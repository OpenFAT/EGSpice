//! The lane GPU kernels (phase 4): one device type over `n` independent
//! lanes (sweep points, Monte Carlo samples), lane-major rows and planes
//! in, each lane's stamps, lim values, latches, charges and breakpoints
//! out. One thread owns its lane's cells, so adds land in tape order with
//! no staging or atomics, and each lane equals its own serial run.
//! Cites dev/research/plans/phase4 and 06 6.11 for the colour ordering.
//!
//! Contents:
//! - `pub fn LaneSink(comptime D: type) type`
//! - `pub fn LaneEvalKernel(comptime D: type, comptime block_size: u32) type`
//! - `pub fn LaneStateKernel(comptime D: type, comptime block_size: u32) type`
//! - `pub fn LaneCtlKernel(comptime D: type, comptime block_size: u32) type`
//! - `pub fn LaneChargeKernel(comptime D: type, comptime block_size: u32) type`
//! - `pub fn LaneBreakKernel(comptime D: type, comptime block_size: u32) type`
//! - each kernel: `pub fn run(...) callconv(gompute.kernel_callconv) void`;
//!   the sink: `skip_g`, `skip_c`, `on_device`, `x`, `gath`, `rhsRow`, `lim`,
//!   `model`, `inst`, `scatterRes`, `scatterJac`, `scatterQ`, `scatterQJac`, `tapeQ`

const std = @import("std");
const contract = @import("contract");
const gompute = @import("gompute");
const abi = @import("device_abi");
const breakpoints = @import("breakpoints.zig");
const dual = @import("dual.zig");
const layout = @import("layout.zig");
const stamp = @import("stamp.zig");
const traits = @import("traits.zig");
const gpu_state_call = @import("kernels.zig").gpu_state_call;
const SimState = abi.SimState;
const StateCtlOp = abi.StateCtlOp;
const LimMap = layout.LimMap;
const SlotMap = layout.SlotMap;
const SrcBrk = breakpoints.SrcBrk;
const RealFor = dual.RealFor;
const Real = dual.Real;

/// Memory access for `evalRange` over lane `l` of `n` lanes, lane-major:
/// unknown gi at `xs[gi * n + l]`, plane cell c at `plane[c * n + l]`,
/// instance id's row at `instances[id * n + l]` and its Model at
/// `models[model_of[id * n + l]]`; the gather, scatter and slot tapes are
/// the batch's own, shared by every lane. One thread walks every instance
/// of its lane, so it alone writes its lane's cells: adds land in tape
/// order with no staging and no atomics (dev/research/plans/phase4).
pub fn LaneSink(comptime D: type) type {
    const n_u: usize = comptime contract.nU(D);
    return struct {
        n: u32,
        l: u32,
        xs: gompute.GlobalPtr(f64),
        gath_: gompute.GlobalPtr(u32),
        rhs_idx_: gompute.GlobalPtr(u32),
        slots_: gompute.GlobalPtr(u32),
        models_: gompute.GlobalPtr(D.Model),
        model_of_: gompute.GlobalPtr(u32),
        instances_: gompute.GlobalPtr(D.Instance),
        g_vals: gompute.GlobalPtr(f64),
        c_vals: gompute.GlobalPtr(f64),
        rhs: gompute.GlobalPtr(f64),
        q_vec: gompute.GlobalPtr(f64),
        /// Lane-major lim plane: instance id's limited unknown `at(u)` at
        /// `(id * k + at(u)) * n + l`.
        lim_: gompute.GlobalPtr(f64),
        /// Lane l's entry 0 of its tape slot, entries `n` apart
        /// (`LaneChargeKernel`), else undefined.
        tape_: gompute.GlobalPtr(f64) = undefined,

        pub const skip_g = false;
        pub const skip_c = false;
        pub const on_device = true;
        const Sk = @This();

        inline fn at(s: *const Sk, i: usize) usize {
            return i * s.n + s.l;
        }
        pub inline fn x(s: *const Sk, gi: u32) f64 {
            return s.xs[s.at(gi)];
        }
        pub inline fn gath(s: *const Sk, id: u32, u: usize) u32 {
            return s.gath_[@as(usize, id) * n_u + u];
        }
        pub inline fn rhsRow(s: *const Sk, id: u32, ru: usize) u32 {
            return s.rhs_idx_[@as(usize, id) * n_u + ru];
        }
        pub inline fn lim(s: *const Sk, id: u32, comptime u: usize) f64 {
            return s.lim_[s.at(@as(usize, id) * LimMap(D).k + comptime LimMap(D).at(u))];
        }
        pub inline fn model(s: *const Sk, id: u32) *const D.Model {
            return @addrSpaceCast(&s.models_[s.model_of_[s.at(id)]]);
        }
        pub inline fn inst(s: *const Sk, id: u32) if (@hasDecl(D, "mutable_eval") and D.mutable_eval) *D.Instance else *const D.Instance {
            return @addrSpaceCast(&s.instances_[s.at(id)]);
        }
        inline fn slot(s: *const Sk, id: u32, comptime ru: usize, comptime cu: usize) u32 {
            return s.slots_[@as(usize, id) * SlotMap(D).k + comptime SlotMap(D).at(ru, cu)];
        }
        pub inline fn scatterRes(s: *const Sk, row: u32, val: f64) void {
            s.rhs[s.at(row)] += val;
        }
        pub inline fn scatterJac(s: *const Sk, id: u32, comptime ru: usize, comptime cu: usize, val: f64) void {
            s.g_vals[s.at(s.slot(id, ru, cu))] += val;
        }
        pub inline fn scatterQ(s: *const Sk, row: u32, qv: f64) void {
            s.q_vec[s.at(row)] += qv;
        }
        pub inline fn scatterQJac(s: *const Sk, id: u32, comptime ru: usize, comptime cu: usize, val: f64) void {
            s.c_vals[s.at(s.slot(id, ru, cu))] += val;
        }
        /// LTE site `j` of instance `id` into lane l's tape slot.
        pub inline fn tapeQ(s: *const Sk, id: u32, j: usize, qv: f64) void {
            s.tape_[(@as(usize, id) * (comptime layout.lteSites(D).len) + j) * s.n] = qv;
        }
    };
}

/// GPU lane eval for D: thread l is lane l of `n_lanes`, evaluating every
/// instance of the batch at its own x under its own `sims[l]`, skipped
/// when `active[l] == 0` (a frozen or finished lane). `limiting[0]` is the
/// batch's lim-active word. Planes must be zeroed first; each plane cell
/// then sums its contributions in the host stamp's order.
pub fn LaneEvalKernel(comptime D: type, comptime block_size: u32) type {
    return struct {
        pub fn run(
            n_lanes: u64,
            count: u64,
            sims: gompute.GlobalPtr(SimState),
            active: gompute.GlobalPtr(u8),
            limiting: gompute.GlobalPtr(u32),
            xs: gompute.GlobalPtr(f64),
            gath: gompute.GlobalPtr(u32),
            rhs_idx: gompute.GlobalPtr(u32),
            slots: gompute.GlobalPtr(u32),
            models: gompute.GlobalPtr(D.Model),
            model_of: gompute.GlobalPtr(u32),
            instances: gompute.GlobalPtr(D.Instance),
            g_vals: gompute.GlobalPtr(f64),
            c_vals: gompute.GlobalPtr(f64),
            rhs: gompute.GlobalPtr(f64),
            q_vec: gompute.GlobalPtr(f64),
            lim: gompute.GlobalPtr(f64),
            // `laneRange` maps threads to (lane, instances).
            ids: gompute.GlobalPtr(u32),
            ids_off: u64,
            n_ids: u64,
        ) callconv(gompute.kernel_callconv) void {
            const r = laneRange(gompute.globalIdX(block_size), n_lanes, count, ids, ids_off, n_ids) orelse return;
            const l = r.l;
            if (active[l] == 0) return;
            var sink: LaneSink(D) = .{
                .n = @intCast(n_lanes),
                .l = l,
                .xs = xs,
                .gath_ = gath,
                .rhs_idx_ = rhs_idx,
                .slots_ = slots,
                .models_ = models,
                .model_of_ = model_of,
                .instances_ = instances,
                .g_vals = g_vals,
                .c_vals = c_vals,
                .rhs = rhs,
                .q_vec = q_vec,
                .lim_ = lim,
            };
            stamp.evalRange(D, false, false, &sink, r.first, r.last, sims[l], limiting[l] != 0);
        }
    };
}

/// GPU lane limit/state kernel: `StateKernel` per (instance, lane), lanes
/// innermost (thread t is lane t % n_lanes of instance t / n_lanes), for
/// the lanes with `mask[l] != 0`. ORs bit 0 (not converged), bit 1 (a
/// reject request) and bit 2 (a finite `$bound_step`) into `flags[l]`; the
/// lane path hands a lane with bit 1 or 2 back to the host.
pub fn LaneStateKernel(comptime D: type, comptime block_size: u32) type {
    const n_u: usize = comptime contract.nU(D);
    const has_limit = @hasDecl(D, "limit");
    const has_state = @hasDecl(D, "State");
    const StateT = if (has_state) D.State else u8;
    return struct {
        pub fn run(
            n_lanes: u64,
            count: u64,
            mask: gompute.GlobalPtr(u8),
            xs: gompute.GlobalPtr(f64),
            x_old: gompute.GlobalPtr(f64),
            gath: gompute.GlobalPtr(u32),
            models: gompute.GlobalPtr(D.Model),
            model_of: gompute.GlobalPtr(u32),
            instances: gompute.GlobalPtr(D.Instance),
            lim: gompute.GlobalPtr(f64),
            states: gompute.GlobalPtr(StateT),
            lim_active: gompute.GlobalPtr(u32),
            halves: u64,
            flags: gompute.GlobalPtr(u32),
            sims: gompute.GlobalPtr(SimState),
        ) callconv(gompute.kernel_callconv) void {
            const tid = gompute.globalIdX(block_size);
            if (tid >= n_lanes * count) return;
            const n: usize = @intCast(n_lanes);
            const l: usize = @intCast(tid % n_lanes);
            const id: usize = @intCast(tid / n_lanes);
            if (mask[l] == 0) return;
            const sim = sims[l];
            const model: *const D.Model = @addrSpaceCast(&models[model_of[id * n + l]]);
            var flag: u32 = 0;
            var cur: [n_u]f64 = undefined;
            inline for (0..n_u) |u| cur[u] = xs[@as(usize, gath[id * n_u + u]) * n + l];
            if (comptime has_limit) if (halves & abi.lim_pass_limit != 0) {
                const writes = comptime layout.limitWritesOf(D);
                const k = comptime LimMap(D).k;
                const inst_c: *const D.Instance = @addrSpaceCast(&instances[id * n + l]);
                var old: [n_u]f64 = undefined;
                inline for (0..n_u) |u| {
                    const xo = x_old[@as(usize, gath[id * n_u + u]) * n + l];
                    old[u] = if (comptime layout.has(writes, u))
                        (if (lim_active[l] != 0) lim[(id * k + comptime LimMap(D).at(u)) * n + l] else xo)
                    else
                        xo;
                }
                var lm = D.limit(Real, model, inst_c, cur, old, sim);
                stamp.snapLimited(n_u, comptime layout.limitReadsOf(D), writes, &lm.x, cur);
                if (!lm.converged) flag |= 1;
                inline for (0..n_u) |u| if (comptime layout.has(writes, u)) {
                    lim[(id * k + comptime LimMap(D).at(u)) * n + l] = lm.x[u];
                };
            };
            if (comptime has_state) if (halves & abi.lim_pass_state != 0) {
                const inst_m: *D.Instance = @addrSpaceCast(&instances[id * n + l]);
                const st: *StateT = @addrSpaceCast(&states[id * n + l]);
                switch (@call(gpu_state_call, D.updateState, .{ Real, model, inst_m, cur, st, sim })) {
                    .ok => {},
                    .request_reject_at => flag |= 2,
                }
                // A finite §9.17.2 `$bound_step`: the host's step control
                // must read it, so the lane leaves the device (bit 2).
                if (comptime @hasField(D.Instance, "bound_step")) if (inst_m.bound_step < std.math.inf(f64)) {
                    flag |= 4;
                };
            };
            if (flag != 0) _ = @atomicRmw(u32, &flags[l], .Or, flag, .monotonic);
        }
    };
}

/// GPU lane accepted-step kernel: `CtlKernel` per (instance, lane), lanes
/// innermost, running `D.stateCtl` with lane l's own op (`ops[l]`, a
/// `StateCtlOp` + 1; 0 skips the lane). ORs a query hit into `flags[l]`.
pub fn LaneCtlKernel(comptime D: type, comptime block_size: u32) type {
    const StateT = if (@hasDecl(D, "State")) D.State else u8;
    return struct {
        pub fn run(
            n_lanes: u64,
            count: u64,
            ops: gompute.GlobalPtr(u8),
            models: gompute.GlobalPtr(D.Model),
            model_of: gompute.GlobalPtr(u32),
            instances: gompute.GlobalPtr(D.Instance),
            states: gompute.GlobalPtr(StateT),
            flags: gompute.GlobalPtr(u32),
        ) callconv(gompute.kernel_callconv) void {
            const tid = gompute.globalIdX(block_size);
            if (tid >= n_lanes * count) return;
            const n: usize = @intCast(n_lanes);
            const l: usize = @intCast(tid % n_lanes);
            const id: usize = @intCast(tid / n_lanes);
            const op = ops[l];
            if (op == 0) return;
            const model: *const D.Model = @addrSpaceCast(&models[model_of[id * n + l]]);
            const inst: *D.Instance = @addrSpaceCast(&instances[id * n + l]);
            const st: *StateT = @addrSpaceCast(&states[id * n + l]);
            const sop: StateCtlOp = @fromBackingInt(@intCast(op - 1));
            if (D.stateCtl(model, inst, st, sop))
                _ = @atomicRmw(u32, &flags[l], .Or, 1, .monotonic);
        }
    };
}

/// GPU lane charge kernel: the device side of `Circuit.evalQ` for lane l of
/// `n_lanes` (thread l walks every instance, or with `n_ids` > 0 a thread
/// per (instance colour, lane), as `LaneEvalKernel`):
/// `evalQRange` adds each instance's charges into lane l's q plane (zeroed
/// first) and writes its LTE sites into slot `slot0[l]` of lane l's
/// lane-minor tape ring, site j of instance id at `((l * n_hist +
/// slot0[l]) * n_qt) + off + id * sites + j`. Skips `mask[l] == 0`.
pub fn LaneChargeKernel(comptime D: type, comptime block_size: u32) type {
    return struct {
        pub fn run(
            n_lanes: u64,
            count: u64,
            mask: gompute.GlobalPtr(u8),
            sims: gompute.GlobalPtr(SimState),
            xs: gompute.GlobalPtr(f64),
            gath: gompute.GlobalPtr(u32),
            rhs_idx: gompute.GlobalPtr(u32),
            models: gompute.GlobalPtr(D.Model),
            model_of: gompute.GlobalPtr(u32),
            instances: gompute.GlobalPtr(D.Instance),
            q_vec: gompute.GlobalPtr(f64),
            tape: gompute.GlobalPtr(f64),
            slot0: gompute.GlobalPtr(u8),
            n_hist: u64,
            n_qt: u64,
            off: u64,
            ids: gompute.GlobalPtr(u32),
            ids_off: u64,
            n_ids: u64,
        ) callconv(gompute.kernel_callconv) void {
            _ = n_hist; // the lane-major tape ring needs no per-lane base
            const r = laneRange(gompute.globalIdX(block_size), n_lanes, count, ids, ids_off, n_ids) orelse return;
            const l = r.l;
            if (mask[l] == 0) return;
            var sink: LaneSink(D) = .{
                .n = @intCast(n_lanes),
                .l = l,
                .xs = xs,
                .gath_ = gath,
                .rhs_idx_ = rhs_idx,
                .slots_ = undefined,
                .models_ = models,
                .model_of_ = model_of,
                .instances_ = instances,
                .g_vals = undefined,
                .c_vals = undefined,
                .rhs = undefined,
                .q_vec = q_vec,
                .lim_ = undefined,
                // Lane-major ring: entry j of slot k at `(k * n_qt + j) * N + l`.
                .tape_ = tape + ((@as(usize, slot0[l]) * @as(usize, @intCast(n_qt)) + @as(usize, @intCast(off))) * @as(usize, @intCast(n_lanes)) + l),
            };
            stamp.evalQRange(D, RealFor(@hasDecl(D, "collapse")), &sink, r.first, r.last, sims[l], null);
        }
    };
}

/// Device step control's `nextBreakpointFn` per lane of `mask`, after the
/// commit: `best[l] = min(best[l], the earliest breakpoint after tt[l][1])`,
/// with `tt[l][0]` the accepted time the host's PULSE/PWL corners round
/// from. The source caches live per instance and lane (`brk`, instance
/// `id` of lane l at `id * N + l`), seeded from `GpuPayload.brk`, so a
/// corner cached at an earlier request keeps that request's rounding as it
/// does on the host. The host's timer-only cache changes no answer and has
/// no device copy.
pub fn LaneBreakKernel(comptime D: type, comptime block_size: u32) type {
    return struct {
        pub fn run(
            n_lanes: u64,
            count: u64,
            mask: gompute.GlobalPtr(u8),
            tt: gompute.GlobalPtr([2]f64),
            models: gompute.GlobalPtr(D.Model),
            model_of: gompute.GlobalPtr(u32),
            instances: gompute.GlobalPtr(D.Instance),
            brk: gompute.GlobalPtr(SrcBrk),
            best: gompute.GlobalPtr(f64),
        ) callconv(gompute.kernel_callconv) void {
            const tid = gompute.globalIdX(block_size);
            if (tid >= n_lanes) return;
            const l: usize = @intCast(tid);
            if (mask[l] == 0) return;
            const N: usize = @intCast(n_lanes);
            const t_acc = tt[l][0];
            const t = tt[l][1];
            var b = best[l];
            if (comptime @hasDecl(D, "nextBreakpoint")) for (0..@as(usize, @intCast(count))) |id| {
                const m: *const D.Model = @addrSpaceCast(&models[model_of[id * N + l]]);
                if (comptime traits.hasSpiceBreaks(D)) {
                    b = @min(b, breakpoints.srcBreak(D, @addrSpaceCast(&brk[id * N + l]), m, t_acc, t));
                    continue;
                }
                if (D.nextBreakpoint(m, t)) |bp| b = @min(b, bp);
            };
            if (comptime traits.walksPending(D)) for (0..@as(usize, @intCast(count))) |id| {
                const inst: *const D.Instance = @addrSpaceCast(&instances[id * N + l]);
                if (D.pendingBreakpoint(inst, t)) |bp| b = @min(b, bp);
            };
            best[l] = b;
        }
    };
}

// ---- private ----

/// Thread `tid`'s lane and instance range in a lane kernel: with `n_ids` > 0
/// instance `ids[ids_off + tid / n_lanes]` on lane `tid % n_lanes` (lanes
/// innermost, 06 6.11, `gpu/lane_color.zig`), else lane `tid` over all
/// `count` instances. Null for a thread past the grid's end.
inline fn laneRange(tid: usize, n_lanes: u64, count: u64, ids: gompute.GlobalPtr(u32), ids_off: u64, n_ids: u64) ?struct { l: u32, first: u32, last: u32 } {
    if (n_ids == 0) {
        if (tid >= n_lanes) return null;
        return .{ .l = @intCast(tid), .first = 0, .last = @intCast(count) };
    }
    if (tid >= n_ids * n_lanes) return null;
    const first = ids[@intCast(ids_off + tid / n_lanes)];
    return .{ .l = @intCast(tid % n_lanes), .first = first, .last = first + 1 };
}
