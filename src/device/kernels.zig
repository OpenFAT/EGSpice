//! The per-instance GPU kernels: one thread per instance of one device
//! type, the batch's resident tapes, rows and planes in, staging cells,
//! lim values, latches and charge tapes out. Every kernel runs the host's
//! own body (`evalRange`, `limit`, `updateState`, `stateCtl`, `q`), so the
//! CPU path is its oracle. Cites dev/devices/gpu-evaluation.md, #15 for the
//! split limit and state halves, and 01 1.6 for the constant baseline.
//!
//! Contents:
//! - `pub const gpu_state_call: std.lang.CallModifier`
//! - `pub fn DeviceKernel(comptime D: type, comptime block_size: u32, comptime skip_const: bool) type`
//! - `pub fn StateKernel(comptime D: type, comptime block_size: u32) type`
//! - `pub fn ReduceKernel(comptime _: type, comptime block_size: u32) type`
//! - `pub inline fn segmentSum(stage: gompute.GlobalPtr(f64), first: u32, end: u32) f64`
//! - `pub fn ChargeKernel(comptime D: type, comptime block_size: u32) type`
//! - `pub fn QTapeKernel(comptime D: type, comptime block_size: u32) type`
//! - `pub fn CtlKernel(comptime D: type, comptime block_size: u32) type`
//! - each kernel: `pub fn run(...) callconv(gompute.kernel_callconv) void`

const std = @import("std");
const builtin = @import("builtin");
const contract = @import("contract");
const gompute = @import("gompute");
const abi = @import("device_abi");
const dual = @import("dual.zig");
const layout = @import("layout.zig");
const stamp = @import("stamp.zig");
const SimState = abi.SimState;
const StateCtlOp = abi.StateCtlOp;
const LimMap = layout.LimMap;
const RealFor = dual.RealFor;
const Real = dual.Real;
const Sink = @import("sink.zig").Sink;

// Zig 0.18.0-dev.349's AMDGPU backend aborts with "Cannot scavenge register
// in FI elimination!" on the outlined coupled-line updateState call.
// Inline it into both state kernels to avoid that call's frame lowering.
// Keep CUDA's existing inlining policy. `zig build test`'s HIP probe emits machine
// code for all three coupled-line widths, including their lane kernels.
pub const gpu_state_call: std.lang.CallModifier = if (builtin.target.cpu.arch == .amdgcn) .always_inline else .auto;

/// GPU eval kernel for D, one thread per instance, running the same
/// `evalRange` as the host through a device `Sink`.
///
/// `skip_const` (`arp_evalc_<model>`, only for a batch whose whole Jacobian
/// is constant) drops the g and c stamps, which the launcher keeps in a
/// device baseline instead (01 1.6); residuals and charges stamp as usual.
pub fn DeviceKernel(comptime D: type, comptime block_size: u32, comptime skip_const: bool) type {
    return struct {
        pub fn run(
            count: u64,
            sim: SimState,
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
            limiting: u64,
        ) callconv(gompute.kernel_callconv) void {
            const tid = gompute.globalIdX(block_size);
            if (tid >= count) return;
            var sink = Sink(D, true, skip_const, false){
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
                .xo = undefined, // read only by limitRange
                .q_tape = {},
            };
            const id: u32 = @intCast(tid);
            // Always the wide basis here: the kernel arguments carry no
            // `narrow_count`, and they are part of the frozen GPU boundary.
            // ponytail: pass `narrow_count` once narrowing is measured on a GPU.
            stamp.evalRange(D, false, false, &sink, id, id + 1, sim, limiting != 0);
        }
    };
}

/// GPU limit and state-latch kernel, one thread per instance: the device side
/// of `applyLimits` and of `updateState`. `halves` (`abi.lim_pass_*` bits)
/// picks which run; the launcher runs the limit half per Newton iterate and
/// the state half once per converged solve, as the host does (#15).
///
/// Sets bit 0 of `flags[0]` when a clamp reports not converged, and bit 1
/// when an `updateState` returns anything but `.ok`, which the launcher
/// treats as a fault and answers by falling back to the CPU.
pub fn StateKernel(comptime D: type, comptime block_size: u32) type {
    const n_u: usize = comptime contract.nU(D);
    const has_limit = @hasDecl(D, "limit");
    const has_state = @hasDecl(D, "State");
    const StateT = if (has_state) D.State else u8;
    return struct {
        pub fn run(
            count: u64,
            xs: gompute.GlobalPtr(f64),
            x_old: gompute.GlobalPtr(f64),
            gath: gompute.GlobalPtr(u32),
            models: gompute.GlobalPtr(D.Model),
            model_of: gompute.GlobalPtr(u32),
            instances: gompute.GlobalPtr(D.Instance),
            lim: gompute.GlobalPtr(f64),
            states: gompute.GlobalPtr(StateT),
            lim_active: u64,
            halves: u64,
            flags: gompute.GlobalPtr(u32),
            sim: SimState,
        ) callconv(gompute.kernel_callconv) void {
            const tid = gompute.globalIdX(block_size);
            if (tid >= count) return;
            const id: usize = @intCast(tid);
            const model: *const D.Model = @addrSpaceCast(&models[model_of[id]]);
            var flag: u32 = 0;
            var cur: [n_u]f64 = undefined;
            inline for (0..n_u) |u| cur[u] = xs[gath[id * n_u + u]];
            if (comptime has_limit) if (halves & abi.lim_pass_limit != 0) {
                // Must use the same write set as `limitRange`: host and
                // device share one `lim_x` plane.
                const writes = comptime layout.limitWritesOf(D);
                const inst_c: *const D.Instance = @addrSpaceCast(&instances[id]);
                var old: [n_u]f64 = undefined;
                inline for (0..n_u) |u| if (comptime layout.has(writes, u)) {
                    old[u] = if (lim_active != 0) lim[id * LimMap(D).k + comptime LimMap(D).at(u)] else x_old[gath[id * n_u + u]];
                } else {
                    old[u] = x_old[gath[id * n_u + u]];
                };
                var lm = D.limit(Real, model, inst_c, cur, old, sim);
                stamp.snapLimited(n_u, comptime layout.limitReadsOf(D), writes, &lm.x, cur);
                if (!lm.converged) flag |= 1;
                inline for (0..n_u) |u| if (comptime layout.has(writes, u)) {
                    lim[id * LimMap(D).k + comptime LimMap(D).at(u)] = lm.x[u];
                };
            };
            if (comptime has_state) if (halves & abi.lim_pass_state != 0) {
                const inst_m: *D.Instance = @addrSpaceCast(&instances[id]);
                const st: *StateT = @addrSpaceCast(&states[id]);
                switch (@call(gpu_state_call, D.updateState, .{ Real, model, inst_m, cur, st, sim })) {
                    .ok => {},
                    .request_reject_at => flag |= 2,
                }
            };
            if (flag != 0) _ = @atomicRmw(u32, &flags[0], .Or, flag, .monotonic);
        }
    };
}

/// Segmented sum, one output cell per thread: `out[i] = sum(in[seg[2i] ..
/// seg[2i + 1]])`. The launcher lays each cell's staging contributions out
/// contiguously in tape order, so this sums them in the order the host stamp
/// does, on every launch. Explicit ends let ranges skip the staging cells
/// that no plane cell reads (ground and structural-zero contributions). The
/// output is overwritten, not accumulated.
///
/// Strict left-to-right with one accumulator on purpose: no `.optimized`
/// float mode, and no interleaved partial sums. The tape emits rows in `ru`
/// order, so interleaving would group all the +1.8 contributions of a supply
/// row in one lane and all the -1.8 in another, measured at 4e-12 error
/// against 1.6e-17 in order. The launcher bounds the chain length instead.
/// The loads go out eight at a time ahead of the in-order adds, so a thread
/// waits on memory once per eight contributions, not once per contribution.
pub fn ReduceKernel(comptime _: type, comptime block_size: u32) type {
    return struct {
        pub fn run(
            n_cells: u64,
            seg: gompute.GlobalPtr(u32),
            stage: gompute.GlobalPtr(f64),
            plane: gompute.GlobalPtr(f64),
        ) callconv(gompute.kernel_callconv) void {
            const tid = gompute.globalIdX(block_size);
            if (tid >= n_cells) return;
            const i: usize = @intCast(tid);
            plane[i] = segmentSum(stage, seg[2 * i], seg[2 * i + 1]);
        }
    };
}

/// `stage[first..end]` summed strictly left to right, the loads issued eight at
/// a time. Split out of `ReduceKernel` so the host tests it against the plain
/// loop: tests/eval.zig.
pub inline fn segmentSum(stage: gompute.GlobalPtr(f64), first: u32, end: u32) f64 {
    var sum: f64 = 0;
    var k = first;
    while (k + 8 <= end) : (k += 8) {
        var v: [8]f64 = undefined;
        inline for (&v, 0..) |*e, j| e.* = stage[k + j];
        inline for (v) |e| sum += e;
    }
    while (k < end) : (k += 1) sum += stage[k];
    return sum;
}

/// GPU charge-only kernel, one thread per instance: `evalQRange`'s q rows
/// into the q staging (`q_vec`, through `rhs_idx` as the eval kernel), and
/// with `with_tape` the LTE charges into `tape` as `QTapeKernel` writes them.
/// The host's `evalQ` pass: values only, at `xs` unlimited, so the
/// accepted-point charge re-read costs a fraction of a full eval (04 4.4).
pub fn ChargeKernel(comptime D: type, comptime block_size: u32) type {
    const n_u: usize = comptime contract.nU(D);
    const sites = comptime layout.lteSites(D);
    const q_row = comptime layout.writtenRows(D, "q_rows");
    return struct {
        pub fn run(
            count: u64,
            sim: SimState,
            xs: gompute.GlobalPtr(f64),
            gath: gompute.GlobalPtr(u32),
            rhs_idx: gompute.GlobalPtr(u32),
            models: gompute.GlobalPtr(D.Model),
            model_of: gompute.GlobalPtr(u32),
            instances: gompute.GlobalPtr(D.Instance),
            q_vec: gompute.GlobalPtr(f64),
            tape: gompute.GlobalPtr(f64),
            with_tape: u64,
        ) callconv(gompute.kernel_callconv) void {
            const tid = gompute.globalIdX(block_size);
            if (tid >= count) return;
            const id: usize = @intCast(tid);
            const S = RealFor(@hasDecl(D, "collapse"));
            var lx: [n_u]f64 = undefined;
            inline for (0..n_u) |u| lx[u] = xs[gath[id * n_u + u]];
            const qs = D.q(S, &lx, @addrSpaceCast(&models[model_of[id]]), @addrSpaceCast(&instances[id]), sim);
            const qo = contract.qRows(D, S, qs);
            const mf = stamp.mfactorOf(D, &instances[id]);
            // Stores, as the device `Sink`; the q rows are not ground-masked
            // there either.
            inline for (0..n_u) |ru| if (comptime q_row[ru]) {
                q_vec[rhs_idx[id * n_u + ru]] = qo[ru].v * mf;
            };
            if (with_tape != 0) inline for (sites, 0..) |k, j| {
                tape[id * sites.len + j] = qs[k].v * mf;
            };
        }
    };
}

/// GPU charge-tape kernel, one thread per instance: the LTE charges
/// `evalQRange` records on the host, at `xs`, into `tape` (`count *
/// layout.lteSites(D).len`). Same `RealFor` basis, so the charges match the host
/// tape to the device's rounding. The launcher runs it lazily, once per
/// transient step, instead of widening the frozen eval kernel's arguments.
pub fn QTapeKernel(comptime D: type, comptime block_size: u32) type {
    const n_u: usize = comptime contract.nU(D);
    const sites = comptime layout.lteSites(D);
    return struct {
        pub fn run(
            count: u64,
            sim: SimState,
            xs: gompute.GlobalPtr(f64),
            gath: gompute.GlobalPtr(u32),
            models: gompute.GlobalPtr(D.Model),
            model_of: gompute.GlobalPtr(u32),
            instances: gompute.GlobalPtr(D.Instance),
            tape: gompute.GlobalPtr(f64),
        ) callconv(gompute.kernel_callconv) void {
            const tid = gompute.globalIdX(block_size);
            if (tid >= count) return;
            const id: usize = @intCast(tid);
            const S = RealFor(@hasDecl(D, "collapse"));
            var lx: [n_u]f64 = undefined;
            inline for (0..n_u) |u| lx[u] = xs[gath[id * n_u + u]];
            const qs = D.q(S, &lx, @addrSpaceCast(&models[model_of[id]]), @addrSpaceCast(&instances[id]), sim);
            const mf = stamp.mfactorOf(D, &instances[id]);
            inline for (sites, 0..) |k, j| tape[id * sites.len + j] = qs[k].v * mf;
        }
    };
}

/// GPU accepted-step kernel, one thread per instance: runs `D.stateCtl` with
/// `op` on the resident Instance/State blobs and ORs its result into bit 0
/// of `flags[0]`.
pub fn CtlKernel(comptime D: type, comptime block_size: u32) type {
    const StateT = if (@hasDecl(D, "State")) D.State else u8;
    return struct {
        pub fn run(
            count: u64,
            models: gompute.GlobalPtr(D.Model),
            model_of: gompute.GlobalPtr(u32),
            instances: gompute.GlobalPtr(D.Instance),
            states: gompute.GlobalPtr(StateT),
            op: u64,
            flags: gompute.GlobalPtr(u32),
        ) callconv(gompute.kernel_callconv) void {
            const tid = gompute.globalIdX(block_size);
            if (tid >= count) return;
            const id: usize = @intCast(tid);
            const model: *const D.Model = @addrSpaceCast(&models[model_of[id]]);
            const inst: *D.Instance = @addrSpaceCast(&instances[id]);
            const st: *StateT = @addrSpaceCast(&states[id]);
            const sop: StateCtlOp = @fromBackingInt(@intCast(@as(u8, @truncate(op))));
            if (D.stateCtl(model, inst, st, sop))
                _ = @atomicRmw(u32, &flags[0], .Or, 1, .monotonic);
        }
    };
}
