//! The frozen per-type batch: one device type's staged rows in, SoA
//! columns for every instance plus the shared tapes out, reached through
//! `Batch` and the comptime `hooks` table. A hook is null when D lacks the
//! decl it needs. The hook bodies live in batch_state.zig, batch_params.zig,
//! batch_ac.zig and batch_bypass.zig; this file holds the columns, the
//! eval entry points and the batch's life cycle.
//! Cites VerA device ABI 6 (`setup` per Model row) and
//! dev/devices/gpu-evaluation.md for the payload.
//!
//! Contents:
//! - `pub fn DeviceBatch(comptime D: type) type`, with its fields and
//!   `shares_models`, `hooks`, `of`, `evalSplit`, `model`, `ownsRows`,
//!   `simAt`, `localX`, `setupRows`, `reprep`, `commitBirth`, `binding`

const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;
const contract = @import("contract");
const abi = @import("device_abi");
const layout = @import("layout.zig");
const stamp = @import("stamp.zig");
const traits = @import("traits.zig");
const breakpoints = @import("breakpoints.zig");
const dual = @import("dual.zig");
const staging_gpa = @import("proto_store.zig").staging_gpa;
const Sink = @import("sink.zig").Sink;
const Bypass = @import("batch_bypass.zig").Bypass;
const Params = @import("batch_params.zig").Params;
const SmallSignal = @import("batch_ac.zig").SmallSignal;
const State = @import("batch_state.zig").State;
const Batch = abi.Batch;
const Hooks = abi.Hooks;
const Planes = abi.Planes;
const SimState = abi.SimState;
const GpuPayload = abi.GpuPayload;
const LimMap = layout.LimMap;
const SrcBrk = breakpoints.SrcBrk;
const RealFor = dual.RealFor;
const Real = dual.Real;

/// The frozen per-type batch: SoA columns for every instance of D plus the
/// shared tapes, reached through `Batch` and the comptime `hooks` table.
pub fn DeviceBatch(comptime D: type) type {
    const n_u: usize = comptime contract.nU(D);
    const has_state = @hasDecl(D, "State");
    const has_q = @hasDecl(D, "q");
    const has_limit = @hasDecl(D, "limit");
    const narrowable = layout.canNarrow(D);
    // Only timer-only models: their breakpoints are pure in (model, t).
    // Native lines rewrite `Model.brk` as they step, so theirs are not.
    const has_bp = @hasDecl(D, "nextBreakpoint") and traits.skipsTimerState(D);
    const has_src_brk = traits.hasSpiceBreaks(D);
    const Ac = traits.AcTerms(D);
    // `update_state` stages: not the delay lines (`commit_state`), not
    // timer-only state.
    const stages_update = @hasDecl(D, "updateState") and !traits.hasAbsdelayState(D) and !traits.skipsTimerState(D);
    const has_ac_dyn = @hasDecl(Ac, "ac_dyn_slots");

    return struct {
        /// Instances whose cards bind bit-identical Models share one row of
        /// `models`: nothing but the host's parameter writes changes a Model
        /// after `derive`. A model that caches per-instance breakpoints or
        /// rewrites itself keeps one row each.
        pub const shares_models = !has_bp and !has_src_brk and !@hasDecl(D, "nextBreakpoint");

        count: u32,
        /// Prepared templates own the tapes; instances made from them borrow.
        owns_tapes: bool = true,
        /// Instances `[0, narrow_count)` collapse maximally and run on the
        /// narrow basis; `ProtoStore.finalize` sorted them to the front.
        narrow_count: if (narrowable) u32 else void,
        /// One row per distinct Model (`shares_models`), else per instance.
        models: []D.Model,
        /// Per instance, its row of `models`. Owned per batch, unlike the
        /// tapes: `unshareModels` rewrites it.
        model_of: []u32,
        /// `unshareModels` replaced `models` with a `staging_gpa` copy and
        /// parked the shared rows here until `destroy`. It cannot use the
        /// batch's allocator, which the worker thread that unshares may not
        /// own (a template's is the Problem arena).
        unshared: bool,
        models_shared: []D.Model,
        /// `writeParams` appended private rows, so `models.len` no longer
        /// tells whether every instance owns its row.
        cowed: bool,
        /// `nextBreakpointFn` returned `hi` (inf for none) at `lo`. That
        /// answer holds for every `t` in `[lo, hi)`: no fire lies strictly
        /// between. `lo = inf` is empty. Every model write goes through
        /// `reprep`, which empties it.
        bp: if (has_bp) struct { lo: f64, hi: f64 } else void,
        /// Per model, ngspice's VSRCbreak_time: the PULSE or PWL breakpoint `at`
        /// requested at query time `req`, kept until a query reaches it
        /// (or goes back before `req`, a new run). `req = inf` is empty.
        src_brk: if (has_src_brk) []SrcBrk else void,
        lim_x: if (has_limit) []f64 else void,
        lim_active: if (has_limit) bool else void,
        /// The analysis state every device call receives; `eval` takes `t`
        /// from its own argument.
        sim: SimState = .{},
        /// `updateState` has latched the §5.2.1 `analog initial` results
        /// (held, §5.10), so later calls run with `analog_initial` false and
        /// skip the block. `reprep` (new parameters or temperature) clears it.
        held_initial: bool = false,
        /// `held_initial` at the last `stateCtl(.commit)`, which a
        /// `.revert` restores along with the held values it rolls back.
        committed_initial: bool = false,
        instances: []D.Instance,
        states: if (has_state) []D.State else void,
        gath: []u32,
        rhs_idx: []u32,
        slots: []u32,
        /// Empty means identity; otherwise frozen row -> original staged row.
        original_ids: []u32,
        /// Charge per instance and LTE site, indexed `id * layout.lteSites(D).len +
        /// j`. Written by `Sink.tapeQ` on the host only; see `Hooks.q_tape`.
        q_tape: if (has_q) []f64 else void,
        /// Latency bypass records (`Hooks.set_bypass`), null when off.
        bypass: ?*Bp = null,

        const Self = @This();
        const Bp = Bypass(D);
        const St = State(D);
        const Pm = Params(D);
        const Sm = SmallSignal(D);

        /// The type's comptime hook table, shared by every batch of D. A hook
        /// is null when D lacks what it needs, so callers test the optional.
        pub const hooks: Hooks = .{
            .instantiate = instantiate,
            .snapshot = snapshot,
            .copy_state = copyState,
            .set_limit_active = if (has_limit) setLimitActive else null,
            .stage = stage,
            .scatter_tapes = scatterTapes,
            .q_tape = if (has_q) qTape else null,
            .charge_identity = if (has_q and layout.lteSites(D).len > 0) chargeIdentity else null,
            .derivative_counts = derivativeCounts,
            .eval_q = if (has_q) evalQOnly else null,
            // Only where `update_state` is the hook that stages (not the
            // delay lines' `commit_state`, not timer-only state).
            .accept_q = if (has_q and @hasDecl(D, "acceptQ") and !traits.hasAbsdelayState(D) and !traits.skipsTimerState(D)) acceptQOnly else null,
            .apply_limits = if (has_limit) St.applyLimits else null,
            .clear_limits = if (has_limit) St.clearLimits else null,
            .limit_range = if (has_limit) St.limitRangeHook else null,
            .arm_limits = if (has_limit) St.armLimits else null,
            .advance_iteration = if (@hasDecl(D, "advanceIteration")) St.advanceIteration else null,
            .check_convergence = if (@hasDecl(D, "checkConvergence")) St.checkConvergence else null,
            .predict_first_iterate = if (traits.predictsOverDeltaOld2(D)) St.predictFirstIterate else null,
            .mark_load_check_rows = if (traits.load_check.get(traits.baseName(D)) != null) St.markInternalRows else null,
            .seed = if (@hasDecl(D, "seed")) St.seedFn else null,
            .seed_ic = if (traits.idtUnknowns(D).len != 0) St.seedIc else null,
            .seed_nodeset = if (@hasDecl(D, "u_nodeset")) St.seedNodeset else null,
            .mark_current_rows = if (@hasDecl(D, "u_kinds")) St.markCurrentRows else null,
            // Delay history runs once per accepted point; everything else
            // per converged solve. See `Hooks.commit_state`.
            .update_state = if (stages_update) St.updateState else null,
            .update_range = if (stages_update) St.updateRange else null,
            .set_bypass = if (Bp.bypassable) Bp.setBypass else null,
            .update_done = if (stages_update) St.updateDone else null,
            .commit_state = if (@hasDecl(D, "updateState") and traits.hasAbsdelayState(D)) St.updateState else null,
            .state_ctl = if (@hasDecl(D, "stateCtl") and !traits.skipsTimerState(D)) St.stateCtl else null,
            // Only `updateState` writes `bound_step`.
            .bound_step = if (@hasDecl(D, "updateState") and @hasField(D.Instance, "bound_step") and !traits.skipsTimerState(D)) St.boundStep else null,
            .set_temp = if (@hasField(D.Model, "temperature__")) St.setTemp else null,
            .set_homotopy = if (@hasField(D.Model, "gmin__") or @hasField(D.Model, "source_scale__")) St.setHomotopy else null,
            .set_sim_state = St.setSimState,
            .min_delay = if (@hasDecl(D, "delays")) St.minDelay else null,
            .next_breakpoint = if (@hasDecl(D, "nextBreakpoint") or traits.walksPending(D)) St.nextBreakpointFn else null,
            .status = if (@hasDecl(D, "status_sites")) St.statusFn else null,
            .say = if (@hasDecl(D, "say")) St.sayFn else null,
            .collect_params = Pm.collectParams,
            .param_layout = Pm.paramLayout,
            .write_params = Pm.writeParams,
            .isolate_rows = Pm.isolateRows,
            // A generator is only priced by the device's own `noisePsd`.
            .collect_noise = if (@hasDecl(D, "noise_gens")) blk: {
                if (!@hasDecl(D, "noisePsd")) @compileError(@typeName(D) ++
                    " declares noise_gens without noisePsd; see dev/analysis/ac-small-signal-noise.md \"Noise: the in-device convention\"");
                break :blk Sm.collectNoise;
            } else null,
            .noise_names = if (@hasDecl(D, "noise_gens")) &Sm.noise_names else &.{},
            .collect_ac_dyn = if (has_ac_dyn) Sm.collectAcDyn else null,
            .ac_dyn = if (has_ac_dyn) Sm.acDyn else null,
            .recompute = if (@hasDecl(D, "collapse") or @hasDecl(D, "precompute") or @hasDecl(D, "setup") or has_bp) St.recomputePrecomputed else null,
            .gpu_payload = if (traits.gpuEligible(D)) gpuPayload else null,
            .deinit = destroy,
        };

        /// The batch behind a type-erased hook context.
        pub inline fn of(ctx: *anyopaque) *Self {
            // SAFETY: every `ctx` a hook receives is the `*Self` `binding` put in its `Batch`.
            return @ptrCast(@alignCast(ctx));
        }

        fn eval(ctx: *anyopaque, pl: *const Planes, first: u32, last: u32, x: []const f64, t: f64) void {
            evalInner(ctx, pl, first, last, x, t, false, false);
        }

        fn evalNewton(ctx: *anyopaque, pl: *const Planes, first: u32, last: u32, x: []const f64, t: f64) void {
            evalInner(ctx, pl, first, last, x, t, true, false);
        }

        fn evalQOnly(ctx: *anyopaque, pl: *const Planes, first: u32, last: u32, x: []const f64, t: f64) void {
            evalQInner(ctx, pl, first, last, x, t, false);
        }

        fn evalQInner(ctx: *anyopaque, pl: *const Planes, first: u32, last: u32, x: []const f64, t: f64, comptime staged: bool) void {
            const self = of(ctx);
            var sink = Sink(D, false, false, staged).host(self, pl, x, undefined);
            // Without `always_inline` LLVM moves mos6's charge core out of
            // line, measured +0.9% on devices/mos6_inverter.
            @call(.always_inline, stamp.evalQRange, .{ D, RealFor(@hasDecl(D, "collapse")), &sink, first, last, self.simAt(t), null });
        }

        /// `evalQOnly` at `self.sim` plus `updateState`'s staging, one core
        /// run per instance.
        fn acceptQOnly(ctx: *anyopaque, pl: *const Planes, first: u32, last: u32, x: []const f64) void {
            const self = of(ctx);
            var sink = Sink(D, false, false, false).host(self, pl, x, undefined);
            @call(.always_inline, stamp.evalQRange, .{ D, RealFor(@hasDecl(D, "collapse")), &sink, first, last, self.sim, self });
        }

        fn stage(ctx: *anyopaque, pl: *const Planes, first: u32, last: u32, x: []const f64, t: f64, mode: abi.StampMode) void {
            switch (mode) {
                .full => evalInner(ctx, pl, first, last, x, t, false, true),
                .newton => evalInner(ctx, pl, first, last, x, t, true, true),
                .charge => if (comptime has_q) evalQInner(ctx, pl, first, last, x, t, true),
            }
        }

        fn scatterTapes(ctx: *anyopaque) abi.ScatterTapes {
            const self = of(ctx);
            return .{ .slots = self.slots, .rows = self.rhs_idx };
        }

        fn qTape(ctx: *anyopaque) []const f64 {
            const self = of(ctx);
            return self.q_tape;
        }

        fn derivativeCounts(ctx: *anyopaque) abi.DerivativeCounts {
            const self = of(ctx);
            const narrow: u32 = if (narrowable) self.narrow_count else 0;
            return .{
                .wide = self.count - narrow,
                .narrow = narrow,
                .wide_lanes = @intCast(comptime layout.basisOf(D, false).w),
                .narrow_lanes = if (narrowable) @intCast(comptime layout.basisOf(D, true).w) else 0,
            };
        }

        fn chargeIdentity(ctx: *anyopaque, index: usize) abi.ChargeIdentity {
            const self = of(ctx);
            const sites = comptime layout.lteSites(D);
            const id = index / sites.len;
            assert(index < self.q_tape.len);
            return .{
                .instance = if (self.original_ids.len == 0) @intCast(id) else self.original_ids[id],
                .site = @intCast(sites[index % sites.len]),
                .rows = self.gath[id * n_u ..][0..n_u],
            };
        }

        fn evalInner(ctx: *anyopaque, pl: *const Planes, first: u32, last: u32, x: []const f64, t: f64, comptime skip_const: bool, comptime staged: bool) void {
            const self = of(ctx);
            // Once per batch call, outside the per-instance loop.
            assert(first <= last);
            assert(last <= self.count);
            const limiting = if (comptime has_limit) self.lim_active else false;
            var sink = Sink(D, false, skip_const, staged).host(self, pl, x, undefined);
            const sim = self.simAt(t);
            // A replay needs the recorded Jacobian (and charge Jacobian) to
            // move the residual to the new point.
            const SinkT = @TypeOf(sink);
            if (comptime Bp.bypassable and skip_const and !SinkT.skip_g and !(has_q and SinkT.skip_c)) if (self.bypass) |bp| if (!bp.off) {
                Bp.evalBypass(self, bp, &sink, first, last, sim, limiting, pl, staged);
                return;
            };
            self.evalSplit(&sink, first, last, sim, limiting);
        }

        /// `evalRange` over `[first, last)`, the narrow basis on the part
        /// below `narrow_count`: a ParEval slice may straddle the boundary.
        pub inline fn evalSplit(self: *const Self, sink: anytype, first: u32, last: u32, sim: SimState, limiting: bool) void {
            if (comptime narrowable) {
                const split = std.math.clamp(self.narrow_count, first, last);
                if (split > first) stamp.evalRange(D, true, stamp.host_pad, sink, first, split, sim, limiting);
                if (last > split) stamp.evalRange(D, false, stamp.host_pad, sink, split, last, sim, limiting);
            } else {
                stamp.evalRange(D, false, stamp.host_pad, sink, first, last, sim, limiting);
            }
        }

        /// Instance `id`'s Model.
        pub inline fn model(self: *const Self, id: usize) *D.Model {
            return &self.models[self.model_of[id]];
        }

        /// Every instance owns its Model row, and `model_of` is the identity.
        pub fn ownsRows(self: *const Self) bool {
            return self.models.len == self.count and !self.cowed;
        }

        /// The stored analysis state at eval time `t`.
        pub fn simAt(self: *const Self, t: f64) SimState {
            var sim = self.sim;
            sim.t = t;
            return sim;
        }

        pub fn localX(self: *Self, x: []const f64, id: usize) [n_u]f64 {
            var out: [n_u]f64 = undefined;
            inline for (0..n_u) |u| out[u] = x[self.gath[id * n_u + u]];
            return out;
        }

        /// VerA's `setup` once per Model row (the solve-invariant cache and
        /// the temperature live on the row, VerA device ABI 6), then
        /// `setupInstance` for every instance that caches from its row.
        pub fn setupRows(self: *Self) void {
            if (comptime @hasDecl(D, "setup")) for (self.models) |*m| D.setup(Real, m);
            if (comptime @hasDecl(D, "setupInstance")) for (self.instances, 0..) |*inst, id| D.setupInstance(self.model(id), inst);
        }

        pub fn reprep(self: *Self) void {
            self.held_initial = false;
            self.sim.analog_initial = true;
            if (comptime has_bp) self.bp.lo = std.math.inf(f64);
            if (comptime has_src_brk) @memset(self.src_brk, .{});
            self.setupRows();
            if (comptime @hasDecl(D, "precompute")) {
                for (self.instances, 0..) |*inst, id| D.precompute(inst, self.model(id));
            }
        }

        /// Commits the initial state, so a `stateCtl(.revert)` before the
        /// first accepted point restores it rather than the `State` defaults.
        pub fn commitBirth(self: *Self) void {
            if (comptime @hasDecl(D, "stateCtl")) _ = St.stateCtl(self, .commit);
        }

        pub fn binding(self: *Self) Batch {
            return .{
                .ctx = self,
                .eval = eval,
                .eval_newton = evalNewton,
                .count = self.count,
                .n_u = n_u,
                .has_charge = has_q,
                .has_const_jacobian = comptime traits.constJacobian(D),
                .type_name = comptime traits.baseName(D),
                .hooks = &hooks,
            };
        }

        fn instantiate(ctx: *const anyopaque, gpa: Allocator) abi.DeviceResult(Batch) {
            return abi.DeviceResult(Batch).fromLocal(duplicate(ctx, gpa, false));
        }

        fn snapshot(ctx: *const anyopaque, gpa: Allocator) abi.DeviceResult(Batch) {
            return abi.DeviceResult(Batch).fromLocal(duplicate(ctx, gpa, true));
        }

        fn copyState(dst_ctx: *anyopaque, src_ctx: *const anyopaque) void {
            const dst: *Self = @ptrCast(@alignCast(dst_ctx));
            const src: *const Self = @ptrCast(@alignCast(src_ctx));
            // Either side may have been unshared or written since the copy
            // was made.
            if (dst.models.len == src.models.len and std.mem.eql(u32, dst.model_of, src.model_of)) {
                @memcpy(dst.models, src.models);
            } else for (0..dst.count) |id| {
                dst.model(id).* = src.model(id).*;
            }
            @memcpy(dst.instances, src.instances);
            if (comptime has_state) @memcpy(dst.states, src.states);
            if (comptime has_q) @memcpy(dst.q_tape, src.q_tape);
            if (comptime has_limit) {
                @memcpy(dst.lim_x, src.lim_x);
                dst.lim_active = src.lim_active;
            }
            if (comptime has_bp) dst.bp = src.bp;
            if (comptime has_src_brk) @memcpy(dst.src_brk, src.src_brk);
            dst.held_initial = src.held_initial;
            dst.committed_initial = src.committed_initial;
            dst.sim.analog_initial = !dst.held_initial;
        }

        fn setLimitActive(ctx: *anyopaque, active: bool) void {
            const self = of(ctx);
            self.lim_active = active;
        }

        /// A new batch sharing the template's tapes with private copies of
        /// all mutable state. `accepted` copies the template's history
        /// (`snapshot`); otherwise history starts fresh (`instantiate`).
        fn duplicate(ctx: *const anyopaque, gpa: Allocator, comptime accepted: bool) !Batch {
            const template: *const Self = @ptrCast(@alignCast(ctx));
            const self = try gpa.create(Self);
            self.* = template.*;
            self.bypass = null;
            self.owns_tapes = false;
            self.unshared = false;
            self.models_shared = &.{};
            if (!accepted) {
                self.held_initial = false;
                self.committed_initial = false;
                self.sim.analog_initial = true;
            }
            self.models = &.{};
            self.model_of = &.{};
            self.instances = &.{};
            if (comptime has_state) self.states = &.{};
            if (comptime has_q) self.q_tape = &.{};
            if (comptime has_limit) {
                self.lim_x = &.{};
                self.lim_active = if (accepted) template.lim_active else false;
            }
            if (comptime has_src_brk) self.src_brk = &.{};
            errdefer destroy(self, gpa);

            // Model/Instance are POD, so a byte copy is a deep copy.
            self.models = try gpa.dupe(D.Model, template.models);
            self.model_of = try gpa.dupe(u32, template.model_of);
            self.instances = try gpa.dupe(D.Instance, template.instances);
            if (comptime has_src_brk) self.src_brk = try gpa.dupe(SrcBrk, template.src_brk);
            if (comptime has_limit) {
                self.lim_x = try gpa.alloc(f64, self.count * LimMap(D).k);
                if (accepted) @memcpy(self.lim_x, template.lim_x) else @memset(self.lim_x, 0);
            }
            if (comptime has_q) {
                self.q_tape = try gpa.alloc(f64, self.count * comptime layout.lteSites(D).len);
                if (accepted) @memcpy(self.q_tape, template.q_tape) else @memset(self.q_tape, 0);
            }
            if (comptime has_state) {
                self.states = try gpa.alloc(D.State, self.count);
                if (accepted) {
                    @memcpy(self.states, template.states);
                } else {
                    for (self.states, self.instances, 0..) |*state, *instance, id|
                        state.* = D.initState(self.model(id), instance);
                    self.commitBirth();
                }
            }
            return self.binding();
        }

        fn destroy(ctx: *anyopaque, gpa: Allocator) void {
            const self = of(ctx);
            Bp.freeBypass(self);
            if (self.unshared) {
                staging_gpa.free(self.models);
                gpa.free(self.models_shared);
            } else gpa.free(self.models);
            gpa.free(self.model_of);
            if (comptime has_limit) gpa.free(self.lim_x);
            if (comptime has_src_brk) gpa.free(self.src_brk);
            gpa.free(self.instances);
            if (comptime has_state) gpa.free(self.states);
            if (self.owns_tapes) {
                gpa.free(self.original_ids);
                gpa.free(self.gath);
                gpa.free(self.rhs_idx);
                gpa.free(self.slots);
            }
            if (comptime has_q) gpa.free(self.q_tape);
            gpa.destroy(self);
        }

        /// This batch's GPU working set. The slices stay owned by the batch.
        fn gpuPayload(ctx: *anyopaque) GpuPayload {
            const self = of(ctx);
            return .{
                .kernel = comptime traits.kernelSym(D, "eval"),
                .count = self.count,
                .n_u = n_u,
                .models = std.mem.sliceAsBytes(self.models),
                .model_of = self.model_of,
                .instances = std.mem.sliceAsBytes(self.instances),
                .gath = self.gath,
                .rhs_idx = self.rhs_idx,
                .slots = self.slots,
                .lim_kernel = comptime if (traits.hasStateKernel(D)) traits.kernelSym(D, "lim") else "",
                .ctl_kernel = comptime if (traits.hasCtlKernel(D)) traits.kernelSym(D, "ctl") else "",
                .reduce_kernel = comptime traits.kernelSym(D, "reduce"),
                .lim_x = if (comptime has_limit) self.lim_x else &.{},
                .states = if (comptime has_state) std.mem.sliceAsBytes(self.states) else &.{},
                .lim_active = if (comptime has_limit) self.lim_active else false,
                .brk = if (comptime has_src_brk) std.mem.sliceAsBytes(self.src_brk) else &.{},
            };
        }
    };
}
