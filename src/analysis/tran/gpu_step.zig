//! Device step control for the GPU transient lanes: the lanes' `Sim`s
//! after `Sim.init` in, `LK.LaneStep` rows to the device and launches of up
//! to `GpuLanes.max_rounds` rounds out, then each launch's accepted rows
//! drained back into each lane's waveform, stream and step state. Also the
//! lane outcome bookkeeping and the per-lane fan-out both GPU loops share.
//! Cites: none (phase 4 of dev/analysis/lane-batched-simulation.md).
//!
//! Contents:
//! - `pub const GpuLane = struct`, with `done`, `kill`, `failAll`
//! - `pub fn deviceStepOk(ckt: *const root.Circuit, opts: Options) bool`
//! - `pub fn laneStep(s: *const Sim, opts: Options, max_iter: u16) LK.LaneStep`
//! - `pub fn deviceLoop(p_in: DeviceLanes, nr_opts: converger.Options, progress: anytype) void`
//! - `pub fn gpuError(e: anyerror) LaneError`
//! - `pub fn forLanesThreads(n: usize) usize`
//! - `pub fn forLanes(n: usize, ctx: anytype, comptime f: fn (@TypeOf(ctx), usize) void) void`

const std = @import("std");
const converger = @import("solver").converger;
const LK = @import("solver").lane_kernels;
const measure = @import("measure");
const root = @import("../types.zig");
const gpu_lanes = @import("../gpu/lanes.zig");
const Run = @import("Run.zig");
const Sim = @import("Sim.zig");
const tran = @import("tran.zig");

const LaneError = tran.LaneError;
const LaneOutcome = tran.LaneOutcome;
const Options = tran.Options;
const run = tran.run;

/// `runGpuLanes`' lane outcomes: a lane leaves `live` once, with its
/// result or error in `out`.
pub const GpuLane = struct {
    pub fn done(rs: []Run, on: []bool, o: []LaneOutcome, l: usize, ok: bool) void {
        o[l] = rs[l].take(ok);
        rs[l].deinit();
        on[l] = false;
    }
    pub fn kill(rs: []Run, on: []bool, o: []LaneOutcome, l: usize, err: LaneError) void {
        o[l] = err;
        rs[l].deinit();
        on[l] = false;
    }
    pub fn failAll(rs: []Run, on: []bool, o: []LaneOutcome, err: LaneError) void {
        for (0..rs.len) |l| if (on[l]) kill(rs, on, o, l, err);
    }
};

/// Lanes per claim in `forLanes`, and the fewest lanes per thread.
const lanes_per_claim = 64;

/// What `deviceLoop` shares with its per-lane drain.
const DeviceLanes = struct {
    runs: []Run,
    live: []bool,
    out: []LaneOutcome,
    ctxs: []const root.RunCtx,
    opts: Options,
    d: *gpu_lanes.GpuLanes,
    gpa: std.mem.Allocator,
    /// Per lane, what the drain left for the serial tail (`Fin`).
    fin: []Fin = &.{},
    /// Ring slots the last launch filled.
    rounds: usize = 1,
    /// The ring slot of each `.stop` lane's resolving row.
    stop_slot: []u8 = &.{},
    /// The `.meas` passes run on the device (`GpuLanes.initMeasures`).
    meas: bool = false,
    /// Per lane, its device passes all ended in an earlier launch.
    meas_done: []bool = &.{},

    const Fin = enum(u8) { none, finish, underflow, parked, stop };

    /// `Sim.end`'s tail for each of lane l's accepted rows of the launch,
    /// in round order: record, stream, the measure bound and early stop;
    /// then what the device's last status asks.
    fn drain(p: *const DeviceLanes, l: usize) void {
        p.fin[l] = .none;
        if (!p.live[l]) return;
        const s = &p.runs[l].sim;
        const opts = p.opts;
        for (0..p.rounds) |r| {
            const rep = p.d.slotReport(r, l);
            if (rep.event != LK.step_accept) continue;
            const t_new = rep.t_new;
            const row = p.d.slotRow(r, l);
            if (t_new >= opts.t_start) s.waveform.record(t_new, row, s.probes) catch |e| {
                GpuLane.kill(p.runs, p.live, p.out, l, e);
                return p.d.requestStop(l);
            };
            if (s.stream) |sm| if (t_new >= opts.t_start) {
                if (p.meas) {
                    // The device fed the row: the measure bound and early
                    // stop read the round that finished every pass.
                    const at = p.d.measResolved(l);
                    if (opts.point.abort_after) |bound| if (t_new > bound and !p.meas_done[l] and (at == null or at.? > r)) {
                        GpuLane.kill(p.runs, p.live, p.out, l, error.MeasureBoundExceeded);
                        return p.d.requestStop(l);
                    };
                    if (opts.point.early_stop) if (at) |a| if (a == r) {
                        s.resolved = true;
                        p.fin[l] = .stop;
                        p.stop_slot[l] = a;
                        return;
                    };
                    continue;
                }
                sm.feed(t_new, row, s.waveform.data());
                if (opts.point.abort_after) |bound| if (t_new > bound and sm.unresolved()) {
                    GpuLane.kill(p.runs, p.live, p.out, l, error.MeasureBoundExceeded);
                    return p.d.requestStop(l);
                };
                if (opts.point.early_stop and sm.open == 0) {
                    s.resolved = true;
                    p.fin[l] = .stop;
                    p.stop_slot[l] = @intCast(r);
                    return;
                }
            };
        }
        if (p.meas and p.d.measResolved(l) != null) p.meas_done[l] = true;
        const rep = p.d.slotReport(p.rounds - 1, l);
        p.fin[l] = switch (rep.status) {
            LK.step_done => .finish,
            LK.step_underflow => .underflow,
            LK.step_parked => .parked,
            else => .none,
        };
    }

    /// A parked lane (`reject_at` or `$bound_step`) reruns alone on the CPU.
    fn rerun(p: *const DeviceLanes, l: usize) void {
        if (p.fin[l] != .parked) return;
        p.runs[l].deinit();
        p.live[l] = false;
        p.d.detached[l] = true;
        p.out[l] = run(&p.ctxs[l], p.opts);
    }
};

/// Whether `runGpuLanes` hands step control to the device (`LK.LaneStep`):
/// not under `ESPICE_GPU_HOST_STEP`, its oracle, nor under the step
/// diagnostics, which print from `Sim` mid-step.
pub fn deviceStepOk(ckt: *const root.Circuit, opts: Options) bool {
    const k = ckt.knobs;
    return !k.gpu_host_step and !k.tran_stats and !k.tran_lte_trace and opts.tangents.len == 0;
}

/// Lane `s`'s `LK.LaneStep` as `Sim.init` left it.
pub fn laneStep(s: *const Sim, opts: Options, max_iter: u16) LK.LaneStep {
    const Ctl = @import("device").abi.StateCtlOp;
    const tran_kind: @FieldType(@import("device").abi.SimState, "kind") = .tran;
    const st = s.st;
    return .{
        .t_stop = opts.t_stop,
        .dt_min = opts.dt_min,
        .xmu = opts.xmu,
        .dt_max = s.effective_dt_max,
        .delmin = s.delmin,
        .min_break = s.min_break,
        .state_eps = s.state_eps,
        .reltol = opts.tol.reltol,
        .abstol = opts.tol.abstol,
        .chgtol = opts.tol.chgtol,
        .trtol = opts.tol.trtol,
        .t = s.t,
        .dt = s.dt,
        .dt_old = s.dt_old,
        .attempted_dt = s.attempted_dt,
        .bp_next = s.bp_next orelse std.math.inf(f64),
        .bp_save_dt = s.bp_save_dt,
        .t_new = s.t,
        .dt_next = s.dt,
        .attempts = st.attempts,
        .nr_iters = st.nr_iters,
        .rej_newton = st.rej_newton,
        .rej_state = st.rej_state,
        .rej_lte = st.rej_lte,
        .order_drops = st.order_drops,
        .bp_landings = st.bp_landings,
        .by_order = st.by_order,
        .first_dt = st.first_dt,
        .max_dt = st.max_dt,
        .steps = s.steps,
        .order_run = s.order_run,
        .max_steps = opts.max_steps,
        .max_iter = max_iter,
        .iteration = s.ckt.sim.iteration,
        .order = s.order,
        .run_order = s.run_order,
        .bp_clamped = @intFromBool(s.bp_clamped),
        .status = LK.step_run,
        .top = switch (opts.method) {
            .backward_euler => 0,
            .trapezoidal => 1,
            .gear_2 => 2,
            else => unreachable, // gear-3..6 are not lane-eligible
        },
        .charge = @intFromBool(s.has_charge),
        .state_q = @intFromBool(s.has_charge and s.ckt.has_state_q),
        .event = LK.step_idle,
        .op_revert = 1 + @as(u8, @backingInt(Ctl.revert)),
        .op_commit = 1 + @as(u8, @backingInt(Ctl.commit)),
        .kind_tran = @backingInt(tran_kind),
    };
}

/// `runGpuLanes`' rounds under device step control: `deviceLaunch`es of up
/// to `GpuLanes.max_rounds` rounds, the drain of their accepted rows in
/// round order, then the lanes that left. The worker's checkpoint fires
/// once per launch, from the first live lane.
pub fn deviceLoop(p_in: DeviceLanes, nr_opts: converger.Options, progress: anytype) void {
    var p = p_in;
    const g = p.runs.len;
    const d = p.d;
    p.fin = p.gpa.alloc(DeviceLanes.Fin, g) catch return GpuLane.failAll(p.runs, p.live, p.out, error.OutOfMemory);
    defer p.gpa.free(p.fin);
    p.stop_slot = p.gpa.alloc(u8, g) catch return GpuLane.failAll(p.runs, p.live, p.out, error.OutOfMemory);
    defer p.gpa.free(p.stop_slot);
    p.meas_done = p.gpa.alloc(bool, g) catch return GpuLane.failAll(p.runs, p.live, p.out, error.OutOfMemory);
    defer p.gpa.free(p.meas_done);
    @memset(p.meas_done, false);
    // Early stop and the measure bound are the drain's verdicts, so a lane
    // they end mid-launch has run past its resolving row. That is unseen
    // (its rows stop there, its step state comes from that row's report,
    // and its x is scratch it frees) unless a device can latch a status
    // `refuseLatched` reads off the device rows: then one round per launch.
    const host_stops = for (p.runs) |*r| {
        if (r.sim.stream != null and (p.opts.point.early_stop or p.opts.point.abort_after != null)) break true;
    } else false;
    const latches = for (p.runs[0].sim.ckt.batches) |b| {
        if (b.hooks.status != null) break true;
    } else false;
    const max_rounds: usize = if ((host_stops and latches) or !d.multiRound()) 1 else gpu_lanes.GpuLanes.max_rounds;
    // Rounds in the next launch, doubling up to the ring.
    var next: usize = 1;
    while (std.mem.indexOfScalar(bool, p.live, true)) |lead| {
        // No lane has attempted before the first launch.
        if (progress) |cb| if (d.prof.rounds != 0) {
            const rep = d.slotReport(p.rounds - 1, lead);
            if (rep.attempts != 0) cb.checkpoint(.{
                .phase = .transient,
                .completed = rep.attempts,
                .simulation_time = rep.t,
                .step_size = rep.attempted_dt,
                .next_step = rep.dt,
                .accepted = rep.steps,
            }) catch |e| return GpuLane.failAll(p.runs, p.live, p.out, e);
        };
        d.prof.lap(.host);
        p.rounds = next;
        d.deviceLaunch(nr_opts, @intCast(p.rounds)) catch |e| return GpuLane.failAll(p.runs, p.live, p.out, gpuError(e));
        d.prof.lap(.newton);
        const t_drain = gpu_lanes.Prof.now();
        forLanes(g, @as(*const DeviceLanes, &p), DeviceLanes.drain);
        if (d.prof.on) d.prof.drain_ns += gpu_lanes.Prof.now() - t_drain;
        // The lanes that left, serially. A lane's x is scratch it frees, so
        // only its step state comes back: the device's for a lane the device
        // ended, its resolving row's report for one early stop ended.
        for (p.fin, 0..) |f, l| if (f == .underflow or f == .stop) d.requestStop(l);
        var left = false;
        for (p.fin) |f| left = left or f != .none;
        if (left) {
            const steps = p.gpa.alloc(LK.LaneStep, g) catch return GpuLane.failAll(p.runs, p.live, p.out, error.OutOfMemory);
            defer p.gpa.free(steps);
            d.fetchSteps(steps) catch |e| return GpuLane.failAll(p.runs, p.live, p.out, gpuError(e));
            // The device passes of the lanes that left go back to their streams.
            if (p.meas) {
                d.fetchMeasures() catch |e| return GpuLane.failAll(p.runs, p.live, p.out, gpuError(e));
                const lane_passes = p.gpa.alloc(measure.passes.Dev, d.n_meas) catch return GpuLane.failAll(p.runs, p.live, p.out, error.OutOfMemory);
                defer p.gpa.free(lane_passes);
                for (p.fin, 0..) |f, l| if (f == .finish or f == .stop or f == .underflow) if (p.runs[l].sim.stream) |sm| {
                    d.lanePasses(l, lane_passes);
                    sm.adoptScans(lane_passes);
                };
            }
            for (p.fin, 0..) |f, l| switch (f) {
                .none, .parked => {},
                .finish, .stop => {
                    const s = &p.runs[l].sim;
                    adoptStep(s, steps[l]);
                    if (f == .stop) {
                        const rep = d.slotReport(p.stop_slot[l], l);
                        s.t = rep.t;
                        s.dt = rep.dt;
                        s.attempted_dt = rep.attempted_dt;
                        s.steps = rep.steps;
                        s.st.attempts = rep.attempts;
                    }
                    const ok = s.finish() catch |e| {
                        GpuLane.kill(p.runs, p.live, p.out, l, e);
                        continue;
                    };
                    GpuLane.done(p.runs, p.live, p.out, l, ok);
                },
                .underflow => {
                    adoptStep(&p.runs[l].sim, steps[l]);
                    GpuLane.done(p.runs, p.live, p.out, l, false);
                },
            };
            forLanes(g, @as(*const DeviceLanes, &p), DeviceLanes.rerun);
        }
        next = @min(2 * p.rounds, max_rounds);
        d.prof.lap(.host);
    }
}

/// A driver failure mid-run, as the lanes' error: out of memory stays
/// itself; anything else cancels the batch.
pub fn gpuError(e: anyerror) LaneError {
    return if (e == error.OutOfMemory) error.OutOfMemory else error.QueryCancelled;
}

/// Threads `forLanes` uses for `n` lanes: one below 256 lanes, so small
/// batches keep running serially on the analysis thread.
pub fn forLanesThreads(n: usize) usize {
    if (@import("builtin").single_threaded or n < 256) return 1;
    const cpus = std.Thread.getCpuCount() catch 1;
    return @max(1, @min(cpus, n / lanes_per_claim, 64));
}

/// `f(ctx, l)` for every lane l < n, claimed `lanes_per_claim` at a time by
/// `forLanesThreads(n)` threads. `f` must touch only lane l's state, so
/// the split cannot change any lane's result.
// ponytail: threads spawned per pass, two passes per round; a persistent
// pool if spawning ever shows beside the passes' 20-40 ms at 10k lanes.
pub fn forLanes(n: usize, ctx: anytype, comptime f: fn (@TypeOf(ctx), usize) void) void {
    const t = forLanesThreads(n);
    if (t <= 1) {
        for (0..n) |l| f(ctx, l);
        return;
    }
    const Work = struct {
        fn run(next: *std.atomic.Value(usize), count: usize, c: @TypeOf(ctx)) void {
            while (true) {
                const lo = next.fetchAdd(lanes_per_claim, .monotonic);
                if (lo >= count) return;
                for (lo..@min(lo + lanes_per_claim, count)) |l| f(c, l);
            }
        }
    };
    var next = std.atomic.Value(usize).init(0);
    var threads: [63]std.Thread = undefined;
    var spawned: usize = 0;
    // A failed spawn leaves fewer helpers; the calling thread still claims.
    while (spawned < t - 1) : (spawned += 1) {
        threads[spawned] = std.Thread.spawn(.{ .stack_size = @import("../worker.zig").stack_size }, Work.run, .{ &next, n, ctx }) catch break;
    }
    Work.run(&next, n, ctx);
    for (threads[0..spawned]) |th| th.join();
}

// ---- private ----

/// The device's step state back into lane `s` before `finish`: what
/// `finish`, the stats line and `take` read.
fn adoptStep(s: *Sim, st: LK.LaneStep) void {
    s.t = st.t;
    s.dt = st.dt;
    s.dt_old = st.dt_old;
    s.attempted_dt = st.attempted_dt;
    s.steps = st.steps;
    s.order = st.order;
    s.run_order = st.run_order;
    s.order_run = st.order_run;
    s.bp_next = if (st.bp_next == std.math.inf(f64)) null else st.bp_next;
    s.bp_clamped = st.bp_clamped != 0;
    s.bp_save_dt = st.bp_save_dt;
    s.st.attempts = st.attempts;
    s.st.nr_iters = st.nr_iters;
    s.st.rej_newton = st.rej_newton;
    s.st.rej_state = st.rej_state;
    s.st.rej_lte = st.rej_lte;
    s.st.order_drops = st.order_drops;
    s.st.bp_landings = st.bp_landings;
    s.st.by_order = st.by_order;
    s.st.first_dt = st.first_dt;
    s.st.max_dt = st.max_dt;
}
