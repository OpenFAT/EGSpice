//! One transient between its attempts: the step machine. An accepted state
//! (t, dt, the order, the charge rings, the pending breakpoint) and one
//! Newton outcome in, the next attempt or the run's verdict out. The scalar
//! driver (`drive`), the CPU lane driver (`tran.runLanes`) and the GPU lane
//! driver (`gpu_drive.zig`) share every statement but the Newton solve;
//! each backend answers the numerics through an `ops` value (`HostOps`, the
//! oracle, or `gpu/lanes.zig`'s `Ops`), whose fields and calls here are
//! that seam's frozen surface. Pinned: `init` fills it in place.
//! Cites: ngspice dctran.c (step control, order promotion, breakpoint
//! landing, the first step), niconv.c, AlmostEqualUlps
//! (maths/misc/equality.c); Verilog-AMS LRM §5.10.2, §9.10, §9.17.2,
//! §12.31.3 and §12.36.
//!
//! Contents:
//! - `pub const TranHook = integrator.Hook(null)`
//! - `pub const Attempt = struct`
//! - `pub const Verdict = enum { go, stop, underflow }`
//! - `pub fn qHist(s: *Sim) [][]f64`, `pub fn qtHist(s: *Sim) [][]f64`, `pub fn lteHist(s: *Sim) [][]f64`
//! - `pub fn promoteMethod(s: *const Sim) Method`
//! - `pub fn lteIp(s: *const Sim) []const f64`, `pub fn lteG(s: *const Sim) []const f64`
//! - `pub const init = sim_init.init`
//! - `pub fn deinit(s: *Sim) void`
//! - `pub fn drive(s: *Sim) !bool`
//! - `pub fn begin(s: *Sim, ops: anytype) !?Attempt`
//! - `pub fn end(s: *Sim, a: *const Attempt, nr: converger.Result, ops: anytype) !Verdict`
//! - `pub fn floorRetry(s: *Sim, comptime reason: []const u8, dt: f64) Verdict`
//! - `pub fn finish(s: *Sim) !bool`
//! - `pub fn event(s: *const Sim, next_step: f64) worker.Event`
//! - `pub fn newtonOptions(options: Options) converger.Options`
//! - `pub fn predict(trial: []f64, cur: []const f64, prev: []const f64, xfact: f64) void`
//! - `pub fn nextBp(c: *root.Circuit, echo: []const f64, extra: ?f64, after: f64) ?f64`
//! - `pub fn methodAt(top: Method, k: u8) Method`
//! - `pub fn atStop(t: f64, t_stop: f64) bool`
//! - `pub fn almostEqualUlps(a: f64, b: f64, max_ulps: i64) bool`

const std = @import("std");
const assert = std.debug.assert;
const num = @import("core").numerics;
const converger = @import("solver").converger;
const Stream = @import("measure").Stream;
const root = @import("../types.zig");
const host_ops = @import("host_ops.zig");
const integrator = @import("integrator.zig");
const sim_init = @import("sim_init.zig");
const Tangents = @import("Tangents.zig");
const tran_types = @import("types.zig");

const HostOps = host_ops.HostOps;
const Method = tran_types.Method;
const Options = tran_types.Options;
const Waveform = tran_types.Waveform;

/// Borrowed: the circuit this run integrates, with its workspace `ws`.
ckt: *root.Circuit,
/// Borrowed: the operating point in, the last accepted solution out.
x: []f64,
probes: []const u32,
waveform: *Waveform,
options: Options,
/// Allocates every scratch slice below; `deinit` frees them.
allocator: std.mem.Allocator,
tangents: ?*Tangents,
stream: ?*Stream,
ws: *converger.Workspace,
n: usize,
has_charge: bool,
/// The two step buffers besides `x`; `cur`, `trial` and `prev` rotate
/// through the three.
x_try: []f64,
x_prev: []f64,
/// The method's highest order, and the charge slots its LTE reads
/// (max(4, top_order + 2)).
top_order: u8,
n_hist: usize,
/// G + ag0*C on the pattern, in the workspace (`ws.ensureAVals`).
a_vals: []f64,
/// Row-plane charge state, one block: the dynamic current `i_prev`, the
/// charge ring `q_ring` [cur, prev, ...] and gear's `g_row`.
row_state: []f64,
i_prev: []f64,
q_ring: [9][]f64,
g_row: []f64,
/// LTE states on the device charge tape; 0 runs the LTE on the rows.
/// The tape history mirrors the row one (`lte_state`, `qt_*`), with
/// `lte_zero` spans cleared and `lte_rows` appended (coupled inductors).
n_qt: usize,
lte_rows: []u32,
lte_zero: [][2]u32,
lte_state: []f64,
qt_i_prev: []f64,
qt_ring: [9][]f64,
qt_g: []f64,
/// The step ceiling, ngspice delmin and CKTminBreak, and the state-flip
/// landing resolution, all in seconds.
effective_dt_max: f64,
delmin: f64,
min_break: f64,
state_eps: f64,
/// Delayed breakpoint echoes (t + k*td), the first `n_echo` live.
echo_bps: [256]f64,
n_echo: usize,
echo_td: ?f64,
uic_bp: ?f64,
vpi: ?*const @import("device").Library.Vpi,
vpi_point: @import("device").Library.AnalogPoint,
/// The accepted point, the attempt and the point before `cur`.
cur: []f64,
trial: []f64,
prev: []f64,
/// The accepted time and the next attempt's step, in seconds; `dt_old`
/// holds the accepted steps before it, newest first.
t: f64,
dt: f64,
dt_old: [6]f64,
steps: u32,
stats_on: bool,
st: Stats,
/// The integration order of the next attempt, and the run of accepted
/// steps at `run_order` (gear-3..6 order control).
order: u8,
run_order: u8,
order_run: u32,
/// The breakpoint the last accepted step clamped toward, whether it
/// clamped, and the step it would have taken (`bp_save_dt`).
bp_next: ?f64,
bp_clamped: bool,
bp_save_dt: f64,
attempted_dt: f64,
/// VPI `again` verdicts in a row; the 60th ends the run.
vpi_again: u32,
/// Every `.meas` card resolved and the run stopped early.
resolved: bool,

/// Newton hook: companion RHS from the q plane, matrix G + ag0*C.
pub const TranHook = integrator.Hook(null);

/// What one attempt asks of Newton: `trial` from the predictor, solved at
/// `t_dev` under `hook` with `nr_opts`. `dts`, `ag` and `c` are the step's
/// own history and coefficients, which `end` reads again.
pub const Attempt = struct {
    t_dev: f64,
    method: Method,
    gear_hi: bool,
    dts: [7]f64,
    ag: [7]f64,
    c: integrator.Coeffs,
    hook: TranHook,
    nr_opts: converger.Options,
};

/// How `Sim.end` left the run: `.go` attempts again (a retry or the next
/// step), `.stop` ends it early with every card resolved, `.underflow` ends
/// it short of t_stop (dt below `dt_min`, or the VPI retry bound).
pub const Verdict = enum { go, stop, underflow };

/// Fills `s` in place (`sim_init.zig`); `deinit` undoes it.
pub const init = sim_init.init;

const Sim = @This();

/// ZP_TRAN_STATS step-economics counters: wall time in a slow transient is
/// attempts x Newton iterations x eval cost, and these say which factor.
const Stats = struct {
    attempts: u64 = 0,
    nr_iters: u64 = 0,
    rej_newton: u64 = 0,
    rej_state: u64 = 0,
    rej_vpi: u64 = 0,
    rej_lte: u64 = 0,
    order_drops: u64 = 0,
    bp_landings: u64 = 0,
    first_dt: f64 = 0,
    max_dt: f64 = 0,
    /// Accepted steps per integration order 1..6.
    by_order: [6]u64 = @splat(0),
};

pub fn qHist(s: *Sim) [][]f64 {
    return s.q_ring[0..s.n_hist];
}
pub fn qtHist(s: *Sim) [][]f64 {
    return s.qt_ring[0..s.n_hist];
}
/// The LTE history: per state when the tape is live, per row otherwise.
/// Same kernel either way, so the row path is the scalar oracle for the
/// tape path. The slice is the ring, so it follows the rotation.
pub fn lteHist(s: *Sim) [][]f64 {
    return if (s.n_qt > 0) s.qtHist() else s.qHist();
}
/// The method of the order-1-to-2 promotion probe.
pub fn promoteMethod(s: *const Sim) Method {
    return methodAt(s.options.method, 2);
}
pub fn lteIp(s: *const Sim) []const f64 {
    return if (s.n_qt > 0) s.qt_i_prev else s.i_prev;
}
pub fn lteG(s: *const Sim) []const f64 {
    return if (s.n_qt > 0) s.qt_g else s.g_row;
}

/// Frees the scratch `init` allocated and clears `land_rejects`.
pub fn deinit(s: *Sim) void {
    // Every exit reports, including underflow, cancellation and errors.
    // Output samples are deliberately separate from accepted solver steps.
    if (s.stats_on) {
        const st = s.st;
        std.debug.print(
            "tran-stats: complete={} resolved={} t={e:.17} stop={e:.17} samples={d} n_qt={d} accepted={d} attempts={d} nr_iters={d} rej[newton={d} lte={d} state={d} vpi={d}] order_drops={d} bp_landings={d} by_order={any} first_dt={e:.17} max_dt={e:.17} avg_dt={e:.17}\n",
            .{ s.t >= s.options.t_stop or atStop(s.t, s.options.t_stop), s.resolved, s.t, s.options.t_stop, s.waveform.len, s.n_qt, s.steps, st.attempts, st.nr_iters, st.rej_newton, st.rej_lte, st.rej_state, st.rej_vpi, st.order_drops, st.bp_landings, st.by_order, st.first_dt, st.max_dt, if (s.steps > 0) s.t / @as(f64, @floatFromInt(s.steps)) else 0 },
        );
    }
    const a = s.allocator;
    a.free(s.lte_state);
    a.free(s.lte_zero);
    a.free(s.lte_rows);
    a.free(s.row_state);
    a.free(s.x_prev);
    a.free(s.x_try);
    s.ckt.land_rejects = false;
    s.* = undefined;
}

/// The scalar loop: each attempt's Newton through `converger.run`, a
/// numerical failure retrying the step. Returns `finish`'s value, or
/// false when the run ended short of t_stop.
pub fn drive(s: *Sim) !bool {
    while (try s.begin(HostOps{})) |a| {
        const nr = converger.run(s.ckt, s.ws, s.trial, a.t_dev, a.nr_opts, a.hook) catch |err| switch (err) {
            error.QueryCancelled, error.OutOfMemory => return err,
            else => converger.Result{ .converged = false, .iterations = 0 }, // else: a numerical failure retries the step
        };
        switch (try s.end(&a, nr, HostOps{})) {
            .go => {},
            .stop => break,
            .underflow => return false,
        }
    }
    return s.finish();
}

/// The step loop's head: null once the run has reached t_stop or
/// `max_steps`, else the next attempt, with `trial` predicted and
/// limited and the device state published for it.
pub fn begin(s: *Sim, ops: anytype) !?Attempt {
    const options = s.options;
    if (!(s.t < options.t_stop and !atStop(s.t, options.t_stop) and s.steps < options.max_steps)) return null;
    const ckt = s.ckt;
    if (s.st.attempts != 0) try ckt.checkpoint(s.event(s.dt));
    s.attempted_dt = s.dt;
    const t = s.t;
    const dt = s.dt;
    // The time the devices evaluate at; OPtran holds it at 0.
    const t_dev = if (options.sources_at_zero) 0 else t + dt;
    // Publish the point this attempt aims at before anything evaluates
    // it: generated devices read Instance.abstime (§9.10 `$abstime`,
    // `ddt`), not the `t` argument. A rejected step comes back here with
    // the shrunken dt, so the last write before an accept is the accepted
    // dt. §5.10.2: initial_step is the analysis' first step, final_step
    // the one that lands on t_stop.
    ckt.setSimState(.{
        .t = t_dev,
        .dt = dt,
        .kind = .tran,
        .initial_step = s.steps == 0,
        .final_step = t + dt >= options.t_stop,
    });
    if (s.vpi) |v| {
        var p = s.vpi_point;
        p.t = t + dt;
        p.dt = dt;
        v.point(v.ctx, p);
    }
    const eff_method = methodAt(options.method, s.order);
    // Gear-3..6 solves its coefficients from the whole step history
    // and folds the older charges into g once per attempt.
    const gear_hi = s.order >= 3;
    var a: Attempt = .{
        .t_dev = t_dev,
        .method = eff_method,
        .gear_hi = gear_hi,
        .dts = [1]f64{dt} ++ s.dt_old,
        .ag = undefined,
        .c = undefined,
        .hook = undefined,
        .nr_opts = undefined,
    };
    a.c = if (gear_hi) blk: {
        a.ag = integrator.gearAg(s.order, &a.dts);
        break :blk .{ .ag0 = a.ag[0], .ag2 = 0 };
    } else integrator.coeffs(eff_method, dt, s.dt_old[0], options.xmu);
    ops.attempt(s, &a);
    a.nr_opts = newtonOptions(options);
    return a;
}

/// The step loop's body after Newton: rejects (Newton, a device's
/// `reject_at`, a state flip, the LTE, VPI) shrink `dt` and return
/// `.go`; an accepted step promotes or keeps the order, advances the
/// histories, commits, lands breakpoints, records and streams, and sets
/// the next dt.
pub fn end(s: *Sim, a: *const Attempt, nr: converger.Result, ops: anytype) !Verdict {
    const ckt = s.ckt;
    const options = s.options;
    const has_charge = s.has_charge;
    const q_hist = s.qHist();
    const t = s.t;
    const dt = s.dt;
    const t_dev = a.t_dev;
    const eff_method = a.method;
    const cf = a.c;
    const gear_hi = a.gear_hi;
    s.st.attempts += 1;
    s.st.nr_iters += nr.iterations;

    if (!nr.converged) {
        s.st.rej_newton += 1;
        // Restore FSM devices to the last accepted state.
        ops.revert(s);
        // Cut dt by 8 and drop to order 1 in one retry (dctran.c:815, :823).
        if (s.order > 1) s.st.order_drops += 1;
        s.order = 1;
        s.dt = dt / 8.0;
        return s.floorRetry("newton", dt);
    }

    // A device located an event inside this step and asked for the step
    // to end on it (`request_reject_at`): retry landing there. A time
    // within state_eps of the step's end counts as the end, the
    // resolution the flip query below keeps too.
    if (ops.rejectAt(s)) |tr| if (tr > t and t + dt - tr > s.state_eps and tr - t >= options.dt_min) {
        s.st.rej_state += 1;
        ops.revert(s);
        s.dt = tr - t;
        return .go;
    };

    // A device state flipped inside this step (a switch crossed its
    // threshold): reject and shrink so the conductance step lands within
    // state_eps of the crossing instead of smeared across dt.
    if (ops.flipped(s, a, dt > s.state_eps)) {
        s.st.rej_state += 1;
        ops.revert(s);
        s.order = 1;
        s.dt = @max(0.25 * dt, s.state_eps);
        return .go;
    }

    // The first accepted point skips CKTtrunc (dctran.c firsttime), so dt
    // repeats. §9.17.2 `$bound_step` is read after the step and applies
    // to the next one; devices only set it in `updateState`, so it cannot
    // be folded into effective_dt_max.
    var dt_next = if (s.steps == 0) dt else @min(dt * 2.0, s.effective_dt_max);
    if (ops.boundStep(s)) |bs| dt_next = @min(dt_next, bs);

    // CKTterr reads the charge of the published point. The converger
    // returns x_k+1 while the planes hold q(x_k) (or a JFNK matvec's x),
    // so re-read q at the solution: the LTE, advanceCurrent and the next
    // residual all see q(trial).
    // `q_fresh`: the converger's `updateStates` already stamped them.
    if (has_charge) ops.charges(s, a);
    const lte: integrator.LteIn = .{
        .dt = dt,
        .dt1 = s.dt_old[0],
        .dt2 = s.dt_old[1],
        .reltol = options.tol.reltol,
        .abstol = options.tol.abstol,
        .chgtol = options.tol.chgtol,
        .trtol = options.tol.trtol,
    };
    // This step's own bound, read again by the gear order control.
    var del: f64 = undefined;
    if (has_charge and s.steps > 0) {
        del = if (gear_hi)
            ops.boundGear(s, a, eff_method, lte)
        else
            ops.bound(s, a, eff_method, lte);
        if (!(del >= 0.9 * dt)) {
            s.st.rej_lte += 1;
            // Reconstruct only from captured history, before rollback.
            // GPU lanes own separate histories; report that limitation
            // explicitly rather than inspect stale host ring storage.
            if (s.ckt.knobs.tran_lte_trace and s.st.rej_lte <= 16) {
                if (@TypeOf(ops) == HostOps) host_ops.traceLte(s, a, nr, lte, del) else std.debug.print("tran-lte: captured-history diagnostic unavailable for this lane backend\n", .{});
            }
            ops.revert(s);
            // Retry at the LTE's dt and the same order (dctran.c:966
            // `CKTdelta = newdelta`); only a Newton failure drops the
            // order. del < 0.9*dt here, so every retry shrinks. A NaN
            // bound (a non-finite charge) retries as a Newton failure.
            if (std.math.isNan(del)) {
                s.order = 1;
                s.dt = dt / 8.0;
            } else s.dt = del;
            return s.floorRetry("lte", dt);
        }
        // Growth capped at 2x per accepted step, as in ngspice, and
        // lower at gear-3..6 (`maxGrowth`).
        dt_next = @min(@max(del, options.dt_min), integrator.maxGrowth(eff_method) * dt, s.effective_dt_max);
    }

    // LRM 12.31.3 acbConvergenceTest, last before acceptance: a rejection
    // (or 12.36 vpiRejectTransientStep) backs up to half the step from
    // the last accepted point, as VerA's own walk does; 12.36
    // vpiTransientFailConverge solves the same step again.
    if (s.vpi) |v| switch (v.verdict(v.ctx)) {
        .accept => s.vpi_again = 0,
        .reject => {
            s.st.rej_vpi += 1;
            ops.revert(s);
            s.dt = dt / 2.0;
            if (s.dt < options.dt_min) return .underflow;
            return .go;
        },
        .again => {
            s.st.rej_vpi += 1;
            ops.revert(s);
            s.vpi_again += 1;
            if (s.vpi_again == 60) return .underflow;
            return .go;
        },
    };

    // Accepted from here on; the history below is still this step's.
    if (s.tangents) |tg| {
        try tg.advance(ckt, s.ws, s.a_vals, s.trial, t_dev, .{ .method = eff_method, .c = cf, .q1 = q_hist[1], .q2 = q_hist[2], .i = s.i_prev });
        if (t + dt >= options.t_start) try tg.record(t + dt, s.probes);
    }

    if (has_charge) {
        // Order promotion (dctran.c:901-913): recompute the trunc at
        // order 2 and adopt min(2*dt, del2) as the next dt whether or not
        // the order changes, as ngspice's `CKTdelta = newdelta` does.
        s.st.by_order[s.order - 1] += 1;
        s.order_run = if (s.order == s.run_order) s.order_run + 1 else 1;
        s.run_order = s.order;
        if (s.steps > 0 and s.order == 1) {
            const trial_del = ops.bound(s, a, s.promoteMethod(), lte);
            // A NaN probe (prev3 non-finite) keeps BE and the step.
            if (!std.math.isNan(trial_del)) {
                const nd2 = @min(2.0 * dt, trial_del);
                if (nd2 > 1.05 * dt) s.order = 2;
                dt_next = @min(@max(nd2, options.dt_min), s.effective_dt_max);
                if (ops.boundStep(s)) |bs| dt_next = @min(dt_next, bs);
            }
        } else if (s.steps > 0 and s.top_order >= 3 and s.order_run > s.order) {
            // Gear-3..6 order control, an espice policy (dctran.c stops
            // at order 2): after order + 1 accepted steps at one order,
            // move one order up or down (not below 2) when that order's
            // CKTterr bound beats this one's by 5%, the margin of the
            // 1-to-2 promotion; the next step is then that bound under
            // the new order's `maxGrowth`.
            // ponytail: one order per change; DASSL-style selection on
            // error trends would move faster.
            var best = del;
            var best_order = s.order;
            for ([_]u8{ s.order + 1, s.order - 1 }) |k| {
                if (k > s.top_order or k < 2) continue;
                const d = ops.boundGear(s, a, methodAt(options.method, k), lte);
                if (d > 1.05 * best) {
                    best = d;
                    best_order = k;
                }
            }
            if (best_order != s.order) {
                s.order = best_order;
                dt_next = @min(@max(best, options.dt_min), integrator.maxGrowth(methodAt(options.method, s.order)) * dt, s.effective_dt_max);
                if (ops.boundStep(s)) |bs| dt_next = @min(dt_next, bs);
            }
        }

        // The dynamic current under the method actually used. Both index
        // spaces run it: ngspice keeps it per state in
        // CKTstates[0][qcap+1], where CKTterr's volttol reads it.
        // Then [cur, prev, prev2, ...] -> [stale, cur, prev, ...].
        ops.advance(s, a);
    }
    std.mem.copyBackwards(f64, s.dt_old[1..], s.dt_old[0 .. s.dt_old.len - 1]);
    s.dt_old[0] = dt;

    // Rotate the slices; accepted states need no copy.
    const stale = s.prev;
    s.prev = s.cur;
    s.cur = s.trial;
    s.trial = stale;
    s.t = t + dt;
    if (s.steps == 0) s.st.first_dt = dt;
    s.st.max_dt = @max(s.st.max_dt, dt);
    s.steps += 1;
    const t_new = s.t;
    // §4.5.2 delay-line bookkeeping, once per accepted point
    // (`Hooks.commit_state`), then the commit that latches every stage.
    ops.commit(s);

    // Landed on a breakpoint: drop to BE and resume at
    // 0.1*min(saveDelta, gap to the next break), spice3 dctran's rule,
    // which resolves paired edges instead of stepping over them. The
    // history is kept; the promotion check re-promotes next step. A step
    // that was not clamped lands too when it ends within 100 ulps of the
    // breakpoint or delmin short of it (dctran.c:559); only min_break
    // separates a clamped landing from its target. The line echoes are
    // left out of that: they stand in for traload breakpoints ngspice
    // sets only on a sharp input, and tran/bench_tline_delay_line steps
    // through its t = td echo without a cut there.
    if (s.bp_next) |bp| {
        const landed = if (s.bp_clamped)
            @abs(t_new - bp) <= s.min_break
        else
            (bp - t_new <= s.delmin or almostEqualUlps(t_new, bp, 100)) and
                std.mem.indexOfScalar(f64, s.echo_bps[0..s.n_echo], bp) == null;
        if (landed) {
            s.st.bp_landings += 1;
            s.order = 1;
            // Re-emit one line delay later; dedupe within min_break and
            // drop when the table is full.
            if (s.echo_td) |td_| {
                const e = t_new + td_;
                if (e < options.t_stop and s.n_echo < s.echo_bps.len) {
                    const dup = for (s.echo_bps[0..s.n_echo]) |old| {
                        if (@abs(old - e) <= s.min_break) break true;
                    } else false;
                    if (!dup) {
                        s.echo_bps[s.n_echo] = e;
                        s.n_echo += 1;
                    }
                }
            }
            var shrink = s.bp_save_dt;
            if (nextBp(ckt, s.echo_bps[0..s.n_echo], s.uic_bp, t_new + s.min_break)) |nb| shrink = @min(shrink, nb - t_new);
            dt_next = @min(dt_next, 0.1 * shrink);
        }
        s.bp_next = null;
    }

    // Stateful-charge devices: the commits above can move a device's q
    // away from what q_hist recorded. Re-read it so the next residual
    // starts from q(cur); otherwise it opens on alpha*(q_committed -
    // q_recorded), which doubles with every dt halving. The i_prev
    // correction is the same alpha*dq for every method. dq is not zero
    // after the pre-LTE re-read: dropping this changes the output of
    // txl2_3_line, hfet_inverter, mesa_oscillator and mos6_inverter.
    // `evalQ` computes the same q bits as `eval`, charges only.
    ops.rebase(s, a);

    // ngspice neither steps onto TSTART nor prints it: dctran.c dumps
    // the accepted points at t >= CKTinitTime and nothing else (#167).
    if (t_new >= options.t_start) try s.waveform.record(t_new, ops.accepted(s), s.probes);
    // Before the breakpoint search below, so a time a callback forces
    // from this point is landed on.
    if (s.vpi) |v| v.accept(v.ctx, false, t_new >= options.t_stop or atStop(t_new, options.t_stop));
    if (s.stream) |sm| if (t_new >= options.t_start) {
        sm.feed(t_new, ops.accepted(s), s.waveform.data());
        if (options.point.abort_after) |bound| if (t_new > bound and sm.unresolved()) return error.MeasureBoundExceeded;
        if (options.point.early_stop and sm.open == 0 and (s.tangents == null or s.tangents.?.open() == 0)) {
            s.resolved = true;
            return .stop;
        }
    };

    // Clamp dt to land on the next breakpoint, skipping those within
    // min_break of now (ngspice CKTminBreak merge).
    s.bp_next = nextBp(ckt, s.echo_bps[0..s.n_echo], s.uic_bp, t_new + s.min_break);
    s.bp_clamped = false;
    if (s.bp_next) |bp| {
        const dt_to_bp = bp - t_new;
        if (dt_to_bp < dt_next) {
            s.bp_save_dt = dt_next;
            dt_next = dt_to_bp;
            s.bp_clamped = true;
        }
    }

    s.dt = dt_next;
    if (t_new + s.dt > options.t_stop) s.dt = options.t_stop - t_new;
    return .go;
}

/// A rejected step of `dt` has proposed `s.dt`: below `dt_min` it is
/// retried once at the floor, since a sub-floor proposal does not prove
/// the floor fails, and a rejection from the floor (or a floor that does
/// not advance t) ends the run (dctran.c:974).
pub fn floorRetry(s: *Sim, comptime reason: []const u8, dt: f64) Verdict {
    const t = s.t;
    const dt_min = s.options.dt_min;
    if (s.dt < dt_min and dt > dt_min and t + dt_min > t) {
        if (s.stats_on) std.debug.print("tran-floor-retry: reason=" ++ reason ++ " t={e:.17} old_dt={e:.17} proposed={e:.17} retry={e:.17}\n", .{ t, dt, s.dt, dt_min });
        s.dt = dt_min;
    }
    if (!(s.dt < dt_min)) return .go;
    if (s.stats_on) std.debug.print(
        "tran-stats: DT UNDERFLOW (" ++ reason ++ ") at t={e:.6} dt={e:.3} accepted={d} attempts={d} nr_iters={d}\n",
        .{ t, s.dt, s.steps, s.st.attempts, s.st.nr_iters },
    );
    return .underflow;
}

/// The step loop's tail: the last checkpoint, the stats line, `x`
/// holding the last accepted solution. True on reaching t_stop or once
/// every card resolved.
pub fn finish(s: *Sim) !bool {
    const options = s.options;
    try s.ckt.checkpoint(s.event(if (s.t < options.t_stop) s.dt else 0));
    if (s.cur.ptr != s.x.ptr) num.copySimd(s.x, s.cur);
    return s.resolved or s.t >= options.t_stop or atStop(s.t, options.t_stop);
}

/// The progress checkpoint at the last accepted point, `next_step` the
/// step after it.
pub fn event(s: *const Sim, next_step: f64) @import("../worker.zig").Event {
    return .{ .phase = .transient, .completed = s.st.attempts, .simulation_time = s.t, .step_size = s.attempted_dt, .next_step = next_step, .accepted = s.steps };
}

/// The Newton options of every transient step. ngspice's transient Newton
/// accepts on the per-row delta test alone (niconv.c): a step that has
/// converged in x but leaves a residual on a row, e.g. a Meyer gate cap's
/// curvature at a source-driven node, is not iterated again.
pub fn newtonOptions(options: Options) converger.Options {
    var o = converger.optionsFromTolerances(options.tol, options.tol.itl4);
    o.residual_tol = std.math.inf(f64);
    return o;
}

/// Writes the MODEINITPRED first iterate, trial = cur + xfact·(cur - prev),
/// into `trial[0..cur.len]`; `prev` must be at least as long as `cur`.
pub fn predict(trial: []f64, cur: []const f64, prev: []const f64, xfact: f64) void {
    for (trial[0..cur.len], cur, prev[0..cur.len]) |*t, c, p| t.* = c + xfact * (c - p);
}

/// The earliest breakpoint at or after `after` among the devices', the line
/// echoes and `extra`; null when there is none.
pub fn nextBp(c: *root.Circuit, echo: []const f64, extra: ?f64, after: f64) ?f64 {
    var best = std.math.inf(f64);
    if (c.nextBreakpoint(after)) |bp| best = bp;
    for (echo) |e| {
        if (e >= after and e < best) best = e;
    }
    if (extra) |e| {
        if (e >= after and e < best) best = e;
    }
    return if (best == std.math.inf(f64)) null else best;
}

/// The method an order-`k` step integrates with: BE at 1, else `top` itself
/// when it stops at order 2 (trap, gear-2, or BE), else gear-k.
pub fn methodAt(top: Method, k: u8) Method {
    if (k == 1) return .backward_euler;
    return if (integrator.order(top) <= 2) top else integrator.gear(k);
}

/// ngspice's end test (dctran.c:503, AlmostEqualUlps(t, tfinal, 100)): a
/// step that lands within 100 ulps of t_stop ends the run. Without it a
/// breakpoint sum that rounds one ulp short of t_stop leaves a 1e-23 s step
/// that fails as TimestepTooSmall.
pub fn atStop(t: f64, t_stop: f64) bool {
    return almostEqualUlps(t, t_stop, 100);
}

/// ngspice's AlmostEqualUlps (maths/misc/equality.c): `a` and `b` are at
/// most `max_ulps` representable doubles apart, across zero included.
pub fn almostEqualUlps(a: f64, b: f64, max_ulps: i64) bool {
    if (a == b) return true;
    const lex = struct {
        fn f(x: f64) i128 {
            const i: i64 = @bitCast(x);
            return if (i < 0) @as(i128, std.math.minInt(i64)) - i else i;
        }
    }.f;
    return @abs(lex(a) - lex(b)) <= max_ulps;
}

// ---- private ----
test "atStop: within 100 ulps of t_stop, not past them" {
    const t_stop = 2.5e-7;
    try std.testing.expect(atStop(std.math.nextAfter(f64, t_stop, 0), t_stop));
    try std.testing.expect(!atStop(t_stop * (1 - 1e-12), t_stop));
}

test {
    _ = @import("floor_retry_test.zig");
}
