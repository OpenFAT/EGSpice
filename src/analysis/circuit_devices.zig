//! A circuit's device-state walks: the Newton bookkeeping (limits, seeds,
//! iteration count), the accepted-state protocol (stage, commit, revert,
//! query), the published simulation state, and parameter moves (temperature,
//! homotopy, recompute). Each routes to the attached GPU context, the
//! threaded stamp (ParEval) or the batches serially. `Circuit` exposes each
//! function here as a method.
//! In: a circuit and an iterate. Out: device state moved, limits applied,
//! reject times.
//! Cites VAMS LRM §9.4 (display tasks), §9.15 (`$simparam`), §9.17.2
//! (`$bound_step`), §12.31.3 (VPI forced times).
//!
//! Contents:
//! - `pub fn evalFollows(self: *const Circuit, x: []const f64, t: f64, charge: bool) void`
//! - `pub fn evalFollowsFirst(self: *const Circuit, x: []const f64, t: f64) void`
//! - `pub fn applyLimits(self: *const Circuit, x: []f64, x_old: []const f64) bool`
//! - `pub fn seedIc`, `pub fn seedNodesets`, `pub fn seedJunctions`, `pub fn clearLimits`
//! - `pub fn beginSolve(self: *Circuit) void`
//! - `pub fn advanceIteration(self: *Circuit, previous_x: []const f64) void`
//! - `pub fn predictFirstIterate(self: *const Circuit, trial: []f64, cur: []const f64, prev: []const f64, xfact2: f64) void`
//! - `pub fn checkConvergence(self: *const Circuit, x: []const f64) bool`
//! - `pub fn updateStates(self: *Circuit, x: []const f64) ?f64`
//! - `pub fn commitStates(self: *const Circuit, x: []const f64) ?f64`
//! - `pub fn stateCtl(self: *Circuit, sop: StateCtlOp) bool`
//! - `pub fn minDelay`, `pub fn boundStep`, `pub fn nextBreakpoint`
//! - `pub fn setCircuitTemp(self: *Circuit, temp_c: f32) void`
//! - `pub fn setHomotopy(self: *Circuit, gmin: f64, source_scale: f64) void`
//! - `pub fn paramsMoved(self: *Circuit) void`
//! - `pub fn setBypass(self: *Circuit, tol: ?BypassTol) void`
//! - `pub fn setSimState(self: *Circuit, st: SimState) void`
//! - `pub fn recompute(self: *Circuit) error{TopologyChanged}!void`
//! - `pub fn recomputeType(self: *Circuit, t: DeviceType) error{TopologyChanged}!void`
//! - `pub fn collectNoiseSources(self: *const Circuit, x: []const f64, gpa: Allocator) ![]NoiseSource`

const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;
const abi = @import("device").abi;
const Circuit = @import("Circuit.zig").Circuit;
const Display = @import("Circuit.zig").Display;
const par_eval = @import("par_eval.zig");

/// Tells the circuit that the next `applyLimits` or `stateCtl(.query)` is
/// followed by `eval` (or, with `charge`, `evalQ`) at `x` and `t`, with
/// nothing in between that changes device state. `x` is read when that call
/// runs, so for `applyLimits` it is the limited iterate. The GPU path then
/// runs both on one host wait; the host path ignores it. A wrong promise
/// costs one wasted device eval, never a wrong stamp: the eval checks its x
/// and its whole `SimState`. Without `charge` the eval is the next Newton
/// iterate, one `advanceIteration` on.
pub fn evalFollows(self: *const Circuit, x: []const f64, t: f64, charge: bool) void {
    var sim = self.sim;
    sim.t = t;
    if (!charge) sim.iteration +|= 1;
    if (self.gpuContext()) |g| g.evalFollows(x, sim, charge);
}

/// `evalFollows` for the first iterate of the next solve, which `beginSolve`
/// numbers 1: the predictor's and a sweep's warm start.
pub fn evalFollowsFirst(self: *const Circuit, x: []const f64, t: f64) void {
    var sim = self.sim;
    sim.t = t;
    sim.iteration = 1;
    if (self.gpuContext()) |g| g.evalFollows(x, sim, false);
}

/// Limits the Newton update in place (pnjlim/fetlim). Returns true when any
/// device clamped `x`. Asserts that both iterates hold `n` values.
pub fn applyLimits(self: *const Circuit, x: []f64, x_old: []const f64) bool {
    assert(x.len >= self.n);
    assert(x_old.len >= self.n);
    if (self.gpuContext()) |g| return g.applyLimits(x, x_old, self.sim);
    if (self.par_eval) |p| return p.limit(self.batches, x, x_old);
    return par_eval.limitBatches(self.batches, x, x_old);
}

/// Sets every `idt` state unknown with an ic to that ic (`Hooks.seed_ic`):
/// the uic start's stand-in for the operating point, which solves them.
/// Needs a static sim state.
pub fn seedIc(self: *const Circuit, x: []f64) void {
    for (self.batches) |b| if (b.hooks.seed_ic) |f| f(b.ctx, x);
}

/// Writes the devices' §3.6.3.2 net initializers into `x`
/// (`Hooks.seed_nodeset`). Cold starts only.
pub fn seedNodesets(self: *const Circuit, x: []f64) void {
    for (self.batches) |b| if (b.hooks.seed_nodeset) |f| f(b.ctx, x);
}

/// SPICE MODEINITJCT: devices write junction seed voltages into a zeroed
/// `x`, so iteration 1 linearizes at vcrit/vto instead of 0 and the
/// limiters clamp against the seed. Cold starts only.
pub fn seedJunctions(self: *const Circuit, x: []f64) void {
    if (self.gpuContext()) |g| return g.seedJunctions(x);
    par_eval.seedBatches(self.batches, x);
}

/// Resets device-private limiting state after a Newton solve, so later
/// evals (waveform, AC, noise) see the node vector itself.
pub fn clearLimits(self: *const Circuit) void {
    if (self.gpuContext()) |g| return g.clearLimits();
    par_eval.clearLimitBatches(self.batches);
}

/// Starts a nonlinear solve: `$simparam("iteration")` reads 1.
pub fn beginSolve(self: *Circuit) void {
    self.lin.valid = false;
    if (self.load_check) |*lc| lc.valid = false;
    self.sim.iteration = 1;
    publishSim(self);
}

/// Hands every device the previous Newton iterate, then counts the iteration.
pub fn advanceIteration(self: *Circuit, previous_x: []const f64) void {
    self.lin.valid = false;
    for (self.batches) |b| if (b.hooks.advance_iteration) |f| f(b.ctx, previous_x);
    self.sim.iteration +|= 1;
    publishSim(self);
}

/// Lets each device that predicts its first transient iterate over the step
/// two back (`Hooks.predict_first_iterate`) overwrite its internal unknowns
/// in `trial` with `cur + xfact2 (cur - prev)`.
pub fn predictFirstIterate(self: *const Circuit, trial: []f64, cur: []const f64, prev: []const f64, xfact2: f64) void {
    for (self.batches) |b| if (b.hooks.predict_first_iterate) |f| f(b.ctx, trial, cur, prev, xfact2);
}

/// Returns false when any device vetoes convergence at `x`.
pub fn checkConvergence(self: *const Circuit, x: []const f64) bool {
    for (self.batches) |b| if (b.hooks.check_convergence) |f| {
        if (!f(b.ctx, x)) return false;
    };
    return true;
}

/// Advances device state (operator history, held variables, latches) from
/// the last accepted point to `x`. Returns the earliest time a device asks
/// the step to be rejected at, or null. A second call before the next
/// commit or revert (a flip that forced another iterate, a continuation
/// rung) reverts first, since `stateCtl(.revert)` is exact only across one
/// `updateState`. Under `land_rejects` the reject time goes to `reject_at`
/// and this returns null. Stages the attached `Display` lines for the next
/// `stateCtl`.
pub fn updateStates(self: *Circuit, x: []const f64) ?f64 {
    if (self.state_staged) _ = stateCtl(self, .revert); // The query result is for `.query` only.
    self.state_staged = true;
    if (self.display) |d| say(self, d, x);
    // The transient reads q at the converged point next, and it must be the
    // charge before `updateState` latches the `$prev` values `q` reads (the
    // Meyer caps average against the last accepted point).
    const accept = self.land_rejects and self.has_charge;
    const tr = if (accept) acceptStates(self, x) else if (self.gpuContext()) |g| g.updateStates(x, self.sim) else hostUpdate(self, x);
    assert(self.state_staged);
    if (!self.land_rejects) return tr;
    self.reject_at = tr;
    return null;
}

/// Accepted-point half of `updateStates`, for delay-line history no static
/// solve may push (`Hooks.commit_state`). The transient calls it once per
/// accepted point, before `stateCtl(.commit)`.
pub fn commitStates(self: *const Circuit, x: []const f64) ?f64 {
    return par_eval.minReject(self.batches, "commit_state", x);
}

/// Commits, reverts or queries the accepted device state. `.commit` at every
/// accepted point, the operating point included; `.revert` after every
/// rejected attempt. For `.query`, returns true when a cross/above flip
/// moved the working state off the last accepted one.
pub fn stateCtl(self: *Circuit, sop: abi.StateCtlOp) bool {
    if (sop != .query) self.state_staged = false;
    if (self.display) |d| if (sop != .query) {
        // ponytail: a failed write to `out` drops the point's lines.
        if (sop == .commit) d.out.writeAll(d.staged.written()) catch {};
        d.staged.clearRetainingCapacity();
    };
    if (self.gpuContext()) |g| return g.stateCtl(sop, self.sim);
    return par_eval.stateCtlBatches(self.batches, sop);
}

/// Returns the smallest model delay (`D.delays`) in the circuit, or null.
pub fn minDelay(self: *const Circuit) ?f64 {
    var min_td = std.math.inf(f64);
    for (self.batches) |b| if (b.hooks.min_delay) |f| {
        min_td = @min(min_td, f(b.ctx));
    };
    return if (min_td == std.math.inf(f64)) null else min_td;
}

/// Returns the tightest `$bound_step` (VAMS 9.17.2) any device asked for, or
/// null. Only meaningful after an accepted step ran `updateStates`.
pub fn boundStep(self: *const Circuit) ?f64 {
    var best = std.math.inf(f64);
    for (self.batches) |b| if (b.hooks.bound_step) |f| {
        best = @min(best, f(b.ctx));
    };
    return if (best == std.math.inf(f64)) null else best;
}

/// Returns the earliest device breakpoint or VPI forced time (LRM 12.31.3
/// acbAbsTime/acbElapsedTime) after `t`, or null.
pub fn nextBreakpoint(self: *const Circuit, t: f64) ?f64 {
    var best = std.math.inf(f64);
    for (self.batches) |b| if (b.hooks.next_breakpoint) |f| {
        if (f(b.ctx, t)) |bp| best = @min(best, bp);
    };
    if (self.vpi) |v| if (v.next_forced(v.ctx, t)) |f| {
        best = @min(best, f);
    };
    return if (best == std.math.inf(f64)) null else best;
}

/// Installs a circuit temperature in degrees Celsius. Call `recompute`
/// before solving, which also detects a temperature-driven topology change.
pub fn setCircuitTemp(self: *Circuit, temp_c: f32) void {
    self.temp_c = temp_c;
    self.lin.valid = false;
    for (self.batches) |b| if (b.hooks.set_temp) |f| f(b.ctx, temp_c);
    markGpuDirty(self);
}

/// Publishes the stepping rungs' gmin (S) and source factor to the models
/// that read §9.15 `$simparam("gmin")` or `("sourceScaleFactor")`. The
/// caller restores the deck's gmin and 1 when stepping ends. A circuit
/// without such a model is untouched; otherwise the constant-Jacobian
/// baseline is dropped, so call `computeBaseline` before the next solve.
pub fn setHomotopy(self: *Circuit, gmin: f64, source_scale: f64) void {
    var any = false;
    for (self.batches) |b| if (b.hooks.set_homotopy) |f| {
        f(b.ctx, gmin, source_scale);
        any = true;
    };
    if (any) paramsMoved(self);
}

/// Device parameters moved: drops the `linearize` memo and the
/// constant-Jacobian baseline, and marks the resident GPU parameters and
/// every bypass recording stale.
pub fn paramsMoved(self: *Circuit) void {
    self.lin.valid = false;
    self.has_baseline = false;
    markGpuDirty(self);
}

/// Turns the device latency bypass on with `tol`, or off with null, on every
/// batch that supports it; turning it on again drops every recording.
pub fn setBypass(self: *Circuit, tol: ?abi.BypassTol) void {
    self.bypass_tol = tol;
    for (self.batches) |b| if (b.hooks.set_bypass) |f| f(b.ctx, tol);
}

/// Publishes the analysis state (`$abstime`, timestep, `analysis()`,
/// `initial_step`/`final_step`) every later device call receives; `eval`
/// takes `$abstime` from its own `t`. Keeps the Newton iteration count.
/// Must run before the eval it describes.
pub fn setSimState(self: *Circuit, st: abi.SimState) void {
    const iteration = self.sim.iteration;
    self.sim = st;
    self.sim.iteration = iteration;
    publishSim(self);
}

/// Re-derives every batch's numeric parameters. Drops the constant-Jacobian
/// baseline (`computeBaseline` rebuilds it) and marks the GPU context's
/// resident parameters stale. `error.TopologyChanged` means internal wiring
/// moved and the circuit must be rebuilt; restore the parameters and
/// recompute before reusing it.
pub fn recompute(self: *Circuit) error{TopologyChanged}!void {
    paramsMoved(self);
    for (self.batches) |b| if (b.hooks.recompute) |f| {
        if (!f(b.ctx)) return error.TopologyChanged;
    };
}

/// `recompute` restricted to batches of type `t` (`ParamRef.type`).
///
/// A `.dc` point writes one parameter, and a full recompute re-runs every
/// compact model's temperature preamble (1,811 BJT preambles per sweep on a
/// one-instance deck). Sound only while nothing global has moved: a
/// temperature change can re-wire a device of another type (a BJT whose
/// RB(T) reaches zero). The caller owns that distinction; `.dc` takes the
/// full walk on the first point of every inner sweep and narrows after.
pub fn recomputeType(self: *Circuit, t: abi.DeviceType) error{TopologyChanged}!void {
    paramsMoved(self);
    for (self.batches, self.batch_types) |b, bt| if (bt == t) if (b.hooks.recompute) |f| {
        if (!f(b.ctx)) return error.TopologyChanged;
    };
}

/// Returns every device's noise generators at `x`, with each device's own
/// PSDs (which already carry its temperature). Pure in `x`, so `.pnoise`
/// calls it per PSS sample. Caller owns the slice and frees it with `gpa`.
pub fn collectNoiseSources(self: *const Circuit, x: []const f64, gpa: Allocator) ![]abi.NoiseSource {
    var list: std.ArrayList(abi.NoiseSource) = .empty;
    errdefer list.deinit(gpa);

    for (self.batches) |b| if (b.hooks.collect_noise) |f| try f(b.ctx, x, gpa, &list).unwrap();
    return try list.toOwnedSlice(gpa);
}

// ---- private ----

/// `updateBatches` on the host, over ParEval's lanes when it exists.
fn hostUpdate(self: *const Circuit, x: []const f64) ?f64 {
    if (self.par_eval) |p| return p.update(self.batches, x);
    return par_eval.updateBatches(self.batches, x);
}

/// Stages every `Hooks.say` line at x, before `updateState` moves the
/// instances on from the state eval saw.
fn say(self: *const Circuit, d: *Display, x: []const f64) void {
    assert(d.names.len == self.batches.len);
    for (self.batches, d.names) |b, names| {
        const f = b.hooks.say orelse continue;
        assert(names.len == b.count);
        for (names, 0..) |name, i| f(b.ctx, x, @intCast(i), name, &d.staged.writer);
    }
}

/// `evalQ` at x and the published time, then `updateBatches`. Serially
/// through `accept_q` where the batch has it, so its model core runs once
/// for both.
// ponytail: the GPU and ParEval paths run two passes; a ParEval `.accept`
// mode would fuse them when lanes are on.
fn acceptStates(self: *Circuit, x: []const f64) ?f64 {
    // The device runs the charges, the state halves and the query on one
    // wait (`GpuContext.acceptStates`).
    if (self.gpuContext()) |g| if (g.acceptStates(x, self.sim.t, self.sim)) |tr| {
        self.lin.valid = false;
        self.bindPlanes(g.planes());
        self.q_fresh = true;
        return tr;
    };
    if (self.gpuContext() != null or self.par_eval != null) {
        self.evalQ(x, self.sim.t);
        const tr = if (self.gpuContext()) |g| g.updateStates(x, self.sim) else hostUpdate(self, x);
        self.q_fresh = true;
        return tr;
    }
    self.lin.valid = false;
    self.clearPlanes(.charge);
    const pl = self.ownPlanes();
    var min_reject: ?f64 = null;
    for (self.batches) |b| {
        if (b.hooks.accept_q) |f| {
            f(b.ctx, &pl, 0, b.count, x);
            continue;
        }
        if (b.hooks.eval_q) |f| f(b.ctx, &pl, 0, b.count, x, self.sim.t);
        if (b.hooks.update_state) |f| if (f(b.ctx, x)) |tr| {
            min_reject = if (min_reject) |cur| @min(cur, tr) else tr;
        };
    }
    self.q_fresh = true;
    return min_reject;
}

/// Copies `sim` into every batch: one store each.
fn publishSim(self: *const Circuit) void {
    for (self.batches) |b| b.hooks.set_sim_state(b.ctx, self.sim);
}

fn markGpuDirty(self: *Circuit) void {
    if (self.gpuContext()) |g| g.markDirty();
    // Device parameters moved: every bypass recording is stale.
    if (self.bypass_tol) |tol| setBypass(self, tol);
}
