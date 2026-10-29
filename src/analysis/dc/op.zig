//! Operating point (`.op`): the static solve every other analysis starts
//! from. Sets the first static step, holds the `.ic` rows under TRANOP,
//! forces the `.nodeset` rows once, then climbs the continuation ladder
//! (`op_ladder.zig`); `op_lanes.zig` runs the same solve W lanes wide.
//! In: a frozen circuit, `.nodeset` and `.ic` rows. Out: the operating point
//! in `x` and the committed device state.
//! Cites ngspice cktop.c, cktload.c (MODETRANOP, MODEINITJCT) and cktic.c;
//! Verilog-AMS LRM §4.6.1 and §5.10.2; dev/analysis/operating-point-homotopy.md.
//!
//! Contents:
//! - `pub const Options = core.query.Op`
//! - `pub const LaneOutcome = ErrorOf(solve)!converger.Result`
//! - `pub const gminFactor = op_ladder.gminFactor`
//! - `pub fn run(ctx: *const RunCtx, _: Options) !Result`
//! - `pub fn solve(ckt: *Circuit, x: []f64, options: Options, nodeset: []const Ic, ic: []const Ic) !converger.Result`
//! - `pub fn beginSolve(ckt: *Circuit, x: []f64, options: Options) !void`
//! - `pub fn coldStart(ckt: *Circuit, x: []f64) void`
//! - `pub fn nodesetLadder(ckt, ws, x, options, force, n_hold) !converger.Result`
//! - `pub fn endSolve(ckt: *Circuit, converged: bool, options: Options) void`
//! - `pub fn lanesAllowed(c: *const Circuit) bool`

const std = @import("std");
const assert = std.debug.assert;
const converger = @import("solver").converger;
const Ic = @import("core").Ic;
const root = @import("../types.zig");
const op_ladder = @import("op_ladder.zig");

/// Query options, defined in core/query.zig.
pub const Options = @import("core").query.Op;

/// One lane's `solve` outcome.
pub const LaneOutcome = root.ErrorOf(solve)!converger.Result;

/// The gmin stepping rung's next factor (`op_ladder.gminFactor`).
pub const gminFactor = op_ladder.gminFactor;

/// Contract entry: the executor's solved `ctx.x_op`, one point per probe.
pub fn run(ctx: *const root.RunCtx, _: Options) !root.Result {
    const x = ctx.x_op;
    // LRM 12.31.3: the one solution is the first and the last. It has no
    // earlier time to back up to, so acbConvergenceTest is not asked.
    if (ctx.circuit.vpi) |v| {
        v.point(v.ctx, .{ .kind = .op });
        v.accept(v.ctx, true, true);
    }
    const names = try root.probeNames(ctx, null);
    const data = try ctx.allocator.alloc(f64, names.len);
    assert(data.len == ctx.probes.len);
    for (ctx.probes, data) |node, *out| out.* = x[node];
    return .{
        .plotname = "Operating Point",
        .varnames = names,
        .is_complex = false,
        .npoints = 1,
        .data = data,
    };
}

/// Solves the operating point into caller-owned `x` (cold-started unless
/// `options.warm_start`), after a first solve that holds the `nodeset` rows
/// at their guesses. When `options.tran_op`, every rung also holds the `ic`
/// rows at their values, ngspice's MODETRANOP without MODEUIC (cktload.c);
/// a standalone operating point ignores `ic`. On convergence the devices
/// commit their state, and the circuit is left in the static simulation
/// state later evals at this point expect. `iterations` sums every rung
/// tried, saturating at the u16 maximum. Fails with error.FloatingNode when
/// a node has no DC path to ground outside a TRANOP; non-convergence is
/// `converged = false`, not an error. Asserts that `x.len` is `ckt.n`.
pub fn solve(ckt: *root.Circuit, x: []f64, options: Options, nodeset: []const Ic, ic: []const Ic) !converger.Result {
    assert(x.len == ckt.n);
    try beginSolve(ckt, x, options);
    const ws = try ckt.workspace();
    const gpa = ws.slv.gpa;
    const held = if (options.tran_op) ic else &.{};
    // Holds first, then the nodesets on rows no `.ic` holds: the `.ic` value
    // wins where both name a node.
    var force: std.ArrayList(converger.Force) = .empty;
    defer force.deinit(gpa);

    try force.ensureTotalCapacityPrecise(gpa, held.len + nodeset.len);
    for (held) |h| force.appendAssumeCapacity(.{ .slot = ckt.diag_slots[h.node], .row = h.node, .value = h.value });
    for (nodeset) |ns| {
        for (held) |h| {
            if (h.node == ns.node) break;
        } else force.appendAssumeCapacity(.{ .slot = ckt.diag_slots[ns.node], .row = ns.node, .value = ns.value });
    }
    assert(force.items.len >= held.len);
    const r = try nodesetLadder(ckt, ws, x, options, force.items, held.len);
    endSolve(ckt, r.converged, options);
    return r;
}

/// `solve`'s setup: the static simulation state of the first step, the cold
/// start unless `options.warm_start`, then the const-Jacobian baseline.
/// LRM §4.6.1: the operating point is a static solve, so `analysis("dc")`
/// holds and `$abstime` = 0; §5.10.2 `initial_step` is this solve, so a
/// switch latches its power-on state from `ic` here. The baseline evaluates
/// const-Jacobian batches, so the state goes in first.
pub fn beginSolve(ckt: *root.Circuit, x: []f64, options: Options) !void {
    assert(x.len == ckt.n);
    ckt.setSimState(.{ .kind = if (options.tran_op) .ic else .dc, .initial_step = true });
    if (!options.warm_start) coldStart(ckt, x);
    try ckt.computeBaseline();
}

/// Zeroes `x`, writes the devices' §3.6.3.2 net initializers (Newton's
/// guess, never held), then applies the SPICE MODEINITJCT junction seeds so
/// the first iteration linearizes at vcrit/vto instead of 0; a limiter
/// without a seed starts from the initializer. Asserts `x.len == ckt.n`.
pub fn coldStart(ckt: *root.Circuit, x: []f64) void {
    assert(x.len == ckt.n);
    @memset(x, 0);
    ckt.seedNodesets(x);
    ckt.seedJunctions(x);
}

/// The `.nodeset` step, then the ladder holding the first `n_hold` rows of
/// `force` (the `.ic` holds; the rest are nodesets). With nodesets given,
/// `x` first takes every `force` value on its row. Every cold CKTop in
/// ngspice starts in MODEINITJCT, where cktload.c applies the nodesets, so
/// the operating point and each cold DC-sweep point both start here.
/// Asserts that the holds lie in `force`.
pub fn nodesetLadder(ckt: *root.Circuit, ws: *converger.Workspace, x: []f64, options: Options, force: []const converger.Force, n_hold: usize) !converger.Result {
    assert(n_hold <= force.len);
    assert(x.len == ckt.n);
    // A converged forcing step leaves a warm start, as ngspice's INITFIX
    // hands INITFLOAT its iterate.
    var ladder = options;
    if (force.len > n_hold) {
        // CKTic starts rhsOld at the nodeset and ic values, so the devices
        // first linearize there: a latch whose x = 0 is an exact
        // (metastable) solution only leaves it this way.
        for (force) |f| x[f.row] = f.value;
        if (try forceNodeset(ckt, ws, x, options, force)) ladder.warm_start = true;
    }
    return op_ladder.solve(ckt, ws, x, ladder, force[0..n_hold]);
}

/// `solve`'s close: commits the device state of a converged point so later
/// analyses start from it, and leaves the initial step, since every later
/// eval at this point (ac, tf, noise, post-processing) must not re-latch.
pub fn endSolve(ckt: *root.Circuit, converged: bool, options: Options) void {
    if (converged) _ = ckt.stateCtl(.commit); // The flip query is for transients.
    ckt.setSimState(.{ .kind = if (options.tran_op) .ic else .dc });
    assert(!ckt.sim.initial_step);
}

/// Returns whether `c`'s Newton may run W lanes wide: a JFNK pin and
/// `-Dcertify` both need each lane's own scalar `converger.run`. Shared by
/// every lane-batched analysis (`.op`, `.dc`, the sweep lanes).
pub fn lanesAllowed(c: *const root.Circuit) bool {
    const pin = c.knobs.newton.pin;
    return (pin == .auto or pin == .direct) and !root.Circuit.certify;
}

// ---- private ----

/// The `.nodeset` step before the ladder, ngspice's MODEINITJCT and
/// MODEINITFIX iterations with the nodes held (cktload.c): Newton from `x`
/// with every `force` row tied to its value by `converger.force_g`. True
/// when it converged, leaving that point in `x`; false restores `x`.
fn forceNodeset(ckt: *root.Circuit, ws: *converger.Workspace, x: []f64, options: Options, force: []const converger.Force) !bool {
    const gpa = ws.slv.gpa;
    const start = try gpa.dupe(f64, x);
    defer gpa.free(start);

    var copts = converger.optionsFromTolerances(options.tol, null);
    copts.init_fix = true;
    copts.force = force;
    const r = converger.run(ckt, ws, x, 0, copts, root.EvalHook{}) catch |e| switch (e) {
        error.SingularMatrix => op_ladder.failed,
        else => return e, // Only a singular Jacobian fails the forcing step.
    };
    if (ws.knobs.opdbg) std.debug.print("ladder: nodeset conv={} it={d}\n", .{ r.converged, r.iterations });
    if (!r.converged) @memcpy(x, start);
    return r.converged;
}
