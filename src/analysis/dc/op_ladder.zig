//! The operating point's continuation ladder over one Workspace: plain
//! Newton, dynamic gmin stepping, source stepping, JFNK, then OPtran. The
//! pattern is frozen, so the ordering and symbolic factorization happen once
//! for the whole ladder. Every rung's Newton holds the `hold` rows.
//! In: a circuit, a start `x`, the holds. Out: `x` and the Newton result.
//! Cites ngspice cktop.c (dynamic_gmin at :207-222, gillespie_src, the
//! optran gate at :94-97), optran.c, cktload.c; Verilog-AMS LRM §9.15; #53,
//! #90; dev/analysis/operating-point-homotopy.md.
//!
//! Contents:
//! - `pub const failed: converger.Result`
//! - `pub fn solve(ckt: *Circuit, ws: *Workspace, x: []f64, options: Options, hold: []const Force) !converger.Result`
//! - `pub fn afterPlain(ckt, ws, x, options, hold, plain: ?converger.Result) !converger.Result`
//! - `pub fn gminFactor(factor: f64, iterations: u16, budget: u16) f64`

const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;
const converger = @import("solver").converger;
const root = @import("../types.zig");
const tran = @import("../tran/tran.zig");
const op = @import("op.zig");
const gmin_stamps = @import("gmin_stamps.zig");
const Options = op.Options;
const Force = converger.Force;

/// A rung that did not converge and counted no iterations.
pub const failed: converger.Result = .{ .converged = false, .iterations = 0 };

/// Solves at most per stepping rung. ngspice has no cap (each rung ends on
/// its step floor); 100 ended chain_bsim4_100k's gmin stepping at 1.35e-6
/// while every step still converged.
const ladder_solves_max = 1000;

/// The ladder from `x` as given. A circuit with a capacitor-only island
/// (`needs_tran_op`) and no holds skips the static rungs and goes straight
/// to OPtran. Fails with error.FloatingNode when a node has no DC path to
/// ground outside a TRANOP.
pub fn solve(ckt: *root.Circuit, ws: *converger.Workspace, x: []f64, options: Options, hold: []const Force) !converger.Result {
    assert(x.len == ckt.n);
    // Holds can pin the floating nodes, as they do in ngspice, so a held
    // solve tries the static rungs before settling.
    if (ckt.needs_tran_op and hold.len == 0) {
        // A node with no DC path has an all-zero G row, so the static
        // operating point is not unique and the transient only reports
        // wherever settling from zero landed. ngspice accepts that under
        // TRANOP, where the transient owns the initial condition; a
        // standalone .op/.ac/.pz has no such owner, so the deck is rejected.
        if (!options.tran_op) {
            std.log.err("topology: a node has no DC path to ground (capacitor-only island) — the operating point is not unique", .{});
            return error.FloatingNode;
        }
        return transientOp(ckt, ws, x, options, hold);
    }
    // Rung 1: plain Newton with no diagonal gmin. ngspice's NIiter loads one
    // only during gmin stepping (junction gmin lives in the device models);
    // an always-on shunt moves every solution off ngspice's (voltage_divider
    // by 2.5e-9) and settles a floating bridge on a common mode ngspice
    // never picks.
    const plain = newtonRun(ckt, ws, x, options.tol, hold, 0.0, &.{}, null, !options.warm_start) catch |e| switch (e) {
        error.SingularMatrix => null,
        else => return e, // Only a singular Jacobian moves on to the next rung.
    };
    return afterPlain(ckt, ws, x, options, hold, plain);
}

/// Rungs 2 to 5, given the plain rung's outcome (null: a singular
/// Jacobian): returns it when it converged. `iterations` sums every rung.
pub fn afterPlain(ckt: *root.Circuit, ws: *converger.Workspace, x: []f64, options: Options, hold: []const Force, plain: ?converger.Result) !converger.Result {
    assert(x.len == ckt.n);
    if (ws.knobs.opdbg) std.debug.print("ladder: plain conv={?}\n", .{if (plain) |p| p.converged else null});
    if (plain) |p| if (p.converged) return p;

    // Last converged solution, the restart point of both stepping rungs.
    const gpa = ws.slv.gpa;
    const x_good = try gpa.alloc(f64, ckt.n);
    defer gpa.free(x_good);

    var total: u16 = if (plain) |p| p.iterations else 0;
    if (try gminStepping(ckt, ws, x, options, hold, x_good, &total)) return .{ .converged = true, .iterations = total };
    try sourceStepping(ckt, ws, x, options, hold, x_good, &total);

    const final = newtonRun(ckt, ws, x, options.tol, hold, 0.0, &.{}, null, false) catch |e| switch (e) {
        error.SingularMatrix => failed,
        else => return e, // Only a singular Jacobian moves on to JFNK.
    };
    total +|= final.iterations;
    if (final.converged) return .{ .converged = true, .iterations = total };

    // Rung 4: JFNK. Damping plus residual backtracking catches circuits
    // where the factored Newton step wedges. It has no holds, so a held
    // solve skips it.
    if (hold.len == 0) {
        op.coldStart(ckt, x);
        // converger.run clears device limiting state on exit; a direct jfnk
        // call must too, so post-solve evals see clean state.
        defer ckt.clearLimits();
        const jr = try converger.jfnk(ckt, ws, x, 0, converger.optionsFromTolerances(options.tol, null), root.EvalHook{});
        total +|= jr.iterations;
        if (jr.converged) return .{ .converged = true, .iterations = total };
    }
    var result = try transientOp(ckt, ws, x, options, hold);
    result.iterations +|= total;
    return result;
}

/// Returns the next gmin step factor after a step that converged in
/// `iterations` of a Newton `budget` (cktop.c:207-222): an easy step (a
/// quarter of the budget or less) accelerates, capped at 10; a hard one
/// (over three quarters) slows down before it fails, floored at 1.00005.
/// ngspice compares with itl2 (50) while NIiter raises the budget to 100,
/// so steps of 13-37 iterations never moved the factor; the thresholds here
/// are the budget's.
pub fn gminFactor(factor: f64, iterations: u16, budget: u16) f64 {
    assert(factor >= 1);
    if (iterations <= budget / 4) return @min(factor * @sqrt(factor), 10.0);
    if (iterations > 3 * (budget / 4)) return @max(@sqrt(factor), 1.00005);
    return factor;
}

// ---- private ----

/// Rung 2, dynamic gmin stepping (cktop.c dynamic_gmin): descend gmin by
/// `factor` (`gminFactor`); on a failed step, back up toward the last good
/// gmin with the 4th root of the factor (the square root when the LU
/// refused the matrix) and retry from the last converged x; give up once the
/// factor is ~1. True when the last, shunt-free solve converged. Leaves the
/// deck's gmin published either way.
fn gminStepping(ckt: *root.Circuit, ws: *converger.Workspace, x: []f64, options: Options, hold: []const Force, x_good: []f64, total: *u16) !bool {
    const gpa = ws.slv.gpa;
    // The planes still hold the plain rung's last assembly, which is where
    // ngspice's preorder reads its twins.
    const stamps = try gmin_stamps.build(ckt, gpa);
    defer gpa.free(stamps);

    op.coldStart(ckt, x);
    const gtarget = options.tol.gmin;
    // §9.15 `$simparam("gmin")` reads the rung's gmin, the deck's after.
    errdefer ckt.setHomotopy(gtarget, 1.0);
    var factor: f64 = 10.0;
    var good_gmin = options.tol.gmin_start; // The bound to back up toward.
    var gmin_val = good_gmin / factor;
    var have_good = false;
    // The Newton budget each step gets: NIiter's floor of 100 over itl2.
    const budget = converger.optionsFromTolerances(options.tol, options.tol.itl2).max_iter;
    var solves: u32 = 0;
    while (solves < ladder_solves_max) : (solves += 1) {
        ckt.setHomotopy(gmin_val, 1.0);
        try ckt.computeBaseline();
        var refused = false;
        const r = newtonRun(ckt, ws, x, options.tol, hold, gmin_val, stamps, options.tol.itl2, !have_good) catch |e| switch (e) {
            error.SingularMatrix => blk: {
                refused = true;
                break :blk failed;
            },
            else => return e, // Only a singular Jacobian is a failed step.
        };
        total.* +|= r.iterations;
        if (ws.knobs.opdbg) std.debug.print("ladder: gmin={e:.3} conv={} it={d}\n", .{ gmin_val, r.converged, r.iterations });
        if (!r.converged) {
            if (factor < 1.00005) break; // Wedged against the last good step.
            // A matrix the LU refused (a singular or fill-capped factor at
            // it = 0) says less about the step than a Newton that ran out:
            // back up by the square root only.
            factor = if (refused) @sqrt(factor) else @sqrt(@sqrt(factor));
            gmin_val = good_gmin / factor;
            if (have_good) @memcpy(x, x_good) else op.coldStart(ckt, x);
            continue;
        }
        if (gmin_val <= gtarget) {
            // dynamic_gmin removes diagGmin for the last solve: the answer
            // must not carry the shunt.
            ckt.setHomotopy(gtarget, 1.0);
            try ckt.computeBaseline();
            const clean = newtonRun(ckt, ws, x, options.tol, hold, 0.0, &.{}, null, false) catch |err| switch (err) {
                error.QueryCancelled, error.OutOfMemory => return err,
                else => failed, // A numerical failure ends the ladder.
            };
            total.* +|= clean.iterations;
            if (clean.converged) return true;
            break;
        }
        @memcpy(x_good, x);
        have_good = true;
        good_gmin = gmin_val;
        factor = gminFactor(factor, r.iterations, budget);
        if (gmin_val < factor * gtarget) {
            factor = gmin_val / gtarget;
            gmin_val = gtarget;
        } else gmin_val /= factor;
    }
    ckt.setHomotopy(gtarget, 1.0); // Rung 3 rebuilds the baseline.
    return false;
}

/// Rung 3, source stepping through §9.15 `$simparam("sourceScaleFactor")`
/// with an adaptive step: grow 1.5x on success, halve on failure and retry
/// from the last good lambda and x. The holds scale with the sources, as
/// cktload.c multiplies each `.ic` by CKTsrcFact. Leaves full sources and a
/// rebuilt baseline whatever the outcome.
fn sourceStepping(ckt: *root.Circuit, ws: *converger.Workspace, x: []f64, options: Options, hold: []const Force, x_good: []f64, total: *u16) !void {
    const gpa = ws.slv.gpa;
    op.coldStart(ckt, x);
    const scaled = try gpa.dupe(Force, hold);
    defer gpa.free(scaled);

    var lambda: f64 = 0.0;
    var lambda_good: f64 = -1.0; // None converged yet.
    var delta: f64 = 0.25;
    var solves: u32 = 0;
    while (solves < ladder_solves_max) : (solves += 1) {
        ckt.setHomotopy(options.tol.gmin, lambda);
        try ckt.rebuildBaseline();
        for (scaled, hold) |*s, h| s.value = h.value * lambda;
        var sr = newtonRun(ckt, ws, x, options.tol, scaled, 0.0, &.{}, options.tol.itl2, lambda_good < 0.0) catch |e| switch (e) {
            error.SingularMatrix => failed,
            else => { // Only a singular Jacobian is a failed step; the rest unwinds the sources.
                try restoreSources(ckt, options);
                return e;
            },
        };
        total.* +|= sr.iterations;
        if (ws.knobs.opdbg) std.debug.print("ladder: src lambda={e:.3} conv={} it={d}\n", .{ lambda, sr.converged, sr.iterations });
        if (!sr.converged and lambda_good < 0.0) {
            sr.converged = sourcesOffGmin(ckt, ws, x, options, scaled, total) catch |e| {
                try restoreSources(ckt, options);
                return e;
            };
        }
        if (sr.converged) {
            if (lambda >= 1.0) break; // Full sources reached.
            lambda_good = lambda;
            @memcpy(x_good, x);
            delta *= 1.5;
            lambda = @min(lambda + delta, 1.0);
        } else {
            delta *= 0.5;
            if (delta < 1e-4) break;
            // Nothing converged yet means the lambda = 0 start failed, and a
            // retry would be the same cold solve.
            if (lambda_good < 0.0) break;
            @memcpy(x, x_good);
            lambda = @min(lambda_good + delta, 1.0);
        }
    }
    try restoreSources(ckt, options);
}

/// Puts the full sources and the deck's gmin back, with their baseline.
fn restoreSources(ckt: *root.Circuit, options: Options) !void {
    ckt.setHomotopy(options.tol.gmin, 1.0);
    try ckt.rebuildBaseline();
}

/// gillespie_src's second try at the sources-off start: when the plain solve
/// at lambda = 0 fails, solve it again under a diagonal gmin from 1e10 times
/// the deck's gmin down to the deck's, a decade per solve, each from the
/// last. True when all eleven converge; `x` then holds the last, the start
/// source stepping raises from. Without it a circuit whose supplies-off
/// state is floating (every MOSFET off, as on chain_bsim4_100k) failed its
/// whole source-stepping rung at lambda = 0.
fn sourcesOffGmin(ckt: *root.Circuit, ws: *converger.Workspace, x: []f64, options: Options, hold: []const Force, total: *u16) !bool {
    if (!(options.tol.gmin > 0)) return false; // No gmin to step down to.
    const gpa = ws.slv.gpa;
    // The planes hold the failed lambda = 0 assembly: the twins' values.
    const stamps = try gmin_stamps.build(ckt, gpa);
    defer gpa.free(stamps);

    op.coldStart(ckt, x);
    var g = options.tol.gmin * 1e10;
    for (0..11) |i| {
        const r = newtonRun(ckt, ws, x, options.tol, hold, g, stamps, options.tol.itl2, i == 0) catch |e| switch (e) {
            error.SingularMatrix => failed,
            else => return e, // Only a singular Jacobian is a failed step.
        };
        total.* +|= r.iterations;
        if (ws.knobs.opdbg) std.debug.print("ladder: src-off gmin={e:.3} conv={} it={d}\n", .{ g, r.converged, r.iterations });
        if (!r.converged) return false;
        g /= 10;
    }
    return true;
}

/// Rung 5, ngspice OPtran (optran.c): a real transient with full sources
/// held at their t = 0 values, steps of at most 10 ns out to 1 us, no ramp
/// and no extra regularization, so the device capacitances do the
/// conditioning the static rungs could not. The settled state only seeds a
/// clean Newton, whose result is the answer; reporting the settled state
/// itself would be a false success (stress/scaling_inverter_chain_4k).
/// Under `needs_tran_op` with no holds, the settled state supplies the
/// capacitor-only nodes, which that Newton holds there (`floatingHolds`).
/// The transient commits device state at every step, so it runs between a
/// save and a restore: only `x` leaves it.
fn transientOp(ckt: *root.Circuit, ws: *converger.Workspace, x: []f64, options: Options, hold: []const Force) !converger.Result {
    const gpa = ws.slv.gpa;
    @memset(x, 0);
    // The held rows start at their values, where the capacitors keep them.
    for (hold) |h| x[h.row] = h.value;
    var wf = try tran.Waveform.init(gpa, 0, 16);
    defer wf.deinit();

    const saved = try ckt.saveState(gpa);
    defer saved.deinit(gpa);

    const op_sim = ckt.sim;
    // Its steps are no analysis's accepted points, so display tasks stay quiet.
    const display = ckt.display;
    ckt.display = null;
    defer ckt.display = display;
    const settled = tran.simulate(ckt, x, &.{}, &wf, .{
        .tol = options.tol,
        .t_stop = 1e-6,
        .dt_init = 1e-8,
        .dt_max = 1e-8,
        .uic = true,
        .sources_at_zero = true,
    }, gpa) catch |err| switch (err) {
        error.QueryCancelled, error.OutOfMemory => return err,
        else => false, // A numerical failure fails this rung.
    };
    // The transient left .tran device state behind; the op contract is a
    // static circuit whatever the outcome.
    ckt.restoreState(saved);
    ckt.setSimState(op_sim);
    try ckt.rebuildBaseline();
    if (!settled) return failed;
    // A held solve keeps its own holds; only an unheld one holds the
    // capacitor-only nodes at their settled values. The settled x belongs to
    // the device state at t = 1 us, not the restored one, so the rest is
    // re-solved at the restored state (#53).
    const floating = if (ckt.needs_tran_op and hold.len == 0) try floatingHolds(ckt, x, gpa) else &.{};
    defer gpa.free(floating);

    const held = if (floating.len > 0) floating else hold;
    return newtonRun(ckt, ws, x, options.tol, held, 0.0, &.{}, null, false) catch |err| switch (err) {
        error.QueryCancelled, error.OutOfMemory => return err,
        else => failed, // A numerical failure fails this rung.
    };
}

/// A hold at its value in `x` on one row of every floating component of the
/// static Jacobian at `x`, so a static Newton is not singular there. A
/// component floats when shifting all its node voltages together leaves
/// every row unchanged (each row's node columns sum to zero): a
/// capacitor-only node, or an island behind a 0 V source or a resistor
/// (#90). Evaluates `ckt` at `x` and clears the limiters. Caller owns the
/// result.
fn floatingHolds(ckt: *root.Circuit, x: []const f64, gpa: Allocator) ![]Force {
    assert(x.len == ckt.n);
    ckt.eval(x, 0);
    ckt.clearLimits();
    const n = ckt.n;
    const parent = try gpa.alloc(u32, n);
    defer gpa.free(parent);

    const sum = try gpa.alloc(f64, 2 * n); // Row sums over node columns, then |G| row sums.
    defer gpa.free(sum);

    const grounded = try gpa.alloc(bool, n);
    defer gpa.free(grounded);

    for (parent, 0..) |*p, i| p.* = @intCast(i);
    @memset(sum, 0);
    @memset(grounded, false);
    for (0..n) |c| for (ckt.col_ptr[c]..ckt.col_ptr[c + 1]) |p| {
        const r = ckt.row_idx[p];
        const g = ckt.g_vals[p];
        if (g == 0) continue;
        parent[findRoot(parent, r)] = findRoot(parent, @intCast(c));
        if (!ckt.current_row[c]) sum[r] += g;
        sum[n + r] += @abs(g);
    };
    for (0..n) |r| {
        if (@abs(sum[r]) > 64 * std.math.floatEps(f64) * sum[n + r]) grounded[findRoot(parent, @intCast(r))] = true;
    }
    assert(grounded[findRoot(parent, root.GROUND)]);
    // One hold per floating root, on its first node row (else its first row).
    var holds: std.ArrayList(Force) = .empty;
    errdefer holds.deinit(gpa);

    for (0..n) |i| {
        const root_i = findRoot(parent, @intCast(i));
        if (grounded[root_i]) continue;
        const k = for (0..n) |j| {
            if (!ckt.current_row[j] and findRoot(parent, @intCast(j)) == root_i) break j;
        } else i;
        try holds.append(gpa, .{ .slot = ckt.diag_slots[k], .row = @intCast(k), .value = x[k] });
        grounded[root_i] = true;
    }
    return holds.toOwnedSlice(gpa);
}

/// Union-find root of `start`, halving the path on the way.
fn findRoot(parent: []u32, start: u32) u32 {
    var i = start;
    var steps: usize = 0;
    while (parent[i] != i) : (i = parent[i]) {
        assert(steps < parent.len);
        steps += 1;
        parent[i] = parent[parent[i]];
    }
    return i;
}

/// One rung's Newton. `max_iter` null means itl1; the stepping rungs pass
/// itl2 (CKTdcTrcvMaxIter, cktop.c:194 and the source-stepping NIiter
/// calls). `init_fix` marks a solve from the cold start, which ngspice runs
/// in MODEINITJCT until a rung's first success switches it to continuemode.
fn newtonRun(ckt: *root.Circuit, ws: *converger.Workspace, x: []f64, tol: converger.Tolerances, hold: []const Force, gmin: f64, stamps: []const converger.GminStamp, max_iter: ?u16, init_fix: bool) !converger.Result {
    var copts = converger.optionsFromTolerances(tol, max_iter);
    copts.gmin = gmin;
    copts.gmin_stamps = stamps;
    copts.init_fix = init_fix;
    copts.force = hold;
    return converger.run(ckt, ws, x, 0, copts, root.EvalHook{});
}
