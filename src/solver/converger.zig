//! Newton convergence loops, generic over the system and its assembly hook:
//! direct Newton (stamp, sparse LU, one solve per step) and JFNK (one
//! `gmres.Gmres` cycle on finite-difference Jacobian products), plus the shared
//! acceptance gates and per-circuit workspace.
//!
//! In: a system, its assembly hook and a start x. Out: x at a converged
//! point (or the last iterate) and a `Result`.
//!
//! `sys` is a pointer with fields `n`, `nnz`, `diag_slots`, `rhs` and
//! `current_row`, an optional `solver_execution`, and optional methods
//! beginSolve, advanceIteration, checkpoint, loadCheck, deviceSolve,
//! evalFollows, checkConvergence, applyLimits, updateStates, clearLimits and
//! nodeName, plus `certify: bool` with certifyPoint(hook, x, t, opts), which
//! `newton` calls on every converged x outside the INITFIX pass, before
//! staging, when `certify` is true. `hook` provides assemble(sys, x, t),
//! vals(sys) (the Jacobian in pattern order), diagAt(sys, slot), and
//! optionally residual(sys, x, t) for a residual-only evaluation.
//!
//! Cites ngspice NIiter (maths/ni/niiter.c), NIconvTest (maths/ni/niconv.c)
//! and CKTload's nodeset hold (cktload.c); dev/solvers/gpu-lu.md (the device
//! factor); #3, #53, #152 (the runaway bound and the branch-row gate).
//!
//! Contents:
//! - `pub const SolverPin = enum`
//!   - `pub const names = std.StaticStringMap(SolverPin).initComptime(.{`
//! - `pub const Knobs = struct`
//! - `pub const Workspace = newton_workspace.Workspace`
//! - `pub const LaneWorkspace = newton_workspace.LaneWorkspace`
//! - `pub const Prof = newton_workspace.Prof`
//! - `pub const jfnk = jfnk_mod.jfnk`
//! - `pub const Tolerances = num.Tolerances`
//! - `pub const Options = struct`
//! - `pub const force_g = 1e10`
//! - `pub const Force = struct`
//! - `pub const GminStamp = struct`
//! - `pub const Result = struct`
//! - `pub const x_plausible = 1e9`
//! - `pub const test_access = if (@import("builtin").is_test) .{ .updateAndNorm = struct {`
//! - `pub fn optionsFromTolerances(tol: Tolerances, max_iter_override: ?u16) Options`
//! - `pub fn newton(...)`
//! - `pub fn WorkspaceOf(comptime W: usize) type`
//! - `pub fn ResultOf(comptime W: usize) type`
//! - `pub fn LaneResult(comptime W: usize) type`
//! - `pub fn newtonLanes(...)`
//! - `pub fn loadExtras(sys: anytype, v: anytype, x: anytype, opts: Options) void`
//! - `pub fn residualGate(opts: Options, diag: anytype, xi: @TypeOf(diag)) @TypeOf(diag)`
//! - `pub fn plausibleLimit(sys: anytype, x: anytype, f: anytype, vals: anytype) @TypeOf(x[0])`
//! - `pub fn run(...)`
//! - `pub fn updateAndNorm(comptime W: usize, x: []Lanes(W).V, dx: []const Lanes(W).V, x_old: []Lanes(W).V, current_row: []const bool, reltol: f64, abstol: f64, vntol: f64, active: Lanes(W).Mask) Lanes(W).V`

const std = @import("std");
const assert = std.debug.assert;
const direct = @import("direct.zig");
const Gmres = @import("gmres.zig").Gmres;
const num = @import("core").numerics;
const BbdInfo = num.BbdInfo;
const Execution = num.Execution;
const lanes = @import("lane/lanes.zig");
const Lanes = lanes.Lanes;
const LaneSolver = @import("lane/solver.zig").LaneSolver;
const newton_sys = @import("newton_sys.zig");
const newton_workspace = @import("newton_workspace.zig");
const jfnk_mod = @import("jfnk.zig");

/// `ESPICE_SOLVER` overrides `run`'s choice: direct, jfnk (LU-preconditioned
/// GMRES) or jfnk-nolu (Jacobi-preconditioned, no factorization at all).
pub const SolverPin = enum {
    auto,
    direct,
    jfnk,
    jfnk_nolu,

    /// The `ESPICE_SOLVER` spellings; anything else is `.auto`.
    pub const names = std.StaticStringMap(SolverPin).initComptime(.{
        .{ "direct", .direct }, .{ "jfnk", .jfnk }, .{ "jfnk-nolu", .jfnk_nolu },
    });
};

/// Developer switches (`Workspace.knobs`). The facade reads them from the
/// environment once (`espice.zig`); the solver reads no environment.
pub const Knobs = struct {
    /// `ESPICE_SOLVER`.
    pin: SolverPin = .auto,
    /// `ZP_OPDBG`: trace operating-point iterates and continuation rungs.
    opdbg: bool = false,
    /// `ZP_NEWTON_DEBUG`: trace every Newton iterate.
    newton_debug: bool = false,
};

/// Per-circuit Newton scratch (`newton_workspace.zig`).
pub const Workspace = newton_workspace.Workspace;

/// Newton scratch for W lanes (`newton_workspace.zig`).
pub const LaneWorkspace = newton_workspace.LaneWorkspace;

/// `--timing-in-depth` wall-time split of `newton` (`newton_workspace.zig`).
pub const Prof = newton_workspace.Prof;

/// Jacobian-free Newton-Krylov (`jfnk.zig`).
pub const jfnk = jfnk_mod.jfnk;

/// The user-facing accuracy profile (SPICE .options).
pub const Tolerances = num.Tolerances;

/// Controls for one nonlinear solve.
pub const Options = struct {
    max_iter: u16 = 100,
    /// Current-row absolute tolerance, amperes.
    abstol: f64 = 1e-12,
    reltol: f64 = 1e-3,
    /// Voltage-row absolute tolerance, volts.
    vntol: f64 = 1e-6,
    /// Floor of the row-scaled residual gate.
    residual_tol: f64 = 1e-9,
    /// Conductance added to every diagonal (and gmin * x to the residual).
    gmin: f64 = 1e-12,
    /// Where `gmin` lands instead of every diagonal, when non-empty.
    gmin_stamps: []const GminStamp = &.{},
    /// The solve is a cold operating point, ngspice's MODEINITJCT/INITFIX:
    /// the first iterate that passes every gate only switches to INITFLOAT,
    /// without staging device states, and the next must pass too, so
    /// `newton` publishes the second passing iterate (niiter.c, the
    /// MODEINITFIX branch; #53). Direct Newton only.
    init_fix: bool = false,
    /// Rows held near a value by a `force_g` conductance, as ngspice holds
    /// `.nodeset` nodes during MODEINITJCT/INITFIX (cktload.c). Direct Newton
    /// only.
    force: []const Force = &.{},
};

/// Conductance, in siemens, of a `Force` hold.
pub const force_g = 1e10;

/// `vals[slot] += force_g` and `rhs[row] += force_g * (x[row] - value)`: a
/// `force_g` conductance from row `row` to a source at `value`.
pub const Force = struct { slot: u32, row: u32, value: f64 };

/// One `gmin` load: `vals[slot] += gmin` and `rhs[row] += gmin * x[col]`,
/// the residual form of a conductance at (row, col).
pub const GminStamp = struct { slot: u32, row: u32, col: u32 };

/// Outcome of a nonlinear solve.
pub const Result = struct {
    converged: bool,
    iterations: u16,
};

/// The floor of the largest |x| an accepted iterate may hold. Every
/// acceptance test is relative: the delta test to |x|, the residual gate to
/// the diagonal conductance. A Newton that ran away carries both with it (a
/// FET fed from a current source at 0 V reached 5e24 V with 1e45 S
/// diagonals, and its 3e39 A residual passed), so a runaway iterate is
/// refused outright (#3).
pub const x_plausible = 1e9;

/// Private kernels for tests.zig; void outside test builds.
pub const test_access = if (@import("builtin").is_test) .{ .updateAndNorm = struct {
    fn f(x: []f64, dx: []const f64, x_old: []f64, current_row: []const bool, reltol: f64, abstol: f64, vntol: f64) f64 {
        return updateAndNorm(1, x, dx, x_old, current_row, reltol, abstol, vntol, true);
    }
}.f } else {};

/// Which gate refused an iterate, for the `ZP_OPDBG` trace. ngspice's
/// NIconvTest (maths/ni/niconv.c) has only the delta test, the first-iterate
/// floor and the device limiting flag; `.residual` and `.runaway` are ours.
/// Disabling `.residual` on vacask/mul changed the Newton count by 0.01%.
/// ngspice's CKTconvTest device tests are dead in 44.2: they bump
/// CKTnoncon, then NIiter assigns NIconvTest's return over it, and
/// CKTconvTest returns OK (niiter.c:288).
const Reject = enum { converged, first_iter, delta, limited, flipped, residual, device, runaway };

comptime {
    // Per-entry tables the loads walk: no padding.
    assert(@sizeOf(Force) == 16);
    assert(@sizeOf(GminStamp) == 12);
}

const inf = std.math.inf(f64);

/// Newton options for `tol`, with `max_iter_override` replacing ITL1. The
/// cap is never below 100: ngspice's NIiter raises any smaller maxIter to
/// 100 (niiter.c:38), so ITL2 and ITL4 only steer the callers' own
/// iteration-count heuristics, never the Newton loop itself.
pub fn optionsFromTolerances(tol: Tolerances, max_iter_override: ?u16) Options {
    return .{
        .max_iter = @max(max_iter_override orelse tol.itl1, 100),
        .abstol = tol.abstol,
        .reltol = tol.reltol,
        .vntol = tol.vntol,
        .residual_tol = tol.residual_tol,
        // Diagonal gmin is opt-in (the op gmin-stepping rung). ngspice's
        // NIiter loads none; junction gmin lives in the device models.
        .gmin = 0,
    };
}

/// Direct Newton: assemble, factor, solve, accept. Errors are the factorization's and the checkpoint's.
///
/// A converged `x` is the last linearization point x_k, not the x_k+1 of the
/// final solve, which only feeds the acceptance gates. ngspice's NIiter
/// returns before swapping CKTrhs into CKTrhsOld (niiter.c), and CKTdump and
/// every device state read CKTrhsOld. A NaN or infinite iterate never
/// converges. `ws` must come from `Workspace.init` on `sys`'s n and pattern.
/// Asserts that `ws` and `x` are sized for `sys`.
pub fn newton(
    sys: anytype,
    ws: *Workspace,
    x: []f64,
    t: f64,
    opts: Options,
    hook: anytype,
) !Result {
    return newtonLanes(1, sys, ws, x, t, opts, hook);
}

/// `Workspace` at W = 1, `LaneWorkspace(W)` above.
pub fn WorkspaceOf(comptime W: usize) type {
    return if (W == 1) Workspace else LaneWorkspace(W);
}

/// `Result` at W = 1, `LaneResult(W)` above.
pub fn ResultOf(comptime W: usize) type {
    return if (W == 1) Result else LaneResult(W);
}

/// Outcome of W lanes' nonlinear solves.
pub fn LaneResult(comptime W: usize) type {
    return struct {
        converged: Lanes(W).Mask,
        /// Lanes whose own factor failed with `error.SingularMatrix`, where
        /// a scalar `newton` would have returned that error.
        singular: Lanes(W).Mask,
        /// Per lane, as `Result.iterations`; `max_iter` for a lane that
        /// did not converge.
        iterations: [W]u16,
    };
}

/// `newton` over W lanes, each bitwise the scalar `newton` on its own
/// system: lanes share the iteration count but never a decision. A lane
/// leaves `active` when it converges or its factor fails, and freezes;
/// its slots still compute and are discarded. At W = 1 this is `newton`
/// (the `gen_newton` probe). Above, `sys` is a lane system whose methods
/// take and return lane masks and read `sys.active` (`lane_system.LaneSystem`),
/// and `t` is per lane; the lanes `sys.active` holds on entry are solved.
/// Every other error fails all lanes: a checkpoint
/// cancel, or out of memory in a lane's factor. Traces (`ZP_OPDBG`,
/// `ZP_NEWTON_DEBUG`), `evalFollows`, the device LU and `certify` are W = 1
/// only.
pub fn newtonLanes(
    comptime W: usize,
    sys: anytype,
    ws: *WorkspaceOf(W),
    x: []Lanes(W).V,
    t: Lanes(W).V,
    opts: Options,
    hook: anytype,
) !ResultOf(W) {
    const L = Lanes(W);
    const S = newton_sys.Deref(@TypeOf(sys));
    assert(ws.slv.n == sys.n);
    assert(ws.dx.len >= sys.n);
    assert(ws.x_old.len >= sys.n);
    assert(x.len >= sys.n);
    const slv = &ws.slv;
    const dx = ws.dx;
    const x_old = ws.x_old;
    var iter: u16 = 0;
    var init_fix: L.Mask = if (opts.init_fix) L.every else L.none;
    // Above W = 1 the caller marks the lanes it fills (`sys.active`).
    var active: L.Mask = if (W == 1) L.every else sys.active;
    // The lanes' outcomes; void at W = 1, which returns from the loop.
    var out: if (W == 1) void else LaneResult(W) = if (W == 1) {} else .{ .converged = L.none, .singular = L.none, .iterations = @splat(opts.max_iter) };
    var limit: L.V = L.splat(x_plausible);
    if (comptime @hasDecl(S, "beginSolve")) sys.beginSolve();
    const prof = &ws.prof;

    while (iter < opts.max_iter) : (iter += 1) {
        if (W > 1) sys.active = active;
        if (comptime @hasDecl(S, "checkpoint")) if (iter != 0)
            try sys.checkpoint(.{ .phase = .nonlinear, .completed = iter, .total = opts.max_iter });
        if (comptime @hasDecl(S, "advanceIteration")) if (iter != 0) sys.advanceIteration(x_old);
        prof.start();
        hook.assemble(sys, x, t);
        prof.lap(.eval);
        const v = hook.vals(sys);
        loadExtras(sys, v, x, opts);
        if (iter == 0) limit = plausibleLimit(sys, x[0..sys.n], sys.rhs[0..sys.n], v);
        var norm_f: f64 = 0; // read only by the traces
        if (W == 1) {
            if (ws.knobs.newton_debug or ws.knobs.opdbg) {
                norm_f = num.normInf(sys.rhs[0..sys.n]);
            }
            if (ws.knobs.newton_debug)
                std.debug.print("  it={d} |F|={e} x={any}\n", .{ iter, norm_f, x[0..@min(sys.n, 8)] });
        }
        // Before the factor and solve, which may overwrite `v` and `rhs`.
        const load_ok: L.Mask = if (comptime @hasDecl(S, "loadCheck")) sys.loadCheck(x, v, sys.rhs[0..sys.n], opts.reltol, opts.abstol) else L.every;
        prof.lap(.load);
        if (W == 1) {
            // The device LU (dev/solvers/gpu-lu.md) is bitwise this factor and
            // solve; false leaves both to the host.
            const on_device = if (comptime @hasDecl(S, "deviceSolve")) sys.deviceSolve(slv, v, dx) orelse false else false;
            if (!on_device) {
                slv.factor(v, newton_sys.executionOf(sys)) catch |e| {
                    if (ws.knobs.opdbg) {
                        var nan_cnt: usize = 0;
                        for (v[0..sys.nnz]) |vi| {
                            if (!std.math.isFinite(vi)) nan_cnt += 1;
                        }
                        const max_x = num.normInf(x[0..sys.n]);
                        std.debug.print("  newton it={d} FACTOR FAIL {} nan_vals={d} max|x|={e:.3} |F|={e:.3}\n", .{ iter, e, nan_cnt, max_x, norm_f });
                    }
                    return e;
                };
                prof.lap(.factor);
                slv.solveNeg(sys.rhs, dx);
            }
        } else {
            const bad = try slv.factorSolveNeg(v, sys.rhs[0..sys.n], dx, active, newton_sys.executionOf(sys));
            out.singular |= bad;
            active &= ~bad;
            prof.lap(.factor);
            if (!L.any(active)) break;
        }
        prof.lap(.solve);
        // The INITFIX pass does not stage: its converged iterate is not the
        // answer, and staging there restarted a Newton-history limiter from a
        // stale value, so the extra pass solved F(lim(x)) = 0 (#53).
        const st = finalizeStep(W, sys, hook, x, dx, x_old, sys.rhs, v, iter, t, opts, limit, load_ok, L.not(init_fix), active);
        prof.lap(.update);
        if (W == 1) if (ws.knobs.opdbg) {
            var fi: usize = 0;
            var di: usize = 0;
            for (0..sys.n) |i| {
                if (@abs(sys.rhs[i]) > @abs(sys.rhs[fi])) fi = i;
                if (@abs(dx[i]) > @abs(dx[di])) di = i;
            }
            const name_fi = newton_sys.sysNodeName(sys, @intCast(fi));
            const name_di = newton_sys.sysNodeName(sys, @intCast(di));
            std.debug.print("  newton it={d} |F|={e:.3}@{d}({s}) dx={e:.3}@{d}({s}) x={e:.3} scaled={e:.3} conv={} why={s}\n", .{ iter, norm_f, fi, name_fi, dx[di], di, name_di, x[di], st.scaled, L.any(st.converged), @tagName(st.why) });
        };
        // INITFIX lanes that passed switch to INITFLOAT; the rest are done.
        // The loop exit (class B): at W = 1 the INITFIX pass or the
        // return; above, each lane that passed leaves INITFIX or freezes.
        if (W == 1) {
            if (st.converged and init_fix) {
                init_fix = false;
            } else if (st.converged) {
                @memcpy(x[0..sys.n], x_old[0..sys.n]);
                return .{ .converged = true, .iterations = iter + 1 };
            }
        } else {
            const done = st.converged & ~init_fix;
            init_fix &= ~st.converged;
            var it = L.each(done);
            while (it.next()) |l| out.iterations[l] = iter + 1;
            for (x[0..sys.n], x_old[0..sys.n]) |*xi, xo| xi.* = L.select(done, xo, xi.*);
            out.converged |= done;
            active &= ~done;
            if (!L.any(active)) break;
        }
    }
    if (W == 1) return .{ .converged = false, .iterations = opts.max_iter };
    return out;
}

/// Adds the loads `newton` stamps on top of `hook.assemble`: the diagonal
/// (or `opts.gmin_stamps`) gmin and the `opts.force` holds, into `v` and
/// `sys.rhs`. `v` is `hook.vals(sys)`. Asserts that `x` holds n entries
/// and `v` nnz.
pub fn loadExtras(sys: anytype, v: anytype, x: anytype, opts: Options) void {
    const L = lanes.Of(@TypeOf(v[0]));
    assert(x.len >= sys.n);
    assert(v.len >= sys.nnz);
    newton_sys.loadGmin(L.V, sys, v, sys.rhs[0..sys.n], x, opts);
    for (opts.force) |f| {
        v[f.slot] += L.splat(force_g);
        sys.rhs[f.row] += L.splat(force_g) * (x[f.row] - L.splat(f.value));
    }
}

/// The row-scaled residual gate's tolerance on a row whose diagonal is
/// `diag` and unknown `xi`: `newton` and `jfnk` accept only when every row
/// has |F_i| at or below it. A zero `diag` (a voltage-defined branch row)
/// is gated at scale 1, a tolerance in volts (#152).
pub fn residualGate(opts: Options, diag: anytype, xi: @TypeOf(diag)) @TypeOf(diag) {
    const L = lanes.Of(@TypeOf(diag));
    const scale = @abs(diag);
    // A zero diagonal is a voltage-defined branch row (V/E/H source): it is
    // gated at scale 1, a loose 10*(reltol*|x| + vntol) that an exact
    // solution's O(gain * eps) residual (a 1e9-gain E source reads 2.7e-8)
    // still passes, but that refuses x = 0 when the only residual sits on
    // such a row (#152).
    const s = L.select(L.cmp(.eq, scale, L.splat(0)), L.splat(1.0), scale);
    return @max(L.splat(opts.residual_tol), L.splat(10.0) * s * (L.splat(opts.reltol) * @abs(xi) + L.splat(opts.vntol)));
}

/// The runaway bound of one solve, read off its first assembly: 1e3 times
/// the scale the circuit sets there, the start |x| or the drive |F| times
/// the largest diagonal conductance, between `x_plausible` and 1e21. A 1e12 V
/// source or a 1e-12 ohm resistor (1e12 A from 1 V) solves; with the fixed
/// 1e9 every iterate was refused and an operating point published zeros
/// (#152).
/// ponytail: the 1e18 scale cap keeps a solve that starts already run away
/// (#3's 5e24 V) refused; a bound from the circuit's sources and constant
/// stamps alone would need no cap.
pub fn plausibleLimit(sys: anytype, x: anytype, f: anytype, vals: anytype) @TypeOf(x[0]) {
    const L = lanes.Of(@TypeOf(x[0]));
    var g: L.V = L.splat(1);
    for (sys.diag_slots[0..sys.n]) |slot| g = @max(g, @abs(vals[slot]));
    const scale = @min(@max(normInf(L.V, x), normInf(L.V, f) * g), L.splat(1e18));
    // Negated, so a NaN start keeps the floor.
    return L.select(L.not(L.cmp(.gt, L.splat(1e3) * scale, L.splat(x_plausible))), L.splat(x_plausible), L.splat(1e3) * scale);
}

/// Solves with `newton`, or with `jfnk` under an `ESPICE_SOLVER` pin (falling
/// back to `newton` when it fails). Clears device limiting on exit.
pub fn run(
    sys: anytype,
    ws: *Workspace,
    x: []f64,
    t: f64,
    opts: Options,
    hook: anytype,
) !Result {
    const S = newton_sys.Deref(@TypeOf(sys));
    // Function scope: a defer inside an `if` block would run before the solve.
    defer if (comptime @hasDecl(S, "clearLimits")) sys.clearLimits();

    const pin = ws.knobs.pin;
    if ((pin == .jfnk or pin == .jfnk_nolu) and sys.n > 0) {
        if (jfnk_mod.jfnk(sys, ws, x, t, opts, hook)) |r| {
            if (r.converged) return r;
        } else |err| if (err == error.QueryCancelled) return err;
    }
    // Direct Newton is the default because it was faster on every fixture
    // size, both backends, op and tran alike (best-of-2 wall seconds):
    //
    //   fixture                unkn  backend  direct   jfnk  jfnk-nolu
    //   resistor_grid_32x32     ~1k  cpu        0.01   0.02       0.16
    //   resistor_grid_100x100  ~10k  cpu        0.19   0.21      10.76
    //   rc_ladder_1k            ~1k  cpu        0.22   5.65       8.74
    //   rc_ladder_10k          ~10k  cpu        2.19   4.76      39.40
    //   rc_ladder_10k          ~10k  cuda       2.03   4.77      39.88
    //
    // JFNK still factors the Jacobian as its preconditioner and adds up to
    // 30 full device sweeps per step. It is not a superset either:
    // scaling/parallel_inverters_100 converges only under JFNK, and only when
    // JFNK drives the whole op ladder, while convergence/diode_bridge goes the
    // other way. An automatic retry after a failed direct solve cost 10x on
    // ensemble/pvt_corners, because `run` is called per rung and per
    // timestep. ponytail: the pin is the interface until the op ladder itself
    // learns to restart under JFNK.
    return newton(sys, ws, x, t, opts, hook);
}

/// The delta gate: x_old = x; x += dx; returns max |dx| / (reltol *
/// max(|x|, |x_old|) + atol), atol = abstol on current rows, vntol
/// elsewhere, per lane. Lanes outside `active` keep x and x_old (their score
/// is garbage). A row whose new x or dx is NaN or infinite scores inf; a
/// zero step on a zero tolerance (0/0) scores 0. The max runs on the i64
/// bit patterns of the nonnegative scores: exact and order-independent, and
/// the loop vectorizes (float `@max` keeps it scalar: 1.2 against 0.8 ns per
/// row).
pub fn updateAndNorm(comptime W: usize, x: []Lanes(W).V, dx: []const Lanes(W).V, x_old: []Lanes(W).V, current_row: []const bool, reltol: f64, abstol: f64, vntol: f64, active: Lanes(W).Mask) Lanes(W).V {
    const L = Lanes(W);
    const I = if (W == 1) i64 else @Vector(W, i64);
    // Up front, so the loop carries no per-slice bounds checks.
    assert(dx.len >= x.len);
    assert(x_old.len >= x.len);
    assert(current_row.len >= x.len);
    var worst: I = if (W == 1) 0 else @splat(0);
    for (x, dx[0..x.len], x_old[0..x.len], current_row[0..x.len]) |*xi, d, *xoi, cur| {
        const xo = xi.*;
        const xn = L.select(active, xo + d, xo);
        xoi.* = L.select(active, xo, xoi.*);
        xi.* = xn;
        const tol = L.splat(reltol) * @max(@abs(xn), @abs(xo)) + L.splat(if (cur) abstol else vntol);
        // A lazy `and` at W = 1: LLVM turns it into an integer finite test
        // and the eager `both` into a float compare pair, which is not
        // the scalar kernel's code (gen_update_and_norm).
        const finite = if (W == 1) @abs(xn) < inf and @abs(d) < inf else L.both(L.cmp(.lt, @abs(xn), L.splat(inf)), L.cmp(.lt, @abs(d), L.splat(inf)));
        const q = @abs(d) / tol;
        const score = L.select(L.not(finite), L.splat(inf), L.select(L.cmp(.eq, q, q), q, L.splat(0)));
        worst = @max(worst, @as(I, @bitCast(score)));
    }
    const result: L.V = @bitCast(worst);
    assert(!L.any(L.cmp(.lt, result, L.splat(0))));
    return result;
}

// ---- private ----

/// `finalizeStep`'s verdict: the lanes it accepted and each lane's delta
/// score; `why` names the first gate that refused, read by the W = 1 trace.
fn Step(comptime W: usize) type {
    return struct { converged: Lanes(W).Mask, scaled: Lanes(W).V, why: Reject = .converged };
}

/// `num.normInf` per lane: the max of |x_i|'s bit patterns, NaN winning.
fn normInf(comptime V: type, x: []const V) V {
    if (V == f64) return num.normInf(x);
    const I = @Vector(lanes.Of(V).width, u64);
    var m: I = @splat(0);
    for (x) |xi| m = @max(m, @as(I, @bitCast(@abs(xi))));
    return @bitCast(m);
}

/// Whether every unknown is finite and within `limit`. Chunked: each
/// 64-unknown block is a branchless OR the loop vectorizer takes, with one
/// branch per block; `!(a <= b)` fails NaN too. 0.07-0.09 ns per unknown
/// against the early-return loop's 0.54 (ReleaseFast, n = 64 to 16384).
fn plausible(x: []const f64, limit: f64) bool {
    const block = 64;
    var i: usize = 0;
    while (i + block <= x.len) : (i += block) {
        var bad: u64 = 0;
        for (x[i..][0..block]) |xi| bad |= @intFromBool(!(@abs(xi) <= limit));
        if (bad != 0) return false;
    }
    var bad: u64 = 0;
    for (x[i..]) |xi| bad |= @intFromBool(!(@abs(xi) <= limit));
    return bad == 0;
}

/// The lanes with every unknown finite and within their `limit` lane:
/// `plausible` per lane.
fn plausibleLanes(comptime V: type, x: []const V, limit: V) lanes.Of(V).Mask {
    const L = lanes.Of(V);
    if (V == f64) return plausible(x, limit);
    var bad: L.Mask = L.none;
    for (x) |xi| bad |= ~L.cmp(.lte, @abs(xi), limit);
    return ~bad;
}

/// The lanes with no `residualMisses` row.
fn residualLanes(comptime V: type, opts: Options, residual: []const V, x: []const V, vals: []const V, diag_slots: []const u32) lanes.Of(V).Mask {
    const L = lanes.Of(V);
    if (V == f64) return residualMisses(opts, residual, x, vals, diag_slots) == 0;
    assert(residual.len >= x.len);
    assert(diag_slots.len >= x.len);
    var bad: L.Mask = L.none;
    for (residual[0..x.len], x, diag_slots[0..x.len]) |ri, xi, slot|
        bad |= ~L.cmp(.lte, @abs(ri), residualGate(opts, vals[slot], xi));
    return ~bad;
}

/// Rows whose |F_i| exceeds `residualGate`, NaN included. A branchless
/// count, not an early return: it runs only on an iterate that passed the
/// delta gate, where it nearly always passes and reads every row anyway.
/// 0.52 ns per row against the early-return loop's 1.22 (ReleaseFast, n =
/// 64 to 16384).
fn residualMisses(opts: Options, residual: []const f64, x: []const f64, vals: []const f64, diag_slots: []const u32) usize {
    assert(residual.len >= x.len);
    assert(diag_slots.len >= x.len);
    var misses: usize = 0;
    for (residual[0..x.len], x, diag_slots[0..x.len]) |ri, xi, slot|
        misses += @intFromBool(!(@abs(ri) <= residualGate(opts, vals[slot], xi)));
    return misses;
}

/// Applies `dx` in the `active` lanes and runs the acceptance gates in
/// order: device limiting, first iterate, per-row delta, row-scaled
/// residual, the device load test (`load_ok`, `sys.loadCheck` at the
/// assembled x), device convergence, then state staging in the lanes of
/// `stage`. A lane refused by a gate meets no later one that calls into
/// its devices; the pure gates run on every lane until none is left.
fn finalizeStep(
    comptime W: usize,
    sys: anytype,
    hook: anytype,
    x: []Lanes(W).V,
    dx: []const Lanes(W).V,
    x_old: []Lanes(W).V,
    residual: []const Lanes(W).V,
    vals: []const Lanes(W).V,
    iter: u16,
    t: Lanes(W).V,
    opts: Options,
    limit: Lanes(W).V,
    load_ok: Lanes(W).Mask,
    stage: Lanes(W).Mask,
    active: Lanes(W).Mask,
) Step(W) {
    const L = Lanes(W);
    const S = newton_sys.Deref(@TypeOf(sys));
    const n = sys.n;
    assert(residual.len >= n);
    assert(vals.len >= sys.nnz);
    const scaled = updateAndNorm(W, x[0..n], dx[0..n], x_old[0..n], sys.current_row[0..n], opts.reltol, opts.abstol, opts.vntol, active);

    // The first-iterate and delta gates are already decided and neither
    // reads the limiter, so the next assemble at the limited x is certain:
    // a system that batches its limit pass with that eval may start both.
    if (W == 1) if (comptime @hasDecl(S, "evalFollows")) {
        if ((iter == 0 or scaled >= 1.0) and iter + 1 < opts.max_iter) sys.evalFollows(x, t, false);
    };

    const limited: L.Mask = if (comptime @hasDecl(S, "applyLimits")) sys.applyLimits(x, x_old) else L.none;

    var ok = L.both(active, L.not(limited));
    if (!L.any(ok)) return .{ .converged = L.none, .scaled = scaled, .why = .limited };
    if (iter == 0) return .{ .converged = L.none, .scaled = scaled, .why = .first_iter };
    ok = L.both(ok, L.not(L.cmp(.gte, scaled, L.splat(1.0))));
    if (!L.any(ok)) return .{ .converged = L.none, .scaled = scaled, .why = .delta };
    ok = L.both(ok, plausibleLanes(L.V, x[0..n], limit));
    if (!L.any(ok)) return .{ .converged = L.none, .scaled = scaled, .why = .runaway };
    ok = L.both(ok, residualLanes(L.V, opts, residual[0..n], x[0..n], vals, sys.diag_slots[0..n]));
    if (!L.any(ok)) return .{ .converged = L.none, .scaled = scaled, .why = .residual };
    ok = L.both(ok, load_ok);
    if (!L.any(ok)) return .{ .converged = L.none, .scaled = scaled, .why = .device };
    if (comptime @hasDecl(S, "checkConvergence")) {
        ok = L.both(ok, if (W == 1) sys.checkConvergence(x) else sys.checkConvergence(x, ok));
        if (!L.any(ok)) return .{ .converged = L.none, .scaled = scaled, .why = .device };
    }
    // Last, at the x_k `newton` publishes: staging per iterate re-entered every
    // model core once per instance per iteration (18% of
    // scaling/parallel_inverters_100), and devices whose core reads the
    // staged latches (cswitch/vswitch hysteresis) saw F change mid-solve.
    // A device that flips here forces one more iterate.
    // Before staging: a device whose eval reads its staged history (LTRA
    // appends the point it stages) would see this x twice (#53).
    if (W == 1) if (comptime @hasDecl(S, "certify") and S.certify) if (L.any(stage)) sys.certifyPoint(hook, x_old[0..n], t, opts);
    if (comptime @hasDecl(S, "updateStates")) {
        const staging = L.both(ok, stage);
        if (L.any(staging)) {
            const flipped: L.Mask = if (W == 1) sys.updateStates(x_old[0..n]) != null else sys.updateStates(x_old[0..n], staging);
            ok = L.both(ok, L.not(flipped));
            if (!L.any(ok)) return .{ .converged = L.none, .scaled = scaled, .why = .flipped };
        }
    }
    // The first-iterate and delta gates passed.
    assert(iter > 0);
    assert(!L.any(L.both(ok, L.cmp(.gte, scaled, L.splat(1.0)))));
    return .{ .converged = ok, .scaled = scaled };
}

test "plausible and residualMisses match a scalar oracle at every length, NaN and inf included" {
    var prng = std.Random.DefaultPrng.init(0xC0);
    const r = prng.random();
    const len_max = 200;
    var x: [len_max]f64 = undefined;
    var res: [len_max]f64 = undefined;
    var vals: [len_max + 1]f64 = undefined;
    var slots: [len_max]u32 = undefined;
    const opts: Options = .{};
    for (0..len_max + 1) |n| for (0..4) |plant| {
        for (x[0..n], res[0..n], slots[0..n], 0..) |*xi, *ri, *si, i| {
            xi.* = (r.float(f64) - 0.5) * 10;
            ri.* = (r.float(f64) - 0.5) * 1e-8;
            si.* = @intCast((i * 7) % (len_max + 1));
        }
        for (&vals, 0..) |*v, i| v.* = if (i % 5 == 0) 0 else r.float(f64) * 1e-2;
        if (n > 0) {
            const at = r.uintLessThan(usize, n);
            switch (plant) {
                0 => {},
                1 => {
                    x[at] = std.math.nan(f64);
                    res[at] = std.math.nan(f64);
                },
                2 => {
                    x[at] = -std.math.inf(f64);
                    res[at] = std.math.inf(f64);
                },
                else => x[at] = 5, // else: plant 3, exactly at the limit (plausible)
            }
        }
        var want_plausible = true;
        var want_misses: usize = 0;
        for (0..n) |i| {
            if (!(@abs(x[i]) <= 5)) want_plausible = false;
            const diag = @abs(vals[slots[i]]);
            const scale = if (diag == 0) 1.0 else diag;
            const gate = @max(opts.residual_tol, 10.0 * scale * (opts.reltol * @abs(x[i]) + opts.vntol));
            if (!(@abs(res[i]) <= gate)) want_misses += 1;
        }
        try std.testing.expectEqual(want_plausible, plausible(x[0..n], 5));
        try std.testing.expectEqual(want_misses, residualMisses(opts, res[0..n], x[0..n], &vals, slots[0..n]));
    };
}
