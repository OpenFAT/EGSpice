//! Jacobian-free Newton-Krylov: a system, its hook and a start x in, x at a
//! converged point (or the last iterate) and a `Result` out. Each step
//! solves J dx = -F by one GMRES cycle (`gmres.Gmres`) on finite-difference
//! products J v, left-preconditioned by the factored Jacobian or its
//! diagonal. Cites: Knoll and Keyes, "Jacobian-free Newton-Krylov methods",
//! J. Comput. Phys. 193(2), 2004; #152 (the branch-row residual gate).
//!
//! Contents:
//! - `pub fn jfnk(...)`
//!     - `pub fn matvec(op: *@This(), v: []const f64, w: []f64) void`

const std = @import("std");
const assert = std.debug.assert;
const num = @import("core").numerics;
const direct = @import("direct.zig");
const converger = @import("converger.zig");
const newton_sys = @import("newton_sys.zig");

const inf = std.math.inf(f64);

const sqrt_eps: f64 = 0x1p-26; // sqrt(f64 machine epsilon)

/// Jacobian-free Newton-Krylov: each step solves J dx = -F by one unrestarted cycle of
/// GMRES(min(30, n)) (`gmres.Gmres`) with finite-difference products J v,
/// left-preconditioned by the factored Jacobian (Jacobi only under
/// ESPICE_SOLVER=jfnk-nolu). A cycle that stops short is taken as it stands.
/// Same gates as `newton`, with the residual gate on F, but a converged `x`
/// is x_k+1. A failed checkpoint returns `error.QueryCancelled`; the first
/// call on a workspace allocates its GMRES scratch (`error.OutOfMemory`).
/// Asserts that `ws` and `x_full` are sized for `sys` and that `sys` has an
/// unknown.
pub fn jfnk(
    sys: anytype,
    ws: *converger.Workspace,
    x_full: []f64,
    t: f64,
    opts: converger.Options,
    hook: anytype,
) !converger.Result {
    const S = newton_sys.Deref(@TypeOf(sys));
    const n: usize = sys.n;
    assert(n > 0);
    assert(ws.slv.n == n);
    assert(x_full.len >= n);
    const krylov = try ws.ensureKrylov();
    const x_pert = ws.jf[0..n];
    const f0 = ws.jf[n..][0..n]; // F(x) + gmin * x at the outer iterate
    const b = ws.jf[2 * n ..][0..n]; // -M^-1 f0, the Krylov right-hand side
    const diag = ws.jf[3 * n ..][0..n]; // Jacobi inverse diagonal
    const r = ws.dx[0..n]; // becomes dx after the Krylov solve
    const x_old = ws.x_old[0..n];
    const x = x_full[0..n];
    // A null solver is what makes this matrix-free.
    const slv: ?*direct.Solver = if (ws.knobs.pin == .jfnk_nolu) null else &ws.slv;

    // w = M^-1 (F(x + eps v) - f0) / eps, eps = sqrt(eps_mach) * max(||x||, 1) / ||v||:
    // left preconditioning inside the operator, so GMRES's residual and
    // stop are on M^-1 (F + J dx), as the hand-rolled solver's were.
    // Right preconditioning (a `precond` decl) left the 2x2 circle test
    // (solver tests) 1e-6 from its root; the hand-rolled left form landed
    // within 1e-8.
    const Sys = @TypeOf(sys);
    const Hook = @TypeOf(hook);
    const FdOp = struct {
        sys: Sys,
        hook: Hook,
        x: []const f64,
        x_norm: f64 = 0,
        x_pert: []f64,
        f0: []const f64,
        diag: []const f64,
        slv: ?*direct.Solver,
        t: f64,
        opts: converger.Options,

        /// Writes w = M^-1 (F(x + eps v) - f0) / eps, the preconditioned
        /// finite-difference product J v.
        pub fn matvec(op: *@This(), v: []const f64, w: []f64) void {
            const v_norm = num.norm2(v);
            // J 0 = 0 exactly: GMRES's initial residual costs no device eval.
            if (v_norm == 0) {
                @memset(w, 0);
                return;
            }
            const eps = if (v_norm > 1e-30) sqrt_eps * op.x_norm / v_norm else sqrt_eps;
            @memcpy(op.x_pert, op.x);
            num.axpy(op.x_pert, eps, v);
            assembleShifted(op.sys, op.x_pert, op.t, op.opts, op.hook);
            // The finite-difference product; a plain loop LLVM vectorizes.
            const inv = 1.0 / eps;
            for (w, op.sys.rhs[0..w.len], op.f0[0..w.len]) |*wi, fi, f0i| wi.* = (fi - f0i) * inv;
            applyPreconditioner(w, op.diag, op.slv);
        }
    };
    var op: FdOp = .{ .sys = sys, .hook = hook, .x = x, .x_pert = x_pert, .f0 = f0, .diag = diag, .slv = slv, .t = t, .opts = opts };

    if (comptime @hasDecl(S, "beginSolve")) sys.beginSolve();
    @memcpy(x_old, x);

    var limit: f64 = converger.x_plausible;
    var iter: u16 = 0;
    while (iter < opts.max_iter) : (iter += 1) {
        if (comptime @hasDecl(S, "checkpoint")) if (iter != 0) {
            sys.checkpoint(.{ .phase = .nonlinear, .completed = iter, .total = opts.max_iter }) catch
                return error.QueryCancelled;
        };
        if (comptime @hasDecl(S, "advanceIteration")) if (iter != 0) sys.advanceIteration(x_old);

        assembleShifted(sys, x, t, opts, hook);
        @memcpy(f0, sys.rhs[0..n]);
        const v = hook.vals(sys);
        if (iter == 0) limit = converger.plausibleLimit(sys, x, f0, v);
        newton_sys.loadGmin(f64, sys, v, null, x, opts);
        // Before the factor, which may overwrite `v`.
        const load_ok = if (comptime @hasDecl(S, "loadCheck")) sys.loadCheck(x, v, f0, opts.reltol, opts.abstol) else true;
        buildDiagPreconditioner(sys, v, diag);
        // A failed factor leaves `s.factored` false, and `applyPreconditioner`
        // then falls back to the Jacobi diagonal: every error is handled there.
        if (slv) |s| s.factor(v, newton_sys.executionOf(sys)) catch |err| switch (err) {
            error.SingularMatrix, error.OutOfMemory, error.OutOfWorkspace => assert(!s.factored),
        };

        // ||M^-1 f0||: the preconditioned step already within abstol.
        // b = -M^-1 f0, beta = ||b||.
        num.scale(b, -1, f0);
        applyPreconditioner(b, diag, slv);
        const beta = num.norm2(b);
        if (beta < opts.abstol and residualConverged(sys, hook, x, f0, opts, limit)) {
            @memset(r, 0);
            @memcpy(x_old, x);
            const limited = newton_sys.applyLimits(sys, x, x_old);
            if (iter > 0 and !limited and load_ok and newton_sys.acceptStep(sys, x))
                return .{ .converged = true, .iterations = iter + 1 };
            continue;
        }

        // GMRES stops at ||M^-1 (F + J dx)|| <= abstol / 10, but takes at
        // least one Arnoldi step: a beta already under that would otherwise
        // return dx = 0 and stall on the residual gate.
        @memset(r, 0);
        op.x_norm = @max(num.norm2(x), 1.0);
        _ = krylov.solve(&op, b, r, @min(0.1 * opts.abstol / beta, 0.5), 0);

        // Limiting may clamp x, so it runs before the residual gate.
        const scaled = converger.updateAndNorm(1, x, r, x_old, sys.current_row[0..n], opts.reltol, opts.abstol, opts.vntol, true);
        const limited = newton_sys.applyLimits(sys, x, x_old);
        const residual_ok = residualConverged(sys, hook, x, f0, opts, limit);
        // Publishes x_k+1: JFNK is not ngspice's iteration, and its inexact
        // steps leave x_k further out than a direct solve does.
        if (iter > 0 and scaled < 1.0 and residual_ok and !limited and load_ok and newton_sys.acceptStep(sys, x))
            return .{ .converged = true, .iterations = iter + 1 };
    }
    return .{ .converged = false, .iterations = opts.max_iter };
}

// ---- private ----

/// Residual at x plus the gmin load into `sys.rhs`.
fn assembleShifted(sys: anytype, x: []const f64, t: f64, opts: converger.Options, hook: anytype) void {
    assert(x.len == sys.n);
    if (@hasDecl(@TypeOf(hook), "residual"))
        hook.residual(sys, x, t)
    else
        hook.assemble(sys, x, t);
    newton_sys.loadGmin(f64, sys, null, sys.rhs[0..x.len], x, opts);
}

/// Inverse diagonal of `vals`, which already carries the gmin load.
fn buildDiagPreconditioner(sys: anytype, vals: []const f64, diag: []f64) void {
    assert(diag.len >= sys.n);
    for (0..sys.n) |i| {
        const d = vals[sys.diag_slots[i]];
        diag[i] = if (@abs(d) > 1e-30) 1.0 / d else 1.0;
    }
}

/// r = M^-1 r: the LU when it factored, else the Jacobi diagonal.
fn applyPreconditioner(r: []f64, diag: []const f64, slv: ?*direct.Solver) void {
    assert(diag.len >= r.len);
    if (slv) |s| {
        if (s.factored) {
            s.solve(r, r);
            return;
        }
    }
    for (r, diag[0..r.len]) |*ri, di| ri.* *= di;
}

/// Row-scaled residual gate, as in `finalizeStep`, that also rejects a
/// non-finite residual or an iterate past `limit`. A row with no diagonal
/// (a V source's branch row) holds its KVL residual in volts to
/// 10 (reltol |x| + vntol): JFNK has no delta test to govern it, and
/// skipping the row accepted a cold x = 0 whose only residual was the
/// source's own value (#152).
fn residualConverged(sys: anytype, hook: anytype, x: []const f64, residual: []const f64, opts: converger.Options, limit: f64) bool {
    assert(x.len == sys.n);
    var ok = true;
    for (x, residual, 0..) |xi, ri, i| {
        const rt = converger.residualGate(opts, hook.diagAt(sys, sys.diag_slots[i]), xi);
        // Negated comparisons reject NaN too.
        if (!(@abs(ri) <= rt) or !(@abs(xi) <= limit)) ok = false;
    }
    return ok;
}

test "the Jacobi preconditioner scales each row by its inverse diagonal" {
    var r = [_]f64{ 2, -3, 4, std.math.nan(f64) };
    const diag = [_]f64{ 0.5, 2, -0.25, 1, 9 };
    applyPreconditioner(&r, &diag, null);
    try std.testing.expectEqualSlices(f64, &.{ 1, -6, -1 }, r[0..3]);
    try std.testing.expect(std.math.isNan(r[3]));
}
