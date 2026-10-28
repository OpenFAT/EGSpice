//! MATEX: matrix-exponential transient for linear circuits (R-MATEX). A
//! linear circuit at its operating point in, the probes at every step out,
//! as `Waveform` rows. Two factorizations serve every step: Arnoldi on
//! (C + gamma*G)^-1 C approximates exp(hA), and G gives the source terms.
//! Only source transition spots and `h_output_cap` bound the step. G must be
//! regular.
//! Cites: the MATEX paper's eq. 5 (the piecewise-linear source update),
//! expokit's breakdown tolerance, Verilog-AMS LRM §4.6.1, #51 and #125.
//! ponytail: no nonlinear (EPIRK) path; add one when a const-Jacobian check
//! says a circuit needs linearization.
//!
//! Contents:
//! - `pub const Options = core.query.Matex`
//! - `pub const test_access`
//! - `pub fn run(ctx: *const root.RunCtx, opts: Options) !root.Result`

const std = @import("std");
const assert = std.debug.assert;
const num = @import("core").numerics;
const solvers = @import("solver");
const root = @import("../types.zig");
const combinePlanes = @import("../Circuit.zig").combinePlanes;
const Waveform = @import("types.zig").Waveform;
/// The waveform rows `run` reserves up front (core/validate.zig, which query
/// validation shares).
const estPoints = @import("core").validate.matexEstPoints;

const dense_lu = solvers.dense_lu;
const Solver = solvers.direct.Solver;

/// `.matex` query options, defined in core/query.zig.
pub const Options = @import("core").query.Matex;

/// Private implementation access for the analysis test suite.
pub const test_access = if (@import("builtin").is_test) .{
    .denseMatMul = denseMatMul,
    .expmSmall = expmSmall,
    .cscMulVec = cscMulVec,
} else {};

const W = std.simd.suggestVectorLength(f64) orelse 8;
const V = @Vector(W, f64);

/// Most steps one run takes: a runaway guard, past any real spot count.
const steps_max = 10_000_000;
/// Most source breakpoints `collectTransitionSpots` walks.
const spots_max = 100_000;
/// Spots and step starts closer than this, in seconds, are one time.
const spot_merge_s = 1e-18;
/// A step shorter than this, in seconds, ends the march.
const step_min_s = 1e-30;
/// A finite f64 is below 2^1024, so 1025 halvings bring it under 0.5.
const halvings_max = 1025;

/// One Arnoldi run's outcome.
const ArnoldiResult = struct {
    /// Krylov dimension reached; 0 when v is zero.
    m: u32,
    /// ||v||.
    beta: f64,
    /// Lucky breakdown or an accepted posterior; false when m_max was hit
    /// first, which fails the step (`Options.m_max`).
    converged: bool,
};

/// Returns the march from ctx.x_op with the piecewise-linear source integral
/// (MATEX eq. 5), stepping to each transition spot or h_output_cap, as
/// point-major rows (time, probes...) in `ctx.allocator`. Fails with
/// error.EmptyCircuit for n = 0; error.MatexDidNotConverge when Arnoldi
/// reaches `m_max` without converging, expm meets an inf, or a step comes
/// out non-finite; error.Singular from the dense LU of H_m (a DAE of index
/// above 1); and the sparse solver's errors from factoring G or
/// C + gamma*G.
pub fn run(ctx: *const root.RunCtx, opts: Options) !root.Result {
    const ckt = ctx.circuit;
    if (ckt.n == 0) return error.EmptyCircuit;
    const n: usize = ckt.n;
    const nn: u32 = ckt.n;
    assert(ctx.x_op.len >= n);
    // `ctx.allocator` holds only the Result. The fixed-size work buffers share
    // one lifetime and one arena; the solvers use scratch directly so their
    // frees are real.
    var work = std.heap.ArenaAllocator.init(ctx.scratch_allocator);
    defer work.deinit();
    const arena = work.allocator();

    const gamma = opts.gamma orelse opts.t_stop / 1000.0;
    const h_cap = opts.h_output_cap orelse opts.t_stop / 200.0;
    // A Krylov subspace of R^n has at most n dimensions, and the step update
    // reuses the n-long arnoldi_tmp1 as an m-long dense RHS, so m <= n is
    // also a memory-safety bound.
    const m_max = @min(opts.m_max, nn);
    const mm: usize = m_max;
    const h_size = mm * mm;

    const x = try arena.dupe(f64, ctx.x_op[0..n]);
    // One eval fills the G and C planes.
    try ckt.computeBaseline();
    ckt.eval(x, 0);

    // Factor (C + gamma*G) once for the whole run; the extra slot is the
    // ground-stamp trash slot.
    const nnz: usize = ckt.nnz;
    const combined_vals = try arena.alloc(f64, nnz + 1);
    if (ckt.has_charge)
        combinePlanes(combined_vals[0..nnz], ckt.c_vals[0..nnz], ckt.g_vals[0..nnz], gamma)
    else
        num.scale(combined_vals[0..nnz], gamma, ckt.g_vals[0..nnz]);
    combined_vals[nnz] = 0;
    var slv = try Solver.init(ctx.scratch_allocator, nn, ckt.col_ptr, ckt.row_idx, ckt.bbd, ckt.knobs.lu);
    defer slv.deinit();
    try slv.factor(combined_vals, ckt.solver_execution);

    // G alone carries the particular solution: the source terms of eq. 5
    // are A^-1 C^-1 b = -G^-1 b and A^-2 C^-1 b = G^-1 C G^-1 b.
    var slv_g = try Solver.init(ctx.scratch_allocator, nn, ckt.col_ptr, ckt.row_idx, ckt.bbd, ckt.knobs.lu);
    defer slv_g.deinit();
    try slv_g.factor(ckt.g_vals, ckt.solver_execution);

    // The C plane for the Arnoldi matvec, saved because later evals
    // overwrite the planes.
    const c_vals = try arena.alloc(f64, nnz + 1);
    if (ckt.has_charge) @memcpy(c_vals[0..nnz], ckt.c_vals[0..nnz]) else @memset(c_vals[0..nnz], 0);

    const spots = try collectTransitionSpots(arena, ckt, opts.t_stop);
    const basis = try arena.alloc(f64, n * (mm + 1));
    // H is m_max x m_max plus one row: the last non-breakdown Arnoldi
    // iteration stores the subdiagonal at H[m_max*m_max + m_max-1], which
    // nothing reads.
    const h_mat = try arena.alloc(f64, h_size + mm);
    // tmp1 doubles as posteriorOk's 7*m*m expm scratch.
    const tmp1 = try arena.alloc(f64, @max(n, 7 * h_size));
    const tmp2 = try arena.alloc(f64, n);
    const piv = try arena.alloc(u32, mm);
    const expm_scratch = try arena.alloc(f64, @max(5 * h_size, 1));
    const expm_out = try arena.alloc(f64, h_size);
    // Plus m_max for u = H_m^-1 e1 past the m x m block.
    const h_copy = try arena.alloc(f64, h_size + mm);
    var b_t = try arena.alloc(f64, n); // b(t)
    var b_th = try arena.alloc(f64, n); // b(t+h)
    var ainv_bt = try arena.alloc(f64, n); // A^-1 b(t)
    var ainv_bth = try arena.alloc(f64, n); // A^-1 b(t+h)
    const ainv2_db = try arena.alloc(f64, n); // A^-2 (b(t+h) - b(t))/h
    const db = try arena.alloc(f64, n); // (b(t+h) - b(t))/h
    const pwl_tmp = try arena.alloc(f64, n);
    const v_vec = try arena.alloc(f64, n); // the vector to exponentiate
    const x_new = try arena.alloc(f64, n);

    // Recorded straight into the results arena; the Result borrows it.
    var wf = try Waveform.init(ctx.allocator, @intCast(ctx.probes.len), estPoints(opts));
    errdefer wf.deinit();
    try wf.record(0, x, ctx.probes);

    var t: f64 = 0;
    var spot: usize = 0;
    var steps: u32 = 0;
    while (t < opts.t_stop and steps < steps_max) : (steps += 1) {
        if (steps != 0) try ckt.checkpoint(.{ .phase = .transient, .completed = steps });
        // h: the distance to the next transition spot, capped.
        var h = opts.t_stop - t;
        while (spot < spots.len and spots[spot] <= t + spot_merge_s) : (spot += 1) {}
        if (spot < spots.len) h = @min(h, spots[spot] - t);
        h = @min(h, h_cap);
        if (h < step_min_s) break;
        if (t + h > opts.t_stop) h = opts.t_stop - t;

        // After the first step b(t) and A^-1 b(t) are last step's b(t+h)
        // terms, swapped in below: evalSourceRhs is a pure function of t.
        if (steps == 0) {
            evalSourceRhs(ckt, t, b_t);
            negGinvMul(&slv_g, b_t, ainv_bt);
        }
        evalSourceRhs(ckt, t + h, b_th);

        // MATEX eq. 5, the exact update for piecewise-linear sources:
        //   x(t+h) = expm(hA) [x(t) + A^-1 b(t) + A^-2 db]
        //          - A^-1 b(t+h) - A^-2 db,     db = (b(t+h) - b(t))/h
        const inv_h = 1.0 / h;
        for (db, b_t, b_th) |*d, bt, bth| d.* = inv_h * (bth - bt);
        negGinvMul(&slv_g, b_th, ainv_bth);
        // A^-2 db = -G^-1 C (-G^-1 db); db is free scratch after this.
        negGinvMul(&slv_g, db, pwl_tmp);
        cscMulVec(nn, ckt.col_ptr, ckt.row_idx, c_vals, pwl_tmp, db);
        negGinvMul(&slv_g, db, ainv2_db);
        for (v_vec, x, ainv_bt, ainv2_db) |*v, xv, a1, a2| v.* = xv + a1 + a2;

        // expm(hA) z = g(S) S z for S = (C + gamma*G)^-1 C, the
        // shift-inverted operator, and g(mu) = exp((h/gamma)(1 - 1/mu)) / mu.
        // Starting the Krylov space at S z drops the algebraic (mu = 0)
        // components, which expm(hA) sends to zero, so H_m stays regular.
        cscMulVec(nn, ckt.col_ptr, ckt.row_idx, c_vals, v_vec, pwl_tmp);
        slv.solve(pwl_tmp, db);
        const ar = arnoldi(nn, db, &slv, ckt.col_ptr, ckt.row_idx, c_vals, basis, h_mat, m_max, h, opts.krylov_tol, tmp1, tmp2, piv);
        if (!ar.converged) return error.MatexDidNotConverge;
        if (ar.m == 0) {
            // Zero vector: only the correction terms remain.
            num.zeroSimd(x_new);
        } else {
            const m: usize = ar.m;
            const msq = m * m;
            for (0..m) |i| num.copySimd(h_copy[i * m ..][0..m], h_mat[i * mm ..][0..m]);
            // H_m approximates (C + gamma*G)^-1 C, so hA maps to
            // T = (h/gamma)(I - H_m^-1) and expm(hA) v ~ beta V_m expm(T) e1.
            // ponytail: inverts H_m by dense LU; fine at Krylov sizes.
            const h_inv = expm_out[0..msq];
            // Singular only for a DAE of index > 1, which this path cannot
            // integrate.
            try dense_lu.factorize(m, h_copy[0..msq], piv[0..m]);
            for (0..m) |j| {
                num.zeroSimd(tmp1[0..m]);
                tmp1[j] = 1.0;
                dense_lu.solveFactored(m, h_copy[0..msq], piv[0..m], tmp1[0..m], tmp1[0..m]);
                for (0..m) |i| h_inv[i * m + j] = tmp1[i];
            }
            // u = H_m^-1 e1, kept for g(H_m) e1 = expm(T) u: expmSmall
            // overwrites h_inv.
            const u = h_copy[msq..][0..m];
            for (u, 0..) |*ui, i| ui.* = h_inv[i * m];
            const scale = h / gamma;
            num.scale(h_copy[0..msq], -scale, h_inv);
            for (0..m) |i| h_copy[i * m + i] += scale;
            if (!expmSmall(m, h_copy, expm_out, expm_scratch, piv)) return error.MatexDidNotConverge;
            // x_new = beta * V_m * expm(T) * u
            num.zeroSimd(x_new);
            for (0..m) |k| num.axpy(x_new, ar.beta * num.dot(expm_out[k * m ..][0..m], u), basis[k * n ..][0..n]);
        }
        // Subtract the correction terms A^-1 b(t+h) + A^-2 db.
        for (x_new, ainv_bth, ainv2_db) |*xv, a1, a2| xv.* = xv.* - a1 - a2;
        // A non-finite step is a failed step, not a waveform (#51); NaN and
        // inf propagate through the sum.
        if (!std.math.isFinite(num.sum(x_new))) return error.MatexDidNotConverge;
        num.copySimd(x, x_new);
        t += h;
        std.mem.swap([]f64, &b_t, &b_th);
        std.mem.swap([]f64, &ainv_bt, &ainv_bth);
        try wf.record(t, x, ctx.probes);
    }
    assert(wf.len == steps + 1);

    return .{
        .plotname = "MATEX Transient Analysis",
        .varnames = try root.probeNames(ctx, "time"),
        .is_complex = false,
        .npoints = wf.len,
        .data = wf.data(),
    };
}

// ---- private ----

/// Source breakpoints in (0, t_stop], sorted, with t_stop appended and
/// points closer than `spot_merge_s` merged. Allocated on `arena`.
fn collectTransitionSpots(arena: std.mem.Allocator, ckt: *root.Circuit, t_stop: f64) ![]f64 {
    assert(t_stop > 0);
    var spots: std.ArrayList(f64) = .empty;
    defer spots.deinit(arena);
    var t: f64 = 0;
    var iters: u32 = 0;
    while (t < t_stop and iters < spots_max) : (iters += 1) {
        const bp = ckt.nextBreakpoint(t) orelse break;
        if (bp > t_stop) break;
        try spots.append(arena, bp);
        t = bp;
    }
    try spots.append(arena, t_stop);

    const items = try spots.toOwnedSlice(arena);
    std.mem.sort(f64, items, {}, std.sort.asc(f64));
    var write: usize = 0;
    for (items, 0..) |v, i| {
        if (i == 0 or v - items[write - 1] > spot_merge_s) {
            items[write] = v;
            write += 1;
        }
    }
    assert(write >= 1);
    return items[0..write];
}

/// b(t) = B*u(t) for C x' + G x = b: the negated rhs plane (the residual
/// current I(x) = G x - b) of an eval at x = 0, so only sources contribute.
/// Sets the transient phase, since a SIN/PULSE/PWL card only follows its
/// waveform under `analysis("tran")` (§4.6.1).
fn evalSourceRhs(ckt: *root.Circuit, t: f64, b_out: []f64) void {
    assert(b_out.len == ckt.n);
    // b_out doubles as the zero state vector; the eval result overwrites it.
    num.zeroSimd(b_out);
    ckt.setSimState(.{ .t = t, .kind = .tran });
    ckt.eval(b_out, t);
    num.scale(b_out, -1.0, ckt.rhs[0..b_out.len]);
}

/// dest = -G^-1 v, which is A^-1 (C^-1 v) for A = -C^-1 G. C^-1 never
/// appears, so a singular C (algebraic MNA rows) is fine; G must be regular.
/// `v` and `dest` must not alias.
fn negGinvMul(slv_g: *Solver, v: []const f64, dest: []f64) void {
    assert(v.len == dest.len);
    assert(v.ptr != dest.ptr);
    slv_g.solve(v, dest);
    num.scale(dest, -1.0, dest);
}

/// y = M*x for the n x n CSC matrix M; zero columns of x are skipped.
fn cscMulVec(n: u32, col_ptr: []const u32, row_idx: []const u32, vals: []const f64, x: []const f64, y: []f64) void {
    assert(col_ptr.len > n);
    assert(x.len >= n);
    num.zeroSimd(y[0..n]);
    for (0..n) |j| {
        const xj = x[j];
        if (xj == 0) continue;
        for (col_ptr[j]..col_ptr[j + 1]) |p| y[row_idx[p]] += vals[p] * xj;
    }
}

/// Arnoldi on (C + gamma*G)^-1 C from v, with modified Gram-Schmidt. Each
/// matvec is a CSC product with C and one solve against the prefactored
/// `slv`. Fills `basis` (n x (m_max+1), column-major) and H (row-major,
/// stride m_max, with one extra row for the last subdiagonal). Stops on
/// breakdown, when `posteriorOk` accepts, or at m_max.
fn arnoldi(
    n: u32,
    v: []const f64,
    slv: *Solver,
    col_ptr: []const u32,
    row_idx: []const u32,
    c_vals: []const f64,
    basis: []f64,
    H: []f64,
    m_max: u32,
    h_step: f64,
    krylov_tol: f64,
    // tmp1: max(n, 7*m_max^2), since the posterior reuses it; tmp2: n.
    tmp1: []f64,
    tmp2: []f64,
    piv: []u32,
) ArnoldiResult {
    const nn: usize = n;
    assert(basis.len >= nn * (@as(usize, m_max) + 1));
    assert(H.len >= @as(usize, m_max) * m_max + m_max);
    const beta = @sqrt(num.dot(v[0..nn], v[0..nn]));
    if (beta < 1e-300) return .{ .m = 0, .beta = 0, .converged = true };
    num.scale(basis[0..nn], 1.0 / beta, v[0..nn]);
    num.zeroSimd(H[0 .. @as(usize, m_max) * m_max]);

    var j: u32 = 0;
    while (j < m_max) : (j += 1) {
        const jj: usize = j;
        cscMulVec(n, col_ptr, row_idx, c_vals, basis[jj * nn ..][0..nn], tmp1[0..nn]);
        slv.solve(tmp1[0..nn], tmp2[0..nn]);
        const w_norm = @sqrt(num.dot(tmp2[0..nn], tmp2[0..nn]));
        for (0..jj + 1) |k| {
            const vk = basis[k * nn ..][0..nn];
            const h_kj = num.dot(tmp2[0..nn], vk);
            H[k * m_max + jj] = h_kj;
            num.axpy(tmp2[0..nn], -h_kj, vk);
        }
        const h_next = @sqrt(num.dot(tmp2[0..nn], tmp2[0..nn]));
        // Lucky breakdown: the subspace is invariant. The test is relative,
        // since orthogonalization leaves a residual of round-off size, and a
        // normalized round-off vector would put a spurious near-zero on H_m's
        // diagonal that its inverse blows up to inf (#51: a 1.13e-11 residual
        // on a rank-3 C reached m = 4). sqrt(eps)-scale, as expokit's btol.
        if (h_next <= 1e-8 * w_norm) return .{ .m = j + 1, .beta = beta, .converged = true };
        H[(jj + 1) * m_max + jj] = h_next;
        num.scale(basis[(jj + 1) * nn ..][0..nn], 1.0 / h_next, tmp2[0..nn]);
        // ponytail: the full expm posterior every 5 iterations from m = 5;
        // see posteriorOk for what it approximates.
        if (j >= 3 and (j + 1) % 5 == 0 and posteriorOk(j + 1, m_max, H, h_next, h_step, beta, krylov_tol, tmp1, piv))
            return .{ .m = j + 1, .beta = beta, .converged = true };
    }
    // A space as large as the system is the exact one.
    return .{ .m = m_max, .beta = beta, .converged = m_max >= n };
}

/// Accepts the m-dimensional Krylov approximation when
/// beta*h_{m+1,m}*|expm(h_step*H_m)[m-1, 0]| < tol*beta. `scratch` needs
/// 7*m*m (false otherwise), `piv` m.
/// ponytail: takes expm of the shift-inverted H directly instead of mapping
/// it back to A-space first. Conservative (may use a few extra vectors);
/// switch to the exact R-MATEX posterior if m keeps hitting m_max.
fn posteriorOk(m: u32, m_max: u32, H: []const f64, h_next: f64, h_step: f64, beta: f64, tol: f64, scratch: []f64, piv: []u32) bool {
    assert(m <= m_max);
    assert(piv.len >= m);
    const mm: usize = m;
    const msq = mm * mm;
    if (scratch.len < 7 * msq) return false;
    const h_copy = scratch[0..msq];
    const expm_out = scratch[msq .. 2 * msq];
    const expm_scratch = scratch[2 * msq .. 7 * msq];
    // The leading m x m block of the m_max-strided H.
    for (0..mm) |i| num.copySimd(h_copy[i * mm ..][0..mm], H[i * @as(usize, m_max) ..][0..mm]);
    num.scale(h_copy, h_step, h_copy);
    if (!expmSmall(mm, h_copy, expm_out, expm_scratch, piv)) return false;
    return beta * h_next * @abs(expm_out[(mm - 1) * mm]) < tol * beta;
}

/// expm(H) for m x m row-major H by scaling and squaring with Pade(6,6):
/// U = H*(b1*I + b3*H^2 + b5*H^4), V = b0*I + b2*H^2 + b4*H^4 + b6*H^6,
/// expm = (V - U)^-1 (V + U); four matmuls and one dense LU. Writes `out`,
/// destroys H. `scratch` holds at least 5*m*m, `piv_buf` at least m. A
/// singular V - U yields the identity. Returns false, writing nothing, when
/// H holds an inf: no scaling brings it under 0.5 (#125). A NaN entry passes
/// through to `out`, where the callers' finiteness checks see it.
fn expmSmall(m: usize, H: []f64, out: []f64, scratch: []f64, piv_buf: []u32) bool {
    const mm = m * m;
    assert(H.len >= mm);
    assert(out.len >= mm);
    assert(scratch.len >= 5 * mm);
    assert(piv_buf.len >= m);
    // b_j = (12-j)! 6! / (12! j! (6-j)!)
    const b = [_]f64{ 1.0, 0.5, 5.0 / 44.0, 1.0 / 66.0, 1.0 / 792.0, 1.0 / 15840.0, 1.0 / 665280.0 };

    // Scale until max |H_ij| <= 0.5.
    // ponytail: the max element suffices for the scaling decision.
    var norm_h = num.normInf(H[0..mm]);
    if (!std.math.isFinite(norm_h)) return false;
    var s: u32 = 0;
    while (norm_h > 0.5) : (s += 1) {
        assert(s < halvings_max);
        norm_h *= 0.5;
        num.scale(H[0..mm], 0.5, H[0..mm]);
    }

    const h2 = scratch[0..mm];
    const h4 = scratch[mm .. 2 * mm];
    const t = scratch[2 * mm .. 3 * mm];
    const u = scratch[3 * mm .. 4 * mm];
    const v = scratch[4 * mm .. 5 * mm];
    denseMatMul(m, H, H, h2);
    denseMatMul(m, h2, h2, h4);
    denseMatMul(m, h4, h2, t); // H^6
    // Even part.
    for (v, h2, h4, t) |*o, x2, x4, x6| o.* = b[2] * x2 + b[4] * x4 + b[6] * x6;
    for (0..m) |i| v[i * m + i] += b[0];
    // Odd part: t becomes the inner polynomial, then u = H*t.
    for (t, h2, h4) |*o, x2, x4| o.* = b[3] * x2 + b[5] * x4;
    for (0..m) |i| t[i * m + i] += b[1];
    denseMatMul(m, H, t, u);
    // out = V + U, H = V - U.
    for (out[0..mm], H[0..mm], u, v) |*o, *h, ui, vi| {
        o.* = vi + ui;
        h.* = vi - ui;
    }

    const piv = piv_buf[0..m];
    dense_lu.factorize(m, H, piv) catch {
        @memset(out[0..mm], 0);
        for (0..m) |i| out[i * m + i] = 1;
        return true;
    };
    // Solve (V - U) X = (V + U) one column at a time, in place in `out`.
    const col = h2;
    for (0..m) |j| {
        for (0..m) |i| col[i] = out[i * m + j];
        dense_lu.solveFactored(m, H, piv, col[0..m], col[0..m]);
        for (0..m) |i| out[i * m + j] = col[i];
    }
    // Undo the scaling: out = out^(2^s).
    for (0..s) |_| {
        num.copySimd(h2, out[0..mm]);
        denseMatMul(m, h2, h2, out);
    }
    return true;
}

/// C = A*B for m x m row-major matrices; C must not alias A or B.
/// Vectorized over W adjacent output columns, each lane summing k in
/// 0..m order, so every element rounds exactly as the scalar loop would.
/// The lanes are output columns, not one loop's iterations: the plain i-k-j
/// loop vectorizes too but measured 1.1-2.7x slower (m = 30..5).
fn denseMatMul(m: usize, A: []const f64, B: []const f64, C: []f64) void {
    assert(A.len >= m * m);
    assert(B.len >= m * m);
    assert(C.len >= m * m);
    for (0..m) |i| {
        const a_row = A[i * m ..][0..m];
        const c_row = C[i * m ..][0..m];
        var j: usize = 0;
        while (j + W <= m) : (j += W) {
            var acc: V = @splat(@as(f64, 0));
            for (0..m) |k| acc += @as(V, @splat(a_row[k])) * @as(V, B[k * m + j ..][0..W].*);
            c_row[j..][0..W].* = acc;
        }
        while (j < m) : (j += 1) {
            var sum: f64 = 0;
            for (0..m) |k| sum += a_row[k] * B[k * m + j];
            c_row[j] = sum;
        }
    }
}
