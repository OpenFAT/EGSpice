//! The `.disto` Volterra kernels: the second-order kernel as its nonzero
//! terms (a central difference of the analytic Jacobians, two evals per
//! unknown), its complex bilinear form, and the third-order contractions
//! d3(V1,V1,V1) and d3(V1,V1,V2*) as directional second differences of the
//! Jacobians, never a stored tensor.
//! In: a circuit at its operating point and phasors. Out: the nonlinear
//! currents the harmonic solves of `disto.zig` take as right-hand sides.
//! Cites ngspice dkerproc.c (:43-52, the harmonic scale).
//!
//! Contents:
//! - `pub const cubic_step: f64`
//! - `pub const Term = struct { row, a, b, coeff, qcoeff }`
//! - `pub const Kernels = struct` (`terms`, `row_start`, `build`, `deinit`)
//! - `pub fn bilinear(terms, row_start, a_re, a_im, b_re, b_im, scale: f64, omega: f64, dst: []f64) void`
//! - `pub fn scatter(dst: []f64, k: usize, probes: []const u32, x: []const f64) void`
//! - `pub fn mixedCubic(ckt, x_op, p, q, c_re, c_im, h, omega, gc0, g_work, g_tap, planes, x_work, dst) void`
//! - `pub fn cubicForms(ckt, x_op, p, q, h, omega, gc0, g_work, g_minus, x_work, cubic) [2][]const f64`

const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;
const num = @import("core").numerics;
const root = @import("../types.zig");

/// Step of the third-order kernel's second difference, along a unit
/// direction (volts, for the node rows that dominate V1). It is not
/// `fd_eps`: a second difference loses ε·|G|/h² to roundoff, and V1 is often
/// dominated by a nearly linear output node, so the controlling voltage of
/// the nonlinearity moves only a fraction of h. At h = 1e-6 that left the
/// BJT common-emitter HD3 as roundoff, moving 1.5% for a 1e-5 K temperature
/// change (disto/bench_disto_bjt_ce). The fourth-order stencil keeps the
/// truncation at (h/Vt)^4/90, 2.4e-8 relative for an exponential at 1e-3 V;
/// that deck's HD3 is flat to four digits for h from 1e-4 to 1e-2.
pub const cubic_step: f64 = 1e-3;

/// One nonzero of the second-order kernels: d²F[row]/dx_a dx_b (`coeff`)
/// and d²Q[row]/dx_a dx_b (`qcoeff`).
/// `bilinear` reads 24 of its 32 bytes (`row` only orders the terms).
pub const Term = struct { row: u32, a: u32, b: u32, coeff: f64, qcoeff: f64 };

comptime {
    assert(@sizeOf(Term) == 32);
}

/// The second-order kernels d2F/dxa dxb = dG[.,a]/dx_b and d2Q = dC/dx as
/// their nonzero terms, row by row in (a, b) order. Invariant: `row_start`
/// has n + 1 ascending offsets into `terms`.
pub const Kernels = struct {
    terms: []Term,
    row_start: []u32,

    /// d2[row][a][b] = (G(x_op + eps e_b) - G(x_op - eps e_b))[row][a] / 2eps,
    /// and its charge twin from C, kept as the nonzero terms only. G and C
    /// share one pattern and both planes are 0 outside it, so only pattern
    /// slots can contribute. Central, not forward: the forward difference's
    /// O(eps d3) error was the whole 2nd-harmonic residual of
    /// disto/bench_disto_diode_clipper. Leaves the planes at `x_op`. Free
    /// with `deinit` and the same allocator. Asserts that `x_op` holds `n`.
    pub fn build(ckt: *root.Circuit, x_op: []const f64, eps: f64, gpa: Allocator) !Kernels {
        const n: usize = ckt.n;
        assert(x_op.len >= n);
        assert(eps > 0);
        const inv_2eps = 0.5 / eps;
        const nnz: usize = ckt.nnz;
        const gc_plus = try gpa.alloc(f64, 2 * nnz);
        defer gpa.free(gc_plus);

        const g_plus = gc_plus[0..nnz];
        const c_plus = gc_plus[nnz..];
        const x_pert = try gpa.alloc(f64, n);
        defer gpa.free(x_pert);

        var term_list: std.ArrayList(Term) = .empty;
        errdefer term_list.deinit(gpa);

        for (0..n) |b| {
            if (b != 0) try ckt.checkpoint(.{ .phase = .prepare, .completed = b, .total = n });
            @memcpy(x_pert, x_op[0..n]);
            x_pert[b] += eps;
            ckt.eval(x_pert, 0);
            @memcpy(g_plus, ckt.g_vals[0..nnz]);
            @memcpy(c_plus, ckt.c_vals[0..nnz]);
            x_pert[b] = x_op[b] - eps;
            ckt.eval(x_pert, 0);
            for (0..n) |a| for (ckt.col_ptr[a]..ckt.col_ptr[a + 1]) |slot| {
                // (-0 compares equal to 0 and is skipped; NaN is kept.)
                const coeff = (g_plus[slot] - ckt.g_vals[slot]) * inv_2eps;
                const qcoeff = (c_plus[slot] - ckt.c_vals[slot]) * inv_2eps;
                if (coeff == 0 and qcoeff == 0) continue;
                try term_list.append(gpa, .{ .row = ckt.row_idx[slot], .a = @intCast(a), .b = @intCast(b), .coeff = coeff, .qcoeff = qcoeff });
            };
        }
        ckt.eval(x_op, 0);
        // Row by row in (a, b) order: the per-frequency contractions add
        // exactly the terms a dense d2 sweep with a zero skip would, in the
        // same order, so the sums are bitwise the same. The sort is stable
        // and b was appended in ascending order.
        const terms = try term_list.toOwnedSlice(gpa);
        errdefer gpa.free(terms);

        std.sort.block(Term, terms, {}, rowThenA);
        const row_start = try gpa.alloc(u32, n + 1);
        var t: u32 = 0;
        for (0..n) |row| {
            row_start[row] = t;
            while (t < terms.len and terms[t].row == row) t += 1;
        }
        row_start[n] = t;
        assert(t == terms.len);
        return .{ .terms = terms, .row_start = row_start };
    }

    /// Frees both tables; `gpa` is the allocator `build` took.
    pub fn deinit(self: *Kernels, gpa: Allocator) void {
        gpa.free(self.terms);
        gpa.free(self.row_start);
    }
};

/// The harmonic plots report the sinusoid amplitude, twice the one-sided
/// phasor the Volterra recursion solves for (DkerProc, dkerproc.c:43-52).
/// The summary plot does not apply it.
const harmonic_scale: f64 = 2.0;

/// dst (stacked 2n) := scale·(F''(a, b) + jω·Q''(a, b)), the complex
/// bilinear form of the second-order kernels `terms` (rows `row_start`) on
/// phasors a and b given as real and imaginary parts, for a product at ω.
pub fn bilinear(terms: []const Term, row_start: []const u32, a_re: []const f64, a_im: []const f64, b_re: []const f64, b_im: []const f64, scale: f64, omega: f64, dst: []f64) void {
    const n = a_re.len;
    assert(row_start.len == n + 1);
    assert(dst.len >= 2 * n);
    for (0..n) |row| {
        var re: f64 = 0;
        var im: f64 = 0;
        var q_re: f64 = 0;
        var q_im: f64 = 0;
        for (terms[row_start[row]..row_start[row + 1]]) |t| {
            const p_re = a_re[t.a] * b_re[t.b] - a_im[t.a] * b_im[t.b];
            const p_im = a_re[t.a] * b_im[t.b] + a_im[t.a] * b_re[t.b];
            re += t.coeff * p_re;
            im += t.coeff * p_im;
            q_re += t.qcoeff * p_re;
            q_im += t.qcoeff * p_im;
        }
        dst[row] = scale * (re - omega * q_im);
        dst[n + row] = scale * (im + omega * q_re);
    }
}

/// Point k's row of a harmonic plot (`Out.h2` layout): the stacked 2n
/// solution `x` at every probe, as a sinusoid amplitude.
pub fn scatter(dst: []f64, k: usize, probes: []const u32, x: []const f64) void {
    const n = x.len / 2;
    assert(dst.len >= (k + 1) * probes.len * 2);
    for (probes, 0..) |row, i| {
        dst[(k * probes.len + i) * 2] = harmonic_scale * x[row];
        dst[(k * probes.len + i) * 2 + 1] = harmonic_scale * x[n + row];
    }
}

/// `dst` (stacked 2n) := K'''(a, a, c) = F'''(a, a, c) + jω·Q'''(a, a, c)
/// for complex a = p + jq and c, without forming d3. With S(u) the second
/// difference of the Jacobians along u (`secondDirDeriv`, scaled back from a
/// unit direction), for each of F and Q
///   d3(a, a, c) = (S(p) - S(q))·c + j·(S(p+q) - S(p) - S(q))·c,
/// the second term being 2·d3(p, q, ·) by polarization. Twelve evals.
/// `gc0` is the G/C pair at x_op (2n·n), `planes` 4n·n scratch,
/// `g_work`/`g_tap` 2n·n, `x_work` n.
pub fn mixedCubic(
    ckt: *root.Circuit,
    x_op: []const f64,
    p: []const f64,
    q: []const f64,
    c_re: []const f64,
    c_im: []const f64,
    h: f64,
    omega: f64,
    gc0: []const f64,
    g_work: []f64,
    g_tap: []f64,
    planes: []f64,
    x_work: []f64,
    dst: []f64,
) void {
    const n = p.len;
    const n2 = n * n;
    assert(planes.len == 4 * n2);
    assert(dst.len >= 2 * n);
    const m1 = planes[0 .. 2 * n2];
    const m2 = planes[2 * n2 ..][0 .. 2 * n2];
    @memset(planes, 0);
    // (direction, weight into m1, weight into m2); p + q goes through x_work's tail.
    const sum = dst[0..n];
    for (sum, p, q) |*s, a, b| s.* = a + b;
    const dirs = [_]struct { []const f64, f64, f64 }{ .{ p, 1, -1 }, .{ q, -1, -1 }, .{ sum, 0, 1 } };
    for (dirs) |d| {
        const nrm = norm(d[0]);
        if (nrm == 0) continue;
        secondDirDeriv(ckt, x_op, d[0], 1.0 / nrm, h, gc0, g_work, g_tap, x_work);
        // a + (w·s2)·g, the rounding the per-element loop had.
        const s2 = nrm * nrm;
        num.axpy(m1, d[1] * s2, g_work);
        num.axpy(m2, d[2] * s2, g_work);
    }
    // Per plane Re = m1·c_re - m2·c_im, Im = m1·c_im + m2·c_re; then F + jω·Q.
    for (0..n) |row| {
        var acc: [2][2]f64 = .{ .{ 0, 0 }, .{ 0, 0 } };
        for (&acc, 0..) |*plane, k| {
            const off = k * n2 + row * n;
            for (m1[off..][0..n], m2[off..][0..n], c_re, c_im) |a, b, cr, ci| {
                plane[0] += a * cr - b * ci;
                plane[1] += a * ci + b * cr;
            }
        }
        dst[row] = acc[0][0] - omega * acc[1][1];
        dst[n + row] = acc[0][1] + omega * acc[1][0];
    }
}

/// `K'''(V1, V1, V1)` at ω, F''' + jω·Q''', without forming d3, returned as
/// `.{ re, im }`, each an n-vector aliasing `cubic`.
///
/// It is a directional second difference of the analytic Jacobians: for a
/// unit direction u, with G(k) = G(x + k·h·u),
///   S(u)[row,a] = (16(G(1) + G(−1)) − (G(2) + G(−2)) − 30G(0))[row,a] / 12h²
///               = d3[row,a,·,·](u,u) + O(h⁴)
/// and the cubic form T(w,u,u)[row] = Σ_a w[a]·S(u)[row,a] by symmetry of d3,
/// likewise for C. With V1 = p + jq that is eight evals, S(p̂) and S(q̂), and
/// per plane
///   A = T(p,p,p) − 3T(p,q,q),  B = 3T(p,p,q) − T(q,q,q),
/// so F'''(V1,V1,V1) = A_F + jB_F and the charge part adds jω(A_Q + jB_Q).
/// `gc0` is the G/C pair at x_op (2n·n), `g_work`/`g_minus` 2n·n scratch,
/// `x_work` n scratch, `cubic` 8n.
pub fn cubicForms(
    ckt: *root.Circuit,
    x_op: []const f64,
    p: []const f64,
    q: []const f64,
    h: f64,
    omega: f64,
    gc0: []const f64,
    g_work: []f64,
    g_minus: []f64,
    x_work: []f64,
    cubic: []f64,
) [2][]const f64 {
    const n = p.len;
    const n2 = n * n;
    assert(cubic.len == 8 * n);
    assert(q.len == n);
    @memset(cubic, 0);

    // cubic[k·4n ..] holds plane k's T(p,p,p), T(p,q,q), T(p,p,q), T(q,q,q).
    const norm_p = norm(p);
    const norm_q = norm(q);
    if (norm_p > 0) {
        secondDirDeriv(ckt, x_op, p, 1.0 / norm_p, h, gc0, g_work, g_minus, x_work);
        for (0..2) |k| {
            const t = cubic[k * 4 * n ..];
            contract(g_work[k * n2 ..][0..n2], p, 1.0 / norm_p, t[0..n]);
            if (norm_q > 0) contract(g_work[k * n2 ..][0..n2], q, 1.0 / norm_q, t[2 * n ..][0..n]);
        }
    }
    if (norm_q > 0) {
        secondDirDeriv(ckt, x_op, q, 1.0 / norm_q, h, gc0, g_work, g_minus, x_work);
        for (0..2) |k| {
            const t = cubic[k * 4 * n ..];
            contract(g_work[k * n2 ..][0..n2], q, 1.0 / norm_q, t[3 * n ..][0..n]);
            if (norm_p > 0) contract(g_work[k * n2 ..][0..n2], p, 1.0 / norm_p, t[n..][0..n]);
        }
    }

    // Undo the unit-direction normalization: every form is cubic in its inputs.
    const ppp = norm_p * norm_p * norm_p;
    const pqq = norm_p * norm_q * norm_q;
    const ppq = norm_p * norm_p * norm_q;
    const qqq = norm_q * norm_q * norm_q;
    for (0..n) |row| {
        var ab: [2][2]f64 = undefined;
        for (&ab, 0..) |*plane, k| {
            const t = cubic[k * 4 * n ..];
            plane.* = .{
                ppp * t[row] - 3.0 * pqq * t[n + row],
                3.0 * ppq * t[2 * n + row] - qqq * t[3 * n + row],
            };
        }
        cubic[row] = ab[0][0] - omega * ab[1][1];
        cubic[n + row] = ab[0][1] + omega * ab[1][0];
    }
    return .{ cubic[0..n], cubic[n .. 2 * n] };
}

// ---- private ----

fn rowThenA(_: void, x: Term, y: Term) bool {
    return x.row < y.row or (x.row == y.row and x.a < y.a);
}

fn norm(v: []const f64) f64 {
    return @sqrt(num.dot(v, v));
}

/// g_out := the second derivative of the G/C pair (2n·n, G first) along
/// `scale·u` at x_op, by the five-point stencil in `cubicForms`. `g0` is the
/// pair at x_op.
fn secondDirDeriv(
    ckt: *root.Circuit,
    x_op: []const f64,
    u: []const f64,
    scale: f64,
    h: f64,
    g0: []const f64,
    g_out: []f64,
    g_tap: []f64,
    x_work: []f64,
) void {
    const n = u.len;
    const n2 = n * n;
    // (k, weight) of the stencil's off-centre points.
    const taps = [_]struct { f64, f64 }{ .{ 1, 16 }, .{ -1, 16 }, .{ 2, -1 }, .{ -2, -1 } };
    num.scale(g_out[0..n2], -30.0, g0[0..n2]);
    @memset(g_out[n2..], 0);
    for (taps) |tap| {
        const step = tap[0] * h * scale;
        @memcpy(x_work, x_op[0..n]);
        num.axpy(x_work[0..n], step, u);
        ckt.eval(x_work, 0);
        ckt.denseG(g_tap[0..n2]);
        ckt.denseC(g_tap[n2..]);
        num.axpy(g_out[0..n2], tap[1], g_tap[0..n2]);
        // C as differences from C(0): exactly zero for a linear capacitor.
        for (g_out[n2..], g_tap[n2..], g0[n2..]) |*out, c, c0| out.* += tap[1] * (c - c0);
    }
    const inv = 1.0 / (12.0 * h * h);
    num.scale(g_out, inv, g_out);
}

/// dst[row] := (Σ_a s[row·n + a] · w[a]) · scale.
fn contract(s: []const f64, w: []const f64, scale: f64, dst: []f64) void {
    for (dst, 0..) |*d, row| d.* = num.dot(s[row * w.len ..][0..w.len], w) * scale;
}

test "bilinear: complex form matches a dense per-term oracle" {
    // Two rows over n = 2; row 1 has no terms and must come out zero.
    const terms = [_]Term{
        .{ .row = 0, .a = 0, .b = 1, .coeff = 2, .qcoeff = 0.5 },
        .{ .row = 0, .a = 1, .b = 1, .coeff = -3, .qcoeff = 0 },
    };
    const row_start = [_]u32{ 0, 2, 2 };
    const a_re = [_]f64{ 1, 2 };
    const a_im = [_]f64{ 0.5, -1 };
    const b_re = [_]f64{ 3, -2 };
    const b_im = [_]f64{ 1, 0.25 };
    var dst: [4]f64 = undefined;
    bilinear(&terms, &row_start, &a_re, &a_im, &b_re, &b_im, -0.5, 10, &dst);

    const C = std.math.Complex(f64);
    var f = C.init(0, 0);
    var q = C.init(0, 0);
    for (terms) |t| {
        const p = C.init(a_re[t.a], a_im[t.a]).mul(C.init(b_re[t.b], b_im[t.b]));
        f = f.add(p.mul(C.init(t.coeff, 0)));
        q = q.add(p.mul(C.init(t.qcoeff, 0)));
    }
    const want = f.add(C.init(0, 10).mul(q)).mul(C.init(-0.5, 0));
    try std.testing.expectApproxEqRel(want.re, dst[0], 1e-15);
    try std.testing.expectApproxEqRel(want.im, dst[2], 1e-15);
    try std.testing.expectEqual(@as(f64, 0), dst[1]);
    try std.testing.expectEqual(@as(f64, 0), dst[3]);
}

test "contract: row-major matrix-vector product, then scaled" {
    const s = [_]f64{ 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    const w = [_]f64{ 1, 0, -1 };
    var dst: [3]f64 = undefined;
    contract(&s, &w, 2, &dst);
    try std.testing.expectEqualSlices(f64, &.{ -4, -4, -4 }, &dst);
}

test "scatter: probes land point-major with (re, im) adjacent, doubled" {
    var dst: [8]f64 = @splat(0);
    const x = [_]f64{ 1, 2, 3, 10, 20, 30 }; // re | im, n = 3
    scatter(&dst, 1, &.{ 2, 0 }, &x);
    try std.testing.expectEqualSlices(f64, &.{ 0, 0, 0, 0, 6, 60, 2, 20 }, &dst);
}
