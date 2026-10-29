//! Per-frequency network math for `.sp` and `.lin` on small dense complex
//! matrices: the power waves of each port read off an MNA solution, the S
//! matrix of a `.net` port pair, S to Y, Z and H, Rollett's K and
//! Edwards-Sinsky's mu, and the two-port noise parameters.
//! Wave variables (Kurokawa power waves):
//!   a_k = (V_k + z0_k I_k) / (2 sqrt z0_k),  b_k = (V_k - z0_k I_k) / (2 sqrt z0_k)
//! where I_k = -i_br_k (branch stamps F_p = +i_br, so the DUT current is
//! negated). Driving port p with a unit source voltage (rhs[b_p] = 1) gives
//! a_p = 1/(2 sqrt z0_p) and a_j = 0 elsewhere, so column p of S is b_j / a_p.
//! In: stacked-real solutions or a row-major S. Out: S, Y, Z, H, K, mu, noise.
//! Cites HSPICE [SA Ch.17]; Kurokawa power waves; Hillbrand and Russer, IEEE
//! Trans. CAS 23(4), 1976; dev/analysis/s-parameters.md.
//!
//! Contents:
//! - `pub const k_boltzmann`, `pub const t0_kelvin`
//! - `pub const noise_count`, `pub const stability_count`
//! - `pub fn writeColumn(n: usize, ports: []const Port, x: []const f64, s_row: []f64, p: usize) void`
//! - `pub fn netRow(n: usize, ports: []const Port, x: []const f64, s_row: []f64) void`
//! - `pub fn paramCount(n_ports: usize) usize`
//! - `pub fn params(z0: []const f64, pq: [2]usize, p: []Complex, tmp: []Complex) void`
//! - `pub fn stability(s: [4]Complex) [stability_count]f64`
//! - `pub fn noiseParams(s: [4]Complex, z0: [2]f64, cv: [2]f64, c12: Complex) [noise_count]Complex`
//! - `pub fn block(s: []const Complex, n: usize, pq: [2]usize) [4]Complex`
//! - `pub fn readComplex(src: []const f64, dst: []Complex) void`
//! - `pub fn writeComplex(src: []const Complex, dst: []f64) void`

const std = @import("std");
const assert = std.debug.assert;
const testing = std.testing;
const Complex = @import("core").numerics.Complex;
const Port = @import("core").query.Port;
const root = @import("../types.zig");
const at = @import("../dc/xf.zig").at;

/// ngspice CONSTboltz (const.h), the constant the device noise models use.
pub const k_boltzmann = 1.38064852e-23;
/// Noise figure reference temperature, K (IEEE; HSPICE's RN and GN too).
pub const t0_kelvin = 290.0;
/// The two-port noise columns `noiseParams` fills: NFMIN, NF, RN, YOPT, GAMMA_OPT.
pub const noise_count = 5;
/// The two-port stability columns `stability` fills: K and mu.
pub const stability_count = 2;

/// Writes column p of S, driven with a unit source on port p's branch
/// (incident wave 1/(2 sqrt z0_p)), into the stacked (re, im) S matrix
/// `s_row` of one frequency. Asserts that `s_row` holds the matrix.
pub fn writeColumn(n: usize, ports: []const Port, x: []const f64, s_row: []f64, p: usize) void {
    const n_ports = ports.len;
    assert(p < n_ports);
    assert(s_row.len >= 2 * n_ports * n_ports);
    const a_p = 1.0 / (2.0 * @sqrt(ports[p].z0));
    for (ports, 0..) |port, k| {
        const br: usize = port.branch;
        var v_k = at(x, n, port.node);
        if (port.neg != root.GROUND) v_k = v_k.sub(at(x, n, port.neg));
        const i_k = Complex{ .re = -x[br], .im = -x[n + br] };
        const b_k = v_k.sub(i_k.scale(port.z0)).scale(1.0 / (2.0 * @sqrt(port.z0)));
        const s = b_k.scale(1.0 / a_p);
        const idx = (k * n_ports + p) * 2;
        s_row[idx] = s.re;
        s_row[idx + 1] = s.im;
    }
}

/// Writes the S matrix of one `.net` frequency from the ideal-port solutions
/// `x` (one stacked block per port drive). Each drive j gives every port's
/// voltage and current into the network (a branch port's -i_br, otherwise
/// the unit drive or nothing), so V = Vm d and I = Im d, Z = Vm Im^-1, and
/// S = (Zn - E)(Zn + E)^-1 with Zn = F^-1 Z F^-1, F = diag(sqrt z0).
/// Asserts one or two ports.
pub fn netRow(n: usize, ports: []const Port, x: []const f64, s_row: []f64) void {
    const np = ports.len;
    assert(np >= 1);
    assert(np <= 2);
    var vt: [4]Complex = undefined;
    var it: [4]Complex = undefined;
    for (0..np) |j| {
        const xj = x[j * 2 * n ..][0 .. 2 * n];
        for (ports, 0..) |port, k| {
            // Transposed: row j is drive j.
            vt[j * np + k] = at(xj, n, port.node).sub(at(xj, n, port.neg));
            it[j * np + k] = if (port.branch != Port.no_branch)
                .{ .re = -xj[port.branch], .im = -xj[n + port.branch] }
            else
                .{ .re = @floatFromInt(@intFromBool(j == k)), .im = 0 };
        }
    }
    // Im^T Z^T = Vm^T, then (Zn^T + E) S^T = (Zn^T - E).
    solveDense(np, it[0 .. np * np], vt[0 .. np * np]);
    for (0..np) |i| for (0..np) |j| {
        const zn = vt[i * np + j].scale(1 / @sqrt(ports[i].z0 * ports[j].z0));
        const eye: f64 = @floatFromInt(@intFromBool(i == j));
        it[i * np + j] = .{ .re = zn.re + eye, .im = zn.im };
        vt[i * np + j] = .{ .re = zn.re - eye, .im = zn.im };
    };
    solveDense(np, it[0 .. np * np], vt[0 .. np * np]);
    for (0..np) |k| for (0..np) |p| {
        s_row[(k * np + p) * 2] = vt[p * np + k].re;
        s_row[(k * np + p) * 2 + 1] = vt[p * np + k].im;
    };
}

/// Returns the complex entries of S, Y, Z and (two or more ports) H, in that order.
pub fn paramCount(n_ports: usize) usize {
    return 3 * n_ports * n_ports + @as(usize, if (n_ports >= 2) 4 else 0);
}

/// Given S in `p[0..n^2]` (row-major), fills Y, Z and H after it
/// (`paramCount` entries). With F = diag(sqrt z0):
///   Y = F^-1 (E + S)^-1 (E - S) F^-1,   Z = F (E - S)^-1 (E + S) F,
/// and H over the two-port `pq` comes from the Z of its 2x2 S block (the
/// other ports terminated in z0, as HSPICE defines it). A singular E +- S
/// (an open or a short port) gives non-finite entries. Asserts that `p`
/// holds `paramCount` entries and `tmp` n^2.
pub fn params(z0: []const f64, pq: [2]usize, p: []Complex, tmp: []Complex) void {
    const n = z0.len;
    const n2 = n * n;
    assert(p.len >= paramCount(n));
    assert(tmp.len >= n2);
    const s = p[0..n2];
    const y = p[n2..][0..n2];
    const z = p[2 * n2 ..][0..n2];
    const lhs = tmp[0..n2];
    for (0..2) |which| {
        // which 0: (E + S) Y' = (E - S); which 1: (E - S) Z' = (E + S).
        const sign: f64 = if (which == 0) 1 else -1;
        const out = if (which == 0) y else z;
        for (0..n) |i| for (0..n) |j| {
            const eye: f64 = if (i == j) 1 else 0;
            lhs[i * n + j] = .{ .re = eye + sign * s[i * n + j].re, .im = sign * s[i * n + j].im };
            out[i * n + j] = .{ .re = eye - sign * s[i * n + j].re, .im = -sign * s[i * n + j].im };
        };
        solveDense(n, lhs, out);
        for (0..n) |i| for (0..n) |j| {
            const f = @sqrt(z0[i] * z0[j]);
            out[i * n + j] = out[i * n + j].scale(if (which == 0) 1 / f else f);
        };
    }
    if (n < 2) return;
    // Z of the two-port block: (E - S2)^-1 (E + S2), scaled as above.
    const s2 = block(s, n, pq);
    var l2 = [4]Complex{ .{ .re = 1 - s2[0].re, .im = -s2[0].im }, s2[1].scale(-1), s2[2].scale(-1), .{ .re = 1 - s2[3].re, .im = -s2[3].im } };
    var z2 = [4]Complex{ .{ .re = 1 + s2[0].re, .im = s2[0].im }, s2[1], s2[2], .{ .re = 1 + s2[3].re, .im = s2[3].im } };
    solveDense(2, &l2, &z2);
    for (0..2) |i| for (0..2) |j| {
        z2[i * 2 + j] = z2[i * 2 + j].scale(@sqrt(z0[pq[i]] * z0[pq[j]]));
    };
    // V1 = h11 I1 + h12 V2, I2 = h21 I1 + h22 V2.
    const h = p[3 * n2 ..][0..4];
    const inv22 = Complex.div(.{ .re = 1, .im = 0 }, z2[3]);
    h[0] = z2[0].mul(z2[3]).sub(z2[1].mul(z2[2])).mul(inv22);
    h[1] = z2[1].mul(inv22);
    h[2] = z2[2].mul(inv22).scale(-1);
    h[3] = inv22;
}

/// Returns Rollett's K and Edwards-Sinsky's mu of the 2x2 S block `s`
/// (row-major), as HSPICE defines them [SA Ch.17]: with
/// D = S11 S22 - S12 S21,
///   K = (1 - |S11|^2 - |S22|^2 + |D|^2) / (2 |S12 S21|),
///   mu = (1 - |S11|^2) / (|S22 - D S11*| + |S12 S21|).
/// A unilateral block (S12 S21 = 0) gives an infinite K.
pub fn stability(s: [4]Complex) [stability_count]f64 {
    const p = s[1].mul(s[2]);
    const delta = s[0].mul(s[3]).sub(p);
    const k = (1 - s[0].magSq() - s[3].magSq() + delta.magSq()) / (2 * @sqrt(p.magSq()));
    const d = s[3].sub(delta.mul(.{ .re = s[0].re, .im = -s[0].im }));
    const mu = (1 - s[0].magSq()) / (@sqrt(d.magSq()) + @sqrt(p.magSq()));
    return .{ k, mu };
}

/// Returns the two-port noise parameters (NFMIN, NF, RN, YOPT, GAMMA_OPT)
/// from the 2x2 S block `s` (row-major), the port impedances and the port
/// voltage noise correlation (one-sided, V^2/Hz) with both ports terminated
/// in z0: diagonal `cv` and C12 = <v1 v2*>.
///
/// The terminated port voltages are the noise waves times sqrt z0 (a = 0,
/// so b = v / sqrt z0). The short-circuit noise currents are
/// i = -2 F^-1 (E + S)^-1 c (Hillbrand and Russer), and the chain form puts
/// v_n = -i2/Y21 and i_n = i1 - (Y11/Y21) i2 at the input. With S_vv, S_ii
/// and S_iv = <i_n v_n*> (4 k T0 = 4 k 290 K):
///   RN = S_vv / 4kT0,  YOPT = (sqrt(S_ii S_vv - Im^2 S_iv) - j Im S_iv) / S_vv,
///   NFMIN = 1 + (Re S_iv + sqrt(S_ii S_vv - Im^2 S_iv)) / 2kT0,
///   NF = 1 + (S_ii + S_vv |Y_s|^2 + 2 Re(Y_s* S_iv)) / (4kT0 G_s), Y_s = 1/z0_1.
/// NFMIN and NF are power ratios; GAMMA_OPT is YOPT's reflection against
/// port 1's z0.
pub fn noiseParams(s: [4]Complex, z0: [2]f64, cv: [2]f64, c12: Complex) [noise_count]Complex {
    assert(z0[0] > 0);
    assert(z0[1] > 0);
    // G = 2 F^-1 (E + S)^-1, rows scaled by 1/sqrt z0.
    var e_s = [4]Complex{ .{ .re = 1 + s[0].re, .im = s[0].im }, s[1], s[2], .{ .re = 1 + s[3].re, .im = s[3].im } };
    var g = [4]Complex{ .{ .re = 2, .im = 0 }, Complex.zero, Complex.zero, .{ .re = 2, .im = 0 } };
    solveDense(2, &e_s, &g);
    for (0..2) |i| for (0..2) |j| {
        g[i * 2 + j] = g[i * 2 + j].scale(1 / @sqrt(z0[i]));
    };
    // Noise-wave correlation C_S = C_v / sqrt(z0_i z0_j), then C_Y = G C_S G^H.
    const cs = [4]Complex{
        .{ .re = cv[0] / z0[0], .im = 0 },
        c12.scale(1 / @sqrt(z0[0] * z0[1])),
        .{ .re = c12.re / @sqrt(z0[0] * z0[1]), .im = -c12.im / @sqrt(z0[0] * z0[1]) },
        .{ .re = cv[1] / z0[1], .im = 0 },
    };
    // Y of the block: G (E - S) / 2 F^-1 = F^-1 (E + S)^-1 (E - S) F^-1.
    var y: [4]Complex = undefined;
    for (0..2) |i| for (0..2) |j| {
        var acc = Complex.zero;
        for (0..2) |k| {
            const eye: f64 = if (k == j) 1 else 0;
            acc = acc.add(g[i * 2 + k].mul(.{ .re = eye - s[k * 2 + j].re, .im = -s[k * 2 + j].im }));
        }
        y[i * 2 + j] = acc.scale(0.5 / @sqrt(z0[j]));
    };
    // T = [[0, -1/Y21], [1, -Y11/Y21]] maps the Y-form currents to
    // (v_n, i_n); M = T G maps the noise waves to them directly.
    const inv21 = Complex.div(.{ .re = -1, .im = 0 }, y[2]);
    const t = [4]Complex{ Complex.zero, inv21, .{ .re = 1, .im = 0 }, y[0].mul(inv21) };
    var mm: [4]Complex = undefined;
    for (0..2) |i| for (0..2) |j| {
        mm[i * 2 + j] = t[i * 2].mul(g[j]).add(t[i * 2 + 1].mul(g[2 + j]));
    };
    // C_A = M C_S M^H; only S_vv, S_ii and S_iv are needed.
    const v_row = [2]Complex{ mm[0], mm[1] };
    const i_row = [2]Complex{ mm[2], mm[3] };
    const s_vv = quadForm(v_row, cs, v_row).re;
    const s_ii = quadForm(i_row, cs, i_row).re;
    const s_iv = quadForm(i_row, cs, v_row);
    const kt = k_boltzmann * t0_kelvin;
    const root_term = @sqrt(@max(s_ii * s_vv - s_iv.im * s_iv.im, 0));
    const yopt = Complex{ .re = root_term / s_vv, .im = -s_iv.im / s_vv };
    const gs = 1 / z0[0];
    const f_min = 1 + (s_iv.re + root_term) / (2 * kt);
    const f_50 = 1 + (s_ii + s_vv * gs * gs + 2 * gs * s_iv.re) / (4 * kt * gs);
    const zy = yopt.scale(z0[0]);
    const gamma = Complex.div(.{ .re = 1 - zy.re, .im = -zy.im }, .{ .re = 1 + zy.re, .im = zy.im });
    return .{
        .{ .re = f_min, .im = 0 },
        .{ .re = f_50, .im = 0 },
        .{ .re = s_vv / (4 * kt), .im = 0 },
        yopt,
        gamma,
    };
}

/// Returns the 2x2 block of rows and columns `pq` of the n x n row-major `s`.
pub fn block(s: []const Complex, n: usize, pq: [2]usize) [4]Complex {
    assert(pq[0] < n);
    assert(pq[1] < n);
    return .{ s[pq[0] * n + pq[0]], s[pq[0] * n + pq[1]], s[pq[1] * n + pq[0]], s[pq[1] * n + pq[1]] };
}

/// Reads interleaved (re, im) pairs into `dst`. Asserts `src` holds them.
pub fn readComplex(src: []const f64, dst: []Complex) void {
    assert(src.len >= 2 * dst.len);
    for (dst, 0..) |*z, i| z.* = .{ .re = src[2 * i], .im = src[2 * i + 1] };
}

/// Writes `src` as interleaved (re, im) pairs. Asserts `dst` holds them.
pub fn writeComplex(src: []const Complex, dst: []f64) void {
    assert(dst.len >= 2 * src.len);
    for (src, 0..) |z, i| {
        dst[2 * i] = z.re;
        dst[2 * i + 1] = z.im;
    }
}

// ---- private ----

/// Solves A X = B in place (B becomes X) for n x n row-major complex A and
/// B, by Gaussian elimination with partial pivoting. Destroys A.
fn solveDense(n: usize, a: []Complex, b: []Complex) void {
    assert(a.len == n * n);
    assert(b.len == n * n);
    for (0..n) |c| {
        var piv = c;
        for (c + 1..n) |r| if (a[r * n + c].magSq() > a[piv * n + c].magSq()) {
            piv = r;
        };
        if (piv != c) for (0..n) |j| {
            std.mem.swap(Complex, &a[c * n + j], &a[piv * n + j]);
            std.mem.swap(Complex, &b[c * n + j], &b[piv * n + j]);
        };
        const inv = Complex.div(.{ .re = 1, .im = 0 }, a[c * n + c]);
        for (c + 1..n) |r| {
            const f = a[r * n + c].mul(inv);
            for (c..n) |j| a[r * n + j] = a[r * n + j].sub(f.mul(a[c * n + j]));
            for (0..n) |j| b[r * n + j] = b[r * n + j].sub(f.mul(b[c * n + j]));
        }
    }
    var c = n;
    while (c > 0) {
        c -= 1;
        const inv = Complex.div(.{ .re = 1, .im = 0 }, a[c * n + c]);
        for (0..n) |j| {
            var acc = b[c * n + j];
            for (c + 1..n) |k| acc = acc.sub(a[c * n + k].mul(b[k * n + j]));
            b[c * n + j] = acc.mul(inv);
        }
    }
}

/// u C w^H over 2-vectors.
fn quadForm(u: [2]Complex, c: [4]Complex, w: [2]Complex) Complex {
    var acc = Complex.zero;
    for (0..2) |i| for (0..2) |j| {
        acc = acc.add(u[i].mul(c[i * 2 + j]).mul(.{ .re = w[j].re, .im = -w[j].im }));
    };
    return acc;
}

fn expectNear(want: Complex, got: Complex, tol: f64) !void {
    try testing.expectApproxEqAbs(want.re, got.re, tol);
    try testing.expectApproxEqAbs(want.im, got.im, tol);
}

test solveDense {
    // Certificate: A X = B for a dense complex A that needs a row swap
    // (zero leading pivot), X solved for B = I.
    const n = 3;
    const a0 = [n * n]Complex{
        .zero,                  .{ .re = 2, .im = -1 },  .{ .re = 0.5, .im = 0 },
        .{ .re = 1, .im = 1 },  .{ .re = -1, .im = 0 },  .{ .re = 0, .im = 3 },
        .{ .re = 4, .im = -2 }, .{ .re = 0, .im = 0.5 }, .{ .re = 1, .im = 0 },
    };
    var a = a0;
    var x: [n * n]Complex = @splat(.zero);
    for (0..n) |i| x[i * n + i] = .{ .re = 1, .im = 0 };
    solveDense(n, &a, &x);
    for (0..n) |i| for (0..n) |j| {
        var acc = Complex.zero;
        for (0..n) |k| acc = acc.add(a0[i * n + k].mul(x[k * n + j]));
        try expectNear(.{ .re = @floatFromInt(@intFromBool(i == j)), .im = 0 }, acc, 1e-14);
    };
}

test params {
    // Matched ports (S = 0): Y = diag(1/z0), Z = diag(z0), and the two-port
    // H is z0_1 at h11 and 1/z0_2 at h22.
    const z0 = [_]f64{ 50, 75, 25 };
    const np = z0.len;
    var p: [paramCount(np)]Complex = @splat(.zero);
    var tmp: [np * np]Complex = undefined;
    params(&z0, .{ 0, 1 }, &p, &tmp);
    const y = p[np * np ..][0 .. np * np];
    const zm = p[2 * np * np ..][0 .. np * np];
    const h = p[3 * np * np ..][0..4];
    for (0..np) |i| for (0..np) |j| {
        const on: f64 = @floatFromInt(@intFromBool(i == j));
        try expectNear(.{ .re = on / z0[i], .im = 0 }, y[i * np + j], 1e-15);
        try expectNear(.{ .re = on * z0[i], .im = 0 }, zm[i * np + j], 1e-12);
    };
    try expectNear(.{ .re = 50, .im = 0 }, h[0], 1e-12);
    try expectNear(.zero, h[1], 1e-15);
    try expectNear(.zero, h[2], 1e-15);
    try expectNear(.{ .re = 1.0 / 75.0, .im = 0 }, h[3], 1e-15);

    // Y Z = E for a general S (certificate of the two inversions).
    const s = [_]Complex{
        .{ .re = 0.2, .im = 0.1 },   .{ .re = -0.3, .im = 0.05 }, .{ .re = 0.1, .im = 0 },
        .{ .re = 0.4, .im = -0.2 },  .{ .re = 0.1, .im = 0.3 },   .{ .re = 0, .im = -0.1 },
        .{ .re = 0.05, .im = 0.05 }, .{ .re = -0.1, .im = 0 },    .{ .re = 0.3, .im = 0.2 },
    };
    @memcpy(p[0 .. np * np], &s);
    params(&z0, .{ 0, 1 }, &p, &tmp);
    for (0..np) |i| for (0..np) |j| {
        var acc = Complex.zero;
        for (0..np) |k| acc = acc.add(y[i * np + k].mul(zm[k * np + j]));
        try expectNear(.{ .re = @floatFromInt(@intFromBool(i == j)), .im = 0 }, acc, 1e-12);
    };
}

test stability {
    // A matched amplifier (S11 = S22 = 0) has K = (1 + |p|^2)/(2|p|) and
    // mu = 1/|p|, p = S12 S21; a unilateral one has K = inf.
    const s = [4]Complex{ .zero, .{ .re = 0.1, .im = 0 }, .{ .re = 0, .im = 5 }, .zero };
    const ks = stability(s);
    try testing.expectApproxEqRel((1 + 0.25) / (2 * 0.5), ks[0], 1e-15);
    try testing.expectApproxEqRel(1 / 0.5, ks[1], 1e-15);
    const uni = stability(.{ .{ .re = 0.5, .im = 0 }, .zero, .{ .re = 3, .im = 0 }, .{ .re = 0.2, .im = 0 } });
    try testing.expect(std.math.isPositiveInf(uni[0]));
}

test noiseParams {
    // A noiseless two-port: no noise resistance and a 1.0 noise figure.
    const s = [4]Complex{ .{ .re = 0.1, .im = 0.2 }, .{ .re = 0.05, .im = 0 }, .{ .re = 2, .im = -1 }, .{ .re = -0.2, .im = 0.1 } };
    const out = noiseParams(s, .{ 50, 50 }, .{ 0, 0 }, .zero);
    try testing.expectEqual(@as(f64, 1), out[1].re);
    try testing.expectEqual(@as(f64, 0), out[2].re);
}

test netRow {
    // One ideal V port reading 1 V and drawing 10 mA (i_br = -10 mA): Z =
    // 100 ohm against z0 = 50 ohm, so S = (2 - 1)/(2 + 1).
    const n = 3;
    var x: [2 * n]f64 = @splat(0);
    x[1] = 1;
    x[2] = -0.01;
    const ports = [_]Port{.{ .node = 1, .branch = 2, .z0 = 50 }};
    var s_row: [2]f64 = undefined;
    netRow(n, &ports, &x, &s_row);
    try testing.expectApproxEqRel(1.0 / 3.0, s_row[0], 1e-14);
    try testing.expectApproxEqAbs(0, s_row[1], 1e-16);
}

test writeColumn {
    // A port terminated in its own z0 reads V = z0 I, so b = 0: S11 = 0.
    // Driven with 1 V through series z0 into a matched load, V = 0.5 and
    // I = 10 mA (i_br = -10 mA).
    const n = 3;
    var x: [2 * n]f64 = @splat(0);
    x[1] = 0.5;
    x[2] = -0.01;
    const ports = [_]Port{.{ .node = 1, .branch = 2, .z0 = 50 }};
    var s_row: [2]f64 = undefined;
    writeColumn(n, &ports, &x, &s_row, 0);
    try testing.expectApproxEqAbs(0, s_row[0], 1e-15);
    try testing.expectApproxEqAbs(0, s_row[1], 1e-15);
}
