//! Passivity by residue perturbation. In: a `Fit` and the samples it was
//! fitted to. Out: the same fit with its residues and direct terms moved
//! the least (in the fitted response over the data points) so that
//! λ_min(Re Y(jω)) ≥ 0 on a dense grid; and the symmetric eigensolver the
//! W element shares.
//! Cites: Gustavsen and Semlyen, IEEE Trans. Power Systems 16(1), 2001
//! (passivity enforcement by residue perturbation); Jacobi eigenvalue
//! method, Golub and Van Loan §8.5.
//!
//! Contents:
//! - `pub fn enforcePassivity(arena: Allocator, net: Network, f: *Fit) Error!bool`
//! - `pub fn symEig(n: usize, a: []f64, v: []f64) void`

const std = @import("std");
const assert = std.debug.assert;
const testing = std.testing;
const Allocator = std.mem.Allocator;
const core = @import("core");
const eigen = core.eigen;
const numerics = core.numerics;
const sparam_touchstone = @import("sparam_touchstone.zig");
const sparam_fit = @import("sparam_fit.zig");
const sparam_lsq = @import("sparam_lsq.zig");
const Network = sparam_touchstone.Network;
const Error = sparam_touchstone.Error;
const Fit = sparam_fit.Fit;
const Cx = std.math.Complex(f64);

/// Makes `f` passive for a P-port: symmetric direct term positive
/// semidefinite and λ_min(Re Y(jω)) ≥ 0 on a dense grid, by the
/// minimum-norm residue and direct-term change (over the data points) that
/// lifts every violation to a small positive margin. Returns false when
/// 30 rounds leave a violation. `f` must be a fit of `net` (same ports);
/// its residues and direct terms change in place, its poles never. Scratch
/// lives in `arena`.
pub fn enforcePassivity(arena: Allocator, net: Network, f: *Fit) Error!bool {
    const p = net.ports;
    const ne = p * p;
    const kk = sparam_fit.realOrder(f.poles);
    const npar = kk + 1;
    var ymax: f64 = 0;
    for (0..ne) |e| for (0..net.freq.len) |k| {
        ymax = @max(ymax, net.at(e, k).magnitude());
    };
    const margin = 1e-9 * ymax;
    // Objective metric: Φ's R factor over the data points (shared by entries).
    const rows = 2 * net.freq.len;
    const r_fac = try arena.alloc(f64, rows * npar);
    const s = try arena.alloc(f64, net.freq.len);
    for (s, net.freq) |*o, fr| o.* = 2 * std.math.pi * fr;
    sparam_fit.fillBasis(f.poles, s, r_fac);
    // Work in column-normalized parameters Δ'_c = n_c·Δ_c: a residue's
    // basis column is ~1/|p| next to the direct term's 1, and R⁻¹ of the
    // unscaled basis loses the constraint to roundoff.
    const col_norm = try arena.alloc(f64, npar);
    for (col_norm, 0..) |*n, c| {
        const col = r_fac[c * rows ..][0..rows];
        const v = numerics.dot(col, col);
        n.* = if (v > 0) @sqrt(v) else 1;
        numerics.scale(col, 1 / n.*, col);
    }
    const junk = try arena.alloc(f64, rows);
    @memset(junk, 0);
    sparam_lsq.householder(rows, npar, r_fac, junk);
    // Check grid: log-spaced over and past the band, the data points and
    // each pole's resonance.
    var grid: std.ArrayList(f64) = .empty;
    const w_hi = 2 * std.math.pi * net.freq[net.freq.len - 1];
    const w_lo = @max(2 * std.math.pi * net.freq[0], w_hi * 1e-6);
    try grid.append(arena, 0);
    const npts = 40 * (kk + 5);
    for (0..npts) |i| try grid.append(arena, w_lo / 10 * std.math.pow(f64, 1000 * w_hi / w_lo, @as(f64, @floatFromInt(i)) / @as(f64, @floatFromInt(npts - 1))));
    for (s) |w| try grid.append(arena, w);
    for (f.poles) |pole| if (pole.im > 0) {
        for ([_]f64{ -1, -0.5, 0, 0.5, 1 }) |t| try grid.append(arena, @max(0, pole.im + t * @abs(pole.re)));
    };
    std.mem.sort(f64, grid.items, {}, std.sort.asc(f64));

    const g = try arena.alloc(Cx, ne);
    const q = try arena.alloc(Cx, p);
    // minEig's 2P×2P matrix and eigenvectors, reused at every grid point.
    const work = try arena.alloc(f64, 8 * p * p);
    var cons: std.ArrayList(f64) = .empty; // rows of B, ne * npar each
    var rhs: std.ArrayList(f64) = .empty;
    var round: usize = 0;
    while (round < 30) : (round += 1) {
        cons.clearRetainingCapacity();
        rhs.clearRetainingCapacity();
        // λ_min over the grid; one constraint per local minimum below zero.
        var prev: f64 = std.math.inf(f64);
        var falling = false;
        var cand_w: f64 = 0;
        var cand_l: f64 = 0;
        for (grid.items, 0..) |w, gi| {
            hermitianPart(f.*, p, w, g);
            const l = minEig(p, g, null, work);
            if (l < prev) {
                falling = true;
                cand_w = w;
                cand_l = l;
            } else if (falling) {
                falling = false;
                if (cand_l < 0) try addConstraint(arena, f.*, p, cand_w, cand_l, margin, g, q, work, &cons, &rhs);
            }
            if (gi == grid.items.len - 1 and falling and cand_l < 0) try addConstraint(arena, f.*, p, cand_w, cand_l, margin, g, q, work, &cons, &rhs);
            prev = l;
        }
        // Direct term: sym(D) ⪰ 0 bounds Re Y at infinite frequency.
        for (0..p) |i| for (0..p) |j| {
            g[i * p + j] = .{ .re = 0.5 * (f.d[i * p + j] + f.d[j * p + i]), .im = 0 };
        };
        const ld = minEig(p, g, q, work);
        if (ld < 0) try appendRow(arena, f.*, p, null, q, -ld + margin, &cons, &rhs);
        const nc = rhs.items.len;
        if (nc == 0) return true;
        // Minimum ‖R·Δ_e‖ subject to B·Δ = c: with z_e = R·Δ_e,
        // z = Mᵀ(MMᵀ)⁻¹c, M = B·blockdiag(R⁻¹).
        const mm = try arena.alloc(f64, nc * ne * npar);
        for (0..nc) |ci| for (0..ne) |e| {
            // Row (B_c,e)·R⁻¹: solve Rᵀ·y = B_c,eᵀ.
            const bce = cons.items[(ci * ne + e) * npar ..][0..npar];
            const y = mm[(ci * ne + e) * npar ..][0..npar];
            for (0..npar) |c| {
                // R's column c above the diagonal is contiguous.
                const rc = r_fac[c * rows ..][0 .. c + 1];
                const v = bce[c] / col_norm[c] - numerics.dot(rc[0..c], y[0..c]);
                y[c] = if (@abs(rc[c]) > 1e-300) v / rc[c] else 0;
            }
        };
        // MMᵀ, symmetric: the upper triangle, mirrored.
        const gram = try arena.alloc(f64, nc * nc);
        const row_len = ne * npar;
        for (0..nc) |a| for (a..nc) |b| {
            const v = numerics.dot(mm[a * row_len ..][0..row_len], mm[b * row_len ..][0..row_len]);
            gram[a * nc + b] = v;
            gram[b * nc + a] = v;
        };
        const lam = try arena.alloc(f64, nc);
        @memcpy(lam, rhs.items);
        // Symmetric, so its row-major layout is also the column-major one
        // leastSquares takes; it destroys `gram`, which nothing reads after.
        sparam_lsq.leastSquares(nc, nc, gram, try arena.dupe(f64, lam), lam);
        // Δ_e = R⁻¹·z_e, z = Mᵀλ.
        const z = try arena.alloc(f64, npar);
        const delta = try arena.alloc(f64, npar);
        for (0..ne) |e| {
            @memset(z, 0);
            for (0..nc) |ci| numerics.axpy(z, lam[ci], mm[(ci * ne + e) * npar ..][0..npar]);
            // Column-oriented back substitution, as in leastSquares; it
            // consumes `z`, which the next entry rebuilds.
            var c = npar;
            while (c > 0) {
                c -= 1;
                const rc = r_fac[c * rows ..][0 .. c + 1];
                delta[c] = if (@abs(rc[c]) > 1e-300) z[c] / rc[c] else 0;
                numerics.axpy(z[0..c], -delta[c], rc[0..c]);
            }
            for (delta, col_norm) |*v, n| v.* /= n;
            var at: usize = 0;
            for (f.poles, f.res[e * f.poles.len ..][0..f.poles.len]) |pole, *r| {
                if (pole.im != 0) {
                    r.* = r.add(.{ .re = delta[at], .im = delta[at + 1] });
                    at += 2;
                } else {
                    r.re += delta[at];
                    at += 1;
                }
            }
            f.d[e] += delta[kk];
        }
    }
    return false;
}

/// Cyclic Jacobi on the real symmetric row-major n×n `a`: afterwards `a`'s
/// diagonal holds the eigenvalues and column i of `v` the unit eigenvector
/// of `a[i * n + i]`. Unordered. At most 100 sweeps, O(n³) each.
/// Asserts that `a` and `v` are n×n.
pub fn symEig(n: usize, a: []f64, v: []f64) void {
    std.debug.assert(a.len == n * n and v.len == n * n);
    @memset(v, 0);
    for (0..n) |i| v[i * n + i] = 1;
    for (0..100) |_| {
        var off: f64 = 0;
        for (0..n) |i| for (0..n) |j| {
            if (i != j) off += a[i * n + j] * a[i * n + j];
        };
        var diag: f64 = 0;
        for (0..n) |i| diag += a[i * n + i] * a[i * n + i];
        if (off <= 1e-30 * @max(diag, 1e-300)) break;
        for (0..n) |pp| for (pp + 1..n) |qq| {
            const apq = a[pp * n + qq];
            if (apq == 0) continue;
            const theta = (a[qq * n + qq] - a[pp * n + pp]) / (2 * apq);
            const t = std.math.sign(theta + 0.5 * @as(f64, if (theta == 0) 1 else 0)) / (@abs(theta) + @sqrt(theta * theta + 1));
            const c = 1 / @sqrt(t * t + 1);
            const sn = t * c;
            for (0..n) |k| {
                const akp = a[k * n + pp];
                const akq = a[k * n + qq];
                a[k * n + pp] = c * akp - sn * akq;
                a[k * n + qq] = sn * akp + c * akq;
            }
            for (0..n) |k| {
                const apk = a[pp * n + k];
                const aqk = a[qq * n + k];
                a[pp * n + k] = c * apk - sn * aqk;
                a[qq * n + k] = sn * apk + c * aqk;
            }
            for (0..n) |k| {
                const vkp = v[k * n + pp];
                const vkq = v[k * n + qq];
                v[k * n + pp] = c * vkp - sn * vkq;
                v[k * n + qq] = sn * vkp + c * vkq;
            }
        };
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

// ---- private ----

fn addConstraint(arena: Allocator, f: Fit, p: usize, w: f64, l: f64, margin: f64, g: []Cx, q: []Cx, work: []f64, cons: *std.ArrayList(f64), rhs: *std.ArrayList(f64)) Error!void {
    hermitianPart(f, p, w, g);
    _ = minEig(p, g, q, work);
    try appendRow(arena, f, p, w, q, -l + margin, cons, rhs);
}

/// One constraint row: Re(qᴴ·ΔY(jω)·q) = c over every entry's parameters
/// [basis coefficients..., d]; `w` null is the direct term alone.
fn appendRow(arena: Allocator, f: Fit, p: usize, w: ?f64, q: []const Cx, c: f64, cons: *std.ArrayList(f64), rhs: *std.ArrayList(f64)) Error!void {
    const npar = sparam_fit.realOrder(f.poles) + 1;
    for (0..p) |i| for (0..p) |j| {
        const qq = q[i].conjugate().mul(q[j]);
        if (w) |om| {
            for (f.poles) |pole| {
                const bv = sparam_fit.phi(pole, om);
                for (bv[0..if (pole.im != 0) 2 else 1]) |v| try cons.append(arena, qq.mul(v).re);
            }
        } else try cons.appendNTimes(arena, 0, npar - 1);
        try cons.append(arena, qq.re);
    };
    try rhs.append(arena, c);
}

/// (Y + Yᴴ)/2 of the fit at jω into `g` (row-major P×P).
fn hermitianPart(f: Fit, p: usize, w: f64, g: []Cx) void {
    for (0..p) |i| for (0..p) |j| {
        g[i * p + j] = f.eval(i * p + j, w);
    };
    for (0..p) |i| for (i..p) |j| {
        const a = g[i * p + j];
        const b = g[j * p + i].conjugate();
        const h = a.add(b).mul(.{ .re = 0.5, .im = 0 });
        g[i * p + j] = h;
        g[j * p + i] = h.conjugate();
    };
}

/// The smallest eigenvalue of the Hermitian P×P `g`, and its unit
/// eigenvector into `vec` when given: cyclic Jacobi on the real symmetric
/// 2P×2P embedding [[Re, −Im], [Im, Re]], whose spectrum doubles g's.
/// `work` holds 2·(2P)² scratch values.
fn minEig(p: usize, g: []const Cx, vec: ?[]Cx, work: []f64) f64 {
    const n = 2 * p;
    const a = work[0 .. n * n];
    const v = work[n * n ..][0 .. n * n];
    for (0..p) |i| for (0..p) |j| {
        const z = g[i * p + j];
        a[i * n + j] = z.re;
        a[(p + i) * n + (p + j)] = z.re;
        a[i * n + (p + j)] = -z.im;
        a[(p + i) * n + j] = z.im;
    };
    symEig(n, a, v);
    var best: usize = 0;
    for (1..n) |i| if (a[i * n + i] < a[best * n + best]) {
        best = i;
    };
    if (vec) |out| {
        var norm: f64 = 0;
        for (0..p) |i| {
            out[i] = .{ .re = v[i * n + best], .im = v[(p + i) * n + best] };
            norm += out[i].magnitude() * out[i].magnitude();
        }
        const inv = 1 / @sqrt(norm);
        for (out) |*o| o.* = o.mul(.{ .re = inv, .im = 0 });
    }
    return a[best * n + best];
}

test symEig {
    // Certificate: A·v_i = λ_i·v_i for every returned pair, unit v_i.
    const a0 = [9]f64{ 4, 1, 0.5, 1, 3, 0.2, 0.5, 0.2, 1 };
    var a = a0;
    var v: [9]f64 = undefined;
    symEig(3, &a, &v);
    for (0..3) |i| {
        var norm: f64 = 0;
        for (0..3) |r| {
            var av: f64 = 0;
            for (0..3) |c| av += a0[r * 3 + c] * v[c * 3 + i];
            try testing.expectApproxEqAbs(a[i * 3 + i] * v[r * 3 + i], av, 1e-12);
            norm += v[r * 3 + i] * v[r * 3 + i];
        }
        try testing.expectApproxEqAbs(@as(f64, 1), norm, 1e-12);
    }
    var one = [1]f64{-2};
    var vone: [1]f64 = undefined;
    symEig(1, &one, &vone);
    try testing.expectEqual(@as(f64, -2), one[0]);
    try testing.expectEqual(@as(f64, 1), vone[0]);
}

test "enforcePassivity lifts a negative direct term to the margin" {
    // Y(jω) = d = −1 everywhere: one grid minimum and the direct term
    // both violate; the minimum-norm change sets d to the 1e-9 margin.
    var arena_state = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const net: Network = .{ .ports = 1, .freq = &.{ 1e6, 2e6 }, .re = &.{ 1, 1 }, .im = &.{ 0, 0 } };
    var d = [_]f64{-1};
    var f: Fit = .{ .poles = &.{}, .res = try arena.alloc(Cx, 0), .d = &d };
    try testing.expect(try enforcePassivity(arena, net, &f));
    try testing.expect(d[0] >= 0 and d[0] < 1e-6);
}
