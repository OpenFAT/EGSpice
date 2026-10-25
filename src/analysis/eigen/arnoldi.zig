//! The largest-magnitude eigenvalues of a real operator known only through
//! its products, by restarted Arnoldi: the sparse `.pz` path runs it on a
//! shift-inverted pencil, whose largest eigenvalues are the roots nearest
//! the shift.
//!
//! Each restart keeps the real invariant subspace of the wanted Ritz
//! vectors of H (the real and imaginary parts of each, orthonormalized),
//! which is Stewart's Krylov-Schur restart with eigenvectors in place of
//! Schur vectors: AV = VT + v·bᵀ stays an exact Krylov decomposition, and
//! the next expansion appends Arnoldi columns to it. H is then Hessenberg
//! but for its full row under T, which the dense QR reduces anyway.
//! ponytail: eigenvectors, not Schur vectors, span the kept subspace, so a
//! nearly defective wanted pair loses accuracy in the restart; switch to a
//! reordered real Schur form if one ever stalls.
//!
//! Cites Stewart, "A Krylov-Schur algorithm for large eigenproblems" (SIAM
//! J. Matrix Anal. Appl. 23, 2001) and ARPACK's convergence test (dnaup2).
const std = @import("std");
const num = @import("core").numerics;
const qr = @import("core").eigen;
const dense_lu = @import("solver").dense_lu;

const Complex = num.Complex;

/// A Ritz pair is converged when ‖Ay − θy‖ ≤ tol·|θ|.
const tol = 1e-10;
const max_restarts = 300;
/// Deflation tolerance and step limit of the QR on H.
const qr_tol = 1e-12;
const qr_max_iter = 1000;

/// Writes the (up to) k largest-magnitude eigenvalues of the n×n operator `op`
/// (`op.apply(x, y)` writes y = A·x, never aliased), largest first, into
/// `out`, which needs k + 1 slots: a complex pair is never split, so k grows
/// by one when the k-th value is half of one. Values below n·eps·|θ₁| count
/// as zero (the infinite roots of a shift-inverted pencil) and are never
/// wanted, so fewer than k come back when A has fewer nonzero eigenvalues.
/// On running out of restarts only the converged ones are written, with
/// `converged` false. The start vector is a fixed pseudo-random one pushed
/// through `op` once, so a run is deterministic and starts inside A's range.
/// With `stable_only`, a Ritz value that a 1e-12 change of H moves by more
/// than 1e-3 relative is never wanted: the ring a defective eigenvalue of
/// high order leaves (see below). That drops a defective genuine
/// eigenvalue too, which without it comes back as the same ring a dense QR
/// gives. error.PzDidNotConverge when the QR on H runs out of steps.
pub fn dominant(gpa: std.mem.Allocator, n: usize, k: usize, op: anytype, out: []Complex, stable_only: bool) !qr.Eigs {
    std.debug.assert(k >= 1 and out.len >= k + 1);
    const m = @min(n, 2 * k + 20);
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    // V: m + 1 basis columns of n; `kept` holds the restart's new columns.
    const v = try a.alloc(f64, n * (m + 1));
    const kept = try a.alloc(f64, n * m);
    const w = try a.alloc(f64, n);
    // H: (m + 1) × m row-major, stride m; the last row holds β.
    const h = try a.alloc(f64, (m + 1) * m);
    const hs = try a.alloc(f64, m * m);
    const theta = try a.alloc(Complex, m);
    // H's eigenvalues after a small change, then a copy of `theta`.
    const theta_moved = try a.alloc(Complex, 2 * m);
    // Ritz vectors of H, `keep` of them as (re, im) column pairs of m.
    const y = try a.alloc(f64, 2 * m * m);
    const q = try a.alloc(f64, m * m);
    const hq = try a.alloc(f64, m * m);
    const lu = try a.alloc(f64, 4 * m * m);
    const piv = try a.alloc(u32, 2 * m);
    const rhs = try a.alloc(f64, 2 * m);
    const conv = try a.alloc(bool, m);

    var prng: std.Random.DefaultPrng = .init(0x9e3779b97f4a7c15);
    for (w) |*x| x.* = 2 * prng.random().float(f64) - 1;
    op.apply(w, v[0..n]);
    const beta0 = @sqrt(num.dot(v[0..n], v[0..n]));
    if (!(beta0 > 0)) return .{ .count = 0, .converged = true };
    num.scale(v[0..n], 1 / beta0, v[0..n]);
    @memset(h, 0);

    var j0: usize = 0;
    var restart: usize = 0;
    while (true) : (restart += 1) {
        // Expand to m columns, or until the space is invariant.
        var mm = m;
        var beta: f64 = 0;
        var j = j0;
        while (j < m) : (j += 1) {
            op.apply(v[j * n ..][0..n], w);
            const w_norm = @sqrt(num.dot(w, w));
            // Classical Gram-Schmidt, twice (DGKS): orthogonal to roundoff.
            for (0..2) |_| {
                for (0..j + 1) |i| hq[i] = num.dot(w, v[i * n ..][0..n]);
                for (0..j + 1) |i| {
                    h[i * m + j] += hq[i];
                    num.axpy(w, -hq[i], v[i * n ..][0..n]);
                }
            }
            beta = @sqrt(num.dot(w, w));
            if (beta <= 1e-14 * w_norm) {
                beta = 0;
                mm = j + 1;
                break;
            }
            h[(j + 1) * m + j] = beta;
            num.scale(v[(j + 1) * n ..][0..n], 1 / beta, w);
        }

        for (0..mm) |r| @memcpy(hs[r * mm ..][0..mm], h[r * m ..][0..mm]);
        const e = qr.eigenvalues(mm, hs[0 .. mm * mm], theta, qr_tol, qr_max_iter);
        if (!e.converged) return error.PzDidNotConverge;
        // Largest first; a complex pair adjacent, positive imaginary first.
        std.mem.sort(Complex, theta[0..mm], {}, struct {
            fn before(_: void, l: Complex, r: Complex) bool {
                const lm = l.magSq();
                const rm = r.magSq();
                if (lm != rm) return lm > rm;
                if (l.re != r.re) return l.re > r.re;
                return l.im > r.im;
            }
        }.before);

        // Genuine Ritz values first, in that order: above the zero cutoff,
        // and (`stable_only`) moved by at most 1e-3 relative when H moves
        // by 1e-12. A defective eigenvalue of order r (the infinite roots
        // of a pencil of high index) comes back as a ring of roundoff Ritz
        // values of radius ~eps^(1/r), which such a change regrows to
        // (1e-12)^(1/r) at new angles, while a simple eigenvalue moves by
        // its condition number times 1e-12.
        const cutoff = @as(f64, @floatFromInt(n)) * std.math.floatEps(f64) * theta[0].mag();
        var moved: qr.Eigs = .{ .count = 0, .converged = true };
        if (stable_only) {
            var h_norm: f64 = 0;
            for (0..mm) |r| h_norm += num.dot(h[r * m ..][0..mm], h[r * m ..][0..mm]);
            h_norm = @sqrt(h_norm);
            for (0..mm) |r| for (0..mm) |c| {
                const pattern: f64 = @floatFromInt((r * 7 + c * 13) % 17);
                hs[r * mm + c] = h[r * m + c] + 1e-12 * h_norm * (pattern / 16 - 0.5);
            };
            moved = qr.eigenvalues(mm, hs[0 .. mm * mm], theta_moved, qr_tol, qr_max_iter);
            if (!moved.converged) return error.PzDidNotConverge;
        }
        var genuine: usize = 0;
        @memcpy(theta_moved[mm..][0..mm], theta[0..mm]);
        var rest = mm;
        for (theta_moved[mm..][0..mm]) |t| {
            const ok = t.mag() > cutoff and (!stable_only or for (theta_moved[0..moved.count]) |o| {
                if (o.sub(t).mag() <= 1e-3 * t.mag()) break true;
            } else false);
            if (ok) {
                theta[genuine] = t;
                genuine += 1;
            } else {
                rest -= 1;
                theta[rest] = t;
            }
        }
        std.mem.reverse(Complex, theta[genuine..mm]);

        var want = @min(k, genuine);
        if (want > 0 and want < genuine and theta[want - 1].im > 0) want += 1;

        var keep = @max(want, want + (genuine - want) / 2);
        if (keep < mm and keep > 0 and theta[keep - 1].im > 0) keep += 1;
        if (beta != 0 and keep >= mm) keep = mm - 1;
        if (keep > 0 and keep < mm and theta[keep - 1].im > 0) keep -= 1;
        keep = @max(keep, want);

        // Ritz vectors and residuals β·|e_mᵀy| (exact space: zero).
        var all = true;
        for (0..keep) |i| {
            if (theta[i].im < 0) continue;
            try ritzVector(mm, h, m, theta[i], y[2 * i * m ..][0..m], y[(2 * i + 1) * m ..][0..m], lu, piv, rhs);
            if (i >= want) continue;
            const last: Complex = .{ .re = y[2 * i * m + mm - 1], .im = y[(2 * i + 1) * m + mm - 1] };
            conv[i] = beta * last.mag() <= tol * theta[i].mag();
            if (theta[i].im > 0) conv[i + 1] = conv[i];
            all = all and conv[i];
        }
        if (all or restart + 1 == max_restarts) {
            var count: usize = 0;
            for (theta[0..want], conv[0..want]) |t, c| if (all or c) {
                out[count] = t;
                count += 1;
            };
            return .{ .count = count, .converged = all };
        }

        // Orthonormal basis Q (columns of mm) of the kept Ritz vectors.
        var p: usize = 0;
        for (0..keep) |i| {
            if (theta[i].im < 0) continue;
            const parts: usize = if (theta[i].im > 0) 2 else 1;
            for (0..parts) |part| {
                const col = q[p * mm ..][0..mm];
                @memcpy(col, y[(2 * i + part) * m ..][0..mm]);
                for (0..2) |_| for (0..p) |c| {
                    const qc = q[c * mm ..][0..mm];
                    num.axpy(col, -num.dot(col, qc), qc);
                };
                const len = @sqrt(num.dot(col, col));
                // A nearly dependent part adds nothing the others lack.
                if (len <= 1e-10) continue;
                num.scale(col, 1 / len, col);
                p += 1;
            }
        }

        // T = QᵀHQ, and the new residual row β·(last row of Q).
        for (0..p) |c| for (0..mm) |r| {
            var s: f64 = 0;
            for (0..mm) |t| s += h[r * m + t] * q[c * mm + t];
            hq[c * mm + r] = s;
        };
        @memset(h, 0);
        for (0..p) |r| for (0..p) |c| {
            h[r * m + c] = num.dot(q[r * mm ..][0..mm], hq[c * mm ..][0..mm]);
        };
        for (0..p) |c| h[p * m + c] = beta * q[c * mm + mm - 1];

        // V ← V·Q, then the old residual direction becomes column p.
        for (0..p) |c| {
            const dst = kept[c * n ..][0..n];
            @memset(dst, 0);
            for (0..mm) |r| num.axpy(dst, q[c * mm + r], v[r * n ..][0..n]);
        }
        @memcpy(v[0 .. p * n], kept[0 .. p * n]);
        @memcpy(v[p * n ..][0..n], v[mm * n ..][0..n]);
        j0 = p;
    }
}

/// The unit eigenvector y = yr + i·yi of the leading mm×mm block of H (row
/// stride `stride`) for its eigenvalue θ = a + ib, by two steps of inverse
/// iteration on H − aI, stacked as [[H − aI, bI], [−bI, H − aI]] when b ≠ 0.
/// A real θ gets a real y (yi = 0), so its one kept column is never a
/// vanishing real part of a complex multiple.
fn ritzVector(mm: usize, h: []const f64, stride: usize, theta: Complex, yr: []f64, yi: []f64, lu: []f64, piv: []u32, rhs: []f64) !void {
    const blocks: usize = if (theta.im == 0) 1 else 2;
    const nn = blocks * mm;
    var shift = theta;
    for (0..2) |attempt| {
        @memset(lu[0 .. nn * nn], 0);
        for (0..blocks) |b| for (0..mm) |r| for (0..mm) |c| {
            lu[(r + b * mm) * nn + c + b * mm] = h[r * stride + c] - if (r == c) shift.re else 0;
        };
        if (blocks == 2) for (0..mm) |d| {
            lu[d * nn + d + mm] = shift.im;
            lu[(d + mm) * nn + d] = -shift.im;
        };
        dense_lu.factorize(nn, lu[0 .. nn * nn], piv[0..nn]) catch |err| {
            // An exact eigenvalue can give an exactly zero pivot: nudge it.
            if (attempt == 1) return err;
            shift.re += 1e-13 * @max(theta.mag(), std.math.floatMin(f64));
            continue;
        };
        break;
    }
    for (rhs[0..nn], 0..) |*b, i| b.* = 1 / @as(f64, @floatFromInt(i + 1));
    for (0..2) |_| {
        dense_lu.solveFactored(nn, lu[0 .. nn * nn], piv[0..nn], rhs[0..nn], rhs[0..nn]);
        const len = @sqrt(num.dot(rhs[0..nn], rhs[0..nn]));
        num.scale(rhs[0..nn], 1 / len, rhs[0..nn]);
    }
    @memcpy(yr[0..mm], rhs[0..mm]);
    if (blocks == 2) @memcpy(yi[0..mm], rhs[mm..nn]) else @memset(yi[0..mm], 0);
}

test dominant {
    // A block-diagonal operator with known eigenvalues: 300 real ones 1/i
    // and three rotation blocks r·e^{±iφ}. The largest are the pairs at 4
    // and 3, so k = 3 grows to 4 to keep the second pair whole.
    const n = 306;
    const Op = struct {
        fn apply(_: @This(), x: []const f64, out: []f64) void {
            for (0..300) |i| out[i] = x[i] / @as(f64, @floatFromInt(i + 1));
            for ([_]f64{ 4, 3, 0.5 }, 0..) |r, b| {
                const o = 300 + 2 * b;
                const c = r * @cos(0.3 * @as(f64, @floatFromInt(b + 1)));
                const s = r * @sin(0.3 * @as(f64, @floatFromInt(b + 1)));
                out[o] = c * x[o] - s * x[o + 1];
                out[o + 1] = s * x[o] + c * x[o + 1];
            }
        }
    };
    var out: [4]Complex = undefined;
    const e = try dominant(std.testing.allocator, n, 3, Op{}, &out, true);
    try std.testing.expect(e.converged);
    try std.testing.expectEqual(@as(usize, 4), e.count);
    const want = [_]Complex{
        .{ .re = 4 * @cos(0.3), .im = 4 * @sin(0.3) },
        .{ .re = 4 * @cos(0.3), .im = -4 * @sin(0.3) },
        .{ .re = 3 * @cos(0.6), .im = 3 * @sin(0.6) },
        .{ .re = 3 * @cos(0.6), .im = -3 * @sin(0.6) },
    };
    for (want, out) |wv, got| {
        try std.testing.expectApproxEqRel(wv.re, got.re, 1e-9);
        try std.testing.expectApproxEqAbs(wv.im, got.im, 1e-9 * wv.mag());
    }
}
