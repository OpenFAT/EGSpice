//! Dense least squares for the vector fit. In: an m×n column-major matrix
//! and a right-hand side. Out: the minimum-residual solution, by
//! Householder QR that reads and updates whole columns, so every inner loop
//! is a contiguous dot or axpy (`core.numerics`).
//! Cites: Golub and Van Loan, Matrix Computations, 4th ed., §5.1-§5.3.
//!
//! Contents:
//! - `pub fn householder(m: usize, n: usize, a: []f64, b: []f64) void`
//! - `pub fn leastSquares(m: usize, n: usize, a: []f64, b: []f64, x: []f64) void`

const std = @import("std");
const assert = std.debug.assert;
const testing = std.testing;
const Allocator = std.mem.Allocator;
const numerics = @import("core").numerics;

/// In-place Householder QR of the column-major m×n `a`, applied to `b` as
/// well: afterwards `a`'s upper triangle is R, the rest is zero, and `b` is
/// Qᵀb. With m < n the columns past m are only reflected.
pub fn householder(m: usize, n: usize, a: []f64, b: []f64) void {
    std.debug.assert(a.len >= m * n and b.len >= m);
    for (0..@min(n, m)) |c| {
        // v = (v0, tail); H = I − 2vvᵀ/(vᵀv).
        const col = a[c * m + c .. (c + 1) * m];
        const tail = col[1..];
        const tail2 = numerics.dot(tail, tail);
        const norm = @sqrt(col[0] * col[0] + tail2);
        if (norm == 0) continue;
        const alpha = if (col[0] > 0) -norm else norm;
        const v0 = col[0] - alpha;
        const vtv = v0 * v0 + tail2;
        if (vtv == 0) continue;
        for (c + 1..n) |j| reflect(a[j * m + c .. (j + 1) * m], v0, tail, vtv);
        reflect(b[c..m], v0, tail, vtv);
        col[0] = alpha;
        @memset(tail, 0);
    }
}

/// min ‖A·x − b‖ for column-major m×n `a` (m ≥ n; destroyed, as is `b`),
/// with column scaling for conditioning and a zero for a null column.
pub fn leastSquares(m: usize, n: usize, a: []f64, b: []f64, x: []f64) void {
    std.debug.assert(m >= n and x.len == n);
    // `x` holds each column's scale until its solution component replaces it.
    for (x, 0..) |*s, c| {
        const col = a[c * m ..][0..m];
        const norm2 = numerics.dot(col, col);
        s.* = if (norm2 > 0) 1 / @sqrt(norm2) else 0;
        numerics.scale(col, s.*, col);
    }
    householder(m, n, a, b);
    // Column-oriented back substitution: each solved component is removed
    // from the rows above it with one contiguous axpy down its column.
    var c = n;
    while (c > 0) {
        c -= 1;
        const col = a[c * m ..][0 .. c + 1];
        const y = if (@abs(col[c]) > 1e-14) b[c] / col[c] else 0;
        numerics.axpy(b[0..c], -y, col[0..c]);
        x[c] *= y;
    }
}

// ---- private ----

/// w −= 2·v·(vᵀw)/(vᵀv) for v = (v0, tail), w.len = 1 + tail.len.
fn reflect(w: []f64, v0: f64, tail: []const f64, vtv: f64) void {
    const f = 2 * (v0 * w[0] + numerics.dot(tail, w[1..])) / vtv;
    w[0] -= f * v0;
    numerics.axpy(w[1..], -f, tail);
}

test "leastSquares meets the normal equations at every column-length tail" {
    // Certificate: the LS solution x makes the residual orthogonal to every
    // column, Aᵀ(Ax − b) = 0. Lengths sweep past 3×8 lanes plus one so every
    // dot/axpy tail inside the QR runs.
    var prng = std.Random.DefaultPrng.init(0x5ea1);
    const rnd = prng.random();
    const max_m = 4 + 3 * 8 + 2;
    var a0: [max_m * 4]f64 = undefined;
    var a: [max_m * 4]f64 = undefined;
    var b0: [max_m]f64 = undefined;
    var b: [max_m]f64 = undefined;
    var x: [4]f64 = undefined;
    for (1..5) |n| for (n..max_m) |m| {
        for (a0[0 .. m * n]) |*v| v.* = rnd.float(f64) * 2 - 1;
        for (b0[0..m]) |*v| v.* = rnd.float(f64) * 2 - 1;
        @memcpy(a[0 .. m * n], a0[0 .. m * n]);
        @memcpy(b[0..m], b0[0..m]);
        leastSquares(m, n, a[0 .. m * n], b[0..m], x[0..n]);
        var na: f64 = 0;
        for (a0[0 .. m * n]) |v| na += v * v;
        var nx: f64 = 0;
        for (x[0..n]) |v| nx += v * v;
        var nb: f64 = 0;
        for (b0[0..m]) |v| nb += v * v;
        const bound = 1e-9 * @sqrt(na) * (@sqrt(na) * @sqrt(nx) + @sqrt(nb));
        for (0..n) |c| {
            var g: f64 = 0;
            for (0..m) |r| {
                var ax: f64 = 0;
                for (0..n) |j| ax += a0[j * m + r] * x[j];
                g += a0[c * m + r] * (ax - b0[r]);
            }
            try testing.expect(@abs(g) <= bound);
        }
    };
}

test "leastSquares zeroes a null column" {
    // Column 0 all ones, column 1 all zeros: x = (mean b, 0).
    var a = [_]f64{ 1, 1, 1, 1, 1, 0, 0, 0, 0, 0 };
    var b = [_]f64{ 1, 2, 3, 4, 5 };
    var x: [2]f64 = undefined;
    leastSquares(5, 2, &a, &b, &x);
    try testing.expectApproxEqRel(@as(f64, 3), x[0], 1e-12);
    try testing.expectEqual(@as(f64, 0), x[1]);
}
