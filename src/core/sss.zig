//! Scaled-sigma sampling (Sun, Li, Liu, Luo, Gu, "Fast statistical analysis
//! of rare circuit failure events via scaled-sigma sampling", IEEE TCAD
//! 34(7), 2015): Monte Carlo trials drawn with every sigma times s > 1 fail
//! often enough to count, and the failure probability follows
//! ln P(s) = α + β ln s - γ/s². A weighted least-squares fit over the
//! scales extrapolates it to s = 1. Pure arithmetic; the driver counts.
const std = @import("std");
const numerics = @import("numerics.zig");
const assert = std.debug.assert;

/// Φ⁻¹(0.975): the half-width of a two-sided 95 % interval, in sigmas.
const z975 = 1.959963984540054;

/// Why `fit` gives no estimate.
pub const Error = error{
    /// Fewer than three scales saw some, but not all, of their trials fail.
    TooFewScales,
    /// The scales do not tell α, β and γ apart.
    Singular,
};

/// The fitted model and its extrapolation to s = 1.
pub const Estimate = struct {
    alpha: f64,
    beta: f64,
    gamma: f64,
    /// ln P_f(1) = α - γ.
    log_p: f64,
    /// Var(ln P_f(1)) from the binomial counts by the delta method. It is
    /// statistical only: the model's own error at s = 1 is not in it.
    log_var: f64,

    /// P_f(1), at most 1.
    pub fn p(e: Estimate) f64 {
        return @min(1, @exp(e.log_p));
    }

    /// The 95 % interval of P_f(1), symmetric in ln P, at most 1.
    pub fn interval(e: Estimate) [2]f64 {
        const h = z975 * @sqrt(e.log_var);
        return .{ @min(1, @exp(e.log_p - h)), @min(1, @exp(e.log_p + h)) };
    }
};

/// The equivalent sigma of failure probability `p`, Φ⁻¹(1 - p), as
/// -Φ⁻¹(p) so that 1 - p is never formed; negative above 1/2.
pub fn sigmaOf(p: f64) f64 {
    return -numerics.normalQuantile(std.math.clamp(p, std.math.floatMin(f64), 1 - std.math.floatEps(f64)));
}

/// Fits α, β, γ to `failures[j]` of `trials[j]` at `scales[j]` by weighted
/// least squares on y = ln p̂, row x = [1, ln s, -1/s²], weight
/// 1/Var(ln p̂) = k/(1 - p̂), with the variances known (no residual
/// rescaling, so three scales give an interval too). Only the scales with
/// 0 < k < n enter: ln 0 is no observation, and k = n has no variance.
/// Asserts that the slices have one length and no count passes its trials.
pub fn fit(scales: []const f64, trials: []const u64, failures: []const u64) Error!Estimate {
    assert(scales.len == trials.len and scales.len == failures.len);
    var a: [3][3]f64 = @splat(@splat(0));
    var b: [3]f64 = @splat(0);
    var used: usize = 0;
    for (scales, trials, failures) |s, n, k| {
        assert(k <= n);
        if (k == 0 or k == n) continue;
        used += 1;
        const kf: f64 = @floatFromInt(k);
        const p = kf / @as(f64, @floatFromInt(n));
        const w = kf / (1 - p);
        const x = [3]f64{ 1, @log(s), -1 / (s * s) };
        for (0..3) |i| {
            b[i] += w * @log(p) * x[i];
            for (0..3) |j| a[i][j] += w * x[i] * x[j];
        }
    }
    if (used < 3) return error.TooFewScales;
    // A = L Lᵀ; a pivot this small is a rank the scales cannot carry.
    var l: [3][3]f64 = @splat(@splat(0));
    const trace = a[0][0] + a[1][1] + a[2][2];
    for (0..3) |j| {
        var d = a[j][j];
        for (0..j) |k| d -= l[j][k] * l[j][k];
        if (!(d > 1e-14 * trace)) return error.Singular;
        l[j][j] = @sqrt(d);
        for (j + 1..3) |i| {
            var v = a[i][j];
            for (0..j) |k| v -= l[i][k] * l[j][k];
            l[i][j] = v / l[j][j];
        }
    }
    // θ = A⁻¹ b, and Var(x₀ᵀθ) = x₀ᵀ A⁻¹ x₀ = |L⁻¹ x₀|² at x₀ = [1, 0, -1].
    const theta = backward(l, forward(l, b));
    const t = forward(l, .{ 1, 0, -1 });
    return .{
        .alpha = theta[0],
        .beta = theta[1],
        .gamma = theta[2],
        .log_p = theta[0] - theta[2],
        .log_var = t[0] * t[0] + t[1] * t[1] + t[2] * t[2],
    };
}

/// Solves L y = v.
fn forward(l: [3][3]f64, v: [3]f64) [3]f64 {
    var y: [3]f64 = undefined;
    for (0..3) |i| {
        var r = v[i];
        for (0..i) |k| r -= l[i][k] * y[k];
        y[i] = r / l[i][i];
    }
    return y;
}

/// Solves Lᵀ x = y.
fn backward(l: [3][3]f64, y: [3]f64) [3]f64 {
    var x: [3]f64 = undefined;
    var i: usize = 3;
    while (i > 0) {
        i -= 1;
        var r = y[i];
        for (i + 1..3) |k| r -= l[k][i] * x[k];
        x[i] = r / l[i][i];
    }
    return x;
}

const testing = std.testing;

test "counts drawn from the model itself give its parameters back (zero residual)" {
    // Oracle: k = n·P(s) to the count's last digit, so every ln p̂ sits on
    // the model and least squares must return (α, β, γ) exactly.
    const alpha: f64 = -3;
    const beta: f64 = 1.5;
    const gamma: f64 = 8;
    const scales = [_]f64{ 2, 2.5, 3, 3.5, 4 };
    var n: [5]u64 = @splat(1 << 50);
    var k: [5]u64 = undefined;
    for (scales, &k) |s, *c| c.* = @intFromFloat(@round(0x1p50 * @exp(alpha + beta * @log(s) - gamma / (s * s))));
    const e = try fit(&scales, &n, &k);
    try testing.expectApproxEqAbs(alpha, e.alpha, 1e-9);
    try testing.expectApproxEqAbs(beta, e.beta, 1e-9);
    try testing.expectApproxEqAbs(gamma, e.gamma, 1e-9);
    try testing.expectApproxEqAbs(alpha - gamma, e.log_p, 1e-9);
    // A scale that saw no failure, or nothing but failures, is no row.
    n[0] = 10;
    k[0] = 10;
    try testing.expectApproxEqAbs(alpha - gamma, (try fit(&scales, &n, &k)).log_p, 1e-9);
}

test "three scales: the estimate interpolates, and its variance is Σ λ²(1 - p)/k by Cramer's rule" {
    // Oracle: with three rows the fit is the interpolation, so ln P(1) =
    // λᵀy with Xᵀλ = x₀, and Var = Σ λ_j² Var(ln p̂_j). λ comes from
    // Cramer's rule here, not from the normal equations.
    const scales = [_]f64{ 1.5, 2, 3 };
    const n = [_]u64{ 1000, 1000, 1000 };
    const k = [_]u64{ 100, 300, 600 };
    const e = try fit(&scales, &n, &k);
    // Columns of Xᵀ are the rows x_j.
    var m: [3][3]f64 = undefined;
    for (scales, 0..) |s, j| {
        m[0][j] = 1;
        m[1][j] = @log(s);
        m[2][j] = -1 / (s * s);
    }
    const x0 = [3]f64{ 1, 0, -1 };
    const det = det3(m);
    var log_p: f64 = 0;
    var log_var: f64 = 0;
    for (0..3) |j| {
        var mj = m;
        for (0..3) |i| mj[i][j] = x0[i];
        const lambda = det3(mj) / det;
        const p = @as(f64, @floatFromInt(k[j])) / 1000;
        log_p += lambda * @log(p);
        log_var += lambda * lambda * (1 - p) / @as(f64, @floatFromInt(k[j]));
    }
    try testing.expectApproxEqRel(log_p, e.log_p, 1e-10);
    try testing.expectApproxEqRel(log_var, e.log_var, 1e-10);
    const iv = e.interval();
    try testing.expectApproxEqRel(@exp(log_p - z975 * @sqrt(log_var)), iv[0], 1e-9);
    try testing.expect(iv[0] < e.p() and e.p() < iv[1]);
}

fn det3(m: [3][3]f64) f64 {
    return m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
        m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
        m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
}

test "fewer than three informative scales, or three equal ones, give no estimate" {
    try testing.expectError(error.TooFewScales, fit(&.{ 1, 2, 3 }, &.{ 100, 100, 100 }, &.{ 0, 5, 100 }));
    try testing.expectError(error.TooFewScales, fit(&.{ 2, 3 }, &.{ 100, 100 }, &.{ 5, 10 }));
    try testing.expectError(error.Singular, fit(&.{ 2, 2, 2 }, &.{ 100, 100, 100 }, &.{ 5, 6, 7 }));
}

test "equivalent sigma: Φ⁻¹(1 - p), past the reach of 1 - p" {
    try testing.expectApproxEqAbs(@as(f64, 5), sigmaOf(numerics.normalTail(5)), 1e-7);
    try testing.expectApproxEqAbs(@as(f64, 0), sigmaOf(0.5), 1e-12);
    try testing.expect(sigmaOf(0.9) < 0 and std.math.isFinite(sigmaOf(0)) and std.math.isFinite(sigmaOf(1)));
}

/// Failures among `n` Bernoulli trials of probability `p`.
fn bernoulli(r: std.Random, n: usize, p: f64) u64 {
    var k: u64 = 0;
    for (0..n) |_| k += @intFromBool(r.float(f64) < p);
    return k;
}

test "coverage: the 95 % interval holds the closed form Q(5) in at least 34 of 40 seeds" {
    // A margin linear in Gaussians fails at s with P = Q(5/s) exactly. One
    // seed proves nothing about an interval; for true 95 % coverage, 33 or
    // fewer of 40 happen with probability 0.34 % (binomial).
    const scales = [_]f64{ 2, 2.5, 3, 3.5, 4 };
    const n: [5]u64 = @splat(2000);
    const truth = numerics.normalTail(5);
    var held: u32 = 0;
    for (0..40) |seed| {
        var prng = std.Random.DefaultPrng.init(seed);
        var k: [5]u64 = undefined;
        for (scales, &k) |s, *c| c.* = bernoulli(prng.random(), 2000, numerics.normalTail(5 / s));
        const iv = (try fit(&scales, &n, &k)).interval();
        held += @intFromBool(iv[0] <= truth and truth <= iv[1]);
    }
    try testing.expect(held >= 34);
}

test "a quadratic margin: the scaled-sigma interval meets brute force's at s = 1" {
    // Oracle: 10⁶ plain trials of g(z) = 4 - z₁ - 0.1 z₂² < 0 (about 7e-5,
    // no closed form), and their own 95 % interval.
    var prng = std.Random.DefaultPrng.init(239);
    const r = prng.random();
    const fails = struct {
        fn at(rand: std.Random, s: f64) bool {
            const z1 = s * rand.floatNorm(f64);
            const z2 = s * rand.floatNorm(f64);
            return 4 - z1 - 0.1 * z2 * z2 < 0;
        }
    }.at;
    const brute = 1_000_000;
    var k_brute: u64 = 0;
    for (0..brute) |_| k_brute += @intFromBool(fails(r, 1));
    try testing.expect(k_brute > 20);
    const kb: f64 = @floatFromInt(k_brute);
    const pb = kb / brute;
    const hb = z975 * @sqrt((1 - pb) / kb);
    const scales = [_]f64{ 2, 2.5, 3, 3.5, 4 };
    const n: [5]u64 = @splat(2000);
    var k: [5]u64 = @splat(0);
    for (scales, &k) |s, *c| for (0..2000) |_| {
        c.* += @intFromBool(fails(r, s));
    };
    const iv = (try fit(&scales, &n, &k)).interval();
    try testing.expect(iv[0] <= pb * @exp(hb) and pb * @exp(-hb) <= iv[1]);
}
