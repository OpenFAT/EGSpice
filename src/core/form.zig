//! First-order reliability (FORM) for `.wcd` (#240): the worst-case
//! distance β_w of one spec, the distance in standard-normal space from the
//! origin to the nearest point where the spec fails (Hasofer and Lind, J.
//! Eng. Mech. 100(1), 1974; Antreich, Graeb and Wieser, IEEE TCAD 13(1),
//! 1994). The iteration is iHL-RF: Rackwitz and Fiessler's step to the
//! linearized boundary, kept by a line search on a merit function (Zhang
//! and Der Kiureghian, Struct. Safety 17(2), 1995). The caller owns
//! evaluation, as with `lm.zig`: `points` yields the points to simulate and
//! `feed` takes the spec's margin g at each, g ≥ 0 where the spec passes.
const std = @import("std");
const numerics = @import("numerics.zig");
const assert = std.debug.assert;
const dot = numerics.dot;

/// The forward-difference step of the gradient, in sigmas.
pub const step = 0.1;
/// Halvings of the line-search step before the search gives up.
const max_halvings = 8;
/// Armijo's sufficient-decrease fraction. Not 1/2: on a linear margin the
/// full step lands exactly on m(z) + slope/2, a tie rounding would decide.
const armijo = 0.1;

/// `.wcd`'s `MAXITER=` and `TOL=`.
pub const Options = struct {
    /// Gradient rounds before the search stops unconverged.
    maxiter: u32 = 20,
    /// Converged when the linearized step would change β by less than
    /// this and the boundary is nearer than this, both in sigmas.
    tol: f64 = 0.01,
};

/// Why the search stopped; `running` until it has. `no_gradient`: no
/// coordinate moves the margin. `line_search`: no step along the
/// linearized direction lowered the merit. `failed`: a gradient point could
/// not be evaluated (a NaN margin).
pub const Status = enum(u8) { running, converged, maxiter, no_gradient, line_search, failed };

/// One search. All storage comes from the arena passed to `init`.
pub const Form = struct {
    /// Coordinates, at least 1.
    d: u32,
    opts: Options,
    /// The last gradient round's point: the one `beta` and the report name.
    z: []f64,
    /// Its margin and forward-difference gradient.
    g: f64 = std.math.nan(f64),
    a: []f64,
    /// The step from `z` to the nearest point of the linearized boundary.
    dir: []f64,
    /// Pending points, `d` values each: `z` then `z + step·e_i` in a
    /// gradient round, `z + lambda·dir` in a line round.
    pts: []f64,
    pending: u32,
    phase: Phase = .gradient,
    /// +1 when the nominal point passes, -1 when it fails; 0 before the
    /// first round.
    sign: f64 = 0,
    /// The merit m(z) = ‖z‖²/2 + c|g(z)| at `z`, its penalty c (never
    /// lowered), its slope along `dir`, and the line-search step.
    c: f64 = 0,
    merit: f64 = 0,
    slope: f64 = 0,
    lambda: f64 = 1,
    halvings: u32 = 0,
    /// Gradient rounds and line points fed.
    iterations: u32 = 0,
    line_points: u32 = 0,
    status: Status = .running,
    /// sign·‖z‖ once a gradient round's margin is finite: the signed
    /// distance of a point that was simulated, never a predicted one.
    beta: ?f64 = null,

    const Phase = enum(u8) { gradient, line };

    /// Starts at the origin, every draw at its nominal. Asserts at least one
    /// coordinate, one round and a positive tolerance. Fails only when
    /// `arena` is out of memory.
    pub fn init(arena: std.mem.Allocator, d: u32, opts: Options) !Form {
        assert(d >= 1);
        assert(opts.maxiter >= 1);
        assert(opts.tol > 0);
        var s: Form = .{
            .d = d,
            .opts = opts,
            .z = try arena.alloc(f64, d),
            .a = try arena.alloc(f64, d),
            .dir = try arena.alloc(f64, d),
            .pts = try arena.alloc(f64, (@as(usize, d) + 1) * d),
            .pending = 0,
        };
        @memset(s.z, 0);
        s.gradientPoints();
        return s;
    }

    /// The points to evaluate next, `d` values each, row-major. Empty once
    /// `status` is no longer `running`.
    pub fn points(s: *const Form) []const f64 {
        if (s.status != .running) return &.{};
        assert(s.pending == 1 or s.pending == s.d + 1);
        return s.pts[0 .. @as(usize, s.pending) * s.d];
    }

    /// Whether `points()` is a gradient round, `z` and its `d` neighbours,
    /// whose margins a caller may take from one run's tangent rows.
    pub fn batched(s: *const Form) bool {
        return s.status == .running and s.phase == .gradient;
    }

    /// Whether the search met `tol`.
    pub fn converged(s: *const Form) bool {
        return s.status == .converged;
    }

    /// Takes the margins of `points()`, one per point in the same order; a
    /// NaN marks a point that could not be evaluated. Asserts that the
    /// search is still going and that there is one margin per point.
    pub fn feed(s: *Form, margins: []const f64) void {
        assert(s.status == .running);
        assert(margins.len == s.pending);
        switch (s.phase) {
            .gradient => s.gradient(margins),
            .line => s.line(margins[0]),
        }
    }

    /// A gradient round: the stop tests at `z`, else the Rackwitz-Fiessler
    /// direction and the line search's first point.
    fn gradient(s: *Form, g: []const f64) void {
        s.iterations += 1;
        s.g = g[0];
        if (!std.math.isFinite(g[0])) return s.stop(.failed);
        if (s.sign == 0) s.sign = if (g[0] >= 0) 1 else -1;
        const z_norm = numerics.norm2(s.z);
        s.beta = s.sign * z_norm;
        for (s.a, g[1..]) |*a, gi| a.* = (gi - g[0]) / step;
        const aa = dot(s.a, s.a);
        if (!std.math.isFinite(aa)) return s.stop(.failed);
        if (aa == 0) return s.stop(.no_gradient);
        const a_norm = @sqrt(aa);
        // The linearized boundary's point nearest the origin is t·a.
        const t = (dot(s.a, s.z) - g[0]) / aa;
        for (s.dir, s.a, s.z) |*d, a, z| d.* = t * a - z;
        const hl_norm = @abs(t) * a_norm;
        // Both, or Δβ alone could stall off the boundary and report a guess.
        if (@abs(hl_norm - z_norm) < s.opts.tol and @abs(g[0]) / a_norm < s.opts.tol) return s.stop(.converged);
        if (s.iterations >= s.opts.maxiter) return s.stop(.maxiter);
        // Zhang and Der Kiureghian's penalty, twice the least that makes
        // `dir` a descent direction of the merit.
        var c = z_norm / a_norm;
        if (g[0] != 0) c = @max(c, 0.5 * hl_norm * hl_norm / @abs(g[0]));
        s.c = @max(s.c, 2 * c);
        s.merit = 0.5 * z_norm * z_norm + s.c * @abs(g[0]);
        // The |g| term's derivative along `dir` is c·sign(g)·aᵀdir, and
        // aᵀdir = -g: 0 on the boundary.
        const sg: f64 = if (g[0] > 0) 1 else if (g[0] < 0) -1 else 0;
        s.slope = dot(s.z, s.dir) + s.c * sg * dot(s.a, s.dir);
        // Off the stop tests the slope is at most -tol²/2 (by Cauchy-Schwarz
        // and the penalty), far from rounding.
        assert(s.slope < 0);
        s.lambda = 1;
        s.halvings = 0;
        s.linePoint();
    }

    /// A line round: Armijo's test on the merit at `z + lambda·dir`; the
    /// first point that passes is the next gradient round's `z`. A NaN
    /// margin fails the test, so the step halves away from it.
    fn line(s: *Form, g: f64) void {
        s.line_points += 1;
        const at = s.pts[0..s.d];
        const m = 0.5 * dot(at, at) + s.c * @abs(g);
        if (m <= s.merit + armijo * s.lambda * s.slope) {
            @memcpy(s.z, at);
            return s.gradientPoints();
        }
        if (s.halvings == max_halvings) return s.stop(.line_search);
        s.halvings += 1;
        s.lambda /= 2;
        s.linePoint();
    }

    fn gradientPoints(s: *Form) void {
        const d = s.d;
        for (0..d + 1) |p| {
            const row = s.pts[p * d ..][0..d];
            @memcpy(row, s.z);
            if (p != 0) row[p - 1] += step;
        }
        s.pending = d + 1;
        s.phase = .gradient;
    }

    fn linePoint(s: *Form) void {
        for (s.pts[0..s.d], s.z, s.dir) |*p, z, d| p.* = z + s.lambda * d;
        s.pending = 1;
        s.phase = .line;
    }

    fn stop(s: *Form, status: Status) void {
        s.status = status;
        s.pending = 0;
    }
};

/// Runs `s` to its end against `f`, a test oracle with `eval(z) f64`.
/// Terminates: MAXITER caps the gradient rounds, `max_halvings` each line.
fn solve(s: *Form, f: anytype) void {
    var g: [8]f64 = undefined;
    while (s.points().len != 0) {
        const pts = s.points();
        const count = pts.len / s.d;
        for (0..count) |p| g[p] = f.eval(pts[p * s.d ..][0..s.d]);
        s.feed(g[0..count]);
    }
}

/// g = l - 3 z1 - 4 z2: linear, ‖a‖ = 5.
const Linear = struct {
    l: f64,
    fn eval(self: Linear, z: []const f64) f64 {
        return self.l - 3 * z[0] - 4 * z[1];
    }
};

/// g = b - z1 - (κ/2) z2²: a parabola curving toward the origin.
const Parabola = struct {
    b: f64,
    kappa: f64,
    fn eval(self: Parabola, z: []const f64) f64 {
        return self.b - z[0] - 0.5 * self.kappa * z[1] * z[1];
    }
};

test "a margin linear in the coordinates gives the closed-form beta in two gradient rounds, signed by the nominal side" {
    // Oracle: the closed form. g = l - aᵀz fails past the plane aᵀz = l,
    // whose nearest point is l·a/‖a‖², at distance l/‖a‖ = l/5. The full
    // step lands on it, so the second round converges with d = 0.
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    for ([_]f64{ 10, -10 }) |l| {
        var s = try Form.init(arena.allocator(), 2, .{});
        solve(&s, Linear{ .l = l });
        try std.testing.expectEqual(Status.converged, s.status);
        try std.testing.expectApproxEqAbs(l / 5, s.beta.?, 1e-12);
        try std.testing.expectApproxEqAbs(l * 3 / 25, s.z[0], 1e-12);
        try std.testing.expectApproxEqAbs(l * 4 / 25, s.z[1], 1e-12);
        try std.testing.expectEqual(@as(u32, 2), s.iterations);
        try std.testing.expectEqual(@as(u32, 1), s.line_points);
    }
}

test "a curved margin converges to its exactly known design point" {
    // Oracle: the KKT point. g = b - z1 - (κ/2) z2² with κb < 1: on the
    // boundary ‖z‖² = (b - κ z2²/2)² + z2², whose minimum is at z2 = 0
    // (second derivative 2(1 - κb) > 0), so z* = (b, 0) and β = b. The
    // forward difference in z2 reads -κ(z2 + step/2), which moves the
    // fixed point to z2 = κ·step·b / (2(1 - κb)) = 0.0214 and β by 5e-5.
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var s = try Form.init(arena.allocator(), 2, .{});
    solve(&s, Parabola{ .b = 3, .kappa = 0.1 });
    try std.testing.expectEqual(Status.converged, s.status);
    try std.testing.expectApproxEqAbs(@as(f64, 3), s.beta.?, s.opts.tol);
    try std.testing.expectApproxEqAbs(@as(f64, 3), s.z[0], s.opts.tol);
    try std.testing.expect(@abs(s.z[1]) < 0.03);
}

test "FORM's failure probability on a curved margin is off by Breitung's curvature factor, as the method says" {
    // Oracle: the exact P_f = ∫ φ(t) Q(b - κt²/2) dt by composite Simpson
    // over [-10, 10], 2000 panels (the tails past it are below 1e-20).
    // This documents the method's error, not the implementation: Φ(-β)
    // misses P_f by about (1 - κb)^(-1/2) = 1.195 (Breitung, J. Eng. Mech.
    // 110(3), 1984); the higher-order terms add about 1.3 % at b = 3.
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const b = 3.0;
    const kappa = 0.1;
    var s = try Form.init(arena.allocator(), 2, .{});
    solve(&s, Parabola{ .b = b, .kappa = kappa });
    const n = 2000;
    const h = 20.0 / @as(f64, n);
    var sum: f64 = 0;
    for (0..n + 1) |k| {
        const t = -10 + h * @as(f64, @floatFromInt(k));
        const w: f64 = if (k == 0 or k == n) 1 else if (k % 2 == 1) 4 else 2;
        sum += w * @exp(-0.5 * t * t) / @sqrt(2 * std.math.pi) * numerics.normalTail(b - 0.5 * kappa * t * t);
    }
    const exact = sum * h / 3;
    const form = numerics.normalTail(s.beta.?);
    try std.testing.expect(exact > 1.1 * form);
    try std.testing.expectApproxEqRel(1 / @sqrt(1 - kappa * b), exact / form, 0.03);
}

test "a margin no coordinate moves has no gradient and no converged beta" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var s = try Form.init(arena.allocator(), 3, .{});
    solve(&s, struct {
        fn eval(_: @This(), _: []const f64) f64 {
            return 2;
        }
    }{});
    try std.testing.expectEqual(Status.no_gradient, s.status);
    try std.testing.expect(!s.converged());
    try std.testing.expectEqual(@as(u32, 1), s.iterations);
}

test "MAXITER stops unconverged at the last simulated point, never at the predicted one" {
    // One round at the origin: β there is 0, though the linearized step
    // points at about 3.
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var s = try Form.init(arena.allocator(), 2, .{ .maxiter = 1 });
    solve(&s, Parabola{ .b = 3, .kappa = 0.1 });
    try std.testing.expectEqual(Status.maxiter, s.status);
    try std.testing.expect(!s.converged());
    try std.testing.expectEqual(@as(f64, 0), s.beta.?);
    try std.testing.expectEqual(@as(u32, 0), s.line_points);
    try std.testing.expectEqual(@as(usize, 0), s.points().len);
}

test "a margin that cannot be evaluated fails the search; a NaN line point halves the step" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var s = try Form.init(arena.allocator(), 2, .{});
    solve(&s, struct {
        fn eval(_: @This(), _: []const f64) f64 {
            return std.math.nan(f64);
        }
    }{});
    try std.testing.expectEqual(Status.failed, s.status);
    try std.testing.expectEqual(@as(?f64, null), s.beta);
    // The linear margin, NaN off the axes: the origin's gradient round is
    // finite, every point along its direction (1.2, 1.6) is not, so the
    // full step and each of its halvings fail Armijo's test.
    var t = try Form.init(arena.allocator(), 2, .{});
    solve(&t, struct {
        fn eval(_: @This(), z: []const f64) f64 {
            return if (z[0] != 0 and z[1] != 0) std.math.nan(f64) else 10 - 3 * z[0] - 4 * z[1];
        }
    }{});
    try std.testing.expectEqual(Status.line_search, t.status);
    try std.testing.expectEqual(@as(u32, max_halvings + 1), t.line_points);
}
