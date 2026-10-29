//! Monte Carlo draws. In: a deck's `.option seed`/`sampling` and a trial.
//! Out: counter-based draws per distribution site (Latin hypercube
//! optional), and the live values of a point or trial set on the netlist.
//! Cites: HSPICE [SA] Ch.25 (Monte Carlo); Acklam's normal quantile.
//!
//! Contents:
//! - `pub const Sampler`
//! - `pub fn init(arena: std.mem.Allocator, config: []const netlist.Config, n: u32) !Sampler`
//! - `pub fn uniform(s: *Sampler, site: u64, trial: u32, index: u32) f64`
//! - `pub const OneSigma`
//! - `pub fn sample(self: OneSigma, site: u64, f: expr.Fn, args: []const f64) f64`
//! - `pub const Trial`
//! - `pub fn sample(self: TrialDraw, site: u64, f: expr.Fn, args: []const f64) f64`
//! - `pub fn setPoint(scratch: std.mem.Allocator, stack: *std.ArrayList(expr.Val), nl: *Netlist, pt: variants.Point) !void`
//! - `pub fn drawOnce(scratch: std.mem.Allocator, nl: *Netlist) !void`
//! - `pub fn normalQuantile(u: f64) f64`

const std = @import("std");
const assert = std.debug.assert;
const testing = std.testing;
const Allocator = std.mem.Allocator;
const core = @import("core");
const device = @import("device");
const netlist = @import("netlist");
const builder = @import("builder");
const analyses = @import("analyses.zig");
const expr = netlist.expr;
const Netlist = netlist.Netlist;
const ParamRef = device.abi.ParamRef;
const variants = @import("variants.zig");

/// Draws for one sweep's trials: counter-based, so trial k's value at a
/// site depends only on (seed, k, site), never on order or other sites.
/// Latin hypercube stratifies each site over the sweep's trials.
pub const Sampler = struct {
    seed: u64,
    n: u32,
    lhs: bool,
    arena: std.mem.Allocator,
    /// Latin hypercube: per site, trial index to stratum.
    strata: std.AutoHashMapUnmanaged(u64, []const u32) = .empty,

    /// Reads `.option seed=` (HSPICE default 1) and `sampling_method=lhs`;
    /// any method other than `lhs` or `srs` is `UnsupportedCard`.
    pub fn init(arena: std.mem.Allocator, config: []const netlist.Config, n: u32) !Sampler {
        var s: Sampler = .{ .seed = 1, .n = n, .lhs = false, .arena = arena };
        for (config) |c| if (!c.temp) for (c.args, 0..) |a, i| {
            if (a != .name or i + 1 >= c.args.len) continue;
            const next = c.args[i + 1];
            // lossyCast: a NaN or out-of-range seed saturates, never traps.
            if (std.ascii.eqlIgnoreCase(a.name, "seed") and next == .num) s.seed = std.math.lossyCast(u64, @abs(next.num));
            if (std.ascii.eqlIgnoreCase(a.name, "sampling_method") and next == .name) {
                if (std.ascii.eqlIgnoreCase(next.name, "lhs")) s.lhs = true else if (!std.ascii.eqlIgnoreCase(next.name, "srs")) return error.UnsupportedCard;
            }
        };
        return s;
    }

    /// Uniform in (0, 1) for HSPICE trial number `trial` at `site`.
    /// `index` (0-based within the sweep) picks the Latin hypercube
    /// stratum: it must be below `n`, which only the safety-checked
    /// `strata[index]` enforces. The first Latin hypercube draw at a site
    /// allocates its strata in `arena` (on allocation failure the draw
    /// falls back to plain sampling).
    pub fn uniform(s: *Sampler, site: u64, trial: u32, index: u32) f64 {
        const bits = netlist.siteKey(netlist.siteKey(s.seed, trial), site);
        const u = (@as(f64, @floatFromInt(bits >> 11)) + 0.5) * 0x1.0p-53;
        if (!s.lhs) return u;
        const strata = s.strataOf(site) catch return u;
        return (@as(f64, @floatFromInt(strata[index])) + u) / @as(f64, @floatFromInt(s.n));
    }

    /// A random permutation of the `n` strata for `site`.
    fn strataOf(s: *Sampler, site: u64) ![]const u32 {
        const gop = try s.strata.getOrPut(s.arena, site);
        if (gop.found_existing) return gop.value_ptr.*;
        const perm = try s.arena.alloc(u32, s.n);
        for (perm, 0..) |*x, i| x.* = @intCast(i);
        var prng = std.Random.DefaultPrng.init(netlist.siteKey(s.seed, site));
        // A u64 index draws the same bits on every target; `shuffle`'s
        // usize index permuted differently on wasm32.
        prng.random().shuffleWithIndex(u32, perm, u64);
        gop.value_ptr.* = perm;
        return perm;
    }
};

/// Holds every distribution call at its nominal except those at `site`,
/// which sit one sigma above it: a uniform's half range over sqrt(3), a
/// `limit`'s whole range (`Planner.siteVariations`).
pub const OneSigma = struct {
    site: u64,

    /// Asserts that `args` holds at least the nominal value.
    pub fn sample(self: OneSigma, site: u64, f: expr.Fn, args: []const f64) f64 {
        std.debug.assert(args.len >= 1);
        const nom = args[0];
        if (site != self.site) return nom;
        const width = if (args.len > 1) args[1] else 0;
        const gaussian = f == .agauss or f == .gauss;
        const sigmas = if (gaussian and args.len > 2 and args[2] != 0) args[2] else 1;
        return nom + switch (f) {
            .agauss => width / sigmas,
            .gauss => nom * width / sigmas,
            .aunif => width / @sqrt(3.0),
            .unif => nom * width / @sqrt(3.0),
            else => width, // else: `limit`; only distribution calls reach a sampler
        };
    }
};

/// One Monte Carlo trial of a sweep.
pub const Trial = struct {
    sampler: *Sampler,
    /// HSPICE trial number; keys the draws.
    trial: u32,
    /// Position within the sweep; picks the Latin hypercube stratum.
    index: u32,

    fn uniform(t: Trial, site: u64) f64 {
        return t.sampler.uniform(site, t.trial, t.index);
    }
};

/// Answers the distribution calls of one trial [SA Ch.20 "Monte Carlo
/// Parameter Distribution"]: `agauss(nom, abs, sigmas[, mult])`,
/// `gauss(nom, rel, sigmas[, mult])`, `aunif(nom, abs[, mult])`,
/// `unif(nom, rel[, mult])`, `limit(nom, abs)`. With a multiplier m the
/// draw of largest deviation among m is kept.
const TrialDraw = struct {
    trial: Trial,

    /// The trial's draw for distribution call `f` at `site`. Asserts that
    /// `args` holds at least the nominal value.
    pub fn sample(self: TrialDraw, site: u64, f: expr.Fn, args: []const f64) f64 {
        std.debug.assert(args.len >= 1);
        const nom = args[0];
        const width = if (args.len > 1) args[1] else 0;
        const gaussian = f == .agauss or f == .gauss;
        const sigmas = if (gaussian and args.len > 2 and args[2] != 0) args[2] else 1;
        const mult_at: usize = if (gaussian) 3 else 2;
        const mult: u32 = if (f != .limit and args.len > mult_at and args[mult_at] >= 1) @intFromFloat(@min(args[mult_at], 1e6)) else 1;
        const scale = switch (f) {
            .agauss => width / sigmas,
            .gauss => nom * width / sigmas,
            .unif => nom * width,
            else => width,
        };
        var best: f64 = 0;
        for (0..mult) |m| {
            const u = self.trial.uniform(netlist.siteKey(site, m));
            const d = switch (f) {
                .agauss, .gauss => scale * normalQuantile(u),
                .limit => if (u < 0.5) -scale else scale,
                else => scale * (2 * u - 1),
            };
            if (m == 0 or @abs(d) > @abs(best)) best = d;
        }
        return nom + best;
    }
};

/// Sets `nl`'s live values to `pt`'s, drawing its Monte Carlo trial if it
/// has one. `stack` is evaluation scratch, grown with `scratch`.
pub fn setPoint(scratch: std.mem.Allocator, stack: *std.ArrayList(expr.Val), nl: *Netlist, pt: variants.Point) !void {
    if (pt.trial) |t| {
        _ = try nl.setLive(scratch, stack, pt.live, TrialDraw{ .trial = t });
    } else _ = try nl.setLive(scratch, stack, pt.live, null);
}

/// Draws every distribution site of `nl` once, as trial 1 of the deck's
/// `.option seed`, into its live values (`Deck.draw_seed`). `scratch` holds
/// the sampler and the evaluation stack.
pub fn drawOnce(scratch: std.mem.Allocator, nl: *Netlist) !void {
    var sampler = try Sampler.init(scratch, nl.deck.config, 1);
    var stack: std.ArrayList(expr.Val) = .empty;
    _ = try nl.setLive(scratch, &stack, nl.live.nominal, TrialDraw{ .trial = .{ .sampler = &sampler, .trial = 1, .index = 0 } });
}

/// The standard normal quantile Φ⁻¹(u), u in (0, 1): Acklam's rational
/// approximation, relative error below 1.2e-9. At 0 or 1 it returns NaN,
/// so callers draw from the open interval.
pub fn normalQuantile(u: f64) f64 {
    const a = [_]f64{ -3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00 };
    const b = [_]f64{ -5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01 };
    const c = [_]f64{ -7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00 };
    const d = [_]f64{ 7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00 };
    const low = 0.02425;
    if (u < low or u > 1 - low) {
        const q = @sqrt(-2 * @log(if (u < low) u else 1 - u));
        const x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
        return if (u < low) x else -x;
    }
    const q = u - 0.5;
    const r = q * q;
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q /
        (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
}

// ---- private ----

test "normal quantile: symmetric, and the 97.5% point is 1.96" {
    try std.testing.expectApproxEqAbs(@as(f64, 0), normalQuantile(0.5), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.959963984540054), normalQuantile(0.975), 1e-8);
    try std.testing.expectApproxEqAbs(-normalQuantile(0.01), normalQuantile(0.99), 1e-12);
}

test "Sampler.init reads seed and sampling method, and a wild seed saturates" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const cfg = struct {
        fn of(args: []const netlist.Value) [1]netlist.Config {
            return .{.{ .temp = false, .args = args }};
        }
    }.of;
    var c = cfg(&.{ .{ .name = "seed" }, .{ .num = -7 }, .{ .name = "sampling_method" }, .{ .name = "LHS" } });
    const s = try Sampler.init(a, &c, 4);
    try std.testing.expectEqual(@as(u64, 7), s.seed);
    try std.testing.expect(s.lhs);
    c = cfg(&.{ .{ .name = "seed" }, .{ .num = comptime std.math.nan(f64) } });
    try std.testing.expectEqual(@as(u64, 0), (try Sampler.init(a, &c, 4)).seed);
    c = cfg(&.{ .{ .name = "seed" }, .{ .num = 1e300 } });
    try std.testing.expectEqual(@as(u64, std.math.maxInt(u64)), (try Sampler.init(a, &c, 4)).seed);
    c = cfg(&.{ .{ .name = "sampling_method" }, .{ .name = "sobol" } });
    try std.testing.expectError(error.UnsupportedCard, Sampler.init(a, &c, 4));
    // A trailing key with no value is skipped.
    c = cfg(&.{.{ .name = "seed" }});
    try std.testing.expectEqual(@as(u64, 1), (try Sampler.init(a, &c, 4)).seed);
}

test "TrialDraw keeps the largest deviation of a multiplier's draws" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var sampler = try Sampler.init(arena.allocator(), &.{}, 1);
    const draw: TrialDraw = .{ .trial = .{ .sampler = &sampler, .trial = 1, .index = 0 } };
    const one = draw.sample(5, .agauss, &.{ 1, 0.1, 1 });
    const many = draw.sample(5, .agauss, &.{ 1, 0.1, 1, 8 });
    // m = 0 is the single draw, so the max over eight is at least as far.
    try std.testing.expect(@abs(many - 1) >= @abs(one - 1));
    // `limit` lands on nom +- abs exactly; `aunif` stays inside the band.
    try std.testing.expectEqual(@as(f64, 0.5), @abs(draw.sample(9, .limit, &.{ 2, 0.5 }) - 2));
    try std.testing.expect(@abs(draw.sample(9, .aunif, &.{ 2, 0.5 }) - 2) < 0.5);
    // Nominal only: no spread.
    try std.testing.expectEqual(@as(f64, 3), draw.sample(9, .agauss, &.{3}));
}

test "normal quantile is monotone across the tail seam" {
    var prev = normalQuantile(1e-12);
    for ([_]f64{ 1e-6, 0.02, 0.02425, 0.0243, 0.3, 0.5, 0.7, 0.97575, 0.98, 1 - 1e-12 }) |u| {
        const x = normalQuantile(u);
        try std.testing.expect(x > prev);
        prev = x;
    }
    try std.testing.expect(std.math.isNan(normalQuantile(0)));
}
