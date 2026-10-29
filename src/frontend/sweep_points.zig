//! Sweep grids. In: a `lin`/`dec`/`oct`/`poi` grid and its bounds. Out:
//! its points, and the SPICE 2G6 polynomial term order.
//! Cites: ngspice dctran/acan grids; XSPICE spice2poly NXTPWR.
//!
//! Contents:
//! - `pub const Grid`
//! - `pub const grids`
//! - `pub fn points(arena: Allocator, grid: Grid, start: f64, stop: f64, x: f64, mode: enum { per_unit, total }) Error![]const f64`
//! - `pub fn nextPower(pw: []u32) void`

const std = @import("std");
const Allocator = std.mem.Allocator;
const lines = @import("lines.zig");
const spectre_rewrite = @import("spectre.zig");
const source = @import("source.zig");
const hypergraph = @import("hypergraph.zig");
const expr = @import("expr.zig");
const measure = @import("measure.zig");
const pattern = @import("pattern.zig");
const core = @import("core");
const Name = core.Name;
const InternPool = core.InternPool;
const assert = std.debug.assert;
const card_rows = @import("card_rows.zig");
const Error = card_rows.Error;

/// A sweep grid: `lin` from an increment or a point count, `dec`/`oct`
/// from points per decade or octave, `poi` an explicit list.
pub const Grid = enum { lin, dec, oct, poi };

pub const grids = std.StaticStringMap(Grid).initComptime(.{ .{ "lin", .lin }, .{ "dec", .dec }, .{ "oct", .oct }, .{ "poi", .poi } });

/// The points of `grid` from `start` to `stop`. `x` is the increment
/// (`per_unit`) or the point count (`total`) for `lin`, and points per
/// decade or octave for `dec`/`oct`. The end point is kept within a
/// relative 1e-9, as a SPICE sweep keeps it.
pub fn points(arena: Allocator, grid: Grid, start: f64, stop: f64, x: f64, mode: enum { per_unit, total }) Error![]const f64 {
    const max_points = 1 << 24;
    var n: f64 = undefined;
    var ratio: f64 = 1;
    switch (grid) {
        .poi => unreachable,
        .lin => if (mode == .total) {
            if (!(x >= 1) or x != @trunc(x)) return error.ParseError;
            n = x;
        } else {
            if (x == 0 or !std.math.isFinite((stop - start) / x) or (stop - start) / x < 0) return error.ParseError;
            n = @floor((stop - start) / x + 1e-9) + 1;
        },
        .dec, .oct => {
            if (!(start > 0) or !(stop >= start) or !(x > 0)) return error.ParseError;
            ratio = std.math.pow(f64, if (grid == .dec) 10 else 2, 1 / x);
            n = @floor(@log(stop / start) / @log(ratio) + 1e-9) + 1;
        },
    }
    if (!(n <= max_points)) return error.ParseError;
    assert(n >= 1);
    const out = try arena.alloc(f64, @intFromFloat(n));
    for (out, 0..) |*o, k| {
        const kf: f64 = @floatFromInt(k);
        o.* = switch (grid) {
            .lin => if (mode == .total)
                (if (out.len == 1) start else start + kf * (stop - start) / (n - 1))
            else
                start + kf * x,
            else => start * std.math.pow(f64, ratio, kf),
        };
    }
    return out;
}

/// Steps `pw` to the exponents of the next SPICE 2G6 polynomial term: a
/// literal port of NXTPWR (XSPICE spice2poly), which orders the terms as
/// the expansion of (a + (b + c))^k, k = 1, 2, ...
pub fn nextPower(pw: []u32) void {
    assert(pw.len > 0);
    const dim = pw.len;
    if (dim == 1) {
        pw[0] += 1;
        return;
    }
    var k = dim;
    while (k > 0 and pw[k - 1] == 0) k -= 1;
    if (k == 0) {
        pw[0] += 1;
        return;
    }
    if (k != dim) {
        pw[k - 1] -= 1;
        pw[k] += 1;
        return;
    }
    for (pw[0 .. k - 1]) |e| {
        if (e != 0) break;
    } else {
        pw[0] = pw[dim - 1] + 1;
        pw[dim - 1] = 0;
        return;
    }
    var psum: u32 = 1;
    k = dim;
    // Bounded: some exponent before the last is nonzero (the loop above).
    while (pw[k - 2] < 1) : (k -= 1) {
        assert(k > 2);
        psum += pw[k - 1];
        pw[k - 1] = 0;
    }
    pw[k - 1] += psum;
    pw[k - 2] -= 1;
}

// ---- private ----

test "nextPower: SPICE 2G6 term order" {
    var pw: [2]u32 = .{ 0, 0 };
    const want = [_][2]u32{ .{ 1, 0 }, .{ 0, 1 }, .{ 2, 0 }, .{ 1, 1 }, .{ 0, 2 }, .{ 3, 0 }, .{ 2, 1 } };
    for (want) |e| {
        nextPower(&pw);
        try std.testing.expectEqual(e, pw);
    }
    var p3: [3]u32 = .{ 0, 0, 0 };
    const want3 = [_][3]u32{ .{ 1, 0, 0 }, .{ 0, 1, 0 }, .{ 0, 0, 1 }, .{ 2, 0, 0 }, .{ 1, 1, 0 }, .{ 1, 0, 1 }, .{ 0, 2, 0 }, .{ 0, 1, 1 }, .{ 0, 0, 2 }, .{ 3, 0, 0 } };
    for (want3) |e| {
        nextPower(&p3);
        try std.testing.expectEqual(e, p3);
    }
}

test points {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    // The end point survives the increment's rounding.
    const lin = try points(a, .lin, 0, 1, 0.1, .per_unit);
    try std.testing.expectEqual(@as(usize, 11), lin.len);
    try std.testing.expectApproxEqAbs(@as(f64, 1), lin[10], 1e-12);
    try std.testing.expectEqualSlices(f64, &.{0}, try points(a, .lin, 0, 1, 1, .total));
    try std.testing.expectEqualSlices(f64, &.{ 0, 0.5, 1 }, try points(a, .lin, 0, 1, 3, .total));
    // A falling sweep with a falling increment.
    try std.testing.expectEqualSlices(f64, &.{ 2, 1, 0 }, try points(a, .lin, 2, 0, -1, .per_unit));
    const dec = try points(a, .dec, 1, 100, 2, .per_unit);
    try std.testing.expectEqual(@as(usize, 5), dec.len);
    try std.testing.expectApproxEqRel(@as(f64, 100), dec[4], 1e-12);
    try std.testing.expectEqual(@as(usize, 4), (try points(a, .oct, 1, 8, 1, .per_unit)).len);
    for ([_]struct { Grid, f64, f64, f64 }{
        .{ .lin, 0, 1, 0 }, // no increment
        .{ .lin, 0, 1, -0.1 }, // increment away from the stop
        .{ .lin, 0, 1e9, 1 }, // past 1 << 24 points
        .{ .dec, 0, 1, 10 }, // log grid through zero
        .{ .dec, 2, 1, 10 }, // falling log grid
        .{ .oct, 1, 2, 0 }, // no points per octave
        .{ .dec, 1, 10, std.math.nan(f64) },
    }) |c| try std.testing.expectError(error.ParseError, points(a, c[0], c[1], c[2], c[3], .per_unit));
    try std.testing.expectError(error.ParseError, points(a, .lin, 0, 1, 0.5, .total));
    try std.testing.expectError(error.ParseError, points(a, .lin, 0, 1, 0, .total));
}
