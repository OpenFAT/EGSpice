//! The optional members of a Newton system (`converger.zig`'s `sys`
//! contract): a system pointer in, each optional method's call or its
//! default out, plus the gmin load every solver path stamps. Cites ngspice
//! CKTload's gmin (cktload.c) for `loadGmin`.
//!
//! Contents:
//! - `pub fn Deref(comptime P: type) type`
//! - `pub fn executionOf(sys: anytype) Execution`
//! - `pub fn loadGmin(comptime V: type, sys: anytype, vals: ?[]V, rhs: ?[]V, x: []const V, opts: converger.Options) void`
//! - `pub fn applyLimits(sys: anytype, x: []f64, x_old: []f64) bool`
//! - `pub fn acceptStep(sys: anytype, x: []f64) bool`
//! - `pub fn sysNodeName(sys: anytype, idx: u32) []const u8`

const std = @import("std");
const assert = std.debug.assert;
const num = @import("core").numerics;
const Execution = num.Execution;
const lanes = @import("lane/lanes.zig");
const converger = @import("converger.zig");

/// Returns the type a system pointer points at (or `P` itself).
pub fn Deref(comptime P: type) type {
    return if (@typeInfo(P) == .pointer) @typeInfo(P).pointer.child else P;
}

/// Returns the BBD factor scheduler `sys` carries as `solver_execution`; serial
/// for a system without one.
pub fn executionOf(sys: anytype) Execution {
    return if (comptime @hasField(Deref(@TypeOf(sys)), "solver_execution")) sys.solver_execution else .{};
}

/// Loads `opts.gmin` as `converger.Options` documents: at each of `gmin_stamps`, or
/// on every diagonal when there are none. Adds it to `vals` when non-null
/// and its residual form, gmin * x[col], to `rhs` when non-null. Asserts
/// that `x` holds the system's n unknowns.
pub fn loadGmin(comptime V: type, sys: anytype, vals: ?[]V, rhs: ?[]V, x: []const V, opts: converger.Options) void {
    assert(x.len >= sys.n);
    if (!(opts.gmin > 0)) return;
    const gmin = lanes.Of(V).splat(opts.gmin);
    const stamps = opts.gmin_stamps;
    if (vals) |v| {
        if (stamps.len > 0) {
            for (stamps) |g| v[g.slot] += gmin;
        } else for (sys.diag_slots[0..sys.n]) |slot| v[slot] += gmin;
    }
    if (rhs) |r| {
        if (stamps.len > 0) {
            for (stamps) |g| r[g.row] += gmin * x[g.col];
        } else for (r[0..sys.n], x[0..sys.n]) |*ri, xi| ri.* += gmin * xi;
    }
}

/// Returns whether `sys.applyLimits` clamped `x` (false without one).
pub fn applyLimits(sys: anytype, x: []f64, x_old: []f64) bool {
    return if (comptime @hasDecl(Deref(@TypeOf(sys)), "applyLimits")) sys.applyLimits(x, x_old) else false;
}

/// Runs device convergence, then state staging; false refuses the iterate.
pub fn acceptStep(sys: anytype, x: []f64) bool {
    const S = Deref(@TypeOf(sys));
    if (comptime @hasDecl(S, "checkConvergence")) {
        if (!sys.checkConvergence(x)) return false;
    }
    if (comptime @hasDecl(S, "updateStates")) {
        if (sys.updateStates(x) != null) return false;
    }
    return true;
}

/// Returns `sys.nodeName(idx)`, or "?" for a system without names.
pub fn sysNodeName(sys: anytype, idx: u32) []const u8 {
    const S = Deref(@TypeOf(sys));
    if (comptime @hasDecl(S, "nodeName")) return sys.nodeName(idx);
    return "?";
}

test "loadGmin stamps the diagonal or the listed stamps, into vals and rhs independently" {
    const Sys = struct { n: u32 = 3, diag_slots: []const u32 = &.{ 0, 4, 8 } };
    const sys: Sys = .{};
    const x = [_]f64{ 1, -2, 3 };
    var vals: [9]f64 = @splat(0);
    var rhs: [3]f64 = @splat(1);
    loadGmin(f64, &sys, &vals, &rhs, &x, .{ .gmin = 0.5 });
    try std.testing.expectEqualSlices(f64, &.{ 0.5, 0, 0, 0, 0.5, 0, 0, 0, 0.5 }, &vals);
    try std.testing.expectEqualSlices(f64, &.{ 1.5, 0, 2.5 }, &rhs);
    // Stamps: (row 0, col 2) at slot 6, twice; rhs only.
    const stamps = [_]converger.GminStamp{ .{ .slot = 6, .row = 0, .col = 2 }, .{ .slot = 6, .row = 0, .col = 2 } };
    loadGmin(f64, &sys, null, &rhs, &x, .{ .gmin = 0.25, .gmin_stamps = &stamps });
    try std.testing.expectEqualSlices(f64, &.{ 3, 0, 2.5 }, &rhs);
    loadGmin(f64, &sys, &vals, null, &x, .{ .gmin = 0.25, .gmin_stamps = &stamps });
    try std.testing.expectEqual(@as(f64, 0.5), vals[6]);
    // Zero, negative and NaN gmin load nothing.
    const before = vals;
    for ([_]f64{ 0, -1, std.math.nan(f64) }) |g| loadGmin(f64, &sys, &vals, &rhs, &x, .{ .gmin = g });
    try std.testing.expectEqualSlices(f64, &before, &vals);
    try std.testing.expectEqualSlices(f64, &.{ 3, 0, 2.5 }, &rhs);
}
