//! The published deck tables. In: the parse arena's `.meas`, `.stim`,
//! `.ic` and `.save` cards and the builder's AC overrides. Out: the same
//! rows copied into the session arena, on circuit rows, so the deck outlives
//! the parse.
//! Cites: none (house deck layout, core/deck.zig).
//!
//! Contents:
//! - `pub fn keepSaved(scratch: Allocator, probes: *[]u32, labels: *[][]const u8, saves: []const []const u8) Allocator.Error!void`
//! - `pub fn stims(arena: Allocator, cards: []const core.Stim) Allocator.Error![]const core.Stim`
//! - `pub fn nodeRows(arena: Allocator, nb: *const builder.NetBuilder, items: []const netlist.Ic) Allocator.Error![]const core.Ic`
//! - `pub fn measures(arena: Allocator, cards: []const core.Measure) Allocator.Error![]const core.Measure`
//! - `pub fn acOverrides(arena: Allocator, cards: []const requests.CardRef, names: []const []const u8, values: []const f64) error{ OutOfMemory, InvalidAcOverride }![]const core.AcOverride`

const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;
const core = @import("core");
const requests = core.query;
const netlist = @import("netlist");
const builder = @import("builder");
const NO_NODE = netlist.none;

/// Narrows the published outputs to the `.save` labels, in place. The
/// probes keep their order; a label nothing publishes is dropped.
pub fn keepSaved(scratch: Allocator, probes: *[]u32, labels: *[][]const u8, saves: []const []const u8) Allocator.Error!void {
    var wanted: std.StringHashMapUnmanaged(void) = .empty;
    for (saves) |s| try wanted.put(scratch, s, {});
    const p = probes.*;
    const l = labels.*;
    assert(p.len == l.len);
    var kept: usize = 0;
    // One buffer for every lowercased label, not an allocation per label.
    var lower: std.ArrayList(u8) = .empty;
    for (p, l) |row, label| {
        try lower.resize(scratch, label.len);
        if (!wanted.contains(std.ascii.lowerString(lower.items, label))) continue;
        p[kept] = row;
        l[kept] = label;
        kept += 1;
    }
    assert(kept <= p.len);
    probes.* = p[0..kept];
    labels.* = l[0..kept];
}

/// `.stim` cards with their strings copied out of the parse arena.
pub fn stims(arena: Allocator, cards: []const core.Stim) Allocator.Error![]const core.Stim {
    const out = try arena.dupe(core.Stim, cards);
    for (out) |*s| {
        if (s.file) |file| s.file = try arena.dupe(u8, file);
        s.dataname = try arena.dupe(u8, s.dataname);
        s.signals = try measures(arena, s.signals);
        const nodes = try arena.dupe([2][]const u8, s.nodes);
        for (nodes) |*pair| for (pair) |*n| {
            n.* = try arena.dupe(u8, n.*);
        };
        s.nodes = nodes;
        s.times = try arena.dupe(f64, s.times);
    }
    return out;
}

/// `.ic`-style values on circuit rows. A node no device touches is dropped,
/// like every other unresolvable directive name.
pub fn nodeRows(arena: Allocator, nb: *const builder.NetBuilder, items: []const netlist.Ic) Allocator.Error![]const core.Ic {
    var out: std.ArrayList(core.Ic) = .empty;
    for (items) |item| {
        const row = nb.frozenRow(item.net.index());
        if (row == NO_NODE or row == core.GROUND) continue;
        try out.append(arena, .{ .node = row, .value = item.value });
    }
    return out.items;
}

/// `.meas` cards with their strings copied out of the parse arena.
pub fn measures(arena: Allocator, cards: []const core.Measure) Allocator.Error![]const core.Measure {
    const out = try arena.dupe(core.Measure, cards);
    for (out) |*m| {
        m.name = try arena.dupe(u8, m.name);
        m.expr = try measureOps(arena, m.expr);
        m.check.expect = try arena.dupe([2]f64, m.check.expect);
        for ([_]*core.MeasureClause{ &m.first, &m.second }) |c| {
            c.vec = try arena.dupe(u8, c.vec);
            c.vec2 = try arena.dupe(u8, c.vec2);
            c.ops = try measureOps(arena, c.ops);
            c.ops2 = try measureOps(arena, c.ops2);
            const refs = try arena.dupe(core.MeasureRef, c.refs);
            for (refs) |*ref| ref.expr = try measureOps(arena, ref.expr);
            c.refs = refs;
        }
    }
    return out;
}

/// Each `ac=` resistance as a (type, instance, parameter) override, so every
/// analysis clone binds its own parameter pointer.
pub fn acOverrides(
    arena: Allocator,
    cards: []const requests.CardRef,
    names: []const []const u8,
    values: []const f64,
) ![]const core.AcOverride {
    const overrides = try arena.alloc(core.AcOverride, names.len);
    for (names, values, overrides) |name, value, *override| {
        for (cards) |card| {
            if (!std.mem.eql(u8, card.name, name)) continue;
            override.* = .{ .type = card.type, .index = card.index, .param_name = "r", .value = value };
            break;
        } else return error.InvalidAcOverride;
    }
    return overrides;
}

// ---- private ----

fn measureOps(arena: Allocator, ops: []const core.MeasureOp) Allocator.Error![]const core.MeasureOp {
    const out = try arena.dupe(core.MeasureOp, ops);
    for (out) |*op| if (op.* == .vector) {
        op.vector = try arena.dupe(u8, op.vector);
    };
    return out;
}

test keepSaved {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    // A label past 64 bytes is still matched (it used to be dropped).
    const long = "v(xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx)";
    var probe_buf = [_]u32{ 1, 2, 3 };
    var label_buf = [_][]const u8{ "v(A)", long, "v(c)" };
    var probes: []u32 = &probe_buf;
    var labels: [][]const u8 = &label_buf;
    try keepSaved(arena.allocator(), &probes, &labels, &.{ long, "v(a)", "v(nowhere)" });
    try std.testing.expectEqualSlices(u32, &.{ 1, 2 }, probes);
    try std.testing.expectEqualStrings("v(A)", labels[0]);
    try std.testing.expectEqualStrings(long, labels[1]);
    try keepSaved(arena.allocator(), &probes, &labels, &.{"v(none)"});
    try std.testing.expectEqual(@as(usize, 0), probes.len);
}
