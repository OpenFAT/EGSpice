//! What an expanded card is read under. In: the subcircuit definitions and
//! `.param` scopes. Out: `Frame` (path, port map, scopes), `Subckt` and
//! model-line rows, and the analysis-net resolution.
//! Cites: ngspice subckt.c (modtranslate: local models), inp.c; #64.
//!
//! Contents:
//! - `pub const Entry`
//! - `pub const Scope`
//! - `pub const Frame`
//! - `pub const PortIndex`
//! - `pub const port_scan_max`
//! - `pub const subckt_depth_max`
//! - `pub const Subckt`
//! - `pub const ModelRow`
//! - `pub const LocalModel`
//! - `pub const ModelLine`
//! - `pub const top_level`
//! - `pub fn groupNode(value: Value, which: usize, buf: *[24]u8) ?[]const u8`
//! - `pub fn nodeText(value: Value, buf: *[24]u8) ?[]const u8`
//! - `pub fn resolve(a: *Analysis, lookup: anytype) void`
//! - `pub fn appendAnalysis(arena: Allocator, list: *std.ArrayList(Analysis), a: Analysis) Allocator.Error!void`
//! - `pub fn appendSpan(comptime T: type, arena: Allocator, list: *std.ArrayList(T), items: []const T) Error!Span`

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
const netlist_view = @import("netlist_view.zig");
const card_table = @import("card_table.zig");
const VertexId = card_rows.VertexId;
const none = card_rows.none;
const Error = card_rows.Error;
const Span = card_rows.Span;
const Value = card_rows.Value;
const Analysis = card_rows.Analysis;
const isGround = netlist_view.isGround;
const cond = card_table.cond;

pub const Entry = union(enum) { num: f64, text: []const u8 };

// Insertion-ordered, so a walk over the globals is deck order (#64).
pub const Scope = std.StringArrayHashMapUnmanaged(Entry);

/// Where an expanded card is read: the enclosing `X` card's full name, its
/// port map, and the parameter scopes, outermost first.
pub const Frame = struct {
    path: ?[]const u8 = null,
    ports: []const []const u8 = &.{},
    actuals: []const VertexId = &.{},
    /// `ports` by name (`Subckt.port_index`), when long enough to keep one.
    port_index: ?*const PortIndex = null,
    scopes: []const *const Scope,
    depth: u8 = 0,
    instance: u32 = 0,
    /// The subcircuit being expanded, null at top level.
    sub: ?u16 = null,
    /// Where this instance's rows start in `Reader.instance_models`.
    models_mark: u32 = 0,
};

/// A subcircuit's port names to their first position.
pub const PortIndex = std.StringHashMapUnmanaged(u32);

/// Ports past this count are looked up through a `PortIndex` instead of a
/// scan, so a wrapper holding a whole extracted network costs pins, not
/// pins × ports.
pub const port_scan_max = 16;

/// Nested `.subckt` definitions past this depth are refused; ngspice's
/// syntax check warns past 10 (ngspice-syntax.md §4.1).
pub const subckt_depth_max = 16;

pub const Subckt = struct {
    name: []const u8,
    ports: []const []const u8,
    defaults: []const struct { key: []const u8, text: []const u8 },
    /// Lines between `.subckt` and `.ends`. Dot cards among them are global,
    /// except `.model` (`LocalModel`).
    first: u32,
    end: u32,
    /// This subcircuit's `.model` cards, `Reader.local_models[models_lo..models_hi]`.
    models_lo: u32 = 0,
    models_hi: u32 = 0,
    /// Some line in the body is an `.if` chain card: `expand` reads the dot
    /// cards only then, since a PDK wrapper carries ~180 `.model` bins.
    has_cond: bool = false,
    /// The body holds a nested `.subckt` definition, whose lines `expand`
    /// skips (they are that subcircuit's).
    has_nested: bool = false,
    /// `ports` by name past `port_scan_max` ports, else null.
    port_index: ?*const PortIndex = null,
};

pub const ModelRow = struct { name: []const u8, kind: []const u8, kv: Span };

/// A `.model` card inside a subcircuit. As in ngspice (subckt.c
/// modtranslate), only cards of that subcircuit's own body see it. `top` is
/// the card read at global scope. Every instance shares it when `shared`;
/// otherwise each `X` instance reads the card again under its own
/// parameters, because its values need one the global scope lacks (IHP's
/// `pre_layout`, or the subcircuit's `l`/`w`) or it sits under the body's
/// `.if` (`cond`), which only an instance can decide.
pub const LocalModel = struct { name: []const u8, line: u32, top: u32, shared: bool, cond: bool };

/// A `.model` line, the innermost subcircuit whose body holds it (`top`
/// for none), and whether an `.if` there guards it.
pub const ModelLine = struct { line: u32, cond: bool, owner: u16 = top_level };

/// `ModelLine.owner` of a model outside every subcircuit.
pub const top_level = std.math.maxInt(u16);

/// Name inside a `v(...)` group, or a bare name; `which` 1 is the reference.
pub fn groupNode(value: Value, which: usize, buf: *[24]u8) ?[]const u8 {
    return switch (value) {
        .name => |n| if (which == 0) n else null,
        .group => |g| if (std.ascii.eqlIgnoreCase(g.name(), "v") and g.args().len > which) nodeText(g.args()[which], buf) else null,
        else => null,
    };
}

/// A node spelled as a bare argument: a name, or an integral number (`.pz 1 0 3 0`).
pub fn nodeText(value: Value, buf: *[24]u8) ?[]const u8 {
    return switch (value) {
        .name => |n| n,
        .num => |n| if (n == @trunc(n) and @abs(n) < 1e9)
            std.fmt.bufPrint(buf, "{d}", .{@as(i64, @intFromFloat(n))}) catch null
        else
            null,
        else => null,
    };
}

/// Output and `.pz` nets of `a`, through `lookup.node(name) u32`.
pub fn resolve(a: *Analysis, lookup: anytype) void {
    const arg: usize = if (a.kind == .four) 1 else 0;
    var buf: [24]u8 = undefined;
    if (arg < a.args.len) {
        if (groupNode(a.args[arg], 0, &buf)) |n| a.pos = lookup.node(n);
        if (groupNode(a.args[arg], 1, &buf)) |n| a.neg = lookup.node(n);
    }
    // `.lstb ... localgnd=n` (VACASK `acstb`): `pos` is the local ground,
    // ground itself when absent.
    if (a.kind == .lstb) {
        a.pos = 0;
        for (a.args[0..a.args.len -| 1], 1..) |v, i| if (v == .name and std.ascii.eqlIgnoreCase(v.name, "localgnd")) {
            const n = nodeText(a.args[i], &buf) orelse continue;
            a.pos = if (isGround(a.dialect, n)) 0 else lookup.node(n);
        };
    }
    if (a.kind != .pz) return;
    for (&a.ports, 0..) |*id, i| {
        if (i >= a.args.len) break;
        const port = nodeText(a.args[i], &buf) orelse continue;
        id.* = if (isGround(a.dialect, port)) 0 else lookup.node(port);
    }
}

/// Appends `a` to `list`, a `.four` naming several outputs as one card per
/// output (the frequency, that output, and any trailing numbers).
pub fn appendAnalysis(arena: Allocator, list: *std.ArrayList(Analysis), a: Analysis) Allocator.Error!void {
    var outputs: usize = 0;
    if (a.kind == .four) for (a.args) |v| {
        if (v == .group) outputs += 1;
    };
    if (outputs < 2) return list.append(arena, a);
    const before = list.items.len;
    for (a.args[1..]) |v| {
        if (v != .group) continue;
        var args: std.ArrayList(Value) = .empty;
        try args.appendSlice(arena, &.{ a.args[0], v });
        for (a.args[1..]) |t| if (t != .group) try args.append(arena, t);
        var part = a;
        part.args = args.items;
        part.split = true;
        try list.append(arena, part);
    }
    assert(list.items.len == before + outputs);
}

/// Appends `items` to `list` and returns their rows. `CircuitTooLarge`
/// when the table would pass u32 rows; `list` is unchanged then.
pub fn appendSpan(comptime T: type, arena: Allocator, list: *std.ArrayList(T), items: []const T) Error!Span {
    const start = list.items.len;
    if (start + items.len > none) return error.CircuitTooLarge;
    try list.appendSlice(arena, items);
    assert(list.items.len == start + items.len);
    return .{ .start = @intCast(start), .len = @intCast(items.len) };
}

// ---- private ----
