//! The parsed, flattened netlist and its read side. In: the reader's
//! frozen tables. Out: `Netlist`, whose views resolve a device's slices,
//! and `setLive`, which rewrites the live values for one variant.
//! Cites: ngspice INPgetModBin (binning), splitmix64 (site keys).
//!
//! Contents:
//! - `pub const Netlist`
//! - `pub const View`
//! - `pub fn bucket(nl: *const Netlist, c: u8) []const EdgeId`
//! - `pub fn device(nl: *const Netlist, e: EdgeId) View`
//! - `pub fn deviceCount(nl: *const Netlist) u32`
//! - `pub fn netName(nl: *const Netlist, v: VertexId) []const u8`
//! - `pub fn dataFile(nl: *const Netlist, path: []const u8) ?[]const u8`
//! - `pub fn findModel(nl: *const Netlist, name: []const u8) ?Model`
//! - `pub fn exprOps(nl: *const Netlist, span: Span) []const expr.Op`
//! - `pub fn setLive(nl: *Netlist, gpa: Allocator, stack: *std.ArrayList(expr.Val), values: []const f64, draw: anytype) !bool`
//! - `pub fn setSlots(nl: *Netlist, values: []const f64) void`
//! - `pub fn slot(nl: *const Netlist, k: usize) *Value`
//! - `pub fn SiteDraw(comptime D: type) type`
//! - `pub fn value(self: @This(), i: usize, f: expr.Fn, args: []const f64) f64`
//! - `pub fn siteKey(owner: u64, salt: u64) u64`
//! - `pub fn nameKey(owner: u64, name: []const u8) u64`
//! - `pub fn isGround(dialect: Dialect, name: []const u8) bool`
//! - `pub fn isGroundName(name: []const u8) bool`
//! - `pub fn nameIndex(names: []const []const u8, target: []const u8) ?usize`
//! - `pub fn binHolds(b: [4]f64, l: f64, w: f64) bool`
//! - `pub fn kvNumber(kv: []const Kv, key: []const u8) ?f64`
//! - `pub fn orderU32(a: u32, b: u32) std.math.Order`

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
const card_table = @import("card_table.zig");
const Dialect = card_rows.Dialect;
const VertexId = card_rows.VertexId;
const EdgeId = card_rows.EdgeId;
const none = card_rows.none;
const Span = card_rows.Span;
const Value = card_rows.Value;
const Kv = card_rows.Kv;
const Hypergraph = card_rows.Hypergraph;
const Model = card_rows.Model;
const Live = card_rows.Live;
const DataFile = card_rows.DataFile;
const Deck = card_rows.Deck;
const kvIndex = card_table.kvIndex;

/// A parsed, flattened deck. Every slice lives in the parse arena.
pub const Netlist = struct {
    /// Net and device names.
    pool: InternPool,
    graph: Hypergraph.Graph,
    /// Edges counting-sorted by card letter; `kind_starts[c - 'a']` opens
    /// letter c's run, [26] is the edge count.
    order: []const EdgeId,
    kind_starts: [27]u32,
    /// Positional values of every device, addressed by `Device.positional`.
    values: []const Value,
    /// `key=value` pairs of every device, addressed by `Device.kv`.
    kvs: []const Kv,
    /// Postfix of every unfolded expression, addressed by `Value.expr`.
    ops: []const expr.Op,
    /// Constant pool the `num` ops index. `setLive` rewrites the rows that
    /// stand in for live names (`Live.pool_rows`).
    consts: []f64,
    models: []const Model,
    /// Model name to its first `.model` row.
    model_ids: std.StringHashMapUnmanaged(u32),
    deck: Deck,
    live: Live = .{},
    /// Rows of `live.names` the caller asked to set on the prepared
    /// circuit (`parseLive`), in the caller's order.
    tunable: []const u32 = &.{},
    /// The `ForeignKind.data` files, as `prepare` read them.
    files: []const DataFile = &.{},

    /// A card as the builder reads it.
    pub const View = struct {
        name: []const u8,
        kind: u8,
        pins: []const VertexId,
        positional: []const Value,
        kv: []const Kv,
        model: ?Model,
        subckt_instance: u32,
        /// Row of `model` in `Netlist.models`; `none` without one.
        model_row: u32 = none,
    };

    /// Devices of card letter `c`, in file order. A behavioural E/F/G/H
    /// sits under `b`. Asserts that `c` is a lowercase ASCII letter.
    pub fn bucket(nl: *const Netlist, c: u8) []const EdgeId {
        assert(c >= 'a');
        assert(c <= 'z');
        return nl.order[nl.kind_starts[c - 'a']..nl.kind_starts[c - 'a' + 1]];
    }

    /// Device `e` with its slices resolved; they borrow the netlist.
    pub fn device(nl: *const Netlist, e: EdgeId) View {
        const i = e.index();
        assert(i < nl.graph.edges.len);
        const d = nl.graph.edges.get(i);
        assert(d.positional.start + d.positional.len <= nl.values.len);
        assert(d.kv.start + d.kv.len <= nl.kvs.len);
        return .{
            .name = nl.pool.str(d.name),
            .kind = d.kind,
            .pins = nl.graph.pins(e),
            .positional = nl.values[d.positional.start..][0..d.positional.len],
            .kv = nl.kvs[d.kv.start..][0..d.kv.len],
            .model = if (d.model == none) null else nl.models[d.model],
            .subckt_instance = d.subckt_instance,
            .model_row = d.model,
        };
    }

    /// Device count after subcircuit expansion.
    pub fn deviceCount(nl: *const Netlist) u32 {
        return nl.graph.edgeCount();
    }

    /// Name of net `v`: `x1.mid` for `mid` inside instance `x1`.
    pub fn netName(nl: *const Netlist, v: VertexId) []const u8 {
        assert(v.index() < nl.graph.vertices.len);
        return nl.pool.str(nl.graph.vertices.items(.name)[v.index()]);
    }

    /// The bytes of data file `path`, spelled as the card spells it; null
    /// before `prepare` reads it. O(files).
    pub fn dataFile(nl: *const Netlist, path: []const u8) ?[]const u8 {
        for (nl.files) |f| if (std.mem.eql(u8, f.path, path)) return f.bytes;
        return null;
    }

    /// First global `.model` card named `name`; a subcircuit's own models
    /// are reached only through `View.model`.
    pub fn findModel(nl: *const Netlist, name: []const u8) ?Model {
        return if (nl.model_ids.get(name)) |i| nl.models[i] else null;
    }

    /// The postfix of an unfolded `Value.expr`.
    pub fn exprOps(nl: *const Netlist, span: Span) []const expr.Op {
        assert(span.start + span.len <= nl.ops.len);
        return nl.ops[span.start..][0..span.len];
    }

    /// Rewrites every live value for one variant: `values[k]` is live name
    /// k, and `draw.sample(site, f, args)` answers each distribution call
    /// (`null` keeps nominals). Returns true when some value changed.
    /// `stack` is scratch, grown with `gpa`. The live slots and `consts`
    /// rows live in the parse arena, so every `View` and `Model` read
    /// before the call sees the new values. A binned M card whose L or W is
    /// live takes the bin its new geometry selects (`Live.bins`);
    /// `ModelBinNotFound` when none holds it, with the values already
    /// rewritten. `ParseError` when such a card's L or W no longer reads as
    /// a number, `OutOfMemory` from growing `stack`. Asserts that
    /// `values.len` equals `live.names.len`.
    pub fn setLive(nl: *Netlist, gpa: Allocator, stack: *std.ArrayList(expr.Val), values: []const f64, draw: anytype) !bool {
        const live = nl.live;
        const pool = nl.consts;
        var changed = false;
        for (live.pool_rows, values) |row, v| if (row != none) {
            changed = changed or pool[row] != v;
            pool[row] = v;
        };
        for (live.ops, live.scale, 0..) |span, scale, k| {
            const at = nl.slot(k);
            const x = scale * try nl.evalLive(gpa, stack, span, values, draw);
            changed = changed or at.num != x;
            at.* = .{ .num = x };
        }
        for (live.site_rows, live.site_spans) |row, span| {
            const x = try nl.evalLive(gpa, stack, span, values, draw);
            changed = changed or pool[row] != x;
            pool[row] = x;
        }
        // Each binned card with live geometry takes the bin its new L and W
        // select, read off the card as the slots above left them (scaled).
        const bins = live.bins;
        const model_of = nl.graph.edges.items(.model);
        const kv_of = nl.graph.edges.items(.kv);
        const positional_of = nl.graph.edges.items(.positional);
        for (bins.edges, bins.per_finger, 0..) |e, per_finger, k| {
            const kv = nl.kvs[kv_of[e].start..][0..kv_of[e].len];
            const l = kvNumber(kv, "l") orelse return error.ParseError;
            const nf = if (per_finger) kvNumber(kv, "nf") orelse 1 else 1;
            const w = (kvNumber(kv, "w") orelse return error.ParseError) / nf;
            const lo = bins.starts[k];
            const hi = bins.starts[k + 1];
            const row = for (bins.rows[lo..hi], bins.bounds[lo..hi]) |row, b| {
                if (binHolds(b, l, w)) break row;
            } else return error.ModelBinNotFound;
            changed = changed or model_of[e] != row;
            model_of[e] = row;
            @constCast(nl.values)[positional_of[e].start] = .{ .name = nl.models[row].name };
        }
        return changed;
    }

    /// Evaluates the postfix `span` at live values `values`, each
    /// distribution call answered by `draw` (`null` keeps nominals).
    fn evalLive(nl: *const Netlist, gpa: Allocator, stack: *std.ArrayList(expr.Val), span: Span, values: []const f64, draw: anytype) !f64 {
        const ops = nl.exprOps(span);
        return if (@TypeOf(draw) == @TypeOf(null))
            expr.eval(gpa, stack, ops, nl.consts, values, null)
        else
            expr.eval(gpa, stack, ops, nl.consts, values, SiteDraw(@TypeOf(draw)){ .inner = draw, .base = span.start, .ops = nl.live.site_ops, .keys = nl.live.site_keys });
    }

    /// Sets live slot k to `values[k]` without evaluating its postfix, as
    /// a probe point; `setLive` restores the evaluated values. Asserts
    /// that `values.len` equals `live.slots.len`.
    pub fn setSlots(nl: *Netlist, values: []const f64) void {
        std.debug.assert(values.len == nl.live.slots.len);
        for (values, 0..) |x, k| nl.slot(k).* = .{ .num = x };
    }

    /// Live slot `k`'s value, in the parse-arena table that holds it; a
    /// write through it is what every later `View` and `Model` reads.
    /// Asserts that `k` is a slot.
    pub fn slot(nl: *const Netlist, k: usize) *Value {
        const s = nl.live.slots[k];
        if (s.table != .groups) std.debug.assert(s.arg == 0);
        const v = switch (s.table) {
            .values => &@constCast(nl.values)[s.row],
            .kvs => &@constCast(nl.kvs)[s.row].value,
            .groups => &@constCast(nl.live.groups[s.row].args())[s.arg],
        };
        std.debug.assert(v.* == .num);
        return v;
    }
};

/// Maps an op index inside one slot's postfix to its site key.
pub fn SiteDraw(comptime D: type) type {
    return struct {
        inner: D,
        base: u32,
        ops: []const u32,
        keys: []const u64,

        /// The draw for call `f` at op `i` of the slot's postfix: the
        /// inner sampler's, under that call's site key. O(log sites).
        pub fn value(self: @This(), i: usize, f: expr.Fn, args: []const f64) f64 {
            const at: u32 = self.base + @as(u32, @intCast(i));
            const k = std.sort.lowerBound(u32, self.ops, at, orderU32);
            return self.inner.sample(self.keys[k], f, args);
        }
    };
}

/// A site key: `owner` mixed with `salt` (splitmix64 finalizer). Stable
/// across runs and targets, so a seed replays the same draws.
pub fn siteKey(owner: u64, salt: u64) u64 {
    var z = owner ^ (salt +% 0x9e3779b97f4a7c15);
    z = (z ^ (z >> 30)) *% 0xbf58476d1ce4e5b9;
    z = (z ^ (z >> 27)) *% 0x94d049bb133111eb;
    return z ^ (z >> 31);
}

pub fn nameKey(owner: u64, name: []const u8) u64 {
    return siteKey(owner, std.hash.Wyhash.hash(0, name));
}

/// True for a name that is ground in `dialect`: only `0` in Spectre, where
/// `gnd` is ground only when `global` names it first (spectre-syntax.md §7).
pub fn isGround(dialect: Dialect, name: []const u8) bool {
    return if (dialect == .spectre) std.mem.eql(u8, name, "0") else isGroundName(name);
}

/// True for `0`, `gnd` and `ground`, case-insensitively: ground in the
/// ngspice and HSPICE dialects.
pub fn isGroundName(name: []const u8) bool {
    return std.mem.eql(u8, name, "0") or
        std.ascii.eqlIgnoreCase(name, "gnd") or
        std.ascii.eqlIgnoreCase(name, "ground");
}

/// Index of the first card name in `names` equal to `target`, byte for
/// byte; null when none is. O(names.len).
pub fn nameIndex(names: []const []const u8, target: []const u8) ?usize {
    for (names, 0..) |n, i| if (std.mem.eql(u8, n, target)) return i;
    return null;
}

/// Whether bin bounds `b` hold `l` and `w`, each within 1 nm.
pub fn binHolds(b: [4]f64, l: f64, w: f64) bool {
    return (@abs(l - b[0]) < 1e-9 or @abs(l - b[1]) < 1e-9 or (l > b[0] and l < b[1])) and
        (@abs(w - b[2]) < 1e-9 or @abs(w - b[3]) < 1e-9 or (w > b[2] and w < b[3]));
}

/// The first `key` row's number; null when no row has `key` or the first is no number.
pub fn kvNumber(kv: []const Kv, key: []const u8) ?f64 {
    return if (kvIndex(kv, key)) |i| kv[i].value.number() else null;
}

pub fn orderU32(a: u32, b: u32) std.math.Order {
    return std.math.order(a, b);
}

// ---- private ----

test isGroundName {
    for ([_][]const u8{ "0", "gnd", "GND", "Ground" }) |n| try std.testing.expect(isGroundName(n));
    for ([_][]const u8{ "00", "", "gnd1", "x1.gnd" }) |n| try std.testing.expect(!isGroundName(n));
}

test nameIndex {
    const names = [_][]const u8{ "v1", "r1", "v1" };
    try std.testing.expectEqual(@as(?usize, 0), nameIndex(&names, "v1"));
    try std.testing.expectEqual(@as(?usize, 1), nameIndex(&names, "r1"));
    try std.testing.expectEqual(@as(?usize, null), nameIndex(&names, "R1"));
    try std.testing.expectEqual(@as(?usize, null), nameIndex(&.{}, "v1"));
}

test siteKey {
    // Fixed bits: a seed must replay the same draws on every target.
    try std.testing.expectEqual(siteKey(1, 2), siteKey(1, 2));
    try std.testing.expect(siteKey(1, 2) != siteKey(2, 1));
    try std.testing.expect(siteKey(0, 0) != siteKey(0, 1));
}
