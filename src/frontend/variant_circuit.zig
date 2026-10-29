//! What tells two variant circuits apart. In: two netlists or two frozen
//! circuits. Out: whether parameter writes alone can turn one into the
//! other (same topology, same cards, same kept rows), and the nominal
//! circuit rebuilt from a netlist's current live values.
//! Cites: dev/analysis/variants.md (rebuild rules); #130 (zero crossings).
//!
//! Contents:
//! - `pub const Kept`
//! - `pub fn circuitOf(lib: *const device.Library, gpa: std.mem.Allocator, scratch: std.mem.Allocator, nl: *const Netlist) !device.Circuit`
//! - `pub fn sameTopology(a: *const device.Circuit, b: *const device.Circuit) bool`
//! - `pub fn sameCards(a: *const Netlist, b: *const Netlist) bool`
//! - `pub fn crossesZero(a: f64, b: f64) bool`
//! - `pub fn signClass(x: f64) u8`
//! - `pub fn refKey(a: std.mem.Allocator, t: core.DeviceType, name: []const u8) ![]const u8`

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

/// What a circuit holds that no parameter write reaches, read off the
/// netlist it was built from: each binned card's model row (bins differ in
/// which model parameters they give) and each constant-pool row a kept
/// expression reads (a B source's tape is not `ParamRef` storage). A point
/// that moves any of them needs a circuit of its own.
pub const Kept = struct {
    bins: []const u32 = &.{},
    /// `Live.pool_rows` then `Live.site_rows`, `none` rows read as 0.
    pool: []const f64 = &.{},

    /// The netlist's current values, allocated in `gpa`.
    fn of(gpa: std.mem.Allocator, nl: *const Netlist) !Kept {
        const live = nl.live;
        const bins = try gpa.alloc(u32, live.bins.edges.len);
        const model_of = nl.graph.edges.items(.model);
        for (bins, live.bins.edges) |*row, e| row.* = model_of[e];
        const pool = try gpa.alloc(f64, live.pool_rows.len + live.site_rows.len);
        for (pool[0..live.pool_rows.len], live.pool_rows) |*v, row| v.* = if (row == netlist.none) 0 else nl.consts[row];
        for (pool[live.pool_rows.len..], live.site_rows) |*v, row| v.* = nl.consts[row];
        return .{ .bins = bins, .pool = pool };
    }

    /// Whether `nl` at its current values still holds `k` bit for bit.
    fn holds(k: Kept, nl: *const Netlist) bool {
        const live = nl.live;
        std.debug.assert(k.bins.len == live.bins.edges.len);
        std.debug.assert(k.pool.len == live.pool_rows.len + live.site_rows.len);
        const model_of = nl.graph.edges.items(.model);
        for (k.bins, live.bins.edges) |row, e| if (model_of[e] != row) return false;
        // By value: a `0*agauss(...)` row is -0 or +0 by the draw's sign,
        // the same number to the tape.
        for (k.pool[0..live.pool_rows.len], live.pool_rows) |v, row| if (row != netlist.none and v != nl.consts[row]) return false;
        for (k.pool[live.pool_rows.len..], live.site_rows) |v, row| if (v != nl.consts[row]) return false;
        return true;
    }
};

/// The frozen circuit `nl` binds to, as `prepare.build` constructs it, with
/// `nl`'s live values as they stand. Caller owns the circuit and releases
/// it with `deinit`; `scratch` holds the construction tables.
pub fn circuitOf(lib: *const device.Library, gpa: std.mem.Allocator, scratch: std.mem.Allocator, nl: *const Netlist) !device.Circuit {
    const deck_opts = try analyses.deckOptions(nl.deck.config, nl.deck.dialect);
    var b = try builder.Builder.init(gpa, lib);
    var compiled = false;
    errdefer if (!compiled) b.deinit();
    b.nom_temp_c = deck_opts.tnom_c;
    b.tol = deck_opts.tol;
    var nb = try builder.NetBuilder.init(scratch, &b, nl.*);
    try nb.build();
    var perm: ?[]const u32 = null;
    const c = try b.compilePerm(&perm);
    compiled = true;
    return c;
}

/// Same matrix pattern, node count and devices per type: parameter writes
/// can turn `a` into `b`.
pub fn sameTopology(a: *const device.Circuit, b: *const device.Circuit) bool {
    if (a.n != b.n or a.nnz != b.nnz or a.batches.len != b.batches.len) return false;
    if (!std.mem.eql(u32, a.col_ptr, b.col_ptr) or !std.mem.eql(u32, a.row_idx, b.row_idx)) return false;
    if (!std.mem.eql(bool, a.current_row, b.current_row)) return false;
    for (a.batches, b.batches, a.batch_types, b.batch_types) |x, y, tx, ty|
        if (tx != ty or x.count != y.count) return false;
    return true;
}

/// Same cards on the same nets, and the same analyses: an `.alter` run
/// that only changed values.
pub fn sameCards(a: *const Netlist, b: *const Netlist) bool {
    if (a.deviceCount() != b.deviceCount() or a.deck.analyses.len != b.deck.analyses.len or a.deck.config.len != b.deck.config.len) return false;
    for (a.deck.analyses, b.deck.analyses) |x, y| if (!std.mem.eql(u8, x.line, y.line)) return false;
    for (a.deck.config, b.deck.config) |x, y| if (!std.mem.eql(u8, x.line, y.line)) return false;
    for (a.order, b.order) |ea, eb| {
        const x = a.device(ea);
        const y = b.device(eb);
        if (x.kind != y.kind or !std.mem.eql(u8, x.name, y.name) or x.pins.len != y.pins.len) return false;
        for (x.pins, y.pins) |px, py| if (!std.mem.eql(u8, a.netName(px), b.netName(py))) return false;
    }
    return true;
}

/// Whether a slot that held `a` reaches, leaves or crosses zero at `b`,
/// which may collapse a node: only a rebuild can tell.
pub fn crossesZero(a: f64, b: f64) bool {
    return signClass(a) != signClass(b);
}

/// 0 for zero, 1 for negative, 2 otherwise (NaN included): what
/// `crossesZero` compares.
pub fn signClass(x: f64) u8 {
    return if (x == 0) 0 else if (x < 0) 1 else 2;
}

pub fn refKey(a: std.mem.Allocator, t: core.DeviceType, name: []const u8) ![]const u8 {
    return std.fmt.allocPrint(a, "{d}:{s}", .{ @backingInt(t), name });
}

// ---- private ----

test crossesZero {
    // (from, to, crosses): reaching, leaving and crossing zero count; -0 is 0.
    for ([_]struct { f64, f64, bool }{
        .{ 1, 2, false },           .{ -1, -2, false }, .{ 0, -0.0, false }, .{ 0, 0, false },
        .{ 0, 1e-300, true },       .{ 1, 0, true },    .{ 1, -1, true },    .{ -1, 1, true },
        .{ 5e-324, -5e-324, true },
    }) |c| try std.testing.expectEqual(c[2], crossesZero(c[0], c[1]));
}
