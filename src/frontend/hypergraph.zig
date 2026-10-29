//! Nets-by-devices hypergraph. In: nets, then devices with their pin lists,
//! appended in card order. Out: a frozen device-major CSR `Graph` whose net
//! and device payloads are SoA columns.
//! Cites: none (house layout, after cktImg src/csr.zig, MIT, Omar El-Sawy).
//!
//! A device's pin list keeps its order (the terminal role) and its repeats
//! (a bulk tied to its source).
//!
//! Contents:
//! - `pub const VertexId = enum(u32)`
//! - `pub fn from(i: usize) VertexId`
//! - `pub fn index(self: VertexId) u32`
//! - `pub const EdgeId = enum(u32)`
//! - `pub fn from(i: usize) EdgeId`
//! - `pub fn index(self: EdgeId) u32`
//! - `pub const Span = struct`
//! - `pub const Net = struct`
//! - `pub const Device = struct`
//! - `pub const Error = error{...} || Allocator.Error`
//! - `pub const Builder = struct`
//! - `pub fn init(gpa: Allocator) Allocator.Error!Builder`
//! - `pub fn ensureTotalCapacity(self: *Builder, gpa: Allocator, vertex_count: usize, edge_count: usize, pin_count: usize) Allocator.Error!void`
//! - `pub fn addVertex(self: *Builder, gpa: Allocator, net: Net) Error!VertexId`
//! - `pub fn addEdge(self: *Builder, gpa: Allocator, dev: Device, pins: []const VertexId) Error!EdgeId`
//! - `pub fn finish(self: *Builder, gpa: Allocator) Allocator.Error!Graph`
//! - `pub const Graph = struct`
//! - `pub fn vertexCount(self: Graph) u32`
//! - `pub fn edgeCount(self: Graph) u32`
//! - `pub fn pins(self: Graph, e: EdgeId) []const VertexId`

const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;
const Name = @import("core").Name;

/// A net: a row of the vertex columns.
pub const VertexId = enum(u32) {
    _,

    /// The id of row `i`. Asserts that `i` fits in a u32.
    pub inline fn from(i: usize) VertexId {
        return @fromBackingInt(@as(u32, @intCast(i)));
    }

    /// The row of this net in the vertex columns.
    pub inline fn index(self: VertexId) u32 {
        return @backingInt(self);
    }
};

/// A device: a row of the edge columns and of `Graph.offsets`.
pub const EdgeId = enum(u32) {
    _,

    /// The id of row `i`. Asserts that `i` fits in a u32.
    pub inline fn from(i: usize) EdgeId {
        return @fromBackingInt(@as(u32, @intCast(i)));
    }

    /// The row of this device in the edge columns.
    pub inline fn index(self: EdgeId) u32 {
        return @backingInt(self);
    }
};

/// Rows `start..start+len` of a flat table.
pub const Span = struct { start: u32 = 0, len: u32 = 0 };

/// Vertex payload: the net's flattened name in the netlist's pool.
pub const Net = struct { name: Name };

/// Edge payload. `kind` is the lowercase card letter.
pub const Device = struct {
    kind: u8,
    /// Flattened name: `r.x1.r1` for `r1` inside instance `x1`.
    name: Name,
    /// `.model` row the first positional names, `maxInt(u32)` otherwise.
    model: u32,
    /// Rows of the netlist's positional values.
    positional: Span,
    /// Rows of the netlist's `key=value` pairs.
    kv: Span,
    /// Subcircuit expansion ordinal, 0 at top level.
    subckt_instance: u32,
};

comptime {
    // SoA device rows: 29 bytes over six columns, no padding.
    var row: usize = 0;
    for (@typeInfo(Device).@"struct".field_types) |t| row += @sizeOf(t);
    assert(row == 29);
    assert(@sizeOf(Net) == 4);
    assert(@sizeOf(Span) == 8);
}

/// Why `addVertex` or `addEdge` refused. The builder is unchanged after any.
pub const Error = error{
    /// A pin is not a net added so far.
    InvalidVertex,
    /// The net count would pass maxInt(u32).
    TooManyVertices,
    /// The device count would pass maxInt(u32).
    TooManyEdges,
    /// The pin count of all devices would pass maxInt(u32).
    TooManyIncidences,
} || Allocator.Error;

/// Append-only tables; `finish` freezes them into a `Graph`.
/// Invariant: `offsets.len == edges.len + 1` and `offsets` ends at `members.len`.
pub const Builder = struct {
    vertices: std.MultiArrayList(Net) = .empty,
    edges: std.MultiArrayList(Device) = .empty,
    offsets: std.ArrayList(u32) = .empty,
    members: std.ArrayList(VertexId) = .empty,

    /// An empty builder; its tables allocate in `gpa`.
    pub fn init(gpa: Allocator) Allocator.Error!Builder {
        var b: Builder = .{};
        try b.offsets.append(gpa, 0);
        assert(b.offsets.items.len == 1);
        return b;
    }

    /// Reserves room for the given totals so the adds below do not
    /// reallocate. Invalidates slices of every column if it grows.
    pub fn ensureTotalCapacity(self: *Builder, gpa: Allocator, vertex_count: usize, edge_count: usize, pin_count: usize) Allocator.Error!void {
        try self.vertices.ensureTotalCapacity(gpa, vertex_count);
        try self.edges.ensureTotalCapacity(gpa, edge_count);
        try self.offsets.ensureTotalCapacity(gpa, edge_count + 1);
        try self.members.ensureTotalCapacity(gpa, pin_count);
        assert(self.offsets.capacity > edge_count);
    }

    /// Appends a net; ids are dense from 0 in call order.
    /// Invalidates slices of the vertex columns if they grow.
    pub fn addVertex(self: *Builder, gpa: Allocator, net: Net) Error!VertexId {
        if (self.vertices.len >= std.math.maxInt(u32)) return error.TooManyVertices;
        const id: VertexId = .from(self.vertices.len);
        try self.vertices.append(gpa, net);
        assert(id.index() + 1 == self.vertices.len);
        return id;
    }

    /// Appends a device with `pins` in the given order, repeats kept; no
    /// pins is a valid device. On error the builder is unchanged.
    /// Invalidates slices of the edge columns if they grow.
    pub fn addEdge(self: *Builder, gpa: Allocator, dev: Device, pins: []const VertexId) Error!EdgeId {
        assert(self.offsets.items.len == self.edges.len + 1);
        for (pins) |v| if (v.index() >= self.vertices.len) return error.InvalidVertex;
        if (self.edges.len >= std.math.maxInt(u32)) return error.TooManyEdges;
        if (self.members.items.len + pins.len > std.math.maxInt(u32)) return error.TooManyIncidences;
        try self.members.ensureUnusedCapacity(gpa, pins.len);
        try self.offsets.ensureUnusedCapacity(gpa, 1);
        try self.edges.ensureUnusedCapacity(gpa, 1);

        self.members.appendSliceAssumeCapacity(pins);
        const id: EdgeId = .from(self.edges.len);
        self.edges.appendAssumeCapacity(dev);
        self.offsets.appendAssumeCapacity(@intCast(self.members.items.len));
        assert(self.offsets.items.len == self.edges.len + 1);
        return id;
    }

    /// Consumes the builder. The graph's tables live in `gpa`; it has no
    /// `deinit`, since the frontend hands it its parse arena.
    pub fn finish(self: *Builder, gpa: Allocator) Allocator.Error!Graph {
        assert(self.offsets.items.len == self.edges.len + 1);
        const offsets = try self.offsets.toOwnedSlice(gpa);
        errdefer gpa.free(offsets);
        const g: Graph = .{
            .vertices = self.vertices,
            .edges = self.edges,
            .offsets = offsets,
            .members = try self.members.toOwnedSlice(gpa),
        };
        self.* = undefined;
        assert(g.offsets[g.offsets.len - 1] == g.members.len);
        return g;
    }
};

/// Frozen hypergraph. Device `e`'s pins are `members[offsets[e]..offsets[e + 1]]`.
pub const Graph = struct {
    vertices: std.MultiArrayList(Net),
    edges: std.MultiArrayList(Device),
    offsets: []const u32,
    members: []const VertexId,

    /// Number of nets.
    pub inline fn vertexCount(self: Graph) u32 {
        return @intCast(self.vertices.len);
    }

    /// Number of devices.
    pub inline fn edgeCount(self: Graph) u32 {
        return @intCast(self.edges.len);
    }

    /// Pins of `e`, in terminal order. Asserts that `e` is a device here.
    pub inline fn pins(self: Graph, e: EdgeId) []const VertexId {
        assert(e.index() < self.edges.len);
        return self.members[self.offsets[e.index()]..self.offsets[e.index() + 1]];
    }
};

// ---- private ----

fn testDevice(kind: u8) Device {
    return .{ .kind = kind, .name = @fromBackingInt(0), .model = 0, .positional = .{}, .kv = .{}, .subckt_instance = 0 };
}

test "pins keep order and repeats" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const gpa = arena.allocator();
    var b: Builder = try .init(gpa);
    for (0..5) |_| _ = try b.addVertex(gpa, .{ .name = @fromBackingInt(0) });
    const v: [5]VertexId = .{ .from(0), .from(1), .from(2), .from(3), .from(4) };
    _ = try b.addEdge(gpa, testDevice('r'), &.{ v[2], v[0], v[1] });
    _ = try b.addEdge(gpa, testDevice('m'), &.{ v[3], v[2], v[2], v[2] });
    try std.testing.expectError(error.InvalidVertex, b.addEdge(gpa, testDevice('c'), &.{.from(9)}));
    const g = try b.finish(gpa);
    try std.testing.expectEqual(2, g.edgeCount());
    try std.testing.expectEqual(5, g.vertexCount());
    try std.testing.expectEqualSlices(VertexId, &.{ v[2], v[0], v[1] }, g.pins(.from(0)));
    try std.testing.expectEqualSlices(VertexId, &.{ v[3], v[2], v[2], v[2] }, g.pins(.from(1)));
    try std.testing.expectEqual('m', g.edges.items(.kind)[1]);
}

test "addEdge is all-or-nothing under allocation failure" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var fail_index: usize = 0;
    while (true) : (fail_index += 1) {
        var b: Builder = try .init(arena.allocator());
        const v0 = try b.addVertex(arena.allocator(), .{ .name = @fromBackingInt(0) });
        const v1 = try b.addVertex(arena.allocator(), .{ .name = @fromBackingInt(1) });
        var failing = std.testing.FailingAllocator.init(arena.allocator(), .{ .fail_index = fail_index, .resize_fail_index = fail_index });
        if (b.addEdge(failing.allocator(), testDevice('r'), &.{ v1, v0, v1 })) |e| {
            try std.testing.expectEqual(0, e.index());
            break;
        } else |err| {
            try std.testing.expectEqual(error.OutOfMemory, err);
            try std.testing.expectEqual(0, b.edges.len);
            try std.testing.expectEqual(0, b.members.items.len);
            try std.testing.expectEqualSlices(u32, &.{0}, b.offsets.items);
        }
    }
    // An empty device and an empty graph are both valid.
    var b: Builder = try .init(arena.allocator());
    const g0 = try b.finish(arena.allocator());
    try std.testing.expectEqual(0, g0.edgeCount());
    try std.testing.expectEqualSlices(u32, &.{0}, g0.offsets);
    b = try .init(arena.allocator());
    _ = try b.addVertex(arena.allocator(), .{ .name = @fromBackingInt(0) });
    const e = try b.addEdge(arena.allocator(), testDevice('v'), &.{});
    const g = try b.finish(arena.allocator());
    try std.testing.expectEqual(0, g.pins(e).len);
}
