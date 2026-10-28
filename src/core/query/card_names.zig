//! Device card names for result labels: the builder's `CardRef` rows
//! (type, per-type index, name) in, a dense per-type name column out, so a
//! label lookup is two loads instead of a scan (#175).
//! Cites: none.
//!
//! Contents:
//! - `pub const CardRef = struct { type: DeviceType, index: u32, name: []const u8 }`
//! - `pub const CardNames = struct`, `CardNames.empty`
//! - `CardNames.init(arena: Allocator, cards: []const CardRef) Allocator.Error!CardNames`
//! - `CardNames.get(self: CardNames, t: DeviceType, index: u32) ?[]const u8`

const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;
const DeviceType = @import("../ids.zig").DeviceType;

/// The card name of one device instance.
pub const CardRef = struct {
    type: DeviceType,
    /// Instance index within `type`.
    index: u32,
    name: []const u8,
};

/// The `CardRef` table as a dense per-type name column, so a label is one
/// load: instance `index` of type `t` is `names[starts[t] + index]`, `""`
/// for an instance no card names (an internal add). The builder hands out
/// `index` densely per type, so a type's rows run to its highest named
/// index. A view: it borrows the arena it is built on and every card name.
pub const CardNames = struct {
    /// Per type, its first row in `names`; types + 1 entries, ascending.
    starts: []const u32 = &.{},
    names: []const []const u8 = &.{},

    pub const empty: CardNames = .{};

    /// The table over `cards`; where a (type, index) pair repeats, the
    /// first row wins. Caller owns the result through `arena`. Asserts that
    /// every card has a type and a name, and that the rows fit u32.
    pub fn init(arena: Allocator, cards: []const CardRef) Allocator.Error!CardNames {
        var n_types: usize = 0;
        for (cards) |c| {
            assert(c.type != .unset);
            assert(c.name.len != 0);
            n_types = @max(n_types, @as(usize, @backingInt(c.type)) + 1);
        }
        const starts = try arena.alloc(u32, n_types + 1);
        @memset(starts, 0);
        // One past each type's highest index at starts[t + 1], then the prefix sum.
        for (cards) |c| {
            const end = &starts[@as(usize, @backingInt(c.type)) + 1];
            end.* = @max(end.*, c.index + 1);
        }
        for (1..starts.len) |t| starts[t] += starts[t - 1];
        const names = try arena.alloc([]const u8, starts[n_types]);
        @memset(names, "");
        for (cards) |c| {
            const name = &names[starts[@backingInt(c.type)] + c.index];
            if (name.len == 0) name.* = c.name;
        }
        assert(starts[0] == 0);
        assert(starts[n_types] == names.len);
        return .{ .starts = starts, .names = names };
    }

    /// The card name of instance `index` of type `t`, if a card names it.
    pub fn get(self: CardNames, t: DeviceType, index: u32) ?[]const u8 {
        const ti: usize = @backingInt(t);
        if (ti + 1 >= self.starts.len) return null;
        const row = @as(u64, self.starts[ti]) + index;
        if (row >= self.starts[ti + 1]) return null;
        assert(row < self.names.len);
        const name = self.names[@intCast(row)];
        return if (name.len == 0) null else name;
    }
};

// ---- private ----

/// The linear scan `CardNames` replaced (#175): the first row matching
/// (type, index). The differential oracle of the dense table.
fn scanName(cards: []const CardRef, t: DeviceType, index: u32) ?[]const u8 {
    for (cards) |c| if (c.index == index and c.type == t) return c.name;
    return null;
}

test "CardNames keys on type and index together" {
    const t = std.testing;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const r: DeviceType = @fromBackingInt(0);
    const c: DeviceType = @fromBackingInt(1);
    const cards = [_]CardRef{ .{ .type = r, .index = 0, .name = "r1" }, .{ .type = c, .index = 0, .name = "c1" } };
    const names = try CardNames.init(arena.allocator(), &cards);
    try t.expectEqualStrings("c1", names.get(c, 0).?);
    try t.expectEqualStrings("r1", names.get(r, 0).?);
    try t.expectEqual(null, names.get(r, 1));
    try t.expectEqual(null, names.get(.unset, 0));
    try t.expectEqual(null, CardNames.empty.get(r, 0));
    try t.expectEqual(null, (try CardNames.init(arena.allocator(), &.{})).get(r, 0));
}

test "CardNames answers as the linear scan does over a shuffled table" {
    const t = std.testing;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var prng = std.Random.DefaultPrng.init(175);
    const rand = prng.random();
    for (0..50) |_| {
        // Builder order: types interleaved, indices ascending per type,
        // some instances unnamed, the odd repeated (type, index).
        const n_types = rand.intRangeAtMost(u16, 1, 6);
        var cards: std.ArrayList(CardRef) = .empty;
        var counts: [6]u32 = @splat(0);
        for (0..rand.uintLessThan(usize, 200)) |k| {
            const ti = rand.uintLessThan(u16, n_types);
            const index = counts[ti];
            counts[ti] += 1;
            if (rand.uintLessThan(u8, 4) == 0) continue;
            const name = try a.print("d{d}", .{k});
            try cards.append(a, .{ .type = @fromBackingInt(ti), .index = index, .name = name });
            if (index > 0 and rand.uintLessThan(u8, 16) == 0)
                try cards.append(a, .{ .type = @fromBackingInt(ti), .index = index - 1, .name = "dup" });
        }
        rand.shuffle(CardRef, cards.items);
        const names = try CardNames.init(a, cards.items);
        for (0..n_types + 2) |ti| for (0..counts[@min(ti, 5)] + 3) |index| {
            const ty: DeviceType = @fromBackingInt(@intCast(ti));
            const want = scanName(cards.items, ty, @intCast(index));
            const got = names.get(ty, @intCast(index));
            try t.expectEqual(want == null, got == null);
            if (want) |w| try t.expectEqualStrings(w, got.?);
        };
    }
}

test "CardNames.init fails cleanly at every allocation" {
    const cards = [_]CardRef{ .{ .type = @fromBackingInt(2), .index = 1, .name = "r2" }, .{ .type = @fromBackingInt(0), .index = 0, .name = "v1" } };
    try std.testing.checkAllAllocationFailures(std.testing.allocator, struct {
        fn run(gpa: std.mem.Allocator, cs: []const CardRef) !void {
            var arena = std.heap.ArenaAllocator.init(gpa);
            defer arena.deinit();
            const names = try CardNames.init(arena.allocator(), cs);
            try std.testing.expectEqualStrings("r2", names.get(@fromBackingInt(2), 1).?);
            try std.testing.expectEqual(null, names.get(@fromBackingInt(1), 0));
        }
    }.run, .{&cards});
}
