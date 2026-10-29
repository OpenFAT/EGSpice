//! Physical lines, the layer under every dialect's logical lines. In: a
//! deck's bytes. Out: lowercased bytes with their newline count, one
//! physical line at a time, and the joined text of a continued statement.
//! Cites: none (house line layer; the dialect rules live in lines_*.zig).
//!
//! Contents:
//! - `pub fn normalize(dst: []u8, src: []const u8) usize`
//! - `pub fn nextPhysicalLine(rest: *[]const u8) ?[]const u8`
//! - `pub fn startsStatement(rest: []const u8, comptime keep: []const u8) bool`
//! - `pub fn join(arena: Allocator, joined: *?std.ArrayList(u8), head: []const u8, sep: []const u8, piece: []const u8) Allocator.Error!void`

const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;

/// Copies `src` lowercased into `dst` (at least as long) and returns its
/// newline count, in one pass over 32-byte blocks.
// ponytail: hand vectors on purpose; a plain map plus `countScalar`
// vectorizes too, but measured 0.07 against 0.05 ns/byte (two passes).
pub fn normalize(dst: []u8, src: []const u8) usize {
    assert(dst.len >= src.len);
    const V = @Vector(32, u8);
    var lines: usize = 0;
    var i: usize = 0;
    while (i + 32 <= src.len) : (i += 32) {
        const v: V = src[i..][0..32].*;
        const upper = (v >= @as(V, @splat('A'))) & (v <= @as(V, @splat('Z')));
        dst[i..][0..32].* = v | @select(u8, upper, @as(V, @splat(0x20)), @as(V, @splat(0)));
        lines += @popCount(@as(u32, @bitCast(v == @as(V, @splat('\n')))));
    }
    for (src[i..], dst[i..src.len]) |c, *d| d.* = c | @as(u8, @intFromBool(c -% 'A' < 26)) << 5;
    return lines + std.mem.countScalar(u8, src[i..], '\n');
}

/// Appends `sep` and a continuation piece to a logical line, copying
/// `head` on first use.
pub fn join(arena: Allocator, joined: *?std.ArrayList(u8), head: []const u8, sep: []const u8, piece: []const u8) Allocator.Error!void {
    if (joined.* == null) {
        joined.* = .empty;
        try joined.*.?.appendSlice(arena, head);
    }
    try joined.*.?.appendSlice(arena, sep);
    try joined.*.?.appendSlice(arena, piece);
}

/// The next physical line without its `\r`, advancing `rest`.
// ponytail: all dialects share physical lines; comment and continuation rules stay local.
pub fn nextPhysicalLine(rest: *[]const u8) ?[]const u8 {
    const src = rest.*;
    if (src.len == 0) return null;
    const nl = std.mem.indexOfScalar(u8, src, '\n') orelse src.len;
    const line = std.mem.trimEnd(u8, src[0..nl], "\r");
    rest.* = if (nl == src.len) src[nl..] else src[nl + 1 ..];
    assert(rest.len < src.len);
    return line;
}

/// Whether the next physical line in `rest` starts a new statement for
/// certain: its first byte past blanks is neither a line end nor one of
/// `keep`, the bytes after which a continuation check must read the line.
/// Lets that check leave the line unread instead of splitting it twice.
pub fn startsStatement(rest: []const u8, comptime keep: []const u8) bool {
    var k: usize = 0;
    while (k < rest.len and (rest[k] == ' ' or rest[k] == '\t')) k += 1;
    if (k == rest.len) return false;
    return switch (rest[k]) {
        '\r', '\n' => false,
        else => |c| std.mem.indexOfScalar(u8, keep, c) == null, // else: any other byte is text
    };
}

test normalize {
    var buf: [40]u8 = undefined;
    const src = "R1 A b\nC1 X 0 1P\nV1 IN 0 PWL(0 0 1N 1)\n";
    try std.testing.expectEqual(3, normalize(&buf, src));
    try std.testing.expectEqualStrings("r1 a b\nc1 x 0 1p\nv1 in 0 pwl(0 0 1n 1)\n", buf[0..src.len]);
    var rest: []const u8 = "a\r\n\n b";
    try std.testing.expectEqualStrings("a", nextPhysicalLine(&rest).?);
    try std.testing.expect(!startsStatement(rest, "+"));
    try std.testing.expectEqualStrings("", nextPhysicalLine(&rest).?);
    try std.testing.expect(startsStatement(rest, "+"));
    try std.testing.expectEqualStrings(" b", nextPhysicalLine(&rest).?);
    try std.testing.expectEqual(null, nextPhysicalLine(&rest));
}
