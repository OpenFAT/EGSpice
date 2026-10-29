//! HSPICE `.alter`. In: a deck's text. Out: one full source per run, the
//! deck up to the first `.alter` and then each block applied on top of the
//! runs before it.
//! Cites: HSPICE [SA] Ch.4 (.ALTER), [CR] .ALTER, .DEL LIB.
//!
//! Contents:
//! - `pub const Error = error{ OutOfMemory, InvalidAlter }`
//! - `pub fn splitAlters(arena: Allocator, src: []const u8) Error![]const []const u8`

const std = @import("std");
const assert = std.debug.assert;
const mem = std.mem;
const Allocator = std.mem.Allocator;
const source_search = @import("source_search.zig");
const Fields = source_search.Fields;

/// Every way `splitAlters` fails; partial runs stay in the arena.
pub const Error = error{
    OutOfMemory,
    /// A nameless `.model`, `.subckt` or `.lib`, a `.del` other than
    /// `.del lib`, or a `.del` that matches no `.lib` card.
    InvalidAlter,
};

/// One logical card of the split: its text (continuations included) and
/// what a block's card of the same key replaces.
const Card = struct { text: []const u8, key: []const u8 };

/// The runs of a deck: the deck up to the first `.alter`, then one full
/// source per `.alter` block. Blocks are cumulative, as HSPICE reads the
/// input again from the previous run's netlist: block k applies to the deck
/// as blocks 1..k-1 left it. In a block, an element card replaces the
/// top-level card of the same name, `.model` the model of that name,
/// `.subckt ... .ends` the subcircuit of that name, and `.lib file sec`
/// the `.lib` line naming the same file; `.del lib file sec` removes one.
/// Any other card is appended (`.param`: the last definition wins); a blank
/// line is no card and is dropped from the later runs. A deck without
/// `.alter` is returned as its single run, unchanged. Runs borrow `src` or
/// live in `arena`.
pub fn splitAlters(arena: Allocator, src: []const u8) Error![]const []const u8 {
    // Most decks have no `.alter`: borrow `src` whole and index nothing.
    if (!hasAlter(src)) return try arena.dupe([]const u8, &.{src});
    var cards: std.ArrayList(Card) = .empty;
    var runs: std.ArrayList([]const u8) = .empty;
    var it = mem.splitScalar(u8, src, '\n');
    const title = it.first();
    var block: ?std.ArrayList(Card) = null;
    var base_end: usize = src.len;
    var pending: ?Card = null;
    while (true) {
        const at = it.index orelse src.len;
        const next = it.next();
        const line = next orelse "";
        const t = mem.trim(u8, line, " \t\r");
        if (next != null and t.len > 0 and t[0] == '+' and pending != null) {
            pending.?.text = src[@intFromPtr(pending.?.text.ptr) - @intFromPtr(src.ptr) .. at + line.len];
            continue;
        }
        // A blank line is no card, and a `+` after it still continues the
        // card before it.
        if (next != null and t.len == 0) continue;
        if (pending) |c| {
            if (block) |*b| try b.append(arena, c) else try cards.append(arena, c);
            pending = null;
        }
        if (next == null) break;
        const head = source_search.firstWord(t);
        if (std.ascii.eqlIgnoreCase(head, ".alter") or std.ascii.eqlIgnoreCase(head, ".end")) {
            if (block) |b| {
                try apply(arena, &cards, b.items);
                try runs.append(arena, try render(arena, title, cards.items));
            } else base_end = at;
            block = null;
            if (std.ascii.eqlIgnoreCase(head, ".end")) break;
            block = .empty;
            continue;
        }
        pending = .{ .text = src[at .. at + line.len], .key = try keyOf(arena, t) };
    }
    if (block) |b| {
        try apply(arena, &cards, b.items);
        try runs.append(arena, try render(arena, title, cards.items));
    }
    if (runs.items.len == 0) return try arena.dupe([]const u8, &.{src});
    try runs.insert(arena, 0, src[0..base_end]);
    assert(mem.startsWith(u8, src, runs.items[0]));
    return runs.items;
}

// ---- private ----

/// Whether any line after the title starts with `.alter`.
fn hasAlter(src: []const u8) bool {
    var it = mem.splitScalar(u8, src, '\n');
    _ = it.first();
    while (it.next()) |line| {
        if (std.ascii.eqlIgnoreCase(source_search.firstWord(mem.trim(u8, line, " \t\r")), ".alter")) return true;
    }
    return false;
}

/// `e:<name>`, `model:<name>`, `subckt:<name>`, `lib:<file>`, `del:<file>`
/// or "" for a card that only appends.
fn keyOf(arena: Allocator, t: []const u8) Error![]const u8 {
    if (t.len == 0 or t[0] == '*') return "";
    var fields = Fields.init(t);
    const head = fields.next() orelse return "";
    const kind: []const u8 = if (std.ascii.isAlphabetic(head[0])) "e" else if (std.ascii.eqlIgnoreCase(head, ".model")) "model" else if (std.ascii.eqlIgnoreCase(head, ".subckt")) "subckt" else if (std.ascii.eqlIgnoreCase(head, ".lib")) "lib" else if (std.ascii.eqlIgnoreCase(head, ".del")) "del" else if (std.ascii.eqlIgnoreCase(head, ".ends")) "ends" else return "";
    if (mem.eql(u8, kind, "e") or mem.eql(u8, kind, "ends")) return std.ascii.allocLowerString(arena, try arena.print("{s}:{s}", .{ kind, head }));
    if (mem.eql(u8, kind, "del")) {
        const what = fields.next() orelse return error.InvalidAlter;
        if (!std.ascii.eqlIgnoreCase(what, "lib")) return error.InvalidAlter;
    }
    const name = (source_search.word(&fields) catch return error.InvalidAlter) orelse return error.InvalidAlter;
    return std.ascii.allocLowerString(arena, try arena.print("{s}:{s}", .{ kind, name }));
}

/// The span of cards a `.subckt` card at `i` opens, through its `.ends`.
fn unitLen(cards: []const Card, i: usize) usize {
    assert(i < cards.len);
    if (!mem.startsWith(u8, cards[i].key, "subckt:")) return 1;
    var k = i + 1;
    while (k < cards.len and !mem.startsWith(u8, cards[k].key, "ends:")) : (k += 1) {}
    return @min(k + 1, cards.len) - i;
}

/// Applies one `.alter` block to `cards` in place.
fn apply(arena: Allocator, cards: *std.ArrayList(Card), block: []const Card) Error!void {
    var i: usize = 0;
    while (i < block.len) {
        const c = block[i];
        const n = unitLen(block, i);
        assert(n >= 1);
        defer i += n;
        const target = if (mem.startsWith(u8, c.key, "del:")) try arena.print("lib:{s}", .{c.key[4..]}) else c.key;
        const replaceable = target.len != 0 and !mem.startsWith(u8, target, "ends:");
        // Top-level cards only: skip subcircuit bodies.
        var k: usize = 0;
        const found: ?usize = while (replaceable and k < cards.items.len) {
            if (mem.eql(u8, cards.items[k].key, target)) break k;
            k += unitLen(cards.items, k);
        } else null;
        if (found) |at| {
            const old = unitLen(cards.items, at);
            const new: []const Card = if (target.ptr != c.key.ptr) &.{} else block[i..][0..n];
            try cards.replaceRange(arena, at, old, new);
        } else if (target.ptr == c.key.ptr) {
            try cards.appendSlice(arena, block[i..][0..n]);
        } else return error.InvalidAlter;
    }
}

fn render(arena: Allocator, title: []const u8, cards: []const Card) Error![]const u8 {
    var out: std.ArrayList(u8) = .empty;
    try out.appendSlice(arena, title);
    try out.append(arena, '\n');
    for (cards) |c| {
        try out.appendSlice(arena, c.text);
        try out.append(arena, '\n');
    }
    try out.appendSlice(arena, ".end\n");
    assert(mem.endsWith(u8, out.items, ".end\n"));
    return out.items;
}

test keyOf {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    try std.testing.expectEqualStrings("e:r1", try keyOf(a, "R1 a b 1"));
    try std.testing.expectEqualStrings("model:nm", try keyOf(a, ".model NM nmos"));
    try std.testing.expectEqualStrings("del:f", try keyOf(a, ".del lib 'f' tt"));
    try std.testing.expectEqualStrings("", try keyOf(a, ".param a=1"));
    try std.testing.expectEqualStrings("", try keyOf(a, "* c"));
    try std.testing.expectError(error.InvalidAlter, keyOf(a, ".del x"));
    try std.testing.expectError(error.InvalidAlter, keyOf(a, ".model"));
}
