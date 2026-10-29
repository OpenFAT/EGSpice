//! The word list of a `.meas` or `.check` card. In: the card text after its
//! keyword, lowercased. Out: `Word` rows, `lhs` or `lhs=rhs`, borrowing the
//! text. Cites: ngspice com_measure2.c (measure_parse_line), HSPICE [CR]
//! .MEASURE and [SA] Ch.27 for `GOAL < v`.
//!
//! Contents:
//! - `pub const Word = struct`
//! - `pub const Error = error{ OutOfMemory, ParseError }`
//! - `pub fn words(arena: Allocator, text: []const u8) Error![]Word`
//! - `pub fn find(w: []const Word, key: []const u8) ?usize`
//! - `pub fn groupWords(arena: Allocator, group: []const u8) Error![]Word`

const std = @import("std");
const assert = std.debug.assert;
const mem = std.mem;
const Allocator = std.mem.Allocator;

/// `lhs`, or `lhs=rhs` with blanks around the `=` dropped. `lhs` is never
/// empty.
pub const Word = struct { lhs: []const u8, rhs: ?[]const u8 = null };

/// Every way `words` fails; nothing of the list is returned.
pub const Error = error{
    OutOfMemory,
    /// A leading `=` (no key) or a trailing `=` (no value).
    ParseError,
};

/// Blank- or comma-separated words; `{..}`, `'..'` and `(..)` stay whole,
/// and blanks around `=` join its sides. The words borrow `text`.
pub fn words(arena: Allocator, text: []const u8) Error![]Word {
    var out: std.ArrayList(Word) = .empty;
    var i: usize = 0;
    var pending_eq = false;
    while (i < text.len) {
        const c = text[i];
        if (c == ' ' or c == '\t' or c == ',') {
            i += 1;
            continue;
        }
        const start = i;
        var depth: u32 = 0;
        var quote = false;
        while (i < text.len) : (i += 1) {
            const b = text[i];
            if (b == '\'' or b == '"') quote = !quote;
            if (quote) continue;
            if (b == '{' or b == '(') depth += 1;
            if ((b == '}' or b == ')') and depth > 0) depth -= 1;
            if (depth == 0 and (b == ' ' or b == '\t' or b == ',')) break;
        }
        const tok = text[start..i];
        assert(tok.len > 0);
        const eq = mem.indexOfScalar(u8, tok, '=');
        if (tok[0] == '=' and out.items.len == 0) return error.ParseError;
        if (pending_eq or tok[0] == '=') {
            // `a =b`, `a = b` or `a= b`: this token completes the last word.
            const last = &out.items[out.items.len - 1];
            const value = if (tok[0] == '=') tok[1..] else tok;
            pending_eq = value.len == 0;
            // `goal > v`: the rhs runs from the operator to the value.
            const op = last.rhs orelse "";
            last.rhs = if (op.len != 0 and (op[0] == '<' or op[0] == '>')) text[@intFromPtr(op.ptr) - @intFromPtr(text.ptr) .. i] else value;
            continue;
        }
        if ((tok[0] == '<' or tok[0] == '>') and out.items.len > 0 and out.items[out.items.len - 1].rhs == null and
            mem.eql(u8, out.items[out.items.len - 1].lhs, "goal"))
        {
            // HSPICE `GOAL < v` / `GOAL > v` [SA Ch.27]: the operator stays in the rhs.
            out.items[out.items.len - 1].rhs = tok;
            pending_eq = tok.len == 1;
            continue;
        }
        if (eq) |e| {
            pending_eq = e + 1 == tok.len;
            try out.append(arena, .{ .lhs = tok[0..e], .rhs = tok[e + 1 ..] });
        } else try out.append(arena, .{ .lhs = tok });
    }
    if (pending_eq) return error.ParseError;
    return out.items;
}

/// Index of the first bare word `key`.
pub fn find(w: []const Word, key: []const u8) ?usize {
    assert(key.len > 0);
    for (w, 0..) |x, i| if (x.rhs == null and mem.eql(u8, x.lhs, key)) return i;
    return null;
}

/// The words inside a `( ... )` group.
pub fn groupWords(arena: Allocator, group: []const u8) Error![]Word {
    if (group.len < 2 or group[0] != '(' or group[group.len - 1] != ')') return error.ParseError;
    return words(arena, group[1 .. group.len - 1]);
}

test words {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const w = try words(a, "v(a, b) val = 1 x= 2,y =3 goal > 4 {p q}");
    try std.testing.expectEqual(6, w.len);
    try std.testing.expectEqualStrings("v(a, b)", w[0].lhs);
    try std.testing.expectEqualStrings("1", w[1].rhs.?);
    try std.testing.expectEqualStrings("2", w[2].rhs.?);
    try std.testing.expectEqualStrings("3", w[3].rhs.?);
    try std.testing.expectEqualStrings("> 4", w[4].rhs.?);
    try std.testing.expectEqualStrings("{p q}", w[5].lhs);
    try std.testing.expectError(error.ParseError, words(a, "=x"));
    try std.testing.expectError(error.ParseError, words(a, "x ="));
    try std.testing.expectEqual(5, find(w, "{p q}"));
}
