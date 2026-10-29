//! Spectre parameter lists and values. In: a statement's tokens after its
//! name and master. Out: `key=value` rows, and each value as SPICE card
//! text (a name, a plain number, or a quoted expression).
//! Cites: dev/research/spectre-syntax.md §4.2 (parameters), §5 (vectors).
//!
//! Contents:
//! - `pub const Param = struct`
//! - `pub fn params(arena: Allocator, toks: []const Tok) Error![]const Param`
//! - `pub fn find(ps: []const Param, key: []const u8) ?[]const Tok`
//! - `pub fn word(v: []const Tok) []const u8`
//! - `pub fn value(arena: Allocator, toks: []const Tok) Error![]const u8`
//! - `pub fn quoted(arena: Allocator, text: []const u8) Error![]const u8`
//! - `pub fn vector(arena: Allocator, toks: []const Tok) Error![]const []const u8`

const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;
const spectre_expr = @import("spectre_expr.zig");
const Error = spectre_expr.Error;
const Tok = spectre_expr.Tok;

/// A parameter as written: its key and value tokens (a vector's included).
pub const Param = struct { key: []const u8, value: []const Tok };

/// Splits `toks` into `key=value` parameters: a value runs to the next
/// `key=` at depth 0 (§4.2). Parentheses around one (`(c=1p)`) are dropped.
pub fn params(arena: Allocator, toks: []const Tok) Error![]const Param {
    var out: std.ArrayList(Param) = .empty;
    var i: usize = 0;
    while (i < toks.len) {
        if (toks[i].is("(") and i + 2 < toks.len and toks[i + 2].is("=")) {
            // `(c=1p)`: find its close and read the inside.
            var depth: usize = 0;
            var j = i;
            while (j < toks.len) : (j += 1) {
                if (toks[j].is("(")) depth += 1;
                if (toks[j].is(")")) {
                    depth -= 1;
                    if (depth == 0) break;
                }
            }
            if (j == toks.len) return error.ParseError;
            try out.appendSlice(arena, try params(arena, toks[i + 1 .. j]));
            i = j + 1;
            continue;
        }
        if (toks[i].kind != .word or i + 1 >= toks.len or !toks[i + 1].is("=")) return error.ParseError;
        const key = toks[i].text;
        var j = i + 2;
        var depth: usize = 0;
        while (j < toks.len) : (j += 1) {
            const t = toks[j];
            if (t.is("(") or t.is("[")) depth += 1;
            if (t.is(")") or t.is("]")) depth -|= 1;
            if (depth == 0 and t.kind == .word and j + 1 < toks.len and toks[j + 1].is("=") and j > i + 2) break;
        }
        try out.append(arena, .{ .key = key, .value = toks[i + 2 .. j] });
        i = j;
    }
    return out.items;
}

/// The value tokens of parameter `key`; null when it is not given.
pub fn find(ps: []const Param, key: []const u8) ?[]const Tok {
    assert(key.len > 0);
    for (ps) |p| if (std.mem.eql(u8, p.key, key)) return p.value;
    return null;
}

/// A one-word value's text; "" for anything else, which no keyword matches.
pub fn word(v: []const Tok) []const u8 {
    return if (v.len == 1 and v[0].kind == .word) v[0].text else "";
}

/// A parameter's value as SPICE text: a name, a plain number, or an
/// expression in quotes, which every card field reads as one (a bare `(`
/// would start a group).
pub fn value(arena: Allocator, toks: []const Tok) Error![]const u8 {
    if (toks.len == 1 and toks[0].kind == .word and !std.ascii.isDigit(toks[0].text[0]) and toks[0].text[0] != '.' and
        spectre_expr.constants.get(toks[0].text) == null and !std.mem.eql(u8, toks[0].text, "temp")) return toks[0].text;
    if (toks.len > 0 and toks[0].is("[")) return error.UnsupportedCard; // a vector where one is not read
    return quoted(arena, try spectre_expr.expression(arena, toks));
}

/// `text` as one card field: as is when it is a number, a name or a
/// string, in quotes when it is an expression.
pub fn quoted(arena: Allocator, text: []const u8) Error![]const u8 {
    assert(text.len > 0);
    if (text[0] == '"' or std.mem.indexOfAny(u8, text, "()+*/%?:<>=!&|") == null and !std.mem.startsWith(u8, text, "-")) return text;
    return arena.print("\"{s}\"", .{text});
}

/// A vector's elements as SPICE text (§5).
pub fn vector(arena: Allocator, toks: []const Tok) Error![]const []const u8 {
    if (toks.len < 2 or !toks[0].is("[") or !toks[toks.len - 1].is("]")) return error.ParseError;
    var out: std.ArrayList([]const u8) = .empty;
    const inner = toks[1 .. toks.len - 1];
    var i: usize = 0;
    while (i < inner.len) {
        if (inner[i].is(",")) {
            i += 1;
            continue;
        }
        // An element: a signed constant or name, or a parenthesized expression.
        var j = i + 1;
        if (inner[i].kind == .op and j < inner.len) j += 1;
        if (inner[i].is("(")) {
            var depth: usize = 0;
            j = i;
            while (j < inner.len) : (j += 1) {
                if (inner[j].is("(")) depth += 1;
                if (inner[j].is(")")) {
                    depth -= 1;
                    if (depth == 0) break;
                }
            }
            j += 1;
        }
        assert(j > i);
        try out.append(arena, try quoted(arena, try spectre_expr.expression(arena, inner[i..j])));
        i = j;
    }
    return out.items;
}
