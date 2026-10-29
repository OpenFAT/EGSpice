//! Spectre statements to SPICE cards: the Spectre dialect's reader, run on
//! the logical lines before netlist.zig reads any card, as pattern.zig
//! rewrites PAT sources. In: Spectre logical lines (the title taken). Out:
//! SPICE cards and, parallel to them, each device card's letter, since a
//! Spectre instance name says nothing about its device. Stages: pass 1
//! declares names (spectre_state.zig), then each statement is tokenized
//! (spectre_expr.zig) and dispatched to its reader (spectre_blocks.zig,
//! spectre_devices.zig).
//! Cites: dev/frontend.md "Spectre dialect" (the mapping);
//! dev/research/spectre-syntax.md (§ numbers); tests/frontend/grammar/spectre.peg.
//!
//! Contents:
//! - `pub const Error = error{ OutOfMemory, ParseError, UnsupportedCard }`
//! - `pub const Translated = struct`
//! - `pub const switchesTo = lines.spectre.Lines.switchesTo`
//! - `pub fn translate(arena: Allocator, in: []const []const u8) Error!Translated`

const std = @import("std");
const assert = std.debug.assert;
const testing = std.testing;
const Allocator = std.mem.Allocator;
const lines = @import("lines.zig");
const spectre_expr = @import("spectre_expr.zig");
const spectre_params = @import("spectre_params.zig");
const spectre_state = @import("spectre_state.zig");
const spectre_blocks = @import("spectre_blocks.zig");
const spectre_devices = @import("spectre_devices.zig");
const Tok = spectre_expr.Tok;
const Translator = spectre_state.Translator;

/// `ParseError`: a malformed statement; `UnsupportedCard`: a statement or
/// form ESPice does not read (logged with the line).
pub const Error = spectre_expr.Error;

/// SPICE cards and, parallel to them, each device card's letter (0 for a
/// card that is no Spectre instance, whose name carries its own).
pub const Translated = struct {
    /// The SPICE cards, in `translate`'s arena.
    lines: []const []const u8,
    /// `kinds[i]` is card i's device letter, 0 for none; `kinds.len == lines.len`.
    kinds: []const u8,
};

/// Whether `line` is `simulator lang=<lang> ...` (the line reader's own test).
pub const switchesTo = lines.spectre.Lines.switchesTo;

/// Rewrites Spectre logical lines (the title already taken) into SPICE cards
/// in `arena`, with each device card's letter. A failing statement is
/// logged with its line; `ParseError` also when a `subckt` or block is
/// still open at the end.
pub fn translate(arena: Allocator, in: []const []const u8) Error!Translated {
    var t: Translator = .{ .arena = arena };
    try t.declare(in);
    var spice = false;
    for (in) |line| {
        if (spice) {
            if (switchesTo(line, "spectre")) {
                spice = false;
                continue;
            }
            try t.out.append(arena, try spiceCard(arena, line));
            continue;
        }
        if (switchesTo(line, "spice")) {
            spice = true;
            continue;
        }
        statement(&t, line) catch |err| {
            if (!@import("builtin").is_test) std.log.err("spectre: {s}: {s}", .{ @errorName(err), line });
            return err;
        };
    }
    if (t.subckts.items.len != 0 or t.blocks.items.len != 0) return error.ParseError;
    assert(t.kinds.items.len <= t.out.items.len);
    try t.kinds.appendNTimes(arena, 0, t.out.items.len - t.kinds.items.len);
    return .{ .lines = t.out.items, .kinds = t.kinds.items };
}

// ---- private ----

fn statement(t: *Translator, line: []const u8) Error!void {
    const toks = try spectre_expr.tokenize(t.arena, line);
    if (toks.len == 0) return;
    if (t.blocks.items.len > 0) switch (t.blocks.items[t.blocks.items.len - 1]) {
        .skip => return spectre_blocks.skipped(t, toks),
        .group => return spectre_blocks.groupRow(t, toks),
        .func => return spectre_blocks.funcBody(t, toks),
        .cond => {},
    };
    const head = toks[0];
    if (head.is("}")) return spectre_blocks.closeBrace(t, toks);
    if (head.kind != .word) return error.ParseError;
    const w = head.text;
    const rest = toks[1..];
    if (std.mem.eql(u8, w, "simulator")) return; // `insensitive=yes` and the like
    if (std.mem.eql(u8, w, "parameters")) return spectre_blocks.parameters(t, rest);
    if (std.mem.eql(u8, w, "model")) return spectre_blocks.model(t, rest);
    if (std.mem.eql(u8, w, "subckt")) return spectre_blocks.subckt(t, rest);
    if (std.mem.eql(u8, w, "inline") and rest.len > 0 and rest[0].is("subckt")) return spectre_blocks.subckt(t, rest[1..]);
    if (std.mem.eql(u8, w, "ends")) return spectre_blocks.ends(t);
    if (std.mem.eql(u8, w, "global")) return spectre_blocks.global(t, rest);
    if (std.mem.eql(u8, w, "ic") or std.mem.eql(u8, w, "nodeset")) return spectre_blocks.initial(t, w, rest);
    if (std.mem.eql(u8, w, "save")) return saveChecked(rest);
    if (std.mem.eql(u8, w, "ahdl_include")) {
        if (rest.len != 1 or rest[0].kind != .string) return error.ParseError;
        return t.emit(".hdl {s}", .{rest[0].text});
    }
    if (std.mem.eql(u8, w, "if")) return spectre_blocks.cond(t, rest, ".if");
    if (std.mem.eql(u8, w, "real")) return spectre_blocks.func(t, rest);
    if (std.mem.eql(u8, w, "statistics")) {
        std.log.warn("spectre: statistics blocks are read and set aside (no Monte Carlo variations)", .{});
        return t.blocks.append(t.arena, .skip);
    }
    if (std.mem.eql(u8, w, "include") or std.mem.eql(u8, w, "library") or std.mem.eql(u8, w, "section"))
        return error.UnsupportedCard;
    return spectre_devices.named(t, toks);
}

/// A card of a `simulator lang=spice` region (§3.2): lowercased, its numbers
/// written plain (SPICE suffixes, `meg` is 1e6), and `'expr'` as `"expr"`,
/// the Spectre field splitter's quote.
fn spiceCard(arena: Allocator, line: []const u8) Error![]const u8 {
    var out: std.ArrayList(u8) = .empty;
    var i: usize = 0;
    while (i < line.len) {
        const c = line[i];
        const prev: u8 = if (i == 0) ' ' else line[i - 1];
        const starts = std.ascii.isDigit(c) or (c == '.' and i + 1 < line.len and std.ascii.isDigit(line[i + 1]));
        if (starts and !(std.ascii.isAlphanumeric(prev) or prev == '_' or prev == '.')) {
            var j = i;
            while (j < line.len and (std.ascii.isAlphanumeric(line[j]) or line[j] == '.' or
                ((line[j] == '+' or line[j] == '-') and j > i and std.ascii.toLower(line[j - 1]) == 'e'))) j += 1;
            const lower = try std.ascii.allocLowerString(arena, line[i..j]);
            if (lines.hspice.parseNum(lower)) |v| {
                try out.print(arena, "{e}", .{v});
            } else try out.appendSlice(arena, lower);
            i = j;
            continue;
        }
        try out.append(arena, if (c == '\'') '"' else std.ascii.toLower(c));
        i += 1;
    }
    return out.items;
}

/// `save X[:t] ... [key=value]` (§4.11): checked, and not acted on, since
/// every vector is written. A `:` needs a terminal or output after it.
fn saveChecked(toks: []const Tok) Error!void {
    for (toks, 0..) |tok, i| if (tok.is(":")) {
        if (i + 1 == toks.len or toks[i + 1].kind != .word) return error.ParseError;
    };
}

test "translate keeps each device card's letter in a column parallel to the cards" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const out = try translate(arena.allocator(), &.{
        "load (a 0) resistor r=1k",
        "vin (a 0) vsource dc=1",
        "simulator lang=spice",
        "r2 a 0 1",
    });
    try testing.expectEqual(out.lines.len, out.kinds.len);
    var letters: std.ArrayList(u8) = .empty;
    for (out.lines, out.kinds) |line, k| if (k != 0) {
        try letters.append(arena.allocator(), k);
        try testing.expect(std.mem.startsWith(u8, line, if (k == 'r') "load" else "vin"));
    };
    try testing.expectEqualStrings("rv", letters.items);
    // The SPICE region's card names its own letter.
    try testing.expectEqual(@as(u8, 0), out.kinds[out.kinds.len - 1]);
}

test {
    _ = spectre_expr;
    _ = spectre_params;
    _ = spectre_blocks;
    _ = spectre_devices;
    _ = @import("tests/spectre.zig");
}
