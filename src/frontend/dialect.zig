//! Which dialect a deck is in. In: a CLI dialect name, or a deck's text.
//! Out: the `Dialect`, from the name or from markers only HSPICE or Spectre
//! decks carry; ngspice otherwise.
//! Cites: dev/research/{hspice,spectre}-syntax.md (the markers); the corpus
//! decks' `tokenizer` fields are the oracle (`detectDialect` test).
//!
//! Contents:
//! - `pub fn parseDialect(name: []const u8) ?Dialect`
//! - `pub fn detectDialect(text: []const u8) Dialect`

const std = @import("std");
const assert = std.debug.assert;
const Dialect = @import("netlist").Dialect;

/// The dialect a CLI name or short alias (`ng`, `hs`, `scs`) selects.
pub fn parseDialect(name: []const u8) ?Dialect {
    return std.StaticStringMap(Dialect).initComptime(.{
        .{ "ngspice", .ngspice }, .{ "ng", .ngspice },
        .{ "hspice", .hspice },   .{ "hs", .hspice },
        .{ "spectre", .spectre }, .{ "scs", .spectre },
    }).get(name);
}

/// The dialect a deck's own text is in, from markers no ngspice deck of
/// the corpus carries (a test pins that): Spectre's `simulator lang=` or a
/// `name (nodes) master` instance, then an HSPICE-only card or keyword.
/// Anything else is ngspice, HSPICE decks with ngspice syntax included
/// (their meaning can still differ, as TNOM does). Reads neither the title
/// line nor the includes.
pub fn detectDialect(text: []const u8) Dialect {
    const hspice_cards = [_][]const u8{
        ".data", ".enddata", ".alter", ".variation", ".mosra", ".appendmodel", ".lin",      ".net",     ".fft",
        ".stim", ".vec",     ".pat",   ".check",     ".dout",  ".biaschk",     ".power",    ".connect", ".dcvolt",
        ".load", ".sample",  ".sn",    ".snac",      ".snxf",  ".snnoise",     ".ptdnoise",
    };
    var hspice = false;
    var card: []const u8 = "";
    var it = std.mem.splitScalar(u8, text, '\n');
    var title = true;
    while (it.next()) |raw| : (title = false) {
        const line = std.mem.trim(u8, raw, " \t\r");
        if (std.ascii.startsWithIgnoreCase(line, "simulator") and std.ascii.startsWithIgnoreCase(std.mem.trimStart(u8, line[9..], " \t"), "lang")) return .spectre;
        if (title or line.len == 0 or line[0] == '*') continue;
        var rest = line;
        if (line[0] == '+') rest = line[1..] else {
            const end = std.mem.indexOfAny(u8, line, " \t(") orelse line.len;
            card = line[0..end];
            rest = line[end..];
            // `r1 (a b) resistor`: Spectre's instance form.
            const tail = std.mem.trimStart(u8, rest, " \t");
            if (std.ascii.isAlphabetic(line[0]) and tail.len != 0 and tail[0] == '(') if (std.mem.indexOfScalar(u8, tail, ')')) |close| {
                const after = tail[close + 1 ..];
                const master = std.mem.trimStart(u8, after, " \t");
                if (master.len != 0 and master.len < after.len and std.ascii.isAlphabetic(master[0])) return .spectre;
            };
            // Past the first HSPICE marker only a Spectre one can change
            // the answer.
            if (hspice) continue;
            if (std.mem.startsWith(u8, card, ".")) for (hspice_cards) |c| {
                hspice = hspice or std.ascii.eqlIgnoreCase(card, c);
            };
            // `.lib 'file' tt`: a quoted path and a section.
            if (std.ascii.eqlIgnoreCase(card, ".lib") and tail.len != 0 and tail[0] == '\'') if (std.mem.indexOfScalarPos(u8, tail, 1, '\'')) |close| {
                hspice = hspice or std.mem.trim(u8, tail[close + 1 ..], " \t").len != 0;
            };
        }
        // Every word test below needs an `=` or a `sweep` on the line.
        if (hspice or (std.mem.indexOfScalar(u8, rest, '=') == null and std.ascii.findIgnoreCase(rest, "sweep") == null)) continue;
        var words = std.mem.tokenizeAny(u8, rest, " \t");
        var sweep = false;
        while (words.next()) |w| {
            // `sweep data=`, `sweep monte=`, `sweep optimize=` on an analysis card.
            if (sweep) for ([_][]const u8{ "data", "monte", "optimize" }) |k| {
                hspice = hspice or std.ascii.startsWithIgnoreCase(w, k);
            };
            sweep = std.ascii.eqlIgnoreCase(w, "sweep");
            hspice = hspice or std.ascii.startsWithIgnoreCase(w, "rlgcmodel=") or std.ascii.startsWithIgnoreCase(w, "tstonefile=") or
                std.ascii.eqlIgnoreCase(w, "modeltype=rlgc") or
                (std.ascii.startsWithIgnoreCase(card, ".opt") and std.ascii.startsWithIgnoreCase(w, "search=")) or
                (std.ascii.startsWithIgnoreCase(card, "p") and std.ascii.startsWithIgnoreCase(w, "port="));
        }
    }
    return if (hspice) .hspice else .ngspice;
}

// ---- private ----

test parseDialect {
    try std.testing.expectEqual(Dialect.spectre, parseDialect("scs").?);
    try std.testing.expectEqual(Dialect.ngspice, parseDialect("ngspice").?);
    // Only the listed names and aliases, no prefixes.
    try std.testing.expectEqual(null, parseDialect("h"));
    try std.testing.expectEqual(null, parseDialect(""));
}

test detectDialect {
    try std.testing.expectEqual(Dialect.ngspice, detectDialect("t\nv1 a 0 1\nr1 a 0 {2*rv}\n.tran 1n 1u\n.meas tran x max v(a)\n.end\n"));
    try std.testing.expectEqual(Dialect.hspice, detectDialect("t\n.param rl=1k\n.data rows rl\n+ 1k\n.enddata\n.dc data=rows\n.end\n"));
    try std.testing.expectEqual(Dialect.hspice, detectDialect("t\nr1 a 0 1k\n.tran 1n 1u\n+ sweep monte=10\n.end\n"));
    try std.testing.expectEqual(Dialect.hspice, detectDialect("t\n.lib 'models.lib' tt\n.op\n"));
    try std.testing.expectEqual(Dialect.ngspice, detectDialect("t\n.lib 'models.lib'\n* .data in a comment\n.op\n"));
    try std.testing.expectEqual(Dialect.spectre, detectDialect("simulator lang=spectre\nr1 (a 0) resistor r=1k\n"));
    try std.testing.expectEqual(Dialect.spectre, detectDialect("title\nr1 (out 0) resistor r=1k\nt1 tran stop=1u\n"));
    // The title line may say anything.
    try std.testing.expectEqual(Dialect.ngspice, detectDialect(".data in the title\nv1 a 0 1\n.op\n"));
}

// Every corpus deck whose oracle names no tokenizer reads as ngspice, and
// a deck detected as HSPICE or Spectre is one: the markers are precise.
test "detectDialect: no false positive over the corpus" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var dir = std.Io.Dir.cwd().openDir(io, "tests/conformance/tests/netlist", .{ .iterate = true }) catch return error.SkipZigTest;
    defer dir.close(io);
    var walker = try dir.walk(a);
    defer walker.deinit();
    var found: usize = 0;
    var total: usize = 0;
    while (try walker.next(io)) |entry| {
        if (!std.mem.endsWith(u8, entry.path, ".expected.json")) continue;
        const json = try dir.readFileAlloc(io, entry.path, a, .limited(1 << 24));
        defer a.free(json);
        const deck_path = try std.mem.concat(a, u8, &.{ entry.path[0 .. entry.path.len - ".expected.json".len], ".sp" });
        defer a.free(deck_path);
        const deck = dir.readFileAlloc(io, deck_path, a, .limited(1 << 26)) catch continue;
        defer a.free(deck);
        const wanted: Dialect = for ([_]Dialect{ .hspice, .spectre }) |d| {
            if (std.mem.indexOf(u8, json, switch (d) {
                .hspice => "\"tokenizer\": \"hspice\"",
                else => "\"tokenizer\": \"spectre\"",
            }) != null) break d;
        } else .ngspice;
        const got = detectDialect(deck);
        try std.testing.expect(got == .ngspice or got == wanted);
        if (wanted != .ngspice) total += 1;
        if (wanted != .ngspice and got == wanted) found += 1;
    }
    // 66 of the 102 on 2026-10-20; the rest use only ngspice syntax.
    try std.testing.expect(found * 2 >= total);
}
