//! Xyce's native binary raw output. Only decks with one supported analysis
//! are admitted: Xyce can otherwise silently choose only one of several
//! analyses. Unsupported dialect cards are refused rather than rewritten.
const std = @import("std");
const common = @import("job.zig");
/// The flag whose output the report header quotes as Xyce's version.
pub const version_flag = "-v";

/// A job writing `raw`, or a refusal naming why Xyce would not run the same
/// analysis. A deck with `.save` runs from `xyce-input.cir` beside `raw`, its
/// selection rewritten as `.PRINT ... FORMAT=RAW`; otherwise Xyce reads the
/// deck itself with `-r`, which dumps every solution vector.
pub fn prepare(io: std.Io, a: std.mem.Allocator, bin: []const u8, source: []const u8, raw: []const u8, dialect: ?[]const u8) common.Error!common.Job {
    if (dialect) |d| if (!std.mem.eql(u8, d, "ngspice")) return .{ .refused = "Xyce adapter requires an ngspice-dialect deck" };
    const text = std.Io.Dir.cwd().readFileAlloc(io, source, a, .unlimited) catch return error.DeckFailed;
    if (refusal(text)) |reason| return .{ .refused = reason };
    // SPICE .save selects raw-file vectors; Xyce's .SAVE is a different
    // command. Use native .PRINT FORMAT=RAW for the same selected vectors.
    // -r would override the selection and dump the entire solution vector.
    const selected = selectedDeck(a, text, std.fs.path.basename(raw)) catch return error.OutOfMemory;
    if (selected) |deck| {
        const directory = std.fs.path.dirname(raw).?;
        const translated = try std.fs.path.join(a, &.{ directory, "xyce-input.cir" });
        std.Io.Dir.cwd().writeFile(io, .{ .sub_path = translated, .data = deck }) catch return error.DeckFailed;
        return .{
            .argv = try a.dupe([]const u8, &.{ bin, "-quiet", translated }),
            .cwd = .{ .path = directory },
            .raw = raw,
        };
    }
    return .{
        .argv = try a.dupe([]const u8, &.{ bin, "-quiet", "-r", raw, source }),
        .cwd = .{ .path = std.fs.path.dirname(source).? },
        .raw = raw,
    };
}

/// Why Xyce would read `text` differently from ngspice, or null.
fn refusal(text: []const u8) ?[]const u8 {
    var lines = std.mem.splitScalar(u8, text, '\n');
    _ = lines.next(); // title
    var analyses: usize = 0;
    var save_continuation = false;
    var ended = false;
    while (lines.next()) |line| {
        var words = std.mem.tokenizeAny(u8, line, " \t\r");
        const word = words.next() orelse continue;
        if (word[0] == '*') continue;
        if (word[0] == '+' and save_continuation) return "continued .save selection is not supported";
        save_continuation = false;
        if (word[0] != '.') continue;
        var buf: [32]u8 = undefined;
        if (word.len > buf.len) return "unsupported Xyce directive";
        const card = std.ascii.lowerString(&buf, word);
        if (std.mem.eql(u8, card, ".op") or std.mem.eql(u8, card, ".dc") or std.mem.eql(u8, card, ".ac") or std.mem.eql(u8, card, ".tran")) {
            analyses += 1;
            continue;
        }
        if (std.mem.eql(u8, card, ".save")) {
            save_continuation = true;
            var count: usize = 0;
            while (words.next()) |name| {
                if (name.len < 4 or (std.ascii.toLower(name[0]) != 'v' and std.ascii.toLower(name[0]) != 'i') or name[1] != '(' or name[name.len - 1] != ')')
                    return "Xyce adapter only translates .save v(...) and i(...)";
                count += 1;
            }
            if (count == 0) return "empty .save selection";
            continue;
        }
        if (std.mem.eql(u8, card, ".print") or std.mem.eql(u8, card, ".probe"))
            return "Xyce adapter does not translate .print/.probe output selectors";
        if (std.mem.eql(u8, card, ".end")) ended = true;
        // Include/library contents may contain analysis or dialect cards;
        // refuse until the adapter can inspect them recursively.
        if (!declarations.has(card)) return "unsupported Xyce directive (only standalone OP/DC/AC/transient decks)";
    }
    if (!ended) return "Xyce adapter requires an explicit .end";
    return if (analyses == 1) null else "Xyce adapter requires exactly one analysis";
}

const declarations = std.StaticStringMap(void).initComptime(.{
    .{ ".model", {} }, .{ ".subckt", {} }, .{ ".ends", {} },
    .{ ".param", {} }, .{ ".global", {} }, .{ ".end", {} },
});

/// `text` with its `.save` cards folded into one `.PRINT FORMAT=RAW` before
/// `.end`, or null when the deck saves nothing.
fn selectedDeck(a: std.mem.Allocator, text: []const u8, raw_name: []const u8) !?[]const u8 {
    var selected: std.Io.Writer.Allocating = .init(a);
    var analysis: []const u8 = "";
    var lines = std.mem.splitScalar(u8, text, '\n');
    _ = lines.next();
    while (lines.next()) |line| {
        var words = std.mem.tokenizeAny(u8, line, " \t\r");
        const card = words.next() orelse continue;
        if (std.ascii.eqlIgnoreCase(card, ".save")) {
            while (words.next()) |name| try selected.writer.print(" {s}", .{name});
        } else if (std.ascii.eqlIgnoreCase(card, ".op") or std.ascii.eqlIgnoreCase(card, ".dc")) {
            analysis = "DC";
        } else if (std.ascii.eqlIgnoreCase(card, ".tran")) {
            analysis = "TRAN";
        } else if (std.ascii.eqlIgnoreCase(card, ".ac")) {
            analysis = "AC";
        }
    }
    if (selected.written().len == 0) return null;
    var result: std.Io.Writer.Allocating = .init(a);
    lines.reset();
    try result.writer.print("{s}\n", .{lines.next().?});
    while (lines.next()) |line| {
        var words = std.mem.tokenizeAny(u8, line, " \t\r");
        const card = words.next() orelse "";
        if (std.ascii.eqlIgnoreCase(card, ".save")) continue;
        if (std.ascii.eqlIgnoreCase(card, ".end"))
            try result.writer.print(".PRINT {s} FORMAT=RAW FILE={s}{s}\n", .{ analysis, raw_name, selected.written() });
        try result.writer.print("{s}\n", .{line});
    }
    return try result.toOwnedSlice();
}

test "Xyce refuses partial analysis coverage and unhandled dialect controls" {
    const t = std.testing;
    try t.expect(refusal("title\nR1 a 0 1k\nV1 a 0 1\n.op\n.end\n") == null);
    try t.expect(refusal("title\n.ac dec 10 1 1e6\n.tran 1n 1u\n.end\n") != null);
    try t.expect(refusal("title\n.op\n.control\nrun\n.endc\n.end\n") != null);
    try t.expect(refusal("title\n.options reltol=1e-6\n.op\n.end\n") != null);
    try t.expect(refusal("title\n.include hidden.sp\n.op\n.end\n") != null);
    try t.expect(refusal("title\n.hb 1e6\n.end\n") != null);
}

test "Xyce translates raw vector selection without changing the circuit or analysis" {
    const t = std.testing;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const original = "title\nV1 in 0 1\nR1 in out 1k\nC1 out 0 1n\n.tran 1n 1u\n.save v(out) i(V1)\n.end\n";
    try t.expect(refusal(original) == null);
    const selected = (try selectedDeck(arena.allocator(), original, "xyce.raw")).?;
    try t.expectEqualStrings("title\nV1 in 0 1\nR1 in out 1k\nC1 out 0 1n\n.tran 1n 1u\n.PRINT TRAN FORMAT=RAW FILE=xyce.raw v(out) i(V1)\n.end\n\n", selected);
    try t.expect(refusal("title\n.op\n.save @m1[id]\n.end\n") != null);
    try t.expect(refusal("title\n.op\n.print dc v(out)\n.end\n") != null);
    try t.expect(refusal("title\n.op\n.save v(out)\n+ i(v1)\n.end\n") != null);
    try t.expect(refusal("title\n.op\n.save v(out)\n") != null);
}
