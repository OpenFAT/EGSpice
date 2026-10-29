//! The Spectre dialect's lines. In: a deck's bytes, case kept. Out:
//! logical lines (the title line, `*` comment lines, `//` comments, `\` or
//! `+` continuation, and HSPICE's rules inside a `simulator lang=spice`
//! region), their fields and numbers.
//! Cites: tests/frontend/grammar/spectre_lines.peg (the spec);
//! dev/research/spectre-syntax.md §1.2-§1.4, §2.2, §3.1, §3.2.
//!
//! Contents:
//! - `pub const title_line = true`
//! - `pub const remainder_ops = "%"`
//! - `pub const unary_not = true`
//! - `pub const fold_case = false`
//! - `pub const Split = Fields("\"", false)`
//! - `pub const Lines = struct`
//! - `pub fn switchesTo(line: []const u8, lang: []const u8) bool`
//! - `pub fn next(self: *Lines) Allocator.Error!?[]const u8`
//! - `pub fn parseNum(text: []const u8) ?f64`

const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;
const Fields = @import("fields.zig").Fields;
const spice_number = @import("spice_number.zig");
const physical_lines = @import("physical_lines.zig");
const hspice = @import("lines_hspice.zig");

/// §1.2: line 1 of the top-level netlist is the title (UG p.29, p.50).
pub const title_line = true;
/// Operator bytes beyond `* /` at their level: `%`, which ESPice reads
/// (tests/frontend/grammar/spectre.peg MulOp); `\` is no Spectre operator (#203).
pub const remainder_ops = "%";
/// Unary `!`, which ESPice reads (tests/frontend/grammar/spectre.peg SignOp).
pub const unary_not = true;
/// Cards are case-sensitive: the text is parsed as written.
pub const fold_case = false;
/// Field splitter over one logical line.
pub const Split = Fields("\"", false);

/// Logical-line iterator; a joined line is allocated in `arena`.
pub const Lines = struct {
    rest: []const u8,
    arena: Allocator,
    /// Inside a `simulator lang=spice` region (§3.1).
    spice: bool = false,

    /// `line` up to its `//` comment, outside strings (a backslash
    /// escapes a quote or any byte of a name), trimmed.
    fn stripComment(line: []const u8) []const u8 {
        var i: usize = 0;
        var quoted = false;
        while (i < line.len) : (i += 1) {
            // Skip the bytes none of the branches below reads.
            while (i < line.len and line[i] != '\\' and line[i] != '"' and line[i] != '/') i += 1;
            if (i == line.len) break;
            const c = line[i];
            if (c == '\\') {
                i += 1;
            } else if (c == '"') {
                quoted = !quoted;
            } else if (!quoted and c == '/' and i + 1 < line.len and line[i + 1] == '/') {
                return std.mem.trim(u8, line[0..i], " \t");
            }
        }
        return std.mem.trim(u8, line, " \t");
    }

    /// A line the statement skips: blank, `*` in the first column, or
    /// only a `//` comment (§1.3).
    fn isSkip(raw: []const u8) bool {
        return (raw.len > 0 and raw[0] == '*') or stripComment(raw).len == 0;
    }

    /// A physical line's text: up to its comment, and without the
    /// backslash that continues it (§1.4) when it has one. The
    /// backslash must end the line, so a line with a comment has none.
    fn read(raw: []const u8) struct { []const u8, bool } {
        const text = stripComment(raw);
        if (text.len < std.mem.trim(u8, raw, " \t").len) return .{ text, false };
        return trailingBackslash(text);
    }

    /// `text` without the backslash that continues it, and whether it
    /// had one. A pair of backslashes is an escaped one, a name
    /// character (§2.2), so only an odd run ends in a break.
    fn trailingBackslash(text: []const u8) struct { []const u8, bool } {
        var run: usize = 0;
        while (run < text.len and text[text.len - 1 - run] == '\\') run += 1;
        if (run % 2 == 1)
            return .{ std.mem.trimEnd(u8, text[0 .. text.len - 1], " \t"), true };
        return .{ text, false };
    }

    /// True for `simulator lang=<lang> ...`, compared case-insensitively
    /// as a SPICE region writes it; fields after the language are ignored.
    pub fn switchesTo(line: []const u8, lang: []const u8) bool {
        var f = Fields("\"", false).init(line);
        const words = [_][]const u8{ "simulator", "lang", "=" };
        for (words) |w| if (!std.ascii.eqlIgnoreCase(f.next() orelse return false, w)) return false;
        return std.ascii.eqlIgnoreCase(f.next() orelse return false, lang);
    }

    /// The next logical line, continuations joined; null at the end.
    /// The line borrows `rest`'s text, or lives in `arena` when it joins
    /// continuations. Fails only with `OutOfMemory` while joining.
    pub fn next(self: *Lines) Allocator.Error!?[]const u8 {
        if (self.spice) {
            // §3.2: a SPICE region reads as HSPICE does.
            var h: hspice.Lines = .{ .rest = self.rest, .arena = self.arena };
            const line = try h.next();
            self.rest = h.rest;
            if (line) |l| if (switchesTo(l, "spectre")) {
                self.spice = false;
            };
            return line;
        }
        var raw: []const u8 = undefined;
        while (true) {
            raw = physical_lines.nextPhysicalLine(&self.rest) orelse return null;
            if (!isSkip(raw)) break;
        }
        var text, var cont = read(raw);
        const head = text;
        var joined: ?std.ArrayList(u8) = null;
        // Each pass consumes a physical line or ends the statement.
        const passes_max = self.rest.len + 1;
        var passes: usize = 0;
        while (true) : (passes += 1) {
            assert(passes <= passes_max);
            // `*` skips a line only in the first column; `keep` errs safe.
            if (!cont and physical_lines.startsStatement(self.rest, "+*/")) break;
            const save = self.rest;
            const next_raw = physical_lines.nextPhysicalLine(&self.rest) orelse break;
            if (cont) {
                text, cont = read(next_raw);
            } else {
                const t = std.mem.trimStart(u8, next_raw, " \t");
                if (t.len > 0 and t[0] == '+') {
                    text, cont = read(t[1..]);
                } else if (isSkip(next_raw)) {
                    continue;
                } else {
                    self.rest = save;
                    break;
                }
            }
            try physical_lines.join(self.arena, &joined, head, " ", text);
        }
        const line = if (joined) |j| j.items else head;
        if (switchesTo(line, "spice")) self.spice = true;
        return line;
    }
};

/// A number literal with a case-sensitive Spectre suffix (`M` mega, `m` milli).
pub fn parseNum(text: []const u8) ?f64 {
    return spice_number.spectreNumber(text);
}
