//! The HSPICE dialect's lines. In: a deck's lowercased bytes. Out: logical
//! lines (like ngspice, but `"` also quotes, `$` starts a comment only after
//! a separator or a number, and a trailing `\` or `\\` continues the line),
//! their fields and numbers, and the value-form checks HSPICE refuses by.
//! Cites: tests/frontend/grammar/hspice_lines.peg (the spec);
//! dev/research/hspice-syntax.md L2-L6, T2, T3, X1, X2 (UG08, CR10 p. 238).
//!
//! Contents:
//! - `pub const title_line = true`
//! - `pub const remainder_ops = ""`
//! - `pub const unary_not = false`
//! - `pub const fold_case = true`
//! - `pub const Split = Fields("'\"", false)`
//! - `pub const Lines = struct`
//! - `pub fn next(self: *Lines) Allocator.Error!?[]const u8`
//! - `pub fn parseNum(text: []const u8) ?f64`
//! - `pub fn exponentWithScale(line: []const u8) bool`
//! - `pub fn bareValue(text: []const u8) bool`

const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;
const Fields = @import("fields.zig").Fields;
const spice_number = @import("spice_number.zig");
const physical_lines = @import("physical_lines.zig");

/// The first line is the deck title, not a card.
pub const title_line = true;
/// Operator bytes beyond `* /` at their level: none. The manual's
/// operator table has no `%` or `\` (hspice-syntax.md X2, UG08
/// pp. 279-283), and a deck that uses one is refused (#203).
pub const remainder_ops = "";
/// No unary `!` either: the manual's table lists `!=` only (X2), and a
/// deck that negates with `!` is refused (#203).
pub const unary_not = false;
/// Cards are case-insensitive: the text is lowercased before parsing.
pub const fold_case = true;
/// Field splitter over one logical line. Braces are name bytes, as
/// HSPICE's manual reads them (hspice-syntax.md T2), not an expression
/// (#203, reversing #153's reading).
pub const Split = Fields("'\"", false);

/// Logical-line iterator; a joined line is allocated in `arena`.
pub const Lines = struct {
    rest: []const u8,
    arena: Allocator,

    /// A token that ends in a `$` comment [L3]: a number with optional
    /// scale and unit letters (`1k`, `1w`, `.5e-3v`).
    fn isNumberToken(t: []const u8) bool {
        var i: usize = 0;
        if (i < t.len and (t[i] == '+' or t[i] == '-')) i += 1;
        const mantissa = i;
        while (i < t.len and std.ascii.isDigit(t[i])) i += 1;
        const int_digits = i > mantissa;
        if (i < t.len and t[i] == '.') i += 1;
        const frac = i;
        while (i < t.len and std.ascii.isDigit(t[i])) i += 1;
        if (!int_digits and i == frac) return false;
        if (i + 1 < t.len and std.ascii.toLower(t[i]) == 'e' or i + 1 < t.len and std.ascii.toLower(t[i]) == 'd') {
            var j = i + 1;
            if (t[j] == '+' or t[j] == '-') j += 1;
            const exp = j;
            while (j < t.len and std.ascii.isDigit(t[j])) j += 1;
            if (j > exp) i = j;
        }
        while (i < t.len and std.ascii.isAlphabetic(t[i])) i += 1;
        return i == t.len;
    }

    /// A physical line's text and how it continues.
    const Part = struct { text: []const u8, brk: Break };

    /// `line` up to its inline `$` comment [L3], trimmed, and its trailing
    /// break [L6]. The comment is a `$` after a separator (blank, comma,
    /// `=`, parenthesis), after a closing quote or after a number token,
    /// outside quotes; a `$` inside a name is a name character (`a$b`).
    /// A line with a comment has no break: the backslashes are not at its
    /// end. `quote` is the quote byte open at the start of the line (0
    /// for none) and is left as the one open at its end, so a string
    /// continued by a backslash keeps its `$` signs.
    fn read(line: []const u8, quote: *u8) Part {
        const text = stripComment(line, quote);
        if (text.len < std.mem.trimEnd(u8, line, " \t").len) return .{ .text = text, .brk = .none };
        return trailingBreak(text);
    }

    fn stripComment(line: []const u8, quote: *u8) []const u8 {
        var token: usize = 0;
        for (line, 0..) |c, i| {
            if (quote.* != 0) {
                if (c == quote.*) {
                    quote.* = 0;
                    token = i + 1;
                }
                continue;
            }
            switch (c) {
                '\'', '"' => quote.* = c,
                ' ', '\t', ',', '=', '(', ')' => token = i + 1,
                '$' => if (token == i or isNumberToken(line[token..i])) return std.mem.trim(u8, line[0..i], " \t"),
                else => {},
            }
        }
        return std.mem.trim(u8, line, " \t");
    }

    /// How a stripped line continues [L6]: `\\` joins the next line with
    /// nothing between, a blank and `\` with one blank.
    const Break = enum { none, glue, keep };

    fn trailingBreak(line: []const u8) Part {
        if (std.mem.endsWith(u8, line, "\\\\"))
            return .{ .text = std.mem.trimEnd(u8, line[0 .. line.len - 2], " \t"), .brk = .glue };
        if (line.len >= 2 and line[line.len - 1] == '\\' and (line[line.len - 2] == ' ' or line[line.len - 2] == '\t'))
            return .{ .text = std.mem.trimEnd(u8, line[0 .. line.len - 1], " \t"), .brk = .keep };
        return .{ .text = line, .brk = .none };
    }

    /// The next logical line, continuations joined; null at the end.
    /// The line borrows `rest`'s text, or lives in `arena` when it joins
    /// continuations. Fails only with `OutOfMemory` while joining.
    pub fn next(self: *Lines) Allocator.Error!?[]const u8 {
        var part: Part = undefined;
        var quote: u8 = 0;
        // L2: comment lines (`*` or `$` first) and blank lines are skipped.
        while (true) {
            const raw = physical_lines.nextPhysicalLine(&self.rest) orelse return null;
            const t = std.mem.trim(u8, raw, " \t");
            if (t.len == 0 or t[0] == '*' or t[0] == '$') continue;
            part = read(t, &quote);
            break;
        }
        const head = part.text;
        var joined: ?std.ArrayList(u8) = null;
        // Each pass consumes a physical line or ends the statement.
        const passes_max = self.rest.len + 1;
        var passes: usize = 0;
        while (true) : (passes += 1) {
            assert(passes <= passes_max);
            if (part.brk != .none) {
                // L6: the next physical line is text, `+` and all.
                const raw = physical_lines.nextPhysicalLine(&self.rest) orelse break;
                const brk = part.brk;
                part = read(std.mem.trim(u8, raw, " \t"), &quote);
                try physical_lines.join(self.arena, &joined, head, if (brk == .glue) "" else " ", part.text);
                continue;
            }
            // L5: a `+` line continues the statement, past comment and blank lines.
            if (physical_lines.startsStatement(self.rest, "+*$")) break;
            const save = self.rest;
            const raw = physical_lines.nextPhysicalLine(&self.rest) orelse break;
            const t = std.mem.trim(u8, raw, " \t");
            if (t.len > 0 and t[0] == '+') {
                quote = 0;
                part = read(t[1..], &quote);
                try physical_lines.join(self.arena, &joined, head, " ", part.text);
            } else if (t.len == 0 or t[0] == '*' or t[0] == '$') {
                continue;
            } else {
                self.rest = save;
                break;
            }
        }
        return if (joined) |j| j.items else head;
    }
};

/// A number literal with an HSPICE scale suffix.
pub fn parseNum(text: []const u8) ?f64 {
    return spice_number.spiceNumber(text, true);
}

/// Whether `line` holds a number with both an exponent and a scale
/// factor (`1e-6u`), which HSPICE forbids (hspice-syntax.md T3, UG08
/// p. 49): a deck that writes one is refused (#203). A number starts
/// after a byte that cannot continue a name, so `n1e3k` is a name.
/// tests/frontend/lower.zig refuses the same lines.
pub fn exponentWithScale(line: []const u8) bool {
    for (0..line.len) |i| {
        if (i > 0 and (std.ascii.isAlphanumeric(line[i - 1]) or line[i - 1] == '_' or line[i - 1] == '.')) continue;
        var j = i;
        if (line[j] == '+' or line[j] == '-') j += 1;
        var digits = false;
        while (j < line.len and std.ascii.isDigit(line[j])) : (j += 1) digits = true;
        if (j < line.len and line[j] == '.') {
            j += 1;
            while (j < line.len and std.ascii.isDigit(line[j])) : (j += 1) digits = true;
        }
        if (!digits or j == line.len or std.mem.indexOfScalar(u8, "eEdD", line[j]) == null) continue;
        j += 1;
        if (j < line.len and (line[j] == '+' or line[j] == '-')) j += 1;
        const exponent = j;
        while (j < line.len and std.ascii.isDigit(line[j])) j += 1;
        if (j == exponent or j == line.len) continue;
        if (std.mem.indexOfScalar(u8, "tgxkmunpfaTGXKMUNPFA", line[j]) != null) return true;
    }
    return false;
}

/// Whether `text`, an unquoted value after `key=`, is one HSPICE takes:
/// a number, or an expression that starts with a letter and holds no
/// blank (hspice-syntax.md X1, CR10 p. 238: "unless the expression
/// begins with an alphabetic character and contains no spaces"). Any
/// other expression needs quotes, and is refused without them (#203).
/// A blank inside a call's parentheses is no blank of the expression:
/// the manual's own `.PARAM x=AGAUSS(1k, 100, 3)` holds them.
pub fn bareValue(text: []const u8) bool {
    const t = std.mem.trim(u8, text, " \t");
    if (t.len == 0) return false;
    if (std.ascii.isAlphabetic(t[0])) {
        var depth: usize = 0;
        for (t) |c| {
            if (c == '(') depth += 1 else if (c == ')') depth -|= 1 else if (depth == 0 and (c == ' ' or c == '\t')) return false;
        }
        return true;
    }
    // A number: sign, mantissa, then exponent, scale and unit bytes.
    var i: usize = 0;
    if (t[0] == '+' or t[0] == '-') i = 1;
    if (i == t.len) return false;
    if (!std.ascii.isDigit(t[i]) and !(t[i] == '.' and i + 1 < t.len and std.ascii.isDigit(t[i + 1]))) return false;
    for (t[i..], i..) |c, k| {
        if (std.ascii.isAlphanumeric(c) or c == '_' or c == '.') continue;
        if ((c == '+' or c == '-') and std.mem.indexOfScalar(u8, "eEdD", t[k - 1]) != null) continue;
        return false;
    }
    return true;
}
test "hspice.bareValue: a number, or a letter-led expression with no blank (#203)" {
    // Hand-listed from CR10 p. 238.
    for ([_][]const u8{ "1k", "-1", "+.5", "2.5e-3", "10pf", "a", "x*2+b", "v(1,0)*2", " cos(2)+sin(2) ", "agauss(1k, 100, 3)", "opt1(1k, 100, 10k)*2" }) |text|
        try std.testing.expect(bareValue(text));
    for ([_][]const u8{ "", "2*a", "6k * 2 / 3", "a + b", "(a+b)", "-a", "!a", ".", "1k-1", "f(a) + 1", "agauss(1k, 1) *2" }) |text|
        try std.testing.expect(!bareValue(text));
}

test "hspice.exponentWithScale finds a number with an exponent and a scale (#203)" {
    // Hand-listed from T3: an exponent and a scale together, never a name.
    for ([_][]const u8{ "r1 a 0 1e-6u", "c1 a 0 c=2.5e3p", "v1 a 0 '1d3k*2'", "r1 a 0 -1e+2meg" }) |line|
        try std.testing.expect(exponentWithScale(line));
    for ([_][]const u8{ "r1 a 0 1e-6", "r1 a 0 1u", "r1 n1e3k 0 1k", "r1 a 0 1e3v", "r1 a 0 1e", "r1 a_1e3k 0 1" }) |line|
        try std.testing.expect(!exponentWithScale(line));
}
