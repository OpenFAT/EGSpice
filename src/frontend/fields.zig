//! One logical line to fields. In: a line. Out: its fields, slices of the
//! line: `=`, `(`, `)` and `,` are one-byte fields, a `{...}` or quoted
//! field keeps its delimiters (its kind is its first byte), and anything
//! else is a word.
//! Cites: tests/frontend/grammar/{ngspice,hspice,spectre}_lines.peg.
//!
//! Contents:
//! - `pub fn Fields(comptime quotes: []const u8, comptime braces: bool) type`
//! - `pub fn init(line: []const u8) Self`
//! - `pub fn rest(self: Self) []const u8`
//! - `pub fn peek(self: Self) ?[]const u8`
//! - `pub fn nextByte(self: *Self) ?u8`
//! - `pub fn takeEq(self: *Self) bool`
//! - `pub fn isWord(f: []const u8) bool`
//! - `pub fn isQuote(c: u8) bool`
//! - `pub fn body(f: []const u8) []const u8`
//! - `pub fn next(self: *Self) ?[]const u8`

const std = @import("std");
const assert = std.debug.assert;

/// A field splitter over one line. `quotes` are the quote bytes, `braces`
/// keeps `{...}` whole. A comptime generic (axis: the dialect's quoting;
/// three cases today, one per dialect, plus the include reader's).
pub fn Fields(comptime quotes: []const u8, comptime braces: bool) type {
    return struct {
        line: []const u8,
        pos: usize = 0,

        const Self = @This();

        /// A splitter at the start of `line`; fields borrow `line`.
        pub fn init(line: []const u8) Self {
            return .{ .line = line };
        }

        /// The unread part of the line, blank-trimmed.
        pub fn rest(self: Self) []const u8 {
            return std.mem.trim(u8, self.line[self.pos..], " \t");
        }

        /// The next field, without consuming it.
        pub fn peek(self: Self) ?[]const u8 {
            var copy = self;
            return copy.next();
        }

        /// First byte of the next field, skipping blanks but not consuming it.
        pub fn nextByte(self: *Self) ?u8 {
            while (self.pos < self.line.len and (self.line[self.pos] == ' ' or self.line[self.pos] == '\t')) self.pos += 1;
            return if (self.pos < self.line.len) self.line[self.pos] else null;
        }

        /// Consumes a `=` field if one is next: the word just read is a key.
        pub fn takeEq(self: *Self) bool {
            if (self.nextByte() != '=') return false;
            self.pos += 1;
            return true;
        }

        /// True unless `f` is a punctuation, quoted or braced field.
        /// Asserts `f` is non-empty, as every field `next` returns is.
        pub fn isWord(f: []const u8) bool {
            assert(f.len > 0);
            return !breaks[f[0]];
        }

        const breaks: [256]bool = blk: {
            var t: [256]bool = @splat(false);
            for (" \t=(),") |c| t[c] = true;
            for (quotes) |q| t[q] = true;
            if (braces) t['{'] = true;
            break :blk t;
        };

        /// True for this dialect's quote bytes.
        pub fn isQuote(c: u8) bool {
            inline for (quotes) |q| if (c == q) return true;
            return false;
        }

        /// A `{...}` or quoted field without its delimiters; an unterminated
        /// one loses only its opening byte. Asserts `f` is non-empty.
        pub fn body(f: []const u8) []const u8 {
            assert(f.len > 0);
            const close: u8 = if (f[0] == '{') '}' else f[0];
            return if (f.len >= 2 and f[f.len - 1] == close) f[1 .. f.len - 1] else f[1..];
        }

        /// Consumes and returns the next field; null at the end of the line.
        /// An unterminated quote or brace runs to the end of the line.
        pub fn next(self: *Self) ?[]const u8 {
            const line = self.line;
            while (self.pos < line.len and (line[self.pos] == ' ' or line[self.pos] == '\t')) self.pos += 1;
            if (self.pos >= line.len) return null;
            const start = self.pos;
            const c = line[start];
            switch (c) {
                '=', '(', ')', ',' => {
                    self.pos += 1;
                    return line[start..self.pos];
                },
                else => {},
            }
            if (isQuote(c)) {
                const end = std.mem.indexOfScalarPos(u8, line, start + 1, c) orelse line.len;
                self.pos = @min(end + 1, line.len);
                return line[start..self.pos];
            }
            if (braces and c == '{') {
                var depth: usize = 0;
                var i = start;
                while (i < line.len) : (i += 1) {
                    if (line[i] == '{') depth += 1;
                    if (line[i] == '}') {
                        depth -= 1;
                        if (depth == 0) break;
                    }
                }
                self.pos = @min(i + 1, line.len);
                return line[start..self.pos];
            }
            while (self.pos < line.len and !breaks[line[self.pos]]) self.pos += 1;
            assert(self.pos > start);
            return line[start..self.pos];
        }
    };
}

test Fields {
    const F = Fields("'", true);
    var f = F.init("r1 a=1 'x y' {a {b}} (c,d)");
    try std.testing.expectEqualStrings("r1", f.peek().?);
    for ([_][]const u8{ "r1", "a", "=", "1", "'x y'", "{a {b}}", "(", "c", ",", "d", ")" }) |want|
        try std.testing.expectEqualStrings(want, f.next().?);
    try std.testing.expectEqual(null, f.next());
    try std.testing.expectEqual(null, f.peek());
    // Unterminated quotes and braces run to the end of the line.
    f = F.init("'abc");
    try std.testing.expectEqualStrings("abc", F.body(f.next().?));
    f = F.init("{a {b} c");
    try std.testing.expectEqualStrings("a {b} c", F.body(f.next().?));
    try std.testing.expectEqualStrings("", F.body("'"));
    try std.testing.expectEqualStrings("", F.body("''"));
    f = F.init("k = v  ");
    try std.testing.expectEqualStrings("k", f.next().?);
    try std.testing.expect(f.takeEq());
    try std.testing.expectEqualStrings("v", f.rest());
    try std.testing.expect(!f.takeEq());
    try std.testing.expectEqual('v', f.nextByte());
    try std.testing.expect(F.isWord("abc") and !F.isWord("=") and !F.isWord("'x'") and !F.isWord("{x}"));
    // HSPICE: `"` quotes too, and braces are name bytes (#203).
    var h = Fields("'\"", false).init("{a+b} {a b} \"b c\"");
    try std.testing.expectEqualStrings("{a+b}", h.next().?);
    try std.testing.expectEqualStrings("{a", h.next().?);
    try std.testing.expectEqualStrings("b}", h.next().?);
    try std.testing.expect(Fields("'\"", false).isWord("{x}"));
    try std.testing.expectEqualStrings("\"b c\"", h.next().?);
    try std.testing.expect(Fields("'\"", false).isQuote('"') and !Fields("'", true).isQuote('"'));
}
