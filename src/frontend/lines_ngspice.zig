//! The ngspice dialect's lines. In: a deck's lowercased bytes. Out:
//! logical lines (title line, `*` comments, `$`/`;` tails, `+` and `\\`
//! continuation) and their fields and numbers.
//! Cites: tests/frontend/grammar/ngspice_lines.peg (the spec);
//! dev/research/ngspice-syntax.md §1.3, §1.5, §1.6, §2.2.
//!
//! Contents:
//! - `pub const title_line = true`
//! - `pub const remainder_ops = "%\\"`
//! - `pub const unary_not = true`
//! - `pub const fold_case = true`
//! - `pub const Split = Fields("'", true)`
//! - `pub const Lines = struct`
//! - `pub fn next(self: *Lines) Allocator.Error!?[]const u8`
//! - `pub fn parseNum(text: []const u8) ?f64`

const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;
const Fields = @import("fields.zig").Fields;
const spice_number = @import("spice_number.zig");
const physical_lines = @import("physical_lines.zig");

/// The first line is the deck title, not a card.
pub const title_line = true;
/// Operator bytes beyond `* /` at their level: numparam's remainder and
/// integer division (ngspice-syntax.md §2.2).
pub const remainder_ops = "%\\";
/// Unary `!` is numparam's logical not (ngspice-syntax.md §2.2).
pub const unary_not = true;
/// Cards are case-insensitive: the text is lowercased before parsing.
pub const fold_case = true;
/// Field splitter over one logical line.
pub const Split = Fields("'", true);

/// Logical-line iterator; a joined line is allocated in `arena`.
/// tests/frontend/grammar/ngspice_lines.peg is the spec; section numbers cite
/// dev/research/ngspice-syntax.md.
pub const Lines = struct {
    rest: []const u8,
    arena: Allocator,

    /// A first byte that makes the line a comment [§1.5]: `*`, `#`, `$`,
    /// `;` silently, and the punctuation ngspice rewrites to `*` with a
    /// warning.
    fn isCommentLead(c: u8) bool {
        return std.mem.indexOfScalar(u8, "*#$;=[]?()&%\"!:,\x0c", c) != null;
    }

    /// `line` up to its end-of-line comment [§1.5], trimmed: `;` or `//`
    /// anywhere, `$` at the start or after a blank, tab or comma, all
    /// outside quotes (a backslash escapes a quote). Braces do not
    /// protect. One scalar pass: cards are short, and stdlib vector
    /// scans cost more in setup than they save.
    fn stripComment(line: []const u8) []const u8 {
        const special = comptime blk: {
            var t: [256]bool = @splat(false);
            for ("'\";/$") |c| t[c] = true;
            break :blk t;
        };
        var quote: u8 = 0;
        var i: usize = 0;
        while (i < line.len) : (i += 1) {
            // Most bytes are none of the switch's: skip them in a tight loop.
            if (quote == 0) {
                while (i < line.len and !special[line[i]]) i += 1;
            }
            if (i == line.len) break;
            const c = line[i];
            if (quote != 0) {
                if (c == '\\') {
                    i += 1;
                } else if (c == quote) quote = 0;
                continue;
            }
            switch (c) {
                '\'', '"' => quote = c,
                ';' => break,
                '/' => if (i + 1 < line.len and line[i + 1] == '/') break,
                '$' => if (i == 0 or line[i - 1] == ' ' or line[i - 1] == '\t' or line[i - 1] == ',') break,
                else => {},
            }
        }
        return std.mem.trim(u8, line[0..@min(i, line.len)], " \t");
    }

    /// True when `line` ends in `\\` and blanks [§1.6]: it continues on
    /// the next line, whatever comment sits before the backslashes.
    fn endsGlued(line: []const u8) bool {
        return std.mem.endsWith(u8, std.mem.trimEnd(u8, line, " \t"), "\\\\");
    }

    /// `line` without its trailing `\\` and blanks.
    fn unglued(line: []const u8) []const u8 {
        const t = std.mem.trimEnd(u8, line, " \t");
        return t[0 .. t.len - 2];
    }

    /// The next logical line, continuations joined; null at the end.
    /// The line borrows `rest`'s text, or lives in `arena` when it joins
    /// continuations. Fails only with `OutOfMemory` while joining.
    pub fn next(self: *Lines) Allocator.Error!?[]const u8 {
        var glued: bool = undefined;
        var head: []const u8 = undefined;
        while (true) {
            const raw = physical_lines.nextPhysicalLine(&self.rest) orelse return null;
            const t = std.mem.trim(u8, raw, " \t");
            if (t.len == 0 or isCommentLead(t[0])) continue;
            // §1.3: a line starting with `@` ends the file.
            if (t[0] == '@') {
                self.rest = "";
                return null;
            }
            // §1.6: a `+` line with nothing to continue is ignored.
            if (t[0] == '+') continue;
            glued = endsGlued(t);
            head = stripComment(if (glued) unglued(t) else t);
            // §1.5: a line holding only an end-of-line comment (`//x`)
            // becomes a comment line, so a `+` after it continues nothing.
            if (!glued and head.len == 0) continue;
            break;
        }
        var joined: ?std.ArrayList(u8) = null;
        // Each pass consumes a physical line or ends the statement.
        const passes_max = self.rest.len + 1;
        var passes: usize = 0;
        while (true) : (passes += 1) {
            assert(passes <= passes_max);
            if (!glued and physical_lines.startsStatement(self.rest, "+*#$")) break;
            const save = self.rest;
            const raw = physical_lines.nextPhysicalLine(&self.rest) orelse break;
            var u = std.mem.trim(u8, raw, " \t");
            if (glued) {
                // §1.6: the line after a `\\` continues the statement as is.
            } else if (u.len > 0 and u[0] == '+') {
                u = u[1..];
            } else if (u.len == 0 or u[0] == '*' or u[0] == '#' or u[0] == '$') {
                continue;
            } else {
                self.rest = save;
                break;
            }
            glued = endsGlued(u);
            try physical_lines.join(self.arena, &joined, head, " ", stripComment(if (glued) unglued(u) else u));
        }
        return if (joined) |j| j.items else head;
    }
};

/// A number literal with an ngspice scale suffix; null when `text` is not one.
pub fn parseNum(text: []const u8) ?f64 {
    return spice_number.spiceNumber(text, false);
}
