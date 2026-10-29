//! Spectre tokens and expressions. In: one Spectre logical line. Out: its
//! tokens (`tokenize`), and any expression among them as ESPice expression
//! text, fully parenthesized so expr.zig sees Spectre's tree (`expression`).
//! Cites: dev/research/spectre-syntax.md §2.2 (names and escapes), §6.1
//! (operators), §6.2 (functions), §6.3 (built-in constants);
//! tests/frontend/grammar/spectre.peg.
//!
//! Contents:
//! - `pub const Error = error{ OutOfMemory, ParseError, UnsupportedCard }`
//! - `pub const Tok = struct`
//! - `pub const Kind = enum { word, string, punct, op }`
//! - `pub fn is(t: Tok, s: []const u8) bool`
//! - `pub const constants: std.StaticStringMap(f64)`
//! - `pub const depth_max = 256`
//! - `pub fn tokenize(arena: Allocator, line: []const u8) Error![]const Tok`
//! - `pub fn expression(arena: Allocator, toks: []const Tok) Error![]const u8`

const std = @import("std");
const assert = std.debug.assert;
const testing = std.testing;
const Allocator = std.mem.Allocator;
const lines = @import("lines.zig");

/// `ParseError`: a malformed statement; `UnsupportedCard`: a statement or
/// form ESPice does not read (logged with the line).
pub const Error = error{ OutOfMemory, ParseError, UnsupportedCard };

/// One token: a word (name, number, node), a string, punctuation or an
/// operator; `text` borrows the line.
pub const Tok = struct {
    kind: Kind,
    text: []const u8,

    /// What the token is.
    pub const Kind = enum { word, string, punct, op };

    /// Whether the token is punctuation, an operator or a word spelled `s`.
    pub fn is(t: Tok, s: []const u8) bool {
        return (t.kind == .punct or t.kind == .op or t.kind == .word) and std.mem.eql(u8, t.text, s);
    }
};

/// §6.3: the built-in constants, as their values.
pub const constants = std.StaticStringMap(f64).initComptime(.{
    .{ "M_E", std.math.e },             .{ "M_LOG2E", std.math.log2e },            .{ "M_LOG10E", std.math.log10e },
    .{ "M_LN2", std.math.ln2 },         .{ "M_LN10", std.math.ln10 },              .{ "M_PI", std.math.pi },
    .{ "M_TWO_PI", 2.0 * std.math.pi }, .{ "M_PI_2", std.math.pi / 2.0 },          .{ "M_PI_4", std.math.pi / 4.0 },
    .{ "M_1_PI", 1.0 / std.math.pi },   .{ "M_2_PI", 2.0 / std.math.pi },          .{ "M_2_SQRTPI", std.math.two_sqrtpi },
    .{ "M_SQRT2", std.math.sqrt2 },     .{ "M_SQRT1_2", std.math.sqrt1_2 },        .{ "M_DEGPERRAD", 180.0 / std.math.pi },
    .{ "P_Q", 1.6021918e-19 },          .{ "P_C", 2.997924562e8 },                 .{ "P_K", 1.3806226e-23 },
    .{ "P_H", 6.6260755e-34 },          .{ "P_EPS0", 8.85418792394420013968e-12 }, .{ "P_U0", 4.0e-7 * std.math.pi },
    .{ "P_CELSIUS0", 273.15 },
});

/// Nesting an expression or a chain of one-line `if`s may reach; deeper is
/// a ParseError, not a stack overflow.
pub const depth_max = 256;

/// A Spectre expression -> ESPice expression text, fully parenthesized so
/// expr.zig sees Spectre's tree (§6.1). `toks` is consumed from `at`.
const Expr = struct {
    toks: []const Tok,
    at: usize = 0,
    out: *std.ArrayList(u8),
    arena: Allocator,
    depth: u32 = 0,

    fn peek(e: *Expr) ?Tok {
        return if (e.at < e.toks.len) e.toks[e.at] else null;
    }

    fn take(e: *Expr) ?Tok {
        const t = e.peek() orelse return null;
        e.at += 1;
        return t;
    }

    fn put(e: *Expr, s: []const u8) Error!void {
        try e.out.appendSlice(e.arena, s);
    }

    /// Binary levels, loosest first; `**` is the right-associative one.
    const levels = [_][]const []const u8{
        &.{"||"}, &.{"&&"}, &.{ "==", "!=" }, &.{ "<", "<=", ">", ">=" }, &.{ "+", "-" }, &.{ "*", "/", "%" },
    };

    fn ternary(e: *Expr) Error!void {
        if (e.depth == depth_max) return error.ParseError;
        e.depth += 1;
        defer e.depth -= 1;
        const mark = e.out.items.len;
        try e.binary(0);
        if (e.peek()) |t| if (t.is("?")) {
            _ = e.take();
            const cond = try e.arena.dupe(u8, e.out.items[mark..]);
            e.out.shrinkRetainingCapacity(mark);
            try e.put("(");
            try e.put(cond);
            try e.put("?");
            try e.ternary();
            if (!(e.take() orelse return error.ParseError).is(":")) return error.ParseError;
            try e.put(":");
            try e.ternary();
            try e.put(")");
        };
    }

    fn binary(e: *Expr, level: usize) Error!void {
        if (level == levels.len) return e.power();
        const mark = e.out.items.len;
        try e.binary(level + 1);
        while (e.peek()) |t| {
            if (t.kind != .op) break;
            const op = for (levels[level]) |o| {
                if (std.mem.eql(u8, t.text, o)) break o;
            } else break;
            _ = e.take();
            const lhs = try e.arena.dupe(u8, e.out.items[mark..]);
            e.out.shrinkRetainingCapacity(mark);
            try e.put("(");
            try e.put(lhs);
            try e.put(op);
            try e.binary(level + 1);
            try e.put(")");
        }
    }

    /// `a ** b ** c` is `a ** (b ** c)`; a unary operand binds tighter
    /// than `**` (§6.1: `-2**2` is 4).
    fn power(e: *Expr) Error!void {
        const mark = e.out.items.len;
        try e.unary();
        if (e.peek()) |t| if (t.is("**")) {
            _ = e.take();
            const lhs = try e.arena.dupe(u8, e.out.items[mark..]);
            e.out.shrinkRetainingCapacity(mark);
            try e.put("(");
            try e.put(lhs);
            try e.put("**");
            if (e.depth == depth_max) return error.ParseError;
            e.depth += 1;
            defer e.depth -= 1;
            try e.power();
            try e.put(")");
        };
    }

    fn unary(e: *Expr) Error!void {
        const t = e.peek() orelse return error.ParseError;
        if (t.kind == .op and (t.is("-") or t.is("+") or t.is("!"))) {
            _ = e.take();
            if (e.depth == depth_max) return error.ParseError;
            e.depth += 1;
            defer e.depth -= 1;
            try e.put("(");
            try e.put(t.text);
            try e.unary();
            try e.put(")");
            return;
        }
        if (t.kind == .op) return error.UnsupportedCard; // bitwise, shift
        return e.primary();
    }

    fn primary(e: *Expr) Error!void {
        const t = e.take() orelse return error.ParseError;
        switch (t.kind) {
            .string => return e.put(t.text),
            .punct => {
                if (!t.is("(")) return error.ParseError;
                try e.put("(");
                try e.ternary();
                if (!(e.take() orelse return error.ParseError).is(")")) return error.ParseError;
                return e.put(")");
            },
            .op => unreachable,
            .word => {},
        }
        const w = t.text;
        if (std.ascii.isDigit(w[0]) or w[0] == '.') return e.number(w);
        if (e.peek()) |n| if (n.is("(")) return e.call(w);
        if (constants.get(w)) |v| return e.out.print(e.arena, "{e}", .{v});
        // `temp` is the circuit temperature (§6.3); `$time` a behavioural
        // expression's time.
        if (std.mem.eql(u8, w, "temp")) return e.put("temper");
        if (std.mem.eql(u8, w, "$time")) return e.put("time");
        return e.put(w);
    }

    fn number(e: *Expr, w: []const u8) Error!void {
        const v = lines.spectre.parseNum(w) orelse return error.ParseError;
        try e.out.print(e.arena, "{e}", .{v});
    }

    /// A call. `fmod` is `%` and `hypot` its square root; the rest keep
    /// their names, which expr.zig reads (§6.2).
    fn call(e: *Expr, name: []const u8) Error!void {
        _ = e.take(); // (
        var args: std.ArrayList([]const u8) = .empty;
        const first = e.peek() orelse return error.ParseError;
        if (first.is(")")) {
            _ = e.take();
        } else while (true) {
            const mark = e.out.items.len;
            try e.ternary();
            try args.append(e.arena, try e.arena.dupe(u8, e.out.items[mark..]));
            e.out.shrinkRetainingCapacity(mark);
            const sep = e.take() orelse return error.ParseError;
            if (sep.is(")")) break;
            if (!sep.is(",")) return error.ParseError;
        }
        if (std.mem.eql(u8, name, "fmod") and args.items.len == 2)
            return e.out.print(e.arena, "({s}%{s})", .{ args.items[0], args.items[1] });
        if (std.mem.eql(u8, name, "hypot") and args.items.len == 2)
            return e.out.print(e.arena, "sqrt({s}*{s}+{s}*{s})", .{ args.items[0], args.items[0], args.items[1], args.items[1] });
        if (std.mem.eql(u8, name, "sign") and args.items.len == 2)
            return e.out.print(e.arena, "(sgn({s})*abs({s}))", .{ args.items[1], args.items[0] });
        if (std.mem.eql(u8, name, "atan2")) return error.UnsupportedCard;
        try e.put(name);
        try e.put("(");
        for (args.items, 0..) |a, i| {
            if (i > 0) try e.put(",");
            try e.put(a);
        }
        try e.put(")");
    }
};

/// Splits one Spectre line into words (names, numbers, nodes; a backslash
/// escapes the next byte, §2.2), strings, punctuation and operators.
pub fn tokenize(arena: Allocator, line: []const u8) Error![]const Tok {
    var out: std.ArrayList(Tok) = .empty;
    var i: usize = 0;
    while (i < line.len) {
        const c = line[i];
        if (c == ' ' or c == '\t') {
            i += 1;
            continue;
        }
        const start = i;
        if (c == '"') {
            i += 1;
            while (i < line.len and line[i] != '"') : (i += 1) if (line[i] == '\\') {
                i += 1;
            };
            if (i >= line.len) return error.ParseError;
            i += 1;
            try out.append(arena, .{ .kind = .string, .text = line[start..i] });
            continue;
        }
        if (isWordByte(c) or c == '\\') {
            while (i < line.len and (isWordByte(line[i]) or line[i] == '\\')) {
                if (line[i] == '\\') i += 1;
                i += 1;
                // An exponent's sign belongs to a number (`1e-9`).
                if (i + 1 < line.len and (line[i] == '+' or line[i] == '-') and
                    (line[i - 1] == 'e' or line[i - 1] == 'E') and std.ascii.isDigit(line[start]) and std.ascii.isDigit(line[i + 1])) i += 1;
            }
            try out.append(arena, .{ .kind = .word, .text = line[start..@min(i, line.len)] });
            continue;
        }
        const two = if (i + 1 < line.len) line[i .. i + 2] else "";
        const pair = for ([_][]const u8{ "**", "<=", ">=", "==", "!=", "&&", "||", "<<", ">>", "~^", "^~" }) |op| {
            if (std.mem.eql(u8, two, op)) break true;
        } else false;
        if (pair) {
            try out.append(arena, .{ .kind = .op, .text = two });
            i += 2;
        } else if (std.mem.indexOfScalar(u8, "()[]{}=,:;", c) != null) {
            try out.append(arena, .{ .kind = .punct, .text = line[i .. i + 1] });
            i += 1;
        } else {
            try out.append(arena, .{ .kind = .op, .text = line[i .. i + 1] });
            i += 1;
        }
    }
    assert(i >= line.len);
    return out.items;
}

/// The ESPice text of the Spectre expression `toks`, all of them.
pub fn expression(arena: Allocator, toks: []const Tok) Error![]const u8 {
    if (toks.len == 0) return error.ParseError;
    var out: std.ArrayList(u8) = .empty;
    var e: Expr = .{ .toks = toks, .out = &out, .arena = arena };
    try e.ternary();
    assert(e.depth == 0);
    if (e.at != toks.len) {
        // §6.1's bitwise and shift operators are Spectre, not read here.
        const t = toks[e.at];
        if (t.kind == .op and std.mem.indexOfAny(u8, t.text, "&|^~<>") != null) return error.UnsupportedCard;
        return error.ParseError;
    }
    return out.items;
}

// ---- private ----

fn isWordByte(c: u8) bool {
    return std.ascii.isAlphanumeric(c) or c == '_' or c == '!' or c == '$' or c == '.';
}

test "expressions keep Spectre's tree (§6.1)" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const cases = [_][2][]const u8{
        .{ "-2**2", "((-2e0)**2e0)" },
        .{ "2**3**2", "(2e0**(3e0**2e0))" },
        .{ "a<b==c", "((a<b)==c)" },
        .{ "1+2*3", "(1e0+(2e0*3e0))" },
        .{ "c?1k:2M", "(c?1e3:2e6)" },
        .{ "fmod(7,3)", "(7e0%3e0)" },
        .{ "temp+M_PI", "(temper+3.141592653589793e0)" },
    };
    for (cases) |c| try testing.expectEqualStrings(c[1], try expression(a, try tokenize(a, c[0])));
    try testing.expectError(error.UnsupportedCard, expression(a, try tokenize(a, "a&b")));
}
