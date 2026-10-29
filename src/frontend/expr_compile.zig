//! Expression text to postfix. In: expression text in one dialect `S`
//! (lines.zig's ngspice, hspice or spectre). Out: `Op` rows, constants and
//! names appended to a `Scratch`; names and probe arguments stay indices
//! into `Scratch.names` until netlist.zig `subst` splices parameters in.
//! A recursive-descent parser bounded by `max_depth`.
//! Cites: dev/research/ngspice-syntax.md §2.2, §2.6; ngspice xpressn.c.
//!
//! Contents:
//! - `pub const max_depth = 512`
//! - `pub fn compile(comptime S: type, gpa: Allocator, s: *Scratch, text: []const u8, pos: usize) CompileError!usize`
//! - `pub fn compileAll(comptime S: type, gpa: Allocator, s: *Scratch, text: []const u8) CompileError!void`
//! - `pub fn numberLen(text: []const u8) usize`
//! - `pub fn isOperator(c: u8) bool`

const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;
const lines = @import("lines.zig");
const expr_ops = @import("expr_ops.zig");
const CompileError = expr_ops.CompileError;
const Scratch = expr_ops.Scratch;
const Op = expr_ops.Op;
const Code = expr_ops.Code;
const Fn = expr_ops.Fn;
const none = expr_ops.none;

/// Recursion bound of `compile`, in parser calls: a level of parentheses,
/// quotes or call arguments costs two, a ternary branch one, a unary
/// operator one or two. Deeper input is a ParseError rather than a stack overflow.
pub const max_depth = 512;

/// Compiles the expression starting at `text[pos]` into `s` and returns
/// where it ends. Grammar: ternary `?:` loosest, then `||`, `&&`,
/// comparisons, `+ -`, `* /`, `^`/`**` (all left associative); unary `-`/`+`
/// bind below power. Nesting deeper than `max_depth` is a ParseError;
/// `%` or `\` after an operand, where `S.remainder_ops` lacks it, is
/// `UnsupportedCard`. Invalidates slices of `s`'s tables, which grow in `gpa`.
pub fn compile(comptime S: type, gpa: Allocator, s: *Scratch, text: []const u8, pos: usize) CompileError!usize {
    var p: Compiler(S) = .{ .text = text, .pos = pos, .gpa = gpa, .s = s };
    try p.bin(0);
    assert(p.pos >= pos);
    assert(p.depth == 0);
    return p.pos;
}

/// `compile` over the whole of `text`; trailing input is a ParseError.
/// Invalidates slices of `s`'s tables, which grow in `gpa`.
pub fn compileAll(comptime S: type, gpa: Allocator, s: *Scratch, text: []const u8) CompileError!void {
    var p: Compiler(S) = .{ .text = text, .gpa = gpa, .s = s };
    try p.bin(0);
    assert(p.depth == 0);
    if (p.peek() != null) return error.ParseError;
}

/// Length of the number literal `text` starts with: digits and dots, an
/// exponent, then a letter suffix (`2.5e-3`, `10meg`). `E` is an exponent
/// too, as `lines.zig` reads it, for Spectre's case-kept text.
pub fn numberLen(text: []const u8) usize {
    var i: usize = 0;
    while (i < text.len) : (i += 1) {
        const ch = text[i];
        if (std.ascii.isDigit(ch) or ch == '.') continue;
        if ((ch == 'e' or ch == 'E') and i + 1 < text.len and
            (std.ascii.isDigit(text[i + 1]) or
                ((text[i + 1] == '-' or text[i + 1] == '+') and i + 2 < text.len and std.ascii.isDigit(text[i + 2]))))
        {
            i += 1;
            if (text[i] == '+' or text[i] == '-') i += 1;
            continue;
        }
        break;
    }
    while (i < text.len and std.ascii.isAlphabetic(text[i])) i += 1;
    return i;
}

/// A byte that continues an expression after an operand: the binary and
/// ternary operator bytes, `%` and `\` included. `compile` reads those two
/// where the dialect has them and refuses them elsewhere (#203).
pub fn isOperator(c: u8) bool {
    return switch (c) {
        '|', '&', '<', '>', '=', '!', '+', '-', '*', '/', '%', '\\', '^', '?' => true,
        else => false,
    };
}

// ---- private ----

fn Compiler(comptime S: type) type {
    return struct {
        text: []const u8,
        pos: usize = 0,
        gpa: Allocator,
        s: *Scratch,
        depth: u16 = 0,

        const P = @This();

        fn peek(p: *P) ?u8 {
            while (p.pos < p.text.len and (p.text[p.pos] == ' ' or p.text[p.pos] == '\t')) p.pos += 1;
            return if (p.pos < p.text.len) p.text[p.pos] else null;
        }

        fn emit(p: *P, op: Op) CompileError!void {
            try p.s.ops.append(p.gpa, op);
        }

        fn name(p: *P, n: []const u8) CompileError!u32 {
            try p.s.names.append(p.gpa, n);
            return @intCast(p.s.names.items.len - 1);
        }

        fn bin(p: *P, min_prec: u8) CompileError!void {
            if (p.depth == max_depth) return error.ParseError;
            p.depth += 1;
            defer p.depth -= 1;
            try p.unary();
            while (true) {
                const c = p.peek() orelse break;
                if (c == '?' and min_prec == 0) {
                    p.pos += 1;
                    try p.bin(0);
                    if (p.peek() != ':') return error.ParseError;
                    p.pos += 1;
                    try p.bin(0);
                    try p.emit(.{ .code = .call, .a = @backingInt(Fn.ternary), .b = 3 });
                    continue;
                }
                // A dialect without `%` or `\` refuses the deck by name
                // rather than end the expression before it (#203).
                if ((c == '%' or c == '\\') and std.mem.indexOfScalar(u8, S.remainder_ops, c) == null) return error.UnsupportedCard;
                const power = c == '*' and p.pos + 1 < p.text.len and p.text[p.pos + 1] == '*';
                const prec: u8 = switch (c) {
                    '|' => 1,
                    '&' => 2,
                    '<', '>', '=', '!' => 3,
                    '+', '-' => 4,
                    '*', '/', '%', '\\' => if (power) 6 else 5,
                    '^' => 6,
                    else => break,
                };
                if (prec < min_prec) break;
                p.pos += 1;
                var code: Code = switch (c) {
                    '|' => .@"or",
                    '&' => .@"and",
                    '<' => .lt,
                    '>' => .gt,
                    '=' => .eq,
                    '!' => .ne,
                    '+' => .add,
                    '-' => .sub,
                    '*' => if (power) .pow else .mul,
                    '/' => .div,
                    '%' => .mod,
                    '\\' => .idiv,
                    else => .pow,
                };
                if (p.pos < p.text.len) {
                    const next = p.text[p.pos];
                    if (power) p.pos += 1;
                    if (next == '=' and (c == '<' or c == '>' or c == '=' or c == '!')) {
                        if (c == '<') code = .le;
                        if (c == '>') code = .ge;
                        p.pos += 1;
                    } else if (c == '<' and next == '>') {
                        // `<>` is `!=` in numparam and B sources (§2.2, §2.6).
                        code = .ne;
                        p.pos += 1;
                    } else if (c == '=' or c == '!') return error.ParseError;
                    if (c == '&' or c == '|') {
                        if (next != c) return error.ParseError;
                        p.pos += 1;
                    }
                }
                try p.bin(prec + 1);
                // `b` marks an HSPICE power, which `fold` reads its own way.
                try p.emit(.{ .code = code, .b = @intFromBool(code == .pow and S == lines.hspice) });
            }
        }

        // Every recursion passes through `bin` or `unary`, so the guard in
        // both bounds them all.
        fn unary(p: *P) CompileError!void {
            if (p.depth == max_depth) return error.ParseError;
            p.depth += 1;
            defer p.depth -= 1;
            const c = p.peek() orelse return error.ParseError;
            switch (c) {
                '-' => {
                    p.pos += 1;
                    try p.bin(6);
                    try p.emit(.{ .code = .neg });
                },
                '+' => {
                    p.pos += 1;
                    try p.bin(6);
                },
                '!' => {
                    // HSPICE has no unary `!` (#203).
                    if (!S.unary_not) return error.UnsupportedCard;
                    p.pos += 1;
                    try p.unary();
                    try p.emit(.{ .code = .not });
                },
                else => try p.atom(),
            }
        }

        /// A probe argument: up to `,` `)` or a blank, kept as a name (node
        /// `10` is a node, and a node `1n` must not read as 1e-9).
        fn nodeArg(p: *P) CompileError![]const u8 {
            _ = p.peek();
            const start = p.pos;
            while (p.pos < p.text.len) : (p.pos += 1) {
                const ch = p.text[p.pos];
                if (ch == ',' or ch == ')' or ch == ' ' or ch == '\t') break;
            }
            if (p.pos == start) return error.ParseError;
            return p.text[start..p.pos];
        }

        fn atom(p: *P) CompileError!void {
            const c = p.peek() orelse return error.ParseError;
            if (c == '"' or c == '\'' or c == '(') {
                p.pos += 1;
                try p.bin(0);
                if (p.peek() != if (c == '(') ')' else c) return error.ParseError;
                p.pos += 1;
                return;
            }
            if (std.ascii.isDigit(c) or c == '.') {
                const start = p.pos;
                p.pos += numberLen(p.text[start..]);
                const n = S.parseNum(p.text[start..p.pos]) orelse return error.ParseError;
                try p.s.consts.append(p.gpa, n);
                return p.emit(.{ .code = .num, .a = @intCast(p.s.consts.items.len - 1) });
            }
            if (!std.ascii.isAlphabetic(c) and c != '_') return error.ParseError;
            const start = p.pos;
            while (p.pos < p.text.len) : (p.pos += 1) {
                const ch = p.text[p.pos];
                if (!(std.ascii.isAlphanumeric(ch) or ch == '_' or ch == '.')) break;
            }
            const word = p.text[start..p.pos];
            if (p.peek() != '(') return p.emit(.{ .code = .ident, .a = try p.name(word) });
            p.pos += 1;
            const probe: ?Code = if (word.len != 1) null else switch (word[0]) {
                'v', 'V' => .vprobe,
                'i', 'I' => .iprobe,
                else => null,
            };
            var args: [2]u32 = .{ none, none };
            var argc: u32 = 0;
            if (p.peek() != ')') while (true) {
                if (probe != null) {
                    const node = try p.nodeArg();
                    if (argc < 2) args[argc] = try p.name(node);
                } else try p.bin(0);
                argc += 1;
                if ((p.peek() orelse return error.ParseError) != ',') break;
                p.pos += 1;
            };
            if (p.peek() != ')') return error.ParseError;
            p.pos += 1;
            if (probe) |code| return p.emit(.{ .code = code, .a = args[0], .b = args[1] });
            const table = if (S == lines.hspice) expr_ops.hspice_fns else expr_ops.fns;
            if (table.get(word)) |f| return p.emit(.{ .code = .call, .a = @backingInt(f), .b = argc });
            try p.emit(.{ .code = .ucall, .a = try p.name(word), .b = argc });
        }
    };
}
