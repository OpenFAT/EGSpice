//! The postfix expression tape's rows. In: nothing (definitions only).
//! Out: the opcode and built-in tables every expression stage reads:
//! `Code`, `Fn`, `Op`, the compile `Scratch`, and the built-in name maps.
//! Cites: dev/research/ngspice-syntax.md §2.2 and ngspice xpressn.c for
//! numparam's operators and built-ins; Star-Hspice Table 10-1 for the
//! HSPICE dialect's sqrt, log and pow (#39).
//!
//! Contents:
//! - `pub const Error = error{ OutOfMemory, ParseError }`
//! - `pub const CompileError = Error || error{UnsupportedCard}`
//! - `pub const none: u32`
//! - `pub const Code = enum(u8)`
//! - `pub const Fn = enum(u8)`
//! - `pub const Op = struct`
//! - `pub const Scratch = struct`
//! - `pub const Mark = struct`
//! - `pub fn mark(s: Scratch) Mark`
//! - `pub fn reset(s: *Scratch, m: Mark) void`
//! - `pub const fns: std.StaticStringMap(Fn)`
//! - `pub const hspice_fns: std.StaticStringMap(Fn)`
//! - `pub fn isDistributionName(name: []const u8) bool`
//! - `pub fn isDistribution(f: Fn) bool`

const std = @import("std");
const assert = std.debug.assert;

/// `ParseError`: malformed expression text, or nesting past `max_depth`.
/// On either error `compile` leaves what it appended in the `Scratch`;
/// roll it back with `Scratch.mark` and `Scratch.reset`. `fold` and `eval`
/// fail only with `OutOfMemory`.
pub const Error = error{ OutOfMemory, ParseError };
/// `compile`'s errors: `UnsupportedCard` for `%` or `\` used as an operator,
/// or a unary `!`, in a dialect without it (`S.remainder_ops`,
/// `S.unary_not`, #203).
pub const CompileError = Error || error{UnsupportedCard};
/// Absent operand.
pub const none = std.math.maxInt(u32);

/// Postfix opcode; `a`/`b` operands as noted, none for the operators.
pub const Code = enum(u8) {
    /// a: constant pool index.
    num,
    /// A name no scope defines. As compiled, a is its index into
    /// `Scratch.names`; in final ops a = 1 for `l`, `w`, `mult`, 2 for
    /// `time`, 3 for `temper`, 0 otherwise.
    ident,
    /// A swept global parameter kept symbolic: a indexes the live table
    /// (`fold`'s `live`). Only netlist.zig `subst` emits it.
    live,
    /// `v(p[,n])`: a, b are the nets (names before `subst`), `none` if absent.
    vprobe,
    /// `i(device)`: a is the device name (a name before `subst`, its
    /// flattened `Netlist.pool` index after).
    iprobe,
    neg,
    not,
    add,
    sub,
    mul,
    div,
    /// `a % b`: a - b·trunc(a/b), as numparam's `%` (ngspice-syntax.md §2.2).
    mod,
    /// `a \ b`: trunc(|a/b|), numparam's integer division.
    idiv,
    pow,
    lt,
    gt,
    le,
    ge,
    eq,
    ne,
    @"and",
    @"or",
    /// a: `Fn`, b: argument count.
    call,
    /// A call to a function outside `Fn`, as compiled: a is its name (an
    /// index into `Scratch.names`), b the argument count. netlist.zig
    /// `subst` inlines a `.func` it names and turns any other into a
    /// `call` of `Fn.other`, so final ops never hold one.
    ucall,
};

/// Built-in functions; `other` is any name the table does not know. The
/// statistical distributions `agauss`, `gauss`, `unif`, `aunif` and HSPICE's
/// `limit(nom, var)` fold to their nominal, the first argument; a Monte Carlo
/// trial draws them instead (`eval`). `table(x, d, x1, y1, ...)` is the
/// smoothed transfer of an E/G TABLE or PWL(1) card, built only by the
/// B-source tape (models/bsource.va opcode 37).
pub const Fn = enum(u8) { sqrt, abs, min, max, pow, pwr, exp, ln, log, log10, sin, cos, tan, atan, floor, ceil, ternary, tanh, sqr, int, nint, sgn, sinh, cosh, asin, acos, asinh, acosh, atanh, agauss, gauss, unif, aunif, limit, table, other, hsqrt, hlog, hlog10, hpow };

/// One postfix op.
pub const Op = struct { code: Code, a: u32 = 0, b: u32 = 0 };

comptime {
    // A row of the postfix tape: a 1-byte opcode and two u32 operands.
    assert(@sizeOf(Code) == 1);
    assert(@sizeOf(Fn) == 1);
    assert(@sizeOf(Op) == 12);
}

/// Compile output before parameters are spliced in.
pub const Scratch = struct {
    ops: std.ArrayList(Op) = .empty,
    /// The constant pool `.num` operands index.
    consts: std.ArrayList(f64) = .empty,
    /// Slices of the compiled text that `.ident`, probe and `.ucall`
    /// operands index.
    names: std.ArrayList([]const u8) = .empty,

    /// Table lengths to roll back to.
    pub const Mark = struct { ops: usize, consts: usize, names: usize };

    /// The current table lengths, for a later `reset`.
    pub fn mark(s: Scratch) Mark {
        return .{ .ops = s.ops.items.len, .consts = s.consts.items.len, .names = s.names.items.len };
    }

    /// Drops everything appended since `m`.
    pub fn reset(s: *Scratch, m: Mark) void {
        assert(m.ops <= s.ops.items.len);
        assert(m.names <= s.names.items.len);
        s.ops.shrinkRetainingCapacity(m.ops);
        s.consts.shrinkRetainingCapacity(m.consts);
        s.names.shrinkRetainingCapacity(m.names);
    }
};

/// Built-in names, lower case, to their `Fn`.
pub const fns = std.StaticStringMap(Fn).initComptime(.{
    .{ "sqrt", .sqrt },           .{ "abs", .abs },     .{ "min", .min },       .{ "max", .max },
    .{ "pow", .pow },             .{ "pwr", .pwr },     .{ "exp", .exp },       .{ "ln", .ln },
    .{ "log", .log },             .{ "log10", .log10 }, .{ "sin", .sin },       .{ "cos", .cos },
    .{ "tan", .tan },             .{ "atan", .atan },   .{ "floor", .floor },   .{ "ceil", .ceil },
    .{ "ternary", .ternary },     .{ "tanh", .tanh },   .{ "agauss", .agauss }, .{ "gauss", .gauss },
    .{ "unif", .unif },           .{ "aunif", .aunif }, .{ "limit", .limit },   .{ "table", .table },
    // numparam's other built-ins (ngspice-syntax.md §2.2, xpressn.c fmathS).
    .{ "sqr", .sqr },             .{ "int", .int },     .{ "nint", .nint },     .{ "sgn", .sgn },
    .{ "sinh", .sinh },           .{ "cosh", .cosh },   .{ "asin", .asin },     .{ "acos", .acos },
    .{ "asinh", .asinh },         .{ "acosh", .acosh }, .{ "atanh", .atanh },   .{ "arctan", .atan },
    .{ "ternary_fcn", .ternary },
});

/// The HSPICE dialect's meanings [Star-Hspice Table 10-1]: sqrt(-x) is
/// -sqrt(x), log and log10 of -x are -log(x), and pow uses the integer
/// part of its exponent (#39).
pub const hspice_fns = blk: {
    const overrides = .{ .{ "sqrt", .hsqrt }, .{ "log", .hlog }, .{ "log10", .hlog10 }, .{ "pow", .hpow } };
    var kvs: [fns.keys().len]struct { []const u8, Fn } = undefined;
    for (fns.keys(), fns.values(), &kvs) |k, v, *out| {
        out.* = .{ k, v };
        for (overrides) |o| if (std.mem.eql(u8, k, o[0])) {
            out.* = .{ k, o[1] };
        };
    }
    break :blk std.StaticStringMap(Fn).initComptime(kvs);
};

/// True when `name` (lower case) calls a function a Monte Carlo trial draws.
pub fn isDistributionName(name: []const u8) bool {
    const f = fns.get(name) orelse return false;
    return isDistribution(f);
}

/// True for the functions a Monte Carlo trial draws.
pub fn isDistribution(f: Fn) bool {
    return switch (f) {
        .agauss, .gauss, .unif, .aunif, .limit => true,
        else => false,
    };
}
