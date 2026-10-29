//! How a `.meas` card reads samples: a result's row-major data in, one f64
//! per sample out, from a plain column (real, or one part of a complex one)
//! or a `par()` waveform folded over other columns sample by sample.
//! Cites: ngspice 45 com_measure2.c get_value (the parts of a complex
//! vector), HSPICE `par('expr')` [CR .MEASURE].
//!
//! Contents:
//! - `pub const Part = enum(u8) { real, imag, mag, phase, db, degrees }` with `pub fn fromVectype(u8) Part`
//! - `pub const Column = struct` with `pub fn get(Column, usize) f64`
//! - `pub const Waveform = union(enum) { column: Column, par: Par }` with `pub fn get(Waveform, usize) f64`
//! - `pub const Par = struct` with `pub const cache_len`
//! - `Sample.leaf`, public for `fold`
//! - `pub fn columnIndex(names: []const []const u8, name: []const u8) ?usize`
//! - `pub fn fold(ops: []const core.MeasureOp, leaves: anytype) f64`

const std = @import("std");
const assert = std.debug.assert;
const core = @import("core");

/// Which number a complex sample gives a card (ngspice get_value). A real
/// column ignores it.
pub const Part = enum(u8) {
    real,
    imag,
    mag,
    /// Radians: ngspice's radtodeg() converts only under `set units=degrees`.
    phase,
    db,
    /// `lstb(p)`'s phase in degrees.
    degrees,

    /// Returns the part a clause's `vectype` letter names: `m`, `i`, `p`,
    /// `d`, `g`; any other letter (`r`, 0) reads the real part.
    pub fn fromVectype(vectype: u8) Part {
        return switch (vectype) {
            'm' => .mag,
            'i' => .imag,
            'p' => .phase,
            'd' => .db,
            'g' => .degrees,
            else => .real, // else: 'r' and an untyped column read the real part
        };
    }
};

/// One column of a result: sample `i` sits at `data[i * stride + offset]`,
/// its imaginary part next to it when `complex`. Borrows `data`.
pub const Column = struct {
    data: []const f64,
    stride: usize,
    offset: usize,
    complex: bool,
    part: Part,

    /// Returns sample `i`. Asserts that the data holds it.
    pub fn get(c: Column, i: usize) f64 {
        const at = i * c.stride + c.offset;
        assert(at + @intFromBool(c.complex) < c.data.len);
        const re = c.data[at];
        if (!c.complex) return re;
        const im = c.data[at + 1];
        return switch (c.part) {
            .real => re,
            .imag => im,
            .mag => std.math.hypot(re, im),
            .phase => std.math.atan2(im, re),
            .db => 20 * std.math.log10(std.math.hypot(re, im)),
            .degrees => std.math.radiansToDegrees(std.math.atan2(im, re)),
        };
    }
};

/// A `par()` waveform: postfix `ops` over the columns `names` labels, each
/// read like `base` but at its own offset. Borrows `ops` and `names`.
pub const Par = struct {
    base: Column,
    ops: []const core.MeasureOp,
    names: []const []const u8,
    /// Column index of each of the first `cached` vector ops in `ops`,
    /// resolved once by name instead of per sample.
    // ponytail: 8 slots; vectors past the 8th fall back to a name lookup
    // per sample. Widen if par() expressions grow.
    columns: [cache_len]u32 = undefined,
    cached: u8 = 0,

    pub const cache_len = 8;
};

/// What a card reads: a plain column or a `par()` waveform.
pub const Waveform = union(enum) {
    column: Column,
    par: Par,

    /// Returns sample `i`; a `par()` waveform that names a missing column
    /// past its cached ones reads NaN there.
    pub fn get(w: Waveform, i: usize) f64 {
        return switch (w) {
            .column => |c| c.get(i),
            .par => |p| fold(p.ops, Sample{ .par = p, .i = i }),
        };
    }
};

const nan = std.math.nan(f64);

/// Sample `i` of the columns a `par()` waveform reads.
const Sample = struct {
    par: Par,
    i: usize,

    pub fn leaf(s: Sample, op: core.MeasureOp, v: usize) f64 {
        return switch (op) {
            .num => |x| x,
            .vector => |name| blk: {
                const p = s.par;
                assert(p.cached <= Par.cache_len);
                const k = if (v < p.cached) p.columns[v] else (columnIndex(p.names, name) orelse break :blk nan);
                var c = p.base;
                c.offset = k * @as(usize, if (c.complex) 2 else 1);
                break :blk c.get(s.i);
            },
            .measure => nan,
            .neg, .add, .sub, .mul, .div, .pow => unreachable,
        };
    }
};

/// Returns the column labelled `name`; a bare node name reads `v(name)`,
/// as ngspice's vector lookup does. Case-insensitive. O(columns).
pub fn columnIndex(names: []const []const u8, name: []const u8) ?usize {
    var buf: [256]u8 = undefined;
    const alt = std.fmt.bufPrint(&buf, "v({s})", .{name}) catch name;
    for (names, 0..) |v, i| {
        if (std.ascii.eqlIgnoreCase(v, name) or std.ascii.eqlIgnoreCase(v, alt)) return i;
    }
    return null;
}

/// Folds measure postfix; `leaves.leaf(op, v)` values each `num`, `measure`
/// and `vector` op, `v` counting the vector ops before it. Returns NaN on a
/// stack overflow or underflow, or a malformed tail.
// comptime interface: the leaves vary (a PARAM card's earlier results, a
// `par()` sample's columns); two cases, and a runtime pointer per op would
// cost a call per sample.
pub fn fold(ops: []const core.MeasureOp, leaves: anytype) f64 {
    var stack: [32]f64 = undefined;
    var n: usize = 0;
    var vectors: usize = 0;
    for (ops) |op| switch (op) {
        .num, .measure, .vector => {
            if (n == stack.len) return nan;
            stack[n] = leaves.leaf(op, vectors);
            vectors += @intFromBool(op == .vector);
            n += 1;
        },
        .neg => {
            if (n == 0) return nan;
            stack[n - 1] = -stack[n - 1];
        },
        .add, .sub, .mul, .div, .pow => {
            if (n < 2) return nan;
            n -= 1;
            const b = stack[n];
            const x = &stack[n - 1];
            x.* = switch (op) {
                .add => x.* + b,
                .sub => x.* - b,
                .mul => x.* * b,
                .div => x.* / b,
                .pow => std.math.pow(f64, x.*, b),
                .num, .measure, .vector, .neg => unreachable,
            };
        },
    };
    assert(n <= stack.len);
    assert(vectors <= ops.len);
    return if (n == 1) stack[0] else nan;
}

// ---- private ----

test "a complex column reads the part its vectype names" {
    // Oracle: 3 + 4j has magnitude 5, 20·log10(5) dB, atan2(4, 3) rad.
    const c: Column = .{ .data = &.{ 0, 0, 3, 4 }, .stride = 4, .offset = 2, .complex = true, .part = .real };
    var x = c;
    try std.testing.expectEqual(3, x.get(0));
    x.part = .fromVectype('i');
    try std.testing.expectEqual(4, x.get(0));
    x.part = .fromVectype('m');
    try std.testing.expectEqual(5, x.get(0));
    x.part = .fromVectype('d');
    try std.testing.expectApproxEqRel(20 * std.math.log10(5.0), x.get(0), 1e-15);
    x.part = .fromVectype('p');
    try std.testing.expectApproxEqRel(0.9272952180016122, x.get(0), 1e-15);
    x.part = .fromVectype('g');
    try std.testing.expectApproxEqRel(53.13010235415598, x.get(0), 1e-14);
    try std.testing.expectEqual(Part.real, Part.fromVectype('r'));
}

test columnIndex {
    const names: []const []const u8 = &.{ "time", "V(Out)", "i(v1)" };
    try std.testing.expectEqual(1, columnIndex(names, "out").?);
    try std.testing.expectEqual(2, columnIndex(names, "I(V1)").?);
    try std.testing.expectEqual(null, columnIndex(names, "v(z)"));
}
