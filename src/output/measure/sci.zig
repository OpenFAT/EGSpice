//! C's `%.*e` for `.meas` listings: an f64 and a precision in, the digits
//! glibc's printf prints out, and a discarding writer that skips them when
//! only a card's value is wanted.
//! Cites: ISO C11 §7.21.6.1 (the `e` conversion); glibc rounds the exact
//! binary value half-to-even.
//!
//! Contents:
//! - `pub const Sci = struct` with `pub fn format(Sci, *Writer) Writer.Error!void`
//! - `pub fn sci(x: f64, comptime precision: u8) Sci`
//! - `pub fn quiet(buffer: []u8) Writer.Discarding`

const std = @import("std");
const assert = std.debug.assert;
const Writer = std.Io.Writer;

/// A number to print as C's `%.*e`, digit for digit: the exact binary value
/// rounded half-to-even, at least two exponent digits and an explicit sign.
/// Infinities and NaN print as Zig's scientific form (`inf`, `nan`).
pub const Sci = struct {
    x: f64,
    precision: u8,

    /// Writes the digits, or nothing into a `quiet` writer.
    pub fn format(s: Sci, w: *Writer) Writer.Error!void {
        assert(s.precision < digits_max);
        if (w.vtable == &quiet_vtable) return;
        if (!std.math.isFinite(s.x)) {
            var buf: [64]u8 = undefined;
            return w.writeAll(std.fmt.float.render(&buf, s.x, .{ .mode = .scientific }) catch "?");
        }
        // The exact decimal of an f64 has at most 767 significant digits.
        var mem: [4096]u8 = undefined;
        var fba: std.heap.FixedBufferAllocator = .init(&mem);
        const exact, const k = exactDecimal(fba.allocator(), @abs(s.x)) catch return w.writeAll("?");
        const q = @as(usize, s.precision) + 1;
        var d: [digits_max]u8 = @splat('0');
        const n = @min(exact.len, q);
        @memcpy(d[0..n], exact[0..n]);
        var e10: i32 = @as(i32, @intCast(exact.len)) - 1 + k;
        if (exact.len > q and roundsUp(exact, q, d[q - 1])) {
            var i = q;
            while (i > 0) : (i -= 1) {
                if (d[i - 1] != '9') {
                    d[i - 1] += 1;
                    break;
                }
                d[i - 1] = '0';
            } else {
                d[0] = '1'; // 9.99..9 carried into 10.00..0
                e10 += 1;
            }
        }
        if (s.x == 0) e10 = 0;
        if (std.math.signbit(s.x)) try w.writeByte('-');
        try w.writeByte(d[0]);
        if (q > 1) try w.print(".{s}", .{d[1..q]});
        try w.print("e{c}{d:0>2}", .{ @as(u8, if (e10 < 0) '-' else '+'), @abs(e10) });
    }
};

/// Mantissa digits `Sci` holds: the precision plus the leading digit.
const digits_max = 32;

const quiet_vtable: Writer.VTable = .{ .drain = Writer.Discarding.drain, .sendFile = Writer.Discarding.sendFile };

/// Returns `x` as `%.<precision>e` for `{f}`.
pub fn sci(x: f64, comptime precision: u8) Sci {
    comptime assert(precision < digits_max);
    return .{ .x = x, .precision = precision };
}

/// Returns a discarding writer for an evaluation whose text nobody reads,
/// only its value: `Sci` prints nothing into it, which skips the exact
/// big-integer decimal (most of a card's cost when `point_measures` or a
/// `Stream` re-evaluates thousands of points). `buffer` must outlive it.
pub fn quiet(buffer: []u8) Writer.Discarding {
    var d: Writer.Discarding = .init(buffer);
    d.writer.vtable = &quiet_vtable;
    return d;
}

// ---- private ----

/// Whether the `q` kept digits of `exact` round up: the next digit above 5,
/// or 5 with anything after it, or 5 alone on an odd last kept digit.
fn roundsUp(exact: []const u8, q: usize, kept_last: u8) bool {
    assert(exact.len > q);
    assert(q > 0);
    const r = exact[q];
    const odd = (kept_last - '0') & 1 == 1;
    const rest = std.mem.trimEnd(u8, exact[q + 1 ..], "0").len != 0;
    return r > '5' or (r == '5' and (rest or odd));
}

/// The exact decimal digits of finite `x >= 0` and their exponent: `x = digits × 10^k`.
/// An f64 is m·2^e exactly, and 2^-e = 5^e / 10^e, so the digits are an integer.
fn exactDecimal(a: std.mem.Allocator, x: f64) !struct { []const u8, i32 } {
    assert(std.math.isFinite(x));
    assert(!std.math.signbit(x));
    const Big = std.math.big.int.Managed;
    const bits: u64 = @bitCast(x);
    const biased: i32 = @intCast(bits >> 52 & 0x7ff);
    const frac = bits & (1 << 52) - 1;
    const m = if (biased == 0) frac else frac | 1 << 52;
    const e = @as(i32, @max(biased, 1)) - 1075;
    var d = try Big.initSet(a, m);
    if (e >= 0) {
        try d.shiftLeft(&d, @intCast(e));
    } else {
        var five = try Big.initSet(a, 5);
        try five.pow(&five, @intCast(-e));
        try d.mul(&d, &five);
    }
    return .{ try d.toString(a, 10, .lower), @min(e, 0) };
}

test "a quiet writer skips the decimal, a fixed one still gets it" {
    var buf: [64]u8 = undefined;
    var q = quiet(&buf);
    try q.writer.print("{f}", .{sci(0.1, 6)});
    try std.testing.expectEqual(@as(u64, 0), q.fullCount());
    var w: Writer = .fixed(&buf);
    try w.print("{f}", .{sci(0.1, 6)});
    try std.testing.expectEqualStrings("1.000000e-01", w.buffered());
}

test "sci matches C %e digit for digit" {
    const libc = struct {
        extern "c" fn snprintf(buf: [*]u8, n: usize, fmt: [*:0]const u8, ...) c_int;
    };
    const cases = [_]f64{
        0,                      -0.0,                       1,        -1,
        1.0000005,              2.5e-7,                     0.125,    9.9999995,
        9.9999996,              std.math.floatTrueMin(f64), 5e-324,   2.2250738585072014e-308,
        std.math.floatMax(f64), 1e300,                      123456.5, 0.1,
        999999.5,               1e-5,                       -3.25e10, 7.0000005e-12,
    };
    var prng: std.Random.DefaultPrng = .init(0x5c1);
    const rand = prng.random();
    var buf: [64]u8 = undefined;
    var want: [64]u8 = undefined;
    for (0..20000 + cases.len) |i| {
        const x: f64 = if (i < cases.len) cases[i] else blk: {
            const v: f64 = @bitCast(rand.int(u64));
            break :blk if (std.math.isFinite(v)) v else 1.5;
        };
        inline for (.{ 5, 6 }) |p| {
            var w: Writer = .fixed(&buf);
            try w.print("{f}", .{sci(x, p)});
            const len: usize = @intCast(libc.snprintf(&want, want.len, "%.*e", @as(c_int, p), x));
            try std.testing.expectEqualStrings(want[0..len], w.buffered());
        }
    }
}

test "sci prints C's %e: signed exponent of two digits or more" {
    const nan = std.math.nan(f64);
    for ([_]f64{ 0, -1.5e-300, 1e100, 12345.678, std.math.inf(f64), nan }, [_][]const u8{ "0.000000e+00", "-1.500000e-300", "1.000000e+100", "1.234568e+04", "inf", "nan" }) |x, want| {
        var buf: [32]u8 = undefined;
        try std.testing.expectEqualStrings(want, try std.fmt.bufPrint(&buf, "{f}", .{sci(x, 6)}));
    }
}
