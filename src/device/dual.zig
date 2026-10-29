//! Forward-mode AD scalars: a device's unknowns in, a residual value and
//! its Jacobian row out, one `evalRange` pass for both. `DualFor` carries
//! only the lanes an expression reads; `RealFor` is the value-only twin
//! every non-eval entry point runs on, bit-identical in the value half.
//! Cites VerA's scalar family contract (`contract.family_fns`,
//! `contract.expectFamily`) and LRM §4.3.1 for `pow`'s negative bases.
//!
//! Contents:
//! - `pub const Real`
//! - `pub fn Dual(comptime N: usize) type`
//! - `pub fn DualFor(comptime lane: []const u8, comptime pad: bool, comptime collapsed: bool) type`
//! - `pub fn RealFor(comptime collapsed: bool) type`
//! - per family: `collapse_applied`, `V`, `lanes`, `Of`, `con`, `probe`, `sel`,
//!   `to`, `val`, `ddxAt`, `grad`, `add`, `sub`, `neg`, `mul`, `div`,
//!   `scale`, `addC`, `exp`, `log`, `expm1`, `log1p`, `sqrt`, `sin`, `cos`,
//!   `tanh`, `pow`, `atan`, `sinh`, `cosh`, `lt`, `le`, `eq`

const std = @import("std");
const builtin = @import("builtin");
const contract = @import("contract");
const gompute = @import("gompute");
const layout = @import("layout.zig");
const Mask = layout.Mask;
// NVPTX/AMDGCN have no libm, so `@exp`/`@log`/`@sin` and std's pow/sinh/cosh
// do not compile there; gompute.math replaces them. exp, exp2, log, log2,
// log10 and pow run one body on host and device (no libm forwarding);
// tanh, sinh, cosh and atan still call std.math on the host. `expm1` comes
// from it too: std's uses `doNotOptimizeAway`, which AMDGCN cannot lower.
const dmath = gompute.math;

/// The value-only family every non-eval entry point runs on.
pub const Real = RealFor(false);

/// Whether `@mulAdd` is a native instruction. Without the feature it lowers to
/// a libm `fma()` call per lane, far slower than the two-rounding form.
/// It is `@mulAdd` rather than `@setFloatMode(.optimized)` on purpose: that
/// mode also grants `nnan`/`ninf`, and `Dual.div` must be allowed to divide by
/// zero (Verilog-A permits `V/r` with r = 0) so SparseLu's isFinite check can
/// catch the result.
// ponytail: an arch switch covers every shipped target; add a `-Dfma` option
// only when a target needs to override it.
const fma_ok = switch (builtin.target.cpu.arch) {
    .x86_64 => std.Target.x86.featureSetHas(builtin.target.cpu.features, .fma),
    .aarch64, .aarch64_be => true,
    .nvptx64, .amdgcn => true,
    else => false, // else: no shipped target beyond these has a fused multiply-add guaranteed
};

/// A `Dual` family over lanes `0..n` with unknown u on lane u, every value
/// carrying all n lanes. For tests and one-off evaluations; `evalRange`
/// builds its families from a device's `Basis`.
pub fn Dual(comptime N: usize) type {
    const lane = comptime blk: {
        var l: [N]u8 = undefined;
        for (&l, 0..) |*e, u| e.* = u;
        break :blk l;
    };
    return DualFor(&lane, false, false);
}

/// Forward-mode AD family: the contract's scalar family `S`
/// (`contract.family_fns`, numerics per `contract.expectFamily`). One eval
/// pass yields the residual and every partial.
///
/// `Of(m)` carries only the lanes of `m`'s unknowns, and a binary operation
/// joins its operands' lanes. `m` is a `contract.MaskOf(lane.len)`.
///
/// `lane[u]` is the derivative lane unknown `u` seeds, or
/// `contract.no_lane`; several unknowns may share one (the narrow basis).
/// `pad` rounds each value's lane count up to a power of two: LLVM splits a
/// 12- or 26-wide vector into shuffles and spills (gummel_poon's eval body
/// doubled). The pad lanes hold zeros (or a NaN from a 0/0) that nothing
/// reads.
pub fn DualFor(comptime lane: []const u8, comptime pad: bool, comptime collapsed: bool) type {
    const n_lanes = laneCount(lane);
    // A lane set, one bit per lane.
    const L = Mask(n_lanes);
    // A mask over the unknowns.
    const M = Mask(lane.len);
    return struct {
        /// Read by the generated device: the builder already applied
        /// `D.collapse` to the scatter tapes.
        pub const collapse_applied = collapsed;
        /// The value type (contract `RefFamily`).
        pub const V = f64;

        /// The lanes the unknowns in `m` occupy, one bit per lane.
        fn lanesOf(comptime m: M) L {
            @setEvalBranchQuota(10 * lane.len + 1000);
            var s: L = 0;
            for (lane, 0..) |l, u| {
                if (layout.has(m, u) and l != contract.no_lane) s |= @as(L, 1) << l;
            }
            return s;
        }

        fn width(comptime ls: L) usize {
            const n = @popCount(ls);
            return if (pad and n != 0) std.math.ceilPowerOfTwoAssert(usize, n) else n;
        }

        fn unknownBit(comptime u: usize) M {
            return @as(M, 1) << u;
        }

        /// The type of a value that depends on the unknowns in `m`.
        pub fn Of(comptime m: M) type {
            return Val(lanesOf(m));
        }
        /// A constant: every lane +0.
        pub fn con(c: f64) Of(0) {
            return .{ .v = c, .d = @splat(0) };
        }
        /// Unknown `u` at `v`, its lane seeded with 1. An unknown without a
        /// lane (`contract.no_lane`) comes back as a constant.
        pub fn probe(comptime u: usize, v: f64) Of(unknownBit(u)) {
            const T = Of(unknownBit(u));
            var d: T.Lanes = @splat(0);
            if (comptime lane[u] != contract.no_lane) d[comptime T.pos(lane[u])] = 1;
            return .{ .v = v, .d = d };
        }
        /// `a` where the indicator `c` is nonzero (NaN included), else `b`,
        /// lanes included.
        pub fn sel(c: anytype, a: anytype, b: anytype) Val(@TypeOf(a).lanes | @TypeOf(b).lanes) {
            const r = @TypeOf(a).lanes | @TypeOf(b).lanes;
            // `c.v != 0.0` from the bits: Zig 0.17's self-hosted x86_64
            // backend reads `!=`'s parity-or-nonzero flags a second time
            // after clobbering them, so `pick` was true at c == 0 (#25,
            // dev/zig-backend.md). Same answer for every c, NaN included.
            const pick = @as(u64, @bitCast(c.v)) << 1 != 0;
            // A lane select, not a branch: on the GPU each thread's indicator
            // differs, and selection is exact, so the bits are the branch's.
            const d = if (comptime width(r) == 0) a.spread(r) else @select(f64, @as(@Vector(width(r), bool), @splat(pick)), a.spread(r), b.spread(r));
            return .{ .v = if (pick) a.v else b.v, .d = d };
        }

        /// A value carrying the lanes in `ls`, in ascending lane order.
        fn Val(comptime ls: L) type {
            return struct {
                v: f64,
                /// Aligned to the element, not the vector: natural vector
                /// alignment pads 8 f64 lanes from 72 to 128 bytes, and
                /// nothing reads `d` through a pointer that needs it.
                d: Lanes align(@alignOf(f64)),

                /// The lanes this value carries, one bit per lane.
                pub const lanes = ls;
                const Lanes = @Vector(width(ls), f64);
                const T = @This();

                comptime {
                    // The value, then the lanes with no gap.
                    assert(@sizeOf(T) == std.mem.alignForward(usize, @sizeOf(f64) + @sizeOf(Lanes), @alignOf(f64)));
                }

                /// Where lane `l` sits in `d`.
                fn pos(comptime l: usize) usize {
                    return @popCount(ls & ((@as(L, 1) << l) - 1));
                }
                inline fn k(c: f64) Lanes {
                    return @splat(c);
                }
                inline fn map(a: T, v: f64, c: f64) T {
                    return .{ .v = v, .d = a.d * k(c) };
                }
                /// These lanes laid out as `to_ls`'s. New and pad lanes are +0.
                inline fn spread(a: T, comptime to_ls: L) @Vector(width(to_ls), f64) {
                    if (ls & ~to_ls != 0) @compileError("Dual: lanes do not widen");
                    if (ls == to_ls) return a.d;
                    if (ls == 0) return @splat(0);
                    const idx = comptime blk: {
                        var idx: [width(to_ls)]i32 = @splat(-1);
                        var j: usize = 0;
                        for (0..n_lanes) |l| if (layout.has(to_ls, l)) {
                            if (layout.has(ls, l)) idx[j] = pos(l);
                            j += 1;
                        };
                        break :blk idx;
                    };
                    return @shuffle(f64, a.d, @as(@Vector(1, f64), @splat(0)), idx);
                }
                fn toLanes(a: T, comptime to_ls: L) Val(to_ls) {
                    return .{ .v = a.v, .d = a.spread(to_ls) };
                }

                /// Widens to `Of(m)`, new lanes +0. A compile error when `m`
                /// drops a lane this value carries.
                pub fn to(a: T, comptime m: M) Of(m) {
                    return a.toLanes(Of(m).lanes);
                }
                pub fn val(a: T) f64 {
                    return a.v;
                }
                /// The partial with respect to unknown `u`.
                pub fn ddxAt(a: T, comptime u: usize) f64 {
                    const l = lane[u];
                    if (comptime l == contract.no_lane or !layout.has(ls, l)) return 0.0;
                    return a.d[comptime pos(l)];
                }
                /// Every lane, pad lanes included.
                pub inline fn grad(a: T) @Vector(width(ls), f64) {
                    return a.d;
                }

                pub fn add(a: T, b: anytype) Val(ls | @TypeOf(b).lanes) {
                    const r = ls | @TypeOf(b).lanes;
                    return .{ .v = a.v + b.v, .d = a.spread(r) + b.spread(r) };
                }
                pub fn sub(a: T, b: anytype) Val(ls | @TypeOf(b).lanes) {
                    const r = ls | @TypeOf(b).lanes;
                    return .{ .v = a.v - b.v, .d = a.spread(r) - b.spread(r) };
                }
                pub fn neg(a: T) T {
                    return .{ .v = -a.v, .d = -a.d };
                }
                /// Lanes by the product rule, fused into one rounding where
                /// `fma_ok`, so they differ in the last bit between targets
                /// with and without FMA; the value never does.
                pub fn mul(a: T, b: anytype) Val(ls | @TypeOf(b).lanes) {
                    const B = @TypeOf(b);
                    const J = Val(ls | B.lanes);
                    if (B.lanes == 0) return .{ .v = a.v * b.v, .d = a.spread(J.lanes) * J.k(b.v) };
                    if (ls == 0) return .{ .v = a.v * b.v, .d = b.spread(J.lanes) * J.k(a.v) };
                    return .{ .v = a.v * b.v, .d = mulAddV(J.Lanes, b.spread(J.lanes), J.k(a.v), a.spread(J.lanes) * J.k(b.v)) };
                }
                /// The value is the IEEE quotient (contract §3); the lanes
                /// keep the reciprocal.
                pub fn div(a: T, b: anytype) Val(ls | @TypeOf(b).lanes) {
                    const B = @TypeOf(b);
                    const J = Val(ls | B.lanes);
                    const inv = 1.0 / b.v;
                    const q = a.v / b.v;
                    if (B.lanes == 0) return .{ .v = q, .d = a.spread(J.lanes) * J.k(inv) };
                    return .{ .v = q, .d = mulAddV(J.Lanes, b.spread(J.lanes), J.k(-q), a.spread(J.lanes)) * J.k(inv) };
                }
                pub fn scale(a: T, c: f64) T {
                    return map(a, a.v * c, c);
                }
                pub fn addC(a: T, c: f64) T {
                    return .{ .v = a.v + c, .d = a.d };
                }
                pub fn exp(a: T) T {
                    const e = dmath.exp(a.v);
                    return map(a, e, e);
                }
                pub fn log(a: T) T {
                    return map(a, dmath.log(a.v), 1.0 / a.v);
                }
                /// Same value operation as VerA's precompute scalar, so the
                /// two agree.
                pub fn expm1(a: T) T {
                    return map(a, dmath.expm1(a.v), dmath.exp(a.v));
                }
                /// gompute's port: Zig 0.17's `std.math.log1p` emits an asm
                /// operand AMDGCN cannot select. Bit-identical to std.
                pub fn log1p(a: T) T {
                    return map(a, dmath.log1p(a.v), 1.0 / (1.0 + a.v));
                }
                /// Slope `0.5/sqrt(x)`, and 0 rather than inf or NaN where
                /// `sqrt(x)` is not positive.
                pub fn sqrt(a: T) T {
                    const s = @sqrt(a.v);
                    return map(a, s, if (s > 0.0) 0.5 / s else 0.0);
                }
                pub fn sin(a: T) T {
                    return map(a, dmath.sin(a.v), dmath.cos(a.v));
                }
                pub fn cos(a: T) T {
                    return map(a, dmath.cos(a.v), -dmath.sin(a.v));
                }
                pub fn tanh(a: T) T {
                    const th = dmath.tanh(a.v);
                    return map(a, th, 1.0 - th * th);
                }
                /// `x^c`, slope `c·x^c/x`: one `pow`, exact for x != 0 (LRM
                /// §4.3.1 negative bases included). At x == 0 the slope takes
                /// a second `pow` because `c·p/x` is 0/0 there and c == 1
                /// must still give slope 1. A non-finite slope becomes 0.
                pub fn pow(a: T, c: f64) T {
                    const p = dmath.pow(a.v, c);
                    const slope = if (a.v != 0.0) c * p / a.v else c * dmath.pow(a.v, c - 1.0);
                    return map(a, p, if (std.math.isFinite(slope)) slope else 0.0);
                }
                pub fn atan(a: T) T {
                    return map(a, dmath.atan(a.v), 1.0 / (1.0 + a.v * a.v));
                }
                pub fn sinh(a: T) T {
                    return map(a, dmath.sinh(a.v), dmath.cosh(a.v));
                }
                pub fn cosh(a: T) T {
                    return map(a, dmath.cosh(a.v), dmath.sinh(a.v));
                }
                // Comparisons return a 0/1 indicator without lanes; `sel`
                // carries the lanes of the operand it picks.

                /// 1 when `a < b`, else 0 (NaN on either side gives 0).
                pub fn lt(a: T, b: anytype) Of(0) {
                    return con(@floatFromInt(@intFromBool(a.v < b.v)));
                }
                /// 1 when `a <= b`, else 0 (NaN on either side gives 0).
                pub fn le(a: T, b: anytype) Of(0) {
                    return con(@floatFromInt(@intFromBool(a.v <= b.v)));
                }
                /// 1 when `a == b`, else 0 (NaN on either side gives 0).
                pub fn eq(a: T, b: anytype) Of(0) {
                    return con(@floatFromInt(@intFromBool(a.v == b.v)));
                }
            };
        }
    };
}

/// Value-only family for the paths that read no partial: `evalQRange`, the
/// charge-tape kernel and every value entry point (`setup`, `limit`,
/// `updateState`, ...). Every method is the `.v` line of the matching
/// `DualFor` method verbatim, so it computes the bits `eval` computes
/// without paying for the gradient.
pub fn RealFor(comptime collapsed: bool) type {
    return struct {
        v: f64,

        const Self = @This();
        pub const collapse_applied = collapsed;
        pub const V = f64;
        pub const lanes: u64 = 0;

        pub fn Of(comptime _: anytype) type {
            return Self;
        }
        pub fn con(c: f64) Self {
            return .{ .v = c };
        }
        pub fn probe(comptime _: usize, v: f64) Self {
            return .{ .v = v };
        }
        pub fn sel(c: Self, a: Self, b: Self) Self {
            // The bit test `DualFor.sel` uses, not `c.v != 0.0` (#25).
            return if (@as(u64, @bitCast(c.v)) << 1 != 0) a else b;
        }
        pub fn to(a: Self, comptime _: anytype) Self {
            return a;
        }
        pub fn val(a: Self) f64 {
            return a.v;
        }
        /// Always 0; the contract requires the accessor.
        pub fn ddxAt(_: Self, comptime _: usize) f64 {
            return 0.0;
        }
        pub fn add(a: Self, b: Self) Self {
            return .{ .v = a.v + b.v };
        }
        pub fn sub(a: Self, b: Self) Self {
            return .{ .v = a.v - b.v };
        }
        pub fn neg(a: Self) Self {
            return .{ .v = -a.v };
        }
        pub fn mul(a: Self, b: Self) Self {
            return .{ .v = a.v * b.v };
        }
        pub fn div(a: Self, b: Self) Self {
            return .{ .v = a.v / b.v };
        }
        pub fn scale(a: Self, c: f64) Self {
            return .{ .v = a.v * c };
        }
        pub fn addC(a: Self, c: f64) Self {
            return .{ .v = a.v + c };
        }
        pub fn exp(a: Self) Self {
            return .{ .v = dmath.exp(a.v) };
        }
        pub fn log(a: Self) Self {
            return .{ .v = dmath.log(a.v) };
        }
        pub fn expm1(a: Self) Self {
            return .{ .v = dmath.expm1(a.v) };
        }
        pub fn log1p(a: Self) Self {
            return .{ .v = dmath.log1p(a.v) };
        }
        pub fn sqrt(a: Self) Self {
            return .{ .v = @sqrt(a.v) };
        }
        pub fn sin(a: Self) Self {
            return .{ .v = dmath.sin(a.v) };
        }
        pub fn cos(a: Self) Self {
            return .{ .v = dmath.cos(a.v) };
        }
        pub fn tanh(a: Self) Self {
            return .{ .v = dmath.tanh(a.v) };
        }
        pub fn sinh(a: Self) Self {
            return .{ .v = dmath.sinh(a.v) };
        }
        pub fn cosh(a: Self) Self {
            return .{ .v = dmath.cosh(a.v) };
        }
        pub fn atan(a: Self) Self {
            return .{ .v = dmath.atan(a.v) };
        }
        pub fn pow(a: Self, c: f64) Self {
            return .{ .v = dmath.pow(a.v, c) };
        }
        pub fn lt(a: Self, b: Self) Self {
            return con(@floatFromInt(@intFromBool(a.v < b.v)));
        }
        pub fn le(a: Self, b: Self) Self {
            return con(@floatFromInt(@intFromBool(a.v <= b.v)));
        }
        pub fn eq(a: Self, b: Self) Self {
            return con(@floatFromInt(@intFromBool(a.v == b.v)));
        }
    };
}

// ---- private ----

/// Number of lanes `lane` maps unknowns onto.
fn laneCount(comptime lane: []const u8) usize {
    var n: usize = 0;
    for (lane) |l| if (l != contract.no_lane) {
        n = @max(n, @as(usize, l) + 1);
    };
    return n;
}

inline fn mulAddV(comptime Vec: type, a: Vec, b: Vec, c: Vec) Vec {
    return if (fma_ok) @mulAdd(Vec, a, b, c) else a * b + c;
}
