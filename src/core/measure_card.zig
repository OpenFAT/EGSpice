//! A parsed `.meas` card: what it computes over which result, its clauses
//! with ngspice's defaults, and the optimizer error its GOAL defines. Plain
//! data, filled by the frontend and evaluated by the output layer; the one
//! transform here is `Measure.goalError`, result value in, weighted miss out.
//! Cites: ngspice com_measure2.c (`MeasureClause` defaults, :1389 for VAL);
//! HSPICE [CR .MEASURE, .MEASURE (Error Function), .MEASURE FFT; SA Ch.27
//! "Optimization Statements"].
//!
//! Contents:
//! - `pub const MeasureFunc = enum(u8)`
//! - `pub const Check = struct`, `Check.Edge = enum(u8) { rise, fall, both }`
//! - `pub const MeasureOp = union(enum)`
//! - `pub const MeasureRef = struct`, `MeasureRef.Field = enum(u8)`
//! - `pub const measure_unset: i32`, `pub const measure_last: i32`, `pub const measure_no_at: f64`
//! - `pub const MeasureClause = struct`
//! - `pub const GoalBound = enum(u8) { equal, below, above }`
//! - `pub const Measure = struct`, `Measure.goalError(m: Measure, value: f64) ?f64`

const std = @import("std");
const assert = std.debug.assert;
const Kind = @import("query.zig").Kind;

/// What a `.meas` card computes, after ngspice com_measure2.c, plus the
/// HSPICE forms: `param` (an expression over other results) and the `err`
/// relative-error family [CR .MEASURE (Error Function)], and the FFT
/// figures THD, SNR, SNDR, ENOB and SFDR [CR .MEASURE FFT], and the
/// recovered electromigration average `em_avg`, and `.jitter`'s time
/// interval error. The `check_*` and `dout` forms count the violations of
/// an HSPICE `.check`, `.biaschk` or `.dout` card (see `Check`).
pub const MeasureFunc = enum(u8) { trig_targ, find, when, avg, min, max, min_at, max_at, pp, rms, integ, deriv, param, err, err1, err2, err3, thd, snr, sndr, enob, sfdr, em_avg, jitter, check_slew, check_forbid, check_require, check_level, dout };

/// What a violation-counting measure tests on the `first` clause's
/// waveform, and on the `second` clause's reference where it has one.
/// Transitions are read between the logic thresholds: a rise leaves `lo`
/// and reaches `hi` without falling back, and it happens at the midpoint of
/// those two crossings.
pub const Check = struct {
    lo: f64 = 0,
    hi: f64 = 0,
    /// `check_slew`: the allowed transition time. `check_forbid`: the window
    /// around each reference edge no transition may land in (SETUP is
    /// [-duration, 0], HOLD [0, duration]). `check_require`: the window one
    /// must land in (EDGE). `check_level`: the allowed band.
    min: f64 = -std.math.inf(f64),
    max: f64 = std.math.inf(f64),
    /// `check_level`: the longest excursion out of the band that passes; 0
    /// counts every one.
    dur: f64 = 0,
    edge: Edge = .both,
    ref_edge: Edge = .both,
    /// `dout`: (time, state) pairs, state 0 or 1; don't-care states are
    /// left out.
    expect: []const [2]f64 = &.{},

    /// Which transitions count; `both` takes either.
    pub const Edge = enum(u8) { rise, fall, both };
};

/// One postfix op of a `PARAM=` measure or a `par('expr')` waveform: a
/// constant, the result of the `measure`-th card, a result vector by label
/// (`v(out)`, `i(v1)`; waveforms only), or an arithmetic operator.
pub const MeasureOp = union(enum) { num: f64, measure: u32, vector: []const u8, neg, add, sub, mul, div, pow };

/// A clause value HSPICE lets name earlier `.meas` results (`FROM=t10`,
/// `TD='tr+1n'`): `expr` is evaluated over them before the clause runs and
/// overrides `field`.
pub const MeasureRef = struct {
    field: Field,
    expr: []const MeasureOp,

    /// The `MeasureClause` field the evaluated expression replaces.
    pub const Field = enum(u8) { val, td, from, to, at };
};

/// Event count not given (ngspice MEASURE_DEFAULT).
pub const measure_unset: i32 = -1;
/// `RISE=LAST` and friends (ngspice MEASURE_LAST_TRANSITION).
pub const measure_last: i32 = -2;
/// `AT` not given.
pub const measure_no_at: f64 = 1e99;

/// One clause of a `.meas` card: ngspice's `struct measure`, with its
/// defaults. For `.meas dc`, `from`/`to` default to -1e99/1e99.
pub const MeasureClause = struct {
    /// Result variable name as the result labels it (`v(out)`).
    vec: []const u8 = "",
    /// Right side of `WHEN vec=vec2`; empty when the level is `val`.
    vec2: []const u8 = "",
    /// `par('expr')` waveforms standing in for `vec` and `vec2`, sample by
    /// sample; empty for a plain vector.
    ops: []const MeasureOp = &.{},
    ops2: []const MeasureOp = &.{},
    /// Values read from earlier results, applied before the clause runs.
    refs: []const MeasureRef = &.{},
    /// HSPICE `TD=TRIG` on a TARG: count events only after the trigger.
    td_trig: bool = false,
    /// How an AC value is read: `m`, `p` (radians), `r`, `i`, `d` from
    /// `vm(..)` and the like, `g` (degrees) from `lstb(p)`, 0 for the real
    /// part.
    vectype: u8 = 0,
    /// The level to cross; ngspice's unset m_val is 1e99 (com_measure2.c:1389),
    /// so a TRIG or TARG without VAL never fires.
    val: f64 = 1e99,
    rise: i32 = measure_unset,
    fall: i32 = measure_unset,
    cross: i32 = measure_unset,
    td: f64 = 0,
    from: f64 = 0,
    to: f64 = 0,
    at: f64 = measure_no_at,
    /// ERR family: the smallest |meas_var| used as a denominator, and the
    /// |meas_var| band outside which a point is skipped (HSPICE defaults).
    minval: f64 = 1e-12,
    ymin: f64 = 1e-15,
    ymax: f64 = 1e15,
    /// FFT figures: the highest harmonic counted as distortion (0: every
    /// one in the spectrum), and the bins either side of the fundamental
    /// counted as signal. MINFREQ and MAXFREQ are `from` and `to`.
    /// EM_AVG keeps `.option em_recovery` in `val`.
    nbharm: u32 = 0,
    binsiz: u32 = 0,
    /// HSPICE optimization target (`GOAL=`) and the weight of its error.
    goal: ?f64 = null,
    weight: f64 = 1,
    /// `GOAL < v` and `GOAL > v` [SA Ch.27 "Optimization Statements"]:
    /// the result only has to stay on that side of the goal.
    goal_bound: GoalBound = .equal,
};

/// How a `.meas` result is held to its GOAL: equal, or only below or
/// above it (an inequality constraint, no error while it holds).
pub const GoalBound = enum(u8) { equal, below, above };

/// A parsed `.meas` card. Its slices borrow from the arena the deck was
/// parsed into.
pub const Measure = struct {
    /// `tran`, `ac`, `dc`, `fft` or `tran_noise`: the results it is
    /// evaluated over.
    analysis: Kind,
    name: []const u8,
    func: MeasureFunc,
    /// TRIG, FIND's vector, WHEN's condition, or the vector a window
    /// function (AVG, MIN, RMS, ...) reads.
    first: MeasureClause,
    /// TARG, or FIND's WHEN clause.
    second: MeasureClause = .{},
    /// The expression of a `param` card.
    expr: []const MeasureOp = &.{},
    /// HSPICE `TRAN_CONT`/`AC_CONT`/`DC_CONT`: every event from the given
    /// count on, one result each.
    cont: bool = false,
    /// The `check_*` and `dout` forms' test.
    check: Check = .{},

    /// The card's optimization error for result `value`, HSPICE's
    /// WEIGHT * (result - GOAL) / max(|GOAL|, MINVAL), from the first
    /// clause that carries `GOAL=`; 0 while an inequality goal holds; null
    /// for a card without a goal.
    pub fn goalError(m: Measure, value: f64) ?f64 {
        const c, const goal = if (m.first.goal) |g|
            .{ m.first, g }
        else if (m.second.goal) |g|
            .{ m.second, g }
        else
            return null;
        const holds = switch (c.goal_bound) {
            .equal => false,
            .below => value < goal,
            .above => value > goal,
        };
        if (holds) return 0;
        const denominator = @max(@abs(goal), c.minval);
        assert(!(denominator < 0));
        return c.weight * (value - goal) / denominator;
    }
};

test "goalError weighs the miss against the goal, floored at MINVAL, and is zero while a bound holds" {
    const t = std.testing;
    var m: Measure = .{ .analysis = .tran, .name = "m", .func = .find, .first = .{ .goal = 2, .weight = 3 } };
    try t.expectEqual(@as(?f64, 3 * (2.5 - 2) / 2.0), m.goalError(2.5));
    m.first.goal = 0;
    try t.expectApproxEqRel(3e-3 / 1e-12, m.goalError(1e-3).?, 1e-15);
    m.first.goal = 2;
    m.first.goal_bound = .below;
    try t.expectEqual(@as(?f64, 0), m.goalError(1));
    try t.expectEqual(@as(?f64, 3.0 * (3 - 2) / 2.0), m.goalError(3));
    m.first.goal_bound = .above;
    try t.expectEqual(@as(?f64, 0), m.goalError(3));
    // The goal may ride on the second clause; neither clause is no goal.
    m.first = .{};
    try t.expectEqual(@as(?f64, null), m.goalError(1));
    m.second = .{ .goal = -4 };
    try t.expectEqual(@as(?f64, (1.0 + 4) / 4.0), m.goalError(1));
}

test "an unset measure level is ngspice's 1e99" {
    // com_measure2.c:1389 m_val = 1e99: a TRIG without VAL never fires (#28).
    const c: MeasureClause = .{};
    try std.testing.expectEqual(@as(f64, 1e99), c.val);
}
