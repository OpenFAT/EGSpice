//! How a transient measures each `.tran` card as it runs: a deck's cards in,
//! a `Rule` per card out, and from those the columns a measures-only run
//! must still record.
//! Cites: dev/research/lane-batched-simulation.md §7 (measuring as the run
//! goes).
//!
//! Contents:
//! - `pub const Rule = enum { end, param, scan, replay }`
//! - `pub fn ruleOf(measures: []const core.Measure, i: usize) Rule`
//! - `pub fn markColumns(measures: []const core.Measure, names: []const []const u8, keep: []bool, streamed: bool) void`

const std = @import("std");
const assert = std.debug.assert;
const core = @import("core");
const columns = @import("column.zig");
const card = @import("card.zig");
const Clause = core.MeasureClause;

/// How `Stream` measures a card: from the first row without a recording
/// (`scan`), over the recording (`replay`), once the cards it reads resolve
/// (`param`), or never (`end`).
pub const Rule = enum { end, param, scan, replay };

/// Returns how `Stream` measures card `i` of `measures`. Asserts that `i`
/// names a card.
pub fn ruleOf(measures: []const core.Measure, i: usize) Rule {
    assert(i < measures.len);
    const m = measures[i];
    if (m.cont) return .end;
    // `print` reads a later card, or a PARAM, in its second pass and
    // `evaluateAll` reads NaN; a PARAM reading a later PARAM reads NaN.
    var refs = false;
    var at_ref = false;
    for ([_]Clause{ m.first, m.second }) |c| for (c.refs) |ref| {
        refs = true;
        at_ref = at_ref or ref.field == .at;
        for (ref.expr) |op| if (op == .measure) {
            if (op.measure >= i or (op.measure < measures.len and measures[op.measure].func == .param)) return .end;
        };
    };
    for (m.expr) |op| if (op == .measure and op.measure >= i and op.measure < measures.len and measures[op.measure].func == .param) return .end;
    const recorded = refs or switch (m.func) {
        // The targ count starts at the trigger.
        .trig_targ => m.second.td_trig and m.first.at == core.measure_no_at,
        // The value at the event reads rows before it, as far back as a
        // window start or the last event.
        .find, .deriv => m.first.at == core.measure_no_at and (m.second.td != 0 or m.second.from != 0 or card.lastEvent(m.second)),
        else => false, // else: every other function reads each row once
    };
    const by_rows: Rule = if (recorded) .replay else .scan;
    return switch (m.func) {
        .param => .param,
        .when, .trig_targ, .avg, .min, .max, .min_at, .max_at, .pp, .rms, .integ, .em_avg, .err, .err1, .err2, .err3 => by_rows,
        // A bare FIND reads a one-row plot.
        .find, .deriv => if (m.first.at != core.measure_no_at or at_ref or m.second.vec.len != 0) by_rows else .end,
        else => .end, // else: JITTER, the FFT figures, the checks and DOUT read the whole run
    };
}

/// Sets `keep[k]` for every column of `names` (a transient's labels, the
/// scale first) that a `.tran` card of `measures` reads, and the scale;
/// with `streamed`, only for the cards a `Stream` cannot measure without
/// the recording. Evaluating over just those columns gives the same values,
/// to the bit, as over all of them: lookups are by name. Asserts that
/// `keep.len == names.len`.
pub fn markColumns(measures: []const core.Measure, names: []const []const u8, keep: []bool, streamed: bool) void {
    assert(keep.len == names.len);
    @memset(keep, false);
    if (keep.len > 0) keep[0] = true;
    for (measures, 0..) |m, i| if (m.analysis == .tran and !(streamed and ruleOf(measures, i) == .scan)) for ([_]Clause{ m.first, m.second }) |c| {
        for ([_][]const u8{ c.vec, c.vec2 }) |name| if (name.len > 0) if (columns.columnIndex(names, name)) |k| {
            keep[k] = true;
        };
        for ([_][]const core.MeasureOp{ c.ops, c.ops2 }) |ops| for (ops) |op| if (op == .vector) if (columns.columnIndex(names, op.vector)) |k| {
            keep[k] = true;
        };
    };
}

test "a card reading a later card or a PARAM never resolves; WHEN scans; TD=TRIG replays" {
    // Oracle: the rules ruleOf's doc comment states, case by case.
    const later: []const core.MeasureOp = &.{.{ .measure = 1 }};
    const cards = [_]core.Measure{
        .{ .analysis = .tran, .name = "w", .func = .when, .first = .{ .vec = "v(a)", .refs = &.{.{ .field = .td, .expr = later }} } },
        .{ .analysis = .tran, .name = "x", .func = .when, .first = .{ .vec = "v(a)" } },
        .{ .analysis = .tran, .name = "t", .func = .trig_targ, .first = .{ .vec = "v(a)" }, .second = .{ .vec = "v(b)", .td_trig = true } },
        .{ .analysis = .tran, .name = "p", .func = .param, .first = .{} },
        .{ .analysis = .tran, .name = "j", .func = .jitter, .first = .{ .vec = "v(a)" } },
        .{ .analysis = .tran, .name = "f", .func = .find, .first = .{ .vec = "v(a)" } },
    };
    const want = [_]Rule{ .end, .scan, .replay, .param, .end, .end };
    for (want, 0..) |r, i| try std.testing.expectEqual(r, ruleOf(&cards, i));
    var keep: [3]bool = undefined;
    markColumns(&cards, &.{ "time", "v(a)", "v(b)" }, &keep, true);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true }, &keep);
    markColumns(cards[1..2], &.{ "time", "v(a)", "v(b)" }, &keep, true);
    try std.testing.expectEqualSlices(bool, &.{ true, false, false }, &keep);
}
