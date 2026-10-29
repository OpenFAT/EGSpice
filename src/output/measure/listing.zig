//! `.meas` listings: a deck's cards and one result (or many Monte Carlo
//! trials) in, ngspice's `Measurements for ... Analysis` block, or the
//! trial statistics, out. A card that names a missing vector or whose event
//! never happens writes ngspice's error line to the error writer instead.
//! Cites: ngspice 45 com_measure2.c and measure.c (headings, error lines,
//! PARAM cards evaluated after the others).
//!
//! Contents:
//! - `pub const Point = struct { row: u32, label: []const u8 }`
//! - `pub fn print(out: *Writer, err: *Writer, measures: []const core.Measure, analysis: Kind, result: core.Result, point: ?Point) Writer.Error!void`
//! - `pub fn printFailed(out: *Writer, measures: []const core.Measure, analysis: Kind, point: Point) Writer.Error!void`
//! - `pub fn printStatistics(out: *Writer, measures: []const core.Measure, analysis: Kind, trials: []const core.Result) Writer.Error!void`

const std = @import("std");
const assert = std.debug.assert;
const Writer = std.Io.Writer;
const core = @import("core");
const card = @import("card.zig");
const Wave = @import("Wave.zig");
const sci = @import("sci.zig").sci;
const quiet = @import("sci.zig").quiet;
const Clause = core.MeasureClause;
const Kind = core.query.Kind;

/// A sweep point's row in the variant table and its label (`rv=1k, ...`),
/// which names its `.meas` block.
pub const Point = struct { row: u32, label: []const u8 };

/// Cards `print` keeps results for, and `printStatistics` reports; a PARAM
/// card reads NaN for any card past it.
const cards_max = 256;
/// Bytes of listing text `print`'s first pass keeps; a card past them is
/// evaluated again in the second.
const kept_bytes_max = 16 * 1024;

/// What `print`'s first pass kept of a card: nothing (`fresh`, evaluate it
/// again), or its text and how it ended.
const Kept = enum(u8) { fresh, value, no_such_vector, out_of_interval };

const nan = std.math.nan(f64);

/// v(a) = t and v(b) = 4 - t on t = 0..4, sampled every 1 (test fixture).
const ramp: core.Result = .{ .plotname = "t", .varnames = &.{ "time", "v(a)", "v(b)" }, .is_complex = false, .npoints = 5, .data = &.{ 0, 0, 4, 1, 1, 3, 2, 2, 2, 3, 3, 1, 4, 4, 0 } };

comptime {
    assert(kept_bytes_max <= std.math.maxInt(u16)); // `kept_len` holds any kept text.
}

/// Writes every card in `measures` that targets `analysis`, evaluated over
/// `result`, under ngspice's heading. Writes nothing when none targets it.
/// A card that names a missing vector or whose event never happens writes
/// ngspice's error line to `err` and no value; at a sweep `point`, which
/// the heading names, it also writes `name = NaN`. A PARAM card reads NaN
/// for any card past the 256th.
pub fn print(out: *Writer, err: *Writer, measures: []const core.Measure, analysis: Kind, result: core.Result, point: ?Point) Writer.Error!void {
    // As ngspice's measure.c, PARAM cards read every other card's result,
    // so those are evaluated first; each card still prints in deck order.
    var values: [cards_max]f64 = @splat(nan);
    var done: Wave.Done = undefined;
    // The first pass keeps each card's printed text (and error), so the
    // second prints it instead of evaluating the card again: a sweep prints
    // a block per point, and each number goes through `%e`'s exact decimal
    // conversion. The kept texts lie back to back in deck order.
    var text: [kept_bytes_max]u8 = undefined;
    var used: usize = 0;
    var kept: [cards_max]Kept = @splat(.fresh);
    var kept_len: [cards_max]u16 = @splat(0);
    for (measures, 0..) |m, i| {
        if (!card.targets(m, analysis, result) or m.func == .param or i >= values.len) continue;
        var w: Writer = .fixed(text[used..]);
        const r = card.evaluate(&w, m, card.cardWave(result, analysis, &values, i, &done));
        if (r == error.WriteFailed) {
            // Out of buffer: the value as before, through a discarding writer.
            var discard_buf: [64]u8 = undefined;
            var discard = quiet(&discard_buf);
            values[i] = card.evaluate(&discard.writer, m, card.cardWave(result, analysis, &values, i, &done)) catch nan;
            continue;
        }
        values[i] = r catch nan;
        kept[i] = if (r) |_| .value else |e| switch (e) {
            error.NoSuchVector => .no_such_vector,
            error.OutOfInterval => .out_of_interval,
            error.WriteFailed => unreachable, // handled above
        };
        kept_len[i] = @intCast(w.end);
        used += w.end;
    }
    assert(used <= text.len);
    var headed = false;
    var replayed: usize = 0;
    for (measures, 0..) |m, i| {
        if (!card.targets(m, analysis, result)) continue;
        if (!headed) try writeHeading(out, analysis, point);
        headed = true;
        const status: Kept = if (i < kept.len) kept[i] else .fresh;
        const outcome: card.Error!f64 = switch (status) {
            .fresh => card.evaluate(out, m, card.cardWave(result, analysis, &values, i, &done)),
            .value, .no_such_vector, .out_of_interval => blk: {
                try out.writeAll(text[replayed..][0..kept_len[i]]);
                replayed += kept_len[i];
                break :blk switch (status) {
                    .value => values[i],
                    .no_such_vector => error.NoSuchVector,
                    .out_of_interval => error.OutOfInterval,
                    .fresh => unreachable,
                };
            },
        };
        const value = outcome catch |e| switch (e) {
            error.WriteFailed => return error.WriteFailed,
            error.NoSuchVector => blk: {
                try err.print("\nError: measure  {s} : no such vector\n", .{m.name});
                break :blk nan;
            },
            error.OutOfInterval => blk: {
                try err.print("\nError: measure  {s} : out of interval\n", .{m.name});
                break :blk nan;
            },
        };
        if (point != null and std.math.isNan(value)) try out.print("{s:<20}=  NaN\n", .{m.name});
        if (m.func == .param and i < values.len) values[i] = value;
    }
    assert(replayed == used);
}

/// `print` for a sweep point whose `analysis` failed: its heading and
/// `name = NaN` for every card of that analysis.
pub fn printFailed(out: *Writer, measures: []const core.Measure, analysis: Kind, point: Point) Writer.Error!void {
    var headed = false;
    for (measures) |m| {
        if (m.analysis != analysis) continue;
        if (!headed) try writeHeading(out, analysis, point);
        headed = true;
        try out.print("{s:<20}=  NaN\n", .{m.name});
    }
}

/// Writes mean, sigma, min and max of each card over Monte Carlo trials,
/// as `mean(name) = x` lines (sigma with Bessel's correction). `trials`
/// holds the results the cards are evaluated over; a card that fails in a
/// trial leaves that trial out. Cards past the 256th are not reported.
/// O(cards · trials · samples).
pub fn printStatistics(out: *Writer, measures: []const core.Measure, analysis: Kind, trials: []const core.Result) Writer.Error!void {
    var discard_buf: [64]u8 = undefined;
    var discard = quiet(&discard_buf);
    // Welford accumulators, one per card; every field moves together.
    const Acc = struct { n: f64 = 0, mean: f64 = 0, m2: f64 = 0, lo: f64 = std.math.inf(f64), hi: f64 = -std.math.inf(f64) };
    var acc: [cards_max]Acc = @splat(.{});
    const cards = measures[0..@min(measures.len, acc.len)];
    for (trials) |result| {
        // One pass in deck order: a card reads only the cards before it (a
        // later one is still NaN), so each card sees what it would alone.
        var values: [cards_max]f64 = @splat(nan);
        var done: Wave.Done = undefined;
        for (cards, 0..) |m, k| {
            if (m.analysis != analysis) continue;
            values[k] = card.evaluate(&discard.writer, m, card.cardWave(result, analysis, &values, k, &done)) catch nan;
            const x = values[k];
            if (!std.math.isFinite(x)) continue;
            const a = &acc[k];
            a.n += 1;
            const delta = x - a.mean;
            a.mean += delta / a.n;
            a.m2 += delta * (x - a.mean);
            a.lo = @min(a.lo, x);
            a.hi = @max(a.hi, x);
        }
    }
    var heading = false;
    for (cards, acc[0..cards.len]) |m, a| {
        if (a.n == 0) continue;
        if (!heading) {
            heading = true;
            try out.print("\n  Monte Carlo statistics over {d} trials\n\n", .{trials.len});
        }
        const sigma = if (a.n > 1) @sqrt(a.m2 / (a.n - 1)) else 0;
        try out.print("mean({s}) = {f}\nsigma({s}) = {f}\nmin({s}) = {f}\nmax({s}) = {f}\n", .{ m.name, sci(a.mean, 6), m.name, sci(sigma, 6), m.name, sci(a.lo, 6), m.name, sci(a.hi, 6) });
    }
}

// ---- private ----

/// ngspice's block heading, naming the sweep point when there is one.
fn writeHeading(out: *Writer, analysis: Kind, point: ?Point) Writer.Error!void {
    try out.print("\n  Measurements for {s} Analysis", .{switch (analysis) {
        .tran => "Transient",
        .tran_noise => "Transient Noise",
        .ac => "AC",
        .sp => "SP",
        .fft => "FFT",
        .dcmatch => "DC Mismatch",
        .acmatch => "AC Mismatch",
        .lstb => "Loop Stability",
        .phasenoise => "Phase Noise",
        .pnoise => "Periodic Time-Dependent Noise",
        .op => "Operating Point",
        else => "DC", // else: any other analysis takes the DC heading; a kind with its own title adds an arm
    }});
    if (point) |p| try out.print(", point {d} ({s})", .{ p.row, p.label });
    try out.writeAll("\n\n");
}

test "measurements match ngspice on a sampled ramp" {
    // v(a) = t on t = 0..4, sampled every 1; v(b) falls 4..0.
    const data = [_]f64{ 0, 0, 4, 1, 1, 3, 2, 2, 2, 3, 3, 1, 4, 4, 0 };
    const result: core.Result = .{ .plotname = "t", .varnames = &.{ "time", "v(a)", "v(b)" }, .is_complex = false, .npoints = 5, .data = &data };
    const cards = [_]core.Measure{
        .{ .analysis = .tran, .name = "d", .func = .trig_targ, .first = .{ .vec = "v(a)", .val = 0.5, .rise = 1 }, .second = .{ .vec = "b", .val = 1.5, .fall = 1 } },
        .{ .analysis = .tran, .name = "f", .func = .find, .first = .{ .vec = "v(b)", .at = 2.5 } },
        .{ .analysis = .tran, .name = "i", .func = .integ, .first = .{ .vec = "v(a)" } },
        .{ .analysis = .tran, .name = "mx", .func = .max, .first = .{ .vec = "v(a)", .to = 3 } },
        .{ .analysis = .tran, .name = "gone", .func = .when, .first = .{ .vec = "v(a)", .val = 9 } },
        .{ .analysis = .ac, .name = "skipped", .func = .max, .first = .{ .vec = "v(a)" } },
        // HSPICE forms: DERIVATIVE, ERR2 (t = 0 falls under YMIN), PARAM.
        .{ .analysis = .tran, .name = "s", .func = .deriv, .first = .{ .vec = "v(b)", .at = 2.5 } },
        .{ .analysis = .tran, .name = "e", .func = .err2, .first = .{ .vec = "v(a)", .vec2 = "v(b)" } },
        .{ .analysis = .tran, .name = "p", .func = .param, .first = .{}, .expr = &.{ .{ .measure = 0 }, .{ .num = 2 }, .mul } },
    };
    var out_buf: [1024]u8 = undefined;
    var err_buf: [256]u8 = undefined;
    var out: Writer = .fixed(&out_buf);
    var err: Writer = .fixed(&err_buf);
    try print(&out, &err, &cards, .tran, result, null);
    // Rise 1 of v(a) happens in the first interval, where ngspice only
    // initialises its counters: it is measured on the next one, by
    // extrapolating that segment (exact here, as v(a) is a line).
    try std.testing.expectEqualStrings(
        \\
        \\  Measurements for Transient Analysis
        \\
        \\d                   =  2.000000e+00 targ=  2.500000e+00 trig=  5.000000e-01
        \\f                   =  1.500000e+00
        \\i                   =   8.00000e+00 from=  0.00000e+00 to=  4.00000e+00
        \\mx                  =  3.000000e+00 at=  3.000000e+00
        \\s                   =  -1.000000e+00
        \\e                   =  9.166667e-01
        \\p                   =  4.00000e+00
        \\
    , out.buffered());
    try std.testing.expect(std.mem.indexOf(u8, err.buffered(), "gone") != null);
}

test "a _CONT FIND or DERIV at a fixed AT= has one event and prints one line" {
    const cards = [_]core.Measure{
        .{ .analysis = .tran, .name = "x", .func = .find, .cont = true, .first = .{ .vec = "v(a)", .at = 1.5 } },
        .{ .analysis = .tran, .name = "d", .func = .deriv, .cont = true, .first = .{ .vec = "v(b)", .at = 1.5 } },
    };
    var values: [cards.len]f64 = undefined;
    card.evaluateAll(&cards, .tran, ramp, &values);
    try std.testing.expectEqualSlices(f64, &.{ 1.5, -1 }, &values);
    var buf: [256]u8 = undefined;
    var out: Writer = .fixed(&buf);
    try print(&out, &out, &cards, .tran, ramp, null);
    try std.testing.expectEqualStrings(
        \\
        \\  Measurements for Transient Analysis
        \\
        \\x[1]                =  1.500000e+00
        \\d[1]                =  -1.000000e+00
        \\
    , out.buffered());
}

test "every function over a result with no samples reads NaN or zero, never panics" {
    const funcs = comptime std.enums.values(core.MeasureFunc);
    var cards: [funcs.len]core.Measure = undefined;
    for (funcs, &cards) |f, *c| c.* = .{ .analysis = .tran, .name = "m", .func = f, .first = .{ .vec = "v(a)", .vec2 = "v(b)" }, .second = .{ .vec = "v(b)" } };
    var empty = ramp;
    empty.npoints = 0;
    empty.data = &.{};
    // A streamed result keeps its point count but no samples.
    var streamed = ramp;
    streamed.data = &.{};
    var buf: [64]u8 = undefined;
    var sink: Writer.Discarding = .init(&buf);
    for ([_]core.Result{ empty, streamed }) |result| {
        var values: [funcs.len]f64 = undefined;
        card.evaluateAll(&cards, .tran, result, &values);
        for (cards, values) |c, v| switch (c.func) {
            // ngspice measure_minMaxAvg reports 0 over an empty window, and
            // a check over no samples counts no violation.
            .avg, .min, .max, .min_at, .max_at, .pp, .check_slew, .check_forbid, .check_require, .check_level, .dout => try std.testing.expectEqual(0, v),
            else => try std.testing.expect(std.math.isNan(v)), // else: every other function has no value over no samples
        };
        try print(&sink.writer, &sink.writer, &cards, .tran, result, null);
        var out: [3]f64 = undefined;
        try card.sample(.{ .vec = "v(a)" }, result, &.{ -1, 0, 9 }, &out);
        for (out) |v| try std.testing.expect(std.math.isNan(v));
    }
}

test printStatistics {
    var doubled = ramp;
    doubled.data = &.{ 0, 0, 0, 1, 2, 0, 2, 4, 0, 3, 6, 0, 4, 8, 0 };
    const cards = [_]core.Measure{
        .{ .analysis = .tran, .name = "mx", .func = .max, .first = .{ .vec = "a" } },
        .{ .analysis = .tran, .name = "p", .func = .param, .first = .{}, .expr = &.{ .{ .measure = 0 }, .{ .num = 2 }, .mul } },
        // A PARAM reads only the cards before it: `late` is NaN here.
        .{ .analysis = .tran, .name = "q", .func = .param, .first = .{}, .expr = &.{.{ .measure = 3 }} },
        .{ .analysis = .tran, .name = "late", .func = .max, .first = .{ .vec = "v(a)" } },
        .{ .analysis = .ac, .name = "other", .func = .max, .first = .{ .vec = "v(a)" } },
    };
    var buf: [1024]u8 = undefined;
    var out: Writer = .fixed(&buf);
    try printStatistics(&out, &cards, .tran, &.{ ramp, doubled });
    try std.testing.expectEqualStrings(
        \\
        \\  Monte Carlo statistics over 2 trials
        \\
        \\mean(mx) = 6.000000e+00
        \\sigma(mx) = 2.828427e+00
        \\min(mx) = 4.000000e+00
        \\max(mx) = 8.000000e+00
        \\mean(p) = 1.200000e+01
        \\sigma(p) = 5.656854e+00
        \\min(p) = 8.000000e+00
        \\max(p) = 1.600000e+01
        \\mean(late) = 6.000000e+00
        \\sigma(late) = 2.828427e+00
        \\min(late) = 4.000000e+00
        \\max(late) = 8.000000e+00
        \\
    , out.buffered());
}

test "crossings, windows, NaN samples and a single point match ngspice 45" {
    // Each expected listing is what ngspice 45 prints for the same cards over
    // the same samples, loaded from an ASCII raw file; a failed card prints
    // nothing to `out`.
    const tri: core.Result = .{ .plotname = "t", .varnames = &.{ "time", "v(a)" }, .is_complex = false, .npoints = 9, .data = &.{ 0, 0, 1, 2, 2, 0, 3, 2, 4, 0, 5, 2, 6, 0, 7, 2, 8, 0 } };
    const with_nan: core.Result = .{ .plotname = "t", .varnames = &.{ "time", "v(a)" }, .is_complex = false, .npoints = 5, .data = &.{ 0, 0, 1, 1, 2, nan, 3, 3, 4, 2 } };
    const one: core.Result = .{ .plotname = "t", .varnames = &.{ "time", "v(a)" }, .is_complex = false, .npoints = 1, .data = &.{ 0, 5 } };
    const last = core.measure_last;
    const a: Clause = .{ .vec = "v(a)" };
    const tri_cards = [_]core.Measure{
        // RISE=1 falls in the first interval, which only primes the counters:
        // ngspice reports the next crossing, the fall at 1.5.
        .{ .analysis = .tran, .name = "r1", .func = .when, .first = .{ .vec = "v(a)", .val = 1, .rise = 1 } },
        .{ .analysis = .tran, .name = "r2", .func = .when, .first = .{ .vec = "v(a)", .val = 1, .rise = 2 } },
        .{ .analysis = .tran, .name = "f1", .func = .when, .first = .{ .vec = "v(a)", .val = 1, .fall = 1 } },
        .{ .analysis = .tran, .name = "c1", .func = .when, .first = .{ .vec = "v(a)", .val = 1, .cross = 1 } },
        .{ .analysis = .tran, .name = "c3", .func = .when, .first = .{ .vec = "v(a)", .val = 1, .cross = 3 } },
        .{ .analysis = .tran, .name = "fl", .func = .when, .first = .{ .vec = "v(a)", .val = 1, .fall = last } },
        .{ .analysis = .tran, .name = "rl", .func = .when, .first = .{ .vec = "v(a)", .val = 1, .rise = last } },
        .{ .analysis = .tran, .name = "cl", .func = .when, .first = .{ .vec = "v(a)", .val = 1, .cross = last } },
        .{ .analysis = .tran, .name = "w0", .func = .when, .first = .{ .vec = "v(a)", .val = 1 } },
        .{ .analysis = .tran, .name = "tt", .func = .trig_targ, .first = .{ .vec = "v(a)", .val = 1, .rise = 2 }, .second = .{ .vec = "v(a)", .val = 1, .fall = 2 } },
        .{ .analysis = .tran, .name = "x", .func = .find, .first = .{ .vec = "v(a)", .at = 2.25 } },
        .{ .analysis = .tran, .name = "mx", .func = .max, .first = .{ .vec = "v(a)", .from = 0.5, .to = 3.5 } },
        .{ .analysis = .tran, .name = "av", .func = .avg, .first = .{ .vec = "v(a)", .to = 4 } },
        .{ .analysis = .tran, .name = "ig", .func = .integ, .first = .{ .vec = "v(a)", .from = 0.5, .to = 3.5 } },
        .{ .analysis = .tran, .name = "rm", .func = .rms, .first = a },
        .{ .analysis = .tran, .name = "pp", .func = .pp, .first = a },
        .{ .analysis = .tran, .name = "mn", .func = .min, .first = .{ .vec = "v(a)", .from = 1, .to = 3 } },
        .{ .analysis = .tran, .name = "wd", .func = .when, .first = .{ .vec = "v(a)", .val = 1, .td = 2.6 } },
    };
    const sparse_cards = [_]core.Measure{
        .{ .analysis = .tran, .name = "mx", .func = .max, .first = a },
        .{ .analysis = .tran, .name = "mn", .func = .min, .first = a },
        .{ .analysis = .tran, .name = "w", .func = .when, .first = .{ .vec = "v(a)", .val = 1.5 } },
        .{ .analysis = .tran, .name = "w2", .func = .when, .first = .{ .vec = "v(a)", .val = 2.5, .fall = 1 } },
        .{ .analysis = .tran, .name = "x", .func = .find, .first = .{ .vec = "v(a)", .at = 0.5 } },
        .{ .analysis = .tran, .name = "x0", .func = .find, .first = .{ .vec = "v(a)", .at = 0 } },
        .{ .analysis = .tran, .name = "av", .func = .avg, .first = a },
        .{ .analysis = .tran, .name = "ig", .func = .integ, .first = a },
        .{ .analysis = .tran, .name = "rm", .func = .rms, .first = a },
        .{ .analysis = .tran, .name = "pp", .func = .pp, .first = a },
    };
    const cases = [_]struct { core.Result, []const core.Measure, []const u8 }{
        .{
            tri, &tri_cards,
            \\r1                  =   1.50000e+00
            \\r2                  =   2.50000e+00
            \\f1                  =   1.50000e+00
            \\c3                  =   2.50000e+00
            \\fl                  =   7.50000e+00
            \\rl                  =   6.50000e+00
            \\cl                  =   7.50000e+00
            \\w0                  =   1.50000e+00
            \\tt                  =  1.000000e+00 targ=  3.500000e+00 trig=  2.500000e+00
            \\x                   =  5.000000e-01
            \\mx                  =  2.000000e+00 at=  3.000000e+00
            \\av                  =  1.000000e+00 from=  0.000000e+00 to=  5.000000e+00
            \\ig                  =   2.50000e+00 from=  5.00000e-01 to=  3.50000e+00
            \\rm                  =   1.47196e+00 from=  0.00000e+00 to=  8.00000e+00
            \\pp                  =  2.000000e+00 from=  0.000000e+00 to=  0.000000e+00
            \\mn                  =  0.000000e+00 at=  2.000000e+00
            \\wd                  =   4.50000e+00
            \\
        },
        .{
            with_nan, &sparse_cards,
            \\mx                  =  3.000000e+00 at=  3.000000e+00
            \\mn                  =  0.000000e+00 at=  0.000000e+00
            \\w2                  =   3.50000e+00
            \\x                   =  5.000000e-01
            \\x0                  =  0.000000e+00
            \\pp                  =  3.000000e+00 from=  0.000000e+00 to=  0.000000e+00
            \\
        },
        .{
            one, &sparse_cards,
            \\mx                  =  5.000000e+00 at=  0.000000e+00
            \\mn                  =  5.000000e+00 at=  0.000000e+00
            \\ig                  =   0.00000e+00 from=  0.00000e+00 to=  0.00000e+00
            \\pp                  =  0.000000e+00 from=  0.000000e+00 to=  0.000000e+00
            \\
        },
    };
    for (cases) |case| {
        var buf: [2048]u8 = undefined;
        var out: Writer = .fixed(&buf);
        var sink: Writer.Discarding = .init(&.{});
        try print(&out, &sink.writer, case[1], .tran, case[0], null);
        const heading = "\n  Measurements for Transient Analysis\n\n";
        try std.testing.expectEqualStrings(heading, out.buffered()[0..heading.len]);
        try std.testing.expectEqualStrings(case[2], out.buffered()[heading.len..]);
    }
}
