//! One `.meas` card evaluated: a card and the `Wave` it reads in, its
//! listing line (ngspice's `name = value` text) and its value out, plus the
//! batch forms over every card of a deck that only want the values.
//! Cites: ngspice 45 com_measure2.c (measure functions and their printed
//! lines), measure.c (PARAM cards read every other card); HSPICE [CR
//! .MEASURE] for DERIV, ERR*, EM_AVG, JITTER, the FFT figures, `_CONT`
//! and `.stim`; [SA Ch.15] for the FFT figures.
//!
//! Contents:
//! - `pub const Error = Writer.Error || error{ NoSuchVector, OutOfInterval }`
//! - `pub fn evaluate(out: *Writer, m: core.Measure, w: Wave) Error!f64`
//! - `pub fn evaluateAll(measures: []const core.Measure, analysis: Kind, result: core.Result, out: []f64) void`
//! - `pub fn evaluateOutcomes(measures: []const core.Measure, analysis: Kind, result: core.Result, out: []f64, missing: ?[]bool) void`
//! - `pub fn sample(c: Clause, result: core.Result, at: []const f64, out: []f64) error{NoSuchVector}!void`
//! - `pub fn targets(m: core.Measure, analysis: Kind, result: core.Result) bool`
//! - `pub fn cardWave(result: core.Result, analysis: Kind, values: []const f64, i: usize, done: *Wave.Done) Wave`
//! - `pub fn resolve(c: Clause, values: []const f64) error{OutOfInterval}!Clause`
//! - `pub fn lastEvent(c: Clause) bool`
//! - `pub fn paramValue(ops: []const core.MeasureOp, values: []const f64) f64`
//! - `Results.leaf`, public for `column.fold`

const std = @import("std");
const assert = std.debug.assert;
const Writer = std.Io.Writer;
const core = @import("core");
const scan = @import("scan.zig");
const columns = @import("column.zig");
const checks = @import("checks.zig");
const Wave = @import("Wave.zig");
const sci = @import("sci.zig").sci;
const quiet = @import("sci.zig").quiet;
const Clause = core.MeasureClause;
const Kind = core.query.Kind;

/// Why a card has no value. `NoSuchVector`: a clause names a vector the
/// result lacks. `OutOfInterval`: its event never happens, or a value it
/// reads is NaN. A `WriteFailed` leaves part of the card's line written.
pub const Error = Writer.Error || error{ NoSuchVector, OutOfInterval };

const nan = std.math.nan(f64);

/// `paramValue`'s leaves: earlier cards' results by index.
const Results = struct {
    values: []const f64,

    pub fn leaf(s: Results, op: core.MeasureOp, _: usize) f64 {
        return switch (op) {
            .num => |x| x,
            .measure => |k| if (k < s.values.len) s.values[k] else nan,
            .vector, .neg, .add, .sub, .mul, .div, .pow => nan,
        };
    }
};

/// v(a) = t and v(b) = 4 - t on t = 0..4, sampled every 1 (test fixture).
const ramp: core.Result = .{ .plotname = "t", .varnames = &.{ "time", "v(a)", "v(b)" }, .is_complex = false, .npoints = 5, .data = &.{ 0, 0, 4, 1, 1, 3, 2, 2, 2, 3, 3, 1, 4, 4, 0 } };

/// Writes card `m`'s listing line to `out` and returns its value, the one
/// a PARAM card reads. A failed card writes nothing but a check's
/// violation lines.
pub fn evaluate(out: *Writer, m: core.Measure, w: Wave) Error!f64 {
    if (m.cont) return evaluateCont(out, m, w);
    const a = try resolve(m.first, w.values);
    var b = try resolve(m.second, w.values);
    switch (m.func) {
        .trig_targ => {
            const trig = try defined(if (a.at == core.measure_no_at) try w.when(a) else a.at);
            if (b.td_trig) b.td = trig;
            const targ = try defined(if (b.at == core.measure_no_at) try w.when(b) else b.at);
            try out.print("{s:<20}=  {f} targ=  {f} trig=  {f}\n", .{ m.name, sci(targ - trig, 6), sci(targ, 6), sci(trig, 6) });
            return targ - trig;
        },
        .find, .deriv => {
            // A bare FIND: the value of a one-row plot.
            if (a.at == core.measure_no_at and b.vec.len == 0) {
                if (w.len() != 1) return error.OutOfInterval;
                const v = try defined((try w.waveform(a.vec, a.ops, a.vectype)).get(0));
                try out.print("{s:<20}=  {f}\n", .{ m.name, sci(v, 6) });
                return v;
            }
            const at = if (a.at == core.measure_no_at) try defined(try w.when(b)) else a.at;
            const v = try defined(if (m.func == .find) try w.valueAt(a, at) else try w.slopeAt(a, at));
            try out.print("{s:<20}=  {f}\n", .{ m.name, sci(v, 6) });
            return v;
        },
        .when => {
            const v = try defined(try w.when(a));
            try out.print("{s:<20}=   {f}\n", .{ m.name, sci(v, 5) });
            return v;
        },
        .rms, .integ => {
            const r = try w.rmsInteg(a, m.func == .rms);
            try out.print("{s:<20}=   {f} from=  {f} to=  {f}\n", .{ m.name, sci(try defined(r.value), 5), sci(r.from, 5), sci(r.to, 5) });
            return r.value;
        },
        .jitter => {
            const j = try w.jitter(a);
            try out.print("{s:<20}=  {f} pp=  {f} period=  {f} edges=  {d}\n", .{ m.name, sci(j.rms, 6), sci(j.pp, 6), sci(j.period, 6), j.edges });
            return j.rms;
        },
        .em_avg => {
            const r = try w.emAvg(a);
            try out.print("{s:<20}=  {f} from=  {f} to=  {f}\n", .{ m.name, sci(try defined(r.value), 6), sci(r.from, 6), sci(r.to, 6) });
            return r.value;
        },
        .avg => {
            const r = try w.extremum(a, .avg);
            try out.print("{s:<20}=  {f} from=  {f} to=  {f}\n", .{ m.name, sci(try defined(r.value), 6), sci(if (a.at == core.measure_no_at) a.from else a.at, 6), sci(r.at, 6) });
            return r.value;
        },
        .min, .max => {
            const r = try w.extremum(a, if (m.func == .min) .min else .max);
            try out.print("{s:<20}=  {f} at=  {f}\n", .{ m.name, sci(try defined(r.value), 6), sci(r.at, 6) });
            return r.value;
        },
        .min_at, .max_at => {
            const r = try w.extremum(a, if (m.func == .min_at) .min else .max);
            try out.print("{s:<20}=  {f} with=  {f}\n", .{ m.name, sci(try defined(r.at), 6), sci(r.value, 6) });
            return r.at;
        },
        .pp => {
            const lo = try defined((try w.extremum(a, .min)).value);
            const hi = try defined((try w.extremum(a, .max)).value);
            try out.print("{s:<20}=  {f} from=  {f} to=  {f}\n", .{ m.name, sci(hi - lo, 6), sci(a.from, 6), sci(a.to, 6) });
            return hi - lo;
        },
        .param => {
            const v = try defined(paramValue(m.expr, w.values));
            try out.print("{s:<20}=  {f}\n", .{ m.name, sci(v, 5) });
            return v;
        },
        .err, .err1, .err2, .err3 => {
            const v = try defined(try w.relError(a, m.func));
            try out.print("{s:<20}=  {f}\n", .{ m.name, sci(v, 6) });
            return v;
        },
        .thd, .snr, .sndr, .enob, .sfdr => {
            const v = try defined(fftFigure(try w.column(a.vec, 'm'), w.scale(), w.len(), a, m.func));
            try out.print("{s:<20}=  {f}\n", .{ m.name, sci(v, 6) });
            return v;
        },
        .check_slew, .check_forbid, .check_require, .check_level, .dout => {
            const n = try checks.violations(w, out, m.name, a, b, m.func, m.check);
            try out.print("{s:<20}=  {d}\n", .{ m.name, n });
            return @floatFromInt(n);
        },
    }
}

/// Writes every card in `measures` that targets `analysis`, evaluated over
/// `result`, into `out` (parallel to `measures`), PARAM cards last in deck
/// order, as `print` does. A card that does not target the result, or
/// whose event never happens, reads NaN. Asserts that
/// `out.len == measures.len`.
pub fn evaluateAll(measures: []const core.Measure, analysis: Kind, result: core.Result, out: []f64) void {
    evaluateOutcomes(measures, analysis, result, out, null);
}

/// `evaluateAll`, also setting `missing[i]` (when given, parallel to
/// `measures`) for a card that names a vector `result` lacks, ngspice's
/// "no such vector", and clearing it for every other card. Asserts both
/// lengths.
pub fn evaluateOutcomes(measures: []const core.Measure, analysis: Kind, result: core.Result, out: []f64, missing: ?[]bool) void {
    assert(out.len == measures.len);
    if (missing) |ms| assert(ms.len == measures.len);
    @memset(out, nan);
    if (missing) |ms| @memset(ms, false);
    var discard_buf: [64]u8 = undefined;
    var discard = quiet(&discard_buf);
    var done: Wave.Done = undefined;
    for ([_]bool{ false, true }) |param| for (measures, out, 0..) |m, *v, i| {
        if (!targets(m, analysis, result) or (m.func == .param) != param) continue;
        v.* = evaluate(&discard.writer, m, cardWave(result, analysis, out, i, &done)) catch |e| blk: {
            if (e == error.NoSuchVector) if (missing) |ms| {
                ms[i] = true;
            };
            break :blk nan;
        };
    };
}

/// Writes clause `c`'s waveform over transient `result` at each time in
/// `at` into `out`, interpolated linearly and held flat past either end. A
/// result with no samples (none computed, or streamed to a file) reads NaN
/// everywhere. O(len(at) · log samples). Asserts that `at.len == out.len`.
pub fn sample(c: Clause, result: core.Result, at: []const f64, out: []f64) error{NoSuchVector}!void {
    assert(at.len == out.len);
    const w: Wave = .{ .result = result, .analysis = .tran };
    const x = w.scale();
    const y = try w.waveform(c.vec, c.ops, c.vectype);
    const n = w.len();
    if (n == 0) return @memset(out, nan);
    for (at, out) |t, *v| {
        // The first sample at or after t.
        var lo: usize = 0;
        var hi: usize = n;
        while (lo < hi) {
            const mid = (lo + hi) / 2;
            if (x.get(mid) < t) lo = mid + 1 else hi = mid;
        }
        assert(lo <= n);
        // ngspice measure_interpolate, 'y' mode, between the samples around t.
        v.* = if (lo == 0) y.get(0) else if (lo == n) y.get(n - 1) else scan.line(x.get(lo - 1), y.get(lo - 1), x.get(lo), y.get(lo), t);
    }
}

/// Whether card `m` reads `result`. Each `.fft` card has its own plot, and
/// `.lstb` and `.ptdnoise` publish two kinds under one analysis, so those
/// cards read the plot holding their vector.
pub fn targets(m: core.Measure, analysis: Kind, result: core.Result) bool {
    if (m.analysis != analysis) return false;
    if ((analysis != .fft and analysis != .lstb and analysis != .pnoise) or m.func == .param) return true;
    return columns.columnIndex(result.varnames, m.first.vec) != null;
}

/// Returns the wave card `i` reads over `result`: its columns, or what its
/// run found (`core.Result.measured`) through `done`, which must outlive
/// the wave.
pub fn cardWave(result: core.Result, analysis: Kind, values: []const f64, i: usize, done: *Wave.Done) Wave {
    var w: Wave = .{ .result = result, .analysis = analysis, .values = values };
    if (i < result.measured.len and result.measured[i].scans > 0) {
        const m = &result.measured[i];
        assert(m.scans <= m.found.len);
        done.* = .{ .found = m.found[0..m.scans] };
        w.done = done;
    }
    return w;
}

/// Returns `c` with its refs to earlier results applied; `OutOfInterval`
/// when one has no value.
pub fn resolve(c: Clause, values: []const f64) error{OutOfInterval}!Clause {
    var r = c;
    for (c.refs) |ref| {
        const v = try defined(paramValue(ref.expr, values));
        switch (ref.field) {
            .val => r.val = v,
            .td => r.td = v,
            .from => r.from = v,
            .to => r.to = v,
            .at => r.at = v,
        }
    }
    return r;
}

/// Whether `c` counts its LAST rise, fall or crossing.
pub fn lastEvent(c: Clause) bool {
    return c.rise == core.measure_last or c.fall == core.measure_last or c.cross == core.measure_last;
}

/// Returns a PARAM expression folded over the results in `values`; NaN
/// when a result it reads is missing or it reads a vector.
pub fn paramValue(ops: []const core.MeasureOp, values: []const f64) f64 {
    return columns.fold(ops, Results{ .values = values });
}

// ---- private ----

fn defined(x: f64) error{OutOfInterval}!f64 {
    return if (std.math.isNan(x)) error.OutOfInterval else x;
}

/// A `_CONT` card: the event it names and every later one, printed as
/// `name[k]`. Returns the first; `OutOfInterval` when there is none.
fn evaluateCont(out: *Writer, card: core.Measure, w: Wave) Error!f64 {
    assert(card.cont);
    var m = card;
    m.first = try resolve(card.first, w.values);
    m.second = try resolve(card.second, w.values);
    var first: f64 = nan;
    var buf: [96]u8 = undefined;
    var k: i32 = 0;
    // Ends when the events run out: `nth` is null past an AT clause, and
    // event k + 1 needs k + 1 crossings, so once k passes the sample count
    // `when` finds none and `evaluate` fails OutOfInterval.
    while (true) : (k += 1) {
        assert(@as(usize, @intCast(k)) <= w.len() + 1);
        var one = m;
        one.cont = false;
        one.name = std.fmt.bufPrint(&buf, "{s}[{d}]", .{ m.name, k + 1 }) catch m.name;
        switch (m.func) {
            .trig_targ => {
                one.first = nth(m.first, k) orelse break;
                one.second = nth(m.second, k) orelse break;
            },
            // FIND or DERIV at a fixed AT= has the one event (#68).
            .find, .deriv => if (m.first.at != core.measure_no_at) {
                one.first = nth(m.first, k) orelse break;
            } else {
                one.second = nth(m.second, k) orelse break;
            },
            .when => one.first = nth(m.first, k) orelse break,
            else => return evaluate(out, one, w), // else: a function without event vectors runs once
        }
        const v = evaluate(out, one, w) catch |e| switch (e) {
            error.OutOfInterval => break,
            else => return e, // else: every other failure is the caller's
        };
        if (k == 0) first = v;
        if (lastEvent(m.first) or lastEvent(m.second)) break;
    }
    return defined(first);
}

/// `c` counting `k` events past the one it names (CROSS=1 when it names
/// none); null for an `AT=` clause past its one value.
fn nth(c: Clause, k: i32) ?Clause {
    assert(k >= 0);
    if (c.at != core.measure_no_at) return if (k == 0) c else null;
    var r = c;
    if (c.rise > 0) r.rise = c.rise + k else if (c.fall > 0) r.fall = c.fall + k else if (c.cross > 0) r.cross = c.cross + k else if (!lastEvent(c)) r.cross = 1 + k;
    return r;
}

/// An FFT figure of merit over the magnitude spectrum `mag` on `freq`
/// [SA Ch.15, CR .MEASURE FFT]. The fundamental is the largest non-DC bin;
/// its harmonics are its bin multiples up to NBHARM (every one when 0) and
/// MAXFREQ; BINSIZ bins either side of it count as signal. THD is the ratio
/// sqrt(Σ harmonic²)/fundamental; SNR leaves the harmonics out of the noise,
/// SNDR keeps them, both in dB over every other non-DC bin; ENOB is
/// (SNDR − 1.76)/6.02; SFDR is the fundamental over the largest other bin
/// in [MINFREQ, MAXFREQ], in dB. NaN for a spectrum with no non-DC bin.
fn fftFigure(mag: columns.Column, freq: columns.Column, n: usize, c: Clause, func: core.MeasureFunc) f64 {
    if (n < 2) return nan;
    var k0: usize = 0;
    for (1..n) |k| if (k0 == 0 or mag.get(k) > mag.get(k0)) {
        k0 = k;
    };
    if (k0 == 0) return nan;
    assert(k0 < n);
    const fund = mag.get(k0);
    var harmonics: f64 = 0;
    var noise: f64 = 0;
    var spur: f64 = 0;
    for (1..n) |k| {
        const d = if (k > k0) k - k0 else k0 - k;
        if (d <= c.binsiz) continue;
        const v = mag.get(k);
        const f = freq.get(k);
        const h = k / k0;
        const harmonic = k % k0 == 0 and (c.nbharm == 0 or h <= c.nbharm) and f <= c.to;
        if (harmonic) harmonics += v * v else noise += v * v;
        if (f >= c.from and f <= c.to) spur = @max(spur, v);
    }
    const sndr = 10 * std.math.log10(fund * fund / (harmonics + noise));
    return switch (func) {
        .thd => @sqrt(harmonics) / fund,
        .snr => 10 * std.math.log10(fund * fund / noise),
        .sndr => sndr,
        .enob => (sndr - 1.76) / 6.02,
        .sfdr => 20 * std.math.log10(fund / spur),
        else => unreachable, // else: `fftFigure` is called with the five FFT functions only
    };
}

test "AT= naming an earlier result reads that result, _CONT too" {
    const t50: []const core.MeasureOp = &.{.{ .measure = 0 }};
    const at_t50: []const core.MeasureRef = &.{.{ .field = .at, .expr = t50 }};
    const cards = [_]core.Measure{
        .{ .analysis = .tran, .name = "t", .func = .when, .first = .{ .vec = "v(a)", .val = 2.5 } },
        .{ .analysis = .tran, .name = "v", .func = .find, .first = .{ .vec = "v(b)", .refs = at_t50 } },
        .{ .analysis = .tran, .name = "c", .func = .find, .cont = true, .first = .{ .vec = "v(b)", .refs = at_t50 } },
    };
    var values: [cards.len]f64 = undefined;
    evaluateAll(&cards, .tran, ramp, &values);
    try std.testing.expectEqualSlices(f64, &.{ 2.5, 1.5, 1.5 }, &values);
}

test sample {
    var out: [5]f64 = undefined;
    try sample(.{ .vec = "v(a)" }, ramp, &.{ -1, 0, 2.5, 3, 9 }, &out);
    try std.testing.expectEqualSlices(f64, &.{ 0, 0, 2.5, 3, 4 }, &out);
    try std.testing.expectError(error.NoSuchVector, sample(.{ .vec = "v(z)" }, ramp, &.{0}, out[0..1]));
}

test "FFT figures of merit over a synthetic spectrum" {
    // DC 5 (ignored), fundamental 1 at bin 2, harmonics 0.1 at bins 4 and
    // 6, noise 0.01 in every odd bin.
    const spectrum: core.Result = .{ .plotname = "fft", .varnames = &.{ "frequency", "v(o)" }, .is_complex = false, .npoints = 8, .data = &.{ 0, 5, 1, 0.01, 2, 1, 3, 0.01, 4, 0.1, 5, 0.01, 6, 0.1, 7, 0.01 } };
    const clause: Clause = .{ .vec = "v(o)", .to = 100 };
    const cards = [_]core.Measure{
        .{ .analysis = .fft, .name = "thd", .func = .thd, .first = clause },
        .{ .analysis = .fft, .name = "snr", .func = .snr, .first = clause },
        .{ .analysis = .fft, .name = "sfdr", .func = .sfdr, .first = clause },
    };
    var values: [cards.len]f64 = undefined;
    evaluateAll(&cards, .fft, spectrum, &values);
    try std.testing.expectApproxEqRel(@sqrt(0.02), values[0], 1e-12);
    try std.testing.expectApproxEqRel(10 * std.math.log10(2500.0), values[1], 1e-12);
    try std.testing.expectApproxEqRel(20, values[2], 1e-12);
}

test "par() waveforms resolve their vectors, past the cached eight too" {
    const v: core.MeasureOp = .{ .vector = "v(a)" };
    const nine = [_]core.MeasureOp{ v, v, .add, v, .add, v, .add, v, .add, v, .add, v, .add, v, .add, v, .add };
    const cards = [_]core.Measure{
        .{ .analysis = .tran, .name = "sum", .func = .max, .first = .{ .ops = &.{ .{ .vector = "a" }, .{ .vector = "v(b)" }, .add } } },
        .{ .analysis = .tran, .name = "nine", .func = .max, .first = .{ .ops = &nine } },
        .{ .analysis = .tran, .name = "missing", .func = .max, .first = .{ .ops = &.{.{ .vector = "v(z)" }} } },
    };
    var values: [cards.len]f64 = undefined;
    evaluateAll(&cards, .tran, ramp, &values);
    try std.testing.expectEqual(4, values[0]);
    try std.testing.expectEqual(36, values[1]);
    try std.testing.expect(std.math.isNan(values[2]));
}

test "malformed PARAM postfix reads NaN" {
    try std.testing.expect(std.math.isNan(paramValue(&.{.add}, &.{})));
    try std.testing.expect(std.math.isNan(paramValue(&.{.neg}, &.{})));
    try std.testing.expect(std.math.isNan(paramValue(&.{ .{ .num = 1 }, .{ .num = 2 } }, &.{})));
    try std.testing.expect(std.math.isNan(paramValue(&.{.{ .measure = 7 }}, &.{1})));
    const deep: [33]core.MeasureOp = @splat(.{ .num = 1 });
    try std.testing.expect(std.math.isNan(paramValue(&deep, &.{})));
    try std.testing.expectEqual(-6, paramValue(&.{ .{ .num = 2 }, .{ .measure = 0 }, .pow, .neg, .{ .num = 2 }, .sub }, &.{2}));
}
