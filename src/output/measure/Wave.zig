//! The result one `.meas` card reads, and the whole-waveform measurements
//! over it: a result (or what its run already found, `Done`) and a clause
//! in, an event time, a value at a point, an extremum, an average, an
//! integral, an error figure or jitter out. Each runs a one-sample-at-a-time
//! pass (scan.zig, or `EmAvg`, `RelError` here) over every sample in order,
//! so a `Stream` that feeds the same passes row by row gets the same bits.
//! Cites: ngspice 45 com_measure2.c (com_measure_when, measure_at,
//! measure_minMaxAvg, measure_rms_integral); HSPICE [SA Ch.11 "Error
//! Equations"], [CR .MEASURE (AVG, EM_AVG, ...)].
//!
//! Contents:
//! - fields `result`, `analysis`, `values`, `done`
//! - `pub const Done = struct` with `pub fn take(*Done) error{NoSuchVector}![3]f64`
//! - `pub const Error = error{NoSuchVector}`
//! - `pub const Jitter`, `pub const EmAvg` and `pub const RelError` (each `pub fn feed`, `result`/`value`)
//! - `pub fn bounds(c: Clause) scan.Bounds`, `pub fn modeOf(analysis: Kind) scan.Mode`
//! - `pub fn len(Wave) usize`, `pub fn scale(Wave) Column`, `pub fn col(Wave, usize, Part) Column`,
//!   `pub fn column(Wave, []const u8, u8) Error!Column`,
//!   `pub fn waveform(Wave, []const u8, []const core.MeasureOp, u8) Error!Waveform`
//! - `pub fn when(Wave, Clause) Error!f64`, `pub fn crossings(Wave, Clause, []f64) Error!usize`
//! - `pub fn valueAt(Wave, Clause, f64) Error!f64`, `pub fn slopeAt(Wave, Clause, f64) Error!f64`
//! - `pub fn relError(Wave, Clause, core.MeasureFunc) Error!f64`
//! - `pub fn extremum(Wave, Clause, scan.Extremum.Op) Error!scan.Stat`
//! - `pub fn jitter(Wave, Clause) error{ NoSuchVector, OutOfInterval }!Jitter`
//! - `pub fn emAvg(Wave, Clause) Error!scan.Integral`, `pub fn rmsInteg(Wave, Clause, bool) Error!scan.Integral`

const std = @import("std");
const assert = std.debug.assert;
const core = @import("core");
const scan = @import("scan.zig");
const columns = @import("column.zig");
const Column = columns.Column;
const Waveform = columns.Waveform;
const Part = columns.Part;
const Clause = core.MeasureClause;
const Kind = core.query.Kind;
const Wave = @import("Wave.zig");

result: core.Result,
analysis: Kind,
/// Results of the cards evaluated so far, by card index.
values: []const f64 = &.{},
/// What the card found while its run went, read in place of `result`.
done: ?*Done = null,

/// A card's passes as its run found them (`core.Measured`), handed to the
/// evaluator in the order it reads them. Borrows `found`.
pub const Done = struct {
    found: []const core.Measured.Found,
    next: usize = 0,

    /// Returns the next pass's numbers; `NoSuchVector` for a pass that named
    /// a vector the run lacked. Asserts that a pass is left.
    pub fn take(d: *Done) Error![3]f64 {
        assert(d.next < d.found.len);
        const f = d.found[d.next];
        d.next += 1;
        assert(d.next <= d.found.len);
        return if (f.missing) error.NoSuchVector else f.v;
    }
};

/// A clause names a vector the result lacks (ngspice "no such vector").
pub const Error = error{NoSuchVector};

/// `jitter`'s figures: the RMS and peak-to-peak time interval error, the
/// recovered period and the events read.
pub const Jitter = struct { rms: f64, pp: f64, period: f64, edges: u32 };

/// `emAvg`'s pass.
pub const EmAvg = struct {
    c: Clause,
    pos: f64 = 0,
    neg: f64 = 0,
    from: f64 = nan,
    to: f64 = nan,
    pv: f64 = 0,

    /// Reads sample (`xv`, `v`); true at the first sample past a nonzero `to`.
    pub fn feed(s: *EmAvg, xv: f64, v: f64) bool {
        if (xv < s.c.from) return false;
        if (s.c.to != 0 and xv > s.c.to) return true;
        if (!std.math.isNan(s.from)) {
            s.pos += positiveArea(s.pv, v, xv - s.to);
            s.neg += positiveArea(-s.pv, -v, xv - s.to);
        } else s.from = xv;
        s.to = xv;
        s.pv = v;
        return false;
    }

    /// Returns the figure over the samples read; NaN over an empty window.
    pub fn result(s: EmAvg) scan.Integral {
        const span = s.to - s.from;
        if (!(span > 0)) return .{ .value = nan, .from = s.from, .to = s.to };
        return .{ .value = (@max(s.pos, s.neg) - s.c.val * @min(s.pos, s.neg)) / span, .from = s.from, .to = s.to };
    }
};

/// `relError`'s pass.
pub const RelError = struct {
    c: Clause,
    dc: bool,
    func: core.MeasureFunc,
    sum: f64 = 0,
    count: f64 = 0,

    /// Reads sample `xv` of the measured (`m`) and calculated (`calc`)
    /// vectors; true at the first sample past a nonzero `to` (not in DC).
    pub fn feed(s: *RelError, xv: f64, m: f64, calc: f64) bool {
        const c = s.c;
        if (s.dc) {
            if (xv < c.from or xv > c.to) return false;
        } else {
            if (xv < c.from) return false;
            if (c.to != 0 and xv > c.to) return true;
        }
        if (@abs(m) < c.ymin or @abs(m) > c.ymax) return false;
        const d = if (@abs(m) < c.minval) c.minval else m;
        // ponytail: the manual gives no reduction for ERR3; RMS, as ERR1.
        const e = if (s.func == .err3) std.math.sign(m / calc) * @log(@abs(m / calc)) / @log(d) else (m - calc) / d;
        s.sum += if (s.func == .err2) @abs(e) else e * e;
        s.count += 1;
        return false;
    }

    /// Returns the figure over the points read; NaN when none was.
    pub fn value(s: RelError) f64 {
        if (s.count == 0) return nan;
        return if (s.func == .err2) s.sum / s.count else @sqrt(s.sum / s.count);
    }
};

const nan = std.math.nan(f64);
/// Events `jitter` reads; later ones are ignored.
const jitter_events_max = 4096;

/// Returns the numbers of clause `c` a shared pass reads.
pub fn bounds(c: Clause) scan.Bounds {
    return .{ .from = c.from, .to = c.to, .val = c.val, .td = c.td, .rise = c.rise, .fall = c.fall, .cross = c.cross, .has_y2 = c.vec2.len > 0 };
}

/// Returns `analysis` as the shared passes tell scales apart.
pub fn modeOf(analysis: Kind) scan.Mode {
    return switch (analysis) {
        .tran, .tran_noise => .tran,
        .dc => .dc,
        .ac, .sp => .ac,
        else => .other, // else: no other scale has a pass-specific rule
    };
}

/// Returns the sample count; 0 for a result whose rows streamed to a file.
pub fn len(w: Wave) usize {
    return if (w.result.data.len == 0) 0 else w.result.npoints;
}

/// Returns column 0, the scale (time, frequency, sweep).
pub fn scale(w: Wave) Column {
    return w.col(0, .real);
}

/// Returns column `i` read as `part`. Asserts that the result has it.
pub fn col(w: Wave, i: usize, part: Part) Column {
    assert(i < w.result.varnames.len);
    const width: usize = if (w.result.is_complex) 2 else 1;
    return .{ .data = w.result.data, .stride = w.result.varnames.len * width, .offset = i * width, .complex = w.result.is_complex, .part = part };
}

/// Returns the column labelled `name` (`columnIndex`), read as `vectype`.
pub fn column(w: Wave, name: []const u8, vectype: u8) Error!Column {
    return w.col(columns.columnIndex(w.result.varnames, name) orelse return error.NoSuchVector, .fromVectype(vectype));
}

/// Returns the vector `name`, or the `par()` waveform `ops` when given,
/// each of its vectors resolved once.
pub fn waveform(w: Wave, name: []const u8, ops: []const core.MeasureOp, vectype: u8) Error!Waveform {
    if (ops.len == 0) return .{ .column = try w.column(name, vectype) };
    var p: columns.Par = .{ .base = w.col(0, .fromVectype(vectype)), .ops = ops, .names = w.result.varnames };
    for (ops) |op| if (op == .vector) {
        const k = columns.columnIndex(w.result.varnames, op.vector) orelse return error.NoSuchVector;
        if (p.cached < columns.Par.cache_len) {
            p.columns[p.cached] = @intCast(k);
            p.cached += 1;
        }
    };
    assert(p.cached <= columns.Par.cache_len);
    return .{ .par = p };
}

/// Returns the scale value of the event `c` names; NaN when it never
/// happens (ngspice com_measure_when).
pub fn when(w: Wave, c: Clause) Error!f64 {
    if (w.done) |d| return (try d.take())[0];
    var t: [1]f64 = .{nan};
    _ = try w.crossings(c, &t);
    return t[0];
}

/// Writes the events `c` names into `times`, in one scan: `times[k]` is
/// what `when` returns for `c` with its count raised by k. Returns how many
/// were found, at most `times.len`. A clause with no count or a LAST count
/// names one event. Asserts that `times` is not empty.
pub fn crossings(w: Wave, c: Clause, times: []f64) Error!usize {
    assert(times.len > 0);
    const x = w.scale();
    const y = try w.waveform(c.vec, c.ops, c.vectype);
    const y2: ?Waveform = if (c.vec2.len > 0) try w.waveform(c.vec2, c.ops2, c.vectype) else null;
    var s: scan.Cross = .init(bounds(c), modeOf(w.analysis));
    for (0..w.len()) |i| {
        if (s.feed(times, x.get(i), y.get(i), if (y2) |c2| c2.get(i) else nan)) break;
    }
    const n = s.finish(times);
    assert(n <= times.len);
    return n;
}

/// Returns the vector interpolated at scale value `at`; NaN outside the
/// sweep (ngspice measure_at).
pub fn valueAt(w: Wave, c: Clause, at: f64) Error!f64 {
    return w.segment(c, .{ .at = at, .dc = w.analysis == .dc });
}

/// Returns the slope of the vector against the scale at `at`, off the
/// sample pair around it; NaN outside the sweep.
pub fn slopeAt(w: Wave, c: Clause, at: f64) Error!f64 {
    return w.segment(c, .{ .at = at, .dc = w.analysis == .dc, .slope = true });
}

/// Returns HSPICE's ERR family over the window [SA Ch.11 "Error
/// Equations"]: with e = (M - C) / M, M = `c.vec` floored at `c.minval`,
/// C = `c.vec2`, ERR and ERR1 give the RMS of e and ERR2 the mean of |e|.
/// ERR3 takes e = +-log|M/C| / log(M), signed as M/C, reduced as ERR1.
/// Points with |M| outside [ymin, ymax] are skipped; NaN when none is left.
pub fn relError(w: Wave, c: Clause, func: core.MeasureFunc) Error!f64 {
    if (w.done) |d| return (try d.take())[0];
    const x = w.scale();
    const mv = try w.waveform(c.vec, c.ops, c.vectype);
    const cv = try w.column(c.vec2, c.vectype);
    var s: RelError = .{ .c = c, .dc = w.analysis == .dc, .func = func };
    for (0..w.len()) |i| if (s.feed(x.get(i), mv.get(i), cv.get(i))) break;
    return s.value();
}

/// Returns MIN/MAX (value, where) or AVG (trapezoidal mean, last scale
/// read) over the window, without interpolating its ends (ngspice
/// measure_minMaxAvg).
pub fn extremum(w: Wave, c: Clause, op: scan.Extremum.Op) Error!scan.Stat {
    if (w.done) |d| {
        const v = try d.take();
        return .{ .value = v[0], .at = v[1] };
    }
    const x = w.scale();
    const y = try w.waveform(c.vec, c.ops, c.vectype);
    var s: scan.Extremum = .{ .c = bounds(c), .dc = w.analysis == .dc, .op = op };
    for (0..w.len()) |i| if (s.feed(x.get(i), y.get(i))) break;
    return s.stat();
}

/// Returns the time interval error of the repeating event `c` names
/// (rising edges unless it counts falls or crossings) after `c.td`: each
/// event time minus the least-squares line through event time against
/// index, the ideal clock the edges recover. `OutOfInterval` below three
/// events. Reads the first `jitter_events_max` events.
pub fn jitter(w: Wave, c: Clause) error{ NoSuchVector, OutOfInterval }!Jitter {
    var times: [jitter_events_max]f64 = undefined;
    var e = c;
    if (c.fall > 0) e.fall = 1 else if (c.cross > 0) e.cross = 1 else e.rise = 1;
    const n = try w.crossings(e, &times);
    if (n < 3) return error.OutOfInterval;
    const count: f64 = @floatFromInt(n);
    var sx: f64 = 0;
    var st: f64 = 0;
    for (times[0..n], 0..) |t, i| {
        sx += @floatFromInt(i);
        st += t;
    }
    const mx = sx / count;
    const mt = st / count;
    var sxx: f64 = 0;
    var sxt: f64 = 0;
    for (times[0..n], 0..) |t, i| {
        const dx = @as(f64, @floatFromInt(i)) - mx;
        sxx += dx * dx;
        sxt += dx * (t - mt);
    }
    const period = sxt / sxx;
    var sum2: f64 = 0;
    var lo: f64 = std.math.inf(f64);
    var hi: f64 = -std.math.inf(f64);
    for (times[0..n], 0..) |t, i| {
        const tie = t - (mt + period * (@as(f64, @floatFromInt(i)) - mx));
        sum2 += tie * tie;
        lo = @min(lo, tie);
        hi = @max(hi, tie);
    }
    return .{ .rms = @sqrt(sum2 / count), .pp = hi - lo, .period = period, .edges = @intCast(n) };
}

/// Returns HSPICE EM_AVG over the window [CR .MEASURE (AVG, EM_AVG, ...)]:
/// with I+ and I- the trapezoidal averages of the positive and negative
/// parts (each segment split at its zero crossing), max(I+, I-) minus
/// `c.val` (`.option em_recovery`) times min(I+, I-). The window ends are
/// samples, as AVG's.
pub fn emAvg(w: Wave, c: Clause) Error!scan.Integral {
    if (w.done) |d| {
        const v = try d.take();
        return .{ .value = v[0], .from = v[1], .to = v[2] };
    }
    const x = w.scale();
    const y = try w.waveform(c.vec, c.ops, c.vectype);
    var s: EmAvg = .{ .c = c };
    for (0..w.len()) |i| if (s.feed(x.get(i), y.get(i))) break;
    return s.result();
}

/// Returns the integral, or with `rms` the RMS, of the vector over [from,
/// to], ends interpolated, by composite Simpson 3/8, Simpson 1/3 and
/// trapezoid panels as the sample spacing allows (ngspice
/// measure_rms_integral).
pub fn rmsInteg(w: Wave, c: Clause, rms: bool) Error!scan.Integral {
    if (w.done) |d| {
        const v = try d.take();
        return .{ .value = v[0], .from = v[1], .to = v[2] };
    }
    const x = w.scale();
    const y = try w.waveform(c.vec, c.ops, c.vectype);
    var s: scan.Integ = .{ .c = bounds(c), .rms = rms };
    for (0..w.len()) |i| if (s.feed(x.get(i), y.get(i))) break;
    return s.result();
}

// ---- private ----

fn segment(w: Wave, c: Clause, pass: scan.At) Error!f64 {
    if (w.done) |d| return (try d.take())[0];
    const x = w.scale();
    const y = try w.waveform(c.vec, c.ops, c.vectype);
    var s = pass;
    for (0..w.len()) |i| if (s.feed(x.get(i), y.get(i))) break;
    return s.value;
}

/// Area under max(v, 0) over one linear segment from `v0` to `v1`, `dx` wide.
fn positiveArea(v0: f64, v1: f64, dx: f64) f64 {
    if (v0 >= 0 and v1 >= 0) return 0.5 * (v0 + v1) * dx;
    if (v0 <= 0 and v1 <= 0) return 0;
    const hi = @max(v0, v1);
    return 0.5 * hi * dx * hi / (hi - @min(v0, v1));
}

test "one crossings scan finds what when finds for each raised count" {
    // A two-tone wave on uneven steps, crossing 0.2 and the second vector.
    var data: [3 * 400]f64 = undefined;
    var t: f64 = 0;
    for (0..400) |i| {
        data[3 * i ..][0..3].* = .{ t, @sin(t) + 0.5 * @sin(3.1 * t), 0.3 * @cos(0.7 * t) };
        t += 0.05 + 0.04 * @abs(@sin(11 * t));
    }
    const r: core.Result = .{ .plotname = "t", .varnames = &.{ "time", "v(a)", "v(b)" }, .is_complex = false, .npoints = 400, .data = &data };
    const w: Wave = .{ .result = r, .analysis = .tran };
    for ([_]Clause{
        .{ .vec = "v(a)", .val = 0.2, .rise = 1 },
        .{ .vec = "v(a)", .val = 0.2, .fall = 2 },
        .{ .vec = "v(a)", .val = 0.2, .cross = 1, .td = 3 },
        .{ .vec = "v(a)", .vec2 = "v(b)", .rise = 1, .from = 1, .to = 25 },
    }) |c| {
        var times: [64]f64 = undefined;
        const n = try w.crossings(c, &times);
        try std.testing.expect(n >= 3 and n < times.len);
        for (0..n + 1) |k| {
            var e = c;
            const raise: i32 = @intCast(k);
            if (c.rise > 0) e.rise += raise else if (c.fall > 0) e.fall += raise else e.cross += raise;
            const want = try w.when(e);
            if (k == n) try std.testing.expect(std.math.isNan(want)) else try std.testing.expectEqual(want, times[k]);
        }
    }
}

test "jitter: an ideal clock has none, an alternating one its residual" {
    // Low from 0, then rising through 0.5 at each edge time (a 0.1-wide ramp
    // centred on it) and falling back 1 later. An edge in the first interval
    // would be ngspice's primed count, read off the next segment.
    const cases = [_]struct { [4]f64, f64, f64, f64 }{
        .{ .{ 0.55, 2.55, 4.55, 6.55 }, 2, 0, 0 },
        // Least squares: period 10.1/5, residuals -0.02, 0.06, -0.06, 0.02.
        .{ .{ 0.55, 2.65, 4.55, 6.65 }, 2.02, @sqrt(0.002), 0.12 },
    };
    for (cases) |case| {
        var data: [2 + 2 * 4 * 4]f64 = @splat(0);
        for (case[0], 0..) |e, k| data[2 + 8 * k ..][0..8].* = .{ e - 0.05, 0, e + 0.05, 1, e + 0.95, 1, e + 1.05, 0 };
        const r: core.Result = .{ .plotname = "t", .varnames = &.{ "time", "v(c)" }, .is_complex = false, .npoints = 17, .data = &data };
        const j = try (Wave{ .result = r, .analysis = .tran }).jitter(.{ .vec = "v(c)", .val = 0.5 });
        try std.testing.expectEqual(4, j.edges);
        try std.testing.expectApproxEqAbs(case[1], j.period, 1e-12);
        try std.testing.expectApproxEqAbs(case[2], j.rms, 1e-12);
        try std.testing.expectApproxEqAbs(case[3], j.pp, 1e-12);
    }
}
