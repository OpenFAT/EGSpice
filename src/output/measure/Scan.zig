//! One pass of a `Stream` card: the stream's current row in, the pass's
//! state advanced out, and once it reads no more rows, what it found as the
//! `Wave` function it stands for returns it (`core.Measured.Found`).
//! Cites: none beyond scan.zig and Wave.zig, whose passes it runs.
//!
//! Contents:
//! - fields `y`, `y2`, `missing`, `over`, `time`, `pass`
//! - `pub const Pass = union(enum) { cross, at, extremum, integ, em_avg, rel_error }`
//! - `pub const Second = enum { none, waveform, column }`
//! - `pub fn init(w: Wave, c: Clause, pass: Pass, second: Second) Scan`
//! - `pub fn feed(s: *Scan, x: f64) void`
//! - `pub fn feedValues(s: *Scan, x: f64, y: f64, y2: f64) void`
//! - `pub fn found(s: *Scan) core.Measured.Found`

const std = @import("std");
const assert = std.debug.assert;
const core = @import("core");
const scan = @import("scan.zig");
const columns = @import("column.zig");
const Wave = @import("Wave.zig");
const Clause = core.MeasureClause;
const Scan = @import("Scan.zig");

/// The vector it reads, and the level or comparison vector when it has one.
/// `y` is undefined when `missing`.
y: columns.Waveform = undefined,
y2: ?columns.Waveform = null,
/// It names a vector the run lacks, which the evaluator fails it for.
missing: bool = false,
/// No later row changes what it found.
over: bool = false,
time: [1]f64 = .{nan},
pass: Pass,

/// The pass a card runs, as Wave.zig runs it over a whole recording.
pub const Pass = union(enum) { cross: scan.Cross, at: scan.At, extremum: scan.Extremum, integ: scan.Integ, em_avg: Wave.EmAvg, rel_error: Wave.RelError };

/// The second vector a pass reads: none, a level waveform (`WHEN vec=vec2`),
/// or ERR's comparison column.
pub const Second = enum { none, waveform, column };

const nan = std.math.nan(f64);

/// Returns `pass` over clause `c`'s vector on `w`, with its `second`
/// vector; a pass naming a vector `w` lacks is `missing` and `over` at once.
pub fn init(w: Wave, c: Clause, pass: Pass, second: Second) Scan {
    const missing: Scan = .{ .missing = true, .over = true, .pass = pass };
    var s: Scan = .{ .y = w.waveform(c.vec, c.ops, c.vectype) catch return missing, .pass = pass };
    switch (second) {
        .none => {},
        .waveform => if (c.vec2.len > 0) {
            s.y2 = w.waveform(c.vec2, c.ops2, c.vectype) catch return missing;
        },
        .column => s.y2 = .{ .column = w.column(c.vec2, c.vectype) catch return missing },
    }
    assert(!s.missing);
    return s;
}

/// Reads the row its columns point at, scale `x`, unless it is over.
pub fn feed(s: *Scan, x: f64) void {
    if (s.over) return;
    s.feedValues(x, s.y.get(0), if (s.y2) |c| c.get(0) else nan);
}

/// Reads sample (`x`, `y`, `y2`), `y2` NaN without a second vector, and
/// sets `over` where the pass stops. Asserts that it is not over.
pub fn feedValues(s: *Scan, x: f64, y: f64, y2: f64) void {
    assert(!s.over);
    assert(!s.missing);
    s.over = switch (s.pass) {
        .cross => |*c| c.feed(&s.time, x, y, y2),
        // A transient's times rise: past AT=, no later segment holds it.
        .at => |*a| a.feed(x, y) or x > a.at,
        .extremum => |*e| e.feed(x, y),
        .integ => |*g| g.feed(x, y),
        .em_avg => |*e| e.feed(x, y),
        .rel_error => |*r| r.feed(x, y, y2),
    };
}

/// Returns what it found over the rows read, as the `Wave` function its
/// pass stands for returns it. Call only once it reads no more rows: an
/// integral folds its pending panels.
pub fn found(s: *Scan) core.Measured.Found {
    if (s.missing) return .{ .missing = true, .v = @splat(nan) };
    return .{ .v = switch (s.pass) {
        .cross => |c| blk: {
            _ = c.finish(&s.time);
            break :blk .{ s.time[0], 0, 0 };
        },
        .at => |a| .{ a.value, 0, 0 },
        .extremum => |e| blk: {
            const r = e.stat();
            break :blk .{ r.value, r.at, 0 };
        },
        .integ => |*g| blk: {
            const r = g.result();
            break :blk .{ r.value, r.from, r.to };
        },
        .em_avg => |e| blk: {
            const r = e.result();
            break :blk .{ r.value, r.from, r.to };
        },
        .rel_error => |r| .{ r.value(), 0, 0 },
    } };
}
