//! Violation counting for the HSPICE check forms: a waveform, its reference
//! edges and a `core.Check` in, one listing line per violation and their
//! count out. Transitions are read between the logic thresholds: a rise
//! leaves `lo` and reaches `hi` without falling back, and happens at the
//! midpoint of those two crossings.
//! Cites: HSPICE [CR .CHECK SLEW, SETUP/HOLD/EDGE, IRDROP/LEVEL; .DOUT;
//! .BIASCHK].
//!
//! Contents:
//! - `pub const Error = Writer.Error || Wave.Error`
//! - `pub fn violations(w: Wave, out: *Writer, name: []const u8, c: Clause, ref: Clause, func: core.MeasureFunc, k: core.Check) Error!u32`

const std = @import("std");
const assert = std.debug.assert;
const Writer = std.Io.Writer;
const core = @import("core");
const columns = @import("column.zig");
const sci = @import("sci.zig").sci;
const Wave = @import("Wave.zig");
const Clause = core.MeasureClause;

/// A listing write failed, or a clause names a missing vector.
pub const Error = Writer.Error || Wave.Error;

/// Logic transitions of a waveform between thresholds `lo` and `hi`.
const Edges = struct {
    x: columns.Column,
    y: columns.Waveform,
    n: usize,
    lo: f64,
    hi: f64,
    i: usize = 1,
    state: enum { low, high, rising, falling, unknown } = .unknown,
    start: f64 = 0,

    const Edge = struct {
        rise: bool,
        start: f64,
        end: f64,

        fn at(e: Edge) f64 {
            return 0.5 * (e.start + e.end);
        }

        fn is(e: Edge, want: core.Check.Edge) bool {
            return want == .both or e.rise == (want == .rise);
        }
    };

    fn next(e: *Edges) ?Edge {
        if (e.i == 1 and e.n > 0) {
            const y0 = e.y.get(0);
            e.state = if (y0 <= e.lo) .low else if (y0 >= e.hi) .high else .unknown;
        }
        while (e.i < e.n) {
            const x0 = e.x.get(e.i - 1);
            const x1 = e.x.get(e.i);
            const y0 = e.y.get(e.i - 1);
            const y1 = e.y.get(e.i);
            e.i += 1;
            if (e.state == .low and y1 > e.lo) {
                e.state = .rising;
                e.start = crossing(x0, y0, x1, y1, e.lo);
            } else if (e.state == .high and y1 < e.hi) {
                e.state = .falling;
                e.start = crossing(x0, y0, x1, y1, e.hi);
            }
            switch (e.state) {
                .rising => if (y1 >= e.hi) {
                    e.state = .high;
                    return .{ .rise = true, .start = e.start, .end = crossing(x0, y0, x1, y1, e.hi) };
                } else if (y1 <= e.lo) {
                    e.state = .low;
                },
                .falling => if (y1 <= e.lo) {
                    e.state = .low;
                    return .{ .rise = false, .start = e.start, .end = crossing(x0, y0, x1, y1, e.lo) };
                } else if (y1 >= e.hi) {
                    e.state = .high;
                },
                .unknown => e.state = if (y1 <= e.lo) .low else if (y1 >= e.hi) .high else .unknown,
                .low, .high => {},
            }
        }
        assert(e.i >= e.n or e.n == 0);
        return null;
    }
};

/// Writes one line per violation of check `k` on `c`'s waveform (`ref`'s
/// is the reference edge) to `out` and returns their count. Asserts that
/// `func` is a check function or DOUT.
// ponytail: SETUP/HOLD/EDGE rescan the node per reference edge,
// O(edges x samples); merge the two edge streams if decks get long.
pub fn violations(w: Wave, out: *Writer, name: []const u8, c: Clause, ref: Clause, func: core.MeasureFunc, k: core.Check) Error!u32 {
    const x = w.scale();
    const y = try w.waveform(c.vec, c.ops, c.vectype);
    const node: Edges = .{ .x = x, .y = y, .n = w.len(), .lo = k.lo, .hi = k.hi };
    var n: u32 = 0;
    switch (func) {
        .check_slew => {
            var it = node;
            while (it.next()) |e| if (e.is(k.edge)) {
                const d = e.end - e.start;
                if (d >= k.min and d <= k.max) continue;
                n += 1;
                try out.print("  {s} violation: {s} at {f} took {f}\n", .{ name, if (e.rise) "rise" else "fall", sci(e.at(), 6), sci(d, 6) });
            };
        },
        .check_forbid, .check_require => {
            var refs: Edges = .{ .x = x, .y = try w.waveform(ref.vec, ref.ops, ref.vectype), .n = w.len(), .lo = k.lo, .hi = k.hi };
            while (refs.next()) |r| if (r.is(k.ref_edge)) {
                var hits: u32 = 0;
                var it = node;
                while (it.next()) |e| if (e.is(k.edge) and e.at() >= r.at() + k.min and e.at() <= r.at() + k.max) {
                    hits += 1;
                    if (func == .check_forbid) try out.print("  {s} violation: {s} at {f}, reference edge at {f}\n", .{ name, if (e.rise) "rise" else "fall", sci(e.at(), 6), sci(r.at(), 6) });
                };
                if (func == .check_forbid) n += hits else if (hits == 0) {
                    n += 1;
                    try out.print("  {s} violation: no edge after reference edge at {f}\n", .{ name, sci(r.at(), 6) });
                }
            };
        },
        .check_level => n = try levelViolations(w, out, name, c, k, x, y),
        .dout => for (k.expect) |e| {
            const v = try w.valueAt(c, e[0]);
            const one = if (k.lo == k.hi) v > k.hi else v >= k.hi;
            const zero = if (k.lo == k.hi) !one else v <= k.lo;
            if (if (e[1] == 1) one else zero) continue;
            n += 1;
            try out.print("  {s} violation: expected {d} at {f}, got {f}\n", .{ name, @as(u8, @intFromFloat(e[1])), sci(e[0], 6), sci(v, 6) });
        },
        else => unreachable, // else: `violations` is called with the check functions only
    }
    return n;
}

// ---- private ----

/// `check_level`: excursions out of [min, max] inside the window lasting
/// past `dur` (every one when `dur` is 0).
fn levelViolations(w: Wave, out: *Writer, name: []const u8, c: Clause, k: core.Check, x: columns.Column, y: columns.Waveform) Error!u32 {
    var n: u32 = 0;
    var entered: ?f64 = null;
    for (0..w.len()) |i| {
        const v = y.get(i);
        const live = x.get(i) >= c.from and (c.to == 0 or x.get(i) <= c.to);
        const outside = live and (v < k.min or v > k.max);
        if (outside and entered == null) {
            entered = if (i == 0) x.get(0) else crossing(x.get(i - 1), y.get(i - 1), x.get(i), v, if (v > k.max) k.max else k.min);
        } else if (!outside and entered != null) {
            const left = crossing(x.get(i - 1), y.get(i - 1), x.get(i), v, if (y.get(i - 1) > k.max) k.max else k.min);
            if (left - entered.? > k.dur or k.dur == 0) {
                n += 1;
                try out.print("  {s} violation: from {f} to {f}\n", .{ name, sci(entered.?, 6), sci(left, 6) });
            }
            entered = null;
        }
    }
    if (entered) |t| if (x.get(w.len() - 1) - t > k.dur or k.dur == 0) {
        n += 1;
        try out.print("  {s} violation: from {f} to the end\n", .{ name, sci(t, 6) });
    };
    return n;
}

/// Where the segment (x0, y0)-(x1, y1) meets `level`.
fn crossing(x0: f64, y0: f64, x1: f64, y1: f64, level: f64) f64 {
    return if (y1 == y0) x1 else x0 + (level - y0) * (x1 - x0) / (y1 - y0);
}

test "a slow edge, a forbidden edge and a level excursion each count once" {
    // Oracle, by hand: v(d) ramps 0, 0.5, 1 over t = 1, 2, 3 (crossing 0.2
    // at 1.4 and 0.8 at 2.6: midpoint 2, 1.2 long), then holds; v(r) ramps
    // 0 -> 1 over [1, 2].
    const data = [_]f64{ 0, 0, 0, 1, 0, 0, 2, 0.5, 1, 3, 1, 1, 4, 1, 1 };
    const r: core.Result = .{ .plotname = "t", .varnames = &.{ "time", "v(d)", "v(r)" }, .is_complex = false, .npoints = 5, .data = &data };
    const w: Wave = .{ .result = r, .analysis = .tran };
    var buf: [512]u8 = undefined;
    var out: Writer = .fixed(&buf);
    const d: Clause = .{ .vec = "v(d)" };
    const check: core.Check = .{ .lo = 0.2, .hi = 0.8, .min = 0, .max = 1 };
    try std.testing.expectEqual(1, try violations(w, &out, "s", d, .{}, .check_slew, check));
    try std.testing.expectEqualStrings("  s violation: rise at 2.000000e+00 took 1.200000e+00\n", out.buffered());
    // The reference rises at its own midpoint 1.5 (0.2 at 1.2, 0.8 at 1.8);
    // d's edge at 2 lands in [1.5, 2.5].
    var forbid = check;
    forbid.max = 1;
    out = .fixed(&buf);
    try std.testing.expectEqual(1, try violations(w, &out, "h", d, .{ .vec = "v(r)" }, .check_forbid, forbid));
    // v(d) leaves [0, 0.75] at 2.5 and never returns.
    out = .fixed(&buf);
    try std.testing.expectEqual(1, try violations(w, &out, "l", d, .{}, .check_level, .{ .min = 0, .max = 0.75 }));
    try std.testing.expectEqualStrings("  l violation: from 2.500000e+00 to the end\n", out.buffered());
}
