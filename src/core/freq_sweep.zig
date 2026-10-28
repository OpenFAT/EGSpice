//! The frequency grid every frequency-domain analysis runs on: a sweep card's
//! spelling (kind, points, ends, or a POI list) in, the grid's point count
//! and frequencies out, point by point (`at`), in bulk (`fill`) or as an
//! iterator. Pure arithmetic on a value; nothing allocates.
//! Cites: ngspice acan.c:89-99 (the `.ac` stretched ratio), acan.c:239 (the
//! stop tolerance).
//!
//! Contents:
//! - `pub const SweepKind = enum { dec, oct, lin, poi }`
//! - `pub const FreqSweep = struct`
//! - `FreqSweep.count(self: FreqSweep) u32`
//! - `FreqSweep.at(self: FreqSweep, k: u32) f64`
//! - `FreqSweep.fill(self: FreqSweep, freqs: ?[]f64, omegas: []f64) void`
//! - `FreqSweep.iter(self: FreqSweep) Iter`, `Iter.next(self: *Iter) ?f64`

const std = @import("std");
const assert = std.debug.assert;

/// SPICE's `.ac`/`.noise`/`.sp` grid spellings, plus HSPICE's POI point list.
pub const SweepKind = enum { dec, oct, lin, poi };

/// A sweep card's grid. `count` and `at` define it; every other method is
/// a view of those two.
pub const FreqSweep = struct {
    /// Hz.
    f_start: f64,
    /// Hz. A geometric grid may stop short of it (see `count`).
    f_stop: f64,
    /// Points per decade (`dec`), per octave (`oct`), or in total (`lin`,
    /// `poi`).
    points: u32 = 10,
    kind: SweepKind = .dec,
    /// The `poi` frequencies in Hz, ascending, `points` of them; `f_start`
    /// and `f_stop` are its ends. Unused by the other kinds. Borrowed: it
    /// must outlive the sweep.
    list: []const f64 = &.{},
    /// ngspice `.ac` (acan.c:89-99): a `dec` grid spanning at least a
    /// decade stretches its ratio so the last point lands on `f_stop`
    /// (`.ac dec 3 10 730` is 6 points ending at 730). ngspice's noise and
    /// sp grids keep the plain ratio, so only the `.ac` card sets it.
    land_on_stop: bool = false,

    /// Yields the grid frequencies in order, in Hz.
    pub const Iter = struct {
        sweep: FreqSweep,
        n: u32,
        k: u32 = 0,

        /// Null once all `n` points are out.
        pub fn next(self: *Iter) ?f64 {
            assert(self.k <= self.n);
            if (self.k == self.n) return null;
            defer self.k += 1;
            return self.sweep.at(self.k);
        }
    };

    /// Number of grid points, at least 1 and at most maxInt(u32). A grid it
    /// cannot step (a non-positive, NaN or infinite span) is one point.
    /// A geometric grid steps by a fixed ratio and keeps every point up to
    /// `f_stop` plus ngspice's stop tolerance (acan.c:239): `dec 10 1 9.999`
    /// is 11 points ending at 10.
    pub fn count(self: FreqSweep) u32 {
        switch (self.kind) {
            .poi => return @max(self.points, 1),
            // A `.dc` lin axis may descend; only an empty span is one point.
            .lin => return if (self.f_stop != self.f_start) @max(self.points, 1) else 1,
            .dec, .oct => {},
        }
        if (!(self.f_start > 0) or !(self.f_stop >= self.f_start) or self.points == 0) return 1;
        const ln_step = @log(self.base()) / @as(f64, @floatFromInt(self.points));
        // The ratio keeps ngspice's rounding; a span whose ratio overflows
        // (1e-300 to 1e300) falls back to the log difference.
        const ratio = self.f_stop / self.f_start;
        const ln_span = if (std.math.isFinite(ratio)) @log(ratio) else @log(self.f_stop) - @log(self.f_start);
        // ngspice runs while freq <= f_stop + step * f_stop * reltol, at the
        // default reltol. ponytail: the deck's reltol is not threaded here;
        // it moves only a point within 0.1% of f_stop.
        const steps = self.stretchedSteps() orelse (ln_span + std.math.log1p(@exp(ln_step) * 1e-3)) / ln_step;
        if (!std.math.isFinite(steps) or steps < 0) return 1;
        // The cap keeps an absurd span (`dec 4294967295 1e-300 1e300`)
        // inside u32 instead of trapping.
        const capped = @min(@floor(steps), @as(f64, std.math.maxInt(u32) - 1));
        assert(capped >= 0);
        const n = @as(u32, @intFromFloat(capped)) + 1;
        assert(n >= 1);
        return n;
    }

    /// Frequency of point `k`, in Hz. Asserts that `k < count()`; for `poi`,
    /// that `list` holds point `k` instead.
    pub fn at(self: FreqSweep, k: u32) f64 {
        if (self.kind == .poi) {
            assert(k < self.list.len);
            return self.list[k];
        }
        const n = self.count();
        assert(k < n);
        const kf: f64 = @floatFromInt(k);
        if (self.kind == .lin) {
            if (n == 1) return self.f_start;
            const frac = kf / @as(f64, @floatFromInt(n - 1));
            return self.f_start + frac * (self.f_stop - self.f_start);
        }
        // pow, not a running product: the product drifts off the decade
        // boundaries the oracles are written on.
        if (self.stretchedSteps()) |steps| return self.f_start * std.math.pow(f64, self.f_stop / self.f_start, kf / steps);
        return self.f_start * std.math.pow(f64, self.base(), kf / @as(f64, @floatFromInt(@max(self.points, 1))));
    }

    /// Writes each grid frequency (Hz) into `freqs`, when given, and its
    /// angular frequency (rad/s) into `omegas`. Asserts that each holds at
    /// least `count()` entries.
    pub fn fill(self: FreqSweep, freqs: ?[]f64, omegas: []f64) void {
        const n = self.count();
        assert(omegas.len >= n);
        if (freqs) |fr| assert(fr.len >= n);
        for (0..n) |i| {
            const f = self.at(@intCast(i));
            if (freqs) |fr| fr[i] = f;
            omegas[i] = 2.0 * std.math.pi * f;
        }
    }

    /// The grid as an iterator, `count()` points in the order of `at`.
    pub fn iter(self: FreqSweep) Iter {
        return .{ .sweep = self, .n = self.count() };
    }

    // ---- private ----

    /// ngspice's stretched step count, floor(log10(stop/start) * points),
    /// when `land_on_stop` applies; null for a plain grid.
    fn stretchedSteps(self: FreqSweep) ?f64 {
        if (!self.land_on_stop or self.kind != .dec or !(self.f_stop / 10.0 >= self.f_start)) return null;
        const steps = @floor(std.math.log10(self.f_stop / self.f_start) * @as(f64, @floatFromInt(self.points)));
        return if (steps >= 1 and std.math.isFinite(steps)) steps else null;
    }

    fn base(self: FreqSweep) f64 {
        assert(self.kind == .dec or self.kind == .oct);
        return if (self.kind == .oct) 2.0 else 10.0;
    }
};

test "FreqSweep matches the ngspice grids the oracles were taken on" {
    const t = std.testing;
    // ngspice 45 `.ac`: the dec ratio stretches to land on f_stop.
    const dec: FreqSweep = .{ .f_start = 10, .f_stop = 730, .points = 3, .kind = .dec, .land_on_stop = true };
    try t.expectEqual(@as(u32, 6), dec.count());
    try t.expectApproxEqRel(@as(f64, 23.58656), dec.at(1), 1e-6);
    try t.expectApproxEqRel(@as(f64, 730), dec.at(5), 1e-12);
    const short: FreqSweep = .{ .f_start = 1, .f_stop = 99.9, .points = 2, .kind = .dec, .land_on_stop = true };
    try t.expectEqual(@as(u32, 4), short.count());
    try t.expectApproxEqRel(@as(f64, 4.640041), short.at(1), 1e-6);
    // Under a decade, and on the noise/sp grids, the ratio stays 10^(1/N);
    // the stop tolerance keeps 10 past 9.999.
    inline for (.{ true, false }) |land| {
        const under: FreqSweep = .{ .f_start = 1, .f_stop = 9.999, .points = 10, .kind = .dec, .land_on_stop = land };
        try t.expectEqual(@as(u32, 11), under.count());
        try t.expectApproxEqRel(@as(f64, 10), under.at(10), 1e-12);
    }
    const plain: FreqSweep = .{ .f_start = 10, .f_stop = 730, .points = 3, .kind = .dec };
    try t.expectEqual(@as(u32, 6), plain.count());
    try t.expectApproxEqRel(@as(f64, 464.15888336128), plain.at(5), 1e-12);
    const oct3: FreqSweep = .{ .f_start = 10, .f_stop = 100, .points = 3, .kind = .oct };
    try t.expectEqual(@as(u32, 10), oct3.count());
    try t.expectApproxEqRel(@as(f64, 80), oct3.at(9), 1e-12);
    const flat: FreqSweep = .{ .f_start = 100, .f_stop = 100, .points = 5, .kind = .lin };
    try t.expectEqual(@as(u32, 1), flat.count());
    const oct: FreqSweep = .{ .f_start = 10, .f_stop = 1280, .points = 2, .kind = .oct };
    try t.expectEqual(@as(u32, 15), oct.count());
    try t.expectApproxEqRel(@as(f64, 1280), oct.at(14), 1e-12);
    const lin: FreqSweep = .{ .f_start = 0, .f_stop = 1000, .points = 9, .kind = .lin };
    try t.expectEqual(@as(u32, 9), lin.count());
    try t.expectEqual(@as(f64, 125), lin.at(1));
    const one: FreqSweep = .{ .f_start = 100, .f_stop = 100, .points = 1, .kind = .lin };
    try t.expectEqual(@as(u32, 1), one.count());
    try t.expectEqual(@as(f64, 100), one.at(0));

    // The iterator walks the same grid and keeps both endpoints.
    const grid: FreqSweep = .{ .f_start = 10, .f_stop = 1000, .points = 2 };
    var sweep = grid.iter();
    const expected = [_]f64{ 10, @sqrt(1000.0), 100, @sqrt(100000.0), 1000 };
    try t.expectEqual(expected.len, sweep.n);
    for (expected) |frequency| try t.expectApproxEqRel(frequency, sweep.next().?, 1e-12);
    try t.expect(sweep.next() == null);
    const point: FreqSweep = .{ .f_start = 7, .f_stop = 7, .points = 10 };
    var single = point.iter();
    try t.expectApproxEqRel(@as(f64, 7), single.next().?, 1e-12);
    try t.expect(single.next() == null);
}

test "FreqSweep is one point on a span it cannot step, and caps an absurd one" {
    const t = std.testing;
    const nan = std.math.nan(f64);
    const inf = std.math.inf(f64);
    for ([_][2]f64{ .{ 0, 10 }, .{ -1, 10 }, .{ 10, 1 }, .{ nan, 10 }, .{ 10, nan }, .{ 10, inf } }) |span| {
        try t.expectEqual(@as(u32, 1), (FreqSweep{ .f_start = span[0], .f_stop = span[1] }).count());
        try t.expectEqual(@as(u32, 1), (FreqSweep{ .f_start = span[0], .f_stop = span[1], .kind = .oct }).count());
    }
    // No points per decade is one point, not a division by zero.
    try t.expectEqual(@as(u32, 1), (FreqSweep{ .f_start = 1, .f_stop = 1e3, .points = 0 }).count());
    try t.expectEqual(@as(f64, 1), (FreqSweep{ .f_start = 1, .f_stop = 1e3, .points = 0 }).at(0));
    try t.expectEqual(@as(u32, 1), (FreqSweep{ .f_start = 1, .f_stop = 1e3, .points = 0, .kind = .lin }).count());
    const huge: FreqSweep = .{ .f_start = 1e-300, .f_stop = 1e300, .points = std.math.maxInt(u32) };
    try t.expectEqual(@as(u32, std.math.maxInt(u32)), huge.count());
    // A descending linear grid walks down to its stop.
    const down: FreqSweep = .{ .f_start = 100, .f_stop = 0, .points = 5, .kind = .lin };
    try t.expectEqual(@as(f64, 75), down.at(1));
    try t.expectEqual(@as(f64, 0), down.at(4));
}

test "FreqSweep.fill writes the poi list and its angular frequencies" {
    const t = std.testing;
    const list = [_]f64{ 1, 20, 300 };
    const poi: FreqSweep = .{ .f_start = 1, .f_stop = 300, .points = 3, .kind = .poi, .list = &list };
    var hz: [3]f64 = undefined;
    var rad: [3]f64 = undefined;
    poi.fill(&hz, &rad);
    try t.expectEqualSlices(f64, &list, &hz);
    for (list, rad) |f, w| try t.expectEqual(2 * std.math.pi * f, w);
    // Without a frequency buffer only the omegas are written.
    var rad_only: [3]f64 = undefined;
    poi.fill(null, &rad_only);
    try t.expectEqualSlices(f64, &rad, &rad_only);
}

test "FreqSweep.fill and the iterator write exactly at(k), and a geometric grid ascends" {
    const t = std.testing;
    const grids = [_]FreqSweep{
        .{ .f_start = 1, .f_stop = 1e6, .points = 7 },
        .{ .f_start = 10, .f_stop = 730, .points = 3, .land_on_stop = true },
        .{ .f_start = 3, .f_stop = 1e3, .points = 5, .kind = .oct },
        .{ .f_start = 0, .f_stop = 5, .points = 6, .kind = .lin },
    };
    var hz: [64]f64 = undefined;
    var rad: [64]f64 = undefined;
    for (grids) |g| {
        const n = g.count();
        g.fill(hz[0..n], rad[0..n]);
        var it = g.iter();
        for (0..n) |k| {
            try t.expectEqual(g.at(@intCast(k)), hz[k]);
            try t.expectEqual(2 * std.math.pi * hz[k], rad[k]);
            try t.expectEqual(hz[k], it.next().?);
            if (k > 0) try t.expect(hz[k] > hz[k - 1]);
        }
        try t.expectEqual(null, it.next());
        // Every point stays within the stop tolerance of f_stop.
        try t.expect(hz[n - 1] <= g.f_stop * (1 + 1e-2));
    }
}
