//! The `.meas` passes that read a waveform one sample at a time, in order
//! (ngspice com_measure2.c): event counting, value at a point, extremum and
//! average, and the integral. `feed` returns true where the whole-waveform
//! loop would stop. `measure.zig`'s `Wave` runs them over a recording, its
//! `Stream` over a transient as it goes, and the GPU lanes on the device
//! (`solver/lane_device.zig`, performance TODO 06 6.8), so all three give
//! the same values, to the bit. std only, extern layouts: the device keeps
//! the states in lane-major arrays the host reads back.

const std = @import("std");
const nan = std.math.nan(f64);

/// `core.measure_unset`, `core.measure_last` (measure.zig asserts them).
pub const unset: i32 = -1;
pub const last: i32 = -2;

/// The numbers a pass reads off its `core.MeasureClause`.
pub const Bounds = extern struct {
    from: f64 = 0,
    to: f64 = 0,
    val: f64 = 1e99,
    td: f64 = 0,
    rise: i32 = unset,
    fall: i32 = unset,
    cross: i32 = unset,
    /// A level vector (`WHEN vec=vec2`) instead of `val`.
    has_y2: bool = false,
};

/// The analysis the scale belongs to, as the passes tell them apart.
pub const Mode = enum(u8) { tran, dc, ac, other };

pub const Stat = extern struct { value: f64, at: f64 };
pub const Integral = extern struct { value: f64, from: f64, to: f64 };

/// The line through (x0, y0) and (x1, y1) at `at`, in slope-intercept form
/// (ngspice measure_interpolate, 'y' mode).
pub fn line(x0: f64, y0: f64, x1: f64, y1: f64, at: f64) f64 {
    const slope = (y1 - y0) / (x1 - x0);
    const intercept = y0 - slope * x0;
    return slope * at + intercept;
}

/// ngspice AlmostEqualUlps(a, b, 100) (maths/misc/equality.c).
pub fn almostEqualUlps(a: f64, b: f64) bool {
    if (a == b) return true;
    const ordered = struct {
        fn f(d: f64) i64 {
            const i: i64 = @bitCast(d);
            return if (i < 0) std.math.minInt(i64) -% i else i;
        }
    }.f;
    return @abs(ordered(a) -% ordered(b)) <= 100;
}

/// `Wave.crossings`' event count (ngspice com_measure_when).
pub const Cross = extern struct {
    c: Bounds,
    mode: Mode,
    td: f64,
    /// The counts the next event needs.
    want: Bounds,
    rise: i32 = 0,
    fall: i32 = 0,
    cross: i32 = 0,
    above: bool = false,
    pending: bool = false,
    last: f64 = nan,
    first: u32 = 0,
    prev: f64 = 0,
    prev2: f64 = 0,
    prev_x: f64 = 0,
    rows: u64 = 0,
    /// Events found, and whether the scan stopped on one (LAST is not added then).
    n: u64 = 0,
    ended: bool = false,

    pub fn init(c: Bounds, mode: Mode) Cross {
        return .{ .c = c, .mode = mode, .td = c.td, .want = c };
    }

    /// Reads sample (`xv`, `v`), `v2` being the level vector's (NaN
    /// without one), and adds the events it ends to `times`.
    pub fn feed(s: *Cross, times: []f64, xv: f64, v: f64, v2: f64) bool {
        const c = s.c;
        const dc = s.mode == .dc;
        const i = s.rows;
        s.rows += 1;
        // A DC sweep may start anywhere: ngspice keeps its origin in td.
        if (dc and i == 0) s.td = xv;
        if (s.mode == .tran and xv < s.td) return false;
        if (s.mode == .ac and xv < 0) return false;
        if (dc) {
            if (xv < c.from or xv > c.to) return false;
        } else {
            if (xv < c.from) return false;
            if (c.to != 0 and xv > c.to) return true;
        }
        if (s.first > 1 and dc and s.td == xv) s.first = 1;
        const level = if (c.has_y2) v2 else c.val;
        const prev_level = if (c.has_y2) s.prev2 else c.val;
        if (s.first == 1) {
            s.cross = 0;
            s.above = !(v < level);
            if (!s.above and s.prev >= prev_level) {
                s.fall = 1;
                s.cross = 1;
            } else if (s.above and s.prev < prev_level) {
                s.rise = 1;
                s.cross = 1;
            }
        }
        if (s.first > 1) {
            if (!s.above and v >= level) {
                s.above = true;
                s.cross += 1;
                s.rise += 1;
                if (c.fall != last) s.pending = true;
            } else if (s.above and v <= level) {
                s.above = false;
                s.cross += 1;
                s.fall += 1;
                if (c.rise != last) s.pending = true;
            }
            const hit = s.prev_x + (prev_level - s.prev) * (xv - s.prev_x) / (v - s.prev - level + prev_level);
            // Counters move by at most one per sample, so the event the
            // raised count names comes on a later sample.
            if (s.cross == s.want.cross or s.rise == s.want.rise or s.fall == s.want.fall) {
                times[@intCast(s.n)] = hit;
                s.n += 1;
                if (s.n == times.len) {
                    s.ended = true;
                    return true;
                }
                if (s.cross == s.want.cross) s.want.cross += 1;
                if (s.rise == s.want.rise) s.want.rise += 1;
                if (s.fall == s.want.fall) s.want.fall += 1;
            }
            if (s.pending) {
                if (c.cross == unset and c.rise == unset and c.fall == unset) {
                    times[@intCast(s.n)] = hit;
                    s.n += 1;
                    s.ended = true;
                    return true;
                }
                if (c.cross == last or c.rise == last or c.fall == last) s.last = hit;
                s.pending = false;
            }
        }
        s.first += 1;
        s.prev = v;
        if (c.has_y2) s.prev2 = v2;
        s.prev_x = xv;
        return false;
    }

    /// How many events `times` holds, a LAST count's appended; once the
    /// scan has stopped or read every sample.
    pub fn finish(s: Cross, times: []f64) usize {
        if (s.ended or std.math.isNan(s.last)) return @intCast(s.n);
        times[@intCast(s.n)] = s.last;
        return @as(usize, @intCast(s.n)) + 1;
    }
};

/// `Wave.valueAt` and `slopeAt`: the first segment holding `at`.
pub const At = extern struct {
    at: f64,
    dc: bool = false,
    slope: bool = false,
    rows: u64 = 0,
    px: f64 = 0,
    pv: f64 = 0,
    value: f64 = nan,

    pub fn feed(s: *At, xv: f64, v: f64) bool {
        if (s.rows > 0 and ((s.px <= s.at and xv >= s.at) or (s.dc and s.px >= s.at and xv <= s.at))) {
            s.value = if (s.slope) (v - s.pv) / (xv - s.px) else s.pv + (s.at - s.px) * (v - s.pv) / (xv - s.px);
            return true;
        }
        s.rows += 1;
        s.px = xv;
        s.pv = v;
        return false;
    }
};

/// `Wave.extremum`.
pub const Extremum = extern struct {
    c: Bounds,
    dc: bool,
    op: Op,
    first: bool = false,
    m: f64 = 0,
    m_at: f64 = 0,
    xv: f64 = 0,
    pv: f64 = 0,
    px: f64 = 0,
    span: f64 = 0,

    pub const Op = enum(u8) { min, max, avg };

    pub fn feed(s: *Extremum, xv: f64, v: f64) bool {
        s.xv = xv;
        if (s.dc) {
            if (xv < s.c.from or xv > s.c.to) return false;
        } else {
            if (xv < s.c.from) return false;
            if (s.c.to != 0 and xv > s.c.to) return true;
        }
        if (!s.first) {
            s.first = true;
            s.m = if (s.op == .avg) 0 else v;
            s.m_at = xv;
            s.pv = v;
            s.px = xv;
            return false;
        }
        switch (s.op) {
            .min => if (v <= s.m) {
                s.m = v;
                s.m_at = xv;
            },
            .max => if (v >= s.m) {
                s.m = v;
                s.m_at = xv;
            },
            .avg => {
                s.m += 0.5 * (v + s.pv) * (xv - s.px);
                s.span += xv - s.px;
                s.pv = v;
                s.px = xv;
            },
        }
        return false;
    }

    pub fn stat(s: Extremum) Stat {
        return if (s.op == .avg) .{ .value = s.m / (if (s.first) s.span else 1), .at = s.xv } else .{ .value = s.m, .at = s.m_at };
    }
};

/// `Wave.rmsInteg`: the window's points, from the first sample at or past
/// `from` (read as the value at a nonzero `from`) to the first past a
/// nonzero `to` (read as the value at `to`), each panel taking the next
/// four points' spacing; points past the window read zero, as in
/// ngspice's zeroed buffers.
pub const Integ = extern struct {
    c: Bounds,
    rms: bool,
    inside: bool = false,
    /// Samples read, and the last one raw: an interpolated end's left side.
    rows: u64 = 0,
    px: f64 = 0,
    py: f64 = 0,
    /// Window points not in a panel yet, oldest first, as (scale, value or
    /// value squared).
    p: [4][2]f64 = undefined,
    np: u64 = 0,
    count: u64 = 0,
    first_x: f64 = nan,
    cut: bool = false,
    cut_x: f64 = nan,
    sum: [3]f64 = .{ 0, 0, 0 },

    pub fn feed(s: *Integ, xv: f64, v: f64) bool {
        defer {
            s.rows += 1;
            s.px = xv;
            s.py = v;
        }
        const c = s.c;
        if (!s.inside) {
            if (xv < c.from) return false;
            s.inside = true;
        }
        s.cut = c.to != 0 and xv > c.to;
        var pt: [2]f64 = .{ xv, v };
        if (s.cut) {
            if (s.rows > 0 and !almostEqualUlps(xv, c.to)) pt = .{ c.to, line(s.px, s.py, xv, v, c.to) };
        } else if (s.count == 0 and c.from != 0 and s.rows > 0 and !almostEqualUlps(xv, c.from)) {
            pt = .{ c.from, line(s.px, s.py, xv, v, c.from) };
        }
        if (s.rms) pt[1] = pt[1] * pt[1];
        if (s.count == 0) s.first_x = pt[0];
        if (s.cut) s.cut_x = pt[0];
        s.count += 1;
        s.p[@intCast(s.np)] = pt;
        s.np += 1;
        if (s.np == s.p.len) s.panel(@intCast(s.np));
        return s.cut;
    }

    /// Folds the oldest points into one panel; `avail` of `p` are points,
    /// the rest read zero, and a spacing past them zero.
    pub fn panel(s: *Integ, avail: usize) void {
        std.debug.assert(avail >= 2 and avail <= s.p.len);
        var p = s.p;
        for (p[avail..]) |*q| q.* = .{ 0, 0 };
        var width: [3]f64 = undefined;
        for (&width, 0..) |*d, j| d.* = if (j + 1 < avail) p[j + 1][0] - p[j][0] else 0;
        const used: usize = if (almostEqualUlps(width[0], width[1]) and almostEqualUlps(width[0], width[2])) blk: {
            s.sum[0] += 3 * width[0] * (p[0][1] + 3 * (p[1][1] + p[2][1]) + p[3][1]) / 8.0;
            break :blk 3;
        } else if (almostEqualUlps(width[0], width[1])) blk: {
            s.sum[1] += width[0] * (p[0][1] + 4 * p[1][1] + p[2][1]) / 3.0;
            break :blk 2;
        } else blk: {
            s.sum[2] += width[0] * (p[0][1] + p[1][1]) / 2;
            break :blk 1;
        };
        s.np = avail -| used;
        const keep: usize = @intCast(s.np);
        std.mem.copyForwards([2]f64, s.p[0..keep], s.p[avail - keep .. avail]);
    }

    pub fn result(s: *Integ) Integral {
        if (s.rows == 0) return .{ .value = nan, .from = nan, .to = nan };
        while (s.np > 1) s.panel(@intCast(s.np));
        const from = if (s.count > @intFromBool(s.cut)) s.first_x else nan;
        const to = if (s.cut) s.cut_x else s.px;
        const total = s.sum[0] + s.sum[1] + s.sum[2];
        return .{ .value = if (s.rms) @sqrt(total / (to - from)) else total, .from = from, .to = to };
    }
};

/// FIND or DERIV ... WHEN on a card measured as it goes (`measure.Stream`'s
/// stage `.find`): the pass before it in its card is the WHEN's `Cross`.
/// Until that ends, this keeps the FIND vector's last three rows; at the
/// event it becomes the `At` there, fed those rows first.
pub const Find = extern struct {
    at: At,
    /// (scale, value) of the last rows, oldest first.
    ring: [3][2]f64,
    ring_len: u8,
    state: State = .waiting,
    /// Rows fed so far, this one included once pushed.
    rows: u64,

    pub const State = enum(u8) { waiting, armed, none };

    /// `Stream.next`'s stage step for the row (`x`, `y`) after `when` read
    /// it; true where the pass is over.
    fn feed(f: *Find, when: *const Dev, x: f64, y: f64) bool {
        if (f.state == .armed) return f.at.feed(x, y) or x > f.at.at;
        if (f.ring_len == f.ring.len) std.mem.copyForwards([2]f64, f.ring[0..2], f.ring[1..3]);
        f.ring_len = @min(f.ring_len + 1, f.ring.len);
        f.ring[f.ring_len - 1] = .{ x, y };
        f.rows += 1;
        if (!when.over) return false;
        // `Scan.found`'s event: the count's own, or a LAST one.
        var time: [1]f64 = .{when.time};
        _ = when.pass.cross.finish(&time);
        if (std.math.isNan(time[0])) {
            f.state = .none;
            return true;
        }
        f.state = .armed;
        f.at.at = time[0];
        if (f.rows > f.ring_len and time[0] <= f.ring[0][0]) return true;
        for (f.ring[0..f.ring_len]) |r| if (f.at.feed(r[0], r[1]) or r[0] > f.at.at) return true;
        return false;
    }
};

/// One pass of a lane's `.meas` card on the device: which pass, whether
/// it has read its last sample (`over`), the event time a `Cross` found,
/// and the pass's state. `measure.Stream` hands these out and takes them
/// back (`deviceScans`, `adoptScans`).
pub const Dev = extern struct {
    kind: Kind,
    over: bool = false,
    time: f64 = nan,
    pass: extern union { cross: Cross, at: At, extremum: Extremum, integ: Integ, find: Find },

    pub const Kind = enum(u8) { cross, at, extremum, integ, find };

    /// `Stream`'s `Scan.feedValues` for these kinds: reads sample (`x`,
    /// `y`, `y2`) unless `over`, and sets `over` where the pass stops. A
    /// `.find` pass reads `prev`, its card's WHEN pass, fed this row first.
    pub fn feed(d: *Dev, prev: *const Dev, x: f64, y: f64, y2: f64) void {
        if (d.over) return;
        var time: [1]f64 = .{d.time};
        d.over = switch (d.kind) {
            .cross => d.pass.cross.feed(&time, x, y, y2),
            // A transient's times rise: past AT=, no later segment holds it.
            .at => d.pass.at.feed(x, y) or x > d.pass.at.at,
            .extremum => d.pass.extremum.feed(x, y),
            .integ => d.pass.integ.feed(x, y),
            .find => d.pass.find.feed(prev, x, y),
        };
        d.time = time[0];
    }
};

/// A `par()` waveform on the device: `measure.zig`'s `fold` over its
/// postfix ops, each vector read as an unknown (or the scale), to the bit.
/// No `**`: the host's `pow` takes libm's `exp` and `log`, the device's
/// its own, so a power keeps the host's rows.
pub const Program = extern struct {
    len: u8 = 0,
    ops: [max]Op = undefined,

    /// Longest program the device takes; a longer one keeps the host's rows.
    pub const max = 16;
    pub const Op = extern struct { code: Code, unknown: u32 = 0, num: f64 = 0 };
    pub const Code = enum(u8) { num, nan, unknown, scale, neg, add, sub, mul, div };

    /// The waveform at scale `x`, `at(u)` reading unknown u.
    pub fn eval(p: *const Program, x: f64, ctx: anytype) f64 {
        var stack: [32]f64 = undefined;
        var n: usize = 0;
        for (p.ops[0..p.len]) |op| switch (op.code) {
            .num, .nan, .unknown, .scale => {
                if (n == stack.len) return nan;
                stack[n] = switch (op.code) {
                    .num => op.num,
                    .unknown => ctx.at(op.unknown),
                    .scale => x,
                    else => nan, // else: `.nan`, an earlier card's value
                };
                n += 1;
            },
            .neg => {
                if (n == 0) return nan;
                stack[n - 1] = -stack[n - 1];
            },
            .add, .sub, .mul, .div => {
                if (n < 2) return nan;
                n -= 1;
                const b = stack[n];
                const a = &stack[n - 1];
                a.* = switch (op.code) {
                    .add => a.* + b,
                    .sub => a.* - b,
                    .mul => a.* * b,
                    else => a.* / b, // else: `.div`
                };
            },
        };
        return if (n == 1) stack[0] else nan;
    }

    pub fn eql(a: *const Program, b: *const Program) bool {
        if (a.len != b.len) return false;
        for (a.ops[0..a.len], b.ops[0..b.len]) |x, y| {
            if (x.code != y.code or x.unknown != y.unknown or @as(u64, @bitCast(x.num)) != @as(u64, @bitCast(y.num))) return false;
        }
        return true;
    }
};

/// Where a `Dev` pass reads its samples: the unknown behind `y` and the
/// level vector's (`none` without one), or a `par()` waveform for either.
pub const DevColumns = extern struct {
    y: u32,
    y2: u32,
    prog: Program = .{},
    prog2: Program = .{},

    pub const none = std.math.maxInt(u32);

    /// The sample (`y`, `y2`) at scale `x`, `ctx.at(u)` reading unknown u.
    pub fn read(c: *const DevColumns, x: f64, ctx: anytype) [2]f64 {
        const y = if (c.prog.len != 0) c.prog.eval(x, ctx) else ctx.at(c.y);
        const y2 = if (c.prog2.len != 0) c.prog2.eval(x, ctx) else if (c.y2 == none) nan else ctx.at(c.y2);
        return .{ y, y2 };
    }

    pub fn eql(a: *const DevColumns, b: *const DevColumns) bool {
        return a.y == b.y and a.y2 == b.y2 and a.prog.eql(&b.prog) and a.prog2.eql(&b.prog2);
    }
};
