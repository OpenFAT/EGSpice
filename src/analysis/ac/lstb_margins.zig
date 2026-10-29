//! HSPICE's `.lstb` stability margins from a loop-gain sweep: bracket the
//! first unity-magnitude and the first negative-real crossing between
//! neighbouring sweep points, then refine each on the circuit itself by
//! Illinois regula falsi, so the margins do not depend on the grid. The
//! `Refiner` pauses before every solve, so a host loop (`crossing`) and the
//! GPU rounds (`lstb_lanes.zig`) take the same steps.
//! In: the sweep's frequencies and loop gains, and a one-frequency solver.
//! Out: five margins.
//! Cites HSPICE [CR .MEASURE LSTB]; dev/analysis/stability.md.
//!
//! Contents:
//! - `pub const Margins = struct` (`names`, `values`, `unbracketed`, `setUnity`, `setNegativeReal`, `find`)
//! - `pub const Crossing = struct { f, w }`
//! - `pub const Refiner = struct` (`Mode`, `init`, `next`, `accept`)
//! - `pub const Probe = struct` (`init`, `deinit`, `gain`)

const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;
const Complex = @import("core").numerics.Complex;
const freq_solve = @import("solver").freq_solve;
const FreqSolver = freq_solve.FreqSolver;
const root = @import("../types.zig");
const lstb = @import("lstb.zig");

/// HSPICE's `.lstb` scalars, plus the frequency the gain margin is read at.
/// A margin whose crossing the sweep never brackets is NaN.
pub const Margins = struct {
    /// dB: -20 log10 |W| where W is real and negative.
    gain_margin: f64,
    /// Hz.
    phase_crossover_freq: f64,
    /// Degrees: 180 + angle(W) at |W| = 1.
    phase_margin: f64,
    /// Hz.
    unity_gain_freq: f64,
    /// dB: 20 log10 |W| at the sweep's first frequency.
    loop_gain_minifreq: f64,

    /// The published column names, in `values` order.
    pub const names = [_][]const u8{ "gain_margin", "phase_crossover_freq", "phase_margin", "unity_gain_freq", "loop_gain_minifreq" };

    /// Returns the published row, in `names` order.
    pub fn values(m: Margins) [names.len]f64 {
        return .{ m.gain_margin, m.phase_crossover_freq, m.phase_margin, m.unity_gain_freq, m.loop_gain_minifreq };
    }

    /// Returns every margin NaN until a crossing sets it; `first` is the
    /// loop gain at the sweep's first frequency.
    pub fn unbracketed(first: Complex) Margins {
        const nan = std.math.nan(f64);
        return .{ .gain_margin = nan, .phase_crossover_freq = nan, .phase_margin = nan, .unity_gain_freq = nan, .loop_gain_minifreq = 20 * std.math.log10(first.mag()) };
    }

    /// Sets the phase margin at the unity-magnitude crossing, or warns when
    /// the sweep never brackets one.
    pub fn setUnity(m: *Margins, cross: ?Crossing) void {
        const c = cross orelse return std.debug.print("warning: .lstb: |loop gain| never crosses 1 in the sweep; no phase margin\n", .{});
        m.unity_gain_freq = c.f;
        m.phase_margin = std.math.radiansToDegrees(reversed(c.w));
    }

    /// Sets the gain margin at the negative-real crossing, or warns when the
    /// sweep never brackets one.
    pub fn setNegativeReal(m: *Margins, cross: ?Crossing) void {
        const c = cross orelse return std.debug.print("warning: .lstb: loop gain never crosses -180 degrees in the sweep; no gain margin\n", .{});
        m.phase_crossover_freq = c.f;
        m.gain_margin = -20 * std.math.log10(c.w.mag());
    }

    /// Returns the margins of `loop` over `freqs`, each crossing refined on
    /// `probe`. Asserts one loop gain per frequency.
    pub fn find(probe: *Probe, freqs: []const f64, loop: []const Complex) !Margins {
        assert(loop.len == freqs.len);
        assert(loop.len >= 1);
        var m: Margins = .unbracketed(loop[0]);
        m.setUnity(try crossing(probe, freqs, loop, .unity));
        m.setNegativeReal(try crossing(probe, freqs, loop, .negative_real));
        return m;
    }
};

/// A refined crossing: its frequency and the loop gain there.
pub const Crossing = struct { f: f64, w: Complex };

/// The Illinois crossing loop, paused immediately before each circuit
/// solve. Every point owns its bracket and stopping decision. Invariant
/// while not `done`: `lo < hi` and `g_lo`, `g_hi` have opposite signs.
pub const Refiner = struct {
    mode: Mode,
    lo: f64 = undefined,
    hi: f64 = undefined,
    g_lo: f64 = undefined,
    g_hi: f64 = undefined,
    f: f64 = undefined,
    best: ?Crossing = null,
    side: i2 = 0,
    steps: u8 = 0,
    done: bool = true,

    /// Which crossing the refiner brackets.
    pub const Mode = enum { unity, negative_real };

    /// Returns a refiner on the first sign change of `mode`'s function
    /// between neighbouring sweep points; done at once when a grid point is
    /// the crossing or no neighbours bracket one.
    pub fn init(freqs: []const f64, loop: []const Complex, mode: Mode) Refiner {
        assert(loop.len == freqs.len);
        for (1..loop.len) |k| {
            const g_lo = value(mode, loop[k - 1]) orelse continue;
            const g_hi = value(mode, loop[k]) orelse continue;
            if (g_lo == 0) return .{ .mode = mode, .best = .{ .f = freqs[k - 1], .w = loop[k - 1] } };
            if (g_hi == 0) return .{ .mode = mode, .best = .{ .f = freqs[k], .w = loop[k] } };
            if ((g_lo < 0) == (g_hi < 0)) continue;
            return .{ .mode = mode, .lo = freqs[k - 1], .hi = freqs[k], .g_lo = g_lo, .g_hi = g_hi, .best = .{ .f = freqs[k], .w = loop[k] }, .done = false };
        }
        return .{ .mode = mode };
    }

    /// Returns the next frequency to solve at, or null when refined.
    pub fn next(self: *Refiner) ?f64 {
        if (self.done) return null;
        assert(self.lo < self.hi);
        var f = self.hi - self.g_hi * (self.hi - self.lo) / (self.g_hi - self.g_lo);
        if (!(f > self.lo and f < self.hi)) f = 0.5 * (self.lo + self.hi);
        self.f = f;
        return f;
    }

    /// Takes the loop gain at the frequency `next` returned. Ends after 100
    /// solves, at a relative width of 1e-13, on an exact root, or when the
    /// function leaves its domain. Asserts that a solve was pending.
    pub fn accept(self: *Refiner, w: Complex) void {
        assert(!self.done);
        const gv = value(self.mode, w) orelse {
            self.done = true;
            return;
        };
        self.best = .{ .f = self.f, .w = w };
        self.steps += 1;
        if (gv == 0 or self.hi - self.lo <= 1e-13 * self.hi) {
            self.done = true;
            return;
        }
        if ((gv < 0) == (self.g_hi < 0)) {
            self.hi = self.f;
            self.g_hi = gv;
            if (self.side == 1) self.g_lo *= 0.5;
            self.side = 1;
        } else {
            self.lo = self.f;
            self.g_lo = gv;
            if (self.side == -1) self.g_hi *= 0.5;
            self.side = -1;
        }
        self.done = self.steps == steps_max;
    }
};

/// The loop gain at any one frequency, for refining a crossing between two
/// sweep points: one single-lane `solveBatch` on the sweep's solver.
pub const Probe = struct {
    fs: *FreqSolver,
    ckt: *root.Circuit,
    x_op: []const f64,
    rhs: []const f64,
    opts: lstb.Options,
    /// Both solutions, then the `acDyn` terms re and im.
    buf: []f64,

    /// Borrows every argument for the probe's life; free with `deinit` and
    /// the same allocator.
    pub fn init(gpa: Allocator, fs: *FreqSolver, ckt: *root.Circuit, x_op: []const f64, rhs: []const f64, opts: lstb.Options) Allocator.Error!Probe {
        const buf = try gpa.alloc(f64, rhs.len + 2 * ckt.ac_dyn_slots.len);
        return .{ .fs = fs, .ckt = ckt, .x_op = x_op, .rhs = rhs, .opts = opts, .buf = buf };
    }

    /// Frees the solution buffer.
    pub fn deinit(self: *Probe, gpa: Allocator) void {
        gpa.free(self.buf);
    }

    /// Returns the total loop gain at `f` Hz.
    pub fn gain(self: *Probe, f: f64) !Complex {
        const omega = [1]f64{2 * std.math.pi * f};
        const slots = self.ckt.ac_dyn_slots.len;
        const x = self.buf[0..self.rhs.len];
        const re = self.buf[self.rhs.len..][0..slots];
        const im = self.buf[self.rhs.len + slots ..][0..slots];
        if (slots != 0) self.ckt.acDyn(self.x_op, &omega, re, im);
        const dyn: freq_solve.Dyn = .{ .slots = self.ckt.ac_dyn_slots, .re = re, .im = im };
        try self.fs.solveBatch(&omega, dyn, self.rhs, x, false);
        return lstb.Gains.of(lstb.responses(self.opts, self.ckt.n, x)).w;
    }
};

/// Most solves one crossing takes.
const steps_max = 100;

// ---- private ----

/// angle(-W) in radians, (-pi, pi]: the phase margin angle.
fn reversed(w: Complex) f64 {
    return std.math.atan2(-w.im, -w.re);
}

/// The function whose sign change `mode` brackets: ln|W|, or angle(-W),
/// defined only near the negative real axis (Re W < 0), where it is
/// continuous and zero at the crossing.
fn value(mode: Refiner.Mode, w: Complex) ?f64 {
    return switch (mode) {
        .unity => @log(w.mag()),
        .negative_real => if (w.re < 0) reversed(w) else null,
    };
}

/// The refined `mode` crossing of `loop`, solving on `probe` at each step.
/// Null when no neighbours bracket a root.
fn crossing(probe: *Probe, freqs: []const f64, loop: []const Complex, mode: Refiner.Mode) !?Crossing {
    var r = Refiner.init(freqs, loop, mode);
    while (r.next()) |f| r.accept(try probe.gain(f));
    return r.best;
}

test crossing {
    // A sign change exactly on a grid point returns that point without a
    // solve; no bracket returns null. Neither path touches `probe`.
    const freqs = [_]f64{ 1, 10, 100 };
    const above = [_]Complex{ .{ .re = 4, .im = 0 }, .{ .re = 2, .im = 0 }, .{ .re = 1.5, .im = 0 } };
    var probe: Probe = undefined;
    try std.testing.expectEqual(null, try crossing(&probe, &freqs, &above, .unity));
    const on_grid = [_]Complex{ .{ .re = 4, .im = 0 }, .{ .re = 1, .im = 0 }, .{ .re = 0.5, .im = 0 } };
    const c = (try crossing(&probe, &freqs, &on_grid, .unity)).?;
    try std.testing.expectEqual(@as(f64, 10), c.f);
    // angle(-W) is undefined off the negative real axis, so a loop that
    // stays in the right half plane never brackets a phase crossover.
    try std.testing.expectEqual(null, try crossing(&probe, &freqs, &above, .negative_real));
}

test Refiner {
    // Oracle: |W| = 10/f crosses 1 at f = 10 exactly; the refiner fed the
    // analytic gain converges there within its 1e-13 width.
    const freqs = [_]f64{ 1, 3, 30 };
    const loop = [_]Complex{ .{ .re = 10, .im = 0 }, .{ .re = 10.0 / 3.0, .im = 0 }, .{ .re = 1.0 / 3.0, .im = 0 } };
    var r = Refiner.init(&freqs, &loop, .unity);
    while (r.next()) |f| r.accept(.{ .re = 10 / f, .im = 0 });
    try std.testing.expectApproxEqRel(@as(f64, 10), r.best.?.f, 1e-12);
    try std.testing.expect(r.steps <= steps_max);
}
