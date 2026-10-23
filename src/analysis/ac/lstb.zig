//! Loop stability by double injection (`.lstb`, HSPICE; VACASK `acstb`).
//! Each frequency solves two right-hand sides on one lane factorization: a
//! unit current into the probe's `+` node, and a unit voltage across the
//! probe. VACASK's two-port reading of the four responses (coreacstb.cpp)
//! gives the DUT y-parameters and the forward, reverse and total loop gain;
//! the total is Tian's loop gain, exact for a bilateral loop and any
//! loading at the break. `diff` and `comm` drive a probe pair ±1 and read
//! the modal half of each response.
//! Cites VACASK coreacstb.cpp, HSPICE [CR .MEASURE LSTB] and
//! dev/analysis/stability.md.
const std = @import("std");
const freq = @import("freq.zig");
const root = @import("../types.zig");
const core = @import("core");
const Complex = core.numerics.Complex;
const FreqSolver = @import("solver").freq_solve.FreqSolver;
const at = @import("../dc/xf.zig").at;
const Dyn = @import("solver").freq_solve.Dyn;
const rounds = @import("../gpu/freq.zig");

/// Query options, defined in core/query.zig.
pub const Options = core.query.Lstb;

/// The probe responses in VACASK's names: the probe current (A, B) and the
/// `+` node voltage (C, D) under current (A, C) and voltage (B, D)
/// injection.
pub const Responses = struct { a: Complex, b: Complex, c: Complex, d: Complex };

/// Everything `.lstb` publishes at one frequency.
pub const Gains = struct {
    /// Total loop gain W = Wf + Wr, the return ratio: positive at DC for
    /// negative feedback, unstable at W = −1.
    w: Complex,
    /// The forward and reverse parts of `w`.
    wf: Complex,
    wr: Complex,
    /// The DUT y-parameters seen across the break.
    y11: Complex,
    y12: Complex,
    y21: Complex,
    y22: Complex,

    /// VACASK coreacstb.cpp's formulas. The y-parameters divide by C, so
    /// they are infinite when the probe's `+` node is an ideal source.
    pub fn of(r: Responses) Gains {
        const one: Complex = .{ .re = 1, .im = 0 };
        const adbc = r.a.mul(r.d).sub(r.b.mul(r.c));
        const fwd = r.a.sub(adbc);
        const rev = r.d.sub(adbc);
        const den = one.add(adbc.scale(2)).sub(r.a).sub(r.d);
        const wf = fwd.div(den);
        const wr = rev.div(den);
        return .{
            .w = wf.add(wr),
            .wf = wf,
            .wr = wr,
            .y11 = one.add(adbc).sub(r.a).sub(r.d).div(r.c),
            .y12 = rev.div(r.c),
            .y21 = fwd.div(r.c),
            .y22 = adbc.div(r.c),
        };
    }
};

/// Excitation weight of each probe: `single` drives the first alone, `diff`
/// the pair +1/−1, `comm` +1/+1.
fn weights(mode: Options.Mode) [2]f64 {
    return switch (mode) {
        .single => .{ 1, 0 },
        .diff => .{ 1, -1 },
        .comm => .{ 1, 1 },
    };
}

/// The two stacked 2n right-hand sides: current injection from the local
/// ground, then voltage injection. `rhs` must be zeroed. An unweighted probe
/// (`single`'s second) is never read, as in `responses`.
fn fill(opts: Options, n: usize, rhs: []f64) void {
    for (opts.probes, weights(opts.mode)) |p, e| {
        if (e == 0) continue;
        if (p.p != root.GROUND) rhs[p.p] += e;
        if (opts.local_gnd != root.GROUND) rhs[opts.local_gnd] -= e;
        rhs[2 * n + p.branch] += e;
    }
}

/// The modal responses in `x` (both solutions): each probe's reading times
/// its weight, averaged over the driven probes, so a symmetric pair reads
/// its half circuit.
fn responses(opts: Options, n: usize, x: []const f64) Responses {
    const e = weights(opts.mode);
    const share: f64 = if (opts.mode == .single) 1 else 0.5;
    var r: Responses = .{ .a = .zero, .b = .zero, .c = .zero, .d = .zero };
    for (opts.probes, e) |p, w| {
        if (w == 0) continue;
        const s = w * share;
        r.a = r.a.add(at(x, n, p.branch).scale(s));
        r.c = r.c.add(at(x, n, p.p).sub(at(x, n, opts.local_gnd)).scale(s));
        r.b = r.b.add(at(x[2 * n ..], n, p.branch).scale(s));
        r.d = r.d.add(at(x[2 * n ..], n, p.p).sub(at(x[2 * n ..], n, opts.local_gnd)).scale(s));
    }
    return r;
}

/// The two-injection sweep. Margin refinement continues the same
/// frequency solver after the sweep, so it retains its live pivot history.
pub fn plan(ctx: *const root.RunCtx, opts: Options) !?freq.Plan {
    if (opts.margins) return null;
    const scratch = ctx.scratch_allocator;
    const n: usize = ctx.circuit.n;
    try validateProbes(opts, n);
    const rhs = try scratch.alloc(f64, 4 * n);
    errdefer scratch.free(rhs);
    @memset(rhs, 0);
    fill(opts, n, rhs);
    const freqs, const omegas = try freq.axis(scratch, opts.sweep);
    return .{ .freqs = freqs, .omegas = omegas, .rhs = rhs, .adjoint = false };
}

fn validateProbes(opts: Options, n: usize) !void {
    const used: usize = if (opts.mode == .single) 1 else 2;
    for (opts.probes[0..used]) |p| if (p.p >= n or p.n >= n or p.branch >= n) return error.InvalidProbe;
    if (opts.local_gnd >= n) return error.InvalidProbe;
}

/// Contract entry. The sweep plot is complex (frequency, loop_gain,
/// loop_gain_forward, loop_gain_reverse, y11, y12, y21, y22); with
/// `margins` it is one real row instead (see `Margins`).
///
/// Returns error.InvalidProbe when a probe row is out of range.
pub fn run(ctx: *const root.RunCtx, opts: Options) !root.Result {
    const a = ctx.allocator;
    const scratch = ctx.scratch_allocator;
    const ckt = ctx.circuit;
    const n: usize = ckt.n;
    try validateProbes(opts, n);

    var fs = try FreqSolver.fromCircuit(scratch, ckt, ctx.x_op);
    defer fs.deinit();

    const n_points: usize = opts.sweep.count();
    const freqs, const omegas = try freq.axis(scratch, opts.sweep);
    defer freq.freeAxis(scratch, freqs);

    const rhs = try scratch.alloc(f64, 4 * n);
    defer scratch.free(rhs);
    @memset(rhs, 0);
    fill(opts, n, rhs);

    const loop = try scratch.alloc(Complex, n_points);
    defer scratch.free(loop);
    const cols = 8;
    const data = try a.alloc(f64, if (opts.margins) 5 else n_points * cols * 2);
    errdefer a.free(data);

    var stream = try freq.Stream.init(scratch, &fs, ckt, ctx.x_op, omegas, rhs, false);
    defer stream.deinit(scratch);
    stream.pre = ctx.freq_pre;
    while (try stream.next(ckt)) |pt| {
        const g = Gains.of(responses(opts, n, pt.x));
        loop[pt.k] = g.w;
        if (opts.margins) continue;
        const row = data[pt.k * cols * 2 ..][0 .. cols * 2];
        row[0..2].* = .{ freqs[pt.k], 0 };
        for ([_]Complex{ g.w, g.wf, g.wr, g.y11, g.y12, g.y21, g.y22 }, 0..) |v, i| row[2 + 2 * i ..][0..2].* = .{ v.re, v.im };
    }

    if (!opts.margins) return .{
        .plotname = "Loop Stability Analysis",
        .varnames = try a.dupe([]const u8, &.{ "frequency", "loop_gain", "loop_gain_forward", "loop_gain_reverse", "y11", "y12", "y21", "y22" }),
        .is_complex = true,
        .npoints = n_points,
        .data = data,
    };

    var probe: Probe = try .init(scratch, &fs, ckt, ctx.x_op, rhs, opts);
    defer probe.deinit(scratch);
    const m = try Margins.find(&probe, freqs, loop);
    data[0..5].* = .{ m.gain_margin, m.phase_crossover_freq, m.phase_margin, m.unity_gain_freq, m.loop_gain_minifreq };
    return .{
        .plotname = "Loop Stability Margins",
        .varnames = try a.dupe([]const u8, &.{ "gain_margin", "phase_crossover_freq", "phase_margin", "unity_gain_freq", "loop_gain_minifreq" }),
        .is_complex = false,
        .npoints = 1,
        .data = data,
    };
}

pub const LaneOutcome = root.ErrorOf(run)!root.Result;
pub const LaneStats = struct {
    rounds: usize = 0,
    device_lanes: usize = 0,
    device_groups: usize = 0,
    device_peels: usize = 0,
    refinement_requests: usize = 0,
    device_refinement_lanes: usize = 0,
};

/// Completion-quantum GPU batches keep every point's solver alive from
/// its sweep through both adaptive margin refinements. Ordinary sweeps
/// keep the existing precompute path.
pub fn runGpuLanes(ctxs: []const root.RunCtx, opts: Options, backend: @import("../gpu/context.zig").Request, out: []LaneOutcome) void {
    const stats = runLanes(ctxs, opts, backend, out) catch |err| {
        for (out) |*result| result.* = err;
        return;
    };
    if (ctxs.len != 0 and ctxs[0].circuit.knobs.gpu_stats != .off)
        std.debug.print("gpu-lstb: {d} points, {d} rounds, {d} device frequency lanes, {d} groups, {d} peels, {d} refinement requests, {d} device refinement lanes\n", .{ ctxs.len, stats.rounds, stats.device_lanes, stats.device_groups, stats.device_peels, stats.refinement_requests, stats.device_refinement_lanes });
}

/// Also accepts .cpu for an independent comparison against run(). Each
/// active point emits at most one original frequency chunk per round.
/// Failed or cancelled points stop without changing a peer's history.
fn runLanes(ctxs: []const root.RunCtx, opts: Options, backend: @import("../gpu/context.zig").Request, out: []LaneOutcome) std.mem.Allocator.Error!LaneStats {
    std.debug.assert(ctxs.len == out.len);
    if (ctxs.len == 0) return .{};
    @memset(out, error.QueryCancelled);
    if (!opts.margins) {
        for (ctxs, out) |*ctx, *result| result.* = run(ctx, opts);
        return .{};
    }
    const allocator = ctxs[0].scratch_allocator;
    const states = try allocator.alloc(?MarginState, ctxs.len);
    defer allocator.free(states);
    @memset(states, null);
    defer for (states) |*state| if (state.*) |*live| live.deinit();
    const requests = try allocator.alloc(rounds.RoundPoint, ctxs.len);
    defer allocator.free(requests);
    const indices = try allocator.alloc(usize, ctxs.len);
    defer allocator.free(indices);
    const failures = try allocator.alloc(?rounds.SolveError, ctxs.len);
    defer allocator.free(failures);
    var driver = rounds.RoundDriver.init(allocator, backend);
    defer driver.deinit();
    for (ctxs, states, out) |*ctx, *state, *result| {
        state.* = MarginState.init(ctx, opts) catch |err| {
            result.* = err;
            continue;
        };
    }
    var stats: LaneStats = .{};
    while (true) {
        var count: usize = 0;
        var sweep_pending = false;
        for (states, out, 0..) |*state, *result, index| {
            const live = if (state.*) |*s| s else continue;
            const request = live.next() catch |err| {
                result.* = err;
                live.deinit();
                state.* = null;
                continue;
            };
            if (request) |r| {
                requests[count] = r;
                indices[count] = index;
                count += 1;
                stats.refinement_requests += @intFromBool(live.phase != .sweep);
                sweep_pending = sweep_pending or live.phase == .sweep;
            } else {
                result.* = live.result();
                live.deinit();
                state.* = null;
            }
        }
        if (count == 0) break;
        // Round setup allocates before it touches any solver. If that
        // storage is unavailable, finish these same live requests on CPU
        // without invalidating a result that an earlier round completed.
        const stat = driver.solve(requests[0..count], failures[0..count]) catch
            rounds.solveRound(allocator, .cpu, requests[0..count], failures[0..count]) catch unreachable;
        stats.rounds += 1;
        stats.device_lanes += stat.lanes;
        stats.device_groups += stat.groups;
        stats.device_peels += stat.peels;
        // This conservative counter includes only rounds whose requests
        // are all refinements, so a sweep can never satisfy the hardware
        // admission gate for the adaptive part.
        if (!sweep_pending) stats.device_refinement_lanes += stat.lanes;
        for (failures[0..count], indices[0..count]) |failure, index| {
            if (failure) |err| {
                out[index] = err;
                states[index].?.deinit();
                states[index] = null;
            }
        }
    }
    return stats;
}

const MarginState = struct {
    const Phase = enum { sweep, unity, negative_real, done };
    ctx: *const root.RunCtx,
    opts: Options,
    fs: FreqSolver,
    freqs: []f64,
    omegas: []f64,
    rhs: []f64,
    solutions: []f64,
    dynamic: []f64,
    loop: []Complex,
    phase: Phase = .sweep,
    sweep_base: usize = 0,
    sweep_end: usize = 0,
    pending: bool = false,
    omega: [1]f64 = undefined,
    refiner: Refiner = undefined,
    margins: Margins = undefined,

    fn init(ctx: *const root.RunCtx, opts: Options) !MarginState {
        const scratch = ctx.scratch_allocator;
        const ckt = ctx.circuit;
        try validateProbes(opts, ckt.n);
        var fs = try FreqSolver.fromCircuit(scratch, ckt, ctx.x_op);
        errdefer fs.deinit();
        const freqs, const omegas = try freq.axis(scratch, opts.sweep);
        errdefer freq.freeAxis(scratch, freqs);
        const rhs = try scratch.alloc(f64, 4 * @as(usize, ckt.n));
        errdefer scratch.free(rhs);
        @memset(rhs, 0);
        fill(opts, ckt.n, rhs);
        const solutions = try scratch.alloc(f64, FreqSolver.W * rhs.len);
        errdefer scratch.free(solutions);
        const dynamic = try scratch.alloc(f64, 2 * FreqSolver.W * ckt.ac_dyn_slots.len);
        errdefer scratch.free(dynamic);
        const loop = try scratch.alloc(Complex, freqs.len);
        return .{ .ctx = ctx, .opts = opts, .fs = fs, .freqs = freqs, .omegas = omegas, .rhs = rhs, .solutions = solutions, .dynamic = dynamic, .loop = loop };
    }

    fn deinit(self: *MarginState) void {
        const scratch = self.ctx.scratch_allocator;
        self.fs.deinit();
        freq.freeAxis(scratch, self.freqs);
        scratch.free(self.rhs);
        scratch.free(self.solutions);
        scratch.free(self.dynamic);
        scratch.free(self.loop);
        self.* = undefined;
    }

    fn request(self: *MarginState, omegas: []const f64) rounds.RoundPoint {
        const ckt = self.ctx.circuit;
        const terms = ckt.ac_dyn_slots.len * omegas.len;
        const re = self.dynamic[0..terms];
        const im = self.dynamic[self.dynamic.len / 2 ..][0..terms];
        if (terms != 0) ckt.acDyn(self.ctx.x_op, omegas, re, im);
        self.pending = true;
        return .{ .fs = &self.fs, .omegas = omegas, .dyn = .{ .slots = ckt.ac_dyn_slots, .re = re, .im = im }, .rhs = self.rhs, .out = self.solutions[0 .. omegas.len * self.rhs.len], .adjoint = false };
    }

    fn next(self: *MarginState) !?rounds.RoundPoint {
        if (self.pending) {
            self.pending = false;
            if (self.phase == .sweep) {
                // Stream checkpoints after solving a chunk and before
                // returning any of its rows; preserve that exact boundary.
                if (self.sweep_end % freq.quantum == 0 or self.sweep_end == self.omegas.len)
                    try self.ctx.circuit.checkpoint(.{ .phase = .frequency, .completed = self.sweep_end, .total = self.omegas.len });
                for (self.sweep_base..self.sweep_end, 0..) |k, local|
                    self.loop[k] = Gains.of(responses(self.opts, self.ctx.circuit.n, self.solutions[local * self.rhs.len ..][0..self.rhs.len])).w;
                self.sweep_base = self.sweep_end;
            } else {
                const gain = Gains.of(responses(self.opts, self.ctx.circuit.n, self.solutions[0..self.rhs.len])).w;
                self.refiner.accept(gain);
            }
        }
        while (true) {
            switch (self.phase) {
                .sweep => {
                    if (self.sweep_base != self.omegas.len) {
                        self.sweep_end = self.sweep_base + @min(FreqSolver.W, self.omegas.len - self.sweep_base);
                        return self.request(self.omegas[self.sweep_base..self.sweep_end]);
                    }
                    const nan = std.math.nan(f64);
                    self.margins = .{ .gain_margin = nan, .phase_crossover_freq = nan, .phase_margin = nan, .unity_gain_freq = nan, .loop_gain_minifreq = 20 * std.math.log10(self.loop[0].mag()) };
                    self.refiner = Refiner.init(self.freqs, self.loop, .unity);
                    self.phase = .unity;
                },
                .unity, .negative_real => {
                    if (self.refiner.next()) |f| {
                        self.omega[0] = 2 * std.math.pi * f;
                        return self.request(&self.omega);
                    }
                    if (self.phase == .unity) {
                        if (self.refiner.best) |cross| {
                            self.margins.unity_gain_freq = cross.f;
                            self.margins.phase_margin = std.math.radiansToDegrees(reversed(cross.w));
                        } else std.debug.print("warning: .lstb: |loop gain| never crosses 1 in the sweep; no phase margin\n", .{});
                        self.refiner = Refiner.init(self.freqs, self.loop, .negative_real);
                        self.phase = .negative_real;
                    } else {
                        if (self.refiner.best) |cross| {
                            self.margins.phase_crossover_freq = cross.f;
                            self.margins.gain_margin = -20 * std.math.log10(cross.w.mag());
                        } else std.debug.print("warning: .lstb: loop gain never crosses -180 degrees in the sweep; no gain margin\n", .{});
                        self.phase = .done;
                    }
                },
                .done => return null,
            }
        }
    }

    fn result(self: *const MarginState) !root.Result {
        std.debug.assert(self.phase == .done);
        const a = self.ctx.allocator;
        const m = self.margins;
        const data = try a.dupe(f64, &.{ m.gain_margin, m.phase_crossover_freq, m.phase_margin, m.unity_gain_freq, m.loop_gain_minifreq });
        errdefer a.free(data);
        return .{
            .plotname = "Loop Stability Margins",
            .varnames = try a.dupe([]const u8, &.{ "gain_margin", "phase_crossover_freq", "phase_margin", "unity_gain_freq", "loop_gain_minifreq" }),
            .is_complex = false,
            .npoints = 1,
            .data = data,
        };
    }
};

/// The loop gain at any one frequency, for refining a crossing between two
/// sweep points: one single-lane `solveBatch` on the sweep's solver.
const Probe = struct {
    fs: *FreqSolver,
    ckt: *root.Circuit,
    x_op: []const f64,
    rhs: []const f64,
    opts: Options,
    /// Both solutions, then the `acDyn` terms re and im.
    buf: []f64,

    fn init(allocator: std.mem.Allocator, fs: *FreqSolver, ckt: *root.Circuit, x_op: []const f64, rhs: []const f64, opts: Options) !Probe {
        const buf = try allocator.alloc(f64, rhs.len + 2 * ckt.ac_dyn_slots.len);
        return .{ .fs = fs, .ckt = ckt, .x_op = x_op, .rhs = rhs, .opts = opts, .buf = buf };
    }

    fn deinit(self: *Probe, allocator: std.mem.Allocator) void {
        allocator.free(self.buf);
    }

    fn gain(self: *Probe, f: f64) !Complex {
        const omega = [1]f64{2 * std.math.pi * f};
        const slots = self.ckt.ac_dyn_slots.len;
        const x = self.buf[0..self.rhs.len];
        const re = self.buf[self.rhs.len..][0..slots];
        const im = self.buf[self.rhs.len + slots ..][0..slots];
        if (slots != 0) self.ckt.acDyn(self.x_op, &omega, re, im);
        const dyn: Dyn = .{ .slots = self.ckt.ac_dyn_slots, .re = re, .im = im };
        try self.fs.solveBatch(&omega, dyn, self.rhs, x, false);
        return Gains.of(responses(self.opts, self.ckt.n, x)).w;
    }
};

/// HSPICE's `.lstb` scalars [CR .MEASURE LSTB], plus the frequency the
/// gain margin is read at. A margin whose crossing the sweep never brackets
/// is NaN.
pub const Margins = struct {
    /// dB: −20·log10|W| where W is real and negative.
    gain_margin: f64,
    /// Hz.
    phase_crossover_freq: f64,
    /// Degrees: 180 + ∠W at |W| = 1.
    phase_margin: f64,
    /// Hz.
    unity_gain_freq: f64,
    /// dB: 20·log10|W| at the sweep's first frequency.
    loop_gain_minifreq: f64,

    /// Brackets the first unity-magnitude and first negative-real crossing
    /// of `loop` over `freqs`, then refines each on the circuit itself
    /// (`crossing`), so the margins do not depend on the grid.
    fn find(probe: *Probe, freqs: []const f64, loop: []const Complex) !Margins {
        const nan = std.math.nan(f64);
        var m: Margins = .{ .gain_margin = nan, .phase_crossover_freq = nan, .phase_margin = nan, .unity_gain_freq = nan, .loop_gain_minifreq = 20 * std.math.log10(loop[0].mag()) };
        if (try crossing(probe, freqs, loop, logMag)) |c| {
            m.unity_gain_freq = c.f;
            m.phase_margin = std.math.radiansToDegrees(reversed(c.w));
        } else std.debug.print("warning: .lstb: |loop gain| never crosses 1 in the sweep; no phase margin\n", .{});
        if (try crossing(probe, freqs, loop, reversedPhase)) |c| {
            m.phase_crossover_freq = c.f;
            m.gain_margin = -20 * std.math.log10(c.w.mag());
        } else std.debug.print("warning: .lstb: loop gain never crosses -180 degrees in the sweep; no gain margin\n", .{});
        return m;
    }
};

/// ∠(−W) in radians, (−π, π]: the phase margin angle.
fn reversed(w: Complex) f64 {
    return std.math.atan2(-w.im, -w.re);
}

fn logMag(w: Complex) ?f64 {
    return @log(w.mag());
}

/// ∠(−W), defined only near the negative real axis (Re W < 0), where it is
/// continuous and zero at the crossing.
fn reversedPhase(w: Complex) ?f64 {
    return if (w.re < 0) reversed(w) else null;
}

/// A refined crossing: its frequency and the loop gain there.
const Crossing = struct { f: f64, w: Complex };

/// The original Illinois crossing loop, paused immediately before each
/// circuit solve. Every point owns its bracket and stopping decision.
const Refiner = struct {
    const Mode = enum { unity, negative_real };
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

    fn value(mode: Mode, w: Complex) ?f64 {
        return switch (mode) {
            .unity => logMag(w),
            .negative_real => reversedPhase(w),
        };
    }

    fn init(freqs: []const f64, loop: []const Complex, mode: Mode) Refiner {
        for (1..loop.len) |k| {
            const g_lo = value(mode, loop[k - 1]) orelse continue;
            const g_hi = value(mode, loop[k]) orelse continue;
            if (g_lo == 0) return .{ .mode = mode, .best = .{ .f = freqs[k - 1], .w = loop[k - 1] } };
            if ((g_lo < 0) == (g_hi < 0) or g_hi == 0) {
                if (g_hi == 0) return .{ .mode = mode, .best = .{ .f = freqs[k], .w = loop[k] } };
                continue;
            }
            return .{ .mode = mode, .lo = freqs[k - 1], .hi = freqs[k], .g_lo = g_lo, .g_hi = g_hi, .best = .{ .f = freqs[k], .w = loop[k] }, .done = false };
        }
        return .{ .mode = mode };
    }

    fn next(self: *Refiner) ?f64 {
        if (self.done) return null;
        var f = self.hi - self.g_hi * (self.hi - self.lo) / (self.g_hi - self.g_lo);
        if (!(f > self.lo and f < self.hi)) f = 0.5 * (self.lo + self.hi);
        self.f = f;
        return f;
    }

    fn accept(self: *Refiner, w: Complex) void {
        std.debug.assert(!self.done);
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
        self.done = self.steps == 100;
    }
};

/// The first sign change of `g` between neighbouring sweep points,
/// refined by Illinois regula falsi on `probe` to a relative frequency
/// width of 1e-13 (at most 100 solves). Null when no neighbours bracket a
/// root.
fn crossing(probe: *Probe, freqs: []const f64, loop: []const Complex, comptime g: fn (Complex) ?f64) !?Crossing {
    for (1..loop.len) |k| {
        var g_lo = g(loop[k - 1]) orelse continue;
        var g_hi = g(loop[k]) orelse continue;
        if (g_lo == 0) return .{ .f = freqs[k - 1], .w = loop[k - 1] };
        if ((g_lo < 0) == (g_hi < 0) or g_hi == 0) {
            if (g_hi == 0) return .{ .f = freqs[k], .w = loop[k] };
            continue;
        }
        var lo = freqs[k - 1];
        var hi = freqs[k];
        var best: Crossing = .{ .f = hi, .w = loop[k] };
        var side: i2 = 0;
        for (0..100) |_| {
            var f = hi - g_hi * (hi - lo) / (g_hi - g_lo);
            if (!(f > lo and f < hi)) f = 0.5 * (lo + hi);
            const w = try probe.gain(f);
            const gv = g(w) orelse break;
            best = .{ .f = f, .w = w };
            if (gv == 0 or hi - lo <= 1e-13 * hi) break;
            if ((gv < 0) == (g_hi < 0)) {
                hi = f;
                g_hi = gv;
                if (side == 1) g_lo *= 0.5;
                side = 1;
            } else {
                lo = f;
                g_lo = gv;
                if (side == -1) g_hi *= 0.5;
                side = -1;
            }
        }
        return best;
    }
    return null;
}

test fill {
    const n = 6;
    var rhs: [4 * n]f64 = @splat(0);
    // `single` never touches the second probe, whatever it holds.
    const far = std.math.maxInt(u32);
    var opts: Options = .{ .sweep = .{ .f_start = 1, .f_stop = 1 }, .probes = .{ .{ .p = 1, .n = 2, .branch = 4 }, .{ .p = far, .n = far, .branch = far } }, .local_gnd = 3 };
    fill(opts, n, &rhs);
    var want: [4 * n]f64 = @splat(0);
    want[1] = 1;
    want[3] = -1;
    want[2 * n + 4] = 1;
    try std.testing.expectEqualSlices(f64, &want, &rhs);

    // `diff`: ±1 on each probe; the local-ground returns cancel.
    opts.mode = .diff;
    opts.probes[1] = .{ .p = 2, .n = 1, .branch = 5 };
    @memset(&rhs, 0);
    fill(opts, n, &rhs);
    want = @splat(0);
    want[1] = 1;
    want[2] = -1;
    want[2 * n + 4] = 1;
    want[2 * n + 5] = -1;
    try std.testing.expectEqualSlices(f64, &want, &rhs);
}

test responses {
    // An antisymmetric pair read in `diff` gives its half circuit: each
    // probe's reading times its weight, averaged.
    const n = 6;
    var x: [4 * n]f64 = @splat(0);
    const opts: Options = .{ .sweep = .{ .f_start = 1, .f_stop = 1 }, .mode = .diff, .probes = .{ .{ .p = 1, .n = 0, .branch = 4 }, .{ .p = 2, .n = 0, .branch = 5 } } };
    // Current injection: branch currents ±0.5+0.25j, node voltages ±2.
    x[4] = 0.5;
    x[n + 4] = 0.25;
    x[5] = -0.5;
    x[n + 5] = -0.25;
    x[1] = 2;
    x[2] = -2;
    // Voltage injection: node voltages ±0.125j.
    x[2 * n + n + 1] = 0.125;
    x[2 * n + n + 2] = -0.125;
    const r = responses(opts, n, &x);
    try std.testing.expectEqual(Complex{ .re = 0.5, .im = 0.25 }, r.a);
    try std.testing.expectEqual(Complex{ .re = 2, .im = 0 }, r.c);
    try std.testing.expectEqual(Complex.zero, r.b);
    try std.testing.expectEqual(Complex{ .re = 0, .im = 0.125 }, r.d);
}

test crossing {
    // A sign change exactly on a grid point returns that point without a
    // solve; no bracket returns null. Neither path touches `probe`.
    const freqs = [_]f64{ 1, 10, 100 };
    const above = [_]Complex{ .{ .re = 4, .im = 0 }, .{ .re = 2, .im = 0 }, .{ .re = 1.5, .im = 0 } };
    var probe: Probe = undefined;
    try std.testing.expectEqual(null, try crossing(&probe, &freqs, &above, logMag));
    const on_grid = [_]Complex{ .{ .re = 4, .im = 0 }, .{ .re = 1, .im = 0 }, .{ .re = 0.5, .im = 0 } };
    const c = (try crossing(&probe, &freqs, &on_grid, logMag)).?;
    try std.testing.expectEqual(@as(f64, 10), c.f);
    // reversedPhase is undefined off the negative real axis, so a loop that
    // stays in the right half plane never brackets a phase crossover.
    try std.testing.expectEqual(null, try crossing(&probe, &freqs, &above, reversedPhase));
}

test "live LSTB margin rounds preserve scalar refinements, failures and cancellation in every order" {
    try marginRoundOracle(.cpu, false);
}

test "GPU LSTB margin rounds execute both sweep and adaptive refinement" {
    const images = @import("gompute_kernels");
    if (comptime images.has_cuda) return marginRoundOracle(.cuda, true);
    if (comptime images.has_hip) return marginRoundOracle(.hip, true);
    return error.SkipZigTest;
}

fn marginRoundOracle(backend: @import("../gpu/context.zig").Request, require_device: bool) !void {
    const t = std.testing;
    const Builder = @import("builder").Builder;
    const Library = @import("device").Library;
    const models = @import("models");
    const count = 7;
    const gains = [_]f64{ 2, 0.1, 10, 2, std.math.nan(f64), 2, 0.1 };
    const Cancel = struct {
        fn checkpoint(_: *anyopaque, _: @import("../worker.zig").Event) error{QueryCancelled}!void {
            return error.QueryCancelled;
        }
    };
    var cancel_context: u8 = 0;
    var library = try Library.init(t.allocator);
    defer library.deinit();
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    var prepared: [count]@import("device").Circuit = undefined;
    var circuits: [count]root.Circuit = undefined;
    var x: [count][]f64 = undefined;
    var contexts: [count]root.RunCtx = undefined;
    var frozen: usize = 0;
    defer for (prepared[0..frozen]) |*p| p.deinit();
    var instantiated: usize = 0;
    defer for (circuits[0..instantiated]) |*c| c.deinit();
    var built: usize = 0;
    defer for (x[0..built]) |values| t.allocator.free(values);
    var options: Options = undefined;
    for (gains, 0..) |gain, p| {
        var builder = try Builder.init(t.allocator, &library);
        const drive = try builder.addNode();
        const a2 = try builder.addNode();
        const b = try builder.addNode();
        const c = try builder.addNode();
        const d = try builder.addNode();
        const e = try builder.addNode();
        const feedback = try builder.addNode();
        try builder.addDevice(models.vcvs, "", &.{ .gain = gain }, &.{}, .{ a2, root.GROUND, root.GROUND, drive });
        try builder.addDevice(models.resistor, "", &.{ .r = 1e3 }, &.{}, .{ a2, b });
        try builder.addDevice(models.capacitor, "", &.{ .c = 1.5915494309189535e-7 }, &.{}, .{ b, root.GROUND });
        try builder.addDevice(models.vcvs, "", &.{ .gain = 2 }, &.{}, .{ c, root.GROUND, b, root.GROUND });
        try builder.addDevice(models.resistor, "", &.{ .r = 1e3 }, &.{}, .{ c, d });
        try builder.addDevice(models.capacitor, "", &.{ .c = 1.5915494309189535e-7 }, &.{}, .{ d, root.GROUND });
        try builder.addDevice(models.vcvs, "", &.{ .gain = 1 }, &.{}, .{ e, root.GROUND, d, root.GROUND });
        try builder.addDevice(models.resistor, "", &.{ .r = 1e3 }, &.{}, .{ e, feedback });
        try builder.addDevice(models.capacitor, "", &.{ .c = 1.5915494309189535e-7 }, &.{}, .{ feedback, root.GROUND });
        const branch = builder.n;
        try builder.addDevice(models.vsource, "", &.{ .dc = 0 }, &.{}, .{ drive, feedback });
        // The independent passive rows keep this oracle above the dense
        // crossover so its hardware half must execute sparse replay.
        for (0..12) |_| {
            const extra = try builder.addNode();
            try builder.addDevice(models.resistor, "", &.{ .r = 700 }, &.{}, .{ extra, root.GROUND });
            try builder.addDevice(models.capacitor, "", &.{ .c = 2e-10 }, &.{}, .{ extra, root.GROUND });
        }
        var permutation: ?[]const u32 = null;
        prepared[p] = try builder.compilePerm(&permutation);
        frozen += 1;
        defer if (permutation) |map| t.allocator.free(map);
        circuits[p] = try root.Circuit.instantiate(&prepared[p], t.allocator);
        instantiated += 1;
        x[p] = try t.allocator.alloc(f64, circuits[p].n);
        built += 1;
        @memset(x[p], 0);
        circuits[p].setSimState(.{ .kind = .ac });
        if (p == 5) circuits[p].progress = .{ .ctx = &cancel_context, .yield_fn = Cancel.checkpoint };
        contexts[p] = .{
            .circuit = &circuits[p],
            .x_op = x[p],
            .probes = &.{},
            .source_node = root.GROUND,
            .source_branch = root.GROUND,
            .allocator = arena.allocator(),
            .scratch_allocator = t.allocator,
        };
        const opts: Options = .{
            .sweep = .{ .f_start = 10, .f_stop = 1e5, .points = 17 },
            .probes = @splat(.{
                .p = if (permutation) |map| map[drive] else drive,
                .n = if (permutation) |map| map[feedback] else feedback,
                .branch = if (permutation) |map| map[branch] else branch,
            }),
            .margins = true,
        };
        if (p == 0) options = opts else {
            try t.expectEqual(options.probes[0], opts.probes[0]);
            try t.expect(circuits[p].n > 16);
        }
    }
    var expected: [count]LaneOutcome = undefined;
    for (&contexts, &expected) |*ctx, *result| result.* = run(ctx, options);
    try t.expectError(error.QueryCancelled, expected[5]);
    var failures: usize = 0;
    for (expected) |result| if (result) |_| {} else |_| {
        failures += 1;
    };
    try t.expect(failures >= 2 and failures < count);
    const orders = [_][count]usize{
        .{ 0, 1, 2, 3, 4, 5, 6 },
        .{ 6, 5, 4, 3, 2, 1, 0 },
        .{ 4, 0, 5, 2, 6, 3, 1 },
    };
    for (orders) |order| {
        var ordered: [count]root.RunCtx = undefined;
        for (&ordered, order) |*ctx, p| ctx.* = contexts[p];
        var actual: [count]LaneOutcome = undefined;
        const stats = try runLanes(&ordered, options, backend, &actual);
        try t.expect(stats.rounds > 1 and stats.refinement_requests > count);
        if (require_device) {
            try t.expect(stats.device_groups > 0);
            try t.expect(stats.device_refinement_lanes > 0);
        }
        for (actual, order) |result, p| {
            if (expected[p]) |oracle| {
                const got = try result;
                try t.expectEqualStrings(oracle.plotname, got.plotname);
                try t.expectEqual(oracle.npoints, got.npoints);
                try t.expectEqual(oracle.data.len, got.data.len);
                for (oracle.data, got.data) |want, value| {
                    if (std.math.isNan(want)) {
                        try t.expect(std.math.isNan(value));
                    } else try t.expectEqual(@as(u64, @bitCast(want)), @as(u64, @bitCast(value)));
                }
            } else |err| try t.expectError(err, result);
        }
    }
}
