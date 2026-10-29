//! Loop stability by double injection (`.lstb`, HSPICE; VACASK `acstb`).
//! Each frequency solves two right-hand sides on one lane factorization: a
//! unit current into the probe's `+` node, and a unit voltage across the
//! probe. VACASK's two-port reading of the four responses gives the DUT
//! y-parameters and the forward, reverse and total loop gain; the total is
//! Tian's loop gain, exact for a bilateral loop and any loading at the
//! break. `diff` and `comm` drive a probe pair +-1 and read the modal half
//! of each response. The margins are `lstb_margins.zig`; a batch of margin
//! queries runs in rounds in `lstb_lanes.zig`.
//! In: a run context and the probes. Out: the complex sweep, or one real
//! row of margins.
//! Cites VACASK coreacstb.cpp; HSPICE [CR .MEASURE LSTB];
//! dev/analysis/stability.md.
//!
//! Contents:
//! - `pub const Options = core.query.Lstb`
//! - `pub const Responses = struct { a, b, c, d }`
//! - `pub const Gains = struct` (`of`)
//! - `pub const LaneOutcome = ErrorOf(run)!Result`
//! - `pub const runGpuLanes = lstb_lanes.runGpuLanes`
//! - `pub fn plan(ctx: *const RunCtx, opts: Options) !?freq.Plan`
//! - `pub fn run(ctx: *const RunCtx, opts: Options) !Result`
//! - `pub fn sweepPlan(scratch: Allocator, opts: Options, n: usize) !freq.Plan`
//! - `pub fn responses(opts: Options, n: usize, x: []const f64) Responses`

const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;
const core = @import("core");
const Complex = core.numerics.Complex;
const FreqSolver = @import("solver").freq_solve.FreqSolver;
const root = @import("../types.zig");
const at = @import("../dc/xf.zig").at;
const freq = @import("freq.zig");
const lstb_margins = @import("lstb_margins.zig");
const lstb_lanes = @import("lstb_lanes.zig");

/// Query options, defined in core/query.zig.
pub const Options = core.query.Lstb;

/// The probe responses in VACASK's names: the probe current (A, B) and the
/// `+` node voltage (C, D) under current (A, C) and voltage (B, D) injection.
pub const Responses = struct { a: Complex, b: Complex, c: Complex, d: Complex };

/// Everything `.lstb` publishes at one frequency.
pub const Gains = struct {
    /// Total loop gain W = Wf + Wr, the return ratio: positive at DC for
    /// negative feedback, unstable at W = -1.
    w: Complex,
    /// The forward and reverse parts of `w`.
    wf: Complex,
    wr: Complex,
    /// The DUT y-parameters seen across the break.
    y11: Complex,
    y12: Complex,
    y21: Complex,
    y22: Complex,

    /// Returns VACASK coreacstb.cpp's gains. The y-parameters divide by C,
    /// so they are infinite when the probe's `+` node is an ideal source.
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

/// One point's `run` outcome from `runGpuLanes`.
pub const LaneOutcome = root.ErrorOf(run)!root.Result;

/// A batch of `.lstb` queries in rounds (`lstb_lanes.runGpuLanes`).
pub const runGpuLanes = lstb_lanes.runGpuLanes;

/// The sweep plot's columns, in row order.
const sweep_names = [_][]const u8{ "frequency", "loop_gain", "loop_gain_forward", "loop_gain_reverse", "y11", "y12", "y21", "y22" };

/// Returns the two-injection sweep; null with `margins`, whose refinement
/// continues the same frequency solver after the sweep and so keeps its
/// sweep live. Fails with error.InvalidProbe as `run` does, before
/// allocating.
pub fn plan(ctx: *const root.RunCtx, opts: Options) !?freq.Plan {
    if (opts.margins) return null;
    return try sweepPlan(ctx.scratch_allocator, opts, ctx.circuit.n);
}

/// Contract entry. The sweep plot is complex (`sweep_names`); with
/// `margins` it is one real row instead (`lstb_margins.Margins`). Fails with
/// error.InvalidProbe when a probe row is out of range.
pub fn run(ctx: *const root.RunCtx, opts: Options) !root.Result {
    const a = ctx.allocator;
    const scratch = ctx.scratch_allocator;
    const ckt = ctx.circuit;
    const n: usize = ckt.n;
    const p = try sweepPlan(scratch, opts, n);
    defer p.deinit(scratch);

    var fs = try FreqSolver.fromCircuit(scratch, ckt, ctx.x_op);
    defer fs.deinit();

    const n_points: usize = opts.sweep.count();
    const loop = try scratch.alloc(Complex, n_points);
    defer scratch.free(loop);

    const cols = sweep_names.len;
    const data = try a.alloc(f64, if (opts.margins) lstb_margins.Margins.names.len else n_points * cols * 2);
    errdefer a.free(data);

    var stream = try freq.Stream.init(scratch, &fs, ckt, ctx.x_op, p.omegas, p.rhs, false);
    defer stream.deinit(scratch);

    stream.pre = ctx.freq_pre;
    while (try stream.next(ckt)) |pt| {
        const g = Gains.of(responses(opts, n, pt.x));
        loop[pt.k] = g.w;
        if (opts.margins) continue;
        const row = data[pt.k * cols * 2 ..][0 .. cols * 2];
        row[0..2].* = .{ p.freqs[pt.k], 0 };
        for ([_]Complex{ g.w, g.wf, g.wr, g.y11, g.y12, g.y21, g.y22 }, 0..) |v, i| row[2 + 2 * i ..][0..2].* = .{ v.re, v.im };
    }
    if (!opts.margins) return .{
        .plotname = "Loop Stability Analysis",
        .varnames = try a.dupe([]const u8, &sweep_names),
        .is_complex = true,
        .npoints = n_points,
        .data = data,
    };
    var probe: lstb_margins.Probe = try .init(scratch, &fs, ckt, ctx.x_op, p.rhs, opts);
    defer probe.deinit(scratch);

    const m = try lstb_margins.Margins.find(&probe, p.freqs, loop);
    data[0..lstb_margins.Margins.names.len].* = m.values();
    return .{
        .plotname = "Loop Stability Margins",
        .varnames = try a.dupe([]const u8, &lstb_margins.Margins.names),
        .is_complex = false,
        .npoints = 1,
        .data = data,
    };
}

/// Returns the two-injection sweep of `run`, `plan` and the lane rounds.
/// Fails with error.InvalidProbe before allocating.
pub fn sweepPlan(scratch: Allocator, opts: Options, n: usize) !freq.Plan {
    try validateProbes(opts, n);
    const rhs = try scratch.alloc(f64, 4 * n);
    errdefer scratch.free(rhs);

    @memset(rhs, 0);
    fill(opts, n, rhs);
    const freqs, const omegas = try freq.axis(scratch, opts.sweep);
    return .{ .freqs = freqs, .omegas = omegas, .rhs = rhs, .adjoint = false };
}

/// Returns the modal responses in `x` (both solutions): each probe's reading
/// times its weight, averaged over the driven probes, so a symmetric pair
/// reads its half circuit. Asserts that `x` holds both stacked solutions.
pub fn responses(opts: Options, n: usize, x: []const f64) Responses {
    assert(x.len == 4 * n);
    const share: f64 = if (opts.mode == .single) 1 else 0.5;
    var r: Responses = .{ .a = .zero, .b = .zero, .c = .zero, .d = .zero };
    for (opts.probes, weights(opts.mode)) |p, w| {
        if (w == 0) continue;
        const s = w * share;
        r.a = r.a.add(at(x, n, p.branch).scale(s));
        r.c = r.c.add(at(x, n, p.p).sub(at(x, n, opts.local_gnd)).scale(s));
        r.b = r.b.add(at(x[2 * n ..], n, p.branch).scale(s));
        r.d = r.d.add(at(x[2 * n ..], n, p.p).sub(at(x[2 * n ..], n, opts.local_gnd)).scale(s));
    }
    return r;
}

// ---- private ----

/// Excitation weight of each probe: `single` drives the first alone, `diff`
/// the pair +1/-1, `comm` +1/+1.
fn weights(mode: Options.Mode) [2]f64 {
    return switch (mode) {
        .single => .{ 1, 0 },
        .diff => .{ 1, -1 },
        .comm => .{ 1, 1 },
    };
}

fn validateProbes(opts: Options, n: usize) error{InvalidProbe}!void {
    const used: usize = if (opts.mode == .single) 1 else 2;
    for (opts.probes[0..used]) |p| if (p.p >= n or p.n >= n or p.branch >= n) return error.InvalidProbe;
    if (opts.local_gnd >= n) return error.InvalidProbe;
}

/// The two stacked 2n right-hand sides: current injection from the local
/// ground, then voltage injection. `rhs` must be zeroed. An unweighted probe
/// (`single`'s second) is never read, as in `responses`.
fn fill(opts: Options, n: usize, rhs: []f64) void {
    assert(rhs.len == 4 * n);
    for (opts.probes, weights(opts.mode)) |p, e| {
        if (e == 0) continue;
        if (p.p != root.GROUND) rhs[p.p] += e;
        if (opts.local_gnd != root.GROUND) rhs[opts.local_gnd] -= e;
        rhs[2 * n + p.branch] += e;
    }
}

test {
    _ = lstb_margins;
    _ = lstb_lanes;
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

    // `diff`: +-1 on each probe; the local-ground returns cancel.
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
    // Current injection: branch currents +-0.5+0.25j, node voltages +-2.
    x[4] = 0.5;
    x[n + 4] = 0.25;
    x[5] = -0.5;
    x[n + 5] = -0.25;
    x[1] = 2;
    x[2] = -2;
    // Voltage injection: node voltages +-0.125j.
    x[2 * n + n + 1] = 0.125;
    x[2 * n + n + 2] = -0.125;
    const r = responses(opts, n, &x);
    try std.testing.expectEqual(Complex{ .re = 0.5, .im = 0.25 }, r.a);
    try std.testing.expectEqual(Complex{ .re = 2, .im = 0 }, r.c);
    try std.testing.expectEqual(Complex.zero, r.b);
    try std.testing.expectEqual(Complex{ .re = 0, .im = 0.125 }, r.d);
}
