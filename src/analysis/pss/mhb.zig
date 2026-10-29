//! Harmonic balance over any number of tones: a circuit and its tones in,
//! each unknown's spectrum over the kept mixing products out, published as
//! one row per line. This file is the query spine and the module's surface:
//! the spectrum (`mhb_spectrum.zig`), the transform pair
//! (`mhb_transform.zig`), the Newton with its matrix-free GMRES
//! (`mhb_newton.zig`) and the GPU batch (`mhb_gpu.zig`) sit beside it.
//!
//! Unknowns are node-major real blocks [dc, c_1, s_1, ..., c_{m-1}, s_{m-1}]
//! with x(t) = dc + Σ c_j cos(ω_j t) + s_j sin(ω_j t), the layout
//! `hb.solveSpectrum` uses for one tone, so one tone is the lines 0..K.
//!
//! Cites: HSPICE `.hb` [CR .HB], VACASK `lib/spurs.cpp` and
//! `lib/corehbcoloc.cpp`, Verilog-A LRM §4.6.1, #83.
//!
//! Contents:
//! - `pub const Options = @import("core").query.Hb`
//! - `pub const LaneOutcome = @import("hb.zig").LaneOutcome`
//! - `pub const Spectrum = mhb_spectrum.Spectrum`
//! - `pub const max_tones = mhb_spectrum.max_tones`
//! - `pub const spectrum = mhb_spectrum.spectrum`
//! - `pub const gvProduct = mhb_newton.gvProduct`
//! - `pub const LineSearch = mhb_newton.LineSearch`
//! - `pub const dcSeed = mhb_newton.dcSeed`
//! - `pub const solve = mhb_newton.solve`
//! - `pub const gpu_test_access = mhb_gpu.gpu_test_access`
//! - `pub const runGpuLanes = mhb_gpu.runGpuLanes`
//! - `pub const Orbit = struct`
//! - `Orbit`: `deinit`
//! - `pub const test_access` (tests only)
//! - `pub fn querySpectrum(gpa: std.mem.Allocator, opts: Options, tones: *[max_tones]f64) !Spectrum`
//! - `pub fn run(ctx: *const root.RunCtx, opts: Options) !root.Result`
//! - `pub fn publish(ctx: *const root.RunCtx, opts: Options, spec: Spectrum, x_hat: []const f64) !root.Result`
//! - `pub fn orbit(ckt: *root.Circuit, opts: Options, allocator: std.mem.Allocator) !Orbit`
//! - `pub fn lineCount(gpa: std.mem.Allocator, opts: Options) !usize`
//! - `pub fn solveOneTone(ckt: *root.Circuit, x_hat: []f64, options: Options, allocator: std.mem.Allocator) !SolveResult`

const std = @import("std");
const root = @import("../types.zig");
const take = root.take;
const pss = @import("pss.zig");
const num = @import("core").numerics;
const solvers = @import("solver");
const converger = solvers.converger;
const dense_lu = solvers.dense_lu;
const FreqSolver = solvers.freq_solve.FreqSolver;
const Gmres = solvers.gmres.Gmres;
const Gcrodr = solvers.gcrodr.Gcrodr;
const SolveResult = pss.SolveResult;
const mhb_gpu = @import("mhb_gpu.zig");
const mhb_newton = @import("mhb_newton.zig");
const mhb_spectrum = @import("mhb_spectrum.zig");
const mhb_transform = @import("mhb_transform.zig");
const Transform = mhb_transform.Transform;
const Fold = mhb_transform.Fold;
const transform = mhb_transform.transform;
const transformSamples = mhb_transform.transformSamples;
const idft = mhb_transform.idft;
const dft = mhb_transform.dft;
const takePlanes = mhb_newton.takePlanes;
const sampleDt = mhb_newton.sampleDt;
const sample = mhb_newton.sample;

/// The `.hb` query, read for its tone list (`f0` then `extra_tones`).
pub const Options = @import("core").query.Hb;

/// One point's `.hb` outcome, as `hb.run` returns it.
pub const LaneOutcome = @import("hb.zig").LaneOutcome;

/// The kept spectral lines (`mhb_spectrum.zig`).
pub const Spectrum = mhb_spectrum.Spectrum;

/// The most tones a spectrum takes.
pub const max_tones = mhb_spectrum.max_tones;

/// Builds the kept lines of a tone set (`mhb_spectrum.zig`).
pub const spectrum = mhb_spectrum.spectrum;

/// The sampled sparse product G(t)·v per instant (`mhb_newton.zig`).
pub const gvProduct = mhb_newton.gvProduct;

/// The Newton step's backtracking state (`mhb_newton.zig`).
pub const LineSearch = mhb_newton.LineSearch;

/// Seeds a spectrum with the DC solution (`mhb_newton.zig`).
pub const dcSeed = mhb_newton.dcSeed;

/// One point's harmonic-balance Newton (`mhb_newton.zig`).
pub const solve = mhb_newton.solve;

/// The GPU batch's test counters (`mhb_gpu.zig`).
pub const gpu_test_access = mhb_gpu.gpu_test_access;

/// Driven sparse points as one GPU batch (`mhb_gpu.zig`).
pub const runGpuLanes = mhb_gpu.runGpuLanes;

/// A converged multi-tone solution sampled for linearization: the
/// spectrum, its transform, and x, G and C at every collocation instant.
pub const Orbit = struct {
    spec: Spectrum,
    tr: Transform,
    /// Node-major, `tr.nt` samples each.
    x_td: []const f64,
    /// Slot-major over the circuit's pattern, `tr.nt` samples each.
    g_td: []const f64,
    c_td: []const f64,
    buf: []f64,

    /// Frees the orbit; `gpa` is the allocator `orbit` took.
    pub fn deinit(self: Orbit, gpa: std.mem.Allocator) void {
        self.spec.deinit(gpa);
        gpa.free(self.buf);
    }
};

/// Private implementation access for the analysis test suite; void outside tests.
pub const test_access = if (@import("builtin").is_test) .{
    .transform = transform,
    .transformSamples = transformSamples,
    .Fold = Fold,
    .idft = idft,
    .dft = dft,
} else {};

/// `spectrum` of `opts`' tones, f0 first, which land in `tones[0..spec.tones]`.
/// error.InvalidQueryOptions for more than `max_tones` tones or a harmonic
/// list that does not match them, and as `spectrum` otherwise.
pub fn querySpectrum(gpa: std.mem.Allocator, opts: Options, tones: *[max_tones]f64) !Spectrum {
    const count = 1 + opts.extra_tones.len;
    if (count > max_tones or opts.extra_harmonics.len != opts.extra_tones.len) return error.InvalidQueryOptions;
    var nharms: [max_tones]u16 = undefined;
    tones[0] = opts.f0;
    nharms[0] = opts.n_harmonics;
    @memcpy(tones[1..count], opts.extra_tones);
    @memcpy(nharms[1..count], opts.extra_harmonics);
    return spectrum(gpa, tones[0..count], nharms[0..count], opts.intmodmax, opts.subharms);
}

/// Contract entry for `.hb TONES=` (any tone count) and the positional
/// card with several tones: one row per spectral line, DC first then by
/// frequency. With `opts.phasors` the rows are complex, (frequency, each
/// probe's phasor c - j·s), x(t) = Re{X·e^(jωt)}, DC real; otherwise
/// (frequency, each probe's amplitude sqrt(c² + s²)) with a signed DC.
/// Non-convergence is error.HbDidNotConverge.
pub fn run(ctx: *const root.RunCtx, opts: Options) !root.Result {
    const scratch = ctx.scratch_allocator;
    var tones: [max_tones]f64 = undefined;
    const spec = try querySpectrum(scratch, opts, &tones);
    defer spec.deinit(scratch);
    if (spec.freqs.len < 2) return error.InvalidQueryOptions;
    const nf = 2 * spec.freqs.len - 1;
    const x_hat = try scratch.alloc(f64, @as(usize, ctx.circuit.n) * nf);
    defer scratch.free(x_hat);
    const st = try solve(ctx.circuit, x_hat, spec, tones[0..spec.tones], opts, scratch);
    if (!st.converged) return error.HbDidNotConverge;

    return publish(ctx, opts, spec, x_hat);
}

/// `run`'s published rows from the converged `x_hat`, in `ctx.allocator`.
pub fn publish(ctx: *const root.RunCtx, opts: Options, spec: Spectrum, x_hat: []const f64) !root.Result {
    const a = ctx.allocator;
    const nf = 2 * spec.freqs.len - 1;
    const names = try root.probeNames(ctx, "frequency");
    const width: usize = if (opts.phasors) 2 else 1;
    const ncols = names.len * width;
    const data = try a.alloc(f64, spec.freqs.len * ncols);
    @memset(data, 0);
    for (spec.freqs, 0..) |f, j| {
        const row = data[j * ncols ..][0..ncols];
        row[0] = f;
        for (ctx.probes, 1..) |node, p| {
            const x = x_hat[node * nf ..][0..nf];
            const v = row[p * width ..][0..width];
            if (j == 0) {
                v[0] = x[0];
            } else if (opts.phasors) {
                v[0] = x[2 * j - 1];
                v[1] = -x[2 * j];
            } else v[0] = std.math.hypot(x[2 * j - 1], x[2 * j]);
        }
    }
    return .{
        .plotname = "Harmonic Balance",
        .varnames = names,
        .is_complex = opts.phasors,
        .npoints = spec.freqs.len,
        .data = data,
    };
}

/// Solves `opts`' harmonic balance and samples the solution for
/// linearization. error.HbDidNotConverge when Newton stops short,
/// error.InvalidQueryOptions when the spectrum has fewer than two lines,
/// and as `solve` and `spectrum` otherwise. Free with `Orbit.deinit` on
/// `allocator`.
pub fn orbit(ckt: *root.Circuit, opts: Options, allocator: std.mem.Allocator) !Orbit {
    var tones: [max_tones]f64 = undefined;
    const spec = try querySpectrum(allocator, opts, &tones);
    errdefer spec.deinit(allocator);
    const tone_list = tones[0..spec.tones];
    if (spec.freqs.len < 2) return error.InvalidQueryOptions;
    const n: usize = ckt.n;
    const nnz: usize = ckt.nnz;
    const nf = 2 * spec.freqs.len - 1;
    const nt = transformSamples(spec);
    const len = n * nf + nt + 2 * nt * nf + 3 * n * nt + 2 * nnz * nt + n;
    const buf = try allocator.alloc(f64, len);
    errdefer allocator.free(buf);
    var off: usize = 0;
    const x_hat = take(buf, &off, n * nf);
    const tr_buf = take(buf, &off, nt + 2 * nt * nf);
    const p = takePlanes(buf, &off, n * nt, nnz * nt);
    const x_sample = take(buf, &off, n);
    std.debug.assert(off == len);
    const st = try solve(ckt, x_hat, spec, tone_list, opts, allocator);
    if (!st.converged) return error.HbDidNotConverge;
    const tr = try transform(spec, tone_list, tr_buf, allocator);
    @memset(p.c, 0);
    sample(ckt, tr, nf, sampleDt(spec), x_hat, x_sample, p);
    return .{ .spec = spec, .tr = tr, .x_td = p.x, .g_td = p.g, .c_td = p.c, .buf = buf };
}

/// Spectral lines of `opts`' spectrum, DC included. Fails as `querySpectrum`.
pub fn lineCount(gpa: std.mem.Allocator, opts: Options) !usize {
    var tones: [max_tones]f64 = undefined;
    const spec = try querySpectrum(gpa, opts, &tones);
    defer spec.deinit(gpa);
    return spec.freqs.len;
}

/// Solves one tone at f0 with K = `options.n_harmonics` through `solve`;
/// `x_hat` in `hb.solveSpectrum`'s layout. Asserts K >= 1.
pub fn solveOneTone(ckt: *root.Circuit, x_hat: []f64, options: Options, allocator: std.mem.Allocator) !SolveResult {
    const tones = [_]f64{options.f0};
    const spec = try spectrum(allocator, &tones, &.{options.n_harmonics}, 0, 1);
    defer spec.deinit(allocator);
    return solve(ckt, x_hat, spec, &tones, options, allocator);
}
