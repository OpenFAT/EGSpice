//! Small-signal noise of the device generators, measured at an output node.
//! Each frequency is one adjoint solve (a lane of `freq.Stream`); the source
//! PSDs come from the devices' noise models. Densities are V^2/Hz internally
//! and the published plots are V/sqrt(Hz) and V rms. The same sweep feeds
//! `.acphasenoise` (dBc/Hz), the per-generator contribution columns
//! (`noise_columns.zig`) and the sampler-folded `onoise_sampled` column.
//! In: a run context and `.noise` options. Out: one of three real plots.
//! Cites ngspice noisean.c (:376, :436-439, :487-488), ninteg.c (:24-45),
//! noisedef.h (:105-113) and resnoise.c (:145-150); Verilog-AMS LRM §4.6.4;
//! IEEE 1139 (L(f) = S(f)/2); #83, #84; dev/analysis/ac-small-signal-noise.md
//! and dev/analysis/phase-noise.md.
//!
//! Contents:
//! - `pub const Options = core.query.Noise`
//! - `pub const test_access`
//! - `pub fn plan(ctx: *const RunCtx, opts: Options) !freq.Plan`
//! - `pub fn run(ctx: *const RunCtx, opts: Options) !Result`
//! - `pub fn columnCount(gpa: Allocator, batches: []const Batch, types: []const DeviceType, opts: Options) !usize`
//! - `pub fn sourcePsd(src: NoiseSource, f: f64) f64`
//! - `pub fn transfer(rows: []const NoiseSource, y: []const f64, n: usize) [2]f64`

const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;
const abi = @import("device").abi;
const NoiseSample = @import("core").query.NoiseSample;
const FreqSolver = @import("solver").freq_solve.FreqSolver;
const root = @import("../types.zig");
const freq = @import("freq.zig");
const noise_columns = @import("noise_columns.zig");
const Columns = noise_columns.Columns;
const NoiseSource = root.NoiseSource;

/// Query options, defined in core/query.zig.
pub const Options = @import("core").query.Noise;

/// Private implementation access for the analysis test suite.
pub const test_access = if (@import("builtin").is_test) .{
    .Band = Band,
    .nintegrate = nintegrate,
} else {};

// ngspice noisedef.h:105-113.
const n_minlog = 1e-38;
const n_intfthresh = 1e-10;
const n_intuselog = 1e-10;
/// N_MINGAIN (noisean.c:487-488): the floor of the input-referred gain.
const gain_sq_min = 1e-20;

/// The per-interval geometry `nintegrate` needs (noisean.c:436-439).
const Band = struct { del_freq: f64, del_ln_freq: f64, ln_freq: f64, ln_last_freq: f64 };

comptime {
    assert(@sizeOf(Band) == 32);
}

/// Band integrals in V^2: output-referred, and referred back through the
/// gain to the input source's terminals (zero when the deck named no input).
const Integrals = struct { onoise: f64, inoise: f64 };

/// Per-generator results for the contribution columns: `dens` is
/// point-major (`n_points * n_sources`, V^2/Hz at the output); `out` and
/// `in` are each generator's own band integral in V^2. A correlated group's
/// values land on its first row only, so the caller zeroes all three.
const PerSource = struct { dens: []f64, out: []f64, in: []f64 };

/// One sweep's output tables, each `n_points` long.
const Spectra = struct { freqs: []f64, density: []f64, in_density: []f64 };

/// Returns the main sweep `run` solves (`sweep`'s adjoint stream): the
/// output's unit adjoint drive at every omega of `opts`. The sampled
/// re-sweep is not part of it.
pub fn plan(ctx: *const root.RunCtx, opts: Options) !freq.Plan {
    const scratch = ctx.scratch_allocator;
    const rhs = try scratch.alloc(f64, 2 * @as(usize, ctx.circuit.n));
    errdefer scratch.free(rhs);

    root.outputDrive(rhs, opts.out_node, opts.out_neg);
    const freqs, const omegas = try freq.axis(scratch, opts.sweep);
    return .{ .freqs = freqs, .omegas = omegas, .rhs = rhs, .adjoint = true };
}

/// Contract entry: the device generators measured at `opts.out_node`, with
/// no drive source. Real; either the spectrum (frequency, onoise_spectrum,
/// inoise_spectrum) or, with `opts.integrated`, one row of rms totals. With
/// `opts.contributions` each device instance adds its columns ahead of the
/// totals, as ngspice's `.noise ... pts` does: per generator name and per
/// instance, `onoise_<inst>_<gen>` and `onoise_<inst>` in the spectrum,
/// `v(onoise_total_<inst>_<gen>)`, `v(inoise_total_<inst>_<gen>)` and the
/// instance sums in the integrated plot. `opts.phase` replaces all of it
/// with the single-sideband phase noise in dBc/Hz, and `opts.sample` adds
/// the folded `onoise_sampled` column to the spectrum.
pub fn run(ctx: *const root.RunCtx, opts: Options) !root.Result {
    const scratch = ctx.scratch_allocator;
    const srcs = try ctx.circuit.collectNoiseSources(ctx.x_op, scratch);
    defer scratch.free(srcs);

    const n_points: usize = opts.sweep.count();
    const work = try scratch.alloc(f64, n_points * 3);
    defer scratch.free(work);

    const sp: Spectra = .{ .freqs = work[0..n_points], .density = work[n_points..][0..n_points], .in_density = work[2 * n_points ..][0..n_points] };
    const per_buf = try scratch.alloc(f64, if (opts.contributions) srcs.len * (n_points + 2) else 0);
    defer scratch.free(per_buf);

    var columns: Columns = .{};
    var per_source: ?PerSource = null;
    if (opts.contributions) {
        columns = try Columns.build(ctx.allocator, ctx.circuit.batches, ctx.circuit.batch_types, opts.card_names, srcs.len);
        @memset(per_buf, 0);
        per_source = .{ .dens = per_buf[0 .. srcs.len * n_points], .out = per_buf[srcs.len * n_points ..][0..srcs.len], .in = per_buf[srcs.len * (n_points + 1) ..] };
    }
    const integrated = try sweep(ctx.circuit, ctx.x_op, srcs, sp, per_source, opts, scratch, ctx.freq_pre);
    // ngspice's two noise plots, with ngspice's names and units: the curves
    // are amplitude spectra (V/sqrt(Hz)) and the totals are V rms. `inoise`
    // is the same noise referred to the input source's terminals.
    if (opts.phase) return phasePlot(ctx.allocator, sp);
    if (opts.integrated) return integratedPlot(ctx, opts, columns, per_source, integrated);
    const folded = if (opts.sample) |s| try sampled(ctx.circuit, ctx.x_op, srcs, sp.freqs, s, opts, scratch) else null;
    defer if (folded) |f| scratch.free(f);

    return spectrumPlot(ctx.allocator, sp, columns, per_source, srcs.len, folded);
}

/// Returns the columns `run` publishes for `opts` over `batches`, which
/// `session.schemaOf` reports before the query runs (#83).
pub fn columnCount(gpa: Allocator, batches: []const abi.Batch, types: []const abi.DeviceType, opts: Options) Columns.Error!usize {
    if (opts.phase) return 2;
    var n_cols: usize = 0;
    if (opts.contributions) {
        var arena = std.heap.ArenaAllocator.init(gpa);
        defer arena.deinit();

        n_cols = (try Columns.build(arena.allocator(), batches, types, opts.card_names, noise_columns.sourceCount(batches))).labels.len;
    }
    return if (opts.integrated) 2 * n_cols + 2 else n_cols + 3 + @intFromBool(opts.sample != null);
}

/// Returns the source's unit-coefficient PSD at `f`: white plus flicker /
/// f^ef plus its table, in A^2/Hz. At f <= 0 the flicker half has no value
/// and drops out. `coeff` is not in it: it scales the transfer (`transfer`).
pub fn sourcePsd(src: NoiseSource, f: f64) f64 {
    const tab = src.tableAt(f);
    if (src.flicker == 0 or f <= 0) return src.white + tab;
    return src.white + src.flicker / std.math.pow(f64, f, src.ef) + tab;
}

/// Returns the (re, im) transfer of one correlated group `rows` to the
/// output, the sum of coeff (y[node_p] - y[node_n]), from the stacked
/// adjoint `y` of `n` unknowns. Asserts that `y` is stacked-real.
pub fn transfer(rows: []const NoiseSource, y: []const f64, n: usize) [2]f64 {
    assert(y.len >= 2 * n);
    var h: [2]f64 = .{ 0, 0 };
    for (rows) |src| {
        if (src.node_p != root.GROUND) {
            h[0] += src.coeff * y[src.node_p];
            h[1] += src.coeff * y[n + src.node_p];
        }
        if (src.node_n != root.GROUND) {
            h[0] -= src.coeff * y[src.node_n];
            h[1] -= src.coeff * y[n + src.node_n];
        }
    }
    return h;
}

// ---- private ----

/// `exp`, linear past 700 so a steep fit cannot overflow (ninteg.c:24).
fn limexp(x: f64) f64 {
    return if (x > 700.0) @exp(@as(f64, 700.0)) * (1.0 + x - 700.0) else @exp(x);
}

/// Integral of the power law `a * f^exponent` fitted between two points of
/// one source's log-log spectrum (ninteg.c:27-45 Nintegrate). A near-flat
/// slope degenerates to the rectangle rule and a slope near -1 to the
/// logarithmic one. The fit has to be per source because a sum of two
/// different power laws is not a power law.
fn nintegrate(dens: f64, ln_dens: f64, ln_last_dens: f64, b: Band) f64 {
    const exponent = (ln_dens - ln_last_dens) / b.del_ln_freq;
    if (@abs(exponent) < n_intfthresh) return dens * b.del_freq;
    const a = limexp(ln_dens - exponent * b.ln_freq);
    const e1 = exponent + 1.0;
    if (@abs(e1) < n_intuselog) return a * (b.ln_freq - b.ln_last_freq);
    return a * (limexp(e1 * b.ln_freq) - limexp(e1 * b.ln_last_freq)) / e1;
}

/// Fills `sp` (V^2/Hz) and returns the band integrals. `in_density` is all
/// zeros unless `options.in_branch` or `options.in_nodes` names a source.
/// `per_source`, when given, also receives every generator's density and
/// integrals. Every work buffer comes from and returns to `gpa`;
/// error.QueryCancelled unwinds at a checkpoint. `pre` is the stream's
/// solutions solved up front (`freq.Stream.pre`), or empty.
fn sweep(ckt: *root.Circuit, x_op: []const f64, noise_sources: []const NoiseSource, sp: Spectra, per_source: ?PerSource, options: Options, gpa: Allocator, pre: []const f64) !Integrals {
    const n: usize = ckt.n;
    const n_points = sp.freqs.len;
    assert(sp.density.len == n_points);
    assert(sp.in_density.len == n_points);
    // Adjoint: A^H y = e_out per omega. The conjugate drops out of |H|^2, so
    // the stacked-real transpose solve is enough.
    var fs = try FreqSolver.fromCircuit(gpa, ckt, x_op);
    defer fs.deinit();

    const omegas = try gpa.alloc(f64, n_points);
    defer gpa.free(omegas);

    options.sweep.fill(sp.freqs, omegas);
    // Unit excitation at the output: e_pos - e_neg for `v(a,b)`.
    const e = try gpa.alloc(f64, 2 * n);
    defer gpa.free(e);

    root.outputDrive(e, options.out_node, options.out_neg);
    var stream = try freq.Stream.init(gpa, &fs, ckt, x_op, omegas, e, true);
    defer stream.deinit(gpa);

    stream.pre = pre;
    // ln of each source's output density at the previous point (ngspice's
    // `nVar[LNLSTDENS][i]`), the other end of the per-source fit; then the
    // same densities unlogged. The input side divides the previous point's
    // output density by the CURRENT gain, as every ngspice device does
    // (resnoise.c:145-150): the gain is held constant across the interval.
    const ln_last = try gpa.alloc(f64, 2 * noise_sources.len);
    defer gpa.free(ln_last);

    const dens_last = ln_last[noise_sources.len..];
    const referred = options.in_branch != null or options.in_nodes != null;
    var integrated: Integrals = .{ .onoise = 0, .inoise = 0 };
    // noisean.c:376 sets lstFreq = freq before the loop, so the first point
    // has delFreq == 0 and only seeds the history.
    var prev_freq: f64 = if (n_points != 0) sp.freqs[0] else 0;
    while (try stream.next(ckt)) |pt| {
        const k = pt.k;
        const f = sp.freqs[k];
        const y = pt.x;
        const ln_freq = @log(@max(f, n_minlog));
        const ln_prev = @log(@max(prev_freq, n_minlog));
        const band: Band = .{ .del_freq = f - prev_freq, .ln_freq = ln_freq, .ln_last_freq = ln_prev, .del_ln_freq = ln_freq - ln_prev };
        const gain_sq: f64 = if (referred) inputGainSq(options, y, n) else 0;
        var total_density: f64 = 0;
        var total_in_density: f64 = 0;
        // One density per correlated group, kept at its first row; the other
        // rows of a group stay zero, so the per-source arrays keep one slot
        // per row. `groupEnd` always advances, so this walks each row once.
        var i: usize = 0;
        while (i < noise_sources.len) {
            const end = NoiseSource.groupEnd(noise_sources, i);
            assert(end > i);
            defer i = end;
            const h = transfer(noise_sources[i..end], y, n);
            const dens = (h[0] * h[0] + h[1] * h[1]) * sourcePsd(noise_sources[i], f);
            total_density += dens;
            if (per_source) |ps| ps.dens[k * noise_sources.len + i] = dens;
            const ln_dens = @log(@max(dens, n_minlog));
            if (band.del_freq != 0) {
                const part = nintegrate(dens, ln_dens, ln_last[i], band);
                integrated.onoise += part;
                if (per_source) |ps| ps.out[i] += part;
            }
            ln_last[i] = ln_dens;
            if (!referred) continue;
            const dens_in = dens / gain_sq;
            total_in_density += dens_in;
            const ln_dens_in = @log(@max(dens_in, n_minlog));
            if (band.del_freq != 0) {
                const part = nintegrate(dens_in, ln_dens_in, @log(@max(dens_last[i] / gain_sq, n_minlog)), band);
                integrated.inoise += part;
                if (per_source) |ps| ps.in[i] += part;
            }
            dens_last[i] = dens;
        }
        sp.density[k] = total_density;
        sp.in_density[k] = total_in_density;
        prev_freq = f;
    }
    return integrated;
}

/// |gain|^2 from the input source to the output without a second solve: the
/// adjoint y is the transfer row, so a unit drive on the input branch gives
/// v_out = y[in_branch] (e_out^T A^-1 e_in = (A^-T e_out)^T e_in). A current
/// input drives its node pair: |y[n+] - y[n-]|. Floored at N_MINGAIN.
fn inputGainSq(options: Options, y: []const f64, n: usize) f64 {
    var g_re: f64 = 0;
    var g_im: f64 = 0;
    if (options.in_branch) |br| {
        g_re = y[br];
        g_im = y[n + br];
    } else for (options.in_nodes.?, [2]f64{ 1, -1 }) |row, sign| if (row != root.GROUND) {
        g_re += sign * y[row];
        g_im += sign * y[n + row];
    };
    return @max(g_re * g_re + g_im * g_im, gain_sq_min);
}

/// The output density at each of `freqs` as a sampler at `s.fs` sees it:
/// the density at every |f + k fs| up to `s.max_fold * fs`, each weighted
/// by the integrator's sinc^2(pi g beta / fs), summed. Caller frees with
/// `gpa`.
fn sampled(ckt: *root.Circuit, x_op: []const f64, srcs: []const NoiseSource, freqs: []const f64, s: NoiseSample, opts: Options, gpa: Allocator) ![]f64 {
    const f_top = s.max_fold * s.fs;
    const k_max: i64 = @intFromFloat(@ceil(s.max_fold) + 1);
    var grid: std.ArrayList(f64) = try .initCapacity(gpa, freqs.len * @as(usize, @intCast(2 * k_max + 1)));
    defer grid.deinit(gpa);

    for (freqs) |f| {
        var k = -k_max;
        while (k <= k_max) : (k += 1) {
            const g = @abs(f + @as(f64, @floatFromInt(k)) * s.fs);
            if (g > 0 and g <= f_top) grid.appendAssumeCapacity(g);
        }
    }
    std.mem.sort(f64, grid.items, {}, std.sort.asc(f64));
    var unique: usize = 0;
    for (grid.items) |g| {
        if (unique != 0 and grid.items[unique - 1] == g) continue;
        grid.items[unique] = g;
        unique += 1;
    }
    const at = grid.items[0..unique];
    const work = try gpa.alloc(f64, 3 * unique);
    defer gpa.free(work);

    var o = opts;
    o.sweep = .{ .f_start = if (unique != 0) at[0] else 0, .f_stop = if (unique != 0) at[unique - 1] else 0, .points = @intCast(unique), .kind = .poi, .list = at };
    o.contributions = false;
    const fine: Spectra = .{ .freqs = work[0..unique], .density = work[unique..][0..unique], .in_density = work[2 * unique ..] };
    if (unique != 0) _ = try sweep(ckt, x_op, srcs, fine, null, o, gpa, &.{}); // Only the density is folded.
    const out = try gpa.alloc(f64, freqs.len);
    for (freqs, out) |f, *total| {
        total.* = 0;
        var k = -k_max;
        while (k <= k_max) : (k += 1) {
            const g = @abs(f + @as(f64, @floatFromInt(k)) * s.fs);
            if (!(g > 0 and g <= f_top)) continue;
            const i = std.sort.lowerBound(f64, at, g, orderF64);
            const x = std.math.pi * g * s.beta / s.fs;
            const sinc = if (x == 0) 1 else @sin(x) / x;
            total.* += fine.density[i] * sinc * sinc;
        }
    }
    return out;
}

fn orderF64(a: f64, b: f64) std.math.Order {
    return std.math.order(a, b);
}

/// `.acphasenoise`: the output PSD is in rad^2/Hz, and phase noise is the
/// single-sideband L(f) = S(f)/2 (IEEE 1139; the HSPICE manual gives no
/// formula, unconfirmed).
fn phasePlot(a: Allocator, sp: Spectra) Allocator.Error!root.Result {
    const names = try a.dupe([]const u8, &.{ "frequency", "phnoise" });
    const data = try a.alloc(f64, 2 * sp.freqs.len);
    for (sp.freqs, sp.density, 0..) |f, d, i| {
        data[2 * i] = f;
        data[2 * i + 1] = 10 * std.math.log10(d / 2);
    }
    return .{ .plotname = "AC Phase Noise Analysis", .varnames = names, .is_complex = false, .npoints = sp.freqs.len, .data = data };
}

/// The integrated plot: per-contribution rms totals, then the two totals.
fn integratedPlot(ctx: *const root.RunCtx, opts: Options, columns: Columns, per_source: ?PerSource, integrated: Integrals) Allocator.Error!root.Result {
    const a = ctx.allocator;
    const n_cols = columns.labels.len;
    const names = try a.alloc([]const u8, 2 * n_cols + 2);
    const data = try a.alloc(f64, names.len);
    if (per_source) |ps| {
        const sums = try ctx.scratch_allocator.alloc(f64, 2 * n_cols);
        defer ctx.scratch_allocator.free(sums);

        @memset(sums, 0);
        columns.sum(ps.out, sums[0..n_cols]);
        columns.sum(ps.in, sums[n_cols..]);
        for (columns.labels, 0..) |label, c| {
            names[2 * c] = try a.print("v(onoise_total_{s})", .{label});
            names[2 * c + 1] = try a.print("v(inoise_total_{s})", .{label});
            data[2 * c] = @sqrt(sums[c]);
            data[2 * c + 1] = @sqrt(sums[n_cols + c]);
        }
    }
    names[2 * n_cols] = "v(onoise_total)";
    // ngspice types the input-referred total by the input source: a current
    // source's is i(inoise_total) (noisean.c, #84).
    names[2 * n_cols + 1] = if (opts.in_nodes != null) "i(inoise_total)" else "v(inoise_total)";
    data[2 * n_cols] = @sqrt(integrated.onoise);
    data[2 * n_cols + 1] = @sqrt(integrated.inoise);
    return .{ .plotname = "Integrated Noise", .varnames = names, .is_complex = false, .npoints = 1, .data = data };
}

/// The spectrum plot: frequency, the contribution columns, the two spectra,
/// and the folded column when the sampler re-sweep ran.
fn spectrumPlot(a: Allocator, sp: Spectra, columns: Columns, per_source: ?PerSource, n_sources: usize, folded: ?[]const f64) Allocator.Error!root.Result {
    const n_points = sp.freqs.len;
    const n_cols = columns.labels.len;
    const width = n_cols + 3 + @intFromBool(folded != null);
    const names = try a.alloc([]const u8, width);
    names[0] = "frequency";
    for (columns.labels, names[1 .. 1 + n_cols]) |label, *name| name.* = try a.print("onoise_{s}", .{label});
    names[n_cols + 1] = "onoise_spectrum";
    names[n_cols + 2] = "inoise_spectrum";
    if (folded != null) names[width - 1] = "onoise_sampled";
    const data = try a.alloc(f64, n_points * width);
    for (0..n_points) |i| {
        const row = data[i * width ..][0..width];
        row[0] = sp.freqs[i];
        const cols = row[1 .. 1 + n_cols];
        if (per_source) |ps| {
            @memset(cols, 0);
            columns.sum(ps.dens[i * n_sources ..][0..n_sources], cols);
            for (cols) |*v| v.* = @sqrt(v.*);
        }
        row[n_cols + 1] = @sqrt(sp.density[i]);
        row[n_cols + 2] = @sqrt(sp.in_density[i]);
        if (folded) |f| row[width - 1] = @sqrt(f[i]);
    }
    return .{ .plotname = "Noise Spectral Density Curves", .varnames = names, .is_complex = false, .npoints = n_points, .data = data };
}
