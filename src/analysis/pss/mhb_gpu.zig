//! Driven sparse HB/MHB points as one GPU batch: lane contexts in, one
//! `Result` or error per point out. Points share sampled evaluation and
//! resident Krylov work; every point keeps its own Newton iteration and
//! line search (`mhb_newton.zig`), and a point the batch cannot take runs
//! `hb.run` alone.
//! Cites: none (dev/analysis/lane-batched-simulation.md).
//!
//! Contents:
//! - `pub const gpu_test_access` (tests only)
//! - `gpu_test_access`: `admitted`, `backtracks`
//! - `pub fn runGpuLanes(ctxs: []const root.RunCtx, options: Options, out: []LaneOutcome) void`

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
const mhb = @import("mhb.zig");
const mhb_newton = @import("mhb_newton.zig");
const mhb_spectrum = @import("mhb_spectrum.zig");
const mhb_transform = @import("mhb_transform.zig");
const Options = mhb.Options;
const max_tones = mhb_spectrum.max_tones;
const transformSamples = mhb_transform.transformSamples;
const idft = mhb_transform.idft;
const takePlanes = mhb_newton.takePlanes;
const sample = mhb_newton.sample;
const gmres_tol = mhb_newton.gmres_tol;
const gmres_restart = mhb_newton.gmres_restart;
const gmres_max_restarts = mhb_newton.gmres_max_restarts;
const Newton = mhb_newton.Newton;
const LaneOutcome = mhb.LaneOutcome;
const querySpectrum = mhb.querySpectrum;
const publish = mhb.publish;

/// Test-only counters of points the device batch admitted and of their line
/// search backtracks; empty outside tests.
pub const gpu_test_access = if (@import("builtin").is_test) struct {
    /// Points the device batch admitted.
    pub var admitted = std.atomic.Value(usize).init(0);
    /// Line-search backtracks of the admitted points.
    pub var backtracks = std.atomic.Value(usize).init(0);
} else struct {};

/// Driven sparse HB/MHB points share sampled evaluation and resident
/// Krylov work. Every point keeps its own Newton iteration and line search.
pub fn runGpuLanes(ctxs: []const root.RunCtx, options: Options, out: []LaneOutcome) void {
    const artifacts = @import("gompute_kernels");
    std.debug.assert(ctxs.len == out.len and ctxs.len != 0);
    if (comptime artifacts.has_cuda) return dispatchGpuBatch(.cuda, ctxs, options, out);
    if (comptime artifacts.has_hip) return dispatchGpuBatch(.hip, ctxs, options, out);
    for (ctxs, out) |*ctx, *result| result.* = @import("hb.zig").run(ctx, options);
}
// ---- private ----

fn dispatchGpuBatch(comptime backend: @import("gompute").Backend, ctxs: []const root.RunCtx, options: Options, out: []LaneOutcome) void {
    runGpuBatch(backend, ctxs, options, out) catch |err| {
        for (out) |*o| o.* = err;
    };
}

/// Writes `input` as point `point`'s column of the point-minor batch `out`:
/// out[row * points + point] = input[row].
fn interleave(out: []f64, input: []const f64, point: usize, points: usize) void {
    std.debug.assert(out.len == input.len * points);
    for (input, 0..) |value, row| out[row * points + point] = value;
}

/// `interleave`'s inverse: point `point`'s column of `input` into `out`.
fn scatter(out: []f64, input: []const f64, point: usize, points: usize) void {
    std.debug.assert(input.len == out.len * points);
    for (out, 0..) |*value, row| value.* = input[row * points + point];
}

fn runGpuBatch(comptime backend: @import("gompute").Backend, ctxs: []const root.RunCtx, options: Options, out: []LaneOutcome) !void {
    const Device = @import("../gpu/periodic.zig").Engine(backend);
    const Samples = @import("../gpu/lanes.zig").Samples;
    const a = ctxs[0].scratch_allocator;
    const N = ctxs.len;
    const n: usize = ctxs[0].circuit.n;
    const nnz: usize = ctxs[0].circuit.nnz;
    var tones: [max_tones]f64 = undefined;
    const spec = try querySpectrum(a, options, &tones);
    defer spec.deinit(a);
    if (spec.freqs.len < 2) return error.InvalidQueryOptions;
    const nf = 2 * spec.freqs.len - 1;
    const nt = transformSamples(spec);
    const states = try a.alloc(?Newton, N);
    defer a.free(states);
    @memset(states, null);
    const spectra = try a.alloc([]f64, N);
    defer a.free(spectra);
    @memset(spectra, &.{});
    defer for (states, spectra, ctxs) |*state, x, ctx| {
        if (state.*) |*s| s.deinit();
        ctx.scratch_allocator.free(x);
    };
    @memset(out, error.QueryCancelled);
    var first: ?usize = null;
    for (ctxs, 0..) |ctx, p| {
        ctx.circuit.refuseDigital("hb") catch |err| {
            out[p] = err;
            continue;
        };
        if (ctx.circuit.n != n or ctx.circuit.nnz != nnz or ctx.circuit.has_charge != ctxs[0].circuit.has_charge or
            !std.mem.eql(u32, ctx.circuit.col_ptr, ctxs[0].circuit.col_ptr) or
            !std.mem.eql(u32, ctx.circuit.row_idx, ctxs[0].circuit.row_idx))
        {
            out[p] = @import("hb.zig").run(&ctx, options);
            continue;
        }
        spectra[p] = ctx.scratch_allocator.alloc(f64, n * nf) catch |err| {
            out[p] = err;
            continue;
        };
        states[p] = Newton.init(ctx.circuit, spectra[p], spec, tones[0..spec.tones], options, ctx.scratch_allocator, false) catch |err| {
            out[p] = err;
            continue;
        };
        if (first == null) first = p;
    }
    const first_point = first orelse return;
    const head = &states[first_point].?;
    const values = std.math.mul(usize, n * nf, N) catch return error.OutOfMemory;
    const sample_values = std.math.mul(usize, n * nt, N) catch return error.OutOfMemory;
    const plane_values = std.math.mul(usize, nnz * nt, N) catch return error.OutOfMemory;
    const sample_size = std.math.add(
        usize,
        std.math.mul(usize, sample_values, 3) catch return error.OutOfMemory,
        std.math.mul(usize, plane_values, 2) catch return error.OutOfMemory,
    ) catch return error.OutOfMemory;
    const sample_slab = try a.alloc(f64, sample_size);
    defer a.free(sample_slab);
    var sample_offset: usize = 0;
    const sampled = takePlanes(sample_slab, &sample_offset, sample_values, plane_values);
    const rhs = try a.alloc(f64, values);
    defer a.free(rhs);
    const dx = try a.alloc(f64, values);
    defer a.free(dx);
    const omegas = try a.alloc(f64, head.omegas.len * N);
    defer a.free(omegas);
    for (0..N) |p| interleave(omegas, head.omegas, p, N);
    const frequencies = try a.dupe(f64, head.omegas);
    defer a.free(frequencies);
    const factors = try a.alloc(*FreqSolver, N);
    defer a.free(factors);
    const results = try a.alloc(Gmres.SolveResult, N);
    defer a.free(results);
    const active = try a.alloc(u8, N);
    defer a.free(active);
    const ckts = try a.alloc(*root.Circuit, N);
    defer a.free(ckts);
    for (states, ckts) |*state, *ckt| ckt.* = if (state.*) |*s| s.ckt else head.ckt;
    var sampler: ?*Samples = Samples.init(a, ckts, head.tr.times, head.dt) catch |err| blk: {
        if (head.ckt.knobs.gpu_stats != .off) std.debug.print("gpu-hb: sample provider declined ({s})\n", .{@errorName(err)});
        break :blk null;
    };
    defer if (sampler) |s| s.deinit();
    var device: ?Device = Device.init(a, .{
        .points = N,
        .n = @intCast(n),
        .nf = @intCast(nf),
        .nt = @intCast(nt),
        .charge = head.ckt.has_charge,
        .col_ptr = head.ckt.col_ptr,
        .row_idx = head.ckt.row_idx,
        .inv = head.tr.inv,
        .fwd = head.tr.fwd,
        .omegas = omegas,
    }, head.ckt.knobs.gpu_stats != .off) catch null;
    defer if (device) |*d| d.deinit();
    var device_samples: usize = 0;
    var device_solves: usize = 0;
    while (true) {
        @memset(active, 0);
        var any = false;
        for (states, 0..) |*state, p| if (state.*) |*s| {
            if (s.phase == .sample) {
                s.checkpoint() catch |err| {
                    out[p] = err;
                    s.deinit();
                    state.* = null;
                    continue;
                };
                active[p] = 1;
                any = true;
            }
            idft(s.tr, s.nf, s.x_hat, s.planes.x);
            interleave(sampled.x, s.planes.x, p, N);
        } else {
            for (0..n * nt) |row| sampled.x[row * N + p] = 0;
        };
        if (!any) break;
        const used_device = if (sampler) |s| blk: {
            s.evaluate(sampled.x, sampled.f, sampled.q, sampled.g, sampled.c, active) catch {
                s.deinit();
                sampler = null;
                break :blk false;
            };
            device_samples += 1;
            break :blk true;
        } else false;
        for (active, states, 0..) |enabled, *state, p| {
            if (enabled == 0) continue;
            const s = &state.*.?;
            if (used_device) {
                scatter(s.planes.f, sampled.f, p, N);
                scatter(s.planes.q, sampled.q, p, N);
                scatter(s.planes.g, sampled.g, p, N);
                scatter(s.planes.c, sampled.c, p, N);
            } else {
                // The checkpoint already ran for this iteration.
                sample(s.ckt, s.tr, s.nf, s.dt, s.x_hat, s.x_sample, s.planes);
            }
            s.sampled() catch |err| {
                out[p] = err;
                s.deinit();
                state.* = null;
            };
        }
        const leader = for (states) |*state| {
            if (state.*) |*s| if (s.phase == .linear) break s;
        } else continue;
        for (states, 0..) |*state, p| {
            const s = if (state.*) |*live| if (live.phase == .linear) live else leader else leader;
            interleave(sampled.g, s.planes.g, p, N);
            interleave(sampled.c, s.planes.c, p, N);
            factors[p] = &s.fs;
            const solving = if (state.*) |*live| live.phase == .linear else false;
            active[p] = @intFromBool(solving);
            if (solving) {
                interleave(rhs, s.f_hat, p, N);
            } else {
                for (0..n * nf) |row| rhs[row * N + p] = 0;
            }
        }
        @memset(dx, 0);
        const solved_device = if (device) |*d| blk: {
            d.factor(sampled.g, sampled.c, factors, frequencies, active) catch {
                d.deinit();
                device = null;
                break :blk false;
            };
            d.solveHost(rhs, dx, @intCast(@min(n * nf, gmres_restart)), gmres_tol, gmres_max_restarts, results) catch {
                d.deinit();
                device = null;
                break :blk false;
            };
            device_solves += 1;
            break :blk true;
        } else false;
        for (states, 0..) |*state, p| if (state.*) |*s| {
            if (s.phase != .linear) continue;
            if (solved_device) {
                scatter(s.dx, dx, p, N);
                s.solved(results[p]);
            } else {
                s.factorHost() catch |err| {
                    out[p] = err;
                    s.deinit();
                    state.* = null;
                    continue;
                };
                s.solveAlone();
            }
        };
    }
    for (states, 0..) |state, p| if (state) |s| {
        out[p] = if (s.result.converged) publish(&ctxs[p], options, spec, s.x_hat) else error.HbDidNotConverge;
        if (ctxs[p].circuit.knobs.gpu_stats != .off and device_solves != 0)
            std.debug.print("gpu-hb-point: {d}, {d} iterations, {d} backtracks, converged={}\n", .{ p, s.result.iterations, s.backtracks, s.result.converged });
    };
    if (ctxs[0].circuit.knobs.gpu_stats != .off and device_solves != 0)
        std.debug.print("gpu-hb: {d} points, {d} sampled device rounds, {d} Krylov rounds\n", .{ N, device_samples, device_solves });
    if (@import("builtin").is_test and device_samples != 0 and device_solves != 0) {
        _ = gpu_test_access.admitted.fetchAdd(N, .monotonic);
        for (states) |state| if (state) |s| {
            _ = gpu_test_access.backtracks.fetchAdd(s.backtracks, .monotonic);
        };
    }
}
