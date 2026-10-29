//! One harmonic-balance point's Newton: a circuit, its spectrum and the
//! tones in, the converged (or last) spectral iterate and its status out.
//! Each step samples the circuit at the transform's instants, forms the
//! spectral residual, and solves the Jacobian by GMRES preconditioned with
//! the block diagonal G0 + jω_j·C0, one `FreqSolver` lane per line; the
//! Jacobian itself is never formed (IDFT, sampled G(t) and C(t), DFT). A
//! backtracking line search damps each step. Sampling and solves are
//! separate requests so `mhb_gpu.zig` can batch points.
//! Cites: Verilog-A LRM §4.6.1 (sources follow their waveform only in the
//! transient phase), #83 (NaN residuals).
//!
//! Contents:
//! - `pub const Operator = struct`
//! - `Operator`: `matvec`, `precond`
//! - `pub const Planes = struct { x: []f64, f: []f64, q: []f64, g: []f64, c: []f64 }`
//! - `pub const LineSearch = struct`
//! - `LineSearch`: `backtrack`
//! - `pub const gmres_tol = 1e-7`
//! - `pub const gmres_restart = 40`
//! - `pub const gmres_max_restarts = 20`
//! - `pub const gmres_recycle = 10`
//! - `pub const Newton = struct`
//! - `Newton`: `init`, `deinit`, `checkpoint`, `sampleHost`, `sampled`, `iterationComplete`, `solved`, `solveAlone`, `factorHost`
//! - `pub fn addOmega(omegas: []const f64, q_hat: []const f64, dst: []f64) void`
//! - `pub fn gvProduct(w_td: []f64, g_td: []const f64, v_td: []const f64, col_ptr: []const u32, row_idx: []const u32, nt: usize) void`
//! - `pub fn takePlanes(slab: []f64, off: *usize, node_len: usize, slot_len: usize) Planes`
//! - `pub fn sampleDt(spec: Spectrum) f64`
//! - `pub fn sample(ckt: *root.Circuit, tr: Transform, nf: usize, dt: f64, x_hat: []const f64, x_sample: []f64, p: Planes) void`
//! - `pub fn dcSeed(ckt: *root.Circuit, x_hat: []f64, x_sample: []f64, nf: usize, tol: converger.Tolerances) !void`
//! - `pub fn solve(ckt: *root.Circuit, x_hat: []f64, spec: Spectrum, tones: []const f64, options: Options, allocator: std.mem.Allocator) !SolveResult`

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
const mhb_spectrum = @import("mhb_spectrum.zig");
const mhb_transform = @import("mhb_transform.zig");
const Options = mhb.Options;
const Spectrum = mhb_spectrum.Spectrum;
const Transform = mhb_transform.Transform;
const Fold = mhb_transform.Fold;
const transform = mhb_transform.transform;
const transformSamples = mhb_transform.transformSamples;
const idft = mhb_transform.idft;
const dft = mhb_transform.dft;

/// The Newton linear system for GMRES: `matvec` is the HB Jacobian at the
/// last residual's samples, `precond` the held G0 + jω_j·C0 blocks.
pub const Operator = struct {
    ckt: *const root.Circuit,
    tr: Transform,
    omegas: []const f64,
    nf: usize,
    /// G and C at every sample, slot-major, samples contiguous.
    g_td: []const f64,
    c_td: []const f64,
    /// Matvec scratch: two n·nt time planes and one n·nf spectrum.
    v_td: []f64,
    w_td: []f64,
    u_hat: []f64,
    fs: *FreqSolver,
    gpu_held: ?*@import("../gpu/freq.zig").Held = null,
    gpu_operator: ?*@import("../gpu/periodic.zig").Operator = null,
    /// Preconditioner right-hand sides and solutions, m stacked 2n blocks.
    p_rhs: []f64,
    p_x: []f64,

    /// w = J v: IDFT, G(t)·v and C(t)·v per sample, DFT, plus Ω on the charge.
    pub fn matvec(self: *Operator, v: []const f64, w: []f64) void {
        if (self.gpu_operator) |device| if (device.matvec(v, w)) return;
        const nf = self.nf;
        const ckt = self.ckt;
        idft(self.tr, nf, v, self.v_td);
        gvProduct(self.w_td, self.g_td, self.v_td, ckt.col_ptr, ckt.row_idx, self.tr.nt);
        dft(self.tr, nf, self.w_td, w);
        if (!ckt.has_charge) return;
        gvProduct(self.w_td, self.c_td, self.v_td, ckt.col_ptr, ckt.row_idx, self.tr.nt);
        dft(self.tr, nf, self.w_td, self.u_hat);
        addOmega(self.omegas, self.u_hat, w);
    }

    /// r := M^-1 r. Line j's (c, s) rows are the phasor X = c - js of
    /// (G0 + jω_j C0)·X = r_c - j·r_s.
    pub fn precond(self: *Operator, r: []f64) void {
        if (self.gpu_operator) |device| if (device.precond(r, r)) return;
        const nf = self.nf;
        const n = r.len / nf;
        const nn = 2 * n;
        for (0..self.omegas.len) |j| {
            const b = self.p_rhs[j * nn ..][0..nn];
            for (0..n) |node| {
                b[node] = r[node * nf + ((2 * j) -| 1)];
                b[n + node] = if (j == 0) 0 else -r[node * nf + 2 * j];
            }
        }
        const device_solved = if (self.gpu_held) |held| held.solve(self.p_rhs, self.p_x, false) else false;
        if (!device_solved) self.fs.solveEach(self.p_rhs, self.p_x, false);
        for (0..self.omegas.len) |j| {
            const x = self.p_x[j * nn ..][0..nn];
            for (0..n) |node| {
                if (j == 0) {
                    r[node * nf] = x[node];
                } else {
                    r[node * nf + 2 * j - 1] = x[node];
                    r[node * nf + 2 * j] = -x[n + node];
                }
            }
        }
    }
};

/// The sampled planes: x, residual f and charge q node-major, G and C
/// slot-major, `nt` samples contiguous in each.
pub const Planes = struct { x: []f64, f: []f64, q: []f64, g: []f64, c: []f64 };

/// The backtracking line search both HB Newtons take on their own residual
/// evaluation: a step that made the residual worse (or non-finite, hence
/// the negated comparison) is undone and retaken at half length, and a step
/// that helped earns the length back. A full Newton step can hand a diode
/// the whole source swing at once.
pub const LineSearch = struct {
    step: f64 = 1,
    prev_residual: f64 = std.math.inf(f64),
    retrying: bool = false,

    /// The shortest step before the search stops shortening.
    const min_step: f64 = 1.0 / 1024.0;

    /// Judges the step that produced `residual`. True: rewind it and retake
    /// it at the halved `step`. False: the step stands and `step` is the
    /// next one's length. `first` (the starting point, or a judgement that
    /// must not rewind) never backtracks.
    pub fn backtrack(ls: *LineSearch, first: bool, residual: f64) bool {
        if (!first and !(residual <= ls.prev_residual) and ls.step > min_step) {
            ls.step *= 0.5;
            ls.retrying = true;
            return true;
        }
        ls.prev_residual = residual;
        if (!ls.retrying) ls.step = @min(ls.step * 2, 1);
        ls.retrying = false;
        return false;
    }
};

/// GMRES stopping tolerance relative to the Newton residual.
pub const gmres_tol: f64 = 1e-7;

/// GMRES restart depth and restarts per Newton step.
pub const gmres_restart: u32 = 40;

/// GMRES restarts per Newton step.
pub const gmres_max_restarts: u32 = 20;

/// Harmonic Ritz vectors the host solve recycles across restarts and
/// Newton steps (`Gcrodr`), out of `gmres_restart`.
pub const gmres_recycle: u32 = 10;

/// One point's Newton and backtracking continuation. Sampling and linear
/// solves are separate requests so independent points can share device
/// work without sharing a convergence or damping decision.
pub const Newton = struct {
    const Phase = enum { sample, linear, done };
    allocator: std.mem.Allocator,
    ckt: *root.Circuit,
    options: Options,
    phase: Phase = .sample,
    result: SolveResult = undefined,
    slab: []f64,
    x_hat: []f64,
    tr: Transform,
    nf: usize,
    dt: f64,
    f_hat: []f64,
    q_hat: []f64,
    dx: []f64,
    x_prev: []f64,
    planes: Planes,
    g0: []f64,
    c0: []f64,
    p_rhs: []f64,
    p_x: []f64,
    omegas: []f64,
    x_sample: []f64,
    fs: FreqSolver,
    factor_on_host: bool,
    factors_ready: bool = false,
    gmres: Gcrodr,
    gpu_held: ?@import("../gpu/freq.zig").Held = null,
    gpu_held_disabled: bool = false,
    gpu_operator: ?@import("../gpu/periodic.zig").Operator = null,
    ls: LineSearch = .{},
    iter: u16 = 0,
    backtracks: usize = 0,

    /// Returns one point's Newton state from `x_hat`'s start. Caller owns it;
    /// free with `deinit`.
    pub fn init(ckt: *root.Circuit, x_hat: []f64, spec: Spectrum, tones: []const f64, options: Options, allocator: std.mem.Allocator, single_gpu: bool) !Newton {
        const n: usize = ckt.n;
        const nnz: usize = ckt.nnz;
        const m = spec.freqs.len;
        const nf = 2 * m - 1;
        const nt = transformSamples(spec);
        const total = n * nf;
        std.debug.assert(x_hat.len == total and m >= 2);
        const len = nt + 2 * nt * nf + 4 * total + 3 * n * nt + 2 * nnz * nt + 2 * nnz + 4 * m * n + m + n;
        const slab = try allocator.alloc(f64, len);
        errdefer allocator.free(slab);
        var off: usize = 0;
        var tr = try transform(spec, tones, take(slab, &off, nt + 2 * nt * nf), allocator);
        if (spec.tones == 1) tr.fold = try Fold.init(allocator, spec, tr);
        errdefer if (tr.fold) |f| f.deinit(allocator);
        const f_hat = take(slab, &off, total);
        const q_hat = take(slab, &off, total);
        const dx = take(slab, &off, total);
        const x_prev = take(slab, &off, total);
        const planes = takePlanes(slab, &off, n * nt, nnz * nt);
        const g0 = take(slab, &off, nnz);
        const c0 = take(slab, &off, nnz);
        const p_rhs = take(slab, &off, 2 * m * n);
        const p_x = take(slab, &off, 2 * m * n);
        const omegas = take(slab, &off, m);
        const x_sample = take(slab, &off, n);
        std.debug.assert(off == len);
        for (omegas, spec.freqs) |*w, f| w.* = 2 * std.math.pi * f;
        if (!ckt.has_charge) @memset(planes.c, 0);
        try dcSeed(ckt, x_hat, x_sample, nf, options.tol);
        var fs = try FreqSolver.fromCircuit(allocator, ckt, x_sample);
        errdefer fs.deinit();
        // Complex held factors on the host solve; the grouped and device
        // drivers read stacked tapes.
        fs.complex_held = fs.complex_lanes and single_gpu and ckt.gpu == null;
        const restart: u32 = @intCast(@min(total, gmres_restart));
        var gmres = try Gcrodr.init(allocator, @intCast(total), restart, @min(gmres_recycle, restart / 4));
        errdefer gmres.deinit(allocator);
        const device: ?@import("../gpu/periodic.zig").Operator = if (single_gpu and ckt.gpu != null)
            @import("../gpu/periodic.zig").Operator.init(allocator, .{
                .points = 1,
                .n = @intCast(n),
                .nf = @intCast(nf),
                .nt = @intCast(nt),
                .charge = ckt.has_charge,
                .col_ptr = ckt.col_ptr,
                .row_idx = ckt.row_idx,
                .inv = tr.inv,
                .fwd = tr.fwd,
                .omegas = omegas,
            }, ckt.knobs.gpu_stats != .off) catch null
        else
            null;
        return .{
            .allocator = allocator,
            .ckt = ckt,
            .options = options,
            .slab = slab,
            .x_hat = x_hat,
            .tr = tr,
            .nf = nf,
            .dt = sampleDt(spec),
            .f_hat = f_hat,
            .q_hat = q_hat,
            .dx = dx,
            .x_prev = x_prev,
            .planes = planes,
            .g0 = g0,
            .c0 = c0,
            .p_rhs = p_rhs,
            .p_x = p_x,
            .omegas = omegas,
            .x_sample = x_sample,
            .fs = fs,
            .gmres = gmres,
            .gpu_operator = device,
            .factor_on_host = single_gpu,
        };
    }

    /// Frees what `init` allocated.
    pub fn deinit(self: *Newton) void {
        if (self.tr.fold) |f| f.deinit(self.allocator);
        if (self.gpu_operator) |*device| device.deinit();
        if (self.gpu_held) |*held| held.deinit();
        self.gmres.deinit(self.allocator);
        self.fs.deinit();
        self.allocator.free(self.slab);
        self.* = undefined;
    }

    /// Reports progress; fails when the query is cancelled.
    pub fn checkpoint(self: *Newton) !void {
        if (self.iter != 0) try self.ckt.checkpoint(.{ .phase = .harmonic, .completed = self.iter });
    }

    /// Samples the circuit at every instant on the host.
    pub fn sampleHost(self: *Newton) !void {
        try self.checkpoint();
        sample(self.ckt, self.tr, self.nf, self.dt, self.x_hat, self.x_sample, self.planes);
    }

    /// Consumes the samples: the residual, the line search, and the next
    /// phase.
    pub fn sampled(self: *Newton) !void {
        std.debug.assert(self.phase == .sample);
        dft(self.tr, self.nf, self.planes.f, self.f_hat);
        if (self.ckt.has_charge) {
            dft(self.tr, self.nf, self.planes.q, self.q_hat);
            addOmega(self.omegas, self.q_hat, self.f_hat);
        }
        const residual = num.normInf(self.f_hat);
        if (self.ckt.knobs.hb_trace) std.debug.print("MHB iter={d} res={e} step={e}\n", .{ self.iter, residual, self.ls.step });
        if (residual < self.options.hb_tol) {
            self.result = .{ .converged = true, .iterations = self.iter + 1, .residual_norm = residual };
            self.phase = .done;
            return;
        }
        if (self.ls.backtrack(self.iter == 0, residual)) {
            self.backtracks += 1;
            num.copySimd(self.x_hat, self.x_prev);
            num.axpy(self.x_hat, self.ls.step, self.dx);
            self.iterationComplete();
            return;
        }
        const nt = self.tr.nt;
        const inv_nt = 1 / @as(f64, @floatFromInt(nt));
        for (0..self.ckt.nnz) |slot| {
            self.g0[slot] = num.sum(self.planes.g[slot * nt ..][0..nt]) * inv_nt;
            self.c0[slot] = num.sum(self.planes.c[slot * nt ..][0..nt]) * inv_nt;
        }
        self.fs.setPlanes(self.g0, self.c0);
        self.factors_ready = false;
        if (self.factor_on_host) try self.factorHost();
        num.scale(self.f_hat, -1, self.f_hat);
        @memset(self.dx, 0);
        self.phase = .linear;
    }

    /// Ends one Newton iteration.
    pub fn iterationComplete(self: *Newton) void {
        self.iter += 1;
        if (self.iter == self.options.max_iter) {
            self.result = .{ .converged = false, .iterations = self.options.max_iter, .residual_norm = num.normInf(self.f_hat) };
            self.phase = .done;
        } else self.phase = .sample;
    }

    /// Consumes the linear solve's outcome.
    pub fn solved(self: *Newton, result: Gmres.SolveResult) void {
        std.debug.assert(self.phase == .linear);
        if (self.ckt.knobs.hb_trace) std.debug.print("  gmres it={d} res={e} conv={}\n", .{ result.iterations, result.residual, result.converged });
        num.copySimd(self.x_prev, self.x_hat);
        num.axpy(self.x_hat, self.ls.step, self.dx);
        self.iterationComplete();
    }

    /// Runs the whole Newton on the host.
    pub fn solveAlone(self: *Newton) void {
        std.debug.assert(self.factors_ready);
        if (self.gpu_operator) |*device| {
            device.refresh(self.planes.g, self.planes.c, &self.fs) catch {
                device.deinit();
                self.gpu_operator = null;
            };
        }
        if (self.gpu_operator == null)
            @import("../gpu/freq.zig").prepareHeld(&self.gpu_held, &self.gpu_held_disabled, self.ckt, &self.fs, self.omegas.len, self.allocator);
        var op: Operator = .{
            .ckt = self.ckt,
            .tr = self.tr,
            .omegas = self.omegas,
            .nf = self.nf,
            .g_td = self.planes.g,
            .c_td = self.planes.c,
            .v_td = self.planes.x,
            .w_td = self.planes.f,
            .u_hat = self.q_hat,
            .fs = &self.fs,
            .p_rhs = self.p_rhs,
            .p_x = self.p_x,
            .gpu_operator = if (self.gpu_operator) |*device| device else null,
            .gpu_held = if (self.gpu_held) |*held| held else null,
        };
        const total = self.f_hat.len;
        const result = (if (self.gpu_operator) |*device|
            device.solve(self.f_hat, self.dx, @intCast(@min(total, gmres_restart)), gmres_tol, gmres_max_restarts)
        else
            null) orelse self.gmres.solve(&op, self.f_hat, self.dx, gmres_tol, gmres_max_restarts);
        self.solved(result);
    }

    /// Factors the preconditioner blocks on the host.
    pub fn factorHost(self: *Newton) !void {
        if (self.factors_ready) return;
        try self.fs.ensureHeldOnHost(self.omegas);
        self.factors_ready = true;
    }
};

/// dst += Ω·q_hat: dq/dt of q(t) = a cos(ω t) + b sin(ω t) is
/// ω b cos - ω a sin, so line j's cos row takes +ω_j·b and its sin row
/// -ω_j·a.
pub fn addOmega(omegas: []const f64, q_hat: []const f64, dst: []f64) void {
    const nf = 2 * omegas.len - 1;
    const n = q_hat.len / nf;
    for (0..n) |node| {
        const q = q_hat[node * nf ..][0..nf];
        const d = dst[node * nf ..][0..nf];
        for (omegas[1..], 1..) |w, j| {
            d[2 * j - 1] += w * q[2 * j];
            d[2 * j] -= w * q[2 * j - 1];
        }
    }
}

/// Time-domain Jacobian-vector product w_td[., s] = G(t_s) * v_td[., s] over
/// the CSC pattern; g_td is slot-major, samples contiguous. Each structural
/// entry is one axpy over its nt samples, W at a time, with the scalar tail
/// as the width-one oracle. Each row still sums its columns in ascending
/// order from +0, and an entry outside the pattern could only add ±0, so for
/// finite v this is bitwise the dense product. Overwrites every w_td slot.
pub fn gvProduct(w_td: []f64, g_td: []const f64, v_td: []const f64, col_ptr: []const u32, row_idx: []const u32, nt: usize) void {
    const W = std.simd.suggestVectorLength(f64) orelse 8;
    const V = @Vector(W, f64);
    @memset(w_td, 0);
    for (0..col_ptr.len - 1) |col| {
        const vc = v_td[col * nt ..][0..nt];
        for (col_ptr[col]..col_ptr[col + 1]) |slot| {
            const g = g_td[slot * nt ..][0..nt];
            const w = w_td[@as(usize, row_idx[slot]) * nt ..][0..nt];
            var s: usize = 0;
            while (s + W <= nt) : (s += W) {
                const gv: V = g[s..][0..W].*;
                const vv: V = vc[s..][0..W].*;
                const wv: V = w[s..][0..W].*;
                w[s..][0..W].* = wv + gv * vv;
            }
            while (s < nt) : (s += 1) w[s] += g[s] * vc[s];
        }
    }
}

/// Carves `Planes` out of `slab` at `off`: x, f and q `node_len` each, then
/// g and c `slot_len` each.
pub fn takePlanes(slab: []f64, off: *usize, node_len: usize, slot_len: usize) Planes {
    return .{
        .x = take(slab, off, node_len),
        .f = take(slab, off, node_len),
        .q = take(slab, off, node_len),
        .g = take(slab, off, slot_len),
        .c = take(slab, off, slot_len),
    };
}

/// The step `sample` declares to the devices, 1/(nt·f_1) with f_1 the
/// lowest line.
pub fn sampleDt(spec: Spectrum) f64 {
    return 1 / (spec.freqs[1] * @as(f64, @floatFromInt(transformSamples(spec))));
}

/// Evaluates the circuit at every instant of `tr` on `x_hat`'s waveform
/// into `p`; `p.c` is left alone without charge. `x_sample` (n) is scratch.
pub fn sample(ckt: *root.Circuit, tr: Transform, nf: usize, dt: f64, x_hat: []const f64, x_sample: []f64, p: Planes) void {
    const n: usize = ckt.n;
    const nnz: usize = ckt.nnz;
    const nt = tr.nt;
    idft(tr, nf, x_hat, p.x);
    // Sample s is the circuit at t_s in the transient phase, the only
    // phase in which a source follows its waveform (§4.6.1).
    // The grouped driver evaluates these instants through gpu_lanes.Samples
    // when every model supports independent sampled evaluation.
    for (0..nt) |s| {
        const t = tr.times[s];
        for (0..n) |node| x_sample[node] = p.x[node * nt + s];
        ckt.setSimState(.{ .t = t, .dt = dt, .kind = .tran });
        ckt.eval(x_sample, t);
        for (0..n) |node| p.f[node * nt + s] = ckt.rhs[node];
        for (0..n) |node| p.q[node * nt + s] = ckt.q_vec[node];
        for (ckt.g_vals[0..nnz], 0..) |g, slot| p.g[slot * nt + s] = g;
        if (ckt.has_charge) for (ckt.c_vals[0..nnz], 0..) |c, slot| {
            p.c[slot * nt + s] = c;
        };
    }
}

/// Zeroes `x_hat` (node-major, `nf` coefficients per node) and seeds each
/// node's DC coefficient with the DC operating point, solved into
/// `x_sample`, which cuts the iteration count on biased circuits. Best
/// effort: a failed solve seeds its last iterate; only cancellation and out
/// of memory propagate. Asserts that `x_hat` holds `nf` per unknown.
pub fn dcSeed(ckt: *root.Circuit, x_hat: []f64, x_sample: []f64, nf: usize, tol: converger.Tolerances) !void {
    const n: usize = ckt.n;
    std.debug.assert(x_hat.len == n * nf);
    std.debug.assert(x_sample.len >= n);
    @memset(x_hat, 0);
    const ws = try ckt.workspace();
    ckt.setSimState(.{ .kind = .dc });
    @memset(x_sample, 0);
    _ = converger.run(ckt, ws, x_sample, 0, converger.optionsFromTolerances(tol, null), root.EvalHook{}) catch |err| switch (err) {
        error.QueryCancelled, error.OutOfMemory => return err,
        else => {}, // else: the DC seed is best effort
    };
    for (0..n) |node| x_hat[node * nf] = x_sample[node];
}

/// Driven harmonic balance on `spec` (tones `tones` Hz) from the DC
/// operating point. `x_hat` (n·(2m-1), caller-owned) receives node-major
/// blocks [dc, c_1, s_1, ...], the last iterate when Newton does not
/// converge. Newton takes the `LineSearch`; each step
/// is GMRES on the matrix-free Jacobian with the block-diagonal
/// preconditioner. Memory is O(nnz·nt + m·LU) for the sampled planes and
/// the held factors. Asserts `spec` has two or more lines and x_hat.len is
/// n·(2m-1). Non-convergence is `converged = false`, not an error; a GMRES
/// step that stops short is taken as it stands. Errors: OutOfMemory,
/// QueryCancelled, error.HbDidNotConverge when the collocation transform
/// of several tones is singular, and `FreqSolver.factorEach`'s.
pub fn solve(ckt: *root.Circuit, x_hat: []f64, spec: Spectrum, tones: []const f64, options: Options, allocator: std.mem.Allocator) !SolveResult {
    var state = try Newton.init(ckt, x_hat, spec, tones, options, allocator, true);
    defer state.deinit();
    while (state.phase != .done) switch (state.phase) {
        .sample => {
            try state.sampleHost();
            try state.sampled();
        },
        .linear => state.solveAlone(),
        .done => unreachable,
    };
    return state.result;
}
