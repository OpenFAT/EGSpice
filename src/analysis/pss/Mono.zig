//! The shooting Jacobian by the trapezoid's own sensitivity recurrence: one
//! recorded base period (its x_k) in, G_k, C_k and A_k = G_k + αC_k
//! evaluated once per shooting iteration, then Φ·v (and the autonomous
//! period column) as one linear solve per step out, no device evaluation.
//! Cites: none (the trapezoid recurrence below is derived from period.zig's
//! own discrete map).
//!
//! Contents:
//! - `pub const MonoKrylovCtx = struct`, `MonoKrylovCtx.matvec(ctx: *MonoKrylovCtx, v: []const f64, w: []f64) void`
//! - `pub fn init(gpa: std.mem.Allocator, ckt: *const root.Circuit, steps: usize, autonomous: bool) !?Mono`
//! - `pub fn deinit(m: *Mono, gpa: std.mem.Allocator) void`
//! - `pub fn build(m: *Mono, ckt: *root.Circuit, opts: Options) void`
//! - `pub fn apply(m: *Mono, gpa: std.mem.Allocator, ckt: *const root.Circuit, opts: Options, blk: []f64, r: usize, period_col: ?usize) error{OutOfMemory}!bool`

const std = @import("std");
const num = @import("core").numerics;
const direct = @import("solver").direct;
const root = @import("../types.zig");
const period = @import("period.zig");

const Options = @import("core").query.Pss;
const stepTime = period.stepTime;

/// The shooting Jacobian Φ = dx(T)/dx0 by the trapezoid's own sensitivity
/// recurrence over one recorded period,
/// instead of one period integration per column or matvec. The base
/// integration records every step's x_k; `build` evaluates G_k and C_k
/// there once per shooting iteration, and each product Φ·v is then one
/// linear solve per step on A_k = G_k + αC_k, no device evaluation:
///
///   s_0 = v, dq_0 = C_0 v, σ_0 = -mask·G_0 v (the i_prev seed's derivative)
///   A_k s_k = α dq_{k-1} + σ_{k-1}
///   dq_k = C_k s_k,  σ_k = α (dq_k - dq_{k-1}) - σ_{k-1}
///
/// The Jacobians are taken at the converged x_k, not at the step's last
/// Newton iterate, so Φ is the exact derivative of the discrete map to
/// within the step tolerance. An autonomous circuit's period column
/// d x_N / d(ln T) runs the same recurrence from s_0 = 0, driven by α's
/// dependence on T (dα/d(ln T) = -α, and α(q_k - q_{k-1}) = i_k + i_{k-1}):
///
///   A_k s_k = α dq_{k-1} + σ_{k-1} + α(q_k - q_{k-1}) - t_k ∂f/∂t
///   σ_k = α (dq_k - dq_{k-1}) - α(q_k - q_{k-1}) - σ_{k-1}
///
/// with t_k ∂f/∂t, the sources' share, a finite difference in time at the
/// recorded x_k (zero when every source is constant).
/// 0..n-1, the base request's probe list: its wave is the trajectory.
rows: []u32,
/// (n_samples + 1) rows [t, x...] of the base period.
traj: []f64,
g0: []f64,
/// C_k for k = 0..n_samples and A_k for k = 1..n_samples, nnz each.
c: []f64,
a: []f64,
/// Autonomous only, else empty: q_k for k = 0..n_samples and
/// t_k ∂f/∂t for k = 1..n_samples, n each.
q: []f64 = &.{},
ft: []f64 = &.{},
/// dq, σ, the right-hand side and C s: n each per column; grown.
work: []f64 = &.{},
slv: direct.Solver,
/// The trajectory of the last base period was recorded and `build` ran.
ready: bool = false,

/// GMRES operator on the recorded monodromy: w = (Φ - I) v, the
/// autonomous period column from `dphi_dlnt`. A singular step Jacobian
/// sets `singular` and returns w = 0; the caller then falls back.
pub const MonoKrylovCtx = struct {
    mono: *Mono,
    gpa: std.mem.Allocator,
    ckt: *const root.Circuit,
    options: Options,
    n: usize,
    dphi_dlnt: []const f64,
    singular: bool = false,
    failure: ?error{OutOfMemory} = null,

    /// The GMRES operator. Reports allocation failure in `failure` and a
    /// singular step Jacobian in `singular`, since GMRES takes a void matvec.
    pub fn matvec(ctx: *MonoKrylovCtx, v: []const f64, w: []f64) void {
        const n = ctx.n;
        const osc = ctx.options.osc_node;
        @memcpy(w[0..n], v[0..n]);
        if (osc != root.GROUND) w[osc] = 0;
        const ok = ctx.mono.apply(ctx.gpa, ctx.ckt, ctx.options, w[0..n], 1, null) catch |err| blk: {
            ctx.failure = err;
            break :blk false;
        };
        if (!ok) {
            ctx.singular = true;
            @memset(w[0..n], 0);
            return;
        }
        for (w[0..n], v[0..n], 0..) |*wi, vi, i| wi.* -= if (i == osc and osc != root.GROUND) 0 else vi;
        if (osc != root.GROUND) num.axpy(w[0..n], v[osc], ctx.dphi_dlnt);
    }
};

const Mono = @This();

/// f64s the monodromy recurrence may hold (planes and trajectory) before the
/// solve keeps the finite-difference columns instead: 512 MB.
/// ponytail: one fixed budget; store factors instead of planes, or
/// checkpoint the trajectory, if long periods on large circuits need it.
const mono_budget: usize = 1 << 26;

/// Returns the recurrence storage for `steps` steps, or null when it
/// would pass `mono_budget`. Caller owns it; free with `deinit` and `gpa`.
pub fn init(gpa: std.mem.Allocator, ckt: *const root.Circuit, steps: usize, autonomous: bool) !?Mono {
    const n: usize = ckt.n;
    const nnz: usize = ckt.nnz;
    const period_planes = if (autonomous) (2 * steps + 1) * n else 0;
    if ((2 * steps + 2) * nnz + (steps + 1) * (n + 1) + period_planes > mono_budget) return null;
    const rows = try gpa.alloc(u32, n);
    errdefer gpa.free(rows);
    for (rows, 0..) |*r, i| r.* = @intCast(i);
    const traj = try gpa.alloc(f64, (steps + 1) * (n + 1));
    errdefer gpa.free(traj);
    const g0 = try gpa.alloc(f64, nnz);
    errdefer gpa.free(g0);
    const c = try gpa.alloc(f64, (steps + 1) * nnz);
    errdefer gpa.free(c);
    const a = try gpa.alloc(f64, steps * nnz);
    errdefer gpa.free(a);
    const q = try gpa.alloc(f64, if (autonomous) (steps + 1) * n else 0);
    errdefer gpa.free(q);
    const ft = try gpa.alloc(f64, if (autonomous) steps * n else 0);
    errdefer gpa.free(ft);
    var slv = try direct.Solver.init(gpa, ckt.n, ckt.col_ptr, ckt.row_idx, null, ckt.knobs.lu);
    // Each step's values differ; the bypass copy never matches.
    slv.dropBypass();
    return .{ .rows = rows, .traj = traj, .g0 = g0, .c = c, .a = a, .q = q, .ft = ft, .slv = slv };
}

/// Frees what `init` and `apply` allocated on `gpa`.
pub fn deinit(m: *Mono, gpa: std.mem.Allocator) void {
    m.slv.deinit();
    inline for (.{ m.traj, m.g0, m.c, m.a, m.q, m.ft, m.work }) |b| gpa.free(b);
    gpa.free(m.rows);
}

/// Evaluates G and C at every recorded x_k of the base period, and for
/// an autonomous circuit q_k and t_k ∂f/∂t (one more eval per step).
pub fn build(m: *Mono, ckt: *root.Circuit, opts: Options) void {
    const n: usize = ckt.n;
    const nnz: usize = ckt.nnz;
    const steps: usize = opts.n_samples;
    const dt = opts.period / @as(f64, @floatFromInt(steps));
    for (0..steps + 1) |k| {
        const x = m.traj[k * (n + 1) + 1 ..][0..n];
        const t = if (k == 0) 0 else stepTime(opts, k - 1);
        const ft: []f64 = if (m.ft.len != 0 and k != 0) m.ft[(k - 1) * n ..][0..n] else &.{};
        if (ft.len != 0) {
            // f at t(1 + ε): the stretched period's step time.
            const t_eps = t * (1 + opts.fd_epsilon);
            ckt.setSimState(.{ .t = t_eps, .dt = dt, .kind = .tran });
            ckt.eval(x, t_eps);
            @memcpy(ft, ckt.rhs[0..n]);
        }
        ckt.setSimState(.{ .t = t, .dt = dt, .kind = .tran });
        ckt.eval(x, t);
        if (ft.len != 0) for (ft, ckt.rhs[0..n]) |*d, f| {
            d.* = (d.* - f) / opts.fd_epsilon;
        };
        if (m.q.len != 0) {
            const qk = m.q[k * n ..][0..n];
            if (ckt.has_charge) @memcpy(qk, ckt.q_vec[0..n]) else @memset(qk, 0);
        }
        @memcpy(m.c[k * nnz ..][0..nnz], ckt.c_vals[0..nnz]);
        if (k == 0) {
            @memcpy(m.g0, ckt.g_vals[0..nnz]);
        } else {
            ckt.combineGC(2.0 / dt, m.a[(k - 1) * nnz ..][0..nnz]);
        }
    }
    m.ready = true;
}

/// Replaces each of the r columns of `blk` (column-major, n each) by
/// Φ times it; column `period_col`, which must enter zero, receives
/// d x_N / d(ln T) instead (autonomous `Mono` only). False when some
/// A_k is singular (`blk` is garbage then).
pub fn apply(m: *Mono, gpa: std.mem.Allocator, ckt: *const root.Circuit, opts: Options, blk: []f64, r: usize, period_col: ?usize) error{OutOfMemory}!bool {
    std.debug.assert(m.ready);
    const n: usize = ckt.n;
    const nnz: usize = ckt.nnz;
    const steps: usize = opts.n_samples;
    const alpha = 2.0 * @as(f64, @floatFromInt(steps)) / opts.period;
    std.debug.assert(blk.len == n * r);
    if (period_col) |pc| std.debug.assert(pc < r and m.q.len != 0);
    if (m.work.len < 2 * n * r + 2 * n) {
        gpa.free(m.work);
        m.work = &.{};
        m.work = try gpa.alloc(f64, 2 * n * r + 2 * n);
    }
    const dq = m.work[0 .. n * r];
    const sig = m.work[n * r ..][0 .. n * r];
    const rhs = m.work[2 * n * r ..][0..n];
    const cs = m.work[2 * n * r + n ..][0..n];
    const c0 = m.c[0..nnz];
    for (0..r) |j| {
        const s = blk[j * n ..][0..n];
        spmv(ckt, c0, s, dq[j * n ..][0..n]);
        const sg = sig[j * n ..][0..n];
        spmv(ckt, m.g0, s, sg);
        // As `periodStart` seeds i_prev = -rhs on rows with a diagonal C.
        sg[0] = 0;
        for (1..n) |i| sg[i] = if (c0[ckt.diag_slots[i]] != 0) -sg[i] else 0;
        if (!ckt.has_charge) @memset(sg, 0);
    }
    for (1..steps + 1) |k| {
        m.slv.factor(m.a[(k - 1) * nnz ..][0..nnz], .{}) catch |err| switch (err) {
            error.OutOfMemory => return error.OutOfMemory,
            else => return false, // else: a singular step Jacobian
        };
        const ck = m.c[k * nnz ..][0..nnz];
        for (0..r) |j| {
            const s = blk[j * n ..][0..n];
            const dqj = dq[j * n ..][0..n];
            const sg = sig[j * n ..][0..n];
            const drive = period_col != null and period_col.? == j;
            for (rhs, dqj, sg) |*b, d, g| b.* = alpha * d + g;
            if (drive) for (rhs, m.q[k * n ..][0..n], m.q[(k - 1) * n ..][0..n], m.ft[(k - 1) * n ..][0..n]) |*b, qk, qp, f| {
                b.* += alpha * (qk - qp) - f;
            };
            m.slv.solve(rhs, s);
            spmv(ckt, ck, s, cs);
            for (sg, cs, dqj) |*g, c, d| g.* = alpha * (c - d) - g.*;
            if (drive) for (sg, m.q[k * n ..][0..n], m.q[(k - 1) * n ..][0..n]) |*g, qk, qp| {
                g.* -= alpha * (qk - qp);
            };
            @memcpy(dqj, cs);
        }
    }
    return true;
}

// ---- private ----
/// y = A x over the circuit pattern.
fn spmv(ckt: *const root.Circuit, vals: []const f64, x: []const f64, y: []f64) void {
    @memset(y, 0);
    for (0..ckt.n) |j| for (ckt.col_ptr[j]..ckt.col_ptr[j + 1]) |p| {
        y[ckt.row_idx[p]] += vals[p] * x[j];
    };
}
