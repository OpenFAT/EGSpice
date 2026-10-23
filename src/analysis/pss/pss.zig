//! Periodic steady state by shooting Newton: find x0 with
//! phi(x0) = x(T; x0) - x0 = 0, integrating one period with fixed-step
//! trapezoid. The Newton system (Phi - I) dx0 = -phi is a dense
//! finite-difference Jacobian below krylov_threshold unknowns, else JFNK GMRES.
//!
//! An autonomous circuit (an oscillator, `Options.osc_node` set) has no
//! period to impose. There T joins the unknowns and the oscillator node's
//! x0 leaves them: its value stays pinned at t = 0, which fixes the phase
//! the time shift would otherwise leave free. The Jacobian keeps its n x n
//! shape, the osc column holding d(phi)/d(ln T) instead of d(phi)/d(x0[osc]).
//!
//! Cites: dev/analysis/pss-shooting-harmonic-balance.md, Verilog-A LRM
//! §4.6.1 (sources follow their waveform only in the transient phase).
const std = @import("std");
const root = @import("../types.zig");
const take = root.take;
const num = @import("core").numerics;
const converger = @import("solver").converger;
const integrator = @import("../tran/integrator.zig");
const dense_lu = @import("solver").dense_lu;
const Gmres = @import("solver").gmres.Gmres;
const solver = @import("solver");
const Lanes = solver.lanes.Lanes;
const gpu_lanes = @import("../gpu/lanes.zig");
const direct = solver.direct;

/// The lane system one period's step Newtons run on: each lane's own
/// circuit and its own `PeriodHook` (phase 3's per-lane hooks).
fn LaneSys(comptime W: usize) type {
    return solver.lane_system.LaneSystem(W, root.Circuit, PeriodHook);
}

/// ponytail: node count at which the Newton solve switches from the dense FD
/// Jacobian (n+1 period integrations, O(n^2) storage) to matrix-free GMRES
/// (one integration per matvec).
const krylov_threshold: usize = 50;

/// Guess periods an autonomous solve records to measure the period from.
const measure_periods: usize = 8;

/// The largest relative period change one autonomous Newton step takes;
/// a longer step is shortened, all unknowns together.
const max_period_step: f64 = 0.2;

/// Volts the oscillator node is pushed off its DC equilibrium to start it.
const osc_kick: f64 = 1e-3;

/// Cap on the periods an autonomous start settles past
/// `osc_settle_periods` while the swing still grows by over 1 % a period.
const max_settle_periods: usize = 2000;

/// The `.pss` query. With `osc_node` set, `period` is only the first guess.
pub const Options = @import("core").query.Pss;

/// Outcome of a periodic solve (shooting, HB, multi-tone HB and qpss).
pub const SolveResult = struct {
    converged: bool,
    /// Newton (shooting or spectral) iterations taken.
    iterations: u16,
    /// Infinity norm of the final residual; inf when an integration failed.
    residual_norm: f64,
};

/// Trapezoidal companion state, reused across every period integration.
const Companion = struct {
    /// q at the previous accepted step.
    q_prev: []f64,
    /// q(x) from the last Newton assemble.
    q_cur: []f64,
    /// Dynamic current of the previous accepted step.
    i_prev: []f64,
    /// G + alpha*C, nnz long.
    a_vals: []f64,
};

/// Newton hook for one fixed step: residual adds
/// alpha*(q(x) - q_prev) - i_prev with alpha = 2/dt, matrix G + alpha*C.
const PeriodHook = integrator.Hook(.trapezoidal);

/// Integrates x in place over [0, T] in n_samples trapezoid steps. A
/// non-empty `wave` receives n_samples+1 point-major rows
/// [t, v(probes)...] of stride 1 + probes.len. Returns false on the first
/// step whose Newton fails.
fn integrateOnePeriod(
    ckt: *root.Circuit,
    ws: *converger.Workspace,
    sc: *Companion,
    x: []f64,
    probes: []const u32,
    wave: []f64,
    options: Options,
) error{ QueryCancelled, OutOfMemory }!bool {
    const steps: usize = options.n_samples;
    const dt = options.period / @as(f64, @floatFromInt(steps));
    periodStart(ckt, sc, x, probes, wave, options);
    for (0..steps) |k| {
        const t = stepTime(options, k);
        const hook = periodHook(sc, 2.0 / dt, ckt.has_charge);
        ckt.setSimState(.{ .t = t, .dt = dt, .kind = .tran });
        const nr = converger.run(ckt, ws, x, t, .{
            .max_iter = options.max_newton_iter,
            .abstol = options.newton_tol,
        }, hook) catch |err| switch (err) {
            error.QueryCancelled, error.OutOfMemory => |e| return e,
            else => null, // else: a numerical failure fails the period
        };
        if (!periodAccept(ckt, &hook, x, probes, wave, k, t, nr != null and nr.?.converged)) return false;
    }
    return true;
}

/// `integrateOnePeriod`'s start: the transient-phase sim state, the
/// trapezoid seam state at x, and the t = 0 wave row.
fn periodStart(ckt: *root.Circuit, sc: *Companion, x: []const f64, probes: []const u32, wave: []f64, options: Options) void {
    const n: usize = ckt.n;
    const dt = options.period / @as(f64, @floatFromInt(options.n_samples));
    // Sources follow their waveform only under `analysis("tran")` (§4.6.1),
    // so every evaluated point declares the transient phase.
    ckt.setSimState(.{ .t = 0, .dt = dt, .kind = .tran });
    if (ckt.has_charge) {
        ckt.eval(x, 0);
        num.copySimd(sc.q_prev[0..n], ckt.q_vec[0..n]);
        // Carry the trapezoid state across the period seam: a converged step
        // leaves i_prev = -f(x), so at x0 it is -rhs. A zero seed loses
        // dt*i0/2 of charge per period. Being a function of x0, it sits
        // inside the shooting Jacobian.
        // ponytail: masked to rows with a diagonal C entry, since an unmasked
        // seed rings undamped on algebraic rows; charge without a diagonal C
        // entry keeps a zero seed.
        sc.i_prev[0] = 0;
        for (1..n) |i| sc.i_prev[i] = if (ckt.c_vals[ckt.diag_slots[i]] != 0) -ckt.rhs[i] else 0;
    }
    if (wave.len != 0) {
        const row = wave[0 .. 1 + probes.len];
        row[0] = 0;
        for (probes, 0..) |node, p| row[1 + p] = x[node];
    }
}

/// The end time of step k of a period.
fn stepTime(options: Options, k: usize) f64 {
    return options.period * @as(f64, @floatFromInt(k + 1)) / @as(f64, @floatFromInt(options.n_samples));
}

/// The companion hook of one step at `alpha` = 2/dt.
fn periodHook(sc: *Companion, alpha: f64, has_charge: bool) PeriodHook {
    return .{
        .method = {},
        .c = .{ .ag0 = alpha, .ag2 = 0 },
        .q_prev = sc.q_prev,
        .i_prev = sc.i_prev,
        .q_cur = sc.q_cur,
        .a_vals = sc.a_vals,
        .has_charge = has_charge,
    };
}

/// Step k's end: a failed Newton reverts the devices and fails the
/// period (false); a converged one commits them, advances the trapezoid
/// history and records wave row k + 1 at t.
fn periodAccept(ckt: *root.Circuit, hook: *const PeriodHook, x: []const f64, probes: []const u32, wave: []f64, k: usize, t: f64, converged: bool) bool {
    if (!converged) {
        _ = ckt.stateCtl(.revert);
        return false;
    }
    _ = ckt.stateCtl(.commit);
    if (hook.has_charge) hook.accept(ckt.n);
    if (wave.len != 0) {
        const ncols = 1 + probes.len;
        const row = wave[(k + 1) * ncols ..][0..ncols];
        row[0] = t;
        for (probes, 0..) |node, p| row[1 + p] = x[node];
    }
    return true;
}

/// Integrates one period from x0 into x_end, leaving x0 untouched.
inline fn integrateFrom(
    ckt: *root.Circuit,
    ws: *converger.Workspace,
    sc: *Companion,
    x0: []const f64,
    x_end: []f64,
    options: Options,
) error{ QueryCancelled, OutOfMemory }!bool {
    num.copySimd(x_end, x0);
    return integrateOnePeriod(ckt, ws, sc, x_end, &.{}, &.{}, options);
}

/// GMRES matvec context: one FD directional derivative of the shooting map.
const ShootingKrylovCtx = struct {
    ckt: *root.Circuit,
    ws: *converger.Workspace,
    sc: *Companion,
    /// Current shooting iterate.
    x0: []const f64,
    /// phi(x0), already computed.
    phi: []const f64,
    /// Scratch, n long: the perturbed start and its endpoint.
    x_pert: []f64,
    x_end_pert: []f64,
    options: Options,
    n: usize,
    /// Set when an integration fails on cancellation or memory, which the
    /// void matvec cannot return; `krylovSolve` returns it.
    failure: ?error{ QueryCancelled, OutOfMemory } = null,

    /// The GMRES operator, `shootingMatvec`.
    pub const matvec = shootingMatvec;
};

/// w = (Phi - I) v by one FD period integration:
/// w = (phi(x0 + eps*v) - phi(x0)) / eps. A failed integration returns
/// w = 0, which GMRES treats as a null direction. Autonomous: v[osc] moves
/// ln T instead of x0[osc].
fn shootingMatvec(ctx: *ShootingKrylovCtx, v: []const f64, w: []f64) void {
    const n = ctx.n;
    const inv_eps = 1.0 / ctx.options.fd_epsilon;

    num.copySimd(ctx.x_pert[0..n], ctx.x0[0..n]);
    num.axpy(ctx.x_pert[0..n], ctx.options.fd_epsilon, v[0..n]);
    var opts = ctx.options;
    const osc = opts.osc_node;
    if (osc != root.GROUND) {
        ctx.x_pert[osc] = ctx.x0[osc];
        opts.period *= 1 + ctx.options.fd_epsilon * v[osc];
    }

    const ok = integrateFrom(ctx.ckt, ctx.ws, ctx.sc, ctx.x_pert, ctx.x_end_pert, opts) catch |err| blk: {
        ctx.failure = ctx.failure orelse err;
        break :blk false;
    };
    if (!ok) {
        // ponytail: a zero column is safer than propagating NaN.
        @memset(w[0..n], 0);
        return;
    }
    fdColumn(w[0..n], ctx.x_end_pert, ctx.x_pert, ctx.phi, inv_eps);
}

/// out = ((x_end - x_start) - phi) * inv_eps, the FD derivative of phi;
/// `out` may alias `x_end`.
fn fdColumn(out: []f64, x_end: []const f64, x_start: []const f64, phi: []const f64, inv_eps: f64) void {
    num.sub(out, x_end, x_start);
    num.sub(out, out, phi);
    num.scale(out, inv_eps, out);
}

/// f64s the monodromy recurrence may hold (planes and trajectory) before the
/// solve keeps the finite-difference columns instead: 512 MB.
/// ponytail: one fixed budget; store factors instead of planes, or
/// checkpoint the trajectory, if long periods on large circuits need it.
const mono_budget: usize = 1 << 26;

/// The shooting Jacobian Φ = dx(T)/dx0 by the trapezoid's own sensitivity
/// recurrence over one recorded period (dev/solvers/monodromy-krylov.md),
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
const Mono = struct {
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

    fn init(gpa: std.mem.Allocator, ckt: *const root.Circuit, steps: usize, autonomous: bool) !?Mono {
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

    fn deinit(m: *Mono, gpa: std.mem.Allocator) void {
        m.slv.deinit();
        inline for (.{ m.traj, m.g0, m.c, m.a, m.q, m.ft, m.work }) |b| gpa.free(b);
        gpa.free(m.rows);
    }

    /// Evaluates G and C at every recorded x_k of the base period, and for
    /// an autonomous circuit q_k and t_k ∂f/∂t (one more eval per step).
    fn build(m: *Mono, ckt: *root.Circuit, opts: Options) void {
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

    /// y = A x over the circuit pattern.
    fn spmv(ckt: *const root.Circuit, vals: []const f64, x: []const f64, y: []f64) void {
        @memset(y, 0);
        for (0..ckt.n) |j| for (ckt.col_ptr[j]..ckt.col_ptr[j + 1]) |p| {
            y[ckt.row_idx[p]] += vals[p] * x[j];
        };
    }

    /// Replaces each of the r columns of `blk` (column-major, n each) by
    /// Φ times it; column `period_col`, which must enter zero, receives
    /// d x_N / d(ln T) instead (autonomous `Mono` only). False when some
    /// A_k is singular (`blk` is garbage then).
    fn apply(m: *Mono, gpa: std.mem.Allocator, ckt: *const root.Circuit, opts: Options, blk: []f64, r: usize, period_col: ?usize) error{OutOfMemory}!bool {
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
};

/// GMRES operator on the recorded monodromy: w = (Φ - I) v, the
/// autonomous period column from `dphi_dlnt`. A singular step Jacobian
/// sets `singular` and returns w = 0; the caller then falls back.
const MonoKrylovCtx = struct {
    mono: *Mono,
    gpa: std.mem.Allocator,
    ckt: *const root.Circuit,
    options: Options,
    n: usize,
    dphi_dlnt: []const f64,
    singular: bool = false,
    failure: ?error{OutOfMemory} = null,

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

/// Krylov path: solves (Phi - I) dx0 = -phi with matrix-free GMRES from a
/// zero start.
fn krylovSolve(
    ckt: *root.Circuit,
    ws: *converger.Workspace,
    sc: *Companion,
    n: usize,
    x0: []const f64,
    phi: []const f64,
    x_pert: []f64,
    x_end_pert: []f64,
    dx0: []f64,
    neg_phi: []f64,
    krylov: *Gmres,
    options: Options,
) error{ QueryCancelled, OutOfMemory }!void {
    num.scale(neg_phi[0..n], -1.0, phi[0..n]);

    var ctx = ShootingKrylovCtx{
        .ckt = ckt,
        .ws = ws,
        .sc = sc,
        .x0 = x0,
        .phi = phi,
        .x_pert = x_pert,
        .x_end_pert = x_end_pert,
        .options = options,
        .n = n,
    };

    @memset(dx0[0..n], 0);

    _ = krylov.solve(&ctx, neg_phi[0..n], dx0[0..n], options.gmres_tol, options.gmres_max_restarts);
    if (ctx.failure) |err| return err;
}

/// One shooting solve between its period integrations, so `solve` and
/// `runLanes` share every step but the integration: `next` names the
/// period the solve needs, the caller integrates it, `done` consumes the
/// outcome. The statements are `solve`'s own, in its order. Pinned once
/// `init` ran: `sc` and `krylov` are reached by pointer.
const Shoot = struct {
    ckt: *root.Circuit,
    ws: *converger.Workspace,
    probes: []const u32,
    wave: []f64,
    /// The query's options; `opts` carries the period the solve is at.
    options: Options,
    opts: Options,
    allocator: std.mem.Allocator,
    n: usize,
    arena: []f64,
    x0: []f64,
    x_end: []f64,
    phi: []f64,
    x0_pert: []f64,
    x_end_pert: []f64,
    dx0: []f64,
    /// Dense path's FD Jacobian (n*n), else empty.
    j_phi: []f64,
    /// Krylov path's right-hand side (n), else empty.
    neg_phi: []f64,
    sc: Companion,
    krylov: ?Gmres,
    /// The monodromy recurrence, unless over `mono_budget`.
    mono: ?Mono = null,
    iter: u16 = 0,
    res_norm: f64 = std.math.inf(f64),
    stage: Stage = .base,
    /// The FD column the `.column` stage integrates next.
    col: usize = 0,
    /// Set when `stage` reaches `.done`.
    result: SolveResult = undefined,

    const Stage = enum { base, column, final, done };

    /// One period integration: `x` holds its start and receives its end;
    /// a non-empty `wave` records it as `integrateOnePeriod` documents.
    const Request = struct { x: []f64, probes: []const u32, wave: []f64, opts: Options };

    /// `solve`'s setup: buffers, the GMRES workspace past
    /// `krylov_threshold`, the baseline, and for an autonomous circuit the
    /// oscillator start. Errors as `solve`'s; on error nothing is held.
    fn init(s: *Shoot, ckt: *root.Circuit, x_dc: []const f64, probes: []const u32, wave: []f64, options: Options, allocator: std.mem.Allocator) !void {
        const n: usize = ckt.n;
        const nnz: usize = ckt.nnz;
        const ncols = 1 + probes.len;
        std.debug.assert(wave.len == 0 or wave.len == (@as(usize, options.n_samples) + 1) * ncols);
        @memset(wave, 0);

        const use_krylov = n >= krylov_threshold;
        // One block for the whole continuation, every size known up front:
        //   x0, x_end, phi, x0_pert, x_end_pert, dx0   6n
        //   q_prev, q_cur, i_prev                      3n
        //   a_vals                                     nnz
        //   j_phi (dense) or neg_phi (Krylov)          n*n or n
        const dense_extra = if (!use_krylov) n * n else 0;
        const krylov_extra = if (use_krylov) n else 0;
        const arena = try allocator.alloc(f64, 9 * n + nnz + dense_extra + krylov_extra);
        errdefer allocator.free(arena);
        var off: usize = 0;
        const x0 = take(arena, &off, n);
        const x_end = take(arena, &off, n);
        const phi = take(arena, &off, n);
        const x0_pert = take(arena, &off, n);
        const x_end_pert = take(arena, &off, n);
        const dx0 = take(arena, &off, n);
        const sc: Companion = .{
            .q_prev = take(arena, &off, n),
            .q_cur = take(arena, &off, n),
            .i_prev = take(arena, &off, n),
            .a_vals = take(arena, &off, nnz),
        };
        // The unused path's buffer slices the arena to zero length so both
        // stay mutable []f64.
        const j_phi = if (!use_krylov) arena[off..][0 .. n * n] else arena[0..0];
        const neg_phi = if (use_krylov) arena[off..][0..n] else arena[0..0];
        var krylov: ?Gmres = null;
        if (use_krylov) {
            const m = @min(options.gmres_restart, @as(u32, @intCast(n)));
            krylov = try Gmres.init(allocator, @intCast(n), m);
        }
        errdefer if (krylov) |*k| k.deinit(allocator);

        var mono = try Mono.init(allocator, ckt, options.n_samples, options.osc_node != root.GROUND);
        errdefer if (mono) |*m| m.deinit(allocator);
        try ckt.computeBaseline();
        const ws = try ckt.workspace();
        s.* = .{
            .mono = mono,
            .ckt = ckt,
            .ws = ws,
            .probes = probes,
            .wave = wave,
            .options = options,
            .opts = options,
            .allocator = allocator,
            .n = n,
            .arena = arena,
            .x0 = x0,
            .x_end = x_end,
            .phi = phi,
            .x0_pert = x0_pert,
            .x_end_pert = x_end_pert,
            .dx0 = dx0,
            .j_phi = j_phi,
            .neg_phi = neg_phi,
            .sc = sc,
            .krylov = krylov,
        };
        num.copySimd(x0, x_dc[0..n]);
        if (options.osc_node != root.GROUND) s.opts.period = try startOscillator(ckt, ws, &s.sc, x0, options, allocator);
    }

    fn deinit(s: *Shoot) void {
        if (s.mono) |*m| m.deinit(s.allocator);
        if (s.krylov) |*k| k.deinit(s.allocator);
        s.allocator.free(s.arena);
        s.* = undefined;
    }

    /// The next period the solve needs, or null once `result` is set.
    /// Fails only when a checkpoint cancels.
    fn next(s: *Shoot) error{QueryCancelled}!?Request {
        const n = s.n;
        sw: switch (s.stage) {
            .base => {
                if (s.iter >= s.options.max_shooting_iter) {
                    s.stage = .final;
                    continue :sw .final;
                }
                if (s.iter != 0) try s.ckt.checkpoint(.{ .phase = .periodic, .completed = s.iter });
                // integrateFrom(x0 -> x_end), recording the trajectory
                // the monodromy recurrence linearizes around.
                num.copySimd(s.x_end, s.x0);
                if (s.mono) |*m| return .{ .x = s.x_end, .probes = m.rows, .wave = m.traj, .opts = s.opts };
                return .{ .x = s.x_end, .probes = &.{}, .wave = &.{}, .opts = s.opts };
            },
            .column => {
                // denseFdSolve's column `col`.
                const j = s.col;
                num.copySimd(s.x0_pert[0..n], s.x0[0..n]);
                var o = s.opts;
                if (o.osc_node != root.GROUND and j == o.osc_node) o.period *= 1 + o.fd_epsilon else s.x0_pert[j] += o.fd_epsilon;
                num.copySimd(s.x_end_pert, s.x0_pert);
                return .{ .x = s.x_end_pert, .probes = &.{}, .wave = &.{}, .opts = o };
            },
            .final => {
                num.copySimd(s.x_end, s.x0);
                return .{ .x = s.x_end, .probes = s.probes, .wave = s.wave, .opts = s.opts };
            },
            .done => return null,
        }
    }

    /// Consumes the outcome of the request `next` returned: `ok` is that
    /// integration's return. Errors as `solve`'s: a Krylov solve's
    /// cancellation or memory, `dense_lu`'s singular Jacobian.
    fn done(s: *Shoot, ok: bool) !void {
        const n = s.n;
        switch (s.stage) {
            .base => {
                if (!ok) {
                    s.result = .{ .converged = false, .iterations = s.iter, .residual_norm = std.math.inf(f64) };
                    s.stage = .done;
                    return;
                }
                // x_end + (-1)*x0 is exactly x_end - x0.
                num.copySimd(s.phi, s.x_end);
                num.axpy(s.phi, -1.0, s.x0);
                s.res_norm = num.normInf(s.phi);
                if (s.res_norm < s.options.shooting_tol) {
                    s.stage = .final;
                    return;
                }
                if (try s.monoStep()) return;
                if (s.krylov) |*k| {
                    // ponytail: unpreconditioned; add an ILU preconditioner if GMRES stalls.
                    try krylovSolve(s.ckt, s.ws, &s.sc, n, s.x0, s.phi, s.x0_pert, s.x_end_pert, s.dx0, s.neg_phi, k, s.opts);
                    s.update();
                } else {
                    s.stage = .column;
                    s.col = 0;
                }
            },
            .column => {
                const j = s.col;
                if (!ok) {
                    for (0..n) |row| s.j_phi[row * n + j] = if (row == j) @as(f64, 1.0) else @as(f64, 0.0);
                } else {
                    // Column j is stride n in row-major J, so the store stays scalar.
                    const col = s.x_end_pert[0..n];
                    fdColumn(col, col, s.x0_pert[0..n], s.phi[0..n], 1.0 / s.opts.fd_epsilon);
                    for (col, 0..) |d, row| s.j_phi[row * n + j] = d;
                }
                s.col += 1;
                if (s.col < n) return;
                try dense_lu.factorizeSolveNeg(n, s.j_phi, s.phi[0..n], s.dx0);
                s.stage = .base;
                s.update();
            },
            .final => {
                if (!ok) @memset(s.wave, 0);
                s.result = .{ .converged = s.res_norm < s.options.shooting_tol, .iterations = s.iter, .residual_norm = s.res_norm };
                s.stage = .done;
            },
            .done => unreachable,
        }
    }

    /// The Newton step on the recorded monodromy (`Mono`): every column of
    /// the dense Jacobian in one recurrence pass, or GMRES on (Φ - I) with
    /// one pass per matvec. The autonomous period column is one more
    /// recurrence column. False, nothing changed, when there is no record
    /// or a step Jacobian is singular: the caller keeps its FD path.
    fn monoStep(s: *Shoot) !bool {
        const m = if (s.mono) |*m| m else return false;
        const n = s.n;
        m.build(s.ckt, s.opts);
        defer m.ready = false;
        const osc = s.opts.osc_node;
        if (s.krylov) |*k| {
            // d(phi)/d(ln T) into x_end_pert.
            if (osc != root.GROUND) {
                @memset(s.x_end_pert[0..n], 0);
                if (!try m.apply(s.allocator, s.ckt, s.opts, s.x_end_pert[0..n], 1, 0)) return false;
            }
            num.scale(s.neg_phi[0..n], -1.0, s.phi[0..n]);
            var ctx: MonoKrylovCtx = .{ .mono = m, .gpa = s.allocator, .ckt = s.ckt, .options = s.opts, .n = n, .dphi_dlnt = s.x_end_pert[0..n] };
            @memset(s.dx0[0..n], 0);
            _ = k.solve(&ctx, s.neg_phi[0..n], s.dx0[0..n], s.options.gmres_tol, s.options.gmres_max_restarts);
            if (ctx.failure) |err| return err;
            if (ctx.singular) return false;
            s.update();
            return true;
        }
        // Dense: J = Φ - I, row-major, from the n unit columns.
        const blk = try s.allocator.alloc(f64, n * n);
        defer s.allocator.free(blk);
        @memset(blk, 0);
        for (0..n) |j| blk[j * n + j] = if (j == osc and osc != root.GROUND) 0 else 1;
        const period_col: ?usize = if (osc != root.GROUND) osc else null;
        if (!try m.apply(s.allocator, s.ckt, s.opts, blk, n, period_col)) return false;
        for (0..n) |j| for (0..n) |row| {
            const col_osc = j == osc and osc != root.GROUND;
            s.j_phi[row * n + j] = blk[j * n + row] - @as(f64, if (row == j and !col_osc) 1 else 0);
        };
        try dense_lu.factorizeSolveNeg(n, s.j_phi, s.phi[0..n], s.dx0);
        s.update();
        return true;
    }

    /// The damped Newton update that ends a shooting iteration.
    fn update(s: *Shoot) void {
        var damping: f64 = 1.0;
        const osc = s.options.osc_node;
        if (osc != root.GROUND) {
            const d_ln_t = s.dx0[osc];
            s.dx0[osc] = 0;
            if (@abs(d_ln_t) > max_period_step) damping = max_period_step / @abs(d_ln_t);
            s.opts.period *= 1 + damping * d_ln_t;
        }
        num.axpy(s.x0, damping, s.dx0);
        s.iter += 1;
    }
};

/// Shooting Newton from x_dc. A non-empty `wave`, sized
/// (n_samples+1) * (1 + probes.len), receives point-major rows
/// [t, v(probes)...] of one period from the final x0, converged or not; it is
/// left zeroed if that integration fails. An autonomous solve starts with
/// `startOscillator` and returns its period as the last row's t;
/// error.OscillatorDidNotStart when the circuit does not oscillate.
/// Allocates O(n^2) below `krylov_threshold` unknowns, O(n * gmres_restart)
/// past it, and O(n * n_samples) more to start an oscillator, all freed
/// before return. Asserts `wave` is empty or that size. A period whose
/// integration fails ends the solve unconverged with an infinite residual,
/// not an error; the errors are OutOfMemory, QueryCancelled, error.Singular
/// from the dense Jacobian, and error.OscillatorDidNotStart.
pub fn solve(
    ckt: *root.Circuit,
    x_dc: []const f64,
    probes: []const u32,
    wave: []f64,
    options: Options,
    allocator: std.mem.Allocator,
) !SolveResult {
    var s: Shoot = undefined;
    try s.init(ckt, x_dc, probes, wave, options, allocator);
    defer s.deinit();
    while (try s.next()) |r| try s.done(try integrateOnePeriod(ckt, s.ws, &s.sc, r.x, r.probes, r.wave, r.opts));
    return s.result;
}

/// Starts an oscillator for the autonomous solve: kicks `x` (the DC point)
/// off its equilibrium at the osc node, integrates `osc_settle_periods`
/// guess periods and then until the swing settles (at most
/// `max_settle_periods`), then records `measure_periods` more. The period
/// is the mean spacing of the osc node's rising mid-swing crossings in
/// that record. Leaves `x` at the first sample after the last crossing, the
/// phase anchor the solve pins, and returns the period.
fn startOscillator(
    ckt: *root.Circuit,
    ws: *converger.Workspace,
    sc: *Companion,
    x: []f64,
    options: Options,
    allocator: std.mem.Allocator,
) !f64 {
    const n: usize = ckt.n;
    const osc = options.osc_node;
    const steps: usize = options.n_samples;
    x[osc] += osc_kick;
    for (0..options.osc_settle_periods) |_|
        if (!try integrateOnePeriod(ckt, ws, sc, x, &.{}, &.{}, options)) return error.OscillatorDidNotStart;
    // A weakly unstable tank grows from the kick over many periods: keep
    // settling until the osc node's swing moves under 1 % a period.
    const swing = try allocator.alloc(f64, 2 * (steps + 1));
    defer allocator.free(swing);
    var prev_swing: f64 = 0;
    for (0..max_settle_periods) |_| {
        if (!try integrateOnePeriod(ckt, ws, sc, x, &.{osc}, swing, options)) return error.OscillatorDidNotStart;
        var lo = std.math.inf(f64);
        var hi = -std.math.inf(f64);
        for (0..steps + 1) |k| {
            lo = @min(lo, swing[2 * k + 1]);
            hi = @max(hi, swing[2 * k + 1]);
        }
        if (@abs(hi - lo - prev_swing) <= 0.01 * (hi - lo)) break;
        prev_swing = hi - lo;
    }

    const rows = try allocator.alloc(u32, n);
    defer allocator.free(rows);
    for (rows, 0..) |*r, i| r.* = @intCast(i);
    const stride = (steps + 1) * (n + 1);
    const record = try allocator.alloc(f64, measure_periods * stride);
    defer allocator.free(record);
    for (0..measure_periods) |p|
        if (!try integrateOnePeriod(ckt, ws, sc, x, rows, record[p * stride ..][0..stride], options)) return error.OscillatorDidNotStart;

    // Sample g of the record is row g % steps of period g / steps; the last
    // row of each period repeats the next period's first.
    const total = measure_periods * steps + 1;
    const Sample = struct {
        fn row(rec: []const f64, g: usize, s: usize, cols: usize) []const f64 {
            const p = @min(g / s, measure_periods - 1);
            return rec[p * (s + 1) * cols + (g - p * s) * cols + 1 ..][0 .. cols - 1];
        }
    };
    var lo = std.math.inf(f64);
    var hi = -std.math.inf(f64);
    for (0..total) |g| {
        const v = Sample.row(record, g, steps, n + 1)[osc];
        lo = @min(lo, v);
        hi = @max(hi, v);
    }
    const level = 0.5 * (lo + hi);
    var first: f64 = 0;
    var last: f64 = 0;
    var anchor: usize = 0;
    var crossings: u32 = 0;
    var prev = Sample.row(record, 0, steps, n + 1)[osc];
    for (1..total) |g| {
        const v = Sample.row(record, g, steps, n + 1)[osc];
        if (prev < level and v >= level) {
            const at = @as(f64, @floatFromInt(g - 1)) + (level - prev) / (v - prev);
            if (crossings == 0) first = at;
            last = at;
            anchor = g;
            crossings += 1;
        }
        prev = v;
    }
    if (crossings < 2 or !(hi - lo > osc_kick)) return error.OscillatorDidNotStart;
    num.copySimd(x[0..n], Sample.row(record, anchor, steps, n + 1));
    const dt = options.period / @as(f64, @floatFromInt(steps));
    return (last - first) / @as(f64, @floatFromInt(crossings - 1)) * dt;
}

/// Contract entry: shoot from ctx.x_op and return one period as point-major
/// rows (time, probes...), n_samples+1 of them. Non-convergence prints a
/// warning and still returns the last period.
pub fn run(ctx: *const root.RunCtx, opts: Options) !root.Result {
    try ctx.circuit.refuseDigital("pss");
    const names = try root.probeNames(ctx, "time");
    const data = try ctx.allocator.alloc(f64, (@as(usize, opts.n_samples) + 1) * names.len);
    errdefer ctx.allocator.free(data);
    const res = try solve(ctx.circuit, ctx.x_op, ctx.probes, data, opts, ctx.scratch_allocator);
    return finish(res, names, data, opts);
}

/// `run`'s setup for a lane: the same refusals, names and row buffer,
/// then the shooting state in `shot`.
fn start(ctx: *const root.RunCtx, opts: Options, shot: *Shoot, names: *[]const []const u8, data: *[]f64) !void {
    try ctx.circuit.refuseDigital("pss");
    names.* = try root.probeNames(ctx, "time");
    data.* = try ctx.allocator.alloc(f64, (@as(usize, opts.n_samples) + 1) * names.len);
    errdefer ctx.allocator.free(data.*);
    try shot.init(ctx.circuit, ctx.x_op, ctx.probes, data.*, opts, ctx.scratch_allocator);
}

/// The published period. Non-convergence prints a warning and still
/// returns the last period.
fn finish(res: SolveResult, names: []const []const u8, data: []f64, opts: Options) root.Result {
    if (!res.converged)
        std.debug.print("Warning: pss: shooting did not converge (residual {e})\n", .{res.residual_norm});
    return .{
        .plotname = "Periodic Steady State",
        .varnames = names,
        .is_complex = false,
        .npoints = @as(usize, opts.n_samples) + 1,
        .data = data,
    };
}

/// `integrateOnePeriod` for every lane of `m`, lane l on `shots[l]` and
/// request `reqs[l]`: the start, commit/revert, history and wave rows run
/// per lane and each step's Newton runs as lanes, so lane l is bitwise its
/// scalar integration. Returns the lanes whose period completed. Every
/// request takes the same `n_samples` (one batch shares its options; an
/// autonomous lane's own period changes only its t and dt). Cancellation
/// and memory fail every lane.
fn integrateLanes(
    comptime W: usize,
    sys: *LaneSys(W),
    lws: *converger.LaneWorkspace(W),
    xv: []Lanes(W).V,
    shots: *[W]Shoot,
    reqs: *const [W]Shoot.Request,
    m: Lanes(W).Mask,
) !Lanes(W).Mask {
    const L = Lanes(W);
    const S = LaneSys(W);
    const steps: usize = reqs[@ctz(m)].opts.n_samples;
    var it = L.each(m);
    while (it.next()) |l| {
        std.debug.assert(reqs[l].opts.n_samples == steps);
        periodStart(shots[l].ckt, &shots[l].sc, reqs[l].x, reqs[l].probes, reqs[l].wave, reqs[l].opts);
    }
    var live = m;
    var hooks: [W]PeriodHook = undefined;
    var times: [W]f64 = @splat(0);
    for (0..steps) |k| {
        if (!L.any(live)) break;
        it = L.each(live);
        while (it.next()) |l| {
            const o = reqs[l].opts;
            const ckt = shots[l].ckt;
            const dt = o.period / @as(f64, @floatFromInt(steps));
            times[l] = stepTime(o, k);
            hooks[l] = periodHook(&shots[l].sc, 2.0 / dt, ckt.has_charge);
            sys.hooks[l] = hooks[l];
            ckt.setSimState(.{ .t = times[l], .dt = dt, .kind = .tran });
            S.putLane(xv, reqs[l].x, l);
        }
        sys.active = live;
        const o0 = reqs[@ctz(live)].opts;
        const res = converger.newtonLanes(W, sys, lws, xv, times, .{
            .max_iter = o0.max_newton_iter,
            .abstol = o0.newton_tol,
        }, S.LaneHook{});
        // As `converger.run` ends each lane's solve.
        it = L.each(live);
        while (it.next()) |l| shots[l].ckt.clearLimits();
        const r = try res;
        it = L.each(live);
        while (it.next()) |l| {
            S.getLane(reqs[l].x, xv, l);
            const bit = @as(L.Mask, 1) << @intCast(l);
            const ok = r.converged & bit != 0 and r.singular & bit == 0;
            if (!periodAccept(shots[l].ckt, &hooks[l], reqs[l].x, reqs[l].probes, reqs[l].wave, k, times[l], ok)) live &= ~bit;
        }
    }
    return live;
}

/// `run` for the lanes of one batch: `.pss` queries on one pattern, each
/// from its own operating point and variant row (#210). Each lane keeps
/// its own shooting loop, FD columns, monodromy LU, GMRES and period; the
/// period integrations of a round run together, one lane Newton per time
/// step (`integrateLanes`). A Krylov lane's matvecs run scalar inside its
/// own `done`. Lane l's result lands in `out[l]`, bitwise `run` on
/// `ctxs[l]`. Under a JFNK pin or `-Dcertify` every lane runs `run`.
/// Asserts 1 <= `ctxs.len` <= W.
pub fn runLanes(comptime W: usize, ctxs: []const root.RunCtx, opts: Options, out: []LaneOutcome) void {
    const g = ctxs.len;
    std.debug.assert(g >= 1 and g <= W and out.len == g);
    const pin = ctxs[0].circuit.knobs.newton.pin;
    if (W == 1 or g == 1 or !(pin == .auto or pin == .direct) or root.Circuit.certify) {
        for (ctxs, out) |*c, *o| o.* = run(c, opts);
        return;
    }
    const L = Lanes(W);
    const S = LaneSys(W);
    var shots: [W]Shoot = undefined;
    var names: [W][]const []const u8 = undefined;
    var data: [W][]f64 = undefined;
    var live: L.Mask = 0;
    for (ctxs, 0..) |*c, l| {
        start(c, opts, &shots[l], &names[l], &data[l]) catch |e| {
            out[l] = e;
            continue;
        };
        live |= @as(L.Mask, 1) << @intCast(l);
    }
    defer {
        var it = L.each(live);
        while (it.next()) |l| shots[l].deinit();
    }
    const kill = struct {
        fn f(sh: []Shoot, m: *L.Mask, o: []LaneOutcome, l: usize, err: LaneError) void {
            o[l] = err;
            sh[l].deinit();
            m.* &= ~(@as(L.Mask, 1) << @intCast(l));
        }
    }.f;
    // Every lane fails with the batch's own error (cancel, out of memory).
    const fail = struct {
        fn f(sh: []Shoot, m: *L.Mask, o: []LaneOutcome, err: LaneError) void {
            var it = L.each(m.*);
            while (it.next()) |l| {
                o[l] = err;
                sh[l].deinit();
            }
            m.* = 0;
        }
    }.f;
    if (live == 0) return;
    const first = @ctz(live);
    var lanes: [W]*root.Circuit = undefined;
    var wss: [W]*converger.Workspace = undefined;
    for (&lanes, &wss, 0..) |*c, *w, l| {
        // Unused lanes repeat the first live one and stay inactive.
        const k = if (l < g and (live >> @intCast(l)) & 1 != 0) l else first;
        c.* = shots[k].ckt;
        w.* = shots[k].ws;
    }
    const gpa = ctxs[first].scratch_allocator;
    // The hook only sizes the value plane here (alpha 0 rewrites `a_vals`
    // as G, which every assemble overwrites); each step sets its own.
    var sys = S.init(gpa, lanes, periodHook(&shots[first].sc, 0, lanes[first].has_charge)) catch |e| return fail(&shots, &live, out, e);
    defer sys.deinit(gpa);
    var lws = converger.LaneWorkspace(W).init(gpa, wss) catch |e| return fail(&shots, &live, out, e);
    defer lws.deinit(gpa);
    const xv = gpa.alloc(L.V, lanes[first].n) catch |e| return fail(&shots, &live, out, e);
    defer gpa.free(xv);

    var reqs: [W]Shoot.Request = undefined;
    while (true) {
        var round: L.Mask = 0;
        var it = L.each(live);
        while (it.next()) |l| {
            const bit = @as(L.Mask, 1) << @intCast(l);
            const r = shots[l].next() catch |e| {
                kill(&shots, &live, out, l, e);
                continue;
            };
            if (r) |req| {
                reqs[l] = req;
                round |= bit;
                continue;
            }
            out[l] = finish(shots[l].result, names[l], data[l], opts);
            shots[l].deinit();
            live &= ~bit;
        }
        if (round == 0) break;
        const ok = integrateLanes(W, &sys, &lws, xv, &shots, &reqs, round) catch |e| return fail(&shots, &live, out, e);
        it = L.each(round);
        while (it.next()) |l| shots[l].done(ok & (@as(L.Mask, 1) << @intCast(l)) != 0) catch |e| kill(&shots, &live, out, l, e);
    }
}

/// The fixed-period GPU driver has no host callbacks or stateful-charge
/// rebase. State flips discovered during a period rerun it on the host
/// from its unchanged start.
fn gpuRefusal(ckt: *const root.Circuit) ?[]const u8 {
    if (!gpu_lanes.available) return "no GPU images";
    if (ckt.vpi != null or ckt.display != null or ckt.load_check != null) return "host callbacks";
    if (ckt.minDelay() != null or ckt.has_state_q) return "stateful charge or delay";
    for (ckt.batches) |b| {
        const h = b.hooks;
        if (h.gpu_payload == null or h.commit_state != null or h.predict_first_iterate != null or
            h.check_convergence != null or h.advance_iteration != null or
            std.mem.eql(u8, b.type_name, "kinduc")) return b.type_name;
        const p = h.gpu_payload.?(b.ctx);
        // Timer-only sources carry generated State bytes but their host
        // hooks are intentionally null (device/eval.zig skipsTimerState).
        // Gate the work that runs, not the presence of unused storage.
        if ((h.update_state != null and p.lim_kernel.len == 0) or
            (h.state_ctl != null and p.ctl_kernel.len == 0))
            return "device shooting state";
    }
    return null;
}

/// Arbitrary-width shooting lanes. Each lane retains the same host Shoot
/// state machine as runLanes; each round's fixed periods run on the device.
/// Unsupported circuits and Newton pins use the scalar path explicitly.
/// Krylov directional integrations inside Shoot.done remain on the host.
pub fn runGpuLanes(ctxs: []const root.RunCtx, opts: Options, out: []LaneOutcome) void {
    std.debug.assert(ctxs.len > 0 and out.len == ctxs.len);
    if (comptime !gpu_lanes.available) {
        for (ctxs, out) |*c, *o| o.* = run(c, opts);
        return;
    }
    const pin = ctxs[0].circuit.knobs.newton.pin;
    var refusal: ?[]const u8 = if (!(pin == .auto or pin == .direct) or root.Circuit.certify) "Newton pin or certification" else null;
    for (ctxs) |c| {
        if (refusal != null) break;
        refusal = gpuRefusal(c.circuit);
        if (c.circuit.n != ctxs[0].circuit.n or c.circuit.nnz != ctxs[0].circuit.nnz) refusal = "different patterns";
    }
    if (refusal) |why| {
        if (ctxs[0].circuit.knobs.gpu_stats != .off) std.debug.print("gpu-pss: off ({s})\n", .{why});
        for (ctxs, out) |*c, *o| o.* = run(c, opts);
        return;
    }
    gpuShots(ctxs, opts, out) catch |err| for (out) |*o| {
        o.* = err;
    };
}

fn gpuShots(ctxs: []const root.RunCtx, opts: Options, out: []LaneOutcome) LaneError!void {
    const gpa = ctxs[0].scratch_allocator;
    const N = ctxs.len;
    const shots = try gpa.alloc(Shoot, N);
    defer gpa.free(shots);
    const live = try gpa.alloc(bool, N);
    defer gpa.free(live);
    @memset(live, false);
    defer for (shots, live) |*s, on| if (on) s.deinit();
    const names = try gpa.alloc([]const []const u8, N);
    defer gpa.free(names);
    const data = try gpa.alloc([]f64, N);
    defer gpa.free(data);
    const requests = try gpa.alloc(Shoot.Request, N);
    defer gpa.free(requests);
    const periods = try gpa.alloc(gpu_lanes.FixedLane, N);
    defer gpa.free(periods);
    const indices = try gpa.alloc(usize, N);
    defer gpa.free(indices);
    const ok = try gpa.alloc(bool, N);
    defer gpa.free(ok);
    for (ctxs, 0..) |*c, l| {
        start(c, opts, &shots[l], &names[l], &data[l]) catch |err| {
            out[l] = err;
            continue;
        };
        live[l] = true;
    }
    while (true) {
        var count: usize = 0;
        for (shots, live, 0..) |*s, *on, l| {
            if (!on.*) continue;
            const r = s.next() catch |err| {
                out[l] = err;
                s.deinit();
                on.* = false;
                continue;
            };
            if (r) |req| {
                requests[count] = req;
                indices[count] = l;
                periodStart(s.ckt, &s.sc, req.x, req.probes, req.wave, req.opts);
                periods[count] = .{
                    .ckt = s.ckt,
                    .x = req.x,
                    .q_prev = s.sc.q_prev,
                    .i_prev = s.sc.i_prev,
                    .period = req.opts.period,
                    .n_samples = req.opts.n_samples,
                    .probes = req.probes,
                    .wave = req.wave,
                };
                count += 1;
            } else {
                out[l] = finish(s.result, names[l], data[l], opts);
                s.deinit();
                on.* = false;
            }
        }
        if (count == 0) break;
        const options: converger.Options = .{ .max_iter = opts.max_newton_iter, .abstol = opts.newton_tol };
        gpu_lanes.integrateFixed(gpa, periods[0..count], options, ok[0..count]) catch |err| switch (err) {
            error.QueryCancelled, error.OutOfMemory => |e| return e,
            else => {
                std.log.warn("gpu pss: {s}; running {d} periods on the CPU", .{ @errorName(err), count });
                for (indices[0..count], requests[0..count], ok[0..count]) |l, req, *done| {
                    const s = &shots[l];
                    done.* = try integrateOnePeriod(s.ckt, s.ws, &s.sc, req.x, req.probes, req.wave, req.opts);
                }
            },
        };
        for (indices[0..count], ok[0..count]) |l, done| shots[l].done(done) catch |err| {
            out[l] = err;
            shots[l].deinit();
            live[l] = false;
        };
    }
}

const LaneError = root.ErrorOf(run);
/// One lane's `run` outcome.
pub const LaneOutcome = LaneError!root.Result;

/// Private implementation access for the analysis test suite; void outside tests.
pub const test_access = if (@import("builtin").is_test) .{
    .krylov_threshold = krylov_threshold,
} else {};
