//! One shooting solve between its period integrations: the DC point (or,
//! for an oscillator, its started orbit) in, a periodic x0, its period and
//! the solve's status out. `next` names the period the solve needs, the
//! caller integrates it (on the host, as lanes, or on the GPU), `done`
//! consumes the outcome; the Newton system (Phi - I) dx0 = -phi is the
//! monodromy recurrence (`Mono.zig`), a dense finite-difference Jacobian
//! below `krylov_threshold` unknowns, or JFNK GMRES. Pinned once `init`
//! ran: `sc` and `krylov` are reached by pointer.
//! Cites: none beyond pss.zig's.
//!
//! Contents:
//! - `pub const Request = struct`
//! - `pub const krylov_threshold: usize`
//! - `ShootingKrylovCtx.matvec` (the GMRES operator, `shootingMatvec`)
//! - `pub fn init(s: *Shoot, ckt: *root.Circuit, x_dc: []const f64, probes: []const u32, wave: []f64, options: Options, allocator: std.mem.Allocator) !void`
//! - `pub fn deinit(s: *Shoot) void`
//! - `pub fn next(s: *Shoot) error{QueryCancelled}!?Request`
//! - `pub fn done(s: *Shoot, ok: bool) !void`
//! - `pub fn runAlone(s: *Shoot) !SolveResult`

const std = @import("std");
const num = @import("core").numerics;
const converger = @import("solver").converger;
const dense_lu = @import("solver").dense_lu;
const Gmres = @import("solver").gmres.Gmres;
const root = @import("../types.zig");
const Mono = @import("Mono.zig");
const period = @import("period.zig");
const pss = @import("pss.zig");

const Companion = period.Companion;
const MonoKrylovCtx = Mono.MonoKrylovCtx;
const Options = pss.Options;
const SolveResult = pss.SolveResult;
const integrateOnePeriod = period.integrateOnePeriod;
const take = root.take;

/// One shooting solve between its period integrations, so `solve` and
/// `runLanes` share every step but the integration: `next` names the
/// period the solve needs, the caller integrates it, `done` consumes the
/// outcome. The statements are `solve`'s own, in its order. Pinned once
/// `init` ran: `sc` and `krylov` are reached by pointer.
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

/// One period integration: `x` holds its start and receives its end;
/// a non-empty `wave` records it as `integrateOnePeriod` documents.
pub const Request = struct { x: []f64, probes: []const u32, wave: []f64, opts: Options };

/// ponytail: node count at which the Newton solve switches from the dense FD
/// Jacobian (n+1 period integrations, O(n^2) storage) to matrix-free GMRES
/// (one integration per matvec).
pub const krylov_threshold: usize = 50;

const Shoot = @This();

const Stage = enum { base, column, final, done };

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

/// `solve`'s setup: buffers, the GMRES workspace past
/// `krylov_threshold`, the baseline, and for an autonomous circuit the
/// oscillator start. Errors as `solve`'s; on error nothing is held.
pub fn init(s: *Shoot, ckt: *root.Circuit, x_dc: []const f64, probes: []const u32, wave: []f64, options: Options, allocator: std.mem.Allocator) !void {
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

/// Frees what `init` allocated.
pub fn deinit(s: *Shoot) void {
    if (s.mono) |*m| m.deinit(s.allocator);
    if (s.krylov) |*k| k.deinit(s.allocator);
    s.allocator.free(s.arena);
    s.* = undefined;
}

/// The next period the solve needs, or null once `result` is set.
/// Fails only when a checkpoint cancels.
pub fn next(s: *Shoot) error{QueryCancelled}!?Request {
    const n = s.n;
    sw: switch (s.stage) {
        .base => {
            if (s.iter >= s.options.max_shooting_iter) {
                s.stage = .final;
                continue :sw .final;
            }
            if (s.iter != 0) try s.ckt.checkpoint(.{ .phase = .periodic, .completed = s.iter });
            // x0 -> x_end, recording the trajectory the monodromy
            // recurrence linearizes around.
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
pub fn done(s: *Shoot, ok: bool) !void {
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

/// Integrates every period the solve asks for on the host, to `result`.
pub fn runAlone(s: *Shoot) !SolveResult {
    while (try s.next()) |r| try s.done(try integrateOnePeriod(s.ckt, s.ws, &s.sc, r.x, r.probes, r.wave, r.opts));
    return s.result;
}

// ---- private ----
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

    num.copySimd(ctx.x_end_pert, ctx.x_pert);
    const ok = integrateOnePeriod(ctx.ckt, ctx.ws, ctx.sc, ctx.x_end_pert, &.{}, &.{}, opts) catch |err| blk: {
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
