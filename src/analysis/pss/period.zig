//! One shooting period: a start state x0 and a fixed step count in, x(T)
//! out by fixed-step trapezoid, with each step's Newton on the trapezoid
//! companion, the device commits, and (when asked) one wave row per step.
//! The lane drivers in pss.zig run the same start, hook and accept per lane
//! around a lane Newton.
//! Cites: Verilog-A LRM §4.6.1 (sources follow their waveform only in the
//! transient phase).
//!
//! Contents:
//! - `pub const Companion = struct`
//! - `pub const PeriodHook = integrator.Hook(.trapezoidal)`
//! - `pub fn integrateOnePeriod(ckt: *root.Circuit, ws: *converger.Workspace, sc: *Companion, x: []f64, probes: []const u32, wave: []f64, options: Options) error{ QueryCancelled, OutOfMemory }!bool`
//! - `pub fn periodStart(ckt: *root.Circuit, sc: *Companion, x: []const f64, probes: []const u32, wave: []f64, options: Options) void`
//! - `pub fn stepTime(options: Options, k: usize) f64`
//! - `pub fn periodHook(sc: *Companion, alpha: f64, has_charge: bool) PeriodHook`
//! - `pub fn periodAccept(ckt: *root.Circuit, hook: *const PeriodHook, x: []const f64, probes: []const u32, wave: []f64, k: usize, t: f64, converged: bool) bool`

const std = @import("std");
const num = @import("core").numerics;
const converger = @import("solver").converger;
const root = @import("../types.zig");
const integrator = @import("../tran/integrator.zig");
const Options = @import("core").query.Pss;

/// Trapezoidal companion state, reused across every period integration.
pub const Companion = struct {
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
pub const PeriodHook = integrator.Hook(.trapezoidal);

/// Integrates x in place over [0, T] in n_samples trapezoid steps. A
/// non-empty `wave` receives n_samples+1 point-major rows
/// [t, v(probes)...] of stride 1 + probes.len. Returns false on the first
/// step whose Newton fails.
pub fn integrateOnePeriod(
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
pub fn periodStart(ckt: *root.Circuit, sc: *Companion, x: []const f64, probes: []const u32, wave: []f64, options: Options) void {
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
pub fn stepTime(options: Options, k: usize) f64 {
    return options.period * @as(f64, @floatFromInt(k + 1)) / @as(f64, @floatFromInt(options.n_samples));
}

/// The companion hook of one step at `alpha` = 2/dt.
pub fn periodHook(sc: *Companion, alpha: f64, has_charge: bool) PeriodHook {
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
pub fn periodAccept(ckt: *root.Circuit, hook: *const PeriodHook, x: []const f64, probes: []const u32, wave: []f64, k: usize, t: f64, converged: bool) bool {
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
