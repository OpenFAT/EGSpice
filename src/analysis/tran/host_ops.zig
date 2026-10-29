//! The host numerics of one step attempt (`HostOps`, the oracle of every
//! lane backend): a `Sim` and its `Attempt` in, the predictor, the charge
//! re-read, the LTE bounds, the history rotation and the commits out, each
//! at the point `Sim.begin` and `Sim.end` call it. Also the LTE state
//! snapshot and the rejected-step LTE trace.
//! Cites: ngspice dctran.c:794 and DEVpred (MODEINITPRED), cktterr.c and
//! niinteg.c (CKTstates[0][qcap+1]).
//!
//! Contents:
//! - `pub const HostOps = struct`
//! - `HostOps`: `attempt`, `revert`, `rejectAt`, `flipped`, `boundStep`, `charges`, `bound`, `boundGear`, `advance`, `commit`, `rebase`, `accepted`
//! - `pub fn lteSnap(c: *root.Circuit, dst: []f64, zero: []const [2]u32, rows: []const u32) void`
//! - `pub fn traceLte(s: *Sim, a: *const Attempt, nr: converger.Result, lte: integrator.LteIn, bound: f64) void`

const std = @import("std");
const assert = std.debug.assert;
const num = @import("core").numerics;
const converger = @import("solver").converger;
const root = @import("../types.zig");
const integrator = @import("integrator.zig");
const Sim = @import("Sim.zig");
const Attempt = Sim.Attempt;
const Method = @import("types.zig").Method;
const predict = Sim.predict;

/// What `Sim.begin` and `Sim.end` ask of the numerics, so one copy of the
/// step control drives the host (`HostOps`, the oracle) and the device
/// (`gpu_lanes.Ops`). Each method runs exactly the statements it names, at
/// the point it is called. A backend may answer from numbers it computed
/// earlier for the whole round, as long as they equal what the statements
/// would compute now.
pub const HostOps = struct {
    /// begin: gear history, the hook, predict, predictFirstIterate,
    /// evalFollowsFirst, applyLimits, reject_at = null.
    pub inline fn attempt(_: HostOps, s: *Sim, a: *Attempt) void {
        const ckt = s.ckt;
        const q_hist = s.qHist();
        if (a.gear_hi) integrator.gearHistory(s.g_row, q_hist[0 .. s.order + 1], a.ag);
        a.hook = .{
            .method = a.method,
            .c = a.c,
            .q_prev = q_hist[1],
            .i_prev = s.i_prev,
            .q_prev2 = if (a.gear_hi) s.g_row else q_hist[2],
            .q_cur = {},
            .a_vals = s.a_vals,
            .has_charge = s.has_charge,
        };
        // MODEINITPRED (dctran.c:794, DEVpred in dioload.c/mos1load.c): the
        // first iterate extrapolates the last two accepted points by
        // dt/dt_prev and is limited against the last accepted one. HFET and
        // MESA divide by the step two back instead (CKTdeltaOld[2]).
        const xfact = s.dt / s.dt_old[0];
        predict(s.trial, s.cur, s.prev, xfact);
        ckt.predictFirstIterate(s.trial, s.cur, s.prev, s.dt / s.dt_old[1]);
        ckt.evalFollowsFirst(s.trial, a.t_dev);
        _ = ckt.applyLimits(s.trial, s.cur);
        ckt.reject_at = null;
    }
    /// Reverts every device latch to the last accepted state.
    pub inline fn revert(_: HostOps, s: *Sim) void {
        _ = s.ckt.stateCtl(.revert);
    }
    /// Returns the time a device asked the step to end on, if any.
    pub inline fn rejectAt(_: HostOps, s: *Sim) ?f64 {
        return s.ckt.reject_at;
    }
    /// The `evalFollows` hint, then (only when `ask`) the flip query.
    pub inline fn flipped(_: HostOps, s: *Sim, a: *const Attempt, ask: bool) bool {
        if (s.has_charge and !s.ckt.q_fresh) s.ckt.evalFollows(s.trial, a.t_dev, true);
        return ask and s.ckt.stateCtl(.query);
    }
    /// Returns the devices' `$bound_step` for the next step, if any.
    pub inline fn boundStep(_: HostOps, s: *Sim) ?f64 {
        return s.ckt.boundStep();
    }
    /// The charge re-read at the solution into ring slot 0, rows and tape.
    pub inline fn charges(_: HostOps, s: *Sim, a: *const Attempt) void {
        const ckt = s.ckt;
        const qt_hist = s.qtHist();
        if (!ckt.q_fresh) ckt.evalQ(s.trial, a.t_dev);
        num.copySimd(s.qHist()[0], ckt.q_vec[0..s.n]);
        if (s.n_qt > 0) lteSnap(ckt, qt_hist[0], s.lte_zero, s.lte_rows);
        if (a.gear_hi and s.n_qt > 0) integrator.gearHistory(s.qt_g, qt_hist[0 .. s.order + 1], a.ag);
    }
    /// `stepBound` (order <= 2) of `method` after a step at `a.method`.
    pub inline fn bound(_: HostOps, s: *Sim, a: *const Attempt, method: Method, lte: integrator.LteIn) f64 {
        const h = s.lteHist();
        return integrator.stepBound(W, method, a.method, .{ h[0], h[1], h[2], h[3] }, s.lteIp(), a.c, lte);
    }
    /// `stepBoundGear` of `method` after a step at `a.method`.
    pub inline fn boundGear(_: HostOps, s: *Sim, a: *const Attempt, method: Method, lte: integrator.LteIn) f64 {
        return integrator.stepBoundGear(method, a.method, s.lteHist(), s.lteIp(), s.lteG(), a.c, lte, &a.dts);
    }
    /// advanceCurrent on both index spaces, then the ring rotation.
    pub inline fn advance(_: HostOps, s: *Sim, a: *const Attempt) void {
        const q_hist = s.qHist();
        const qt_hist = s.qtHist();
        integrator.advanceCurrent(a.method, s.i_prev, q_hist[0], q_hist[1], if (a.gear_hi) s.g_row else q_hist[2], a.c);
        if (s.n_qt > 0)
            integrator.advanceCurrent(a.method, s.qt_i_prev, qt_hist[0], qt_hist[1], if (a.gear_hi) s.qt_g else qt_hist[2], a.c);
        std.mem.rotate([]f64, q_hist, s.n_hist - 1);
        if (s.n_qt > 0) std.mem.rotate([]f64, qt_hist, s.n_hist - 1);
    }
    /// commitStates and the commit, at the rotated `cur`.
    pub inline fn commit(_: HostOps, s: *Sim) void {
        _ = s.ckt.commitStates(s.cur);
        _ = s.ckt.stateCtl(.commit);
    }
    /// The stateful-charge re-read at `cur` (`has_state_q`).
    pub inline fn rebase(_: HostOps, s: *Sim, a: *const Attempt) void {
        const ckt = s.ckt;
        const q_hist = s.qHist();
        const qt_hist = s.qtHist();
        if (!(s.has_charge and ckt.has_state_q)) return;
        ckt.evalQ(s.cur, s.t);
        integrator.rebaseCurrent(s.i_prev, ckt.q_vec[0..s.n], q_hist[1], a.c.ag0);
        num.copySimd(q_hist[1], ckt.q_vec[0..s.n]);
        if (s.n_qt > 0) {
            lteSnap(ckt, qt_hist[0], s.lte_zero, s.lte_rows);
            integrator.rebaseCurrent(s.qt_i_prev, qt_hist[0], qt_hist[1], a.c.ag0);
            // Swap, not copy: slot 0 is the ring's scratch slot, and the
            // next accepted attempt overwrites it before any read.
            std.mem.swap([]f64, &qt_hist[0], &qt_hist[1]);
        }
    }
    /// The accepted solution `record` and `feed` read (probe rows only).
    pub inline fn accepted(_: HostOps, s: *Sim) []const f64 {
        return s.cur;
    }
};

/// Lanes of the LTE vector kernel.
const W = std.simd.suggestVectorLength(f64) orelse 8;

/// The LTE states at the circuit's current charges into `dst`: the device
/// charge tape, its `zero` spans cleared, then `q_vec` at each of `rows`.
/// `dst` is `n_qt + rows.len` long.
pub fn lteSnap(c: *root.Circuit, dst: []f64, zero: []const [2]u32, rows: []const u32) void {
    const n_tape = dst.len - rows.len;
    c.snapshotQTape(dst[0..n_tape]);
    for (zero) |span| @memset(dst[span[0]..span[1]], 0);
    for (rows, dst[n_tape..]) |r, *d| d.* = c.q_vec[r];
}

/// Prints the operands of the state that bound a rejected step
/// (`ESPICE_TRAN_LTE_TRACE`), replayed from the captured history. Asserts
/// that the replay reproduces `bound` bit for bit.
pub fn traceLte(s: *Sim, a: *const Attempt, nr: converger.Result, lte: integrator.LteIn, bound: f64) void {
    const hit = integrator.diagnoseBound(a.method, a.method, s.lteHist(), s.lteIp(), s.lteG(), a.c, lte, &a.dts);
    const matches = @as(u64, @bitCast(hit.bound)) == @as(u64, @bitCast(bound)) or
        (std.math.isNan(hit.bound) and std.math.isNan(bound));
    std.debug.print(
        "tran-lte: matches={} accepted_t={e:.17} trial_t={e:.17} dt={e:.17} bound={e:.17} diagnostic_bound={e:.17} method={s} order={d} nr_iters_attempt={d} index={d}\n",
        .{ matches, s.t, s.t + s.dt, s.dt, bound, hit.bound, @tagName(a.method), integrator.order(a.method), nr.iterations, hit.index },
    );
    std.debug.assert(matches);
    if (!matches) return;
    if (s.n_qt == 0) {
        std.debug.print("tran-lte-state: row={d}\n", .{hit.index});
    } else if (hit.index >= s.n_qt) {
        std.debug.print("tran-lte-state: appended_row={d}\n", .{s.lte_rows[hit.index - s.n_qt]});
    } else {
        var off: usize = 0;
        for (s.ckt.batches, 0..) |b, bi| {
            const tape = b.hooks.q_tape orelse continue;
            const len = tape(b.ctx).len;
            if (hit.index < off + len) {
                if (b.hooks.charge_identity) |identity| {
                    const id = identity(b.ctx, hit.index - off);
                    std.debug.print("tran-lte-state: batch={d} type={s} original_instance={d} generated_site={d} rows={any}\n", .{ bi, b.type_name, id.instance, id.site, id.rows });
                } else std.debug.print("tran-lte-state: batch={d} type={s} local_tape_index={d} identity_unavailable=true\n", .{ bi, b.type_name, hit.index - off });
                break;
            }
            off += len;
        }
    }
    for (s.lteHist(), 0..) |h, k| std.debug.print("tran-lte-q: history={d} q={e:.17}\n", .{ k, h[hit.index] });
    for (a.dts, 0..) |dt, k| std.debug.print("tran-lte-dt: history={d} dt={e:.17}\n", .{ k, dt });
    std.debug.print(
        "tran-lte-operands: ip={e:.17} i_new={e:.17} current_tol={e:.17} charge_tol={e:.17} tol={e:.17} dd={e:.17} lte_coefficient={e:.17} unrooted_bound={e:.17} ag0={e:.17} ag1={e:.17} ag2={e:.17} trtol={e:.17}\n",
        .{ hit.i_prev, hit.i_new, hit.current_tol, hit.charge_tol, hit.tolerance, hit.difference, hit.coefficient, hit.unrooted_bound, a.c.ag0, a.c.ag1, a.c.ag2, lte.trtol },
    );
}
