//! The lane backend's transient-step kernels, exported by `lane_device.zig`:
//! per-lane step state, scalars and history rings in device memory, the
//! predictor, LTE bound, accepted-step update, step decisions and probe
//! ring out. One thread per lane; bodies in `kernels/lane_step.zig`.
//! Cites: dev/research/plans/phase4-gpu-lanes.md G5.
//!
//! Contents:
//! - `pub fn predict(n_lanes: u64, n: u64, sc: P(LaneScalars), cur: P(f64), prev: P(f64), x: P(f64), x_old: P(f64)) callconv(gompute.kernel_callconv) void`
//! - `pub fn lte(n_lanes: u64, sc: P(LaneScalars), ring_qt: P(f64), perm_qt: P(u8), ip_qt: P(f64), n_hist: u64, n_qt: u64, lte_in: P(lk.LteIn), out: P([2]f64)) callconv(gompute.kernel_callconv) void`
//! - `pub fn accept(n_lanes: u64, n: u64, n_qt: u64, n_hist: u64, charge: u64, sc: P(LaneScalars), mask: P(u8), x: P(f64), cur: P(f64), prev: P(f64), q: P(f64), ring: P(f64), ip: P(f64), ring_qt: P(f64), ip_qt: P(f64)) callconv(gompute.kernel_callconv) void`
//! - `pub fn rebase(n_lanes: u64, n: u64, n_qt: u64, n_hist: u64, sc: P(LaneScalars), mask: P(u8), q: P(f64), ring: P(f64), ip: P(f64), ring_qt: P(f64), ip_qt: P(f64)) callconv(gompute.kernel_callconv) void`
//! - `pub fn gather(n_lanes: u64, sc: P(LaneScalars), x: P(f64), probes: P(u32), n_probes: u64, out: P(f64)) callconv(gompute.kernel_callconv) void`
//! - `pub fn stepBegin(n_lanes: u64, steps: P(lk.LaneStep), sc: P(LaneScalars), sims: P(SimState), lte_in: P(lk.LteIn), round: P(u8), ops: P(u8), acc: P(u8)) callconv(gompute.kernel_callconv) void`
//! - `pub fn settle(n_lanes: u64, steps: P(lk.LaneStep), active: P(u8), decide: P(u8), sc: P(LaneScalars)) callconv(gompute.kernel_callconv) void`
//! - `pub fn setLimw(n_lanes: u64, mask: P(u8), limw: P(u32), value: u64) callconv(gompute.kernel_callconv) void`
//! - `pub fn newtonInit(n_lanes: u64, max_iter: u64, round: P(u8), gate: P(Gate), sims: P(SimState), active: P(u8), fail: P(u8)) callconv(gompute.kernel_callconv) void`
//! - `pub fn setOps(n_lanes: u64, mask: P(u8), ops: P(u8), code: u64) callconv(gompute.kernel_callconv) void`
//! - `pub fn slotZero(n_lanes: u64, n_hist: u64, rebase_slot: u64, perm_qt: P(u8), sc: P(LaneScalars), slot0: P(u8)) callconv(gompute.kernel_callconv) void`
//! - `pub fn stepEndA(n_lanes: u64, n_hist: u64, decide: P(u8), steps: P(lk.LaneStep), gate: P(Gate), query: P(u32), lte_out: P([2]f64), ops: P(u8), acc: P(u8), sc: P(LaneScalars), perm_q: P(u8), perm_qt: P(u8), brk_t: P([2]f64), best: P(f64)) callconv(gompute.kernel_callconv) void`
//! - `pub fn stepEndB(n_lanes: u64, steps: P(lk.LaneStep), best: P(f64), report: P(lk.StepReport)) callconv(gompute.kernel_callconv) void`
//! - `pub fn ringPut(n_lanes: u64, n_probes: u64, slot: P(u32), report: P(lk.StepReport), ring: P(lk.StepReport), probe_out: P(f64), rows: P(f64)) callconv(gompute.kernel_callconv) void`
//! - `pub fn ringNext(slot: P(u32)) callconv(gompute.kernel_callconv) void`
//! - `pub fn stepStop(n_lanes: u64, stop: P(u8), steps: P(lk.LaneStep)) callconv(gompute.kernel_callconv) void`

const gompute = @import("gompute");
const lk = @import("kernels/lane.zig");
const lane_device = @import("lane_device.zig");
const P = gompute.GlobalPtr;
const SimState = lk.SimState;
const LaneScalars = lk.LaneScalars;
const Gate = lk.Gate;
const block = lane_device.block;
const lane = lane_device.lane;

/// trial = cur + xfact (cur - prev), per (row, lane), for the lanes in the
/// round (`sc[l].flags & 1`); x_old = cur for the predictor's limit pass.
pub fn predict(n_lanes: u64, n: u64, sc: P(LaneScalars), cur: P(f64), prev: P(f64), x: P(f64), x_old: P(f64)) callconv(gompute.kernel_callconv) void {
    const tid = gompute.globalIdX(block);
    if (tid >= n * n_lanes) return;
    const l: usize = @intCast(tid % n_lanes);
    if (sc[l].flags & 1 == 0) return;
    const i: usize = @intCast(tid);
    const c = cur[i];
    x[i] = c + sc[l].xfact * (c - prev[i]);
    x_old[i] = c;
}

/// The step's LTE bounds per round lane, from its tape ring (lane-major:
/// entry j of lane l's physical slot k at `(k * n_qt + j) * N + l`):
/// `out[l] = { del at the step's method, del of the order-2 promotion
/// probe (order-1 lanes) }`.
pub fn lte(n_lanes: u64, sc: P(LaneScalars), ring_qt: P(f64), perm_qt: P(u8), ip_qt: P(f64), n_hist: u64, n_qt: u64, lte_in: P(lk.LteIn), out: P([2]f64)) callconv(gompute.kernel_callconv) void {
    const l = lane();
    if (l >= n_lanes) return;
    const s = sc[l];
    if (s.flags & 1 == 0) return;
    const nh: usize = @intCast(n_hist);
    const m: usize = @intCast(n_qt);
    const L: usize = l;
    const N: usize = @intCast(n_lanes);
    var q: [4]P(f64) = undefined;
    inline for (0..4) |k| q[k] = ring_qt + @as(usize, perm_qt[L * nh + k]) * m * N + L;
    const ip = ip_qt + L;
    const method: u8 = @intCast(s.method);
    const c = lte_in[l];
    const del = lk.stepBound(method, method, q[0], q[1], q[2], q[3], ip, m, N, c);
    const del2 = if (method == 0) lk.stepBound(@intCast((s.flags >> 4) & 3), method, q[0], q[1], q[2], q[3], ip, m, N, c) else del;
    out[l] = .{ del, del2 };
}

/// An accepted step per (row or tape entry, lane), lanes innermost, for
/// the lanes with `accept[l]`: rows `prev <- cur`, `cur <- x`, the q plane
/// into the row ring's logical slot 0, and `advanceCurrent` (`ip = ag0 (q0
/// - q1) - history`) over rows and over the tape. The slots are the
/// pre-rotation ones in `sc[l].sq`/`sqt`; the host has rotated the
/// permutations for the next step already.
pub fn accept(n_lanes: u64, n: u64, n_qt: u64, n_hist: u64, charge: u64, sc: P(LaneScalars), mask: P(u8), x: P(f64), cur: P(f64), prev: P(f64), q: P(f64), ring: P(f64), ip: P(f64), ring_qt: P(f64), ip_qt: P(f64)) callconv(gompute.kernel_callconv) void {
    const tid = gompute.globalIdX(block);
    _ = n_hist; // the lane-major tape ring needs no per-lane base
    if (tid >= (n + n_qt) * n_lanes) return;
    const l: usize = @intCast(tid % n_lanes);
    if (mask[l] == 0) return;
    const s = sc[l];
    const N: usize = @intCast(n_lanes);
    const e: usize = @intCast(tid / n_lanes);
    const nn: usize = @intCast(n);
    if (e < nn) {
        const i = e * N + l;
        prev[i] = cur[i];
        cur[i] = x[i];
        if (charge == 0) return;
        const q0 = q[i];
        ring[(@as(usize, s.sq[0]) * nn + e) * N + l] = q0;
        const q1 = ring[(@as(usize, s.sq[1]) * nn + e) * N + l];
        const q2 = ring[(@as(usize, s.sq[2]) * nn + e) * N + l];
        ip[i] = advance(s, q0, q1, q2, ip[i]);
    } else {
        const j = e - nn;
        const m: usize = @intCast(n_qt);
        const q0 = ring_qt[(@as(usize, s.sqt[0]) * m + j) * N + l];
        const q1 = ring_qt[(@as(usize, s.sqt[1]) * m + j) * N + l];
        const q2 = ring_qt[(@as(usize, s.sqt[2]) * m + j) * N + l];
        ip_qt[j * N + l] = advance(s, q0, q1, q2, ip_qt[j * N + l]);
    }
}

/// The stateful-charge re-read of an accepted step per (row or tape entry,
/// lane), for the lanes with `mask[l]`, after the eval at cur and the tape
/// kernel into the tape's logical slot 0: `rebaseCurrent` (`ip += ag0 (q -
/// q1)`) and q into the row ring's logical slot 1, then the same over the
/// tape. Post-rotation slots: rows' logical 1 is `sq[0]`; the tape's
/// logical 0 is `sqt[3]` and logical 1 `sqt[0]`.
pub fn rebase(n_lanes: u64, n: u64, n_qt: u64, n_hist: u64, sc: P(LaneScalars), mask: P(u8), q: P(f64), ring: P(f64), ip: P(f64), ring_qt: P(f64), ip_qt: P(f64)) callconv(gompute.kernel_callconv) void {
    const tid = gompute.globalIdX(block);
    _ = n_hist; // the lane-major tape ring needs no per-lane base
    if (tid >= (n + n_qt) * n_lanes) return;
    const l: usize = @intCast(tid % n_lanes);
    if (mask[l] == 0) return;
    const s = sc[l];
    const N: usize = @intCast(n_lanes);
    const e: usize = @intCast(tid / n_lanes);
    const nn: usize = @intCast(n);
    if (e < nn) {
        const i = e * N + l;
        const r1 = (@as(usize, s.sq[0]) * nn + e) * N + l;
        ip[i] += s.ag0 * (q[i] - ring[r1]);
        ring[r1] = q[i];
    } else {
        const j = e - nn;
        const m: usize = @intCast(n_qt);
        ip_qt[j * N + l] += s.ag0 * (ring_qt[(@as(usize, s.sqt[3]) * m + j) * N + l] - ring_qt[(@as(usize, s.sqt[0]) * m + j) * N + l]);
    }
}

/// `out[l * n_probes + k] = x[probes[k]]` of lane l, for the lanes in the
/// round.
pub fn gather(n_lanes: u64, sc: P(LaneScalars), x: P(f64), probes: P(u32), n_probes: u64, out: P(f64)) callconv(gompute.kernel_callconv) void {
    const tid = gompute.globalIdX(block);
    if (tid >= n_probes * n_lanes) return;
    const l: usize = @intCast(tid % n_lanes);
    if (sc[l].flags & 1 == 0) return;
    const k: usize = @intCast(tid / n_lanes);
    out[l * @as(usize, @intCast(n_probes)) + k] = x[@as(usize, probes[k]) * @as(usize, @intCast(n_lanes)) + l];
}

/// `lk.stepBegin` per lane: the round's scalars, `SimState`, LTE inputs and
/// `round` mask, with `ops` and `accept` cleared for every lane.
pub fn stepBegin(n_lanes: u64, steps: P(lk.LaneStep), sc: P(LaneScalars), sims: P(SimState), lte_in: P(lk.LteIn), round: P(u8), ops: P(u8), acc: P(u8)) callconv(gompute.kernel_callconv) void {
    const l = lane();
    if (l >= n_lanes) return;
    ops[l] = 0;
    acc[l] = 0;
    // A lane whose Newton carries over from the last round keeps its
    // attempt; bit 0 of its flags clears so `predict` skips it, and
    // `settle` sets it back when its Newton ends.
    if (steps[l].event == lk.step_attempt) {
        round[l] = 0;
        sc[l].flags &= ~@as(u32, 1);
        return;
    }
    const on = lk.stepBegin(&steps[l], &sc[l], &sims[l], &lte_in[l]);
    round[l] = @intFromBool(on);
    // `predict` and the LTE kernel read bit 0 for round membership.
    if (!on) sc[l].flags = 0;
}

/// After a round's ticks: `decide[l]` for each lane whose Newton ended
/// this round (attempting, no longer active), which takes back flag bit 0
/// for the probe; lanes still in Newton carry over.
pub fn settle(n_lanes: u64, steps: P(lk.LaneStep), active: P(u8), decide: P(u8), sc: P(LaneScalars)) callconv(gompute.kernel_callconv) void {
    const l = lane();
    if (l >= n_lanes) return;
    const done = steps[l].event == lk.step_attempt and active[l] == 0;
    decide[l] = @intFromBool(done);
    if (done) sc[l].flags |= 1;
}

/// A batch's per-lane lim-active word: `value` for the lanes of `mask`.
pub fn setLimw(n_lanes: u64, mask: P(u8), limw: P(u32), value: u64) callconv(gompute.kernel_callconv) void {
    const l = lane();
    if (l >= n_lanes or mask[l] == 0) return;
    limw[l] = @intCast(value);
}

/// `GpuLanes.newton`'s setup for the lanes that began this round
/// (`round`): each one's gate, `beginSolve`'s iteration, the active mask
/// and its fail flag. A lane carrying its Newton over keeps all four.
pub fn newtonInit(n_lanes: u64, max_iter: u64, round: P(u8), gate: P(Gate), sims: P(SimState), active: P(u8), fail: P(u8)) callconv(gompute.kernel_callconv) void {
    const l = lane();
    if (l >= n_lanes or round[l] == 0) return;
    gate[l] = .{ .limit = 0, .scaled = 0, .iterations = @intCast(max_iter), .bits = 1 };
    sims[l].iteration = 1;
    active[l] = 1;
    fail[l] = 0;
}

/// `ops[l] = code` for the lanes of `mask`, 0 for the rest.
pub fn setOps(n_lanes: u64, mask: P(u8), ops: P(u8), code: u64) callconv(gompute.kernel_callconv) void {
    const l = lane();
    if (l >= n_lanes) return;
    ops[l] = if (mask[l] != 0) @intCast(code) else 0;
}

/// The tape slot a charge pass writes, per lane: the tape's logical slot 0
/// (`perm_qt[l * nh]`, `probe`) or the rebase's `sc[l].sqt[3]`.
pub fn slotZero(n_lanes: u64, n_hist: u64, rebase_slot: u64, perm_qt: P(u8), sc: P(LaneScalars), slot0: P(u8)) callconv(gompute.kernel_callconv) void {
    const l = lane();
    if (l >= n_lanes) return;
    slot0[l] = if (rebase_slot != 0) sc[l].sqt[3] else perm_qt[@as(usize, l) * @as(usize, @intCast(n_hist))];
}

/// `lk.stepEndA` per lane, then the breakpoint request of each accepted
/// one: `brk_t[l] = { t_new, t_new + min_break }` and `best[l] = inf`.
pub fn stepEndA(n_lanes: u64, n_hist: u64, decide: P(u8), steps: P(lk.LaneStep), gate: P(Gate), query: P(u32), lte_out: P([2]f64), ops: P(u8), acc: P(u8), sc: P(LaneScalars), perm_q: P(u8), perm_qt: P(u8), brk_t: P([2]f64), best: P(f64)) callconv(gompute.kernel_callconv) void {
    const l = lane();
    if (l >= n_lanes or decide[l] == 0) return;
    const nh: usize = @intCast(n_hist);
    const L: usize = l;
    const s = &steps[l];
    lk.stepEndA(s, gate[l], query[l], lte_out[l], &ops[l], &acc[l], &sc[l], perm_q + L * nh, perm_qt + L * nh, nh);
    if (s.event == lk.step_accept) {
        brk_t[l] = .{ s.t_new, s.t_new + s.min_break };
        best[l] = @import("std").math.inf(f64);
    }
}

/// `lk.stepEndB` per lane from its breakpoint query, then its report.
pub fn stepEndB(n_lanes: u64, steps: P(lk.LaneStep), best: P(f64), report: P(lk.StepReport)) callconv(gompute.kernel_callconv) void {
    const l = lane();
    if (l >= n_lanes) return;
    const s = &steps[l];
    lk.stepEndB(s, best[l]);
    report[l] = .{ .t_new = s.t_new, .t = s.t, .dt = s.dt, .attempted_dt = s.attempted_dt, .attempts = s.attempts, .steps = s.steps, .event = s.event, .status = s.status };
}

/// The round's report and probe rows of every lane into ring slot
/// `slot[0]` of a multi-round launch: `ring[slot * N + l]` and rows at
/// `(slot * N + l) * n_probes`.
pub fn ringPut(n_lanes: u64, n_probes: u64, slot: P(u32), report: P(lk.StepReport), ring: P(lk.StepReport), probe_out: P(f64), rows: P(f64)) callconv(gompute.kernel_callconv) void {
    const l = lane();
    if (l >= n_lanes) return;
    const N: usize = @intCast(n_lanes);
    const np: usize = @intCast(n_probes);
    const at = @as(usize, slot[0]) * N + l;
    ring[at] = report[l];
    for (0..np) |k| rows[at * np + k] = probe_out[@as(usize, l) * np + k];
}

/// Advances the ring slot after every lane's `ringPut`.
pub fn ringNext(slot: P(u32)) callconv(gompute.kernel_callconv) void {
    if (gompute.globalIdX(block) == 0) slot[0] += 1;
}

/// The host's stop requests: a lane with `stop[l]` leaves stepping.
pub fn stepStop(n_lanes: u64, stop: P(u8), steps: P(lk.LaneStep)) callconv(gompute.kernel_callconv) void {
    const l = lane();
    if (l >= n_lanes or stop[l] == 0) return;
    if (steps[l].status == lk.step_run) steps[l].status = lk.step_stopped;
    stop[l] = 0;
}

// ---- private ----

/// `integrator.companionAt(m, false, ...)` at one element.
inline fn advance(s: LaneScalars, q0: f64, q1: f64, q2: f64, i_prev: f64) f64 {
    const d = s.ag0 * (q0 - q1);
    return switch (s.method) {
        1 => d - s.ag1 * i_prev,
        2 => d - s.ag2 * (q1 - q2),
        else => d,
    };
}
