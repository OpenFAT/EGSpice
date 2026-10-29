//! Per-lane bodies of the GPU lane transient step: one lane's step state,
//! its scalars and the converged Newton gate in, the next step's decision
//! (accept, reject, retry with a cut step, stop) and the LTE bound out.
//! Each body is the host transient's own arithmetic in its order
//! (`analysis/tran`'s `integrator.stepBound`, ngspice CKTterr), so a host
//! run is bitwise the oracle. std only: the kernel root compiles this file
//! for NVPTX and AMDGCN.
//! Cites: ngspice dctran.c and cktterr.c (step control, LTE);
//! dev/research/plans/phase4.
//!
//! Contents:
//! - `pub const LteIn = extern struct`
//! - `pub const dense_peel_max_n = 64`
//! - `pub const LaneScalars = extern struct`
//! - `pub const LaneStep = extern struct`
//! - `pub const StepReport = extern struct`
//! - `pub const step_run: u8 = 0`
//! - `pub const step_done: u8 = 1`
//! - `pub const step_underflow: u8 = 2`
//! - `pub const step_parked: u8 = 3`
//! - `pub const step_stopped: u8 = 4`
//! - `pub const step_idle: u8 = 0`
//! - `pub const step_attempt: u8 = 1`
//! - `pub const step_accept: u8 = 2`
//! - `pub fn stepBound(method: u8, cur_method: u8, q0: anytype, q1: anytype, q2: anytype, q3: anytype, ip: anytype, len: usize, stride: usize, c: LteIn) f64`
//! - `pub fn denseSolveNeg(n: usize, m: anytype, x: anytype) bool`
//! - `pub fn stepBegin(s: anytype, sc: anytype, sim: anytype, lin: anytype) bool`
//! - `pub fn stepEndA(s: anytype, gate: lk.Gate, query: u32, lte: [2]f64, ops: anytype, accept: anytype, sc: anytype, pq: anytype, pqt: anytype, nh: usize) void`
//! - `pub fn stepEndB(s: anytype, nb: f64) void`

const std = @import("std");
const lk = @import("lane.zig");

/// The LTE inputs of one step, per lane (uploaded with `LaneScalars`).
pub const LteIn = extern struct { dt: f64, dt1: f64, dt2: f64, reltol: f64, abstol: f64, chgtol: f64, trtol: f64, ag0: f64, ag1: f64, ag2: f64 };

/// Largest system a failed lane factors densely on the device
/// (`arp_lane_peel_dense`); past it the host peels.
pub const dense_peel_max_n = 64;

/// Per-lane step inputs, uploaded each round (`gpu_lanes.Ops.attempt`).
pub const LaneScalars = extern struct {
    t_dev: f64,
    dt: f64,
    xfact: f64,
    ag0: f64,
    ag1: f64,
    ag2: f64,
    /// 0 BE, 1 trap, 2 gear-2.
    method: u32,
    /// Bit 0: in this round. Bits 4-5: the order-2 promotion probe's
    /// method code, read by `lte` for a lane at order 1.
    flags: u32,
    /// The row ring's physical slots of logical 0, 1, 2 before this step's
    /// rotation, for `accept`.
    sq: [4]u8,
    /// The same for the LTE tape ring.
    sqt: [4]u8,
};

/// One lane's transient step machine for device step control: `tran.Sim`'s
/// scalars and stats, and the bodies `stepBegin`, `stepEndA` and
/// `stepEndB`, which restate `Sim.begin` and `Sim.end` statement for
/// statement for the shapes lanes admit (order <= 2, no VPI, tangents,
/// delay lines or uic). The host drains each accepted row and runs the
/// measures and early stop; `$bound_step` and `reject_at` park the lane.
/// Host step control over the same device numerics is the bitwise oracle.
pub const LaneStep = extern struct {
    // Fixed for the run (`Sim.init`, `Options`).
    t_stop: f64,
    dt_min: f64,
    xmu: f64,
    dt_max: f64,
    delmin: f64,
    min_break: f64,
    state_eps: f64,
    reltol: f64,
    abstol: f64,
    chgtol: f64,
    trtol: f64,
    // `Sim`'s state.
    t: f64,
    dt: f64,
    dt_old: [6]f64,
    attempted_dt: f64,
    /// inf for none.
    bp_next: f64,
    bp_save_dt: f64,
    /// Between `stepEndA` and `stepEndB` of an accepted step.
    t_new: f64,
    dt_next: f64,
    // `Stats`.
    attempts: u64,
    nr_iters: u64,
    rej_newton: u64,
    rej_state: u64,
    rej_lte: u64,
    order_drops: u64,
    bp_landings: u64,
    by_order: [6]u64,
    first_dt: f64,
    max_dt: f64,
    steps: u32,
    order_run: u32,
    max_steps: u32,
    max_iter: u32,
    /// `ckt.sim.iteration` as `Sim.begin`'s `setSimState` keeps it.
    iteration: u32,
    order: u8,
    run_order: u8,
    bp_clamped: u8,
    status: u8,
    /// Method code (0 BE, 1 trap, 2 gear-2) of `Options.method`.
    top: u8,
    charge: u8,
    state_q: u8,
    /// This round: `step_attempt`, `step_accept` or `step_reject`.
    event: u8,
    /// `1 + StateCtlOp` codes for revert and commit (`ops`).
    op_revert: u8,
    op_commit: u8,
    /// `AnalysisKind.tran`.
    kind_tran: u8,
    _pad: [5]u8 = @splat(0),
};

/// What the host reads of each lane after a device-controlled round
/// (`step_end_b`): enough to drain an accepted row, finish a lane and
/// report progress without downloading the whole `LaneStep`.
pub const StepReport = extern struct {
    t_new: f64,
    t: f64,
    dt: f64,
    attempted_dt: f64,
    attempts: u64,
    steps: u32,
    event: u8,
    status: u8,
    _pad: [2]u8 = @splat(0),
};

/// `LaneStep.status`: stepping, reached t_stop or `max_steps`, dt below
/// `dt_min`, handed back to the host (`reject_at`, `$bound_step`), stopped by
/// the host (early stop, a measure bound, an error).
pub const step_run: u8 = 0;

pub const step_done: u8 = 1;

pub const step_underflow: u8 = 2;

pub const step_parked: u8 = 3;

pub const step_stopped: u8 = 4;

/// `LaneStep.event`.
pub const step_idle: u8 = 0;

pub const step_attempt: u8 = 1;

pub const step_accept: u8 = 2;

const step_reject: u8 = 3;

/// `integrator.stepBound(w, method, cur_method, ...)` for one lane over its
/// history: `qk` is logical slot k and `ip` the dynamic current, `len`
/// entries each `stride` apart (1 contiguous, the lane count lane-major),
/// as slices or many-item pointers of any address space; methods as in
/// `companionRow`. NaN as `stepBound`. Order 1 or 2 only.
pub fn stepBound(method: u8, cur_method: u8, q0: anytype, q1: anytype, q2: anytype, q3: anytype, ip: anytype, len: usize, stride: usize, c: LteIn) f64 {
    const coeff: f64 = switch (method) {
        0 => 0.5,
        1 => 0.08333333333,
        else => 0.2222222222,
    };
    const sum01 = c.dt + c.dt1;
    const sum12 = c.dt1 + c.dt2;
    const sum012 = c.dt + (c.dt1 + c.dt2);
    var vmin = std.math.inf(f64);
    var poison: f64 = 0;
    for (0..len) |jj| {
        const j = jj * stride;
        const qc = q0[j];
        const qp = q1[j];
        const qp2 = q2[j];
        const qp3 = q3[j];
        const i_p = ip[j];
        const i_new = switch (cur_method) {
            1 => c.ag0 * (qc - qp) - c.ag1 * i_p,
            2 => c.ag0 * (qc - qp) - c.ag2 * (qp - qp2),
            else => c.ag0 * (qc - qp),
        };
        const volttol = c.abstol + c.reltol * @max(@abs(i_new), @abs(i_p));
        const chargetol = c.reltol * @max(@max(@abs(qc), @abs(qp)), c.chgtol) / c.dt;
        const tol = @max(volttol, chargetol);
        poison += (qc - qc) + (qp - qp) + (qp2 - qp2) + (i_p - i_p);
        const f12 = (qp - qp2) / c.dt1;
        var dd = ((qc - qp) / c.dt - f12) / sum01;
        if (method != 0) {
            poison += qp3 - qp3;
            const f123 = (f12 - (qp2 - qp3) / c.dt2) / sum12;
            dd = (dd - f123) / sum012;
        }
        const del = c.trtol * tol / @max(c.abstol, coeff * @abs(dd));
        poison += if (del == del) 0 else del;
        vmin = @min(vmin, del);
    }
    if (std.math.isNan(poison)) return std.math.nan(f64);
    return if (method == 0) vmin else @sqrt(vmin);
}

/// x = -A^-1 b by LU with partial pivoting, for a lane whose replayed
/// pivots failed (`arp_lane_peel_dense`). `m` is A row-major, n x n, and
/// `x` holds b on entry; both are clobbered. False when a pivot is below
/// eps^2 or not finite. The elimination is `dense_lu`'s unblocked loops op
/// for op, so pivots and singularity agree with it exactly; the back
/// substitution sums in index order where `dense_lu` dots in SIMD order,
/// so x agrees within rounding.
pub fn denseSolveNeg(n: usize, m: anytype, x: anytype) bool {
    const tiny = std.math.floatEps(f64) * std.math.floatEps(f64);
    for (0..n) |i| x[i] = -x[i];
    for (0..n) |k| {
        var max_row = k;
        var max_val = @abs(m[k * n + k]);
        for (k + 1..n) |i| {
            const v = @abs(m[i * n + k]);
            if (v > max_val) {
                max_val = v;
                max_row = i;
            }
        }
        const p = m[max_row * n + k];
        if (!(@abs(p) >= tiny and @abs(p) <= std.math.floatMax(f64))) return false;
        if (max_row != k) {
            for (0..n) |j| {
                const t = m[k * n + j];
                m[k * n + j] = m[max_row * n + j];
                m[max_row * n + j] = t;
            }
            const t = x[k];
            x[k] = x[max_row];
            x[max_row] = t;
        }
        const inv = @as(f64, 1.0) / m[k * n + k];
        for (k + 1..n) |i| {
            const f = m[i * n + k] * inv;
            for (k + 1..n) |j| m[i * n + j] -= f * m[k * n + j];
            x[i] -= f * x[k];
        }
    }
    var ki = n;
    while (ki > 0) {
        ki -= 1;
        var s: f64 = 0;
        for (ki + 1..n) |j| s += m[ki * n + j] * x[j];
        x[ki] = (x[ki] - s) / m[ki * n + ki];
    }
    return true;
}

/// `Sim.begin` and `gpu_lanes.Ops.attempt`: false when the lane does not
/// attempt this round (not stepping, or finished: then `status` says why).
/// Fills the round's scalars, `SimState` and LTE inputs.
pub fn stepBegin(s: anytype, sc: anytype, sim: anytype, lin: anytype) bool {
    s.event = step_idle;
    if (s.status != step_run) return false;
    if (!(s.t < s.t_stop and !atStop(s.t, s.t_stop) and s.steps < s.max_steps)) {
        s.status = step_done;
        return false;
    }
    s.attempted_dt = s.dt;
    const t = s.t;
    const dt = s.dt;
    const t_dev = t + dt;
    sim.* = .{
        .t = t_dev,
        .dt = dt,
        .kind = s.kind_tran,
        .initial_step = s.steps == 0,
        .final_step = t + dt >= s.t_stop,
        .analog_initial = true,
        .iteration = s.iteration,
    };
    const m = methodAt(s.top, s.order);
    const c = coeffs(m, dt, s.dt_old[0], s.xmu);
    sc.* = .{
        .t_dev = t_dev,
        .dt = dt,
        .xfact = dt / s.dt_old[0],
        .ag0 = c[0],
        .ag1 = c[1],
        .ag2 = c[2],
        .method = m,
        .flags = 1 | (@as(u32, methodAt(s.top, 2)) << 4),
        .sq = @splat(0),
        .sqt = @splat(0),
    };
    lin.* = .{ .dt = dt, .dt1 = s.dt_old[0], .dt2 = s.dt_old[1], .reltol = s.reltol, .abstol = s.abstol, .chgtol = s.chgtol, .trtol = s.trtol, .ag0 = c[0], .ag1 = c[1], .ag2 = c[2] };
    s.event = step_attempt;
    return true;
}

/// `Sim.end` up to the commit, for a lane that attempted this round, from
/// its Newton gate, flip query and LTE pair: a rejection sets `ops` to
/// revert and the retry dt; an accepted step rotates the ring permutations
/// (`Ops.advance`, `Ops.rebase`), sets `ops` to commit and `accept`, and
/// leaves `t_new` and `dt_next` for `stepEndB`. A gate asking for
/// `reject_at` or `$bound_step` parks the lane. `pq`/`pqt` are the lane's
/// `nh` ring slots.
pub fn stepEndA(s: anytype, gate: lk.Gate, query: u32, lte: [2]f64, ops: anytype, accept: anytype, sc: anytype, pq: anytype, pqt: anytype, nh: usize) void {
    if (s.event != step_attempt) return;
    // The host stopped the lane while its Newton ran.
    if (s.status != step_run) {
        s.event = step_idle;
        return;
    }
    if (gate.bits & 32 != 0) {
        s.status = step_parked;
        s.event = step_idle;
        return;
    }
    const singular = gate.bits & 8 != 0;
    const converged = !singular and gate.bits & 4 != 0;
    const iterations: u64 = if (singular) 0 else @min(gate.iterations, s.max_iter);
    const t = s.t;
    const dt = s.dt;
    const method = methodAt(s.top, s.order);
    s.attempts += 1;
    s.nr_iters += iterations;
    if (!converged) {
        s.rej_newton += 1;
        ops.* = s.op_revert;
        if (s.order > 1) s.order_drops += 1;
        s.order = 1;
        return retry(s, dt, dt / 8.0);
    }
    // `reject_at` never answers on lanes (`Ops.rejectAt`).
    if (dt > s.state_eps and query & 1 != 0) {
        s.rej_state += 1;
        ops.* = s.op_revert;
        s.order = 1;
        s.dt = @max(0.25 * dt, s.state_eps);
        s.event = step_reject;
        return;
    }
    var dt_next = if (s.steps == 0) dt else @min(dt * 2.0, s.dt_max);
    if (s.charge != 0 and s.steps > 0) {
        const del = lte[0];
        if (!(del >= 0.9 * dt)) {
            s.rej_lte += 1;
            ops.* = s.op_revert;
            if (std.math.isNan(del)) {
                s.order = 1;
                return retry(s, dt, dt / 8.0);
            }
            return retry(s, dt, del);
        }
        dt_next = @min(@max(del, s.dt_min), 2 * dt, s.dt_max);
    }
    if (s.charge != 0) {
        s.by_order[s.order - 1] += 1;
        s.order_run = if (s.order == s.run_order) s.order_run + 1 else 1;
        s.run_order = s.order;
        if (s.steps > 0 and s.order == 1) {
            // `Ops.bound`: the step's own method reads the first bound.
            const promote = methodAt(s.top, 2);
            const trial_del = if (promote == method) lte[0] else lte[1];
            if (!std.math.isNan(trial_del)) {
                const nd2 = @min(2.0 * dt, trial_del);
                if (nd2 > 1.05 * dt) s.order = 2;
                dt_next = @min(@max(nd2, s.dt_min), s.dt_max);
            }
        }
        // `Ops.advance`: record the pre-rotation slots, then rotate as
        // `std.mem.rotate(u8, p, nh - 1)` does.
        inline for (0..3) |k| {
            sc.sq[k] = pq[k];
            sc.sqt[k] = pqt[k];
        }
        rotateRight(pq, nh);
        rotateRight(pqt, nh);
    }
    var i: usize = s.dt_old.len - 1;
    while (i > 0) : (i -= 1) s.dt_old[i] = s.dt_old[i - 1];
    s.dt_old[0] = dt;
    s.t_new = t + dt;
    if (s.steps == 0) s.first_dt = dt;
    s.max_dt = @max(s.max_dt, dt);
    s.steps += 1;
    ops.* = s.op_commit;
    accept.* = 1;
    // `Ops.rebase` (it runs after the landing, which reads no ring).
    if (s.charge != 0 and s.state_q != 0) {
        sc.sqt[3] = pqt[0];
        const p0 = pqt[0];
        pqt[0] = pqt[1];
        pqt[1] = p0;
    }
    s.dt_next = dt_next;
    s.event = step_accept;
}

/// `Sim.end` after the commit, for an accepted lane: the breakpoint landing
/// and the next step's clamp. `nb` is `nextBp(t_new + min_break)` at the
/// committed state (inf for none); the host asks it twice there and its
/// caches make both answers this one.
pub fn stepEndB(s: anytype, nb: f64) void {
    if (s.event != step_accept) return;
    const t_new = s.t_new;
    var dt_next = s.dt_next;
    s.t = t_new;
    const inf = std.math.inf(f64);
    if (s.bp_next != inf) {
        const bp = s.bp_next;
        const landed = if (s.bp_clamped != 0)
            @abs(t_new - bp) <= s.min_break
        else
            bp - t_new <= s.delmin or almostEqualUlps(t_new, bp, 100);
        if (landed) {
            s.bp_landings += 1;
            s.order = 1;
            var shrink = s.bp_save_dt;
            if (nb != inf) shrink = @min(shrink, nb - t_new);
            dt_next = @min(dt_next, 0.1 * shrink);
        }
        s.bp_next = inf;
    }
    s.bp_next = nb;
    s.bp_clamped = 0;
    if (nb != inf) {
        const dt_to_bp = nb - t_new;
        if (dt_to_bp < dt_next) {
            s.bp_save_dt = dt_next;
            dt_next = dt_to_bp;
            s.bp_clamped = 1;
        }
    }
    s.dt = dt_next;
    if (t_new + s.dt > s.t_stop) s.dt = s.t_stop - t_new;
}

// ---- private ----

/// `tran.methodAt` over method codes, for a `top` of order <= 2.
fn methodAt(top: u8, k: u8) u8 {
    return if (k == 1) 0 else top;
}

/// `integrator.coeffs` over method codes; `Coeffs.ag1` defaults to 1.
fn coeffs(method: u8, dt: f64, dt_prev: f64, xmu: f64) [3]f64 {
    return switch (method) {
        0 => .{ 1.0 / dt, 1, 0 },
        1 => .{ 1.0 / dt / (1.0 - xmu), xmu / (1.0 - xmu), 0 },
        else => blk: {
            const r = dt / dt_prev;
            break :blk .{ (1.0 + 2.0 * r) / ((1.0 + r) * dt), 1, r * r / ((1.0 + r) * dt) };
        },
    };
}

/// `tran.atStop`.
fn atStop(t: f64, t_stop: f64) bool {
    if (t == t_stop) return true;
    if ((t < 0) != (t_stop < 0)) return false;
    const a: i64 = @bitCast(t);
    const b: i64 = @bitCast(t_stop);
    return @abs(a - b) <= 100;
}

/// `tran.almostEqualUlps`.
fn almostEqualUlps(a: f64, b: f64, max_ulps: i64) bool {
    if (a == b) return true;
    const lex = struct {
        fn f(x: f64) i128 {
            const i: i64 = @bitCast(x);
            return if (i < 0) @as(i128, std.math.minInt(i64)) - i else i;
        }
    }.f;
    return @abs(lex(a) - lex(b)) <= max_ulps;
}

/// A rejected attempt's tail in `Sim.end`: the retry dt and the floor
/// retry, or underflow.
fn retry(s: anytype, dt: f64, next: f64) void {
    s.event = step_reject;
    s.dt = next;
    if (s.dt < s.dt_min and dt > s.dt_min and s.t + s.dt_min > s.t) s.dt = s.dt_min;
    if (s.dt < s.dt_min) s.status = step_underflow;
}

fn rotateRight(p: anytype, nh: usize) void {
    const last = p[nh - 1];
    var i: usize = nh - 1;
    while (i > 0) : (i -= 1) p[i] = p[i - 1];
    p[0] = last;
}
