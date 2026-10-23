//! Per-lane bodies of the GPU lane backend (dev/research/plans/phase4),
//! generic over the pointer type so the host runs them as the oracle tests'
//! loops and the device as one thread per lane (`lane_device.zig`):
//! `LaneLu`'s refactor and solve restated over lane-major values, the
//! Newton update, gates and finalize of `converger.newtonLanes`, the
//! transient's companion residual, assembly, predictor, LTE bound and
//! accepted-step update. Lane l of n holds element i at `i * n + l`
//! (lane-major); the LTE tape ring is lane-minor. Each body matches its
//! oracle's operation order, so a host run is bitwise the oracle.
//! std only: the kernel root compiles this file for NVPTX and AMDGCN.

const std = @import("std");

/// Offsets (u32 words) of a frozen `SparseLu` tape inside one index buffer
/// `x`, plus its scalars. Passed by value. Built by `buildTab` from the
/// lead lane's factored `SparseLu`; every lane replays it.
pub const LaneTab = extern struct {
    n: u32,
    nnz: u32,
    nnz_l: u32,
    nnz_u: u32,
    growth: f64,
    /// 1 / `direct.pivot_tol`: `LaneSolver.wideMultipliers`' bound.
    wide: f64,
    col_ptr: u32, // n + 1: A's CSC columns
    prow: u32, // nnz: A entry -> pivot position
    q: u32, // n: pivot step -> column
    pinv: u32, // n: row -> pivot step
    lp: u32, // n + 1
    li: u32, // nnz_l
    up: u32, // n + 1
    ui: u32, // nnz_u
    flags: u32, // n: bit 0 void column, bit 1 scaled pivot
    /// `SparseLu.void_slots`: the A entries a fabricated pivot needs zero.
    void_slots: u32,
    n_void_slots: u32,
};

pub const void_bit: u32 = 1;
pub const scaled_bit: u32 = 2;

pub fn Bodies(comptime P: fn (type) type) type {
    return struct {
        /// `LaneLu.refactor` for lane l, plus `LaneSolver.wideMultipliers`:
        /// true when lane l must peel (dead or non-finite pivot, growth,
        /// a non-finite L/U entry, a wide multiplier, or a void column).
        /// `a` is A lane-major in pattern order (nnz rows); `w` is n rows of
        /// zeros on entry and on exit.
        pub fn refactor(t: LaneTab, x: P(u32), a: P(f64), lx: P(f64), ux: P(f64), udiag: P(f64), w: P(f64), n_lanes: u32, l: u32) bool {
            const N: usize = n_lanes;
            const L: usize = l;
            var bad = false;
            var nonfinite: u64 = 0;
            const inf = std.math.inf(f64);
            // `SparseLu.refactor`'s precheck: a fabricated pivot's slots
            // must still be zero.
            for (0..t.n_void_slots) |v| if (a[@as(usize, x[t.void_slots + v]) * N + L] != 0) return true;
            for (0..t.n) |k| {
                const c = x[t.q + k];
                for (x[t.col_ptr + c]..x[t.col_ptr + c + 1]) |p| w[@as(usize, x[t.prow + p]) * N + L] = a[p * N + L];
                for (x[t.up + k]..x[t.up + k + 1]) |p| {
                    const i: usize = x[t.ui + p];
                    const uki = w[i * N + L];
                    w[i * N + L] = 0;
                    ux[p * N + L] = uki;
                    nonfinite |= @as(u64, @bitCast(uki - uki));
                    for (x[t.lp + i]..x[t.lp + i + 1]) |pl| w[@as(usize, x[t.li + pl]) * N + L] -= lx[pl * N + L] * uki;
                }
                if (x[t.flags + k] & void_bit != 0) {
                    // `SparseLu.refactorColumns`' fabricated pivot.
                    udiag[k * N + L] = 1;
                    for (x[t.lp + k]..x[t.lp + k + 1]) |p| {
                        lx[p * N + L] = 0;
                        w[@as(usize, x[t.li + p]) * N + L] = 0;
                    }
                    w[k * N + L] = 0;
                    for (x[t.up + k]..x[t.up + k + 1]) |p| {
                        const i: usize = x[t.ui + p];
                        for (x[t.lp + i]..x[t.lp + i + 1]) |pl| w[@as(usize, x[t.li + pl]) * N + L] = 0;
                    }
                    continue;
                }
                var d = w[k * N + L];
                w[k * N + L] = 0;
                const dead = d == 0 or !(@abs(d) < inf);
                bad = bad or dead;
                if (dead) d = 1;
                udiag[k * N + L] = d;
                if (t.growth > 0 and x[t.flags + k] & scaled_bit == 0) {
                    var cbits: i64 = @bitCast(@abs(d));
                    for (x[t.lp + k]..x[t.lp + k + 1]) |p| {
                        const r: usize = x[t.li + p];
                        const v = w[r * N + L];
                        w[r * N + L] = 0;
                        cbits = @max(cbits, @as(i64, @bitCast(@abs(v))));
                        lx[p * N + L] = v / d;
                    }
                    const cmax: f64 = @bitCast(cbits);
                    bad = bad or !(@abs(d) >= t.growth * cmax);
                } else {
                    for (x[t.lp + k]..x[t.lp + k + 1]) |p| {
                        const r: usize = x[t.li + p];
                        const lv = w[r * N + L] / d;
                        nonfinite |= @as(u64, @bitCast(lv - lv));
                        lx[p * N + L] = lv;
                        w[r * N + L] = 0;
                    }
                }
            }
            if (nonfinite != 0) return true;
            // `wideMultipliers`: an unscaled column's L entry past 1 / pivot_tol.
            var worst: u64 = 0;
            for (0..t.n) |k| {
                if (x[t.flags + k] & scaled_bit != 0) continue;
                for (x[t.lp + k]..x[t.lp + k + 1]) |p| worst = @max(worst, @as(u64, @bitCast(@abs(lx[p * N + L]))));
            }
            return bad or !(@as(f64, @bitCast(worst)) <= t.wide);
        }

        /// `LaneLu.solve` of -rhs for lane l: dx = -A_l^-1 rhs_l, bitwise
        /// `LaneSolver.factorSolveNeg`'s vector lane. `y` is n rows of
        /// scratch.
        pub fn solveNeg(t: LaneTab, x: P(u32), lx: P(f64), ux: P(f64), udiag: P(f64), rhs: P(f64), y: P(f64), dx: P(f64), n_lanes: u32, l: u32) void {
            const N: usize = n_lanes;
            const L: usize = l;
            for (0..t.n) |r| y[@as(usize, x[t.pinv + r]) * N + L] = -rhs[r * N + L];
            for (0..t.n) |k| {
                const yk = y[k * N + L];
                if (yk != 0) for (x[t.lp + k]..x[t.lp + k + 1]) |p| {
                    y[@as(usize, x[t.li + p]) * N + L] -= lx[p * N + L] * yk;
                };
            }
            var k: usize = t.n;
            while (k > 0) {
                k -= 1;
                const zk = y[k * N + L] / udiag[k * N + L];
                y[k * N + L] = zk;
                if (zk != 0) for (x[t.up + k]..x[t.up + k + 1]) |p| {
                    y[@as(usize, x[t.ui + p]) * N + L] -= ux[p * N + L] * zk;
                };
            }
            for (0..t.n) |j| dx[@as(usize, x[t.q + j]) * N + L] = y[j * N + L];
        }

        /// `converger.updateAndNorm` at W = 1 for lane l: x_old = x,
        /// x += dx, the delta score. Same operations in the same order.
        pub fn updateNorm(n: u32, x: P(f64), dx: P(f64), x_old: P(f64), current_row: P(u8), reltol: f64, abstol: f64, vntol: f64, n_lanes: u32, l: u32) f64 {
            const N: usize = n_lanes;
            const L: usize = l;
            const inf = std.math.inf(f64);
            var worst: i64 = 0;
            for (0..n) |i| {
                const xo = x[i * N + L];
                const d = dx[i * N + L];
                const xn = xo + d;
                x_old[i * N + L] = xo;
                x[i * N + L] = xn;
                const tol = reltol * @max(@abs(xn), @abs(xo)) + (if (current_row[i] != 0) abstol else vntol);
                const finite = @abs(xn) < inf and @abs(d) < inf;
                const q = @abs(d) / tol;
                const score = if (!finite) inf else if (q == q) q else 0;
                worst = @max(worst, @as(i64, @bitCast(score)));
            }
            return @bitCast(worst);
        }

        /// `plausible` for lane l: every |x_i| finite and <= limit.
        pub fn plausible(n: u32, x: P(f64), limit: f64, n_lanes: u32, l: u32) bool {
            for (0..n) |i| if (!(@abs(x[i * @as(usize, n_lanes) + l]) <= limit)) return false;
            return true;
        }

        /// The row-scaled residual gate, including the finite gate used by
        /// shooting PSS. A NaN residual fails even with an infinite floor.
        pub fn residual(n: u32, opts: NewtonOpts, x: P(f64), rhs: P(f64), vals: P(f64), diag: P(u32), n_lanes: u32, l: u32) bool {
            const N: usize = n_lanes;
            for (0..n) |i| {
                const scale = @abs(vals[@as(usize, diag[i]) * N + l]);
                const s: f64 = if (scale == 0) 1 else scale;
                const tol = @max(opts.residual_tol, 10.0 * s * (opts.reltol * @abs(x[i * N + l]) + opts.vntol));
                if (!(@abs(rhs[i * N + l]) <= tol)) return false;
            }
            return true;
        }

        /// `plausibleLimit` for lane l: the runaway bound read off the
        /// solve's first assembly (x, residual f, matrix values).
        pub fn plausibleLimit(n: u32, x: P(f64), f: P(f64), vals: P(f64), diag: P(u32), n_lanes: u32, l: u32) f64 {
            const N: usize = n_lanes;
            var g: f64 = 1;
            for (0..n) |i| g = @max(g, @abs(vals[@as(usize, diag[i]) * N + l]));
            var xm: u64 = 0;
            var fm: u64 = 0;
            for (0..n) |i| {
                xm = @max(xm, @as(u64, @bitCast(@abs(x[i * N + l]))));
                fm = @max(fm, @as(u64, @bitCast(@abs(f[i * N + l]))));
            }
            const scale = @min(@max(@as(f64, @bitCast(xm)), @as(f64, @bitCast(fm)) * g), 1e18);
            return if (!(1e3 * scale > x_plausible)) x_plausible else 1e3 * scale;
        }

        /// The transient companion residual and matrix of lane l at row/slot
        /// granularity: `integrator.companionAt(m, true, ...)` per row and
        /// `combineGC(ag0)` per slot. `method`: 0 BE, 1 trap, 2 gear-2.
        /// q1/q2 are the ring's logical slots 1 and 2 (`ring` physical
        /// slots via `perm`).
        pub fn companionRow(i: usize, method: u8, ag0: f64, ag1: f64, ag2: f64, rhs: P(f64), q: P(f64), ring: P(f64), perm: P(u8), n_hist: u32, n: u32, i_prev: P(f64), n_lanes: u32, l: u32) void {
            const N: usize = n_lanes;
            const L: usize = l;
            const s1: usize = perm[L * n_hist + 1];
            const s2: usize = perm[L * n_hist + 2];
            const a = q[i * N + L];
            const b = ring[(s1 * n + i) * N + L];
            const d = ag0 * (a - b);
            const base = rhs[i * N + L] + d;
            rhs[i * N + L] = switch (method) {
                1 => base - ag1 * i_prev[i * N + L],
                2 => base - ag2 * (b - ring[(s2 * n + i) * N + L]),
                else => base,
            };
        }

        /// A = G + ag0 C at slot p for lane l (`combinePlanes`'s order:
        /// g + alpha * c).
        pub fn combineSlot(p: usize, ag0: f64, g: P(f64), c: P(f64), a: P(f64), n_lanes: u32, l: u32) void {
            const i = p * @as(usize, n_lanes) + l;
            a[i] = g[i] + ag0 * c[i];
        }
    };
}

/// `converger.x_plausible`, the runaway floor. Kept equal by a host test.
pub const x_plausible = 1e9;

/// The LTE inputs of one step, per lane (uploaded with `LaneScalars`).
pub const LteIn = extern struct { dt: f64, dt1: f64, dt2: f64, reltol: f64, abstol: f64, chgtol: f64, trtol: f64, ag0: f64, ag1: f64, ag2: f64 };

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

/// Largest system a failed lane factors densely on the device
/// (`arp_lane_peel_dense`); past it the host peels.
pub const dense_peel_max_n = 64;

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
pub const step_reject: u8 = 3;

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

/// A rejected attempt's tail in `Sim.end`: the retry dt and the floor
/// retry, or underflow.
fn retry(s: anytype, dt: f64, next: f64) void {
    s.event = step_reject;
    s.dt = next;
    if (s.dt < s.dt_min and dt > s.dt_min and s.t + s.dt_min > s.t) s.dt = s.dt_min;
    if (s.dt < s.dt_min) s.status = step_underflow;
}

/// `Sim.end` up to the commit, for a lane that attempted this round, from
/// its Newton gate, flip query and LTE pair: a rejection sets `ops` to
/// revert and the retry dt; an accepted step rotates the ring permutations
/// (`Ops.advance`, `Ops.rebase`), sets `ops` to commit and `accept`, and
/// leaves `t_new` and `dt_next` for `stepEndB`. A gate asking for
/// `reject_at` or `$bound_step` parks the lane. `pq`/`pqt` are the lane's
/// `nh` ring slots.
pub fn stepEndA(s: anytype, gate: lk_Gate, query: u32, lte: [2]f64, ops: anytype, accept: anytype, sc: anytype, pq: anytype, pqt: anytype, nh: usize) void {
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

fn rotateRight(p: anytype, nh: usize) void {
    const last = p[nh - 1];
    var i: usize = nh - 1;
    while (i > 0) : (i -= 1) p[i] = p[i - 1];
    p[0] = last;
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

const lk_Gate = Gate;

/// Newton per-lane gate state, one record per lane.
pub const Gate = extern struct {
    /// `plausibleLimit`, set at iteration 0.
    limit: f64,
    scaled: f64,
    iterations: u32,
    /// Bit 0 active, 1 init_fix, 2 converged, 3 singular, 4 gates passed,
    /// 5 a device asked to reject the step or bound the next one
    /// (`updateState`), which hands the lane back to the host.
    bits: u32,
    /// The lane's Newton iteration (0 at its first tick): each lane keeps
    /// its own, so a lane may carry its Newton over into the next round.
    tick: u32 = 0,
    _pad: u32 = 0,
};

pub const NewtonOpts = extern struct {
    reltol: f64,
    abstol: f64,
    vntol: f64,
    gmin: f64 = 0,
    residual_tol: f64 = std.math.inf(f64),
    /// `converger.Options.max_iter`: `finalize` retires a lane after this
    /// many ticks, unconverged.
    max_iter: u32 = std.math.maxInt(u32),
    _pad: u32 = 0,
};

test "lane_kernels finite residual gate matches converger for zero diagonals and nonfinite residuals" {
    const converger = @import("../converger.zig");
    const B = Bodies(HostP);
    const N = 3;
    const n = 4;
    var x: [n * N]f64 = @splat(0.25);
    var rhs: [n * N]f64 = @splat(0);
    var vals: [n * N]f64 = @splat(0);
    var diag = [_]u32{ 2, 0, 3, 1 };
    const residuals = [_]f64{ 0, 1e-12, 1e-6, 0.25, std.math.inf(f64), std.math.nan(f64) };
    const floors = [_]f64{ 1e-9, std.math.inf(f64) };
    for (floors) |floor| {
        const opts: converger.Options = .{ .residual_tol = floor };
        const kopts: NewtonOpts = .{ .reltol = opts.reltol, .abstol = opts.abstol, .vntol = opts.vntol, .residual_tol = floor };
        for (residuals, 0..) |r, trial| {
            for (0..n) |i| for (0..N) |l| {
                x[i * N + l] = @as(f64, @floatFromInt(i + l)) * 0.125;
                vals[i * N + l] = if ((i + l) % 2 == 0) 0 else 2.5e-3;
                rhs[i * N + l] = if (i == (trial + l) % n) r else 0;
            };
            for (0..N) |l| {
                var want = true;
                for (0..n) |i| {
                    want = want and @abs(rhs[i * N + l]) <= converger.residualGate(opts, vals[diag[i] * N + l], x[i * N + l]);
                }
                try testing.expectEqual(want, B.residual(n, kopts, &x, &rhs, &vals, &diag, N, @intCast(l)));
            }
        }
    }
}

/// VerA's `contract.SimState` (24 bytes, extern), restated so this file
/// stays std only; `analysis/gpu/lanes.zig` asserts the two layouts equal.
pub const SimState = extern struct {
    t: f64 = 0,
    dt: f64 = 0,
    kind: u8 = 0,
    initial_step: bool = false,
    final_step: bool = false,
    analog_initial: bool = true,
    iteration: u32 = 1,
};

/// The `LaneTab` of a factored sparse LU `lu` (`SparseLu`'s fields) over the
/// CSC column pointers `col_ptr`, its index arrays concatenated into one
/// buffer the caller owns. `growth` is the solver's refactor growth limit,
/// `pivot_tol` `direct.pivot_tol`.
pub fn buildTab(gpa: std.mem.Allocator, lu: anytype, col_ptr: []const u32, growth: f64, pivot_tol: f64) !struct { idx: []u32, tab: LaneTab } {
    const n = lu.n;
    const nnz = col_ptr[n];
    const nl: u32 = @intCast(lu.li.items.len);
    const nu: u32 = @intCast(lu.ui.items.len);
    var t: LaneTab = .{ .n = n, .nnz = nnz, .nnz_l = nl, .nnz_u = nu, .growth = growth, .wide = 1 / pivot_tol, .col_ptr = 0, .prow = 0, .q = 0, .pinv = 0, .lp = 0, .li = 0, .up = 0, .ui = 0, .flags = 0, .void_slots = 0, .n_void_slots = @intCast(lu.void_slots.items.len) };
    const parts = [_]struct { *u32, []const u32 }{
        .{ &t.col_ptr, col_ptr[0 .. n + 1] }, .{ &t.prow, lu.prow[0..nnz] }, .{ &t.q, lu.q[0..n] },
        .{ &t.pinv, lu.pinv[0..n] },          .{ &t.lp, lu.lp[0 .. n + 1] }, .{ &t.li, lu.li.items },
        .{ &t.up, lu.up[0 .. n + 1] },        .{ &t.ui, lu.ui.items },       .{ &t.void_slots, lu.void_slots.items },
    };
    var len: usize = n;
    for (parts) |p| len += p[1].len;
    const idx = try gpa.alloc(u32, len);
    var at: u32 = 0;
    for (parts) |p| {
        p[0].* = at;
        @memcpy(idx[at..][0..p[1].len], p[1]);
        at += @intCast(p[1].len);
    }
    t.flags = at;
    for (idx[at..][0..n], lu.void_col[0..n], lu.scaled_pivot[0..n]) |*f, v, s|
        f.* = (if (v) void_bit else 0) | (if (s) scaled_bit else 0);
    return .{ .idx = idx, .tab = t };
}

/// `Bodies`' pointer type on the host: a plain many-item pointer.
pub fn HostP(comptime T: type) type {
    return [*]T;
}

const testing = std.testing;

test "lane_kernels refactor and solveNeg over N lanes are bitwise LaneLu(W)'s lanes, and fail exactly where it fails or a multiplier runs wide" {
    const gpa = testing.allocator;
    const direct = @import("../direct.zig");
    const SparseLu = @import("../sparse_lu.zig").SparseLu;
    const LaneLu = @import("../lane/lu.zig").LaneLu;
    const W = 4;
    const V = @Vector(W, f64);
    const B = Bodies(HostP);
    var prng = std.Random.DefaultPrng.init(0x1a7e51);
    const r = prng.random();
    var good: usize = 0;
    var failed: usize = 0;
    for (0..64) |trial| {
        const n: u32 = @intCast(2 + trial % 20);
        const sys = try SparseLu.test_access.TestSystem.init(gpa, r, n, 0.3);
        defer sys.deinit(gpa);
        const nnz = sys.vals.len;
        var lead = try direct.Solver.init(gpa, n, sys.col_ptr, sys.row_idx, null, .{});
        defer lead.deinit();
        lead.factor(sys.vals, .{}) catch continue;
        const lu = lead.sparse() orelse continue;
        // LaneLu refuses a fabricated pivot outright; the bodies replay it
        // as `SparseLu.refactor` does, so those trials have no lane oracle.
        if (std.mem.indexOfScalar(bool, lu.void_col, true) != null) continue;
        var ll = try LaneLu(W).init(gpa, lu);
        defer ll.deinit(gpa);
        const built = try buildTab(gpa, lu, sys.col_ptr, lead.params.refactor_growth_limit, direct.pivot_tol);
        defer gpa.free(built.idx);
        const t = built.tab;
        // Lane-major values: lane l scales, redraws an entry or zeroes a column.
        const vals = try gpa.alloc(V, nnz);
        defer gpa.free(vals);
        const rhs = try gpa.alloc(V, n);
        defer gpa.free(rhs);
        const x_ll = try gpa.alloc(V, n);
        defer gpa.free(x_ll);
        const arr: [][W]f64 = @ptrCast(vals);
        for (arr, sys.vals) |*v, a| v.* = @splat(a);
        for (0..W) |l| switch (r.uintLessThan(u8, 6)) {
            0 => arr[r.uintLessThan(usize, nnz)][l] = 4 * r.float(f64) - 2,
            1 => {
                const c = r.uintLessThan(usize, n);
                for (sys.col_ptr[c]..sys.col_ptr[c + 1]) |p| arr[p][l] = 0;
            },
            else => for (arr, sys.vals) |*v, a| {
                v[l] = a * (0.95 + 0.1 * r.float(f64));
            },
        };
        for (rhs) |*v| {
            var a: [W]f64 = undefined;
            for (&a) |*e| e.* = 2 * r.float(f64) - 1;
            v.* = a;
        }
        var bad = ll.refactor(LaneLu(W).Plane{ .col_ptr = sys.col_ptr, .vals = vals }, lead.params.refactor_growth_limit);
        // LaneSolver.wideMultipliers.
        var worst: @Vector(W, u64) = @splat(0);
        for (0..n) |k| {
            if (lu.scaled_pivot[k]) continue;
            for (ll.lx[lu.lp[k]..lu.lp[k + 1]]) |e| worst = @max(worst, @as(@Vector(W, u64), @bitCast(@abs(e))));
        }
        const wide: [W]bool = !(@as(V, @bitCast(worst)) <= @as(V, @splat(1 / direct.pivot_tol)));
        for (wide, 0..) |w_, l| bad |= @as(u64, @intFromBool(w_)) << @intCast(l);
        for (x_ll, rhs) |*d, b| d.* = -b;
        ll.solve(x_ll, x_ll);

        // The same lanes through the per-lane bodies.
        const N = W;
        const a_f: [*]f64 = @ptrCast(vals.ptr);
        const lx = try gpa.alloc(f64, @max(t.nnz_l, 1) * N);
        defer gpa.free(lx);
        const ux = try gpa.alloc(f64, @max(t.nnz_u, 1) * N);
        defer gpa.free(ux);
        const scratch = try gpa.alloc(f64, 4 * n * N);
        defer gpa.free(scratch);
        @memset(scratch, 0);
        const udiag = scratch[0 .. n * N];
        const w = scratch[n * N ..][0 .. n * N];
        const y = scratch[2 * n * N ..][0 .. n * N];
        const dx = scratch[3 * n * N ..][0 .. n * N];
        const rhs_f: [*]f64 = @ptrCast(rhs.ptr);
        for (0..W) |l| {
            const fail = B.refactor(t, built.idx.ptr, a_f, lx.ptr, ux.ptr, udiag.ptr, w.ptr, N, @intCast(l));
            try testing.expectEqual((bad >> @intCast(l)) & 1 != 0, fail);
            for (w) |e| try testing.expectEqual(@as(f64, 0), e);
            if (fail) {
                failed += 1;
                continue;
            }
            good += 1;
            B.solveNeg(t, built.idx.ptr, lx.ptr, ux.ptr, udiag.ptr, rhs_f, y.ptr, dx.ptr, N, @intCast(l));
            for (0..n) |i| {
                const lanes_i: [W]f64 = x_ll[i];
                try testing.expectEqual(@as(u64, @bitCast(lanes_i[l])), @as(u64, @bitCast(dx[i * N + l])));
            }
        }
    }
    try testing.expect(good > 0 and failed > 0);
}

test "lane_kernels updateNorm and plausibleLimit at one lane are bitwise converger's W = 1 forms" {
    const converger = @import("../converger.zig");
    const B = Bodies(HostP);
    try testing.expectEqual(converger.x_plausible, x_plausible);
    var prng = std.Random.DefaultPrng.init(0x1a7e52);
    const r = prng.random();
    const N = 3;
    const n = 40;
    const special = [_]f64{ std.math.nan(f64), std.math.inf(f64), -std.math.inf(f64), -0.0, 0.0, 1e300, -1e-300 };
    for (0..50) |_| {
        var x: [n * N]f64 = undefined;
        var dx: [n * N]f64 = undefined;
        var xo: [n * N]f64 = undefined;
        var f: [n * N]f64 = undefined;
        var vals: [n * N]f64 = undefined;
        var cur: [n]u8 = undefined;
        var cur_b: [n]bool = undefined;
        var diag: [n]u32 = undefined;
        for (&x, &dx, &f, &vals) |*a, *b, *c, *d| {
            a.* = if (r.uintLessThan(u8, 20) == 0) special[r.uintLessThan(usize, special.len)] else 10 * r.float(f64) - 5;
            b.* = if (r.uintLessThan(u8, 20) == 0) special[r.uintLessThan(usize, special.len)] else r.float(f64) - 0.5;
            c.* = 1e3 * r.float(f64) - 500;
            d.* = 1e4 * r.float(f64);
        }
        for (&cur, &cur_b, &diag, 0..) |*c, *cb, *d, i| {
            cb.* = r.boolean();
            c.* = @intFromBool(cb.*);
            d.* = @intCast((i * 7) % n);
        }
        for (0..N) |l| {
            var xl: [n]f64 = undefined;
            var dl: [n]f64 = undefined;
            var ol: [n]f64 = undefined;
            var fl: [n]f64 = undefined;
            var vl: [n]f64 = undefined;
            for (0..n) |i| {
                xl[i] = x[i * N + l];
                dl[i] = dx[i * N + l];
                fl[i] = f[i * N + l];
                vl[i] = vals[i * N + l];
            }
            const Sys = struct { n: u32, diag_slots: []const u32 };
            const want_lim = converger.plausibleLimit(Sys{ .n = n, .diag_slots = &diag }, @as([]const f64, &xl), @as([]const f64, &fl), @as([]const f64, &vl));
            const got_lim = B.plausibleLimit(n, &x, &f, &vals, &diag, N, @intCast(l));
            try testing.expectEqual(@as(u64, @bitCast(want_lim)), @as(u64, @bitCast(got_lim)));
            const want = converger.updateAndNorm(1, &xl, &dl, &ol, &cur_b, 1e-3, 1e-12, 1e-6, true);
            const got = B.updateNorm(n, &x, &dx, &xo, &cur, 1e-3, 1e-12, 1e-6, N, @intCast(l));
            try testing.expectEqual(@as(u64, @bitCast(want)), @as(u64, @bitCast(got)));
            for (0..n) |i| {
                try testing.expectEqual(@as(u64, @bitCast(xl[i])), @as(u64, @bitCast(x[i * N + l])));
                try testing.expectEqual(@as(u64, @bitCast(ol[i])), @as(u64, @bitCast(xo[i * N + l])));
            }
            try testing.expectEqual(B.plausible(n, &x, got_lim, N, @intCast(l)), blk: {
                for (xl) |e| if (!(@abs(e) <= got_lim)) break :blk false;
                break :blk true;
            });
        }
    }
}
