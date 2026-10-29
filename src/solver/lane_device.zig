//! GPU kernel root for the lane backend's model-independent kernels: lane-
//! major buffers in device memory in (element i of lane l of N at
//! `i * N + l`, the LTE tape ring included), the same buffers updated out.
//! One thread per lane, or per (entry, lane) with lanes innermost. Bodies:
//! `kernels/lane.zig`, `kernels/frequency.zig`, `kernels/periodic.zig`,
//! `kernels/gmres.zig`; the transient-step kernels live in
//! `lane_step_device.zig`. Every kernel is exported and launched at `block`
//! threads, which HIP bakes into `globalIdX`. A kernel image root: no host
//! module imports it.
//! Cites: dev/research/plans/phase4-gpu-lanes.md G5.
//!
//! Contents:
//! - `pub const block = 64`
//! - `pub fn lane() u32`
//!     - `pub fn at(s: @This(), u: u32) f64`

const gompute = @import("gompute");
const lk = @import("kernels/lane.zig");
const P = gompute.GlobalPtr;
const FB = @import("kernels/frequency.zig").Bodies(P);
const pk = @import("kernels/periodic.zig");
const gmres_lanes = @import("kernels/gmres.zig");
const SimState = lk.SimState;
const scan = @import("measure_scan");
const step = @import("lane_step_device.zig");
const LaneScalars = lk.LaneScalars;
const Gate = lk.Gate;
const NewtonOpts = lk.NewtonOpts;

/// `device_abi.gpu_lane_block_size`; the host asserts the two equal.
pub const block = 64;

const B = lk.Bodies(P);

const PB = pk.Bodies(P);

const GB = gmres_lanes.Bodies(P);

comptime {
    if (gompute.is_device) {
        gompute.exportRaw("arp_lane_refactor", &refactor);
        gompute.exportRaw("arp_lane_step_begin", &step.stepBegin);
        gompute.exportRaw("arp_lane_settle", &step.settle);
        gompute.exportRaw("arp_lane_set_limw", &step.setLimw);
        gompute.exportRaw("arp_lane_newton_init", &step.newtonInit);
        gompute.exportRaw("arp_lane_set_ops", &step.setOps);
        gompute.exportRaw("arp_lane_slot_zero", &step.slotZero);
        gompute.exportRaw("arp_lane_step_end_a", &step.stepEndA);
        gompute.exportRaw("arp_lane_step_end_b", &step.stepEndB);
        gompute.exportRaw("arp_lane_step_stop", &step.stepStop);
        gompute.exportRaw("arp_lane_ring_put", &step.ringPut);
        gompute.exportRaw("arp_lane_ring_next", &step.ringNext);
        gompute.exportRaw("arp_lane_peel_dense", &peelDense);
        gompute.exportRaw("arp_lane_pack", &pack);
        gompute.exportRaw("arp_lane_unpack", &unpack);
        gompute.exportRaw("arp_lane_solve", &solveNeg);
        gompute.exportRaw("arp_freq_fill", &frequencyFill);
        gompute.exportRaw("arp_freq_solve", &frequencySolve);
        gompute.exportRaw("arp_freq_cfill", &frequencyComplexFill);
        gompute.exportRaw("arp_freq_crefactor", &frequencyComplexRefactor);
        gompute.exportRaw("arp_freq_csolve", &frequencyComplexSolve);
        gompute.exportRaw("arp_periodic_inverse", &periodicInverse);
        gompute.exportRaw("arp_periodic_product", &periodicProduct);
        gompute.exportRaw("arp_periodic_forward", &periodicForward);
        gompute.exportRaw("arp_periodic_pack", &periodicPack);
        gompute.exportRaw("arp_periodic_unpack", &periodicUnpack);
        gompute.exportRaw("arp_periodic_copy", &periodicCopy);
        gompute.exportRaw("arp_gmres_start", &gmresStart);
        gompute.exportRaw("arp_gmres_advance", &gmresAdvance);
        gompute.exportRaw("arp_gmres_advance_only", &gmresAdvanceOnly);
        gompute.exportRaw("arp_lane_assemble", &assemble);
        gompute.exportRaw("arp_lane_update", &update);
        gompute.exportRaw("arp_lane_gates", &gates);
        gompute.exportRaw("arp_lane_gmin", &gmin);
        gompute.exportRaw("arp_lane_clear_q", &clearQ);
        gompute.exportRaw("arp_lane_hub_reduce", &hubReduce);
        gompute.exportRaw("arp_lane_meas", &meas);
        gompute.exportRaw("arp_lane_finalize", &finalize);
        gompute.exportRaw("arp_lane_predict", &step.predict);
        gompute.exportRaw("arp_lane_lte", &step.lte);
        gompute.exportRaw("arp_lane_accept", &step.accept);
        gompute.exportRaw("arp_lane_rebase", &step.rebase);
        gompute.exportRaw("arp_lane_gather", &step.gather);
    }
}

/// The calling thread's lane: one thread per lane, `block` per block.
pub fn lane() u32 {
    return @intCast(gompute.globalIdX(block));
}

// ---- private ----

/// A converged lane retains its final assembled charge while slower peers
/// continue Newton. Fixed-period PSS accepts this exact charge, unlike an
/// adaptive transient's later charge re-evaluation.
fn clearQ(n_lanes: u64, rows: u64, active: P(u8), q: P(f64)) callconv(gompute.kernel_callconv) void {
    const tid = gompute.globalIdX(block);
    if (tid >= rows * n_lanes) return;
    const l: usize = @intCast(tid % n_lanes);
    if (active[l] != 0) q[tid] = 0;
}

/// The device `.meas` passes (06 6.8), a thread per lane: a lane whose step
/// this round `report` accepted at or past `t_start` feeds each pass not
/// over the accepted solution `x` (`scan.Dev.feed`, `Stream.feed`'s
/// arithmetic); once every pass is over it records this round's ring slot
/// in `resolved[l]`, which stays 0xff until then, and feeds no more.
fn meas(n_lanes: u64, n_scans: u64, t_start: f64, report: P(lk.StepReport), x: P(f64), cols: P(scan.DevColumns), states: P(scan.Dev), resolved: P(u8), ring_slot: P(u32)) callconv(gompute.kernel_callconv) void {
    const l = lane();
    if (l >= n_lanes or resolved[l] != 0xff) return;
    const rep = report[l];
    if (rep.event != lk.step_accept or rep.t_new < t_start) return;
    const N: usize = @intCast(n_lanes);
    const Lane = struct {
        x: P(f64),
        n: usize,
        l: usize,
        /// Entry `u` of the gathered row.
        pub fn at(s: @This(), u: u32) f64 {
            return s.x[@as(usize, u) * s.n + s.l];
        }
    };
    const ctx: Lane = .{ .x = x, .n = N, .l = l };
    var all = true;
    for (0..@intCast(n_scans)) |k| {
        // In place: a pass touches a few of its fields per row. A `.find`
        // pass reads its card's WHEN pass, the one before it.
        const c: *const scan.DevColumns = @addrSpaceCast(&cols[k]);
        const d: *scan.Dev = @addrSpaceCast(&states[k * N + l]);
        const prev: *const scan.Dev = @addrSpaceCast(&states[(k -| 1) * N + l]);
        const y = c.read(rep.t_new, ctx);
        d.feed(prev, rep.t_new, y[0], y[1]);
        all = all and d.over;
    }
    if (all) resolved[l] = @intCast(ring_slot[0]);
}

/// Thread per (hub, lane) of `active`: the hub cell takes its staging cells
/// in order, in both planes, and they go back to -0.0
/// (`analysis/gpu/lane_color.zig`).
fn hubReduce(n_lanes: u64, n_hubs: u64, active: P(u8), cells: P(u32), starts: P(u32), entries: P(u32), a: P(f64), b: P(f64)) callconv(gompute.kernel_callconv) void {
    const tid = gompute.globalIdX(block);
    if (tid >= n_hubs * n_lanes) return;
    const N: usize = @intCast(n_lanes);
    const l: usize = @intCast(tid % n_lanes);
    if (active[l] == 0) return;
    const h: usize = @intCast(tid / n_lanes);
    const at = @as(usize, cells[h]) * N + l;
    var va = a[at];
    var vb = b[at];
    for (starts[h]..starts[h + 1]) |k| {
        const e = @as(usize, entries[k]) * N + l;
        va += a[e];
        vb += b[e];
        a[e] = -0.0;
        b[e] = -0.0;
    }
    a[at] = va;
    b[at] = vb;
}

/// Also counts the failing lanes into `fails[0]` (zeroed by the host), which
/// the host reads to decide whether to peel this tick. `gpu/freq.zig`
/// launches it too, with these arguments.
fn refactor(n_lanes: u64, t: lk.LaneTab, x: P(u32), active: P(u8), a: P(f64), lx: P(f64), ux: P(f64), udiag: P(f64), w: P(f64), fail: P(u8), fails: P(u32)) callconv(gompute.kernel_callconv) void {
    const l = lane();
    if (l >= n_lanes or active[l] == 0) return;
    const bad = B.refactor(t, x, a, lx, ux, udiag, w, @intCast(n_lanes), l);
    fail[l] = @intFromBool(bad);
    if (bad) _ = @atomicRmw(u32, &fails[0], .Add, 1, .monotonic);
}

/// A lane whose replay failed solves on its own dense LU
/// (`lk.denseSolveNeg`) in a scratch slot of n * n + n entries it claims
/// from `slot_next[0]` (zeroed by the host); A comes from the pattern's
/// CSC (`col_ptr`, `row_idx`). Solved: dx written, fail = 0. Singular:
/// fail = 2, which `finalize` treats as any failure and the host peel
/// skips, as a host factor's `SingularMatrix` leaves the lane failed. A
/// lane past the pool keeps fail = 1 and counts into `over[0]` for the
/// host. `first[0]` keeps this tick's lowest failed lane (all ones from the
/// host), and `seen[0]` counts every failed lane and `seen_lo[0]` keeps the
/// lowest across a whole multi-round launch (reset per launch), for the
/// stale-tape tests.
fn peelDense(n_lanes: u64, n: u64, col_ptr: P(u32), row_idx: P(u32), active: P(u8), fail: P(u8), a: P(f64), rhs: P(f64), dx: P(f64), scratch: P(f64), n_slots: u64, slot_next: P(u32), over: P(u32), first: P(u32), seen: P(u32), seen_lo: P(u32)) callconv(gompute.kernel_callconv) void {
    const l = lane();
    if (l >= n_lanes or active[l] == 0 or fail[l] != 1) return;
    _ = @atomicRmw(u32, &first[0], .Min, l, .monotonic);
    _ = @atomicRmw(u32, &seen[0], .Add, 1, .monotonic);
    _ = @atomicRmw(u32, &seen_lo[0], .Min, l, .monotonic);
    const slot: u64 = @atomicRmw(u32, &slot_next[0], .Add, 1, .monotonic);
    if (slot >= n_slots) {
        _ = @atomicRmw(u32, &over[0], .Add, 1, .monotonic);
        return;
    }
    const nn: usize = @intCast(n);
    const N: usize = @intCast(n_lanes);
    const L: usize = l;
    const m = scratch + @as(usize, @intCast(slot)) * (nn * nn + nn);
    const xs = m + nn * nn;
    for (0..nn * nn) |i| m[i] = 0;
    for (0..nn) |c| for (col_ptr[c]..col_ptr[c + 1]) |p| {
        m[@as(usize, row_idx[p]) * nn + c] = a[p * N + L];
    };
    for (0..nn) |i| xs[i] = rhs[i * N + L];
    if (!lk.denseSolveNeg(nn, m, xs)) {
        fail[l] = 2;
        return;
    }
    for (0..nn) |i| dx[i * N + L] = xs[i];
    fail[l] = 0;
}

/// Lanes `idx[0..k)` of the lane-major `src` (`rows` rows) packed lane by
/// lane into `staged` from entry `at`: `staged[at + j * rows + r] =
/// src[r * N + idx[j]]`, so a host peel downloads only the lanes it
/// factors.
fn pack(n_lanes: u64, rows: u64, k: u64, idx: P(u32), src: P(f64), staged: P(f64), at: u64) callconv(gompute.kernel_callconv) void {
    const tid = gompute.globalIdX(block);
    if (tid >= rows * k) return;
    const j: usize = @intCast(tid / rows);
    const r: usize = @intCast(tid % rows);
    staged[@as(usize, @intCast(at)) + tid] = src[r * @as(usize, @intCast(n_lanes)) + idx[j]];
}

/// `pack` inverted: `dst[r * N + idx[j]] = staged[at + j * rows + r]`.
fn unpack(n_lanes: u64, rows: u64, k: u64, idx: P(u32), staged: P(f64), at: u64, dst: P(f64)) callconv(gompute.kernel_callconv) void {
    const tid = gompute.globalIdX(block);
    if (tid >= rows * k) return;
    const j: usize = @intCast(tid / rows);
    const r: usize = @intCast(tid % rows);
    dst[r * @as(usize, @intCast(n_lanes)) + idx[j]] = staged[@as(usize, @intCast(at)) + tid];
}

fn solveNeg(n_lanes: u64, t: lk.LaneTab, x: P(u32), active: P(u8), fail: P(u8), lx: P(f64), ux: P(f64), udiag: P(f64), rhs: P(f64), y: P(f64), dx: P(f64)) callconv(gompute.kernel_callconv) void {
    const l = lane();
    if (l >= n_lanes or active[l] == 0 or fail[l] != 0) return;
    B.solveNeg(t, x, lx, ux, udiag, rhs, y, dx, @intCast(n_lanes), l);
}

fn frequencyFill(n_lanes: u64, n_points: u64, nnz: u64, positions: P(u32), g: P(f64), c: P(f64), point: P(u32), omega: P(f64), a: P(f64)) callconv(gompute.kernel_callconv) void {
    const i = gompute.globalIdX(block);
    if (i >= nnz * n_lanes) return;
    FB.fill(@intCast(i / n_lanes), @intCast(i % n_lanes), @intCast(n_lanes), @intCast(n_points), positions, g, c, point, omega, a);
}

fn frequencySolve(n_lanes: u64, t: lk.LaneTab, x: P(u32), fail: P(u8), lx: P(f64), ux: P(f64), diag: P(f64), rhs: P(f64), y: P(f64), out: P(f64), adjoint: u64) callconv(gompute.kernel_callconv) void {
    const l = lane();
    if (l >= n_lanes or fail[l] != 0) return;
    FB.solve(t, x, lx, ux, diag, rhs, y, out, @intCast(n_lanes), l, adjoint != 0);
}

fn frequencyComplexFill(n_lanes: u64, n_points: u64, nnz: u64, g: P(f64), c: P(f64), point: P(u32), omega: P(f64), a: P(f64)) callconv(gompute.kernel_callconv) void {
    const i = gompute.globalIdX(block);
    if (i >= nnz * n_lanes) return;
    FB.complexFill(@intCast(i / n_lanes), @intCast(i % n_lanes), @intCast(n_lanes), @intCast(n_points), @intCast(nnz), g, c, point, omega, a);
}

fn frequencyComplexRefactor(n_lanes: u64, t: lk.LaneTab, x: P(u32), active: P(u8), a: P(f64), lx: P(f64), ux: P(f64), d: P(f64), w: P(f64), fail: P(u8), fails: P(u32)) callconv(gompute.kernel_callconv) void {
    const l = lane();
    if (l >= n_lanes or active[l] == 0) return;
    const bad = FB.complexRefactor(t, x, a, lx, ux, d, w, @intCast(n_lanes), l);
    fail[l] = @intFromBool(bad);
    if (bad) _ = @atomicRmw(u32, &fails[0], .Add, 1, .monotonic);
}

/// `n_rhs` right-hand sides per lane, each 2n rows of `rhs` and `out`.
fn frequencyComplexSolve(n_lanes: u64, n_rhs: u64, t: lk.LaneTab, x: P(u32), fail: P(u8), lx: P(f64), ux: P(f64), d: P(f64), rhs: P(f64), y: P(f64), out: P(f64), adjoint: u64) callconv(gompute.kernel_callconv) void {
    const l = lane();
    if (l >= n_lanes or fail[l] != 0) return;
    const stride = 2 * @as(usize, t.n) * @as(usize, @intCast(n_lanes));
    for (0..@intCast(n_rhs)) |r| FB.complexSolve(t, x, lx, ux, d, rhs + r * stride, y, out + r * stride, @intCast(n_lanes), l, adjoint != 0);
}

fn periodicInverse(t: pk.Shape, active: P(u8), inv: P(f64), x: P(f64), time: P(f64)) callconv(gompute.kernel_callconv) void {
    const i = gompute.globalIdX(block);
    if (i >= @as(u64, t.n) * t.nt * t.points) return;
    PB.inverse(t, @intCast(i / t.points), @intCast(i % t.points), active, inv, x, time);
}

fn periodicProduct(t: pk.Shape, active: P(u8), row_ptr: P(u32), columns: P(u32), slots: P(u32), g: P(f64), c: P(f64), time: P(f64), gv: P(f64), cv: P(f64)) callconv(gompute.kernel_callconv) void {
    const i = gompute.globalIdX(block);
    if (i >= @as(u64, t.n) * t.nt * t.points) return;
    PB.product(t, @intCast(i / t.points), @intCast(i % t.points), active, row_ptr, columns, slots, g, c, time, gv, cv);
}

fn periodicForward(t: pk.Shape, active: P(u8), fwd: P(f64), omega: P(f64), gv: P(f64), cv: P(f64), out: P(f64)) callconv(gompute.kernel_callconv) void {
    const i = gompute.globalIdX(block);
    if (i >= @as(u64, t.n) * t.nf * t.points) return;
    PB.forward(t, @intCast(i / t.points), @intCast(i % t.points), active, fwd, omega, gv, cv, out);
}

fn periodicPack(t: pk.Shape, p: u64, active: P(u8), input: P(f64), rhs: P(f64)) callconv(gompute.kernel_callconv) void {
    const i = gompute.globalIdX(block);
    if (i >= @as(u64, 2) * t.n * ((t.nf + 1) / 2)) return;
    PB.pack(t, @intCast(i), @intCast(p), active, input, rhs);
}

fn periodicUnpack(t: pk.Shape, p: u64, active: P(u8), identity: P(u8), rhs: P(f64), x: P(f64), out: P(f64)) callconv(gompute.kernel_callconv) void {
    const i = gompute.globalIdX(block);
    if (i >= @as(u64, t.n) * t.nf) return;
    PB.unpack(t, @intCast(i), @intCast(p), active, identity, rhs, x, out);
}

fn periodicCopy(t: pk.Shape, p: u64, active: P(u8), input: P(f64), out: P(f64)) callconv(gompute.kernel_callconv) void {
    const i = gompute.globalIdX(block);
    if (i >= @as(u64, t.n) * t.nf) return;
    PB.copy(t, @intCast(i), @intCast(p), active, input, out);
}

fn gmresRequest(point: u64, request: gmres_lanes.Request, matvec_mask: P(u8), precond_mask: P(u8), counts: P(u32)) void {
    matvec_mask[point] = @intFromBool(request == .matvec);
    precond_mask[point] = @intFromBool(request == .precond);
    switch (request) {
        .done => {},
        .matvec => _ = @atomicRmw(u32, &counts[0], .Add, 1, .monotonic),
        .precond => _ = @atomicRmw(u32, &counts[1], .Add, 1, .monotonic),
    }
}

fn gmresStart(shape: gmres_lanes.Shape, states: P(gmres_lanes.State), b: P(f64), x: P(f64), input: P(f64), matvec_mask: P(u8), precond_mask: P(u8), counts: P(u32)) callconv(gompute.kernel_callconv) void {
    const point = gompute.globalIdX(block);
    if (point >= shape.points) return;
    gmresRequest(point, GB.start(shape, states, b, x, input, point), matvec_mask, precond_mask, counts);
}

fn gmresAdvance(shape: gmres_lanes.Shape, layout: gmres_lanes.Layout, states: P(gmres_lanes.State), b: P(f64), x: P(f64), work: P(f64), input: P(f64), output: P(f64), matvec_mask: P(u8), precond_mask: P(u8), counts: P(u32)) callconv(gompute.kernel_callconv) void {
    const point = gompute.globalIdX(block);
    if (point >= shape.points) return;
    gmresRequest(point, GB.advance(shape, layout, states, b, x, work, input, output, point), matvec_mask, precond_mask, counts);
}

/// `gmresAdvance` for the points `only` selects, whose request the host
/// just served; the others keep theirs. `only` may alias a request mask:
/// each point reads its own entry before rewriting it.
fn gmresAdvanceOnly(shape: gmres_lanes.Shape, layout: gmres_lanes.Layout, states: P(gmres_lanes.State), b: P(f64), x: P(f64), work: P(f64), input: P(f64), output: P(f64), matvec_mask: P(u8), precond_mask: P(u8), counts: P(u32), only: P(u8)) callconv(gompute.kernel_callconv) void {
    const point = gompute.globalIdX(block);
    if (point >= shape.points or only[point] == 0) return;
    gmresRequest(point, GB.advance(shape, layout, states, b, x, work, input, output, point), matvec_mask, precond_mask, counts);
}

/// The ground pin, then the companion rows and the combined matrix, per
/// (row or slot, lane), lanes innermost: thread t < n * N does row t / N,
/// the rest slot (t - n N) / N. Row 0 takes `rhs += x` and slot `d0` `g +=
/// 1` first, as `Circuit.groundStamp` does after the stamp. Without charge
/// A is G. Skips inactive lanes.
fn assemble(n_lanes: u64, n: u64, nnz: u64, d0: u64, charge: u64, sc: P(LaneScalars), active: P(u8), x: P(f64), rhs: P(f64), q: P(f64), ring: P(f64), perm: P(u8), n_hist: u64, i_prev: P(f64), g: P(f64), c: P(f64), a: P(f64)) callconv(gompute.kernel_callconv) void {
    const tid = gompute.globalIdX(block);
    if (tid >= (n + nnz) * n_lanes) return;
    const l: u32 = @intCast(tid % n_lanes);
    if (active[l] == 0) return;
    const s = sc[l];
    const N: usize = @intCast(n_lanes);
    const e: usize = @intCast(tid / n_lanes);
    if (e < n) {
        if (e == 0) rhs[l] += x[l];
        if (charge != 0) B.companionRow(e, @intCast(s.method), s.ag0, s.ag1, s.ag2, rhs, q, ring, perm, @intCast(n_hist), @intCast(n), i_prev, @intCast(n_lanes), l);
    } else {
        const p = e - @as(usize, @intCast(n));
        if (p == d0) g[p * N + l] += 1;
        if (charge != 0) B.combineSlot(p, s.ag0, g, c, a, @intCast(n_lanes), l) else a[p * N + l] = g[p * N + l];
    }
}

/// Separate from assembly so the diagonal update cannot race the slot
/// workers that write A. Only needed for a nonzero Newton gmin.
fn gmin(n_lanes: u64, n: u64, value: f64, active: P(u8), x: P(f64), rhs: P(f64), a: P(f64), diag: P(u32)) callconv(gompute.kernel_callconv) void {
    const tid = gompute.globalIdX(block);
    if (tid >= n * n_lanes) return;
    const l: usize = @intCast(tid % n_lanes);
    if (active[l] == 0) return;
    const i: usize = @intCast(tid / n_lanes);
    const N: usize = @intCast(n_lanes);
    rhs[i * N + l] += value * x[i * N + l];
    a[@as(usize, diag[i]) * N + l] += value;
}

/// The delta update for every active lane: x_old = x, x += dx, the score;
/// the runaway bound at iteration 0; clears the lane's flag word for the
/// limit pass that follows. `iter` is the device iteration word.
fn update(n_lanes: u64, n: u64, opts: NewtonOpts, active: P(u8), gate: P(Gate), x: P(f64), dx: P(f64), x_old: P(f64), cur_row: P(u8), rhs: P(f64), a: P(f64), diag: P(u32), flags: P(u32)) callconv(gompute.kernel_callconv) void {
    const l = lane();
    if (l >= n_lanes or active[l] == 0) return;
    if (gate[l].tick == 0) gate[l].limit = B.plausibleLimit(@intCast(n), x, rhs, a, diag, @intCast(n_lanes), l);
    gate[l].scaled = B.updateNorm(@intCast(n), x, dx, x_old, cur_row, opts.reltol, opts.abstol, opts.vntol, @intCast(n_lanes), l);
    flags[l] = 0;
}

/// `finalizeStep`'s gates after the limit pass, in its order: limited,
/// first iterate, delta, plausible, then the row-scaled residual gate.
/// Sets `stage[l]` for the lanes the state pass must stage.
fn gates(n_lanes: u64, n: u64, opts: NewtonOpts, active: P(u8), gate: P(Gate), x: P(f64), rhs: P(f64), a: P(f64), diag: P(u32), flags: P(u32), stage: P(u8)) callconv(gompute.kernel_callconv) void {
    const l = lane();
    if (l >= n_lanes) return;
    stage[l] = 0;
    if (active[l] == 0) return;
    var g = gate[l];
    var ok = flags[l] & 1 == 0 and g.tick != 0 and !(g.scaled >= 1.0);
    if (ok) ok = B.plausible(@intCast(n), x, g.limit, @intCast(n_lanes), l);
    if (ok) ok = B.residual(@intCast(n), opts, x, rhs, a, diag, @intCast(n_lanes), l);
    g.bits = (g.bits & ~@as(u32, 16)) | (if (ok) @as(u32, 16) else 0);
    gate[l] = g;
    // INITFIX lanes do not stage (`stage = not(init_fix)`).
    stage[l] = @intFromBool(ok and g.bits & 2 == 0);
    if (stage[l] != 0) flags[l] = 0;
}

/// The loop exit per lane, after the state pass. Under `land_rejects`
/// (every transient) `updateStates` never reports a flip, so a lane that
/// passed leaves INITFIX or converges (x = x_old, as `newton` publishes
/// x_k); a state pass that asked for a reject or set a finite
/// `$bound_step` sets bit 5. A converged or
/// singular lane leaves `active`. Counts the lanes still active into
/// `count[0]` (zeroed by the host), and advances every active lane's
/// `$simparam("iteration")`.
fn finalize(n_lanes: u64, n: u64, max_iter: u64, active: P(u8), stage: P(u8), gate: P(Gate), flags: P(u32), fail: P(u8), x: P(f64), x_old: P(f64), sims: P(SimState), count: P(u32)) callconv(gompute.kernel_callconv) void {
    const l = lane();
    if (l >= n_lanes or active[l] == 0) return;
    var g = gate[l];
    if (fail[l] != 0) {
        g.bits |= 8;
        active[l] = 0;
        gate[l] = g;
        return;
    }
    if (stage[l] != 0 and flags[l] & 6 != 0) g.bits |= 32;
    const passed = g.bits & 16 != 0;
    if (passed and g.bits & 2 != 0) {
        g.bits &= ~@as(u32, 2); // INITFIX -> INITFLOAT
    } else if (passed) {
        const N: usize = @intCast(n_lanes);
        for (0..@as(usize, @intCast(n))) |i| x[i * N + l] = x_old[i * N + l];
        g.bits |= 4;
        g.iterations = g.tick + 1;
        active[l] = 0;
        gate[l] = g;
        return;
    }
    sims[l].iteration +|= 1;
    g.tick += 1;
    gate[l] = g;
    // The host loop's bound, per lane: past `max_iter` ticks the lane
    // leaves unconverged.
    if (g.tick >= max_iter) {
        active[l] = 0;
        return;
    }
    _ = @atomicRmw(u32, &count[0], .Add, 1, .monotonic);
}
