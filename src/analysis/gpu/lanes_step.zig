//! Device step control of a GPU lane run. In: each lane's `LaneStep` and
//! its sources' breakpoint caches, and optionally its `.meas` passes. Out:
//! up to `max_rounds` whole rounds per launch on the device (begin,
//! predict, Newton, probe, decide, apply, next breakpoint), each round's
//! reports and probe rows in a ring the host drains after one sync.
//! Cites: phase4-gpu-lanes.md 06 6.8 (device `.meas`).
//!
//! Contents:
//! - `pub fn initSteps(self: *GpuLanes, steps: []const lk.LaneStep) !void`
//! - `pub fn initMeasures(self: *GpuLanes, scans: []const passes.Dev, cols: []const passes.DevColumns, t_start: f64) !void`
//! - `pub fn deviceLaunch(self: *GpuLanes, opts: converger.Options, rounds: u32) !void`
//! - `pub fn multiRound(self: *const GpuLanes) bool`
//! - `pub fn slotReport(self: *const GpuLanes, r: usize, l: usize) StepReport`
//! - `pub fn slotRow(self: *GpuLanes, r: usize, l: usize) []const f64`
//! - `pub fn measResolved(self: *const GpuLanes, l: usize) ?u8`
//! - `pub fn fetchMeasures(self: *GpuLanes) !void`
//! - `pub fn lanePasses(self: *const GpuLanes, l: usize, out: []passes.Dev) void`
//! - `pub fn fetchSteps(self: *GpuLanes, out: []lk.LaneStep) !void`
//! - `pub fn requestStop(self: *GpuLanes, l: usize) void`

const std = @import("std");
const assert = std.debug.assert;
const gompute = @import("gompute");
const device_ir = @import("device").abi;
const solver = @import("solver");
const passes = @import("measure").passes;
const devices = @import("devices.zig");
const hdl = @import("hdl.zig");
const lanes = @import("lanes.zig");
const lanes_batch = @import("lanes_batch.zig");
const lanes_newton = @import("lanes_newton.zig");
const lanes_round = @import("lanes_round.zig");
const lanes_tick = @import("lanes_tick.zig");
const lk = solver.lane_kernels;
const converger = solver.converger;
const StateCtlOp = device_ir.StateCtlOp;
const StepReport = lk.StepReport;
const backend = devices.backend;
const GpuLanes = lanes.GpuLanes;
const Error = lanes.Error;
const arg = gompute.interface.arg;
const max_rounds = GpuLanes.max_rounds;

/// Newton ticks per round of a multi-round launch. A lane needing more
/// carries its Newton over; most transient steps converge in 2 or 3.
const phased_ticks = 3;

/// Hands step control to the device: uploads each lane's `LaneStep`
/// (`tran.runGpuLanes` fills them after `Sim.init`), loads every
/// breakpoint batch's lane kernel and seeds its source caches from each
/// lane's `GpuPayload.brk`. Rounds then go through `deviceLaunch`.
pub fn initSteps(self: *GpuLanes, steps: []const lk.LaneStep) !void {
    const N: usize = self.n_lanes;
    std.debug.assert(steps.len == N);
    try self.buf(.step).upload(steps.ptr, N * @sizeOf(lk.LaneStep));
    const c0 = self.ckts[0];
    for (self.batches, 0..) |*b, bi| {
        const h0 = c0.batches[bi].hooks;
        if (h0.next_breakpoint == null or b.count == 0) continue;
        var name: [160]u8 = undefined;
        const p0 = h0.gpu_payload.?(c0.batches[bi].ctx);
        b.brk = hdl.kernel(backend.?, @ptrCast(h0.gpu_payload.?), lanes_batch.laneName(&name, p0.kernel, "brk") orelse return Error.NoLaneImage) catch |e|
            return if (e == error.KernelNotFound) Error.NoLaneImage else e;
        if (p0.brk.len == 0) continue;
        const W = 2 * @sizeOf(f64);
        std.debug.assert(p0.brk.len == b.count * W);
        const host = try self.gpa.alloc(u8, b.count * N * W);
        defer self.gpa.free(host);
        for (self.ckts, 0..) |c, l| {
            const p = c.batches[bi].hooks.gpu_payload.?(c.batches[bi].ctx);
            for (0..b.count) |id| @memcpy(host[(id * N + l) * W ..][0..W], p.brk[id * W ..][0..W]);
        }
        b.d_brk = try lanes_batch.upload(&self.k.refactor, host);
    }
    self.device_step = true;
}

/// Hands the `.meas` passes to the device (06 6.8): `scans` holds every
/// lane's `n` passes lane-major (pass k of lane l at `k * N + l`, as
/// `measure.Stream.deviceScans` gives them), `cols` the unknowns each
/// reads. Each round, a lane that accepts a step at or past `t_start`
/// feeds them its solution, and `measResolved` tells the round they
/// all finished in.
pub fn initMeasures(self: *GpuLanes, scans: []const passes.Dev, cols: []const passes.DevColumns, t_start: f64) !void {
    const N: usize = self.n_lanes;
    std.debug.assert(scans.len == cols.len * N and cols.len > 0);
    const k0 = &self.k.refactor;
    self.buf(.meas_cols).free();
    self.buf(.meas_cols).* = try lanes_batch.upload(k0, std.mem.sliceAsBytes(cols));
    self.buf(.meas).free();
    self.buf(.meas).* = try lanes_batch.upload(k0, std.mem.sliceAsBytes(scans));
    self.buf(.meas_resolved).free();
    self.buf(.meas_resolved).* = try k0.alloc(N);
    try self.buf(.meas_resolved).upload(self.meas_resolved.ptr, N);
    self.meas_host = try self.gpa.alloc(passes.Dev, scans.len);
    self.n_meas = @intCast(cols.len);
    self.meas_t_start = t_start;
    lanes_newton.clearGraphs(self);
}

/// `rounds` rounds under device step control (1 unless `multiRound`):
/// every stepping lane begins, predicts, solves Newton, probes, decides,
/// applies its decision, asks for its next breakpoint and lands it, all
/// on the device; each round's `report` and probe rows go to ring slot
/// r (`ring_report[r * N + l]`, rows at `(r * N + l) * n_probes`). One
/// sync at the end brings the ring down. Stops the host queued in `stop`
/// take effect before the first round. A multi-round launch replays one
/// captured round with its Newton unrolled to `max_iter` masked ticks,
/// and refreshes a stale tape only at its end.
pub fn deviceLaunch(self: *GpuLanes, opts: converger.Options, rounds: u32) !void {
    std.debug.assert(self.device_step);
    std.debug.assert(rounds >= 1 and rounds <= max_rounds);
    std.debug.assert(rounds == 1 or multiRound(self));
    std.debug.assert(opts.gmin_stamps.len == 0 and opts.force.len == 0 and !opts.init_fix);
    const N: usize = self.n_lanes;
    var nl: u64 = N;
    const nopts = lanes_newton.newtonOpts(opts);
    lanes_newton.prepareGraphs(self, nopts);
    // ponytail: N bytes up every launch for the host's stops; a flag
    // word first if that ever shows at 100k lanes.
    try self.put(self.buf(.stop), self.stop);
    try self.launch(&self.k.step_stop, N, &.{ arg(&nl), self.buf(.stop).argPtr(), self.buf(.step).argPtr() });
    try self.buf(.ring_slot).fillAsync(0, 4, &self.stream);
    try self.buf(.seen).fillAsync(0, 4, &self.stream);
    try self.buf(.seen_lo).fillAsync(0xff, 4, &self.stream);
    if (rounds == 1) {
        try roundKernels(self, nopts, opts.max_iter, false);
    } else {
        if (self.graph_round == null) {
            try self.stream.synchronize();
            self.graph_round = try lanes_newton.capture(self, roundKernels, .{ nopts, opts.max_iter, true });
        }
        for (0..rounds) |_| try self.graph_round.?.launch(&self.stream);
    }
    const np: usize = self.n_probes;
    try self.get(self.ring_report[0 .. rounds * N], self.buf(.ring_report));
    if (np > 0) try self.get(self.ring_rows[0 .. rounds * N * np], self.buf(.ring_rows));
    if (self.n_meas > 0) try self.get(self.meas_resolved, self.buf(.meas_resolved));
    try self.buf(.seen).downloadAtAsync(&self.words[0], 0, 4, &self.stream);
    try self.buf(.seen_lo).downloadAtAsync(&self.words[2], 0, 4, &self.stream);
    try self.stream.synchronize();
    // The stops' copy has run.
    @memset(self.stop, 0);
    self.prof.rounds += rounds;
    // A multi-round launch read no per-tick words (`ticks` counts a
    // single round's): refresh a tape most lanes kept failing on now,
    // from the lowest failing lane.
    if (rounds > 1) {
        self.prof.dense_peels += self.words[0];
        if (2 * @as(u64, self.words[0]) >= N * rounds) try lanes_newton.retapeFrom(self, self.words[2]);
    }
}

/// Whether `deviceLaunch` may run more than one round: every failed
/// replay solves on the device (one dense slot per lane), so no tick
/// needs the host, and the driver can capture graphs.
pub fn multiRound(self: *const GpuLanes) bool {
    return self.dense_slots >= self.n_lanes and self.k.refactor.hasGraphs() and
        !self.prof.phases and !self.ckts[0].knobs.gpu_nographs;
}

/// Lane l's report in ring slot r of the last `deviceLaunch`.
pub fn slotReport(self: *const GpuLanes, r: usize, l: usize) StepReport {
    return self.ring_report[r * self.n_lanes + l];
}

/// Lane l's accepted row of ring slot r into `rows`, as `probe` leaves
/// rows: the probe positions of an n-long vector.
pub fn slotRow(self: *GpuLanes, r: usize, l: usize) []const f64 {
    const np: usize = self.n_probes;
    const row = self.rows[l * self.n ..][0..self.n];
    const src = self.ring_rows[(r * self.n_lanes + l) * np ..][0..np];
    for (self.probes, src) |p, v| row[p] = v;
    return row;
}

/// The ring slot of the last launch whose round finished lane l's
/// passes, or null.
pub fn measResolved(self: *const GpuLanes, l: usize) ?u8 {
    const r = self.meas_resolved[l];
    return if (r == 0xff) null else r;
}

/// Every lane's passes into `meas_host`, for `lanePasses`.
pub fn fetchMeasures(self: *GpuLanes) !void {
    try self.buf(.meas).download(self.meas_host.ptr, self.meas_host.len * @sizeOf(passes.Dev));
}

/// Lane l's passes of the last `fetchMeasures` into `out`.
pub fn lanePasses(self: *const GpuLanes, l: usize, out: []passes.Dev) void {
    const N: usize = self.n_lanes;
    for (out, 0..) |*o, k| o.* = self.meas_host[k * N + l];
}

/// Every lane's `LaneStep` into `out`, for the lanes the host finishes.
pub fn fetchSteps(self: *GpuLanes, out: []lk.LaneStep) !void {
    std.debug.assert(out.len == self.n_lanes);
    try self.buf(.step).download(out.ptr, out.len * @sizeOf(lk.LaneStep));
}

/// Stops lane l before the next round begins. Touches only lane l's
/// byte, so the threaded drain may call it; `deviceLaunch` uploads every
/// lane's byte each round.
pub fn requestStop(self: *GpuLanes, l: usize) void {
    self.stop[l] = 1;
}

// ---- private ----

/// One device-controlled round's kernels, `ringPut` included. Newton's
/// ticks are host-driven (`ticks`), or under `unrolled` `max_iter`
/// masked ticks that read nothing back, for capture.
fn roundKernels(self: *GpuLanes, nopts: lk.NewtonOpts, max_iter: u16, unrolled: bool) !void {
    const N: usize = self.n_lanes;
    var nl: u64 = N;
    var nh: u64 = self.n_hist;
    // Begin and predict.
    try self.launch(&self.k.step_begin, N, &.{ arg(&nl), self.buf(.step).argPtr(), self.buf(.sc).argPtr(), self.buf(.sims).argPtr(), self.buf(.lte_in).argPtr(), self.buf(.round).argPtr(), self.buf(.ops).argPtr(), self.buf(.accept).argPtr() });
    try lanes_tick.predictKernels(self);
    // Newton.
    var mi: u64 = max_iter;
    try self.launch(&self.k.newton_init, N, &.{ arg(&nl), arg(&mi), self.buf(.round).argPtr(), self.buf(.gate).argPtr(), self.buf(.sims).argPtr(), self.buf(.active).argPtr(), self.buf(.fail).argPtr() });
    if (unrolled) {
        // Each lane counts its own ticks, so one whose Newton is not
        // done carries it into the next round instead of every round
        // paying `max_iter` masked ticks.
        for (0..phased_ticks) |_| {
            try lanes_tick.linearTick(self, nopts);
            try lanes_tick.updateTick(self, nopts);
        }
    } else {
        // The round's lane count is on the device; N bounds it, which
        // only makes the stale-tape test stricter on the first tick.
        try lanes_newton.ticks(self, nopts, max_iter, N);
    }
    // The lanes whose Newton ended: `clearLimits` for them, and the
    // probe's mask.
    try self.launch(&self.k.settle, N, &.{ arg(&nl), self.buf(.step).argPtr(), self.buf(.active).argPtr(), self.buf(.decide).argPtr(), self.buf(.sc).argPtr() });
    for (self.batches) |*b| if (b.lim != null and b.has_lim) try lanes_tick.setLimw(self, b, self.buf(.decide), 0);
    // Probe: the flip query, the charges at x into the tape's logical
    // slot 0, the LTE pair and the probe rows.
    var query: u64 = 1 + @as(u64, @backingInt(StateCtlOp.query));
    try self.launch(&self.k.set_ops, N, &.{ arg(&nl), self.buf(.decide).argPtr(), self.buf(.ops).argPtr(), arg(&query) });
    try self.buf(.flags).fillAsync(0, N * 4, &self.stream);
    try lanes_round.ctlPass(self);
    var probe_slot: u64 = 0;
    var rebase_slot: u64 = 1;
    if (self.has_charge) {
        try self.launch(&self.k.slot_zero, N, &.{ arg(&nl), arg(&nh), arg(&probe_slot), self.buf(.perm_qt).argPtr(), self.buf(.sc).argPtr(), self.buf(.slot0).argPtr() });
        try lanes_round.chargeKernels(self, self.buf(.decide), self.buf(.x));
    }
    if (self.n_qt > 0) try lanes_round.lteKernel(self);
    if (self.n_probes > 0) try lanes_round.gatherKernel(self);
    // Decide (`Sim.end` to the commit).
    try self.launch(&self.k.step_end_a, N, &.{
        arg(&nl),                    arg(&nh),                 self.buf(.decide).argPtr(), self.buf(.step).argPtr(), self.buf(.gate).argPtr(),   self.buf(.flags).argPtr(),
        self.buf(.lte_out).argPtr(), self.buf(.ops).argPtr(),  self.buf(.accept).argPtr(), self.buf(.sc).argPtr(),   self.buf(.perm_q).argPtr(), self.buf(.perm_qt).argPtr(),
        self.buf(.brk_t).argPtr(),   self.buf(.best).argPtr(),
    });
    // Apply: the latches, the accepted histories, the stateful rebase.
    try self.buf(.flags).fillAsync(0, N * 4, &self.stream);
    try lanes_round.ctlPass(self);
    try lanes_round.acceptKernel(self);
    if (self.state_q) {
        try self.launch(&self.k.slot_zero, N, &.{ arg(&nl), arg(&nh), arg(&rebase_slot), self.buf(.perm_qt).argPtr(), self.buf(.sc).argPtr(), self.buf(.slot0).argPtr() });
        try lanes_round.chargeKernels(self, self.buf(.accept), self.buf(.cur));
        try lanes_round.rebaseKernel(self);
    }
    // The next breakpoint at the committed state, then the landing.
    for (self.batches) |*b| {
        const k = if (b.brk) |*k| k else continue;
        var count: u64 = b.count;
        const cache = if (b.d_brk) |*c| c.argPtr() else self.buf(.best).argPtr();
        try self.launch(k, N, &.{ arg(&nl), arg(&count), self.buf(.accept).argPtr(), self.buf(.brk_t).argPtr(), b.d_models.argPtr(), b.d_model_of.argPtr(), b.d_instances.argPtr(), cache, self.buf(.best).argPtr() });
    }
    try self.launch(&self.k.step_end_b, N, &.{ arg(&nl), self.buf(.step).argPtr(), self.buf(.best).argPtr(), self.buf(.report).argPtr() });
    var np: u64 = self.n_probes;
    try self.launch(&self.k.ring_put, N, &.{ arg(&nl), arg(&np), self.buf(.ring_slot).argPtr(), self.buf(.report).argPtr(), self.buf(.ring_report).argPtr(), self.buf(.probe_out).argPtr(), self.buf(.ring_rows).argPtr() });
    if (self.n_meas > 0) {
        var n_meas: u64 = self.n_meas;
        var t_start: f64 = self.meas_t_start;
        try self.launch(&self.k.meas, N, &.{ arg(&nl), arg(&n_meas), arg(&t_start), self.buf(.report).argPtr(), self.buf(.x).argPtr(), self.buf(.meas_cols).argPtr(), self.buf(.meas).argPtr(), self.buf(.meas_resolved).argPtr(), self.buf(.ring_slot).argPtr() });
    }
    try self.launch(&self.k.ring_next, 1, &.{self.buf(.ring_slot).argPtr()});
}
