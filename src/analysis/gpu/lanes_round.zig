//! The host-stepped end of a GPU lane round. In: each round lane's Newton
//! solution. Out: the step-end numbers `tran.Sim.end` decides from (the
//! state-flip query, the charges and their tape at x, the LTE bounds, the
//! probe rows), then the decisions' device mutations: latch commits and
//! reverts, the accepted histories, the stateful-charge rebase.
//! Cites: phase4-gpu-lanes.md G1 and G6.
//!
//! Contents:
//! - `pub fn probe(self: *GpuLanes) !void`
//! - `pub fn apply(self: *GpuLanes) !void`
//! - `pub fn queueAccept(self: *GpuLanes, l: usize) void`
//! - `pub fn ctlPass(self: *GpuLanes) !void`
//! - `pub fn chargePass(self: *GpuLanes, mask: *Buffer, x: *Buffer) !void`
//! - `pub fn chargeKernels(self: *GpuLanes, mask: *Buffer, x: *Buffer) !void`
//! - `pub fn acceptKernel(self: *GpuLanes) !void`
//! - `pub fn rebaseKernel(self: *GpuLanes) !void`
//! - `pub fn lteKernel(self: *GpuLanes) !void`
//! - `pub fn gatherKernel(self: *GpuLanes) !void`

const std = @import("std");
const assert = std.debug.assert;
const gompute = @import("gompute");
const device_ir = @import("device").abi;
const root = @import("../types.zig");
const lanes = @import("lanes.zig");
const lanes_batch = @import("lanes_batch.zig");
const Circuit = root.Circuit;
const StateCtlOp = device_ir.StateCtlOp;
const GpuLanes = lanes.GpuLanes;
const Buffer = lanes_batch.Buffer;
const block = lanes_batch.block;
const arg = gompute.interface.arg;

/// The step-end numbers of the round lanes at their Newton solution:
/// the state-flip query, the charges and their tape at x, the LTE
/// bounds and the probe rows. One sync.
pub fn probe(self: *GpuLanes) !void {
    const N: usize = self.n_lanes;
    // 1. The flip query, read-only.
    for (self.ops, self.round) |*o, r| o.* = if (r != 0) 1 + @as(u8, @backingInt(StateCtlOp.query)) else 0;
    try self.put(self.buf(.ops), self.ops);
    try self.buf(.flags).fillAsync(0, N * 4, &self.stream);
    try ctlPass(self);
    // 2. The charges at x (`Sim.end`'s `evalQ`), and the tape into each
    // lane's logical slot 0.
    if (self.has_charge) {
        const nh: usize = self.n_hist;
        for (self.slot0, 0..) |*s, l| s.* = self.perm_qt[l * nh];
        try chargePass(self, self.buf(.round), self.buf(.x));
    }
    if (self.n_qt > 0) {
        // 3. The LTE bounds.
        try lteKernel(self);
        try self.get(self.lte_out, self.buf(.lte_out));
    }
    // 4. The probe rows.
    if (self.n_probes > 0) {
        try gatherKernel(self);
        try self.get(self.probe_out[0 .. N * self.n_probes], self.buf(.probe_out));
    }
    try self.get(self.query, self.buf(.flags));
    try self.stream.synchronize();
    // The query's copy has run; `Sim.end` fills `ops` afresh.
    @memset(self.ops, 0);
    const n: usize = self.n;
    for (self.round, 0..) |r, l| {
        if (r == 0) continue;
        const row = self.rows[l * n ..][0..n];
        for (self.probes, 0..) |p, k| row[p] = self.probe_out[l * self.n_probes + k];
    }
}

/// Runs the round's decisions (`Ops`) on the device: revert or commit
/// per lane, then each accepted lane's history update. Clears them.
pub fn apply(self: *GpuLanes) !void {
    const N: usize = self.n_lanes;
    try self.put(self.buf(.ops), self.ops);
    try self.buf(.flags).fillAsync(0, N * 4, &self.stream);
    try ctlPass(self);
    try self.put(self.buf(.sc), self.sc);
    try self.put(self.buf(.accept), self.accept_);
    try acceptKernel(self);
    if (self.state_q) try rebaseRound(self);
    try self.put(self.buf(.perm_q), self.perm_q);
    try self.put(self.buf(.perm_qt), self.perm_qt);
    try self.stream.synchronize();
    @memset(self.ops, 0);
    @memset(self.accept_, 0);
}

/// Records lane l's pre-rotation ring slots for `accept`'s history
/// update (an accepted step with charge) and rotates
/// its permutations as `std.mem.rotate` moves the host rings.
pub fn queueAccept(self: *GpuLanes, l: usize) void {
    const nh: usize = self.n_hist;
    const pq = self.perm_q[l * nh ..][0..nh];
    const pqt = self.perm_qt[l * nh ..][0..nh];
    for (0..3) |k| {
        self.sc[l].sq[k] = pq[k];
        self.sc[l].sqt[k] = pqt[k];
    }
    std.mem.rotate(u8, pq, nh - 1);
    std.mem.rotate(u8, pqt, nh - 1);
}

/// Every batch's lane ctl kernel with the per-lane `ops` on the device.
pub fn ctlPass(self: *GpuLanes) !void {
    const N: usize = self.n_lanes;
    for (self.batches) |*b| {
        const k = if (b.ctl) |*k| k else continue;
        if (b.count == 0) continue;
        var nl: u64 = N;
        var count: u64 = b.count;
        try self.launch(k, b.count * N, &.{ arg(&nl), arg(&count), self.buf(.ops).argPtr(), b.d_models.argPtr(), b.d_model_of.argPtr(), b.d_instances.argPtr(), b.d_states.argPtr(), self.buf(.flags).argPtr() });
    }
}

/// `Circuit.evalQ` at `x` for the lanes of `mask`: the q plane zeroed,
/// then every charged batch's lane charge kernel, which also writes the
/// tape into each lane's slot `slot0[l]` (uploaded here).
pub fn chargePass(self: *GpuLanes, mask: *Buffer, x: *Buffer) !void {
    try self.put(self.buf(.slot0), self.slot0);
    try chargeKernels(self, mask, x);
}

/// `chargePass` with `d_slot0` already on the device.
pub fn chargeKernels(self: *GpuLanes, mask: *Buffer, x: *Buffer) !void {
    const N: usize = self.n_lanes;
    try self.buf(.q).fillAsync(0, (self.n + 1) * N * @sizeOf(f64), &self.stream);
    for (self.batches) |*b| {
        const k = if (b.chg) |*k| k else continue;
        if (b.count == 0) continue;
        var nl: u64 = N;
        var count: u64 = b.count;
        var n_hist: u64 = self.n_hist;
        var n_qt: u64 = self.n_qt;
        var off: u64 = b.tape_off;
        const col = if (b.color) |*c| c else {
            var none: u64 = 0;
            try self.launch(k, N, &.{
                arg(&nl),              arg(&count),                 mask.argPtr(),             self.buf(.sims).argPtr(), x.argPtr(),
                b.d_gath.argPtr(),     b.d_rhs_idx.argPtr(),        b.d_models.argPtr(),       b.d_model_of.argPtr(),    b.d_instances.argPtr(),
                self.buf(.q).argPtr(), self.buf(.ring_qt).argPtr(), self.buf(.slot0).argPtr(), arg(&n_hist),             arg(&n_qt),
                arg(&off),             b.d_slots.argPtr(),          arg(&none),                arg(&none),
            });
            continue;
        };
        // As `evalAll` colours eval (06 6.12): one launch per colour,
        // the q rows of hub cells staged, then the row hubs, with q as
        // both planes (it reads, sums and resets each cell once per
        // plane, so the second pass rewrites the same sums).
        for (col.starts[0 .. col.starts.len - 1], col.starts[1..]) |lo, hi| {
            var ids_off: u64 = lo;
            var n_ids: u64 = hi - lo;
            try self.launch(k, (hi - lo) * N, &.{
                arg(&nl),              arg(&count),                 mask.argPtr(),             self.buf(.sims).argPtr(), x.argPtr(),
                b.d_gath.argPtr(),     col.d_rows.argPtr(),         b.d_models.argPtr(),       b.d_model_of.argPtr(),    b.d_instances.argPtr(),
                self.buf(.q).argPtr(), self.buf(.ring_qt).argPtr(), self.buf(.slot0).argPtr(), arg(&n_hist),             arg(&n_qt),
                arg(&off),             col.d_order.argPtr(),        arg(&ids_off),             arg(&n_ids),
            });
        }
        const hubs = &col.row_hubs;
        var n_hubs: u64 = hubs.n;
        try self.launch(&self.k.hub_reduce, hubs.n * N, &.{
            arg(&nl),                arg(&n_hubs),          mask.argPtr(),         hubs.d_cells.argPtr(), hubs.d_starts.argPtr(),
            hubs.d_entries.argPtr(), self.buf(.q).argPtr(), self.buf(.q).argPtr(),
        });
    }
}

/// `arp_lane_accept`: each accepted lane's `prev <- cur <- x` and its
/// charge and tape histories (`d_accept`, `d_sc`'s ring slots).
pub fn acceptKernel(self: *GpuLanes) !void {
    const N: usize = self.n_lanes;
    var nl: u64 = N;
    var n: u64 = self.n;
    var n_qt: u64 = self.n_qt;
    var nh: u64 = self.n_hist;
    var charge: u64 = @intFromBool(self.has_charge);
    try self.launch(&self.k.accept, (self.n + self.n_qt) * N, &.{
        arg(&nl),               arg(&n),                    arg(&n_qt),             arg(&nh),                    arg(&charge),
        self.buf(.sc).argPtr(), self.buf(.accept).argPtr(), self.buf(.x).argPtr(),  self.buf(.cur).argPtr(),     self.buf(.prev).argPtr(),
        self.buf(.q).argPtr(),  self.buf(.ring).argPtr(),   self.buf(.ip).argPtr(), self.buf(.ring_qt).argPtr(), self.buf(.ip_qt).argPtr(),
    });
}

/// `arp_lane_rebase` for the accepted lanes, after their charges at
/// cur landed in the q plane and the tape's slot 0.
pub fn rebaseKernel(self: *GpuLanes) !void {
    const N: usize = self.n_lanes;
    var nl: u64 = N;
    var n: u64 = self.n;
    var n_qt: u64 = self.n_qt;
    var nh: u64 = self.n_hist;
    try self.launch(&self.k.rebase, (self.n + self.n_qt) * N, &.{
        arg(&nl),                   arg(&n),               arg(&n_qt),               arg(&nh),               self.buf(.sc).argPtr(),
        self.buf(.accept).argPtr(), self.buf(.q).argPtr(), self.buf(.ring).argPtr(), self.buf(.ip).argPtr(), self.buf(.ring_qt).argPtr(),
        self.buf(.ip_qt).argPtr(),
    });
}

/// `arp_lane_lte`: each lane's LTE pair into `d_lte_out`.
pub fn lteKernel(self: *GpuLanes) !void {
    const N: usize = self.n_lanes;
    var nl: u64 = N;
    var nh: u64 = self.n_hist;
    var n_qt: u64 = self.n_qt;
    try self.launch(&self.k.lte, N, &.{
        arg(&nl), self.buf(.sc).argPtr(), self.buf(.ring_qt).argPtr(), self.buf(.perm_qt).argPtr(), self.buf(.ip_qt).argPtr(), arg(&nh), arg(&n_qt), self.buf(.lte_in).argPtr(), self.buf(.lte_out).argPtr(),
    });
}

/// `arp_lane_gather`: each lane's probe rows of `d_x` into `d_probe_out`.
pub fn gatherKernel(self: *GpuLanes) !void {
    const N: usize = self.n_lanes;
    var nl: u64 = N;
    var np: u64 = self.n_probes;
    try self.launch(&self.k.gather, self.n_probes * N, &.{ arg(&nl), self.buf(.sc).argPtr(), self.buf(.x).argPtr(), self.buf(.probes).argPtr(), arg(&np), self.buf(.probe_out).argPtr() });
}

// ---- private ----

/// `Sim.end`'s stateful-charge block for the accepted lanes, after the
/// commits: the planes at cur, the tape into the tape's logical slot 0
/// (`sqt[3]`), then `arp_lane_rebase`.
fn rebaseRound(self: *GpuLanes) !void {
    for (self.slot0, self.sc) |*s, sc| s.* = sc.sqt[3];
    try chargePass(self, self.buf(.accept), self.buf(.cur));
    try rebaseKernel(self);
}
