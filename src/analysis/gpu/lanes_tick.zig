//! One Newton tick of a GPU lane run, as kernel launches on the lane stream.
//! In: the lanes' iterates and scalars on the device. Out: the predictor
//! and its limit pass, every batch's lane eval (by colour, with its hub
//! reduce, for coloured batches), the assembly, the LU replay and solve
//! with the device dense peel (graph A), and the update, limit and state
//! passes with the next iteration mask (graph B). No launch reads anything
//! back, so each half captures as one graph.
//! Cites: phase4-gpu-lanes.md G6, 06 6.11 and 6.12.
//!
//! Contents:
//! - `pub fn laneOps(self: *GpuLanes, l: usize) Ops`
//! - `pub fn predictRound(self: *GpuLanes) !void`
//! - `pub fn predictKernels(self: *GpuLanes) !void`
//! - `pub fn limitPass(self: *GpuLanes, mask: *Buffer, x: *Buffer, x_old: *Buffer, halves_: u64) !void`
//! - `pub fn linearTick(self: *GpuLanes, opts: lk.NewtonOpts) !void`
//! - `pub fn updateTick(self: *GpuLanes, opts: lk.NewtonOpts) !void`
//! - `pub fn setLimw(self: *GpuLanes, b: *LaneBatch, mask: *Buffer, value: u64) !void`

const std = @import("std");
const assert = std.debug.assert;
const gompute = @import("gompute");
const device_ir = @import("device").abi;
const solver = @import("solver");
const passes = @import("measure").passes;
const devices = @import("devices.zig");
const lanes = @import("lanes.zig");
const lanes_batch = @import("lanes_batch.zig");
const lk = solver.lane_kernels;
const backend = devices.backend;
const GpuLanes = lanes.GpuLanes;
const Prof = lanes.Prof;
const Buffer = lanes_batch.Buffer;
const LaneBatch = lanes_batch.LaneBatch;
const arg = gompute.interface.arg;

/// The backend `tran.Sim` drives lane l with.
pub fn laneOps(self: *GpuLanes, l: usize) lanes.Ops {
    return .{ .g = self, .l = @intCast(l) };
}

/// Starts a round: uploads the round lanes' scalars (`Ops.attempt`
/// filled them), predicts each lane's first iterate and runs its
/// predictor limit pass against cur. `round` marks the lanes in it.
pub fn predictRound(self: *GpuLanes) !void {
    try self.put(self.buf(.sc), self.sc);
    try self.put(self.buf(.sims), self.sims);
    try self.put(self.buf(.lte_in), self.lte_in);
    try self.put(self.buf(.round), self.round);
    try predictKernels(self);
}

/// The predictor and its limit pass for the lanes of `d_round`, from
/// the scalars already in `d_sc`.
pub fn predictKernels(self: *GpuLanes) !void {
    const N: usize = self.n_lanes;
    var nl: u64 = N;
    var n: u64 = self.n;
    try self.launch(&self.k.predict, self.n * N, &.{ arg(&nl), arg(&n), self.buf(.sc).argPtr(), self.buf(.cur).argPtr(), self.buf(.prev).argPtr(), self.buf(.x).argPtr(), self.buf(.x_old).argPtr() });
    // `applyLimits` leaves each limiting batch's lim plane live
    // (`limitPass` sets the round lanes' words).
    try limitPass(self, self.buf(.round), self.buf(.x), self.buf(.x_old), device_ir.lim_pass_limit);
}

/// The limit or state half of every batch's lane state kernel over the
/// lanes of `mask`, at `x` against `x_old`.
pub fn limitPass(self: *GpuLanes, mask: *Buffer, x: *Buffer, x_old: *Buffer, halves_: u64) !void {
    const N: usize = self.n_lanes;
    for (self.batches) |*b| {
        const k = if (b.lim) |*k| k else continue;
        if (b.count == 0) continue;
        if (halves_ == device_ir.lim_pass_limit and !b.has_lim) continue;
        var nl: u64 = N;
        var count: u64 = b.count;
        var halves = halves_;
        try self.launch(k, b.count * N, &.{
            arg(&nl),            arg(&count),         mask.argPtr(),         x.argPtr(),                x_old.argPtr(),
            b.d_gath.argPtr(),   b.d_models.argPtr(), b.d_model_of.argPtr(), b.d_instances.argPtr(),    b.d_lim.argPtr(),
            b.d_states.argPtr(), b.d_limw.argPtr(),   arg(&halves),          self.buf(.flags).argPtr(), self.buf(.sims).argPtr(),
        });
        // The predictor's limit pass makes the lim plane live, and so
        // does a fixed-period solve's first Newton pass, which runs
        // without the predictor. Per lane: a lane whose Newton carries
        // over keeps its word while others begin.
        if (halves_ == device_ir.lim_pass_limit and b.has_lim) try setLimw(self, b, mask, 1);
    }
}

/// Graph A: everything before the host handles a rare failed refactor.
pub fn linearTick(self: *GpuLanes, opts: lk.NewtonOpts) !void {
    const N: usize = self.n_lanes;
    var nl: u64 = N;
    var nn: u64 = self.n;
    var nnz: u64 = self.nnz;
    var d0: u64 = self.d0;
    var charge: u64 = @intFromBool(self.has_charge);
    var nh: u64 = self.n_hist;
    var tab = self.tab;
    var mark = Prof.now();
    try evalAll(self, self.buf(.active), self.buf(.x));
    try tickLap(self, &mark, .eval);
    try self.launch(&self.k.assemble, (self.n + self.nnz) * N, &.{
        arg(&nl),                 arg(&nn),                   arg(&nnz),             arg(&d0),                arg(&charge),
        self.buf(.sc).argPtr(),   self.buf(.active).argPtr(), self.buf(.x).argPtr(), self.buf(.rhs).argPtr(), self.buf(.q).argPtr(),
        self.buf(.ring).argPtr(), self.buf(.perm_q).argPtr(), arg(&nh),              self.buf(.ip).argPtr(),  self.buf(.g).argPtr(),
        self.buf(.c).argPtr(),    self.buf(.a).argPtr(),
    });
    if (opts.gmin > 0) {
        var value = opts.gmin;
        try self.launch(&self.k.gmin, self.n * N, &.{
            arg(&nl),                arg(&nn),              arg(&value),              self.buf(.active).argPtr(), self.buf(.x).argPtr(),
            self.buf(.rhs).argPtr(), self.buf(.a).argPtr(), self.buf(.diag).argPtr(),
        });
    }
    try tickLap(self, &mark, .assemble);
    try self.buf(.fails).fillAsync(0, 4, &self.stream);
    try self.buf(.first).fillAsync(0xff, 4, &self.stream);
    try self.launch(&self.k.refactor, N, &.{
        arg(&nl),                  arg(&tab),              self.buf(.tab_idx).argPtr(), self.buf(.active).argPtr(), self.buf(.a).argPtr(),
        self.buf(.lx).argPtr(),    self.buf(.ux).argPtr(), self.buf(.udiag).argPtr(),   self.buf(.w).argPtr(),      self.buf(.fail).argPtr(),
        self.buf(.fails).argPtr(),
    });
    try self.launch(&self.k.solve, N, &.{
        arg(&nl),               arg(&tab),              self.buf(.tab_idx).argPtr(), self.buf(.active).argPtr(), self.buf(.fail).argPtr(),
        self.buf(.lx).argPtr(), self.buf(.ux).argPtr(), self.buf(.udiag).argPtr(),   self.buf(.rhs).argPtr(),    self.buf(.y).argPtr(),
        self.buf(.dx).argPtr(),
    });
    if (self.dense_slots > 0) {
        try self.buf(.slot_next).fillAsync(0, 4, &self.stream);
        try self.buf(.over).fillAsync(0, 4, &self.stream);
        var slots: u64 = self.dense_slots;
        try self.launch(&self.k.peel_dense, N, &.{
            arg(&nl),                    arg(&nn),                      self.buf(.col_ptr).argPtr(), self.buf(.row_idx).argPtr(), self.buf(.active).argPtr(),
            self.buf(.fail).argPtr(),    self.buf(.a).argPtr(),         self.buf(.rhs).argPtr(),     self.buf(.dx).argPtr(),      self.buf(.dense).argPtr(),
            arg(&slots),                 self.buf(.slot_next).argPtr(), self.buf(.over).argPtr(),    self.buf(.first).argPtr(),   self.buf(.seen).argPtr(),
            self.buf(.seen_lo).argPtr(),
        });
    }
    try tickLap(self, &mark, .lu);
}

/// Graph B: the update, both state passes, and the next iteration mask.
pub fn updateTick(self: *GpuLanes, opts: lk.NewtonOpts) !void {
    const N: usize = self.n_lanes;
    var nl: u64 = N;
    var nn: u64 = self.n;
    var nopts = opts;
    var mark = Prof.now();
    try self.launch(&self.k.update, N, &.{
        arg(&nl),              arg(&nn),                 arg(&nopts),               self.buf(.active).argPtr(),  self.buf(.gate).argPtr(),
        self.buf(.x).argPtr(), self.buf(.dx).argPtr(),   self.buf(.x_old).argPtr(), self.buf(.cur_row).argPtr(), self.buf(.rhs).argPtr(),
        self.buf(.a).argPtr(), self.buf(.diag).argPtr(), self.buf(.flags).argPtr(),
    });
    try limitPass(self, self.buf(.active), self.buf(.x), self.buf(.x_old), device_ir.lim_pass_limit);
    try self.launch(&self.k.gates, N, &.{
        arg(&nl),                  arg(&nn),                arg(&nopts),           self.buf(.active).argPtr(), self.buf(.gate).argPtr(),
        self.buf(.x).argPtr(),     self.buf(.rhs).argPtr(), self.buf(.a).argPtr(), self.buf(.diag).argPtr(),   self.buf(.flags).argPtr(),
        self.buf(.stage).argPtr(),
    });
    try tickLap(self, &mark, .gates);
    try limitPass(self, self.buf(.stage), self.buf(.x_old), self.buf(.x_old), device_ir.lim_pass_state);
    try self.buf(.count).fillAsync(0, 4, &self.stream);
    var max_iter: u64 = opts.max_iter;
    try self.launch(&self.k.finalize, N, &.{
        arg(&nl),                 arg(&nn),                  arg(&max_iter),           self.buf(.active).argPtr(), self.buf(.stage).argPtr(),
        self.buf(.gate).argPtr(), self.buf(.flags).argPtr(), self.buf(.fail).argPtr(), self.buf(.x).argPtr(),      self.buf(.x_old).argPtr(),
        self.buf(.sims).argPtr(), self.buf(.count).argPtr(),
    });
    try tickLap(self, &mark, .state);
}

/// Batch b's lim-active word for the lanes of `mask`.
pub fn setLimw(self: *GpuLanes, b: *LaneBatch, mask: *Buffer, value: u64) !void {
    var nl: u64 = self.n_lanes;
    var v = value;
    try self.launch(&self.k.set_limw, self.n_lanes, &.{ arg(&nl), mask.argPtr(), b.d_limw.argPtr(), arg(&v) });
}

// ---- private ----

/// Zeroes the planes, then every batch's lane eval at `x` for the lanes
/// of `mask`.
fn evalAll(self: *GpuLanes, mask: *Buffer, x: *Buffer) !void {
    const N: usize = self.n_lanes;
    const f8 = @sizeOf(f64);
    try self.buf(.g).fillAsync(0, (self.nnz + 1) * N * f8, &self.stream);
    try self.buf(.c).fillAsync(0, (self.nnz + 1) * N * f8, &self.stream);
    try self.buf(.rhs).fillAsync(0, (self.n + 1) * N * f8, &self.stream);
    // Freeze q for lanes that have already passed Newton. Shooting's
    // trapezoidal history uses their last assembly, and another lane
    // taking an extra iteration must not erase it.
    var q_lanes: u64 = N;
    var q_rows: u64 = self.n + 1;
    try self.launch(&self.k.clear_q, (self.n + 1) * N, &.{ arg(&q_lanes), arg(&q_rows), mask.argPtr(), self.buf(.q).argPtr() });
    for (self.batches) |*b| {
        if (b.count == 0) continue;
        var nl: u64 = N;
        var count: u64 = b.count;
        const col = if (b.color) |*c| c else {
            var none: u64 = 0;
            try self.launch(&b.eval, N, &.{
                arg(&nl),              arg(&count),            self.buf(.sims).argPtr(), mask.argPtr(),         b.d_limw.argPtr(),
                x.argPtr(),            b.d_gath.argPtr(),      b.d_rhs_idx.argPtr(),     b.d_slots.argPtr(),    b.d_models.argPtr(),
                b.d_model_of.argPtr(), b.d_instances.argPtr(), self.buf(.g).argPtr(),    self.buf(.c).argPtr(), self.buf(.rhs).argPtr(),
                self.buf(.q).argPtr(), b.d_lim.argPtr(),       b.d_slots.argPtr(),       arg(&none),            arg(&none),
            });
            continue;
        };
        // One launch per colour, in order, then the hubs.
        for (col.starts[0 .. col.starts.len - 1], col.starts[1..]) |lo, hi| {
            var off: u64 = lo;
            var n_ids: u64 = hi - lo;
            try self.launch(&b.eval, (hi - lo) * N, &.{
                arg(&nl),              arg(&count),            self.buf(.sims).argPtr(), mask.argPtr(),         b.d_limw.argPtr(),
                x.argPtr(),            b.d_gath.argPtr(),      col.d_rows.argPtr(),      col.d_slots.argPtr(),  b.d_models.argPtr(),
                b.d_model_of.argPtr(), b.d_instances.argPtr(), self.buf(.g).argPtr(),    self.buf(.c).argPtr(), self.buf(.rhs).argPtr(),
                self.buf(.q).argPtr(), b.d_lim.argPtr(),       col.d_order.argPtr(),     arg(&off),             arg(&n_ids),
            });
        }
        inline for (.{ .{ "slot_hubs", .g, .c }, .{ "row_hubs", .rhs, .q } }) |h| {
            const hubs = &@field(col, h[0]);
            var n_hubs: u64 = hubs.n;
            try self.launch(&self.k.hub_reduce, hubs.n * N, &.{
                arg(&nl),                arg(&n_hubs),            mask.argPtr(),           hubs.d_cells.argPtr(), hubs.d_starts.argPtr(),
                hubs.d_entries.argPtr(), self.buf(h[1]).argPtr(), self.buf(h[2]).argPtr(),
            });
        }
    }
}

fn tickLap(self: *GpuLanes, last: *u64, phase: Prof.TickPhase) !void {
    if (!self.prof.phases) return;
    try self.stream.synchronize();
    const now = Prof.now();
    self.prof.tick_ns[@backingInt(phase)] += now - last.*;
    last.* = now;
}
