//! Operating points and `.dc` points as GPU lanes (06 6.9). In: N lane
//! circuits at their start x, each `SimState` set. Out: the lanes' Newton
//! solutions and verdicts, one device Newton per point with no predictor
//! and no charge history, and the next `.dc` point's host rows reloaded
//! without rebuilding the lane run.
//! Cites: phase4-gpu-lanes.md 06 6.9.
//!
//! Contents:
//! - `pub fn initStatic(gpa: std.mem.Allocator, ckts: []const *Circuit, xs: []const []const f64) !*GpuLanes`
//! - `pub fn reload(self: *GpuLanes, xs: []const []const f64, limits_live: bool) !void`
//! - `pub fn solveStatic(self: *GpuLanes, xs: []const []f64, opts: converger.Options, out: []LaneNewton) !void`

const std = @import("std");
const assert = std.debug.assert;
const device_ir = @import("device").abi;
const solver = @import("solver");
const root = @import("../types.zig");
const lanes = @import("lanes.zig");
const lanes_batch = @import("lanes_batch.zig");
const lanes_init = @import("lanes_init.zig");
const lanes_newton = @import("lanes_newton.zig");
const lanes_round = @import("lanes_round.zig");
const lanes_tick = @import("lanes_tick.zig");

const lk = solver.lane_kernels;
const converger = solver.converger;
const Circuit = root.Circuit;
const StateCtlOp = device_ir.StateCtlOp;
const GpuLanes = lanes.GpuLanes;
const LaneNewton = lanes.LaneNewton;
const available = lanes.available;

/// `init` for operating points and `.dc` points (06 6.9): no step machine
/// and no charge history; lane l starts at `xs[l]`. Evaluates lane 0 at its
/// start on the host for the LU tape, so each lane's `SimState` must be set
/// first. Rounds go through `solveStatic`. Fails as `init`, or with
/// `error.BackendUnavailable` in a build without images.
pub fn initStatic(gpa: std.mem.Allocator, ckts: []const *Circuit, xs: []const []const f64) !*GpuLanes {
    // A build without GPU images has no device buffers to fill.
    if (comptime !available) return error.BackendUnavailable;
    assert(xs.len == ckts.len);
    const Seed = struct {
        n_hist: usize = 0,
        n_qt: usize = 0,
        dt: f64 = std.math.inf(f64),
        cur: []const f64,
        q_ring: []const []const f64 = &.{},
        i_prev: []const f64 = &.{},
        qt_ring: []const []const f64 = &.{},
        qt_i_prev: []const f64 = &.{},
    };
    const seeds = try gpa.alloc(Seed, ckts.len);
    defer gpa.free(seeds);
    for (seeds, xs) |*s, x| s.* = .{ .cur = x };
    // `init` factors lane 0 in its workspace, which an operating point
    // makes only on its first solve.
    if (ckts.len > 0) {
        _ = try ckts[0].workspace();
        ckts[0].evalNewton(xs[0], 0);
    }
    const self = try lanes_init.init(gpa, ckts, seeds, &.{});
    errdefer lanes_init.deinit(self);
    try putX(self, xs);
    @memset(self.round, 1);
    return self;
}

/// The next `.dc` point of a static batch: every attached lane's host rows
/// (Model, Instance, State and the lim plane, which the host's `recompute`
/// and predictor limit pass just wrote) and its start `xs[l]`. With
/// `limits_live` the lim planes are live for the first iterate, as after
/// `Circuit.applyLimits`. Clears `detached`. Invalidates the captured
/// graphs when the Model table grew.
pub fn reload(self: *GpuLanes, xs: []const []const f64, limits_live: bool) !void {
    if (comptime !available) return error.BackendUnavailable;
    assert(xs.len == self.n_lanes);
    const gpa = self.gpa;
    const k0 = &self.k.refactor;
    const ps = try gpa.alloc(device_ir.GpuPayload, self.n_lanes);
    defer gpa.free(ps);
    try self.stream.synchronize();
    for (self.batches, 0..) |*b, bi| {
        if (b.count == 0) continue;
        for (ps, self.ckts) |*p, c| p.* = c.batches[bi].hooks.gpu_payload.?(c.batches[bi].ctx);
        const pk = try lanes_batch.pack(gpa, ps);
        defer pk.deinit(gpa);
        // A grown Model table moves, and the captured graphs hold its old
        // address.
        if (pk.models.len > b.d_models.bytes) {
            const m = try lanes_batch.upload(k0, pk.models);
            b.d_models.free();
            b.d_models = m;
            lanes_newton.clearGraphs(self);
            self.graph_opts = null;
        } else if (pk.models.len > 0) try b.d_models.upload(pk.models.ptr, pk.models.len);
        try b.d_model_of.upload(pk.model_of.ptr, pk.model_of.len * 4);
        if (pk.instances.len > 0) try b.d_instances.upload(pk.instances.ptr, pk.instances.len);
        if (pk.states.len > 0) try b.d_states.upload(pk.states.ptr, pk.states.len);
        if (pk.lim.len > 0) try b.d_lim.upload(pk.lim.ptr, pk.lim.len * @sizeOf(f64));
    }
    try putX(self, xs);
    @memset(self.detached, false);
    @memset(self.round, 1);
    try self.put(self.buf(.round), self.round);
    for (self.batches) |*b| {
        if (b.lim == null or !b.has_lim or b.count == 0) continue;
        try lanes_tick.setLimw(self, b, self.buf(.round), @intFromBool(limits_live));
    }
}

/// Newton for the lanes of `round` (every lane after `initStatic` and
/// `reload`), each from its x in `d_x` at its circuit's `SimState`, as
/// `converger.newton` solves an operating point or a `.dc` point: no
/// predictor, and INITFIX first under `opts.init_fix`. Then the converged
/// lanes' latch query: DC Newton iterates on past a flipped latch, which
/// the device does not, so a flipped lane, and one whose state pass asked
/// for a reject or a bound, comes back `reject` for the host to solve. Lane
/// l's x lands in `xs[l]` and, for a round lane, its outcome in `out[l]`.
/// Asserts one row of `xs` and `out` per lane and a static batch.
pub fn solveStatic(self: *GpuLanes, xs: []const []f64, opts: converger.Options, out: []LaneNewton) !void {
    if (comptime !available) return error.BackendUnavailable;
    const N: usize = self.n_lanes;
    const n: usize = self.n;
    assert(xs.len == N);
    assert(out.len == N);
    assert(self.n_hist == 0);
    for (self.sims, self.ckts) |*s, c| s.* = c.sim;
    @memset(self.sc, std.mem.zeroes(lk.LaneScalars));
    for (self.sc) |*s| s.flags = 1;
    try self.put(self.buf(.sc), self.sc);
    self.prof.rounds += 1;
    try lanes_newton.newton(self, opts, out);
    const query = 1 + @as(u8, @backingInt(StateCtlOp.query));
    for (self.ops, self.round, out) |*o, on, r| o.* = if (on != 0 and r.result.converged and !r.singular) query else 0;
    try self.put(self.buf(.ops), self.ops);
    try self.buf(.flags).fillAsync(0, N * 4, &self.stream);
    try lanes_round.ctlPass(self);
    try self.get(self.query, self.buf(.flags));
    const host = try self.gpa.alloc(f64, (n + 1) * N);
    defer self.gpa.free(host);
    try self.buf(.x).downloadAtAsync(host.ptr, 0, host.len * @sizeOf(f64), &self.stream);
    try self.stream.synchronize();
    @memset(self.ops, 0);
    for (out, self.round, self.query) |*o, on, q| if (on != 0 and q & 1 != 0) {
        o.reject = true;
    };
    for (xs, 0..) |x, l| for (x[0..n], 0..) |*v, i| {
        v.* = host[i * N + l];
    };
}

// ---- private ----

/// Each lane's start `xs[l]` into `d_x`.
fn putX(self: *GpuLanes, xs: []const []const f64) !void {
    const N: usize = self.n_lanes;
    const n: usize = self.n;
    assert(xs.len == N);
    const host = try self.gpa.alloc(f64, (n + 1) * N);
    defer self.gpa.free(host);
    @memset(host, 0);
    for (xs, 0..) |x, l| for (x[0..n], 0..) |v, i| {
        host[i * N + l] = v;
    };
    try self.buf(.x).upload(host.ptr, host.len * @sizeOf(f64));
}
