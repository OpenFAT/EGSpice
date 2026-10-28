//! Fixed-period shooting on the GPU lanes. In: each lane's seeded period
//! (x, q history, i_prev) and its fixed trapezoidal step count. Out: each
//! period integrated by device Newton, its waveform rows, and the accepted
//! Instance and State rows published to the host only when every copy
//! succeeded, so a failure reruns the period on the host from unchanged
//! state.
//! Cites: none (pss.zig's shooting).
//!
//! Contents:
//! - `pub fn integrateFixed(gpa: std.mem.Allocator, periods: []const FixedLane, opts: converger.Options, out: []bool) !void`

const std = @import("std");
const assert = std.debug.assert;
const device_ir = @import("device").abi;
const solver = @import("solver");
const root = @import("../types.zig");
const devices = @import("devices.zig");
const lanes = @import("lanes.zig");
const lanes_round = @import("lanes_round.zig");
const converger = solver.converger;
const Circuit = root.Circuit;
const SimState = device_ir.SimState;
const StateCtlOp = device_ir.StateCtlOp;
const backend = devices.backend;
const GpuLanes = lanes.GpuLanes;
const LaneNewton = lanes.LaneNewton;
const FixedLane = lanes.FixedLane;
const Error = lanes.Error;

/// The Newton work of fixed-period shooting, resident for the full period.
/// The host keeps each lane's shooting iteration and period. No transient
/// predictor or LTE probe runs here: PSS accepts the charge from the last
/// Newton assembly and preserves its finite residual gate and gmin.
/// On a device error, host x and device instances remain unchanged so the
/// caller can rerun the period through the scalar integrator.
pub fn integrateFixed(gpa: std.mem.Allocator, periods: []const FixedLane, opts: converger.Options, out: []bool) !void {
    if (comptime backend == null) return Error.NoLaneImage;
    std.debug.assert(periods.len > 0 and out.len == periods.len);
    const N = periods.len;
    const n: usize = periods[0].ckt.n;
    const steps = periods[0].n_samples;
    const Seed = struct {
        n_hist: usize = 3,
        n_qt: usize = 0,
        dt: f64,
        cur: []f64,
        q_ring: [3][]f64,
        i_prev: []f64,
        qt_ring: [3][]const f64 = @splat(&.{}),
        qt_i_prev: []const f64 = &.{},
    };
    const ckts = try gpa.alloc(*Circuit, N);
    defer gpa.free(ckts);
    const seeds = try gpa.alloc(Seed, N);
    defer gpa.free(seeds);
    for (periods, ckts, seeds) |l, *c, *s| {
        std.debug.assert(l.ckt.n == n and l.n_samples == steps and !l.ckt.has_state_q);
        c.* = l.ckt;
        s.* = .{ .dt = l.period / @as(f64, @floatFromInt(steps)), .cur = l.x, .q_ring = @splat(l.q_prev), .i_prev = l.i_prev };
        if (!c.*.has_charge) c.*.eval(l.x, 0);
    }
    const dev = try GpuLanes.init(gpa, ckts, seeds, &.{});
    defer dev.deinit();
    @memset(dev.detached, true);
    dev.prof.on = periods[0].ckt.knobs.gpu_stats != .off;
    if (dev.prof.on) std.debug.print("gpu-pss: {d} lanes, {d} fixed steps\n", .{ N, steps });
    const host = try gpa.alloc(f64, (n + 1) * N);
    defer gpa.free(host);
    const nr = try gpa.alloc(LaneNewton, N);
    defer gpa.free(nr);
    @memset(host, 0);
    for (periods, 0..) |l, k| for (l.x[0..n], 0..) |x, i| {
        host[i * N + k] = x;
    };
    try dev.buf(.x).upload(host.ptr, host.len * @sizeOf(f64));
    @memset(dev.round, 1);
    @memset(out, true);
    const records = for (periods) |l| {
        if (l.wave.len != 0) break true;
    } else false;
    for (0..steps) |step| {
        if (std.mem.indexOfScalar(u8, dev.round, 1) == null) break;
        for (periods, dev.round, 0..) |l, on, k| {
            if (on == 0) continue;
            try l.ckt.checkpoint(.{ .phase = .periodic, .completed = step, .total = steps });
            const dt = l.period / @as(f64, @floatFromInt(steps));
            const t = l.period * @as(f64, @floatFromInt(step + 1)) / @as(f64, @floatFromInt(steps));
            l.ckt.setSimState(.{ .t = t, .dt = dt, .kind = .tran });
            dev.sims[k] = l.ckt.sim;
            dev.sc[k] = .{ .t_dev = t, .dt = dt, .xfact = 0, .ag0 = 2.0 / dt, .ag1 = 1, .ag2 = 0, .method = 1, .flags = 1, .sq = @splat(0), .sqt = @splat(0) };
        }
        try dev.put(dev.buf(.sc), dev.sc);
        dev.prof.rounds += 1;
        try dev.newton(opts, nr);
        for (dev.round, nr, out, 0..) |*on, r, *ok, k| {
            if (on.* == 0) continue;
            if (r.reject) return error.FixedStateRejected;
            ok.* = r.result.converged and !r.singular;
            if (!ok.*) {
                dev.ops[k] = 1 + @as(u8, @backingInt(StateCtlOp.revert));
                on.* = 0;
                continue;
            }
            dev.ops[k] = 1 + @as(u8, @backingInt(StateCtlOp.commit));
            dev.accept_[k] = 1;
            lanes_round.queueAccept(dev, k);
        }
        try dev.apply();
        if (records) {
            try dev.buf(.x).download(host.ptr, host.len * @sizeOf(f64));
            for (periods, out, 0..) |l, ok, k| {
                if (!ok or l.wave.len == 0) continue;
                const cols = 1 + l.probes.len;
                const row = l.wave[(step + 1) * cols ..][0..cols];
                row[0] = dev.sc[k].t_dev;
                for (l.probes, 0..) |p, j| row[j + 1] = host[@as(usize, p) * N + k];
            }
        }
    }
    try dev.buf(.x).download(host.ptr, host.len * @sizeOf(f64));
    try dev.buf(.sims).download(dev.sims.ptr, N * @sizeOf(SimState));
    // Stage every accepted payload before changing its host copy. The
    // destructor's best-effort transfer cannot make fixed-period acceptance
    // transactional: a late allocation or download failure must rerun the
    // period with the original host Instance/State rows.
    const Payload = struct { instances: []u8 = &.{}, states: []u8 = &.{} };
    const payloads = try gpa.alloc(Payload, dev.batches.len);
    @memset(payloads, .{});
    defer {
        for (payloads) |p| {
            gpa.free(p.instances);
            gpa.free(p.states);
        }
        gpa.free(payloads);
    }
    for (dev.batches, payloads) |*b, *p| {
        p.instances = try gpa.alloc(u8, b.count * N * b.inst_size);
        p.states = try gpa.alloc(u8, b.count * N * b.state_size);
        if (p.instances.len != 0) try b.d_instances.download(p.instances.ptr, p.instances.len);
        if (p.states.len != 0) try b.d_states.download(p.states.ptr, p.states.len);
    }
    for (dev.batches, payloads, 0..) |b, p, bi| for (periods, 0..) |l, k| {
        const hb = l.ckt.batches[bi];
        const dst = hb.hooks.gpu_payload.?(hb.ctx);
        for (0..b.count) |id| {
            const is = b.inst_size;
            const ss = b.state_size;
            @memcpy(@constCast(dst.instances)[id * is ..][0..is], p.instances[(id * N + k) * is ..][0..is]);
            @memcpy(@constCast(dst.states)[id * ss ..][0..ss], p.states[(id * N + k) * ss ..][0..ss]);
        }
    };
    for (periods, 0..) |l, k| for (l.x[0..n], 0..) |*x, i| {
        x.* = host[i * N + k];
    };
    for (periods, dev.sims) |l, sim| {
        l.ckt.state_staged = false;
        // setSimState preserves iteration, so explicitly publish the final
        // device value for the next host seam or directional integration.
        l.ckt.sim.iteration = sim.iteration;
        l.ckt.setSimState(l.ckt.sim);
    }
    // Host publication is complete; keep the destructor from copying again.
}

// ---- private ----
