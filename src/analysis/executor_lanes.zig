//! An executor's lane batches: a query and its `peers` built, then run
//! together through each analysis's lane driver (operating points, `.dc`,
//! `.mc`/`.temp` rows, transients, shooting `.pss`, the AC family), on the
//! host W wide or as GPU lanes in waves; each member publishes as it would
//! alone. `Executor` calls these as its own methods.
//! In: the batch's executors. Out: every member's outcome.
//! Cites dev/research/lane-batched-simulation.md §6;
//! dev/research/plans/phase5-ac-lanes.md; #213.
//!
//! Contents:
//! - `pub fn executeBatch(self: *Executor) QueryError!Result`
//! - `pub fn executeGpuBatch(self: *Executor) QueryError!Result`
//! - `pub fn runGpuWave(self: *Executor, members, outcomes, live) QueryError!void`
//! - `pub fn runGpuOps(self: *Executor, members, live, outcomes) QueryError!void`
//! - `pub fn executeGpuFreqBatch(self: *Executor) QueryError!Result`
//! - `pub fn precompute(self: *Executor, ctx: *const RunCtx) ![]const f64`
//! - `pub fn frequencyPlan(self: *Executor, ctx: *const RunCtx) !?freq.Plan`
//! - `pub fn gpuMembers(self: *Executor) !struct { []*Executor, []QueryError!Result }`

const std = @import("std");
const assert = std.debug.assert;
const Circuit = @import("device").Circuit;
const types = @import("types.zig");
const op = @import("dc/op.zig");
const op_lanes = @import("dc/op_lanes.zig");
const sweep_lanes = @import("sweep/lanes.zig");
const tran = @import("tran/tran.zig");
const exec_config = @import("exec_config.zig");
const dispatch = @import("dispatch.zig");
const executor = @import("executor.zig");
const Executor = executor.Executor;
const QueryError = executor.QueryError;
const lane_width = exec_config.lane_width;
const freqLaneKind = exec_config.freqLaneKind;
const gpuFreqLaneKind = exec_config.gpuFreqLaneKind;
const module = dispatch.module;
const run = dispatch.run;
const simKind = dispatch.simKind;

/// The host bytes a GPU batch's wave may hold (`gpuLaneBytes` each).
// ponytail: a fixed 4 GiB; size it from the machine's memory if decks
// with tens of thousands of large lanes need more per wave.
const gpu_wave_bytes: usize = 4 << 30;
/// Fewest lanes per wave, so a huge circuit still batches.
const gpu_wave_min: usize = 256;

/// Runs this query and its `peers` as one lane batch: each builds its
/// circuit, then the operating points (`op.solveLanes`), `.dc` sweeps
/// (`dc.runLanes`) or transients (`tran.runLanes`) run as lanes and each publishes as it would alone. A
/// batch whose devices print display tasks runs its queries one after
/// another instead. Every peer ends with `published` or `peer_failure`;
/// returns this query's own.
pub fn executeBatch(self: *Executor) QueryError!types.Result {
    if (self.config.gpu_lanes and (self.job == .tran or self.job == .pss or self.job == .hb or
        (self.job == .lstb and self.job.lstb.margins) or (self.config.gpu_static and (self.job == .op or self.job == .dc)))) return self.executeGpuBatch();
    if (self.config.gpu_lanes and gpuFreqLaneKind(self.job)) return self.executeGpuFreqBatch();
    const g = 1 + self.peers.len;
    assert(g <= lane_width);
    assert(self.job == .op or self.job == .dc or self.job == .mc or self.job == .temp or self.job == .tran or self.job == .pss or freqLaneKind(self.job));
    var members: [lane_width]*Executor = undefined;
    members[0] = self;
    @memcpy(members[1..g], self.peers);
    var outcomes: [lane_width]QueryError!types.Result = @splat(error.QueryCancelled);
    defer publishPeers(members[0..g], outcomes[0..g]);
    // Lanes whose circuit built, as indices into `members`.
    var live: [lane_width]usize = undefined;
    var n_live: usize = 0;
    for (members[0..g], 0..) |e, k| {
        e.materialize() catch |err| {
            outcomes[k] = err;
            continue;
        };
        e.circuit.progress = self.controller.callback();
        live[n_live] = k;
        n_live += 1;
    }
    if (n_live != 0 and (try members[live[0]].sayNames()).len != 0) {
        for (live[0..n_live]) |k| outcomes[k] = members[k].finishAlone();
        return outcomes[0];
    }
    if (freqLaneKind(self.job)) {
        runFreqLanes(members[0..g], live[0..n_live], outcomes[0..g]);
        return outcomes[0];
    }
    switch (self.job) {
        // Every laned analysis past the OP but `.tran` (its `uic` start,
        // below): each lane runs from its own circuit and context, and
        // publishes as it would alone.
        inline .dc, .pss => |opts, tag| {
            const M = module(tag);
            var ctxs: [lane_width]types.RunCtx = undefined;
            laneCtxs(members[0..g], live[0..n_live], &ctxs);
            var swept: [lane_width]M.LaneOutcome = undefined;
            if (n_live != 0) M.runLanes(lane_width, ctxs[0..n_live], opts, swept[0..n_live]);
            for (live[0..n_live], swept[0..n_live]) |k, o|
                outcomes[k] = members[k].settleLabeled(o);
            return outcomes[0];
        },
        // `.mc`/`.temp` lane rows (#213), each its own lane (`Session.gather`).
        inline .mc, .temp => |_, tag| {
            const M = module(tag);
            var ctxs: [lane_width]types.RunCtx = undefined;
            laneCtxs(members[0..g], live[0..n_live], &ctxs);
            var opts: [lane_width]M.Options = undefined;
            for (live[0..n_live], opts[0..n_live]) |k, *o| o.* = @field(members[k].job, @tagName(tag));
            var swept: [lane_width]sweep_lanes.LaneOutcome = undefined;
            if (n_live != 0) sweep_lanes.runBatch(lane_width, M, ctxs[0..n_live], opts[0..n_live], swept[0..n_live]);
            for (live[0..n_live], swept[0..n_live]) |k, o|
                outcomes[k] = members[k].settleLabeled(o);
            return outcomes[0];
        },
        else => {},
    }
    if (self.job == .tran) {
        // `runQuery`'s transient start: the `uic` x when no operating
        // point ran.
        for (live[0..n_live]) |k| if (members[k].job.uic()) members[k].seedUic() catch |err| {
            for (live[0..n_live]) |j| outcomes[j] = err;
            return outcomes[0];
        };
        var ctxs: [lane_width]types.RunCtx = undefined;
        laneCtxs(members[0..g], live[0..n_live], &ctxs);
        var ran: [lane_width]tran.LaneOutcome = undefined;
        if (n_live != 0) tran.runLanes(lane_width, ctxs[0..n_live], self.job.tran, ran[0..n_live]);
        for (live[0..n_live], ran[0..n_live]) |k, o|
            outcomes[k] = members[k].settleLabeled(o);
        return outcomes[0];
    }
    var ckts: [lane_width]*types.Circuit = undefined;
    var xs: [lane_width][]f64 = undefined;
    var options: op.Options = undefined;
    for (live[0..n_live], 0..) |k, l| {
        const e = members[k];
        e.circuit.setSimState(.{ .kind = simKind(e.job) });
        options = e.opStart() catch |err| {
            for (live[0..n_live]) |j| outcomes[j] = err;
            return outcomes[0];
        };
        ckts[l] = &e.circuit;
        xs[l] = e.x.?;
    }
    var solved: [lane_width]op.LaneOutcome = undefined;
    if (n_live != 0) op_lanes.solveLanes(lane_width, ckts[0..n_live], xs[0..n_live], options, self.deck.nodeset, self.deck.ic, solved[0..n_live]) catch |err| {
        for (live[0..n_live]) |k| outcomes[k] = err;
        return outcomes[0];
    };
    for (live[0..n_live], solved[0..n_live]) |k, o| {
        const e = members[k];
        outcomes[k] = e.settle(blk: {
            const r = o catch |err| break :blk err;
            if (!r.converged) break :blk error.OpDidNotConverge;
            e.circuit.setSimState(.{ .kind = simKind(e.job) });
            break :blk e.finish();
        });
    }
    return outcomes[0];
}

/// `executeBatch` for a transient, periodic or live LSTB GPU batch: `peers` may
/// be thousands wide, so the per-member arrays come from the work arena.
pub fn executeGpuBatch(self: *Executor) QueryError!types.Result {
    const members, const outcomes = try self.gpuMembers();
    defer publishPeers(members, outcomes);
    const g = members.len;
    // Waves bound the host: each member's circuit, run and solver state
    // exist only for its wave, and one nothing reads afterwards is freed
    // with it. One wave unless the lanes would pass `gpu_wave_bytes`.
    const per_lane = gpuLaneBytes(self.topology);
    const wave = @max(gpu_wave_min, @min(g, gpu_wave_bytes / per_lane));
    const live = try self.work.allocator().alloc(usize, @min(g, wave));
    var first: usize = 0;
    while (first < g) : (first += wave) {
        const end = @min(g, first + wave);
        try self.runGpuWave(members[first..end], outcomes[first..end], live);
        for (members[first..end]) |e| if (e.no_readers and e.built) e.circuit.release();
    }
    return outcomes[0];
}

/// One wave of `executeGpuBatch`: materialize its members, run their
/// lanes, label their outcomes and check their latches.
pub fn runGpuWave(self: *Executor, members: []const *Executor, outcomes: []QueryError!types.Result, live: []usize) QueryError!void {
    const a = self.work.allocator();
    var n_live: usize = 0;
    // ponytail: serial materialize; thousands of lanes take seconds here.
    // Parallelize over a thread pool if it shows above 5% of the batch.
    const t0 = @import("gpu/lanes.zig").Prof.now();
    for (members, 0..) |e, k| {
        e.materialize() catch |err| {
            outcomes[k] = err;
            continue;
        };
        // UIC omits the operating point; its initial vector still
        // has to exist before the GPU admission or host fallback
        // receives a RunCtx, just as in runQuery and CPU lane batches.
        if (e.job.uic()) e.seedUic() catch |err| {
            outcomes[k] = err;
            continue;
        };
        e.circuit.progress = self.controller.callback();
        live[n_live] = k;
        n_live += 1;
    }
    if (self.config.knobs.gpu_stats != .off) std.debug.print("gpu-lanes: materialize {d} lanes {d:.1} ms\n", .{ members.len, @as(f64, @floatFromInt(@import("gpu/lanes.zig").Prof.now() - t0)) / 1e6 });
    if (self.job == .op) return self.runGpuOps(members, live[0..n_live], outcomes);
    const ctxs = try a.alloc(types.RunCtx, n_live);
    laneCtxs(members, live[0..n_live], ctxs);
    switch (self.job) {
        inline .tran, .pss, .hb, .lstb, .dc => |opts, tag| {
            const M = module(tag);
            const ran = try a.alloc(M.LaneOutcome, n_live);
            if (n_live != 0) {
                if (comptime tag == .lstb)
                    M.runGpuLanes(ctxs, opts, self.config.backend, ran)
                else if (comptime tag == .tran)
                    // Every usable device takes a slice (multi-GPU stage 1).
                    @import("gpu/multi.zig").runLanes(M, self.allocator, self.config.threads, ctxs, opts, ran)
                else
                    M.runGpuLanes(ctxs, opts, ran);
            }
            for (live[0..n_live], ran) |k, o|
                outcomes[k] = members[k].settleLabeled(o);
        },
        else => unreachable,
    }
}

/// `runGpuWave` for operating points: each live member's start
/// (`opStart`), `op.solveGpuLanes` over all of them, then each ends as
/// `executeBatch`'s operating points do.
pub fn runGpuOps(self: *Executor, members: []const *Executor, live: []const usize, outcomes: []QueryError!types.Result) QueryError!void {
    const a = self.work.allocator();
    const ckts = try a.alloc(*types.Circuit, live.len);
    const xs = try a.alloc([]f64, live.len);
    var options: op.Options = undefined;
    for (live, ckts, xs) |k, *c, *x| {
        const e = members[k];
        e.circuit.setSimState(.{ .kind = simKind(e.job) });
        options = e.opStart() catch |err| {
            for (live) |j| outcomes[j] = err;
            return;
        };
        c.* = &e.circuit;
        x.* = e.x.?;
    }
    const solved = try a.alloc(op.LaneOutcome, live.len);
    if (live.len != 0) op_lanes.solveGpuLanes(lane_width, ckts, xs, options, self.deck.nodeset, self.deck.ic, solved) catch |err| {
        for (live) |k| outcomes[k] = err;
        return;
    };
    for (live, solved) |k, o| {
        const e = members[k];
        outcomes[k] = e.settle(blk: {
            const r = o catch |err| break :blk err;
            if (!r.converged) break :blk error.OpDidNotConverge;
            break :blk e.finish();
        });
    }
}

/// The leading AC-family sweeps across a wide GPU batch. Every query
/// keeps its own solver/plan and reuses its ordinary readout. A point
/// whose setup or precompute fails runs live, preserving its numerical
/// failure, device-backend policy and partial streaming behavior.
pub fn executeGpuFreqBatch(self: *Executor) QueryError!types.Result {
    const frequency = @import("ac/freq.zig");
    const provider = @import("gpu/freq.zig");
    const FS = @import("solver").freq_solve.FreqSolver;
    const a = self.work.allocator();
    const members, const outcomes = try self.gpuMembers();
    defer publishPeers(members, outcomes);
    const g = members.len;
    const ctxs = try a.alloc(?types.RunCtx, g);
    @memset(ctxs, null);
    const plans = try a.alloc(?frequency.Plan, g);
    @memset(plans, null);
    const solvers = try a.alloc(?FS, g);
    @memset(solvers, null);
    defer for (plans, solvers, ctxs) |p, *fs, ctx| {
        if (fs.*) |*f| f.deinit();
        if (p) |plan| plan.deinit(ctx.?.scratch_allocator);
    };
    const points = try a.alloc(provider.Point, g);
    const point_ids = try a.alloc(usize, g);
    var n: usize = 0;
    var pre_bytes: usize = 0;
    for (members, 0..) |e, k| {
        e.materialize() catch |err| {
            outcomes[k] = err;
            continue;
        };
        e.circuit.progress = self.controller.callback();
        e.circuit.setSimState(.{ .kind = simKind(e.job) });
        ctxs[k] = e.runCtx();
        // Stateful display tasks must run only through their ordinary
        // analysis; the precompute would evaluate them an extra time.
        const display_names = e.sayNames() catch continue;
        if (display_names.len != 0) continue;
        plans[k] = e.frequencyPlan(&ctxs[k].?) catch continue;
        const plan = plans[k] orelse continue;
        const rhs_bytes = std.math.mul(usize, plan.rhs.len, @sizeOf(f64)) catch continue;
        const bytes = std.math.mul(usize, plan.omegas.len, rhs_bytes) catch continue;
        // Bound the extra precompute storage of a wide group. Large
        // individual sweeps keep their bounded Stream working set.
        if (bytes > 256 * 1024 * 1024 - pre_bytes) continue;
        solvers[k] = FS.fromCircuit(ctxs[k].?.scratch_allocator, &e.circuit, ctxs[k].?.x_op) catch continue;
        plan.configure(&solvers[k].?);
        const out = e.work.allocator().alloc(f64, bytes / @sizeOf(f64)) catch continue;
        points[n] = .{
            .fs = &solvers[k].?,
            .circuit = &e.circuit,
            .x_op = ctxs[k].?.x_op,
            .omegas = plan.omegas,
            .rhs = plan.rhs,
            .out = out,
            .adjoint = plan.adjoint,
        };
        point_ids[n] = k;
        n += 1;
        pre_bytes += bytes;
    }
    const computed = blk: {
        const stats = provider.solveAll(a, self.config.backend, points[0..n]) catch |err| {
            if (self.config.knobs.gpu_stats != .off)
                std.debug.print("gpu-freq: precompute declined ({s}); running queries live\n", .{@errorName(err)});
            break :blk false;
        };
        if (self.config.knobs.gpu_stats != .off and stats.lanes != 0)
            std.debug.print("gpu-freq: {d} points, {d} frequency lanes, {d} groups, {d} peels\n", .{ stats.points, stats.lanes, stats.groups, stats.peels });
        break :blk true;
    };
    if (computed) {
        for (points[0..n], point_ids[0..n]) |p, k| ctxs[k].?.freq_pre = p.out;
    }
    for (members, ctxs, 0..) |e, *ctx, k| {
        if (ctx.*) |*c| {
            if (c.freq_pre.len == 0) {
                outcomes[k] = e.finishAlone();
                continue;
            }
            outcomes[k] = e.settleLabeled(run(c, e.job));
        }
    }
    return outcomes[0];
}

/// The query's leading frequency sweep, solved as its own `run` would
/// solve it (`freq.solveAll` over its `plan`), in the query's work
/// arena; empty for a kind that solves live.
pub fn precompute(self: *Executor, ctx: *const types.RunCtx) ![]const f64 {
    const freq = @import("ac/freq.zig");
    const scratch = ctx.scratch_allocator;
    const p = (try self.frequencyPlan(ctx)) orelse return &.{};
    defer p.deinit(scratch);
    var fs = try @import("solver").freq_solve.FreqSolver.fromCircuit(scratch, ctx.circuit, ctx.x_op);
    defer fs.deinit();
    p.configure(&fs);
    const out = try self.work.allocator().alloc(f64, p.omegas.len * p.rhs.len);
    try freq.solveAll(scratch, &fs, ctx.circuit, ctx.x_op, p.omegas, p.rhs, p.adjoint, out);
    return out;
}

pub fn frequencyPlan(self: *Executor, ctx: *const types.RunCtx) !?@import("ac/freq.zig").Plan {
    return switch (self.job) {
        .ac => |o| try @import("ac/ac.zig").plan(ctx, o),
        .acxf => |o| try @import("ac/xf.zig").plan(ctx, o),
        .stb => |o| try @import("ac/stb.zig").plan(ctx, o),
        .lstb => |o| try @import("ac/lstb.zig").plan(ctx, o),
        .sp => |o| try @import("ac/sp.zig").plan(ctx, o),
        .noise => |o| try @import("ac/noise.zig").plan(ctx, o),
        else => null,
    };
}

/// This query and its `peers` in the work arena, this one first, each
/// outcome a cancel until its lane settles it; `publishPeers` hands the
/// peers theirs. A GPU batch may be thousands wide.
pub fn gpuMembers(self: *Executor) !struct { []*Executor, []QueryError!types.Result } {
    const a = self.work.allocator();
    const members = try a.alloc(*Executor, 1 + self.peers.len);
    members[0] = self;
    @memcpy(members[1..], self.peers);
    const outcomes = try a.alloc(QueryError!types.Result, members.len);
    @memset(outcomes, error.QueryCancelled);
    return .{ members, outcomes };
}

// ---- private ----

/// Host bytes one GPU lane holds while its wave runs, estimated from the
/// circuit's size: its materialized circuit, its run and step machine (about
/// 20 n-vectors and the charge rings) and its solver's factors. Measured at
/// about 2.3 KB per unknown on a 72-unknown deck; this errs high.
fn gpuLaneBytes(topology: *const Circuit) usize {
    return 4096 * (@as(usize, topology.n) + @as(usize, topology.nnz) / 4) + 64 * 1024;
}

/// `executeBatch`'s AC-family branch: every live lane's leading
/// frequency sweep solved up front (`precompute`), then each lane's
/// analysis replaying it (`RunCtx.freq_pre`) and labelled as alone. The
/// two passes are the GPU lane backend's shape: it replaces the first
/// with (point, ω) lane rounds (dev/research/plans/phase5-ac-lanes.md).
/// A lane whose precompute fails runs live, so it fails or streams as
/// it would alone.
// ponytail: holds points·ω·rhs values per lane until its run ends;
// chunk the precompute if a long sweep on a large n needs it.
fn runFreqLanes(members: []const *Executor, live: []const usize, outcomes: []QueryError!types.Result) void {
    var ctxs: [lane_width]types.RunCtx = undefined;
    laneCtxs(members, live, &ctxs);
    for (live, ctxs[0..live.len]) |k, *c| c.freq_pre = members[k].precompute(c) catch &.{};
    for (live, 0..) |k, l| {
        const e = members[k];
        outcomes[k] = e.settleLabeled(run(&ctxs[l], e.job));
    }
}

/// Each live member's `RunCtx`, into `ctxs[0..live.len]`, its sim state set
/// as `runQuery` sets it (`simKind`).
fn laneCtxs(members: []const *Executor, live: []const usize, ctxs: []types.RunCtx) void {
    for (live, ctxs[0..live.len]) |k, *c| {
        members[k].circuit.setSimState(.{ .kind = simKind(members[k].job) });
        c.* = members[k].runCtx();
    }
}

/// Hands each peer, `members[1..]`, its lane's outcome.
fn publishPeers(members: []const *Executor, outcomes: []const QueryError!types.Result) void {
    for (members[1..], outcomes[1..]) |p, o| {
        if (o) |r| p.published = r else |err| p.peer_failure = err;
    }
}
