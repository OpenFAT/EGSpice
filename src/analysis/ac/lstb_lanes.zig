//! `.lstb` margins for a batch of points in completion-quantum rounds: each
//! point keeps its own frequency solver alive from its sweep through both
//! adaptive margin refinements (`lstb_margins.Refiner`), and every round
//! hands one request per live point to `gpu/freq.zig`'s `RoundDriver`.
//! Failed or cancelled points stop without changing a peer's history.
//! In: run contexts on one pattern. Out: one `lstb.LaneOutcome` per point,
//! bitwise `lstb.run` on that point.
//! Cites dev/analysis/stability.md.
//!
//! Contents:
//! - `pub const LaneStats = struct`
//! - `pub fn runGpuLanes(ctxs: []const RunCtx, opts: Options, backend: Request, out: []LaneOutcome) void`
//! - `pub fn runLanes(ctxs: []const RunCtx, opts: Options, backend: Request, out: []LaneOutcome) Allocator.Error!LaneStats`

const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;
const Complex = @import("core").numerics.Complex;
const FreqSolver = @import("solver").freq_solve.FreqSolver;
const root = @import("../types.zig");
const rounds = @import("../gpu/freq.zig");
const Request = @import("../gpu/context.zig").Request;
const freq = @import("freq.zig");
const lstb = @import("lstb.zig");
const margins_mod = @import("lstb_margins.zig");
const Margins = margins_mod.Margins;
const Refiner = margins_mod.Refiner;
const Options = lstb.Options;
const LaneOutcome = lstb.LaneOutcome;

/// What a `runLanes` batch put on the device, for the `gpu_stats` line.
pub const LaneStats = struct {
    rounds: usize = 0,
    device_lanes: usize = 0,
    device_groups: usize = 0,
    device_peels: usize = 0,
    refinement_requests: usize = 0,
    device_refinement_lanes: usize = 0,
};

/// One point's margins between rounds: its solver, its sweep, and the
/// refiner it is in.
const MarginState = struct {
    ctx: *const root.RunCtx,
    opts: Options,
    fs: FreqSolver,
    sweep: freq.Plan,
    solutions: []f64,
    dynamic: []f64,
    loop: []Complex,
    phase: Phase = .sweep,
    sweep_base: usize = 0,
    sweep_end: usize = 0,
    pending: bool = false,
    omega: [1]f64 = undefined,
    refiner: Refiner = undefined,
    margins: Margins = undefined,

    const Phase = enum { sweep, unity, negative_real, done };

    fn init(ctx: *const root.RunCtx, opts: Options) !MarginState {
        const scratch = ctx.scratch_allocator;
        const ckt = ctx.circuit;
        const p = try lstb.sweepPlan(scratch, opts, ckt.n);
        errdefer p.deinit(scratch);

        var fs = try FreqSolver.fromCircuit(scratch, ckt, ctx.x_op);
        errdefer fs.deinit();

        const solutions = try scratch.alloc(f64, FreqSolver.W * p.rhs.len);
        errdefer scratch.free(solutions);

        const dynamic = try scratch.alloc(f64, 2 * FreqSolver.W * ckt.ac_dyn_slots.len);
        errdefer scratch.free(dynamic);

        const loop = try scratch.alloc(Complex, p.freqs.len);
        return .{ .ctx = ctx, .opts = opts, .fs = fs, .sweep = p, .solutions = solutions, .dynamic = dynamic, .loop = loop };
    }

    fn deinit(self: *MarginState) void {
        const scratch = self.ctx.scratch_allocator;
        self.fs.deinit();
        self.sweep.deinit(scratch);
        scratch.free(self.solutions);
        scratch.free(self.dynamic);
        scratch.free(self.loop);
        self.* = undefined;
    }

    fn request(self: *MarginState, omegas: []const f64) rounds.RoundPoint {
        assert(!self.pending);
        assert(omegas.len <= FreqSolver.W);
        const ckt = self.ctx.circuit;
        const terms = ckt.ac_dyn_slots.len * omegas.len;
        const re = self.dynamic[0..terms];
        const im = self.dynamic[self.dynamic.len / 2 ..][0..terms];
        if (terms != 0) ckt.acDyn(self.ctx.x_op, omegas, re, im);
        self.pending = true;
        return .{ .fs = &self.fs, .omegas = omegas, .dyn = .{ .slots = ckt.ac_dyn_slots, .re = re, .im = im }, .rhs = self.sweep.rhs, .out = self.solutions[0 .. omegas.len * self.sweep.rhs.len], .adjoint = false };
    }

    fn next(self: *MarginState) !?rounds.RoundPoint {
        const m = self.sweep.rhs.len;
        const n = self.ctx.circuit.n;
        if (self.pending) {
            self.pending = false;
            if (self.phase == .sweep) {
                // Stream checkpoints after solving a chunk and before
                // returning any of its rows; keep that exact boundary.
                if (self.sweep_end % freq.quantum == 0 or self.sweep_end == self.sweep.omegas.len)
                    try self.ctx.circuit.checkpoint(.{ .phase = .frequency, .completed = self.sweep_end, .total = self.sweep.omegas.len });
                for (self.sweep_base..self.sweep_end, 0..) |k, local|
                    self.loop[k] = lstb.Gains.of(lstb.responses(self.opts, n, self.solutions[local * m ..][0..m])).w;
                self.sweep_base = self.sweep_end;
            } else self.refiner.accept(lstb.Gains.of(lstb.responses(self.opts, n, self.solutions[0..m])).w);
        }
        // Each pass advances the phase or returns, so at most four run.
        for (0..4) |_| switch (self.phase) {
            .sweep => {
                if (self.sweep_base != self.sweep.omegas.len) {
                    self.sweep_end = self.sweep_base + @min(FreqSolver.W, self.sweep.omegas.len - self.sweep_base);
                    return self.request(self.sweep.omegas[self.sweep_base..self.sweep_end]);
                }
                self.margins = .unbracketed(self.loop[0]);
                self.refiner = Refiner.init(self.sweep.freqs, self.loop, .unity);
                self.phase = .unity;
            },
            .unity, .negative_real => {
                if (self.refiner.next()) |f| {
                    self.omega[0] = 2 * std.math.pi * f;
                    return self.request(&self.omega);
                }
                if (self.phase == .unity) {
                    self.margins.setUnity(self.refiner.best);
                    self.refiner = Refiner.init(self.sweep.freqs, self.loop, .negative_real);
                    self.phase = .negative_real;
                } else {
                    self.margins.setNegativeReal(self.refiner.best);
                    self.phase = .done;
                }
            },
            .done => return null,
        };
        unreachable; // The phases above reach `.done` in four passes.
    }

    fn result(self: *const MarginState) !root.Result {
        assert(self.phase == .done);
        const a = self.ctx.allocator;
        const data = try a.dupe(f64, &self.margins.values());
        errdefer a.free(data);

        return .{
            .plotname = "Loop Stability Margins",
            .varnames = try a.dupe([]const u8, &Margins.names),
            .is_complex = false,
            .npoints = 1,
            .data = data,
        };
    }
};

/// `runLanes`, its own out-of-memory failing every point, then the
/// `gpu_stats` line. Asserts that `out` matches `ctxs`.
pub fn runGpuLanes(ctxs: []const root.RunCtx, opts: Options, backend: Request, out: []LaneOutcome) void {
    assert(out.len == ctxs.len);
    const stats = runLanes(ctxs, opts, backend, out) catch |err| return @memset(out, err);
    if (ctxs.len != 0 and ctxs[0].circuit.knobs.gpu_stats != .off)
        std.debug.print("gpu-lstb: {d} points, {d} rounds, {d} device frequency lanes, {d} groups, {d} peels, {d} refinement requests, {d} device refinement lanes\n", .{ ctxs.len, stats.rounds, stats.device_lanes, stats.device_groups, stats.device_peels, stats.refinement_requests, stats.device_refinement_lanes });
}

/// Runs every point's margins in rounds; `.cpu` runs the rounds on the host,
/// an independent comparison against `lstb.run`. Each live point emits at
/// most one original frequency chunk per round. A query without `margins`
/// runs `lstb.run` per point. Asserts that `out` matches `ctxs`.
pub fn runLanes(ctxs: []const root.RunCtx, opts: Options, backend: Request, out: []LaneOutcome) Allocator.Error!LaneStats {
    assert(ctxs.len == out.len);
    if (ctxs.len == 0) return .{};
    @memset(out, error.QueryCancelled);
    if (!opts.margins) {
        for (ctxs, out) |*ctx, *result| result.* = lstb.run(ctx, opts);
        return .{};
    }
    const gpa = ctxs[0].scratch_allocator;
    const states = try gpa.alloc(?MarginState, ctxs.len);
    defer gpa.free(states);

    @memset(states, null);
    defer for (states) |*state| if (state.*) |*live| live.deinit();
    const requests = try gpa.alloc(rounds.RoundPoint, ctxs.len);
    defer gpa.free(requests);

    const indices = try gpa.alloc(usize, ctxs.len);
    defer gpa.free(indices);

    const failures = try gpa.alloc(?rounds.SolveError, ctxs.len);
    defer gpa.free(failures);

    var driver = rounds.RoundDriver.init(gpa, backend);
    defer driver.deinit();

    for (ctxs, states, out) |*ctx, *state, *result| {
        state.* = MarginState.init(ctx, opts) catch |err| {
            result.* = err;
            continue;
        };
    }
    var stats: LaneStats = .{};
    // Every round finishes at least one solve of each live point, and each
    // point needs a bounded number (its sweep chunks plus two refinements).
    while (true) {
        var count: usize = 0;
        var sweep_pending = false;
        for (states, out, 0..) |*state, *result, index| {
            const live = if (state.*) |*s| s else continue;
            const request = live.next() catch |err| {
                result.* = err;
                live.deinit();
                state.* = null;
                continue;
            };
            if (request) |r| {
                requests[count] = r;
                indices[count] = index;
                count += 1;
                stats.refinement_requests += @intFromBool(live.phase != .sweep);
                sweep_pending = sweep_pending or live.phase == .sweep;
            } else {
                result.* = live.result();
                live.deinit();
                state.* = null;
            }
        }
        if (count == 0) break;
        // Round setup allocates before it touches any solver. If that
        // storage is unavailable, finish these same live requests on the CPU
        // without invalidating a result an earlier round completed.
        const stat = driver.solve(requests[0..count], failures[0..count]) catch
            rounds.solveRound(gpa, .cpu, requests[0..count], failures[0..count]) catch unreachable;
        stats.rounds += 1;
        stats.device_lanes += stat.lanes;
        stats.device_groups += stat.groups;
        stats.device_peels += stat.peels;
        // Counts only rounds whose requests are all refinements, so a sweep
        // can never satisfy the hardware admission gate for the adaptive part.
        if (!sweep_pending) stats.device_refinement_lanes += stat.lanes;
        for (failures[0..count], indices[0..count]) |failure, index| {
            const err = failure orelse continue;
            out[index] = err;
            states[index].?.deinit();
            states[index] = null;
        }
    }
    return stats;
}

// ---- private ----

/// Oracle: `lstb.run` on each point alone, bitwise, under three point orders.
fn marginRoundOracle(backend: Request, require_device: bool) !void {
    const t = std.testing;
    const Builder = @import("builder").Builder;
    const Library = @import("device").Library;
    const models = @import("models");
    const count = 7;
    const gains = [_]f64{ 2, 0.1, 10, 2, std.math.nan(f64), 2, 0.1 };
    const Cancel = struct {
        fn checkpoint(_: *anyopaque, _: @import("../worker.zig").Event) error{QueryCancelled}!void {
            return error.QueryCancelled;
        }
    };
    var cancel_context: u8 = 0;
    var library = try Library.init(t.allocator);
    defer library.deinit();
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    var prepared: [count]@import("device").Circuit = undefined;
    var circuits: [count]root.Circuit = undefined;
    var x: [count][]f64 = undefined;
    var contexts: [count]root.RunCtx = undefined;
    var frozen: usize = 0;
    defer for (prepared[0..frozen]) |*p| p.deinit();
    var instantiated: usize = 0;
    defer for (circuits[0..instantiated]) |*c| c.deinit();
    var built: usize = 0;
    defer for (x[0..built]) |values| t.allocator.free(values);
    var options: Options = undefined;
    for (gains, 0..) |gain, p| {
        var builder = try Builder.init(t.allocator, &library);
        const drive = try builder.addNode();
        const a2 = try builder.addNode();
        const b = try builder.addNode();
        const c = try builder.addNode();
        const d = try builder.addNode();
        const e = try builder.addNode();
        const feedback = try builder.addNode();
        try builder.addDevice(models.vcvs, "", &.{ .gain = gain }, &.{}, .{ a2, root.GROUND, root.GROUND, drive });
        try builder.addDevice(models.resistor, "", &.{ .r = 1e3 }, &.{}, .{ a2, b });
        try builder.addDevice(models.capacitor, "", &.{ .c = 1.5915494309189535e-7 }, &.{}, .{ b, root.GROUND });
        try builder.addDevice(models.vcvs, "", &.{ .gain = 2 }, &.{}, .{ c, root.GROUND, b, root.GROUND });
        try builder.addDevice(models.resistor, "", &.{ .r = 1e3 }, &.{}, .{ c, d });
        try builder.addDevice(models.capacitor, "", &.{ .c = 1.5915494309189535e-7 }, &.{}, .{ d, root.GROUND });
        try builder.addDevice(models.vcvs, "", &.{ .gain = 1 }, &.{}, .{ e, root.GROUND, d, root.GROUND });
        try builder.addDevice(models.resistor, "", &.{ .r = 1e3 }, &.{}, .{ e, feedback });
        try builder.addDevice(models.capacitor, "", &.{ .c = 1.5915494309189535e-7 }, &.{}, .{ feedback, root.GROUND });
        const branch = builder.n;
        try builder.addDevice(models.vsource, "", &.{ .dc = 0 }, &.{}, .{ drive, feedback });
        // The independent passive rows keep this oracle above the dense
        // crossover so its hardware half must execute sparse replay.
        for (0..12) |_| {
            const extra = try builder.addNode();
            try builder.addDevice(models.resistor, "", &.{ .r = 700 }, &.{}, .{ extra, root.GROUND });
            try builder.addDevice(models.capacitor, "", &.{ .c = 2e-10 }, &.{}, .{ extra, root.GROUND });
        }
        var permutation: ?[]const u32 = null;
        prepared[p] = try builder.compilePerm(&permutation);
        frozen += 1;
        defer if (permutation) |map| t.allocator.free(map);
        circuits[p] = try root.Circuit.instantiate(&prepared[p], t.allocator);
        instantiated += 1;
        x[p] = try t.allocator.alloc(f64, circuits[p].n);
        built += 1;
        @memset(x[p], 0);
        circuits[p].setSimState(.{ .kind = .ac });
        if (p == 5) circuits[p].progress = .{ .ctx = &cancel_context, .yield_fn = Cancel.checkpoint };
        contexts[p] = .{
            .circuit = &circuits[p],
            .x_op = x[p],
            .probes = &.{},
            .source_node = root.GROUND,
            .source_branch = root.GROUND,
            .allocator = arena.allocator(),
            .scratch_allocator = t.allocator,
        };
        const opts: Options = .{
            .sweep = .{ .f_start = 10, .f_stop = 1e5, .points = 17 },
            .probes = @splat(.{
                .p = if (permutation) |map| map[drive] else drive,
                .n = if (permutation) |map| map[feedback] else feedback,
                .branch = if (permutation) |map| map[branch] else branch,
            }),
            .margins = true,
        };
        if (p == 0) options = opts else {
            try t.expectEqual(options.probes[0], opts.probes[0]);
            try t.expect(circuits[p].n > 16);
        }
    }
    var expected: [count]LaneOutcome = undefined;
    for (&contexts, &expected) |*ctx, *result| result.* = lstb.run(ctx, options);
    try t.expectError(error.QueryCancelled, expected[5]);
    var failures: usize = 0;
    for (expected) |result| if (result) |_| {} else |_| {
        failures += 1;
    };
    try t.expect(failures >= 2 and failures < count);
    const orders = [_][count]usize{
        .{ 0, 1, 2, 3, 4, 5, 6 },
        .{ 6, 5, 4, 3, 2, 1, 0 },
        .{ 4, 0, 5, 2, 6, 3, 1 },
    };
    for (orders) |order| {
        var ordered: [count]root.RunCtx = undefined;
        for (&ordered, order) |*ctx, p| ctx.* = contexts[p];
        var actual: [count]LaneOutcome = undefined;
        const stats = try runLanes(&ordered, options, backend, &actual);
        try t.expect(stats.rounds > 1 and stats.refinement_requests > count);
        if (require_device) {
            try t.expect(stats.device_groups > 0);
            try t.expect(stats.device_refinement_lanes > 0);
        }
        for (actual, order) |result, p| {
            if (expected[p]) |oracle| {
                const got = try result;
                try t.expectEqualStrings(oracle.plotname, got.plotname);
                try t.expectEqual(oracle.npoints, got.npoints);
                try t.expectEqual(oracle.data.len, got.data.len);
                for (oracle.data, got.data) |want, value| {
                    if (std.math.isNan(want)) {
                        try t.expect(std.math.isNan(value));
                    } else try t.expectEqual(@as(u64, @bitCast(want)), @as(u64, @bitCast(value)));
                }
            } else |err| try t.expectError(err, result);
        }
    }
}

test "live LSTB margin rounds preserve scalar refinements, failures and cancellation in every order" {
    try marginRoundOracle(.cpu, false);
}

test "GPU LSTB margin rounds execute both sweep and adaptive refinement" {
    const images = @import("gompute_kernels");
    if (comptime images.has_cuda) return marginRoundOracle(.cuda, true);
    if (comptime images.has_hip) return marginRoundOracle(.hip, true);
    return error.SkipZigTest;
}
