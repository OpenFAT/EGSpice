//! DC sweep (`.dc`): steps one parameter (a source value, a device
//! parameter or the temperature) through its ParamRef, one warm-started
//! Newton per point, and records the probes; `Sweep.zig` owns the state
//! between steps. Three drivers share it and differ only in where the warm
//! Newton runs: one circuit, W host lanes, or every lane on the GPU.
//! In: run contexts and `.dc` options. Out: one Result per context.
//! Cites ngspice dctrcurv.c; dev/research/lane-batched-simulation.md §6 (the
//! `.dc` row); 06 6.9 (GPU lanes).
//!
//! Contents:
//! - `pub const Options = core.query.Dc`
//! - `pub const LaneOutcome = LaneError!Result`
//! - `pub fn run(ctx: *const RunCtx, opts: Options) !Result`
//! - `pub fn runLanes(comptime W: usize, ctxs: []const RunCtx, opts: Options, out: []LaneOutcome) void`
//! - `pub fn runGpuLanes(ctxs: []const RunCtx, opts: Options, out: []LaneOutcome) void`

const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;
const solver = @import("solver");
const converger = solver.converger;
const root = @import("../types.zig");
const gpu_lanes = @import("../gpu/lanes.zig");
const op = @import("op.zig");
const Sweep = @import("Sweep.zig");

/// Query options, defined in core/query.zig.
pub const Options = @import("core").query.Dc;

/// One lane's `run` outcome.
pub const LaneOutcome = LaneError!root.Result;

const LaneError = root.ErrorOf(run);

/// `runGpuLanes`' per-lane tables, one allocation each, freed together.
const GpuTables = struct {
    sweeps: []Sweep,
    ckts: []*root.Circuit,
    xs: [][]f64,
    res: []gpu_lanes.LaneNewton,
    /// Still sweeping.
    live: []bool,
    /// Runs a warm Newton this point.
    warm: []bool,

    fn init(gpa: Allocator, g: usize) Allocator.Error!GpuTables {
        var t: GpuTables = .{ .sweeps = &.{}, .ckts = &.{}, .xs = &.{}, .res = &.{}, .live = &.{}, .warm = &.{} };
        errdefer t.deinit(gpa);
        t.sweeps = try gpa.alloc(Sweep, g);
        t.ckts = try gpa.alloc(*root.Circuit, g);
        t.xs = try gpa.alloc([]f64, g);
        t.res = try gpa.alloc(gpu_lanes.LaneNewton, g);
        t.live = try gpa.alloc(bool, g);
        t.warm = try gpa.alloc(bool, g);
        return t;
    }

    fn deinit(t: GpuTables, gpa: Allocator) void {
        gpa.free(t.sweeps);
        gpa.free(t.ckts);
        gpa.free(t.xs);
        gpa.free(t.res);
        gpa.free(t.live);
        gpa.free(t.warm);
    }
};

/// Contract entry: real, point-major (sweep value, probes...), outer blocks
/// concatenated, then one more such block per `opts.tangents` row. A point
/// that does not converge records NaN probes, in every block. The swept
/// values are restored afterwards so the cached operating point stays valid
/// for later jobs. Fails with error.DcSweepSourceNotFound when a target
/// names no parameter.
pub fn run(ctx: *const root.RunCtx, opts: Options) !root.Result {
    var s: Sweep = undefined;
    try s.init(ctx, opts);
    defer s.deinit();

    for (0..s.n_outer) |po| {
        try s.beginBlock(po);
        for (0..s.n_inner) |pt| {
            var converged = false;
            if (try s.beginPoint(pt)) {
                s.prepareWarm();
                converged = try warmNewton(&s);
            }
            try s.endPoint(pt, converged);
        }
    }
    return s.take();
}

/// `run` for the lanes of one batch: `.dc` queries on one pattern that sweep
/// the same values, each from its own operating point and variant row. The
/// sweep axis stays sequential and every lane keeps its own predictor and
/// warm start; each point's warm Newton runs as lanes, and a lane it leaves
/// unconverged takes its own cold ladder, as `run` does. Lane l's result
/// lands in `out[l]`, bitwise `run` on `ctxs[l]`. Under a JFNK pin or
/// `-Dcertify` every lane runs `run`. Asserts 1 <= `ctxs.len` <= W and that
/// `out` matches it.
pub fn runLanes(comptime W: usize, ctxs: []const root.RunCtx, opts: Options, out: []LaneOutcome) void {
    const g = ctxs.len;
    assert(g >= 1);
    assert(g <= W);
    assert(out.len == g);
    if (W == 1 or g == 1 or !op.lanesAllowed(ctxs[0].circuit)) {
        for (ctxs, out) |*c, *o| o.* = run(c, opts);
        return;
    }
    const L = solver.lanes.Lanes(W);
    const System = solver.lane_system.LaneSystem(W, root.Circuit, root.EvalHook);
    var sweeps: [W]Sweep = undefined;
    var live: L.Mask = 0;
    for (ctxs, 0..) |*c, l| {
        sweeps[l].init(c, opts) catch |e| {
            out[l] = e;
            continue;
        };
        live |= bit(L, l);
    }
    const batch: Lanes(L) = .{ .sweeps = &sweeps, .live = &live, .out = out };
    defer batch.endAll();

    if (live == 0) return;
    const first = @ctz(live);
    var lanes: [W]*root.Circuit = undefined;
    var wss: [W]*converger.Workspace = undefined;
    for (&lanes, &wss, 0..) |*c, *w, l| {
        const k = if (l < g and live & bit(L, l) != 0) l else first;
        c.* = ctxs[k].circuit;
        w.* = c.*.workspace() catch |e| return batch.fail(e);
    }
    const gpa = ctxs[first].scratch_allocator;
    var sys = System.init(gpa, lanes, .{}) catch |e| return batch.fail(e);
    defer sys.deinit(gpa);

    var lws = converger.LaneWorkspace(W).init(gpa, wss) catch |e| return batch.fail(e);
    defer lws.deinit(gpa);

    const xv = gpa.alloc(L.V, lanes[0].n) catch |e| return batch.fail(e);
    defer gpa.free(xv);

    for (0..sweeps[first].n_outer) |po| {
        var it = L.each(live);
        while (it.next()) |l| sweeps[l].beginBlock(po) catch |e| batch.kill(l, e);
        for (0..sweeps[first].n_inner) |pt| {
            var warm: L.Mask = 0;
            it = L.each(live);
            while (it.next()) |l| {
                const w = sweeps[l].beginPoint(pt) catch |e| {
                    batch.kill(l, e);
                    continue;
                };
                if (!w) continue;
                sweeps[l].prepareWarm();
                System.putLane(xv, sweeps[l].sol(), l);
                warm |= bit(L, l);
            }
            var converged: L.Mask = 0;
            if (warm != 0) {
                sys.active = warm;
                const res = converger.newtonLanes(W, &sys, &lws, xv, @splat(0), sweeps[@ctz(warm)].warmOptions(), System.LaneHook{});
                it = L.each(warm);
                // As `converger.run` ends each lane's solve.
                while (it.next()) |l| lanes[l].clearLimits();
                const r = res catch |e| return batch.fail(e);
                converged = r.converged;
                it = L.each(warm);
                while (it.next()) |l| System.getLane(sweeps[l].sol(), xv, l);
            }
            it = L.each(live);
            while (it.next()) |l| sweeps[l].endPoint(pt, converged & bit(L, l) != 0) catch |e| batch.kill(l, e);
        }
    }
    var it = L.each(live);
    while (it.next()) |l| batch.finish(l);
    assert(live == 0);
}

/// `run` for any number of lanes with each point's warm Newton for every
/// lane on the GPU (`gpu_lanes.GpuLanes.solveStatic`): the swept value,
/// recompute, predictor, cold ladder, commit and rows stay on the host, as
/// `run` does them, and each point moves the lanes' rows to the device and
/// the accepted lanes' device state back. Within tolerance of `run`, not
/// bitwise (one LU pivot order). A batch the device cannot take runs each
/// lane's `run`; a device error part way runs the rest of the warm Newtons
/// on the host. Asserts that `out` matches `ctxs`.
pub fn runGpuLanes(ctxs: []const root.RunCtx, opts: Options, out: []LaneOutcome) void {
    const g = ctxs.len;
    assert(out.len == g);
    if (g == 0) return;
    const c0 = ctxs[0].circuit;
    const why: ?[]const u8 = if (g < 2) "one lane" else if (!op.lanesAllowed(c0)) "a JFNK pin or certify" else gpu_lanes.staticRefusal(c0);
    if (why) |w| {
        if (c0.knobs.gpu_stats != .off) std.debug.print("gpu-lanes: dc lanes on the CPU: {s}\n", .{w});
        for (ctxs, out) |*c, *o| o.* = run(c, opts);
        return;
    }
    const gpa = ctxs[0].scratch_allocator;
    const t = GpuTables.init(gpa, g) catch |e| return @memset(out, e);
    defer t.deinit(gpa);

    for (t.sweeps, ctxs, 0..) |*s, *c, l| {
        s.init(c, opts) catch |e| {
            // ponytail: a lane that fails its setup sends the batch to `run`.
            for (t.sweeps[0..l]) |*d| d.deinit();
            out[l] = e;
            for (ctxs, out, 0..) |*cc, *o, k| if (k != l) {
                o.* = run(cc, opts);
            };
            return;
        };
        t.ckts[l] = c.circuit;
        t.xs[l] = s.sol();
    }
    @memset(t.live, true);
    defer for (t.sweeps, t.live) |*s, on| if (on) s.deinit();

    var dev: ?*gpu_lanes.GpuLanes = null;
    // The host already holds every lane's state (`copyBack` each point).
    defer if (dev) |d| {
        @memset(d.detached, true);
        d.deinit();
    };
    var on_device = true;
    for (0..t.sweeps[0].n_outer) |po| {
        for (t.sweeps, t.live, out) |*s, *on, *o| if (on.*) s.beginBlock(po) catch |e| drop(s, on, o, e);
        for (0..t.sweeps[0].n_inner) |pt| {
            var any = false;
            for (t.sweeps, t.live, t.warm, out) |*s, *on, *w, *o| {
                w.* = false;
                if (!on.*) continue;
                w.* = s.beginPoint(pt) catch |e| {
                    drop(s, on, o, e);
                    continue;
                };
                if (!w.*) continue;
                s.prepareWarm();
                any = true;
            }
            if (any and on_device) devicePoint(gpa, &dev, t) catch |e| {
                if (c0.knobs.gpu_stats != .off) std.debug.print("gpu-lanes: dc lanes leave the device ({s})\n", .{@errorName(e)});
                on_device = false;
            };
            for (t.sweeps, t.live, t.warm, t.res, out) |*s, *on, w, r, *o| {
                if (!on.*) continue;
                var converged = false;
                if (w and on_device) {
                    converged = r.result.converged and !r.singular and !r.reject;
                    // As `converger.run` ends each lane's solve.
                    s.ckt.clearLimits();
                } else if (w) {
                    converged = warmNewton(s) catch |e| {
                        drop(s, on, o, e);
                        continue;
                    };
                }
                s.endPoint(pt, converged) catch |e| drop(s, on, o, e);
            }
        }
    }
    for (t.sweeps, t.live, out) |*s, *on, *o| if (on.*) {
        o.* = s.take();
        s.deinit();
        on.* = false;
    };
}

// ---- private ----

/// `runLanes`' lane bookkeeping: a lane leaves `live` once, with its
/// outcome written and its sweep freed.
fn Lanes(comptime L: type) type {
    return struct {
        sweeps: []Sweep,
        live: *L.Mask,
        out: []LaneOutcome,

        const Self = @This();

        /// One lane's own error ends its sweep.
        fn kill(self: Self, l: usize, err: LaneError) void {
            assert(self.live.* & bit(L, l) != 0);
            self.out[l] = err;
            self.sweeps[l].deinit();
            self.live.* &= ~bit(L, l);
        }

        /// The batch's own error (out of memory) fails every live lane.
        fn fail(self: Self, err: LaneError) void {
            var it = L.each(self.live.*);
            while (it.next()) |l| self.kill(l, err);
            assert(self.live.* == 0);
        }

        /// A finished lane publishes its table.
        fn finish(self: Self, l: usize) void {
            self.out[l] = self.sweeps[l].take();
            self.sweeps[l].deinit();
            self.live.* &= ~bit(L, l);
        }

        /// Frees the sweeps of the lanes still live when the batch unwinds.
        fn endAll(self: Self) void {
            var it = L.each(self.live.*);
            while (it.next()) |l| self.sweeps[l].deinit();
        }
    };
}

fn bit(comptime L: type, l: usize) L.Mask {
    return @as(L.Mask, 1) << @intCast(l);
}

/// The warm Newton on one sweep's circuit; true when it converged. A
/// SingularMatrix (NaN stamps from a bad warm guess) does not abort the
/// sweep: the point falls to the cold ladder like any other failure.
fn warmNewton(s: *Sweep) !bool {
    const r = converger.run(s.ckt, s.ws, s.sol(), 0, s.warmOptions(), root.EvalHook{}) catch |e| switch (e) {
        error.SingularMatrix => return false,
        else => return e, // Only a singular warm guess falls to the ladder.
    };
    return r.converged;
}

/// A lane's own error ends its sweep.
fn drop(s: *Sweep, on: *bool, o: *LaneOutcome, e: LaneError) void {
    assert(on.*);
    o.* = e;
    s.deinit();
    on.* = false;
}

/// One point's warm Newton on the device for the lanes of `t.warm`: the
/// device batch built on the first call, every lane's rows and start
/// uploaded, then each converged lane's x in `t.xs` and its device state
/// copied back for `Sweep.endPoint` to commit. `t.res[l]` holds a warm
/// lane's outcome.
fn devicePoint(gpa: Allocator, dev: *?*gpu_lanes.GpuLanes, t: GpuTables) !void {
    const d = dev.* orelse blk: {
        const made = try gpu_lanes.GpuLanes.initStatic(gpa, t.ckts, t.xs);
        made.prof.on = t.ckts[0].knobs.gpu_stats != .off;
        dev.* = made;
        break :blk made;
    };
    try d.reload(t.xs, true);
    var first: usize = 0;
    for (d.round, t.warm, 0..) |*r, w, l| {
        r.* = @intFromBool(w);
        if (w and !t.warm[first]) first = l;
    }
    assert(t.warm[first]);
    try d.solveStatic(t.xs, t.sweeps[first].warmOptions(), t.res);
    for (d.detached, t.warm, t.res) |*det, w, r| det.* = !(w and r.result.converged and !r.singular and !r.reject);
    d.copyBack();
}

test {
    _ = Sweep;
}
