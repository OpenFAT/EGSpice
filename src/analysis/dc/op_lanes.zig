//! The operating point for the lanes of one batch: the setup and the plain
//! rung run as W host lanes (`solveLanes`) or every lane on the GPU
//! (`solveGpuLanes`); a lane the plain rung leaves unconverged or singular
//! peels to its own scalar ladder from rung 2 (class E: the stepping rungs
//! are not masked yet).
//! In: W circuits on one pattern and their start vectors. Out: one
//! `op.LaneOutcome` per lane.
//! Cites dev/research/lane-batched-simulation.md §6 (the `.op` ladder);
//! 06 6.9 (GPU lanes).
//!
//! Contents:
//! - `pub fn solveLanes(comptime W: usize, ckts, xs, options, nodeset, ic, out: []LaneOutcome) !void`
//! - `pub fn solveGpuLanes(comptime W: usize, ckts, xs, options, nodeset, ic, out: []LaneOutcome) !void`

const std = @import("std");
const assert = std.debug.assert;
const solver = @import("solver");
const converger = solver.converger;
const Ic = @import("core").Ic;
const root = @import("../types.zig");
const gpu_lanes = @import("../gpu/lanes.zig");
const op = @import("op.zig");
const op_ladder = @import("op_ladder.zig");
const Options = op.Options;
const LaneOutcome = op.LaneOutcome;

/// `op.solve` for the lanes of one batch, each lane bitwise its own
/// `op.solve`. A batch with nodesets, a floating start, a JFNK pin or
/// `-Dcertify` runs every lane's scalar `op.solve`. Lane l's outcome lands
/// in `out[l]`; the call itself fails only when the lane solve does (out of
/// memory), which fails every lane. Asserts 1 <= `ckts.len` <= W and that
/// `xs` and `out` match it.
pub fn solveLanes(comptime W: usize, ckts: []const *root.Circuit, xs: []const []f64, options: Options, nodeset: []const Ic, ic: []const Ic, out: []LaneOutcome) !void {
    const g = ckts.len;
    assert(g >= 1);
    assert(g <= W);
    assert(xs.len == g);
    assert(out.len == g);
    const c0 = ckts[0];
    const held = if (options.tran_op) ic else &.{};
    const laned = W > 1 and g > 1 and nodeset.len == 0 and !(c0.needs_tran_op and held.len == 0) and op.lanesAllowed(c0);
    if (!laned) {
        for (ckts, xs, out) |c, x, *o| o.* = op.solve(c, x, options, nodeset, ic);
        return;
    }
    const System = solver.lane_system.LaneSystem(W, root.Circuit, root.EvalHook);
    const L = solver.lanes.Lanes(W);
    // Unused lanes repeat lane 0 and stay inactive.
    var lanes: [W]*root.Circuit = undefined;
    var wss: [W]*converger.Workspace = undefined;
    for (&lanes, &wss, 0..) |*c, *w, l| {
        c.* = ckts[if (l < g) l else 0];
        if (l < g) {
            try op.beginSolve(c.*, xs[l], options);
            w.* = try c.*.workspace();
        } else w.* = wss[0];
    }
    const gpa = wss[0].slv.gpa;
    const force = try gpa.alloc(converger.Force, held.len);
    defer gpa.free(force);

    for (held, force) |h, *f| f.* = .{ .slot = c0.diag_slots[h.node], .row = h.node, .value = h.value };
    var sys = try System.init(gpa, lanes, .{});
    defer sys.deinit(gpa);

    sys.active = @truncate(@as(u64, std.math.maxInt(L.Mask)) >> @intCast(W - g));
    var lws = try converger.LaneWorkspace(W).init(gpa, wss);
    defer lws.deinit(gpa);

    const xv = try gpa.alloc(L.V, c0.n);
    defer gpa.free(xv);

    for (xs, 0..) |x, l| System.putLane(xv, x, l);
    // The plain rung, as `op_ladder.solve` and `converger.run` run it.
    var copts = converger.optionsFromTolerances(options.tol, null);
    copts.gmin = 0;
    copts.init_fix = !options.warm_start;
    copts.force = force;
    const res = converger.newtonLanes(W, &sys, &lws, xv, @splat(0), copts, System.LaneHook{});
    sys.clearLimits();
    const plain = try res;
    for (xs, 0..) |x, l| System.getLane(x, xv, l);

    for (ckts, xs, out, 0..) |c, x, *o, l| {
        const bit = @as(L.Mask, 1) << @intCast(l);
        const p: ?converger.Result = if (plain.singular & bit != 0) null else .{ .converged = plain.converged & bit != 0, .iterations = plain.iterations[l] };
        const r = op_ladder.afterPlain(c, wss[l], x, options, force, p) catch |e| {
            o.* = e;
            continue;
        };
        op.endSolve(c, r.converged, options);
        o.* = r;
    }
}

/// `solveLanes` for any number of lanes with every lane's plain rung on the
/// GPU: one device Newton from INITFIX, as `op.solve` runs it; a lane it
/// leaves unconverged, singular or with a flipped latch takes its own
/// ladder from rung 2 on the host, from the device state it started with.
/// Within tolerance of `solveLanes`, not bitwise: the device replays one LU
/// pivot order. A batch the device cannot take (nodesets, `.ic` holds, a
/// floating start, a JFNK pin, `-Dcertify`, `gpu_lanes.staticRefusal`, a
/// driver error) runs `solveLanes` W lanes at a time. Asserts that `xs` and
/// `out` match `ckts`.
pub fn solveGpuLanes(comptime W: usize, ckts: []const *root.Circuit, xs: []const []f64, options: Options, nodeset: []const Ic, ic: []const Ic, out: []LaneOutcome) !void {
    assert(xs.len == ckts.len);
    assert(out.len == ckts.len);
    if (ckts.len == 0) return;
    const c0 = ckts[0];
    const held = if (options.tran_op) ic else &.{};
    const why: ?[]const u8 = if (ckts.len < 2 or nodeset.len != 0 or held.len != 0 or c0.needs_tran_op)
        "nodesets, holds or a floating start"
    else if (!op.lanesAllowed(c0))
        "a JFNK pin or certify"
    else
        gpu_lanes.staticRefusal(c0);
    if (why) |w| {
        if (c0.knobs.gpu_stats != .off) std.debug.print("gpu-lanes: op lanes on the CPU: {s}\n", .{w});
    } else if (solveOnDevice(ckts, xs, options, out)) return else |e| switch (e) {
        error.OutOfMemory => return error.OutOfMemory,
        else => if (c0.knobs.gpu_stats != .off) std.debug.print("gpu-lanes: op lanes declined ({s})\n", .{@errorName(e)}),
    }
    var i: usize = 0;
    while (i < ckts.len) : (i += W) {
        const end = @min(ckts.len, i + W);
        try solveLanes(W, ckts[i..end], xs[i..end], options, nodeset, ic, out[i..end]);
    }
}

// ---- private ----

/// `solveGpuLanes`' device half. Fails before any lane's outcome is set (a
/// driver error or `GpuLanes` refusal) or with out of memory.
fn solveOnDevice(ckts: []const *root.Circuit, xs: []const []f64, options: Options, out: []LaneOutcome) !void {
    assert(ckts.len >= 2);
    const gpa = ckts[0].gpa;
    for (ckts, xs) |c, x| try op.beginSolve(c, x, options);
    const res = try gpa.alloc(gpu_lanes.LaneNewton, ckts.len);
    defer gpa.free(res);

    {
        const dev = try gpu_lanes.GpuLanes.initStatic(gpa, ckts, xs);
        defer dev.deinit();

        dev.prof.on = ckts[0].knobs.gpu_stats != .off;
        var copts = converger.optionsFromTolerances(options.tol, null);
        copts.gmin = 0;
        copts.init_fix = !options.warm_start;
        try dev.solveStatic(xs, copts, res);
        // `deinit` copies back the device state of the lanes it settled; the
        // rest keep the host's start.
        for (res, dev.detached) |r, *d| d.* = !(r.result.converged and !r.singular and !r.reject);
    }
    for (ckts, xs, out, res) |c, x, *o, r| {
        const plain: ?converger.Result = if (r.singular) null else if (r.reject) .{ .converged = false, .iterations = r.result.iterations } else r.result;
        o.* = blk: {
            const ws = c.workspace() catch |e| break :blk e;
            // Rung 2 reads the plain rung's last assembly off the planes.
            if (plain == null or !plain.?.converged) c.evalNewton(x, 0);
            break :blk op_ladder.afterPlain(c, ws, x, options, &.{}, plain);
        };
        op.endSolve(c, if (o.*) |rr| rr.converged else |_| false, options);
    }
}
