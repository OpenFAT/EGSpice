//! One `.dc` analysis' state between its steps, so the scalar sweep and the
//! lane drivers in `dc.zig` share every step but the warm Newton. Blocks are
//! the outer sweep's levels, each sweeping the inner parameter: every point
//! warm-starts from the previous solution; a block's first point, and any
//! point whose warm Newton fails, cold-starts through the `.nodeset` step and
//! the full OP ladder, as dctrcurv.c's CKTop calls do.
//! In: a run context and `.dc` options. Out: the point-major result table
//! (sweep value, probes...), outer blocks concatenated, then one block per
//! `tangents` variant row.
//! Cites ngspice dctrcurv.c (predictor, state rotation at :290 and
//! :457-462, accumulated steps at :469) and DCTsetup;
//! dev/analysis/dc-sweep.md and dev/analysis/optimize.md (tangent rows).
//!
//! Contents:
//! - fields (`ctx`, `ckt`, `opts`, `t`, `saved`, `t2`, `saved2`, `sweeps_temp`, `temp_c`, `n_inner`, `n_outer`, `ncols`, `data`, `hist`, `force`, `ws`, `block_index`, `cold`, `v`, `v2`)
//! - `pub fn init(s: *Sweep, ctx: *const RunCtx, opts: Options) !void`
//! - `pub fn deinit(s: *Sweep) void`
//! - `pub fn take(s: *Sweep) Allocator.Error!Result`
//! - `pub fn sol(s: *const Sweep) []f64`
//! - `pub fn warmOptions(s: *const Sweep) converger.Options`
//! - `pub fn beginBlock(s: *Sweep, po: usize) !void`
//! - `pub fn beginPoint(s: *Sweep, pt: usize) !bool`
//! - `pub fn prepareWarm(s: *Sweep) void`
//! - `pub fn endPoint(s: *Sweep, pt: usize, warm_converged: bool) !void`

const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;
const converger = @import("solver").converger;
const Library = @import("device").Library;
const root = @import("../types.zig");
const op = @import("op.zig");
const Options = @import("core").query.Dc;

ctx: *const root.RunCtx,
ckt: *root.Circuit,
opts: Options,
/// The inner target; null sweeps the temperature.
t: ?root.ParamRef,
saved: f64,
t2: ?root.ParamRef = null,
saved2: f64 = 0,
sweeps_temp: bool,
temp_c: f32,
n_inner: usize,
n_outer: usize,
ncols: usize,
/// The result data, owned until `take` moves it out (then empty).
data: []f64,
/// The last two points for the predictor, carried across outer blocks as
/// ngspice's state rotation is, then `tangentRows`' scratch: F(x) and dx,
/// and the saved values of the widest row's writes.
hist: []f64,
force: []converger.Force,
ws: *converger.Workspace = undefined,
/// The outer level `beginBlock` installed.
block_index: usize = 0,
cold: bool = true,
v: f64 = 0,
v2: f64 = 0,

const Sweep = @This();

/// `run`'s setup through the outer target. On an error everything set up so
/// far is undone and freed. Fails with error.DcSweepSourceNotFound when a
/// target names no parameter.
pub fn init(s: *Sweep, ctx: *const root.RunCtx, opts: Options) !void {
    const ckt = ctx.circuit;
    // DC simulation state for the whole sweep, whatever the shared op left
    // behind. A deck with a .tran solves its op in the ic phase, where
    // analysis("tran") is true and sources bias at waveform(0), which would
    // make the swept dc value a no-op (rtlinv).
    ckt.setSimState(.{ .kind = .dc });
    // Matching the device type as well as the index lets a deck hold both a
    // V and an I card (their per-type indices overlap) and lets the sweep
    // name a resistor.
    const t = try targetRef(ckt, opts.target);
    // A temperature sweep at either level writes every Model row's
    // temperature; `deinit` puts the netlist's back.
    s.* = .{
        .ctx = ctx,
        .ckt = ckt,
        .opts = opts,
        .t = t,
        .saved = if (t) |r| r.get() else 0,
        .sweeps_temp = opts.target == .temp or (opts.hasOuter() and opts.target2.? == .temp),
        .temp_c = ckt.temp_c,
        .n_inner = if (opts.points.len > 0) opts.points.len else sweepCount(opts.start, opts.stop, opts.step),
        .n_outer = if (!opts.hasOuter()) 1 else if (opts.points2.len > 0) opts.points2.len else sweepCount(opts.start2, opts.stop2, opts.step2),
        .ncols = ctx.probes.len + 1,
        .data = &.{},
        .hist = &.{},
        .force = &.{},
    };
    errdefer s.deinit();

    const npoints = s.n_inner * s.n_outer;
    s.data = try ctx.allocator.alloc(f64, npoints * s.ncols * (1 + opts.tangents.len));
    var widest: usize = 0;
    for (opts.tangents) |row| widest = @max(widest, ctx.variants.writes(row)[0].len);
    s.hist = try ctx.scratch_allocator.alloc(f64, 2 * ckt.n + if (opts.tangents.len == 0) 0 else 2 * ckt.n + widest);
    @memset(s.hist[0 .. 2 * ckt.n], 0);
    s.force = try ctx.scratch_allocator.alloc(converger.Force, ctx.nodeset.len);
    for (s.force, ctx.nodeset) |*f, ns| f.* = .{ .slot = ckt.diag_slots[ns.node], .row = ns.node, .value = ns.value };
    // ngspice's second variable is the outer loop: the whole inner sweep
    // replays for each src2 value and the raw file concatenates the blocks.
    if (opts.hasOuter()) {
        s.t2 = try targetRef(ckt, opts.target2.?);
        s.saved2 = if (s.t2) |r| r.get() else 0;
        // Accumulated, not start + k*step: ngspice steps both levels with
        // `+= TRCVvStep` and its axis carries that roundoff (1.4975e-13
        // where start + k*step gives 0), above the oracle's axis tolerance.
        s.v2 = opts.start2;
    }
    assert(s.n_inner >= 1);
    assert(s.n_outer >= 1);
}

/// Restores what the sweep moved (the outer target, the temperature, then
/// the inner target with a full recompute) and frees its buffers, the data
/// too unless `take` moved it out.
pub fn deinit(s: *Sweep) void {
    if (s.t2) |r| r.set(s.saved2);
    if (s.sweeps_temp) s.ckt.setCircuitTemp(s.temp_c);
    if (s.t) |r| r.set(s.saved);
    // The original values were installed by this circuit once, so they cannot re-wire it.
    s.ckt.recompute() catch unreachable;
    s.ctx.allocator.free(s.data);
    s.ctx.scratch_allocator.free(s.hist);
    s.ctx.scratch_allocator.free(s.force);
}

/// Moves the table out as the query's Result; the results arena owns it.
pub fn take(s: *Sweep) Allocator.Error!root.Result {
    const varnames = try root.probeNames(s.ctx, sweepColumn(s.opts.target));
    const data = s.data;
    s.data = &.{};
    assert(data.len == s.n_inner * s.n_outer * (1 + s.opts.tangents.len) * s.ncols);
    return .{
        .plotname = "DC transfer characteristic",
        .varnames = varnames,
        .is_complex = false,
        .npoints = s.n_inner * s.n_outer * (1 + s.opts.tangents.len),
        .data = data,
    };
}

/// Returns the working solution, the first `n` of `hist`.
pub fn sol(s: *const Sweep) []f64 {
    return s.hist[0..s.ckt.n];
}

/// Returns the warm Newton's options: the deck's tolerances under itl2.
pub fn warmOptions(s: *const Sweep) converger.Options {
    return converger.optionsFromTolerances(s.opts.tol, s.opts.tol.itl2);
}

/// Installs outer level `po` and starts its block cold at `start`.
/// Asserts that `po` is a level of the sweep.
pub fn beginBlock(s: *Sweep, po: usize) !void {
    assert(po < s.n_outer);
    if (s.opts.hasOuter()) {
        if (po != 0) s.v2 += s.opts.step2;
        if (s.opts.points2.len > 0) s.v2 = s.opts.points2[po];
        if (s.t2) |r| r.set(s.v2) else s.ckt.setCircuitTemp(@floatCast(s.v2));
    }
    s.block_index = po;
    // The pattern is frozen, so one workspace (one symbolic factorization)
    // serves every point.
    s.ws = try s.ckt.workspace();
    s.cold = true;
    s.v = s.opts.start;
}

/// Installs point `pt` of the block and writes its predictor into `sol()`;
/// returns true when the point runs a warm Newton first. Asserts that `pt`
/// is a point of the block.
pub fn beginPoint(s: *Sweep, pt: usize) !bool {
    assert(pt < s.n_inner);
    const ckt = s.ckt;
    const opts = s.opts;
    if (pt != 0) try ckt.checkpoint(.{ .phase = .dc, .completed = pt, .total = s.n_inner });
    // Accumulated like ngspice (see `init`).
    if (pt != 0) s.v += opts.step;
    if (opts.points.len > 0) s.v = opts.points[pt];
    if (s.t) |r| r.set(s.v) else ckt.setCircuitTemp(@floatCast(s.v));
    // Only `t`'s device type moved, so later points re-derive just that type
    // (a full walk cost 1,811 BJT preamble runs on a one-instance deck).
    // Point 0 takes the full walk: an outer `.dc ... temp` loop may just have
    // re-wired a device (`Circuit.recomputeType`). A temperature point moves
    // every type.
    if (s.t) |r| {
        if (pt == 0) try ckt.recompute() else try ckt.recomputeType(r.type);
    } else try ckt.recompute();
    try ckt.computeBaseline();

    // MODEINITPRED (dctrcurv.c:375): start from the last two points
    // extrapolated (xfact = 1 on the uniform step), limited against the last
    // one. A cold point overwrites the guess; the rotation still runs.
    for (s.sol(), s.previous()) |*xi, *xp| {
        const last = xi.*;
        xi.* = 2.0 * last - xp.*;
        xp.* = last;
    }
    return !s.cold;
}

/// Writes the warm point's start: the predictor limited against the last point.
pub fn prepareWarm(s: *Sweep) void {
    assert(!s.cold);
    s.ckt.evalFollowsFirst(s.sol(), 0);
    _ = s.ckt.applyLimits(s.sol(), s.previous()); // A limited start needs no flag here.
}

/// Ends point `pt`: the cold ladder when the warm Newton did not converge
/// (or did not run), then the point's row and tangent rows. A point that
/// does not converge records NaN probes, in every block.
pub fn endPoint(s: *Sweep, pt: usize, warm_converged: bool) !void {
    assert(pt < s.n_inner);
    const ckt = s.ckt;
    const ctx = s.ctx;
    const opts = s.opts;
    const ncols = s.ncols;
    const x = s.sol();
    // Rows in one tangent block: every point of every outer level.
    const block = s.n_inner * s.n_outer;
    var converged = warm_converged;
    if (!converged and s.isOpPoint(pt)) {
        @memcpy(x, ctx.x_op);
        converged = true;
    }
    if (!converged) {
        op.coldStart(ckt, x);
        converged = (try op.nodesetLadder(ckt, s.ws, x, .{ .tol = opts.tol }, s.force, 0)).converged;
    }
    const at = s.block_index * s.n_inner + pt;
    const row = s.data[at * ncols ..][0..ncols];
    row[0] = s.v;
    for (0..opts.tangents.len) |k| s.data[((k + 1) * block + at) * ncols] = s.v;
    if (!converged) {
        for (0..1 + opts.tangents.len) |k| @memset(s.data[(k * block + at) * ncols ..][1..ncols], std.math.nan(f64));
        s.cold = true;
        return;
    }
    // Each sweep point is an accepted point for device state.
    _ = ckt.stateCtl(.commit);
    for (ctx.probes, row[1..]) |node, *o| o.* = x[node];
    if (opts.tangents.len != 0) try tangentRows(ctx, ckt, s.ws, x, opts.tangents, s.hist[2 * ckt.n ..], s.data[(block + at) * ncols ..], block * ncols, ncols);
    s.cold = false;
    // ngspice copies state0 into state1 after the first point
    // (dctrcurv.c:457-462), so the second point's MODEINITPRED holds the
    // first point instead of doubling it.
    if (s.block_index == 0 and pt == 0) @memcpy(s.previous(), x);
}

// ---- private ----

/// The predictor's previous point, the second `n` of `hist`.
fn previous(s: *const Sweep) []f64 {
    return s.hist[s.ckt.n..][0..s.ckt.n];
}

/// Point `pt` of the current block is the deck's operating point: the first
/// point of the first block, with every swept device value at its netlist
/// value. Its cold ladder is then the OP prerequisite's own solve, whose
/// answer is `ctx.x_op` (04 4.10).
fn isOpPoint(s: *const Sweep, pt: usize) bool {
    if (pt != 0 or s.block_index != 0 or s.t == null or s.v != s.saved) return false;
    if (s.opts.hasOuter() and (s.t2 == null or s.v2 != s.saved2)) return false;
    return s.ctx.x_op.len == s.ckt.n;
}

/// The ref `target` sweeps, null for the temperature (`.dc TEMP ...`).
fn targetRef(ckt: *root.Circuit, target: Options.SweepTarget) !?root.ParamRef {
    return switch (target) {
        .temp => null,
        .device => |d| try findTarget(ckt, d) orelse return error.DcSweepSourceNotFound,
    };
}

/// The parameter `want` names. Only the batches of its type are collected,
/// so the other batches keep their Model rows shared; the ref stays valid as
/// `collectParams`' do.
fn findTarget(ckt: *root.Circuit, want: Options.SweepTarget.Param) !?root.ParamRef {
    var refs: std.ArrayList(root.ParamRef) = .empty;
    defer refs.deinit(ckt.gpa);

    for (ckt.batches, ckt.batch_types) |b, t| {
        if (t != want.type) continue;
        refs.clearRetainingCapacity();
        try b.hooks.collect_params(b.ctx, ckt.gpa, &refs).unwrap();
        for (refs.items) |ref| {
            if (ref.index != want.index or !std.mem.eql(u8, ref.param_name, want.param_name)) continue;
            var r = ref;
            r.type = t;
            return r;
        }
    }
    return null;
}

/// The sweep column's name, after the swept quantity as ngspice names it.
fn sweepColumn(target: Options.SweepTarget) []const u8 {
    const d = switch (target) {
        .temp => return "temp-sweep",
        .device => |d| d,
    };
    inline for (.{
        .{ "isource", "i(i-sweep)" },
        .{ "resistor", "res-sweep" },
        .{ "capacitor", "cap-sweep" },
        .{ "inductor", "ind-sweep" },
    }) |pair| if (d.type == Library.builtin(pair[0])) return pair[1];
    return "v(v-sweep)";
}

/// The probes of each `tangents` row at the solution `x`, to first order:
/// x - J^-1 (F(x, row) - F(x)), F with every parameter as the row writes it
/// and J factored at `x`, so one solve per row replaces the Newton solve of
/// a sweep at that row. Row k's probes go to `out[k * stride + 1 ..]`; NaN
/// when J at `x` is singular. `buf` holds 2n values plus the widest row's
/// writes. Leaves every parameter as it found it.
fn tangentRows(ctx: *const root.RunCtx, ckt: *root.Circuit, ws: *converger.Workspace, x: []const f64, tangents: []const u32, buf: []f64, out: []f64, stride: usize, ncols: usize) !void {
    const n = ckt.n;
    assert(x.len == n);
    assert(buf.len >= 2 * n);
    const rhs = buf[0..n];
    const dx = buf[n..][0..n];
    // A full stamp: the baseline would hold the const-Jacobian devices
    // (resistors) at their nominal values, as in sens.zig.
    ckt.eval(x, 0);
    @memcpy(rhs, ckt.rhs[0..n]);
    const factored = if (ws.slv.factor(ckt.g_vals, ckt.solver_execution)) true else |e| switch (e) {
        error.SingularMatrix => false,
        else => return e, // Only a singular J has an answer (NaN rows).
    };
    for (tangents, 0..) |row, k| {
        const probes = out[k * stride + 1 ..][0 .. ncols - 1];
        if (!factored) {
            @memset(probes, std.math.nan(f64));
            continue;
        }
        const refs, const values = ctx.variants.writes(row);
        const saved = buf[2 * n ..][0..refs.len];
        // All read before any is written, so a ref written twice saves its
        // nominal both times.
        for (refs, saved) |r, *old| old.* = (try ckt.paramAt(r)).value;
        try ckt.setParams(refs, values, false);
        // A row that re-wires a device has no first-order answer here.
        const rewired = if (recomputeTypes(ckt, refs)) false else |e| switch (e) {
            error.TopologyChanged => true,
            else => |other| return other,
        };
        if (!rewired) {
            ckt.eval(x, 0);
            for (dx, ckt.rhs[0..n], rhs) |*d, f, f0| d.* = f - f0;
        }
        try ckt.setParams(refs, saved, false);
        recomputeTypes(ckt, refs) catch |e| switch (e) {
            error.TopologyChanged => unreachable, // Restores the values it was built at.
            else => |other| return other,
        };
        if (rewired) {
            @memset(probes, std.math.nan(f64));
            continue;
        }
        ws.slv.solveNeg(dx, dx);
        for (ctx.probes, probes) |node, *p| p.* = x[node] + dx[node];
    }
}

/// `Circuit.recomputeType` once for each device type `refs` writes.
fn recomputeTypes(ckt: *root.Circuit, refs: []const u32) !void {
    for (refs, 0..) |r, i| {
        const t = (try ckt.paramAt(r)).type;
        for (refs[0..i]) |earlier| {
            if ((try ckt.paramAt(earlier)).type == t) break;
        } else try ckt.recomputeType(t);
    }
}

/// Points from `start` to `stop` by `step`, endpoint inclusive (DCTsetup
/// loops `v <= stop`); 1 for a zero step or one pointing away from `stop`.
fn sweepCount(start: f64, stop: f64, step: f64) usize {
    if (step == 0 or (stop - start) * std.math.sign(step) < 0) return 1;
    // An exact-integer ratio arrives just under itself in f64
    // ((0.95-0.3)/0.005 is 129.9999999), and a bare floor would drop the
    // last point (hicum2_gummel: 130 vs ngspice's 131). The 1e-6-step nudge
    // absorbs the ~1e-13 division error and is far below any fractional step
    // a deck means.
    return @as(usize, @intFromFloat(@floor((stop - start) / step + 1e-6))) + 1;
}

test sweepCount {
    const t = std.testing;
    // (0.95 - 0.3) / 0.005 lands just under 130 in f64; the endpoint stays.
    try t.expectEqual(@as(usize, 131), sweepCount(0.3, 0.95, 0.005));
    try t.expectEqual(@as(usize, 11), sweepCount(0, 1, 0.1));
    // A step that does not divide the span stops short of `stop`.
    try t.expectEqual(@as(usize, 4), sweepCount(0, 1, 0.3));
    try t.expectEqual(@as(usize, 1), sweepCount(2, 2, 1));
    // Descending with a negative step counts like ascending.
    try t.expectEqual(@as(usize, 6), sweepCount(5, 0, -1));
    // A zero step, or one pointing away from `stop`, is one point.
    try t.expectEqual(@as(usize, 1), sweepCount(0, 1, 0));
    try t.expectEqual(@as(usize, 1), sweepCount(0, 1, -0.1));
    try t.expectEqual(@as(usize, 1), sweepCount(1, 0, 0.1));
}
