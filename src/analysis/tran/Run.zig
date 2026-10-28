//! One `.tran` query's outputs around its `Sim`: a `RunCtx` and the options
//! in, the probes it records, its `.meas` stream, waveform and tangents,
//! then the published `Result` out, so `tran.run` and both lane drivers
//! build and publish each query the same way. Pinned: `init` fills it in
//! place, and `sim` points at `wf`, `stream` and `tangents`.
//! Cites: HSPICE `.op <time>` (the snapshot plot).
//!
//! Contents:
//! - `pub fn init(r: *Run, ctx: *const root.RunCtx, opts: Options) !void`
//! - `pub fn deinit(r: *Run) void`
//! - `pub fn take(r: *Run, sim_ok: bool) !root.Result`

const std = @import("std");
const assert = std.debug.assert;
const num = @import("core").numerics;
const measure = @import("measure");
const root = @import("../types.zig");
const Sim = @import("Sim.zig");
const Tangents = @import("Tangents.zig");
const tran = @import("tran.zig");
const tran_types = @import("types.zig");

const Stream = measure.Stream;
const Options = tran_types.Options;
const Waveform = tran_types.Waveform;
const initialCapacity = tran_types.initialCapacity;
const plotname = tran.plotname;
const scale = tran.scale;

ctx: *const root.RunCtx,
opts: Options,
x: []f64,
kept: bool,
streamed: bool,
names: []const []const u8,
own_probes: []u32,
stream: ?Stream,
wf: Waveform,
tangents: ?Tangents,
/// `take` handed `wf`'s rows to the result.
published: bool = false,
sim: Sim,

const Run = @This();

/// `run`'s setup, then `Sim.init`.
pub fn init(r: *Run, ctx: *const root.RunCtx, opts: Options) !void {
    const a = ctx.allocator;
    const scratch = ctx.scratch_allocator;
    const x_op = ctx.x_op;
    const x = try scratch.alloc(f64, x_op.len);
    errdefer scratch.free(x);
    num.copySimd(x, x_op);

    // The recording is the published result, so it lives in the results
    // arena and `data` is borrowed below. A `.op <time>` snapshot publishes
    // only x, and a streamed run writes its rows to `ctx.stream` as it goes,
    // so either keeps its recording in scratch and drops it here.
    const kept = !opts.snapshot and ctx.stream == null;
    const all_names = try root.probeNames(ctx, scale);
    var names = all_names;
    var probes = ctx.probes;
    var own_probes: []u32 = &.{};
    errdefer scratch.free(own_probes);
    const tran_cards = for (ctx.measures) |m| {
        if (m.analysis == .tran) break true;
    } else false;
    // Measured as the run goes, in place of their columns.
    const streamed = kept and opts.point.measures_only and opts.tangents.len == 0 and tran_cards;
    if (kept and opts.point.measures_only) {
        const keep = try scratch.alloc(bool, names.len);
        defer scratch.free(keep);
        measure.markColumns(ctx.measures, names, keep, streamed);
        const sub_names = try a.alloc([]const u8, std.mem.count(bool, keep, &.{true}));
        const sub_probes = try scratch.alloc(u32, sub_names.len - 1);
        var k: usize = 0;
        for (names, keep, 0..) |name, on, i| if (on) {
            sub_names[k] = name;
            if (k > 0) sub_probes[k - 1] = ctx.probes[i - 1];
            k += 1;
        };
        names = sub_names;
        probes = sub_probes;
        own_probes = sub_probes;
    }
    const watched = streamed or (kept and tran_cards and (opts.point.early_stop or opts.point.abort_after != null));
    var stream: ?Stream = if (watched) try .init(scratch, ctx.measures, all_names, ctx.probes, names) else null;
    errdefer if (stream) |*sm| sm.deinit();
    if (stream) |*sm| if (opts.point.abort_measure.len != 0) {
        sm.abort_card = for (ctx.measures, 0..) |m, k| {
            if (std.ascii.eqlIgnoreCase(m.name, opts.point.abort_measure)) break k;
        } else null;
    };
    const n_probes: u32 = @intCast(probes.len);
    var wf = if (kept)
        try Waveform.init(a, n_probes, initialCapacity(opts))
    else if (opts.snapshot)
        try Waveform.init(scratch, n_probes, initialCapacity(opts))
    else
        try Waveform.initStream(scratch, n_probes, ctx.stream.?);
    errdefer wf.deinit();
    var tangents = if (kept and opts.tangents.len != 0) try Tangents.init(ctx, opts, n_probes, scratch) else null;
    errdefer if (tangents) |*tg| tg.deinit(scratch);
    if (tangents) |*tg| if (watched and opts.point.early_stop) {
        const streams = try scratch.alloc(Stream, tg.wf.len);
        for (streams, 0..) |*sm, k| sm.* = Stream.init(scratch, ctx.measures, all_names, ctx.probes, names) catch |e| {
            for (streams[0..k]) |*done| done.deinit();
            scratch.free(streams);
            return e;
        };
        tg.streams = streams;
    };
    r.* = .{
        .ctx = ctx,
        .opts = opts,
        .x = x,
        .kept = kept,
        .streamed = streamed,
        .names = names,
        .own_probes = own_probes,
        .stream = stream,
        .wf = wf,
        .tangents = tangents,
        .sim = undefined,
    };
    try r.sim.init(ctx.circuit, x, probes, &r.wf, opts, scratch, if (r.tangents) |*tg| tg else null, if (r.stream) |*sm| sm else null);
}

/// Frees what `init` made, `wf` unless `take` published it.
pub fn deinit(r: *Run) void {
    const scratch = r.ctx.scratch_allocator;
    r.sim.deinit();
    if (r.tangents) |*tg| tg.deinit(scratch);
    if (!r.published) r.wf.deinit();
    if (r.stream) |*sm| sm.deinit();
    scratch.free(r.own_probes);
    scratch.free(r.x);
    r.* = undefined;
}

/// The query's result from a run that ended with `sim_ok` (`Sim.drive`
/// or `Sim.finish`): `run`'s tail.
pub fn take(r: *Run, sim_ok: bool) !root.Result {
    const ctx = r.ctx;
    const a = ctx.allocator;
    const opts = r.opts;
    if (!sim_ok) return error.TimestepTooSmall;
    // HSPICE `.op <time>`: the state at t_stop, laid out as `.op` lays it out.
    if (opts.snapshot) {
        const op_names = try root.probeNames(ctx, null);
        const data = try a.alloc(f64, op_names.len);
        for (ctx.probes, data) |node, *out| out.* = r.x[node];
        return .{
            .plotname = try std.fmt.allocPrint(a, "Operating Point (time={e})", .{opts.t_stop}),
            .varnames = op_names,
            .is_complex = false,
            .npoints = 1,
            .data = data,
        };
    }
    const tg = if (r.tangents) |*tg| tg else {
        const res: root.Result = .{
            .plotname = plotname,
            .varnames = r.names,
            .is_complex = false,
            .npoints = r.wf.len,
            .data = r.wf.data(),
            .measured = if (r.streamed) try r.stream.?.measured(a) else &.{},
        };
        r.published = r.kept;
        return res;
    };
    // Each row's block on the base run's time points.
    const block = r.wf.data().len;
    const blocks = try a.alloc(f64, block * tg.wf.len);
    for (tg.wf, 0..) |w, k| @memcpy(blocks[k * block ..][0..block], w.data());
    r.published = r.kept;
    return .{
        .plotname = plotname,
        .varnames = r.names,
        .is_complex = false,
        .npoints = r.wf.len,
        .data = r.wf.data(),
        .tangents = blocks,
    };
}
