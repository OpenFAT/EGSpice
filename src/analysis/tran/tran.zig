//! Transient analysis: a circuit at its operating point (or the `.ic`
//! values under uic) and `Options` in, the probes at every accepted point
//! out, as `Waveform` rows. Newton per timestep on A = G + ag0*C, with
//! ngspice's LTE step control, order promotion and breakpoint landing
//! (`Sim.zig`); the companion residual uses the exact q(x) plane and the
//! Jacobian the analytic C plane. This file is the query spine: one run,
//! the CPU lane batch, and the GPU lane batch (`gpu_drive.zig`).
//! Cites: ngspice dctran.c, traninit.c, cktsetbk.c, ckttrunc.c, niconv.c,
//! indload.c, capload.c and AlmostEqualUlps (maths/misc/equality.c);
//! Verilog-AMS LRM §4.5, §5.10.2, §9.10 and §9.17.2; #161.
//!
//! Contents:
//! - `pub const plotname: []const u8`, `pub const scale: []const u8`
//! - `pub const Options = types.Options`, `pub const Waveform = types.Waveform`,
//!   `pub const Column = types.Column`, `pub const initialCapacity = types.initialCapacity`
//! - `pub const LaneError = root.ErrorOf(run)`, `pub const LaneOutcome = LaneError!root.Result`
//! - `pub const runGpuLanes = gpu_drive.runGpuLanes`
//! - `pub const test_access`
//! - `pub fn simulate(ckt: *root.Circuit, x: []f64, probes: []const u32, waveform: *Waveform, options: Options, allocator: std.mem.Allocator) !bool`
//! - `pub fn run(ctx: *const root.RunCtx, opts: Options) !root.Result`
//! - `pub fn runEach(ctxs: []const root.RunCtx, opts: Options, out: []LaneOutcome) void`
//! - `pub fn runLanes(comptime width: usize, ctxs: []const root.RunCtx, opts: Options, out: []LaneOutcome) void`

const std = @import("std");
const assert = std.debug.assert;
const converger = @import("solver").converger;
const Stream = @import("measure").Stream;
const root = @import("../types.zig");
const gpu_drive = @import("gpu_drive.zig");
const host_ops = @import("host_ops.zig");
const integrator = @import("integrator.zig");
const Run = @import("Run.zig");
const Sim = @import("Sim.zig");
const Tangents = @import("Tangents.zig");
const tran_types = @import("types.zig");

const Attempt = Sim.Attempt;
const HostOps = host_ops.HostOps;
const TranHook = Sim.TranHook;

/// The swept plot's name (`session.schemaOf` reports it).
pub const plotname = "Transient Analysis";
/// The scale column's name, column 0 of every row `run` publishes.
pub const scale = "time";
/// `.tran` query options, defined in core/query.zig.
pub const Options = tran_types.Options;
/// The recorded waveform (types.zig).
pub const Waveform = tran_types.Waveform;
/// A borrowed waveform column (types.zig).
pub const Column = tran_types.Column;
/// The waveform's initial capacity for a query (types.zig).
pub const initialCapacity = tran_types.initialCapacity;
/// Every error `run` returns.
pub const LaneError = root.ErrorOf(run);
/// One lane's `run` outcome.
pub const LaneOutcome = LaneError!root.Result;
/// `runLanes` on the GPU (`gpu_drive.zig`).
pub const runGpuLanes = gpu_drive.runGpuLanes;

/// Private implementation access for the analysis test suite.
pub const test_access = if (@import("builtin").is_test) .{
    .W = std.simd.suggestVectorLength(f64) orelse 8,
    .integrator = integrator,
    .predict = Sim.predict,
    .almostEqualUlps = Sim.almostEqualUlps,
} else {};

/// Integrates from the operating point in `x` to options.t_stop, recording
/// accepted points at t >= t_start into `waveform`. When the loop ends at
/// t_stop or at `max_steps`, `x` holds the last accepted solution; after a
/// dt underflow its contents are unspecified (one of the rotating step
/// buffers was live). `allocator` backs per-run scratch only. Returns
/// true on reaching t_stop; a dt underflow or `max_steps` returns false,
/// not an error; the errors are cancellation, circuit setup and allocation (scratch or
/// `waveform`). Leaves the device state committed at the last accepted point.
/// Ignores `options.tangents`.
pub fn simulate(
    ckt: *root.Circuit,
    x: []f64,
    probes: []const u32,
    waveform: *Waveform,
    options: Options,
    allocator: std.mem.Allocator,
) !bool {
    return simulateWith(ckt, x, probes, waveform, options, allocator, null, null);
}

/// Contract entry: integrate from the operating point and return the
/// waveform as point-major rows (time, probes...). A run that stops short of
/// t_stop is error.TimestepTooSmall, as in ngspice. The result's names and
/// data live in `ctx.allocator`; `opts.snapshot` returns the one row at
/// t_stop instead, and a streamed run (`ctx.stream`) returns empty `data`.
/// `opts.point.measures_only` measures `ctx.measures`' `.tran` cards as the
/// run goes (`measure.Stream`) and keeps only the columns the rest read,
/// the result's `measured` holding what the others found; under tangents
/// it keeps every column those cards read instead. `early_stop` ends the
/// run, with the rows so far, once every such card has resolved, and
/// `abort_after` fails it with `error.MeasureBoundExceeded` once
/// past that time with one unresolved, or with `abort_measure` unresolved. A streamed or snapshot run ignores all three.
pub fn run(ctx: *const root.RunCtx, opts: Options) !root.Result {
    var r: Run = undefined;
    try r.init(ctx, opts);
    defer r.deinit();
    return r.take(try r.sim.drive());
}

/// `run` for the lanes of one batch: `.tran` queries on one pattern, each
/// from its own operating point (or `uic` start) and variant row, in
/// rounds. Each round every live lane
/// `begin`s an attempt at its own (t, dt), one `newtonLanes` solves them,
/// and each `end`s it with its own step control, so no decision couples
/// lanes and lane l's result lands in `out[l]` bitwise `run` on `ctxs[l]`.
/// A JFNK pin, `-Dcertify` or a VPI application runs every lane's `run`.
/// Asserts 1 <= `ctxs.len` <= `width`.
pub fn runLanes(comptime width: usize, ctxs: []const root.RunCtx, opts: Options, out: []LaneOutcome) void {
    const g = ctxs.len;
    std.debug.assert(g >= 1 and g <= width and out.len == g);
    const c0 = ctxs[0].circuit;
    const pin = c0.knobs.newton.pin;
    if (width == 1 or g == 1 or !(pin == .auto or pin == .direct) or root.Circuit.certify or c0.vpi != null)
        return runEach(ctxs, opts, out);
    const L = @import("solver").lanes.Lanes(width);
    const System = @import("solver").lane_system.LaneSystem(width, root.Circuit, TranHook);
    const bit = struct {
        fn f(l: usize) L.Mask {
            return @as(L.Mask, 1) << @intCast(l);
        }
    }.f;
    var runs: [width]Run = undefined;
    var live: L.Mask = 0;
    for (ctxs, 0..) |*c, l| {
        runs[l].init(c, opts) catch |e| {
            out[l] = e;
            continue;
        };
        live |= bit(l);
    }
    defer {
        var it = L.each(live);
        while (it.next()) |l| runs[l].deinit();
    }
    const Lane = struct {
        /// Lane l ends with `ok` (`Sim.finish`'s value, or false short of
        /// t_stop): its result, then its scratch.
        fn done(rs: []Run, m: *L.Mask, o: []LaneOutcome, l: usize, ok: bool) void {
            o[l] = rs[l].take(ok);
            rs[l].deinit();
            m.* &= ~bit(l);
        }
        fn kill(rs: []Run, m: *L.Mask, o: []LaneOutcome, l: usize, err: LaneError) void {
            o[l] = err;
            rs[l].deinit();
            m.* &= ~bit(l);
        }
        /// Every live lane fails with the batch's own error (out of memory,
        /// a cancel).
        fn fail(rs: []Run, m: *L.Mask, o: []LaneOutcome, err: LaneError) void {
            var it = L.each(m.*);
            while (it.next()) |l| kill(rs, m, o, l, err);
        }
    };
    if (live == 0) return;
    const gpa = ctxs[@ctz(live)].scratch_allocator;
    const n = c0.n;

    var sys: ?System = null;
    defer if (sys) |*s| s.deinit(gpa);
    var lws: ?@import("solver").converger.LaneWorkspace(width) = null;
    defer if (lws) |*w| w.deinit(gpa);
    const xv = gpa.alloc(L.V, n) catch |e| return Lane.fail(&runs, &live, out, e);
    defer gpa.free(xv);
    var atts: [width]Attempt = undefined;

    while (live != 0) {
        var round: L.Mask = 0;
        var tv: [width]f64 = @splat(0);
        var it = L.each(live);
        while (it.next()) |l| {
            const sim = &runs[l].sim;
            const a = sim.begin(HostOps{}) catch |e| {
                Lane.kill(&runs, &live, out, l, e);
                continue;
            } orelse {
                const ok = sim.finish() catch |e| {
                    Lane.kill(&runs, &live, out, l, e);
                    continue;
                };
                Lane.done(&runs, &live, out, l, ok);
                continue;
            };
            atts[l] = a;
            tv[l] = a.t_dev;
            round |= bit(l);
        }
        if (round == 0) break;
        const lead = @ctz(round);
        if (sys == null) {
            // Unused and finished lanes repeat the lead and stay inactive.
            var lanes: [width]*root.Circuit = undefined;
            var wss: [width]*converger.Workspace = undefined;
            for (&lanes, &wss, 0..) |*c, *w, l| {
                const k = if (l < g and (live >> @intCast(l)) & 1 != 0) l else lead;
                c.* = runs[k].sim.ckt;
                w.* = runs[k].sim.ws;
            }
            sys = System.init(gpa, lanes, atts[lead].hook) catch |e| return Lane.fail(&runs, &live, out, e);
            lws = converger.LaneWorkspace(width).init(gpa, wss) catch |e| return Lane.fail(&runs, &live, out, e);
        }
        const s = &sys.?;
        it = L.each(round);
        while (it.next()) |l| {
            s.hooks[l] = atts[l].hook;
            System.putLane(xv, runs[l].sim.trial, l);
        }
        s.active = round;
        const res = converger.newtonLanes(width, s, &lws.?, xv, tv, atts[lead].nr_opts, System.LaneHook{});
        // As `converger.run` ends each lane's solve.
        it = L.each(round);
        while (it.next()) |l| runs[l].sim.ckt.clearLimits();
        const r = res catch |e| return Lane.fail(&runs, &live, out, e);
        it = L.each(round);
        while (it.next()) |l| {
            const sim = &runs[l].sim;
            System.getLane(sim.trial, xv, l);
            const b = bit(l);
            // A singular lane is `converger.run`'s caught numerical failure.
            const nr: converger.Result = if (r.singular & b != 0)
                .{ .converged = false, .iterations = 0 }
            else
                .{ .converged = r.converged & b != 0, .iterations = r.iterations[l] };
            const v = sim.end(&atts[l], nr, HostOps{}) catch |e| {
                Lane.kill(&runs, &live, out, l, e);
                continue;
            };
            switch (v) {
                .go => {},
                .stop => {
                    const ok = sim.finish() catch |e| {
                        Lane.kill(&runs, &live, out, l, e);
                        continue;
                    };
                    Lane.done(&runs, &live, out, l, ok);
                },
                .underflow => Lane.done(&runs, &live, out, l, false),
            }
        }
    }
}

/// Each lane's `run`, one after another.
pub fn runEach(ctxs: []const root.RunCtx, opts: Options, out: []LaneOutcome) void {
    for (ctxs, out) |*c, *o| o.* = run(c, opts);
}

// ---- private ----
/// `simulate`, also advancing and recording `tangents` at the operating
/// point and every accepted step, and feeding each recorded row to `stream`
/// for `options.point`'s early stop (returns true once every card has
/// resolved, the tangent rows' too) and abort bound
/// (`error.MeasureBoundExceeded` at the first row past it with a card
/// unresolved).
fn simulateWith(
    ckt: *root.Circuit,
    x: []f64,
    probes: []const u32,
    waveform: *Waveform,
    options: Options,
    allocator: std.mem.Allocator,
    tangents: ?*Tangents,
    stream: ?*Stream,
) !bool {
    var s: Sim = undefined;
    try s.init(ckt, x, probes, waveform, options, allocator, tangents, stream);
    defer s.deinit();
    return s.drive();
}
