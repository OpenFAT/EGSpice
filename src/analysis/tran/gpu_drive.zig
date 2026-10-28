//! `tran.runLanes` on the GPU: any number of `.tran` queries on one pattern
//! in, one `Result` or error per lane out. Each lane keeps its own `Sim` on
//! the host (step control, breakpoints, measures) over one `GpuLanes` that
//! holds every lane's numerics; the device answers `Sim.end` through
//! `gpu/lanes.zig`'s `Ops`. Under device step control the round loop is
//! `gpu_step.zig`'s.
//! Cites: none (phase 4 of dev/analysis/lane-batched-simulation.md).
//!
//! Contents:
//! - `pub fn runGpuLanes(ctxs: []const root.RunCtx, opts: Options, out: []LaneOutcome) void`

const std = @import("std");
const measure = @import("measure");
const LK = @import("solver").lane_kernels;
const root = @import("../types.zig");
const gpu_lanes = @import("../gpu/lanes.zig");
const gpu_step = @import("gpu_step.zig");
const integrator = @import("integrator.zig");
const Run = @import("Run.zig");
const Sim = @import("Sim.zig");
const tran = @import("tran.zig");

const Attempt = Sim.Attempt;
const GpuLane = gpu_step.GpuLane;
const LaneOutcome = tran.LaneOutcome;
const Options = tran.Options;
const deviceLoop = gpu_step.deviceLoop;
const deviceStepOk = gpu_step.deviceStepOk;
const forLanes = gpu_step.forLanes;
const forLanesThreads = gpu_step.forLanesThreads;
const gpuError = gpu_step.gpuError;
const laneStep = gpu_step.laneStep;
const newtonOptions = Sim.newtonOptions;
const run = tran.run;
const runEach = tran.runEach;

/// `runLanes` on the GPU: any
/// number of `.tran` queries on one pattern, each its own `Sim` on the host
/// (step control, breakpoints, measures) over one `GpuLanes` that holds
/// every lane's numerics. Each round: every live lane `begin`s, the device
/// predicts and solves Newton for all of them and computes their step-end
/// numbers, each lane `end`s from those, and the device applies the
/// verdicts. Within tolerance of `runLanes`, not bitwise. Falls back to
/// each lane's `run` when the lanes are not eligible or the device cannot
/// take them; a lane whose device asks to reject a step reruns on the CPU.
/// Asserts `ctxs` is non-empty and as long as `out`.
pub fn runGpuLanes(ctxs: []const root.RunCtx, opts: Options, out: []LaneOutcome) void {
    const g = ctxs.len;
    std.debug.assert(g >= 1 and out.len == g);
    if (comptime !gpu_lanes.available) return runEach(ctxs, opts, out);
    const gpa = ctxs[0].scratch_allocator;
    const runs = gpa.alloc(Run, g) catch return runEach(ctxs, opts, out);
    defer gpa.free(runs);
    const live = gpa.alloc(bool, g) catch return runEach(ctxs, opts, out);
    defer gpa.free(live);
    @memset(live, false);
    var n_live: usize = 0;
    const t_init = gpu_lanes.Prof.now();
    for (ctxs, runs, live, out) |*c, *r, *on, *o| {
        r.init(c, opts) catch |e| {
            o.* = e;
            continue;
        };
        on.* = true;
        n_live += 1;
    }
    defer for (runs, live) |*r, on| if (on) r.deinit();
    if (n_live == 0) return;
    const first = std.mem.indexOfScalar(bool, live, true).?;
    const refusal = gpuLaneRefusal(ctxs[first].circuit, opts);
    if (refusal) |why| if (ctxs[first].circuit.knobs.gpu_stats != .off) std.debug.print("gpu-lanes: off ({s})\n", .{why});
    var eligible = refusal == null;
    for (ctxs, live) |c, on| if (on and c.circuit.n != ctxs[first].circuit.n) {
        eligible = false;
    };
    const ckts = gpa.alloc(*root.Circuit, g) catch null;
    defer if (ckts) |cs| gpa.free(cs);
    const sims = gpa.alloc(*Sim, g) catch null;
    defer if (sims) |ss| gpa.free(ss);
    // Only the unknowns a lane's `record` and `.meas` feed read come down
    // per round: under measures-only a few columns, not the deck's whole
    // probe list. `GpuLanes` borrows it, so it is freed after `d.deinit`.
    const all_probes = ctxs[first].probes;
    // The `.meas` passes run on the device when every lane can:
    // device step control, a measure bound only on all cards (the device
    // knows when all its passes are over, not one card), and every card a
    // one-row pass (`measure.Stream.deviceScans`). Their columns then stay
    // there.
    const dev_meas = eligible and deviceStepOk(ctxs[first].circuit, opts) and opts.point.abort_measure.len == 0 and devMeasOk(gpa, runs, live) and
        !(@import("builtin").is_test and gpu_lanes.test_access.host_meas);
    if (@import("builtin").is_test and dev_meas) _ = gpu_lanes.test_access.dev_meas.fetchAdd(1, .monotonic);
    const probes = laneProbes(gpa, &runs[first], !dev_meas) catch all_probes;
    defer if (probes.ptr != all_probes.ptr) gpa.free(probes);
    const dev: ?*gpu_lanes.GpuLanes = if (!eligible or ckts == null or sims == null or n_live != g) null else blk: {
        for (ckts.?, sims.?, runs) |*c, *s, *r| {
            c.* = r.sim.ckt;
            s.* = &r.sim;
        }
        break :blk gpu_lanes.GpuLanes.init(gpa, ckts.?, sims.?, probes) catch |e| {
            std.log.warn("gpu lanes: {s}; running {d} transients on the CPU", .{ @errorName(e), g });
            break :blk null;
        };
    };
    if (ctxs[first].circuit.knobs.gpu_stats != .off) std.debug.print("gpu-lanes: setup {d:.1} ms (Run.init and GpuLanes.init)\n", .{@as(f64, @floatFromInt(gpu_lanes.Prof.now() - t_init)) / 1e6});
    const d = dev orelse {
        // Each lane alone, from a fresh `Run`.
        for (runs, live, ctxs, out) |*r, *on, *c, *o| if (on.*) {
            r.deinit();
            on.* = false;
            o.* = run(c, opts);
        };
        return;
    };
    defer d.deinit();
    d.prof.on = ctxs[first].circuit.knobs.gpu_stats != .off;
    if (d.prof.on) std.debug.print("gpu-lanes: {d} lanes\n", .{g});

    const atts = gpa.alloc(Attempt, g) catch |e| return GpuLane.failAll(runs, live, out, e);
    defer gpa.free(atts);
    const nr = gpa.alloc(gpu_lanes.LaneNewton, g) catch |e| return GpuLane.failAll(runs, live, out, e);
    defer gpa.free(nr);
    const nr_opts = newtonOptions(opts);
    const Lane = GpuLane;
    // The two host passes per round, one lane per call. Each touches only
    // lane l's run, outcome and slots of `d`, so `forLanes` may split them.
    const Pass = struct {
        runs: []Run,
        live: []bool,
        out: []LaneOutcome,
        atts: []Attempt,
        nr: []const gpu_lanes.LaneNewton,
        d: *gpu_lanes.GpuLanes,
        ctxs: []const root.RunCtx,
        opts: Options,

        fn begin(p: *const @This(), l: usize) void {
            if (!p.live[l]) return;
            const r = &p.runs[l];
            const maybe = r.sim.begin(p.d.laneOps(l)) catch |e| return Lane.kill(p.runs, p.live, p.out, l, e);
            p.atts[l] = maybe orelse {
                const ok = r.sim.finish() catch |e| return Lane.kill(p.runs, p.live, p.out, l, e);
                return Lane.done(p.runs, p.live, p.out, l, ok);
            };
        }

        fn end(p: *const @This(), l: usize) void {
            if (!p.live[l] or p.d.round[l] == 0) return;
            const r = &p.runs[l];
            if (p.nr[l].reject) {
                // The device asked for `reject_at` or a finite
                // `$bound_step`, which the lanes do not take: this lane
                // reruns alone on the CPU.
                r.deinit();
                p.live[l] = false;
                p.d.detached[l] = true;
                p.out[l] = run(&p.ctxs[l], p.opts);
                return;
            }
            const v = r.sim.end(&p.atts[l], p.nr[l].result, p.d.laneOps(l)) catch |e| return Lane.kill(p.runs, p.live, p.out, l, e);
            switch (v) {
                .go => {},
                .stop => {
                    const ok = r.sim.finish() catch |e| return Lane.kill(p.runs, p.live, p.out, l, e);
                    Lane.done(p.runs, p.live, p.out, l, ok);
                },
                .underflow => Lane.done(p.runs, p.live, p.out, l, false),
            }
        }
    };
    const pass: Pass = .{ .runs = runs, .live = live, .out = out, .atts = atts, .nr = nr, .d = d, .ctxs = ctxs, .opts = opts };
    // Threaded passes cannot call the worker's checkpoint, which parks the
    // analysis thread: the lanes run without it and the driver makes one
    // per round for the batch, from the first live lane. Every lane of a
    // batch carries the batch's one callback (`executeGpuBatch`).
    const threaded = forLanesThreads(g) > 1;
    const progress = ctxs[first].circuit.progress;
    if (threaded) for (ctxs) |c| {
        c.circuit.progress = null;
    };
    defer if (threaded) for (ctxs) |c| {
        c.circuit.progress = progress;
    };
    if (deviceStepOk(ctxs[first].circuit, opts)) device: {
        const steps = gpa.alloc(LK.LaneStep, g) catch break :device;
        defer gpa.free(steps);
        for (runs, steps) |*r, *st| st.* = laneStep(&r.sim, opts, nr_opts.max_iter);
        d.initSteps(steps) catch |e| {
            std.log.warn("gpu lanes: {s}; keeping step control on the host", .{@errorName(e)});
            if (dev_meas) return GpuLane.failAll(runs, live, out, gpuError(e));
            break :device;
        };
        if (dev_meas) initDevMeasures(gpa, d, runs, opts.t_start) catch |e| return GpuLane.failAll(runs, live, out, gpuError(e));
        return deviceLoop(.{ .runs = runs, .live = live, .out = out, .ctxs = ctxs, .opts = opts, .d = d, .gpa = gpa, .meas = dev_meas }, nr_opts, progress);
    }
    while (std.mem.indexOfScalar(bool, live, true)) |lead| {
        @memset(d.round, 0);
        for (d.sc) |*sc| sc.flags = 0;
        if (threaded) if (progress) |cb| {
            const s = &runs[lead].sim;
            if (s.st.attempts != 0) cb.checkpoint(s.event(s.dt)) catch |e| return Lane.failAll(runs, live, out, e);
        };
        forLanes(g, &pass, Pass.begin);
        // `Ops.attempt` marks every lane that attempts this round.
        if (std.mem.indexOfScalar(u8, d.round, 1) == null) break;
        d.prof.rounds += 1;
        d.prof.lap(.host);
        d.predictRound() catch |e| return Lane.failAll(runs, live, out, gpuError(e));
        d.prof.lap(.predict);
        d.newton(nr_opts, nr) catch |e| return Lane.failAll(runs, live, out, gpuError(e));
        d.prof.lap(.newton);
        d.probe() catch |e| return Lane.failAll(runs, live, out, gpuError(e));
        d.prof.lap(.probe);
        forLanes(g, &pass, Pass.end);
        d.prof.lap(.host);
        d.apply() catch |e| return Lane.failAll(runs, live, out, gpuError(e));
        d.prof.lap(.apply);
    }
}

// ---- private ----
/// Why `runGpuLanes` cannot serve this circuit, after `Run.init`, and these
/// options, or null when it can: every batch needs a device payload and no
/// host-side step hook (a `$bound_step` is read on the device and hands its
/// lane back), and none of the features whose step-end numbers the device
/// does not compute (phase 4 G7, plus Newton hooks the lane kernels do not
/// run: `check_convergence`, `advance_iteration` and a load check).
fn gpuLaneRefusal(ckt: *const root.Circuit, opts: Options) ?[]const u8 {
    if (!gpu_lanes.available) return "no GPU images";
    if (opts.uic or opts.snapshot or opts.sources_at_zero or opts.tangents.len != 0) return "uic, snapshot, OPtran or tangents";
    if (integrator.order(opts.method) > 2) return "gear-3..6";
    if (ckt.vpi != null or ckt.display != null or ckt.load_check != null) return "VPI, display tasks or a load check";
    if (ckt.minDelay() != null) return "a delay line";
    if (ckt.has_charge and (ckt.knobs.no_qtape or ckt.qTapeLen() == 0)) return "no charge tape";
    for (ckt.batches) |b| {
        const h = b.hooks;
        const why: ?[]const u8 = if (h.gpu_payload == null)
            "no device payload"
        else if (h.commit_state != null)
            "commit_state"
        else if (h.predict_first_iterate != null)
            "predict_first_iterate"
        else if (h.check_convergence != null)
            "check_convergence"
        else if (h.advance_iteration != null)
            "advance_iteration"
        else if (std.mem.eql(u8, b.type_name, "kinduc"))
            "coupled inductors"
        else if (h.gpu_payload.?(b.ctx).lim_kernel.len == 0 and (h.update_state != null or h.state_ctl != null))
            "host-stepped" // state hooks without a device state kernel
        else
            null;
        if (why) |w| {
            if (ckt.knobs.gpu_stats != .off) std.debug.print("gpu-lanes: {s}: {s}\n", .{ b.type_name, w });
            return b.type_name;
        }
    }
    return null;
}

/// Whether every live lane measures with a stream whose cards all run on
/// the device (`measure.Stream.deviceScans`); the deck is shared, so the
/// first lane answers for all.
fn devMeasOk(gpa: std.mem.Allocator, runs: []Run, live: []const bool) bool {
    var first: ?*Run = null;
    for (runs, live) |*r, on| if (on) {
        if (r.stream == null) return false;
        if (first == null) first = r;
    };
    const sm = (first orelse return false).stream.?;
    const got = (sm.deviceScans(gpa) catch return false) orelse return false;
    defer gpa.free(got.scans);
    defer gpa.free(got.cols);
    // `initDevMeasures` takes lane 0's columns for all: every lane must
    // be at the same stage of the same cards.
    for (runs, live) |*r, on| if (on and r != first.?) {
        const other = (r.stream.?.deviceScans(gpa) catch return false) orelse return false;
        defer gpa.free(other.scans);
        defer gpa.free(other.cols);
        if (got.cols.len != other.cols.len) return false;
        for (got.cols, other.cols, got.scans, other.scans) |*a, *b, x, y| if (!a.eql(b) or x.kind != y.kind) return false;
    };
    return true;
}

/// The unknowns lane `r` reads from an accepted row: its recorded probes
/// and, with `fed_rows`, its `.meas` stream's columns, sorted and without
/// repeats. Every lane of a batch shares the deck, so the first lane's set
/// serves all. The caller frees the result with `gpa`.
fn laneProbes(gpa: std.mem.Allocator, r: *const Run, fed_rows: bool) ![]u32 {
    const rec = r.sim.probes;
    const fed: []const u32 = if (!fed_rows) &.{} else if (r.stream) |sm| sm.nodes else &.{};
    const out = try gpa.alloc(u32, rec.len + fed.len);
    errdefer gpa.free(out);
    @memcpy(out[0..rec.len], rec);
    @memcpy(out[rec.len..], fed);
    std.mem.sortUnstable(u32, out, {}, std.sort.asc(u32));
    var k: usize = 0;
    for (out, 0..) |p, i| if (i == 0 or p != out[k - 1]) {
        out[k] = p;
        k += 1;
    };
    return gpa.realloc(out, k);
}

/// Every lane's passes, lane-major, onto the device (`initMeasures`).
fn initDevMeasures(gpa: std.mem.Allocator, d: *gpu_lanes.GpuLanes, runs: []Run, t_start: f64) !void {
    const N = runs.len;
    var cols: []measure.passes.DevColumns = &.{};
    defer gpa.free(cols);
    var all: []measure.passes.Dev = &.{};
    defer gpa.free(all);
    for (runs, 0..) |*r, l| {
        const got = (try r.stream.?.deviceScans(gpa)) orelse return error.OutOfMemory;
        defer gpa.free(got.scans);
        defer gpa.free(got.cols);
        if (l == 0) {
            cols = try gpa.dupe(measure.passes.DevColumns, got.cols);
            all = try gpa.alloc(measure.passes.Dev, got.scans.len * N);
        }
        std.debug.assert(got.scans.len == cols.len);
        for (got.scans, 0..) |sc, k| all[k * N + l] = sc;
    }
    try d.initMeasures(all, cols, t_start);
}
