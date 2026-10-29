//! Periodic steady state by shooting Newton: a circuit at its DC point and
//! `Options` in, one period of the probes out, as point-major rows. Finds
//! x0 with phi(x0) = x(T; x0) - x0 = 0, integrating one period with
//! fixed-step trapezoid (`period.zig`) under the shooting state machine
//! (`Shoot.zig`). This file is the query spine: one solve, the CPU lane
//! batch and the GPU lane batch.
//!
//! An autonomous circuit (an oscillator, `Options.osc_node` set) has no
//! period to impose. There T joins the unknowns and the oscillator node's
//! x0 leaves them: its value stays pinned at t = 0, which fixes the phase
//! the time shift would otherwise leave free. The Jacobian keeps its n x n
//! shape, the osc column holding d(phi)/d(ln T) instead of d(phi)/d(x0[osc]).
//!
//! Cites: Verilog-A LRM §4.6.1 (sources follow their waveform only in the
//! transient phase).
//!
//! Contents:
//! - `pub const Options = @import("core").query.Pss`
//! - `pub const SolveResult = struct`
//! - `pub const LaneError = root.ErrorOf(run)`
//! - `pub const LaneOutcome = LaneError!root.Result`
//! - `pub const test_access` (tests only)
//! - `pub fn solve(ckt: *root.Circuit, x_dc: []const f64, probes: []const u32, wave: []f64, options: Options, allocator: std.mem.Allocator) !SolveResult`
//! - `pub fn run(ctx: *const root.RunCtx, opts: Options) !root.Result`
//! - `pub fn runLanes(comptime W: usize, ctxs: []const root.RunCtx, opts: Options, out: []LaneOutcome) void`
//! - `pub fn runGpuLanes(ctxs: []const root.RunCtx, opts: Options, out: []LaneOutcome) void`

const std = @import("std");
const converger = @import("solver").converger;
const solver = @import("solver");
const root = @import("../types.zig");
const gpu_lanes = @import("../gpu/lanes.zig");
const period = @import("period.zig");
const Shoot = @import("Shoot.zig");
const Lanes = solver.lanes.Lanes;
const PeriodHook = period.PeriodHook;
const integrateOnePeriod = period.integrateOnePeriod;
const periodAccept = period.periodAccept;
const periodHook = period.periodHook;
const periodStart = period.periodStart;
const stepTime = period.stepTime;

/// The `.pss` query. With `osc_node` set, `period` is only the first guess.
pub const Options = @import("core").query.Pss;

/// Outcome of a periodic solve (shooting, HB, multi-tone HB and qpss).
pub const SolveResult = struct {
    converged: bool,
    /// Newton (shooting or spectral) iterations taken.
    iterations: u16,
    /// Infinity norm of the final residual; inf when an integration failed.
    residual_norm: f64,
};

/// Every error `run` returns.
pub const LaneError = root.ErrorOf(run);

/// One lane's `run` outcome.
pub const LaneOutcome = LaneError!root.Result;

/// Private implementation access for the analysis test suite; void outside tests.
pub const test_access = if (@import("builtin").is_test) .{
    .krylov_threshold = Shoot.krylov_threshold,
} else {};

/// Shooting Newton from x_dc. A non-empty `wave`, sized
/// (n_samples+1) * (1 + probes.len), receives point-major rows
/// [t, v(probes)...] of one period from the final x0, converged or not; it is
/// left zeroed if that integration fails. An autonomous solve starts with
/// `startOscillator` and returns its period as the last row's t;
/// error.OscillatorDidNotStart when the circuit does not oscillate.
/// Allocates O(n^2) below `krylov_threshold` unknowns, O(n * gmres_restart)
/// past it, and O(n * n_samples) more to start an oscillator, all freed
/// before return. Asserts `wave` is empty or that size. A period whose
/// integration fails ends the solve unconverged with an infinite residual,
/// not an error; the errors are OutOfMemory, QueryCancelled, error.Singular
/// from the dense Jacobian, and error.OscillatorDidNotStart.
pub fn solve(
    ckt: *root.Circuit,
    x_dc: []const f64,
    probes: []const u32,
    wave: []f64,
    options: Options,
    allocator: std.mem.Allocator,
) !SolveResult {
    var s: Shoot = undefined;
    try s.init(ckt, x_dc, probes, wave, options, allocator);
    defer s.deinit();
    return s.runAlone();
}

/// Contract entry: shoot from ctx.x_op and return one period as point-major
/// rows (time, probes...), n_samples+1 of them. Non-convergence prints a
/// warning and still returns the last period.
pub fn run(ctx: *const root.RunCtx, opts: Options) !root.Result {
    var shot: Shoot = undefined;
    var names: []const []const u8 = undefined;
    var data: []f64 = undefined;
    try start(ctx, opts, &shot, &names, &data);
    defer shot.deinit();
    errdefer ctx.allocator.free(data);
    return finish(try shot.runAlone(), names, data, opts);
}

/// `run` for the lanes of one batch: `.pss` queries on one pattern, each
/// from its own operating point and variant row (#210). Each lane keeps
/// its own shooting loop, FD columns, monodromy LU, GMRES and period; the
/// period integrations of a round run together, one lane Newton per time
/// step (`integrateLanes`). A Krylov lane's matvecs run scalar inside its
/// own `done`. Lane l's result lands in `out[l]`, bitwise `run` on
/// `ctxs[l]`. Under a JFNK pin or `-Dcertify` every lane runs `run`.
/// Asserts 1 <= `ctxs.len` <= W.
pub fn runLanes(comptime W: usize, ctxs: []const root.RunCtx, opts: Options, out: []LaneOutcome) void {
    const g = ctxs.len;
    std.debug.assert(g >= 1 and g <= W and out.len == g);
    const pin = ctxs[0].circuit.knobs.newton.pin;
    if (W == 1 or g == 1 or !(pin == .auto or pin == .direct) or root.Circuit.certify) {
        for (ctxs, out) |*c, *o| o.* = run(c, opts);
        return;
    }
    const L = Lanes(W);
    const S = LaneSys(W);
    var shots: [W]Shoot = undefined;
    var names: [W][]const []const u8 = undefined;
    var data: [W][]f64 = undefined;
    var live: L.Mask = 0;
    for (ctxs, 0..) |*c, l| {
        start(c, opts, &shots[l], &names[l], &data[l]) catch |e| {
            out[l] = e;
            continue;
        };
        live |= @as(L.Mask, 1) << @intCast(l);
    }
    defer {
        var it = L.each(live);
        while (it.next()) |l| shots[l].deinit();
    }
    const kill = struct {
        fn f(sh: []Shoot, m: *L.Mask, o: []LaneOutcome, l: usize, err: LaneError) void {
            o[l] = err;
            sh[l].deinit();
            m.* &= ~(@as(L.Mask, 1) << @intCast(l));
        }
    }.f;
    // Every lane fails with the batch's own error (cancel, out of memory).
    const fail = struct {
        fn f(sh: []Shoot, m: *L.Mask, o: []LaneOutcome, err: LaneError) void {
            var it = L.each(m.*);
            while (it.next()) |l| {
                o[l] = err;
                sh[l].deinit();
            }
            m.* = 0;
        }
    }.f;
    if (live == 0) return;
    const first = @ctz(live);
    var lanes: [W]*root.Circuit = undefined;
    var wss: [W]*converger.Workspace = undefined;
    for (&lanes, &wss, 0..) |*c, *w, l| {
        // Unused lanes repeat the first live one and stay inactive.
        const k = if (l < g and (live >> @intCast(l)) & 1 != 0) l else first;
        c.* = shots[k].ckt;
        w.* = shots[k].ws;
    }
    const gpa = ctxs[first].scratch_allocator;
    // The hook only sizes the value plane here (alpha 0 rewrites `a_vals`
    // as G, which every assemble overwrites); each step sets its own.
    var sys = S.init(gpa, lanes, periodHook(&shots[first].sc, 0, lanes[first].has_charge)) catch |e| return fail(&shots, &live, out, e);
    defer sys.deinit(gpa);
    var lws = converger.LaneWorkspace(W).init(gpa, wss) catch |e| return fail(&shots, &live, out, e);
    defer lws.deinit(gpa);
    const xv = gpa.alloc(L.V, lanes[first].n) catch |e| return fail(&shots, &live, out, e);
    defer gpa.free(xv);

    var reqs: [W]Shoot.Request = undefined;
    while (true) {
        var round: L.Mask = 0;
        var it = L.each(live);
        while (it.next()) |l| {
            const bit = @as(L.Mask, 1) << @intCast(l);
            const r = shots[l].next() catch |e| {
                kill(&shots, &live, out, l, e);
                continue;
            };
            if (r) |req| {
                reqs[l] = req;
                round |= bit;
                continue;
            }
            out[l] = finish(shots[l].result, names[l], data[l], opts);
            shots[l].deinit();
            live &= ~bit;
        }
        if (round == 0) break;
        const ok = integrateLanes(W, &sys, &lws, xv, &shots, &reqs, round) catch |e| return fail(&shots, &live, out, e);
        it = L.each(round);
        while (it.next()) |l| shots[l].done(ok & (@as(L.Mask, 1) << @intCast(l)) != 0) catch |e| kill(&shots, &live, out, l, e);
    }
}

/// Arbitrary-width shooting lanes. Each lane retains the same host Shoot
/// state machine as runLanes; each round's fixed periods run on the device.
/// Unsupported circuits and Newton pins use the scalar path explicitly.
/// Krylov directional integrations inside Shoot.done remain on the host.
pub fn runGpuLanes(ctxs: []const root.RunCtx, opts: Options, out: []LaneOutcome) void {
    std.debug.assert(ctxs.len > 0 and out.len == ctxs.len);
    if (comptime !gpu_lanes.available) {
        for (ctxs, out) |*c, *o| o.* = run(c, opts);
        return;
    }
    const pin = ctxs[0].circuit.knobs.newton.pin;
    var refusal: ?[]const u8 = if (!(pin == .auto or pin == .direct) or root.Circuit.certify) "Newton pin or certification" else null;
    for (ctxs) |c| {
        if (refusal != null) break;
        refusal = gpuRefusal(c.circuit);
        if (c.circuit.n != ctxs[0].circuit.n or c.circuit.nnz != ctxs[0].circuit.nnz) refusal = "different patterns";
    }
    if (refusal) |why| {
        if (ctxs[0].circuit.knobs.gpu_stats != .off) std.debug.print("gpu-pss: off ({s})\n", .{why});
        for (ctxs, out) |*c, *o| o.* = run(c, opts);
        return;
    }
    gpuShots(ctxs, opts, out) catch |err| for (out) |*o| {
        o.* = err;
    };
}
// ---- private ----

/// The lane system one period's step Newtons run on: each lane's own
/// circuit and its own `PeriodHook` (phase 3's per-lane hooks).
fn LaneSys(comptime W: usize) type {
    return solver.lane_system.LaneSystem(W, root.Circuit, PeriodHook);
}

/// `run`'s setup, shared with the lane drivers: the refusals, names and row
/// buffer, then the shooting state in `shot`.
fn start(ctx: *const root.RunCtx, opts: Options, shot: *Shoot, names: *[]const []const u8, data: *[]f64) !void {
    try ctx.circuit.refuseDigital("pss");
    names.* = try root.probeNames(ctx, "time");
    data.* = try ctx.allocator.alloc(f64, (@as(usize, opts.n_samples) + 1) * names.len);
    errdefer ctx.allocator.free(data.*);
    try shot.init(ctx.circuit, ctx.x_op, ctx.probes, data.*, opts, ctx.scratch_allocator);
}

/// The published period. Non-convergence prints a warning and still
/// returns the last period.
fn finish(res: SolveResult, names: []const []const u8, data: []f64, opts: Options) root.Result {
    if (!res.converged)
        std.debug.print("Warning: pss: shooting did not converge (residual {e})\n", .{res.residual_norm});
    return .{
        .plotname = "Periodic Steady State",
        .varnames = names,
        .is_complex = false,
        .npoints = @as(usize, opts.n_samples) + 1,
        .data = data,
    };
}

/// `integrateOnePeriod` for every lane of `m`, lane l on `shots[l]` and
/// request `reqs[l]`: the start, commit/revert, history and wave rows run
/// per lane and each step's Newton runs as lanes, so lane l is bitwise its
/// scalar integration. Returns the lanes whose period completed. Every
/// request takes the same `n_samples` (one batch shares its options; an
/// autonomous lane's own period changes only its t and dt). Cancellation
/// and memory fail every lane.
fn integrateLanes(
    comptime W: usize,
    sys: *LaneSys(W),
    lws: *converger.LaneWorkspace(W),
    xv: []Lanes(W).V,
    shots: *[W]Shoot,
    reqs: *const [W]Shoot.Request,
    m: Lanes(W).Mask,
) !Lanes(W).Mask {
    const L = Lanes(W);
    const S = LaneSys(W);
    const steps: usize = reqs[@ctz(m)].opts.n_samples;
    var it = L.each(m);
    while (it.next()) |l| {
        std.debug.assert(reqs[l].opts.n_samples == steps);
        periodStart(shots[l].ckt, &shots[l].sc, reqs[l].x, reqs[l].probes, reqs[l].wave, reqs[l].opts);
    }
    var live = m;
    var hooks: [W]PeriodHook = undefined;
    var times: [W]f64 = @splat(0);
    for (0..steps) |k| {
        if (!L.any(live)) break;
        it = L.each(live);
        while (it.next()) |l| {
            const o = reqs[l].opts;
            const ckt = shots[l].ckt;
            const dt = o.period / @as(f64, @floatFromInt(steps));
            times[l] = stepTime(o, k);
            hooks[l] = periodHook(&shots[l].sc, 2.0 / dt, ckt.has_charge);
            sys.hooks[l] = hooks[l];
            ckt.setSimState(.{ .t = times[l], .dt = dt, .kind = .tran });
            S.putLane(xv, reqs[l].x, l);
        }
        sys.active = live;
        const o0 = reqs[@ctz(live)].opts;
        const res = converger.newtonLanes(W, sys, lws, xv, times, .{
            .max_iter = o0.max_newton_iter,
            .abstol = o0.newton_tol,
        }, S.LaneHook{});
        // As `converger.run` ends each lane's solve.
        it = L.each(live);
        while (it.next()) |l| shots[l].ckt.clearLimits();
        const r = try res;
        it = L.each(live);
        while (it.next()) |l| {
            S.getLane(reqs[l].x, xv, l);
            const bit = @as(L.Mask, 1) << @intCast(l);
            const ok = r.converged & bit != 0 and r.singular & bit == 0;
            if (!periodAccept(shots[l].ckt, &hooks[l], reqs[l].x, reqs[l].probes, reqs[l].wave, k, times[l], ok)) live &= ~bit;
        }
    }
    return live;
}

/// The fixed-period GPU driver has no host callbacks or stateful-charge
/// rebase. State flips discovered during a period rerun it on the host
/// from its unchanged start.
fn gpuRefusal(ckt: *const root.Circuit) ?[]const u8 {
    if (!gpu_lanes.available) return "no GPU images";
    if (ckt.vpi != null or ckt.display != null or ckt.load_check != null) return "host callbacks";
    if (ckt.minDelay() != null or ckt.has_state_q) return "stateful charge or delay";
    for (ckt.batches) |b| {
        const h = b.hooks;
        if (h.gpu_payload == null or h.commit_state != null or h.predict_first_iterate != null or
            h.check_convergence != null or h.advance_iteration != null or
            std.mem.eql(u8, b.type_name, "kinduc")) return b.type_name;
        const p = h.gpu_payload.?(b.ctx);
        // Timer-only sources carry generated State bytes but their host
        // hooks are intentionally null (device/eval.zig skipsTimerState).
        // Gate the work that runs, not the presence of unused storage.
        if ((h.update_state != null and p.lim_kernel.len == 0) or
            (h.state_ctl != null and p.ctl_kernel.len == 0))
            return "device shooting state";
    }
    return null;
}

fn gpuShots(ctxs: []const root.RunCtx, opts: Options, out: []LaneOutcome) LaneError!void {
    const gpa = ctxs[0].scratch_allocator;
    const N = ctxs.len;
    const shots = try gpa.alloc(Shoot, N);
    defer gpa.free(shots);
    const live = try gpa.alloc(bool, N);
    defer gpa.free(live);
    @memset(live, false);
    defer for (shots, live) |*s, on| if (on) s.deinit();
    const names = try gpa.alloc([]const []const u8, N);
    defer gpa.free(names);
    const data = try gpa.alloc([]f64, N);
    defer gpa.free(data);
    const requests = try gpa.alloc(Shoot.Request, N);
    defer gpa.free(requests);
    const periods = try gpa.alloc(gpu_lanes.FixedLane, N);
    defer gpa.free(periods);
    const indices = try gpa.alloc(usize, N);
    defer gpa.free(indices);
    const ok = try gpa.alloc(bool, N);
    defer gpa.free(ok);
    for (ctxs, 0..) |*c, l| {
        start(c, opts, &shots[l], &names[l], &data[l]) catch |err| {
            out[l] = err;
            continue;
        };
        live[l] = true;
    }
    while (true) {
        var count: usize = 0;
        for (shots, live, 0..) |*s, *on, l| {
            if (!on.*) continue;
            const r = s.next() catch |err| {
                out[l] = err;
                s.deinit();
                on.* = false;
                continue;
            };
            if (r) |req| {
                requests[count] = req;
                indices[count] = l;
                periodStart(s.ckt, &s.sc, req.x, req.probes, req.wave, req.opts);
                periods[count] = .{
                    .ckt = s.ckt,
                    .x = req.x,
                    .q_prev = s.sc.q_prev,
                    .i_prev = s.sc.i_prev,
                    .period = req.opts.period,
                    .n_samples = req.opts.n_samples,
                    .probes = req.probes,
                    .wave = req.wave,
                };
                count += 1;
            } else {
                out[l] = finish(s.result, names[l], data[l], opts);
                s.deinit();
                on.* = false;
            }
        }
        if (count == 0) break;
        const options: converger.Options = .{ .max_iter = opts.max_newton_iter, .abstol = opts.newton_tol };
        gpu_lanes.integrateFixed(gpa, periods[0..count], options, ok[0..count]) catch |err| switch (err) {
            error.QueryCancelled, error.OutOfMemory => |e| return e,
            else => {
                std.log.warn("gpu pss: {s}; running {d} periods on the CPU", .{ @errorName(err), count });
                for (indices[0..count], requests[0..count], ok[0..count]) |l, req, *done| {
                    const s = &shots[l];
                    done.* = try integrateOnePeriod(s.ckt, s.ws, &s.sc, req.x, req.probes, req.wave, req.opts);
                }
            },
        };
        for (indices[0..count], ok[0..count]) |l, done| shots[l].done(done) catch |err| {
            out[l] = err;
            shots[l].deinit();
            live[l] = false;
        };
    }
}
