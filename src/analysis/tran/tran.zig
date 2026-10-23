//! Transient analysis: Newton per timestep on A = G + ag0*C, with ngspice's
//! LTE step control, order promotion and breakpoint landing. The companion
//! residual uses the exact q(x) plane and the Jacobian the analytic C plane.
//! In: a circuit at its operating point (or the `.ic` values under uic) and
//! `Options`. Out: the probes at every accepted point, as `Waveform` rows.
//! Cites ngspice dctran.c, traninit.c, cktsetbk.c, ckttrunc.c, niconv.c,
//! indload.c, capload.c and AlmostEqualUlps (maths/misc/equality.c);
//! Verilog-AMS LRM §4.5, §5.10.2, §9.10 and §9.17.2; #161;
//! dev/analysis/transient-integration.md.
const std = @import("std");
const root = @import("../types.zig");
const converger = @import("solver").converger;
const integrator = @import("integrator.zig");
const num = @import("core").numerics;

const W = std.simd.suggestVectorLength(f64) orelse 8;

// Re-exported so the post-processors and tests import one file; each is
// documented in types.zig.
const tran_types = @import("types.zig");

/// The swept plot's name and scale column (`session.schemaOf` reports them).
pub const plotname = "Transient Analysis";
pub const scale = "time";
const Method = tran_types.Method;
/// See types.zig `Options`.
pub const Options = tran_types.Options;
/// See types.zig `Waveform`.
pub const Waveform = tran_types.Waveform;
/// See types.zig `Column`.
pub const Column = tran_types.Column;
/// See types.zig `initialCapacity`.
pub const initialCapacity = tran_types.initialCapacity;
const measure = @import("measure");
const Stream = measure.Stream;

/// Writes the MODEINITPRED first iterate, trial = cur + xfact·(cur - prev),
/// into `trial[0..cur.len]`; `prev` must be at least as long as `cur`.
fn predict(trial: []f64, cur: []const f64, prev: []const f64, xfact: f64) void {
    for (trial[0..cur.len], cur, prev[0..cur.len]) |*t, c, p| t.* = c + xfact * (c - p);
}

/// ZP_TRAN_STATS step-economics counters: wall time in a slow transient is
/// attempts x Newton iterations x eval cost, and these say which factor.
const Stats = struct {
    attempts: u64 = 0,
    nr_iters: u64 = 0,
    rej_newton: u64 = 0,
    rej_state: u64 = 0,
    rej_vpi: u64 = 0,
    rej_lte: u64 = 0,
    order_drops: u64 = 0,
    bp_landings: u64 = 0,
    first_dt: f64 = 0,
    max_dt: f64 = 0,
    /// Accepted steps per integration order 1..6.
    by_order: [6]u64 = @splat(0),
};

/// Newton hook: companion RHS from the q plane, matrix G + ag0*C.
const TranHook = integrator.Hook(null);

/// The t = 0 row: `x` itself after an operating point. Under uic no
/// operating point ran and `x` holds only the `.ic` values, which break KVL
/// (V1 = 2 V read as 0 V, #161), so the row is instead one step solve over
/// `hook`'s vanishing dt into `scratch`: every charge and flux stays at its
/// initial value and the rest of the network is solved around them. The
/// device state it leaves is reverted. Falls back to `x` when that solve
/// fails. ngspice prints no t = 0 row under uic at all.
fn uicStart(ckt: *root.Circuit, ws: *converger.Workspace, x: []const f64, scratch: []f64, hook: TranHook, options: Options) ![]const f64 {
    if (!options.uic) return x;
    ckt.setSimState(.{ .t = 0, .dt = 1 / hook.c.ag0, .kind = .tran, .initial_step = true });
    defer ckt.setSimState(.{ .t = 0, .dt = 0, .kind = .tran, .initial_step = true });
    num.copySimd(scratch, x);
    var nr_opts = converger.optionsFromTolerances(options.tol, options.tol.itl4);
    nr_opts.residual_tol = std.math.inf(f64);
    ckt.reject_at = null;
    const nr = converger.run(ckt, ws, scratch, 0, nr_opts, hook) catch |err| switch (err) {
        error.QueryCancelled, error.OutOfMemory => return err,
        else => null, // else: a numerical failure keeps the `.ic` row
    };
    _ = ckt.stateCtl(.revert);
    return if (nr != null and nr.?.converged) scratch else x;
}

/// The first-order trajectories of `Options.tangents`, one per variant row,
/// on the base run's own time grid (dev/analysis/optimize.md, "Direct
/// sensitivities"). At the operating point and at every accepted step,
/// `advance` moves each row's state by -A⁻¹ (R(x, row) - R(x)): R the
/// step's residual, with the row's parameter values, charge history and
/// device state (Meyer `$prev`, delay lines) on the row side, A the step's
/// matrix at the base solution x. Each row keeps its device state, with
/// its parameter values, as a `Circuit.SavedState` swapped in for its turn,
/// so a GPU-resident circuit, which `restoreState` leaves alone, turns every
/// row to NaN, as do a singular A and a row that re-wires a device.
const Tangents = struct {
    rows: []const u32,
    variants: @import("core").Variants,
    params: []const root.ParamRef,
    /// Per row, 5 n values: x, then the charge ring q0, q1, q2, then the
    /// dynamic current, as the base run's `q_hist` and `i_prev`.
    state: []f64,
    dead: []bool,
    wf: []Waveform,
    /// Each row's device state between turns, then the base run's.
    devices: []root.Circuit.SavedState,
    /// The base residual, n values.
    r0: []f64,
    /// Each row's `.meas` cards under early stop, else empty.
    streams: []Stream = &.{},

    /// The companion terms of the step the base run just accepted.
    const Step = struct { method: Method, c: integrator.Coeffs, q1: []const f64, q2: []const f64, i: []const f64 };

    fn init(ctx: *const root.RunCtx, opts: Options, n_probes: u32, gpa: std.mem.Allocator) !Tangents {
        const ckt = ctx.circuit;
        const k = opts.tangents.len;
        var tg: Tangents = .{
            .rows = opts.tangents,
            .variants = ctx.variants,
            .params = try ckt.collectParams(),
            .state = &.{},
            .dead = &.{},
            .wf = &.{},
            .devices = &.{},
            .r0 = &.{},
        };
        errdefer tg.deinit(gpa);
        tg.state = try gpa.alloc(f64, 5 * ckt.n * k);
        tg.dead = try gpa.alloc(bool, k);
        @memset(tg.dead, ckt.gpu != null);
        tg.r0 = try gpa.alloc(f64, ckt.n);
        tg.wf = try gpa.alloc(Waveform, k);
        for (tg.wf, 0..) |*w, i| w.* = Waveform.init(gpa, n_probes, initialCapacity(opts)) catch |e| {
            tg.wf = tg.wf[0..i];
            return e;
        };
        tg.devices = try gpa.alloc(root.Circuit.SavedState, k + 1);
        for (tg.devices, 0..) |*d, i| d.* = ckt.saveState(gpa) catch |e| {
            tg.devices = tg.devices[0..i];
            return e;
        };
        return tg;
    }

    fn deinit(tg: *Tangents, gpa: std.mem.Allocator) void {
        for (tg.streams) |*s| s.deinit();
        gpa.free(tg.streams);
        for (tg.wf) |*w| w.deinit();
        for (tg.devices) |d| d.deinit(gpa);
        gpa.free(tg.wf);
        gpa.free(tg.devices);
        gpa.free(tg.dead);
        gpa.free(tg.state);
        gpa.free(tg.r0);
    }

    /// Moves every row to the base solution `x` at `t`: after a static
    /// solve when `step` is null (the operating point, where each row takes
    /// its parameter values and seeds its charge history), else after that
    /// companion step, its device state staged but not committed. Leaves the
    /// base run's device state and parameters as it found them; the first
    /// call drops the constant-Jacobian baseline (`Circuit.recompute`).
    fn advance(tg: *Tangents, ckt: *root.Circuit, ws: *converger.Workspace, a_vals: []f64, x: []const f64, t: f64, step: ?Step) !void {
        const n = ckt.n;
        const charge = ckt.has_charge;
        const base = tg.devices[tg.rows.len];
        // The rows' accepted points print nothing.
        const display = ckt.display;
        ckt.display = null;
        defer ckt.display = display;
        ckt.storeState(base);
        defer ckt.restoreState(base);
        // The device state the step's Newton solved under.
        _ = ckt.stateCtl(.revert);
        // A full stamp: the baseline holds the const-Jacobian devices at
        // their nominal values (dc.zig `tangentRows`).
        ckt.eval(x, t);
        num.copySimd(tg.r0, ckt.rhs[0..n]);
        if (step) |s| if (charge) companion(s, tg.r0, ckt.q_vec[0..n], s.q1, s.q2, s.i);
        const vals = if (step != null and charge) blk: {
            ckt.combineGC(step.?.c.ag0, a_vals);
            break :blk a_vals;
        } else ckt.g_vals;
        const factored = if (ws.slv.factor(vals, ckt.solver_execution)) true else |e| switch (e) {
            error.SingularMatrix => false,
            else => return e, // else: only a singular A has an answer (NaN rows)
        };
        for (tg.rows, tg.dead, tg.devices[0..tg.rows.len], 0..) |row, *dead, devices, k| {
            const st = tg.state[5 * n * k ..][0 .. 5 * n];
            const xk = st[0..n];
            const q0 = st[n..][0..n];
            const q1 = st[2 * n ..][0..n];
            const q2 = st[3 * n ..][0..n];
            const ik = st[4 * n ..][0..n];
            if (!factored) dead.* = true;
            if (dead.*) {
                @memset(xk, std.math.nan(f64));
                continue;
            }
            ckt.restoreState(devices);
            if (step == null) {
                const refs, const values = tg.variants.writes(row);
                for (refs, values) |r, v| tg.params[r].set(v);
                ckt.recompute() catch |e| switch (e) {
                    // No first-order answer on this pattern.
                    error.TopologyChanged => {
                        dead.* = true;
                        @memset(xk, std.math.nan(f64));
                        continue;
                    },
                };
            }
            ckt.eval(x, t);
            // xk holds dx until the solve is added to x.
            num.copySimd(xk, ckt.rhs[0..n]);
            if (step) |s| if (charge) companion(s, xk, ckt.q_vec[0..n], q1, q2, ik);
            for (xk, tg.r0) |*d, r| d.* -= r;
            ws.slv.solveNeg(xk, xk);
            for (xk, x) |*xi, xb| xi.* += xb;
            if (step) |s| if (charge) {
                ckt.evalQ(xk, t);
                num.copySimd(q0, ckt.q_vec[0..n]);
                integrator.advanceCurrent(s.method, ik, q0, q1, q2, s.c);
                num.copySimd(q2, q1);
                num.copySimd(q1, q0);
            };
            // The row's accepted point, as the base run commits its own.
            _ = ckt.updateStates(xk);
            _ = ckt.commitStates(xk);
            _ = ckt.stateCtl(.commit);
            if (charge) if (step) |s| {
                if (ckt.has_state_q) {
                    ckt.evalQ(xk, t);
                    integrator.rebaseCurrent(ik, ckt.q_vec[0..n], q1, s.c.ag0);
                    num.copySimd(q1, ckt.q_vec[0..n]);
                }
            } else {
                ckt.evalQ(xk, t);
                num.copySimd(q1, ckt.q_vec[0..n]);
                num.copySimd(q2, q1);
                num.zeroSimd(ik);
            };
            ckt.storeState(devices);
        }
    }

    /// Adds to `out` the step's dynamic current over the history `q1`,
    /// `q2`, `i`, as `TranHook.assemble` adds it to the residual.
    fn companion(s: Step, out: []f64, q: []const f64, q1: []const f64, q2: []const f64, i: []const f64) void {
        switch (s.method) {
            inline else => |m| integrator.companionAt(m, true, out, q, q1, q2, i, s.c),
        }
    }

    fn record(tg: *Tangents, t: f64, probes: []const u32) !void {
        const n = tg.r0.len;
        for (tg.wf, 0..) |*w, k| {
            const xk = tg.state[5 * n * k ..][0..n];
            try w.record(t, xk, probes);
            if (tg.streams.len != 0) tg.streams[k].feed(t, xk, w.data());
        }
    }

    /// `.meas` cards still open over the rows that have values.
    fn open(tg: Tangents) usize {
        var n: usize = 0;
        for (tg.streams, tg.dead) |s, dead| n += if (dead) 0 else s.open;
        return n;
    }
};

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

/// What one attempt asks of Newton: `trial` from the predictor, solved at
/// `t_dev` under `hook` with `nr_opts`. `dts`, `ag` and `c` are the step's
/// own history and coefficients, which `end` reads again.
const Attempt = struct {
    t_dev: f64,
    method: Method,
    gear_hi: bool,
    dts: [7]f64,
    ag: [7]f64,
    c: integrator.Coeffs,
    hook: TranHook,
    nr_opts: converger.Options,
};

/// How `Sim.end` left the run: `.go` attempts again (a retry or the next
/// step), `.stop` ends it early with every card resolved, `.underflow` ends
/// it short of t_stop (dt below `dt_min`, or the VPI retry bound).
const Verdict = enum { go, stop, underflow };

/// One transient between its attempts: `simulateWith`'s state, so the
/// scalar driver (`drive`) and the lane driver (`runLanes`) share every
/// statement but the Newton solve. Pinned: `init` fills it in place.
/// `begin` prepares an attempt, the caller solves it, `end` takes the
/// verdict, and `finish` closes a run that left `begin`'s loop.
/// dev/research/lane-batched-simulation.md §7.
const Sim = struct {
    ckt: *root.Circuit,
    x: []f64,
    probes: []const u32,
    waveform: *Waveform,
    options: Options,
    allocator: std.mem.Allocator,
    tangents: ?*Tangents,
    stream: ?*Stream,
    ws: *converger.Workspace,
    n: usize,
    has_charge: bool,
    x_try: []f64,
    x_prev: []f64,
    top_order: u8,
    n_hist: usize,
    a_vals: []f64,
    row_state: []f64,
    i_prev: []f64,
    q_ring: [9][]f64,
    g_row: []f64,
    n_qt: usize,
    lte_rows: []u32,
    lte_zero: [][2]u32,
    lte_state: []f64,
    qt_i_prev: []f64,
    qt_ring: [9][]f64,
    qt_g: []f64,
    effective_dt_max: f64,
    delmin: f64,
    min_break: f64,
    state_eps: f64,
    echo_bps: [256]f64,
    n_echo: usize,
    echo_td: ?f64,
    uic_bp: ?f64,
    vpi: ?*const @import("device").Library.Vpi,
    vpi_point: @import("device").Library.AnalogPoint,
    cur: []f64,
    trial: []f64,
    prev: []f64,
    t: f64,
    dt: f64,
    dt_old: [6]f64,
    steps: u32,
    stats_on: bool,
    st: Stats,
    order: u8,
    run_order: u8,
    order_run: u32,
    bp_next: ?f64,
    bp_clamped: bool,
    bp_save_dt: f64,
    attempted_dt: f64,
    vpi_again: u32,
    resolved: bool,

    fn qHist(s: *Sim) [][]f64 {
        return s.q_ring[0..s.n_hist];
    }
    fn qtHist(s: *Sim) [][]f64 {
        return s.qt_ring[0..s.n_hist];
    }
    /// The LTE history: per state when the tape is live, per row otherwise.
    /// Same kernel either way, so the row path is the scalar oracle for the
    /// tape path. The slice is the ring, so it follows the rotation.
    fn lteHist(s: *Sim) [][]f64 {
        return if (s.n_qt > 0) s.qtHist() else s.qHist();
    }
    /// The method of the order-1-to-2 promotion probe.
    pub fn promoteMethod(s: *const Sim) Method {
        return methodAt(s.options.method, 2);
    }
    fn lteIp(s: *const Sim) []const f64 {
        return if (s.n_qt > 0) s.qt_i_prev else s.i_prev;
    }
    fn lteG(s: *const Sim) []const f64 {
        return if (s.n_qt > 0) s.qt_g else s.g_row;
    }

    /// Everything `simulateWith` did before its step loop, in its order:
    /// scratch, the charge rings, uic bookkeeping, the baseline, history
    /// seeding, the tangents' operating point, delay-line seeding, the
    /// breakpoint and VPI setup, and the t = 0 row. Leaves the device state
    /// committed at t = 0 and `land_rejects` set; `deinit` clears it.
    fn init(
        s: *Sim,
        ckt: *root.Circuit,
        x: []f64,
        probes: []const u32,
        waveform: *Waveform,
        options: Options,
        allocator: std.mem.Allocator,
        tangents: ?*Tangents,
        stream: ?*Stream,
    ) !void {
        const n: usize = ckt.n;
        const has_charge = ckt.has_charge;

        const ws = try ckt.workspace();
        // A device's `request_reject_at` retries the step ending at that time.
        ckt.land_rejects = true;
        errdefer ckt.land_rejects = false;
        const x_try = try allocator.alloc(f64, n);
        errdefer allocator.free(x_try);
        // The accepted point before `cur`, for the predictor. Equal to `cur`
        // until the first step is accepted, so that step starts from the OP.
        const x_prev = try allocator.alloc(f64, n);
        errdefer allocator.free(x_prev);
        num.copySimd(x_prev, x);

        // Row-plane charge state, one block: the dynamic current, then the
        // charge ring [cur, prev, prev2, ...]. Slot 0 takes the charge of each
        // converged attempt; the companion residual reads slots 1 and 2. The LTE
        // at order k reads k + 2 slots, so the ring holds max(4, top + 2), and
        // a gear-3..6 run keeps its step's `gearHistory` g in one more part.
        const top_order = integrator.order(options.method);
        const n_hist: usize = @max(4, top_order + 2);
        const n_parts: usize = 1 + n_hist + @intFromBool(top_order >= 3);
        const row_state = try allocator.alloc(f64, if (has_charge) n_parts * n else 0);
        errdefer allocator.free(row_state);
        const q_ring = ring(row_state, n_parts);

        // The LTE runs per device charge state, as ngspice's CKTterr does
        // (ckttrunc.c and the per-device trunc routines), not per matrix row:
        // charges that share a node add their divided differences, so a row's
        // slope belongs to no real state. The device `q_tape` holds one charge
        // per `ddt()` site the model leaves in the LTE (`vera_lte`), and the host
        // keeps a second history over it. The companion residual still
        // integrates the row plane.
        //
        // n_qt == 0 (no charge, or a GPU plane-stamp hook that bypasses the host
        // batches) runs the same kernel over the row plane. ZP_NO_QTAPE forces
        // that fallback for A/B runs; `n_qt` in the stats line says which is live.
        const n_qt: usize = if (has_charge and !ckt.knobs.no_qtape) ckt.qTapeLen() else 0;
        // Coupled inductors: ngspice truncates one state per inductor, INDflux =
        // L*i + sum M*i_other (indload.c:72-76; MUT has no trunc routine). The
        // tape holds L*i and each M*i as separate states, which bind where
        // INDflux does not. With a K card, `lteSnap` zeroes the inductor and
        // kinduc tape spans (a flat zero history never binds) and appends the row
        // plane at every inductor current row, since an inductor's branch row
        // sums exactly its INDflux. Both are resolved here, once: `lte_zero`
        // holds the [offset, end) tape spans to clear.
        var lte_rows: []u32 = &.{};
        var lte_zero: [][2]u32 = &.{};
        errdefer allocator.free(lte_rows);
        errdefer allocator.free(lte_zero);
        if (n_qt > 0) for (ckt.batches) |kb| {
            if (!std.mem.eql(u8, kb.type_name, "kinduc")) continue;
            var rows: std.ArrayList(u32) = .empty;
            errdefer rows.deinit(allocator);
            const is_l = try allocator.alloc(bool, n);
            defer allocator.free(is_l);
            @memset(is_l, false);
            ckt.markCurrentRowsOf("inductor", is_l);
            for (is_l, 0..) |l, r| if (l) try rows.append(allocator, @intCast(r));
            lte_rows = try rows.toOwnedSlice(allocator);
            var spans: std.ArrayList([2]u32) = .empty;
            errdefer spans.deinit(allocator);
            var off: u32 = 0;
            for (ckt.batches) |b| {
                const f = b.hooks.q_tape orelse continue;
                const end_ = off + @as(u32, @intCast(f(b.ctx).len));
                if (std.mem.eql(u8, b.type_name, "inductor") or std.mem.eql(u8, b.type_name, "kinduc"))
                    try spans.append(allocator, .{ off, end_ });
                off = end_;
            }
            lte_zero = try spans.toOwnedSlice(allocator);
            break;
        };
        const n_lt = n_qt + lte_rows.len;
        // The same layout over the n_lt LTE states.
        const lte_state = try allocator.alloc(f64, if (n_qt > 0) n_parts * n_lt else 0);
        errdefer allocator.free(lte_state);
        const qt_ring = ring(lte_state, n_parts);

        s.* = .{
            .ckt = ckt,
            .x = x,
            .probes = probes,
            .waveform = waveform,
            .options = options,
            .allocator = allocator,
            .tangents = tangents,
            .stream = stream,
            .ws = ws,
            .n = n,
            .has_charge = has_charge,
            .x_try = x_try,
            .x_prev = x_prev,
            .top_order = top_order,
            .n_hist = n_hist,
            .a_vals = &.{},
            .row_state = row_state,
            .i_prev = row_state[0 .. row_state.len / n_parts],
            .q_ring = q_ring,
            .g_row = q_ring[n_hist],
            .n_qt = n_qt,
            .lte_rows = lte_rows,
            .lte_zero = lte_zero,
            .lte_state = lte_state,
            .qt_i_prev = lte_state[0 .. lte_state.len / n_parts],
            .qt_ring = qt_ring,
            .qt_g = qt_ring[n_hist],
            // Set below, in `simulateWith`'s order.
            .effective_dt_max = undefined,
            .delmin = undefined,
            .min_break = undefined,
            .state_eps = undefined,
            .echo_bps = undefined,
            .n_echo = 0,
            .echo_td = null,
            .uic_bp = null,
            .vpi = null,
            .vpi_point = undefined,
            .cur = x,
            .trial = x_try,
            .prev = x_prev,
            .t = 0,
            .dt = undefined,
            .dt_old = undefined,
            .steps = 0,
            .stats_on = ckt.knobs.tran_stats,
            .st = .{},
            .order = 1,
            .run_order = 1,
            .order_run = 0,
            .bp_next = null,
            .bp_clamped = false,
            .bp_save_dt = options.t_stop / 50.0,
            .attempted_dt = undefined,
            .vpi_again = 0,
            .resolved = false,
        };
        const q_hist = s.qHist();
        const qt_hist = s.qtHist();

        // uic skipped op.solve, so do its device bookkeeping here: latch power-on
        // FSM state under `initial_step` (§5.10.2), commit it, and leave a static
        // kind for the charge seeding below. `.ic`, not `.dc`: the uic start is
        // the transient's ic phase, so waveform sources evaluate at t = 0.
        // An idt's ic is its value at t = 0 (LRM §4.5.4) and the charge the
        // integration starts from, as a capacitor's IC is under ngspice's
        // MODEUIC (capload.c), so it is seeded here rather than left at 0.
        if (options.uic) {
            ckt.setSimState(.{ .kind = .ic, .initial_step = true });
            _ = ckt.stateCtl(.commit);
            ckt.setSimState(.{ .kind = .ic });
            ckt.seedIc(x);
        }
        try ckt.computeBaseline();

        if (has_charge) {
            s.a_vals = try ws.ensureAVals(ckt.nnz);
            num.zeroSimd(s.i_prev);
            num.zeroSimd(s.qt_i_prev);
            // No setSimState first: q_prev must be the charge the operating point
            // saw, so this eval runs in the static state op.solve left behind.
            ckt.eval(x, 0);
            // A flat q(0) history makes every divided difference vanish, so LTE
            // control runs from the first step.
            for (q_hist[1..]) |h| num.copySimd(h, ckt.q_vec[0..n]);
            if (n_qt > 0) {
                lteSnap(ckt, qt_hist[1], lte_zero, lte_rows);
                for (qt_hist[2..]) |h| num.copySimd(h, qt_hist[1]);
            }
        }
        // Still in the operating point's static state, as the seeding above.
        if (tangents) |tg| {
            std.debug.assert(!options.uic);
            try tg.advance(ckt, ws, s.a_vals, x, 0, null);
            try ckt.computeBaseline();
        }

        // Seed absdelay rings with the operating point: the first §4.5.7
        // zHistPush fills the whole ring with (0, v_op), so no delay line reads
        // 0 V for t < td. Needs kind = .tran, the only branch on which the
        // generated core computes the delay inputs (dt = 0 keeps zAbsdelay the
        // DC identity).
        ckt.setSimState(.{ .t = 0, .dt = 0, .kind = .tran, .initial_step = true });
        _ = ckt.commitStates(x);
        _ = ckt.stateCtl(.commit);

        // ngspice tmax defaults to (tstop - tstart)/50; the circuit's minimum
        // delay caps it further.
        s.effective_dt_max = options.requestedCeiling(.api);
        if (ckt.minDelay()) |td_min| s.effective_dt_max = @min(s.effective_dt_max, td_min);
        // The analysis' own CKTminBreak: delmin = 1e-11*maxStep and minBreak =
        // 10*delmin (traninit.c:36, dctran.c:170). Breakpoints closer than this
        // to the current time or to each other merge (dctran.c:636,
        // cktsetbk.c:45). optran.c:419's 5e-5*tmax is the OP transient's rule and
        // would merge away ns-scale source corners.
        s.delmin = 1e-11 * s.effective_dt_max;
        s.min_break = 10.0 * s.delmin;
        // How sharply a device state flip must land before the step is accepted.
        // An espice Newton/FSM tolerance with no ngspice counterpart; the switch
        // fixtures are tuned against it, so it is not min_break.
        s.state_eps = 5e-5 * s.effective_dt_max;

        // Delayed breakpoint echoes, standing in for ngspice traload's t + td
        // breakpoint on a sharp line input: every landed breakpoint re-emits one
        // echo at t + td, and echo landings re-emit in turn (t + 2td, 3td, ...).
        // ponytail: one td (the circuit's minimum delay) for every history
        // device; enumerate per-device delays if mixed-td circuits show edge smear.
        s.echo_td = ckt.minDelay();
        if (s.echo_td) |td_| {
            // t = 0 is itself a breakpoint (source edges often start there).
            s.echo_bps[0] = td_;
            s.n_echo = 1;
        }
        // Under uic ngspice sets a breakpoint at tstep "to reduce ringing of
        // current in devices" (dctran.c:582-584), after sizing the first step,
        // so landing there drops the order once (#166). Past t_stop it is never
        // reached, as CKTsetBreak inserts nothing past CKTfinalTime.
        s.uic_bp = if (options.uic) options.dt_init else null;

        // LRM 12.31.3 sees the transient proper, not the OPtran fallback.
        s.vpi = if (options.sources_at_zero) null else ckt.vpi;
        s.vpi_point = .{ .kind = .tran, .start = options.t_start, .stop = options.t_stop, .max_step = s.effective_dt_max };
        if (s.vpi) |v| v.point(v.ctx, s.vpi_point);

        // tstart suppresses output, never the solve: [0, t_start) is integrated
        // with full history and not recorded.
        if (options.t_start <= 0) {
            const x0 = try uicStart(ckt, ws, x, x_try, .{
                .method = .backward_euler,
                .c = integrator.coeffs(.backward_euler, s.delmin, s.delmin, options.xmu),
                .q_prev = q_hist[1],
                .i_prev = s.i_prev,
                .q_prev2 = q_hist[2],
                .a_vals = s.a_vals,
                .q_cur = {},
                .has_charge = has_charge,
            }, options);
            try waveform.record(0, x0, probes);
            if (stream) |sm| sm.feed(0, x0, waveform.data());
        }
        if (s.vpi) |v| v.accept(v.ctx, true, false);
        if (tangents) |tg| if (options.t_start <= 0) try tg.record(0, probes);

        // ngspice's first step: min(tstep, tstop/100)/10 clamped to tmax outside
        // the min (dctran.c:134), then the t = 0 breakpoint clamp 0.1*breaks[1]
        // (dctran.c:578-586), then the firsttime /10. The operand order sets the
        // phase of the whole accepted grid.
        var dt: f64 = @min(@min(options.dt_init, options.t_stop / 100.0) / 10.0, s.effective_dt_max);
        if (nextBp(ckt, s.echo_bps[0..s.n_echo], null, s.min_break)) |bp0| dt = @min(dt, 0.1 * bp0);
        dt /= 10.0;
        if (options.t_start > 0 and dt > options.t_start) dt = options.t_start;
        s.dt = dt;
        // The accepted steps [dt_prev, dt_prev2, ...], newest first. CKTdeltaOld[]
        // starts at CKTmaxStep (dctran.c:312), which the first divided
        // differences read.
        s.dt_old = @splat(s.effective_dt_max);
        s.attempted_dt = dt;
        if (s.stats_on) std.debug.print(
            "tran-config: dt_init={e:.17} requested_ceiling={e:.17} effective_ceiling={e:.17} initial_dt={e:.17} dt_min={e:.17} method={s} lte={s} uic={} tol={any}\n",
            .{ options.dt_init, options.requestedCeiling(.api), s.effective_dt_max, dt, options.dt_min, @tagName(options.method), if (s.n_qt > 0) "charge-tape" else "row", options.uic, options.tol },
        );
    }

    /// Frees the scratch `init` allocated and clears `land_rejects`.
    fn deinit(s: *Sim) void {
        // Every exit reports, including underflow, cancellation and errors.
        // Output samples are deliberately separate from accepted solver steps.
        if (s.stats_on) {
            const st = s.st;
            std.debug.print(
                "tran-stats: complete={} resolved={} t={e:.17} stop={e:.17} samples={d} n_qt={d} accepted={d} attempts={d} nr_iters={d} rej[newton={d} lte={d} state={d} vpi={d}] order_drops={d} bp_landings={d} by_order={any} first_dt={e:.17} max_dt={e:.17} avg_dt={e:.17}\n",
                .{ s.t >= s.options.t_stop or atStop(s.t, s.options.t_stop), s.resolved, s.t, s.options.t_stop, s.waveform.len, s.n_qt, s.steps, st.attempts, st.nr_iters, st.rej_newton, st.rej_lte, st.rej_state, st.rej_vpi, st.order_drops, st.bp_landings, st.by_order, st.first_dt, st.max_dt, if (s.steps > 0) s.t / @as(f64, @floatFromInt(s.steps)) else 0 },
            );
        }
        const a = s.allocator;
        a.free(s.lte_state);
        a.free(s.lte_zero);
        a.free(s.lte_rows);
        a.free(s.row_state);
        a.free(s.x_prev);
        a.free(s.x_try);
        s.ckt.land_rejects = false;
        s.* = undefined;
    }

    /// The scalar loop: each attempt's Newton through `converger.run`, a
    /// numerical failure retrying the step. Returns `finish`'s value, or
    /// false when the run ended short of t_stop.
    fn drive(s: *Sim) !bool {
        while (try s.begin(HostOps{})) |a| {
            const nr = converger.run(s.ckt, s.ws, s.trial, a.t_dev, a.nr_opts, a.hook) catch |err| switch (err) {
                error.QueryCancelled, error.OutOfMemory => return err,
                else => converger.Result{ .converged = false, .iterations = 0 }, // else: a numerical failure retries the step
            };
            switch (try s.end(&a, nr, HostOps{})) {
                .go => {},
                .stop => break,
                .underflow => return false,
            }
        }
        return s.finish();
    }

    /// The step loop's head: null once the run has reached t_stop or
    /// `max_steps`, else the next attempt, with `trial` predicted and
    /// limited and the device state published for it.
    fn begin(s: *Sim, ops: anytype) !?Attempt {
        const options = s.options;
        if (!(s.t < options.t_stop and !atStop(s.t, options.t_stop) and s.steps < options.max_steps)) return null;
        const ckt = s.ckt;
        if (s.st.attempts != 0) try ckt.checkpoint(.{
            .phase = .transient,
            .completed = s.st.attempts,
            .simulation_time = s.t,
            .step_size = s.attempted_dt,
            .next_step = s.dt,
            .accepted = s.steps,
        });
        s.attempted_dt = s.dt;
        const t = s.t;
        const dt = s.dt;
        // The time the devices evaluate at; OPtran holds it at 0.
        const t_dev = if (options.sources_at_zero) 0 else t + dt;
        // Publish the point this attempt aims at before anything evaluates
        // it: generated devices read Instance.abstime (§9.10 `$abstime`,
        // `ddt`), not the `t` argument. A rejected step comes back here with
        // the shrunken dt, so the last write before an accept is the accepted
        // dt. §5.10.2: initial_step is the analysis' first step, final_step
        // the one that lands on t_stop.
        ckt.setSimState(.{
            .t = t_dev,
            .dt = dt,
            .kind = .tran,
            .initial_step = s.steps == 0,
            .final_step = t + dt >= options.t_stop,
        });
        if (s.vpi) |v| {
            var p = s.vpi_point;
            p.t = t + dt;
            p.dt = dt;
            v.point(v.ctx, p);
        }
        const eff_method = methodAt(options.method, s.order);
        // Gear-3..6 solves its coefficients from the whole step history
        // and folds the older charges into g once per attempt.
        const gear_hi = s.order >= 3;
        var a: Attempt = .{
            .t_dev = t_dev,
            .method = eff_method,
            .gear_hi = gear_hi,
            .dts = [1]f64{dt} ++ s.dt_old,
            .ag = undefined,
            .c = undefined,
            .hook = undefined,
            .nr_opts = undefined,
        };
        a.c = if (gear_hi) blk: {
            a.ag = integrator.gearAg(s.order, &a.dts);
            break :blk .{ .ag0 = a.ag[0], .ag2 = 0 };
        } else integrator.coeffs(eff_method, dt, s.dt_old[0], options.xmu);
        ops.attempt(s, &a);
        // ngspice's transient Newton accepts on the per-row delta test alone
        // (niconv.c): a step that has converged in x but leaves a residual on
        // a row, e.g. a Meyer gate cap's curvature at a source-driven node,
        // is not iterated again.
        a.nr_opts = converger.optionsFromTolerances(options.tol, options.tol.itl4);
        a.nr_opts.residual_tol = std.math.inf(f64);
        return a;
    }

    /// The step loop's body after Newton: rejects (Newton, a device's
    /// `reject_at`, a state flip, the LTE, VPI) shrink `dt` and return
    /// `.go`; an accepted step promotes or keeps the order, advances the
    /// histories, commits, lands breakpoints, records and streams, and sets
    /// the next dt.
    fn end(s: *Sim, a: *const Attempt, nr: converger.Result, ops: anytype) !Verdict {
        const ckt = s.ckt;
        const options = s.options;
        const has_charge = s.has_charge;
        const q_hist = s.qHist();
        const t = s.t;
        const dt = s.dt;
        const t_dev = a.t_dev;
        const eff_method = a.method;
        const cf = a.c;
        const gear_hi = a.gear_hi;
        s.st.attempts += 1;
        s.st.nr_iters += nr.iterations;

        if (!nr.converged) {
            s.st.rej_newton += 1;
            // Restore FSM devices to the last accepted state.
            ops.revert(s);
            // Cut dt by 8 and drop to order 1 in one retry (dctran.c:815, :823).
            if (s.order > 1) s.st.order_drops += 1;
            s.order = 1;
            s.dt = dt / 8.0;
            // A sub-floor proposal does not prove the floor will fail. Try it
            // once; rejection from the floor still terminates (dctran.c:974).
            if (s.dt < options.dt_min and dt > options.dt_min and t + options.dt_min > t) {
                if (s.stats_on) std.debug.print("tran-floor-retry: reason=newton t={e:.17} old_dt={e:.17} proposed={e:.17} retry={e:.17}\n", .{ t, dt, s.dt, options.dt_min });
                s.dt = options.dt_min;
            }
            if (s.dt < options.dt_min) {
                // The stats block in `finish` is skipped by this return.
                if (s.stats_on) std.debug.print(
                    "tran-stats: DT UNDERFLOW (newton) at t={e:.6} dt={e:.3} accepted={d} attempts={d} nr_iters={d}\n",
                    .{ t, s.dt, s.steps, s.st.attempts, s.st.nr_iters },
                );
                return .underflow;
            }
            return .go;
        }

        // A device located an event inside this step and asked for the step
        // to end on it (`request_reject_at`): retry landing there. A time
        // within state_eps of the step's end counts as the end, the
        // resolution the flip query below keeps too.
        if (ops.rejectAt(s)) |tr| if (tr > t and t + dt - tr > s.state_eps and tr - t >= options.dt_min) {
            s.st.rej_state += 1;
            ops.revert(s);
            s.dt = tr - t;
            return .go;
        };

        // A device state flipped inside this step (a switch crossed its
        // threshold): reject and shrink so the conductance step lands within
        // state_eps of the crossing instead of smeared across dt.
        if (ops.flipped(s, a, dt > s.state_eps)) {
            s.st.rej_state += 1;
            ops.revert(s);
            s.order = 1;
            s.dt = @max(0.25 * dt, s.state_eps);
            return .go;
        }

        // The first accepted point skips CKTtrunc (dctran.c firsttime), so dt
        // repeats. §9.17.2 `$bound_step` is read after the step and applies
        // to the next one; devices only set it in `updateState`, so it cannot
        // be folded into effective_dt_max.
        var dt_next = if (s.steps == 0) dt else @min(dt * 2.0, s.effective_dt_max);
        if (ops.boundStep(s)) |bs| dt_next = @min(dt_next, bs);

        // CKTterr reads the charge of the published point. The converger
        // returns x_k+1 while the planes hold q(x_k) (or a JFNK matvec's x),
        // so re-read q at the solution: the LTE, advanceCurrent and the next
        // residual all see q(trial).
        // `q_fresh`: the converger's `updateStates` already stamped them.
        if (has_charge) ops.charges(s, a);
        const lte: integrator.LteIn = .{
            .dt = dt,
            .dt1 = s.dt_old[0],
            .dt2 = s.dt_old[1],
            .reltol = options.tol.reltol,
            .abstol = options.tol.abstol,
            .chgtol = options.tol.chgtol,
            .trtol = options.tol.trtol,
        };
        // This step's own bound, read again by the gear order control.
        var del: f64 = undefined;
        if (has_charge and s.steps > 0) {
            del = if (gear_hi)
                ops.boundGear(s, a, eff_method, lte)
            else
                ops.bound(s, a, eff_method, lte);
            if (!(del >= 0.9 * dt)) {
                s.st.rej_lte += 1;
                // Reconstruct only from captured history, before rollback.
                // GPU lanes own separate histories; report that limitation
                // explicitly rather than inspect stale host ring storage.
                if (s.ckt.knobs.tran_lte_trace and s.st.rej_lte <= 16) {
                    if (@TypeOf(ops) == HostOps) s.traceLte(a, nr, lte, del) else std.debug.print("tran-lte: captured-history diagnostic unavailable for this lane backend\n", .{});
                }
                ops.revert(s);
                // Retry at the LTE's dt and the same order (dctran.c:966
                // `CKTdelta = newdelta`); only a Newton failure drops the
                // order. del < 0.9*dt here, so every retry shrinks. A NaN
                // bound (a non-finite charge) retries as a Newton failure.
                if (std.math.isNan(del)) {
                    s.order = 1;
                    s.dt = dt / 8.0;
                } else s.dt = del;
                // Recompute the unchanged LTE test at the existing floor
                // before declaring failure, with the same retry guard as Newton.
                if (s.dt < options.dt_min and dt > options.dt_min and t + options.dt_min > t) {
                    if (s.stats_on) std.debug.print("tran-floor-retry: reason=lte t={e:.17} old_dt={e:.17} proposed={e:.17} retry={e:.17}\n", .{ t, dt, s.dt, options.dt_min });
                    s.dt = options.dt_min;
                }
                if (s.dt < options.dt_min) {
                    if (s.stats_on) std.debug.print(
                        "tran-stats: DT UNDERFLOW (lte) at t={e:.6} dt={e:.3} accepted={d} attempts={d} nr_iters={d}\n",
                        .{ t, s.dt, s.steps, s.st.attempts, s.st.nr_iters },
                    );
                    return .underflow;
                }
                return .go;
            }
            // Growth capped at 2x per accepted step, as in ngspice, and
            // lower at gear-3..6 (`maxGrowth`).
            dt_next = @min(@max(del, options.dt_min), integrator.maxGrowth(eff_method) * dt, s.effective_dt_max);
        }

        // LRM 12.31.3 acbConvergenceTest, last before acceptance: a rejection
        // (or 12.36 vpiRejectTransientStep) backs up to half the step from
        // the last accepted point, as VerA's own walk does; 12.36
        // vpiTransientFailConverge solves the same step again.
        if (s.vpi) |v| switch (v.verdict(v.ctx)) {
            .accept => s.vpi_again = 0,
            .reject => {
                s.st.rej_vpi += 1;
                ops.revert(s);
                s.dt = dt / 2.0;
                if (s.dt < options.dt_min) return .underflow;
                return .go;
            },
            .again => {
                s.st.rej_vpi += 1;
                ops.revert(s);
                s.vpi_again += 1;
                if (s.vpi_again == 60) return .underflow;
                return .go;
            },
        };

        // Accepted from here on; the history below is still this step's.
        if (s.tangents) |tg| {
            try tg.advance(ckt, s.ws, s.a_vals, s.trial, t_dev, .{ .method = eff_method, .c = cf, .q1 = q_hist[1], .q2 = q_hist[2], .i = s.i_prev });
            if (t + dt >= options.t_start) try tg.record(t + dt, s.probes);
        }

        if (has_charge) {
            // Order promotion (dctran.c:901-913): recompute the trunc at
            // order 2 and adopt min(2*dt, del2) as the next dt whether or not
            // the order changes, as ngspice's `CKTdelta = newdelta` does.
            s.st.by_order[s.order - 1] += 1;
            s.order_run = if (s.order == s.run_order) s.order_run + 1 else 1;
            s.run_order = s.order;
            if (s.steps > 0 and s.order == 1) {
                const trial_del = ops.bound(s, a, methodAt(options.method, 2), lte);
                // A NaN probe (prev3 non-finite) keeps BE and the step.
                if (!std.math.isNan(trial_del)) {
                    const nd2 = @min(2.0 * dt, trial_del);
                    if (nd2 > 1.05 * dt) s.order = 2;
                    dt_next = @min(@max(nd2, options.dt_min), s.effective_dt_max);
                    if (ops.boundStep(s)) |bs| dt_next = @min(dt_next, bs);
                }
            } else if (s.steps > 0 and s.top_order >= 3 and s.order_run > s.order) {
                // Gear-3..6 order control, an espice policy (dctran.c stops
                // at order 2): after order + 1 accepted steps at one order,
                // move one order up or down (not below 2) when that order's
                // CKTterr bound beats this one's by 5%, the margin of the
                // 1-to-2 promotion; the next step is then that bound under
                // the new order's `maxGrowth`.
                // ponytail: one order per change; DASSL-style selection on
                // error trends would move faster.
                var best = del;
                var best_order = s.order;
                for ([_]u8{ s.order + 1, s.order - 1 }) |k| {
                    if (k > s.top_order or k < 2) continue;
                    const d = ops.boundGear(s, a, methodAt(options.method, k), lte);
                    if (d > 1.05 * best) {
                        best = d;
                        best_order = k;
                    }
                }
                if (best_order != s.order) {
                    s.order = best_order;
                    dt_next = @min(@max(best, options.dt_min), integrator.maxGrowth(methodAt(options.method, s.order)) * dt, s.effective_dt_max);
                    if (ops.boundStep(s)) |bs| dt_next = @min(dt_next, bs);
                }
            }

            // The dynamic current under the method actually used. Both index
            // spaces run it: ngspice keeps it per state in
            // CKTstates[0][qcap+1], where CKTterr's volttol reads it.
            // Then [cur, prev, prev2, ...] -> [stale, cur, prev, ...].
            ops.advance(s, a);
        }
        std.mem.copyBackwards(f64, s.dt_old[1..], s.dt_old[0 .. s.dt_old.len - 1]);
        s.dt_old[0] = dt;

        // Rotate the slices; accepted states need no copy.
        const stale = s.prev;
        s.prev = s.cur;
        s.cur = s.trial;
        s.trial = stale;
        s.t = t + dt;
        if (s.steps == 0) s.st.first_dt = dt;
        s.st.max_dt = @max(s.st.max_dt, dt);
        s.steps += 1;
        const t_new = s.t;
        // §4.5.2 delay-line bookkeeping, once per accepted point
        // (`Hooks.commit_state`), then the commit that latches every stage.
        ops.commit(s);

        // Landed on a breakpoint: drop to BE and resume at
        // 0.1*min(saveDelta, gap to the next break), spice3 dctran's rule,
        // which resolves paired edges instead of stepping over them. The
        // history is kept; the promotion check re-promotes next step. A step
        // that was not clamped lands too when it ends within 100 ulps of the
        // breakpoint or delmin short of it (dctran.c:559); only min_break
        // separates a clamped landing from its target. The line echoes are
        // left out of that: they stand in for traload breakpoints ngspice
        // sets only on a sharp input, and tran/bench_tline_delay_line steps
        // through its t = td echo without a cut there.
        if (s.bp_next) |bp| {
            const landed = if (s.bp_clamped)
                @abs(t_new - bp) <= s.min_break
            else
                (bp - t_new <= s.delmin or almostEqualUlps(t_new, bp, 100)) and
                    std.mem.indexOfScalar(f64, s.echo_bps[0..s.n_echo], bp) == null;
            if (landed) {
                s.st.bp_landings += 1;
                s.order = 1;
                // Re-emit one line delay later; dedupe within min_break and
                // drop when the table is full.
                if (s.echo_td) |td_| {
                    const e = t_new + td_;
                    if (e < options.t_stop and s.n_echo < s.echo_bps.len) {
                        var dup = false;
                        for (s.echo_bps[0..s.n_echo]) |old| {
                            if (@abs(old - e) <= s.min_break) {
                                dup = true;
                                break;
                            }
                        }
                        if (!dup) {
                            s.echo_bps[s.n_echo] = e;
                            s.n_echo += 1;
                        }
                    }
                }
                var shrink = s.bp_save_dt;
                if (nextBp(ckt, s.echo_bps[0..s.n_echo], s.uic_bp, t_new + s.min_break)) |nb| shrink = @min(shrink, nb - t_new);
                dt_next = @min(dt_next, 0.1 * shrink);
            }
            s.bp_next = null;
        }

        // Stateful-charge devices: the commits above can move a device's q
        // away from what q_hist recorded. Re-read it so the next residual
        // starts from q(cur); otherwise it opens on alpha*(q_committed -
        // q_recorded), which doubles with every dt halving. The i_prev
        // correction is the same alpha*dq for every method. dq is not zero
        // after the pre-LTE re-read: dropping this changes the output of
        // txl2_3_line, hfet_inverter, mesa_oscillator and mos6_inverter.
        // `evalQ` computes the same q bits as `eval`, charges only.
        ops.rebase(s, a);

        // ngspice neither steps onto TSTART nor prints it: dctran.c dumps
        // the accepted points at t >= CKTinitTime and nothing else (#167).
        if (t_new >= options.t_start) try s.waveform.record(t_new, ops.accepted(s), s.probes);
        // Before the breakpoint search below, so a time a callback forces
        // from this point is landed on.
        if (s.vpi) |v| v.accept(v.ctx, false, t_new >= options.t_stop or atStop(t_new, options.t_stop));
        if (s.stream) |sm| if (t_new >= options.t_start) {
            sm.feed(t_new, ops.accepted(s), s.waveform.data());
            if (options.point.abort_after) |bound| if (t_new > bound and sm.unresolved()) return error.MeasureBoundExceeded;
            if (options.point.early_stop and sm.open == 0 and (s.tangents == null or s.tangents.?.open() == 0)) {
                s.resolved = true;
                return .stop;
            }
        };

        // Clamp dt to land on the next breakpoint, skipping those within
        // min_break of now (ngspice CKTminBreak merge).
        s.bp_next = nextBp(ckt, s.echo_bps[0..s.n_echo], s.uic_bp, t_new + s.min_break);
        s.bp_clamped = false;
        if (s.bp_next) |bp| {
            const dt_to_bp = bp - t_new;
            if (dt_to_bp < dt_next) {
                s.bp_save_dt = dt_next;
                dt_next = dt_to_bp;
                s.bp_clamped = true;
            }
        }

        s.dt = dt_next;
        if (t_new + s.dt > options.t_stop) s.dt = options.t_stop - t_new;
        return .go;
    }

    /// The step loop's tail: the last checkpoint, the stats line, `x`
    /// holding the last accepted solution. True on reaching t_stop or once
    /// every card resolved.
    fn finish(s: *Sim) !bool {
        const options = s.options;
        try s.ckt.checkpoint(.{
            .phase = .transient,
            .completed = s.st.attempts,
            .simulation_time = s.t,
            .step_size = s.attempted_dt,
            .next_step = if (s.t < options.t_stop) s.dt else 0,
            .accepted = s.steps,
        });
        if (s.cur.ptr != s.x.ptr) num.copySimd(s.x, s.cur);
        return s.resolved or s.t >= options.t_stop or atStop(s.t, options.t_stop);
    }

    fn traceLte(s: *Sim, a: *const Attempt, nr: converger.Result, lte: integrator.LteIn, bound: f64) void {
        const hit = integrator.diagnoseBound(a.method, a.method, s.lteHist(), s.lteIp(), s.lteG(), a.c, lte, &a.dts);
        const matches = @as(u64, @bitCast(hit.bound)) == @as(u64, @bitCast(bound)) or
            (std.math.isNan(hit.bound) and std.math.isNan(bound));
        std.debug.print(
            "tran-lte: matches={} accepted_t={e:.17} trial_t={e:.17} dt={e:.17} bound={e:.17} diagnostic_bound={e:.17} method={s} order={d} nr_iters_attempt={d} index={d}\n",
            .{ matches, s.t, s.t + s.dt, s.dt, bound, hit.bound, @tagName(a.method), integrator.order(a.method), nr.iterations, hit.index },
        );
        std.debug.assert(matches);
        if (!matches) return;
        if (s.n_qt == 0) {
            std.debug.print("tran-lte-state: row={d}\n", .{hit.index});
        } else if (hit.index >= s.n_qt) {
            std.debug.print("tran-lte-state: appended_row={d}\n", .{s.lte_rows[hit.index - s.n_qt]});
        } else {
            var off: usize = 0;
            for (s.ckt.batches, 0..) |b, bi| {
                const tape = b.hooks.q_tape orelse continue;
                const len = tape(b.ctx).len;
                if (hit.index < off + len) {
                    if (b.hooks.charge_identity) |identity| {
                        const id = identity(b.ctx, hit.index - off);
                        std.debug.print("tran-lte-state: batch={d} type={s} original_instance={d} generated_site={d} rows={any}\n", .{ bi, b.type_name, id.instance, id.site, id.rows });
                    } else std.debug.print("tran-lte-state: batch={d} type={s} local_tape_index={d} identity_unavailable=true\n", .{ bi, b.type_name, hit.index - off });
                    break;
                }
                off += len;
            }
        }
        for (s.lteHist(), 0..) |h, k| std.debug.print("tran-lte-q: history={d} q={e:.17}\n", .{ k, h[hit.index] });
        for (a.dts, 0..) |dt, k| std.debug.print("tran-lte-dt: history={d} dt={e:.17}\n", .{ k, dt });
        std.debug.print(
            "tran-lte-operands: ip={e:.17} i_new={e:.17} current_tol={e:.17} charge_tol={e:.17} tol={e:.17} dd={e:.17} lte_coefficient={e:.17} unrooted_bound={e:.17} ag0={e:.17} ag1={e:.17} ag2={e:.17} trtol={e:.17}\n",
            .{ hit.i_prev, hit.i_new, hit.current_tol, hit.charge_tol, hit.tolerance, hit.difference, hit.coefficient, hit.unrooted_bound, a.c.ag0, a.c.ag1, a.c.ag2, lte.trtol },
        );
    }
};

/// What `Sim.begin` and `Sim.end` ask of the numerics, so one copy of the
/// step control drives the host (`HostOps`, the oracle) and the device
/// (`gpu_lanes.Ops`). Each method runs exactly the statements it names, at
/// the point it is called. A backend may answer from numbers it computed
/// earlier for the whole round, as long as they equal what the statements
/// would compute now.
const HostOps = struct {
    /// begin: gear history, the hook, predict, predictFirstIterate,
    /// evalFollowsFirst, applyLimits, reject_at = null.
    inline fn attempt(_: HostOps, s: *Sim, a: *Attempt) void {
        const ckt = s.ckt;
        const q_hist = s.qHist();
        if (a.gear_hi) integrator.gearHistory(s.g_row, q_hist[0 .. s.order + 1], a.ag);
        a.hook = .{
            .method = a.method,
            .c = a.c,
            .q_prev = q_hist[1],
            .i_prev = s.i_prev,
            .q_prev2 = if (a.gear_hi) s.g_row else q_hist[2],
            .q_cur = {},
            .a_vals = s.a_vals,
            .has_charge = s.has_charge,
        };
        // MODEINITPRED (dctran.c:794, DEVpred in dioload.c/mos1load.c): the
        // first iterate extrapolates the last two accepted points by
        // dt/dt_prev and is limited against the last accepted one. HFET and
        // MESA divide by the step two back instead (CKTdeltaOld[2]).
        const xfact = s.dt / s.dt_old[0];
        predict(s.trial, s.cur, s.prev, xfact);
        ckt.predictFirstIterate(s.trial, s.cur, s.prev, s.dt / s.dt_old[1]);
        ckt.evalFollowsFirst(s.trial, a.t_dev);
        _ = ckt.applyLimits(s.trial, s.cur);
        ckt.reject_at = null;
    }
    inline fn revert(_: HostOps, s: *Sim) void {
        _ = s.ckt.stateCtl(.revert);
    }
    inline fn rejectAt(_: HostOps, s: *Sim) ?f64 {
        return s.ckt.reject_at;
    }
    /// The `evalFollows` hint, then (only when `ask`) the flip query.
    inline fn flipped(_: HostOps, s: *Sim, a: *const Attempt, ask: bool) bool {
        if (s.has_charge and !s.ckt.q_fresh) s.ckt.evalFollows(s.trial, a.t_dev, true);
        return ask and s.ckt.stateCtl(.query);
    }
    inline fn boundStep(_: HostOps, s: *Sim) ?f64 {
        return s.ckt.boundStep();
    }
    /// The charge re-read at the solution into ring slot 0, rows and tape.
    inline fn charges(_: HostOps, s: *Sim, a: *const Attempt) void {
        const ckt = s.ckt;
        const qt_hist = s.qtHist();
        if (!ckt.q_fresh) ckt.evalQ(s.trial, a.t_dev);
        num.copySimd(s.qHist()[0], ckt.q_vec[0..s.n]);
        if (s.n_qt > 0) lteSnap(ckt, qt_hist[0], s.lte_zero, s.lte_rows);
        if (a.gear_hi and s.n_qt > 0) integrator.gearHistory(s.qt_g, qt_hist[0 .. s.order + 1], a.ag);
    }
    /// `stepBound` (order <= 2) of `method` after a step at `a.method`.
    inline fn bound(_: HostOps, s: *Sim, a: *const Attempt, method: Method, lte: integrator.LteIn) f64 {
        const h = s.lteHist();
        return integrator.stepBound(W, method, a.method, .{ h[0], h[1], h[2], h[3] }, s.lteIp(), a.c, lte);
    }
    inline fn boundGear(_: HostOps, s: *Sim, a: *const Attempt, method: Method, lte: integrator.LteIn) f64 {
        return integrator.stepBoundGear(method, a.method, s.lteHist(), s.lteIp(), s.lteG(), a.c, lte, &a.dts);
    }
    /// advanceCurrent on both index spaces, then the ring rotation.
    inline fn advance(_: HostOps, s: *Sim, a: *const Attempt) void {
        const q_hist = s.qHist();
        const qt_hist = s.qtHist();
        integrator.advanceCurrent(a.method, s.i_prev, q_hist[0], q_hist[1], if (a.gear_hi) s.g_row else q_hist[2], a.c);
        if (s.n_qt > 0)
            integrator.advanceCurrent(a.method, s.qt_i_prev, qt_hist[0], qt_hist[1], if (a.gear_hi) s.qt_g else qt_hist[2], a.c);
        std.mem.rotate([]f64, q_hist, s.n_hist - 1);
        if (s.n_qt > 0) std.mem.rotate([]f64, qt_hist, s.n_hist - 1);
    }
    /// commitStates and the commit, at the rotated `cur`.
    inline fn commit(_: HostOps, s: *Sim) void {
        _ = s.ckt.commitStates(s.cur);
        _ = s.ckt.stateCtl(.commit);
    }
    /// The stateful-charge re-read at `cur` (`has_state_q`).
    inline fn rebase(_: HostOps, s: *Sim, a: *const Attempt) void {
        const ckt = s.ckt;
        const q_hist = s.qHist();
        const qt_hist = s.qtHist();
        if (!(s.has_charge and ckt.has_state_q)) return;
        ckt.evalQ(s.cur, s.t);
        integrator.rebaseCurrent(s.i_prev, ckt.q_vec[0..s.n], q_hist[1], a.c.ag0);
        num.copySimd(q_hist[1], ckt.q_vec[0..s.n]);
        if (s.n_qt > 0) {
            lteSnap(ckt, qt_hist[0], s.lte_zero, s.lte_rows);
            integrator.rebaseCurrent(s.qt_i_prev, qt_hist[0], qt_hist[1], a.c.ag0);
            // Swap, not copy: slot 0 is the ring's scratch slot, and the
            // next accepted attempt overwrites it before any read.
            std.mem.swap([]f64, &qt_hist[0], &qt_hist[1]);
        }
    }
    /// The accepted solution `record` and `feed` read (probe rows only).
    inline fn accepted(_: HostOps, s: *Sim) []const f64 {
        return s.cur;
    }
};

/// `lteSnap` and `nextBp`, hoisted out of `simulateWith` unchanged so
/// `Sim` can call them.
fn lteSnap(c: *root.Circuit, dst: []f64, zero: []const [2]u32, rows: []const u32) void {
    const n_tape = dst.len - rows.len;
    c.snapshotQTape(dst[0..n_tape]);
    for (zero) |span| @memset(dst[span[0]..span[1]], 0);
    for (rows, dst[n_tape..]) |r, *d| d.* = c.q_vec[r];
}

fn nextBp(c: *root.Circuit, echo: []const f64, extra: ?f64, after: f64) ?f64 {
    var best = std.math.inf(f64);
    if (c.nextBreakpoint(after)) |bp| best = bp;
    for (echo) |e| {
        if (e >= after and e < best) best = e;
    }
    if (extra) |e| {
        if (e >= after and e < best) best = e;
    }
    return if (best == std.math.inf(f64)) null else best;
}

/// Parts 1..parts-1 of `block`, which holds `parts` (at most 10: gear-6's
/// eight charge slots and g) equal parts (all empty for an empty block): the
/// slots after the leading dynamic-current part. Entries past parts - 1 are
/// empty.
fn ring(block: []f64, parts: usize) [9][]f64 {
    const m = block.len / parts;
    var out: [9][]f64 = @splat(block[0..0]);
    for (out[0 .. parts - 1], 1..) |*o, i| o.* = block[i * m ..][0..m];
    return out;
}

/// The method an order-`k` step integrates with: BE at 1, else `top` itself
/// when it stops at order 2 (trap, gear-2, or BE), else gear-k.
fn methodAt(top: Method, k: u8) Method {
    if (k == 1) return .backward_euler;
    return if (integrator.order(top) <= 2) top else integrator.gear(k);
}

/// ngspice's end test (dctran.c:503, AlmostEqualUlps(t, tfinal, 100)): a
/// step that lands within 100 ulps of t_stop ends the run. Without it a
/// breakpoint sum that rounds one ulp short of t_stop leaves a 1e-23 s step
/// that fails as TimestepTooSmall.
fn atStop(t: f64, t_stop: f64) bool {
    if (t == t_stop) return true;
    if ((t < 0) != (t_stop < 0)) return false;
    const a: i64 = @bitCast(t);
    const b: i64 = @bitCast(t_stop);
    return @abs(a - b) <= 100;
}

test "atStop: within 100 ulps of t_stop, not past them" {
    const t_stop = 2.5e-7;
    try std.testing.expect(atStop(std.math.nextAfter(f64, t_stop, 0), t_stop));
    try std.testing.expect(!atStop(t_stop * (1 - 1e-12), t_stop));
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

/// One `.tran` query's outputs around its `Sim`: the probes it records,
/// its `.meas` stream, waveform and tangents, so `run` and `runLanes`
/// build and publish each query the same way. Pinned: `init` fills it in
/// place, and `sim` points at `wf`, `stream` and `tangents`.
const Run = struct {
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

    /// `run`'s setup, then `Sim.init`.
    fn init(r: *Run, ctx: *const root.RunCtx, opts: Options) !void {
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
    fn deinit(r: *Run) void {
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
    fn take(r: *Run, sim_ok: bool) !root.Result {
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
};

const LaneError = root.ErrorOf(run);
/// One lane's `run` outcome.
pub const LaneOutcome = LaneError!root.Result;

/// `run` for the lanes of one batch: `.tran` queries on one pattern, each
/// from its own operating point (or `uic` start) and variant row, in rounds
/// (dev/research/lane-batched-simulation.md §7). Each round every live lane
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
    if (width == 1 or g == 1 or !(pin == .auto or pin == .direct) or root.Circuit.certify or c0.vpi != null) {
        for (ctxs, out) |*c, *o| o.* = run(c, opts);
        return;
    }
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

const gpu_lanes = @import("../gpu/lanes.zig");

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

/// `runLanes` on the GPU (dev/research/plans/phase4-gpu-lanes.md G7): any
/// number of `.tran` queries on one pattern, each its own `Sim` on the host
/// (step control, breakpoints, measures) over one `GpuLanes` that holds
/// every lane's numerics. Each round: every live lane `begin`s, the device
/// predicts and solves Newton for all of them and computes their step-end
/// numbers, each lane `end`s from those, and the device applies the
/// verdicts. Within tolerance of `runLanes`, not bitwise. Falls back to
/// each lane's `run` when the lanes are not eligible or the device cannot
/// take them; a lane whose device asks to reject a step reruns on the CPU.
/// The unknowns lane `r` reads from an accepted row: its recorded probes
/// and its `.meas` stream's columns, sorted and without repeats. Every
/// lane of a batch shares the deck, so the first lane's set serves all.
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

pub fn runGpuLanes(ctxs: []const root.RunCtx, opts: Options, out: []LaneOutcome) void {
    const g = ctxs.len;
    std.debug.assert(g >= 1 and out.len == g);
    if (comptime !gpu_lanes.available) {
        for (ctxs, out) |*c, *o| o.* = run(c, opts);
        return;
    }
    const gpa = ctxs[0].scratch_allocator;
    const runs = gpa.alloc(Run, g) catch {
        for (ctxs, out) |*c, *o| o.* = run(c, opts);
        return;
    };
    defer gpa.free(runs);
    const live = gpa.alloc(bool, g) catch {
        for (ctxs, out) |*c, *o| o.* = run(c, opts);
        return;
    };
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
    // The `.meas` passes run on the device when every lane can (06 6.8):
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

    const atts = gpa.alloc(Attempt, g) catch |e| {
        for (runs, live, out) |*r, *on, *o| if (on.*) {
            r.deinit();
            on.* = false;
            o.* = e;
        };
        return;
    };
    defer gpa.free(atts);
    const nr = gpa.alloc(gpu_lanes.LaneNewton, g) catch |e| {
        for (runs, live, out) |*r, *on, *o| if (on.*) {
            r.deinit();
            on.* = false;
            o.* = e;
        };
        return;
    };
    defer gpa.free(nr);
    const nr_opts = blk: {
        var o = converger.optionsFromTolerances(opts.tol, opts.tol.itl4);
        o.residual_tol = std.math.inf(f64);
        break :blk o;
    };
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
            if (s.st.attempts != 0) cb.checkpoint(.{
                .phase = .transient,
                .completed = s.st.attempts,
                .simulation_time = s.t,
                .step_size = s.attempted_dt,
                .next_step = s.dt,
                .accepted = s.steps,
            }) catch |e| return Lane.failAll(runs, live, out, e);
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

const LK = @import("solver").lane_kernels;

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

/// Whether `runGpuLanes` hands step control to the device (`LK.LaneStep`):
/// not under `ESPICE_GPU_HOST_STEP`, its oracle, nor under the step
/// diagnostics, which print from `Sim` mid-step.
fn deviceStepOk(ckt: *const root.Circuit, opts: Options) bool {
    const k = ckt.knobs;
    return !k.gpu_host_step and !k.tran_stats and !k.tran_lte_trace and opts.tangents.len == 0;
}

/// Lane `s`'s `LK.LaneStep` as `Sim.init` left it.
fn laneStep(s: *const Sim, opts: Options, max_iter: u16) LK.LaneStep {
    const Ctl = @import("device").abi.StateCtlOp;
    const tran_kind: @FieldType(@import("device").abi.SimState, "kind") = .tran;
    const st = s.st;
    return .{
        .t_stop = opts.t_stop,
        .dt_min = opts.dt_min,
        .xmu = opts.xmu,
        .dt_max = s.effective_dt_max,
        .delmin = s.delmin,
        .min_break = s.min_break,
        .state_eps = s.state_eps,
        .reltol = opts.tol.reltol,
        .abstol = opts.tol.abstol,
        .chgtol = opts.tol.chgtol,
        .trtol = opts.tol.trtol,
        .t = s.t,
        .dt = s.dt,
        .dt_old = s.dt_old,
        .attempted_dt = s.attempted_dt,
        .bp_next = s.bp_next orelse std.math.inf(f64),
        .bp_save_dt = s.bp_save_dt,
        .t_new = s.t,
        .dt_next = s.dt,
        .attempts = st.attempts,
        .nr_iters = st.nr_iters,
        .rej_newton = st.rej_newton,
        .rej_state = st.rej_state,
        .rej_lte = st.rej_lte,
        .order_drops = st.order_drops,
        .bp_landings = st.bp_landings,
        .by_order = st.by_order,
        .first_dt = st.first_dt,
        .max_dt = st.max_dt,
        .steps = s.steps,
        .order_run = s.order_run,
        .max_steps = opts.max_steps,
        .max_iter = max_iter,
        .iteration = s.ckt.sim.iteration,
        .order = s.order,
        .run_order = s.run_order,
        .bp_clamped = @intFromBool(s.bp_clamped),
        .status = LK.step_run,
        .top = switch (opts.method) {
            .backward_euler => 0,
            .trapezoidal => 1,
            .gear_2 => 2,
            else => unreachable, // gear-3..6 are not lane-eligible
        },
        .charge = @intFromBool(s.has_charge),
        .state_q = @intFromBool(s.has_charge and s.ckt.has_state_q),
        .event = LK.step_idle,
        .op_revert = 1 + @as(u8, @backingInt(Ctl.revert)),
        .op_commit = 1 + @as(u8, @backingInt(Ctl.commit)),
        .kind_tran = @intFromEnum(tran_kind),
    };
}

/// The device's step state back into lane `s` before `finish`: what
/// `finish`, the stats line and `take` read.
fn adoptStep(s: *Sim, st: LK.LaneStep) void {
    s.t = st.t;
    s.dt = st.dt;
    s.dt_old = st.dt_old;
    s.attempted_dt = st.attempted_dt;
    s.steps = st.steps;
    s.order = st.order;
    s.run_order = st.run_order;
    s.order_run = st.order_run;
    s.bp_next = if (st.bp_next == std.math.inf(f64)) null else st.bp_next;
    s.bp_clamped = st.bp_clamped != 0;
    s.bp_save_dt = st.bp_save_dt;
    s.st.attempts = st.attempts;
    s.st.nr_iters = st.nr_iters;
    s.st.rej_newton = st.rej_newton;
    s.st.rej_state = st.rej_state;
    s.st.rej_lte = st.rej_lte;
    s.st.order_drops = st.order_drops;
    s.st.bp_landings = st.bp_landings;
    s.st.by_order = st.by_order;
    s.st.first_dt = st.first_dt;
    s.st.max_dt = st.max_dt;
}

/// What `deviceLoop` shares with its per-lane drain.
const DeviceLanes = struct {
    runs: []Run,
    live: []bool,
    out: []LaneOutcome,
    ctxs: []const root.RunCtx,
    opts: Options,
    d: *gpu_lanes.GpuLanes,
    gpa: std.mem.Allocator,
    /// Per lane, what the drain left for the serial tail (`Fin`).
    fin: []Fin = &.{},
    /// Ring slots the last launch filled.
    rounds: usize = 1,
    /// The ring slot of each `.stop` lane's resolving row.
    stop_slot: []u8 = &.{},
    /// The `.meas` passes run on the device (`GpuLanes.initMeasures`).
    meas: bool = false,
    /// Per lane, its device passes all ended in an earlier launch.
    meas_done: []bool = &.{},

    const Fin = enum(u8) { none, finish, underflow, parked, stop };

    /// `Sim.end`'s tail for each of lane l's accepted rows of the launch,
    /// in round order: record, stream, the measure bound and early stop;
    /// then what the device's last status asks.
    fn drain(p: *const DeviceLanes, l: usize) void {
        p.fin[l] = .none;
        if (!p.live[l]) return;
        const s = &p.runs[l].sim;
        const opts = p.opts;
        for (0..p.rounds) |r| {
            const rep = p.d.slotReport(r, l);
            if (rep.event != LK.step_accept) continue;
            const t_new = rep.t_new;
            const row = p.d.slotRow(r, l);
            if (t_new >= opts.t_start) s.waveform.record(t_new, row, s.probes) catch |e| {
                GpuLane.kill(p.runs, p.live, p.out, l, e);
                return p.d.requestStop(l);
            };
            if (s.stream) |sm| if (t_new >= opts.t_start) {
                if (p.meas) {
                    // The device fed the row: the measure bound and early
                    // stop read the round that finished every pass.
                    const at = p.d.measResolved(l);
                    if (opts.point.abort_after) |bound| if (t_new > bound and !p.meas_done[l] and (at == null or at.? > r)) {
                        GpuLane.kill(p.runs, p.live, p.out, l, error.MeasureBoundExceeded);
                        return p.d.requestStop(l);
                    };
                    if (opts.point.early_stop) if (at) |a| if (a == r) {
                        s.resolved = true;
                        p.fin[l] = .stop;
                        p.stop_slot[l] = a;
                        return;
                    };
                    continue;
                }
                sm.feed(t_new, row, s.waveform.data());
                if (opts.point.abort_after) |bound| if (t_new > bound and sm.unresolved()) {
                    GpuLane.kill(p.runs, p.live, p.out, l, error.MeasureBoundExceeded);
                    return p.d.requestStop(l);
                };
                if (opts.point.early_stop and sm.open == 0) {
                    s.resolved = true;
                    p.fin[l] = .stop;
                    p.stop_slot[l] = @intCast(r);
                    return;
                }
            };
        }
        if (p.meas and p.d.measResolved(l) != null) p.meas_done[l] = true;
        const rep = p.d.slotReport(p.rounds - 1, l);
        p.fin[l] = switch (rep.status) {
            LK.step_done => .finish,
            LK.step_underflow => .underflow,
            LK.step_parked => .parked,
            else => .none,
        };
    }

    /// A parked lane (`reject_at` or `$bound_step`) reruns alone on the CPU.
    fn rerun(p: *const DeviceLanes, l: usize) void {
        if (p.fin[l] != .parked) return;
        p.runs[l].deinit();
        p.live[l] = false;
        p.d.detached[l] = true;
        p.out[l] = run(&p.ctxs[l], p.opts);
    }
};

/// `runGpuLanes`' rounds under device step control: `deviceLaunch`es of up
/// to `GpuLanes.max_rounds` rounds, the drain of their accepted rows in
/// round order, then the lanes that left. The worker's checkpoint fires
/// once per launch, from the first live lane.
fn deviceLoop(p_in: DeviceLanes, nr_opts: converger.Options, progress: anytype) void {
    var p = p_in;
    const g = p.runs.len;
    const d = p.d;
    p.fin = p.gpa.alloc(DeviceLanes.Fin, g) catch return GpuLane.failAll(p.runs, p.live, p.out, error.OutOfMemory);
    defer p.gpa.free(p.fin);
    p.stop_slot = p.gpa.alloc(u8, g) catch return GpuLane.failAll(p.runs, p.live, p.out, error.OutOfMemory);
    defer p.gpa.free(p.stop_slot);
    p.meas_done = p.gpa.alloc(bool, g) catch return GpuLane.failAll(p.runs, p.live, p.out, error.OutOfMemory);
    defer p.gpa.free(p.meas_done);
    @memset(p.meas_done, false);
    // Early stop and the measure bound are the drain's verdicts, so a lane
    // they end mid-launch has run past its resolving row. That is unseen
    // (its rows stop there, its step state comes from that row's report,
    // and its x is scratch it frees) unless a device can latch a status
    // `refuseLatched` reads off the device rows: then one round per launch.
    const host_stops = for (p.runs) |*r| {
        if (r.sim.stream != null and (p.opts.point.early_stop or p.opts.point.abort_after != null)) break true;
    } else false;
    const latches = for (p.runs[0].sim.ckt.batches) |b| {
        if (b.hooks.status != null) break true;
    } else false;
    const max_rounds: usize = if ((host_stops and latches) or !d.multiRound()) 1 else gpu_lanes.GpuLanes.max_rounds;
    // Rounds in the next launch, doubling up to the ring.
    var next: usize = 1;
    while (std.mem.indexOfScalar(bool, p.live, true)) |lead| {
        // No lane has attempted before the first launch.
        if (progress) |cb| if (d.prof.rounds != 0) {
            const rep = d.slotReport(p.rounds - 1, lead);
            if (rep.attempts != 0) cb.checkpoint(.{
                .phase = .transient,
                .completed = rep.attempts,
                .simulation_time = rep.t,
                .step_size = rep.attempted_dt,
                .next_step = rep.dt,
                .accepted = rep.steps,
            }) catch |e| return GpuLane.failAll(p.runs, p.live, p.out, e);
        };
        d.prof.lap(.host);
        p.rounds = next;
        d.deviceLaunch(nr_opts, @intCast(p.rounds)) catch |e| return GpuLane.failAll(p.runs, p.live, p.out, gpuError(e));
        d.prof.lap(.newton);
        const t_drain = gpu_lanes.Prof.now();
        forLanes(g, @as(*const DeviceLanes, &p), DeviceLanes.drain);
        if (d.prof.on) d.prof.drain_ns += gpu_lanes.Prof.now() - t_drain;
        // The lanes that left, serially. A lane's x is scratch it frees, so
        // only its step state comes back: the device's for a lane the device
        // ended, its resolving row's report for one early stop ended.
        for (p.fin, 0..) |f, l| if (f == .underflow or f == .stop) d.requestStop(l);
        var left = false;
        for (p.fin) |f| left = left or f != .none;
        if (left) {
            const steps = p.gpa.alloc(LK.LaneStep, g) catch return GpuLane.failAll(p.runs, p.live, p.out, error.OutOfMemory);
            defer p.gpa.free(steps);
            d.fetchSteps(steps) catch |e| return GpuLane.failAll(p.runs, p.live, p.out, gpuError(e));
            // The device passes of the lanes that left go back to their streams.
            if (p.meas) {
                d.fetchMeasures() catch |e| return GpuLane.failAll(p.runs, p.live, p.out, gpuError(e));
                const lane_passes = p.gpa.alloc(measure.passes.Dev, d.n_meas) catch return GpuLane.failAll(p.runs, p.live, p.out, error.OutOfMemory);
                defer p.gpa.free(lane_passes);
                for (p.fin, 0..) |f, l| if (f == .finish or f == .stop or f == .underflow) if (p.runs[l].sim.stream) |sm| {
                    d.lanePasses(l, lane_passes);
                    sm.adoptScans(lane_passes);
                };
            }
            for (p.fin, 0..) |f, l| switch (f) {
                .none, .parked => {},
                .finish, .stop => {
                    const s = &p.runs[l].sim;
                    adoptStep(s, steps[l]);
                    if (f == .stop) {
                        const rep = d.slotReport(p.stop_slot[l], l);
                        s.t = rep.t;
                        s.dt = rep.dt;
                        s.attempted_dt = rep.attempted_dt;
                        s.steps = rep.steps;
                        s.st.attempts = rep.attempts;
                    }
                    const ok = s.finish() catch |e| {
                        GpuLane.kill(p.runs, p.live, p.out, l, e);
                        continue;
                    };
                    GpuLane.done(p.runs, p.live, p.out, l, ok);
                },
                .underflow => {
                    adoptStep(&p.runs[l].sim, steps[l]);
                    GpuLane.done(p.runs, p.live, p.out, l, false);
                },
            };
            forLanes(g, @as(*const DeviceLanes, &p), DeviceLanes.rerun);
        }
        next = @min(2 * p.rounds, max_rounds);
        d.prof.lap(.host);
    }
}

/// `runGpuLanes`' lane outcomes: a lane leaves `live` once, with its
/// result or error in `out`.
const GpuLane = struct {
    fn done(rs: []Run, on: []bool, o: []LaneOutcome, l: usize, ok: bool) void {
        o[l] = rs[l].take(ok);
        rs[l].deinit();
        on[l] = false;
    }
    fn kill(rs: []Run, on: []bool, o: []LaneOutcome, l: usize, err: LaneError) void {
        o[l] = err;
        rs[l].deinit();
        on[l] = false;
    }
    fn failAll(rs: []Run, on: []bool, o: []LaneOutcome, err: LaneError) void {
        for (0..rs.len) |l| if (on[l]) kill(rs, on, o, l, err);
    }
};

/// Lanes per claim in `forLanes`, and the fewest lanes per thread.
const lanes_per_claim = 64;

/// Threads `forLanes` uses for `n` lanes: one below 256 lanes, so small
/// batches keep running serially on the analysis thread.
fn forLanesThreads(n: usize) usize {
    if (@import("builtin").single_threaded or n < 256) return 1;
    const cpus = std.Thread.getCpuCount() catch 1;
    return @max(1, @min(cpus, n / lanes_per_claim, 64));
}

/// `f(ctx, l)` for every lane l < n, claimed `lanes_per_claim` at a time by
/// `forLanesThreads(n)` threads. `f` must touch only lane l's state, so
/// the split cannot change any lane's result.
// ponytail: threads spawned per pass, two passes per round; a persistent
// pool if spawning ever shows beside the passes' 20-40 ms at 10k lanes.
fn forLanes(n: usize, ctx: anytype, comptime f: fn (@TypeOf(ctx), usize) void) void {
    const t = forLanesThreads(n);
    if (t <= 1) {
        for (0..n) |l| f(ctx, l);
        return;
    }
    const Work = struct {
        fn run(next: *std.atomic.Value(usize), count: usize, c: @TypeOf(ctx)) void {
            while (true) {
                const lo = next.fetchAdd(lanes_per_claim, .monotonic);
                if (lo >= count) return;
                for (lo..@min(lo + lanes_per_claim, count)) |l| f(c, l);
            }
        }
    };
    var next = std.atomic.Value(usize).init(0);
    var threads: [63]std.Thread = undefined;
    var spawned: usize = 0;
    // A failed spawn leaves fewer helpers; the calling thread still claims.
    while (spawned < t - 1) : (spawned += 1) {
        threads[spawned] = std.Thread.spawn(.{ .stack_size = @import("../worker.zig").stack_size }, Work.run, .{ &next, n, ctx }) catch break;
    }
    Work.run(&next, n, ctx);
    for (threads[0..spawned]) |th| th.join();
}

/// A driver failure mid-run, as the lanes' error: out of memory stays
/// itself; anything else cancels the batch.
fn gpuError(e: anyerror) LaneError {
    return if (e == error.OutOfMemory) error.OutOfMemory else error.QueryCancelled;
}

// Private implementation access for the analysis test suite.
pub const test_access = if (@import("builtin").is_test) .{
    .W = W,
    .integrator = integrator,
    .predict = predict,
    .almostEqualUlps = almostEqualUlps,
} else {};

/// ngspice's AlmostEqualUlps (maths/misc/equality.c): `a` and `b` are at
/// most `max_ulps` representable doubles apart, across zero included.
fn almostEqualUlps(a: f64, b: f64, max_ulps: i64) bool {
    if (a == b) return true;
    const lex = struct {
        fn f(x: f64) i128 {
            const i: i64 = @bitCast(x);
            return if (i < 0) @as(i128, std.math.minInt(i64)) - i else i;
        }
    }.f;
    return @abs(lex(a) - lex(b)) <= max_ulps;
}

// Controller tests use real generated devices and HostOps histories. Only the
// LTE proposal/Newton outcome is injected, so each retry branch is deterministic.
const FloorRetryTestOps = struct {
    proposal: f64,
    reverts: usize = 0,
    advances: usize = 0,
    commits: usize = 0,
    fn revert(o: *@This(), s: *Sim) void {
        o.reverts += 1;
        (HostOps{}).revert(s);
    }
    fn rejectAt(_: *@This(), s: *Sim) ?f64 {
        return (HostOps{}).rejectAt(s);
    }
    fn flipped(_: *@This(), s: *Sim, a: *const Attempt, ask: bool) bool {
        return (HostOps{}).flipped(s, a, ask);
    }
    fn boundStep(_: *@This(), s: *Sim) ?f64 {
        return (HostOps{}).boundStep(s);
    }
    fn charges(_: *@This(), s: *Sim, a: *const Attempt) void {
        (HostOps{}).charges(s, a);
    }
    fn bound(o: *@This(), _: *Sim, _: *const Attempt, _: Method, _: integrator.LteIn) f64 {
        return o.proposal;
    }
    fn boundGear(o: *@This(), _: *Sim, _: *const Attempt, _: Method, _: integrator.LteIn) f64 {
        return o.proposal;
    }
    fn advance(o: *@This(), s: *Sim, a: *const Attempt) void {
        o.advances += 1;
        (HostOps{}).advance(s, a);
    }
    fn commit(o: *@This(), s: *Sim) void {
        o.commits += 1;
        (HostOps{}).commit(s);
    }
    fn rebase(_: *@This(), s: *Sim, a: *const Attempt) void {
        (HostOps{}).rebase(s, a);
    }
    fn accepted(_: *@This(), s: *Sim) []const f64 {
        return (HostOps{}).accepted(s);
    }
};

fn checkFloorRetryController(newton_failure: bool, nonadvancing: bool, endpoint: bool, persistent_failure: bool) !void {
    const testing = std.testing;
    const gpa = testing.allocator;
    var lib = try @import("device").Library.init(gpa);
    defer lib.deinit();
    var b = try @import("builder").Builder.init(gpa, &lib);
    const node = try b.addNode();
    try b.addDevice(@import("models").vsource, "", &.{ .dc = 1 }, &.{}, .{ node, root.GROUND });
    try b.addDevice(@import("models").capacitor, "", &.{ .c = 1e-12 }, &.{}, .{ node, root.GROUND });
    var prepared = try b.compile();
    defer prepared.deinit();
    var ckt = try root.Circuit.instantiate(&prepared, gpa);
    defer ckt.deinit();
    const x = try gpa.alloc(f64, ckt.n);
    defer gpa.free(x);
    @memset(x, 0);
    var wf = try Waveform.init(gpa, 1, 32);
    defer wf.deinit();
    var s: Sim = undefined;
    try s.init(&ckt, x, &.{node}, &wf, .{ .t_stop = 2e-12, .dt_init = 2e-14, .dt_max = 2e-14, .dt_min = 1e-18, .uic = true }, gpa, null, null);
    defer s.deinit();
    // Seed an actual accepted point and both row/tape charge histories.
    const first = (try s.begin(HostOps{})).?;
    const first_nr = try converger.run(&ckt, s.ws, s.trial, first.t_dev, first.nr_opts, first.hook);
    try testing.expect(first_nr.converged);
    try testing.expectEqual(Verdict.go, try s.end(&first, first_nr, HostOps{}));
    try testing.expectEqual(@as(u32, 1), s.steps);
    try testing.expect(s.n_qt > 0);

    s.dt = 4 * s.options.dt_min;
    if (nonadvancing) {
        // Small finite times, far from the 100-ulp stop tolerance: the old
        // attempt advances by one ulp, but the requested floor rounds to t.
        s.t = 1e-9;
        s.options.t_stop = s.t + 1e-20;
        s.options.dt_min = 4e-26;
        s.dt = 2e-25;
        try testing.expect(s.t + s.dt > s.t);
        try testing.expectEqual(s.t, s.t + s.options.dt_min);
    } else if (endpoint) {
        s.options.t_stop = s.t + s.dt;
        s.dt = s.options.t_stop - s.t;
        s.uic_bp = null;
    } else {
        // Model an already-clipped pending breakpoint. `uic_bp` is an actual
        // nextBp source, so the accepted short retry must rediscover it.
        s.uic_bp = s.t + s.dt;
        s.bp_next = s.uic_bp;
        s.bp_clamped = true;
        s.bp_save_dt = 8 * s.options.dt_min;
    }
    const old_t = s.t;
    const old_dt = s.dt_old;
    const old_steps = s.steps;
    const old_cur = s.cur;
    const old_prev = s.prev;
    const old_bp = s.bp_next;
    var rows: [8][]f64 = undefined;
    var tapes: [8][]f64 = undefined;
    for (1..s.n_hist) |k| {
        rows[k - 1] = try gpa.dupe(f64, s.qHist()[k]);
        tapes[k - 1] = try gpa.dupe(f64, s.qtHist()[k]);
    }
    defer for (1..s.n_hist) |k| {
        gpa.free(rows[k - 1]);
        gpa.free(tapes[k - 1]);
    };
    const ip = try gpa.dupe(f64, s.i_prev);
    defer gpa.free(ip);
    const qip = try gpa.dupe(f64, s.qt_i_prev);
    defer gpa.free(qip);

    var ops: FloorRetryTestOps = .{ .proposal = s.options.dt_min / 4 };
    const a = (try s.begin(HostOps{})).?;
    var nr = try converger.run(&ckt, s.ws, s.trial, a.t_dev, a.nr_opts, a.hook);
    try testing.expect(nr.converged);
    if (newton_failure) nr.converged = false;
    try testing.expectEqual(if (nonadvancing) Verdict.underflow else Verdict.go, try s.end(&a, nr, &ops));
    try testing.expectEqual(@as(usize, 1), ops.reverts);
    try testing.expectEqual(@as(usize, 0), ops.advances);
    try testing.expectEqual(@as(usize, 0), ops.commits);
    try testing.expectEqual(old_t, s.t);
    try testing.expectEqual(old_steps, s.steps);
    try testing.expectEqualSlices(f64, &old_dt, &s.dt_old);
    try testing.expectEqual(old_cur.ptr, s.cur.ptr);
    try testing.expectEqual(old_prev.ptr, s.prev.ptr);
    try testing.expectEqual(old_bp, s.bp_next);
    for (1..s.n_hist) |k| {
        try testing.expectEqualSlices(f64, rows[k - 1], s.qHist()[k]);
        try testing.expectEqualSlices(f64, tapes[k - 1], s.qtHist()[k]);
    }
    try testing.expectEqualSlices(f64, ip, s.i_prev);
    try testing.expectEqualSlices(f64, qip, s.qt_i_prev);
    if (nonadvancing) return;
    try testing.expectEqual(s.options.dt_min, s.dt);
    const floor = s.dt;
    ops.proposal = 2 * floor;
    const retry = (try s.begin(HostOps{})).?;
    try testing.expectEqual(old_t + floor, retry.t_dev);
    nr = try converger.run(&ckt, s.ws, s.trial, retry.t_dev, retry.nr_opts, retry.hook);
    try testing.expect(nr.converged);
    if (persistent_failure) {
        if (newton_failure) nr.converged = false else ops.proposal = floor / 4;
        try testing.expectEqual(Verdict.underflow, try s.end(&retry, nr, &ops));
        try testing.expectEqual(@as(usize, 2), ops.reverts);
        try testing.expectEqual(@as(usize, 0), ops.advances);
        try testing.expectEqual(@as(usize, 0), ops.commits);
        try testing.expectEqual(old_t, s.t);
        try testing.expectEqual(old_steps, s.steps);
        try testing.expectEqualSlices(f64, &old_dt, &s.dt_old);
        try testing.expect(s.dt < s.options.dt_min);
        return;
    }
    try testing.expectEqual(Verdict.go, try s.end(&retry, nr, &ops));
    try testing.expectEqual(@as(usize, 1), ops.advances);
    try testing.expectEqual(@as(usize, 1), ops.commits);
    try testing.expectEqual(old_steps + 1, s.steps);
    try testing.expectEqual(old_t + floor, s.t);
    try testing.expectEqual(floor, s.dt_old[0]);
    try testing.expectEqual(old_dt[0], s.dt_old[1]);
    try testing.expectEqualSlices(f64, rows[0], s.qHist()[2]);
    try testing.expectEqualSlices(f64, tapes[0], s.qtHist()[2]);
    if (!endpoint) {
        try testing.expectEqual(old_bp, s.bp_next);
        try testing.expectEqual(@as(u32, 0), s.st.bp_landings);
    }
    const target = if (endpoint) s.options.t_stop else old_bp.?;
    // Continue genuine Newton solves under the controlled passing LTE bound.
    // The floor retry must neither skip the pending event nor cross TSTOP.
    var landed = false;
    for (0..8) |_| {
        const next = (try s.begin(HostOps{})) orelse break;
        try testing.expect(next.t_dev <= target);
        nr = try converger.run(&ckt, s.ws, s.trial, next.t_dev, next.nr_opts, next.hook);
        try testing.expect(nr.converged);
        try testing.expectEqual(Verdict.go, try s.end(&next, nr, &ops));
        if (s.t == target) {
            landed = true;
            break;
        }
    }
    try testing.expect(landed);
    if (endpoint) {
        try testing.expectEqual(@as(?Attempt, null), try s.begin(HostOps{}));
    } else try testing.expectEqual(@as(u32, 1), s.st.bp_landings);
}

test "floor retry: rejected Newton and LTE preserve histories and pending breakpoint" {
    for ([_]bool{ false, true }) |nr| try checkFloorRetryController(nr, false, false, false);
}

test "floor retry: Newton and LTE floor retries retain clipped endpoint" {
    for ([_]bool{ false, true }) |nr| try checkFloorRetryController(nr, false, true, false);
}

test "floor retry: nonrepresentable floor aborts Newton and LTE at small finite time" {
    for ([_]bool{ false, true }) |nr| try checkFloorRetryController(nr, true, false, false);
}

test "floor retry: persistent Newton and LTE rejection at the floor terminates" {
    for ([_]bool{ false, true }) |nr| try checkFloorRetryController(nr, false, false, true);
}
