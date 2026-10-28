//! `Sim.init`: a circuit at its operating point (or its `.ic` values under
//! uic) and the `.tran` options in, a pinned `Sim` at t = 0 out, with its
//! scratch, charge rings, LTE tape, seeded histories, breakpoint and VPI
//! setup, the first step and the t = 0 row.
//! Cites: ngspice dctran.c (the first step, CKTdeltaOld, the uic
//! breakpoint), traninit.c (delmin, minBreak), ckttrunc.c and indload.c
//! (one LTE state per charge, INDflux), capload.c (MODEUIC); Verilog-AMS
//! LRM §4.5.4, §4.5.7, §5.10.2 and §12.31.3; #161, #166.
//!
//! Contents:
//! - `pub fn init(s: *Sim, ckt: *root.Circuit, x: []f64, probes: []const u32, waveform: *Waveform, options: Options, allocator: std.mem.Allocator, tangents: ?*Tangents, stream: ?*Stream) !void`

const std = @import("std");
const assert = std.debug.assert;
const num = @import("core").numerics;
const converger = @import("solver").converger;
const Stream = @import("measure").Stream;
const root = @import("../types.zig");
const host_ops = @import("host_ops.zig");
const integrator = @import("integrator.zig");
const Sim = @import("Sim.zig");
const Tangents = @import("Tangents.zig");
const tran_types = @import("types.zig");

const Options = tran_types.Options;
const Waveform = tran_types.Waveform;
const TranHook = Sim.TranHook;
const lteSnap = host_ops.lteSnap;
const nextBp = Sim.nextBp;
const newtonOptions = Sim.newtonOptions;

/// Everything `simulateWith` did before its step loop, in its order:
/// scratch, the charge rings, uic bookkeeping, the baseline, history
/// seeding, the tangents' operating point, delay-line seeding, the
/// breakpoint and VPI setup, and the t = 0 row. Leaves the device state
/// committed at t = 0 and `land_rejects` set; `deinit` clears it.
pub fn init(
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

// ---- private ----
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
    ckt.reject_at = null;
    const nr = converger.run(ckt, ws, scratch, 0, newtonOptions(options), hook) catch |err| switch (err) {
        error.QueryCancelled, error.OutOfMemory => return err,
        else => null, // else: a numerical failure keeps the `.ic` row
    };
    _ = ckt.stateCtl(.revert);
    return if (nr != null and nr.?.converged) scratch else x;
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
