//! Floor-retry controller tests for `Sim.end`: real generated devices and
//! HostOps histories, with only the LTE proposal or the Newton outcome
//! injected, so each retry branch is deterministic.
//! Oracle: the state a rejected step must leave untouched (histories, the
//! pending breakpoint, t, the step count), and ngspice dctran.c:974's floor
//! rule for what the retry must do.
//!
//! Contents: none; `FloorRetryTestOps` is private, its `pub` methods (`revert`,
//! `rejectAt`, `flipped`, `boundStep`, `charges`, `bound`, `boundGear`,
//! `advance`, `commit`, `rebase`, `accepted`) the `ops` seam `Sim.end` calls.

const std = @import("std");
const converger = @import("solver").converger;
const root = @import("../types.zig");
const integrator = @import("integrator.zig");
const Sim = @import("Sim.zig");
const tran_types = @import("types.zig");

const Attempt = Sim.Attempt;
const HostOps = @import("host_ops.zig").HostOps;
const Method = tran_types.Method;
const Verdict = Sim.Verdict;
const Waveform = tran_types.Waveform;

// Controller tests use real generated devices and HostOps histories. Only the
// LTE proposal/Newton outcome is injected, so each retry branch is deterministic.
const FloorRetryTestOps = struct {
    proposal: f64,
    reverts: usize = 0,
    advances: usize = 0,
    commits: usize = 0,
    pub fn revert(o: *@This(), s: *Sim) void {
        o.reverts += 1;
        (HostOps{}).revert(s);
    }
    pub fn rejectAt(_: *@This(), s: *Sim) ?f64 {
        return (HostOps{}).rejectAt(s);
    }
    pub fn flipped(_: *@This(), s: *Sim, a: *const Attempt, ask: bool) bool {
        return (HostOps{}).flipped(s, a, ask);
    }
    pub fn boundStep(_: *@This(), s: *Sim) ?f64 {
        return (HostOps{}).boundStep(s);
    }
    pub fn charges(_: *@This(), s: *Sim, a: *const Attempt) void {
        (HostOps{}).charges(s, a);
    }
    pub fn bound(o: *@This(), _: *Sim, _: *const Attempt, _: Method, _: integrator.LteIn) f64 {
        return o.proposal;
    }
    pub fn boundGear(o: *@This(), _: *Sim, _: *const Attempt, _: Method, _: integrator.LteIn) f64 {
        return o.proposal;
    }
    pub fn advance(o: *@This(), s: *Sim, a: *const Attempt) void {
        o.advances += 1;
        (HostOps{}).advance(s, a);
    }
    pub fn commit(o: *@This(), s: *Sim) void {
        o.commits += 1;
        (HostOps{}).commit(s);
    }
    pub fn rebase(_: *@This(), s: *Sim, a: *const Attempt) void {
        (HostOps{}).rebase(s, a);
    }
    pub fn accepted(_: *@This(), s: *Sim) []const f64 {
        return (HostOps{}).accepted(s);
    }
};

// ---- private ----

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
