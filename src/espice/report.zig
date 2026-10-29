//! Finished results in, the text a deck's cards ask for out: each
//! optimization's summary, each result's `.meas` block in request order,
//! the Monte Carlo statistics, and the `--tangents` slopes. `run_all`
//! streams the blocks as their turn comes (`printReady`); `print_measures`
//! writes what is left.
//! Cites: none for the `.meas` blocks (output/measure.zig owns their
//! format); the optimization listings follow HSPICE's [SA Ch.19].
//!
//! Contents:
//! - `pub const Cursor = struct { headed: bool, session: usize, next: usize }`
//! - `pub const LiveMeasures = struct`
//! - `pub fn printMeasures(self: *Problem, out: *std.Io.Writer, err: *std.Io.Writer) !void`
//! - `pub fn printTangents(self: *const Problem, out: *std.Io.Writer) !void`
//! - `pub fn printReady(self: *Problem) void`

const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;
const analysis = @import("analysis");
const core = @import("core");
const frontend = @import("frontend");
const output = @import("output");
const config = @import("config.zig");
const points = @import("points.zig");
const Problem = @import("Problem.zig");
const QueryId = core.query.QueryId;
const Session = analysis.session.Session;

/// The next `.meas` block to print: whether the optimization summaries are
/// out, the session (0 the main one, k the k-th run's) and its output.
pub const Cursor = struct { headed: bool = false, session: usize = 0, next: usize = 0 };

/// `Options.measures` and how far `run_all` has printed to it. `failure` is
/// the first error printing there; once set nothing more prints, and
/// `print_measures` returns it.
pub const LiveMeasures = struct {
    to: config.MeasureWriters,
    cursor: Cursor = .{},
    failure: ?anyerror = null,
};

/// What `printFrom` does at a result still running.
const Running = enum { wait_running, skip_running };

/// Writes each optimization's summary, then the deck's `.meas` results for
/// every completed result a card can read, in request order, then the
/// Monte Carlo statistics (`Problem.print_measures` documents the
/// contract).
pub fn printMeasures(self: *Problem, out: *std.Io.Writer, err: *std.Io.Writer) !void {
    const measures = self.prepared.deck.measures;
    if (self.live_measures) |*live| {
        if (live.failure) |e| return e;
        try printFrom(self, &live.cursor, out, err, .skip_running);
    } else {
        var cursor: Cursor = .{};
        try printFrom(self, &cursor, out, err, .skip_running);
    }
    if (measures.len == 0) return;
    // Statistics over the Monte Carlo trials of each analysis kind.
    var trials: std.ArrayList(output.Result) = .empty;
    defer trials.deinit(self.gpa);
    inline for (.{ .tran, .ac, .dc }) |kind| {
        trials.clearRetainingCapacity();
        try monteTrials(self.gpa, &self.session, kind, &trials);
        for (self.runs) |*run| try monteTrials(self.gpa, &run.session, kind, &trials);
        if (trials.items.len != 0) try output.printMeasureStatistics(out, measures, kind, trials.items);
    }
}

/// With `Options.point.tangent_step`, writes d(card)/d(live parameter) of
/// every `.tran` card at the deck's first transient, one
/// `d(card)/d(param) = value` line each; NaN where a card has no value.
/// Nothing without tangents.
pub fn printTangents(self: *const Problem, out: *std.Io.Writer) !void {
    const id = for (self.session.outputs.items) |id| {
        if (self.tangent_steps.contains(id)) break id;
    } else return;
    var names: std.ArrayList([]const u8) = .empty;
    defer names.deinit(self.gpa);
    for (self.prepared.deck.measures) |m| if (m.analysis == .tran) try names.append(self.gpa, m.name);
    if (names.items.len == 0) return;
    const p = self.live.len;
    assert(p == self.live_names.len);
    const slopes = try self.gpa.alloc(f64, names.items.len * p);
    defer self.gpa.free(slopes);
    try points.pointTangents(self, id, 1, names.items, slopes);
    try out.writeAll("\n  Tangents for Transient Analysis\n\n");
    for (names.items, 0..) |name, c| for (self.live_names, 0..) |param, k| {
        try out.print("d({s})/d({s}) = {e:.6}\n", .{ name, param, slopes[c * p + k] });
    };
}

/// Prints to `Options.measures` every `.meas` block whose turn has come and
/// flushes both writers; the first error stops it for good.
pub fn printReady(self: *Problem) void {
    const live = if (self.live_measures) |*live| live else return;
    if (live.failure != null) return;
    printFrom(self, &live.cursor, live.to.out, live.to.err, .wait_running) catch |e| {
        live.failure = e;
        return;
    };
    live.to.out.flush() catch |e| {
        live.failure = e;
    };
    live.to.err.flush() catch |e| {
        live.failure = e;
    };
}

// ---- private ----

/// The optimization summaries, then each session's `.meas` blocks in
/// request order from `cursor` on; at a result still running it stops
/// (`.wait_running`) or prints nothing for it (`.skip_running`).
fn printFrom(self: *const Problem, cursor: *Cursor, out: *std.Io.Writer, err: *std.Io.Writer, running: Running) !void {
    const measures = self.prepared.deck.measures;
    if (!cursor.headed) {
        for (self.optimization, 0..) |*lm, k| try printOptimization(out, lm, &self.prepared.tuner.?, k, measures);
        for (self.bisection, 0..) |*b, k| try printBisection(out, b, &self.prepared.tuner.?, k, measures);
        cursor.headed = true;
    }
    if (measures.len == 0) return;
    while (cursor.session <= self.runs.len) : ({
        cursor.session += 1;
        cursor.next = 0;
    }) {
        const session = if (cursor.session == 0) &self.session else &self.runs[cursor.session - 1].session;
        while (cursor.next < session.outputs.items.len) : (cursor.next += 1) {
            const id = session.outputs.items[cursor.next];
            if (running == .wait_running and !(try session.info(id)).status.terminal()) return;
            try printQueryMeasures(session, id, out, err, measures);
        }
    }
    assert(cursor.session == self.runs.len + 1);
}

/// `.meas` results over query `id`'s result when it completed, of a kind a
/// card can read; a transient's own run, not its tangent blocks. A sweep
/// point's block names its row and label, and prints NaN for each card
/// without a value, a failed point's for all.
fn printQueryMeasures(session: *const Session, id: QueryId, out: *std.Io.Writer, err: *std.Io.Writer, measures: []const core.Measure) !void {
    const info = try session.info(id);
    switch (info.kind) {
        .op, .tran, .ac, .dc, .fft, .tran_noise, .dcmatch, .acmatch, .lstb, .phasenoise, .pnoise, .sp => {},
        else => return,
    }
    if (session.released(id)) return;
    const row = points.variantOf(try session.query(id));
    const point: ?output.MeasurePoint = if (row) |v| .{ .row = v, .label = session.deck.variants.labels[v] } else null;
    if (info.status == .complete) {
        try output.printMeasures(out, err, measures, info.kind, try points.block(session, id, 0), point);
    } else if (point) |p| if (info.status.terminal()) try output.printFailedMeasures(out, measures, info.kind, p);
}

/// Appends `session`'s completed Monte Carlo trial results of `kind`.
fn monteTrials(gpa: Allocator, session: *const Session, kind: core.query.Kind, trials: *std.ArrayList(output.Result)) !void {
    for (session.outputs.items) |id| {
        const info = try session.info(id);
        // `releaseDelivered` keeps every trial.
        if (info.status != .complete or info.kind != kind or session.released(id)) continue;
        const res = try session.result(id);
        if (std.mem.indexOf(u8, res.plotname, "(monte=") != null) try trials.append(gpa, res);
    }
}

/// The optimizer's summary, after HSPICE's: one line per accepted
/// iteration, the stop reason, the final figures, the optimized parameters
/// and each RESULTS card's error at the optimum.
fn printOptimization(out: *std.Io.Writer, lm: *const core.lm.Lm, t: *const frontend.Tuner, base: usize, measures: []const core.Measure) !void {
    assert(base < t.bases.len);
    const o = t.options;
    try out.print("\n  Optimization {s} (Levenberg-Marquardt, model {s}){s}{s}\n\n  iter  evals  residual sum of squares  marquardt param", .{ t.spec.name, t.spec.model, if (t.bases[base].label.len != 0) " at " else "", t.bases[base].label });
    for (t.names) |name| try out.print("  {s:>14}", .{name});
    try out.writeAll("\n");
    const h = lm.history;
    for (h.rss.items, h.lambda.items, h.evaluations.items, 0..) |rss, lambda, evals, k| {
        try out.print("  {d:>4}  {d:>5}  {e:>23.6}  {e:>15.6}", .{ k, evals, rss, lambda });
        for (h.x.items[k * lm.n ..][0..lm.n]) |x| try out.print("  {e:>14.6}", .{x});
        try out.writeAll("\n");
    }
    try out.writeAll("\n");
    var pinned = false;
    for (0..lm.n) |j| pinned = pinned or lm.pinned(j) != 0;
    if (lm.status != .failed and pinned) {
        try out.writeAll("  optimization stopped at a limit: the goals are not reachable inside the parameter ranges\n");
    } else switch (lm.status) {
        .relin => try out.print("  optimization completed: RELIN = {e} on last iteration\n", .{o.relin}),
        .relout => try out.print("  optimization completed: RELOUT = {e} on last iteration\n", .{o.relout}),
        .grad => try out.print("  optimization completed: norm of the gradient < GRAD = {e}\n", .{o.grad}),
        .itropt => try out.print("  optimization incomplete: ITROPT = {d} iterations reached\n", .{o.itropt}),
        .max => try out.print("  optimization stopped: marquardt parameter above MAX = {e}\n", .{o.max}),
        .failed, .running => try out.writeAll("  optimization failed: a point could not be simulated or measured\n"),
    }
    for (t.names, 0..) |name, j| switch (lm.pinned(j)) {
        -1 => try out.print("  {s} is held at its lower limit {e:.6}\n", .{ name, lm.lo[j] }),
        1 => try out.print("  {s} is held at its upper limit {e:.6}\n", .{ name, lm.hi[j] }),
        else => {},
    };
    try out.print(
        \\  residual sum of squares       = {e:.6}
        \\  norm of the gradient          = {e:.6}
        \\  marquardt scaling parameter   = {e:.6}
        \\  no. of function evaluations   = {d}
        \\  no. of iterations             = {d}
        \\
        \\  optimized parameters {s} -- final values
        \\
    , .{ lm.rss, lm.gradientNorm(), lm.lambda, lm.evaluations, lm.iterations, t.spec.name });
    for (t.names, lm.x, t.initial, lm.lo, lm.hi, lm.grid) |name, x, x0, lo, hi, g| {
        try out.print("  {s} = {e:.6} $ initial {e:.6}, range {e:.6} to {e:.6}", .{ name, x, x0, lo, hi });
        if (g > 0) try out.print(", grid {e:.6}", .{g});
        try out.writeAll("\n");
    }
    try out.writeAll("\n  goal errors at the optimum (weight * (result - goal) / max(|goal|, minval))\n");
    for (t.results, lm.r) |card, r| try out.print("  error({s}) = {e:.6}\n", .{ measures[card].name, r });
}

/// The bisection summary: one line per midpoint test with the window it
/// halved, the stop reason, and the last passing value [SA Ch.19].
fn printBisection(out: *std.Io.Writer, b: *const core.bisect.Bisect, t: *const frontend.Tuner, base: usize, measures: []const core.Measure) !void {
    assert(base < t.bases.len);
    assert(t.results.len != 0);
    const o = t.options;
    try out.print("\n  Optimization {s} ({s}, model {s}){s}{s}\n\n  iter             xlo             xhi               x      goal error\n", .{ t.spec.name, @tagName(b.method), t.spec.model, if (t.bases[base].label.len != 0) " at " else "", t.bases[base].label });
    const h = b.history;
    for (h.lo.items, h.hi.items, h.x.items, h.r.items, 1..) |lo, hi, x, r, k|
        try out.print("  {d:>4}  {e:>14.6}  {e:>14.6}  {e:>14.6}  {e:>14.6}\n", .{ k, lo, hi, x, r });
    try out.writeAll("\n");
    switch (b.status) {
        .converged => if (o.absin > 0)
            try out.print("  optimization completed: ABSIN = {e} satisfied\n", .{o.absin})
        else if (b.method == .passfail)
            try out.print("  optimization completed: RELIN = {e} satisfied\n", .{o.relin})
        else
            try out.print("  optimization completed: RELIN = {e} and {s} = {e} satisfied\n", .{ o.relin, if (o.absout > 0) "ABSOUT" else "RELOUT", if (o.absout > 0) o.absout else o.relout }),
        .itropt => try out.print("  optimization incomplete: ITROPT = {d} iterations reached\n", .{o.itropt}),
        .bounds => try out.print("  optimization failed: both limits of {s} {s}, so they do not bracket the goal\n", .{ t.names[0], if (b.pass_hi) "pass" else "fail" }),
        .failed, .running => try out.writeAll("  optimization failed: a point could not be simulated or measured\n"),
    }
    try out.print(
        \\  no. of function evaluations   = {d}
        \\  no. of iterations             = {d}
        \\
        \\  optimized parameters {s} -- final values
        \\  {s} = {e:.6} $ last passing value, range {e:.6} to {e:.6}
        \\
    , .{ b.evaluations, b.iterations, t.spec.name, t.names[0], b.x, t.spec.lo[0], t.spec.hi[0] });
    if (b.grid > 0) try out.print("  grid {e:.6}\n", .{b.grid});
    try out.print("  error({s}) = {e:.6}\n", .{ measures[t.results[0]].name, b.r });
}
