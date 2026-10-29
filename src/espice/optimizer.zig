//! A deck's pending whole-run work in, rewritten variant rows out: the
//! HSPICE `.model OPT` optimizations (Levenberg-Marquardt fits or
//! bisections), each point batch run as one session of the optimized card's
//! queries, and the MOSRA stress run with its degradation table. Both run
//! once, before any query of the Problem advances.
//! Cites: dev/analysis/optimize.md, dev/analysis/mosra.md, [SA Ch.19].
//!
//! Contents:
//! - `pub fn runPending(self: *Problem) !void`
//! - `pub fn tangentable(self: *const Problem, job: Query) bool`

const std = @import("std");
const assert = std.debug.assert;
const analysis = @import("analysis");
const core = @import("core");
const frontend = @import("frontend");
const output = @import("output");
const Problem = @import("Problem.zig");
const requests = core.query;
const Query = requests.Query;
const QueryId = requests.QueryId;
const Session = analysis.session.Session;

/// Runs the deck's optimizations and then its MOSRA stress run, each the
/// first time only; a deck with neither returns at once. Fails with the
/// first failing point batch's or stress run's error.
pub fn runPending(self: *Problem) !void {
    try optimize(self);
    try age(self);
    if (self.prepared.tuner) |t| assert(self.optimization.len + self.bisection.len == t.bases.len);
    if (self.prepared.deck.mosra != null) assert(self.aged);
}

/// Whether `job` can give the optimizer its Jacobian as tangents of its own
/// run. A transient cannot from an operating point it does not solve on its
/// own (`uic`, `.ic` holds, a capacitor-only node's OPtran), under
/// Gear-3..6, under VPI callbacks, or off the CPU: its rows swap host
/// device state (`Tran.tangents`).
pub fn tangentable(self: *const Problem, job: Query) bool {
    const c = &self.prepared.circuit;
    return switch (job) {
        .dc => true,
        // Tangents carry Gear-2's charge history only (validate.zig).
        .tran => |o| !o.uic and !o.snapshot and !o.sources_at_zero and self.prepared.deck.ic.len == 0 and
            @backingInt(o.method) <= @backingInt(requests.Method.gear_2) and
            !c.needs_tran_op and c.vpi == null and self.session.config.backend == .cpu,
        else => false,
    };
}

// ---- private ----

/// Runs the deck's HSPICE optimizations, one per `Tuner.bases` entry in
/// turn. Each batch of optimizer points is one session of the optimized
/// card's queries, a variant row per point, run `max_parallel` at a time.
/// Each optimum's writes then fill the rows its optimized card and every
/// later card run.
fn optimize(self: *Problem) !void {
    const t = if (self.prepared.tuner) |*tuner| tuner else return;
    if (self.optimization.len != 0 or self.bisection.len != 0) return;
    const arena = self.arena.allocator();
    const deck = &self.prepared.deck;
    const n = t.bases.len;
    if (t.method == .lm) self.optimization = try arena.alloc(core.lm.Lm, n) else self.bisection = try arena.alloc(core.bisect.Bisect, n);
    const best = try arena.alloc(core.Variants, n);
    var template: std.ArrayList(Query) = .empty;
    for (best, 0..) |*row, base| {
        template.clearRetainingCapacity();
        for (deck.queries) |job| switch (job) {
            inline else => |o| if (o.tol.variant == @as(u32, @intCast(base))) try template.append(arena, job),
        };
        const x: []const f64 = if (t.method == .lm) fit: {
            // DC and transient goals take their Jacobian from the trial
            // point's own run (`Dc.tangents`, `Tran.tangents`), so every
            // round is batched.
            const direct = !self.session.config.knobs.no_tangents and for (template.items) |job| {
                if (!tangentable(self, job)) break false;
            } else true;
            const lm = &self.optimization[base];
            if (base == 0) for (t.initial, t.spec.dels, t.names) |x0, del, name| if (t.options.stepOverreach(x0, del)) |ratio|
                std.log.warn("optimize {s}: {s} = {e} is far below PARMIN = {e}, so its finite-difference step is {d:.0}x the value and can land on a bound; set PARMIN near the parameter's scale", .{ t.spec.name, name, x0, t.options.parmin, ratio });
            lm.* = try core.lm.Lm.init(arena, t.initial, t.spec.lo, t.spec.hi, t.spec.dels, t.spec.grid, @intCast(t.results.len), t.options, direct or self.limits.max_parallel > 1);
            try search(self, t, template.items, base, lm, direct);
            break :fit lm.x;
        } else halve: {
            const goal = deck.measures[t.results[0]];
            const c = if (goal.first.goal != null) goal.first else goal.second;
            const scale = @max(@abs(c.goal.?), c.minval) / c.weight;
            const b = &self.bisection[base];
            b.* = core.bisect.Bisect.init(arena, t.method, t.spec.lo[0], t.spec.hi[0], t.spec.grid[0], scale, t.options);
            try search(self, t, template.items, base, b, false);
            break :halve (&b.x)[0..1];
        };
        row.* = try t.rows(arena, &self.prepared.circuit, x, base);
    }
    deck.variants = try frontend.Tuner.optimized(arena, deck.variants, best);
}

/// HSPICE MOSRA: runs the stress transient with the bound MOSFETs'
/// terminals as its only outputs, fills the aged rows' `delvto`/`mulu0`
/// writes and publishes the degradation table.
// ponytail: the stress transient repeats the fresh one; fold them into one
// run when MOSRA decks get large.
fn age(self: *Problem) !void {
    const m = self.prepared.deck.mosra orelse return;
    if (self.aged) return;
    const arena = self.arena.allocator();
    var deck = self.prepared.deck;
    const probes = try arena.alloc(u32, 3 * m.terminals.len);
    for (m.terminals, 0..) |t, i| probes[3 * i ..][0..3].* = t;
    const labels = try arena.alloc([]const u8, probes.len);
    @memset(labels, "");
    deck.probes = probes;
    deck.probe_labels = labels;
    var session = Session.init(self.query_memory.allocator(), self.io, &self.prepared.circuit, &deck, self.session.config);
    defer session.deinit();
    var id: [1]QueryId = undefined;
    _ = try session.append(&.{.{ .tran = m.tran }}, &id);
    try session.runAll(.{ .max_parallel = self.limits.max_parallel, .quantum = .completion });
    if (session.failure()) |err| return err;
    const aged = try analysis.mosra.age(arena, m, try session.result(id[0]));
    if (m.aged_runs) self.prepared.deck.variants.values = aged.values;
    self.aged = true;
    try self.delivery.publish(self.io, .{ .title = deck.title, .result = aged.table });
}

/// Feeds `solver` (`*core.lm.Lm` or `*core.bisect.Bisect`) the residuals of
/// the points it asks for until it stops. A Levenberg-Marquardt point's
/// operating point starts from the one of the iterate its step left
/// (dev/analysis/optimize.md, "Warm starts"), so the path does not depend
/// on batching; the start's round and every bisection point start cold.
/// `direct` as `evaluatePoints` says.
// Comptime `solver`: the axis is the search method, LM and bisection both
// exist today, and their `points`/`feed` loops differ only in the warm
// start, which the comptime `lm` flag selects.
fn search(self: *Problem, t: *frontend.Tuner, template: []const Query, base: usize, solver: anytype, direct: bool) !void {
    const n = t.spec.live.len;
    const m = t.results.len;
    const lm = @TypeOf(solver) == *core.lm.Lm;
    comptime assert(lm or @TypeOf(solver) == *core.bisect.Bisect);
    const gpa = self.gpa;
    var residuals: std.ArrayList(f64) = .empty;
    defer residuals.deinit(gpa);
    // The operating points of the accepted iterate, of row 0 of the last
    // round, and the one this round starts from.
    var origin: std.ArrayList(f64) = .empty;
    defer origin.deinit(gpa);
    var first: std.ArrayList(f64) = .empty;
    defer first.deinit(gpa);
    var guess: std.ArrayList(f64) = .empty;
    defer guess.deinit(gpa);
    while (solver.points().len != 0) {
        const points = solver.points();
        assert(points.len % n == 0);
        if (lm) if (solver.stepping()) {
            guess.clearRetainingCapacity();
            try guess.appendSlice(gpa, origin.items);
        };
        try residuals.resize(gpa, points.len / n * m);
        try evaluatePoints(self, t, template, base, points, if (lm) direct and solver.batched() else false, guess.items, residuals.items, &first);
        const accepted = if (lm) solver.history.rss.items.len else 0;
        try solver.feed(residuals.items);
        if (lm) if (solver.history.rss.items.len > accepted) std.mem.swap(std.ArrayList(f64), &origin, &first);
    }
}

/// Residuals of each point of `points` (the optimized parameters' values, n
/// per point, on top of `t.bases[base]`) into `out`, m per point: the
/// RESULTS cards' goal errors over the `template` queries run at that
/// point, NaN where a query failed or a card found no value. Every
/// operating point starts from `guess` (cold when empty); `first` takes the
/// one row 0's first query solved, empty when it has none. With `direct` (a
/// batched round of `tangentable` queries) only row 0 runs, and the other
/// points' results are its runs' first-order predictions of them
/// (`Dc.tangents`, `Tran.tangents`, dev/analysis/optimize.md "Direct
/// sensitivities").
fn evaluatePoints(self: *Problem, t: *frontend.Tuner, template: []const Query, base: usize, points: []const f64, direct: bool, guess: []const f64, out: []f64, first: *std.ArrayList(f64)) !void {
    var arena_state = std.heap.ArenaAllocator.init(self.gpa);
    defer arena_state.deinit();
    const scratch = arena_state.allocator();
    const count = points.len / t.spec.live.len;
    const m = t.results.len;
    if (direct) assert(count == t.spec.live.len + 1);
    assert(out.len == count * m);
    var deck = self.prepared.deck;
    deck.variants = try t.rows(scratch, &self.prepared.circuit, points, base);
    deck.op_guess = guess;
    const runs = if (direct) 1 else count;
    const tangents = try scratch.alloc(u32, if (direct) count - 1 else 0);
    for (tangents, 1..) |*row, p| row.* = @intCast(p);
    const jobs = try scratch.alloc(Query, runs * template.len);
    for (0..runs) |p| for (template, jobs[p * template.len ..][0..template.len]) |job, *copy| {
        copy.* = job;
        switch (copy.*) {
            inline else => |*o| o.tol.variant = @intCast(p),
        }
        if (direct) switch (copy.*) {
            .dc => |*o| o.tangents = tangents,
            .tran => |*o| o.tangents = tangents,
            else => unreachable, // else: `tangentable` admits these two
        };
    };
    var session = Session.init(self.query_memory.allocator(), self.io, &self.prepared.circuit, &deck, self.session.config);
    defer session.deinit();
    const ids = try scratch.alloc(QueryId, jobs.len);
    _ = try session.append(jobs, ids);
    try session.runAll(.{ .max_parallel = self.limits.max_parallel, .quantum = .completion });
    first.clearRetainingCapacity();
    if (try session.operatingPoint(ids[0])) |x| try first.appendSlice(self.gpa, x);
    const values = try scratch.alloc(f64, deck.measures.len);
    @memset(out, std.math.nan(f64));
    for (0..count) |p| for (ids[(if (direct) 0 else p) * template.len ..][0..template.len]) |id| {
        const info = try session.info(id);
        if (info.status != .complete) continue;
        var res = try session.result(id);
        if (direct) {
            // Block p of the sweep: its own points, or row p's prediction.
            // A transient keeps its tangent rows apart (`Result.tangents`,
            // as `points.block` reads them); a `.dc` sweep holds them inline.
            if (res.tangents.len != 0) {
                if (p != 0) {
                    const n = res.npoints * res.varnames.len;
                    res.data = res.tangents[(p - 1) * n ..][0..n];
                    res.measured = &.{};
                }
                res.tangents = &.{};
            } else {
                res.npoints /= count;
                res.data = res.data[p * res.npoints * res.varnames.len ..][0 .. res.npoints * res.varnames.len];
            }
        }
        output.measureValues(deck.measures, info.kind, res, values);
        for (t.results, out[p * m ..][0..m]) |card, *r| {
            const measure = deck.measures[card];
            // Never null: Tuner.init refuses a RESULTS card without GOAL=
            // (`goalError(0) != null`), and only the goal decides null.
            if (measure.analysis == info.kind) r.* = measure.goalError(values[card]).?;
        }
    };
}
