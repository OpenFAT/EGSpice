//! Live-parameter points in, query rows and their measures out: the
//! `set_params` and `append_points` rows a sizing loop appends, the tangent
//! rows beside them, the warm operating point they start from, and the
//! `.meas` values, slopes and releases read back per point. A point is a
//! run of consecutive requested queries on one variant row.
//! Cites: dev/analysis/variants.md §5 (re-evaluation without a re-parse).
//!
//! Contents:
//! - `pub const Reason = enum(u32)`
//! - `pub fn setParams(self: *Problem, names: []const []const u8, values: []const f64) !void`
//! - `pub fn appendPoints(self: *Problem, values: []const f64, count: usize, options: requests.PointOptions) !QueryId`
//! - `pub fn checkPoint(self: *const Problem, point: requests.PointOptions) error{UnknownMeasure}!void`
//! - `pub fn tangentDeck(self: *Problem, jobs: []Query, h: f64, steps: [][]const f64) !void`
//! - `pub fn solveWarmOp(self: *Problem, jobs: []const Query) !void`
//! - `pub fn pointMeasures(self: *const Problem, first: QueryId, names: []const []const u8, out: []f64, reasons: []Reason) !void`
//! - `pub fn releasePoints(self: *Problem, first: QueryId, count: usize) !void`
//! - `pub fn pointTangents(self: *const Problem, first: QueryId, count: usize, names: []const []const u8, out: []f64) !void`
//! - `pub fn variantOf(job: Query) ?u32`
//! - `pub fn block(session: *const Session, id: QueryId, k: usize) !output.Result`

const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;
const analysis = @import("analysis");
const core = @import("core");
const frontend = @import("frontend");
const output = @import("output");
const optimizer = @import("optimizer.zig");
const Problem = @import("Problem.zig");
const requests = core.query;
const Query = requests.Query;
const QueryId = requests.QueryId;
const Session = analysis.session.Session;

/// Why a point produced no value, or not all of them: the C API's
/// ESPICE_REASON_* codes.
pub const Reason = enum(u32) {
    ok = 0,
    no_convergence = 1,
    timestep_too_small = 2,
    singular_matrix = 3,
    measure_not_found = 4,
    measure_unresolved = 5,
    aborted = 6,
    model_bin_not_found = 7,
    topology_changed = 8,
    invalid_parameter = 9,
    device_refused = 10,
    other = 255,

    /// The reason a query that failed with `err` gives.
    pub fn of(err: anyerror) Reason {
        return switch (err) {
            error.OpDidNotConverge, error.HbDidNotConverge, error.QpssDidNotConverge, error.PzDidNotConverge, error.MatexDidNotConverge, error.EnvelopeDidNotConverge, error.PacDidNotConverge => .no_convergence,
            error.TimestepTooSmall => .timestep_too_small,
            error.SingularMatrix, error.Singular, error.FloatingNode => .singular_matrix,
            error.MeasureBoundExceeded => .aborted,
            error.ModelBinNotFound => .model_bin_not_found,
            error.TopologyChanged, error.NoiseTopologyChanged => .topology_changed,
            error.NonFiniteParameter => .invalid_parameter,
            error.DeviceRefused => .device_refused,
            // `err` is anyerror: the full name stays with the query
            // (`QueryInfo.failure`).
            else => .other,
        };
    }
};

/// `Prepared.tuning`'s payload: the re-evaluator of the live parameters.
const Tuning = @typeInfo(@FieldType(frontend.Prepared, "tuning")).optional.child;

/// Sets each global `.param` in `names` to its value for every query
/// appended afterwards (`Problem.set_params` documents the contract).
/// Asserts that the new template index exists.
pub fn setParams(self: *Problem, names: []const []const u8, values: []const f64) !void {
    const tuning = self.prepared.tuning orelse return error.UnknownParameter;
    if (names.len != values.len) return error.InvalidArgument;
    // The optimizer and MOSRA rewrite the variant table once; let them
    // run before rows are appended to it.
    try optimizer.runPending(self);
    try self.session.templates.ensureUnusedCapacity(self.session.allocator, 1);
    const change = try tuning.set(&self.prepared.circuit, self.prepared.deck.variants, names, values);
    // The capacity was reserved above.
    if (change.built) |c| self.session.addTemplate(c) catch unreachable;
    assert(change.template <= self.session.templates.items.len);
    self.session.template = change.template;
    self.prepared.deck.variants = change.variants;
    self.variant = change.variants.count() - 1;
    assert(self.variant.? < self.prepared.deck.variants.count());
}

/// Appends `count` points, each the deck's analyses at one row of `values`
/// (`Problem.append_points` documents the contract), and returns point 0's
/// first query id. Asserts that `values` holds `count` rows.
pub fn appendPoints(self: *Problem, values: []const f64, count: usize, options: requests.PointOptions) !QueryId {
    const tuning = self.prepared.tuning orelse return error.NoLiveParameters;
    const p = self.live.len;
    assert(values.len == count * p);
    if (count == 0) return error.InvalidArgument;
    try checkPoint(self, options);
    const deck = &self.prepared.deck;
    for (deck.queries) |job| if (variantOf(job) != null) return error.SweptTemplate;
    try optimizer.runPending(self);
    const lone_op = [_]Query{.{ .op = .{ .tol = deck.deck_tol } }};
    const template_jobs: []const Query = if (deck.queries.len == 0) &lone_op else deck.queries;
    try self.checkOutput(&self.prepared, template_jobs);
    const tangent = options.tangent_step > 0;
    if (tangent) for (template_jobs) |job| if (job == .tran and !optimizer.tangentable(self, job)) return error.TangentsUnavailable;
    if (options.warm_start) try solveWarmOp(self, template_jobs);
    var arena_state = std.heap.ArenaAllocator.init(self.gpa);
    defer arena_state.deinit();
    const scratch = arena_state.allocator();
    const j = template_jobs.len;
    const jobs = try scratch.alloc(Query, count * j);
    const templates = try scratch.alloc(u32, count * j);
    const refused = try scratch.alloc(?analysis.session.PointError, count);
    // Each point's tangent steps, kept with its transient.
    const steps = try scratch.alloc([]const f64, count);
    // A built template becomes the session's current one; `set_params`
    // rows appended later keep theirs.
    const current = self.session.template;
    defer self.session.template = current;
    for (0..count) |i| {
        const point_values = values[i * p ..][0..p];
        try self.session.templates.ensureUnusedCapacity(self.session.allocator, 1);
        refused[i] = null;
        const template = if (tuning.setAt(&self.prepared.circuit, deck.variants, self.live, point_values)) |change| built: {
            // The capacity was reserved above.
            if (change.built) |c| self.session.addTemplate(c) catch unreachable;
            deck.variants = change.variants;
            break :built change.template;
        } else |err| failed: {
            refused[i] = switch (err) {
                error.UnsupportedCard => error.TopologyChanged,
                error.ModelBinNotFound => error.ModelBinNotFound,
                error.InvalidArgument => error.NonFiniteParameter,
                error.OutOfMemory => |e| return e,
                // Any other failure belongs to this point's values (a device
                // rejecting them as it builds), not to the batch.
                else => |e| other: {
                    std.log.warn("point {d}: {s}", .{ i, @errorName(e) });
                    break :other error.PointRefused;
                },
            };
            deck.variants = try tuning.refusedRow(deck.variants);
            break :failed 0;
        };
        const row = deck.variants.count() - 1;
        const rows: []const u32 = if (tangent and refused[i] == null) try tangentRows(self, tuning, point_values, options.tangent_step, &steps[i]) else &.{};
        if (rows.len == 0) steps[i] = &.{};
        for (template_jobs, jobs[i * j ..][0..j], templates[i * j ..][0..j]) |job, *copy, *t| {
            copy.* = job;
            switch (copy.*) {
                inline else => |*o| o.tol.variant = row,
            }
            if (copy.* == .tran) {
                copy.tran.point = options;
                copy.tran.tangents = rows;
            }
            t.* = template;
        }
    }
    const ids = try scratch.alloc(QueryId, jobs.len);
    _ = try self.session.appendAt(jobs, templates, ids);
    // Each id was just appended, so every lookup below finds it.
    for (refused, 0..) |r, i| if (r) |err| for (ids[i * j ..][0..j]) |id| self.session.refuse(id, err) catch unreachable;
    // A point owns its OP: nothing appended later reads either.
    for (ids) |id| self.session.markFinal(id) catch unreachable;
    if (tangent) for (jobs, ids, 0..) |job, id, k| if (job == .tran and steps[k / j].len != 0) {
        try self.tangent_steps.put(self.arena.allocator(), id, steps[k / j]);
    };
    assert(ids.len == count * j);
    return ids[0];
}

/// `error.UnknownMeasure` for a `point.abort_measure` no card names.
pub fn checkPoint(self: *const Problem, point: requests.PointOptions) error{UnknownMeasure}!void {
    if (point.abort_measure.len == 0) return;
    if (measureIndex(self.prepared.deck.measures, point.abort_measure) == null) return error.UnknownMeasure;
}

/// Gives each transient of `jobs` one tangent row per live parameter at its
/// nominal value, moved by `h` as `appendPoints` moves it, and its steps in
/// `steps`. `error.NoLiveParameters` without live parameters,
/// `error.TangentsUnavailable` for a transient that cannot integrate them.
/// Asserts that `steps` is as long as `jobs`.
pub fn tangentDeck(self: *Problem, jobs: []Query, h: f64, steps: [][]const f64) !void {
    assert(steps.len == jobs.len);
    assert(h > 0);
    const tuning = self.prepared.tuning orelse return error.NoLiveParameters;
    const values = try self.arena.allocator().alloc(f64, self.live.len);
    for (values, self.live) |*v, k| v.* = tuning.values[k];
    const current = self.session.template;
    defer self.session.template = current;
    for (jobs, steps) |*job, *s| if (job.* == .tran) {
        if (!optimizer.tangentable(self, job.*)) return error.TangentsUnavailable;
        job.tran.tangents = try tangentRows(self, tuning, values, h, s);
    };
}

/// Solves the template's nominal operating point into `Deck.warm_op`, once,
/// with the tolerances of the first transient in `jobs` (else the first
/// query's); nothing without a job that reads one. A nominal point that
/// does not converge leaves it empty, so every point starts cold.
pub fn solveWarmOp(self: *Problem, jobs: []const Query) !void {
    if (self.prepared.deck.warm_op.len != 0) return;
    // Nothing reads it when every job starts from `.ic` (`uic`): on
    // chain_bsim4_100k that cold OP alone ran for hours before a `uic`
    // transient that needs none.
    for (jobs) |job| {
        if (!job.uic()) break;
    } else return;
    assert(jobs.len != 0);
    const tran_op, var tol = for (jobs) |job| {
        if (job == .tran) break .{ true, job.tran.tol };
    } else switch (jobs[0]) {
        inline else => |o| .{ false, o.tol },
    };
    tol.warm_op = false;
    tol.variant = null;
    tol.temp_c = null;
    var deck = self.prepared.deck;
    var session = Session.init(self.query_memory.allocator(), self.io, &self.prepared.circuit, &deck, self.session.config);
    defer session.deinit();
    var id: [1]QueryId = undefined;
    _ = try session.append(&.{.{ .op = .{ .tol = tol, .tran_op = tran_op } }}, &id);
    try session.runAll(.{ .quantum = .completion });
    const x = try session.operatingPoint(id[0]) orelse return;
    self.prepared.deck.warm_op = try self.arena.allocator().dupe(f64, x);
    assert(self.prepared.deck.warm_op.len == x.len);
}

/// Writes the `.meas` cards `names` of `reasons.len` points into `out`
/// (`Problem.point_measures` documents the contract). Asserts that
/// `out.len == reasons.len * names.len`.
pub fn pointMeasures(self: *const Problem, first: QueryId, names: []const []const u8, out: []f64, reasons: []Reason) !void {
    assert(out.len == reasons.len * names.len);
    const session = &self.session;
    const measures = self.prepared.deck.measures;
    var arena_state = std.heap.ArenaAllocator.init(self.gpa);
    defer arena_state.deinit();
    const scratch = arena_state.allocator();
    const cards = try cardIndices(scratch, measures, names);
    const outputs = session.outputs.items;
    const ends = try pointEnds(self, scratch, first, reasons.len);
    const values = try scratch.alloc(f64, measures.len);
    const missing = try scratch.alloc(bool, measures.len);
    for (reasons, 0..) |*reason, k| {
        const row = out[k * names.len ..][0..names.len];
        @memset(row, std.math.nan(f64));
        const ids = outputs[ends[k]..ends[k + 1]];
        reason.* = for (ids) |id| {
            if (try queryFailure(session, id)) |err| break .of(err);
        } else .ok;
        if (reason.* != .ok) continue;
        var evaluated: ?QueryId = null;
        for (cards, row) |card, *dst| {
            const kind = measures[card].analysis;
            const source = try measureSource(session, ids, kind);
            var why: Reason = .measure_not_found;
            if (source) |q| {
                // Cards of one analysis share its evaluation.
                if (evaluated != q) output.measureOutcomes(measures, kind, try block(session, q, 0), values, missing);
                evaluated = q;
                dst.* = values[card];
                if (!missing[card]) why = .measure_unresolved;
            }
            if (std.math.isNan(dst.*) and reason.* == .ok) reason.* = why;
        }
    }
}

/// Frees `count` points from `first` (`Problem.release_points` documents
/// the contract). Errors as `pointMeasures`.
pub fn releasePoints(self: *Problem, first: QueryId, count: usize) !void {
    var arena_state = std.heap.ArenaAllocator.init(self.gpa);
    defer arena_state.deinit();
    const ends = try pointEnds(self, arena_state.allocator(), first, count);
    assert(ends.len == count + 1);
    const outputs = self.session.outputs.items;
    // Dependents before the OPs they read.
    var k = ends[count];
    while (k > ends[0]) {
        k -= 1;
        try self.session.release(outputs[k]);
        _ = self.tangent_steps.remove(outputs[k]);
    }
    assert(k == ends[0]);
}

/// The slopes d(card)/d(live parameter) of the `.tran` cards `names` at
/// `count` points from `first` (`Problem.point_tangents` documents the
/// contract). Asserts that `out.len == count * names.len * live params`.
pub fn pointTangents(self: *const Problem, first: QueryId, count: usize, names: []const []const u8, out: []f64) !void {
    const p = self.live.len;
    assert(out.len == count * names.len * p);
    const measures = self.prepared.deck.measures;
    var arena_state = std.heap.ArenaAllocator.init(self.gpa);
    defer arena_state.deinit();
    const scratch = arena_state.allocator();
    // Validates the names and points and gives each point's reason.
    const base = try scratch.alloc(f64, count * names.len);
    const reasons = try scratch.alloc(Reason, count);
    try pointMeasures(self, first, names, base, reasons);
    // `pointMeasures` checked every name.
    const cards = cardIndices(scratch, measures, names) catch |err| switch (err) {
        error.OutOfMemory => |e| return e,
        error.UnknownMeasure => unreachable,
    };
    const values = try scratch.alloc(f64, measures.len);
    const v0 = try scratch.alloc(f64, measures.len);
    @memset(out, std.math.nan(f64));
    const session = &self.session;
    const outputs = session.outputs.items;
    // `pointMeasures` found `first` among the outputs.
    var at = (try session.outputIndex(first)).?;
    for (0..count) |i| {
        // Point i's queries share one variant row; its transient carries the steps.
        const row = variantOf(try session.query(outputs[at]));
        var tran: ?QueryId = null;
        while (at < outputs.len and variantOf(try session.query(outputs[at])) == row) : (at += 1) {
            if (tran == null and self.tangent_steps.contains(outputs[at])) tran = outputs[at];
        }
        const q = tran orelse continue;
        if ((try session.info(q)).status != .complete) continue;
        const steps = self.tangent_steps.get(q).?;
        assert(steps.len == p);
        output.measureValues(measures, .tran, try block(session, q, 0), v0);
        for (steps, 0..) |h, k| {
            output.measureValues(measures, .tran, try block(session, q, k + 1), values);
            for (cards, 0..) |card, c| out[(i * names.len + c) * p + k] = (values[card] - v0[card]) / h;
        }
    }
}

/// The variant row `job` runs at, or null for the nominal circuit.
pub fn variantOf(job: Query) ?u32 {
    return switch (job) {
        inline else => |o| o.tol.variant,
    };
}

/// Block `k` of query `id`'s result: its own run at 0, else the prediction
/// at its `k`-th tangent row (`Result.tangents`). Asserts that the result
/// holds that row.
pub fn block(session: *const Session, id: QueryId, k: usize) !output.Result {
    var res = try session.result(id);
    if (k == 0) return res;
    const n = res.npoints * res.varnames.len;
    assert(k * n <= res.tangents.len);
    res.data = res.tangents[(k - 1) * n ..][0..n];
    res.tangents = &.{};
    res.measured = &.{};
    assert(res.data.len == n);
    return res;
}

// ---- private ----

/// One variant row per live parameter of the point at `values`, that
/// parameter moved by `h`·|value| (`h` at 0), and the steps into `steps.*`;
/// a row `set_params` refuses gets a NaN step. Both live in the Problem's
/// arena: the queries hold the rows.
fn tangentRows(self: *Problem, tuning: Tuning, values: []const f64, h: f64, steps: *[]const f64) ![]const u32 {
    assert(values.len == self.live.len);
    const arena = self.arena.allocator();
    const deck = &self.prepared.deck;
    const rows = try arena.alloc(u32, values.len);
    const dp = try arena.alloc(f64, values.len);
    const moved = try arena.dupe(f64, values);
    defer arena.free(moved);
    for (rows, dp, 0..) |*row, *d, k| {
        d.* = if (values[k] == 0) h else h * @abs(values[k]);
        moved[k] = values[k] + d.*;
        defer moved[k] = values[k];
        try self.session.templates.ensureUnusedCapacity(self.session.allocator, 1);
        if (tuning.setAt(&self.prepared.circuit, deck.variants, self.live, moved)) |change| {
            // The capacity was reserved above.
            if (change.built) |c| self.session.addTemplate(c) catch unreachable;
            deck.variants = change.variants;
        } else |err| switch (err) {
            error.UnsupportedCard, error.ModelBinNotFound, error.InvalidArgument => {
                deck.variants = try tuning.refusedRow(deck.variants);
                d.* = std.math.nan(f64);
            },
            else => |e| return e,
        }
        row.* = deck.variants.count() - 1;
    }
    steps.* = dp;
    assert(steps.len == rows.len);
    return rows;
}

/// The index of each of `names` among `measures`, in `scratch`.
fn cardIndices(scratch: Allocator, measures: []const core.Measure, names: []const []const u8) error{ OutOfMemory, UnknownMeasure }![]u32 {
    const cards = try scratch.alloc(u32, names.len);
    for (names, cards) |name, *card| card.* = measureIndex(measures, name) orelse return error.UnknownMeasure;
    return cards;
}

/// The index of the `.meas` card named `name` (case-insensitive), the first
/// when several share it.
fn measureIndex(measures: []const core.Measure, name: []const u8) ?u32 {
    for (measures, 0..) |m, k| if (std.ascii.eqlIgnoreCase(m.name, name)) return @intCast(k);
    return null;
}

/// Where each of `count` points from `first` lies in `session.outputs`:
/// point k's queries are `outputs[ends[k]..ends[k + 1]]`, one variant
/// row's. `error.InvalidQuery` for a `first` that is not requested or fewer
/// points after it, `error.ResultUnavailable` while one of them has not
/// ended.
fn pointEnds(self: *const Problem, scratch: Allocator, first: QueryId, count: usize) ![]usize {
    const session = &self.session;
    const outputs = session.outputs.items;
    const ends = try scratch.alloc(usize, count + 1);
    ends[0] = try session.outputIndex(first) orelse return error.InvalidQuery;
    for (ends[1..], ends[0..count]) |*end, begin| {
        if (begin == outputs.len) return error.InvalidQuery;
        const row = variantOf(try session.query(outputs[begin]));
        var e = begin;
        while (e < outputs.len and variantOf(try session.query(outputs[e])) == row) : (e += 1) {
            if (!(try session.info(outputs[e])).status.terminal()) return error.ResultUnavailable;
        }
        end.* = e;
    }
    assert(ends[count] <= outputs.len);
    return ends;
}

/// The error query `id` ended with, its prerequisite's for a query that
/// never ran because of it; null when it completed. Asserts it is terminal.
fn queryFailure(session: *const Session, id: QueryId) !?anyerror {
    const info = try session.info(id);
    assert(info.status.terminal());
    if (info.status == .complete) return null;
    // `?anyerror` payloads: a bare error value here would return as this
    // function's own error.
    if (info.failure) |err| if (err == error.DependencyFailed) if (info.dependency) |dep| {
        if ((try session.info(dep)).failure) |cause| return @as(?anyerror, cause);
    };
    return @as(?anyerror, info.failure orelse error.QueryCancelled);
}

/// The query of `ids` a card of `kind` reads: the point's first analysis of
/// that kind, else for an `op` card the point's own operating point.
fn measureSource(session: *const Session, ids: []const QueryId, kind: requests.Kind) !?QueryId {
    for (ids) |id| if ((try session.info(id)).kind == kind) return id;
    if (kind != .op) return null;
    for (ids) |id| if ((try session.info(id)).dependency) |dep| return dep;
    return null;
}
