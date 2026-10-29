//! Building a session's query graph: appending queries (validated,
//! deep-copied and interned), the OP prerequisite each shares, the lane rows
//! a split query becomes, and the row marks a caller sets afterwards.
//! `Session` exposes each function here as a method.
//! In: query requests. Out: rows and their ids.
//! Cites #13, #55, #213.
//!
//! Contents:
//! - `append`
//! - `appendAt`
//! - `JobRef`
//! - `intern`
//! - `jobAt`
//! - `kindAt`
//! - `opKey`
//! - `addRow`
//! - `markFinal`
//! - `final`
//! - `refuse`
//! - `addTemplate`

const std = @import("std");
const assert = std.debug.assert;
const core = @import("core");
const requests = core.query;
const Circuit = @import("device").Circuit;
const execution = @import("executor.zig");
const progress = @import("worker.zig");
const types = @import("types.zig");
const session = @import("session.zig");
const jobs_mod = @import("session_jobs.zig");
const Session = session.Session;
const Status = session.Status;
const Scope = session.Scope;
const Limits = session.Limits;
const Advance = session.Advance;
const PrintOptions = session.PrintOptions;
const Failure = session.Failure;
const PointError = session.PointError;
const Pool = session.Pool;
const QueryId = session.QueryId;
const Quantum = progress.Quantum;
const Result = types.Result;
const none = jobs_mod.none;
const no_output = jobs_mod.no_output;
const no_variant = jobs_mod.no_variant;
const lane_window = jobs_mod.lane_window;
const sameLaneJob = jobs_mod.sameLaneJob;
const batchable = jobs_mod.batchable;
const splitLanes = jobs_mod.splitLanes;
const prerequisite = jobs_mod.prerequisite;
const withoutVariant = jobs_mod.withoutVariant;
const copyValue = jobs_mod.copyValue;
const JobKey = jobs_mod.JobKey;
const OpKey = jobs_mod.OpKey;
const validateDeck = core.validate.validateDeck;

/// One appended job: its `jobs` entry, its variant, and its
/// prerequisite OP's entry (`no_variant` without one).
pub const JobRef = struct { job: u32, variant: u32, op: u32 };

/// Adds `jobs`, writing their ids to `ids`, and returns `jobs.len`. Each
/// non-OP job shares an existing matching OP prerequisite or gets a new
/// one. All-or-nothing: every job is validated and deep-copied before any
/// row or id is published. Fails with `error.BufferTooSmall` when `ids`
/// is shorter than `jobs`, a validation error for the first bad job, or
/// `error.TooManyQueries` past u32 rows.
pub fn append(self: *Session, jobs: []const requests.Query, ids: []QueryId) !usize {
    return self.appendAt(jobs, null, ids);
}

/// `append` with job k instantiating `templates[k]` (as `template`
/// numbers them) instead of `template`, so one call can hold points on
/// several templates. Asserts `templates` matches `jobs` in length.
pub fn appendAt(self: *Session, jobs: []const requests.Query, templates: ?[]const u32, ids: []QueryId) !usize {
    if (templates) |t| assert(t.len == jobs.len);
    if (ids.len < jobs.len) return error.BufferTooSmall;
    if (jobs.len == 0) return 0;
    // Each job adds at least its own row, at most also an OP and its
    // lane rows; the first bound needs only the length (#55).
    if (jobs.len >= std.math.maxInt(u32) - self.rows.len) return error.TooManyQueries;
    var added: usize = 0;
    for (jobs) |job| added = std.math.add(usize, added, 2 + @as(usize, splitLanes(job))) catch return error.TooManyQueries;
    const max_rows = std.math.add(usize, self.rows.len, added) catch return error.TooManyQueries;
    if (max_rows >= std.math.maxInt(u32)) return error.TooManyQueries;
    const old_len = self.rows.len;
    for (jobs) |job| try validateDeck(job, self.topology.n, self.deck);
    // Everything that can fail first: interning, then room for every
    // row, OP index entry and output, so the rows below change
    // infallibly. A failed append leaves at most unread `jobs` entries.
    const refs = try self.allocator.alloc(JobRef, jobs.len);
    defer self.allocator.free(refs);
    for (jobs, refs) |job, *r| {
        const stripped, const variant = withoutVariant(job);
        r.* = .{
            .job = try self.intern(stripped),
            .variant = variant,
            .op = if (prerequisite(stripped)) |op| try self.intern(.{ .op = op }) else no_variant,
        };
    }
    try self.rows.ensureTotalCapacity(self.allocator, max_rows);
    try self.ops.ensureUnusedCapacity(self.allocator, @intCast(max_rows - old_len));
    try self.outputs.ensureUnusedCapacity(self.allocator, jobs.len);
    try self.urgent.ensureTotalCapacity(self.allocator, max_rows);
    var roots: u32 = 0;
    for (refs, ids[0..jobs.len], 0..) |r, *id, k| {
        const template = if (templates) |t| t[k] else self.template;
        assert(template <= self.templates.items.len);
        const job = self.jobs.items[r.job];
        // An explicit OP can expose an already shared prerequisite product.
        if (job == .op) {
            if (self.ops.get(.{ .template = template, .job = r.job, .variant = r.variant })) |existing| {
                const e = @backingInt(existing);
                id.* = existing;
                if (!self.rows.items(.requested)[e]) {
                    self.rows.items(.requested)[e] = true;
                    self.rows.items(.output)[e] = @intCast(self.outputs.items.len);
                    self.outputs.appendAssumeCapacity(existing);
                }
                continue;
            }
        }
        const n_lanes = splitLanes(job);
        for (0..n_lanes) |_| _ = self.addRow(r, true, template, false, &roots);
        id.* = self.addRow(r, false, template, true, &roots);
        if (n_lanes != 0) {
            const g = @backingInt(id.*);
            self.rows.items(.lanes)[g] = n_lanes;
            for (g - n_lanes..g) |l| {
                assert(self.rows.items(.dependency)[l] == self.rows.items(.dependency)[g]);
                self.rows.items(.owner)[l] = id.*;
            }
        }
    }
    self.root_readers += roots;
    return jobs.len;
}

/// `job`'s index in `jobs`, appending a deep copy when no equal entry
/// exists (slices by content, floats by bits).
pub fn intern(self: *Session, job: requests.Query) !u32 {
    const jobs = self.jobs.items;
    const got = try self.job_index.getOrPutContextAdapted(self.allocator, job, JobKey{ .jobs = jobs }, .{ .jobs = jobs });
    if (got.found_existing) return got.key_ptr.*;
    errdefer self.job_index.removeByPtr(got.key_ptr);
    try self.jobs.ensureUnusedCapacity(self.allocator, 1);
    const k: u32 = @intCast(self.jobs.items.len);
    self.jobs.appendAssumeCapacity(try copyValue(self.job_arena.allocator(), job));
    got.key_ptr.* = k;
    return k;
}

/// Row `i`'s query as it was appended: its variant set and, on a lane
/// row, its lane.
pub fn jobAt(self: *const Session, i: usize) requests.Query {
    var job = self.jobs.items[self.rows.items(.job)[i]];
    const v = self.rows.items(.variant)[i];
    switch (job) {
        inline else => |*o| o.tol.variant = if (v == no_variant) null else v,
    }
    const owner = self.rows.items(.owner)[i];
    if (owner != none) {
        const g = @backingInt(owner);
        const lane: u32 = @intCast(i - (g - self.rows.items(.lanes)[g]));
        switch (job) {
            inline .mc, .temp => |*o| o.lane = lane,
            else => unreachable, // else: only `.mc` and `.temp` split (`splitLanes`)
        }
    }
    return job;
}

pub fn kindAt(self: *const Session, i: usize) requests.Kind {
    return self.jobs.items[self.rows.items(.job)[i]];
}

pub fn opKey(self: *const Session, i: usize) OpKey {
    return .{ .template = self.rows.items(.template)[i], .job = self.rows.items(.job)[i], .variant = self.rows.items(.variant)[i] };
}

/// Appends job `r` after the OP it needs, shared when one matches, and
/// returns its id; a `requested` row joins `outputs`. A split query
/// (`splitLanes`) reads its OP only through its `lane` rows, so it is
/// not one of the OP's readers. Infallible: `appendAt` reserved every
/// row, OP index entry and output first.
pub fn addRow(self: *Session, r: JobRef, lane: bool, template: u32, requested: bool, roots: *u32) QueryId {
    const job = self.jobs.items[r.job];
    const split = if (lane) 0 else splitLanes(job);
    var dependency = none;
    if (r.op != no_variant) {
        const key: OpKey = .{ .template = template, .job = r.op, .variant = r.variant };
        dependency = self.ops.get(key) orelse blk: {
            const dep: QueryId = @fromBackingInt(@intCast(self.rows.len));
            self.rows.appendAssumeCapacity(.{
                .job = r.op,
                .variant = r.variant,
                .component = @backingInt(dep),
                .requested = false,
                .template = template,
            });
            self.ops.putAssumeCapacity(key, dep);
            roots.* += 1;
            break :blk dep;
        };
    }
    const id: QueryId = @fromBackingInt(@intCast(self.rows.len));
    var sibling = none;
    if (dependency == none) {
        assert(split == 0);
        roots.* += 1;
    } else if (split == 0) {
        const d = @backingInt(dependency);
        self.rows.items(.readers)[d] += 1;
        sibling = self.rows.items(.child)[d];
        self.rows.items(.child)[d] = id;
    }
    self.rows.appendAssumeCapacity(.{
        .job = r.job,
        .variant = r.variant,
        .dependency = dependency,
        .component = if (dependency == none) @backingInt(id) else self.rows.items(.component)[@backingInt(dependency)],
        .requested = requested,
        .template = template,
        .sibling = sibling,
        .output = if (requested) @intCast(self.outputs.items.len) else no_output,
    });
    if (requested) self.outputs.appendAssumeCapacity(id);
    if (job == .op) self.ops.putAssumeCapacity(.{ .template = template, .job = r.job, .variant = r.variant }, id);
    return id;
}

/// Marks query `id` and its prerequisite final: the caller appends
/// nothing that reads their state (a point of `Problem.append_points`
/// owns its OP), so a dependent may take the OP's state instead of
/// copying it and each circuit is freed once nothing reads it, as under
/// `Config.final_plan`. Results stay.
pub fn markFinal(self: *Session, id: QueryId) !void {
    const i = try self.index(id);
    self.rows.items(.final)[i] = true;
    const dep = self.rows.items(.dependency)[i];
    if (dep != none) self.rows.items(.final)[@backingInt(dep)] = true;
}

pub fn final(self: *const Session, i: usize) bool {
    return self.config.final_plan or self.rows.items(.final)[i];
}

/// Ends pending query `id`, and its pending prerequisite, with `err`
/// before anything runs: a point `Problem.append_points` could not set.
/// The prerequisite must be the query's own (no other row reads it).
pub fn refuse(self: *Session, id: QueryId, err: PointError) !void {
    const i = try self.index(id);
    const dep = self.rows.items(.dependency)[i];
    for ([_]QueryId{ id, dep }) |row| if (row != none) {
        const r = @backingInt(row);
        assert(self.rows.items(.executor)[r] == null);
        self.rows.items(.status)[r] = .failed;
        self.rows.items(.failure)[r] = err;
    };
}

/// Makes `circuit` the template of every row appended from now on
/// (`template`, which a caller may point back at an earlier one).
/// Asserts that it has `topology`'s node count and pattern size; the
/// caller keeps it alive for the session's lifetime.
pub fn addTemplate(self: *Session, circuit: *const Circuit) !void {
    assert(circuit.n == self.topology.n);
    assert(circuit.nnz == self.topology.nnz);
    try self.templates.append(self.allocator, circuit);
    self.template = @intCast(self.templates.items.len);
}
