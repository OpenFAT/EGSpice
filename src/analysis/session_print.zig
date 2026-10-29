//! A session's query graph as a tree on a writer: every root OP and its
//! dependents, with each row's status and what the previewed advancement
//! would run.
//! `Session` exposes each function here as a method.
//! In: the rows. Out: text.
//! Cites no outside source.
//!
//! Contents:
//! - `print`
//! - `containsComponent`
//! - `printQuery`

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

/// Writes the query tree for `options.scope`, marking what the chosen
/// preview would run next. Runs nothing; uses the same readiness order and
/// validation as advancement.
pub fn print(self: *const Session, writer: *std.Io.Writer, options: PrintOptions) !void {
    try self.validateScope(options.scope);
    const limits = options.limits orelse Limits{};
    if (limits.max_parallel == 0) return error.InvalidConcurrency;
    const a = self.allocator;
    const next = try a.alloc(QueryId, self.rows.len);
    defer a.free(next);
    var n_next: usize = 0;
    switch (options.preview) {
        .run_all => n_next = @min(try self.readyQueries(options.scope, next), limits.max_parallel),
        .advance => |id| n_next = @min(try self.readyQueries(.{ .query = id }, next), 1),
        .advance_ready => |ids| {
            try self.validateReady(ids, limits);
            @memcpy(next[0..ids.len], ids);
            n_next = ids.len;
        },
    }
    const seen = try a.alloc(bool, self.rows.len);
    defer a.free(seen);
    @memset(seen, false);
    const continuations = try a.alloc(bool, self.rows.len + 1);
    defer a.free(continuations);
    var components: usize = 0;
    for (self.rows.items(.component), 0..) |component, i|
        if (component == i and self.containsComponent(options.scope, component)) {
            components += 1;
        };
    try writer.print("Problem: {d} requested queries, {d} total queries, {d} components\n", .{ self.outputs.items.len, self.rows.len, components });
    // Each component's rows in row order, CSR by component (a
    // component is numbered by its first row), so the tree is one pass.
    const starts = try a.alloc(u32, self.rows.len + 1);
    defer a.free(starts);
    @memset(starts, 0);
    for (self.rows.items(.component)) |c| starts[c + 1] += 1;
    for (1..starts.len) |c| starts[c] += starts[c - 1];
    const members = try a.alloc(u32, self.rows.len);
    defer a.free(members);
    const fill = try a.dupe(u32, starts[0..self.rows.len]);
    defer a.free(fill);
    for (self.rows.items(.component), 0..) |c, j| {
        members[fill[c]] = @intCast(j);
        fill[c] += 1;
    }
    for (self.rows.items(.component), 0..) |component, i| {
        if (component != i or !self.containsComponent(options.scope, component)) continue;
        components -= 1;
        continuations[0] = components != 0;
        try writer.print("{s} Component {d}\n", .{ connector(options.ascii, components == 0), component });
        const rows = members[starts[component]..starts[component + 1]];
        var last_query: usize = i;
        for (rows) |j| {
            if (self.contains(options.scope, j) and self.rows.items(.requested)[j]) last_query = j;
        }
        for (rows) |j| {
            if (!self.contains(options.scope, j) or !self.rows.items(.requested)[j]) continue;
            try self.printQuery(writer, j, 1, j == last_query, continuations, next[0..n_next], seen, options.ascii);
        }
        // A scope naming an implicit prerequisite still shows that query.
        if (options.scope == .query) {
            const j = @backingInt(options.scope.query);
            if (!seen[j]) try self.printQuery(writer, j, 1, j == last_query, continuations, next[0..n_next], seen, options.ascii);
        }
    }
}

pub fn containsComponent(self: *const Session, scope: Scope, component: u32) bool {
    return switch (scope) {
        .all => true,
        .component => |c| component == c,
        .query => |q| component == self.rows.items(.component)[@backingInt(q)],
    };
}

pub fn printQuery(self: *const Session, writer: *std.Io.Writer, i: usize, depth: usize, last: bool, continuations: []bool, next: []const QueryId, seen: []bool, ascii: bool) !void {
    const q = try self.info(@fromBackingInt(@intCast(i)));
    for (continuations[0..depth]) |continues|
        try writer.writeAll(if (!continues) "    " else if (ascii) "|   " else "│   ");
    try writer.print("{s} {s}q{d} {s} [{s}", .{
        connector(ascii, last),
        if (seen[i]) (if (ascii) "-> " else "↪ ") else "",
        i,
        @tagName(q.kind),
        if (q.status == .pending and self.ready(i)) "ready" else @tagName(q.status),
    });
    if (seen[i]) try writer.writeAll("; shared");
    const selected = std.mem.indexOfScalar(QueryId, next, @fromBackingInt(@intCast(i))) != null;
    if (selected) try writer.writeAll("; NEXT") else if (self.ready(i)) try writer.writeAll("; deferred by selection");
    if (q.dependency) |dep| if (!self.ready(i) and !q.status.terminal())
        try writer.print("; waiting for q{d}", .{@backingInt(dep)});
    if (q.progress) |p| try writer.print("; {s} {d}/{d}", .{ @tagName(p.phase), p.completed, p.total });
    if (q.failure) |err| try writer.print("; {s}", .{@errorName(err)});
    try writer.writeAll("]\n");
    if (seen[i]) return;
    seen[i] = true;
    continuations[depth] = !last;
    if (q.dependency) |dep| try self.printQuery(writer, @backingInt(dep), depth + 1, true, continuations, next, seen, ascii);
}

// ---- private ----

fn connector(ascii: bool, last: bool) []const u8 {
    return if (ascii) (if (last) "`--" else "|--") else (if (last) "└──" else "├──");
}
