//! A session's job table helpers: the comparisons and hashes that intern a
//! job by content, the lane-batch admission rule, the lane rows a query
//! splits into, a query's OP prerequisite, and deep copies of options.
//! In: queries. Out: keys, verdicts, copies.
//! Cites #213.
//!
//! Contents:
//! - `none`
//! - `no_output`
//! - `no_variant`
//! - `lane_window`
//! - `sameLaneJob`
//! - `batchable`
//! - `stripTol`
//! - `sameDeep`
//! - `OpKey`
//! - `OpIndex`
//! - `JobIndex`
//! - `JobKey`
//! - `withoutVariant`
//! - `hashFields`
//! - `max_split`
//! - `splitLanes`
//! - `prerequisite`
//! - `copyValue`
//! - members: `hash`, `eql`

const std = @import("std");
const requests = @import("core").query;
const execution = @import("executor.zig");
const temp_sweep = @import("sweep/temp_sweep.zig");
const QueryId = requests.QueryId;

/// The row id that names no row.
pub const none = requests.invalid_query;

/// `Row.output` of a row nobody requested.
pub const no_output = std.math.maxInt(u32);

/// `Row.variant` of a row without one.
pub const no_variant = std.math.maxInt(u32);

/// How far ahead of the queue head `gather` looks for a batch's peers.
pub const lane_window = 8 * execution.lane_width;

/// An OP row's template, `jobs` entry and variant: equal keys are equal
/// OPs.
pub const OpKey = struct { template: u32, job: u32, variant: u32 };

pub const OpIndex = std.AutoHashMapUnmanaged(OpKey, QueryId);

/// `Session.jobs` indices, hashed and compared by the entry they name.
pub const JobIndex = std.HashMapUnmanaged(u32, void, struct {
    jobs: []const requests.Query,
    pub fn hash(c: @This(), k: u32) u64 {
        return JobKey.hash(.{ .jobs = c.jobs }, c.jobs[k]);
    }
    pub fn eql(_: @This(), a: u32, b: u32) bool {
        return a == b;
    }
}, std.hash_map.default_max_load_percentage);

/// `JobIndex` looked up by a query.
pub const JobKey = struct {
    jobs: []const requests.Query,
    pub fn hash(_: JobKey, job: requests.Query) u64 {
        var h: std.hash.Wyhash = .init(0);
        hashFields(&h, job);
        return h.final();
    }
    pub fn eql(c: JobKey, job: requests.Query, k: u32) bool {
        return sameDeep(job, c.jobs[k]);
    }
};

/// Largest query `append` splits into lane rows; a longer one runs whole,
/// which publishes the same rows.
/// ponytail: one row per lane; batch lanes into rows of several if huge
/// sweeps make the row table the memory bottleneck.
pub const max_split = 1 << 16;

/// Whether two rows may share a lane batch: the same kind with equal
/// options but for the variant row and temperature, which only the circuit
/// each lane builds reads. Slices compare by content, floats by bits.
pub fn sameLaneJob(a: requests.Query, b: requests.Query) bool {
    return switch (a) {
        // HB's GPU provider owns its Newton rounds; it is deliberately not
        // CPU-batchable, so ordinary CPU query workers remain independent.
        .hb => |o| b == .hb and sameDeep(stripTol(o), stripTol(b.hb)),
        // The AC family's list is `execution.freqLaneKind`'s; keep the two adjacent in review.
        inline .op, .dc, .mc, .temp, .tran, .pss, .ac, .noise, .sp, .stb, .lstb, .acxf, .tf, .dcxf, .dcinc, .sens, .dcmatch, .dcsens, .acmatch => |o, tag| b == tag and batchable(b) and sameDeep(stripTol(o), stripTol(@field(b, @tagName(tag)))),
        else => false,
    };
}

/// Kinds a lane batch runs: `.op`, `.dc`, `.tran`, `.pss`, the AC family
/// (`execution.freqLaneKind`), and the lane rows a split `.mc` or `.temp`
/// became (`splitLanes`).
pub fn batchable(job: requests.Query) bool {
    return switch (job) {
        .op, .dc, .tran, .pss => true,
        inline .mc, .temp => |o| o.lane != null,
        else => execution.freqLaneKind(job),
    };
}

/// `o` without what only the circuit each lane builds reads: the variant
/// row, the temperature and a split row's lane.
pub fn stripTol(o: anytype) @TypeOf(o) {
    var c = o;
    c.tol.variant = null;
    c.tol.temp_c = null;
    if (@hasField(@TypeOf(o), "lane")) c.lane = null;
    return c;
}

pub fn sameDeep(a: anytype, b: @TypeOf(a)) bool {
    const T = @TypeOf(a);
    switch (@typeInfo(T)) {
        .@"struct" => |s| {
            inline for (s.field_names) |name| if (!sameDeep(@field(a, name), @field(b, name))) return false;
            return true;
        },
        .@"union" => {
            if (std.meta.activeTag(a) != std.meta.activeTag(b)) return false;
            return switch (a) {
                inline else => |payload, tag| sameDeep(payload, @field(b, @tagName(tag))),
            };
        },
        .optional => {
            if (a == null or b == null) return a == null and b == null;
            return sameDeep(a.?, b.?);
        },
        .pointer => |p| {
            if (p.size != .slice) return a == b;
            if (a.len != b.len) return false;
            for (a, b) |x, y| if (!sameDeep(x, y)) return false;
            return true;
        },
        .float => return std.mem.eql(u8, std.mem.asBytes(&a), std.mem.asBytes(&b)),
        .array => {
            for (a, b) |x, y| if (!sameDeep(x, y)) return false;
            return true;
        },
        else => return a == b,
    }
}

/// `job` with `tol.variant` cleared, and the variant (`no_variant` for none).
pub fn withoutVariant(job: requests.Query) struct { requests.Query, u32 } {
    var j = job;
    switch (j) {
        inline else => |*o| {
            const v = o.tol.variant orelse no_variant;
            o.tol.variant = null;
            return .{ j, v };
        },
    }
}

/// Feeds `value` to `h` as `sameDeep` compares it: slices by content,
/// floats by their bits (which `std.hash.autoHash` refuses).
pub fn hashFields(h: *std.hash.Wyhash, value: anytype) void {
    switch (@typeInfo(@TypeOf(value))) {
        .@"struct" => |s| inline for (s.field_names) |name| hashFields(h, @field(value, name)),
        .@"union" => switch (value) {
            inline else => |payload, tag| {
                std.hash.autoHash(h, tag);
                hashFields(h, payload);
            },
        },
        .optional => if (value) |v| {
            h.update(&.{1});
            hashFields(h, v);
        } else h.update(&.{0}),
        .pointer => |p| if (p.size != .slice) std.hash.autoHash(h, value) else {
            std.hash.autoHash(h, value.len);
            if (p.child == u8) h.update(value) else for (value) |x| hashFields(h, x);
        },
        .array => for (value) |x| hashFields(h, x),
        .float => h.update(std.mem.asBytes(&value)),
        else => std.hash.autoHash(h, value),
    }
}

/// How many lane rows `append` splits `job` into (#213): one per `.mc`
/// trial or variant and per `.temp` temperature, each solved alone; 0 for
/// a job it keeps whole.
pub fn splitLanes(job: requests.Query) u32 {
    const n: usize = switch (job) {
        .mc => |o| if (o.lane != null) 0 else if (o.variants.count != 0) o.variants.count else o.n_trials,
        // Unvalidated: NaN and huge spans fail the comparison.
        .temp => |o| if (o.lane == null and o.t_step > 0 and (o.t_stop - o.t_start) / o.t_step < max_split) temp_sweep.numPoints(o) else 0,
        else => 0,
    };
    return if (n >= 2 and n <= max_split) @intCast(n) else 0;
}

/// The OP a query needs first: its tolerances, transient-flavoured for
/// time-domain kinds, warm when a transient point asks. Null for `.op`
/// itself and a `uic` start.
pub fn prerequisite(job: requests.Query) ?requests.Op {
    if (job == .op or job.uic()) return null;
    const tol = switch (job) {
        inline else => |o| o.tol,
    };
    return .{ .tol = tol, .tran_op = @as(requests.Kind, job).transient(), .warm_start = tol.warm_op or (job == .tran and job.tran.point.warm_start) };
}

/// Deep-copies every slice reachable from `value` into `allocator`.
pub fn copyValue(allocator: std.mem.Allocator, value: anytype) std.mem.Allocator.Error!@TypeOf(value) {
    const T = @TypeOf(value);
    switch (@typeInfo(T)) {
        .@"struct" => |s| {
            var result = value;
            inline for (s.field_names) |name| @field(result, name) = try copyValue(allocator, @field(value, name));
            return result;
        },
        .@"union" => switch (value) {
            inline else => |payload, tag| return @unionInit(T, @tagName(tag), try copyValue(allocator, payload)),
        },
        .optional => return if (value) |payload| try copyValue(allocator, payload) else null,
        .pointer => |p| {
            if (p.size != .slice) return value;
            const result = try allocator.alloc(p.child, value.len);
            for (value, result) |src, *dst| dst.* = try copyValue(allocator, src);
            return result;
        },
        else => return value,
    }
}

test copyValue {
    // The copy shares no slice with the request, so a caller may free or
    // reuse its option buffers once `append` returns.
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var tones = [_]f64{ 2e3, 3e3 };
    var harms = [_]u16{ 2, 3 };
    const job: requests.Query = .{ .hb = .{ .f0 = 1e3, .extra_tones = &tones, .extra_harmonics = &harms } };
    const copy = try copyValue(arena.allocator(), job);
    tones[0] = 0;
    harms[1] = 0;
    try std.testing.expectEqual(@as(f64, 1e3), copy.hb.f0);
    try std.testing.expectEqualSlices(f64, &.{ 2e3, 3e3 }, copy.hb.extra_tones);
    try std.testing.expectEqualSlices(u16, &.{ 2, 3 }, copy.hb.extra_harmonics);
    // An empty slice copies to an empty slice.
    const plain = try copyValue(arena.allocator(), requests.Query{ .hb = .{ .f0 = 1e3 } });
    try std.testing.expectEqual(@as(usize, 0), plain.hb.extra_tones.len);
}

test prerequisite {
    try std.testing.expectEqual(null, prerequisite(.{ .op = .{} }));
    try std.testing.expectEqual(null, prerequisite(.{ .tran = .{ .t_stop = 1e-6, .uic = true } }));
    // A transient-family query gets a MODETRANOP point with its own tolerances.
    const tran = prerequisite(.{ .tran = .{ .t_stop = 1e-6, .tol = .{ .reltol = 1e-4 } } }).?;
    try std.testing.expect(tran.tran_op);
    try std.testing.expectEqual(@as(f64, 1e-4), tran.tol.reltol);
    try std.testing.expect(!tran.warm_start);
    try std.testing.expect(prerequisite(.{ .tran = .{ .t_stop = 1e-6, .point = .{ .warm_start = true } } }).?.warm_start);
    const ac = prerequisite(.{ .ac = .{ .sweep = .{ .f_start = 1, .f_stop = 1e3 } } }).?;
    try std.testing.expect(!ac.tran_op);
    try std.testing.expect(!ac.warm_start);
}
