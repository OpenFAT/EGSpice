//! GPU frequency replay over (point, frequency) lanes. In: every point's
//! `FreqSolver`, its frequencies and right-hand sides. Out: each point's
//! solutions, written exactly where its host `solveBatch` would write them.
//! The host owns each point's solver and advances exactly its usual
//! W-frequency chunks; only equal pivot tapes share a launch
//! (`freq_engine.zig`). Failed lanes peel in point-local frequency order
//! before the next round, so readout stays unchanged. The sweep precompute
//! (`solveAll`) replays the host's complex tapes while a point's complex
//! lanes are on, its stacked tapes after; live rounds and held
//! preconditioner factors (`freq_held.zig`) use stacked tapes.
//! Cites: none (FreqSolver's lane-chunk protocol).
//!
//! Contents:
//! - `pub const test_hooks` (`factor_failure_after`, `factor_candidates`)
//! - `pub const Point = struct`
//! - `pub const RoundPoint = struct`
//! - `pub const SolveError`
//! - `pub const Stats = struct { points, lanes, groups, peels }`
//! - `pub const RoundDriver = struct` (`init`, `deinit`, `solve`)
//! - `pub const Held = freq_held.Held`
//! - `pub const prepareHeld = freq_held.prepareHeld`
//! - `pub const DeviceHeld = freq_held.DeviceHeld`
//! - `pub fn solveAll(gpa: std.mem.Allocator, requested: gpu.Request, points: []const Point) !Stats`
//! - `pub fn solveRound(gpa: std.mem.Allocator, requested: gpu.Request, points: []const RoundPoint, failures: []?SolveError) std.mem.Allocator.Error!Stats`

const std = @import("std");
const assert = std.debug.assert;
const gompute = @import("gompute");
const artifacts = @import("gompute_kernels");
const solver = @import("solver");
const root = @import("../types.zig");
const gpu = @import("context.zig");
const freq = @import("../ac/freq.zig");
const freq_engine = @import("freq_engine.zig");
const freq_held = @import("freq_held.zig");

const FS = solver.freq_solve.FreqSolver;
const Chunk = FS.LaneChunk;
const CChunk = FS.ComplexChunk;
const Engine = freq_engine.Engine;

pub const test_hooks = if (@import("builtin").is_test) struct {
    /// Fail one replay only after its host pivot candidate was selected.
    /// Other points may already have completed resident factors.
    pub var factor_failure_after: ?usize = null;
    pub var factor_candidates: usize = 0;
} else struct {};

/// Borrowed precompute inputs. `out` has omegas.len*rhs.len values and is
/// owned by the query, as for freq.solveAll. No buffers are retained.
pub const Point = struct {
    fs: *FS,
    circuit: *root.Circuit,
    x_op: []const f64,
    omegas: []const f64,
    rhs: []const f64,
    out: []f64,
    adjoint: bool,
};

/// One live point's next frequency chunk. The owning analysis may resume
/// and change its next request only after solveRound returns. The solver
/// carries its original pivot history across all rounds.
pub const RoundPoint = struct {
    fs: *FS,
    omegas: []const f64,
    dyn: solver.freq_solve.Dyn = .{},
    rhs: []const f64,
    out: []f64,
    adjoint: bool,
};

/// A point's own numerical failure, as its host `solveBatch` reports it.
pub const SolveError = root.ErrorOf(FS.solveBatch);

/// Points served, device lanes run, launch groups, and lanes that failed a
/// device refactor and peeled to the host.
pub const Stats = struct {
    points: usize = 0,
    lanes: usize = 0,
    groups: usize = 0,
    peels: usize = 0,
};

/// Keeps kernel modules and scratch buffers across live requests. A device
/// failure disables this driver for its remaining rounds; each point still
/// continues on its own original solver.
pub const RoundDriver = struct {
    allocator: std.mem.Allocator,
    requested: gpu.Request,
    cuda: if (artifacts.has_cuda) ?Engine(.cuda) else void = if (artifacts.has_cuda) null else {},
    hip: if (artifacts.has_hip) ?Engine(.hip) else void = if (artifacts.has_hip) null else {},
    disabled: bool = false,

    /// Touches no driver: the engine loads on the first round that needs it.
    pub fn init(allocator: std.mem.Allocator, requested: gpu.Request) RoundDriver {
        return .{ .allocator = allocator, .requested = requested };
    }

    /// Frees the engine, if one loaded.
    pub fn deinit(self: *RoundDriver) void {
        if (comptime artifacts.has_cuda) if (self.cuda) |*engine| engine.deinit();
        if (comptime artifacts.has_hip) if (self.hip) |*engine| engine.deinit();
        self.* = undefined;
    }

    /// `solveRound` on this driver's engine. Asserts that `failures` has
    /// one slot per point, that each point asks for 1 to `FS.W`
    /// frequencies, and that no two points share a solver.
    pub fn solve(self: *RoundDriver, points: []const RoundPoint, failures: []?SolveError) std.mem.Allocator.Error!Stats {
        std.debug.assert(points.len == failures.len);
        // Every point brings its own solver and one chunk of frequencies.
        for (points, 0..) |p, i| {
            std.debug.assert(p.omegas.len > 0 and p.omegas.len <= FS.W);
            for (points[0..i]) |q| std.debug.assert(q.fs != p.fs);
        }
        @memset(failures, null);
        if (points.len == 0) return .{};
        if (!self.disabled) switch (self.requested) {
            .cpu => {},
            .cuda => if (comptime artifacts.has_cuda) return runRound(.cuda, self.allocator, points, failures, &self.cuda, &self.disabled),
            .hip => if (comptime artifacts.has_hip) return runRound(.hip, self.allocator, points, failures, &self.hip, &self.disabled),
            .auto => {
                if (comptime artifacts.has_cuda) return runRound(.cuda, self.allocator, points, failures, &self.cuda, &self.disabled);
                if (comptime artifacts.has_hip) return runRound(.hip, self.allocator, points, failures, &self.hip, &self.disabled);
            },
        };
        for (points, failures) |p, *failure| solveHost(p, failure);
        return .{ .points = points.len };
    }
};

/// Retained harmonic factors on the device (`freq_held.zig`).
pub const Held = freq_held.Held;
/// Refreshes a query's held factors (`freq_held.zig`).
pub const prepareHeld = freq_held.prepareHeld;
/// Backend-typed held factors (`freq_held.zig`).
pub const DeviceHeld = freq_held.DeviceHeld;

/// Every point's leading sweep. Dense engines and frequency-dependent
/// device terms use their existing host solve. A driver/allocation/solve
/// failure is returned so the caller can discard the partial precompute
/// and run the original analysis live, including its streaming behavior.
pub fn solveAll(gpa: std.mem.Allocator, requested: gpu.Request, points: []const Point) !Stats {
    if (points.len == 0) return .{};
    switch (requested) {
        .cpu => return error.GpuBackendUnavailable,
        .cuda => if (comptime artifacts.has_cuda) {
            return run(.cuda, gpa, points);
        },
        .hip => if (comptime artifacts.has_hip) {
            return run(.hip, gpa, points);
        },
        .auto => {
            if (comptime artifacts.has_cuda) return run(.cuda, gpa, points);
            if (comptime artifacts.has_hip) return run(.hip, gpa, points);
        },
    }
    return error.GpuBackendUnavailable;
}

/// Executes exactly one original chunk per point, without checkpoints.
/// Numerical failures belong to their point. Dynamic terms use that
/// point's original host solve. A failed device group resumes through its
/// own live solvers before the caller can observe any output.
pub fn solveRound(gpa: std.mem.Allocator, requested: gpu.Request, points: []const RoundPoint, failures: []?SolveError) std.mem.Allocator.Error!Stats {
    var driver = RoundDriver.init(gpa, requested);
    defer driver.deinit();
    return driver.solve(points, failures);
}

// ---- private ----

fn solveHost(p: RoundPoint, failure: *?SolveError) void {
    p.fs.solveBatch(p.omegas, p.dyn, p.rhs, p.out, p.adjoint) catch |err| {
        failure.* = err;
    };
}

fn runRound(comptime backend: gompute.Backend, gpa: std.mem.Allocator, points: []const RoundPoint, failures: []?SolveError, engine: *?Engine(backend), disabled: *bool) std.mem.Allocator.Error!Stats {
    const bases = try gpa.alloc(usize, points.len);
    defer gpa.free(bases);
    const counts = try gpa.alloc(usize, points.len);
    defer gpa.free(counts);
    const chunks = try gpa.alloc(?Chunk, points.len);
    defer gpa.free(chunks);
    const cchunks = try gpa.alloc(?CChunk, points.len);
    defer gpa.free(cchunks);
    const used = try gpa.alloc(bool, points.len);
    defer gpa.free(used);
    const ids = try gpa.alloc(usize, points.len);
    defer gpa.free(ids);
    const bad = try gpa.alloc(u64, points.len);
    defer gpa.free(bad);
    const solved = try gpa.alloc(bool, points.len);
    defer gpa.free(solved);
    @memset(bases, 0);
    @memset(chunks, null);
    @memset(cchunks, null);
    @memset(bad, 0);
    @memset(solved, false);
    var stats: Stats = .{ .points = points.len };
    // Each point's chunk takes the lanes its host `solveBatch` would: the
    // complex lanes while they are on, else the stacked ones, so a device
    // replay and every host fallback are bitwise the scalar solve.
    for (points, counts, chunks, cchunks, failures) |p, *count, *chunk, *cchunk, *failure| {
        count.* = p.omegas.len;
        std.debug.assert(count.* > 0 and count.* <= FS.W);
        std.debug.assert(p.out.len == p.omegas.len * p.rhs.len);
        if (p.dyn.slots.len != 0) {
            solveHost(p, failure);
            continue;
        }
        if (p.fs.complex_lanes) {
            cchunk.* = p.fs.beginComplexChunk(p.omegas, 0, count.*, p.rhs, p.out, p.adjoint) catch |err| {
                failure.* = err;
                continue;
            };
            // Null with the lanes still on: solved serially.
            if (cchunk.* != null or p.fs.complex_lanes) continue;
        }
        chunk.* = p.fs.beginLaneChunk(p.omegas, 0, count.*, p.rhs, p.out, p.adjoint) catch |err| {
            failure.* = err;
            continue;
        };
    }
    @memset(used, false);
    for (chunks, 0..) |candidate, first| {
        if (candidate == null or used[first]) continue;
        const n = gather(points, chunks, first, used, ids);
        if (deviceGroup(backend, gpa, engine, disabled, points, bases, counts, chunks, ids[0..n], bad, &stats))
            for (ids[0..n]) |id| {
                solved[id] = true;
            };
    }
    @memset(used, false);
    for (cchunks, 0..) |candidate, first| {
        if (candidate == null or used[first]) continue;
        const n = gather(points, cchunks, first, used, ids);
        if (deviceGroup(backend, gpa, engine, disabled, points, bases, counts, cchunks, ids[0..n], bad, &stats))
            for (ids[0..n]) |id| {
                solved[id] = true;
            };
    }
    for (points, counts, chunks, cchunks, bad, solved, failures) |p, count, chunk, cchunk, mask, on_device, *failure| {
        // A chunk the device did not solve has only laid its tape and run
        // no peel, so `solveBatch` replays exactly that tape and finishes
        // in scalar order.
        if (chunk == null and cchunk == null) continue;
        if (!on_device) {
            solveHost(p, failure);
        } else if (cchunk) |cc| {
            // False: the fresh tape failed at its own centre, the complex
            // lanes are off, and `solveBatch` redoes the chunk stacked.
            const kept = p.fs.finishComplexChunk(p.omegas, 0, count, mask, cc.fresh, p.rhs, p.out, p.adjoint) catch |err| {
                failure.* = err;
                continue;
            };
            if (!kept) solveHost(p, failure);
        } else {
            p.fs.finishLaneChunk(p.omegas, 0, count, mask, p.rhs, p.out, p.adjoint) catch |err| {
                failure.* = err;
            };
        }
    }
    return stats;
}

/// Runs one launch group (`Engine.group` for stacked chunks, `groupComplex`
/// for complex ones) unless the device is `disabled`; true when it ran. A
/// driver failure disables the device for the rest of the driver's life.
fn deviceGroup(comptime backend: gompute.Backend, gpa: std.mem.Allocator, engine: *?Engine(backend), disabled: *bool, points: []const RoundPoint, bases: []const usize, counts: []const usize, chunks: anytype, ids: []const usize, bad: []u64, stats: *Stats) bool {
    if (!disabled.* and engine.* == null) {
        engine.* = Engine(backend).init() catch null;
        disabled.* = engine.* == null;
    }
    if (disabled.*) return false;
    const e = &engine.*.?;
    const result = if (@TypeOf(chunks) == []const ?CChunk or @TypeOf(chunks) == []?CChunk)
        e.groupComplex(gpa, points, bases, counts, chunks, ids, bad, stats)
    else
        e.group(gpa, points, bases, counts, chunks, ids, bad, stats);
    result catch {
        // A group synchronizes before unwinding its host buffers.
        e.deinit();
        engine.* = null;
        disabled.* = true;
        return false;
    };
    return true;
}

fn run(comptime backend: gompute.Backend, gpa: std.mem.Allocator, points: []const Point) !Stats {
    const bases = try gpa.alloc(usize, points.len);
    defer gpa.free(bases);
    const counts = try gpa.alloc(usize, points.len);
    defer gpa.free(counts);
    const chunks = try gpa.alloc(?Chunk, points.len);
    defer gpa.free(chunks);
    const cchunks = try gpa.alloc(?CChunk, points.len);
    defer gpa.free(cchunks);
    const used = try gpa.alloc(bool, points.len);
    defer gpa.free(used);
    const ids = try gpa.alloc(usize, points.len);
    defer gpa.free(ids);
    const bad = try gpa.alloc(u64, points.len);
    defer gpa.free(bad);
    @memset(bases, 0);
    var stats: Stats = .{ .points = points.len };
    var engine: ?Engine(backend) = null;
    defer if (engine) |*e| e.deinit();
    for (points, bases) |p, *base| {
        std.debug.assert(p.out.len == p.omegas.len * p.rhs.len);
        if (p.circuit.ac_dyn_slots.len != 0) {
            try freq.solveAll(gpa, p.fs, p.circuit, p.x_op, p.omegas, p.rhs, p.adjoint, p.out);
            base.* = p.omegas.len;
        }
    }
    // Chunks per point per round: up to a quantum (64 frequencies, the
    // failure mask's width), bounded by the device planes a lane needs.
    // ponytail: a lane is priced at 16 doubles per stacked unknown (the
    // 4 nnz fill, L/U and the solve planes on a typical MNA pattern); size
    // from the built tab if that misjudges.
    const lane_bytes = 16 * @sizeOf(f64) * @as(usize, @max(points[0].fs.nn, 1));
    const span_chunks = std.math.clamp((1 << 30) / (lane_bytes * points.len * FS.W), 1, freq.quantum / FS.W);
    while (true) {
        var any = false;
        @memset(used, false);
        @memset(bad, 0);
        for (points, bases, counts, chunks, cchunks) |p, base, *count, *chunk, *cchunk| {
            const first = @min(FS.W, p.omegas.len - base);
            count.* = first;
            chunk.* = null;
            cchunk.* = null;
            if (first == 0) continue;
            any = true;
            // The host's complex lanes first, as `solveBatch` runs them; a
            // null with the lanes still on solved the chunk serially.
            if (p.fs.complex_lanes) cchunk.* = try p.fs.beginComplexChunk(p.omegas, base, first, p.rhs, p.out, p.adjoint);
            if (cchunk.* == null and !p.fs.complex_lanes)
                chunk.* = try p.fs.beginLaneChunk(p.omegas, base, first, p.rhs, p.out, p.adjoint);
            // Later chunks replay the same tape: without a failed lane the
            // host solver's history does not change between chunks, so the
            // whole span is the chunks `solveBatch` would run one by one.
            // Chunks past the first failure are redone next round.
            if (chunk.* != null or cchunk.* != null) {
                const end = @min(p.omegas.len, (base / freq.quantum + 1) * freq.quantum, base + span_chunks * FS.W);
                count.* = end - base;
            }
        }
        if (!any) break;
        for (chunks, 0..) |candidate, first| {
            if (candidate == null or used[first]) continue;
            const n = gather(points, chunks, first, used, ids);
            if (engine == null) engine = try Engine(backend).init();
            try engine.?.group(gpa, points, bases, counts, chunks, ids[0..n], bad, &stats);
        }
        @memset(used, false);
        for (cchunks, 0..) |candidate, first| {
            if (candidate == null or used[first]) continue;
            const n = gather(points, cchunks, first, used, ids);
            if (engine == null) engine = try Engine(backend).init();
            try engine.?.groupComplex(gpa, points, bases, counts, cchunks, ids[0..n], bad, &stats);
        }
        for (points, bases, counts, chunks, cchunks, bad) |p, *base, count, chunk, cchunk, mask| {
            if (count == 0) continue;
            if (cchunk) |cc| {
                // As the stacked span below; a fresh tape that fails at
                // its own centre leaves the chunk to the stacked lanes.
                var off: usize = 0;
                while (off < count) {
                    const cnt = @min(FS.W, count - off);
                    const sub = (mask >> @intCast(off)) & ((@as(u64, 1) << @intCast(cnt)) - 1);
                    off += cnt;
                    if (sub == 0) continue;
                    if (!try p.fs.finishComplexChunk(p.omegas, base.* + off - cnt, cnt, sub, cc.fresh and off == cnt, p.rhs, p.out, p.adjoint)) off -= cnt;
                    break;
                }
                base.* += off;
            } else if (chunk != null) {
                // Accept chunks in order up to and including the first with
                // a failed lane, which peels as `finishLaneChunk` does.
                var off: usize = 0;
                while (off < count) {
                    const cnt = @min(FS.W, count - off);
                    const sub = (mask >> @intCast(off)) & ((@as(u64, 1) << @intCast(cnt)) - 1);
                    off += cnt;
                    if (sub == 0) continue;
                    try p.fs.finishLaneChunk(p.omegas, base.* + off - cnt, cnt, sub, p.rhs, p.out, p.adjoint);
                    break;
                }
                base.* += off;
            } else base.* += count;
            if (base.* % freq.quantum == 0 or base.* == p.omegas.len)
                try p.circuit.checkpoint(.{ .phase = .frequency, .completed = base.*, .total = p.omegas.len });
        }
    }
    return stats;
}

/// Lists in `ids` every unused chunk from `first` on (`first` included)
/// that shares `chunks[first]`'s tape, adjoint flag and rhs width, marks
/// them `used`, and returns how many. `chunks[first]` must be non-null.
fn gather(points: anytype, chunks: anytype, first: usize, used: []bool, ids: []usize) usize {
    const head = chunks[first].?;
    var n: usize = 0;
    for (chunks[first..], first..) |other, k| {
        if (used[k] or points[k].adjoint != points[first].adjoint or points[k].rhs.len != points[first].rhs.len) continue;
        const chunk = other orelse continue;
        if (!head.sameTape(chunk)) continue;
        ids[n] = k;
        n += 1;
        used[k] = true;
    }
    return n;
}
