//! GPU frequency replay over (point, frequency) lanes. The host owns each
//! point's FreqSolver and advances exactly its usual W-frequency chunks;
//! only equal pivot tapes share a launch. Failed lanes peel in point-local
//! frequency order before the next round. Analysis readout stays unchanged.
//! The sweep precompute (`solveAll`) replays the host's complex tapes while
//! a point's complex lanes are on, its stacked tapes after; live rounds and
//! held preconditioner factors use stacked tapes.
const std = @import("std");
const gompute = @import("gompute");
const artifacts = @import("gompute_kernels");
const solver = @import("solver");
const lk = solver.lane_kernels;
const fk = solver.frequency_kernels;
const FS = solver.freq_solve.FreqSolver;
const Chunk = FS.LaneChunk;
const CChunk = FS.ComplexChunk;
const root = @import("../types.zig");
const gpu = @import("context.zig");
const freq = @import("../ac/freq.zig");

pub const available = artifacts.has_cuda or artifacts.has_hip;
const block = @import("device").abi.gpu_lane_block_size;
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

pub const SolveError = root.ErrorOf(FS.solveBatch);

pub const Stats = struct {
    points: usize = 0,
    lanes: usize = 0,
    groups: usize = 0,
    peels: usize = 0,
};

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

/// Keeps kernel modules and scratch buffers across live requests. A device
/// failure disables this driver for its remaining rounds; each point still
/// continues on its own original solver.
pub const RoundDriver = struct {
    allocator: std.mem.Allocator,
    requested: gpu.Request,
    cuda: if (artifacts.has_cuda) ?Engine(.cuda) else void = if (artifacts.has_cuda) null else {},
    hip: if (artifacts.has_hip) ?Engine(.hip) else void = if (artifacts.has_hip) null else {},
    disabled: bool = false,

    pub fn init(allocator: std.mem.Allocator, requested: gpu.Request) RoundDriver {
        return .{ .allocator = allocator, .requested = requested };
    }

    pub fn deinit(self: *RoundDriver) void {
        if (comptime artifacts.has_cuda) if (self.cuda) |*engine| engine.deinit();
        if (comptime artifacts.has_hip) if (self.hip) |*engine| engine.deinit();
        self.* = undefined;
    }

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
    const used = try gpa.alloc(bool, points.len);
    defer gpa.free(used);
    const ids = try gpa.alloc(usize, points.len);
    defer gpa.free(ids);
    const bad = try gpa.alloc(u64, points.len);
    defer gpa.free(bad);
    @memset(bases, 0);
    @memset(chunks, null);
    @memset(used, false);
    @memset(bad, 0);
    var stats: Stats = .{ .points = points.len };
    // The device replays stacked tapes; host fallbacks must match them.
    for (points) |p| p.fs.complex_lanes = false;
    for (points, counts, chunks, failures) |p, *count, *chunk, *failure| {
        count.* = p.omegas.len;
        std.debug.assert(count.* > 0 and count.* <= FS.W);
        std.debug.assert(p.out.len == p.omegas.len * p.rhs.len);
        if (p.dyn.slots.len != 0) {
            solveHost(p, failure);
            continue;
        }
        chunk.* = p.fs.beginLaneChunk(p.omegas, 0, count.*, p.rhs, p.out, p.adjoint) catch |err| {
            failure.* = err;
            continue;
        };
    }
    for (chunks, 0..) |candidate, first| {
        const head = candidate orelse continue;
        if (used[first]) continue;
        var n: usize = 0;
        for (chunks[first..], first..) |other, k| {
            if (used[k] or points[k].adjoint != points[first].adjoint or points[k].rhs.len != points[first].rhs.len) continue;
            const chunk = other orelse continue;
            if (!head.sameTape(chunk)) continue;
            ids[n] = k;
            n += 1;
            used[k] = true;
        }
        if (!disabled.* and engine.* == null) {
            engine.* = Engine(backend).init() catch null;
            disabled.* = engine.* == null;
        }
        if (!disabled.*) engine.*.?.group(gpa, points, bases, counts, chunks, ids[0..n], bad, &stats) catch {
            // group synchronizes before unwinding its host buffers.
            engine.*.?.deinit();
            engine.* = null;
            disabled.* = true;
        };
        for (ids[0..n]) |id| {
            const p = points[id];
            if (disabled.*) {
                // beginLaneChunk has only selected this chunk's center
                // tape. No peel has run, so its ordinary solveBatch now
                // replays exactly that tape and finishes in scalar order.
                solveHost(p, &failures[id]);
            } else {
                p.fs.finishLaneChunk(p.omegas, 0, counts[id], bad[id], p.rhs, p.out, p.adjoint) catch |err| {
                    failures[id] = err;
                };
            }
        }
    }
    return stats;
}

/// Retained harmonic/sideband factors for a periodic Krylov operator.
/// `factorEach` still chooses its tapes and identity blocks on the host;
/// each preconditioner application solves all non-identity harmonics on
/// the device. The type erases only backend dispatch, not numerical work.
pub const Held = struct {
    context: *anyopaque,
    refresh_fn: *const fn (*anyopaque, *const FS, usize) anyerror!void,
    solve_fn: *const fn (*anyopaque, []const f64, []f64, bool) bool,
    deinit_fn: *const fn (*anyopaque) void,

    pub fn init(gpa: std.mem.Allocator, fs: *const FS, count: usize, stats: bool) !Held {
        if (comptime artifacts.has_cuda) return DeviceHeld(.cuda).create(gpa, fs, count, stats);
        if (comptime artifacts.has_hip) return DeviceHeld(.hip).create(gpa, fs, count, stats);
        return error.GpuBackendUnavailable;
    }

    pub fn refresh(self: *Held, fs: *const FS, count: usize) !void {
        try self.refresh_fn(self.context, fs, count);
    }

    /// False on a driver failure: the caller uses its unchanged host
    /// factorEach result. No partial device result is published.
    pub fn solve(self: *Held, rhs: []const f64, out: []f64, adjoint: bool) bool {
        return self.solve_fn(self.context, rhs, out, adjoint);
    }

    pub fn deinit(self: *Held) void {
        self.deinit_fn(self.context);
        self.* = undefined;
    }
};

/// Refresh the periodic preconditioner when this query already uses the
/// GPU. A machine failure disables this optional acceleration for the
/// rest of the query; a dense or all-identity factor stays on the host.
pub fn prepareHeld(held: *?Held, disabled: *bool, ckt: *const root.Circuit, fs: *const FS, count: usize, gpa: std.mem.Allocator) void {
    if (disabled.* or ckt.gpu == null) return;
    if (fs.heldTape() == null) {
        if (held.*) |*h| h.deinit();
        held.* = null;
        return;
    }
    if (held.*) |*h| {
        h.refresh(fs, count) catch {
            h.deinit();
            held.* = null;
            disabled.* = true;
        };
    } else {
        held.* = Held.init(gpa, fs, count, ckt.knobs.gpu_stats != .off) catch {
            disabled.* = true;
            return;
        };
    }
}

/// Backend-typed factors for another resident GPU operator. Its stream
/// must finish before these factors are refreshed or deinitialized.
pub fn DeviceHeld(comptime backend: gompute.Backend) type {
    return struct {
        const Self = @This();
        const Raw = gompute.RawByName(backend);
        engine: Engine(backend),
        gpa: std.mem.Allocator,
        tab: lk.LaneTab = undefined,
        count: usize = 0,
        nn: usize = 0,
        work: []f64 = &.{},
        identity: []u8 = &.{},
        poisoned: bool = false,
        stats: bool,
        solves: usize = 0,
        factors: usize = 0,
        device_refactors: usize = 0,

        fn create(gpa: std.mem.Allocator, fs: *const FS, count: usize, stats: bool) !Held {
            const self = try gpa.create(Self);
            errdefer gpa.destroy(self);
            self.* = try init(gpa, fs, count, stats);
            return .{ .context = self, .refresh_fn = refresh, .solve_fn = apply, .deinit_fn = destroy };
        }

        pub fn init(gpa: std.mem.Allocator, fs: *const FS, count: usize, stats: bool) !Self {
            var self: Self = .{ .engine = try Engine(backend).init(), .gpa = gpa, .stats = stats };
            errdefer {
                self.engine.deinit();
                gpa.free(self.work);
                gpa.free(self.identity);
            }
            try self.update(fs, count);
            return self;
        }

        /// A factorEach provider whose numeric factors never leave the
        /// device. Only the host's full pivot candidates and failure-mask
        /// decisions remain serial, exactly as FreqSolver.factorEach.
        pub fn initFactor(gpa: std.mem.Allocator, fs: *FS, omegas: []const f64, stats: bool) !Self {
            var self: Self = .{ .engine = try Engine(backend).init(), .gpa = gpa, .stats = stats };
            errdefer self.deinit();
            _ = try self.factorEach(fs, omegas);
            return self;
        }

        /// Returns false for an all-identity result. The caller must not
        /// call solveDevice then, because no sparse tape may exist.
        pub fn factorEach(self: *Self, fs: *FS, omegas: []const f64) !bool {
            if (self.poisoned) return error.GpuBackendUnavailable;
            std.debug.assert(omegas.len > 0);
            if (self.identity.len != omegas.len) {
                const next = try self.gpa.alloc(u8, omegas.len);
                self.gpa.free(self.identity);
                self.identity = next;
            }
            self.count = omegas.len;
            self.nn = fs.nn;
            var tried: [3]usize = undefined;
            var tries: usize = 0;
            var best: usize = undefined;
            var best_held: usize = std.math.maxInt(usize);
            var candidate = omegas.len / 2;
            while (tries < tried.len) {
                tried[tries] = candidate;
                tries += 1;
                const held = try self.replayHeld(fs, omegas, candidate);
                if (held < best_held) {
                    best = candidate;
                    best_held = held;
                }
                if (held == 0) break;
                candidate = for (fs.held_id.items) |k| {
                    if (std.mem.indexOfScalar(usize, tried[0..tries], k) == null) break k;
                } else break;
            }
            if (best != tried[tries - 1]) _ = try self.replayHeld(fs, omegas, best);
            fs.completeHeldPolicy(omegas.len, self.poisoned);
            self.factors += 1;
            // A failed device replay finished this same candidate policy
            // on the host. Tell the caller to stop device Krylov, without
            // discarding or re-factoring those recovered host factors.
            if (self.poisoned) return error.GpuBackendUnavailable;
            return fs.held_id.items.len != omegas.len;
        }

        fn replayHeld(self: *Self, fs: *FS, omegas: []const f64, candidate: usize) !usize {
            const tape = (try fs.beginHeldTape(omegas, candidate)) orelse {
                @memset(self.identity, 1);
                return omegas.len;
            };
            if (self.poisoned) return fs.replayCurrentHeld(omegas);
            return self.replayDeviceHeld(fs, tape, omegas) catch {
                self.engine.stream.synchronize() catch {};
                self.poisoned = true;
                // The full factor already selected this candidate. Only
                // replay its numeric harmonic blocks, then let the
                // original tried/best loop select any remaining tapes.
                return fs.replayCurrentHeld(omegas);
            };
        }

        fn replayDeviceHeld(self: *Self, fs: *FS, tape: Chunk, omegas: []const f64) !usize {
            if (@import("builtin").is_test) {
                const sequence = test_hooks.factor_candidates;
                test_hooks.factor_candidates += 1;
                if (test_hooks.factor_failure_after) |at| if (sequence == at) {
                    test_hooks.factor_failure_after = null;
                    return error.InjectedDriverFailure;
                };
            }
            var arena = std.heap.ArenaAllocator.init(self.gpa);
            defer arena.deinit();
            defer self.engine.stream.synchronize() catch {};
            const a = arena.allocator();
            const N = omegas.len;
            const nnz = tape.g.len;
            const nn: usize = fs.nn;
            var built = try lk.buildTab(a, tape.lu, tape.col_ptr, tape.growth, solver.direct.pivot_tol);
            built.tab.wide = std.math.inf(f64);
            const positions = try a.alloc(u32, 4 * nnz);
            fk.positions(tape.src_col_ptr, positions);
            const point = try a.alloc(u32, N);
            @memset(point, 0);
            const active = try a.alloc(u8, N);
            @memset(active, 1);
            try self.engine.upload(.idx, built.idx);
            try self.engine.upload(.positions, positions);
            try self.engine.upload(.g, tape.g);
            try self.engine.upload(.c, tape.c);
            try self.engine.upload(.point, point);
            try self.engine.upload(.omega, omegas);
            try self.engine.upload(.active, active);
            try self.engine.ensure(.a, 4 * nnz * N * @sizeOf(f64));
            try self.engine.ensure(.lx, @as(usize, built.tab.nnz_l) * N * @sizeOf(f64));
            try self.engine.ensure(.ux, @as(usize, built.tab.nnz_u) * N * @sizeOf(f64));
            try self.engine.ensure(.diag, nn * N * @sizeOf(f64));
            try self.engine.zero(.w, nn * N * @sizeOf(f64));
            try self.engine.ensure(.y, nn * N * @sizeOf(f64));
            try self.engine.ensure(.fail, N);
            try self.engine.zero(.fails, @sizeOf(u32));
            const nl: u64 = N;
            const points: u64 = 1;
            const nz: u64 = nnz;
            const scalar = gompute.interface.arg;
            try self.engine.launch(&self.engine.fill, nnz * N, &.{
                scalar(&nl),         scalar(&points),     scalar(&nz),             self.engine.arg(.positions),
                self.engine.arg(.g), self.engine.arg(.c), self.engine.arg(.point), self.engine.arg(.omega),
                self.engine.arg(.a),
            });
            try self.engine.launch(&self.engine.refactor, N, &.{
                scalar(&nl),             scalar(&built.tab),   self.engine.arg(.idx),  self.engine.arg(.active), self.engine.arg(.a),
                self.engine.arg(.lx),    self.engine.arg(.ux), self.engine.arg(.diag), self.engine.arg(.w),      self.engine.arg(.fail),
                self.engine.arg(.fails),
            });
            try self.engine.buffer(.fail).downloadAtAsync(self.identity.ptr, 0, N, &self.engine.stream);
            try self.engine.stream.synchronize();
            fs.finishHeldTape(self.identity);
            self.tab = built.tab;
            self.device_refactors += N;
            return fs.held_id.items.len;
        }

        fn refresh(context: *anyopaque, fs: *const FS, count: usize) !void {
            const self: *Self = @ptrCast(@alignCast(context));
            try self.update(fs, count);
        }

        pub fn update(self: *Self, fs: *const FS, count: usize) !void {
            if (self.poisoned) return error.GpuBackendUnavailable;
            const tape = fs.heldTape() orelse return error.NoLaneTape;
            var arena = std.heap.ArenaAllocator.init(self.gpa);
            defer arena.deinit();
            defer self.engine.stream.synchronize() catch {};
            const a = arena.allocator();
            const built = try lk.buildTab(a, tape.lu, tape.col_ptr, tape.growth, solver.direct.pivot_tol);
            const lower = try a.alloc(f64, tape.lu.lx.items.len * count);
            const upper = try a.alloc(f64, tape.lu.ux.items.len * count);
            const diag = try a.alloc(f64, fs.nn * count);
            fs.packHeld(count, lower, upper, diag);
            const values = @as(usize, fs.nn) * count;
            if (self.work.len != 2 * values) {
                const next = try self.gpa.alloc(f64, 2 * values);
                self.gpa.free(self.work);
                self.work = next;
            }
            if (self.identity.len != count) {
                const next = try self.gpa.alloc(u8, count);
                self.gpa.free(self.identity);
                self.identity = next;
            }
            @memset(self.identity, 0);
            for (fs.held_id.items) |k| self.identity[k] = 1;
            try self.engine.upload(.idx, built.idx);
            try self.engine.upload(.lx, lower);
            try self.engine.upload(.ux, upper);
            try self.engine.upload(.diag, diag);
            try self.engine.upload(.fail, self.identity);
            try self.engine.ensure(.rhs, values * @sizeOf(f64));
            try self.engine.ensure(.y, values * @sizeOf(f64));
            try self.engine.ensure(.out, values * @sizeOf(f64));
            self.tab = built.tab;
            self.count = count;
            self.nn = fs.nn;
            self.factors += 1;
        }

        fn apply(context: *anyopaque, rhs: []const f64, out: []f64, adjoint: bool) bool {
            const self: *Self = @ptrCast(@alignCast(context));
            if (self.poisoned) return false;
            self.solve(rhs, out, adjoint) catch {
                self.engine.stream.synchronize() catch {};
                self.poisoned = true;
                return false;
            };
            self.solves += 1;
            return true;
        }

        fn solve(self: *Self, rhs: []const f64, out: []f64, adjoint: bool) !void {
            const N = self.count;
            const n = self.nn;
            std.debug.assert(rhs.len == n * N and out.len == rhs.len);
            const input = self.work[0..rhs.len];
            const output = self.work[rhs.len..];
            for (0..N) |k| for (0..n) |i| {
                input[i * N + k] = rhs[k * n + i];
            };
            try self.engine.upload(.rhs, input);
            try self.solveDevice(self.engine.buffer(.rhs), self.engine.buffer(.out), adjoint, &self.engine.stream);
            try self.engine.buffer(.out).downloadAtAsync(output.ptr, 0, output.len * @sizeOf(f64), &self.engine.stream);
            try self.engine.stream.synchronize();
            for (0..N) |k| for (0..n) |i| {
                out[k * n + i] = if (self.identity[k] != 0) rhs[k * n + i] else output[i * N + k];
            };
        }

        /// RHS/output are row-major with harmonic lanes innermost.
        /// Identity harmonics are masked; the caller copies their RHS.
        pub fn solveDevice(self: *Self, rhs: *Raw.Buffer, out: *Raw.Buffer, adjoint: bool, stream: *Raw.Stream) !void {
            const nl: u64 = self.count;
            const trans: u64 = @intFromBool(adjoint);
            const scalar = gompute.interface.arg;
            try self.engine.solve.launchOn(stream, gompute.Dim3.linear(@intCast(self.count), block), .{ .x = block }, 0, &.{
                scalar(&nl),          scalar(&self.tab),    self.engine.arg(.idx),  self.engine.arg(.fail),
                self.engine.arg(.lx), self.engine.arg(.ux), self.engine.arg(.diag), rhs.argPtr(),
                self.engine.arg(.y),  out.argPtr(),         scalar(&trans),
            });
        }

        pub fn identityBuffer(self: *Self) *Raw.Buffer {
            return self.engine.buffer(.fail);
        }

        pub fn deinit(self: *Self) void {
            if (self.stats and self.solves != 0)
                std.debug.print("gpu-freq-held: {d} harmonics, {d} factorizations, {d} device solves\n", .{ self.count, self.factors, self.solves });
            self.engine.deinit();
            self.gpa.free(self.work);
            self.gpa.free(self.identity);
            self.* = undefined;
        }

        fn destroy(context: *anyopaque) void {
            const self: *Self = @ptrCast(@alignCast(context));
            const gpa = self.gpa;
            self.deinit();
            gpa.destroy(self);
        }
    };
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
            const head = candidate orelse continue;
            if (used[first]) continue;
            var n: usize = 0;
            for (chunks[first..], first..) |other, k| {
                if (used[k] or points[k].adjoint != points[first].adjoint or points[k].rhs.len != points[first].rhs.len) continue;
                const c = other orelse continue;
                if (!head.sameTape(c)) continue;
                ids[n] = k;
                n += 1;
                used[k] = true;
            }
            if (engine == null) engine = try Engine(backend).init();
            try engine.?.group(gpa, points, bases, counts, chunks, ids[0..n], bad, &stats);
        }
        @memset(used, false);
        for (cchunks, 0..) |candidate, first| {
            const head = candidate orelse continue;
            if (used[first]) continue;
            var n: usize = 0;
            for (cchunks[first..], first..) |other, k| {
                if (used[k] or points[k].adjoint != points[first].adjoint or points[k].rhs.len != points[first].rhs.len) continue;
                const c = other orelse continue;
                if (!head.sameTape(c)) continue;
                ids[n] = k;
                n += 1;
                used[k] = true;
            }
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

fn Engine(comptime backend: gompute.Backend) type {
    return struct {
        const Self = @This();
        const Raw = gompute.RawByName(backend);
        const Buffer = Raw.Buffer;
        const Id = enum { idx, positions, g, c, point, omega, a, lx, ux, diag, w, rhs, y, out, fail, fails, active };
        const Storage = struct {
            value: ?Buffer = null,
            capacity: usize = 0,
        };
        fill: Raw,
        refactor: Raw,
        solve: Raw,
        cfill: Raw,
        crefactor: Raw,
        csolve: Raw,
        stream: Raw.Stream,
        buffers: [@typeInfo(Id).@"enum".field_names.len]Storage = @splat(.{}),

        fn init() !Self {
            var fill = try gompute.rawKernelByName(backend, "arp_freq_fill", 0);
            errdefer fill.deinit();
            var refactor = try gompute.rawKernelByName(backend, "arp_lane_refactor", 0);
            errdefer refactor.deinit();
            var solve = try gompute.rawKernelByName(backend, "arp_freq_solve", 0);
            errdefer solve.deinit();
            var cfill = try gompute.rawKernelByName(backend, "arp_freq_cfill", 0);
            errdefer cfill.deinit();
            var crefactor = try gompute.rawKernelByName(backend, "arp_freq_crefactor", 0);
            errdefer crefactor.deinit();
            var csolve = try gompute.rawKernelByName(backend, "arp_freq_csolve", 0);
            errdefer csolve.deinit();
            const stream = try fill.createStream();
            return .{ .fill = fill, .refactor = refactor, .solve = solve, .cfill = cfill, .crefactor = crefactor, .csolve = csolve, .stream = stream };
        }

        fn deinit(self: *Self) void {
            self.stream.synchronize() catch {};
            for (&self.buffers) |*s| if (s.value) |*b| b.free();
            self.stream.deinit();
            self.fill.deinit();
            self.refactor.deinit();
            self.solve.deinit();
            self.cfill.deinit();
            self.crefactor.deinit();
            self.csolve.deinit();
        }

        fn ensure(self: *Self, id: Id, bytes: usize) !void {
            const s = &self.buffers[@backingInt(id)];
            const need = @max(bytes, 1);
            if (s.capacity >= need) return;
            const b = try self.fill.alloc(need);
            if (s.value) |*old| old.free();
            s.* = .{ .value = b, .capacity = need };
        }

        fn buffer(self: *Self, id: Id) *Buffer {
            return &self.buffers[@backingInt(id)].value.?;
        }

        fn upload(self: *Self, id: Id, values: anytype) !void {
            const bytes = std.mem.sliceAsBytes(values);
            try self.ensure(id, bytes.len);
            if (bytes.len != 0) try self.buffer(id).upload(bytes.ptr, bytes.len);
        }

        fn zero(self: *Self, id: Id, bytes: usize) !void {
            try self.ensure(id, bytes);
            if (bytes != 0) try self.buffer(id).fillAsync(0, bytes, &self.stream);
        }

        fn arg(self: *Self, id: Id) gompute.interface.Arg {
            return self.buffer(id).argPtr();
        }

        fn launch(self: *Self, kernel: *Raw, threads: usize, args: []const gompute.interface.Arg) !void {
            try kernel.launchOn(&self.stream, gompute.Dim3.linear(@intCast(threads), block), .{ .x = block }, 0, args);
        }

        fn group(self: *Self, gpa: std.mem.Allocator, points: anytype, bases: []const usize, counts: []const usize, chunks: []const ?Chunk, ids: []const usize, bad: []u64, stats: *Stats) !void {
            // Complete queued work before host temporaries or a replaced
            // device allocation can be released after a launch failure.
            var arena = std.heap.ArenaAllocator.init(gpa);
            defer arena.deinit();
            defer self.stream.synchronize() catch {};
            const a = arena.allocator();
            const first = ids[0];
            const head = chunks[first].?;
            const nnz = head.g.len;
            const nn: usize = points[first].fs.nn;
            var N: usize = 0;
            for (ids) |id| N += counts[id];
            const np = ids.len;
            var built = try lk.buildTab(a, head.lu, head.col_ptr, head.growth, solver.direct.pivot_tol);
            // FreqSolver's LaneLu uses the growth monitor, without
            // LaneSolver's additional Newton multiplier threshold.
            built.tab.wide = std.math.inf(f64);
            const pos = try a.alloc(u32, 4 * nnz);
            fk.positions(head.src_col_ptr, pos);
            const g = try a.alloc(f64, nnz * np);
            const c = try a.alloc(f64, nnz * np);
            const point = try a.alloc(u32, N);
            const omega = try a.alloc(f64, N);
            const rhs = try a.alloc(f64, nn * N);
            const out = try a.alloc(f64, nn * N);
            const fail = try a.alloc(u8, N);
            const active = try a.alloc(u8, N);
            @memset(active, 1);
            var lane: usize = 0;
            for (ids, 0..) |id, p| {
                const chunk = chunks[id].?;
                for (chunk.g, chunk.c, 0..) |gv, cv, slot| {
                    g[slot * np + p] = gv;
                    c[slot * np + p] = cv;
                }
                for (0..counts[id]) |k| {
                    point[lane] = @intCast(p);
                    omega[lane] = points[id].omegas[bases[id] + k];
                    lane += 1;
                }
            }
            try self.upload(.idx, built.idx);
            try self.upload(.positions, pos);
            try self.upload(.g, g);
            try self.upload(.c, c);
            try self.upload(.point, point);
            try self.upload(.omega, omega);
            try self.upload(.active, active);
            try self.ensure(.a, 4 * nnz * N * @sizeOf(f64));
            try self.ensure(.lx, @as(usize, built.tab.nnz_l) * N * @sizeOf(f64));
            try self.ensure(.ux, @as(usize, built.tab.nnz_u) * N * @sizeOf(f64));
            try self.ensure(.diag, nn * N * @sizeOf(f64));
            try self.zero(.w, nn * N * @sizeOf(f64));
            try self.ensure(.y, nn * N * @sizeOf(f64));
            try self.ensure(.out, nn * N * @sizeOf(f64));
            try self.ensure(.fail, N);
            try self.zero(.fails, @sizeOf(u32));
            const nl: u64 = N;
            const np64: u64 = np;
            const nz64: u64 = nnz;
            const adjoint: u64 = @intFromBool(points[first].adjoint);
            const scalar = gompute.interface.arg;
            try self.launch(&self.fill, nnz * N, &.{
                scalar(&nl),  scalar(&np64), scalar(&nz64),    self.arg(.positions),
                self.arg(.g), self.arg(.c),  self.arg(.point), self.arg(.omega),
                self.arg(.a),
            });
            try self.launch(&self.refactor, N, &.{
                scalar(&nl),      scalar(&built.tab), self.arg(.idx),  self.arg(.active), self.arg(.a),
                self.arg(.lx),    self.arg(.ux),      self.arg(.diag), self.arg(.w),      self.arg(.fail),
                self.arg(.fails),
            });
            try self.buffer(.fail).downloadAtAsync(fail.ptr, 0, N, &self.stream);
            try self.stream.synchronize();
            lane = 0;
            for (ids) |id| for (0..counts[id]) |k| {
                if (fail[lane] != 0) {
                    bad[id] |= @as(u64, 1) << @intCast(k);
                    stats.peels += 1;
                }
                lane += 1;
            };
            var r: usize = 0;
            while (r < points[first].rhs.len) : (r += nn) {
                lane = 0;
                for (ids) |id| for (0..counts[id]) |_| {
                    for (points[id].rhs[r..][0..nn], 0..) |v, row| rhs[row * N + lane] = v;
                    lane += 1;
                };
                try self.upload(.rhs, rhs);
                try self.launch(&self.solve, N, &.{
                    scalar(&nl),   scalar(&built.tab), self.arg(.idx),   self.arg(.fail),
                    self.arg(.lx), self.arg(.ux),      self.arg(.diag),  self.arg(.rhs),
                    self.arg(.y),  self.arg(.out),     scalar(&adjoint),
                });
                try self.buffer(.out).downloadAtAsync(out.ptr, 0, out.len * @sizeOf(f64), &self.stream);
                try self.stream.synchronize();
                lane = 0;
                for (ids) |id| for (0..counts[id]) |k| {
                    if (fail[lane] == 0) {
                        const dst = points[id].out[(bases[id] + k) * points[id].rhs.len + r ..][0..nn];
                        for (dst, 0..) |*v, row| v.* = out[row * N + lane];
                    }
                    lane += 1;
                };
            }
            stats.lanes += N;
            stats.groups += 1;
        }

        /// `group` on the complex lanes: `kernels/frequency.zig`'s complex
        /// bodies over the n x n proxy tape, every plane re then im.
        fn groupComplex(self: *Self, gpa: std.mem.Allocator, points: anytype, bases: []const usize, counts: []const usize, chunks: []const ?CChunk, ids: []const usize, bad: []u64, stats: *Stats) !void {
            var arena = std.heap.ArenaAllocator.init(gpa);
            defer arena.deinit();
            defer self.stream.synchronize() catch {};
            const a = arena.allocator();
            const first = ids[0];
            const head = chunks[first].?;
            const nnz = head.g.len;
            const nn: usize = points[first].fs.nn;
            var N: usize = 0;
            for (ids) |id| N += counts[id];
            const np = ids.len;
            var built = try lk.buildTab(a, head.lu, head.src_col_ptr, head.growth, solver.direct.pivot_tol);
            built.tab.wide = std.math.inf(f64);
            const g = try a.alloc(f64, nnz * np);
            const c = try a.alloc(f64, nnz * np);
            const point = try a.alloc(u32, N);
            const omega = try a.alloc(f64, N);
            // Every right-hand side in one launch: noise's adjoints, sp's ports.
            const n_rhs = points[first].rhs.len / nn;
            const rhs = try a.alloc(f64, n_rhs * nn * N);
            const out = try a.alloc(f64, n_rhs * nn * N);
            const fail = try a.alloc(u8, N);
            const active = try a.alloc(u8, N);
            @memset(active, 1);
            var lane: usize = 0;
            for (ids, 0..) |id, p| {
                const chunk = chunks[id].?;
                for (chunk.g, chunk.c, 0..) |gv, cv, slot| {
                    g[slot * np + p] = gv;
                    c[slot * np + p] = cv;
                }
                for (0..counts[id]) |k| {
                    point[lane] = @intCast(p);
                    omega[lane] = points[id].omegas[bases[id] + k];
                    lane += 1;
                }
            }
            try self.upload(.idx, built.idx);
            try self.upload(.g, g);
            try self.upload(.c, c);
            try self.upload(.point, point);
            try self.upload(.omega, omega);
            try self.upload(.active, active);
            try self.ensure(.a, 2 * nnz * N * @sizeOf(f64));
            try self.ensure(.lx, 2 * @as(usize, built.tab.nnz_l) * N * @sizeOf(f64));
            try self.ensure(.ux, 2 * @as(usize, built.tab.nnz_u) * N * @sizeOf(f64));
            try self.ensure(.diag, nn * N * @sizeOf(f64));
            try self.zero(.w, nn * N * @sizeOf(f64));
            try self.ensure(.y, nn * N * @sizeOf(f64));
            try self.ensure(.out, out.len * @sizeOf(f64));
            try self.ensure(.fail, N);
            try self.zero(.fails, @sizeOf(u32));
            const nl: u64 = N;
            const np64: u64 = np;
            const nz64: u64 = nnz;
            const adjoint: u64 = @intFromBool(points[first].adjoint);
            const scalar = gompute.interface.arg;
            try self.launch(&self.cfill, nnz * N, &.{
                scalar(&nl),     scalar(&np64),    scalar(&nz64), self.arg(.g),
                self.arg(.c),    self.arg(.point), self.arg(.omega), self.arg(.a),
            });
            try self.launch(&self.crefactor, N, &.{
                scalar(&nl),     scalar(&built.tab), self.arg(.idx),  self.arg(.active), self.arg(.a),
                self.arg(.lx),   self.arg(.ux),      self.arg(.diag), self.arg(.w),      self.arg(.fail),
                self.arg(.fails),
            });
            // The solve skips the failed lanes itself: one wait per group.
            lane = 0;
            for (ids) |id| for (0..counts[id]) |_| {
                for (points[id].rhs, 0..) |v, row| rhs[row * N + lane] = v;
                lane += 1;
            };
            try self.upload(.rhs, rhs);
            const nr64: u64 = n_rhs;
            try self.launch(&self.csolve, N, &.{
                scalar(&nl),     scalar(&nr64),  scalar(&built.tab), self.arg(.idx),
                self.arg(.fail), self.arg(.lx),  self.arg(.ux),      self.arg(.diag),
                self.arg(.rhs),  self.arg(.y),   self.arg(.out),     scalar(&adjoint),
            });
            try self.buffer(.fail).downloadAtAsync(fail.ptr, 0, N, &self.stream);
            try self.buffer(.out).downloadAtAsync(out.ptr, 0, out.len * @sizeOf(f64), &self.stream);
            try self.stream.synchronize();
            lane = 0;
            for (ids) |id| for (0..counts[id]) |k| {
                if (fail[lane] != 0) {
                    bad[id] |= @as(u64, 1) << @intCast(k);
                    stats.peels += 1;
                } else {
                    const m = points[id].rhs.len;
                    for (points[id].out[(bases[id] + k) * m ..][0..m], 0..) |*v, row| v.* = out[row * N + lane];
                }
                lane += 1;
            };
            stats.lanes += N;
            stats.groups += 1;
        }
    };
}
