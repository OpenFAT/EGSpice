//! Retained harmonic and sideband factors of a periodic Krylov operator on
//! the device. In: a `FreqSolver`'s held factors (`factorEach` picks the
//! pivot tapes and identity blocks on the host), or its candidate tapes for
//! a device factorization. Out: the factors resident on the device, and
//! each preconditioner application solving every non-identity harmonic
//! there. A driver failure poisons the factors and hands the caller back
//! its unchanged host result.
//! Cites: none (FreqSolver's held-factor policy).
//!
//! Contents:
//! - `pub const Held = struct` (`init`, `refresh`, `solve`, `deinit`)
//! - `pub fn prepareHeld(held: *?Held, disabled: *bool, ckt: *const root.Circuit, fs: *const FS, count: usize, gpa: std.mem.Allocator) void`
//! - `pub fn DeviceHeld(comptime backend: gompute.Backend) type` (`init`, `initFactor`, `factorEach`, `update`,
//!   `solveDevice`, `identityBuffer`, `deinit`)

const std = @import("std");
const assert = std.debug.assert;
const gompute = @import("gompute");
const artifacts = @import("gompute_kernels");
const solver = @import("solver");
const root = @import("../types.zig");
const freq = @import("freq.zig");
const freq_engine = @import("freq_engine.zig");

const lk = solver.lane_kernels;
const fk = solver.frequency_kernels;
const FS = solver.freq_solve.FreqSolver;
const Chunk = FS.LaneChunk;
const Engine = freq_engine.Engine;
const block = freq_engine.block;
const test_hooks = freq.test_hooks;

/// Retained harmonic/sideband factors for a periodic Krylov operator.
/// `factorEach` still chooses its tapes and identity blocks on the host;
/// each preconditioner application solves all non-identity harmonics on
/// the device. The type erases only backend dispatch, not numerical work.
pub const Held = struct {
    context: *anyopaque,
    refresh_fn: *const fn (*anyopaque, *const FS, usize) anyerror!void,
    solve_fn: *const fn (*anyopaque, []const f64, []f64, bool) bool,
    deinit_fn: *const fn (*anyopaque) void,

    /// Uploads `fs`'s held factors for `count` harmonics. Fails with
    /// `error.GpuBackendUnavailable` without images, `error.NoLaneTape`
    /// when `fs` holds no sparse tape, or the driver's error. Free with
    /// `deinit`.
    pub fn init(gpa: std.mem.Allocator, fs: *const FS, count: usize, stats: bool) !Held {
        if (comptime artifacts.has_cuda) return DeviceHeld(.cuda).create(gpa, fs, count, stats);
        if (comptime artifacts.has_hip) return DeviceHeld(.hip).create(gpa, fs, count, stats);
        return error.GpuBackendUnavailable;
    }

    /// Re-uploads `fs`'s current held factors. After a failure the caller
    /// must `deinit` and use the host factors.
    pub fn refresh(self: *Held, fs: *const FS, count: usize) !void {
        try self.refresh_fn(self.context, fs, count);
    }

    /// False on a driver failure: the caller uses its unchanged host
    /// factorEach result. No partial device result is published.
    pub fn solve(self: *Held, rhs: []const f64, out: []f64, adjoint: bool) bool {
        return self.solve_fn(self.context, rhs, out, adjoint);
    }

    /// Frees the device factors and the context.
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

        /// Loads the kernels on device 0 and uploads `fs`'s held factors
        /// (`update`). Free with `deinit`.
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

        /// Uploads the host-factored held tape of `fs` for `count`
        /// harmonics, its identity harmonics marked. Fails with
        /// `error.GpuBackendUnavailable` once poisoned or
        /// `error.NoLaneTape` without a held tape.
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

        /// One byte per harmonic, nonzero for an identity block, on the
        /// device. Valid until the next `update` or `factorEach`.
        pub fn identityBuffer(self: *Self) *Raw.Buffer {
            return self.engine.buffer(.fail);
        }

        /// Prints the `ESPICE_GPU_STATS` line, then frees everything. The
        /// caller's stream must have finished with `identityBuffer`.
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
