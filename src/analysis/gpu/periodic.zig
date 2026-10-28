//! The resident sampled HB Jacobian and harmonic preconditioner. In: the
//! sampled problem (`Setup`), each Newton step's sampled g and c planes and
//! each point's held factors. Out: the Jacobian product and the harmonic
//! preconditioner applied on the device, as masked callbacks on a caller's
//! stream for device GMRES (`krylov.zig`), or as host-facing calls for one
//! point (`Operator`). A point's vectors and matrices are interleaved with
//! points innermost; masks never mix convergence or arithmetic between
//! points. The tests sit in periodic_tests.zig.
//! Cites: none (mhb.zig's sampled operator).
//!
//! Contents:
//! - `pub const Setup = struct`
//! - `pub const Operator = struct` (`init`, `refresh`, `matvec`, `precond`, `solve`, `deinit`)
//! - `pub fn Engine(comptime backend: gompute.Backend) type` (`Raw`, `Buffer`, `Stream`, `init`, `deinit`, `refresh`,
//!   `factor`, `matvec`, `precond`, `solveHost`, and for periodic_tests.zig `applyHost`, `buffer`, `upload`)

const std = @import("std");
const assert = std.debug.assert;
const gompute = @import("gompute");
const artifacts = @import("gompute_kernels");
const solver = @import("solver");

const pk = solver.periodic_kernels;
const FS = solver.freq_solve.FreqSolver;
const SolveResult = solver.gmres.Gmres.SolveResult;
const block = @import("device").abi.gpu_lane_block_size;

/// The sampled HB problem shared by every point: `n` unknowns, `nf`
/// (odd) frequency values per unknown, `nt` time samples, the pattern's
/// CSC and the `nt * nf` inverse and forward transforms. Borrowed by
/// `Engine.init`, which copies it to the device.
pub const Setup = struct {
    points: usize,
    n: u32,
    nf: u32,
    nt: u32,
    charge: bool,
    col_ptr: []const u32,
    row_idx: []const u32,
    inv: []const f64,
    fwd: []const f64,
    /// Harmonic-major, points innermost, including DC.
    omegas: []const f64,
};

/// Scalar-analysis adapter. The resident engine below also exposes masked
/// device-buffer callbacks for point-wise GPU Krylov.
pub const Operator = struct {
    context: *anyopaque,
    refresh_fn: *const fn (*anyopaque, []const f64, []const f64, *const FS) anyerror!void,
    matvec_fn: *const fn (*anyopaque, []const f64, []f64) bool,
    precond_fn: *const fn (*anyopaque, []const f64, []f64) bool,
    solve_fn: *const fn (*anyopaque, []const f64, []f64, u32, f64, u32) ?SolveResult,
    deinit_fn: *const fn (*anyopaque) void,

    /// One point's engine on the first backend this build carries. Asserts
    /// `setup.points == 1`. Fails with `error.GpuBackendUnavailable`
    /// without images, or the driver's error. Free with `deinit`.
    pub fn init(gpa: std.mem.Allocator, setup: Setup, stats: bool) !Operator {
        std.debug.assert(setup.points == 1);
        if (comptime artifacts.has_cuda) return Adapter(.cuda).create(gpa, setup, stats);
        if (comptime artifacts.has_hip) return Adapter(.hip).create(gpa, setup, stats);
        return error.GpuBackendUnavailable;
    }

    /// `Engine.refresh` with this Newton step's sampled planes and `fs`'s
    /// held factors. Fails with `error.GpuBackendUnavailable` once a
    /// driver fault poisoned the operator.
    pub fn refresh(self: *Operator, g: []const f64, c: []const f64, fs: *const FS) !void {
        try self.refresh_fn(self.context, g, c, fs);
    }

    /// The Jacobian product on the device. False on a driver fault (and
    /// every call after it): the caller computes `out` on the host.
    pub fn matvec(self: *Operator, input: []const f64, out: []f64) bool {
        return self.matvec_fn(self.context, input, out);
    }

    /// The harmonic preconditioner on the device; false as `matvec`.
    /// `input` and `out` may alias.
    pub fn precond(self: *Operator, input: []const f64, out: []f64) bool {
        return self.precond_fn(self.context, input, out);
    }

    /// The whole preconditioned GMRES solve on the device, from guess `x`.
    /// Null on a driver fault: `x` is unchanged and the caller solves on
    /// the host.
    pub fn solve(self: *Operator, b: []const f64, x: []f64, m: u32, tolerance: f64, max_restarts: u32) ?SolveResult {
        return self.solve_fn(self.context, b, x, m, tolerance, max_restarts);
    }

    /// Frees the engine and the context.
    pub fn deinit(self: *Operator) void {
        self.deinit_fn(self.context);
        self.* = undefined;
    }
};

/// Callbacks accept a stream owned by their caller, so Krylov can enqueue
/// them without downloading its vectors. Calls on one engine are ordered.
pub fn Engine(comptime backend: gompute.Backend) type {
    return struct {
        const Self = @This();
        pub const Raw = gompute.RawByName(backend);
        pub const Buffer = Raw.Buffer;
        pub const Stream = Raw.Stream;
        const Factor = @import("freq.zig").DeviceHeld(backend);
        const Krylov = @import("krylov.zig").Engine(backend);
        /// `arp_periodic_<tag>`, loaded in tag order into `kernels`.
        const Kernel = enum { inverse, product, forward, pack, unpack, copy };
        const kernel_names = @typeInfo(Kernel).@"enum".field_names;
        const Id = enum { row_ptr, columns, slots, inv, fwd, omega, g, c, time, gv, cv, input, output, mask, rhs, solution };
        const Storage = struct { value: ?Buffer = null, capacity: usize = 0 };

        gpa: std.mem.Allocator,
        shape: pk.Shape,
        nnz: usize,
        kernels: [kernel_names.len]Raw,
        stream: Stream,
        buffers: [@typeInfo(Id).@"enum".field_names.len]Storage = @splat(.{}),
        factors: []?Factor,
        host_output: []f64,
        stats: bool,
        matvecs: usize = 0,
        preconditions: usize = 0,
        krylov: ?Krylov = null,
        krylov_solves: usize = 0,
        krylov_rounds: usize = 0,
        factor_rounds: usize = 0,
        factor_lanes: usize = 0,

        /// Loads the kernels on device 0 and uploads `setup`'s pattern,
        /// transforms and omegas. Asserts the shape is consistent (odd
        /// `nf`, transform and omega lengths). Free with `deinit`.
        pub fn init(a: std.mem.Allocator, setup: Setup, stats: bool) !Self {
            std.debug.assert(setup.points > 0 and setup.n > 0 and setup.nf > 0 and setup.nf % 2 == 1 and setup.nt > 0);
            std.debug.assert(setup.inv.len == @as(usize, setup.nt) * setup.nf and setup.fwd.len == setup.inv.len);
            std.debug.assert(setup.omegas.len == ((setup.nf + 1) / 2) * setup.points);
            var kernels: [kernel_names.len]Raw = undefined;
            var loaded: usize = 0;
            errdefer for (kernels[0..loaded]) |*k| k.deinit();
            inline for (kernel_names) |name| {
                kernels[loaded] = try gompute.rawKernelByName(backend, "arp_periodic_" ++ name, 0);
                loaded += 1;
            }
            var stream = try kernels[0].createStream();
            errdefer stream.deinit();
            const factors = try a.alloc(?Factor, setup.points);
            errdefer a.free(factors);
            @memset(factors, null);
            const host_output = try a.alloc(f64, @as(usize, setup.n) * setup.nf * setup.points);
            errdefer a.free(host_output);
            var self: Self = .{
                .gpa = a,
                .shape = .{ .points = setup.points, .n = setup.n, .nf = setup.nf, .nt = setup.nt, .charge = @intFromBool(setup.charge) },
                .nnz = setup.col_ptr[setup.n],
                .kernels = kernels,
                .stream = stream,
                .factors = factors,
                .host_output = host_output,
                .stats = stats,
            };
            errdefer {
                self.stream.synchronize() catch {};
                for (&self.buffers) |*s| if (s.value) |*v| v.free();
            }
            const rows = try pk.Rows.init(a, setup.col_ptr, setup.row_idx);
            defer rows.deinit(a);
            try self.upload(.row_ptr, rows.ptr);
            try self.upload(.columns, rows.columns);
            try self.upload(.slots, rows.slots);
            try self.upload(.inv, setup.inv);
            try self.upload(.fwd, setup.fwd);
            try self.upload(.omega, setup.omegas);
            const samples = @as(usize, setup.n) * setup.nt * setup.points;
            inline for (.{ .time, .gv, .cv }) |id| try self.ensure(id, samples * @sizeOf(f64));
            const values = @as(usize, setup.n) * setup.nf * setup.points;
            inline for (.{ .input, .output }) |id| try self.ensure(id, values * @sizeOf(f64));
            const harmonics = @as(usize, setup.n) * (setup.nf + 1);
            inline for (.{ .rhs, .solution }) |id| try self.ensure(id, harmonics * @sizeOf(f64));
            const mask = try a.alloc(u8, setup.points);
            defer a.free(mask);
            @memset(mask, 1);
            try self.upload(.mask, mask);
            return self;
        }

        /// Drains the stream, prints the `ESPICE_GPU_STATS` lines, then
        /// frees everything.
        pub fn deinit(self: *Self) void {
            self.stream.synchronize() catch {};
            if (self.stats and self.matvecs != 0)
                std.debug.print("gpu-periodic: {d} points, {d} device matvecs, {d} device preconditions\n", .{ self.shape.points, self.matvecs, self.preconditions });
            if (self.stats and self.krylov_solves != 0)
                std.debug.print("gpu-krylov: {d} points, {d} solves, {d} rounds\n", .{ self.shape.points, self.krylov_solves, self.krylov_rounds });
            if (self.stats and self.factor_lanes != 0)
                std.debug.print("gpu-periodic-factor: {d} points, {d} rounds, {d} harmonic lanes\n", .{ self.shape.points, self.factor_rounds, self.factor_lanes });
            if (self.krylov) |*k| k.deinit();
            for (self.factors) |*f| if (f.*) |*held| held.deinit();
            self.gpa.free(self.factors);
            self.gpa.free(self.host_output);
            for (&self.buffers) |*s| if (s.value) |*v| v.free();
            self.stream.deinit();
            for (&self.kernels) |*k| k.deinit();
            self.* = undefined;
        }

        /// Sample planes are (slot, sample, point); factors are one
        /// independent host factorEach result per point. The caller must
        /// complete external stream work before refreshing.
        pub fn refresh(self: *Self, g: []const f64, c: []const f64, fs: []const *const FS) !void {
            std.debug.assert(g.len == self.nnz * self.shape.nt * self.shape.points and c.len == g.len);
            std.debug.assert(fs.len == self.factors.len);
            try self.upload(.g, g);
            try self.upload(.c, c);
            for (fs, self.factors) |f, *held| {
                if (f.heldTape() == null) {
                    if (f.held_id.items.len != (self.shape.nf + 1) / 2)
                        return error.UnsupportedPeriodicPreconditioner;
                    if (held.*) |*h| h.deinit();
                    held.* = null;
                } else if (held.*) |*h| {
                    try h.update(f, (self.shape.nf + 1) / 2);
                } else {
                    held.* = try Factor.init(self.gpa, f, (self.shape.nf + 1) / 2, false);
                }
            }
        }

        /// Factor active points' harmonic blocks directly on the device.
        /// FreqSolver chooses each point's pivot tape and keeps its own
        /// identity decisions; no numeric harmonic factors are downloaded.
        /// A caller falling back to host solveEach calls ensureHeldOnHost
        /// to retain the completed candidate policy and pivot history.
        pub fn factor(self: *Self, g: []const f64, c: []const f64, fs: []const *FS, omegas: []const f64, active: []const u8) !void {
            std.debug.assert(g.len == self.nnz * self.shape.nt * self.shape.points and c.len == g.len);
            std.debug.assert(fs.len == self.factors.len and active.len == fs.len);
            std.debug.assert(omegas.len == (self.shape.nf + 1) / 2);
            try self.upload(.g, g);
            try self.upload(.c, c);
            for (fs, self.factors, active) |f, *held, on| {
                if (on == 0) continue;
                const before = if (held.*) |h| h.device_refactors else 0;
                const usable = if (held.*) |*h|
                    try h.factorEach(f, omegas)
                else blk: {
                    held.* = try Factor.initFactor(self.gpa, f, omegas, false);
                    break :blk f.held_id.items.len != omegas.len;
                };
                self.factor_lanes += held.*.?.device_refactors - before;
                if (!usable) {
                    if (held.*) |*h| h.deinit();
                    held.* = null;
                }
            }
            self.factor_rounds += 1;
        }

        /// Queues the sampled Jacobian product on `stream` for the points
        /// `mask` selects (one byte each); other points' outputs are left
        /// as they were. Unwaited.
        pub fn matvec(self: *Self, input: *Buffer, output: *Buffer, mask: *Buffer, stream: *Stream) !void {
            const scalar = gompute.interface.arg;
            const samples = @as(usize, self.shape.n) * self.shape.nt * @as(usize, @intCast(self.shape.points));
            const values = @as(usize, self.shape.n) * self.shape.nf * @as(usize, @intCast(self.shape.points));
            try self.launch(.inverse, samples, stream, &.{
                scalar(&self.shape), mask.argPtr(), self.arg(.inv), input.argPtr(), self.arg(.time),
            });
            try self.launch(.product, samples, stream, &.{
                scalar(&self.shape), mask.argPtr(), self.arg(.row_ptr), self.arg(.columns), self.arg(.slots),
                self.arg(.g),        self.arg(.c),  self.arg(.time),    self.arg(.gv),      self.arg(.cv),
            });
            try self.launch(.forward, values, stream, &.{
                scalar(&self.shape), mask.argPtr(), self.arg(.fwd),  self.arg(.omega),
                self.arg(.gv),       self.arg(.cv), output.argPtr(),
            });
            self.matvecs += 1;
        }

        /// Queues the harmonic preconditioner on `stream` for the points
        /// `mask` selects: each point's device factors, or a copy for a
        /// point without them. Unwaited.
        pub fn precond(self: *Self, input: *Buffer, output: *Buffer, mask: *Buffer, stream: *Stream) !void {
            const scalar = gompute.interface.arg;
            const values = @as(usize, self.shape.n) * self.shape.nf;
            const harmonics = @as(usize, self.shape.n) * (self.shape.nf + 1);
            for (self.factors, 0..) |*held, p| {
                const point: u64 = p;
                if (held.*) |*f| {
                    try self.launch(.pack, harmonics, stream, &.{
                        scalar(&self.shape), scalar(&point), mask.argPtr(), input.argPtr(), self.arg(.rhs),
                    });
                    try f.solveDevice(self.buffer(.rhs), self.buffer(.solution), false, stream);
                    try self.launch(.unpack, values, stream, &.{
                        scalar(&self.shape), scalar(&point),      mask.argPtr(),   f.identityBuffer().argPtr(),
                        self.arg(.rhs),      self.arg(.solution), output.argPtr(),
                    });
                } else {
                    try self.launch(.copy, values, stream, &.{
                        scalar(&self.shape), scalar(&point), mask.argPtr(), input.argPtr(), output.argPtr(),
                    });
                }
            }
            self.preconditions += 1;
        }

        pub fn applyHost(self: *Self, input: []const f64, out: []f64, precondition: bool) !void {
            std.debug.assert(input.len == @as(usize, self.shape.n) * self.shape.nf * self.shape.points and out.len == input.len);
            try self.upload(.input, input);
            if (precondition)
                try self.precond(self.buffer(.input), self.buffer(.output), self.buffer(.mask), &self.stream)
            else
                try self.matvec(self.buffer(.input), self.buffer(.output), self.buffer(.mask), &self.stream);
            try self.buffer(.output).downloadAtAsync(self.host_output.ptr, 0, out.len * @sizeOf(f64), &self.stream);
            try self.stream.synchronize();
            @memcpy(out, self.host_output);
        }

        /// Host boundary for complete independently masked solves. The
        /// original x is retained on any driver/operator failure.
        pub fn solveHost(self: *Self, b: []const f64, x: []f64, m: u32, tolerance: f64, max_restarts: u32, results: []SolveResult) !void {
            std.debug.assert(b.len == self.host_output.len and x.len == b.len and results.len == self.shape.points);
            if (self.krylov == null) self.krylov = try Krylov.init(self.gpa);
            try self.upload(.input, b);
            try self.upload(.output, x);
            const stats = try self.krylov.?.solve(self, self.buffer(.input), self.buffer(.output), @intCast(self.shape.points), self.shape.n * self.shape.nf, m, tolerance, max_restarts);
            const got = try self.krylov.?.downloadResults(self.gpa);
            defer self.gpa.free(got);
            try self.buffer(.output).downloadAtAsync(self.host_output.ptr, 0, self.host_output.len * @sizeOf(f64), &self.stream);
            try self.stream.synchronize();
            @memcpy(x, self.host_output);
            @memcpy(results, got);
            self.krylov_solves += 1;
            self.krylov_rounds += stats.rounds;
        }

        fn ensure(self: *Self, id: Id, size: usize) !void {
            const need = @max(size, 8);
            const entry = &self.buffers[@backingInt(id)];
            if (entry.capacity >= need) return;
            const value = try self.kernels[0].alloc(need);
            if (entry.value) |*old| old.free();
            entry.* = .{ .value = value, .capacity = need };
        }

        pub fn buffer(self: *Self, id: Id) *Buffer {
            return &self.buffers[@backingInt(id)].value.?;
        }

        pub fn upload(self: *Self, id: Id, data: anytype) !void {
            const bytes = std.mem.sliceAsBytes(data);
            try self.ensure(id, bytes.len);
            if (bytes.len != 0) try self.buffer(id).upload(bytes.ptr, bytes.len);
        }

        fn arg(self: *Self, id: Id) gompute.interface.Arg {
            return self.buffer(id).argPtr();
        }

        fn launch(self: *Self, kernel: Kernel, threads: usize, stream: *Stream, args: []const gompute.interface.Arg) !void {
            try self.kernels[@backingInt(kernel)].launchOn(stream, gompute.Dim3.linear(@intCast(threads), block), .{ .x = block }, 0, args);
        }
    };
}

// ---- private ----

fn Adapter(comptime backend: gompute.Backend) type {
    return struct {
        const Self = @This();
        engine: Engine(backend),
        poisoned: bool = false,

        fn create(a: std.mem.Allocator, setup: Setup, stats: bool) !Operator {
            const self = try a.create(Self);
            errdefer a.destroy(self);
            self.* = .{ .engine = try Engine(backend).init(a, setup, stats) };
            return .{ .context = self, .refresh_fn = refresh, .matvec_fn = matvec, .precond_fn = precond, .solve_fn = solve, .deinit_fn = destroy };
        }

        fn refresh(context: *anyopaque, g: []const f64, c: []const f64, fs: *const FS) !void {
            const self: *Self = @ptrCast(@alignCast(context));
            if (self.poisoned) return error.GpuBackendUnavailable;
            try self.engine.refresh(g, c, &.{fs});
        }

        fn matvec(context: *anyopaque, input: []const f64, out: []f64) bool {
            return apply(context, input, out, false);
        }

        fn precond(context: *anyopaque, input: []const f64, out: []f64) bool {
            return apply(context, input, out, true);
        }

        fn apply(context: *anyopaque, input: []const f64, out: []f64, precondition: bool) bool {
            const self: *Self = @ptrCast(@alignCast(context));
            if (self.poisoned) return false;
            self.engine.applyHost(input, out, precondition) catch {
                self.engine.stream.synchronize() catch {};
                self.poisoned = true;
                return false;
            };
            return true;
        }

        fn solve(context: *anyopaque, b: []const f64, x: []f64, m: u32, tolerance: f64, max_restarts: u32) ?SolveResult {
            const self: *Self = @ptrCast(@alignCast(context));
            if (self.poisoned) return null;
            var result: [1]SolveResult = undefined;
            self.engine.solveHost(b, x, m, tolerance, max_restarts, &result) catch {
                self.poisoned = true;
                return null;
            };
            return result[0];
        }

        fn destroy(context: *anyopaque) void {
            const self: *Self = @ptrCast(@alignCast(context));
            const a = self.engine.gpa;
            self.engine.deinit();
            a.destroy(self);
        }
    };
}

test {
    _ = @import("periodic_tests.zig");
}
