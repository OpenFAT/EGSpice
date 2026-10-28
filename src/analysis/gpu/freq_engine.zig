//! The device engine of the frequency replay. In: launch groups of
//! (point, frequency) lanes that share one pivot tape, their g and c planes
//! and right-hand sides. Out: each lane's solution written into its point's
//! output, and each lane that failed the device refactor marked for its
//! point to peel on the host. Stacked real tapes (`group`) and complex
//! tapes (`groupComplex`); work buffers grow to the largest group and are
//! kept.
//! Cites: `kernels/frequency.zig` (the lane bodies).
//!
//! Contents:
//! - `pub const block: u32`
//! - `pub fn Engine(comptime backend: gompute.Backend) type` (`init`, `deinit`, `ensure`, `buffer`, `upload`,
//!   `zero`, `arg`, `launch`, `group`, `groupComplex`)

const std = @import("std");
const assert = std.debug.assert;
const gompute = @import("gompute");
const solver = @import("solver");
const freq = @import("freq.zig");

const lk = solver.lane_kernels;
const fk = solver.frequency_kernels;
const FS = solver.freq_solve.FreqSolver;
const Chunk = FS.LaneChunk;
const CChunk = FS.ComplexChunk;
const Stats = freq.Stats;

/// The lane kernels' exported block width.
pub const block: u32 = @import("device").abi.gpu_lane_block_size;

pub fn Engine(comptime backend: gompute.Backend) type {
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

        pub fn init() !Self {
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

        pub fn deinit(self: *Self) void {
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

        pub fn ensure(self: *Self, id: Id, bytes: usize) !void {
            const s = &self.buffers[@backingInt(id)];
            const need = @max(bytes, 1);
            if (s.capacity >= need) return;
            const b = try self.fill.alloc(need);
            if (s.value) |*old| old.free();
            s.* = .{ .value = b, .capacity = need };
        }

        pub fn buffer(self: *Self, id: Id) *Buffer {
            return &self.buffers[@backingInt(id)].value.?;
        }

        pub fn upload(self: *Self, id: Id, values: anytype) !void {
            const bytes = std.mem.sliceAsBytes(values);
            try self.ensure(id, bytes.len);
            if (bytes.len != 0) try self.buffer(id).upload(bytes.ptr, bytes.len);
        }

        pub fn zero(self: *Self, id: Id, bytes: usize) !void {
            try self.ensure(id, bytes);
            if (bytes != 0) try self.buffer(id).fillAsync(0, bytes, &self.stream);
        }

        pub fn arg(self: *Self, id: Id) gompute.interface.Arg {
            return self.buffer(id).argPtr();
        }

        pub fn launch(self: *Self, kernel: *Raw, threads: usize, args: []const gompute.interface.Arg) !void {
            try kernel.launchOn(&self.stream, gompute.Dim3.linear(@intCast(threads), block), .{ .x = block }, 0, args);
        }

        /// Uploads `ids`' chunks for one launch: their g and c planes
        /// slot-major with the points innermost, each lane's point and
        /// omega, and an all-on lane mask. Returns the lane count.
        fn uploadGroup(self: *Self, a: std.mem.Allocator, points: anytype, bases: []const usize, counts: []const usize, chunks: anytype, ids: []const usize) !usize {
            const nnz = chunks[ids[0]].?.g.len;
            const np = ids.len;
            var N: usize = 0;
            for (ids) |id| N += counts[id];
            const g = try a.alloc(f64, nnz * np);
            const c = try a.alloc(f64, nnz * np);
            const point = try a.alloc(u32, N);
            const omega = try a.alloc(f64, N);
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
            try self.upload(.g, g);
            try self.upload(.c, c);
            try self.upload(.point, point);
            try self.upload(.omega, omega);
            try self.upload(.active, active);
            return N;
        }

        pub fn group(self: *Self, gpa: std.mem.Allocator, points: anytype, bases: []const usize, counts: []const usize, chunks: []const ?Chunk, ids: []const usize, bad: []u64, stats: *Stats) !void {
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
            const np = ids.len;
            var built = try lk.buildTab(a, head.lu, head.col_ptr, head.growth, solver.direct.pivot_tol);
            // FreqSolver's LaneLu uses the growth monitor, without
            // LaneSolver's additional Newton multiplier threshold.
            built.tab.wide = std.math.inf(f64);
            const pos = try a.alloc(u32, 4 * nnz);
            fk.positions(head.src_col_ptr, pos);
            try self.upload(.idx, built.idx);
            try self.upload(.positions, pos);
            const N = try self.uploadGroup(a, points, bases, counts, chunks, ids);
            const rhs = try a.alloc(f64, nn * N);
            const out = try a.alloc(f64, nn * N);
            const fail = try a.alloc(u8, N);
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
            var lane: usize = 0;
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
        pub fn groupComplex(self: *Self, gpa: std.mem.Allocator, points: anytype, bases: []const usize, counts: []const usize, chunks: []const ?CChunk, ids: []const usize, bad: []u64, stats: *Stats) !void {
            var arena = std.heap.ArenaAllocator.init(gpa);
            defer arena.deinit();
            defer self.stream.synchronize() catch {};
            const a = arena.allocator();
            const first = ids[0];
            const head = chunks[first].?;
            const nnz = head.g.len;
            const nn: usize = points[first].fs.nn;
            const np = ids.len;
            var built = try lk.buildTab(a, head.lu, head.src_col_ptr, head.growth, solver.direct.pivot_tol);
            built.tab.wide = std.math.inf(f64);
            try self.upload(.idx, built.idx);
            const N = try self.uploadGroup(a, points, bases, counts, chunks, ids);
            // Every right-hand side in one launch: noise's adjoints, sp's ports.
            const n_rhs = points[first].rhs.len / nn;
            const rhs = try a.alloc(f64, n_rhs * nn * N);
            const out = try a.alloc(f64, n_rhs * nn * N);
            const fail = try a.alloc(u8, N);
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
                scalar(&nl),  scalar(&np64),    scalar(&nz64),    self.arg(.g),
                self.arg(.c), self.arg(.point), self.arg(.omega), self.arg(.a),
            });
            try self.launch(&self.crefactor, N, &.{
                scalar(&nl),      scalar(&built.tab), self.arg(.idx),  self.arg(.active), self.arg(.a),
                self.arg(.lx),    self.arg(.ux),      self.arg(.diag), self.arg(.w),      self.arg(.fail),
                self.arg(.fails),
            });
            // The solve skips the failed lanes itself: one wait per group.
            var lane: usize = 0;
            for (ids) |id| for (0..counts[id]) |_| {
                for (points[id].rhs, 0..) |v, row| rhs[row * N + lane] = v;
                lane += 1;
            };
            try self.upload(.rhs, rhs);
            const nr64: u64 = n_rhs;
            try self.launch(&self.csolve, N, &.{
                scalar(&nl),     scalar(&nr64), scalar(&built.tab), self.arg(.idx),
                self.arg(.fail), self.arg(.lx), self.arg(.ux),      self.arg(.diag),
                self.arg(.rhs),  self.arg(.y),  self.arg(.out),     scalar(&adjoint),
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
