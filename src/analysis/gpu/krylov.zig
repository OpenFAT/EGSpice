//! GPU-resident, independently masked GMRES systems. The host dispatches
//! operator requests and reads two counts per round; vectors, basis, norms,
//! rotations and iteration decisions remain on the device. A callback must
//! write only points selected by its u8 mask and preserve inactive outputs.
const std = @import("std");
const gompute = @import("gompute");
const lanes = @import("solver").gmres_lanes;
const Gmres = @import("solver").gmres.Gmres;
const block = @import("device").abi.gpu_lane_block_size;

pub const Stats = struct {
    rounds: usize = 0,
    matvec_rounds: usize = 0,
    precond_rounds: usize = 0,
};

pub fn Engine(comptime backend: gompute.Backend) type {
    return struct {
        const Self = @This();
        pub const Raw = gompute.RawByName(backend);
        pub const Buffer = Raw.Buffer;
        const Id = enum { states, work, input, output, matvec_mask, precond_mask, counts };
        const Storage = struct { buffer: ?Buffer = null, capacity: usize = 0 };

        allocator: std.mem.Allocator,
        start_kernel: Raw,
        advance_kernel: Raw,
        stream: Raw.Stream,
        buffers: [@typeInfo(Id).@"enum".field_names.len]Storage = @splat(.{}),
        count: usize = 0,
        completed: bool = false,

        pub fn init(allocator: std.mem.Allocator) !Self {
            var start_kernel = try gompute.rawKernelByName(backend, "arp_gmres_start", 0);
            errdefer start_kernel.deinit();
            var advance_kernel = try gompute.rawKernelByName(backend, "arp_gmres_advance", 0);
            errdefer advance_kernel.deinit();
            const stream = try start_kernel.createStream();
            return .{ .allocator = allocator, .start_kernel = start_kernel, .advance_kernel = advance_kernel, .stream = stream };
        }

        pub fn deinit(self: *Self) void {
            self.stream.synchronize() catch {};
            for (&self.buffers) |*storage| if (storage.buffer) |*b| b.free();
            self.stream.deinit();
            self.start_kernel.deinit();
            self.advance_kernel.deinit();
            self.* = undefined;
        }

        fn ensure(self: *Self, id: Id, bytes: usize) !void {
            const storage = &self.buffers[@backingInt(id)];
            const required = @max(bytes, 1);
            if (storage.capacity >= required) return;
            const allocation = try self.start_kernel.alloc(required);
            if (storage.buffer) |*old| old.free();
            storage.* = .{ .buffer = allocation, .capacity = required };
        }

        fn buffer(self: *Self, id: Id) *Buffer {
            return &self.buffers[@backingInt(id)].buffer.?;
        }

        fn arg(self: *Self, id: Id) gompute.interface.Arg {
            return self.buffer(id).argPtr();
        }

        fn counts(self: *Self) ![2]u32 {
            var pending: [2]u32 = undefined;
            errdefer self.stream.synchronize() catch {};
            try self.buffer(.counts).downloadAtAsync(&pending, 0, @sizeOf(@TypeOf(pending)), &self.stream);
            try self.stream.synchronize();
            return pending;
        }

        /// b and x contain n*count f64s, element-major with points
        /// innermost. x is the initial guess and receives each point's best
        /// iterate. Operator failures propagate; no result is then valid.
        /// Call downloadResults after success to inspect convergence.
        pub fn solve(self: *Self, op: anytype, b: *Buffer, x: *Buffer, count: usize, n: u32, m: u32, tolerance: f64, max_restarts: u32) !Stats {
            self.completed = false;
            self.count = count;
            if (count == 0) {
                self.completed = true;
                return .{};
            }
            if (count > std.math.maxInt(u32)) return error.TooManyKrylovPoints;
            const layout = try lanes.Layout.init(n, m);
            const has_precond = comptime @hasDecl(std.meta.Child(@TypeOf(op)), "precond");
            const shape: lanes.Shape = .{
                .points = count,
                .n = n,
                .m = m,
                .max_restarts = max_restarts,
                .preconditioned = @intFromBool(has_precond),
                .tolerance = tolerance,
            };
            const entries = try std.math.mul(usize, n, count);
            const vector_bytes = try std.math.mul(usize, entries, @sizeOf(f64));
            if (b.bytes < vector_bytes or x.bytes < vector_bytes) return error.InvalidKrylovShape;
            const work_rows = std.math.cast(usize, layout.rows) orelse return error.Overflow;
            const work_entries = try std.math.mul(usize, work_rows, count);
            const work_bytes = try std.math.mul(usize, work_entries, @sizeOf(f64));
            const state_bytes = try std.math.mul(usize, count, @sizeOf(lanes.State));
            try self.ensure(.states, state_bytes);
            try self.ensure(.work, work_bytes);
            try self.ensure(.input, vector_bytes);
            try self.ensure(.output, vector_bytes);
            try self.ensure(.matvec_mask, count);
            try self.ensure(.precond_mask, count);
            try self.ensure(.counts, 2 * @sizeOf(u32));
            errdefer self.stream.synchronize() catch {};
            try self.buffer(.counts).fillAsync(0, 2 * @sizeOf(u32), &self.stream);
            const scalar = gompute.interface.arg;
            try self.start_kernel.launchOn(&self.stream, gompute.Dim3.linear(@intCast(count), block), .{ .x = block }, 0, &.{
                scalar(&shape),         self.arg(.states),       b.argPtr(),        x.argPtr(), self.arg(.input),
                self.arg(.matvec_mask), self.arg(.precond_mask), self.arg(.counts),
            });
            var pending = try self.counts();
            var stats: Stats = .{};
            while (pending[0] != 0 or pending[1] != 0) {
                if (pending[0] != 0) {
                    try op.matvec(self.buffer(.input), self.buffer(.output), self.buffer(.matvec_mask), &self.stream);
                    stats.matvec_rounds += 1;
                }
                if (comptime has_precond) {
                    if (pending[1] != 0) {
                        try op.precond(self.buffer(.input), self.buffer(.output), self.buffer(.precond_mask), &self.stream);
                        stats.precond_rounds += 1;
                    }
                } else std.debug.assert(pending[1] == 0);
                try self.buffer(.counts).fillAsync(0, 2 * @sizeOf(u32), &self.stream);
                try self.advance_kernel.launchOn(&self.stream, gompute.Dim3.linear(@intCast(count), block), .{ .x = block }, 0, &.{
                    scalar(&shape),   scalar(&layout),   self.arg(.states),      b.argPtr(),              x.argPtr(),        self.arg(.work),
                    self.arg(.input), self.arg(.output), self.arg(.matvec_mask), self.arg(.precond_mask), self.arg(.counts),
                });
                pending = try self.counts();
                stats.rounds += 1;
            }
            self.completed = true;
            return stats;
        }

        pub fn downloadResults(self: *Self, allocator: std.mem.Allocator) ![]Gmres.SolveResult {
            if (!self.completed) return error.KrylovResultUnavailable;
            const results = try allocator.alloc(Gmres.SolveResult, self.count);
            errdefer allocator.free(results);
            if (self.count == 0) return results;
            const states = try self.allocator.alloc(lanes.State, self.count);
            defer self.allocator.free(states);
            // This is a synchronous download: temporary host storage must
            // remain alive even if the backend reports a transfer failure.
            try self.buffer(.states).download(states.ptr, states.len * @sizeOf(lanes.State));
            for (states, results) |state, *result| result.* = .{
                .iterations = state.iterations,
                .residual = state.residual,
                .converged = state.converged != 0,
            };
            return results;
        }
    };
}
