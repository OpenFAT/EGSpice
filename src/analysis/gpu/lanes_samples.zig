//! Sampled periodic device evaluation for harmonic balance. In: the point
//! circuits and the sample times. Out: every (sample, point) lane's
//! residual, charge and both Jacobian planes, each lane on its own storage,
//! with no LU, Newton state or integration history.
//! Cites: none (mhb.zig's sampled operator).
//!
//! Contents:
//! - `pub const Samples = struct (`init`, `deinit`, `evaluate`)`

const std = @import("std");
const assert = std.debug.assert;
const gompute = @import("gompute");
const device_ir = @import("device").abi;
const root = @import("../types.zig");
const devices = @import("devices.zig");
const hdl = @import("hdl.zig");
const lanes = @import("lanes.zig");
const lanes_batch = @import("lanes_batch.zig");
const Circuit = root.Circuit;
const SimState = device_ir.SimState;
const backend = devices.backend;
const Error = lanes.Error;
const Buffer = lanes_batch.Buffer;
const Stream = lanes_batch.Stream;
const LaneBatch = lanes_batch.LaneBatch;
const block = lanes_batch.block;
const arg = gompute.interface.arg;

/// Sampled periodic device evaluation, with independent storage for each
/// (sample, point). It owns no LU, Newton state or integration history.
/// Flattened lane = sample * points + point; input/output vectors and
/// planes have node/slot outermost. The point circuits are read-only and
/// must outlive this object.
pub const Samples = struct {
    gpa: std.mem.Allocator,
    points: usize,
    nt: usize,
    n: usize,
    nnz: usize,
    ground_slot: usize,
    batches: []LaneBatch,
    active: []u8,
    stream: Stream,
    d_x: Buffer,
    d_f: Buffer,
    d_q: Buffer,
    d_g: Buffer,
    d_c: Buffer,
    d_sims: Buffer,
    d_active: Buffer,
    d_limw: Buffer,

    const buffers = .{ "d_x", "d_f", "d_q", "d_g", "d_c", "d_sims", "d_active", "d_limw" };

    /// Packs and uploads every (sample, point) lane of `ckts` at `times`
    /// with step `dt`. The sampled operator does not step analog state, so
    /// a history-dependent circuit (VPI, display tasks, a load check,
    /// stateful charge, a delay, or any state hook) fails with
    /// `error.UnsupportedSampleState` before packing rather than sampling
    /// one point's mutable history in another lane; timer-only source
    /// State is unused, its state hooks null. Fails with
    /// `error.InvalidSamples` on no lanes or a nonfinite time or step,
    /// `error.IncompatibleSamples` across patterns, `Error.NoLaneImage`, or
    /// the driver's error. Free with `deinit`.
    pub fn init(gpa: std.mem.Allocator, ckts: []const *Circuit, times: []const f64, dt: f64) !*Samples {
        if (comptime backend == null) return Error.NoLaneImage;
        if (ckts.len == 0 or times.len == 0 or !std.math.isFinite(dt) or dt <= 0) return error.InvalidSamples;
        const c0 = ckts[0];
        const n: usize = c0.n;
        const nnz: usize = c0.nnz;
        const N = try std.math.mul(usize, ckts.len, times.len);
        // Every sample lane's lim-active word, never set: the sampled
        // operator runs no limit pass.
        const no_lim = try gpa.alloc(u8, N * 4);
        defer gpa.free(no_lim);
        @memset(no_lim, 0);
        if (N > std.math.maxInt(u32)) return error.InvalidSamples;
        for (times) |time| if (!std.math.isFinite(time)) return error.InvalidSamples;
        for (ckts) |c| {
            if (!lanes_batch.samePattern(c, c0)) return error.IncompatibleSamples;
            if (c.vpi != null or c.display != null or c.load_check != null or c.has_state_q or c.minDelay() != null)
                return error.UnsupportedSampleState;
            for (c.batches) |b| {
                const h = b.hooks;
                if (h.gpu_payload == null) return Error.NoLaneImage;
                if (h.status != null or h.update_state != null or h.state_ctl != null or h.commit_state != null or
                    h.advance_iteration != null or h.check_convergence != null or h.predict_first_iterate != null)
                    return error.UnsupportedSampleState;
            }
        }
        if (c0.batches.len == 0) return Error.NoLaneImage;
        const batches = try gpa.alloc(LaneBatch, c0.batches.len);
        errdefer gpa.free(batches);
        var n_batches: usize = 0;
        errdefer for (batches[0..n_batches]) |*b| b.deinit(gpa);
        const ps = try gpa.alloc(device_ir.GpuPayload, N);
        defer gpa.free(ps);
        for (c0.batches, 0..) |b0, bi| {
            const p0 = b0.hooks.gpu_payload.?(b0.ctx);
            for (ps, 0..) |*p, lane_id| {
                const b = ckts[lane_id % ckts.len].batches[bi];
                p.* = b.hooks.gpu_payload.?(b.ctx);
                if (!lanes_batch.sameRows(p.*, p0)) return error.IncompatibleSamples;
            }
            const pk = try lanes_batch.pack(gpa, ps);
            defer pk.deinit(gpa);
            var name: [160]u8 = undefined;
            var eval = try hdl.kernel(backend.?, @ptrCast(b0.hooks.gpu_payload.?), lanes_batch.laneName(&name, p0.kernel, "eval") orelse return Error.NoLaneImage);
            errdefer eval.deinit();
            const Bufs = struct {
                models: Buffer,
                model_of: Buffer,
                instances: Buffer,
                states: Buffer,
                lim: Buffer,
                gath: Buffer,
                rhs_idx: Buffer,
                slots: Buffer,
                limw: Buffer,
            };
            var bs: Bufs = undefined;
            const data = .{
                pk.models,                    std.mem.sliceAsBytes(pk.model_of), pk.instances,                     pk.states,
                std.mem.sliceAsBytes(pk.lim), std.mem.sliceAsBytes(p0.gath),     std.mem.sliceAsBytes(p0.rhs_idx), std.mem.sliceAsBytes(p0.slots),
                no_lim,
            };
            var made: usize = 0;
            errdefer inline for (@typeInfo(Bufs).@"struct".field_names, 0..) |f, j| {
                if (j < made) @field(bs, f).free();
            };
            inline for (@typeInfo(Bufs).@"struct".field_names, 0..) |f, j| {
                @field(bs, f) = try lanes_batch.upload(&eval, data[j]);
                made += 1;
            }
            batches[bi] = .{
                .eval = eval,
                .lim = null,
                .ctl = null,
                .chg = null,
                .count = p0.count,
                .has_lim = false,
                .tape_off = 0,
                .inst_size = if (p0.count == 0) 0 else p0.instances.len / p0.count,
                .state_size = if (p0.count == 0) 0 else p0.states.len / p0.count,
                .d_models = bs.models,
                .d_model_of = bs.model_of,
                .d_instances = bs.instances,
                .d_states = bs.states,
                .d_lim = bs.lim,
                .d_gath = bs.gath,
                .d_rhs_idx = bs.rhs_idx,
                .d_slots = bs.slots,
                .d_limw = bs.limw,
            };
            n_batches += 1;
        }
        const active = try gpa.alloc(u8, N);
        errdefer gpa.free(active);
        @memset(active, 0);
        const self = try gpa.create(Samples);
        errdefer gpa.destroy(self);
        self.* = .{
            .gpa = gpa,
            .points = ckts.len,
            .nt = times.len,
            .n = n,
            .nnz = nnz,
            .ground_slot = c0.diag_slots[0],
            .batches = batches,
            .active = active,
            .stream = try batches[0].eval.createStream(),
            .d_x = undefined,
            .d_f = undefined,
            .d_q = undefined,
            .d_g = undefined,
            .d_c = undefined,
            .d_sims = undefined,
            .d_active = undefined,
            .d_limw = undefined,
        };
        errdefer self.stream.deinit();
        const sizes = .{
            (n + 1) * N * @sizeOf(f64),   (n + 1) * N * @sizeOf(f64),   (n + 1) * N * @sizeOf(f64),
            (nnz + 1) * N * @sizeOf(f64), (nnz + 1) * N * @sizeOf(f64), N * @sizeOf(SimState),
            N,                            @as(usize, 4),
        };
        var made: usize = 0;
        errdefer inline for (buffers, 0..) |f, j| {
            if (j < made) @field(self, f).free();
        };
        inline for (buffers, 0..) |f, j| {
            @field(self, f) = try lanes_batch.zeros(&batches[0].eval, sizes[j]);
            made += 1;
        }
        const sims = try gpa.alloc(SimState, N);
        defer gpa.free(sims);
        for (sims, 0..) |*s, lane_id| s.* = .{
            .t = times[lane_id / ckts.len],
            .dt = dt,
            .kind = .tran,
            .iteration = ckts[lane_id % ckts.len].sim.iteration,
        };
        try self.d_sims.upload(sims.ptr, N * @sizeOf(SimState));
        try self.stream.synchronize();
        if (c0.knobs.gpu_stats != .off) std.debug.print("gpu-samples: {d} points, {d} sample lanes\n", .{ ckts.len, N });
        return self;
    }

    /// Drains the stream and frees everything; the point circuits are
    /// untouched.
    pub fn deinit(self: *Samples) void {
        if (comptime backend == null) return;
        self.stream.synchronize() catch {};
        inline for (buffers) |f| @field(self, f).free();
        for (self.batches) |*b| b.deinit(self.gpa);
        self.gpa.free(self.batches);
        self.gpa.free(self.active);
        self.stream.deinit();
        self.gpa.destroy(self);
    }

    /// Evaluate each active point's samples independently. Inactive points
    /// never run a model kernel and return zero planes. No sample's
    /// Instance/State is copied into the point circuit: the scalar oracle
    /// only evaluates, and samples start from the same fixed model state.
    pub fn evaluate(self: *Samples, x: []const f64, f: []f64, q: []f64, g: []f64, c: []f64, active: []const u8) !void {
        if (comptime backend == null) return Error.NoLaneImage;
        const N = self.points * self.nt;
        std.debug.assert(x.len == self.n * N and f.len == x.len and q.len == x.len);
        std.debug.assert(g.len == self.nnz * N and c.len == g.len);
        std.debug.assert(active.len == self.points);
        for (self.active, 0..) |*on, lane_id| on.* = @intFromBool(active[lane_id % self.points] != 0);
        try self.d_active.upload(self.active.ptr, N);
        try self.d_x.upload(x.ptr, x.len * @sizeOf(f64));
        try self.d_f.fillAsync(0, (self.n + 1) * N * @sizeOf(f64), &self.stream);
        try self.d_q.fillAsync(0, (self.n + 1) * N * @sizeOf(f64), &self.stream);
        try self.d_g.fillAsync(0, (self.nnz + 1) * N * @sizeOf(f64), &self.stream);
        try self.d_c.fillAsync(0, (self.nnz + 1) * N * @sizeOf(f64), &self.stream);
        for (self.batches) |*b| {
            if (b.count == 0) continue;
            var nl: u64 = N;
            var count: u64 = b.count;
            var none: u64 = 0;
            try b.eval.launchOn(&self.stream, gompute.Dim3.linear(@intCast(N), block), .{ .x = block }, 0, &.{
                arg(&nl),              arg(&count),            self.d_sims.argPtr(), self.d_active.argPtr(), self.d_limw.argPtr(),
                self.d_x.argPtr(),     b.d_gath.argPtr(),      b.d_rhs_idx.argPtr(), b.d_slots.argPtr(),     b.d_models.argPtr(),
                b.d_model_of.argPtr(), b.d_instances.argPtr(), self.d_g.argPtr(),    self.d_c.argPtr(),      self.d_f.argPtr(),
                self.d_q.argPtr(),     b.d_lim.argPtr(),       b.d_slots.argPtr(),   arg(&none),             arg(&none),
            });
        }
        try self.d_f.downloadAtAsync(f.ptr, 0, f.len * @sizeOf(f64), &self.stream);
        try self.d_q.downloadAtAsync(q.ptr, 0, q.len * @sizeOf(f64), &self.stream);
        try self.d_g.downloadAtAsync(g.ptr, 0, g.len * @sizeOf(f64), &self.stream);
        try self.d_c.downloadAtAsync(c.ptr, 0, c.len * @sizeOf(f64), &self.stream);
        try self.stream.synchronize();
        for (0..N) |lane_id| {
            if (self.active[lane_id] == 0) continue;
            f[lane_id] += x[lane_id];
            g[self.ground_slot * N + lane_id] += 1;
        }
    }
};

// ---- private ----
