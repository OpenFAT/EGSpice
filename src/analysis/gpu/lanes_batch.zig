//! One device batch of a GPU lane run. In: every lane's `GpuPayload` for one
//! host batch. Out: the batch's lane kernels loaded, its Model rows
//! deduplicated by bytes, its Instance, State and lim rows packed lane-major
//! (instance outermost, lane innermost) and uploaded, and, while N lanes
//! underfill the card, its instance colours (`lane_color.zig`). Also the
//! model-independent lane kernels and the buffer helpers every lane file
//! shares.
//! Cites: lane-batched-simulation.md §3.2 (packed rows); phase4-gpu-lanes.md
//! 06 6.11 (colours).
//!
//! Contents:
//! - `pub const Raw`, `pub const Buffer`, `pub const Stream`, `pub const GraphExec`
//! - `pub const block: u32`
//! - `pub const LaneBatch = struct` (`Color`, `Color.Hubs`, `deinit`)
//! - `pub const Packed = struct` (`deinit`)
//! - `pub const Kernels = struct` (`init`, `deinit`)
//! - `pub fn loadBatch(gpa: Allocator, k0: *Raw, ps: []const device_ir.GpuPayload, payload_fn: *const anyopaque, has_charge: bool, tape_off: u32) !LaneBatch`
//! - `pub fn colorBatch(gpa: Allocator, k0: *Raw, b: *LaneBatch, t: lane_color.Tables, force: bool) !?lane_color.Coloring`
//! - `pub fn samePattern(c: *const Circuit, c0: *const Circuit) bool`
//! - `pub fn sameRows(p: device_ir.GpuPayload, p0: device_ir.GpuPayload) bool`
//! - `pub fn pack(gpa: Allocator, ps: []const device_ir.GpuPayload) !Packed`
//! - `pub fn laneName(buf: []u8, eval_kernel: []const u8, what: []const u8) ?[:0]const u8`
//! - `pub fn multiprocessorCount() ?u32`
//! - `pub fn fp64Ratio() ?u32`
//! - `pub fn staged(k: *Raw, cells: usize, tail: usize, N: usize) !Buffer`
//! - `pub fn upload(k: *Raw, bytes: []const u8) !Buffer`
//! - `pub fn zeros(k: *Raw, bytes: usize) !Buffer`

const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;
const gompute = @import("gompute");
const device_ir = @import("device").abi;
const root = @import("../types.zig");
const devices = @import("devices.zig");
const hdl = @import("hdl.zig");
const lane_color = @import("lane_color.zig");
const lanes = @import("lanes.zig");

const Circuit = root.Circuit;
const Error = lanes.Error;
const backend = devices.backend;

/// `void` in a build with no device images; every use sits behind
/// `comptime backend != null`.
pub const Raw = if (backend) |be| gompute.RawByName(be) else void;
pub const Buffer = if (backend != null) Raw.Buffer else void;
pub const Stream = if (backend != null) Raw.Stream else void;
pub const GraphExec = if (backend != null) Raw.GraphExec else void;
/// The lane kernels' exported block width.
pub const block: u32 = device_ir.gpu_lane_block_size;

/// One device batch over every lane: its shared tapes, its deduplicated
/// Model rows, lane-major Instance/State/lim rows, and its lane kernels.
pub const LaneBatch = struct {
    eval: Raw,
    lim: ?Raw,
    ctl: ?Raw,
    /// `arp_lchg_<model>`: charges into the q plane and the tape.
    chg: ?Raw,
    count: u32,
    has_lim: bool,
    /// This batch's offset into the charge tape.
    tape_off: u32,
    inst_size: usize,
    state_size: usize,
    d_models: Buffer,
    d_model_of: Buffer,
    d_instances: Buffer,
    d_states: Buffer,
    d_lim: Buffer,
    d_gath: Buffer,
    d_rhs_idx: Buffer,
    d_slots: Buffer,
    /// The batch's lim-active word per lane: 1 between the predictor's
    /// limit pass and the end of Newton (`clearLimits`).
    d_limw: Buffer,
    /// `arp_lbrk_<model>` and its per-instance-and-lane source breakpoint
    /// caches, under device step control (`GpuLanes.initSteps`).
    brk: ?Raw = null,
    d_brk: ?Buffer = null,
    /// This batch's colours (`colorBatch`), when it launches by colour.
    color: ?Color = null,

    /// The batch's colour launches on the device.
    pub const Color = struct {
        /// Colour c is `d_order[starts[c]..starts[c + 1]]`; owned by the
        /// lane run's `gpa`.
        starts: []u32,
        d_order: Buffer,
        d_slots: Buffer,
        d_rows: Buffer,
        slot_hubs: Hubs,
        row_hubs: Hubs,

        /// One plane's hub cells and their staging cells on the device.
        pub const Hubs = struct {
            n: u32,
            d_cells: Buffer,
            d_starts: Buffer,
            d_entries: Buffer,
        };

        // ---- private ----

        fn deinit(c: *Color, gpa: Allocator) void {
            gpa.free(c.starts);
            inline for (.{ "d_order", "d_slots", "d_rows" }) |f| @field(c, f).free();
            inline for (.{ &c.slot_hubs, &c.row_hubs }) |h| inline for (.{ "d_cells", "d_starts", "d_entries" }) |f| @field(h, f).free();
        }
    };

    /// Frees every buffer and kernel; `gpa` frees the colour table.
    pub fn deinit(self: *LaneBatch, gpa: Allocator) void {
        inline for (.{ "d_models", "d_model_of", "d_instances", "d_states", "d_lim", "d_gath", "d_rhs_idx", "d_slots", "d_limw" }) |f| @field(self, f).free();
        if (self.d_brk) |*b| b.free();
        if (self.color) |*c| c.deinit(gpa);
        if (self.brk) |*k| k.deinit();
        if (self.lim) |*k| k.deinit();
        if (self.ctl) |*k| k.deinit();
        if (self.chg) |*k| k.deinit();
        self.eval.deinit();
    }
};

/// Lane-major rows of one batch over every lane.
pub const Packed = struct {
    models: []u8,
    model_of: []u32,
    instances: []u8,
    states: []u8,
    lim: []f64,

    /// Frees every row table.
    pub fn deinit(p: Packed, gpa: Allocator) void {
        gpa.free(p.models);
        gpa.free(p.model_of);
        gpa.free(p.instances);
        gpa.free(p.states);
        gpa.free(p.lim);
    }
};

/// The model-independent lane kernels, `arp_lane_<field>`.
pub const Kernels = struct {
    refactor: Raw,
    solve: Raw,
    assemble: Raw,
    gmin: Raw,
    clear_q: Raw,
    update: Raw,
    gates: Raw,
    finalize: Raw,
    settle: Raw,
    set_limw: Raw,
    predict: Raw,
    lte: Raw,
    accept: Raw,
    rebase: Raw,
    gather: Raw,
    peel_dense: Raw,
    pack: Raw,
    unpack: Raw,
    step_begin: Raw,
    newton_init: Raw,
    set_ops: Raw,
    slot_zero: Raw,
    step_end_a: Raw,
    step_end_b: Raw,
    step_stop: Raw,
    ring_put: Raw,
    ring_next: Raw,
    hub_reduce: Raw,
    meas: Raw,

    /// Loads every kernel on this thread's device. Fails with
    /// `Error.NoLaneImage` when the image lacks one, or the driver's error;
    /// nothing stays loaded then.
    pub fn init() !Kernels {
        // Two inline loops over every field, the errdefer's unrolled at each `try`.
        @setEvalBranchQuota(10_000);
        var k: Kernels = undefined;
        var done: usize = 0;
        errdefer inline for (@typeInfo(Kernels).@"struct".field_names, 0..) |f, i| {
            if (i < done) @field(k, f).deinit();
        };
        inline for (@typeInfo(Kernels).@"struct".field_names) |f| {
            @field(k, f) = gompute.rawKernelByName(backend.?, "arp_lane_" ++ f, devices.ordinal) catch |e| return if (e == error.KernelNotFound) Error.NoLaneImage else e;
            done += 1;
        }
        assert(done == @typeInfo(Kernels).@"struct".field_names.len);
        return k;
    }

    /// Unloads every kernel.
    pub fn deinit(k: *Kernels) void {
        inline for (@typeInfo(Kernels).@"struct".field_names) |f| @field(k, f).deinit();
    }
};

/// Loads batch `ps[0]`'s lane kernels and uploads its rows packed over the
/// lanes (`ps[l]` is lane l's payload). `payload_fn` is the batch's
/// `Hooks.gpu_payload`, which finds a runtime `.hdl` image. Fails with
/// `Error.NoLaneImage` when the image lacks the eval kernel, or the driver's
/// or allocator's error; nothing stays allocated then. Caller owns the
/// result (`LaneBatch.deinit`). Asserts that `ps` holds a lane.
pub fn loadBatch(gpa: Allocator, k0: *Raw, ps: []const device_ir.GpuPayload, payload_fn: *const anyopaque, has_charge: bool, tape_off: u32) !LaneBatch {
    assert(ps.len > 0);
    const p0 = ps[0];
    const N = ps.len;
    var name: [160]u8 = undefined;
    const eval_name = laneName(&name, p0.kernel, "eval") orelse return Error.NoLaneImage;
    var eval = hdl.kernel(backend.?, payload_fn, eval_name) catch |e| return if (e == error.KernelNotFound) Error.NoLaneImage else e;
    errdefer eval.deinit();
    var lim: ?Raw = null;
    errdefer if (lim) |*x| x.deinit();
    var ctl: ?Raw = null;
    errdefer if (ctl) |*x| x.deinit();
    var chg: ?Raw = null;
    errdefer if (chg) |*x| x.deinit();
    if (p0.lim_kernel.len > 0) lim = try hdl.kernel(backend.?, payload_fn, laneName(&name, p0.kernel, "lim").?);
    if (p0.ctl_kernel.len > 0) ctl = try hdl.kernel(backend.?, payload_fn, laneName(&name, p0.kernel, "ctl").?);
    if (has_charge) chg = try hdl.kernel(backend.?, payload_fn, laneName(&name, p0.kernel, "chg").?);
    const pk = try pack(gpa, ps);
    defer pk.deinit(gpa);
    // Uploaded one at a time so a failure frees the ones already made.
    const srcs = [_][]const u8{ pk.models, std.mem.sliceAsBytes(pk.model_of), pk.instances, pk.states, std.mem.sliceAsBytes(pk.lim), std.mem.sliceAsBytes(p0.gath), std.mem.sliceAsBytes(p0.rhs_idx), std.mem.sliceAsBytes(p0.slots) };
    var d: [srcs.len + 1]Buffer = undefined;
    var n_d: usize = 0;
    errdefer for (d[0..n_d]) |*x| x.free();
    for (srcs) |s| {
        d[n_d] = try upload(k0, s);
        n_d += 1;
    }
    d[n_d] = try zeros(k0, N * 4);
    n_d += 1;
    assert(n_d == d.len);
    return .{
        .eval = eval,
        .lim = lim,
        .ctl = ctl,
        .chg = chg,
        .count = p0.count,
        .has_lim = p0.lim_x.len > 0,
        .tape_off = tape_off,
        .inst_size = if (p0.count == 0) 0 else p0.instances.len / p0.count,
        .state_size = if (p0.count == 0) 0 else p0.states.len / p0.count,
        .d_models = d[0],
        .d_model_of = d[1],
        .d_instances = d[2],
        .d_states = d[3],
        .d_lim = d[4],
        .d_gath = d[5],
        .d_rhs_idx = d[6],
        .d_slots = d[7],
        .d_limw = d[8],
    };
}

/// Colours batch `b` (`lane_color.color` over `t`) and uploads the colour
/// tables into `b.color`, unless fewer than two instances per colour would
/// launch more than they gain (and not `force`). Returns the colouring for
/// the caller's staging counts and report; caller frees it with `gpa`. On
/// error `b.color` stays null and nothing stays allocated.
pub fn colorBatch(gpa: Allocator, k0: *Raw, b: *LaneBatch, t: lane_color.Tables, force: bool) !?lane_color.Coloring {
    assert(b.color == null);
    assert(t.count == b.count);
    const col = try lane_color.color(gpa, t);
    errdefer col.deinit(gpa);
    if (2 * col.colors() > b.count and !force) {
        col.deinit(gpa);
        return null;
    }
    const starts = try gpa.dupe(u32, col.starts);
    errdefer gpa.free(starts);
    var d_order = try upload(k0, std.mem.sliceAsBytes(col.order));
    errdefer d_order.free();
    var d_slots = try upload(k0, std.mem.sliceAsBytes(col.slots));
    errdefer d_slots.free();
    var d_rows = try upload(k0, std.mem.sliceAsBytes(col.rows));
    errdefer d_rows.free();
    var slot_hubs = try uploadHubs(k0, col.slot_hubs);
    errdefer inline for (.{ "d_cells", "d_starts", "d_entries" }) |f| @field(slot_hubs, f).free();
    const row_hubs = try uploadHubs(k0, col.row_hubs);
    b.color = .{ .starts = starts, .d_order = d_order, .d_slots = d_slots, .d_rows = d_rows, .slot_hubs = slot_hubs, .row_hubs = row_hubs };
    return col;
}

/// Whether `c` shares `c0`'s unknowns, sparse pattern and batch count, so
/// both can be lanes of one device batch.
pub fn samePattern(c: *const Circuit, c0: *const Circuit) bool {
    return c.n == c0.n and c.nnz == c0.nnz and c.batches.len == c0.batches.len and
        std.mem.eql(u32, c.col_ptr, c0.col_ptr) and std.mem.eql(u32, c.row_idx, c0.row_idx);
}

/// Whether batch payload `p` packs beside `p0`: the same kernel, row sizes
/// and scatter tapes (`pack` asserts the tapes).
pub fn sameRows(p: device_ir.GpuPayload, p0: device_ir.GpuPayload) bool {
    return p.count == p0.count and p.n_u == p0.n_u and
        p.instances.len == p0.instances.len and p.states.len == p0.states.len and
        p.lim_x.len == p0.lim_x.len and std.mem.eql(u8, p.kernel, p0.kernel) and
        std.mem.eql(u32, p.gath, p0.gath) and std.mem.eql(u32, p.rhs_idx, p0.rhs_idx) and
        std.mem.eql(u32, p.slots, p0.slots);
}

/// Lane-major rows of batch b over every lane: Model rows deduplicated by
/// bytes, model_of, instances, states and the lim plane. `ps[l]` is lane
/// l's `GpuPayload` for b; the dedup map's keys borrow its frozen Model
/// bytes. Caller owns the result (`Packed.deinit`). Asserts that every lane
/// shares lane 0's count and tapes.
pub fn pack(gpa: Allocator, ps: []const device_ir.GpuPayload) !Packed {
    assert(ps.len > 0);
    const p0 = ps[0];
    const n_lanes = ps.len;
    const count: usize = p0.count;
    var rows_n: u32 = 0;
    for (p0.model_of) |r| rows_n = @max(rows_n, r + 1);
    const ms = if (rows_n == 0) 0 else p0.models.len / rows_n;
    const is = if (count == 0) 0 else p0.instances.len / count;
    const ss = if (count == 0) 0 else p0.states.len / count;
    const kl = if (count == 0) 0 else p0.lim_x.len / count;
    var index: std.StringHashMapUnmanaged(u32) = .empty;
    defer index.deinit(gpa);
    var models: std.ArrayList(u8) = .empty;
    errdefer models.deinit(gpa);
    const model_of = try gpa.alloc(u32, count * n_lanes);
    errdefer gpa.free(model_of);
    const inst = try gpa.alloc(u8, count * n_lanes * is);
    errdefer gpa.free(inst);
    const states = try gpa.alloc(u8, count * n_lanes * ss);
    errdefer gpa.free(states);
    const lim = try gpa.alloc(f64, count * kl * n_lanes);
    errdefer gpa.free(lim);
    for (ps, 0..) |p, l| {
        assert(p.count == p0.count);
        assert(std.mem.eql(u32, p.gath, p0.gath));
        assert(std.mem.eql(u32, p.slots, p0.slots));
        for (0..count) |id| {
            const row = p.models[@as(usize, p.model_of[id]) * ms ..][0..ms];
            const gop = try index.getOrPut(gpa, row);
            if (!gop.found_existing) {
                gop.value_ptr.* = @intCast(models.items.len / @max(ms, 1));
                try models.appendSlice(gpa, row);
            }
            model_of[id * n_lanes + l] = gop.value_ptr.*;
            @memcpy(inst[(id * n_lanes + l) * is ..][0..is], p.instances[id * is ..][0..is]);
            if (ss != 0) @memcpy(states[(id * n_lanes + l) * ss ..][0..ss], p.states[id * ss ..][0..ss]);
            for (0..kl) |a| lim[(id * kl + a) * n_lanes + l] = p.lim_x[id * kl + a];
        }
    }
    return .{ .models = try models.toOwnedSlice(gpa), .model_of = model_of, .instances = inst, .states = states, .lim = lim };
}

/// `arp_l<what>_<model>` for the batch whose eval kernel is
/// `arp_eval_<model>`, into `buf`; null when the name has another shape or
/// does not fit.
pub fn laneName(buf: []u8, eval_kernel: []const u8, what: []const u8) ?[:0]const u8 {
    const prefix = "arp_eval_";
    if (!std.mem.startsWith(u8, eval_kernel, prefix)) return null;
    return std.fmt.bufPrintSentinel(buf, "arp_l{s}_{s}", .{ what, eval_kernel[prefix.len..] }, 0) catch null;
}

/// The device's streaming multiprocessors (compute units on HIP 6.0+), or
/// null (older HIP, no driver). Starts the driver.
pub fn multiprocessorCount() ?u32 {
    if (comptime @import("builtin").cpu.arch.isWasm()) return null;
    var compute = gompute.runtime.dynamic.Compute.init(null) catch return null;
    defer compute.deinit();
    return compute.multiprocessorCount() catch null;
}

/// The device's FP64:FP32 ratio, or null (HIP, no driver). Starts the
/// driver.
pub fn fp64Ratio() ?u32 {
    if (comptime @import("builtin").cpu.arch.isWasm()) return null;
    var compute = gompute.runtime.dynamic.Compute.init(null) catch return null;
    defer compute.deinit();
    return compute.fp64Ratio() catch null;
}

/// `cells` zeroed rows of N lanes followed by `tail` rows of -0.0
/// (`lane_color.zig`'s staging cells). Caller frees the buffer.
pub fn staged(k: *Raw, cells: usize, tail: usize, N: usize) !Buffer {
    if (tail == 0) return zeros(k, cells * N * @sizeOf(f64));
    const host = try std.heap.page_allocator.alloc(f64, (cells + tail) * N);
    defer std.heap.page_allocator.free(host);
    @memset(host[0 .. cells * N], 0);
    @memset(host[cells * N ..], -0.0);
    return upload(k, std.mem.sliceAsBytes(host));
}

/// A device copy of `bytes`, at least one byte long so an empty table still
/// has a kernel-argument pointer. Caller frees the buffer.
pub fn upload(k: *Raw, bytes: []const u8) !Buffer {
    var buf = try k.alloc(@max(bytes.len, 1));
    errdefer buf.free();
    if (bytes.len > 0) try buf.upload(bytes.ptr, bytes.len);
    return buf;
}

/// `bytes` zeroed device bytes (at least one). Caller frees the buffer.
pub fn zeros(k: *Raw, bytes: usize) !Buffer {
    var buf = try k.alloc(@max(bytes, 1));
    errdefer buf.free();
    const z = try std.heap.page_allocator.alloc(u8, @max(bytes, 1));
    defer std.heap.page_allocator.free(z);
    @memset(z, 0);
    try buf.upload(z.ptr, z.len);
    return buf;
}

// ---- private ----

fn uploadHubs(k: *Raw, h: lane_color.Coloring.Hubs) !LaneBatch.Color.Hubs {
    var cells = try upload(k, std.mem.sliceAsBytes(h.cells));
    errdefer cells.free();
    var starts = try upload(k, std.mem.sliceAsBytes(h.starts));
    errdefer starts.free();
    const entries = try upload(k, std.mem.sliceAsBytes(h.entries));
    return .{ .n = @intCast(h.cells.len), .d_cells = cells, .d_starts = starts, .d_entries = entries };
}
