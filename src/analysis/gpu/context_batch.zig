//! One resident device batch. In: a host batch's GPU payload. Out: its
//! kernels loaded and its models, instances, tapes, lim plane and states
//! uploaded, resident for the query, or a verdict that it stays on the host
//! (no image in this build, or a cold heavy image under `auto`).
//! Cites #139.
//!
//! Contents:
//! - `pub const Raw`
//! - `pub const Buffer`
//! - `pub const Stream`
//! - `pub const block_size: u32`
//! - `pub const BatchGpu = struct (`deinit`, `launch`)`
//! - `pub fn loadBatch(gpa: std.mem.Allocator, bg: *BatchGpu, b: device_ir.Batch, explicit: bool, report: bool, knobs: Knobs) !bool`
//! - `pub fn uploadBytes(kernel: *Raw, bytes: []const u8) !Buffer`
//! - `pub fn pinnedF64(kernel: *Raw, n: usize) ![]f64`
//! - `pub fn view(base: Buffer, byte_off: usize) Buffer`

const std = @import("std");
const assert = std.debug.assert;
const gompute = @import("gompute");
const device_ir = @import("device").abi;
const Knobs = @import("../Circuit.zig").Knobs;
const devices = @import("devices.zig");
const hdl = @import("hdl.zig");
const context = @import("context.zig");
const ctx_cost = @import("context_cost.zig");
const backend = devices.backend;
const Host = context.Host;
const Error = context.Error;

/// `void` in a build with no device images; every use sits behind
/// `comptime backend != null`.
pub const Raw = if (backend) |be| gompute.RawByName(be) else void;

pub const Buffer = if (backend != null) Raw.Buffer else void;

pub const Stream = if (backend != null) Raw.Stream else void;

/// The block width the kernels were exported with. HIP bakes it into
/// `globalIdX`; NVPTX reads the launch's real width, so CUDA may launch
/// narrower blocks (`evalBlock`).
pub const block_size: u32 = device_ir.gpu_block_size;

/// One batch's resident working set and the kernels that consume it.
pub const BatchGpu = struct {
    kernel: Raw,
    /// `arp_lim_<model>`: the limit/state pass, when the device has one.
    /// The limit half per limit pass, the state half once per converged
    /// solve, at the x the converger stages (`updateStatesOnGpu`).
    lim_kernel: ?Raw,
    /// `arp_ctl_<model>`: the accepted-step latch pass, when the device
    /// declares `stateCtl`.
    ctl_kernel: ?Raw,
    /// `arp_qtp_<model>`: the LTE charge tape, when the host batch has one.
    qtp_kernel: ?Raw,
    /// `arp_qonly_<model>`: q rows and tape at x, values only, when the
    /// image has it and the batch has charge (`evalQ`).
    q_kernel: ?Raw,
    /// `arp_evalc_<model>`, for a batch whose whole Jacobian is constant:
    /// the eval without g and c stamps, which sit in the device baseline
    /// instead. Null keeps the batch's g/c in the staging.
    const_kernel: ?Raw,
    /// Resident for the simulation. `repack` re-uploads only models and
    /// instances; the tapes never change.
    d_models: Buffer,
    d_model_of: Buffer,
    d_instances: Buffer,
    d_gath: Buffer,
    d_rhs_idx: Buffer,
    d_slots: Buffer,
    /// Lim plane (`GpuPayload.lim_x`'s length), `[]D.State` and the charge tape;
    /// 1-byte dummies when absent, so every kernel signature is uniform.
    d_lim: Buffer,
    d_states: Buffer,
    d_tape: Buffer,
    /// The host batch's `q_tape`, which `syncTape` fills.
    tape: []f64,
    /// The host batch, so `repack` can re-read its parameter arrays.
    ctx: *anyopaque,
    payload: *const fn (*anyopaque) device_ir.GpuPayload,
    set_limit_active: ?*const fn (*anyopaque, bool) void,
    count: u32,
    /// Bytes `d_models` holds; `repack` regrows it when unsharing changed it.
    models_len: usize,
    grid: gompute.Dim3,
    block: u32,
    has_lim: bool,
    /// The state half of `lim_kernel` can do anything: the batch has device
    /// state or a host state hook. A limit-only model (BSIM4) skips that
    /// launch and its wait.
    has_state: bool,
    /// The host batch carries charge (`Batch.has_charge`).
    has_charge: bool,
    /// The host batch has `Hooks.status`: eval latches `vera_status__` in
    /// the resident Instance, which `syncStatus` brings home.
    has_status: bool,
    /// Host-stepped (`eval.gpuHostStepped`): state hooks but no state
    /// kernel. The host walks step its Instance, and `pushStepped` uploads
    /// the rows they changed before the next launch that reads them.
    stepped: bool,
    /// The device lim plane holds live clamp state (after a seed upload or a
    /// limit launch). Mirrors the host batch's `lim_active`.
    lim_active: bool = false,
    /// Host `seedJunctions` wrote the host lim plane; upload it before the
    /// next launch that reads it.
    lim_dirty: bool = false,
    /// The Instance bytes host and device last agreed on, kept only when a
    /// kernel writes the resident Instance (the status latch, the limit/state
    /// pass or the accepted-step latch); empty otherwise. `syncInstances`
    /// diffs the host against it to tell parameter writes from latches.
    base: []u8,
    /// Scratch the device Instance bytes download into, `base`'s length, so
    /// the per-step merge allocates nothing (#174). Page-locked (`kernel`'s
    /// `freePinned`). Contents are dead between calls.
    dev: []u8,

    pub fn deinit(self: *BatchGpu, gpa: std.mem.Allocator) void {
        gpa.free(self.base);
        if (self.dev.len > 0) self.kernel.freePinned(self.dev);
        for ([_]*Buffer{ &self.d_models, &self.d_model_of, &self.d_instances, &self.d_gath, &self.d_rhs_idx, &self.d_slots, &self.d_lim, &self.d_states, &self.d_tape }) |b| b.free();
        for ([_]*?Raw{ &self.lim_kernel, &self.ctl_kernel, &self.qtp_kernel, &self.q_kernel, &self.const_kernel }) |opt| if (opt.*) |*k| k.deinit();
        self.kernel.deinit();
    }

    /// `launchOn` with this batch's grid.
    /// Asserts that a host-stepped batch's host rows have been pushed
    /// (`pushStepped`), checked in safe builds only (one memcmp).
    pub fn launch(self: *BatchGpu, k: *Raw, stream: *Stream, args: []const gompute.interface.Arg) !void {
        if (std.debug.runtime_safety and self.stepped)
            std.debug.assert(std.mem.eql(u8, self.payload(self.ctx).instances, self.base));
        try k.launchOn(stream, self.grid, .{ .x = self.block }, 0, args);
    }
};

/// Loads batch `b`'s kernels and uploads its working set into `bg`.
/// False when it stays on the host: no GPU payload or no image in this
/// build. Under `auto`, an image the driver has not compiled yet fails
/// with `ImageNotCompiled`.
pub fn loadBatch(gpa: std.mem.Allocator, bg: *BatchGpu, b: device_ir.Batch, explicit: bool, report: bool, knobs: Knobs) !bool {
    const get = b.hooks.gpu_payload orelse return false;
    const p = get(b.ctx);
    const image = ctx_cost.imageOf(p.kernel) orelse hdl.runtimeImage(@ptrCast(get)) orelse {
        if (report) std.debug.print("note: GPU batch '{s}' ({d} instances) stays on the CPU: no kernel image in this build\n", .{ b.type_name, p.count });
        return false;
    };
    const heavy = image.len >= ctx_cost.heavy_image_bytes;
    const cold = heavy and !ctx_cost.imageWarm(image, knobs);
    if (cold and !explicit) {
        std.debug.print(
            "note: GPU kernel for '{s}' is not compiled yet ({d} MB of PTX), so this run stays on the CPU; `--backend cuda` compiles it once per build\n",
            .{ b.type_name, image.len >> 20 },
        );
        return Error.ImageNotCompiled;
    }
    if (cold and report) std.debug.print(
        "note: compiling the GPU kernel for '{s}' ({d} MB of PTX); the driver caches it for later runs\n",
        .{ b.type_name, image.len >> 20 },
    );

    var kernel = try hdl.kernel(backend.?, @ptrCast(get), p.kernel);
    errdefer kernel.deinit();
    if (heavy) ctx_cost.markWarm(image, knobs);
    // The other entry points share the eval kernel's image, so a missing
    // one means a stale image: fail rather than run half-resident.
    var lim_kernel: ?Raw = if (p.lim_kernel.len > 0) try hdl.kernel(backend.?, @ptrCast(get), p.lim_kernel) else null;
    errdefer if (lim_kernel) |*k| k.deinit();
    var ctl_kernel: ?Raw = if (p.ctl_kernel.len > 0) try hdl.kernel(backend.?, @ptrCast(get), p.ctl_kernel) else null;
    errdefer if (ctl_kernel) |*k| k.deinit();
    // The kernels below are named after the eval kernel, not carried in
    // the frozen payload; an older image without one keeps the host's
    // path for it (the row LTE, the full eval, the staged g/c).
    var name: [128]u8 = undefined;
    var qtp_kernel: ?Raw = null;
    if (b.hooks.q_tape != null) if (renamed(&name, p.kernel, "arp_qtp_")) |n| {
        qtp_kernel = try optional(hdl.kernel(backend.?, @ptrCast(get), n));
    };
    errdefer if (qtp_kernel) |*k| k.deinit();
    var q_kernel: ?Raw = null;
    if (b.has_charge) if (renamed(&name, p.kernel, "arp_qonly_")) |n| {
        q_kernel = try optional(hdl.kernel(backend.?, @ptrCast(get), n));
    };
    errdefer if (q_kernel) |*k| k.deinit();
    var const_kernel: ?Raw = null;
    if (b.has_const_jacobian and !knobs.gpu_nofuse) if (renamed(&name, p.kernel, "arp_evalc_")) |n| {
        const_kernel = try optional(hdl.kernel(backend.?, @ptrCast(get), n));
    };
    errdefer if (const_kernel) |*k| k.deinit();
    const tape: []f64 = if (qtp_kernel != null) @constCast(b.hooks.q_tape.?(b.ctx)) else &.{};

    // One errdefer per buffer: `bg` counts only once it is whole, so a
    // failure part way through frees what this batch already holds.
    var d_models = try uploadBytes(&kernel, p.models);
    errdefer d_models.free();
    var d_model_of = try uploadBytes(&kernel, std.mem.sliceAsBytes(p.model_of));
    errdefer d_model_of.free();
    var d_instances = try uploadBytes(&kernel, p.instances);
    errdefer d_instances.free();
    var d_gath = try uploadBytes(&kernel, std.mem.sliceAsBytes(p.gath));
    errdefer d_gath.free();
    var d_rhs_idx = try uploadBytes(&kernel, std.mem.sliceAsBytes(p.rhs_idx));
    errdefer d_rhs_idx.free();
    var d_slots = try uploadBytes(&kernel, std.mem.sliceAsBytes(p.slots));
    errdefer d_slots.free();
    // Left unwritten: reads are gated by `lim_active`, like the host's
    // `lim_x`.
    var d_lim = try kernel.alloc(@max(p.lim_x.len * @sizeOf(f64), 1));
    errdefer d_lim.free();
    var d_states = try uploadBytes(&kernel, p.states);
    errdefer d_states.free();
    var d_tape = try kernel.alloc(@max(tape.len * @sizeOf(f64), 1));
    errdefer d_tape.free();
    // `eval.gpuHostStepped`: state hooks but no device state kernel.
    const stepped = lim_kernel == null and
        (b.hooks.update_state != null or b.hooks.commit_state != null or b.hooks.state_ctl != null);
    // A stepped batch keeps `base` too: `pushStepped` diffs against it.
    const writes_instances = b.hooks.status != null or lim_kernel != null or ctl_kernel != null or stepped;
    const base: []u8 = if (writes_instances) try gpa.dupe(u8, p.instances) else &.{};
    errdefer gpa.free(base);
    // Pinned, so the per-step Instance downloads queue on the stream.
    const dev: []u8 = if (base.len > 0) try kernel.allocPinned(base.len) else &.{};

    const block = evalBlock(knobs);
    bg.* = .{
        .kernel = kernel,
        .lim_kernel = lim_kernel,
        .ctl_kernel = ctl_kernel,
        .qtp_kernel = qtp_kernel,
        .q_kernel = q_kernel,
        .const_kernel = const_kernel,
        .d_models = d_models,
        .d_model_of = d_model_of,
        .d_instances = d_instances,
        .d_gath = d_gath,
        .d_rhs_idx = d_rhs_idx,
        .d_slots = d_slots,
        .d_lim = d_lim,
        .d_states = d_states,
        .d_tape = d_tape,
        .tape = tape,
        .ctx = b.ctx,
        .payload = get,
        .set_limit_active = b.hooks.set_limit_active,
        .count = p.count,
        .models_len = p.models.len,
        .grid = gompute.Dim3.linear(p.count, block),
        .block = block,
        .has_lim = p.lim_x.len > 0,
        .has_state = p.states.len > 0 or b.hooks.update_state != null or b.hooks.commit_state != null,
        .has_charge = b.has_charge,
        .has_status = b.hooks.status != null,
        .stepped = stepped,
        .base = base,
        .dev = dev,
    };
    return true;
}

pub fn uploadBytes(kernel: *Raw, bytes: []const u8) !Buffer {
    // Empty arrays still need a valid kernel-argument pointer, and
    // drivers reject zero-byte allocations.
    var buf = try kernel.alloc(@max(bytes.len, 1));
    errdefer buf.free();
    if (bytes.len > 0) try buf.upload(bytes.ptr, bytes.len);
    return buf;
}

/// Page-locked `[]f64` from the driver; release with `freePinned`.
pub fn pinnedF64(kernel: *Raw, n: usize) ![]f64 {
    const bytes = try kernel.allocPinned(n * @sizeOf(f64));
    return @alignCast(std.mem.bytesAsSlice(f64, bytes));
}

/// A non-owning buffer `byte_off` bytes into `base`, for kernel arguments.
/// Never `free` it.
pub fn view(base: Buffer, byte_off: usize) Buffer {
    var v = base;
    v.bytes -= byte_off;
    v.handle = switch (@typeInfo(@TypeOf(base.handle))) {
        .int => base.handle + byte_off,
        else => @ptrFromInt(@intFromPtr(base.handle) + byte_off), // else: HIP hands out pointers
    };
    return v;
}

// ---- private ----

/// The CUDA block width for a batch's per-instance kernels. Compact models
/// take ~255 registers a thread, so one 256-wide block fills an SM's register
/// file; 2000 instances were then 8 blocks on 24 SMs. 64 spreads any batch
/// over every SM: per eval against 256, bsim4va 679 us against 866, psp103
/// 1071 against 1284; 32 measured the same as 64. HIP keeps the exported
/// width, which its `globalIdX` bakes in. `ESPICE_GPU_BLOCK` overrides it
/// (`Knobs.gpu_block`).
fn evalBlock(knobs: Knobs) u32 {
    if (comptime backend != .cuda) return block_size;
    return knobs.gpu_block;
}

/// The kernel `lookup` loaded, or null when the image lacks it.
fn optional(lookup: anytype) !?Raw {
    return lookup catch |e| switch (e) {
        error.KernelNotFound => null,
        else => return e, // else: any other driver error is fatal for the batch
    };
}

/// `eval_kernel` (`arp_eval_<model>`) with its prefix swapped for `prefix`
/// (`arp_qtp_<model>`, ...), or null when the name has another shape or the
/// result does not fit `buf`.
fn renamed(buf: []u8, eval_kernel: []const u8, prefix: []const u8) ?[]const u8 {
    const eval_prefix = "arp_eval_";
    if (!std.mem.startsWith(u8, eval_kernel, eval_prefix)) return null;
    return std.fmt.bufPrint(buf, "{s}{s}", .{ prefix, eval_kernel[eval_prefix.len..] }) catch null;
}

test renamed {
    var buf: [32]u8 = undefined;
    try std.testing.expectEqualStrings("arp_qtp_mos1", renamed(&buf, "arp_eval_mos1", "arp_qtp_").?);
    try std.testing.expectEqualStrings("arp_qtp_", renamed(&buf, "arp_eval_", "arp_qtp_").?);
    try std.testing.expect(renamed(&buf, "arp_lim_mos1", "arp_qtp_") == null);
    try std.testing.expect(renamed(&buf, "", "arp_qtp_") == null);
    // Too long for the buffer: no tape kernel, not a panic.
    try std.testing.expect(renamed(&buf, "arp_eval_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx", "arp_qtp_") == null);
}
