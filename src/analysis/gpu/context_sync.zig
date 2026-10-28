//! Host and device Instance bytes of the resident batches. In: the host
//! copies, the device copies and the bytes both last agreed on. Out: one
//! merged copy on both sides, where a host write (a parameter, `recompute`)
//! wins and every other byte takes the device's latches; and the device
//! state brought home for dependents.
//! Cites #174 (the pinned merge scratch).
//!
//! Contents:
//! - `pub noinline fn markDirty(self: *GpuContext) void`
//! - `pub fn syncStatus(self: *GpuContext) void`
//! - `pub fn syncHostState(self: *GpuContext) !void`
//! - `pub fn repack(self: *GpuContext) !void`
//! - `pub fn pullStepped(self: *GpuContext) !void`
//! - `pub fn pushStepped(self: *GpuContext) !bool`

const std = @import("std");
const assert = std.debug.assert;
const walk = @import("../par_eval.zig");
const devices = @import("devices.zig");
const context = @import("context.zig");
const ctx_batch = @import("context_batch.zig");
const ctx_eval = @import("context_eval.zig");
const backend = devices.backend;
const GpuContext = context.GpuContext;
const BatchGpu = ctx_batch.BatchGpu;

/// Marks the resident models and instances stale after a host
/// parameter write; the context re-uploads before its next launch.
pub noinline fn markDirty(self: *GpuContext) void {
    self.query_hit = null;
    self.pre = null;
    self.params_dirty = true;
}

/// Downloads the resident instances when a batch can latch a VerA
/// `$fatal`/`$error` (`Hooks.status`), so the host reads and formats it
/// as it would for a host batch. The device stores the latch with the
/// same arithmetic as the host. Nothing to do after a fallback: the host
/// walk then evaluated, and latched, on its own instances.
pub fn syncStatus(self: *GpuContext) void {
    self.query_hit = null;
    if (comptime backend == null) return;
    if (self.demoted) return;
    if (self.poisoned or self.warned_eval) return;
    for (self.batches) |bg| if (bg.has_status) {
        syncInstances(self, false) catch {};
        return;
    };
}

/// Downloads the resident instance, state and lim data into the host
/// batches, so dependents copy an OP that matches the device. Fails with
/// `error.GpuStateUnavailable` after any fallback, which can leave the two
/// sides at different accepted points, or with the driver's error from a
/// download, which can leave the host copy part-written.
pub fn syncHostState(self: *GpuContext) !void {
    self.query_hit = null;
    if (comptime backend == null) return;
    if (self.demoted) return;
    if (self.poisoned or self.warned_eval or self.warned_state)
        return error.GpuStateUnavailable;
    try ctx_eval.sync(self, .other);
    try syncInstances(self, false);
    for (self.batches) |*batch| {
        const payload = batch.payload(batch.ctx);
        try ctx_eval.download(self, &batch.d_states, @constCast(payload.states));
        if (batch.has_lim and !batch.lim_dirty) {
            if (batch.lim_active)
                try ctx_eval.download(self, &batch.d_lim, std.mem.sliceAsBytes(@constCast(payload.lim_x)));
            if (batch.set_limit_active) |set| set(batch.ctx, batch.lim_active);
        }
    }
    // Pageable destinations: the copies above already blocked, but say
    // so rather than lean on the driver.
    try ctx_eval.sync(self, .other);
}

/// Re-uploads models after a host parameter change and merges the
/// instances both ways (`syncInstances`), so the device keeps its
/// latches. The tapes never change.
pub fn repack(self: *GpuContext) !void {
    if (comptime backend == null) return;
    for (self.batches) |*bg| {
        const p = bg.payload(bg.ctx);
        // Unsharing (a parameter write) grows `models` past the buffer.
        if (p.models.len != bg.models_len) {
            // The old buffer may still feed a queued launch.
            try ctx_eval.sync(self, .other);
            var models = try bg.kernel.alloc(@max(p.models.len, 1));
            errdefer models.free();
            try ctx_eval.upload(self, &models, p.models);
            bg.d_models.free();
            bg.d_models = models;
            try ctx_eval.upload(self, &bg.d_model_of, std.mem.sliceAsBytes(p.model_of));
            bg.models_len = p.models.len;
        } else try ctx_eval.upload(self, &bg.d_models, p.models);
    }
    try syncInstances(self, true);
    // A parameter can flip `jacConstApplies`, leaving a cell its batch
    // no longer writes.
    try self.d_stage.fillAsync(0, @max(self.n_stage, 1) * @sizeOf(f64), &self.stream);
    try ctx_eval.uploadBase(self);
    self.params_dirty = false;
}

/// Merges the device Instance bytes of each host-stepped batch whose eval
/// can latch a status into the host copy before a host walk reads them;
/// bytes the host wrote since `base` win. Eval writes nothing else in a
/// stepped batch, so the others need no download. One wait when any.
pub fn pullStepped(self: *GpuContext) !void {
    if (comptime backend == null) return;
    var any = false;
    for (self.batches) |*bg| {
        if (!bg.stepped or !bg.has_status) continue;
        try ctx_eval.download(self, &bg.d_instances, bg.dev);
        any = true;
    }
    if (!any) return;
    try ctx_eval.sync(self, .stepped);
    for (self.batches) |*bg| {
        if (!bg.stepped or !bg.has_status) continue;
        const host: []u8 = @constCast(bg.payload(bg.ctx).instances);
        std.debug.assert(bg.dev.len == host.len);
        _ = mergeInstances(host, bg.base, bg.dev);
        @memcpy(bg.base, bg.dev);
    }
}

/// Uploads each host-stepped batch whose host Instance bytes moved since
/// `base` (an `updateState`, `commit_state` or revert on the host walk),
/// merging device latches first when eval can write one. True when it
/// uploaded anything. One memcmp per stepped batch when nothing moved.
pub fn pushStepped(self: *GpuContext) !bool {
    if (comptime backend == null) return false;
    var pushed = false;
    for (self.batches) |*bg| {
        if (!bg.stepped) continue;
        const host: []u8 = @constCast(bg.payload(bg.ctx).instances);
        std.debug.assert(bg.base.len == host.len);
        if (std.mem.eql(u8, host, bg.base)) continue;
        if (bg.has_status) {
            // Normally `pullStepped` already merged; this covers a host
            // write outside `updateStates` (a revert, `commit_state`).
            try ctx_eval.download(self, &bg.d_instances, bg.dev);
            try ctx_eval.sync(self, .stepped);
            _ = mergeInstances(host, bg.base, bg.dev);
        }
        try ctx_eval.upload(self, &bg.d_instances, host);
        @memcpy(bg.base, host);
        pushed = true;
    }
    return pushed;
}

// ---- private ----

/// Brings each resident batch's host and device Instance bytes to the
/// same value: a byte the host wrote since `BatchGpu.base` (a parameter
/// write, `recompute`) wins, and every other byte takes the device's (its
/// latches). A batch whose kernels never write instances downloads
/// nothing, because the host copy is current, and uploads only under
/// `push`.
fn syncInstances(self: *GpuContext, push: bool) !void {
    if (comptime backend == null) return;
    var any = false;
    for (self.batches) |*bg| {
        const p = bg.payload(bg.ctx);
        if (p.instances.len == 0) continue;
        if (bg.base.len == 0) {
            if (push) try ctx_eval.upload(self, &bg.d_instances, p.instances);
            continue;
        }
        std.debug.assert(bg.dev.len == p.instances.len);
        try ctx_eval.download(self, &bg.d_instances, bg.dev);
        any = true;
    }
    if (!any) return;
    try ctx_eval.sync(self, .other);
    for (self.batches) |*bg| {
        if (bg.base.len == 0) continue;
        const host: []u8 = @constCast(bg.payload(bg.ctx).instances);
        if (mergeInstances(host, bg.base, bg.dev)) try ctx_eval.upload(self, &bg.d_instances, host);
        @memcpy(bg.base, host);
    }
}

/// Three-way merges one batch's Instance bytes into `host`: a byte that
/// differs from `base` was written by the host and stays; every other byte
/// takes `device`'s. Returns whether the host wrote any byte, so the device
/// needs the merged copy. All three are the same length.
// ponytail: byte granularity, so a field both sides rewrote since `base`
// would tear. The host writes parameters and derived values and the
// device writes latches, so no field has two writers today.
fn mergeInstances(host: []u8, base: []const u8, device: []const u8) bool {
    std.debug.assert(host.len == base.len and base.len == device.len);
    // Branchless, so it vectorizes: 0.065 against 0.40 ns/byte for the
    // `if`/`else` store (1 MiB, 1% host writes, i9-14900HX).
    var wrote: u8 = 0;
    for (host, base, device) |*h, b, d| {
        const mine = h.* != b;
        wrote |= @intFromBool(mine);
        h.* = if (mine) h.* else d;
    }
    return wrote != 0;
}

test "mergeInstances keeps host parameter writes and device latches" {
    // No padding: every byte compared is a field byte.
    const Inst = extern struct { param: f64, status: u64, pq: f64 };
    const base = [2]Inst{ .{ .param = 1, .status = 0, .pq = 0 }, .{ .param = 2, .status = 0, .pq = 0 } };
    // The device latched a status and committed charge on both...
    var device = base;
    device[0].status = 3;
    device[0].pq = 0.5;
    device[1].pq = -0.25;
    // ...while the host wrote instance 1's parameter after the last eval.
    var host = base;
    host[1].param = 7;
    try std.testing.expect(mergeInstances(std.mem.asBytes(&host), std.mem.asBytes(&base), std.mem.asBytes(&device)));
    try std.testing.expectEqual(Inst{ .param = 1, .status = 3, .pq = 0.5 }, host[0]);
    try std.testing.expectEqual(Inst{ .param = 7, .status = 0, .pq = -0.25 }, host[1]);

    // Nothing written on the host: a plain pull, and nothing to push.
    var pulled = base;
    try std.testing.expect(!mergeInstances(std.mem.asBytes(&pulled), std.mem.asBytes(&base), std.mem.asBytes(&device)));
    try std.testing.expectEqualSlices(u8, std.mem.asBytes(&device), std.mem.asBytes(&pulled));
}

test "mergeInstances matches the bytewise rule at every length and offset" {
    var prng = std.Random.DefaultPrng.init(0xbeef);
    const rnd = prng.random();
    var host: [204]u8 = undefined;
    var base: [204]u8 = undefined;
    var dev: [204]u8 = undefined;
    var want: [204]u8 = undefined;
    for (0..201) |n| for (0..4) |off| {
        const h = host[off..][0..n];
        const b = base[3 - off ..][0..n];
        const d = dev[(off + 1) % 4 ..][0..n];
        rnd.bytes(b);
        rnd.bytes(d);
        @memcpy(h, b);
        var wrote = false;
        for (h) |*x| if (rnd.uintLessThan(u8, 16) == 0) {
            x.* +%= 1 + rnd.uintLessThan(u8, 255); // never a multiple of 256
            wrote = true;
        };
        for (want[0..n], h, b, d) |*w, x, y, z| w.* = if (x != y) x else z;
        try std.testing.expectEqual(wrote, mergeInstances(h, b, d));
        try std.testing.expectEqualSlices(u8, want[0..n], h);
    };
}
