//! A transient GPU lane batch across every usable device (multi-GPU stage
//! 1): contiguous row-order slices by device capacity (`devices.split`),
//! slice 0 on the leader's thread, the rest on pooled threads, every slice
//! replaying the pivot tape of the batch's lane 0 (`devices.TapeShare`), so
//! a lane's bits do not depend on the device count.

const std = @import("std");
const devices = @import("devices.zig");
const worker = @import("../worker.zig");
const now = @import("lanes.zig").Prof.now;

/// `M.runGpuLanes(ctxs, opts, out)` split over the devices; one device (or
/// too few rows for two slices) runs it as before, on that device. `gpa` is
/// thread-safe; `threads` lends the slices' threads (null spawns them).
pub fn runLanes(comptime M: type, gpa: std.mem.Allocator, threads: ?*worker.Threads, ctxs: anytype, opts: anytype, out: anytype) void {
    var ends: [16]usize = undefined;
    var who: [16]c_int = undefined;
    var weights: [16]devices.Device = undefined;
    const k = devices.split(devices.weighted(&weights), ctxs.len, &ends, &who);
    const prev = devices.ordinal;
    defer devices.ordinal = prev;
    if (k == 1) {
        devices.ordinal = who[0];
        return M.runGpuLanes(ctxs, opts, out);
    }
    if (ctxs[0].circuit.knobs.gpu_stats != .off) {
        std.debug.print("gpu-devices: {d} lanes in {d} slices:", .{ ctxs.len, k });
        for (ends[0..k], who[0..k]) |e, o| std.debug.print(" ..{d} on {d}", .{ e, o });
        std.debug.print("\n", .{});
    }
    var share: devices.TapeShare = .{ .gpa = gpa };
    defer share.deinit();
    const Slice = struct {
        ctxs: @TypeOf(ctxs),
        opts: @TypeOf(opts),
        out: @TypeOf(out),
        ordinal: c_int,
        share: *devices.TapeShare,
        builds: bool,

        fn run(raw: *anyopaque) void {
            const s: *@This() = @ptrCast(@alignCast(raw));
            const t0 = now();
            defer devices.record(s.ordinal, s.ctxs.len, now() -| t0);
            devices.ordinal = s.ordinal;
            devices.tape = s.share;
            devices.tape_builds = s.builds;
            defer {
                devices.tape = null;
                devices.tape_builds = false;
            }
            M.runGpuLanes(s.ctxs, s.opts, s.out);
            // Slice 0 that ran no device init frees the waiting slices.
            if (s.builds) s.share.fail();
        }
    };
    var slices: [16]Slice = undefined;
    var start: usize = 0;
    for (slices[0..k], ends[0..k], who[0..k], 0..) |*s, end, ord, i| {
        s.* = .{ .ctxs = ctxs[start..end], .opts = opts, .out = out[start..end], .ordinal = ord, .share = &share, .builds = i == 0 };
        start = end;
    }
    // Helper threads first: they wait on slice 0's tape, which this thread
    // builds.
    var pooled: [16]?*worker.Threads.Slot = @splat(null);
    var spawned: [16]?std.Thread = @splat(null);
    for (slices[1..k], 1..) |*s, i| {
        const task: worker.Threads.Task = .{ .run = Slice.run, .ctx = s };
        if (threads) |t| {
            pooled[i] = t.spawn(task) catch null;
            if (pooled[i] != null) continue;
        }
        spawned[i] = std.Thread.spawn(.{ .stack_size = worker.stack_size }, Slice.run, .{@as(*anyopaque, s)}) catch null;
        // ponytail: no thread at all runs the slice here, after slice 0,
        // from the tape slice 0 published.
    }
    Slice.run(&slices[0]);
    for (slices[1..k], 1..) |*s, i| {
        if (pooled[i]) |slot| threads.?.join(slot) else if (spawned[i]) |th| th.join() else Slice.run(s);
    }
}
