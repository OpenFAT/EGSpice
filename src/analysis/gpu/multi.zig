//! A transient GPU lane batch across every usable device (multi-GPU stage
//! 1). In: one batch's run contexts and options. Out: each lane's outcome,
//! run as contiguous row-order slices by device capacity (`devices.split`),
//! slice 0 on the leader's thread and the rest on pooled threads, every
//! slice replaying the pivot tape of the batch's lane 0
//! (`devices.TapeShare`), so a lane's bits do not depend on the device count.
//! Cites: none.
//!
//! Contents:
//! - `pub fn runLanes(comptime M: type, gpa: std.mem.Allocator, threads: ?*worker.Threads, ctxs: anytype, opts: anytype, out: anytype) void`

const std = @import("std");
const assert = std.debug.assert;
const worker = @import("../worker.zig");
const devices = @import("devices.zig");
const now = @import("lanes.zig").Prof.now;

/// `M.runGpuLanes(ctxs, opts, out)` split over the devices; one device (or
/// too few rows for two slices) runs it whole, on that device. `gpa` is
/// thread-safe; `threads` lends the slices' threads (null spawns them).
/// Asserts that `ctxs` and `out` have one row per lane. Restores this
/// thread's `devices.ordinal` on return.
// Generic over `M` (the analysis: tran, pss, hb, dc, lstb), the second
// case existing today; the slices' types are M's own.
pub fn runLanes(comptime M: type, gpa: std.mem.Allocator, threads: ?*worker.Threads, ctxs: anytype, opts: anytype, out: anytype) void {
    assert(ctxs.len == out.len);
    var ends: [devices.devices_max]usize = undefined;
    var who: [devices.devices_max]c_int = undefined;
    var weights: [devices.devices_max]devices.Device = undefined;
    const k = devices.split(devices.weighted(&weights), ctxs.len, &ends, &who);
    assert(k >= 1 and k <= devices.devices_max);
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
    const Slice = SliceOf(M, @TypeOf(ctxs), @TypeOf(opts), @TypeOf(out));
    var slices: [devices.devices_max]Slice = undefined;
    var start: usize = 0;
    for (slices[0..k], ends[0..k], who[0..k], 0..) |*s, end, ord, i| {
        s.* = .{ .ctxs = ctxs[start..end], .opts = opts, .out = out[start..end], .ordinal = ord, .share = &share, .builds = i == 0 };
        start = end;
    }
    assert(start == ctxs.len);
    // Helper threads first: they wait on slice 0's tape, which this thread
    // builds.
    var pooled: [devices.devices_max]?*worker.Threads.Slot = @splat(null);
    var spawned: [devices.devices_max]?std.Thread = @splat(null);
    for (slices[1..k], 1..) |*s, i| {
        const task: worker.Threads.Task = .{ .run = Slice.run, .ctx = s };
        if (threads) |t| {
            pooled[i] = t.spawn(task) catch null;
            if (pooled[i] != null) continue;
        }
        // ponytail: with no thread at all the slice runs below, after
        // slice 0, from the tape slice 0 published.
        spawned[i] = std.Thread.spawn(.{ .stack_size = worker.stack_size }, Slice.run, .{@as(*anyopaque, s)}) catch null;
    }
    Slice.run(&slices[0]);
    for (slices[1..k], 1..) |*s, i| {
        if (pooled[i]) |slot| threads.?.join(slot) else if (spawned[i]) |th| th.join() else Slice.run(s);
    }
}

// ---- private ----

/// One device's slice of a batch and the thread body that runs it there.
fn SliceOf(comptime M: type, comptime Ctxs: type, comptime Opts: type, comptime Out: type) type {
    return struct {
        ctxs: Ctxs,
        opts: Opts,
        out: Out,
        ordinal: c_int,
        share: *devices.TapeShare,
        /// This slice holds lane 0 and builds the shared tape.
        builds: bool,

        fn run(raw: *anyopaque) void {
            // SAFETY: `runLanes` hands every task a `*@This()` it keeps alive
            // until the join.
            const s: *@This() = @ptrCast(@alignCast(raw));
            assert(s.ctxs.len == s.out.len);
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
}
