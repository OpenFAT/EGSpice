//! `ESPICE_GPU_LU_BENCH` (E2 of dev/solvers/gpu-lu.md), compiled only under
//! `-Dlu-instrument`. In: each device refactor the device LU runs. Out: the
//! same refactor and solve timed on the host (`SparseLu`, one thread) and
//! with the kernel bodies on 1 and `ESPICE_GPU_LU_THREADS` (default 8) host
//! threads, against the device kernels by event timing; medians printed at
//! teardown, or after N samples for `ESPICE_GPU_LU_BENCH=N`.
//! Cites: dev/solvers/gpu-lu.md E2.
//!
//! Contents:
//! - `pub const Bench = struct` (`ev`, `init`, `sample`, `deinit`)

const std = @import("std");
const assert = std.debug.assert;
const solver = @import("solver");
const numerics = @import("core").numerics;
const gpu_lu = @import("lu.zig");
const ctx_batch = @import("context_batch.zig");

const K = solver.lu_kernels;
const direct = solver.direct;
const GpuLu = gpu_lu.GpuLu;
const Raw = ctx_batch.Raw;
const Event = gpu_lu.Event;
const nowNs = @import("context.zig").nowNs;

/// `ESPICE_GPU_LU_BENCH` (E2): after each device refactor, times the same
/// refactor and solve on the host (`SparseLu`, one thread) and with the
/// kernel bodies on 1 and `ESPICE_GPU_LU_THREADS` (default 8) host threads,
/// against the device kernels by event timing. Prints medians at teardown,
/// or after N samples for `ESPICE_GPU_LU_BENCH=N` (then stops sampling).
pub const Bench = struct {
    /// Events around the refactor, the L solve and the U solve (`GpuLu.enqueueCall`).
    ev: [4]Event,
    threads: u32,
    limit: usize = 0,
    /// Microseconds per sample: device refactor kernel, device solve
    /// kernels, device call wall (upload to download), host SparseLu
    /// refactor, host solve, kernel bodies on 1 thread, on `threads`.
    s: [8]std.ArrayList(f64) = @splat(.empty),
    val: []f64 = &.{},
    y: []f64 = &.{},
    dx: []f64 = &.{},
    sync: []u32 = &.{},
    stamp: u32 = 0,
    /// `GpuLu.st.epochs` the buffers above are sized for.
    epoch: u64 = 0,

    pub fn init(gpa: std.mem.Allocator, k: *Raw, limit: usize, threads: u32) !*Bench {
        if (!k.hasEvents()) return error.Unsupported;
        const b = try gpa.create(Bench);
        errdefer gpa.destroy(b);
        b.* = .{ .ev = undefined, .threads = threads, .limit = limit };
        var made: usize = 0;
        errdefer for (b.ev[0..made]) |*e| e.deinit();
        for (&b.ev) |*e| {
            e.* = try k.createEvent(true);
            made += 1;
        }
        return b;
    }

    pub fn sample(b: *Bench, lu_ctx: *GpuLu, slv: *direct.Solver, vals: []const f64, rhs: []const f64, wall_ns: u64) !void {
        const gpa = lu_ctx.gpa;
        const t = lu_ctx.tab;
        if (b.limit != 0 and b.s[0].items.len >= b.limit) return;
        defer if (b.limit != 0 and b.s[0].items.len == b.limit) b.report(lu_ctx);
        if (b.epoch != lu_ctx.st.epochs) {
            for ([_][]f64{ b.val, b.y, b.dx }) |p| gpa.free(p);
            gpa.free(b.sync);
            b.val = try gpa.alloc(f64, t.n_val);
            b.y = try gpa.alloc(f64, t.n);
            b.dx = try gpa.alloc(f64, t.n);
            b.sync = try gpa.alloc(u32, t.syncLen());
            @memset(b.sync, 0);
            b.epoch = lu_ctx.st.epochs;
        }
        try b.s[0].append(gpa, try Event.elapsedUs(&b.ev[0], &b.ev[1]));
        try b.s[1].append(gpa, try Event.elapsedUs(&b.ev[1], &b.ev[3]));
        try b.s[7].append(gpa, try Event.elapsedUs(&b.ev[1], &b.ev[2]));
        try b.s[2].append(gpa, @as(f64, @floatFromInt(wall_ns)) * 1e-3);
        const lu = slv.sparse().?;
        var t0 = nowNs();
        try lu.refactor(slv.col_ptr, vals, slv.params.refactor_growth_limit);
        var t1 = nowNs();
        try b.s[3].append(gpa, @as(f64, @floatFromInt(t1 - t0)) * 1e-3);
        const neg = b.dx; // scratch until the kernel runs below
        numerics.scale(neg, -1, rhs[0..t.n]);
        t0 = nowNs();
        lu.solve(neg, neg);
        t1 = nowNs();
        try b.s[4].append(gpa, @as(f64, @floatFromInt(t1 - t0)) * 1e-3);
        slv.host_stale = false; // the refactor above brought lu to vcopy
        const bufs: K.HostBufs = .{ .idx = lu_ctx.idx, .val = b.val, .a = vals, .rhs = rhs, .y = b.y, .dx = b.dx, .sync = b.sync };
        for ([_]u32{ 1, b.threads }, 5..) |th, slot| {
            b.stamp += 1;
            t0 = nowNs();
            _ = try K.runHost(t, bufs, b.stamp, th);
            t1 = nowNs();
            try b.s[slot].append(gpa, @as(f64, @floatFromInt(t1 - t0)) * 1e-3);
        }
    }

    fn report(b: *Bench, lu_ctx: *GpuLu) void {
        const names = [_][]const u8{ "dev_refactor", "dev_solves", "dev_call_wall", "host_refactor", "host_solve", "kern_1thr", "kern_nthr", "dev_lsolve" };
        const t = lu_ctx.tab;
        const x = lu_ctx.idx;
        var wide: u32 = 0;
        for (0..t.n) |k| wide += @intFromBool(x[t.wlev + k] != x[t.wlev + k + 1]);
        // Head span: per level, the longest row list.
        var chains: [2]u64 = .{ 0, 0 };
        for ([_][3]u32{ .{ t.l_levels, t.l_lev, t.l_rows }, .{ t.u_levels, t.u_lev, t.u_rows } }, [_]u32{ t.l_ptr, t.u_ptr }, 0..) |lv, ptr, side| {
            for (0..lv[0]) |v| {
                var mx: u32 = 0;
                for (x[lv[1] + v]..x[lv[1] + v + 1]) |j| {
                    const r = x[lv[2] + j];
                    mx = @max(mx, x[ptr + r + 1] - x[ptr + r]);
                }
                chains[side] += mx;
            }
        }
        std.debug.print("gpu-lu-bench: n={d} F={d} l_levels={d} u_levels={d} head_chain={d}+{d} tail={d} gather_cols={d} samples={d} threads={d} checked={d} mismatches={d} (median us)", .{ lu_ctx.n, lu_ctx.flops, t.l_levels, t.u_levels, chains[0], chains[1], t.n - t.t0, wide, b.s[0].items.len, b.threads, if (lu_ctx.knobs.gpu_lu_check) lu_ctx.st.refactors else 0, lu_ctx.st.mismatches });
        for (&b.s, names) |*s, name| {
            std.mem.sort(f64, s.items, {}, std.sort.asc(f64));
            const m = if (s.items.len == 0) 0 else s.items[s.items.len / 2];
            std.debug.print(" {s}={d:.1}", .{ name, m });
        }
        std.debug.print("\n", .{});
    }

    pub fn deinit(b: *Bench, lu_ctx: *GpuLu) void {
        const gpa = lu_ctx.gpa;
        if (b.limit == 0 or b.s[0].items.len < b.limit) b.report(lu_ctx);
        for (&b.s) |*s| s.deinit(gpa);
        for ([_][]f64{ b.val, b.y, b.dx }) |p| gpa.free(p);
        gpa.free(b.sync);
        for (&b.ev) |*e| e.deinit();
        gpa.destroy(b);
    }
};
