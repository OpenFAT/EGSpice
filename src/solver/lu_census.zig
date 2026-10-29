//! The `-Dlu-instrument` census of the direct solver: a factor's pattern,
//! values and schedule in, one stderr line per measurement out. Every
//! function here runs only when `direct.instrument` is compiled in and the
//! `ZP_LU_STATS` / `ESPICE_LU_PAR_STATS` knobs ask for it; none changes a
//! factor's numbers.
//! Cites: dev/solvers/gpu-lu.md ("Multicore host refactor"), TODO 2.5 and 2.8.
//!
//! Contents:
//! - `pub fn btfCensus(gpa: Allocator, n: u32, col_ptr: []const u32, row_idx: []const u32) (Allocator.Error || order.Error)!void`
//! - `pub fn reachCensus(gpa: Allocator, lu: *const SparseLu, col_ptr: []const u32, old: []const f64, vals: []const f64) Allocator.Error!void`
//! - `pub fn benchSubtree(p: *subtree.Plan, lu: *SparseLu, col_ptr: []const u32, vals: []const f64, growth: f64, io: std.Io, mult_scan: *const fn (*const SparseLu) f64) void`
//! - `pub fn census(self: *const SparseLu, gpa: Allocator, col_ptr: []const u32, row_idx: []const u32) Allocator.Error!void`

const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;
const SparseLu = @import("sparse_lu.zig").SparseLu;
const order = @import("order.zig");
const subtree = @import("subtree.zig");

/// The census for keeping the BTF blocks through the factor (TODO 2.5):
/// the block count, singletons, the largest block and the entries outside
/// every diagonal block, which a block factor would leave unfactored.
pub fn btfCensus(gpa: Allocator, n: u32, col_ptr: []const u32, row_idx: []const u32) (Allocator.Error || order.Error)!void {
    assert(col_ptr.len == @as(usize, n) + 1);
    const buf = try gpa.alloc(u32, 3 * @as(usize, n) + 1 + order.wsSize(n, col_ptr[n]));
    defer gpa.free(buf);
    const emit = buf[0..n];
    const block_ptr = buf[n..][0 .. n + 1];
    const blk_of = buf[2 * n + 1 ..][0..n];
    var ws: order.Ws = .init(buf[3 * n + 1 ..]);
    const nb = try order.blocks(n, col_ptr, row_idx, emit, block_ptr, &ws);
    var largest: u32 = 0;
    var singletons: u32 = 0;
    for (0..nb) |b| {
        const size = block_ptr[b + 1] - block_ptr[b];
        largest = @max(largest, size);
        singletons += @intFromBool(size == 1);
        for (emit[block_ptr[b]..block_ptr[b + 1]]) |v| blk_of[v] = @intCast(b);
    }
    var off: u64 = 0;
    for (0..n) |j| for (row_idx[col_ptr[j]..col_ptr[j + 1]]) |r| {
        off += @intFromBool(blk_of[r] != blk_of[j]);
    };
    std.debug.print("lu-btf: n={d} blocks={d} singletons={d} largest={d} ({d:.1}% of n) off_block_nnz={d} ({d:.1}% of nnz)\n", .{
        n,                                                                            nb,  singletons,                                                                        largest,
        100 * @as(f64, @floatFromInt(largest)) / @as(f64, @floatFromInt(@max(n, 1))), off, 100 * @as(f64, @floatFromInt(off)) / @as(f64, @floatFromInt(@max(col_ptr[n], 1))),
    });
}

/// The census for a partial refactor (TODO 2.8): the share of A entries
/// that changed since the last factor, and of pivot steps a replay would
/// have to redo, those whose column changed or whose U names a redone step.
pub fn reachCensus(gpa: Allocator, lu: *const SparseLu, col_ptr: []const u32, old: []const f64, vals: []const f64) Allocator.Error!void {
    const n = lu.n;
    assert(old.len == vals.len);
    assert(col_ptr.len == @as(usize, n) + 1);
    const redo = try gpa.alloc(bool, n);
    defer gpa.free(redo);
    var changed: u64 = 0;
    var steps: u64 = 0;
    var flops: u64 = 0;
    var redo_flops: u64 = 0;
    for (0..n) |k| {
        const c = lu.q[k];
        var r = false;
        for (col_ptr[c]..col_ptr[c + 1]) |p| if (!(old[p] == vals[p])) {
            changed += 1;
            r = true;
        };
        var f: u64 = 0;
        for (lu.ui.items[lu.up[k]..lu.up[k + 1]]) |i| {
            r = r or redo[i];
            f += lu.lp[i + 1] - lu.lp[i];
        }
        redo[k] = r;
        flops += f;
        if (r) {
            steps += 1;
            redo_flops += f;
        }
    }
    const pct = struct {
        fn f(a: u64, b: u64) f64 {
            return 100 * @as(f64, @floatFromInt(a)) / @as(f64, @floatFromInt(@max(b, 1)));
        }
    }.f;
    std.debug.print("lu-reach: changed={d:.2}% steps={d:.2}% flops={d:.2}%\n", .{ pct(changed, col_ptr[n]), pct(steps, n), pct(redo_flops, flops) });
}

/// `-Dlu-instrument` with `ESPICE_LU_PAR_STATS`: the subtree refactor, and
/// the same with its tail run serially, against the serial one on this
/// epoch's first values, medians of 15 interleaved runs each. Leaves `lu`
/// refactored on `vals` (or failed, as the caller's own refactor would).
/// `mult_scan` is the multiplier scan the replay bar runs, timed beside it.
pub fn benchSubtree(p: *subtree.Plan, lu: *SparseLu, col_ptr: []const u32, vals: []const f64, growth: f64, io: std.Io, mult_scan: *const fn (*const SparseLu) f64) void {
    assert(lu.factored);
    assert(p.width >= 1);
    const reps = 15;
    var ser: [reps]u64 = undefined;
    var par: [reps]u64 = undefined;
    var scan: [reps]u64 = undefined;
    var stail: [reps]u64 = undefined;
    var sink: f64 = 0;
    const done = p.done;
    defer p.done = done;
    for (0..reps) |i| {
        const t0 = std.Io.Timestamp.now(io, .awake);
        lu.refactor(col_ptr, vals, growth) catch |e| {
            std.debug.print("lu-subtree-bench: refactor {s}\n", .{@errorName(e)});
            return;
        };
        const t1 = std.Io.Timestamp.now(io, .awake);
        p.run(lu, col_ptr, vals, growth, io) catch return;
        const t2 = std.Io.Timestamp.now(io, .awake);
        lu.replay_mult = null;
        sink += mult_scan(lu);
        const t3 = std.Io.Timestamp.now(io, .awake);
        p.done = &.{};
        p.run(lu, col_ptr, vals, growth, io) catch return;
        p.done = done;
        const t4 = std.Io.Timestamp.now(io, .awake);
        ser[i] = @intCast(t0.durationTo(t1).nanoseconds);
        par[i] = @intCast(t1.durationTo(t2).nanoseconds);
        scan[i] = @intCast(t2.durationTo(t3).nanoseconds);
        stail[i] = @intCast(t3.durationTo(t4).nanoseconds);
    }
    std.sort.pdq(u64, &stail, {}, std.sort.asc(u64));
    std.debug.print("lu-subtree-bench: serial-tail lanes={d}us\n", .{stail[reps / 2] / 1000});
    std.mem.doNotOptimizeAway(sink);
    std.sort.pdq(u64, &scan, {}, std.sort.asc(u64));
    std.sort.pdq(u64, &ser, {}, std.sort.asc(u64));
    std.sort.pdq(u64, &par, {}, std.sort.asc(u64));
    std.debug.print("lu-subtree-bench: serial={d}us lanes={d}us L-scan={d}us speedup={d:.2}x\n", .{ ser[reps / 2] / 1000, par[reps / 2] / 1000, scan[reps / 2] / 1000, @as(f64, @floatFromInt(ser[reps / 2])) / @as(f64, @floatFromInt(par[reps / 2])) });
}

/// E1's census (dev/solvers/gpu-lu.md §5) of the factor just built,
/// all O(nnz(L+U)): F, the refactor's multiply-adds (one per U entry
/// (j, k) and row of L[:,j]); `S_r`, the §4.2 span of the sync-free
/// column kernel; the level count; the solve spans `S_L`/`S_U` as
/// longest dependency paths; the widest U column; the share of F in
/// levels narrower than 64 columns; A's rows and columns above 1,000
/// entries (rails); and the §4.1 epoch bytes.
pub fn census(self: *const SparseLu, gpa: Allocator, col_ptr: []const u32, row_idx: []const u32) Allocator.Error!void {
    const n = self.n;
    assert(col_ptr.len == @as(usize, n) + 1);
    assert(self.lp[n] == self.li.items.len);
    const scratch = try gpa.alloc(u32, 4 * @as(usize, n));
    defer gpa.free(scratch);
    const finish = scratch[0..n];
    const level = scratch[n .. 2 * n];
    const span = scratch[2 * n .. 3 * n];
    const width = scratch[3 * n ..];
    var flops: u64 = 0;
    var s_r: u32 = 0;
    var levels: u32 = 0;
    var widest: u32 = 0;
    for (0..n) |k| {
        var t: u32 = 0;
        var lv: u32 = 0;
        const col = self.ui.items[self.up[k]..self.up[k + 1]];
        for (col) |j| {
            t = @max(t, finish[j]) + 1;
            lv = @max(lv, level[j] + 1);
            flops += self.lp[j + 1] - self.lp[j];
        }
        finish[k] = t + 1;
        level[k] = lv;
        s_r = @max(s_r, t + 1);
        levels = @max(levels, lv + 1);
        widest = @max(widest, @as(u32, @intCast(col.len)));
    }
    @memset(width, 0);
    for (level) |lv| width[lv] += 1;
    var narrow: u64 = 0;
    for (0..n) |k| if (width[level[k]] < 64) {
        for (self.ui.items[self.up[k]..self.up[k + 1]]) |j| narrow += self.lp[j + 1] - self.lp[j];
    };
    // Forward solve: y[r] waits on y[k] for each L[r, k], k < r.
    @memset(span, 0);
    var s_l: u32 = 0;
    for (0..n) |k| for (self.li.items[self.lp[k]..self.lp[k + 1]]) |r| {
        span[r] = @max(span[r], span[k] + 1);
        s_l = @max(s_l, span[r]);
    };
    // Back solve: z[i] waits on z[k] for each U[i, k], i < k.
    @memset(span, 0);
    var s_u: u32 = 0;
    var k = n;
    while (k > 0) {
        k -= 1;
        for (self.ui.items[self.up[k]..self.up[k + 1]]) |i| {
            span[i] = @max(span[i], span[k] + 1);
            s_u = @max(s_u, span[i]);
        }
    }
    @memset(width, 0);
    var rail_cols: u32 = 0;
    for (0..n) |c| {
        rail_cols += @intFromBool(col_ptr[c + 1] - col_ptr[c] > 1000);
        for (row_idx[col_ptr[c]..col_ptr[c + 1]]) |r| width[r] += 1;
    }
    var rail_rows: u32 = 0;
    for (width) |w| rail_rows += @intFromBool(w > 1000);
    const nl: u64 = self.li.items.len;
    const nu: u64 = self.ui.items.len;
    const epoch = 8 * (nu + n + nl + 1) + 4 * (2 * (n + 1) + nu + (nu + 1) + flops + col_ptr[n]) + n +
        4 * (2 * (n + 1) + 2 * nl + 2 * nu) + 8 * @as(u64, n);
    std.debug.print("lu-census: F={d} S_r={d} levels={d} S_L={d} S_U={d} widest_U={d} narrow_F={d:.1}% rails={d}r/{d}c epoch={d:.1}MB\n", .{
        flops,                                                                           s_r,       levels,    s_l,                                  s_u, widest,
        100 * @as(f64, @floatFromInt(narrow)) / @as(f64, @floatFromInt(@max(flops, 1))), rail_rows, rail_cols, @as(f64, @floatFromInt(epoch)) / 1e6,
    });
}
