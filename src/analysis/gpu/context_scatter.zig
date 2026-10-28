//! The deterministic scatter of the device eval. In: every resident batch's
//! slot and row tapes. Out: the stable counting-sort permutation that sends
//! each contribution to its own staging cell, and the two-level segment
//! tables that sum each plane cell's run in the serial CPU stamp's order, so
//! a replay at the same x is bitwise the same (atomic accumulation is not:
//! parallel_inverters_2000's Vdd row moved 2.1e-10 between replays).
//! Cites #93.
//!
//! Contents:
//! - `pub const Layout = struct (`off`, `cells`)`
//! - `pub const Order = struct (`deinit`, `upload`, `build`, `chunk`)`
//! - `pub fn scratchPlanes(buf: []f64, ng: usize, nr: usize) Planes`
//! - `pub fn baseCells(gpa: std.mem.Allocator, batches: []BatchGpu, trash_slot: u32, ng: usize) ![]u32`
//! - `pub fn copyPlanes(dst: Planes, src: Planes) void`

const std = @import("std");
const assert = std.debug.assert;
const device_ir = @import("device").abi;
const context = @import("context.zig");
const ctx_batch = @import("context_batch.zig");
const Planes = device_ir.Planes;
const Host = context.Host;
const Error = context.Error;
const Stream = ctx_batch.Stream;
const BatchGpu = ctx_batch.BatchGpu;

/// The planes as the device and the pinned landing area lay them out: g, rhs,
/// c, q back to back. c and q exist on the device only when a resident batch
/// has charge.
pub const Layout = struct {
    ng: usize,
    nr: usize,

    pub fn off(self: Layout, comptime plane: enum { g, rhs, c, q }) usize {
        return switch (plane) {
            .g => 0,
            .rhs => self.ng,
            .c => self.ng + self.nr,
            .q => 2 * self.ng + self.nr,
        };
    }

    /// Cells in the first `n_planes` planes.
    pub fn cells(self: Layout, n_planes: usize) usize {
        return if (n_planes == 2) self.off(.c) else 2 * self.off(.c);
    }
};

/// The permutation and segment tables behind the deterministic scatter,
/// fused over every device plane so an eval costs one clear, two reduce
/// launches and one download.
///
/// A contribution is one tape entry, `(batch, id, ru, cu)` for g/c and
/// `(batch, id, ru)` for rhs/q, numbered in the order the serial CPU stamp
/// visits them and counting-sorted stably by destination cell. `perm[k]` is
/// k's staging cell within its plane's stage; g and c share the slot
/// permutation, rhs and q the row permutation.
///
/// Stages sit back to back (`stage_base`), each a run per plane cell and
/// then a trash tail (ground and structural-zero contributions) that no
/// range covers. Level 1 sums `chunk`-sized pieces of each cell's run,
/// level 2 each cell's pieces, both with the same kernel over `[start,
/// end)` pairs. A run of at most `chunk` is one piece and sums exactly as
/// the CPU does.
///
/// Summing the trash tails instead (one pad cell per plane) cost 104 us of
/// a 170 us mos1_2000 eval: level 2 walked the pad's 1,600 pieces on one
/// thread.
pub const Order = struct {
    perm_slot: []u32,
    perm_row: []u32,
    /// Piece `j` is stage `[seg1[2j], seg1[2j+1])`.
    seg1: []u32,
    /// Device plane cell `i` is pieces `[seg2[2i], seg2[2i+1])`.
    seg2: []u32,
    stage_base: [4]usize,
    n_stage: usize,
    n_pieces: usize,
    /// The first piece of plane 3 (q), `n_pieces` with two planes.
    q_piece0: usize,
    /// Baseline cells past `n_stage`: `base_cells.len` per slot plane.
    n_base: usize,

    /// Contributions per level-1 piece: an 8000-deep row becomes 125
    /// threads of 64, then one thread of 125.
    pub const chunk: u32 = 64;

    pub fn deinit(self: Order, gpa: std.mem.Allocator) void {
        gpa.free(self.perm_slot);
        gpa.free(self.perm_row);
        gpa.free(self.seg1);
        gpa.free(self.seg2);
    }

    /// Overwrites each batch's resident slot and row tapes with its slice
    /// of the permutation, keeping their length and layout.
    /// On `stream`, so a query starting beside another's device work
    /// does not fence it; the driver stages the pageable tables before
    /// returning, so `deinit` may follow at once.
    pub fn upload(self: Order, batches: []BatchGpu, stream: *Stream) !void {
        var off_s: usize = 0;
        var off_r: usize = 0;
        for (batches) |*bg| {
            const p = bg.payload(bg.ctx);
            // A constant batch's kernel reads no slot tape.
            const n_slots = if (bg.const_kernel != null) 0 else p.slots.len;
            if (n_slots > 0)
                try bg.d_slots.uploadAtAsync(self.perm_slot[off_s..].ptr, 0, n_slots * @sizeOf(u32), stream);
            if (p.rhs_idx.len > 0)
                try bg.d_rhs_idx.uploadAtAsync(self.perm_row[off_r..].ptr, 0, p.rhs_idx.len * @sizeOf(u32), stream);
            off_s += n_slots;
            off_r += p.rhs_idx.len;
        }
    }

    /// `slot_tapes[b]`/`row_tapes[b]` are resident batch b's `slots` and
    /// `rhs_idx`, in `batches` order; `trash_slot`/`trash_row` the
    /// cells that absorb ground and dead entries. Caller frees with
    /// `deinit`.
    /// `base_cells` (ascending slot cells) each get one more piece, ahead
    /// of their run in g and c: baseline value j of plane g at stage
    /// `n_stage + j`, of plane c at `n_stage + base_cells.len + j`.
    pub fn build(gpa: std.mem.Allocator, slot_tapes: []const []const u32, row_tapes: []const []const u32, trash_slot: u32, trash_row: u32, layout: Layout, n_planes: usize, base_cells: []const u32) !Order {
        std.debug.assert(slot_tapes.len == row_tapes.len);
        std.debug.assert(n_planes == 2 or n_planes == 4);
        var n_slot: usize = 0;
        for (slot_tapes) |t| n_slot += t.len;
        var n_row: usize = 0;
        for (row_tapes) |t| n_row += t.len;
        const n_stage = if (n_planes == 2) n_slot + n_row else 2 * (n_slot + n_row);
        // The u32 tapes and seg tables index the stage, and a piece's end
        // is computed as `at + chunk`. Past u32 fall back to the CPU,
        // never truncate or wrap.
        const n_base = base_cells.len * (n_planes / 2);
        if (n_stage + n_base > std.math.maxInt(u32) - chunk) return Error.CircuitNotEligible;

        const perm_slot = try gpa.alloc(u32, n_slot);
        errdefer gpa.free(perm_slot);
        const perm_row = try gpa.alloc(u32, n_row);
        errdefer gpa.free(perm_row);
        // Cell -> start of its run within the space's stage.
        const seg_slot = try gpa.alloc(u32, layout.ng + 1);
        defer gpa.free(seg_slot);
        const seg_row = try gpa.alloc(u32, layout.nr + 1);
        defer gpa.free(seg_row);
        permute(slot_tapes, trash_slot, seg_slot, perm_slot);
        permute(row_tapes, trash_row, seg_row, perm_row);
        const spaces = [2][]const u32{ seg_slot, seg_row };

        // g, rhs, c, q: slot, row, slot, row.
        var stage_base: [4]usize = .{ 0, n_slot, n_slot + n_row, 2 * n_slot + n_row };
        if (n_planes == 2) stage_base[2..].* = .{ 0, 0 };
        var pieces_per_pair: usize = 0;
        for (spaces) |seg| for (seg[0 .. seg.len - 1], seg[1..]) |a, b| {
            pieces_per_pair += (b - a + chunk - 1) / chunk;
        };
        const n_pieces = pieces_per_pair * (n_planes / 2) + n_base;
        const seg1 = try gpa.alloc(u32, 2 * n_pieces);
        errdefer gpa.free(seg1);
        const n_cells = layout.cells(n_planes);
        const seg2 = try gpa.alloc(u32, 2 * n_cells);
        errdefer gpa.free(seg2);
        var cell: usize = 0;
        var piece: u32 = 0;
        var q_piece0: usize = n_pieces;
        for (0..n_planes) |plane| {
            if (plane == 3) q_piece0 = piece;
            const seg = spaces[plane % 2];
            const base: u32 = @intCast(stage_base[plane]);
            // Plane g's baseline values, then plane c's.
            const base_at: u32 = @intCast(n_stage + (plane / 2) * base_cells.len);
            var next_base: usize = 0;
            for (0..seg.len - 1) |i| {
                seg2[2 * cell] = piece;
                if (plane % 2 == 0 and next_base < base_cells.len and base_cells[next_base] == i) {
                    const at = base_at + @as(u32, @intCast(next_base));
                    seg1[2 * piece ..][0..2].* = .{ at, at + 1 };
                    piece += 1;
                    next_base += 1;
                }
                var at = seg[i];
                while (at < seg[i + 1]) : (at += chunk) {
                    seg1[2 * piece ..][0..2].* = .{ base + at, base + @min(at + chunk, seg[i + 1]) };
                    piece += 1;
                }
                seg2[2 * cell + 1] = piece;
                cell += 1;
            }
        }
        std.debug.assert(cell == n_cells and piece == n_pieces);
        return .{
            .perm_slot = perm_slot,
            .perm_row = perm_row,
            .seg1 = seg1,
            .seg2 = seg2,
            .stage_base = stage_base,
            .n_stage = n_stage,
            .n_pieces = n_pieces,
            .q_piece0 = q_piece0,
            .n_base = n_base,
        };
    }

    /// Counting-sorts one space's contributions by destination cell into
    /// `perm` and fills `seg` (cell -> run start, `seg[n_cells]` = run
    /// end). Trash contributions go to the tail after the runs, each on
    /// its own cell: the kernel stores through the tape unconditionally,
    /// and a shared cell would be a contended write.
    fn permute(tapes: []const []const u32, trash: u32, seg: []u32, perm: []u32) void {
        std.debug.assert(seg.len >= 1);
        @memset(seg, 0);
        for (tapes) |tape| for (tape) |dest| {
            if (dest != trash) seg[dest + 1] += 1;
        };
        var run: u32 = 0;
        for (seg) |*s| {
            run += s.*;
            s.* = run;
        }
        // `seg[c]` is cell c's cursor: it ends at cell c + 1's start,
        // so the shift below restores the starts.
        var tail: u32 = run;
        var k: usize = 0;
        for (tapes) |tape| for (tape) |dest| {
            if (dest == trash) {
                perm[k] = tail;
                tail += 1;
            } else {
                perm[k] = seg[dest];
                seg[dest] += 1;
            }
            k += 1;
        };
        std.mem.copyBackwards(u32, seg[1..], seg[0 .. seg.len - 1]);
        seg[0] = 0;
        std.debug.assert(k == perm.len and tail == perm.len);
    }
};

/// Host planes g, c, rhs, q back to back at the start of `buf`, for the
/// host stamps into scratch (the cost probe, the baseline, the fault path).
pub fn scratchPlanes(buf: []f64, ng: usize, nr: usize) Planes {
    return .{ .g_vals = buf[0..ng], .c_vals = buf[ng..][0..ng], .rhs = buf[2 * ng ..][0..nr], .q_vec = buf[2 * ng + nr ..][0..nr] };
}

/// The slot cells the constant batches stamp, ascending, trash excluded.
/// Caller frees.
pub fn baseCells(gpa: std.mem.Allocator, batches: []BatchGpu, trash_slot: u32, ng: usize) ![]u32 {
    const hit = try gpa.alloc(bool, ng);
    defer gpa.free(hit);
    @memset(hit, false);
    var n: usize = 0;
    for (batches) |*bg| {
        if (bg.const_kernel == null) continue;
        for (bg.payload(bg.ctx).slots) |cell| if (cell != trash_slot and !hit[cell]) {
            hit[cell] = true;
            n += 1;
        };
    }
    const cells = try gpa.alloc(u32, n);
    var k: usize = 0;
    for (hit, 0..) |h, i| if (h) {
        cells[k] = @intCast(i);
        k += 1;
    };
    return cells;
}

pub fn copyPlanes(dst: Planes, src: Planes) void {
    @memcpy(dst.g_vals, src.g_vals);
    @memcpy(dst.c_vals, src.c_vals);
    @memcpy(dst.rhs, src.rhs);
    @memcpy(dst.q_vec, src.q_vec);
}

// ---- private ----

// TESTHELPER
fn buildOrderAndFree(gpa: std.mem.Allocator, slot_tapes: []const []const u32, row_tapes: []const []const u32) !void {
    const order = try Order.build(gpa, slot_tapes, row_tapes, 2, 1, .{ .ng = 3, .nr = 2 }, 4, &.{1});
    order.deinit(gpa);
}

// TESTHELPER
/// Host model of the two `arp_reduce_*` levels over `order`'s tables: the
/// device planes an eval would download from `stage`.
fn reduceOnHost(gpa: std.mem.Allocator, order: Order, stage: []const f64, planes: []f64) !void {
    const mid = try gpa.alloc(f64, order.n_pieces);
    defer gpa.free(mid);
    for (mid, 0..) |*m, j| {
        m.* = 0;
        for (stage[order.seg1[2 * j]..order.seg1[2 * j + 1]]) |v| m.* += v;
    }
    for (planes, 0..) |*c, i| {
        c.* = 0;
        for (mid[order.seg2[2 * i]..order.seg2[2 * i + 1]]) |v| c.* += v;
    }
}

test "Layout puts g, rhs, c, q back to back" {
    const l: Layout = .{ .ng = 5, .nr = 3 };
    try std.testing.expectEqual(@as(usize, 0), l.off(.g));
    try std.testing.expectEqual(@as(usize, 5), l.off(.rhs));
    try std.testing.expectEqual(@as(usize, 8), l.off(.c));
    try std.testing.expectEqual(@as(usize, 13), l.off(.q));
    try std.testing.expectEqual(@as(usize, 8), l.cells(2));
    try std.testing.expectEqual(@as(usize, 16), l.cells(4));
}

test "Order reduces each cell in the serial stamp's order" {
    const gpa = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x5eed);
    const rnd = prng.random();
    // Run lengths across the level-1 chunk: empty, one, chunk +- 1, two
    // chunks +- 1. The last cell of each space is its trash cell.
    const slot_runs = [_]u32{ 0, 1, 63, 64, 65, 127, 128, 129, 2 };
    const row_runs = [_]u32{ 3, 0, 64, 130, 1 };
    const layout: Layout = .{ .ng = slot_runs.len + 1, .nr = row_runs.len + 1 };
    const trash_slot: u32 = slot_runs.len;
    const trash_row: u32 = row_runs.len;

    var spaces: [2][]u32 = undefined;
    for (&spaces, [_][]const u32{ &slot_runs, &row_runs }, [_]u32{ trash_slot, trash_row }) |*space, runs, trash| {
        var dests: std.ArrayList(u32) = .empty;
        errdefer dests.deinit(gpa);
        for (runs, 0..) |r, cell| try dests.appendNTimes(gpa, @intCast(cell), r);
        try dests.appendNTimes(gpa, trash, 7);
        rnd.shuffle(u32, dests.items);
        space.* = try dests.toOwnedSlice(gpa);
    }
    defer for (spaces) |sp| gpa.free(sp);
    // Three batches, the middle one empty in both spaces.
    const cut_s = spaces[0].len / 3;
    const cut_r = spaces[1].len / 2;
    const slot_tapes = [_][]const u32{ spaces[0][0..cut_s], &.{}, spaces[0][cut_s..] };
    const row_tapes = [_][]const u32{ spaces[1][0..cut_r], &.{}, spaces[1][cut_r..] };

    for ([_]usize{ 2, 4 }) |n_planes| {
        const order = try Order.build(gpa, &slot_tapes, &row_tapes, trash_slot, trash_row, layout, n_planes, &.{});
        defer order.deinit(gpa);
        const n_cells = layout.cells(n_planes);
        const plane_offs = [4]usize{ layout.off(.g), layout.off(.rhs), layout.off(.c), layout.off(.q) };
        // Every piece is one nonempty chunk of one cell's run.
        var pieces: usize = 0;
        for (slot_runs ++ [_]u32{0}) |r| pieces += (r + Order.chunk - 1) / Order.chunk;
        for (row_runs ++ [_]u32{0}) |r| pieces += (r + Order.chunk - 1) / Order.chunk;
        try std.testing.expectEqual(pieces * (n_planes / 2), order.n_pieces);

        // Each plane's contributions in tape order, scattered as the eval
        // kernels do (`perm`), against the serial stamp's `+=`.
        const stage = try gpa.alloc(f64, order.n_stage);
        defer gpa.free(stage);
        // Only the trash tails keep the NaN; a piece over them would read it.
        @memset(stage, std.math.nan(f64));
        const serial = try gpa.alloc(f64, n_cells);
        defer gpa.free(serial);
        @memset(serial, 0);
        for (0..n_planes) |plane| {
            const is_slot = plane % 2 == 0;
            const tape = spaces[if (is_slot) 0 else 1];
            const perm = if (is_slot) order.perm_slot else order.perm_row;
            const trash = if (is_slot) trash_slot else trash_row;
            const plane_off = plane_offs[plane];
            for (tape, perm) |dest, at| {
                const v = rnd.float(f64) * 2 - 1;
                stage[order.stage_base[plane] + at] = v;
                if (dest != trash) serial[plane_off + dest] += v;
            }
        }
        const planes = try gpa.alloc(f64, n_cells);
        defer gpa.free(planes);
        try reduceOnHost(gpa, order, stage, planes);

        for (0..n_planes) |plane| {
            const runs: []const u32 = if (plane % 2 == 0) &(slot_runs ++ [_]u32{0}) else &(row_runs ++ [_]u32{0});
            const plane_off = plane_offs[plane];
            for (runs, 0..) |r, cell| {
                const got = planes[plane_off + cell];
                const want = serial[plane_off + cell];
                if (r <= Order.chunk) {
                    // One piece: the CPU's sum, bit for bit.
                    try std.testing.expectEqual(@as(u64, @bitCast(want)), @as(u64, @bitCast(got)));
                } else {
                    // Two levels reassociate: the classic r^2 eps bound on |v| <= 1.
                    const rf: f64 = @floatFromInt(r);
                    try std.testing.expectApproxEqAbs(want, got, rf * rf * std.math.floatEps(f64));
                }
            }
        }
    }
}

test "Order.build puts each baseline value ahead of its cell's run" {
    const gpa = std.testing.allocator;
    // Slot cells 0..2 (2 is trash), row cell 0 (1 is trash); base on cells 0 and 1.
    const slots = [_]u32{ 1, 0, 1, 2 };
    const rows = [_]u32{ 0, 0 };
    const order = try Order.build(gpa, &.{&slots}, &.{&rows}, 2, 1, .{ .ng = 3, .nr = 2 }, 4, &.{ 0, 1 });
    defer order.deinit(gpa);
    try std.testing.expectEqual(@as(usize, 4), order.n_base);
    const stage = try gpa.alloc(f64, order.n_stage + order.n_base);
    defer gpa.free(stage);
    @memset(stage, 0);
    // Contributions: g gets 1, 2, 4 on cells 1, 0, 1; c gets 8, 16, 32.
    for (slots, order.perm_slot, 0..) |dest, at, k| if (dest != 2) {
        stage[order.stage_base[0] + at] = @floatFromInt(@as(u32, 1) << @intCast(k));
        stage[order.stage_base[2] + at] = @floatFromInt(@as(u32, 8) << @intCast(k));
    };
    // Baselines: g 100, 200 and c 300, 400 on cells 0 and 1.
    stage[order.n_stage..][0..4].* = .{ 100, 200, 300, 400 };
    const planes = try gpa.alloc(f64, (Layout{ .ng = 3, .nr = 2 }).cells(4));
    defer gpa.free(planes);
    try reduceOnHost(gpa, order, stage, planes);
    // g: 0 1 2 = 102, 205, 0; rhs; c at offset 5: 316, 440, 0.
    try std.testing.expectEqualSlices(f64, &.{ 102, 205, 0 }, planes[0..3]);
    try std.testing.expectEqualSlices(f64, &.{ 316, 440, 0 }, planes[5..8]);
}

test "Order.permute is a stable counting sort with trash in the tail" {
    const trash: u32 = 3;
    const tapes = [_][]const u32{ &.{ 2, 3, 0, 2 }, &.{}, &.{ 3, 0, 2 } };
    var seg: [5]u32 = undefined;
    var perm: [7]u32 = undefined;
    Order.permute(&tapes, trash, &seg, &perm);
    // Cells 0, 1, 2, trash: runs of 2, 0, 3, 0; the two trash entries after.
    try std.testing.expectEqualSlices(u32, &.{ 0, 2, 2, 5, 5 }, &seg);
    try std.testing.expectEqualSlices(u32, &.{ 2, 5, 0, 3, 6, 1, 4 }, &perm);
}

test "Order.permute on empty tapes" {
    const tapes = [_][]const u32{ &.{}, &.{} };
    var seg: [3]u32 = undefined;
    var perm: [0]u32 = .{};
    Order.permute(&tapes, 1, &seg, &perm);
    try std.testing.expectEqualSlices(u32, &.{ 0, 0, 0 }, &seg);
}

test "Order.permute: random tapes against the stable-sort ranks" {
    var prng = std.Random.DefaultPrng.init(7);
    const rnd = prng.random();
    var dests: [300]u32 = undefined;
    var perm: [300]u32 = undefined;
    var seg: [10]u32 = undefined; // nine cells
    for (0..200) |_| {
        const len = rnd.uintLessThan(usize, dests.len + 1);
        const trash = rnd.uintLessThan(u32, 9);
        for (dests[0..len]) |*d| d.* = rnd.uintLessThan(u32, 9);
        const cut = rnd.uintLessThan(usize, len + 1);
        const all = dests[0..len];
        Order.permute(&.{ all[0..cut], all[cut..] }, trash, &seg, perm[0..len]);
        var n_live: u32 = 0;
        for (all) |d| n_live += @intFromBool(d != trash);
        // A live entry's rank: live entries of lower cells, then earlier
        // entries of its own. Trash ranks after every live entry, in order.
        for (all, 0..) |d, k| {
            var want: u32 = if (d == trash) n_live else 0;
            if (d != trash) for (all) |e| {
                want += @intFromBool(e != trash and e < d);
            };
            for (all[0..k]) |e| want += @intFromBool(e == d);
            try std.testing.expectEqual(want, perm[k]);
        }
        for (seg, 0..) |s, c| {
            var want: u32 = 0;
            for (all) |e| want += @intFromBool(e != trash and e < c);
            try std.testing.expectEqual(want, s);
        }
    }
}

test "Order.build frees everything on every allocation failure" {
    // Cell 0's run spans two pieces.
    var slots: [160]u32 = undefined;
    for (&slots, 0..) |*d, i| d.* = ([_]u32{ 0, 2, 1, 0 })[i % 4];
    const rows = [_]u32{ 0, 1, 0 };
    try std.testing.checkAllAllocationFailures(std.testing.allocator, buildOrderAndFree, .{ &[_][]const u32{ &slots, &.{} }, &[_][]const u32{ &rows, &.{} } });
}

test "Order.build: no contributions, and a stage past u32" {
    const gpa = std.testing.allocator;
    const order = try Order.build(gpa, &.{&.{}}, &.{&.{}}, 1, 0, .{ .ng = 2, .nr = 1 }, 4, &.{});
    defer order.deinit(gpa);
    try std.testing.expectEqual(@as(usize, 0), order.n_stage);
    try std.testing.expectEqual(@as(usize, 0), order.n_pieces);
    // Every cell is an empty range of pieces.
    try std.testing.expectEqualSlices(u32, &@as([12]u32, @splat(0)), order.seg2);

    // Only the lengths are read before the bar, so a dangling base is safe.
    const huge = @as([*]const u32, @ptrFromInt(@alignOf(u32)))[0 .. std.math.maxInt(u32) - Order.chunk + 1];
    try std.testing.expectError(Error.CircuitNotEligible, Order.build(gpa, &.{huge}, &.{&.{}}, 0, 0, .{ .ng = 1, .nr = 1 }, 2, &.{}));
}
