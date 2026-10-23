//! Thread-per-(instance, lane) evaluation for GPU lanes (performance TODO 06
//! 6.11, opt-in `ESPICE_GPU_LANE_COLOR`). Thread-per-lane walks a batch's
//! instances in order, which holds only N threads per batch: 10k lanes fill
//! a laptop part but about 30 % of an H100. Here one launch per colour runs
//! a thread per (instance of that colour, lane).
//!
//! Exactness: the host adds a batch's stamps into each cell in instance
//! order, and within an instance in the eval's scatter order (rows
//! ascending, columns ascending: the slot table's order). Colours are levels
//! in instance order: an instance takes one colour past every earlier
//! instance sharing a cell with it, so each cell's instances come in
//! increasing colours and colours launch in order. A hub cell (touched by
//! more than `hub_min` instances) would make one colour per toucher;
//! instead each of its touches gets a staging cell of its own, which the
//! eval writes once (-0.0 + v = v), and a reduce adds them into the hub in
//! (instance, entry) order, the host's order, then resets them to -0.0
//! (x + -0.0 = x, so an entry the eval skipped changes nothing, not even a
//! zero's sign). The trash slot and the ground row take no part: their
//! writes race, and nothing reads them.

const std = @import("std");

/// One batch's evaluation order and remapped tables.
pub const Coloring = struct {
    /// Instance ids by colour: colour c is `order[starts[c]..starts[c + 1]]`.
    order: []u32,
    starts: []u32,
    /// The slot and row tables with every hub touch moved to its staging
    /// cell (`stage_slot0 + k`, `stage_row0 + k`).
    slots: []u32,
    rows: []u32,
    slot_hubs: Hubs,
    row_hubs: Hubs,
    n_slot_stage: u32,
    n_row_stage: u32,

    /// Hub h's cell `cells[h]` takes its staging cells
    /// `entries[starts[h]..starts[h + 1]]` in that order.
    pub const Hubs = struct {
        cells: []u32,
        starts: []u32,
        entries: []u32,

        fn deinit(h: Hubs, gpa: std.mem.Allocator) void {
            gpa.free(h.cells);
            gpa.free(h.starts);
            gpa.free(h.entries);
        }
    };

    pub fn deinit(c: Coloring, gpa: std.mem.Allocator) void {
        gpa.free(c.order);
        gpa.free(c.starts);
        gpa.free(c.slots);
        gpa.free(c.rows);
        c.slot_hubs.deinit(gpa);
        c.row_hubs.deinit(gpa);
    }

    pub fn colors(c: Coloring) usize {
        return c.starts.len - 1;
    }
};

/// Cells touched by more than this many instances of a batch are hubs.
// ponytail: a fixed bound; tune per card once a datacenter part can measure it.
pub const hub_min = 4;

/// The colouring of a batch of `count` instances whose slot table holds
/// `ks` entries per instance (cells < `trash_slot`, the trash slot itself
/// ignored) and row table `n_u` per instance (rows < `ground_row`, the
/// ground row ignored). Staging cells start at `stage_slot0` and
/// `stage_row0`, so batches sharing planes get disjoint ranges.
pub fn color(gpa: std.mem.Allocator, count: u32, ks: u32, slots: []const u32, n_u: u32, rows: []const u32, trash_slot: u32, ground_row: u32, stage_slot0: u32, stage_row0: u32) !Coloring {
    std.debug.assert(slots.len == @as(usize, count) * ks and rows.len == @as(usize, count) * n_u);
    var out: Coloring = undefined;
    const slot_hub = try hubMask(gpa, count, ks, slots, trash_slot);
    defer gpa.free(slot_hub);
    const row_hub = try hubMask(gpa, count, n_u, rows, ground_row);
    defer gpa.free(row_hub);
    out.slots = try gpa.dupe(u32, slots);
    errdefer gpa.free(out.slots);
    out.rows = try gpa.dupe(u32, rows);
    errdefer gpa.free(out.rows);
    out.slot_hubs, out.n_slot_stage = try stage(gpa, slot_hub, out.slots, stage_slot0);
    errdefer out.slot_hubs.deinit(gpa);
    out.row_hubs, out.n_row_stage = try stage(gpa, row_hub, out.rows, stage_row0);
    errdefer out.row_hubs.deinit(gpa);

    // Levels over the cells left: one past the last colour on each.
    const level = try gpa.alloc(u32, count);
    defer gpa.free(level);
    const last_slot = try gpa.alloc(u32, trash_slot);
    defer gpa.free(last_slot);
    const last_row = try gpa.alloc(u32, ground_row);
    defer gpa.free(last_row);
    @memset(last_slot, 0);
    @memset(last_row, 0);
    var n_colors: u32 = 0;
    for (level, 0..) |*lv, i| {
        var c: u32 = 0;
        for (out.slots[i * ks ..][0..ks]) |s| if (s < trash_slot) {
            c = @max(c, last_slot[s]);
        };
        for (out.rows[i * n_u ..][0..n_u]) |r| if (r < ground_row) {
            c = @max(c, last_row[r]);
        };
        lv.* = c + 1;
        for (out.slots[i * ks ..][0..ks]) |s| if (s < trash_slot) {
            last_slot[s] = lv.*;
        };
        for (out.rows[i * n_u ..][0..n_u]) |r| if (r < ground_row) {
            last_row[r] = lv.*;
        };
        n_colors = @max(n_colors, lv.*);
    }
    // Counting sort by colour, instance order kept.
    out.starts = try gpa.alloc(u32, n_colors + 1);
    errdefer gpa.free(out.starts);
    @memset(out.starts, 0);
    for (level) |lv| out.starts[lv] += 1;
    for (1..out.starts.len) |c| out.starts[c] += out.starts[c - 1];
    out.order = try gpa.alloc(u32, count);
    const fill = try gpa.dupe(u32, out.starts[0..n_colors]);
    defer gpa.free(fill);
    for (level, 0..) |lv, i| {
        out.order[fill[lv - 1]] = @intCast(i);
        fill[lv - 1] += 1;
    }
    return out;
}

/// `mask[cell]`: cell (below `trash`) is touched by more than `hub_min`
/// instances.
fn hubMask(gpa: std.mem.Allocator, count: u32, k: u32, table: []const u32, trash: u32) ![]bool {
    const touches = try gpa.alloc(u32, trash);
    defer gpa.free(touches);
    const seen = try gpa.alloc(u32, trash);
    defer gpa.free(seen);
    @memset(touches, 0);
    @memset(seen, std.math.maxInt(u32));
    for (0..count) |i| for (table[i * k ..][0..k]) |cell| if (cell < trash and seen[cell] != i) {
        seen[cell] = @intCast(i);
        touches[cell] += 1;
    };
    const mask = try gpa.alloc(bool, trash);
    for (mask, touches) |*m, t| m.* = t > hub_min;
    return mask;
}

/// Moves every hub entry of `table` to the next staging cell from `base`,
/// in table order, and lists each hub's staging cells in that order.
fn stage(gpa: std.mem.Allocator, hub: []const bool, table: []u32, base: u32) !struct { Coloring.Hubs, u32 } {
    var n_hubs: u32 = 0;
    const index = try gpa.alloc(u32, hub.len);
    defer gpa.free(index);
    for (hub, index) |h, *ix| {
        ix.* = n_hubs;
        n_hubs += @intFromBool(h);
    }
    var hubs: Coloring.Hubs = .{
        .cells = try gpa.alloc(u32, n_hubs),
        .starts = try gpa.alloc(u32, n_hubs + 1),
        .entries = undefined,
    };
    errdefer {
        gpa.free(hubs.cells);
        gpa.free(hubs.starts);
    }
    @memset(hubs.starts, 0);
    var n_stage: u32 = 0;
    for (table) |cell| if (cell < hub.len and hub[cell]) {
        hubs.starts[index[cell] + 1] += 1;
        n_stage += 1;
    };
    for (hub, index, 0..) |h, ix, cell| if (h) {
        hubs.cells[ix] = @intCast(cell);
    };
    for (1..hubs.starts.len) |h| hubs.starts[h] += hubs.starts[h - 1];
    hubs.entries = try gpa.alloc(u32, n_stage);
    const fill = try gpa.dupe(u32, hubs.starts[0..n_hubs]);
    defer gpa.free(fill);
    var k: u32 = 0;
    for (table) |*cell| if (cell.* < hub.len and hub[cell.*]) {
        const h = index[cell.*];
        cell.* = base + k;
        hubs.entries[fill[h]] = base + k;
        fill[h] += 1;
        k += 1;
    };
    return .{ hubs, n_stage };
}

test "lane colouring: per-cell sums are bitwise the instance-order walk" {
    const gpa = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x6c11);
    const r = prng.random();
    for (0..40) |_| {
        const count = r.intRangeAtMost(u32, 1, 60);
        const ks = r.intRangeAtMost(u32, 1, 9);
        const n_u = r.intRangeAtMost(u32, 1, 4);
        const nnz = r.intRangeAtMost(u32, 1, 30);
        const n = r.intRangeAtMost(u32, 1, 12);
        const slots = try gpa.alloc(u32, count * ks);
        defer gpa.free(slots);
        const rows = try gpa.alloc(u32, count * n_u);
        defer gpa.free(rows);
        // Cell 0 and row 0 are hubs most of the time; nnz and n are trash.
        for (slots) |*s| s.* = if (r.boolean()) 0 else r.intRangeAtMost(u32, 0, nnz);
        for (rows) |*x| x.* = if (r.boolean()) 0 else r.intRangeAtMost(u32, 0, n);
        const vs = try gpa.alloc(f64, slots.len);
        defer gpa.free(vs);
        for (vs) |*v| v.* = (r.float(f64) - 0.5) * std.math.pow(f64, 10, @floatFromInt(r.intRangeAtMost(i32, -8, 8)));
        // The host: instance order, table order, starting from a prior value.
        const want = try gpa.alloc(f64, nnz + 1);
        defer gpa.free(want);
        for (want, 0..) |*w, cell| w.* = @floatFromInt(cell);
        for (slots, vs) |s, v| want[s] += v;
        const c = try color(gpa, count, ks, slots, n_u, rows, nnz, n, nnz + 1, n + 1);
        defer c.deinit(gpa);
        const got = try gpa.alloc(f64, nnz + 1 + c.n_slot_stage);
        defer gpa.free(got);
        for (got[0 .. nnz + 1], 0..) |*g, cell| g.* = @floatFromInt(cell);
        @memset(got[nnz + 1 ..], -0.0);
        // Colours in order, instances of a colour in reverse, as a race may.
        for (0..c.colors()) |col| {
            const ids = c.order[c.starts[col]..c.starts[col + 1]];
            var j = ids.len;
            while (j > 0) {
                j -= 1;
                const i = ids[j];
                for (c.slots[i * ks ..][0..ks], vs[i * ks ..][0..ks]) |s, v| got[s] += v;
            }
            // No two instances of a colour share a non-hub, non-trash cell.
            for (ids, 0..) |a, x| for (ids[x + 1 ..]) |b| for (c.slots[a * ks ..][0..ks]) |sa| if (sa < nnz) for (c.slots[b * ks ..][0..ks]) |sb| {
                try std.testing.expect(sa != sb);
            };
        }
        for (c.slot_hubs.cells, 0..) |cell, h| for (c.slot_hubs.entries[c.slot_hubs.starts[h]..c.slot_hubs.starts[h + 1]]) |e| {
            got[cell] += got[e];
            got[e] = -0.0;
        };
        for (0..nnz) |cell| try std.testing.expectEqual(@as(u64, @bitCast(want[cell])), @as(u64, @bitCast(got[cell])));
        for (got[nnz + 1 ..]) |g| try std.testing.expectEqual(@as(u64, @bitCast(@as(f64, -0.0))), @as(u64, @bitCast(g)));
    }
}
