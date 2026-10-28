//! Thread-per-(instance, lane) evaluation order for GPU lanes. In: one
//! batch's slot and row scatter tables. Out: its instances grouped into
//! colours that launch in order, and the tables with every hub touch moved
//! to a staging cell of its own. Thread-per-lane walks a batch's instances
//! in order, which holds only N threads per batch: 10k lanes fill a laptop
//! part but about 30 % of an H100. One launch per colour runs a thread per
//! (instance of that colour, lane).
//!
//! Exactness: the host adds a batch's stamps into each cell in instance
//! order, and within an instance in the eval's scatter order (rows
//! ascending, columns ascending: the slot table's order). Colours are levels
//! in instance order: an instance takes one colour past every earlier
//! instance sharing a cell with it, so each cell's instances come in
//! increasing colours and colours launch in order. A hub cell (touched by
//! more than `hub_min` instances) would make one colour per toucher;
//! instead each of its touches gets a staging cell, which the eval writes
//! once (-0.0 + v = v), and a reduce adds them into the hub in (instance,
//! entry) order, the host's order, then resets them to -0.0 (x + -0.0 = x,
//! so an entry the eval skipped changes nothing, not even a zero's sign).
//! The trash slot and the ground row take no part: their writes race, and
//! nothing reads them.
//! Cites: dev/research/plans/phase4-gpu-lanes.md 06 6.11 and 6.12.
//!
//! Contents:
//! - `pub const Tables = struct`
//! - `pub const Coloring = struct` with `pub const Hubs`, `pub fn deinit`, `pub fn colors`
//! - `pub fn color(gpa: std.mem.Allocator, t: Tables) std.mem.Allocator.Error!Coloring`

const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;

/// One batch's scatter tables, borrowed, and where its staging cells start.
pub const Tables = struct {
    count: u32,
    /// Slot entries per instance; `slots` holds `count * ks`, each below
    /// `trash_slot` or equal to it (ignored).
    ks: u32,
    slots: []const u32,
    /// Row entries per instance; `rows` holds `count * n_u`, each below
    /// `ground_row` or equal to it (ignored).
    n_u: u32,
    rows: []const u32,
    trash_slot: u32,
    ground_row: u32,
    /// First staging cells, so batches sharing planes get disjoint ranges.
    stage_slot0: u32,
    stage_row0: u32,
};

/// One batch's evaluation order and remapped tables. Every slice is owned;
/// free with `deinit`.
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

        // ---- private ----

        fn deinit(h: Hubs, gpa: Allocator) void {
            gpa.free(h.cells);
            gpa.free(h.starts);
            gpa.free(h.entries);
        }
    };

    /// Frees every table `color` allocated in `gpa`.
    pub fn deinit(c: Coloring, gpa: Allocator) void {
        gpa.free(c.order);
        gpa.free(c.starts);
        gpa.free(c.slots);
        gpa.free(c.rows);
        c.slot_hubs.deinit(gpa);
        c.row_hubs.deinit(gpa);
    }

    /// The number of colours, which launch in order 0, 1, ...
    pub fn colors(c: Coloring) usize {
        assert(c.starts.len >= 1);
        return c.starts.len - 1;
    }
};

/// Cells touched by more than this many instances of a batch are hubs.
// ponytail: a fixed bound; tune per card once a datacenter part can measure it.
const hub_min = 4;

/// The colouring of batch `t`. Caller owns the result and frees it with
/// `Coloring.deinit` and the same `gpa`; on error nothing is left allocated.
/// Asserts that the table lengths match `count`. O(count * (ks + n_u)).
pub fn color(gpa: Allocator, t: Tables) Allocator.Error!Coloring {
    assert(t.slots.len == @as(usize, t.count) * t.ks);
    assert(t.rows.len == @as(usize, t.count) * t.n_u);
    const slot_hub = try hubMask(gpa, t.count, t.ks, t.slots, t.trash_slot);
    defer gpa.free(slot_hub);
    const row_hub = try hubMask(gpa, t.count, t.n_u, t.rows, t.ground_row);
    defer gpa.free(row_hub);
    const slots = try gpa.dupe(u32, t.slots);
    errdefer gpa.free(slots);
    const rows = try gpa.dupe(u32, t.rows);
    errdefer gpa.free(rows);
    const slot_hubs, const n_slot_stage = try stage(gpa, slot_hub, slots, t.stage_slot0);
    errdefer slot_hubs.deinit(gpa);
    const row_hubs, const n_row_stage = try stage(gpa, row_hub, rows, t.stage_row0);
    errdefer row_hubs.deinit(gpa);

    const level = try gpa.alloc(u32, t.count);
    defer gpa.free(level);
    const n_colors = try levels(gpa, t, slots, rows, level);
    const starts, const order = try countingSort(gpa, level, n_colors);
    assert(starts[n_colors] == t.count);
    return .{
        .order = order,
        .starts = starts,
        .slots = slots,
        .rows = rows,
        .slot_hubs = slot_hubs,
        .row_hubs = row_hubs,
        .n_slot_stage = n_slot_stage,
        .n_row_stage = n_row_stage,
    };
}

// ---- private ----

/// `mask[cell]`: cell (below `trash`) is touched by more than `hub_min`
/// instances. Caller owns the mask.
fn hubMask(gpa: Allocator, count: u32, k: u32, table: []const u32, trash: u32) Allocator.Error![]bool {
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
    for (mask, touches) |*m, n| m.* = n > hub_min;
    return mask;
}

/// Moves every hub entry of `table` to the next staging cell from `base`,
/// in table order, and lists each hub's staging cells in that order.
/// Returns the hubs (caller owns them) and the staging cells used.
fn stage(gpa: Allocator, hub: []const bool, table: []u32, base: u32) Allocator.Error!struct { Coloring.Hubs, u32 } {
    var n_hubs: u32 = 0;
    const index = try gpa.alloc(u32, hub.len);
    defer gpa.free(index);
    for (hub, index) |h, *ix| {
        ix.* = n_hubs;
        n_hubs += @intFromBool(h);
    }
    const cells = try gpa.alloc(u32, n_hubs);
    errdefer gpa.free(cells);
    const starts = try gpa.alloc(u32, n_hubs + 1);
    errdefer gpa.free(starts);
    @memset(starts, 0);
    var n_stage: u32 = 0;
    for (table) |cell| if (cell < hub.len and hub[cell]) {
        starts[index[cell] + 1] += 1;
        n_stage += 1;
    };
    for (hub, index, 0..) |h, ix, cell| if (h) {
        cells[ix] = @intCast(cell);
    };
    for (1..starts.len) |h| starts[h] += starts[h - 1];
    const entries = try gpa.alloc(u32, n_stage);
    errdefer gpa.free(entries);
    const fill = try gpa.dupe(u32, starts[0..n_hubs]);
    defer gpa.free(fill);
    var k: u32 = 0;
    for (table) |*cell| if (cell.* < hub.len and hub[cell.*]) {
        const h = index[cell.*];
        cell.* = base + k;
        entries[fill[h]] = base + k;
        fill[h] += 1;
        k += 1;
    };
    assert(k == n_stage);
    return .{ .{ .cells = cells, .starts = starts, .entries = entries }, n_stage };
}

/// Each instance's colour (1-based) into `level`, one past the last colour
/// on any non-hub cell it touches; returns the colour count.
fn levels(gpa: Allocator, t: Tables, slots: []const u32, rows: []const u32, level: []u32) Allocator.Error!u32 {
    assert(level.len == t.count);
    const last_slot = try gpa.alloc(u32, t.trash_slot);
    defer gpa.free(last_slot);
    const last_row = try gpa.alloc(u32, t.ground_row);
    defer gpa.free(last_row);
    @memset(last_slot, 0);
    @memset(last_row, 0);
    var n_colors: u32 = 0;
    for (level, 0..) |*lv, i| {
        const own_slots = slots[i * t.ks ..][0..t.ks];
        const own_rows = rows[i * t.n_u ..][0..t.n_u];
        var c: u32 = 0;
        for (own_slots) |s| if (s < t.trash_slot) {
            c = @max(c, last_slot[s]);
        };
        for (own_rows) |r| if (r < t.ground_row) {
            c = @max(c, last_row[r]);
        };
        lv.* = c + 1;
        for (own_slots) |s| if (s < t.trash_slot) {
            last_slot[s] = lv.*;
        };
        for (own_rows) |r| if (r < t.ground_row) {
            last_row[r] = lv.*;
        };
        n_colors = @max(n_colors, lv.*);
    }
    assert(n_colors <= t.count);
    return n_colors;
}

/// Counting sort of instance ids by colour, instance order kept within a
/// colour: the colour starts and the order. Caller owns both.
fn countingSort(gpa: Allocator, level: []const u32, n_colors: u32) Allocator.Error!struct { []u32, []u32 } {
    const starts = try gpa.alloc(u32, n_colors + 1);
    errdefer gpa.free(starts);
    @memset(starts, 0);
    for (level) |lv| starts[lv] += 1;
    for (1..starts.len) |c| starts[c] += starts[c - 1];
    const order = try gpa.alloc(u32, level.len);
    errdefer gpa.free(order);
    const fill = try gpa.dupe(u32, starts[0..n_colors]);
    defer gpa.free(fill);
    for (level, 0..) |lv, i| {
        order[fill[lv - 1]] = @intCast(i);
        fill[lv - 1] += 1;
    }
    return .{ starts, order };
}

test "lane colouring: per-cell sums are bitwise the instance-order walk" {
    // Oracle: the host's serial stamp, instance order then table order.
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
        const c = try color(gpa, .{ .count = count, .ks = ks, .slots = slots, .n_u = n_u, .rows = rows, .trash_slot = nnz, .ground_row = n, .stage_slot0 = nnz + 1, .stage_row0 = n + 1 });
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

test "color frees everything on every allocation failure" {
    // Slot cell 0 and row 0 are hubs (six touchers), cell 1 is not.
    const slots = [_]u32{ 0, 1, 0, 3, 0, 1, 0, 2, 0, 3, 0, 1 };
    const rows = [_]u32{ 0, 0, 1, 0, 0, 0 };
    const run = struct {
        fn f(gpa: Allocator, s: []const u32, r: []const u32) !void {
            const c = try color(gpa, .{ .count = 6, .ks = 2, .slots = s, .n_u = 1, .rows = r, .trash_slot = 3, .ground_row = 2, .stage_slot0 = 4, .stage_row0 = 3 });
            c.deinit(gpa);
        }
    }.f;
    try std.testing.checkAllAllocationFailures(std.testing.allocator, run, .{ &slots, &rows });
}
