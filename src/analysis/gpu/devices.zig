//! The devices GPU work spreads over (multi-GPU stage 1): the device table,
//! each device's capacity in host cores, the contiguous row-order slices a
//! lane batch splits into, and the pivot tape every slice of one batch
//! shares. A thread's GPU handles open on `ordinal`; the batch leader sets
//! it on each slice's thread, so every lane kernel, buffer and stream of a
//! slice lives on its device. Every usable device is used: the device set is
//! the driver's (`CUDA_VISIBLE_DEVICES` / `HIP_VISIBLE_DEVICES`).

const std = @import("std");
const gompute = @import("gompute");
const lk = @import("solver").lane_kernels;
const artifacts = @import("gompute_kernels");
const backend: ?gompute.Backend = if (artifacts.has_cuda) .cuda else if (artifacts.has_hip) .hip else null;

/// The device this thread's GPU handles open on (`rawKernelByName`'s
/// ordinal). 0 unless a batch leader handed this thread a slice.
pub threadlocal var ordinal: c_int = 0;

/// The tape share of the batch this thread runs a slice of; null outside a
/// split batch (`GpuLanes.init` then builds its own tape).
pub threadlocal var tape: ?*TapeShare = null;
/// This thread runs the slice holding the batch's lane 0, which builds the
/// tape (`TapeShare.publish`); the others take it.
pub threadlocal var tape_builds: bool = false;

/// A lane pivot tape: the index tables and their offsets.
pub const Tape = struct { idx: []u32, tab: lk.LaneTab };

/// One usable device: its ordinal and its share of the GPU work, in host
/// cores (`lanes.share`'s unit).
pub const Device = struct { ordinal: c_int, cap: f64 };

/// E1 measured the RTX 4060 Laptop (24 SMs, FP64 at 1/64) at the work of
/// about 13 host cores per instance evaluation; FP64 throughput and the SM
/// count scale that.
fn cardCap(info: gompute.runtime.dynamic.DeviceInfo) f64 {
    const ratio: f64 = @floatFromInt(@max(info.fp64_ratio, 1));
    return 13.0 * 64.0 / ratio * (@as(f64, @floatFromInt(@max(info.sms, 1))) / 24.0);
}

var table_buf: [16]Device = undefined;
var table_len: usize = 0;
var table_state: std.atomic.Value(u8) = .init(0); // 0 unread, 1 reading, 2 read

/// Every device that opens and reports its FP64 ratio, in PCI order (a
/// stable order whatever the driver's enumeration), each with its
/// capacity. Contexts on one card (same uuid: `GOMPUTE_FAKE_DEVICES`) split
/// the card's capacity, so the card's total share of a batch does not
/// change with them. Empty without a driver or images. Read once.
pub fn table() []const Device {
    if (table_state.load(.acquire) != 2) {
        if (table_state.cmpxchgStrong(0, 1, .acquire, .monotonic) == null) {
            table_len = read(&table_buf);
            table_state.store(2, .release);
        } else while (table_state.load(.acquire) != 2) std.Thread.yield() catch {};
    }
    return table_buf[0..table_len];
}

fn read(out: []Device) usize {
    const be = backend orelse return 0;
    const Compute = gompute.runtime.dynamic.Compute;
    const rt: gompute.runtime.dynamic.Backend = if (be == .cuda) .cuda else .hip;
    const count = gompute.runtime.dynamic.deviceCount(rt) catch return 0;
    var infos: [16]gompute.runtime.dynamic.DeviceInfo = undefined;
    var n: usize = 0;
    for (0..@min(count, out.len)) |o| {
        var c = Compute.initDevice(rt, @intCast(o)) catch continue;
        defer c.deinit();
        const info = c.info() catch continue;
        if (info.fp64_ratio == 0) continue;
        infos[n] = info;
        n += 1;
    }
    const Key = struct {
        fn lt(_: void, a: gompute.runtime.dynamic.DeviceInfo, b: gompute.runtime.dynamic.DeviceInfo) bool {
            const ka = [_]u64{ a.pci_domain, a.pci_bus, a.pci_device, @intCast(a.ordinal) };
            const kb = [_]u64{ b.pci_domain, b.pci_bus, b.pci_device, @intCast(b.ordinal) };
            return std.mem.order(u64, &ka, &kb) == .lt;
        }
    };
    std.mem.sortUnstable(gompute.runtime.dynamic.DeviceInfo, infos[0..n], {}, Key.lt);
    for (infos[0..n], out[0..n]) |info, *d| {
        var same: f64 = 0;
        for (infos[0..n]) |o| same += @floatFromInt(@intFromBool(std.mem.eql(u8, &o.uuid, &info.uuid)));
        d.* = .{ .ordinal = info.ordinal, .cap = cardCap(info) / same };
    }
    return n;
}

/// The cards' summed capacity in host cores (`lanes.share`); null without
/// a usable device.
pub fn totalCap() ?f64 {
    const t = table();
    if (t.len == 0) return null;
    var sum: f64 = 0;
    for (t) |d| sum += d.cap;
    return sum;
}

/// Lanes per second each device ran in the last split batch (`record`),
/// by `table` index; 0 until measured. Stage 3: a measured rate replaces
/// the capacity formula for the NEXT batch's split, never mid-batch.
var rates: [16]std.atomic.Value(u64) = @splat(.init(0));

/// Books `rows` lanes run in `ns` on device `ord` (`multi.runLanes`).
pub fn record(ord: c_int, rows: usize, ns: u64) void {
    if (rows == 0 or ns == 0) return;
    for (table(), 0..) |d, i| if (d.ordinal == ord) {
        const r: f64 = @as(f64, @floatFromInt(rows)) * 1e9 / @as(f64, @floatFromInt(ns));
        rates[i].store(@bitCast(r), .monotonic);
    };
}

/// The devices weighted for the next split: measured rates when every
/// device has one, else capacities. Written into `buf`.
pub fn weighted(buf: []Device) []const Device {
    const t = table();
    var all = true;
    for (t, 0..) |_, i| all = all and rates[i].load(.monotonic) != 0;
    for (t, buf[0..t.len], 0..) |d, *w, i| w.* = .{ .ordinal = d.ordinal, .cap = if (all) @bitCast(rates[i].load(.monotonic)) else d.cap };
    return buf[0..t.len];
}

/// Single-circuit GPU queries live on each device, for `place`.
var live: [16]std.atomic.Value(u32) = @splat(.init(0));

/// A query's device (`place`): its handles open on it until `release`.
pub const Placement = struct {
    slot: ?usize,
    prev: c_int,

    pub fn release(self: Placement) void {
        ordinal = self.prev;
        if (self.slot) |i| _ = live[i].fetchSub(1, .monotonic);
    }
};

/// Puts this thread's next GPU handles (`ordinal`) on the device with the
/// fewest live queries per unit of capacity, the first on a tie; with one
/// device or none, on device 0 as before. Device choice changes no bits:
/// the device eval and the device LU are the host's arithmetic.
/// ponytail: a count at placement, not a load measure; queries of very
/// different sizes can pile on one card.
pub fn place() Placement {
    const prev = ordinal;
    const t = table();
    if (t.len <= 1) return .{ .slot = null, .prev = prev };
    var best: usize = 0;
    var best_load = std.math.inf(f64);
    for (t, 0..) |d, i| {
        const load = @as(f64, @floatFromInt(live[i].load(.monotonic) + 1)) / d.cap;
        if (load < best_load) {
            best = i;
            best_load = load;
        }
    }
    _ = live[best].fetchAdd(1, .monotonic);
    ordinal = t[best].ordinal;
    return .{ .slot = best, .prev = prev };
}

/// A device's slice smaller than this many rows goes to the slice before:
/// its context and kernel loads would cost more than its lanes return.
pub const min_slice = 64;

/// Cuts `rows` lanes into contiguous row-order slices by capacity, one per
/// device that gets at least `min_slice` rows (the first device takes what
/// the others leave). Writes each slice's end row and device into `ends`
/// and `who` and returns the slice count, at least 1. A pure function of
/// the table and `rows`, so a batch splits the same way every run.
pub fn split(devs: []const Device, rows: usize, ends: []usize, who: []c_int) usize {
    std.debug.assert(ends.len >= @max(devs.len, 1) and who.len == ends.len);
    if (devs.len <= 1) {
        ends[0] = rows;
        who[0] = if (devs.len == 1) devs[0].ordinal else 0;
        return 1;
    }
    var sum: f64 = 0;
    for (devs) |d| sum += d.cap;
    var k: usize = 0;
    var start: usize = 0;
    for (devs, 0..) |d, i| {
        const want: usize = @intFromFloat(@ceil(d.cap / sum * @as(f64, @floatFromInt(rows))));
        const take = if (i + 1 == devs.len) rows - start else @min(want, rows - start);
        if (take < min_slice and k > 0) {
            ends[k - 1] += take;
        } else if (take > 0 or k == 0) {
            ends[k] = start + take;
            who[k] = d.ordinal;
            k += 1;
        }
        start += take;
    }
    // A short first slice folds into the next one instead.
    if (k > 1 and ends[0] < min_slice) {
        std.mem.copyForwards(usize, ends[0 .. k - 1], ends[1..k]);
        std.mem.copyForwards(c_int, who[0 .. k - 1], who[1..k]);
        k -= 1;
    }
    return k;
}

/// The pivot tape of a split batch: slice 0 (the batch's lane 0) builds it
/// from lane 0's matrix, as one device would, and every other slice replays
/// the same tape, so a lane's bits do not depend on which device ran it.
pub const TapeShare = struct {
    /// 0 pending, 1 published, 2 slice 0 built none.
    state: std.atomic.Value(u8) = .init(0),
    tab: lk.LaneTab = undefined,
    idx: []u32 = &.{},
    gpa: std.mem.Allocator,

    /// Slice 0's tape, a copy in `gpa`; null when slice 0 built none.
    /// Waits for slice 0. ponytail: a yield loop; slice 0 publishes within
    /// its `GpuLanes.init` (milliseconds), or before running its lanes on
    /// the CPU (`fail`).
    pub fn take(self: *TapeShare, gpa: std.mem.Allocator) !?Tape {
        while (self.state.load(.acquire) == 0) std.Thread.yield() catch {};
        if (self.state.load(.acquire) == 2) return null;
        return .{ .idx = try gpa.dupe(u32, self.idx), .tab = self.tab };
    }

    pub fn publish(self: *TapeShare, tab: lk.LaneTab, idx: []const u32) void {
        if (self.state.load(.acquire) != 0) return;
        self.idx = self.gpa.dupe(u32, idx) catch return self.fail();
        self.tab = tab;
        self.state.store(1, .release);
    }

    /// Slice 0 builds no tape: the other slices build their own.
    pub fn fail(self: *TapeShare) void {
        if (self.state.load(.acquire) == 0) self.state.store(2, .release);
    }

    pub fn deinit(self: *TapeShare) void {
        self.gpa.free(self.idx);
    }
};

test "split cuts rows by capacity in device order and folds short slices" {
    var ends: [3]usize = undefined;
    var who: [3]c_int = undefined;
    const two = [_]Device{ .{ .ordinal = 0, .cap = 6.5 }, .{ .ordinal = 1, .cap = 6.5 } };
    try std.testing.expectEqual(@as(usize, 2), split(&two, 1000, &ends, &who));
    try std.testing.expectEqualSlices(usize, &.{ 500, 1000 }, ends[0..2]);
    try std.testing.expectEqualSlices(c_int, &.{ 0, 1 }, who[0..2]);
    // 1001 rows: the first slice rounds up.
    _ = split(&two, 1001, &ends, &who);
    try std.testing.expectEqualSlices(usize, &.{ 501, 1001 }, ends[0..2]);
    // Too few rows for two slices: one device keeps them all.
    try std.testing.expectEqual(@as(usize, 1), split(&two, 100, &ends, &who));
    try std.testing.expectEqual(@as(usize, 100), ends[0]);
    const three = [_]Device{ .{ .ordinal = 2, .cap = 100 }, .{ .ordinal = 0, .cap = 1 }, .{ .ordinal = 1, .cap = 100 } };
    try std.testing.expectEqual(@as(usize, 2), split(&three, 1000, &ends, &who));
    try std.testing.expectEqualSlices(usize, &.{ 503, 1000 }, ends[0..2]);
    try std.testing.expectEqualSlices(c_int, &.{ 2, 1 }, who[0..2]);
    try std.testing.expectEqual(@as(usize, 1), split(&.{}, 7, &ends, &who));
    try std.testing.expectEqual(@as(usize, 7), ends[0]);
}
