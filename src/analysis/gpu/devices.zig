//! The devices GPU work spreads over (multi-GPU stages 1 to 3). In: the
//! driver's device list (`CUDA_VISIBLE_DEVICES` / `HIP_VISIBLE_DEVICES`
//! decide it). Out: the device table with each card's capacity in host
//! cores, contiguous row-order slices of a lane batch, a query's placement,
//! and the pivot tape every slice of one split batch replays. A thread's GPU
//! handles open on `ordinal`; a batch leader sets it on each slice's thread.
//! Cites: dev/research/gpu-lane-e1.md (the E1 capacity measurement).
//!
//! Contents:
//! - `pub const backend: ?gompute.Backend`
//! - `pub threadlocal var ordinal: c_int`
//! - `pub threadlocal var tape: ?*TapeShare`
//! - `pub threadlocal var tape_builds: bool`
//! - `pub const devices_max: usize`
//! - `pub const Tape = struct { idx, tab }`
//! - `pub const Device = struct { ordinal, cap }`
//! - `pub const Placement = struct` with `pub fn release(self: Placement) void`
//! - `pub const TapeShare = struct` with `pub fn take`, `pub fn publish`, `pub fn fail`, `pub fn deinit`
//! - `pub fn table() []const Device`
//! - `pub fn totalCap() ?f64`
//! - `pub fn record(ord: c_int, rows: usize, ns: u64) void`
//! - `pub fn weighted(buf: []Device) []const Device`
//! - `pub fn place() Placement`
//! - `pub fn split(devs: []const Device, rows: usize, ends: []usize, who: []c_int) usize`

const std = @import("std");
const assert = std.debug.assert;
const gompute = @import("gompute");
const artifacts = @import("gompute_kernels");
const lk = @import("solver").lane_kernels;

const DeviceInfo = gompute.runtime.dynamic.DeviceInfo;

/// The backend this binary carries kernel images for, CUDA first. Comptime,
/// because naming `gompute.RawByName(.cuda)` is a compile error in a build
/// without CUDA artifacts.
pub const backend: ?gompute.Backend = if (artifacts.has_cuda) .cuda else if (artifacts.has_hip) .hip else null;

/// The device this thread's GPU handles open on (`rawKernelByName`'s
/// ordinal). 0 unless a batch leader handed this thread a slice or
/// `place` chose a device.
pub threadlocal var ordinal: c_int = 0;

/// The tape share of the batch this thread runs a slice of; null outside a
/// split batch (`GpuLanes.init` then builds its own tape).
pub threadlocal var tape: ?*TapeShare = null;
/// This thread runs the slice holding the batch's lane 0, which builds the
/// tape (`TapeShare.publish`); the others take it.
pub threadlocal var tape_builds: bool = false;

/// Most devices the table holds; a machine with more uses the first ones in
/// PCI order.
pub const devices_max: usize = 16;

/// A lane pivot tape: the index tables and their offsets.
pub const Tape = struct { idx: []u32, tab: lk.LaneTab };

/// One usable device: its ordinal and its share of the GPU work, in host
/// cores (`lanes.share`'s unit).
pub const Device = struct { ordinal: c_int, cap: f64 };

/// A query's device (`place`): its handles open on it until `release`.
pub const Placement = struct {
    /// The table row whose live count this query holds; null with one
    /// device or none.
    slot: ?usize,
    /// This thread's `ordinal` before `place`.
    prev: c_int,

    /// Restores this thread's previous `ordinal` and drops the query from
    /// its device's live count. Call once.
    pub fn release(self: Placement) void {
        ordinal = self.prev;
        if (self.slot) |i| {
            assert(i < devices_max);
            const before = registry.live[i].fetchSub(1, .monotonic);
            assert(before > 0); // Pairs with `place`'s fetchAdd.
        }
    }
};

/// The pivot tape of a split batch: slice 0 (the batch's lane 0) builds it
/// from lane 0's matrix, as one device would, and every other slice replays
/// the same tape, so a lane's bits do not depend on which device ran it.
pub const TapeShare = struct {
    state: std.atomic.Value(State) = .init(.pending),
    tab: lk.LaneTab = undefined,
    idx: []u32 = &.{},
    /// Owns `idx` once published.
    gpa: std.mem.Allocator,

    const State = enum(u8) { pending, published, failed };

    /// Slice 0's tape, a copy the caller owns in `gpa`; null when slice 0
    /// built none. Waits for slice 0.
    /// ponytail: a yield loop; slice 0 publishes within its
    /// `GpuLanes.init` (milliseconds), or before running its lanes on the
    /// CPU (`fail`).
    pub fn take(self: *TapeShare, gpa: std.mem.Allocator) std.mem.Allocator.Error!?Tape {
        var state = self.state.load(.acquire);
        while (state == .pending) : (state = self.state.load(.acquire)) std.Thread.yield() catch {};
        if (state == .failed) return null;
        assert(self.idx.len > 0 or self.tab.n == 0);
        return .{ .idx = try gpa.dupe(u32, self.idx), .tab = self.tab };
    }

    /// Hands slice 0's tape to the waiting slices, copying `idx` into
    /// `gpa`. A no-op once published or failed; an allocation failure
    /// `fail`s instead.
    pub fn publish(self: *TapeShare, tab: lk.LaneTab, idx: []const u32) void {
        if (self.state.load(.acquire) != .pending) return;
        self.idx = self.gpa.dupe(u32, idx) catch return self.fail();
        self.tab = tab;
        self.state.store(.published, .release);
    }

    /// Slice 0 builds no tape: the other slices build their own.
    pub fn fail(self: *TapeShare) void {
        _ = self.state.cmpxchgStrong(.pending, .failed, .release, .monotonic);
        assert(self.state.load(.monotonic) != .pending);
    }

    /// Frees the published copy. Every slice must have finished `take`.
    pub fn deinit(self: *TapeShare) void {
        self.gpa.free(self.idx);
        self.idx = &.{};
    }
};

/// A device slice smaller than this many rows goes to the slice before:
/// its context and kernel loads would cost more than its lanes return.
const slice_rows_min = 64;

/// E1 measured the RTX 4060 Laptop (24 SMs, FP64 at 1/64) at the work of
/// about 13 host cores per instance evaluation; FP64 throughput and the SM
/// count scale that.
const e1_cores = 13.0;
const e1_fp64_ratio = 64.0;
const e1_sms = 24.0;

/// Process-wide facts of the machine's driver: the device table (read once),
/// each device's measured rate and its live query count. A global because
/// the driver itself is process-wide state.
const Registry = struct {
    devs: [devices_max]Device = undefined,
    len: usize = 0,
    state: std.atomic.Value(Phase) = .init(.unread),
    /// Lanes per second each device ran in the last split batch, as f64
    /// bits by table row; 0 until measured.
    rates: [devices_max]std.atomic.Value(u64) = @splat(.init(0)),
    /// Single-circuit GPU queries open on each device (`place`).
    live: [devices_max]std.atomic.Value(u32) = @splat(.init(0)),

    const Phase = enum(u8) { unread, reading, read };
};

var registry: Registry = .{};

/// Every device that opens and reports its FP64 ratio, in PCI order (a
/// stable order whatever the driver's enumeration), each with its capacity.
/// Contexts on one card (same uuid: `GOMPUTE_FAKE_DEVICES`) split the card's
/// capacity, so the card's total share of a batch does not change with
/// them. Empty without a driver or images. Reads the driver on the first
/// call only; later calls return the same slice.
pub fn table() []const Device {
    if (registry.state.load(.acquire) != .read) {
        if (registry.state.cmpxchgStrong(.unread, .reading, .acquire, .monotonic) == null) {
            registry.len = readDevices(&registry.devs);
            registry.state.store(.read, .release);
        } else while (registry.state.load(.acquire) != .read) std.Thread.yield() catch {};
    }
    assert(registry.len <= devices_max);
    return registry.devs[0..registry.len];
}

/// The cards' summed capacity in host cores (`lanes.share`); null without
/// a usable device.
pub fn totalCap() ?f64 {
    const t = table();
    if (t.len == 0) return null;
    var sum: f64 = 0;
    for (t) |d| sum += d.cap;
    assert(sum > 0);
    return sum;
}

/// Books `rows` lanes run in `ns` on device `ord` (`multi.runLanes`). The
/// rate weighs the next batch's split, never the running one's.
pub fn record(ord: c_int, rows: usize, ns: u64) void {
    if (rows == 0 or ns == 0) return;
    for (table(), 0..) |d, i| if (d.ordinal == ord) {
        const rate: f64 = @as(f64, @floatFromInt(rows)) * 1e9 / @as(f64, @floatFromInt(ns));
        assert(rate > 0);
        registry.rates[i].store(@bitCast(rate), .monotonic);
    };
}

/// The devices weighted for the next split, written into `buf`: measured
/// rates when every device has one, else capacities. Asserts that `buf`
/// holds `devices_max` rows.
pub fn weighted(buf: []Device) []const Device {
    assert(buf.len >= devices_max);
    const t = table();
    var all = true;
    for (registry.rates[0..t.len]) |*r| all = all and r.load(.monotonic) != 0;
    for (t, buf[0..t.len], registry.rates[0..t.len]) |d, *w, *r| {
        w.* = .{ .ordinal = d.ordinal, .cap = if (all) @bitCast(r.load(.monotonic)) else d.cap };
    }
    return buf[0..t.len];
}

/// Puts this thread's next GPU handles (`ordinal`) on the device with the
/// fewest live queries per unit of capacity, the first on a tie; with one
/// device or none, on device 0 as before. Device choice changes no bits:
/// the device eval and the device LU are the host's arithmetic. Pair with
/// `Placement.release`.
/// ponytail: a count at placement, not a load measure; queries of very
/// different sizes can pile on one card.
pub fn place() Placement {
    const prev = ordinal;
    const t = table();
    if (t.len <= 1) return .{ .slot = null, .prev = prev };
    var best: usize = 0;
    var best_load = std.math.inf(f64);
    for (t, registry.live[0..t.len], 0..) |d, *live, i| {
        const load = @as(f64, @floatFromInt(live.load(.monotonic) + 1)) / d.cap;
        if (load < best_load) {
            best = i;
            best_load = load;
        }
    }
    assert(best < t.len);
    _ = registry.live[best].fetchAdd(1, .monotonic);
    ordinal = t[best].ordinal;
    return .{ .slot = best, .prev = prev };
}

/// Cuts `rows` lanes into contiguous row-order slices by capacity, one per
/// device that gets at least `slice_rows_min` rows (the first device takes
/// what the others leave). Writes each slice's end row and device into
/// `ends` and `who` and returns the slice count, at least 1. A pure
/// function of `devs` and `rows`, so a batch splits the same way every run.
/// Asserts that `ends` and `who` have one row per device (at least one).
pub fn split(devs: []const Device, rows: usize, ends: []usize, who: []c_int) usize {
    assert(ends.len >= @max(devs.len, 1));
    assert(who.len == ends.len);
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
        if (take < slice_rows_min and k > 0) {
            ends[k - 1] += take;
        } else if (take > 0 or k == 0) {
            ends[k] = start + take;
            who[k] = d.ordinal;
            k += 1;
        }
        start += take;
    }
    // A short first slice folds into the next one instead.
    if (k > 1 and ends[0] < slice_rows_min) {
        std.mem.copyForwards(usize, ends[0 .. k - 1], ends[1..k]);
        std.mem.copyForwards(c_int, who[0 .. k - 1], who[1..k]);
        k -= 1;
    }
    assert(k >= 1);
    assert(ends[k - 1] == rows);
    return k;
}

// ---- private ----

/// The device table into `out`; returns its length.
fn readDevices(out: []Device) usize {
    const be = backend orelse return 0;
    const dynamic = gompute.runtime.dynamic;
    const rt: dynamic.Backend = if (be == .cuda) .cuda else .hip;
    const count = dynamic.deviceCount(rt) catch return 0;
    var infos: [devices_max]DeviceInfo = undefined;
    var n: usize = 0;
    for (0..@min(count, out.len)) |o| {
        var c = dynamic.Compute.initDevice(rt, @intCast(o)) catch continue;
        defer c.deinit();
        const info = c.info() catch continue;
        if (info.fp64_ratio == 0) continue;
        infos[n] = info;
        n += 1;
    }
    std.mem.sortUnstable(DeviceInfo, infos[0..n], {}, pciLessThan);
    for (infos[0..n], out[0..n]) |info, *d| {
        var same: f64 = 0;
        for (infos[0..n]) |o| same += @floatFromInt(@intFromBool(std.mem.eql(u8, &o.uuid, &info.uuid)));
        assert(same >= 1); // A card shares its uuid with itself.
        d.* = .{ .ordinal = info.ordinal, .cap = cardCap(info) / same };
    }
    return n;
}

fn pciLessThan(_: void, a: DeviceInfo, b: DeviceInfo) bool {
    const ka = [_]u64{ a.pci_domain, a.pci_bus, a.pci_device, @intCast(a.ordinal) };
    const kb = [_]u64{ b.pci_domain, b.pci_bus, b.pci_device, @intCast(b.ordinal) };
    return std.mem.order(u64, &ka, &kb) == .lt;
}

fn cardCap(info: DeviceInfo) f64 {
    const ratio: f64 = @floatFromInt(@max(info.fp64_ratio, 1));
    const sms: f64 = @floatFromInt(@max(info.sms, 1));
    return e1_cores * e1_fp64_ratio / ratio * (sms / e1_sms);
}

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

test "split covers every row exactly once over random capacities" {
    // Oracle: the slices partition [0, rows) in order, each but the first
    // device's at least `slice_rows_min` rows.
    var prng = std.Random.DefaultPrng.init(0xd3);
    const r = prng.random();
    var devs: [devices_max]Device = undefined;
    var ends: [devices_max]usize = undefined;
    var who: [devices_max]c_int = undefined;
    for (0..500) |_| {
        const n = r.intRangeAtMost(usize, 1, devices_max);
        for (devs[0..n], 0..) |*d, i| d.* = .{ .ordinal = @intCast(i), .cap = 0.1 + r.float(f64) * 50 };
        const rows = r.uintLessThan(usize, 5000);
        const k = split(devs[0..n], rows, &ends, &who);
        try std.testing.expect(k >= 1 and k <= n);
        try std.testing.expectEqual(rows, ends[k - 1]);
        var start: usize = 0;
        for (ends[0..k], 0..) |e, i| {
            try std.testing.expect(e >= start);
            if (k > 1) try std.testing.expect(e - start >= slice_rows_min or i == k - 1 or i == 0);
            start = e;
        }
    }
}

test "TapeShare hands one published copy to every taker and none after fail" {
    const gpa = std.testing.allocator;
    var share: TapeShare = .{ .gpa = gpa };
    defer share.deinit();
    const tab = std.mem.zeroes(lk.LaneTab);
    share.publish(tab, &.{ 3, 1, 4 });
    // A second publish or a late fail changes nothing.
    share.publish(tab, &.{9});
    share.fail();
    const got = (try share.take(gpa)).?;
    defer gpa.free(got.idx);
    try std.testing.expectEqualSlices(u32, &.{ 3, 1, 4 }, got.idx);

    var failed: TapeShare = .{ .gpa = gpa };
    defer failed.deinit();
    failed.fail();
    failed.publish(tab, &.{1});
    try std.testing.expect(try failed.take(gpa) == null);
}
