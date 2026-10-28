//! The `auto` backend's price of a GPU query. In: the circuit's batches, a
//! host probe of their eval time, the caller's eval estimate and the
//! measurements earlier runs left in the cache directory. Out: an admit or a
//! decline before the first driver call, the audit that holds an admitted
//! query to its price, and the JIT warmth of the heavy kernel images.
//! Cites #126 and #139 (the measurements behind the constants).
//!
//! Contents:
//! - `pub const Price = struct`
//! - `pub const Audit = struct (`from`, `at`)`
//! - `pub const CostCache = struct (`key`, `read`, `learn`, `write`, `Rec`)`
//! - `pub const heavy_image_bytes: usize`
//! - `pub fn admit(gpa: std.mem.Allocator, host: Host, threads: u32, evals: f64, driver_up: bool) !Price`
//! - `pub fn imageOf(kernel: []const u8) ?[:0]const u8`
//! - `pub fn imageWarm(image: []const u8, knobs: Knobs) bool`
//! - `pub fn markWarm(image: []const u8, knobs: Knobs) void`

const std = @import("std");
const assert = std.debug.assert;
const gompute = @import("gompute");
const artifacts = @import("gompute_kernels");
const device_ir = @import("device").abi;
const Knobs = @import("../Circuit.zig").Knobs;
const devices = @import("devices.zig");
const context = @import("context.zig");
const ctx_scatter = @import("context_scatter.zig");
const Planes = device_ir.Planes;
const backend = devices.backend;
const Host = context.Host;
const Error = context.Error;
const nowNs = context.nowNs;

/// The host's price per eval in microseconds, and the circuit's
/// `CostCache` key.
pub const Price = struct { cpu_us: f64, key: u64 };

/// The device path's measured price under `auto`: wall time
/// inside the entry points that evaluate (eval, the limit pass with its
/// prefetch, the charge passes) per device eval, from eval `from` to
/// eval `at`, against the host price per eval `Cost.admit` measured.
/// Checked once; past it the query stays where it is.
pub const Audit = struct {
    cpu_us: f64,
    spent_ns: u64 = 0,
    done: bool = false,
    /// Where the verdict is kept for the next run (`CostCache`).
    key: u64,
    knobs: Knobs,
    /// False under an explicit `cuda`/`hip`: the audit only measures, for a
    /// circuit `auto` declined on its probe (`CostCache.learn`).
    demote: bool = true,

    /// The first evals pay cold caches and the module's first launches.
    pub const from: u64 = 10;
    pub const at: u64 = 40;
};

/// The prices `Audit` measured for one circuit on this card,
/// kept beside the JIT markers in `<cache>/espice/gpu-cost/<key>` as the
/// host and device microseconds per eval. `Cost.admit` prices from them
/// instead of probing, and declines a circuit the device lost before
/// without starting the driver. The key covers the card, the circuit's
/// shape and models, the kernel images, the host threads and the query's
/// expected evals (an OP and a transient of one circuit price apart), so a
/// rebuild, another card or another thread count measures again.
pub const CostCache = struct {
    pub const Rec = struct { cpu_us: f64, gpu_us: f64 };

    pub fn key(host: Host, threads: u32, evals: f64) u64 {
        var h = std.hash.Wyhash.init(0x5e1ce);
        h.update(std.mem.asBytes(&cardId()));
        for ([_]u64{ host.n, host.planes.g_vals.len, @intFromBool(host.has_charge), threads, @bitCast(evals) }) |v| h.update(std.mem.asBytes(&v));
        for (host.batches) |b| {
            const get = b.hooks.gpu_payload orelse continue;
            const p = get(b.ctx);
            h.update(b.type_name);
            h.update(std.mem.asBytes(&p.count));
            // A runtime `.hdl` image would start the driver: its kernel's
            // name stands in.
            const image = imageOf(p.kernel) orelse p.kernel;
            const ih = std.hash.Wyhash.hash(0, image);
            h.update(std.mem.asBytes(&ih));
        }
        return h.final();
    }

    /// The uuid of the card this thread's queries open on
    /// (`devices.place`), hashed; 0 without one.
    fn cardId() u64 {
        const be = backend orelse return 0;
        const dyn = gompute.runtime.dynamic;
        var c = dyn.Compute.initDevice(if (be == .cuda) .cuda else .hip, devices.ordinal) catch return 0;
        defer c.deinit();
        const info = c.info() catch return 0;
        return std.hash.Wyhash.hash(0, &info.uuid);
    }

    pub fn read(k: u64, knobs: Knobs) ?Rec {
        var buf: [512]u8 = undefined;
        const p = cachePath(&buf, "gpu-cost", k, knobs) orelse return null;
        const fd = std.c.open(p, .{ .ACCMODE = .RDONLY }, @as(c_uint, 0));
        if (fd < 0) return null;
        defer _ = std.c.close(fd);
        var rec: [2]f64 = undefined;
        if (std.c.read(fd, @ptrCast(&rec), @sizeOf([2]f64)) != @sizeOf([2]f64)) return null;
        // gpu_us 0: `auto` declined on the probe; nothing measured the device.
        if (!(rec[0] > 0) or !(rec[1] >= 0)) return null;
        return .{ .cpu_us = rec[0], .gpu_us = rec[1] };
    }

    /// The measuring audit for an explicit `cuda`/`hip` query of a circuit
    /// `auto` priced before (a record exists): it books the device's cost
    /// per eval beside the host's, so the next `auto` run decides from both
    /// measurements instead of the estimate that declined it. Null for a
    /// circuit with no record.
    pub fn learn(host: Host, threads: u32, evals: f64, knobs: Knobs) ?Audit {
        if (evals < 20) return null;
        const k = key(host, threads, evals);
        const rec = read(k, knobs) orelse return null;
        return .{ .cpu_us = rec.cpu_us, .key = k, .knobs = knobs, .demote = false };
    }

    /// Best effort, as `markWarm`: a missing record only means the next
    /// run probes.
    pub fn write(k: u64, knobs: Knobs, rec: Rec) void {
        var buf: [512]u8 = undefined;
        const p = cachePath(&buf, "gpu-cost", k, knobs) orelse return;
        if (!makeParents(&buf, p)) return;
        const fd = std.c.open(p, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true }, @as(c_uint, 0o644));
        if (fd < 0) return;
        defer _ = std.c.close(fd);
        const v = [2]f64{ rec.cpu_us, rec.gpu_us };
        _ = std.c.write(fd, @ptrCast(&v), @sizeOf([2]f64));
    }
};

/// Kernel images larger than this JIT for seconds to minutes on first use
/// (bsim4va 3.6 MB of PTX in 15 s, psp103 7.8 MB in 35 s, hisimhv_va
/// 29.3 MB in 1001 s), so `auto` only runs them once a `--backend cuda` run
/// has compiled them into the driver cache.
pub const heavy_image_bytes: usize = 1 << 20;

// TESTHELPER
/// A batch for `Cost.admit` that busy-waits `spin_ns` per eval and counts
/// its evals.
const StubBatch = struct {
    spin_ns: u64,
    evals: u32 = 0,

    fn eval(ctx: *anyopaque, _: *const Planes, _: u32, _: u32, _: []const f64, _: f64) void {
        const self: *StubBatch = @ptrCast(@alignCast(ctx));
        self.evals += 1;
        const t0 = nowNs();
        while (nowNs() - t0 < self.spin_ns) {}
    }

    fn payload(_: *anyopaque) device_ir.GpuPayload {
        return .{
            .kernel = "",
            .count = 1,
            .n_u = 2,
            .models = &.{},
            .model_of = &.{},
            .instances = &.{},
            .gath = &.{},
            .rhs_idx = &.{},
            .slots = &.{ 0, 1 },
            .lim_kernel = "",
            .ctl_kernel = "",
            .reduce_kernel = "",
            .lim_x = &.{},
            .states = &.{},
            .lim_active = false,
        };
    }

    fn batch(self: *StubBatch, hooks: *const device_ir.Hooks) device_ir.Batch {
        return .{ .ctx = self, .eval = eval, .eval_newton = eval, .count = 1, .n_u = 2, .has_charge = false, .has_const_jacobian = false, .type_name = "stub", .hooks = hooks };
    }
};

/// The price of the device path, measured on an RTX 4060 Laptop (sm_89,
/// PCIe 4.0 x8) against an i9-14900HX; see `Cost`.
///
/// Per eval: launches and the one host wait, with the limit pass sharing it
/// (mos1_2000 phases: 10 us upload and clear, 42 us kernels, 40 us reduce,
/// 19 us download, and ~40 us of launch and wait around them).
const cost_fixed_us: f64 = 40;

/// Pinned-memory transfer rate, bytes per microsecond (~11 GB/s).
const cost_bus_b_per_us: f64 = 11_000;

/// Staging clear and the two reduce levels, bytes of staging per
/// microsecond (mos1_2000: 4.8 MB in 50 us).
const cost_stage_b_per_us: f64 = 100_000;

/// Device-kernel time as a fraction of one host thread's eval of the same
/// batches, 2000 inverters: mos1 42 us against ~450 us, bsim4va ~700 us
/// against ~7.3 ms.
const cost_kernel_ratio: f64 = 0.12;

/// One-time driver setup (cuInit, the context, module loads) in the first
/// query of a Problem (`Driver`); later queries reuse it.
const cost_init_us: f64 = 350_000;

/// Host lanes ParEval is worth per thread past the first. Measured on 8
/// threads against 1: bsim4_2000 3.3x, psp103_2000 1.8x, bsim3_2000 and the
/// mos1 decks 1x or slower. 0.1 (1.7x on 8) keeps the heavy models, where
/// the GPU's lead is large, and does not credit the light ones with lanes
/// they do not get.
const cost_lane_gain: f64 = 0.1;

/// The GPU must beat the CPU by this factor over the whole query: wall time
/// on a shared machine wanders by more than the model's error.
const cost_margin: f64 = 1.3;

/// ...and by this factor per eval. The probe's host time moves by up to 2x
/// between queries of one process under load (scaling_rc_ladder_100k: 1.5
/// and 2.9 ms), and that deck's GPU path costs more per step than its evals
/// show (8.3-8.9 ms against 8.2 on the CPU). The decks where the GPU wins
/// clear this easily: bsim3_2000 5.8x, bsim4va and psp103 ~5x.
const cost_min_ratio: f64 = 2.0;

pub fn admit(gpa: std.mem.Allocator, host: Host, threads: u32, evals: f64, driver_up: bool) !Price {
    const stats = host.knobs.gpu_stats != .off;
    const init_us: f64 = if (driver_up) 0 else cost_init_us;
    // The probe costs three host evals; a query with few more than
    // that cannot repay the setup.
    if (evals < 20) return decline(stats, 0, 0, "too few evals");
    const key = CostCache.key(host, threads, evals);
    if (CostCache.read(key, host.knobs)) |rec| if (rec.gpu_us > 0) {
        if (!measuredWins(rec, evals, init_us)) return decline(stats, rec.cpu_us, rec.gpu_us, "measured on this card");
        if (stats) std.debug.print(
            "gpu-stats: cost per eval cpu={d:.0} us gpu={d:.0} us (measured on this card), {d:.0} evals, setup {d:.0} ms: GPU\n",
            .{ rec.cpu_us, rec.gpu_us, evals, init_us * 1e-3 },
        );
        return .{ .cpu_us = rec.cpu_us, .key = key };
    };
    var slots: u64 = 0;
    for (host.batches) |b| {
        const get = b.hooks.gpu_payload orelse continue;
        const p = get(b.ctx);
        // A constant batch's g/c stamps sit in the baseline.
        if (!b.has_const_jacobian) slots += p.slots.len;
    }
    // Scratch planes: the probe must not disturb a linearization a
    // prerequisite query left behind.
    const ng = host.planes.g_vals.len;
    const nr = host.planes.rhs.len;
    const g = try gpa.alloc(f64, 2 * ng + 2 * nr);
    defer gpa.free(g);
    @memset(g, 0);
    const pl = ctx_scatter.scratchPlanes(g, ng, nr);
    // Spread, not zero: at x = 0 every MOSFET sits in cutoff, which
    // evaluates in half the time of the operating mix (mos1: 197 us
    // against ~400 us per eval on parallel_inverters_2000).
    const x = try gpa.alloc(f64, host.n + 1);
    defer gpa.free(x);
    for (x, 0..) |*xi, i| xi.* = @as(f64, @floatFromInt(i % 7)) * 0.3;
    var host_ns: u64 = std.math.maxInt(u64);
    for (0..3) |_| {
        const t0 = nowNs();
        for (host.batches) |b| {
            if (b.hooks.gpu_payload != null) b.eval(b.ctx, &pl, 0, b.count, x, 0);
        }
        host_ns = @min(host_ns, nowNs() - t0);
    }
    const host_us = @as(f64, @floatFromInt(host_ns)) * 1e-3;

    const n_planes: usize = if (host.has_charge) 4 else 2;
    const bus_bytes: f64 = @floatFromInt((n_planes * (ng + nr) / 2 + 2 * host.n) * @sizeOf(f64));
    const stage_bytes: f64 = @floatFromInt(slots * n_planes / 2 * @sizeOf(f64));
    const gpu_us = cost_fixed_us + bus_bytes / cost_bus_b_per_us +
        stage_bytes / cost_stage_b_per_us + host_us * cost_kernel_ratio;
    const lanes = 1.0 + cost_lane_gain * @as(f64, @floatFromInt(@max(1, threads) - 1));
    const cpu_us = host_us / lanes;
    if (!gpuWins(cpu_us, gpu_us, evals, init_us)) {
        // The host's price, for an explicit device run to set beside
        // its measured one (`CostCache.learn`).
        CostCache.write(key, host.knobs, .{ .cpu_us = cpu_us, .gpu_us = 0 });
        return decline(stats, cpu_us, gpu_us, "the CPU is priced faster");
    }
    if (stats) std.debug.print(
        "gpu-stats: cost per eval cpu={d:.0} us gpu={d:.0} us, {d:.0} evals, setup {d:.0} ms: GPU\n",
        .{ cpu_us, gpu_us, evals, init_us * 1e-3 },
    );
    return .{ .cpu_us = cpu_us, .key = key };
}

/// The embedded image that exports `kernel`, or null.
pub fn imageOf(kernel: []const u8) ?[:0]const u8 {
    if (comptime backend == null) return null;
    const idx = if (comptime backend == .cuda) artifacts.cuda_index else artifacts.hip_index;
    const images = if (comptime backend == .cuda) artifacts.cuda_images else artifacts.hip_images;
    const e = idx.get(kernel) orelse return null;
    return images[e.blob];
}

pub fn imageWarm(image: []const u8, knobs: Knobs) bool {
    return sassFits() or markedWarm(image, knobs);
}

/// Records `image` as compiled. Best effort: a missing marker only means the
/// next `auto` run keeps that batch on the CPU.
pub fn markWarm(image: []const u8, knobs: Knobs) void {
    var buf: [512]u8 = undefined;
    const path = markerPath(&buf, image, knobs) orelse return;
    if (!makeParents(&buf, path)) return;
    const fd = std.c.open(path, .{ .ACCMODE = .WRONLY, .CREAT = true }, @as(c_uint, 0o644));
    if (fd >= 0) _ = std.c.close(fd);
}

// ---- private ----

/// `gpuWins` for prices `Audit` measured: no per-eval safety ratio
/// (the audit demotes at 1), the query margin and the setup kept.
fn measuredWins(rec: CostCache.Rec, evals: f64, init_us: f64) bool {
    return rec.cpu_us > rec.gpu_us and cost_margin * (init_us + evals * rec.gpu_us) < evals * rec.cpu_us;
}

/// The GPU must beat the CPU by `cost_min_ratio` per eval and by
/// `cost_margin` over the whole query, setup included. Any NaN input
/// prices the CPU.
fn gpuWins(cpu_us: f64, gpu_us: f64, evals: f64, init_us: f64) bool {
    return cpu_us >= cost_min_ratio * gpu_us and cost_margin * (init_us + evals * gpu_us) < evals * cpu_us;
}

fn decline(stats: bool, cpu_us: f64, gpu_us: f64, why: []const u8) Error {
    if (stats) std.debug.print("gpu-stats: cost per eval cpu={d:.0} us gpu={d:.0} us: CPU ({s})\n", .{ cpu_us, gpu_us, why });
    return Error.NotEnoughGpuWork;
}

/// The images carry SASS for this card (`-Dcuda-sass`), so loading them
/// needs no JIT. Asked once per process; any failure answers false.
fn sassFits() bool {
    if (comptime backend != .cuda) return false;
    const want: u32 = comptime if (artifacts.cuda_sass) |arch| std.fmt.parseInt(u32, arch["sm_".len..], 10) catch 0 else 0;
    if (want == 0) return false;
    const Cache = struct {
        var state: std.atomic.Value(u8) = .init(0); // 0 unknown, 1 fits, 2 not
    };
    switch (Cache.state.load(.acquire)) {
        1 => return true,
        2 => return false,
        else => {},
    }
    const fits = blk: {
        var compute = gompute.runtime.dynamic.Compute.init(null) catch break :blk false;
        defer compute.deinit();
        break :blk (compute.computeCapability() catch break :blk false) == want;
    };
    Cache.state.store(if (fits) 1 else 2, .release);
    return fits;
}

/// The marker file that records `image` as compiled into the driver cache:
/// `$XDG_CACHE_HOME/espice/gpu-jit/<hash>` or `$HOME/.cache/...`. The hash
/// covers the image bytes, so a rebuild that changes the kernel starts cold.
/// ponytail: the driver cache also keys on the driver version, so an upgrade
/// leaves stale markers and `auto` pays one cold JIT; key on the version too
/// if that ever matters.
fn markerPath(buf: []u8, image: []const u8, knobs: Knobs) ?[:0]const u8 {
    return cachePath(buf, "gpu-jit", std.hash.Wyhash.hash(0, image), knobs);
}

/// `<cache>/espice/<dir>/<key as 16 hex digits>` in `buf`, `<cache>` being
/// `$XDG_CACHE_HOME`, else `$HOME/.cache`. Null without either, or past
/// `buf` (no path, never a truncated one).
fn cachePath(buf: []u8, comptime dir: []const u8, key: u64, knobs: Knobs) ?[:0]const u8 {
    if (knobs.xdg_cache_home) |x|
        return std.mem.printSentinel(buf, "{s}/espice/" ++ dir ++ "/{x:0>16}", .{ x, key }, 0) catch null;
    const home = knobs.home orelse return null;
    return std.mem.printSentinel(buf, "{s}/.cache/espice/" ++ dir ++ "/{x:0>16}", .{ home, key }, 0) catch null;
}

/// mkdir -p for `path`'s `espice/` and `<dir>/` (`cachePath`), whose bytes
/// live at the start of `buf`; EEXIST is the common case. False when
/// `path` has fewer than two slashes. Best effort: a failed mkdir surfaces
/// as the caller's failed open.
fn makeParents(buf: []u8, path: []const u8) bool {
    const leaf = std.mem.lastIndexOfScalar(u8, path, '/') orelse return false;
    const app = std.mem.lastIndexOfScalar(u8, path[0..leaf], '/') orelse return false;
    for ([_]usize{ app, leaf }) |end| {
        buf[end] = 0;
        _ = std.c.mkdir(@ptrCast(buf.ptr), 0o755);
        buf[end] = '/';
    }
    return true;
}

/// Whether `markWarm` recorded `image` under `knobs`' cache root.
fn markedWarm(image: []const u8, knobs: Knobs) bool {
    var buf: [512]u8 = undefined;
    const path = markerPath(&buf, image, knobs) orelse return false;
    return std.c.access(path, 0) == 0;
}

test "CostCache: a record reads back under its key and nowhere else" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var root: [64]u8 = undefined;
    const knobs: Knobs = .{ .xdg_cache_home = try std.fmt.bufPrint(&root, ".zig-cache/tmp/{s}", .{&tmp.sub_path}) };
    try std.testing.expect(CostCache.read(42, knobs) == null);
    CostCache.write(42, knobs, .{ .cpu_us = 120, .gpu_us = 30 });
    const rec = CostCache.read(42, knobs) orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(@as(f64, 120), rec.cpu_us);
    try std.testing.expectEqual(@as(f64, 30), rec.gpu_us);
    try std.testing.expect(CostCache.read(43, knobs) == null);
    // A declined probe books the host's price alone; it reads back for
    // `learn`, with no device price to admit on.
    CostCache.write(44, knobs, .{ .cpu_us = 120, .gpu_us = 0 });
    try std.testing.expectEqual(@as(f64, 0), (CostCache.read(44, knobs) orelse return error.TestUnexpectedResult).gpu_us);
    try std.testing.expect(measuredWins(rec, 1000, 0));
    try std.testing.expect(!measuredWins(.{ .cpu_us = 30, .gpu_us = 120 }, 1000, 0));
}

test "Cost.gpuWins needs both margins and refuses NaN" {
    // 10x per eval, cheap setup: the GPU.
    try std.testing.expect(gpuWins(1000, 100, 1000, 0));
    // Under the per-eval ratio, however many evals.
    try std.testing.expect(!gpuWins(199, 100, 1e9, 0));
    // Fast per eval, but setup outweighs the query.
    try std.testing.expect(!gpuWins(1000, 100, 10, cost_init_us));
    try std.testing.expect(!gpuWins(std.math.nan(f64), 100, 1000, 0));
    try std.testing.expect(!gpuWins(1000, std.math.nan(f64), 1000, 0));
}

test "Cost.gpuWins at its bars and on infinities" {
    // `cost_min_ratio` exactly passes.
    try std.testing.expect(gpuWins(cost_min_ratio * 100, 100, 1e6, 0));
    // Setup: 1.3 * (init + 1e5) < 1e6 holds below init = 669,230.8 us.
    try std.testing.expect(gpuWins(1000, 100, 1000, 669_000));
    try std.testing.expect(!gpuWins(1000, 100, 1000, 670_000));
    try std.testing.expect(!gpuWins(1000, std.math.inf(f64), 1000, 0));
    // Unbounded evals: inf < inf is false, so the CPU.
    try std.testing.expect(!gpuWins(1000, 100, std.math.inf(f64), 0));
}

test "Cost.admit prices a measured host eval and leaves the planes alone" {
    const gpa = std.testing.allocator;
    // `admit` reads only `gpu_payload` off the hooks.
    var gpu_hooks: device_ir.Hooks = undefined;
    gpu_hooks.gpu_payload = StubBatch.payload;
    var host_hooks: device_ir.Hooks = undefined;
    host_hooks.gpu_payload = null;
    var vals: [8]f64 = @splat(7);
    const planes: Planes = .{ .g_vals = vals[0..2], .c_vals = vals[2..4], .rhs = vals[4..6], .q_vec = vals[6..8] };
    const Case = struct { spin_ns: u64, evals: f64, driver_up: bool, admitted: bool, probed: bool };
    const cases = [_]Case{
        // Too few evals: declined before the probe.
        .{ .spin_ns = 2_000_000, .evals = 19, .driver_up = true, .admitted = false, .probed = false },
        // A free host eval never repays the 40 us launch floor.
        .{ .spin_ns = 0, .evals = 1e6, .driver_up = true, .admitted = false, .probed = true },
        // >= 2 ms against <= 0.28 ms + 0.12 of it on the device.
        .{ .spin_ns = 2_000_000, .evals = 1e6, .driver_up = true, .admitted = true, .probed = true },
        // The same, but 20 evals cannot repay a cold 350 ms driver setup.
        .{ .spin_ns = 2_000_000, .evals = 20, .driver_up = false, .admitted = false, .probed = true },
    };
    for (cases) |c| {
        var dev: StubBatch = .{ .spin_ns = c.spin_ns };
        var other: StubBatch = .{ .spin_ns = 0 };
        const batches = [_]device_ir.Batch{ other.batch(&host_hooks), dev.batch(&gpu_hooks) };
        const host: Host = .{ .batches = &batches, .planes = planes, .n = 2, .trash_slot = 2, .has_charge = false, .knobs = .{} };
        const got = admit(gpa, host, 1, c.evals, c.driver_up);
        if (c.admitted) try std.testing.expect((try got).cpu_us > 0) else try std.testing.expectError(Error.NotEnoughGpuWork, got);
        try std.testing.expectEqual(@as(u32, if (c.probed) 3 else 0), dev.evals);
        try std.testing.expectEqual(@as(u32, 0), other.evals);
        try std.testing.expectEqualSlices(f64, &@as([8]f64, @splat(7)), &vals);
    }
}

test markerPath {
    var buf: [64]u8 = undefined;
    const Want = struct { knobs: Knobs, prefix: ?[]const u8 };
    for ([_]Want{
        .{ .knobs = .{ .xdg_cache_home = "/x", .home = "/h" }, .prefix = "/x/espice/gpu-jit/" },
        .{ .knobs = .{ .home = "/h" }, .prefix = "/h/.cache/espice/gpu-jit/" },
        .{ .knobs = .{}, .prefix = null },
        // Past the buffer: no marker, not a truncated one.
        .{ .knobs = .{ .home = "/hhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh" }, .prefix = null },
    }) |w| {
        const got = markerPath(&buf, "image", w.knobs);
        const prefix = w.prefix orelse {
            try std.testing.expect(got == null);
            continue;
        };
        try std.testing.expect(std.mem.startsWith(u8, got.?, prefix));
        const hash = got.?[prefix.len..];
        try std.testing.expectEqual(@as(usize, 16), hash.len);
        for (hash) |ch| try std.testing.expect(std.ascii.isHex(ch));
    }
    // The name is keyed on the image bytes.
    var other: [64]u8 = undefined;
    try std.testing.expect(!std.mem.eql(u8, markerPath(&buf, "image", .{ .home = "/h" }).?, markerPath(&other, "imagf", .{ .home = "/h" }).?));
}

test "markWarm marks that image only" {
    // markedWarm, not imageWarm: a build whose SASS fits this card is warm
    // without a marker.
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var root: [64]u8 = undefined;
    const knobs: Knobs = .{ .xdg_cache_home = try std.fmt.bufPrint(&root, ".zig-cache/tmp/{s}", .{&tmp.sub_path}) };
    try std.testing.expect(!markedWarm("a", knobs));
    markWarm("a", knobs);
    try std.testing.expect(markedWarm("a", knobs));
    try std.testing.expect(!markedWarm("b", knobs));
    // A second mark, with both directories present, is harmless.
    markWarm("a", knobs);
    try std.testing.expect(markedWarm("a", knobs));
}
