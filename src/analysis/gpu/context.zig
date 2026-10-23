//! GPU device evaluation feeding the host solver. In: the analysis
//! circuit's batches and planes (`Host`). Out: the planes stamped on the
//! device, plus the limit, state and accepted-step passes of the resident
//! batches. Every eligible batch's models, instances and tapes are uploaded
//! once and stay resident; per Newton iteration the bus carries `x` up and
//! one contiguous block of value planes down, and the Newton update and
//! convergence test stay on the CPU (the device LU is gpu/lu.zig's, and
//! independent of this context). That round trip is the floor, so the GPU
//! pays off on device work, not circuit size; `init` prices both sides
//! before `auto` commits.
//!
//! Cites: dev/decisions.md S5 (the `noinline` entry points); #15, #93,
//! #126, #139 for the decisions marked in the body.

const std = @import("std");
const hdl = @import("hdl.zig");
const device_ir = @import("device").abi;
const gompute = @import("gompute");
const walk = @import("../par_eval.zig");

const Knobs = @import("../Circuit.zig").Knobs;
const devices = @import("devices.zig");
const gpu_lu = @import("lu.zig");
const direct = @import("solver").direct;
const Planes = device_ir.Planes;
const SimState = device_ir.SimState;

/// The backend this binary carries kernel images for. Comptime, because
/// naming `gompute.RawByName(.cuda)` is a compile error in a build without
/// CUDA artifacts.
const artifacts = @import("gompute_kernels");
const backend: ?gompute.Backend = if (artifacts.has_cuda)
    .cuda
else if (artifacts.has_hip)
    .hip
else
    null;

/// This binary carries kernel images, so a context can exist at all.
pub const available = backend != null;

/// What `--backend` can ask for. `auto` may decline for any reason and fall
/// back to the CPU; `cuda`/`hip` are explicit and either run there or fail
/// saying why (see `fallsBack`).
pub const Request = enum { cpu, auto, cuda, hip };

/// Why `init` refused, in the three kinds a caller acts on differently.
pub const Decline = enum {
    /// A performance estimate: the CPU is priced faster, or a heavy image
    /// is not compiled yet. An explicit request overrides it, so only
    /// `auto` sees one.
    policy,
    /// Nothing in this circuit has a kernel image. A reported CPU fallback
    /// under `auto`; `cuda`/`hip` fail.
    capability,
    /// The driver, hardware or build said no (including a build with no
    /// images). An explicit request fails naming the detected hardware.
    machine,
};

/// Why `GpuContext.init` failed: one of `Error`, `error.OutOfMemory`, or a
/// driver error gompute reports (bringing up the device, loading a module,
/// allocating or copying device memory). `declineKind` sorts them.
pub const InitError = Error || std.mem.Allocator.Error || gompute.Error;

/// Classifies an `init` error. Anything outside `Error`, a driver or
/// allocation failure included, is `.machine`.
pub fn declineKind(e: InitError) Decline {
    return switch (e) {
        Error.NotEnoughGpuWork, Error.ImageNotCompiled => .policy,
        Error.CircuitNotEligible => .capability,
        else => .machine, // else: every driver and allocator error is the machine saying no
    };
}

/// Whether a query runs on the CPU after `GpuContext.init` failed with
/// `e`. `cuda` and `hip` never fall back. `auto` does unless `explicit` (the
/// user named the GPU) and the machine said no.
pub fn fallsBack(req: Request, explicit: bool, e: InitError) bool {
    return switch (req) {
        .cpu => true,
        .auto => !explicit or declineKind(e) != .machine,
        .cuda, .hip => false,
    };
}

/// Returns "cuda", "hip" or "none": the first driver that brings up a device
/// on this machine, CUDA first, whatever this binary carries. Costs a driver
/// init and retains device 0's primary context, so call it when reporting a
/// failure.
pub fn detectedHardware() []const u8 {
    // wasm has no drivers to load (and gompute's loader lock needs sched_yield).
    if (comptime @import("builtin").cpu.arch.isWasm()) return "none";
    var compute = gompute.runtime.dynamic.Compute.init(null) catch return "none";
    defer compute.deinit();
    return switch (compute.backend) {
        .cpu => "none",
        .cuda, .hip => |b| @tagName(b),
    };
}

/// False for an explicit `cuda`/`hip` this binary cannot serve, so the
/// request fails before any simulation runs. `auto` and `cpu` always pass.
pub fn requestSupported(req: Request) bool {
    return switch (req) {
        .cpu, .auto => true,
        .cuda => backend == .cuda,
        .hip => backend == .hip,
    };
}

/// `void` in a build with no device images; every use sits behind
/// `comptime backend != null`.
const Raw = if (backend) |be| gompute.RawByName(be) else void;
const Buffer = if (backend != null) Raw.Buffer else void;
const Stream = if (backend != null) Raw.Stream else void;

/// The block width the kernels were exported with. HIP bakes it into
/// `globalIdX`; NVPTX reads the launch's real width, so CUDA may launch
/// narrower blocks (`evalBlock`).
const block_size: u32 = device_ir.gpu_block_size;

/// The refusals `GpuContext.init` makes on its own, before or instead of a
/// driver error; `declineKind` sorts them.
pub const Error = error{
    /// The build machine had no GPU, so no kernel images were emitted.
    NoGpuArtifacts,
    /// No device in this circuit has a kernel image (see `eval.gpuEligible`).
    CircuitNotEligible,
    /// The cost model priced the CPU faster.
    NotEnoughGpuWork,
    /// `auto` met a heavy image the driver has not compiled yet. The whole
    /// query runs on the CPU rather than leave that one model behind (#139).
    ImageNotCompiled,
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

/// Whether a context came up, so `auto`'s cost model stops pricing the
/// driver setup. The facade owns one per Problem and lends it to every query
/// (`Config.gpu_driver`). Queries on other threads read it while one sets
/// it: monotonic is enough, since a stale read only overprices one query
/// (#126).
pub const Driver = struct {
    up: std.atomic.Value(bool) = .init(false),
    /// `prestart`'s thread; `deinit` joins it.
    warm: ?std.Thread = null,

    /// Brings the driver and device 0's primary context up on a thread
    /// (05 5.12), so the ~350 ms start overlaps the frontend instead of the
    /// first query. The primary context stays retained for the process
    /// (gompute's `Compute.deinit` drops the handle only); a failure here
    /// leaves the first `GpuContext.init` to report it. Call once, before
    /// any query; pair with `deinit`.
    pub fn prestart(self: *Driver) void {
        if (comptime backend == null) return;
        std.debug.assert(self.warm == null);
        self.warm = std.Thread.spawn(.{}, warmUp, .{self}) catch null;
    }

    fn warmUp(self: *Driver) void {
        var compute = gompute.runtime.dynamic.Compute.init(if (backend.? == .cuda) .cuda else .hip) catch return;
        compute.deinit();
        self.up.store(true, .monotonic);
        // Every other device's context too: the device table opens each one
        // (multi-GPU stage 1). ponytail: one after another; a thread per
        // device when a box has many.
        _ = devices.table();
    }

    /// The driver is up, or `prestart` is bringing it up beside the
    /// frontend: either way a query no longer pays the start. `warm` is
    /// set before the first query and cleared after the last.
    pub fn started(self: *const Driver) bool {
        return self.warm != null or self.up.load(.monotonic);
    }

    /// Joins `prestart`'s thread, if any.
    pub fn deinit(self: *Driver) void {
        if (self.warm) |t| t.join();
        self.warm = null;
    }
};

/// Monotonic nanoseconds, for the cost probe and `Prof`.
fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

/// `ESPICE_GPU_STATS` wall-time split of the device path, printed at deinit.
const Prof = struct {
    on: bool = false,
    last: u64 = 0,
    evals: u64 = 0,
    /// Evals served by a prefetch, with no wait of their own.
    fused: u64 = 0,
    /// Host time to enqueue an eval, the wait for the device, and the host
    /// batches stamped after it.
    enqueue: u64 = 0,
    wait: u64 = 0,
    host: u64 = 0,
    lims: u64 = 0,
    lim: u64 = 0,
    /// `ESPICE_GPU_STATS=phases`: a sync after each eval stage, so the
    /// stage times below add up to the device time (upload+clear, eval
    /// kernels, reduce level 1, level 2, download). Serializes the
    /// pipeline; diagnosis only.
    phases: bool = false,
    phase: [5]u64 = @splat(0),
    /// Host waits on the stream, all causes, and by cause (`Wait`).
    syncs: u64 = 0,
    waits: [@typeInfo(Wait).@"enum".field_names.len]u64 = @splat(0),
    /// The census (01 1.1): evals that carried charge (`evalQ`), accepted
    /// steps (`stateCtl(.commit)`, the operating point included), and the
    /// bytes each way over the bus, every copy counted.
    charge_evals: u64 = 0,
    steps: u64 = 0,
    up: u64 = 0,
    down: u64 = 0,

    fn start(self: *Prof) void {
        if (self.on) self.last = nowNs();
    }

    /// Adds the time since the previous lap to `field`.
    fn lap(self: *Prof, comptime field: []const u8) void {
        if (!self.on) return;
        const t = nowNs();
        @field(self, field) += t - self.last;
        self.last = t;
    }

    fn print(self: Prof) void {
        if (!self.on) return;
        const us = struct {
            fn f(ns: u64, n: u64) f64 {
                return if (n == 0) 0 else @as(f64, @floatFromInt(ns)) * 1e-3 / @as(f64, @floatFromInt(n));
            }
        }.f;
        std.debug.print(
            "gpu-prof: syncs={d} evals={d} fused={d} per eval us: enqueue={d:.1} wait={d:.1} host={d:.1}; limits={d} {d:.1} us each\n",
            .{ self.syncs, self.evals, self.fused, us(self.enqueue, self.evals), us(self.wait, self.evals), us(self.host, self.evals), self.lims, us(self.lim, self.lims) },
        );
        const per = struct {
            fn f(v: u64, n: u64) f64 {
                return if (n == 0) 0 else @as(f64, @floatFromInt(v)) / @as(f64, @floatFromInt(n));
            }
        }.f;
        const newton = self.evals - self.charge_evals;
        std.debug.print(
            "gpu-census: newton evals={d} charge evals={d} steps={d}; syncs per newton eval={d:.2} per step={d:.2}; KB per newton eval up={d:.1} down={d:.1}; KB per step up={d:.1} down={d:.1}\n",
            .{ newton, self.charge_evals, self.steps, per(self.syncs, newton), per(self.syncs, self.steps), per(self.up, newton) / 1024, per(self.down, newton) / 1024, per(self.up, self.steps) / 1024, per(self.down, self.steps) / 1024 },
        );
        std.debug.print("gpu-census: waits", .{});
        inline for (@typeInfo(Wait).@"enum".field_names, self.waits) |name, w| std.debug.print(" {s}={d}", .{ name, w });
        std.debug.print("\n", .{});
        if (self.phases) std.debug.print(
            "gpu-prof: phases per eval us: upload+clear={d:.1} eval={d:.1} reduce1={d:.1} reduce2={d:.1} download={d:.1}\n",
            .{ us(self.phase[0], self.evals), us(self.phase[1], self.evals), us(self.phase[2], self.evals), us(self.phase[3], self.evals), us(self.phase[4], self.evals) },
        );
    }
};

/// Why the host waited on the stream, for the `ESPICE_GPU_STATS` census:
/// a Newton or charge eval nobody prefetched, a limit pass (with the eval
/// it prefetched), the state half, the accepted-state query (with the
/// charge eval it prefetched), a lazy tape pass, a host-stepped batch's
/// Instance merge, the eval check's replay, and everything else
/// (parameter repacks, `syncHostState`).
const Wait = enum { eval, charge, lim, state, query, tape, stepped, check, other };

/// The device path's measured price under `auto` (01 1.9b): wall time
/// inside the entry points that evaluate (eval, the limit pass with its
/// prefetch, the charge passes) per device eval, from eval `from` to
/// eval `at`, against the host price per eval `Cost.admit` measured.
/// Checked once; past it the query stays where it is.
const Audit = struct {
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
    const from: u64 = 10;
    const at: u64 = 40;
};

/// The prices `Audit` measured for one circuit on this card (01 1.9a),
/// kept beside the JIT markers in `<cache>/espice/gpu-cost/<key>` as the
/// host and device microseconds per eval. `Cost.admit` prices from them
/// instead of probing, and declines a circuit the device lost before
/// without starting the driver. The key covers the card, the circuit's
/// shape and models, the kernel images, the host threads and the query's
/// expected evals (an OP and a transient of one circuit price apart), so a
/// rebuild, another card or another thread count measures again.
const CostCache = struct {
    const Rec = struct { cpu_us: f64, gpu_us: f64 };

    fn key(host: Host, threads: u32, evals: f64) u64 {
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

    fn path(buf: []u8, k: u64, knobs: Knobs) ?[:0]const u8 {
        if (knobs.xdg_cache_home) |x|
            return std.mem.printSentinel(buf, "{s}/espice/gpu-cost/{x:0>16}", .{ x, k }, 0) catch null;
        const home = knobs.home orelse return null;
        return std.mem.printSentinel(buf, "{s}/.cache/espice/gpu-cost/{x:0>16}", .{ home, k }, 0) catch null;
    }

    fn read(k: u64, knobs: Knobs) ?Rec {
        var buf: [512]u8 = undefined;
        const p = path(&buf, k, knobs) orelse return null;
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
    fn learn(host: Host, threads: u32, evals: f64, knobs: Knobs) ?Audit {
        if (evals < 20) return null;
        const k = key(host, threads, evals);
        const rec = read(k, knobs) orelse return null;
        return .{ .cpu_us = rec.cpu_us, .key = k, .knobs = knobs, .demote = false };
    }

    /// Best effort, as `markWarm`: a missing record only means the next
    /// run probes.
    fn write(k: u64, knobs: Knobs, rec: Rec) void {
        var buf: [512]u8 = undefined;
        const p = path(&buf, k, knobs) orelse return;
        const leaf = std.mem.lastIndexOfScalar(u8, p, '/') orelse return;
        const app = std.mem.lastIndexOfScalar(u8, p[0..leaf], '/') orelse return;
        for ([_]usize{ app, leaf }) |end| {
            buf[end] = 0;
            _ = std.c.mkdir(@ptrCast(&buf), 0o755);
            buf[end] = '/';
        }
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
const heavy_image_bytes: usize = 1 << 20;

/// The embedded image that exports `kernel`, or null.
fn imageOf(kernel: []const u8) ?[:0]const u8 {
    if (comptime backend == null) return null;
    const idx = if (comptime backend == .cuda) artifacts.cuda_index else artifacts.hip_index;
    const images = if (comptime backend == .cuda) artifacts.cuda_images else artifacts.hip_images;
    const e = idx.get(kernel) orelse return null;
    return images[e.blob];
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
    const hash = std.hash.Wyhash.hash(0, image);
    if (knobs.xdg_cache_home) |x|
        return std.mem.printSentinel(buf, "{s}/espice/gpu-jit/{x:0>16}", .{ x, hash }, 0) catch null;
    const home = knobs.home orelse return null;
    return std.mem.printSentinel(buf, "{s}/.cache/espice/gpu-jit/{x:0>16}", .{ home, hash }, 0) catch null;
}

fn imageWarm(image: []const u8, knobs: Knobs) bool {
    if (sassFits()) return true;
    var buf: [512]u8 = undefined;
    const path = markerPath(&buf, image, knobs) orelse return false;
    return std.c.access(path, 0) == 0;
}

/// Records `image` as compiled. Best effort: a missing marker only means the
/// next `auto` run keeps that batch on the CPU.
fn markWarm(image: []const u8, knobs: Knobs) void {
    var buf: [512]u8 = undefined;
    const path = markerPath(&buf, image, knobs) orelse return;
    // mkdir -p for `espice/` and `gpu-jit/`; EEXIST is the common case.
    const jit_dir = std.mem.lastIndexOfScalar(u8, path, '/') orelse return;
    const app_dir = std.mem.lastIndexOfScalar(u8, path[0..jit_dir], '/') orelse return;
    for ([_]usize{ app_dir, jit_dir }) |end| {
        buf[end] = 0;
        _ = std.c.mkdir(@ptrCast(&buf), 0o755);
        buf[end] = '/';
    }
    const fd = std.c.open(path, .{ .ACCMODE = .WRONLY, .CREAT = true }, @as(c_uint, 0o644));
    if (fd >= 0) _ = std.c.close(fd);
}

/// One batch's resident working set and the kernels that consume it.
const BatchGpu = struct {
    kernel: Raw,
    /// `arp_lim_<model>`: the limit/state pass, when the device has one.
    /// The limit half per limit pass, the state half once per converged
    /// solve, at the x the converger stages (`updateStatesOnGpu`).
    lim_kernel: ?Raw,
    /// `arp_ctl_<model>`: the accepted-step latch pass, when the device
    /// declares `stateCtl`.
    ctl_kernel: ?Raw,
    /// `arp_qtp_<model>`: the LTE charge tape, when the host batch has one.
    qtp_kernel: ?Raw,
    /// `arp_qonly_<model>`: q rows and tape at x, values only, when the
    /// image has it and the batch has charge (`evalQ`, 04 4.4).
    q_kernel: ?Raw,
    /// `arp_evalc_<model>`, for a batch whose whole Jacobian is constant:
    /// the eval without g and c stamps, which sit in the device baseline
    /// instead (01 1.6). Null keeps the batch's g/c in the staging.
    const_kernel: ?Raw,
    /// Resident for the simulation. `repack` re-uploads only models and
    /// instances; the tapes never change.
    d_models: Buffer,
    d_model_of: Buffer,
    d_instances: Buffer,
    d_gath: Buffer,
    d_rhs_idx: Buffer,
    d_slots: Buffer,
    /// Lim plane (`GpuPayload.lim_x`'s length), `[]D.State` and the charge tape;
    /// 1-byte dummies when absent, so every kernel signature is uniform.
    d_lim: Buffer,
    d_states: Buffer,
    d_tape: Buffer,
    /// The host batch's `q_tape`, which `syncTape` fills.
    tape: []f64,
    /// The host batch, so `repack` can re-read its parameter arrays.
    ctx: *anyopaque,
    payload: *const fn (*anyopaque) device_ir.GpuPayload,
    set_limit_active: ?*const fn (*anyopaque, bool) void,
    count: u32,
    /// Bytes `d_models` holds; `repack` regrows it when unsharing changed it.
    models_len: usize,
    grid: gompute.Dim3,
    block: u32,
    has_lim: bool,
    /// The state half of `lim_kernel` can do anything: the batch has device
    /// state or a host state hook. A limit-only model (BSIM4) skips that
    /// launch and its wait.
    has_state: bool,
    /// The host batch carries charge (`Batch.has_charge`).
    has_charge: bool,
    /// The host batch has `Hooks.status`: eval latches `vera_status__` in
    /// the resident Instance, which `syncStatus` brings home.
    has_status: bool,
    /// Host-stepped (`eval.gpuHostStepped`): state hooks but no state
    /// kernel. The host walks step its Instance, and `pushStepped` uploads
    /// the rows they changed before the next launch that reads them.
    stepped: bool,
    /// The device lim plane holds live clamp state (after a seed upload or a
    /// limit launch). Mirrors the host batch's `lim_active`.
    lim_active: bool = false,
    /// Host `seedJunctions` wrote the host lim plane; upload it before the
    /// next launch that reads it.
    lim_dirty: bool = false,
    /// The Instance bytes host and device last agreed on, kept only when a
    /// kernel writes the resident Instance (the status latch, the limit/state
    /// pass or the accepted-step latch); empty otherwise. `syncInstances`
    /// diffs the host against it to tell parameter writes from latches.
    base: []u8,
    /// Scratch the device Instance bytes download into, `base`'s length, so
    /// the per-step merge allocates nothing (#174). Page-locked (`kernel`'s
    /// `freePinned`). Contents are dead between calls.
    dev: []u8,

    fn deinit(self: *BatchGpu, gpa: std.mem.Allocator) void {
        gpa.free(self.base);
        if (self.dev.len > 0) self.kernel.freePinned(self.dev);
        self.d_models.free();
        self.d_model_of.free();
        self.d_instances.free();
        self.d_gath.free();
        self.d_rhs_idx.free();
        self.d_slots.free();
        self.d_lim.free();
        self.d_states.free();
        self.d_tape.free();
        if (self.lim_kernel) |*k| k.deinit();
        if (self.ctl_kernel) |*k| k.deinit();
        if (self.qtp_kernel) |*k| k.deinit();
        if (self.q_kernel) |*k| k.deinit();
        if (self.const_kernel) |*k| k.deinit();
        self.kernel.deinit();
    }

    /// `launchOn` with this batch's grid.
    /// Asserts that a host-stepped batch's host rows have been pushed
    /// (`pushStepped`), checked in safe builds only (one memcmp).
    fn launch(self: *BatchGpu, k: *Raw, stream: *Stream, args: []const gompute.interface.Arg) !void {
        if (std.debug.runtime_safety and self.stepped)
            std.debug.assert(std.mem.eql(u8, self.payload(self.ctx).instances, self.base));
        try k.launchOn(stream, self.grid, .{ .x = self.block }, 0, args);
    }
};

/// The CUDA block width for a batch's per-instance kernels. Compact models
/// take ~255 registers a thread, so one 256-wide block fills an SM's register
/// file; 2000 instances were then 8 blocks on 24 SMs. 64 spreads any batch
/// over every SM: per eval against 256, bsim4va 679 us against 866, psp103
/// 1071 against 1284; 32 measured the same as 64. HIP keeps the exported
/// width, which its `globalIdX` bakes in. `ESPICE_GPU_BLOCK` overrides it
/// (`Knobs.gpu_block`).
fn evalBlock(knobs: Knobs) u32 {
    if (comptime backend != .cuda) return block_size;
    return knobs.gpu_block;
}

/// `arp_qtp_<model>` for the eval kernel `arp_eval_<model>`, or null when
/// the name has another shape or the result does not fit `buf` (the batch
/// then keeps the row LTE).
fn qtpName(buf: []u8, eval_kernel: []const u8) ?[]const u8 {
    return renamed(buf, eval_kernel, "arp_qtp_");
}

/// `eval_kernel` (`arp_eval_<model>`) with its prefix swapped for `prefix`,
/// or null as `qtpName`.
fn renamed(buf: []u8, eval_kernel: []const u8, prefix: []const u8) ?[]const u8 {
    const eval_prefix = "arp_eval_";
    if (!std.mem.startsWith(u8, eval_kernel, eval_prefix)) return null;
    return std.fmt.bufPrint(buf, "{s}{s}", .{ prefix, eval_kernel[eval_prefix.len..] }) catch null;
}

/// One Newton eval's matrix request (02 2.3): A = G + alpha·C. Under
/// `device` the device LU forms A from `d_planes` (`GpuLu.solveResident`)
/// and the host keeps g and c on the diagonal only; otherwise the device
/// forms A here and A comes down in place of g and c.
pub const GcReq = struct { alpha: f64, device: bool };

/// What `GpuContext.evalGc` left in the circuit's planes: `failed` as
/// `eval`'s false, `full` whole planes (the host combines), `host_a` A in
/// `gcVals`'s pinned buffer, `device` A on the device for `gcSolve`.
pub const GcServed = enum { failed, full, host_a, device };

/// The resident Newton matrix's buffers (02 2.3): the gather slots (the
/// diagonal, the ground pin's first), their g then c per plane buffer, and
/// A per plane buffer. Kernels from the LU image (`lu_device.zig`), launched
/// at `gc_block` threads.
const Gc = struct {
    combine: Raw,
    gather: Raw,
    d_k: Buffer,
    d_kv: Buffer,
    d_a: Buffer,
    /// Whole pinned allocations, sliced per plane buffer below.
    kv_all: []f64,
    a_all: []f64,
    pin_kv: [2][]f64,
    pin_a: [2][]f64,
    k: []const u32,
    nnz: u32,
    /// `ESPICE_GPU_LU_CHECK`: device-formed matrices compared with the
    /// host's, and how many differed.
    checks: u64 = 0,
    mismatches: u64 = 0,
};

/// `lu_device.zig`'s `gc_block`.
const gc_block: u32 = 256;

/// The planes as the device and the pinned landing area lay them out: g, rhs,
/// c, q back to back. c and q exist on the device only when a resident batch
/// has charge.
const Layout = struct {
    ng: usize,
    nr: usize,

    fn off(self: Layout, comptime plane: enum { g, rhs, c, q }) usize {
        return switch (plane) {
            .g => 0,
            .rhs => self.ng,
            .c => self.ng + self.nr,
            .q => 2 * self.ng + self.nr,
        };
    }

    /// Cells in the first `n_planes` planes.
    fn cells(self: Layout, n_planes: usize) usize {
        return if (n_planes == 2) self.off(.c) else 2 * self.off(.c);
    }
};

/// What `GpuContext.init` reads off the analysis circuit. The context keeps
/// `batches` (borrowed: the circuit's own, which outlive it) and the plane
/// allocations, and copies the rest.
pub const Host = struct {
    /// Every batch, resident or not; the fallback walks run over all of them.
    batches: []const device_ir.Batch,
    /// The circuit's own planes, with their current values.
    planes: Planes,
    n: u32,
    trash_slot: u32,
    has_charge: bool,
    knobs: Knobs,
};

/// The resident device state of one query. Heap-allocated; the circuit it
/// is attached to (`Circuit.attachGpu`) holds its address until detached.
/// Every entry point that launches takes the circuit's analysis state
/// (`sim`) as it stands at the call. The entry points are `noinline`: the
/// circuit's methods inline into the Newton and transient loops, and an
/// inlined GPU branch costs those loops registers on the host path
/// (dev/decisions.md S5). Out of line, each is the opaque call the old
/// function-pointer hook was, without the indirection.
pub const GpuContext = struct {
    gpa: std.mem.Allocator,
    /// The circuit's batches (`Host.batches`), for the fallback walks.
    all_batches: []const device_ir.Batch,
    n: u32,
    /// Some batch, resident or not, carries charge (`Host.has_charge`).
    has_charge: bool,
    /// The analysis state of the entry point running now, for the launches.
    sim: SimState = .{},
    /// The eligible batches, resident on the device.
    batches: []BatchGpu,
    /// Everything else: whatever `eval.gpuEligible` turns down, plus eligible
    /// batches demoted for a missing or cold image. They stamp on the host.
    cpu_batches: []const device_ir.Batch,
    /// The resident host-stepped batches (`BatchGpu.stepped`), which the
    /// state walks visit beside `cpu_batches`. A view of `cpu_owned`'s tail.
    stepped_batches: []const device_ir.Batch,
    /// Full-length allocations behind `batches` and `cpu_batches`. Demotions
    /// are only known after loading, so both are sized for the worst case and
    /// narrowed; `deinit` frees these, not the narrowed views.
    batches_owned: []BatchGpu,
    cpu_owned: []device_ir.Batch,

    /// The circuit's own plane allocations, swapped out for `pin_planes` while
    /// the context lives and handed back (with the last values) by
    /// `restorePlanes`.
    host_planes: Planes,
    /// Two page-locked landing areas in `Layout` order, back to back. The
    /// circuit's planes point into buffer `cur`, so the one download per eval
    /// is the whole plane update and the host batches stamp on top after it
    /// lands: no host clear, no host add. A prefetched eval (`pre`) lands
    /// in the other buffer, because the converger still reads this one's
    /// residual after the limit pass that launched it.
    pin_planes: []f64,
    cur: u1 = 0,
    layout: Layout,
    /// Page-locked staging for the eval's `x` (one per plane buffer: buffer
    /// `cur`'s holds the x of the last eval served, the other a prefetch's
    /// key), the limit pass's iterate and `x_old`, the flag word and the
    /// concatenated charge tapes.
    pin_x: [2][]f64,
    pin_x2: []f64,
    pin_xold: []f64,
    pin_flags: []u8,
    pin_tape: []f64,

    /// The eval's x, paired with `pin_x`: `d_x[cur]` holds the x of the last
    /// eval served, which the lazy tape pass reuses. The limit pass takes its
    /// iterate in `d_x2` and `x_old` in `d_xold`.
    d_x: [2]Buffer,
    d_xold: Buffer,
    d_x2: Buffer,
    /// The flag words the state kernels OR into: word 0 bit 0 limited, bit 1
    /// reject requested; word 1 the accepted-state query `acceptStates`
    /// fuses.
    d_flags: Buffer,
    d_planes: Buffer,

    /// Deterministic scatter. Kernels write each contribution to its own
    /// staging cell, and `reduce` sums each plane cell's run of cells in the
    /// order the serial CPU stamp visits them.
    ///
    /// Atomic accumulation reorders the sum from pass to pass. On
    /// parallel_inverters_2000 the Vdd row takes 8000 contributions that
    /// cancel to 3.6e-9; two replays at the same x differed by 2.1e-10, the
    /// converger rejected every iterate and dt underflowed. In tape order each
    /// instance's +1.8 meets its own -1.8 two entries later and the sum is
    /// exact.
    ///
    /// Built beside the frozen tapes, not through them: `d_slots` and
    /// `d_rhs_idx` keep their `[id][ru][cu]` u32 layout and length; only their
    /// values change from plane index to staging index.
    reduce: Raw,
    /// Staging for all planes back to back (`Order`), and views of it at each
    /// plane's base, which the eval kernels take as their plane pointers.
    d_stage: Buffer,
    v_stage: [4]Buffer,
    d_seg1: Buffer,
    d_seg2: Buffer,
    d_mid: Buffer,
    n_stage: usize,
    n_pieces: usize,
    n_cells: usize,
    /// Slot cells the constant batches stamp (`BatchGpu.const_kernel`), in
    /// order; their g then c baseline values sit at stage `n_stage` on,
    /// each a level-1 piece of its own ahead of the cell's run.
    base_cells: []u32,
    /// The q plane's first level-1 piece; its pieces run to `n_pieces`.
    q_piece0: usize,
    /// Every resident batch with charge has `q_kernel` and the device has a
    /// q plane, so `evalQ` runs charges only.
    q_only: bool,

    /// `pin_tape` holds the tape for plane buffer `tape_buf`'s eval; null
    /// when it holds nothing current. Host tapes then match with a copy.
    tape_buf: ?u1 = null,
    /// The host tapes match `d_x[cur]`; cleared by every eval.
    tape_fresh: bool = false,
    /// Set by `evalFollows`, consumed by the next limit pass or state query,
    /// and dropped by every other hook.
    follow: ?Follow = null,
    /// A prefetched eval at `pin_x[cur ^ 1]` sits in plane buffer `cur ^ 1`.
    /// Any hook that changes device state drops it.
    pre: ?Pre = null,
    /// Every resident batch with a host `q_tape` has a tape kernel, so the
    /// transient can keep its per-device LTE (`Circuit.qTapeLen`).
    tape_ok: bool,
    last_t: f64 = 0,

    /// A resident device requested a step reject, which the GPU path cannot
    /// deliver, or a tape sync faulted. Every later eval, limit and state
    /// call takes the host path.
    poisoned: bool = false,
    /// Host parameters changed; re-upload models and instances before the
    /// next launch. Lazy, so a pair of writes costs one upload.
    params_dirty: bool = false,

    /// Every copy and launch goes on this stream, never the NULL stream,
    /// which would synchronize with the copies this overlaps. One stream is
    /// enough: an iteration is a strict chain.
    stream: Stream,

    /// One-shot fallback warnings, one per failure kind so neither silences
    /// the other.
    warned_eval: bool = false,
    warned_state: bool = false,

    /// `evalCheck` scratch (a copy of the device planes); non-empty only
    /// under `ESPICE_GPU_EVAL_CHECK`.
    chk: []f64 = &.{},
    /// Worst reproducibility gap printed so far.
    chk_worst: f64 = 0,

    /// Some resident batch produces charge. When none does, the device c/q
    /// planes would only carry zeros, so they are neither reduced nor
    /// downloaded; the host batches stamp them alone.
    resident_charge: bool,

    /// `auto` admitted this query on a price: `Audit` holds the device to it.
    audit: ?Audit = null,
    /// `Audit` found the device losing and moved the rest of the query to
    /// the host, state included: every entry point walks on the host,
    /// quietly, and `syncHostState` has nothing left to bring home.
    demoted: bool = false,

    /// The device half of the next `stateCtl(.query)`, which `acceptStates`
    /// ran on its wait; dropped by every other entry point.
    query_hit: ?bool = null,

    prof: Prof = .{},
    /// False under `ESPICE_GPU_NOFUSE`: the unfused sequence, for A/B runs.
    /// Read once at `init`, since `evalFollows` fires every iteration.
    fuse: bool = true,

    /// The resident Newton matrix (`evalGc`), allocated on the first
    /// request; null before, and for good once `gc_off`.
    gc: ?Gc = null,
    gc_off: bool = false,
    /// `ESPICE_GPU_LU_CHECK`, read once at `init`.
    lu_check: bool = false,
    /// Length of the last eval's x (`gcRescue` restamps at it).
    x_len: usize = 0,
    /// What each plane buffer's last eval left on the host: null for whole
    /// planes, else the request it served (rhs and q whole, g and c only at
    /// the gathered slots).
    gc_buf: [2]?GcReq = .{ null, null },
    /// The request of the last Newton eval, which a prefetch inherits.
    gc_last: ?GcReq = null,

    const Self = @This();

    /// `sim` is the analysis state the announced eval runs under: a
    /// model that reads `$simparam("iteration")` stamps differently at the
    /// next iterate (the Meyer caps at iteration 1).
    const Follow = struct { x: []const f64, sim: SimState, charge: bool };
    const Pre = struct { sim: SimState, charge: bool, gc: ?GcReq };

    /// Uploads every eligible batch and keeps it resident. The caller
    /// classifies a failure with `declineKind`.
    ///
    /// `explicit` (the user named the GPU) skips the cost model and loads
    /// cold images; otherwise `init` prices the query both ways (`Cost`,
    /// from `evals`, the caller's estimate of its device evals, and
    /// `threads` host lanes) and declines before the first driver call when
    /// the CPU wins. The `auto` probe evaluates the eligible batches three
    /// times on the host, which rewrites their `q_tape`.
    ///
    /// On success the page-locked planes (`planes`) hold `host.planes`'
    /// values, ready for `Circuit.attachGpu`; `restorePlanes` copies them
    /// back. A success marks `driver` up, which later queries' cost models
    /// read; a null `driver` prices the setup every time. Caller owns the
    /// result and frees it with `deinit`.
    pub fn init(gpa: std.mem.Allocator, host: Host, driver: ?*Driver, explicit: bool, threads: u32, evals: f64) InitError!*Self {
        if (comptime backend == null) return Error.NoGpuArtifacts;
        const knobs = host.knobs;
        const report = explicit or knobs.gpu_stats != .off;

        var n_gpu: usize = 0;
        for (host.batches) |b| {
            if (b.hooks.gpu_payload != null) n_gpu += 1;
        }
        if (n_gpu == 0) return Error.CircuitNotEligible;
        const auto_price: ?Cost.Price = if (explicit) null else try Cost.admit(gpa, host, threads, evals, if (driver) |d| d.started() else false);

        const self = try gpa.create(Self);
        errdefer gpa.destroy(self);

        const batches = try gpa.alloc(BatchGpu, n_gpu);
        errdefer gpa.free(batches);
        // Sized for every batch: an eligible model with no image, or a cold
        // one under `auto`, demotes into this array below.
        const cpu_batches = try gpa.alloc(device_ir.Batch, host.batches.len);
        errdefer gpa.free(cpu_batches);

        var n_up: usize = 0;
        var n_cpu: usize = 0;
        var resident_charge = false;
        var tape_ok = true;
        var tape_len: usize = 0;
        errdefer for (batches[0..n_up]) |*bg| bg.deinit(gpa);

        var n_stepped: usize = 0;
        var q_only = true;
        for (host.batches) |b| {
            // Host-only batches first: once every eligible batch is up,
            // `batches[n_up]` is one past the end (#93).
            if (b.hooks.gpu_payload == null) {
                cpu_batches[n_cpu] = b;
                n_cpu += 1;
                continue;
            }
            const up = try loadBatch(gpa, &batches[n_up], b, explicit, report, knobs);
            if (!up) {
                cpu_batches[n_cpu] = b;
                n_cpu += 1;
                continue;
            }
            const bg = &batches[n_up];
            if (bg.stepped) {
                n_stepped += 1;
                cpu_batches[cpu_batches.len - n_stepped] = b;
            }
            resident_charge = resident_charge or b.has_charge;
            if (b.hooks.q_tape != null and bg.qtp_kernel == null) tape_ok = false;
            if (bg.has_charge and bg.q_kernel == null) q_only = false;
            tape_len += bg.tape.len;
            n_up += 1;
        }

        // Every eligible batch demoted; `batches[0]` below needs one.
        if (n_up == 0) return Error.CircuitNotEligible;
        if (knobs.gpu_stats != .off) std.debug.print(
            "gpu-stats: resident batches={d} (host-stepped {d}) host batches={d} charge planes={s} lte tape={s}\n",
            .{ n_up, n_stepped, n_cpu, if (resident_charge) "device" else "host-only", if (tape_ok) "device" else "row-plane" },
        );

        // Every handle allocates from the same primary context, so buffers
        // made through batch 0 are valid in every launch.
        const k0 = &batches[0].kernel;
        const layout: Layout = .{ .ng = host.planes.g_vals.len, .nr = host.planes.rhs.len };
        const n_planes: usize = if (resident_charge) 4 else 2;
        const x_bytes = (host.n + 1) * @sizeOf(f64);

        const base_cells = try baseCells(gpa, batches[0..n_up], host.trash_slot, layout.ng);
        errdefer gpa.free(base_cells);
        const order = blk: {
            const tapes = try gpa.alloc([]const u32, 2 * n_up);
            defer gpa.free(tapes);
            for (batches[0..n_up], tapes[0..n_up], tapes[n_up..]) |*bg, *st, *rt| {
                const p = bg.payload(bg.ctx);
                // A constant batch's g/c live in the baseline, not the stage.
                st.* = if (bg.const_kernel != null) &.{} else p.slots;
                rt.* = p.rhs_idx;
            }
            break :blk try Order.build(gpa, tapes[0..n_up], tapes[n_up..], host.trash_slot, host.n, layout, n_planes, base_cells);
        };
        defer order.deinit(gpa);
        // Same image as the eval kernels, so a miss is a stale build: fatal.
        var reduce = try hdl.kernel(backend.?, @ptrCast(batches[0].payload), batches[0].payload(batches[0].ctx).reduce_kernel);
        errdefer reduce.deinit();
        var stream = try batches[0].kernel.createStream();
        errdefer stream.deinit();
        try order.upload(batches[0..n_up], &stream);

        // One errdefer per resource, so a failure part way through releases
        // exactly what was made. Pinned memory goes through batch 0's handle,
        // which the batch errdefer above releases after these run.
        const pin_planes = try pinnedF64(k0, 2 * layout.cells(4));
        errdefer k0.freePinned(std.mem.sliceAsBytes(pin_planes));
        const pin_x0 = try pinnedF64(k0, host.n + 1);
        errdefer k0.freePinned(std.mem.sliceAsBytes(pin_x0));
        const pin_x1 = try pinnedF64(k0, host.n + 1);
        errdefer k0.freePinned(std.mem.sliceAsBytes(pin_x1));
        const pin_xold = try pinnedF64(k0, host.n + 1);
        errdefer k0.freePinned(std.mem.sliceAsBytes(pin_xold));
        const pin_x2 = try pinnedF64(k0, host.n + 1);
        errdefer k0.freePinned(std.mem.sliceAsBytes(pin_x2));
        // Word 0 the limit and state verdicts, word 1 a fused query's.
        const pin_flags = try k0.allocPinned(8);
        errdefer k0.freePinned(pin_flags);
        const pin_tape: []f64 = if (tape_len > 0) try pinnedF64(k0, tape_len) else &.{};
        errdefer if (pin_tape.len > 0) k0.freePinned(std.mem.sliceAsBytes(pin_tape));
        var d_x0 = try k0.alloc(x_bytes);
        errdefer d_x0.free();
        var d_x1 = try k0.alloc(x_bytes);
        errdefer d_x1.free();
        var d_xold = try k0.alloc(x_bytes);
        errdefer d_xold.free();
        var d_x2 = try k0.alloc(x_bytes);
        errdefer d_x2.free();
        var d_flags = try k0.alloc(8);
        errdefer d_flags.free();
        var d_planes = try k0.alloc(layout.cells(n_planes) * @sizeOf(f64));
        errdefer d_planes.free();
        var d_stage = try k0.alloc(@max(order.n_stage + order.n_base, 1) * @sizeOf(f64));
        errdefer d_stage.free();
        var d_seg1 = try uploadBytes(k0, std.mem.sliceAsBytes(order.seg1));
        errdefer d_seg1.free();
        var d_seg2 = try uploadBytes(k0, std.mem.sliceAsBytes(order.seg2));
        errdefer d_seg2.free();
        var d_mid = try k0.alloc(@max(order.n_pieces, 1) * @sizeOf(f64));
        errdefer d_mid.free();
        const chk: []f64 = if (knobs.gpu_eval_check)
            try gpa.alloc(f64, layout.cells(n_planes))
        else
            &.{};

        if (knobs.gpu_stats != .off) std.debug.print("gpu-stats: stage={d} pieces={d} cells={d}\n", .{ order.n_stage, order.n_pieces, layout.cells(n_planes) });
        // The eval kernels store every staging cell they own each pass; the
        // cells no pass writes keep this zero (01 1.10).
        try d_stage.fillAsync(0, @max(order.n_stage, 1) * @sizeOf(f64), &stream);
        // Charge-free planes read zero until a host batch stamps them.
        @memset(pin_planes, 0);
        var v_stage: [4]Buffer = undefined;
        for (&v_stage, order.stage_base) |*v, base| v.* = view(d_stage, base * @sizeOf(f64));

        self.* = .{
            .gpa = gpa,
            .all_batches = host.batches,
            .n = host.n,
            .has_charge = host.has_charge,
            .batches = batches[0..n_up],
            .cpu_batches = cpu_batches[0..n_cpu],
            .stepped_batches = cpu_batches[cpu_batches.len - n_stepped ..],
            .batches_owned = batches,
            .cpu_owned = cpu_batches,
            .host_planes = host.planes,
            .pin_planes = pin_planes,
            .layout = layout,
            .pin_x = .{ pin_x0, pin_x1 },
            .pin_xold = pin_xold,
            .pin_x2 = pin_x2,
            .pin_flags = pin_flags,
            .pin_tape = pin_tape,
            .d_x = .{ d_x0, d_x1 },
            .d_xold = d_xold,
            .d_x2 = d_x2,
            .d_flags = d_flags,
            .d_planes = d_planes,
            .reduce = reduce,
            .d_stage = d_stage,
            .v_stage = v_stage,
            .d_seg1 = d_seg1,
            .d_seg2 = d_seg2,
            .d_mid = d_mid,
            .n_stage = order.n_stage,
            .n_pieces = order.n_pieces,
            .n_cells = layout.cells(n_planes),
            .q_piece0 = order.q_piece0,
            .base_cells = base_cells,
            .q_only = q_only and resident_charge,
            .tape_ok = tape_ok,
            .stream = stream,
            .resident_charge = resident_charge,
            .chk = chk,
            .prof = .{ .on = knobs.gpu_stats != .off, .phases = knobs.gpu_stats == .phases },
            .fuse = !knobs.gpu_nofuse,
            .lu_check = knobs.gpu_lu_check,
            .audit = if (auto_price) |p| .{ .cpu_us = p.cpu_us, .key = p.key, .knobs = knobs } else CostCache.learn(host, threads, evals, knobs),
        };
        if (driver) |d| d.up.store(true, .monotonic);
        // A failure here unwinds through the errdefers above.
        try self.uploadBase();
        // The pinned planes start from the circuit's values.
        copyPlanes(self.pinnedPlanes(0), host.planes);
        return self;
    }

    /// Loads batch `b`'s kernels and uploads its working set into `bg`.
    /// False when it stays on the host: no GPU payload or no image in this
    /// build. Under `auto`, an image the driver has not compiled yet fails
    /// with `ImageNotCompiled`.
    fn loadBatch(gpa: std.mem.Allocator, bg: *BatchGpu, b: device_ir.Batch, explicit: bool, report: bool, knobs: Knobs) !bool {
        const get = b.hooks.gpu_payload orelse return false;
        const p = get(b.ctx);
        const image = imageOf(p.kernel) orelse hdl.runtimeImage(@ptrCast(get)) orelse {
            if (report) std.debug.print("note: GPU batch '{s}' ({d} instances) stays on the CPU: no kernel image in this build\n", .{ b.type_name, p.count });
            return false;
        };
        const heavy = image.len >= heavy_image_bytes;
        if (heavy and !explicit and !imageWarm(image, knobs)) {
            std.debug.print(
                "note: GPU kernel for '{s}' is not compiled yet ({d} MB of PTX), so this run stays on the CPU; `--backend cuda` compiles it once per build\n",
                .{ b.type_name, image.len >> 20 },
            );
            return Error.ImageNotCompiled;
        }
        if (heavy and report and !imageWarm(image, knobs)) std.debug.print(
            "note: compiling the GPU kernel for '{s}' ({d} MB of PTX); the driver caches it for later runs\n",
            .{ b.type_name, image.len >> 20 },
        );

        var kernel = try hdl.kernel(backend.?, @ptrCast(get), p.kernel);
        errdefer kernel.deinit();
        if (heavy) markWarm(image, knobs);
        // The other entry points share the eval kernel's image, so a missing
        // one means a stale image: fail rather than run half-resident.
        var lim_kernel: ?Raw = if (p.lim_kernel.len > 0) try hdl.kernel(backend.?, @ptrCast(get), p.lim_kernel) else null;
        errdefer if (lim_kernel) |*k| k.deinit();
        var ctl_kernel: ?Raw = if (p.ctl_kernel.len > 0) try hdl.kernel(backend.?, @ptrCast(get), p.ctl_kernel) else null;
        errdefer if (ctl_kernel) |*k| k.deinit();
        // `arp_qtp_<model>` is named after the eval kernel, not carried in
        // the frozen payload; an older image without it keeps the row LTE.
        var qtp_name: [128]u8 = undefined;
        var qtp_kernel: ?Raw = null;
        if (b.hooks.q_tape != null) if (qtpName(&qtp_name, p.kernel)) |name| {
            qtp_kernel = hdl.kernel(backend.?, @ptrCast(get), name) catch |e| switch (e) {
                error.KernelNotFound => null,
                else => return e, // else: any other driver error is fatal for the batch
            };
        };
        errdefer if (qtp_kernel) |*k| k.deinit();
        var q_kernel: ?Raw = null;
        if (b.has_charge) if (renamed(&qtp_name, p.kernel, "arp_qonly_")) |name| {
            q_kernel = gompute.rawKernelByName(backend.?, name, devices.ordinal) catch |e| switch (e) {
                error.KernelNotFound => null,
                else => return e, // else: any other driver error is fatal for the batch
            };
        };
        errdefer if (q_kernel) |*k| k.deinit();
        var const_kernel: ?Raw = null;
        if (b.has_const_jacobian and !knobs.gpu_nofuse) if (renamed(&qtp_name, p.kernel, "arp_evalc_")) |name| {
            const_kernel = gompute.rawKernelByName(backend.?, name, devices.ordinal) catch |e| switch (e) {
                error.KernelNotFound => null,
                else => return e, // else: any other driver error is fatal for the batch
            };
        };
        errdefer if (const_kernel) |*k| k.deinit();
        const tape: []f64 = if (qtp_kernel != null) @constCast(b.hooks.q_tape.?(b.ctx)) else &.{};

        // One errdefer per buffer: `bg` counts only once it is whole, so a
        // failure part way through frees what this batch already holds.
        var d_models = try uploadBytes(&kernel, p.models);
        errdefer d_models.free();
        var d_model_of = try uploadBytes(&kernel, std.mem.sliceAsBytes(p.model_of));
        errdefer d_model_of.free();
        var d_instances = try uploadBytes(&kernel, p.instances);
        errdefer d_instances.free();
        var d_gath = try uploadBytes(&kernel, std.mem.sliceAsBytes(p.gath));
        errdefer d_gath.free();
        var d_rhs_idx = try uploadBytes(&kernel, std.mem.sliceAsBytes(p.rhs_idx));
        errdefer d_rhs_idx.free();
        var d_slots = try uploadBytes(&kernel, std.mem.sliceAsBytes(p.slots));
        errdefer d_slots.free();
        // Left unwritten: reads are gated by `lim_active`, like the host's
        // `lim_x`.
        var d_lim = try kernel.alloc(@max(p.lim_x.len * @sizeOf(f64), 1));
        errdefer d_lim.free();
        var d_states = try uploadBytes(&kernel, p.states);
        errdefer d_states.free();
        var d_tape = try kernel.alloc(@max(tape.len * @sizeOf(f64), 1));
        errdefer d_tape.free();
        // `eval.gpuHostStepped`: state hooks but no device state kernel.
        const stepped = lim_kernel == null and
            (b.hooks.update_state != null or b.hooks.commit_state != null or b.hooks.state_ctl != null);
        // A stepped batch keeps `base` too: `pushStepped` diffs against it.
        const writes_instances = b.hooks.status != null or lim_kernel != null or ctl_kernel != null or stepped;
        const base: []u8 = if (writes_instances) try gpa.dupe(u8, p.instances) else &.{};
        errdefer gpa.free(base);
        // Pinned, so the per-step Instance downloads queue on the stream.
        const dev: []u8 = if (base.len > 0) try kernel.allocPinned(base.len) else &.{};

        const block = evalBlock(knobs);
        bg.* = .{
            .kernel = kernel,
            .lim_kernel = lim_kernel,
            .ctl_kernel = ctl_kernel,
            .qtp_kernel = qtp_kernel,
            .q_kernel = q_kernel,
            .const_kernel = const_kernel,
            .d_models = d_models,
            .d_model_of = d_model_of,
            .d_instances = d_instances,
            .d_gath = d_gath,
            .d_rhs_idx = d_rhs_idx,
            .d_slots = d_slots,
            .d_lim = d_lim,
            .d_states = d_states,
            .d_tape = d_tape,
            .tape = tape,
            .ctx = b.ctx,
            .payload = get,
            .set_limit_active = b.hooks.set_limit_active,
            .count = p.count,
            .models_len = p.models.len,
            .grid = gompute.Dim3.linear(p.count, block),
            .block = block,
            .has_lim = p.lim_x.len > 0,
            .has_state = p.states.len > 0 or b.hooks.update_state != null or b.hooks.commit_state != null,
            .has_charge = b.has_charge,
            .has_status = b.hooks.status != null,
            .stepped = stepped,
            .base = base,
            .dev = dev,
        };
        return true;
    }

    /// Prices a query both ways before `auto` touches the driver.
    ///
    /// The host side is measured: the best of three serial evals of every
    /// eligible batch into scratch planes, divided over the ParEval lanes.
    /// The probe also writes those batches' host `q_tape`, which every reader
    /// refills first (the transient evals at its start). The device side is
    /// modelled per eval (`cost_*`: launch and wait, bus, staging, kernels
    /// at `cost_kernel_ratio` of the host time) plus the driver setup once
    /// per Problem (`Driver`). Both are multiplied by the caller's eval estimate. A
    /// query with too few evals is declined before the probe.
    const Cost = struct {
        /// The host's price per eval in microseconds, and the circuit's
        /// `CostCache` key.
        const Price = struct { cpu_us: f64, key: u64 };

        fn admit(gpa: std.mem.Allocator, host: Host, threads: u32, evals: f64, driver_up: bool) !Price {
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
            const pl: Planes = .{
                .g_vals = g[0..ng],
                .c_vals = g[ng..][0..ng],
                .rhs = g[2 * ng ..][0..nr],
                .q_vec = g[2 * ng + nr ..][0..nr],
            };
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
    const Order = struct {
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
        const chunk: u32 = 64;

        fn deinit(self: Order, gpa: std.mem.Allocator) void {
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
        fn upload(self: Order, batches: []BatchGpu, stream: *Stream) !void {
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
        fn build(gpa: std.mem.Allocator, slot_tapes: []const []const u32, row_tapes: []const []const u32, trash_slot: u32, trash_row: u32, layout: Layout, n_planes: usize, base_cells: []const u32) !Order {
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

    /// The slot cells the constant batches stamp, ascending, trash excluded.
    /// Caller frees.
    fn baseCells(gpa: std.mem.Allocator, batches: []BatchGpu, trash_slot: u32, ng: usize) ![]u32 {
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

    /// Stamps the constant batches on the host into scratch planes (their
    /// Jacobian is the same at every x) and uploads g and c at the base
    /// cells behind the staging. Writes those batches' host `q_tape`, so the
    /// tapes are marked stale.
    // ponytail: the baseline values are the host stamp's sum of the
    // constant batches, then reduced ahead of the rest of the cell: the sum
    // order differs from the full restamp (tolerance class, 01 1.6).
    fn uploadBase(self: *Self) !void {
        if (self.base_cells.len == 0) return;
        const l = self.layout;
        const scratch = try self.gpa.alloc(f64, 2 * l.ng + 2 * l.nr + self.n + 1);
        defer self.gpa.free(scratch);
        @memset(scratch, 0);
        const pl: Planes = .{
            .g_vals = scratch[0..l.ng],
            .c_vals = scratch[l.ng..][0..l.ng],
            .rhs = scratch[2 * l.ng ..][0..l.nr],
            .q_vec = scratch[2 * l.ng + l.nr ..][0..l.nr],
        };
        const x0 = scratch[2 * l.ng + 2 * l.nr ..];
        for (self.batches) |*bg| {
            if (bg.const_kernel == null) continue;
            for (self.all_batches) |b| if (b.ctx == bg.ctx) b.eval(b.ctx, &pl, 0, b.count, x0, self.last_t);
        }
        self.tape_fresh = false;
        const n_base = self.base_cells.len;
        const n_planes: usize = if (self.resident_charge) 2 else 1;
        const vals = try self.gpa.alloc(f64, n_planes * n_base);
        defer self.gpa.free(vals);
        for (self.base_cells, vals[0..n_base]) |cell, *v| v.* = pl.g_vals[cell];
        if (n_planes == 2) for (self.base_cells, vals[n_base..]) |cell, *v| {
            v.* = pl.c_vals[cell];
        };
        try self.d_stage.uploadAtAsync(vals.ptr, self.n_stage * @sizeOf(f64), vals.len * @sizeOf(f64), &self.stream);
        self.prof.up += vals.len * @sizeOf(f64);
        // `vals` is pageable and freed on return: the driver staged it, but
        // wait anyway so the bytes are on the device before any launch.
        try self.sync(.other);
    }

    /// Page-locked `[]f64` from the driver; release with `freePinned`.
    fn pinnedF64(kernel: *Raw, n: usize) ![]f64 {
        const bytes = try kernel.allocPinned(n * @sizeOf(f64));
        return @alignCast(std.mem.bytesAsSlice(f64, bytes));
    }

    /// A non-owning buffer `byte_off` bytes into `base`, for kernel arguments.
    /// Never `free` it.
    fn view(base: Buffer, byte_off: usize) Buffer {
        var v = base;
        v.bytes -= byte_off;
        v.handle = switch (@typeInfo(@TypeOf(base.handle))) {
            .int => base.handle + byte_off,
            else => @ptrFromInt(@intFromPtr(base.handle) + byte_off), // else: HIP hands out pointers
        };
        return v;
    }

    /// Plane buffer `buf` (all four planes) within `pin_planes`.
    fn pinnedBuf(self: *const Self, buf: u1) []f64 {
        const n = self.layout.cells(4);
        return self.pin_planes[@as(usize, buf) * n ..][0..n];
    }

    /// The four planes as slices of plane buffer `buf`.
    fn pinnedPlanes(self: *const Self, buf: u1) Planes {
        const l = self.layout;
        const b = self.pinnedBuf(buf);
        return .{
            .g_vals = b[l.off(.g)..][0..l.ng],
            .rhs = b[l.off(.rhs)..][0..l.nr],
            .c_vals = b[l.off(.c)..][0..l.ng],
            .q_vec = b[l.off(.q)..][0..l.nr],
        };
    }

    /// The planes the last eval wrote, which the attached circuit points at
    /// (`Circuit.attachGpu`, `Circuit.evalGpu`).
    pub fn planes(self: *const Self) Planes {
        return self.pinnedPlanes(self.cur);
    }

    /// Drains the stream and copies the current planes into the circuit's
    /// own allocations (`Host.planes`), which it returns for
    /// `Circuit.detachGpu` to rebind. Call once, before `deinit`.
    pub fn restorePlanes(self: *Self) Planes {
        if (comptime backend == null) return self.host_planes;
        // `.commit`/`.revert` latches stay queued unwaited. A fault here was
        // already reported or is moot.
        self.stream.synchronize() catch {};
        // ponytail: after a resident Newton eval (`evalGc`) g and c hold the
        // diagonal only; every reader of the circuit's planes evaluates
        // first. A reader that does not would need the planes downloaded
        // again here, while `d_planes` still holds them.
        copyPlanes(self.host_planes, self.planes());
        return self.host_planes;
    }

    fn copyPlanes(dst: Planes, src: Planes) void {
        @memcpy(dst.g_vals, src.g_vals);
        @memcpy(dst.c_vals, src.c_vals);
        @memcpy(dst.rhs, src.rhs);
        @memcpy(dst.q_vec, src.q_vec);
    }

    /// Frees every device and pinned buffer and `self`. Detach it first
    /// (`Circuit.detachGpu`, which takes the planes back). Device-side state
    /// is dropped; call `syncStatus` or `syncHostState` first to keep it.
    pub fn deinit(self: *Self) void {
        if (comptime backend == null) return;
        self.prof.print();
        // `.commit`/`.revert` latches stay queued unwaited; drain them before
        // their buffers go. A fault here was already reported or is moot.
        self.stream.synchronize() catch {};
        // Pinned memory first, while batch 0's handle that owns it is alive.
        const k0 = &self.batches[0].kernel;
        if (self.gc) |*gc| {
            if (self.lu_check) std.debug.print("gpu-gc: checked={d} mismatches={d}\n", .{ gc.checks, gc.mismatches });
            k0.freePinned(std.mem.sliceAsBytes(gc.kv_all));
            k0.freePinned(std.mem.sliceAsBytes(gc.a_all));
            for ([_]*Buffer{ &gc.d_k, &gc.d_kv, &gc.d_a }) |b| b.free();
            gc.combine.deinit();
            gc.gather.deinit();
        }
        for ([_][]f64{ self.pin_planes, self.pin_x[0], self.pin_x[1], self.pin_x2, self.pin_xold, self.pin_tape }) |p| {
            if (p.len > 0) k0.freePinned(std.mem.sliceAsBytes(p));
        }
        k0.freePinned(self.pin_flags);
        self.stream.deinit();
        for ([_]*Buffer{ &self.d_x[0], &self.d_x[1], &self.d_xold, &self.d_x2, &self.d_flags, &self.d_planes, &self.d_stage, &self.d_seg1, &self.d_seg2, &self.d_mid }) |b| b.free();
        self.reduce.deinit();
        for (self.batches) |*bg| bg.deinit(self.gpa);
        if (self.chk.len > 0) self.gpa.free(self.chk);
        self.gpa.free(self.base_cells);
        self.gpa.free(self.batches_owned);
        self.gpa.free(self.cpu_owned);
        self.gpa.destroy(self);
    }

    fn uploadBytes(kernel: *Raw, bytes: []const u8) !Buffer {
        // Empty arrays still need a valid kernel-argument pointer, and
        // drivers reject zero-byte allocations.
        var buf = try kernel.alloc(@max(bytes.len, 1));
        errdefer buf.free();
        if (bytes.len > 0) try buf.upload(bytes.ptr, bytes.len);
        return buf;
    }

    /// Stamps all four planes: the resident batches on the device, whose one
    /// download lands straight in the circuit's planes, then the host batches
    /// on top; the circuit adds the ground pin. Always zero and restamp, never the
    /// constant-Jacobian baseline. Limit devices read their resident lim
    /// plane while `lim_active`, like the host eval.
    ///
    /// ponytail: the host batches wait for the download instead of stamping
    /// beside it, which costs their eval time per iteration (sources: a few
    /// microseconds). A heavy ineligible batch next to resident ones would
    /// want private host planes and an add over its footprint again.
    fn evalOnGpu(self: *Self, x: []const f64, t: f64, charge: bool, want: ?GcReq) !?GcReq {
        if (comptime backend == null) return Error.NoGpuArtifacts;
        self.follow = null;
        if (self.poisoned) return error.GpuStateReject;
        if (self.params_dirty) try self.repack();
        self.prof.start();
        self.prof.evals += 1;
        if (charge) self.prof.charge_evals += 1 else self.gc_last = want;
        self.x_len = x.len;

        const pre = self.pre;
        self.pre = null;
        // A host step since the prefetch (`commit_state`, a revert) leaves
        // it reading old rows.
        const pushed = try self.pushStepped();
        var served = want;
        // The prefetch ran on the same device state with the same launches,
        // so its planes are bitwise what this eval would produce. Bytes, not
        // `==`: -0.0 and +0.0 compare equal but may stamp differently.
        if (!pushed and pre != null and gcFits(pre.?.gc, want) and std.meta.eql(pre.?.sim, self.simAt(t)) and
            std.mem.eql(u8, std.mem.sliceAsBytes(x), std.mem.sliceAsBytes(self.pin_x[self.cur ^ 1][0..x.len])))
        {
            self.cur ^= 1;
            self.tape_buf = if (pre.?.charge) self.cur else null;
            self.prof.fused += 1;
            served = pre.?.gc;
            // A first iterate prefetched before the step's alpha was known
            // (`evalFollowsFirst`): form A again from the same planes.
            if (served) |p| if (!p.device and @as(u64, @bitCast(p.alpha)) != @as(u64, @bitCast(want.?.alpha))) {
                try self.enqueueCombine(self.cur, want.?.alpha);
                self.gc_buf[self.cur] = want;
                served = want;
                try self.sync(.eval);
            };
        } else {
            try self.enqueueEval(x, self.simAt(t), self.cur, charge, want);
            self.prof.lap("enqueue");
            if (charge) try self.sync(.charge) else try self.sync(.eval);
            self.prof.lap("wait");
        }
        const pl = self.planes();
        if (served) |r| {
            self.gcScatter(self.cur, r);
        } else if (!self.resident_charge and self.has_charge) {
            // The device writes neither c nor q when no resident batch has
            // charge; the host batches stamp them from zero.
            @memset(pl.c_vals, 0);
            @memset(pl.q_vec, 0);
        }
        self.tape_fresh = false;
        self.last_t = t;

        if (self.chk.len > 0) try self.evalCheck(x, t);

        // None under a resident matrix (`gcInit`).
        for (self.cpu_batches) |b| b.eval(b.ctx, &pl, 0, b.count, x, t);
        self.prof.lap("host");
        return served;
    }

    /// Whether planes a prefetch left under `have` serve an eval that asks
    /// for `want`: whole planes serve anything, and a resident matrix serves
    /// a request for the same side (a host-side A at another alpha is formed
    /// again, `evalOnGpu`).
    fn gcFits(have: ?GcReq, want: ?GcReq) bool {
        const h = have orelse return true;
        const w = want orelse return false;
        return h.device == w.device;
    }

    /// `evalOnGpu` for a Newton iterate whose matrix is `G + req.alpha·C`
    /// (02 2.3, `Circuit.evalNewtonGC`). Returns what the planes hold:
    /// `full` when the query cannot take a resident matrix, `failed` as
    /// `eval`'s false. `diag` (n slots) and `nnz` are the circuit's, the
    /// same every call.
    pub noinline fn evalGc(self: *Self, x: []const f64, t: f64, sim: SimState, req: GcReq, diag: []const u32, nnz: u32) GcServed {
        self.query_hit = null;
        const t0 = self.auditStart();
        defer self.auditEnd(t0);
        self.sim = sim;
        const want: ?GcReq = if (self.gcInit(diag, nnz)) req else null;
        const got = self.evalOnGpu(x, t, false, want) catch |e| {
            self.evalFailed(e);
            return .failed;
        };
        const r = got orelse return .full;
        return if (r.device) .device else .host_a;
    }

    /// Allocates `gc` on the first call; false for good when the query
    /// cannot take a resident matrix: host batches stamp g and c anywhere,
    /// the eval check compares whole planes, and without device charge
    /// planes there is no c to combine.
    fn gcInit(self: *Self, diag: []const u32, nnz: u32) bool {
        if (self.gc != null) return !self.poisoned;
        if (self.gc_off) return false;
        self.gc_off = true;
        if (!self.fuse or self.poisoned or !self.resident_charge or self.cpu_batches.len != 0 or self.chk.len > 0) return false;
        self.gc = self.gcAlloc(diag, nnz) catch return false;
        self.gc_off = false;
        return true;
    }

    fn gcAlloc(self: *Self, diag: []const u32, nnz: u32) !Gc {
        const be = backend.?;
        std.debug.assert(diag.len == self.n);
        var combine = try gompute.rawKernelByName(be, "arp_gc_combine", self.batches[0].kernel.context.ordinal);
        errdefer combine.deinit();
        var gather = try gompute.rawKernelByName(be, "arp_gc_gather", self.batches[0].kernel.context.ordinal);
        errdefer gather.deinit();
        const k0 = &self.batches[0].kernel;
        const nk: usize = @max(diag.len, 1);
        var d_k = try k0.alloc(nk * @sizeOf(u32));
        errdefer d_k.free();
        var d_kv = try k0.alloc(2 * nk * @sizeOf(f64));
        errdefer d_kv.free();
        var d_a = try k0.alloc(@max(nnz, 1) * @sizeOf(f64));
        errdefer d_a.free();
        const kv_all = try pinnedF64(k0, 4 * nk);
        errdefer k0.freePinned(std.mem.sliceAsBytes(kv_all));
        const na: usize = @max(nnz, 1);
        const a_all = try pinnedF64(k0, 2 * na);
        errdefer k0.freePinned(std.mem.sliceAsBytes(a_all));
        try self.upload(&d_k, std.mem.sliceAsBytes(diag));
        try self.sync(.other);
        return .{
            .combine = combine,
            .gather = gather,
            .d_k = d_k,
            .d_kv = d_kv,
            .d_a = d_a,
            .kv_all = kv_all,
            .a_all = a_all,
            .pin_kv = .{ kv_all[0 .. 2 * nk], kv_all[2 * nk ..] },
            .pin_a = .{ a_all[0..nnz], a_all[na..][0..nnz] },
            .k = diag,
            .nnz = nnz,
        };
    }

    /// Gather slots a request downloads: the whole diagonal under the device
    /// LU, else only the ground pin's.
    fn gcCount(self: *const Self, r: GcReq) usize {
        return if (r.device) self.gc.?.k.len else @min(self.gc.?.k.len, 1);
    }

    /// Queues the planes a request needs into plane buffer `buf`: rhs and q
    /// whole, g and c at the gather slots, and A unless the device LU forms
    /// it. Under `ESPICE_GPU_LU_CHECK` the whole planes come down too.
    fn enqueueGcDown(self: *Self, buf: u1, r: GcReq) !void {
        const l = self.layout;
        const gc = &self.gc.?;
        const pb = self.pinnedBuf(buf);
        if (self.lu_check) {
            try self.d_planes.downloadAtAsync(pb.ptr, 0, self.n_cells * @sizeOf(f64), &self.stream);
            self.prof.down += self.n_cells * @sizeOf(f64);
        } else {
            var rhs = view(self.d_planes, l.off(.rhs) * @sizeOf(f64));
            try rhs.downloadAtAsync(pb[l.off(.rhs)..].ptr, 0, l.nr * @sizeOf(f64), &self.stream);
            var q = view(self.d_planes, l.off(.q) * @sizeOf(f64));
            try q.downloadAtAsync(pb[l.off(.q)..].ptr, 0, l.nr * @sizeOf(f64), &self.stream);
            self.prof.down += 2 * l.nr * @sizeOf(f64);
        }
        const count = self.gcCount(r);
        if (count > 0) {
            var n_k: u64 = count;
            var c_plane = view(self.d_planes, l.off(.c) * @sizeOf(f64));
            try gc.gather.launchOn(&self.stream, gompute.Dim3.linear(count, gc_block), .{ .x = gc_block }, 0, &.{
                gompute.interface.arg(&n_k), gc.d_k.argPtr(), self.d_planes.argPtr(), c_plane.argPtr(), gc.d_kv.argPtr(),
            });
            try gc.d_kv.downloadAtAsync(gc.pin_kv[buf].ptr, 0, 2 * count * @sizeOf(f64), &self.stream);
            self.prof.down += 2 * count * @sizeOf(f64);
        }
        if (!r.device) try self.enqueueCombine(buf, r.alpha);
    }

    /// Queues `A = g + alpha·c` over the last eval's planes into `gc.d_a`
    /// and its download into plane buffer `buf`'s pinned A.
    fn enqueueCombine(self: *Self, buf: u1, alpha: f64) !void {
        const gc = &self.gc.?;
        if (gc.nnz == 0) return;
        var n: u64 = gc.nnz;
        var a = alpha;
        var c_plane = view(self.d_planes, self.layout.off(.c) * @sizeOf(f64));
        try gc.combine.launchOn(&self.stream, gompute.Dim3.linear(gc.nnz, gc_block), .{ .x = gc_block }, 0, &.{
            gompute.interface.arg(&n), gompute.interface.arg(&a), self.d_planes.argPtr(), c_plane.argPtr(), gc.d_a.argPtr(),
        });
        try gc.d_a.downloadAtAsync(gc.pin_a[buf].ptr, 0, @as(usize, gc.nnz) * @sizeOf(f64), &self.stream);
        self.prof.down += @as(usize, gc.nnz) * @sizeOf(f64);
    }

    /// Writes the gathered g and c into plane buffer `buf`'s planes.
    fn gcScatter(self: *Self, buf: u1, r: GcReq) void {
        const gc = &self.gc.?;
        const pl = self.pinnedPlanes(buf);
        const count = self.gcCount(r);
        const kv = gc.pin_kv[buf];
        for (gc.k[0..count], kv[0..count], kv[count..][0..count]) |s, g, c| {
            pl.g_vals[s] = g;
            pl.c_vals[s] = c;
        }
    }

    /// `Circuit.newtonGC` after `evalGc` served `got` (`host_a` or
    /// `device`), the ground pin added: under `host_a` the pinned A with
    /// the ground pin's entry formed again from the planes; under `device`
    /// `out` with the diagonal formed from the planes (the device LU forms
    /// the rest, `gcSolve`). `diag` and `nnz` as `evalGc`'s.
    pub fn gcVals(self: *Self, got: GcServed, alpha: f64, out: []f64, diag: []const u32, nnz: u32) []f64 {
        const pl = self.planes();
        if (got == .host_a) {
            const a = self.gc.?.pin_a[self.cur];
            if (diag.len > 0) a[diag[0]] = pl.g_vals[diag[0]] + alpha * pl.c_vals[diag[0]];
            if (self.lu_check) self.gcCheck(a, alpha, &.{}, &.{});
            return a[0..nnz];
        }
        std.debug.assert(got == .device);
        for (diag) |s| out[s] = pl.g_vals[s] + alpha * pl.c_vals[s];
        return out;
    }

    /// `Circuit.deviceSolve` after `evalGc` served `device`: the device LU
    /// forms A from this eval's planes and `vals` on the diagonal
    /// (`GpuLu.solveResident`). On a null or false return `vals` holds all
    /// of A. A driver fault stamps g and c on the host at the eval's x and
    /// combines them (`gcRescue`) before returning null.
    pub noinline fn gcSolve(self: *Self, lu: *gpu_lu.GpuLu, slv: *direct.Solver, vals: []f64, rhs: []const f64, dx: []f64, alpha: f64) ?bool {
        const gc = &self.gc.?;
        const c_plane = view(self.d_planes, self.layout.off(.c) * @sizeOf(f64));
        const r: gpu_lu.Resident = .{ .alpha = alpha, .g = self.d_planes, .c = c_plane, .k = gc.d_k, .k_host = gc.k };
        const out = lu.solveResident(slv, vals, rhs, dx, r) catch {
            self.gcRescue(vals, alpha);
            return null;
        };
        if (self.lu_check and out == true) self.gcCheck(slv.vcopy, alpha, vals, gc.k);
        return out;
    }

    /// `gcSolve`'s fault path: g and c stamped on the host at the last
    /// eval's x into scratch, combined into `vals` off the diagonal, which
    /// keeps the host's own entries. Tolerance class (the host stamp's sum
    /// order), as every fault fallback.
    fn gcRescue(self: *Self, vals: []f64, alpha: f64) void {
        const l = self.layout;
        const scratch = self.gpa.alloc(f64, 2 * l.ng + 2 * l.nr) catch @panic("OOM in the device-LU fault fallback");
        defer self.gpa.free(scratch);
        @memset(scratch, 0);
        const pl: Planes = .{ .g_vals = scratch[0..l.ng], .c_vals = scratch[l.ng..][0..l.ng], .rhs = scratch[2 * l.ng ..][0..l.nr], .q_vec = scratch[2 * l.ng + l.nr ..][0..l.nr] };
        const x = self.pin_x[self.cur][0..self.x_len];
        for (self.all_batches) |b| walk.stampRange(b, &pl, 0, b.count, x, self.last_t, .full);
        const gc = &self.gc.?;
        const keep = self.gpa.alloc(f64, gc.k.len) catch @panic("OOM in the device-LU fault fallback");
        defer self.gpa.free(keep);
        for (gc.k, keep) |s, *v| v.* = vals[s];
        for (vals[0..gc.nnz], pl.g_vals[0..gc.nnz], pl.c_vals[0..gc.nnz]) |*v, g, c| v.* = g + alpha * c;
        for (gc.k, keep) |s, v| vals[s] = v;
    }

    /// `ESPICE_GPU_LU_CHECK` (the whole planes came down too): compares the
    /// device-formed `a` with `G + alpha·C` from the planes, overlaid with
    /// `over` at `slots`, bitwise.
    fn gcCheck(self: *Self, a: []const f64, alpha: f64, over: []const f64, slots: []const u32) void {
        const gc = &self.gc.?;
        const pl = self.planes();
        gc.checks += 1;
        const ref = self.gpa.alloc(f64, gc.nnz) catch return;
        defer self.gpa.free(ref);
        for (ref, pl.g_vals[0..gc.nnz], pl.c_vals[0..gc.nnz]) |*v, g, c| v.* = g + alpha * c;
        for (slots) |s| ref[s] = over[s];
        for (ref, a[0..gc.nnz], 0..) |u, v, i| if (@as(u64, @bitCast(u)) != @as(u64, @bitCast(v))) {
            gc.mismatches += 1;
            if (gc.mismatches <= 3) std.debug.print("gpu-gc check: A slot {d} differs from the host ({e} vs {e})\n", .{ i, v, u });
            return;
        };
    }

    /// The circuit's analysis state at eval time `t`, as the host batches
    /// build it (`DeviceBatch.simAt`).
    fn simAt(self: *const Self, t: f64) SimState {
        var sim = self.sim;
        sim.t = t;
        return sim;
    }

    /// Enqueues the device half without waiting: upload x, clear the stage,
    /// launch every resident batch, reduce, download into the pinned planes.
    /// Separate so `evalCheck` can replay the same pass.
    fn enqueueEval(self: *Self, x: []const f64, sim_at: SimState, buf: u1, charge: bool, gc: ?GcReq) !void {
        if (comptime backend == null) return Error.NoGpuArtifacts;
        _ = try self.pushStepped();
        @memcpy(self.pin_x[buf][0..x.len], x);
        try self.d_x[buf].uploadAtAsync(self.pin_x[buf].ptr, 0, x.len * @sizeOf(f64), &self.stream);
        self.prof.up += x.len * @sizeOf(f64);
        // No staging clear: the kernels store each cell they own every pass,
        // and the cells none writes hold `init`'s zero (01 1.10).
        try self.phaseLap(0);

        for (self.batches) |*bg| {
            if (bg.count == 0) continue;
            try self.syncSeededLim(bg);
            // Scalars are passed by pointer and must outlive the launch call.
            var count: u64 = bg.count;
            var sim = sim_at;
            var limiting: u64 = @intFromBool(bg.lim_active);
            try bg.launch(if (bg.const_kernel) |*k| k else &bg.kernel, &self.stream, &.{
                gompute.interface.arg(&count),
                gompute.interface.arg(&sim),
                self.d_x[buf].argPtr(),
                bg.d_gath.argPtr(),
                bg.d_rhs_idx.argPtr(),
                bg.d_slots.argPtr(),
                bg.d_models.argPtr(),
                bg.d_model_of.argPtr(),
                bg.d_instances.argPtr(),
                self.v_stage[0].argPtr(),
                self.v_stage[2].argPtr(),
                self.v_stage[1].argPtr(),
                self.v_stage[3].argPtr(),
                bg.d_lim.argPtr(),
                gompute.interface.arg(&limiting),
            });
        }

        try self.phaseLap(1);
        try self.launchReduce(&self.d_seg1, &self.d_stage, &self.d_mid, self.n_pieces);
        try self.phaseLap(2);
        try self.launchReduce(&self.d_seg2, &self.d_mid, &self.d_planes, self.n_cells);
        try self.phaseLap(3);
        self.gc_buf[buf] = gc;
        if (gc) |r| try self.enqueueGcDown(buf, r) else {
            try self.d_planes.downloadAtAsync(self.pinnedBuf(buf).ptr, 0, self.n_cells * @sizeOf(f64), &self.stream);
            self.prof.down += self.n_cells * @sizeOf(f64);
        }
        try self.phaseLap(4);
        // `pin_tape` is about to hold buffer `buf`'s tape, or stale bytes.
        self.tape_buf = null;
        if (charge) try self.enqueueTape(buf, sim_at);
    }

    /// Waits for this stream only: a context-wide sync would also wait on
    /// work this process does not own. Counted for `ESPICE_GPU_STATS`.
    fn sync(self: *Self, comptime why: Wait) !void {
        self.prof.syncs += 1;
        self.prof.waits[@intFromEnum(why)] += 1;
        try self.stream.synchronize();
    }

    /// Under `Prof.phases`, waits for the stream and books the time since
    /// the last lap to stage `i`.
    fn phaseLap(self: *Self, comptime i: usize) !void {
        if (!self.prof.phases) return;
        if (i == 0) self.prof.last = nowNs();
        try self.stream.synchronize();
        const t = nowNs();
        self.prof.phase[i] += t - self.prof.last;
        self.prof.last = t;
    }

    /// One `arp_reduce_*` launch: `out[i] = sum(in[seg[i]..seg[i+1]])`.
    fn launchReduce(self: *Self, seg: *Buffer, in: *Buffer, out: *Buffer, cells: usize) !void {
        if (cells == 0) return;
        var n_cells: u64 = cells;
        try self.reduce.launchOn(
            &self.stream,
            gompute.Dim3.linear(cells, block_size),
            .{ .x = block_size },
            0,
            &.{ gompute.interface.arg(&n_cells), seg.argPtr(), in.argPtr(), out.argPtr() },
        );
    }

    /// `ESPICE_GPU_EVAL_CHECK`: replays the device pass at the same x and
    /// prints the worst device plane cell that moved, when it beats the last
    /// record. Newton cannot converge on a stamp that is not a function of x,
    /// so any output here means the deterministic scatter regressed. Restores
    /// the first pass, so a checked run stamps what an unchecked one would.
    fn evalCheck(self: *Self, x: []const f64, t: f64) !void {
        if (comptime backend == null) return Error.NoGpuArtifacts;
        const first = self.pinnedBuf(self.cur)[0..self.n_cells];
        @memcpy(self.chk, first);
        try self.enqueueEval(x, self.simAt(t), self.cur, false, null);
        try self.sync(.check);
        var worst: f64 = 0;
        var where: usize = 0;
        for (self.chk, first, 0..) |a, b, i| {
            const d = @abs(a - b);
            if (d > worst) {
                worst = d;
                where = i;
            }
        }
        if (worst > self.chk_worst) {
            self.chk_worst = worst;
            std.debug.print(
                "gpu-check: eval is not reproducible at t={e:.6}: |Δ plane cell {d}|={e:.3} (was {e:.16}, now {e:.16})\n",
                .{ t, where, worst, self.chk[where], first[where] },
            );
        }
        @memcpy(first, self.chk);
    }

    /// Stamps all four planes at `x`, ground pin aside; with `charge`, also
    /// downloads the resident batches' charge tapes on the same wait, so the
    /// next `syncQTape` copies them without another (`Circuit.evalQ`).
    /// Serves `eval`, `evalNewton` and `evalQ`: the device path never uses
    /// the constant-Jacobian baseline. False after a driver fault (warned
    /// once) or once poisoned: no analysis can unwind from one mid-solve, so
    /// the caller restamps on the host instead (`Circuit.evalGpu`).
    pub noinline fn eval(self: *Self, x: []const f64, t: f64, sim: SimState, charge: bool) bool {
        self.query_hit = null;
        const t0 = self.auditStart();
        defer self.auditEnd(t0);
        self.sim = sim;
        _ = self.evalOnGpu(x, t, charge, null) catch |e| {
            self.evalFailed(e);
            return false;
        };
        return true;
    }

    fn evalFailed(self: *Self, e: anyerror) void {
        if (!self.warned_eval) {
            self.warned_eval = true;
            std.debug.print(
                "warning: GPU device eval failed ({s}); falling back to the CPU stamp\n",
                .{@errorName(e)},
            );
        }
        // The host stamp the caller runs next writes every batch's
        // planes and `q_tape` at x; nothing the failed pass queued may
        // stand in for it.
        self.pre = null;
        self.tape_buf = null;
        self.tape_fresh = true;
    }

    fn auditStart(self: *const Self) u64 {
        const a = self.audit orelse return 0;
        return if (a.done or self.prof.evals < Audit.from) 0 else nowNs();
    }

    /// Books the time since `t0` and, at `Audit.at` evals, demotes the
    /// query to the host when the device costs more per eval than the host
    /// was priced at.
    fn auditEnd(self: *Self, t0: u64) void {
        if (t0 == 0) return;
        const a = &self.audit.?;
        a.spent_ns += nowNs() - t0;
        if (self.prof.evals < Audit.at) return;
        a.done = true;
        const gpu_us = @as(f64, @floatFromInt(a.spent_ns)) * 1e-3 / @as(f64, @floatFromInt(self.prof.evals - Audit.from));
        const losing = gpu_us > a.cpu_us;
        CostCache.write(a.key, a.knobs, .{ .cpu_us = a.cpu_us, .gpu_us = gpu_us });
        if (self.prof.on) std.debug.print("gpu-stats: measured {d:.0} us per device eval against the CPU's {d:.0}: {s}\n", .{ gpu_us, a.cpu_us, if (losing) "the rest of the query runs on the CPU" else "GPU" });
        if (losing and a.demote) self.demote();
    }

    /// Moves the rest of the query to the host through the fault fallback,
    /// after bringing the device state home, so it continues exactly where
    /// the device left it. Stays on the device if that download fails.
    fn demote(self: *Self) void {
        self.syncHostState() catch return;
        self.demoted = true;
        self.poisoned = true;
        // The fallbacks' warnings are for faults, not for a price.
        self.warned_eval = true;
        self.warned_state = true;
        self.pre = null;
        self.follow = null;
        self.query_hit = null;
        // The host evals refill every `q_tape` from here on.
        self.tape_fresh = true;
    }

    /// Stamps only the charges at `x` (`Circuit.evalQ`): q and the tapes
    /// from `arp_qonly_<model>` at x unlimited, as the host's charge pass,
    /// then the host batches' charge stamps; g, c and rhs keep the last
    /// eval's values. False under `ESPICE_GPU_NOFUSE`, when some resident
    /// batch has no charge kernel, or
    /// on a driver fault (warned once): the caller then runs the full eval.
    pub noinline fn evalQ(self: *Self, x: []const f64, t: f64, sim: SimState) bool {
        self.query_hit = null;
        const t0 = self.auditStart();
        defer self.auditEnd(t0);
        // `ESPICE_GPU_NOFUSE` keeps the full pass, for A/B runs.
        if (!self.fuse or !self.q_only or self.poisoned) return false;
        self.sim = sim;
        self.evalQOnGpu(x, t) catch |e| {
            if (!self.warned_eval) {
                self.warned_eval = true;
                std.debug.print("warning: GPU charge eval failed ({s}); falling back to the full eval\n", .{@errorName(e)});
            }
            self.pre = null;
            self.tape_buf = null;
            self.q_only = false;
            return false;
        };
        return true;
    }

    fn evalQOnGpu(self: *Self, x: []const f64, t: f64) !void {
        if (comptime backend == null) return Error.NoGpuArtifacts;
        self.follow = null;
        // The tape below lands in `pin_tape`, which a prefetch also targets.
        self.pre = null;
        if (self.params_dirty) try self.repack();
        try self.enqueueCharge(x, t);
        self.prof.lap("enqueue");
        try self.sync(.charge);
        self.prof.lap("wait");
        self.finishCharge(x, t);
        self.prof.lap("host");
    }

    /// The device half of `evalQOnGpu`: x up, the charge kernels, the q
    /// reduce and the q plane and tapes down, unwaited.
    fn enqueueCharge(self: *Self, x: []const f64, t: f64) !void {
        _ = try self.pushStepped();
        self.prof.start();
        self.prof.evals += 1;
        self.prof.charge_evals += 1;
        const buf = self.cur;
        // `d_x[cur]` holds the x behind the planes' charge, which a lazy
        // tape pass reads.
        @memcpy(self.pin_x[buf][0..x.len], x);
        try self.d_x[buf].uploadAtAsync(self.pin_x[buf].ptr, 0, x.len * @sizeOf(f64), &self.stream);
        self.prof.up += x.len * @sizeOf(f64);
        const sim_at = self.simAt(t);
        var off: usize = 0;
        for (self.batches) |*bg| {
            const k = if (bg.q_kernel) |*k| k else continue;
            if (bg.count == 0) continue;
            var count: u64 = bg.count;
            var sim = sim_at;
            var with_tape: u64 = @intFromBool(bg.tape.len > 0);
            try bg.launch(k, &self.stream, &.{
                gompute.interface.arg(&count),
                gompute.interface.arg(&sim),
                self.d_x[buf].argPtr(),
                bg.d_gath.argPtr(),
                bg.d_rhs_idx.argPtr(),
                bg.d_models.argPtr(),
                bg.d_model_of.argPtr(),
                bg.d_instances.argPtr(),
                self.v_stage[3].argPtr(),
                bg.d_tape.argPtr(),
                gompute.interface.arg(&with_tape),
            });
            if (bg.tape.len > 0) {
                try bg.d_tape.downloadAtAsync(self.pin_tape[off..].ptr, 0, bg.tape.len * @sizeOf(f64), &self.stream);
                self.prof.down += bg.tape.len * @sizeOf(f64);
            }
            off += bg.tape.len;
        }
        // Batches without a q kernel carry no charge, so their tapes are
        // empty and `pin_tape` is whole.
        std.debug.assert(off == self.pin_tape.len);
        // Level 1 over the q plane's pieces, level 2 over its cells: the
        // seg tables index the whole stage and the piece array absolutely.
        const l = self.layout;
        var seg1 = view(self.d_seg1, 2 * self.q_piece0 * @sizeOf(u32));
        var mid = view(self.d_mid, self.q_piece0 * @sizeOf(f64));
        try self.launchReduce(&seg1, &self.d_stage, &mid, self.n_pieces - self.q_piece0);
        var seg2 = view(self.d_seg2, 2 * l.off(.q) * @sizeOf(u32));
        var q_plane = view(self.d_planes, l.off(.q) * @sizeOf(f64));
        try self.launchReduce(&seg2, &self.d_mid, &q_plane, l.nr);
        try q_plane.downloadAtAsync(self.pinnedBuf(buf)[l.off(.q)..].ptr, 0, l.nr * @sizeOf(f64), &self.stream);
        self.prof.down += l.nr * @sizeOf(f64);
        self.tape_buf = buf;
    }

    /// The host half of `evalQOnGpu`, once the q plane has landed.
    fn finishCharge(self: *Self, x: []const f64, t: f64) void {
        self.tape_fresh = false;
        self.last_t = t;
        const pl = self.planes();
        for (self.cpu_batches) |b| walk.stampRange(b, &pl, 0, b.count, x, t, .charge);
    }

    /// The transient's accepted-point pass (`Circuit.acceptStates`): the
    /// charges at `x`, then every resident state half, then the accepted-
    /// state query, queued as one chain behind one wait (01 1.1), in the
    /// order the separate calls run them: the charges read the state before
    /// `updateState` latches it, and the query reads it after. The query's
    /// device half is kept for the `stateCtl(.query)` that follows. Returns
    /// null when the charge-only path is off (the caller then runs `evalQ`
    /// and `updateStates`), else what `updateStates` returns. A fault stamps
    /// the charges and walks the states of every batch on the host.
    pub noinline fn acceptStates(self: *Self, x: []const f64, t: f64, sim: SimState) ??f64 {
        self.query_hit = null;
        const t0 = self.auditStart();
        defer self.auditEnd(t0);
        if (!self.fuse or !self.q_only or self.poisoned) return null;
        self.sim = sim;
        return self.acceptStatesOnGpu(x, t) catch {
            self.warnStateFallback();
            self.pre = null;
            self.tape_buf = null;
            self.tape_fresh = true;
            self.query_hit = null;
            const pl = self.planes();
            @memset(pl.q_vec, 0);
            for (self.all_batches) |b| walk.stampRange(b, &pl, 0, b.count, x, t, .charge);
            return walk.updateBatches(self.all_batches, x);
        };
    }

    fn acceptStatesOnGpu(self: *Self, x: []const f64, t: f64) !?f64 {
        if (comptime backend == null) return Error.NoGpuArtifacts;
        self.follow = null;
        self.pre = null;
        if (self.params_dirty) try self.repack();
        try self.enqueueCharge(x, t);
        const states = try self.enqueueStates(x);
        var word1 = view(self.d_flags, 4);
        const query = try self.enqueueCtl(.query, &word1);
        if (states or query) {
            try self.d_flags.downloadAtAsync(self.pin_flags.ptr, 0, 8, &self.stream);
            self.prof.down += 8;
        }
        self.prof.lap("enqueue");
        try self.sync(.charge);
        self.prof.lap("wait");
        self.finishCharge(x, t);
        self.prof.lap("host");
        if (states and std.mem.readInt(u32, self.pin_flags[0..4], .little) & 2 != 0) {
            self.poisoned = true;
            return error.GpuStateReject;
        }
        const tr = try self.finishStates(x);
        if (query) self.query_hit = std.mem.readInt(u32, self.pin_flags[4..8], .little) != 0;
        return tr;
    }

    /// Fills every resident batch's host `q_tape` with the LTE charges at the
    /// last eval's x, which `d_x[cur]` still holds. Lazy: the transient reads the
    /// tape once per step, and Newton evals many times in between.
    fn syncTape(self: *Self) !void {
        if (comptime backend == null) return;
        if (self.tape_fresh or self.pin_tape.len == 0) return;
        if (self.poisoned) return error.GpuStateReject;
        if (self.tape_buf != self.cur) {
            try self.enqueueTape(self.cur, self.simAt(self.last_t));
            try self.sync(.tape);
        }
        var off: usize = 0;
        for (self.batches) |*bg| {
            @memcpy(bg.tape, self.pin_tape[off..][0..bg.tape.len]);
            off += bg.tape.len;
        }
        self.tape_fresh = true;
    }

    /// Queues every tape kernel at `d_x[buf]` and the download into
    /// `pin_tape`, which then holds buffer `buf`'s tape once the stream
    /// drains.
    fn enqueueTape(self: *Self, buf: u1, sim_at: SimState) !void {
        _ = try self.pushStepped();
        var off: usize = 0;
        for (self.batches) |*bg| {
            const k = if (bg.qtp_kernel) |*k| k else continue;
            if (bg.tape.len == 0) continue;
            var count: u64 = bg.count;
            var sim = sim_at;
            try bg.launch(k, &self.stream, &.{
                gompute.interface.arg(&count),
                gompute.interface.arg(&sim),
                self.d_x[buf].argPtr(),
                bg.d_gath.argPtr(),
                bg.d_models.argPtr(),
                bg.d_model_of.argPtr(),
                bg.d_instances.argPtr(),
                bg.d_tape.argPtr(),
            });
            try bg.d_tape.downloadAtAsync(self.pin_tape[off..].ptr, 0, bg.tape.len * @sizeOf(f64), &self.stream);
            self.prof.down += bg.tape.len * @sizeOf(f64);
            off += bg.tape.len;
        }
        self.tape_buf = buf;
    }

    /// Brings the resident batches' host `q_tape` up to the last eval. A
    /// fault leaves the resident tapes at their last value for this step,
    /// warns once, and poisons the context so every later step runs on the
    /// host.
    pub noinline fn syncQTape(self: *Self, sim: SimState) void {
        self.query_hit = null;
        self.sim = sim;
        self.syncTape() catch {
            self.warnStateFallback();
            self.poisoned = true;
        };
    }

    /// Uploads seed voltages a host `seedJunctions` left in the batch's lim
    /// plane and arms device limiting. Synchronous, so it lands before the
    /// launch that reads it.
    fn syncSeededLim(self: *Self, bg: *BatchGpu) !void {
        if (!bg.lim_dirty) return;
        bg.lim_dirty = false;
        if (!bg.has_lim) return;
        const p = bg.payload(bg.ctx);
        if (!p.lim_active) return;
        try self.upload(&bg.d_lim, std.mem.sliceAsBytes(p.lim_x));
        bg.lim_active = true;
    }

    /// Copies `bytes` (any host memory) to the start of `buf` on the
    /// context's stream, never the legacy default stream, which would fence
    /// every other stream on the device. From pageable memory the driver
    /// stages the bytes before returning, so the caller may reuse them at
    /// once.
    fn upload(self: *Self, buf: *Buffer, bytes: []const u8) !void {
        if (bytes.len == 0) return;
        try buf.uploadAtAsync(bytes.ptr, 0, bytes.len, &self.stream);
        self.prof.up += bytes.len;
    }

    /// Queues a copy of `buf`'s first `bytes.len` bytes into `bytes`; they
    /// land by the next `sync`. Asynchronous only when `bytes` is pinned
    /// (`BatchGpu.dev`); into pageable memory the driver blocks until done.
    fn download(self: *Self, buf: *Buffer, bytes: []u8) !void {
        if (bytes.len == 0) return;
        try buf.downloadAtAsync(bytes.ptr, 0, bytes.len, &self.stream);
        self.prof.down += bytes.len;
    }

    /// Limits the resident batches in one launch, walking the host batches
    /// meanwhile. Synchronous, because the converger reads the answer at
    /// once. No state half: the host stages state only at the converged
    /// iterate, and so does `updateStatesOnGpu`.
    ///
    /// After `evalFollows`, the eval at the limited x goes on the stream
    /// behind the limit pass, once the host limiters have finished with x,
    /// and one wait covers both (`evalOnGpu` picks it up). The device runs
    /// the same launches in the same order as the two-wait sequence, so the
    /// planes are bitwise the same. mos1_2000: 2 waits per Newton iteration
    /// become 1.
    fn applyLimitsOnGpu(self: *Self, x: []f64, x_old: []const f64) !bool {
        if (comptime backend == null) return false;
        const follow = self.follow;
        self.follow = null;
        self.pre = null;
        // A prefetch whose wait failed holds no planes.
        errdefer self.pre = null;
        if (self.poisoned) return error.GpuStateReject;
        if (self.params_dirty) try self.repack();
        const n = self.n;
        self.prof.start();
        self.prof.lims += 1;
        defer self.prof.lap("lim");

        var launched = false;
        for (self.batches) |*bg| {
            const lk = if (bg.lim_kernel) |*k| k else continue;
            // A state-only batch has nothing to run per iterate.
            if (bg.count == 0 or !bg.has_lim) continue;
            if (!launched) {
                // `x_old` explicitly: under JFNK the last eval was a
                // finite-difference probe.
                @memcpy(self.pin_x2[0..n], x[0..n]);
                try self.d_x2.uploadAtAsync(self.pin_x2.ptr, 0, n * @sizeOf(f64), &self.stream);
                @memcpy(self.pin_xold[0..n], x_old[0..n]);
                try self.d_xold.uploadAtAsync(self.pin_xold.ptr, 0, n * @sizeOf(f64), &self.stream);
                self.prof.up += 2 * n * @sizeOf(f64);
                try self.d_flags.fillAsync(0, 4, &self.stream);
            }
            try self.syncSeededLim(bg);
            var count: u64 = bg.count;
            var lim_active: u64 = @intFromBool(bg.lim_active);
            var halves: u64 = device_ir.lim_pass_limit;
            var sim = self.sim;
            try bg.launch(lk, &self.stream, &.{
                gompute.interface.arg(&count),
                self.d_x2.argPtr(),
                self.d_xold.argPtr(),
                bg.d_gath.argPtr(),
                bg.d_models.argPtr(),
                bg.d_model_of.argPtr(),
                bg.d_instances.argPtr(),
                bg.d_lim.argPtr(),
                bg.d_states.argPtr(),
                gompute.interface.arg(&lim_active),
                gompute.interface.arg(&halves),
                self.d_flags.argPtr(),
                gompute.interface.arg(&sim),
            });
            bg.lim_active = bg.lim_active or bg.has_lim;
            launched = true;
        }
        if (launched) {
            try self.d_flags.downloadAtAsync(self.pin_flags.ptr, 0, 4, &self.stream);
            self.prof.down += 4;
        }

        var any = walk.limitBatches(self.cpu_batches, x, x_old);

        if (launched) {
            // Without a limit launch there is no wait to share.
            if (follow) |f| try self.prefetch(f);
            try self.sync(.lim);
            const flags = std.mem.readInt(u32, self.pin_flags[0..4], .little);
            if (flags & 2 != 0) {
                // A step reject the GPU path cannot deliver: `gpuEligible`
                // admitted a device it should not have.
                self.poisoned = true;
                return error.GpuStateReject;
            }
            if (flags & 1 != 0) any = true;
        }
        return any;
    }

    /// Announces an eval at `x` under `sim` right after the next
    /// `applyLimits` or `stateCtl(.query)` (`Circuit.evalFollows`). `x` is
    /// borrowed until that call. A no-op under `ESPICE_GPU_NOFUSE`.
    pub noinline fn evalFollows(self: *Self, x: []const f64, sim: SimState, charge: bool) void {
        if (!self.fuse) return;
        self.follow = .{ .x = x, .sim = sim, .charge = charge };
    }

    /// Enqueues the eval `f` announced into the other plane buffer, behind
    /// whatever the stream holds; the caller's wait covers it.
    fn prefetch(self: *Self, f: Follow) !void {
        const gc = if (f.charge) null else self.gc_last;
        try self.enqueueEval(f.x, f.sim, self.cur ^ 1, f.charge, gc);
        self.pre = .{ .sim = f.sim, .charge = f.charge, .gc = gc };
    }

    /// Runs the device limit pass and the host walk on the rest. A fault
    /// falls back to the host walk over every batch; the resident batches'
    /// stale host lim/state self-heal, since
    /// limiting restarts from x_old and the latches recompute from x alone.
    pub noinline fn applyLimits(self: *Self, x: []f64, x_old: []const f64, sim: SimState) bool {
        self.query_hit = null;
        const t0 = self.auditStart();
        defer self.auditEnd(t0);
        self.sim = sim;
        return self.applyLimitsOnGpu(x, x_old) catch {
            self.warnStateFallback();
            return walk.limitBatches(self.all_batches, x, x_old);
        };
    }

    /// Runs every resident batch's state half at the converged `x` and the
    /// host walk on the rest. The converger stages at x_k, the iterate
    /// `newton` publishes, after its last limit pass ran at x_k+1; staging
    /// in that pass committed state one iterate ahead of the CPU and drifted
    /// tran/device_hfet_inverter's incremental Meyer charges 0.34% (#15). A
    /// resident batch cannot ask for a step reject: the kernel flags any
    /// non-`.ok` `updateState` as a fault instead.
    fn updateStatesOnGpu(self: *Self, x: []const f64) !?f64 {
        if (comptime backend == null) return walk.updateBatches(self.cpu_batches, x);
        if (self.poisoned) return error.GpuStateReject;
        // The host walk waits for the verdict: a second `updateState` reads
        // the held values the first one wrote, so the fallback's walk over
        // every batch must be the only one.
        if (try self.enqueueStates(x)) {
            try self.d_flags.downloadAtAsync(self.pin_flags.ptr, 0, 4, &self.stream);
            self.prof.down += 4;
            try self.sync(.state);
            if (std.mem.readInt(u32, self.pin_flags[0..4], .little) & 2 != 0) {
                self.poisoned = true;
                return error.GpuStateReject;
            }
        }
        return self.finishStates(x);
    }

    /// Queues the state half of every resident batch that has one at `x`,
    /// its verdict in flag word 0, unwaited. True when it launched any.
    fn enqueueStates(self: *Self, x: []const f64) !bool {
        const n = self.n;
        var launched = false;
        for (self.batches) |*bg| {
            const lk = if (bg.lim_kernel) |*k| k else continue;
            if (bg.count == 0 or !bg.has_state) continue;
            if (!launched) {
                // A prefetched eval read the state this launch rewrites.
                self.pre = null;
                if (self.params_dirty) try self.repack();
                @memcpy(self.pin_x2[0..n], x[0..n]);
                try self.d_x2.uploadAtAsync(self.pin_x2.ptr, 0, n * @sizeOf(f64), &self.stream);
                self.prof.up += n * @sizeOf(f64);
                try self.d_flags.fillAsync(0, 4, &self.stream);
            }
            var count: u64 = bg.count;
            var lim_active: u64 = 0;
            var halves: u64 = device_ir.lim_pass_state;
            var sim = self.sim;
            try bg.launch(lk, &self.stream, &.{
                gompute.interface.arg(&count),
                self.d_x2.argPtr(),
                self.d_x2.argPtr(),
                bg.d_gath.argPtr(),
                bg.d_models.argPtr(),
                bg.d_model_of.argPtr(),
                bg.d_instances.argPtr(),
                bg.d_lim.argPtr(),
                bg.d_states.argPtr(),
                gompute.interface.arg(&lim_active),
                gompute.interface.arg(&halves),
                self.d_flags.argPtr(),
                gompute.interface.arg(&sim),
            });
            launched = true;
        }
        return launched;
    }

    /// The host half of `updateStates`, after the device verdict.
    fn finishStates(self: *Self, x: []const f64) !?f64 {
        const host = walk.updateBatches(self.cpu_batches, x);
        try self.pullStepped();
        const stepped = walk.updateBatches(self.stepped_batches, x);
        // Now, so the eval `stateCtl(.query)` prefetches reads the new rows.
        if (try self.pushStepped()) self.pre = null;
        if (host == null) return stepped;
        return if (stepped) |s| @min(host.?, s) else host;
    }

    /// Runs the resident batches' state pass and walks the host ones. Every
    /// batch walks on the host once poisoned.
    pub noinline fn updateStates(self: *Self, x: []const f64, sim: SimState) ?f64 {
        self.query_hit = null;
        self.sim = sim;
        return self.updateStatesOnGpu(x) catch {
            self.warnStateFallback();
            return walk.updateBatches(self.all_batches, x);
        };
    }

    /// Disarms device limiting and walks the host batches' `clearLimits`.
    pub noinline fn clearLimits(self: *Self) void {
        // Keeps `query_hit`: Newton clears its limits between the fused
        // accepted-point pass and the query, and the lim planes are not
        // state the query reads.
        self.pre = null;
        for (self.batches) |*bg| {
            bg.lim_active = false;
            bg.lim_dirty = false;
        }
        walk.clearLimitBatches(if (self.poisoned) self.all_batches else self.cpu_batches);
    }

    /// Runs the accepted-step latch (path commit `pb <- wb, pq += wq`) on the
    /// resident instance blobs and the host walk on the rest. The kernels OR
    /// each instance's verdict into `d_flags`, so `.query` costs one launch
    /// and a 4-byte sync.
    fn stateCtlOnGpu(self: *Self, op: device_ir.StateCtlOp) !bool {
        if (comptime backend == null) return false;
        const follow = self.follow;
        self.follow = null;
        self.pre = null;
        errdefer self.pre = null;
        if (self.poisoned) return error.GpuStateReject;
        if (op == .commit) self.prof.steps += 1;
        const hit = self.query_hit;
        self.query_hit = null;
        if (op == .query) if (hit) |device_dirty| {
            // `acceptStates` already ran the device half on its wait.
            var dirty = walk.stateCtlBatches(self.cpu_batches, op);
            if (walk.stateCtlBatches(self.stepped_batches, op)) dirty = true;
            return dirty or device_dirty;
        };
        if (self.params_dirty) try self.repack();
        const launched = try self.enqueueCtl(op, &self.d_flags);
        // Every caller discards what `.commit` and `.revert` return, so they
        // stay on the stream unwaited; the next wait surfaces a fault. Only a
        // `.query` reads the flag, and it shares its wait with the charge
        // eval the transient announced.
        const wait = launched and op == .query;
        // Queued before the host walk so it drains meanwhile, and so a failed
        // enqueue happens before the walk: the fallback re-walks every batch,
        // and the path commit `pq += wq` is not idempotent.
        if (wait) {
            try self.d_flags.downloadAtAsync(self.pin_flags.ptr, 0, 4, &self.stream);
            self.prof.down += 4;
            if (follow) |f| try self.prefetch(f);
        }

        var dirty = walk.stateCtlBatches(self.cpu_batches, op);
        if (walk.stateCtlBatches(self.stepped_batches, op)) dirty = true;
        if (wait) {
            try self.sync(.query);
            if (std.mem.readInt(u32, self.pin_flags[0..4], .little) != 0) dirty = true;
        }
        return dirty;
    }

    /// Queues every resident `arp_ctl_<model>` with `op`, verdicts ORed into
    /// the word at `flags` (cleared first), unwaited. True when any ran.
    fn enqueueCtl(self: *Self, op: device_ir.StateCtlOp, flags: *Buffer) !bool {
        var launched = false;
        for (self.batches) |*bg| {
            const ck = if (bg.ctl_kernel) |*k| k else continue;
            if (bg.count == 0) continue;
            if (!launched) try flags.fillAsync(0, 4, &self.stream);
            var count: u64 = bg.count;
            var opv: u64 = @backingInt(op);
            try bg.launch(ck, &self.stream, &.{
                gompute.interface.arg(&count),
                bg.d_models.argPtr(),
                bg.d_model_of.argPtr(),
                bg.d_instances.argPtr(),
                bg.d_states.argPtr(),
                gompute.interface.arg(&opv),
                flags.argPtr(),
            });
            launched = true;
        }
        return launched;
    }

    /// Commits, reverts or queries the accepted state: accepted-step
    /// latches mutate device-resident instance blobs, so a host walk cannot
    /// stand in for a resident batch. A fault walks every batch on the host.
    pub noinline fn stateCtl(self: *Self, op: device_ir.StateCtlOp, sim: SimState) bool {
        self.sim = sim;
        return self.stateCtlOnGpu(op) catch {
            self.warnStateFallback();
            return walk.stateCtlBatches(self.all_batches, op);
        };
    }

    /// Every batch seeds the host `x`, then each resident lim plane is
    /// marked for upload before its next launch.
    pub noinline fn seedJunctions(self: *Self, x: []f64) void {
        self.query_hit = null;
        self.pre = null;
        walk.seedBatches(self.all_batches, x);
        for (self.batches) |*bg| {
            if (bg.has_lim) bg.lim_dirty = true;
        }
    }

    /// Merges the device Instance bytes of each host-stepped batch whose eval
    /// can latch a status into the host copy before a host walk reads them;
    /// bytes the host wrote since `base` win. Eval writes nothing else in a
    /// stepped batch, so the others need no download. One wait when any.
    fn pullStepped(self: *Self) !void {
        if (comptime backend == null) return;
        var any = false;
        for (self.batches) |*bg| {
            if (!bg.stepped or !bg.has_status) continue;
            try self.download(&bg.d_instances, bg.dev);
            any = true;
        }
        if (!any) return;
        try self.sync(.stepped);
        for (self.batches) |*bg| {
            if (!bg.stepped or !bg.has_status) continue;
            const host: []u8 = @constCast(bg.payload(bg.ctx).instances);
            std.debug.assert(bg.dev.len == host.len);
            _ = mergeInstances(host, bg.base, bg.dev);
            @memcpy(bg.base, bg.dev);
        }
    }

    /// Uploads each host-stepped batch whose host Instance bytes moved since
    /// `base` (an `updateState`, `commit_state` or revert on the host walk),
    /// merging device latches first when eval can write one. True when it
    /// uploaded anything. One memcmp per stepped batch when nothing moved.
    fn pushStepped(self: *Self) !bool {
        if (comptime backend == null) return false;
        var pushed = false;
        for (self.batches) |*bg| {
            if (!bg.stepped) continue;
            const host: []u8 = @constCast(bg.payload(bg.ctx).instances);
            std.debug.assert(bg.base.len == host.len);
            if (std.mem.eql(u8, host, bg.base)) continue;
            if (bg.has_status) {
                // Normally `pullStepped` already merged; this covers a host
                // write outside `updateStates` (a revert, `commit_state`).
                try self.download(&bg.d_instances, bg.dev);
                try self.sync(.stepped);
                _ = mergeInstances(host, bg.base, bg.dev);
            }
            try self.upload(&bg.d_instances, host);
            @memcpy(bg.base, host);
            pushed = true;
        }
        return pushed;
    }

    /// Brings each resident batch's host and device Instance bytes to the
    /// same value: a byte the host wrote since `BatchGpu.base` (a parameter
    /// write, `recompute`) wins, and every other byte takes the device's (its
    /// latches). A batch whose kernels never write instances downloads
    /// nothing, because the host copy is current, and uploads only under
    /// `push`.
    fn syncInstances(self: *Self, push: bool) !void {
        if (comptime backend == null) return;
        var any = false;
        for (self.batches) |*bg| {
            const p = bg.payload(bg.ctx);
            if (p.instances.len == 0) continue;
            if (bg.base.len == 0) {
                if (push) try self.upload(&bg.d_instances, p.instances);
                continue;
            }
            std.debug.assert(bg.dev.len == p.instances.len);
            try self.download(&bg.d_instances, bg.dev);
            any = true;
        }
        if (!any) return;
        try self.sync(.other);
        for (self.batches) |*bg| {
            if (bg.base.len == 0) continue;
            const host: []u8 = @constCast(bg.payload(bg.ctx).instances);
            if (mergeInstances(host, bg.base, bg.dev)) try self.upload(&bg.d_instances, host);
            @memcpy(bg.base, host);
        }
    }

    fn warnStateFallback(self: *Self) void {
        if (self.warned_state) return;
        self.warned_state = true;
        std.debug.print(
            "warning: GPU limit/state pass failed; falling back to the CPU walk\n",
            .{},
        );
    }

    /// Downloads the resident instances when a batch can latch a VerA
    /// `$fatal`/`$error` (`Hooks.status`), so the host reads and formats it
    /// as it would for a host batch. The device stores the latch with the
    /// same arithmetic as the host. Nothing to do after a fallback: the host
    /// walk then evaluated, and latched, on its own instances.
    pub fn syncStatus(self: *Self) void {
        self.query_hit = null;
        if (comptime backend == null) return;
        if (self.demoted) return;
        if (self.poisoned or self.warned_eval) return;
        for (self.batches) |bg| if (bg.has_status) {
            self.syncInstances(false) catch {};
            return;
        };
    }

    /// Downloads the resident instance, state and lim data into the host
    /// batches, so dependents copy an OP that matches the device. Fails with
    /// `error.GpuStateUnavailable` after any fallback, which can leave the two
    /// sides at different accepted points, or with the driver's error from a
    /// download, which can leave the host copy part-written.
    pub fn syncHostState(self: *Self) !void {
        self.query_hit = null;
        if (comptime backend == null) return;
        if (self.demoted) return;
        if (self.poisoned or self.warned_eval or self.warned_state)
            return error.GpuStateUnavailable;
        try self.sync(.other);
        try self.syncInstances(false);
        for (self.batches) |*batch| {
            const payload = batch.payload(batch.ctx);
            try self.download(&batch.d_states, @constCast(payload.states));
            if (batch.has_lim and !batch.lim_dirty) {
                if (batch.lim_active)
                    try self.download(&batch.d_lim, std.mem.sliceAsBytes(@constCast(payload.lim_x)));
                if (batch.set_limit_active) |set| set(batch.ctx, batch.lim_active);
            }
        }
        // Pageable destinations: the copies above already blocked, but say
        // so rather than lean on the driver.
        try self.sync(.other);
    }

    /// Re-uploads models after a host parameter change and merges the
    /// instances both ways (`syncInstances`), so the device keeps its
    /// latches. The tapes never change.
    fn repack(self: *Self) !void {
        if (comptime backend == null) return;
        for (self.batches) |*bg| {
            const p = bg.payload(bg.ctx);
            // Unsharing (a parameter write) grows `models` past the buffer.
            if (p.models.len != bg.models_len) {
                // The old buffer may still feed a queued launch.
                try self.sync(.other);
                var models = try bg.kernel.alloc(@max(p.models.len, 1));
                errdefer models.free();
                try self.upload(&models, p.models);
                bg.d_models.free();
                bg.d_models = models;
                try self.upload(&bg.d_model_of, std.mem.sliceAsBytes(p.model_of));
                bg.models_len = p.models.len;
            } else try self.upload(&bg.d_models, p.models);
        }
        try self.syncInstances(true);
        // A parameter can flip `jacConstApplies`, leaving a cell its batch
        // no longer writes.
        try self.d_stage.fillAsync(0, @max(self.n_stage, 1) * @sizeOf(f64), &self.stream);
        try self.uploadBase();
        self.params_dirty = false;
    }

    /// Marks the resident models and instances stale after a host
    /// parameter write; the context re-uploads before its next launch.
    pub noinline fn markDirty(self: *Self) void {
        self.query_hit = null;
        self.pre = null;
        self.params_dirty = true;
    }
};

/// Three-way merges one batch's Instance bytes into `host`: a byte that
/// differs from `base` was written by the host and stays; every other byte
/// takes `device`'s. Returns whether the host wrote any byte, so the device
/// needs the merged copy. All three are the same length.
// ponytail: byte granularity, so a field both sides rewrote since `base`
// would tear. The host writes parameters and derived values and the
// device writes latches, so no field has two writers today.
fn mergeInstances(host: []u8, base: []const u8, device: []const u8) bool {
    std.debug.assert(host.len == base.len and base.len == device.len);
    // Branchless, so it vectorizes: 0.065 against 0.40 ns/byte for the
    // `if`/`else` store (1 MiB, 1% host writes, i9-14900HX).
    var wrote: u8 = 0;
    for (host, base, device) |*h, b, d| {
        const mine = h.* != b;
        wrote |= @intFromBool(mine);
        h.* = if (mine) h.* else d;
    }
    return wrote != 0;
}

test "mergeInstances keeps host parameter writes and device latches" {
    // No padding: every byte compared is a field byte.
    const Inst = extern struct { param: f64, status: u64, pq: f64 };
    const base = [2]Inst{ .{ .param = 1, .status = 0, .pq = 0 }, .{ .param = 2, .status = 0, .pq = 0 } };
    // The device latched a status and committed charge on both...
    var device = base;
    device[0].status = 3;
    device[0].pq = 0.5;
    device[1].pq = -0.25;
    // ...while the host wrote instance 1's parameter after the last eval.
    var host = base;
    host[1].param = 7;
    try std.testing.expect(mergeInstances(std.mem.asBytes(&host), std.mem.asBytes(&base), std.mem.asBytes(&device)));
    try std.testing.expectEqual(Inst{ .param = 1, .status = 3, .pq = 0.5 }, host[0]);
    try std.testing.expectEqual(Inst{ .param = 7, .status = 0, .pq = -0.25 }, host[1]);

    // Nothing written on the host: a plain pull, and nothing to push.
    var pulled = base;
    try std.testing.expect(!mergeInstances(std.mem.asBytes(&pulled), std.mem.asBytes(&base), std.mem.asBytes(&device)));
    try std.testing.expectEqualSlices(u8, std.mem.asBytes(&device), std.mem.asBytes(&pulled));
}

test declineKind {
    try std.testing.expectEqual(Decline.policy, declineKind(Error.NotEnoughGpuWork));
    try std.testing.expectEqual(Decline.capability, declineKind(Error.CircuitNotEligible));
    try std.testing.expectEqual(Decline.machine, declineKind(Error.NoGpuArtifacts));
    // Driver and allocator errors never come from `Error`.
    try std.testing.expectEqual(Decline.machine, declineKind(error.InitFailed));
    try std.testing.expectEqual(Decline.machine, declineKind(error.OutOfMemory));
}

test requestSupported {
    try std.testing.expect(requestSupported(.cpu));
    try std.testing.expect(requestSupported(.auto));
    try std.testing.expectEqual(backend == .cuda, requestSupported(.cuda));
    try std.testing.expectEqual(backend == .hip, requestSupported(.hip));
}

test fallsBack {
    const errs = [_]InitError{ Error.NotEnoughGpuWork, Error.CircuitNotEligible, Error.NoGpuArtifacts, error.InitFailed };
    for (errs) |e| {
        // An explicit backend never runs on the CPU, nothing eligible included.
        try std.testing.expect(!fallsBack(.cuda, true, e));
        try std.testing.expect(!fallsBack(.hip, true, e));
        try std.testing.expect(fallsBack(.auto, false, e));
        try std.testing.expectEqual(declineKind(e) != .machine, fallsBack(.auto, true, e));
    }
}

test detectedHardware {
    // Whatever the machine has, the answer is one of the three names and
    // does not depend on the build's images.
    const name = detectedHardware();
    try std.testing.expect(std.mem.eql(u8, name, "none") or std.mem.eql(u8, name, "cuda") or std.mem.eql(u8, name, "hip"));
}

test qtpName {
    var buf: [32]u8 = undefined;
    try std.testing.expectEqualStrings("arp_qtp_mos1", qtpName(&buf, "arp_eval_mos1").?);
    try std.testing.expectEqualStrings("arp_qtp_", qtpName(&buf, "arp_eval_").?);
    try std.testing.expect(qtpName(&buf, "arp_lim_mos1") == null);
    try std.testing.expect(qtpName(&buf, "") == null);
    // Too long for the buffer: no tape kernel, not a panic.
    try std.testing.expect(qtpName(&buf, "arp_eval_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx") == null);
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
    try std.testing.expect(GpuContext.Cost.measuredWins(rec, 1000, 0));
    try std.testing.expect(!GpuContext.Cost.measuredWins(.{ .cpu_us = 30, .gpu_us = 120 }, 1000, 0));
}

test "Cost.gpuWins needs both margins and refuses NaN" {
    const W = GpuContext.Cost;
    // 10x per eval, cheap setup: the GPU.
    try std.testing.expect(W.gpuWins(1000, 100, 1000, 0));
    // Under the per-eval ratio, however many evals.
    try std.testing.expect(!W.gpuWins(199, 100, 1e9, 0));
    // Fast per eval, but setup outweighs the query.
    try std.testing.expect(!W.gpuWins(1000, 100, 10, cost_init_us));
    try std.testing.expect(!W.gpuWins(std.math.nan(f64), 100, 1000, 0));
    try std.testing.expect(!W.gpuWins(1000, std.math.nan(f64), 1000, 0));
}

/// Host model of the two `arp_reduce_*` levels over `order`'s tables: the
/// device planes an eval would download from `stage`.
fn reduceOnHost(gpa: std.mem.Allocator, order: GpuContext.Order, stage: []const f64, planes: []f64) !void {
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
        const order = try GpuContext.Order.build(gpa, &slot_tapes, &row_tapes, trash_slot, trash_row, layout, n_planes, &.{});
        defer order.deinit(gpa);
        const n_cells = layout.cells(n_planes);
        const plane_offs = [4]usize{ layout.off(.g), layout.off(.rhs), layout.off(.c), layout.off(.q) };
        // Every piece is one nonempty chunk of one cell's run.
        var pieces: usize = 0;
        for (slot_runs ++ [_]u32{0}) |r| pieces += (r + GpuContext.Order.chunk - 1) / GpuContext.Order.chunk;
        for (row_runs ++ [_]u32{0}) |r| pieces += (r + GpuContext.Order.chunk - 1) / GpuContext.Order.chunk;
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
                if (r <= GpuContext.Order.chunk) {
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
    const order = try GpuContext.Order.build(gpa, &.{&slots}, &.{&rows}, 2, 1, .{ .ng = 3, .nr = 2 }, 4, &.{ 0, 1 });
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
    GpuContext.Order.permute(&tapes, trash, &seg, &perm);
    // Cells 0, 1, 2, trash: runs of 2, 0, 3, 0; the two trash entries after.
    try std.testing.expectEqualSlices(u32, &.{ 0, 2, 2, 5, 5 }, &seg);
    try std.testing.expectEqualSlices(u32, &.{ 2, 5, 0, 3, 6, 1, 4 }, &perm);
}

test "Order.permute on empty tapes" {
    const tapes = [_][]const u32{ &.{}, &.{} };
    var seg: [3]u32 = undefined;
    var perm: [0]u32 = .{};
    GpuContext.Order.permute(&tapes, 1, &seg, &perm);
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
        GpuContext.Order.permute(&.{ all[0..cut], all[cut..] }, trash, &seg, perm[0..len]);
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

fn buildOrderAndFree(gpa: std.mem.Allocator, slot_tapes: []const []const u32, row_tapes: []const []const u32) !void {
    const order = try GpuContext.Order.build(gpa, slot_tapes, row_tapes, 2, 1, .{ .ng = 3, .nr = 2 }, 4, &.{1});
    order.deinit(gpa);
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
    const order = try GpuContext.Order.build(gpa, &.{&.{}}, &.{&.{}}, 1, 0, .{ .ng = 2, .nr = 1 }, 4, &.{});
    defer order.deinit(gpa);
    try std.testing.expectEqual(@as(usize, 0), order.n_stage);
    try std.testing.expectEqual(@as(usize, 0), order.n_pieces);
    // Every cell is an empty range of pieces.
    try std.testing.expectEqualSlices(u32, &@as([12]u32, @splat(0)), order.seg2);

    // Only the lengths are read before the bar, so a dangling base is safe.
    const huge = @as([*]const u32, @ptrFromInt(@alignOf(u32)))[0 .. std.math.maxInt(u32) - GpuContext.Order.chunk + 1];
    try std.testing.expectError(Error.CircuitNotEligible, GpuContext.Order.build(gpa, &.{huge}, &.{&.{}}, 0, 0, .{ .ng = 1, .nr = 1 }, 2, &.{}));
}

test "mergeInstances matches the bytewise rule at every length and offset" {
    var prng = std.Random.DefaultPrng.init(0xbeef);
    const rnd = prng.random();
    var host: [204]u8 = undefined;
    var base: [204]u8 = undefined;
    var dev: [204]u8 = undefined;
    var want: [204]u8 = undefined;
    for (0..201) |n| for (0..4) |off| {
        const h = host[off..][0..n];
        const b = base[3 - off ..][0..n];
        const d = dev[(off + 1) % 4 ..][0..n];
        rnd.bytes(b);
        rnd.bytes(d);
        @memcpy(h, b);
        var wrote = false;
        for (h) |*x| if (rnd.uintLessThan(u8, 16) == 0) {
            x.* +%= 1 + rnd.uintLessThan(u8, 255); // never a multiple of 256
            wrote = true;
        };
        for (want[0..n], h, b, d) |*w, x, y, z| w.* = if (x != y) x else z;
        try std.testing.expectEqual(wrote, mergeInstances(h, b, d));
        try std.testing.expectEqualSlices(u8, want[0..n], h);
    };
}

test "Cost.gpuWins at its bars and on infinities" {
    const W = GpuContext.Cost;
    // `cost_min_ratio` exactly passes.
    try std.testing.expect(W.gpuWins(cost_min_ratio * 100, 100, 1e6, 0));
    // Setup: 1.3 * (init + 1e5) < 1e6 holds below init = 669,230.8 us.
    try std.testing.expect(W.gpuWins(1000, 100, 1000, 669_000));
    try std.testing.expect(!W.gpuWins(1000, 100, 1000, 670_000));
    try std.testing.expect(!W.gpuWins(1000, std.math.inf(f64), 1000, 0));
    // Unbounded evals: inf < inf is false, so the CPU.
    try std.testing.expect(!W.gpuWins(1000, 100, std.math.inf(f64), 0));
}

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
        const got = GpuContext.Cost.admit(gpa, host, 1, c.evals, c.driver_up);
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

test "markWarm makes imageWarm true for that image only" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var root: [64]u8 = undefined;
    const knobs: Knobs = .{ .xdg_cache_home = try std.fmt.bufPrint(&root, ".zig-cache/tmp/{s}", .{&tmp.sub_path}) };
    try std.testing.expect(!imageWarm("a", knobs));
    markWarm("a", knobs);
    try std.testing.expect(imageWarm("a", knobs));
    try std.testing.expect(!imageWarm("b", knobs));
    // A second mark, with both directories present, is harmless.
    markWarm("a", knobs);
    try std.testing.expect(imageWarm("a", knobs));
}

test "without kernel images init declines as the machine and auto falls back" {
    if (comptime backend != null) return error.SkipZigTest;
    const empty: Planes = .{ .g_vals = &.{}, .c_vals = &.{}, .rhs = &.{}, .q_vec = &.{} };
    const host: Host = .{ .batches = &.{}, .planes = empty, .n = 0, .trash_slot = 0, .has_charge = false, .knobs = .{} };
    const got = GpuContext.init(std.testing.allocator, host, null, true, 1, 1e6);
    try std.testing.expectError(Error.NoGpuArtifacts, got);
    try std.testing.expect(!available);
    try std.testing.expect(fallsBack(.auto, false, Error.NoGpuArtifacts));
}

/// Private access for the analysis test suite.
pub const test_access = if (@import("builtin").is_test) .{
    .backend = backend,
} else {};
