//! The GPU lane backend's device state (dev/research/plans/phase4-gpu-lanes.md
//! G6): N transient lanes resident on one device, lane-major, with the
//! model-independent kernels of `solver/lane_device.zig` and each model's
//! lane kernels (`eval.zig` `arp_l*`). Per round the host uploads each
//! lane's step scalars, the device predicts, solves Newton for every lane,
//! and computes the step-end numbers (`probe`): the state-flip query, the
//! charge tape at the solution, the LTE bounds and the probe rows. The
//! host's `tran.Sim.end` decides each lane from them through `Ops`, and
//! `apply` runs the decisions' device mutations. The CPU lane path
//! (`tran.runLanes`) is the oracle; results match it within tolerance, not
//! bitwise.
//!
//! In: N lane circuits after `tran.Sim.init`, on one pattern. Out: per lane
//! Newton results and probe numbers; the lanes' device instances copied back
//! to the host at `deinit`.

const std = @import("std");
const device_ir = @import("device").abi;
const gompute = @import("gompute");
const solver = @import("solver");
const lk = solver.lane_kernels;
const direct = solver.direct;
const converger = solver.converger;
const root = @import("../types.zig");
const Circuit = root.Circuit;
const SimState = device_ir.SimState;
const StepReport = lk.StepReport;
const StateCtlOp = device_ir.StateCtlOp;

const passes = @import("measure").passes;
const lane_color = @import("lane_color.zig");
const hdl = @import("hdl.zig");
const devices = @import("devices.zig");
const backend = devices.backend;

/// This binary carries kernel images; whether they include the lane
/// kernels is found when `init` loads them.
pub const available = backend != null;

/// Differential hardware tests must distinguish device execution from a
/// CPU fallback. This counter does not exist in production builds.
pub const test_access = if (@import("builtin").is_test) struct {
    pub var admitted = std.atomic.Value(usize).init(0);
    /// Colour every batch whatever the card (`colorPolicy`).
    pub var force_color = false;
    /// What `staticPolicy` answers in tests.
    pub var static = false;
    /// Measure on the host under device step control, the oracle of the
    /// device `.meas` passes (06 6.8).
    pub var host_meas = false;
    /// Lane batches whose `.meas` passes ran on the device.
    pub var dev_meas = std.atomic.Value(usize).init(0);
} else struct {};

const Raw = if (backend) |be| gompute.RawByName(be) else void;
const Buffer = if (backend != null) Raw.Buffer else void;
const Stream = if (backend != null) Raw.Stream else void;
const GraphExec = if (backend != null) Raw.GraphExec else void;
const block: u32 = device_ir.gpu_lane_block_size;

comptime {
    std.debug.assert(@sizeOf(lk.SimState) == @sizeOf(SimState));
    std.debug.assert(@offsetOf(lk.SimState, "iteration") == @offsetOf(SimState, "iteration"));
}

/// Whether this Problem batches transients as GPU lanes: an explicit
/// `cuda`/`hip` does unless `ESPICE_LANES=0`; `auto` only on a device whose
/// FP64 rate is at least a quarter of FP32 (datacenter parts report 2; the
/// RTX 4060 Laptop 64, where E1 measured the GPU at 1.4-2x the 24-core CPU,
/// so `auto` keeps the CPU there).
pub fn policy(config: anytype) bool {
    if (config.knobs.lanes) |on| if (!on) return false;
    return switch (config.backend) {
        .cpu => false,
        .cuda, .hip => available,
        // An unknown ratio (HIP, no driver) keeps the CPU.
        .auto => available and (fp64Ratio() orelse return false) <= 4,
    };
}

/// `Config.gpu_share` for a Problem whose transients batch as GPU lanes,
/// with `cpus` host threads beside the cards: the cards' work over all of
/// it, each counted in host cores (`devices.table`). 1 (the devices keep
/// every point) when no device reports its FP64 ratio, and in tests, so the
/// hardware tests count every point onto the device.
pub fn share(cpus: usize) f64 {
    if (@import("builtin").is_test) return 1;
    const cards = devices.totalCap() orelse return 1;
    return cards / (cards + @as(f64, @floatFromInt(@max(cpus, 1))));
}

/// Whether `GpuLanes.init` evaluates a thread per (instance colour, lane)
/// (`lane_color.zig`): while N lanes underfill the card. Measured on the RTX
/// 4060 Laptop (24 SMs) with 80 mos6 in 9 colours: Newton -29 % at 512
/// lanes, +13 % at 1,335, +20 % at 4,096, +40 % at 8,665, so the crossover
/// is near 40 lanes per SM: the bar is 40 x SMs (960 here, 4,000-5,300 on
/// a 100-132 SM part), 1,024 when the driver does not say.
// ponytail: one card and one model measured; refit the 40 on another.
fn colorPolicy(n_lanes: usize) bool {
    if (@import("builtin").is_test and test_access.force_color) return true;
    const sms = multiprocessorCount() orelse return n_lanes < 1024;
    return n_lanes < 40 * @as(usize, sms);
}

/// The device's streaming multiprocessors (compute units on HIP 6.0+), or
/// null (older HIP, no driver).
fn multiprocessorCount() ?u32 {
    if (comptime @import("builtin").cpu.arch.isWasm()) return null;
    var compute = gompute.runtime.dynamic.Compute.init(null) catch return null;
    defer compute.deinit();
    return compute.multiprocessorCount() catch null;
}

/// The device's FP64:FP32 ratio, or null (HIP, no driver).
fn fp64Ratio() ?u32 {
    if (comptime @import("builtin").cpu.arch.isWasm()) return null;
    var compute = gompute.runtime.dynamic.Compute.init(null) catch return null;
    defer compute.deinit();
    return compute.fp64Ratio() catch null;
}

/// `arp_l<what>_<model>` for the batch whose eval kernel is
/// `arp_eval_<model>`, into `buf`; null when the name has another shape.
fn laneName(buf: []u8, eval_kernel: []const u8, what: []const u8) ?[:0]const u8 {
    const prefix = "arp_eval_";
    if (!std.mem.startsWith(u8, eval_kernel, prefix)) return null;
    return std.fmt.bufPrintSentinel(buf, "arp_l{s}_{s}", .{ what, eval_kernel[prefix.len..] }, 0) catch null;
}

/// Whether a Problem whose transients batch as GPU lanes also solves its
/// operating points and `.dc` sweeps there (06 6.9), with `cpus` host
/// threads beside the card: always under an explicit `cuda`/`hip`, as
/// `policy`; under `auto` only when the card does more instance
/// evaluations than they do, in `share`'s host-core units. A static point
/// is a few Newton iterates, so its per-lane setup (packing, upload, the
/// tape) costs as much as its solve and the card must outrun the whole
/// pool. Off under `auto` on the RTX 4060 Laptop (about 13 cores against
/// 32 threads).
pub fn staticPolicy(config: anytype, cpus: usize) bool {
    if (@import("builtin").is_test) return test_access.static;
    if (config.backend != .auto) return true;
    const ratio = fp64Ratio() orelse return false;
    return 13.0 * 64.0 / @as(f64, @floatFromInt(@max(ratio, 1))) > @as(f64, @floatFromInt(@max(cpus, 1)));
}

/// Why `ckt`'s pattern cannot solve operating points or `.dc` points as
/// GPU lanes, or null: the transient lanes' device rules
/// (`tran.gpuLaneRefusal`) without the step machine's.
pub fn staticRefusal(ckt: *const Circuit) ?[]const u8 {
    if (!available) return "no GPU images";
    if (ckt.vpi != null or ckt.display != null or ckt.load_check != null) return "VPI, display tasks or a load check";
    for (ckt.batches) |b| {
        const h = b.hooks;
        const p = h.gpu_payload orelse return "no device payload";
        if (h.commit_state != null or h.check_convergence != null or h.advance_iteration != null) return "Newton hooks";
        // `Circuit.load_check` exists only once a solve made the workspace.
        if (h.mark_load_check_rows != null) return "a load check";
        if (std.mem.eql(u8, b.type_name, "kinduc")) return "coupled inductors";
        if (p(b.ctx).lim_kernel.len == 0 and (h.update_state != null or h.state_ctl != null)) return "host-stepped";
    }
    return null;
}

/// Why `GpuLanes.init` refused before any driver error.
pub const Error = error{
    /// No kernel image for the lane kernels, or a batch without one.
    NoLaneImage,
    /// The lead lane holds no factored sparse LU to replay.
    NoLaneTape,
    /// Points do not share the sparse pattern and device scatter tapes.
    IncompatibleLanes,
};

/// One device batch over every lane: its shared tapes, its deduplicated
/// Model rows, lane-major Instance/State/lim rows, and its lane kernels.
const LaneBatch = struct {
    eval: Raw,
    lim: ?Raw,
    ctl: ?Raw,
    /// `arp_lchg_<model>`: charges into the q plane and the tape.
    chg: ?Raw,
    count: u32,
    has_lim: bool,
    /// This batch's offset into the charge tape.
    tape_off: u32,
    inst_size: usize,
    state_size: usize,
    d_models: Buffer,
    d_model_of: Buffer,
    d_instances: Buffer,
    d_states: Buffer,
    d_lim: Buffer,
    d_gath: Buffer,
    d_rhs_idx: Buffer,
    d_slots: Buffer,
    /// The batch's lim-active word: 1 between the predictor's limit pass
    /// and the end of Newton (`clearLimits`).
    d_limw: Buffer,
    /// `arp_lbrk_<model>` and its per-instance-and-lane source breakpoint
    /// caches, under device step control (`GpuLanes.initSteps`).
    brk: ?Raw = null,
    d_brk: ?Buffer = null,
    /// `Knobs.gpu_lane_color`: this batch's colours (`lane_color.zig`).
    color: ?Color = null,

    const Color = struct {
        /// Colour c is `d_order[starts[c]..starts[c + 1]]`.
        starts: []u32,
        d_order: Buffer,
        d_slots: Buffer,
        d_rows: Buffer,
        slot_hubs: Hubs,
        row_hubs: Hubs,

        const Hubs = struct {
            n: u32,
            d_cells: Buffer,
            d_starts: Buffer,
            d_entries: Buffer,
        };

        fn deinit(c: *Color, gpa: std.mem.Allocator) void {
            gpa.free(c.starts);
            inline for (.{ "d_order", "d_slots", "d_rows" }) |f| @field(c, f).free();
            inline for (.{ &c.slot_hubs, &c.row_hubs }) |h| inline for (.{ "d_cells", "d_starts", "d_entries" }) |f| @field(h, f).free();
        }
    };

    fn deinit(self: *LaneBatch) void {
        inline for (.{ "d_models", "d_model_of", "d_instances", "d_states", "d_lim", "d_gath", "d_rhs_idx", "d_slots", "d_limw" }) |f| @field(self, f).free();
        if (self.d_brk) |*b| b.free();
        if (self.color) |*c| c.deinit(std.heap.page_allocator);
        if (self.brk) |*k| k.deinit();
        if (self.lim) |*k| k.deinit();
        if (self.ctl) |*k| k.deinit();
        if (self.chg) |*k| k.deinit();
        self.eval.deinit();
    }
};

/// Whether `c` shares `c0`'s unknowns, sparse pattern and batch count, so
/// both can be lanes of one device batch.
fn samePattern(c: *const Circuit, c0: *const Circuit) bool {
    return c.n == c0.n and c.nnz == c0.nnz and c.batches.len == c0.batches.len and
        std.mem.eql(u32, c.col_ptr, c0.col_ptr) and std.mem.eql(u32, c.row_idx, c0.row_idx);
}

/// Whether batch payload `p` packs beside `p0`: the same kernel, row sizes
/// and scatter tapes (`pack` asserts the tapes).
fn sameRows(p: device_ir.GpuPayload, p0: device_ir.GpuPayload) bool {
    return p.count == p0.count and p.n_u == p0.n_u and
        p.instances.len == p0.instances.len and p.states.len == p0.states.len and
        p.lim_x.len == p0.lim_x.len and std.mem.eql(u8, p.kernel, p0.kernel) and
        std.mem.eql(u32, p.gath, p0.gath) and std.mem.eql(u32, p.rhs_idx, p0.rhs_idx) and
        std.mem.eql(u32, p.slots, p0.slots);
}

/// Lane-major rows of one batch over every lane.
const Packed = struct {
    models: []u8,
    model_of: []u32,
    instances: []u8,
    states: []u8,
    lim: []f64,

    fn deinit(p: Packed, gpa: std.mem.Allocator) void {
        gpa.free(p.models);
        gpa.free(p.model_of);
        gpa.free(p.instances);
        gpa.free(p.states);
        gpa.free(p.lim);
    }
};

/// Lane-major rows of batch b over every lane: Model rows deduplicated by
/// bytes (the "packed rows" of lane-batched-simulation.md §3.2),
/// model_of, instances, states and the lim plane. `ps[l]` is lane l's
/// `GpuPayload` for b; the map's keys borrow its frozen Model bytes.
fn pack(gpa: std.mem.Allocator, ps: []const device_ir.GpuPayload) !Packed {
    const p0 = ps[0];
    const n_lanes = ps.len;
    const count: usize = p0.count;
    var rows_n: u32 = 0;
    for (p0.model_of) |r| rows_n = @max(rows_n, r + 1);
    const ms = if (rows_n == 0) 0 else p0.models.len / rows_n;
    const is = if (count == 0) 0 else p0.instances.len / count;
    const ss = if (count == 0) 0 else p0.states.len / count;
    const kl = if (count == 0) 0 else p0.lim_x.len / count;
    var index: std.StringHashMapUnmanaged(u32) = .empty;
    defer index.deinit(gpa);
    var models: std.ArrayList(u8) = .empty;
    errdefer models.deinit(gpa);
    const model_of = try gpa.alloc(u32, count * n_lanes);
    errdefer gpa.free(model_of);
    const inst = try gpa.alloc(u8, count * n_lanes * is);
    errdefer gpa.free(inst);
    const states = try gpa.alloc(u8, count * n_lanes * ss);
    errdefer gpa.free(states);
    const lim = try gpa.alloc(f64, count * kl * n_lanes);
    errdefer gpa.free(lim);
    for (ps, 0..) |p, l| {
        std.debug.assert(p.count == p0.count and std.mem.eql(u32, p.gath, p0.gath) and std.mem.eql(u32, p.slots, p0.slots));
        for (0..count) |id| {
            const row = p.models[@as(usize, p.model_of[id]) * ms ..][0..ms];
            const gop = try index.getOrPut(gpa, row);
            if (!gop.found_existing) {
                gop.value_ptr.* = @intCast(models.items.len / @max(ms, 1));
                try models.appendSlice(gpa, row);
            }
            model_of[id * n_lanes + l] = gop.value_ptr.*;
            @memcpy(inst[(id * n_lanes + l) * is ..][0..is], p.instances[id * is ..][0..is]);
            if (ss != 0) @memcpy(states[(id * n_lanes + l) * ss ..][0..ss], p.states[id * ss ..][0..ss]);
            for (0..kl) |a| lim[(id * kl + a) * n_lanes + l] = p.lim_x[id * kl + a];
        }
    }
    return .{ .models = try models.toOwnedSlice(gpa), .model_of = model_of, .instances = inst, .states = states, .lim = lim };
}

/// The model-independent lane kernels.
const Kernels = struct {
    refactor: Raw,
    solve: Raw,
    assemble: Raw,
    gmin: Raw,
    clear_q: Raw,
    update: Raw,
    gates: Raw,
    finalize: Raw,
    settle: Raw,
    set_limw: Raw,
    predict: Raw,
    lte: Raw,
    accept: Raw,
    rebase: Raw,
    gather: Raw,
    peel_dense: Raw,
    pack: Raw,
    unpack: Raw,
    step_begin: Raw,
    newton_init: Raw,
    set_ops: Raw,
    slot_zero: Raw,
    step_end_a: Raw,
    step_end_b: Raw,
    step_stop: Raw,
    ring_put: Raw,
    ring_next: Raw,
    hub_reduce: Raw,
    meas: Raw,

    fn init() !Kernels {
        // Two inline loops over every field, the errdefer's unrolled at each `try`.
        @setEvalBranchQuota(10_000);
        var k: Kernels = undefined;
        var done: usize = 0;
        errdefer inline for (@typeInfo(Kernels).@"struct".field_names, 0..) |f, i| {
            if (i < done) @field(k, f).deinit();
        };
        inline for (@typeInfo(Kernels).@"struct".field_names) |f| {
            @field(k, f) = gompute.rawKernelByName(backend.?, "arp_lane_" ++ f, devices.ordinal) catch |e| return if (e == error.KernelNotFound) Error.NoLaneImage else e;
            done += 1;
        }
        return k;
    }

    fn deinit(k: *Kernels) void {
        inline for (@typeInfo(Kernels).@"struct".field_names) |f| @field(k, f).deinit();
    }
};

/// `ESPICE_GPU_STATS` wall time per round phase, printed at `deinit`.
pub const Prof = struct {
    on: bool = false,
    /// `ESPICE_GPU_STATS=phases`: direct launches with waits between stages.
    /// This diagnoses the kernel split; regular timing keeps graph replay.
    phases: bool = false,
    /// Nanoseconds: host step control, predict, Newton, probe, apply.
    ns: [5]u64 = @splat(0),
    tick_ns: [5]u64 = @splat(0),
    rounds: u64 = 0,
    ticks: u64 = 0,
    graph_ticks: u64 = 0,
    lane_rounds: u64 = 0,
    lane_ticks: u64 = 0,
    peels: u64 = 0,
    retapes: u64 = 0,
    /// Failed replays the device dense peel solved.
    dense_peels: u64 = 0,
    /// Device step control: the per-launch drain of the ring rows (record,
    /// `.meas` stream, early stop), within `ns[host]`.
    drain_ns: u64 = 0,
    last: u64 = 0,

    /// The `ns` slot `lap` charges.
    pub const Phase = enum(u3) { host, predict, newton, probe, apply };
    const TickPhase = enum(u3) { eval, assemble, lu, gates, state };

    /// Monotonic nanoseconds; 0 off Linux, where the stats read zero.
    pub fn now() u64 {
        if (comptime @import("builtin").os.tag != .linux) return 0;
        var ts: std.os.linux.timespec = undefined;
        _ = std.os.linux.clock_gettime(.MONOTONIC, &ts);
        return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
    }

    /// Charges the time since the last lap to `p`.
    pub fn lap(self: *Prof, p: Phase) void {
        if (!self.on) return;
        const t = now();
        if (self.last != 0) self.ns[@backingInt(p)] += t - self.last;
        self.last = t;
    }
};

/// One lane's Newton outcome, as `converger.Result` and its singular flag.
pub const LaneNewton = struct { result: converger.Result, singular: bool, reject: bool };

/// Device state of N transient lanes.
pub const GpuLanes = struct {
    gpa: std.mem.Allocator,
    ckts: []const *Circuit,
    n_lanes: u32,
    n: u32,
    nnz: u32,
    n_hist: u32,
    n_qt: u32,
    n_probes: u32,
    has_charge: bool,
    /// Stateful charge (`Circuit.has_state_q`): every accepted step re-reads
    /// q at cur (`rebaseRound`).
    state_q: bool,
    d0: u32,
    batches: []LaneBatch,
    k: Kernels,
    stream: Stream,
    tab: lk.LaneTab,

    // Lane-major: n + 1 rows (x, cur, prev, rhs, q), nnz + 1 slots (g, c).
    d_x: Buffer,
    d_x_old: Buffer,
    d_dx: Buffer,
    d_cur: Buffer,
    d_prev: Buffer,
    d_g: Buffer,
    d_c: Buffer,
    d_rhs: Buffer,
    d_q: Buffer,
    d_a: Buffer,
    d_lx: Buffer,
    d_ux: Buffer,
    d_udiag: Buffer,
    d_w: Buffer,
    d_y: Buffer,
    d_ring: Buffer,
    d_ip: Buffer,
    // The LTE tape ring, lane-major too: entry j of physical slot k at
    // `(k * n_qt + j) * N + l`, and `d_ip_qt` entry j at `j * N + l`.
    d_ring_qt: Buffer,
    d_ip_qt: Buffer,
    // Per lane.
    d_sc: Buffer,
    d_sims: Buffer,
    d_active: Buffer,
    d_round: Buffer,
    d_stage: Buffer,
    d_gate: Buffer,
    d_flags: Buffer,
    d_fail: Buffer,
    d_ops: Buffer,
    d_accept: Buffer,
    d_slot0: Buffer,
    d_perm_q: Buffer,
    d_perm_qt: Buffer,
    d_lte_in: Buffer,
    d_lte_out: Buffer,
    d_probe_out: Buffer,
    // Tables.
    d_tab_idx: Buffer,
    d_cur_row: Buffer,
    d_diag: Buffer,
    d_probes: Buffer,
    // Device words.
    /// Lanes whose Newton ended this round (`settle`): the probe's mask.
    d_decide: Buffer,
    d_count: Buffer,
    d_fails: Buffer,
    /// The lowest lane whose replay failed this tick and the dense peel took
    /// (all ones for none).
    d_first: Buffer,
    /// Failed lanes past the dense pool this tick, left to the host.
    d_over: Buffer,
    d_slot_next: Buffer,
    // The device dense peel (`dense_slots` > 0): the pattern's CSC and
    // `dense_slots` scratch slots of n * n + n entries.
    d_col_ptr: Buffer,
    d_row_idx: Buffer,
    d_dense: Buffer,
    // The host peel's packed lanes: their indices, then A, rhs and dx.
    d_peel_idx: Buffer,
    d_peel: Buffer,
    // Device step control (`initSteps`): each lane's `LaneStep`, its
    // breakpoint request and answer, its report, and the host's stops.
    d_step: Buffer,
    d_brk_t: Buffer,
    d_best: Buffer,
    d_report: Buffer,
    d_stop: Buffer,
    // A launch's ring of `max_rounds` rounds: reports, probe rows, the slot
    // word, and the dense peel's launch-wide count and lowest failing lane.
    d_ring_report: Buffer,
    d_ring_rows: Buffer,
    d_ring_slot: Buffer,
    d_seen: Buffer,
    d_seen_lo: Buffer,
    // Device `.meas` passes (`initMeasures`): each pass's columns, every
    // (pass, lane)'s state lane-major, and each lane's resolving ring slot.
    d_meas_cols: Buffer,
    d_meas: Buffer,
    d_meas_resolved: Buffer,

    // Host mirrors of what moves each round, page-locked (`pinned`) so the
    // copies queue on `stream` instead of staging through a fenced copy.
    sc: []lk.LaneScalars,
    sims: []SimState,
    lte_in: []lk.LteIn,
    lte_out: [][2]f64,
    gate: []lk.Gate,
    query: []u32,
    ops: []u8,
    accept_: []u8,
    round: []u8,
    perm_q: []u8,
    perm_qt: []u8,
    slot0: []u8,
    fail: []u8,
    probe_out: []f64,
    /// Device step control: the host's stop requests for the next launch,
    /// and the last launch's ring of reports and probe rows.
    stop: []u8,
    ring_report: []StepReport,
    ring_rows: []f64,
    /// Each lane's resolving ring slot of the last launch (0xff: none yet).
    meas_resolved: []u8,
    /// The device passes as the last `fetchMeasures` found them, lane-major.
    meas_host: []passes.Dev = &.{},
    /// Device `.meas` passes per lane (0: the host measures the rows).
    n_meas: u32 = 0,
    meas_t_start: f64 = 0,
    /// The per-tick words, pinned so their downloads stay asynchronous
    /// instead of staging through pageable memory: fails, count, first,
    /// over.
    words: []u32,
    probes: []const u32,
    /// Lanes that left the device for a CPU rerun: `deinit` copies no
    /// state back into them.
    detached: []bool,
    /// Per lane, an n-long host vector whose probe rows hold the last
    /// probe's x (`Ops.accepted`).
    rows: []f64,
    prof: Prof = .{},
    /// Two immutable tick graphs, separated at the exceptional host LU peel.
    /// Their pointers and dimensions stay valid for this batch's lifetime;
    /// all per-round values live in device buffers. Only Newton tolerances
    /// are captured scalars, so changing them rebuilds the graphs.
    graph_a: ?GraphExec = null,
    graph_b: ?GraphExec = null,
    graph_opts: ?lk.NewtonOpts = null,
    /// One device-controlled round with unrolled ticks (`roundKernels`).
    graph_round: ?GraphExec = null,
    /// Entries `d_lx`/`d_ux` hold per lane; a `retape` past them regrows.
    lx_cap: u32 = 0,
    ux_cap: u32 = 0,
    /// Lanes the device dense peel can take at once; 0 when n is past
    /// `lk.dense_peel_max_n`. With one slot per lane no lane ever needs
    /// the host, so a tick reads its words once, after graph B.
    dense_slots: u32 = 0,
    /// `initSteps` ran: rounds go through `deviceLaunch`.
    device_step: bool = false,
    /// Lanes `d_peel_idx`/`d_peel` hold.
    peel_cap: u32 = 0,
    /// Staging cells past the slots and rows of the planes, for the hub
    /// touches of coloured batches (`lane_color.zig`).
    stage_slots: u32 = 0,
    stage_rows: u32 = 0,

    const Self = @This();

    /// `init`'s pivot tape from `c0`'s first step matrix (`dt` as `Sim.init`
    /// left it), factored on `c0`'s host solver.
    fn ownTape(gpa: std.mem.Allocator, c0: *Circuit, dt: f64, n_hist: usize) !devices.Tape {
        const nnz: usize = c0.nnz;
        const lead = &c0.ws.?.slv;
        {
            const a0 = try gpa.alloc(f64, nnz);
            defer gpa.free(a0);
            if (c0.has_charge and n_hist != 0) c0.combineGC(1 / dt, a0) else @memcpy(a0, c0.g_vals[0..nnz]);
            lead.factor(a0, c0.solver_execution) catch return Error.NoLaneTape;
        }
        const lu0 = lead.sparse() orelse return Error.NoLaneTape;
        if (!lu0.factored) return Error.NoLaneTape;
        const built = try lk.buildTab(gpa, lu0, lead.col_ptr, lead.params.refactor_growth_limit, direct.pivot_tol);
        return .{ .idx = built.idx, .tab = built.tab };
    }

    /// Uploads every lane after its `Sim.init`: the pattern's tapes, each
    /// batch's packed rows, x_op, the charge rings and histories. `sims[l]`
    /// is lane l's step machine (`tran.Sim`), read for its rings. Fails with
    /// `Error` or a driver or allocation error; the caller then runs the
    /// lanes on the CPU.
    pub fn init(gpa: std.mem.Allocator, ckts: []const *Circuit, sims: anytype, probes: []const u32) !*Self {
        if (comptime backend == null) return Error.NoLaneImage;
        if (ckts.len == 0 or ckts.len > std.math.maxInt(u32)) return Error.IncompatibleLanes;
        const c0 = ckts[0];
        const n_lanes: u32 = @intCast(ckts.len);
        const N: usize = n_lanes;
        const n: usize = c0.n;
        const nnz: usize = c0.nnz;
        const s0 = sims[0];
        const n_hist: usize = s0.n_hist;
        const n_qt: usize = s0.n_qt;
        const n_probes: usize = probes.len;

        for (ckts) |c| {
            if (!samePattern(c, c0)) return Error.IncompatibleLanes;
            for (c.batches, c0.batches) |b, b0| {
                const payload = b.hooks.gpu_payload orelse return Error.NoLaneImage;
                const payload0 = b0.hooks.gpu_payload orelse return Error.NoLaneImage;
                if (!sameRows(payload(b.ctx), payload0(b0.ctx))) return Error.IncompatibleLanes;
            }
        }

        // The LU tape: lane 0's pivot order on its first step's matrix, G +
        // C/dt at x_op as `Sim.init` left the planes (G alone without charge).
        // A slice of a batch split across devices (`devices.tape`) replays
        // the tape of the batch's lane 0 instead of its own lane 0's.
        const shared = if (devices.tape) |ts| (if (devices.tape_builds) null else try ts.take(gpa)) else null;
        const built: devices.Tape = shared orelse ownTape(gpa, c0, s0.dt, n_hist) catch |e| {
            if (devices.tape) |ts| ts.fail();
            return e;
        };
        defer gpa.free(built.idx);
        if (shared == null) if (devices.tape) |ts| ts.publish(built.tab, built.idx);
        const t = built.tab;

        const self = try gpa.create(Self);
        errdefer gpa.destroy(self);
        var k = try Kernels.init();
        errdefer k.deinit();
        const k0 = &k.refactor;
        var stream = try k0.createStream();
        errdefer stream.deinit();

        // Per batch: each lane's payload, packed, uploaded.
        const n_b = c0.batches.len;
        const batches = try gpa.alloc(LaneBatch, n_b);
        errdefer gpa.free(batches);
        var n_up: usize = 0;
        errdefer for (batches[0..n_up]) |*b| b.deinit();
        const ps = try gpa.alloc(device_ir.GpuPayload, N);
        defer gpa.free(ps);
        var tape_off: u32 = 0;
        for (0..n_b) |bi| {
            const b0 = c0.batches[bi];
            for (ps, ckts) |*p, c| p.* = c.batches[bi].hooks.gpu_payload.?(c.batches[bi].ctx);
            const p0 = ps[0];
            var name: [160]u8 = undefined;
            const eval_name = laneName(&name, p0.kernel, "eval") orelse return Error.NoLaneImage;
            var eval = hdl.kernel(backend.?, @ptrCast(b0.hooks.gpu_payload.?), eval_name) catch |e| return if (e == error.KernelNotFound) Error.NoLaneImage else e;
            errdefer eval.deinit();
            var lim: ?Raw = null;
            errdefer if (lim) |*x| x.deinit();
            var ctl: ?Raw = null;
            errdefer if (ctl) |*x| x.deinit();
            var chg: ?Raw = null;
            errdefer if (chg) |*x| x.deinit();
            if (p0.lim_kernel.len > 0) lim = try hdl.kernel(backend.?, @ptrCast(b0.hooks.gpu_payload.?), laneName(&name, p0.kernel, "lim").?);
            if (p0.ctl_kernel.len > 0) ctl = try hdl.kernel(backend.?, @ptrCast(b0.hooks.gpu_payload.?), laneName(&name, p0.kernel, "ctl").?);
            const tape_len: u32 = if (b0.hooks.q_tape) |f| @intCast(f(b0.ctx).len) else 0;
            if (b0.has_charge) chg = try hdl.kernel(backend.?, @ptrCast(b0.hooks.gpu_payload.?), laneName(&name, p0.kernel, "chg").?);
            const pk = try pack(gpa, ps);
            defer pk.deinit(gpa);
            // Uploaded one at a time so a failure frees the ones already made.
            const srcs = [_][]const u8{ pk.models, std.mem.sliceAsBytes(pk.model_of), pk.instances, pk.states, std.mem.sliceAsBytes(pk.lim), std.mem.sliceAsBytes(p0.gath), std.mem.sliceAsBytes(p0.rhs_idx), std.mem.sliceAsBytes(p0.slots) };
            var d: [srcs.len + 1]Buffer = undefined;
            var n_d: usize = 0;
            errdefer for (d[0..n_d]) |*x| x.free();
            for (srcs) |s| {
                d[n_d] = try upload(k0, s);
                n_d += 1;
            }
            d[n_d] = try zeros(k0, N * 4);
            n_d += 1;
            batches[bi] = .{
                .eval = eval,
                .lim = lim,
                .ctl = ctl,
                .chg = chg,
                .count = p0.count,
                .has_lim = p0.lim_x.len > 0,
                .tape_off = tape_off,
                .inst_size = if (p0.count == 0) 0 else p0.instances.len / p0.count,
                .state_size = if (p0.count == 0) 0 else p0.states.len / p0.count,
                .d_models = d[0],
                .d_model_of = d[1],
                .d_instances = d[2],
                .d_states = d[3],
                .d_lim = d[4],
                .d_gath = d[5],
                .d_rhs_idx = d[6],
                .d_slots = d[7],
                .d_limw = d[8],
            };
            tape_off += tape_len;
            n_up += 1;
        }
        std.debug.assert(tape_off == n_qt or n_qt == 0);
        // Colours, each batch's staging cells after the last batch's.
        var stage_slots: u32 = 0;
        var stage_rows: u32 = 0;
        if (colorPolicy(N)) for (batches, c0.batches) |*b, hb| {
            if (b.count == 0) continue;
            const p0 = hb.hooks.gpu_payload.?(hb.ctx);
            const ks: u32 = @intCast(p0.slots.len / b.count);
            const col = try lane_color.color(gpa, b.count, ks, p0.slots, p0.n_u, p0.rhs_idx, @intCast(nnz), @intCast(n), @intCast(nnz + 1 + stage_slots), @intCast(n + 1 + stage_rows));
            defer col.deinit(gpa);
            // Fewer than two instances per colour launch more than they gain.
            if (2 * col.colors() > b.count and !(@import("builtin").is_test and test_access.force_color)) continue;
            stage_slots += col.n_slot_stage;
            stage_rows += col.n_row_stage;
            const hubs = struct {
                fn up(raw: *Raw, h: lane_color.Coloring.Hubs) !LaneBatch.Color.Hubs {
                    return .{ .n = @intCast(h.cells.len), .d_cells = try upload(raw, std.mem.sliceAsBytes(h.cells)), .d_starts = try upload(raw, std.mem.sliceAsBytes(h.starts)), .d_entries = try upload(raw, std.mem.sliceAsBytes(h.entries)) };
                }
            };
            b.color = .{
                .starts = try std.heap.page_allocator.dupe(u32, col.starts),
                .d_order = try upload(k0, std.mem.sliceAsBytes(col.order)),
                .d_slots = try upload(k0, std.mem.sliceAsBytes(col.slots)),
                .d_rows = try upload(k0, std.mem.sliceAsBytes(col.rows)),
                .slot_hubs = try hubs.up(k0, col.slot_hubs),
                .row_hubs = try hubs.up(k0, col.row_hubs),
            };
            if (c0.knobs.gpu_stats != .off) std.debug.print("gpu-lanes: {s}: {d} instances in {d} colours, {d} + {d} hub cells\n", .{ hb.type_name, b.count, col.colors(), col.slot_hubs.cells.len, col.row_hubs.cells.len });
        };

        self.* = .{
            .gpa = gpa,
            .ckts = ckts,
            .n_lanes = n_lanes,
            .n = @intCast(n),
            .nnz = @intCast(nnz),
            .n_hist = @intCast(n_hist),
            .n_qt = @intCast(n_qt),
            .n_probes = @intCast(n_probes),
            // A static solve (`initStatic`, no history) reads no charge.
            .has_charge = c0.has_charge and n_hist != 0,
            .state_q = c0.has_charge and n_hist != 0 and c0.has_state_q,
            .d0 = c0.diag_slots[0],
            .batches = batches,
            .k = k,
            .stream = stream,
            .tab = t,
            .d_x = undefined,
            .d_x_old = undefined,
            .d_dx = undefined,
            .d_cur = undefined,
            .d_prev = undefined,
            .d_g = undefined,
            .d_c = undefined,
            .d_rhs = undefined,
            .d_q = undefined,
            .d_a = undefined,
            .d_lx = undefined,
            .d_ux = undefined,
            .d_udiag = undefined,
            .d_w = undefined,
            .d_y = undefined,
            .d_ring = undefined,
            .d_ip = undefined,
            .d_ring_qt = undefined,
            .d_ip_qt = undefined,
            .d_sc = undefined,
            .d_sims = undefined,
            .d_active = undefined,
            .d_round = undefined,
            .d_stage = undefined,
            .d_gate = undefined,
            .d_flags = undefined,
            .d_fail = undefined,
            .d_ops = undefined,
            .d_accept = undefined,
            .d_slot0 = undefined,
            .d_perm_q = undefined,
            .d_perm_qt = undefined,
            .d_lte_in = undefined,
            .d_lte_out = undefined,
            .d_probe_out = undefined,
            .d_tab_idx = undefined,
            .d_cur_row = undefined,
            .d_diag = undefined,
            .d_probes = undefined,
            .d_decide = undefined,
            .d_count = undefined,
            .d_fails = undefined,
            .d_first = undefined,
            .d_over = undefined,
            .d_slot_next = undefined,
            .d_col_ptr = undefined,
            .d_row_idx = undefined,
            .d_dense = undefined,
            .d_peel_idx = undefined,
            .d_peel = undefined,
            .d_step = undefined,
            .d_brk_t = undefined,
            .d_best = undefined,
            .d_report = undefined,
            .d_stop = undefined,
            .stop = undefined,
            .ring_report = undefined,
            .ring_rows = undefined,
            .d_ring_report = undefined,
            .d_ring_rows = undefined,
            .d_ring_slot = undefined,
            .d_seen = undefined,
            .d_seen_lo = undefined,
            .d_meas_cols = undefined,
            .d_meas = undefined,
            .d_meas_resolved = undefined,
            .meas_resolved = undefined,
            .sc = undefined,
            .sims = undefined,
            .lte_in = undefined,
            .lte_out = undefined,
            .gate = undefined,
            .query = undefined,
            .ops = undefined,
            .accept_ = undefined,
            .round = undefined,
            .perm_q = undefined,
            .perm_qt = undefined,
            .slot0 = undefined,
            .fail = undefined,
            .probe_out = undefined,
            .words = undefined,
            .probes = probes,
            .detached = undefined,
            .rows = undefined,
            .prof = .{ .phases = c0.knobs.gpu_stats == .phases },
        };
        // ponytail: every buffer below is freed by `deinit` only; a failure
        // part way through leaks the device buffers made so far until the
        // context goes down. Thread errdefers per buffer if that ever matters.
        const f8 = @sizeOf(f64);
        const rows_b = (n + 1) * N * f8;
        self.d_x = try zeros(k0, rows_b);
        self.d_x_old = try zeros(k0, rows_b);
        self.d_dx = try zeros(k0, rows_b);
        self.d_cur = try zeros(k0, rows_b);
        self.d_prev = try zeros(k0, rows_b);
        // Staging tails start at -0.0 (`lane_color.zig`); every pass that
        // clears the planes clears only the cells before them.
        self.stage_slots = stage_slots;
        self.stage_rows = stage_rows;
        self.d_g = try staged(k0, nnz + 1, stage_slots, N);
        self.d_c = try staged(k0, nnz + 1, stage_slots, N);
        self.d_rhs = try staged(k0, n + 1, stage_rows, N);
        self.d_q = try staged(k0, n + 1, stage_rows, N);
        self.d_a = try zeros(k0, (nnz + 1) * N * f8);
        self.lx_cap = @max(t.nnz_l, 1);
        self.ux_cap = @max(t.nnz_u, 1);
        self.d_lx = try zeros(k0, self.lx_cap * N * f8);
        self.d_ux = try zeros(k0, self.ux_cap * N * f8);
        self.d_udiag = try zeros(k0, n * N * f8);
        self.d_w = try zeros(k0, n * N * f8);
        self.d_y = try zeros(k0, n * N * f8);
        self.d_ring = try zeros(k0, n_hist * n * N * f8);
        self.d_ip = try zeros(k0, n * N * f8);
        self.d_ring_qt = try zeros(k0, N * n_hist * n_qt * f8);
        self.d_ip_qt = try zeros(k0, N * n_qt * f8);
        self.d_sc = try zeros(k0, N * @sizeOf(lk.LaneScalars));
        self.d_sims = try zeros(k0, N * @sizeOf(SimState));
        self.d_active = try zeros(k0, N);
        self.d_round = try zeros(k0, N);
        self.d_stage = try zeros(k0, N);
        self.d_gate = try zeros(k0, N * @sizeOf(lk.Gate));
        self.d_flags = try zeros(k0, N * 4);
        self.d_fail = try zeros(k0, N);
        self.d_ops = try zeros(k0, N);
        self.d_accept = try zeros(k0, N);
        self.d_slot0 = try zeros(k0, N);
        self.d_perm_q = try zeros(k0, N * n_hist);
        self.d_perm_qt = try zeros(k0, N * n_hist);
        self.d_lte_in = try zeros(k0, N * @sizeOf(lk.LteIn));
        self.d_lte_out = try zeros(k0, N * 16);
        self.d_probe_out = try zeros(k0, N * n_probes * f8);
        self.d_tab_idx = try upload(k0, std.mem.sliceAsBytes(built.idx));
        self.d_decide = try zeros(k0, N);
        self.d_count = try zeros(k0, 4);
        self.d_fails = try zeros(k0, 4);
        self.d_first = try zeros(k0, 4);
        self.d_over = try zeros(k0, 4);
        self.d_slot_next = try zeros(k0, 4);
        self.d_col_ptr = try upload(k0, std.mem.sliceAsBytes(c0.col_ptr[0 .. n + 1]));
        self.d_row_idx = try upload(k0, std.mem.sliceAsBytes(c0.row_idx[0..nnz]));
        // ponytail: a fixed 256 MB pool for the dense peel; size it from
        // the device's free memory if large lane counts ever crowd it.
        const slot_bytes = (n * n + n) * f8;
        self.dense_slots = if (n <= lk.dense_peel_max_n) @intCast(@min(N, (256 << 20) / slot_bytes)) else 0;
        self.d_dense = try k0.alloc(@max(@as(usize, self.dense_slots) * slot_bytes, 1));
        self.d_peel_idx = try k0.alloc(4);
        self.d_peel = try k0.alloc(8);
        self.d_step = try k0.alloc(N * @sizeOf(lk.LaneStep));
        self.d_brk_t = try k0.alloc(N * 16);
        self.d_best = try k0.alloc(N * f8);
        self.d_report = try k0.alloc(N * @sizeOf(StepReport));
        self.d_stop = try zeros(k0, N);
        self.d_ring_report = try k0.alloc(max_rounds * N * @sizeOf(StepReport));
        self.d_ring_rows = try k0.alloc(@max(max_rounds * N * n_probes, 1) * f8);
        self.d_ring_slot = try zeros(k0, 4);
        self.d_seen = try zeros(k0, 4);
        self.d_seen_lo = try zeros(k0, 4);
        self.d_meas_cols = try zeros(k0, 1);
        self.d_meas = try zeros(k0, 1);
        self.d_meas_resolved = try zeros(k0, 1);
        const cur_row = try gpa.alloc(u8, n);
        defer gpa.free(cur_row);
        for (cur_row, c0.current_row[0..n]) |*o, b| o.* = @intFromBool(b);
        self.d_cur_row = try upload(k0, cur_row);
        self.d_diag = try upload(k0, std.mem.sliceAsBytes(c0.diag_slots[0..n]));
        self.d_probes = try upload(k0, std.mem.sliceAsBytes(probes));

        self.sc = try pinned(k0, lk.LaneScalars, N);
        self.sims = try pinned(k0, SimState, N);
        self.lte_in = try pinned(k0, lk.LteIn, N);
        self.lte_out = try pinned(k0, [2]f64, N);
        self.gate = try pinned(k0, lk.Gate, N);
        self.query = try pinned(k0, u32, N);
        self.ops = try pinned(k0, u8, N);
        self.accept_ = try pinned(k0, u8, N);
        self.round = try pinned(k0, u8, N);
        self.perm_q = try pinned(k0, u8, N * n_hist);
        self.perm_qt = try pinned(k0, u8, N * n_hist);
        self.slot0 = try pinned(k0, u8, N);
        self.fail = try pinned(k0, u8, N);
        self.probe_out = try pinned(k0, f64, @max(N * n_probes, 1));
        self.stop = try pinned(k0, u8, N);
        @memset(self.stop, 0);
        self.ring_report = try pinned(k0, StepReport, max_rounds * N);
        self.ring_rows = try pinned(k0, f64, @max(max_rounds * N * n_probes, 1));
        self.meas_resolved = try pinned(k0, u8, N);
        @memset(self.meas_resolved, 0xff);
        self.words = try pinned(k0, u32, 4);
        self.rows = try gpa.alloc(f64, N * n);
        self.detached = try gpa.alloc(bool, N);
        @memset(self.detached, false);
        @memset(self.sc, std.mem.zeroes(lk.LaneScalars));
        @memset(self.ops, 0);
        @memset(self.accept_, 0);
        @memset(self.round, 0);
        @memset(self.rows, 0);
        for (0..N) |l| for (0..n_hist) |s| {
            self.perm_q[l * n_hist + s] = @intCast(s);
            self.perm_qt[l * n_hist + s] = @intCast(s);
        };

        // Each lane's start, as its `Sim.init` left it: x_op in cur and
        // prev, the row ring (physical = logical), i_prev, the tape ring.
        const host = try gpa.alloc(f64, @max(n_hist * n * N, @max((n + 1) * N, N * n_hist * n_qt)));
        defer gpa.free(host);
        @memset(host[0 .. (n + 1) * N], 0);
        for (sims, 0..) |s, l| for (s.cur[0..n], 0..) |v, i| {
            host[i * N + l] = v;
        };
        try self.d_cur.upload(host.ptr, (n + 1) * N * f8);
        try self.d_prev.upload(host.ptr, (n + 1) * N * f8);
        if (self.has_charge) {
            for (sims, 0..) |s, l| for (0..n_hist) |slot| for (s.q_ring[slot][0..n], 0..) |v, i| {
                host[(slot * n + i) * N + l] = v;
            };
            try self.d_ring.upload(host.ptr, n_hist * n * N * f8);
            for (sims, 0..) |s, l| for (s.i_prev[0..n], 0..) |v, i| {
                host[i * N + l] = v;
            };
            try self.d_ip.upload(host.ptr, n * N * f8);
        }
        if (n_qt > 0) {
            for (sims, 0..) |s, l| for (0..n_hist) |slot| for (s.qt_ring[slot][0..n_qt], 0..) |v, j| {
                host[(slot * n_qt + j) * N + l] = v;
            };
            try self.d_ring_qt.upload(host.ptr, N * n_hist * n_qt * f8);
            for (sims, 0..) |s, l| for (s.qt_i_prev[0..n_qt], 0..) |v, j| {
                host[j * N + l] = v;
            };
            try self.d_ip_qt.upload(host.ptr, N * n_qt * f8);
        }
        try self.d_perm_q.upload(self.perm_q.ptr, N * n_hist);
        try self.d_perm_qt.upload(self.perm_qt.ptr, N * n_hist);
        if (@import("builtin").is_test) _ = test_access.admitted.fetchAdd(N, .monotonic);
        return self;
    }

    /// Copies each lane's device Instance rows back into its host batch
    /// (the latches `refuseLatched` reads), then frees everything.
    pub fn deinit(self: *Self) void {
        if (comptime backend == null) return;
        const gpa = self.gpa;
        const N: usize = self.n_lanes;
        self.stream.synchronize() catch {};
        if (self.prof.on) {
            const p = self.prof;
            const ms = struct {
                fn f(ns: u64) f64 {
                    return @as(f64, @floatFromInt(ns)) / 1e6;
                }
            }.f;
            std.debug.print("gpu-lanes: {d} retapes, {d} dense peels on the device ({d} slots)\n", .{ p.retapes, p.dense_peels, self.dense_slots });
            std.debug.print("gpu-lanes: {d} lanes, {d} rounds, {d} Newton ticks, {d} peels; ms host={d:.1} predict={d:.1} newton={d:.1} probe={d:.1} apply={d:.1}\n", .{ N, p.rounds, p.ticks, p.peels, ms(p.ns[0]), ms(p.ns[1]), ms(p.ns[2]), ms(p.ns[3]), ms(p.ns[4]) });
            std.debug.print("gpu-lanes: {d} ticks replayed with graphs; drain {d:.1} ms\n", .{ p.graph_ticks, ms(p.drain_ns) });
            const round_capacity = p.rounds * N;
            const tick_capacity = p.ticks * N;
            const round_idle = if (round_capacity == 0) 0 else 1 - @as(f64, @floatFromInt(p.lane_rounds)) / @as(f64, @floatFromInt(round_capacity));
            const tick_idle = if (tick_capacity == 0) 0 else 1 - @as(f64, @floatFromInt(p.lane_ticks)) / @as(f64, @floatFromInt(tick_capacity));
            std.debug.print("gpu-lanes: idle round={d:.3}% Newton={d:.3}%\n", .{ 100 * round_idle, 100 * tick_idle });
            if (p.phases) std.debug.print("gpu-lanes: direct tick ms eval={d:.1} assemble={d:.1} lu={d:.1} gates={d:.1} state={d:.1}\n", .{ ms(p.tick_ns[0]), ms(p.tick_ns[1]), ms(p.tick_ns[2]), ms(p.tick_ns[3]), ms(p.tick_ns[4]) });
        }
        self.clearGraphs();
        self.copyBack();
        for (self.batches) |*b| b.deinit();
        gpa.free(self.batches);
        inline for (@typeInfo(Self).@"struct".field_names) |f| {
            if (comptime std.mem.startsWith(u8, f, "d_")) @field(self, f).free();
        }
        inline for (pinned_mirrors) |f| self.k.refactor.freePinned(std.mem.sliceAsBytes(@field(self, f)));
        gpa.free(self.rows);
        gpa.free(self.detached);
        gpa.free(self.meas_host);
        self.stream.deinit();
        self.k.deinit();
        gpa.destroy(self);
    }

    /// Copies each attached lane's device Instance and State rows back into
    /// its host batch. Best effort: a failed allocation or download leaves
    /// that batch's host rows as they were. Called with the stream idle.
    pub fn copyBack(self: *Self) void {
        if (comptime !available) unreachable; // no batch is ever built without GPU images
        const gpa = self.gpa;
        const N: usize = self.n_lanes;
        for (self.batches, 0..) |*b, bi| {
            const is = b.inst_size;
            if (b.count == 0 or is == 0) continue;
            const buf = gpa.alloc(u8, b.count * N * is) catch continue;
            defer gpa.free(buf);
            b.d_instances.download(buf.ptr, buf.len) catch continue;
            for (self.ckts, 0..) |c, l| {
                if (self.detached[l]) continue;
                const hb = c.batches[bi];
                const dst = @constCast(hb.hooks.gpu_payload.?(hb.ctx).instances);
                for (0..b.count) |id| @memcpy(dst[id * is ..][0..is], buf[(id * N + l) * is ..][0..is]);
            }
            const ss = b.state_size;
            if (ss != 0) {
                const states = gpa.alloc(u8, b.count * N * ss) catch continue;
                defer gpa.free(states);
                b.d_states.download(states.ptr, states.len) catch continue;
                for (self.ckts, 0..) |c, l| {
                    if (self.detached[l]) continue;
                    const hb = c.batches[bi];
                    const dst = @constCast(hb.hooks.gpu_payload.?(hb.ctx).states);
                    for (0..b.count) |id| @memcpy(dst[id * ss ..][0..ss], states[(id * N + l) * ss ..][0..ss]);
                }
            }
        }
    }

    /// `init` for operating points and `.dc` points (06 6.9): no step
    /// machine and no charge history; lane l starts at `xs[l]`. Evaluates
    /// lane 0 at its start on the host for the LU tape, so each lane's
    /// `SimState` must be set first. Rounds go through `solveStatic`.
    pub fn initStatic(gpa: std.mem.Allocator, ckts: []const *Circuit, xs: []const []const f64) !*Self {
        // A build without GPU images has no device buffers to fill.
        if (comptime !available) return error.BackendUnavailable;
        const Seed = struct {
            n_hist: usize = 0,
            n_qt: usize = 0,
            dt: f64 = std.math.inf(f64),
            cur: []const f64,
            q_ring: []const []const f64 = &.{},
            i_prev: []const f64 = &.{},
            qt_ring: []const []const f64 = &.{},
            qt_i_prev: []const f64 = &.{},
        };
        const seeds = try gpa.alloc(Seed, ckts.len);
        defer gpa.free(seeds);
        for (seeds, xs) |*s, x| s.* = .{ .cur = x };
        // `init` factors lane 0 in its workspace, which an operating point
        // makes only on its first solve.
        if (ckts.len > 0) {
            _ = try ckts[0].workspace();
            ckts[0].evalNewton(xs[0], 0);
        }
        const self = try init(gpa, ckts, seeds, &.{});
        errdefer self.deinit();
        try self.putX(xs);
        @memset(self.round, 1);
        return self;
    }

    /// Each lane's start `xs[l]` into `d_x`.
    fn putX(self: *Self, xs: []const []const f64) !void {
        const N: usize = self.n_lanes;
        const n: usize = self.n;
        const host = try self.gpa.alloc(f64, (n + 1) * N);
        defer self.gpa.free(host);
        @memset(host, 0);
        for (xs, 0..) |x, l| for (x[0..n], 0..) |v, i| {
            host[i * N + l] = v;
        };
        try self.d_x.upload(host.ptr, host.len * @sizeOf(f64));
    }

    /// The next `.dc` point of a static batch: every attached lane's host
    /// rows (Model, Instance, State and the lim plane, which the host's
    /// `recompute` and predictor limit pass just wrote) and its start
    /// `xs[l]`. With `limits_live` the lim planes are live for the first
    /// iterate, as after `Circuit.applyLimits`. Clears `detached`.
    pub fn reload(self: *Self, xs: []const []const f64, limits_live: bool) !void {
        if (comptime !available) return error.BackendUnavailable;
        const gpa = self.gpa;
        const k0 = &self.k.refactor;
        const ps = try gpa.alloc(device_ir.GpuPayload, self.n_lanes);
        defer gpa.free(ps);
        try self.stream.synchronize();
        for (self.batches, 0..) |*b, bi| {
            if (b.count == 0) continue;
            for (ps, self.ckts) |*p, c| p.* = c.batches[bi].hooks.gpu_payload.?(c.batches[bi].ctx);
            const pk = try pack(gpa, ps);
            defer pk.deinit(gpa);
            // A grown Model table moves, and the captured graphs hold its
            // old address.
            if (pk.models.len > b.d_models.bytes) {
                const m = try upload(k0, pk.models);
                b.d_models.free();
                b.d_models = m;
                self.clearGraphs();
                self.graph_opts = null;
            } else if (pk.models.len > 0) try b.d_models.upload(pk.models.ptr, pk.models.len);
            try b.d_model_of.upload(pk.model_of.ptr, pk.model_of.len * 4);
            if (pk.instances.len > 0) try b.d_instances.upload(pk.instances.ptr, pk.instances.len);
            if (pk.states.len > 0) try b.d_states.upload(pk.states.ptr, pk.states.len);
            if (pk.lim.len > 0) try b.d_lim.upload(pk.lim.ptr, pk.lim.len * @sizeOf(f64));
        }
        try self.putX(xs);
        @memset(self.detached, false);
        @memset(self.round, 1);
        try self.put(&self.d_round, self.round);
        for (self.batches) |*b| {
            if (b.lim == null or !b.has_lim or b.count == 0) continue;
            try self.setLimw(b, &self.d_round, @intFromBool(limits_live));
        }
    }

    /// Newton for the lanes of `round` (every lane after `initStatic` and
    /// `reload`), each from its x in `d_x` at its circuit's `SimState`, as
    /// `converger.newton` solves an operating point or a `.dc` point: no
    /// predictor, and INITFIX first under `opts.init_fix`. Then the converged lanes' latch query: DC
    /// Newton iterates on past a flipped latch, which the device does not,
    /// so a flipped lane, and one whose state pass asked for a reject or a
    /// bound, comes back `reject` for the host to solve. Lane l's x lands
    /// in `xs[l]` and, for a round lane, its outcome in `out[l]`.
    pub fn solveStatic(self: *Self, xs: []const []f64, opts: converger.Options, out: []LaneNewton) !void {
        if (comptime !available) return error.BackendUnavailable;
        const N: usize = self.n_lanes;
        const n: usize = self.n;
        std.debug.assert(xs.len == N and out.len == N and self.n_hist == 0);
        for (self.sims, self.ckts) |*s, c| s.* = c.sim;
        @memset(self.sc, std.mem.zeroes(lk.LaneScalars));
        for (self.sc) |*s| s.flags = 1;
        try self.put(&self.d_sc, self.sc);
        self.prof.rounds += 1;
        try self.newton(opts, out);
        const query = 1 + @as(u8, @backingInt(StateCtlOp.query));
        for (self.ops, self.round, out) |*o, on, r| o.* = if (on != 0 and r.result.converged and !r.singular) query else 0;
        try self.put(&self.d_ops, self.ops);
        try self.d_flags.fillAsync(0, N * 4, &self.stream);
        try self.ctlPass();
        try self.get(self.query, &self.d_flags);
        const host = try self.gpa.alloc(f64, (n + 1) * N);
        defer self.gpa.free(host);
        try self.d_x.downloadAtAsync(host.ptr, 0, host.len * @sizeOf(f64), &self.stream);
        try self.stream.synchronize();
        @memset(self.ops, 0);
        for (out, self.round, self.query) |*o, on, q| if (on != 0 and q & 1 != 0) {
            o.reject = true;
        };
        for (xs, 0..) |x, l| for (x[0..n], 0..) |*v, i| {
            v.* = host[i * N + l];
        };
    }

    const pinned_mirrors = .{ "sc", "sims", "lte_in", "lte_out", "gate", "query", "ops", "accept_", "round", "perm_q", "perm_qt", "slot0", "fail", "probe_out", "words", "stop", "ring_report", "ring_rows", "meas_resolved" };

    /// `n` elements of page-locked host memory, freed with `freePinned`.
    fn pinned(k: *Raw, comptime T: type, n: usize) ![]T {
        const bytes = try k.allocPinned(@max(n, 1) * @sizeOf(T));
        // Driver allocations are page aligned.
        const typed: [*]T = @ptrCast(@alignCast(bytes.ptr));
        return typed[0..n];
    }

    /// `src` into `dst` on the lane stream. `src` must be pinned and left
    /// alone until the stream next synchronizes.
    fn put(self: *Self, dst: *Buffer, src: anytype) !void {
        const bytes = std.mem.sliceAsBytes(src);
        try dst.uploadAtAsync(bytes.ptr, 0, bytes.len, &self.stream);
    }

    /// `src` into pinned `dst` on the lane stream; valid after the next
    /// synchronize.
    fn get(self: *Self, dst: anytype, src: *Buffer) !void {
        const bytes = std.mem.sliceAsBytes(dst);
        try src.downloadAtAsync(bytes.ptr, 0, bytes.len, &self.stream);
    }

    /// `cells` zeroed rows of N lanes followed by `tail` rows of -0.0.
    fn staged(k: *Raw, cells: usize, tail: usize, N: usize) !Buffer {
        if (tail == 0) return zeros(k, cells * N * @sizeOf(f64));
        const host = try std.heap.page_allocator.alloc(f64, (cells + tail) * N);
        defer std.heap.page_allocator.free(host);
        @memset(host[0 .. cells * N], 0);
        @memset(host[cells * N ..], -0.0);
        return upload(k, std.mem.sliceAsBytes(host));
    }

    fn upload(k: *Raw, bytes: []const u8) !Buffer {
        var buf = try k.alloc(@max(bytes.len, 1));
        errdefer buf.free();
        if (bytes.len > 0) try buf.upload(bytes.ptr, bytes.len);
        return buf;
    }

    fn zeros(k: *Raw, bytes: usize) !Buffer {
        var buf = try k.alloc(@max(bytes, 1));
        errdefer buf.free();
        const z = try std.heap.page_allocator.alloc(u8, @max(bytes, 1));
        defer std.heap.page_allocator.free(z);
        @memset(z, 0);
        try buf.upload(z.ptr, z.len);
        return buf;
    }

    const arg = gompute.interface.arg;

    /// Launches `k` over `threads` threads at the lane block width.
    fn launch(self: *Self, k: *Raw, threads: usize, args: []const gompute.interface.Arg) !void {
        if (threads == 0) return;
        try k.launchOn(&self.stream, gompute.Dim3.linear(@intCast(threads), block), .{ .x = block }, 0, args);
    }

    /// The backend `tran.Sim` drives lane l with.
    pub fn laneOps(self: *Self, l: usize) Ops {
        return .{ .g = self, .l = @intCast(l) };
    }

    /// Starts a round: uploads the round lanes' scalars (`Ops.attempt`
    /// filled them), predicts each lane's first iterate and runs its
    /// predictor limit pass against cur. `round` marks the lanes in it.
    pub fn predictRound(self: *Self) !void {
        try self.put(&self.d_sc, self.sc);
        try self.put(&self.d_sims, self.sims);
        try self.put(&self.d_lte_in, self.lte_in);
        try self.put(&self.d_round, self.round);
        try self.predictKernels();
    }

    /// The predictor and its limit pass for the lanes of `d_round`, from
    /// the scalars already in `d_sc`.
    fn predictKernels(self: *Self) !void {
        const N: usize = self.n_lanes;
        var nl: u64 = N;
        var n: u64 = self.n;
        try self.launch(&self.k.predict, self.n * N, &.{ arg(&nl), arg(&n), self.d_sc.argPtr(), self.d_cur.argPtr(), self.d_prev.argPtr(), self.d_x.argPtr(), self.d_x_old.argPtr() });
        // `applyLimits` leaves each limiting batch's lim plane live
        // (`limitPass` sets the round lanes' words).
        try self.limitPass(&self.d_round, &self.d_x, &self.d_x_old, device_ir.lim_pass_limit);
    }

    /// The limit or state half of every batch's lane state kernel over the
    /// lanes of `mask`, at `x` against `x_old`.
    fn limitPass(self: *Self, mask: *Buffer, x: *Buffer, x_old: *Buffer, halves_: u64) !void {
        const N: usize = self.n_lanes;
        for (self.batches) |*b| {
            const k = if (b.lim) |*k| k else continue;
            if (b.count == 0) continue;
            if (halves_ == device_ir.lim_pass_limit and !b.has_lim) continue;
            var nl: u64 = N;
            var count: u64 = b.count;
            var halves = halves_;
            try self.launch(k, b.count * N, &.{
                arg(&nl),            arg(&count),         mask.argPtr(),         x.argPtr(),             x_old.argPtr(),
                b.d_gath.argPtr(),   b.d_models.argPtr(), b.d_model_of.argPtr(), b.d_instances.argPtr(), b.d_lim.argPtr(),
                b.d_states.argPtr(), b.d_limw.argPtr(),   arg(&halves),          self.d_flags.argPtr(),  self.d_sims.argPtr(),
            });
            // The predictor's limit pass makes the lim plane live, and so
            // does a fixed-period solve's first Newton pass, which runs
            // without the predictor. Per lane: a lane whose Newton carries
            // over keeps its word while others begin.
            if (halves_ == device_ir.lim_pass_limit and b.has_lim) try self.setLimw(b, mask, 1);
        }
    }

    /// Zeroes the planes, then every batch's lane eval at `x` for the lanes
    /// of `mask`.
    fn evalAll(self: *Self, mask: *Buffer, x: *Buffer) !void {
        const N: usize = self.n_lanes;
        const f8 = @sizeOf(f64);
        try self.d_g.fillAsync(0, (self.nnz + 1) * N * f8, &self.stream);
        try self.d_c.fillAsync(0, (self.nnz + 1) * N * f8, &self.stream);
        try self.d_rhs.fillAsync(0, (self.n + 1) * N * f8, &self.stream);
        // Freeze q for lanes that have already passed Newton. Shooting's
        // trapezoidal history uses their last assembly, and another lane
        // taking an extra iteration must not erase it.
        var q_lanes: u64 = N;
        var q_rows: u64 = self.n + 1;
        try self.launch(&self.k.clear_q, (self.n + 1) * N, &.{ arg(&q_lanes), arg(&q_rows), mask.argPtr(), self.d_q.argPtr() });
        for (self.batches) |*b| {
            if (b.count == 0) continue;
            var nl: u64 = N;
            var count: u64 = b.count;
            const col = if (b.color) |*c| c else {
                var none: u64 = 0;
                try self.launch(&b.eval, N, &.{
                    arg(&nl),              arg(&count),            self.d_sims.argPtr(), mask.argPtr(),      b.d_limw.argPtr(),
                    x.argPtr(),            b.d_gath.argPtr(),      b.d_rhs_idx.argPtr(), b.d_slots.argPtr(), b.d_models.argPtr(),
                    b.d_model_of.argPtr(), b.d_instances.argPtr(), self.d_g.argPtr(),    self.d_c.argPtr(),  self.d_rhs.argPtr(),
                    self.d_q.argPtr(),     b.d_lim.argPtr(),       b.d_slots.argPtr(),   arg(&none),         arg(&none),
                });
                continue;
            };
            // One launch per colour, in order, then the hubs.
            for (col.starts[0 .. col.starts.len - 1], col.starts[1..]) |lo, hi| {
                var off: u64 = lo;
                var n_ids: u64 = hi - lo;
                try self.launch(&b.eval, (hi - lo) * N, &.{
                    arg(&nl),              arg(&count),            self.d_sims.argPtr(), mask.argPtr(),        b.d_limw.argPtr(),
                    x.argPtr(),            b.d_gath.argPtr(),      col.d_rows.argPtr(),  col.d_slots.argPtr(), b.d_models.argPtr(),
                    b.d_model_of.argPtr(), b.d_instances.argPtr(), self.d_g.argPtr(),    self.d_c.argPtr(),    self.d_rhs.argPtr(),
                    self.d_q.argPtr(),     b.d_lim.argPtr(),       col.d_order.argPtr(), arg(&off),            arg(&n_ids),
                });
            }
            inline for (.{ .{ "slot_hubs", "d_g", "d_c" }, .{ "row_hubs", "d_rhs", "d_q" } }) |h| {
                const hubs = &@field(col, h[0]);
                var n_hubs: u64 = hubs.n;
                try self.launch(&self.k.hub_reduce, hubs.n * N, &.{
                    arg(&nl),                arg(&n_hubs),                mask.argPtr(),               hubs.d_cells.argPtr(), hubs.d_starts.argPtr(),
                    hubs.d_entries.argPtr(), @field(self, h[1]).argPtr(), @field(self, h[2]).argPtr(),
                });
            }
        }
    }

    /// Batch b's lim-active word for the lanes of `mask`.
    fn setLimw(self: *Self, b: *LaneBatch, mask: *Buffer, value: u64) !void {
        var nl: u64 = self.n_lanes;
        var v = value;
        try self.launch(&self.k.set_limw, self.n_lanes, &.{ arg(&nl), mask.argPtr(), b.d_limw.argPtr(), arg(&v) });
    }

    fn clearGraphs(self: *Self) void {
        inline for (.{ "graph_a", "graph_b", "graph_round" }) |f| if (@field(self, f)) |*g| {
            g.deinit();
            @field(self, f) = null;
        };
    }

    /// `body(self, args...)` captured as a graph. Capture on a temporary
    /// nonblocking stream, then replay on the normal blocking stream.
    /// Keeping that execution stream preserves ordering with the existing
    /// synchronous uploads/downloads on the NULL stream.
    fn capture(self: *Self, comptime body: anytype, args: anytype) !GraphExec {
        var capture_stream = try self.k.refactor.createStreamNonBlocking();
        defer capture_stream.deinit();
        std.mem.swap(Stream, &self.stream, &capture_stream);
        defer std.mem.swap(Stream, &self.stream, &capture_stream);
        try self.stream.beginCapture(.thread_local);
        var capturing = true;
        defer if (capturing) {
            // Even an invalidated capture must be ended before destroying
            // its stream. No captured work has executed.
            if (self.stream.endCapture()) |abandoned| {
                var graph = abandoned;
                graph.deinit();
            } else |_| {}
        };
        try @call(.auto, body, .{self} ++ args);
        const ended = self.stream.endCapture();
        capturing = false;
        var graph = try ended;
        defer graph.deinit();
        return graph.instantiate();
    }

    fn createGraphs(self: *Self, opts: lk.NewtonOpts) !void {
        try self.stream.synchronize();
        var a = try self.capture(linearTick, .{opts});
        errdefer a.deinit();
        const b = try self.capture(updateTick, .{opts});
        self.graph_a = a;
        self.graph_b = b;
    }

    fn prepareGraphs(self: *Self, opts: lk.NewtonOpts) void {
        if (self.graph_opts) |saved| if (std.meta.eql(saved, opts)) return;
        self.clearGraphs();
        self.graph_opts = opts;
        if (self.prof.phases or self.ckts[0].knobs.gpu_nographs or !self.k.refactor.hasGraphs()) return;
        self.createGraphs(opts) catch |e| {
            std.log.warn("GPU lane graph capture failed ({s}); using direct launches", .{@errorName(e)});
        };
    }

    fn tickLap(self: *Self, last: *u64, phase: Prof.TickPhase) !void {
        if (!self.prof.phases) return;
        try self.stream.synchronize();
        const now = Prof.now();
        self.prof.tick_ns[@backingInt(phase)] += now - last.*;
        last.* = now;
    }

    /// Graph A: everything before the host handles a rare failed refactor.
    fn linearTick(self: *Self, opts: lk.NewtonOpts) !void {
        const N: usize = self.n_lanes;
        var nl: u64 = N;
        var nn: u64 = self.n;
        var nnz: u64 = self.nnz;
        var d0: u64 = self.d0;
        var charge: u64 = @intFromBool(self.has_charge);
        var nh: u64 = self.n_hist;
        var tab = self.tab;
        var mark = Prof.now();
        try self.evalAll(&self.d_active, &self.d_x);
        try self.tickLap(&mark, .eval);
        try self.launch(&self.k.assemble, (self.n + self.nnz) * N, &.{
            arg(&nl),             arg(&nn),               arg(&nnz),         arg(&d0),            arg(&charge),
            self.d_sc.argPtr(),   self.d_active.argPtr(), self.d_x.argPtr(), self.d_rhs.argPtr(), self.d_q.argPtr(),
            self.d_ring.argPtr(), self.d_perm_q.argPtr(), arg(&nh),          self.d_ip.argPtr(),  self.d_g.argPtr(),
            self.d_c.argPtr(),    self.d_a.argPtr(),
        });
        if (opts.gmin > 0) {
            var value = opts.gmin;
            try self.launch(&self.k.gmin, self.n * N, &.{
                arg(&nl),            arg(&nn),          arg(&value),          self.d_active.argPtr(), self.d_x.argPtr(),
                self.d_rhs.argPtr(), self.d_a.argPtr(), self.d_diag.argPtr(),
            });
        }
        try self.tickLap(&mark, .assemble);
        try self.d_fails.fillAsync(0, 4, &self.stream);
        try self.d_first.fillAsync(0xff, 4, &self.stream);
        try self.launch(&self.k.refactor, N, &.{
            arg(&nl),              arg(&tab),          self.d_tab_idx.argPtr(), self.d_active.argPtr(), self.d_a.argPtr(),
            self.d_lx.argPtr(),    self.d_ux.argPtr(), self.d_udiag.argPtr(),   self.d_w.argPtr(),      self.d_fail.argPtr(),
            self.d_fails.argPtr(),
        });
        try self.launch(&self.k.solve, N, &.{
            arg(&nl),           arg(&tab),          self.d_tab_idx.argPtr(), self.d_active.argPtr(), self.d_fail.argPtr(),
            self.d_lx.argPtr(), self.d_ux.argPtr(), self.d_udiag.argPtr(),   self.d_rhs.argPtr(),    self.d_y.argPtr(),
            self.d_dx.argPtr(),
        });
        if (self.dense_slots > 0) {
            try self.d_slot_next.fillAsync(0, 4, &self.stream);
            try self.d_over.fillAsync(0, 4, &self.stream);
            var slots: u64 = self.dense_slots;
            try self.launch(&self.k.peel_dense, N, &.{
                arg(&nl),                arg(&nn),                  self.d_col_ptr.argPtr(), self.d_row_idx.argPtr(), self.d_active.argPtr(),
                self.d_fail.argPtr(),    self.d_a.argPtr(),         self.d_rhs.argPtr(),     self.d_dx.argPtr(),      self.d_dense.argPtr(),
                arg(&slots),             self.d_slot_next.argPtr(), self.d_over.argPtr(),    self.d_first.argPtr(),   self.d_seen.argPtr(),
                self.d_seen_lo.argPtr(),
            });
        }
        try self.tickLap(&mark, .lu);
    }

    /// Graph B: the update, both state passes, and the next iteration mask.
    fn updateTick(self: *Self, opts: lk.NewtonOpts) !void {
        const N: usize = self.n_lanes;
        var nl: u64 = N;
        var nn: u64 = self.n;
        var nopts = opts;
        var mark = Prof.now();
        try self.launch(&self.k.update, N, &.{
            arg(&nl),          arg(&nn),             arg(&nopts),           self.d_active.argPtr(),  self.d_gate.argPtr(),
            self.d_x.argPtr(), self.d_dx.argPtr(),   self.d_x_old.argPtr(), self.d_cur_row.argPtr(), self.d_rhs.argPtr(),
            self.d_a.argPtr(), self.d_diag.argPtr(), self.d_flags.argPtr(),
        });
        try self.limitPass(&self.d_active, &self.d_x, &self.d_x_old, device_ir.lim_pass_limit);
        try self.launch(&self.k.gates, N, &.{
            arg(&nl),              arg(&nn),            arg(&nopts),       self.d_active.argPtr(), self.d_gate.argPtr(),
            self.d_x.argPtr(),     self.d_rhs.argPtr(), self.d_a.argPtr(), self.d_diag.argPtr(),   self.d_flags.argPtr(),
            self.d_stage.argPtr(),
        });
        try self.tickLap(&mark, .gates);
        try self.limitPass(&self.d_stage, &self.d_x_old, &self.d_x_old, device_ir.lim_pass_state);
        try self.d_count.fillAsync(0, 4, &self.stream);
        var max_iter: u64 = opts.max_iter;
        try self.launch(&self.k.finalize, N, &.{
            arg(&nl),             arg(&nn),              arg(&max_iter),       self.d_active.argPtr(), self.d_stage.argPtr(),
            self.d_gate.argPtr(), self.d_flags.argPtr(), self.d_fail.argPtr(), self.d_x.argPtr(),      self.d_x_old.argPtr(),
            self.d_sims.argPtr(), self.d_count.argPtr(),
        });
        try self.tickLap(&mark, .state);
    }

    /// Newton for the round lanes, every lane its own `converger.newton`
    /// at its own t (`sims`), with `opts` (lane-uniform). Fills `out[l]`
    /// for the round lanes. `clearLimits` at the end.
    pub fn newton(self: *Self, opts: converger.Options, out: []LaneNewton) !void {
        std.debug.assert(opts.gmin_stamps.len == 0 and opts.force.len == 0);
        const nopts = newtonOpts(opts);
        self.prepareGraphs(nopts);
        var active: u64 = 0;
        for (self.round) |r| active += @intFromBool(r != 0);
        self.prof.lane_rounds += active;
        // `predictRound`'s queued copies still read `sims`.
        try self.stream.synchronize();
        const fix: u32 = if (opts.init_fix) 2 else 0;
        for (self.gate, self.round, self.sims) |*g, r, *s| {
            g.* = .{ .limit = 0, .scaled = 0, .iterations = opts.max_iter, .bits = if (r != 0) r | fix else 0 };
            // `beginSolve`.
            s.iteration = 1;
        }
        try self.put(&self.d_gate, self.gate);
        try self.put(&self.d_sims, self.sims);
        try self.put(&self.d_active, self.round);
        @memset(self.fail, 0);
        try self.put(&self.d_fail, self.fail);
        try self.ticks(nopts, opts.max_iter, active);
        // `clearLimits`, every lane's word.
        for (self.batches) |*b| try b.d_limw.fillAsync(0, 4 * @as(usize, self.n_lanes), &self.stream);
        try self.get(self.gate, &self.d_gate);
        try self.stream.synchronize();
        for (self.gate, self.round, out) |g, r, *o| {
            if (r == 0) continue;
            const singular = g.bits & 8 != 0;
            o.* = .{
                .result = if (singular) .{ .converged = false, .iterations = 0 } else .{ .converged = g.bits & 4 != 0, .iterations = @intCast(@min(g.iterations, opts.max_iter)) },
                .singular = singular,
                .reject = g.bits & 32 != 0,
            };
        }
    }

    /// The lane-uniform tolerances the tick kernels take, from `opts`.
    fn newtonOpts(opts: converger.Options) lk.NewtonOpts {
        return .{ .reltol = opts.reltol, .abstol = opts.abstol, .vntol = opts.vntol, .gmin = opts.gmin, .residual_tol = opts.residual_tol, .max_iter = opts.max_iter };
    }

    /// Newton's ticks for the lanes in `d_active` (`active` of them, or an
    /// upper bound), until every lane leaves or `max_iter`.
    fn ticks(self: *Self, nopts: lk.NewtonOpts, max_iter: u16, active_in: u64) !void {
        var active = active_in;
        // Every lane has a dense slot: no failure can need the host this
        // tick, so its words come down once, after graph B.
        const device_only = self.dense_slots >= self.n_lanes;
        var iter: u16 = 0;
        while (iter < max_iter) : (iter += 1) {
            self.prof.lane_ticks += active;
            if (self.graph_a) |*graph| try graph.launch(&self.stream) else try self.linearTick(nopts);
            if (!device_only) {
                try self.readWords();
                const fails = self.words[0];
                // Most lanes failing means the tape went stale, not a few
                // lanes going bad: refresh it rather than peel every tick.
                const stale = 2 * @as(u64, fails) >= active;
                const host = if (self.dense_slots > 0) self.words[3] else fails;
                if (host > 0) {
                    self.prof.peels += host;
                    try self.peel(stale);
                } else if (fails > 0 and stale) try self.retapeFrom(self.words[2]);
            }

            if (self.graph_b) |*graph| {
                try graph.launch(&self.stream);
                self.prof.graph_ticks += 1;
            } else try self.updateTick(nopts);
            self.prof.ticks += 1;
            try self.readWords();
            const count = self.words[1];
            if (device_only) {
                const fails = self.words[0];
                self.prof.dense_peels += fails;
                if (fails > 0 and 2 * @as(u64, fails) >= active) try self.retapeFrom(self.words[2]);
            }
            if (count == 0) break;
            active = count;
        }
    }

    /// The lanes whose device refactor failed this tick: each factors and
    /// solves on its own solver from its downloaded A and rhs, as
    /// `LaneSolver` peels; a singular one stays failed (`finalize` marks it).
    /// With `refresh`, the first lane that factors becomes the tape's lead
    /// (`retape`), as `LaneSolver`'s lead follows its own solver.
    fn peel(self: *Self, refresh: bool) !void {
        const gpa = self.gpa;
        const n: usize = self.n;
        const nnz: usize = self.nnz;
        try self.get(self.fail, &self.d_fail);
        try self.stream.synchronize();
        // Only `fail == 1` lanes are the host's: 2 is the dense peel's
        // singular verdict, which stands.
        var k: usize = 0;
        for (self.fail) |f| k += @intFromBool(f == 1);
        if (k == 0) return;
        const idx = try gpa.alloc(u32, k);
        defer gpa.free(idx);
        k = 0;
        for (self.fail, 0..) |f, l| if (f == 1) {
            idx[k] = @intCast(l);
            k += 1;
        };
        // Per lane: A (nnz), then rhs (n), then dx (n).
        const per = nnz + 2 * n;
        const host = try gpa.alloc(f64, k * per);
        defer gpa.free(host);
        try self.ensurePeel(k);
        try self.d_peel_idx.upload(idx.ptr, k * 4);
        try self.packLanes(k, nnz, &self.d_a, 0);
        try self.packLanes(k, n, &self.d_rhs, k * nnz);
        try self.d_peel.download(host.ptr, k * (nnz + n) * @sizeOf(f64));
        const xs = host[k * (nnz + n) ..][0 .. k * n];
        var lead: ?usize = null;
        for (idx, 0..) |l32, j| {
            const l: usize = l32;
            const slv = &self.ckts[l].ws.?.slv;
            slv.factor(host[j * nnz ..][0..nnz], self.ckts[l].solver_execution) catch |e| switch (e) {
                error.SingularMatrix => {
                    @memset(xs[j * n ..][0..n], 0);
                    continue;
                },
                else => |other| return other,
            };
            slv.solveNeg(host[k * nnz + j * n ..][0..n], xs[j * n ..][0..n]);
            self.fail[l] = 0;
            if (lead == null) lead = l;
        }
        // Singular lanes keep fail = 1, so their scattered zeros are unread.
        try self.d_peel.uploadAt(xs.ptr, k * (nnz + n) * @sizeOf(f64), k * n * @sizeOf(f64));
        try self.unpackLanes(k, n, k * (nnz + n), &self.d_dx);
        try self.put(&self.d_fail, self.fail);
        try self.stream.synchronize();
        if (refresh) if (lead) |l| try self.retape(l);
    }

    /// The per-tick device words into `words`, one sync.
    fn readWords(self: *Self) !void {
        try self.d_fails.downloadAtAsync(&self.words[0], 0, 4, &self.stream);
        try self.d_count.downloadAtAsync(&self.words[1], 0, 4, &self.stream);
        try self.d_first.downloadAtAsync(&self.words[2], 0, 4, &self.stream);
        try self.d_over.downloadAtAsync(&self.words[3], 0, 4, &self.stream);
        try self.stream.synchronize();
    }

    /// Room in `d_peel_idx`/`d_peel` for `k` packed lanes.
    fn ensurePeel(self: *Self, k: usize) !void {
        if (k <= self.peel_cap) return;
        const k0 = &self.k.refactor;
        const per = (self.nnz + 2 * @as(usize, self.n)) * @sizeOf(f64);
        var idx = try k0.alloc(k * 4);
        errdefer idx.free();
        const buf = try k0.alloc(k * per);
        self.d_peel_idx.free();
        self.d_peel.free();
        self.d_peel_idx = idx;
        self.d_peel = buf;
        self.peel_cap = @intCast(k);
    }

    /// `arp_lane_pack` of `rows` rows of `src` for the `k` lanes in
    /// `d_peel_idx`, into `d_peel` from entry `at`.
    fn packLanes(self: *Self, k: usize, rows: usize, src: *Buffer, at: usize) !void {
        var nl: u64 = self.n_lanes;
        var r: u64 = rows;
        var kk: u64 = k;
        var off: u64 = at;
        try self.launch(&self.k.pack, rows * k, &.{ arg(&nl), arg(&r), arg(&kk), self.d_peel_idx.argPtr(), src.argPtr(), self.d_peel.argPtr(), arg(&off) });
    }

    fn unpackLanes(self: *Self, k: usize, rows: usize, at: usize, dst: *Buffer) !void {
        var nl: u64 = self.n_lanes;
        var r: u64 = rows;
        var kk: u64 = k;
        var off: u64 = at;
        try self.launch(&self.k.unpack, rows * k, &.{ arg(&nl), arg(&r), arg(&kk), self.d_peel_idx.argPtr(), self.d_peel.argPtr(), arg(&off), dst.argPtr() });
    }

    /// The device dense peel saw most lanes fail: factor lane `first`'s
    /// matrix on its host solver and make it the tape's lead (`retape`).
    /// A singular lead leaves the tape as it was. Called with the stream
    /// idle.
    fn retapeFrom(self: *Self, first: u32) !void {
        if (first >= self.n_lanes) return;
        const nnz: usize = self.nnz;
        const vals = try self.gpa.alloc(f64, nnz);
        defer self.gpa.free(vals);
        try self.ensurePeel(1);
        try self.d_peel_idx.upload(&first, 4);
        try self.packLanes(1, nnz, &self.d_a, 0);
        try self.d_peel.download(vals.ptr, nnz * @sizeOf(f64));
        const l: usize = first;
        const slv = &self.ckts[l].ws.?.slv;
        slv.factor(vals, self.ckts[l].solver_execution) catch |e| switch (e) {
            error.SingularMatrix => return,
            else => |other| return other,
        };
        try self.retape(l);
    }

    /// Rebuilds the replay tape from lane l's solver, just factored on its
    /// own matrix, and uploads it. The lanes' L/U values are rewritten by
    /// the next refactor, so only their capacity carries over. The captured
    /// graphs hold the old tape, so they go; the next `newton` recaptures.
    /// Called with the stream idle.
    fn retape(self: *Self, l: usize) !void {
        const slv = &self.ckts[l].ws.?.slv;
        const lu = slv.sparse() orelse return;
        if (!lu.factored) return;
        const built = try lk.buildTab(self.gpa, lu, slv.col_ptr, slv.params.refactor_growth_limit, direct.pivot_tol);
        defer self.gpa.free(built.idx);
        const t = built.tab;
        const k0 = &self.k.refactor;
        const per_lane = @as(usize, self.n_lanes) * @sizeOf(f64);
        var idx = try upload(k0, std.mem.sliceAsBytes(built.idx));
        errdefer idx.free();
        if (t.nnz_l > self.lx_cap) {
            const lx = try zeros(k0, t.nnz_l * per_lane);
            self.d_lx.free();
            self.d_lx = lx;
            self.lx_cap = t.nnz_l;
        }
        if (t.nnz_u > self.ux_cap) {
            const ux = try zeros(k0, t.nnz_u * per_lane);
            self.d_ux.free();
            self.d_ux = ux;
            self.ux_cap = t.nnz_u;
        }
        self.d_tab_idx.free();
        self.d_tab_idx = idx;
        self.tab = t;
        self.clearGraphs();
        self.graph_opts = null;
        self.prof.retapes += 1;
    }

    /// The step-end numbers of the round lanes at their Newton solution:
    /// the state-flip query, the charges and their tape at x, the LTE
    /// bounds and the probe rows. One sync.
    pub fn probe(self: *Self) !void {
        const N: usize = self.n_lanes;
        // 1. The flip query, read-only.
        for (self.ops, self.round) |*o, r| o.* = if (r != 0) 1 + @as(u8, @backingInt(StateCtlOp.query)) else 0;
        try self.put(&self.d_ops, self.ops);
        try self.d_flags.fillAsync(0, N * 4, &self.stream);
        try self.ctlPass();
        // 2. The charges at x (`Sim.end`'s `evalQ`), and the tape into each
        // lane's logical slot 0.
        if (self.has_charge) {
            const nh: usize = self.n_hist;
            for (self.slot0, 0..) |*s, l| s.* = self.perm_qt[l * nh];
            try self.chargePass(&self.d_round, &self.d_x);
        }
        if (self.n_qt > 0) {
            // 3. The LTE bounds.
            try self.lteKernel();
            try self.get(self.lte_out, &self.d_lte_out);
        }
        // 4. The probe rows.
        if (self.n_probes > 0) {
            try self.gatherKernel();
            try self.get(self.probe_out[0 .. N * self.n_probes], &self.d_probe_out);
        }
        try self.get(self.query, &self.d_flags);
        try self.stream.synchronize();
        // The query's copy has run; `Sim.end` fills `ops` afresh.
        @memset(self.ops, 0);
        const n: usize = self.n;
        for (self.round, 0..) |r, l| {
            if (r == 0) continue;
            const row = self.rows[l * n ..][0..n];
            for (self.probes, 0..) |p, k| row[p] = self.probe_out[l * self.n_probes + k];
        }
    }

    /// Every batch's lane ctl kernel with the per-lane `ops` on the device.
    fn ctlPass(self: *Self) !void {
        const N: usize = self.n_lanes;
        for (self.batches) |*b| {
            const k = if (b.ctl) |*k| k else continue;
            if (b.count == 0) continue;
            var nl: u64 = N;
            var count: u64 = b.count;
            try self.launch(k, b.count * N, &.{ arg(&nl), arg(&count), self.d_ops.argPtr(), b.d_models.argPtr(), b.d_model_of.argPtr(), b.d_instances.argPtr(), b.d_states.argPtr(), self.d_flags.argPtr() });
        }
    }

    /// Runs the round's decisions (`Ops`) on the device: revert or commit
    /// per lane, then each accepted lane's history update. Clears them.
    pub fn apply(self: *Self) !void {
        const N: usize = self.n_lanes;
        try self.put(&self.d_ops, self.ops);
        try self.d_flags.fillAsync(0, N * 4, &self.stream);
        try self.ctlPass();
        try self.put(&self.d_sc, self.sc);
        try self.put(&self.d_accept, self.accept_);
        try self.acceptKernel();
        if (self.state_q) try self.rebaseRound();
        try self.put(&self.d_perm_q, self.perm_q);
        try self.put(&self.d_perm_qt, self.perm_qt);
        try self.stream.synchronize();
        @memset(self.ops, 0);
        @memset(self.accept_, 0);
    }

    /// Hands step control to the device: uploads each lane's `LaneStep`
    /// (`tran.runGpuLanes` fills them after `Sim.init`), loads every
    /// breakpoint batch's lane kernel and seeds its source caches from each
    /// lane's `GpuPayload.brk`. Rounds then go through `deviceLaunch`.
    pub fn initSteps(self: *Self, steps: []const lk.LaneStep) !void {
        const N: usize = self.n_lanes;
        std.debug.assert(steps.len == N);
        try self.d_step.upload(steps.ptr, N * @sizeOf(lk.LaneStep));
        const c0 = self.ckts[0];
        for (self.batches, 0..) |*b, bi| {
            const h0 = c0.batches[bi].hooks;
            if (h0.next_breakpoint == null or b.count == 0) continue;
            var name: [160]u8 = undefined;
            const p0 = h0.gpu_payload.?(c0.batches[bi].ctx);
            b.brk = hdl.kernel(backend.?, @ptrCast(h0.gpu_payload.?), laneName(&name, p0.kernel, "brk") orelse return Error.NoLaneImage) catch |e|
                return if (e == error.KernelNotFound) Error.NoLaneImage else e;
            if (p0.brk.len == 0) continue;
            const W = 2 * @sizeOf(f64);
            std.debug.assert(p0.brk.len == b.count * W);
            const host = try self.gpa.alloc(u8, b.count * N * W);
            defer self.gpa.free(host);
            for (self.ckts, 0..) |c, l| {
                const p = c.batches[bi].hooks.gpu_payload.?(c.batches[bi].ctx);
                for (0..b.count) |id| @memcpy(host[(id * N + l) * W ..][0..W], p.brk[id * W ..][0..W]);
            }
            b.d_brk = try upload(&self.k.refactor, host);
        }
        self.device_step = true;
    }

    /// Hands the `.meas` passes to the device (06 6.8): `scans` holds every
    /// lane's `n` passes lane-major (pass k of lane l at `k * N + l`, as
    /// `measure.Stream.deviceScans` gives them), `cols` the unknowns each
    /// reads. Each round, a lane that accepts a step at or past `t_start`
    /// feeds them its solution, and `measResolved` tells the round they
    /// all finished in.
    pub fn initMeasures(self: *Self, scans: []const passes.Dev, cols: []const passes.DevColumns, t_start: f64) !void {
        const N: usize = self.n_lanes;
        std.debug.assert(scans.len == cols.len * N and cols.len > 0);
        const k0 = &self.k.refactor;
        self.d_meas_cols.free();
        self.d_meas_cols = try upload(k0, std.mem.sliceAsBytes(cols));
        self.d_meas.free();
        self.d_meas = try upload(k0, std.mem.sliceAsBytes(scans));
        self.d_meas_resolved.free();
        self.d_meas_resolved = try k0.alloc(N);
        try self.d_meas_resolved.upload(self.meas_resolved.ptr, N);
        self.meas_host = try self.gpa.alloc(passes.Dev, scans.len);
        self.n_meas = @intCast(cols.len);
        self.meas_t_start = t_start;
        self.clearGraphs();
    }

    /// The ring slot of the last launch whose round finished lane l's
    /// passes, or null.
    pub fn measResolved(self: *const Self, l: usize) ?u8 {
        const r = self.meas_resolved[l];
        return if (r == 0xff) null else r;
    }

    /// Every lane's passes into `meas_host`, for `lanePasses`.
    pub fn fetchMeasures(self: *Self) !void {
        try self.d_meas.download(self.meas_host.ptr, self.meas_host.len * @sizeOf(passes.Dev));
    }

    /// Lane l's passes of the last `fetchMeasures` into `out`.
    pub fn lanePasses(self: *const Self, l: usize, out: []passes.Dev) void {
        const N: usize = self.n_lanes;
        for (out, 0..) |*o, k| o.* = self.meas_host[k * N + l];
    }

    /// Most rounds one `deviceLaunch` runs; the ring holds this many.
    pub const max_rounds = 16;
    /// Newton ticks per round of a multi-round launch. A lane needing more
    /// carries its Newton over; most transient steps converge in 2 or 3.
    const phased_ticks = 3;

    /// Whether `deviceLaunch` may run more than one round: every failed
    /// replay solves on the device (one dense slot per lane), so no tick
    /// needs the host, and the driver can capture graphs.
    pub fn multiRound(self: *const Self) bool {
        return self.dense_slots >= self.n_lanes and self.k.refactor.hasGraphs() and
            !self.prof.phases and !self.ckts[0].knobs.gpu_nographs;
    }

    /// `rounds` rounds under device step control (1 unless `multiRound`):
    /// every stepping lane begins, predicts, solves Newton, probes, decides,
    /// applies its decision, asks for its next breakpoint and lands it, all
    /// on the device; each round's `report` and probe rows go to ring slot
    /// r (`ring_report[r * N + l]`, rows at `(r * N + l) * n_probes`). One
    /// sync at the end brings the ring down. Stops the host queued in `stop`
    /// take effect before the first round. A multi-round launch replays one
    /// captured round with its Newton unrolled to `max_iter` masked ticks,
    /// and refreshes a stale tape only at its end.
    pub fn deviceLaunch(self: *Self, opts: converger.Options, rounds: u32) !void {
        std.debug.assert(self.device_step);
        std.debug.assert(rounds >= 1 and rounds <= max_rounds);
        std.debug.assert(rounds == 1 or self.multiRound());
        std.debug.assert(opts.gmin_stamps.len == 0 and opts.force.len == 0 and !opts.init_fix);
        const N: usize = self.n_lanes;
        var nl: u64 = N;
        const nopts = newtonOpts(opts);
        self.prepareGraphs(nopts);
        // ponytail: N bytes up every launch for the host's stops; a flag
        // word first if that ever shows at 100k lanes.
        try self.put(&self.d_stop, self.stop);
        try self.launch(&self.k.step_stop, N, &.{ arg(&nl), self.d_stop.argPtr(), self.d_step.argPtr() });
        try self.d_ring_slot.fillAsync(0, 4, &self.stream);
        try self.d_seen.fillAsync(0, 4, &self.stream);
        try self.d_seen_lo.fillAsync(0xff, 4, &self.stream);
        if (rounds == 1) {
            try self.roundKernels(nopts, opts.max_iter, false);
        } else {
            if (self.graph_round == null) {
                try self.stream.synchronize();
                self.graph_round = try self.capture(roundKernels, .{ nopts, opts.max_iter, true });
            }
            for (0..rounds) |_| try self.graph_round.?.launch(&self.stream);
        }
        const np: usize = self.n_probes;
        try self.get(self.ring_report[0 .. rounds * N], &self.d_ring_report);
        if (np > 0) try self.get(self.ring_rows[0 .. rounds * N * np], &self.d_ring_rows);
        if (self.n_meas > 0) try self.get(self.meas_resolved, &self.d_meas_resolved);
        try self.d_seen.downloadAtAsync(&self.words[0], 0, 4, &self.stream);
        try self.d_seen_lo.downloadAtAsync(&self.words[2], 0, 4, &self.stream);
        try self.stream.synchronize();
        // The stops' copy has run.
        @memset(self.stop, 0);
        self.prof.rounds += rounds;
        // A multi-round launch read no per-tick words (`ticks` counts a
        // single round's): refresh a tape most lanes kept failing on now,
        // from the lowest failing lane.
        if (rounds > 1) {
            self.prof.dense_peels += self.words[0];
            if (2 * @as(u64, self.words[0]) >= N * rounds) try self.retapeFrom(self.words[2]);
        }
    }

    /// One device-controlled round's kernels, `ringPut` included. Newton's
    /// ticks are host-driven (`ticks`), or under `unrolled` `max_iter`
    /// masked ticks that read nothing back, for capture.
    fn roundKernels(self: *Self, nopts: lk.NewtonOpts, max_iter: u16, unrolled: bool) !void {
        const N: usize = self.n_lanes;
        var nl: u64 = N;
        var nh: u64 = self.n_hist;
        // Begin and predict.
        try self.launch(&self.k.step_begin, N, &.{ arg(&nl), self.d_step.argPtr(), self.d_sc.argPtr(), self.d_sims.argPtr(), self.d_lte_in.argPtr(), self.d_round.argPtr(), self.d_ops.argPtr(), self.d_accept.argPtr() });
        try self.predictKernels();
        // Newton.
        var mi: u64 = max_iter;
        try self.launch(&self.k.newton_init, N, &.{ arg(&nl), arg(&mi), self.d_round.argPtr(), self.d_gate.argPtr(), self.d_sims.argPtr(), self.d_active.argPtr(), self.d_fail.argPtr() });
        if (unrolled) {
            // Each lane counts its own ticks, so one whose Newton is not
            // done carries it into the next round instead of every round
            // paying `max_iter` masked ticks.
            for (0..phased_ticks) |_| {
                try self.linearTick(nopts);
                try self.updateTick(nopts);
            }
        } else {
            // The round's lane count is on the device; N bounds it, which
            // only makes the stale-tape test stricter on the first tick.
            try self.ticks(nopts, max_iter, N);
        }
        // The lanes whose Newton ended: `clearLimits` for them, and the
        // probe's mask.
        try self.launch(&self.k.settle, N, &.{ arg(&nl), self.d_step.argPtr(), self.d_active.argPtr(), self.d_decide.argPtr(), self.d_sc.argPtr() });
        for (self.batches) |*b| if (b.lim != null and b.has_lim) try self.setLimw(b, &self.d_decide, 0);
        // Probe: the flip query, the charges at x into the tape's logical
        // slot 0, the LTE pair and the probe rows.
        var query: u64 = 1 + @as(u64, @backingInt(StateCtlOp.query));
        try self.launch(&self.k.set_ops, N, &.{ arg(&nl), self.d_decide.argPtr(), self.d_ops.argPtr(), arg(&query) });
        try self.d_flags.fillAsync(0, N * 4, &self.stream);
        try self.ctlPass();
        var probe_slot: u64 = 0;
        var rebase_slot: u64 = 1;
        if (self.has_charge) {
            try self.launch(&self.k.slot_zero, N, &.{ arg(&nl), arg(&nh), arg(&probe_slot), self.d_perm_qt.argPtr(), self.d_sc.argPtr(), self.d_slot0.argPtr() });
            try self.chargeKernels(&self.d_decide, &self.d_x);
        }
        if (self.n_qt > 0) try self.lteKernel();
        if (self.n_probes > 0) try self.gatherKernel();
        // Decide (`Sim.end` to the commit).
        try self.launch(&self.k.step_end_a, N, &.{
            arg(&nl),                arg(&nh),             self.d_decide.argPtr(), self.d_step.argPtr(), self.d_gate.argPtr(),   self.d_flags.argPtr(),
            self.d_lte_out.argPtr(), self.d_ops.argPtr(),  self.d_accept.argPtr(), self.d_sc.argPtr(),   self.d_perm_q.argPtr(), self.d_perm_qt.argPtr(),
            self.d_brk_t.argPtr(),   self.d_best.argPtr(),
        });
        // Apply: the latches, the accepted histories, the stateful rebase.
        try self.d_flags.fillAsync(0, N * 4, &self.stream);
        try self.ctlPass();
        try self.acceptKernel();
        if (self.state_q) {
            try self.launch(&self.k.slot_zero, N, &.{ arg(&nl), arg(&nh), arg(&rebase_slot), self.d_perm_qt.argPtr(), self.d_sc.argPtr(), self.d_slot0.argPtr() });
            try self.chargeKernels(&self.d_accept, &self.d_cur);
            try self.rebaseKernel();
        }
        // The next breakpoint at the committed state, then the landing.
        for (self.batches) |*b| {
            const k = if (b.brk) |*k| k else continue;
            var count: u64 = b.count;
            const cache = if (b.d_brk) |*c| c.argPtr() else self.d_best.argPtr();
            try self.launch(k, N, &.{ arg(&nl), arg(&count), self.d_accept.argPtr(), self.d_brk_t.argPtr(), b.d_models.argPtr(), b.d_model_of.argPtr(), b.d_instances.argPtr(), cache, self.d_best.argPtr() });
        }
        try self.launch(&self.k.step_end_b, N, &.{ arg(&nl), self.d_step.argPtr(), self.d_best.argPtr(), self.d_report.argPtr() });
        var np: u64 = self.n_probes;
        try self.launch(&self.k.ring_put, N, &.{ arg(&nl), arg(&np), self.d_ring_slot.argPtr(), self.d_report.argPtr(), self.d_ring_report.argPtr(), self.d_probe_out.argPtr(), self.d_ring_rows.argPtr() });
        if (self.n_meas > 0) {
            var n_meas: u64 = self.n_meas;
            var t_start: f64 = self.meas_t_start;
            try self.launch(&self.k.meas, N, &.{ arg(&nl), arg(&n_meas), arg(&t_start), self.d_report.argPtr(), self.d_x.argPtr(), self.d_meas_cols.argPtr(), self.d_meas.argPtr(), self.d_meas_resolved.argPtr(), self.d_ring_slot.argPtr() });
        }
        try self.launch(&self.k.ring_next, 1, &.{self.d_ring_slot.argPtr()});
    }

    /// Lane l's report in ring slot r of the last `deviceLaunch`.
    pub fn slotReport(self: *const Self, r: usize, l: usize) StepReport {
        return self.ring_report[r * self.n_lanes + l];
    }

    /// Lane l's accepted row of ring slot r into `rows`, as `probe` leaves
    /// rows: the probe positions of an n-long vector.
    pub fn slotRow(self: *Self, r: usize, l: usize) []const f64 {
        const np: usize = self.n_probes;
        const row = self.rows[l * self.n ..][0..self.n];
        const src = self.ring_rows[(r * self.n_lanes + l) * np ..][0..np];
        for (self.probes, src) |p, v| row[p] = v;
        return row;
    }

    /// Every lane's `LaneStep` into `out`, for the lanes the host finishes.
    pub fn fetchSteps(self: *Self, out: []lk.LaneStep) !void {
        std.debug.assert(out.len == self.n_lanes);
        try self.d_step.download(out.ptr, out.len * @sizeOf(lk.LaneStep));
    }

    /// Stops lane l before the next round begins. Touches only lane l's
    /// byte, so the threaded drain may call it; `deviceLaunch` uploads every
    /// lane's byte each round.
    pub fn requestStop(self: *Self, l: usize) void {
        self.stop[l] = 1;
    }

    /// `Sim.end`'s stateful-charge block for the accepted lanes, after the
    /// commits: the planes at cur, the tape into the tape's logical slot 0
    /// (`sqt[3]`), then `arp_lane_rebase`.
    fn rebaseRound(self: *Self) !void {
        for (self.slot0, self.sc) |*s, sc| s.* = sc.sqt[3];
        try self.chargePass(&self.d_accept, &self.d_cur);
        try self.rebaseKernel();
    }

    /// `arp_lane_accept`: each accepted lane's `prev <- cur <- x` and its
    /// charge and tape histories (`d_accept`, `d_sc`'s ring slots).
    fn acceptKernel(self: *Self) !void {
        const N: usize = self.n_lanes;
        var nl: u64 = N;
        var n: u64 = self.n;
        var n_qt: u64 = self.n_qt;
        var nh: u64 = self.n_hist;
        var charge: u64 = @intFromBool(self.has_charge);
        try self.launch(&self.k.accept, (self.n + self.n_qt) * N, &.{
            arg(&nl),           arg(&n),                arg(&n_qt),         arg(&nh),                arg(&charge),
            self.d_sc.argPtr(), self.d_accept.argPtr(), self.d_x.argPtr(),  self.d_cur.argPtr(),     self.d_prev.argPtr(),
            self.d_q.argPtr(),  self.d_ring.argPtr(),   self.d_ip.argPtr(), self.d_ring_qt.argPtr(), self.d_ip_qt.argPtr(),
        });
    }

    /// `arp_lane_rebase` for the accepted lanes, after their charges at
    /// cur landed in the q plane and the tape's slot 0.
    fn rebaseKernel(self: *Self) !void {
        const N: usize = self.n_lanes;
        var nl: u64 = N;
        var n: u64 = self.n;
        var n_qt: u64 = self.n_qt;
        var nh: u64 = self.n_hist;
        try self.launch(&self.k.rebase, (self.n + self.n_qt) * N, &.{
            arg(&nl),               arg(&n),           arg(&n_qt),           arg(&nh),           self.d_sc.argPtr(),
            self.d_accept.argPtr(), self.d_q.argPtr(), self.d_ring.argPtr(), self.d_ip.argPtr(), self.d_ring_qt.argPtr(),
            self.d_ip_qt.argPtr(),
        });
    }

    /// `arp_lane_lte`: each lane's LTE pair into `d_lte_out`.
    fn lteKernel(self: *Self) !void {
        const N: usize = self.n_lanes;
        var nl: u64 = N;
        var nh: u64 = self.n_hist;
        var n_qt: u64 = self.n_qt;
        try self.launch(&self.k.lte, N, &.{
            arg(&nl), self.d_sc.argPtr(), self.d_ring_qt.argPtr(), self.d_perm_qt.argPtr(), self.d_ip_qt.argPtr(), arg(&nh), arg(&n_qt), self.d_lte_in.argPtr(), self.d_lte_out.argPtr(),
        });
    }

    /// `arp_lane_gather`: each lane's probe rows of `d_x` into `d_probe_out`.
    fn gatherKernel(self: *Self) !void {
        const N: usize = self.n_lanes;
        var nl: u64 = N;
        var np: u64 = self.n_probes;
        try self.launch(&self.k.gather, self.n_probes * N, &.{ arg(&nl), self.d_sc.argPtr(), self.d_x.argPtr(), self.d_probes.argPtr(), arg(&np), self.d_probe_out.argPtr() });
    }

    /// `Circuit.evalQ` at `x` for the lanes of `mask`: the q plane zeroed,
    /// then every charged batch's lane charge kernel, which also writes the
    /// tape into each lane's slot `slot0[l]` (uploaded here).
    fn chargePass(self: *Self, mask: *Buffer, x: *Buffer) !void {
        try self.put(&self.d_slot0, self.slot0);
        try self.chargeKernels(mask, x);
    }

    /// `chargePass` with `d_slot0` already on the device.
    fn chargeKernels(self: *Self, mask: *Buffer, x: *Buffer) !void {
        const N: usize = self.n_lanes;
        try self.d_q.fillAsync(0, (self.n + 1) * N * @sizeOf(f64), &self.stream);
        for (self.batches) |*b| {
            const k = if (b.chg) |*k| k else continue;
            if (b.count == 0) continue;
            var nl: u64 = N;
            var count: u64 = b.count;
            var n_hist: u64 = self.n_hist;
            var n_qt: u64 = self.n_qt;
            var off: u64 = b.tape_off;
            const col = if (b.color) |*c| c else {
                var none: u64 = 0;
                try self.launch(k, N, &.{
                    arg(&nl),          arg(&count),             mask.argPtr(),         self.d_sims.argPtr(),  x.argPtr(),
                    b.d_gath.argPtr(), b.d_rhs_idx.argPtr(),    b.d_models.argPtr(),   b.d_model_of.argPtr(), b.d_instances.argPtr(),
                    self.d_q.argPtr(), self.d_ring_qt.argPtr(), self.d_slot0.argPtr(), arg(&n_hist),          arg(&n_qt),
                    arg(&off),         b.d_slots.argPtr(),      arg(&none),            arg(&none),
                });
                continue;
            };
            // As `evalAll` colours eval (06 6.12): one launch per colour,
            // the q rows of hub cells staged, then the row hubs, with q as
            // both planes (it reads, sums and resets each cell once per
            // plane, so the second pass rewrites the same sums).
            for (col.starts[0 .. col.starts.len - 1], col.starts[1..]) |lo, hi| {
                var ids_off: u64 = lo;
                var n_ids: u64 = hi - lo;
                try self.launch(k, (hi - lo) * N, &.{
                    arg(&nl),          arg(&count),             mask.argPtr(),         self.d_sims.argPtr(),  x.argPtr(),
                    b.d_gath.argPtr(), col.d_rows.argPtr(),     b.d_models.argPtr(),   b.d_model_of.argPtr(), b.d_instances.argPtr(),
                    self.d_q.argPtr(), self.d_ring_qt.argPtr(), self.d_slot0.argPtr(), arg(&n_hist),          arg(&n_qt),
                    arg(&off),         col.d_order.argPtr(),    arg(&ids_off),         arg(&n_ids),
                });
            }
            const hubs = &col.row_hubs;
            var n_hubs: u64 = hubs.n;
            try self.launch(&self.k.hub_reduce, hubs.n * N, &.{
                arg(&nl),                arg(&n_hubs),      mask.argPtr(),     hubs.d_cells.argPtr(), hubs.d_starts.argPtr(),
                hubs.d_entries.argPtr(), self.d_q.argPtr(), self.d_q.argPtr(),
            });
        }
    }

    /// Records lane l's pre-rotation ring slots for `accept`'s history
    /// update (an accepted step with charge) and rotates
    /// its permutations as `std.mem.rotate` moves the host rings.
    fn queueAccept(self: *Self, l: usize) void {
        const nh: usize = self.n_hist;
        const pq = self.perm_q[l * nh ..][0..nh];
        const pqt = self.perm_qt[l * nh ..][0..nh];
        for (0..3) |k| {
            self.sc[l].sq[k] = pq[k];
            self.sc[l].sqt[k] = pqt[k];
        }
        std.mem.rotate(u8, pq, nh - 1);
        std.mem.rotate(u8, pqt, nh - 1);
    }
};

/// One fixed trapezoidal period, after the host has seeded its seam state.
/// Stateful charge is excluded. A device requesting a rejection during
/// Newton reruns this period on the host from its unchanged starting state.
pub const FixedLane = struct {
    ckt: *Circuit,
    x: []f64,
    q_prev: []f64,
    i_prev: []f64,
    period: f64,
    n_samples: u32,
    probes: []const u32,
    wave: []f64,
};

/// Sampled periodic device evaluation, with independent storage for each
/// (sample, point). It owns no LU, Newton state or integration history.
/// Flattened lane = sample * points + point; input/output vectors and
/// planes have node/slot outermost. The point circuits are read-only and
/// must outlive this object.
pub const Samples = struct {
    gpa: std.mem.Allocator,
    points: usize,
    nt: usize,
    n: usize,
    nnz: usize,
    ground_slot: usize,
    batches: []LaneBatch,
    active: []u8,
    stream: Stream,
    d_x: Buffer,
    d_f: Buffer,
    d_q: Buffer,
    d_g: Buffer,
    d_c: Buffer,
    d_sims: Buffer,
    d_active: Buffer,
    d_limw: Buffer,

    const buffers = .{ "d_x", "d_f", "d_q", "d_g", "d_c", "d_sims", "d_active", "d_limw" };

    /// Packs and uploads every (sample, point) lane of `ckts` at `times`
    /// with step `dt`. The sampled operator does not step analog state, so
    /// a history-dependent circuit (VPI, display tasks, a load check,
    /// stateful charge, a delay, or any state hook) fails with
    /// `error.UnsupportedSampleState` before packing rather than sampling
    /// one point's mutable history in another lane; timer-only source
    /// State is unused, its state hooks null. Fails with
    /// `error.InvalidSamples` on no lanes or a nonfinite time or step,
    /// `error.IncompatibleSamples` across patterns, `Error.NoLaneImage`, or
    /// the driver's error. Free with `deinit`.
    pub fn init(gpa: std.mem.Allocator, ckts: []const *Circuit, times: []const f64, dt: f64) !*Samples {
        if (comptime backend == null) return Error.NoLaneImage;
        if (ckts.len == 0 or times.len == 0 or !std.math.isFinite(dt) or dt <= 0) return error.InvalidSamples;
        const c0 = ckts[0];
        const n: usize = c0.n;
        const nnz: usize = c0.nnz;
        const N = try std.math.mul(usize, ckts.len, times.len);
        // Every sample lane's lim-active word, never set: the sampled
        // operator runs no limit pass.
        const no_lim = try gpa.alloc(u8, N * 4);
        defer gpa.free(no_lim);
        @memset(no_lim, 0);
        if (N > std.math.maxInt(u32)) return error.InvalidSamples;
        for (times) |time| if (!std.math.isFinite(time)) return error.InvalidSamples;
        for (ckts) |c| {
            if (!samePattern(c, c0)) return error.IncompatibleSamples;
            if (c.vpi != null or c.display != null or c.load_check != null or c.has_state_q or c.minDelay() != null)
                return error.UnsupportedSampleState;
            for (c.batches) |b| {
                const h = b.hooks;
                if (h.gpu_payload == null) return Error.NoLaneImage;
                if (h.status != null or h.update_state != null or h.state_ctl != null or h.commit_state != null or
                    h.advance_iteration != null or h.check_convergence != null or h.predict_first_iterate != null)
                    return error.UnsupportedSampleState;
            }
        }
        if (c0.batches.len == 0) return Error.NoLaneImage;
        const batches = try gpa.alloc(LaneBatch, c0.batches.len);
        errdefer gpa.free(batches);
        var n_batches: usize = 0;
        errdefer for (batches[0..n_batches]) |*b| b.deinit();
        const ps = try gpa.alloc(device_ir.GpuPayload, N);
        defer gpa.free(ps);
        for (c0.batches, 0..) |b0, bi| {
            const p0 = b0.hooks.gpu_payload.?(b0.ctx);
            for (ps, 0..) |*p, lane_id| {
                const b = ckts[lane_id % ckts.len].batches[bi];
                p.* = b.hooks.gpu_payload.?(b.ctx);
                if (!sameRows(p.*, p0)) return error.IncompatibleSamples;
            }
            const pk = try pack(gpa, ps);
            defer pk.deinit(gpa);
            var name: [160]u8 = undefined;
            var eval = try hdl.kernel(backend.?, @ptrCast(b0.hooks.gpu_payload.?), laneName(&name, p0.kernel, "eval") orelse return Error.NoLaneImage);
            errdefer eval.deinit();
            const Bufs = struct {
                models: Buffer,
                model_of: Buffer,
                instances: Buffer,
                states: Buffer,
                lim: Buffer,
                gath: Buffer,
                rhs_idx: Buffer,
                slots: Buffer,
                limw: Buffer,
            };
            var bs: Bufs = undefined;
            const data = .{
                pk.models,                    std.mem.sliceAsBytes(pk.model_of), pk.instances,                     pk.states,
                std.mem.sliceAsBytes(pk.lim), std.mem.sliceAsBytes(p0.gath),     std.mem.sliceAsBytes(p0.rhs_idx), std.mem.sliceAsBytes(p0.slots),
                no_lim,
            };
            var made: usize = 0;
            errdefer inline for (@typeInfo(Bufs).@"struct".field_names, 0..) |f, j| {
                if (j < made) @field(bs, f).free();
            };
            inline for (@typeInfo(Bufs).@"struct".field_names, 0..) |f, j| {
                @field(bs, f) = try GpuLanes.upload(&eval, data[j]);
                made += 1;
            }
            batches[bi] = .{
                .eval = eval,
                .lim = null,
                .ctl = null,
                .chg = null,
                .count = p0.count,
                .has_lim = false,
                .tape_off = 0,
                .inst_size = if (p0.count == 0) 0 else p0.instances.len / p0.count,
                .state_size = if (p0.count == 0) 0 else p0.states.len / p0.count,
                .d_models = bs.models,
                .d_model_of = bs.model_of,
                .d_instances = bs.instances,
                .d_states = bs.states,
                .d_lim = bs.lim,
                .d_gath = bs.gath,
                .d_rhs_idx = bs.rhs_idx,
                .d_slots = bs.slots,
                .d_limw = bs.limw,
            };
            n_batches += 1;
        }
        const active = try gpa.alloc(u8, N);
        errdefer gpa.free(active);
        @memset(active, 0);
        const self = try gpa.create(Samples);
        errdefer gpa.destroy(self);
        self.* = .{
            .gpa = gpa,
            .points = ckts.len,
            .nt = times.len,
            .n = n,
            .nnz = nnz,
            .ground_slot = c0.diag_slots[0],
            .batches = batches,
            .active = active,
            .stream = try batches[0].eval.createStream(),
            .d_x = undefined,
            .d_f = undefined,
            .d_q = undefined,
            .d_g = undefined,
            .d_c = undefined,
            .d_sims = undefined,
            .d_active = undefined,
            .d_limw = undefined,
        };
        errdefer self.stream.deinit();
        const sizes = .{
            (n + 1) * N * @sizeOf(f64),   (n + 1) * N * @sizeOf(f64),   (n + 1) * N * @sizeOf(f64),
            (nnz + 1) * N * @sizeOf(f64), (nnz + 1) * N * @sizeOf(f64), N * @sizeOf(SimState),
            N,                            @as(usize, 4),
        };
        var made: usize = 0;
        errdefer inline for (buffers, 0..) |f, j| {
            if (j < made) @field(self, f).free();
        };
        inline for (buffers, 0..) |f, j| {
            @field(self, f) = try GpuLanes.zeros(&batches[0].eval, sizes[j]);
            made += 1;
        }
        const sims = try gpa.alloc(SimState, N);
        defer gpa.free(sims);
        for (sims, 0..) |*s, lane_id| s.* = .{
            .t = times[lane_id / ckts.len],
            .dt = dt,
            .kind = .tran,
            .iteration = ckts[lane_id % ckts.len].sim.iteration,
        };
        try self.d_sims.upload(sims.ptr, N * @sizeOf(SimState));
        try self.stream.synchronize();
        if (c0.knobs.gpu_stats != .off) std.debug.print("gpu-samples: {d} points, {d} sample lanes\n", .{ ckts.len, N });
        return self;
    }

    /// Drains the stream and frees everything; the point circuits are
    /// untouched.
    pub fn deinit(self: *Samples) void {
        if (comptime backend == null) return;
        self.stream.synchronize() catch {};
        inline for (buffers) |f| @field(self, f).free();
        for (self.batches) |*b| b.deinit();
        self.gpa.free(self.batches);
        self.gpa.free(self.active);
        self.stream.deinit();
        self.gpa.destroy(self);
    }

    /// Evaluate each active point's samples independently. Inactive points
    /// never run a model kernel and return zero planes. No sample's
    /// Instance/State is copied into the point circuit: the scalar oracle
    /// only evaluates, and samples start from the same fixed model state.
    pub fn evaluate(self: *Samples, x: []const f64, f: []f64, q: []f64, g: []f64, c: []f64, active: []const u8) !void {
        if (comptime backend == null) return Error.NoLaneImage;
        const N = self.points * self.nt;
        std.debug.assert(x.len == self.n * N and f.len == x.len and q.len == x.len);
        std.debug.assert(g.len == self.nnz * N and c.len == g.len);
        std.debug.assert(active.len == self.points);
        for (self.active, 0..) |*on, lane_id| on.* = @intFromBool(active[lane_id % self.points] != 0);
        try self.d_active.upload(self.active.ptr, N);
        try self.d_x.upload(x.ptr, x.len * @sizeOf(f64));
        try self.d_f.fillAsync(0, (self.n + 1) * N * @sizeOf(f64), &self.stream);
        try self.d_q.fillAsync(0, (self.n + 1) * N * @sizeOf(f64), &self.stream);
        try self.d_g.fillAsync(0, (self.nnz + 1) * N * @sizeOf(f64), &self.stream);
        try self.d_c.fillAsync(0, (self.nnz + 1) * N * @sizeOf(f64), &self.stream);
        for (self.batches) |*b| {
            if (b.count == 0) continue;
            var nl: u64 = N;
            var count: u64 = b.count;
            var none: u64 = 0;
            try b.eval.launchOn(&self.stream, gompute.Dim3.linear(@intCast(N), block), .{ .x = block }, 0, &.{
                GpuLanes.arg(&nl),     GpuLanes.arg(&count),   self.d_sims.argPtr(), self.d_active.argPtr(), self.d_limw.argPtr(),
                self.d_x.argPtr(),     b.d_gath.argPtr(),      b.d_rhs_idx.argPtr(), b.d_slots.argPtr(),     b.d_models.argPtr(),
                b.d_model_of.argPtr(), b.d_instances.argPtr(), self.d_g.argPtr(),    self.d_c.argPtr(),      self.d_f.argPtr(),
                self.d_q.argPtr(),     b.d_lim.argPtr(),       b.d_slots.argPtr(),   GpuLanes.arg(&none),    GpuLanes.arg(&none),
            });
        }
        try self.d_f.downloadAtAsync(f.ptr, 0, f.len * @sizeOf(f64), &self.stream);
        try self.d_q.downloadAtAsync(q.ptr, 0, q.len * @sizeOf(f64), &self.stream);
        try self.d_g.downloadAtAsync(g.ptr, 0, g.len * @sizeOf(f64), &self.stream);
        try self.d_c.downloadAtAsync(c.ptr, 0, c.len * @sizeOf(f64), &self.stream);
        try self.stream.synchronize();
        for (0..N) |lane_id| {
            if (self.active[lane_id] == 0) continue;
            f[lane_id] += x[lane_id];
            g[self.ground_slot * N + lane_id] += 1;
        }
    }
};

/// The Newton work of fixed-period shooting, resident for the full period.
/// The host keeps each lane's shooting iteration and period. No transient
/// predictor or LTE probe runs here: PSS accepts the charge from the last
/// Newton assembly and preserves its finite residual gate and gmin.
/// On a device error, host x and device instances remain unchanged so the
/// caller can rerun the period through the scalar integrator.
pub fn integrateFixed(gpa: std.mem.Allocator, lanes: []const FixedLane, opts: converger.Options, out: []bool) !void {
    if (comptime backend == null) return Error.NoLaneImage;
    std.debug.assert(lanes.len > 0 and out.len == lanes.len);
    const N = lanes.len;
    const n: usize = lanes[0].ckt.n;
    const steps = lanes[0].n_samples;
    const Seed = struct {
        n_hist: usize = 3,
        n_qt: usize = 0,
        dt: f64,
        cur: []f64,
        q_ring: [3][]f64,
        i_prev: []f64,
        qt_ring: [3][]const f64 = @splat(&.{}),
        qt_i_prev: []const f64 = &.{},
    };
    const ckts = try gpa.alloc(*Circuit, N);
    defer gpa.free(ckts);
    const seeds = try gpa.alloc(Seed, N);
    defer gpa.free(seeds);
    for (lanes, ckts, seeds) |l, *c, *s| {
        std.debug.assert(l.ckt.n == n and l.n_samples == steps and !l.ckt.has_state_q);
        c.* = l.ckt;
        s.* = .{ .dt = l.period / @as(f64, @floatFromInt(steps)), .cur = l.x, .q_ring = @splat(l.q_prev), .i_prev = l.i_prev };
        if (!c.*.has_charge) c.*.eval(l.x, 0);
    }
    const dev = try GpuLanes.init(gpa, ckts, seeds, &.{});
    defer dev.deinit();
    @memset(dev.detached, true);
    dev.prof.on = lanes[0].ckt.knobs.gpu_stats != .off;
    if (dev.prof.on) std.debug.print("gpu-pss: {d} lanes, {d} fixed steps\n", .{ N, steps });
    const host = try gpa.alloc(f64, (n + 1) * N);
    defer gpa.free(host);
    const nr = try gpa.alloc(LaneNewton, N);
    defer gpa.free(nr);
    @memset(host, 0);
    for (lanes, 0..) |l, k| for (l.x[0..n], 0..) |x, i| {
        host[i * N + k] = x;
    };
    try dev.d_x.upload(host.ptr, host.len * @sizeOf(f64));
    @memset(dev.round, 1);
    @memset(out, true);
    const records = for (lanes) |l| {
        if (l.wave.len != 0) break true;
    } else false;
    for (0..steps) |step| {
        if (std.mem.indexOfScalar(u8, dev.round, 1) == null) break;
        for (lanes, dev.round, 0..) |l, on, k| {
            if (on == 0) continue;
            try l.ckt.checkpoint(.{ .phase = .periodic, .completed = step, .total = steps });
            const dt = l.period / @as(f64, @floatFromInt(steps));
            const t = l.period * @as(f64, @floatFromInt(step + 1)) / @as(f64, @floatFromInt(steps));
            l.ckt.setSimState(.{ .t = t, .dt = dt, .kind = .tran });
            dev.sims[k] = l.ckt.sim;
            dev.sc[k] = .{ .t_dev = t, .dt = dt, .xfact = 0, .ag0 = 2.0 / dt, .ag1 = 1, .ag2 = 0, .method = 1, .flags = 1, .sq = @splat(0), .sqt = @splat(0) };
        }
        try dev.put(&dev.d_sc, dev.sc);
        dev.prof.rounds += 1;
        try dev.newton(opts, nr);
        for (dev.round, nr, out, 0..) |*on, r, *ok, k| {
            if (on.* == 0) continue;
            if (r.reject) return error.FixedStateRejected;
            ok.* = r.result.converged and !r.singular;
            if (!ok.*) {
                dev.ops[k] = 1 + @as(u8, @backingInt(StateCtlOp.revert));
                on.* = 0;
                continue;
            }
            dev.ops[k] = 1 + @as(u8, @backingInt(StateCtlOp.commit));
            dev.accept_[k] = 1;
            dev.queueAccept(k);
        }
        try dev.apply();
        if (records) {
            try dev.d_x.download(host.ptr, host.len * @sizeOf(f64));
            for (lanes, out, 0..) |l, ok, k| {
                if (!ok or l.wave.len == 0) continue;
                const cols = 1 + l.probes.len;
                const row = l.wave[(step + 1) * cols ..][0..cols];
                row[0] = dev.sc[k].t_dev;
                for (l.probes, 0..) |p, j| row[j + 1] = host[@as(usize, p) * N + k];
            }
        }
    }
    try dev.d_x.download(host.ptr, host.len * @sizeOf(f64));
    try dev.d_sims.download(dev.sims.ptr, N * @sizeOf(SimState));
    // Stage every accepted payload before changing its host copy. The
    // destructor's best-effort transfer cannot make fixed-period acceptance
    // transactional: a late allocation or download failure must rerun the
    // period with the original host Instance/State rows.
    const Payload = struct { instances: []u8 = &.{}, states: []u8 = &.{} };
    const payloads = try gpa.alloc(Payload, dev.batches.len);
    @memset(payloads, .{});
    defer {
        for (payloads) |p| {
            gpa.free(p.instances);
            gpa.free(p.states);
        }
        gpa.free(payloads);
    }
    for (dev.batches, payloads) |*b, *p| {
        p.instances = try gpa.alloc(u8, b.count * N * b.inst_size);
        p.states = try gpa.alloc(u8, b.count * N * b.state_size);
        if (p.instances.len != 0) try b.d_instances.download(p.instances.ptr, p.instances.len);
        if (p.states.len != 0) try b.d_states.download(p.states.ptr, p.states.len);
    }
    for (dev.batches, payloads, 0..) |b, p, bi| for (lanes, 0..) |l, k| {
        const hb = l.ckt.batches[bi];
        const dst = hb.hooks.gpu_payload.?(hb.ctx);
        for (0..b.count) |id| {
            const is = b.inst_size;
            const ss = b.state_size;
            @memcpy(@constCast(dst.instances)[id * is ..][0..is], p.instances[(id * N + k) * is ..][0..is]);
            @memcpy(@constCast(dst.states)[id * ss ..][0..ss], p.states[(id * N + k) * ss ..][0..ss]);
        }
    };
    for (lanes, 0..) |l, k| for (l.x[0..n], 0..) |*x, i| {
        x.* = host[i * N + k];
    };
    for (lanes, dev.sims) |l, sim| {
        l.ckt.state_staged = false;
        // setSimState preserves iteration, so explicitly publish the final
        // device value for the next host seam or directional integration.
        l.ckt.sim.iteration = sim.iteration;
        l.ckt.setSimState(l.ckt.sim);
    }
    // Host publication is complete; keep the destructor from copying again.
}

/// `tran.Sim`'s numerics backend for lane l of a `GpuLanes` round (the
/// phase 4 G1 surface): the device did the numerics for every lane at once
/// (`GpuLanes.probe`); this answers `Sim.end`'s calls from those numbers
/// and queues lane l's mutations for `GpuLanes.apply`.
pub const Ops = struct {
    g: *GpuLanes,
    l: u32,

    fn methodCode(m: anytype) u32 {
        return switch (m) {
            .backward_euler => 0,
            .trapezoidal => 1,
            .gear_2 => 2,
            else => unreachable, // gear-3..6 are not lane-eligible
        };
    }

    /// Queues lane l into the next round with the step `a` that `Sim` `s`
    /// attempts: its scalars, `SimState` and LTE inputs. Gear-3..6 is
    /// unreachable.
    pub fn attempt(o: Ops, s: anytype, a: anytype) void {
        const g = o.g;
        const promote = methodCode(s.promoteMethod());
        g.sc[o.l] = .{
            .t_dev = a.t_dev,
            .dt = s.dt,
            .xfact = s.dt / s.dt_old[0],
            .ag0 = a.c.ag0,
            .ag1 = a.c.ag1,
            .ag2 = a.c.ag2,
            .method = methodCode(a.method),
            .flags = 1 | (promote << 4),
            .sq = @splat(0),
            .sqt = @splat(0),
        };
        g.sims[o.l] = s.ckt.sim;
        g.lte_in[o.l] = .{ .dt = s.dt, .dt1 = s.dt_old[0], .dt2 = s.dt_old[1], .reltol = s.options.tol.reltol, .abstol = s.options.tol.abstol, .chgtol = s.options.tol.chgtol, .trtol = s.options.tol.trtol, .ag0 = a.c.ag0, .ag1 = a.c.ag1, .ag2 = a.c.ag2 };
        g.round[o.l] = 1;
    }
    /// Queues the latch revert of a rejected step for `GpuLanes.apply`.
    pub fn revert(o: Ops, _: anytype) void {
        o.g.ops[o.l] = 1 + @as(u8, @backingInt(StateCtlOp.revert));
    }
    /// Always null: a reject request fails the lane instead
    /// (`tran.runGpuLanes`).
    pub fn rejectAt(_: Ops, _: anytype) ?f64 {
        return null;
    }
    /// The device state-flip query `probe` ran, when `ask`.
    pub fn flipped(o: Ops, _: anytype, _: anytype, ask: bool) bool {
        return ask and o.g.query[o.l] & 1 != 0;
    }
    /// Always null: a finite bound hands the lane back (`LaneNewton.reject`).
    pub fn boundStep(_: Ops, _: anytype) ?f64 {
        return null;
    }
    /// A no-op: `probe` evaluated the round's charges.
    pub fn charges(_: Ops, _: anytype, _: anytype) void {}
    /// The LTE bound `probe` computed for `method`.
    pub fn bound(o: Ops, _: anytype, a: anytype, method: anytype, _: anytype) f64 {
        const out = o.g.lte_out[o.l];
        // `end` asks for the step's own method, or the order-2 promotion
        // probe (the only other order <= 2 call).
        return if (method == a.method) out[0] else out[1];
    }
    /// Unreachable: gear-3..6 are not lane-eligible.
    pub fn boundGear(_: Ops, _: anytype, _: anytype, _: anytype, _: anytype) f64 {
        unreachable;
    }
    /// Rotates lane l's history permutations (`queueAccept`).
    pub fn advance(o: Ops, _: anytype, _: anytype) void {
        o.g.queueAccept(o.l);
    }
    /// Every accepted step: the commit, and `accept`'s `prev <- cur <- x`.
    pub fn commit(o: Ops, _: anytype) void {
        o.g.ops[o.l] = 1 + @as(u8, @backingInt(StateCtlOp.commit));
        o.g.accept_[o.l] = 1;
    }
    /// Records the tape's logical slot 0 for `rebaseRound` and swaps the
    /// tape's slots 0 and 1, as `std.mem.swap` moves the host ring.
    pub fn rebase(o: Ops, s: anytype, _: anytype) void {
        if (!(s.has_charge and s.ckt.has_state_q)) return;
        const nh: usize = o.g.n_hist;
        const p = o.g.perm_qt[@as(usize, o.l) * nh ..][0..2];
        o.g.sc[o.l].sqt[3] = p[0];
        std.mem.swap(u8, &p[0], &p[1]);
    }
    /// Lane l's n-long row: only its probe positions hold the solution.
    pub fn accepted(o: Ops, _: anytype) []const f64 {
        const n: usize = o.g.n;
        return o.g.rows[@as(usize, o.l) * n ..][0..n];
    }
};
