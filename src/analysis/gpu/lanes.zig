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
//! to the host at `deinit`. The transforms live beside this file:
//! lanes_init.zig (upload, copy-back), lanes_static.zig (operating and `.dc` points), lanes_tick.zig (one
//! Newton tick's kernels), lanes_newton.zig (Newton's tick loop, graphs,
//! peels and retapes), lanes_round.zig (host-stepped probe and apply),
//! lanes_step.zig (device step control), lanes_samples.zig, lanes_fixed.zig,
//! lanes_ops.zig, lanes_batch.zig (one device batch).
//! Cites: phase4-gpu-lanes.md G1, G6, 06 6.8, 6.9, 6.11, 6.12.
//!
//! Contents:
//! - `pub const available: bool`
//! - `pub const test_access` (`admitted`, `force_color`, `static`, `host_meas`, `dev_meas`)
//! - `pub const Error = error{ NoLaneImage, NoLaneTape, IncompatibleLanes }`
//! - `pub const Prof = struct` (`Phase`, `TickPhase`, `now`, `lap`)
//! - `pub const LaneNewton = struct { result, singular, reject }`
//! - `pub const FixedLane = struct`
//! - `pub const Dev = enum(u8)`, `pub const dev_count`
//! - `pub const GpuLanes = struct` (`pinned_mirrors`, `max_rounds`, `buf`, `put`, `get`, `launch`, and the entry
//!   points `init`, `initStatic`, `deinit`, `copyBack`, `reload`, `solveStatic`, `laneOps`, `predictRound`, `newton`,
//!   `probe`, `apply`, `initSteps`, `initMeasures`, `measResolved`, `fetchMeasures`, `lanePasses`, `multiRound`,
//!   `deviceLaunch`, `slotReport`, `slotRow`, `fetchSteps`, `requestStop`)
//! - `pub const Samples = lanes_samples.Samples`
//! - `pub const Ops = lanes_ops.Ops`
//! - `pub const integrateFixed = lanes_fixed.integrateFixed`
//! - `pub fn policy(config: anytype) bool`
//! - `pub fn share(cpus: usize) f64`
//! - `pub fn staticPolicy(config: anytype, cpus: usize) bool`
//! - `pub fn staticRefusal(ckt: *const Circuit) ?[]const u8`

const std = @import("std");
const assert = std.debug.assert;
const gompute = @import("gompute");
const device_ir = @import("device").abi;
const solver = @import("solver");
const root = @import("../types.zig");
const passes = @import("measure").passes;
const devices = @import("devices.zig");
const lanes_batch = @import("lanes_batch.zig");
const lanes_fixed = @import("lanes_fixed.zig");
const lanes_init = @import("lanes_init.zig");
const lanes_newton = @import("lanes_newton.zig");
const lanes_ops = @import("lanes_ops.zig");
const lanes_round = @import("lanes_round.zig");
const lanes_samples = @import("lanes_samples.zig");
const lanes_static = @import("lanes_static.zig");
const lanes_step = @import("lanes_step.zig");
const lanes_tick = @import("lanes_tick.zig");

const lk = solver.lane_kernels;
const converger = solver.converger;
const Circuit = root.Circuit;
const SimState = device_ir.SimState;
const StepReport = lk.StepReport;
const backend = devices.backend;
const Raw = lanes_batch.Raw;
const Buffer = lanes_batch.Buffer;
const Stream = lanes_batch.Stream;
const GraphExec = lanes_batch.GraphExec;
const LaneBatch = lanes_batch.LaneBatch;
const Kernels = lanes_batch.Kernels;
const block = lanes_batch.block;

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

/// Why `GpuLanes.init` refused before any driver error.
pub const Error = error{
    /// No kernel image for the lane kernels, or a batch without one.
    NoLaneImage,
    /// The lead lane holds no factored sparse LU to replay.
    NoLaneTape,
    /// Points do not share the sparse pattern and device scatter tapes.
    IncompatibleLanes,
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
    /// The `tick_ns` slot `tickLap` charges under `phases`.
    pub const TickPhase = enum(u3) { eval, assemble, lu, gates, state };

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

/// The device buffers of a `GpuLanes`, one per row of `bufs`. Lane-major:
/// entry i of lane l at `i * N + l`. x, x_old, dx, cur, prev, rhs, q hold
/// n + 1 rows, g, c, a nnz + 1 slots (rhs, q, g, c plus staging tails for
/// coloured batches). The LTE tape ring holds entry j of physical slot k at
/// `(k * n_qt + j) * N + l`, `ip_qt` entry j at `j * N + l`. Then per-lane
/// scalars and flags; the tables (tape, CSC, current rows, diagonal,
/// probes); the device words (decide: lanes whose Newton ended this round;
/// first: the lowest lane the dense peel took, all ones for none; over:
/// failed lanes past the dense pool); the dense peel's slots of n * n + n
/// entries; the host peel's packed lanes (indices, then A, rhs, dx); device
/// step control (each lane's `LaneStep`, breakpoint request and answer,
/// report, stops); a launch's ring of `max_rounds` reports and probe rows,
/// its slot word and the dense peel's launch-wide count and lowest failing
/// lane; the device `.meas` passes (columns, every (pass, lane) state lane
/// major, each lane's resolving ring slot).
pub const Dev = enum(u8) {
    x,
    x_old,
    dx,
    cur,
    prev,
    g,
    c,
    rhs,
    q,
    a,
    lx,
    ux,
    udiag,
    w,
    y,
    ring,
    ip,
    ring_qt,
    ip_qt,
    sc,
    sims,
    active,
    round,
    stage,
    gate,
    flags,
    fail,
    ops,
    accept,
    slot0,
    perm_q,
    perm_qt,
    lte_in,
    lte_out,
    probe_out,
    tab_idx,
    cur_row,
    diag,
    probes,
    decide,
    count,
    fails,
    first,
    over,
    slot_next,
    col_ptr,
    row_idx,
    dense,
    peel_idx,
    peel,
    step,
    brk_t,
    best,
    report,
    stop,
    ring_report,
    ring_rows,
    ring_slot,
    seen,
    seen_lo,
    meas_cols,
    meas,
    meas_resolved,
};

/// Rows of `GpuLanes.bufs`.
pub const dev_count = @typeInfo(Dev).@"enum".field_names.len;

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

    /// Every device buffer, by `Dev` (`buf`); `init` sizes them once
    /// (`lanes_init.zig`), and a `retape`, a peel or `initMeasures` may
    /// replace one in place.
    bufs: [dev_count]Buffer,
    /// One page-locked block behind every pinned mirror below
    /// (`pinned_mirrors`), carved at `init` and freed at `deinit`.
    pin_block: []u8,

    // Host mirrors of what moves each round, page-locked (`pinned`) so the
    // copies queue on `stream` instead of staging through a fenced copy.
    sc: []lk.LaneScalars = &.{},
    sims: []SimState = &.{},
    lte_in: []lk.LteIn = &.{},
    lte_out: [][2]f64 = &.{},
    gate: []lk.Gate = &.{},
    query: []u32 = &.{},
    ops: []u8 = &.{},
    accept_: []u8 = &.{},
    round: []u8 = &.{},
    perm_q: []u8 = &.{},
    perm_qt: []u8 = &.{},
    slot0: []u8 = &.{},
    fail: []u8 = &.{},
    probe_out: []f64 = &.{},
    /// Device step control: the host's stop requests for the next launch,
    /// and the last launch's ring of reports and probe rows.
    stop: []u8 = &.{},
    ring_report: []StepReport = &.{},
    ring_rows: []f64 = &.{},
    /// Each lane's resolving ring slot of the last launch (0xff: none yet).
    meas_resolved: []u8 = &.{},
    /// The device passes as the last `fetchMeasures` found them, lane-major.
    meas_host: []passes.Dev = &.{},
    /// Device `.meas` passes per lane (0: the host measures the rows).
    n_meas: u32 = 0,
    meas_t_start: f64 = 0,
    /// The per-tick words, pinned so their downloads stay asynchronous
    /// instead of staging through pageable memory: fails, count, first,
    /// over.
    words: []u32 = &.{},
    probes: []const u32,
    /// Lanes that left the device for a CPU rerun: `deinit` copies no
    /// state back into them.
    detached: []bool = &.{},
    /// Per lane, an n-long host vector whose probe rows hold the last
    /// probe's x (`Ops.accepted`).
    rows: []f64 = &.{},
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

    /// The pinned mirror fields, in their order in `pin_block`.
    pub const pinned_mirrors = .{ "sc", "sims", "lte_in", "lte_out", "gate", "query", "ops", "accept_", "round", "perm_q", "perm_qt", "slot0", "fail", "probe_out", "words", "stop", "ring_report", "ring_rows", "meas_resolved" };

    /// Most rounds one `deviceLaunch` runs; the ring holds this many.
    pub const max_rounds = 16;

    // The entry points, one transform per sibling file.
    pub const init = lanes_init.init;
    pub const initStatic = lanes_static.initStatic;
    pub const deinit = lanes_init.deinit;
    pub const copyBack = lanes_init.copyBack;
    pub const reload = lanes_static.reload;
    pub const solveStatic = lanes_static.solveStatic;
    pub const laneOps = lanes_tick.laneOps;
    pub const predictRound = lanes_tick.predictRound;
    pub const newton = lanes_newton.newton;
    pub const probe = lanes_round.probe;
    pub const apply = lanes_round.apply;
    pub const initSteps = lanes_step.initSteps;
    pub const initMeasures = lanes_step.initMeasures;
    pub const measResolved = lanes_step.measResolved;
    pub const fetchMeasures = lanes_step.fetchMeasures;
    pub const lanePasses = lanes_step.lanePasses;
    pub const multiRound = lanes_step.multiRound;
    pub const deviceLaunch = lanes_step.deviceLaunch;
    pub const slotReport = lanes_step.slotReport;
    pub const slotRow = lanes_step.slotRow;
    pub const fetchSteps = lanes_step.fetchSteps;
    pub const requestStop = lanes_step.requestStop;

    /// Device buffer `id`. Valid until a call that replaces it (`retape`,
    /// `ensurePeel`, `initMeasures`).
    pub fn buf(self: *Self, id: Dev) *Buffer {
        return &self.bufs[@backingInt(id)];
    }

    /// `src` into `dst` on the lane stream. `src` must be pinned and left
    /// alone until the stream next synchronizes.
    pub fn put(self: *Self, dst: *Buffer, src: anytype) !void {
        const bytes = std.mem.sliceAsBytes(src);
        try dst.uploadAtAsync(bytes.ptr, 0, bytes.len, &self.stream);
    }

    /// `src` into pinned `dst` on the lane stream; valid after the next
    /// synchronize.
    pub fn get(self: *Self, dst: anytype, src: *Buffer) !void {
        const bytes = std.mem.sliceAsBytes(dst);
        try src.downloadAtAsync(bytes.ptr, 0, bytes.len, &self.stream);
    }

    /// Launches `k` over `threads` threads at the lane block width; none
    /// for zero threads (an empty grid is not a launch).
    pub fn launch(self: *Self, k: *Raw, threads: usize, args: []const gompute.interface.Arg) !void {
        if (threads == 0) return;
        try k.launchOn(&self.stream, gompute.Dim3.linear(@intCast(threads), block), .{ .x = block }, 0, args);
    }
};

/// Sampled periodic device evaluation (`lanes_samples.zig`).
pub const Samples = lanes_samples.Samples;
/// `tran.Sim`'s numerics backend for one lane (`lanes_ops.zig`).
pub const Ops = lanes_ops.Ops;
/// One fixed trapezoidal period per lane on the device (`lanes_fixed.zig`).
pub const integrateFixed = lanes_fixed.integrateFixed;

comptime {
    assert(@sizeOf(lk.SimState) == @sizeOf(SimState));
    assert(@offsetOf(lk.SimState, "iteration") == @offsetOf(SimState, "iteration"));
    assert(dev_count <= std.math.maxInt(u8));
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
        .auto => available and (lanes_batch.fp64Ratio() orelse return false) <= 4,
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
    const ratio = lanes_batch.fp64Ratio() orelse return false;
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
