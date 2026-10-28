//! A GPU lane run's lifetime. In: N lane circuits on one pattern, each after
//! its `tran.Sim.init` (or, for operating points and `.dc` points, at its
//! start x, `lanes_static.zig`). Out: the resident `GpuLanes`: the pivot tape of lane 0 (or of
//! the split batch's lane 0, `devices.TapeShare`), every batch packed and
//! uploaded, every device buffer sized once from the lane count and the
//! pattern (`Dev`), the pinned mirrors carved from one page-locked block,
//! and each lane's start state; at `deinit`, the lanes' Instance and State
//! rows copied back to their host batches. A failure at any point frees
//! everything made so far.
//! Cites: phase4-gpu-lanes.md G6, 06 6.9 (static points).
//!
//! Contents:
//! - `pub fn init(gpa: std.mem.Allocator, ckts: []const *Circuit, sims: anytype, probes: []const u32) !*GpuLanes`
//! - `pub fn deinit(self: *GpuLanes) void`
//! - `pub fn copyBack(self: *GpuLanes) void`

const std = @import("std");
const assert = std.debug.assert;
const device_ir = @import("device").abi;
const solver = @import("solver");
const root = @import("../types.zig");
const devices = @import("devices.zig");
const lane_color = @import("lane_color.zig");
const lanes = @import("lanes.zig");
const lanes_batch = @import("lanes_batch.zig");
const lanes_newton = @import("lanes_newton.zig");
const lanes_round = @import("lanes_round.zig");
const lanes_tick = @import("lanes_tick.zig");

const lk = solver.lane_kernels;
const direct = solver.direct;
const converger = solver.converger;
const Circuit = root.Circuit;
const SimState = device_ir.SimState;
const StateCtlOp = device_ir.StateCtlOp;
const StepReport = lk.StepReport;
const GpuLanes = lanes.GpuLanes;
const LaneNewton = lanes.LaneNewton;
const Dev = lanes.Dev;
const Error = lanes.Error;
const test_access = lanes.test_access;
const Raw = lanes_batch.Raw;
const Buffer = lanes_batch.Buffer;
const LaneBatch = lanes_batch.LaneBatch;
const Kernels = lanes_batch.Kernels;
const backend = devices.backend;
const available = lanes.available;
const max_rounds = GpuLanes.max_rounds;

/// Bytes of the device dense peel's pool.
// ponytail: a fixed 256 MB pool; size it from the device's free memory if
// large lane counts ever crowd it.
const dense_pool_bytes: usize = 256 << 20;

/// Alignment of each pinned mirror within `pin_block`.
const mirror_align = 64;

/// The sizes every device buffer and pinned mirror is cut to.
const Shape = struct {
    N: usize,
    n: usize,
    nnz: usize,
    n_hist: usize,
    n_qt: usize,
    n_probes: usize,
    lx_cap: usize,
    ux_cap: usize,
    stage_slots: usize,
    stage_rows: usize,
    dense_slots: usize,
};

/// Staging cells past the planes, for the hub touches of coloured batches.
const Staging = struct { slots: u32, rows: u32 };

/// The tables `init` uploads as they are.
const Tables = struct {
    tab_idx: []const u32,
    col_ptr: []const u32,
    row_idx: []const u32,
    cur_row: []const u8,
    diag: []const u32,
    probes: []const u32,
};

/// Uploads every lane after its `Sim.init`: the pattern's tapes, each
/// batch's packed rows, x_op, the charge rings and histories. `sims[l]` is
/// lane l's step machine (`tran.Sim`), read for its rings. Fails with
/// `Error` or a driver or allocation error, with nothing left allocated;
/// the caller then runs the lanes on the CPU. Caller owns the result
/// (`deinit`).
pub fn init(gpa: std.mem.Allocator, ckts: []const *Circuit, sims: anytype, probes: []const u32) !*GpuLanes {
    if (comptime backend == null) return Error.NoLaneImage;
    if (ckts.len == 0 or ckts.len > std.math.maxInt(u32)) return Error.IncompatibleLanes;
    assert(sims.len == ckts.len);
    const c0 = ckts[0];
    const N: usize = ckts.len;
    const n: usize = c0.n;
    const nnz: usize = c0.nnz;
    const s0 = sims[0];
    for (ckts) |c| {
        if (!lanes_batch.samePattern(c, c0)) return Error.IncompatibleLanes;
        for (c.batches, c0.batches) |b, b0| {
            const payload = b.hooks.gpu_payload orelse return Error.NoLaneImage;
            const payload0 = b0.hooks.gpu_payload orelse return Error.NoLaneImage;
            if (!lanes_batch.sameRows(payload(b.ctx), payload0(b0.ctx))) return Error.IncompatibleLanes;
        }
    }

    // The LU tape: lane 0's pivot order on its first step's matrix, G +
    // C/dt at x_op as `Sim.init` left the planes (G alone without charge).
    // A slice of a batch split across devices (`devices.tape`) replays the
    // tape of the batch's lane 0 instead of its own lane 0's.
    const shared = if (devices.tape) |ts| (if (devices.tape_builds) null else try ts.take(gpa)) else null;
    const built: devices.Tape = shared orelse ownTape(gpa, c0, s0.dt, s0.n_hist) catch |e| {
        if (devices.tape) |ts| ts.fail();
        return e;
    };
    defer gpa.free(built.idx);
    if (shared == null) if (devices.tape) |ts| ts.publish(built.tab, built.idx);
    const t = built.tab;

    const self = try gpa.create(GpuLanes);
    errdefer gpa.destroy(self);
    var k = try Kernels.init();
    errdefer k.deinit();
    const k0 = &k.refactor;
    var stream = try k0.createStream();
    errdefer stream.deinit();

    const batches = try gpa.alloc(LaneBatch, c0.batches.len);
    errdefer gpa.free(batches);
    var n_up: usize = 0;
    errdefer for (batches[0..n_up]) |*b| b.deinit(gpa);
    const n_qt: usize = s0.n_qt;
    const ps = try gpa.alloc(device_ir.GpuPayload, N);
    defer gpa.free(ps);
    var tape_off: u32 = 0;
    for (batches, c0.batches, 0..) |*b, b0, bi| {
        for (ps, ckts) |*p, c| p.* = c.batches[bi].hooks.gpu_payload.?(c.batches[bi].ctx);
        b.* = try lanes_batch.loadBatch(gpa, k0, ps, @ptrCast(b0.hooks.gpu_payload.?), b0.has_charge, tape_off);
        n_up += 1;
        tape_off += if (b0.hooks.q_tape) |f| @intCast(f(b0.ctx).len) else 0;
    }
    assert(n_up == batches.len);
    assert(tape_off == n_qt or n_qt == 0);
    const stage = try colorBatches(gpa, k0, batches, c0, N);

    const slot_bytes = (n * n + n) * @sizeOf(f64);
    const shape: Shape = .{
        .N = N,
        .n = n,
        .nnz = nnz,
        .n_hist = s0.n_hist,
        .n_qt = n_qt,
        .n_probes = probes.len,
        .lx_cap = @max(t.nnz_l, 1),
        .ux_cap = @max(t.nnz_u, 1),
        .stage_slots = stage.slots,
        .stage_rows = stage.rows,
        .dense_slots = if (n <= lk.dense_peel_max_n) @min(N, dense_pool_bytes / slot_bytes) else 0,
    };
    const cur_row = try gpa.alloc(u8, n);
    defer gpa.free(cur_row);
    for (cur_row, c0.current_row[0..n]) |*o, on| o.* = @intFromBool(on);
    const tables: Tables = .{
        .tab_idx = built.idx,
        .col_ptr = c0.col_ptr[0 .. n + 1],
        .row_idx = c0.row_idx[0..nnz],
        .cur_row = cur_row,
        .diag = c0.diag_slots[0..n],
        .probes = probes,
    };
    self.* = .{
        .gpa = gpa,
        .ckts = ckts,
        .n_lanes = @intCast(N),
        .n = @intCast(n),
        .nnz = @intCast(nnz),
        .n_hist = @intCast(shape.n_hist),
        .n_qt = @intCast(n_qt),
        .n_probes = @intCast(probes.len),
        // A static solve (`initStatic`, no history) reads no charge.
        .has_charge = c0.has_charge and shape.n_hist != 0,
        .state_q = c0.has_charge and shape.n_hist != 0 and c0.has_state_q,
        .d0 = c0.diag_slots[0],
        .batches = batches,
        .k = k,
        .stream = stream,
        .tab = t,
        .bufs = undefined,
        .pin_block = &.{},
        .probes = probes,
        .prof = .{ .phases = c0.knobs.gpu_stats == .phases },
        .lx_cap = @intCast(shape.lx_cap),
        .ux_cap = @intCast(shape.ux_cap),
        .dense_slots = @intCast(shape.dense_slots),
        .stage_slots = @intCast(shape.stage_slots),
        .stage_rows = @intCast(shape.stage_rows),
    };
    try allocBuffers(self, k0, shape, tables);
    errdefer for (&self.bufs) |*b| b.free();
    try carveMirrors(self, k0, shape);
    errdefer k0.freePinned(self.pin_block);
    self.rows = try gpa.alloc(f64, N * n);
    errdefer gpa.free(self.rows);
    self.detached = try gpa.alloc(bool, N);
    errdefer gpa.free(self.detached);
    @memset(self.rows, 0);
    @memset(self.detached, false);
    try seedLanes(self, sims);
    if (@import("builtin").is_test) _ = test_access.admitted.fetchAdd(N, .monotonic);
    return self;
}

/// Copies each lane's device Instance rows back into its host batch (the
/// latches `refuseLatched` reads), prints the `ESPICE_GPU_STATS` lines,
/// then frees everything and `self`.
pub fn deinit(self: *GpuLanes) void {
    if (comptime backend == null) return;
    const gpa = self.gpa;
    // A fault here was already reported or is moot.
    self.stream.synchronize() catch {};
    printProf(self);
    lanes_newton.clearGraphs(self);
    copyBack(self);
    for (self.batches) |*b| b.deinit(gpa);
    gpa.free(self.batches);
    for (&self.bufs) |*b| b.free();
    self.k.refactor.freePinned(self.pin_block);
    gpa.free(self.rows);
    gpa.free(self.detached);
    gpa.free(self.meas_host);
    self.stream.deinit();
    self.k.deinit();
    gpa.destroy(self);
}

/// Copies each attached lane's device Instance and State rows back into its
/// host batch. Best effort: a failed allocation or download leaves that
/// batch's host rows as they were. Call with the stream idle.
pub fn copyBack(self: *GpuLanes) void {
    if (comptime !available) unreachable; // no batch is ever built without GPU images
    const gpa = self.gpa;
    const N: usize = self.n_lanes;
    assert(self.detached.len == N);
    for (self.batches, 0..) |*b, bi| {
        const is = b.inst_size;
        if (b.count == 0 or is == 0) continue;
        const rows = gpa.alloc(u8, b.count * N * is) catch continue;
        defer gpa.free(rows);
        b.d_instances.download(rows.ptr, rows.len) catch continue;
        for (self.ckts, 0..) |c, l| {
            if (self.detached[l]) continue;
            const hb = c.batches[bi];
            const dst = @constCast(hb.hooks.gpu_payload.?(hb.ctx).instances);
            for (0..b.count) |id| @memcpy(dst[id * is ..][0..is], rows[(id * N + l) * is ..][0..is]);
        }
        const ss = b.state_size;
        if (ss == 0) continue;
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

// ---- private ----

/// `init`'s pivot tape from `c0`'s first step matrix (`dt` as `Sim.init`
/// left it), factored on `c0`'s host solver. Caller owns `idx`.
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
    assert(built.tab.n == c0.n);
    return .{ .idx = built.idx, .tab = built.tab };
}

/// Colours every batch while `N` lanes underfill the card (`colorPolicy`),
/// each batch's staging cells after the last batch's. Returns the staging
/// cells used past the planes' `nnz + 1` slots and `n + 1` rows. On error
/// the batches keep the colours they took; `LaneBatch.deinit` frees them.
fn colorBatches(gpa: std.mem.Allocator, k0: *Raw, batches: []LaneBatch, c0: *const Circuit, N: usize) !Staging {
    var stage: Staging = .{ .slots = 0, .rows = 0 };
    if (!colorPolicy(N)) return stage;
    const force = @import("builtin").is_test and test_access.force_color;
    for (batches, c0.batches) |*b, hb| {
        if (b.count == 0) continue;
        const p0 = hb.hooks.gpu_payload.?(hb.ctx);
        const t: lane_color.Tables = .{
            .count = b.count,
            .ks = @intCast(p0.slots.len / b.count),
            .slots = p0.slots,
            .n_u = p0.n_u,
            .rows = p0.rhs_idx,
            .trash_slot = c0.nnz,
            .ground_row = c0.n,
            .stage_slot0 = c0.nnz + 1 + stage.slots,
            .stage_row0 = c0.n + 1 + stage.rows,
        };
        const col = try lanes_batch.colorBatch(gpa, k0, b, t, force) orelse continue;
        defer col.deinit(gpa);
        stage.slots += col.n_slot_stage;
        stage.rows += col.n_row_stage;
        if (c0.knobs.gpu_stats != .off) std.debug.print("gpu-lanes: {s}: {d} instances in {d} colours, {d} + {d} hub cells\n", .{ hb.type_name, b.count, col.colors(), col.slot_hubs.cells.len, col.row_hubs.cells.len });
    }
    return stage;
}

/// Whether `init` evaluates a thread per (instance colour, lane)
/// (`lane_color.zig`): while N lanes underfill the card. Measured on the RTX
/// 4060 Laptop (24 SMs) with 80 mos6 in 9 colours: Newton -29 % at 512
/// lanes, +13 % at 1,335, +20 % at 4,096, +40 % at 8,665, so the crossover
/// is near 40 lanes per SM: the bar is 40 x SMs (960 here, 4,000-5,300 on
/// a 100-132 SM part), 1,024 when the driver does not say.
// ponytail: one card and one model measured; refit the 40 on another.
fn colorPolicy(n_lanes: usize) bool {
    if (@import("builtin").is_test and test_access.force_color) return true;
    const sms = lanes_batch.multiprocessorCount() orelse return n_lanes < 1024;
    return n_lanes < 40 * @as(usize, sms);
}

/// Allocates every row of `self.bufs` for `s`, uploading `t`'s tables and
/// zeroing the rest the kernels read before writing. On error none stays
/// allocated.
fn allocBuffers(self: *GpuLanes, k0: *Raw, s: Shape, t: Tables) !void {
    var made: usize = 0;
    errdefer for (self.bufs[0..made]) |*b| b.free();
    for (std.enums.values(Dev), 0..) |id, i| {
        assert(@backingInt(id) == i);
        self.bufs[i] = try makeBuffer(k0, id, s, t);
        made += 1;
    }
    assert(made == lanes.dev_count);
}

/// Device buffer `id` for shape `s`: zeroed (`zeros`), uninitialized
/// (written before any kernel reads it), staged (-0.0 tails) or uploaded.
fn makeBuffer(k0: *Raw, id: Dev, s: Shape, t: Tables) !Buffer {
    const f8 = @sizeOf(f64);
    const N = s.N;
    const z = lanes_batch.zeros;
    const up = lanes_batch.upload;
    const bytes = std.mem.sliceAsBytes;
    return switch (id) {
        .x, .x_old, .dx, .cur, .prev => z(k0, (s.n + 1) * N * f8),
        .g, .c => lanes_batch.staged(k0, s.nnz + 1, s.stage_slots, N),
        .rhs, .q => lanes_batch.staged(k0, s.n + 1, s.stage_rows, N),
        .a => z(k0, (s.nnz + 1) * N * f8),
        .lx => z(k0, s.lx_cap * N * f8),
        .ux => z(k0, s.ux_cap * N * f8),
        .udiag, .w, .y, .ip => z(k0, s.n * N * f8),
        .ring => z(k0, s.n_hist * s.n * N * f8),
        .ring_qt => z(k0, N * s.n_hist * s.n_qt * f8),
        .ip_qt => z(k0, N * s.n_qt * f8),
        .sc => z(k0, N * @sizeOf(lk.LaneScalars)),
        .sims => z(k0, N * @sizeOf(SimState)),
        .active, .round, .stage, .fail, .ops, .accept, .slot0, .decide, .stop => z(k0, N),
        .gate => z(k0, N * @sizeOf(lk.Gate)),
        .flags => z(k0, N * 4),
        .perm_q, .perm_qt => z(k0, N * s.n_hist),
        .lte_in => z(k0, N * @sizeOf(lk.LteIn)),
        .lte_out => z(k0, N * 16),
        .probe_out => z(k0, N * s.n_probes * f8),
        .count, .fails, .first, .over, .slot_next, .ring_slot, .seen, .seen_lo => z(k0, 4),
        .meas_cols, .meas, .meas_resolved => z(k0, 1),
        .tab_idx => up(k0, bytes(t.tab_idx)),
        .col_ptr => up(k0, bytes(t.col_ptr)),
        .row_idx => up(k0, bytes(t.row_idx)),
        .cur_row => up(k0, t.cur_row),
        .diag => up(k0, bytes(t.diag)),
        .probes => up(k0, bytes(t.probes)),
        .dense => k0.alloc(@max(s.dense_slots * (s.n * s.n + s.n) * f8, 1)),
        .peel_idx => k0.alloc(4),
        .peel => k0.alloc(8),
        .step => k0.alloc(N * @sizeOf(lk.LaneStep)),
        .brk_t => k0.alloc(N * 16),
        .best => k0.alloc(N * f8),
        .report => k0.alloc(N * @sizeOf(StepReport)),
        .ring_report => k0.alloc(max_rounds * N * @sizeOf(StepReport)),
        .ring_rows => k0.alloc(@max(max_rounds * N * s.n_probes, 1) * f8),
    };
}

/// Each pinned mirror's element count for shape `s`, in `pinned_mirrors`
/// order.
fn mirrorCounts(s: Shape) [GpuLanes.pinned_mirrors.len]usize {
    const N = s.N;
    return .{ N, N, N, N, N, N, N, N, N, N * s.n_hist, N * s.n_hist, N, N, @max(N * s.n_probes, 1), 4, N, max_rounds * N, @max(max_rounds * N * s.n_probes, 1), N };
}

/// Carves every pinned mirror from one page-locked block, each at a
/// `mirror_align` boundary, and sets their start values. On error nothing
/// stays allocated.
fn carveMirrors(self: *GpuLanes, k0: *Raw, s: Shape) !void {
    const counts = mirrorCounts(s);
    var total: usize = 0;
    inline for (GpuLanes.pinned_mirrors, counts) |f, c| {
        total = std.mem.alignForward(usize, total, mirror_align) + c * @sizeOf(MirrorElem(f));
    }
    const block = try k0.allocPinned(@max(total, 1));
    // Driver allocations are page aligned.
    assert(std.mem.isAligned(@intFromPtr(block.ptr), mirror_align));
    self.pin_block = block;
    var at: usize = 0;
    inline for (GpuLanes.pinned_mirrors, counts) |f, c| {
        at = std.mem.alignForward(usize, at, mirror_align);
        const T = MirrorElem(f);
        // SAFETY: `at` is `mirror_align`-aligned within the page-aligned
        // block, and `c` elements of `T` end before `total`.
        @field(self, f) = @as([*]T, @ptrCast(@alignCast(block.ptr + at)))[0..c];
        at += c * @sizeOf(T);
    }
    assert(at == total);
    @memset(self.stop, 0);
    @memset(self.meas_resolved, 0xff);
    @memset(self.sc, std.mem.zeroes(lk.LaneScalars));
    @memset(self.ops, 0);
    @memset(self.accept_, 0);
    @memset(self.round, 0);
    const nh = s.n_hist;
    for (0..s.N) |l| for (0..nh) |slot| {
        self.perm_q[l * nh + slot] = @intCast(slot);
        self.perm_qt[l * nh + slot] = @intCast(slot);
    };
}

fn MirrorElem(comptime f: []const u8) type {
    return @typeInfo(@FieldType(GpuLanes, f)).pointer.child;
}

/// Each lane's start, as its `Sim.init` left it: x_op in cur and prev, the
/// row ring (physical = logical), i_prev, the tape ring, the permutations.
fn seedLanes(self: *GpuLanes, sims: anytype) !void {
    const N: usize = self.n_lanes;
    const n: usize = self.n;
    const n_hist: usize = self.n_hist;
    const n_qt: usize = self.n_qt;
    const f8 = @sizeOf(f64);
    const host = try self.gpa.alloc(f64, @max(n_hist * n * N, @max((n + 1) * N, N * n_hist * n_qt)));
    defer self.gpa.free(host);
    @memset(host[0 .. (n + 1) * N], 0);
    for (sims, 0..) |s, l| for (s.cur[0..n], 0..) |v, i| {
        host[i * N + l] = v;
    };
    try self.buf(.cur).upload(host.ptr, (n + 1) * N * f8);
    try self.buf(.prev).upload(host.ptr, (n + 1) * N * f8);
    if (self.has_charge) {
        for (sims, 0..) |s, l| for (0..n_hist) |slot| for (s.q_ring[slot][0..n], 0..) |v, i| {
            host[(slot * n + i) * N + l] = v;
        };
        try self.buf(.ring).upload(host.ptr, n_hist * n * N * f8);
        for (sims, 0..) |s, l| for (s.i_prev[0..n], 0..) |v, i| {
            host[i * N + l] = v;
        };
        try self.buf(.ip).upload(host.ptr, n * N * f8);
    }
    if (n_qt > 0) {
        for (sims, 0..) |s, l| for (0..n_hist) |slot| for (s.qt_ring[slot][0..n_qt], 0..) |v, j| {
            host[(slot * n_qt + j) * N + l] = v;
        };
        try self.buf(.ring_qt).upload(host.ptr, N * n_hist * n_qt * f8);
        for (sims, 0..) |s, l| for (s.qt_i_prev[0..n_qt], 0..) |v, j| {
            host[j * N + l] = v;
        };
        try self.buf(.ip_qt).upload(host.ptr, N * n_qt * f8);
    }
    try self.buf(.perm_q).upload(self.perm_q.ptr, N * n_hist);
    try self.buf(.perm_qt).upload(self.perm_qt.ptr, N * n_hist);
}

/// The `ESPICE_GPU_STATS` lines of a finished lane run.
fn printProf(self: *const GpuLanes) void {
    if (!self.prof.on) return;
    const N: usize = self.n_lanes;
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
