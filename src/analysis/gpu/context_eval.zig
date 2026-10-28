//! The device eval of a GPU query. In: the Newton iterate `x`, the time
//! and the analysis state. Out: the four planes stamped on the device and
//! downloaded in one copy, the host batches stamped on top, the charge-only
//! pass and the lazy LTE charge tapes. Per Newton iteration the bus carries
//! `x` up and one contiguous block of planes down; the update and the
//! convergence test stay on the CPU.
//! Cites #126 (the audit).
//!
//! Contents:
//! - `pub noinline fn eval(self: *GpuContext, x: []const f64, t: f64, sim: SimState, charge: bool) bool`
//! - `pub noinline fn evalQ(self: *GpuContext, x: []const f64, t: f64, sim: SimState) bool`
//! - `pub noinline fn syncQTape(self: *GpuContext, sim: SimState) void`
//! - `pub fn evalOnGpu(self: *GpuContext, x: []const f64, t: f64, charge: bool, want: ?GcReq) !?GcReq`
//! - `pub fn evalFailed(self: *GpuContext, e: anyerror) void`
//! - `pub fn auditStart(self: *const GpuContext) u64`
//! - `pub fn auditEnd(self: *GpuContext, t0: u64) void`
//! - `pub fn enqueueEval(self: *GpuContext, x: []const f64, sim_at: SimState, buf: u1, charge: bool, gc: ?GcReq) !void`
//! - `pub fn enqueueCharge(self: *GpuContext, x: []const f64, t: f64) !void`
//! - `pub fn finishCharge(self: *GpuContext, x: []const f64, t: f64) void`
//! - `pub fn uploadBase(self: *GpuContext) !void`
//! - `pub fn sync(self: *GpuContext, comptime why: Wait) !void`
//! - `pub fn upload(self: *GpuContext, buf: *Buffer, bytes: []const u8) !void`
//! - `pub fn download(self: *GpuContext, buf: *Buffer, bytes: []u8) !void`

const std = @import("std");
const assert = std.debug.assert;
const gompute = @import("gompute");
const device_ir = @import("device").abi;
const walk = @import("../par_eval.zig");
const devices = @import("devices.zig");
const context = @import("context.zig");
const ctx_batch = @import("context_batch.zig");
const ctx_cost = @import("context_cost.zig");
const ctx_matrix = @import("context_matrix.zig");
const ctx_scatter = @import("context_scatter.zig");
const ctx_state = @import("context_state.zig");
const ctx_stats = @import("context_stats.zig");
const ctx_sync = @import("context_sync.zig");
const SimState = device_ir.SimState;
const backend = devices.backend;
const GpuContext = context.GpuContext;
const Error = context.Error;
const GcReq = context.GcReq;
const nowNs = context.nowNs;
const Buffer = ctx_batch.Buffer;
const BatchGpu = ctx_batch.BatchGpu;
const block_size = ctx_batch.block_size;
const Audit = ctx_cost.Audit;
const CostCache = ctx_cost.CostCache;
const Wait = ctx_stats.Wait;

/// Stamps all four planes at `x`, ground pin aside; with `charge`, also
/// downloads the resident batches' charge tapes on the same wait, so the
/// next `syncQTape` copies them without another (`Circuit.evalQ`).
/// Serves `eval`, `evalNewton` and `evalQ`: the device path never uses
/// the constant-Jacobian baseline. False after a driver fault (warned
/// once) or once poisoned: no analysis can unwind from one mid-solve, so
/// the caller restamps on the host instead (`Circuit.evalGpu`).
pub noinline fn eval(self: *GpuContext, x: []const f64, t: f64, sim: SimState, charge: bool) bool {
    self.query_hit = null;
    const t0 = auditStart(self);
    defer auditEnd(self, t0);
    self.sim = sim;
    _ = evalOnGpu(self, x, t, charge, null) catch |e| {
        evalFailed(self, e);
        return false;
    };
    return true;
}

/// Stamps only the charges at `x` (`Circuit.evalQ`): q and the tapes
/// from `arp_qonly_<model>` at x unlimited, as the host's charge pass,
/// then the host batches' charge stamps; g, c and rhs keep the last
/// eval's values. False under `ESPICE_GPU_NOFUSE`, when some resident
/// batch has no charge kernel, or
/// on a driver fault (warned once): the caller then runs the full eval.
pub noinline fn evalQ(self: *GpuContext, x: []const f64, t: f64, sim: SimState) bool {
    self.query_hit = null;
    const t0 = auditStart(self);
    defer auditEnd(self, t0);
    // `ESPICE_GPU_NOFUSE` keeps the full pass, for A/B runs.
    if (!self.fuse or !self.q_only or self.poisoned) return false;
    self.sim = sim;
    evalQOnGpu(self, x, t) catch |e| {
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

/// Brings the resident batches' host `q_tape` up to the last eval. A
/// fault leaves the resident tapes at their last value for this step,
/// warns once, and poisons the context so every later step runs on the
/// host.
pub noinline fn syncQTape(self: *GpuContext, sim: SimState) void {
    self.query_hit = null;
    self.sim = sim;
    syncTape(self) catch {
        ctx_state.warnStateFallback(self);
        self.poisoned = true;
    };
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
pub fn evalOnGpu(self: *GpuContext, x: []const f64, t: f64, charge: bool, want: ?GcReq) !?GcReq {
    if (comptime backend == null) return Error.NoGpuArtifacts;
    self.follow = null;
    if (self.poisoned) return error.GpuStateReject;
    if (self.params_dirty) try ctx_sync.repack(self);
    self.prof.start();
    self.prof.evals += 1;
    if (charge) self.prof.charge_evals += 1 else self.gc_last = want;
    self.x_len = x.len;

    const pre = self.pre;
    self.pre = null;
    // A host step since the prefetch (`commit_state`, a revert) leaves
    // it reading old rows.
    const pushed = try ctx_sync.pushStepped(self);
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
            try ctx_matrix.enqueueCombine(self, self.cur, want.?.alpha);
            self.gc_buf[self.cur] = want;
            served = want;
            try sync(self, .eval);
        };
    } else {
        try enqueueEval(self, x, self.simAt(t), self.cur, charge, want);
        self.prof.lap("enqueue");
        if (charge) try sync(self, .charge) else try sync(self, .eval);
        self.prof.lap("wait");
    }
    const pl = self.planes();
    if (served) |r| {
        ctx_matrix.gcScatter(self, self.cur, r);
    } else if (!self.resident_charge and self.has_charge) {
        // The device writes neither c nor q when no resident batch has
        // charge; the host batches stamp them from zero.
        @memset(pl.c_vals, 0);
        @memset(pl.q_vec, 0);
    }
    self.tape_fresh = false;
    self.last_t = t;

    if (self.chk.len > 0) try evalCheck(self, x, t);

    // None under a resident matrix (`gcInit`).
    for (self.cpu_batches) |b| b.eval(b.ctx, &pl, 0, b.count, x, t);
    self.prof.lap("host");
    return served;
}

pub fn evalFailed(self: *GpuContext, e: anyerror) void {
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

pub fn auditStart(self: *const GpuContext) u64 {
    const a = self.audit orelse return 0;
    return if (a.done or self.prof.evals < Audit.from) 0 else nowNs();
}

/// Books the time since `t0` and, at `Audit.at` evals, demotes the
/// query to the host when the device costs more per eval than the host
/// was priced at.
pub fn auditEnd(self: *GpuContext, t0: u64) void {
    if (t0 == 0) return;
    const a = &self.audit.?;
    a.spent_ns += nowNs() - t0;
    if (self.prof.evals < Audit.at) return;
    a.done = true;
    const gpu_us = @as(f64, @floatFromInt(a.spent_ns)) * 1e-3 / @as(f64, @floatFromInt(self.prof.evals - Audit.from));
    const losing = gpu_us > a.cpu_us;
    CostCache.write(a.key, a.knobs, .{ .cpu_us = a.cpu_us, .gpu_us = gpu_us });
    if (self.prof.on) std.debug.print("gpu-stats: measured {d:.0} us per device eval against the CPU's {d:.0}: {s}\n", .{ gpu_us, a.cpu_us, if (losing) "the rest of the query runs on the CPU" else "GPU" });
    if (losing and a.demote) demote(self);
}

/// Enqueues the device half without waiting: upload x, clear the stage,
/// launch every resident batch, reduce, download into the pinned planes.
/// Separate so `evalCheck` can replay the same pass.
pub fn enqueueEval(self: *GpuContext, x: []const f64, sim_at: SimState, buf: u1, charge: bool, gc: ?GcReq) !void {
    if (comptime backend == null) return Error.NoGpuArtifacts;
    _ = try ctx_sync.pushStepped(self);
    @memcpy(self.pin_x[buf][0..x.len], x);
    try upload(self, &self.d_x[buf], std.mem.sliceAsBytes(self.pin_x[buf][0..x.len]));
    // No staging clear: the kernels store each cell they own every pass,
    // and the cells none writes hold `init`'s zero.
    try phaseLap(self, 0);

    for (self.batches) |*bg| {
        if (bg.count == 0) continue;
        try ctx_state.syncSeededLim(self, bg);
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

    try phaseLap(self, 1);
    try launchReduce(self, &self.d_seg1, &self.d_stage, &self.d_mid, self.n_pieces);
    try phaseLap(self, 2);
    try launchReduce(self, &self.d_seg2, &self.d_mid, &self.d_planes, self.n_cells);
    try phaseLap(self, 3);
    self.gc_buf[buf] = gc;
    if (gc) |r| try ctx_matrix.enqueueGcDown(self, buf, r) else {
        try download(self, &self.d_planes, std.mem.sliceAsBytes(self.pinnedBuf(buf)[0..self.n_cells]));
    }
    try phaseLap(self, 4);
    // `pin_tape` is about to hold buffer `buf`'s tape, or stale bytes.
    self.tape_buf = null;
    if (charge) try enqueueTape(self, buf, sim_at);
}

/// The device half of `evalQOnGpu`: x up, the charge kernels, the q
/// reduce and the q plane and tapes down, unwaited.
pub fn enqueueCharge(self: *GpuContext, x: []const f64, t: f64) !void {
    _ = try ctx_sync.pushStepped(self);
    self.prof.start();
    self.prof.evals += 1;
    self.prof.charge_evals += 1;
    const buf = self.cur;
    // `d_x[cur]` holds the x behind the planes' charge, which a lazy
    // tape pass reads.
    @memcpy(self.pin_x[buf][0..x.len], x);
    try upload(self, &self.d_x[buf], std.mem.sliceAsBytes(self.pin_x[buf][0..x.len]));
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
            try download(self, &bg.d_tape, std.mem.sliceAsBytes(self.pin_tape[off..][0..bg.tape.len]));
        }
        off += bg.tape.len;
    }
    // Batches without a q kernel carry no charge, so their tapes are
    // empty and `pin_tape` is whole.
    std.debug.assert(off == self.pin_tape.len);
    // Level 1 over the q plane's pieces, level 2 over its cells: the
    // seg tables index the whole stage and the piece array absolutely.
    const l = self.layout;
    var seg1 = ctx_batch.view(self.d_seg1, 2 * self.q_piece0 * @sizeOf(u32));
    var mid = ctx_batch.view(self.d_mid, self.q_piece0 * @sizeOf(f64));
    try launchReduce(self, &seg1, &self.d_stage, &mid, self.n_pieces - self.q_piece0);
    var seg2 = ctx_batch.view(self.d_seg2, 2 * l.off(.q) * @sizeOf(u32));
    var q_plane = ctx_batch.view(self.d_planes, l.off(.q) * @sizeOf(f64));
    try launchReduce(self, &seg2, &self.d_mid, &q_plane, l.nr);
    try download(self, &q_plane, std.mem.sliceAsBytes(self.pinnedBuf(buf)[l.off(.q)..][0..l.nr]));
    self.tape_buf = buf;
}

/// The host half of `evalQOnGpu`, once the q plane has landed.
pub fn finishCharge(self: *GpuContext, x: []const f64, t: f64) void {
    self.tape_fresh = false;
    self.last_t = t;
    const pl = self.planes();
    for (self.cpu_batches) |b| walk.stampRange(b, &pl, 0, b.count, x, t, .charge);
}

/// Stamps the constant batches on the host into scratch planes (their
/// Jacobian is the same at every x) and uploads g and c at the base
/// cells behind the staging. Writes those batches' host `q_tape`, so the
/// tapes are marked stale.
// ponytail: the baseline values are the host stamp's sum of the
// constant batches, then reduced ahead of the rest of the cell: the sum
// order differs from the full restamp (tolerance class).
pub fn uploadBase(self: *GpuContext) !void {
    if (self.base_cells.len == 0) return;
    const l = self.layout;
    const scratch = try self.gpa.alloc(f64, 2 * l.ng + 2 * l.nr + self.n + 1);
    defer self.gpa.free(scratch);
    @memset(scratch, 0);
    const pl = ctx_scatter.scratchPlanes(scratch, l.ng, l.nr);
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
    var base = ctx_batch.view(self.d_stage, self.n_stage * @sizeOf(f64));
    try upload(self, &base, std.mem.sliceAsBytes(vals));
    // `vals` is pageable and freed on return: the driver staged it, but
    // wait anyway so the bytes are on the device before any launch.
    try sync(self, .other);
}

/// Waits for this stream only: a context-wide sync would also wait on
/// work this process does not own. Counted for `ESPICE_GPU_STATS`.
pub fn sync(self: *GpuContext, comptime why: Wait) !void {
    self.prof.syncs += 1;
    self.prof.waits[@backingInt(why)] += 1;
    try self.stream.synchronize();
}

/// Copies `bytes` (any host memory) to the start of `buf` on the
/// context's stream, never the legacy default stream, which would fence
/// every other stream on the device. From pageable memory the driver
/// stages the bytes before returning, so the caller may reuse them at
/// once.
pub fn upload(self: *GpuContext, buf: *Buffer, bytes: []const u8) !void {
    if (bytes.len == 0) return;
    try buf.uploadAtAsync(bytes.ptr, 0, bytes.len, &self.stream);
    self.prof.up += bytes.len;
}

/// Queues a copy of `buf`'s first `bytes.len` bytes into `bytes`; they
/// land by the next `sync`. Asynchronous only when `bytes` is pinned
/// (`BatchGpu.dev`); into pageable memory the driver blocks until done.
pub fn download(self: *GpuContext, buf: *Buffer, bytes: []u8) !void {
    if (bytes.len == 0) return;
    try buf.downloadAtAsync(bytes.ptr, 0, bytes.len, &self.stream);
    self.prof.down += bytes.len;
}

// ---- private ----

/// Whether planes a prefetch left under `have` serve an eval that asks
/// for `want`: whole planes serve anything, and a resident matrix serves
/// a request for the same side (a host-side A at another alpha is formed
/// again, `evalOnGpu`).
fn gcFits(have: ?GcReq, want: ?GcReq) bool {
    const h = have orelse return true;
    const w = want orelse return false;
    return h.device == w.device;
}

/// Under `Prof.phases`, waits for the stream and books the time since
/// the last lap to stage `i`.
fn phaseLap(self: *GpuContext, comptime i: usize) !void {
    if (!self.prof.phases) return;
    if (i == 0) self.prof.last = nowNs();
    try self.stream.synchronize();
    const t = nowNs();
    self.prof.phase[i] += t - self.prof.last;
    self.prof.last = t;
}

/// One `arp_reduce_*` launch: `out[i] = sum(in[seg[i]..seg[i+1]])`.
fn launchReduce(self: *GpuContext, seg: *Buffer, in: *Buffer, out: *Buffer, cells: usize) !void {
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
fn evalCheck(self: *GpuContext, x: []const f64, t: f64) !void {
    if (comptime backend == null) return Error.NoGpuArtifacts;
    const first = self.pinnedBuf(self.cur)[0..self.n_cells];
    @memcpy(self.chk, first);
    try enqueueEval(self, x, self.simAt(t), self.cur, false, null);
    try sync(self, .check);
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

/// Moves the rest of the query to the host through the fault fallback,
/// after bringing the device state home, so it continues exactly where
/// the device left it. Stays on the device if that download fails.
fn demote(self: *GpuContext) void {
    ctx_sync.syncHostState(self) catch return;
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

fn evalQOnGpu(self: *GpuContext, x: []const f64, t: f64) !void {
    if (comptime backend == null) return Error.NoGpuArtifacts;
    self.follow = null;
    // The tape below lands in `pin_tape`, which a prefetch also targets.
    self.pre = null;
    if (self.params_dirty) try ctx_sync.repack(self);
    try enqueueCharge(self, x, t);
    self.prof.lap("enqueue");
    try sync(self, .charge);
    self.prof.lap("wait");
    finishCharge(self, x, t);
    self.prof.lap("host");
}

/// Fills every resident batch's host `q_tape` with the LTE charges at the
/// last eval's x, which `d_x[cur]` still holds. Lazy: the transient reads the
/// tape once per step, and Newton evals many times in between.
fn syncTape(self: *GpuContext) !void {
    if (comptime backend == null) return;
    if (self.tape_fresh or self.pin_tape.len == 0) return;
    if (self.poisoned) return error.GpuStateReject;
    if (self.tape_buf != self.cur) {
        try enqueueTape(self, self.cur, self.simAt(self.last_t));
        try sync(self, .tape);
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
fn enqueueTape(self: *GpuContext, buf: u1, sim_at: SimState) !void {
    _ = try ctx_sync.pushStepped(self);
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
        try download(self, &bg.d_tape, std.mem.sliceAsBytes(self.pin_tape[off..][0..bg.tape.len]));
        off += bg.tape.len;
    }
    self.tape_buf = buf;
}
