//! The limit, state and accepted-step passes of the resident batches. In:
//! the iterate and its predecessor, or the converged x, or a step-control
//! op. Out: the limit verdict, the state halves' step bound, the latch
//! commits, reverts and queries, run on the device beside the host walks of
//! the other batches, and fused with the next eval's launch when announced.
//! Cites #15 (staging at the converged iterate).
//!
//! Contents:
//! - `pub noinline fn applyLimits(self: *GpuContext, x: []f64, x_old: []const f64, sim: SimState) bool`
//! - `pub noinline fn evalFollows(self: *GpuContext, x: []const f64, sim: SimState, charge: bool) void`
//! - `pub noinline fn updateStates(self: *GpuContext, x: []const f64, sim: SimState) ?f64`
//! - `pub noinline fn acceptStates(self: *GpuContext, x: []const f64, t: f64, sim: SimState) ??f64`
//! - `pub noinline fn clearLimits(self: *GpuContext) void`
//! - `pub noinline fn stateCtl(self: *GpuContext, op: device_ir.StateCtlOp, sim: SimState) bool`
//! - `pub noinline fn seedJunctions(self: *GpuContext, x: []f64) void`
//! - `pub fn syncSeededLim(self: *GpuContext, bg: *BatchGpu) !void`
//! - `pub fn warnStateFallback(self: *GpuContext) void`

const std = @import("std");
const assert = std.debug.assert;
const gompute = @import("gompute");
const device_ir = @import("device").abi;
const walk = @import("../par_eval.zig");
const devices = @import("devices.zig");
const context = @import("context.zig");
const ctx_batch = @import("context_batch.zig");
const ctx_eval = @import("context_eval.zig");
const ctx_sync = @import("context_sync.zig");
const SimState = device_ir.SimState;
const backend = devices.backend;
const GpuContext = context.GpuContext;
const Error = context.Error;
const Buffer = ctx_batch.Buffer;
const BatchGpu = ctx_batch.BatchGpu;

/// Runs the device limit pass and the host walk on the rest. A fault
/// falls back to the host walk over every batch; the resident batches'
/// stale host lim/state self-heal, since
/// limiting restarts from x_old and the latches recompute from x alone.
pub noinline fn applyLimits(self: *GpuContext, x: []f64, x_old: []const f64, sim: SimState) bool {
    self.query_hit = null;
    const t0 = ctx_eval.auditStart(self);
    defer ctx_eval.auditEnd(self, t0);
    self.sim = sim;
    return applyLimitsOnGpu(self, x, x_old) catch {
        warnStateFallback(self);
        return walk.limitBatches(self.all_batches, x, x_old);
    };
}

/// Announces an eval at `x` under `sim` right after the next
/// `applyLimits` or `stateCtl(.query)` (`Circuit.evalFollows`). `x` is
/// borrowed until that call. A no-op under `ESPICE_GPU_NOFUSE`.
pub noinline fn evalFollows(self: *GpuContext, x: []const f64, sim: SimState, charge: bool) void {
    if (!self.fuse) return;
    self.follow = .{ .x = x, .sim = sim, .charge = charge };
}

/// Runs the resident batches' state pass and walks the host ones. Every
/// batch walks on the host once poisoned.
pub noinline fn updateStates(self: *GpuContext, x: []const f64, sim: SimState) ?f64 {
    self.query_hit = null;
    self.sim = sim;
    return updateStatesOnGpu(self, x) catch {
        warnStateFallback(self);
        return walk.updateBatches(self.all_batches, x);
    };
}

/// The transient's accepted-point pass (`Circuit.acceptStates`): the
/// charges at `x`, then every resident state half, then the accepted-
/// state query, queued as one chain behind one wait, in the
/// order the separate calls run them: the charges read the state before
/// `updateState` latches it, and the query reads it after. The query's
/// device half is kept for the `stateCtl(.query)` that follows. Returns
/// null when the charge-only path is off (the caller then runs `evalQ`
/// and `updateStates`), else what `updateStates` returns. A fault stamps
/// the charges and walks the states of every batch on the host.
pub noinline fn acceptStates(self: *GpuContext, x: []const f64, t: f64, sim: SimState) ??f64 {
    self.query_hit = null;
    const t0 = ctx_eval.auditStart(self);
    defer ctx_eval.auditEnd(self, t0);
    if (!self.fuse or !self.q_only or self.poisoned) return null;
    self.sim = sim;
    return acceptStatesOnGpu(self, x, t) catch {
        warnStateFallback(self);
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

/// Disarms device limiting and walks the host batches' `clearLimits`.
pub noinline fn clearLimits(self: *GpuContext) void {
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

/// Commits, reverts or queries the accepted state: accepted-step
/// latches mutate device-resident instance blobs, so a host walk cannot
/// stand in for a resident batch. A fault walks every batch on the host.
pub noinline fn stateCtl(self: *GpuContext, op: device_ir.StateCtlOp, sim: SimState) bool {
    self.sim = sim;
    return stateCtlOnGpu(self, op) catch {
        warnStateFallback(self);
        return walk.stateCtlBatches(self.all_batches, op);
    };
}

/// Every batch seeds the host `x`, then each resident lim plane is
/// marked for upload before its next launch.
pub noinline fn seedJunctions(self: *GpuContext, x: []f64) void {
    self.query_hit = null;
    self.pre = null;
    walk.seedBatches(self.all_batches, x);
    for (self.batches) |*bg| {
        if (bg.has_lim) bg.lim_dirty = true;
    }
}

/// Uploads seed voltages a host `seedJunctions` left in the batch's lim
/// plane and arms device limiting. Synchronous, so it lands before the
/// launch that reads it.
pub fn syncSeededLim(self: *GpuContext, bg: *BatchGpu) !void {
    if (!bg.lim_dirty) return;
    bg.lim_dirty = false;
    if (!bg.has_lim) return;
    const p = bg.payload(bg.ctx);
    if (!p.lim_active) return;
    try ctx_eval.upload(self, &bg.d_lim, std.mem.sliceAsBytes(p.lim_x));
    bg.lim_active = true;
}

pub fn warnStateFallback(self: *GpuContext) void {
    if (self.warned_state) return;
    self.warned_state = true;
    std.debug.print(
        "warning: GPU limit/state pass failed; falling back to the CPU walk\n",
        .{},
    );
}

// ---- private ----

fn acceptStatesOnGpu(self: *GpuContext, x: []const f64, t: f64) !?f64 {
    if (comptime backend == null) return Error.NoGpuArtifacts;
    self.follow = null;
    self.pre = null;
    if (self.params_dirty) try ctx_sync.repack(self);
    try ctx_eval.enqueueCharge(self, x, t);
    const states = try enqueueStates(self, x);
    var word1 = ctx_batch.view(self.d_flags, 4);
    const query = try enqueueCtl(self, .query, &word1);
    if (states or query) {
        try ctx_eval.download(self, &self.d_flags, self.pin_flags[0..8]);
    }
    self.prof.lap("enqueue");
    try ctx_eval.sync(self, .charge);
    self.prof.lap("wait");
    ctx_eval.finishCharge(self, x, t);
    self.prof.lap("host");
    if (states and std.mem.readInt(u32, self.pin_flags[0..4], .little) & 2 != 0) {
        self.poisoned = true;
        return error.GpuStateReject;
    }
    const tr = try finishStates(self, x);
    if (query) self.query_hit = std.mem.readInt(u32, self.pin_flags[4..8], .little) != 0;
    return tr;
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
fn applyLimitsOnGpu(self: *GpuContext, x: []f64, x_old: []const f64) !bool {
    if (comptime backend == null) return false;
    const follow = self.follow;
    self.follow = null;
    self.pre = null;
    // A prefetch whose wait failed holds no planes.
    errdefer self.pre = null;
    if (self.poisoned) return error.GpuStateReject;
    if (self.params_dirty) try ctx_sync.repack(self);
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
            try ctx_eval.upload(self, &self.d_x2, std.mem.sliceAsBytes(self.pin_x2[0..n]));
            @memcpy(self.pin_xold[0..n], x_old[0..n]);
            try ctx_eval.upload(self, &self.d_xold, std.mem.sliceAsBytes(self.pin_xold[0..n]));
            try self.d_flags.fillAsync(0, 4, &self.stream);
        }
        try syncSeededLim(self, bg);
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
        try ctx_eval.download(self, &self.d_flags, self.pin_flags[0..4]);
    }

    var any = walk.limitBatches(self.cpu_batches, x, x_old);

    if (launched) {
        // Without a limit launch there is no wait to share.
        if (follow) |f| try prefetch(self, f);
        try ctx_eval.sync(self, .lim);
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

/// Enqueues the eval `f` announced into the other plane buffer, behind
/// whatever the stream holds; the caller's wait covers it.
fn prefetch(self: *GpuContext, f: GpuContext.Follow) !void {
    const gc = if (f.charge) null else self.gc_last;
    try ctx_eval.enqueueEval(self, f.x, f.sim, self.cur ^ 1, f.charge, gc);
    self.pre = .{ .sim = f.sim, .charge = f.charge, .gc = gc };
}

/// Runs every resident batch's state half at the converged `x` and the
/// host walk on the rest. The converger stages at x_k, the iterate
/// `newton` publishes, after its last limit pass ran at x_k+1; staging
/// in that pass committed state one iterate ahead of the CPU and drifted
/// tran/device_hfet_inverter's incremental Meyer charges 0.34% (#15). A
/// resident batch cannot ask for a step reject: the kernel flags any
/// non-`.ok` `updateState` as a fault instead.
fn updateStatesOnGpu(self: *GpuContext, x: []const f64) !?f64 {
    if (comptime backend == null) return walk.updateBatches(self.cpu_batches, x);
    if (self.poisoned) return error.GpuStateReject;
    // The host walk waits for the verdict: a second `updateState` reads
    // the held values the first one wrote, so the fallback's walk over
    // every batch must be the only one.
    if (try enqueueStates(self, x)) {
        try ctx_eval.download(self, &self.d_flags, self.pin_flags[0..4]);
        try ctx_eval.sync(self, .state);
        if (std.mem.readInt(u32, self.pin_flags[0..4], .little) & 2 != 0) {
            self.poisoned = true;
            return error.GpuStateReject;
        }
    }
    return finishStates(self, x);
}

/// Queues the state half of every resident batch that has one at `x`,
/// its verdict in flag word 0, unwaited. True when it launched any.
fn enqueueStates(self: *GpuContext, x: []const f64) !bool {
    const n = self.n;
    var launched = false;
    for (self.batches) |*bg| {
        const lk = if (bg.lim_kernel) |*k| k else continue;
        if (bg.count == 0 or !bg.has_state) continue;
        if (!launched) {
            // A prefetched eval read the state this launch rewrites.
            self.pre = null;
            if (self.params_dirty) try ctx_sync.repack(self);
            @memcpy(self.pin_x2[0..n], x[0..n]);
            try ctx_eval.upload(self, &self.d_x2, std.mem.sliceAsBytes(self.pin_x2[0..n]));
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
fn finishStates(self: *GpuContext, x: []const f64) !?f64 {
    const host = walk.updateBatches(self.cpu_batches, x);
    try ctx_sync.pullStepped(self);
    const stepped = walk.updateBatches(self.stepped_batches, x);
    // Now, so the eval `stateCtl(.query)` prefetches reads the new rows.
    if (try ctx_sync.pushStepped(self)) self.pre = null;
    if (host == null) return stepped;
    return if (stepped) |s| @min(host.?, s) else host;
}

/// Runs the accepted-step latch (path commit `pb <- wb, pq += wq`) on the
/// resident instance blobs and the host walk on the rest. The kernels OR
/// each instance's verdict into `d_flags`, so `.query` costs one launch
/// and a 4-byte sync.
fn stateCtlOnGpu(self: *GpuContext, op: device_ir.StateCtlOp) !bool {
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
    if (self.params_dirty) try ctx_sync.repack(self);
    const launched = try enqueueCtl(self, op, &self.d_flags);
    // Every caller discards what `.commit` and `.revert` return, so they
    // stay on the stream unwaited; the next wait surfaces a fault. Only a
    // `.query` reads the flag, and it shares its wait with the charge
    // eval the transient announced.
    const wait = launched and op == .query;
    // Queued before the host walk so it drains meanwhile, and so a failed
    // enqueue happens before the walk: the fallback re-walks every batch,
    // and the path commit `pq += wq` is not idempotent.
    if (wait) {
        try ctx_eval.download(self, &self.d_flags, self.pin_flags[0..4]);
        if (follow) |f| try prefetch(self, f);
    }

    var dirty = walk.stateCtlBatches(self.cpu_batches, op);
    if (walk.stateCtlBatches(self.stepped_batches, op)) dirty = true;
    if (wait) {
        try ctx_eval.sync(self, .query);
        if (std.mem.readInt(u32, self.pin_flags[0..4], .little) != 0) dirty = true;
    }
    return dirty;
}

/// Queues every resident `arp_ctl_<model>` with `op`, verdicts ORed into
/// the word at `flags` (cleared first), unwaited. True when any ran.
fn enqueueCtl(self: *GpuContext, op: device_ir.StateCtlOp, flags: *Buffer) !bool {
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
