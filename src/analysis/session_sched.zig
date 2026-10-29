//! A session's cooperative scheduler: building each row's executor,
//! gathering ready rows into lane batches, launching and settling quanta,
//! following finished rows to their dependents, and the advance loops.
//! `Session` exposes each function here as a method.
//! In: ready rows. Out: rows run, settled and released.
//! Cites #13; dev/research/lane-batched-simulation.md; 05 5.3b.
//!
//! Contents:
//! - `makeExecutor`
//! - `gather`
//! - `peerOf`
//! - `lead`
//! - `gpuLaned`
//! - `staticWide`
//! - `gatherWide`
//! - `shareWithHost`
//! - `settlePeers`
//! - `launch`
//! - `settle`
//! - `laneDone`
//! - `assemble`
//! - `readerDone`
//! - `releaseIfRead`
//! - `advanceReady`
//! - `event`
//! - `requeue`
//! - `nextReady`
//! - `follow`
//! - `followOne`
//! - `runAll`
//! - `advance`
//! - `advanceDone`

const std = @import("std");
const assert = std.debug.assert;
const core = @import("core");
const requests = core.query;
const Circuit = @import("device").Circuit;
const execution = @import("executor.zig");
const progress = @import("worker.zig");
const types = @import("types.zig");
const session = @import("session.zig");
const jobs_mod = @import("session_jobs.zig");
const Session = session.Session;
const Status = session.Status;
const Scope = session.Scope;
const Limits = session.Limits;
const Advance = session.Advance;
const PrintOptions = session.PrintOptions;
const Failure = session.Failure;
const PointError = session.PointError;
const Pool = session.Pool;
const QueryId = session.QueryId;
const Quantum = progress.Quantum;
const Result = types.Result;
const none = jobs_mod.none;
const no_output = jobs_mod.no_output;
const no_variant = jobs_mod.no_variant;
const lane_window = jobs_mod.lane_window;
const sameLaneJob = jobs_mod.sameLaneJob;
const batchable = jobs_mod.batchable;
const splitLanes = jobs_mod.splitLanes;
const prerequisite = jobs_mod.prerequisite;
const withoutVariant = jobs_mod.withoutVariant;
const copyValue = jobs_mod.copyValue;
const JobKey = jobs_mod.JobKey;
const OpKey = jobs_mod.OpKey;
const validateDeck = core.validate.validateDeck;

pub fn makeExecutor(self: *Session, i: usize) !void {
    if (self.rows.items(.executor)[i] != null) return;
    const dep = self.rows.items(.dependency)[i];
    const initial = if (dep == none) null else self.rows.items(.executor)[@backingInt(dep)];
    // Query `i` may take its source's state only as its last reader,
    // once every other reader has stopped a quantum: their copies are done.
    const last = if (dep == none) self.last_template_reader and self.root_readers == 1 else self.rows.items(.readers)[@backingInt(dep)] == 1;
    self.rows.items(.executor)[i] = try execution.Executor.create(
        self.allocator,
        self.io,
        self.templateOf(i),
        self.deck,
        self.jobAt(i),
        initial,
        (if (dep == none) self.config.final_plan else self.final(@backingInt(dep))) and last,
        self.config,
    );
    if (self.stream) |s| if (@backingInt(s.id) == i) {
        self.rows.items(.executor)[i].?.stream = s.writer;
    };
    self.rows.items(.executor)[i].?.row = @intCast(i);
    if (self.config.timing_in_depth)
        self.rows.items(.executor)[i].?.controller.options.timing_query = @fromBackingInt(@intCast(i));
}

/// The grouping pass (lane-batched-simulation.md §2): makes ready `.op`,
/// `batchable` row `i` (`.op`, `.dc`, `.tran`, `.pss`, an AC-family kind or
/// a `.mc`/`.temp` lane row) the leader of a lane batch with up to
/// `lane_width - 1` other ready, never-started rows of its kind on its
/// template, with a prerequisite exactly when it has one, whose options
/// differ at most in `tol.variant`, `tol.temp_c` and a lane row's lane,
/// taken from the queue ahead.
/// Each lane is bitwise its query run alone, so grouping is free to
/// follow scheduling order. A row whose executor cannot be built is
/// left to run alone.
/// ponytail: looks `lane_window` rows into `urgent` and the queue, not
/// all of them; a sweep's rows are adjacent in row order.
pub fn gather(self: *Session, i: usize) void {
    const lane_width = execution.lane_width;
    const config = self.config;
    if (config.gpu_lanes and self.gpuLaned(i)) return self.gatherWide(i);
    if (!execution.lanesOn(config) or config.backend != .cpu or config.device_threads != 1 or config.timing_in_depth) return;
    const ji = self.jobAt(i);
    if (!batchable(ji) or self.rows.items(.executor)[i] != null) return;
    var peers: [lane_width - 1]usize = undefined;
    var n: usize = 0;
    // A finished batch's dependents wait in `urgent`, newest last.
    const urgent = self.urgent.items;
    const from_urgent = @min(urgent.len, lane_window);
    const queued = self.queue.items[@min(self.queue_head, self.queue.items.len)..];
    for (0..from_urgent + @min(queued.len, lane_window)) |at| {
        if (n == peers.len) break;
        const j = if (at < from_urgent) urgent[urgent.len - 1 - at] else queued[at - from_urgent];
        if (!self.peerOf(i, ji, j)) continue;
        peers[n] = j;
        n += 1;
    }
    _ = self.lead(i, peers[0..n]);
}

/// Row `j` may join leader `i` (whose query is `ji`) in a lane batch: a
/// ready, never-started row other than `i` on its template, with a
/// prerequisite exactly when `i` has one, and `sameLaneJob` options.
pub fn peerOf(self: *const Session, i: usize, ji: requests.Query, j: usize) bool {
    const rows = self.rows.slice();
    return j != i and self.ready(j) and !rows.items(.running)[j] and rows.items(.executor)[j] == null and
        rows.items(.template)[j] == rows.items(.template)[i] and
        (rows.items(.dependency)[j] == none) == (rows.items(.dependency)[i] == none) and
        sameLaneJob(ji, self.jobAt(j));
}

/// Makes row `i` the leader of `peers` (rows): builds every executor
/// and marks each peer running. A peer whose executor cannot be built
/// is left out to run alone. Null, with no peer taken, when `peers` is
/// empty or the leader or its list cannot be built.
pub fn lead(self: *Session, i: usize, peers: []const usize) ?*execution.Executor {
    if (peers.len == 0) return null;
    self.makeExecutor(i) catch return null;
    const leader = self.rows.items(.executor)[i].?;
    const list = leader.work.allocator().alloc(*execution.Executor, peers.len) catch return null;
    var taken: usize = 0;
    for (peers) |j| {
        self.makeExecutor(j) catch continue;
        list[taken] = self.rows.items(.executor)[j].?;
        taken += 1;
        self.rows.items(.running)[j] = true;
    }
    leader.peers = list[0..taken];
    return leader;
}

/// Row i is a never-built transient, shooting, HB, AC-family or
/// `staticWide` row: under `Config.gpu_lanes` it gathers as GPU lanes.
/// A row `shareWithHost` gave the CPU already has its executor.
pub fn gpuLaned(self: *const Session, i: usize) bool {
    const kind = self.kindAt(i);
    return (kind == .tran or kind == .pss or kind == .hb or execution.gpuFreqLaneKind(kind) or self.staticWide(i)) and
        self.rows.items(.executor)[i] == null;
}

/// Row i is an operating point or `.dc` sweep that batches as GPU
/// lanes (`Config.gpu_static`).
pub fn staticWide(self: *const Session, i: usize) bool {
    const job = self.kindAt(i);
    return self.config.gpu_static and (job == .op or job == .dc);
}

/// The GPU lane form of `gather`: up to `gpu_lane_cap - 1` ready,
/// never-started transient, shooting, HB or AC-family rows of row i's
/// template and options, from all of `urgent` and the queue (the rows are ready together
/// under breadth-first order). A row whose executor cannot be built is
/// left to run alone.
/// ponytail: an ineligible sweep (`tran.gpuLaneRefusal`) still gathers
/// and then runs its lanes one after another on one worker; check the
/// template before gathering if that ever costs a `--jobs` run.
pub fn gatherWide(self: *Session, i: usize) void {
    const ji = self.jobAt(i);
    var peers: std.ArrayList(usize) = .empty;
    defer peers.deinit(self.allocator);
    // Bound the actual GPU lanes: frequencies for AC, collocation
    // samples for HB. The full tone box is an upper bound after
    // frequency merging/truncation; one tone uses 2x oversampling.
    const width: u32 = if (execution.gpuFreqLaneKind(ji))
        @import("solver").freq_solve.FreqSolver.W
    else if (ji == .hb) blk: {
        var samples: u32 = 2 * (2 * @as(u32, ji.hb.n_harmonics) + 1);
        for (ji.hb.extra_harmonics) |h|
            samples = std.math.mul(u32, samples, 2 * @as(u32, h) + 1) catch std.math.maxInt(u32);
        break :blk samples;
    } else 1;
    const cap = @max(self.config.gpu_lane_cap / width, 1) - 1;
    const queued = self.queue.items[@min(self.queue_head, self.queue.items.len)..];
    for ([_][]const u32{ self.urgent.items, queued }) |list| for (list) |j32| {
        if (peers.items.len == cap) break;
        const j: usize = j32;
        if (!self.peerOf(i, ji, j) or std.mem.indexOfScalar(usize, peers.items, j) != null) continue;
        peers.append(self.allocator, j) catch break;
    };
    // Lanes in row order, whatever order the OPs finished in, so lane 0
    // (the LU plan's source) and the host share are the same each run.
    std.mem.sortUnstable(usize, peers.items, {}, std.sort.asc(usize));
    self.shareWithHost(&peers);
    const leader = self.lead(i, peers.items) orelse return;
    // Under a final plan a member nothing reads drops its circuit with
    // its wave (`Executor.executeGpuBatch`), as `releaseIfRead` would
    // once the batch ends.
    leader.no_readers = self.final(i) and self.rows.items(.readers)[i] == 0;
    for (leader.peers) |p| p.no_readers = self.final(p.row) and self.rows.items(.readers)[p.row] == 0;
}

/// `Config.gpu_share` f: of a GPU batch leader and its gathered
/// `peers`, the first ceil(f * all) stay on the device and the rest
/// leave `peers` with a CPU executor, which the pool runs alongside the
/// batch. The split is by row order, so a run is reproducible.
pub fn shareWithHost(self: *Session, peers: *std.ArrayList(usize)) void {
    const f = self.config.gpu_share;
    if (!(f > 0 and f < 1)) return;
    const all: f64 = @floatFromInt(1 + peers.items.len);
    const keep: usize = @intFromFloat(@max(1, @ceil(f * all)) - 1);
    if (keep >= peers.items.len) return;
    for (peers.items[keep..]) |j| {
        self.makeExecutor(j) catch continue;
        const e = self.rows.items(.executor)[j].?;
        e.config.backend = .cpu;
        e.config.gpu_lanes = false;
    }
    peers.shrinkRetainingCapacity(keep);
}

/// Settles the peers of batch leader `i`, which has stopped: each takes
/// the outcome its lane left, or the leader's cancel.
pub fn settlePeers(self: *Session, i: usize) void {
    const leader = self.rows.items(.executor)[i].?;
    const cancelled = self.rows.items(.status)[i] == .cancelled;
    for (leader.peers) |p| {
        const j = p.row;
        assert(self.rows.items(.running)[j]);
        self.rows.items(.running)[j] = false;
        const first = self.rows.items(.status)[j] == .pending;
        if (p.published != null) {
            self.rows.items(.status)[j] = .complete;
        } else if (cancelled or p.peer_failure == null) {
            self.rows.items(.status)[j] = .cancelled;
        } else {
            self.rows.items(.status)[j] = .failed;
            self.rows.items(.failure)[j] = p.peer_failure;
        }
        if (first) self.readerDone(j);
        self.releaseIfRead(j);
        self.laneDone(j);
    }
}

/// Starts query `i`'s next quantum, building its executor first, with
/// `signal` posted when it stops. False, the row failed and settled,
/// when it could not start.
pub fn launch(self: *Session, i: usize, quantum: Quantum, signal: *std.Io.Semaphore) bool {
    const started = blk: {
        self.makeExecutor(i) catch |err| break :blk err;
        const exec = self.rows.items(.executor)[i].?;
        if (self.stream_opener) |o| if (exec.stream == null and exec.peers.len == 0 and
            self.rows.items(.status)[i] == .pending and self.rows.items(.output)[i] != no_output and
            self.rows.items(.owner)[i] == none)
        {
            exec.stream = o.open(o.ctx, @fromBackingInt(@intCast(i)), self.jobAt(i));
        };
        exec.start(quantum, signal) catch |err| break :blk err;
        self.rows.items(.running)[i] = true;
        return true;
    };
    const first = self.rows.items(.status)[i] == .pending;
    self.rows.items(.status)[i] = .failed;
    self.rows.items(.failure)[i] = started;
    if (first) self.readerDone(i);
    self.laneDone(i);
    return false;
}

/// Records the quantum of running query `i`, which has stopped. Under
/// `Config.final_plan`, a completed query's circuit, or its source's, is
/// released once no dependent still has to copy it.
pub fn settle(self: *Session, i: usize) void {
    assert(self.rows.items(.running)[i]);
    self.rows.items(.running)[i] = false;
    const first = self.rows.items(.status)[i] == .pending;
    // A started worker always has an outcome; `NotStarted` cannot happen.
    const outcome = self.rows.items(.executor)[i].?.wait() catch unreachable;
    switch (outcome) {
        .progress => |p| {
            self.rows.items(.status)[i] = .paused;
            self.rows.items(.progress)[i] = p;
        },
        .complete => self.rows.items(.status)[i] = .complete,
        .failed => |err| {
            self.rows.items(.status)[i] = .failed;
            self.rows.items(.failure)[i] = err;
        },
        .cancelled => self.rows.items(.status)[i] = .cancelled,
    }
    if (first) self.readerDone(i);
    self.releaseIfRead(i);
    self.laneDone(i);
    self.settlePeers(i);
}

/// Assembles the split query lane row `i` belongs to once `i`, the last
/// of its lanes to end, has.
pub fn laneDone(self: *Session, i: usize) void {
    const owner = self.rows.items(.owner)[i];
    if (owner == none or !self.rows.items(.status)[i].terminal()) return;
    const g = @backingInt(owner);
    for (g - self.rows.items(.lanes)[g]..g) |l| if (!self.rows.items(.status)[l].terminal()) return;
    self.assemble(g) catch |err| {
        self.rows.items(.status)[g] = .failed;
        self.rows.items(.failure)[g] = err;
    };
}

/// Ends split query `g`, whose lanes all ended: as its first lane (in
/// lane order) that did not complete, else complete with the lanes'
/// rows in lane order, as the query run whole publishes them.
pub fn assemble(self: *Session, g: usize) !void {
    const first = g - self.rows.items(.lanes)[g];
    var npoints: usize = 0;
    for (first..g) |l| {
        const status = self.rows.items(.status)[l];
        if (status != .complete) {
            self.rows.items(.status)[g] = status;
            self.rows.items(.failure)[g] = self.rows.items(.failure)[l];
            return;
        }
        npoints += self.rows.items(.executor)[l].?.result().?.npoints;
    }
    const job = self.jobAt(g);
    const e = try execution.Executor.create(self.allocator, self.io, self.templateOf(g), self.deck, job, null, false, self.config);
    self.rows.items(.executor)[g] = e;
    var res = self.rows.items(.executor)[first].?.result().?;
    const ncols = res.varnames.len;
    const data = try e.results.allocator().alloc(f64, npoints * ncols);
    var at: usize = 0;
    for (first..g) |l| {
        const part = self.rows.items(.executor)[l].?.result().?.data;
        @memcpy(data[at..][0..part.len], part);
        at += part.len;
    }
    // A plain `.mc` numbers its converged trials.
    if (job == .mc and job.mc.variants.count == 0) {
        for (0..npoints) |r| data[r * ncols] = @floatFromInt(r);
    }
    res.data = data;
    res.npoints = npoints;
    e.published = res;
    self.rows.items(.status)[g] = .complete;
}

/// Query `i` no longer reads its source's state.
pub fn readerDone(self: *Session, i: usize) void {
    const dep = self.rows.items(.dependency)[i];
    if (dep == none) {
        self.root_readers -= 1;
    } else {
        self.rows.items(.readers)[@backingInt(dep)] -= 1;
        self.releaseIfRead(@backingInt(dep));
    }
}

/// Under a final plan nothing reads a finished query's circuit once
/// its dependents have copied it; its result lives elsewhere.
pub fn releaseIfRead(self: *Session, i: usize) void {
    if (self.final(i) and self.rows.items(.status)[i] == .complete and self.rows.items(.readers)[i] == 0)
        self.rows.items(.executor)[i].?.circuit.release();
}

/// Advances each of `ids` by one quantum, at most `limits.max_parallel` at
/// a time, starting the next as soon as one stops, and writes one event
/// per id. Every id must be ready and unique;
/// that is checked for all of them before any runs: fails with
/// `error.InvalidConcurrency` for a zero `max_parallel`,
/// `error.QueryNotReady`, `error.DuplicateQuery`, or
/// `error.BufferTooSmall` when `events` is shorter than `ids`. A query
/// whose executor cannot be built or started is marked failed, not
/// returned as an error. Under `Config.final_plan`, a completed query's
/// circuit is released once no dependent still has to snapshot it.
pub fn advanceReady(self: *Session, ids: []const QueryId, limits: Limits, events: []Advance) !usize {
    try self.validateReady(ids, limits);
    if (events.len < ids.len) return error.BufferTooSmall;
    if (ids.len == 0) return 0;
    // Positions into `ids` of the running queries.
    const running = try self.allocator.alloc(usize, @min(ids.len, limits.max_parallel));
    defer self.allocator.free(running);
    var signal: std.Io.Semaphore = .{};
    var n_running: usize = 0;
    var next: usize = 0;
    while (true) {
        while (n_running < running.len and next < ids.len) : (next += 1) {
            if (self.launch(@backingInt(ids[next]), limits.quantum, &signal)) {
                running[n_running] = next;
                n_running += 1;
            } else try self.event(ids[next], &events[next]);
        }
        if (n_running == 0) break;
        signal.waitUncancelable(self.io);
        var k: usize = 0;
        while (k < n_running) {
            const at = running[k];
            const i = @backingInt(ids[at]);
            if (!self.rows.items(.executor)[i].?.stopped()) {
                k += 1;
                continue;
            }
            self.settle(i);
            try self.event(ids[at], &events[at]);
            n_running -= 1;
            running[k] = running[n_running];
        }
    }
    self.cursor = @intCast((@backingInt(ids[ids.len - 1]) + 1) % self.rows.len);
    return ids.len;
}

pub fn event(self: *const Session, id: QueryId, out: *Advance) !void {
    const q = try self.info(id);
    out.* = .{ .requested = id, .advanced = id, .status = q.status, .target_status = q.status, .progress = q.progress, .failure = q.failure };
}

/// Queues every query that is ready now for `Pool.fill`, in row order;
/// queries that turn ready later join as they do.
pub fn requeue(self: *Session) !void {
    self.queue.clearRetainingCapacity();
    self.urgent.clearRetainingCapacity();
    self.queue_head = 0;
    // A row is queued in `urgent` at most once at a time.
    try self.urgent.ensureTotalCapacity(self.allocator, self.rows.len);
    self.advanceDone();
    for (self.done_rows..self.rows.len) |i| if (self.ready(i) and !self.rows.items(.running)[i]) {
        try self.queue.append(self.allocator, @intCast(i));
    };
}

/// The next query `Pool.fill` starts: a parked query or a finished
/// OP's dependent first, so each point's OP and analysis run back to
/// back, then the queue in row order.
pub fn nextReady(self: *Session) ?usize {
    // Breadth-first under GPU lanes: OPs drain before their dependents,
    // so the analyses become ready together and gather N wide.
    if (!self.config.gpu_lanes) while (self.urgent.pop()) |i| if (self.ready(i) and !self.rows.items(.running)[i]) return i;
    while (self.queue_head < self.queue.items.len) {
        const i = self.queue.items[self.queue_head];
        self.queue_head += 1;
        if (self.ready(i) and !self.rows.items(.running)[i]) return i;
    }
    if (self.config.gpu_lanes) {
        // A laned analysis waits while anything still runs, so the rest of
        // its sweep's OPs finish and it gathers with all of them.
        // ponytail: one scan of `running` per call; count it if a sweep
        // of 100k points makes the scan show.
        const top = self.urgent.getLastOrNull() orelse return null;
        const laned = self.gpuLaned(top);
        if (laned and std.mem.indexOfScalar(bool, self.rows.items(.running), true) != null) return null;
        // The batch leader is the lowest ready row, not the last OP to finish.
        if (laned) std.mem.sortUnstable(u32, self.urgent.items, {}, std.sort.desc(u32));
        while (self.urgent.pop()) |i| if (self.ready(i) and !self.rows.items(.running)[i]) return i;
    }
    return null;
}

/// Queues for `nextReady` what settled query `i` (and its batch's
/// peers) made ready: itself when parked, its dependents when complete.
pub fn follow(self: *Session, i: usize) void {
    for (self.rows.items(.executor)[i].?.peers) |p| self.followOne(p.row);
    self.followOne(i);
}

pub fn followOne(self: *Session, i: usize) void {
    switch (self.rows.items(.status)[i]) {
        .paused => self.urgent.appendAssumeCapacity(@intCast(i)),
        .complete => {
            var c = self.rows.items(.child)[i];
            while (c != none) : (c = self.rows.items(.sibling)[@backingInt(c)]) {
                const j = @backingInt(c);
                if (self.ready(j) and !self.rows.items(.running)[j]) self.urgent.appendAssumeCapacity(j);
            }
        },
        else => {},
    }
}

/// Runs every query to its end through a pool of `limits.max_parallel`.
/// Fails with `error.SchedulingFailure` when queries remain that can
/// never become ready.
pub fn runAll(self: *Session, limits: Limits) !void {
    var pool: Pool = try .init(self.allocator, self.io, limits);
    defer pool.deinit();
    try pool.add(self);
    while (true) {
        pool.fill(self);
        if (!pool.wait()) break;
    }
    if (!self.finished()) return error.SchedulingFailure;
}

/// Advances `target`, or its prerequisite first, by one quantum. A
/// terminal target returns its status without running anything. Fails
/// as `advanceReady` does.
pub fn advance(self: *Session, target: QueryId) !Advance {
    var events: [1]Advance = undefined;
    if ((try self.info(target)).status.terminal()) {
        try self.event(target, &events[0]);
        return events[0];
    }
    // The first ready query in scheduler order: the prerequisite, the
    // target, or one of a split target's lanes.
    const next: QueryId = for (0..self.rows.len) |offset| {
        const i = (offset + self.cursor) % self.rows.len;
        if (self.contains(.{ .query = target }, i) and self.ready(i)) break @fromBackingInt(@intCast(i));
    } else return error.SchedulingFailure;
    _ = try self.advanceReady(&.{next}, .{}, &events);
    events[0].requested = target;
    events[0].target_status = (try self.info(target)).status;
    return events[0];
}

/// Moves `done_rows` and `done_outputs` past the rows and outputs that
/// have ended; a terminal row stays terminal. O(rows ended since).
pub fn advanceDone(self: *Session) void {
    while (self.done_rows < self.rows.len and self.effectiveStatus(self.done_rows).terminal()) self.done_rows += 1;
    const outputs = self.outputs.items;
    while (self.done_outputs < outputs.len and self.effectiveStatus(@backingInt(outputs[self.done_outputs])).terminal()) self.done_outputs += 1;
}
