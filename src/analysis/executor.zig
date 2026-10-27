//! Executor: one query's mutable numerical state and the worker thread that
//! runs it. Prepared topology and the deck are borrowed; an accepted OP from a
//! dependency is copied before use, or moved under `Config.final_plan`.
//! Dispatches to the analysis leaves, with the query's ParEval, GPU context
//! and device LU attached for the run. Cites VAMS LRM Table 4-22 and
//! ngspice osdiload.c (the small-signal `analysis()` kind), §9.4 display
//! tasks, #13.
const std = @import("std");
const Circuit = @import("device").Circuit;
const Deck = @import("core").Deck;
const requests = @import("core").query;
const types = @import("types.zig");
const op = @import("dc/op.zig");
const sweep_lanes = @import("sweep/lanes.zig");
const tran = @import("tran/tran.zig");
const gpu = @import("gpu/context.zig");
const gpu_lu = @import("gpu/lu.zig");
const ParEval = @import("par_eval.zig").ParEval;
const Controller = @import("worker.zig").Worker(QueryError, types.Result);
const Quantum = @import("worker.zig").Quantum;
const converger = @import("solver").converger;

/// Why a started query ended without a result: whatever an analysis leaf
/// or its setup in `runQuery` can return (`error.QueryCancelled` for a
/// cancel, `error.OutOfMemory`, the numerical failures such as
/// `error.TimestepTooSmall`), or `error.DeviceRefused` when a device latched
/// `$fatal`/`$error`. Names are part of the API: `c_api.zig` maps some to
/// status codes.
// Spelled from runQuery's parts, not `ErrorOf(Executor.runQuery)`: the
// executor holds its worker, whose type needs this set, so naming its own
// method's inferred set is a dependency loop. A part missing here fails
// to compile at `execute`.
pub const QueryError = types.ErrorOf(run) || types.ErrorOf(op.solve) || types.ErrorOf(ParEval.init) ||
    gpu.InitError || types.ErrorOf(gpu.GpuContext.syncHostState) || types.ErrorOf(types.Circuit.workspace) ||
    std.mem.Allocator.Error || error{ OpDidNotConverge, DeviceRefused, QueryCancelled } ||
    // `materialize`'s parts.
    types.ErrorOf(types.Circuit.fromSnapshot) || types.ErrorOf(types.Circuit.fromSnapshotMove) ||
    types.ErrorOf(types.Circuit.instantiate) || types.ErrorOf(types.Circuit.instantiateMove) ||
    types.ErrorOf(types.Circuit.recompute) || types.ErrorOf(types.Circuit.collectParams) || types.ErrorOf(types.Circuit.setParams) || error{InvalidAcOverride};

/// Per-problem execution settings, shared by every query.
pub const Config = struct {
    /// The `--backend` choices: `auto`, `cpu`, `cuda`, `hip`.
    pub const Backend = gpu.Request;
    backend: gpu.Request = .cpu,
    /// The user named the GPU: bypass the work gate, and under `auto` fail
    /// rather than fall back when the machine cannot run it. `cuda` and `hip`
    /// imply it and never fall back (`gpu.fallsBack`).
    gpu_explicit: bool = false,
    /// The caller left the backend at its default `auto` (the CLI in a
    /// build with kernels): a query the GPU declines runs on the CPU
    /// without a warning, since the user asked for nothing.
    gpu_quiet: bool = false,
    /// Solver threads; above 1 it also fixes the multicore refactor's
    /// lanes, which otherwise follow `device_threads` (`materialize`).
    solver_threads: u8 = 1,
    /// Device-stamp lanes; above 1 each query builds a ParEval.
    device_threads: u32 = 1,
    /// Extra lanes a large query may borrow for its device stamp and
    /// multicore refactor when the caller fixed no thread count; null
    /// keeps `device_threads` and `solver_threads` as given.
    thread_pool: ?*ThreadPool = null,
    /// Reusable query threads (`worker.Threads`); null spawns one per query.
    threads: ?*@import("worker.zig").Threads = null,
    /// Workspaces shared by the Problem's queries (`ws_pool.WsPool`).
    ws_pool: ?*@import("ws_pool.zig").WsPool = null,
    /// Print per-query setup and per-checkpoint timing to stderr.
    timing_in_depth: bool = false,
    /// The caller adds no query after the session is built (the CLI), so a
    /// completed prerequisite's device state may move into its last
    /// dependent instead of being copied. Appending callers keep copies.
    final_plan: bool = false,
    /// Developer switches the facade read from the environment.
    knobs: types.Knobs = .{},
    /// `--lanes`: batch ready operating points, `.dc` sweeps, transients,
    /// shooting `.pss` and the AC family as lanes (`lanesOn`), each bitwise
    /// its query run alone.
    lanes: bool = false,
    /// Batch ready transients as GPU lanes (dev/research/plans/phase4):
    /// decided once per Problem by `gpu_lanes.policy`, true for an explicit
    /// `cuda`/`hip` and, under `auto`, on an FP64-class device only.
    gpu_lanes: bool = false,
    /// False when the deck has nothing to batch (one transient, no runs or
    /// tuning): its operating point is alone and may take the device LU
    /// under `gpu_lanes` too.
    may_batch: bool = true,
    /// Lanes per GPU batch at most.
    gpu_lane_cap: u32 = 16384,
    /// The share of a GPU lane batch's points the device keeps; the CPU
    /// workers run the rest beside it (`Session.shareWithHost`, 05 5.3b).
    /// The facade sets it from `gpu_lanes.share`; 1 keeps every point.
    gpu_share: f64 = 1,
    /// Operating points and `.dc` sweeps batch as GPU lanes too (06 6.9);
    /// the facade sets it from `gpu_lanes.staticPolicy` under `gpu_lanes`.
    gpu_static: bool = false,

    /// The facade's record of whether the GPU driver is up, shared by every
    /// query of the Problem; null prices the driver setup in every query.
    gpu_driver: ?*gpu.Driver = null,
    /// Where `.hdl` display-task lines go; null is the process's stdout.
    /// Each accepted point's lines arrive in one `writeAll` from the thread
    /// running the query, so concurrent queries write concurrently.
    display: ?*std.Io.Writer = null,
};

/// Fails with `error.GpuBackendUnavailable` for a backend this binary has no
/// kernels for (printing the GPU hardware the probe detected) and
/// `error.InvalidThreadCount` for a zero thread count. A machine without the
/// GPU fails later, when the first query brings up the driver.
pub fn validateBackend(config: Config) !void {
    if (!gpu.requestSupported(config.backend)) {
        const name = @tagName(config.backend);
        // The library's diagnostic channel, which an embedder routes with
        // its own `std_options.logFn`; the CLI's prints to stderr (#13).
        std.log.warn("--backend {s}: this build has no {s} kernels; detected GPU hardware: {s}", .{ name, name, gpu.detectedHardware() });
        return error.GpuBackendUnavailable;
    }
    if (config.solver_threads == 0 or config.device_threads == 0) return error.InvalidThreadCount;
}

/// Threads a Problem lends its queries beyond each worker's own. A query
/// whose device work or matrix is large enough to gain from more lanes
/// (`borrowLanes`) takes up to `per_query - 1` of them for its run and
/// gives them back when it ends, so `--jobs` workers plus the pool bound
/// the threads in use: one big transient gets the machine, a sweep of
/// small points stays one thread each.
pub const ThreadPool = struct {
    free: std.atomic.Value(u32),

    /// Most lanes one query runs. ParEval measured flat or worse past 8
    /// on a hybrid CPU (dev/analysis/refactoring.md, #138).
    pub const per_query: u32 = 8;

    /// Half the logical CPUs (SMT siblings share a core's FP units), at
    /// least none.
    pub fn init(cpus: usize) ThreadPool {
        return .{ .free = .init(@intCast(@min(cpus / 2, std.math.maxInt(u16)))) };
    }

    /// Up to `want` threads, fewer when others hold them; never blocks.
    pub fn take(pool: *ThreadPool, want: u32) u32 {
        var cur = pool.free.load(.monotonic);
        while (true) {
            const got = @min(cur, want);
            if (got == 0) return 0;
            cur = pool.free.cmpxchgWeak(cur, cur - got, .monotonic, .monotonic) orelse return got;
        }
    }

    /// Returns `n` threads a `take` handed out.
    pub fn give(pool: *ThreadPool, n: u32) void {
        if (n != 0) _ = pool.free.fetchAdd(n, .monotonic);
    }
};

/// Device weight (`ParEval`'s per-instance stamp estimate summed over the
/// circuit) from which extra lanes pay: about 800 BSIM4 instances. ParEval
/// measured 3.3x on 4,000 bsim4 at 8 threads and no gain on 4,000 mos1,
/// whose summed weight sits below this (`gpu/context.zig` cost notes).
const borrow_min_weight: u64 = 4_000_000;
/// Matrix entries from which a refactor may be thick enough for the
/// multicore LU; its own per-epoch admission (`direct.zig`) still decides.
const borrow_min_nnz: u32 = 50_000;

/// Lanes for a query of `ckt`: 1 below both gates.
fn wantLanes(ckt: *const types.Circuit) u32 {
    var weight: u64 = 0;
    for (ckt.batches) |b| weight += @as(u64, b.count) * @import("par_eval.zig").instanceWeight(b);
    return if (weight >= borrow_min_weight or ckt.nnz >= borrow_min_nnz) ThreadPool.per_query else 1;
}

/// Lanes per batch: the CPU's f64 vector width (4 on AVX2, 8 on AVX-512).
pub const lane_width = std.simd.suggestVectorLength(f64) orelse 4;

/// Whether `Session` groups ready queries into lane batches:
/// `ESPICE_LANES` when set, else `Config.lanes`. Off by default on every
/// backend: on the CPU lanes measured no faster (#211).
pub fn lanesOn(config: Config) bool {
    return config.knobs.lanes orelse config.lanes;
}

/// Most points `Session` runs together as one lane batch under `config`:
/// `gpu_lane_cap` under GPU lanes, `lane_width` under CPU lanes (as
/// `Session.gather` admits them), else 1. Advisory: batching never changes
/// a point's result.
pub fn batchWidth(config: Config) u32 {
    if (config.gpu_lanes) return config.gpu_lane_cap;
    if (lanesOn(config) and config.backend == .cpu and config.device_threads == 1 and !config.timing_in_depth) return lane_width;
    return 1;
}

/// The AC-family kinds a lane batch runs (`runFreqLanes`): small-signal
/// analyses at their own operating point, each lane bitwise its query run
/// alone (lane-batched-simulation.md §6, the AC family rows).
pub fn freqLaneKind(kind: requests.Kind) bool {
    return switch (kind) {
        .ac, .noise, .sp, .stb, .lstb, .acxf, .tf, .dcxf, .dcinc, .sens, .dcmatch, .dcsens, .acmatch => true,
        else => false,
    };
}

/// Sweeps with a GPU (point, frequency) provider. LSTB margins keep their
/// live solvers across the leading sweep and crossover refinements.
pub fn gpuFreqLaneKind(kind: requests.Kind) bool {
    return switch (kind) {
        .ac, .noise, .sp, .stb, .lstb, .acxf => true,
        else => false,
    };
}

/// The analysis state a query's devices see while it runs (`runQuery`).
/// Small-signal linearizations run as analysis("ac") or ("noise"), LRM
/// Table 4-22, as ngspice's MODEINITSMSIG load sets ANALYSIS_AC for ac,
/// noise, pz and disto (osdiload.c:165).
fn simKind(job: requests.Query) SimKind {
    const transient = if (job == .op) job.op.tran_op else @as(requests.Kind, job).transient();
    return switch (job) {
        .ac, .sp, .stb, .pz, .disto, .lstb, .acxf => .ac,
        .noise => .noise,
        else => if (transient) .ic else .dc,
    };
}

const SimKind = @FieldType(@import("device").abi.SimState, "kind");

/// One query: its own Circuit instance, work and result arenas, and worker.
/// Heap-pinned because the worker and the progress callback hold its address.
pub const Executor = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    topology: *const Circuit,
    deck: *const Deck,
    job: requests.Query,
    config: Config,
    circuit: types.Circuit,
    work: std.heap.ArenaAllocator,
    results: std.heap.ArenaAllocator,
    controller: Controller,
    x: ?[]f64 = null,
    published: ?types.Result = null,
    /// What `materialize` builds `circuit` from (`create`).
    initial: ?*Executor,
    take: bool,
    /// `circuit` holds a built circuit that `destroy` frees.
    built: bool = false,
    /// `RunCtx.stream` for this query.
    stream: ?*std.Io.Writer = null,
    /// Queries this one runs as the other lanes of its batch, on its own
    /// worker (`Session` groups them); empty for a lone query.
    peers: []const *Executor = &.{},
    /// The session row, for settling a batch's peers.
    row: u32 = 0,
    /// A peer's failure, set by its batch; `published` holds its success.
    peer_failure: ?QueryError = null,
    /// No query copies this one's circuit once it completes (a final plan
    /// and no readers), so a GPU batch frees it with its wave.
    no_readers: bool = false,

    /// What one `wait` saw: a checkpoint, the result, an error or a cancel.
    pub const Outcome = Controller.Outcome;

    /// Builds the executor without starting it; its circuit is built on
    /// the worker thread when the first quantum starts (`materialize`), so
    /// the coordinator only books the query. `initial`, when given, must be
    /// a completed `.op` query over the same topology; its device state and
    /// operating point seed this one, and the caller keeps them unchanged
    /// until this query's first quantum stops. With `take`, this is the last
    /// query that will read `initial` (or, without one, the template), so
    /// its device state moves here instead of being copied, and `initial`'s
    /// circuit may be left empty: its result and operating point stay
    /// readable, but nothing may evaluate it again. Fails with
    /// `error.InvalidOperatingPoint` when `initial` is not such a query and
    /// with `validateBackend`'s errors. The run fails with
    /// `error.InvalidAcOverride` when an AC override names no device
    /// parameter, and with `error.TopologyChanged` when the query's variant
    /// or temperature re-wires a device. Caller frees it with `destroy`.
    pub fn create(allocator: std.mem.Allocator, io: std.Io, topology: *const Circuit, deck: *const Deck, job: requests.Query, initial: ?*Executor, take: bool, config: Config) !*Executor {
        try validateBackend(config);
        if (initial) |source| {
            if (source.topology != topology or source.operatingPoint() == null)
                return error.InvalidOperatingPoint;
        }
        const self = try allocator.create(Executor);
        self.* = .{
            .allocator = allocator,
            .io = io,
            .topology = topology,
            .deck = deck,
            .job = job,
            .config = config,
            .circuit = undefined,
            .work = std.heap.ArenaAllocator.init(allocator),
            .results = std.heap.ArenaAllocator.init(allocator),
            .controller = undefined,
            .initial = initial,
            .take = take,
        };
        self.controller = Controller.init(io, self, execute, .{ .report_nonlinear = job == .op, .threads = config.threads });
        return self;
    }

    /// Whether this query runs at another temperature than the nominal
    /// point, whose operating point then sits too far away for a warm start
    /// to land where the cold ladder does (variants/step_temp_source.sp).
    fn movesTemp(self: *const Executor) bool {
        if (queryTemp(self.job) != null) return true;
        const v = queryVariant(self.job) orelse return false;
        return self.deck.variants.tempAt(v) != null;
    }

    /// The circuit copy, variant and AC overrides `create` deferred, built
    /// on the worker thread before the analysis runs.
    fn materialize(self: *Executor) !void {
        const started = if (self.controller.options.timing_query != null) std.Io.Timestamp.now(self.io, .awake) else null;
        const topology = self.topology;
        const allocator = self.allocator;
        self.circuit = if (self.initial) |source|
            if (self.take)
                try types.Circuit.fromSnapshotMove(topology, &source.circuit, allocator)
            else
                try types.Circuit.fromSnapshot(topology, &source.circuit, allocator)
        else if (self.take)
            try types.Circuit.instantiateMove(topology, allocator)
        else
            try types.Circuit.instantiate(topology, allocator);
        self.built = true;
        // The multicore refactor follows the device threads unless the
        // solver's own count is set (ESPICE_SOLVER_THREADS).
        const config = self.config;
        const lu_threads: u8 = @intCast(@min(if (config.solver_threads > 1) config.solver_threads else config.device_threads, 16));
        self.circuit.configure(.{ .io = self.io, .threads = config.solver_threads, .lu_threads = lu_threads }, config.knobs);
        self.circuit.ws_pool = config.ws_pool;
        if (self.initial) |source| self.x = try self.work.allocator().dupe(f64, source.operatingPoint().?);
        // A dependent copies its prerequisite's circuit, variant included.
        if (self.initial == null) try self.installVariant();
        try self.bindAcOverrides();
        self.circuit.progress = self.controller.callback();
        if (started) |t0| {
            const elapsed = t0.durationTo(std.Io.Timestamp.now(self.io, .awake)).nanoseconds;
            std.debug.print("timing: query {d} {s} setup: {d:.6}ms\n", .{
                @backingInt(self.controller.options.timing_query.?), @tagName(self.job), @as(f64, @floatFromInt(elapsed)) / 1e6,
            });
        }
    }

    /// Writes the query's variant parameters and temperature, then
    /// re-derives the devices once: every batch under a temperature, else
    /// only the batches the variant writes. A nominal query at the deck
    /// temperature touches nothing.
    fn installVariant(self: *Executor) !void {
        const variant = queryVariant(self.job);
        const variant_temp = if (variant) |v| self.deck.variants.tempAt(v) else null;
        const temp = queryTemp(self.job) orelse variant_temp orelse self.deck.deck_temp;
        if (variant) |v| {
            const refs, const values = self.deck.variants.writes(v);
            if (refs.len != 0) try self.circuit.setParams(refs, values, temp == null);
        }
        if (temp) |t| {
            self.circuit.setCircuitTemp(@floatCast(t));
            try self.circuit.recompute();
        }
    }

    fn bindAcOverrides(self: *Executor) !void {
        if (self.deck.ac_overrides.len == 0) return;
        const params = try self.circuit.collectParams();
        const mapped = try self.work.allocator().alloc(types.AcParam, self.deck.ac_overrides.len);
        for (self.deck.ac_overrides, mapped) |override, *target| {
            for (params) |param| {
                if (param.index == override.index and !param.is_instance and
                    param.type == override.type and
                    std.mem.eql(u8, param.param_name, override.param_name))
                {
                    target.* = .{ .ptr = param, .ac_value = override.value };
                    break;
                }
            } else return error.InvalidAcOverride;
        }
        self.circuit.ac_params = mapped;
    }

    /// Runs the query up to where `quantum` lets it stop, without waiting.
    /// `signal`, when given, is posted once when the quantum stops.
    pub fn start(self: *Executor, quantum: Quantum, signal: ?*std.Io.Semaphore) !void {
        try self.controller.start(quantum, signal);
    }

    /// Waits for the quantum `start` began and publishes a completed result.
    pub fn wait(self: *Executor) !Outcome {
        const outcome = try self.controller.wait();
        if (outcome == .complete) self.published = outcome.complete;
        return outcome;
    }

    /// Whether the started quantum has stopped, so `wait` will not block.
    pub fn stopped(self: *Executor) bool {
        return self.controller.stopped();
    }

    /// Asks the worker to stop at its next checkpoint; see `Worker.cancel`.
    pub fn cancel(self: *Executor) void {
        self.controller.cancel();
    }

    /// Returns the completed result, valid until `destroy`.
    pub fn result(self: *const Executor) ?types.Result {
        return self.published;
    }

    /// Returns the solved operating point of a completed `.op` query, null
    /// for any other query. Valid until `destroy`, even after the circuit
    /// is released.
    pub fn operatingPoint(self: *const Executor) ?[]const f64 {
        return if (self.job == .op and self.published != null) self.x else null;
    }

    /// Cancels and joins the worker, then frees everything the query owns.
    pub fn destroy(self: *Executor) void {
        self.controller.deinit();
        if (self.built) self.circuit.deinit();
        self.work.deinit();
        self.results.deinit();
        self.allocator.destroy(self);
    }

    /// Runs the query, then refuses it if any device latched a VerA
    /// `$fatal`/`$error` (`Hooks.status`), whether the analysis failed on it
    /// or not: a latched device stamps zero rows, which can still solve.
    // ponytail: checked once per query, not per eval; a refused device
    // costs the analysis's own failure path first. Check per Newton
    // iterate if that turns out slow.
    fn execute(ctx: *anyopaque) QueryError!types.Result {
        const self: *Executor = @ptrCast(@alignCast(ctx));
        if (self.peers.len != 0) return self.executeBatch();
        try self.materialize();
        return self.finishAlone();
    }

    /// Runs this query and its `peers` as one lane batch: each builds its
    /// circuit, then the operating points (`op.solveLanes`), `.dc` sweeps
    /// (`dc.runLanes`) or transients (`tran.runLanes`) run as lanes and each publishes as it would alone. A
    /// batch whose devices print display tasks runs its queries one after
    /// another instead. Every peer ends with `published` or `peer_failure`;
    /// returns this query's own.
    fn executeBatch(self: *Executor) QueryError!types.Result {
        if (self.config.gpu_lanes and (self.job == .tran or self.job == .pss or self.job == .hb or
            (self.job == .lstb and self.job.lstb.margins) or (self.config.gpu_static and (self.job == .op or self.job == .dc)))) return self.executeGpuBatch();
        if (self.config.gpu_lanes and gpuFreqLaneKind(self.job)) return self.executeGpuFreqBatch();
        const g = 1 + self.peers.len;
        std.debug.assert(g <= lane_width and (self.job == .op or self.job == .dc or self.job == .mc or self.job == .temp or self.job == .tran or self.job == .pss or freqLaneKind(self.job)));
        var members: [lane_width]*Executor = undefined;
        members[0] = self;
        @memcpy(members[1..g], self.peers);
        var outcomes: [lane_width]QueryError!types.Result = @splat(error.QueryCancelled);
        defer publishPeers(members[0..g], outcomes[0..g]);
        // Lanes whose circuit built, as indices into `members`.
        var live: [lane_width]usize = undefined;
        var n_live: usize = 0;
        for (members[0..g], 0..) |e, k| {
            e.materialize() catch |err| {
                outcomes[k] = err;
                continue;
            };
            e.circuit.progress = self.controller.callback();
            live[n_live] = k;
            n_live += 1;
        }
        if (n_live != 0 and (try members[live[0]].sayNames()).len != 0) {
            for (live[0..n_live]) |k| outcomes[k] = members[k].finishAlone();
            return outcomes[0];
        }
        if (freqLaneKind(self.job)) {
            runFreqLanes(members[0..g], live[0..n_live], outcomes[0..g]);
            return outcomes[0];
        }
        switch (self.job) {
            // Every laned analysis past the OP but `.tran` (its `uic` start,
            // below): each lane runs from its own circuit and context, and
            // publishes as it would alone.
            inline .dc, .pss => |opts, tag| {
                const M = module(tag);
                var ctxs: [lane_width]types.RunCtx = undefined;
                laneCtxs(members[0..g], live[0..n_live], &ctxs);
                var swept: [lane_width]M.LaneOutcome = undefined;
                if (n_live != 0) M.runLanes(lane_width, ctxs[0..n_live], opts, swept[0..n_live]);
                for (live[0..n_live], swept[0..n_live]) |k, o|
                    outcomes[k] = members[k].settleLabeled(o);
                return outcomes[0];
            },
            // `.mc`/`.temp` lane rows (#213), each its own lane (`Session.gather`).
            inline .mc, .temp => |_, tag| {
                const M = module(tag);
                var ctxs: [lane_width]types.RunCtx = undefined;
                laneCtxs(members[0..g], live[0..n_live], &ctxs);
                var opts: [lane_width]M.Options = undefined;
                for (live[0..n_live], opts[0..n_live]) |k, *o| o.* = @field(members[k].job, @tagName(tag));
                var swept: [lane_width]sweep_lanes.LaneOutcome = undefined;
                if (n_live != 0) sweep_lanes.runBatch(lane_width, M, ctxs[0..n_live], opts[0..n_live], swept[0..n_live]);
                for (live[0..n_live], swept[0..n_live]) |k, o|
                    outcomes[k] = members[k].settleLabeled(o);
                return outcomes[0];
            },
            else => {},
        }
        if (self.job == .tran) {
            // `runQuery`'s transient start: the `uic` x when no operating
            // point ran.
            for (live[0..n_live]) |k| if (members[k].job.uic()) members[k].seedUic() catch |err| {
                for (live[0..n_live]) |j| outcomes[j] = err;
                return outcomes[0];
            };
            var ctxs: [lane_width]types.RunCtx = undefined;
            laneCtxs(members[0..g], live[0..n_live], &ctxs);
            var ran: [lane_width]tran.LaneOutcome = undefined;
            if (n_live != 0) tran.runLanes(lane_width, ctxs[0..n_live], self.job.tran, ran[0..n_live]);
            for (live[0..n_live], ran[0..n_live]) |k, o|
                outcomes[k] = members[k].settleLabeled(o);
            return outcomes[0];
        }
        var ckts: [lane_width]*types.Circuit = undefined;
        var xs: [lane_width][]f64 = undefined;
        var options: op.Options = undefined;
        for (live[0..n_live], 0..) |k, l| {
            const e = members[k];
            e.circuit.setSimState(.{ .kind = simKind(e.job) });
            options = e.opStart() catch |err| {
                for (live[0..n_live]) |j| outcomes[j] = err;
                return outcomes[0];
            };
            ckts[l] = &e.circuit;
            xs[l] = e.x.?;
        }
        var solved: [lane_width]op.LaneOutcome = undefined;
        if (n_live != 0) op.solveLanes(lane_width, ckts[0..n_live], xs[0..n_live], options, self.deck.nodeset, self.deck.ic, solved[0..n_live]) catch |err| {
            for (live[0..n_live]) |k| outcomes[k] = err;
            return outcomes[0];
        };
        for (live[0..n_live], solved[0..n_live]) |k, o| {
            const e = members[k];
            outcomes[k] = e.settle(blk: {
                const r = o catch |err| break :blk err;
                if (!r.converged) break :blk error.OpDidNotConverge;
                e.circuit.setSimState(.{ .kind = simKind(e.job) });
                break :blk e.finish();
            });
        }
        return outcomes[0];
    }

    /// Host bytes one GPU lane holds while its wave runs, estimated from the
    /// circuit's size: its materialized circuit, its run and step machine (about
    /// 20 n-vectors and the charge rings) and its solver's factors. Measured at
    /// about 2.3 KB per unknown on a 72-unknown deck; this errs high.
    fn gpuLaneBytes(topology: *const Circuit) usize {
        return 4096 * (@as(usize, topology.n) + @as(usize, topology.nnz) / 4) + 64 * 1024;
    }

    /// Device bytes one GPU lane holds in `GpuLanes.init`, over-estimated
    /// where its sizes are known only later: 36 n-vectors (the planes and
    /// iterates, gear 6's history ring, a probe ring over every unknown), 6
    /// nnz-planes (G, C, A and an LU fill of three times nnz), each batch's
    /// packed rows and 4 KiB of per-lane scalars and reports.
    // ponytail: the transient lane state only; HB's harmonic planes are not
    // counted.
    fn gpuLaneDeviceBytes(topology: *const Circuit) usize {
        var bytes: usize = 8 * (36 * (@as(usize, topology.n) + 1) + 6 * (@as(usize, topology.nnz) + 1)) + 4096;
        for (topology.batches) |b| if (b.hooks.gpu_payload) |payload| {
            const p = payload(b.ctx);
            bytes += p.models.len + 4 * p.model_of.len + p.instances.len + p.states.len + 8 * p.lim_x.len;
        };
        return bytes;
    }

    /// The host bytes a GPU batch's wave may hold (`gpuLaneBytes` each).
    // ponytail: a fixed 4 GiB; size it from the machine's memory if decks
    // with tens of thousands of large lanes need more per wave.
    const gpu_wave_bytes: usize = 4 << 30;
    /// Fewest lanes per wave, so a huge circuit still batches.
    const gpu_wave_min: usize = 256;

    /// `executeBatch` for a transient, periodic or live LSTB GPU batch: `peers` may
    /// be thousands wide, so the per-member arrays come from the work arena.
    fn executeGpuBatch(self: *Executor) QueryError!types.Result {
        const members, const outcomes = try self.gpuMembers();
        defer publishPeers(members, outcomes);
        const g = members.len;
        // Waves bound the host: each member's circuit, run and solver state
        // exist only for its wave, and one nothing reads afterwards is freed
        // with it. One wave unless the lanes would pass `gpu_wave_bytes`, or
        // the device budget, which binds only where the wave would not fit
        // the device, and then over the host's `gpu_wave_min`.
        const per_lane = gpuLaneBytes(self.topology);
        const host_wave = @max(gpu_wave_min, @min(g, gpu_wave_bytes / per_lane));
        const devices = @import("gpu/devices.zig");
        const device_wave = devices.waveLanes(devices.table(), gpuLaneDeviceBytes(self.topology));
        const wave = @min(host_wave, device_wave);
        if (self.config.knobs.gpu_stats != .off and wave < g) std.debug.print("gpu-lanes: {d} lanes in waves of {d} (host {d}, device {d})\n", .{ g, wave, host_wave, device_wave });
        const live = try self.work.allocator().alloc(usize, @min(g, wave));
        var first: usize = 0;
        while (first < g) : (first += wave) {
            const end = @min(g, first + wave);
            try self.runGpuWave(members[first..end], outcomes[first..end], live);
            for (members[first..end]) |e| if (e.no_readers and e.built) e.circuit.release();
        }
        return outcomes[0];
    }

    /// One wave of `executeGpuBatch`: materialize its members, run their
    /// lanes, label their outcomes and check their latches.
    fn runGpuWave(self: *Executor, members: []const *Executor, outcomes: []QueryError!types.Result, live: []usize) QueryError!void {
        const a = self.work.allocator();
        var n_live: usize = 0;
        // ponytail: serial materialize; thousands of lanes take seconds here.
        // Parallelize over a thread pool if it shows above 5% of the batch.
        const t0 = @import("gpu/lanes.zig").Prof.now();
        for (members, 0..) |e, k| {
            e.materialize() catch |err| {
                outcomes[k] = err;
                continue;
            };
            // UIC omits the operating point; its initial vector still
            // has to exist before the GPU admission or host fallback
            // receives a RunCtx, just as in runQuery and CPU lane batches.
            if (e.job.uic()) e.seedUic() catch |err| {
                outcomes[k] = err;
                continue;
            };
            e.circuit.progress = self.controller.callback();
            live[n_live] = k;
            n_live += 1;
        }
        if (self.config.knobs.gpu_stats != .off) std.debug.print("gpu-lanes: materialize {d} lanes {d:.1} ms\n", .{ members.len, @as(f64, @floatFromInt(@import("gpu/lanes.zig").Prof.now() - t0)) / 1e6 });
        if (self.job == .op) return self.runGpuOps(members, live[0..n_live], outcomes);
        const ctxs = try a.alloc(types.RunCtx, n_live);
        laneCtxs(members, live[0..n_live], ctxs);
        switch (self.job) {
            inline .tran, .pss, .hb, .lstb, .dc => |opts, tag| {
                const M = module(tag);
                const ran = try a.alloc(M.LaneOutcome, n_live);
                if (n_live != 0) {
                    if (comptime tag == .lstb)
                        M.runGpuLanes(ctxs, opts, self.config.backend, ran)
                    else if (comptime tag == .tran)
                        // Every usable device takes a slice (multi-GPU stage 1).
                        @import("gpu/multi.zig").runLanes(M, self.allocator, self.config.threads, ctxs, opts, ran)
                    else
                        M.runGpuLanes(ctxs, opts, ran);
                }
                for (live[0..n_live], ran) |k, o|
                    outcomes[k] = members[k].settleLabeled(o);
            },
            else => unreachable,
        }
    }

    /// `runGpuWave` for operating points: each live member's start
    /// (`opStart`), `op.solveGpuLanes` over all of them, then each ends as
    /// `executeBatch`'s operating points do.
    fn runGpuOps(self: *Executor, members: []const *Executor, live: []const usize, outcomes: []QueryError!types.Result) QueryError!void {
        const a = self.work.allocator();
        const ckts = try a.alloc(*types.Circuit, live.len);
        const xs = try a.alloc([]f64, live.len);
        var options: op.Options = undefined;
        for (live, ckts, xs) |k, *c, *x| {
            const e = members[k];
            e.circuit.setSimState(.{ .kind = simKind(e.job) });
            options = e.opStart() catch |err| {
                for (live) |j| outcomes[j] = err;
                return;
            };
            c.* = &e.circuit;
            x.* = e.x.?;
        }
        const solved = try a.alloc(op.LaneOutcome, live.len);
        if (live.len != 0) op.solveGpuLanes(lane_width, ckts, xs, options, self.deck.nodeset, self.deck.ic, solved) catch |err| {
            for (live) |k| outcomes[k] = err;
            return;
        };
        for (live, solved) |k, o| {
            const e = members[k];
            outcomes[k] = e.settle(blk: {
                const r = o catch |err| break :blk err;
                if (!r.converged) break :blk error.OpDidNotConverge;
                break :blk e.finish();
            });
        }
    }

    /// The leading AC-family sweeps across a wide GPU batch. Every query
    /// keeps its own solver/plan and reuses its ordinary readout. A point
    /// whose setup or precompute fails runs live, preserving its numerical
    /// failure, device-backend policy and partial streaming behavior.
    fn executeGpuFreqBatch(self: *Executor) QueryError!types.Result {
        const frequency = @import("ac/freq.zig");
        const provider = @import("gpu/freq.zig");
        const FS = @import("solver").freq_solve.FreqSolver;
        const a = self.work.allocator();
        const members, const outcomes = try self.gpuMembers();
        defer publishPeers(members, outcomes);
        const g = members.len;
        const ctxs = try a.alloc(?types.RunCtx, g);
        @memset(ctxs, null);
        const plans = try a.alloc(?frequency.Plan, g);
        @memset(plans, null);
        const solvers = try a.alloc(?FS, g);
        @memset(solvers, null);
        defer for (plans, solvers, ctxs) |p, *fs, ctx| {
            if (fs.*) |*f| f.deinit();
            if (p) |plan| plan.deinit(ctx.?.scratch_allocator);
        };
        const points = try a.alloc(provider.Point, g);
        const point_ids = try a.alloc(usize, g);
        var n: usize = 0;
        var pre_bytes: usize = 0;
        for (members, 0..) |e, k| {
            e.materialize() catch |err| {
                outcomes[k] = err;
                continue;
            };
            e.circuit.progress = self.controller.callback();
            e.circuit.setSimState(.{ .kind = simKind(e.job) });
            ctxs[k] = e.runCtx();
            // Stateful display tasks must run only through their ordinary
            // analysis; the precompute would evaluate them an extra time.
            const display_names = e.sayNames() catch continue;
            if (display_names.len != 0) continue;
            plans[k] = e.frequencyPlan(&ctxs[k].?) catch continue;
            const plan = plans[k] orelse continue;
            const rhs_bytes = std.math.mul(usize, plan.rhs.len, @sizeOf(f64)) catch continue;
            const bytes = std.math.mul(usize, plan.omegas.len, rhs_bytes) catch continue;
            // Bound the extra precompute storage of a wide group. Large
            // individual sweeps keep their bounded Stream working set.
            if (bytes > 256 * 1024 * 1024 - pre_bytes) continue;
            solvers[k] = FS.fromCircuit(ctxs[k].?.scratch_allocator, &e.circuit, ctxs[k].?.x_op) catch continue;
            plan.configure(&solvers[k].?);
            const out = e.work.allocator().alloc(f64, bytes / @sizeOf(f64)) catch continue;
            points[n] = .{
                .fs = &solvers[k].?,
                .circuit = &e.circuit,
                .x_op = ctxs[k].?.x_op,
                .omegas = plan.omegas,
                .rhs = plan.rhs,
                .out = out,
                .adjoint = plan.adjoint,
            };
            point_ids[n] = k;
            n += 1;
            pre_bytes += bytes;
        }
        const computed = blk: {
            const stats = provider.solveAll(a, self.config.backend, points[0..n]) catch |err| {
                if (self.config.knobs.gpu_stats != .off)
                    std.debug.print("gpu-freq: precompute declined ({s}); running queries live\n", .{@errorName(err)});
                break :blk false;
            };
            if (self.config.knobs.gpu_stats != .off and stats.lanes != 0)
                std.debug.print("gpu-freq: {d} points, {d} frequency lanes, {d} groups, {d} peels\n", .{ stats.points, stats.lanes, stats.groups, stats.peels });
            break :blk true;
        };
        if (computed) {
            for (points[0..n], point_ids[0..n]) |p, k| ctxs[k].?.freq_pre = p.out;
        }
        for (members, ctxs, 0..) |e, *ctx, k| {
            if (ctx.*) |*c| {
                if (c.freq_pre.len == 0) {
                    outcomes[k] = e.finishAlone();
                    continue;
                }
                outcomes[k] = e.settleLabeled(run(c, e.job));
            }
        }
        return outcomes[0];
    }

    /// `executeBatch`'s AC-family branch: every live lane's leading
    /// frequency sweep solved up front (`precompute`), then each lane's
    /// analysis replaying it (`RunCtx.freq_pre`) and labelled as alone. The
    /// two passes are the GPU lane backend's shape: it replaces the first
    /// with (point, ω) lane rounds (dev/research/plans/phase5-ac-lanes.md).
    /// A lane whose precompute fails runs live, so it fails or streams as
    /// it would alone.
    // ponytail: holds points·ω·rhs values per lane until its run ends;
    // chunk the precompute if a long sweep on a large n needs it.
    fn runFreqLanes(members: []const *Executor, live: []const usize, outcomes: []QueryError!types.Result) void {
        var ctxs: [lane_width]types.RunCtx = undefined;
        laneCtxs(members, live, &ctxs);
        for (live, ctxs[0..live.len]) |k, *c| c.freq_pre = members[k].precompute(c) catch &.{};
        for (live, 0..) |k, l| {
            const e = members[k];
            outcomes[k] = e.settleLabeled(run(&ctxs[l], e.job));
        }
    }

    /// The query's leading frequency sweep, solved as its own `run` would
    /// solve it (`freq.solveAll` over its `plan`), in the query's work
    /// arena; empty for a kind that solves live.
    fn precompute(self: *Executor, ctx: *const types.RunCtx) ![]const f64 {
        const freq = @import("ac/freq.zig");
        const scratch = ctx.scratch_allocator;
        const p = (try self.frequencyPlan(ctx)) orelse return &.{};
        defer p.deinit(scratch);
        var fs = try @import("solver").freq_solve.FreqSolver.fromCircuit(scratch, ctx.circuit, ctx.x_op);
        defer fs.deinit();
        p.configure(&fs);
        const out = try self.work.allocator().alloc(f64, p.omegas.len * p.rhs.len);
        try freq.solveAll(scratch, &fs, ctx.circuit, ctx.x_op, p.omegas, p.rhs, p.adjoint, out);
        return out;
    }

    fn frequencyPlan(self: *Executor, ctx: *const types.RunCtx) !?@import("ac/freq.zig").Plan {
        return switch (self.job) {
            .ac => |o| try @import("ac/ac.zig").plan(ctx, o),
            .acxf => |o| try @import("ac/xf.zig").plan(ctx, o),
            .stb => |o| try @import("ac/stb.zig").plan(ctx, o),
            .lstb => |o| try @import("ac/lstb.zig").plan(ctx, o),
            .sp => |o| try @import("ac/sp.zig").plan(ctx, o),
            .noise => |o| try @import("ac/noise.zig").plan(ctx, o),
            else => null,
        };
    }

    /// This query and its `peers` in the work arena, this one first, each
    /// outcome a cancel until its lane settles it; `publishPeers` hands the
    /// peers theirs. A GPU batch may be thousands wide.
    fn gpuMembers(self: *Executor) !struct { []*Executor, []QueryError!types.Result } {
        const a = self.work.allocator();
        const members = try a.alloc(*Executor, 1 + self.peers.len);
        members[0] = self;
        @memcpy(members[1..], self.peers);
        const outcomes = try a.alloc(QueryError!types.Result, members.len);
        @memset(outcomes, error.QueryCancelled);
        return .{ members, outcomes };
    }

    /// `settle` of a lane's analysis outcome, its result labelled as alone.
    fn settleLabeled(self: *Executor, outcome: anytype) QueryError!types.Result {
        return self.settle(if (outcome) |r| self.label(r) else |err| err);
    }

    /// `execute` past `materialize`, for a batch lane that runs alone.
    fn finishAlone(self: *Executor) QueryError!types.Result {
        return self.settle(self.runQuery());
    }

    /// `outcome`, unless a device latched a refusal (`refuseLatched`),
    /// which wins whether the run failed or not.
    fn settle(self: *const Executor, outcome: QueryError!types.Result) QueryError!types.Result {
        try self.refuseLatched();
        return outcome;
    }

    /// `error.DeviceRefused` after logging `<instance>: <file>:<line>:
    /// fatal|error: <message>` for the first instance with a latched status.
    /// A warning, not `log.err`: a test runner fails any test that logs an
    /// error, and refusing is this path's expected outcome.
    fn refuseLatched(self: *const Executor) error{DeviceRefused}!void {
        var buf: [512]u8 = undefined;
        for (self.circuit.batches, self.circuit.batch_types) |b, t| {
            const status = b.hooks.status orelse continue;
            const hit = status(b.ctx, &buf) orelse continue;
            const name = self.deck.card_names.get(t, hit.index) orelse b.type_name;
            std.log.warn("{s}: {s}", .{ name, buf[0..hit.len] });
            return error.DeviceRefused;
        }
    }

    fn runQuery(self: *Executor) !types.Result {
        // Borrowed lanes serve the device stamp and the multicore refactor;
        // both are bitwise their one-thread forms, so only speed moves.
        var lanes = self.config.device_threads;
        var borrowed: u32 = 0;
        if (self.config.thread_pool) |pool| if (self.circuit.ws == null) {
            const want = wantLanes(&self.circuit);
            borrowed = pool.take(want - 1);
            lanes = 1 + borrowed;
            var execution = self.circuit.solver_execution;
            execution.lu_threads = @intCast(@min(lanes, 16));
            self.circuit.configure(execution, self.circuit.knobs);
        };
        defer if (self.config.thread_pool) |pool| pool.give(borrowed);
        var par: ?ParEval = null;
        if (lanes > 1) {
            par = try ParEval.init(self.allocator, self.io, self.circuit.batches, self.circuit.nnz, self.circuit.n, self.circuit.has_charge, lanes);
            self.circuit.par_eval = &par.?;
        }
        defer {
            self.circuit.par_eval = null;
            if (par) |*p| p.deinit();
        }
        // The query's device eval and device LU open on the least loaded
        // device (multi-GPU stage 2); one device keeps them on it. An
        // explicit `--backend cpu` opens no device: reading the device table
        // starts the driver, ~140 ms a run on a machine with a GPU.
        const devices = @import("gpu/devices.zig");
        const placed: devices.Placement = if (self.config.backend == .cpu) .{ .slot = null, .prev = devices.ordinal } else devices.place();
        defer placed.release();
        const gpu_context = try self.prepareGpu(lanes);
        // The bypass serves host evaluations only.
        if (self.config.knobs.bypass and gpu_context == null) {
            // A tenth of reltol: a replay's current error is about reltol
            // times the device currents, which a net current that nearly
            // cancels (a supply at a switching edge) would show at full reltol.
            const tol = self.deck.deck_tol;
            self.circuit.setBypass(.{ .reltol = tol.reltol / 10, .vntol = tol.vntol, .abstol = tol.abstol });
        }
        defer if (gpu_context) |g| {
            // `refuseLatched` reads statuses off the host instances.
            g.syncStatus();
            self.circuit.detachGpu();
            g.deinit();
        };
        const lu_context = self.prepareGpuLu();
        defer {
            self.circuit.lu = null;
            if (lu_context) |l| l.deinit();
        }
        // §9.4 display tasks print to stdout by default, unbuffered, so each
        // accepted point's lines land in order with the rest of the run's output.
        var stdout: std.Io.File.Writer = .initStreaming(.stdout(), self.io, &.{});
        const out = self.config.display orelse &stdout.interface;
        var display: types.Display = .{ .names = try self.sayNames(), .staged = .init(self.allocator), .out = out };
        defer display.staged.deinit();
        if (display.names.len != 0) self.circuit.display = &display;
        defer self.circuit.display = null;
        if (self.config.timing_in_depth) (try self.circuit.workspace()).prof.io = self.io;
        defer if (self.config.timing_in_depth) printNewtonSplit(&self.circuit.ws.?, self.job);
        // It is what gives a host-integrated idt its 1/(jw) instead of its
        // DC form (`simKind`).
        self.circuit.setSimState(.{ .kind = simKind(self.job) });
        if (self.job == .op) {
            const options = try self.opStart();
            const solved = try op.solve(&self.circuit, self.x.?, options, self.deck.nodeset, self.deck.ic);
            if (!solved.converged) return error.OpDidNotConverge;
            if (gpu_context) |g| try g.syncHostState();
            // The OP ladder publishes its own sim state (`initial_step`);
            // restore the plain one that dependents inherit.
            self.circuit.setSimState(.{ .kind = simKind(self.job) });
        } else if (self.job.uic()) try self.seedUic();
        return self.finish();
    }

    /// A `uic` start's x: zero, then every `.ic` and `uic` value.
    fn seedUic(self: *Executor) !void {
        self.x = try self.work.allocator().alloc(f64, self.circuit.n);
        @memset(self.x.?, 0);
        for (self.deck.ic) |ic| self.x.?[ic.node] = ic.value;
        for (self.deck.uic_ic) |ic| self.x.?[ic.node] = ic.value;
    }

    /// The operating point's start: `x` from the deck's guess, the
    /// template's warm point or zero, and the options that start implies.
    fn opStart(self: *Executor) !op.Options {
        self.x = try self.work.allocator().alloc(f64, self.circuit.n);
        @memset(self.x.?, 0);
        var options = self.job.op;
        if (self.deck.op_guess.len != 0) {
            std.debug.assert(self.deck.op_guess.len == self.circuit.n);
            @memcpy(self.x.?, self.deck.op_guess);
            options.warm_start = true;
        } else if ((options.warm_start or options.tol.warm_op) and self.deck.warm_op.len != 0 and !self.movesTemp()) {
            std.debug.assert(self.deck.warm_op.len == self.circuit.n);
            @memcpy(self.x.?, self.deck.warm_op);
            options.warm_start = true;
            // From another point's solution Newton stops within reltol of
            // this one's, on the far side from where the cold ladder stops:
            // converge a thousand times tighter (one or two more quadratic
            // iterations) so the answer is the point's own, not the path's.
            options.tol.reltol *= 1e-3;
            options.tol.vntol *= 1e-3;
            options.tol.abstol *= 1e-3;
        } else options.warm_start = false;
        return options;
    }

    /// The analysis proper, from `x` (the operating point, or a `uic` start).
    fn finish(self: *Executor) !types.Result {
        const run_ctx = self.runCtx();
        return self.label(try run(&run_ctx, self.job));
    }

    fn runCtx(self: *Executor) types.RunCtx {
        return .{
            .circuit = &self.circuit,
            .x_op = self.x.?,
            .probes = self.deck.probes,
            .probe_labels = self.deck.probe_labels,
            .source_node = self.deck.source_node,
            .source_branch = self.deck.source_branch,
            .nodeset = self.deck.nodeset,
            .ac_drive = self.deck.ac_drive,
            .variants = self.deck.variants,
            .allocator = self.results.allocator(),
            .scratch_allocator = self.allocator,
            .stream = self.stream,
            .measures = self.deck.measures,
        };
    }

    /// Names a finished result after the query's temperature and variant.
    fn label(self: *Executor, result_in: types.Result) !types.Result {
        var res = result_in;
        // Copies of one card at several temperatures keep apart by name.
        if (queryTemp(self.job)) |temp| res.plotname = try self.results.allocator().print("{s} (temp={d})", .{ res.plotname, temp });
        if (queryVariant(self.job)) |v| res.plotname = try self.results.allocator().print("{s} ({s})", .{ res.plotname, self.deck.variants.labelAt(v) });
        return res;
    }

    /// `Display.names` from the deck's cards, an unlisted instance under its
    /// type name; empty when no batch has `Hooks.say`.
    fn sayNames(self: *Executor) ![]const []const []const u8 {
        const batches = self.circuit.batches;
        for (batches) |b| {
            if (b.hooks.say != null) break;
        } else return &.{};
        const a = self.work.allocator();
        const names = try a.alloc([]const []const u8, batches.len);
        for (names, batches, self.circuit.batch_types) |*row, b, t| {
            if (b.hooks.say == null) {
                row.* = &.{};
                continue;
            }
            const own = try a.alloc([]const u8, b.count);
            for (own, 0..) |*name, i| name.* = self.deck.card_names.get(t, @intCast(i)) orelse b.type_name;
            row.* = own;
        }
        return names;
    }

    /// The device LU: `Knobs.gpu_lu` (`ESPICE_GPU_LU=1`) forces it on under a
    /// GPU backend, any other value off; otherwise it runs where the matrix is big enough to be
    /// worth asking the driver, on every pivot epoch with enough flops per
    /// column or in all (`GpuLu.admits`), whatever the card's FP64 rate. A
    /// driver that refuses leaves the host LU.
    fn prepareGpuLu(self: *Executor) ?*gpu_lu.GpuLu {
        // A lone operating point refactors on the device (logic_bsim4_10k
        // under `cuda`: 57 iterations, 6.3 s against 7.5-40 s of host LU).
        if (self.config.backend == .cpu or (self.config.gpu_lanes and self.config.may_batch and self.job == .op)) return null;
        var forced = false;
        if (self.config.knobs.gpu_lu) |on| {
            if (!on) return null;
            forced = true;
        } else if (self.circuit.n < gpu_lu_min_n) return null;
        const context = gpu_lu.GpuLu.init(self.allocator, self.circuit.n, self.circuit.nnz, self.config.knobs) catch |err| {
            if (forced) std.debug.print("warning: device LU unavailable ({s})\n", .{@errorName(err)});
            return null;
        };
        context.forced = forced;
        self.circuit.lu = context;
        return context;
    }

    fn prepareGpu(self: *Executor, lanes: u32) !?*gpu.GpuContext {
        // Under GPU lanes the operating points run on the CPU workers, in
        // parallel, and only the transients' lanes take the device.
        if (self.config.backend == .cpu or (self.config.gpu_lanes and self.job == .op)) return null;
        const explicit = self.config.gpu_explicit or self.config.backend != .auto;
        const c = &self.circuit;
        const host: gpu.Host = .{ .batches = c.batches, .planes = c.ownPlanes(), .n = c.n, .trash_slot = c.trash_slot, .has_charge = c.has_charge, .knobs = c.knobs };
        const context = gpu.GpuContext.init(self.allocator, host, self.config.gpu_driver, explicit, lanes, expectedEvals(self.job)) catch |err| {
            if (!gpu.fallsBack(self.config.backend, explicit, err)) {
                std.log.warn("--backend {s}: GPU unavailable ({s}); detected GPU hardware: {s}", .{ @tagName(self.config.backend), @errorName(err), gpu.detectedHardware() });
                return err;
            }
            if (gpu.declineKind(err) != .policy and !self.config.gpu_quiet)
                std.log.warn("GPU unavailable ({s}); running on the CPU", .{@errorName(err)});
            return null;
        };
        self.circuit.attachGpu(context);
        return context;
    }
};

/// Below this many unknowns `auto` never asks the driver about the device
/// LU. ponytail: the E2 decks start at 17k; measure a data-center card
/// before trusting the bar.
const gpu_lu_min_n = 10_000;

/// The temperature `job` runs at when it overrides the deck's.
fn queryTemp(job: requests.Query) ?f64 {
    return switch (job) {
        inline else => |o| o.tol.temp_c,
    };
}

fn queryVariant(job: requests.Query) ?u32 {
    return switch (job) {
        inline else => |o| o.tol.variant,
    };
}

/// `--timing-in-depth`: where the query's Newton time went.
/// The matrix size and the flat LU's fill (L + U + diagonal) close the line.
fn printNewtonSplit(ws: *const converger.Workspace, job: requests.Query) void {
    const p = ws.prof;
    const ms = struct {
        fn f(ns: u64) f64 {
            return @as(f64, @floatFromInt(ns)) / 1e6;
        }
    }.f;
    std.debug.print("timing: {s} newton: iterations={d} factors={d} eval={d:.3}ms load={d:.3}ms factor={d:.3}ms solve={d:.3}ms update={d:.3}ms n={d} lu_nnz={d}\n", .{
        @tagName(job), p.counts[0], p.counts[1], ms(p.ns[0]), ms(p.ns[1]), ms(p.ns[2]), ms(p.ns[3]), ms(p.ns[4]), ws.dx.len,
        switch (ws.slv.engine) {
            .lu => |lu| lu.li.items.len + lu.ui.items.len + lu.n,
            .tri, .bbd => 0,
        },
    });
}

/// A rough count of the device evals `job` runs, for the GPU cost model:
/// four per transient step (measured 4.3 on mos1_2000 and 5.2 on
/// vacask_ring) at the step the deck asks for, three per dc point, and a
/// Newton solve's worth for an operating point. Small-signal analyses
/// linearize once.
/// ponytail: a static guess; count evals as the query runs and move to the
/// GPU mid-run if a guess ever misprices a deck.
fn expectedEvals(job: requests.Query) f64 {
    return switch (job) {
        .op => 30,
        .dc => |d| blk: {
            const inner = @abs(d.stop - d.start) / @max(@abs(d.step), 1e-300) + 1;
            const outer = if (d.target2 != null) @abs(d.stop2 - d.start2) / @max(@abs(d.step2), 1e-300) + 1 else 1;
            break :blk 3 * inner * outer;
        },
        .tran => |t| 4 * t.t_stop / @min(t.dt_init, t.requestedCeiling(.api)),
        .ac, .noise, .sp, .stb, .tf, .pz, .disto, .four, .dcmatch, .sens, .lstb, .acxf, .dcxf, .dcinc, .acmatch, .dcsens => 1,
        else => 200,
    };
}

/// Each live member's `RunCtx`, into `ctxs[0..live.len]`, its sim state set
/// as `runQuery` sets it (`simKind`).
fn laneCtxs(members: []const *Executor, live: []const usize, ctxs: []types.RunCtx) void {
    for (live, ctxs[0..live.len]) |k, *c| {
        members[k].circuit.setSimState(.{ .kind = simKind(members[k].job) });
        c.* = members[k].runCtx();
    }
}

/// Hands each peer, `members[1..]`, its lane's outcome.
fn publishPeers(members: []const *Executor, outcomes: []const QueryError!types.Result) void {
    for (members[1..], outcomes[1..]) |p, o| {
        if (o) |r| p.published = r else |err| p.peer_failure = err;
    }
}

fn module(comptime kind: requests.Kind) type {
    return switch (kind) {
        .op => op,
        .dc => @import("dc/dc.zig"),
        .tf => @import("dc/tf.zig"),
        .dcmatch => @import("dc/dcmatch.zig"),
        .dcsens => @import("dc/dcmatch.zig").Sens,
        .acmatch => @import("ac/acmatch.zig"),
        .ac => @import("ac/ac.zig"),
        .noise => @import("ac/noise.zig"),
        .sp => @import("ac/sp.zig"),
        .stb => @import("ac/stb.zig"),
        .lstb => @import("ac/lstb.zig"),
        .acxf => @import("ac/xf.zig"),
        .dcxf => @import("dc/xf.zig"),
        .dcinc => @import("dc/xf.zig").Inc,
        .tran => @import("tran/tran.zig"),
        .tran_noise => @import("tran/tran_noise.zig"),
        .envelope => @import("tran/envelope.zig"),
        .matex => @import("tran/matex.zig"),
        .pss => @import("pss/pss.zig"),
        .pac => @import("pss/pac.zig"),
        .pnoise => @import("pss/pnoise.zig"),
        .hb => @import("pss/hb.zig"),
        .pxf => @import("pss/pxf.zig"),
        .hbac => @import("pss/hb_lptv.zig").Ac,
        .hbxf => @import("pss/hb_lptv.zig").Xf,
        .hbnoise => @import("pss/hb_lptv.zig").Noise,
        .hblin => @import("pss/hb_lptv.zig").Lin,
        .phasenoise => @import("pss/phasenoise.zig"),
        .qpss => @import("pss/qpss.zig"),
        .four => @import("post/four.zig"),
        .disto => @import("post/disto.zig"),
        .fft => @import("post/fft.zig"),
        .sens => @import("sweep/sens.zig"),
        .mc => @import("sweep/mc.zig"),
        .temp => @import("sweep/temp_sweep.zig"),
        .pz => @import("eigen/pz.zig"),
    };
}

fn run(ctx: *const types.RunCtx, job: requests.Query) !types.Result {
    switch (job) {
        inline else => |opts, tag| return module(tag).run(ctx, opts),
    }
}
