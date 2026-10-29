//! Executor: one query's mutable numerical state and the worker thread that
//! runs it. Prepared topology and the deck are borrowed; an accepted OP from
//! a dependency is copied before use, or moved under `Config.final_plan`.
//! Dispatches to the analysis leaves (`dispatch.zig`), with the query's
//! ParEval, GPU context and device LU attached for the run; a lane batch
//! runs through `executor_lanes.zig`. The shared settings are
//! `exec_config.zig`, re-exported here.
//! In: a prepared circuit, the deck, one query. Out: its result, or why it
//! has none.
//! Cites VAMS LRM §9.4 (display tasks); #13.
//!
//! Contents:
//! - `pub const QueryError`
//! - `pub const Config`, `ThreadPool`, `lane_width`, `validateBackend`,
//!   `lanesOn`, `batchWidth`, `freqLaneKind`, `gpuFreqLaneKind` (`exec_config.zig`)
//! - `pub const Executor = struct` (`Outcome`, `create`, `start`, `wait`,
//!   `stopped`, `cancel`, `result`, `operatingPoint`, `destroy`, and the
//!   batch members `materialize`, `finishAlone`, `settleLabeled`, `settle`,
//!   `seedUic`, `opStart`, `finish`, `runCtx`, `sayNames`, `executeBatch`,
//!   `executeGpuBatch`, `runGpuWave`, `runGpuOps`, `executeGpuFreqBatch`,
//!   `precompute`, `frequencyPlan`, `gpuMembers`)

const std = @import("std");
const assert = std.debug.assert;
const Circuit = @import("device").Circuit;
const Deck = @import("core").Deck;
const requests = @import("core").query;
const converger = @import("solver").converger;
const types = @import("types.zig");
const op = @import("dc/op.zig");
const gpu = @import("gpu/context.zig");
const gpu_lu = @import("gpu/lu.zig");
const gpu_devices = @import("gpu/devices.zig");
const par_eval = @import("par_eval.zig");
const ParEval = par_eval.ParEval;
const worker = @import("worker.zig");
const exec_config = @import("exec_config.zig");
const executor_lanes = @import("executor_lanes.zig");
const dispatch = @import("dispatch.zig");
const Quantum = worker.Quantum;
const simKind = dispatch.simKind;
const queryTemp = dispatch.queryTemp;
const queryVariant = dispatch.queryVariant;

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
pub const QueryError = types.ErrorOf(dispatch.run) || types.ErrorOf(op.solve) || types.ErrorOf(ParEval.init) ||
    gpu.InitError || types.ErrorOf(gpu.GpuContext.syncHostState) || types.ErrorOf(types.Circuit.workspace) ||
    std.mem.Allocator.Error || error{ OpDidNotConverge, DeviceRefused, QueryCancelled } ||
    // `materialize`'s parts.
    types.ErrorOf(types.Circuit.fromSnapshot) || types.ErrorOf(types.Circuit.fromSnapshotMove) ||
    types.ErrorOf(types.Circuit.instantiate) || types.ErrorOf(types.Circuit.instantiateMove) ||
    types.ErrorOf(types.Circuit.recompute) || types.ErrorOf(types.Circuit.collectParams) || types.ErrorOf(types.Circuit.setParams) || error{InvalidAcOverride};

/// Per-problem execution settings (`exec_config.Config`).
pub const Config = exec_config.Config;
/// Threads a Problem lends its large queries (`exec_config.ThreadPool`).
pub const ThreadPool = exec_config.ThreadPool;
/// Lanes per batch (`exec_config.lane_width`).
pub const lane_width = exec_config.lane_width;
/// Refuses a backend without kernels or a zero thread count.
pub const validateBackend = exec_config.validateBackend;
/// Whether `Session` groups ready queries into lane batches.
pub const lanesOn = exec_config.lanesOn;
/// Most points one lane batch runs together; advisory.
pub const batchWidth = exec_config.batchWidth;
/// The AC-family kinds a lane batch runs.
pub const freqLaneKind = exec_config.freqLaneKind;
/// The sweeps with a GPU (point, frequency) provider.
pub const gpuFreqLaneKind = exec_config.gpuFreqLaneKind;

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
    /// Lane batches (`executor_lanes.zig`).
    pub const executeBatch = executor_lanes.executeBatch;
    pub const executeGpuBatch = executor_lanes.executeGpuBatch;
    pub const runGpuWave = executor_lanes.runGpuWave;
    pub const runGpuOps = executor_lanes.runGpuOps;
    pub const executeGpuFreqBatch = executor_lanes.executeGpuFreqBatch;
    pub const precompute = executor_lanes.precompute;
    pub const frequencyPlan = executor_lanes.frequencyPlan;
    pub const gpuMembers = executor_lanes.gpuMembers;

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
        return self.deck.variants.temp_c[v] != null;
    }

    /// The circuit copy, variant and AC overrides `create` deferred, built
    /// on the worker thread before the analysis runs.
    pub fn materialize(self: *Executor) !void {
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
        const variant_temp = if (variant) |v| self.deck.variants.temp_c[v] else null;
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
        // SAFETY: `create` hands the controller `self` as its ctx.
        const self: *Executor = @ptrCast(@alignCast(ctx));
        if (self.peers.len != 0) return self.executeBatch();
        try self.materialize();
        return self.finishAlone();
    }

    /// `settle` of a lane's analysis outcome, its result labelled as alone.
    pub fn settleLabeled(self: *Executor, outcome: anytype) QueryError!types.Result {
        return self.settle(if (outcome) |r| self.label(r) else |err| err);
    }

    /// `execute` past `materialize`, for a batch lane that runs alone.
    pub fn finishAlone(self: *Executor) QueryError!types.Result {
        return self.settle(self.runQuery());
    }

    /// `outcome`, unless a device latched a refusal (`refuseLatched`),
    /// which wins whether the run failed or not.
    pub fn settle(self: *const Executor, outcome: QueryError!types.Result) QueryError!types.Result {
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
        const devices = gpu_devices;
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
    pub fn seedUic(self: *Executor) !void {
        self.x = try self.work.allocator().alloc(f64, self.circuit.n);
        @memset(self.x.?, 0);
        for (self.deck.ic) |ic| self.x.?[ic.node] = ic.value;
        for (self.deck.uic_ic) |ic| self.x.?[ic.node] = ic.value;
    }

    /// The operating point's start: `x` from the deck's guess, the
    /// template's warm point or zero, and the options that start implies.
    pub fn opStart(self: *Executor) !op.Options {
        self.x = try self.work.allocator().alloc(f64, self.circuit.n);
        @memset(self.x.?, 0);
        var options = self.job.op;
        if (self.deck.op_guess.len != 0) {
            assert(self.deck.op_guess.len == self.circuit.n);
            @memcpy(self.x.?, self.deck.op_guess);
            options.warm_start = true;
        } else if ((options.warm_start or options.tol.warm_op) and self.deck.warm_op.len != 0 and !self.movesTemp()) {
            assert(self.deck.warm_op.len == self.circuit.n);
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
    pub fn finish(self: *Executor) !types.Result {
        const run_ctx = self.runCtx();
        return self.label(try dispatch.run(&run_ctx, self.job));
    }

    pub fn runCtx(self: *Executor) types.RunCtx {
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
        if (queryVariant(self.job)) |v| res.plotname = try self.results.allocator().print("{s} ({s})", .{ res.plotname, self.deck.variants.labels[v] });
        return res;
    }

    /// `Display.names` from the deck's cards, an unlisted instance under its
    /// type name; empty when no batch has `Hooks.say`.
    pub fn sayNames(self: *Executor) ![]const []const []const u8 {
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
        const context = gpu.GpuContext.init(self.allocator, host, self.config.gpu_driver, explicit, lanes, dispatch.expectedEvals(self.job)) catch |err| {
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

const Controller = worker.Worker(QueryError, types.Result);

/// Below this many unknowns `auto` never asks the driver about the device
/// LU. ponytail: the E2 decks start at 17k; measure a data-center card
/// before trusting the bar.
const gpu_lu_min_n = 10_000;

/// Device weight (`ParEval`'s per-instance stamp estimate summed over the
/// circuit) from which extra lanes pay: about 800 BSIM4 instances. ParEval
/// measured 3.3x on 4,000 bsim4 at 8 threads and no gain on 4,000 mos1,
/// whose summed weight sits below this (`gpu/context.zig` cost notes).
const borrow_min_weight: u64 = 4_000_000;
/// Matrix entries from which a refactor may be thick enough for the
/// multicore LU; its own per-epoch admission (`direct.zig`) still decides.
const borrow_min_nnz: u32 = 50_000;

// ---- private ----

/// Lanes for a query of `ckt`: 1 below both gates.
fn wantLanes(ckt: *const types.Circuit) u32 {
    var weight: u64 = 0;
    for (ckt.batches) |b| weight += @as(u64, b.count) * par_eval.instanceWeight(b);
    return if (weight >= borrow_min_weight or ckt.nnz >= borrow_min_nnz) ThreadPool.per_query else 1;
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
