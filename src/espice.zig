//! The `Problem` facade: one netlist prepared once, a DAG of analysis queries
//! run over it, and their results delivered to one output. main.zig and
//! c_api.zig import only this module.
const std = @import("std");
const frontend = @import("frontend");
const analysis = @import("analysis");
const output = @import("output");
const core = @import("core");
/// Query request types (`core.query`): what `append_queries` takes.
pub const requests = core.query;
/// Dense DAG index, `0..query_count()`. Appends never renumber an ID.
pub const QueryId = requests.QueryId;
/// One analysis request; `append_queries` copies its slices.
pub const Query = requests.Query;
/// A netlist path, or in-memory text with the origin it is named by.
pub const Source = frontend.Source;
/// The netlist syntax family, which decides SPICE letter and LEVEL meaning.
pub const Dialect = frontend.Dialect;
/// Maps a CLI dialect name (`ngspice`, `hspice`, `spectre`); null if unknown.
pub const parseDialect = frontend.parseDialect;
/// The dialect a deck's text is in, from markers only HSPICE or Spectre
/// decks carry; ngspice otherwise (the CLI's `--tokenizer` default).
pub const detectDialect = frontend.detectDialect;
/// Requested compute backend; `Problem.init` rejects one the build lacks; a
/// query run with an explicit cuda/hip fails when no GPU is usable.
pub const Request = analysis.ExecutionConfig.Backend;
/// This build carries GPU kernel images.
pub const gpu_available = analysis.gpu_available;
/// A query's lifecycle state; `terminal()` says whether it can still run.
pub const Status = analysis.session.Status;
/// Which part of the DAG a call looks at: all, one query's chain, or one component.
pub const Scope = analysis.session.Scope;
/// Concurrency and quantum for one advancing call.
pub const Limits = analysis.session.Limits;
/// Which call's next frontier `print` marks.
pub const Preview = analysis.session.Preview;
/// `print` settings; a null `limits` takes the Problem's.
pub const PrintOptions = analysis.session.PrintOptions;
/// A query's kind, status, prerequisite, component and progress.
pub const QueryInfo = analysis.session.QueryInfo;
/// What one quantum did: which query ran, where it and its target stand.
pub const Advance = analysis.session.Advance;
/// A published plot. Its slices borrow Problem storage until `deinit`.
pub const Result = output.Result;
/// Output file format; some (touchstone, citi) hold only S-parameter queries.
pub const Format = output.Format;
/// Output format and path. A null path keeps results in memory only.
pub const Selection = output.Selection;
/// Maps a CLI format name (`binary`, `csv`, ...); null if unknown.
pub const parseFormat = output.parseFormat;

/// Everything `Problem.init` needs. Its slices are copied during init.
pub const Options = struct {
    source: Source,
    dialect: Dialect = .ngspice,
    /// Where results go. `.save` and `.stim` files land in its path's directory.
    output: Selection = .{},
    /// Device backend and thread counts. `final_plan` lets a lone leading
    /// transient or AC sweep stream to the output instead of being kept,
    /// and `run_all` frees each result once its plot and `.meas` block are
    /// out (`summary` keeps its name and shape).
    backend: analysis.ExecutionConfig = .{},
    /// Queries `run_all` advances concurrently. Zero is `error.InvalidConcurrency`.
    max_parallel: u16 = 1,
    /// Let a large query borrow extra device and refactor lanes from a
    /// per-Problem pool of half the logical CPUs, in place of the fixed
    /// `backend.device_threads` and `backend.solver_threads`. Results are
    /// bitwise those of one thread.
    auto_threads: bool = false,
    /// Print per-phase wall times to stderr.
    timing_in_depth: bool = false,
    /// The Zig compiler and `share/espice` directory a `.hdl` model builds
    /// with; null fields read `$ZIG` and the executable's path (#13).
    hdl_toolchain: HdlToolchain = .{},
    /// Global `.param` names `set_params` may change later. Each stays live
    /// through the build, as a `.step` name does, so a name an analysis
    /// card, `.meas`, `.ic` or `.options` reads is `error.UnsupportedCard`,
    /// and one no `.param` defines is `error.UnknownParameter`.
    live_params: []const []const u8 = &.{},
    /// Where `.hdl` display-task lines (`$strobe`, `$display`, ...) go; null
    /// prints them on stdout. Each accepted point's lines arrive in one
    /// `writeAll` from the thread running its query; with `max_parallel`
    /// above 1 that is several threads at once, so the writer must
    /// serialize them. It must outlive the Problem.
    display: ?*std.Io.Writer = null,
    /// VPI applications (LRM 12.33) to load before the deck's `.hdl` models,
    /// as given; the deck may add more with `.options vpi=`. They need `vpi`.
    vpi_apps: []const []const u8 = &.{},
    /// The VPI host that loads them, which the espice executable supplies;
    /// null in an embedding, where any application is `VpiUnavailable`.
    /// It must outlive the Problem.
    vpi: ?*const Vpi = null,
    /// What every `.tran` of the deck does, as `append_points` asks of a
    /// point (the CLI's `--measures-only`, `--early-stop`, `--abort-after`).
    /// `measures_only` is dropped for a deck with `.save`, `.stim` or `.option soa`, which
    /// read every column.
    point: requests.PointOptions = .{},
    /// The CLI's default: also turn on `point`'s `measures_only`,
    /// `early_stop` and `warm_start` when nothing but the `.meas` cards
    /// reads the deck's transient waveforms (`waveformsConsumed`).
    auto_point: bool = false,
    /// Where `run_all` prints each result's `.meas` block as soon as it and
    /// every result before it have ended, flushing both writers, instead of
    /// leaving them all to `print_measures`; the optimization summaries
    /// come first. The same bytes in the same order as `print_measures`
    /// alone, at any `max_parallel`. Both writers must outlive the Problem.
    measures: ?MeasureWriters = null,
    /// Also keep every `.meas` result printed, as JSON, for
    /// `write_measures_json` (the CLI's `--measures=FILE`, #235).
    measures_json: bool = false,
};

/// `Options.measures`: the `.meas` values and the cards' error lines.
pub const MeasureWriters = struct { out: *std.Io.Writer, err: *std.Io.Writer };

/// The VPI host interface (`Options.vpi`), which vpi_host.zig implements.
pub const Vpi = frontend.Library.Vpi;
/// Why `Vpi.load_app` or `Vpi.attach` failed; Library.zig names each cause.
pub const VpiError = frontend.Library.VpiError;
/// The solution an analysis is attempting, as `Vpi.point` hands it over.
pub const VpiAnalogPoint = frontend.Library.AnalogPoint;
/// What the VPI callbacks decided about the point (`Vpi.verdict`).
pub const VpiVerdict = frontend.Library.Verdict;

/// Why publication stopped (`Problem.output_error`): an encoder or file
/// error from the output session (`publish`, `endStream`, `finish`) or the Touchstone
/// file writer (`output.write`), `error.OutOfMemory` included. The session
/// lookups' `error.InvalidQuery` and `error.ResultUnavailable` mean the
/// query bookkeeping broke (a bug), reported instead of a panic.
pub const DeliveryError = errorOf(output.Session.publish) || errorOf(output.Session.endStream) || errorOf(output.Session.endSide) || errorOf(output.Session.finish) || errorOf(output.write) ||
    errorOf(analysis.session.Session.info) || errorOf(analysis.session.Session.result) || errorOf(analysis.session.Session.query);

fn errorOf(comptime f: anytype) type {
    return @typeInfo(@typeInfo(@TypeOf(f)).@"fn".return_type.?).error_union.error_set;
}

/// Where runtime `.hdl` compilation finds its toolchain (`Options.hdl_toolchain`).
pub const HdlToolchain = frontend.Library.Toolchain;

/// A prepared circuit, its query DAG and the output its results go to.
///
/// Not thread-safe: the caller serializes every call, reads and `deinit`
/// included. Parallelism lives inside `advance_ready` and `run_all`. Result
/// slices stay valid until `deinit`; `copy_result` copies for callers that
/// cannot hold them.
pub const Problem = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    /// Serializes `allocator` for query workers; see `workerAllocator`.
    allocation_mutex: std.Io.Mutex = .init,
    /// Bytes the query workers hold through `workerAllocator` now, and the
    /// most they held at once (`queryMemory`). Counted under
    /// `allocation_mutex`.
    query_live: usize = 0,
    query_peak: usize = 0,
    /// Owns the prepared circuit and deck for the Problem's lifetime.
    arena: std.heap.ArenaAllocator,
    /// Every device type this problem can instantiate; outlives `prepared`.
    library: frontend.Library,
    prepared: frontend.Prepared,
    session: analysis.session.Session,
    delivery: output.Session,
    limits: Limits,
    /// Index into `session.outputs` of the next result to publish. Results
    /// publish in request order, so a finished query waits for earlier ones.
    next_output: usize = 0,
    /// `final_plan`: the main session's outputs below this index were
    /// released once written (`releaseDelivered`), and the shape of each
    /// released result, for `summary`.
    released_upto: usize = 0,
    released: std.AutoHashMapUnmanaged(QueryId, Summary) = .empty,
    /// First output failure. Once set, nothing else is published.
    delivery_error: ?DeliveryError = null,
    /// The query whose rows stream to the output as it runs (`openStream`).
    streaming: ?QueryId = null,
    /// Main-session queries whose rows stream to a side file as they run
    /// (`openSide`), until delivered.
    sides: std.AutoHashMapUnmanaged(QueryId, *output.Session.Stream) = .empty,
    /// The variant row `set_params` last made, which queries appended
    /// since then run at; null before any call.
    variant: ?u32 = null,
    /// `Options.live_params` as live slots (`Tuning.index`), resolved once
    /// so a point of `append_points` costs no name lookups.
    live: []const u32 = &.{},
    /// `Options.live_params`, as given.
    live_names: []const []const u8 = &.{},
    /// Where an HSPICE `.save` writes the operating point; null without one.
    save_path: ?[]const u8 = null,
    /// Where `.save` and `.stim` files go: beside the output file, as HSPICE
    /// writes beside its listing.
    out_dir: []const u8 = ".",
    /// One session per `prepared.runs` entry (variants with their own
    /// topology), run after the main session by `run_all`.
    runs: []Run = &.{},
    /// The deck's file name (`Source` path or origin), which names the
    /// Touchstone files `.lin format=touchstone` writes beside it.
    origin: []const u8 = "",

    /// Holds the netlist an HSPICE optimization re-evaluates
    /// (`prepared.tuner`); null for a deck without one.
    parse_arena: ?std.heap.ArenaAllocator = null,
    /// The finished optimizations, once `optimize` has run, one per
    /// `Tuner.bases` entry: Levenberg-Marquardt fits or bisections, as the
    /// `.model OPT` says.
    optimization: []core.lm.Lm = &.{},
    bisection: []core.bisect.Bisect = &.{},
    /// The MOSRA degradation table, once `age` has run.
    degradation: ?Result = null,
    /// Whether a query of this Problem brought the GPU driver up, which every
    /// session's queries read through `ExecutionConfig.gpu_driver`.
    gpu_driver: analysis.GpuDriver = .{},
    /// `Options.auto_threads`' pool, lent through `ExecutionConfig.thread_pool`.
    thread_pool: analysis.ThreadPool = .{ .free = .init(0) },
    /// The query threads every session's workers run on.
    threads: analysis.Threads,
    /// The solver workspaces sweep points hand on (`ExecutionConfig.ws_pool`).
    ws_pool: analysis.WsPool,
    /// Each transient appended with `PointOptions.tangent_step`: the step
    /// of each live parameter its tangent rows took (`point_tangents`).
    tangent_steps: std.AutoHashMapUnmanaged(QueryId, []const f64) = .empty,
    /// Bit v of `point_rows`: `append_points` made variant row v for a
    /// point (its own row or a tangent row), so nothing else reads it. Of
    /// `freed_rows`: nothing reads row v again (its point was released, or
    /// its call failed). `release_points` drops the table's freed prefix.
    point_rows: std.DynamicBitSetUnmanaged = .{},
    freed_rows: std.DynamicBitSetUnmanaged = .{},
    /// `Options.measures`, how far `run_all` has printed to it, and the
    /// first error printing there, which `print_measures` returns.
    measures_to: ?MeasureWriters = null,
    measure_cursor: MeasureCursor = .{},
    measure_error: ?anyerror = null,
    /// `Options.measures_json`: each result printed so far as a JSON
    /// object, each led by `,\n`.
    measure_json: ?std.Io.Writer.Allocating = null,
    /// The outputs `run_all` has checked for failures, and the first it
    /// found, which every later call returns too.
    checked_outputs: usize = 0,
    first_failure: ?anyerror = null,
    /// `Options.point.emir`: `write_emir_json` reads every transient after
    /// the run, so none is released.
    emir: bool = false,

    /// A variant run's queries and its next result to publish.
    const Run = struct { session: analysis.session.Session, next_output: usize = 0 };

    /// The next `.meas` block to print: whether the optimization summaries
    /// are out, the session (0 the main one, k the k-th run's) and its output.
    const MeasureCursor = struct { headed: bool = false, session: usize = 0, next: usize = 0 };

    /// A result's name and shape, which outlive its release.
    pub const Summary = struct { plotname: []const u8, npoints: usize, nvars: usize };

    /// Parses and prepares `options.source`, then queues the deck's analyses
    /// (a lone `.op` when the deck names none). Nothing runs yet.
    ///
    /// Fails on a netlist error, a backend the build lacks, zero
    /// `max_parallel`, or a query the output format cannot hold. The caller
    /// frees the result with `deinit`.
    pub fn init(allocator: std.mem.Allocator, io: std.Io, options: Options) !*Problem {
        var lap = if (options.timing_in_depth) std.Io.Timestamp.now(io, .awake) else null;
        if (options.max_parallel == 0) return error.InvalidConcurrency;
        try analysis.validateBackend(options.backend);
        const self = try allocator.create(Problem);
        errdefer allocator.destroy(self);
        // The undefined fields are set below, each before its first read.
        // An optimization keeps the parse arena: its tuner re-evaluates the netlist.
        // `self` is heap-allocated, so the arena keeps its address.
        self.* = .{
            .allocator = allocator,
            .io = io,
            .arena = .init(allocator),
            .library = undefined,
            .prepared = undefined,
            .session = undefined,
            .delivery = undefined,
            .limits = .{ .max_parallel = options.max_parallel },
            .parse_arena = .init(allocator),
            .threads = undefined,
            .ws_pool = undefined,
        };
        self.threads = .init(self.workerAllocator(), io);
        self.ws_pool = try .init(self.workerAllocator(), io, options.max_parallel);
        errdefer self.ws_pool.deinit();
        errdefer self.arena.deinit();
        if (analysis.gpu_available and options.backend.backend != .cpu) {
            // The driver reads the JIT cache cap at its start.
            raiseJitCache();
            if (likelyGpu(options)) self.gpu_driver.prestart();
        }
        errdefer self.gpu_driver.deinit();
        const a = self.arena.allocator();
        self.origin = try a.dupe(u8, switch (options.source) {
            .file => |path| path,
            .bytes => |bytes| bytes.origin,
        });
        var keep_parse = false;
        defer if (!keep_parse) {
            self.parse_arena.?.deinit();
            self.parse_arena = null;
        };
        const scratch = self.parse_arena.?.allocator();
        self.library = try frontend.Library.init(allocator);
        errdefer self.library.deinit();
        self.library.toolchain = try toolchainFromEnv(a, options.hdl_toolchain);
        self.library.vpi = options.vpi;
        const ast = try frontend.prepare(io, scratch, options.source, options.dialect, options.live_params);
        try frontend.loadModels(io, &self.library, scratch, ast, self.origin, options.vpi_apps);
        timingLap(io, &lap, "frontend (source, parsing, HDL)");
        self.prepared = try frontend.build(&self.library, a, scratch, ast);
        timingLap(io, &lap, "Problem creation (expansion, binding, topology)");
        errdefer self.prepared.deinit();
        if (self.prepared.tuning) |t| {
            const live = try a.alloc(u32, options.live_params.len);
            for (options.live_params, live) |name, *k| k.* = try t.index(name);
            self.live = live;
            self.live_names = try dupeAll(a, options.live_params);
        }
        self.out_dir = try a.dupe(u8, if (options.output.path) |out| std.fs.path.dirname(out) orelse "." else ".");
        if (self.prepared.deck.save_op) |save| {
            // `FILE=x.ic` names the `.ic#` family (one file per `.alter` run in
            // HSPICE); ESPice writes `#` = 0, which `.load FILE=x.ic` reads.
            const name = if (save.file) |f|
                try std.mem.concat(a, u8, &.{ f, if (std.ascii.eqlIgnoreCase(std.fs.path.extension(f), ".ic")) "0" else "" })
            else
                try std.mem.concat(a, u8, &.{ std.fs.path.stem(self.origin), ".ic0" });
            self.save_path = try std.fs.path.resolve(a, &.{ self.out_dir, name });
        }
        self.delivery = try output.Session.init(allocator, options.output);
        errdefer self.delivery.deinit();
        var execution = options.backend;
        execution.timing_in_depth = options.timing_in_depth;
        execution.knobs = knobsFromEnv();
        // A test whose oracle is the plain scalar run turns the bypass off.
        execution.knobs.bypass = options.backend.knobs.bypass;
        execution.gpu_driver = &self.gpu_driver;
        execution.threads = &self.threads;
        execution.ws_pool = &self.ws_pool;
        if (options.auto_threads and !@import("builtin").single_threaded) {
            self.thread_pool = .init(std.Thread.getCpuCount() catch 1);
            execution.thread_pool = &self.thread_pool;
        }
        // Under a bare `auto` the lane policy asks the driver for the FP64
        // rate, a driver start (~250 ms); only a deck that can batch pays it.
        execution.may_batch = mayBatch(&self.prepared, options);
        execution.gpu_lanes = if (execution.backend == .auto and !execution.gpu_explicit and !execution.may_batch)
            false
        else
            analysis.gpu_lanes.policy(execution);
        if (execution.gpu_lanes) {
            const cpus = std.Thread.getCpuCount() catch 1;
            execution.gpu_share = analysis.gpu_lanes.share(cpus);
            execution.gpu_static = analysis.gpu_lanes.staticPolicy(execution, cpus);
        }
        execution.display = options.display;
        self.measures_to = options.measures;
        // Allocates on its first write only, so no errdefer.
        if (options.measures_json) self.measure_json = .init(allocator);
        self.session = analysis.session.Session.init(self.workerAllocator(), io, &self.prepared.circuit, &self.prepared.deck, execution);
        errdefer self.session.deinit();
        // An optimization's sessions run before this one; nothing after it
        // instantiates the template.
        // `set_params` diffs later points against the template, so it is
        // never handed to a query to keep.
        if (self.prepared.tuning == null) self.session.markLastTemplateReader();
        var point = options.point;
        try self.checkPoint(point);
        if (options.auto_point and !self.waveformsConsumed(options)) {
            point.measures_only = true;
            point.early_stop = true;
            point.warm_start = true;
        }
        // A sweep's points start from the nominal operating point: the same
        // point within Newton's tolerances, in a few iterations instead of
        // the cold ladder (145 → 2 per point on a 20-stage CMOS chain).
        if (options.auto_point and isSweep(&self.prepared.deck)) point.warm_start = true;
        if (self.save_path != null or self.prepared.deck.stims.len != 0 or self.prepared.deck.soa.limits.len != 0) point.measures_only = false;
        // `--emir` reads every accepted point to t_stop (#237).
        if (point.emir) point.early_stop = false;
        self.emir = point.emir;
        // No analysis card runs an OP, unless it would have no columns (a
        // title-only deck): ngspice runs nothing there either.
        const jobs: []Query = if (self.prepared.deck.queries.len == 0 and self.prepared.deck.probes.len == 0)
            try scratch.alloc(Query, 0)
        else
            try withPoint(scratch, if (self.prepared.deck.queries.len == 0)
                &[_]Query{.{ .op = .{ .tol = self.prepared.deck.deck_tol } }}
            else
                self.prepared.deck.queries, point);
        if (point.warm_start) try self.solveWarmOp(jobs);
        // `tangent_step` on the deck's own transients: tangent rows at the
        // nominal live values, which `print_tangents` reads.
        const steps = try scratch.alloc([]const f64, jobs.len);
        if (point.tangent_step > 0) try self.tangentDeck(jobs, point.tangent_step, steps);
        const ids = try scratch.alloc(QueryId, jobs.len);
        _ = try self.append_queries(jobs, ids);
        if (point.tangent_step > 0) for (jobs, ids, steps) |job, id, s| if (job == .tran) {
            try self.tangent_steps.put(a, id, s);
        };
        self.runs = try a.alloc(Run, self.prepared.runs.len);
        for (self.runs, self.prepared.runs, 0..) |*run, *prep, k| {
            errdefer for (self.runs[0..k]) |*done| done.session.deinit();
            run.* = .{ .session = analysis.session.Session.init(self.workerAllocator(), io, &prep.circuit, &prep.deck, execution) };
            run.session.markLastTemplateReader();
            errdefer run.session.deinit();
            try self.checkOutput(prep, prep.deck.queries);
            _ = try run.session.append(try withPoint(scratch, prep.deck.queries, point), try scratch.alloc(QueryId, prep.deck.queries.len));
        }
        timingLap(io, &lap, "query graph and output setup");
        keep_parse = self.prepared.tuner != null or self.prepared.tuning != null;
        return self;
    }

    /// Gives each transient of `jobs` one tangent row per live parameter
    /// at its nominal value, moved by `h` as `append_points` moves it, and
    /// its steps in `steps`. `error.NoLiveParameters` without live
    /// parameters, `error.TangentsUnavailable` for a transient that cannot
    /// integrate them.
    fn tangentDeck(self: *Problem, jobs: []Query, h: f64, steps: [][]const f64) !void {
        const tuning = self.prepared.tuning orelse return error.NoLiveParameters;
        const values = try self.arena.allocator().alloc(f64, self.live.len);
        for (values, self.live) |*v, k| v.* = tuning.values[k];
        const current = self.session.template;
        defer self.session.template = current;
        for (jobs, steps) |*job, *s| if (job.* == .tran) {
            if (!self.tangentable(job.*)) return error.TangentsUnavailable;
            job.tran.tangents = try self.tangentRows(tuning, values, h, s);
        };
    }

    /// Whether anything besides the `.meas` cards reads the deck's
    /// transient waveforms: an output file, `.save`, `.stim`, `.option soa`, an
    /// optimization, tangents, or a device's display tasks, which print at
    /// every accepted step. `.print`, `.probe` and `.plot` are ignored
    /// cards, and `.four`, `.fft` and MOSRA run transients of their own.
    fn waveformsConsumed(self: *const Problem, options: Options) bool {
        const deck = &self.prepared.deck;
        if (options.output.path != null or self.save_path != null or deck.stims.len != 0 or deck.soa.limits.len != 0) return true;
        if (self.prepared.tuner != null or options.point.tangent_step > 0) return true;
        for (self.prepared.circuit.batches) |b| if (b.hooks.say != null) return true;
        for (self.prepared.runs) |run| for (run.circuit.batches) |b| if (b.hooks.say != null) return true;
        return false;
    }

    /// Whether `prepared` can hand the scheduler several transients of one
    /// template to batch: two in the deck, a variant run, or a template
    /// later points append to.
    fn mayBatch(prepared: *const frontend.Prepared, options: Options) bool {
        if (prepared.runs.len != 0 or prepared.tuning != null or options.live_params.len != 0) return true;
        var trans: usize = 0;
        for (prepared.deck.queries) |q| trans += @intFromBool(q == .tran);
        return trans >= 2;
    }

    /// Whether any of `deck`'s queries runs a variant row (`.step`, `.data`,
    /// `.mc`, `.alter` parameters, a temperature list).
    fn isSweep(deck: *const core.Deck) bool {
        for (deck.queries) |q| switch (q) {
            inline else => |o| if (o.tol.variant != null or o.tol.temp_c != null) return true,
        };
        return false;
    }

    /// A copy of `queries` whose transients carry `point`.
    fn withPoint(a: std.mem.Allocator, queries: []const Query, point: requests.PointOptions) ![]Query {
        const copy = try a.dupe(Query, queries);
        for (copy) |*q| {
            if (q.* == .tran) q.tran.point = point;
            if (point.warm_start) switch (q.*) {
                inline else => |*o| o.tol.warm_op = true,
            };
        }
        return copy;
    }

    /// Cancels and joins any paused query workers, then frees everything.
    /// Invalidates every result slice handed out. A Problem destroyed
    /// before its last query ended drops the plots published into binary
    /// or ASCII raw or `.print`, and the file keeps its old bytes.
    pub fn deinit(self: *Problem) void {
        const a = self.allocator;
        for (self.runs) |*run| run.session.deinit();
        self.released.deinit(a);
        self.point_rows.deinit(a);
        self.freed_rows.deinit(a);
        self.session.deinit();
        self.gpu_driver.deinit();
        self.sides.deinit(a);
        self.ws_pool.deinit();
        self.threads.deinit();
        self.delivery.deinit();
        self.prepared.deinit();
        self.library.deinit();
        if (self.measure_json) |*j| j.deinit();
        if (self.parse_arena) |*arena| arena.deinit();
        self.arena.deinit();
        a.destroy(self);
    }

    /// The deck's title line.
    pub fn title(self: *const Problem) []const u8 {
        return self.prepared.deck.title;
    }

    /// Device instances in the flattened circuit, subcircuits expanded.
    pub fn device_count(self: *const Problem) u32 {
        return self.prepared.deck.n_devices;
    }

    /// The output failure that stopped publication, if any.
    pub fn output_error(self: *const Problem) ?DeliveryError {
        return self.delivery_error;
    }

    /// True once every requested query has ended, completed or not: the
    /// stop condition of a stepping loop over `ready_queries`.
    pub fn finished(self: *const Problem) bool {
        return self.session.finished();
    }

    /// Queries in the DAG, prerequisites included. IDs run `0..query_count()`.
    pub fn query_count(self: *const Problem) u32 {
        return self.session.count();
    }

    /// A snapshot of one query's state. `error.InvalidQuery` for an ID
    /// outside `0..query_count()`.
    pub fn query_info(self: *const Problem, id: QueryId) !QueryInfo {
        return self.session.info(id);
    }

    /// How many queries in `scope` can advance now: the length
    /// `ready_queries` needs.
    pub fn ready_count(self: *const Problem, scope: Scope) !usize {
        return self.session.readyCount(scope);
    }

    /// Writes the queries in `scope` that can advance now into `ids` and
    /// returns how many there are. Fails with `error.BufferTooSmall`, `ids`
    /// untouched, when `ids` is shorter than `ready_count`.
    pub fn ready_queries(self: *const Problem, scope: Scope, ids: []QueryId) !usize {
        return self.session.readyQueries(scope, ids);
    }

    /// Adds `queries` and their prerequisites to the DAG, writing each one's
    /// ID into `ids`. Returns `queries.len`.
    ///
    /// When `ids` is too short this is a sizing call: nothing is validated or
    /// added. Otherwise the append is all-or-nothing: a query the output
    /// format cannot hold, or any other failure, leaves the DAG and `ids`
    /// unchanged. An `.op` identical to an existing OP, an implicit
    /// prerequisite included, returns that OP's ID; every other query gets
    /// a new ID. After `set_params`, a query that names no variant of its
    /// own runs at the values the last call set.
    pub fn append_queries(self: *Problem, queries: []const Query, ids: []QueryId) !usize {
        if (ids.len < queries.len) return queries.len;
        const jobs: []const Query = if (self.variant) |v| blk: {
            const copy = try self.allocator.dupe(Query, queries);
            for (copy) |*q| switch (q.*) {
                inline else => |*o| if (o.tol.variant == null) {
                    o.tol.variant = v;
                },
            };
            break :blk copy;
        } else queries;
        defer if (self.variant != null) self.allocator.free(jobs);
        try self.checkOutput(&self.prepared, jobs);
        return self.session.append(jobs, ids);
    }

    /// Sets each global `.param` in `names` to the matching value for
    /// every query appended afterwards; the others keep their last values.
    /// Queries appended earlier, finished or not, keep the values they were
    /// appended with, so a sizing loop appends its analyses again after
    /// each call (their plots are named `<plot> (<name>=<value>, ...)`).
    ///
    /// Nothing is re-parsed: the point is re-evaluated from the prepared
    /// netlist like a `.step` point, a binned MOSFET whose L or W moves
    /// takes its new bin, and the circuit is rebuilt only when parameter
    /// writes cannot express the point (dev/analysis/variants.md §5).
    ///
    /// The first call also runs the deck's pending `.model OPT` optimization
    /// and MOSRA stress run, so it can take as long as they do and fail with
    /// their errors.
    ///
    /// All-or-nothing. Fails with `error.UnknownParameter` for a name not
    /// in `Options.live_params` (nor a `.step` name), `error.InvalidArgument`
    /// when the lengths differ or a value is not finite,
    /// `error.ModelBinNotFound` when no bin holds a new geometry, and
    /// `error.UnsupportedCard` when the values would flip a top-level `.if`
    /// or change the topology. Each call keeps one row until `deinit`, and
    /// each distinct bin selection it reaches keeps one circuit.
    pub fn set_params(self: *Problem, names: []const []const u8, values: []const f64) !void {
        const tuning = self.prepared.tuning orelse return error.UnknownParameter;
        if (names.len != values.len) return error.InvalidArgument;
        // The optimizer and MOSRA rewrite the variant table once; let them
        // run before rows are appended to it.
        try self.optimize();
        try self.age();
        try self.session.templates.ensureUnusedCapacity(self.session.allocator, 1);
        const change = try tuning.set(&self.prepared.circuit, self.prepared.deck.variants, names, values);
        if (change.built) |c| self.session.addTemplate(c) catch unreachable;
        std.debug.assert(change.template <= self.session.templates.items.len);
        self.session.template = change.template;
        self.prepared.deck.variants = change.variants;
        self.variant = change.variants.count() - 1;
    }

    /// Appends `count` points, each the deck's analyses run at one row of
    /// `values` (`count` rows of `Options.live_params` values, row-major),
    /// as a `.step` point would run them, and returns point 0's first
    /// query id. Points are consecutive in request order (implicit
    /// operating points sit between their ids). Each point is one
    /// `set_params` row; a point it refuses is still appended, its queries
    /// failed with `TopologyChanged` (a topology change or a flipped
    /// top-level `.if`), `ModelBinNotFound`, `NonFiniteParameter` or, for
    /// any other error building its circuit, `PointRefused`. The
    /// first call also runs the deck's `.model OPT` and MOSRA runs.
    ///
    /// All-or-nothing. `error.NoLiveParameters` without
    /// `Options.live_params`, `error.SweptTemplate` for a deck whose
    /// analyses already sweep (`.step`, `.data`, `.alter`, Monte Carlo),
    /// `error.InvalidArgument` for zero points, `error.TangentsUnavailable`
    /// for `options.tangent_step` on a transient that cannot integrate
    /// tangents (`tangentable`, a Gear-3..6 method). Asserts that `values`
    /// holds `count` rows.
    ///
    /// With `options.tangent_step` h, each point's transient also integrates
    /// the first-order prediction (`Tran.tangents`) of the point with live
    /// parameter k moved by h·|value| (h when the value is 0), one per
    /// parameter; `point_tangents` reads the measures' slopes off them.
    pub fn append_points(self: *Problem, values: []const f64, count: usize, options: requests.PointOptions) !QueryId {
        const tuning = self.prepared.tuning orelse return error.NoLiveParameters;
        const p = self.live.len;
        std.debug.assert(values.len == count * p);
        if (count == 0) return error.InvalidArgument;
        try self.checkPoint(options);
        const deck = &self.prepared.deck;
        for (deck.queries) |job| switch (job) {
            inline else => |o| if (o.tol.variant != null) return error.SweptTemplate,
        };
        try self.optimize();
        try self.age();
        const template_jobs = if (deck.queries.len == 0) &[_]Query{.{ .op = .{ .tol = deck.deck_tol } }} else deck.queries;
        try self.checkOutput(&self.prepared, template_jobs);
        const tangent = options.tangent_step > 0;
        if (tangent) for (template_jobs) |job| if (job == .tran and !self.tangentable(job)) return error.TangentsUnavailable;
        if (options.warm_start) try self.solveWarmOp(template_jobs);
        var arena = std.heap.ArenaAllocator.init(self.allocator);
        defer arena.deinit();
        const a = arena.allocator();
        const j = template_jobs.len;
        const jobs = try a.alloc(Query, count * j);
        const templates = try a.alloc(u32, count * j);
        const refused = try a.alloc(?analysis.session.PointError, count);
        // Each point's tangent rows and steps, kept with its transient.
        const steps = try a.alloc([]const f64, count);
        // A built template becomes the session's current one; `set_params`
        // rows appended later keep theirs.
        const current = self.session.template;
        defer self.session.template = current;
        // Room to mark every row this call makes, a point's own and its
        // tangent rows, before it makes any; a failed call's rows are freed.
        const rows_before = deck.variants.count();
        const rows_most = rows_before + count * (1 + if (tangent) p else 0);
        if (self.point_rows.bit_length < rows_most) try self.point_rows.resize(self.allocator, rows_most, false);
        if (self.freed_rows.bit_length < rows_most) try self.freed_rows.resize(self.allocator, rows_most, false);
        var appended = false;
        errdefer if (!appended) self.freed_rows.setRangeValue(.{ .start = rows_before, .end = deck.variants.count() }, true);
        for (0..count) |i| {
            try self.session.templates.ensureUnusedCapacity(self.session.allocator, 1);
            refused[i] = null;
            const template = if (tuning.setAt(&self.prepared.circuit, deck.variants, self.live, values[i * p ..][0..p])) |change| blk: {
                if (change.built) |c| self.session.addTemplate(c) catch unreachable;
                deck.variants = change.variants;
                break :blk change.template;
            } else |err| blk: {
                refused[i] = switch (err) {
                    error.UnsupportedCard => error.TopologyChanged,
                    error.ModelBinNotFound => error.ModelBinNotFound,
                    error.InvalidArgument => error.NonFiniteParameter,
                    error.OutOfMemory => |e| return e,
                    // Any other failure belongs to this point's values (a
                    // device rejecting them as it builds), not the batch.
                    else => |e| blk2: {
                        std.log.warn("point {d}: {s}", .{ i, @errorName(e) });
                        break :blk2 error.PointRefused;
                    },
                };
                deck.variants = try tuning.refusedRow(deck.variants);
                break :blk 0;
            };
            const row = deck.variants.count() - 1;
            const rows: []const u32 = if (tangent and refused[i] == null) try self.tangentRows(tuning, values[i * p ..][0..p], options.tangent_step, &steps[i]) else &.{};
            if (rows.len == 0) steps[i] = &.{};
            for (template_jobs, jobs[i * j ..][0..j], templates[i * j ..][0..j]) |job, *copy, *t| {
                copy.* = job;
                switch (copy.*) {
                    inline else => |*o| o.tol.variant = row,
                }
                if (copy.* == .tran) {
                    copy.tran.point = options;
                    copy.tran.tangents = rows;
                }
                t.* = template;
            }
        }
        const ids = try a.alloc(QueryId, jobs.len);
        _ = try self.session.appendAt(jobs, templates, ids);
        appended = true;
        self.point_rows.setRangeValue(.{ .start = rows_before, .end = deck.variants.count() }, true);
        for (refused, 0..) |r, i| if (r) |err| for (ids[i * j ..][0..j]) |id| self.session.refuse(id, err) catch unreachable;
        // A point owns its OP: nothing appended later reads either.
        for (ids) |id| self.session.markFinal(id) catch unreachable;
        if (tangent) for (jobs, ids, 0..) |job, id, k| if (job == .tran and steps[k / j].len != 0) {
            try self.tangent_steps.put(self.arena.allocator(), id, steps[k / j]);
        };
        return ids[0];
    }

    /// `error.UnknownMeasure` for a `point.abort_measure` no card names.
    fn checkPoint(self: *const Problem, point: requests.PointOptions) !void {
        if (point.abort_measure.len != 0 and measureIndex(self.prepared.deck.measures, point.abort_measure) == null) return error.UnknownMeasure;
    }

    /// One variant row per live parameter of the point at `values`, that
    /// parameter moved by `h`·|value| (`h` at 0), and the steps into
    /// `steps.*`; a row `set_params` refuses gets a NaN step. Both live in
    /// the Problem's arena: the queries hold the rows.
    fn tangentRows(self: *Problem, tuning: @typeInfo(@FieldType(frontend.Prepared, "tuning")).optional.child, values: []const f64, h: f64, steps: *[]const f64) ![]const u32 {
        const a = self.arena.allocator();
        const deck = &self.prepared.deck;
        const rows = try a.alloc(u32, values.len);
        const dp = try a.alloc(f64, values.len);
        const moved = try a.dupe(f64, values);
        defer a.free(moved);
        for (rows, dp, 0..) |*row, *d, k| {
            d.* = if (values[k] == 0) h else h * @abs(values[k]);
            moved[k] = values[k] + d.*;
            defer moved[k] = values[k];
            try self.session.templates.ensureUnusedCapacity(self.session.allocator, 1);
            if (tuning.setAt(&self.prepared.circuit, deck.variants, self.live, moved)) |change| {
                if (change.built) |c| self.session.addTemplate(c) catch unreachable;
                deck.variants = change.variants;
            } else |err| switch (err) {
                error.UnsupportedCard, error.ModelBinNotFound, error.InvalidArgument => {
                    deck.variants = try tuning.refusedRow(deck.variants);
                    d.* = std.math.nan(f64);
                },
                else => |e| return e,
            }
            row.* = deck.variants.count() - 1;
        }
        steps.* = dp;
        return rows;
    }

    /// Solves the template's nominal operating point into `Deck.warm_op`,
    /// once, with the tolerances of the first transient in `jobs`; nothing
    /// without one. A nominal point that does not converge leaves it empty,
    /// so every point starts cold.
    fn solveWarmOp(self: *Problem, jobs: []const Query) !void {
        if (self.prepared.deck.warm_op.len != 0) return;
        // Nothing reads it when every job starts from `.ic` (`uic`): on
        // chain_bsim4_100k that cold OP alone ran for hours before a `uic`
        // transient that needs none.
        for (jobs) |job| {
            if (!job.uic()) break;
        } else return;
        // A transient's flavour when the deck has one, else the first query's.
        const tran_op, var tol = for (jobs) |job| {
            if (job == .tran) break .{ true, job.tran.tol };
        } else switch ((if (jobs.len == 0) return else jobs[0])) {
            inline else => |o| .{ false, o.tol },
        };
        tol.warm_op = false;
        tol.variant = null;
        tol.temp_c = null;
        var deck = self.prepared.deck;
        var session = analysis.session.Session.init(self.workerAllocator(), self.io, &self.prepared.circuit, &deck, self.session.config);
        defer session.deinit();
        var id: [1]QueryId = undefined;
        _ = try session.append(&.{.{ .op = .{ .tol = tol, .tran_op = tran_op } }}, &id);
        try session.runAll(.{ .quantum = .completion });
        const x = try session.operatingPoint(id[0]) orelse return;
        self.prepared.deck.warm_op = try self.arena.allocator().dupe(f64, x);
    }

    /// Why a point produced no value, or not all of them: the C API's
    /// ESPICE_REASON_* codes.
    pub const Reason = enum(u32) {
        ok = 0,
        no_convergence = 1,
        timestep_too_small = 2,
        singular_matrix = 3,
        measure_not_found = 4,
        measure_unresolved = 5,
        aborted = 6,
        model_bin_not_found = 7,
        topology_changed = 8,
        invalid_parameter = 9,
        device_refused = 10,
        other = 255,

        /// The reason a query that failed with `err` gives.
        pub fn of(err: anyerror) Reason {
            return switch (err) {
                error.OpDidNotConverge, error.HbDidNotConverge, error.QpssDidNotConverge, error.PzDidNotConverge, error.MatexDidNotConverge, error.EnvelopeDidNotConverge, error.PacDidNotConverge => .no_convergence,
                error.TimestepTooSmall => .timestep_too_small,
                error.SingularMatrix, error.Singular, error.FloatingNode => .singular_matrix,
                error.MeasureBoundExceeded => .aborted,
                error.ModelBinNotFound => .model_bin_not_found,
                error.TopologyChanged, error.NoiseTopologyChanged => .topology_changed,
                error.NonFiniteParameter => .invalid_parameter,
                error.DeviceRefused => .device_refused,
                // `err` is anyerror: the full name stays with the query
                // (`QueryInfo.failure`).
                else => .other,
            };
        }
    };

    /// Writes the `.meas` cards `names` (case-insensitive) of `reasons.len`
    /// points into `out` (`reasons.len` rows of `names.len`, row-major),
    /// and each point's reason. A point is a run of consecutive requested
    /// queries on one variant row, starting at `first`: each point of
    /// `append_points` or of a swept deck, or the whole of an unswept deck.
    /// A card reads the point's first analysis of its kind, and an `op`
    /// card the point's own operating point when it has no `.op`. A point
    /// that failed reads NaN everywhere with its failure's reason; else NaN
    /// where a card found no value, the reason naming the first such card
    /// (`measure_not_found` for a missing vector or analysis,
    /// `measure_unresolved` for an event that never happened).
    ///
    /// `error.UnknownMeasure` for a name no card has, `error.InvalidQuery`
    /// for a `first` that is not a requested query or fewer points after
    /// it, `error.ResultUnavailable` while one of them is unfinished;
    /// nothing is written then. Asserts `out.len == reasons.len * names.len`.
    pub fn point_measures(self: *const Problem, first: QueryId, names: []const []const u8, out: []f64, reasons: []Reason) !void {
        std.debug.assert(out.len == reasons.len * names.len);
        const session = &self.session;
        const measures = self.prepared.deck.measures;
        var arena = std.heap.ArenaAllocator.init(self.allocator);
        defer arena.deinit();
        const a = arena.allocator();
        const cards = try a.alloc(u32, names.len);
        for (names, cards) |name, *card| card.* = measureIndex(measures, name) orelse return error.UnknownMeasure;
        const outputs = session.outputs.items;
        const ends = try self.pointEnds(a, first, reasons.len);
        const nan = std.math.nan(f64);
        const values = try a.alloc(f64, measures.len);
        const missing = try a.alloc(bool, measures.len);
        for (reasons, 0..) |*reason, k| {
            const row = out[k * names.len ..][0..names.len];
            @memset(row, nan);
            const ids = outputs[ends[k]..ends[k + 1]];
            reason.* = for (ids) |id| {
                if (try queryFailure(session, id)) |err| break .of(err);
            } else .ok;
            if (reason.* != .ok) continue;
            var evaluated: ?QueryId = null;
            for (cards, row) |card, *dst| {
                const kind = measures[card].analysis;
                var source: ?QueryId = null;
                for (ids) |id| if ((try session.info(id)).kind == kind) {
                    source = id;
                    break;
                };
                if (source == null and kind == .op) for (ids) |id| if ((try session.info(id)).dependency) |dep| {
                    source = dep;
                    break;
                };
                var why: Reason = .measure_not_found;
                if (source) |q| {
                    // Cards of one analysis share its evaluation.
                    if (evaluated != q) output.measureOutcomes(measures, kind, try block(&self.session, q, 0), values, missing);
                    evaluated = q;
                    dst.* = values[card];
                    if (!missing[card]) why = .measure_unresolved;
                }
                if (std.math.isNan(dst.*) and reason.* == .ok) reason.* = why;
            }
        }
    }

    /// Where each of `count` points from `first` lies in `session.outputs`:
    /// point k's queries are `outputs[ends[k]..ends[k + 1]]`, one variant
    /// row's. `error.InvalidQuery` for a `first` that is not requested or
    /// fewer points after it, `error.ResultUnavailable` while one of them
    /// has not ended.
    fn pointEnds(self: *const Problem, a: std.mem.Allocator, first: QueryId, count: usize) ![]usize {
        const session = &self.session;
        const outputs = session.outputs.items;
        const ends = try a.alloc(usize, count + 1);
        ends[0] = try session.outputIndex(first) orelse return error.InvalidQuery;
        for (ends[1..], ends[0..count]) |*end, begin| {
            if (begin == outputs.len) return error.InvalidQuery;
            const row = variantOf(try session.query(outputs[begin]));
            var e = begin;
            while (e < outputs.len and variantOf(try session.query(outputs[e])) == row) : (e += 1) {
                if (!(try session.info(outputs[e])).status.terminal()) return error.ResultUnavailable;
            }
            end.* = e;
        }
        return ends;
    }

    /// Frees `count` points from `first` (as `point_measures` counts
    /// them): each query's circuit, work and result, its OP's, and its
    /// tangent steps, so a long-lived Problem appended to batch after batch
    /// stays the same size. Each query keeps its id and status;
    /// `result`, `point_measures` and `point_tangents` then fail with
    /// `error.ResultUnavailable`, and nothing publishes it. A point already
    /// released is left alone. The variant rows of `append_points` points
    /// go too, at a call made while no query runs; a query appended with
    /// one of them (`tol.variant`) must not outlive its point, and is
    /// refused once it is gone. Errors as `point_measures`.
    // ponytail: each query's row (a few hundred bytes) stays until `deinit`;
    // drop a released prefix of them if sessions reach billions of points.
    // Rows drop as a prefix: a `set_params` row is never freed, so no row
    // after it is dropped.
    pub fn release_points(self: *Problem, first: QueryId, count: usize) !void {
        var arena = std.heap.ArenaAllocator.init(self.allocator);
        defer arena.deinit();
        const ends = try self.pointEnds(arena.allocator(), first, count);
        const outputs = self.session.outputs.items;
        // Dependents before the OPs they read.
        var k = ends[count];
        while (k > ends[0]) {
            k -= 1;
            try self.session.release(outputs[k]);
            _ = self.tangent_steps.remove(outputs[k]);
        }
        for (outputs[ends[0]..ends[count]]) |id| {
            const job = try self.session.query(id);
            if (variantOf(job)) |v| self.freeRow(v);
            if (job == .tran) for (job.tran.tangents) |v| self.freeRow(v);
        }
        self.dropFreedRows();
    }

    /// Marks variant row `v` freed when `append_points` made it.
    fn freeRow(self: *Problem, v: u32) void {
        if (v < self.point_rows.bit_length and self.point_rows.isSet(v)) self.freed_rows.set(v);
    }

    /// Drops the variant table's prefix of freed rows when no query runs
    /// (a running one reads its own copy of the table) and the prefix is at
    /// least as long as the rows kept, so each row moves O(1) times. A
    /// table that cannot move keeps its rows.
    fn dropFreedRows(self: *Problem) void {
        const tuning = self.prepared.tuning orelse return;
        const rows = &self.prepared.deck.variants;
        var below = rows.base;
        while (below < self.freed_rows.bit_length and self.freed_rows.isSet(below)) below += 1;
        if (below == rows.base or below - rows.base < rows.count() - below or !self.session.idle()) return;
        rows.* = tuning.drop(self.allocator, below) catch return;
    }

    /// Most points the executor runs together as one lane batch on this
    /// Problem's backend (`analysis.batchWidth`). Advisory.
    pub fn lane_width(self: *const Problem) u32 {
        return analysis.batchWidth(self.session.config);
    }

    /// The slopes d(card)/d(live parameter) of the `.tran` cards `names` at
    /// `count` points appended with `PointOptions.tangent_step`, starting
    /// at `first`, into `out` (`count` × `names.len` × live parameters,
    /// row-major): each card over the parameter's tangent prediction minus
    /// over the point's own run, over the step. NaN where either has no
    /// value, for a point that failed or was appended without tangents,
    /// for a card of another analysis, and for a step `set_params` refused.
    /// Errors as `point_measures`. Asserts `out.len == count * names.len *
    /// live parameters`.
    pub fn point_tangents(self: *const Problem, first: QueryId, count: usize, names: []const []const u8, out: []f64) !void {
        const p = self.live.len;
        std.debug.assert(out.len == count * names.len * p);
        const measures = self.prepared.deck.measures;
        const a = self.allocator;
        // Validates the names and points and gives each point's reason.
        const base = try a.alloc(f64, count * names.len);
        defer a.free(base);
        const reasons = try a.alloc(Reason, count);
        defer a.free(reasons);
        try self.point_measures(first, names, base, reasons);
        const cards = try a.alloc(u32, names.len);
        defer a.free(cards);
        for (names, cards) |name, *card| card.* = measureIndex(measures, name).?; // `point_measures` checked every name
        const values = try a.alloc(f64, measures.len);
        defer a.free(values);
        const v0 = try a.alloc(f64, measures.len);
        defer a.free(v0);
        @memset(out, std.math.nan(f64));
        const outputs = self.session.outputs.items;
        var at = (try self.session.outputIndex(first)).?;
        for (0..count) |i| {
            // Point i's queries share one variant row; its transient carries the steps.
            const row = variantOf(try self.session.query(outputs[at]));
            var tran: ?QueryId = null;
            while (at < outputs.len and variantOf(try self.session.query(outputs[at])) == row) : (at += 1) {
                if (tran == null and self.tangent_steps.contains(outputs[at])) tran = outputs[at];
            }
            const q = tran orelse continue;
            if ((try self.session.info(q)).status != .complete) continue;
            const steps = self.tangent_steps.get(q).?;
            std.debug.assert(steps.len == p);
            output.measureValues(measures, .tran, try block(&self.session, q, 0), v0);
            for (steps, 0..) |h, k| {
                output.measureValues(measures, .tran, try block(&self.session, q, k + 1), values);
                for (cards, 0..) |card, c| out[(i * names.len + c) * p + k] = (values[card] - v0[card]) / h;
            }
        }
    }

    /// Refuses, before anything runs, the first of `queries` whose result
    /// the output format cannot hold over `prep`'s circuit and deck.
    fn checkOutput(self: *const Problem, prep: *const frontend.Prepared, queries: []const Query) !void {
        for (queries) |query| {
            const schema = try analysis.schemaOf(self.allocator, &prep.circuit, &prep.deck, query);
            try output.validateQuery(self.delivery.selection.format, schema, prep.deck.title, prep.deck.probe_labels);
        }
    }

    /// `append_queries` for SPICE analysis directives (`.ac dec 10 1 1meg`),
    /// resolved against the prepared circuit. A device card is
    /// `error.UnsupportedDirectiveMutation`; text with no analysis card is
    /// `error.InvalidAnalysisArguments`.
    pub fn append_directives(self: *Problem, text: []const u8, ids: []QueryId) !usize {
        var scratch = std.heap.ArenaAllocator.init(self.allocator);
        defer scratch.deinit();
        const jobs = try frontend.resolveQueries(scratch.allocator(), &self.prepared, text);
        return self.append_queries(jobs, ids);
    }

    /// Runs one quantum of `id`, or of its first unfinished prerequisite, then
    /// publishes whatever completed; in binary or ASCII raw or `.print` the
    /// plots reach the file in one replacement when the last query ends,
    /// as with `run_all` (#179). A terminal `id` returns its status without
    /// running anything. A numerical failure is reported in the
    /// event, not as an error, and an output failure in its
    /// `delivery_error`. `error.InvalidQuery` for an ID outside
    /// `0..query_count()`. The first of `advance`, `advance_ready`,
    /// `run_all` and `set_params` also runs the deck's `.model OPT`
    /// optimization and MOSRA stress run.
    pub fn advance(self: *Problem, id: QueryId) !Advance {
        try self.optimize();
        try self.age();
        var event = try self.session.advance(id);
        self.deliverFrom(&self.session, &self.next_output);
        self.finishWhenDone();
        event.delivery_error = self.delivery_error;
        return event;
    }

    /// Advances every query in `ids` by one quantum, up to
    /// `limits.max_parallel` at a time, writing one event per query in `ids`
    /// order, and returns `ids.len`. `ids` must be ready and distinct
    /// (`error.QueryNotReady`, `error.DuplicateQuery`), `limits.max_parallel`
    /// nonzero (`error.InvalidConcurrency`) and `events` at least as long
    /// as `ids` (`error.BufferTooSmall`); a bad frontier starts nothing.
    /// Publishes as `advance` does.
    pub fn advance_ready(self: *Problem, ids: []const QueryId, limits: Limits, events: []Advance) !usize {
        const n = try self.stepReady(ids, limits, events);
        self.finishWhenDone();
        for (events[0..n]) |*event| event.delivery_error = self.delivery_error;
        return n;
    }

    /// `advance_ready` without finishing the output, which `run_all` does
    /// itself once the `.step` runs have published too.
    fn stepReady(self: *Problem, ids: []const QueryId, limits: Limits, events: []Advance) !usize {
        try self.optimize();
        try self.age();
        const n = try self.session.advanceReady(ids, limits, events);
        self.deliverFrom(&self.session, &self.next_output);
        return n;
    }

    /// Finishes the output once every query has ended and published, so
    /// a caller that steps to the end gets the file `run_all` writes.
    /// `run_all` alone runs the `.step` variant runs, and publishes their
    /// plots in a later `finish`.
    fn finishWhenDone(self: *Problem) void {
        if (self.delivery_error != null or !self.finished()) return;
        if (self.next_output != self.session.outputs.items.len) return;
        self.delivery.finish(self.io) catch |err| {
            self.delivery_error = err;
        };
    }

    /// Runs every query to completion, publishes the results and finishes
    /// the output, then writes the `.save` and `.stim` files. Returns the
    /// first query failure, else `error.DeliveryFailed` if output failed,
    /// once every query has ended; no `.save` or `.stim` file is written
    /// then. Calling it again is a no-op for finished queries and publishes
    /// nothing twice.
    pub fn run_all(self: *Problem) !void {
        var lap = if (self.session.config.timing_in_depth) std.Io.Timestamp.now(self.io, .awake) else null;
        defer timingLap(self.io, &lap, "run total (analysis, scheduling, output)");
        try self.optimize();
        try self.age();
        try self.openStream();
        if (self.sidesAllowed()) self.session.stream_opener = .{ .ctx = self, .open = openSide };
        // Whole-query quanta: nothing here reads intermediate progress,
        // except the in-depth timing, which reports per checkpoint. The
        // variant runs share the pool, after the main session's ready queries.
        var pool: analysis.session.Pool = try .init(self.allocator, self.io, .{ .max_parallel = self.limits.max_parallel, .quantum = if (self.session.config.timing_in_depth) .interval else .completion });
        defer pool.deinit();
        try pool.add(&self.session);
        for (self.runs) |*run| try pool.add(&run.session);
        while (true) {
            pool.fill(&self.session);
            for (self.runs) |*run| pool.fill(&run.session);
            if (!pool.wait()) break;
            // Refill the slots `wait` freed before encoding and printing,
            // so workers run while the coordinator delivers.
            pool.fill(&self.session);
            for (self.runs) |*run| pool.fill(&run.session);
            self.deliverRuns();
            self.printReady();
            self.releaseDelivered();
        }
        self.deliverRuns();
        self.printReady();
        self.releaseDelivered();
        if (!self.finished()) return error.SchedulingFailure;
        for (self.runs) |*run| if (!run.session.finished()) return error.SchedulingFailure;
        if (self.first_failure == null) self.first_failure = self.session.failureFrom(self.checked_outputs);
        self.checked_outputs = self.session.outputs.items.len;
        if (self.first_failure) |err| return err;
        for (self.runs) |*run| if (run.session.failure()) |err| return err;
        if (self.delivery_error != null) return error.DeliveryFailed;
        if (self.session.config.timing_in_depth) {
            const mem = self.queryMemory();
            std.debug.print("timing: query memory: peak {d} B, {d} B held at the end\n", .{ mem.peak, mem.live });
        }
        var output_lap = if (self.session.config.timing_in_depth) std.Io.Timestamp.now(self.io, .awake) else null;
        defer timingLap(self.io, &output_lap, "output finish");
        self.delivery.finish(self.io) catch |err| {
            self.delivery_error = err;
            return error.DeliveryFailed;
        };
        // Both build their text in a `Writer.Allocating`, which reports
        // running out of memory as `error.WriteFailed`.
        if (self.save_path) |path| self.saveOperatingPoint(path, self.prepared.deck.save_op.?) catch |err|
            return if (err == error.WriteFailed) error.OutOfMemory else err;
        for (self.prepared.deck.stims) |stim| self.writeStim(stim) catch |err|
            return if (err == error.WriteFailed) error.OutOfMemory else err;
    }

    /// Writes an HSPICE `.stim` card's file from the main session's first
    /// transient: one PWL source per signal, or one `.data` table with a row
    /// per time. No transient only warns.
    fn writeStim(self: *Problem, stim: core.Stim) !void {
        const res = for (self.session.outputs.items) |id| {
            const info = try self.session.info(id);
            if (info.status == .complete and info.kind == .tran) {
                const r = try self.session.result(id);
                if (r.npoints != 0) break r;
            }
        } else {
            std.log.warn(".stim: no transient result to write", .{});
            return;
        };
        const gpa = self.allocator;
        var times: std.ArrayList(f64) = .empty;
        defer times.deinit(gpa);
        const width = res.varnames.len;
        try times.ensureTotalCapacity(gpa, if (stim.times.len != 0) stim.times.len else if (stim.npoints > 1) stim.npoints else res.npoints);
        if (stim.times.len != 0) {
            times.appendSliceAssumeCapacity(stim.times);
        } else if (stim.npoints > 1) {
            const to = @min(stim.to, res.data[(res.npoints - 1) * width]);
            for (0..stim.npoints) |k| times.appendAssumeCapacity(stim.from + (to - stim.from) * @as(f64, @floatFromInt(k)) / @as(f64, @floatFromInt(stim.npoints - 1)));
        } else for (0..res.npoints) |k| {
            const t = res.data[k * width];
            if (t >= stim.from and t <= stim.to) times.appendAssumeCapacity(t);
        }
        // Column-major: signal k's samples at values[k * n ..].
        const n = times.items.len;
        const values = try gpa.alloc(f64, n * stim.signals.len);
        defer gpa.free(values);
        for (stim.signals, 0..) |m, k| output.sampleMeasure(m.first, res, times.items, values[k * n ..][0..n]) catch {
            std.log.warn(".stim: no such vector for {s}", .{m.name});
            return;
        };
        var text: std.Io.Writer.Allocating = .init(gpa);
        defer text.deinit();
        const w = &text.writer;
        try w.print("* {s}\n* written by .stim\n", .{self.prepared.deck.title});
        if (stim.data) {
            try w.print(".data {s}{s}", .{ stim.dataname, if (stim.indepout) " time" else "" });
            for (stim.signals) |m| try w.print(" {s}", .{m.name});
            for (times.items, 0..) |t, i| {
                try w.writeAll("\n+");
                if (stim.indepout) try w.print(" {e}", .{t});
                for (0..stim.signals.len) |k| try w.print(" {e}", .{values[k * n + i]});
            }
            try w.writeAll("\n.enddata\n");
        } else for (stim.signals, stim.nodes, 0..) |m, pair, k| {
            try w.print("{s} {s} {s} pwl(", .{ m.name, pair[0], pair[1] });
            for (times.items, 0..) |t, i| try w.print("\n+ {e} {e}", .{ t, values[k * n + i] });
            try w.writeAll(")\n");
        }
        const name = try std.fmt.allocPrint(gpa, "{s}.{s}{d}_tr0", .{ stim.file orelse std.fs.path.stem(self.origin), if (stim.data) "dat" else "pwl", stim.serial });
        defer gpa.free(name);
        const path = try std.fs.path.resolve(gpa, &.{ self.out_dir, name });
        defer gpa.free(path);
        try std.Io.Dir.cwd().writeFile(self.io, .{ .sub_path = path, .data = text.written() });
    }

    /// Under a final plan (the CLI), a transient or AC sweep that publishes
    /// first and that nothing reads afterwards (no `.meas` over it, no
    /// `.save` or `.stim`, no optimization) writes its rows straight to a binary raw
    /// output, or drops them without one, instead of keeping them all in
    /// memory. Its Result then has empty `data`. Plotname and scale column
    /// come from `analysis.schemaOf`; `endStream` checks the row count.
    fn openStream(self: *Problem) !void {
        const deck = &self.prepared.deck;
        if (!self.sidesAllowed() or self.next_output != 0 or self.streaming != null) return;
        if (self.session.outputs.items.len == 0) return;
        const id = self.session.outputs.items[0];
        // A `.four`/`.fft` reads a `.tran`'s rows after it ends (#231).
        if ((try self.session.info(id)).status != .pending or self.session.resultRead(id)) return;
        const job = try self.session.query(id);
        for (deck.measures) |m| if (m.analysis == job) return;
        const schema = try analysis.schemaOf(self.allocator, &self.prepared.circuit, deck, job);
        if (schema.plotname.len == 0) return;
        switch (job) {
            inline else => |o| if (o.tol.temp_c != null or o.tol.variant != null) return,
        }
        const names = try self.allocator.alloc([]const u8, deck.probe_labels.len + 1);
        defer self.allocator.free(names);
        names[0] = schema.scale;
        @memcpy(names[1..], deck.probe_labels);
        const writer = try self.delivery.beginStream(self.io, .{ .title = deck.title, .result = .{
            .plotname = schema.plotname,
            .varnames = names,
            .is_complex = job == .ac,
            .npoints = 0,
            .data = &.{},
        } }) orelse return;
        if (!try self.session.streamTo(id, writer)) return self.delivery.abortStream();
        self.streaming = id;
    }

    /// What `openStream` asks of the whole run before any plot streams: a
    /// final plan, nothing that reads results after the run, and a probe
    /// label per column, with no device operating-point variable computed
    /// from the finished rows.
    fn sidesAllowed(self: *const Problem) bool {
        const deck = &self.prepared.deck;
        return self.session.config.final_plan and self.save_path == null and deck.stims.len == 0 and
            self.prepared.tuner == null and deck.soa.limits.len == 0 and deck.probe_labels.len == deck.probes.len and deck.opvars.len == 0;
    }

    /// `Session.stream_opener`: a transient or AC plot that no `.meas` card
    /// and no tangent row reads streams to a side file of the output from
    /// its worker, and `deliverOne` appends it in request order, so `-r`
    /// over a sweep holds one row buffer per running point instead of
    /// every waveform. Null (keep the rows) for anything else, or when the
    /// side cannot open.
    fn openSide(ctx: *anyopaque, id: QueryId, job: Query) ?*std.Io.Writer {
        const self: *Problem = @ptrCast(@alignCast(ctx));
        switch (job) {
            .tran => |o| if (o.snapshot or o.tangents.len != 0) return null,
            .ac => {},
            else => return null, // else: only a transient or AC sweep streams its rows
        }
        if (self.streaming == id or self.tangent_steps.contains(id) or self.session.resultRead(id)) return null;
        for (self.prepared.deck.measures) |m| if (m.analysis == job) return null;
        self.sides.ensureUnusedCapacity(self.allocator, 1) catch return null;
        const s = (self.delivery.beginSide(self.io) catch return null) orelse return null;
        self.sides.putAssumeCapacity(id, s);
        return s.writer();
    }

    /// Writes the node voltages of the main session's operating point as
    /// `.nodeset` or `.ic` cards: the first `.op` result, else row 0 of the
    /// first transient (its operating point) or DC sweep (its first point,
    /// as HSPICE saves only a sweep's first). `save.time` > 0 reads the first
    /// transient at that time, interpolated. Nothing to save only warns.
    fn saveOperatingPoint(self: *Problem, path: []const u8, save: core.SaveOp) !void {
        const found: ?struct { result: Result, row: usize, frac: f64 } = for (self.session.outputs.items) |id| {
            const info = try self.session.info(id);
            if (info.status != .complete) continue;
            const res = try self.session.result(id);
            if (res.is_complex or res.npoints == 0) continue;
            if (save.time > 0) {
                if (info.kind != .tran) continue;
                const width = res.varnames.len;
                var i: usize = 0;
                while (i + 1 < res.npoints and res.data[(i + 1) * width] < save.time) i += 1;
                if (i + 1 >= res.npoints) continue;
                const t0 = res.data[i * width];
                const t1 = res.data[(i + 1) * width];
                break .{ .result = res, .row = i, .frac = if (t1 > t0) (save.time - t0) / (t1 - t0) else 0 };
            }
            switch (info.kind) {
                .op, .tran, .dc => break .{ .result = res, .row = 0, .frac = 0 },
                else => {},
            }
        } else null;
        const op = found orelse {
            std.log.warn(".save: no {s} result to save", .{if (save.time > 0) "transient" else "operating point"});
            return;
        };
        var text: std.Io.Writer.Allocating = .init(self.allocator);
        defer text.deinit();
        const w = &text.writer;
        try w.print("* {s}\n* operating point written by .save\n", .{self.prepared.deck.title});
        const width = op.result.varnames.len;
        for (op.result.varnames, 0..) |name, k| {
            if (!std.mem.startsWith(u8, name, "v(") or std.mem.indexOfScalar(u8, name, ',') != null) continue;
            if (save.top_only and std.mem.indexOfScalar(u8, name, '.') != null) continue;
            const v0 = op.result.data[op.row * width + k];
            const v = if (op.frac == 0) v0 else v0 + op.frac * (op.result.data[(op.row + 1) * width + k] - v0);
            try w.print("{s} {s}={e}\n", .{ if (save.ic) ".ic" else ".nodeset", name, v });
        }
        try std.Io.Dir.cwd().writeFile(self.io, .{ .sub_path = path, .data = text.written() });
    }

    /// Writes the query DAG with the frontier `options.preview` would run
    /// next. Changes no state. A null `options.limits` uses the Problem's.
    pub fn print(self: *const Problem, writer: *std.Io.Writer, options: PrintOptions) !void {
        var resolved = options;
        if (resolved.limits == null) resolved.limits = self.limits;
        try self.session.print(writer, resolved);
    }

    /// Writes each `.model OPT` optimization's summary, then the deck's
    /// `.meas` results for every completed result a card can read (tran,
    /// transient noise, AC, DC, FFT, S-parameter, mismatch, loop stability,
    /// phase and periodic noise), in request order, as ngspice prints them,
    /// then the Monte Carlo statistics of each tran, AC and DC card, or its
    /// SIGMA_SCALE report (#239). Cards that cannot be measured are
    /// reported on `err`; a refused SIGMA_SCALE fit fails with
    /// `error.TooFewScales` or `error.Singular` once everything is printed.
    /// With `Options.measures`, it writes only what `run_all` has not
    /// printed there, and returns the first error printing there.
    pub fn print_measures(self: *Problem, out: *std.Io.Writer, err: *std.Io.Writer) !void {
        if (self.measures_to != null) {
            if (self.measure_error) |e| return e;
            try self.printMeasuresFrom(&self.measure_cursor, out, err, false);
        } else {
            // Each call prints every block again, so the JSON starts over.
            if (self.measure_json) |*j| j.clearRetainingCapacity();
            var cursor: MeasureCursor = .{};
            try self.printMeasuresFrom(&cursor, out, err, false);
        }
        try self.writeStatistics(out, false);
    }

    /// Writes every `.meas` result `print_measures` (or `run_all`, with
    /// `Options.measures`) has printed as one JSON document,
    /// `espice-measures/1` (#235): `deck`, the `results` in the printed
    /// order, each with its analysis, its `variant` (`output.writeVariantJson`)
    /// and the `from`/`to`/`at` its line prints, a failed one with a null
    /// `value` and an `error`; then the Monte Carlo `statistics`; then
    /// under `.option soa` every excursion `print_soa` counts (#238), as
    /// `instance`, `quantity`, `key`, `limit`, `peak`, its window `from` and
    /// `to` (null at an operating point, `to` null past the last point), and
    /// its result's `analysis` and `variant`.
    /// `error.MeasuresJsonOff` without `Options.measures_json`.
    pub fn write_measures_json(self: *Problem, out: *std.Io.Writer, deck: []const u8) !void {
        const results = if (self.measure_json) |*j| j.written() else return error.MeasuresJsonOff;
        try out.writeAll("{\n  \"format\": \"espice-measures/1\",\n  \"deck\": ");
        try std.json.Stringify.encodeJsonString(deck, .{}, out);
        // Each object is led by `,\n`; the first one's comma goes.
        try out.print(",\n  \"results\": [{s}\n  ],\n  \"statistics\": [", .{if (results.len == 0) "" else results[1..]});
        var stats: std.Io.Writer.Allocating = .init(self.allocator);
        defer stats.deinit();
        try self.writeStatistics(&stats.writer, true);
        const s = stats.written();
        try out.print("{s}\n  ],\n  \"soa\": [", .{if (s.len == 0) "" else s[1..]});
        var soa: std.Io.Writer.Allocating = .init(self.allocator);
        defer soa.deinit();
        var tally: SoaTally = .{ .gpa = self.allocator, .json = &soa.writer };
        defer tally.instances.deinit(self.allocator);
        try tally.session(&self.session);
        for (self.runs) |*run| try tally.session(&run.session);
        const e = soa.written();
        try out.print("{s}\n  ]\n}}\n", .{if (e.len == 0) "" else e[1..]});
    }

    /// Writes `--emir`'s document `k` (#237), `espice-emir/1`, and returns
    /// true; false, writing nothing, past the last one. One document per
    /// variant (a variant row and temperature of the main session or of a
    /// variant run, in output order): its first completed transient's
    /// terminal currents (`Result.emir`) under its `variant` label
    /// (`output.writeVariantJson`). An instance is named by its card, a
    /// terminal by its device's port and a net by its node; an independent
    /// source reports its first port, under `sources`. `em_avg` is what
    /// `.meas EM_AVG` makes of the same halves with `.option em_recovery`.
    pub fn write_emir_json(self: *const Problem, out: *std.Io.Writer, deck: []const u8, k: usize) !bool {
        var n: usize = 0;
        for (0..1 + self.runs.len) |s| {
            const session = if (s == 0) &self.session else &self.runs[s - 1].session;
            const outputs = session.outputs.items;
            for (outputs, 0..) |id, j| {
                if (!try emirFirst(session, outputs[0..j], id)) continue;
                if (n == k) {
                    try writeEmir(out, deck, &self.library, if (s == 0) &self.prepared else &self.prepared.runs[s - 1], session, id);
                    return true;
                }
                n += 1;
            }
        }
        return false;
    }

    /// The `.meas` statistics over the Monte Carlo trials of each tran, AC
    /// and DC card, as text or as JSON objects; for a SIGMA_SCALE card's
    /// analysis, its scaled-sigma report instead (#239). As text, a refused
    /// fit fails with its error once everything is written.
    fn writeStatistics(self: *const Problem, out: *std.Io.Writer, json: bool) !void {
        const measures = self.prepared.deck.measures;
        if (measures.len == 0) return;
        var trials: std.ArrayList(Result) = .empty;
        defer trials.deinit(self.allocator);
        var refused: ?core.sss.Error = null;
        inline for (.{ .tran, .ac, .dc }) |kind| {
            trials.clearRetainingCapacity();
            try monteTrials(self.allocator, &self.session, kind, &trials);
            for (self.runs) |*run| try monteTrials(self.allocator, &run.session, kind, &trials);
            const scaled = for (self.prepared.sss) |s| {
                if (s.kind == kind) break s;
            } else null;
            if (scaled) |s| {
                if (try writeSss(self.allocator, out, json, measures, s, trials.items)) |e| refused = refused orelse e;
            } else if (trials.items.len != 0) {
                if (json) try output.writeMeasureStatistics(out, measures, kind, trials.items) else try output.printMeasureStatistics(out, measures, kind, trials.items);
            }
        }
        if (refused) |e| if (!json) return e;
    }

    /// With `Options.point.tangent_step`, writes d(card)/d(live parameter)
    /// of every `.tran` card at the deck's first transient
    /// (`point_tangents`), one `d(card)/d(param) = value` line each; NaN
    /// where a card has no value. Nothing without tangents.
    pub fn print_tangents(self: *const Problem, out: *std.Io.Writer) !void {
        const id = for (self.session.outputs.items) |id| {
            if (self.tangent_steps.contains(id)) break id;
        } else return;
        var names: std.ArrayList([]const u8) = .empty;
        defer names.deinit(self.allocator);
        for (self.prepared.deck.measures) |m| if (m.analysis == .tran) try names.append(self.allocator, m.name);
        if (names.items.len == 0) return;
        const p = self.live.len;
        const slopes = try self.allocator.alloc(f64, names.items.len * p);
        defer self.allocator.free(slopes);
        try self.point_tangents(id, 1, names.items, slopes);
        try out.writeAll("\n  Tangents for Transient Analysis\n\n");
        for (names.items, 0..) |name, c| for (self.live_names, 0..) |param, k| {
            try out.print("d({s})/d({s}) = {e:.6}\n", .{ name, param, slopes[c * p + k] });
        };
    }

    /// With `.option soa=1` (ngspice `warn=1`), holds every accepted point
    /// of each completed operating point, DC sweep and transient, the
    /// variant runs' too, to the deck's model-card limits (`core.Soa`): one
    /// `soa:` line per instance, limit and excursion, at most `maxwarns` of
    /// them, then the count of every one; nothing when none happened. An
    /// excursion runs from the first point past the limit to the first one
    /// back at or under it, so a point at the limit passes. A limit whose
    /// terminal a result lacks (left out by `.save`) is counted as skipped.
    pub fn print_soa(self: *const Problem, out: *std.Io.Writer) !void {
        const soa = self.prepared.deck.soa;
        if (soa.limits.len == 0) return;
        var tally: SoaTally = .{ .gpa = self.allocator, .out = out, .max_warns = soa.max_warns };
        defer tally.instances.deinit(self.allocator);
        try tally.session(&self.session);
        for (self.runs) |*run| try tally.session(&run.session);
        if (tally.excursions != 0) try out.print("soa: {d} excursions on {d} instances\n", .{ tally.excursions, tally.instances.count() });
        if (tally.skipped != 0) try out.print("soa: {d} limit checks skipped: a terminal is not saved\n", .{tally.skipped});
    }

    /// Prints to `Options.measures` every `.meas` block whose turn has come
    /// and flushes; the first error stops it for good.
    fn printReady(self: *Problem) void {
        const to = self.measures_to orelse return;
        if (self.measure_error != null) return;
        self.printMeasuresFrom(&self.measure_cursor, to.out, to.err, true) catch |e| {
            self.measure_error = e;
            return;
        };
        to.out.flush() catch |e| {
            self.measure_error = e;
        };
        to.err.flush() catch |e| {
            self.measure_error = e;
        };
    }

    /// The optimization summaries, then each session's `.meas` blocks in
    /// request order (`printQueryMeasures`) from `cursor` on; with `wait`,
    /// it stops at the first result still running, else it skips those.
    fn printMeasuresFrom(self: *Problem, cursor: *MeasureCursor, out: *std.Io.Writer, err: *std.Io.Writer, wait: bool) !void {
        const measures = self.prepared.deck.measures;
        if (!cursor.headed) {
            for (self.optimization, 0..) |*lm, k| try printOptimization(out, lm, &self.prepared.tuner.?, k, measures);
            for (self.bisection, 0..) |*b, k| try printBisection(out, b, &self.prepared.tuner.?, k, measures);
            cursor.headed = true;
        }
        if (measures.len == 0) return;
        while (cursor.session <= self.runs.len) : ({
            cursor.session += 1;
            cursor.next = 0;
        }) {
            const session = if (cursor.session == 0) &self.session else &self.runs[cursor.session - 1].session;
            while (cursor.next < session.outputs.items.len) : (cursor.next += 1) {
                const id = session.outputs.items[cursor.next];
                if (wait and !(try session.info(id)).status.terminal()) return;
                try printQueryMeasures(self.allocator, session, id, out, err, if (self.measure_json) |*j| &j.writer else null, measures);
            }
        }
    }

    /// The completed results of the variant runs with their own topology
    /// (an `.alter` that swaps elements, a point that collapses a node), in
    /// output order. The caller frees the slice with `gpa`; the results stay
    /// valid until `deinit`.
    pub fn run_results(self: *const Problem, gpa: std.mem.Allocator) ![]Result {
        var list: std.ArrayList(Result) = .empty;
        errdefer list.deinit(gpa);
        for (self.runs) |*run| for (run.session.outputs.items) |id| {
            if ((try run.session.info(id)).status == .complete) try list.append(gpa, try run.session.result(id));
        };
        return list.toOwnedSlice(gpa);
    }

    /// A completed query's result, valid until `deinit`.
    /// `error.ResultUnavailable` until the query completes.
    pub fn result(self: *const Problem, id: QueryId) !Result {
        return self.session.result(id);
    }

    /// `id`'s plot name and shape, also after `run_all` released it.
    pub fn summary(self: *const Problem, id: QueryId) !Summary {
        if (self.released.get(id)) |s| return s;
        const res = try self.result(id);
        return .{ .plotname = res.plotname, .npoints = res.npoints, .nvars = res.varnames.len };
    }

    /// `final_plan`: frees each main-session query whose plot is written
    /// and whose `.meas` block is printed, so a `-r` sweep holds the points
    /// still running rather than the whole run. Nothing with `.save` or
    /// `.stim`. Keeps the Monte Carlo
    /// trials (`print_measures` reads them at the end), the tangent runs,
    /// and a requested OP that later queries still read.
    fn releaseDelivered(self: *Problem) void {
        // `.save`, `.stim`, `print_soa` and `write_emir_json` read the results after the run.
        if (!self.session.config.final_plan or self.save_path != null or self.prepared.deck.stims.len != 0 or self.prepared.deck.soa.limits.len != 0 or self.emir) return;
        const outputs = self.session.outputs.items;
        const printed = if (self.prepared.deck.measures.len == 0 or self.measure_cursor.session > 0) outputs.len else self.measure_cursor.next;
        const upto = @min(self.next_output, printed);
        while (self.released_upto < upto) : (self.released_upto += 1) {
            const id = outputs[self.released_upto];
            if (self.tangent_steps.contains(id)) continue;
            const s = self.summary(id) catch continue;
            if (std.mem.indexOf(u8, s.plotname, "(monte=") != null) continue;
            const kept: Summary = .{ .plotname = self.arena.allocator().dupe(u8, s.plotname) catch return, .npoints = s.npoints, .nvars = s.nvars };
            self.released.ensureUnusedCapacity(self.allocator, 1) catch return;
            self.session.release(id) catch continue;
            self.released.putAssumeCapacity(id, kept);
        }
    }

    /// Copies a result's data into `buffer` and returns its length
    /// (`result(id).data.len`); a longer buffer keeps its tail. Fails with
    /// `error.BufferTooSmall`, `buffer` untouched, when it is shorter, and
    /// with `error.ResultUnavailable` until the query completes.
    pub fn copy_result(self: *const Problem, id: QueryId, buffer: []f64) !usize {
        const data = (try self.result(id)).data;
        if (buffer.len < data.len) return error.BufferTooSmall;
        @memcpy(buffer[0..data.len], data);
        return data.len;
    }

    /// Runs the deck's HSPICE optimizations once, before any query advances
    /// (dev/analysis/optimize.md), one per `Tuner.bases` entry in turn.
    /// Each batch of optimizer points is one session of the optimized
    /// card's queries, a variant row per point, run `max_parallel` at a
    /// time. Each optimum's writes then fill the rows its optimized card
    /// and every later card run.
    fn optimize(self: *Problem) !void {
        const t = if (self.prepared.tuner) |*tuner| tuner else return;
        if (self.optimization.len != 0 or self.bisection.len != 0) return;
        const a = self.arena.allocator();
        const deck = &self.prepared.deck;
        const n = t.bases.len;
        if (t.method == .lm) self.optimization = try a.alloc(core.lm.Lm, n) else self.bisection = try a.alloc(core.bisect.Bisect, n);
        const best = try a.alloc(core.Variants, n);
        var template: std.ArrayList(Query) = .empty;
        for (best, 0..) |*row, base| {
            template.clearRetainingCapacity();
            for (deck.queries) |job| switch (job) {
                inline else => |o| if (o.tol.variant == @as(u32, @intCast(base))) try template.append(a, job),
            };
            const x: []const f64 = if (t.method == .lm) blk: {
                // DC and transient goals take their Jacobian from the trial
                // point's own run (`Dc.tangents`, `Tran.tangents`), so every
                // round is batched.
                const direct = !self.session.config.knobs.no_tangents and for (template.items) |job| {
                    if (!self.tangentable(job)) break false;
                } else true;
                const lm = &self.optimization[base];
                if (base == 0) for (t.initial, t.spec.dels, t.names) |x0, del, name| if (t.options.stepOverreach(x0, del)) |ratio|
                    std.log.warn("optimize {s}: {s} = {e} is far below PARMIN = {e}, so its finite-difference step is {d:.0}x the value and can land on a bound; set PARMIN near the parameter's scale", .{ t.spec.name, name, x0, t.options.parmin, ratio });
                lm.* = try core.lm.Lm.init(a, t.initial, t.spec.lo, t.spec.hi, t.spec.dels, t.spec.grid, @intCast(t.results.len), t.options, direct or self.limits.max_parallel > 1);
                try self.search(t, template.items, base, lm, direct);
                break :blk lm.x;
            } else blk: {
                const goal = deck.measures[t.results[0]];
                const c = if (goal.first.goal != null) goal.first else goal.second;
                const scale = @max(@abs(c.goal.?), c.minval) / c.weight;
                const b = &self.bisection[base];
                b.* = core.bisect.Bisect.init(a, t.method, t.spec.lo[0], t.spec.hi[0], t.spec.grid[0], scale, t.options);
                try self.search(t, template.items, base, b, false);
                break :blk (&b.x)[0..1];
            };
            row.* = try t.rows(a, &self.prepared.circuit, x, base);
        }
        deck.variants = try frontend.Tuner.optimized(a, deck.variants, best);
    }

    /// Whether `job` can give the optimizer its Jacobian as tangents of its
    /// own run. A transient cannot from an operating point it does not
    /// solve on its own (`uic`, `.ic` holds, a capacitor-only node's
    /// OPtran), under Gear-3..6, under VPI callbacks, or off the CPU: its rows swap host
    /// device state (`Tran.tangents`).
    fn tangentable(self: *const Problem, job: Query) bool {
        const c = &self.prepared.circuit;
        return switch (job) {
            .dc => true,
            // Tangents carry Gear-2's charge history only (validate.zig).
            .tran => |o| !o.uic and !o.snapshot and !o.sources_at_zero and self.prepared.deck.ic.len == 0 and @backingInt(o.method) <= @backingInt(requests.Method.gear_2) and
                !c.needs_tran_op and c.vpi == null and self.session.config.backend == .cpu,
            else => false,
        };
    }

    /// HSPICE MOSRA, once before any query advances
    /// (dev/analysis/mosra.md): runs the stress transient with the bound
    /// MOSFETs' terminals as its only outputs, fills the aged rows'
    /// `delvto`/`mulu0` writes and publishes the degradation table.
    /// ponytail: the stress transient repeats the fresh one; fold them
    /// into one run when MOSRA decks get large.
    fn age(self: *Problem) !void {
        const m = self.prepared.deck.mosra orelse return;
        if (self.degradation != null) return;
        const a = self.arena.allocator();
        var deck = self.prepared.deck;
        const probes = try a.alloc(u32, 3 * m.terminals.len);
        for (m.terminals, 0..) |t, i| probes[3 * i ..][0..3].* = t;
        const labels = try a.alloc([]const u8, probes.len);
        @memset(labels, "");
        deck.probes = probes;
        deck.probe_labels = labels;
        deck.opvars = &.{};
        var session = analysis.session.Session.init(self.workerAllocator(), self.io, &self.prepared.circuit, &deck, self.session.config);
        defer session.deinit();
        var id: [1]QueryId = undefined;
        _ = try session.append(&.{.{ .tran = m.tran }}, &id);
        try session.runAll(.{ .max_parallel = self.limits.max_parallel, .quantum = .completion });
        if (session.failure()) |err| return err;
        const aged = try analysis.mosra.age(a, m, try session.result(id[0]));
        if (m.aged_runs) self.prepared.deck.variants.values = aged.values;
        self.degradation = aged.table;
        try self.delivery.publish(self.io, .{ .title = deck.title, .result = aged.table });
    }

    /// Feeds `solver` (`core.lm.Lm` or `core.bisect.Bisect`) the residuals
    /// of the points it asks for until it stops.
    /// A Levenberg-Marquardt point's operating point starts from the one of
    /// the iterate its step left (dev/analysis/optimize.md, "Warm starts"),
    /// so the path does not depend on batching; the start's round and every
    /// bisection point start cold. `direct` as `evaluatePoints` says.
    fn search(self: *Problem, t: *frontend.Tuner, template: []const Query, base: usize, solver: anytype, direct: bool) !void {
        const n = t.spec.live.len;
        const m = t.results.len;
        const lm = @TypeOf(solver) == *core.lm.Lm;
        const gpa = self.allocator;
        var residuals: std.ArrayList(f64) = .empty;
        defer residuals.deinit(gpa);
        // The operating points of the accepted iterate, of row 0 of the
        // last round, and the one this round starts from.
        var origin: std.ArrayList(f64) = .empty;
        defer origin.deinit(gpa);
        var first: std.ArrayList(f64) = .empty;
        defer first.deinit(gpa);
        var guess: std.ArrayList(f64) = .empty;
        defer guess.deinit(gpa);
        while (solver.points().len != 0) {
            const points = solver.points();
            if (lm) if (solver.stepping()) {
                guess.clearRetainingCapacity();
                try guess.appendSlice(gpa, origin.items);
            };
            try residuals.resize(gpa, points.len / n * m);
            try self.evaluatePoints(t, template, base, points, if (lm) direct and solver.batched() else false, guess.items, residuals.items, &first);
            const accepted = if (lm) solver.history.rss.items.len else 0;
            try solver.feed(residuals.items);
            if (lm) if (solver.history.rss.items.len > accepted) std.mem.swap(std.ArrayList(f64), &origin, &first);
        }
    }

    /// Residuals of each point of `points` (the optimized parameters'
    /// values, n per point, on top of `t.bases[base]`) into `out`, m per
    /// point: the RESULTS cards'
    /// goal errors over the `template` queries run at that point, NaN where
    /// a query failed or a card found no value. Every operating point
    /// starts from `guess` (cold when empty); `first` takes the one row 0's
    /// first query solved, empty when it has none. With `direct` (a
    /// batched round of `tangentable` queries) only row 0 runs, and the
    /// other points' results are its runs' first-order predictions of them
    /// (`Dc.tangents`, `Tran.tangents`, dev/analysis/optimize.md "Direct
    /// sensitivities").
    fn evaluatePoints(self: *Problem, t: *frontend.Tuner, template: []const Query, base: usize, points: []const f64, direct: bool, guess: []const f64, out: []f64, first: *std.ArrayList(f64)) !void {
        var arena = std.heap.ArenaAllocator.init(self.allocator);
        defer arena.deinit();
        const a = arena.allocator();
        const count = points.len / t.spec.live.len;
        std.debug.assert(!direct or count == t.spec.live.len + 1);
        var deck = self.prepared.deck;
        deck.variants = try t.rows(a, &self.prepared.circuit, points, base);
        deck.op_guess = guess;
        const runs = if (direct) 1 else count;
        const tangents = try a.alloc(u32, if (direct) count - 1 else 0);
        for (tangents, 1..) |*row, p| row.* = @intCast(p);
        const jobs = try a.alloc(Query, runs * template.len);
        for (0..runs) |p| for (template, jobs[p * template.len ..][0..template.len]) |job, *copy| {
            copy.* = job;
            switch (copy.*) {
                inline else => |*o| o.tol.variant = @intCast(p),
            }
            if (direct) switch (copy.*) {
                .dc => |*o| o.tangents = tangents,
                .tran => |*o| o.tangents = tangents,
                else => unreachable, // else: `tangentable` admits these two
            };
        };
        var session = analysis.session.Session.init(self.workerAllocator(), self.io, &self.prepared.circuit, &deck, self.session.config);
        defer session.deinit();
        const ids = try a.alloc(QueryId, jobs.len);
        _ = try session.append(jobs, ids);
        try session.runAll(.{ .max_parallel = self.limits.max_parallel, .quantum = .completion });
        first.clearRetainingCapacity();
        if (try session.operatingPoint(ids[0])) |x| try first.appendSlice(self.allocator, x);
        const values = try a.alloc(f64, deck.measures.len);
        const m = t.results.len;
        @memset(out, std.math.nan(f64));
        for (0..count) |p| for (ids[(if (direct) 0 else p) * template.len ..][0..template.len]) |id| {
            const info = try session.info(id);
            if (info.status != .complete) continue;
            var res = try session.result(id);
            if (direct) {
                // Block p of the sweep: its own points, or row p's prediction.
                // A transient keeps its tangent rows apart (`Result.tangents`,
                // as `block` reads them); a `.dc` sweep holds them inline.
                if (res.tangents.len != 0) {
                    if (p != 0) {
                        const n = res.npoints * res.varnames.len;
                        res.data = res.tangents[(p - 1) * n ..][0..n];
                        res.measured = &.{};
                    }
                    res.tangents = &.{};
                } else {
                    res.npoints /= count;
                    res.data = res.data[p * res.npoints * res.varnames.len ..][0 .. res.npoints * res.varnames.len];
                }
            }
            output.measureValues(deck.measures, info.kind, res, values);
            for (t.results, out[p * m ..][0..m]) |card, *r| {
                const measure = deck.measures[card];
                // Never null: Tuner.init refuses a RESULTS card without GOAL=
                // (`goalError(0) != null`), and only the goal decides null.
                if (measure.analysis == info.kind) r.* = measure.goalError(values[card]).?;
            }
        };
    }

    /// Publishes what has completed, the main session's results first, then
    /// each variant run's once every earlier one has published all of its.
    fn deliverRuns(self: *Problem) void {
        self.deliverFrom(&self.session, &self.next_output);
        if (self.next_output != self.session.outputs.items.len) return;
        for (self.runs) |*run| {
            self.deliverFrom(&run.session, &run.next_output);
            if (run.next_output != run.session.outputs.items.len) return;
        }
    }

    /// Publishes `session`'s completed results in request order from
    /// `next_output.*`, stopping at the first unfinished query or the first
    /// output failure.
    fn deliverFrom(self: *Problem, session: *const analysis.session.Session, next_output: *usize) void {
        if (self.delivery_error != null) return;
        while (next_output.* < session.outputs.items.len) : (next_output.* += 1) {
            const delivered = self.deliverOne(session, session.outputs.items[next_output.*]) catch |err| {
                self.delivery_error = err;
                return;
            };
            if (!delivered) return;
        }
    }

    /// Publishes (or ends the stream of) `id` once it is terminal; false
    /// while it still runs. `session.outputs` holds only appended ids, and a
    /// row reads `.complete` only after its executor published, so the
    /// session lookups fail only if Session breaks that bookkeeping; the
    /// error then stops delivery like any output failure instead of
    /// aborting a C host.
    fn deliverOne(self: *Problem, session: *const analysis.session.Session, id: QueryId) DeliveryError!bool {
        const status = (try session.info(id)).status;
        if (!status.terminal()) return false;
        if (session.released(id)) return true;
        if (session == &self.session and self.streaming == id) {
            self.streaming = null;
            if (status != .complete) {
                self.delivery.abortStream();
                return true;
            }
            const res = try session.result(id);
            try self.delivery.endStream(self.io, res.npoints, res.varnames.len);
        } else if (session == &self.session and self.sides.contains(id)) {
            const side = self.sides.fetchRemove(id).?.value;
            if (status != .complete) {
                self.delivery.dropSide(side);
                return true;
            }
            const res = try session.result(id);
            var lap = if (self.session.config.timing_in_depth) std.Io.Timestamp.now(self.io, .awake) else null;
            defer timingLap(self.io, &lap, "output delivery");
            try self.delivery.endSide(self.io, side, .{ .title = session.deck.title, .result = res });
        } else if (status == .complete) {
            const res = try session.result(id);
            var lap = if (self.session.config.timing_in_depth) std.Io.Timestamp.now(self.io, .awake) else null;
            defer timingLap(self.io, &lap, "output delivery");
            const job = try session.query(id);
            // An `.sp` carries its ports' z0, which a Touchstone output records (#60).
            var z0: std.ArrayList(f64) = .empty;
            defer z0.deinit(self.allocator);
            if (job == .sp) try modeZ0(self.allocator, job.sp.ports, &z0);
            const plot: output.Plot = .{ .title = session.deck.title, .result = res, .z0 = z0.items };
            try self.delivery.publish(self.io, plot);
            if (job == .sp) if (job.sp.lin) |lin| if (lin.touchstone)
                try self.writeTouchstone(lin.file, plot);
        }
        return true;
    }

    /// Writes a `.lin format=touchstone` result as `<file>.s<N>p` beside the
    /// deck, `file` defaulting to the deck's name without its extension
    /// (HSPICE names it after the netlist), against the ports' own z0.
    fn writeTouchstone(self: *Problem, file: []const u8, plot: output.Plot) !void {
        const stem = if (file.len > 0) file else std.fs.path.stem(self.origin);
        const name = try std.fmt.allocPrint(self.allocator, "{s}.s{d}p", .{ stem, @max(plot.z0.len, 1) });
        defer self.allocator.free(name);
        const path = try std.fs.path.join(self.allocator, &.{ std.fs.path.dirname(self.origin) orelse ".", name });
        defer self.allocator.free(path);
        try output.write(self.io, path, .touchstone, plot);
    }

    /// Appends the reference impedance of each of `ports`' modes to `z0`, in
    /// sp.zig's `Basis` order: each port's first mode (a balanced one's
    /// differential, 2·z0), then the common modes (z0/2).
    fn modeZ0(gpa: std.mem.Allocator, ports: []const requests.Port, z0: *std.ArrayList(f64)) !void {
        for (ports) |p| try z0.append(gpa, if (p.balanced != null) 2 * p.z0 else p.z0);
        for (ports) |p| if (p.balanced != null) try z0.append(gpa, p.z0 / 2);
    }

    /// `allocator` behind a mutex, so parallel query arenas can share it
    /// without the caller having to supply a thread-safe allocator.
    fn workerAllocator(self: *Problem) std.mem.Allocator {
        return .{ .ptr = self, .vtable = &.{ .alloc = alloc, .resize = resize, .remap = remap, .free = free } };
    }

    /// The query workers' allocations (sessions, executors, circuits,
    /// results): bytes held now and the peak so far, in bytes requested.
    /// The preparation (`Prepared`) is not counted.
    pub fn queryMemory(self: *Problem) struct { live: usize, peak: usize } {
        self.allocation_mutex.lockUncancelable(self.io);
        defer self.allocation_mutex.unlock(self.io);
        return .{ .live = self.query_live, .peak = self.query_peak };
    }

    /// Moves the counted bytes from `old` to `new`; under the mutex.
    fn account(self: *Problem, old: usize, new: usize) void {
        self.query_live = self.query_live - old + new;
        self.query_peak = @max(self.query_peak, self.query_live);
    }

    fn alloc(ctx: *anyopaque, n: usize, alignment: std.mem.Alignment, ret: usize) ?[*]u8 {
        const self: *Problem = @ptrCast(@alignCast(ctx));
        self.allocation_mutex.lockUncancelable(self.io);
        defer self.allocation_mutex.unlock(self.io);
        const p = self.allocator.rawAlloc(n, alignment, ret) orelse return null;
        self.account(0, n);
        return p;
    }

    fn resize(ctx: *anyopaque, memory: []u8, alignment: std.mem.Alignment, n: usize, ret: usize) bool {
        const self: *Problem = @ptrCast(@alignCast(ctx));
        self.allocation_mutex.lockUncancelable(self.io);
        defer self.allocation_mutex.unlock(self.io);
        if (!self.allocator.rawResize(memory, alignment, n, ret)) return false;
        self.account(memory.len, n);
        return true;
    }

    fn remap(ctx: *anyopaque, memory: []u8, alignment: std.mem.Alignment, n: usize, ret: usize) ?[*]u8 {
        const self: *Problem = @ptrCast(@alignCast(ctx));
        self.allocation_mutex.lockUncancelable(self.io);
        defer self.allocation_mutex.unlock(self.io);
        const p = self.allocator.rawRemap(memory, alignment, n, ret) orelse return null;
        self.account(memory.len, n);
        return p;
    }

    fn free(ctx: *anyopaque, memory: []u8, alignment: std.mem.Alignment, ret: usize) void {
        const self: *Problem = @ptrCast(@alignCast(ctx));
        self.allocation_mutex.lockUncancelable(self.io);
        defer self.allocation_mutex.unlock(self.io);
        self.allocator.rawFree(memory, alignment, ret);
        self.account(memory.len, 0);
    }
};

/// `Problem.print_soa`'s running count: the excursions, the instances they
/// happened on, and the (limit, result) pairs a missing column skipped.
/// Each excursion prints to `out` while `max_warns` allows, and with `json`
/// is also a JSON object there, led by `,\n` (`write_measures_json`).
const SoaTally = struct {
    gpa: std.mem.Allocator,
    out: ?*std.Io.Writer = null,
    json: ?*std.Io.Writer = null,
    max_warns: u32 = 0,
    excursions: u64 = 0,
    instances: std.StringHashMapUnmanaged(void) = .empty,
    skipped: u64 = 0,

    /// Every SOA limit of `s`'s deck over each of its completed operating
    /// points, DC sweeps and transients, in request order.
    fn session(tally: *SoaTally, s: *const analysis.session.Session) !void {
        for (s.outputs.items) |id| {
            const info = try s.info(id);
            if (!soaChecked(info.kind) or info.status != .complete or s.released(id)) continue;
            try tally.result(s, try s.query(id), info.kind, try s.result(id));
        }
    }

    /// Every SOA limit of `s`'s deck over `res`, the result of `job`.
    fn result(tally: *SoaTally, s: *const analysis.session.Session, job: Query, kind: requests.Kind, res: Result) !void {
        for (s.deck.soa.limits) |lim| try tally.scan(lim, res, s, job, kind);
    }

    /// `lim` over every row of `res`; `kind` says what column 0 is: a
    /// transient's time, a DC sweep's value, nothing for an operating point.
    fn scan(tally: *SoaTally, lim: core.SoaLimit, res: Result, s: *const analysis.session.Session, job: Query, kind: requests.Kind) !void {
        const w = res.varnames.len;
        if (res.data.len != res.npoints * w) {
            tally.skipped += 1;
            return;
        }
        var cols: [2]?usize = .{ null, null };
        for (&cols, [2][]const u8{ lim.pos, lim.neg }) |*col, name| if (name.len != 0) {
            col.* = for (res.varnames, 0..) |v, k| {
                if (std.mem.eql(u8, v, name)) break k;
            } else {
                tally.skipped += 1;
                return;
            };
        };
        var from: ?f64 = null;
        var peak: f64 = 0;
        for (0..res.npoints) |i| {
            const row = res.data[i * w ..][0..w];
            const d = (if (cols[0]) |c| row[c] else 0) - (if (cols[1]) |c| row[c] else 0);
            const v = if (lim.abs) @abs(d) else d;
            const at = if (kind == .op) 0 else row[0];
            if (v > lim.limit) {
                peak = if (from == null) v else @max(peak, v);
                if (from == null) from = at;
            } else if (from) |f| {
                try tally.excursion(lim, peak, f, at, s, job, kind);
                from = null;
            }
        }
        if (from) |f| try tally.excursion(lim, peak, f, null, s, job, kind);
    }

    /// Counts one excursion, `from` to `to` (null: past the last row) at
    /// `peak`, prints it while `max_warns` allows and writes it to `json`.
    fn excursion(tally: *SoaTally, lim: core.SoaLimit, peak: f64, from: f64, to: ?f64, s: *const analysis.session.Session, job: Query, kind: requests.Kind) !void {
        tally.excursions += 1;
        try tally.instances.put(tally.gpa, lim.name, {});
        const row = variantOf(job);
        if (tally.json) |j| {
            const nan = std.math.nan(f64);
            const windowed = kind != .op;
            try j.writeAll(",\n    {\"instance\": ");
            try std.json.Stringify.encodeJsonString(lim.name, .{}, j);
            try j.print(", \"quantity\": \"{s}\", \"key\": \"{s}\", \"limit\": ", .{ lim.quantity, lim.key });
            try output.writeJsonNumber(j, lim.limit);
            try j.writeAll(", \"peak\": ");
            try output.writeJsonNumber(j, peak);
            try j.writeAll(", \"from\": ");
            try output.writeJsonNumber(j, if (windowed) from else nan);
            try j.writeAll(", \"to\": ");
            try output.writeJsonNumber(j, if (windowed) (to orelse nan) else nan);
            try j.print(", \"analysis\": \"{t}\", \"variant\": ", .{kind});
            try output.writeVariantJson(j, s.deck.variants, row, tempOf(job));
            try j.writeAll("}");
        }
        const out = tally.out orelse return;
        if (tally.excursions > tally.max_warns) return;
        const label = if (row) |v| s.deck.variants.labelAt(v) else "";
        try out.print("soa: {s} {s} = {d:.4} V > {s} {d} V", .{ lim.name, lim.quantity, peak, lim.key, lim.limit });
        if (kind != .op) {
            try out.print(" from {e:.3}", .{from});
            if (to) |t| try out.print(" to {e:.3}", .{t}) else try out.writeAll(" to the end");
            if (kind == .tran) try out.writeAll(" s");
        }
        try out.print(" ({t}{s}{s})\n", .{ kind, if (label.len != 0) " " else "", label });
    }
};

/// Whether `.option soa` reads a `kind` result: only these publish the
/// accepted points.
fn soaChecked(kind: requests.Kind) bool {
    return kind == .op or kind == .dc or kind == .tran;
}

/// The index of the `.meas` card named `name` (case-insensitive), the first
/// when several share it.
fn measureIndex(measures: []const core.Measure, name: []const u8) ?u32 {
    for (measures, 0..) |m, k| if (std.ascii.eqlIgnoreCase(m.name, name)) return @as(u32, @intCast(k));
    return null;
}

/// The variant row `job` runs at, or null for the nominal circuit.
fn variantOf(job: Query) ?u32 {
    return switch (job) {
        inline else => |o| o.tol.variant,
    };
}

/// Whether `id` is a completed transient with terminal currents and the
/// first such of its variant after those in `before`.
// ponytail: quadratic in the outputs; a key set once sweeps reach thousands of points.
fn emirFirst(session: *const analysis.session.Session, before: []const QueryId, id: QueryId) !bool {
    if (!try hasEmir(session, id)) return false;
    const job = try session.query(id);
    for (before) |b| if (try hasEmir(session, b)) {
        const other = try session.query(b);
        if (std.meta.eql(variantOf(other), variantOf(job)) and std.meta.eql(tempOf(other), tempOf(job))) return false;
    };
    return true;
}

/// Whether `id` is a completed, kept transient that read terminal currents.
fn hasEmir(session: *const analysis.session.Session, id: QueryId) !bool {
    const info = try session.info(id);
    if (info.kind != .tran or info.status != .complete or session.released(id)) return false;
    return (try session.result(id)).emir.terminals.len != 0;
}

/// `Problem.write_emir_json`'s document for transient `id` of `session`,
/// which runs `prepared`.
fn writeEmir(out: *std.Io.Writer, deck: []const u8, lib: *const frontend.Library, prepared: *const frontend.Prepared, session: *const analysis.session.Session, id: QueryId) !void {
    const e = (try session.result(id)).emir;
    const job = try session.query(id);
    const d = &prepared.deck;
    const ckt = &prepared.circuit;
    try out.writeAll("{\n  \"format\": \"espice-emir/1\",\n  \"deck\": ");
    try std.json.Stringify.encodeJsonString(deck, .{}, out);
    try out.writeAll(",\n  \"analysis\": \"tran\",\n  \"variant\": ");
    try output.writeVariantJson(out, d.variants, variantOf(job), tempOf(job));
    try out.writeAll(",\n  \"temp\": ");
    if (tempOf(job) orelse d.deck_temp) |temp| try output.writeJsonNumber(out, temp) else try out.writeAll("null");
    try out.writeAll(",\n  \"window\": [");
    try output.writeJsonNumber(out, e.from);
    try out.writeAll(", ");
    try output.writeJsonNumber(out, e.to);
    try out.writeAll("],\n  \"em_recovery\": ");
    try output.writeJsonNumber(out, d.em_recovery);
    // `Wave.emAvg`'s value from the same halves and span.
    const span = e.to - e.from;
    inline for (.{ false, true }) |sources| {
        try out.writeAll(if (sources) "\n  ],\n  \"sources\": [" else ",\n  \"terminals\": [");
        var first = true;
        for (e.terminals) |term| {
            const b = ckt.batches[term.batch];
            const source = std.mem.eql(u8, b.type_name, "vsource") or std.mem.eql(u8, b.type_name, "isource");
            if (source != sources or (sources and term.port != 0)) continue;
            try out.writeAll(if (first) "\n    {\"instance\": " else ",\n    {\"instance\": ");
            first = false;
            const t = ckt.batch_types[term.batch];
            try std.json.Stringify.encodeJsonString(d.card_names.get(t, term.instance) orelse b.type_name, .{}, out);
            if (!sources) {
                try out.writeAll(", \"device\": ");
                try std.json.Stringify.encodeJsonString(b.type_name, .{}, out);
                try out.writeAll(", \"terminal\": ");
                const ports = lib.vtable(t).unknown_names;
                if (term.port < ports.len) try std.json.Stringify.encodeJsonString(ports[term.port], .{}, out) else try out.print("\"{d}\"", .{term.port});
            }
            try out.writeAll(", \"net\": ");
            try std.json.Stringify.encodeJsonString(if (term.row == core.GROUND or term.row >= ckt.n) "0" else ckt.nodeName(term.row), .{}, out);
            const em_avg = if (span > 0) (@max(term.pos, term.neg) - d.em_recovery * @min(term.pos, term.neg)) / span else std.math.nan(f64);
            for ([_][]const u8{ "avg", "em_avg", "rms", "peak" }, [_]f64{ term.avg, em_avg, term.rms, term.peak }) |key, v| {
                try out.print(", \"{s}\": ", .{key});
                try output.writeJsonNumber(out, v);
            }
            try out.writeAll("}");
        }
    }
    try out.writeAll("\n  ]\n}\n");
}

/// The error query `id` ended with, its prerequisite's for a query that
/// never ran because of it; null when it completed. Asserts it is terminal.
fn queryFailure(session: *const analysis.session.Session, id: QueryId) !?anyerror {
    const info = try session.info(id);
    std.debug.assert(info.status.terminal());
    if (info.status == .complete) return null;
    // `?anyerror` payloads: a bare error value here would return as this
    // function's own error.
    if (info.failure) |err| if (err == error.DependencyFailed) if (info.dependency) |dep| {
        if ((try session.info(dep)).failure) |cause| return @as(?anyerror, cause);
    };
    return @as(?anyerror, info.failure orelse error.QueryCancelled);
}

/// Appends `session`'s completed Monte Carlo trial results of `kind`.
fn monteTrials(gpa: std.mem.Allocator, session: *const analysis.session.Session, kind: requests.Kind, trials: *std.ArrayList(Result)) !void {
    for (session.outputs.items) |id| {
        const info = try session.info(id);
        // `releaseDelivered` keeps every trial.
        if (info.status != .complete or info.kind != kind or session.released(id)) continue;
        const res = try session.result(id);
        if (std.mem.indexOf(u8, res.plotname, "(monte=") != null) try trials.append(gpa, res);
    }
}

/// Counts the failures of SIGMA_SCALE card `s` over its completed `trials`
/// (#239), fits them and writes the report; returns the fit's refusal. A
/// trial fails when FAIL holds or reads a card without a value. Counted by
/// subtraction, so a trial with no completed result (no convergence,
/// cancelled) fails too. ponytail: every trial's result stays until the
/// end, as plain MONTE's do; reduce each to its `.meas` row on delivery if
/// large runs need it.
fn writeSss(gpa: std.mem.Allocator, out: *std.Io.Writer, json: bool, measures: []const core.Measure, s: frontend.Sss, trials: []const Result) !?core.sss.Error {
    const m = s.monte;
    const passed = try gpa.alloc(bool, m.n * m.scales.len);
    defer gpa.free(passed);
    @memset(passed, false);
    const values = try gpa.alloc(f64, measures.len);
    defer gpa.free(values);
    for (trials) |res| {
        const at = std.mem.indexOf(u8, res.plotname, "(monte=").? + "(monte=".len;
        const tail = res.plotname[at..];
        const trial = std.fmt.parseInt(u64, tail[0 .. std.mem.indexOfNone(u8, tail, "0123456789") orelse tail.len], 10) catch continue;
        if (s.scaleOf(trial) == null) continue;
        output.measureValues(measures, s.kind, res, values);
        passed[@intCast(trial - m.first)] = !try s.fails(gpa, values);
    }
    const failures = try gpa.alloc(u64, m.scales.len);
    defer gpa.free(failures);
    const each = try gpa.alloc(u64, m.scales.len);
    defer gpa.free(each);
    @memset(each, m.n);
    for (failures, 0..) |*k, j| k.* = m.n - std.mem.count(bool, passed[j * m.n ..][0..m.n], &.{true});
    const estimate = core.sss.fit(m.scales, each, failures);
    const report: output.Sss = .{ .fail = m.fail.text, .scales = m.scales, .trials = m.n, .failures = failures, .estimate = estimate };
    if (json) try output.writeSss(out, s.kind, report) else try output.printSss(out, report);
    // An error value returned as is would be this function's own error.
    _ = estimate catch |e| return @as(?core.sss.Error, e);
    return null;
}

/// Block `k` of query `id`'s result: its own run at 0, else the
/// prediction at its `k`-th tangent row (`Result.tangents`).
fn block(session: *const analysis.session.Session, id: QueryId, k: usize) !Result {
    var res = try session.result(id);
    if (k == 0) return res;
    const n = res.npoints * res.varnames.len;
    std.debug.assert(k * n <= res.tangents.len);
    res.data = res.tangents[(k - 1) * n ..][0..n];
    res.tangents = &.{};
    res.measured = &.{};
    return res;
}

/// `.meas` results over query `id`'s result when it completed, of a kind a
/// card can read; a transient's own run, not its tangent blocks. A sweep
/// point's block names its row and label, and prints NaN for each card
/// without a value, a failed point's for all.
fn printQueryMeasures(gpa: std.mem.Allocator, session: *const analysis.session.Session, id: QueryId, out: *std.Io.Writer, err: *std.Io.Writer, json: ?*std.Io.Writer, measures: []const core.Measure) !void {
    const info = try session.info(id);
    switch (info.kind) {
        .op, .tran, .ac, .dc, .fft, .tran_noise, .dcmatch, .acmatch, .lstb, .phasenoise, .pnoise, .sp => {},
        else => return,
    }
    if (session.released(id)) return;
    const job = try session.query(id);
    const row = variantOf(job);
    const point: ?output.MeasurePoint = if (row) |v| .{ .row = v, .label = session.deck.variants.labelAt(v) } else null;
    // The fields every JSON result of the block shares (#235).
    var head: std.Io.Writer.Allocating = .init(gpa);
    defer head.deinit();
    if (json != null) {
        try head.writer.print("\"analysis\": \"{t}\", \"variant\": ", .{info.kind});
        try output.writeVariantJson(&head.writer, session.deck.variants, row, tempOf(job));
    }
    const sink: ?output.MeasureJson = if (json) |j| .{ .out = j, .head = head.written() } else null;
    if (info.status == .complete) {
        var res = try block(session, id, 0);
        // `soa_count()` (#238): this result's excursions, when none was skipped.
        if (session.deck.soa.limits.len != 0 and soaChecked(info.kind)) {
            var tally: SoaTally = .{ .gpa = gpa };
            defer tally.instances.deinit(gpa);
            try tally.result(session, job, info.kind, res);
            if (tally.skipped == 0) res.soa_count = @floatFromInt(tally.excursions);
        }
        try output.printMeasures(out, err, measures, info.kind, res, point, sink);
    } else if (point) |p| if (info.status.terminal()) try output.printFailedMeasures(out, measures, info.kind, p, sink);
}

/// The HSPICE `.temp` list entry `job` runs at; null at the deck's.
fn tempOf(job: Query) ?f64 {
    return switch (job) {
        inline else => |o| o.tol.temp_c,
    };
}

/// The optimizer's summary, after HSPICE's: one line per accepted
/// iteration, the stop reason, the final figures, the optimized parameters
/// and each RESULTS card's error at the optimum.
fn printOptimization(out: *std.Io.Writer, lm: *const core.lm.Lm, t: *const frontend.Tuner, base: usize, measures: []const core.Measure) !void {
    const o = t.options;
    try out.print("\n  Optimization {s} (Levenberg-Marquardt, model {s}){s}{s}\n\n  iter  evals  residual sum of squares  marquardt param", .{ t.spec.name, t.spec.model, if (t.bases[base].label.len != 0) " at " else "", t.bases[base].label });
    for (t.names) |name| try out.print("  {s:>14}", .{name});
    try out.writeAll("\n");
    const h = lm.history;
    for (h.rss.items, h.lambda.items, h.evaluations.items, 0..) |rss, lambda, evals, k| {
        try out.print("  {d:>4}  {d:>5}  {e:>23.6}  {e:>15.6}", .{ k, evals, rss, lambda });
        for (h.x.items[k * lm.n ..][0..lm.n]) |x| try out.print("  {e:>14.6}", .{x});
        try out.writeAll("\n");
    }
    try out.writeAll("\n");
    var pinned = false;
    for (0..lm.n) |j| pinned = pinned or lm.pinned(j) != 0;
    if (lm.status != .failed and pinned) {
        try out.writeAll("  optimization stopped at a limit: the goals are not reachable inside the parameter ranges\n");
    } else switch (lm.status) {
        .relin => try out.print("  optimization completed: RELIN = {e} on last iteration\n", .{o.relin}),
        .relout => try out.print("  optimization completed: RELOUT = {e} on last iteration\n", .{o.relout}),
        .grad => try out.print("  optimization completed: norm of the gradient < GRAD = {e}\n", .{o.grad}),
        .itropt => try out.print("  optimization incomplete: ITROPT = {d} iterations reached\n", .{o.itropt}),
        .max => try out.print("  optimization stopped: marquardt parameter above MAX = {e}\n", .{o.max}),
        .failed, .running => try out.writeAll("  optimization failed: a point could not be simulated or measured\n"),
    }
    for (t.names, 0..) |name, j| switch (lm.pinned(j)) {
        -1 => try out.print("  {s} is held at its lower limit {e:.6}\n", .{ name, lm.lo[j] }),
        1 => try out.print("  {s} is held at its upper limit {e:.6}\n", .{ name, lm.hi[j] }),
        else => {},
    };
    try out.print(
        \\  residual sum of squares       = {e:.6}
        \\  norm of the gradient          = {e:.6}
        \\  marquardt scaling parameter   = {e:.6}
        \\  no. of function evaluations   = {d}
        \\  no. of iterations             = {d}
        \\
        \\  optimized parameters {s} -- final values
        \\
    , .{ lm.rss, lm.gradientNorm(), lm.lambda, lm.evaluations, lm.iterations, t.spec.name });
    for (t.names, lm.x, t.initial, lm.lo, lm.hi, lm.grid) |name, x, x0, lo, hi, g| {
        try out.print("  {s} = {e:.6} $ initial {e:.6}, range {e:.6} to {e:.6}", .{ name, x, x0, lo, hi });
        if (g > 0) try out.print(", grid {e:.6}", .{g});
        try out.writeAll("\n");
    }
    try out.writeAll("\n  goal errors at the optimum (weight * (result - goal) / max(|goal|, minval))\n");
    for (t.results, lm.r) |card, r| try out.print("  error({s}) = {e:.6}\n", .{ measures[card].name, r });
}

/// The bisection summary: one line per midpoint test with the window it
/// halved, the stop reason, and the last passing value [SA Ch.19].
fn printBisection(out: *std.Io.Writer, b: *const core.bisect.Bisect, t: *const frontend.Tuner, base: usize, measures: []const core.Measure) !void {
    const o = t.options;
    try out.print("\n  Optimization {s} ({s}, model {s}){s}{s}\n\n  iter             xlo             xhi               x      goal error\n", .{ t.spec.name, @tagName(b.method), t.spec.model, if (t.bases[base].label.len != 0) " at " else "", t.bases[base].label });
    const h = b.history;
    for (h.lo.items, h.hi.items, h.x.items, h.r.items, 1..) |lo, hi, x, r, k|
        try out.print("  {d:>4}  {e:>14.6}  {e:>14.6}  {e:>14.6}  {e:>14.6}\n", .{ k, lo, hi, x, r });
    try out.writeAll("\n");
    switch (b.status) {
        .converged => if (o.absin > 0)
            try out.print("  optimization completed: ABSIN = {e} satisfied\n", .{o.absin})
        else if (b.method == .passfail)
            try out.print("  optimization completed: RELIN = {e} satisfied\n", .{o.relin})
        else
            try out.print("  optimization completed: RELIN = {e} and {s} = {e} satisfied\n", .{ o.relin, if (o.absout > 0) "ABSOUT" else "RELOUT", if (o.absout > 0) o.absout else o.relout }),
        .itropt => try out.print("  optimization incomplete: ITROPT = {d} iterations reached\n", .{o.itropt}),
        .bounds => try out.print("  optimization failed: both limits of {s} {s}, so they do not bracket the goal\n", .{ t.names[0], if (b.pass_hi) "pass" else "fail" }),
        .failed, .running => try out.writeAll("  optimization failed: a point could not be simulated or measured\n"),
    }
    try out.print(
        \\  no. of function evaluations   = {d}
        \\  no. of iterations             = {d}
        \\
        \\  optimized parameters {s} -- final values
        \\  {s} = {e:.6} $ last passing value, range {e:.6} to {e:.6}
        \\
    , .{ b.evaluations, b.iterations, t.spec.name, t.names[0], b.x, t.spec.lo[0], t.spec.hi[0] });
    if (b.grid > 0) try out.print("  grid {e:.6}\n", .{b.grid});
    try out.print("  error({s}) = {e:.6}\n", .{ measures[t.results[0]].name, b.r });
}

/// Prints the time since `start.*` under `label` and restarts the lap.
/// A null `start` means timing is off.
fn timingLap(io: std.Io, start: *?std.Io.Timestamp, label: []const u8) void {
    const before = start.* orelse return;
    const now = std.Io.Timestamp.now(io, .awake);
    const elapsed = before.durationTo(now).nanoseconds;
    std.debug.print("timing: {s}: {d:.6}ms\n", .{ label, @as(f64, @floatFromInt(elapsed)) / 1e6 });
    start.* = std.Io.Timestamp.now(io, .awake);
}

/// libc's getenv; null on wasm, which links no libc and has no environment.
fn getenv(name: [*:0]const u8) ?[*:0]const u8 {
    if (comptime @import("builtin").cpu.arch.isWasm()) return null;
    return std.c.getenv(name);
}

/// Copies `items`, the list and each string, into `a`.
fn dupeAll(a: std.mem.Allocator, items: []const []const u8) ![]const []const u8 {
    const out = try a.alloc([]const u8, items.len);
    for (out, items) |*o, s| o.* = try a.dupe(u8, s);
    return out;
}

/// `given` with each null path the environment sets filled in, copied into
/// `a`: `zig` from `$ZIG`, and `cache_dir` from `$ESPICE_CACHE/hdl`, else
/// `$XDG_CACHE_HOME/espice/hdl`, else `$HOME/.cache/espice/hdl`, else
/// `$TMPDIR/espice-cache/hdl` (default `/tmp`). An empty variable counts as
/// unset for the cache, as `$ZIG` never is.
fn toolchainFromEnv(a: std.mem.Allocator, given: HdlToolchain) !HdlToolchain {
    const env = struct {
        fn get(name: [*:0]const u8) ?[]const u8 {
            const v = std.mem.span(getenv(name) orelse return null);
            return if (v.len == 0) null else v;
        }
    }.get;
    const cache_dir = if (given.cache_dir) |d| try a.dupe(u8, d) else if (env("ESPICE_CACHE")) |d|
        try std.fs.path.join(a, &.{ d, "hdl" })
    else if (env("XDG_CACHE_HOME")) |d|
        try std.fs.path.join(a, &.{ d, "espice", "hdl" })
    else if (env("HOME")) |d|
        try std.fs.path.join(a, &.{ d, ".cache", "espice", "hdl" })
    else
        try std.fs.path.join(a, &.{ env("TMPDIR") orelse "/tmp", "espice-cache", "hdl" });
    const zig = given.zig orelse if (getenv("ZIG")) |z| std.mem.span(z) else null;
    return .{
        .zig = if (zig) |z| try a.dupe(u8, z) else null,
        .share = if (given.share) |d| try a.dupe(u8, d) else null,
        .cache_dir = cache_dir,
        .include_dirs = try dupeAll(a, given.include_dirs),
    };
}

/// Whether a run will probably take the GPU, judged before parsing: an
/// explicit `cuda`/`hip`, or a deck file of at least `likely_gpu_bytes`,
/// the size where `auto`'s cost model starts admitting post-layout decks.
// ponytail: the top file's size only; a small deck that includes a large
// netlist starts the driver with its first query, as before 05 5.12.
fn likelyGpu(options: Options) bool {
    if (options.backend.gpu_explicit or options.backend.backend != .auto) return true;
    const path = switch (options.source) {
        .file => |p| p,
        .bytes => |b| return b.data.len >= likely_gpu_bytes,
    };
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const z = std.mem.printSentinel(&buf, "{s}", .{path}, 0) catch return false;
    const fd = std.c.open(z, .{ .ACCMODE = .RDONLY });
    if (fd < 0) return false;
    defer _ = std.c.close(fd);
    return std.c.lseek(fd, 0, std.c.SEEK.END) >= likely_gpu_bytes;
}

/// 1 MiB: post-layout decks (chain_bsim4_10k is 3.3 MB; `auto` takes the
/// GPU for it, 6.6 → 6.2 s with the start overlapped). Below it `auto`
/// mostly stays on the CPU, and a started context costs a CPU run its
/// teardown (stress/scaling_parallel_inverters_2000, 199 KiB: 0.6 → 1.0 s).
const likely_gpu_bytes = 1 << 20;

/// Keeps the driver's JIT cache (`~/.nv/ComputeCache`) from evicting the
/// compact-model kernels, whose cold compiles take 15 s (bsim4va) to 1001 s
/// (hisimhv_va). The default cap is 256 MiB on older drivers and this
/// machine's cache already held 321 MB before they were added. 4 GiB is the
/// driver's maximum. A user's own setting wins. It must run before the
/// driver initializes, which the first GPU query of the process does, so
/// `Problem.init` calls it whenever the backend may use the GPU.
fn raiseJitCache() void {
    const setenv = struct {
        extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;
    }.setenv;
    _ = setenv("CUDA_CACHE_MAXSIZE", "4294967296", 0);
}

/// The developer switches (`analysis.Knobs`), read from the environment once
/// per Problem. Nothing below the facade reads the environment, so every
/// knob is parsed here; the field docs name each variable.
fn knobsFromEnv() analysis.Knobs {
    const env = struct {
        fn get(name: [:0]const u8) ?[]const u8 {
            return std.mem.span(getenv(name) orelse return null);
        }
        fn on(name: [:0]const u8) bool {
            return getenv(name) != null;
        }
    };
    var k: analysis.Knobs = .{};
    if (env.get("ESPICE_SOLVER")) |s| k.newton.pin = @TypeOf(k.newton.pin).names.get(s) orelse .auto;
    k.newton.opdbg = env.on("ZP_OPDBG");
    k.newton.newton_debug = env.on("ZP_NEWTON_DEBUG");
    k.lu.no_bbd = env.on("ESPICE_NO_BBD");
    if (env.get("ESPICE_LU_PAR")) |v| k.lu.par = v.len > 0 and v[0] == '1';
    k.lu.par_stats = env.on("ESPICE_LU_PAR_STATS");
    if (env.get("ESPICE_LANES")) |v| k.lanes = !std.mem.eql(u8, v, "0");
    k.hb_trace = env.on("ESPICE_HB_TRACE");
    k.no_qtape = env.on("ZP_NO_QTAPE");
    k.tran_stats = env.on("ZP_TRAN_STATS");
    k.tran_lte_trace = env.on("ZP_TRAN_LTE_TRACE");
    k.no_tangents = env.on("ZP_NO_TANGENTS");
    if (env.get("ESPICE_GPU_LU")) |v| k.gpu_lu = std.mem.eql(u8, v, "1");
    k.gpu_lu_check = env.on("ESPICE_GPU_LU_CHECK");
    k.gpu_lu_nogather = env.on("ESPICE_GPU_LU_NOGATHER");
    if (env.get("ESPICE_GPU_LU_TAIL")) |v| k.gpu_lu_tail = std.fmt.parseInt(u32, v, 10) catch 0;
    if (analysis.Knobs.instrument) {
        k.lu.stats = env.on("ZP_LU_STATS");
        if (env.get("ESPICE_GPU_LU_BENCH")) |v| k.gpu_lu_bench = std.fmt.parseInt(usize, v, 10) catch 0;
        if (env.get("ESPICE_GPU_LU_THREADS")) |v| k.gpu_lu_threads = std.fmt.parseInt(u32, v, 10) catch 8;
    }
    if (env.get("ESPICE_GPU_STATS")) |v| k.gpu_stats = if (std.mem.eql(u8, v, "phases")) .phases else .on;
    k.gpu_block = analysis.Knobs.parseBlock(env.get("ESPICE_GPU_BLOCK"));
    k.gpu_eval_check = env.on("ESPICE_GPU_EVAL_CHECK");
    k.gpu_nofuse = env.on("ESPICE_GPU_NOFUSE");
    k.gpu_nographs = env.on("ESPICE_GPU_NOGRAPHS");
    k.gpu_host_step = env.on("ESPICE_GPU_HOST_STEP");
    k.xdg_cache_home = env.get("XDG_CACHE_HOME");
    k.home = env.get("HOME");
    return k;
}
