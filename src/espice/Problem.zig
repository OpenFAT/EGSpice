//! A prepared circuit, its query DAG and the output its results go to:
//! `Options` in, results, published plots, `.meas` text and the deck's side
//! files out. `init` parses and prepares once and queues the deck's own
//! analyses; every later call appends to the DAG, advances it, or reads it.
//!
//! Not thread-safe: the caller serializes every call, reads and `deinit`
//! included. Parallelism lives inside `advance_ready` and `run_all`. Result
//! slices stay valid until `deinit`; `copy_result` copies for callers that
//! cannot hold them.
//! Cites: none.
//!
//! Contents:
//! - `pub const Summary = struct { plotname: []const u8, npoints: usize, nvars: usize }`
//! - `pub const Reason = points.Reason`
//! - `pub fn init(gpa: Allocator, io: std.Io, options: Options) !*Problem`
//! - `pub fn deinit(self: *Problem) void`
//! - `pub fn title(self: *const Problem) []const u8`
//! - `pub fn device_count(self: *const Problem) u32`
//! - `pub fn output_error(self: *const Problem) ?publish.DeliveryError`
//! - `pub fn finished(self: *const Problem) bool`
//! - `pub fn query_count(self: *const Problem) u32`
//! - `pub fn query_info(self: *const Problem, id: QueryId) !QueryInfo`
//! - `pub fn ready_count(self: *const Problem, scope: Scope) !usize`
//! - `pub fn ready_queries(self: *const Problem, scope: Scope, ids: []QueryId) !usize`
//! - `pub fn append_queries(self: *Problem, queries: []const Query, ids: []QueryId) !usize`
//! - `pub fn append_directives(self: *Problem, text: []const u8, ids: []QueryId) !usize`
//! - `pub fn checkOutput(self: *const Problem, prep: *const frontend.Prepared, queries: []const Query) !void`
//! - `pub const set_params = points.setParams`
//! - `pub const append_points = points.appendPoints`
//! - `pub const point_measures = points.pointMeasures`
//! - `pub const release_points = points.releasePoints`
//! - `pub const point_tangents = points.pointTangents`
//! - `pub fn lane_width(self: *const Problem) u32`
//! - `pub fn advance(self: *Problem, id: QueryId) !Advance`
//! - `pub fn advance_ready(self: *Problem, ids: []const QueryId, limits: Limits, events: []Advance) !usize`
//! - `pub fn run_all(self: *Problem) !void`
//! - `pub fn print(self: *const Problem, writer: *std.Io.Writer, options: PrintOptions) !void`
//! - `pub const print_measures = report.printMeasures`
//! - `pub const print_tangents = report.printTangents`
//! - `pub fn run_results(self: *const Problem, gpa: Allocator) ![]Result`
//! - `pub fn result(self: *const Problem, id: QueryId) !Result`
//! - `pub fn summary(self: *const Problem, id: QueryId) !Summary`
//! - `pub fn copy_result(self: *const Problem, id: QueryId, buffer: []f64) !usize`
//! - `pub fn queryMemory(self: *Problem) CountingAllocator.Usage`
//! - `pub fn lapStart(io: std.Io, on: bool) ?std.Io.Timestamp`
//! - `pub fn timingLap(io: std.Io, start: *?std.Io.Timestamp, label: []const u8) void`

const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;
const builtin = @import("builtin");
const analysis = @import("analysis");
const core = @import("core");
const frontend = @import("frontend");
const output = @import("output");
const config = @import("config.zig");
const CountingAllocator = @import("CountingAllocator.zig");
const deck_files = @import("deck_files.zig");
const optimizer = @import("optimizer.zig");
const points = @import("points.zig");
const publish = @import("publish.zig");
const report = @import("report.zig");
const Options = config.Options;
const requests = core.query;
const Query = requests.Query;
const QueryId = requests.QueryId;
const Session = analysis.session.Session;
const QueryInfo = analysis.session.QueryInfo;
const Advance = analysis.session.Advance;
const Scope = analysis.session.Scope;
const Limits = analysis.session.Limits;
const PrintOptions = analysis.session.PrintOptions;
const Result = output.Result;

/// A result's name and shape, which outlive its release.
pub const Summary = struct { plotname: []const u8, npoints: usize, nvars: usize };

/// Why a point produced no value, or not all of them (`point_measures`).
pub const Reason = points.Reason;

/// Sets each global `.param` in `names` to the matching value for every
/// query appended afterwards; the others keep their last values. Queries
/// appended earlier, finished or not, keep the values they were appended
/// with, so a sizing loop appends its analyses again after each call (their
/// plots are named `<plot> (<name>=<value>, ...)`).
///
/// Nothing is re-parsed: the point is re-evaluated from the prepared
/// netlist like a `.step` point, a binned MOSFET whose L or W moves takes
/// its new bin, and the circuit is rebuilt only when parameter writes
/// cannot express the point (dev/analysis/variants.md §5).
///
/// The first call also runs the deck's pending `.model OPT` optimization
/// and MOSRA stress run, so it can take as long as they do and fail with
/// their errors.
///
/// All-or-nothing. Fails with `error.UnknownParameter` for a name not in
/// `Options.live_params` (nor a `.step` name), `error.InvalidArgument` when
/// the lengths differ or a value is not finite, `error.ModelBinNotFound`
/// when no bin holds a new geometry, and `error.UnsupportedCard` when the
/// values would flip a top-level `.if` or change the topology. Each call
/// keeps one row until `deinit`, and each distinct bin selection it reaches
/// keeps one circuit.
pub const set_params = points.setParams;

/// Appends `count` points, each the deck's analyses run at one row of
/// `values` (`count` rows of `Options.live_params` values, row-major), as a
/// `.step` point would run them, and returns point 0's first query id.
/// Points are consecutive in request order (implicit operating points sit
/// between their ids). Each point is one `set_params` row; a point it
/// refuses is still appended, its queries failed with `TopologyChanged` (a
/// topology change or a flipped top-level `.if`), `ModelBinNotFound`,
/// `NonFiniteParameter` or, for any other error building its circuit,
/// `PointRefused`. The first call also runs the deck's `.model OPT` and
/// MOSRA runs.
///
/// All-or-nothing. `error.NoLiveParameters` without `Options.live_params`,
/// `error.SweptTemplate` for a deck whose analyses already sweep (`.step`,
/// `.data`, `.alter`, Monte Carlo), `error.InvalidArgument` for zero
/// points, `error.UnknownMeasure` for an `abort_measure` no card names,
/// `error.TangentsUnavailable` for `options.tangent_step` on a transient
/// that cannot integrate tangents (a Gear-3..6 method). Asserts that
/// `values` holds `count` rows.
///
/// With `options.tangent_step` h, each point's transient also integrates
/// the first-order prediction (`Tran.tangents`) of the point with live
/// parameter k moved by h·|value| (h when the value is 0), one per
/// parameter; `point_tangents` reads the measures' slopes off them.
pub const append_points = points.appendPoints;

/// Writes the `.meas` cards `names` (case-insensitive) of `reasons.len`
/// points into `out` (`reasons.len` rows of `names.len`, row-major), and
/// each point's reason. A point is a run of consecutive requested queries
/// on one variant row, starting at `first`: each point of `append_points`
/// or of a swept deck, or the whole of an unswept deck. A card reads the
/// point's first analysis of its kind, and an `op` card the point's own
/// operating point when it has no `.op`. A point that failed reads NaN
/// everywhere with its failure's reason; else NaN where a card found no
/// value, the reason naming the first such card (`measure_not_found` for a
/// missing vector or analysis, `measure_unresolved` for an event that never
/// happened).
///
/// `error.UnknownMeasure` for a name no card has, `error.InvalidQuery` for
/// a `first` that is not a requested query or fewer points after it,
/// `error.ResultUnavailable` while one of them is unfinished; nothing is
/// written then. Asserts that `out.len == reasons.len * names.len`.
pub const point_measures = points.pointMeasures;

/// Frees `count` points from `first` (as `point_measures` counts them):
/// each query's circuit, work and result, its OP's, and its tangent steps,
/// so a long-lived Problem appended to batch after batch stays the same
/// size. Each query keeps its id and status; `result`, `point_measures` and
/// `point_tangents` then fail with `error.ResultUnavailable`, and nothing
/// publishes it. A point already released is left alone. Errors as
/// `point_measures`.
// ponytail: each query's row (a few hundred bytes) and the point's variant
// row (its label and parameter writes) stay until `deinit`; drop a released
// prefix of both if sessions reach billions of points.
pub const release_points = points.releasePoints;

/// The slopes d(card)/d(live parameter) of the `.tran` cards `names` at
/// `count` points appended with `PointOptions.tangent_step`, starting at
/// `first`, into `out` (`count` × `names.len` × live parameters,
/// row-major): each card over the parameter's tangent prediction minus over
/// the point's own run, over the step. NaN where either has no value, for a
/// point that failed or was appended without tangents, for a card of
/// another analysis, and for a step `set_params` refused. Errors as
/// `point_measures`. Asserts that `out.len == count * names.len * live
/// parameters`.
pub const point_tangents = points.pointTangents;

/// Writes each `.model OPT` optimization's summary, then the deck's `.meas`
/// results for every completed result a card can read (tran, transient
/// noise, AC, DC, FFT, S-parameter, mismatch, loop stability, phase and
/// periodic noise), in request order, as ngspice prints them, then the
/// Monte Carlo statistics of each tran, AC and DC card. Cards that cannot
/// be measured are reported on `err`. With `Options.measures`, it writes
/// only what `run_all` has not printed there, and returns the first error
/// printing there.
pub const print_measures = report.printMeasures;

/// With `Options.point.tangent_step`, writes d(card)/d(live parameter) of
/// every `.tran` card at the deck's first transient (`point_tangents`), one
/// `d(card)/d(param) = value` line each; NaN where a card has no value.
/// Nothing without tangents.
pub const print_tangents = report.printTangents;

/// What `init` allocated the Problem from; every per-call buffer comes
/// from it too. It need not be thread-safe: workers use `query_memory`.
gpa: Allocator,
io: std.Io,
/// `gpa` serialized and counted, for the query workers (sessions,
/// executors, circuits, results).
query_memory: CountingAllocator,
/// Owns the prepared circuit and deck for the Problem's lifetime.
arena: std.heap.ArenaAllocator,
/// Every device type this problem can instantiate; outlives `prepared`.
library: frontend.Library,
prepared: frontend.Prepared,
session: Session,
delivery: output.Session,
limits: Limits,
/// Index into `session.outputs` of the next result to publish. Results
/// publish in request order, so a finished query waits for earlier ones.
next_output: usize = 0,
/// Under a final plan, the main session's outputs below this index were
/// released once written (`publish.releaseDelivered`), and the shape of
/// each released result, for `summary`. `released_upto <= next_output`.
released_upto: usize = 0,
released: std.AutoHashMapUnmanaged(QueryId, Summary) = .empty,
/// First output failure. Once set, nothing else is published.
delivery_error: ?publish.DeliveryError = null,
/// The query whose rows stream to the output as it runs (`openStream`).
streaming: ?QueryId = null,
/// Main-session queries whose rows stream to a side file as they run
/// (`openSide`), until delivered. At most `max_parallel` at once.
sides: std.AutoHashMapUnmanaged(QueryId, *output.Session.Stream) = .empty,
/// The variant row `set_params` last made, which queries appended since
/// then run at; null before any call.
variant: ?u32 = null,
/// `Options.live_params` as live slots (`Tuning.index`), resolved once so a
/// point of `append_points` costs no name lookups, and the names as given.
/// Equal lengths; both empty without live parameters.
live: []const u32 = &.{},
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
/// Holds the netlist an HSPICE optimization or a live parameter
/// re-evaluates (`prepared.tuner`, `prepared.tuning`); null otherwise.
parse_arena: ?std.heap.ArenaAllocator = null,
/// The finished optimizations, once `optimizer.runPending` has run, one per
/// `Tuner.bases` entry: Levenberg-Marquardt fits or bisections, as the
/// `.model OPT` says. At most one of the two is non-empty.
optimization: []core.lm.Lm = &.{},
bisection: []core.bisect.Bisect = &.{},
/// The MOSRA stress run has published its degradation table.
aged: bool = false,
/// Whether a query of this Problem brought the GPU driver up, which every
/// session's queries read through `ExecutionConfig.gpu_driver`.
gpu_driver: analysis.GpuDriver = .{},
/// `Options.auto_threads`' pool, lent through `ExecutionConfig.thread_pool`.
thread_pool: analysis.ThreadPool = .{ .free = .init(0) },
/// The query threads every session's workers run on.
threads: analysis.Threads,
/// The solver workspaces sweep points hand on (`ExecutionConfig.ws_pool`).
ws_pool: analysis.WsPool,
/// Each transient appended with `PointOptions.tangent_step`: the step of
/// each live parameter its tangent rows took (`point_tangents`).
tangent_steps: std.AutoHashMapUnmanaged(QueryId, []const f64) = .empty,
/// `Options.measures` and how far `run_all` has printed to it.
live_measures: ?report.LiveMeasures = null,
/// The outputs `run_all` has checked for failures, and the first it found,
/// which every later call returns too.
checked_outputs: usize = 0,
first_failure: ?anyerror = null,

const Problem = @This();

/// A variant run's queries and its next result to publish.
const Run = struct { session: Session, next_output: usize = 0 };

/// Parses and prepares `options.source`, then queues the deck's analyses (a
/// lone `.op` when the deck names none). Nothing runs yet, except the
/// nominal operating point a warm start reads.
///
/// Fails on a netlist error, a backend the build lacks, zero
/// `max_parallel` (`error.InvalidConcurrency`), an unknown
/// `point.abort_measure` (`error.UnknownMeasure`), or a query the output
/// format cannot hold. Caller owns the result and frees it with `deinit`.
pub fn init(gpa: Allocator, io: std.Io, options: Options) !*Problem {
    var lap = lapStart(io, options.timing_in_depth);
    if (options.max_parallel == 0) return error.InvalidConcurrency;
    try analysis.validateBackend(options.backend);
    const self = try gpa.create(Problem);
    errdefer gpa.destroy(self);
    // The undefined fields are set below, each before its first read.
    // `self` is heap-allocated, so the arenas and the counting allocator
    // keep their addresses.
    self.* = .{
        .gpa = gpa,
        .io = io,
        .query_memory = .init(gpa, io),
        .arena = .init(gpa),
        .library = undefined,
        .prepared = undefined,
        .session = undefined,
        .delivery = undefined,
        .limits = .{ .max_parallel = options.max_parallel },
        .parse_arena = .init(gpa),
        .threads = undefined,
        .ws_pool = undefined,
    };
    const workers = self.query_memory.allocator();
    self.threads = .init(workers, io);
    self.ws_pool = try .init(workers, io, options.max_parallel);
    errdefer self.ws_pool.deinit();
    errdefer self.arena.deinit();
    if (analysis.gpu_available and options.backend.backend != .cpu) {
        // The driver reads the JIT cache cap at its start.
        config.raiseJitCache();
        if (config.likelyGpu(options)) self.gpu_driver.prestart();
    }
    errdefer self.gpu_driver.deinit();
    const arena = self.arena.allocator();
    self.origin = try arena.dupe(u8, switch (options.source) {
        .file => |path| path,
        .bytes => |bytes| bytes.origin,
    });
    // An optimization or a live parameter keeps the parse arena: its tuner
    // re-evaluates the netlist.
    var keep_parse = false;
    defer if (!keep_parse) {
        self.parse_arena.?.deinit();
        self.parse_arena = null;
    };
    const scratch = self.parse_arena.?.allocator();

    // Front end: source, parse, `.hdl` models, then the build.
    self.library = try frontend.Library.init(gpa);
    errdefer self.library.deinit();
    self.library.toolchain = try config.toolchainFromEnv(arena, options.hdl_toolchain);
    self.library.vpi = options.vpi;
    const ast = try frontend.prepare(io, scratch, options.source, options.dialect, options.live_params);
    try frontend.loadModels(io, &self.library, scratch, ast, self.origin, options.vpi_apps);
    timingLap(io, &lap, "frontend (source, parsing, HDL)");
    self.prepared = try frontend.build(&self.library, arena, scratch, ast);
    timingLap(io, &lap, "Problem creation (expansion, binding, topology)");
    errdefer self.prepared.deinit();
    if (self.prepared.tuning) |t| {
        const live = try arena.alloc(u32, options.live_params.len);
        for (options.live_params, live) |name, *k| k.* = try t.index(name);
        self.live = live;
        self.live_names = try config.dupeAll(arena, options.live_params);
    }
    assert(self.live.len == self.live_names.len);
    self.out_dir = try arena.dupe(u8, if (options.output.path) |out| std.fs.path.dirname(out) orelse "." else ".");
    self.save_path = try deck_files.savePath(arena, self.prepared.deck.save_op, self.out_dir, self.origin);
    self.delivery = try output.Session.init(gpa, options.output);
    errdefer self.delivery.deinit();

    // The config every session shares.
    if (options.auto_threads and !builtin.single_threaded) self.thread_pool = .init(std.Thread.getCpuCount() catch 1);
    const execution = config.execution(options, &self.prepared, .{
        .gpu_driver = &self.gpu_driver,
        .threads = &self.threads,
        .ws_pool = &self.ws_pool,
        .thread_pool = if (options.auto_threads and !builtin.single_threaded) &self.thread_pool else null,
    });
    if (options.measures) |to| self.live_measures = .{ .to = to };
    self.session = Session.init(workers, io, &self.prepared.circuit, &self.prepared.deck, execution);
    errdefer self.session.deinit();
    // An optimization's sessions run before this one; nothing after it
    // instantiates the template. `set_params` diffs later points against
    // the template, so with live parameters it is never handed to a query
    // to keep.
    if (self.prepared.tuning == null) self.session.markLastTemplateReader();

    // The deck's own queries.
    try points.checkPoint(self, options.point);
    const point = config.pointOptions(options, &self.prepared, self.save_path);
    const jobs = try config.deckJobs(scratch, &self.prepared.deck, point);
    if (point.warm_start) try points.solveWarmOp(self, jobs);
    // `tangent_step` on the deck's own transients: tangent rows at the
    // nominal live values, which `print_tangents` reads.
    const steps = try scratch.alloc([]const f64, jobs.len);
    if (point.tangent_step > 0) try points.tangentDeck(self, jobs, point.tangent_step, steps);
    const ids = try scratch.alloc(QueryId, jobs.len);
    _ = try self.append_queries(jobs, ids);
    if (point.tangent_step > 0) for (jobs, ids, steps) |job, id, s| if (job == .tran) {
        try self.tangent_steps.put(arena, id, s);
    };

    // One session per variant run with its own topology.
    self.runs = try arena.alloc(Run, self.prepared.runs.len);
    for (self.runs, self.prepared.runs, 0..) |*run, *prep, k| {
        errdefer for (self.runs[0..k]) |*done| done.session.deinit();
        run.* = .{ .session = Session.init(workers, io, &prep.circuit, &prep.deck, execution) };
        run.session.markLastTemplateReader();
        errdefer run.session.deinit();
        try self.checkOutput(prep, prep.deck.queries);
        _ = try run.session.append(try config.withPoint(scratch, prep.deck.queries, point), try scratch.alloc(QueryId, prep.deck.queries.len));
    }
    timingLap(io, &lap, "query graph and output setup");
    keep_parse = self.prepared.tuner != null or self.prepared.tuning != null;
    assert(self.runs.len == self.prepared.runs.len);
    return self;
}

/// Cancels and joins any paused query workers, then frees everything.
/// Invalidates every result slice handed out. A Problem destroyed before
/// its last query ended drops the plots published into binary or ASCII raw
/// or `.print`, and the file keeps its old bytes.
pub fn deinit(self: *Problem) void {
    const gpa = self.gpa;
    for (self.runs) |*run| run.session.deinit();
    self.released.deinit(gpa);
    self.session.deinit();
    self.gpu_driver.deinit();
    self.sides.deinit(gpa);
    self.ws_pool.deinit();
    self.threads.deinit();
    self.delivery.deinit();
    self.prepared.deinit();
    self.library.deinit();
    if (self.parse_arena) |*parse| parse.deinit();
    self.arena.deinit();
    gpa.destroy(self);
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
pub fn output_error(self: *const Problem) ?publish.DeliveryError {
    return self.delivery_error;
}

/// True once every requested query has ended, completed or not: the stop
/// condition of a stepping loop over `ready_queries`.
pub fn finished(self: *const Problem) bool {
    return self.session.finished();
}

/// Queries in the DAG, prerequisites included. IDs run `0..query_count()`.
pub fn query_count(self: *const Problem) u32 {
    return self.session.count();
}

/// A snapshot of one query's state. `error.InvalidQuery` for an ID outside
/// `0..query_count()`.
pub fn query_info(self: *const Problem, id: QueryId) !QueryInfo {
    return self.session.info(id);
}

/// How many queries in `scope` can advance now: the length `ready_queries`
/// needs.
pub fn ready_count(self: *const Problem, scope: Scope) !usize {
    return self.session.readyCount(scope);
}

/// Writes the queries in `scope` that can advance now into `ids` and
/// returns how many there are. Fails with `error.BufferTooSmall`, `ids`
/// untouched, when `ids` is shorter than `ready_count`.
pub fn ready_queries(self: *const Problem, scope: Scope, ids: []QueryId) !usize {
    return self.session.readyQueries(scope, ids);
}

/// Adds `queries` and their prerequisites to the DAG, writing each one's ID
/// into `ids`. Returns `queries.len`.
///
/// When `ids` is too short this is a sizing call: nothing is validated or
/// added. Otherwise the append is all-or-nothing: a query the output format
/// cannot hold, or any other failure, leaves the DAG and `ids` unchanged.
/// An `.op` identical to an existing OP, an implicit prerequisite included,
/// returns that OP's ID; every other query gets a new ID. After
/// `set_params`, a query that names no variant of its own runs at the
/// values the last call set.
pub fn append_queries(self: *Problem, queries: []const Query, ids: []QueryId) !usize {
    if (ids.len < queries.len) return queries.len;
    const jobs: []const Query = if (self.variant) |v| at_variant: {
        const copy = try self.gpa.dupe(Query, queries);
        for (copy) |*q| switch (q.*) {
            inline else => |*o| if (o.tol.variant == null) {
                o.tol.variant = v;
            },
        };
        break :at_variant copy;
    } else queries;
    defer if (self.variant != null) self.gpa.free(jobs);
    try self.checkOutput(&self.prepared, jobs);
    const n = try self.session.append(jobs, ids);
    assert(n == queries.len);
    return n;
}

/// `append_queries` for SPICE analysis directives (`.ac dec 10 1 1meg`),
/// resolved against the prepared circuit. A device card is
/// `error.UnsupportedDirectiveMutation`; text with no analysis card is
/// `error.InvalidAnalysisArguments`.
pub fn append_directives(self: *Problem, text: []const u8, ids: []QueryId) !usize {
    var scratch = std.heap.ArenaAllocator.init(self.gpa);
    defer scratch.deinit();
    const jobs = try frontend.resolveQueries(scratch.allocator(), &self.prepared, text);
    return self.append_queries(jobs, ids);
}

/// Refuses, before anything runs, the first of `queries` whose result the
/// output format cannot hold over `prep`'s circuit and deck.
pub fn checkOutput(self: *const Problem, prep: *const frontend.Prepared, queries: []const Query) !void {
    for (queries) |query| {
        const schema = try analysis.schemaOf(self.gpa, &prep.circuit, &prep.deck, query);
        try output.validateQuery(self.delivery.selection.format, schema, prep.deck.title, prep.deck.probe_labels);
    }
}

/// Most points the executor runs together as one lane batch on this
/// Problem's backend (`analysis.batchWidth`). Advisory.
pub fn lane_width(self: *const Problem) u32 {
    return analysis.batchWidth(self.session.config);
}

/// Runs one quantum of `id`, or of its first unfinished prerequisite, then
/// publishes whatever completed; in binary or ASCII raw or `.print` the
/// plots reach the file in one replacement when the last query ends, as
/// with `run_all` (#179). A terminal `id` returns its status without
/// running anything. A numerical failure is reported in the event, not as
/// an error, and an output failure in its `delivery_error`.
/// `error.InvalidQuery` for an ID outside `0..query_count()`. The first of
/// `advance`, `advance_ready`, `run_all` and `set_params` also runs the
/// deck's `.model OPT` optimization and MOSRA stress run.
pub fn advance(self: *Problem, id: QueryId) !Advance {
    try optimizer.runPending(self);
    var event = try self.session.advance(id);
    publish.deliverFrom(self, &self.session, &self.next_output);
    publish.finishWhenDone(self);
    event.delivery_error = self.delivery_error;
    return event;
}

/// Advances every query in `ids` by one quantum, up to `limits.max_parallel`
/// at a time, writing one event per query in `ids` order, and returns
/// `ids.len`. `ids` must be ready and distinct (`error.QueryNotReady`,
/// `error.DuplicateQuery`), `limits.max_parallel` nonzero
/// (`error.InvalidConcurrency`) and `events` at least as long as `ids`
/// (`error.BufferTooSmall`); a bad frontier starts nothing. Publishes as
/// `advance` does.
pub fn advance_ready(self: *Problem, ids: []const QueryId, limits: Limits, events: []Advance) !usize {
    try optimizer.runPending(self);
    const n = try self.session.advanceReady(ids, limits, events);
    assert(n <= events.len);
    publish.deliverFrom(self, &self.session, &self.next_output);
    publish.finishWhenDone(self);
    for (events[0..n]) |*event| event.delivery_error = self.delivery_error;
    return n;
}

/// Runs every query to completion, publishes the results and finishes the
/// output, then writes the `.save` and `.stim` files. Returns the first
/// query failure, else `error.DeliveryFailed` if output failed, once every
/// query has ended; no `.save` or `.stim` file is written then. Calling it
/// again is a no-op for finished queries and publishes nothing twice.
pub fn run_all(self: *Problem) !void {
    const timing = self.session.config.timing_in_depth;
    var lap = lapStart(self.io, timing);
    defer timingLap(self.io, &lap, "run total (analysis, scheduling, output)");
    try optimizer.runPending(self);
    try publish.openStream(self);
    if (publish.sidesAllowed(self)) self.session.stream_opener = .{ .ctx = self, .open = publish.openSide };
    // Whole-query quanta: nothing here reads intermediate progress, except
    // the in-depth timing, which reports per checkpoint. The variant runs
    // share the pool, after the main session's ready queries.
    var pool: analysis.session.Pool = try .init(self.gpa, self.io, .{ .max_parallel = self.limits.max_parallel, .quantum = if (timing) .interval else .completion });
    defer pool.deinit();
    try pool.add(&self.session);
    for (self.runs) |*run| try pool.add(&run.session);
    while (true) {
        self.fillPool(&pool);
        if (!pool.wait()) break;
        // Refill the slots `wait` freed before encoding and printing, so
        // workers run while the coordinator delivers.
        self.fillPool(&pool);
        self.deliverReady();
    }
    self.deliverReady();
    if (!self.finished()) return error.SchedulingFailure;
    for (self.runs) |*run| if (!run.session.finished()) return error.SchedulingFailure;
    if (self.first_failure == null) self.first_failure = self.session.failureFrom(self.checked_outputs);
    self.checked_outputs = self.session.outputs.items.len;
    if (self.first_failure) |err| return err;
    for (self.runs) |*run| if (run.session.failure()) |err| return err;
    if (self.delivery_error != null) return error.DeliveryFailed;
    // Every query ended and nothing failed, so every result published.
    assert(self.next_output == self.session.outputs.items.len);
    if (timing) {
        const usage = self.query_memory.usage();
        std.debug.print("timing: query memory: peak {d} B, {d} B held at the end\n", .{ usage.peak, usage.live });
    }
    var output_lap = lapStart(self.io, timing);
    defer timingLap(self.io, &output_lap, "output finish");
    self.delivery.finish(self.io) catch |err| {
        self.delivery_error = err;
        return error.DeliveryFailed;
    };
    // Both build their text in a `Writer.Allocating`, which reports running
    // out of memory as `error.WriteFailed`.
    const deck = &self.prepared.deck;
    if (self.save_path) |path| deck_files.writeSave(self.gpa, self.io, &self.session, deck.title, path, deck.save_op.?) catch |err|
        return if (err == error.WriteFailed) error.OutOfMemory else err;
    for (deck.stims) |stim| deck_files.writeStim(self.gpa, self.io, &self.session, deck.title, stim, self.origin, self.out_dir) catch |err|
        return if (err == error.WriteFailed) error.OutOfMemory else err;
}

/// Writes the query DAG with the frontier `options.preview` would run next.
/// Changes no state. A null `options.limits` uses the Problem's.
pub fn print(self: *const Problem, writer: *std.Io.Writer, options: PrintOptions) !void {
    var resolved = options;
    if (resolved.limits == null) resolved.limits = self.limits;
    try self.session.print(writer, resolved);
}

/// The completed results of the variant runs with their own topology (an
/// `.alter` that swaps elements, a point that collapses a node), in output
/// order. Caller owns the slice and frees it with `gpa`; the results stay
/// valid until `deinit`.
pub fn run_results(self: *const Problem, gpa: Allocator) ![]Result {
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

/// Copies a result's data into `buffer` and returns its length
/// (`result(id).data.len`); a longer buffer keeps its tail. Fails with
/// `error.BufferTooSmall`, `buffer` untouched, when it is shorter, and with
/// `error.ResultUnavailable` until the query completes.
pub fn copy_result(self: *const Problem, id: QueryId, buffer: []f64) !usize {
    const data = (try self.result(id)).data;
    if (buffer.len < data.len) return error.BufferTooSmall;
    @memcpy(buffer[0..data.len], data);
    return data.len;
}

/// The query workers' allocations (sessions, executors, circuits, results):
/// bytes held now and the peak so far, in bytes requested. The preparation
/// (`Prepared`) is not counted.
pub fn queryMemory(self: *Problem) CountingAllocator.Usage {
    return self.query_memory.usage();
}

/// The start of a timed phase, or null when timing is off.
pub fn lapStart(io: std.Io, on: bool) ?std.Io.Timestamp {
    return if (on) std.Io.Timestamp.now(io, .awake) else null;
}

/// Prints the time since `start.*` under `label` and restarts the lap. A
/// null `start` means timing is off.
pub fn timingLap(io: std.Io, start: *?std.Io.Timestamp, label: []const u8) void {
    const before = start.* orelse return;
    const now = std.Io.Timestamp.now(io, .awake);
    const elapsed = before.durationTo(now).nanoseconds;
    std.debug.print("timing: {s}: {d:.6}ms\n", .{ label, @as(f64, @floatFromInt(elapsed)) / 1e6 });
    start.* = std.Io.Timestamp.now(io, .awake);
}

// ---- private ----

/// Hands each session's ready queries to `pool`, the main session first.
fn fillPool(self: *Problem, pool: *analysis.session.Pool) void {
    pool.fill(&self.session);
    for (self.runs) |*run| pool.fill(&run.session);
}

/// Publishes what has completed, prints the `.meas` blocks whose turn came,
/// and frees what both are done with.
fn deliverReady(self: *Problem) void {
    publish.deliverRuns(self);
    report.printReady(self);
    publish.releaseDelivered(self);
}
