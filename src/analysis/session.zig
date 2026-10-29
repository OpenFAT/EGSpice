//! Session: the query graph and its cooperative scheduler. In: query
//! requests. Out: rows with their status, progress and results. Each row is
//! one query, an implicit OP prerequisite included; numerical state lives in
//! the row's Executor. Cites #13 (the C API's calling contract) and #154
//! (the `.sens` schema).
//!
//! Contents (every `pub` name, the Session methods included): `Failure`,
//! `PointError`, `QueryId`, `Status`, `terminal`, `Scope`, `Limits`,
//! `Preview`, `PrintOptions`, `QueryInfo`, `Advance`, `Session`, `append`,
//! `appendAt`, `JobRef`, `intern`, `jobAt`, `kindAt`, `opKey`, `addRow`,
//! `markFinal`, `final`, `refuse`, `addTemplate`, `makeExecutor`, `gather`,
//! `peerOf`, `lead`, `gpuLaned`, `staticWide`, `gatherWide`, `shareWithHost`,
//! `settlePeers`, `launch`, `settle`, `laneDone`, `assemble`, `readerDone`,
//! `releaseIfRead`, `advanceReady`, `event`, `requeue`, `nextReady`,
//! `follow`, `followOne`, `runAll`, `advance`, `advanceDone`, `print`,
//! `containsComponent`, `printQuery`, `init`, `markLastTemplateReader`,
//! `StreamOpener`, `streamTo`, `deinit`, `count`, `index`, `effectiveStatus`,
//! `info`, `validateScope`, `contains`, `ready`, `readyCount`,
//! `readyQueries`, `validateReady`, `templateOf`, `finished`, `released`,
//! `outputIndex`, `release`, `failure`, `failureFrom`, `query`, `result`,
//! `operatingPoint`, `Pool`, `init`, `deinit`, `add`, `fill`, `wait`,
//! `schemaOf`

const std = @import("std");
const assert = std.debug.assert;
const core = @import("core");
const requests = core.query;
const Deck = core.Deck;
const Circuit = @import("device").Circuit;
const ParamRef = @import("device").abi.ParamRef;
const execution = @import("executor.zig");
const progress = @import("worker.zig");
const types = @import("types.zig");
const Result = types.Result;
const jobs_mod = @import("session_jobs.zig");
const session_build = @import("session_build.zig");
const session_sched = @import("session_sched.zig");
const session_print = @import("session_print.zig");
const schema = @import("schema.zig");
const none = jobs_mod.none;
const no_output = jobs_mod.no_output;
const no_variant = jobs_mod.no_variant;
const OpIndex = jobs_mod.OpIndex;
const JobIndex = jobs_mod.JobIndex;
const Quantum = progress.Quantum;

/// Why a query ended without a result: its run failed (`QueryError`,
/// `error.QueryCancelled` included), its executor could not be built
/// (`Executor.create`: `error.OutOfMemory`, `error.InvalidOperatingPoint`,
/// a bad variant or AC override) or started (a thread spawn), or its
/// prerequisite ended without one (`error.DependencyFailed`). Names are part
/// of the API: `c_api.zig` maps some to status codes.
pub const Failure = execution.QueryError || types.ErrorOf(execution.Executor.create) ||
    types.ErrorOf(execution.Executor.start) || error{DependencyFailed} || PointError;
/// Why a point of `Problem.append_points` was refused before it ran
/// (`Session.refuse`): a value that needs another topology or flips a
/// top-level `.if`, a geometry no model bin holds, a non-finite value, or
/// any other failure to build the point's circuit (`PointRefused`).
pub const PointError = error{ TopologyChanged, ModelBinNotFound, NonFiniteParameter, PointRefused };

/// A row index into the session, stable for its lifetime.
pub const QueryId = requests.QueryId;

/// Lifecycle of one query row.
pub const Status = enum(u8) {
    pending,
    paused,
    complete,
    failed,
    dependency_failed,
    cancelled,

    /// True once the row will never run again.
    pub fn terminal(self: Status) bool {
        return switch (self) {
            .pending, .paused => false,
            else => true,
        };
    }
};
/// A query selection: everything, one query with its prerequisite, or one
/// connected component (a shared OP and its dependents).
pub const Scope = union(enum) { all, query: QueryId, component: u32 };
/// How many ready queries one scheduling round may run at once, and how far
/// each runs.
pub const Limits = struct { max_parallel: u16 = 1, quantum: Quantum = .checkpoint };
/// Which advancement `print` previews.
pub const Preview = union(enum) { run_all, advance: QueryId, advance_ready: []const QueryId };
/// What `Session.print` shows; `limits` previews a concurrency cap.
pub const PrintOptions = struct {
    scope: Scope = .all,
    preview: Preview = .run_all,
    ascii: bool = false,
    limits: ?Limits = null,
};
/// Snapshot of one query row.
pub const QueryInfo = struct {
    id: QueryId,
    kind: requests.Kind,
    status: Status,
    dependency: ?QueryId,
    component: u32,
    requested: bool,
    progress: ?progress.Event,
    failure: ?Failure,
};
/// What one advancement did: `advanced` may be the prerequisite of
/// `requested`, and `target_status` is the requested query's status after it.
pub const Advance = struct {
    requested: QueryId,
    advanced: QueryId,
    status: Status,
    target_status: Status,
    /// The facade's first output failure (`espice.Problem.output_error`);
    /// analysis cannot name the output module's error set.
    delivery_error: ?anyerror = null,
    progress: ?progress.Event = null,
    failure: ?Failure = null,
};

/// The query graph over one prepared circuit and deck, both borrowed for the
/// session's lifetime. `QueryId` is the row index; every call that takes one
/// fails with `error.InvalidQuery` for an id at or past `count`.
pub const Session = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    topology: *const Circuit,
    deck: *const Deck,
    config: execution.Config,
    rows: std.MultiArrayList(Row) = .empty,
    /// Requested queries in first-request order, for delivery.
    outputs: std.ArrayList(QueryId) = .empty,
    /// Every distinct job appended, its `tol.variant` cleared, deep-copied
    /// into `job_arena`: a row holds an index and its variant (8.7), so a
    /// sweep's points and their OPs hold one copy of their options.
    jobs: std.ArrayList(requests.Query) = .empty,
    job_index: JobIndex = .empty,
    job_arena: std.heap.ArenaAllocator,
    /// Round-robin start for `readyQueries`.
    cursor: u32 = 0,
    /// Every OP row by template and options, for `append` to share one.
    ops: OpIndex = .empty,
    /// Rows with no dependency that have not yet stopped a first quantum:
    /// the template's readers, as `Row.readers` counts a row's.
    root_readers: u32 = 0,
    /// A `Pool`'s view: ready rows in row order from `queue_head`, and the
    /// rows that turned ready since (a finished OP's dependents, a parked
    /// query), taken first.
    queue: std.ArrayList(u32) = .empty,
    queue_head: usize = 0,
    urgent: std.ArrayList(u32) = .empty,
    /// Every row before it is terminal, and every output before
    /// `done_outputs`: where `requeue` and `finished` start, so a session
    /// appended to and run over and over costs what each batch adds.
    done_rows: usize = 0,
    done_outputs: usize = 0,
    /// No other session reads `topology`'s device state after this one, so
    /// under `config.final_plan` its last root query evaluates the template's
    /// own batches instead of a copy (`Circuit.instantiateMove`). Set by
    /// `markLastTemplateReader`.
    last_template_reader: bool = false,
    /// Circuits with `topology`'s pattern that rows can instantiate instead
    /// (a model bin that moved under `Problem.set_params`), oldest first;
    /// each must outlive the session.
    templates: std.ArrayList(*const Circuit) = .empty,
    /// The circuit rows appended from now on instantiate: 0 for `topology`,
    /// k for `templates[k - 1]`.
    template: u32 = 0,
    /// A transient or AC query that writes its rows to `writer` as it records them
    /// instead of keeping them; its Result has empty `data`. Set by `streamTo`.
    stream: ?struct { id: QueryId, writer: *std.Io.Writer } = null,
    /// Asked once for each requested query as it first launches on its own
    /// (no lane peers, not a lane row) and no `streamTo` writer took it:
    /// a writer for its rows, as `streamTo` gives, or null to keep them.
    /// Runs on the thread that calls `Pool.fill`.
    stream_opener: ?StreamOpener = null,

    /// `session_build.zig`.
    pub const append = session_build.append;
    pub const appendAt = session_build.appendAt;
    pub const JobRef = session_build.JobRef;
    pub const intern = session_build.intern;
    pub const jobAt = session_build.jobAt;
    pub const kindAt = session_build.kindAt;
    pub const opKey = session_build.opKey;
    pub const addRow = session_build.addRow;
    pub const markFinal = session_build.markFinal;
    pub const final = session_build.final;
    pub const refuse = session_build.refuse;
    pub const addTemplate = session_build.addTemplate;
    /// `session_sched.zig`.
    pub const makeExecutor = session_sched.makeExecutor;
    pub const gather = session_sched.gather;
    pub const peerOf = session_sched.peerOf;
    pub const lead = session_sched.lead;
    pub const gpuLaned = session_sched.gpuLaned;
    pub const staticWide = session_sched.staticWide;
    pub const gatherWide = session_sched.gatherWide;
    pub const shareWithHost = session_sched.shareWithHost;
    pub const settlePeers = session_sched.settlePeers;
    pub const launch = session_sched.launch;
    pub const settle = session_sched.settle;
    pub const laneDone = session_sched.laneDone;
    pub const assemble = session_sched.assemble;
    pub const readerDone = session_sched.readerDone;
    pub const releaseIfRead = session_sched.releaseIfRead;
    pub const advanceReady = session_sched.advanceReady;
    pub const event = session_sched.event;
    pub const requeue = session_sched.requeue;
    pub const nextReady = session_sched.nextReady;
    pub const follow = session_sched.follow;
    pub const followOne = session_sched.followOne;
    pub const runAll = session_sched.runAll;
    pub const advance = session_sched.advance;
    pub const advanceDone = session_sched.advanceDone;
    /// `session_print.zig`.
    pub const print = session_print.print;
    pub const containsComponent = session_print.containsComponent;
    pub const printQuery = session_print.printQuery;

    /// An empty session. Allocates nothing; `topology` and `deck` must
    /// outlive it.
    pub fn init(allocator: std.mem.Allocator, io: std.Io, topology: *const Circuit, deck: *const Deck, config: execution.Config) Session {
        return .{ .allocator = allocator, .io = io, .topology = topology, .deck = deck, .config = config, .job_arena = .init(allocator) };
    }

    /// Marks this session as the last reader of `topology`'s device state
    /// (`last_template_reader`). Asserts that nothing has been appended, so
    /// every row sees the same answer (#13).
    pub fn markLastTemplateReader(self: *Session) void {
        assert(self.rows.len == 0);
        self.last_template_reader = true;
    }

    /// The `stream_opener` callback: `open(ctx, id, job)` returns the writer
    /// query `id`'s rows go to, or null to keep them. `ctx` must outlive
    /// the session.
    pub const StreamOpener = struct {
        ctx: *anyopaque,
        open: *const fn (ctx: *anyopaque, id: QueryId, job: requests.Query) ?*std.Io.Writer,
    };

    /// Makes transient or AC query `id` write its rows to `writer` as it
    /// records them, its Result keeping empty `data`. False, and nothing
    /// set, once `id` has started: its rows already went elsewhere (#13).
    /// Asserts that no other query streams and that `id` is a tran or AC row.
    pub fn streamTo(self: *Session, id: QueryId, writer: *std.Io.Writer) !bool {
        const i = try self.index(id);
        assert(self.stream == null);
        const job = self.kindAt(i);
        assert(job == .tran or job == .ac);
        if (self.rows.items(.executor)[i] != null) return false;
        self.stream = .{ .id = id, .writer = writer };
        return true;
    }

    /// Cancels and joins every started query, then frees every row, result
    /// and copied option. Every `Result` the session handed out dies here.
    pub fn deinit(self: *Session) void {
        // Newest first: dependents join before the OP products they copied from.
        var i = self.rows.len;
        while (i != 0) {
            i -= 1;
            if (self.rows.items(.executor)[i]) |worker| worker.destroy();
        }
        self.rows.deinit(self.allocator);
        self.outputs.deinit(self.allocator);
        self.templates.deinit(self.allocator);
        self.ops.deinit(self.allocator);
        self.queue.deinit(self.allocator);
        self.urgent.deinit(self.allocator);
        self.jobs.deinit(self.allocator);
        self.job_index.deinit(self.allocator);
        self.job_arena.deinit();
        self.* = undefined;
    }

    /// Returns the row count, implicit prerequisites included.
    pub fn count(self: *const Session) u32 {
        return @intCast(self.rows.len);
    }

    pub fn index(self: *const Session, id: QueryId) !usize {
        const i = @backingInt(id);
        if (i >= self.rows.len) return error.InvalidQuery;
        return i;
    }

    pub fn effectiveStatus(self: *const Session, i: usize) Status {
        const status = self.rows.items(.status)[i];
        if (status.terminal()) return status;
        const dep = self.rows.items(.dependency)[i];
        if (dep != none) {
            const ds = self.rows.items(.status)[@backingInt(dep)];
            if (ds.terminal() and ds != .complete) return .dependency_failed;
        }
        return status;
    }

    /// Returns a snapshot of query `id`. A non-terminal query whose
    /// prerequisite ended without a result reads as `.dependency_failed`.
    pub fn info(self: *const Session, id: QueryId) !QueryInfo {
        const i = try self.index(id);
        const dep = self.rows.items(.dependency)[i];
        const status = self.effectiveStatus(i);
        return .{
            .id = id,
            .kind = self.kindAt(i),
            .status = status,
            .dependency = if (dep == none) null else dep,
            .component = self.rows.items(.component)[i],
            .requested = self.rows.items(.requested)[i],
            .progress = self.rows.items(.progress)[i],
            .failure = if (status == .dependency_failed) error.DependencyFailed else self.rows.items(.failure)[i],
        };
    }

    pub fn validateScope(self: *const Session, scope: Scope) !void {
        switch (scope) {
            .all => {},
            .query => |id| _ = try self.index(id),
            .component => |component| {
                for (self.rows.items(.component)) |c| if (c == component) return;
                return error.InvalidComponent;
            },
        }
    }

    pub fn contains(self: *const Session, scope: Scope, i: usize) bool {
        return switch (scope) {
            .all => true,
            .component => |component| self.rows.items(.component)[i] == component,
            .query => |id| @backingInt(id) == i or self.rows.items(.owner)[i] == id or
                self.rows.items(.dependency)[@backingInt(id)] == @as(QueryId, @fromBackingInt(@intCast(i))),
        };
    }

    pub fn ready(self: *const Session, i: usize) bool {
        if (self.rows.items(.lanes)[i] != 0 or self.effectiveStatus(i).terminal()) return false;
        const dep = self.rows.items(.dependency)[i];
        return dep == none or self.rows.items(.status)[@backingInt(dep)] == .complete;
    }

    /// How many queries in `scope` can advance now: the length
    /// `readyQueries` needs.
    pub fn readyCount(self: *const Session, scope: Scope) !usize {
        try self.validateScope(scope);
        var n: usize = 0;
        for (0..self.rows.len) |i| if (self.contains(scope, i) and self.ready(i)) {
            n += 1;
        };
        return n;
    }

    /// Copies the ready queries in `scope` into `ids` in scheduler order and
    /// returns how many there are. Fails with `error.BufferTooSmall`, writing
    /// nothing, when `ids` is shorter than `readyCount`, as every other
    /// copying call here does (#13).
    pub fn readyQueries(self: *const Session, scope: Scope, ids: []QueryId) !usize {
        const n = try self.readyCount(scope);
        if (ids.len < n) return error.BufferTooSmall;
        var out: usize = 0;
        for (0..self.rows.len) |offset| {
            const i = (offset + self.cursor) % self.rows.len;
            if (!self.contains(scope, i) or !self.ready(i)) continue;
            ids[out] = @fromBackingInt(@intCast(i));
            out += 1;
        }
        return n;
    }

    pub fn validateReady(self: *const Session, ids: []const QueryId, limits: Limits) !void {
        if (limits.max_parallel == 0) return error.InvalidConcurrency;
        for (ids, 0..) |id, pos| {
            const i = try self.index(id);
            if (!self.ready(i)) return error.QueryNotReady;
            for (ids[0..pos]) |earlier| if (earlier == id) return error.DuplicateQuery;
        }
    }

    /// The circuit row `i` instantiates (`Row.template`).
    pub fn templateOf(self: *const Session, i: usize) *const Circuit {
        const t = self.rows.items(.template)[i];
        return if (t == 0) self.topology else self.templates.items[t - 1];
    }

    /// True when every requested query is terminal.
    pub fn finished(self: *const Session) bool {
        for (self.outputs.items[self.done_outputs..]) |id| if (!self.effectiveStatus(@backingInt(id)).terminal()) return false;
        return true;
    }

    /// Whether `release` freed query `id`.
    pub fn released(self: *const Session, id: QueryId) bool {
        return self.rows.items(.released)[@backingInt(id)];
    }

    /// The position of requested query `id` in `outputs`; null when it was
    /// never requested.
    pub fn outputIndex(self: *const Session, id: QueryId) !?usize {
        const k = self.rows.items(.output)[try self.index(id)];
        return if (k == no_output) null else k;
    }

    /// Frees ended query `id`'s executor: its circuit, work and result.
    /// Its status stays; `result` then fails `error.ResultUnavailable`. Its
    /// prerequisite OP goes too once no unreleased query reads it, and
    /// leaves the OP index, so a later query gets a new one. A query
    /// already released is left alone. `error.QueryNotReady` while it, or
    /// a query reading its state, has not ended.
    pub fn release(self: *Session, id: QueryId) error{ InvalidQuery, QueryNotReady }!void {
        const i = try self.index(id);
        if (self.rows.items(.released)[i]) return;
        if (!self.effectiveStatus(i).terminal() or self.rows.items(.running)[i]) return error.QueryNotReady;
        var c = self.rows.items(.child)[i];
        while (c != none) : (c = self.rows.items(.sibling)[@backingInt(c)]) {
            if (!self.rows.items(.released)[@backingInt(c)]) return error.QueryNotReady;
        }
        if (self.rows.items(.executor)[i]) |worker| worker.destroy();
        self.rows.items(.executor)[i] = null;
        self.rows.items(.released)[i] = true;
        if (self.kindAt(i) == .op) {
            const key = self.opKey(i);
            if (self.ops.get(key)) |at| if (@backingInt(at) == i) {
                _ = self.ops.remove(key);
            };
        }
        const dep = self.rows.items(.dependency)[i];
        if (dep != none and !self.rows.items(.requested)[@backingInt(dep)]) self.release(dep) catch |err| switch (err) {
            // Another of its readers has not been released yet.
            error.QueryNotReady => {},
            else => unreachable, // else: `dep` is a valid row
        };
    }

    /// Returns the error of the first requested query that ended without a
    /// result, or null. A query whose prerequisite failed reports the
    /// prerequisite's error, so an implicit OP's OutOfMemory is not renamed
    /// DependencyFailed.
    pub fn failure(self: *const Session) ?Failure {
        return self.failureFrom(0);
    }

    /// `failure` over `outputs[from..]`.
    pub fn failureFrom(self: *const Session, from: usize) ?Failure {
        for (self.outputs.items[from..]) |id| {
            const i = @backingInt(id);
            const status = self.effectiveStatus(i);
            if (status == .complete or !status.terminal()) continue;
            if (self.rows.items(.failure)[i]) |err| return err;
            if (status == .cancelled) return error.QueryCancelled;
            const dep = self.rows.items(.dependency)[i];
            if (dep != none) if (self.rows.items(.failure)[@backingInt(dep)]) |err| return err;
            return error.DependencyFailed;
        }
        return null;
    }

    /// Returns the request `id` was appended as. Its slices live until
    /// `deinit`.
    pub fn query(self: *const Session, id: QueryId) !requests.Query {
        return self.jobAt(try self.index(id));
    }

    /// Returns a completed query's result, valid until `deinit`. Fails with
    /// `error.ResultUnavailable` before completion.
    pub fn result(self: *const Session, id: QueryId) !Result {
        const i = try self.index(id);
        if (self.effectiveStatus(i) != .complete) return error.ResultUnavailable;
        const worker = self.rows.items(.executor)[i] orelse return error.ResultUnavailable;
        return worker.result() orelse error.ResultUnavailable;
    }

    /// The solved operating point query `id` started from: its
    /// prerequisite's, or its own for an `.op`. Null when there is none or
    /// it has not completed. Valid until `deinit`.
    pub fn operatingPoint(self: *const Session, id: QueryId) !?[]const f64 {
        const i = try self.index(id);
        const dep = self.rows.items(.dependency)[i];
        const worker = self.rows.items(.executor)[if (dep == none) i else @backingInt(dep)] orelse return null;
        return worker.operatingPoint();
    }
};

/// A refilling scheduler over the queries of one or more sessions: up to
/// `limit` quanta in flight, a stopped one replaced at once. Every worker
/// posts `signal` when its quantum stops, so the coordinator wakes on
/// whichever stops first. Run it as `add` each session, then `fill` and
/// `wait` until `wait` returns false. Results do not depend on the order
/// queries start in, so output is the same at every `limit`.
pub const Pool = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    limit: u16,
    quantum: Quantum,
    signal: std.Io.Semaphore = .{},
    running: std.ArrayList(Running) = .empty,

    const Running = struct { session: *Session, row: u32 };

    /// An empty pool. `error.InvalidConcurrency` for a zero `max_parallel`.
    pub fn init(allocator: std.mem.Allocator, io: std.Io, limits: Limits) !Pool {
        if (limits.max_parallel == 0) return error.InvalidConcurrency;
        var pool: Pool = .{ .allocator = allocator, .io = io, .limit = limits.max_parallel, .quantum = limits.quantum };
        try pool.running.ensureTotalCapacity(allocator, limits.max_parallel);
        return pool;
    }

    /// Cancels and settles whatever still runs (an error ended the loop
    /// early), then frees the pool.
    pub fn deinit(pool: *Pool) void {
        for (pool.running.items) |r| r.session.rows.items(.executor)[r.row].?.cancel();
        for (pool.running.items) |r| r.session.settle(r.row);
        pool.running.deinit(pool.allocator);
    }

    /// Queues `session`'s ready queries. Call once per run, before `fill`.
    pub fn add(pool: *Pool, session: *Session) !void {
        _ = pool;
        try session.requeue();
    }

    /// Starts `session`'s next ready queries while the pool has room.
    pub fn fill(pool: *Pool, session: *Session) void {
        while (pool.running.items.len < pool.limit) {
            const i = session.nextReady() orelse return;
            // A batch publishes peer outcomes only when its leader ends.
            // Interactive checkpoint quanta must keep independent workers:
            // a parked leader has not completed or cancelled its peers.
            if (pool.quantum == .completion) session.gather(i);
            if (session.launch(i, pool.quantum, &pool.signal)) {
                pool.running.appendAssumeCapacity(.{ .session = session, .row = @intCast(i) });
            } else if (session.rows.items(.executor)[i]) |leader| {
                // The batch never started: its peers run on their own.
                for (leader.peers) |p| session.rows.items(.running)[p.row] = false;
                leader.peers = &.{};
            }
        }
    }

    /// Waits until a running quantum stops, then settles every stopped
    /// one and queues what it made ready. False, without waiting, when
    /// nothing runs.
    pub fn wait(pool: *Pool) bool {
        if (pool.running.items.len == 0) return false;
        pool.signal.waitUncancelable(pool.io);
        var k: usize = 0;
        while (k < pool.running.items.len) {
            const r = pool.running.items[k];
            if (!r.session.rows.items(.executor)[r.row].?.stopped()) {
                k += 1;
                continue;
            }
            r.session.settle(r.row);
            r.session.follow(r.row);
            _ = pool.running.swapRemove(k);
        }
        return true;
    }
};

/// The result shape of a query, without running it (`schema.zig`).
pub const schemaOf = schema.schemaOf;

/// One query row's columns (`Session.rows`).
const Row = struct {
    /// Index into `Session.jobs`; a lane row names its split query's.
    job: u32,
    /// The `tol.variant` the row runs, which `jobs` leaves out so a
    /// sweep's points share one entry; `no_variant` for none.
    variant: u32 = no_variant,
    dependency: QueryId = none,
    component: u32,
    requested: bool,
    status: Status = .pending,
    progress: ?progress.Event = null,
    failure: ?Failure = null,
    executor: ?*execution.Executor = null,
    /// The circuit the query instantiates: 0 for `topology`, k for
    /// `templates[k - 1]`.
    template: u32 = 0,
    /// Rows that read this one's device state (its dependents) and have
    /// not yet stopped a first quantum, which is when their copy is done.
    readers: u32 = 0,
    /// The newest dependent and the next older one sharing `dependency`.
    child: QueryId = none,
    sibling: QueryId = none,
    /// A quantum is in flight; `settle` collects it.
    running: bool = false,
    /// A query split into its lanes (`splitLanes`): its lane rows are the
    /// `lanes` rows just before it. It never runs; `assemble` gathers its
    /// result once every lane has ended.
    lanes: u32 = 0,
    /// On a lane row, the split query it belongs to.
    owner: QueryId = none,
    /// `release` dropped its executor and result.
    released: bool = false,
    /// No query appended later reads its state (`markFinal`), so it is
    /// treated as under `Config.final_plan`.
    final: bool = false,
    /// Its position in `outputs`, or `no_output` when not requested.
    output: u32 = no_output,
};

test "Status.terminal" {
    for ([_]Status{ .pending, .paused }) |s| try std.testing.expect(!s.terminal());
    for ([_]Status{ .complete, .failed, .dependency_failed, .cancelled }) |s| try std.testing.expect(s.terminal());
}
