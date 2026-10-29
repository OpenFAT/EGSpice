//! `Options` and the process environment in, the settings a `Problem` runs
//! with out: the execution config its sessions share, the `.hdl` toolchain,
//! the developer knobs, and the point options and query list of the deck's
//! own analyses. Nothing below the facade reads the environment; this file
//! reads all of it, once per Problem.
//! Cites: none.
//!
//! Contents:
//! - `pub const Options = struct`
//! - `pub const MeasureWriters = struct { out: *std.Io.Writer, err: *std.Io.Writer }`
//! - `pub const HdlToolchain = frontend.Library.Toolchain`
//! - `pub const Shared = struct`
//! - `pub fn execution(options: Options, prepared: *const frontend.Prepared, shared: Shared) analysis.ExecutionConfig`
//! - `pub fn pointOptions(options: Options, prepared: *const frontend.Prepared, save_path: ?[]const u8) requests.PointOptions`
//! - `pub fn deckJobs(scratch: Allocator, deck: *const core.Deck, point: requests.PointOptions) Allocator.Error![]requests.Query`
//! - `pub fn withPoint(scratch: Allocator, queries: []const requests.Query, point: requests.PointOptions) Allocator.Error![]requests.Query`
//! - `pub fn toolchainFromEnv(arena: Allocator, given: HdlToolchain) Allocator.Error!HdlToolchain`
//! - `pub fn dupeAll(arena: Allocator, items: []const []const u8) Allocator.Error![]const []const u8`
//! - `pub fn likelyGpu(options: Options) bool`
//! - `pub fn raiseJitCache() void`

const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;
const builtin = @import("builtin");
const analysis = @import("analysis");
const core = @import("core");
const frontend = @import("frontend");
const output = @import("output");
const requests = core.query;

/// Everything `Problem.init` needs. Its slices are copied during init.
pub const Options = struct {
    /// A netlist path, or in-memory text with the origin it is named by.
    source: frontend.Source,
    dialect: frontend.Dialect = .ngspice,
    /// Where results go. `.save` and `.stim` files land in its path's directory.
    output: output.Selection = .{},
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
    vpi: ?*const frontend.Library.Vpi = null,
    /// What every `.tran` of the deck does, as `append_points` asks of a
    /// point (the CLI's `--measures-only`, `--early-stop`, `--abort-after`).
    /// `measures_only` is dropped for a deck with `.save` or `.stim`, which
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
};

/// `Options.measures`: the `.meas` values and the cards' error lines.
pub const MeasureWriters = struct { out: *std.Io.Writer, err: *std.Io.Writer };

/// Where runtime `.hdl` compilation finds its toolchain (`Options.hdl_toolchain`).
pub const HdlToolchain = frontend.Library.Toolchain;

/// The Problem-owned state every session's queries share. Each pointer
/// must outlive every session built from the config; `thread_pool` is
/// null unless `Options.auto_threads` made one.
pub const Shared = struct {
    gpu_driver: *analysis.GpuDriver,
    threads: *analysis.Threads,
    ws_pool: *analysis.WsPool,
    thread_pool: ?*analysis.ThreadPool,
};

/// 1 MiB: post-layout decks (chain_bsim4_10k is 3.3 MB; `auto` takes the
/// GPU for it, 6.6 → 6.2 s with the start overlapped). Below it `auto`
/// mostly stays on the CPU, and a started context costs a CPU run its
/// teardown (stress/scaling_parallel_inverters_2000, 199 KiB: 0.6 → 1.0 s).
const likely_gpu_bytes = 1 << 20;

/// Returns the config every session of the Problem runs with: `options`'
/// backend with the environment's knobs, the shared state, and the GPU lane
/// policy. Under a bare `auto` the lane policy asks the driver for the FP64
/// rate, a driver start (~250 ms), so only a deck that can batch pays it.
pub fn execution(options: Options, prepared: *const frontend.Prepared, shared: Shared) analysis.ExecutionConfig {
    var config = options.backend;
    config.timing_in_depth = options.timing_in_depth;
    config.knobs = knobsFromEnv();
    // A test whose oracle is the plain scalar run turns the bypass off.
    config.knobs.bypass = options.backend.knobs.bypass;
    config.gpu_driver = shared.gpu_driver;
    config.threads = shared.threads;
    config.ws_pool = shared.ws_pool;
    if (shared.thread_pool) |pool| config.thread_pool = pool;
    config.may_batch = mayBatch(prepared, options);
    config.gpu_lanes = if (config.backend == .auto and !config.gpu_explicit and !config.may_batch)
        false
    else
        analysis.gpu_lanes.policy(config);
    if (config.gpu_lanes) {
        const cpus = std.Thread.getCpuCount() catch 1;
        config.gpu_share = analysis.gpu_lanes.share(cpus);
        config.gpu_static = analysis.gpu_lanes.staticPolicy(config, cpus);
    }
    config.display = options.display;
    assert(config.gpu_driver == shared.gpu_driver);
    assert(config.threads == shared.threads);
    return config;
}

/// Returns `options.point` with the CLI's defaults applied: under
/// `auto_point`, measures-only, early stop and warm start when nothing but
/// the `.meas` cards reads the transient waveforms, and a warm start for
/// any sweep. A deck with `.save` or `.stim` (`save_path`, `deck.stims`)
/// reads every column, so it never runs measures-only.
pub fn pointOptions(options: Options, prepared: *const frontend.Prepared, save_path: ?[]const u8) requests.PointOptions {
    const deck = &prepared.deck;
    var point = options.point;
    if (options.auto_point and !waveformsConsumed(options, prepared, save_path)) {
        point.measures_only = true;
        point.early_stop = true;
        point.warm_start = true;
    }
    // A sweep's points start from the nominal operating point: the same
    // point within Newton's tolerances, in a few iterations instead of
    // the cold ladder (145 → 2 per point on a 20-stage CMOS chain).
    if (options.auto_point and isSweep(deck)) point.warm_start = true;
    if (save_path != null or deck.stims.len != 0) point.measures_only = false;
    if (point.measures_only) assert(save_path == null);
    if (options.point.warm_start) assert(point.warm_start);
    return point;
}

/// Returns the deck's own queries, each carrying `point`: its analysis
/// cards, a lone `.op` when it names none, and nothing for a title-only
/// deck (no cards and no probes), where ngspice runs nothing either.
/// Caller owns the slice, allocated from `scratch`.
pub fn deckJobs(scratch: Allocator, deck: *const core.Deck, point: requests.PointOptions) Allocator.Error![]requests.Query {
    if (deck.queries.len == 0 and deck.probes.len == 0) return scratch.alloc(requests.Query, 0);
    const lone_op = [_]requests.Query{.{ .op = .{ .tol = deck.deck_tol } }};
    const jobs = try withPoint(scratch, if (deck.queries.len == 0) &lone_op else deck.queries, point);
    assert(jobs.len >= 1);
    return jobs;
}

/// Returns a copy of `queries` whose transients carry `point`, and whose
/// every query starts from the warm operating point when `point` asks.
/// Caller owns the slice, allocated from `scratch`.
pub fn withPoint(scratch: Allocator, queries: []const requests.Query, point: requests.PointOptions) Allocator.Error![]requests.Query {
    const copy = try scratch.dupe(requests.Query, queries);
    for (copy) |*q| {
        if (q.* == .tran) q.tran.point = point;
        if (point.warm_start) switch (q.*) {
            inline else => |*o| o.tol.warm_op = true,
        };
    }
    assert(copy.len == queries.len);
    return copy;
}

/// Returns `given` with each null path the environment sets filled in,
/// copied into `arena`: `zig` from `$ZIG`, and `cache_dir` from
/// `$ESPICE_CACHE/hdl`, else `$XDG_CACHE_HOME/espice/hdl`, else
/// `$HOME/.cache/espice/hdl`, else `$TMPDIR/espice-cache/hdl` (default
/// `/tmp`). An empty variable counts as unset for the cache, as `$ZIG`
/// never is.
pub fn toolchainFromEnv(arena: Allocator, given: HdlToolchain) Allocator.Error!HdlToolchain {
    const join = std.fs.path.join;
    const cache_dir = if (given.cache_dir) |d| try arena.dupe(u8, d) else if (nonEmptyEnv("ESPICE_CACHE")) |d|
        try join(arena, &.{ d, "hdl" })
    else if (nonEmptyEnv("XDG_CACHE_HOME")) |d|
        try join(arena, &.{ d, "espice", "hdl" })
    else if (nonEmptyEnv("HOME")) |d|
        try join(arena, &.{ d, ".cache", "espice", "hdl" })
    else
        try join(arena, &.{ nonEmptyEnv("TMPDIR") orelse "/tmp", "espice-cache", "hdl" });
    const zig = given.zig orelse env("ZIG");
    const toolchain: HdlToolchain = .{
        .zig = if (zig) |z| try arena.dupe(u8, z) else null,
        .share = if (given.share) |d| try arena.dupe(u8, d) else null,
        .cache_dir = cache_dir,
        .include_dirs = try dupeAll(arena, given.include_dirs),
    };
    assert(toolchain.include_dirs.len == given.include_dirs.len);
    return toolchain;
}

/// Returns a copy of `items`, the list and each string, in `arena`.
pub fn dupeAll(arena: Allocator, items: []const []const u8) Allocator.Error![]const []const u8 {
    const out = try arena.alloc([]const u8, items.len);
    for (out, items) |*o, s| o.* = try arena.dupe(u8, s);
    return out;
}

/// Whether a run will probably take the GPU, judged before parsing: an
/// explicit `cuda`/`hip`, or a deck of at least `likely_gpu_bytes`, the
/// size where `auto`'s cost model starts admitting post-layout decks.
// ponytail: the top file's size only; a small deck that includes a large
// netlist starts the driver with its first query, as before 05 5.12.
pub fn likelyGpu(options: Options) bool {
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

/// Keeps the driver's JIT cache (`~/.nv/ComputeCache`) from evicting the
/// compact-model kernels, whose cold compiles take 15 s (bsim4va) to 1001 s
/// (hisimhv_va). The default cap is 256 MiB on older drivers and this
/// machine's cache already held 321 MB before they were added. 4 GiB is the
/// driver's maximum. A user's own setting wins. It must run before the
/// driver initializes, which the first GPU query of the process does, so
/// `Problem.init` calls it whenever the backend may use the GPU.
pub fn raiseJitCache() void {
    const libc = struct {
        extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;
    };
    // Only EINVAL (a bad name) or ENOMEM fail; a missing cap costs a
    // recompile, never a wrong result.
    _ = libc.setenv("CUDA_CACHE_MAXSIZE", "4294967296", 0);
}

// ---- private ----

/// Whether `prepared` can hand the scheduler several transients of one
/// template to batch: two in the deck, a variant run, or a template later
/// points append to.
fn mayBatch(prepared: *const frontend.Prepared, options: Options) bool {
    if (prepared.runs.len != 0 or prepared.tuning != null or options.live_params.len != 0) return true;
    var trans: usize = 0;
    for (prepared.deck.queries) |q| trans += @intFromBool(q == .tran);
    assert(trans <= prepared.deck.queries.len);
    return trans >= 2;
}

/// Whether anything besides the `.meas` cards reads the deck's transient
/// waveforms: an output file, `.save`, `.stim`, an optimization, tangents,
/// or a device's display tasks, which print at every accepted step.
/// `.print`, `.probe` and `.plot` are ignored cards, and `.four`, `.fft`
/// and MOSRA run transients of their own.
fn waveformsConsumed(options: Options, prepared: *const frontend.Prepared, save_path: ?[]const u8) bool {
    const deck = &prepared.deck;
    if (options.output.path != null or save_path != null or deck.stims.len != 0) return true;
    if (prepared.tuner != null or options.point.tangent_step > 0) return true;
    for (prepared.circuit.batches) |b| if (b.hooks.say != null) return true;
    for (prepared.runs) |run| for (run.circuit.batches) |b| if (b.hooks.say != null) return true;
    return false;
}

/// Whether any of `deck`'s queries runs a variant row (`.step`, `.data`,
/// `.mc`, `.alter` parameters, a temperature list).
fn isSweep(deck: *const core.Deck) bool {
    for (deck.queries) |q| switch (q) {
        inline else => |o| if (o.tol.variant != null or o.tol.temp_c != null) return true,
    };
    return false;
}

/// The developer switches (`analysis.Knobs`); the field docs name each
/// variable.
fn knobsFromEnv() analysis.Knobs {
    var k: analysis.Knobs = .{};
    if (env("ESPICE_SOLVER")) |s| k.newton.pin = @TypeOf(k.newton.pin).names.get(s) orelse .auto;
    k.newton.opdbg = isSet("ZP_OPDBG");
    k.newton.newton_debug = isSet("ZP_NEWTON_DEBUG");
    k.lu.no_bbd = isSet("ESPICE_NO_BBD");
    if (env("ESPICE_LU_PAR")) |v| k.lu.par = v.len > 0 and v[0] == '1';
    k.lu.par_stats = isSet("ESPICE_LU_PAR_STATS");
    if (env("ESPICE_LANES")) |v| k.lanes = !std.mem.eql(u8, v, "0");
    k.hb_trace = isSet("ESPICE_HB_TRACE");
    k.no_qtape = isSet("ZP_NO_QTAPE");
    k.tran_stats = isSet("ZP_TRAN_STATS");
    k.tran_lte_trace = isSet("ZP_TRAN_LTE_TRACE");
    k.no_tangents = isSet("ZP_NO_TANGENTS");
    if (env("ESPICE_GPU_LU")) |v| k.gpu_lu = std.mem.eql(u8, v, "1");
    k.gpu_lu_check = isSet("ESPICE_GPU_LU_CHECK");
    k.gpu_lu_nogather = isSet("ESPICE_GPU_LU_NOGATHER");
    if (env("ESPICE_GPU_LU_TAIL")) |v| k.gpu_lu_tail = std.fmt.parseInt(u32, v, 10) catch 0;
    if (analysis.Knobs.instrument) {
        k.lu.stats = isSet("ZP_LU_STATS");
        if (env("ESPICE_GPU_LU_BENCH")) |v| k.gpu_lu_bench = std.fmt.parseInt(usize, v, 10) catch 0;
        if (env("ESPICE_GPU_LU_THREADS")) |v| k.gpu_lu_threads = std.fmt.parseInt(u32, v, 10) catch 8;
    }
    if (env("ESPICE_GPU_STATS")) |v| k.gpu_stats = if (std.mem.eql(u8, v, "phases")) .phases else .on;
    k.gpu_block = analysis.Knobs.parseBlock(env("ESPICE_GPU_BLOCK"));
    k.gpu_eval_check = isSet("ESPICE_GPU_EVAL_CHECK");
    k.gpu_nofuse = isSet("ESPICE_GPU_NOFUSE");
    k.gpu_nographs = isSet("ESPICE_GPU_NOGRAPHS");
    k.gpu_host_step = isSet("ESPICE_GPU_HOST_STEP");
    k.xdg_cache_home = env("XDG_CACHE_HOME");
    k.home = env("HOME");
    return k;
}

/// The variable `name`, borrowed from the environment block; null when
/// unset, and always on wasm, which links no libc and has no environment.
fn env(name: [*:0]const u8) ?[]const u8 {
    if (comptime builtin.cpu.arch.isWasm()) return null;
    return std.mem.span(std.c.getenv(name) orelse return null);
}

fn isSet(name: [*:0]const u8) bool {
    return env(name) != null;
}

fn nonEmptyEnv(name: [*:0]const u8) ?[]const u8 {
    const value = env(name) orelse return null;
    return if (value.len == 0) null else value;
}
