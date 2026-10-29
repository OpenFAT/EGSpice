//! Per-Problem execution settings every query shares, and the decisions
//! that read only them: which backend is usable, how many lanes a batch
//! takes, which kinds batch as lanes, and the thread pool large queries
//! borrow from.
//! In: the facade's settings. Out: `Config` and its derived answers.
//! Cites #13 (diagnostics channel), #138, #211; dev/analysis/refactoring.md.
//!
//! Contents:
//! - `pub const Config = struct` (`Backend`)
//! - `pub const ThreadPool = struct` (`per_query`, `init`, `take`, `give`)
//! - `pub const lane_width`
//! - `pub fn validateBackend(config: Config) error{GpuBackendUnavailable, InvalidThreadCount}!void`
//! - `pub fn lanesOn(config: Config) bool`
//! - `pub fn batchWidth(config: Config) u32`
//! - `pub fn freqLaneKind(kind: requests.Kind) bool`
//! - `pub fn gpuFreqLaneKind(kind: requests.Kind) bool`

const std = @import("std");
const assert = std.debug.assert;
const requests = @import("core").query;
const types = @import("types.zig");
const gpu = @import("gpu/context.zig");

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
        // A failed exchange means another taker moved `free`; each retry
        // reads the new value, so the loop ends once contention does.
        while (true) {
            const got = @min(cur, want);
            if (got == 0) return 0;
            cur = pool.free.cmpxchgWeak(cur, cur - got, .monotonic, .monotonic) orelse return got;
        }
    }

    /// Returns `n` threads a `take` handed out.
    pub fn give(pool: *ThreadPool, n: u32) void {
        assert(n <= per_query);
        if (n != 0) _ = pool.free.fetchAdd(n, .monotonic); // The old count is not needed.
    }
};

/// Lanes per batch: the CPU's f64 vector width (4 on AVX2, 8 on AVX-512).
pub const lane_width = std.simd.suggestVectorLength(f64) orelse 4;

/// Fails with `error.GpuBackendUnavailable` for a backend this binary has no
/// kernels for (printing the GPU hardware the probe detected) and
/// `error.InvalidThreadCount` for a zero thread count. A machine without the
/// GPU fails later, when the first query brings up the driver.
pub fn validateBackend(config: Config) error{ GpuBackendUnavailable, InvalidThreadCount }!void {
    if (!gpu.requestSupported(config.backend)) {
        const name = @tagName(config.backend);
        // The library's diagnostic channel, which an embedder routes with
        // its own `std_options.logFn`; the CLI's prints to stderr (#13).
        std.log.warn("--backend {s}: this build has no {s} kernels; detected GPU hardware: {s}", .{ name, name, gpu.detectedHardware() });
        return error.GpuBackendUnavailable;
    }
    if (config.solver_threads == 0 or config.device_threads == 0) return error.InvalidThreadCount;
}

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
