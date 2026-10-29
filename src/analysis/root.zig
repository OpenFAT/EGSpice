//! The analysis module's public seam: the query session, execution config and
//! result schemas. The numerical implementations stay private to the module.
//! In: nothing (re-exports). Out: what `espice.zig` calls.
//! Cites no outside source.
//!
//! Contents:
//! - `pub const session`, `ExecutionConfig`, `Knobs`, `validateBackend`, `batchWidth`
//! - `pub const ThreadPool`, `Threads`, `WsPool`, `GpuDriver`, `gpu_available`
//! - `pub const gpu_lanes`, `schemaOf`, `mosra`

/// The query graph and its scheduler (`Session`).
pub const session = @import("session.zig");
/// Per-Problem execution settings every query shares.
pub const ExecutionConfig = @import("executor.zig").Config;
/// Developer switches `ExecutionConfig.knobs` carries; the facade fills them.
pub const Knobs = @import("types.zig").Knobs;
/// Refuses a backend without kernels in this build or a zero thread count.
pub const validateBackend = @import("executor.zig").validateBackend;
/// Most points one lane batch runs together under a config; advisory.
pub const batchWidth = @import("executor.zig").batchWidth;
/// Threads a Problem lends its large queries (`ExecutionConfig.thread_pool`).
pub const ThreadPool = @import("executor.zig").ThreadPool;
/// Reusable query threads a Problem lends through `ExecutionConfig.threads`.
pub const Threads = @import("worker.zig").Threads;
/// Solver workspaces a Problem's queries share (`ExecutionConfig.ws_pool`).
pub const WsPool = @import("ws_pool.zig").WsPool;
/// Whether the GPU driver came up; the facade owns one per Problem and
/// lends it through `ExecutionConfig.gpu_driver`.
pub const GpuDriver = @import("gpu/context.zig").Driver;
/// This build carries GPU kernel images, so a non-`cpu` backend can use the
/// driver at all.
pub const gpu_available = @import("gpu/context.zig").available;
/// The GPU transient lane backend (phase 4): `policy` decides
/// `ExecutionConfig.gpu_lanes` once per Problem.
pub const gpu_lanes = @import("gpu/lanes.zig");
/// HSPICE MOSRA level 1 aging over a stress transient.
pub const mosra = @import("post/mosra.zig");
/// The result shape of a query, without running it.
pub const schemaOf = session.schemaOf;

test {
    _ = @import("tests/ac.zig");
    _ = @import("tests/circuit.zig");
    // Parking and resuming a worker needs its thread; single-threaded, a
    // worker runs to completion on `start` (worker.zig).
    if (!@import("builtin").single_threaded) _ = @import("tests/executor.zig");
    _ = @import("tests/four.zig");
    _ = @import("tests/gpu.zig");
    _ = @import("gpu/periodic.zig");
    _ = @import("gpu/lane_color.zig");
    _ = @import("gpu/devices.zig");
    _ = @import("tests/periodic.zig");
    _ = @import("tests/pz.zig");
    _ = @import("tests/session.zig");
    _ = @import("tests/sweep.zig");
    _ = @import("tests/transient.zig");
    _ = @import("tests/integration.zig");
    _ = @import("tests/jacobian.zig");
    _ = @import("tests/certify.zig");
    _ = @import("post/mosra.zig");
    _ = @import("Circuit.zig");
    _ = @import("certify.zig");
    _ = @import("circuit_params.zig");
    _ = @import("dc/dc.zig");
    _ = @import("ac/lstb.zig");
    _ = @import("ac/sp.zig");
    _ = @import("ac/acmatch.zig");
    _ = @import("eigen/pz.zig");
    _ = @import("post/disto.zig");
    _ = @import("post/fft.zig");
    _ = @import("sweep/mc.zig");
    _ = @import("sweep/sens.zig");
    _ = @import("dc/xf.zig");
    _ = @import("eigen/qz.zig");
    _ = @import("eigen/arnoldi.zig");
    _ = @import("par_eval.zig");
    _ = @import("session.zig");

    // Imported for semantic analysis only: Zig never type-checks an
    // unreferenced function, and the test step does not link the executor's
    // dispatch into these files.
    _ = @import("pss/hb.zig");
    _ = @import("pss/hb_lptv.zig");
    _ = @import("pss/phasenoise.zig");
    _ = @import("pss/pac.zig");
    _ = @import("pss/pnoise.zig");
    _ = @import("pss/pss.zig");
    _ = @import("pss/pxf.zig");
    _ = @import("pss/qpss.zig");
}
