//! Linear and nonlinear solvers. Imports only `core`; `analysis` is the
//! only module build.zig wires it into. The re-exports below are the module's API;
//! the engines behind `direct` (sparse_lu, tridiag, bbd, order) and
//! `freq_solve` (lane/lu) are reached through those two. `lane/` holds the
//! W-wide CPU lane solves, `kernels/` the std-only bodies shared with the GPU
//! kernel images. The two kernel roots, lu_device.zig and lane_device.zig,
//! stay beside this file: a module may import only files under its root's
//! directory, and their bodies reach sparse_lu, converger and lane/.
//! In: nothing; out: the namespaces below. Cites nothing.
//!
//! Contents:
//! - `pub const direct = @import("direct.zig")`
//! - `pub const lu_kernels = @import("kernels/lu.zig")`
//! - `pub const dense_lu = @import("dense_lu.zig")`
//! - `pub const freq_solve = @import("freq_solve.zig")`
//! - `pub const fft = @import("fft.zig")`
//! - `pub const gmres = @import("gmres.zig")`
//! - `pub const gcrodr = @import("gcrodr.zig")`
//! - `pub const converger = @import("converger.zig")`
//! - `pub const lanes = @import("lane/lanes.zig")`
//! - `pub const lane_system = @import("lane/system.zig")`
//! - `pub const lane_kernels = @import("kernels/lane.zig")`
//! - `pub const frequency_kernels = @import("kernels/frequency.zig")`
//! - `pub const gmres_lanes = @import("kernels/gmres.zig")`
//! - `pub const periodic_kernels = @import("kernels/periodic.zig")`
pub const direct = @import("direct.zig");
pub const lu_kernels = @import("kernels/lu.zig");
pub const dense_lu = @import("dense_lu.zig");
pub const freq_solve = @import("freq_solve.zig");
pub const fft = @import("fft.zig");
pub const gmres = @import("gmres.zig");
pub const gcrodr = @import("gcrodr.zig");
pub const converger = @import("converger.zig");
pub const lanes = @import("lane/lanes.zig");
pub const lane_system = @import("lane/system.zig");
pub const lane_kernels = @import("kernels/lane.zig");
pub const frequency_kernels = @import("kernels/frequency.zig");
pub const gmres_lanes = @import("kernels/gmres.zig");
pub const periodic_kernels = @import("kernels/periodic.zig");

test {
    _ = @import("tests.zig");
    // In-file tests of the files the API reaches only through other files.
    _ = @import("sparse_lu.zig");
    _ = @import("lane/lu.zig");
    _ = @import("lane/clu.zig");
    _ = @import("gcrodr.zig");
    _ = @import("lane/lanes.zig");
    _ = @import("lane/solver.zig");
    _ = @import("lane/system.zig");
    _ = @import("kernels/lane.zig");
    _ = @import("kernels/frequency.zig");
    _ = @import("kernels/gmres.zig");
    _ = @import("kernels/periodic.zig");
    _ = @import("tridiag.zig");
    _ = @import("order.zig");
    _ = @import("bbd.zig");
    _ = @import("subtree.zig");
}
