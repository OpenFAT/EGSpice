//! The device module's public face: the model catalog, the per-Problem
//! `Library` of device types, the frozen `Circuit` and the runtime-`.hdl`
//! GPU kernels. `abi` is the neutral device ABI; eval.zig implements it.
//! Cites nothing: it only re-exports.
//!
//! Contents:
//! - `pub const abi`
//! - `pub const Circuit`
//! - `pub const hdl_gpu`
//! - `pub const Library`
//! - `pub const DeviceType`
//! - `pub const catalog`
//! - `pub const modelName`
//! - `pub const vtable`
//! - `pub const byName`
//! - `pub const has`

const cat = @import("catalog.zig");

/// The neutral device ABI (abi.zig), shared with every device object.
pub const abi = @import("device_abi");
/// The frozen circuit `Circuit.freeze` builds from the protos.
pub const Circuit = @import("Circuit.zig");
/// Device kernels for runtime `.hdl` models.
pub const hdl_gpu = @import("hdl_gpu.zig");
/// The per-Problem table of device types under dense ids.
pub const Library = @import("Library.zig");
/// A dense device-type id into a `Library`.
pub const DeviceType = abi.DeviceType;
/// Every decl of `models`, in declaration order.
pub const catalog = cat.catalog;
/// Returns the binding name of a device type, or null.
pub const modelName = cat.modelName;
/// Returns a built-in model's vtable from its linked object.
pub const vtable = cat.vtable;
/// Returns the device type registered under a name.
pub const byName = cat.byName;
/// Returns whether a model is registered under a name.
pub const has = cat.has;

test {
    _ = cat;
    _ = @import("tests/catalog.zig");
    _ = @import("tests/jacobian_fd.zig");
    _ = Circuit;
    _ = Library;
    _ = @import("loader.zig");
    _ = hdl_gpu;
}
