//! Ids every layer shares that no other core file owns: the dense device
//! type id and the ground row. Nothing in, two constants out. A leaf with no
//! import but std, because every device object and GPU kernel module
//! imports it as its whole `core` (build.zig `abi_dev`, `k_abi`): an edit
//! here rebuilds every model.
//! Cites: none (ESPice's own MNA row convention).
//!
//! Contents:
//! - `pub const DeviceType = enum(u16) { unset = maxInt(u16), _ }`
//! - `pub const GROUND: u32`

const std = @import("std");

/// Dense id of a device type in a `device.Library`: built-ins first, in
/// catalog order, then runtime-loaded ones. A device never knows its own id;
/// the host stamps it where one is needed (`ParamRef.type`, `CardRef.type`).
pub const DeviceType = enum(u16) { unset = std.math.maxInt(u16), _ };

/// Row 0 of every circuit: the reference node.
pub const GROUND: u32 = 0;
