//! The build-time model catalog: the generated `models` module in, one
//! (binding name, device type) row per decl out, plus each built-in's vtable
//! from its separately compiled object. A leaf, so `Library.zig` reads it
//! without importing `root.zig` (`restricted_imports` in build.zig).
//! Cites nothing: it reflects build.zig's model list.
//!
//! Contents:
//! - `pub const models`
//! - `pub const Entry = struct`
//! - `pub const catalog: []const Entry`
//! - `pub fn modelName(comptime D: type) ?[]const u8`
//! - `pub fn vtable(comptime name: []const u8) *const abi.DeviceVtable`
//! - `pub fn byName(comptime name: []const u8) type`
//! - `pub fn has(comptime name: []const u8) bool`

const std = @import("std");
const assert = std.debug.assert;
const abi = @import("device_abi");

/// Every build-time device type, one decl per binding name.
pub const models = @import("models");

/// One catalog row: a model's binding name and its generated device type.
pub const Entry = struct { name: []const u8, type: type };

/// Every decl of `models` in declaration order; a new models/NAME.va appears
/// here without a hand-kept list.
pub const catalog: []const Entry = blk: {
    const decls = @typeInfo(models).@"struct".decl_names;
    var rows: [decls.len]Entry = undefined;
    for (decls, &rows) |name, *row| row.* = .{ .name = name, .type = @field(models, name) };
    const frozen = rows;
    break :blk &frozen;
};

/// Returns the binding name of device type `D`, or null when `D` is not in
/// the catalog.
pub fn modelName(comptime D: type) ?[]const u8 {
    @setEvalBranchQuota(100_000);
    inline for (catalog) |e| {
        if (e.type == D) return e.name;
    }
    return null;
}

/// Returns built-in `name`'s vtable from the `arp_device_<name>` symbol its
/// object exports, valid for the life of the process. Everything the host
/// needs from a generated device goes through it, so the executable never
/// instantiates a device body. Asserts that the ports come first among the
/// unknowns and that the unknown names cover all or none.
// ponytail: no layout-hash check, unlike loader.zig's `.so` path: these
// objects come from this build graph at the same target and mode. Add the
// exported hash check if that stops being true.
pub fn vtable(comptime name: []const u8) *const abi.DeviceVtable {
    const get = @extern(*const fn () callconv(.c) *const abi.DeviceVtable, .{
        .name = "arp_device_" ++ name,
    });
    const vt = get();
    assert(vt.n_u >= vt.num_ports);
    assert(vt.unknown_names.len == 0 or vt.unknown_names.len == vt.n_u);
    return vt;
}

/// Returns the device type registered as `name`; a compile error when there
/// is none.
pub fn byName(comptime name: []const u8) type {
    if (!has(name)) @compileError("devices: no model named '" ++ name ++ "'");
    return @field(models, name);
}

/// Returns whether a model is registered as `name`.
pub fn has(comptime name: []const u8) bool {
    return @hasDecl(models, name);
}
