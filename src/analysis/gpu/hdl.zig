//! Kernel lookup that also finds a runtime `.hdl` model's kernels. In: a
//! kernel name and the payload function of the batch that wants it. Out: a
//! loaded kernel, from the build-time images first, then from the PTX
//! `device.hdl_gpu` compiles for the loaded library that payload lives in.
//! Cites: none (gompute's raw-kernel API).
//!
//! Contents:
//! - `pub fn kernel(comptime be: gompute.Backend, payload_fn: *const anyopaque, name: []const u8) !gompute.RawByName(be)`
//! - `pub fn runtimeImage(payload_fn: *const anyopaque) ?[:0]const u8`

const std = @import("std");
const assert = std.debug.assert;
const gompute = @import("gompute");
const hdl_gpu = @import("device").hdl_gpu;
const devices = @import("devices.zig");

/// A build with CUDA images; the runtime path is CUDA only.
const runtime_on = hdl_gpu.enabled and @import("gompute_kernels").has_cuda;

/// Longest kernel name the runtime lookup takes (its NUL-terminated copy).
const name_max = 159;

/// `gompute.rawKernelByName` on this thread's device (`devices.ordinal`),
/// or, when no built-in image has `name`, the kernel of that name in the
/// runtime PTX of the loaded model whose library holds `payload_fn` (its
/// `Hooks.gpu_payload`). Fails with `error.KernelNotFound` when neither has
/// it, or with the driver's error. Caller owns the result (`deinit`).
pub fn kernel(comptime be: gompute.Backend, payload_fn: *const anyopaque, name: []const u8) !gompute.RawByName(be) {
    assert(name.len > 0);
    return gompute.rawKernelByName(be, name, devices.ordinal) catch |e| {
        if (comptime be != .cuda or !runtime_on) return e;
        if (e != error.KernelNotFound) return e;
        return runtimeKernel(payload_fn, name) catch e;
    };
}

/// The runtime PTX of the loaded model whose library holds `payload_fn`, or
/// null (a built-in model, no device, Debug, a failed compile). The bytes
/// live as long as the loaded library.
pub fn runtimeImage(payload_fn: *const anyopaque) ?[:0]const u8 {
    if (comptime !runtime_on) return null;
    var context = gompute.runtime.cuda.Context.init(devices.ordinal) catch return null;
    defer context.deinit();
    return hdl_gpu.ptx(payload_fn, context.computeCapability() catch return null);
}

// ---- private ----

fn runtimeKernel(payload_fn: *const anyopaque, name: []const u8) !gompute.RawByName(.cuda) {
    if (name.len > name_max) return error.KernelNotFound;
    const rt = gompute.runtime.cuda;
    var context = try rt.Context.init(devices.ordinal);
    errdefer context.deinit();
    const image = hdl_gpu.ptx(payload_fn, try context.computeCapability()) orelse return error.KernelNotFound;
    // Cached per image by the runtime: one JIT per process.
    var module = try context.loadModuleFromMemory(image);
    var z: [name_max + 1:0]u8 = undefined;
    @memcpy(z[0..name.len], name);
    z[name.len] = 0;
    assert(std.mem.len(@as([*:0]const u8, &z)) == name.len);
    const k = try module.getKernel(&z);
    return .{ .context = context, .module = module, .kernel = k };
}
