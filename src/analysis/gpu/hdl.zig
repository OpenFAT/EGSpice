//! Kernel lookup for batch kernels that also finds a runtime `.hdl` model's:
//! the build-time images first, then the PTX `device.hdl_gpu` compiles for
//! the loaded library the batch's code lives in.

const gompute = @import("gompute");
const hdl_gpu = @import("device").hdl_gpu;
const devices = @import("devices.zig");
/// A build with CUDA images; the runtime path is CUDA only.
const on = hdl_gpu.enabled and @import("gompute_kernels").has_cuda;

/// `gompute.rawKernelByName` on this thread's device (`devices.ordinal`),
/// or, when no built-in image has `name`, the kernel of that name in the
/// runtime PTX of the loaded model whose library holds `payload_fn` (its
/// `Hooks.gpu_payload`). Fails with `error.KernelNotFound` when neither has
/// it, as a built-in miss does. Caller owns the result (`deinit`).
pub fn kernel(comptime be: gompute.Backend, payload_fn: *const anyopaque, name: []const u8) !gompute.RawByName(be) {
    return gompute.rawKernelByName(be, name, devices.ordinal) catch |e| {
        if (comptime be != .cuda or !on) return e;
        if (e != error.KernelNotFound) return e;
        return runtimeKernel(payload_fn, name) catch e;
    };
}

fn runtimeKernel(payload_fn: *const anyopaque, name: []const u8) !gompute.RawByName(.cuda) {
    const rt = gompute.runtime.cuda;
    var context = try rt.Context.init(devices.ordinal);
    errdefer context.deinit();
    const image = hdl_gpu.ptx(payload_fn, try context.computeCapability()) orelse return error.KernelNotFound;
    // Cached per image by the runtime: one JIT per process.
    var module = try context.loadModuleFromMemory(image);
    var z: [160:0]u8 = undefined;
    if (name.len >= z.len) return error.KernelNotFound;
    @memcpy(z[0..name.len], name);
    z[name.len] = 0;
    const k = try module.getKernel(&z);
    return .{ .context = context, .module = module, .kernel = k };
}

/// The runtime PTX of the loaded model whose library holds `payload_fn`, or
/// null (a built-in model, no device, Debug, a failed compile).
pub fn runtimeImage(payload_fn: *const anyopaque) ?[:0]const u8 {
    if (comptime !on) return null;
    var context = gompute.runtime.cuda.Context.init(devices.ordinal) catch return null;
    defer context.deinit();
    return hdl_gpu.ptx(payload_fn, context.computeCapability() catch return null);
}
