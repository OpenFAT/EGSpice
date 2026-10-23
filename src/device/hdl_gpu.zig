//! Device kernels for runtime `.hdl` models (performance TODO 01 1.8). The
//! loader builds a host library per model; a GPU build's kernel images are
//! fixed at build time, so a loaded model's batches stayed on the host. Each
//! library the loader opens is recorded here, and the first time the GPU asks
//! for one of its kernels `ptx` compiles the same generated device against
//! `eval.zig` for NVPTX, the way build.zig compiles a built-in model's kernel
//! root, cached beside the library as `gpu-<sm>.ptx`.
//!
//! Only outside Debug: a Debug host library keeps VerA's Debug-only `su_ok`
//! Model field (`eval.vera_validate_contract`), which the `.fast` kernels
//! drop, so their rows would read at the wrong stride.

const std = @import("std");
const builtin = @import("builtin");
const BuildPaths = @import("loader.zig").BuildPaths;

/// Whether loaded models are recorded at all.
pub const enabled = builtin.mode != .debug and !builtin.cpu.arch.isWasm() and builtin.os.tag == .linux;

const Source = struct {
    /// The library as `dlopen` got it, which `dladdr` reports back.
    so_path: []const u8,
    /// Its build directory: `device.zig` and its `u/` tree.
    work_dir: []const u8,
    paths: BuildPaths,
    /// The loading Problem's I/O, which runs the compiler.
    // ponytail: the first loader's; a Problem outliving it would need its own.
    run_io: std.Io,
    /// The PTX once built, or the build failed (null image, `tried`).
    image: ?[:0]const u8 = null,
    tried: bool = false,
};

// ponytail: process-lifetime registry on the page allocator, never freed; a
// deck loads a handful of models.
var sources: std.ArrayList(Source) = .empty;
var mutex: std.Io.Mutex = .init;

fn io() std.Io {
    return std.Io.Threaded.global_single_threaded.io();
}

/// Records the library at `so_path`, built in `work_dir` against `paths`.
/// Best effort: out of memory only leaves the model on the host.
pub fn register(run_io: std.Io, so_path: []const u8, work_dir: []const u8, paths: BuildPaths) void {
    if (comptime !enabled) return;
    const a = std.heap.page_allocator;
    mutex.lockUncancelable(io());
    defer mutex.unlock(io());
    for (sources.items) |s| if (std.mem.eql(u8, s.so_path, so_path)) return;
    const s: Source = .{
        .so_path = a.dupe(u8, so_path) catch return,
        .work_dir = a.dupe(u8, work_dir) catch return,
        .run_io = run_io,
        .paths = .{
            .work_dir = a.dupe(u8, paths.work_dir) catch return,
            .contract = a.dupe(u8, paths.contract) catch return,
            .dyn = a.dupe(u8, paths.dyn) catch return,
            .gompute = a.dupe(u8, paths.gompute) catch return,
            .device_abi = a.dupe(u8, paths.device_abi) catch return,
            .core = a.dupe(u8, paths.core) catch return,
            .zig = a.dupe(u8, paths.zig) catch return,
        },
    };
    sources.append(a, s) catch return;
}

const DlInfo = extern struct {
    fname: ?[*:0]const u8,
    fbase: ?*anyopaque,
    sname: ?[*:0]const u8,
    saddr: ?*anyopaque,
};
extern "c" fn dladdr(addr: *const anyopaque, info: *DlInfo) c_int;

/// The PTX for the loaded model whose code holds `fn_addr` (any function
/// of its library, such as its `Hooks.gpu_payload`), for NVPTX `sm_<sm>`;
/// null for a built-in model, in Debug, or when the build fails (reported
/// once on stderr). Compiles on first use and caches on disk.
pub fn ptx(fn_addr: *const anyopaque, sm: u32) ?[:0]const u8 {
    if (comptime !enabled) return null;
    var info: DlInfo = undefined;
    if (dladdr(fn_addr, &info) == 0) return null;
    const name = std.mem.span(info.fname orelse return null);
    mutex.lockUncancelable(io());
    defer mutex.unlock(io());
    const s = for (sources.items) |*s| {
        if (std.mem.eql(u8, s.so_path, name)) break s;
    } else return null;
    if (!s.tried) {
        s.tried = true;
        s.image = build(s.*, sm) catch |e| blk: {
            std.debug.print("hdl: no device kernels for {s} ({s}); it stays on the host\n", .{ s.so_path, @errorName(e) });
            break :blk null;
        };
    }
    return s.image;
}

/// `gpu-<sm>.ptx` in the model's build directory, compiled when missing.
fn build(s: Source, sm: u32) ![:0]const u8 {
    const a = std.heap.page_allocator;
    const cwd = std.Io.Dir.cwd();
    const gpu_dir = try std.fs.path.join(a, &.{ s.work_dir, "gpu" });
    const out = try std.fmt.allocPrint(a, "{s}/gpu-sm_{d}.ptx", .{ s.work_dir, sm });
    if (cwd.readFileAllocOptions(io(), out, a, .unlimited, .of(u8), 0)) |text| return text else |_| {}
    try cwd.createDirPath(io(), gpu_dir);
    // The decl name is the library's module name, `device`, so the kernel
    // symbols match the ones its `gpuPayload` names.
    const models = try std.fs.path.join(a, &.{ gpu_dir, "models.zig" });
    try cwd.writeFile(io(), .{ .sub_path = models, .data = "pub const device = @import(\"device\");\npub const gpu_kernels = true;\n" });
    const lane_opts = try std.fs.path.join(a, &.{ gpu_dir, "lane_options.zig" });
    try cwd.writeFile(io(), .{ .sub_path = lane_opts, .data = "pub const enabled = true;\n" });
    const core_dir = std.fs.path.dirname(s.paths.core) orelse ".";
    const gompute_dir = std.fs.path.dirname(s.paths.gompute) orelse ".";
    const tmp = try std.fmt.allocPrint(a, "{s}.{d}.tmp", .{ out, std.os.linux.getpid() });
    // Each module's own mode, strip and deps precede its -M, as the build
    // runner passes them.
    var argv: std.ArrayList([]const u8) = .empty;
    try argv.appendSlice(a, &.{ s.paths.zig, "build-obj", "-target", "nvptx64-cuda", "-mcpu", try std.fmt.allocPrint(a, "sm_{d}", .{sm}), "-fno-emit-bin", try std.fmt.allocPrint(a, "-femit-asm={s}", .{tmp}) });
    const Mod = struct { name: []const u8, root: []const u8, deps: []const []const u8 };
    const mods = [_]Mod{
        .{ .name = "root", .root = s.paths.dyn, .deps = &.{ "models", "contract", "device_abi", "gompute", "lane_options" } },
        .{ .name = "models", .root = models, .deps = &.{"device"} },
        .{ .name = "device", .root = try std.fs.path.join(a, &.{ s.work_dir, "device.zig" }), .deps = &.{"contract"} },
        .{ .name = "contract", .root = s.paths.contract, .deps = &.{} },
        .{ .name = "device_abi", .root = s.paths.device_abi, .deps = &.{ "contract", "core" } },
        .{ .name = "core", .root = try std.fs.path.join(a, &.{ core_dir, "ids.zig" }), .deps = &.{} },
        .{ .name = "gompute", .root = try std.fs.path.join(a, &.{ gompute_dir, "device.zig" }), .deps = &.{} },
        .{ .name = "lane_options", .root = lane_opts, .deps = &.{} },
    };
    for (mods) |m| {
        try argv.appendSlice(a, &.{ "-fstrip", "-Ofast" });
        for (m.deps) |d| try argv.appendSlice(a, &.{ "--dep", d });
        try argv.append(a, try std.fmt.allocPrint(a, "-M{s}={s}", .{ m.name, m.root }));
    }
    try argv.appendSlice(a, &.{
        "--cache-dir",        try std.fs.path.join(a, &.{ s.work_dir, ".zig-cache" }),
        "--global-cache-dir", try std.fs.path.join(a, &.{ s.paths.work_dir, "zig-global" }),
        "--name",             "kernels",
    });
    std.debug.print("hdl: compiling device kernels for {s} (sm_{d}), cached afterwards\n", .{ s.so_path, sm });
    const r = try std.process.run(a, s.run_io, .{ .argv = argv.items, .cwd = .{ .path = gpu_dir } });
    if (!r.term.success()) {
        std.debug.print("{s}", .{r.stderr});
        return error.DeviceCompileFailed;
    }
    try cwd.rename(tmp, cwd, out, io());
    return cwd.readFileAllocOptions(io(), out, a, .unlimited, .of(u8), 0);
}

test "ptx: no runtime image for code outside a loaded library" {
    _ = &build; // type-checks the compile path in every mode
    try std.testing.expect(ptx(@ptrCast(&io), 89) == null);
}
