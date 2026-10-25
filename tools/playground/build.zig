//! The docs playground: espice.wasm (the C API, simulating in the browser)
//! and cktimg.wasm (netlist text in, schematic SVG out). `pages.yml` runs
//! `zig build` here and copies zig-out/wasm/ into docs/wasm/.
const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.resolveTargetQuery(.{ .cpu_arch = .wasm32, .os_tag = .freestanding });
    const optimize: std.lang.Optimize = .small;
    const ck = b.dependency("cktimg", .{ .target = target, .optimize = optimize });
    const np = ck.module("NetlistParser");
    // cktImg's own symbol set, the one its tests and gallery draw with.
    const files = b.addWriteFiles();
    _ = files.addCopyFile(ck.path("tests/symbols.zon"), "symbols.zon");
    const symbols = b.createModule(.{ .root_source_file = files.add("symbols.zig", "pub const text = @embedFile(\"symbols.zon\");\n") });
    // Its SVG writer lives with its tests, on the public API only.
    const svg = b.createModule(.{
        .root_source_file = ck.path("tests/svg.zig"),
        .imports = &.{.{ .name = "NetlistParser", .module = np }},
    });
    const exe = b.addExecutable(.{ .name = "cktimg", .root_module = b.createModule(.{
        .root_source_file = b.path("cktimg.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "NetlistParser", .module = np },
            .{ .name = "svg", .module = svg },
            .{ .name = "symbols", .module = symbols },
        },
    }) });
    exe.entry = .disabled;
    exe.rdynamic = true;
    b.getInstallStep().dependOn(&b.addInstallArtifact(exe, .{ .dest_dir = .{ .override = .{ .custom = "wasm" } } }).step);

    const wasi = b.resolveTargetQuery(.{ .cpu_arch = .wasm32, .os_tag = .wasi, .cpu_features_add = std.Target.wasm.featureSet(&.{.simd128}) });
    const espice = b.dependency("espice", .{ .target = wasi, .optimize = optimize, .gpu = false, .@"cuda-arch" = "none", .@"hip-arch" = "none" });
    const c_api = espice.module("c_api");
    const root = b.createModule(.{
        .root_source_file = b.path("espice.zig"),
        .target = wasi,
        .optimize = optimize,
        .strip = true,
        .single_threaded = true,
        .imports = &.{
            .{ .name = "c_api", .module = c_api },
            .{ .name = "espice_h", .module = c_api.import_table.get("espice_h").? },
        },
    });
    const wexe = b.addExecutable(.{ .name = "espice", .root_module = root });
    wexe.entry = .disabled;
    wexe.rdynamic = true;
    wexe.wasi_exec_model = .reactor;
    wexe.stack_size = 16 << 20; // the corpus peaks at 541 KiB (dc/device_b3soipd_output.sp)
    b.getInstallStep().dependOn(&b.addInstallArtifact(wexe, .{ .dest_dir = .{ .override = .{ .custom = "wasm" } } }).step);
}
