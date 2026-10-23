const std = @import("std");
const gompute_build = @import("gompute");
const Build = std.Build;
const Module = Build.Module;
const Import = Module.Import;
const Compile = Build.Step.Compile;

pub fn build(b: *Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.option(std.lang.Optimize, "optimize", "Prioritize performance, safety, or binary size") orelse .fast;
    const debug_info = b.option(bool, "debug-info", "Emit DWARF in Release builds (slow: ~3x LLVM time)") orelse false;
    const gpu_kernels = b.option(bool, "gpu", "Compile the GPU device kernels (default true)") orelse true;
    const single_threaded = b.option(bool, "single-threaded", "Build without threads, as espice.wasm does");
    const cuda_arch = b.option([]const u8, "cuda-arch", "CUDA arch (sm_75, ...), `auto` to probe the build machine, `none` to omit") orelse "auto";
    const hip_arch = b.option([]const u8, "hip-arch", "HIP arch (gfx1100, ...), `none` to omit") orelse "gfx1100";
    const gpu_lanes = b.option(bool, "gpu-lanes", "Also compile the GPU lane kernels (phase 4; default true, ~2x ptxas time)") orelse true;
    const suite = b.option([]const u8, "suite", "bench-suites: one suite under tests/conformance/external/spice-bench (default: all)");
    const use_llvm = optimize != .debug;

    const gompute = b.dependency("gompute", .{});
    const vera = b.dependency("vera", .{ .target = target, .optimize = optimize });
    const vera_exe = b.dependency("vera", .{ .target = b.graph.host, .optimize = .fast }).artifact("vera");
    const contract: Import = .{ .name = "contract", .module = vera.module("contract") };

    // The runtime .hdl loader rebuilds src/device/eval.zig against these roots.
    const bopts = b.addOptions();
    bopts.addOption([]const u8, "contract_path", pathFromRoot(b, vera.builder, "lib/contract.zig"));
    bopts.addOption([]const u8, "dyn_path", pathFromRoot(b, b, "src/device/eval.zig"));
    bopts.addOption([]const u8, "device_abi_path", pathFromRoot(b, b, "src/device/abi.zig"));
    bopts.addOption([]const u8, "core_path", pathFromRoot(b, b, "src/core/root.zig"));
    bopts.addOption([]const u8, "gompute_path", pathFromRoot(b, gompute.builder, "src/root.zig"));
    bopts.addOption(bool, "certify", b.option(bool, "certify", "Certify every converged Newton point against its KCL residual") orelse false);
    bopts.addOption(bool, "lu_instrument", b.option(bool, "lu-instrument", "Compile the LU census (ZP_LU_STATS) and the device-LU bench (ESPICE_GPU_LU_BENCH)") orelse false);
    const bo = bopts.createModule();
    // Its own module, so a new VerA commit recompiles main.zig alone.
    const vera_build = b.addOptions();
    vera_build.addOption([]const u8, "commit", veraCommit(b, vera.builder));

    const M: Cfg = .{
        .b = b,
        .target = target,
        .optimize = optimize,
        .strip = optimize != .debug and !debug_info,
        .single_threaded = single_threaded,
        .use_llvm = use_llvm,
        .gpu_kernels = gpu_kernels,
    };

    const models = discoverModels(b);
    const gen = b.allocator.alloc(Build.LazyPath, models.len) catch @panic("OOM");
    for (models, gen) |m, *out| {
        const run = b.addRunArtifact(vera_exe);
        run.addArg(b.fmt("--expect-module={s}", .{m.module}));
        if (m.hdl == .verilog_a) {
            run.addArgs(&.{ "--emit-zig", "--allow=W0650", "--allow=W0651", "--allow=W0850", "--color=never", "--check", "--display=drop" });
            if (escape_models.has(m.name)) run.addArg("--allow=W0149");
            run.addArgs(&.{ "--zig", b.graph.zig_exe, "--contract" });
            run.addFileArg(vera.path("lib/contract.zig"));
        }
        if (m.include) |inc| {
            run.addArgs(&.{ "-I", pathFromRoot(b, b, "models") });
            run.addFileInput(b.path(b.fmt("models/{s}", .{inc})));
        }
        for (m.inc_dirs) |dir| {
            run.addArg("-I");
            run.addDirectoryArg(dir);
        }
        run.addArg("-o");
        out.* = run.addOutputFileArg2(b.fmt("{s}.zig", .{m.name}), .{});
        // VerA compiles Verilog-AMS as VAMS-2023 only; an older keyword set
        // comes from `begin_keywords (LRM §10.6) around the unedited source,
        // found through its own directory (`inc_dirs[0]`).
        run.addFileArg(if (m.std) |spec| b.addWriteFiles().add(b.fmt("{s}_std.va", .{m.name}), b.fmt(
            "`begin_keywords \"{s}\"\n`include \"{s}\"\n`end_keywords\n",
            .{ spec, std.fs.path.basename(m.file) },
        )) else m.src);
    }

    const g = graph(M, vera, gompute.module("gompute"), models, gen, bo);
    // Warm the same objects consumed by every host test, CLI and library.
    // No separate archive/configuration: Zig owns their content-addressed cache.
    const build_models = b.step("build-models", "Compile and cache the shared host model objects without linking a simulator");
    for (g.objs) |obj| {
        _ = obj.getEmittedBin();
        build_models.dependOn(&obj.step);
    }

    const frontend_bench = b.addExecutable(.{
        .name = "frontend-bench",
        .use_llvm = use_llvm,
        .root_module = M.make(b.path("tests/benchmark/frontend.zig"), &.{.{ .name = "netlist", .module = g.netlist }}),
    });
    const run_frontend_bench = b.addRunArtifact(frontend_bench);
    run_frontend_bench.addPassthruArgs();
    b.step("bench-frontend", "Measure netlist parsing and expansion").dependOn(&run_frontend_bench.step);

    const exe = b.addExecutable(.{
        .name = "espice",
        .use_llvm = use_llvm,
        .use_lld = use_llvm and !target.result.os.tag.isDarwin(),
        .root_module = M.make(b.path("src/main.zig"), &.{
            .{ .name = "espice", .module = g.espice },
            .{ .name = "vera_vpi", .module = vera.module("vpi") },
            .{ .name = "fastvaf", .module = vera.module("vera") },
            contract,
            .{ .name = "vera_build", .module = vera_build.createModule() },
        }),
    });
    exe.root_module.link_libc = true;
    exe.rdynamic = true; // dlopen'd VPI apps resolve vpi_* against the exe
    for (g.objs) |o| exe.root_module.addObject(o);
    b.installArtifact(exe);
    b.step("build-cli", "Build and install the simulator CLI without the static library").dependOn(&b.addInstallArtifact(exe, .{}).step);
    b.step("compile-cli", "Compile the simulator CLI without installing a duplicate").dependOn(&exe.step);
    for ([_]struct { Build.LazyPath, []const u8 }{
        .{ b.path("src/core"), "core" },
        .{ b.path("src/device"), "device" },
        .{ gompute.path("src"), "gompute" },
        .{ vera.path("lib"), "vera/lib" },
    }) |dir| b.installDirectory(.{
        .source_dir = dir[0],
        .install_dir = .prefix,
        .install_subdir = b.fmt("share/espice/{s}", .{dir[1]}),
        .include_extensions = &.{".zig"},
    });

    var roots: std.ArrayList(gompute_build.KernelRoot) = .empty;
    const lane_opts = b.addOptions();
    lane_opts.addOption(bool, "enabled", gpu_lanes);
    const lane_options: Import = .{ .name = "lane_options", .module = lane_opts.createModule() };
    for (models, g.kernel_models) |m, one| {
        if (!gpu_kernels or gpu_host_only.has(m.name)) continue;
        const imports = b.allocator.dupe(Import, &.{ .{ .name = "models", .module = one }, g.kernel_contract, g.kernel_abi, lane_options }) catch @panic("OOM");
        roots.append(b.allocator, .{
            .name = m.name,
            .root = b.path("src/device/eval.zig"),
            .imports = &deviceKernelImports,
            .imports_ctx = @ptrCast(imports.ptr),
            .heavy = m.heavy,
        }) catch @panic("OOM");
    }
    roots.append(b.allocator, .{ .name = "lu", .root = b.path("src/solver/lu_device.zig") }) catch @panic("OOM");
    // The lane root runs `.meas` passes on the device (06 6.8) from the file
    // measure.zig shares with it.
    const lane_scan = b.allocator.dupe(Import, &.{.{ .name = "measure_scan", .module = b.createModule(.{ .root_source_file = b.path("src/output/scan.zig"), .strip = true }) }}) catch @panic("OOM");
    if (gpu_lanes) roots.append(b.allocator, .{ .name = "lane", .root = b.path("src/solver/lane_device.zig"), .imports = &laneKernelImports, .imports_ctx = @ptrCast(lane_scan.ptr) }) catch @panic("OOM");
    gompute_build.emitKernels(b, gompute, exe, .{
        .kernel_roots = roots.items,
        .heavy_lanes = 2,
        .target = target,
        // -Dgpu=false disables solver kernels as well as model kernels.
        // Gompute embeds SASS for the target whenever ptxas and fatbinary
        // are on PATH (the default shell); `.#ci` builds keep PTX only.
        .cuda = if (gpu_kernels) gpuArch(cuda_arch) else .{ .enabled = false },
        .hip = if (gpu_kernels) gpuArch(hip_arch) else .{ .enabled = false },
        .optimize = if (optimize == .debug) .fast else optimize,
    });
    g.analysis.addImport("gompute_kernels", exe.root_module.import_table.get("gompute_kernels").?);

    // Compile the coupled-line state kernels on CPU-only CI runners too.
    // LLVM's AMDGPU frame lowering has failed here even when host tests pass.
    const hip_codegen = b.step("test-hip-codegen", "Compile coupled-line HIP kernels, including lane state kernels");
    const hip_probe_target = b.resolveTargetQuery(std.Target.Query.parse(.{
        .arch_os_abi = "amdgcn-amdhsa",
        .cpu_features = "gfx1100+trap_handler",
    }) catch unreachable);
    const hip_probe_lanes = b.addOptions();
    hip_probe_lanes.addOption(bool, "enabled", true);
    var previous_hip_probe: ?*Compile = null;
    for (models, g.kernel_models) |m, one| {
        if (!std.mem.eql(u8, m.name, "coupled_ltra") and
            !std.mem.eql(u8, m.name, "coupled_ltra3") and
            !std.mem.eql(u8, m.name, "coupled_ltra4")) continue;
        const probe = b.addObject(.{
            .name = b.fmt("hip_codegen_{s}", .{m.name}),
            .root_module = b.createModule(.{
                .root_source_file = b.path("src/device/eval.zig"),
                .target = hip_probe_target,
                .optimize = .fast,
                .strip = true,
                .imports = &.{
                    .{ .name = "gompute", .module = gompute.module("gompute_device") },
                    .{ .name = "models", .module = one },
                    g.kernel_contract,
                    g.kernel_abi,
                    .{ .name = "lane_options", .module = hip_probe_lanes.createModule() },
                },
            }),
        });
        _ = probe.getEmittedBin(); // Type-checking or IR emission alone misses the LLVM abort.
        const symbols = b.addCheckFile(probe.getEmittedLlvmIr(), .{ .expected_matches = &.{
            b.fmt("define amdgpu_kernel void @arp_lim_{s}(", .{m.name}),
            b.fmt("define amdgpu_kernel void @arp_llim_{s}(", .{m.name}),
        } });
        if (previous_hip_probe) |previous| probe.step.dependOn(&previous.step);
        previous_hip_probe = probe;
        hip_codegen.dependOn(&symbols.step);
    }

    const espice_h = b.addTranslateC(.{ .root_source_file = b.path("include/espice.h"), .target = target, .optimize = optimize }).createModule();
    const c_api_mod = M.make(b.path("src/c_api.zig"), &.{
        .{ .name = "espice", .module = g.espice },
        .{ .name = "espice_h", .module = espice_h },
        .{ .name = "core", .module = g.core },
    });
    c_api_mod.link_libc = true;
    for (g.objs) |o| c_api_mod.addObject(o);
    const c_api_lib = b.addLibrary(.{ .name = "espice", .linkage = .static, .root_module = c_api_mod, .use_llvm = exe.use_llvm, .use_lld = exe.use_lld });
    // Installed as files, not with `installArtifact`, so that a dependent's
    // `artifact("espice")` names the CLI alone.
    b.getInstallStep().dependOn(&b.addInstallLibFile(c_api_lib.getEmittedBin(), c_api_lib.out_filename).step);
    b.getInstallStep().dependOn(&b.addInstallHeaderFile(b.path("include/espice.h"), "espice.h").step);
    c_api_lib.installHeader(b.path("include/espice.h"), "espice.h");
    b.addNamedLazyPath("libespice", c_api_lib.getEmittedBin());

    const c_example_mod = b.createModule(.{ .target = target, .optimize = optimize, .link_libc = true });
    c_example_mod.addCSourceFile(.{ .file = b.path("docs/embed/host.c") });
    c_example_mod.addIncludePath(b.path("include"));
    c_example_mod.linkLibrary(c_api_lib);
    const c_example = b.addExecutable(.{ .name = "c-example", .root_module = c_example_mod });
    b.step("c-example", "Build and run the C API example from the docs").dependOn(&b.addRunArtifact(c_example).step);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    run_cmd.addPassthruArgs();
    b.step("run", "Run ESPice").dependOn(&run_cmd.step);

    const run_vera = b.addRunArtifact(vera_exe);
    run_vera.addPassthruArgs();
    b.step("vera", "Run the vera CLI (any .va/.v/.sv/.vhd)").dependOn(&run_vera.step);

    const test_step = b.step("test", "Run every suite and the numeric SPICE fixtures");
    const t: HostTest = .{
        .b = b,
        .exe = exe,
        .objs = g.objs,
        .check = b.step("check", "Type-check every test root without emitting binaries (use with --watch -fincremental)"),
        .filters = if (b.option([]const u8, "test-filter", "Run unit tests whose names contain this text")) |filter| &.{filter} else &.{},
    };

    const c_api_test_mod = M.make(b.path("src/tests/c_api.zig"), &.{.{ .name = "espice_h", .module = espice_h }});
    c_api_test_mod.link_libc = true;
    c_api_test_mod.linkLibrary(c_api_lib);
    const run_c_api_tests = t.run(c_api_test_mod, &.{}, false);
    run_c_api_tests.setEnvironmentVariable("ZIG", b.graph.zig_exe);
    const run_c_api_unit_tests = t.run(c_api_mod, &.{}, false);
    const run_espice_tests = t.run(M.make(b.path("src/tests/espice.zig"), &.{ .{ .name = "espice", .module = g.espice }, .{ .name = "core", .module = g.core }, .{ .name = "analysis", .module = g.analysis } }), &.{}, true);
    run_espice_tests.setEnvironmentVariable("ZIG", b.graph.zig_exe);

    const limiter_gen = b.addRunArtifact(vera_exe);
    limiter_gen.addArgs(&.{ "--emit-zig", "--allow=W0650", "-o" });
    const limiter_source = limiter_gen.addOutputFileArg2("va_limit_state.zig", .{});
    limiter_gen.addFileArg(b.path("tests/conformance/tests/netlist/hdl/veriloga_limit.assets/va_limit_state.va"));

    // A separate object on purpose: error ordinals differ between compilations.
    const error_object_mod = M.make(b.path("src/device/tests/device_errors_object.zig"), &.{
        .{ .name = "device_eval", .module = g.eval },
        .{ .name = "device_abi", .module = g.abi },
        contract,
    });
    error_object_mod.link_libc = true;
    const error_tests_mod = M.make(b.path("src/device/tests/device_errors.zig"), &.{.{ .name = "device_abi", .module = g.abi }});
    error_tests_mod.addObject(b.addObject(.{ .name = "device_errors", .root_module = error_object_mod, .use_llvm = use_llvm }));

    const wfit_mod = M.make(b.path("src/frontend/wfit.zig"), &.{ .{ .name = "core", .module = g.core }, .{ .name = "netlist", .module = g.netlist } });
    const run_grammar = t.run(M.make(b.path("tests/grammar/root.zig"), &.{
        .{ .name = "netlist", .module = g.netlist },
        .{ .name = "wfit", .module = wfit_mod },
    }), &.{}, false);
    run_grammar.setCwd(b.path("."));

    for ([_]struct { []const u8, []const u8, []const *Build.Step.Run }{
        .{ "test-espice", "Run Problem facade, analysis contract, C ABI and CLI tests", &.{ run_espice_tests, run_c_api_tests, run_c_api_unit_tests, t.run(exe.root_module, &.{}, false) } },
        .{ "test-c-api", "Run C ABI boundary tests", &.{ run_c_api_tests, run_c_api_unit_tests } },
        .{
            "test-frontend", "Run netlist, builder and prepared-circuit tests",
            &.{
                t.run(g.netlist, &.{}, false),
                t.run(wfit_mod, &.{}, false),
                t.run(g.frontend, &.{.{ .name = "build_options", .module = bo }}, true),
                // spice.zig and builder.zig's own tests live only in the builder module.
                t.run(g.builder, &.{}, true),
            },
        },
        .{ "test-analysis", "Run all analysis tests", &.{t.run(g.analysis, &.{
            .{ .name = "builder", .module = g.builder },
            .{ .name = "limiter_device", .module = M.make(limiter_source, &.{contract}) },
            contract,
            .{ .name = "models", .module = g.models },
            .{ .name = "device_eval", .module = g.eval },
        }, true)} },
        .{ "test-core", "Run shared data and numerics tests", &.{t.run(g.core, &.{}, false)} },
        .{ "test-solver", "Run solver tests", &.{t.run(g.solver, &.{}, false)} },
        .{ "test-output", "Run waveform writer tests", &.{ t.run(g.output, &.{}, false), t.run(g.measure, &.{}, false) } },
        .{ "test-device", "Run device catalog, evaluator and ABI tests", &.{
            t.run(g.device, &.{}, true),
            t.run(g.abi, &.{}, false),
            t.run(M.make(b.path("src/device/tests/eval.zig"), &.{
                .{ .name = "device_eval", .module = g.eval },
                contract,
                .{ .name = "models", .module = g.models },
            }), &.{}, true),
            t.run(error_tests_mod, &.{}, false),
            t.run(M.make(b.path("models/ltra_ac.zig"), &.{}), &.{}, false),
        } },
        .{ "test-grammar", "Check every input grammar against the frontend", &.{run_grammar} },
    }) |s| {
        const step = b.step(s[0], s[1]);
        for (s[2]) |r| step.dependOn(&r.step);
        test_step.dependOn(step);
    }

    // The runners use threads for deadlines, whatever -Dsingle-threaded says.
    var MH = M;
    MH.single_threaded = null;
    const vpi_fixture = b.addLibrary(.{
        .name = "vpi_gain",
        .linkage = .dynamic,
        .root_module = b.createModule(.{ .target = target, .optimize = optimize, .link_libc = true }),
    });
    vpi_fixture.root_module.addCSourceFile(.{ .file = b.path("tests/conformance/tests/netlist/hdl/veriloga_vpi_app.assets/vpi_gain.c"), .flags = &.{ "-std=c11", "-Wall", "-Werror" } });
    vpi_fixture.root_module.addIncludePath(vera.path("src/vpi"));
    vpi_fixture.linker_allow_shlib_undefined = true;
    const correctness = b.addExecutable(.{
        .name = "test-correctness",
        .root_module = MH.make(b.path("tests/test_correctness.zig"), &.{.{ .name = "fixture_catalog", .module = @import("tests/fixture_catalog.zig").create(b) }}),
    });
    const run_harness_tests = t.run(correctness.root_module, &.{}, false);
    run_harness_tests.setCwd(b.path("."));

    const run_correctness = corpus(b, correctness, exe, vpi_fixture, &.{});
    run_correctness.step.dependOn(&run_harness_tests.step);
    test_step.dependOn(&run_correctness.step);

    const run_metamorphic = corpus(b, correctness, exe, vpi_fixture, &.{"--metamorphic"});
    run_metamorphic.step.dependOn(&run_harness_tests.step);
    b.step("test-metamorphic", "Check that metamorphic rewrites of the fixture decks keep their results").dependOn(&run_metamorphic.step);
    test_step.dependOn(&run_metamorphic.step);

    // One deck at a time: concurrent CUDA contexts time-slice one GPU.
    const run_gpu_corpus = corpus(b, correctness, exe, vpi_fixture, &.{ "--backend", "cuda", "--jobs", "1" });
    b.step("test-gpu", "Run the numeric SPICE fixtures with --backend cuda").dependOn(&run_gpu_corpus.step);

    {
        var P = M;
        P.target = b.resolveTargetQuery(.{ .cpu_arch = .x86_64, .os_tag = .linux, .cpu_model = .{ .explicit = &std.Target.x86.cpu.x86_64_v3 } });
        P.optimize = .fast;
        P.strip = false;
        P.single_threaded = null;
        const pg = graph(P, b.dependency("vera", .{ .target = P.target, .optimize = P.optimize }), gompute.module("gompute"), models, gen, bo);
        const core: Import = .{ .name = "core", .module = pg.core };
        const probe = b.addObject(.{ .name = "codegen-probe", .root_module = P.make(b.path("tests/codegen.zig"), &.{
            .{ .name = "device_eval", .module = pg.eval },
            .{ .name = "solver", .module = pg.solver },
            .{ .name = "integrator", .module = P.make(b.path("src/analysis/tran/integrator.zig"), &.{core}) },
            core,
        }) });
        const residue = b.addSystemCommand(&.{"python3"});
        residue.addFileArg(b.path("tools/asm_probe.py"));
        residue.addFileArg(probe.getEmittedAsm());
        residue.addArgs(&.{ "gen_dual_mul=hand_dual_mul", "gen_engine_solve=hand_engine_solve", "gen_hook_trap_assemble=hand_hook_trap_assemble", "gen_hook_step_assemble=hand_hook_step_assemble", "gen_hook_diag_at=hand_hook_diag_at", "gen_num_dot=hand_num_dot", "gen_num_sum=hand_num_sum", "gen_num_norminf=hand_num_norminf", "gen_lanes_splat=hand_lanes_splat", "gen_lanes_any=hand_lanes_any", "gen_lanes_all=hand_lanes_all", "gen_lanes_select=hand_lanes_select", "gen_lanes_cmp=hand_lanes_cmp", "gen_lanes_either=hand_lanes_either", "gen_lanes_lane=hand_lanes_lane", "gen_lanes_each=hand_lanes_each", "gen_newton=hand_newton", "gen_update_and_norm=hand_update_and_norm" });
        b.step("test-codegen", "Judge the residue probes in tests/codegen.zig").dependOn(&residue.step);
        test_step.dependOn(&residue.step);
    }

    const bench_runner = b.addExecutable(.{ .name = "bench-runner", .root_module = MH.make(b.path("tests/benchmark/runner.zig"), &.{}) });
    b.step("build-benchmark", "Build the standalone simulator comparison runner").dependOn(&b.addInstallArtifact(bench_runner, .{}).step);
    b.step("bench", "Compare ESPice with ngspice, VACASK and optional --xyce BIN (nix develop .#benchmarking)").dependOn(&corpus(b, bench_runner, exe, null, &.{"tests/conformance/tests/netlist"}).step);

    const gen_postlayout = b.addSystemCommand(&.{ "python3", "tests/conformance/external/spice-bench/postlayout/gen.py", "zig-out/postlayout" });
    gen_postlayout.setCwd(b.path("."));
    const run_postlayout = corpus(b, bench_runner, exe, null, &.{ "zig-out/postlayout", "--out", "zig-out/postlayout-results.md" });
    run_postlayout.step.dependOn(&gen_postlayout.step);
    b.step("bench-postlayout", "Time ESPice on synthetic post-layout decks against ngspice and VACASK").dependOn(&run_postlayout.step);

    const fetch_suites = b.addSystemCommand(&.{ "bash", "tests/conformance/external/spice-bench/fetch.sh", "zig-out/suites" });
    if (suite) |name| fetch_suites.addArg(name);
    fetch_suites.setCwd(b.path("."));
    fetch_suites.has_side_effects = true;
    const suites_dir = if (suite) |name| b.fmt("zig-out/suites/{s}", .{name}) else "zig-out/suites";
    const run_suites = corpus(b, bench_runner, exe, null, &.{ suites_dir, "--out", "zig-out/suites-results.md" });
    run_suites.step.dependOn(&fetch_suites.step);
    b.step("bench-suites", "Fetch the external SPICE suites and time ESPice on them against ngspice and VACASK").dependOn(&run_suites.step);

    // The conformance suite in one step: the oracles first, then every deck
    // timed and compared against each simulator installed (a missing one
    // drops out), espice on the CPU and, in a GPU build, on CUDA. Serial
    // after the oracles, so the timings run on a quiet machine.
    const conformance_args: []const []const u8 = if (gpu_kernels)
        &.{ "tests/conformance/tests/netlist", "--out", "zig-out/conformance-results.md", "--espice", "espice-cpu=--backend=cpu", "--espice", "espice-gpu=--backend=cuda", "--ngspice", "auto", "--vacask", "auto", "--xyce", "auto", "--hspice", "auto", "--spectre", "auto" }
    else
        &.{ "tests/conformance/tests/netlist", "--out", "zig-out/conformance-results.md", "--espice", "espice-cpu=--backend=cpu", "--ngspice", "auto", "--vacask", "auto", "--xyce", "auto", "--hspice", "auto", "--spectre", "auto" };
    const run_conformance_bench = corpus(b, bench_runner, exe, null, conformance_args);
    run_conformance_bench.step.dependOn(&run_correctness.step);
    b.step("test-conformance", "The conformance suite: oracles, then a benchmark against every simulator on PATH (Xyce, VACASK, Spectre, ngspice, HSPICE) and espice CPU/GPU").dependOn(&run_conformance_bench.step);

    const run_bench_tests = t.run(bench_runner.root_module, &.{}, false);
    b.step("test-benchmark", "Test reference adapters and benchmark comparison").dependOn(&run_bench_tests.step);
    test_step.dependOn(&run_bench_tests.step);

    {
        const wasm_step = b.step("wasm", "Build the docs playground (zig-out/wasm/espice.wasm, cktimg.wasm)");
        var W = M;
        W.target = b.resolveTargetQuery(.{ .cpu_arch = .wasm32, .os_tag = .wasi, .cpu_features_add = std.Target.wasm.featureSet(&.{.simd128}) });
        W.optimize = .small;
        W.strip = true;
        W.single_threaded = null;
        W.use_llvm = null;
        const wgompute = b.dependency("gompute", .{ .target = W.target, .optimize = W.optimize }).module("gompute");
        const wg = graph(W, b.dependency("vera", .{ .target = W.target, .optimize = W.optimize }), wgompute, models, gen, bo);
        wg.analysis.addImport("gompute_kernels", wgompute.import_table.get("gompute_kernels").?);
        const wh: Import = .{ .name = "espice_h", .module = b.addTranslateC(.{ .root_source_file = b.path("include/espice.h"), .target = W.target, .optimize = W.optimize, .link_libc = false }).createModule() };
        const wroot = W.make(b.path("tools/playground/espice.zig"), &.{ wh, .{ .name = "c_api", .module = W.make(b.path("src/c_api.zig"), &.{ .{ .name = "espice", .module = wg.espice }, wh }) } });
        for (wg.objs) |o| wroot.addObject(o);
        wroot.single_threaded = true;
        const wexe = b.addExecutable(.{ .name = "espice", .root_module = wroot });
        wexe.entry = .disabled;
        wexe.rdynamic = true;
        wexe.wasi_exec_model = .reactor;
        wexe.stack_size = 16 << 20; // the corpus peaks at 541 KiB (dc/device_b3soipd_output.sp)
        wasm_step.dependOn(&b.addInstallArtifact(wexe, .{ .dest_dir = .{ .override = .{ .custom = "wasm" } } }).step);

        // Build with `-Dcpu=x86_64 -Dhip-arch=none -Dcuda-arch=none` to match simd128's lane count.
        const cmp = b.addSystemCommand(&.{"node"});
        cmp.addFileArg(b.path("tools/playground/compare.js"));
        cmp.addDirectoryArg(wexe.getEmittedBin().dirname());
        cmp.addArtifactArg(exe);
        cmp.addDirectoryArg(b.path("tests/conformance/tests/netlist"));
        cmp.addFileInput(b.path("docs/javascripts/playground.js"));
        b.step("wasm-compare", "Check espice.wasm against the native espice over the corpus (node)").dependOn(&cmp.step);

        const ck = b.addSystemCommand(&.{ b.graph.zig_exe, "build", "--prefix" });
        const ck_out = ck.addOutputDirectoryArg("playground");
        ck.setCwd(b.path("tools/playground"));
        for ([_][]const u8{ "build.zig", "build.zig.zon", "cktimg.zig" }) |f| ck.addFileInput(b.path(b.fmt("tools/playground/{s}", .{f})));
        wasm_step.dependOn(&b.addInstallFileWithDir(ck_out.path(b, "wasm/cktimg.wasm"), .{ .custom = "wasm" }, "cktimg.wasm").step);
    }

    checkModuleGraph(b, exe.root_module);
    checkModuleGraph(b, c_api_mod);
}

const Cfg = struct {
    b: *Build,
    target: Build.ResolvedTarget,
    optimize: std.lang.Optimize,
    strip: bool,
    single_threaded: ?bool,
    use_llvm: ?bool,
    /// `-Dgpu`: device kernels are compiled in, so host device objects build
    /// their Model rows in the kernels' layout (no Debug `su_ok`).
    gpu_kernels: bool = false,

    fn make(self: Cfg, root: Build.LazyPath, imports: []const Import) *Module {
        const mod = self.b.createModule(.{
            .root_source_file = root,
            .target = self.target,
            .optimize = self.optimize,
            .strip = self.strip,
            .single_threaded = self.single_threaded,
            .imports = imports,
        });
        return mod;
    }
};

const Graph = struct {
    core: *Module,
    abi: *Module,
    eval: *Module,
    models: *Module,
    /// Target-less twins of the device models, `contract` and `abi` for the GPU kernel
    /// roots: each kernel compile inherits its root's GPU target. A pinned -Dtarget on the
    /// host modules would compile them as host code inside the amdgcn/nvptx object.
    kernel_models: []*Module,
    kernel_contract: Import,
    kernel_abi: Import,
    objs: []*Compile,
    device: *Module,
    solver: *Module,
    analysis: *Module,
    netlist: *Module,
    builder: *Module,
    frontend: *Module,
    output: *Module,
    measure: *Module,
    espice: *Module,
};

/// The simulator's module graph for one target. `analysis` still needs `gompute_kernels`.
fn graph(c: Cfg, vera: *Build.Dependency, gompute: *Module, models: []const Model, gen: []const Build.LazyPath, bo: *Module) Graph {
    const b = c.b;
    // Device code: always stripped (DWARF on generated models crashed NVPTX).
    var dev = c;
    dev.strip = true;
    const contract: Import = .{ .name = "contract", .module = vera.module("contract") };
    const gomp: Import = .{ .name = "gompute", .module = gompute };
    // emitKernels mutates the public Gompute module to import all GPU blobs.
    // Standalone CPU objects only use its math and host no-ops. Keep their
    // import graph free of artifacts so changing any kernel, GPU architecture
    // or lane option cannot invalidate every host model (or wait on ptxas).
    const object_gomp: Import = .{ .name = "gompute", .module = b.createModule(.{
        .root_source_file = gompute.root_source_file,
        .target = gompute.resolved_target,
        .optimize = gompute.optimize,
        .strip = gompute.strip,
        .single_threaded = gompute.single_threaded,
    }) };
    const build_options: Import = .{ .name = "build_options", .module = bo };
    const core: Import = .{ .name = "core", .module = c.make(b.path("src/core/root.zig"), &.{}) };
    const abi: Import = .{ .name = "device_abi", .module = c.make(b.path("src/device/abi.zig"), &.{ contract, core }) };
    // Device objects see only core's ids leaf (GROUND, DeviceType): an edit elsewhere in core
    // must not recompile every model. abi's own tests still get the full core.
    const abi_dev = dev.make(b.path("src/device/abi.zig"), &.{ contract, .{ .name = "core", .module = dev.make(b.path("src/core/ids.zig"), &.{}) } });
    // Zig 0.18's std.Io.Threaded does not compile for wasm32-wasi with libc
    // (Environ and PATH_MAX); without it, it takes the WASI syscall path.
    const libc = !c.target.result.cpu.arch.isWasm();
    const eval = c.make(b.path("src/device/eval.zig"), &.{ contract, gomp, abi });
    eval.link_libc = libc;

    var agg: std.ArrayList(u8) = .empty;
    const dev_mods = b.allocator.alloc(*Module, models.len) catch @panic("OOM");
    const kernel_models = b.allocator.alloc(*Module, models.len) catch @panic("OOM");
    const kmake = struct {
        fn f(owner: *Build, root: Build.LazyPath, imports: []const Import) *Module {
            return owner.createModule(.{ .root_source_file = root, .strip = true, .imports = imports });
        }
    }.f;
    const k_contract: Import = .{ .name = "contract", .module = kmake(b, vera.module("contract").root_source_file.?, &.{}) };
    const k_abi: Import = .{ .name = "device_abi", .module = kmake(b, b.path("src/device/abi.zig"), &.{ k_contract, .{ .name = "core", .module = kmake(b, b.path("src/core/ids.zig"), &.{}) } }) };
    const objs = b.allocator.alloc(*Compile, models.len) catch @panic("OOM");
    for (models, gen, dev_mods, kernel_models, objs) |m, src, *dm, *kone, *obj| {
        const line = b.fmt("pub const {s} = @import(\"{s}\");\n", .{ m.name, m.name });
        agg.appendSlice(b.allocator, line) catch @panic("OOM");
        dm.* = dev.make(src, &.{contract});
        // The device kernels compile at .fast, where VerA's Debug-only Model
        // field (`su_ok`) does not exist; `gpu_kernels` makes the host's
        // device object drop it too, so a Model row has one size on both sides.
        // Each wrapper needs its own cache entry. A shared WriteFiles step
        // changes every wrapper's path when just one model/option changes.
        // Outside Debug the contract checks are already off, so CPU and GPU
        // builds can use the exact same host object and wrapper.
        // ltra also gets models/ltra_ac.zig, its exact AC term (eval.zig AcTerms).
        const ltra = std.mem.eql(u8, m.name, "ltra");
        const one_src = b.addWriteFiles().add(b.fmt("{s}/models.zig", .{m.name}), b.fmt("{s}{s}{s}", .{
            line,
            if (c.gpu_kernels or c.optimize != .debug) "pub const gpu_kernels = true;\n" else "",
            if (ltra) "pub const ltra_ac = @import(\"ltra_ac\");\n" else "",
        }));
        const one = dev.make(one_src, &.{.{ .name = m.name, .module = dm.* }});
        kone.* = kmake(b, one_src, &.{.{ .name = m.name, .module = kmake(b, src, &.{k_contract}) }});
        if (ltra) {
            one.addImport("ltra_ac", dev.make(b.path("models/ltra_ac.zig"), &.{}));
            kone.*.addImport("ltra_ac", kmake(b, b.path("models/ltra_ac.zig"), &.{}));
        }
        const host = dev.make(b.path("src/device/eval.zig"), &.{ contract, .{ .name = "models", .module = one }, .{ .name = "device_abi", .module = abi_dev }, object_gomp });
        host.link_libc = libc;
        // The callconv(.auto) vtable gains a hidden *StackTrace when tracing is on.
        host.error_tracing = c.optimize == .debug;
        obj.* = b.addObject(.{ .name = b.fmt("dev_{s}", .{m.name}), .root_module = host, .use_llvm = c.use_llvm });
    }
    const models_mod = c.make(b.addWriteFiles().add("models.zig", agg.items), &.{});
    for (models, dev_mods) |m, dm| models_mod.addImport(m.name, dm);

    const device: Import = .{ .name = "device", .module = c.make(b.path("src/device/root.zig"), &.{
        .{ .name = "models", .module = models_mod },
        abi,
        core,
        .{ .name = "fastvaf", .module = vera.module("vera") },
        .{ .name = "vera_sim", .module = vera.module("sim") },
        .{ .name = "vera_digital", .module = vera.module("digital") },
        build_options,
    }) };
    device.module.link_libc = libc;
    const solver = c.make(b.path("src/solver/root.zig"), &.{ core, build_options });
    // `.meas` evaluation: output prints it, a transient watches its cards resolve.
    const measure: Import = .{ .name = "measure", .module = c.make(b.path("src/output/measure.zig"), &.{core}) };
    const analysis = c.make(b.path("src/analysis/root.zig"), &.{ .{ .name = "solver", .module = solver }, core, device, gomp, build_options, measure });
    analysis.link_libc = libc;
    const netlist: Import = .{ .name = "netlist", .module = c.make(b.path("src/frontend/netlist.zig"), &.{core}) };
    const builder: Import = .{ .name = "builder", .module = c.make(b.path("src/frontend/builder.zig"), &.{ core, device, netlist }) };
    const frontend = c.make(b.path("src/frontend/prepare.zig"), &.{ core, device, netlist, builder });
    const output = c.make(b.path("src/output/root.zig"), &.{ core, measure });
    return .{
        .core = core.module,
        .abi = abi.module,
        .eval = eval,
        .models = models_mod,
        .kernel_models = kernel_models,
        .kernel_contract = k_contract,
        .kernel_abi = k_abi,
        .objs = objs,
        .device = device.module,
        .solver = solver,
        .analysis = analysis,
        .netlist = netlist.module,
        .builder = builder.module,
        .frontend = frontend,
        .output = output,
        .measure = measure.module,
        .espice = c.make(b.path("src/espice.zig"), &.{
            core,
            .{ .name = "analysis", .module = analysis },
            .{ .name = "frontend", .module = frontend },
            .{ .name = "output", .module = output },
        }),
    };
}

fn corpus(b: *Build, runner: *Compile, exe: *Compile, vpi: ?*Compile, args: []const []const u8) *Build.Step.Run {
    const r = b.addRunArtifact(runner);
    r.setCwd(b.path("."));
    r.addArtifactArg2(exe, .{});
    if (vpi) |v| {
        r.addArg("--vpi-fixture");
        r.addArtifactArg(v);
    } else r.stdio = .inherit;
    r.setEnvironmentVariable("ZIG", b.graph.zig_exe);
    // Debug runs the slowest decks (stress/scaling_inverter_chain_4k,
    // tran/ltra_tran_long_run_past_8192) in ~620 s under load, past the
    // runner's 300 s default. A later passthrough --timeout still wins.
    if (exe.root_module.optimize == .debug) r.addArgs(&.{ "--timeout", "1200" });
    r.addArgs(args);
    r.addPassthruArgs();
    return r;
}

/// "A module importing `name` must be rooted under one of `allowed`", checked on shipped graphs.
const restricted_imports = [_]struct { name: []const u8, allowed: []const []const u8 }{
    .{ .name = "solver", .allowed = &.{"src/analysis/"} },
    .{ .name = "analysis", .allowed = &.{"src/espice.zig"} },
    .{ .name = "frontend", .allowed = &.{"src/espice.zig"} },
    .{ .name = "output", .allowed = &.{"src/espice.zig"} },
    .{ .name = "netlist", .allowed = &.{"src/frontend/"} },
    .{ .name = "builder", .allowed = &.{"src/frontend/"} },
    .{ .name = "espice", .allowed = &.{ "src/main.zig", "src/c_api.zig" } },
};

fn checkModuleGraph(b: *Build, root: *Module) void {
    for (root.getGraph().modules) |m| {
        const src = m.root_source_file orelse continue;
        if (src != .src_path or src.src_path.owner != b) continue;
        const from = src.src_path.sub_path;
        for (m.import_table.keys()) |name| for (restricted_imports) |rule| {
            if (!std.mem.eql(u8, rule.name, name)) continue;
            for (rule.allowed) |prefix| {
                if (std.mem.startsWith(u8, from, prefix)) break;
            } else std.debug.panic("module graph: {s} imports {s}, which restricted_imports forbids", .{ from, name });
        };
    }
}

/// One test binary from a fresh copy of `m`: `addObject` mutates its module.
const HostTest = struct {
    b: *Build,
    exe: *Compile,
    objs: []const *Compile,
    check: *Build.Step,
    filters: []const []const u8,

    fn run(self: HostTest, m: *Module, extra: []const Import, devices: bool) *Build.Step.Run {
        self.check.dependOn(&self.b.addTest(.{ .name = "check", .root_module = self.copy(m, extra), .use_llvm = false }).step);
        const mod = self.copy(m, extra);
        mod.link_objects.appendSlice(self.b.allocator, m.link_objects.items) catch @panic("OOM");
        mod.include_dirs.appendSlice(self.b.allocator, m.include_dirs.items) catch @panic("OOM");
        if (devices) for (self.objs) |o| mod.addObject(o);
        return self.b.addRunArtifact(self.b.addTest(.{ .root_module = mod, .filters = self.filters, .use_llvm = self.exe.use_llvm, .use_lld = self.exe.use_lld }));
    }

    fn copy(self: HostTest, m: *Module, extra: []const Import) *Module {
        const mod = self.b.createModule(.{
            .root_source_file = m.root_source_file,
            .target = m.resolved_target,
            .optimize = m.optimize,
            .strip = m.strip,
            .single_threaded = m.single_threaded,
            .link_libc = true,
        });
        var it = m.import_table.iterator();
        while (it.next()) |e| mod.addImport(e.key_ptr.*, e.value_ptr.*);
        for (extra) |e| mod.addImport(e.name, e.module);
        return mod;
    }
};

const Model = struct {
    name: []const u8,
    file: []const u8,
    src: Build.LazyPath,
    module: []const u8,
    std: ?[]const u8 = null,
    inc_dirs: []const Build.LazyPath = &.{},
    hdl: Hdl,
    heavy: bool,
    include: ?[]const u8 = null,
};

const Hdl = enum { verilog_a, digital };

const hdl_by_ext = std.StaticStringMap(Hdl).initComptime(.{
    .{ ".va", .verilog_a },
    .{ ".v", .digital },
    .{ ".sv", .digital },
    .{ ".vhd", .digital },
    .{ ".vhdl", .digital },
});

/// Variants of one models/ body, compiled as `defines` then `include "base"`.
/// A module's ports and output branch are fixed, so each port count or output
/// kind stays its own device; only the source is shared.
const model_variants = [_]struct { name: []const u8, base: []const u8, defines: []const u8 }{
    .{ .name = "bsource_i", .base = "bsource.va", .defines = "`define BSRC_CURRENT\n" },
    .{ .name = "bsource_q", .base = "bsource.va", .defines = "`define BSRC_CHARGE\n" },
    .{ .name = "vccs_laplace", .base = "vcvs_laplace.va", .defines = "`define LAPLACE_CURRENT\n" },
    .{ .name = "vcvs_pole", .base = "vcvs_laplace.va", .defines = "`define LAPLACE_ZP\n" },
    .{ .name = "vccs_pole", .base = "vcvs_laplace.va", .defines = "`define LAPLACE_ZP\n`define LAPLACE_CURRENT\n" },
    .{ .name = "sparam_2", .base = "sparam_1.va", .defines = "`define SP_N 2\n`define SP_2\n" },
    .{ .name = "sparam_3", .base = "sparam_1.va", .defines = "`define SP_N 3\n`define SP_3\n" },
    .{ .name = "sparam_4", .base = "sparam_1.va", .defines = "`define SP_N 4\n`define SP_4\n" },
    .{ .name = "wline_2", .base = "wline_1.va", .defines = "`define W_N 2\n`define W_2\n" },
    .{ .name = "wline_3", .base = "wline_1.va", .defines = "`define W_N 3\n`define W_3\n" },
    .{ .name = "wline_4", .base = "wline_1.va", .defines = "`define W_N 4\n`define W_4\n" },
    .{ .name = "coupled_ltra3", .base = "coupled_ltra.va", .defines = "`define CPL_N 3\n`define CPL_MODULE coupled_ltra3\n" },
    .{ .name = "coupled_ltra4", .base = "coupled_ltra.va", .defines = "`define CPL_N 4\n`define CPL_MODULE coupled_ltra4\n" },
};

/// Third-party model releases, vendored in models/ as one file each (the
/// release's include files inlined; licence files as `<release>_LICENSE.txt`).
/// Their module names and keyword sets differ from the file stems, so they
/// are listed here and skipped by the directory scan.
const upstream_models = [_]struct {
    name: []const u8,
    file: []const u8,
    module: []const u8,
    std: ?[]const u8 = null,
}{
    .{ .name = "psp103", .file = "psp103.va", .module = "PSP103VA" },
    .{ .name = "psp103_nqs", .file = "psp103_nqs.va", .module = "PSPNQS103VA" },
    .{ .name = "psp103t", .file = "psp103t.va", .module = "PSP103TVA" },
    .{ .name = "juncap200", .file = "juncap200.va", .module = "JUNCAP200" },
    .{ .name = "psp104", .file = "psp104.va", .module = "PSP104VA" },
    .{ .name = "psp104_nqs", .file = "psp104_nqs.va", .module = "PSPNQS104VA" },
    .{ .name = "psp104t", .file = "psp104t.va", .module = "PSP104TVA" },
    .{ .name = "bsimimg", .file = "bsimimg.va", .module = "bsimimg" },
    .{ .name = "bsimcmg", .file = "bsimcmg.va", .module = "bsimcmg_va", .std = "VAMS-2.3" },
    .{ .name = "bsimbulk", .file = "bsimbulk.va", .module = "bsimbulk" },
    .{ .name = "bsimsoi_va", .file = "bsimsoi.va", .module = "bsimsoi" },
    .{ .name = "l_utsoi", .file = "L_UTSOI_102.va", .module = "l_utsoi" },
    .{ .name = "hisimhv_va", .file = "hisimhv.va", .module = "hisimhv_va" },
    .{ .name = "hisimsoi", .file = "hisimsoi.va", .module = "hisimsoi_va", .std = "VAMS-2.3" },
    .{ .name = "hisimsotb", .file = "hisimsotb.va", .module = "hisimsotb_va", .std = "VAMS-2.3" },
    .{ .name = "mosvar", .file = "mosvar.va", .module = "mosvar" },
    .{ .name = "hicumL2_va", .file = "hicumL2_v320.va", .module = "hicumL2va" },
    .{ .name = "hicumL0", .file = "hicumL0_v2p1p0.va", .module = "hicumL0va" },
    .{ .name = "mextram", .file = "bjt505.va", .module = "bjt505_va" },
    .{ .name = "mextram_t", .file = "bjt505t.va", .module = "bjt505t_va" },
    .{ .name = "diode_cmc", .file = "diode_cmc.va", .module = "DIODE_CMC" },
    .{ .name = "r2_cmc", .file = "r2_cmc.va", .module = "r2_cmc" },
    .{ .name = "r3_cmc", .file = "r3_cmc.va", .module = "r3_cmc" },
    .{ .name = "asmhemt", .file = "asmhemt.va", .module = "asmhemt" },
    .{ .name = "mvsg", .file = "mvsg_cmc_5.0.0.va", .module = "mvsg_cmc" },
    .{ .name = "asmesd", .file = "asmesd.va", .module = "asmesd" },
    .{ .name = "asmesd_dio", .file = "asmesd_dio.va", .module = "asmesd_dio" },
};

/// Whether the models/ scan skips `file`: an upstream release, built from
/// `upstream_models` instead.
fn upstreamFile(file: []const u8) bool {
    for (upstream_models) |u| if (std.mem.eql(u8, u.file, file)) return true;
    return false;
}

/// Source size from which a model's GPU compile goes on the serial heavy lanes.
const heavy_model_bytes: u64 = 20 * 1024;

/// GPU kernel stack frames past the 262,136 B limit (CUDA's and amdgcn's).
/// Empty since VerA v0.11.0's sparse layout: the largest is mvsg at 47 KB on
/// sm_89 and asmhemt at 113 KB on gfx1100 (dev/research/gpu-lane-e1.md §5).
const gpu_host_only = std.StaticStringMap(void).initComptime(.{});

/// Upstream sources with a non-LRM string escape (W0149) we cannot edit.
const escape_models = std.StaticStringMap(void).initComptime(.{ .{"hisim2_va"}, .{"hisimsotb"} });

fn discoverModels(b: *Build) []const Model {
    const io = b.graph.io;
    var out: std.ArrayList(Model) = .empty;
    b.dependOnDirectoryContents(b.path("models"));
    var dir = b.root.openDir(io, "models", .{ .iterate = true }) catch @panic("models/ missing");
    defer dir.close(io);
    var it = dir.iterate();
    while (it.next(io) catch @panic("models/ iterate failed")) |e| {
        if (e.kind != .file) continue;
        const hdl = hdl_by_ext.get(std.fs.path.extension(e.name)) orelse continue;
        if (upstreamFile(e.name)) continue;
        const name = std.fs.path.stem(e.name);
        b.dependOnFileMetadata(b.path(b.fmt("models/{s}", .{e.name})));
        const size = (dir.statFile(io, e.name, .{}) catch @panic("models/ stat failed")).size;
        out.append(b.allocator, .{
            .name = b.dupe(name),
            .file = b.dupe(e.name),
            .src = b.path(b.fmt("models/{s}", .{e.name})),
            .module = b.dupe(name),
            .hdl = hdl,
            .heavy = size >= heavy_model_bytes,
        }) catch @panic("OOM");
    }
    for (model_variants) |v| {
        const size = (dir.statFile(io, v.base, .{}) catch @panic("models/ stat failed")).size;
        out.append(b.allocator, .{
            .name = v.name,
            .file = v.base,
            .src = b.addWriteFiles().add(b.fmt("{s}.va", .{v.name}), b.fmt("{s}`include \"{s}\"\n", .{ v.defines, v.base })),
            .module = v.name,
            .hdl = .verilog_a,
            .heavy = size >= heavy_model_bytes,
            .include = v.base,
        }) catch @panic("OOM");
    }
    for (upstream_models) |u| {
        const inc_dirs = b.allocator.dupe(Build.LazyPath, &.{b.path("models")}) catch @panic("OOM");
        out.append(b.allocator, .{
            .name = u.name,
            .file = u.file,
            .src = b.path(b.fmt("models/{s}", .{u.file})),
            .module = u.module,
            .std = u.std,
            .inc_dirs = inc_dirs,
            .hdl = .verilog_a,
            .heavy = true, // every release is well past heavy_model_bytes
        }) catch @panic("OOM");
    }
    std.mem.sort(Model, out.items, {}, struct {
        fn lt(_: void, a: Model, c: Model) bool {
            return std.mem.lessThan(u8, a.name, c.name);
        }
    }.lt);
    for (out.items[1..], out.items[0 .. out.items.len - 1]) |cur, prev| {
        if (std.mem.eql(u8, cur.name, prev.name))
            std.debug.panic("models/{s} and models/{s} share the stem '{s}'", .{ prev.file, cur.file, cur.name });
    }
    return out.items;
}

fn gpuArch(arch: []const u8) gompute_build.CudaOptions {
    if (std.mem.eql(u8, arch, "none")) return .{ .enabled = false };
    if (std.mem.eql(u8, arch, "auto")) return .{};
    return .{ .gpu = .{ .name = arch } };
}

fn deviceKernelImports(_: *Build, _: Build.ResolvedTarget, _: std.lang.Optimize, ctx: ?*anyopaque) []const Import {
    const imports: [*]const Import = @ptrCast(@alignCast(ctx.?));
    return imports[0..4];
}

fn laneKernelImports(_: *Build, _: Build.ResolvedTarget, _: std.lang.Optimize, ctx: ?*anyopaque) []const Import {
    const imports: [*]const Import = @ptrCast(@alignCast(ctx.?));
    return imports[0..1];
}

/// Absolute `sub_path` under `owner`'s package root, resolved against `b`'s.
/// The VerA commit `espice --version` names: the one build.zig.zon pins,
/// or for a `--fork=<VerA>` checkout its `git describe` ("-dirty" with
/// uncommitted edits), or the fork's path when it is no git checkout.
fn veraCommit(b: *Build, vera: *Build) []const u8 {
    const root = pathFromRoot(b, vera, "");
    if (std.mem.eql(u8, std.fs.path.basename(root), vera.pkg_hash)) {
        const url = @import("build.zig.zon").dependencies.vera.url;
        return url[std.mem.lastIndexOfScalar(u8, url, '#').? + 1 ..];
    }
    // HEAD and the working tree are configure inputs Zig cannot track.
    b.graph.poisonCache();
    // Not a checkout of its own: git would describe an enclosing repository.
    std.Io.Dir.cwd().access(b.graph.io, pathFromRoot(b, vera, ".git"), .{}) catch return root;
    var code: u8 = undefined;
    const out = b.runAllowFail(&.{ "git", "-C", root, "describe", "--always", "--dirty", "--abbrev=40", "--exclude=*" }, &code, .ignore) catch return root;
    return std.mem.trimEnd(u8, out, "\n");
}

fn pathFromRoot(b: *Build, owner: *Build, sub_path: []const u8) []u8 {
    const top = b.root.root_dir.path orelse ".";
    const dep = owner.root.root_dir.path orelse ".";
    return b.pathResolve(&.{ top, b.root.sub_path, dep, owner.root.sub_path, sub_path });
}
