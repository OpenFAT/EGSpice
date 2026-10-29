//! The `espice` command line: argv in, one `Problem` per netlist file run
//! in order, its `.meas` text on stdout and a one-line summary per result on
//! stderr. Exit 0, 1 if any file failed to read, prepare or run, 2 on a
//! usage error. The flags, the exit codes and every printed line are the
//! CLI's contract.
//! Cites: none.
//!
//! Contents:
//! - `pub const std_options: std.Options`
//! - `pub const vera_validate_contract: bool`
//! - `pub fn main(init: std.process.Init) !u8`

const std = @import("std");
const builtin = @import("builtin");
const problem = @import("espice");
const vpi_host = @import("vpi_host.zig");

/// std reserves a 256 KB alternate signal stack per thread for stack-overflow
/// traces, as zero-filled TLS: every thread of every compilation unit pays it
/// (1.5 MB per thread across the device objects before this). Kept in Debug,
/// where the trace is worth it.
pub const std_options: std.Options = .{ .signal_stack_size = if (builtin.mode == .debug) 1 << 18 else null };

/// VerA's device/host contract checks (`contract.validating`), on in Debug
/// only: off, a device build spends 0.4-1.8% fewer instructions.
pub const vera_validate_contract = builtin.mode == .debug;

/// `--version`; a test pins it to build.zig.zon's `.version`.
const version = "0.3.0";

/// The backends compiled in, for `--version`; a GPU one also needs its
/// device at run time.
const backends = blk: {
    const k = @import("gompute_kernels");
    break :blk "cpu" ++ (if (k.has_cuda) ", cuda" else "") ++ (if (k.has_hip) ", hip" else "");
};

const help =
    \\Usage: espice [OPTION]... FILE...
    \\  -b, --batch                 Run all queries (default)
    \\  -r, --rawfile=FILE          Result destination
    \\      --format=FMT            binary|ascii|csv|touchstone|psf|fsdb|sst2|citi|print
    \\      --backend=BE            cpu|auto|cuda|hip (default auto with GPU kernels, else cpu)
    \\      --gpu                   Require an available GPU
    \\      --jobs=N|auto           Maximum parallel queries (default auto: one per CPU)
    \\      --print-dag             Print the next query frontier before running
    \\      --plan                  Print the DAG without running analyses
    \\      --timing-in-depth       Time preparation, queries, and individual steps
    \\      --measures-only         Measure .tran cards as the run goes; keep only the columns whole-run cards read
    \\                              (--measures-only, --early-stop and --warm-start are on by default when
    \\                              nothing reads the waveforms: no -r, .save, .stim, optimization, --tangents
    \\                              or Verilog-A display task)
    \\      --full-waveforms        Turn that default off
    \\      --early-stop            End each .tran once its .meas cards have resolved (never with LAST, _CONT,
    \\                              JITTER, .check/DOUT or a window without TO=)
    \\      --abort-after=T         Fail a .tran still missing a .meas value past T seconds
    \\      --abort-measure=NAME    Bound only .meas NAME with --abort-after
    \\      --warm-start            Start each operating point (a transient's, an AC or DC sweep's, an .op) from the
    \\                              nominal one: the same point within Newton's tolerances, faster on .step sweeps
    \\      --tangents=P,...        Print d(.meas)/d(P) of the first .tran's cards for these .params
    \\      --tangent-step=S        Relative step of --tangents (default 1e-6)
    \\      --lanes                 Solve sweep points' OPs and .dc sweeps as SIMD lanes (CPU)
    \\      --tokenizer=FMT         ngspice|hspice|spectre (default: detected from the deck, else ngspice)
    \\      --vpi=APP               Load a VPI application (.so) for Verilog-A $systf calls; repeatable
    \\  -I, --include=DIR           Search DIR for a .hdl model's `include files, after the model's own
    \\                              directory; repeatable, in order
    \\  -v, --version               Version, the VerA version and commit it was built with, and its backends
    \\
;

/// What `parse` decided: run the decks, print help or the version, or fail
/// as a usage error.
const Parsed = enum { run, help, version, usage };

/// Whether a deck's DAG is printed, and whether it then runs.
const Plan = enum { run, print_then_run, print_only };

/// Every flag's value, as parsed. The lists borrow argv.
const Cli = struct {
    paths: std.ArrayList([]const u8) = .empty,
    vpi_apps: std.ArrayList([]const u8) = .empty,
    include_dirs: std.ArrayList([]const u8) = .empty,
    tangents: std.ArrayList([]const u8) = .empty,
    dialect: ?problem.Dialect = null,
    selection: problem.Selection = .{},
    /// Null when no `--backend` or `--gpu` was given.
    backend: ?problem.Request = null,
    explicit_gpu: bool = false,
    lanes: bool = false,
    /// 0 is `--jobs=auto`, resolved by `main`.
    max_parallel: u16 = 0,
    full_waveforms: bool = false,
    plan: Plan = .run,
    timing_in_depth: bool = false,
    point: problem.requests.PointOptions = .{},

    fn deinit(cli: *Cli, gpa: std.mem.Allocator) void {
        cli.paths.deinit(gpa);
        cli.vpi_apps.deinit(gpa);
        cli.include_dirs.deinit(gpa);
        cli.tangents.deinit(gpa);
    }
};

/// glibc's `mallopt` (malloc.h); M_MMAP_THRESHOLD is -3.
extern "c" fn mallopt(param: c_int, value: c_int) c_int;

/// Runs every netlist named on the command line and returns the exit code:
/// 0, 1 if any file failed to read, prepare or run, 2 on a usage error.
/// Fails only when stdout or stderr cannot be written, or out of memory
/// while reading argv.
pub fn main(init: std.process.Init) !u8 {
    // glibc raises its mmap threshold to each mmapped block it frees (up to
    // 32 MB), so after one query frees a big buffer the next ones come from
    // the heap and stay resident when freed. A fixed 128 KB threshold (its
    // default) hands every big block back at once: peak RSS
    // scaling_rc_ladder_100k 285 -> 277 MB, opamp with two .ac 180 -> 170 MB,
    // for 4% more minor page faults on the ladder.
    if (builtin.target.isGnuLibC()) _ = mallopt(-3, 128 * 1024);
    var cli: Cli = .{};
    defer cli.deinit(init.gpa);
    var args = init.minimal.args.iterate();
    _ = args.skip();
    switch (try parse(init.gpa, &args, &cli)) {
        .run => {},
        .help => {
            // Help and version go to stdout, so `espice --version | ...` and
            // `espice --help | less` see them.
            var out: std.Io.File.Writer = .initStreaming(.stdout(), init.io, &.{});
            try out.interface.writeAll(help);
            return 0;
        },
        .version => {
            var out: std.Io.File.Writer = .initStreaming(.stdout(), init.io, &.{});
            try out.interface.writeAll("espice " ++ version ++ " (VerA " ++ @import("contract").vera_version ++ " " ++ @import("vera_build").commit ++ "; backends: " ++ backends ++ ")\n");
            return 0;
        },
        .usage => return usageFail(),
    }
    std.debug.assert(cli.paths.items.len != 0);
    // With kernels in the build the default is `auto`, which prices each
    // query and stays on the CPU, silently, unless the GPU pays.
    const backend: problem.Request = cli.backend orelse if (problem.gpu_available) .auto else .cpu;
    if (cli.max_parallel == 0) cli.max_parallel = autoJobs();
    if (cli.tangents.items.len == 0) cli.point.tangent_step = 0 else if (cli.point.tangent_step == 0) cli.point.tangent_step = 1e-6;
    // One per process: VerA's VPI is process-global.
    const vpi = vpi_host.init(init.gpa, init.io);
    var failed = false;
    for (cli.paths.items) |path| {
        if (!try runDeck(init, &cli, backend, vpi, path)) failed = true;
    }
    std.debug.assert(cli.max_parallel != 0);
    return if (failed) 1 else 0;
}

// ---- private ----

/// Reads every argument of `args` into `cli`. A usage error prints its own
/// line for an unknown option; the caller prints the usage. Fails only out
/// of memory.
fn parse(gpa: std.mem.Allocator, args: anytype, cli: *Cli) std.mem.Allocator.Error!Parsed {
    while (args.next()) |arg| {
        if (std.mem.eql(u8, arg, "-h") or std.mem.eql(u8, arg, "--help")) {
            return .help;
        } else if (std.mem.eql(u8, arg, "-v") or std.mem.eql(u8, arg, "--version")) {
            return .version;
        } else if (std.mem.eql(u8, arg, "-b") or std.mem.eql(u8, arg, "--batch")) {
            // The default: every query runs.
        } else if (std.mem.eql(u8, arg, "--gpu")) {
            cli.backend = .auto;
            cli.explicit_gpu = true;
        } else if (std.mem.eql(u8, arg, "--plan")) {
            cli.plan = .print_only;
        } else if (std.mem.eql(u8, arg, "--print-dag")) {
            // A later --print-dag keeps an earlier --plan from running anything.
            if (cli.plan == .run) cli.plan = .print_then_run;
        } else if (std.mem.eql(u8, arg, "--timing-in-depth")) {
            cli.timing_in_depth = true;
        } else if (std.mem.eql(u8, arg, "--measures-only")) {
            cli.point.measures_only = true;
        } else if (std.mem.eql(u8, arg, "--early-stop")) {
            cli.point.early_stop = true;
        } else if (std.mem.eql(u8, arg, "--warm-start")) {
            cli.point.warm_start = true;
        } else if (std.mem.eql(u8, arg, "--full-waveforms")) {
            cli.full_waveforms = true;
        } else if (optionValue(arg, "", "--tangents", args)) |value| {
            var it = std.mem.splitScalar(u8, value orelse return .usage, ',');
            while (it.next()) |name| {
                if (name.len == 0) return .usage;
                try cli.tangents.append(gpa, name);
            }
        } else if (optionValue(arg, "", "--tangent-step", args)) |value| {
            const step = std.fmt.parseFloat(f64, value orelse return .usage) catch return .usage;
            if (!(step > 0) or std.math.isInf(step)) return .usage;
            cli.point.tangent_step = step;
        } else if (std.mem.eql(u8, arg, "--lanes")) {
            cli.lanes = true;
        } else if (optionValue(arg, "", "--abort-after", args)) |value| {
            const bound = std.fmt.parseFloat(f64, value orelse return .usage) catch return .usage;
            if (!(bound >= 0) or std.math.isInf(bound)) return .usage;
            cli.point.abort_after = bound;
        } else if (optionValue(arg, "", "--abort-measure", args)) |value| {
            cli.point.abort_measure = value orelse return .usage;
        } else if (optionValue(arg, "-r", "--rawfile", args)) |value| {
            cli.selection.path = value orelse return .usage;
        } else if (optionValue(arg, "", "--format", args)) |value| {
            cli.selection.format = problem.parseFormat(value orelse return .usage) orelse return .usage;
        } else if (optionValue(arg, "", "--backend", args)) |value| {
            const be = std.meta.stringToEnum(problem.Request, value orelse return .usage) orelse return .usage;
            cli.backend = be;
            cli.explicit_gpu = cli.explicit_gpu or be == .cuda or be == .hip;
        } else if (optionValue(arg, "", "--jobs", args)) |value| {
            const v = value orelse return .usage;
            if (std.mem.eql(u8, v, "auto")) {
                cli.max_parallel = 0;
            } else {
                cli.max_parallel = std.fmt.parseInt(u16, v, 10) catch return .usage;
                if (cli.max_parallel == 0) return .usage;
            }
        } else if (optionValue(arg, "", "--vpi", args)) |value| {
            try cli.vpi_apps.append(gpa, value orelse return .usage);
        } else if (optionValue(arg, "-I", "--include", args)) |value| {
            try cli.include_dirs.append(gpa, value orelse return .usage);
        } else if (optionValue(arg, "-tokenizer", "--tokenizer", args)) |value| {
            cli.dialect = problem.parseDialect(value orelse return .usage) orelse return .usage;
        } else if (arg.len > 0 and arg[0] == '-') {
            std.debug.print("Error: unsupported option '{s}'\n", .{arg});
            return .usage;
        } else try cli.paths.append(gpa, arg);
    }
    return if (cli.paths.items.len == 0) .usage else .run;
}

/// Reads, prepares and runs the deck at `path`, printing its `.meas` text
/// and summary lines; false when it failed, after printing why on stderr.
/// The deck is read once: a pipe or FIFO (`/dev/stdin`, `<(...)`) drains on
/// the first read (#234). Fails only when stdout or stderr cannot be written.
fn runDeck(init: std.process.Init, cli: *const Cli, backend: problem.Request, vpi: *const problem.Vpi, path: []const u8) !bool {
    std.debug.assert(cli.max_parallel != 0);
    if (cli.timing_in_depth) std.debug.print("timing: {s}\n", .{path});
    // Each `.meas` block prints as its point ends (`Options.measures`).
    var out_buf: [4096]u8 = undefined;
    var err_buf: [1024]u8 = undefined;
    var stdout: std.Io.File.Writer = .initStreaming(.stdout(), init.io, &out_buf);
    var stderr: std.Io.File.Writer = .initStreaming(.stderr(), init.io, &err_buf);
    const text = std.Io.Dir.cwd().readFileAlloc(init.io, path, init.gpa, .unlimited) catch |err| {
        std.debug.print("Error: {s}: {s}\n", .{ path, @errorName(err) });
        return false;
    };
    defer init.gpa.free(text);
    const dialect = cli.dialect orelse detected: {
        const d = problem.detectDialect(text);
        if (d != .ngspice) std.debug.print("{s}: reading as {t} (--tokenizer to override)\n", .{ path, d });
        break :detected d;
    };
    const p = problem.Problem.init(init.gpa, init.io, .{
        .source = .{ .bytes = .{ .data = text, .origin = path } },
        .dialect = dialect,
        .output = cli.selection,
        .backend = .{
            .backend = backend,
            .gpu_explicit = cli.explicit_gpu,
            .gpu_quiet = cli.backend == null,
            .solver_threads = @intCast(@min(envThreads("ESPICE_SOLVER_THREADS"), 16)),
            .device_threads = envThreads("ESPICE_THREADS"),
            // One deck, run once: no query is appended later.
            .final_plan = true,
            .lanes = cli.lanes,
        },
        .max_parallel = cli.max_parallel,
        // Unset thread variables: large queries borrow lanes.
        .auto_threads = std.c.getenv("ESPICE_THREADS") == null and std.c.getenv("ESPICE_SOLVER_THREADS") == null,
        .timing_in_depth = cli.timing_in_depth,
        .point = cli.point,
        // --plan solves nothing, not even the warm OP.
        .auto_point = !cli.full_waveforms and cli.plan != .print_only,
        .live_params = cli.tangents.items,
        .measures = .{ .out = &stdout.interface, .err = &stderr.interface },
        .vpi_apps = cli.vpi_apps.items,
        .vpi = vpi,
        .hdl_toolchain = .{ .include_dirs = cli.include_dirs.items },
    }) catch |err| {
        std.debug.print("Error: {s}: {s}\n", .{ path, @errorName(err) });
        return false;
    };
    defer p.deinit();
    if (cli.plan != .run) {
        try p.print(&stdout.interface, .{ .limits = .{ .max_parallel = cli.max_parallel } });
        try stdout.interface.flush();
    }
    if (cli.plan == .print_only) return true;
    p.run_all() catch |err| {
        std.debug.print("Error: {s}: {s}\n", .{ path, @errorName(err) });
        if (p.output_error()) |delivery| std.debug.print("Output error: {s}\n", .{@errorName(delivery)});
        return false;
    };
    try p.print_measures(&stdout.interface, &stderr.interface);
    try p.print_tangents(&stdout.interface);
    try stdout.interface.flush();
    try stderr.interface.flush();
    try printSummary(init.gpa, p);
    return true;
}

/// The deck's stderr summary: its title and device count, then each
/// requested result's and each variant run's shape.
fn printSummary(gpa: std.mem.Allocator, p: *problem.Problem) !void {
    std.debug.print("{s}: {d} devices\n", .{ p.title(), p.device_count() });
    for (0..p.query_count()) |i| {
        const id: problem.QueryId = @fromBackingInt(@intCast(i));
        if (!(try p.query_info(id)).requested) continue;
        const s = try p.summary(id);
        std.debug.print("  {s}: {d} points, {d} variables\n", .{ s.plotname, s.npoints, s.nvars });
    }
    const runs = try p.run_results(gpa);
    defer gpa.free(runs);
    for (runs) |result| std.debug.print("  {s}: {d} points, {d} variables\n", .{ result.plotname, result.npoints, result.varnames.len });
}

/// `--jobs=auto`: one query per CPU.
fn autoJobs() u16 {
    const jobs: u16 = @intCast(std.math.clamp(std.Thread.getCpuCount() catch 1, 1, std.math.maxInt(u16)));
    std.debug.assert(jobs >= 1);
    return jobs;
}

/// A positive thread count from the environment, 1 when unset or unparsable.
fn envThreads(comptime name: [:0]const u8) u32 {
    const value = std.c.getenv(name) orelse return 1;
    return @max(1, std.fmt.parseInt(u32, std.mem.span(value), 10) catch 1);
}

/// The value of `short`/`long` given as `-r X`, `--rawfile X` or
/// `--rawfile=X`. Outer null: `arg` is not this flag. Inner null: the value
/// is missing or empty (`--rawfile=`), which is a usage error. A separate
/// value is taken from `args`.
// `args` is anytype: argv's iterator and the tests' split iterator both
// have `next() ?[]const u8`.
fn optionValue(arg: []const u8, short: []const u8, long: []const u8, args: anytype) ??[]const u8 {
    std.debug.assert(long.len > 2);
    if ((short.len != 0 and std.mem.eql(u8, arg, short)) or std.mem.eql(u8, arg, long)) {
        const value = args.next() orelse return @as(?[]const u8, null);
        return if (value.len == 0) @as(?[]const u8, null) else value;
    }
    for ([_][]const u8{ short, long }) |name| {
        if (name.len != 0 and std.mem.startsWith(u8, arg, name) and arg.len > name.len and arg[name.len] == '=')
            return if (arg.len == name.len + 1) @as(?[]const u8, null) else arg[name.len + 1 ..];
    }
    return null;
}

fn usageFail() u8 {
    std.debug.print("Usage: espice [OPTION]... FILE...\nTry 'espice --help' for options.\n", .{});
    return 2;
}

test {
    // Analyzed only from `main` otherwise, which no test build reaches.
    std.testing.refAllDecls(vpi_host);
}

test "--version matches build.zig.zon" {
    const zon = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, "build.zig.zon", std.testing.allocator, .limited(1 << 16));
    defer std.testing.allocator.free(zon);
    try std.testing.expect(std.mem.indexOf(u8, zon, ".version = \"" ++ version ++ "\",") != null);
}

test "optionValue: separate, joined, missing and empty values" {
    const t = std.testing;
    var args = std.mem.splitScalar(u8, "out.raw", ' ');
    try t.expectEqualStrings("out.raw", optionValue("-r", "-r", "--rawfile", &args).?.?);
    try t.expect(args.next() == null); // the value was consumed
    // Value missing at the end of argv.
    try t.expect(optionValue("--rawfile", "-r", "--rawfile", &args).? == null);
    var empty = std.mem.splitScalar(u8, "", ' ');
    try t.expect(optionValue("-r", "-r", "--rawfile", &empty).? == null);
    try t.expectEqualStrings("a=b", optionValue("--rawfile=a=b", "-r", "--rawfile", &args).?.?);
    try t.expect(optionValue("--rawfile=", "-r", "--rawfile", &args).? == null);
    // The single-dash name takes the joined form too (#88).
    try t.expectEqualStrings("hspice", optionValue("-tokenizer=hspice", "-tokenizer", "--tokenizer", &args).?.?);
    // Not this flag: a longer flag sharing the prefix, a bare value, and an
    // empty short name that must not match an empty argument.
    try t.expect(optionValue("--rawfilex", "-r", "--rawfile", &args) == null);
    try t.expect(optionValue("out.raw", "-r", "--rawfile", &args) == null);
    try t.expect(optionValue("", "", "--format", &args) == null);
}

test "envThreads: positive integers only, 1 otherwise" {
    const libc = struct {
        extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;
        extern "c" fn unsetenv(name: [*:0]const u8) c_int;
    };
    const name = "ESPICE_TEST_THREADS";
    defer _ = libc.unsetenv(name);
    _ = libc.unsetenv(name);
    try std.testing.expectEqual(@as(u32, 1), envThreads(name));
    for ([_]struct { [*:0]const u8, u32 }{ .{ "8", 8 }, .{ "0", 1 }, .{ "-3", 1 }, .{ "abc", 1 }, .{ "", 1 } }) |case| {
        try std.testing.expectEqual(@as(c_int, 0), libc.setenv(name, case[0], 1));
        try std.testing.expectEqual(case[1], envThreads(name));
    }
}
