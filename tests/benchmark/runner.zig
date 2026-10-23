//! `zig build bench`: times espice, ngspice and VACASK (and, given their
//! binaries, Xyce, HSPICE and Spectre) on every deck under a directory and writes a
//! markdown table of medians and pairwise waveform agreement. Runs are
//! serial, since concurrent runs distort each other's timings.
//!   nix develop .#benchmarking --command zig build bench -- --iters 3 --filter op/
//! `--espice LABEL=WORD,...` (repeatable) adds an espice column; a word is a
//! flag, or an environment assignment when it has `=` and no leading `-`.
//! Without it the one column is `espice=--backend=cpu`. `--ngspice none` or
//! `--vacask none` drops that reference; `--xyce BIN`, `--hspice BIN` and `--spectre BIN`
//! add one. `auto` for any of them takes the tool from PATH when it is
//! installed and drops it when not (`zig build test-conformance`); an
//! HSPICE or Spectre found that way moves the report to `vendor-NAME.md`.
//!   zig build bench-postlayout -- --espice cpu8=ESPICE_THREADS=8,--backend=cpu
const std = @import("std");
const Io = std.Io;
const ngspice = @import("ngspice.zig");
const vacask = @import("vacask.zig");
const xyce = @import("xyce.zig");
const hspice = @import("hspice.zig");
const spectre = @import("spectre.zig");
const compare = @import("compare.zig");
const completion = @import("completion.zig");
const common = @import("job.zig");
const Allocator = std.mem.Allocator;

/// One espice configuration: its column label, the `KEY=VALUE` environment
/// words and the flags.
const Variant = struct { label: []const u8, env: []const []const u8, flags: []const []const u8 };
const Config = struct {
    espice: []const u8,
    variants: []const Variant = &.{},
    /// The reference binaries, in `Ref` order; "none" skips one.
    refs: std.enums.EnumArray(Ref, []const u8) = .init(.{ .ngspice = "ngspice", .vacask = "vacask", .xyce = "none", .hspice = "none", .spectre = "none" }),
    /// Extra command-line words for HSPICE and Spectre (threads, APS).
    hspice_args: []const []const u8 = &.{},
    spectre_args: []const []const u8 = &.{},
    /// Give HSPICE and Spectre ESPice's tolerances where the deck sets none.
    match_tol: bool = true,
    fixtures: []const u8,
    filter: []const u8 = "",
    iters: u16 = 3,
    timeout: u32 = 300,
    rtol: f64 = 1e-3,
    klu: bool = false,
    vacask_tran_fs: ?f64 = null,
    vacask_tran_step: ?f64 = null,
    list: bool = false,
    report: []const u8 = "zig-out/benchmark-results.md",
};
/// The reference simulators, in column order.
const Ref = enum {
    ngspice,
    vacask,
    xyce,
    hspice,
    spectre,

    fn label(r: Ref) []const u8 {
        return switch (r) {
            .ngspice => "ngspice",
            .vacask => "VACASK",
            .xyce => "Xyce",
            .hspice => "HSPICE",
            .spectre => "Spectre",
        };
    }
    /// The executable `auto` looks for on PATH.
    fn binary(r: Ref) []const u8 {
        return switch (r) {
            .ngspice => "ngspice",
            .vacask => "vacask",
            .xyce => "Xyce",
            .hspice => "hspice",
            .spectre => "spectre",
        };
    }
    /// These references report their own run and analysis times.
    fn vendor(r: Ref) bool {
        return r == .hspice or r == .spectre;
    }
};
/// Median wall time in ms and the plots of one engine on one fixture,
/// allocated in the fixture's arena, plus a vendor tool's own median times.
const Measurement = struct { milliseconds: f64, samples_ms: []const f64, plots: []const compare.Plot, reported: ?common.Reported = null };

pub fn main(init: std.process.Init) !void {
    const a = init.arena.allocator();
    const io = init.io;
    const cfg = try arguments(a, io, init);
    const decks = try findDecks(a, io, cfg.fixtures, cfg.filter);
    if (decks.len == 0) return error.NoMatchingFixtures;
    if (cfg.list) {
        for (decks) |path| std.debug.print("{s}\n", .{path});
        std.debug.print("{d} fixtures\n", .{decks.len});
        return;
    }

    var report: Io.Writer.Allocating = .init(a);
    var notes: Io.Writer.Allocating = .init(a);
    var samples: Io.Writer.Allocating = .init(a);
    const samples_path = try std.fmt.allocPrint(a, "{s}.samples.jsonl", .{cfg.report});
    const w = &report.writer;
    try w.writeAll("# Simulator benchmark\n\n");
    const version_flags: std.enums.EnumArray(Ref, []const u8) = .init(.{ .ngspice = ngspice.version_flag, .vacask = vacask.version_flag, .xyce = xyce.version_flag, .hspice = hspice.version_flag, .spectre = spectre.version_flag });
    var refs: std.ArrayList(Ref) = .empty;
    try printVersion(a, io, w, "espice", cfg.espice, "--version");
    for (std.enums.values(Ref)) |r| {
        if (std.mem.eql(u8, cfg.refs.get(r), "none")) continue;
        try refs.append(a, r);
        try printVersion(a, io, w, @tagName(r), cfg.refs.get(r), version_flags.get(r));
    }
    for (cfg.variants) |v| try w.print("- column {s}: `{s} espice {s}`\n", .{ v.label, try std.mem.join(a, " ", v.env), try std.mem.join(a, " ", v.flags) });
    var vendors: std.ArrayList(Ref) = .empty;
    for (refs.items) |r| if (r.vendor()) try vendors.append(a, r);
    for (vendors.items) |r| try w.print("- {s} arguments: `{s}`\n", .{ r.label(), try std.mem.join(a, " ", if (r == .hspice) cfg.hspice_args else cfg.spectre_args) });
    if (vendors.items.len > 0) try w.print("- {s}: decks run through tests/benchmark/hspice.zig's gate, with the dialect's TNOM/TEMP made explicit{s}. Their run ms is the tool's own elapsed time less licensing, analysis ms the sum of its per-analysis times; load is the 1-minute load average before the deck.\n", .{
        if (vendors.items.len == 2) "HSPICE and Spectre" else vendors.items[0].label(),
        if (cfg.match_tol) " and ESPice's reltol=1e-3 abstol=1e-12 vntol=1e-6 where the deck sets none" else "",
    });
    try w.print("\nMedian of {d} measured runs after one warm-up; milliseconds include process startup and output. Runs are serial, with engine order rotated each repetition. VACASK conversion is outside timing.\n\n", .{cfg.iters});

    // Columns: the espice variants, then each reference in use. Agreement:
    // each variant against each reference, then each pair of references.
    const nv = cfg.variants.len;
    const labels = try a.alloc([]const u8, nv + refs.items.len);
    for (cfg.variants, labels[0..nv]) |v, *l| l.* = v.label;
    for (refs.items, labels[nv..]) |r, *l| l.* = r.label();
    var pairs: std.ArrayList([2]usize) = .empty;
    for (0..nv) |i| for (nv..labels.len) |j| try pairs.append(a, .{ i, j });
    for (nv..labels.len) |i| for (i + 1..labels.len) |j| try pairs.append(a, .{ i, j });
    try w.writeAll("| Fixture |");
    for (labels) |l| try w.print(" {s} ms |", .{l});
    for (vendors.items) |r| try w.print(" {s} run ms | {s} analysis ms |", .{ r.label(), r.label() });
    if (vendors.items.len > 0) try w.writeAll(" load |");
    for (pairs.items) |p| try w.print(" {s}/{s} |", .{ labels[p[0]], labels[p[1]] });
    try w.writeAll("\n|---|");
    for (labels) |_| try w.writeAll("---:|");
    for (vendors.items) |_| try w.writeAll("---:|---:|");
    if (vendors.items.len > 0) try w.writeAll("---:|");
    for (pairs.items) |_| try w.writeAll("---|");
    try w.writeByte('\n');
    std.debug.print("Benchmark: {d} fixtures, {d} repetitions; references from PATH/overrides\n", .{ decks.len, cfg.iters });
    var random: [16]u8 = undefined;
    io.random(&random);
    const scratch = try std.fmt.allocPrint(a, "zig-out/bench-out/{s}", .{std.fmt.bytesToHex(random, .lower)});
    try Io.Dir.cwd().createDirPath(io, scratch);

    try snapshotRun(a, io, scratch, cfg);

    for (decks, 0..) |relative, index| {
        var arena = std.heap.ArenaAllocator.init(init.gpa);
        defer arena.deinit();
        const fa = arena.allocator();
        const source = try Io.Dir.cwd().realPathFileAlloc(io, try std.fmt.allocPrint(fa, "{s}/{s}", .{ cfg.fixtures, relative }), fa);
        const directory = try std.fmt.allocPrint(fa, "{s}/{d}", .{ scratch, index });
        try Io.Dir.cwd().createDirPath(io, directory);
        const absolute = try Io.Dir.cwd().realPathFileAlloc(io, directory, fa);
        const source_bytes = try Io.Dir.cwd().readFileAlloc(io, source, fa, .unlimited);
        const oracle_path = try std.fmt.allocPrint(fa, "{s}.expected.json", .{source[0 .. source.len - 3]});
        const oracle_bytes = Io.Dir.cwd().readFileAlloc(io, oracle_path, fa, .unlimited) catch null;
        const contract_path = try std.fmt.allocPrint(fa, "{s}.completion.json", .{source[0 .. source.len - 3]});
        const contract_bytes = Io.Dir.cwd().readFileAlloc(io, contract_path, fa, .unlimited) catch null;
        const contract = if (contract_bytes) |bytes|
            try std.json.parseFromSliceLeaky(completion.Contract, fa, bytes, .{ .ignore_unknown_fields = true })
        else
            try completion.fromDeck(fa, source_bytes, oracle_bytes);
        try Io.Dir.cwd().writeFile(io, .{ .sub_path = try std.fs.path.join(fa, &.{ absolute, "source.sp" }), .data = source_bytes });
        const results = try fa.alloc(?Measurement, labels.len);
        const reasons = try fa.alloc([]const u8, labels.len);
        const jobs = try fa.alloc(?common.Job, labels.len);
        @memset(results, null);
        @memset(reasons, "");
        @memset(jobs, null);
        // The oracle's dialect, as the correctness runner passes it: an
        // HSPICE deck read as ngspice fails before it runs.
        const dialect = try tokenizerOf(fa, io, source);
        const hspice_dialect = if (dialect) |d| std.mem.eql(u8, d, "hspice") else false;
        // Sampled before the deck's runs, so they do not count themselves.
        const load = loadAverage(io, fa);
        for (labels, 0..) |label, i| {
            const job: common.Job = if (i < nv) blk: {
                const v = cfg.variants[i];
                const raw = try std.fmt.allocPrint(fa, "{s}/{s}.raw", .{ absolute, label });
                const tokenizer: []const []const u8 = if (dialect) |t| &.{ "--tokenizer", t } else &.{};
                const argv = try std.mem.concat(fa, []const u8, &.{
                    if (v.env.len > 0) &.{"env"} else &.{}, v.env,     &.{cfg.espice},
                    v.flags,                                tokenizer, &.{ "--format=binary", "-b", "-r", raw, source },
                });
                break :blk .{ .argv = argv, .cwd = .{ .path = std.fs.path.dirname(source).? }, .raw = raw };
            } else switch (refs.items[i - nv]) {
                .ngspice => try ngspice.prepare(io, fa, cfg.refs.get(.ngspice), source, try std.fmt.allocPrint(fa, "{s}/ngspice.raw", .{absolute}), cfg.klu, hspice_dialect),
                .vacask => try vacask.prepareWithOptions(io, fa, cfg.refs.get(.vacask), source, try std.fmt.allocPrint(fa, "{s}/vacask", .{absolute}), hspice_dialect, .{ .tran_fs = cfg.vacask_tran_fs, .tran_step = cfg.vacask_tran_step }),
                .xyce => try xyce.prepare(io, fa, cfg.refs.get(.xyce), source, try std.fmt.allocPrint(fa, "{s}/xyce.raw", .{absolute}), dialect),
                .hspice => try hspice.prepare(io, fa, cfg.refs.get(.hspice), cfg.hspice_args, source, try std.fmt.allocPrint(fa, "{s}/hspice", .{absolute}), hspice_dialect, cfg.match_tol),
                .spectre => try spectre.prepare(io, fa, cfg.refs.get(.spectre), cfg.spectre_args, source, try std.fmt.allocPrint(fa, "{s}/spectre", .{absolute}), hspice_dialect, cfg.match_tol),
            };
            if (job.refused.len > 0) {
                reasons[i] = job.refused;
                continue;
            }
            jobs[i] = job;
        }
        try snapshotJobs(fa, io, absolute, source, jobs, labels, contract);
        try measureRounds(io, fa, jobs, cfg.iters, cfg.timeout, index, results, reasons, contract);
        const row_start = report.written().len;
        try w.print("| {s} |", .{relative});
        for (results) |result| {
            if (result) |r| try w.print(" {d:.3} |", .{r.milliseconds}) else try w.writeAll(" — |");
        }
        for (refs.items, results[nv..]) |r, result| {
            if (!r.vendor()) continue;
            if (result != null and result.?.reported != null) {
                const t = result.?.reported.?;
                try w.print(" {d:.3} | {d:.3} |", .{ t.run_ms, t.analysis_ms });
            } else try w.writeAll(" — | — |");
        }
        if (vendors.items.len > 0) try w.print(" {s} |", .{load});
        for (pairs.items) |pair| {
            const left = results[pair[0]];
            const right = results[pair[1]];
            const accuracy = if (left != null and right != null) compare.compareRaws(fa, right.?.plots, left.?.plots, cfg.rtol) else null;
            try w.print(" {s} |", .{if (accuracy) |v| if (!v.complete) "incomplete" else if (v.pass) "agree" else "DIFFER" else "unavailable"});
            if (accuracy) |v| if (!v.pass) {
                try notes.writer.print("- {s} / {s}/{s}: complete={}, normalized max={e}, RMS={e}\n", .{ relative, labels[pair[0]], labels[pair[1]], v.complete, v.max_rel, v.rms_rel });
            };
        }
        try w.writeByte('\n');
        std.debug.print("{s}", .{report.written()[row_start..]});
        for (reasons, labels) |reason, label| {
            if (reason.len == 0) continue;
            try notes.writer.print("- {s} / {s}: {s}\n", .{ relative, label, reason });
            std.debug.print("  {s}: {s}\n", .{ label, reason });
        }
        for (results, labels, reasons) |result, label, reason| {
            try std.json.Stringify.value(.{
                .fixture = relative,
                .engine = label,
                .samples_ms = if (result) |r| r.samples_ms else &.{},
                .median_ms = if (result) |r| @as(?f64, r.milliseconds) else null,
                .failure = if (result == null) @as(?[]const u8, reason) else null,
                .scratch = scratch,
                .index = index,
                .source = source,
                .completion_contract = contract,
                .completion = .{ .complete = result != null, .reason = if (result == null) @as(?[]const u8, reason) else null },
                .provenance = try std.fs.path.join(fa, &.{ absolute, "provenance.json" }),
            }, .{}, &samples.writer);
            try samples.writer.writeByte('\n');
        }
        // Publish progress after every case, so a later timeout cannot erase it.
        try Io.Dir.cwd().createDirPath(io, std.fs.path.dirname(cfg.report) orelse ".");
        try Io.Dir.cwd().writeFile(io, .{ .sub_path = cfg.report, .data = report.written() });
        try Io.Dir.cwd().writeFile(io, .{ .sub_path = samples_path, .data = samples.written() });
    }
    try w.print("\n{s}", .{notes.written()});
    try Io.Dir.cwd().writeFile(io, .{ .sub_path = cfg.report, .data = report.written() });
    std.debug.print("Report: {s}\nRaw outputs: {s}\n", .{ cfg.report, scratch });
}

/// Hash binaries once outside timing; argv and exact staged decks are saved per
/// fixture. Repository patch/build provenance is prepared by prepare_comparison.
fn fileHash(a: Allocator, io: Io, path: []const u8) ![]const u8 {
    const file = try Io.Dir.cwd().openFile(io, path, .{});
    defer file.close(io);
    var hash = std.crypto.hash.sha2.Sha256.init(.{});
    var buffer: [65536]u8 = undefined;
    while (true) {
        const n = file.readStreaming(io, &.{&buffer}) catch |err| switch (err) {
            error.EndOfStream => break,
            else => return err,
        };
        if (n == 0) break;
        hash.update(buffer[0..n]);
    }
    return a.dupe(u8, &std.fmt.bytesToHex(hash.finalResult(), .lower));
}

fn binaryPath(a: Allocator, io: Io, bin: []const u8) ![]const u8 {
    if (std.mem.indexOfScalar(u8, bin, '/') != null) return bin;
    const located = try std.process.run(a, io, .{ .argv = &.{ "which", bin } });
    if (located.term != .exited or located.term.exited != 0) return error.MissingExecutable;
    return std.mem.trim(u8, located.stdout, " \t\r\n");
}

fn snapshotRun(a: Allocator, io: Io, scratch: []const u8, cfg: Config) !void {
    const Binary = struct { label: []const u8, path: []const u8, sha256: []const u8 };
    var binaries: std.ArrayList(Binary) = .empty;
    try binaries.append(a, .{ .label = "espice", .path = cfg.espice, .sha256 = try fileHash(a, io, cfg.espice) });
    for (std.enums.values(Ref)) |r| {
        const bin = cfg.refs.get(r);
        if (std.mem.eql(u8, bin, "none")) continue;
        const path = try binaryPath(a, io, bin);
        try binaries.append(a, .{ .label = r.label(), .path = path, .sha256 = try fileHash(a, io, path) });
    }
    const revision = try std.process.run(a, io, .{ .argv = &.{ "git", "rev-parse", "HEAD" } });
    const patch = try std.process.run(a, io, .{ .argv = &.{ "git", "diff", "--binary", "HEAD" }, .stdout_limit = .unlimited });
    if (revision.term != .exited or revision.term.exited != 0 or patch.term != .exited or patch.term.exited != 0) return error.MissingSourceProvenance;
    try Io.Dir.cwd().writeFile(io, .{ .sub_path = try std.fs.path.join(a, &.{ scratch, "working-tree.patch" }), .data = patch.stdout });
    const Untracked = struct { path: []const u8, sha256: []const u8, snapshot: []const u8 };
    var untracked: std.ArrayList(Untracked) = .empty;
    const listed = try std.process.run(a, io, .{ .argv = &.{ "git", "ls-files", "--others", "--exclude-standard", "-z" }, .stdout_limit = .unlimited });
    if (listed.term != .exited or listed.term.exited != 0) return error.MissingSourceProvenance;
    const untracked_dir = try std.fs.path.join(a, &.{ scratch, "untracked" });
    try Io.Dir.cwd().createDirPath(io, untracked_dir);
    var paths = std.mem.tokenizeScalar(u8, listed.stdout, 0);
    while (paths.next()) |path| {
        if (std.mem.indexOf(u8, path, "__pycache__/") != null) continue;
        const sha = try fileHash(a, io, path);
        const bytes = try Io.Dir.cwd().readFileAlloc(io, path, a, .unlimited);
        const target = try std.fmt.allocPrint(a, "untracked/{s}", .{sha});
        try Io.Dir.cwd().writeFile(io, .{ .sub_path = try std.fs.path.join(a, &.{ scratch, target }), .data = bytes });
        try untracked.append(a, .{ .path = path, .sha256 = sha, .snapshot = target });
    }
    const state = try std.json.Stringify.valueAlloc(a, .{
        .schema_version = 1,
        .revision = std.mem.trim(u8, revision.stdout, " \n"),
        .working_tree_patch_sha256 = try fileHash(a, io, try std.fs.path.join(a, &.{ scratch, "working-tree.patch" })),
        .untracked = untracked.items,
        .binaries = binaries.items,
        .runner_build_mode = @tagName(@import("builtin").mode),
        .simulator_build_options = "not inferred; record in prepare_comparison provenance",
        .configuration = cfg,
    }, .{ .whitespace = .indent_2 });
    try Io.Dir.cwd().writeFile(io, .{ .sub_path = try std.fs.path.join(a, &.{ scratch, "provenance.json" }), .data = state });
}

fn snapshotJobs(a: Allocator, io: Io, directory: []const u8, source: []const u8, jobs: []const ?common.Job, labels: []const []const u8, contract: completion.Contract) !void {
    const Input = struct { path: []const u8, sha256: []const u8, snapshot: []const u8 };
    const Invocation = struct { engine: []const u8, argv: []const []const u8, cwd: []const u8 };
    var invocations: std.ArrayList(Invocation) = .empty;
    var inputs: std.ArrayList(Input) = .empty;
    try inputs.append(a, .{ .path = source, .sha256 = try fileHash(a, io, source), .snapshot = "source.sp" });
    // Prepared native decks and adapter scripts are immutable run inputs. Copy
    // them before the first process; the archived source has its original path
    // recorded so relative includes can be resolved against the input manifest.
    var dir = try Io.Dir.cwd().openDir(io, directory, .{ .iterate = true });
    defer dir.close(io);
    var walker = try dir.walk(a);
    defer walker.deinit();
    var paths: std.ArrayList([]const u8) = .empty;
    while (try walker.next(io)) |entry| if (entry.kind == .file and !std.mem.eql(u8, entry.path, "source.sp")) {
        try paths.append(a, try a.dupe(u8, entry.path));
    };
    const archive = try std.fs.path.join(a, &.{ directory, "inputs" });
    try Io.Dir.cwd().createDirPath(io, archive);
    for (paths.items) |relative| {
        const path = try std.fs.path.join(a, &.{ directory, relative });
        const sha = try fileHash(a, io, path);
        const archived = try std.fmt.allocPrint(a, "inputs/{s}", .{sha});
        const bytes = try Io.Dir.cwd().readFileAlloc(io, path, a, .unlimited);
        try dir.writeFile(io, .{ .sub_path = archived, .data = bytes });
        try inputs.append(a, .{ .path = path, .sha256 = sha, .snapshot = archived });
    }
    for (jobs, labels) |maybe_job, label| if (maybe_job) |job| {
        try invocations.append(a, .{ .engine = label, .argv = job.argv, .cwd = job.cwd.path });
    };
    const state = try std.json.Stringify.valueAlloc(a, .{ .schema_version = 1, .inputs = inputs.items, .invocations = invocations.items, .completion_contract = contract }, .{ .whitespace = .indent_2 });
    try dir.writeFile(io, .{ .sub_path = "provenance.json", .data = state });
}

/// The `tokenizer` field of the deck's `.expected.json`, or null when the
/// deck has no oracle (post-layout decks) or the oracle names none.
fn tokenizerOf(a: Allocator, io: Io, deck: []const u8) !?[]const u8 {
    const path = try std.fmt.allocPrint(a, "{s}.expected.json", .{deck[0 .. deck.len - ".sp".len]});
    const bytes = Io.Dir.cwd().readFileAlloc(io, path, a, .limited(64 << 20)) catch |err| switch (err) {
        error.FileNotFound => return null,
        else => return err,
    };
    const Oracle = struct { tokenizer: ?[]const u8 = null };
    const oracle = std.json.parseFromSliceLeaky(Oracle, a, bytes, .{ .ignore_unknown_fields = true }) catch return null;
    return oracle.tokenizer;
}

/// Every `*.sp` under `root` whose relative path contains `filter`, sorted.
/// Files under a deck's side directory (`*.assets/`) are not decks.
fn findDecks(a: Allocator, io: Io, root: []const u8, filter: []const u8) ![]const []const u8 {
    var dir = try Io.Dir.cwd().openDir(io, root, .{ .iterate = true });
    defer dir.close(io);
    var walker = try dir.walk(a);
    defer walker.deinit();
    var paths: std.ArrayList([]const u8) = .empty;
    while (try walker.next(io)) |entry| {
        if (entry.kind != .file or !std.mem.endsWith(u8, entry.path, ".sp")) continue;
        if (std.mem.indexOf(u8, entry.path, ".assets/") != null or !matches(entry.path, filter)) continue;
        try paths.append(a, try a.dupe(u8, entry.path));
    }
    std.mem.sort([]const u8, paths.items, {}, lessStr);
    return paths.items;
}

fn arguments(a: Allocator, io: Io, init: std.process.Init) !Config {
    var args = init.minimal.args.iterate();
    _ = args.skip();
    const engine = args.next() orelse return error.MissingEngine;
    var cfg: Config = .{ .espice = try Io.Dir.cwd().realPathFileAlloc(io, engine, a), .fixtures = args.next() orelse return error.MissingFixtures };
    var variants: std.ArrayList(Variant) = .empty;
    const Flag = enum(u8) { list, klu, native_tol, filter, iters, timeout, rtol, out, ngspice, vacask, xyce, hspice, spectre, hspice_args, spectre_args, espice, vacask_tran_fs, vacask_tran_step };
    const flags = std.StaticStringMap(Flag).initComptime(.{
        .{ "--list", .list },                    .{ "--ngspice-klu", .klu },               .{ "--filter", .filter },
        .{ "--iters", .iters },                  .{ "--timeout", .timeout },               .{ "--rtol", .rtol },
        .{ "--out", .out },                      .{ "--ngspice", .ngspice },               .{ "--vacask", .vacask },
        .{ "--xyce", .xyce },                    .{ "--espice", .espice },                 .{ "--hspice", .hspice },
        .{ "--spectre", .spectre },              .{ "--hspice-args", .hspice_args },       .{ "--spectre-args", .spectre_args },
        .{ "--vendor-tolerances", .native_tol }, .{ "--vacask-tran-fs", .vacask_tran_fs }, .{ "--vacask-tran-step", .vacask_tran_step },
    });
    while (args.next()) |arg| {
        const flag = flags.get(arg) orelse return error.UnknownArgument;
        switch (flag) {
            .list => cfg.list = true,
            .klu => cfg.klu = true,
            .native_tol => cfg.match_tol = false,
            else => {}, // else: value flags, parsed below
        }
        if (flag == .list or flag == .klu or flag == .native_tol) continue;
        const value = args.next() orelse return error.MissingArgument;
        switch (flag) {
            .filter => cfg.filter = value,
            .iters => cfg.iters = try std.fmt.parseInt(u16, value, 10),
            .timeout => cfg.timeout = try std.fmt.parseInt(u32, value, 10),
            .rtol => cfg.rtol = try std.fmt.parseFloat(f64, value),
            .out => cfg.report = value,
            .ngspice => cfg.refs.set(.ngspice, value),
            .vacask => cfg.refs.set(.vacask, value),
            .vacask_tran_fs => cfg.vacask_tran_fs = try std.fmt.parseFloat(f64, value),
            .vacask_tran_step => cfg.vacask_tran_step = try std.fmt.parseFloat(f64, value),
            .xyce => cfg.refs.set(.xyce, value),
            .hspice => cfg.refs.set(.hspice, value),
            .spectre => cfg.refs.set(.spectre, value),
            // Comma-separated, like an espice variant's words.
            .hspice_args => cfg.hspice_args = try splitWords(a, value),
            .spectre_args => cfg.spectre_args = try splitWords(a, value),
            .espice => try variants.append(a, try parseVariant(a, value)),
            .list, .klu, .native_tol => unreachable,
        }
    }
    if (cfg.iters == 0 or cfg.timeout == 0 or cfg.rtol <= 0 or !std.math.isFinite(cfg.rtol)) return error.InvalidArgument;
    if (cfg.vacask_tran_fs) |fraction| {
        if (!std.math.isFinite(fraction) or fraction <= 0 or fraction > 1) return error.InvalidArgument;
    }
    if (cfg.vacask_tran_step) |step| {
        if (!std.math.isFinite(step) or step <= 0) return error.InvalidArgument;
    }
    var auto_vendor = false;
    for (std.enums.values(Ref)) |r| if (std.mem.eql(u8, cfg.refs.get(r), "auto")) {
        const found = if (binaryPath(a, io, r.binary())) |_| true else |_| false;
        cfg.refs.set(r, if (found) r.binary() else "none");
        auto_vendor = auto_vendor or (found and r.vendor());
    };
    // A reference that reports its own times writes only to a
    // `vendor-*.md` file; keep those files out of the repository.
    if (!privateReport(cfg) and auto_vendor) {
        const dir = std.fs.path.dirname(cfg.report) orelse ".";
        cfg.report = try std.fmt.allocPrint(a, "{s}/vendor-{s}", .{ dir, std.fs.path.basename(cfg.report) });
    }
    if (!privateReport(cfg)) {
        std.debug.print("--hspice/--spectre write only to a vendor-*.md report: pass --out zig-out/vendor-NAME.md\n", .{});
        return error.VendorReportNotPrivate;
    }
    for (&cfg.refs.values) |*bin| if (std.mem.indexOfScalar(u8, bin.*, '/') != null) {
        bin.* = try Io.Dir.cwd().realPathFileAlloc(io, bin.*, a);
    };
    if (variants.items.len == 0) try variants.append(a, try parseVariant(a, "espice=--backend=cpu"));
    cfg.variants = variants.items;
    return cfg;
}

/// `LABEL=WORD,WORD,...`. A word with `=` and no leading `-` is an
/// environment assignment; the others are espice flags.
fn parseVariant(a: Allocator, spec: []const u8) !Variant {
    const eq = std.mem.indexOfScalar(u8, spec, '=') orelse return error.InvalidArgument;
    if (eq == 0) return error.InvalidArgument;
    var env: std.ArrayList([]const u8) = .empty;
    var flags: std.ArrayList([]const u8) = .empty;
    var words = std.mem.tokenizeScalar(u8, spec[eq + 1 ..], ',');
    while (words.next()) |word| {
        const is_env = word[0] != '-' and std.mem.indexOfScalar(u8, word, '=') != null;
        try (if (is_env) &env else &flags).append(a, word);
    }
    return .{ .label = spec[0..eq], .env = env.items, .flags = flags.items };
}

/// Whether `cfg` keeps reference results private: no adapter column,
/// or a report named `vendor-*.md`.
fn privateReport(cfg: Config) bool {
    const vendor = !std.mem.eql(u8, cfg.refs.get(.hspice), "none") or !std.mem.eql(u8, cfg.refs.get(.spectre), "none");
    const name = std.fs.path.basename(cfg.report);
    return !vendor or (std.mem.startsWith(u8, name, "vendor-") and std.mem.endsWith(u8, name, ".md"));
}

fn splitWords(a: Allocator, value: []const u8) ![]const []const u8 {
    var words: std.ArrayList([]const u8) = .empty;
    var it = std.mem.tokenizeScalar(u8, value, ',');
    while (it.next()) |word| try words.append(a, word);
    return words.items;
}

/// `- ENGINE: `BIN` — VERSION` from the banner of `BIN FLAG`. HSPICE prints
/// a `Using:` line first, so its banner starts at `Version`.
fn printVersion(a: Allocator, io: Io, w: *Io.Writer, engine: []const u8, bin: []const u8, flag: []const u8) !void {
    const version = try std.process.run(a, io, .{ .argv = &.{ bin, flag }, .stdout_limit = .limited(65536), .stderr_limit = .limited(65536) });
    const banner = if (version.stdout.len > 0) version.stdout else version.stderr;
    const start = if (std.mem.eql(u8, engine, "hspice")) std.mem.indexOf(u8, banner, "Version") orelse 0 else 0;
    const text = std.mem.trim(u8, banner[start..], " \t\r\n*");
    try w.print("- {s}: `{s}` — {s}\n", .{ engine, bin, text[0 .. std.mem.indexOfScalar(u8, text, '\n') orelse text.len] });
}

/// The 1-minute load average, as /proc/loadavg writes it, or "—". The file
/// reports size 0, so it is read as a stream rather than by size.
fn loadAverage(io: Io, a: Allocator) []const u8 {
    const file = Io.Dir.cwd().openFile(io, "/proc/loadavg", .{}) catch return "—";
    defer file.close(io);
    const buf = a.alloc(u8, 128) catch return "—";
    const n = file.readStreaming(io, &.{buf}) catch return "—";
    var words = std.mem.tokenizeAny(u8, buf[0..n], " \n");
    return words.next() orelse "—";
}

fn matches(path: []const u8, filter: []const u8) bool {
    return std.mem.indexOf(u8, path, filter) != null;
}

/// All engines get one warm-up, then one run per round. Rotating the order
/// shares slow temperature/frequency drift across engines. No child runs in
/// parallel with another. A failed repetition invalidates that measurement.
fn measureRounds(io: Io, a: Allocator, jobs: []const ?common.Job, iters: u16, seconds: u32, start_order: usize, results: []?Measurement, reasons: [][]const u8, contract: completion.Contract) !void {
    const Series = struct { wall: []f64, run: []f64, analysis: []f64, reported: bool = true };
    const series = try a.alloc(Series, jobs.len);
    for (series) |*s| s.* = .{ .wall = try a.alloc(f64, iters), .run = try a.alloc(f64, iters), .analysis = try a.alloc(f64, iters) };
    for (0..@as(usize, iters) + 1) |round| {
        for (0..jobs.len) |turn| {
            const k = (start_order + round + turn) % jobs.len;
            const job = jobs[k] orelse continue;
            if (reasons[k].len != 0) continue;
            const sample = runOnce(io, a, job, seconds, round) catch |err| {
                reasons[k] = @errorName(err);
                continue;
            };
            // Every warm-up and measured run must independently fulfill the
            // requested output contract, even if a later repetition succeeds.
            var sample_arena = std.heap.ArenaAllocator.init(a);
            defer sample_arena.deinit();
            const sample_allocator = sample_arena.allocator();
            const plots = readPlots(io, sample_allocator, job) catch |err| {
                reasons[k] = @errorName(err);
                continue;
            };
            if (try completion.check(sample_allocator, plots, contract)) |reason| {
                reasons[k] = reason;
                continue;
            }
            if (round == 0) continue;
            series[k].wall[round - 1] = sample.wall;
            if (sample.reported) |r| {
                series[k].run[round - 1] = r.run_ms;
                series[k].analysis[round - 1] = r.analysis_ms;
            } else series[k].reported = false;
        }
    }
    for (jobs, series, results, reasons) |maybe_job, s, *result, *reason| {
        const job = maybe_job orelse continue;
        if (reason.len != 0) continue;
        const plots = readPlots(io, a, job) catch |err| {
            reason.* = @errorName(err);
            continue;
        };
        result.* = .{
            .milliseconds = medianOf(try a.dupe(f64, s.wall)),
            .samples_ms = s.wall,
            .plots = plots,
            .reported = if (s.reported) .{ .run_ms = medianOf(s.run), .analysis_ms = medianOf(s.analysis) } else null,
        };
    }
}

const Sample = struct { wall: f64, reported: ?common.Reported };

fn runOnce(io: Io, a: Allocator, job: common.Job, seconds: u32, round: usize) !Sample {
    const timeout = try std.fmt.allocPrint(a, "{d}", .{seconds});
    const argv = try std.mem.concat(a, []const u8, &.{ &.{ "timeout", "--kill-after=5", timeout }, job.argv });
    // No repetition can borrow a previous run's output.
    switch (job.reader) {
        .raw => Io.Dir.cwd().deleteFile(io, job.raw) catch |err| switch (err) {
            error.FileNotFound => {},
            else => return err,
        },
        .vacask => try clearRaws(io, job.cwd.path),
        .hspice, .spectre => try clearOutputs(io, job.cwd.path),
    }
    const log_path = try std.fmt.allocPrint(a, "{s}.round-{d}.log", .{ if (job.raw.len > 0) job.raw else job.cwd.path, round });
    const log = try Io.Dir.cwd().createFile(io, log_path, .{});
    defer log.close(io);
    const start = Io.Timestamp.now(io, .awake);
    var child = try std.process.spawn(io, .{ .argv = argv, .cwd = job.cwd, .stdin = .ignore, .stdout = .{ .file = log }, .stderr = .{ .file = log } });
    defer child.kill(io);
    const term = try child.wait(io);
    const elapsed = start.durationTo(Io.Timestamp.now(io, .awake)).nanoseconds;
    if (term != .exited or term.exited != 0) return if (term == .exited and term.exited == 124) error.Timeout else error.SimulatorFailed;
    if (job.reader == .raw) try Io.Dir.cwd().access(io, job.raw, .{});
    const reported: ?common.Reported = switch (job.reader) {
        .hspice => hspice.reported(io, a, job.cwd.path) catch null,
        .spectre => spectre.reported(io, a, job.cwd.path) catch null,
        .raw, .vacask => null,
    };
    return .{ .wall = @as(f64, @floatFromInt(elapsed)) / 1e6, .reported = reported };
}

fn readPlots(io: Io, a: Allocator, job: common.Job) ![]const compare.Plot {
    return switch (job.reader) {
        .raw => compare.parseRawFile(io, a, job.raw) orelse return error.MissingOrInvalidRaw,
        .vacask => try vacaskPlots(io, a, job.cwd.path),
        .hspice => try hspice.readPlots(io, a, job.cwd.path, job.op),
        .spectre => try spectre.readPlots(io, a, job.cwd.path, job.op),
    };
}

/// The median of `xs`, sorting it in place.
fn medianOf(xs: []f64) f64 {
    std.mem.sort(f64, xs, {}, std.sort.asc(f64));
    return if (xs.len % 2 == 1) xs[xs.len / 2] else (xs[xs.len / 2 - 1] + xs[xs.len / 2]) / 2;
}

/// Every file in a vendor job's directory but its deck.
fn clearOutputs(io: Io, path: []const u8) !void {
    var dir = try Io.Dir.cwd().openDir(io, path, .{ .iterate = true });
    defer dir.close(io);
    var it = dir.iterate();
    while (try it.next(io)) |entry| if (entry.kind == .file and !std.mem.eql(u8, entry.name, "deck.sp")) {
        try dir.deleteFile(io, entry.name);
    };
}

fn clearRaws(io: Io, path: []const u8) !void {
    var dir = try Io.Dir.cwd().openDir(io, path, .{ .iterate = true });
    defer dir.close(io);
    var it = dir.iterate();
    while (try it.next(io)) |entry| if (entry.kind == .file and std.mem.endsWith(u8, entry.name, ".raw")) {
        try dir.deleteFile(io, entry.name);
    };
}

/// The plots of every raw file VACASK wrote into `path`, in file-name order.
fn vacaskPlots(io: Io, a: Allocator, path: []const u8) ![]const compare.Plot {
    var dir = try Io.Dir.cwd().openDir(io, path, .{ .iterate = true });
    defer dir.close(io);
    var names: std.ArrayList([]const u8) = .empty;
    var it = dir.iterate();
    while (try it.next(io)) |entry| if (entry.kind == .file and std.mem.endsWith(u8, entry.name, ".raw")) {
        try names.append(a, try a.dupe(u8, entry.name));
    };
    std.mem.sort([]const u8, names.items, {}, lessStr);
    var plots: std.ArrayList(compare.Plot) = .empty;
    for (names.items) |name| {
        const raw = try dir.readFileAlloc(io, name, a, .unlimited);
        try plots.appendSlice(a, compare.parseRawBlob(a, raw) orelse return error.InvalidRaw);
    }
    if (plots.items.len == 0) return error.MissingRaw;
    return plots.toOwnedSlice(a);
}

fn lessStr(_: void, x: []const u8, y: []const u8) bool {
    return std.mem.lessThan(u8, x, y);
}

test "provenance hashing accepts ordinary file EOF" {
    const t = std.testing;
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(t.io, .{ .sub_path = "input", .data = "abc" });
    const path = try std.fmt.allocPrint(t.allocator, ".zig-cache/tmp/{s}/input", .{tmp.sub_path});
    defer t.allocator.free(path);
    const sha = try fileHash(t.allocator, t.io, path);
    defer t.allocator.free(sha);
    try t.expectEqualStrings("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad", sha);
}

test "a vendor run writes only a vendor-*.md report" {
    var cfg: Config = .{ .espice = "espice", .fixtures = "tests/conformance/tests/netlist" };
    try std.testing.expect(privateReport(cfg));
    cfg.refs.set(.spectre, "spectre");
    try std.testing.expect(!privateReport(cfg));
    cfg.report = "zig-out/postlayout-results.md";
    try std.testing.expect(!privateReport(cfg));
    cfg.report = "zig-out/vendor-postlayout.md";
    try std.testing.expect(privateReport(cfg));
}

test "an espice variant splits environment from flags" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const v = try parseVariant(arena.allocator(), "cpu8=ESPICE_THREADS=8,--backend=cpu");
    try std.testing.expectEqualStrings("cpu8", v.label);
    try std.testing.expectEqualStrings("ESPICE_THREADS=8", v.env[0]);
    try std.testing.expectEqual(@as(usize, 1), v.env.len);
    try std.testing.expectEqualStrings("--backend=cpu", v.flags[0]);
    try std.testing.expectError(error.InvalidArgument, parseVariant(arena.allocator(), "unlabelled"));
}

const sample_raw =
    "Title: benchmark test\nPlotname: Operating Point\nFlags: real\n" ++
    "No. Variables: 1\nNo. Points: 1\nVariables:\n\t0\tv(out)\tvoltage\nValues:\n0 1\n";

test "timing rounds rotate engines and retain every measured sample" {
    const t = std.testing;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(t.io, .{ .sub_path = "source.raw", .data = sample_raw });
    const dir = try Io.Dir.cwd().realPathFileAlloc(t.io, try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}", .{tmp.sub_path}), a);
    const source = try std.fs.path.join(a, &.{ dir, "source.raw" });
    const order = try std.fs.path.join(a, &.{ dir, "order" });
    var jobs: [2]?common.Job = undefined;
    for ([_][]const u8{ "A", "B" }, &jobs) |label, *job| {
        const raw = try std.fmt.allocPrint(a, "{s}/{s}.raw", .{ dir, label });
        job.* = .{
            .argv = try a.dupe([]const u8, &.{ "sh", "-c", "printf '%s' \"$1\" >> \"$2\"; cp \"$3\" \"$4\"", "runner-test", label, order, source, raw }),
            .cwd = .{ .path = dir },
            .raw = raw,
        };
    }
    var results: [2]?Measurement = @splat(null);
    var reasons: [2][]const u8 = @splat("");
    try measureRounds(t.io, a, &jobs, 2, 10, 0, &results, &reasons, .{ .requests = &.{.{ .kind = .op }} });
    for (results, reasons) |result, reason| {
        try t.expectEqualStrings("", reason);
        try t.expectEqual(@as(usize, 2), result.?.samples_ms.len);
        try t.expectEqual(@as(usize, 1), result.?.plots.len);
    }
    try t.expectEqualStrings("ABBAAB", try Io.Dir.cwd().readFileAlloc(t.io, order, a, .limited(100)));
}

test "a measured run cannot borrow its warm-up waveform" {
    const t = std.testing;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(t.io, .{ .sub_path = "source.raw", .data = sample_raw });
    const dir = try Io.Dir.cwd().realPathFileAlloc(t.io, try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}", .{tmp.sub_path}), a);
    const source = try std.fs.path.join(a, &.{ dir, "source.raw" });
    const marker = try std.fs.path.join(a, &.{ dir, "warmed" });
    const raw = try std.fs.path.join(a, &.{ dir, "output.raw" });
    const jobs = [_]?common.Job{.{
        .argv = &.{ "sh", "-c", "if [ ! -f \"$1\" ]; then cp \"$2\" \"$3\"; touch \"$1\"; fi", "runner-test", marker, source, raw },
        .cwd = .{ .path = dir },
        .raw = raw,
    }};
    var results: [1]?Measurement = @splat(null);
    var reasons: [1][]const u8 = @splat("");
    try measureRounds(t.io, a, &jobs, 2, 10, 0, &results, &reasons, .{ .requests = &.{.{ .kind = .op }} });
    try t.expect(results[0] == null);
    try t.expectEqualStrings("FileNotFound", reasons[0]);
}

test {
    _ = ngspice;
    _ = vacask;
    _ = xyce;
    _ = hspice;
    _ = spectre;
    _ = compare;
    _ = completion;
}
