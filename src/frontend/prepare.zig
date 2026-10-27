//! The frontend module root: source to netlist to frozen circuit and
//! queries, the entry points the Problem facade calls. Leaves never import
//! this file. Cites no outside source.
const std = @import("std");
const core = @import("core");
const requests = core.query;
const netlist = @import("netlist");
const device = @import("device");
const analyses = @import("analyses.zig");
const builder = @import("builder");
const variants = @import("variants.zig");
const mosra = @import("mosra.zig");
const Builder = builder.Builder;
const Job = requests.Query;
/// The per-Problem device library the facade fills before `build`.
pub const Library = device.Library;
/// HSPICE `OPTIMIZE=`: the optimizer's point-to-rows adapter (variants.zig).
pub const Tuner = variants.Tuner;
/// A SIGMA_SCALE card, for counting its trials' failures (#239).
pub const Sss = variants.Sss;
/// `.wcd`: FORM's points to variant rows (#240, variants.zig).
pub const WorstCase = variants.WorstCase;
const Ic = core.Ic;
const GROUND = core.GROUND;

/// Construction result: the frozen topology and what the deck says about it.
/// The session arena owns every deck slice; `deinit` releases the circuits.
pub const Prepared = struct {
    /// The frozen topology; `deinit` releases it.
    circuit: device.Circuit,
    /// Queries, probes, measures and variants; its slices live in the
    /// session arena.
    deck: core.Deck,
    /// Variant runs whose topology differs from `circuit` (an `.alter`
    /// that swaps elements, a point that collapses a node), each with its
    /// own circuit, deck and variant labels, in output order after this one.
    runs: []Prepared = &.{},
    /// The deck's HSPICE optimization, when it has one. It borrows the
    /// parse arena, which must then outlive the Prepared.
    tuner: ?variants.Tuner = null,
    /// The live `.param` values a caller sets on the prepared circuit
    /// (`prepare`'s `live`); null without any. It borrows the parse arena,
    /// which must then outlive the Prepared.
    tuning: ?*variants.Tuning = null,
    /// The deck's SIGMA_SCALE cards (#239), in the session arena.
    sss: []const variants.Sss = &.{},
    /// The deck's `.wcd` (#240), on the main run only. It borrows the parse
    /// arena, which must then outlive the Prepared.
    wcd: ?variants.WorstCase = null,

    /// Releases this run's circuit and every nested run's, and the tuning's
    /// templates. The deck slices stay with the session arena. Leaves
    /// `self` undefined.
    pub fn deinit(self: *Prepared) void {
        for (self.runs) |*run| run.deinit();
        if (self.tuning) |t| t.deinit();
        self.circuit.deinit();
        self.* = undefined;
    }
};
const NO_NODE = analyses.NO_NODE;

/// The netlist syntax a deck is read in: ngspice, HSPICE or Spectre.
pub const Dialect = netlist.Dialect;

/// How `prepare` reads device cards (#230).
pub const ParseOptions = netlist.ParseOptions;

/// Where a deck comes from.
pub const Source = union(enum) {
    /// Path of the deck file.
    file: []const u8,
    /// Deck text already in memory.
    bytes: Bytes,

    /// In-memory deck; `origin` is a real or virtual filename whose
    /// directory relative includes resolve against.
    pub const Bytes = struct { data: []const u8, origin: []const u8 };
};

/// The dialect a CLI name or short alias (`ng`, `hs`, `scs`) selects.
pub fn parseDialect(name: []const u8) ?Dialect {
    return std.StaticStringMap(Dialect).initComptime(.{
        .{ "ngspice", .ngspice }, .{ "ng", .ngspice },
        .{ "hspice", .hspice },   .{ "hs", .hspice },
        .{ "spectre", .spectre }, .{ "scs", .spectre },
    }).get(name);
}

/// The dialect a deck's own text is in, from markers no ngspice deck of
/// the corpus carries (a test pins that): Spectre's `simulator lang=` or a
/// `name (nodes) master` instance, then an HSPICE-only card or keyword.
/// Anything else is ngspice, HSPICE decks with ngspice syntax included
/// (their meaning can still differ, as TNOM does). Reads neither the title
/// line nor the includes.
pub fn detectDialect(text: []const u8) Dialect {
    const hspice_cards = [_][]const u8{
        ".data", ".enddata", ".alter", ".variation", ".mosra", ".appendmodel", ".lin",      ".net",     ".fft",
        ".stim", ".vec",     ".pat",   ".check",     ".dout",  ".biaschk",     ".power",    ".connect", ".dcvolt",
        ".load", ".sample",  ".sn",    ".snac",      ".snxf",  ".snnoise",     ".ptdnoise",
    };
    var hspice = false;
    var card: []const u8 = "";
    var it = std.mem.splitScalar(u8, text, '\n');
    var title = true;
    while (it.next()) |raw| : (title = false) {
        const line = std.mem.trim(u8, raw, " \t\r");
        if (std.ascii.startsWithIgnoreCase(line, "simulator") and std.ascii.startsWithIgnoreCase(std.mem.trimStart(u8, line[9..], " \t"), "lang")) return .spectre;
        if (title or line.len == 0 or line[0] == '*') continue;
        var rest = line;
        if (line[0] == '+') rest = line[1..] else {
            const end = std.mem.indexOfAny(u8, line, " \t(") orelse line.len;
            card = line[0..end];
            rest = line[end..];
            // `r1 (a b) resistor`: Spectre's instance form.
            const tail = std.mem.trimStart(u8, rest, " \t");
            if (std.ascii.isAlphabetic(line[0]) and tail.len != 0 and tail[0] == '(') if (std.mem.indexOfScalar(u8, tail, ')')) |close| {
                const after = tail[close + 1 ..];
                const master = std.mem.trimStart(u8, after, " \t");
                if (master.len != 0 and master.len < after.len and std.ascii.isAlphabetic(master[0])) return .spectre;
            };
            // Past the first HSPICE marker only a Spectre one can change
            // the answer.
            if (hspice) continue;
            if (std.mem.startsWith(u8, card, ".")) for (hspice_cards) |c| {
                hspice = hspice or std.ascii.eqlIgnoreCase(card, c);
            };
            // `.lib 'file' tt`: a quoted path and a section.
            if (std.ascii.eqlIgnoreCase(card, ".lib") and tail.len != 0 and tail[0] == '\'') if (std.mem.indexOfScalarPos(u8, tail, 1, '\'')) |close| {
                hspice = hspice or std.mem.trim(u8, tail[close + 1 ..], " \t").len != 0;
            };
        }
        // Every word test below needs an `=` or a `sweep` on the line.
        if (hspice or (std.mem.indexOfScalar(u8, rest, '=') == null and std.ascii.findIgnoreCase(rest, "sweep") == null)) continue;
        var words = std.mem.tokenizeAny(u8, rest, " \t");
        var sweep = false;
        while (words.next()) |w| {
            // `sweep data=`, `sweep monte=`, `sweep optimize=` on an analysis card.
            if (sweep) for ([_][]const u8{ "data", "monte", "optimize" }) |k| {
                hspice = hspice or std.ascii.startsWithIgnoreCase(w, k);
            };
            sweep = std.ascii.eqlIgnoreCase(w, "sweep");
            hspice = hspice or std.ascii.startsWithIgnoreCase(w, "rlgcmodel=") or std.ascii.startsWithIgnoreCase(w, "tstonefile=") or
                std.ascii.eqlIgnoreCase(w, "modeltype=rlgc") or
                (std.ascii.startsWithIgnoreCase(card, ".opt") and std.ascii.startsWithIgnoreCase(w, "search=")) or
                (std.ascii.startsWithIgnoreCase(card, "p") and std.ascii.startsWithIgnoreCase(w, "port="));
        }
    }
    return if (hspice) .hspice else .ngspice;
}

test detectDialect {
    try std.testing.expectEqual(Dialect.ngspice, detectDialect("t\nv1 a 0 1\nr1 a 0 {2*rv}\n.tran 1n 1u\n.meas tran x max v(a)\n.end\n"));
    try std.testing.expectEqual(Dialect.hspice, detectDialect("t\n.param rl=1k\n.data rows rl\n+ 1k\n.enddata\n.dc data=rows\n.end\n"));
    try std.testing.expectEqual(Dialect.hspice, detectDialect("t\nr1 a 0 1k\n.tran 1n 1u\n+ sweep monte=10\n.end\n"));
    try std.testing.expectEqual(Dialect.hspice, detectDialect("t\n.lib 'models.lib' tt\n.op\n"));
    try std.testing.expectEqual(Dialect.ngspice, detectDialect("t\n.lib 'models.lib'\n* .data in a comment\n.op\n"));
    try std.testing.expectEqual(Dialect.spectre, detectDialect("simulator lang=spectre\nr1 (a 0) resistor r=1k\n"));
    try std.testing.expectEqual(Dialect.spectre, detectDialect("title\nr1 (out 0) resistor r=1k\nt1 tran stop=1u\n"));
    // The title line may say anything.
    try std.testing.expectEqual(Dialect.ngspice, detectDialect(".data in the title\nv1 a 0 1\n.op\n"));
}

// Every corpus deck whose oracle names no tokenizer reads as ngspice, and
// a deck detected as HSPICE or Spectre is one: the markers are precise.
test "detectDialect: no false positive over the corpus" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var dir = std.Io.Dir.cwd().openDir(io, "tests/conformance/tests/netlist", .{ .iterate = true }) catch return error.SkipZigTest;
    defer dir.close(io);
    var walker = try dir.walk(a);
    defer walker.deinit();
    var found: usize = 0;
    var total: usize = 0;
    while (try walker.next(io)) |entry| {
        if (!std.mem.endsWith(u8, entry.path, ".expected.json")) continue;
        const json = try dir.readFileAlloc(io, entry.path, a, .limited(1 << 24));
        defer a.free(json);
        const deck_path = try std.mem.concat(a, u8, &.{ entry.path[0 .. entry.path.len - ".expected.json".len], ".sp" });
        defer a.free(deck_path);
        const deck = dir.readFileAlloc(io, deck_path, a, .limited(1 << 26)) catch continue;
        defer a.free(deck);
        const wanted: Dialect = for ([_]Dialect{ .hspice, .spectre }) |d| {
            if (std.mem.indexOf(u8, json, switch (d) {
                .hspice => "\"tokenizer\": \"hspice\"",
                else => "\"tokenizer\": \"spectre\"",
            }) != null) break d;
        } else .ngspice;
        const got = detectDialect(deck);
        try std.testing.expect(got == .ngspice or got == wanted);
        if (wanted != .ngspice) total += 1;
        if (wanted != .ngspice and got == wanted) found += 1;
    }
    // 66 of the 102 on 2026-10-20; the rest use only ngspice syntax.
    try std.testing.expect(found * 2 >= total);
}

/// Reads, expands and flattens `input` into `session` and reads its data
/// files; compiles nothing (`loadModels` does, #13). The global `.param`
/// names in `live` stay live, so the built circuit takes new values for
/// them without a re-parse (`Prepared.tuning`). The netlist borrows
/// `session`; release it once `build` has returned, or with the Prepared
/// when `live` is not empty. Spectre input expands its `include`s
/// (`source.expandSpectre`) and nothing else: it splits no `.alter` runs
/// and reads no data files. Fails with `UnknownParameter` for a `live`
/// name no global `.param` defines, with the file error of an
/// unreadable deck, include, data or `soalimits` file, and with
/// `InvalidSoaLimits` for a `soalimits` file `soaRatings` refuses. `parse`
/// says how the device cards are read; every choice gives the same netlist.
pub fn prepare(io: std.Io, session: std.mem.Allocator, input: Source, dialect: Dialect, live: []const []const u8, parse: ParseOptions) !netlist.Netlist {
    const origin = switch (input) {
        .file => |path| path,
        .bytes => |bytes| bytes.origin,
    };
    const raw = switch (input) {
        .file => |path| try std.Io.Dir.cwd().readFileAlloc(io, path, session, .unlimited),
        .bytes => |bytes| try session.dupe(u8, bytes.data),
    };
    if (dialect == .spectre) {
        return netlist.parseLive(session, try netlist.source.expandSpectre(io, session, origin, raw), dialect, live, parse);
    }
    const runs = try netlist.source.splitAlters(session, raw);
    var nl = try netlist.parseLive(session, try netlist.source.expand(io, std.heap.page_allocator, session, origin, runs[0]), dialect, live, parse);
    const alters = try session.alloc([]const u8, runs.len - 1);
    for (alters, runs[1..]) |*text, run| text.* = try netlist.source.expand(io, std.heap.page_allocator, session, origin, run);
    nl.deck.alters = alters;
    nl.files = try loadData(io, session, nl.deck.foreign, origin);
    nl.soa_ratings = try soaRatings(io, session, origin, nl.deck.soa_file);
    return nl;
}

/// The `.options soalimits=` file at `path` (relative to the deck at
/// `origin`; none when empty), checked here, at the trust boundary: one
/// JSON object of model name or name prefix to an object of `soa_keys`
/// limits, each a finite number (#238). `InvalidSoaLimits`, logged with
/// the file, for anything else: bad JSON, a repeated name, an unknown key,
/// a value that is no finite number.
fn soaRatings(io: std.Io, session: std.mem.Allocator, origin: []const u8, path: []const u8) ![]const netlist.SoaRating {
    if (path.len == 0) return &.{};
    const file = try deckRelative(session, origin, path);
    const bytes = std.Io.Dir.cwd().readFileAlloc(io, file, session, .limited(1 << 24)) catch |err| {
        if (!@import("builtin").is_test) std.log.err("netlist: cannot read soalimits file {s}: {s}", .{ file, @errorName(err) });
        return err;
    };
    const Limits = std.json.ArrayHashMap(f64);
    const models = std.json.parseFromSliceLeaky(std.json.ArrayHashMap(Limits), session, bytes, .{}) catch |err| {
        if (!@import("builtin").is_test) std.log.err("netlist: soalimits file {s}: {s}", .{ file, @errorName(err) });
        return error.InvalidSoaLimits;
    };
    var out: std.ArrayList(netlist.SoaRating) = .empty;
    for (models.map.keys(), models.map.values()) |model, limits| for (limits.map.keys(), limits.map.values()) |key, limit| {
        const known = for (soa_keys) |k| {
            if (std.mem.eql(u8, k.key, key)) break true;
        } else false;
        if (!known or !std.math.isFinite(limit)) {
            if (!@import("builtin").is_test) std.log.err("netlist: soalimits file {s}: {s}: {s} is {s}", .{ file, model, key, if (known) "no finite number" else "no SOA limit" });
            return error.InvalidSoaLimits;
        }
        try out.append(session, .{ .model = model, .key = key, .limit = limit });
    };
    return out.items;
}

/// Reads every data file the deck names, relative to the deck's directory.
fn loadData(io: std.Io, session: std.mem.Allocator, foreign: []const netlist.Foreign, origin: []const u8) ![]const netlist.DataFile {
    var files: std.ArrayList(netlist.DataFile) = .empty;
    for (foreign) |f| if (f.kind == .data) {
        const path = try deckRelative(session, origin, f.path);
        const bytes = std.Io.Dir.cwd().readFileAlloc(io, path, session, .unlimited) catch |err| {
            std.log.err("netlist: cannot read data file {s}: {s}", .{ path, @errorName(err) });
            return err;
        };
        try files.append(session, .{ .path = f.path, .bytes = bytes });
    };
    return files.items;
}

/// Compiles and loads into `lib` the HDL models `nl` (from `prepare` on a
/// deck at `origin`) includes, paths relative to the deck's directory.
/// First loads the VPI applications `apps` names (`--vpi`, as given) and the
/// deck's `.options vpi=` (relative to the deck), since a registration
/// decides how a model compiles (LRM 12.32.1).
/// Fails with `OsdiUnsupported` on any OSDI include (`pre_osdi`, `osdi`),
/// with `Library.loadVpiApp`'s error for an application, and with the
/// loader's error for a model that does not compile.
pub fn loadModels(io: std.Io, lib: *device.Library, session: std.mem.Allocator, nl: netlist.Netlist, origin: []const u8, apps: []const []const u8) !void {
    for (apps) |app| try lib.loadVpiApp(app);
    for (nl.deck.vpi_apps) |app| try lib.loadVpiApp(try deckRelative(session, origin, app));
    var paths: std.ArrayList([]const u8) = .empty;
    for (nl.deck.foreign) |f| switch (f.kind) {
        .data => {},
        .verilog_a, .verilog => try paths.append(session, try deckRelative(session, origin, f.path)),
        // ESPice compiles Verilog-A through VerA and never loads OSDI, so an
        // OSDI reference is an error rather than a guess at a substitute.
        .osdi_include, .pre_osdi => {
            if (!@import("builtin").is_test) std.log.err("netlist: {s} {s}: espice does not load OSDI binaries; load the Verilog-A source with .hdl \"model.va\"", .{ @tagName(f.kind), f.path });
            return error.OsdiUnsupported;
        },
    };
    if (paths.items.len == 0) return;
    try lib.load(io, paths.items);
}

/// `path` as written when absolute, else joined onto the directory of the
/// deck at `origin`, allocated in `session`.
fn deckRelative(session: std.mem.Allocator, origin: []const u8, path: []const u8) ![]const u8 {
    if (std.fs.path.isAbsolute(path)) return path;
    return std.fs.path.join(session, &.{ std.fs.path.dirname(origin) orelse ".", path });
}

/// Narrows the published outputs to the `.save` labels, in place. The
/// probes keep their order; a label nothing publishes is dropped.
fn keepSaved(scratch: std.mem.Allocator, probes: *[]u32, labels: *[][]const u8, saves: []const []const u8) !void {
    var wanted: std.StringHashMapUnmanaged(void) = .empty;
    for (saves) |s| try wanted.put(scratch, s, {});
    const p = probes.*;
    const l = labels.*;
    var kept: usize = 0;
    // One buffer for every lowercased label, not an allocation per label.
    var lower: std.ArrayList(u8) = .empty;
    for (p, l) |row, label| {
        try lower.resize(scratch, label.len);
        if (!wanted.contains(std.ascii.lowerString(lower.items, label))) continue;
        p[kept] = row;
        l[kept] = label;
        kept += 1;
    }
    probes.* = p[0..kept];
    labels.* = l[0..kept];
}

/// Builds the frozen circuit and deck data from `nl`, with its variants
/// (`.step`, `SWEEP`, `.alter`, Monte Carlo). `parse_arena` holds
/// construction scratch; `sim_arena` owns every published slice and must
/// outlive the result, as must `lib` when the deck uses loaded devices.
/// Caller owns the result and releases its circuits with `Prepared.deinit`.
pub fn build(lib: *const device.Library, sim_arena: std.mem.Allocator, parse_arena: std.mem.Allocator, nl: netlist.Netlist) !Prepared {
    const owned = try parse_arena.create(netlist.Netlist);
    owned.* = nl;
    return buildRun(lib, sim_arena, parse_arena, owned, .{});
}

// What `buildRun` can fail with, spelled out because it recurses: the
// errors of `buildOne` and of setting a point's live values, each logged
// with its card by the stage that owns it. A failed build releases every
// circuit it froze; its slices stay in the arenas until they die.
const BuildError = ErrorOf(buildOne) || ErrorOf(variants.setPoint);

fn ErrorOf(comptime f: anytype) type {
    return @typeInfo(@typeInfo(@TypeOf(f)).@"fn".return_type.?).error_union.error_set;
}

/// `build` for one run of the deck in `nl`, which must outlive the result
/// when the deck has an HSPICE optimization or a `.wcd` (`Prepared.tuner`
/// and `Prepared.wcd` borrow it).
/// A `run.point` sets `nl`'s live values first; they are nominal again
/// when a run with rebuilt points returns.
fn buildRun(lib: *const device.Library, sim_arena: std.mem.Allocator, parse_arena: std.mem.Allocator, nl: *netlist.Netlist, run: variants.Run) BuildError!Prepared {
    var stack: std.ArrayList(netlist.expr.Val) = .empty;
    if (run.point) |pt| try variants.setPoint(parse_arena, &stack, nl, pt);
    var one = try buildOne(lib, sim_arena, parse_arena, nl, run);
    errdefer one.prepared.deinit();
    // Points the plan could not hold as rows, each its own run. A rebuilt
    // `.alter` netlist has no `.alter` of its own and a point run plans no
    // rebuilds, so this recursion is at most two runs deep.
    const runs = try sim_arena.alloc(Prepared, one.rebuilds.len);
    var built: usize = 0;
    errdefer for (runs[0..built]) |*r| r.deinit();
    for (one.rebuilds, runs) |r, *dst| {
        dst.* = try buildRun(lib, sim_arena, parse_arena, r.nl, r.run);
        built += 1;
    }
    for (one.rebuilds) |r| if (r.run.point != null) {
        try variants.setPoint(parse_arena, &stack, nl, .{ .label = "", .axis = 0, .live = nl.live.nominal });
        break;
    };
    one.prepared.runs = runs;
    return one.prepared;
}

/// One run without its rebuilt points: the prepared run and the runs its
/// plan asks for. `CircuitTooLarge` when the deck has more analysis cards
/// than a u32 job index can fan out.
fn buildOne(lib: *const device.Library, sim_arena: std.mem.Allocator, parse_arena: std.mem.Allocator, nl: *netlist.Netlist, run: variants.Run) !struct { prepared: Prepared, rebuilds: []const variants.Rebuild } {
    if (nl.deck.analyses.len > (std.math.maxInt(u32) - 1) / 3) return error.CircuitTooLarge;
    const deck_opts = try analyses.deckOptions(nl.deck.config, nl.deck.dialect);
    var b = try Builder.init(sim_arena, lib);
    var compiled_ok = false;
    errdefer if (!compiled_ok) b.deinit();

    // `.options tnom` is a model-card default (b4set.c:1950): models read it
    // as they are derived, before the pattern freezes.
    b.nom_temp_c = deck_opts.tnom_c;
    b.tol = deck_opts.tol;

    if (nl.deck.draw_seed and run.point == null and !variants.any(nl.deck)) try variants.drawOnce(parse_arena, nl);
    var nb = try builder.NetBuilder.init(parse_arena, &b, nl.*);
    try nb.build();

    // The card table outlives the Builder; its names live on the parse arena.
    const cards = try sim_arena.dupe(requests.CardRef, b.cards.items);
    for (cards) |*c| {
        c.name = try sim_arena.dupe(u8, c.name);
    }
    var perm: ?[]const u32 = null;
    var circuit = try b.compilePerm(&perm);
    compiled_ok = true;
    errdefer circuit.deinit();
    try frozenIndices(parse_arena, cards, &circuit);
    const card_names = try requests.CardNames.init(sim_arena, cards);

    var out = try nb.publish(sim_arena, &circuit, perm);
    if (nl.deck.saves.len > 0) try keepSaved(parse_arena, &out.probes, &out.probe_labels, nl.deck.saves);

    // Analysis nets to circuit rows.
    const cards_rows = try parse_arena.dupe(netlist.Analysis, nl.deck.analyses);
    for (cards_rows) |*a| {
        a.pos = nb.frozenRow(a.pos);
        if (namesNoOutput(a.*)) a.pos = out.output_node;
        a.neg = nb.frozenRow(a.neg);
        for (&a.ports) |*p| p.* = nb.frozenRow(p.*);
    }
    var tuner: ?variants.Tuner = null;
    // Variation groups for the mismatch and sensitivity cards: the deck's
    // `.variation` block, else the distribution calls in its models and
    // parameters (a PDK's mismatch), else `dcmatch`'s Pelgrom-or-unit
    // fallback, said aloud.
    var card_opts = deck_opts;
    for (nl.deck.analyses) |a| switch (a.kind) {
        .dcmatch, .acmatch, .dcsens => {
            if (nl.deck.variations.len == 0 and nl.live.site_keys.len == 0) {
                if (!@import("builtin").is_test) std.log.warn("variants: {s}: no .variation block and no agauss/gauss/unif/aunif/limit in a model or parameter, so each parameter takes its Pelgrom sigma, or unit variance without one", .{a.line});
                break;
            }
            var planner = try variants.Planner.init(lib, sim_arena, parse_arena, nl, &circuit, cards, "");
            card_opts.variations = if (nl.deck.variations.len != 0) try planner.variations() else try planner.siteVariations();
            if (nl.deck.variations.len == 0 and card_opts.variations.labels.len == 0 and !@import("builtin").is_test)
                std.log.warn("variants: {s}: no distribution call moves a device parameter (a PDK corner without its mismatch switch?), so the mismatch is zero", .{a.line});
            break;
        },
        else => {},
    };
    var aging: ?mosra.Plan = null;
    if (nl.deck.mosra) |m| {
        if (run.point != null or run.prefix.len != 0 or variants.any(nl.deck)) {
            if (!@import("builtin").is_test) std.log.err("netlist: .mosra together with .step, .alter or an analysis sweep: {s}", .{m.line});
            return error.UnsupportedCard;
        }
        aging = try mosra.plan(sim_arena, parse_arena, nl, &nb, &circuit, cards, deck_opts.temp_c orelse 27);
    }
    const plan: variants.Plan = if (aging) |a| .{ .variants = a.variants, .fanout = a.fanout } else if (run.point) |pt| blk: {
        var planner = try variants.Planner.init(lib, sim_arena, parse_arena, nl, &circuit, cards, "");
        var own = pt;
        own.label = run.prefix;
        try planner.add(own);
        break :blk .{ .variants = try planner.table(), .fanout = .{ .global = .{ .count = 1, .nominal = false }, .only = run.card } };
    } else if (run.prefix.len != 0 or variants.any(nl.deck)) blk: {
        var planner = try variants.Planner.init(lib, sim_arena, parse_arena, nl, &circuit, cards, run.prefix);
        const planned = try variants.plan(&planner);
        if (nl.deck.optimize != null) tuner = try variants.Tuner.init(planner);
        break :blk planned;
    } else .{};
    // A planner of its own: the plan's left the live values nominal.
    const wcd = if (nl.deck.wcd != null and run.point == null and run.prefix.len == 0) try variants.WorstCase.init(try variants.Planner.init(lib, sim_arena, parse_arena, nl, &circuit, cards, "")) else null;
    const tuning = if (nl.tunable.len != 0 and run.point == null and run.prefix.len == 0) try variants.Tuning.init(core.gpa, lib, sim_arena, parse_arena, nl, &circuit, cards) else null;
    errdefer if (tuning) |t| t.deinit();
    if (plan.variants.refs.len != 0 or tuning != null or wcd != null) try circuit.countParams();
    card_opts.sample = nl.deck.sample;
    card_opts.card_names = card_names;
    const queries = try analyses.queries(sim_arena, cards_rows, circuit.n, false, out.bindings, cards, card_opts, plan.fanout);
    if (aging) |*a| a.mosra.tran = for (queries) |q| {
        if (q == .tran and q.tran.tol.variant == null) break q.tran;
    } else {
        if (!@import("builtin").is_test) std.log.err("netlist: .mosra needs a .tran to stress the devices: {s}", .{nl.deck.mosra.?.line});
        return error.UnsupportedCard;
    };
    const ov = try opVars(sim_arena, parse_arena, lib, &circuit, cards, out.probes, out.probe_labels, nl, queries);
    return .{ .rebuilds = plan.rebuilds, .prepared = .{ .circuit = circuit, .tuner = tuner, .tuning = tuning, .wcd = wcd, .sss = try variants.sssCards(sim_arena, nl.deck), .deck = .{
        .probes = out.probes,
        .probe_labels = out.probe_labels,
        .source_node = out.source_node,
        .source_branch = out.source_branch,
        .output_node = out.output_node,
        .ac_drive = out.ac_drive,
        .title = try sim_arena.dupe(u8, nl.deck.title),
        .n_devices = nl.deviceCount(),
        .ic = try nodeRows(sim_arena, &nb, nl.deck.ic),
        .nodeset = try nodeRows(sim_arena, &nb, nl.deck.nodeset),
        .uic_ic = out.uic_ic,
        .deck_tol = deck_opts.tol,
        .deck_temp = deck_opts.temp_c,
        .deck_method = deck_opts.method,
        .queries = queries,
        .mosra = if (aging) |a| a.mosra else null,
        .variants = plan.variants,
        .bindings = out.bindings,
        .cards = cards,
        .card_names = card_names,
        .ac_overrides = try acOverrides(sim_arena, cards, nb.ac_res.items(.name), nb.ac_res.items(.value)),
        .measures = try measures(sim_arena, nl.deck.measures),
        .save_op = if (nl.deck.save_op) |s| blk: {
            var own = s;
            if (s.file) |file| own.file = try sim_arena.dupe(u8, file);
            break :blk own;
        } else null,
        .stims = try stims(sim_arena, nl.deck.stims),
        .soa = if (deck_opts.soa) .{ .max_warns = deck_opts.max_warns, .limits = try soaLimits(sim_arena, nl, &nb, &circuit) } else .{},
        .opvars = ov.vars,
        .opvar_probes = ov.probes,
        .opvar_labels = ov.labels,
        .emir_ports = try portCounts(sim_arena, lib, &circuit),
        .em_recovery = deck_opts.em_recovery,
        .card_models = try cardModels(sim_arena, parse_arena, nl, cards),
        .emir_file = if (nl.deck.emir_file) |f| try sim_arena.dupe(u8, f) else null,
    } } };
}

/// The device operating-point variables (`@m1[gm]`, #236) the deck's
/// `.save`, `.probe` and `.meas` cards name, and what records them:
/// `probes` and `labels`, then each named instance's rows, unlabeled. A
/// variable is the model's own (VerA `op_vars`) or, for a MOSFET, a
/// terminal quantity (`core.OpVar.Quantity`). A name that is neither, or no
/// `OpVarName` of a device in the deck, is `UnsupportedCard`, logged with
/// the variables there are. So is one under a `.dc` that sweeps the
/// temperature or a device of its type: the instance would be evaluated at
/// the sweep's last state. HSPICE's `.probe op` adds every variable of
/// every device (`probeAll`).
// ponytail: `.probe op` is a column per variable per device at every point
// of every .op, .dc and .tran; narrow it to the .op if a large circuit's
// transient runs out of memory.
fn opVars(
    sim_arena: std.mem.Allocator,
    scratch: std.mem.Allocator,
    lib: *const device.Library,
    circuit: *const device.Circuit,
    cards: []const requests.CardRef,
    probes: []const u32,
    labels: []const []const u8,
    nl: *const netlist.Netlist,
    queries: []const Job,
) !struct { vars: []const core.OpVar, probes: []const u32, labels: []const []const u8 } {
    const dialect = nl.deck.dialect;
    var names: std.StringArrayHashMapUnmanaged(void) = .empty;
    for (nl.deck.opvar_saves) |s| try names.put(scratch, s, {});
    for (nl.deck.measures) |m| for ([_]core.MeasureClause{ m.first, m.second }) |c| {
        for ([_][]const u8{ c.vec, c.vec2 }) |v| if (netlist.isOpVar(dialect, v)) try names.put(scratch, v, {});
        for ([_][]const core.MeasureOp{ c.ops, c.ops2 }) |ops| for (ops) |op| if (op == .vector and netlist.isOpVar(dialect, op.vector)) {
            try names.put(scratch, op.vector, {});
        };
    };
    if (names.count() == 0 and !nl.deck.probe_op) return .{ .vars = &.{}, .probes = &.{}, .labels = &.{} };
    // The terminal quantities are a MOSFET's (d, g, s, b), whose letter a
    // Spectre card's name need not carry; each maps to whether its card
    // is p-type, as `.mosra` reads it.
    // ponytail: the card kind only; a `type=-1` on an nmos or Verilog-A
    // module card reads as n-type. Read the bound model's polarity if a
    // deck spells it that way.
    var mosfets: std.StringHashMapUnmanaged(bool) = .empty;
    for (nl.bucket('m')) |e| {
        const d = nl.device(e);
        try mosfets.put(scratch, d.name, if (d.model) |m| std.ascii.eqlIgnoreCase(m.kind, "pmos") else false);
    }
    var list: std.ArrayList(core.OpVar) = .empty;
    for (names.keys()) |name| try list.append(sim_arena, try opVar(sim_arena, lib, circuit, cards, &mosfets, name));
    if (nl.deck.probe_op) for (cards) |ref| try probeAll(sim_arena, lib, circuit, ref, mosfets.get(ref.name), &names, &list);
    const vars = list.items;
    var rows: std.ArrayList(u32) = .empty;
    try rows.appendSlice(sim_arena, probes);
    var row_labels: std.ArrayList([]const u8) = .empty;
    try row_labels.appendSlice(sim_arena, labels);
    for (vars) |v| {
        const b = circuit.batches[v.batch];
        for (b.hooks.scatter_tapes(b.ctx).rows[@as(usize, v.index) * b.n_u ..][0..b.n_u]) |row| {
            // A ground unknown's row is `n`.
            if (row >= circuit.n or std.mem.indexOfScalar(u32, rows.items[probes.len..], row) != null) continue;
            try rows.append(sim_arena, row);
            try row_labels.append(sim_arena, "");
        }
    }
    for (queries) |q| if (q == .dc) for ([_]?requests.Dc.SweepTarget{ q.dc.target, q.dc.target2 }) |target| {
        const swept = target orelse continue;
        for (vars) |v| if (swept == .temp or swept.device.type == circuit.batch_types[v.batch]) {
            if (!@import("builtin").is_test) std.log.err("netlist: {s} along a .dc that sweeps the temperature or a device of its type is not supported (#236)", .{v.label});
            return error.UnsupportedCard;
        };
    };
    return .{ .vars = vars, .probes = rows.items, .labels = row_labels.items };
}

/// `core.OpVar.Quantity`'s names, for the error that lists them.
const opvar_names = blk: {
    var s: []const u8 = "";
    for (@typeInfo(core.OpVar.Quantity).@"enum".field_names) |n| s = s ++ " " ++ n;
    break :blk s[1..];
};

/// HSPICE's MOSFET element templates [HSPICE Reference Manual: Elements
/// and Device Models, "MOSFET Output Templates"]: `lx<N>`/`lv<N>` to the
/// variable it reads. Another number refuses.
// ponytail: lx19/lx20 (HSPICE's ∂Qg/∂Vd and ∂Qg/∂Vs, -cgd and -cgs) need a
// sign this table cannot carry; add one if a deck reads them.
const hspice_templates = std.StaticStringMap([]const u8).initComptime(.{
    .{ "lx1", "vbs" }, .{ "lx2", "vgs" }, .{ "lx3", "vds" },  .{ "lx4", "id" },
    .{ "lx7", "gm" },  .{ "lx8", "gds" }, .{ "lx9", "gmbs" }, .{ "lx18", "cgg" },
    .{ "lv9", "vth" }, .{ "lv10", "vdsat" },
});

/// A device operating-point variable's name, split: ngspice
/// `@<card>[<variable>]`, Spectre `<device>:<variable>`, or HSPICE
/// `lx<N>(<device>)`/`lv<N>(<device>)` through `hspice_templates`. A
/// `<device>` (`path`) is the deck's own name (`x1.mn1`), which the card's
/// flattened name leads with its letter (`m.x1.mn1`).
const OpVarName = struct {
    device: []const u8,
    variable: []const u8,
    path: bool,

    /// Null for a name of none of these forms.
    fn of(name: []const u8) ?OpVarName {
        if (name.len != 0 and name[0] == '@') {
            const open = std.mem.indexOfScalar(u8, name, '[') orelse return null;
            if (open < 2 or name[name.len - 1] != ']') return null;
            return .{ .device = name[1..open], .variable = name[open + 1 .. name.len - 1], .path = false };
        }
        if (std.mem.indexOfScalar(u8, name, ':')) |colon| {
            if (colon == 0 or colon + 1 == name.len) return null;
            return .{ .device = name[0..colon], .variable = name[colon + 1 ..], .path = true };
        }
        const open = std.mem.indexOfScalar(u8, name, '(') orelse return null;
        if (open + 2 >= name.len or name[name.len - 1] != ')') return null;
        const variable = hspice_templates.get(name[0..open]) orelse return null;
        return .{ .device = name[open + 1 .. name.len - 1], .variable = variable, .path = true };
    }

    /// Whether `card`, a flattened card name, is the device `n` names.
    fn matches(n: OpVarName, card: []const u8) bool {
        if (std.ascii.eqlIgnoreCase(card, n.device)) return true;
        return n.path and card.len == n.device.len + 2 and card[1] == '.' and std.ascii.eqlIgnoreCase(card[2..], n.device);
    }
};

/// `name`, an `OpVarName`, resolved to its instance; `mosfets` maps the
/// MOSFETs' card names to whether each is p-type. See `opVars` for the
/// refusals.
fn opVar(arena: std.mem.Allocator, lib: *const device.Library, circuit: *const device.Circuit, cards: []const requests.CardRef, mosfets: *const std.StringHashMapUnmanaged(bool), name: []const u8) !core.OpVar {
    const split = OpVarName.of(name) orelse {
        if (!@import("builtin").is_test) std.log.err("netlist: {s}: a device operating-point variable is @<device>[<variable>], <device>:<variable>, or lx1-lx4, lx7-lx9, lx18, lv9 or lv10 (<device>)", .{name});
        return error.UnsupportedCard;
    };
    const ref = for (cards) |c| {
        if (split.matches(c.name)) break c;
    } else {
        if (!@import("builtin").is_test) std.log.err("netlist: {s}: no device {s}", .{ name, split.device });
        return error.UnsupportedCard;
    };
    return opVarOf(arena, lib, circuit, ref, mosfets.get(ref.name), name, split.variable);
}

/// `variable` of instance `ref`, published as `label`; `pmos_card` is null
/// unless its card is a MOSFET's, and then whether it is p-type.
fn opVarOf(arena: std.mem.Allocator, lib: *const device.Library, circuit: *const device.Circuit, ref: requests.CardRef, pmos_card: ?bool, label: []const u8, variable: []const u8) !core.OpVar {
    const batch = std.mem.indexOfScalar(core.DeviceType, circuit.batch_types, ref.type).?;
    const hooks = circuit.batches[batch].hooks;
    const ports = lib.vtable(ref.type).num_ports;
    const mosfet = pmos_card != null and ports >= 3;
    if (hooks.local_eval == null) return refuseOpVar(arena, label, ref.name, variable, &.{}, false);
    var v: core.OpVar = .{ .label = try arena.dupe(u8, label), .batch = @intCast(batch), .index = ref.index, .ports = ports, .pmos = pmos_card orelse false };
    // The model's own variable first: BSIM-CMG's `ids` is its own.
    for (hooks.op_vars, 0..) |own, k| {
        if (std.ascii.eqlIgnoreCase(own, variable)) {
            v.own = @intCast(k);
            break;
        }
    } else {
        const terminal = if (mosfet) std.meta.stringToEnum(core.OpVar.Quantity, variable) else null;
        v.quantity = terminal orelse return refuseOpVar(arena, label, ref.name, variable, hooks.op_vars, mosfet);
    }
    return v;
}

/// Appends to `list` every operating-point variable of instance `ref`
/// that `names` does not hold, as `@<card>[<variable>]` (HSPICE `.probe
/// op`): the model's own, then for a MOSFET (`pmos_card`, as `opVarOf`
/// reads it) the terminal quantities it does not shadow, without the
/// alias `ids` and, with three terminals, the bulk ones.
fn probeAll(arena: std.mem.Allocator, lib: *const device.Library, circuit: *const device.Circuit, ref: requests.CardRef, pmos_card: ?bool, names: *const std.StringArrayHashMapUnmanaged(void), list: *std.ArrayList(core.OpVar)) !void {
    const hooks = circuit.batches[std.mem.indexOfScalar(core.DeviceType, circuit.batch_types, ref.type).?].hooks;
    if (hooks.local_eval == null) return;
    const ports = lib.vtable(ref.type).num_ports;
    const bulk = [_]core.OpVar.Quantity{ .ib, .gmbs, .cgb, .vbs };
    var all: std.ArrayList([]const u8) = .empty;
    try all.appendSlice(arena, hooks.op_vars);
    if (pmos_card != null and ports >= 3) for (std.enums.values(core.OpVar.Quantity)) |q| {
        if (q == .ids or (ports == 3 and std.mem.indexOfScalar(core.OpVar.Quantity, &bulk, q) != null)) continue;
        for (hooks.op_vars) |own| {
            if (std.ascii.eqlIgnoreCase(own, @tagName(q))) break;
        } else try all.append(arena, @tagName(q));
    };
    for (all.items) |variable| {
        const label = try arena.print("@{s}[{s}]", .{ ref.name, variable });
        if (names.contains(std.ascii.lowerString(label, label))) continue;
        try list.append(arena, try opVarOf(arena, lib, circuit, ref, pmos_card, label, variable));
    }
}

/// Logs that `card` has no variable `variable`, listing the model's own
/// variables `own` and, for a MOSFET, the terminal ones; `UnsupportedCard`.
fn refuseOpVar(arena: std.mem.Allocator, name: []const u8, card: []const u8, variable: []const u8, own: []const []const u8, mosfet: bool) error{ OutOfMemory, UnsupportedCard } {
    if (!@import("builtin").is_test) {
        const listed = std.mem.join(arena, " ", own) catch return error.OutOfMemory;
        const sep = if (listed.len != 0 and mosfet) " " else "";
        std.log.err("netlist: {s}: {s} has no variable '{s}'; its variables are: {s}{s}{s}", .{ name, card, variable, listed, sep, if (mosfet) opvar_names else "" });
    }
    return error.UnsupportedCard;
}

/// Renumbers each card's `index` from its add ordinal to its instance's
/// place in the frozen batch, which a mixed batch partitions
/// (`Hooks.original_ids`): the index every tape, hook and `ParamRef` reads.
fn frozenIndices(scratch: std.mem.Allocator, cards: []requests.CardRef, circuit: *const device.Circuit) !void {
    for (circuit.batches, circuit.batch_types) |b, t| {
        const ids: []const u32 = if (b.hooks.original_ids) |original| original(b.ctx) else &.{};
        if (ids.len == 0) continue;
        std.debug.assert(ids.len == b.count);
        const at = try scratch.alloc(u32, ids.len);
        for (ids, 0..) |id, k| at[id] = @intCast(k);
        for (cards) |*c| if (c.type == t) {
            c.index = at[c.index];
        };
    }
}

/// `Deck.card_models`: the `.model` card each card names, by its name, for
/// `--emir` (#237). A binned card's is its bin's.
fn cardModels(sim_arena: std.mem.Allocator, scratch: std.mem.Allocator, nl: *const netlist.Netlist, cards: []const requests.CardRef) !requests.CardNames {
    var by_name: std.StringHashMapUnmanaged([]const u8) = .empty;
    for (nl.order) |e| {
        const dev = nl.device(e);
        if (dev.model) |m| try by_name.put(scratch, dev.name, m.name);
    }
    var named: std.ArrayList(requests.CardRef) = .empty;
    for (cards) |c| if (by_name.get(c.name)) |m| try named.append(scratch, .{ .type = c.type, .index = c.index, .name = try sim_arena.dupe(u8, m) });
    return requests.CardNames.init(sim_arena, named.items);
}

/// `Deck.emir_ports`: each batch's port count, from its type's vtable.
fn portCounts(arena: std.mem.Allocator, lib: *const device.Library, circuit: *const device.Circuit) ![]const u32 {
    const ports = try arena.alloc(u32, circuit.batch_types.len);
    for (ports, circuit.batch_types) |*p, t| p.* = lib.vtable(t).num_ports;
    return ports;
}

/// `.stim` cards with their strings copied out of the parse arena.
fn stims(arena: std.mem.Allocator, cards: []const core.Stim) ![]const core.Stim {
    const out = try arena.dupe(core.Stim, cards);
    for (out) |*s| {
        if (s.file) |file| s.file = try arena.dupe(u8, file);
        s.dataname = try arena.dupe(u8, s.dataname);
        s.signals = try measures(arena, s.signals);
        const nodes = try arena.dupe([2][]const u8, s.nodes);
        for (nodes) |*pair| for (pair) |*n| {
            n.* = try arena.dupe(u8, n.*);
        };
        s.nodes = nodes;
        s.times = try arena.dupe(f64, s.times);
    }
    return out;
}

/// `.ic`-style values on circuit rows. A node no device touches is dropped,
/// like every other unresolvable directive name.
fn nodeRows(arena: std.mem.Allocator, nb: *const builder.NetBuilder, items: []const netlist.Ic) ![]const Ic {
    var out: std.ArrayList(Ic) = .empty;
    for (items) |item| {
        const row = nb.frozenRow(item.net.index());
        if (row == NO_NODE or row == GROUND) continue;
        try out.append(arena, .{ .node = row, .value = item.value });
    }
    return out.items;
}

/// A model-card SOA limit (ngspice's names) of card letter `card`: held
/// across terminals `pos` and `neg`, by magnitude unless `abs` is false.
const SoaKey = struct { card: u8, key: []const u8, quantity: []const u8, pos: u8, neg: u8, abs: bool = true };

/// MOS pins d g s b, BJT c b e, diode a k; a diode's `bv_max` holds the
/// reverse voltage v(k) - v(a), as ngspice's diosoachk.c does.
const soa_keys = [_]SoaKey{
    .{ .card = 'm', .key = "vgs_max", .quantity = "vgs", .pos = 1, .neg = 2 },
    .{ .card = 'm', .key = "vgd_max", .quantity = "vgd", .pos = 1, .neg = 0 },
    .{ .card = 'm', .key = "vgb_max", .quantity = "vgb", .pos = 1, .neg = 3 },
    .{ .card = 'm', .key = "vds_max", .quantity = "vds", .pos = 0, .neg = 2 },
    .{ .card = 'm', .key = "vbs_max", .quantity = "vbs", .pos = 3, .neg = 2 },
    .{ .card = 'm', .key = "vbd_max", .quantity = "vbd", .pos = 3, .neg = 0 },
    .{ .card = 'q', .key = "vbe_max", .quantity = "vbe", .pos = 1, .neg = 2 },
    .{ .card = 'q', .key = "vbc_max", .quantity = "vbc", .pos = 1, .neg = 0 },
    .{ .card = 'q', .key = "vce_max", .quantity = "vce", .pos = 0, .neg = 2 },
    .{ .card = 'd', .key = "bv_max", .quantity = "vr", .pos = 1, .neg = 0, .abs = false },
    .{ .card = 'r', .key = "bv_max", .quantity = "v", .pos = 0, .neg = 1 },
    .{ .card = 'c', .key = "bv_max", .quantity = "v", .pos = 0, .neg = 1 },
};

/// One `core.SoaLimit` per `soa_keys` limit the model card of an M, Q, D,
/// R or C card states, each terminal named by its result column as
/// `NetBuilder.publish` labels it. A model card that states none takes
/// the `soalimits` file's limits for the longest name that is its own or
/// a prefix of it (`Netlist.soa_ratings`), so a card value is never
/// shadowed. A limit that is no number, or names a terminal the card does
/// not connect, is skipped.
fn soaLimits(arena: std.mem.Allocator, nl: *const netlist.Netlist, nb: *const builder.NetBuilder, circuit: *const device.Circuit) ![]const core.SoaLimit {
    var out: std.ArrayList(core.SoaLimit) = .empty;
    for ("mqdrc") |card| for (nl.bucket(card)) |e| {
        const d = nl.device(e);
        const m = d.model orelse continue;
        const stated = for (soa_keys) |k| {
            if (k.card == card and cardValue(m.kv, k.key) != null) break true;
        } else false;
        // ponytail: a scan of the file's rows per device; a map by model
        // name when a kit's file holds thousands.
        var rated: ?[]const u8 = null;
        if (!stated) for (nl.soa_ratings) |r| {
            if (std.ascii.startsWithIgnoreCase(m.name, r.model) and (rated == null or r.model.len > rated.?.len)) rated = r.model;
        };
        for (soa_keys) |k| {
            if (k.card != card or @max(k.pos, k.neg) >= d.pins.len) continue;
            const value: ?f64 = if (rated) |model| rating(nl.soa_ratings, model, k.key) else if (cardValue(m.kv, k.key)) |v| v.number() else null;
            const limit = value orelse continue;
            var columns: [2][]const u8 = undefined;
            for (&columns, [2]u8{ k.pos, k.neg }) |*column, pin| {
                const row = nb.frozenRow(d.pins[pin].index());
                column.* = if (row == NO_NODE or row == GROUND) "" else try std.mem.concat(arena, u8, &.{ "v(", circuit.nodeName(row), ")" });
            }
            try out.append(arena, .{ .name = try arena.dupe(u8, d.name), .quantity = k.quantity, .key = k.key, .limit = limit, .pos = columns[0], .neg = columns[1], .abs = k.abs });
        }
    };
    return out.items;
}

/// The value model-card pairs `kv` give `key`; null when they give none.
fn cardValue(kv: []const netlist.Kv, key: []const u8) ?netlist.Value {
    for (kv) |item| if (std.mem.eql(u8, item.key, key)) return item.value;
    return null;
}

/// The `soalimits` file's `key` limit for `model`; null when it gives none.
fn rating(ratings: []const netlist.SoaRating, model: []const u8, key: []const u8) ?f64 {
    for (ratings) |r| if (std.mem.eql(u8, r.model, model) and std.mem.eql(u8, r.key, key)) return r.limit;
    return null;
}

/// `.meas` cards with their strings copied out of the parse arena.
fn measures(arena: std.mem.Allocator, cards: []const core.Measure) ![]const core.Measure {
    const out = try arena.dupe(core.Measure, cards);
    for (out) |*m| {
        m.name = try arena.dupe(u8, m.name);
        m.expr = try measureOps(arena, m.expr);
        m.check.expect = try arena.dupe([2]f64, m.check.expect);
        for ([_]*core.MeasureClause{ &m.first, &m.second }) |c| {
            c.vec = try arena.dupe(u8, c.vec);
            c.vec2 = try arena.dupe(u8, c.vec2);
            c.ops = try measureOps(arena, c.ops);
            c.ops2 = try measureOps(arena, c.ops2);
            const refs = try arena.dupe(core.MeasureRef, c.refs);
            for (refs) |*ref| ref.expr = try measureOps(arena, ref.expr);
            c.refs = refs;
        }
    }
    return out;
}

fn measureOps(arena: std.mem.Allocator, ops: []const core.MeasureOp) ![]const core.MeasureOp {
    const out = try arena.dupe(core.MeasureOp, ops);
    for (out) |*op| if (op.* == .vector) {
        op.vector = try arena.dupe(u8, op.vector);
    };
    return out;
}

/// Cards whose output is `Deck.output_node` because they name none. HSPICE's
/// `.snxf v(out) ...` names its own.
fn namesNoOutput(a: netlist.Analysis) bool {
    return a.kind == .pac or (a.kind == .pxf and !a.sn) or a.kind == .disto or a.kind == .hbac;
}

/// Frozen-circuit node row by label, for cards appended after the build.
const NodeIndex = struct {
    map: std.StringHashMapUnmanaged(u32),

    fn init(arena: std.mem.Allocator, circuit: *const device.Circuit) !NodeIndex {
        var map: std.StringHashMapUnmanaged(u32) = .empty;
        try map.ensureTotalCapacity(arena, circuit.n);
        for (0..circuit.n) |i| {
            const label = circuit.nodeName(@intCast(i));
            if (label.len != 0) map.putAssumeCapacity(label, @intCast(i));
        }
        return .{ .map = map };
    }

    /// Row of net `name`; NO_NODE when the deck never named it.
    pub fn node(self: NodeIndex, name: []const u8) u32 {
        return self.map.get(name) orelse NO_NODE;
    }
};

/// Resolves analysis cards appended to a built circuit. Topology and deck
/// settings stay fixed; only the queries are allocated, in `arena`. Any card
/// that is not an analysis, or a single-value `.temp`, is
/// `UnsupportedDirectiveMutation`; text holding no card is
/// `InvalidAnalysisArguments`.
pub fn resolveQueries(arena: std.mem.Allocator, prepared: *const Prepared, directive_text: []const u8) ![]const Job {
    const nodes: NodeIndex = try .init(arena, &prepared.circuit);
    const cards = try netlist.parseAnalyses(arena, directive_text, nodes);
    if (cards.len == 0) return error.InvalidAnalysisArguments;
    for (cards) |*a| if (namesNoOutput(a.*)) {
        a.pos = prepared.deck.output_node;
    };
    return analyses.queries(arena, cards, prepared.circuit.n, true, prepared.deck.bindings, prepared.deck.cards, .{
        .tol = prepared.deck.deck_tol,
        .method = prepared.deck.deck_method,
        .temp_c = prepared.deck.deck_temp,
        .card_names = prepared.deck.card_names,
    }, .{});
}

/// Each `ac=` resistance as a (type, instance, parameter) override, so every
/// analysis clone binds its own parameter pointer.
fn acOverrides(
    arena: std.mem.Allocator,
    cards: []const requests.CardRef,
    names: []const []const u8,
    values: []const f64,
) ![]const core.AcOverride {
    const overrides = try arena.alloc(core.AcOverride, names.len);
    for (names, values, overrides) |name, value, *override| {
        for (cards) |card| {
            if (!std.mem.eql(u8, card.name, name)) continue;
            override.* = .{ .type = card.type, .index = card.index, .param_name = "r", .value = value };
            break;
        } else return error.InvalidAcOverride;
    }
    return overrides;
}

test parseDialect {
    try std.testing.expectEqual(Dialect.spectre, parseDialect("scs").?);
    try std.testing.expectEqual(Dialect.ngspice, parseDialect("ngspice").?);
    // Only the listed names and aliases, no prefixes.
    try std.testing.expectEqual(null, parseDialect("h"));
    try std.testing.expectEqual(null, parseDialect(""));
}

test keepSaved {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    // A label past 64 bytes is still matched (it used to be dropped).
    const long = "v(xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx)";
    var probe_buf = [_]u32{ 1, 2, 3 };
    var label_buf = [_][]const u8{ "v(A)", long, "v(c)" };
    var probes: []u32 = &probe_buf;
    var labels: [][]const u8 = &label_buf;
    try keepSaved(arena.allocator(), &probes, &labels, &.{ long, "v(a)", "v(nowhere)" });
    try std.testing.expectEqualSlices(u32, &.{ 1, 2 }, probes);
    try std.testing.expectEqualStrings("v(A)", labels[0]);
    try std.testing.expectEqualStrings(long, labels[1]);
    try keepSaved(arena.allocator(), &probes, &labels, &.{"v(none)"});
    try std.testing.expectEqual(@as(usize, 0), probes.len);
}

test namesNoOutput {
    try std.testing.expect(namesNoOutput(.{ .kind = .pac, .args = &.{} }));
    try std.testing.expect(namesNoOutput(.{ .kind = .pxf, .args = &.{} }));
    try std.testing.expect(!namesNoOutput(.{ .kind = .pxf, .args = &.{}, .sn = true }));
    try std.testing.expect(!namesNoOutput(.{ .kind = .ac, .args = &.{} }));
}

test {
    _ = @import("tests/builder.zig");
    _ = @import("tests/prepared.zig");
    _ = @import("tests/models.zig");
    _ = @import("tests/variants.zig");
    _ = variants;
    _ = analyses;
    _ = mosra;
}
