//! Netlist text to a flat, subcircuit-expanded nets-by-devices hypergraph,
//! plus models and deck cards. Three walks over the logical lines: declarations
//! (subcircuit ranges, `.param`, `.model`, directives), devices (an `X` card
//! re-reads its subcircuit's lines under a frame), then analysis nets. Values
//! are rows of flat tables; an expression that does not fold stays postfix.
//!
//! Cites ngspice (inpcom.c, inp.c recifeval, subckt.c modtranslate,
//! INPgetModBin) for the ngspice dialect, the HSPICE manuals ([SA], [CR])
//! for the HSPICE cards, and dev/research/{ngspice,hspice,spectre}-syntax.md
//! (§ and rule ids) for the grammar; tests/frontend/grammar/ngspice.peg, hspice.peg
//! and spectre.peg are its oracle.
//!
//! Contents:
//! - `pub const Dialect` (card_rows.zig)
//! - `pub const VertexId` (card_rows.zig)
//! - `pub const EdgeId` (card_rows.zig)
//! - `pub const Kind` (card_rows.zig)
//! - `pub const ground` (card_rows.zig)
//! - `pub const none` (card_rows.zig)
//! - `pub const Error` (card_rows.zig)
//! - `pub const Span` (card_rows.zig)
//! - `pub const Value` (card_rows.zig)
//! - `pub const Group` (card_rows.zig)
//! - `pub const Kv` (card_rows.zig)
//! - `pub const Net` (card_rows.zig)
//! - `pub const Device` (card_rows.zig)
//! - `pub const Hypergraph` (card_rows.zig)
//! - `pub const Model` (card_rows.zig)
//! - `pub const StepTarget` (card_rows.zig)
//! - `pub const Step` (card_rows.zig)
//! - `pub const Monte` (card_rows.zig)
//! - `pub const Sweep` (card_rows.zig)
//! - `pub const Optimize` (card_rows.zig)
//! - `pub const Data` (card_rows.zig)
//! - `pub const Variation` (card_rows.zig)
//! - `pub const Live` (card_rows.zig)
//! - `pub const Analysis` (card_rows.zig)
//! - `pub const Config` (card_rows.zig)
//! - `pub const Ic` (card_rows.zig)
//! - `pub const ForeignKind` (card_rows.zig)
//! - `pub const Foreign` (card_rows.zig)
//! - `pub const DataFile` (card_rows.zig)
//! - `pub const Deck` (card_rows.zig)
//! - `pub const Mosra` (card_rows.zig)
//! - `pub const AppendModel` (card_rows.zig)
//! - `pub const Netlist` (netlist_view.zig)
//! - `pub const siteKey` (netlist_view.zig)
//! - `pub const isGround` (netlist_view.zig)
//! - `pub const isGroundName` (netlist_view.zig)
//! - `pub const nameIndex` (netlist_view.zig)
//! - `pub const lines, spectre_rewrite, source, expr`
//! - `pub fn parse(arena: Allocator, src: []const u8, dialect: Dialect) Error!Netlist`
//! - `pub fn parseLive(arena: Allocator, src: []const u8, dialect: Dialect, live: []const []const u8) (Error || error{UnknownParameter})!Netlist`
//! - `pub fn parseAnalyses(arena: Allocator, text: []const u8, lookup: anytype) (Error || error{UnsupportedDirectiveMutation})![]Analysis`

const std = @import("std");
const Allocator = std.mem.Allocator;
/// Logical lines and fields per dialect, re-exported for the frontend tests.
pub const lines = @import("lines.zig");
/// The Spectre statement rewrite, re-exported for the grammar oracle.
pub const spectre_rewrite = @import("spectre.zig");
/// Deck loading and include expansion, re-exported for the frontend tests.
pub const source = @import("source.zig");
const hypergraph = @import("hypergraph.zig");
/// The postfix `Value.expr` spans index, re-exported for the builder.
pub const expr = @import("expr.zig");
const measure = @import("measure.zig");
const pattern = @import("pattern.zig");
const core = @import("core");
const Name = core.Name;
const InternPool = core.InternPool;
const card_rows = @import("card_rows.zig");
const netlist_view = @import("netlist_view.zig");
const card_table = @import("card_table.zig");
const reader_frame = @import("reader_frame.zig");
const sweep_points = @import("sweep_points.zig");
const reader = @import("reader.zig");
const Reader = reader.Reader;
const assert = std.debug.assert;
const reader_run = @import("reader_run.zig");
const reader_walk = @import("reader_walk.zig");
const reader_models = @import("reader_models.zig");
const reader_directives = @import("reader_directives.zig");
const reader_sweeps = @import("reader_sweeps.zig");
const reader_live = @import("reader_live.zig");
const reader_values = @import("reader_values.zig");
const reader_subst = @import("reader_subst.zig");
const reader_measures = @import("reader_measures.zig");
const reader_nets = @import("reader_nets.zig");
const reader_devices = @import("reader_devices.zig");
const reader_behavioural = @import("reader_behavioural.zig");
const appendAnalysis = reader_frame.appendAnalysis;
const cardOf = card_table.cardOf;
const isSn = card_table.isSn;
const resolve = reader_frame.resolve;

/// Input syntax: ngspice, HSPICE or Spectre.
pub const Dialect = card_rows.Dialect;
/// A net: a row of `Netlist.graph`'s vertex table.
pub const VertexId = card_rows.VertexId;
/// A device: a row of `Netlist.graph`'s edge table, in card order.
pub const EdgeId = card_rows.EdgeId;
/// Analysis kind, shared with the query layer.
pub const Kind = card_rows.Kind;
/// Net 0; `0`, `gnd` and `ground` all name it.
pub const ground = card_rows.ground;
/// Missing id in every u32 index space here.
pub const none = card_rows.none;
/// `ParseError`: malformed card; `ModelBinNotFound`: no `.model nm.N` bin (card_rows.zig).
pub const Error = card_rows.Error;
/// Rows `start..start+len` of a flat table.
pub const Span = card_rows.Span;
/// One card value after parameter substitution.
pub const Value = card_rows.Value;
/// `name(args)`, held as one slice: `items[0]` is the name (a `.name` (card_rows.zig).
pub const Group = card_rows.Group;
/// `key=value`; a positional model value has an empty key.
pub const Kv = card_rows.Kv;
/// Vertex payload: the net's flattened name in `Netlist.pool`.
pub const Net = card_rows.Net;
/// Hyperedge payload; `model` is a `.model` row or `none`.
pub const Device = card_rows.Device;
/// Nets by devices; a device's pins are its member list, in terminal order.
pub const Hypergraph = card_rows.Hypergraph;
/// A `.model` card: `kind` is the type word (`nmos`, `d`, a VA module name).
pub const Model = card_rows.Model;
/// What a `.step` card or an HSPICE `SWEEP` varies.
pub const StepTarget = card_rows.StepTarget;
/// One swept quantity and its points, in sweep order.
pub const Step = card_rows.Step;
/// HSPICE `MONTE=n [FIRSTRUN=k]`: trials `first..first + n - 1`; or (card_rows.zig).
pub const Monte = card_rows.Monte;
/// An HSPICE sweep on one analysis card: the card runs once per point.
pub const Sweep = card_rows.Sweep;
/// HSPICE optimization [SA Ch.27]: `<card> SWEEP OPTIMIZE=name (card_rows.zig).
pub const Optimize = card_rows.Optimize;
/// An inline `.data` table: one target per column, `values` row-major.
pub const Data = card_rows.Data;
/// One Monte Carlo variation of a model or element parameter: HSPICE (card_rows.zig).
pub const Variation = card_rows.Variation;
/// Values a variant re-evaluates: device and model values that read a swept (card_rows.zig).
pub const Live = card_rows.Live;
/// An analysis card. `pos`/`neg` are the output `v(a[,b])` nets, `ports` the (card_rows.zig).
pub const Analysis = card_rows.Analysis;
/// `.options` and single-value `.temp` cards, in deck order.
pub const Config = card_rows.Config;
/// One `.ic v(net)=value` entry on a net some card names.
pub const Ic = card_rows.Ic;
/// What an include carries: OSDI, Verilog-A, digital Verilog or a data file.
pub const ForeignKind = card_rows.ForeignKind;
/// An HDL or OSDI include, path as written (relative to the deck).
pub const Foreign = card_rows.Foreign;
/// A `ForeignKind.data` file's contents, read by `prepare`.
pub const DataFile = card_rows.DataFile;
/// Deck data: everything that is not circuit topology.
pub const Deck = card_rows.Deck;
/// HSPICE `.mosra` options as read [SA Ch.29]; SimMode 0 or 2.
pub const Mosra = card_rows.Mosra;
/// `.appendmodel src MOSRA dst NMOS|PMOS`: MOSRA model `src` ages the (card_rows.zig).
pub const AppendModel = card_rows.AppendModel;
/// A parsed, flattened deck. Every slice lives in the parse arena.
pub const Netlist = netlist_view.Netlist;
/// A site key: `owner` mixed with `salt` (splitmix64 finalizer). Stable (netlist_view.zig).
pub const siteKey = netlist_view.siteKey;
/// True for a name that is ground in `dialect`: only `0` in Spectre, where (netlist_view.zig).
pub const isGround = netlist_view.isGround;
/// True for `0`, `gnd` and `ground`, case-insensitively: ground in the (netlist_view.zig).
pub const isGroundName = netlist_view.isGroundName;
/// Index of the first card name in `names` equal to `target`, byte for (netlist_view.zig).
pub const nameIndex = netlist_view.nameIndex;

/// Parses `src` in `dialect` into `arena`. The result borrows `src` (paths
/// and the title keep their original case) and the arena, which should be
/// an arena: nothing is freed piecewise. Fails per `Error`; partial
/// results stay in the arena.
pub fn parse(arena: Allocator, src: []const u8, dialect: Dialect) Error!Netlist {
    return parseLive(arena, src, dialect, &.{}) catch |err| switch (err) {
        error.UnknownParameter => unreachable,
        else => |e| return e,
    };
}

/// `parse`, with the global `.param` names in `live` kept live as a
/// swept name is (`Netlist.tunable`), so a later `setLive` can move them.
/// `UnknownParameter` for a name no global `.param` defines.
pub fn parseLive(arena: Allocator, src: []const u8, dialect: Dialect, live: []const []const u8) (Error || error{UnknownParameter})!Netlist {
    return switch (dialect) {
        inline else => |d| Reader(lines.Syntax(d)).run(arena, src, d, live),
    };
}

/// Parses analysis cards appended to a built circuit, resolving nets through
/// `lookup.node(name) u32`, which sees lowercased names. Any other line, and
/// a single-value `.temp`, is refused with `UnsupportedDirectiveMutation`.
/// The cards and their strings live in `arena`.
pub fn parseAnalyses(arena: Allocator, text: []const u8, lookup: anytype) (Error || error{UnsupportedDirectiveMutation})![]Analysis {
    const lower = try arena.alloc(u8, text.len);
    _ = lines.normalize(lower, text);
    var r: Reader(lines.ngspice) = try .init(arena, text, lower, .ngspice);
    r.global_scopes = .{&r.globals};
    var src: lines.ngspice.Lines = .{ .rest = lower, .arena = arena };
    var out: std.ArrayList(Analysis) = .empty;
    while (try src.next()) |line| {
        if (line.len == 0) continue;
        if (line[0] != '.') return error.UnsupportedDirectiveMutation;
        var f = lines.ngspice.Split.init(line);
        const card = cardOf(f.next().?[1..]) orelse return error.UnsupportedDirectiveMutation;
        if (card != .analysis) return error.UnsupportedDirectiveMutation;
        const args = try r.readArgs(&f);
        // A single `.temp` is deck configuration, fixed at build.
        if (card.analysis == .temp and args.len == 1) return error.UnsupportedDirectiveMutation;
        const first = out.items.len;
        // As readDirective reads a deck's card: its SWEEP tail and the line as written (#64).
        const swept = try r.readSweep(card.analysis, args);
        try appendAnalysis(arena, &out, .{ .kind = card.analysis, .args = swept.args, .sweep = swept.sweep, .line = r.written(line), .sn = isSn(line) });
        for (out.items[first..]) |*a| resolve(a, lookup);
    }
    return out.items;
}

// ---- private ----

test "measure values: empty text is a ParseError, not an index past the end" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var r: Reader(lines.ngspice) = try .init(arena.allocator(), "", "", .ngspice);
    r.global_scopes = .{&r.globals};
    try std.testing.expectError(error.ParseError, r.measureValue(""));
    try std.testing.expectEqual(@as(f64, 2e-3), try r.measureValue("2m"));
    try std.testing.expectEqual(@as(f64, 3), try r.measureValue("{1+2}"));
}

test {
    _ = lines;
    _ = source;
    _ = expr;
    _ = hypergraph;
    _ = card_rows;
    _ = netlist_view;
    _ = card_table;
    _ = reader_frame;
    _ = sweep_points;
    _ = reader;
    _ = reader_run;
    _ = reader_walk;
    _ = reader_models;
    _ = reader_directives;
    _ = reader_sweeps;
    _ = reader_live;
    _ = reader_values;
    _ = reader_subst;
    _ = reader_measures;
    _ = reader_nets;
    _ = reader_devices;
    _ = reader_behavioural;
    _ = measure;
    _ = pattern;
    _ = spectre_rewrite;
    _ = @import("vec.zig");
    _ = @import("tests/netlist.zig");
    _ = @import("tests/fuzz.zig");
    _ = @import("tests/lines.zig");
    _ = @import("tests/cards.zig");
}
