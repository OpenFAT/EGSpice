//! The parsed deck's row types. In: nothing (definitions only). Out: the
//! values, cards, sweeps, variations, live slots and deck tables every
//! stage after the reader reads.
//! Cites: dev/research/{ngspice,hspice,spectre}-syntax.md; HSPICE [SA]
//! Ch.25-29, [CR] (the HSPICE cards each type holds).
//!
//! Contents:
//! - `pub const Dialect`
//! - `pub const VertexId`
//! - `pub const EdgeId`
//! - `pub const Kind`
//! - `pub const ground`
//! - `pub const none`
//! - `pub const Error`
//! - `pub const Span`
//! - `pub const Value`
//! - `pub fn number(v: Value) ?f64`
//! - `pub fn numberAt(args: []const Value, i: usize) ?f64`
//! - `pub fn nameAt(args: []const Value, i: usize) ?[]const u8`
//! - `pub const Group`
//! - `pub fn name(g: Group) []const u8`
//! - `pub fn args(g: Group) []const Value`
//! - `pub const Kv`
//! - `pub const Net`
//! - `pub const Device`
//! - `pub const Hypergraph`
//! - `pub const Model`
//! - `pub const StepTarget`
//! - `pub const Step`
//! - `pub const Monte`
//! - `pub const Sweep`
//! - `pub const Optimize`
//! - `pub const Data`
//! - `pub const Variation`
//! - `pub const sigma_names`
//! - `pub const Dist`
//! - `pub const Live`
//! - `pub const Slot`
//! - `pub const Table`
//! - `pub const Bins`
//! - `pub const Analysis`
//! - `pub const Config`
//! - `pub const Ic`
//! - `pub const ForeignKind`
//! - `pub const Foreign`
//! - `pub const DataFile`
//! - `pub const Deck`
//! - `pub const Mosra`
//! - `pub const AppendModel`

const std = @import("std");
const Allocator = std.mem.Allocator;
const lines = @import("lines.zig");
const spectre_rewrite = @import("spectre.zig");
const source = @import("source.zig");
const hypergraph = @import("hypergraph.zig");
const expr = @import("expr.zig");
const measure = @import("measure.zig");
const pattern = @import("pattern.zig");
const core = @import("core");
const Name = core.Name;
const InternPool = core.InternPool;
const assert = std.debug.assert;

/// Input syntax: ngspice, HSPICE or Spectre.
pub const Dialect = lines.Dialect;

/// A net: a row of `Netlist.graph`'s vertex table.
pub const VertexId = hypergraph.VertexId;

/// A device: a row of `Netlist.graph`'s edge table, in card order.
pub const EdgeId = hypergraph.EdgeId;

/// Analysis kind, shared with the query layer.
pub const Kind = core.query.Kind;

/// Net 0; `0`, `gnd` and `ground` all name it.
pub const ground: VertexId = @fromBackingInt(@intCast(0));

/// Missing id in every u32 index space here.
pub const none = std.math.maxInt(u32);

/// `ParseError`: malformed card; `ModelBinNotFound`: no `.model nm.N` bin
/// holds an M card's L/W; `CircuitTooLarge`: a table passed u32;
/// `UnsupportedCard`: a card or element form ESPice cannot simulate, logged
/// with the line.
pub const Error = error{ OutOfMemory, ParseError, ModelBinNotFound, CircuitTooLarge, UnsupportedCard };

/// Rows `start..start+len` of a flat table.
pub const Span = hypergraph.Span;

/// One card value after parameter substitution.
pub const Value = union(enum) {
    num: f64,
    /// A word no scope defines: a node, model, keyword or source name.
    name: []const u8,
    /// Postfix ops in `Netlist.ops` that did not fold to a number.
    expr: Span,
    /// `name(args)`: a source waveform or a `v(...)`/`i(...)` output.
    group: Group,

    /// The number; null for a name, an unfolded expression or a group.
    pub fn number(v: Value) ?f64 {
        return switch (v) {
            .num => |n| n,
            else => null,
        };
    }

    /// `args[i]` as a number; null past the end or for a non-number.
    pub fn numberAt(args: []const Value, i: usize) ?f64 {
        return if (i < args.len) args[i].number() else null;
    }

    /// `args[i]` as a bare word; null past the end or for a non-name.
    pub fn nameAt(args: []const Value, i: usize) ?[]const u8 {
        if (i >= args.len) return null;
        return switch (args[i]) {
            .name => |n| n,
            else => null,
        };
    }
};

/// `name(args)`, held as one slice: `items[0]` is the name (a `.name`
/// value), the rest the arguments. One slice keeps `Value` at 24 B.
/// ponytail: group items are arena slices, not a table: a few per deck.
pub const Group = struct {
    items: []const Value,

    /// The word before the parenthesis. Asserts that `items[0]` is a name.
    pub fn name(g: Group) []const u8 {
        std.debug.assert(g.items.len > 0);
        return g.items[0].name;
    }

    /// The values inside the parentheses, in card order.
    pub fn args(g: Group) []const Value {
        std.debug.assert(g.items.len > 0);
        return g.items[1..];
    }
};

/// `key=value`; a positional model value has an empty key.
pub const Kv = struct { key: []const u8, value: Value };

/// Vertex payload: the net's flattened name in `Netlist.pool`.
pub const Net = hypergraph.Net;

/// Hyperedge payload; `model` is a `.model` row or `none`.
pub const Device = hypergraph.Device;

/// Nets by devices; a device's pins are its member list, in terminal order.
pub const Hypergraph = hypergraph;

/// A `.model` card: `kind` is the type word (`nmos`, `d`, a VA module name).
pub const Model = struct { name: []const u8, kind: []const u8, kv: []const Kv };

/// What a `.step` card or an HSPICE `SWEEP` varies.
pub const StepTarget = union(enum) {
    /// A global `.param`, by its row in `Live.names`.
    param: u32,
    temp,
    /// A card's primary value (a source's `dc`, a resistor's `r`), by card name.
    card: []const u8,
};

/// One swept quantity and its points, in sweep order.
pub const Step = struct { target: StepTarget, values: []const f64 };

/// HSPICE `MONTE=n [FIRSTRUN=k]`: trials `first..first + n - 1`; or
/// `MONTE=list(a b:c ...)` [CR .DC]: exactly the trials in `list`.
pub const Monte = struct { n: u32, first: u32 = 1, list: []const u32 = &.{} };

/// An HSPICE sweep on one analysis card: the card runs once per point.
/// On a `.dc` card with no sweep of its own (`.dc DATA=d`, `.dc MONTE=n`)
/// the points are the DC sweep.
pub const Sweep = union(enum) {
    none,
    /// `SWEEP DATA=name`: one point per row of that `.data` table.
    data: []const u8,
    monte: Monte,
    step: Step,
    /// `SWEEP OPTIMIZE=`: the card is the one `Deck.optimize` fits.
    optimize,
};

/// HSPICE optimization [SA Ch.27]: `<card> SWEEP OPTIMIZE=name
/// RESULTS=m1,m2 MODEL=optmod` fits the `.param p = name(init, lo, hi
/// [, dels])` and `OPTRANGE(...)` parameters to the `GOAL=` of the named
/// `.meas` cards. The parameter columns are parallel.
pub const Optimize = struct {
    name: []const u8,
    /// `.meas` card names.
    results: []const []const u8,
    /// The `.model name OPT` card with the optimizer's options.
    model: []const u8,
    /// Each parameter's row in `Live.names`; its nominal is the initial value.
    live: []const u32,
    lo: []const f64,
    hi: []const f64,
    /// Finite-difference step; 0 when not given.
    dels: []const f64,
    /// ESPice's fifth argument (#95): the grid each value stays on; 0 when
    /// not given.
    grid: []const f64,
};

/// An inline `.data` table: one target per column, `values` row-major.
pub const Data = struct {
    name: []const u8,
    labels: []const []const u8,
    columns: []const StepTarget,
    values: []const f64,
};

/// One Monte Carlo variation of a model or element parameter: HSPICE
/// `DEV`/`LOT` on a `.model` value, or a `.variation` block row.
pub const Variation = struct {
    /// `DEV` and local variation draw once per device; `LOT` and global
    /// variation once per model.
    per_device: bool,
    dist: Dist,
    /// The `.model` name; empty for element variation, which `letter` names.
    model: []const u8,
    letter: u8 = 0,
    param: []const u8,
    /// One sigma for `gauss`, the half range otherwise; a fraction of the
    /// nominal when `relative`.
    value: f64,
    relative: bool,
    /// A `.variation` sigma written as an expression: its postfix in
    /// `Netlist.ops`, evaluated per device with `live` = that device's
    /// `w`, `l` and `m` (`sigma_names`), then times `value` (1, or 0.01
    /// with `%`). Empty for a number, which `value` holds alone.
    sigma: Span = .{},

    /// The instance values a sigma expression reads, in `live` order.
    pub const sigma_names = [_][]const u8{ "w", "l", "m" };

    /// Normal, uniform over ±value, or one of the two extremes.
    pub const Dist = enum(u8) { gauss, unif, limit };
};

/// Values a variant re-evaluates: device and model values that read a swept
/// `.param` or, when some card runs Monte Carlo, a distribution call.
pub const Live = struct {
    /// Swept global `.param` names; a `.live` op indexes them.
    names: []const []const u8 = &.{},
    /// Each name's value in the deck as written.
    nominal: []const f64 = &.{},
    /// The values a variant rewrites, parallel to `ops`; `Netlist.slot`
    /// reads one.
    slots: []const Slot = &.{},
    /// The groups whose arguments hold a slot (`Slot.table == .groups`);
    /// their items live in the parse arena.
    groups: []const Group = &.{},
    /// Postfix each slot is evaluated from, and the factor `.option scale`
    /// applies to it afterwards.
    ops: []const Span = &.{},
    scale: []const f64 = &.{},
    /// Distribution calls: op index (ascending) and site key. A site draws
    /// once per trial wherever it is read.
    site_ops: []const u32 = &.{},
    site_keys: []const u64 = &.{},
    /// Constant-pool row that stands in for each name inside expressions a
    /// device keeps as postfix (a B source), or `none`. A variant that moves
    /// one changes storage `ParamRef` cannot reach and must be rebuilt.
    pool_rows: []const u32 = &.{},
    /// Constant-pool rows that stand in for each constant subtree holding a
    /// distribution call inside such an expression (#133), and the subtree
    /// each is re-evaluated from, parallel. Moving one rebuilds the same way.
    site_rows: []const u32 = &.{},
    site_spans: []const Span = &.{},
    /// Postfix of each top-level `.if`/`.elseif` condition the parse
    /// evaluated that reads a live name. A variant keeps the nominal
    /// branches only while each still has its nominal truth value.
    conds: []const Span = &.{},
    bins: Bins = .{},

    /// Where a live value sits: row `row` of `Netlist.values` or
    /// `Netlist.kvs`, or argument `arg` of row `row` of `Live.groups`.
    pub const Slot = struct {
        row: u32,
        /// Index into `Group.args()`; 0 outside `.groups`.
        arg: u32 = 0,
        table: Table,

        /// The table `row` indexes.
        pub const Table = enum(u8) { values, kvs, groups };
    };

    /// M cards on a binned model whose L or W reads a live name, and the
    /// bins each may take; `setLive` picks the bin again at every point.
    pub const Bins = struct {
        /// Edge of card k.
        edges: []const u32 = &.{},
        /// Card k's candidate model rows are `rows[starts[k]..starts[k+1]]`,
        /// last-declared first (ngspice's order), with their lmin, lmax,
        /// wmin and wmax in `bounds`.
        starts: []const u32 = &.{},
        rows: []const u32 = &.{},
        bounds: []const [4]f64 = &.{},
        /// Card k divides W by NF before the bin test (`wnflag`).
        per_finger: []const bool = &.{},
    };
};

/// An analysis card. `pos`/`neg` are the output `v(a[,b])` nets, `ports` the
/// four `.pz` nets; `none` where the card names none or an unknown net.
pub const Analysis = struct {
    kind: Kind,
    args: []const Value,
    sweep: Sweep = .none,
    /// The card as written, for diagnostics.
    line: []const u8 = "",
    /// The syntax the card is read in (`.tran` segments, for one).
    dialect: Dialect = .ngspice,
    pos: u32 = none,
    neg: u32 = none,
    ports: [4]u32 = @splat(none),
    /// Written as an HSPICE shooting-Newton card (`.sn`, `.snac`,
    /// `.snnoise`, `.snxf`) or as `.ptdnoise`, whose keyword arguments
    /// differ from `.pss`, `.pac`, `.pnoise` and `.pxf`.
    sn: bool = false,
    /// One output of a `.four` card that named several; each is its own card.
    split: bool = false,
};

/// `.options` and single-value `.temp` cards, in deck order.
pub const Config = struct {
    temp: bool,
    args: []const Value,
    /// The card as written, for diagnostics.
    line: []const u8 = "",
};

/// One `.ic v(net)=value` entry on a net some card names.
pub const Ic = struct { net: VertexId, value: f64 };

/// What an include carries: OSDI, Verilog-A, digital Verilog or a data file.
pub const ForeignKind = source.ForeignKind;

/// An HDL or OSDI include, path as written (relative to the deck).
pub const Foreign = struct { kind: ForeignKind, path: []const u8 };

/// A `ForeignKind.data` file's contents, read by `prepare`.
pub const DataFile = struct { path: []const u8, bytes: []const u8 };

/// Deck data: everything that is not circuit topology.
pub const Deck = struct {
    /// The first line as written, original case.
    title: []const u8,
    dialect: Dialect,
    analyses: []const Analysis,
    config: []const Config,
    ic: []const Ic,
    /// `.nodeset` guesses, on nets some card names.
    nodeset: []const Ic,
    /// `.save` outputs as lowercased `v(net)`/`i(name)` labels. Empty: save
    /// everything, as with no `.save` card or a `.save all`.
    saves: []const []const u8,
    foreign: []const Foreign,
    /// `.options vpi="app.so"` paths as written (relative to the deck): VPI
    /// applications to load (LRM 12.33, #197).
    vpi_apps: []const []const u8 = &.{},
    /// `.meas` cards; their strings borrow the parse arena.
    measures: []const core.Measure,
    /// `.step` cards in deck order; the last varies fastest.
    steps: []const Step = &.{},
    data: []const Data = &.{},
    variations: []const Variation = &.{},
    /// The optimization of the deck's one `SWEEP OPTIMIZE=` card.
    optimize: ?Optimize = null,
    /// Each `.alter` run's full source, cumulative, expanded; set by
    /// `prepare`.
    alters: []const []const u8 = &.{},
    /// HSPICE `.save` (the last one); ngspice's `.save` fills `saves`.
    save_op: ?core.SaveOp = null,
    /// HSPICE `.stim` cards; their strings borrow the parse arena.
    stims: []const core.Stim = &.{},
    /// HSPICE `.sample` (the last one), for the `.noise` spectra.
    sample: ?core.query.NoiseSample = null,
    /// HSPICE `.mosra` (the last one) and its `.appendmodel` bindings.
    mosra: ?Mosra = null,
    appendmodels: []const AppendModel = &.{},
    /// ngspice dialect with `.option seed=N` and no Monte Carlo sweep: the
    /// distribution calls are drawn once from that seed (`variants.drawOnce`)
    /// instead of folding to their nominal, as ngspice draws them every run.
    draw_seed: bool = false,
};

/// HSPICE `.mosra` options as read [SA Ch.29]; SimMode 0 or 2.
pub const Mosra = struct {
    rel_total_time: f64 = 0,
    rel_start_time: f64 = 0,
    rel_step: f64 = 0,
    /// SimMode 2: aged runs after the fresh ones; 0: degradation only.
    aged_runs: bool = true,
    /// RelMode 0 (both), 1 (HCI) or 2 (BTI).
    rel_mode: u2 = 0,
    aging_start: f64 = 0,
    aging_stop: f64 = std.math.inf(f64),
    hci_threshold: f64 = 0,
    nbti_threshold: f64 = 0,
    deg_f: ?f64 = null,
    /// The card as written, for diagnostics.
    line: []const u8 = "",
};

/// `.appendmodel src MOSRA dst NMOS|PMOS`: MOSRA model `src` ages the
/// MOSFETs of model `dst`.
pub const AppendModel = struct { src: []const u8, dst: []const u8 };

comptime {
    if (@sizeOf(usize) == 8) {
        std.debug.assert(@sizeOf(Value) == 24);
        std.debug.assert(@sizeOf(Kv) == 40);
    }
}

// ---- private ----
