//! The netlist reader's state. In: a deck's logical lines in one dialect.
//! Out: the `Reader` row store the three walks fill (declarations, devices
//! with subcircuits expanded, analysis nets), and `run`, the spine that
//! calls them in order and freezes the result into a `Netlist`. The walks'
//! methods live in the reader_*.zig stage files.
//! Cites: ngspice inpcom.c, inp.c (recifeval), subckt.c; HSPICE [SA], [CR].
//!
//! `Reader` is generic over `S` (axis: the dialect's line syntax; three
//! cases today: ngspice, hspice, spectre), so each dialect's comptime line
//! rules fold into its walks.
//!
//! Contents:
//! - `pub fn Reader(comptime S: type) type`
//! - `pub const max_group_depth`
//! - `pub const Func`
//! - `pub const Binding`
//! - `pub const NetLookup`
//! - `pub fn node(self: NetLookup, name: []const u8) u32`
//! - `pub const Geometry`
//! - `pub const LocalReads`
//! - `pub const max_trials`
//! - `pub const Sites`
//! - `pub const Slots`
//! - `pub const Hit`
//! - `pub const Size`
//! - `pub fn init(arena: Allocator, orig: []const u8, text: []const u8, dialect: Dialect) Allocator.Error!R`
//! - `pub const run`
//! - `pub const nodeValues`
//! - `pub const declarations`
//! - `pub const branch`
//! - `pub const condition`
//! - `pub const pathOf`
//! - `pub const readFunc`
//! - `pub const paramText`
//! - `pub const checkBare`
//! - `pub const readModel`
//! - `pub const unresolved`
//! - `pub const namesParam`
//! - `pub const modelRow`
//! - `pub const binsEnd`
//! - `pub const closeBins`
//! - `pub const cardLength`
//! - `pub const localRow`
//! - `pub const localKey`
//! - `pub const localReads`
//! - `pub const sameValues`
//! - `pub const readArgs`
//! - `pub const readVpiOptions`
//! - `pub const isStep`
//! - `pub const readDirective`
//! - `pub const readMosra`
//! - `pub const readSample`
//! - `pub const readSaveOp`
//! - `pub const written`
//! - `pub const failed`
//! - `pub const liveParam`
//! - `pub const foldGlobals`
//! - `pub const stepTarget`
//! - `pub const readStep`
//! - `pub const readSweep`
//! - `pub const readOptimize`
//! - `pub const positiveCount`
//! - `pub const numbers`
//! - `pub const readBlock`
//! - `pub const readVariation`
//! - `pub const sigmaExpr`
//! - `pub const percentValue`
//! - `pub const devLot`
//! - `pub const resolveLive`
//! - `pub const readsCircuit`
//! - `pub const holdsSite`
//! - `pub const liveSlot`
//! - `pub const siteRows`
//! - `pub const liveConds`
//! - `pub const unsupported`
//! - `pub const readValue`
//! - `pub const valueAt`
//! - `pub const kvValue`
//! - `pub const junkNumber`
//! - `pub const numberByte`
//! - `pub const refuseOperatorField`
//! - `pub const nameValue`
//! - `pub const readMeasures`
//! - `pub const skipBad`
//! - `pub const readPower`
//! - `pub const readStim`
//! - `pub const sourcePower`
//! - `pub const measureExpr`
//! - `pub const measureOption`
//! - `pub const measureValue`
//! - `pub const exprValue`
//! - `pub const fold`
//! - `pub const find`
//! - `pub const subst`
//! - `pub const bound`
//! - `pub const inline_`
//! - `pub const compiledText`
//! - `pub const levelPath`
//! - `pub const emitNum`
//! - `pub const internName`
//! - `pub const intern`
//! - `pub const connect`
//! - `pub const joined`
//! - `pub const netOf`
//! - `pub const mixedPort`
//! - `pub const fixedPins`
//! - `pub const behaviouralResistor`
//! - `pub const kindOf`
//! - `pub const readDevice`
//! - `pub const behaviouralCard`
//! - `pub const wrapped`
//! - `pub const exprRest`
//! - `pub const word`
//! - `pub const numberAt`
//! - `pub const controlText`
//! - `pub const tableText`
//! - `pub const polyText`
//! - `pub const readDataElement`
//! - `pub const dataPath`
//! - `pub const portLine`
//! - `pub const isModel`
//! - `pub const splitTerminals`
//! - `pub const commit`
//! - `pub const expand`
//! - `pub const expandedExtra`
//! - `pub const bodySize`
//! - `pub const xTarget`
//! - `pub const skipNested`
//! - `pub const shunts`
//! - `pub const option`
//! - `pub const binOptions`
//! - `pub const modelBins`

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
const card_rows = @import("card_rows.zig");
const netlist_view = @import("netlist_view.zig");
const card_table = @import("card_table.zig");
const reader_frame = @import("reader_frame.zig");
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
const Dialect = card_rows.Dialect;
const VertexId = card_rows.VertexId;
const none = card_rows.none;
const Span = card_rows.Span;
const Value = card_rows.Value;
const Group = card_rows.Group;
const Kv = card_rows.Kv;
const Hypergraph = card_rows.Hypergraph;
const Step = card_rows.Step;
const Optimize = card_rows.Optimize;
const Data = card_rows.Data;
const Variation = card_rows.Variation;
const Live = card_rows.Live;
const Analysis = card_rows.Analysis;
const Config = card_rows.Config;
const Foreign = card_rows.Foreign;
const Mosra = card_rows.Mosra;
const AppendModel = card_rows.AppendModel;
const isGround = netlist_view.isGround;
const cards = card_table.cards;
const Entry = reader_frame.Entry;
const Scope = reader_frame.Scope;
const Subckt = reader_frame.Subckt;
const ModelRow = reader_frame.ModelRow;
const LocalModel = reader_frame.LocalModel;

/// The netlist reader over dialect syntax `S` (lines.zig): every walk's
/// state, and its methods, which live in the reader_*.zig stage files.
pub fn Reader(comptime S: type) type {
    return struct {
        const R = @This();

        arena: Allocator,
        dialect: Dialect,
        orig: []const u8,
        text: []const u8,
        /// The card a walk is reading: a refusal no reader logged names it (#144).
        card: []const u8 = "",
        /// Set once `failed` or `unsupported` has logged the refusal's card.
        logged: bool = false,
        lines: std.ArrayList([]const u8) = .empty,
        hg: Hypergraph.Builder,
        pool: InternPool = .{},
        /// The net of each name, `none` for a device name.
        net_of: std.ArrayList(u32) = .empty,
        /// A flattened name under construction.
        name_buf: std.ArrayList(u8) = .empty,
        values: std.ArrayList(Value) = .empty,
        kvs: std.ArrayList(Kv) = .empty,
        ops: std.ArrayList(expr.Op) = .empty,
        consts: std.ArrayList(f64) = .empty,
        models: std.ArrayList(ModelRow) = .empty,
        model_ids: std.StringHashMapUnmanaged(u32) = .empty,
        subckts: std.ArrayList(Subckt) = .empty,
        subckt_ids: std.StringHashMapUnmanaged(u16) = .empty,
        /// Every subcircuit's `.model` cards, in line order.
        local_models: std.ArrayList(LocalModel) = .empty,
        /// Names in `local_models`, for `isModel`.
        local_names: std.StringHashMapUnmanaged(void) = .empty,
        /// `.option scale` and `wnflag`, read before any device.
        scale: f64 = 1,
        wnflag: bool = false,
        /// Per `local_models` row, the model rows read from its line, one per
        /// distinct set of values, so instances with equal parameters share
        /// a row. Sized on the first `localRow`, once `local_models` is whole.
        variants: []std.ArrayList(u32) = &.{},
        /// Per `local_models` row, the parameter names its line can read
        /// (`localReads`); null until first needed.
        local_reads: []?LocalReads = &.{},
        /// `localRow` results keyed by the row and every scope entry its
        /// line can reach (`localKey`): an instance whose parameters are the
        /// same text as an earlier one's skips re-reading the card.
        local_memo: std.StringHashMapUnmanaged(u32) = .empty,
        local_key: std.ArrayList(u8) = .empty,
        /// `(name, row)` of the instance models the open expansions resolved;
        /// `Frame.models_mark` splits it by instance.
        instance_models: std.ArrayList(struct { name: []const u8, row: u32 }) = .empty,
        globals: Scope = .empty,
        /// The global `.param` names in deck order, for `foldGlobals`.
        global_keys: std.ArrayList([]const u8) = .empty,
        global_scopes: [1]*const Scope = undefined,
        analyses: std.ArrayList(Analysis) = .empty,
        config: std.ArrayList(Config) = .empty,
        ic_cards: std.ArrayList([]const Value) = .empty,
        nodeset_cards: std.ArrayList([]const Value) = .empty,
        meas_lines: std.ArrayList([]const u8) = .empty,
        /// `.global` names: one net at every subcircuit level.
        global_nets: std.StringHashMapUnmanaged(void) = .empty,
        /// Per `lines` row, the device letter `spectre.translate` gave a
        /// card whose name does not carry it (0: none); empty outside the
        /// Spectre dialect.
        kinds: []const u8 = &.{},
        /// `.func` and `.param f(x)=` definitions, by name.
        funcs: std.StringHashMapUnmanaged(Func) = .empty,
        /// The formals of the user functions `subst` is inside, innermost last.
        bindings: std.ArrayList(Binding) = .empty,
        saves: std.ArrayList([]const u8) = .empty,
        save_all: bool = false,
        save_op: ?core.SaveOp = null,
        stims: std.ArrayList(core.Stim) = .empty,
        sample: ?core.query.NoiseSample = null,
        mosra: ?Mosra = null,
        appendmodels: std.ArrayList(AppendModel) = .empty,
        foreign: std.ArrayList(Foreign) = .empty,
        vpi_apps: std.ArrayList([]const u8) = .empty,
        measures: std.ArrayList(core.Measure) = .empty,
        instances: u32 = 1,
        /// `.data` and `.variation` blocks: first and one-past-last line.
        blocks: std.ArrayList([2]u32) = .empty,
        steps: std.ArrayList(Step) = .empty,
        data: std.ArrayList(Data) = .empty,
        variations: std.ArrayList(Variation) = .empty,
        /// Top-level `.if`/`.elseif` conditions evaluated in walk 1, in order.
        cond_texts: std.ArrayList([]const u8) = .empty,
        /// Swept global parameters, registered before any device is read.
        live_names: std.StringArrayHashMapUnmanaged(void) = .empty,
        live_nominal: std.ArrayList(f64) = .empty,
        /// `Live.bins`, column by column.
        bin_edges: std.ArrayList(u32) = .empty,
        bin_starts: std.ArrayList(u32) = .empty,
        bin_rows: std.ArrayList(u32) = .empty,
        bin_bounds: std.ArrayList([4]f64) = .empty,
        bin_per_finger: std.ArrayList(bool) = .empty,
        /// A device or model value is being read: one that reads a live name
        /// or (under `monte`) a distribution keeps its postfix.
        in_card: bool = false,
        /// A live name may appear outside a card (a live name's own value).
        live_ok: bool = false,
        /// Some card runs Monte Carlo.
        monte: bool = false,
        /// `Deck.draw_seed`.
        draw_seed: bool = false,
        optimize: ?Optimize = null,
        site_ops: std.ArrayList(u32) = .empty,
        site_keys: std.ArrayList(u64) = .empty,
        /// Key of the text whose distribution calls are being emitted, and
        /// the next call's ordinal within it.
        site_owner: u64 = 0,
        site_ordinal: u32 = 0,
        // Per-card scratch.
        nodes: std.ArrayList([]const u8) = .empty,
        pins: std.ArrayList(VertexId) = .empty,
        positional: std.ArrayList(Value) = .empty,
        card_kv: std.ArrayList(Kv) = .empty,
        scratch: expr.Scratch = .{},
        /// Parameter and function texts compiled once each (`compiledText`).
        compiled_texts: std.StringHashMapUnmanaged(*const expr.Scratch) = .empty,
        stack: std.ArrayList(expr.Val) = .empty,
        /// Open `name(` groups in the value being read.
        group_depth: u8 = 0,

        pub const max_group_depth = 32;

        /// A user function: its formals and its body as written.
        pub const Func = struct { formals: []const []const u8, body: []const u8 };

        /// A formal of the function `subst` is expanding, bound to the
        /// argument's final ops.
        pub const Binding = struct { name: []const u8, ops: []const expr.Op };

        pub const NetLookup = struct {
            pool: *const InternPool,
            net_of: []const u32,
            dialect: Dialect,
            /// The net `name` names, 0 for ground, `none` for no net.
            pub fn node(self: NetLookup, name: []const u8) u32 {
                if (isGround(self.dialect, name)) return 0;
                const n = self.pool.find(name) orelse return none;
                return self.net_of[n.index()];
            }
        };

        // Walk 1: declarations.

        pub const Geometry = struct { num: f64, live: bool = false };

        pub const LocalReads = union(enum) { stochastic, names: []const []const u8 };

        /// Trials one `MONTE=list(...)` names in all, its `a:b` ranges
        /// expanded: each is an argument row of 24 B, so a short card
        /// cannot ask for gigabytes. A million runs is past any budget.
        pub const max_trials = 1 << 20;

        /// `Live.site_rows` and `Live.site_spans` as `liveSlot` finds them.
        pub const Sites = struct { rows: std.ArrayList(u32) = .empty, spans: std.ArrayList(Span) = .empty };

        /// `Live.slots` and `Live.groups` as `liveSlot` finds them.
        pub const Slots = struct { at: std.ArrayList(Live.Slot) = .empty, groups: std.ArrayList(Group) = .empty };

        pub const Hit = struct { entry: Entry, level: usize };

        pub const Size = struct {
            cards: usize = 0,
            kvs: usize = 0,
            state: enum { new, busy, done } = .new,

            fn add(a: *Size, b: Size) void {
                a.cards +|= b.cards;
                a.kvs +|= b.kvs;
            }
        };

        pub fn init(arena: Allocator, orig: []const u8, text: []const u8, dialect: Dialect) Allocator.Error!R {
            var r: R = .{ .arena = arena, .dialect = dialect, .orig = orig, .text = text, .hg = try .init(arena) };
            const zero = try r.pool.intern(arena, "0");
            _ = r.hg.addVertex(arena, .{ .name = zero }) catch return error.OutOfMemory;
            try r.net_of.append(arena, 0);
            return r;
        }

        // The methods each stage file holds, as decls of this type so
        // `r.name(...)` resolves.
        pub const run = reader_run.Methods(S).run;
        pub const nodeValues = reader_run.Methods(S).nodeValues;
        pub const declarations = reader_walk.Methods(S).declarations;
        pub const branch = reader_walk.Methods(S).branch;
        pub const condition = reader_walk.Methods(S).condition;
        pub const pathOf = reader_walk.Methods(S).pathOf;
        pub const readFunc = reader_walk.Methods(S).readFunc;
        pub const paramText = reader_walk.Methods(S).paramText;
        pub const checkBare = reader_walk.Methods(S).checkBare;
        pub const readModel = reader_models.Methods(S).readModel;
        pub const unresolved = reader_models.Methods(S).unresolved;
        pub const namesParam = reader_models.Methods(S).namesParam;
        pub const modelRow = reader_models.Methods(S).modelRow;
        pub const binsEnd = reader_models.Methods(S).binsEnd;
        pub const closeBins = reader_models.Methods(S).closeBins;
        pub const cardLength = reader_models.Methods(S).cardLength;
        pub const localRow = reader_models.Methods(S).localRow;
        pub const localKey = reader_models.Methods(S).localKey;
        pub const localReads = reader_models.Methods(S).localReads;
        pub const sameValues = reader_models.Methods(S).sameValues;
        pub const readArgs = reader_directives.Methods(S).readArgs;
        pub const readVpiOptions = reader_directives.Methods(S).readVpiOptions;
        pub const isStep = reader_directives.Methods(S).isStep;
        pub const readDirective = reader_directives.Methods(S).readDirective;
        pub const readMosra = reader_directives.Methods(S).readMosra;
        pub const readSample = reader_directives.Methods(S).readSample;
        pub const readSaveOp = reader_directives.Methods(S).readSaveOp;
        pub const written = reader_directives.Methods(S).written;
        pub const failed = reader_directives.Methods(S).failed;
        pub const liveParam = reader_directives.Methods(S).liveParam;
        pub const foldGlobals = reader_directives.Methods(S).foldGlobals;
        pub const stepTarget = reader_directives.Methods(S).stepTarget;
        pub const readStep = reader_directives.Methods(S).readStep;
        pub const readSweep = reader_sweeps.Methods(S).readSweep;
        pub const readOptimize = reader_sweeps.Methods(S).readOptimize;
        pub const positiveCount = reader_sweeps.Methods(S).positiveCount;
        pub const numbers = reader_sweeps.Methods(S).numbers;
        pub const readBlock = reader_sweeps.Methods(S).readBlock;
        pub const readVariation = reader_sweeps.Methods(S).readVariation;
        pub const sigmaExpr = reader_sweeps.Methods(S).sigmaExpr;
        pub const percentValue = reader_sweeps.Methods(S).percentValue;
        pub const devLot = reader_sweeps.Methods(S).devLot;
        pub const resolveLive = reader_live.Methods(S).resolveLive;
        pub const readsCircuit = reader_live.Methods(S).readsCircuit;
        pub const holdsSite = reader_live.Methods(S).holdsSite;
        pub const liveSlot = reader_live.Methods(S).liveSlot;
        pub const siteRows = reader_live.Methods(S).siteRows;
        pub const liveConds = reader_live.Methods(S).liveConds;
        pub const unsupported = reader_live.Methods(S).unsupported;
        pub const readValue = reader_values.Methods(S).readValue;
        pub const valueAt = reader_values.Methods(S).valueAt;
        pub const kvValue = reader_values.Methods(S).kvValue;
        pub const junkNumber = reader_values.Methods(S).junkNumber;
        pub const numberByte = reader_values.Methods(S).numberByte;
        pub const refuseOperatorField = reader_values.Methods(S).refuseOperatorField;
        pub const nameValue = reader_values.Methods(S).nameValue;
        pub const readMeasures = reader_measures.Methods(S).readMeasures;
        pub const skipBad = reader_measures.Methods(S).skipBad;
        pub const readPower = reader_measures.Methods(S).readPower;
        pub const readStim = reader_measures.Methods(S).readStim;
        pub const sourcePower = reader_measures.Methods(S).sourcePower;
        pub const measureExpr = reader_measures.Methods(S).measureExpr;
        pub const measureOption = reader_measures.Methods(S).measureOption;
        pub const measureValue = reader_measures.Methods(S).measureValue;
        pub const exprValue = reader_values.Methods(S).exprValue;
        pub const fold = reader_values.Methods(S).fold;
        pub const find = reader_values.Methods(S).find;
        pub const subst = reader_subst.Methods(S).subst;
        pub const bound = reader_subst.Methods(S).bound;
        pub const inline_ = reader_subst.Methods(S).inline_;
        pub const compiledText = reader_subst.Methods(S).compiledText;
        pub const levelPath = reader_subst.Methods(S).levelPath;
        pub const emitNum = reader_values.Methods(S).emitNum;
        pub const internName = reader_nets.Methods(S).internName;
        pub const intern = reader_nets.Methods(S).intern;
        pub const connect = reader_nets.Methods(S).connect;
        pub const joined = reader_nets.Methods(S).joined;
        pub const netOf = reader_nets.Methods(S).netOf;
        pub const mixedPort = reader_nets.Methods(S).mixedPort;
        pub const fixedPins = reader_nets.Methods(S).fixedPins;
        pub const behaviouralResistor = reader_devices.Methods(S).behaviouralResistor;
        pub const kindOf = reader_devices.Methods(S).kindOf;
        pub const readDevice = reader_devices.Methods(S).readDevice;
        pub const behaviouralCard = reader_behavioural.Methods(S).behaviouralCard;
        pub const wrapped = reader_behavioural.Methods(S).wrapped;
        pub const exprRest = reader_behavioural.Methods(S).exprRest;
        pub const word = reader_behavioural.Methods(S).word;
        pub const numberAt = reader_behavioural.Methods(S).numberAt;
        pub const controlText = reader_behavioural.Methods(S).controlText;
        pub const tableText = reader_behavioural.Methods(S).tableText;
        pub const polyText = reader_behavioural.Methods(S).polyText;
        pub const readDataElement = reader_behavioural.Methods(S).readDataElement;
        pub const dataPath = reader_behavioural.Methods(S).dataPath;
        pub const portLine = reader_behavioural.Methods(S).portLine;
        pub const isModel = reader_devices.Methods(S).isModel;
        pub const splitTerminals = reader_devices.Methods(S).splitTerminals;
        pub const commit = reader_devices.Methods(S).commit;
        pub const expand = reader_nets.Methods(S).expand;
        pub const expandedExtra = reader_nets.Methods(S).expandedExtra;
        pub const bodySize = reader_nets.Methods(S).bodySize;
        pub const xTarget = reader_nets.Methods(S).xTarget;
        pub const skipNested = reader_nets.Methods(S).skipNested;
        pub const shunts = reader_devices.Methods(S).shunts;
        pub const option = reader_devices.Methods(S).option;
        pub const binOptions = reader_devices.Methods(S).binOptions;
        pub const modelBins = reader_devices.Methods(S).modelBins;
    };
}
