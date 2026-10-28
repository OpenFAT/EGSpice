//! Everything a deck says besides its topology: probes, drive sources,
//! options, queries, `.ic`/`.nodeset` values, variants, MOSRA stress rows,
//! `.save`/`.stim` requests and the bindings later directives resolve
//! against. Plain data, filled by the frontend and read by the analysis and
//! output layers; `Variants.writes` is the one transform (row in, its
//! parameter writes out). The `.meas` cards are in `measure_card.zig`.
//! Cites: HSPICE [CR .SAVE, .STIM; SA Ch.29]; dev/analysis/mosra.md,
//! dev/analysis/optimize.md.
//!
//! Contents:
//! - `pub const Ic = struct { node: u32, value: f64 }`
//! - `pub const SaveOp = struct`
//! - `pub const Stim = struct`
//! - `pub const AcOverride = struct`
//! - `pub const Variants = struct`, `Variants.count(self: Variants) u32`,
//!   `Variants.writes(self: Variants, v: u32) struct { []const u32, []const f64 }`
//! - `pub const Mosra = struct`, `Mosra.no_param: u32`
//! - `pub const MosraModel = struct`
//! - `pub const QueryBindings = struct`
//! - `pub const Deck = struct`

const std = @import("std");
const assert = std.debug.assert;
const requests = @import("query.zig");
const numerics = @import("numerics.zig");
const DeviceType = @import("ids.zig").DeviceType;
const Measure = @import("measure_card.zig").Measure;

/// A resolved `.ic` value; all of them apply together when a query starts
/// with UIC.
pub const Ic = struct { node: u32, value: f64 };

/// HSPICE `.save` [CR .SAVE]: write the operating point as `.nodeset` (or
/// `.ic`) cards a later run's `.load` reads.
pub const SaveOp = struct {
    /// FILE=, as written; null for `<deck stem>.ic0`.
    file: ?[]const u8 = null,
    /// TYPE=IC instead of NODESET.
    ic: bool = false,
    /// LEVEL=TOP: top-level nodes only.
    top_only: bool = false,
    /// TIME=: the transient state at this time, in seconds; 0 is the
    /// operating point.
    time: f64 = 0,
};

/// HSPICE `.stim tran pwl|data` [CR .STIM]: after the run, the first
/// transient's signals are written to a file as PWL sources or as one
/// `.data` table, for a later run to read.
pub const Stim = struct {
    /// `.data dataname` instead of PWL sources.
    data: bool = false,
    /// FILENAME=, as written; null for the deck's stem. The file is
    /// `<name>.pwl<serial>_tr0` or `<name>.dat<serial>_tr0`.
    file: ?[]const u8 = null,
    /// Counts the cards of the same form from 0.
    serial: u32 = 0,
    dataname: []const u8 = "",
    /// One `find` measure per signal: `name` is the source or column name,
    /// `first` the waveform.
    signals: []const Measure = &.{},
    /// A PWL source's terminals, parallel to `signals`.
    nodes: []const [2][]const u8 = &.{},
    /// INDEPVAR= times; empty to sample FROM..TO, at NPOINTS evenly spaced
    /// times, or at every transient point when NPOINTS is 0.
    times: []const f64 = &.{},
    from: f64 = 0,
    to: f64 = std.math.inf(f64),
    npoints: u32 = 0,
    /// INDEPOUT: a `time` column in a `.data` table.
    indepout: bool = false,
};

/// A device parameter value that replaces the DC one in AC-family analyses
/// (a resistor's `ac=`). Stored by identity; each analysis clone resolves
/// its own parameter pointer.
pub const AcOverride = struct {
    type: DeviceType,
    /// Instance index within `type`.
    index: u32,
    param_name: []const u8,
    value: f64,
};

/// Circuit variants over one topology: `.step` and `.data` points, `.alter`
/// runs and Monte Carlo trials. Variant `v` writes `values[i]` to parameter
/// `refs[i]` (an index into the circuit's `collectParams` list) for every
/// `i` in `starts[v]..starts[v + 1]`, then runs at `temp_c[v]` when that is
/// set. A query runs variant `Tolerances.variant`; the rows are SoA and
/// `starts` has one entry more than `labels`.
pub const Variants = struct {
    /// Plot-name suffix without parentheses: `p=1.5`, `alter=2`, `monte=3`.
    labels: []const []const u8 = &.{},
    /// Circuit temperature in °C, or null for the deck's.
    temp_c: []const ?f64 = &.{},
    /// The variant's value of the first swept name (the trial number for
    /// Monte Carlo), the sweep axis of a lane query.
    axis: []const f64 = &.{},
    starts: []const u32 = &.{},
    refs: []const u32 = &.{},
    values: []const f64 = &.{},

    /// Number of variants (rows). Asserts that a non-empty table has one
    /// `starts` entry more than rows.
    pub fn count(self: Variants) u32 {
        if (self.labels.len != 0) assert(self.starts.len == self.labels.len + 1);
        return @intCast(self.labels.len);
    }

    /// The parameter writes of variant `v`: `.{ refs, values }`, slices of
    /// the table's own columns. Asserts that `starts` holds row `v`, that
    /// `refs` and `values` have equal lengths, and that the row's range is
    /// not reversed.
    pub fn writes(self: Variants, v: u32) struct { []const u32, []const f64 } {
        assert(v + 1 < self.starts.len);
        assert(self.refs.len == self.values.len);
        const lo = self.starts[v];
        const hi = self.starts[v + 1];
        assert(lo <= hi);
        return .{ self.refs[lo..hi], self.values[lo..hi] };
    }
};

/// HSPICE MOSRA [SA Ch.29] (dev/analysis/mosra.md): the transient that
/// stresses the bound MOSFETs, the reliability times the aged runs extrapolate
/// to, and one SoA row per stressed instance. Aged time `k` is row `k` of
/// `Deck.variants`; each row writes `delvto` then, where the device has
/// one, `mulu0`, device by device: the device's own value shifted by the
/// aged ΔVth, and scaled by the mobility factor.
pub const Mosra = struct {
    /// The stress transient: the deck's first nominal `.tran`.
    tran: requests.Tran,
    /// Reliability times of the aged runs, seconds, ascending.
    rel_times: []const f64,
    /// SimMode 2 runs the aged rows; SimMode 0 only reports degradation.
    aged_runs: bool,
    /// RelMode: which mechanisms age the devices.
    hci: bool = true,
    bti: bool = true,
    /// Stress window within the transient (AgingStart, AgingStop), seconds.
    aging_start: f64 = 0,
    aging_stop: f64 = std.math.inf(f64),
    /// Vgs (polarity-corrected) at or below which a mechanism sees no
    /// stress, volts.
    hci_threshold: f64 = 0,
    bti_threshold: f64 = 0,
    /// DegF: the |ΔVth| that ends a device's life; adds `life(m)` columns.
    deg_f: ?f64 = null,
    /// The `.model ... MOSRA` cards `model` indexes.
    models: []const MosraModel,
    /// Card name of each stressed instance, for the table's columns.
    names: []const []const u8,
    /// Circuit rows of drain, gate and source.
    terminals: []const [3]u32,
    pmos: []const bool,
    /// Device temperature during the stress run, kelvin: the instance's
    /// `temp` where the card gives one, else the circuit's plus its `dtemp`.
    temp_k: []const f64,
    /// Row of `models` aging the instance.
    model: []const u16,
    /// Indices into the circuit's `collectParams` list; `mulu0` is
    /// `no_param` on a device without one.
    delvto: []const u32,
    mulu0: []const u32,
    /// The fresh values the aged ones start from; `mulu0_fresh` is 1 where
    /// the device has no `mulu0`.
    delvto_fresh: []const f64,
    mulu0_fresh: []const f64,

    /// `mulu0` of a device without that parameter.
    pub const no_param = std.math.maxInt(u32);
};

/// One `.model name MOSRA LEVEL=1` card: a power law per mechanism,
/// ΔVth = A·t^n for constant stress, with A = a0·exp(fd·v)·exp(-td/T).
/// Parameter names are ESPice's choice (dev/analysis/mosra.md).
pub const MosraModel = struct {
    /// BTI: v is Vgs.
    tit0: f64 = 0,
    titfd: f64 = 0,
    tittd: f64 = 0,
    tn: f64 = 0.25,
    /// HCI: v is Vds while Vgs is above the threshold.
    hci0: f64 = 0,
    hcifd: f64 = 0,
    hcitd: f64 = 0,
    hcin: f64 = 0.5,
    /// Mobility loss per volt of each mechanism's ΔVth: mulu0 = 1/(1 + Σ mu·Δ).
    titmu: f64 = 0,
    hcimu: f64 = 0,
};

/// Source bindings kept from construction so later analysis directives can
/// name sources and ports. All rows are post-permutation.
pub const QueryBindings = struct {
    /// V card names, parallel to `v_branches`, `v_pos`, `v_neg`, `v_distof1`,
    /// `v_distof2`.
    v_names: []const []const u8,
    /// I card names, parallel to `i_pos` and `i_neg`.
    i_names: []const []const u8,
    /// MNA branch row of each V card.
    v_branches: []const u32,
    /// Node rows of each V card: `+` then `-`.
    v_pos: []const u32,
    v_neg: []const u32,
    /// Node rows of each I card. An I source has no branch row.
    i_pos: []const u32,
    i_neg: []const u32,
    /// `DISTOF1 <mag> <phase>` of each V card.
    v_distof1: []const [2]f64,
    /// `DISTOF2 <mag> <phase>` of each V card; empty is none.
    v_distof2: []const [2]f64 = &.{},
    ports: []const requests.Port,
};

/// The prepared deck. Session-owned; every slice lives in the arena that
/// built the circuit.
pub const Deck = struct {
    /// MNA rows to publish, parallel to `probe_labels`.
    probes: []const u32,
    probe_labels: []const []const u8,
    /// Node and branch row of the drive source that analyses default to.
    source_node: u32,
    source_branch: u32,
    /// Row of the last net the deck introduces: the output of cards that
    /// name none (`.pac`, `.pxf`, `.disto`). Not the last probe, whose place
    /// the BBD permutation moves.
    output_node: u32,
    /// AC excitation: n real rows, then n imaginary rows.
    ac_drive: []const f64,
    title: []const u8,
    n_devices: u32,
    ic: []const Ic,
    /// C and L `ic=` values on their rows (a C's node, an L's branch). Only
    /// a `uic` start applies them, after `ic`; ngspice ignores them otherwise.
    uic_ic: []const Ic = &.{},
    /// `.nodeset` guesses: where the operating point's first solve holds
    /// each node.
    nodeset: []const Ic = &.{},
    /// The first Newton iterate of every operating point, `n` values, in
    /// place of the cold start (`query.Op.warm_start`); empty is cold. The
    /// optimizer sets it on the decks of its point sessions
    /// (dev/analysis/optimize.md).
    op_guess: []const f64 = &.{},
    /// The template's nominal operating point: the first Newton iterate of
    /// every operating point that asks for a warm start
    /// (`query.Op.warm_start`, from a transient's
    /// `PointOptions.warm_start`) when `op_guess` is empty. Empty until
    /// `Problem` solves it, and then fixed, so a point's start never
    /// depends on which points ran before it.
    warm_op: []const f64 = &.{},
    /// `.options` tolerances every query starts from.
    deck_tol: numerics.Tolerances,
    /// `.temp` or `.options temp`, in °C, when given.
    deck_temp: ?f64,
    /// `.options method`, when given.
    deck_method: ?requests.Method,
    queries: []const requests.Query,
    bindings: QueryBindings,
    /// Card name of every device instance, for result labels.
    cards: []const requests.CardRef,
    /// `cards` as a dense per-type column, for per-instance labels.
    card_names: requests.CardNames = .empty,
    ac_overrides: []const AcOverride,
    /// `.meas` cards, in deck order, evaluated over finished results.
    measures: []const Measure = &.{},
    variants: Variants = .{},
    /// HSPICE `.save`, when the deck has one.
    save_op: ?SaveOp = null,
    /// HSPICE `.mosra`, when the deck has one.
    mosra: ?Mosra = null,
    /// HSPICE `.stim` cards, in deck order.
    stims: []const Stim = &.{},
};

test "Variants slice each row's writes out of the shared columns" {
    const t = std.testing;
    const v: Variants = .{
        .labels = &.{ "p=1", "p=2", "p=3" },
        .temp_c = &.{ null, 50, null },
        .axis = &.{ 1, 2, 3 },
        .starts = &.{ 0, 1, 3, 3 },
        .refs = &.{ 4, 4, 7 },
        .values = &.{ 1, 2, 0.5 },
    };
    try t.expectEqual(@as(u32, 3), v.count());
    const row1 = v.writes(1);
    try t.expectEqualSlices(u32, &.{ 4, 7 }, row1[0]);
    try t.expectEqualSlices(f64, &.{ 2, 0.5 }, row1[1]);
    // A row may write nothing and only move the temperature or the axis.
    try t.expectEqual(@as(usize, 0), v.writes(2)[0].len);
    try t.expectEqual(@as(u32, 0), (Variants{}).count());
}
