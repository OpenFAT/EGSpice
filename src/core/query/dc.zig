//! Large-signal DC requests: the operating point, DC sweeps of a source,
//! a card parameter or the temperature, the incremental source response,
//! and the Monte Carlo and temperature ensembles over DC solves. One options
//! struct per card, filled by the frontend and read by its driver.
//! Cites: ngspice MODETRANOP (cktop.c); HSPICE `.dc DATA=`/`MONTE=`
//! [CR .DC]; dev/analysis/optimize.md ("Direct sensitivities").
//!
//! Contents:
//! - `pub const Op = struct`
//! - `pub const Dc = struct`, `Dc.SweepTarget = union(enum)`, `SweepTarget.Param`,
//!   `Dc.hasOuter(self: Dc) bool`
//! - `pub const Dcinc = struct`
//! - `pub const Mc = struct`, `Mc.VariantRange = struct`
//! - `pub const Temp = struct`

const Tolerances = @import("../numerics.zig").Tolerances;
const DeviceType = @import("../ids.zig").DeviceType;

/// The DC operating point (`.op`), which most other analyses also run first.
pub const Op = struct {
    tol: Tolerances = .{},
    /// Newton starts from `Deck.warm_op` instead of the cold start, and
    /// falls back to the full ladder when that fails; cold while the deck
    /// has none. A transient's prerequisite takes it from
    /// `PointOptions.warm_start`. The executor also sets it when
    /// `Deck.op_guess` is given.
    warm_start: bool = false,
    /// This operating point starts a transient (ngspice MODETRANOP), so
    /// waveform sources evaluate at t = 0 instead of at their DC value. The
    /// engine sets it when the deck has a transient-family query.
    tran_op: bool = false,
};

/// DC sweep of a source, a card parameter or the temperature (`.dc`), one
/// or two levels deep.
pub const Dc = struct {
    tol: Tolerances = .{},
    start: f64 = 0,
    stop: f64 = 0,
    step: f64 = 1,
    /// What the inner sweep drives, named by device type and parameter so a
    /// V card, an I card and a resistor with the same index stay distinct.
    target: SweepTarget = .{ .device = .{} },
    /// Optional outer sweep (`.dc src1 ... src2 start2 stop2 incr2`, or
    /// `.dc ... temp ...`).
    target2: ?SweepTarget = null,
    start2: f64 = 0,
    stop2: f64 = 0,
    step2: f64 = 1,
    /// HSPICE `LIN`/`DEC`/`OCT`/`POI` grids as explicit values. Non-empty
    /// replaces `start`/`stop`/`step` (`points2` the outer ones).
    points: []const f64 = &.{},
    points2: []const f64 = &.{},
    /// `Deck.variants` rows the sweep also predicts to first order from its
    /// own solutions, for the optimizer's Jacobian (dev/analysis/optimize.md,
    /// "Direct sensitivities"). Each adds a block of the same points after
    /// the sweep's own: the probes moved by -J⁻¹ (F(x, row) - F(x)) at each
    /// point, on the factorization at that point's solution.
    tangents: []const u32 = &.{},

    /// The device parameter, or the temperature, a `.dc` sweep drives.
    pub const SweepTarget = union(enum) {
        device: Param,
        temp,

        /// A card parameter, keyed like `ParamRef`.
        pub const Param = struct {
            /// Set by the frontend from the swept card; `unset` names no card.
            type: DeviceType = .unset,
            /// Instance index within `type`.
            index: u32 = 0,
            param_name: []const u8 = "dc",
        };
    };

    /// True for a two-level sweep, whose outer level is `target2`.
    pub fn hasOuter(self: Dc) bool {
        return self.target2 != null;
    }
};

/// Incremental DC response to every source's AC magnitude (`.dcinc`).
pub const Dcinc = struct {
    tol: Tolerances = .{},
};

/// Monte Carlo over the DC solution.
pub const Mc = struct {
    tol: Tolerances = .{},
    n_trials: u16 = 100,
    /// PRNG seed; the same seed reproduces the same trials.
    seed: u64 = 42,
    /// Relative spread applied to every primary instance value.
    variation: f64 = 0.05,
    /// Options for each trial's DC solve.
    dc_options: Dc = .{},
    /// Rows `first..first + count` of `Deck.variants` to solve instead of
    /// perturbing primary values (`.dc DATA=`, `.dc MONTE=`): one lane
    /// each, warm-started from the nominal OP.
    variants: VariantRange = .{},
    /// First column of a variant ensemble; `axis_values` fills it.
    axis: []const u8 = "run",
    /// Publish a variant ensemble as a DC sweep rather than Monte Carlo.
    dc_plot: bool = false,
    /// Solve only this trial (or variant `variants.first + lane`): one row
    /// of a query the session split into its lanes. Null solves them all.
    lane: ?u32 = null,

    /// Rows `first..first + count` of `Deck.variants`; none when `count` is 0.
    pub const VariantRange = struct { first: u32 = 0, count: u32 = 0 };
};

/// DC solution over a temperature sweep, in °C.
pub const Temp = struct {
    tol: Tolerances = .{},
    t_start: f64 = -40.0,
    t_stop: f64 = 125.0,
    t_step: f64 = 1.0,
    dc_options: Dc = .{},
    /// Solve only temperature `t_start + lane * t_step`: one row of a query
    /// the session split into its lanes. Null solves them all.
    lane: ?u32 = null,
};
