//! Time-domain requests: the transient and the per-point options the
//! points API adds to it, transient noise, envelope following, the
//! matrix-exponential transient, and the `.four` and `.fft` post-processors
//! over a transient. One options struct per card.
//! Cites: ngspice dctran.c:484 (TSTART), nicomcof.c (XMU), optran.c,
//! dotcards.c (`.four`); HSPICE [CR .FFT, .TRANNOISE, .OP time].
//!
//! Contents:
//! - `pub const Method = enum`
//! - `pub const Tran = struct`, `Tran.requestedCeiling(t: Tran, origin: enum { api, deck }) f64`
//! - `pub const PointOptions = struct`
//! - `pub const TranNoise = struct`
//! - `pub const Envelope = struct`
//! - `pub const Matex = struct`
//! - `pub const Four = struct`, `Four.max_harmonics`
//! - `pub const Fft = struct`, `Fft.Window = enum(u8)`

const std = @import("std");
const Tolerances = @import("../numerics.zig").Tolerances;

/// Transient integration method, and the highest order the step loop
/// promotes to: `gear_k` is variable-order Gear (BDF) of orders 1..k.
pub const Method = enum { backward_euler, trapezoidal, gear_2, gear_3, gear_4, gear_5, gear_6 };

/// Transient analysis (`.tran`, HSPICE `.op <time>`).
pub const Tran = struct {
    tol: Tolerances = .{},
    /// Seconds, like every time field below.
    t_stop: f64,
    dt_init: f64 = 1e-9,
    dt_min: f64 = 1e-18,
    /// ngspice tstart: output suppression only. Integration still starts at
    /// t = 0 with the same history; points before `t_start` are dropped and
    /// none is added at it, as in ngspice (dctran.c:484).
    t_start: f64 = 0,
    /// Explicit maximum step; wins over ceiling_policy.
    dt_max: ?f64 = null,
    /// Opt-in requested ceiling. Device/event bounds still apply.
    ceiling_policy: enum { compatibility, duration } = .compatibility,
    method: Method = .trapezoidal,
    /// ngspice `.options xmu`: the trapezoidal weight on the previous step's
    /// derivative (nicomcof.c). 0.5 is the plain trapezoid; below it each
    /// step damps the trapezoid's undamped ringing by xmu/(1 - xmu), down to
    /// backward Euler at 0. Range [0, 0.5].
    xmu: f64 = 0.5,
    /// Runaway guard only; `dt_min` is the real brake. A 1 s run on a 1 us
    /// grid is already 1e6 accepted points.
    max_steps: u32 = 1_000_000_000,
    /// `.tran ... uic`: no operating point runs; the starting state comes
    /// from the `.ic` cards (zero elsewhere), and the transient does the
    /// setup the operating point would have done.
    uic: bool = false,
    /// HSPICE `.op <time>`: publish only the state at `t_stop`, as an
    /// operating-point plot named for that time.
    snapshot: bool = false,
    /// Every device sees t = 0 while the steps advance, so sources hold
    /// their DC values: the operating point's OPtran rung, as ngspice's
    /// optran.c keeps CKTtime at 0 on its own clock.
    sources_at_zero: bool = false,
    /// `Deck.variants` rows the transient also predicts to first order on
    /// its own time grid, for the optimizer's Jacobian
    /// (dev/analysis/optimize.md, "Direct sensitivities"), as `Dc.tangents`
    /// does per sweep point. Each adds a block of the same time points after
    /// the waveform's own. Not with `uic`, `snapshot` or `sources_at_zero`.
    tangents: []const u32 = &.{},
    /// What `Problem.append_points` asked of this point.
    point: PointOptions = .{},

    /// The output-window span sets the implicit ceiling even though
    /// integration starts at zero. Deck compatibility additionally caps
    /// it at TSTEP; the direct API has historically omitted that cap.
    pub fn requestedCeiling(t: Tran, origin: enum { api, deck }) f64 {
        if (t.dt_max) |explicit| return explicit;
        const span = t.t_stop - t.t_start;
        return switch (t.ceiling_policy) {
            .duration => span / 10,
            .compatibility => if (origin == .deck) @min(t.dt_init, span / 50) else span / 50,
        };
    }
};

/// The points API's per-point requests (`Problem.append_points`), carried
/// on each point's transient (analysis/tran/tran.zig `run`). None changes a
/// `.tran` card's value, but `warm_start` may within Newton's tolerances.
pub const PointOptions = struct {
    /// Record only the columns the `.tran` cards read.
    measures_only: bool = false,
    /// End the transient once every `.tran` card has resolved; its
    /// waveform then stops there.
    early_stop: bool = false,
    /// Start the operating point from the template's nominal one
    /// (`Deck.warm_op`) instead of the cold start, falling back to the full
    /// ladder when that fails.
    warm_start: bool = false,
    /// Seconds past which a `.tran` card still unresolved fails the
    /// transient with `error.MeasureBoundExceeded`; null is off, 0 a bound.
    abort_after: ?f64 = null,
    /// The card `abort_after` bounds, by name; empty bounds every card.
    abort_measure: []const u8 = "",
    /// Above 0: also integrate, per live parameter, the first-order
    /// prediction of the point with that parameter moved by this times
    /// its magnitude (`Tran.tangents`), for d(measure)/d(parameter).
    tangent_step: f64 = 0,
};

/// Transient noise: a transient with sampled noise sources, or the noise
/// covariance carried alongside it (`.trannoise`).
pub const TranNoise = struct {
    tol: Tolerances = .{},
    /// Seconds, like every time field below.
    t_stop: f64,
    dt_init: f64 = 1e-9,
    dt_min: f64 = 1e-18,
    dt_max: f64 = 1e-3,
    /// Runaway guard only, as in `Tran.max_steps`.
    max_steps: u32 = 1_000_000_000,
    /// Noise generator seed; the same seed reproduces the same run.
    seed: u64 = 0xDEAD_BEEF_CAFE_1234,
    /// Multiplies every source's PSD (HSPICE `SCALE`).
    scale: f64 = 1,
    /// Hz; the lowest flicker-noise frequency, 1/t_stop when null (HSPICE
    /// `FMIN`). The highest is the sampling bandwidth 1/(2 dt_max).
    f_min: ?f64 = null,
    /// HSPICE Monte Carlo index of one run of a `SAMPLES>1` card, named in
    /// the plot; 0 for a lone run.
    sample: u32 = 0,
    /// HSPICE `METHOD=SDE`: no sampled noise; the noiseless march carries
    /// the noise covariance and publishes `onoise`, the rms noise of
    /// v(out_node, out_neg).
    sde: bool = false,
    out_node: u32 = 0,
    out_neg: u32 = 0,
    /// Seconds; HSPICE `TIME=`, a time the march lands on exactly.
    t_break: ?f64 = null,
};

/// Envelope-following transient: carrier periods integrated in full, the
/// envelope stepped across them (`.envelope`).
pub const Envelope = struct {
    tol: Tolerances = .{},
    /// Carrier period (1 / f_carrier), in seconds.
    t_carrier: f64,
    /// Total simulated time, in seconds.
    t_stop: f64,
    carrier_steps_per_period: u32 = 64,
    /// Carrier periods per outer envelope step; 1 samples every period.
    periods_per_outer_step: u32 = 1,
    /// Outer steps allowed before giving up.
    max_outer_steps: u32 = 1_000_000,
    /// The outer step halves when the envelope changes by more than this
    /// fraction between two outer steps.
    envelope_reltol: f64 = 0.05,
    /// Outer step bounds, in carrier periods.
    min_periods_per_step: u32 = 1,
    max_periods_per_step: u32 = 16,
};

/// Matrix-exponential transient (MATEX).
pub const Matex = struct {
    tol: Tolerances = .{},
    /// Seconds.
    t_stop: f64,
    /// Krylov posterior tolerance, the exponential's counterpart of reltol.
    krylov_tol: f64 = 1e-10,
    /// Largest Krylov subspace before the step fails.
    m_max: u32 = 80,
    /// R-MATEX shift γ, on the order of the intended step; results are
    /// insensitive to it. Null means t_stop / 1000.
    gamma: ?f64 = null,
    /// Largest step between recorded points. Null means t_stop / 200.
    h_output_cap: ?f64 = null,
    /// Recorded-point ceiling; sizes the initial waveform allocation.
    max_points: u32 = 1 << 22,
};

/// Fourier coefficients of one transient output (`.four`).
pub const Four = struct {
    tol: Tolerances = .{},
    /// Hz.
    f_fundamental: f64,
    n_harmonics: u16 = 9,
    output_node: u32 = 0,
    /// `v(a,b)` reference: the harmonics are of v(output_node) - v(output_neg).
    output_neg: u32 = 0,
    /// The deck's `.tran`, whose last period is analyzed, as ngspice's
    /// `.four` reads the tran plot (dotcards.c). Null, for a deck without a
    /// `.tran`, means 5 fundamental periods at 200 points per period.
    tran_opts: ?Tran = null,
    /// The output as written (`v(b)`) when the card named several: it goes
    /// into the plot name so the plots stay apart. Empty for one output.
    label: []const u8 = "",

    /// Size of the fixed harmonic table the extractor returns by value.
    pub const max_harmonics = 64;
};

/// HSPICE `.fft` [CR .FFT]: the windowed spectrum of one transient output.
pub const Fft = struct {
    tol: Tolerances = .{},
    /// The deck's transient, run to at least `stop`.
    tran: Tran,
    out_pos: u32,
    /// `v(a,b)` reference; GROUND is single-ended.
    out_neg: u32 = 0,
    /// Window on the waveform, seconds.
    start: f64,
    stop: f64,
    /// Uniform samples in [start, stop); a power of two.
    np: u32 = 1024,
    window: Window = .rect,
    /// GAUSS and KAISER shape parameter.
    alfa: f64 = 3,
    /// FORMAT=NORM: magnitudes relative to the largest non-DC bin.
    normalized: bool = true,
    /// The output as written, which names the plot (`v(out)`).
    label: []const u8,

    /// HSPICE's eight windows, named as the card spells them [SA Ch.15 Table 56].
    pub const Window = enum(u8) { rect, bart, hann, hamm, black, harris, gauss, kaiser };
};

test "transient ceiling policy preserves explicit, deck and API precedence" {
    const t = std.testing;
    var q: Tran = .{ .t_stop = 100, .t_start = 20, .dt_init = 0.1 };
    try t.expectEqual(@as(f64, 0.1), q.requestedCeiling(.deck));
    try t.expectEqual(@as(f64, 1.6), q.requestedCeiling(.api));
    q.ceiling_policy = .duration;
    try t.expectEqual(@as(f64, 8), q.requestedCeiling(.api));
    try t.expectEqual(@as(f64, 8), q.requestedCeiling(.deck));
    q.dt_max = 0.2;
    try t.expectEqual(@as(f64, 0.2), q.requestedCeiling(.api));
    try t.expectEqual(@as(f64, 0.2), q.requestedCeiling(.deck));
    const short: Tran = .{ .t_stop = 2e-12, .dt_init = 2e-12 };
    try t.expectEqual(@as(f64, 4e-14), short.requestedCeiling(.deck));
}
