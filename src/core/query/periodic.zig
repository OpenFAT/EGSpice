//! Periodic and quasi-periodic steady-state requests (shooting, harmonic
//! balance) and the small-signal and noise analyses about their orbits. One
//! options struct per card; `HbLptv.hb`, `Hblin.lptv` and `PhaseNoise.hb`
//! derive the `.hb` solve each linearizes about.
//! Cites: HSPICE [CR .HB; RF Ch.7]; ngspice-style `.pss`, `.pac`, `.pnoise`.
//!
//! Contents:
//! - `pub const Pss = struct`
//! - `pub const Hb = struct`
//! - `pub const Pac = struct`
//! - `pub const Pnoise = struct`
//! - `pub const HbLptv = struct`, `HbLptv.hb(self: HbLptv) Hb`
//! - `pub const Hblin = struct`, `Hblin.lptv(self: Hblin) HbLptv`
//! - `pub const PhaseNoise = struct`, `PhaseNoise.Algorithm = enum(u2)`,
//!   `PhaseNoise.hb(self: PhaseNoise) Hb`
//! - `pub const Qpss = struct`

const std = @import("std");
const assert = std.debug.assert;
const Tolerances = @import("../numerics.zig").Tolerances;
const FreqSweep = @import("../freq_sweep.zig").FreqSweep;
const GROUND = @import("../ids.zig").GROUND;
const Port = @import("small_signal.zig").Port;

/// Periodic steady state by shooting (`.pss`; autonomous with `osc_node`).
pub const Pss = struct {
    tol: Tolerances = .{},
    /// Seconds.
    period: f64,
    max_shooting_iter: u16 = 50,
    shooting_tol: f64 = 1e-7,
    /// Perturbation of the finite-difference monodromy Jacobian.
    fd_epsilon: f64 = 1e-7,
    max_newton_iter: u16 = 50,
    newton_tol: f64 = 1e-9,
    /// Fixed trapezoidal steps per period; the waveform has n_samples + 1 rows.
    n_samples: u32 = 256,
    /// GMRES restart depth; used only on the Krylov path for large circuits.
    gmres_restart: u32 = 30,
    gmres_max_restarts: u32 = 10,
    /// Relative tolerance of the inner GMRES solve.
    gmres_tol: f64 = 1e-3,
    /// Oscillator node row of an autonomous solve (`.snosc`). The period is
    /// then an unknown and `period` only its first guess, and this node's
    /// value at t = 0 is pinned as the phase condition. GROUND means a
    /// driven circuit with a known period.
    osc_node: u32 = 0,
    /// Guess periods an autonomous solve integrates from the kicked DC
    /// point before it measures the period, so the oscillator has settled
    /// onto its limit cycle (HSPICE's TRINIT, counted in periods).
    osc_settle_periods: u16 = 30,
};

/// Periodic or quasi-periodic steady state by harmonic balance (`.hb`;
/// autonomous `.hbosc` with `osc_node`).
pub const Hb = struct {
    tol: Tolerances = .{},
    /// Fundamental, in Hz.
    f0: f64,
    n_harmonics: u16 = 8,
    max_iter: u16 = 200,
    hb_tol: f64 = 1e-9,
    /// Oscillator node row of an autonomous solve (`.hbosc`): f0 is then an
    /// unknown seeded from the oscillator's shooting orbit, and this node's
    /// fundamental is held a pure cosine as the phase condition. GROUND
    /// means a driven circuit with a known f0.
    osc_node: u32 = 0,
    /// Tones after `f0`, HSPICE `.hb TONES=f0 f1 ...` [CR .HB]; empty for
    /// one tone. Driven only.
    extra_tones: []const f64 = &.{},
    /// Harmonics kept of each of `extra_tones`, parallel to it (`n_harmonics` is
    /// f0's).
    extra_harmonics: []const u16 = &.{},
    /// HSPICE INTMODMAX: the largest |k_0| + |k_1| + ... a kept mixing
    /// product k_0·f0 + k_1·f1 + ... may have. A single tone's own
    /// harmonics are kept up to its count regardless, and every |k_i| stays
    /// within tone i's count. 0 keeps that whole box.
    intmodmax: u16 = 0,
    /// HSPICE SUBHARMS: `f0` is the card's lowest tone divided by this and
    /// `n_harmonics` its NHARMS times this, so every subharmonic step is a
    /// line. In the INTMODMAX order, `subharms` steps of f0 count as one.
    subharms: u16 = 1,
    /// Publish complex phasors X = c - j·s per line, x(t) = Re{X·e^(jωt)},
    /// instead of magnitudes: HSPICE's `.hb TONES=` form [CR .HB]. The
    /// positional `.hb f0 K` keeps magnitudes.
    phasors: bool = false,
};

/// Periodic AC (`.pac`) and periodic transfer function (`.pxf`) options.
pub const Pac = struct {
    tol: Tolerances = .{},
    /// LO (pump) frequency in Hz: the fundamental periodicity.
    f_lo: f64,
    /// Output row: PAC's probe, PXF's injection point.
    out_node: u32,
    /// LO harmonics kept: sidebands span `-n_harmonics..n_harmonics`.
    n_harmonics: u16 = 3,
    /// Input frequency sweep.
    sweep: FreqSweep,
    /// Time samples per LO period. Must be a power of two and at least
    /// 2 * (2 * n_harmonics + 1); `validate` refuses anything else.
    n_time_samples: u16 = 64,
    pss_newton_tol: f64 = 1e-9,
    pss_max_newton_iter: u16 = 50,
};

/// Periodic noise about the shooting orbit (`.pnoise`, HSPICE `.ptdnoise`).
pub const Pnoise = struct {
    tol: Tolerances = .{},
    out_node: u32,
    /// `v(a,b)` reference: the measurement is v(out_node) - v(out_neg).
    /// GROUND is the single-ended case.
    out_neg: u32 = 0,
    sweep: FreqSweep,
    /// Hz.
    f_fundamental: f64,
    /// Time samples per period, rounded up to a power of two of at least
    /// 2 * (2 * n_sidebands + 1).
    pss_n_samples: u32 = 64,
    pss_shoot_tol: f64 = 1e-6,
    pss_shoot_max_iter: u16 = 50,
    pss_newton_max_iter: u16 = 50,
    pss_newton_tol: f64 = 1e-9,
    /// Sidebands kept on each side of the carrier.
    n_sidebands: u16 = 7,
    /// HSPICE `.ptdnoise`: the noise density at this time of the period, in
    /// seconds, instead of the time average; see pnoise.zig `strobed`.
    strobe: ?f64 = null,
};

/// Small-signal analyses about the harmonic-balance solution: `.hbac`
/// (periodic AC), `.hbxf` (periodic transfer function) and `.hbnoise`
/// (periodic noise) run the `.pac`, `.pxf` and `.pnoise` sweeps on the
/// `.hb` orbit instead of the shooting one.
pub const HbLptv = struct {
    tol: Tolerances = .{},
    /// HB fundamental, in Hz.
    f0: f64,
    /// HB harmonics kept.
    n_harmonics: u16 = 8,
    /// Sidebands kept on each side of the carrier in the conversion matrix.
    n_sidebands: u16 = 8,
    /// Output row: `.hbac`'s probe, `.hbxf`'s injection point, `.hbnoise`'s
    /// measured node.
    out_node: u32,
    /// `.hbnoise v(a,b)` reference node; GROUND is single-ended. `validate`
    /// refuses any other on `.hbac` and `.hbxf`, which read v(out_node).
    out_neg: u32 = 0,
    /// Input (offset) frequency sweep.
    sweep: FreqSweep,
    max_iter: u16 = 200,
    hb_tol: f64 = 1e-9,
    /// The `.hb TONES=` card's other tones, as `Hb`'s fields. With any
    /// extra tone the sidebands are the HB spectrum's lines, both signs,
    /// and `n_sidebands` is unused.
    extra_tones: []const f64 = &.{},
    extra_harmonics: []const u16 = &.{},
    intmodmax: u16 = 0,
    subharms: u16 = 1,

    /// The driven `.hb` solve this analysis linearizes about: every HB
    /// field copied, `osc_node` left at GROUND.
    pub fn hb(self: HbLptv) Hb {
        const out: Hb = .{
            .tol = self.tol,
            .f0 = self.f0,
            .n_harmonics = self.n_harmonics,
            .max_iter = self.max_iter,
            .hb_tol = self.hb_tol,
            .extra_tones = self.extra_tones,
            .extra_harmonics = self.extra_harmonics,
            .intmodmax = self.intmodmax,
            .subharms = self.subharms,
        };
        assert(out.osc_node == GROUND);
        return out;
    }
};

/// HSPICE RF `.hblin`: frequency-translation S-parameters between the
/// deck's ports about the `.hb` orbit, each port read in its own band
/// (`Port.band`).
pub const Hblin = struct {
    tol: Tolerances = .{},
    /// HB fundamental, in Hz; 0 until the deck's `.hb` card fills it.
    f0: f64,
    n_harmonics: u16 = 8,
    /// Sidebands kept on each side of the carrier; every port's band
    /// harmonic must lie within them.
    n_sidebands: u16 = 8,
    /// Input (small-signal tone) frequency sweep.
    sweep: FreqSweep,
    max_iter: u16 = 200,
    hb_tol: f64 = 1e-9,
    ports: []const Port,
    /// `noisecalc=1`: append the port 1 to port 2 noise figure (two or
    /// more ports).
    noise: bool = false,

    /// The HB-orbit small-signal options the orbit and linearization take:
    /// one tone, no output row of its own (the ports are the outputs).
    pub fn lptv(self: Hblin) HbLptv {
        const out: HbLptv = .{
            .tol = self.tol,
            .f0 = self.f0,
            .n_harmonics = self.n_harmonics,
            .n_sidebands = self.n_sidebands,
            .out_node = GROUND,
            .sweep = self.sweep,
            .max_iter = self.max_iter,
            .hb_tol = self.hb_tol,
        };
        assert(out.extra_tones.len == 0);
        return out;
    }
};

/// Oscillator phase noise (`.phasenoise`) about the autonomous HB solution.
pub const PhaseNoise = struct {
    tol: Tolerances = .{},
    /// Oscillation frequency guess, in Hz.
    f0: f64,
    /// HB harmonics kept.
    n_harmonics: u16 = 8,
    /// Oscillator phase node (`Hb.osc_node`); never GROUND.
    osc_node: u32,
    /// Offset-from-carrier sweep.
    sweep: FreqSweep,
    max_iter: u16 = 200,
    hb_tol: f64 = 1e-9,
    method: Algorithm = .nlp,
    /// HSPICE CARRIERINDEX: the harmonic, 1..n_harmonics, whose phase noise
    /// is reported.
    carrier: u16 = 1,

    /// HSPICE METHOD=0|1|2 [RF Ch.7].
    pub const Algorithm = enum(u2) {
        /// The perturbation projection vector projects every source onto
        /// the phase (Demir's method).
        nlp,
        /// Periodic noise at the carrier sideband over the carrier power:
        /// phase and amplitude noise.
        pac,
        /// `nlp` close in, `pac` from where the two first agree.
        bpn,
    };

    /// The autonomous `.hb` solve the phase noise is computed about.
    /// Asserts that `osc_node` is not GROUND.
    pub fn hb(self: PhaseNoise) Hb {
        assert(self.osc_node != GROUND);
        const out: Hb = .{
            .tol = self.tol,
            .f0 = self.f0,
            .n_harmonics = self.n_harmonics,
            .max_iter = self.max_iter,
            .hb_tol = self.hb_tol,
            .osc_node = self.osc_node,
        };
        assert(out.extra_tones.len == 0);
        return out;
    }
};

/// Two-tone quasi-periodic steady state.
pub const Qpss = struct {
    tol: Tolerances = .{},
    /// The two fundamentals, in Hz.
    f1: f64,
    f2: f64,
    /// Harmonics kept of `f1` and `f2`.
    k1: u16 = 5,
    k2: u16 = 5,
    /// Newton iterations of the multi-tone HB solve (`.hb`'s `max_iter`).
    max_newton: u16 = 50,
    hb_tol: f64 = 1e-9,
};

test "the HB-orbit views carry the fields their solve reads" {
    const t = std.testing;
    const tones = [_]f64{1.1e9};
    const harmonics = [_]u16{2};
    const sweep: FreqSweep = .{ .f_start = 1, .f_stop = 10 };
    const l: HbLptv = .{ .tol = .{ .reltol = 1e-4 }, .f0 = 1e9, .n_harmonics = 5, .out_node = 3, .sweep = sweep, .max_iter = 7, .hb_tol = 1e-6, .extra_tones = &tones, .extra_harmonics = &harmonics, .intmodmax = 4, .subharms = 2 };
    const h = l.hb();
    try t.expectEqual(@as(f64, 1e-4), h.tol.reltol);
    try t.expectEqual(@as(f64, 1e9), h.f0);
    try t.expectEqual(@as(u16, 5), h.n_harmonics);
    try t.expectEqual(@as(u16, 7), h.max_iter);
    try t.expectEqual(@as(f64, 1e-6), h.hb_tol);
    try t.expectEqual(@as(usize, 1), h.extra_tones.len);
    try t.expectEqual(@as(usize, 1), h.extra_harmonics.len);
    try t.expectEqual(@as(u16, 4), h.intmodmax);
    try t.expectEqual(@as(u16, 2), h.subharms);
    try t.expectEqual(GROUND, h.osc_node);
    const lin: Hblin = .{ .tol = .{ .reltol = 1e-5 }, .f0 = 2e9, .n_harmonics = 6, .n_sidebands = 3, .sweep = sweep, .max_iter = 9, .hb_tol = 1e-7, .ports = &.{} };
    const lp = lin.lptv();
    try t.expectEqual(@as(f64, 1e-5), lp.tol.reltol);
    try t.expectEqual(@as(f64, 2e9), lp.f0);
    try t.expectEqual(@as(u16, 6), lp.n_harmonics);
    try t.expectEqual(@as(u16, 3), lp.n_sidebands);
    try t.expectEqual(@as(u16, 9), lp.max_iter);
    try t.expectEqual(@as(f64, 1e-7), lp.hb_tol);
    const pn: PhaseNoise = .{ .f0 = 5e6, .osc_node = 4, .sweep = sweep, .n_harmonics = 3 };
    try t.expectEqual(@as(u32, 4), pn.hb().osc_node);
    try t.expectEqual(@as(u16, 3), pn.hb().n_harmonics);
}
