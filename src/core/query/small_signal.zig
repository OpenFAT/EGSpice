//! Small-signal requests about the DC operating point: AC and noise sweeps,
//! S-parameters and loop gain, DC and AC transfer functions, poles and
//! zeros, distortion. One options struct per card, filled by the frontend
//! from the card and read by its analysis driver; node and branch fields
//! are MNA rows, which `validate.zig` checks against the circuit.
//! Cites: ngspice acan.c, cktdisto.c:107-158, distoan.c:143, vsrcpar.c:180-193;
//! HSPICE [CR .SAMPLE, .LIN, .NET, .STB]; VACASK `acstb`.
//!
//! Contents:
//! - `pub const Port = struct`, `Port.Leg`, `Port.Band`, `Port.no_branch`
//! - `pub const Ac = struct`
//! - `pub const Noise = struct`, `pub const NoiseSample = struct`
//! - `pub const Sp = struct`, `Sp.Lin = struct`
//! - `pub const Stb = struct`
//! - `pub const Lstb = struct`, `Lstb.Probe`, `Lstb.Mode = enum(u8)`
//! - `pub const XfSource = struct`, `pub const Dcxf = struct`, `pub const Acxf = struct`
//! - `pub const Tf = struct`
//! - `pub const Pz = struct`, `Pz.Want = enum(u2)`
//! - `pub const Disto = struct`, `Disto.Plot = enum(u8)`

const std = @import("std");
const Tolerances = @import("../numerics.zig").Tolerances;
const FreqSweep = @import("../freq_sweep.zig").FreqSweep;
const GROUND = @import("../ids.zig").GROUND;
const CardNames = @import("card_names.zig").CardNames;

/// One `.sp` port: the node rows and branch row of its source and its
/// reference impedance in ohms. The port voltage is v(node) - v(neg).
pub const Port = struct {
    node: u32,
    branch: u32,
    z0: f64 = 50.0,
    neg: u32 = 0,
    /// The band `.hblin` reads this port in (an HSPICE P card's
    /// `hblin=[harmonic, sign]`): sign·f + harmonic·f0 for input frequency f.
    band: Band = .{},
    /// z0 is already a resistor in the circuit, in series between `node`
    /// and the source (an HSPICE P card), so the port analyses must not
    /// terminate the branch again. False for ngspice's ideal portnum source.
    series_z0: bool = false,
    /// A mixed-mode (balanced) P card's − leg, against the same `neg`
    /// reference and z0; `node`/`branch` are then its + leg. Null for a
    /// single-ended port.
    balanced: ?Leg = null,

    /// Node and branch row of one leg of a balanced port.
    pub const Leg = struct { node: u32, branch: u32 };

    /// `branch` of a port with no source branch, which `.net` drives with
    /// a current into `node`.
    pub const no_branch = std.math.maxInt(u32);

    /// `sign` is +1 or -1; the default reads the port at the input frequency.
    pub const Band = struct { harmonic: i16 = 0, sign: i8 = 1 };
};

/// Small-signal sweep about the operating point (`.ac`).
pub const Ac = struct {
    tol: Tolerances = .{},
    sweep: FreqSweep,
};

/// Small-signal noise at one output over a sweep (`.noise`, HSPICE
/// `.acphasenoise`).
pub const Noise = struct {
    tol: Tolerances = .{},
    out_node: u32,
    /// `v(a,b)` reference: the measurement is v(out_node) - v(out_neg).
    /// GROUND is the single-ended case.
    out_neg: u32 = 0,
    sweep: FreqSweep,
    /// Branch row of the `.noise v(out) SRC ...` input source. It does not
    /// drive the solve; `inoise_spectrum` refers the output noise back to
    /// it. Null emits the output-referred curve only.
    in_branch: ?u32 = null,
    /// The (n+, n-) rows of a current-source input, which drives current
    /// from n+ to n- through itself; set instead of `in_branch`.
    in_nodes: ?[2]u32 = null,
    /// Emit integrated device noise in V rms instead of the measured PSD.
    integrated: bool = false,
    /// Publish each device instance's contribution, per generator name and
    /// in total, ahead of the totals (ngspice `.noise ... pts`, HSPICE
    /// `.noise ... inter`).
    contributions: bool = false,
    /// Card names for the contribution columns. An unnamed instance falls
    /// back to `<type>#<ordinal>`.
    card_names: CardNames = .empty,
    /// HSPICE `.sample`: also publish the output noise folded by a sampler
    /// (`onoise_sampled`); spectrum queries only.
    sample: ?NoiseSample = null,
    /// HSPICE `.acphasenoise`: the output is a phase in radians, published
    /// as phase noise `phnoise` in dBc/Hz instead of the noise curves.
    phase: bool = false,
};

/// HSPICE `.sample FS= [MAXFLD=] [BETA=]` [CR .SAMPLE]: noise sampled at
/// `fs` Hz folds every band up to `max_fold * fs` into [0, fs/2], after an
/// integrator over `beta / fs` seconds (none when `beta` is 0).
pub const NoiseSample = struct {
    fs: f64,
    max_fold: f64 = 10,
    beta: f64 = 1,
};

/// S-parameters between ports over a sweep (`.sp`, HSPICE `.lin`, `.net`).
pub const Sp = struct {
    tol: Tolerances = .{},
    sweep: FreqSweep,
    /// Explicit port list. Empty means one port at the deck's drive source
    /// (`Deck.source_node`/`source_branch`).
    ports: []const Port = &.{},
    /// HSPICE `.lin` network parameters after S; null is the plain `.sp`
    /// S matrix.
    lin: ?Lin = null,
    /// HSPICE `.net`: the ports are ideal (a V card's port is a short, any
    /// other an open) and driven as such, and S comes from the measured
    /// Z against each port's z0 (RIN, ROUT). `lin` is set, without group
    /// delay or noise.
    net: bool = false,

    /// What `.lin` adds to the S matrix: Y and Z always, H for two or more
    /// ports (from the port 1-2 block), then the optional group delays and
    /// two-port noise parameters.
    pub const Lin = struct {
        /// `gdcalc=1`: the group delay of every S, Y, Z and H entry.
        group_delay: bool = false,
        /// `noisecalc=1`: NFMIN, NF, RN, YOPT and GAMMA_OPT between ports
        /// 1 and 2, the others terminated in their z0.
        noise: bool = false,
        /// `format=touchstone`: the facade also writes the result as a
        /// Touchstone file, `<file>.s<N>p` beside the deck.
        touchstone: bool = false,
        /// `filename=`; empty is the deck's file name without extension.
        file: []const u8 = "",
        /// `mixedmode2port=`: whether port 1's and port 2's mode in the
        /// two-port measurements (H, stability, noise) is its common mode
        /// (`c`) rather than its first one (`s` or `d`).
        common: [2]bool = .{ false, false },
    };
};

/// Loop gain through the deck's 0 V probe source (`.stb`).
pub const Stb = struct {
    tol: Tolerances = .{},
    sweep: FreqSweep,
    /// The deck's own 0 V probe source (`.stb Vprobe ...`): its two node rows
    /// and its branch row. `probe_p` is the arriving (driven) side.
    probe_p: u32,
    probe_n: u32,
    probe_branch: u32,
};

/// Loop stability by double injection (`.lstb`, VACASK `acstb`): a current
/// and a voltage injection at the probe, combined into the loop gain, the
/// forward and reverse gains and the DUT y-parameters.
pub const Lstb = struct {
    tol: Tolerances = .{},
    sweep: FreqSweep,
    mode: Mode = .single,
    /// The 0 V probe sources, HSPICE's orientation: `+` faces the loop's
    /// input (drv), `−` its output (fbk). `probes[1]` is read in `diff`
    /// and `comm` modes only.
    probes: [2]Probe,
    /// Publish the one-row margins plot instead of the sweep.
    margins: bool = false,
    /// VACASK's `localgnd`: the node the current injection returns to and
    /// the `+` node voltage is read against.
    local_gnd: u32 = GROUND,

    /// Node rows and branch row of one probe source.
    pub const Probe = struct { p: u32, n: u32, branch: u32 };
    /// Single-ended loop, or the differential (+1/−1) or common-mode
    /// (+1/+1) loop through a pair of probes.
    pub const Mode = enum(u8) { single, diff, comm };
};

/// One independent source of an all-source transfer (`.dcxf`, `.acxf`).
pub const XfSource = struct {
    /// Card name, for the result's column labels.
    name: []const u8,
    /// V card: its branch row. Null for an I card, whose unit current
    /// flows from `nodes[0]` through the card into `nodes[1]`.
    branch: ?u32,
    nodes: [2]u32 = .{ 0, 0 },
};

/// DC transfer from every independent source to one output (`.dcxf`).
pub const Dcxf = struct {
    tol: Tolerances = .{},
    /// Output row; unused when `output_branch` is set.
    output_node: u32,
    /// `v(a,b)` reference node; GROUND is single-ended.
    output_neg: u32 = 0,
    /// `i(Vmeasure)` output: that source's branch current.
    output_branch: ?u32 = null,
    sources: []const XfSource,
    /// Transfer functions only, from one adjoint solve; otherwise one
    /// forward solve per source also gives its input impedance.
    tf_only: bool = false,
};

/// `Dcxf` over a frequency sweep (`.acxf`).
pub const Acxf = struct {
    tol: Tolerances = .{},
    sweep: FreqSweep,
    output_node: u32,
    output_neg: u32 = 0,
    output_branch: ?u32 = null,
    sources: []const XfSource,
    tf_only: bool = false,
};

/// DC small-signal transfer function, input and output resistance (`.tf`).
pub const Tf = struct {
    tol: Tolerances = .{},
    /// `v(a,b)` reference node for the output; GROUND is single-ended.
    output_neg: u32 = 0,
    /// `.tf i(Vmeasure) ...`: the output is that source's branch current.
    /// When set, `output_node` and `output_neg` are unused.
    output_branch: ?u32 = null,
    /// `.tf v(out) Iin`: the input is a current source, which has no branch
    /// row, so the excitation is a unit current into this node pair. When
    /// set, `input_branch` is unused.
    input_nodes: ?[2]u32 = null,
    /// Branch row of the input V source. Null means `Deck.source_branch`.
    input_branch: ?u32 = null,
    /// Output row; unused when `output_branch` is set.
    output_node: u32,
};

/// Poles and zeros of the linearized circuit (`.pz`).
pub const Pz = struct {
    tol: Tolerances = .{},
    qr_max_iter: u32 = 1000,
    qr_tol: f64 = 1e-12,
    /// Output side of `.pz in+ in- out+ out- vol|cur pol|zer|pz`. A bare
    /// `.pz` leaves both at GROUND and asks for the circuit's own poles.
    out_pos: u32 = 0,
    out_neg: u32 = 0,
    /// Where the input drive enters. `vol` drives a voltage source, so this
    /// is its branch row; GROUND selects `cur`, a current injected at
    /// `in_pos`/`in_neg`. Same convention as `Disto.drive_branch`.
    drive_branch: u32 = 0,
    in_pos: u32 = 0,
    in_neg: u32 = 0,
    /// The card's `pol`, `zer` or `pz`.
    want: Want = .poles,

    /// Which roots the card asks for.
    pub const Want = enum(u2) { poles, zeros, both };
};

/// Small-signal distortion at one or two drive frequencies (`.disto`).
pub const Disto = struct {
    tol: Tolerances = .{},
    sweep: FreqSweep,
    /// Branch row of the V card carrying `DISTOF1`, where ngspice puts a
    /// voltage source's F1 drive (cktdisto.c:115). A node row would be
    /// pinned by the source's own branch equation and give V1(out) = 0.
    /// GROUND selects the current-source form below.
    drive_branch: u32 = 0,
    /// Current-source form (ngspice cktdisto.c:151-158): the drive is a
    /// current into this row, which takes -0.5·mag. GROUND means the drive
    /// source (`Deck.source_branch`).
    ac_source_node: u32 = 0,
    /// `DISTOF1 <mag> [<phase deg>]` off the card (ngspice vsrcpar.c:180-193).
    ac_magnitude: f64 = 1.0,
    ac_phase: f64 = 0.0,
    /// Row the summary plot measures. The frontend passes
    /// `Deck.output_node`, since the card names none.
    output_node: u32,
    fd_eps: f64 = 1e-6,
    /// The card's `f2overf1`: F2 = f2_ratio·f_start, held for the whole
    /// sweep as ngspice does (distoan.c:143). 0 means one tone.
    f2_ratio: f64 = 0,
    /// Branch row of the V card carrying `DISTOF2`, the F2 drive, and its
    /// `<mag> [<phase deg>]` (ngspice cktdisto.c:107-117).
    drive2_branch: u32 = 0,
    ac2_magnitude: f64 = 1.0,
    ac2_phase: f64 = 0.0,
    /// Which of the card's plots this query publishes. The frontend fans
    /// one `.disto` card out into all of them.
    plot: Plot = .summary,

    /// `second` and `third` are the harmonic solution vectors ngspice
    /// prints for one tone; with F2, ngspice prints the intermodulation
    /// vectors at f1+f2, f1-f2 and 2f1-f2 instead. `summary` is a 4-column
    /// digest at one node.
    pub const Plot = enum(u8) { summary, second, third, f1pf2, f1mf2, twof1mf2 };
};
