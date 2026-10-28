//! Resolved query descriptions: the contract between the frontend, which
//! fills them from analysis cards (keywords in frontend/netlist.zig `cards`),
//! and the analysis drivers. A card in, one `Query` row out. This file owns
//! the tag (`Kind`) and the union (`Query`); each family of options structs
//! lives in `query/` and is re-exported here, so `core.query.X` names every
//! one. Node and branch fields are MNA rows; `validate.zig` checks the
//! numeric fields and row indices before a driver runs.
//! Cites: the ngspice sources and HSPICE manual sections each family file
//! names; include/espice.h (the `Kind` tag values are its C ABI).
//!
//! Contents:
//! - `pub const QueryId = enum(u32) { _ }`, `pub const invalid_query: QueryId`
//! - `pub const Kind = enum(u8)`, `Kind.transient(kind: Kind) bool`
//! - `pub const Query = union(Kind)`, `Query.uic(job: Query) bool`
//! - `pub const FreqSweep`, `pub const SweepKind` (`freq_sweep.zig`)
//! - small signal (`query/small_signal.zig`): `Port`, `Ac`, `Noise`,
//!   `NoiseSample`, `Sp`, `Stb`, `Lstb`, `XfSource`, `Dcxf`, `Acxf`, `Tf`, `Pz`, `Disto`
//! - DC (`query/dc.zig`): `Op`, `Dc`, `Dcinc`, `Mc`, `Temp`
//! - mismatch (`query/sensitivity.zig`): `Variations`, `Dcmatch`, `Acmatch`, `Dcsens`, `Sens`
//! - periodic (`query/periodic.zig`): `Pss`, `Hb`, `Pac`, `Pnoise`, `HbLptv`,
//!   `Hblin`, `PhaseNoise`, `Qpss`
//! - time domain (`query/transient.zig`): `Method`, `Tran`, `PointOptions`,
//!   `TranNoise`, `Envelope`, `Matex`, `Four`, `Fft`
//! - labels (`query/card_names.zig`): `CardRef`, `CardNames`

const std = @import("std");
const freq_sweep = @import("freq_sweep.zig");
const small_signal = @import("query/small_signal.zig");
const dc = @import("query/dc.zig");
const sensitivity = @import("query/sensitivity.zig");
const periodic = @import("query/periodic.zig");
const transient = @import("query/transient.zig");
const card_names = @import("query/card_names.zig");

/// Dense id of a query within one session.
pub const QueryId = enum(u32) { _ };
/// Sentinel `QueryId` that names no query.
pub const invalid_query: QueryId = @fromBackingInt(std.math.maxInt(u32));

/// Re-exported so a query reads its sweep type from one place.
pub const FreqSweep = freq_sweep.FreqSweep;
/// Reachable by embedders only here: `espice.requests` exports `query`, not `numerics`.
pub const SweepKind = freq_sweep.SweepKind;

pub const Port = small_signal.Port;
pub const Ac = small_signal.Ac;
pub const Noise = small_signal.Noise;
pub const NoiseSample = small_signal.NoiseSample;
pub const Sp = small_signal.Sp;
pub const Stb = small_signal.Stb;
pub const Lstb = small_signal.Lstb;
pub const XfSource = small_signal.XfSource;
pub const Dcxf = small_signal.Dcxf;
pub const Acxf = small_signal.Acxf;
pub const Tf = small_signal.Tf;
pub const Pz = small_signal.Pz;
pub const Disto = small_signal.Disto;

pub const Op = dc.Op;
pub const Dc = dc.Dc;
pub const Dcinc = dc.Dcinc;
pub const Mc = dc.Mc;
pub const Temp = dc.Temp;

pub const Variations = sensitivity.Variations;
pub const Dcmatch = sensitivity.Dcmatch;
pub const Acmatch = sensitivity.Acmatch;
pub const Dcsens = sensitivity.Dcsens;
pub const Sens = sensitivity.Sens;

pub const Pss = periodic.Pss;
pub const Hb = periodic.Hb;
pub const Pac = periodic.Pac;
pub const Pnoise = periodic.Pnoise;
pub const HbLptv = periodic.HbLptv;
pub const Hblin = periodic.Hblin;
pub const PhaseNoise = periodic.PhaseNoise;
pub const Qpss = periodic.Qpss;

pub const Method = transient.Method;
pub const Tran = transient.Tran;
pub const PointOptions = transient.PointOptions;
pub const TranNoise = transient.TranNoise;
pub const Envelope = transient.Envelope;
pub const Matex = transient.Matex;
pub const Four = transient.Four;
pub const Fft = transient.Fft;

pub const CardRef = card_names.CardRef;
pub const CardNames = card_names.CardNames;

/// Tag of `Query`: one per analysis.
pub const Kind = enum(u8) {
    ac,
    dc,
    dcmatch,
    disto,
    envelope,
    four,
    hb,
    matex,
    mc,
    noise,
    op,
    pac,
    pnoise,
    pss,
    pxf,
    pz,
    qpss,
    sens,
    sp,
    stb,
    temp,
    tf,
    tran,
    tran_noise,
    // Appended after the alphabetical block: the tag values are the C ABI
    // (include/espice.h).
    hbac,
    hbnoise,
    hbxf,
    phasenoise,
    lstb,
    acxf,
    dcxf,
    dcinc,
    fft,
    acmatch,
    dcsens,
    hblin,

    /// Runs off a transient operating point (ngspice MODETRANOP) and starts
    /// its devices in `.ic` rather than `.dc` state.
    pub fn transient(kind: Kind) bool {
        return switch (kind) {
            .tran, .four, .fft, .tran_noise, .envelope, .matex, .pss, .qpss, .pnoise, .pac, .pxf => true,
            .ac, .dc, .dcmatch, .disto, .hb, .mc, .noise, .op, .pz, .sens, .sp, .stb, .temp, .tf, .hbac, .hbnoise, .hbxf, .phasenoise, .lstb, .acxf, .dcxf, .dcinc, .acmatch, .dcsens, .hblin => false,
        };
    }
};

/// One resolved analysis request. Slices inside borrow from the deck's arena.
pub const Query = union(Kind) {
    ac: Ac,
    dc: Dc,
    dcmatch: Dcmatch,
    disto: Disto,
    envelope: Envelope,
    four: Four,
    hb: Hb,
    matex: Matex,
    mc: Mc,
    noise: Noise,
    op: Op,
    pac: Pac,
    pnoise: Pnoise,
    pss: Pss,
    pxf: Pac,
    pz: Pz,
    qpss: Qpss,
    sens: Sens,
    sp: Sp,
    stb: Stb,
    temp: Temp,
    tf: Tf,
    tran: Tran,
    tran_noise: TranNoise,
    hbac: HbLptv,
    hbnoise: HbLptv,
    hbxf: HbLptv,
    phasenoise: PhaseNoise,
    lstb: Lstb,
    acxf: Acxf,
    dcxf: Dcxf,
    dcinc: Dcinc,
    fft: Fft,
    acmatch: Acmatch,
    dcsens: Dcsens,
    hblin: Hblin,

    /// The run starts from the `.ic` values with no operating point: a
    /// `.tran ... uic`, or a `.four` over one. An `.fft` carries its own
    /// transient but always runs it from an operating point.
    pub fn uic(job: Query) bool {
        return switch (job) {
            .tran => |o| o.uic,
            .four => |o| if (o.tran_opts) |t| t.uic else false,
            .ac, .dc, .dcmatch, .disto, .envelope, .hb, .matex, .mc, .noise, .op, .pac, .pnoise, .pss, .pxf, .pz, .qpss, .sens, .sp, .stb, .temp, .tf, .tran_noise, .hbac, .hbnoise, .hbxf, .phasenoise, .lstb, .acxf, .dcxf, .dcinc, .fft, .acmatch, .dcsens, .hblin => false,
        };
    }
};

test "every transient-family query runs off the transient operating point" {
    // MATEX integrates in time like .tran, so it starts from MODETRANOP (#27).
    inline for (.{ Kind.tran, Kind.matex, Kind.envelope, Kind.pss }) |k| try std.testing.expect(k.transient());
    try std.testing.expect(!Kind.dc.transient());
}

test "only a uic transient, or a .four over one, skips its operating point" {
    // Oracle: Query.uic's contract, read off the `.tran ... uic` card form.
    const t = std.testing;
    try t.expect((Query{ .tran = .{ .t_stop = 1, .uic = true } }).uic());
    try t.expect(!(Query{ .tran = .{ .t_stop = 1 } }).uic());
    try t.expect((Query{ .four = .{ .f_fundamental = 1, .tran_opts = .{ .t_stop = 1, .uic = true } } }).uic());
    try t.expect(!(Query{ .four = .{ .f_fundamental = 1 } }).uic());
    const fft: Fft = .{ .tran = .{ .t_stop = 1, .uic = true }, .out_pos = 1, .start = 0, .stop = 1, .label = "v(1)" };
    try t.expect(!(Query{ .fft = fft }).uic());
    try t.expect(!(Query{ .op = .{} }).uic());
}
