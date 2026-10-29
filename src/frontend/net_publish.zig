//! The deck tables the queries read. In: the finished card pass. Out: the
//! AC excitation vector, the `.sp` port list, and PULSE defaults resolved
//! against the first `.tran`.
//! Cites: ngspice acan.c:471-490, vsrcacld.c:175, traninit.c (PULSE defaults).
//!
//! Contents:
//! - `pub fn acExcitation(self: *const NetBuilder, arena: std.mem.Allocator, n: usize) ![]f64`
//! - `pub fn portList(self: *const NetBuilder, arena: std.mem.Allocator) ![]requests.Port`
//! - `pub fn resolvePulseDefaults(self: *const NetBuilder, target: anytype) void`

const std = @import("std");
const core = @import("core");
const requests = core.query;
const numerics = core.numerics;
const devices = @import("spice.zig");
const sparam = @import("sparam.zig");
const wfit = @import("wfit.zig");
const device = @import("device");
const netlist = @import("netlist");
const Netlist = netlist.Netlist;
const Device = Netlist.View;
const Model = netlist.Model;
const Kv = netlist.Kv;
const Value = netlist.Value;
const Op = netlist.expr.Op;
const batch = device.abi;
const Library = device.Library;
const DeviceType = batch.DeviceType;
const castField = batch.bind.castField;
const markGiven = batch.bind.markGiven;

const GROUND = core.GROUND;
const Circuit = device.Circuit;
const Proto = batch.Proto;

const net_builder = @import("net_builder.zig");
const net_waves = @import("net_waves.zig");
const NetBuilder = net_builder.NetBuilder;

/// The AC excitation as one stacked vector `[re(0..n), im(0..n)]`,
/// ngspice's post-CKTacLoad (CKTrhs, CKTirhs) pair. Every AC source lands
/// in it, so the sweep is one solve per point. All zero when no card names
/// `AC`. `ac` rows must already be post-permutation.
pub fn acExcitation(self: *const NetBuilder, arena: std.mem.Allocator, n: usize) ![]f64 {
    const exc = try arena.alloc(f64, 2 * n);
    @memset(exc, 0);
    const ac = self.ac.slice();
    for (ac.items(.pos), ac.items(.neg), ac.items(.re), ac.items(.im)) |pos, neg, re, im| {
        if (pos != GROUND) {
            exc[pos] -= re;
            exc[n + pos] -= im;
        }
        if (neg != GROUND) {
            exc[neg] += re;
            exc[n + neg] += im;
        }
    }
    return exc;
}

/// The deck's `.sp` ports ordered by `portnum`, as ngspice sorts CKTrfPorts
/// (vsrctemp.c:110-124). A gapped or duplicated numbering is an error
/// (vsrctemp.c:143-160). Empty when no V card carries `portnum`, which
/// leaves `.sp` on its one-port fallback.
pub fn portList(self: *const NetBuilder, arena: std.mem.Allocator) ![]requests.Port {
    var n_ports: usize = 0;
    for (self.ports.items) |entry| n_ports = @max(n_ports, entry.num);
    if (n_ports == 0) return &.{};
    const ports = try arena.alloc(requests.Port, n_ports);
    for (ports) |*p| p.branch = std.math.maxInt(u32); // unset
    for (self.ports.items) |entry| {
        std.debug.assert(entry.num >= 1); // sourcePort refuses 0
        const slot = &ports[entry.num - 1];
        if (slot.branch != std.math.maxInt(u32)) return error.DuplicatePortNumber;
        slot.* = entry.port;
    }
    for (ports) |p| if (p.branch == std.math.maxInt(u32)) return error.MissingPortNumber;
    return ports;
}

/// ngspice's `.tran`-card defaults (vsrcload.c:102-215): PULSE TR/TF are
/// TSTEP and PW/PER TSTOP when omitted or 0 (the model's -1 sentinel is
/// omitted), SIN's frequency 1/TSTOP, and EXP's TD1, TAU1 and TAU2 TSTEP
/// and TD2 TD1 + TSTEP (applyWaveArgs zeroes the EXP time constants
/// first). A no-op on a struct without the PULSE fields, so it can run
/// over both Model and Instance.
pub fn resolvePulseDefaults(self: *const NetBuilder, target: anytype) void {
    const T = @TypeOf(target.*);
    if (comptime !@hasField(T, "pulse_tr")) return;
    const tstep = self.tran_step;
    const tstop = self.tran_stop;
    // Only a PULSE waveform gets the TRANinit fill. Any other waveform
    // parks TD past every tstop so its unused pulse fields make no breakpoint.
    if (comptime @hasField(T, "waveform")) {
        // SFFM's CKTfinalTime defaults: FM = 5/TSTOP when omitted, FC =
        // 500/TSTOP when omitted or 0 (vsrcload.c:237-243, isrcload.c:215-221).
        if (comptime @hasField(T, "sffm_fm")) if (target.waveform == @backingInt(net_waves.Wave.sffm)) {
            if (target.sffm_fm == -1.0) target.sffm_fm = @floatCast(5.0 / tstop);
            if (target.sffm_fc == 0.0) target.sffm_fc = @floatCast(500.0 / tstop);
        };
        if (comptime @hasField(T, "sin_freq")) if (target.waveform == @backingInt(net_waves.Wave.sin)) {
            if (target.sin_freq == 0) target.sin_freq = @floatCast(1.0 / tstop);
        };
        if (comptime @hasField(T, "exp_td1")) if (target.waveform == @backingInt(net_waves.Wave.exp)) {
            if (target.exp_td1 == 0) target.exp_td1 = @floatCast(tstep);
            if (target.exp_tau1 == 0) target.exp_tau1 = @floatCast(tstep);
            if (target.exp_td2 == 0) target.exp_td2 = @floatCast(target.exp_td1 + tstep);
            if (target.exp_tau2 == 0) target.exp_tau2 = @floatCast(tstep);
        };
        if (target.waveform != @backingInt(net_waves.Wave.pulse)) {
            target.pulse_td = 1e30;
            return;
        }
    }
    if (target.pulse_tr <= 0) target.pulse_tr = tstep;
    if (target.pulse_tf <= 0) target.pulse_tf = tstep;
    if (target.pulse_pw <= 0) target.pulse_pw = tstop;
    if (target.pulse_per <= 0) target.pulse_per = tstop;
}

// ---- private ----
