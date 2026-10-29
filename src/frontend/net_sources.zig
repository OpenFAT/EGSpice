//! Independent source cards. In: V and I cards. Out: their bound Model and
//! Instance, with ngspice's waveform rewrites (PULSE trains, AM).
//! Cites: ngspice vsrcload.c, isrcload.c, vsrctemp.c.
//!
//! Contents:
//! - `pub fn ngspiceWaveform(self: *const NetBuilder, dev: Device) !Device`
//! - `pub fn pulseTrain(self: *const NetBuilder, name: []const u8, args: []const Value) ![]const Value`
//! - `pub fn amArgs(self: *const NetBuilder, name: []const u8, args: []const Value) ![]const Value`
//! - `pub fn bindSource(self: *const NetBuilder, comptime D: type, dev: Device) !struct { D.Model, D.Instance }`

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
const net_publish = @import("net_publish.zig");
const net_waves = @import("net_waves.zig");
const net_cards = @import("net_cards.zig");
const NetBuilder = net_builder.NetBuilder;

/// `dev` with its waveform in the source models' argument order. PULSE's
/// eighth argument is ngspice's pulse count (vsrcload.c:120-135), which
/// becomes a PWL of that many pulses, and the ngspice dialect's
/// `AM(VO VMO VMA FM FC TD PHASEM PHASEC)` (vsrcload.c:283-315) maps onto
/// the models' HSPICE-order AM. Any other card comes back as it is.
pub fn ngspiceWaveform(self: *const NetBuilder, dev: Device) !Device {
    const w = net_waves.waveAt(dev) orelse return dev;
    const args: []const Value = switch (w.kind) {
        .pulse => if (w.args.len < 8) return dev else try pulseTrain(self, dev.name, w.args),
        .am => if (self.nl.deck.dialect == .hspice or w.args.len < 2) return dev else try amArgs(self, dev.name, w.args),
        else => return dev, // else: only PULSE and AM have an order to fix
    };
    const kind: []const u8 = if (w.kind == .am or args.len == 7) @tagName(w.kind) else "pwl";
    const items = try self.arena.alloc(Value, args.len + 1);
    items[0] = .{ .name = kind };
    @memcpy(items[1..], args);
    const positional = try self.arena.alloc(Value, dev.positional.len - (w.end - w.start) + 1);
    @memcpy(positional[0..w.start], dev.positional[0..w.start]);
    positional[w.start] = .{ .group = .{ .items = items } };
    @memcpy(positional[w.start + 1 ..], dev.positional[w.end..]);
    var out = dev;
    out.positional = positional;
    return out;
}

/// PULSE arguments with ngspice's pulse count NP (the eighth) applied:
/// the first seven when the count never ends the train before TSTOP,
/// else PWL points for NP pulses and then V1.
/// ponytail: a PWL, not a model parameter; the models carry no count, and
/// past 4096 pulses the card is refused.
pub fn pulseTrain(self: *const NetBuilder, name: []const u8, args: []const Value) ![]const Value {
    var a: [8]f64 = undefined;
    for (&a, args[0..8]) |*x, v| x.* = v.number() orelse return error.UnresolvedParameter;
    const v1, const v2, const td = .{ a[0], a[1], a[2] };
    const tr = if (a[3] != 0) a[3] else self.tran_step;
    const tf = if (a[4] != 0) a[4] else self.tran_step;
    const pw = if (a[5] != 0) a[5] else self.tran_stop;
    const per = @max(if (a[6] != 0) a[6] else self.tran_stop, tr + pw + tf);
    const end = td + a[7] * per;
    if (!(a[7] > 0) or end >= self.tran_stop) return args[0..7];
    const pulses = @ceil(a[7]);
    if (pulses > 4096) {
        if (!@import("builtin").is_test) std.log.err("{s}: a PULSE count past 4096 pulses", .{name});
        return error.UnsupportedCard;
    }
    var pts: std.ArrayList(Value) = .empty;
    try pts.appendSlice(self.arena, &.{ .{ .num = td }, .{ .num = v1 } });
    var t_prev = td;
    var v_prev = v1;
    var k: f64 = 0;
    corners: while (k < pulses) : (k += 1) {
        const base = td + k * per;
        const corners = [_][2]f64{ .{ base + tr, v2 }, .{ base + tr + pw, v2 }, .{ base + tr + pw + tf, v1 }, .{ base + per, v1 } };
        for (corners) |c| {
            if (c[0] == t_prev) continue;
            if (c[0] > end) {
                // ngspice holds V1 past NP*PER: cut the pulse there.
                const v_end = v_prev + (c[1] - v_prev) * (end - t_prev) / (c[0] - t_prev);
                try pts.appendSlice(self.arena, &.{ .{ .num = end }, .{ .num = v_end }, .{ .num = end }, .{ .num = v1 } });
                break :corners;
            }
            try pts.appendSlice(self.arena, &.{ .{ .num = c[0] }, .{ .num = c[1] } });
            t_prev, v_prev = c;
        }
    }
    return pts.items;
}

/// The ngspice dialect's `AM(VO VMO VMA FM FC TD PHASEM PHASEC)`, value
/// VO + (VMO + VMA sin(2 pi FM t' + PHASEM)) sin(2 pi FC t' + PHASEC) on
/// t' = t - TD, as the models' `AM(VA VO MF FC TD PHASEC PHASES)`, value
/// VA (VO + sin(2 pi MF t + PHASES)) sin(2 pi FC t + PHASEC): the delay
/// moves into the phases, and VMA = 0 holds the envelope at sin(90).
/// ponytail: a nonzero VO offset is refused; the models have no slot for it.
pub fn amArgs(self: *const NetBuilder, name: []const u8, args: []const Value) ![]const Value {
    var a = [8]f64{ 0, 0, 1, 5 / self.tran_stop, 500 / self.tran_stop, 0, 0, 0 };
    for (a[0..@min(args.len, 8)], args[0..@min(args.len, 8)]) |*x, v| x.* = v.number() orelse return error.UnresolvedParameter;
    const vo, const vmo, const vma, const fm, const fc, const td, const phasem, const phasec = a;
    if (vo != 0) {
        if (!@import("builtin").is_test) std.log.err("{s}: AM with a nonzero VO offset", .{name});
        return error.UnsupportedCard;
    }
    const out = try self.arena.alloc(Value, 7);
    const ours = if (vma != 0)
        [7]f64{ vma, vmo / vma, fm, fc, td, phasec - 360 * fc * td, phasem - 360 * fm * td }
    else
        [7]f64{ vmo, 0, 0, fc, td, phasec - 360 * fc * td, 90 };
    for (out, ours) |*o, x| o.* = .{ .num = x };
    return out;
}

/// Binds a V or I card (`V1 a b DC 5 PULSE(0 5 1n 1n 1n 1u 2u)`), all but
/// the nodes. Each step runs over both Model and Instance behind
/// @hasField guards, so a model that moves a parameter keeps binding.
///
/// The order is fixed: `DC <v>` first, then the waveform group, then the
/// card pairs (so `pulse_tr=2n` overrides the group), and only then
/// `resolvePulseDefaults`, so the -1 sentinels survive every legitimate
/// write before the `.tran` card fills them.
pub fn bindSource(self: *const NetBuilder, comptime D: type, dev: Device) !struct { D.Model, D.Instance } {
    try net_waves.refuseUnknownWave(dev);
    var model: D.Model = .{};
    var instance: D.Instance = .{};
    const dc = net_cards.sourceDc(dev);
    _ = try net_builder.setParam(D, &model, &instance, "dc", dc orelse 0);
    net_waves.applySourceWaveform(&model, dev);
    net_waves.applySourceWaveform(&instance, dev);
    try net_cards.applyKv(&model, dev.kv);
    try net_cards.applyKv(&instance, dev.kv);
    // The PWL suffixes `td=` and `r=` (vsrcpar.c VSRC_TD, VSRC_R).
    if (net_cards.kvNumber(dev.kv, "td")) |td| _ = try net_builder.setParam(D, &model, &instance, "pwl_td", td);
    if (net_cards.kvNumber(dev.kv, "r")) |r| if (net_waves.pwlArgs(dev)) |pts| {
        // vsrcpar.c:160-176: the repeat start is one of the table's
        // times and before its last.
        var on_point = false;
        var k: usize = 0;
        while (k + 1 < pts.len) : (k += 2) on_point = on_point or pts[k].number() == r;
        const last = if (pts.len >= 2) pts[pts.len - pts.len % 2 - 2].number() orelse r else r;
        if (!on_point or r >= last) {
            if (!@import("builtin").is_test) std.log.err("{s}: PWL r={e} is not a time point before the last", .{ dev.name, r });
            return error.InvalidParameterValue;
        }
        _ = try net_builder.setParam(D, &model, &instance, "pwl_repeat", r);
    };
    net_publish.resolvePulseDefaults(self, &model);
    net_publish.resolvePulseDefaults(self, &instance);
    // ngspice: a source with a transient spec and no DC value has the
    // t = 0 value as DC. Resolved here so the model reads `dc`
    // unconditionally and a .dc sweep can override it.
    if (dc == null) {
        net_waves.dcFromWaveform(&model);
        net_waves.dcFromWaveform(&instance);
    }
    return .{ model, instance };
}

// ---- private ----
