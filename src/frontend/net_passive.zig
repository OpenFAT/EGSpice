//! Passive and controlled cards. In: R, C, L, K, E, F, G, H and switch
//! cards. Out: their instances; F, H and W cards bound to the V card whose
//! branch they sense.
//! Cites: ngspice resparam.c, capparam.c, indparam.c, mutsetup.c,
//! cccsset.c, ccvsset.c.
//!
//! Contents:
//! - `pub fn addPassive(self: *NetBuilder, comptime D: type, dev: Device, comptime value_field: []const u8, comptime alias: []const u8) !u32`
//! - `pub fn addBranchRef(self: *NetBuilder, comptime D: type, dev: Device, comptime default_gain: ?f64) !void`
//! - `pub fn addKinduc(self: *NetBuilder, dev: Device) !void`

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
const net_branches = @import("net_branches.zig");
const net_devices = @import("net_devices.zig");
const net_cards = @import("net_cards.zig");
const NetBuilder = net_builder.NetBuilder;

/// R, C or L card. The value (`R1 a b 1k`) goes to parameter
/// `value_field`; `alias` is a card spelling only (`resist=1k`), never a
/// field. Returns the row of the device's first internal unknown.
pub fn addPassive(
    self: *NetBuilder,
    comptime D: type,
    dev: Device,
    comptime value_field: []const u8,
    comptime alias: []const u8,
) !u32 {
    if (comptime !@hasDecl(D, "eval")) return error.UnsupportedDevice;
    if (comptime !@hasField(D.Model, value_field) and !@hasField(D.Instance, value_field))
        @compileError(@typeName(D) ++ ": no parameter `" ++ value_field ++ "` on Model or Instance");
    var model: D.Model = .{};
    var instance: D.Instance = .{};
    // A value that did not fold (an undefined parameter) is an error, not
    // the 1 mOhm or zero that a missing value defaults to.
    if (dev.positional.len > 0 and dev.model == null and Value.numberAt(dev.positional, 0) == null) return error.UnresolvedParameter;
    const given = Value.numberAt(dev.positional, 0) orelse try net_cards.numericParameter(dev.kv, value_field) orelse try net_cards.numericParameter(dev.kv, alias);
    var value = if (given) |v| (if (D == devices.resistor) net_cards.resistanceFloor(v) else v) else blk: {
        // Semiconductor resistor (ngspice restemp.c RESupdate_conduct):
        // R = RSH*(L-2*SHORT)/(W-2*NARROW), W defaulting to the model's
        // DEFW. A non-positive result reads as open, as ngspice's
        // unchecked divide leaves it.
        if (comptime D == devices.resistor) {
            if (dev.model) |m| {
                const rsh = net_cards.kvNumber(m.kv, "rsh") orelse 0;
                const l = net_cards.kvNumber(dev.kv, "l") orelse 0;
                const w = net_cards.kvNumber(dev.kv, "w") orelse net_cards.kvNumber(m.kv, "defw") orelse 10e-6;
                if (l * w * rsh > 0) {
                    const narrow = net_cards.kvNumber(m.kv, "narrow") orelse 0;
                    const rshort = net_cards.kvNumber(m.kv, "short") orelse 0;
                    const rr = (l - 2 * rshort) / (w - 2 * narrow) * rsh;
                    break :blk if (std.math.isFinite(rr) and rr > 0) rr else 1e30;
                }
                if (net_cards.kvNumber(m.kv, "r")) |mr| break :blk mr;
            }
            // restemp.c: "resistance too low or not given, set to 1 mOhm".
            break :blk 1e-3;
        }
        break :blk 0;
    };
    // ngspice instance factors, conduct = m/(R*scale), apply to an
    // explicit value too (`R5 6 0 10 scale=1K`, `R4 ... m=2`).
    if (comptime D == devices.resistor) {
        const scale = net_cards.kvNumber(dev.kv, "scale") orelse 1;
        const mult = net_cards.kvNumber(dev.kv, "m") orelse 1;
        value *= scale;
        value /= mult;
        // `ac=` is an AC-only resistance (restemp.c:112-118) with the same
        // instance factors; Circuit.linearizeAc swaps it in.
        if (net_cards.kvNumber(dev.kv, "ac")) |ac_r|
            try self.ac_res.append(self.arena, .{ .name = dev.name, .value = net_cards.resistanceFloor(ac_r) * scale / mult });
    }
    // The model card first (`.model rm R(TC1=2e-3 TNOM=25)`), so the card
    // value and instance pairs win over it, as in ngspice's RESmodel.
    if (dev.model) |m| try net_cards.applyKv(&model, m.kv);
    _ = try net_builder.setParam(D, &model, &instance, value_field, value);
    try net_cards.applyKv(&model, dev.kv);
    try net_cards.applyKv(&instance, dev.kv);
    // `$simparam` defaults after the last write: `tnom` is ngspice's
    // CKTnomTemp for R's tc1/tc2 (restemp.c), a card TNOM still wins.
    net_devices.deriveModel(D, &model, self.b, dev.pins.len);
    const nodes = try net_devices.deviceNodes(self, D, dev);
    const br = self.b.n;
    try self.b.addDevice(D, dev.name, &model, &instance, nodes);
    return br;
}

/// F, H or W card: a device whose 0 V control branch is the next link of
/// the sensed V card's chain (`VCard.tap`). `default_gain` null: no gain parameter.
pub fn addBranchRef(self: *NetBuilder, comptime D: type, dev: Device, comptime default_gain: ?f64) !void {
    if (comptime !@hasDecl(D, "eval")) return error.UnsupportedDevice;
    const ctrl_name = Value.nameAt(dev.positional, 0) orelse return error.MissingControlSource;
    const ctrl = net_branches.vRow(self, ctrl_name) orelse return error.UnknownControlSource;

    var model: D.Model = .{};
    // W card: `W n+ n- Vctrl model`, model at positional[1] after the
    // control source. F/H put a number there, so fall back to positional[0].
    if (Value.nameAt(dev.positional, 1) orelse Value.nameAt(dev.positional, 0)) |name| {
        if (self.nl.findModel(name)) |m| try net_cards.applyKv(&model, m.kv);
    }
    var instance: D.Instance = .{};
    if (comptime default_gain) |dflt|
        _ = try net_builder.setParam(D, &model, &instance, "gain", Value.numberAt(dev.positional, 1) orelse net_cards.kvNumber(dev.kv, "gain") orelse dflt);
    try net_cards.applyKv(&instance, dev.kv);

    // (p, n, cp, cn): the model's `branch (cp, cn) ctrl` at `vsense` = 0
    // is an ammeter in series with the sensed source.
    const left = &self.v.items(.sensors)[ctrl];
    std.debug.assert(left.* != 0); // init counted this card
    left.* -= 1;
    const tap = &self.v.items(.tap)[ctrl];
    const cp = tap.*;
    tap.* = if (left.* == 0) self.v.items(.neg)[ctrl] else try self.b.addNode();
    const nodes = [4]u32{
        if (dev.pins.len > 0) try self.rowOf(dev.pins[0]) else GROUND,
        if (dev.pins.len > 1) try self.rowOf(dev.pins[1]) else GROUND,
        cp,
        tap.*,
    };
    try self.b.addDevice(D, dev.name, &model, &instance, nodes);
    // H also carries its own output branch, and ngspice names it i(h1).
    if (comptime @hasDecl(D, "U") and D == devices.ccvs)
        try net_branches.addBranchProbe(self, dev.name, net_devices.internalRow(D, "flowZ28pZ2cnZ29", self.b.n));
}

/// K card: mutual inductance between two L cards' branches.
pub fn addKinduc(self: *NetBuilder, dev: Device) !void {
    if (comptime !@hasDecl(devices.kinduc, "eval")) return error.UnsupportedDevice;
    const l1_name = Value.nameAt(dev.positional, 0) orelse return error.KinducMissingInductor;
    const l2_name = Value.nameAt(dev.positional, 1) orelse return error.KinducMissingInductor;
    const li1 = net_branches.lRow(self, l1_name) orelse return error.KinducUnknownInductor;
    const li2 = net_branches.lRow(self, l2_name) orelse return error.KinducUnknownInductor;
    const ibr1 = self.l.items(.branch)[li1];
    const ibr2 = self.l.items(.branch)[li2];
    var model: devices.kinduc.Model = .{};
    if (Value.numberAt(dev.positional, 2)) |k| model.k = try castField(@TypeOf(model.k), k);
    if (dev.model) |m| try net_cards.applyKv(&model, m.kv);
    try net_cards.applyKv(&model, dev.kv);
    // The card gives the coupling k; the device stamps M = k*sqrt(L1*L2)
    // (ngspice INDsetup).
    model.k = try castField(@TypeOf(model.k), model.k *
        @sqrt(self.l.items(.value)[li1] * self.l.items(.value)[li2]));
    try self.b.addDevice(devices.kinduc, dev.name, &model, &.{}, [2]u32{ ibr1, ibr2 });
}

// ---- private ----
