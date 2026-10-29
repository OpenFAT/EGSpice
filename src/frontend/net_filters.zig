//! E/G LAPLACE, POLE and DELAY cards. In: the card's coefficients. Out:
//! the rational or delayed controlled source instance.
//! Cites: HSPICE [SA] E/G element (LAPLACE, POLE, DELAY).
//!
//! Contents:
//! - `pub fn addLaplace(self: *NetBuilder, dev: Device, letter: u8) !void`
//! - `pub fn addPole(self: *NetBuilder, dev: Device, letter: u8) !void`
//! - `pub fn rootProduct(r: []const f64) error{UnpairedPoleRoot}!f64`
//! - `pub fn addDelay(self: *NetBuilder, dev: Device, letter: u8) !void`

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
const net_tape = @import("net_tape.zig");
const net_waves = @import("net_waves.zig");
const net_cards = @import("net_cards.zig");
const NetBuilder = net_builder.NetBuilder;

/// E/G LAPLACE card: H(s) = Σ k_i s^i / Σ d_i s^i, ascending coefficients
/// with `laplace=` of them in the numerator [SA E-element Laplace
/// Transform], on models/vcvs_laplace.va or vccs_laplace.va (`laplace_nd`).
/// Improper H, a zero denominator and orders past the arrays are refused.
pub fn addLaplace(self: *NetBuilder, dev: Device, letter: u8) !void {
    for (dev.kv) |item| {
        const known = std.StaticStringMap(void).initComptime(.{ .{"laplace"}, .{"scale"}, .{"m"} });
        if (!known.has(item.key)) return error.UnsupportedLaplaceParameter;
    }
    const nk_f = net_cards.kvNumber(dev.kv, "laplace").?;
    if (!(nk_f >= 1) or nk_f != @round(nk_f) or nk_f >= @as(f64, @floatFromInt(dev.positional.len))) return error.UnsupportedLaplaceOrder;
    const nk: usize = @intFromFloat(nk_f);
    var coef: [2 * 32]f64 = undefined;
    if (dev.positional.len > coef.len) return error.UnsupportedLaplaceOrder;
    for (dev.positional, coef[0..dev.positional.len]) |v, *c| c.* = v.number() orelse return error.UnresolvedParameter;
    var num = coef[0..nk];
    var den = coef[nk..dev.positional.len];
    while (den.len > 0 and den[den.len - 1] == 0) den.len -= 1;
    while (num.len > 0 and num[num.len - 1] == 0) num.len -= 1;
    inline for (.{ devices.vcvs_laplace, devices.vccs_laplace }, "eg") |D, l| if (letter == l) {
        if (comptime !@hasDecl(D, "eval")) return error.UnsupportedDevice;
        const slots = comptime net_waves.slotCount(D.Model, "den");
        if (den.len == 0 or den.len > slots or num.len > den.len) return error.UnsupportedLaplaceOrder;
        var model: D.Model = .{};
        var instance: D.Instance = .{};
        _ = try net_builder.setParam(D, &model, &instance, "gain", net_tape.cardScale(dev, letter));
        inline for (0..slots) |k| {
            @field(model, net_waves.pwlSlot("num", k)) = if (k < num.len) num[k] else 0;
            @field(model, net_waves.pwlSlot("den", k)) = if (k < den.len) den[k] else 0;
        }
        try self.b.addDevice(D, dev.name, &model, &instance, try net_tape.controlledNodes(self, dev));
        if (l == 'e') try net_branches.addBranchProbe(self, dev.name, net_devices.internalRow(D, "flowZ28pZ2cnZ29", self.b.n));
    };
}

/// E/G POLE card: H(s) = a (s - z1)...(s - zn) / (b (s - p1)...(s - pm)),
/// each root written `alpha, f` for s = -alpha + j 2 pi f [SA E-element
/// Pole-Zero Function], on models/vcvs_pole.va or vccs_pole.va
/// (`laplace_zp`). A complex root without its conjugate on the same side
/// is refused (UnpairedPoleRoot), as are more zeros than poles and more
/// than 8 roots a side.
pub fn addPole(self: *NetBuilder, dev: Device, letter: u8) !void {
    for (dev.kv) |item| {
        const known = std.StaticStringMap(void).initComptime(.{ .{"pole"}, .{"scale"}, .{"m"} });
        if (!known.has(item.key)) return error.UnsupportedLaplaceParameter;
    }
    // The parser checked both sides: a constant, then (alpha, f) pairs.
    const nk: usize = @intFromFloat(net_cards.kvNumber(dev.kv, "pole").?);
    var v: [2 * (1 + 2 * 8)]f64 = undefined;
    if (dev.positional.len > v.len) return error.UnsupportedLaplaceOrder;
    for (dev.positional, v[0..dev.positional.len]) |p, *x| x.* = p.number() orelse return error.UnresolvedParameter;
    const zeros = v[1..nk];
    const poles = v[nk + 1 .. dev.positional.len];
    if (zeros.len > poles.len or poles.len > 2 * 8) return error.UnsupportedLaplaceOrder;
    // The roots as laplace_zp takes them, s = -alpha + j 2 pi f, padded with
    // inf, whose (1 - s/inf) term is exactly 1.
    var zr: [2 * 8]f64 = @splat(0);
    var pr: [2 * 8]f64 = @splat(0);
    inline for (.{ zeros, poles }, .{ &zr, &pr }) |side, r| {
        for (0..8) |k| r[2 * k] = std.math.inf(f64);
        for (0..side.len / 2) |k| r[2 * k ..][0..2].* = .{ -side[2 * k], 2 * std.math.pi * side[2 * k + 1] };
    }
    // laplace_zp is prod(1 - s/r), and s - r = -r (1 - s/r) for r != 0.
    const gain = net_tape.cardScale(dev, letter) * v[0] / v[nk] * try rootProduct(zr[0..zeros.len]) / try rootProduct(pr[0..poles.len]);
    inline for (.{ devices.vcvs_pole, devices.vccs_pole }, "eg") |D, l| if (letter == l) {
        if (comptime !@hasDecl(D, "eval")) return error.UnsupportedDevice;
        comptime std.debug.assert(net_waves.slotCount(D.Model, "poles") == 2 * 8);
        var model: D.Model = .{};
        var instance: D.Instance = .{};
        _ = try net_builder.setParam(D, &model, &instance, "gain", gain);
        inline for (0..2 * 8) |k| {
            @field(model, net_waves.pwlSlot("zeros", k)) = zr[k];
            @field(model, net_waves.pwlSlot("poles", k)) = pr[k];
        }
        try self.b.addDevice(D, dev.name, &model, &instance, try net_tape.controlledNodes(self, dev));
        if (l == 'e') try net_branches.addBranchProbe(self, dev.name, net_devices.internalRow(D, "flowZ28pZ2cnZ29", self.b.n));
    };
}

/// prod(-r) over the nonzero roots `r` of one side, (re, im) pairs. Pairs
/// complex roots as VerA's zRootSecs does (the first unused root within 1e-9
/// relative of the conjugate), so a root it would turn into NaN is refused.
pub fn rootProduct(r: []const f64) error{UnpairedPoleRoot}!f64 {
    var used: [8]bool = @splat(false);
    var prod: f64 = 1;
    for (0..r.len / 2) |i| {
        if (used[i]) continue;
        used[i] = true;
        const a = r[2 * i];
        const b = r[2 * i + 1];
        if (b == 0) {
            if (a != 0) prod *= -a;
            continue;
        }
        const tol = 1e-9 * @sqrt(a * a + b * b);
        const j = for (0..r.len / 2) |c| {
            if (!used[c] and @abs(r[2 * c] - a) <= tol and @abs(r[2 * c + 1] + b) <= tol) break c;
        } else return error.UnpairedPoleRoot;
        used[j] = true;
        prod *= a * a + b * b;
    }
    return prod;
}

/// E/G DELAY card: the control voltage delayed by `td=`, times SCALE
/// [SA E-element Delay Element], on models/vcvs_delay.va or vccs_delay.va.
pub fn addDelay(self: *NetBuilder, dev: Device, letter: u8) !void {
    const td = net_cards.kvNumber(dev.kv, "td") orelse return error.UnresolvedParameter;
    if (!(td > 0)) return error.InvalidDelay;
    inline for (.{ devices.vcvs_delay, devices.vccs_delay }, "eg") |D, l| if (letter == l) {
        if (comptime !@hasDecl(D, "eval")) return error.UnsupportedDevice;
        var model: D.Model = .{};
        var instance: D.Instance = .{};
        _ = try net_builder.setParam(D, &model, &instance, "gain", net_tape.cardScale(dev, letter));
        _ = try net_builder.setParam(D, &model, &instance, "td", td);
        try self.b.addDevice(D, dev.name, &model, &instance, try net_tape.controlledNodes(self, dev));
        if (l == 'e') try net_branches.addBranchProbe(self, dev.name, net_devices.internalRow(D, "flowZ28pZ2cnZ29", self.b.n));
    };
}

// ---- private ----
