//! Transmission-line cards. In: T, O, Y, U (URC), P (CPL), HSPICE W and S
//! cards. Out: their device instances: lossless and lossy lines, LTRA, TXL,
//! URC lumps, coupled lines, and the W/S rational models.
//! Cites: ngspice ltraset.c, txlsetup.c, urcsetup.c, cplsetup.c; HSPICE
//! [SI] Ch.2, Ch.3.
//!
//! Contents:
//! - `pub fn addTxl(self: *NetBuilder, dev: Device) !void`
//! - `pub fn addLossyLine(self: *NetBuilder, dev: Device) !void`
//! - `pub fn addUrc(self: *NetBuilder, dev: Device) !void`
//! - `pub fn addCpl(self: *NetBuilder, dev: Device) !void`
//! - `pub fn addWline(self: *NetBuilder, dev: Device) !void`
//! - `pub fn addSparam(self: *NetBuilder, dev: Device) !void`

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
const net_waves = @import("net_waves.zig");
const net_cards = @import("net_cards.zig");
const NetBuilder = net_builder.NetBuilder;

/// TXL (Y card) on txl.va's Pade/history line. The RC case inp2y
/// expands into 3-pi sections (r/l > 1.6e10) is refused, not approximated;
/// a failed fit refuses the run (txl.va $fatal).
pub fn addTxl(self: *NetBuilder, dev: Device) !void {
    if (dev.pins.len != 4) return error.InvalidTransmissionLinePorts;
    var r: f64 = 0;
    var l: f64 = 0;
    var g: f64 = 0;
    var c: f64 = 0;
    var len: f64 = 0;
    if (dev.model) |m| {
        r = try net_cards.numericParameter(m.kv, "r") orelse 0;
        l = try net_cards.numericParameter(m.kv, "l") orelse 0;
        g = try net_cards.numericParameter(m.kv, "g") orelse 0;
        c = try net_cards.numericParameter(m.kv, "c") orelse 0;
        len = try net_cards.numericParameter(m.kv, "length") orelse 0;
    }
    if ((try net_cards.numericParameter(dev.kv, "length")) orelse (try net_cards.numericParameter(dev.kv, "len"))) |v| len = v;
    // numericParameter already rejected non-finite values.
    if (g < 0 or r <= 0 or l <= 0 or c <= 0 or len <= 0 or r / l > 1.6e10 or
        !std.math.isFinite(r * len) or r * len <= 0)
        return error.UnsupportedTransmissionLineParameters;
    var nm: devices.txl.Model = .{ .r = r, .l = l, .g = g, .c = c, .len = len };
    net_devices.deriveModel(devices.txl, &nm, self.b, dev.pins.len);
    // ngspice's inp2y binds the signal nodes and ignores the references.
    const n1 = try self.rowOf(dev.pins[0]);
    const n2 = try self.rowOf(dev.pins[2]);
    try self.b.addDevice(devices.txl, dev.name, &nm, &.{}, [4]u32{ n1, GROUND, n2, GROUND });
    // ngspice writes duplicate i(Y) names; the oracle keeps the last
    // (far-end) branch, as for CPL.
    try net_branches.addBranchProbe(self, dev.name, self.b.n - 1);
}

/// LTRA (O card), routed as ngspice LTRAsetup does:
/// RLC (r,l,c > 0, g = 0) and RC (r,c > 0, l = g = 0) go to the
/// recursive-convolution device (ltra.va: history, coefficients, chop,
/// step limit); LC (r = g = 0) to one exact Bergeron line (tline.va); a
/// static RG line to lossy_tline.va's exact two-port. Anything else is refused.
pub fn addLossyLine(self: *NetBuilder, dev: Device) !void {
    if (dev.pins.len != 4) return error.InvalidTransmissionLinePorts;
    var model: devices.lossy_tline.Model = .{};
    if (dev.model) |m| {
        try net_cards.applyKv(&model, m.kv);
        // TXL model cards spell the line length `length=`; the
        // lossy_tline field is `len`.
        if (try net_cards.numericParameter(m.kv, "length")) |length| model.len = @floatCast(length);
    }
    try net_cards.applyKv(&model, dev.kv);
    if (try net_cards.numericParameter(dev.kv, "length")) |length| model.len = @floatCast(length);

    const len: f64 = @as(f64, model.len);
    if (len <= 0 or model.r < 0 or model.l < 0 or model.g < 0 or model.c < 0)
        return error.UnsupportedTransmissionLineParameters;
    const r_t: f64 = @as(f64, model.r) * len;
    const l_t: f64 = @as(f64, model.l) * len;
    const c_t: f64 = @as(f64, model.c) * len;
    const g_t: f64 = @as(f64, model.g) * len;
    if (!std.math.isFinite(r_t) or !std.math.isFinite(l_t) or !std.math.isFinite(c_t) or !std.math.isFinite(g_t))
        return error.NonFiniteParameter;

    if ((model.r > 0 and r_t == 0) or (model.l > 0 and l_t == 0) or
        (model.c > 0 and c_t == 0) or (model.g > 0 and g_t == 0))
        return error.UnsupportedTransmissionLineParameters;
    // Classify raw parameters: scaling underflow must not change the case.
    const wave = model.l > 0 and model.c > 0 and model.g == 0;
    const rc = model.r > 0 and model.c > 0 and model.l == 0 and model.g == 0;
    if (wave) {
        const impedance = @sqrt(model.l / model.c);
        const delay = @sqrt(model.l * model.c) * len;
        if (!net_cards.finiteLineCoefficients(.{ impedance, 1.0 / impedance, delay, model.r / model.l }) or impedance <= 0 or delay <= 0)
            return error.UnsupportedTransmissionLineParameters;
    } else if (rc) {
        const cbyr = model.c / model.r;
        const rclsqr = model.r * model.c * len * len;
        if (!net_cards.finiteLineCoefficients(.{ cbyr, rclsqr }) or cbyr <= 0 or rclsqr <= 0)
            return error.UnsupportedTransmissionLineParameters;
    }

    // Only the static RG branch of lossy_tline.va is exact; a dynamic line
    // must not fall into its RC/RLC approximation.
    if (!wave and !rc) {
        if (model.r > 0 and model.g > 0 and model.l == 0 and model.c == 0) {
            const rg = model.r * model.g;
            if (rg == 0) return error.UnsupportedTransmissionLineParameters;
            const gl = len * @sqrt(rg);
            const sinhc = if (gl > 1e-9) std.math.sinh(gl) / gl else 1.0 + gl * gl / 6.0;
            const zs = r_t * sinhc;
            if (gl <= 0 or !net_cards.finiteLineCoefficients(.{ gl, sinhc, std.math.cosh(gl), zs, zs * (1.0 + 1e-12) }))
                return error.UnsupportedTransmissionLineParameters;
            net_devices.deriveModel(devices.lossy_tline, &model, self.b, dev.pins.len);
            return self.b.addDevice(devices.lossy_tline, dev.name, &model, &.{}, try net_devices.deviceNodes(self, devices.lossy_tline, dev));
        }
        return error.UnsupportedTransmissionLineParameters;
    }

    var ports: [4]u32 = undefined;
    for (&ports, dev.pins) |*port, pin| port.* = try self.rowOf(pin);

    if (rc or r_t > 0) {
        var nm: devices.ltra.Model = .{
            .r = model.r,
            .l = model.l,
            .g = model.g,
            .c = model.c,
            .len = model.len,
            .compactrel = model.compactrel,
            .compactabs = model.compactabs,
            .rel = model.rel,
            .steplimit = if (model.nosteplimit != 0) 0 else 1,
            .truncdontcut = model.truncdontcut,
        };
        net_devices.deriveModel(devices.ltra, &nm, self.b, dev.pins.len);
        return self.b.addDevice(devices.ltra, dev.name, &nm, &.{}, ports);
    }

    // Lossless LC: one exact Bergeron ideal line.
    const t_model: devices.tline.Model = .{
        .z0 = @floatCast(@sqrt(l_t / c_t)),
        .td = @floatCast(@sqrt(l_t * c_t)),
    };
    if (!net_cards.finiteLineCoefficients(.{ t_model.z0, t_model.td }) or t_model.z0 <= 0 or t_model.td <= 0)
        return error.UnsupportedTransmissionLineParameters;
    try self.b.addDevice(devices.tline, dev.name, &t_model, &.{}, ports);
}

/// URC (U card): `Uxxx n1 n2 ngnd model [l=len] [n=lumps]`. Expanded, as
/// ngspice URCsetup does, into a ladder of R/C lumps (diodes when ISPERL
/// is given) sized geometrically by K from both ends toward the middle, so the
/// totals telescope to L*RPERL and L*CPERL.
pub fn addUrc(self: *NetBuilder, dev: Device) !void {
    // Model defaults from urcsetup.c; a card without a .model is legal.
    var k: f64 = 1.5;
    var fmax: f64 = 1e9;
    var rperl: f64 = 1000;
    var cperl: f64 = 1e-12;
    var isperl: ?f64 = null;
    var rsperl: f64 = 0;
    if (dev.model) |m| {
        k = net_cards.kvNumber(m.kv, "k") orelse k;
        fmax = net_cards.kvNumber(m.kv, "fmax") orelse fmax;
        rperl = net_cards.kvNumber(m.kv, "rperl") orelse rperl;
        cperl = net_cards.kvNumber(m.kv, "cperl") orelse cperl;
        isperl = net_cards.kvNumber(m.kv, "isperl");
        rsperl = net_cards.kvNumber(m.kv, "rsperl") orelse rsperl;
    }
    // K = 1 makes the geometric sizing below 0/0, and K <= 0 sizes no
    // ladder at all: refused rather than stamped as NaN resistors.
    if (!(k > 0) or k == 1) return error.InvalidParameterValue;
    // ngspice defaults the length to 0 (0-ohm lumps); a card without l=
    // means a unit line here.
    const len = net_cards.kvNumber(dev.kv, "l") orelse 1.0;
    const p = k;
    const r0 = len * rperl;
    const c0 = len * cperl;
    const is0 = len * (isperl orelse 0);

    const lumps: u32 = if (try net_cards.numericParameter(dev.kv, "n")) |nv|
        // The clamp keeps @intFromFloat defined on absurd cards.
        @intFromFloat(std.math.clamp(nv, 1, 1000))
    else blk: {
        // URCsetup: enough lumps that the finest one's pole clears FMAX.
        const wnorm = fmax * r0 * c0 * 2.0 * std.math.pi;
        const est = @log(wnorm * ((p - 1) / p) * ((p - 1) / p)) / @log(p);
        // `!(est > 3)` also catches the NaN a K <= 1 card produces,
        // which @intFromFloat must not see.
        break :blk if (wnorm < 35 or !(est > 3)) 3 else @intFromFloat(@min(est, 1000));
    };
    const lumps_f: f64 = @floatFromInt(lumps);

    // Geometric sizing (urcsetup.c): lump i carries r1*K^(i-1), c1*K^(i-1).
    const r1 = r0 * (p - 1) / (2 * std.math.pow(f64, p, lumps_f) - 2);
    const c1 = c0 * (p - 1) / (std.math.pow(f64, p, lumps_f - 1) * (p + 1) - 2);
    const is1 = is0 * (p - 1) / (std.math.pow(f64, p, lumps_f - 1) * (p + 1) - 2);
    const rd = len * lumps_f * rsperl;

    const pos = if (dev.pins.len > 0) try self.rowOf(dev.pins[0]) else GROUND;
    const neg = if (dev.pins.len > 1) try self.rowOf(dev.pins[1]) else GROUND;
    const gnd = if (dev.pins.len > 2) try self.rowOf(dev.pins[2]) else GROUND;

    // ngspice switches to diodes when ISPERL is merely given
    // (urcsetup.c:106), so ISPERL=0 makes is=0 diodes: gmin plus a
    // voltage-dependent depletion capacitance, not a linear capacitor.
    const use_diodes = isperl != null;
    var prop: f64 = 1; // K^(i-1)
    var lowl = pos; // low-side chain head (walks pos -> middle)
    var hir = neg; // high-side chain head (walks neg -> middle)
    for (1..lumps + 1) |i| {
        const last = i == lumps;
        const hil = try self.b.addNode();
        // The chains meet at the last hi node.
        const lowr = if (last) hil else try self.b.addNode();
        const r: f64 = prop * r1;
        try self.b.addDevice(devices.resistor, dev.name, &.{ .r = @floatCast(r) }, &.{}, [2]u32{ lowl, lowr });
        try self.b.addDevice(devices.resistor, dev.name, &.{ .r = @floatCast(r) }, &.{}, [2]u32{ hil, hir });
        if (use_diodes) {
            const Diode = devices.DeviceId.Type(.diode);
            if (comptime !@hasDecl(Diode, "eval")) return error.UnsupportedDevice;
            // ngspice shares one diode model (is=i1, cjo=c1, rs=rd) and
            // scales each lump by area=prop; diode.va has no area, so the
            // scaling goes into per-lump model values.
            var dm: Diode.Model = .{};
            // diosetup.c:216 raises IS below CKTepsmin (1e-28) to it,
            // so ISPERL=0 still conducts ~6e-12 A per unit area at 1 V.
            dm.is = @floatCast(@max(is1, 1e-28) * prop);
            dm.cjo = @floatCast(c1 * prop);
            dm.rs = @floatCast(rd / prop);
            try self.b.addDevice(Diode, dev.name, &dm, &.{}, [2]u32{ lowr, gnd });
            if (!last) try self.b.addDevice(Diode, dev.name, &dm, &.{}, [2]u32{ hil, gnd });
        } else {
            const cm: devices.capacitor.Model = .{ .c = @floatCast(prop * c1) };
            try self.b.addDevice(devices.capacitor, dev.name, &cm, &.{}, [2]u32{ lowr, gnd });
            if (!last) try self.b.addDevice(devices.capacitor, dev.name, &cm, &.{}, [2]u32{ hil, gnd });
        }
        prop *= p;
        lowl = lowr;
        hir = hil;
    }
}

/// CPL (P card) on coupled_ltra.va's modal-fit, accepted-step convolution
/// line, for 2 to 4 conductors. Anything else is refused, never
/// approximated; a failed modal fit refuses the run ($fatal).
pub fn addCpl(self: *NetBuilder, dev: Device) !void {
    if (dev.pins.len < 6 or dev.pins.len % 2 != 0) return error.UnsupportedCoupledLineDimension;
    const n_lines = (dev.pins.len - 2) / 2;
    if (n_lines > 4) return error.UnsupportedCoupledLineDimension;
    const card = dev.model orelse return error.UnsupportedTransmissionLineParameters;
    var rr: [10]f64 = undefined;
    var ll: [10]f64 = undefined;
    var cc: [10]f64 = undefined;
    var gg: [10]f64 = @splat(0);
    const nr = try net_cards.cplVector(card.kv, "r", &rr);
    const nl = try net_cards.cplVector(card.kv, "l", &ll);
    const nc = try net_cards.cplVector(card.kv, "c", &cc);
    const ng = try net_cards.cplVector(card.kv, "g", &gg);
    const length = (try net_cards.numericParameter(dev.kv, "length")) orelse (try net_cards.numericParameter(card.kv, "length")) orelse 0;
    const tri = n_lines * (n_lines + 1) / 2;
    if (nr != tri or nl != tri or nc != tri or (ng != 0 and ng != tri) or !std.math.isFinite(length) or length <= 0)
        return error.UnsupportedTransmissionLineParameters;
    inline for (.{ devices.coupled_ltra, devices.coupled_ltra3, devices.coupled_ltra4 }) |D| {
        const N = (D.num_ports - 2) / 2;
        if (n_lines == N) {
            var model: D.Model = .{ .length = length };
            inline for (0..10) |k| if (k < tri) {
                @field(model, net_waves.pwlSlot("r", k)) = rr[k];
                @field(model, net_waves.pwlSlot("l", k)) = ll[k];
                @field(model, net_waves.pwlSlot("c", k)) = cc[k];
                @field(model, net_waves.pwlSlot("g", k)) = gg[k];
            };
            net_devices.deriveModel(D, &model, self.b, dev.pins.len);
            // ngspice cplsetup binds conductor nodes and ignores references.
            var nodes: [2 * N + 2]u32 = @splat(GROUND);
            for (0..N) |i| nodes[i] = try self.rowOf(dev.pins[i]);
            for (0..N) |i| nodes[N + 1 + i] = try self.rowOf(dev.pins[N + 1 + i]);
            try self.b.addDevice(D, dev.name, &model, &.{}, nodes);
            try net_branches.addBranchProbe(self, dev.name, self.b.n - 1);
            return;
        }
    }
    unreachable;
}

/// HSPICE W element [SI Ch.3]: `Wx i1..iN iR o1..oN oR N= L=` with
/// `RLGCMODEL=` (a `.model m W MODELTYPE=RLGC`) or `RLGCFILE=`, on
/// models/wline_N.va: the modal rational fit of wfit.zig run by the
/// method of characteristics. Refused: other model forms and N > 4.
pub fn addWline(self: *NetBuilder, dev: Device) !void {
    // L, C, R, G, Rs, Gd lower triangles, in RLGC-file order.
    var tri: [6][10]f64 = @splat(@splat(0));
    var n: usize = 0;
    if (net_cards.kvName(dev.kv, "rlgcmodel")) |name| {
        const card = self.nl.findModel(name) orelse return net_cards.refuse(dev, 'W', "no .model named by RLGCMODEL");
        if (!std.mem.eql(u8, card.kind, "w")) return net_cards.refuse(dev, 'W', "RLGCMODEL names a model that is not type W");
        if (net_cards.kvName(card.kv, "modeltype")) |t| if (!std.mem.eql(u8, t, "rlgc")) return net_cards.refuse(dev, 'W', "only MODELTYPE=RLGC is supported");
        // Clamped first: @intFromFloat of N=1e30 is illegal behaviour.
        n = @intFromFloat(std.math.clamp((try net_cards.numericParameter(card.kv, "n")) orelse 0, 0, wfit.max_n + 1));
        if (n == 0 or n > wfit.max_n) return net_cards.refuse(dev, 'W', "N must be 1 to 4");
        const n_tri = n * (n + 1) / 2;
        const keys = [_][2][]const u8{ .{ "lo", "l" }, .{ "co", "c" }, .{ "ro", "r" }, .{ "go", "g" }, .{ "rs", "" }, .{ "gd", "" } };
        for (keys, 0..) |k, which| {
            var got = try net_cards.cplVector(card.kv, k[0], tri[which][0..n_tri]);
            if (got == 0 and k[1].len != 0) got = try net_cards.cplVector(card.kv, k[1], tri[which][0..n_tri]);
            if (got != 0 and got != n_tri) return net_cards.refuse(dev, 'W', "a matrix does not hold N(N+1)/2 lower-triangle entries");
            if (got == 0 and which < 2) return net_cards.refuse(dev, 'W', "Lo and Co are required");
        }
    } else if (net_cards.kvName(dev.kv, "rlgcfile")) |path| {
        const bytes = self.nl.dataFile(path) orelse return net_cards.refuse(dev, 'W', "RLGCFILE was not read");
        const file = wfit.readRlgcFile(bytes) catch |err| return net_cards.refuse(dev, 'W', switch (err) {
            error.TooManyConductors => "N must be 1 to 4",
            error.InvalidRlgcFile => "malformed RLGC file",
        });
        n = file.n;
        for (file.tri[0..file.mats], tri[0..file.mats]) |src, *dst| @memcpy(dst[0..src.len], &src);
    } else return net_cards.refuse(dev, 'W', "only RLGCMODEL= and RLGCFILE= are supported (TABLEMODEL, UMODEL, FSMODEL and SMODEL are not)");
    if (try net_cards.numericParameter(dev.kv, "n")) |nn| if (nn != @as(f64, @floatFromInt(n))) return net_cards.refuse(dev, 'W', "N differs from the model's");
    if (dev.pins.len != 2 * n + 2) return net_cards.refuse(dev, 'W', "the node count is not 2N+2");
    const length = (try net_cards.numericParameter(dev.kv, "l")) orelse return net_cards.refuse(dev, 'W', "L= (length) is required");
    if (!(length > 0)) return net_cards.refuse(dev, 'W', "L must be positive");
    var p: wfit.Rlgc = .{
        .n = n,
        .length = length,
        .fgd = (try net_cards.numericParameter(dev.kv, "fgd")) orelse 0,
        .rs_imag = if (net_cards.kvName(dev.kv, "includersimag")) |v| !std.mem.eql(u8, v, "no") else (net_cards.kvNumber(dev.kv, "includersimag") orelse 1) != 0,
    };
    for ([_]*[wfit.max_n][wfit.max_n]f64{ &p.l, &p.c, &p.r, &p.g, &p.rs, &p.gd }, tri) |dst, packed_tri| {
        var idx: usize = 0;
        for (0..n) |i| for (0..i + 1) |j| {
            dst[i][j] = packed_tri[idx];
            dst[j][i] = packed_tri[idx];
            idx += 1;
        };
    }
    var nodes: [2 * wfit.max_n + 2]u32 = undefined;
    for (dev.pins, nodes[0..dev.pins.len]) |pin, *row| row.* = try self.rowOf(pin);
    inline for (.{ devices.wline_1, devices.wline_2, devices.wline_3, devices.wline_4 }, 1..) |D, N| if (n == N) {
        if (comptime !@hasDecl(D, "eval")) return error.UnsupportedDevice;
        const k = comptime net_waves.slotCount(D.Model, "ycp") / (3 * N);
        const line = wfit.fitLine(self.arena, p, k) catch |err| return net_cards.refuse(dev, 'W', @errorName(err));
        var model: D.Model = .{};
        var tv: [N * N]f64 = undefined;
        var ti: [N * N]f64 = undefined;
        var rdc: [N * N]f64 = undefined;
        var gdc: [N * N]f64 = undefined;
        for (0..N) |i| for (0..N) |j| {
            tv[i * N + j] = line.tv[i][j];
            ti[i * N + j] = line.ti[i][j];
            rdc[i * N + j] = p.r[i][j] * length;
            gdc[i * N + j] = p.g[i][j] * length / 2;
        };
        var z0: [N]f64 = undefined;
        var td: [N]f64 = undefined;
        var ycd: [N]f64 = undefined;
        var hd: [N]f64 = undefined;
        var ycn: [2 * N * k]f64 = @splat(0);
        var hn: [2 * N * k]f64 = @splat(0);
        var ycp: [3 * N * k]f64 = undefined;
        var hp: [3 * N * k]f64 = undefined;
        for (0..N * k) |s| {
            ycp[3 * s ..][0..3].* = .{ 1, 0, 0 };
            hp[3 * s ..][0..3].* = .{ 1, 0, 0 };
        }
        for (line.modes[0..N], 0..) |md, mi| {
            z0[mi] = md.z0;
            td[mi] = md.tau;
            ycd[mi] = md.yc.d;
            hd[mi] = md.h.d;
            for (md.yc.num, md.yc.den, 0..) |nm, dn, s| {
                ycn[2 * (mi * k + s) ..][0..2].* = nm;
                ycp[3 * (mi * k + s) ..][0..3].* = dn;
            }
            for (md.h.num, md.h.den, 0..) |nm, dn, s| {
                hn[2 * (mi * k + s) ..][0..2].* = nm;
                hp[3 * (mi * k + s) ..][0..3].* = dn;
            }
            const err = @max(md.yc.rel_err, md.h.rel_err);
            if (err > 1e-2 and !@import("builtin").is_test)
                std.log.warn("W element '{s}': mode {d} rational fit error {e:.2}", .{ dev.name, mi, err });
        }
        inline for (.{ "tv", "ti", "rdc", "gdc", "z0", "td", "ycd", "hd", "ycn", "hn", "ycp", "hp" }, .{ &tv, &ti, &rdc, &gdc, &z0, &td, &ycd, &hd, &ycn, &hn, &ycp, &hp }) |base, values|
            net_waves.setArray(D.Model, &model, base, values);
        return self.b.addDevice(D, dev.name, &model, &.{}, nodes[0 .. 2 * N + 2].*);
    };
    unreachable;
}

/// HSPICE S element [SI Ch.2]: `Sx nd1..ndN [ndR] MNAME=m`, `.model m S
/// TSTONEFILE=f.sNp`, on models/sparam_N.va: the file's admittance
/// vector fitted with common poles and made passive (sparam.zig), every
/// entry a sum of second-order sections. Refused: FQMODEL/CITIFILE/
/// RFMFILE data, mixed mode, N > 4, and a fit that cannot be made
/// passive.
pub fn addSparam(self: *NetBuilder, dev: Device) !void {
    const model_name = net_cards.kvName(dev.kv, "mname") orelse return net_cards.refuse(dev, 'S', "MNAME= is required (FQMODEL is not supported)");
    const card = self.nl.findModel(model_name) orelse return net_cards.refuse(dev, 'S', "no .model named by MNAME");
    if (!std.mem.eql(u8, card.kind, "s")) return net_cards.refuse(dev, 'S', "MNAME names a model that is not type S");
    for ([_][]const u8{ "fqmodel", "citifile", "rfmfile" }) |k| if (net_cards.kvName(card.kv, k) != null or net_cards.kvName(dev.kv, k) != null)
        return net_cards.refuse(dev, 'S', "only TSTONEFILE data is supported");
    if ((net_cards.kvNumber(card.kv, "mixedmode") orelse 0) != 0 or (net_cards.kvNumber(dev.kv, "mixedmode") orelse 0) != 0)
        return net_cards.refuse(dev, 'S', "mixed-mode data is not supported");
    const path = net_cards.kvName(card.kv, "tstonefile") orelse return net_cards.refuse(dev, 'S', "TSTONEFILE= is required");
    const ext = std.fs.path.extension(path);
    const from_ext: ?usize = if (ext.len >= 4 and std.ascii.toLower(ext[1]) == 's' and std.ascii.toLower(ext[ext.len - 1]) == 'p')
        std.fmt.parseInt(usize, ext[2 .. ext.len - 1], 10) catch null
    else
        null;
    const p: usize = if (try net_cards.numericParameter(card.kv, "n")) |n| @intFromFloat(std.math.clamp(n, 0, 5)) else from_ext orelse return net_cards.refuse(dev, 'S', "the port count is neither N= nor a .sNp extension");
    if (p == 0 or p > 4) return net_cards.refuse(dev, 'S', "1 to 4 ports are supported");
    if (dev.pins.len != p and dev.pins.len != p + 1) return net_cards.refuse(dev, 'S', "the node count is not N or N+1");
    const bytes = self.nl.dataFile(path) orelse return net_cards.refuse(dev, 'S', "TSTONEFILE was not read");
    const net = sparam.parseTouchstone(self.arena, bytes, p) catch |err| return net_cards.refuse(dev, 'S', @errorName(err));
    var nodes: [5]u32 = @splat(GROUND);
    for (dev.pins, nodes[0..dev.pins.len]) |pin, *row| row.* = try self.rowOf(pin);
    inline for (.{ devices.sparam_1, devices.sparam_2, devices.sparam_3, devices.sparam_4 }, 1..) |D, N| if (p == N) {
        if (comptime !@hasDecl(D, "eval")) return error.UnsupportedDevice;
        const k = comptime net_waves.slotCount(D.Model, "den") / 3;
        var fit = try sparam.fit(self.arena, net, .{ .max_order = 2 * k });
        if (!try sparam.enforcePassivity(self.arena, net, &fit)) return net_cards.refuse(dev, 'S', "the rational fit could not be made passive");
        if (fit.rel_err > 1e-3 and !@import("builtin").is_test)
            std.log.warn("S element '{s}': rational fit error {e:.2} of max |Y|", .{ dev.name, fit.rel_err });
        const sec = try sparam.sections(self.arena, fit);
        var model: D.Model = .{};
        var d: [N * N]f64 = undefined;
        @memcpy(&d, fit.d);
        var den: [3 * k]f64 = undefined;
        var num: [2 * N * N * k]f64 = @splat(0);
        for (0..k) |s| den[3 * s ..][0..3].* = if (s < sec.den.len) sec.den[s] else .{ 1, 0, 0 };
        for (0..N * N) |e| for (sec.num[e * sec.den.len ..][0..sec.den.len], 0..) |nm, s| {
            num[2 * (e * k + s) ..][0..2].* = nm;
        };
        inline for (.{ "d", "den", "num" }, .{ &d, &den, &num }) |base, values| net_waves.setArray(D.Model, &model, base, values);
        return self.b.addDevice(D, dev.name, &model, &.{}, nodes[0 .. N + 1].*);
    };
    unreachable;
}

// ---- private ----
