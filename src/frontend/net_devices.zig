//! Built-in device cards. In: a card with a letter and maybe a `.model`.
//! Out: its `DeviceId` (letter, LEVEL or module name), its Model bound once
//! per card and copied per instance, `derive`d, and its instance stored
//! with memoised collapse maps.
//! Cites: ngspice inpdomod.c; Verilog-A LRM §9.18 ($simparam), derive.
//!
//! Contents:
//! - `pub const simparam_fields`
//! - `pub fn resolveDeviceId(letter: u8, dev: Device) !devices.DeviceId`
//! - `pub fn named(dev: Device, err: anyerror) anyerror`
//! - `pub fn inferDeviceFromModel(dev: Device) !?devices.DeviceId`
//! - `pub fn simparamValues(b: *const builder.Builder) [simparam_fields.len]f64`
//! - `pub fn deriveModel(comptime D: type, model: *D.Model, b: *const builder.Builder, pins: usize) void`
//! - `pub fn addSingleDevice(self: *NetBuilder, comptime D: type, dev: Device) !void`
//! - `pub fn deviceNodes(nb: *NetBuilder, comptime D: type, dev: Device) ![D.num_ports]u32`
//! - `pub fn internalRow(comptime D: type, comptime tag: []const u8, end: u32) u32`
//! - `pub fn ownerOf(comptime T: type) struct { []const u8, bool }`

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
const net_cards = @import("net_cards.zig");
const builder = @import("builder.zig");
const NetBuilder = net_builder.NetBuilder;
const Builder = builder.Builder;

/// VerA's reserved Model fields for §9.15 `$simparam`, parallel to
/// `simparamValues`: tnom (degC), reltol, abstol (A), vntol (V), gmin (S).
/// `source_scale__` keeps its default 1 until source stepping. See VerA's
/// `Lower.simparamHostField`.
pub const simparam_fields = [_][]const u8{ "nom_temp__", "reltol__", "abstol__", "vntol__", "gmin__" };

/// The built-in model kinds of each semiconductor letter. A card whose model
/// is another letter's kind is refused, as ngspice's "incorrect model type"
/// (#75); a kind outside these (an HDL module) is not checked.
const letter_kinds = std.StaticStringMap(u8).initComptime(.{
    .{ "nmos", 'm' }, .{ "pmos", 'm' },  .{ "vdmos", 'm' },
    .{ "npn", 'q' },  .{ "pnp", 'q' },   .{ "d", 'd' },
    .{ "njf", 'j' },  .{ "pjf", 'j' },   .{ "nmf", 'z' },
    .{ "pmf", 'z' },  .{ "nhfet", 'z' }, .{ "phfet", 'z' },
});

/// The built-in device a card letter and its model's LEVEL select.
pub fn resolveDeviceId(letter: u8, dev: Device) !devices.DeviceId {
    const id = try deviceIdOf(letter, dev);
    // PSP 103 ships as two devices, as upstream does (psp103.va,
    // psp103_nqs.va): the NQS build carries 48 unknowns per instance where
    // the QS one carries 12, whatever SWNQS is, so only a card that asks
    // for NQS pays for it.
    if (id == .psp103) {
        const swnqs = net_cards.kvNumber(dev.kv, "swnqs") orelse
            if (dev.model) |m| net_cards.kvNumber(m.kv, "swnqs") orelse 0 else 0;
        if (swnqs != 0) return .psp103_nqs;
    }
    return id;
}

/// `err`, logged with the card and its model when it is `UnsupportedDevice`,
/// so the refusal names what to fix (#75).
pub fn named(dev: Device, err: anyerror) anyerror {
    if (err == error.UnsupportedDevice and !@import("builtin").is_test) {
        const m = dev.model orelse return err;
        std.log.err("{s}: model '{s}' ({s}) is not a device this card can use", .{ dev.name, m.name, m.kind });
    }
    return err;
}

/// The device a card with no built-in letter gets from its `.model` kind
/// (`N1 d g s b nmos_model`); null when the kind names none.
pub fn inferDeviceFromModel(dev: Device) !?devices.DeviceId {
    const m = dev.model orelse return null;
    const l = try modelLevel(dev);
    if (std.ascii.eqlIgnoreCase(m.kind, "vdmos"))
        return .vdmos;
    // The Verilog-A module name an OSDI deck loads the model under.
    var lower: [32]u8 = undefined;
    if (m.kind.len <= lower.len) {
        if (devices.module_map.get(std.ascii.lowerString(&lower, m.kind))) |id| return id;
    }
    if (eqlAny(m.kind, &.{ "nmos", "pmos" }))
        return devices.mosfetDeviceId(l) catch null;
    if (eqlAny(m.kind, &.{ "npn", "pnp" }))
        return devices.bjtDeviceId(l) catch null;
    if (std.ascii.eqlIgnoreCase(m.kind, "d"))
        return devices.diodeDeviceId(l) catch null;
    if (eqlAny(m.kind, &.{ "njf", "pjf" }))
        return devices.jfetDeviceId(l) catch null;
    if (eqlAny(m.kind, &.{ "nmf", "pmf", "nhfet", "phfet" }))
        return devices.mesDeviceId(l) catch null;
    return null;
}

pub fn simparamValues(b: *const builder.Builder) [simparam_fields.len]f64 {
    return .{ b.nom_temp_c, b.tol.reltol, b.tol.abstol, b.tol.vntol, b.tol.gmin };
}

/// Runs §6.3.4/§3.4.5 `derive` through the device's own object: calling
/// `D.derive` directly would compile the generated body (thousands of lines
/// for bsim4) into the executable a second time.
///
/// `.options tnom` is written immediately before `derive`: VerA lowers
/// `parameter real tnom = $simparam("tnom")` to `if (!model.tnom__given)
/// model.tnom = model.nom_temp__`, ngspice's `if (!BSIM4tnomGiven)
/// BSIM4tnom = ckt->CKTnomTemp` (b4set.c:1950). A card TNOM/TREF set
/// `__given` in `applyKv`, so it still wins.
///
/// A card with `pins` nodes leaves the module's later ports unconnected:
/// §9.19 `$port_connected` reads `Model.port_connected__` (bit p = port p,
/// all ones by default), written here because `derive` may read it. A
/// 4-terminal card on HiSIM_HV's 6-port module clears `sub` and `temp`, as
/// its 4-node path expects.
pub fn deriveModel(comptime D: type, model: *D.Model, b: *const builder.Builder, pins: usize) void {
    inline for (simparam_fields, simparamValues(b)) |f, v| {
        if (comptime @hasField(D.Model, f)) @field(model, f) = v;
    }
    if (comptime @hasField(D.Model, "port_connected__")) {
        if (pins < 64) model.port_connected__ &= (@as(u64, 1) << @intCast(pins)) - 1;
    }
    if (comptime device.modelName(D)) |name| {
        if (device.vtable(name).derive) |f| f(@ptrCast(model));
    } else if (comptime @hasDecl(D, "derive")) D.derive(model);
}

/// Binds one card onto built-in `D`: model card, polarity, card pairs, derive.
pub fn addSingleDevice(self: *NetBuilder, comptime D: type, dev: Device) !void {
    const b = self.b;
    if (comptime !@hasDecl(D, "eval")) return error.UnsupportedDevice;
    var model: D.Model = .{};
    var instance: D.Instance = .{};
    // VBIC: a card without the fifth (thermal) node has it tied to ground in
    // ngspice (inp2q.c:85-87), so dT = 0 whatever RTH says. Our `dt` is an
    // internal node, so self-heating is off unless the card names it; an
    // explicit SW_ET on the card still wins through applyKv below.
    if (comptime @hasField(D.Model, "sw_et")) {
        if (dev.pins.len < 5) model.sw_et = 0;
    }
    const key = try memoKey(self, D, dev);
    if (key) |k| if (self.memo.get(k)) |hit| return b.addDeviceRow(D, dev.name, hit.row, @ptrCast(@alignCast(hit.instance)), hit.col, try deviceNodes(self, D, dev));
    if (dev.model) |m| model = try boundCard(self, D, model, m, dev.model_row);
    _ = try net_builder.setParam(D, &model, &instance, "gain", Value.numberAt(dev.positional, 0) orelse 0);
    // INP2Q/D/J/Z: a bare number after the model name is the area; area=
    // below still wins.
    switch (dev.kind) {
        'q', 'd', 'j', 'z' => if (Value.numberAt(dev.positional, 1)) |a| {
            _ = try net_builder.setParam(D, &model, &instance, "area", a);
        },
        else => {},
    }
    // Card pairs go to both structs: VerA puts every Verilog-A `parameter`
    // (W and L included) on Model. `model` is a per-device copy, so this
    // overrides the model card for this instance only.
    // LRM §6.3.6: `m=` on the instance card is the multiplier. A model
    // without an `m` of its own (bsim4va, hicum, hisim, bsimsoi), or whose
    // `m` is a shape (diode grading, hfet/mesa knee), takes it as the
    // instance's $mfactor (#48); the shape keeps the model card's value.
    const shape_m = comptime D == devices.DeviceId.Type(.diode) or D == devices.DeviceId.Type(.hfet1) or
        D == devices.DeviceId.Type(.hfet2) or D == devices.DeviceId.Type(.mesa);
    const to_mfactor = comptime (!@hasField(D.Model, "m") or shape_m) and @hasField(D.Instance, "mfactor");
    var kv = dev.kv;
    if (to_mfactor) if (net_cards.kvNumber(dev.kv, "m")) |mv| {
        instance.mfactor = mv;
        const rest = try self.arena.alloc(Kv, dev.kv.len);
        var n: usize = 0;
        for (dev.kv) |item| if (!std.mem.eql(u8, item.key, "m")) {
            rest[n] = item;
            n += 1;
        };
        kv = rest[0..n];
    };
    try net_cards.applyKv(&model, kv);
    try net_cards.applyKv(&instance, kv);
    try refuseUnknownKeys(self.arena, D, dev.name, kv);
    // §6.3.4/§3.4.5: derived parameters after the last write; explicit card
    // values win through `__given`.
    deriveModel(D, &model, b, dev.pins.len);
    try b.addDevice(D, dev.name, &model, &instance, try deviceNodes(self, D, dev));
    const k = key orelse return;
    const t = comptime Library.builtin(device.modelName(D).?);
    const vt = b.lib.vtable(t);
    const last = vt.proto_last_row orelse return;
    const n_u = comptime @typeInfo(D.U).@"enum".field_names.len;
    const col = try self.arena.alloc(i32, n_u);
    @memset(col, -1);
    if (comptime n_u > D.num_ports) if (vt.collapse) |collapse| collapse(@ptrCast(&model), @ptrCast(&instance), col.ptr);
    const kept = try self.arena.create(D.Instance);
    kept.* = instance;
    try self.memo.put(self.arena, try self.arena.dupe(u8, k), .{ .row = last((try b.protoOf(t)).ctx), .instance = kept, .col = col });
}

/// Rows of `D`'s ports from the card's pins; missing pins are ground.
pub fn deviceNodes(nb: *NetBuilder, comptime D: type, dev: Device) ![D.num_ports]u32 {
    var out: [D.num_ports]u32 = undefined;
    inline for (0..D.num_ports) |idx| {
        out[idx] = if (idx < dev.pins.len) try nb.rowOf(dev.pins[idx]) else GROUND;
    }
    return out;
}

/// Row `Builder.addDevice` gave `D`'s internal unknown `tag`, where `end`
/// is `b.n` sampled right after the call. Internal rows follow `D.U` order
/// and end at `end`, so the precondition is that no internal after `tag`
/// collapsed onto another node. Counting from the end, not from `b.n`
/// before the call, keeps it right when the card's own pins or probed nets
/// take fresh rows inside the call, and when an earlier internal collapses.
///
/// VerA names a Verilog-A `branch (a, b)` as the U member `flowZ28aZ2cbZ29`
/// (`(` = Z28, `,` = Z2c, `)` = Z29). Naming the member rather than writing
/// `first + 1` makes a wrong branch a compile error instead of a mislabelled
/// current column: ccvs declares two branches, the sense one first.
pub fn internalRow(comptime D: type, comptime tag: []const u8, end: u32) u32 {
    const off = comptime blk: {
        const names = @typeInfo(D.U).@"enum".field_names;
        for (names, 0..) |name, i| {
            if (std.mem.eql(u8, name, tag)) {
                if (i < D.num_ports) @compileError(@typeName(D) ++ ": `" ++ tag ++ "` is a port, not an internal unknown");
                break :blk names.len - i;
            }
        }
        @compileError(@typeName(D) ++ ": no unknown named `" ++ tag ++ "`");
    };
    return end - @as(u32, @intCast(off));
}

/// The catalog device whose Model (true) or Instance (false) `T` is.
pub fn ownerOf(comptime T: type) struct { []const u8, bool } {
    for (device.catalog) |e| {
        if (@hasDecl(e.type, "Model") and e.type.Model == T) return .{ e.name, true };
        if (@hasDecl(e.type, "Instance") and e.type.Instance == T) return .{ e.name, false };
    }
    @compileError(@typeName(T) ++ " is no catalog device's Model or Instance");
}

// ---- private ----

fn deviceIdOf(letter: u8, dev: Device) !devices.DeviceId {
    if (dev.model) |m| if (std.mem.indexOfScalar(u8, "mqdjz", letter) != null) {
        var buf: [8]u8 = undefined;
        const owner = if (m.kind.len <= buf.len) letter_kinds.get(std.ascii.lowerString(&buf, m.kind)) else null;
        if (owner != null and owner.? != letter) return error.UnsupportedDevice;
        // ponytail: vdmos.va has no polarity parameter, so a P-channel
        // VDMOS is refused rather than run N-channel; add `type` to the .va.
        if (owner == 'm' and std.ascii.eqlIgnoreCase(m.kind, "vdmos")) for (m.kv) |kv| {
            if (kv.key.len == 0 and kv.value == .name and std.ascii.eqlIgnoreCase(kv.value.name, "pchan")) return error.UnsupportedDevice;
        };
    };
    const level = try modelLevel(dev);
    return switch (letter) {
        // `.model X VDMOS(...)` has no LEVEL; the model kind selects it, as
        // in ngspice inpdomod.c.
        'm' => blk: {
            if (dev.model) |mm| {
                if (std.ascii.eqlIgnoreCase(mm.kind, "vdmos")) break :blk .vdmos;
            }
            break :blk try devices.mosfetDeviceId(level);
        },
        'q' => try devices.bjtDeviceId(level),
        'd' => try devices.diodeDeviceId(level),
        'j' => try devices.jfetDeviceId(level),
        'z' => try devices.mesDeviceId(level),
        else => devices.letter_map.get(&.{letter}) orelse
            (try inferDeviceFromModel(dev)) orelse error.UnsupportedDevice,
    };
}

fn eqlAny(a: []const u8, candidates: []const []const u8) bool {
    for (candidates) |c| if (std.ascii.eqlIgnoreCase(a, c)) return true;
    return false;
}

/// Sets P-type polarity under whichever name the model spells it. The names
/// are VerA output: Verilog-A `type` becomes `typeZ` (a Zig keyword), a
/// declared `type_` becomes `typeZ5f`. A misspelled `@hasField` is silently
/// false and runs a P-type card as N-type, so do not rename them.
fn setPolarity(comptime D: type, model: *D.Model) !void {
    if (comptime @hasField(D.Model, "typeZ")) {
        model.typeZ = -1; // i64 on most, f64 on mos1/mos2; `-1` coerces to both
        markGiven(model, "typeZ"); // bsim4va: `if (!$param_given(type)) type = NMOS`
    } else if (comptime @hasField(D.Model, "typeZ5f")) {
        model.typeZ5f = -1; // bsim2 declares `type_`
        markGiven(model, "typeZ5f");
    } else if (comptime @hasField(D.Model, "dev_type")) {
        model.dev_type = -1; // mos3
        markGiven(model, "dev_type");
    } else if (comptime @hasField(D.Model, "mtype")) {
        model.mtype = -1; // mos6/mos9 spell polarity `mtype`
        markGiven(model, "mtype");
    } else if (comptime @hasField(D.Model, "TYPE")) {
        model.TYPE = -1; // bsimsoi/hisim2/hisimhv: `TYPE` +1=NMOS, -1=PMOS
        markGiven(model, "TYPE");
    } else if (comptime @hasField(D.Model, "jtype")) {
        model.jtype = -1; // jfet/jfet2: PJF (#75)
        markGiven(model, "jtype");
    } else if (comptime @hasField(D.Model, "ztype")) {
        model.ztype = -1; // mes: PMF
        markGiven(model, "ztype");
    } else if (comptime @hasField(D.Model, "type_nfet")) {
        model.type_nfet = -1; // hfet2: PHFET
        markGiven(model, "type_nfet");
    } else {
        // ponytail: mesa and vdmos have no polarity parameter, so
        // a P-type card on them is refused rather than run N-type (silent
        // NaN). Upgrade path: a `type` parameter in the .va, as mos1 has.
        return error.UnsupportedDevice;
    }
}

/// `base` with model card `m` (netlist row `row`) and its polarity bound.
/// Cached per row when `base` is `.{}`, which holds for every device but
/// VBIC's `sw_et` pre-set.
fn boundCard(self: *NetBuilder, comptime D: type, base: D.Model, m: Model, row: u32) !D.Model {
    const t = comptime Library.builtin(device.modelName(D).?);
    const cacheable = comptime !@hasField(D.Model, "sw_et");
    if (cacheable and row != netlist.none) if (self.bound_cards[row]) |c| if (c.type == t)
        return @as(*const D.Model, @ptrCast(@alignCast(c.model))).*;
    var model = base;
    try net_cards.applyKv(&model, m.kv);
    // Polarity comes from the model card kind, outside applyKv.
    if (eqlAny(m.kind, &.{ "pmos", "pnp", "pjf", "pmf", "phfet" })) try setPolarity(D, &model);
    if (cacheable and row != netlist.none and self.bound_cards[row] == null) {
        const slot = try self.arena.create(D.Model);
        slot.* = model;
        self.bound_cards[row] = .{ .type = t, .model = slot };
    }
    return model;
}

/// The bytes that decide what `addSingleDevice` binds for `dev`: its type,
/// letter, pin count, model card row, the two positional numbers it reads
/// and every pair (all numbers). Null, so nothing is memoized, for a card
/// without a model row, a pair that is not a number, or a device that
/// calls VPI applications or keeps a row per instance. Borrows
/// `memo_key` until the next call.
fn memoKey(self: *NetBuilder, comptime D: type, dev: Device) !?[]const u8 {
    if (comptime device.modelName(D) == null) return null;
    const t = comptime Library.builtin(device.modelName(D).?);
    const vt = self.b.lib.vtable(t);
    if (vt.proto_add_row == null or vt.systf_calls.len != 0) return null;
    if (dev.model != null and dev.model_row == netlist.none) return null;
    for (dev.kv) |item| if (item.value != .num) return null;
    const k = &self.memo_key;
    k.clearRetainingCapacity();
    const head = [_]u64{
        @backingInt(t),                                                       dev.kind,
        dev.pins.len,                                                         if (dev.model == null) netlist.none else dev.model_row,
        @bitCast(Value.numberAt(dev.positional, 0) orelse std.math.nan(f64)), @bitCast(Value.numberAt(dev.positional, 1) orelse std.math.nan(f64)),
        dev.kv.len,
    };
    try k.appendSlice(self.arena, std.mem.asBytes(&head));
    for (dev.kv) |item| {
        const v: u64 = @bitCast(item.value.num);
        try k.appendSlice(self.arena, std.mem.asBytes(&[_]u64{ item.key.len, v }));
        try k.appendSlice(self.arena, item.key);
    }
    return k.items;
}

/// Refuses a card pair that neither `D.Model` nor `D.Instance` binds, as
/// ngspice refuses an unknown instance parameter: `ww=10u` for `w=10u`
/// would otherwise run with the default width (#35).
fn refuseUnknownKeys(arena: std.mem.Allocator, comptime D: type, name: []const u8, kv: []const Kv) !void {
    const params = try arena.alloc(batch.Param, kv.len);
    for (kv, params) |item, *p| p.* = .{ .key = item.key, .value = null };
    const known = try arena.alloc(bool, kv.len);
    @memset(known, false);
    batch.bind.markKnown(D.Model, params, known);
    batch.bind.markKnown(D.Instance, params, known);
    for (kv, known) |item, hit| if (!hit) {
        if (!@import("builtin").is_test) std.log.err("{s}: unknown instance parameter '{s}'", .{ name, item.key });
        return error.UnknownParameter;
    };
}

/// The model card's LEVEL, 1 when absent.
fn modelLevel(dev: Device) !u16 {
    if (dev.model) |m| {
        if (try net_cards.numericParameter(m.kv, "level")) |l| return try castField(u16, l);
    }
    return 1;
}
