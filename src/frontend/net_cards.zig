//! Card value readers. In: a card's positional and `key=value` values.
//! Out: numbers, names, a source's DC/AC/DISTO/port settings, and refusals
//! logged with the card.
//! Cites: ngspice vsrcpar.c, isrcpar.c (source keywords).
//!
//! Contents:
//! - `pub fn sourceDc(dev: Device) ?f64`
//! - `pub fn isPortCard(dev: Device) bool`
//! - `pub fn sourcePort(dev: Device) !?struct { num: u16, z0: f64, band: requests.Port.Band }`
//! - `pub fn sourceAc(dev: Device) ?struct { re: f64, im: f64 }`
//! - `pub fn sourceDisto(dev: Device, key: []const u8) [2]f64`
//! - `pub fn finiteLineCoefficients(value: anytype) bool`
//! - `pub fn cplVector(kv: []const Kv, key: []const u8, out: []f64) !usize`
//! - `pub fn refuse(dev: Device, letter: u8, why: []const u8) error{UnsupportedCard}`
//! - `pub fn kvName(kv: []const Kv, key: []const u8) ?[]const u8`
//! - `pub fn numericParameter(kv: []const Kv, key: []const u8) !?f64`
//! - `pub fn applyKv(target: anytype, kv: []const Kv) !void`
//! - `pub fn bindKv(bind: *const fn ([*]u8, []const batch.Param) batch.BindStatus, blob: [*]u8, kv: []const Kv) !void`
//! - `pub fn copyNames(arena: std.mem.Allocator, names: []const []const u8) ![]const []const u8`
//! - `pub fn kvNumber(kv: []const Kv, key: []const u8) ?f64`
//! - `pub fn resistanceFloor(r: f64) f64`

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

const net_devices = @import("net_devices.zig");

/// A source card's DC value: `dc=`, `DC v`, `DC(v)` or the first bare
/// number not owed to AC/DISTOF. Null when the card gives none.
pub fn sourceDc(dev: Device) ?f64 {
    if (kvNumber(dev.kv, "dc")) |dc| return dc;
    // Numbers still owed to a preceding AC/DISTOF keyword (mag [phase]).
    var skip: usize = 0;
    for (dev.positional, 0..) |pos, idx| switch (pos) {
        .num => |n| {
            if (skip > 0) {
                skip -= 1;
                continue;
            }
            return n;
        },
        .group => |group| {
            if (std.mem.eql(u8, group.name(), "dc") and group.args().len > 0)
                return group.args()[0].number() orelse 0;
        },
        .name => |name| {
            if (std.mem.eql(u8, name, "dc")) {
                if (Value.numberAt(dev.positional, idx + 1)) |dc| return dc;
            } else if (std.mem.eql(u8, name, "ac") or
                std.mem.eql(u8, name, "distof1") or std.mem.eql(u8, name, "distof2"))
            {
                skip = 2;
            }
        },
        else => {},
    };
    return null;
}

/// An HSPICE P card, which the reader turns into a V card (netlist.zig
/// readDevice): kind `v` under a leaf name that starts with `p`.
pub fn isPortCard(dev: Device) bool {
    const leaf = dev.name[if (std.mem.lastIndexOfScalar(u8, dev.name, '.')) |i| i + 1 else 0..];
    return dev.kind == 'v' and leaf.len > 0 and std.ascii.toLower(leaf[0]) == 'p';
}

/// The RF port of a V card (`VP1 in 0 DC 0 AC 1 portnum 1 z0 50`, or an
/// HSPICE P card's `port=1`), null when `portnum` is not given (ngspice
/// vsrctemp.c:74-82). `z0` defaults to 50 ohm.
/// Both `portnum 1` and `portnum=1` are accepted. A given `portnum` outside
/// 1..1024 or a non-positive `z0` is an error, not a plain source.
pub fn sourcePort(dev: Device) !?struct { num: u16, z0: f64, band: requests.Port.Band } {
    const num_f = blk: {
        if (kvNumber(dev.kv, "portnum") orelse kvNumber(dev.kv, "port")) |v| break :blk v;
        for (dev.positional, 0..) |pos, idx| {
            if (pos != .name or !std.mem.eql(u8, pos.name, "portnum")) continue;
            break :blk Value.numberAt(dev.positional, idx + 1) orelse return null;
        }
        return null;
    };
    const z0 = blk: {
        if (kvNumber(dev.kv, "z0")) |v| break :blk v;
        for (dev.positional, 0..) |pos, idx| {
            if (pos != .name or !std.mem.eql(u8, pos.name, "z0")) continue;
            break :blk Value.numberAt(dev.positional, idx + 1) orelse 50.0;
        }
        break :blk 50.0;
    };
    if (!(num_f >= 1) or num_f > 1024 or !(z0 > 0)) return error.InvalidAnalysisArguments;
    const h = kvNumber(dev.kv, "hblin_h") orelse 0;
    const s = kvNumber(dev.kv, "hblin_s") orelse 1;
    if (h != @trunc(h) or @abs(h) > 1024 or @abs(s) != 1) return error.InvalidAnalysisArguments;
    return .{ .num = @intFromFloat(num_f), .z0 = z0, .band = .{ .harmonic = @intFromFloat(h), .sign = @intFromFloat(s) } };
}

/// The `AC` spec of a source card as its complex excitation
/// mag * e^(j*phase*pi/180); null when the card names no `AC`, so an .ac run
/// does not drive it (ngspice vsrcacld.c:171, isrcacld.c:36). Bare `AC` is
/// mag 1 phase 0 and `AC mag` is phase 0 (vsrcpar.c:59-72, vsrctemp.c:38-43).
/// Phase is in degrees.
pub fn sourceAc(dev: Device) ?struct { re: f64, im: f64 } {
    var mag: f64 = 1;
    var phase: f64 = 0;
    var given = false;
    for (dev.positional, 0..) |pos, idx| {
        switch (pos) {
            // `AC 1 SIN 0 1 1k`: numberAt returns null on the next
            // keyword, so the trailing waveform never reads as a phase.
            .name => |name| {
                if (!std.mem.eql(u8, name, "ac")) continue;
                if (Value.numberAt(dev.positional, idx + 1)) |m| {
                    mag = m;
                    if (Value.numberAt(dev.positional, idx + 2)) |p| phase = p;
                }
            },
            .group => |group| {
                if (!std.mem.eql(u8, group.name(), "ac")) continue;
                if (group.args().len > 0) mag = group.args()[0].number() orelse mag;
                if (group.args().len > 1) phase = group.args()[1].number() orelse phase;
            },
            else => continue,
        }
        given = true;
        break;
    }
    if (!given) return null;
    const rad = phase * (std.math.pi / 180.0);
    return .{ .re = mag * @cos(rad), .im = mag * @sin(rad) };
}

/// `DISTOF1 [mag [phase]]` or `DISTOF2 ...` (`key`) on a source card
/// (ngspice vsrcpar.c:180-205): the bare keyword is mag 1 phase 0, one
/// number sets the magnitude, two set both. `{0, 0}` when the card never
/// names it, so it is no drive.
pub fn sourceDisto(dev: Device, key: []const u8) [2]f64 {
    if (kvNumber(dev.kv, key)) |mag| return .{ mag, 0 };
    for (dev.positional, 0..) |pos, idx| switch (pos) {
        .name => |name| if (std.mem.eql(u8, name, key)) return .{
            Value.numberAt(dev.positional, idx + 1) orelse 1.0,
            Value.numberAt(dev.positional, idx + 2) orelse 0.0,
        },
        else => {},
    };
    return .{ 0, 0 };
}

/// True when every float in `value`, nested arrays and structs included, is
/// finite. Setup-time only.
pub fn finiteLineCoefficients(value: anytype) bool {
    switch (@typeInfo(@TypeOf(value))) {
        .float => return std.math.isFinite(value),
        .bool => {},
        .array => for (value) |item| {
            if (!finiteLineCoefficients(item)) return false;
        },
        .@"struct" => |info| inline for (info.field_names) |name| {
            if (!finiteLineCoefficients(@field(value, name))) return false;
        },
        else => @compileError("unexpected native line coefficient type"),
    }
    return true;
}

/// Reads the CPL matrix `key` into `out` and returns its entry count: the
/// packed upper triangle is the `key=` value and the unnamed values after it.
/// 0 when the card has no `key`.
pub fn cplVector(kv: []const Kv, key: []const u8, out: []f64) !usize {
    for (kv, 0..) |entry, start| {
        if (!std.mem.eql(u8, entry.key, key)) continue;
        var i = start;
        var n: usize = 0;
        while (i < kv.len and (i == start or kv[i].key.len == 0)) : (i += 1) {
            const value = kv[i].value.number() orelse return error.InvalidParameterValue;
            if (!std.math.isFinite(value)) return error.NonFiniteParameter;
            if (n == out.len) return error.UnsupportedTransmissionLineParameters;
            out[n] = value;
            n += 1;
        }
        return n;
    }
    return 0;
}

/// Logs why HSPICE element `dev` (`letter` W or S) is refused.
pub fn refuse(dev: Device, letter: u8, why: []const u8) error{UnsupportedCard} {
    if (!@import("builtin").is_test) std.log.err("{c} element '{s}': {s}", .{ letter, dev.name, why });
    return error.UnsupportedCard;
}

pub fn kvName(kv: []const Kv, key: []const u8) ?[]const u8 {
    for (kv) |item| if (std.mem.eql(u8, item.key, key)) return switch (item.value) {
        .name => |n| n,
        else => null,
    };
    return null;
}

/// The number `key=` gives, null when absent. A present key that is not a
/// finite number is an error, never a silent fall back to the default.
pub fn numericParameter(kv: []const Kv, key: []const u8) !?f64 {
    for (kv) |item| if (std.mem.eql(u8, item.key, key)) {
        const value = item.value.number() orelse return error.UnresolvedParameter;
        if (!std.math.isFinite(value)) return error.NonFiniteParameter;
        return value;
    };
    return null;
}

/// Binds card pairs onto a built-in's Model or Instance through the
/// device's own binder, the one every loaded device uses too.
pub fn applyKv(target: anytype, kv: []const Kv) !void {
    const name, const is_model = comptime net_devices.ownerOf(@TypeOf(target.*));
    const vt = device.vtable(name);
    try bindKv(if (is_model) vt.bind_model else vt.bind_instance, @ptrCast(target), kv);
}

/// Binds `kv` onto a param blob. A non-numeric value reaches the binder as
/// null, which it rejects for a key it knows.
pub fn bindKv(bind: *const fn ([*]u8, []const batch.Param) batch.BindStatus, blob: [*]u8, kv: []const Kv) !void {
    var buf: [4096]u8 align(@alignOf(batch.Param)) = undefined;
    var fallback: std.heap.BufferFirstAllocator = .init(&buf, core.gpa);
    const a = fallback.allocator();
    const params = try a.alloc(batch.Param, kv.len);
    defer a.free(params);
    for (kv, params) |item, *p| p.* = .{ .key = item.key, .value = item.value.number() };
    try bind(blob, params).unwrap();
}

pub fn copyNames(arena: std.mem.Allocator, names: []const []const u8) ![]const []const u8 {
    const copied = try arena.alloc([]const u8, names.len);
    for (names, copied) |name, *copy| copy.* = try arena.dupe(u8, name);
    return copied;
}

pub fn kvNumber(kv: []const Kv, key: []const u8) ?f64 {
    for (kv) |item| if (std.mem.eql(u8, item.key, key)) return item.value.number();
    return null;
}

/// A card's resistance as ngspice resparam.c keeps it: the sign stays and
/// |R| is at least 1e-12 (0 is +1e-12), so `R1 a 0 -2k` is a negative
/// resistor, not the 1 mOhm restemp.c gives a missing value.
pub fn resistanceFloor(r: f64) f64 {
    return if (@abs(r) < 1e-12) std.math.copysign(@as(f64, 1e-12), r) else r;
}

// ---- private ----
