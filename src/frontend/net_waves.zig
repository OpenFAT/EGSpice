//! Source waveforms. In: a V or I card's PULSE, SIN, EXP, PWL, SFFM or AM
//! arguments. Out: the source model's waveform fields.
//! Cites: ngspice vsrcload.c, isrcload.c (wave shapes, defaults).
//!
//! Contents:
//! - `pub const Wave`
//! - `pub fn refuseUnknownWave(dev: Device) error{UnsupportedCard}!void`
//! - `pub fn waveKind(name: []const u8) ?Wave`
//! - `pub fn pwlSlot(comptime base: []const u8, comptime k: usize) []const u8`
//! - `pub fn pwlCapacity(comptime T: type) usize`
//! - `pub fn slotCount(comptime T: type, comptime base: []const u8) usize`
//! - `pub fn setArray(comptime T: type, model: *T, comptime base: []const u8, values: anytype) void`
//! - `pub fn applySourceWaveform(target: anytype, dev: Device) void`
//! - `pub fn waveAt(dev: Device) ?WaveAt`
//! - `pub fn pwlArgs(dev: Device) ?[]const Value`
//! - `pub fn pwlTail(comptime D: type, arena: std.mem.Allocator, dev: Device) ![]const D.Model`
//! - `pub fn dcFromWaveform(target: anytype) void`

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

const net_cards = @import("net_cards.zig");

/// Source waveform shapes; the tag is the model's `waveform` code (0 is DC only).
pub const Wave = enum(u8) { pulse = 1, sin = 2, exp = 3, pwl = 4, sffm = 5, am = 6 };

/// A card's waveform: its kind, its arguments, and the positional range
/// `[start, end)` that spells it (one group, or a keyword and its numbers).
const WaveAt = struct { kind: Wave, args: []const Value, start: usize, end: usize };

/// Source waveforms ngspice or HSPICE define that the source models do not
/// draw. A card naming one is refused rather than run as its DC value
/// (#72), and so is HSPICE's `PWL FILE=`.
/// ponytail: refused, not drawn; TRNOISE/TRRANDOM need ngspice's noise
/// generator per source and breakpoints at each sample.
pub fn refuseUnknownWave(dev: Device) error{UnsupportedCard}!void {
    const unknown = std.StaticStringMap(void).initComptime(.{ .{"trnoise"}, .{"trrandom"}, .{"lfsr"}, .{"hb"} });
    var buf: [8]u8 = undefined;
    for (dev.positional) |v| {
        const name = switch (v) {
            .group => |g| g.name(),
            .name => |n| n,
            else => continue, // else: numbers and expressions name no waveform
        };
        if (name.len > buf.len or !unknown.has(std.ascii.lowerString(&buf, name))) continue;
        if (!@import("builtin").is_test) std.log.err("{s}: the {s} source waveform is not supported", .{ dev.name, name });
        return error.UnsupportedCard;
    }
    for (dev.kv) |item| if (std.ascii.eqlIgnoreCase(item.key, "file")) {
        if (!@import("builtin").is_test) std.log.err("{s}: a PWL FILE= source is not supported", .{dev.name});
        return error.UnsupportedCard;
    };
}

/// The waveform a source keyword names, case-insensitively.
pub fn waveKind(name: []const u8) ?Wave {
    const map = std.StaticStringMap(Wave).initComptime(.{
        .{ "pulse", .pulse }, .{ "sin", .sin },   .{ "exp", .exp },
        .{ "pwl", .pwl },     .{ "sffm", .sffm }, .{ "am", .am },
    });
    var buf: [8]u8 = undefined;
    if (name.len > buf.len) return null;
    return map.get(std.ascii.lowerString(&buf, name));
}

/// Field name of slot `k` of a flattened Verilog-A array. VerA turns
/// `parameter real pwl_times[0:63]` into 64 scalar Model fields named with
/// the escaped subscript (`[` is `Z5b`, `]` is `Z5d`; see VerA
/// lib/backend/naming.zig), so the slot is picked at comptime.
pub fn pwlSlot(comptime base: []const u8, comptime k: usize) []const u8 {
    // 2 arrays x 64 slots of comptime std.fmt exceed the default quota.
    @setEvalBranchQuota(200_000);
    return std.fmt.comptimePrint("{s}Z5b{d}Z5d", .{ base, k });
}

/// PWL table capacity, read off the struct so it follows `max_pwl` in the .va.
pub fn pwlCapacity(comptime T: type) usize {
    return slotCount(T, "pwl_times");
}

/// Slots of the flattened Verilog-A array `base` in `T`.
pub fn slotCount(comptime T: type, comptime base: []const u8) usize {
    // The S and W templates hold arrays of a few hundred slots.
    @setEvalBranchQuota(10_000_000);
    comptime var n: usize = 0;
    inline while (@hasField(T, pwlSlot(base, n))) : (n += 1) {}
    return n;
}

/// Writes `values` over VerA array parameter `base` (fields `base[k]`),
/// whose slot count must equal `values.len`.
pub fn setArray(comptime T: type, model: *T, comptime base: []const u8, values: anytype) void {
    comptime std.debug.assert(slotCount(T, base) == values.len);
    @setEvalBranchQuota(10_000_000);
    inline for (0..values.len) |k| @field(model, pwlSlot(base, k)) = values[k];
}

/// Writes the card's waveform (group or parenless spelling) into `target`,
/// a source Model or Instance; a no-op on a struct without `waveform`.
pub fn applySourceWaveform(target: anytype, dev: Device) void {
    const T = @TypeOf(target.*);
    if (comptime !@hasField(T, "waveform")) return;
    var i: usize = 0;
    while (i < dev.positional.len) : (i += 1) {
        switch (dev.positional[i]) {
            .group => |group| {
                const kind = waveKind(group.name()) orelse continue;
                applyWaveArgs(T, target, kind, group.args());
            },
            // Parenless spelling (`vs a 0 dc=0 sin 0 50 100k`): the keyword is
            // a bare positional name and its args are the numeric positionals
            // that follow.
            .name => |nm| {
                const kind = waveKind(nm) orelse continue;
                const start = i + 1;
                var end = start;
                while (end < dev.positional.len and dev.positional[end] == .num) end += 1;
                applyWaveArgs(T, target, kind, dev.positional[start..end]);
                i = end - 1;
            },
            else => {},
        }
    }
}

/// The first waveform of `dev`, or null when it has none.
pub fn waveAt(dev: Device) ?WaveAt {
    for (dev.positional, 0..) |v, i| switch (v) {
        .group => |g| if (waveKind(g.name())) |kind| return .{ .kind = kind, .args = g.args(), .start = i, .end = i + 1 },
        .name => |nm| if (waveKind(nm)) |kind| {
            var end = i + 1;
            while (end < dev.positional.len and dev.positional[end] == .num) end += 1;
            return .{ .kind = kind, .args = dev.positional[i + 1 .. end], .start = i, .end = end };
        },
        else => {},
    };
    return null;
}

/// The PWL points of `dev`'s `PWL(...)` group, or null when it has none.
pub fn pwlArgs(dev: Device) ?[]const Value {
    const w = waveAt(dev) orelse return null;
    return if (w.kind == .pwl) w.args else null;
}

/// Models for the segments of a PWL past `D`'s table capacity: segment k
/// starts on the last point of the one before it and holds the change from
/// there, so the card's source (the first `capacity` points) plus every
/// segment is the whole waveform: in series for a V source, in parallel for
/// an I source. Empty when the PWL fits. ngspice reads any length; a
/// truncated table would end the waveform early without a word.
pub fn pwlTail(comptime D: type, arena: std.mem.Allocator, dev: Device) ![]const D.Model {
    if (comptime !@hasField(D.Model, pwlSlot("pwl_times", 0))) return &.{};
    const cap = comptime pwlCapacity(D.Model);
    const args = pwlArgs(dev) orelse return &.{};
    const n = args.len / 2;
    if (n <= cap) return &.{};
    if (net_cards.kvNumber(dev.kv, "r") != null) {
        if (!@import("builtin").is_test) std.log.err("{s}: a PWL with r= holds at most {d} points", .{ dev.name, cap });
        return error.UnsupportedCard;
    }
    const td = net_cards.kvNumber(dev.kv, "td") orelse 0;
    var out: std.ArrayList(D.Model) = .empty;
    var start: usize = cap - 1;
    while (start + 1 < n) : (start += cap - 1) {
        const t0 = args[2 * start].number() orelse return error.UnresolvedParameter;
        const v0 = args[2 * start + 1].number() orelse return error.UnresolvedParameter;
        var m: D.Model = .{};
        m.waveform = @backingInt(Wave.pwl);
        m.pwl_td = td;
        const len = @min(cap, n - start);
        inline for (0..cap) |k| if (k < len) {
            const j = start + k;
            @field(m, pwlSlot("pwl_times", k)) = args[2 * j].number() orelse t0;
            @field(m, pwlSlot("pwl_values", k)) = (args[2 * j + 1].number() orelse v0) - v0;
        };
        m.pwl_len = @intCast(len);
        try out.append(arena, m);
    }
    return out.items;
}

/// Sets `dc` to the waveform's t = 0 value (each pre-TD branch of
/// vsource.va/isource.va): PULSE/EXP hold their first level, SIN is
/// VO + VA*sin(2*pi*phase/360), PWL its first point. AM starts at 0, the
/// `dc` default already.
pub fn dcFromWaveform(target: anytype) void {
    const T = @TypeOf(target.*);
    if (comptime !@hasField(T, "waveform") or !@hasField(T, "dc")) return;
    const rd = struct {
        fn f(t: anytype, comptime a: []const u8, comptime b: []const u8) f64 {
            if (comptime @hasField(T, a)) return @field(t, a);
            if (comptime @hasField(T, b)) return @field(t, b);
            return 0;
        }
    }.f;
    const v: f64 = switch (target.waveform) {
        @backingInt(Wave.pulse) => rd(target.*, "pulse_v1", "pulse_i1"),
        @backingInt(Wave.sin) => rd(target.*, "sin_vo", "sin_ioff") +
            rd(target.*, "sin_va", "sin_iamp") *
                @sin(2.0 * std.math.pi * rd(target.*, "sin_phase", "sin_phase") / 360.0),
        @backingInt(Wave.exp) => rd(target.*, "exp_v1", "exp_i1"),
        @backingInt(Wave.pwl) => rd(target.*, pwlSlot("pwl_values", 0), pwlSlot("pwl_values", 0)),
        // ngspice's DCOP evaluates SFFM at time 0: the V source is 0 there
        // (vsrcload.c:271-274), the I source has no delay and reads its phases
        // one slot early (isrcload.c:222-252, see isource.va).
        @backingInt(Wave.sffm) => if (comptime !@hasField(T, "sffm_fm") or T == devices.vsource.Model or T == devices.vsource.Instance) 0 else blk: {
            const mdi = if (target.sffm_mdi > target.sffm_fc / target.sffm_fm)
                target.sffm_fc / target.sffm_fm
            else
                @max(target.sffm_mdi, 0);
            break :blk target.sffm_vo + target.sffm_va *
                @sin(target.sffm_phasem * std.math.pi / 180.0 + mdi * @sin(target.sffm_td * std.math.pi / 180.0));
        },
        else => return,
    };
    target.dc = @floatCast(v);
}

// ---- private ----

fn applyWaveArgs(comptime T: type, target: anytype, kind: Wave, args: []const Value) void {
    target.waveform = @backingInt(kind);
    switch (kind) {
        // `PWL(T1 V1 T2 V2 ...)`: (time, value) pairs into the flattened
        // table. A non-numeric arg leaves its slot at the default; ngspice's
        // `r=`/`td=` suffixes are card pairs that bindSource maps onto
        // `pwl_repeat`/`pwl_td`.
        .pwl => if (comptime @hasField(T, pwlSlot("pwl_times", 0))) {
            const n_pts = @min(args.len / 2, comptime pwlCapacity(T));
            inline for (0..comptime pwlCapacity(T)) |k| {
                if (k < n_pts) {
                    if (args[2 * k].number()) |t|
                        @field(target.*, pwlSlot("pwl_times", k)) = @floatCast(t);
                    if (args[2 * k + 1].number()) |v|
                        @field(target.*, pwlSlot("pwl_values", k)) = @floatCast(v);
                }
            }
            target.pwl_len = @intCast(n_pts);
        },
        // ngspice's omitted EXP time constants are TSTEP, which
        // resolvePulseDefaults fills over 0.
        .exp => {
            if (comptime @hasField(T, "exp_tau1")) {
                target.exp_tau1 = 0;
                target.exp_tau2 = 0;
            }
            applyGroupArgs(T, target, args, comptime fieldPairs(.exp));
        },
        inline else => |cw| applyGroupArgs(T, target, args, comptime fieldPairs(cw)),
    }
}

/// Positional waveform args onto fields; arg k goes to `field_pairs[2k]`,
/// the V-source name, or else `field_pairs[2k + 1]`, the I-source name.
fn applyGroupArgs(comptime T: type, target: anytype, args: []const Value, comptime field_pairs: []const []const u8) void {
    comptime var i: usize = 0;
    inline while (i < field_pairs.len) : (i += 2) {
        const arg_idx = i / 2;
        if (arg_idx < args.len) {
            if (args[arg_idx].number()) |v| {
                if (comptime @hasField(T, field_pairs[i])) @field(target.*, field_pairs[i]) = @floatCast(v) else if (comptime @hasField(T, field_pairs[i + 1])) @field(target.*, field_pairs[i + 1]) = @floatCast(v);
            }
        }
    }
}

/// (V-source field, I-source field) per positional arg of waveform `w`.
fn fieldPairs(comptime w: Wave) []const []const u8 {
    return switch (w) {
        .pulse => &.{ "pulse_v1", "pulse_i1", "pulse_v2", "pulse_i2", "pulse_td", "pulse_td", "pulse_tr", "pulse_tr", "pulse_tf", "pulse_tf", "pulse_pw", "pulse_pw", "pulse_per", "pulse_per", "pulse_phase", "pulse_phase" },
        .sin => &.{ "sin_vo", "sin_ioff", "sin_va", "sin_iamp", "sin_freq", "sin_freq", "sin_td", "sin_td", "sin_theta", "sin_theta", "sin_phase", "sin_phase" },
        .exp => &.{ "exp_v1", "exp_i1", "exp_v2", "exp_i2", "exp_td1", "exp_td1", "exp_tau1", "exp_tau1", "exp_td2", "exp_td2", "exp_tau2", "exp_tau2" },
        .pwl => &.{},
        // ngspice 44 order (vsrcload.c:235-249): FM third, FC fifth.
        .sffm => &.{ "sffm_vo", "sffm_vo", "sffm_va", "sffm_va", "sffm_fm", "sffm_fm", "sffm_mdi", "sffm_mdi", "sffm_fc", "sffm_fc", "sffm_td", "sffm_td", "sffm_phasem", "sffm_phasem", "sffm_phasec", "sffm_phasec" },
        .am => &.{ "am_va", "am_va", "am_vo", "am_vo", "am_mf", "am_mf", "am_fc", "am_fc", "am_td", "am_td", "am_phasec", "am_phasec", "am_phases", "am_phases" },
    };
}
