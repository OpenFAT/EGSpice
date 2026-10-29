//! HSPICE MOSRA. In: a parsed deck's `.mosra`, `.appendmodel` and
//! `.model ... MOSRA LEVEL=1` cards and the frozen circuit. Out: a `Plan`,
//! the stressed MOSFETs as SoA columns plus the aged runs as variant rows
//! whose `delvto`/`mulu0` values the facade fills from the stress transient.
//! Cites: HSPICE [SA] Ch.29 (MOSRA); dev/analysis/mosra.md.
//!
//! Contents:
//! - `pub const Plan = struct`
//! - `pub const Error = error{ OutOfMemory, TooManyInstances, UnsupportedCard }`
//! - `pub fn plan(sim: Allocator, scratch: Allocator, nl: *const Netlist, nb: *const builder.NetBuilder, circuit: *const device.Circuit, cards: []const core.query.CardRef, temp_c: f64) Error!Plan`

const std = @import("std");
const assert = std.debug.assert;
const mem = std.mem;
const Allocator = std.mem.Allocator;
const core = @import("core");
const device = @import("device");
const netlist = @import("netlist");
const builder = @import("builder");
const analyses = @import("analyses.zig");
const variants = @import("variants.zig");
const Netlist = netlist.Netlist;
const no_param = core.Mosra.no_param;

/// The resolved `.mosra`, its aged rows and how every card fans out over
/// them. `mosra.tran` is left for the caller, which holds the queries.
pub const Plan = struct { mosra: core.Mosra, variants: core.Variants, fanout: analyses.Fanout };

/// Every way `plan` fails; partial results stay in `sim` and `scratch`.
pub const Error = error{
    OutOfMemory,
    /// A device batch reported more instances than its table holds.
    TooManyInstances,
    /// The aging cannot bind: a source with no valid MOSRA LEVEL=1 card, a
    /// MOSFET without `delvto`, mobility aging without `mulu0`, no bound
    /// MOSFET, or more than `rel_times_max` times. Logged with the line.
    UnsupportedCard,
};

/// Most aged runs one `.mosra` may ask for.
const rel_times_max = 10_000;

/// The device parameters aging reads, as slots of a per-device ordinal row.
const Aged = enum(u2) { delvto, mulu0, dtemp, temp };

/// One stressed MOSFET, a row of the `core.Mosra` columns.
const Bound = struct {
    name: []const u8,
    terminal: [3]u32,
    pmos: bool,
    temp_k: f64,
    model: u16,
    delvto: u32,
    mulu0: u32,
    delvto_fresh: f64,
    mulu0_fresh: f64,
};

/// Resolves `nl.deck.mosra` against the frozen `circuit`, whose card table
/// is `cards` and whose rows `nb` maps nets to. The plan lives in `sim`;
/// `scratch` holds the lookups. Asserts that `nl.deck.mosra` is set.
pub fn plan(
    sim: Allocator,
    scratch: Allocator,
    nl: *const Netlist,
    nb: *const builder.NetBuilder,
    circuit: *const device.Circuit,
    cards: []const core.query.CardRef,
    temp_c: f64,
) Error!Plan {
    const card = nl.deck.mosra.?;
    const line = card.line;

    // One MOSRA model per distinct `.appendmodel` source; `.appendmodel` k
    // is aged by `models[source[k]]`.
    var models: std.ArrayList(core.MosraModel) = .empty;
    const source = try scratch.alloc(u16, nl.deck.appendmodels.len);
    for (nl.deck.appendmodels, source, 0..) |app, *s, k| {
        s.* = for (nl.deck.appendmodels[0..k], source[0..k]) |seen, row| {
            if (mem.eql(u8, seen.src, app.src)) break row;
        } else blk: {
            const at = nl.model_ids.get(app.src) orelse return refuse(line, "an .appendmodel source with no .model card");
            const m = nl.models[at];
            if (!std.ascii.eqlIgnoreCase(m.kind, "mosra")) return refuse(line, "an .appendmodel source that is no MOSRA model");
            try models.append(sim, try readModel(m, line));
            break :blk @intCast(models.items.len - 1);
        };
    }

    const bound = try bindDevices(sim, scratch, nl, nb, circuit, cards, temp_c, models.items, source);
    if (bound.len == 0) return refuse(line, "a .mosra that binds no MOSFET (.appendmodel)");
    const times = try relTimes(sim, card);

    const mosra: core.Mosra = .{
        .tran = .{ .t_stop = 0 },
        .rel_times = times,
        .aged_runs = card.aged_runs,
        .hci = card.rel_mode != 2,
        .bti = card.rel_mode != 1,
        .aging_start = card.aging_start,
        .aging_stop = card.aging_stop,
        .hci_threshold = card.hci_threshold,
        .bti_threshold = card.nbti_threshold,
        .deg_f = card.deg_f,
        .models = models.items,
        .names = bound.items(.name),
        .terminals = bound.items(.terminal),
        .pmos = bound.items(.pmos),
        .temp_k = bound.items(.temp_k),
        .model = bound.items(.model),
        .delvto = bound.items(.delvto),
        .mulu0 = bound.items(.mulu0),
        .delvto_fresh = bound.items(.delvto_fresh),
        .mulu0_fresh = bound.items(.mulu0_fresh),
    };
    if (!card.aged_runs) return .{ .mosra = mosra, .variants = .{}, .fanout = .{} };
    const rows = try agedRows(sim, times, mosra.delvto, mosra.mulu0);
    assert(rows.count() == times.len);
    return .{
        .mosra = mosra,
        .variants = rows,
        .fanout = .{ .global = .{ .first = 0, .count = rows.count(), .nominal = true } },
    };
}

// ---- private ----

fn refuse(line: []const u8, what: []const u8) error{UnsupportedCard} {
    if (!@import("builtin").is_test) std.log.err("netlist: .mosra: {s}: {s}", .{ what, line });
    return error.UnsupportedCard;
}

/// A `.model name MOSRA LEVEL=1 ...` card's parameters; any other level or
/// key is refused.
fn readModel(m: netlist.Model, line: []const u8) error{UnsupportedCard}!core.MosraModel {
    var md: core.MosraModel = .{};
    for (m.kv) |kv| {
        if (kv.value != .num) return refuse(line, "a MOSRA model value that is not a number");
        const v = kv.value.num;
        if (mem.eql(u8, kv.key, "level")) {
            if (v != 1) return refuse(line, "a MOSRA model LEVEL other than 1");
            continue;
        }
        inline for (@typeInfo(core.MosraModel).@"struct".field_names) |name| {
            if (mem.eql(u8, kv.key, name)) {
                @field(md, name) = v;
                break;
            }
        } else return refuse(line, "a MOSRA model key other than LEVEL, TIT0, TITFD, TITTD, TN, HCI0, HCIFD, HCITD, HCIN, TITMU, HCIMU");
    }
    if (!(md.tn > 0 and md.hcin > 0 and md.tit0 >= 0 and md.hci0 >= 0)) return refuse(line, "a MOSRA model with TN or HCIN <= 0, or a negative TIT0 or HCI0");
    assert(md.tn > 0);
    return md;
}

/// The M cards whose model an `.appendmodel` names, one `Bound` row each,
/// in card order.
fn bindDevices(
    sim: Allocator,
    scratch: Allocator,
    nl: *const Netlist,
    nb: *const builder.NetBuilder,
    circuit: *const device.Circuit,
    cards: []const core.query.CardRef,
    temp_c: f64,
    models: []const core.MosraModel,
    source: []const u16,
) Error!std.MultiArrayList(Bound).Slice {
    assert(source.len == nl.deck.appendmodels.len);
    const line = nl.deck.mosra.?.line;
    var by_name: std.StringHashMapUnmanaged(core.query.CardRef) = .empty;
    for (cards) |c| try by_name.put(scratch, c.name, c);

    // Each device's `Aged` parameter ordinals by (type, instance); a VerA
    // device keeps every parameter in its Model, one copy per instance.
    var aged: std.AutoHashMapUnmanaged(u64, [4]u32) = .empty;
    const params = try variants.collect(scratch, circuit);
    for (0..params.len) |k| {
        const ref = params.at(k);
        const slot = std.meta.stringToEnum(Aged, ref.param_name) orelse continue;
        const got = try aged.getOrPut(scratch, refKey(ref.type, ref.index));
        if (!got.found_existing) got.value_ptr.* = @splat(no_param);
        got.value_ptr[@backingInt(slot)] = @intCast(k);
    }

    var out: std.MultiArrayList(Bound) = .empty;
    for (nl.bucket('m')) |e| {
        const d = nl.device(e);
        const mm = d.model orelse continue;
        const k = for (nl.deck.appendmodels, 0..) |app, k| {
            if (mem.eql(u8, app.dst, mm.name) or mem.eql(u8, app.dst, binBase(mm.name))) break k;
        } else continue;
        const c = by_name.get(d.name) orelse return refuse(line, "a bound MOSFET that is not in the circuit");
        const ords: [4]u32 = aged.get(refKey(c.type, c.index)) orelse @splat(no_param);
        const dv = ords[@backingInt(Aged.delvto)];
        if (dv == no_param) return refuse(line, "a bound MOSFET whose model has no delvto parameter (bsim3, bsim4, BSIM-SOI, mos3 and mos9 have one)");
        const mu = ords[@backingInt(Aged.mulu0)];
        const md = models[source[k]];
        if (mu == no_param and (md.titmu != 0 or md.hcimu != 0))
            return refuse(line, try scratch.print("mobility aging (titmu, hcimu) on {s}, whose model {s} has no mulu0 (only bsim3 has one)", .{ d.name, mm.name }));
        if (d.pins.len < 3) return refuse(line, "a bound MOSFET with fewer than three terminals");
        var t: [3]u32 = undefined;
        for (&t, d.pins[0..3]) |*row, pin| {
            row.* = nb.frozenRow(pin.index());
            if (row.* == netlist.none) return refuse(line, "a bound MOSFET terminal with no circuit row");
        }
        // The device's own rule (mos3.va, ngspice): an explicit `temp`
        // wins, else the circuit temperature plus `dtemp`.
        const own = for (d.kv) |kv| {
            if (mem.eql(u8, kv.key, "temp")) break ords[@backingInt(Aged.temp)];
        } else no_param;
        const dt = ords[@backingInt(Aged.dtemp)];
        try out.append(sim, .{
            .name = try sim.dupe(u8, d.name),
            .terminal = t,
            .pmos = std.ascii.eqlIgnoreCase(mm.kind, "pmos"),
            .temp_k = 273.15 + if (own != no_param) params.get(own) else temp_c + if (dt != no_param) params.get(dt) else 0,
            .model = source[k],
            .delvto = dv,
            .mulu0 = mu,
            .delvto_fresh = params.get(dv),
            .mulu0_fresh = if (mu == no_param) 1 else params.get(mu),
        });
    }
    return out.toOwnedSlice();
}

/// `nch` for a bin card `nch.3`; the name itself otherwise.
fn binBase(name: []const u8) []const u8 {
    const dot = mem.lastIndexOfScalar(u8, name, '.') orelse return name;
    for (name[dot + 1 ..]) |ch| if (!std.ascii.isDigit(ch)) return name;
    return if (dot + 1 < name.len) name[0..dot] else name;
}

fn refKey(t: core.DeviceType, index: u32) u64 {
    return @as(u64, @backingInt(t)) << 32 | index;
}

/// The reliability times: `RelStartTime + k*RelStep` inside the run, then
/// `RelTotalTime` itself.
fn relTimes(sim: Allocator, card: netlist.Mosra) Error![]const f64 {
    var times: std.ArrayList(f64) = .empty;
    if (card.rel_step > 0) {
        var k: f64 = 0;
        while (true) : (k += 1) {
            const t = card.rel_start_time + k * card.rel_step;
            if (t >= card.rel_total_time * (1 - 1e-12)) break;
            if (t > 0) try times.append(sim, t);
            if (times.items.len >= rel_times_max) return refuse(card.line, "more than 10000 reliability times");
        }
    }
    try times.append(sim, card.rel_total_time);
    assert(times.items.len <= rel_times_max);
    return times.items;
}

/// One variant row per reliability time, writing each device's `delvto`
/// and (where it has one) `mulu0`; the values are placeholders until
/// `Problem.age` writes the aged ones.
fn agedRows(sim: Allocator, times: []const f64, delvto: []const u32, mulu0: []const u32) Allocator.Error!core.Variants {
    assert(delvto.len == mulu0.len);
    const rows = times.len;
    const labels = try sim.alloc([]const u8, rows);
    const temps = try sim.alloc(?f64, rows);
    const starts = try sim.alloc(u32, rows + 1);
    var refs: std.ArrayList(u32) = .empty;
    starts[0] = 0;
    for (times, labels, temps, starts[1..]) |t, *label, *temp, *start| {
        label.* = try sim.print("reltime={d}", .{t});
        temp.* = null;
        for (delvto, mulu0) |dv, mu| {
            try refs.append(sim, dv);
            if (mu != no_param) try refs.append(sim, mu);
        }
        start.* = @intCast(refs.items.len);
    }
    const values = try sim.alloc(f64, refs.items.len);
    @memset(values, 0);
    assert(starts[rows] == refs.items.len);
    return .{ .labels = labels, .temp_c = temps, .axis = times, .starts = starts, .refs = refs.items, .values = values };
}

test binBase {
    try std.testing.expectEqualStrings("nch", binBase("nch.3"));
    try std.testing.expectEqualStrings("nch.a", binBase("nch.a"));
    try std.testing.expectEqualStrings("nch.", binBase("nch."));
    try std.testing.expectEqualStrings("nch", binBase("nch"));
    try std.testing.expectEqualStrings("a.b", binBase("a.b.12"));
}

test readModel {
    const kv = netlist.Kv;
    const good = try readModel(.{ .name = "nra", .kind = "mosra", .kv = &[_]kv{
        .{ .key = "level", .value = .{ .num = 1 } },
        .{ .key = "tit0", .value = .{ .num = 1e-4 } },
        .{ .key = "tn", .value = .{ .num = 0.25 } },
    } }, "");
    try std.testing.expectEqual(1e-4, good.tit0);
    try std.testing.expectEqual(0.25, good.tn);
    for ([_][]const kv{
        &.{.{ .key = "level", .value = .{ .num = 2 } }},
        &.{.{ .key = "bogus", .value = .{ .num = 1 } }},
        &.{.{ .key = "tit0", .value = .{ .name = "x" } }},
        &.{.{ .key = "tn", .value = .{ .num = 0 } }},
        &.{.{ .key = "tit0", .value = .{ .num = -1 } }},
    }) |pairs| try std.testing.expectError(error.UnsupportedCard, readModel(.{ .name = "nra", .kind = "mosra", .kv = pairs }, ""));
}

test "relTimes: steps inside the run, then the total; bounded" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    // Hand-derived: start 0 step 1 total 3 gives 1 and 2 (0 is skipped,
    // 3 is the total itself), then 3.
    const t = try relTimes(arena.allocator(), .{ .rel_total_time = 3, .rel_step = 1 });
    try std.testing.expectEqualSlices(f64, &.{ 1, 2, 3 }, t);
    try std.testing.expectEqualSlices(f64, &.{5}, try relTimes(arena.allocator(), .{ .rel_total_time = 5 }));
    try std.testing.expectError(error.UnsupportedCard, relTimes(arena.allocator(), .{ .rel_total_time = 1, .rel_step = 1e-6 }));
}

test "agedRows: one row per time, mulu0 only where the device has one" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const v = try agedRows(arena.allocator(), &.{ 1, 2 }, &.{ 4, 7 }, &.{ 5, no_param });
    try std.testing.expectEqual(2, v.count());
    try std.testing.expectEqualSlices(u32, &.{ 0, 3, 6 }, v.starts);
    try std.testing.expectEqualSlices(u32, &.{ 4, 5, 7, 4, 5, 7 }, v.refs);
    try std.testing.expectEqualStrings("reltime=2", v.labels[1]);
}
