//! Periodic and RF cards to queries. In: one `.pss`, `.hb`, `.qpss`,
//! `.hbac`/`.hbxf`/`.hbnoise`, `.hblin`, `.phasenoise`, `.pac`/`.pxf`,
//! `.pnoise` or `.sp` (`.lin`, `.net`) card, or HSPICE's shooting-Newton
//! and `.fft` forms. Out: its query.
//! Cites: HSPICE [CR] .SN, .SNAC, .SNNOISE, .SNXF, .PTDNOISE, .HBLIN, .LIN,
//! .NET, .FFT, [RF] Ch.7.
//!
//! Contents:
//! - `pub fn pnoiseJob(a: netlist.Analysis, sources: core.QueryBindings, ctx: analysis_args.CardContext) !Job`
//! - `pub fn pssJob(a: netlist.Analysis) !Job`
//! - `pub fn hbJob(a: netlist.Analysis, ctx: analysis_args.CardContext) !Job`
//! - `pub fn qpssJob(a: netlist.Analysis) !Job`
//! - `pub fn hbacJob(a: netlist.Analysis, sources: core.QueryBindings) !Job`
//! - `pub fn hblinJob(a: netlist.Analysis, sources: core.QueryBindings, ctx: analysis_args.CardContext) !Job`
//! - `pub fn phasenoiseJob(a: netlist.Analysis) !Job`
//! - `pub fn pacJob(a: netlist.Analysis, ctx: analysis_args.CardContext) !Job`
//! - `pub fn spJob(a: netlist.Analysis, sources: core.QueryBindings, ctx: analysis_args.CardContext) !Job`

const std = @import("std");
const assert = std.debug.assert;
const testing = std.testing;
const Allocator = std.mem.Allocator;
const core = @import("core");
const requests = core.query;
const numerics = core.numerics;
const netlist = @import("netlist");
const Value = netlist.Value;
const nameAt = Value.nameAt;
const numberAt = Value.numberAt;
const Job = requests.Query;
const GROUND = core.GROUND;
const Library = @import("device").Library;
const analysis_args = @import("analysis_args.zig");
const analysis_time = @import("analysis_time.zig");

/// The `pnoise` card.
pub fn pnoiseJob(a: netlist.Analysis, sources: core.QueryBindings, ctx: analysis_args.CardContext) !Job {
    const args = a.args;
    const node_id = a.pos;
    const node_neg = a.neg;
    // HSPICE `.snnoise v(out) insrc <sweep> [n1 +/-1]` [CR .SNNOISE]
    // at the `.sn` fundamental. Only the n1 = 0 band, the input's
    // own frequency, is the band `.pnoise` measures.
    if (a.sn and std.ascii.startsWithIgnoreCase(a.line, ".ptdnoise")) return .{ .pnoise = try analysis_time.ptdNoise(ctx, args, node_id, node_neg) };
    if (a.sn) {
        const f0 = ctx.sn_f0 orelse return error.MissingAnalysisCard;
        try analysis_args.inputSource(args, 1, sources);
        const grid = try analysis_args.acGrid(ctx.arena, args, 2);
        if (args.len > grid.end + 2) return error.InvalidAnalysisArguments;
        if (grid.end < args.len and try analysis_args.number(args, grid.end) != 0) return error.InvalidAnalysisArguments;
        return .{ .pnoise = .{ .out_node = try analysis_args.outputNode(node_id), .out_neg = try analysis_args.outputNeg(node_neg), .sweep = grid.sweep, .f_fundamental = f0 } };
    }
    try analysis_args.arity(args, 7, 8);
    try analysis_args.inputSource(args, 1, sources);
    const sweep = try analysis_args.frequencySweep(args, 2);
    const sidebands = if (args.len == 8) try analysis_args.number(args, 7) else 7;
    if (sidebands < 0 or sidebands != @trunc(sidebands) or sidebands > 31) return error.InvalidAnalysisArguments;
    return .{ .pnoise = .{ .out_node = try analysis_args.outputNode(node_id), .out_neg = try analysis_args.outputNeg(node_neg), .sweep = sweep, .f_fundamental = try analysis_args.positive(args, 6), .n_sidebands = @intFromFloat(sidebands) } };
}

/// The `pss` card.
pub fn pssJob(a: netlist.Analysis) !Job {
    const args = a.args;
    const node_id = a.pos;
    const node_neg = a.neg;
    try analysis_args.singleEnded(node_neg);
    if (a.sn) return .{ .pss = try shootingNewton(args) };
    // `.pss v(osc) f [n [settle]]` (HSPICE `.snosc`) solves an
    // oscillator: f is the first guess and osc the phase node.
    const osc = args.len != 0 and args[0] == .group;
    const at: usize = @intFromBool(osc);
    try analysis_args.arity(args, at + 1, at + @as(usize, if (osc) 3 else 2));
    var opts: requests.Pss = .{ .period = 1 / try analysis_args.positive(args, at), .n_samples = try analysis_args.count(u32, args, at + 1, 256) };
    if (osc) {
        opts.osc_node = try analysis_args.outputNode(node_id);
        opts.osc_settle_periods = try analysis_args.count(u16, args, at + 2, 30);
    }
    return .{ .pss = opts };
}

/// The `hb` card.
pub fn hbJob(a: netlist.Analysis, ctx: analysis_args.CardContext) !Job {
    const args = a.args;
    const node_id = a.pos;
    const node_neg = a.neg;
    try analysis_args.singleEnded(node_neg);
    // HB drives from whatever sources the deck has, current ones
    // included, so it names none. `.hb v(osc) f [K]` (HSPICE
    // `.hbosc`) solves an oscillator from the first guess f.
    if (nameAt(args, 0) != null) return .{ .hb = try hbTones(ctx.arena, args) };
    const osc = args.len != 0 and args[0] == .group;
    const at: usize = @intFromBool(osc);
    try analysis_args.arity(args, at + 1, at + 2);
    return .{ .hb = .{
        .f0 = try analysis_args.positive(args, at),
        .n_harmonics = try analysis_args.count(u16, args, at + 1, 8),
        .osc_node = if (osc) try analysis_args.outputNode(node_id) else GROUND,
    } };
}

/// The `qpss` card.
pub fn qpssJob(a: netlist.Analysis) !Job {
    const args = a.args;
    // Like HB, QPSS drives from every source the deck stamps.
    try analysis_args.arity(args, 2, 4);
    return .{ .qpss = .{ .f1 = try analysis_args.positive(args, 0), .f2 = try analysis_args.positive(args, 1), .k1 = try analysis_args.count(u16, args, 2, 5), .k2 = try analysis_args.count(u16, args, 3, 5) } };
}

/// The `hbac`, `hbxf`, `hbnoise` cards.
pub fn hbacJob(a: netlist.Analysis, sources: core.QueryBindings) !Job {
    const args = a.args;
    const id = a.kind;
    const node_id = a.pos;
    const node_neg = a.neg;
    // `.hbac sweep`, `.hbxf v(out) sweep` and
    // `.hbnoise v(out[,ref]) [Vsrc] sweep`, each optionally followed
    // by `f0 [K]` (`.hbnoise`: `f0 [K [M]]`). Without f0, `queries`
    // fills the tone from the deck's `.hb` card, as HSPICE does.
    var at: usize = @intFromBool(id != .hbac);
    // As in `.pnoise`, the input source is checked but does not
    // drive the solve.
    if (id == .hbnoise and nameAt(args, at) != null and std.meta.isError(analysis_args.frequencySweep(args, at))) {
        _ = try analysis_args.voltageSource(args, at, sources);
        at += 1;
    }
    const grid = try analysis_args.frequencySweep(args, at);
    at += 4;
    try analysis_args.arity(args, at, at + @as(usize, if (id == .hbnoise) 3 else 2));
    if (id != .hbnoise and node_neg != analysis_args.NO_NODE) return error.InvalidAnalysisArguments;
    var opts: requests.HbLptv = .{ .f0 = 0, .sweep = grid, .out_node = try analysis_args.outputNode(node_id), .out_neg = try analysis_args.outputNeg(node_neg) };
    if (at < args.len) {
        opts.f0 = try analysis_args.positive(args, at);
        opts.n_harmonics = try analysis_args.count(u16, args, at + 1, 8);
        opts.n_sidebands = opts.n_harmonics;
        if (at + 2 < args.len) {
            const m = try analysis_args.number(args, at + 2);
            if (m < 0 or m != @trunc(m) or m > 31) return error.InvalidAnalysisArguments;
            opts.n_sidebands = @intFromFloat(m);
        }
    }
    return switch (id) {
        .hbac => .{ .hbac = opts },
        .hbxf => .{ .hbxf = opts },
        else => .{ .hbnoise = opts },
    };
}

/// The `hblin` card.
pub fn hblinJob(a: netlist.Analysis, sources: core.QueryBindings, ctx: analysis_args.CardContext) !Job {
    const args = a.args;
    // HSPICE RF `.hblin <sweep> [NOISECALC=0|1] [FILENAME= DATAFORMAT=
    // MIXEDMODE2PORT=ss]` [CR .HBLIN] at the `.hb` card's tone, over
    // the deck's P elements. The file keywords are checked and
    // unused; mixed mode is refused.
    const grid = try analysis_args.acGrid(ctx.arena, args, 0);
    var noise = false;
    var i: usize = grid.end;
    while (i < args.len) : (i += 2) {
        var lower: [16]u8 = undefined;
        const key = try analysis_args.keyword(args, i, &lower);
        if (i + 1 >= args.len) return error.InvalidAnalysisArguments;
        if (std.mem.eql(u8, key, "noisecalc")) {
            noise = if (numberAt(args, i + 1)) |v| v != 0 else !std.ascii.eqlIgnoreCase(nameAt(args, i + 1) orelse "", "no");
        } else if (std.mem.eql(u8, key, "mixedmode2port")) {
            if (!std.ascii.eqlIgnoreCase(nameAt(args, i + 1) orelse "", "ss")) return error.UnsupportedAnalysisOutput;
        } else if (!std.mem.eql(u8, key, "filename") and !std.mem.eql(u8, key, "dataformat")) return error.InvalidAnalysisArguments;
    }
    if (sources.ports.len == 0) return error.MissingAnalysisCard;
    // ponytail: mixed-mode P cards only in `.lin`; `.hblin` would
    // need the leg drives in every band.
    for (sources.ports) |p| if (p.balanced != null) return error.UnsupportedAnalysisOutput;
    return .{ .hblin = .{ .f0 = 0, .sweep = grid.sweep, .ports = sources.ports, .noise = noise } };
}

/// The `phasenoise` card.
pub fn phasenoiseJob(a: netlist.Analysis) !Job {
    const args = a.args;
    const node_id = a.pos;
    const node_neg = a.neg;
    // `.phasenoise v(out) sweep [f0 [K]] [METHOD=0|1|2]
    // [CARRIERINDEX=k] [LIST...=] [SPURIOUS=0]` [RF Ch.7]: without
    // f0 the oscillator is the deck's `.hbosc`; with it, v(out) is
    // the phase node. The LIST keywords shape HSPICE's listing and
    // are checked and unused.
    try analysis_args.singleEnded(node_neg);
    const out = try analysis_args.outputNode(node_id);
    var o: requests.PhaseNoise = .{ .f0 = 0, .osc_node = out, .sweep = try analysis_args.frequencySweep(args, 1) };
    var i: usize = 5;
    if (i < args.len and args[i] == .num) {
        o.f0 = try analysis_args.positive(args, i);
        // K is optional: a keyword after f0 starts the pairs.
        if (numberAt(args, i + 1) != null) o.n_harmonics = try analysis_args.count(u16, args, i + 1, 8);
        i += if (i + 1 < args.len and args[i + 1] == .num) 2 else 1;
    }
    const Key = enum { method, carrierindex, listfreq, listcount, listfloor, listsources, spurious };
    while (i < args.len) : (i += 2) {
        var lower: [16]u8 = undefined;
        const key = std.meta.stringToEnum(Key, try analysis_args.keyword(args, i, &lower)) orelse return error.InvalidAnalysisArguments;
        if (i + 1 >= args.len) return error.InvalidAnalysisArguments;
        switch (key) {
            .method => {
                const m = try analysis_args.number(args, i + 1);
                if (m != 0 and m != 1 and m != 2) return error.InvalidAnalysisArguments;
                o.method = @fromBackingInt(@intCast(@as(u2, @intFromFloat(m))));
            },
            .carrierindex => o.carrier = try analysis_args.count(u16, args, i + 1, 1),
            .spurious => if (try analysis_args.number(args, i + 1) != 0) return error.UnsupportedAnalysisOutput,
            .listfreq, .listcount, .listfloor, .listsources => {},
        }
    }
    return .{ .phasenoise = o };
}

/// The `pac`, `pxf` cards.
pub fn pacJob(a: netlist.Analysis, ctx: analysis_args.CardContext) !Job {
    const args = a.args;
    const id = a.kind;
    const node_id = a.pos;
    const node_neg = a.neg;
    try analysis_args.singleEnded(node_neg);
    // HSPICE `.snac <sweep>` and `.snxf v(out) <sweep>` [CR .SNAC,
    // .SNXF] run at the `.sn` fundamental.
    var opts: requests.Pac = undefined;
    if (a.sn) {
        const lo = ctx.sn_f0 orelse return error.MissingAnalysisCard;
        const grid = try analysis_args.acGrid(ctx.arena, args, @intFromBool(id == .pxf));
        if (grid.end != args.len) return error.InvalidAnalysisArguments;
        opts = .{ .f_lo = lo, .sweep = grid.sweep, .out_node = try analysis_args.outputNode(node_id) };
    } else {
        try analysis_args.arity(args, 5, 5);
        const sweep = try analysis_args.frequencySweep(args, 1);
        opts = .{ .f_lo = try analysis_args.positive(args, 0), .sweep = sweep, .out_node = try analysis_args.outputNode(node_id) };
    }
    return if (id == .pac) .{ .pac = opts } else .{ .pxf = opts };
}

/// The `sp` card.
pub fn spJob(a: netlist.Analysis, sources: core.QueryBindings, ctx: analysis_args.CardContext) !Job {
    const args = a.args;
    if (analysis_args.cardIs(a.line, "net")) return .{ .sp = try net(a, sources, ctx) };
    if (analysis_args.cardIs(a.line, "lin")) return .{ .sp = .{ .sweep = ctx.ac orelse return error.MissingAnalysisCard, .ports = sources.ports, .lin = try lin(args, sources.ports) } };
    try analysis_args.arity(args, 4, 4);
    return .{ .sp = .{ .sweep = try analysis_args.frequencySweep(args, 0), .ports = sources.ports } };
}

// ---- private ----

/// HSPICE `.net input [RIN=r]`, `.net input r` and `.net output input
/// [ROUT=r] [RIN=r]` [CR App.A], obsolete, over the `.ac` sweep: the S, Y,
/// Z and H of the network between the input source's terminals (port 1)
/// and the output, `v(n1[,n2])` or `i(Vx)` (port 2), each port driven as
/// what it is (a V card's voltage, an I card's or node pair's current) and
/// S taken against RIN and ROUT, which default to 1 ohm. A V card's port
/// keeps its branch; the others have `Port.no_branch`.
fn net(a: netlist.Analysis, sources: core.QueryBindings, ctx: analysis_args.CardContext) !requests.Sp {
    const args = a.args;
    const ports = try ctx.arena.alloc(requests.Port, 2);
    var n_ports: usize = 1;
    var i: usize = 0;
    if (args.len > 0 and args[0] == .group) {
        // Two-port: the output first.
        ports[1] = if (analysis_args.currentProbeName(args, 0)) |probe| blk: {
            const v = netlist.nameIndex(sources.v_names, probe) orelse return error.AnalysisSourceNotFound;
            break :blk .{ .node = sources.v_pos[v], .neg = sources.v_neg[v], .branch = sources.v_branches[v] };
        } else .{ .node = a.pos, .neg = if (a.neg == netlist.none) GROUND else a.neg, .branch = requests.Port.no_branch };
        if (ports[1].node == netlist.none) return error.AnalysisNodeNotFound;
        n_ports = 2;
        i = 1;
    }
    const input = nameAt(args, i) orelse return error.InvalidAnalysisArguments;
    // An I card delivers its current into n-, so that is the port's + terminal.
    ports[0] = if (netlist.nameIndex(sources.v_names, input)) |v|
        .{ .node = sources.v_pos[v], .neg = sources.v_neg[v], .branch = sources.v_branches[v] }
    else if (netlist.nameIndex(sources.i_names, input)) |k|
        .{ .node = sources.i_neg[k], .neg = sources.i_pos[k], .branch = requests.Port.no_branch }
    else
        return error.AnalysisSourceNotFound;
    i += 1;
    var z0 = [2]f64{ 1, 1 };
    if (n_ports == 1 and i < args.len and args[i] == .num) {
        z0[0] = try analysis_args.positive(args, i);
        i += 1;
    }
    while (i < args.len) : (i += 2) {
        const at: usize = if (analysis_args.isKey(args, i, "rin")) 0 else if (n_ports == 2 and analysis_args.isKey(args, i, "rout")) 1 else return error.InvalidAnalysisArguments;
        z0[at] = try analysis_args.positive(args, i + 1);
    }
    for (ports[0..n_ports], z0[0..n_ports]) |*p, z| p.z0 = z;
    return .{ .sweep = ctx.ac orelse return error.MissingAnalysisCard, .ports = ports[0..n_ports], .lin = .{}, .net = true };
}

/// HSPICE `.lin [sparcalc=1] [noisecalc=0|1|2] [gdcalc=0|1]
/// [mixedmode2port=xy] [format= filename= dataformat= modelname= ...]`
/// [CR .LIN] over the `.ac` sweep. S is always computed. noisecalc=2 (the
/// N-port correlation matrix) publishes the two-port parameters only.
/// mixedmode2port names port 1's and port 2's mode in the two-port
/// measurements: `s` for a single-ended port, `d` or `c` for a balanced
/// one; any other pairing is InvalidAnalysisArguments.
/// `format=touchstone|touchstone2` also writes a Touchstone 1.0 file named
/// by `filename=`; the other formats (`selem`, `citi`) and the listing
/// keywords are checked for a value and unused.
fn lin(args: []const Value, ports: []const requests.Port) !requests.Sp.Lin {
    const Key = enum { sparcalc, noisecalc, gdcalc, mixedmode2port, format, filename, dataformat, modelname, freqdigit, spardigit, listfreq, listcount, listfloor, listsources };
    var o: requests.Sp.Lin = .{};
    var i: usize = 0;
    while (i < args.len) : (i += 2) {
        var lower: [16]u8 = undefined;
        const key = std.meta.stringToEnum(Key, try analysis_args.keyword(args, i, &lower)) orelse return error.InvalidAnalysisArguments;
        if (i + 1 >= args.len) return error.InvalidAnalysisArguments;
        switch (key) {
            .sparcalc, .noisecalc, .gdcalc => {
                const v = try analysis_args.number(args, i + 1);
                if (v != @trunc(v) or v < 0 or v > @as(f64, if (key == .noisecalc) 2 else 1)) return error.InvalidAnalysisArguments;
                if (key == .noisecalc) o.noise = v != 0;
                if (key == .gdcalc) o.group_delay = v != 0;
            },
            .mixedmode2port => {
                const mode = nameAt(args, i + 1) orelse return error.InvalidAnalysisArguments;
                if (mode.len != 2) return error.InvalidAnalysisArguments;
                for (mode, 0..) |c, k| {
                    const balanced = k < ports.len and ports[k].balanced != null;
                    switch (std.ascii.toLower(c)) {
                        's' => if (balanced) return error.InvalidAnalysisArguments,
                        'd', 'c' => if (!balanced) return error.InvalidAnalysisArguments,
                        else => return error.InvalidAnalysisArguments,
                    }
                    o.common[k] = std.ascii.toLower(c) == 'c';
                }
            },
            .format => {
                const f = nameAt(args, i + 1) orelse return error.InvalidAnalysisArguments;
                o.touchstone = std.ascii.startsWithIgnoreCase(f, "touchstone");
            },
            .filename => o.file = nameAt(args, i + 1) orelse return error.InvalidAnalysisArguments,
            else => {},
        }
    }
    return o;
}

/// HSPICE `.hb TONES=f0 [f1 ...] [NHARMS=h0 [h1 ...]] [INTMODMAX=n]
/// [SUBHARMS=s] [SS_TONE=i]` [CR .HB]. NHARMS takes one count per tone and
/// defaults every tone to INTMODMAX; INTMODMAX defaults to the largest
/// NHARMS. One of them is required. SUBHARMS=s puts the lowest tone f's
/// lines at multiples of f/s up to its NHARMS·f; that tone becomes `f0`,
/// the first of the list. SS_TONE=i (from 1) takes tone
/// i out of the large-signal spectrum: it is the small-signal tone, whose
/// frequency the `.hbac`, `.hbxf` or `.hbnoise` sweep supplies.
fn hbTones(arena: std.mem.Allocator, args: []const Value) !requests.Hb {
    const Key = enum { tones, nharms, intmodmax, subharms, ss_tone };
    var tones: std.ArrayList(f64) = .empty;
    var nharms: std.ArrayList(u16) = .empty;
    var single: [3]?u16 = @splat(null); // INTMODMAX, SUBHARMS, SS_TONE
    var key: ?Key = null;
    for (args, 0..) |v, i| switch (v) {
        .name => {
            var lower: [16]u8 = undefined;
            key = std.meta.stringToEnum(Key, try analysis_args.keyword(args, i, &lower)) orelse return error.InvalidAnalysisArguments;
        },
        .num => switch (key orelse return error.InvalidAnalysisArguments) {
            .tones => try tones.append(arena, try analysis_args.positive(args, i)),
            .nharms => try nharms.append(arena, try analysis_args.count(u16, args, i, 0)),
            inline .intmodmax, .subharms, .ss_tone => |k| {
                const slot = &single[@backingInt(k) - @backingInt(Key.intmodmax)];
                if (slot.* != null) return error.InvalidAnalysisArguments;
                slot.* = try analysis_args.count(u16, args, i, 0);
            },
        },
        else => return error.InvalidAnalysisArguments,
    };
    const intmod, const sub, const ss = single;
    const n = tones.items.len;
    if (n == 0 or (nharms.items.len != 0 and nharms.items.len != n)) return error.InvalidAnalysisArguments;
    if (nharms.items.len == 0) {
        const h = intmod orelse return error.InvalidAnalysisArguments;
        try nharms.appendNTimes(arena, h, n);
    }
    if (ss) |i| {
        if (i > n or n == 1) return error.InvalidAnalysisArguments;
        _ = tones.orderedRemove(i - 1);
        _ = nharms.orderedRemove(i - 1);
    }
    const s = sub orelse 1;
    if (s != 1) {
        const lo = std.mem.indexOfMin(f64, tones.items);
        std.mem.swap(f64, &tones.items[0], &tones.items[lo]);
        std.mem.swap(u16, &nharms.items[0], &nharms.items[lo]);
    }
    return .{
        .f0 = tones.items[0] / @as(f64, @floatFromInt(s)),
        .n_harmonics = std.math.mul(u16, nharms.items[0], s) catch return error.InvalidAnalysisArguments,
        .extra_tones = tones.items[1..],
        .extra_harmonics = nharms.items[1..],
        .intmodmax = intmod orelse std.mem.max(u16, nharms.items),
        .subharms = s,
        .phasors = true,
    };
}

/// HSPICE `.sn TRES= PERIOD=` or `.sn TONE= NHARMS=` [CR .SN] as `.pss`:
/// the period, and PERIOD/TRES steps when TRES is given. TRINIT,
/// MAXTRINITCYCLES and NUMPEROUT are read and unused: `.pss` shoots from
/// the operating point and publishes one period.
fn shootingNewton(args: []const Value) !requests.Pss {
    const Key = enum { tres, period, tone, nharms, trinit, maxtrinitcycles, numperout };
    var given: [@typeInfo(Key).@"enum".field_names.len]?f64 = @splat(null);
    var i: usize = 0;
    while (i < args.len) : (i += 2) {
        var lower: [16]u8 = undefined;
        const key = std.meta.stringToEnum(Key, try analysis_args.keyword(args, i, &lower)) orelse return error.InvalidAnalysisArguments;
        given[@backingInt(key)] = try analysis_args.positive(args, i + 1);
    }
    const period = given[@backingInt(Key.period)] orelse
        1 / (given[@backingInt(Key.tone)] orelse return error.InvalidAnalysisArguments);
    var pss: requests.Pss = .{ .period = period };
    if (given[@backingInt(Key.tres)]) |tres| {
        const steps = @round(period / tres);
        if (!(steps >= 1) or steps > std.math.maxInt(u32) - 1) return error.InvalidAnalysisArguments;
        pss.n_samples = @intFromFloat(steps);
    }
    return pss;
}

/// A comptime argument list: strings become names, numbers numbers.
fn argv(comptime xs: anytype) []const Value {
    const final = comptime blk: {
        var out: [xs.len]Value = undefined;
        for (xs, 0..) |x, k| out[k] = if (@typeInfo(@TypeOf(x)) == .pointer) .{ .name = x } else .{ .num = x };
        break :blk out;
    };
    return &final;
}

test hbTones {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const two = try hbTones(a, argv(.{ "tones", 1e3, 3e3, "nharms", 4, 2 }));
    try testing.expectEqual(@as(f64, 1e3), two.f0);
    try testing.expectEqual(@as(u16, 4), two.n_harmonics);
    try testing.expectEqualSlices(f64, &.{3e3}, two.extra_tones);
    try testing.expectEqual(@as(u16, 4), two.intmodmax);
    // INTMODMAX alone fills NHARMS; SS_TONE takes its tone out.
    const ss = try hbTones(a, argv(.{ "tones", 1e3, 3e3, "intmodmax", 5, "ss_tone", 2 }));
    try testing.expectEqual(@as(u16, 5), ss.n_harmonics);
    try testing.expectEqual(@as(usize, 0), ss.extra_tones.len);
    // SUBHARMS puts the lowest tone first and divides it.
    const sub = try hbTones(a, argv(.{ "tones", 3e3, 1e3, "nharms", 2, 3, "subharms", 2 }));
    try testing.expectEqual(@as(f64, 500), sub.f0);
    try testing.expectEqual(@as(u16, 6), sub.n_harmonics);
    try testing.expectEqualSlices(f64, &.{3e3}, sub.extra_tones);
    for ([_][]const Value{
        argv(.{ "tones", 1e3 }), // neither NHARMS nor INTMODMAX
        argv(.{ "tones", 1e3, 2e3, "nharms", 3 }), // one count for two tones
        argv(.{ "tones", 1e3, "intmodmax", 3, "ss_tone", 1 }), // no large-signal tone left
        argv(.{ "tones", 1e3, "intmodmax", 3, "intmodmax", 4 }),
        argv(.{ "tones", 1e3, "nharms", 40000, "subharms", 2 }), // harmonics past u16
        argv(.{ 1e3, "tones" }),
        argv(.{ "tone", 1e3 }),
    }) |args| try testing.expectError(error.InvalidAnalysisArguments, hbTones(a, args));
}

test shootingNewton {
    try testing.expectEqual(@as(f64, 1e-3), (try shootingNewton(argv(.{ "tone", 1e3, "nharms", 8 }))).period);
    try testing.expectEqual(@as(u32, 2000), (try shootingNewton(argv(.{ "period", 2e-6, "tres", 1e-9 }))).n_samples);
    try testing.expectError(error.InvalidAnalysisArguments, shootingNewton(argv(.{ "period", 1e-9, "tres", 1e-8 })));
    try testing.expectError(error.InvalidAnalysisArguments, shootingNewton(argv(.{ "tres", 1e-9 })));
    try testing.expectError(error.InvalidAnalysisArguments, shootingNewton(argv(.{ "period", -1 })));
}
