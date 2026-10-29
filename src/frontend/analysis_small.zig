//! Small-signal cards to queries. In: one `.ac`, `.disto`, `.noise`,
//! `.tf`, `.sens`, `.dcmatch`, `.acmatch`, `.dcsens`, `.pz`, `.stb`,
//! `.lstb`, `.dcxf`, `.acxf` or `.dcinc` card. Out: its query.
//! Cites: ngspice distoan.c, noisean.c, cktdisto.c, pzan.c; HSPICE [CR]
//! .NOISE, .PZ, .ACMATCH, .DCSENS, .ACPHASENOISE; VACASK `acstb`.
//!
//! Contents:
//! - `pub fn xfSources(arena: std.mem.Allocator, sources: core.QueryBindings) ![]const requests.XfSource`
//! - `pub fn lstbJob(args: []const Value, sources: core.QueryBindings, ac: ?numerics.FreqSweep, local_gnd: u32) !requests.Lstb`
//! - `pub fn acJob(a: netlist.Analysis, ctx: analysis_args.CardContext) !Job`
//! - `pub fn distoJob(a: netlist.Analysis, sources: core.QueryBindings) !Job`
//! - `pub fn noiseJob(a: netlist.Analysis, sources: core.QueryBindings, ctx: analysis_args.CardContext) !Job`
//! - `pub fn tfJob(a: netlist.Analysis, sources: core.QueryBindings) !Job`
//! - `pub fn sensJob(a: netlist.Analysis, ctx: analysis_args.CardContext) !Job`
//! - `pub fn acmatchJob(a: netlist.Analysis, ctx: analysis_args.CardContext) !Job`
//! - `pub fn pzJob(a: netlist.Analysis, sources: core.QueryBindings) !Job`
//! - `pub fn stbJob(a: netlist.Analysis, sources: core.QueryBindings) !Job`
//! - `pub fn dcxfJob(a: netlist.Analysis, sources: core.QueryBindings) !Job`
//! - `pub fn acxfJob(a: netlist.Analysis, sources: core.QueryBindings) !Job`
//! - `pub fn dcincJob(a: netlist.Analysis) !Job`

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

/// Every independent source, V cards then I cards, in card order.
pub fn xfSources(arena: std.mem.Allocator, sources: core.QueryBindings) ![]const requests.XfSource {
    var out: std.ArrayList(requests.XfSource) = try .initCapacity(arena, sources.v_names.len + sources.i_names.len);
    for (sources.v_names, sources.v_branches) |name, br| out.appendAssumeCapacity(.{ .name = name, .branch = br });
    for (sources.i_names, sources.i_pos, sources.i_neg) |name, p, m| out.appendAssumeCapacity(.{ .name = name, .branch = null, .nodes = .{ p, m } });
    return out.items;
}

/// `.lstb mode=single|diff|comm vsource=v1[,v2] [localgnd=n] [dec|oct|lin
/// N f1 f2]` [CR .LSTB]: the first three letters of the mode count. Without
/// a sweep the card takes the deck's `.ac` frequencies, `ac`. `local_gnd`
/// is the `localgnd` row `netlist.resolve` found, GROUND without one.
pub fn lstbJob(args: []const Value, sources: core.QueryBindings, ac: ?numerics.FreqSweep, local_gnd: u32) !requests.Lstb {
    const Key = enum { mode, vsource, localgnd, dec, oct, lin };
    const keys = std.StaticStringMap(Key).initComptime(.{
        .{ "mode", .mode }, .{ "vsource", .vsource }, .{ "localgnd", .localgnd },
        .{ "dec", .dec },   .{ "oct", .oct },         .{ "lin", .lin },
    });
    if (local_gnd == analysis_args.NO_NODE) return error.AnalysisNodeNotFound;
    const modes = std.StaticStringMap(requests.Lstb.Mode).initComptime(.{
        .{ "sin", .single }, .{ "dif", .diff }, .{ "com", .comm },
    });
    var sweep = ac;
    var opts: requests.Lstb = .{ .sweep = undefined, .probes = undefined, .local_gnd = local_gnd };
    var n_probes: usize = 0;
    var buf: [16]u8 = undefined;
    var i: usize = 0;
    while (i < args.len) {
        const key = keys.get(try analysis_args.keyword(args, i, &buf)) orelse return error.InvalidAnalysisArguments;
        switch (key) {
            .mode => {
                const m = try analysis_args.keyword(args, i + 1, &buf);
                opts.mode = modes.get(m[0..@min(m.len, 3)]) orelse return error.InvalidAnalysisArguments;
                i += 2;
            },
            .localgnd => i += 2,
            .vsource => {
                i += 1;
                while (i < args.len) : (i += 1) {
                    if (keys.has(analysis_args.keyword(args, i, &buf) catch "")) break;
                    if (n_probes == 2) return error.InvalidAnalysisArguments;
                    const v = try analysis_args.voltageSource(args, i, sources);
                    opts.probes[n_probes] = .{ .p = sources.v_pos[v], .n = sources.v_neg[v], .branch = sources.v_branches[v] };
                    n_probes += 1;
                }
            },
            .dec, .oct, .lin => {
                try analysis_args.arity(args, i + 4, i + 4);
                sweep = try analysis_args.frequencySweep(args, i);
                i += 4;
            },
        }
    }
    if (n_probes != @as(usize, if (opts.mode == .single) 1 else 2)) return error.InvalidAnalysisArguments;
    if (n_probes == 1) opts.probes[1] = opts.probes[0];
    opts.sweep = sweep orelse return error.MissingAnalysisCard;
    return opts;
}

/// The `ac` card.
pub fn acJob(a: netlist.Analysis, ctx: analysis_args.CardContext) !Job {
    const args = a.args;
    const grid = try analysis_args.acGrid(ctx.arena, args, 0);
    if (grid.end != args.len) return error.InvalidAnalysisArguments;
    var sweep = grid.sweep;
    sweep.land_on_stop = true;
    return .{ .ac = .{ .sweep = sweep } };
}

/// The `disto` card.
pub fn distoJob(a: netlist.Analysis, sources: core.QueryBindings) !Job {
    const args = a.args;
    const node_id = a.pos;
    try analysis_args.arity(args, 4, 5);
    const grid = try analysis_args.frequencySweep(args, 0);
    var opts: requests.Disto = .{ .sweep = grid, .output_node = try analysis_args.outputNode(node_id) };
    // `f2overf1` asks for the intermodulation products; ngspice then
    // wants a DISTOF2 source (E_NOF2SRC, distoan.c:425).
    if (args.len == 5) {
        opts.f2_ratio = try analysis_args.positive(args, 4);
        for (sources.v_distof2, 0..) |d, i| {
            if (d[0] == 0) continue;
            opts.drive2_branch = sources.v_branches[i];
            opts.ac2_magnitude = d[0];
            opts.ac2_phase = d[1];
            break;
        } else return error.AnalysisSourceNotFound;
    }
    // ngspice cktdisto.c:100-117: the F1 drive is the card carrying
    // DISTOF1, not the first source, on that card's branch row
    // (`disto/bjt_ce`: Vcc comes first, DISTOF1 is on Vin).
    // ponytail: first such card only. ngspice sums every DISTOF1
    // source into one RHS; sum here when a deck has two.
    for (sources.v_branches, sources.v_distof1) |br, d| {
        if (d[0] == 0) continue;
        opts.drive_branch = br;
        opts.ac_magnitude = d[0];
        opts.ac_phase = d[1];
        break;
    }
    return .{ .disto = opts };
}

/// The `noise` card.
pub fn noiseJob(a: netlist.Analysis, sources: core.QueryBindings, ctx: analysis_args.CardContext) !Job {
    var args = a.args;
    const node_id = a.pos;
    const node_neg = a.neg;
    // ngspice: `v(out) [src] dec|oct|lin N f1 f2 [pts]`; HSPICE:
    // `v(out) src [inter]` over the `.ac` sweep [CR .NOISE]. A
    // nonzero pts or inter asks for the per-device contributions.
    // The input reference source does not drive the noise solve, but
    // a name no card carries is still an error, not a silent success.
    // HSPICE `.acphasenoise out in [interval] carrier= [LIST...=]`
    // [CR .ACPHASENOISE] is the same solve read as phase noise; the
    // carrier only scales HSPICE's jitter listing, unused here.
    const phase = std.ascii.startsWithIgnoreCase(a.line, ".acphasenoise");
    if (phase) {
        const keys = std.StaticStringMap(void).initComptime(.{ .{"carrier"}, .{"listfreq"}, .{"listcount"}, .{"listfloor"}, .{"listsources"} });
        var k: usize = 2;
        while (k < args.len and args[k] == .num) k += 1;
        var carrier = false;
        var kw: [16]u8 = undefined;
        var j = k;
        while (j < args.len) : (j += 2) {
            const key = try analysis_args.keyword(args, j, &kw);
            if (!keys.has(key) or j + 1 >= args.len) return error.InvalidAnalysisArguments;
            if (std.mem.eql(u8, key, "carrier")) {
                _ = try analysis_args.positive(args, j + 1);
                carrier = true;
            }
        }
        if (!carrier) return error.InvalidAnalysisArguments;
        args = args[0..k];
    }
    var i: usize = 1;
    var in_branch: ?u32 = null;
    const grid_words = std.StaticStringMap(void).initComptime(.{ .{"dec"}, .{"oct"}, .{"lin"}, .{"poi"} });
    var lower: [4]u8 = undefined;
    const isGrid = struct {
        fn f(words: anytype, xs: []const Value, at: usize, buf: *[4]u8) bool {
            const name = nameAt(xs, at) orelse return false;
            return name.len <= buf.len and words.has(std.ascii.lowerString(buf[0..name.len], name));
        }
    }.f;
    var in_nodes: ?[2]u32 = null;
    if (i < args.len and !isGrid(grid_words, args, i, &lower)) {
        // A V or an I card (ngspice noisean.c: VSRC or ISRC).
        const name = nameAt(args, i) orelse return error.InvalidAnalysisArguments;
        if (netlist.nameIndex(sources.v_names, name)) |v| {
            in_branch = sources.v_branches[v];
        } else if (netlist.nameIndex(sources.i_names, name)) |k| {
            in_nodes = .{ sources.i_pos[k], sources.i_neg[k] };
        } else return error.AnalysisSourceNotFound;
        i += 1;
    }
    var sweep: numerics.FreqSweep = undefined;
    if (i < args.len and isGrid(grid_words, args, i, &lower)) {
        const g = try analysis_args.acGrid(ctx.arena, args, i);
        sweep = g.sweep;
        i = g.end;
    } else sweep = ctx.ac orelse return error.MissingAnalysisCard;
    const contributions = i < args.len and try analysis_args.number(args, i) > 0;
    if (i < args.len) i += 1;
    if (i != args.len) return error.InvalidAnalysisArguments;
    return .{ .noise = .{
        .out_node = try analysis_args.outputNode(node_id),
        .out_neg = try analysis_args.outputNeg(node_neg),
        .in_branch = in_branch,
        .in_nodes = in_nodes,
        .sweep = sweep,
        .contributions = contributions and !phase,
        .card_names = ctx.card_names,
        .phase = phase,
    } };
}

/// The `tf` card.
pub fn tfJob(a: netlist.Analysis, sources: core.QueryBindings) !Job {
    const args = a.args;
    const node_id = a.pos;
    const node_neg = a.neg;
    try analysis_args.arity(args, 2, 2);
    var opts: requests.Tf = .{ .output_node = GROUND };
    // `.tf i(Vmeasure) ...` measures a branch current.
    if (analysis_args.currentProbeName(args, 0)) |probe| {
        opts.output_branch = sources.v_branches[netlist.nameIndex(sources.v_names, probe) orelse return error.AnalysisSourceNotFound];
    } else {
        opts.output_node = try analysis_args.outputNode(node_id);
        opts.output_neg = try analysis_args.outputNeg(node_neg);
    }
    // The drive is a V card (branch row) or an I card (node pair).
    const drive = nameAt(args, 1) orelse return error.InvalidAnalysisArguments;
    if (netlist.nameIndex(sources.v_names, drive)) |v| {
        opts.input_branch = sources.v_branches[v];
    } else if (netlist.nameIndex(sources.i_names, drive)) |i_idx| {
        opts.input_nodes = .{ sources.i_pos[i_idx], sources.i_neg[i_idx] };
    } else return error.AnalysisSourceNotFound;
    return .{ .tf = opts };
}

/// The `sens`, `dcmatch` cards.
pub fn sensJob(a: netlist.Analysis, ctx: analysis_args.CardContext) !Job {
    const args = a.args;
    const id = a.kind;
    const node_id = a.pos;
    const node_neg = a.neg;
    try analysis_args.arity(args, 1, 1);
    const node = try analysis_args.outputNode(node_id);
    const neg = try analysis_args.outputNeg(node_neg);
    if (id == .sens) return .{ .sens = .{ .output_node = node, .output_neg = neg, .card_names = ctx.card_names } };
    return .{ .dcmatch = .{ .output_node = node, .output_neg = neg, .variations = ctx.variations } };
}

/// The `acmatch`, `dcsens` cards.
pub fn acmatchJob(a: netlist.Analysis, ctx: analysis_args.CardContext) !Job {
    const args = a.args;
    const id = a.kind;
    const node_id = a.pos;
    const node_neg = a.neg;
    // HSPICE `.acmatch v(out) [THRESHOLD= FILE= INTERVAL=
    // VIRTUAL_SENSITIVITY= SENS_THRESHOLD=]` over the `.ac` sweep, and
    // `.dcsens v(out) [FILE= PERTURBATION= INTERVAL= THRESHOLD=
    // GROUPBYDEVICE=]` [CR .ACMATCH, .DCSENS]. Every group is
    // published, so the listing keywords are checked and unused;
    // the derivative is the adjoint one, so PERTURBATION is too.
    if (args.len == 0 or args.len % 2 != 1) return error.InvalidAnalysisArguments;
    const keys = std.StaticStringMap(void).initComptime(.{
        .{"threshold"},      .{"file"},         .{"interval"},      .{"virtual_sensitivity"}, .{"virtual_sens"},
        .{"sens_threshold"}, .{"perturbation"}, .{"groupbydevice"},
    });
    var i: usize = 1;
    while (i < args.len) : (i += 2) {
        var lower: [24]u8 = undefined;
        if (!keys.has(try analysis_args.keyword(args, i, &lower))) return error.InvalidAnalysisArguments;
    }
    const node = try analysis_args.outputNode(node_id);
    const neg = try analysis_args.outputNeg(node_neg);
    if (id == .dcsens) return .{ .dcsens = .{ .output_node = node, .output_neg = neg, .variations = ctx.variations } };
    return .{ .acmatch = .{ .sweep = ctx.ac orelse return error.MissingAnalysisCard, .output_node = node, .output_neg = neg, .variations = ctx.variations } };
}

/// The `pz` card.
pub fn pzJob(a: netlist.Analysis, sources: core.QueryBindings) !Job {
    const args = a.args;
    const node_id = a.pos;
    const node_neg = a.neg;
    const ports = a.ports;
    // Bare `.pz` asks for the circuit's own poles and names no transfer.
    if (args.len == 0) return .{ .pz = .{} };
    // HSPICE `.pz v(out[,ref]) src` [CR .PZ]: the input is the
    // source's own node pair, `vol` for a V card and `cur` for an I
    // card; poles and zeros both.
    if (args.len == 2) {
        if (analysis_args.currentProbeName(args, 0) != null) return error.UnsupportedAnalysisOutput;
        var opts: requests.Pz = .{ .out_pos = try analysis_args.outputNode(node_id), .out_neg = try analysis_args.outputNeg(node_neg), .want = .both };
        const name = nameAt(args, 1) orelse return error.InvalidAnalysisArguments;
        if (netlist.nameIndex(sources.v_names, name)) |v| {
            opts.in_pos = sources.v_pos[v];
            opts.in_neg = sources.v_neg[v];
            opts.drive_branch = sources.v_branches[v];
        } else if (netlist.nameIndex(sources.i_names, name)) |v| {
            opts.in_pos = sources.i_pos[v];
            opts.in_neg = sources.i_neg[v];
        } else return error.AnalysisSourceNotFound;
        // The drive's node pair is unordered for poles and zeros;
        // keep the grounded side second, as the SPICE form must.
        if (opts.in_pos == GROUND) std.mem.swap(u32, &opts.in_pos, &opts.in_neg);
        _ = try analysis_args.outputNode(opts.in_pos);
        return .{ .pz = opts };
    }
    try analysis_args.arity(args, 6, 6);
    const drive_kind = std.StaticStringMap(enum { vol, cur })
        .initComptime(.{ .{ "vol", .vol }, .{ "cur", .cur } });
    const wanted = std.StaticStringMap(requests.Pz.Want)
        .initComptime(.{ .{ "pol", .poles }, .{ "zer", .zeros }, .{ "pz", .both } });
    var lower: [4]u8 = undefined;
    const kind = drive_kind.get(try analysis_args.keyword(args, 4, &lower)) orelse return error.InvalidAnalysisArguments;
    const want = wanted.get(try analysis_args.keyword(args, 5, &lower)) orelse return error.InvalidAnalysisArguments;
    var opts: requests.Pz = .{
        .in_pos = try analysis_args.outputNode(ports[0]),
        .in_neg = try analysis_args.outputNeg(ports[1]),
        .out_pos = try analysis_args.outputNode(ports[2]),
        .out_neg = try analysis_args.outputNeg(ports[3]),
        .want = want,
    };
    // A `vol` transfer is driven by the voltage source the deck already
    // put across the input port: its branch row is the column the
    // numerator's Cramer rule needs, and its presence in the nulled
    // matrix is the short the denominator needs. `cur` needs no card:
    // a current injection is just the node pair.
    if (kind == .vol) {
        opts.drive_branch = for (sources.v_pos, sources.v_neg, sources.v_branches) |p, m, br| {
            if ((p == opts.in_pos and m == opts.in_neg) or (p == opts.in_neg and m == opts.in_pos)) break br;
        } else return error.AnalysisSourceNotFound;
    }
    return .{ .pz = opts };
}

/// The `stb` card.
pub fn stbJob(a: netlist.Analysis, sources: core.QueryBindings) !Job {
    const args = a.args;
    // `.stb Vprobe dec N fstart fstop`: the named 0 V source is the
    // loop break; the sweep drives its branch row directly.
    try analysis_args.arity(args, 5, 5);
    const probe = try analysis_args.voltageSource(args, 0, sources);
    return .{ .stb = .{
        .sweep = try analysis_args.frequencySweep(args, 1),
        .probe_p = sources.v_pos[probe],
        .probe_n = sources.v_neg[probe],
        .probe_branch = sources.v_branches[probe],
    } };
}

/// The `dcxf` card.
pub fn dcxfJob(a: netlist.Analysis, sources: core.QueryBindings) !Job {
    const args = a.args;
    const node_id = a.pos;
    const node_neg = a.neg;
    try analysis_args.arity(args, 1, 2);
    return .{ .dcxf = try xfOutput(requests.Dcxf, args, node_id, node_neg, sources) };
}

/// The `acxf` card.
pub fn acxfJob(a: netlist.Analysis, sources: core.QueryBindings) !Job {
    const args = a.args;
    const node_id = a.pos;
    const node_neg = a.neg;
    try analysis_args.arity(args, 5, 6);
    return .{ .acxf = try xfOutput(requests.Acxf, args, node_id, node_neg, sources) };
}

/// The `dcinc` card.
pub fn dcincJob(a: netlist.Analysis) !Job {
    const args = a.args;
    try analysis_args.arity(args, 0, 0);
    return .{ .dcinc = .{} };
}

// ---- private ----

/// The `v(out)`, `v(a,b)` or `i(Vmeasure)` output of `.dcxf`/`.acxf`, and
/// whether a trailing `tf` asks for transfer functions only.
fn xfOutput(comptime T: type, args: []const Value, node: u32, neg: u32, sources: core.QueryBindings) !T {
    var opts: T = if (T == requests.Acxf)
        .{ .sweep = try analysis_args.frequencySweep(args, 1), .output_node = GROUND, .sources = &.{} }
    else
        .{ .output_node = GROUND, .sources = &.{} };
    if (analysis_args.currentProbeName(args, 0)) |probe| {
        opts.output_branch = sources.v_branches[netlist.nameIndex(sources.v_names, probe) orelse return error.AnalysisSourceNotFound];
    } else {
        opts.output_node = try analysis_args.outputNode(node);
        opts.output_neg = try analysis_args.outputNeg(neg);
    }
    const last = args.len - 1;
    if (last > 0) if (nameAt(args, last)) |word| {
        if (!std.ascii.eqlIgnoreCase(word, "tf")) return error.InvalidAnalysisArguments;
        opts.tf_only = true;
    };
    return opts;
}
