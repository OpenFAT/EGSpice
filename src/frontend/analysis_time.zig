//! Time-domain and DC cards to queries. In: one `.op`, `.tran`,
//! `.trannoise`, `.matex`, `.dc`, `.four`, `.envelope`, `.mc` or `.temp`
//! card. Out: its query.
//! Cites: ngspice dctran.c, traninit.c, dcop.c; HSPICE [CR] .OP, .TRAN, .DC.
//!
//! Contents:
//! - `pub fn dcTarget(args: []const Value, i: usize, cards: []const requests.CardRef) !requests.Dc.SweepTarget`
//! - `pub fn ptdNoise(ctx: analysis_args.CardContext, args: []const Value, pos: u32, neg: u32) !requests.Pnoise`
//! - `pub fn ptdTimes(arena: std.mem.Allocator, args: []const Value, k: *usize) ![]const f64`
//! - `pub fn tranNoiseSamples(args: []const Value) u32`
//! - `pub fn fft(ctx: analysis_args.CardContext, args: []const Value, pos: u32, neg: u32) !requests.Fft`
//! - `pub fn opJob(a: netlist.Analysis) !Job`
//! - `pub fn tranJob(a: netlist.Analysis, ctx: analysis_args.CardContext) !Job`
//! - `pub fn dcJob(a: netlist.Analysis, cards: []const requests.CardRef, ctx: analysis_args.CardContext) !Job`
//! - `pub fn fourJob(a: netlist.Analysis, ctx: analysis_args.CardContext) !Job`
//! - `pub fn envelopeJob(a: netlist.Analysis) !Job`
//! - `pub fn mcJob(a: netlist.Analysis) !Job`
//! - `pub fn tempJob(a: netlist.Analysis) !?Job`

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

/// One `.dc` axis, read from `args[i.*..]` and `i` advanced past it:
/// `start stop incr`, HSPICE `START= STOP= STEP=`, `LIN|DEC|OCT np start
/// stop` or `POI np v1 ... vn` [CR .DC]. The typed grids become explicit
/// `points`, allocated in `arena`.
const Axis = struct { start: f64 = 0, stop: f64 = 0, step: f64 = 1, points: []const f64 = &.{} };

/// The swept quantity of `.dc <card|TEMP> start stop step`, as the
/// (device type, instance index, parameter) key `ParamRef` uses. Card names
/// match case-insensitively, first card first. `InvalidAnalysisArguments`
/// when `args[i]` is no name, `AnalysisSourceNotFound` when no card has it.
pub fn dcTarget(args: []const Value, i: usize, cards: []const requests.CardRef) !requests.Dc.SweepTarget {
    const name = nameAt(args, i) orelse return error.InvalidAnalysisArguments;
    if (std.ascii.eqlIgnoreCase(name, "temp")) return .temp;
    for (cards) |c| {
        if (!std.ascii.eqlIgnoreCase(c.name, name)) continue;
        // ngspice sweeps a card's primary value: `dc` on a source, the
        // element value on a passive.
        const param: []const u8 = switch (c.type) {
            Library.builtin("resistor") => "r",
            Library.builtin("capacitor") => "c",
            Library.builtin("inductor") => "l",
            else => "dc",
        };
        return .{ .device = .{ .type = c.type, .index = c.index, .param_name = param } };
    }
    return error.AnalysisSourceNotFound;
}

/// HSPICE `.ptdnoise v(out) TIME=t|<sweep> [TDELTA=dt] <sweep> [LISTFREQ=
/// LISTCOUNT= LISTFLOOR= LISTSOURCES=]` [CR .PTDNOISE] at the `.sn`
/// fundamental: the noise density at time t of the period, with `strobe`
/// set to the first time (`ptdTimes` lists them all). A `.meas` name for
/// TIME is `UnsupportedAnalysisOutput`; TDELTA only feeds HSPICE's
/// strobed-jitter measure and the LIST keywords its listing, so they are
/// checked and unused.
pub fn ptdNoise(ctx: analysis_args.CardContext, args: []const Value, pos: u32, neg: u32) !requests.Pnoise {
    const f0 = ctx.sn_f0 orelse return error.MissingAnalysisCard;
    var lower: [16]u8 = undefined;
    var k: usize = 2;
    const t = (try ptdTimes(ctx.arena, args, &k))[0];
    if (k < args.len and args[k] == .name and std.mem.eql(u8, try analysis_args.keyword(args, k, &lower), "tdelta")) {
        _ = try analysis_args.positive(args, k + 1);
        k += 2;
    }
    const grid = try analysis_args.acGrid(ctx.arena, args, k);
    const lists = std.StaticStringMap(void).initComptime(.{ .{"listfreq"}, .{"listcount"}, .{"listfloor"}, .{"listsources"} });
    var i = grid.end;
    while (i < args.len) : (i += 2) if (!lists.has(try analysis_args.keyword(args, i, &lower))) return error.InvalidAnalysisArguments;
    return .{ .out_node = try analysis_args.outputNode(pos), .out_neg = try analysis_args.outputNeg(neg), .sweep = grid.sweep, .f_fundamental = f0, .strobe = t };
}

/// The TIME values of a `.ptdnoise` card, one number or a LIN/DEC/OCT/POI
/// sweep starting at `args[k.*]`, `k` advanced past them. Allocated in
/// `arena`; never empty.
pub fn ptdTimes(arena: std.mem.Allocator, args: []const Value, k: *usize) ![]const f64 {
    var lower: [16]u8 = undefined;
    if (args.len < 3 or args[0] != .group or !std.mem.eql(u8, try analysis_args.keyword(args, 1, &lower), "time")) return error.InvalidAnalysisArguments;
    const times = if (args[2] == .num) blk: {
        k.* = 3;
        break :blk try arena.dupe(f64, &.{try analysis_args.number(args, 2)});
    } else blk: {
        const sweeps = std.StaticStringMap(void).initComptime(.{ .{"lin"}, .{"dec"}, .{"oct"}, .{"poi"} });
        if (!sweeps.has(try analysis_args.keyword(args, 2, &lower))) return error.UnsupportedAnalysisOutput;
        k.* = 2;
        break :blk (try dcAxis(arena, args, k)).points;
    };
    for (times) |t| if (!(t >= 0)) return error.InvalidAnalysisArguments;
    return times;
}

/// SAMPLES= of an HSPICE `.trannoise` card, 1 when absent or malformed
/// (`hspiceTranNoise` reports the malformed card).
pub fn tranNoiseSamples(args: []const Value) u32 {
    var i: usize = 1;
    while (i + 1 < args.len) : (i += 2) {
        var lower: [16]u8 = undefined;
        if (std.mem.eql(u8, analysis_args.keyword(args, i, &lower) catch return 1, "samples")) return analysis_args.count(u32, args, i + 1, 1) catch 1;
    }
    return 1;
}

/// HSPICE `.fft v(a[,b]) [START=|FROM=] [STOP=|TO=] [NP=] [FORMAT=NORM|UNORM]
/// [WINDOW=] [ALFA=] [FREQ=] [FMIN=] [FMAX=]` [CR .FFT] over the deck's
/// `.tran`, whose window START and STOP default to. NP rounds up to a power
/// of two. FREQ, FMIN and FMAX only shape HSPICE's printed listing; they are
/// checked and not stored.
pub fn fft(ctx: analysis_args.CardContext, args: []const Value, pos: u32, neg: u32) !requests.Fft {
    const tran = ctx.tran orelse return error.MissingAnalysisCard;
    if (args.len == 0) return error.InvalidAnalysisArguments;
    if (analysis_args.currentProbeName(args, 0) != null) return error.UnsupportedAnalysisOutput;
    var o: requests.Fft = .{
        .tran = tran,
        .out_pos = try analysis_args.outputNode(pos),
        .out_neg = try analysis_args.outputNeg(neg),
        .start = tran.t_start,
        .stop = tran.t_stop,
        .label = try analysis_args.outputLabel(ctx.arena, args[0]),
    };
    const Key = enum { start, from, stop, to, np, format, window, alfa, freq, fmin, fmax };
    var i: usize = 1;
    while (i < args.len) : (i += 2) {
        var lower: [8]u8 = undefined;
        const key = std.meta.stringToEnum(Key, try analysis_args.keyword(args, i, &lower)) orelse return error.InvalidAnalysisArguments;
        switch (key) {
            .start, .from => o.start = try analysis_args.number(args, i + 1),
            .stop, .to => o.stop = try analysis_args.number(args, i + 1),
            .np => {
                const np = try analysis_args.positive(args, i + 1);
                if (np > 1 << 27) return error.InvalidAnalysisArguments;
                o.np = std.math.ceilPowerOfTwoAssert(u32, @intFromFloat(@max(@ceil(np), 4)));
            },
            .format => {
                const word = try analysis_args.keyword(args, i + 1, &lower);
                o.normalized = if (std.mem.eql(u8, word, "norm")) true else if (std.mem.eql(u8, word, "unorm")) false else return error.InvalidAnalysisArguments;
            },
            .window => o.window = std.meta.stringToEnum(requests.Fft.Window, try analysis_args.keyword(args, i + 1, &lower)) orelse return error.InvalidAnalysisArguments,
            .alfa => {
                o.alfa = try analysis_args.number(args, i + 1);
                if (o.alfa < 1 or o.alfa > 20) return error.InvalidAnalysisArguments;
            },
            .freq, .fmin, .fmax => if (try analysis_args.number(args, i + 1) < 0) return error.InvalidAnalysisArguments,
        }
    }
    if (o.start < tran.t_start or o.stop > tran.t_stop or !(o.stop > o.start)) return error.InvalidAnalysisArguments;
    return o;
}

/// The `op` card.
pub fn opJob(a: netlist.Analysis) !Job {
    const args = a.args;
    // HSPICE `.op [format] [time ...]` [CR .OP]; `queries` turns the
    // times into snapshots.
    const formats = std.StaticStringMap(void).initComptime(.{
        .{"all"}, .{"brief"}, .{"current"}, .{"debug"}, .{"none"}, .{"voltage"},
    });
    var lower: [8]u8 = undefined;
    for (args, 0..) |v, i| switch (v) {
        .num => |t| if (!(t >= 0) or !std.math.isFinite(t)) return error.InvalidAnalysisArguments,
        .name => if (!formats.has(try analysis_args.keyword(args, i, &lower))) return error.InvalidAnalysisArguments,
        else => return error.InvalidAnalysisArguments,
    };
    return .{ .op = .{} };
}

/// The `tran`, `tran_noise`, `matex` cards.
pub fn tranJob(a: netlist.Analysis, ctx: analysis_args.CardContext) !Job {
    const args = a.args;
    const id = a.kind;
    const node_id = a.pos;
    const node_neg = a.neg;
    if (id == .tran and a.dialect == .hspice) return .{ .tran = try hspiceTran(args) };
    if (id == .tran_noise and args.len > 0 and args[0] != .num) return .{ .tran_noise = try hspiceTranNoise(ctx, args, node_id, node_neg) };
    try analysis_args.arity(args, 2, if (id == .tran) 5 else 2);
    const step = try analysis_args.positive(args, 0);
    const stop = try analysis_args.positive(args, 1);
    if (id == .matex) return .{ .matex = .{ .t_stop = stop, .h_output_cap = step } };
    if (id == .tran_noise) return .{ .tran_noise = .{ .t_stop = stop, .dt_init = step, .dt_max = step } };
    var numeric_end = args.len;
    const uic = analysis_args.hasUic(args);
    if (uic) numeric_end -= 1;
    if (numeric_end > 4) return error.InvalidAnalysisArguments;
    // ngspice tstart: suppresses OUTPUT before it, never the solve.
    const t_start = if (numeric_end > 2) try analysis_args.number(args, 2) else 0;
    if (!(t_start >= 0) or t_start >= stop) return error.InvalidAnalysisArguments;
    return .{ .tran = .{ .t_stop = stop, .dt_init = step, .t_start = t_start, .dt_max = if (numeric_end > 3) try analysis_args.positive(args, 3) else null, .uic = uic } };
}

/// The `dc` card.
pub fn dcJob(a: netlist.Analysis, cards: []const requests.CardRef, ctx: analysis_args.CardContext) !Job {
    const args = a.args;
    var opts: requests.Dc = .{ .target = try dcTarget(args, 0, cards) };
    var i: usize = 1;
    const inner = try dcAxis(ctx.arena, args, &i);
    opts.start = inner.start;
    opts.stop = inner.stop;
    opts.step = inner.step;
    opts.points = inner.points;
    if (i < args.len) {
        opts.target2 = try dcTarget(args, i, cards);
        i += 1;
        const outer = try dcAxis(ctx.arena, args, &i);
        opts.start2 = outer.start;
        opts.stop2 = outer.stop;
        opts.step2 = outer.step;
        opts.points2 = outer.points;
    }
    if (i != args.len) return error.InvalidAnalysisArguments;
    return .{ .dc = opts };
}

/// The `four` card.
pub fn fourJob(a: netlist.Analysis, ctx: analysis_args.CardContext) !Job {
    const args = a.args;
    const node_id = a.pos;
    const node_neg = a.neg;
    try analysis_args.arity(args, 2, 3);
    return .{ .four = .{
        .f_fundamental = try analysis_args.positive(args, 0),
        .output_node = try analysis_args.outputNode(node_id),
        .output_neg = try analysis_args.outputNeg(node_neg),
        .n_harmonics = try analysis_args.count(u16, args, 2, 9),
        .label = if (a.split) try analysis_args.outputLabel(ctx.arena, args[1]) else "",
        .tran_opts = ctx.tran,
    } };
}

/// The `envelope` card.
pub fn envelopeJob(a: netlist.Analysis) !Job {
    const args = a.args;
    try analysis_args.arity(args, 2, 2);
    return .{ .envelope = .{ .t_carrier = try analysis_args.positive(args, 0), .t_stop = try analysis_args.positive(args, 1) } };
}

/// The `mc` card.
pub fn mcJob(a: netlist.Analysis) !Job {
    const args = a.args;
    try analysis_args.arity(args, 1, 2);
    var opts: requests.Mc = .{ .n_trials = try analysis_args.count(u16, args, 0, 100) };
    if (args.len == 2) opts.variation = try analysis_args.number(args, 1);
    if (opts.variation < 0) return error.InvalidAnalysisArguments;
    return .{ .mc = opts };
}

/// The `temp` card.
pub fn tempJob(a: netlist.Analysis) !?Job {
    const args = a.args;
    // A single-temperature card is deck configuration.
    if (args.len == 1) {
        if (try analysis_args.number(args, 0) <= -273.15) return error.InvalidAnalysisArguments;
        return null;
    }
    try analysis_args.arity(args, 3, 3);
    const opts: requests.Temp = .{ .t_start = try analysis_args.number(args, 0), .t_stop = try analysis_args.number(args, 1), .t_step = try analysis_args.number(args, 2) };
    try analysis_args.checkStep(opts.t_start, opts.t_stop, opts.t_step);
    if (opts.t_step <= 0 or (opts.t_stop - opts.t_start) / opts.t_step >= std.math.maxInt(u32) or
        @min(opts.t_start, opts.t_stop) <= -273.15) return error.InvalidAnalysisArguments;
    return .{ .temp = opts };
}

// ---- private ----

fn dcAxis(arena: std.mem.Allocator, args: []const Value, i: *usize) !Axis {
    const Word = enum { lin, dec, oct, poi, start, stop, step };
    const words = std.StaticStringMap(Word).initComptime(.{
        .{ "lin", .lin },     .{ "dec", .dec },   .{ "oct", .oct },   .{ "poi", .poi },
        .{ "start", .start }, .{ "stop", .stop }, .{ "step", .step },
    });
    const wordAt = struct {
        fn f(a: []const Value, at: usize) ?Word {
            var lower: [8]u8 = undefined;
            const name = nameAt(a, at) orelse return null;
            if (name.len > lower.len) return null;
            return words.get(std.ascii.lowerString(lower[0..name.len], name));
        }
    }.f;
    var axis: Axis = .{};
    switch (wordAt(args, i.*) orelse {
        axis = .{ .start = try analysis_args.number(args, i.*), .stop = try analysis_args.number(args, i.* + 1), .step = try analysis_args.number(args, i.* + 2) };
        try analysis_args.checkStep(axis.start, axis.stop, axis.step);
        i.* += 3;
        return axis;
    }) {
        .start, .stop, .step => {
            var seen: u3 = 0;
            while (wordAt(args, i.*)) |key| : (i.* += 2) {
                const v = try analysis_args.number(args, i.* + 1);
                switch (key) {
                    .start => axis.start = v,
                    .stop => axis.stop = v,
                    .step => axis.step = v,
                    else => return error.InvalidAnalysisArguments,
                }
                seen |= @as(u3, 1) << @intCast(@backingInt(key) - @backingInt(Word.start));
            }
            if (seen != 0b111) return error.InvalidAnalysisArguments;
            try analysis_args.checkStep(axis.start, axis.stop, axis.step);
        },
        .poi => {
            const n = try analysis_args.count(u32, args, i.* + 1, 0);
            if (n == 0 or args.len < i.* + 2 + n) return error.InvalidAnalysisArguments;
            const points = try arena.alloc(f64, n);
            for (points, 0..) |*p, k| p.* = try analysis_args.number(args, i.* + 2 + k);
            i.* += 2 + n;
            axis.points = points;
        },
        .lin, .dec, .oct => |kind| {
            const grid: numerics.FreqSweep = .{
                .kind = if (kind == .lin) .lin else if (kind == .dec) .dec else .oct,
                .points = try analysis_args.count(u32, args, i.* + 1, 0),
                .f_start = try analysis_args.number(args, i.* + 2),
                .f_stop = try analysis_args.number(args, i.* + 3),
            };
            if (grid.points == 0) return error.InvalidAnalysisArguments;
            if (kind != .lin and !(grid.f_start > 0 and grid.f_stop >= grid.f_start)) return error.InvalidAnalysisArguments;
            // ponytail: a 10M-point ceiling keeps a typo from allocating the heap.
            // A finite f64 span is under 2100 octaves, so 1e6 points per
            // octave or decade keeps `count()` inside u32 before it is checked.
            if ((kind != .lin and grid.points > 1_000_000) or grid.count() > 10_000_000) return error.InvalidAnalysisArguments;
            const points = try arena.alloc(f64, grid.count());
            for (points, 0..) |*p, k| p.* = grid.at(@intCast(k));
            i.* += 4;
            axis.points = points;
        },
    }
    if (axis.points.len > 0) {
        axis.start = axis.points[0];
        axis.stop = axis.points[axis.points.len - 1];
    }
    return axis;
}

/// HSPICE `.tran tstep1 tstop1 [tstep2 tstop2 ...] [START=t] [UIC]` [CR
/// .TRAN]: one run to the last tstop. A double-point card whose tstep2 and
/// tstop2 are both below tstop1, with no START=, is the SPICE form
/// `tstep tstop tstart delmax`, as is a three-number card.
fn hspiceTran(args: []const Value) !requests.Tran {
    var nums: [32]f64 = undefined;
    var n: usize = 0;
    var start: ?f64 = null;
    var uic = false;
    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        var lower: [8]u8 = undefined;
        switch (args[i]) {
            .num => |v| {
                if (n == nums.len or !std.math.isFinite(v)) return error.InvalidAnalysisArguments;
                nums[n] = v;
                n += 1;
            },
            .name => {
                const word = try analysis_args.keyword(args, i, &lower);
                if (std.mem.eql(u8, word, "uic")) {
                    uic = true;
                } else if (std.mem.eql(u8, word, "start")) {
                    i += 1;
                    start = try analysis_args.number(args, i);
                } else return error.InvalidAnalysisArguments;
            },
            else => return error.InvalidAnalysisArguments,
        }
    }
    if (n < 2 or !(nums[0] > 0) or !(nums[1] > 0)) return error.InvalidAnalysisArguments;
    var t: requests.Tran = .{ .t_stop = nums[1], .dt_init = nums[0], .uic = uic, .dt_max = null };
    if (n == 3 or (n == 4 and start == null and nums[2] < nums[1] and nums[3] < nums[1])) {
        t.t_start = nums[2];
        if (n == 4) t.dt_max = if (nums[3] > 0) nums[3] else return error.InvalidAnalysisArguments;
    } else {
        if (n % 2 != 0) return error.InvalidAnalysisArguments;
        // ponytail: the finest tstep caps the whole run; a per-segment cap
        // would save steps in the coarse segments.
        var k: usize = 2;
        while (k < n) : (k += 2) {
            if (!(nums[k] > 0) or !(nums[k + 1] > nums[k - 1])) return error.InvalidAnalysisArguments;
            t.dt_init = @min(t.dt_init, nums[k]);
            t.t_stop = nums[k + 1];
        }
    }
    t.t_start = start orelse t.t_start;
    if (!(t.t_start >= 0) or t.t_start >= t.t_stop) return error.InvalidAnalysisArguments;
    return t;
}

/// HSPICE `.trannoise out [METHOD=MC|SDE] [SEED=] [SAMPLES=] [TIME=all|t]
/// [AUTOCORRELATION=] [FMIN=] [FMAX=] [SCALE=]` [CR .TRANNOISE] over the
/// deck's `.tran`, stepped at 1/(2 FMAX), the bandwidth that carries the
/// white noise up to FMAX (default 1/TSTEP). SEED is the Monte Carlo index
/// of the first run (default 2); `queries` turns the SAMPLES runs into one
/// query each and index 1 into the noiseless run. The output
/// names v(out) for SDE's `onoise`; a sampled run records every probe.
fn hspiceTranNoise(ctx: analysis_args.CardContext, args: []const Value, pos: u32, neg: u32) !requests.TranNoise {
    const tran = ctx.tran orelse return error.MissingAnalysisCard;
    if (args[0] != .group) return error.InvalidAnalysisArguments;
    var o: requests.TranNoise = .{ .tol = tran.tol, .t_stop = tran.t_stop, .seed = 2 };
    var f_max = 1 / tran.dt_init;
    const Key = enum { method, seed, samples, autocorrelation, fmin, fmax, scale, time };
    var i: usize = 1;
    while (i < args.len) : (i += 2) {
        var lower: [16]u8 = undefined;
        const key = std.meta.stringToEnum(Key, try analysis_args.keyword(args, i, &lower)) orelse return error.InvalidAnalysisArguments;
        if (i + 1 >= args.len) return error.InvalidAnalysisArguments;
        switch (key) {
            .method => {
                const m = try analysis_args.keyword(args, i + 1, &lower);
                if (std.mem.eql(u8, m, "sde")) o.sde = true else if (!std.mem.eql(u8, m, "mc")) return error.InvalidAnalysisArguments;
            },
            .seed => o.seed = try analysis_args.count(u32, args, i + 1, 1),
            .samples => _ = try analysis_args.count(u32, args, i + 1, 1),
            .autocorrelation => {},
            .fmin => o.f_min = try analysis_args.positive(args, i + 1),
            .fmax => f_max = try analysis_args.positive(args, i + 1),
            .scale => o.scale = try analysis_args.positive(args, i + 1),
            .time => if (args[i + 1] == .num) {
                o.t_break = try analysis_args.positive(args, i + 1);
            } else if (!std.mem.eql(u8, try analysis_args.keyword(args, i + 1, &lower), "all")) return error.InvalidAnalysisArguments,
        }
    }
    if (o.sde) {
        if (tranNoiseSamples(args) != 1) return error.InvalidAnalysisArguments;
        o.out_node = try analysis_args.outputNode(pos);
        o.out_neg = try analysis_args.outputNeg(neg);
    }
    o.dt_max = 0.5 / f_max;
    o.dt_init = o.dt_max;
    return o;
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

test dcAxis {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var i: usize = 0;
    const plain = try dcAxis(a, argv(.{ 0, 1, 0.5, "v2" }), &i);
    try testing.expectEqual(@as(usize, 3), i);
    try testing.expectEqual(@as(f64, 0.5), plain.step);
    i = 1;
    const keyed = try dcAxis(a, argv(.{ "v1", "STEP", 0.25, "start", 1, "stop", 2 }), &i);
    try testing.expectEqual(@as(usize, 7), i);
    try testing.expectEqual(@as(f64, 1), keyed.start);
    try testing.expectEqual(@as(f64, 0.25), keyed.step);
    i = 0;
    const poi = try dcAxis(a, argv(.{ "poi", 3, 5, -1, 2 }), &i);
    try testing.expectEqual(@as(usize, 5), i);
    try testing.expectEqual(@as(f64, 5), poi.start);
    try testing.expectEqual(@as(f64, 2), poi.stop);
    i = 0;
    const lin_axis = try dcAxis(a, argv(.{ "lin", 3, 1, 0 }), &i);
    try testing.expectEqualSlices(f64, &.{ 1, 0.5, 0 }, lin_axis.points);
    i = 0;
    const dec = try dcAxis(a, argv(.{ "dec", 1, 1, 100 }), &i);
    try testing.expectEqual(@as(usize, 3), dec.points.len);
    for ([_][]const Value{
        argv(.{ "start", 0, "stop", 1 }), // no STEP
        argv(.{ "poi", 4e9, 1 }), // more points than given
        argv(.{ "lin", 0, 0, 1 }),
        argv(.{ "dec", 1, 0, 100 }),
        argv(.{ "dec", 2e6, 1, 10 }), // past the per-decade cap
        argv(.{ "dec", 1e6, 1e-300, 1e300 }), // past the 10M point cap, without trapping in count()
        argv(.{ 0, 1 }),
    }) |args| {
        i = 0;
        try testing.expectError(error.InvalidAnalysisArguments, dcAxis(a, args, &i));
    }
}

test hspiceTran {
    const two = try hspiceTran(argv(.{ 1e-9, 1e-6 }));
    try testing.expectEqual(@as(f64, 1e-6), two.t_stop);
    try testing.expectEqual(@as(?f64, null), two.dt_max);
    // Three numbers, and four whose second pair sits below tstop1, are
    // the SPICE `tstep tstop tstart [delmax]`.
    try testing.expectEqual(@as(f64, 2e-7), (try hspiceTran(argv(.{ 1e-9, 1e-6, 2e-7 }))).t_start);
    const spice = try hspiceTran(argv(.{ 1e-9, 1e-6, 2e-7, 5e-9 }));
    try testing.expectEqual(@as(f64, 2e-7), spice.t_start);
    try testing.expectEqual(@as(?f64, 5e-9), spice.dt_max);
    // Segments: the finest tstep for the whole run, to the last tstop.
    const segs = try hspiceTran(argv(.{ 1e-9, 1e-6, 1e-10, 2e-6, "START", 1e-7, "uic" }));
    try testing.expectEqual(@as(f64, 1e-10), segs.dt_init);
    try testing.expectEqual(@as(f64, 2e-6), segs.t_stop);
    try testing.expectEqual(@as(f64, 1e-7), segs.t_start);
    try testing.expect(segs.uic);
    for ([_][]const Value{
        argv(.{1e-9}),
        argv(.{ 0, 1e-6 }),
        argv(.{ 1e-9, 1e-6, 1e-9, 2e-6, 1e-9 }), // odd segment count
        argv(.{ 1e-9, 1e-6, 1e-9, 5e-7, "start", 0 }), // tstop2 before tstop1
        argv(.{ 1e-9, 1e-6, "start", 1e-6 }), // start at the stop
        argv(.{ 1e-9, 1e-6, "start" }),
        argv(.{ 1e-9, 1e-6, "bogus" }),
    }) |args| try testing.expectError(error.InvalidAnalysisArguments, hspiceTran(args));
}

test tranNoiseSamples {
    const out: Value = .{ .group = .{ .items = argv(.{ "v", "o" }) } };
    try testing.expectEqual(@as(u32, 4), tranNoiseSamples(&[_]Value{ out, .{ .name = "seed" }, .{ .num = 3 }, .{ .name = "samples" }, .{ .num = 4 } }));
    try testing.expectEqual(@as(u32, 1), tranNoiseSamples(&[_]Value{ out, .{ .name = "samples" }, .{ .num = -4 } }));
    try testing.expectEqual(@as(u32, 1), tranNoiseSamples(&[_]Value{out}));
}
