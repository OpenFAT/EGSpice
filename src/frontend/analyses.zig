//! Analysis cards and deck options to resolved queries. In: the deck's
//! analysis and config cards, nets already circuit rows, and the
//! construction bindings source and card names resolve against. Out: the
//! query list, fanned out over the deck's variants and `.temp` list. Each
//! card is read by analysis_card.zig; options by analysis_options.zig.
//! Cites: ngspice cktsopt.c, cktntask.c, traninit.c, dctran.c, distoan.c,
//! noisean.c, spiceif.c, inpgval.c; HSPICE [CR, SA, RF], at the code that
//! follows each.
//!
//! Contents:
//! - `pub const NO_NODE`, `CardContext` (analysis_args.zig)
//! - `pub const DeckOptions`, `deckOptions`, `applyDeckOptions` (analysis_options.zig)
//! - `pub const buildJob` (analysis_card.zig), `dcTarget` (analysis_time.zig)
//! - `pub const Fanout = struct`
//! - `pub const Span = struct`
//! - `pub fn queries(arena: Allocator, cards: []const netlist.Analysis, n_rows: u32, appended: bool, sources: core.QueryBindings, card_refs: []const requests.CardRef, deck_opts: DeckOptions, fanout: Fanout) ![]const Job`

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
const analysis_options = @import("analysis_options.zig");
const analysis_time = @import("analysis_time.zig");
const analysis_small = @import("analysis_small.zig");
const analysis_rf = @import("analysis_rf.zig");
const analysis_card = @import("analysis_card.zig");

/// Unresolved or absent node row.
pub const NO_NODE = analysis_args.NO_NODE;
/// What a card borrows from the rest of the deck.
pub const CardContext = analysis_args.CardContext;
/// Parsed `.options` overrides, in deck order.
pub const DeckOptions = analysis_options.DeckOptions;
/// Folds the deck's `.options` and `.temp` cards.
pub const deckOptions = analysis_options.deckOptions;
/// Copies deck options into one query.
pub const applyDeckOptions = analysis_options.applyDeckOptions;
/// The query one analysis card asks for.
pub const buildJob = analysis_card.buildJob;
/// A `.dc` sweep target.
pub const dcTarget = analysis_time.dcTarget;

/// How `queries` copies jobs over `Deck.variants` rows.
pub const Fanout = struct {
    /// Rows every card runs (`.step` points, `.alter` runs): the jobs of
    /// all cards once per row, after the nominal jobs when `nominal`.
    global: Span = .{},
    /// Rows of each card's own HSPICE sweep, parallel to the cards; empty
    /// for a deck without one.
    cards: []const Span = &.{},
    /// Keep only this card's jobs: a rebuilt point of one card's sweep.
    only: ?u32 = null,

    /// Rows `first..first + count` of `Deck.variants`.
    pub const Span = struct {
        first: u32 = 0,
        count: u32 = 0,
        nominal: bool = true,
        /// Solve the rows as one DC lane query (`.dc DATA=`, `.dc MONTE=`)
        /// whose first column is `axis`.
        lanes: ?struct { axis: []const u8, dc_plot: bool } = null,
    };
};

/// Queries for `cards` (nets already circuit rows), in card order, allocated
/// in `arena`. `.noise` also yields its integrated plot, `.lstb` its
/// margins, `.disto` its two harmonic vectors (three intermodulation
/// vectors with F2); a single-value `.temp` yields none. An output `v(...)`
/// names one or two nodes in the deck; `appended` cards must name exactly
/// one, else `UnsupportedAnalysisOutput`. `fanout` copies them over the
/// deck's variants, and an HSPICE `.temp` list copies everything once per
/// temperature. Errors are `buildJob`'s, logged with the card that raised
/// them, and `InvalidAnalysisArguments` for a card whose query
/// `core.validate` refuses on a circuit of `n_rows` unknowns, an HSPICE
/// `.hbac`-family card with no `.hb` tone to take, or a `.temp` list value
/// that is not a temperature above -273.15 degC (not logged);
/// `OutOfMemory` also covers a fan-out count too large for `usize`.
pub fn queries(arena: Allocator, cards: []const netlist.Analysis, n_rows: u32, appended: bool, sources: core.QueryBindings, card_refs: []const requests.CardRef, deck_opts: analysis_options.DeckOptions, fanout: Fanout) ![]const Job {
    for (cards) |c| {
        const args = c.args;
        const arg: usize = if (c.kind == .four) 1 else 0;
        if (arg < args.len) switch (args[arg]) {
            .group => |g| if (g.args().len > @as(usize, if (appended) 1 else 2) or (appended and g.args().len == 0))
                return cardError(c.line, error.UnsupportedAnalysisOutput),
            else => {},
        };
    }
    // What HSPICE forms borrow from the other cards: the first `.ac`,
    // `.tran` and `.sn`. A malformed one reports on its own turn below.
    var ctx: analysis_args.CardContext = .{ .arena = arena, .variations = deck_opts.variations, .card_names = deck_opts.card_names };
    // Fan-out ceiling: `.disto` gives up to four plots per line (two-tone),
    // `.op t1 t2 ...` one per time.
    var fan: usize = 4;
    for (cards) |c| switch (c.kind) {
        .ac => if (ctx.ac == null) {
            if (analysis_args.acGrid(arena, c.args, 0)) |g| {
                ctx.ac = g.sweep;
                ctx.ac.?.land_on_stop = true; // the grid the `.ac` card runs
            } else |_| {}
        },
        .tran => if (ctx.tran == null) {
            if (analysis_card.buildJob(c, sources, card_refs, ctx)) |job| {
                var tran = job.?;
                analysis_options.applyDeckOptions(&tran, deck_opts);
                ctx.tran = tran.tran;
            } else |_| {}
        },
        .pss => if (c.sn and ctx.sn_f0 == null) {
            if (analysis_card.buildJob(c, sources, card_refs, ctx)) |job| ctx.sn_f0 = 1 / job.?.pss.period else |_| {}
        },
        .op => fan = @max(fan, c.args.len),
        .tran_noise => fan = @max(fan, analysis_time.tranNoiseSamples(c.args)),
        .pnoise => if (c.sn and std.ascii.startsWithIgnoreCase(c.line, ".ptdnoise")) {
            var k: usize = 2;
            if (analysis_time.ptdTimes(arena, c.args, &k)) |times| fan = @max(fan, times.len) else |_| {}
        },
        else => {},
    };
    const temps = @max(deck_opts.temp_list.len, 1);
    var per_card: usize = fan;
    for (fanout.cards) |span| per_card = @max(per_card, fan * (span.count + 1));
    // The nominal jobs are built before they are copied over the rows, so
    // the table holds one row even when none survives (#132).
    const rows = @max(fanout.global.count + @intFromBool(fanout.global.nominal), 1);
    // `.trannoise SAMPLES=` and the variant counts are deck input, so the
    // product can overflow before the allocator ever sees it.
    var len: usize = cards.len;
    for ([_]usize{ per_card, temps + 1, rows }) |f| len = std.math.mul(usize, len, f) catch return error.OutOfMemory;
    const jobs = try arena.alloc(Job, len);
    var n: usize = 0;
    // HSPICE's `.hbac`, `.hbxf`, `.hbnoise` and `.phasenoise` take the tone
    // and harmonic count (and oscillator node) of the deck's `.hb` or
    // `.hbosc` card unless the card gives its own.
    var hb: ?requests.Hb = null;
    for (cards) |c| if (c.kind == .hb) {
        hb = ((analysis_card.buildJob(c, sources, card_refs, ctx) catch |err| return cardError(c.line, err)).?).hb;
    };
    var xf_sources: ?[]const requests.XfSource = null;
    for (cards, 0..) |c, ci| {
        if (fanout.only) |only| if (ci != only) continue;
        const span: Fanout.Span = if (ci < fanout.cards.len) fanout.cards[ci] else .{};
        if (span.lanes) |lanes| {
            var job: Job = .{ .mc = .{ .variants = .{ .first = span.first, .count = span.count }, .axis = lanes.axis, .dc_plot = lanes.dc_plot } };
            analysis_options.applyDeckOptions(&job, deck_opts);
            jobs[n] = job;
            n += 1;
            continue;
        }
        const first = n;
        var job = (analysis_card.buildJob(c, sources, card_refs, ctx) catch |err| return cardError(c.line, err)) orelse continue;
        switch (job) {
            inline .hbac, .hbxf, .hbnoise, .hblin => |*o, kind| if (o.f0 == 0) {
                const tone = hb orelse return cardError(c.line, error.InvalidAnalysisArguments);
                o.f0 = tone.f0;
                o.n_harmonics = tone.n_harmonics;
                o.n_sidebands = tone.n_harmonics;
                if (kind == .hblin) {
                    // ponytail: `.hblin` reads one tone's port bands; several
                    // tones need a band per mixing product.
                    if (tone.extra_tones.len != 0) return cardError(c.line, error.InvalidAnalysisArguments);
                } else {
                    o.extra_tones = tone.extra_tones;
                    o.extra_harmonics = tone.extra_harmonics;
                    o.intmodmax = tone.intmodmax;
                    o.subharms = tone.subharms;
                }
            },
            .phasenoise => |*o| if (o.f0 == 0) {
                const tone = hb orelse return cardError(c.line, error.InvalidAnalysisArguments);
                if (tone.osc_node == GROUND) return cardError(c.line, error.InvalidAnalysisArguments);
                o.f0 = tone.f0;
                o.n_harmonics = tone.n_harmonics;
                o.osc_node = tone.osc_node;
            },
            inline .dcxf, .acxf => |*o| {
                if (xf_sources == null) xf_sources = try analysis_small.xfSources(arena, sources);
                o.sources = xf_sources.?;
            },
            else => {},
        }
        analysis_options.applyDeckOptions(&job, deck_opts);
        // Query validation stays the API's backstop; a card it would refuse
        // fails here, with its line (#108).
        core.validate.validate(job, n_rows) catch return cardError(c.line, error.InvalidAnalysisArguments);
        // HSPICE `.op t1 t2 ...` [CR .OP]: one operating point per time, a
        // transient snapshot past t = 0.
        if (job == .op and analysis_args.hasNumber(c.args)) {
            for (c.args) |v| {
                const t = switch (v) {
                    .num => |t| t,
                    else => continue,
                };
                jobs[n] = if (t == 0) job else snapshot(t, ctx.tran);
                analysis_options.applyDeckOptions(&jobs[n], deck_opts);
                n += 1;
            }
            n = first + copyVariants(jobs[first..], n - first, span);
            continue;
        }
        // HSPICE `.trannoise ... SAMPLES=n`: one run per Monte Carlo index
        // from SEED on; index 1 is noiseless [RF Ch.9].
        if (job == .tran_noise and c.args.len > 0 and c.args[0] != .num) {
            const runs = analysis_time.tranNoiseSamples(c.args);
            const base = job.tran_noise;
            for (0..runs) |k| {
                var o = base;
                o.seed = base.seed + k;
                if (o.seed == 1) o.scale = 0;
                if (runs > 1) o.sample = @intCast(o.seed);
                jobs[n] = .{ .tran_noise = o };
                n += 1;
            }
            n = first + copyVariants(jobs[first..], n - first, span);
            continue;
        }
        // HSPICE `.ptdnoise TIME=<sweep>`: one strobed noise query per time.
        // ponytail: each query shoots its own PSS; one orbit serving every
        // strobe is the upgrade if long time sweeps get slow.
        if (job == .pnoise and job.pnoise.strobe != null) {
            var k: usize = 2;
            for (try analysis_time.ptdTimes(arena, c.args, &k)) |t| {
                jobs[n] = job;
                jobs[n].pnoise.strobe = t;
                n += 1;
            }
            n = first + copyVariants(jobs[first..], n - first, span);
            continue;
        }
        jobs[n] = job;
        n += 1;
        // `.lstb` also reports its margins.
        if (job == .lstb) {
            job.lstb.margins = true;
            jobs[n] = job;
            n += 1;
        }
        // The "Integrated Noise" plot always exists; a degenerate band
        // integrates to zero and ngspice still prints the row.
        if (job == .noise and !job.noise.phase) {
            job.noise.integrated = true;
            jobs[n] = job;
            n += 1;
        }
        // ngspice's `.disto` output is the pair of complex harmonic vectors,
        // or with F2 the three intermodulation vectors (distoan.c:510-620).
        if (job == .disto) {
            const plots: []const requests.Disto.Plot = if (job.disto.f2_ratio == 0) &.{ .second, .third } else &.{ .f1pf2, .f1mf2, .twof1mf2 };
            for (plots) |plot| {
                job.disto.plot = plot;
                jobs[n] = job;
                n += 1;
            }
        }
        n = first + copyVariants(jobs[first..], n - first, span);
    }
    n = copyVariants(jobs, n, fanout.global);
    if (deck_opts.sample) |s| sampleNoise(jobs[0..n], s);
    if (temps == 1) return jobs[0..n];
    // An HSPICE `.temp` list runs every query once per temperature [CR .TEMP].
    const base = n;
    n = 0;
    for (deck_opts.temp_list) |t| {
        const temp = switch (t) {
            .num => |v| v,
            else => return error.InvalidAnalysisArguments,
        };
        if (!(temp > -273.15) or !std.math.isFinite(temp)) return error.InvalidAnalysisArguments;
        for (jobs[0..base]) |job| {
            var copy = job;
            switch (copy) {
                inline else => |*opts| opts.tol.temp_c = temp,
            }
            jobs[base + n] = copy;
            n += 1;
        }
    }
    return jobs[base..][0..n];
}

// ---- private ----

/// A transient that publishes only its state at `t` (HSPICE `.op <time>`),
/// stepping as the deck's `.tran` does and at most t / 50.
fn snapshot(t: f64, tran: ?requests.Tran) Job {
    var s: requests.Tran = tran orelse .{ .t_stop = t, .dt_init = t / 50 };
    s.t_stop = t;
    s.t_start = 0;
    s.snapshot = true;
    s.dt_init = @min(s.dt_init, t / 50);
    s.dt_max = @min(s.dt_max orelse t / 50, t / 50);
    return .{ .tran = s };
}

/// Rewrites the first `n` jobs of `jobs` as their copies over `span`'s rows,
/// row-major, after the originals when `span.nominal`, and returns the count:
/// `n` for a span with no rows that keeps the nominal, 0 for one that keeps
/// neither (every point rebuilt as its own run).
fn copyVariants(jobs: []Job, n: usize, span: Fanout.Span) usize {
    const skip: usize = @intFromBool(span.nominal);
    // Back to front, so no copy overwrites a job not yet copied.
    var row = span.count;
    while (row > 0) {
        row -= 1;
        for (0..n) |k| {
            var copy = jobs[k];
            switch (copy) {
                inline else => |*opts| opts.tol.variant = span.first + @as(u32, @intCast(row)),
            }
            jobs[(row + skip) * n + k] = copy;
        }
    }
    return (span.count + skip) * n;
}

/// Logs the card a query could not be built from, then returns `err`.
fn cardError(line: []const u8, err: anytype) @TypeOf(err) {
    // The test runner fails any test that logs an error.
    if (!@import("builtin").is_test) std.log.err("analysis: {s}: {s}", .{ line, @errorName(err) });
    return err;
}

/// Sets HSPICE `.sample` on every `.noise` spectrum of `jobs`; the
/// integrated plot keeps the plain band.
fn sampleNoise(jobs: []Job, sample: requests.NoiseSample) void {
    for (jobs) |*q| if (q.* == .noise and !q.noise.integrated) {
        q.noise.sample = sample;
    };
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

test sampleNoise {
    const sweep: core.numerics.FreqSweep = .{ .f_start = 1, .f_stop = 10 };
    var jobs = [_]Job{
        .{ .noise = .{ .out_node = 1, .sweep = sweep } },
        .{ .noise = .{ .out_node = 1, .sweep = sweep, .integrated = true } },
        .{ .op = .{} },
    };
    sampleNoise(&jobs, .{ .fs = 1e6 });
    try testing.expectEqual(@as(f64, 1e6), jobs[0].noise.sample.?.fs);
    // The integrated plot keeps the plain band.
    try testing.expect(jobs[1].noise.sample == null);
}

test copyVariants {
    var jobs: [6]Job = @splat(.{ .op = .{} });
    // Nominal kept: the originals first, then one block per row.
    try testing.expectEqual(@as(usize, 6), copyVariants(&jobs, 2, .{ .first = 5, .count = 2, .nominal = true }));
    for (jobs, [_]?u32{ null, null, 5, 5, 6, 6 }) |j, want| try testing.expectEqual(want, j.op.tol.variant);
    jobs = @splat(.{ .op = .{} });
    try testing.expectEqual(@as(usize, 4), copyVariants(&jobs, 2, .{ .first = 5, .count = 2, .nominal = false }));
    for (jobs[0..4], [_]?u32{ 5, 5, 6, 6 }) |j, want| try testing.expectEqual(want, j.op.tol.variant);
    try testing.expectEqual(@as(usize, 0), copyVariants(&jobs, 0, .{ .count = 3, .nominal = true }));
    // No rows: the nominal stays, or nothing does.
    try testing.expectEqual(@as(usize, 2), copyVariants(&jobs, 2, .{}));
    try testing.expectEqual(@as(usize, 0), copyVariants(&jobs, 2, .{ .nominal = false }));
}

test "deckOptions: a NaN RUNLVL fails, and the last .temp card wins" {
    const nan = comptime std.math.nan(f64);
    try testing.expectError(error.InvalidAnalysisArguments, analysis_options.deckOptions(&.{.{ .temp = false, .args = argv(.{ "runlvl", nan }) }}, .hspice));
    try testing.expectError(error.InvalidAnalysisArguments, analysis_options.deckOptions(&.{.{ .temp = false, .args = argv(.{ "runlvl", 7 }) }}, .hspice));
    const list_then_one = try analysis_options.deckOptions(&.{ .{ .temp = true, .args = argv(.{ 0, 50 }) }, .{ .temp = true, .args = argv(.{85}) } }, .hspice);
    try testing.expectEqual(@as(usize, 0), list_then_one.temp_list.len);
    try testing.expectEqual(@as(?f64, 85), list_then_one.temp_c);
    const one_then_list = try analysis_options.deckOptions(&.{ .{ .temp = true, .args = argv(.{85}) }, .{ .temp = true, .args = argv(.{ 0, 50 }) } }, .ngspice);
    try testing.expectEqual(@as(usize, 2), one_then_list.temp_list.len);
    try testing.expectEqual(@as(?f64, null), one_then_list.temp_c);
    // Unknown and over-long option names are skipped, not fatal.
    _ = try analysis_options.deckOptions(&.{.{ .temp = false, .args = argv(.{ "a_very_long_option_name", 1, "bogus", 2 }) }}, .ngspice);
}

test queries {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const none: core.QueryBindings = .{ .v_names = &.{}, .i_names = &.{}, .v_branches = &.{}, .v_pos = &.{}, .v_neg = &.{}, .i_pos = &.{}, .i_neg = &.{}, .v_distof1 = &.{}, .ports = &.{} };
    const op: netlist.Analysis = .{ .kind = .op, .args = &.{} };
    try testing.expectEqual(@as(usize, 0), (try queries(a, &.{}, 16, false, none, &.{}, .{}, .{})).len);
    // A `.temp` list runs every job once per temperature.
    const temps = try queries(a, &.{op}, 16, false, none, &.{}, .{ .temp_list = argv(.{ -40, 125 }) }, .{});
    try testing.expectEqual(@as(usize, 2), temps.len);
    try testing.expectEqual(@as(?f64, -40), temps[0].op.tol.temp_c);
    try testing.expectEqual(@as(?f64, 125), temps[1].op.tol.temp_c);
    try testing.expectError(error.InvalidAnalysisArguments, queries(a, &.{op}, 16, false, none, &.{}, .{ .temp_list = argv(.{ 0, -300 }) }, .{}));
    // HSPICE `.op 0 1u`: the operating point, then a snapshot at 1 us.
    const times = try queries(a, &.{.{ .kind = .op, .args = argv(.{ 0, 1e-6 }) }}, 16, false, none, &.{}, .{}, .{});
    try testing.expectEqual(@as(usize, 2), times.len);
    try testing.expect(times[0] == .op);
    try testing.expect(times[1].tran.snapshot);
    try testing.expectEqual(@as(f64, 1e-6), times[1].tran.t_stop);
    // Global rows after the nominal one.
    const rows = try queries(a, &.{op}, 16, false, none, &.{}, .{}, .{ .global = .{ .first = 0, .count = 2, .nominal = true } });
    try testing.expectEqual(@as(usize, 3), rows.len);
    for (rows, [_]?u32{ null, 0, 1 }) |j, want| try testing.expectEqual(want, j.op.tol.variant);
    // Every row rebuilt as its own run: no jobs left here, and none
    // written past the table (#132).
    try testing.expectEqual(@as(usize, 0), (try queries(a, &.{op}, 16, false, none, &.{}, .{}, .{ .global = .{ .first = 0, .count = 0, .nominal = false } })).len);
    const cards: []const Fanout.Span = &.{.{ .count = 0, .nominal = false }};
    try testing.expectEqual(@as(usize, 0), (try queries(a, &.{op}, 16, false, none, &.{}, .{}, .{ .cards = cards })).len);
    // `only` keeps one card's jobs.
    const only = try queries(a, &.{ op, .{ .kind = .dcinc, .args = &.{} } }, 16, false, none, &.{}, .{}, .{ .only = 1 });
    try testing.expectEqual(@as(usize, 1), only.len);
    try testing.expect(only[0] == .dcinc);
}

test {
    _ = analysis_args;
    _ = analysis_options;
    _ = analysis_time;
    _ = analysis_small;
    _ = analysis_rf;
    _ = analysis_card;
}
