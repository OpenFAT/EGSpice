//! Circuit variants over one prepared circuit: `.step` points, HSPICE
//! `SWEEP`/`.data` points, `.alter` runs, Monte Carlo trials and the live
//! `.param` values a caller sets on a Problem (dev/analysis/variants.md).
//! Each point re-evaluates the netlist's live values (no re-parse),
//! rebinds the devices and keeps only the parameters that moved, as
//! `core.Variants` rows the executor writes through `ParamRef`. A point
//! whose topology moves becomes a `Rebuild` request, which the prepare
//! spine builds as its own run; a live value whose model bins move becomes
//! a new template (`Tuning`). In: a netlist and its nominal circuit. Out:
//! a `Plan`. Stages: variant_planner.zig (rows and rebuilds),
//! variant_mismatch.zig, variant_draws.zig, variant_tuning.zig,
//! variant_params.zig.
//! Cites: dev/analysis/variants.md; HSPICE [SA] Ch.25-27.
//!
//! Contents:
//! - `pub const Planner, Tuner, Tuning, Params, collect, Sampler, Trial, setPoint, drawOnce, variations, siteVariations` (stage files)
//! - `pub const Plan`
//! - `pub const Run`
//! - `pub const Rebuild`
//! - `pub const Point`
//! - `pub const CardValue`
//! - `pub fn any(deck: netlist.Deck) bool`
//! - `pub fn plan(p: *variant_planner.Planner) !Plan`

const std = @import("std");
const assert = std.debug.assert;
const testing = std.testing;
const Allocator = std.mem.Allocator;
const core = @import("core");
const device = @import("device");
const netlist = @import("netlist");
const builder = @import("builder");
const analyses = @import("analyses.zig");
const expr = netlist.expr;
const Netlist = netlist.Netlist;
const ParamRef = device.abi.ParamRef;
const variant_draws = @import("variant_draws.zig");
const variant_planner = @import("variant_planner.zig");
const variant_params = @import("variant_params.zig");
const variant_tuning = @import("variant_tuning.zig");
const variant_mismatch = @import("variant_mismatch.zig");
const variant_circuit = @import("variant_circuit.zig");

/// Planning state (variant_planner.zig).
pub const Planner = variant_planner.Planner;
/// HSPICE `OPTIMIZE=` rows (variant_tuning.zig).
pub const Tuner = variant_tuning.Tuner;
/// Live `.param` values set after the build (variant_tuning.zig).
pub const Tuning = variant_tuning.Tuning;
/// A circuit's parameters by ordinal (variant_params.zig).
pub const Params = variant_params.Params;
/// Every parameter of a circuit (variant_params.zig).
pub const collect = variant_params.collect;
/// Monte Carlo draws for one sweep (variant_draws.zig).
pub const Sampler = variant_draws.Sampler;
/// One Monte Carlo trial (variant_draws.zig).
pub const Trial = variant_draws.Trial;
/// Sets a netlist's live values to a point's (variant_draws.zig).
pub const setPoint = variant_draws.setPoint;
/// Draws every distribution site once from `.option seed`.
pub const drawOnce = variant_draws.drawOnce;
/// The `.variation` rows as mismatch groups (variant_mismatch.zig).
pub const variations = variant_mismatch.variations;
/// The deck's distribution sites as mismatch groups (variant_mismatch.zig).
pub const siteVariations = variant_mismatch.siteVariations;

/// A run's variants: the table, how each card fans out over it, and the
/// runs that need their own circuit, for the caller to build in order.
pub const Plan = struct {
    /// One row per point the nominal circuit can hold, in the session arena.
    variants: core.Variants = .{},
    /// Which cards run over which rows (`analyses.queries`).
    fanout: analyses.Fanout = .{},
    /// Points and `.alter` runs that need a circuit of their own.
    rebuilds: []const Rebuild = &.{},
};

/// Which run of a deck `prepare.buildRun` builds.
pub const Run = struct {
    /// Label of the run's rows: `alter=2`, or a rebuilt point's own.
    prefix: []const u8 = "",
    /// A point whose circuit differs from the nominal one: the run sets
    /// its live values before it builds, and the run is that one row.
    point: ?Point = null,
    /// The point belongs to this analysis card's sweep alone.
    card: ?u32 = null,
};

/// A run the plan could not hold as rows: an `.alter` netlist, or the
/// planned netlist at one point. Both live in the planner's scratch arena.
pub const Rebuild = struct { nl: *Netlist, run: Run };

/// What one point sets, before it becomes a row.
pub const Point = struct {
    /// The row's label, `name=value, ...`.
    label: []const u8,
    /// The row's sweep-axis value: the first swept value or the trial
    /// number; 0 when nothing is swept.
    axis: f64,
    /// Circuit temperature in degC; null keeps the deck's.
    temp: ?f64 = null,
    /// Every live name's value (`Live.names`).
    live: []const f64,
    /// Card primary values: a source's `dc`, a resistor's `r`.
    cards: []const CardValue = &.{},
    /// Monte Carlo trial, whose draws answer the distribution calls and
    /// the `DEV`/`LOT`/`.variation` rows.
    trial: ?variant_draws.Trial = null,
};

/// A card's primary value for one point, by card name.
pub const CardValue = struct { name: []const u8, value: f64 };

/// True when `deck` asks for variants: `.step`, `.alter` or a card's sweep.
pub fn any(deck: netlist.Deck) bool {
    for (deck.analyses) |a| if (a.sweep != .none) return true;
    return deck.steps.len != 0 or deck.alters.len != 0;
}

/// Plans every variant of `p`'s deck: each card's own sweep, then the
/// `.step` grid, then the `.alter` runs, as rows over `p.nominal` and runs
/// to rebuild where the topology moves. `p.prefix` labels an `.alter`
/// run's rows. `p.nl`'s live values are nominal again on success.
/// `UnsupportedCard` for a combination the plan cannot hold (SWEEP with
/// `.step`, a DC ensemble whose topology moves, a trial number past u32).
pub fn plan(p: *variant_planner.Planner) !Plan {
    const nl = p.nl;
    const deck = nl.deck;
    if (deck.optimize != null) return p.planOptimize();
    var fanout: analyses.Fanout = .{};
    const card_spans = try p.scratch.alloc(analyses.Fanout.Span, deck.analyses.len);
    @memset(card_spans, .{});
    var swept = false;
    for (deck.analyses, card_spans, 0..) |a, *span, ci| {
        if (a.sweep == .none) continue;
        if (deck.steps.len != 0) return p.refuse(a.line, "SWEEP together with .step");
        swept = true;
        span.first = p.count();
        const runs_before = p.rebuilds.items.len;
        const lanes = a.kind == .dc and a.args.len == 0;
        // Only Monte Carlo keeps the nominal run beside its trials.
        span.nominal = a.sweep == .monte and !lanes;
        p.card = @intCast(ci);
        switch (a.sweep) {
            .none, .optimize => unreachable,
            .step => |st| {
                for (st.values) |v| try p.add(try p.stepPoint(&.{st}, &.{v}));
                // `.dc param start stop incr`: the parameter is the axis.
                if (lanes) span.lanes = .{ .axis = switch (st.target) {
                    .param => |k| nl.live.names[k],
                    .card => |c| c,
                    .temp => "temp",
                }, .dc_plot = true };
            },
            .data => |name| {
                const table = for (deck.data) |d| {
                    if (std.mem.eql(u8, d.name, name)) break d;
                } else return p.refuse(a.line, "no such .data table");
                const cols = table.columns.len;
                var row: usize = 0;
                while (row * cols < table.values.len) : (row += 1)
                    try p.addRow(table, table.values[row * cols ..][0..cols]);
                if (lanes) span.lanes = .{ .axis = table.labels[0], .dc_plot = true };
            },
            .monte => |m| if (m.list.len != 0) {
                // A listed trial t is trial t of a `MONTE=max` run, Latin
                // hypercube stratum included.
                // A rebuilt trial draws from it after `plan` returns.
                const sampler = try p.scratch.create(variant_draws.Sampler);
                sampler.* = try .init(p.scratch, deck.config, std.mem.max(u32, m.list));
                for (m.list) |t| try p.addTrial(sampler, t, t - 1);
                if (lanes) span.lanes = .{ .axis = "run", .dc_plot = false };
            } else {
                const sampler = try p.scratch.create(variant_draws.Sampler);
                sampler.* = try .init(p.scratch, deck.config, m.n);
                if (m.n > std.math.maxInt(u32) - m.first + 1) return p.refuse(a.line, "a MONTE trial number past 4294967295");
                for (0..m.n) |k| try p.addTrial(sampler, m.first + @as(u32, @intCast(k)), @intCast(k));
                if (lanes) span.lanes = .{ .axis = "run", .dc_plot = false };
            },
        }
        p.card = null;
        span.count = p.count() - span.first;
        if (lanes and p.rebuilds.items.len != runs_before) return p.refuse(a.line, "a DC ensemble point that changes the topology");
    }
    if (swept) fanout.cards = try p.sim.dupe(analyses.Fanout.Span, card_spans);
    if (deck.steps.len != 0) {
        const values = try p.scratch.alloc(f64, deck.steps.len);
        fanout.global = .{ .first = p.count(), .nominal = false };
        var points: std.ArrayList(Point) = .empty;
        try p.grid(deck.steps, values, 0, &points);
        for (points.items) |pt| try p.add(pt);
        fanout.global.count = p.count() - fanout.global.first;
    }
    if (deck.alters.len != 0) {
        // With other variants, each `.alter` run carries its own: it is a
        // rebuild, adds no row, and the `.step` rows stay the global span
        // (#66).
        const own = swept or deck.steps.len != 0;
        if (!own) fanout.global = .{ .first = p.count(), .nominal = true };
        for (deck.alters, 1..) |text, k| try p.addAlter(text, k, own);
        if (!own) fanout.global.count = p.count() - fanout.global.first;
    }
    if (p.prefix.len != 0 and p.count() == 0) {
        try p.closeRow(p.prefix, 0, null);
        fanout.global = .{ .first = 0, .count = 1, .nominal = false };
    }
    _ = try nl.setLive(p.scratch, &p.stack, nl.live.nominal, null);
    return .{ .variants = try p.table(), .fanout = fanout, .rebuilds = p.rebuilds.items };
}

// ---- private ----

test {
    _ = variant_params;
    _ = variant_draws;
    _ = variant_tuning;
    _ = variant_planner;
    _ = variant_mismatch;
    _ = variant_circuit;
}
