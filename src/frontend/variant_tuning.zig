//! Live parameters a caller moves after the build. In: the prepared
//! circuit and new values for its live `.param`s or optimized points. Out:
//! variant rows (`Tuner` for HSPICE OPTIMIZE, `Tuning` for
//! `Problem.set_params`), or a new template when bins or topology move.
//! Cites: HSPICE [SA] Ch.27 (optimization); dev/analysis/variants.md.
//!
//! Contents:
//! - `pub const Tuner`
//! - `pub fn init(planner: variant_planner.Planner) !Tuner`
//! - `pub fn rows(t: *Tuner, arena: std.mem.Allocator, nominal: *const device.Circuit, points: []const f64, base: usize) !core.Variants`
//! - `pub fn optimized(arena: std.mem.Allocator, planned: core.Variants, best: []const core.Variants) !core.Variants`
//! - `pub const Tuning`
//! - `pub const Change`
//! - `pub fn init(gpa: std.mem.Allocator, lib: *const device.Library, sim: std.mem.Allocator, scratch: std.mem.Allocator, nl: *Netlist, nominal: *const device.Circuit, cards: []const core.query.CardRef) !*Tuning`
//! - `pub fn deinit(t: *Tuning) void`
//! - `pub fn set(t: *Tuning, nominal: *const device.Circuit, current: core.Variants, names: []const []const u8, values: []const f64) !Change`
//! - `pub fn setAt(t: *Tuning, nominal: *const device.Circuit, current: core.Variants, ks: []const u32, values: []const f64) !Change`
//! - `pub fn index(t: *const Tuning, name: []const u8) !u32`
//! - `pub fn hash(c: NameContext, name: []const u8) u64`
//! - `pub fn eql(c: NameContext, a: []const u8, b: []const u8) bool`
//! - `pub fn refusedRow(t: *Tuning, current: core.Variants) !core.Variants`

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
const variant_circuit = @import("variant_circuit.zig");
const variants = @import("variants.zig");

/// Turns optimizer points into variant rows (HSPICE `OPTIMIZE=`,
/// dev/analysis/optimize.md): each point sets the optimized parameters'
/// live values and goes through `Planner.add`, so it takes the same
/// probe-mapped fast path as a `.step` point. Lives in the parse arena,
/// with the netlist it re-evaluates.
pub const Tuner = struct {
    planner: variant_planner.Planner,
    /// The optimized card's index in the deck's analyses.
    card: u32,
    /// The optimization, its parameters' initial values and names, and
    /// the `.model OPT` options.
    spec: netlist.Optimize,
    initial: []const f64,
    names: []const []const u8,
    options: core.lm.Options,
    method: core.bisect.Method,
    /// The RESULTS cards' indices in the deck's `.meas` cards.
    results: []const u32,
    /// The points each optimization starts from (`Planner.basePoints`): one
    /// optimization per `.step` point.
    bases: []const variants.Point,

    /// A tuner over the planner of the deck's main run. Asserts that the
    /// deck has an optimization and a card with `SWEEP OPTIMIZE=`;
    /// `UnsupportedCard` for RESULTS or `.model OPT` settings it cannot run.
    pub fn init(planner: variant_planner.Planner) !Tuner {
        var pl = planner;
        const nl = planner.nl;
        const spec = nl.deck.optimize.?;
        const card: u32 = for (nl.deck.analyses, 0..) |a, ci| {
            if (a.sweep == .optimize) break @intCast(ci);
        } else unreachable;
        const init_values = try planner.scratch.alloc(f64, spec.live.len);
        const names = try planner.scratch.alloc([]const u8, spec.live.len);
        for (spec.live, init_values, names) |k, *v, *name| {
            v.* = nl.live.nominal[k];
            name.* = nl.live.names[k];
        }
        const line = nl.deck.analyses[card].line;
        const results = try planner.scratch.alloc(u32, spec.results.len);
        for (spec.results, results) |name, *index| {
            index.* = for (nl.deck.measures, 0..) |m, i| {
                if (std.mem.eql(u8, m.name, name) and m.analysis == nl.deck.analyses[card].kind and m.goalError(0) != null) break @intCast(i);
            } else return planner.refuse(line, "a RESULTS name that is no .meas card of this analysis with GOAL=");
        }
        const options, const method = try optOptions(nl, spec.model, line);
        // ponytail: one parameter and one RESULTS card for a bisection; the
        // manual's AND over several bisected parameters is not specified.
        if (method != .lm and (spec.live.len != 1 or results.len != 1)) return planner.refuse(line, "a bisection over more than one parameter or RESULTS card");
        // A bisection passes on the sign of the goal error, which an inequality goal zeroes.
        const goal = nl.deck.measures[results[0]];
        if (method != .lm and (if (goal.first.goal != null) goal.first else goal.second).goal_bound != .equal) return planner.refuse(line, "a bisection on a GOAL < or GOAL > card");
        return .{ .planner = planner, .card = card, .spec = spec, .initial = init_values, .names = names, .options = options, .method = method, .results = results, .bases = try pl.basePoints() };
    }

    /// One row per point of `points` (the optimized parameters' values,
    /// `spec.live.len` per point) on top of `bases[base]`, allocated in
    /// `arena`. `nominal` is the prepared circuit the rows write into.
    /// Resets the planner's rows, so a table from an earlier call stays
    /// valid (it was copied out) but the planner no longer holds it.
    pub fn rows(t: *Tuner, arena: std.mem.Allocator, nominal: *const device.Circuit, points: []const f64, base: usize) !core.Variants {
        const p = &t.planner;
        p.nominal = nominal;
        p.sim = arena;
        inline for (.{ &p.labels, &p.temps, &p.axes, &p.write_refs, &p.write_values }) |list| list.clearRetainingCapacity();
        p.starts.shrinkRetainingCapacity(1);
        const b = t.bases[base];
        const live = try arena.dupe(f64, b.live);
        const n = t.spec.live.len;
        var at: usize = 0;
        while (at < points.len) : (at += n) {
            for (t.spec.live, points[at..][0..n]) |k, v| live[k] = v;
            try p.add(.{ .label = "", .axis = 0, .live = live, .temp = b.temp, .cards = b.cards });
            if (p.rebuilds.items.len != 0) return p.refuse(p.nl.deck.analyses[t.card].line, "an optimized value that changes the topology");
        }
        return p.table();
    }

    /// `planned` (the deck's variant table) with each optimum installed:
    /// rows `0..n` and `n..2n` take `best[v % n]`'s writes, one optimum per
    /// `bases` entry (`n = best.len`), and the rows after keep the plan's.
    /// Labels, temperatures and axes stay the plan's. Allocated in `arena`.
    pub fn optimized(arena: std.mem.Allocator, planned: core.Variants, best: []const core.Variants) !core.Variants {
        const n = best.len;
        var starts: std.ArrayList(u32) = .empty;
        var refs: std.ArrayList(u32) = .empty;
        var values: std.ArrayList(f64) = .empty;
        try starts.append(arena, 0);
        for (0..planned.starts.len - 1) |v| {
            const w_refs, const w_values = if (v < 2 * n) best[v % n].writes(0) else planned.writes(@intCast(v));
            try refs.appendSlice(arena, w_refs);
            try values.appendSlice(arena, w_values);
            try starts.append(arena, @intCast(refs.items.len));
        }
        var out = planned;
        out.starts = starts.items;
        out.refs = refs.items;
        out.values = values.items;
        return out;
    }
};

/// Global `.param` values set on a prepared circuit (`Problem.set_params`,
/// #101; dev/analysis/variants.md §5). Each call re-evaluates the netlist
/// at the new values as a `.step` point does, never re-parsing, and becomes
/// one more row of the deck's variant table over the current template:
/// the probe-mapped writes or a rebuild's diff. When no writes can make the
/// point (a model bin moved, or a kept expression reads a moved value) the
/// row and later rows run on an earlier template that holds the point's
/// bins and kept values, or else on the circuit built at it, so the
/// templates are bounded by the distinct bin selections visited (#129).
/// Lives in the parse arena, with the netlist it re-evaluates.
pub const Tuning = struct {
    gpa: std.mem.Allocator,
    /// The planner's tables, in `arenas[current]`; a planner re-opened over
    /// another template builds in the other one, so a failure keeps the old.
    arenas: [2]std.heap.ArenaAllocator,
    current: u1 = 0,
    planner: variant_planner.Planner,
    /// The template the planner is open over; it differs from `template`
    /// only after a call failed past re-opening it.
    planned: u32 = 0,
    /// Every live name's value now, and the call's candidate values.
    values: []f64,
    next: []f64,
    /// The `.option seed` draw (trial 1), when the deck takes one, so a
    /// point keeps the nominal circuit's mismatch sample.
    trial: ?variant_draws.Trial,
    /// Each live name to its slot, compared as `index` does, so `set`
    /// costs one lookup per name, not a scan of every live name.
    slots: std.HashMapUnmanaged([]const u8, u32, NameContext, std.hash_map.default_max_load_percentage) = .empty,
    /// Template 0 (the nominal circuit, `circuit` null) and those built
    /// where no writes could make the point, oldest first, all in `gpa`;
    /// `Session.templates` numbers them the same way.
    templates: std.ArrayList(Template) = .empty,
    /// The template the last row runs on, and later rows diff against.
    template: u32 = 0,
    /// The deck's variant table with each call's row appended, in the
    /// session arena (`Planner.sim`); seeded on the first call.
    seeded: bool = false,
    labels: std.ArrayList([]const u8) = .empty,
    temps: std.ArrayList(?f64) = .empty,
    axes: std.ArrayList(f64) = .empty,
    starts: std.ArrayList(u32) = .empty,
    refs: std.ArrayList(u32) = .empty,
    write_values: std.ArrayList(f64) = .empty,

    /// What one call changed: the whole variant table, whose last row is
    /// the call's, the template that row runs on (0 for the nominal
    /// circuit, k for the k-th built), and that template's circuit when
    /// this call built it.
    pub const Change = struct { variants: core.Variants, template: u32, built: ?*const device.Circuit };

    /// A circuit rows can run on, the live values it was built at, and what
    /// it holds that writes cannot reach, which a point must match to run
    /// on it.
    const Template = struct {
        circuit: ?*device.Circuit,
        values: []const f64,
        kept: variant_circuit.Kept,

        /// The planner's nominal state over `circuit`, copied into `gpa`.
        fn of(gpa: std.mem.Allocator, circuit: ?*device.Circuit, values: []const f64, kept: variant_circuit.Kept) !Template {
            const v = try gpa.dupe(f64, values);
            errdefer gpa.free(v);
            const bins = try gpa.dupe(u32, kept.bins);
            errdefer gpa.free(bins);
            return .{ .circuit = circuit, .values = v, .kept = .{ .bins = bins, .pool = try gpa.dupe(f64, kept.pool) } };
        }

        /// Frees the copies and the circuit, which `gpa` owns.
        fn deinit(tm: Template, gpa: std.mem.Allocator) void {
            if (tm.circuit) |c| {
                c.deinit();
                gpa.destroy(c);
            }
            gpa.free(tm.values);
            gpa.free(tm.kept.bins);
            gpa.free(tm.kept.pool);
        }
    };

    /// A tuning over `nl`, whose nominal circuit is `nominal` with card
    /// table `cards`. `scratch` (the parse arena) holds the result; `sim`
    /// (the session arena) holds the variant table. Asserts that `nl` has
    /// tunable names. The caller releases it with `deinit`.
    pub fn init(gpa: std.mem.Allocator, lib: *const device.Library, sim: std.mem.Allocator, scratch: std.mem.Allocator, nl: *Netlist, nominal: *const device.Circuit, cards: []const core.query.CardRef) !*Tuning {
        std.debug.assert(nl.tunable.len != 0);
        const t = try scratch.create(Tuning);
        t.* = .{ .gpa = gpa, .arenas = .{ .init(gpa), .init(gpa) }, .planner = undefined, .values = try scratch.dupe(f64, nl.live.nominal), .next = try scratch.dupe(f64, nl.live.nominal), .trial = null };
        errdefer for (&t.arenas) |*a| a.deinit();
        t.planner = try variant_planner.Planner.init(lib, sim, t.arenas[0].allocator(), nl, nominal, cards, "");
        const names: NameContext = .{ .fold = nl.deck.dialect != .spectre };
        try t.slots.ensureTotalCapacityContext(scratch, @intCast(nl.live.names.len), names);
        // The first of two names that compare equal wins, as a scan finds it.
        for (nl.live.names, 0..) |name, k| {
            const gop = t.slots.getOrPutAssumeCapacityContext(name, names);
            if (!gop.found_existing) gop.value_ptr.* = @intCast(k);
        }
        try t.templates.ensureTotalCapacity(gpa, 1);
        errdefer t.templates.deinit(gpa);
        t.templates.appendAssumeCapacity(try .of(gpa, null, nl.live.nominal, t.planner.kept));
        if (nl.deck.draw_seed) {
            const sampler = try scratch.create(variant_draws.Sampler);
            sampler.* = try variant_draws.Sampler.init(scratch, nl.deck.config, 1);
            t.trial = .{ .sampler = sampler, .trial = 1, .index = 0 };
        }
        return t;
    }

    /// Frees the templates and the planner. The variant table stays with
    /// the session arena.
    pub fn deinit(t: *Tuning) void {
        for (t.templates.items) |tm| tm.deinit(t.gpa);
        t.templates.deinit(t.gpa);
        for (&t.arenas) |*a| a.deinit();
    }

    /// Re-opens the planner over template `k`, with the netlist at the
    /// values it was built at. `nominal` is template 0's circuit.
    fn open(t: *Tuning, k: u32, nominal: *const device.Circuit) !void {
        const p = &t.planner;
        const tm = t.templates.items[k];
        try variant_draws.setPoint(p.scratch, &p.stack, p.nl, .{ .label = "", .axis = 0, .live = tm.values, .trial = t.trial });
        const next = &t.arenas[~t.current];
        _ = next.reset(.free_all);
        const planner = try variant_planner.Planner.init(p.lib, p.sim, next.allocator(), p.nl, tm.circuit orelse nominal, p.cards, "");
        _ = t.arenas[t.current].reset(.free_all);
        t.current = ~t.current;
        p.* = planner;
        t.planned = k;
    }

    /// Sets each live name in `names` to the matching value, keeping every
    /// other at its last value, and returns the table with the point's row
    /// appended. `nominal` is the prepared circuit (template 0); `current`
    /// is the deck's table, read on the first call only. Names compare as
    /// the dialect spells them (case-insensitively but in Spectre).
    ///
    /// All-or-nothing: on failure the values, the table and the templates
    /// stay as they were. `UnknownParameter` for a name that is not live,
    /// `InvalidArgument` for a non-finite value, `ModelBinNotFound` when no
    /// bin holds a new L/W, `UnsupportedCard` when the point flips a
    /// top-level `.if` or changes the topology. Asserts that `names` and
    /// `values` have the same length.
    pub fn set(t: *Tuning, nominal: *const device.Circuit, current: core.Variants, names: []const []const u8, values: []const f64) !Change {
        std.debug.assert(names.len == values.len);
        @memcpy(t.next, t.values);
        for (names, values) |name, v| {
            if (!std.math.isFinite(v)) return error.InvalidArgument;
            t.next[try t.index(name)] = v;
        }
        return t.commit(nominal, current);
    }

    /// `set` with the names already resolved by `index`: `ks[i]` is the
    /// live slot of `values[i]`, so a point costs no name lookups.
    pub fn setAt(t: *Tuning, nominal: *const device.Circuit, current: core.Variants, ks: []const u32, values: []const f64) !Change {
        std.debug.assert(ks.len == values.len);
        @memcpy(t.next, t.values);
        for (ks, values) |k, v| {
            if (!std.math.isFinite(v)) return error.InvalidArgument;
            t.next[k] = v;
        }
        return t.commit(nominal, current);
    }

    /// The live slot `setAt` takes for `name`, compared as `set` does.
    /// `UnknownParameter` for a name that is not live.
    pub fn index(t: *const Tuning, name: []const u8) !u32 {
        const names: NameContext = .{ .fold = t.planner.nl.deck.dialect != .spectre };
        return t.slots.getContext(name, names) orelse error.UnknownParameter;
    }

    /// Name equality as the dialect spells it: case-insensitive but in Spectre.
    const NameContext = struct {
        fold: bool,

        pub fn hash(c: NameContext, name: []const u8) u64 {
            if (!c.fold) return std.hash.Wyhash.hash(0, name);
            var h: std.hash.Wyhash = .init(0);
            for (name) |ch| h.update(&.{std.ascii.toLower(ch)});
            return h.final();
        }

        pub fn eql(c: NameContext, a: []const u8, b: []const u8) bool {
            return if (c.fold) std.ascii.eqlIgnoreCase(a, b) else std.mem.eql(u8, a, b);
        }
    };

    /// Appends a row with no writes, labelled `refused`, and returns the
    /// table: the row of a point `set` refused, so its queries stay its
    /// own (`Problem.append_points`). The values and templates are untouched.
    pub fn refusedRow(t: *Tuning, current: core.Variants) !core.Variants {
        try t.reserveRow(current);
        t.closeRow("refused");
        return t.table();
    }

    /// Seeds the table and makes room for one more row, so `closeRow`
    /// cannot fail.
    fn reserveRow(t: *Tuning, current: core.Variants) !void {
        const sim = t.planner.sim;
        try t.seed(current);
        try t.labels.ensureUnusedCapacity(sim, 1);
        try t.temps.ensureUnusedCapacity(sim, 1);
        try t.axes.ensureUnusedCapacity(sim, 1);
        try t.starts.ensureUnusedCapacity(sim, 1);
    }

    /// Ends the row over the writes appended so far. Asserts `reserveRow` ran.
    fn closeRow(t: *Tuning, label: []const u8) void {
        t.labels.appendAssumeCapacity(label);
        t.temps.appendAssumeCapacity(null);
        t.axes.appendAssumeCapacity(0);
        t.starts.appendAssumeCapacity(@intCast(t.refs.items.len));
    }

    /// Copies the deck's table on the first row appended.
    fn seed(t: *Tuning, current: core.Variants) !void {
        if (t.seeded) return;
        const p = &t.planner;
        try t.labels.appendSlice(p.sim, current.labels);
        try t.temps.appendSlice(p.sim, current.temp_c);
        try t.axes.appendSlice(p.sim, current.axis);
        try t.starts.appendSlice(p.sim, if (current.starts.len == 0) &.{0} else current.starts);
        try t.refs.appendSlice(p.sim, current.refs);
        try t.write_values.appendSlice(p.sim, current.values);
        t.seeded = true;
    }

    fn table(t: *const Tuning) core.Variants {
        return .{
            .labels = t.labels.items,
            .temp_c = t.temps.items,
            .axis = t.axes.items,
            .starts = t.starts.items,
            .refs = t.refs.items,
            .values = t.write_values.items,
        };
    }

    /// `set` once `t.next` holds the point's values.
    fn commit(t: *Tuning, nominal: *const device.Circuit, current: core.Variants) !Change {
        std.debug.assert(t.template < t.templates.items.len);
        const p = &t.planner;
        const nl = p.nl;
        var label: std.ArrayList(u8) = .empty;
        for (nl.tunable, 0..) |k, i| try label.print(p.sim, "{s}{s}={d}", .{ if (i == 0) "" else ", ", nl.live.names[k], t.next[k] });
        if (t.planned != t.template) try t.open(t.template, nominal);
        // `nominal` is the Problem's own circuit; the one the planner opened
        // over in `init` was moved into it since.
        p.nominal = t.templates.items[t.planned].circuit orelse nominal;
        // The open row starts empty; a failed call may have left writes.
        for (p.write_refs.items) |r| p.row_at[r] = netlist.none;
        p.write_refs.clearRetainingCapacity();
        p.write_values.clearRetainingCapacity();
        const pt: variants.Point = .{ .label = label.items, .axis = 0, .live = t.next, .trial = t.trial };
        // Room for the row and a template first, so nothing fails after
        // the template list grows.
        try t.reserveRow(current);
        try t.templates.ensureUnusedCapacity(t.gpa, 1);
        var built: ?*device.Circuit = null;
        errdefer if (built != null) {
            // Not published: the next call re-opens the planner.
            t.templates.pop().?.deinit(t.gpa);
            t.planned = std.math.maxInt(u32);
        };
        if (try p.load(pt, label.items) == .circuit) {
            // An earlier template that holds the point's bins and kept values
            // takes it; otherwise the circuit built at it is a new template.
            const reuse: ?u32 = for (t.templates.items, 0..) |tm, k| {
                if (k != t.planned and tm.kept.holds(nl)) break @intCast(k);
            } else null;
            if (reuse) |k| {
                try t.open(k, nominal);
            } else {
                const c = try t.gpa.create(device.Circuit);
                var owned = true;
                errdefer if (owned) t.gpa.destroy(c);
                var scratch = std.heap.ArenaAllocator.init(t.gpa);
                defer scratch.deinit();
                c.* = try variant_circuit.circuitOf(p.lib, t.gpa, scratch.allocator(), nl);
                errdefer if (owned) c.deinit();
                if (!variant_circuit.sameTopology(t.templates.items[t.planned].circuit orelse nominal, c)) return p.refuse(label.items, "a parameter value that changes the topology");
                const kept: variant_circuit.Kept = try .of(scratch.allocator(), nl);
                t.templates.appendAssumeCapacity(try .of(t.gpa, c, t.next, kept));
                owned = false;
                built = c;
                try t.open(@intCast(t.templates.items.len - 1), nominal);
            }
            if (try p.load(pt, label.items) != .row) return p.refuse(label.items, "a parameter value that changes the topology");
        }
        try p.pointWrites(pt);
        try t.refs.appendSlice(p.sim, p.write_refs.items);
        try t.write_values.appendSlice(p.sim, p.write_values.items);
        t.template = t.planned;
        t.closeRow(label.items);
        @memcpy(t.values, t.next);
        return .{ .template = t.template, .built = built, .variants = t.table() };
    }
};

// ---- private ----

/// The optimizer options of `.model name OPT` [CR .MODEL; SA Ch.19, 27]
/// and its search: METHOD=BISECTION|PASSFAIL, else LEVEL 1 (LM), 2
/// (bisection) or 3 (pass/fail); METHOD supersedes LEVEL. Unknown keys are
/// refused; CENDIF and DYNACC (a speed-up through reduced accuracy) are
/// accepted and ignored.
fn optOptions(nl: *const Netlist, name: []const u8, line: []const u8) !struct { core.lm.Options, core.bisect.Method } {
    const model = nl.findModel(name) orelse {
        if (!@import("builtin").is_test) std.log.err("optimize: no .model {s} OPT: {s}", .{ name, line });
        return error.UnsupportedCard;
    };
    var o: core.lm.Options = .{};
    var method: ?core.bisect.Method = null;
    var level: core.bisect.Method = .lm;
    const Key = enum { itropt, relin, relout, close, cut, difsiz, parmin, grad, max, level, method, cendif, absin, absout, dynacc };
    const ok = std.mem.eql(u8, model.kind, "opt") and for (model.kv) |kv| {
        const key = std.meta.stringToEnum(Key, kv.key) orelse break false;
        if (key == .method) {
            if (kv.value != .name) break false;
            method = std.meta.stringToEnum(core.bisect.Method, kv.value.name) orelse break false;
            continue;
        }
        const v = switch (kv.value) {
            .num => |v| v,
            else => break false,
        };
        if (key == .dynacc) continue;
        if (!(v > 0) or !std.math.isFinite(v)) break false;
        switch (key) {
            .itropt => o.itropt = std.math.lossyCast(u32, v),
            .relin => o.relin = v,
            .relout => o.relout = v,
            .close => o.close = v,
            .cut => o.cut = v,
            .difsiz => o.difsiz = v,
            .parmin => o.parmin = v,
            .grad => o.grad = v,
            .max => o.max = v,
            .absin => o.absin = v,
            .absout => o.absout = v,
            .level => level = switch (std.math.lossyCast(u8, v)) {
                1 => .lm,
                2 => .bisection,
                3 => .passfail,
                else => break false,
            },
            .method, .cendif, .dynacc => {},
        }
    } else true;
    if (!ok or o.cut <= 1) {
        if (!@import("builtin").is_test) std.log.err("optimize: unsupported .model {s} OPT option: {s}", .{ name, line });
        return error.UnsupportedCard;
    }
    return .{ o, method orelse level };
}
