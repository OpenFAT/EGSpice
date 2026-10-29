//! The variant planner's state. In: a netlist, its nominal circuit and card
//! table. Out: variant rows (one parameter write per moved parameter) and
//! rebuild requests for points whose topology moves, decided by probing
//! the slot-to-parameter map once.
//! Cites: dev/analysis/variants.md.
//!
//! Contents:
//! - `pub const Planner`
//! - `pub const Targets`
//! - `pub const Fast`
//! - `pub fn init(lib: *const device.Library, sim: std.mem.Allocator, scratch: std.mem.Allocator, nl: *Netlist, nominal: *const device.Circuit, cards: []const core.query.CardRef, prefix: []const u8) !Planner`
//! - `pub fn refuse(_: *const Planner, line: []const u8, what: []const u8) error{UnsupportedCard}`
//! - `pub fn table(p: *Planner) !core.Variants`
//! - `pub fn add(p: *Planner, pt: variants.Point) !void`
//! - `pub fn probeIdentity(p: *Planner) !Fast`
//! - `pub fn paramRef(p: *Planner, t: core.DeviceType, index: u32, name: []const u8) !u32`

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
const variant_params = @import("variant_params.zig");
const variant_draws = @import("variant_draws.zig");
const variant_mismatch = @import("variant_mismatch.zig");
const variant_circuit = @import("variant_circuit.zig");
const variants = @import("variants.zig");

/// Planning state: the netlist, the nominal circuit's parameters, and the
/// rows and rebuilds so far.
pub const Planner = struct {
    lib: *const device.Library,
    /// The session arena: `table` copies the rows into it.
    sim: std.mem.Allocator,
    /// The parse arena: every planning table, which dies after the build.
    scratch: std.mem.Allocator,
    /// The netlist each point is set on in turn; nominal again when
    /// `plan` returns.
    nl: *Netlist,
    /// The circuit every row writes into; its parameters are `refs`.
    nominal: *const device.Circuit,
    cards: []const core.query.CardRef,
    /// Label prefix of an `.alter` run's rows; "" for the main run.
    prefix: []const u8 = "",
    /// Set while a card's own sweep is planned: rebuilt points keep only it.
    card: ?u32 = null,
    refs: variant_params.Params = .{},
    stack: std.ArrayList(expr.Val) = .empty,
    labels: std.ArrayList([]const u8) = .empty,
    temps: std.ArrayList(?f64) = .empty,
    axes: std.ArrayList(f64) = .empty,
    starts: std.ArrayList(u32) = .empty,
    write_refs: std.ArrayList(u32) = .empty,
    write_values: std.ArrayList(f64) = .empty,
    rebuilds: std.ArrayList(variants.Rebuild) = .empty,
    /// (type, parameter) to the ordinal of its instance 0, for card and
    /// variation writes.
    by_name: std.StringHashMapUnmanaged(u32) = .empty,
    /// Each live slot's value in the nominal circuit.
    nominal_slots: []const f64 = &.{},
    /// What the nominal circuit holds that no parameter write reaches. A
    /// run rebuilt at a point opens with that point's values here, not the
    /// deck's, so its own row does not read as moved and rebuild again.
    kept: variant_circuit.Kept = .{},
    fast: Fast = .untested,
    /// Rebuild every moved point; the differential test's oracle.
    force_rebuild: bool = false,
    /// The fast path's (slot, parameter) pairs, parallel.
    fast_slots: []const u32 = &.{},
    fast_refs: []const u32 = &.{},
    /// Per zero-crossing pattern points have met (`classes`), the probe's
    /// map there, or null when that pattern rebuilds every point.
    crossed: std.StringHashMapUnmanaged(?Map) = .empty,
    /// `classes` of the open point, one byte per live slot.
    key: []u8 = &.{},
    /// Per parameter, its write's index in the open row, or `none`.
    row_at: []u32 = &.{},
    targets: ?Targets = null,

    /// The parameters each `.variation` row moves, CSR by row.
    pub const Targets = struct { starts: []const u32, ords: []const u32, keys: []const u64 };

    /// Whether a point can take the probe's map instead of a rebuild.
    pub const Fast = enum { untested, identity, rebuild };

    /// A probe's map, in parameter order: `refs[i]` holds live slot
    /// `slots[i]` unchanged, or, where that is `none`, `values[i]`, which
    /// differs from the nominal circuit without following a slot.
    const Map = struct {
        refs: []const u32 = &.{},
        slots: []const u32 = &.{},
        values: []const f64 = &.{},
    };

    /// A planner over `nl`, whose nominal circuit is `nominal` with card
    /// table `cards`; `prefix` labels an `.alter` run's rows.
    pub fn init(lib: *const device.Library, sim: std.mem.Allocator, scratch: std.mem.Allocator, nl: *Netlist, nominal: *const device.Circuit, cards: []const core.query.CardRef, prefix: []const u8) !Planner {
        var p: Planner = .{ .lib = lib, .sim = sim, .scratch = scratch, .nl = nl, .nominal = nominal, .cards = cards, .prefix = prefix };
        p.refs = try variant_params.collect(scratch, nominal);
        p.row_at = try scratch.alloc(u32, p.refs.len);
        @memset(p.row_at, netlist.none);
        try p.starts.append(scratch, 0);
        const nominal_slots = try scratch.alloc(f64, nl.live.slots.len);
        for (nominal_slots, 0..) |*v, k| v.* = nl.slot(k).num;
        p.nominal_slots = nominal_slots;
        p.kept = try .of(scratch, nl);
        return p;
    }

    fn count(p: *const Planner) u32 {
        return @intCast(p.labels.items.len);
    }

    pub fn refuse(_: *const Planner, line: []const u8, what: []const u8) error{UnsupportedCard} {
        if (!@import("builtin").is_test) std.log.err("variants: {s}: {s}", .{ what, line });
        return error.UnsupportedCard;
    }

    /// HSPICE `OPTIMIZE=`, once per base point (`bases`: each `.step`
    /// point, or the nominal one). With S of them, rows 0..S are the
    /// optimized card's and S..2S every later card's: the base point's
    /// writes, which the optimum's replace once its optimizer has run
    /// (`Tuner`). Under `.step`, rows 2S..3S run the earlier cards at the
    /// initial values; without it they run nominal.
    fn planOptimize(p: *Planner) !variants.Plan {
        const deck = p.nl.deck;
        const bases = try p.basePoints();
        const n: u32 = @intCast(bases.len);
        const stepped = deck.steps.len != 0;
        const spans = try p.sim.alloc(analyses.Fanout.Span, deck.analyses.len);
        var card: ?u32 = null;
        for (deck.analyses, spans, 0..) |a, *span, ci| {
            if (a.sweep == .optimize) card = @intCast(ci) else if (a.sweep != .none) return p.refuse(a.line, "OPTIMIZE together with another SWEEP");
            const block: ?u32 = if (card) |c| @intFromBool(c != ci) else if (stepped) 2 else null;
            span.* = if (block) |b| .{ .first = b * n, .count = n, .nominal = false } else .{};
        }
        if (deck.alters.len != 0) return p.refuse(deck.analyses[card.?].line, "OPTIMIZE together with .alter");
        for (0..@as(u32, if (stepped) 3 else 2)) |block| for (bases) |pt| {
            var row = pt;
            if (block < 2) row.label = if (pt.label.len == 0)
                try std.fmt.allocPrint(p.scratch, "optimize={s}", .{deck.optimize.?.name})
            else
                try std.fmt.allocPrint(p.scratch, "optimize={s}, {s}", .{ deck.optimize.?.name, pt.label });
            try p.add(row);
        };
        if (p.rebuilds.items.len != 0) return p.refuse(deck.analyses[card.?].line, "OPTIMIZE over a .step point that changes the topology");
        return .{ .variants = try p.table(), .fanout = .{ .cards = spans } };
    }

    /// The points an optimization runs from: each `.step` point, or the
    /// nominal one.
    fn basePoints(p: *Planner) ![]const variants.Point {
        const steps = p.nl.deck.steps;
        var out: std.ArrayList(variants.Point) = .empty;
        if (steps.len == 0) {
            try out.append(p.scratch, .{ .label = "", .axis = 0, .live = p.nl.live.nominal });
        } else try p.grid(steps, try p.scratch.alloc(f64, steps.len), 0, &out);
        return out.items;
    }

    /// The rows so far, copied into the session arena.
    pub fn table(p: *Planner) !core.Variants {
        return .{
            .labels = try p.sim.dupe([]const u8, p.labels.items),
            .temp_c = try p.sim.dupe(?f64, p.temps.items),
            .axis = try p.sim.dupe(f64, p.axes.items),
            .starts = try p.sim.dupe(u32, p.starts.items),
            .refs = try p.sim.dupe(u32, p.write_refs.items),
            .values = try p.sim.dupe(f64, p.write_values.items),
        };
    }

    /// The `.step` cartesian product from card `depth` on, appended to
    /// `out`; the last card varies fastest.
    fn grid(p: *Planner, steps: []const netlist.Step, values: []f64, depth: usize, out: *std.ArrayList(variants.Point)) !void {
        if (depth == steps.len) return out.append(p.scratch, try p.stepPoint(steps, values));
        for (steps[depth].values) |v| {
            values[depth] = v;
            try p.grid(steps, values, depth + 1, out);
        }
    }

    /// The point setting `steps[i]`'s target to `values[i]`.
    fn stepPoint(p: *Planner, steps: []const netlist.Step, values: []const f64) !variants.Point {
        const targets = try p.scratch.alloc(netlist.StepTarget, steps.len);
        const names = try p.scratch.alloc([]const u8, steps.len);
        for (steps, targets, names) |st, *t, *name| {
            t.* = st.target;
            name.* = targetName(p.nl, st.target);
        }
        return p.pointOf(targets, names, values);
    }

    /// One `.data` row.
    fn addRow(p: *Planner, data: netlist.Data, row: []const f64) !void {
        try p.add(try p.pointOf(data.columns, data.labels, row));
    }

    /// The point setting `targets[i]` to `values[i]`, labelled
    /// `names[i]=values[i], ...`; the first value is its axis.
    fn pointOf(p: *Planner, targets: []const netlist.StepTarget, names: []const []const u8, values: []const f64) !variants.Point {
        const live = try p.scratch.dupe(f64, p.nl.live.nominal);
        var pt: variants.Point = .{ .label = "", .axis = values[0], .live = live };
        var label: std.ArrayList(u8) = .empty;
        var cards: std.ArrayList(variants.CardValue) = .empty;
        for (targets, names, values) |target, name, v| {
            if (label.items.len != 0) try label.appendSlice(p.scratch, ", ");
            try label.print(p.scratch, "{s}={d}", .{ name, v });
            switch (target) {
                .param => |k| live[k] = v,
                .temp => pt.temp = v,
                .card => |card| try cards.append(p.scratch, .{ .name = card, .value = v }),
            }
        }
        pt.label = label.items;
        pt.cards = cards.items;
        return pt;
    }

    fn addTrial(p: *Planner, sampler: *variant_draws.Sampler, trial: u32, index: u32) !void {
        try p.add(.{
            .label = try std.fmt.allocPrint(p.scratch, "monte={d}", .{trial}),
            .axis = @floatFromInt(trial),
            .live = p.nl.live.nominal,
            .trial = .{ .sampler = sampler, .trial = trial, .index = index },
        });
    }

    /// `.alter` run `k`. Without variants of its own, a run whose circuit
    /// matches the nominal one becomes a row; otherwise it is a rebuild
    /// that plans its own variants.
    fn addAlter(p: *Planner, text: []const u8, k: usize, own: bool) !void {
        const label = try std.fmt.allocPrint(p.sim, "alter={d}", .{k});
        const alt = try p.scratch.create(Netlist);
        alt.* = try netlist.parse(p.scratch, text, p.nl.deck.dialect);
        if (!own and variant_circuit.sameCards(p.nl, alt)) {
            var arena = std.heap.ArenaAllocator.init(core.gpa);
            defer arena.deinit();
            var c = try variant_circuit.circuitOf(p.lib, arena.allocator(), arena.allocator(), alt);
            defer c.deinit();
            if (variant_circuit.sameTopology(p.nominal, &c)) {
                try p.diff(try variant_params.collect(arena.allocator(), &c));
                return p.closeRow(label, @floatFromInt(k), null);
            }
        }
        try p.rebuilds.append(p.scratch, .{ .nl = alt, .run = .{ .prefix = label } });
    }

    /// Turns `pt` into a row, or into a rebuild when its live values
    /// change the topology or storage `ParamRef` cannot reach.
    pub fn add(p: *Planner, pt: variants.Point) !void {
        const label = if (p.prefix.len == 0) try p.sim.dupe(u8, pt.label) else try std.fmt.allocPrint(p.sim, "{s}, {s}", .{ p.prefix, pt.label });
        if (try p.load(pt, label) == .circuit) return p.addRun(label, pt);
        try p.pointWrites(pt);
        try p.closeRow(label, pt.axis, pt.temp);
    }

    const Load = enum { row, circuit };

    /// Sets `pt`'s live values (and the bins they pick), then appends to
    /// the open row the parameter writes that turn the nominal circuit into
    /// it. `.circuit` when only a circuit of its own can hold the point: a
    /// model bin moved, a kept expression reads a moved value, or the
    /// topology changed; the open row is then partial. `UnsupportedCard`
    /// (naming `label`) when a top-level `.if` would take another branch.
    fn load(p: *Planner, pt: variants.Point, label: []const u8) !Load {
        // ponytail: a point that flips a top-level `.if` is refused; re-parse
        // it as its own run if a deck needs branch selection per point.
        for (p.nl.live.conds) |span| {
            const ops = p.nl.exprOps(span);
            const at = try expr.eval(p.scratch, &p.stack, ops, p.nl.consts, pt.live, null);
            const nominal = try expr.eval(p.scratch, &p.stack, ops, p.nl.consts, p.nl.live.nominal, null);
            if ((at != 0) != (nominal != 0)) return p.refuse(label, "a top-level .if condition reads a swept parameter and selects another branch at this point");
        }
        try variant_draws.setPoint(p.scratch, &p.stack, p.nl, pt);
        if (p.key.len != p.nominal_slots.len) p.key = try p.scratch.alloc(u8, p.nominal_slots.len);
        var moved = false;
        var crossed = false;
        for (p.nominal_slots, p.key, 0..) |nominal, *class, k| {
            const x = p.nl.slot(k).num;
            moved = moved or x != nominal;
            class.* = variant_circuit.signClass(x);
            crossed = crossed or class.* != variant_circuit.signClass(nominal);
        }
        if (!p.kept.holds(p.nl)) return .circuit;
        if (!moved) return .row;
        if (p.fast == .untested) {
            p.fast = try p.probeIdentity();
            // The probe left the live slots perturbed.
            if (!p.force_rebuild) try variant_draws.setPoint(p.scratch, &p.stack, p.nl, pt);
        }
        if (p.fast == .identity and !crossed) {
            for (p.fast_slots, p.fast_refs) |k, r| try p.setMoved(r, p.nl.slot(k).num);
            return .row;
        }
        // A value that reaches or leaves zero may collapse a node or pick
        // a builder default: the first point of each crossing pattern
        // rebuilds and probes there, and the pattern's later points take
        // that probe's map (#214).
        var seen = false;
        if (p.fast == .identity) if (p.crossed.get(p.key)) |map| {
            if (map) |m| {
                try p.writeMap(m);
                return .row;
            }
            seen = true;
        };
        // The rebuild's tables die with it, so a long run of points does
        // not grow `scratch`.
        var arena = std.heap.ArenaAllocator.init(core.gpa);
        defer arena.deinit();
        var c = try variant_circuit.circuitOf(p.lib, arena.allocator(), arena.allocator(), p.nl);
        defer c.deinit();
        if (!variant_circuit.sameTopology(p.nominal, &c)) return .circuit;
        const at = try variant_params.collect(arena.allocator(), &c);
        try p.diff(at);
        if (p.fast == .identity and !seen) {
            const base = try p.scratch.alloc(f64, p.nominal_slots.len);
            for (base, 0..) |*x, k| x.* = p.nl.slot(k).num;
            const map = try p.probeAt(base, at);
            try variant_draws.setPoint(p.scratch, &p.stack, p.nl, pt);
            try p.crossed.put(p.scratch, try p.scratch.dupe(u8, p.key), map);
        }
        return .row;
    }

    /// Appends the writes `m` gives at the open point's live values.
    fn writeMap(p: *Planner, m: Map) !void {
        for (m.refs, m.slots, m.values) |r, k, x|
            try p.setMoved(r, if (k == netlist.none) x else p.nl.slot(k).num);
    }

    /// `set`, unless parameter `ord` holds `x` in the nominal circuit.
    fn setMoved(p: *Planner, ord: u32, x: f64) !void {
        if (@as(u64, @bitCast(x)) != @as(u64, @bitCast(p.refs.get(ord)))) try p.set(ord, x);
    }

    /// Finds, with one rebuild at perturbed live values, whether every
    /// parameter the live values reach holds one of them unchanged. Then a
    /// point writes its values straight to those parameters (the C2 fast
    /// path); otherwise each point rebinds the netlist and diffs. Unless
    /// `force_rebuild`, may leave the live slots perturbed.
    pub fn probeIdentity(p: *Planner) !Fast {
        if (p.force_rebuild) return .rebuild;
        const m = try p.probeAt(p.nominal_slots, p.refs) orelse return .rebuild;
        std.debug.assert(std.mem.indexOfScalar(u32, m.slots, netlist.none) == null);
        p.fast_slots = m.slots;
        p.fast_refs = m.refs;
        return .identity;
    }

    /// The map at live slot values `base`, where the circuit's parameters
    /// are `at`, from one rebuild at values perturbed from `base`; null
    /// when some parameter neither holds a slot's value unchanged nor
    /// stays put. Leaves the live slots at the perturbed values.
    fn probeAt(p: *Planner, base: []const f64, at: variant_params.Params) !?Map {
        std.debug.assert(at.len == p.refs.len);
        const probe_at = try p.scratch.alloc(f64, base.len);
        var by_bits: std.AutoHashMapUnmanaged(u64, u32) = .empty;
        defer by_bits.deinit(p.scratch);
        for (probe_at, base, 0..) |*x, b, k| {
            if (!std.math.isFinite(b)) return null;
            // A zero slot stays put and maps nothing: a point that moves it
            // crosses zero, which `load` probes on its own (#130).
            if (b == 0) {
                x.* = 0;
                continue;
            }
            // A distinct nearby value per slot, far from any rounding tie.
            x.* = b * (1 + @as(f64, @floatFromInt(k + 1)) * 0x1.0p-24);
            const gop = try by_bits.getOrPut(p.scratch, @bitCast(x.*));
            if (gop.found_existing) return null;
            gop.value_ptr.* = @intCast(k);
        }
        p.nl.setSlots(probe_at);
        var arena = std.heap.ArenaAllocator.init(core.gpa);
        defer arena.deinit();
        var c = try variant_circuit.circuitOf(p.lib, arena.allocator(), arena.allocator(), p.nl);
        defer c.deinit();
        if (!variant_circuit.sameTopology(p.nominal, &c)) return null;
        const moved = try variant_params.collect(arena.allocator(), &c);
        var refs: std.ArrayList(u32) = .empty;
        var slots: std.ArrayList(u32) = .empty;
        var values: std.ArrayList(f64) = .empty;
        var it_at = at.iterator();
        var it_moved = moved.iterator();
        var it_nominal = p.refs.iterator();
        for (0..at.len) |i| {
            const before = it_at.next().?.*;
            const after = it_moved.next().?.*;
            const nominal = it_nominal.next().?.*;
            var k: u32 = netlist.none;
            if (@as(u64, @bitCast(before)) != @as(u64, @bitCast(after))) {
                k = by_bits.get(@bitCast(after)) orelse return null;
                if (@as(u64, @bitCast(before)) != @as(u64, @bitCast(base[k]))) return null;
            } else if (@as(u64, @bitCast(before)) == @as(u64, @bitCast(nominal))) continue;
            try refs.append(p.scratch, @intCast(i));
            try slots.append(p.scratch, k);
            try values.append(p.scratch, before);
        }
        return .{ .refs = refs.items, .slots = slots.items, .values = values.items };
    }

    fn closeRow(p: *Planner, label: []const u8, axis: f64, temp: ?f64) !void {
        for (p.write_refs.items[p.starts.items[p.starts.items.len - 1]..]) |r| p.row_at[r] = netlist.none;
        try p.labels.append(p.scratch, label);
        try p.axes.append(p.scratch, axis);
        try p.temps.append(p.scratch, temp);
        try p.starts.append(p.scratch, @intCast(p.write_refs.items.len));
    }

    /// A point with its own circuit, rebuilt from the netlist at `pt`.
    fn addRun(p: *Planner, label: []const u8, pt: variants.Point) !void {
        try p.rebuilds.append(p.scratch, .{ .nl = p.nl, .run = .{ .prefix = label, .point = pt, .card = p.card } });
    }

    /// Appends a write for every parameter whose value in `moved`, collected
    /// in `p.refs` order, differs from the nominal circuit's.
    fn diff(p: *Planner, moved: variant_params.Params) !void {
        std.debug.assert(moved.len == p.refs.len);
        var it_moved = moved.iterator();
        var it_ref = p.refs.iterator();
        for (0..p.refs.len) |i| {
            const x = it_moved.next().?.*;
            if (@as(u64, @bitCast(it_ref.next().?.*)) == @as(u64, @bitCast(x))) continue;
            try p.set(@intCast(i), x);
        }
    }

    /// Card values and Monte Carlo variations of `pt`, over the open
    /// row's writes.
    fn pointWrites(p: *Planner, pt: variants.Point) !void {
        for (pt.cards) |cv| try p.set(try p.cardRef(cv.name), cv.value);
        const trial = pt.trial orelse return;
        const targets = try variant_mismatch.variationTargets(p);
        for (p.nl.deck.variations, 0..) |v, i| {
            for (targets.ords[targets.starts[i]..targets.starts[i + 1]], targets.keys[targets.starts[i]..targets.starts[i + 1]]) |ord, key| {
                const base = p.current(ord);
                const u = trial.uniform(key);
                const spread = try variant_mismatch.spreadOf(p, v, ord, base);
                const x = switch (v.dist) {
                    .gauss => base + spread * variant_draws.normalQuantile(u),
                    .unif => base + spread * (2 * u - 1),
                    .limit => base + (if (u < 0.5) -spread else spread),
                };
                try p.set(ord, x);
            }
        }
    }

    /// The value parameter `ord` has in the open row.
    fn current(p: *Planner, ord: u32) f64 {
        const at = p.row_at[ord];
        return if (at != netlist.none) p.write_values.items[at] else p.refs.get(ord);
    }

    /// Writes `value` to `ord` in the open row, once per parameter.
    fn set(p: *Planner, ord: u32, value: f64) !void {
        if (p.row_at[ord] != netlist.none) {
            p.write_values.items[p.row_at[ord]] = value;
            return;
        }
        p.row_at[ord] = @intCast(p.write_refs.items.len);
        try p.write_refs.append(p.scratch, ord);
        try p.write_values.append(p.scratch, value);
    }

    /// The ordinal of card `name`'s primary value, as `.dc` sweeps it.
    fn cardRef(p: *Planner, name: []const u8) !u32 {
        const target = try analyses.dcTarget(&.{.{ .name = name }}, 0, p.cards);
        const d = switch (target) {
            .device => |d| d,
            .temp => unreachable,
        };
        return p.paramRef(d.type, d.index, d.param_name);
    }

    pub fn paramRef(p: *Planner, t: core.DeviceType, index: u32, name: []const u8) !u32 {
        if (p.by_name.count() == 0) try p.refs.indexNames(p.scratch, &p.by_name);
        const first = p.by_name.get(try variant_circuit.refKey(p.scratch, t, name)) orelse return error.UnknownParameter;
        const ord = first + index;
        if (ord >= p.refs.len) return error.UnknownParameter;
        const r = p.refs.at(ord);
        if (r.index != index or r.type != t) return error.UnknownParameter;
        return ord;
    }
};

// ---- private ----

fn targetName(nl: *const Netlist, t: netlist.StepTarget) []const u8 {
    return switch (t) {
        .param => |k| nl.live.names[k],
        .temp => "temp",
        .card => |name| name,
    };
}
