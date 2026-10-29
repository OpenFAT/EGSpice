//! Mismatch groups. In: the planner and the deck's `.variation`/`DEV`/`LOT`
//! rows, or its distribution sites. Out: `core.query.Variations`, the
//! groups `.dcmatch`, `.acmatch` and `.dcsens` read, one sigma per member.
//! Cites: HSPICE [CR] .VARIATION, .DCMATCH; #100, #130, #133.
//!
//! Contents:
//! - `pub fn variationTargets(p: *variant_planner.Planner) !variant_planner.Planner.Targets`
//! - `pub fn variations(p: *variant_planner.Planner) !core.query.Variations`
//! - `pub fn siteVariations(p: *variant_planner.Planner) !core.query.Variations`
//! - `pub fn spreadOf(p: *variant_planner.Planner, v: netlist.Variation, ord: u32, base: f64) !f64`

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
const variant_planner = @import("variant_planner.zig");
const variant_circuit = @import("variant_circuit.zig");
const Planner = variant_planner.Planner;
const Params = variant_params.Params;

/// The parameters each `.variation`/`DEV`/`LOT` row moves and the site
/// each draws from, CSR by row; found on the first trial.
pub fn variationTargets(p: *variant_planner.Planner) !variant_planner.Planner.Targets {
    if (p.targets) |t| return t;
    var by_card: std.StringHashMapUnmanaged(core.query.CardRef) = .empty;
    for (p.cards) |c| try by_card.put(p.scratch, c.name, c);
    var starts: std.ArrayList(u32) = .empty;
    var ords: std.ArrayList(u32) = .empty;
    var keys: std.ArrayList(u64) = .empty;
    try starts.append(p.scratch, 0);
    for (p.nl.deck.variations) |v| {
        const lot = netlist.siteKey(std.hash.Wyhash.hash(0, v.model), std.hash.Wyhash.hash(1, v.param));
        for (p.nl.order) |e| {
            const dev = p.nl.device(e);
            const hit = if (v.letter != 0) dev.kind == v.letter else if (dev.model) |m| std.mem.eql(u8, m.name, v.model) else false;
            if (!hit) continue;
            const card = by_card.get(dev.name) orelse continue;
            try ords.append(p.scratch, try p.paramRef(card.type, card.index, v.param));
            try keys.append(p.scratch, if (v.per_device) netlist.siteKey(std.hash.Wyhash.hash(2, dev.name), std.hash.Wyhash.hash(3, v.param)) else lot);
        }
        try starts.append(p.scratch, @intCast(ords.items.len));
    }
    p.targets = .{ .starts = starts.items, .ords = ords.items, .keys = keys.items };
    return p.targets.?;
}

/// The `.variation`/`DEV`/`LOT` rows as mismatch and sensitivity
/// groups: a per-device row is one group per device it reaches
/// (`<card>@<param>`), a per-model row one group over all of them
/// (`<model>@<param>`). A member's step is one sigma: the row's value
/// (times the nominal when relative), over sqrt(3) for a uniform spread
/// and the half range itself for `limit`.
pub fn variations(p: *variant_planner.Planner) !core.query.Variations {
    const targets = try variationTargets(p);
    const names = try core.query.CardNames.init(p.scratch, p.cards);
    var labels: std.ArrayList([]const u8) = .empty;
    var starts: std.ArrayList(u32) = .empty;
    var params: std.ArrayList(u32) = .empty;
    var sigmas: std.ArrayList(f64) = .empty;
    try starts.append(p.sim, 0);
    for (p.nl.deck.variations, 0..) |v, i| {
        const ords = targets.ords[targets.starts[i]..targets.starts[i + 1]];
        if (ords.len == 0) continue;
        for (ords) |ord| {
            const r = p.refs.at(ord);
            const spread = try spreadOf(p, v, ord, r.get());
            try params.append(p.sim, ord);
            try sigmas.append(p.sim, if (v.dist == .unif) spread / @sqrt(3.0) else spread);
            if (!v.per_device) continue;
            const card = names.get(r.type, r.index) orelse "?";
            try labels.append(p.sim, try std.fmt.allocPrint(p.sim, "{s}@{s}", .{ card, v.param }));
            try starts.append(p.sim, @intCast(params.items.len));
        }
        if (v.per_device) continue;
        try labels.append(p.sim, try std.fmt.allocPrint(p.sim, "{s}@{s}", .{ v.model, v.param }));
        try starts.append(p.sim, @intCast(params.items.len));
    }
    if (labels.items.len == 0) return .{};
    return .{ .labels = labels.items, .starts = starts.items, .params = params.items, .sigmas = sigmas.items };
}

/// The deck's own mismatch as groups, for a deck with no `.variation`
/// block: one group per distribution site the circuit reads, whose
/// members are the parameters a one-sigma draw there moves, each
/// scaled by its move (#100). This is how a PDK such as sky130 states
/// mismatch (`agauss` in its model cards). A site that moves no
/// parameter (sky130's `mc_mm_switch=0`) is no group, and no group at
/// all is an empty table, never the unit-variance fallback. The live
/// values are nominal again on return.
pub fn siteVariations(p: *variant_planner.Planner) !core.query.Variations {
    const live = p.nl.live;
    std.debug.assert(live.site_keys.len == live.site_ops.len);
    var seen: std.AutoHashMapUnmanaged(u64, void) = .empty;
    const names = try core.query.CardNames.init(p.scratch, p.cards);
    var labels: std.ArrayList([]const u8) = .empty;
    var starts: std.ArrayList(u32) = .empty;
    var params: std.ArrayList(u32) = .empty;
    var steps: std.ArrayList(f64) = .empty;
    try starts.append(p.sim, 0);
    if (p.fast == .untested) p.fast = try p.probeIdentity();
    // The steps are taken from every draw at its nominal, which under
    // `.option seed` is not the circuit's own state.
    _ = try p.nl.setLive(p.scratch, &p.stack, live.nominal, null);
    const base_slots = try p.scratch.alloc(f64, live.slots.len);
    for (base_slots, 0..) |*x, k| x.* = p.nl.slot(k).num;
    var base_arena = std.heap.ArenaAllocator.init(core.gpa);
    defer base_arena.deinit();
    var base_c = try variant_circuit.circuitOf(p.lib, base_arena.allocator(), base_arena.allocator(), p.nl);
    defer base_c.deinit();
    const base_refs = try variant_params.collect(base_arena.allocator(), &base_c);
    // Only a site inside a live slot reaches a device; the others sit in
    // model cards no instance binds (a PDK's unused bins).
    for (live.ops) |span| for (live.site_ops, live.site_keys) |op, key| {
        if (op < span.start or op >= span.start + span.len) continue;
        if ((try seen.getOrPut(p.scratch, key)).found_existing) continue;
        _ = try p.nl.setLive(p.scratch, &p.stack, live.nominal, variant_draws.OneSigma{ .site = key });
        const first = params.items.len;
        const crossed = for (base_slots, 0..) |x, k| {
            if (variant_circuit.crossesZero(x, p.nl.slot(k).num)) break true;
        } else false;
        // The probe's slot-to-parameter map gives the steps without a
        // rebuild, unless the draw takes a slot across zero (#130).
        if (p.fast == .identity and !crossed) {
            for (p.fast_slots, p.fast_refs) |k, ord| {
                const step = p.nl.slot(k).num - base_slots[k];
                if (step == 0) continue;
                try params.append(p.sim, ord);
                try steps.append(p.sim, step);
            }
        } else {
            var arena = std.heap.ArenaAllocator.init(core.gpa);
            defer arena.deinit();
            var c = try variant_circuit.circuitOf(p.lib, arena.allocator(), arena.allocator(), p.nl);
            defer c.deinit();
            if (!variant_circuit.sameTopology(p.nominal, &c)) return p.refuse("", "a one-sigma mismatch draw that changes the topology");
            const moved = try variant_params.collect(arena.allocator(), &c);
            var it_moved = moved.iterator();
            var it_base = base_refs.iterator();
            for (0..base_refs.len) |ord| {
                const step = it_moved.next().?.* - it_base.next().?.*;
                if (step == 0) continue;
                try params.append(p.sim, @intCast(ord));
                try steps.append(p.sim, step);
            }
        }
        if (params.items.len == first) continue;
        const r = p.refs.at(params.items[first]);
        const card = names.get(r.type, r.index) orelse "?";
        try labels.append(p.sim, try std.fmt.allocPrint(p.sim, "{s}@{s}", .{ card, r.param_name }));
        try starts.append(p.sim, @intCast(params.items.len));
    };
    _ = try p.nl.setLive(p.scratch, &p.stack, live.nominal, null);
    std.debug.assert(starts.items.len == labels.items.len + 1);
    return .{ .labels = labels.items, .starts = starts.items, .params = params.items, .sigmas = steps.items };
}

/// Row `v`'s sigma (or half range) at parameter `ord` whose value is
/// `base`: its number, or its expression over the device's `w`, `l`
/// and `m`, times `base` when relative. `UnsupportedCard` when the
/// expression is not finite there (a `w` or `l` the device lacks).
pub fn spreadOf(p: *variant_planner.Planner, v: netlist.Variation, ord: u32, base: f64) !f64 {
    var spread = v.value;
    if (v.sigma.len != 0) {
        const r = p.refs.at(ord);
        var at: [netlist.Variation.sigma_names.len]f64 = undefined;
        for (netlist.Variation.sigma_names, &at) |name, *x| x.* = instanceValue(p, r, name) orelse
            if (name[0] == 'm') instanceValue(p, r, "mfactor") orelse 1 else std.math.nan(f64);
        spread *= try expr.eval(p.scratch, &p.stack, p.nl.exprOps(v.sigma), p.nl.consts, &at, null);
        if (!std.math.isFinite(spread)) return p.refuse(v.param, "a .variation sigma that is not a finite number at this device");
    }
    return if (v.relative) spread * base else spread;
}

// ---- private ----

/// Instance parameter `name` of the device that owns `r`, or null.
fn instanceValue(p: *variant_planner.Planner, r: variant_params.Params.Ref, name: []const u8) ?f64 {
    const ord = p.paramRef(r.type, r.index, name) catch return null;
    return p.refs.get(ord);
}
