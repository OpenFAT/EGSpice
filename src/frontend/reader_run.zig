//! The reader spine: a deck's text through the three walks to a frozen
//! `Netlist`, and the `.ic`-style node values it publishes.
//! Cites: ngspice inp.c (walk order); #63, #65, #100, #164, #203.
//!
//! Contents:
//! - `pub fn Methods(comptime S: type) type`
//! - `pub fn run(arena: Allocator, src: []const u8, dialect: Dialect, live_params: []const []const u8) (Error || error{UnknownParameter})!Netlist`
//! - `pub fn nodeValues(arena: Allocator, cards_args: []const []const Value, nets: NetLookup) Error![]const Ic`

const std = @import("std");
const Allocator = std.mem.Allocator;
const lines = @import("lines.zig");
const spectre_rewrite = @import("spectre.zig");
const source = @import("source.zig");
const hypergraph = @import("hypergraph.zig");
const expr = @import("expr.zig");
const measure = @import("measure.zig");
const pattern = @import("pattern.zig");
const core = @import("core");
const Name = core.Name;
const InternPool = core.InternPool;
const assert = std.debug.assert;
const reader = @import("reader.zig");
const card_rows = @import("card_rows.zig");
const netlist_view = @import("netlist_view.zig");
const card_table = @import("card_table.zig");
const reader_frame = @import("reader_frame.zig");
const Dialect = card_rows.Dialect;
const EdgeId = card_rows.EdgeId;
const none = card_rows.none;
const Error = card_rows.Error;
const Value = card_rows.Value;
const Model = card_rows.Model;
const Ic = card_rows.Ic;
const Netlist = netlist_view.Netlist;
const nodeText = reader_frame.nodeText;
const resolve = reader_frame.resolve;
const Frame = reader_frame.Frame;
const ModelLine = reader_frame.ModelLine;
const top_level = reader_frame.top_level;
const seedOption = card_table.seedOption;
const analysisOf = card_table.analysisOf;
const setsKey = card_table.setsKey;

/// The stage's `Reader` methods over dialect syntax `S`; `Reader(S)`
/// aliases each one as its own decl.
pub fn Methods(comptime S: type) type {
    return struct {
        const R = reader.Reader(S);
        const NetLookup = R.NetLookup;
        const isStep = R.isStep;

        pub fn run(arena: Allocator, src: []const u8, dialect: Dialect, live_params: []const []const u8) (Error || error{UnknownParameter})!Netlist {
            var line_hint: usize = undefined;
            const text: []const u8 = if (S.fold_case) blk: {
                const buf = try arena.alloc(u8, src.len);
                line_hint = lines.normalize(buf, src);
                break :blk buf;
            } else blk: {
                line_hint = std.mem.countScalar(u8, src, '\n');
                break :blk src;
            };
            var r: R = try .init(arena, src, text, dialect);
            errdefer if (!r.logged and r.card.len > 0 and !@import("builtin").is_test)
                std.log.err("netlist: cannot read the card: {s}", .{r.written(r.card)});
            r.global_scopes = .{&r.globals};
            var split: S.Lines = .{ .rest = text, .arena = arena };
            var title: []const u8 = "";
            if (S.title_line) {
                // Case folding keeps byte positions, so the title is `src`'s.
                const nl = std.mem.indexOfScalar(u8, text, '\n') orelse text.len;
                title = std.mem.trim(u8, src[0..nl], " \t\r");
                split.rest = if (nl < text.len) text[nl + 1 ..] else "";
            }
            try r.lines.ensureTotalCapacityPrecise(arena, line_hint);
            // A line left blank (`\\` glued to a blank line, or a lone `\\` at the
            // end) is no card (#164, #38).
            while (try split.next()) |line| {
                if (std.mem.trim(u8, line, " \t").len > 0) try r.lines.append(arena, line);
            }
            // HSPICE forbids an exponent and a scale factor together (#203).
            if (dialect == .hspice) for (r.lines.items) |line| if (lines.hspice.exponentWithScale(line)) {
                r.card = line;
                return r.unsupported(line, "a number with both an exponent and a scale factor (`1e-6u`)");
            };
            // Spectre statements become the SPICE cards the walks read.
            if (dialect == .spectre) {
                const translated = try spectre_rewrite.translate(arena, r.lines.items);
                r.lines = .empty;
                try r.lines.appendSlice(arena, translated.lines);
                r.kinds = translated.kinds;
            }
            pattern.rewrite(arena, r.lines.items) catch |err| switch (err) {
                error.OutOfMemory => return error.OutOfMemory,
                error.InvalidPattern => return error.ParseError,
                error.UnsupportedPattern => return error.UnsupportedCard,
            };
            try r.hg.ensureTotalCapacity(arena, line_hint + 1, line_hint, 3 * line_hint);
            // Sized from the line count, so a flat deck never regrows them:
            // about one device name and one new net per card.
            const names_hint: u32 = @intCast(@min(2 * line_hint, none - 1));
            try r.pool.reserve(arena, names_hint);
            try r.net_of.ensureTotalCapacity(arena, names_hint + 1);
            try r.values.ensureTotalCapacity(arena, line_hint);
            // One `key=value` row per `=` of a flat deck.
            const eq_hint = std.mem.countScalar(u8, text, '=');
            try r.kvs.ensureTotalCapacity(arena, eq_hint);

            var top_devices: std.ArrayList(u32) = .empty;
            try top_devices.ensureTotalCapacityPrecise(arena, r.lines.items.len);
            var model_lines: std.ArrayList(ModelLine) = .empty;
            var directive_lines: std.ArrayList(u32) = .empty;
            try r.declarations(&top_devices, &model_lines, &directive_lines);
            // Flattened, a hierarchical deck holds more cards than lines:
            // sized for that too, so `expand` never regrows the tables (an
            // arena regrowth copies them and keeps the old block).
            const extra = try r.expandedExtra(top_devices.items);
            if (extra.cards != 0) {
                const flat = line_hint + extra.cards;
                try r.hg.ensureTotalCapacity(arena, flat + 1, flat, 3 * flat);
                const names: u32 = @intCast(@min(2 * flat, none - 1));
                try r.pool.reserve(arena, names);
                try r.net_of.ensureTotalCapacity(arena, names + 1);
                try r.values.ensureTotalCapacity(arena, flat);
                try r.kvs.ensureTotalCapacity(arena, eq_hint + extra.kvs);
            }
            const top: Frame = .{ .scopes = &r.global_scopes };
            // Three kinds of card make the models read next (a PDK's shared
            // bins) keep their distribution sites. A MONTE sweep draws them
            // per trial; the analysis cards are read after the models
            // (#100). ngspice draws agauss/gauss/unif/aunif on every run, and
            // a seed makes that one reproducible sample. `.dcmatch`,
            // `.acmatch` and `.dcsens` without a `.variation` block read the
            // mismatch off the calls (`variants.Planner.siteVariations`).
            // ponytail: a top-level `.param` read in walk 1 stays nominal;
            // the PDK distributions sit in model cards and subcircuit params.
            var monte_card = false;
            var seed = false;
            var mismatch = false;
            for (directive_lines.items) |i| {
                const line = r.lines.items[i];
                seed = seed or seedOption(line);
                const kind = analysisOf(line) orelse continue;
                monte_card = monte_card or setsKey(line, "monte");
                mismatch = mismatch or kind == .dcmatch or kind == .acmatch or kind == .dcsens;
            }
            seed = seed and dialect == .ngspice;
            r.draw_seed = seed and !monte_card;
            r.monte = monte_card or seed or mismatch;
            // The caller's live names, registered before any model,
            // directive or device reads them, so a card that cannot follow a
            // later value (an analysis, `.ic`, `.options`) is refused.
            const tunable = try arena.alloc(u32, live_params.len);
            for (live_params, tunable) |name, *k| {
                // The live table keeps the key; the caller's bytes may not outlive the call.
                const key = if (S.fold_case) try std.ascii.allocLowerString(arena, name) else try arena.dupe(u8, name);
                k.* = try r.liveParam(key) orelse {
                    if (!@import("builtin").is_test) std.log.err("netlist: live parameter '{s}': no global .param defines it", .{name});
                    return error.UnknownParameter;
                };
            }
            for (r.blocks.items) |b| try r.readBlock(b);
            // `.step` cards first, so their swept names are live before any
            // other directive reads them, wherever they sit in the deck (#63).
            for (directive_lines.items) |i| if (isStep(r.lines.items[i])) try r.readDirective(r.lines.items[i]);
            for (directive_lines.items) |i| if (!isStep(r.lines.items[i])) try r.readDirective(r.lines.items[i]);
            try r.foldGlobals();
            // Global models in line order, then each subcircuit's own, kept
            // together: `models_lo..models_hi` is one range per subcircuit,
            // nested definitions included. They follow the directives, so a
            // `.step` or `.dc` swept name is live when a model value reads it
            // (#65).
            for (model_lines.items) |ml| if (ml.owner == top_level) try r.readModel(r.lines.items[ml.line], &top, true);
            for (r.subckts.items, 0..) |*sub, id| for (model_lines.items) |ml| {
                if (ml.owner != id) continue;
                try r.readModel(r.lines.items[ml.line], &top, false);
                const row: u32 = @intCast(r.models.items.len - 1);
                const name = r.models.items[row].name;
                const shared = !ml.cond and !r.unresolved(r.models.items[row].kv) and !r.namesParam(r.models.items[row].kv, sub.*);
                if (sub.models_hi == 0) sub.models_lo = @intCast(r.local_models.items.len);
                try r.local_models.append(arena, .{ .name = name, .line = ml.line, .top = row, .shared = shared, .cond = ml.cond });
                sub.models_hi = @intCast(r.local_models.items.len);
                try r.local_names.put(arena, name, {});
            };
            try r.binOptions();

            for (top_devices.items) |i| try r.readDevice(r.lines.items[i], r.kindOf(i), &top);
            // The passes below read no one card.
            r.card = "";
            try r.shunts();
            try r.readMeasures();
            var live = try r.resolveLive();
            live.conds = try r.liveConds();
            const scale = try r.arena.alloc(f64, live.slots.len);
            @memset(scale, 1);
            try r.modelBins(live.slots, scale);
            live.scale = scale;
            live.bins = .{ .edges = r.bin_edges.items, .starts = r.bin_starts.items, .rows = r.bin_rows.items, .bounds = r.bin_bounds.items, .per_finger = r.bin_per_finger.items };

            const graph = try r.hg.finish(arena);
            const models = try arena.alloc(Model, r.models.items.len);
            for (models, r.models.items) |*m, row| m.* = .{ .name = row.name, .kind = row.kind, .kv = r.kvs.items[row.kv.start..][0..row.kv.len] };

            // Stable counting sort of devices by card letter.
            const kinds = graph.edges.items(.kind);
            var starts: [27]u32 = @splat(0);
            for (kinds) |k| starts[k - 'a' + 1] += 1;
            for (1..starts.len) |i| starts[i] += starts[i - 1];
            const order = try arena.alloc(EdgeId, kinds.len);
            var next: [26]u32 = starts[0..26].*;
            for (kinds, 0..) |k, e| {
                order[next[k - 'a']] = .from(e);
                next[k - 'a'] += 1;
            }

            const nets: NetLookup = .{ .pool = &r.pool, .net_of = r.net_of.items, .dialect = dialect };
            for (r.analyses.items) |*a| resolve(a, nets);

            return .{
                .pool = r.pool,
                .graph = graph,
                .order = order,
                .kind_starts = starts,
                .values = r.values.items,
                .kvs = r.kvs.items,
                .ops = r.ops.items,
                .consts = r.consts.items,
                .models = models,
                .model_ids = r.model_ids,
                .live = live,
                .tunable = tunable,
                .deck = .{
                    .title = title,
                    .dialect = dialect,
                    .analyses = r.analyses.items,
                    .config = r.config.items,
                    .ic = try nodeValues(arena, r.ic_cards.items, nets),
                    .nodeset = try nodeValues(arena, r.nodeset_cards.items, nets),
                    .saves = if (r.save_all) &.{} else r.saves.items,
                    .foreign = r.foreign.items,
                    .vpi_apps = r.vpi_apps.items,
                    .measures = r.measures.items,
                    .steps = r.steps.items,
                    .data = r.data.items,
                    .variations = r.variations.items,
                    .optimize = r.optimize,
                    .save_op = r.save_op,
                    .stims = r.stims.items,
                    .sample = r.sample,
                    .mosra = r.mosra,
                    .appendmodels = r.appendmodels.items,
                    .draw_seed = r.draw_seed,
                },
            };
        }

        /// `v(net)=value` pairs (or bare `net value`, as `.dcvolt` allows) of
        /// `.ic`-style cards; a net no card names is dropped.
        pub fn nodeValues(arena: Allocator, cards_args: []const []const Value, nets: NetLookup) Error![]const Ic {
            var out: std.ArrayList(Ic) = .empty;
            for (cards_args) |args| {
                var buf: [24]u8 = undefined;
                var i: usize = 0;
                while (i + 1 < args.len) : (i += 2) {
                    const name = switch (args[i]) {
                        .group => |g| if (std.ascii.eqlIgnoreCase(g.name(), "v") and g.args().len > 0) nodeText(g.args()[0], &buf) else null,
                        else => nodeText(args[i], &buf),
                    } orelse continue;
                    const value = switch (args[i + 1]) {
                        .num => |n| n,
                        else => continue,
                    };
                    const id = nets.node(name);
                    if (id == none or id == 0) continue;
                    try out.append(arena, .{ .net = .from(id), .value = value });
                }
            }
            return out.items;
        }
    };
}
