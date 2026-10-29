//! Device cards: the element line read into one hyperedge with its
//! positional and `key=value` rows; shunts and model bins after the walk.
//! Cites: ngspice inpcom.c, INPgetModBin; HSPICE [CR] .OPTION GSHUNT.
//!
//! Contents:
//! - `pub fn Methods(comptime S: type) type`
//! - `pub fn behaviouralResistor(r: *R, line: []const u8) Error!?[]const u8`
//! - `pub fn kindOf(r: *const R, row: usize) u8`
//! - `pub fn readDevice(r: *R, line: []const u8, kind: u8, frame: *const Frame) Error!void`
//! - `pub fn isModel(r: *R, name: []const u8) bool`
//! - `pub fn splitTerminals(r: *R) Error!void`
//! - `pub fn commit(r: *R, head: []const u8, letter: u8, frame: *const Frame) Error!void`
//! - `pub fn shunts(r: *R) Error!void`
//! - `pub fn option(r: *const R, name: []const u8) ?f64`
//! - `pub fn binOptions(r: *R) Error!void`
//! - `pub fn modelBins(r: *R, slots: []const Live.Slot, slot_scale: []f64) Error!void`

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
const VertexId = card_rows.VertexId;
const ground = card_rows.ground;
const none = card_rows.none;
const Error = card_rows.Error;
const Value = card_rows.Value;
const Kv = card_rows.Kv;
const Live = card_rows.Live;
const nameKey = netlist_view.nameKey;
const behavioural = card_table.behavioural;
const Frame = reader_frame.Frame;
const bTransformationWanted = card_table.bTransformationWanted;
const binBounds = card_table.binBounds;
const binHolds = netlist_view.binHolds;
const kvNumber = netlist_view.kvNumber;
const appendSpan = reader_frame.appendSpan;

/// The stage's `Reader` methods over dialect syntax `S`; `Reader(S)`
/// aliases each one as its own decl.
pub fn Methods(comptime S: type) type {
    const F = S.Split;
    return struct {
        const R = reader.Reader(S);
        const fixedPins = R.fixedPins;
        const mixedPort = R.mixedPort;

        /// ngspice inp_compat: an R card whose value reads `v(`, `i(`,
        /// `time`, `temper` or `hertz` is the B source
        /// `b<name> n1 n2 i = v(n1,n2)/(eq) tc1=.. tc2=.. reciproctc=1`, its
        /// temperature factor dividing the current; `m=` multiplies it
        /// (reciprocm=0). Null for any other R card. sky130's poly resistors
        /// (`rbody`) are written this way.
        pub fn behaviouralResistor(r: *R, line: []const u8) Error!?[]const u8 {
            var f = F.init(line);
            const head = f.next() orelse return null;
            const n1 = f.next() orelse return null;
            const n2 = f.next() orelse return null;
            if (!F.isWord(n1) or !F.isWord(n2) or !bTransformationWanted(f.rest())) return null;
            var eq: ?[]const u8 = null;
            var tc1: []const u8 = "0";
            var tc2: []const u8 = "0";
            var m: []const u8 = "1";
            while (f.next()) |t| {
                if (t[0] == '{') {
                    if (eq == null) eq = F.body(t);
                    continue;
                }
                if (!F.isWord(t) or !f.takeEq()) continue;
                const v = f.next() orelse return error.ParseError;
                const val = if (v[0] == '{') F.body(v) else v;
                if (std.mem.eql(u8, t, "r") or std.mem.eql(u8, t, "resistance")) {
                    eq = val;
                } else if (std.mem.eql(u8, t, "tc1")) {
                    tc1 = val;
                } else if (std.mem.eql(u8, t, "tc2")) {
                    tc2 = val;
                } else if (std.mem.eql(u8, t, "m")) {
                    m = val;
                }
            }
            // Without braces it is not ngspice's form: the plain R path rejects it.
            const e = eq orelse return null;
            return try r.arena.print("b{s} {s} {s} i={{v({s},{s})/({s})*({s})}} tc1={{{s}}} tc2={{{s}}} reciproctc=1", .{ head, n1, n2, n1, n2, e, m, tc1, tc2 });
        }

        /// The letter `kinds` gives row `row` of `lines`, 0 when its name
        /// carries its own.
        pub fn kindOf(r: *const R, row: usize) u8 {
            return if (row < r.kinds.len) r.kinds[row] else 0;
        }

        /// Reads device card `line`; `kind` is its letter when its name does
        /// not carry it (`kindOf`), else 0.
        pub fn readDevice(r: *R, line: []const u8, kind: u8, frame: *const Frame) Error!void {
            r.card = line;
            const arena = r.arena;
            const port_card = r.dialect == .hspice and std.ascii.toLower(line[0]) == 'p';
            var f = F.init(if (port_card) try r.portLine(line) else line);
            const head = f.next() orelse return error.ParseError;
            const was_in_card = r.in_card;
            r.in_card = true;
            defer r.in_card = was_in_card;
            if (r.monte) r.site_owner = nameKey(nameKey(0, frame.path orelse ""), head);
            r.site_ordinal = 0;
            if (!F.isWord(head) or !std.ascii.isAlphabetic(head[0])) return error.ParseError;
            // HSPICE's P element is a port: a V card that `port=` numbers
            // for `.lin` (`P1 in 0 port=1 z0=50`). A Spectre card's letter
            // is its master's, kept beside it.
            const letter = if (port_card) 'v' else if (kind != 0) kind else std.ascii.toLower(head[0]);
            // HSPICE reads these letters as IBIS buffers and lumped lines,
            // neither of which is built.
            if (r.dialect == .hspice and std.mem.indexOfScalar(u8, "bu", letter) != null)
                return r.unsupported(line, "unsupported HSPICE element");
            if (r.dialect == .hspice and (letter == 's' or letter == 'w')) return r.readDataElement(head, letter, &f, frame);
            if (letter == 'r' and r.dialect == .ngspice) if (try r.behaviouralResistor(line)) |b| return r.readDevice(b, 0, frame);
            if (std.mem.indexOfScalar(u8, "efgh", letter) != null) {
                var probe = f;
                _ = probe.next();
                _ = probe.next();
                var buf: [16]u8 = undefined;
                const at = probe.pos;
                if (probe.next()) |t| if (t.len <= buf.len) if (behavioural.get(std.ascii.lowerString(&buf, t))) |form| {
                    // `E1 a b VCVS c d 2`: the type word only repeats the letter.
                    // A run of them goes in one copy, so the re-read never recurses.
                    if (form == .kind) {
                        // A word for another letter (`E1 o 0 CCCS ...`) is
                        // refused, as ngspice does, not read as this one (#75).
                        const kinds = std.StaticStringMap(u8).initComptime(.{ .{ "vcvs", 'e' }, .{ "vccs", 'g' }, .{ "ccvs", 'h' }, .{ "cccs", 'f' } });
                        if (kinds.get(std.ascii.lowerString(&buf, t)) != letter) return r.unsupported(line, "a controlled-source type word for another letter");
                        while (probe.peek()) |u| {
                            if (u.len > buf.len or (behavioural.get(std.ascii.lowerString(&buf, u)) orelse .refused) != .kind) break;
                            _ = probe.next();
                        }
                        return r.readDevice(try std.mem.concat(r.arena, u8, &.{ line[0..at], " ", probe.line[probe.pos..] }), letter, frame);
                    }
                    // HSPICE: VCR and VCCAP are G-element forms only (#75).
                    if ((form == .vcr or form == .vccap) and letter != 'g') return r.unsupported(line, "VCR or VCCAP on a card other than G");
                    return r.behaviouralCard(line, head, letter, f, frame, form);
                };
            }
            r.nodes.clearRetainingCapacity();
            r.positional.clearRetainingCapacity();
            r.card_kv.clearRetainingCapacity();
            var kv_text: std.ArrayList([]const u8) = .empty;

            if ((f.nextByte() orelse return error.ParseError) == '(') {
                _ = f.next();
                while (true) {
                    const t = f.next() orelse return error.ParseError;
                    if (t[0] == ')') break;
                    if (t[0] == ',') continue;
                    if (!F.isWord(t)) return error.ParseError;
                    try r.nodes.append(arena, t);
                }
                const model = f.next() orelse return error.ParseError;
                if (!F.isWord(model)) return error.ParseError;
                if (letter != 'x') try r.refuseOperatorField(line, model);
                try r.positional.append(arena, try r.nameValue(model, frame, false));
            } else if (fixedPins(letter)) |fixed| {
                const count = if (port_card and mixedPort(f)) 3 else fixed;
                try r.nodes.ensureUnusedCapacity(arena, count);
                for (0..count) |_| {
                    const t = f.next() orelse return error.ParseError;
                    if (!F.isWord(t)) return error.ParseError;
                    r.nodes.appendAssumeCapacity(t);
                }
                if (letter == 'b') {
                    const out = f.next() orelse return error.ParseError;
                    const eq = f.next() orelse return error.ParseError;
                    if (!F.isWord(out) or eq[0] != '=') return error.ParseError;
                    try r.card_kv.append(arena, .{ .key = out, .value = try r.exprValue(try r.exprRest(&f), frame, false) });
                    // A braced expression may be followed by instance
                    // parameters (`i={...} tc1=1m reciproctc=1`).
                    while (f.next()) |t| {
                        if (!F.isWord(t) or !f.takeEq()) return error.ParseError;
                        try r.card_kv.append(arena, .{ .key = t, .value = try r.kvValue(&f, frame, false) });
                    }
                    return r.commit(head, letter, frame);
                }
            } else {
                // Variable-terminal cards end their word list with the model/subcircuit name.
                while (true) {
                    const save = f.pos;
                    const t = f.next() orelse break;
                    if (!F.isWord(t) or f.takeEq()) {
                        f.pos = save;
                        break;
                    }
                    try r.nodes.append(arena, t);
                }
                // ngspice reads `X1 a b sub params: p=1` (numparam's subcircuit call).
                if (letter == 'x' and r.dialect == .ngspice and r.nodes.items.len > 0 and std.mem.eql(u8, r.nodes.items[r.nodes.items.len - 1], "params:")) _ = r.nodes.pop();
                const model = r.nodes.pop() orelse return error.ParseError;
                if (letter != 'x') try r.refuseOperatorField(line, model);
                try r.positional.append(arena, try r.nameValue(model, frame, false));
            }

            // A third node makes a mixed-mode (balanced) port, not built.
            if (port_card) {
                var probe = f;
                if (probe.next()) |t| if (F.isWord(t) and !probe.takeEq()) return r.unsupported(line, "mixed-mode P element");
            }
            while (f.next()) |t| {
                if (t[0] == ',') {
                    continue;
                } else if (F.isWord(t) and f.takeEq()) {
                    const start = f.pos;
                    try r.card_kv.append(arena, .{ .key = t, .value = try r.kvValue(&f, frame, false) });
                    if (letter == 'x') try kv_text.append(arena, std.mem.trim(u8, f.line[start..f.pos], " \t"));
                } else {
                    if (letter != 'x') try r.refuseOperatorField(line, t);
                    try r.positional.append(arena, try r.valueAt(t, &f, frame, true, false));
                }
            }
            if (letter == 'x') return r.expand(head, frame, kv_text.items);
            // A D card's third node is its thermal node `tj` when a model
            // name follows it (ngspice-syntax.md §3.2, #121).
            if (letter == 'q' or letter == 'm' or letter == 'd') try r.splitTerminals();
            return r.commit(head, letter, frame);
        }

        pub fn isModel(r: *R, name: []const u8) bool {
            return r.model_ids.contains(name) or r.local_names.contains(name);
        }

        /// Q cards carry 3-5 terminals and M cards 3-7 against the fixed 3
        /// and 4 read, so the model name can land on either side of the
        /// node/positional split. Moves it to positional[0].
        pub fn splitTerminals(r: *R) Error!void {
            const nodes = &r.nodes;
            const pos = &r.positional;
            // Too few terminals: the model name was read as the last node.
            if (nodes.items.len > 0 and (pos.items.len == 0 or switch (pos.items[0]) {
                .name => |n| !r.isModel(n),
                else => true,
            })) {
                const last = nodes.items[nodes.items.len - 1];
                if (r.isModel(last)) {
                    _ = nodes.pop();
                    pos.shrinkRetainingCapacity(@min(pos.items.len, 3));
                    try pos.insert(r.arena, 0, .{ .name = last });
                    return;
                }
            }
            // Too many: extra terminals spilled into positional before the model.
            var model_at: ?usize = null;
            for (pos.items, 0..) |p, i| switch (p) {
                .name => |n| if (r.isModel(n)) {
                    model_at = i;
                },
                else => {},
            };
            const mi = model_at orelse return;
            if (mi == 0 or nodes.items.len + mi > 8 or pos.items.len - mi > 4) return;
            for (pos.items[0..mi]) |p| switch (p) {
                .name, .num => {},
                else => return,
            };
            for (pos.items[0..mi]) |p| try nodes.append(r.arena, switch (p) {
                .name => |n| n,
                .num => |v| try r.arena.print("{d}", .{v}),
                else => unreachable,
            });
            try pos.replaceRange(r.arena, 0, mi, &.{});
        }

        pub fn commit(r: *R, head: []const u8, letter: u8, frame: *const Frame) Error!void {
            const arena = r.arena;
            try r.pins.resize(arena, r.nodes.items.len);
            for (r.nodes.items, r.pins.items) |n, *pin| pin.* = try r.netOf(frame, n);
            const positional = try appendSpan(Value, arena, &r.values, r.positional.items);
            const kv = try appendSpan(Kv, arena, &r.kvs, r.card_kv.items);
            // After the spans: reading an instance model reuses the card scratch.
            // An R, C or L card may name its model after its value
            // (`R1 a b 1k rmod`, ngspice INP2R/INP2C/INP2L).
            const pv = r.positional.items;
            const model_at: ?usize = if (pv.len > 0 and pv[0] == .name) 0 else if (std.mem.indexOfScalar(u8, "rcl", letter) != null and pv.len > 1 and pv[0] == .num and pv[1] == .name) 1 else null;
            const model: u32 = if (model_at) |k| try r.modelRow(letter, pv[k].name, positional.start, frame) else none;
            // A behavioural E/F/G/H is built as a B card but keeps its own
            // letter in a flattened name, which is what its i() probes use.
            const name_letter = if (letter == 'b') std.ascii.toLower(head[0]) else letter;
            const name = try r.internName(if (frame.path) |path| try r.joined(&[_][]const u8{ &.{name_letter}, ".", path, ".", head }) else head);
            _ = r.hg.addEdge(arena, .{
                .kind = letter,
                .name = name,
                .model = model,
                .positional = positional,
                .kv = kv,
                .subckt_instance = frame.instance,
            }, r.pins.items) catch |err| return switch (err) {
                error.OutOfMemory => error.OutOfMemory,
                else => error.CircuitTooLarge,
            };
        }

        /// `.options gshunt=G cshunt=C` [CR Ch.3] and ngspice's `rshunt=R`:
        /// a conductance, capacitance or resistance from every net to
        /// ground, added as R and C cards named `r.gshunt.<net>`,
        /// `c.cshunt.<net>` and `r.rshunt.<net>`.
        pub fn shunts(r: *R) Error!void {
            const nets = r.hg.vertices.len;
            inline for (.{ .{ "gshunt", 'r' }, .{ "cshunt", 'c' }, .{ "rshunt", 'r' } }) |s| if (r.option(s[0])) |v| if (v > 0) {
                for (1..nets) |net| {
                    const value: Value = .{ .num = if (comptime std.mem.eql(u8, s[0], "gshunt")) 1 / v else v };
                    const pins = [2]VertexId{ .from(net), ground };
                    const net_name = r.pool.str(r.hg.vertices.items(.name)[net]);
                    const name = try r.internName(try r.arena.print("{c}.{s}.{s}", .{ s[1], s[0], net_name }));
                    _ = r.hg.addEdge(r.arena, .{
                        .kind = s[1],
                        .name = name,
                        .model = none,
                        .positional = try appendSpan(Value, r.arena, &r.values, &.{value}),
                        .kv = .{},
                        .subckt_instance = 0,
                    }, &pins) catch |err| return switch (err) {
                        error.OutOfMemory => error.OutOfMemory,
                        else => error.CircuitTooLarge,
                    };
                }
            };
        }

        /// The last `.options name=value` number in the deck, null when none
        /// sets it.
        pub fn option(r: *const R, name: []const u8) ?f64 {
            var out: ?f64 = null;
            for (r.config.items) |c| if (!c.temp) for (c.args, 0..) |a, i| {
                if (a == .name and std.ascii.eqlIgnoreCase(a.name, name) and i + 1 < c.args.len and c.args[i + 1] == .num) out = c.args[i + 1].num;
            };
            return out;
        }

        /// `.option scale` and `wnflag`, which bin selection reads.
        pub fn binOptions(r: *R) Error!void {
            r.wnflag = r.dialect != .ngspice;
            for (r.config.items) |c| {
                if (c.temp) continue;
                for (c.args, 0..) |arg, i| {
                    if (arg != .name) continue;
                    const is_scale = std.mem.eql(u8, arg.name, "scale");
                    if (!is_scale and !std.mem.eql(u8, arg.name, "wnflag")) continue;
                    if (i + 1 == c.args.len or c.args[i + 1] != .num) return error.ParseError;
                    if (is_scale) r.scale = c.args[i + 1].num else r.wnflag = c.args[i + 1].num != 0;
                }
            }
            if (!(r.scale > 0) or !std.math.isFinite(r.scale)) return error.ParseError;
        }

        /// `.option scale` and ngspice model binning (INPgetModBin): an M card
        /// naming `nm` takes the last-declared global `nm.<n>` whose L/W
        /// bounds hold it, within 1 nm. `modelRow` already binned the cards
        /// that name a subcircuit's own bins. Writes the factor each live
        /// slot of `slots` takes into `slot_scale`, parallel to it.
        pub fn modelBins(r: *R, slots: []const Live.Slot, slot_scale: []f64) Error!void {
            std.debug.assert(slots.len == slot_scale.len);
            const scale = r.scale;
            const wnflag = r.wnflag;
            const arena = r.arena;
            // Slot of each `kvs` row, `none` where it holds none, for the
            // scale and the bin check; empty when the deck has no slot.
            const slot_of = try arena.alloc(u32, if (slots.len == 0) 0 else r.kvs.items.len);
            @memset(slot_of, none);
            for (slots, 0..) |slot, k| if (slot.table == .kvs) {
                slot_of[slot.row] = @intCast(k);
            };
            const models = r.models.items;
            const kvs = r.kvs.items;
            var names: std.StringHashMapUnmanaged(u32) = .empty;
            const next = try arena.alloc(u32, models.len);
            @memset(next, none);
            const bounds = try arena.alloc([4]f64, models.len);
            // Global rows only: a subcircuit's models are its own.
            for (models, 0..) |m, i| if (r.model_ids.contains(m.name)) try names.put(arena, m.name, @intCast(i));
            for (models, 0..) |m, mi| {
                if (!r.model_ids.contains(m.name)) continue;
                const dot = std.mem.lastIndexOfScalar(u8, m.name, '.') orelse continue;
                _ = std.fmt.parseInt(u32, m.name[dot + 1 ..], 10) catch continue;
                bounds[mi] = binBounds(kvs[m.kv.start..][0..m.kv.len]) orelse continue;
                const entry = try names.getOrPut(arena, m.name[0..dot]);
                if (entry.found_existing) {
                    if (std.mem.eql(u8, models[entry.value_ptr.*].name, m.name[0..dot])) continue;
                    next[mi] = entry.value_ptr.*;
                }
                entry.value_ptr.* = @intCast(mi);
            }
            const lengths = std.StaticStringMap(u2).initComptime(.{
                .{ "l", 1 },  .{ "w", 1 },  .{ "pd", 1 }, .{ "ps", 1 }, .{ "sa", 1 }, .{ "sb", 1 }, .{ "sd", 1 },
                .{ "ad", 2 }, .{ "as", 2 },
            });
            const edges = r.hg.edges.slice();
            for (edges.items(.kind), edges.items(.positional), edges.items(.kv), edges.items(.model), 0..) |kind, ps, ks, *model, ei| {
                if (kind != 'm') continue;
                const kv = r.kvs.items[ks.start..][0..ks.len];
                var swept_geometry = false;
                for (kv, @as(usize, ks.start)..) |*item, row| {
                    const power = lengths.get(item.key) orelse continue;
                    if (item.value != .num) return error.ParseError;
                    const factor = if (power == 2) scale * scale else scale;
                    if (slot_of.len != 0 and slot_of[row] != none) {
                        const k = slot_of[row];
                        slot_scale[k] = factor;
                        swept_geometry = swept_geometry or std.mem.eql(u8, item.key, "l") or std.mem.eql(u8, item.key, "w");
                    }
                    if (scale != 1) item.value.num *= factor;
                }
                if (ps.len == 0) continue;
                const first = &r.values.items[ps.start];
                if (first.* != .name) continue;
                var bin = names.get(first.name) orelse continue;
                if (std.mem.eql(u8, first.name, models[bin].name)) continue;
                const l = kvNumber(kv, "l") orelse return error.ParseError;
                const use_nf = if (kvNumber(kv, "wnflag")) |flag| flag != 0 else wnflag;
                const nf = if (use_nf) kvNumber(kv, "nf") orelse 1 else 1;
                const w = (kvNumber(kv, "w") orelse return error.ParseError) / nf;
                // A swept or sampled l/w: every bin is a candidate a point
                // may pick (`Live.bins`).
                if (swept_geometry) {
                    var c = bin;
                    while (c != none) : (c = next[c]) {
                        try r.bin_rows.append(arena, r.model_ids.get(models[c].name) orelse none);
                        try r.bin_bounds.append(arena, bounds[c]);
                    }
                    try r.closeBins(@intCast(ei), use_nf);
                }
                while (bin != none) : (bin = next[bin]) {
                    if (binHolds(bounds[bin], l, w)) {
                        first.* = .{ .name = models[bin].name };
                        model.* = r.model_ids.get(models[bin].name) orelse none;
                        break;
                    }
                }
                if (bin == none) return error.ModelBinNotFound;
            }
        }
    };
}
