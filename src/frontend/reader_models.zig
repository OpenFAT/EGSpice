//! Model cards: global and subcircuit-local `.model` rows, binning bounds,
//! and the per-instance re-reads a local model needs.
//! Cites: ngspice subckt.c modtranslate, INPgetModBin.
//!
//! Contents:
//! - `pub fn Methods(comptime S: type) type`
//! - `pub fn readModel(r: *R, line: []const u8, frame: *const Frame, global: bool) Error!void`
//! - `pub fn unresolved(r: *const R, span: Span) bool`
//! - `pub fn namesParam(r: *const R, span: Span, sub: Subckt) bool`
//! - `pub fn modelRow(r: *R, letter: u8, name: []const u8, pos: u32, frame: *const Frame) Error!u32`
//! - `pub fn binsEnd(r: *const R) u32`
//! - `pub fn closeBins(r: *R, e: u32, per_finger: bool) Error!void`
//! - `pub fn cardLength(r: *R, kv: []const Kv, key: []const u8) Error!?Geometry`
//! - `pub fn localRow(r: *R, lm: usize, frame: *const Frame) Error!u32`
//! - `pub fn localKey(r: *R, lm: usize, frame: *const Frame) Error!?[]const u8`
//! - `pub fn localReads(r: *R, text: []const u8) Error!LocalReads`
//! - `pub fn sameValues(a: []const Kv, b: []const Kv) bool`

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
const none = card_rows.none;
const Error = card_rows.Error;
const Span = card_rows.Span;
const Value = card_rows.Value;
const Kv = card_rows.Kv;
const data_file_keys = card_table.data_file_keys;
const nameKey = netlist_view.nameKey;
const Frame = reader_frame.Frame;
const Subckt = reader_frame.Subckt;
const isBinOf = card_table.isBinOf;
const binBounds = card_table.binBounds;
const binHolds = netlist_view.binHolds;
const kvNumber = netlist_view.kvNumber;
const readsLive = card_table.readsLive;
const appendSpan = reader_frame.appendSpan;

/// The stage's `Reader` methods over dialect syntax `S`; `Reader(S)`
/// aliases each one as its own decl.
pub fn Methods(comptime S: type) type {
    const F = S.Split;
    return struct {
        const R = reader.Reader(S);
        const Geometry = R.Geometry;
        const LocalReads = R.LocalReads;

        /// Reads a `.model` card under `frame` into a new row; `global`
        /// makes it the row its name resolves to outside subcircuits.
        pub fn readModel(r: *R, line: []const u8, frame: *const Frame, global: bool) Error!void {
            r.card = line;
            var f = F.init(line);
            _ = f.next();
            const name = f.next() orelse return error.ParseError;
            const kind = f.next() orelse return error.ParseError;
            if (!F.isWord(name) or !F.isWord(kind)) return error.ParseError;
            const was_in_card = r.in_card;
            r.in_card = true;
            defer r.in_card = was_in_card;
            // Site keys are read only under Monte Carlo (`emit`).
            if (r.monte) r.site_owner = nameKey(nameKey(0, frame.path orelse ""), name);
            r.site_ordinal = 0;
            const matrix_keys = std.StaticStringMap(void).initComptime(.{ .{ "r", {} }, .{ "l", {} }, .{ "g", {} }, .{ "c", {} } });
            // CPL matrices are blank-separated; a negative entry is not a subtraction.
            const cpl = std.mem.eql(u8, kind, "cpl");
            // So are an HSPICE W model's lower-triangle matrices.
            const wline = std.mem.eql(u8, kind, "w");
            const w_matrix_keys = std.StaticStringMap(void).initComptime(.{
                .{ "lo", {} }, .{ "co", {} }, .{ "ro", {} }, .{ "go", {} }, .{ "rs", {} }, .{ "gd", {} },
                .{ "l", {} },  .{ "c", {} },  .{ "r", {} },  .{ "g", {} },
            });
            r.card_kv.clearRetainingCapacity();
            while (f.next()) |t| {
                if (t[0] == '(' or t[0] == ')' or t[0] == ',') continue;
                if (!F.isWord(t)) return error.ParseError;
                if (f.takeEq()) {
                    // DEV/LOT tolerances apply to the value before them;
                    // the global-scope read records them once.
                    const prev = if (r.card_kv.items.len > 0) r.card_kv.items[r.card_kv.items.len - 1].key else "";
                    if (try r.devLot(t, prev, name, &f, line)) |v| {
                        if (global) try r.variations.append(r.arena, v);
                        continue;
                    }
                    if (data_file_keys.has(t)) {
                        try r.card_kv.append(r.arena, .{ .key = t, .value = .{ .name = try r.dataPath(&f) } });
                        continue;
                    }
                    const value = if ((cpl and matrix_keys.has(t)) or (wline and w_matrix_keys.has(t))) try r.readValue(&f, frame, true, true) else try r.kvValue(&f, frame, true);
                    try r.card_kv.append(r.arena, .{ .key = t, .value = value });
                } else {
                    const value: Value = if (S.parseNum(t)) |n| .{ .num = n } else try r.nameValue(t, frame, true);
                    try r.card_kv.append(r.arena, .{ .key = "", .value = value });
                }
            }
            const span = try appendSpan(Kv, r.arena, &r.kvs, r.card_kv.items);
            if (global) {
                const gop = try r.model_ids.getOrPut(r.arena, name);
                if (!gop.found_existing) gop.value_ptr.* = @intCast(r.models.items.len);
            }
            try r.models.append(r.arena, .{ .name = name, .kind = kind, .kv = span });
        }

        /// True when a value in `span` did not fold: it names a parameter the
        /// global scope lacks, which no device could bind.
        pub fn unresolved(r: *const R, span: Span) bool {
            for (r.kvs.items[span.start..][0..span.len]) |kv| if (kv.value == .expr) return true;
            return false;
        }

        /// True when a value in `span` is a bare name `sub` declares as a
        /// parameter: read at global scope it stayed a name, but each
        /// instance substitutes its own value.
        pub fn namesParam(r: *const R, span: Span, sub: Subckt) bool {
            for (r.kvs.items[span.start..][0..span.len]) |kv| if (kv.value == .name) {
                for (sub.defaults) |d| if (std.mem.eql(u8, d.key, kv.value.name)) return true;
            };
            return false;
        }

        /// The model row card `letter` inside `frame` names: a `.model` of
        /// the frame's own subcircuit (for an M card, the `name.N` bin the
        /// card's L and W select, as ngspice bins the renamed subcircuit
        /// models), else the first global row. Rewrites the card's model
        /// name, `r.values[pos]`, to the bin it picks.
        pub fn modelRow(r: *R, letter: u8, name: []const u8, pos: u32, frame: *const Frame) Error!u32 {
            const sub = r.subckts.items[frame.sub orelse return r.model_ids.get(name) orelse none];
            for (r.instance_models.items[frame.models_mark..]) |m| if (std.mem.eql(u8, m.name, name)) return m.row;
            const locals = r.local_models.items[sub.models_lo..sub.models_hi];
            for (locals, sub.models_lo..) |m, lm| if (!m.cond and std.mem.eql(u8, m.name, name)) return r.localRow(lm, frame);
            if (letter != 'm') return r.model_ids.get(name) orelse none;
            // ngspice prepends model declarations (inpmkmod.c): the last
            // matching bin wins, as in `modelBins`.
            // Read before any `localRow`, which reuses the card scratch.
            const kv = r.card_kv.items;
            const lg = try r.cardLength(kv, "l") orelse return r.model_ids.get(name) orelse none;
            const use_nf = if (kvNumber(kv, "wnflag")) |flag| flag != 0 else r.wnflag;
            const nfg: Geometry = if (use_nf) try r.cardLength(kv, "nf") orelse .{ .num = 1 } else .{ .num = 1 };
            const wg = try r.cardLength(kv, "w") orelse return r.model_ids.get(name) orelse none;
            const l = lg.num * r.scale;
            const w = wg.num * r.scale / nfg.num;
            const w_live = wg.live or nfg.live;
            var picked: ?u32 = null;
            var k = locals.len;
            while (k > 0) {
                k -= 1;
                const m = locals[k];
                if (m.cond or !isBinOf(m.name, name)) continue;
                // ponytail: bounds from the global-scope read; a bin whose
                // bounds need instance parameters is never picked.
                const b = binBounds(r.kvs.items[r.models.items[m.top].kv.start..][0..r.models.items[m.top].kv.len]) orelse continue;
                if (!lg.live and !w_live) {
                    if (!binHolds(b, l, w)) continue;
                    r.values.items[pos] = .{ .name = m.name };
                    return r.localRow(sub.models_lo + k, frame);
                }
                // Live geometry: every bin the fixed dimension allows is a
                // candidate a point may pick (`Live.bins`), read now under
                // this instance's parameters.
                if (!lg.live and !binHolds(b, l, b[2])) continue;
                if (!w_live and !binHolds(b, b[0], w)) continue;
                const row = try r.localRow(sub.models_lo + k, frame);
                try r.bin_rows.append(r.arena, row);
                try r.bin_bounds.append(r.arena, b);
                if (picked == null and binHolds(b, l, w)) {
                    picked = row;
                    r.values.items[pos] = .{ .name = m.name };
                }
            }
            if (r.bin_rows.items.len != r.binsEnd()) try r.closeBins(@intCast(r.hg.edges.len), use_nf);
            return picked orelse r.model_ids.get(name) orelse none;
        }

        /// One past the last candidate of the cards `Live.bins` holds.
        pub fn binsEnd(r: *const R) u32 {
            const starts = r.bin_starts.items;
            return if (starts.len == 0) 0 else starts[starts.len - 1];
        }

        /// Records the candidates appended since `binsEnd` as edge `e`'s.
        /// `starts` opens on the first card, so a deck without one
        /// allocates nothing here.
        pub fn closeBins(r: *R, e: u32, per_finger: bool) Error!void {
            std.debug.assert(r.bin_rows.items.len > r.binsEnd());
            if (r.bin_starts.items.len == 0) try r.bin_starts.append(r.arena, 0);
            try r.bin_edges.append(r.arena, e);
            try r.bin_per_finger.append(r.arena, per_finger);
            try r.bin_starts.append(r.arena, @intCast(r.bin_rows.items.len));
        }

        /// Card value `key` as a number: as written, or, when it reads a
        /// live name, folded at the nominal values (`live` set). Null when
        /// the card lacks it or it does not fold.
        pub fn cardLength(r: *R, kv: []const Kv, key: []const u8) Error!?Geometry {
            for (kv) |item| if (std.mem.eql(u8, item.key, key)) switch (item.value) {
                .num => |n| return .{ .num = n },
                .expr => |span| {
                    const ops = r.ops.items[span.start..][0..span.len];
                    if (!readsLive(ops)) return null;
                    const v = try expr.fold(r.arena, &r.stack, ops, r.consts.items, false, r.live_nominal.items);
                    return if (v.known) .{ .num = v.num, .live = true } else null;
                },
                else => return null,
            };
            return null;
        }

        /// The model row of `local_models` row `lm` for the instance `frame`
        /// expands: the shared global-scope row, else one read under the
        /// instance's parameters on first use.
        pub fn localRow(r: *R, lm: usize, frame: *const Frame) Error!u32 {
            const m = r.local_models.items[lm];
            if (m.shared) return m.top;
            const line = m.line;
            const name = m.name;
            if (r.variants.len != r.local_models.items.len) {
                std.debug.assert(r.variants.len == 0);
                r.variants = try r.arena.alloc(std.ArrayList(u32), r.local_models.items.len);
                @memset(r.variants, .empty);
                r.local_reads = try r.arena.alloc(?LocalReads, r.local_models.items.len);
                @memset(r.local_reads, null);
            }
            // A PDK reads each subcircuit-local card once per instance
            // under the instance's parameters (78% of `Problem.init` on
            // c7552_ihp); equal inputs fold to equal rows.
            const memo = try r.localKey(lm, frame);
            if (memo) |key| if (r.local_memo.get(key)) |row| {
                try r.instance_models.append(r.arena, .{ .name = name, .row = row });
                return row;
            };
            const kv_mark = r.kvs.items.len;
            try r.readModel(r.lines.items[line], frame, false);
            var row: u32 = @intCast(r.models.items.len - 1);
            const seen = &r.variants[lm];
            const fresh = r.kvs.items[kv_mark..];
            for (seen.items) |old| {
                const kv = r.models.items[old].kv;
                if (!sameValues(r.kvs.items[kv.start..][0..kv.len], fresh)) continue;
                r.models.shrinkRetainingCapacity(row);
                r.kvs.shrinkRetainingCapacity(kv_mark);
                row = old;
                break;
            } else try seen.append(r.arena, row);
            if (memo) |key| try r.local_memo.put(r.arena, try r.arena.dupe(u8, key), row);
            try r.instance_models.append(r.arena, .{ .name = name, .row = row });
            return row;
        }

        /// The memo key of local model `lm` read in `frame`: `lm`, then for
        /// each name its line can read, every scope's entry for it (all
        /// scopes, not only the innermost, so any lookup order the fold uses
        /// is covered), following text entries' own names. Null when the
        /// line draws a statistical value, which differs per instance site.
        /// Borrows `r.local_key` until the next call.
        pub fn localKey(r: *R, lm: usize, frame: *const Frame) Error!?[]const u8 {
            const reads = r.local_reads[lm] orelse blk: {
                const got = try r.localReads(r.lines.items[r.local_models.items[lm].line]);
                r.local_reads[lm] = got;
                break :blk got;
            };
            const names = switch (reads) {
                .stochastic => return null,
                .names => |ns| ns,
            };
            const key = &r.local_key;
            key.clearRetainingCapacity();
            try key.appendSlice(r.arena, std.mem.asBytes(&@as(u64, lm)));
            // Names reached through text entries, breadth first, each once.
            var queue: std.ArrayList([]const u8) = .empty;
            defer queue.deinit(r.arena);
            var seen: std.StringHashMapUnmanaged(void) = .empty;
            defer seen.deinit(r.arena);
            for (names) |n| if (!(try seen.getOrPut(r.arena, n)).found_existing) try queue.append(r.arena, n);
            var i: usize = 0;
            while (i < queue.items.len) : (i += 1) {
                const n = queue.items[i];
                for (frame.scopes) |scope| {
                    const e = scope.get(n) orelse {
                        try key.append(r.arena, 0);
                        continue;
                    };
                    switch (e) {
                        .num => |v| {
                            try key.append(r.arena, 1);
                            try key.appendSlice(r.arena, std.mem.asBytes(&v));
                        },
                        .text => |t| {
                            try key.append(r.arena, 2);
                            try key.appendSlice(r.arena, std.mem.asBytes(&@as(u64, t.len)));
                            try key.appendSlice(r.arena, t);
                            switch (try r.localReads(t)) {
                                .stochastic => return null,
                                .names => |more| for (more) |m| if (!(try seen.getOrPut(r.arena, m)).found_existing) try queue.append(r.arena, m),
                            }
                        },
                    }
                }
            }
            return key.items;
        }

        /// The identifiers `text` holds, `.func` bodies followed, or
        /// `.stochastic` when it calls a distribution. Keys and function
        /// names count too: a superset only makes the memo key stricter.
        pub fn localReads(r: *R, text: []const u8) Error!LocalReads {
            var out: std.ArrayList([]const u8) = .empty;
            var todo: std.ArrayList([]const u8) = .empty;
            defer todo.deinit(r.arena);
            try todo.append(r.arena, text);
            var bodies: usize = 0;
            while (todo.pop()) |t| {
                var k: usize = 0;
                while (k < t.len) {
                    const c = t[k];
                    if (std.ascii.isDigit(c) or c == '.') {
                        // A number with its exponent and scale suffix.
                        while (k < t.len and (std.ascii.isAlphanumeric(t[k]) or t[k] == '.' or t[k] == '_' or
                            ((t[k] == '+' or t[k] == '-') and k > 0 and (t[k - 1] == 'e' or t[k - 1] == 'E')))) k += 1;
                        continue;
                    }
                    if (!(std.ascii.isAlphabetic(c) or c == '_')) {
                        k += 1;
                        continue;
                    }
                    const start = k;
                    while (k < t.len and (std.ascii.isAlphanumeric(t[k]) or t[k] == '_')) k += 1;
                    const ident = t[start..k];
                    if (expr.isDistributionName(ident)) return .stochastic;
                    if (r.funcs.get(ident)) |func| {
                        bodies += 1;
                        // A recursive `.func` chain: give up on the memo.
                        if (bodies > 64) return .stochastic;
                        try todo.append(r.arena, func.body);
                    }
                    try out.append(r.arena, ident);
                }
            }
            return .{ .names = out.items };
        }

        /// Equal keys and values; an unfolded expression never compares equal.
        pub fn sameValues(a: []const Kv, b: []const Kv) bool {
            if (a.len != b.len) return false;
            for (a, b) |x, y| {
                if (!std.mem.eql(u8, x.key, y.key)) return false;
                const same = switch (x.value) {
                    .num => |n| y.value == .num and y.value.num == n,
                    .name => |n| y.value == .name and std.mem.eql(u8, y.value.name, n),
                    else => false,
                };
                if (!same) return false;
            }
            return true;
        }
    };
}
