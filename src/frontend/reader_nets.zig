//! Nets and subcircuit expansion: interned names, flattened node names, and
//! an `X` card's body re-read under its frame.
//! Cites: ngspice subckt.c; dev/research/ngspice-syntax.md §4.1.
//!
//! Contents:
//! - `pub fn Methods(comptime S: type) type`
//! - `pub fn internName(r: *R, s: []const u8) Error!Name`
//! - `pub fn intern(r: *R, s: []const u8) Error!VertexId`
//! - `pub fn connect(r: *R, a: []const u8, b: []const u8) Error!void`
//! - `pub fn joined(r: *R, parts: []const []const u8) Error![]const u8`
//! - `pub fn netOf(r: *R, frame: *const Frame, node: []const u8) Error!VertexId`
//! - `pub fn mixedPort(f: F) bool`
//! - `pub fn fixedPins(letter: u8) ?usize`
//! - `pub fn expand(r: *R, head: []const u8, frame: *const Frame, kv_text: []const []const u8) Error!void`
//! - `pub fn expandedExtra(r: *R, top: []const u32) Error!struct { cards: usize, kvs: usize }`
//! - `pub fn bodySize(r: *R, sizes: []Size, id: u16, depth: u8) Size`
//! - `pub fn xTarget(r: *const R, i: usize) ?u16`
//! - `pub fn skipNested(card: Card, nested: *u32) bool`

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
const isGround = netlist_view.isGround;
const Card = card_table.Card;
const Branches = card_table.Branches;
const port_source_words = card_table.port_source_words;
const cards = card_table.cards;
const cardOf = card_table.cardOf;
const Scope = reader_frame.Scope;
const Frame = reader_frame.Frame;

/// The stage's `Reader` methods over dialect syntax `S`; `Reader(S)`
/// aliases each one as its own decl.
pub fn Methods(comptime S: type) type {
    const F = S.Split;
    return struct {
        const R = reader.Reader(S);
        const Size = R.Size;

        /// A name into the pool; a new one maps to no net yet.
        pub fn internName(r: *R, s: []const u8) Error!Name {
            const n = try r.pool.intern(r.arena, s);
            if (n.index() == r.net_of.items.len) try r.net_of.append(r.arena, none);
            return n;
        }

        pub fn intern(r: *R, s: []const u8) Error!VertexId {
            if (isGround(r.dialect, s)) return ground;
            const n = try r.internName(s);
            const slot = &r.net_of.items[n.index()];
            if (slot.* == none) {
                const v = r.hg.addVertex(r.arena, .{ .name = n }) catch |err| return switch (err) {
                    error.OutOfMemory => error.OutOfMemory,
                    else => error.CircuitTooLarge,
                };
                slot.* = v.index();
            }
            return .from(slot.*);
        }

        /// Makes nets `a` and `b` one net, before any device card is read.
        /// Two names that are already distinct nets (an alias chain closing
        /// on itself through different roots) are refused.
        pub fn connect(r: *R, a: []const u8, b: []const u8) Error!void {
            // Name rows, not pointers into `net_of`: interning `b` can grow it.
            var rows: [2]?u32 = .{ null, null };
            for ([_][]const u8{ a, b }, &rows) |name, *row| if (!isGround(r.dialect, name)) {
                row.* = (try r.internName(try r.arena.dupe(u8, name))).index();
            };
            var ids: [2]u32 = .{ 0, 0 };
            for (rows, &ids) |row, *id| if (row) |n| {
                id.* = r.net_of.items[n];
            };
            if (ids[0] == none and ids[1] == none) ids[0] = (try r.intern(try r.arena.dupe(u8, a))).index();
            if (ids[0] == none) r.net_of.items[rows[0].?] = ids[1] else if (ids[1] == none) r.net_of.items[rows[1].?] = ids[0] else if (ids[0] != ids[1]) return error.ParseError;
        }

        /// `parts` joined, in reused scratch.
        pub fn joined(r: *R, parts: []const []const u8) Error![]const u8 {
            r.name_buf.clearRetainingCapacity();
            for (parts) |part| try r.name_buf.appendSlice(r.arena, part);
            return r.name_buf.items;
        }

        /// A node as the frame names it: a port maps to the caller's net,
        /// ground (`isGround`) and `.global` names stay global, anything else
        /// is `<path>.<node>`.
        pub fn netOf(r: *R, frame: *const Frame, node: []const u8) Error!VertexId {
            const path = frame.path orelse return r.intern(node);
            if (frame.port_index) |index| {
                if (index.get(node)) |k| return frame.actuals[k];
            } else for (frame.ports, frame.actuals) |p, a| if (std.mem.eql(u8, p, node)) return a;
            if (isGround(r.dialect, node) or r.global_nets.contains(node)) return r.intern(node);
            return r.intern(try r.joined(&.{ path, ".", node }));
        }

        /// A mixed-mode P card, `P1 n+ n- ref port=1 ...` [SA Ch.17]: a
        /// third word that is neither a key (`port=`) nor a source keyword.
        /// The manual writes a P card's DC value as `DC mag`, so a bare
        /// third word is the reference node, not a value.
        pub fn mixedPort(f: F) bool {
            var p = f;
            _ = p.next();
            _ = p.next();
            const t = p.next() orelse return false;
            if (!F.isWord(t)) return false;
            if (p.next()) |u| if (u[0] == '=') return false;
            var buf: [8]u8 = undefined;
            return t.len > buf.len or !port_source_words.has(std.ascii.lowerString(&buf, t));
        }

        pub fn fixedPins(letter: u8) ?usize {
            return switch (letter) {
                // W n+ n- Vctrl model: Vctrl/model are positional words (ngspice INP2W).
                'r', 'c', 'l', 'v', 'i', 'd', 'b', 'f', 'h', 'w' => 2,
                'q', 'z', 'j' => 3,
                'e', 'g', 's', 'm', 't', 'o' => 4,
                'k' => 0,
                else => null,
            };
        }

        /// Flattens an `X` card: reads its subcircuit's lines under a child frame.
        pub fn expand(r: *R, head: []const u8, frame: *const Frame, kv_text: []const []const u8) Error!void {
            const arena = r.arena;
            // A top-level X card expands at depth 0, so 32 levels end at 31.
            if (frame.depth >= 32) {
                if (!@import("builtin").is_test) std.log.err("netlist: '{s}': subcircuits nest more than 32 levels", .{head});
                return error.ParseError;
            }
            if (r.positional.items.len < 1) return error.ParseError;
            const sname = switch (r.positional.items[r.positional.items.len - 1]) {
                .name => |n| n,
                else => return error.ParseError,
            };
            const id = r.subckt_ids.get(sname) orelse return error.ParseError;
            const sub = r.subckts.items[id];
            if (r.nodes.items.len != sub.ports.len) return error.ParseError;
            if (r.instances == std.math.maxInt(u32)) return error.CircuitTooLarge;
            const instance = r.instances;
            r.instances += 1;

            // Shadow order: instance values, subcircuit defaults, enclosing scopes.
            const scope = try arena.create(Scope);
            scope.* = .empty;
            for (sub.defaults) |d| try scope.put(arena, d.key, .{ .text = d.text });
            for (r.card_kv.items, kv_text) |kv, text| try scope.put(arena, kv.key, switch (kv.value) {
                .num => |n| .{ .num = n },
                .name => |n| .{ .text = n },
                else => .{ .text = if (text.len > 0 and (text[0] == '{' or F.isQuote(text[0]))) F.body(text) else text },
            });
            const actuals = try arena.alloc(VertexId, r.nodes.items.len);
            for (r.nodes.items, actuals) |n, *a| a.* = try r.netOf(frame, n);
            const scopes = try std.mem.concat(arena, *const Scope, &.{ frame.scopes, &.{scope} });
            const child: Frame = .{
                .path = if (frame.path) |p| try std.mem.concat(arena, u8, &.{ p, ".", head }) else head,
                .ports = sub.ports,
                .actuals = actuals,
                .port_index = sub.port_index,
                .scopes = scopes,
                .depth = frame.depth + 1,
                .instance = instance,
                .sub = id,
                .models_mark = @intCast(r.instance_models.items.len),
            };
            defer r.instance_models.shrinkRetainingCapacity(child.models_mark);
            // Models first, as ngspice reads every `.model` before any
            // device: the ones under a branch this instance keeps.
            const locals = r.local_models.items[sub.models_lo..sub.models_hi];
            if (for (locals) |m| {
                if (m.cond) break true;
            } else false) {
                var kept: Branches = .{};
                var nested: u32 = 0;
                for (r.lines.items[sub.first..sub.end], sub.first..) |line, i| {
                    if (line[0] != '.') continue;
                    var f = F.init(line);
                    const card = cardOf(f.next().?[1..]) orelse continue;
                    if (sub.has_nested and skipNested(card, &nested)) continue;
                    switch (card) {
                        .cond => |c| try r.branch(&kept, c, &f, &child),
                        .model => if (kept.active()) for (locals, sub.models_lo..) |m, lm| {
                            if (m.cond and m.line == i) _ = try r.localRow(lm, &child);
                        },
                        else => {},
                    }
                }
            }
            var branches: Branches = .{};
            var nested: u32 = 0;
            for (r.lines.items[sub.first..sub.end], sub.first..) |line, i| {
                if (line[0] != '.') {
                    if (nested == 0 and branches.active()) try r.readDevice(line, r.kindOf(i), &child);
                    continue;
                }
                if (!sub.has_cond and !sub.has_nested) continue;
                var f = F.init(line);
                const card = cardOf(f.next().?[1..]) orelse continue;
                if (sub.has_nested and skipNested(card, &nested)) continue;
                if (card == .cond) try r.branch(&branches, card.cond, &f, &child);
            }
            if (branches.depth != 0) return error.ParseError;
        }

        /// Cards and `key=value` rows that flattening the `X` cards among
        /// lines `top` adds, counted off each subcircuit's body (its own
        /// lines, and its `X` cards' subcircuits in turn). A reservation
        /// estimate: both arms of a body's `.if` count, and an `X` card whose
        /// subcircuit this cannot name counts nothing.
        /// ponytail: capped at 16M cards, so a deep or cyclic hierarchy
        /// that `expand` refuses reserves nothing absurd.
        pub fn expandedExtra(r: *R, top: []const u32) Error!struct { cards: usize, kvs: usize } {
            if (r.subckts.items.len == 0) return .{ .cards = 0, .kvs = 0 };
            const sizes = try r.arena.alloc(Size, r.subckts.items.len);
            @memset(sizes, .{});
            var total: Size = .{};
            for (top) |i| if (r.xTarget(i)) |id| total.add(r.bodySize(sizes, id, 0));
            const cap = 1 << 24;
            return .{ .cards = @min(total.cards, cap), .kvs = @min(total.kvs, 8 * cap) };
        }

        /// Subcircuit `id` flattened: its body's cards and `=` signs, its
        /// `X` cards' subcircuits included. Memoized in `sizes`; a
        /// subcircuit reached again while counting (a cycle) or past
        /// `expand`'s depth counts nothing.
        pub fn bodySize(r: *R, sizes: []Size, id: u16, depth: u8) Size {
            if (sizes[id].state == .done) return sizes[id];
            if (sizes[id].state == .busy or depth >= 32) return .{};
            sizes[id].state = .busy;
            const sub = r.subckts.items[id];
            var size: Size = .{};
            var nested: u32 = 0;
            for (r.lines.items[sub.first..sub.end], sub.first..) |line, i| {
                if (line[0] == '.') {
                    if (!sub.has_nested) continue;
                    var f = F.init(line);
                    const card = cardOf(f.next().?[1..]) orelse continue;
                    _ = skipNested(card, &nested);
                    continue;
                }
                if (nested != 0) continue;
                size.add(.{ .cards = 1, .kvs = std.mem.countScalar(u8, line, '=') });
                if (r.xTarget(i)) |x| size.add(r.bodySize(sizes, x, depth + 1));
            }
            size.state = .done;
            sizes[id] = size;
            return size;
        }

        /// The subcircuit line `i` instantiates when it is an `X` card
        /// whose last word before its parameters names one, else null.
        pub fn xTarget(r: *const R, i: usize) ?u16 {
            const line = r.lines.items[i];
            const letter = if (r.kindOf(i) != 0) r.kindOf(i) else std.ascii.toLower(line[0]);
            if (letter != 'x') return null;
            var f = F.init(line);
            _ = f.next();
            var last: ?[]const u8 = null;
            while (true) {
                const t = f.next() orelse break;
                if (!F.isWord(t) or f.takeEq()) break;
                if (!std.mem.eql(u8, t, "params:")) last = t;
            }
            return r.subckt_ids.get(last orelse return null);
        }

        /// Tracks nested `.subckt` ... `.ends` while reading a body: true
        /// for every card from a nested definition's header to its `.ends`.
        pub fn skipNested(card: Card, nested: *u32) bool {
            switch (card) {
                .subckt => {
                    nested.* += 1;
                    return true;
                },
                .ends => if (nested.* > 0) {
                    nested.* -= 1;
                    return true;
                },
                else => {},
            }
            return nested.* > 0;
        }

        // After the walk: shunts and model bins.
    };
}
