//! Behavioural E/F/G/H forms: VALUE, POLY, TABLE, PWL, LAPLACE and the
//! rest rewritten as B-source text, and data-file elements.
//! Cites: ngspice inpcom.c (E/G keywords); XSPICE spice2poly; HSPICE [SA].
//!
//! Contents:
//! - `pub fn Methods(comptime S: type) type`
//! - `pub fn behaviouralCard(r: *R, line: []const u8, head: []const u8, letter: u8, f_in: F, frame: *const Frame, form: Form) Error!void`
//! - `pub fn wrapped(arena: Allocator, pre: []const u8, text: []const u8, post: []const u8) Error!std.ArrayList(u8)`
//! - `pub fn exprRest(r: *R, f: *F) Error![]const u8`
//! - `pub fn word(f: *F) Error![]const u8`
//! - `pub fn numberAt(r: *R, t: []const u8, f: *F, frame: *const Frame) Error!f64`
//! - `pub fn controlText(r: *R, w: *std.ArrayList(u8), f: *F, by_current: bool) Error!void`
//! - `pub fn tableText(r: *R, w: *std.ArrayList(u8), f: *F, frame: *const Frame) Error!f64`
//! - `pub fn polyText(r: *R, w: *std.ArrayList(u8), f: *F, frame: *const Frame, by_current: bool) Error!void`
//! - `pub fn readDataElement(r: *R, head: []const u8, letter: u8, f: *F, frame: *const Frame) Error!void`
//! - `pub fn dataPath(r: *R, f: *F) Error![]const u8`
//! - `pub fn portLine(r: *R, line: []const u8) Error![]const u8`

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
const sweep_points = @import("sweep_points.zig");
const Error = card_rows.Error;
const Kv = card_rows.Kv;
const data_file_keys = card_table.data_file_keys;
const isGround = netlist_view.isGround;
const Form = card_table.Form;
const behavioural = card_table.behavioural;
const kvIndex = card_table.kvIndex;
const nextPower = sweep_points.nextPower;
const Frame = reader_frame.Frame;

/// The stage's `Reader` methods over dialect syntax `S`; `Reader(S)`
/// aliases each one as its own decl.
pub fn Methods(comptime S: type) type {
    const F = S.Split;
    return struct {
        const R = reader.Reader(S);

        /// An E/F/G/H card in a behavioural form, `f` just past the name. Value,
        /// POLY, TABLE and PWL(1) forms, and HSPICE's VCR and VCCAP, become a
        /// B card over one expression (`i=`, `v=`, or `q=` for the charge of
        /// VCCAP); LAPLACE, POLE and DELAY keep their letter and carry the
        /// form in `laplace=` or `pole=<numerator count>`, or `td=`. A POLE
        /// side is a constant and (re, im) root pairs, so both counts are
        /// odd. Anything else is refused.
        pub fn behaviouralCard(r: *R, line: []const u8, head: []const u8, letter: u8, f_in: F, frame: *const Frame, form: Form) Error!void {
            const arena = r.arena;
            var f = f_in;
            // HSPICE quotes these expressions (`VOL='...'`, `TABLE '...'`);
            // a brace is a name byte there, and the form takes no name with
            // one (#203). tests/frontend/lower.zig `element` refuses the same.
            if (r.dialect == .hspice and std.mem.indexOfScalar(u8, f.rest(), '{') != null)
                return r.unsupported(line, "a braced expression (HSPICE quotes expressions: '...')");
            r.nodes.clearRetainingCapacity();
            r.positional.clearRetainingCapacity();
            r.card_kv.clearRetainingCapacity();
            for (0..2) |_| try r.nodes.append(arena, try word(&f));
            _ = f.next();
            const by_current = letter == 'f' or letter == 'h';
            var w: std.ArrayList(u8) = .empty;
            switch (form) {
                .refused, .kind => return r.unsupported(line, "unsupported controlled-source form"),
                .laplace, .pole, .delay => {
                    if (by_current) return r.unsupported(line, "unsupported controlled-source form");
                    for (0..2) |_| try r.nodes.append(arena, try word(&f));
                    var n_num: usize = 0;
                    while (f.next()) |t| {
                        if (t[0] == ',' or t[0] == '(' or t[0] == ')') continue;
                        if (form != .delay and std.mem.eql(u8, t, "/")) {
                            n_num = r.positional.items.len;
                        } else if (F.isWord(t) and f.takeEq()) {
                            try r.card_kv.append(arena, .{ .key = t, .value = try r.kvValue(&f, frame, false) });
                        } else try r.positional.append(arena, .{ .num = try r.numberAt(t, &f, frame) });
                    }
                    if (form != .delay) {
                        const n_den = r.positional.items.len - n_num;
                        if (n_num == 0 or n_den == 0) return error.ParseError;
                        if (form == .pole and (n_num % 2 == 0 or n_den % 2 == 0)) return error.ParseError;
                        try r.card_kv.append(arena, .{ .key = @tagName(form), .value = .{ .num = @floatFromInt(n_num) } });
                    } else if (r.positional.items.len != 0 or kvIndex(r.card_kv.items, "td") == null) return error.ParseError;
                    return r.commit(head, letter, frame);
                },
                .value => {
                    _ = f.takeEq();
                    try w.appendSlice(arena, try r.exprRest(&f));
                },
                .poly => try r.polyText(&w, &f, frame, by_current),
                .table => {
                    // TABLE {expr} = (x1, y1) (x2, y2) ... (ngspice inpcom.c),
                    // corners smoothed over a tenth of the shorter segment.
                    _ = f.takeEq();
                    const t = f.next() orelse return error.ParseError;
                    if (t[0] != '{' and !F.isQuote(t[0])) return error.ParseError;
                    _ = f.takeEq();
                    try w.print(arena, "table(({s}),0.1", .{F.body(t)});
                    _ = try r.tableText(&w, &f, frame);
                },
                .pwl, .vcr, .vccap => {
                    // HSPICE: [VCR|VCCAP] [PWL(1)|POLY(n)] in+ in- ..., or the
                    // linear `VCR in+ in- factor` (a resistance or capacitance).
                    const inner = if (form == .pwl) Form.pwl else blk: {
                        if (by_current) return r.unsupported(line, "unsupported controlled-source form");
                        const t = f.peek() orelse return error.ParseError;
                        var buf: [8]u8 = undefined;
                        const sub = if (t.len <= buf.len) behavioural.get(std.ascii.lowerString(&buf, t)) else null;
                        if (sub == null) break :blk Form.value;
                        if (sub.? != .pwl and sub.? != .poly) return r.unsupported(line, "unsupported controlled-source form");
                        _ = f.next();
                        break :blk sub.?;
                    };
                    switch (form) {
                        .vcr => try w.print(arena, "v({s},{s})/(", .{ r.nodes.items[0], r.nodes.items[1] }),
                        .vccap => try w.print(arena, "v({s},{s})*(", .{ r.nodes.items[0], r.nodes.items[1] }),
                        else => try w.append(arena, '('),
                    }
                    switch (inner) {
                        .poly => try r.polyText(&w, &f, frame, false),
                        .pwl => {
                            if (!std.mem.eql(u8, f.next() orelse "", "(") or !std.mem.eql(u8, f.next() orelse "", "1") or
                                !std.mem.eql(u8, f.next() orelse "", ")")) return r.unsupported(line, "multi-input PWL");
                            var ctl: std.ArrayList(u8) = .empty;
                            try r.controlText(&ctl, &f, by_current);
                            var pts: std.ArrayList(u8) = .empty;
                            const spacing = try r.tableText(&pts, &f, frame);
                            // DELTA, the corner width, defaults to a quarter of
                            // the closest breakpoint spacing [SA G-element parameters].
                            const delta = if (kvIndex(r.card_kv.items, "delta")) |k| switch (r.card_kv.items[k].value) {
                                .num => |x| x,
                                else => return error.ParseError,
                            } else spacing / 4;
                            try w.print(arena, "table({s},{e}{s}", .{ ctl.items, -delta, pts.items });
                        },
                        else => {
                            try w.append(arena, '(');
                            try r.controlText(&w, &f, false);
                            try w.print(arena, ")*({e})", .{try r.numberAt(f.next() orelse return error.ParseError, &f, frame)});
                        },
                    }
                    try w.append(arena, ')');
                },
            }
            // HSPICE modifiers [SA E-element parameters], in this order.
            while (f.next()) |t| {
                if (t[0] == ',') continue;
                if (!F.isWord(t) or !f.takeEq()) return error.ParseError;
                try r.card_kv.append(arena, .{ .key = t, .value = try r.kvValue(&f, frame, false) });
            }
            var tc: [2]Kv = undefined;
            var n_tc: usize = 0;
            for (r.card_kv.items) |kv| {
                const x = switch (kv.value) {
                    .num => |x| x,
                    else => return r.unsupported(line, "non-constant controlled-source parameter"),
                };
                const keys = std.StaticStringMap(u8).initComptime(.{
                    .{ "scale", 's' }, .{ "m", 's' },   .{ "abs", 'a' }, .{ "max", 'x' },   .{ "min", 'n' },
                    .{ "tc1", 't' },   .{ "tc2", 't' }, .{ "ic", 'i' },  .{ "delta", 'i' },
                });
                const k = keys.get(kv.key) orelse return r.unsupported(line, "unsupported controlled-source parameter");
                switch (k) {
                    's' => w = try wrapped(arena, "(", w.items, try arena.print(")*({e})", .{x})),
                    'a' => if (x != 0) {
                        w = try wrapped(arena, "abs(", w.items, ")");
                    },
                    'x' => w = try wrapped(arena, "min(", w.items, try arena.print(",{e})", .{x})),
                    'n' => w = try wrapped(arena, "max(", w.items, try arena.print(",{e})", .{x})),
                    't' => {
                        if (n_tc == tc.len) return error.ParseError;
                        tc[n_tc] = kv;
                        n_tc += 1;
                    },
                    else => {},
                }
            }
            r.card_kv.clearRetainingCapacity();
            const key: []const u8 = if (form == .vccap) "q" else if (letter == 'g' or letter == 'f') "i" else "v";
            try r.card_kv.append(arena, .{ .key = key, .value = try r.exprValue(w.items, frame, false) });
            try r.card_kv.appendSlice(arena, tc[0..n_tc]);
            return r.commit(head, 'b', frame);
        }

        /// `pre ++ text ++ post` in a fresh list.
        pub fn wrapped(arena: Allocator, pre: []const u8, text: []const u8, post: []const u8) Error!std.ArrayList(u8) {
            var out: std.ArrayList(u8) = .empty;
            try out.print(arena, "{s}{s}{s}", .{ pre, text, post });
            return out;
        }

        /// The expression after a B or VALUE `=`. A `{...}` or quoted field
        /// followed by nothing or by `name=` modifiers is its body; one that
        /// only starts the expression (`{g}*V(a)`) runs to the end of the
        /// line, its delimiters grouping like parentheses.
        pub fn exprRest(r: *R, f: *F) Error![]const u8 {
            const start = f.pos;
            const t = f.next() orelse return error.ParseError;
            const all = std.mem.trim(u8, f.line[start..], " \t");
            if (t[0] == '{' or F.isQuote(t[0])) {
                var after = f.*;
                const n = after.next() orelse return F.body(t);
                if (F.isWord(n) and after.takeEq()) return F.body(t);
            }
            f.pos = f.line.len;
            if (t[0] != '{' and !F.isQuote(t[0])) return all;
            const text = try r.arena.dupe(u8, all);
            var open = false;
            for (text) |*c| switch (c.*) {
                '{' => c.* = '(',
                '}' => c.* = ')',
                else => if (F.isQuote(c.*)) {
                    c.* = if (open) ')' else '(';
                    open = !open;
                },
            };
            return text;
        }

        /// The next node or source name, past any `(`, `)` or `,`.
        pub fn word(f: *F) Error![]const u8 {
            while (f.next()) |t| {
                if (t[0] == ',' or t[0] == '(' or t[0] == ')') continue;
                if (!F.isWord(t)) return error.ParseError;
                return t;
            }
            return error.ParseError;
        }

        /// A number field, or a parameter or `{expr}` that folds to one.
        pub fn numberAt(r: *R, t: []const u8, f: *F, frame: *const Frame) Error!f64 {
            return switch (try r.valueAt(t, f, frame, true, false)) {
                .num => |x| x,
                else => error.ParseError,
            };
        }

        /// One controlling quantity as expression text: `v(a,b)` over the
        /// next two nodes, or `i(vname)` over the next source name.
        pub fn controlText(r: *R, w: *std.ArrayList(u8), f: *F, by_current: bool) Error!void {
            if (by_current) return w.print(r.arena, "i({s})", .{try word(f)});
            const a = try word(f);
            const b = try word(f);
            if (isGround(r.dialect, b)) return w.print(r.arena, "v({s})", .{a});
            try w.print(r.arena, "v({s},{s})", .{ a, b });
        }

        /// `(x1, y1) (x2, y2) ...` or `x1,y1 x2,y2 ...` as `,x1,y1,...)`.
        /// A `key=value` among them goes to `card_kv`. Returns the smallest x
        /// spacing.
        pub fn tableText(r: *R, w: *std.ArrayList(u8), f: *F, frame: *const Frame) Error!f64 {
            var n: usize = 0;
            var last: f64 = 0;
            var spacing = std.math.inf(f64);
            while (f.next()) |t| {
                if (t[0] == ',' or t[0] == '(' or t[0] == ')') continue;
                if (F.isWord(t) and f.takeEq()) {
                    try r.card_kv.append(r.arena, .{ .key = t, .value = try r.kvValue(f, frame, false) });
                    continue;
                }
                const x = try r.numberAt(t, f, frame);
                if (n % 2 == 0) {
                    if (n > 0) spacing = @min(spacing, x - last);
                    last = x;
                }
                try w.print(r.arena, ",{e}", .{x});
                n += 1;
            }
            if (n < 4 or n % 2 != 0) return error.ParseError;
            try w.append(r.arena, ')');
            return spacing;
        }

        /// `POLY(n) <controls> p0 p1 ...` as a sum of products in SPICE 2G6
        /// coefficient order (XSPICE spice2poly nxtpwr). HSPICE reads a lone
        /// coefficient as p1 [SA G-element parameters]; ngspice as p0.
        pub fn polyText(r: *R, w: *std.ArrayList(u8), f: *F, frame: *const Frame, by_current: bool) Error!void {
            const arena = r.arena;
            if (!std.mem.eql(u8, f.next() orelse "", "(")) return error.ParseError;
            const dim_f = try r.numberAt(f.next() orelse return error.ParseError, f, frame);
            if (!std.mem.eql(u8, f.next() orelse "", ")") or dim_f < 1 or dim_f > 8 or dim_f != @trunc(dim_f)) return error.ParseError;
            const dim: usize = @intFromFloat(dim_f);
            var ctl: [8][]const u8 = undefined;
            for (ctl[0..dim]) |*c| {
                var t: std.ArrayList(u8) = .empty;
                try r.controlText(&t, f, by_current);
                c.* = t.items;
            }
            var coef: std.ArrayList(f64) = .empty;
            while (f.next()) |t| {
                if (t[0] == ',') continue;
                if (F.isWord(t) and f.takeEq()) {
                    try r.card_kv.append(arena, .{ .key = t, .value = try r.kvValue(f, frame, false) });
                    continue;
                }
                try coef.append(arena, try r.numberAt(t, f, frame));
            }
            if (coef.items.len == 0) return error.ParseError;
            if (coef.items.len == 1 and dim == 1 and r.dialect == .hspice) try coef.insert(arena, 0, 0);
            try w.print(arena, "({e})", .{coef.items[0]});
            var pw: [8]u32 = @splat(0);
            for (coef.items[1..]) |c| {
                nextPower(pw[0..dim]);
                if (c == 0) continue;
                try w.print(arena, "+({e})", .{c});
                for (pw[0..dim], ctl[0..dim]) |e, x| for (0..e) |_| try w.print(arena, "*{s}", .{x});
            }
        }

        /// HSPICE W and S cards: nodes and `key=value` pairs in any order
        /// (`W1 N=3 1 3 5 0 2 4 6 0 RLGCMODEL=m L=0.97`). A file key's value
        /// is the raw path in the deck's own case.
        pub fn readDataElement(r: *R, head: []const u8, letter: u8, f: *F, frame: *const Frame) Error!void {
            r.nodes.clearRetainingCapacity();
            r.positional.clearRetainingCapacity();
            r.card_kv.clearRetainingCapacity();
            while (f.next()) |t| {
                if (t[0] == ',') continue;
                if (!F.isWord(t)) return error.ParseError;
                if (!f.takeEq()) {
                    try r.nodes.append(r.arena, t);
                } else if (data_file_keys.has(t)) {
                    try r.card_kv.append(r.arena, .{ .key = t, .value = .{ .name = try r.dataPath(f) } });
                } else {
                    try r.card_kv.append(r.arena, .{ .key = t, .value = try r.kvValue(f, frame, false) });
                }
            }
            return r.commit(head, letter, frame);
        }

        /// A data file's path, recorded for `prepare` to read.
        pub fn dataPath(r: *R, f: *F) Error![]const u8 {
            const path = try r.pathOf(f);
            try r.foreign.append(r.arena, .{ .kind = .data, .path = path });
            return path;
        }

        /// An HSPICE P card with its `hblin=[h, s]` band vector spelled as
        /// the plain keys `hblin_h=h hblin_s=s` the builder reads. A longer
        /// vector (multi-tone HB) is refused.
        pub fn portLine(r: *R, line: []const u8) Error![]const u8 {
            const at = std.ascii.findIgnoreCase(line, "hblin") orelse return line;
            const open = std.mem.indexOfScalarPos(u8, line, at, '[') orelse return error.ParseError;
            const close = std.mem.indexOfScalarPos(u8, line, open, ']') orelse return error.ParseError;
            var it = std.mem.tokenizeAny(u8, line[open + 1 .. close], ", \t");
            const h = it.next() orelse return error.ParseError;
            const s = it.next() orelse return error.ParseError;
            if (it.next() != null) return r.unsupported(line, "multi-tone HBLIN band");
            return r.arena.print("{s} hblin_h={s} hblin_s={s} {s}", .{ line[0..at], h, s, line[close + 1 ..] });
        }
    };
}
