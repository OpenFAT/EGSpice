//! HSPICE sweeps: `SWEEP DATA=|MONTE=|OPTIMIZE=`, `.data` and `.variation`
//! blocks, `DEV`/`LOT` spreads.
//! Cites: HSPICE [CR] .DATA, .VARIATION, [SA] Ch.25, Ch.27.
//!
//! Contents:
//! - `pub fn Methods(comptime S: type) type`
//! - `pub fn readSweep(r: *R, kind: Kind, args: []const Value) Error!struct { args: []const Value, sweep: Sweep }`
//! - `pub fn readOptimize(r: *R, tail: []const Value) Error!Optimize`
//! - `pub fn positiveCount(args: []const Value, i: usize) Error!u32`
//! - `pub fn numbers(r: *R, args: []const Value) Error![]const f64`
//! - `pub fn readBlock(r: *R, b: [2]u32) Error!void`
//! - `pub fn readVariation(r: *R, b: [2]u32) Error!void`
//! - `pub fn sigmaExpr(r: *R, t: []const u8, line: []const u8) Error!Span`
//! - `pub fn percentValue(f: *F) ?struct { value: f64, relative: bool }`
//! - `pub fn devLot(r: *R, key: []const u8, param: []const u8, model: []const u8, f: *F, line: []const u8) Error!?Variation`

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
const card_table = @import("card_table.zig");
const reader_frame = @import("reader_frame.zig");
const sweep_points = @import("sweep_points.zig");
const Kind = card_rows.Kind;
const none = card_rows.none;
const Error = card_rows.Error;
const Span = card_rows.Span;
const Value = card_rows.Value;
const StepTarget = card_rows.StepTarget;
const Monte = card_rows.Monte;
const Sweep = card_rows.Sweep;
const Optimize = card_rows.Optimize;
const Variation = card_rows.Variation;
const cardOf = card_table.cardOf;
const Frame = reader_frame.Frame;
const grids = sweep_points.grids;
const points = sweep_points.points;

/// The stage's `Reader` methods over dialect syntax `S`; `Reader(S)`
/// aliases each one as its own decl.
pub fn Methods(comptime S: type) type {
    const F = S.Split;
    return struct {
        const R = reader.Reader(S);

        /// Splits an analysis card's HSPICE sweep tail (`SWEEP ...`, or a
        /// bare `DATA=`/`MONTE=` on `.dc`) off its arguments.
        pub fn readSweep(r: *R, kind: Kind, args: []const Value) Error!struct { args: []const Value, sweep: Sweep } {
            var at: ?usize = null;
            for (args, 0..) |a, i| if (a == .name and std.mem.eql(u8, a.name, "sweep")) {
                at = i;
                break;
            };
            if (at == null and kind == .dc and args.len >= 2 and args[0] == .name and
                (std.mem.eql(u8, args[0].name, "data") or std.mem.eql(u8, args[0].name, "monte"))) at = 0;
            // HSPICE `.dc param start stop incr` [CR .DC]: a swept `.param`
            // and no source. Its points are the DC sweep, as `.dc DATA=`'s
            // rows are (#94).
            if (at == null and kind == .dc and args.len == 4 and args[0] == .name) {
                if (try r.liveParam(args[0].name)) |k| {
                    const n = try r.numbers(args[1..]);
                    return .{ .args = args[0..0], .sweep = .{ .step = .{ .target = .{ .param = k }, .values = try points(r.arena, .lin, n[0], n[1], n[2], .per_unit) } } };
                }
            }
            const i = at orelse return .{ .args = args, .sweep = .none };
            const tail = args[i + @intFromBool(args[i] == .name and std.mem.eql(u8, args[i].name, "sweep")) ..];
            if (tail.len < 2 or tail[0] != .name) return error.ParseError;
            const head = tail[0].name;
            const sweep: Sweep = if (std.mem.eql(u8, head, "optimize")) blk: {
                // ponytail: one optimization per deck.
                if (r.optimize != null) return error.UnsupportedCard;
                r.optimize = try r.readOptimize(tail);
                break :blk .optimize;
            } else if (std.mem.eql(u8, head, "data")) blk: {
                if (tail.len != 2 or tail[1] != .name) return error.ParseError;
                break :blk .{ .data = tail[1].name };
            } else if (std.mem.eql(u8, head, "monte")) blk: {
                if (tail[1] == .name and std.mem.eql(u8, tail[1].name, "list")) {
                    const list = try r.arena.alloc(u32, tail.len - 2);
                    for (list, 2..) |*t, k| t.* = try positiveCount(tail, k);
                    if (list.len == 0) return error.ParseError;
                    r.monte = true;
                    break :blk .{ .monte = .{ .n = @intCast(list.len), .list = list } };
                }
                var m: Monte = .{ .n = try positiveCount(tail, 1) };
                if (tail.len == 4 and tail[2] == .name and std.mem.eql(u8, tail[2].name, "firstrun")) {
                    m.first = try positiveCount(tail, 3);
                } else if (tail.len != 2) return error.ParseError;
                r.monte = true;
                break :blk .{ .monte = m };
            } else blk: {
                const target = try r.stepTarget(head);
                if (tail.len >= 2 and tail[1] == .name) {
                    const grid = grids.get(tail[1].name) orelse return error.ParseError;
                    const n = try r.numbers(tail[2..]);
                    if (n.len < 1 or n[0] != @trunc(n[0]) or n[0] < 1) return error.ParseError;
                    if (grid == .poi) {
                        // In f64: a count past usize must not reach @intFromFloat.
                        if (@as(f64, @floatFromInt(n.len)) != 1 + n[0]) return error.ParseError;
                        break :blk .{ .step = .{ .target = target, .values = n[1..] } };
                    }
                    if (n.len != 3) return error.ParseError;
                    break :blk .{ .step = .{ .target = target, .values = try points(r.arena, grid, n[1], n[2], n[0], if (grid == .lin) .total else .per_unit) } };
                }
                const n = try r.numbers(tail[1..]);
                if (n.len != 3) return error.ParseError;
                break :blk .{ .step = .{ .target = target, .values = try points(r.arena, .lin, n[0], n[1], n[2], .per_unit) } };
            };
            return .{ .args = args[0..i], .sweep = sweep };
        }

        /// `OPTIMIZE=name RESULTS=m1,m2 MODEL=optmod`, in any order (`RESULT=`
        /// as in the manual's bisection example [SA Ch.19]), and the
        /// parameters `name(...)` or `OPTRANGE(...)` define, registered live
        /// at their initial values in deck order.
        pub fn readOptimize(r: *R, tail: []const Value) Error!Optimize {
            var opt: Optimize = .{ .name = "", .results = &.{}, .model = "", .live = &.{}, .lo = &.{}, .hi = &.{}, .dels = &.{}, .grid = &.{} };
            var results: std.ArrayList([]const u8) = .empty;
            const Key = enum { none, optimize, results, result, model };
            var key: Key = .none;
            for (tail) |v| {
                if (v != .name) return error.ParseError;
                if (std.meta.stringToEnum(Key, v.name)) |k| {
                    key = k;
                    continue;
                }
                switch (key) {
                    .optimize => opt.name = v.name,
                    .model => opt.model = v.name,
                    .results, .result => try results.append(r.arena, v.name),
                    .none => return error.ParseError,
                }
            }
            if (opt.name.len == 0 or opt.model.len == 0 or results.items.len == 0) return error.ParseError;
            opt.results = results.items;
            const Def = struct { key: []const u8, text: []const u8 };
            var defs: std.ArrayList(Def) = .empty;
            var it = r.globals.iterator();
            while (it.next()) |e| if (e.value_ptr.* == .text) {
                const text = e.value_ptr.text;
                const paren = std.mem.indexOfScalar(u8, text, '(') orelse continue;
                const head = std.mem.trim(u8, text[0..paren], " \t{'");
                if (std.mem.eql(u8, head, opt.name) or std.mem.eql(u8, head, "optrange"))
                    try defs.append(r.arena, .{ .key = e.key_ptr.*, .text = text[paren..] });
            };
            // `globals` iterates in deck order (`Scope`).
            if (defs.items.len == 0) return error.ParseError;
            const n = defs.items.len;
            const live = try r.arena.alloc(u32, n);
            const lo = try r.arena.alloc(f64, n);
            const hi = try r.arena.alloc(f64, n);
            const dels = try r.arena.alloc(f64, n);
            const grid = try r.arena.alloc(f64, n);
            const top: Frame = .{ .scopes = &r.global_scopes };
            for (defs.items, live, lo, hi, dels, grid) |d, *k, *l, *u, *h, *g| {
                // `(init, lo, hi[, dels[, grid]])`, each a number or an
                // expression; the grid is ESPice's, HSPICE stops at dels.
                const close = std.mem.lastIndexOfScalar(u8, d.text, ')') orelse return error.ParseError;
                var args: [5]f64 = @splat(0);
                var count: usize = 0;
                var depth: u32 = 0;
                var start: usize = 1;
                for (d.text[1 .. close + 1], 1..) |c, i| {
                    if (c == '(') depth += 1;
                    if (c == ')' and depth > 0) {
                        depth -= 1;
                        continue;
                    }
                    if (depth != 0 or (c != ',' and i != close)) continue;
                    if (count == args.len) return error.ParseError;
                    args[count] = switch (try r.exprValue(d.text[start..i], &top, false)) {
                        .num => |x| x,
                        else => return error.ParseError,
                    };
                    count += 1;
                    start = i + 1;
                }
                if (count < 3 or !(args[1] <= args[0] and args[0] <= args[2] and args[1] < args[2])) return error.ParseError;
                // A grid needs a multiple of itself inside [lo, hi].
                if (!(args[4] >= 0) or (args[4] > 0 and !(@ceil(args[1] / args[4]) <= @floor(args[2] / args[4])))) return error.ParseError;
                r.globals.getPtr(d.key).?.* = .{ .num = args[0] };
                k.* = (try r.liveParam(d.key)).?;
                l.* = args[1];
                u.* = args[2];
                h.* = args[3];
                g.* = args[4];
            }
            opt.live = live;
            opt.lo = lo;
            opt.hi = hi;
            opt.dels = dels;
            opt.grid = grid;
            return opt;
        }

        /// A positive integer argument.
        pub fn positiveCount(args: []const Value, i: usize) Error!u32 {
            if (i >= args.len or args[i] != .num) return error.ParseError;
            const n = args[i].num;
            if (!(n >= 1) or n != @trunc(n) or n > std.math.maxInt(u32)) return error.ParseError;
            return @intFromFloat(n);
        }

        /// Every value as a finite number.
        pub fn numbers(r: *R, args: []const Value) Error![]const f64 {
            const out = try r.arena.alloc(f64, args.len);
            for (args, out) |a, *o| {
                if (a != .num or !std.math.isFinite(a.num)) return error.ParseError;
                o.* = a.num;
            }
            return out;
        }

        /// A `.data` or `.variation` block, lines `b[0]..b[1]`.
        pub fn readBlock(r: *R, b: [2]u32) Error!void {
            r.card = r.lines.items[b[0]];
            var f = F.init(r.lines.items[b[0]]);
            const card = cardOf(f.next().?[1..]).?;
            if (card == .variation) return r.readVariation(b);
            const top: Frame = .{ .scopes = &r.global_scopes };
            const name = f.next() orelse return r.failed(r.lines.items[b[0]], error.ParseError);
            var labels: std.ArrayList([]const u8) = .empty;
            var columns: std.ArrayList(StepTarget) = .empty;
            var values: std.ArrayList(f64) = .empty;
            for (b[0]..b[1]) |li| {
                const line = r.lines.items[li];
                var lf = F.init(line);
                if (li == b[0]) {
                    _ = lf.next();
                    _ = lf.next();
                }
                while (lf.next()) |t| {
                    if (t[0] == ',' or t[0] == '+') continue;
                    if (F.isWord(t) and S.parseNum(t) == null) {
                        // `MER`/`LAM` and `FILE=` read external tables.
                        if (values.items.len != 0 or lf.nextByte() == '=') return r.failed(line, error.UnsupportedCard);
                        try labels.append(r.arena, t);
                        try columns.append(r.arena, try r.stepTarget(t));
                        continue;
                    }
                    switch (try r.valueAt(t, &lf, &top, false, false)) {
                        .num => |n| try values.append(r.arena, n),
                        else => return r.failed(line, error.ParseError),
                    }
                }
            }
            if (columns.items.len == 0 or values.items.len % columns.items.len != 0)
                return r.failed(r.lines.items[b[0]], error.ParseError);
            try r.data.append(r.arena, .{ .name = name, .labels = labels.items, .columns = columns.items, .values = values.items });
        }

        /// HSPICE `.variation` block [SA Ch.20]: `.global_variation` and
        /// `.local_variation` rows `<type> <model> p=sigma [%] ...`, and
        /// `.element_variation` rows `<letter> p=sigma [%] ...`. Sigmas are
        /// one sigma of a Gaussian, relative when followed by `%`; a sigma
        /// that is not a number is an expression (`sigmaExpr`).
        pub fn readVariation(r: *R, b: [2]u32) Error!void {
            var local = false;
            var element = false;
            for (b[0] + 1..b[1]) |li| {
                const line = r.lines.items[li];
                var f = F.init(line);
                const head = f.next().?;
                if (head[0] == '.') {
                    const markers = std.StaticStringMap([2]?bool).initComptime(.{
                        .{ ".global_variation", .{ false, false } }, .{ ".end_global_variation", .{ null, null } },
                        .{ ".local_variation", .{ true, false } },   .{ ".end_local_variation", .{ null, null } },
                        .{ ".element_variation", .{ null, true } },  .{ ".end_element_variation", .{ null, false } },
                    });
                    const m = markers.get(head) orelse return r.failed(line, error.UnsupportedCard);
                    if (m[0]) |l| local = l;
                    if (m[1]) |e| element = e;
                    continue;
                }
                if (std.mem.eql(u8, head, "option")) continue;
                var model: []const u8 = "";
                var letter: u8 = 0;
                if (element) letter = head[0] else model = f.next() orelse return r.failed(line, error.ParseError);
                while (f.next()) |key| {
                    if (!F.isWord(key) or !f.takeEq()) return r.failed(line, error.ParseError);
                    const t = f.peek() orelse return r.failed(line, error.ParseError);
                    const plain = F.isWord(t) and S.parseNum(if (t[t.len - 1] == '%') t[0 .. t.len - 1] else t) != null;
                    var row: Variation = .{ .per_device = local or element, .dist = .gauss, .model = model, .letter = letter, .param = key, .value = 1, .relative = false };
                    if (plain) {
                        const v = percentValue(&f) orelse return r.failed(line, error.ParseError);
                        row.value = v.value;
                        row.relative = v.relative;
                    } else {
                        row.sigma = try r.sigmaExpr(f.next().?, line);
                        // `'expr' %`: the expression is a percentage.
                        if (f.nextByte() == '%') {
                            _ = f.next();
                            row.value = 0.01;
                            row.relative = true;
                        }
                    }
                    try r.variations.append(r.arena, row);
                }
            }
        }

        /// A `.variation` sigma written as `'expr'`, `{expr}` or a bare
        /// name: its postfix over the global `.param`s at their nominal
        /// values and the device's `w`, `l`, `m` (`Variation.sigma_names`,
        /// `m` also as `mult`), kept as `.live` ops 0..2. Refuses a name it
        /// cannot resolve and a distribution call.
        pub fn sigmaExpr(r: *R, t: []const u8, line: []const u8) Error!Span {
            const mark = r.scratch.mark();
            defer r.scratch.reset(mark);
            try expr.compileAll(S, r.arena, &r.scratch, if (F.isWord(t)) t else F.body(t));
            const slot = std.StaticStringMap(u32).initComptime(.{ .{ "w", 0 }, .{ "l", 1 }, .{ "m", 2 }, .{ "mult", 2 } });
            for (r.scratch.ops.items[mark.ops..]) |*op| if (op.code == .ident) {
                if (slot.get(r.scratch.names.items[op.a])) |k| op.* = .{ .code = .live, .a = k };
            };
            const start = r.ops.items.len;
            const sites = r.site_ops.items.len;
            const top: Frame = .{ .scopes = &r.global_scopes };
            try r.subst(&r.scratch, mark.ops, r.scratch.ops.items.len, &top, &r.global_scopes, 0);
            const ops = r.ops.items[start..];
            const probe = try expr.fold(r.arena, &r.stack, ops, r.consts.items, false, &.{ 1, 1, 1 });
            var drawn = r.site_ops.items.len != sites;
            for (ops) |op| drawn = drawn or (op.code == .call and expr.isDistribution(@fromBackingInt(@intCast(op.a))));
            if (!probe.known or drawn) return r.failed(line, error.ParseError);
            return .{ .start = @intCast(start), .len = @intCast(ops.len) };
        }

        /// A number, optionally followed by (or ending in) `%`.
        pub fn percentValue(f: *F) ?struct { value: f64, relative: bool } {
            const t = f.next() orelse return null;
            const pct = t[t.len - 1] == '%';
            const n = S.parseNum(if (pct) t[0 .. t.len - 1] else t) orelse return null;
            if (!std.math.isFinite(n)) return null;
            if (!pct and f.nextByte() == '%') {
                _ = f.next();
                return .{ .value = n / 100, .relative = true };
            }
            return .{ .value = if (pct) n / 100 else n, .relative = pct };
        }

        /// HSPICE `dev[/n][/dist]=v` or `lot[/n][/dist]=v` after model value
        /// `param`; null when `key` is neither.
        pub fn devLot(r: *R, key: []const u8, param: []const u8, model: []const u8, f: *F, line: []const u8) Error!?Variation {
            var parts = std.mem.splitScalar(u8, key, '/');
            const head = parts.next().?;
            const per_device = std.mem.eql(u8, head, "dev");
            if (!per_device and !std.mem.eql(u8, head, "lot")) return null;
            var dist: Variation.Dist = .unif;
            while (parts.next()) |p| {
                if (p.len > 0 and std.ascii.isDigit(p[0])) continue;
                const dists = std.StaticStringMap(Variation.Dist).initComptime(.{
                    .{ "gauss", .gauss }, .{ "unif", .unif }, .{ "uniform", .unif }, .{ "limit", .limit },
                });
                dist = dists.get(p) orelse return r.failed(line, error.UnsupportedCard);
            }
            if (param.len == 0) return r.failed(line, error.ParseError);
            const v = percentValue(f) orelse return r.failed(line, error.ParseError);
            // A Gaussian's value is its 3-sigma spread [SA Ch.20 "Variations
            // Specified Using DEV and LOT"]; `Variation.value` holds one sigma.
            const value = if (dist == .gauss) v.value / 3 else v.value;
            return .{ .per_device = per_device, .dist = dist, .model = model, .param = param, .value = value, .relative = v.relative };
        }
    };
}
