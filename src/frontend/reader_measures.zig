//! `.meas`, `.check`, `.power` and `.stim` cards, and the measure callbacks
//! measure.zig evaluates values through.
//! Cites: ngspice com_measure2.c; HSPICE [CR] .MEASURE, .STIM, .POWER.
//!
//! Contents:
//! - `pub fn Methods(comptime S: type) type`
//! - `pub fn readMeasures(r: *R) Error!void`
//! - `pub fn skipBad(err: anyerror, line: []const u8) error{OutOfMemory}!void`
//! - `pub fn readPower(r: *R, f: *F, serial: u32) Error!void`
//! - `pub fn readStim(r: *R, f: *F, serials: *[2]u32) Error!void`
//! - `pub fn sourcePower(r: *R, name: []const u8) Error![]const u8`
//! - `pub fn measureExpr(r: *R, text: []const u8) Error![]const core.MeasureOp`
//! - `pub fn measureOption(r: *R, name: []const u8) ?f64`
//! - `pub fn measureValue(r: *R, text: []const u8) Error!f64`

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
const reader_frame = @import("reader_frame.zig");
const Kind = card_rows.Kind;
const Error = card_rows.Error;
const isGround = netlist_view.isGround;
const Frame = reader_frame.Frame;

/// The stage's `Reader` methods over dialect syntax `S`; `Reader(S)`
/// aliases each one as its own decl.
pub fn Methods(comptime S: type) type {
    const F = S.Split;
    return struct {
        const R = reader.Reader(S);

        /// Parses the `.meas` cards in deck order. ngspice reports a bad one
        /// and simulates anyway.
        pub fn readMeasures(r: *R) Error!void {
            var last: ?Kind = null;
            if (r.dialect == .hspice) for (r.analyses.items) |a| switch (a.kind) {
                .tran, .ac, .dc => last = a.kind,
                else => {},
            };
            var jitters: u32 = 0;
            var checks: u32 = 0;
            var powers: u32 = 0;
            var stims: [2]u32 = @splat(0);
            var global_level: ?measure.Levels = null;
            for (r.meas_lines.items) |line| {
                // HSPICE writes a measure's expression as `par('...')` or
                // quoted; a brace is a name byte there, and a measure card
                // holding one is refused, not skipped (#203). lower.zig
                // `prescan` refuses the same.
                if (r.dialect == .hspice and std.mem.indexOfScalar(u8, line, '{') != null)
                    return r.unsupported(line, "a braced expression (HSPICE quotes expressions: '...')");
                var f = F.init(line);
                var text = f.rest();
                const head = f.next().?[1..];
                if (std.mem.eql(u8, head, "check") or std.mem.eql(u8, head, "dout") or std.mem.eql(u8, head, "biaschk")) {
                    const n = r.measures.items.len;
                    measure.parseCheck(r.arena, head, f.rest(), r, &global_level, checks + 1, &r.measures) catch |err| try skipBad(err, line);
                    if (r.measures.items.len > n) checks += 1;
                    continue;
                }
                if (std.mem.eql(u8, head, "power")) {
                    powers += 1;
                    r.readPower(&f, powers) catch |err| try skipBad(err, line);
                    continue;
                }
                if (std.mem.startsWith(u8, head, "stim")) {
                    r.readStim(&f, &stims) catch |err| try skipBad(err, line);
                    continue;
                }
                if (std.mem.eql(u8, head, "jitter")) {
                    // `.jitter <kind> TRIG ...` reads as `.meas <kind>
                    // jitter[N] jitter TRIG ...`, N from the second card on.
                    jitters += 1;
                    const kind = f.next() orelse "";
                    text = if (jitters == 1)
                        try r.arena.print("{s} jitter jitter {s}", .{ kind, f.rest() })
                    else
                        try r.arena.print("{s} jitter{d} jitter {s}", .{ kind, jitters, f.rest() });
                } else text = f.rest();
                const m = measure.parse(r.arena, text, r, last, r.dialect == .hspice) catch |err| switch (err) {
                    error.OutOfMemory => return error.OutOfMemory,
                    else => {
                        std.log.warn("netlist: ignoring malformed card '{s}'", .{line});
                        continue;
                    },
                };
                try r.measures.append(r.arena, m);
            }
        }

        /// Logs a card `readMeasures` cannot take and goes on, as ngspice
        /// does; only `OutOfMemory` fails the parse.
        pub fn skipBad(err: anyerror, line: []const u8) error{OutOfMemory}!void {
            if (err == error.OutOfMemory) return error.OutOfMemory;
            std.log.warn("netlist: ignoring unsupported or malformed card '{s}'", .{line});
        }

        /// HSPICE `.power signal [REF=] [FROM=] [TO=]` [CR .POWER] as four
        /// transient measures, `power<serial>_avg`, `_rms`, `_max` and
        /// `_min`. A bare name is a top-level voltage source, read as the
        /// power it absorbs, v(n+, n-) * i(name); any other signal is an
        /// output variable read as written. REF= only names HSPICE's table
        /// and is dropped.
        pub fn readPower(r: *R, f: *F, serial: u32) Error!void {
            const start = f.pos;
            _ = f.next() orelse return error.ParseError;
            if (f.nextByte() == '(') while (f.next()) |t| if (t[0] == ')') break;
            const sig = std.mem.trim(u8, f.line[start..f.pos], " \t");
            const wave = if (std.mem.indexOfScalar(u8, sig, '(') != null) sig else try r.sourcePower(sig);
            var window: std.ArrayList(u8) = .empty;
            while (f.next()) |t| {
                if (std.mem.eql(u8, t, "ref") and f.takeEq()) {
                    _ = f.next();
                    continue;
                }
                try window.print(r.arena, " {s}", .{t});
            }
            for ([_][]const u8{ "avg", "rms", "max", "min" }) |func| {
                const text = try r.arena.print("tran power{d}_{s} {s} {s}{s}", .{ serial, func, func, wave, window.items });
                try r.measures.append(r.arena, try measure.parse(r.arena, text, r, .tran, true));
            }
        }

        /// HSPICE `.stim [tran] pwl|data [filename=] [dataname] [name=]ovar
        /// [node1= node2=] ... [from= to= npoints= | indepvar=(t ...)]
        /// [indepout]` [CR .STIM]. Each signal is read as a `find` measure.
        /// A signal with no name takes its output variable's, `v(a)` as
        /// `v_a`; a `par()` one must be named, and a PWL source's name says
        /// V or I. A PWL source of `v(a[,b])` defaults to terminals a and b
        /// (else 0). The AC, DC and VEC forms are a ParseError.
        pub fn readStim(r: *R, f: *F, serials: *[2]u32) Error!void {
            var form = f.next() orelse return error.ParseError;
            if (std.mem.eql(u8, form, "tran")) form = f.next() orelse return error.ParseError;
            const data = std.mem.eql(u8, form, "data");
            if (!data and !std.mem.eql(u8, form, "pwl")) return error.ParseError;
            var stim: core.Stim = .{ .data = data, .serial = serials[@intFromBool(data)] };
            var signals: std.ArrayList(core.Measure) = .empty;
            var nodes: std.ArrayList([2][]const u8) = .empty;
            var times: std.ArrayList(f64) = .empty;
            const Key = enum { filename, node1, node2, from, to, npoints, indepvar, indepout };
            var name: ?[]const u8 = null;
            while (f.peek()) |t| {
                const start = f.pos;
                _ = f.next();
                const key = if (F.isWord(t) and f.takeEq()) t else null;
                if (key == null and std.mem.eql(u8, t, "indepout")) {
                    stim.indepout = true;
                    continue;
                }
                const field = if (key) |k| std.meta.stringToEnum(Key, k) else null;
                if (field) |which| switch (which) {
                    .filename => stim.file = try r.pathOf(f),
                    .node1, .node2 => {
                        const node = f.next() orelse return error.ParseError;
                        if (nodes.items.len == 0) return error.ParseError;
                        nodes.items[nodes.items.len - 1][@intFromBool(which == .node2)] = node;
                    },
                    .from, .to => {
                        const v = S.parseNum(f.next() orelse "") orelse return error.ParseError;
                        if (which == .from) stim.from = v else stim.to = v;
                    },
                    .npoints => stim.npoints = std.math.lossyCast(u32, S.parseNum(f.next() orelse "") orelse return error.ParseError),
                    .indepvar => {
                        const paren = f.peek() != null and f.peek().?[0] == '(';
                        if (paren) _ = f.next();
                        while (f.peek()) |v| {
                            if (v[0] == ')') {
                                _ = f.next();
                                break;
                            }
                            const x = S.parseNum(v) orelse if (paren) return error.ParseError else break;
                            _ = f.next();
                            try times.append(r.arena, x);
                        }
                    },
                    .indepout => {
                        const v = f.next() orelse "";
                        stim.indepout = !std.mem.eql(u8, v, "0") and !std.mem.eql(u8, v, "off");
                    },
                } else if (key) |k| {
                    name = k;
                } else if (data and stim.dataname.len == 0 and signals.items.len == 0 and f.peek() != null and f.peek().?[0] != '(') {
                    stim.dataname = t;
                } else {
                    // An output variable: `v(a)`, `i(r1)`, `par('...')`.
                    if (f.nextByte() == '(') while (f.next()) |p| if (p[0] == ')') break;
                    const ovar = std.mem.trim(u8, f.line[start..f.pos], " \t");
                    const text = try r.arena.print("tran s find {s} at=0", .{ovar});
                    var m = try measure.parse(r.arena, text, r, .tran, true);
                    m.name = name orelse blk: {
                        if (m.first.ops.len != 0) return error.ParseError;
                        const own = try r.arena.dupe(u8, ovar);
                        for (own) |*c| if (!std.ascii.isAlphanumeric(c.*)) {
                            c.* = '_';
                        };
                        break :blk std.mem.trimEnd(u8, own, "_");
                    };
                    name = null;
                    if (m.name.len == 0 or (!data and m.name[0] != 'v' and m.name[0] != 'i')) return error.ParseError;
                    var pair: [2][]const u8 = .{ "", "0" };
                    if (std.mem.startsWith(u8, ovar, "v(") and std.mem.endsWith(u8, ovar, ")")) {
                        var it = std.mem.tokenizeAny(u8, ovar[2 .. ovar.len - 1], ", ");
                        pair[0] = it.next() orelse "";
                        pair[1] = it.next() orelse "0";
                    }
                    try signals.append(r.arena, m);
                    try nodes.append(r.arena, pair);
                }
            }
            if (signals.items.len == 0 or (data and stim.dataname.len == 0)) return error.ParseError;
            if (!data) for (nodes.items) |pair| if (pair[0].len == 0) return error.ParseError;
            stim.signals = signals.items;
            stim.nodes = nodes.items;
            stim.times = times.items;
            serials[@intFromBool(data)] += 1;
            try r.stims.append(r.arena, stim);
        }

        /// `par()` power absorbed by top-level voltage source `name`.
        pub fn sourcePower(r: *R, name: []const u8) Error![]const u8 {
            if (name[0] != 'v') return error.ParseError;
            for (r.lines.items) |line| {
                var f = F.init(line);
                if (!std.mem.eql(u8, f.next() orelse continue, name)) continue;
                const p = f.next() orelse return error.ParseError;
                const n = f.next() orelse return error.ParseError;
                if (isGround(r.dialect, n)) return r.arena.print("par('v({s})*i({s})')", .{ p, name });
                return r.arena.print("par('(v({s})-v({s}))*i({s})')", .{ p, n, name });
            }
            return error.ParseError;
        }

        /// A `PARAM=` or `par()` measure expression in postfix: names of
        /// earlier `.meas` cards read their results, global parameters fold
        /// to numbers, `v(a[,b])` and `i(x)` read result vectors. Arithmetic
        /// only; anything else is a ParseError. The ops live in the arena.
        pub fn measureExpr(r: *R, text: []const u8) Error![]const core.MeasureOp {
            const body = if (text.len > 0 and (text[0] == '{' or F.isQuote(text[0]))) F.body(text) else text;
            const mark = r.scratch.mark();
            defer r.scratch.reset(mark);
            try expr.compileAll(S, r.arena, &r.scratch, body);
            // Copies: folding a parameter below reuses the scratch.
            const ops = try r.arena.dupe(expr.Op, r.scratch.ops.items[mark.ops..]);
            const names = try r.arena.dupe([]const u8, r.scratch.names.items);
            const consts = try r.arena.dupe(f64, r.scratch.consts.items);
            var out: std.ArrayList(core.MeasureOp) = .empty;
            for (ops) |op| try out.append(r.arena, switch (op.code) {
                .num => .{ .num = consts[op.a] },
                .vprobe, .iprobe => {
                    const letter = if (op.code == .vprobe) "v" else "i";
                    try out.append(r.arena, .{ .vector = try r.arena.print("{s}({s})", .{ letter, names[op.a] }) });
                    if (op.b == expr.none) continue;
                    try out.append(r.arena, .{ .vector = try r.arena.print("{s}({s})", .{ letter, names[op.b] }) });
                    try out.append(r.arena, .sub);
                    continue;
                },
                .ident => for (r.measures.items, 0..) |m, i| {
                    if (std.mem.eql(u8, m.name, names[op.a])) break .{ .measure = @intCast(i) };
                } else .{ .num = try r.measureValue(names[op.a]) },
                .neg => .neg,
                .add => .add,
                .sub => .sub,
                .mul => .mul,
                .div => .div,
                .pow => .pow,
                else => return error.ParseError,
            });
            return out.items;
        }

        /// A numeric `.option name=value`, the last one given; null when no
        /// card sets it. `name` matches byte for byte, unlike `option`.
        pub fn measureOption(r: *R, name: []const u8) ?f64 {
            var found: ?f64 = null;
            for (r.config.items) |c| if (!c.temp) for (c.args, 0..) |a, i| {
                if (a != .name or i + 1 >= c.args.len or !std.mem.eql(u8, a.name, name)) continue;
                if (c.args[i + 1] == .num) found = c.args[i + 1].num;
            };
            return found;
        }

        /// A `.meas` value: a number or a global parameter expression.
        /// ParseError for empty text or one that does not fold.
        pub fn measureValue(r: *R, text: []const u8) Error!f64 {
            if (text.len == 0) return error.ParseError;
            if (S.parseNum(text)) |n| return n;
            const body = if (text[0] == '{' or F.isQuote(text[0])) F.body(text) else text;
            const top: Frame = .{ .scopes = &r.global_scopes };
            return switch (try r.exprValue(body, &top, false)) {
                .num => |n| n,
                else => error.ParseError,
            };
        }
    };
}
