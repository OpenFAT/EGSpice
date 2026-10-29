//! Card values: a field as a number, a name, a group or postfix, folded
//! against the `.param` scopes.
//! Cites: dev/research/ngspice-syntax.md §2.2; HSPICE [CR10] p. 238 (#203).
//!
//! Contents:
//! - `pub fn Methods(comptime S: type) type`
//! - `pub fn readValue(r: *R, f: *F, frame: *const Frame, subst_names: bool, geometry: bool) Error!Value`
//! - `pub fn valueAt(r: *R, t: []const u8, f: *F, frame: *const Frame, subst_names: bool, geometry: bool) Error!Value`
//! - `pub fn kvValue(r: *R, f: *F, frame: *const Frame, geometry: bool) Error!Value`
//! - `pub fn junkNumber(w: []const u8) bool`
//! - `pub fn numberByte(c: u8) bool`
//! - `pub fn refuseOperatorField(r: *R, line: []const u8, t: []const u8) Error!void`
//! - `pub fn nameValue(r: *R, name: []const u8, frame: *const Frame, geometry: bool) Error!Value`
//! - `pub fn exprValue(r: *R, text: []const u8, frame: *const Frame, geometry: bool) Error!Value`
//! - `pub fn fold(r: *R, from: usize, frame: *const Frame, geometry: bool) Error!Value`
//! - `pub fn find(scopes: []const *const Scope, name: []const u8) ?Hit`
//! - `pub fn emitNum(r: *R, n: f64) Error!void`

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
const Error = card_rows.Error;
const Value = card_rows.Value;
const Scope = reader_frame.Scope;
const Frame = reader_frame.Frame;
const readsLive = card_table.readsLive;

/// The stage's `Reader` methods over dialect syntax `S`; `Reader(S)`
/// aliases each one as its own decl.
pub fn Methods(comptime S: type) type {
    const F = S.Split;
    return struct {
        const R = reader.Reader(S);
        const Hit = R.Hit;
        const max_group_depth = R.max_group_depth;

        /// A positional value: number, name, `{expr}`/quoted expression, or
        /// `name(args)`. `subst_names` substitutes a name a scope defines
        /// (devices and models do; analysis cards only substitute expressions).
        pub fn readValue(r: *R, f: *F, frame: *const Frame, subst_names: bool, geometry: bool) Error!Value {
            return r.valueAt(f.next() orelse return error.ParseError, f, frame, subst_names, geometry);
        }

        /// `readValue` whose first field `t` is already read.
        pub fn valueAt(r: *R, t: []const u8, f: *F, frame: *const Frame, subst_names: bool, geometry: bool) Error!Value {
            if (!F.isWord(t)) {
                if (t[0] == '{' or F.isQuote(t[0])) return r.exprValue(F.body(t), frame, geometry);
                return error.ParseError;
            }
            if (f.nextByte() == '(') {
                _ = f.next();
                // Bounded: each nesting level is a stack frame.
                if (r.group_depth == max_group_depth) return error.ParseError;
                r.group_depth += 1;
                defer r.group_depth -= 1;
                // `v(...)`/`i(...)` name nodes and devices, never numbers.
                const probe = std.ascii.eqlIgnoreCase(t, "v") or std.ascii.eqlIgnoreCase(t, "i");
                // `items[0]` is the name (`Group`).
                var items: std.ArrayList(Value) = .empty;
                try items.append(r.arena, .{ .name = t });
                while (true) {
                    const a = f.next() orelse return error.ParseError;
                    if (a[0] == ')') break;
                    if (a[0] == ',') continue;
                    try items.append(r.arena, if (probe and F.isWord(a)) .{ .name = a } else try r.valueAt(a, f, frame, subst_names, geometry));
                }
                // HSPICE writes SFFM(VO VA FC MDI FS) [CR SFFM]; the source
                // models read ngspice 44's (VO VA FM MDI FC). An omitted MDI
                // is 0 and an omitted FS the models' -1, their 5/TSTOP.
                const args = items.items[1..];
                if (r.dialect == .hspice and args.len > 2 and std.ascii.eqlIgnoreCase(t, "sffm")) {
                    if (args.len == 3) try items.append(r.arena, .{ .num = 0 });
                    if (items.items.len == 5) try items.append(r.arena, .{ .num = -1 });
                    std.mem.swap(Value, &items.items[3], &items.items[5]);
                }
                return .{ .group = .{ .items = items.items } };
            }
            if (S.parseNum(t)) |n| return .{ .num = n };
            return if (subst_names) r.nameValue(t, frame, geometry) else .{ .name = t };
        }

        /// The value after `key=`: an expression read up to where the grammar
        /// stops, or a braced/quoted field.
        pub fn kvValue(r: *R, f: *F, frame: *const Frame, geometry: bool) Error!Value {
            const rest = f.rest();
            if (rest.len == 0) return error.ParseError;
            // HSPICE reads a brace as a name byte, so `{...}` is a word there
            // and goes to the bare-value rules (#203).
            if ((rest[0] == '{' and !F.isWord(rest)) or rest[0] == '\'') return r.readValue(f, frame, true, geometry);
            // A lone number literal (`w=0.5u`) needs no compile; nor does
            // ngspice's `r=10%3`, which it reads as 10 (`junkNumber`, #200).
            if (std.ascii.isDigit(rest[0]) or rest[0] == '.') {
                const save = f.pos;
                const w = f.next().?;
                const lone = expr.numberLen(w) == w.len or (r.dialect == .ngspice and junkNumber(w));
                if (lone and !expr.isOperator(f.nextByte() orelse ' '))
                    return .{ .num = S.parseNum(w) orelse return error.ParseError };
                f.pos = save;
            }
            const mark = r.scratch.mark();
            defer r.scratch.reset(mark);
            const start = f.pos;
            f.pos = try expr.compile(S, r.arena, &r.scratch, f.line, f.pos);
            try r.checkBare(f.line[start..f.pos]);
            const ops = r.scratch.ops.items[mark.ops..];
            if (ops.len == 1 and ops[0].code == .num) return .{ .num = r.scratch.consts.items[ops[0].a] };
            if (ops.len == 1 and ops[0].code == .ident) return r.nameValue(r.scratch.names.items[ops[0].a], frame, geometry);
            return r.fold(mark.ops, frame, geometry);
        }

        /// Whether ngspice reads field `w` as one number token whose rest it
        /// ignores: a number that runs on through `%` or `\` (`10%3`, `2k\3`),
        /// since INPgetTok ends a token at `*`, `/` and `^` but at neither
        /// of those (inpgtok.c; ngspice-syntax.md §1.10). The ngspice.peg
        /// `numtok` rule.
        pub fn junkNumber(w: []const u8) bool {
            if (w.len < 2) return false;
            if (!std.ascii.isDigit(w[0]) and !(w[0] == '.' and std.ascii.isDigit(w[1]))) return false;
            var i: usize = 1;
            while (i < w.len and numberByte(w[i])) i += 1;
            if (i == w.len or (w[i] != '%' and w[i] != '\\')) return false;
            for (w[i..]) |c| if (!numberByte(c) and c != '%' and c != '\\') return false;
            return true;
        }

        /// A byte INPevaluate's token runs on through after its mantissa.
        pub fn numberByte(c: u8) bool {
            return std.ascii.isLower(c) or std.ascii.isDigit(c) or c == '_' or c == '.';
        }

        /// ngspice fails a device card whose value or model field starts
        /// with an operator byte (`R1 a b 10 %3`: "unknown parameter");
        /// ESPice refuses it by name instead of reading a name `%3` (#200).
        /// In HSPICE, which has no `%` or `\` operator, a field led by one
        /// is refused the same way (#203). Node fields are names and stay
        /// read, as do X cards and dot cards. tests/frontend/lower.zig
        /// `element` refuses the same fields.
        pub fn refuseOperatorField(r: *R, line: []const u8, t: []const u8) Error!void {
            if (!F.isWord(t)) return;
            const leads: []const u8 = switch (r.dialect) {
                .ngspice => "%\\*/^|&?",
                .hspice => "%\\",
                .spectre => "",
            };
            if (std.mem.indexOfScalar(u8, leads, t[0]) == null) return;
            return r.unsupported(line, "an operator where a name or value belongs");
        }

        /// A bare name: substituted when a scope defines it.
        pub fn nameValue(r: *R, name: []const u8, frame: *const Frame, geometry: bool) Error!Value {
            if (find(frame.scopes, name) == null) return .{ .name = name };
            const mark = r.scratch.mark();
            defer r.scratch.reset(mark);
            try r.scratch.names.append(r.arena, name);
            try r.scratch.ops.append(r.arena, .{ .code = .ident, .a = @intCast(mark.names) });
            return r.fold(mark.ops, frame, geometry);
        }

        pub fn exprValue(r: *R, text: []const u8, frame: *const Frame, geometry: bool) Error!Value {
            const mark = r.scratch.mark();
            defer r.scratch.reset(mark);
            try expr.compileAll(S, r.arena, &r.scratch, text);
            return r.fold(mark.ops, frame, geometry);
        }

        /// Splices parameters into the scratch ops from `from` on, then folds. A
        /// number keeps nothing; anything else keeps its postfix.
        /// A device or model value that reads a live name, or a distribution
        /// when some card runs Monte Carlo, keeps its postfix too: a variant
        /// re-evaluates it (`resolveLive`). A live name read anywhere else is
        /// `UnsupportedCard`, since nothing would re-read it.
        pub fn fold(r: *R, from: usize, frame: *const Frame, geometry: bool) Error!Value {
            const ops_mark = r.ops.items.len;
            const consts_mark = r.consts.items.len;
            const sites_mark = r.site_ops.items.len;
            try r.subst(&r.scratch, from, r.scratch.ops.items.len, frame, frame.scopes, 0);
            const ops = r.ops.items[ops_mark..];
            const v = try expr.fold(r.arena, &r.stack, ops, r.consts.items, geometry, r.live_nominal.items);
            if (v.known) {
                const reads = readsLive(ops);
                const sited = r.site_ops.items.len > sites_mark;
                if (reads and !r.in_card and !r.live_ok) {
                    if (!@import("builtin").is_test) std.log.err("netlist: a swept parameter is read outside a device or model value", .{});
                    return error.UnsupportedCard;
                }
                if (!(r.in_card and (reads or sited))) {
                    r.ops.shrinkRetainingCapacity(ops_mark);
                    r.consts.shrinkRetainingCapacity(consts_mark);
                    r.site_ops.shrinkRetainingCapacity(sites_mark);
                    r.site_keys.shrinkRetainingCapacity(sites_mark);
                    return .{ .num = v.num };
                }
            }
            return .{ .expr = .{ .start = @intCast(ops_mark), .len = @intCast(r.ops.items.len - ops_mark) } };
        }

        pub fn find(scopes: []const *const Scope, name: []const u8) ?Hit {
            var i = scopes.len;
            while (i > 0) {
                i -= 1;
                if (scopes[i].get(name)) |entry| return .{ .entry = entry, .level = i };
            }
            return null;
        }

        pub fn emitNum(r: *R, n: f64) Error!void {
            try r.consts.append(r.arena, n);
            try r.ops.append(r.arena, .{ .code = .num, .a = @intCast(r.consts.items.len - 1) });
        }

        // Walk 2: devices.
    };
}
