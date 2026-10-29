//! HSPICE `.check`, `.dout` and `.biaschk` cards. In: one card's keyword and
//! its lowercased text. Out: the `core.Measure` rows it expands to, one per
//! node, which the `.meas` evaluator runs.
//! Cites: HSPICE [CR] .CHECK, .DOUT, .BIASCHK.
//!
//! Contents:
//! - `pub const Levels = struct`
//! - `pub fn parseCheck(arena: Allocator, card: []const u8, text: []const u8, ctx: anytype, global: *?Levels, serial: u32, out: *std.ArrayList(core.Measure)) !void`

const std = @import("std");
const assert = std.debug.assert;
const mem = std.mem;
const Allocator = std.mem.Allocator;
const core = @import("core");
const measure_words = @import("measure_words.zig");
const Word = measure_words.Word;

/// Logic thresholds of `.check` cards: low at or below `lo`, high at or
/// above `hi`.
pub const Levels = struct { lo: f64, hi: f64 };

/// The `.check` forms.
const CheckKind = enum { global_level, rise, fall, slew, setup, hold, edge, irdrop };

/// Appends the measures one card expands to, one per node, to `out`.
/// `card` is the card name without its dot and `text` the rest.
/// `.check global_level` only sets `global`, which later cards default to.
/// `serial` numbers the card in its measures' names (`setup2_v1`).
/// `ctx` evaluates values as `measure.parse`'s does.
/// ParseError for a form ESPice does not read: a node wildcard, an element
/// or region `.biaschk`, a `.biaschk` outside the transient; the errors of
/// `ctx`'s callbacks pass through. `out` keeps the rows appended before an
/// error. Invalidates element pointers of `out` if it grows.
pub fn parseCheck(
    arena: Allocator,
    card: []const u8,
    text: []const u8,
    ctx: anytype,
    global: *?Levels,
    serial: u32,
    out: *std.ArrayList(core.Measure),
) !void {
    const w = try measure_words.words(arena, text);
    if (w.len == 0) return error.ParseError;
    const m: core.Measure = .{ .analysis = .tran, .name = "", .func = .check_slew, .first = .{}, .second = .{} };
    if (mem.eql(u8, card, "biaschk")) return out.append(arena, try biasCheck(arena, w, ctx, serial, m));
    if (mem.eql(u8, card, "dout")) return digitalOut(arena, w, ctx, serial, m, out);
    if (!mem.eql(u8, card, "check") or w.len < 2) return error.ParseError;
    return check(arena, w, ctx, global, serial, m, out);
}

// ---- private ----

/// `.biaschk 'expr' [limit=] [max=] [min=] [simulation=tr] [tstart=] [tstop=] ...`
fn biasCheck(arena: Allocator, w: []const Word, ctx: anytype, serial: u32, base: core.Measure) !core.Measure {
    assert(w.len > 0);
    if (w[0].lhs[0] != '\'' and w[0].lhs[0] != '"') return error.ParseError;
    var m = base;
    m.func = .check_level;
    m.first.ops = try ctx.measureExpr(w[0].lhs);
    m.first.vec = w[0].lhs;
    for (w[1..]) |x| {
        const rhs = x.rhs orelse continue; // `autostop`
        if (mem.eql(u8, x.lhs, "simulation")) {
            if (!mem.eql(u8, rhs, "tr")) return error.ParseError;
        } else if (mem.eql(u8, x.lhs, "limit") or mem.eql(u8, x.lhs, "max")) {
            m.check.max = try ctx.measureValue(rhs);
        } else if (mem.eql(u8, x.lhs, "min")) {
            m.check.min = try ctx.measureValue(rhs);
        } else if (mem.eql(u8, x.lhs, "tstart")) {
            m.first.from = try ctx.measureValue(rhs);
        } else if (mem.eql(u8, x.lhs, "tstop")) {
            m.first.to = try ctx.measureValue(rhs);
        } else if (!mem.eql(u8, x.lhs, "noise") and !mem.eql(u8, x.lhs, "interval")) return error.ParseError;
    }
    m.name = try arena.print("biaschk{d}", .{serial});
    return m;
}

/// `.dout nd [nd ...] [VTH | VLO VHI] (time state [state ...] ...)`; the
/// group may touch the last word (`vth(0n 1 ...)`).
fn digitalOut(arena: Allocator, w: []const Word, ctx: anytype, serial: u32, base: core.Measure, out: *std.ArrayList(core.Measure)) !void {
    assert(w.len > 0);
    var last = w[w.len - 1].lhs;
    const open = mem.indexOfScalar(u8, last, '(') orelse return error.ParseError;
    const head = last[0..open];
    last = last[open..];
    var names: std.ArrayList([]const u8) = .empty;
    var levels: std.ArrayList(f64) = .empty;
    for (w[0 .. w.len - 1]) |x| try names.append(arena, x.lhs);
    if (head.len > 0) try names.append(arena, head);
    // The first word is a node; values after it are thresholds.
    while (names.items.len > 1) {
        const v = ctx.measureValue(names.items[names.items.len - 1]) catch |e| {
            if (e == error.OutOfMemory) return e;
            break;
        };
        try levels.insert(arena, 0, v);
        _ = names.pop();
    }
    // ponytail: no threshold given reads the .vec VTH default, 1.65 V.
    const lo, const hi = switch (levels.items.len) {
        0 => .{ 1.65, 1.65 },
        1 => .{ levels.items[0], levels.items[0] },
        2 => .{ levels.items[0], levels.items[1] },
        else => return error.ParseError,
    };
    const row = try measure_words.groupWords(arena, last);
    const width = names.items.len + 1;
    if (row.len == 0 or row.len % width != 0) return error.ParseError;
    const before = out.items.len;
    for (names.items, 0..) |node, k| {
        var expect: std.ArrayList([2]f64) = .empty;
        var r: usize = 0;
        while (r < row.len) : (r += width) {
            const state = row[r + 1 + k].lhs;
            if (mem.eql(u8, state, "0") or mem.eql(u8, state, "1"))
                try expect.append(arena, .{ try ctx.measureValue(row[r].lhs), if (state[0] == '1') 1 else 0 });
        }
        var d = base;
        d.func = .dout;
        d.check = .{ .lo = lo, .hi = hi, .expect = expect.items };
        try appendNode(arena, d, "dout", serial, node, out);
    }
    assert(out.items.len == before + names.items.len);
}

/// `.check <kind> (spec) node... [(levels)]`.
fn check(arena: Allocator, w: []const Word, ctx: anytype, global: *?Levels, serial: u32, base: core.Measure, out: *std.ArrayList(core.Measure)) !void {
    assert(w.len >= 2);
    const kind = std.meta.stringToEnum(CheckKind, w[0].lhs) orelse return error.ParseError;
    const spec = try measure_words.groupWords(arena, w[1].lhs);
    var nodes = w[2..];
    var levels = global.*;
    if (nodes.len > 0 and nodes[nodes.len - 1].lhs[0] == '(') {
        levels = try parseLevels(arena, nodes[nodes.len - 1].lhs, ctx);
        nodes = nodes[0 .. nodes.len - 1];
    }
    if (kind == .global_level) {
        global.* = try parseLevels(arena, w[1].lhs, ctx);
        return;
    }
    if (nodes.len == 0) return error.ParseError;
    var m = base;
    switch (kind) {
        .global_level => unreachable,
        // `(min max)`
        .rise, .fall, .slew => {
            if (spec.len != 2) return error.ParseError;
            m.check.min = try ctx.measureValue(spec[0].lhs);
            m.check.max = try ctx.measureValue(spec[1].lhs);
            m.check.edge = switch (kind) {
                .rise => .rise,
                .fall => .fall,
                else => .both,
            };
        },
        // `(ref RISE|FALL duration RISE|FALL)`, `(ref RISE|FALL min max RISE|FALL)`
        .setup, .hold, .edge => {
            if (spec.len != @as(usize, if (kind == .edge) 5 else 4)) return error.ParseError;
            m.second.vec = try nodeVector(arena, spec[0].lhs);
            m.check.ref_edge = try edgeOf(spec[1].lhs);
            m.check.edge = try edgeOf(spec[spec.len - 1].lhs);
            const a = try ctx.measureValue(spec[2].lhs);
            m.func = if (kind == .edge) .check_require else .check_forbid;
            m.check.min, m.check.max = switch (kind) {
                .setup => .{ -a, 0 },
                .hold => .{ 0, a },
                else => .{ a, try ctx.measureValue(spec[3].lhs) },
            };
        },
        // `(volt duration)`: below a negative level, above a positive one.
        .irdrop => {
            if (spec.len != 2) return error.ParseError;
            const v = try ctx.measureValue(spec[0].lhs);
            m.func = .check_level;
            m.check.dur = try ctx.measureValue(spec[1].lhs);
            if (v < 0) m.check.min = v else m.check.max = v;
        },
    }
    if (m.func != .check_level) {
        const l = levels orelse return error.ParseError;
        m.check.lo = l.lo;
        m.check.hi = l.hi;
    }
    for (nodes) |x| try appendNode(arena, m, @tagName(kind), serial, x.lhs, out);
}

/// `m` reading node `node`, named `<kind><serial>_<node>`.
fn appendNode(arena: Allocator, m: core.Measure, kind: []const u8, serial: u32, node: []const u8, out: *std.ArrayList(core.Measure)) !void {
    assert(kind.len > 0);
    var one = m;
    one.first.vec = try nodeVector(arena, node);
    one.name = try arena.print("{s}{d}_{s}", .{ kind, serial, node });
    try out.append(arena, one);
}

/// A bare node name as its result label, `v(node)`; a `v(..)` or `i(..)`
/// stays as written.
fn nodeVector(arena: Allocator, node: []const u8) ![]const u8 {
    if (mem.indexOfScalar(u8, node, '*') != null) return error.ParseError;
    if (mem.indexOfScalar(u8, node, '(') != null) return node;
    return arena.print("v({s})", .{node});
}

fn edgeOf(word: []const u8) !core.Check.Edge {
    if (mem.eql(u8, word, "rise")) return .rise;
    if (mem.eql(u8, word, "fall")) return .fall;
    return error.ParseError;
}

/// `(hi lo hi_th lo_th)`; a threshold ending in `%` is that share of the
/// swing above `lo`.
fn parseLevels(arena: Allocator, group: []const u8, ctx: anytype) !Levels {
    const g = try measure_words.groupWords(arena, group);
    if (g.len != 4) return error.ParseError;
    const hi = try ctx.measureValue(g[0].lhs);
    const lo = try ctx.measureValue(g[1].lhs);
    var th: [2]f64 = undefined;
    for (g[2..], &th) |x, *t| {
        const s = x.lhs;
        t.* = if (s[s.len - 1] == '%') lo + (hi - lo) * try ctx.measureValue(s[0 .. s.len - 1]) / 100 else try ctx.measureValue(s);
    }
    return .{ .lo = th[1], .hi = th[0] };
}
