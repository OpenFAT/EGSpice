//! Spectre named statements. In: a statement led by a name. Out: the SPICE
//! card of an instance (primitive, model, subcircuit or Verilog-A module),
//! an analysis or a control statement, appended to the `Translator`.
//! Cites: dev/research/spectre-syntax.md §4.1-§4.3, §4.9, §4.10;
//! dev/frontend.md "Spectre dialect" (the primitive list).
//!
//! Contents:
//! - `pub fn named(t: *Translator, toks: []const Tok) Error!void`

const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;
const spectre_expr = @import("spectre_expr.zig");
const spectre_params = @import("spectre_params.zig");
const spectre_state = @import("spectre_state.zig");
const Error = spectre_expr.Error;
const Tok = spectre_expr.Tok;
const Param = spectre_params.Param;
const Translator = spectre_state.Translator;

/// The source parameters `waveform` reads.
const wave_keys = std.StaticStringMap(void).initComptime(.{
    .{"dc"},        .{"mag"},  .{"phase"}, .{"type"},  .{"val0"},   .{"val1"}, .{"period"},
    .{"rise"},      .{"fall"}, .{"width"}, .{"delay"}, .{"sinedc"}, .{"ampl"}, .{"freq"},
    .{"sinephase"}, .{"damp"}, .{"td1"},   .{"tau1"},  .{"td2"},    .{"tau2"}, .{"wave"},
});

/// A statement led by a name: an analysis, a control statement or an
/// instance (§4.1).
pub fn named(t: *Translator, toks: []const Tok) Error!void {
    if (toks[0].kind != .word) return error.ParseError;
    const name = toks[0].text;
    var i: usize = 1;
    var nodes: std.ArrayList([]const u8) = .empty;
    var paren = false;
    if (i < toks.len and toks[i].is("(")) {
        paren = true;
        i += 1;
        while (i < toks.len and !toks[i].is(")")) : (i += 1) {
            if (toks[i].kind != .word) return error.ParseError;
            try nodes.append(t.arena, toks[i].text);
        }
        if (i == toks.len) return error.ParseError;
        i += 1;
    }
    if (i < toks.len and toks[i].kind == .word and !(i + 1 < toks.len and toks[i + 1].is("="))) {
        const kw = toks[i].text;
        if (spectre_state.analysis_kinds.has(kw)) return analysis(t, kw, nodes.items, toks[i + 1 ..]);
        if (!paren) if (spectre_state.control_kinds.get(kw)) |c| return control(t, c, toks[i + 1 ..]);
    }
    // An instance: without parentheses, the master is the last bare word
    // before the first `key=` (§4.2).
    if (!paren) {
        while (i < toks.len and toks[i].kind == .word and !(i + 1 < toks.len and toks[i + 1].is("="))) : (i += 1)
            try nodes.append(t.arena, toks[i].text);
        const master = nodes.pop() orelse return error.ParseError;
        return instance(t, name, nodes.items, master, try spectre_params.params(t.arena, toks[i..]));
    }
    if (i >= toks.len or toks[i].kind != .word) return error.ParseError;
    return instance(t, name, nodes.items, toks[i].text, try spectre_params.params(t.arena, toks[i + 1 ..]));
}

// ---- private ----

fn instance(t: *Translator, name: []const u8, nodes: []const []const u8, master: []const u8, ps: []const Param) Error!void {
    const pins = try std.mem.join(t.arena, " ", nodes);
    if (spectre_state.primitives.get(master)) |letter| return primitive(t, letter, master, name, pins, ps);
    var kv: std.ArrayList(u8) = .empty;
    for (ps) |p| try kv.print(t.arena, " {s}={s}", .{ p.key, try spectre_params.value(t.arena, p.value) });
    if (t.subckt_names.contains(master)) return t.device('x', "{s} {s} {s}{s}", .{ name, pins, master, kv.items });
    if (t.models.get(master)) |mm| {
        const m = spectre_state.masters.get(mm) orelse return error.UnsupportedCard;
        return t.device(m.letter, "{s} {s} {s}{s}", .{ name, pins, master, kv.items });
    }
    // A model group's members are `master.label`; the instance names
    // the group, which bins.
    if (t.ahdl) {
        const gop = try t.ahdl_models.getOrPut(t.arena, master);
        if (!gop.found_existing) try t.emit(".model {s} {s}", .{ master, master });
        return t.device('n', "{s} {s} {s}{s}", .{ name, pins, master, kv.items });
    }
    return error.UnsupportedCard;
}

/// A primitive's card (dev/frontend.md, the primitive list).
fn primitive(t: *Translator, letter: u8, master: []const u8, name: []const u8, pins: []const u8, ps: []const Param) Error!void {
    var pos: std.ArrayList(u8) = .empty;
    var kv: std.ArrayList(u8) = .empty;
    const lead: []const []const u8 = switch (letter) {
        'r' => &.{"r"},
        'c' => &.{"c"},
        'l' => &.{"l"},
        'e' => &.{"gain"},
        'g' => &.{"gm"},
        'h' => &.{ "probe", "rm" },
        'f' => &.{ "probe", "gain" },
        'k' => &.{ "ind1", "ind2", "coupling" },
        else => &.{},
    };
    for (lead) |key| if (spectre_params.find(ps, key)) |v| try pos.print(t.arena, " {s}", .{try spectre_params.value(t.arena, v)});
    var source = false;
    if (letter == 'v' or letter == 'i') {
        source = true;
        if (std.mem.eql(u8, master, "iprobe")) try pos.appendSlice(t.arena, " dc 0");
        try waveform(t, &pos, ps);
    }
    for (ps) |p| {
        const used = for (lead) |key| {
            if (std.mem.eql(u8, key, p.key)) break true;
        } else source and wave_keys.has(p.key);
        if (used) continue;
        try kv.print(t.arena, " {s}={s}", .{ p.key, try spectre_params.value(t.arena, p.value) });
    }
    const sep: []const u8 = if (pins.len > 0) " " else "";
    try t.device(letter, "{s}{s}{s}{s}{s}", .{ name, sep, pins, pos.items, kv.items });
}

/// `dc`, `mag`/`phase` and `type=` waveforms as SPICE source values.
fn waveform(t: *Translator, pos: *std.ArrayList(u8), ps: []const Param) Error!void {
    if (spectre_params.find(ps, "dc")) |v| try pos.print(t.arena, " dc {s}", .{try spectre_params.value(t.arena, v)});
    if (spectre_params.find(ps, "mag")) |v| {
        try pos.print(t.arena, " ac {s}", .{try spectre_params.value(t.arena, v)});
        if (spectre_params.find(ps, "phase")) |ph| try pos.print(t.arena, " {s}", .{try spectre_params.value(t.arena, ph)});
    }
    const ty = spectre_params.find(ps, "type") orelse return;
    if (ty.len != 1) return error.ParseError;
    const kind = ty[0].text;
    const fields: []const []const u8, const fname: []const u8 = if (std.mem.eql(u8, kind, "pulse"))
        .{ &.{ "val0", "val1", "delay", "rise", "fall", "width", "period" }, "pulse" }
    else if (std.mem.eql(u8, kind, "sine"))
        .{ &.{ "sinedc", "ampl", "freq", "delay", "damp", "sinephase" }, "sin" }
    else if (std.mem.eql(u8, kind, "exp"))
        .{ &.{ "val0", "val1", "td1", "tau1", "td2", "tau2" }, "exp" }
    else if (std.mem.eql(u8, kind, "pwl")) {
        const wave = spectre_params.find(ps, "wave") orelse return error.UnsupportedCard;
        if (wave.len < 2 or !wave[0].is("[") or !wave[wave.len - 1].is("]")) return error.ParseError;
        try pos.print(t.arena, " pwl(", .{});
        var first = true;
        for (try spectre_params.vector(t.arena, wave)) |v| {
            if (!first) try pos.append(t.arena, ' ');
            first = false;
            try pos.appendSlice(t.arena, v);
        }
        try pos.append(t.arena, ')');
        return;
    } else if (std.mem.eql(u8, kind, "dc"))
        return
    else
        return error.UnsupportedCard;
    // Trailing fields not given are left off; a gap reads as 0.
    var last: usize = 0;
    for (fields, 0..) |f, k| if (spectre_params.find(ps, f) != null) {
        last = k + 1;
    };
    try pos.print(t.arena, " {s}(", .{fname});
    for (fields[0..last], 0..) |f, k| {
        if (k > 0) try pos.append(t.arena, ' ');
        try pos.appendSlice(t.arena, if (spectre_params.find(ps, f)) |v| try spectre_params.value(t.arena, v) else "0");
    }
    try pos.append(t.arena, ')');
}

fn analysis(t: *Translator, kind: []const u8, nodes: []const []const u8, toks: []const Tok) Error!void {
    if (t.subckts.items.len > 0) return error.UnsupportedCard;
    if (toks.len > 0 and toks[toks.len - 1].is("{")) return error.UnsupportedCard; // child analyses
    const ps = try spectre_params.params(t.arena, toks);
    const num = struct {
        fn of(arena: Allocator, list: []const Param, key: []const u8) Error!?[]const u8 {
            const v = spectre_params.find(list, key) orelse return null;
            return try spectre_params.value(arena, v);
        }
    }.of;
    if (std.mem.eql(u8, kind, "tran")) {
        for (ps) |p| if (!spectre_state.tran_keys.has(p.key)) {
            if (!@import("builtin").is_test) std.log.err("spectre: tran parameter `{s}` is not supported", .{p.key});
            return error.UnsupportedCard;
        };
        const stop = try num(t.arena, ps, "stop") orelse return error.ParseError;
        const step = try num(t.arena, ps, "step") orelse try t.arena.print("\"({s}/50)\"", .{std.mem.trim(u8, stop, "\"")});
        const start = try num(t.arena, ps, "start");
        const maxstep = try num(t.arena, ps, "maxstep");
        // ponytail: `method` becomes the deck's `.options method`, so a
        // second tran with another method sets both. Per-card methods
        // when a deck needs two.
        if (spectre_params.find(ps, "method")) |m| try t.emit(".options {s}", .{spectre_state.tran_methods.get(spectre_params.word(m)) orelse return error.UnsupportedCard});
        // `moderate` is Spectre's default; the other presets retune
        // reltol and the LTE together, which is not mapped.
        if (spectre_params.find(ps, "errpreset")) |e| if (!std.mem.eql(u8, spectre_params.word(e), "moderate")) return error.UnsupportedCard;
        // `skipdc=yes` starts from the `ic` statements, as `uic` does.
        const uic = if (spectre_params.find(ps, "skipdc")) |d|
            (if (std.mem.eql(u8, spectre_params.word(d), "yes")) " uic" else if (std.mem.eql(u8, spectre_params.word(d), "no")) "" else return error.UnsupportedCard)
        else
            "";
        if (maxstep) |m| return t.emit(".tran {s} {s} {s} {s}{s}", .{ step, stop, start orelse "0", m, uic });
        if (start) |s| return t.emit(".tran {s} {s} {s}{s}", .{ step, stop, s, uic });
        return t.emit(".tran {s} {s}{s}", .{ step, stop, uic });
    }
    if (std.mem.eql(u8, kind, "dc")) {
        const target = if (spectre_params.find(ps, "dev")) |d| d[0].text else if (spectre_params.find(ps, "param")) |p| p[0].text else {
            if (ps.len != 0) return error.UnsupportedCard;
            return t.emit(".op", .{});
        };
        if (spectre_params.find(ps, "dev") != null) if (spectre_params.find(ps, "param")) |p| if (!std.mem.eql(u8, p[0].text, "dc")) return error.UnsupportedCard;
        return t.emit(".dc {s} {s}", .{ target, try sweep(t, ps, false) });
    }
    if (std.mem.eql(u8, kind, "ac")) return t.emit(".ac {s}", .{try sweep(t, ps, true)});
    if (std.mem.eql(u8, kind, "noise")) {
        const src = spectre_params.find(ps, "iprobe") orelse return error.UnsupportedCard;
        const out = switch (nodes.len) {
            1 => try t.arena.print("v({s})", .{nodes[0]}),
            2 => try t.arena.print("v({s},{s})", .{ nodes[0], nodes[1] }),
            else => return error.UnsupportedCard,
        };
        return t.emit(".noise {s} {s} {s}", .{ out, src[0].text, try sweep(t, ps, true) });
    }
    return error.UnsupportedCard;
}

/// A sweep's SPICE grid: `dec N a b`, `lin N a b`, `poi n v...` (`freq`
/// set when `ac`); for `.dc`, `a b step` or `poi`.
fn sweep(t: *Translator, ps: []const Param, ac: bool) Error![]const u8 {
    if (spectre_params.find(ps, "values")) |v| {
        const vals = try spectre_params.vector(t.arena, v);
        return t.arena.print("poi {d} {s}", .{ vals.len, try std.mem.join(t.arena, " ", vals) });
    }
    const start = try spectre_params.value(t.arena, spectre_params.find(ps, "start") orelse return error.UnsupportedCard);
    const stop = try spectre_params.value(t.arena, spectre_params.find(ps, "stop") orelse return error.UnsupportedCard);
    if (spectre_params.find(ps, "dec")) |n| return t.arena.print("dec {s} {s} {s}", .{ try spectre_params.value(t.arena, n), start, stop });
    if (spectre_params.find(ps, "lin")) |n| {
        if (ac) return t.arena.print("lin {s} {s} {s}", .{ try spectre_params.value(t.arena, n), start, stop });
        const q = "\"";
        return t.arena.print("{s} {s} \"(({s}-{s})/{s})\"", .{ start, stop, std.mem.trim(u8, stop, q), std.mem.trim(u8, start, q), std.mem.trim(u8, try spectre_params.value(t.arena, n), q) });
    }
    if (spectre_params.find(ps, "step")) |s| {
        if (ac) return error.UnsupportedCard;
        return t.arena.print("{s} {s} {s}", .{ start, stop, try spectre_params.value(t.arena, s) });
    }
    return error.UnsupportedCard;
}

fn control(t: *Translator, c: spectre_state.ControlKind, toks: []const Tok) Error!void {
    switch (c) {
        .refused => return error.UnsupportedCard,
        .ignored => return,
        .options => {
            var text: std.ArrayList(u8) = .empty;
            for (try spectre_params.params(t.arena, toks)) |p| {
                const v = try spectre_params.value(t.arena, p.value);
                if (std.mem.eql(u8, p.key, "temp")) {
                    try t.emit(".temp {s}", .{v});
                    continue;
                }
                const key = if (std.mem.eql(u8, p.key, "vabstol")) "vntol" else if (std.mem.eql(u8, p.key, "iabstol")) "abstol" else p.key;
                try text.print(t.arena, " {s}={s}", .{ key, v });
            }
            if (text.items.len > 0) try t.emit(".options{s}", .{text.items});
        },
    }
}
