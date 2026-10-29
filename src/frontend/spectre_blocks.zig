//! Spectre structure statements. In: one statement's tokens. Out: the SPICE
//! cards for `parameters`, `model` (and model groups), `subckt`/`ends`,
//! `global`, `ic`/`nodeset`, `if`/`else` chains, skipped blocks and `real`
//! functions, appended to the `Translator`.
//! Cites: dev/research/spectre-syntax.md §4.5-§4.8, §4.12;
//! dev/frontend.md "Spectre dialect".
//!
//! Contents:
//! - `pub fn parameters(t: *Translator, toks: []const Tok) Error!void`
//! - `pub fn model(t: *Translator, toks: []const Tok) Error!void`
//! - `pub fn groupRow(t: *Translator, toks: []const Tok) Error!void`
//! - `pub fn subckt(t: *Translator, toks: []const Tok) Error!void`
//! - `pub fn ends(t: *Translator) Error!void`
//! - `pub fn global(t: *Translator, toks: []const Tok) Error!void`
//! - `pub fn initial(t: *Translator, card: []const u8, toks: []const Tok) Error!void`
//! - `pub fn cond(t: *Translator, toks: []const Tok, card: []const u8) Error!void`
//! - `pub fn closeBrace(t: *Translator, toks: []const Tok) Error!void`
//! - `pub fn skipped(t: *Translator, toks: []const Tok) Error!void`
//! - `pub fn func(t: *Translator, toks: []const Tok) Error!void`
//! - `pub fn funcBody(t: *Translator, toks: []const Tok) Error!void`

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
const spectre_devices = @import("spectre_devices.zig");

/// `parameters a=1 b`: global `.param`, or a subcircuit's defaults
/// (one with no value has none, §4.6).
pub fn parameters(t: *Translator, toks: []const Tok) Error!void {
    var text: std.ArrayList(u8) = .empty;
    var i: usize = 0;
    while (i < toks.len) {
        if (toks[i].kind != .word) return error.ParseError;
        const key = toks[i].text;
        if (spectre_state.reserved.has(key)) return error.UnsupportedCard;
        if (i + 1 < toks.len and toks[i + 1].is("=")) {
            var j = i + 2;
            var depth: usize = 0;
            while (j < toks.len) : (j += 1) {
                if (toks[j].is("(") or toks[j].is("[")) depth += 1;
                if (toks[j].is(")") or toks[j].is("]")) depth -|= 1;
                // The next declaration: a key with `=`, or a bare key
                // (no default) after a complete value.
                const after_value = toks[j - 1].kind == .word or toks[j - 1].kind == .string or toks[j - 1].is(")") or toks[j - 1].is("]");
                if (depth == 0 and toks[j].kind == .word and j > i + 2 and
                    ((j + 1 < toks.len and toks[j + 1].is("=")) or (after_value and (j + 1 == toks.len or toks[j + 1].kind == .word)))) break;
            }
            try text.print(t.arena, " {s}={s}", .{ key, try spectre_params.value(t.arena, toks[i + 2 .. j]) });
            i = j;
        } else i += 1;
    }
    if (t.subckts.items.len > 0) {
        const o = &t.subckts.items[t.subckts.items.len - 1];
        try o.defaults.appendSlice(t.arena, text.items);
    } else if (text.items.len > 0) try t.emit(".param{s}", .{text.items});
}

/// `model name master params`, or a model group (`{` rows `}`).
pub fn model(t: *Translator, toks: []const Tok) Error!void {
    if (toks.len < 2 or toks[0].kind != .word or toks[1].kind != .word) return error.ParseError;
    const name = toks[0].text;
    const master = toks[1].text;
    var tail = toks[2..];
    const group = tail.len > 0 and tail[tail.len - 1].is("{");
    if (group) tail = tail[0 .. tail.len - 1];
    const ps = try spectre_params.params(t.arena, tail);
    if (group) return t.blocks.append(t.arena, .{ .group = .{ .name = name, .master = master, .kv = ps } });
    try modelCard(t, name, master, ps, &.{});
}

/// A model group's `label: params` row, or its closing `}`.
pub fn groupRow(t: *Translator, toks: []const Tok) Error!void {
    const g = t.blocks.items[t.blocks.items.len - 1].group;
    if (toks[0].is("}")) {
        _ = t.blocks.pop();
        return;
    }
    if (toks.len < 2 or toks[0].kind != .word or !toks[1].is(":")) return error.ParseError;
    const bin = try t.arena.print("{s}.{s}", .{ g.name, toks[0].text });
    try modelCard(t, bin, g.master, g.kv, try spectre_params.params(t.arena, toks[2..]));
}

pub fn subckt(t: *Translator, toks: []const Tok) Error!void {
    if (toks.len == 0 or toks[0].kind != .word) return error.ParseError;
    var head: std.ArrayList(u8) = .empty;
    try head.print(t.arena, ".subckt {s}", .{toks[0].text});
    var ports = toks[1..];
    // `(ports)` or bare ports (§4.5); an open parenthesis must close.
    if (ports.len > 0 and ports[0].is("(")) {
        if (!ports[ports.len - 1].is(")")) return error.ParseError;
        ports = ports[1 .. ports.len - 1];
    }
    for (ports) |tok| {
        if (tok.kind != .word) return error.ParseError;
        try head.print(t.arena, " {s}", .{tok.text});
    }
    try t.out.append(t.arena, "");
    try t.subckts.append(t.arena, .{ .card = t.out.items.len - 1, .head = head.items });
}

pub fn ends(t: *Translator) Error!void {
    const o = t.subckts.pop() orelse return error.ParseError;
    t.out.items[o.card] = try std.mem.concat(t.arena, u8, &.{ o.head, o.defaults.items });
    try t.emit(".ends", .{});
}

/// `global g n...`: the first name is ground (§4.8).
pub fn global(t: *Translator, toks: []const Tok) Error!void {
    if (toks.len == 0 or toks[0].kind != .word) return error.ParseError;
    const g = toks[0].text;
    if (!std.mem.eql(u8, g, "0")) try t.emit(".connect 0 {s}", .{g});
    if (toks.len == 1) return;
    var text: std.ArrayList(u8) = .empty;
    try text.appendSlice(t.arena, ".global");
    for (toks[1..]) |tok| {
        if (tok.kind != .word) return error.ParseError;
        try text.print(t.arena, " {s}", .{tok.text});
    }
    try t.out.append(t.arena, text.items);
}

/// `ic n=v ...` / `nodeset n=v ...` on nodes; an `inst:terminal` signal
/// is refused.
pub fn initial(t: *Translator, card: []const u8, toks: []const Tok) Error!void {
    var text: std.ArrayList(u8) = .empty;
    try text.print(t.arena, ".{s}", .{card});
    for (try spectre_params.params(t.arena, toks)) |p| {
        if (std.mem.indexOfScalar(u8, p.key, ':') != null) return error.UnsupportedCard;
        try text.print(t.arena, " v({s})={s}", .{ p.key, try spectre_params.value(t.arena, p.value) });
    }
    try t.out.append(t.arena, text.items);
}

/// `if (cond) {`, `if (cond) stmt`, and `} else ...` (§4.12).
pub fn cond(t: *Translator, toks: []const Tok, card: []const u8) Error!void {
    if (toks.len == 0 or !toks[0].is("(")) return error.ParseError;
    if (t.if_depth == spectre_expr.depth_max) return error.ParseError;
    t.if_depth += 1;
    defer t.if_depth -= 1;
    var depth: usize = 0;
    var close: usize = 0;
    for (toks, 0..) |tok, i| {
        if (tok.is("(")) depth += 1;
        if (tok.is(")")) {
            depth -= 1;
            if (depth == 0) {
                close = i;
                break;
            }
        }
    } else return error.ParseError;
    try t.emit("{s} ({s})", .{ card, try spectre_expr.expression(t.arena, toks[1..close]) });
    const tail = toks[close + 1 ..];
    if (tail.len == 1 and tail[0].is("{")) return t.blocks.append(t.arena, .cond);
    if (tail.len == 0) return error.ParseError;
    // One statement, no braces, maybe `else stmt` after it; a nested
    // `if` takes the `else` itself (it binds to the closest, R19 p.491).
    const split = if (tail[0].is("if")) tail.len else for (tail, 0..) |tok, i| {
        if (tok.is("else")) break i;
    } else tail.len;
    if (split == 0) return error.ParseError;
    try namedOrStatement(t, tail[0..split]);
    if (split < tail.len) {
        if (split + 1 == tail.len) return error.ParseError;
        try t.emit(".else", .{});
        try namedOrStatement(t, tail[split + 1 ..]);
    }
    try t.emit(".endif", .{});
}

/// `}` closing a block: a branch, maybe followed by `else`.
pub fn closeBrace(t: *Translator, toks: []const Tok) Error!void {
    const b = t.blocks.pop() orelse return error.ParseError;
    assert(b == .cond);
    if (toks.len == 1) return t.emit(".endif", .{});
    if (!toks[1].is("else")) return error.ParseError;
    const tail = toks[2..];
    if (tail.len >= 1 and tail[0].is("if")) {
        // `} else if (c) {`: one chain.
        try cond(t, tail[1..], ".elseif");
        return;
    }
    if (tail.len == 1 and tail[0].is("{")) {
        try t.emit(".else", .{});
        return t.blocks.append(t.arena, .cond);
    }
    return error.ParseError;
}

pub fn skipped(t: *Translator, toks: []const Tok) Error!void {
    var depth: isize = 0;
    for (toks) |tok| {
        if (tok.is("{")) depth += 1;
        if (tok.is("}")) depth -= 1;
    }
    // A nested block opens on its own line; the block's `}` closes it.
    if (depth > 0) return t.blocks.append(t.arena, .skip);
    if (depth < 0) _ = t.blocks.pop();
}

/// `real f(real a, real b) {` with its `return e` and `}` lines.
pub fn func(t: *Translator, toks: []const Tok) Error!void {
    if (toks.len < 3 or toks[0].kind != .word or !toks[1].is("(")) return error.ParseError;
    var text: std.ArrayList(u8) = .empty;
    try text.print(t.arena, ".func {s}(", .{toks[0].text});
    var i: usize = 2;
    var first = true;
    // `real a, real b`: every argument is typed `real` (§4.7).
    while (i < toks.len and !toks[i].is(")")) {
        if (!first) {
            if (!toks[i].is(",")) return error.ParseError;
            i += 1;
        }
        if (i + 1 >= toks.len or !toks[i].is("real") or toks[i + 1].kind != .word) return error.ParseError;
        if (!first) try text.append(t.arena, ',');
        first = false;
        try text.appendSlice(t.arena, toks[i + 1].text);
        i += 2;
    }
    try text.append(t.arena, ')');
    const tail = if (i + 1 < toks.len) toks[i + 1 ..] else return error.ParseError;
    if (!tail[0].is("{")) return error.ParseError;
    // `{ return e }` on one line, or the body on the lines after.
    if (tail.len > 1) {
        const body = tail[1..];
        if (!body[0].is("return") or !body[body.len - 1].is("}")) return error.ParseError;
        var end = body.len - 1;
        if (end > 1 and body[end - 1].is(";")) end -= 1;
        try text.print(t.arena, " {s}", .{try spectre_expr.expression(t.arena, body[1..end])});
        return t.out.append(t.arena, text.items);
    }
    try t.out.append(t.arena, text.items);
    try t.blocks.append(t.arena, .func);
}

pub fn funcBody(t: *Translator, toks: []const Tok) Error!void {
    if (toks[0].is("}")) {
        _ = t.blocks.pop();
        return;
    }
    if (!toks[0].is("return")) return error.ParseError;
    var end = toks.len;
    if (end > 1 and toks[end - 1].is(";")) end -= 1;
    const last = &t.out.items[t.out.items.len - 1];
    last.* = try t.arena.print("{s} {s}", .{ last.*, try spectre_expr.expression(t.arena, toks[1..end]) });
}

// ---- private ----

fn modelCard(t: *Translator, name: []const u8, master: []const u8, ps: []const Param, more: []const Param) Error!void {
    const m = spectre_state.masters.get(master) orelse return error.UnsupportedCard;
    const ty = spectre_params.find(ps, "type") orelse spectre_params.find(more, "type");
    const p = if (ty) |v| v.len == 1 and (std.mem.eql(u8, v[0].text, "p") or std.mem.eql(u8, v[0].text, "pnp")) else false;
    var text: std.ArrayList(u8) = .empty;
    try text.print(t.arena, ".model {s} {s}", .{ name, if (p) m.p else m.n });
    if (m.level) |lv| try text.print(t.arena, " level={d}", .{lv});
    for ([_][]const Param{ ps, more }) |list| for (list) |kv| {
        if (std.mem.eql(u8, kv.key, "type")) continue;
        try text.print(t.arena, " {s}={s}", .{ kv.key, try spectre_params.value(t.arena, kv.value) });
    };
    try t.out.append(t.arena, text.items);
}

fn namedOrStatement(t: *Translator, toks: []const Tok) Error!void {
    if (toks[0].kind == .word and std.mem.eql(u8, toks[0].text, "if")) return cond(t, toks[1..], ".if");
    return spectre_devices.named(t, toks);
}
