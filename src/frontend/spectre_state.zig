//! The Spectre translator's state and fact tables. In: nothing (the state
//! the statement readers share, and the keyword tables they read). Out: the
//! `Translator` row store: emitted SPICE cards, their device letters, the
//! names pass 1 declared, and the open blocks.
//! Cites: dev/research/spectre-syntax.md §4.3 (primitives), §4.9-§4.10
//! (analysis and control keywords), §6.4 (names used before definition);
//! dev/frontend.md "Spectre dialect" (the master table).
//!
//! Contents:
//! - `pub const tran_keys, tran_methods, reserved, primitives, masters, analysis_kinds, control_kinds`
//! - `pub const Master = struct`
//! - `pub const ControlKind = enum { options, ignored, refused }`
//! - `pub const Translator = struct`
//! - `pub const Open = struct`
//! - `pub const Block = union(enum)`
//! - `pub fn emit(t: *Translator, comptime fmt: []const u8, args: anytype) Error!void`
//! - `pub fn device(t: *Translator, letter: u8, comptime fmt: []const u8, args: anytype) Error!void`
//! - `pub fn declare(t: *Translator, in: []const []const u8) Error!void`

const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;
const lines = @import("lines.zig");
const spectre_expr = @import("spectre_expr.zig");
const spectre_params = @import("spectre_params.zig");
const Error = spectre_expr.Error;
const Param = spectre_params.Param;
const switchesTo = lines.spectre.Lines.switchesTo;

/// The `tran` parameters mapped; any other is refused by name.
pub const tran_keys = std.StaticStringMap(void).initComptime(.{ .{"stop"}, .{"step"}, .{"start"}, .{"maxstep"}, .{"method"}, .{"skipdc"}, .{"errpreset"} });

/// Spectre `tran method=` as the `.options` that select it; `trapgear2` has
/// no ESPice equivalent and is refused.
pub const tran_methods = std.StaticStringMap([]const u8).initComptime(.{
    .{ "euler", "method=gear maxord=1" }, .{ "trap", "method=trap" },      .{ "traponly", "method=trap" },
    .{ "gear2", "method=gear" },          .{ "gear2only", "method=gear" },
});

/// Parameter names Spectre reserves (R11 p.361, R19 p.508).
pub const reserved = std.StaticStringMap(void).initComptime(.{ .{"temp"}, .{"tnom"}, .{"scale"}, .{"scalem"}, .{"freq"}, .{"time"} });

/// §4.3: the primitives and their SPICE letters.
pub const primitives = std.StaticStringMap(u8).initComptime(.{
    .{ "resistor", 'r' }, .{ "capacitor", 'c' },       .{ "inductor", 'l' }, .{ "vsource", 'v' },
    .{ "isource", 'i' },  .{ "vcvs", 'e' },            .{ "vccs", 'g' },     .{ "ccvs", 'h' },
    .{ "cccs", 'f' },     .{ "mutual_inductor", 'k' }, .{ "iprobe", 'v' },   .{ "bsource", 'b' },
    .{ "tline", 't' },
});

/// A model master's SPICE type (its letter, the type word for n/p, the
/// LEVEL) (dev/frontend.md, the model-master table).
pub const Master = struct { letter: u8, n: []const u8, p: []const u8, level: ?u16 = null };
pub const masters = std.StaticStringMap(Master).initComptime(.{
    .{ "resistor", Master{ .letter = 'r', .n = "r", .p = "r" } },
    .{ "capacitor", Master{ .letter = 'c', .n = "c", .p = "c" } },
    .{ "inductor", Master{ .letter = 'l', .n = "l", .p = "l" } },
    .{ "diode", Master{ .letter = 'd', .n = "d", .p = "d", .level = 1 } },
    .{ "bjt", Master{ .letter = 'q', .n = "npn", .p = "pnp", .level = 1 } },
    .{ "vbic", Master{ .letter = 'q', .n = "npn", .p = "pnp", .level = 4 } },
    .{ "hicum", Master{ .letter = 'q', .n = "npn", .p = "pnp", .level = 8 } },
    .{ "mos1", Master{ .letter = 'm', .n = "nmos", .p = "pmos", .level = 1 } },
    .{ "mos2", Master{ .letter = 'm', .n = "nmos", .p = "pmos", .level = 2 } },
    .{ "mos3", Master{ .letter = 'm', .n = "nmos", .p = "pmos", .level = 3 } },
    .{ "bsim3v3", Master{ .letter = 'm', .n = "nmos", .p = "pmos", .level = 8 } },
    .{ "bsim4", Master{ .letter = 'm', .n = "nmos", .p = "pmos", .level = 54 } },
    .{ "bsimsoi", Master{ .letter = 'm', .n = "nmos", .p = "pmos", .level = 10 } },
    .{ "hisim2", Master{ .letter = 'm', .n = "nmos", .p = "pmos", .level = 68 } },
    .{ "hisimhv", Master{ .letter = 'm', .n = "nmos", .p = "pmos", .level = 73 } },
    .{ "psp103", Master{ .letter = 'm', .n = "nmos", .p = "pmos", .level = 1040 } },
    .{ "jfet", Master{ .letter = 'j', .n = "njf", .p = "pjf", .level = 1 } },
});

/// §4.9 and §4.10: analysis and control type keywords after a name.
pub const analysis_kinds = std.StaticStringMap(void).initComptime(.{
    .{"acmatch"}, .{"ac"},   .{"dcmatch"}, .{"dc"},         .{"envlp"}, .{"hbac"},    .{"hbnoise"}, .{"hbsp"},
    .{"hbstb"},   .{"hbxf"}, .{"hb"},      .{"montecarlo"}, .{"noise"}, .{"pac"},     .{"pnoise"},  .{"psp"},
    .{"pss"},     .{"pstb"}, .{"pxf"},     .{"pz"},         .{"qpac"},  .{"qpnoise"}, .{"qpsp"},    .{"qpss"},
    .{"qpxf"},    .{"sp"},   .{"stb"},     .{"sweep"},      .{"tdr"},   .{"tran"},    .{"xf"},      .{"fourier"},
    .{"pdisto"},  .{"sens"},
});
/// What a control statement does here: `options` set options, the rest are
/// read and not acted on, or refused by name.
pub const ControlKind = enum { options, ignored, refused };
pub const control_kinds = std.StaticStringMap(ControlKind).initComptime(.{
    .{ "options", .options }, .{ "info", .ignored },       .{ "set", .ignored },      .{ "shell", .ignored },
    .{ "check", .ignored },   .{ "checklimit", .ignored }, .{ "assert", .ignored },   .{ "paramtest", .ignored },
    .{ "alter", .refused },   .{ "altergroup", .refused }, .{ "paramset", .refused },
});

/// The state every statement reader writes; one per `translate`.
pub const Translator = struct {
    arena: Allocator,
    out: std.ArrayList([]const u8) = .empty,
    /// Per `out` row up to the last device card, its letter (0: none).
    kinds: std.ArrayList(u8) = .empty,
    /// Model names and their masters; subcircuit names (any depth).
    models: std.StringHashMapUnmanaged([]const u8) = .empty,
    subckts: std.ArrayList(Open) = .empty,
    subckt_names: std.StringHashMapUnmanaged(void) = .empty,
    /// `ahdl_include` was given: an unknown master is a Verilog-A module.
    ahdl: bool = false,
    ahdl_models: std.StringHashMapUnmanaged(void) = .empty,
    /// Open braces: `if` branches, a `statistics` block, a model group, a
    /// function body.
    blocks: std.ArrayList(Block) = .empty,
    /// One-line `if` statements nested inside the one being read; past
    /// `Expr.depth_max` the line is a ParseError, not a stack overflow.
    if_depth: u32 = 0,

    /// A subcircuit whose `.subckt` card waits for its body's
    /// `parameters`: the card's index, its name and ports, its defaults.
    pub const Open = struct { card: usize, head: []const u8, defaults: std.ArrayList(u8) = .empty };
    /// An open brace.
    pub const Block = union(enum) { cond, skip, group: struct { name: []const u8, master: []const u8, kv: []const Param }, func };

    /// Appends one formatted card.
    pub fn emit(t: *Translator, comptime fmt: []const u8, args: anytype) Error!void {
        try t.out.append(t.arena, try t.arena.print(fmt, args));
    }

    /// Appends one device card and records its letter.
    pub fn device(t: *Translator, letter: u8, comptime fmt: []const u8, args: anytype) Error!void {
        assert(letter != 0);
        const card = try t.arena.print(fmt, args);
        try t.out.append(t.arena, card);
        try t.kinds.appendNTimes(t.arena, 0, t.out.items.len - 1 - t.kinds.items.len);
        try t.kinds.append(t.arena, letter);
        assert(t.kinds.items.len == t.out.items.len);
    }

    /// Pass 1: model, subcircuit and module names, which instances before
    /// their definitions may name (§6.4).
    pub fn declare(t: *Translator, in: []const []const u8) Error!void {
        var spice = false;
        for (in) |line| {
            if (spice) {
                if (switchesTo(line, "spectre")) spice = false;
                continue;
            }
            if (switchesTo(line, "spice")) {
                spice = true;
                continue;
            }
            const toks = spectre_expr.tokenize(t.arena, line) catch continue;
            if (toks.len < 2 or toks[0].kind != .word) continue;
            const head = toks[0].text;
            if (std.mem.eql(u8, head, "model") and toks.len >= 3) try t.models.put(t.arena, toks[1].text, toks[2].text);
            if (std.mem.eql(u8, head, "subckt")) try t.subckt_names.put(t.arena, toks[1].text, {});
            if (std.mem.eql(u8, head, "inline") and toks.len >= 3) try t.subckt_names.put(t.arena, toks[2].text, {});
            if (std.mem.eql(u8, head, "ahdl_include")) t.ahdl = true;
        }
    }
};
