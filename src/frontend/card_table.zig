//! The dot-card and element-form tables. In: a card's head word or line.
//! Out: which card or behavioural form it is, and the line predicates the
//! walks ask (a `seed=` option, an analysis card, a `key=`).
//! Cites: dev/research/{ngspice,hspice,spectre}-syntax.md (D1, X4, D4;
//! #121); ngspice inpcom.c (b_transformation_wanted, E/G keywords).
//!
//! Contents:
//! - `pub const Card`
//! - `pub const CondCard`
//! - `pub const Branches`
//! - `pub const port_source_words`
//! - `pub const cards`
//! - `pub const variant_cards`
//! - `pub const Form`
//! - `pub const behavioural`
//! - `pub const data_file_keys`
//! - `pub fn an(kind: Kind) Card`
//! - `pub fn cond(c: CondCard) Card`
//! - `pub fn cardOf(head: []const u8) ?Card`
//! - `pub fn isSn(line: []const u8) bool`
//! - `pub fn seedOption(line: []const u8) bool`
//! - `pub fn analysisOf(line: []const u8) ?Kind`
//! - `pub fn setsKey(line: []const u8, key: []const u8) bool`
//! - `pub fn bTransformationWanted(p: []const u8) bool`
//! - `pub fn kvIndex(kvs: []const Kv, key: []const u8) ?usize`
//! - `pub fn isBinOf(model: []const u8, base: []const u8) bool`
//! - `pub fn binBounds(kv: []const Kv) ?[4]f64`
//! - `pub fn readsLive(ops: []const expr.Op) bool`

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
const card_rows = @import("card_rows.zig");
const netlist_view = @import("netlist_view.zig");
const Kind = card_rows.Kind;
const Kv = card_rows.Kv;
const kvNumber = netlist_view.kvNumber;

/// `ignored`: a card that only shapes printed output, which ESPice writes
/// in full anyway.
pub const Card = union(enum) { end, ends, subckt, param, func, model, include, osdi_include, pre_osdi, verilog, control, endc, options, ic, nodeset, global, connect, save, store, sample, meas, ignored, mosra, appendmodel, step, data, enddata, variation, end_variation, analysis: Kind, cond: CondCard };

pub const CondCard = enum { @"if", elseif, @"else", endif };

/// Open `.if` chains, matching ngspice's recifeval (inp.c): each chain keeps
/// the lines of its first true branch and drops the rest.
pub const Branches = struct {
    open: [max]State = undefined,
    depth: u8 = 0,

    /// `live`: in the kept branch; `pending`: no branch held yet; `spent`:
    /// a branch already held, or the whole chain sits in a dropped region.
    const State = enum(u2) { live, pending, spent };
    const max = 32;

    fn active(b: Branches) bool {
        return b.depth == 0 or b.open[b.depth - 1] == .live;
    }
};

/// Source keywords that may follow a P card's nodes (`mixedPort`).
pub const port_source_words = std.StaticStringMap(void).initComptime(.{
    .{"dc"},  .{"ac"},   .{"hb"}, .{"hbac"}, .{"pulse"}, .{"sin"},     .{"exp"},
    .{"pwl"}, .{"sffm"}, .{"am"}, .{"lfsr"}, .{"pat"},   .{"distof1"}, .{"distof2"},
});

/// Every dot card the three dialects accept, looked up lowercased.
pub const cards = std.StaticStringMap(Card).initComptime(.{
    .{ "end", .end },                .{ "ends", .ends },                 .{ "subckt", .subckt },
    // HSPICE's `.macro`/`.eom` and `.parameter(s)`/`.temperature` spellings,
    // ngspice's too for `.macro` (hspice-syntax.md D1, X4, D4; #121).
    .{ "macro", .subckt },           .{ "eom", .ends },                  .{ "parameter", .param },
    .{ "parameters", .param },       .{ "temperature", an(.temp) },      .{ "func", .func },
    .{ "param", .param },            .{ "model", .model },               .{ "include", .include },
    .{ "hdl", .include },            .{ "osdi_include", .osdi_include }, .{ "pre_osdi", .pre_osdi },
    .{ "verilog", .verilog },        .{ "option", .options },            .{ "options", .options },
    .{ "opt", .options },            .{ "opts", .options },              .{ "ic", .ic },
    .{ "ac", an(.ac) },              .{ "dc", an(.dc) },                 .{ "dcmatch", an(.dcmatch) },
    .{ "disto", an(.disto) },        .{ "envelope", an(.envelope) },     .{ "envlp", an(.envelope) },
    .{ "four", an(.four) },          .{ "hb", an(.hb) },                 .{ "matex", an(.matex) },
    .{ "hbac", an(.hbac) },          .{ "hbnoise", an(.hbnoise) },       .{ "hbxf", an(.hbxf) },
    .{ "hbosc", an(.hb) },           .{ "snosc", an(.pss) },             .{ "phasenoise", an(.phasenoise) },
    .{ "mc", an(.mc) },              .{ "montecarlo", an(.mc) },         .{ "noise", an(.noise) },
    .{ "op", an(.op) },              .{ "pac", an(.pac) },               .{ "pnoise", an(.pnoise) },
    .{ "pss", an(.pss) },            .{ "pxf", an(.pxf) },               .{ "pz", an(.pz) },
    .{ "qpss", an(.qpss) },          .{ "sens", an(.sens) },             .{ "sp", an(.sp) },
    .{ "stb", an(.stb) },            .{ "temp", an(.temp) },             .{ "tf", an(.tf) },
    .{ "sn", an(.pss) },             .{ "snac", an(.pac) },              .{ "snnoise", an(.pnoise) },
    .{ "snxf", an(.pxf) },           .{ "fft", an(.fft) },               .{ "ptdnoise", an(.pnoise) },
    .{ "tran", an(.tran) },          .{ "trannoise", an(.tran_noise) },  .{ "tran_noise", an(.tran_noise) },
    .{ "lstb", an(.lstb) },          .{ "acxf", an(.acxf) },             .{ "dcxf", an(.dcxf) },
    .{ "dcinc", an(.dcinc) },        .{ "lin", an(.sp) },                .{ "acmatch", an(.acmatch) },
    .{ "dcsens", an(.dcsens) },      .{ "hblin", an(.hblin) },           .{ "net", an(.sp) },
    .{ "acphasenoise", an(.noise) }, .{ "if", cond(.@"if") },            .{ "elseif", cond(.elseif) },
    .{ "else", cond(.@"else") },     .{ "endif", cond(.endif) },         .{ "meas", .meas },
    .{ "measure", .meas },           .{ "save", .save },                 .{ "dcvolt", .ic },
    .{ "nodeset", .nodeset },        .{ "global", .global },             .{ "connect", .connect },
    .{ "jitter", .meas },            .{ "store", .store },               .{ "sample", .sample },
    .{ "mosra", .mosra },            .{ "appendmodel", .appendmodel },   .{ "check", .meas },
    .{ "dout", .meas },              .{ "biaschk", .meas },              .{ "power", .meas },
    .{ "pat", .ignored },            .{ "stim", .meas },                 .{ "stimuli", .meas },
    .{ "control", .control },        .{ "endc", .endc },                 .{ "print", .ignored },
    .{ "plot", .ignored },           .{ "probe", .ignored },             .{ "graph", .ignored },
    .{ "width", .ignored },          .{ "title", .ignored },             .{ "protect", .ignored },
    .{ "unprotect", .ignored },      .{ "prot", .ignored },              .{ "unprot", .ignored },
});

/// Sweep and variation cards (frontend/variants.zig).
pub const variant_cards = std.StaticStringMap(Card).initComptime(.{
    .{ "step", .step }, .{ "data", .data }, .{ "enddata", .enddata }, .{ "variation", .variation }, .{ "end_variation", .end_variation },
});

/// A behavioural E/F/G/H form, named by the word in the first control-node
/// slot (ngspice inpcom.c, HSPICE's E/G element keywords).
pub const Form = enum { value, poly, table, pwl, laplace, pole, delay, vcr, vccap, kind, refused };

pub const behavioural = std.StaticStringMap(Form).initComptime(.{
    .{ "poly", .poly },    .{ "value", .value },     .{ "vol", .value },   .{ "cur", .value },
    .{ "table", .table },  .{ "laplace", .laplace }, .{ "pole", .pole },   .{ "freq", .refused },
    .{ "vcr", .vcr },      .{ "vccap", .vccap },     .{ "delay", .delay }, .{ "opamp", .refused },
    .{ "npwl", .refused }, .{ "ppwl", .refused },    .{ "pwl", .pwl },     .{ "and", .refused },
    .{ "nand", .refused }, .{ "or", .refused },      .{ "nor", .refused }, .{ "vcvs", .kind },
    .{ "vccs", .kind },    .{ "ccvs", .kind },       .{ "cccs", .kind },   .{ "transformer", .refused },
});

/// Keys whose value is a data file an HSPICE W or S card reads.
pub const data_file_keys = std.StaticStringMap(void).initComptime(.{
    .{ "rlgcfile", {} }, .{ "tstonefile", {} }, .{ "citifile", {} },
});

pub fn an(kind: Kind) Card {
    return .{ .analysis = kind };
}

pub fn cond(c: CondCard) Card {
    return .{ .cond = c };
}

/// The card a `.keyword` names, case-insensitively; null for any other card.
pub fn cardOf(head: []const u8) ?Card {
    var buf: [16]u8 = undefined;
    if (head.len > buf.len) return null;
    const lower = std.ascii.lowerString(buf[0..head.len], head);
    return cards.get(lower) orelse variant_cards.get(lower);
}

/// `.sn`, `.snac`, `.snnoise`, `.snxf` or `.ptdnoise`: the HSPICE
/// shooting-Newton spelling of a periodic card. `.snosc` keeps the `.pss`
/// form.
pub fn isSn(line: []const u8) bool {
    const words = std.StaticStringMap(void).initComptime(.{ .{"sn"}, .{"snac"}, .{"snnoise"}, .{"snxf"}, .{"ptdnoise"} });
    const head = line[1 .. std.mem.indexOfAny(u8, line, " \t") orelse line.len];
    var buf: [8]u8 = undefined;
    return head.len <= buf.len and words.has(std.ascii.lowerString(buf[0..head.len], head));
}

/// A `.option`/`.options` card (lowercased) that sets `seed=`.
pub fn seedOption(line: []const u8) bool {
    return std.mem.startsWith(u8, line, ".opt") and setsKey(line, "seed");
}

/// The analysis a card (lowercased) runs; null for any other card.
pub fn analysisOf(line: []const u8) ?Kind {
    if (line.len == 0 or line[0] != '.') return null;
    // `cardOf` takes the head without its dot.
    const head = line[1 .. std.mem.indexOfAny(u8, line, " \t") orelse line.len];
    const card = cardOf(head) orelse return null;
    return if (card == .analysis) card.analysis else null;
}

/// Whether `line` (lowercased) holds `key=` with `key` a whole word.
pub fn setsKey(line: []const u8, key: []const u8) bool {
    std.debug.assert(key.len > 0);
    var at: usize = 0;
    while (std.mem.indexOfPos(u8, line, at, key)) |k| : (at = k + key.len) {
        if (k > 0 and (std.ascii.isAlphanumeric(line[k - 1]) or line[k - 1] == '_')) continue;
        const rest = std.mem.trimStart(u8, line[k + key.len ..], " \t");
        if (rest.len > 0 and rest[0] == '=') return true;
    }
    return false;
}

/// ngspice b_transformation_wanted (inpcom.c): `p` reads a node voltage,
/// a branch current, `time`, `temper` or `hertz`, so a passive card holding
/// it becomes a B source.
pub fn bTransformationWanted(p: []const u8) bool {
    for (p, 0..) |c, i| {
        if (std.mem.indexOfScalar(u8, "vith", c) == null) continue;
        if (i > 0 and (std.ascii.isAlphanumeric(p[i - 1]) or p[i - 1] == '_')) continue;
        const tail = p[i..];
        if (std.mem.startsWith(u8, tail, "v(") or std.mem.startsWith(u8, tail, "i(")) return true;
        for ([_][]const u8{ "temper", "hertz", "time" }) |w| {
            if (!std.mem.startsWith(u8, tail, w)) continue;
            if (tail.len == w.len or !(std.ascii.isAlphanumeric(tail[w.len]) or tail[w.len] == '_')) return true;
        }
    }
    return false;
}

/// Row of `key` in `kvs`.
pub fn kvIndex(kvs: []const Kv, key: []const u8) ?usize {
    for (kvs, 0..) |kv, i| if (std.mem.eql(u8, kv.key, key)) return i;
    return null;
}

/// Whether `model` is a bin of `base`: `base.<n>`.
pub fn isBinOf(model: []const u8, base: []const u8) bool {
    if (model.len <= base.len + 1 or !std.mem.startsWith(u8, model, base) or model[base.len] != '.') return false;
    _ = std.fmt.parseInt(u32, model[base.len + 1 ..], 10) catch return false;
    return true;
}

/// A bin card's `{lmin, lmax, wmin, wmax}`, null unless all four are numbers.
pub fn binBounds(kv: []const Kv) ?[4]f64 {
    return .{
        kvNumber(kv, "lmin") orelse return null, kvNumber(kv, "lmax") orelse return null,
        kvNumber(kv, "wmin") orelse return null, kvNumber(kv, "wmax") orelse return null,
    };
}

/// Whether postfix `ops` reads a live name.
pub fn readsLive(ops: []const expr.Op) bool {
    for (ops) |op| if (op.code == .live) return true;
    return false;
}

// ---- private ----
