//! Analysis-card argument readers. In: one card's values. Out: its
//! numbers, counts, keywords, output rows, sources and frequency grids,
//! each checked once here, so a malformed card is an error before any
//! query is built.
//! Cites: ngspice's card readers (inpgval.c, spiceif.c, cktsopt.c); HSPICE
//! [CR] for the sweep forms.
//!
//! Contents:
//! - `pub const NO_NODE`
//! - `pub const CardContext`
//! - `pub fn hasNumber(args: []const Value) bool`
//! - `pub fn number(args: []const Value, i: usize) !f64`
//! - `pub fn positive(args: []const Value, i: usize) !f64`
//! - `pub fn count(comptime T: type, args: []const Value, i: usize, default: T) !T`
//! - `pub fn arity(args: []const Value, min: usize, max: usize) !void`
//! - `pub fn keyword(args: []const Value, i: usize, buf: []u8) ![]const u8`
//! - `pub fn hasUic(args: []const Value) bool`
//! - `pub fn outputNode(node: u32) !u32`
//! - `pub fn outputNeg(node: u32) !u32`
//! - `pub fn currentProbeName(args: []const Value, i: usize) ?[]const u8`
//! - `pub fn inputSource(args: []const Value, i: usize, sources: core.QueryBindings) !void`
//! - `pub fn singleEnded(node_neg: u32) !void`
//! - `pub fn voltageSource(args: []const Value, i: usize, sources: core.QueryBindings) !usize`
//! - `pub fn checkStep(start: f64, stop: f64, step: f64) !void`
//! - `pub fn frequencySweep(args: []const Value, offset: usize) !numerics.FreqSweep`
//! - `pub fn acGrid(arena: std.mem.Allocator, args: []const Value, offset: usize) !struct { sweep: numerics.FreqSweep, end: usize }`
//! - `pub fn outputLabel(arena: std.mem.Allocator, value: Value) ![]const u8`
//! - `pub fn cardIs(line: []const u8, name: []const u8) bool`
//! - `pub fn isKey(args: []const Value, i: usize, key: []const u8) bool`

const std = @import("std");
const assert = std.debug.assert;
const testing = std.testing;
const Allocator = std.mem.Allocator;
const core = @import("core");
const requests = core.query;
const numerics = core.numerics;
const netlist = @import("netlist");
const Value = netlist.Value;
const nameAt = Value.nameAt;
const numberAt = Value.numberAt;
const Job = requests.Query;
const GROUND = core.GROUND;
const Library = @import("device").Library;

/// Unresolved or absent node row.
pub const NO_NODE = netlist.none;

/// What a card borrows from the rest of the deck: HSPICE's `.noise` and
/// `.lstb` run over the `.ac` sweep, `.fft` and `.op <time>` over the `.tran`, and
/// `.snac`/`.snnoise`/`.snxf` at the `.sn` fundamental.
pub const CardContext = struct {
    /// Owns the point lists and labels a card allocates.
    arena: std.mem.Allocator,
    /// The first `.ac` card's grid.
    ac: ?numerics.FreqSweep = null,
    /// The first `.tran` card, deck options applied.
    tran: ?requests.Tran = null,
    /// The first `.sn` card's fundamental, in Hz.
    sn_f0: ?f64 = null,
    /// The deck's variation groups (`DeckOptions.variations`).
    variations: requests.Variations = .{},
    /// The deck's card names (`DeckOptions.card_names`).
    card_names: requests.CardNames = .empty,
};

pub fn hasNumber(args: []const Value) bool {
    for (args) |v| if (v == .num) return true;
    return false;
}

pub fn number(args: []const Value, i: usize) !f64 {
    const n = numberAt(args, i) orelse return error.InvalidAnalysisArguments;
    if (!std.math.isFinite(n)) return error.InvalidAnalysisArguments;
    return n;
}

pub fn positive(args: []const Value, i: usize) !f64 {
    const n = try number(args, i);
    if (n <= 0) return error.InvalidAnalysisArguments;
    return n;
}

pub fn count(comptime T: type, args: []const Value, i: usize, default: T) !T {
    if (i >= args.len) return default;
    const n = try positive(args, i);
    if (n != @trunc(n) or n > std.math.maxInt(T)) return error.InvalidAnalysisArguments;
    return @intFromFloat(n);
}

pub fn arity(args: []const Value, min: usize, max: usize) !void {
    if (args.len < min or args.len > max) return error.InvalidAnalysisArguments;
}

/// One case-insensitive keyword argument, lowered into caller storage so the
/// `StaticStringMap` lookup that follows stays allocation-free.
pub fn keyword(args: []const Value, i: usize, buf: []u8) ![]const u8 {
    const name = nameAt(args, i) orelse return error.InvalidAnalysisArguments;
    if (name.len > buf.len) return error.InvalidAnalysisArguments;
    return std.ascii.lowerString(buf[0..name.len], name);
}

/// `uic` is a trailing keyword, not a positional: `.tran 1n 100n uic` and
/// `.tran 1n 100n 0 1n uic` are both legal, so scan rather than index.
pub fn hasUic(args: []const Value) bool {
    for (args) |a| switch (a) {
        .name => |n| if (std.ascii.eqlIgnoreCase(n, "uic")) return true,
        else => {},
    };
    return false;
}

pub fn outputNode(node: u32) !u32 {
    if (node == NO_NODE or node == GROUND) return error.AnalysisNodeNotFound;
    return node;
}

/// `v(a,b)` second node. Absent (single-ended) resolves to GROUND, which is
/// what every consumer already means by "no reference node"; a name the deck
/// never defines is a typo, not a ground reference.
pub fn outputNeg(node: u32) !u32 {
    if (node == NO_NODE) return GROUND;
    return node;
}

/// `i(name)`: a branch-current probe, as opposed to a `v(...)` output.
pub fn currentProbeName(args: []const Value, i: usize) ?[]const u8 {
    if (i >= args.len) return null;
    return switch (args[i]) {
        .group => |g| if (std.ascii.eqlIgnoreCase(g.name(), "i") and g.args().len == 1)
            switch (g.args()[0]) {
                .name => |n| n,
                else => null,
            }
        else
            null,
        else => null,
    };
}

/// A noise card's input source `args[i]`: a V or I card the deck has. It
/// names the input but does not drive the solve.
pub fn inputSource(args: []const Value, i: usize, sources: core.QueryBindings) !void {
    const name = nameAt(args, i) orelse return error.InvalidAnalysisArguments;
    if (netlist.nameIndex(sources.v_names, name) == null and netlist.nameIndex(sources.i_names, name) == null) return error.AnalysisSourceNotFound;
}

/// An output the card reads single-ended: `v(a,b)` with b not ground is
/// `UnsupportedAnalysisOutput`, never silently v(a) (#57).
pub fn singleEnded(node_neg: u32) !void {
    if (node_neg != NO_NODE and node_neg != GROUND) return error.UnsupportedAnalysisOutput;
}

pub fn voltageSource(args: []const Value, i: usize, sources: core.QueryBindings) !usize {
    const name = nameAt(args, i) orelse return error.InvalidAnalysisArguments;
    return netlist.nameIndex(sources.v_names, name) orelse error.AnalysisSourceNotFound;
}

pub fn checkStep(start: f64, stop: f64, step: f64) !void {
    const intervals = (stop - start) / step;
    if (step == 0 or !std.math.isFinite(intervals) or intervals < 0 or
        intervals >= @as(f64, @floatFromInt(std.math.maxInt(usize)))) return error.InvalidAnalysisArguments;
}

/// The `dec|oct|lin N fstart fstop` grid every frequency-domain card shares,
/// read from `offset`. Only `lin` admits fstart = 0.
pub fn frequencySweep(args: []const Value, offset: usize) !numerics.FreqSweep {
    const mode = nameAt(args, offset) orelse return error.InvalidAnalysisArguments;
    const kinds = std.StaticStringMap(numerics.SweepKind).initComptime(.{
        .{ "dec", .dec }, .{ "oct", .oct }, .{ "lin", .lin },
    });
    var lower: [8]u8 = undefined;
    if (mode.len > lower.len) return error.UnsupportedFrequencySweep;
    const kind = kinds.get(std.ascii.lowerString(lower[0..mode.len], mode)) orelse return error.UnsupportedFrequencySweep;
    const first = if (kind == .lin) try number(args, offset + 2) else try positive(args, offset + 2);
    const last = try positive(args, offset + 3);
    if (first < 0 or last < first) return error.InvalidAnalysisArguments;
    return .{ .f_start = first, .f_stop = last, .points = try count(u32, args, offset + 1, 10), .kind = kind };
}

/// A frequency grid at `args[offset..]` and the index past it: the
/// `dec|oct|lin N fstart fstop` grid, or HSPICE's `POI n f1 ... fn` [CR .AC]
/// with its frequencies ascending and positive, allocated in `arena`.
pub fn acGrid(arena: std.mem.Allocator, args: []const Value, offset: usize) !struct { sweep: numerics.FreqSweep, end: usize } {
    var lower: [4]u8 = undefined;
    const poi = nameAt(args, offset) != null and std.mem.eql(u8, keyword(args, offset, &lower) catch "", "poi");
    if (!poi) return .{ .sweep = try frequencySweep(args, offset), .end = offset + 4 };
    const n = try count(u32, args, offset + 1, 0);
    // Checked before the allocation: `poi 4e9 1` must not ask for 32 GB.
    if (n == 0 or args.len < offset + 2 + n) return error.InvalidAnalysisArguments;
    const list = try arena.alloc(f64, n);
    for (list, 0..) |*f, k| {
        f.* = try positive(args, offset + 2 + k);
        if (k > 0 and !(f.* > list[k - 1])) return error.InvalidAnalysisArguments;
    }
    return .{ .sweep = .{ .kind = .poi, .points = n, .list = list, .f_start = list[0], .f_stop = list[n - 1] }, .end = offset + 2 + n };
}

/// An output as the card wrote it, `v(a)` or `v(a,b)`, allocated in `arena`.
pub fn outputLabel(arena: std.mem.Allocator, value: Value) ![]const u8 {
    const g = switch (value) {
        .group => |g| g,
        else => return error.InvalidAnalysisArguments,
    };
    var out: std.Io.Writer.Allocating = .init(arena);
    const w = &out.writer;
    w.print("{s}(", .{g.name()}) catch return error.OutOfMemory;
    for (g.args(), 0..) |v, k| {
        if (k > 0) w.writeByte(',') catch return error.OutOfMemory;
        switch (v) {
            .name => |n| w.writeAll(n) catch return error.OutOfMemory,
            .num => |n| w.print("{d}", .{n}) catch return error.OutOfMemory,
            else => return error.InvalidAnalysisArguments,
        }
    }
    w.writeByte(')') catch return error.OutOfMemory;
    return out.written();
}

/// True when `line` is the dot card `name` (HSPICE `.lin` and `.net` run
/// as `.sp` queries).
pub fn cardIs(line: []const u8, name: []const u8) bool {
    const head = line[@min(line.len, 1) .. std.mem.indexOfAny(u8, line, " \t") orelse line.len];
    return std.ascii.eqlIgnoreCase(head, name);
}

pub fn isKey(args: []const Value, i: usize, key: []const u8) bool {
    return std.ascii.eqlIgnoreCase(nameAt(args, i) orelse return false, key);
}

// ---- private ----

/// A comptime argument list: strings become names, numbers numbers.
fn argv(comptime xs: anytype) []const Value {
    const final = comptime blk: {
        var out: [xs.len]Value = undefined;
        for (xs, 0..) |x, k| out[k] = if (@typeInfo(@TypeOf(x)) == .pointer) .{ .name = x } else .{ .num = x };
        break :blk out;
    };
    return &final;
}

test checkStep {
    try checkStep(0, 1, 0.1);
    try checkStep(1, 0, -0.1);
    try checkStep(1, 1, 0.5);
    for ([_][3]f64{ .{ 0, 1, 0 }, .{ 0, 1, -0.1 }, .{ 0, std.math.nan(f64), 1 }, .{ 0, 1e300, 1e-300 } }) |c|
        try testing.expectError(error.InvalidAnalysisArguments, checkStep(c[0], c[1], c[2]));
}

test frequencySweep {
    const lin_grid = try frequencySweep(argv(.{ "LIN", 5, 0, 10 }), 0);
    try testing.expectEqual(numerics.SweepKind.lin, lin_grid.kind);
    try testing.expectEqual(@as(u32, 5), lin_grid.points);
    try testing.expectError(error.InvalidAnalysisArguments, frequencySweep(argv(.{ "dec", 5, 0, 10 }), 0));
    try testing.expectError(error.InvalidAnalysisArguments, frequencySweep(argv(.{ "lin", 5, -1, 10 }), 0));
    try testing.expectError(error.UnsupportedFrequencySweep, frequencySweep(argv(.{ "log", 5, 1, 10 }), 0));
    try testing.expectError(error.UnsupportedFrequencySweep, frequencySweep(argv(.{ "decadesss", 5, 1, 10 }), 0));
    try testing.expectError(error.InvalidAnalysisArguments, frequencySweep(argv(.{"dec"}), 0));
}

test acGrid {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const poi = try acGrid(a, argv(.{ "x", "POI", 3, 1, 2, 3 }), 1);
    try testing.expectEqual(numerics.SweepKind.poi, poi.sweep.kind);
    try testing.expectEqual(@as(usize, 6), poi.end);
    try testing.expectEqual(@as(f64, 3), poi.sweep.f_stop);
    try testing.expectEqual(@as(usize, 4), (try acGrid(a, argv(.{ "dec", 2, 10, 100 }), 0)).end);
    // Not ascending, a zero count, and a count the card does not carry
    // (which must fail before it allocates 4e9 points).
    for ([_][]const Value{ argv(.{ "poi", 2, 2, 1 }), argv(.{ "poi", 0 }), argv(.{ "poi", 4e9, 1 }), argv(.{ "poi", 3, 1, 2 }) }) |args|
        try testing.expectError(error.InvalidAnalysisArguments, acGrid(a, args, 0));
}

test outputLabel {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    try testing.expectEqualStrings("v(a,2)", try outputLabel(arena.allocator(), .{ .group = .{ .items = argv(.{ "v", "a", 2 }) } }));
    try testing.expectError(error.InvalidAnalysisArguments, outputLabel(arena.allocator(), .{ .name = "v" }));
}

test cardIs {
    try testing.expect(cardIs(".LIN sparcalc=1", "lin"));
    try testing.expect(cardIs(".net\tvin", "net"));
    try testing.expect(!cardIs(".linx", "lin"));
    try testing.expect(!cardIs("", "lin"));
}
