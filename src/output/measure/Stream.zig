//! The `.tran` cards of a deck measured row by row as a transient accepts
//! rows: each accepted row in, each card's passes advanced and its value
//! once no later row can change it out, by the passes `print` makes over a
//! finished result, so with the same values, to the bit. Early stop and the
//! abort bound read which cards have resolved:
//! - WHEN, TRIG/TARG, FIND/DERIV with WHEN or AT: at the event, or past
//!   every TO= (or AT=) without one; a LAST count only past TO=.
//! - AVG, MIN, MAX, PP, RMS, INTEG, EM_AVG, ERR*: past a nonzero TO=.
//! - PARAM: once every card it reads has resolved.
//! - `_CONT`, JITTER, the checks and DOUT, a bare FIND, and cards reading
//!   a later card or a PARAM: never; they read the whole run.
//! Cards without TO= resolve at the run's end, recorded or not. `measured`
//! gives what each `.scan` card found, whose columns need no recording
//! (`markColumns` with `streamed`); every other card reads the recording.
//! A card reading an earlier card's value starts once that resolves,
//! replaying the recording so far, as do TARG with TD=TRIG and FIND/DERIV
//! with a WHEN of TD=, FROM= or a LAST count at their event.
//! O(open cards) per row; a replay is O(rows) once per card.
//! Cites: dev/research/lane-batched-simulation.md §7.
//!
//! Contents:
//! - fields `arena`, `measures`, `cards`, `values`, `open`, `live`, `names`, `nodes`, `row`,
//!   `saved`, `recorded`, `rows`, `abort_card`
//! - `pub const Card = struct` with `pub fn resolved(Card) bool`
//! - `pub const none: u32`
//! - `pub const DeviceScans = struct { scans: []scan.Dev, cols: []scan.DevColumns }`
//! - `pub fn init(gpa: Allocator, measures: []const core.Measure, names: []const []const u8, nodes: []const u32, recorded: []const []const u8) Allocator.Error!Stream`
//! - `pub fn deinit(s: *Stream) void`
//! - `pub fn feed(s: *Stream, t: f64, x: []const f64, data: []const f64) void`
//! - `pub fn unresolved(s: Stream) bool`
//! - `pub fn measured(s: *Stream, gpa: Allocator) Allocator.Error![]core.Measured`
//! - `pub fn deviceScans(s: *const Stream, gpa: Allocator) Allocator.Error!?DeviceScans`
//! - `pub fn adoptScans(s: *Stream, scans: []const scan.Dev) void`
//! - `pub fn wave(s: Stream) Wave`, `pub fn close(s: *Stream, i: usize) void`

const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;
const Writer = std.Io.Writer;
const core = @import("core");
const scan = @import("scan.zig");
const columns = @import("column.zig");
const card = @import("card.zig");
const rule = @import("rule.zig");
const device = @import("device_scans.zig");
const sci = @import("sci.zig");
const Wave = @import("Wave.zig");
const Scan = @import("Scan.zig");
const Clause = core.MeasureClause;
const Stream = @import("Stream.zig");

/// Owns every slice below except `measures`.
arena: std.heap.ArenaAllocator,
/// Borrowed; must outlive the stream.
measures: []const core.Measure,
/// One per card of `measures`.
cards: []Card,
/// Each resolved card's value, NaN until then: what a later card reads.
values: []f64,
/// `.tran` cards not resolved yet, those that never resolve included.
open: usize,
/// The cards `feed` visits, in deck order: open cards that can resolve,
/// non-PARAM ones first (`live[0]`), then PARAM ones (`live[1]`).
live: [2]std.ArrayList(u32),
/// The columns the `.tran` cards read, the scale first, and the
/// unknown of the solution behind each after the scale.
names: []const []const u8,
nodes: []u32,
/// Those columns at the current row, which every scan reads, and the
/// current row kept while a replay borrows `row`.
row: []f64,
saved: []f64,
/// The column of `row` behind each recorded column, `none` for one no
/// card reads.
recorded: []u32,
/// Rows fed so far.
rows: usize = 0,
/// The card an abort bound waits on (`unresolved`); null for all.
abort_card: ?usize = null,

/// One card's measurement as the run goes.
pub const Card = struct {
    rule: rule.Rule,
    done: bool,
    started: bool = false,
    /// Its passes, in the order `card.evaluate` reads them; `n` are built.
    scans: [2]Scan = undefined,
    n: u8 = 0,
    /// Scan 1 starts at scan 0's event, from `clause`: TARG with
    /// TD=TRIG, or the value FIND/DERIV reads at its WHEN.
    stage: enum { none, targ, find } = .none,
    clause: Clause = .{},
    /// DERIV's slope rather than FIND's value at the event.
    slope: bool = false,
    /// A `.scan` FIND/DERIV WHEN: its FIND vector, and (scale, value)
    /// at the last rows, oldest first, which the value pass starts
    /// from (in a run whose times rise, it reads no row before them).
    ring_y: ?columns.Waveform = null,
    ring: [3][2]f64 = undefined,
    ring_len: u8 = 0,

    /// Whether no later row changes what its passes found.
    pub fn resolved(c: Card) bool {
        assert(c.n <= c.scans.len);
        if (!c.started or c.stage != .none) return false;
        for (c.scans[0..c.n]) |s| if (!s.over) return false;
        return true;
    }
};

/// The open cards' passes as device passes and what each reads, in card
/// then pass order. Caller owns both slices.
pub const DeviceScans = struct { scans: []scan.Dev, cols: []scan.DevColumns };

/// The `recorded` entry of a column no card reads.
pub const none = std.math.maxInt(u32);

const nan = std.math.nan(f64);

/// Returns a stream over a transient's columns `names` and `nodes` (the
/// scale first, then each probe's unknown); `recorded` names the columns
/// the run records. Borrows `measures` and the name strings, which must
/// outlive it; free with `deinit`. Asserts one node per name after the
/// scale and fewer than 2^32 cards.
pub fn init(gpa: Allocator, measures: []const core.Measure, names: []const []const u8, nodes: []const u32, recorded: []const []const u8) Allocator.Error!Stream {
    assert(names.len == nodes.len + 1);
    assert(measures.len < none);
    var arena: std.heap.ArenaAllocator = .init(gpa);
    errdefer arena.deinit();
    const a = arena.allocator();
    const keep = try a.alloc(bool, names.len);
    rule.markColumns(measures, names, keep, false);
    const width = std.mem.count(bool, keep, &.{true});
    const own = try a.alloc([]const u8, width);
    const own_nodes = try a.alloc(u32, width - 1);
    var k: usize = 0;
    for (names, keep, 0..) |name, on, i| if (on) {
        own[k] = name;
        if (k > 0) own_nodes[k - 1] = nodes[i - 1];
        k += 1;
    };
    const map = try a.alloc(u32, recorded.len);
    for (recorded, map) |name, *m| m.* = for (own, 0..) |o, j| {
        if (std.mem.eql(u8, o, name)) break @intCast(j);
    } else none;
    var s: Stream = .{
        .arena = undefined,
        .measures = measures,
        .cards = try a.alloc(Card, measures.len),
        .values = try a.alloc(f64, measures.len),
        .open = 0,
        .live = .{ try .initCapacity(a, measures.len), try .initCapacity(a, measures.len) },
        .names = own,
        .nodes = own_nodes,
        .row = try a.alloc(f64, width),
        .saved = try a.alloc(f64, width),
        .recorded = map,
    };
    @memset(s.values, nan);
    @memset(s.row, nan);
    for (measures, s.cards, 0..) |m, *c, i| {
        c.* = .{ .rule = rule.ruleOf(measures, i), .done = m.analysis != .tran };
        if (c.done) continue;
        s.open += 1;
        if (c.rule != .end) s.live[@intFromBool(c.rule == .param)].appendAssumeCapacity(@intCast(i));
        // Read from the first row, so a run with none measures as an
        // empty recording does.
        if (c.rule == .scan) s.start(m, c, &.{});
    }
    s.arena = arena;
    return s;
}

/// Frees everything `init` allocated; slices from `measured` and
/// `deviceScans` are the caller's and survive it.
pub fn deinit(s: *Stream) void {
    s.arena.deinit();
    s.* = undefined;
}

/// Feeds the row at `t` of solution `x`. `data` is the recording, this row
/// last, which a card starting late replays. Call once per row. Asserts
/// that `data` holds one recorded row per row fed, this one included.
pub fn feed(s: *Stream, t: f64, x: []const f64, data: []const f64) void {
    assert(data.len == (s.rows + 1) * s.recorded.len);
    defer s.rows += 1;
    if (s.open == 0) return;
    s.row[0] = t;
    for (s.nodes, s.row[1..]) |node, *v| v.* = x[node];
    // Non-PARAM cards read only earlier ones, PARAM cards any card.
    for (&s.live) |*list| {
        var kept: usize = 0;
        for (list.items) |k| {
            const i: usize = k;
            const c = &s.cards[i];
            const m = s.measures[i];
            assert(!c.done);
            if (c.started) {
                for (c.scans[0..c.n]) |*sc| sc.feed(s.row[0]);
                s.next(c, data);
            } else if (s.ready(i)) s.start(m, c, data);
            if (c.started and c.resolved()) {
                s.close(i);
                continue;
            }
            list.items[kept] = k;
            kept += 1;
        }
        list.shrinkRetainingCapacity(kept);
    }
}

/// Whether the card an abort bound waits on (`abort_card`), or any
/// `.tran` card without one, has no final value yet.
pub fn unresolved(s: Stream) bool {
    return if (s.abort_card) |k| !s.cards[k].done else s.open > 0;
}

/// Returns each card's passes for `core.Result.measured`: what a `.scan`
/// card found, nothing for the others. Caller owns the slice.
pub fn measured(s: *Stream, gpa: Allocator) Allocator.Error![]core.Measured {
    const out = try gpa.alloc(core.Measured, s.cards.len);
    for (s.cards, out) |*c, *o| {
        o.* = .{};
        if (c.rule != .scan or !c.started) continue;
        o.* = .{ .scans = 2, .found = @splat(.{ .v = @splat(nan) }) };
        for (c.scans[0..c.n], o.found[0..c.n]) |*sc, *f| f.* = sc.found();
    }
    return out;
}

/// Returns every pass of the open `.tran` cards as a device pass and what
/// it reads, in card then pass order, so a GPU lane can measure without
/// the host seeing a row (performance TODO 06 6.8); a FIND/DERIV ... WHEN
/// card still at its WHEN adds a `.find` pass after it. Null when a card
/// needs more than one row at a time: one that is not `.scan`, an
/// ERR/EM_AVG pass, or a `par()` waveform past `scan.Program.max` ops.
pub fn deviceScans(s: *const Stream, gpa: Allocator) Allocator.Error!?DeviceScans {
    return device.handOut(s, gpa);
}

/// Takes back the passes `deviceScans` handed out, as the device left
/// them, and resolves each card they finish, as `feed` would have at the
/// row that finished it. Asserts that `scans` is what `deviceScans` gave.
pub fn adoptScans(s: *Stream, scans: []const scan.Dev) void {
    device.takeBack(s, scans);
    for (&s.live) |*list| {
        var kept: usize = 0;
        for (list.items) |k| if (!s.cards[k].done) {
            list.items[kept] = k;
            kept += 1;
        };
        list.shrinkRetainingCapacity(kept);
    }
}

/// Returns the current row as a one-row result, which the scans' columns read.
pub fn wave(s: Stream) Wave {
    return .{ .result = .{ .plotname = "", .varnames = s.names, .is_complex = false, .npoints = 1, .data = s.row }, .analysis = .tran };
}

/// Marks resolved card `i` done and records its value, as `card.evaluate`
/// reads it over its passes. Asserts that it is open and resolved.
pub fn close(s: *Stream, i: usize) void {
    const c = &s.cards[i];
    assert(!c.done);
    assert(c.resolved());
    c.done = true;
    s.values[i] = s.value(s.measures[i], c);
    s.open -= 1;
}

// ---- private ----

/// Whether every card card `i` reads has resolved; a card of another
/// analysis reads NaN whatever this run does.
fn ready(s: Stream, i: usize) bool {
    const m = s.measures[i];
    for ([_]Clause{ m.first, m.second }) |c| for (c.refs) |ref| if (!s.resolvedOps(ref.expr)) return false;
    return s.resolvedOps(m.expr);
}

fn resolvedOps(s: Stream, ops: []const core.MeasureOp) bool {
    for (ops) |op| if (op == .measure and op.measure < s.cards.len and !s.cards[op.measure].done) return false;
    return true;
}

/// Builds card `c`'s passes, as `card.evaluate` makes them, and runs them
/// over the rows so far: the recording `data` for a card starting late.
fn start(s: *Stream, m: core.Measure, c: *Card, data: []const f64) void {
    assert(!c.started);
    c.started = true;
    if (c.rule == .param) return;
    // A ref without a value: `card.evaluate` fails it before any pass.
    const a = card.resolve(m.first, s.values) catch return;
    const b = card.resolve(m.second, s.values) catch return;
    const w = s.wave();
    const cross = struct {
        fn of(cl: Clause) Scan.Pass {
            return .{ .cross = .init(Wave.bounds(cl), .tran) };
        }
    }.of;
    switch (m.func) {
        .when => c.scans[0] = .init(w, a, cross(a), .waveform),
        .trig_targ => {
            if (a.at == core.measure_no_at) c.scans[0] = .init(w, a, cross(a), .waveform);
            c.n = @intFromBool(a.at == core.measure_no_at);
            if (b.at == core.measure_no_at) {
                var targ = b;
                if (b.td_trig and a.at == core.measure_no_at) {
                    c.stage = .targ;
                    c.clause = b;
                } else {
                    if (b.td_trig) targ.td = a.at;
                    c.scans[c.n] = .init(w, targ, cross(targ), .waveform);
                    c.n += 1;
                }
            }
            return s.catchUp(c, data);
        },
        .find, .deriv => if (a.at != core.measure_no_at) {
            c.scans[0] = .init(w, a, .{ .at = .{ .at = a.at, .slope = m.func == .deriv } }, .none);
        } else {
            c.scans[0] = .init(w, b, cross(b), .waveform);
            c.stage = .find;
            c.clause = a;
            c.slope = m.func == .deriv;
            if (c.rule == .scan) c.ring_y = w.waveform(a.vec, a.ops, a.vectype) catch null;
        },
        .avg, .min, .max, .min_at, .max_at => c.scans[0] = .init(w, a, .{
            .extremum = .{
                .c = Wave.bounds(a),
                .dc = false,
                .op = switch (m.func) {
                    .avg => .avg,
                    .min, .min_at => .min,
                    else => .max, // else: MAX and MAX_AT
                },
            },
        }, .none),
        .pp => {
            c.scans = .{
                .init(w, a, .{ .extremum = .{ .c = Wave.bounds(a), .dc = false, .op = .min } }, .none),
                .init(w, a, .{ .extremum = .{ .c = Wave.bounds(a), .dc = false, .op = .max } }, .none),
            };
            c.n = 2;
            return s.catchUp(c, data);
        },
        .rms, .integ => c.scans[0] = .init(w, a, .{ .integ = .{ .c = Wave.bounds(a), .rms = m.func == .rms } }, .none),
        .em_avg => c.scans[0] = .init(w, a, .{ .em_avg = .{ .c = a } }, .none),
        .err, .err1, .err2, .err3 => c.scans[0] = .init(w, a, .{ .rel_error = .{ .c = a, .dc = false, .func = m.func } }, .column),
        else => unreachable, // else: `ruleOf` gives every other function `.end`
    }
    c.n = 1;
    s.catchUp(c, data);
}

/// Runs card `c`'s new passes over the rows so far: none before the
/// first row, else the recording `data`, this row included.
fn catchUp(s: *Stream, c: *Card, data: []const f64) void {
    if (data.len == 0) return;
    assert(c.rule == .replay);
    for (c.scans[0..c.n]) |*sc| s.replay(sc, data);
    s.next(c, data);
}

/// After card `c` has read the current row: keeps a `.scan` FIND's
/// ring, and starts scan 1 at scan 0's event.
fn next(s: *Stream, c: *Card, data: []const f64) void {
    if (c.stage == .none) return;
    if (c.ring_y) |y| scan.Find.push(&c.ring, &c.ring_len, .{ s.row[0], y.get(0) });
    if (!c.scans[0].over) return;
    const event = c.scans[0].found();
    const stage = c.stage;
    c.stage = .none;
    // No event: `card.evaluate` fails the card before scan 1.
    if (event.missing or std.math.isNan(event.v[0])) return;
    const w = s.wave();
    var cl = c.clause;
    c.n = 2;
    if (stage == .targ) {
        cl.td = event.v[0];
        c.scans[1] = .init(w, cl, .{ .cross = .init(Wave.bounds(cl), .tran) }, .waveform);
    } else c.scans[1] = .init(w, cl, .{ .at = .{ .at = event.v[0], .slope = c.slope } }, .none);
    if (c.rule == .replay) return s.replay(&c.scans[1], data);
    // `.scan`: from the ring. Rows before it hold the event only where
    // times repeat, which a transient's accepted rows never do.
    const sc = &c.scans[1];
    if (sc.missing) return;
    if (s.rows + 1 > c.ring_len and event.v[0] <= c.ring[0][0]) {
        sc.over = true; // ponytail: unreachable while recorded times rise
        return;
    }
    for (c.ring[0..c.ring_len]) |r| if (!sc.over) sc.feedValues(r[0], r[1], nan);
}

/// Runs `sc` over the recording `data`, every row so far.
fn replay(s: *Stream, sc: *Scan, data: []const f64) void {
    const width = s.recorded.len;
    assert(data.len == (s.rows + 1) * width);
    @memcpy(s.saved, s.row);
    defer @memcpy(s.row, s.saved);
    for (0..s.rows + 1) |i| {
        for (s.recorded, data[i * width ..][0..width]) |k, v| if (k != none) {
            s.row[k] = v;
        };
        sc.feed(s.row[0]);
    }
}

/// Resolved card `c`'s value, as `card.evaluate` reads it over its passes.
fn value(s: Stream, m: core.Measure, c: *Card) f64 {
    var found: [2]core.Measured.Found = @splat(.{ .v = @splat(nan) });
    for (c.scans[0..c.n], found[0..c.n]) |*sc, *f| f.* = sc.found();
    var done: Wave.Done = .{ .found = &found };
    var discard_buf: [64]u8 = undefined;
    var discard = sci.quiet(&discard_buf);
    const empty: core.Result = .{ .plotname = "", .varnames = &.{}, .is_complex = false, .npoints = 0, .data = &.{} };
    return card.evaluate(&discard.writer, m, .{ .result = empty, .analysis = .tran, .values = s.values, .done = &done }) catch nan;
}

/// Feeds `cards` a recording a row at a time through a `Stream`, keeping
/// only the columns `markColumns` keeps for it, and checks that the values
/// and the printout equal those over the whole recording; with `early`,
/// stopping once every card has resolved. Returns the rows fed.
fn expectStreamed(cards: []const core.Measure, early: bool) !usize {
    const gpa = std.testing.allocator;
    // A two-tone wave and a slow cosine on uneven steps, as a transient
    // would record them.
    const rows = 400;
    var data: [3 * rows]f64 = undefined;
    var t: f64 = 0;
    for (0..rows) |i| {
        data[3 * i ..][0..3].* = .{ t, @sin(t) + 0.5 * @sin(3.1 * t), 0.3 * @cos(0.7 * t) };
        t += 0.05 + 0.04 * @abs(@sin(11 * t));
    }
    const names: []const []const u8 = &.{ "time", "v(a)", "v(b)" };
    const full: core.Result = .{ .plotname = "t", .varnames = names, .is_complex = false, .npoints = rows, .data = &data };
    const want = try gpa.alloc(f64, cards.len);
    defer gpa.free(want);
    card.evaluateAll(cards, .tran, full, want);
    var want_text: Writer.Allocating = .init(gpa);
    defer want_text.deinit();
    try @import("listing.zig").print(&want_text.writer, &want_text.writer, cards, .tran, full, null);
    var keep: [3]bool = undefined;
    rule.markColumns(cards, names, &keep, true);
    var recorded: std.ArrayList([]const u8) = .empty;
    defer recorded.deinit(gpa);
    for (names, keep) |n, on| if (on) try recorded.append(gpa, n);
    var s: Stream = try .init(gpa, cards, names, &.{ 0, 1 }, recorded.items);
    defer s.deinit();
    var rec: std.ArrayList(f64) = .empty;
    defer rec.deinit(gpa);
    var n: usize = 0;
    while (n < rows and !(early and s.open == 0)) : (n += 1) {
        const row = data[3 * n ..][0..3];
        for (keep, row) |on, v| if (on) try rec.append(gpa, v);
        s.feed(row[0], row[1..], rec.items);
    }
    const found = try s.measured(gpa);
    defer gpa.free(found);
    const streamed: core.Result = .{ .plotname = "t", .varnames = recorded.items, .is_complex = false, .npoints = n, .data = rec.items, .measured = found };
    const got = try gpa.alloc(f64, cards.len);
    defer gpa.free(got);
    card.evaluateAll(cards, .tran, streamed, got);
    var got_text: Writer.Allocating = .init(gpa);
    defer got_text.deinit();
    try @import("listing.zig").print(&got_text.writer, &got_text.writer, cards, .tran, streamed, null);
    errdefer std.debug.print("early {}: {d} rows, columns {any}\n{s}\n", .{ early, n, keep, got_text.written() });
    try std.testing.expectEqualSlices(u64, @ptrCast(want), @ptrCast(got));
    try std.testing.expectEqualStrings(want_text.written(), got_text.written());
    return n;
}

test "Stream's device passes, fed outside it, resolve and measure as feeding it does, to the bit" {
    const gpa = std.testing.allocator;
    const last = core.measure_last;
    const sum: []const core.MeasureOp = &.{ .{ .vector = "v(a)" }, .{ .vector = "v(b)" }, .add };
    const scaled: []const core.MeasureOp = &.{ .{ .vector = "v(b)" }, .{ .vector = "v(b)" }, .mul, .neg, .{ .num = 0.5 }, .mul };
    const slope_t: []const core.MeasureOp = &.{ .{ .vector = "time" }, .{ .num = 0.1 }, .mul, .{ .vector = "v(a)" }, .sub, .{ .measure = 0 }, .{ .num = 3 }, .div, .add };
    const cards = [_]core.Measure{
        .{ .analysis = .tran, .name = "w", .func = .when, .first = .{ .vec = "v(a)", .val = 0.2, .rise = 2 } },
        .{ .analysis = .tran, .name = "wv", .func = .when, .first = .{ .vec = "v(a)", .vec2 = "v(b)", .cross = 3, .td = 1 } },
        .{ .analysis = .tran, .name = "wl", .func = .when, .first = .{ .vec = "v(a)", .val = 0.2, .fall = last, .to = 15 } },
        .{ .analysis = .tran, .name = "wn", .func = .when, .first = .{ .vec = "v(a)", .val = 9, .to = 20 } },
        .{ .analysis = .tran, .name = "tt", .func = .trig_targ, .first = .{ .vec = "v(a)", .val = 0.2, .rise = 1 }, .second = .{ .vec = "v(b)", .val = 0, .fall = 1 } },
        .{ .analysis = .tran, .name = "fa", .func = .find, .first = .{ .vec = "v(b)", .at = 4.321 } },
        .{ .analysis = .tran, .name = "da", .func = .deriv, .first = .{ .vec = "v(a)", .at = 7.5 } },
        .{ .analysis = .tran, .name = "mx", .func = .max, .first = .{ .vec = "v(a)", .from = 1, .to = 6 } },
        .{ .analysis = .tran, .name = "mn", .func = .min_at, .first = .{ .vec = "v(a)", .from = 1, .to = 6 } },
        .{ .analysis = .tran, .name = "av", .func = .avg, .first = .{ .vec = "v(b)", .from = 2, .to = 9 } },
        .{ .analysis = .tran, .name = "pp", .func = .pp, .first = .{ .vec = "v(b)", .to = 12 } },
        .{ .analysis = .tran, .name = "ig", .func = .integ, .first = .{ .vec = "v(a)", .from = 0.7, .to = 8.3 } },
        .{ .analysis = .tran, .name = "rm", .func = .rms, .first = .{ .vec = "v(a)", .to = 21 } },
        // Stages and `par()` waveforms (the scale, an earlier card's NaN).
        .{ .analysis = .tran, .name = "fw", .func = .find, .first = .{ .vec = "v(b)" }, .second = .{ .vec = "v(a)", .val = -0.3, .fall = 2 } },
        .{ .analysis = .tran, .name = "dw", .func = .deriv, .first = .{ .ops = sum }, .second = .{ .vec = "v(a)", .val = 0.1, .cross = 4 } },
        .{ .analysis = .tran, .name = "fp", .func = .find, .first = .{ .vec = "v(b)" }, .second = .{ .vec = "v(a)", .val = 0.05, .rise = 1 } },
        .{ .analysis = .tran, .name = "fn", .func = .find, .first = .{ .vec = "v(b)" }, .second = .{ .vec = "v(a)", .val = 9, .rise = 1, .to = 5 } },
        .{ .analysis = .tran, .name = "pa", .func = .avg, .first = .{ .ops = sum, .to = 9 } },
        .{ .analysis = .tran, .name = "pw", .func = .when, .first = .{ .ops = sum, .ops2 = scaled, .vec2 = "v(b)", .cross = 2 } },
        .{ .analysis = .tran, .name = "pt", .func = .max, .first = .{ .ops = slope_t, .to = 14 } },
    };
    const names: []const []const u8 = &.{ "time", "v(a)", "v(b)" };
    const rows = 400;
    for ([_]bool{ false, true }) |early| {
        var host: Stream = try .init(gpa, &cards, names, &.{ 0, 1 }, &.{"time"});
        defer host.deinit();
        var dev: Stream = try .init(gpa, &cards, names, &.{ 0, 1 }, &.{"time"});
        defer dev.deinit();
        const got = (try dev.deviceScans(gpa)).?;
        defer gpa.free(got.scans);
        defer gpa.free(got.cols);
        var t: f64 = 0;
        var rec: std.ArrayList(f64) = .empty;
        defer rec.deinit(gpa);
        var stop_host: ?usize = null;
        var stop_dev: ?usize = null;
        for (0..rows) |i| {
            const x = [_]f64{ @sin(t) + 0.5 * @sin(3.1 * t), 0.3 * @cos(0.7 * t) };
            if (stop_host == null) {
                try rec.append(gpa, t);
                host.feed(t, &x, rec.items);
                if (early and host.open == 0) stop_host = i;
            }
            if (stop_dev == null) {
                var all = true;
                const at = struct {
                    v: *const [2]f64,
                    pub fn at(c: @This(), u: u32) f64 {
                        return c.v[u];
                    }
                }{ .v = &x };
                for (got.scans, got.cols, 0..) |*d, c, k| {
                    const y = c.read(t, at);
                    d.feed(&got.scans[k -| 1], t, y[0], y[1]);
                    all = all and d.over;
                }
                if (early and all) stop_dev = i;
            }
            t += 0.05 + 0.04 * @abs(@sin(11 * t));
        }
        try std.testing.expectEqual(stop_host, stop_dev);
        try std.testing.expect(!early or stop_dev != null);
        dev.adoptScans(got.scans);
        try std.testing.expectEqual(host.open, dev.open);
        try std.testing.expectEqualSlices(u64, @ptrCast(host.values), @ptrCast(dev.values));
        const mh = try host.measured(gpa);
        defer gpa.free(mh);
        const md = try dev.measured(gpa);
        defer gpa.free(md);
        try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(mh), std.mem.sliceAsBytes(md));
    }
    // A card that reads two rows at a time, or a power, keeps the host's rows.
    const pow: []const core.MeasureOp = &.{ .{ .vector = "v(a)" }, .{ .num = 2 }, .pow };
    const host_only = [_]core.Measure{
        .{ .analysis = .tran, .name = "e1", .func = .err1, .first = .{ .vec = "v(a)", .vec2 = "v(b)", .to = 9 } },
        .{ .analysis = .tran, .name = "pw", .func = .max, .first = .{ .ops = pow, .to = 9 } },
    };
    for (0..host_only.len) |k| {
        var s: Stream = try .init(gpa, host_only[k..][0..1], names, &.{ 0, 1 }, &.{"time"});
        defer s.deinit();
        try std.testing.expect(try s.deviceScans(gpa) == null);
    }
}

test "Stream measures each card as the whole recording does, to the bit, early stop too" {
    const last = core.measure_last;
    const sum: []const core.MeasureOp = &.{ .{ .vector = "v(a)" }, .{ .vector = "v(b)" }, .add };
    const streamed = [_]core.Measure{
        .{ .analysis = .tran, .name = "w", .func = .when, .first = .{ .vec = "v(a)", .val = 0.2, .rise = 2 } },
        .{ .analysis = .tran, .name = "wv", .func = .when, .first = .{ .vec = "v(a)", .vec2 = "v(b)", .cross = 3, .td = 1 } },
        .{ .analysis = .tran, .name = "wl", .func = .when, .first = .{ .vec = "v(a)", .val = 0.2, .fall = last, .to = 15 } },
        .{ .analysis = .tran, .name = "wn", .func = .when, .first = .{ .vec = "v(a)", .val = 9, .to = 20 } },
        .{ .analysis = .tran, .name = "w0", .func = .when, .first = .{ .vec = "v(a)", .val = 0.5 } },
        .{ .analysis = .tran, .name = "tt", .func = .trig_targ, .first = .{ .vec = "v(a)", .val = 0.2, .rise = 1 }, .second = .{ .vec = "v(b)", .val = 0, .fall = 1 } },
        .{ .analysis = .tran, .name = "ta", .func = .trig_targ, .first = .{ .at = 2 }, .second = .{ .vec = "v(a)", .val = 0, .cross = 2, .td_trig = true } },
        .{ .analysis = .tran, .name = "fw", .func = .find, .first = .{ .vec = "v(b)" }, .second = .{ .vec = "v(a)", .val = -0.3, .fall = 2 } },
        .{ .analysis = .tran, .name = "dw", .func = .deriv, .first = .{ .ops = sum }, .second = .{ .vec = "v(a)", .val = 0.1, .cross = 4 } },
        .{ .analysis = .tran, .name = "fp", .func = .find, .first = .{ .vec = "v(b)" }, .second = .{ .vec = "v(a)", .val = 0.05, .rise = 1 } },
        .{ .analysis = .tran, .name = "fa", .func = .find, .first = .{ .vec = "v(b)", .at = 4.321 } },
        .{ .analysis = .tran, .name = "da", .func = .deriv, .first = .{ .vec = "v(a)", .at = 7.5 } },
        .{ .analysis = .tran, .name = "mx", .func = .max, .first = .{ .vec = "v(a)", .from = 1, .to = 6 } },
        .{ .analysis = .tran, .name = "mn", .func = .min_at, .first = .{ .vec = "v(a)", .from = 1, .to = 6 } },
        .{ .analysis = .tran, .name = "av", .func = .avg, .first = .{ .ops = sum, .from = 2, .to = 9 } },
        .{ .analysis = .tran, .name = "pp", .func = .pp, .first = .{ .vec = "v(b)", .to = 12 } },
        .{ .analysis = .tran, .name = "ig", .func = .integ, .first = .{ .vec = "v(a)", .from = 0.7, .to = 8.3 } },
        .{ .analysis = .tran, .name = "rm", .func = .rms, .first = .{ .vec = "v(a)", .to = 21 } },
        .{ .analysis = .tran, .name = "em", .func = .em_avg, .first = .{ .vec = "v(a)", .to = 10 } },
        .{ .analysis = .tran, .name = "e1", .func = .err1, .first = .{ .vec = "v(a)", .vec2 = "v(b)", .to = 9 } },
        .{ .analysis = .tran, .name = "nv", .func = .max, .first = .{ .vec = "v(z)" } },
        .{ .analysis = .tran, .name = "pr", .func = .param, .first = .{}, .expr = &.{ .{ .measure = 0 }, .{ .measure = 12 }, .add } },
    };
    // Every card resolves by t = 21 of 30, keeping no column but time.
    const cut = try expectStreamed(&streamed, true);
    try std.testing.expect(cut < 300);
    _ = try expectStreamed(&streamed, false);
    const t0: []const core.MeasureOp = &.{.{ .measure = 0 }};
    const recorded = streamed ++ [_]core.Measure{
        .{ .analysis = .tran, .name = "td", .func = .trig_targ, .first = .{ .vec = "v(b)", .val = 0, .fall = 1 }, .second = .{ .vec = "v(a)", .val = 0, .rise = 1, .td_trig = true } },
        .{ .analysis = .tran, .name = "fd", .func = .find, .first = .{ .vec = "v(b)" }, .second = .{ .vec = "v(a)", .val = 0.1, .cross = 1, .td = 3 } },
        .{ .analysis = .tran, .name = "fr", .func = .find, .first = .{ .vec = "v(b)", .refs = &.{.{ .field = .at, .expr = t0 }} } },
        .{ .analysis = .tran, .name = "wr", .func = .when, .first = .{ .vec = "v(a)", .val = 0, .refs = &.{.{ .field = .td, .expr = t0 }} } },
        // Never resolve: a LAST count, a window and JITTER without TO=.
        .{ .analysis = .tran, .name = "fl", .func = .find, .first = .{ .vec = "v(b)" }, .second = .{ .vec = "v(a)", .val = 0.1, .rise = last } },
        .{ .analysis = .tran, .name = "rn", .func = .rms, .first = .{ .vec = "v(a)" } },
        .{ .analysis = .tran, .name = "ji", .func = .jitter, .first = .{ .vec = "v(a)", .val = 0 } },
    };
    for ([_]bool{ false, true }) |early| _ = try expectStreamed(&recorded, early);
    // The replayed cards resolve too.
    try std.testing.expect(try expectStreamed(recorded[0 .. recorded.len - 3], true) < 300);
}
