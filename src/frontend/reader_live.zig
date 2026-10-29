//! Live values: every device and model value that reads a swept `.param` or
//! a distribution, as slots a variant rewrites.
//! Cites: dev/analysis/variants.md; #133.
//!
//! Contents:
//! - `pub fn Methods(comptime S: type) type`
//! - `pub fn resolveLive(r: *R) Error!Live`
//! - `pub fn readsCircuit(ops: []const expr.Op) bool`
//! - `pub fn holdsSite(r: *const R, start: usize, end: usize) bool`
//! - `pub fn liveSlot(r: *R, v: *Value, at: Live.Slot, slots: *Slots, spans: *std.ArrayList(Span), pool_rows: []u32, sites: *Sites) Error!void`
//! - `pub fn siteRows(r: *R, span: Span, sites: *Sites) Error!Span`
//! - `pub fn liveConds(r: *R) Error![]const Span`
//! - `pub fn unsupported(r: *R, line: []const u8, what: []const u8) Error`

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
const card_table = @import("card_table.zig");
const reader_frame = @import("reader_frame.zig");
const none = card_rows.none;
const Error = card_rows.Error;
const Span = card_rows.Span;
const Value = card_rows.Value;
const Live = card_rows.Live;
const Frame = reader_frame.Frame;
const readsLive = card_table.readsLive;
const orderU32 = netlist_view.orderU32;

/// The stage's `Reader` methods over dialect syntax `S`; `Reader(S)`
/// aliases each one as its own decl.
pub fn Methods(comptime S: type) type {
    return struct {
        const R = reader.Reader(S);
        const Sites = R.Sites;
        const Slots = R.Slots;

        /// Collects the live values after the device walk: each table value
        /// kept as postfix because it read a live name or a distribution
        /// becomes a slot holding its nominal number. Other kept postfix
        /// (a B source's) reads each live name from a constant-pool row.
        pub fn resolveLive(r: *R) Error!Live {
            if (r.live_names.count() == 0 and r.site_ops.items.len == 0) return .{};
            var slots: Slots = .{};
            var spans: std.ArrayList(Span) = .empty;
            var out: Live = .{ .names = r.live_names.keys(), .nominal = r.live_nominal.items };
            const pool_rows = try r.arena.alloc(u32, r.live_names.count());
            @memset(pool_rows, none);
            var sites: Sites = .{};
            for (r.values.items, 0..) |*v, i| try r.liveSlot(v, .{ .table = .values, .row = @intCast(i) }, &slots, &spans, pool_rows, &sites);
            for (r.kvs.items, 0..) |*kv, i| try r.liveSlot(&kv.value, .{ .table = .kvs, .row = @intCast(i) }, &slots, &spans, pool_rows, &sites);
            out.slots = slots.at.items;
            out.groups = slots.groups.items;
            out.ops = spans.items;
            out.site_ops = r.site_ops.items;
            out.site_keys = r.site_keys.items;
            out.pool_rows = pool_rows;
            out.site_rows = sites.rows.items;
            out.site_spans = sites.spans.items;
            return out;
        }

        /// Whether postfix `ops` reads the circuit (a probe, `time` or
        /// `temper`): only a B-source tape does.
        pub fn readsCircuit(ops: []const expr.Op) bool {
            for (ops) |op| switch (op.code) {
                .vprobe, .iprobe => return true,
                .ident => if (op.a >= 2) return true,
                else => {}, // else: no other op reads the circuit
            };
            return false;
        }

        /// Whether one of the distribution calls the parse recorded sits in
        /// ops `start..end` of the netlist.
        pub fn holdsSite(r: *const R, start: usize, end: usize) bool {
            const k = std.sort.lowerBound(u32, r.site_ops.items, @as(u32, @intCast(start)), orderU32);
            return k < r.site_ops.items.len and r.site_ops.items[k] < end;
        }

        /// Records `v`, which sits at `at`, as a slot when it reads a live
        /// name or a distribution, and the arguments of a group the same way.
        pub fn liveSlot(r: *R, v: *Value, at: Live.Slot, slots: *Slots, spans: *std.ArrayList(Span), pool_rows: []u32, sites: *Sites) Error!void {
            switch (v.*) {
                // `valueAt` built the arguments in the parse arena, mutable;
                // `Group` types them const for every reader after the parse.
                // A group keeps its `Live.groups` row only if an argument
                // (or a nested group's) became a slot.
                .group => |g| {
                    if (slots.groups.items.len == none) return error.CircuitTooLarge;
                    const row: u32 = @intCast(slots.groups.items.len);
                    try slots.groups.append(r.arena, g);
                    const mark = slots.at.items.len;
                    for (g.args(), 0..) |*a, j| try r.liveSlot(@constCast(a), .{ .table = .groups, .row = row, .arg = @intCast(j) }, slots, spans, pool_rows, sites);
                    if (slots.at.items.len == mark) {
                        std.debug.assert(slots.groups.items.len == row + 1);
                        slots.groups.items.len = row;
                    }
                },
                .expr => |span| {
                    const sited = r.holdsSite(span.start, span.start + span.len);
                    if (!sited and !readsLive(r.ops.items[span.start..][0..span.len])) return;
                    const folded = try expr.fold(r.arena, &r.stack, r.ops.items[span.start..][0..span.len], r.consts.items, false, r.live_nominal.items);
                    if (folded.known) {
                        try slots.at.append(r.arena, at);
                        try spans.append(r.arena, span);
                        v.* = .{ .num = folded.num };
                        return;
                    }
                    // A B source's tape reads its draws from pool rows. A model
                    // value kept for its `l`/`w`/`mult` is read per instance
                    // as a slot, so its shared copy keeps its calls (#100).
                    const kept = if (sited and readsCircuit(r.ops.items[span.start..][0..span.len])) try r.siteRows(span, sites) else span;
                    v.* = .{ .expr = kept };
                    for (r.ops.items[kept.start..][0..kept.len]) |*op| if (op.code == .live) {
                        if (pool_rows[op.a] == none) {
                            try r.consts.append(r.arena, r.live_nominal.items[op.a]);
                            pool_rows[op.a] = @intCast(r.consts.items.len - 1);
                        }
                        op.* = .{ .code = .num, .a = pool_rows[op.a] };
                    };
                },
                else => {},
            }
        }

        /// A copy of the kept expression at `span` with each largest constant
        /// subtree holding a distribution call replaced by a pool row at its
        /// nominal value, which `setLive` redraws from the subtree left in
        /// place (#133). A subtree that also reads a probe stays as it is.
        pub fn siteRows(r: *R, span: Span, sites: *Sites) Error!Span {
            var rev: std.ArrayList(expr.Op) = .empty;
            var i: usize = span.len;
            while (i > 0) {
                i -= 1;
                const ops = r.ops.items[span.start..][0..span.len];
                const s = expr.subtreeStart(ops, i);
                const sub: Span = .{ .start = span.start + @as(u32, @intCast(s)), .len = @intCast(i + 1 - s) };
                const v = if (r.holdsSite(sub.start, sub.start + sub.len)) try expr.fold(r.arena, &r.stack, ops[s .. i + 1], r.consts.items, false, r.live_nominal.items) else expr.Val{ .known = false };
                if (!v.known) {
                    try rev.append(r.arena, ops[i]);
                    continue;
                }
                try r.consts.append(r.arena, v.num);
                const row: u32 = @intCast(r.consts.items.len - 1);
                try sites.rows.append(r.arena, row);
                try sites.spans.append(r.arena, sub);
                try rev.append(r.arena, .{ .code = .num, .a = row });
                i = s;
            }
            std.mem.reverse(expr.Op, rev.items);
            const start: u32 = @intCast(r.ops.items.len);
            try r.ops.appendSlice(r.arena, rev.items);
            return .{ .start = start, .len = @intCast(rev.items.len) };
        }

        /// Recompiles the top-level conditions walk 1 evaluated, now that the
        /// swept names are known, keeping those that read one.
        pub fn liveConds(r: *R) Error![]const Span {
            if (r.live_names.count() == 0) return &.{};
            const was_in_card = r.in_card;
            r.in_card = true;
            defer r.in_card = was_in_card;
            const top: Frame = .{ .scopes = &r.global_scopes };
            var out: std.ArrayList(Span) = .empty;
            for (r.cond_texts.items) |text| switch (try r.exprValue(text, &top, false)) {
                .expr => |span| try out.append(r.arena, span),
                else => {},
            };
            return out.items;
        }

        /// Logs `line` as `what` and fails the parse with `UnsupportedCard`.
        pub fn unsupported(r: *R, line: []const u8, what: []const u8) Error {
            r.logged = true;
            // The test runner fails any test that logs an error.
            if (!@import("builtin").is_test) std.log.err("netlist: {s}: {s}", .{ what, r.written(line) });
            return error.UnsupportedCard;
        }

        // Values.
    };
}
