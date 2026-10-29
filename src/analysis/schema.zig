//! The result shape of a query over a circuit and deck, without running it:
//! its column count, whether it streams, and its plot and scale names.
//! In: a query, the template and the deck. Out: `core.QuerySchema`.
//! Cites #83, #154.
//!
//! Contents:
//! - `pub fn schemaOf(allocator: Allocator, topology: *const Circuit, deck: *const Deck, query: requests.Query) !core.QuerySchema`

const std = @import("std");
const core = @import("core");
const requests = core.query;
const Deck = core.Deck;
const Circuit = @import("device").Circuit;
const ParamRef = @import("device").abi.ParamRef;

/// Returns the shape `query` publishes over this circuit and deck, without
/// running it.
pub fn schemaOf(allocator: std.mem.Allocator, topology: *const Circuit, deck: *const Deck, query: requests.Query) !core.QuerySchema {
    const columns: usize = switch (query) {
        .op, .dcinc => deck.probes.len,
        // HSPICE `.op <time>` publishes the `.op` layout, with no time column.
        .tran => |o| deck.probes.len + @intFromBool(!o.snapshot),
        .tran_noise => |o| deck.probes.len + 1 + @intFromBool(o.sde),
        .tf => 3,
        .lstb => |o| if (o.margins) 5 else 8,
        .dcxf => |o| (if (o.tf_only) @as(usize, 1) else 3) * o.sources.len,
        .acxf => |o| 1 + (if (o.tf_only) @as(usize, 1) else 3) * o.sources.len,
        .noise => |o| try @import("ac/noise.zig").columnCount(allocator, topology.batches, topology.batch_types, o),
        // Time, then peak and rms per probe.
        .envelope => 1 + 2 * deck.probes.len,
        .four => 4,
        .disto => |o| if (o.plot == .summary) 4 else deck.probes.len + 1,
        .pz, .stb, .pnoise, .hbnoise, .phasenoise, .fft => 2,
        .pac => |o| 2 + 2 * @as(usize, o.n_harmonics),
        .hbac => |o| 1 + try hbSidebands(allocator, o),
        .pxf => |o| 1 + (1 + 2 * @as(usize, o.n_harmonics)) * topology.n,
        .hbxf => |o| 1 + try hbSidebands(allocator, o) * topology.n,
        .hblin => |o| 1 + o.ports.len * o.ports.len + @intFromBool(o.noise and o.ports.len >= 2),
        .sp => |o| @import("ac/sp.zig").columns(@import("ac/sp.zig").modeCount(o.ports), o.lin),
        // The parameters `.sens` lists, which isSensParam narrows (#154).
        .sens => try countParams(allocator, topology, @import("sweep/sens.zig").isSensParam),
        .dcmatch, .dcsens, .acmatch => blk: {
            const groups = switch (query) {
                inline .dcmatch, .dcsens, .acmatch => |o| o.variations.starts.len -| 1,
                else => 0,
            };
            const extra: usize = switch (query) {
                .dcmatch => 1,
                .acmatch => 5,
                else => 0,
            };
            if (groups != 0) break :blk groups + extra;
            break :blk try countParams(allocator, topology, null) + extra;
        },
        else => deck.probes.len + 1,
    };
    const tran = @import("tran/tran.zig");
    const ac = @import("ac/ac.zig");
    const streamable = switch (query) {
        .tran => |o| !o.snapshot,
        .ac => true,
        else => false, // else: only a transient or AC sweep streams its rows
    };
    return .{
        .kind = query,
        .columns = columns,
        .portless = query == .sp and query.sp.ports.len == 0 and deck.source_branch == 0,
        .plotname = if (!streamable) "" else if (query == .tran) tran.plotname else ac.plotname,
        .scale = if (!streamable) "" else if (query == .tran) tran.scale else ac.scale,
    };
}

// ---- private ----

/// Sidebands an HB small-signal analysis reads: 2M + 1 about one tone,
/// every signed line of a multi-tone spectrum.
fn hbSidebands(allocator: std.mem.Allocator, o: requests.HbLptv) !usize {
    if (o.extra_tones.len == 0) return 2 * @as(usize, o.n_sidebands) + 1;
    return 2 * try @import("pss/mhb.zig").lineCount(allocator, o.hb()) - 1;
}

/// How many of `topology`'s device parameters `keep(type_name, param_name)`
/// admits, all of them for null. A batch with a `param_layout` is counted
/// from it, which leaves the template's Model rows shared.
fn countParams(allocator: std.mem.Allocator, topology: *const Circuit, comptime keep: ?fn ([]const u8, []const u8) bool) !usize {
    var refs: std.ArrayList(ParamRef) = .empty;
    defer refs.deinit(allocator);
    var n: usize = 0;
    for (topology.batches) |b| {
        if (b.hooks.param_layout) |layout| {
            const l = try layout(b.ctx).unwrap();
            for (l.fields) |f| n += l.count * @intFromBool(if (keep) |k| k(b.type_name, f.name) else true);
            continue;
        }
        refs.clearRetainingCapacity();
        try b.hooks.collect_params(b.ctx, allocator, &refs).unwrap();
        for (refs.items) |r| n += @intFromBool(if (keep) |k| k(b.type_name, r.param_name) else true);
    }
    return n;
}
