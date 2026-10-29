//! The `.noise` contribution columns: one label per generator name and per
//! instance, and where each noise source lands in them. Generators of one
//! instance that share a name (LRM §4.6.4 `name`) share a column; each
//! instance's total column follows its generator columns.
//! In: the batches, their types, the deck's card names. Out: the labels and
//! two per-source column maps.
//! Cites Verilog-AMS LRM §4.6.4; ngspice's `.noise ... pts` naming; #83.
//!
//! Contents:
//! - `pub const Columns = struct` (`labels`, `gen`, `total`, `Error`, `build`, `sum`)
//! - `pub fn sourceCount(batches: []const Batch) usize`

const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;
const abi = @import("device").abi;
const root = @import("../types.zig");

/// The contribution columns and where each noise source lands in them.
pub const Columns = struct {
    /// `<inst>_<gen>` or `<inst>`, in column order.
    labels: []const []const u8 = &.{},
    /// Per source, in `collectNoiseSources` order: its generator column.
    gen: []const u32 = &.{},
    /// Per source: its instance's total column.
    total: []const u32 = &.{},

    /// Every way `build` fails. Everything is allocated in an arena the
    /// caller frees whole, so nothing is freed on error.
    pub const Error = Allocator.Error || error{
        /// The batches' generator counts do not add up to `n_sources`.
        NoiseLabelMismatch,
    };

    /// Labels the `n_sources` generators `collectNoiseSources` yields, batch
    /// by batch and instance by instance, from each batch's `noise_names`
    /// and the card names in `names`. Allocates from `arena`, which owns the
    /// result.
    pub fn build(arena: Allocator, batches: []const abi.Batch, types: []const abi.DeviceType, names: root.CardNames, n_sources: usize) Error!Columns {
        assert(types.len == batches.len);
        var labels: std.ArrayList([]const u8) = .empty;
        const map = try arena.alloc(u32, 2 * n_sources);
        const gen = map[0..n_sources];
        const total = map[n_sources..];
        var s: usize = 0;
        for (batches, types) |b, t| {
            if (b.hooks.collect_noise == null) continue;
            const gens = b.hooks.noise_names;
            if (s + b.count * gens.len > n_sources) return error.NoiseLabelMismatch;
            if (gens.len == 0) continue;
            for (0..b.count) |id| {
                const inst = names.get(t, @intCast(id)) orelse try arena.print("{s}#{d}", .{ b.type_name, id });
                const first = labels.items.len;
                for (gens, 0..) |name, k| {
                    const label = if (name.len == 0)
                        try arena.print("{s}_{d}", .{ inst, k })
                    else
                        try arena.print("{s}_{s}", .{ inst, name });
                    gen[s + k] = for (labels.items[first..], first..) |seen, c| {
                        if (std.mem.eql(u8, seen, label)) break @intCast(c);
                    } else blk: {
                        try labels.append(arena, label);
                        break :blk @intCast(labels.items.len - 1);
                    };
                }
                @memset(total[s..][0..gens.len], @intCast(labels.items.len));
                try labels.append(arena, inst);
                s += gens.len;
            }
        }
        if (s != n_sources) return error.NoiseLabelMismatch;
        return .{ .labels = labels.items, .gen = gen, .total = total };
    }

    /// Adds each per-source value into its generator and instance columns.
    /// Asserts one value per source.
    pub fn sum(self: Columns, per_source: []const f64, cols: []f64) void {
        assert(per_source.len == self.gen.len);
        for (per_source, self.gen, self.total) |v, g, t| {
            cols[g] += v;
            cols[t] += v;
        }
    }
};

/// Returns the sources `collectNoiseSources` yields over `batches`: one per
/// generator per instance.
pub fn sourceCount(batches: []const abi.Batch) usize {
    var n: usize = 0;
    for (batches) |b| {
        if (b.hooks.collect_noise != null) n += b.count * b.hooks.noise_names.len;
    }
    return n;
}
