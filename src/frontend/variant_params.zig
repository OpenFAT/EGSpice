//! A circuit's parameters by ordinal. In: a frozen circuit. Out: every
//! batch parameter in `Circuit.collectParams` order, addressed through
//! each batch's `param_layout` rather than one `ParamRef` per parameter.
//! Cites: none (house device ABI, device/abi.zig).
//!
//! Contents:
//! - `pub const Params`
//! - `pub const Ref`
//! - `pub fn get(r: Ref) f64`
//! - `pub fn at(ps: Params, ord: usize) Ref`
//! - `pub fn get(ps: Params, ord: usize) f64`
//! - `pub fn iterator(ps: *const Params) Iterator`
//! - `pub const Iterator`
//! - `pub fn next(it: *Iterator) ?*f64`
//! - `pub fn collect(gpa: std.mem.Allocator, c: *const device.Circuit) !Params`

const std = @import("std");
const assert = std.debug.assert;
const testing = std.testing;
const Allocator = std.mem.Allocator;
const core = @import("core");
const device = @import("device");
const netlist = @import("netlist");
const builder = @import("builder");
const analyses = @import("analyses.zig");
const expr = netlist.expr;
const Netlist = netlist.Netlist;
const ParamRef = device.abi.ParamRef;
const variant_circuit = @import("variant_circuit.zig");

/// Every parameter of a circuit's batches in `Circuit.collectParams` order,
/// addressed through each batch's `param_layout` instead of a `ParamRef`
/// per parameter: a sweep over 10k BSIM4 devices held 8.7M refs (555 MB)
/// per table, and the planner builds several. Batches without the hook
/// keep their `collect_params` refs. Dies with the circuit.
pub const Params = struct {
    /// The circuit's batch and type slices: heap storage, which stays put
    /// when the `Circuit` value itself moves (`prepare` returns it).
    batches: []const device.abi.Batch = &.{},
    types: []const device.abi.DeviceType = &.{},
    /// Batch b's parameters are ordinals `starts[b]..starts[b + 1]`.
    starts: []const u32 = &.{0},
    /// Batch b's `param_layout`, null for a batch without the hook.
    layouts: []const ?device.abi.ParamLayout = &.{},
    /// Batch b's `collect_params` refs, empty when it has a layout.
    fallback: []const []const ParamRef = &.{},
    /// Total parameter count, `starts[batches.len]`.
    len: usize = 0,

    /// One parameter: where it lives and which device field it is.
    pub const Ref = struct {
        ptr: *f64,
        type: core.DeviceType,
        param_name: []const u8,
        index: u32,
        is_instance: bool,
        primary: bool,

        /// The parameter's current value.
        pub fn get(r: Ref) f64 {
            return r.ptr.*;
        }
    };

    fn batchOf(ps: Params, ord: usize) usize {
        return std.sort.upperBound(u32, ps.starts, @as(u32, @intCast(ord)), orderU32) - 1;
    }

    fn orderU32(a: u32, b: u32) std.math.Order {
        return std.math.order(a, b);
    }

    /// Parameter `ord`. O(log batches). Asserts `ord < ps.len`.
    pub fn at(ps: Params, ord: usize) Ref {
        const b = ps.batchOf(ord);
        const local: u32 = @intCast(ord - ps.starts[b]);
        if (ps.layouts[b]) |l| {
            const field = l.fields[local / l.count];
            return .{
                .ptr = l.ptr(local),
                .type = ps.types[b],
                .param_name = field.name,
                .index = local % l.count,
                .is_instance = field.is_instance,
                .primary = field.primary,
            };
        }
        const r = ps.fallback[b][local];
        return .{ .ptr = r.ptr, .type = r.type, .param_name = r.param_name, .index = r.index, .is_instance = r.is_instance, .primary = r.primary };
    }

    /// Parameter `ord`'s current value; `at` without building the `Ref`.
    pub fn get(ps: Params, ord: usize) f64 {
        return ps.ptrOf(ord).*;
    }

    fn ptrOf(ps: Params, ord: usize) *f64 {
        const b = ps.batchOf(ord);
        return ps.ptrIn(b, @intCast(ord - ps.starts[b]));
    }

    /// Parameter `local` of batch `b`.
    fn ptrIn(ps: Params, b: usize, local: u32) *f64 {
        return if (ps.layouts[b]) |l| l.ptr(local) else ps.fallback[b][local].ptr;
    }

    /// Every parameter's address in order, without a search per call.
    pub fn iterator(ps: *const Params) Iterator {
        return .{ .ps = ps };
    }

    /// Walks the parameters in ordinal order; see `iterator`.
    pub const Iterator = struct {
        ps: *const Params,
        b: usize = 0,
        ord: usize = 0,

        /// The next parameter's address; null past the last.
        pub fn next(it: *Iterator) ?*f64 {
            const ps = it.ps;
            if (it.ord >= ps.len) return null;
            while (it.ord >= ps.starts[it.b + 1]) it.b += 1;
            const local: u32 = @intCast(it.ord - ps.starts[it.b]);
            it.ord += 1;
            return ps.ptrIn(it.b, local);
        }
    };

    /// `refKey(type, name)` to the ordinal of that field's index-0
    /// parameter, the first entry winning, as a scan of the refs gave.
    fn indexNames(ps: Params, a: std.mem.Allocator, map: *std.StringHashMapUnmanaged(u32)) !void {
        for (ps.layouts, ps.fallback, 0..) |layout, refs, b| {
            if (layout) |l| {
                if (l.count == 0) continue;
                for (l.fields, 0..) |f, k| {
                    const gop = try map.getOrPut(a, try variant_circuit.refKey(a, ps.types[b], f.name));
                    if (!gop.found_existing) gop.value_ptr.* = ps.starts[b] + @as(u32, @intCast(k)) * l.count;
                }
                continue;
            }
            for (refs, 0..) |r, i| {
                const gop = try map.getOrPut(a, try variant_circuit.refKey(a, r.type, r.param_name));
                if (!gop.found_existing) gop.value_ptr.* = @intCast(ps.starts[b] + i - r.index);
            }
        }
    }
};

/// Every parameter of `c`'s batches, typed, in `Circuit.collectParams`
/// order. Caller owns the returned slice and must free it with `gpa`; the
/// refs point into `c`'s storage and die with it.
pub fn collect(gpa: std.mem.Allocator, c: *const device.Circuit) !Params {
    const n = c.batches.len;
    const starts = try gpa.alloc(u32, n + 1);
    const layouts = try gpa.alloc(?device.abi.ParamLayout, n);
    const fallback = try gpa.alloc([]const ParamRef, n);
    starts[0] = 0;
    for (c.batches, c.batch_types, 0..) |b, t, i| {
        fallback[i] = &.{};
        layouts[i] = null;
        if (b.hooks.param_layout) |layout| {
            const l = try layout(b.ctx).unwrap();
            layouts[i] = l;
            starts[i + 1] = starts[i] + @as(u32, @intCast(l.count * l.fields.len));
            continue;
        }
        var list: std.ArrayList(ParamRef) = .empty;
        try b.hooks.collect_params(b.ctx, gpa, &list).unwrap();
        for (list.items) |*r| r.type = t;
        fallback[i] = list.items;
        starts[i + 1] = starts[i] + @as(u32, @intCast(list.items.len));
    }
    return .{ .batches = c.batches, .types = c.batch_types, .starts = starts, .layouts = layouts, .fallback = fallback, .len = starts[n] };
}

// ---- private ----
