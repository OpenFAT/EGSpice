//! A circuit's device parameters by ordinal (`collectParams` numbering):
//! the collected ref table, writes that re-derive only the batches they
//! reach, Model rows unshared on demand, and reads that unshare nothing.
//! `Circuit` exposes each function here as a method.
//! In: a circuit and parameter ordinals. Out: refs, values, writes.
//! Cites no outside source.
//!
//! Contents:
//! - `pub const ParamValue = struct { value, type }`
//! - `pub fn collectParams(self: *Circuit) ![]const ParamRef`
//! - `pub fn setParams(self: *Circuit, refs: []const u32, values: []const f64, rederive: bool) !void`
//! - `pub fn isolatedParams(self: *Circuit, gpa: Allocator, ords: []const u32) ![]ParamRef`
//! - `pub fn paramAt(self: *Circuit, ord: u32) !ParamValue`
//! - `pub fn typeName(self: *const Circuit, t: DeviceType) []const u8`

const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;
const abi = @import("device").abi;
const Circuit = @import("Circuit.zig").Circuit;
const ParamRef = abi.ParamRef;

/// One parameter's value and its device type.
pub const ParamValue = struct { value: f64, type: abi.DeviceType };

/// Returns every device parameter, typed by batch. Built once and freed by
/// `release`; the refs point into frozen batch storage, so they stay valid
/// across `restoreState` and die with the batches.
pub fn collectParams(self: *Circuit) ![]const ParamRef {
    if (self.param_refs) |refs| return refs;
    const gpa = self.gpa;
    var list: std.ArrayList(ParamRef) = .empty;
    errdefer list.deinit(gpa);

    for (self.batches, self.batch_types) |b, t| {
        const first = list.items.len;
        try b.hooks.collect_params(b.ctx, gpa, &list).unwrap();
        for (list.items[first..]) |*ref| ref.type = t;
    }
    self.param_refs = try list.toOwnedSlice(gpa);
    return self.param_refs.?;
}

/// Writes `values` to the parameters numbered `refs`. With `rederive`,
/// re-derives the batches they reach, which is `recompute` when nothing else
/// moved. With the template's `param_starts` only those batches'
/// parameters are collected; without them, all of them. Fails with
/// error.TopologyChanged when a re-derived batch re-wires. Asserts one value
/// per ref.
pub fn setParams(self: *Circuit, refs: []const u32, values: []const f64, rederive: bool) !void {
    assert(refs.len == values.len);
    const starts = self.param_starts;
    if (starts.len == 0) {
        const params = try self.collectParams();
        for (refs, values) |r, v| params[r].set(v);
        if (rederive) try self.recompute();
        return;
    }
    assert(starts.len == self.batches.len + 1);
    const gpa = self.gpa;
    const touched = try gpa.alloc(bool, self.batches.len);
    defer gpa.free(touched);

    @memset(touched, false);
    for (refs) |r| touched[batchOf(starts, r)] = true;
    var list: std.ArrayList(ParamRef) = .empty;
    defer list.deinit(gpa);

    for (self.batches, touched, starts[0..self.batches.len], starts[1..]) |b, t, lo, hi| {
        if (!t) continue;
        if (b.hooks.write_params) |write| {
            try write(b.ctx, refs, values, lo, hi).unwrap();
            continue;
        }
        list.clearRetainingCapacity();
        try b.hooks.collect_params(b.ctx, gpa, &list).unwrap();
        assert(list.items.len == hi - lo);
        for (refs, values) |r, v| if (r >= lo and r < hi) list.items[r - lo].set(v);
    }
    if (!rederive) return;
    self.paramsMoved();
    for (self.batches, touched) |b, t| if (t) if (b.hooks.recompute) |f| {
        if (!f(b.ctx)) return error.TopologyChanged;
    };
}

/// Returns refs to the parameters numbered `ords` that each write their own
/// instance alone: an instance a Model-field ref reaches gets a private copy
/// of its row when others read it (`Hooks.isolate_rows`), and the rest keep
/// sharing, so a mismatch over a few devices of a large circuit copies a few
/// rows. Without the template's `param_starts`, or on a batch without the
/// hooks, every row is unshared (`collectParams`). Valid until the rows move
/// again. Fails with error.InvalidVariation for an ordinal past the
/// circuit's parameters. Caller frees the slice with `gpa`.
pub fn isolatedParams(self: *Circuit, gpa: Allocator, ords: []const u32) ![]ParamRef {
    const out = try gpa.alloc(ParamRef, ords.len);
    errdefer gpa.free(out);

    const starts = self.param_starts;
    const layouts = starts.len != 0 and self.param_refs == null and for (ords) |o| {
        if (o >= starts[self.batches.len]) return error.InvalidVariation;
        const h = self.batches[batchOf(starts, o)].hooks;
        if (h.param_layout == null or h.isolate_rows == null) break false;
    } else true;
    if (!layouts) {
        const all = try self.collectParams();
        for (ords, out) |o, *ref| ref.* = if (o < all.len) all[o] else return error.InvalidVariation;
        return out;
    }
    var ids: std.ArrayList(u32) = .empty;
    defer ids.deinit(gpa);

    for (self.batches, starts[0..self.batches.len], starts[1..]) |b, lo, hi| {
        ids.clearRetainingCapacity();
        for (ords) |o| if (o >= lo and o < hi) {
            const l = try b.hooks.param_layout.?(b.ctx).unwrap();
            if (!l.fields[(o - lo) / l.count].is_instance) try ids.append(gpa, (o - lo) % l.count);
        };
        if (ids.items.len == 0) continue;
        std.mem.sort(u32, ids.items, {}, std.sort.asc(u32));
        var unique: usize = 0;
        for (ids.items) |id| if (unique == 0 or ids.items[unique - 1] != id) {
            ids.items[unique] = id;
            unique += 1;
        };
        try b.hooks.isolate_rows.?(b.ctx, ids.items[0..unique]).unwrap();
    }
    for (ords, out) |o, *ref| {
        const k = batchOf(starts, o);
        const b = self.batches[k];
        const l = try b.hooks.param_layout.?(b.ctx).unwrap();
        const r = o - starts[k];
        const f = l.fields[r / l.count];
        ref.* = .{ .ptr = l.ptr(r), .type = self.batch_types[k], .param_name = f.name, .index = r % l.count, .is_instance = f.is_instance, .primary = f.primary };
    }
    return out;
}

/// Returns parameter `ord`'s value and device type, read without unsharing
/// any Model row when the template has `param_starts` and the batch a
/// `param_layout`.
pub fn paramAt(self: *Circuit, ord: u32) !ParamValue {
    const starts = self.param_starts;
    if (starts.len != 0) {
        const k = batchOf(starts, ord);
        const b = self.batches[k];
        if (b.hooks.param_layout) |layout| return .{ .value = (try layout(b.ctx).unwrap()).ptr(ord - starts[k]).*, .type = self.batch_types[k] };
    }
    const ref = (try self.collectParams())[ord];
    return .{ .value = ref.get(), .type = ref.type };
}

/// Returns the short type name of device type `t`, or "" when absent.
pub fn typeName(self: *const Circuit, t: abi.DeviceType) []const u8 {
    for (self.batches, self.batch_types) |b, bt| if (bt == t) return b.type_name;
    return "";
}

// ---- private ----

/// The batch whose ordinals hold `ord`: the last start at or below it.
fn batchOf(starts: []const u32, ord: u32) usize {
    const k = std.sort.upperBound(u32, starts, ord, orderU32);
    assert(k >= 1);
    return k - 1;
}

fn orderU32(a: u32, b: u32) std.math.Order {
    return std.math.order(a, b);
}
