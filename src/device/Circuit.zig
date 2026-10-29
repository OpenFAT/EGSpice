//! The frozen circuit: the construction protos and node names in, the CSC
//! matrix pattern, one device batch per type and the topology facts out.
//! Analysis instantiates its mutable state from it. Cites ngspice's OPtran
//! for `needs_tran_op` and #13 for `Facts`.
//!
//! Contents:
//! - `allocator`, `n`, `nnz`, `col_ptr`, `row_idx`, `diag_slots`, `batches`,
//!   `batch_types`, `current_row`, `intern_bytes`, `intern_offs`, `bbd`,
//!   `ac_dyn_slots`, `has_charge`, `has_state_q`, `needs_tran_op`, `vpi`,
//!   `param_starts`, `column_order` (fields)
//! - `pub const ColumnOrder = struct`, with `get`, `share`, `destroy`
//! - `pub const Facts = struct`
//! - `pub const FreezeError = error{...}`
//! - `pub fn freeze(allocator: Allocator, n: u32, intern_bytes: []u8, intern_offs: []u32, protos: []const abi.Proto, types: []const abi.DeviceType, bbd: ?numerics.BbdInfo, facts: Facts) FreezeError!Circuit`
//! - `pub fn deinit(self: *Circuit) void`
//! - `pub fn countParams(self: *Circuit) error{ OutOfMemory, TooManyInstances }!void`
//! - `pub fn nodeName(self: Circuit, node: u32) []const u8`
//! - `pub fn findSlot(self: Circuit, row: u32, col: u32) ?u32`

const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;
const abi = @import("device_abi");
const core = @import("core");
const numerics = core.numerics;
const Library = @import("Library.zig");

/// Owns every slice below.
allocator: Allocator,
/// Matrix dimension; every node id is below it.
n: u32,
/// Structural nonzeros; also the trash slot ground stamps land in.
nnz: u32,
/// CSC column starts, n + 1 entries.
col_ptr: []u32,
/// CSC row of each slot, nnz entries, ascending within a column.
row_idx: []u32,
/// CSC slot of each diagonal entry.
diag_slots: []u32,
/// One batch per proto, in `protos` order, each owned.
batches: []abi.Batch,
/// The Library type of each batch, parallel to `batches`.
batch_types: []abi.DeviceType,
/// Rows whose unknown is a branch current rather than a node voltage.
current_row: []bool,
/// Node names, concatenated; name i is `intern_bytes[intern_offs[i]..
/// intern_offs[i + 1]]`.
intern_bytes: []u8,
intern_offs: []u32,
/// The frontend's bordered-block-diagonal row partition, or null; `blocks`
/// is owned and freed by `deinit`.
bbd: ?numerics.BbdInfo,
/// CSC slot of every frequency-dependent small-signal entry, batch-major in
/// `batches` order, then as each batch's `Hooks.collect_ac_dyn` lists them;
/// `nnz` marks a ground entry. Entry e pairs with term e of `Hooks.ac_dyn`.
ac_dyn_slots: []u32,
/// Some batch stamps charge (`Batch.has_charge`).
has_charge: bool,
/// Some charge-bearing batch also keeps accepted-step state, so a transient
/// must restamp q after `update_state`/`commit_state`.
has_state_q: bool,
/// The builder found a node with only capacitors to it, so the operating
/// point falls back to ngspice's OPtran. From `Facts` at the freeze.
needs_tran_op: bool = false,
/// The VPI host whose LRM 12.31.3 callbacks `.op` and `.tran` drive: set
/// when an instance calls a `$systf` through an attached application.
vpi: ?*const Library.Vpi = null,
/// Per batch, the ordinal of its first parameter in the list
/// `Hooks.collect_params` builds batch by batch, then the total; empty
/// until `countParams`. A variant row then reaches the batches it writes
/// without collecting the others.
param_starts: []u32 = &.{},
/// The sparse LU's BTF + AMD column order of this pattern, filled by the
/// first query that factors and copied by the rest (`ColumnOrder`), so a
/// sweep orders its pattern once instead of per point. Heap-held, so it
/// survives the Circuit value moving; null for a hand-made circuit.
column_order: ?*ColumnOrder = null,

/// A once-filled, lock-free slot for a pattern's column order. Safe to
/// `get` and `share` from any thread.
pub const ColumnOrder = struct {
    /// Owns `q`: the template's allocator, whatever allocator the query
    /// that fills it runs on.
    gpa: Allocator,
    q: std.atomic.Value(?[*]u32) = .init(null),

    /// Returns the order, n entries, once some query has shared one.
    pub fn get(c: *const ColumnOrder, n: u32) ?[]const u32 {
        const p = c.q.load(.acquire) orelse return null;
        return p[0..n];
    }

    /// Keeps a copy of `q` unless another query got there first.
    pub fn share(c: *ColumnOrder, q: []const u32) error{OutOfMemory}!void {
        if (c.q.load(.acquire) != null) return;
        const copy = try c.gpa.dupe(u32, q);
        // The loser of a race frees its copy; the winner's stays.
        if (c.q.cmpxchgStrong(null, copy.ptr, .acq_rel, .acquire) != null) c.gpa.free(copy);
        assert(c.q.load(.acquire) != null);
    }

    /// Frees the shared order and the slot.
    pub fn destroy(c: *ColumnOrder, n: u32) void {
        const gpa = c.gpa;
        if (c.q.load(.acquire)) |p| gpa.free(p[0..n]);
        gpa.destroy(c);
    }
};

/// Topology facts `freeze` records on the Circuit.
pub const Facts = struct {
    /// A node has no DC path (a capacitor-only island): only the transient's
    /// own operating point is defined.
    needs_tran_op: bool = false,
    /// Per `DeviceType`, whether the type runs a digital engine
    /// (`Library.digital`); types past its end are analog.
    digital: []const bool = &.{},
    /// Becomes `Circuit.vpi`.
    vpi: ?*const Library.Vpi = null,
};

/// Every way `freeze` fails. On any error the caller keeps `intern_bytes`,
/// `intern_offs` and `bbd`; a proto whose `finalize` failed is only fit for
/// `destroy`.
pub const FreezeError = error{
    /// `n` is 0, or `intern_offs` is not n + 1 ascending offsets ending at
    /// `intern_bytes.len`, or `types` does not hold one id per proto.
    InvalidCircuit,
    /// A proto stages more than `maxInt(u32)` instances.
    TooManyInstances,
    OutOfMemory,
};

const Circuit = @This();

/// A one-instance device that stamps (row, col) and owns one heap word, so a
/// leaked or double-freed batch shows up under the testing allocator.
const FakeProto = struct {
    row: u32,
    col: u32,

    const hooks: abi.Hooks = .{
        .instantiate = undefined,
        .snapshot = undefined,
        .copy_state = undefined,
        .stage = undefined,
        .scatter_tapes = undefined,
        .set_sim_state = undefined,
        .collect_params = undefined,
        .mark_current_rows = markCurrentRows,
        .collect_ac_dyn = collectAcDyn,
        .deinit = deinitBatch,
    };

    fn proto(self: *FakeProto) abi.Proto {
        return .{ .ctx = self, .type_name = "fake", .pattern = pattern, .finalize = finalize, .destroy = destroy, .apply_perm = applyPerm };
    }
    fn pattern(ctx: *anyopaque, gpa: Allocator, pb: *abi.PatternBuilder) abi.DeviceResult(void) {
        const self: *FakeProto = @ptrCast(@alignCast(ctx));
        return abi.DeviceResult(void).fromLocal(pb.add(gpa, self.row, self.col));
    }
    fn finalize(ctx: *anyopaque, gpa: Allocator, view: abi.PatternView) abi.DeviceResult(abi.Batch) {
        const self: *FakeProto = @ptrCast(@alignCast(ctx));
        const slot = gpa.create(u32) catch return .out_of_memory;
        slot.* = view.findSlot(self.row, self.col).?;
        return .{ .ok = .{
            .ctx = slot,
            .eval = undefined,
            .eval_newton = undefined,
            .count = 1,
            .n_u = 2,
            .has_charge = false,
            .has_const_jacobian = true,
            .type_name = "fake",
            .hooks = &hooks,
        } };
    }
    fn destroy(_: *anyopaque, _: Allocator) void {}
    fn applyPerm(_: *anyopaque, _: []const u32) void {}
    fn markCurrentRows(_: *anyopaque, rows: []bool) void {
        rows[rows.len - 1] = true;
    }
    fn collectAcDyn(ctx: *anyopaque, gpa: Allocator, out: *std.ArrayList(u32)) abi.DeviceResult(void) {
        const slot: *u32 = @ptrCast(@alignCast(ctx));
        return abi.DeviceResult(void).fromLocal(out.append(gpa, slot.*));
    }
    fn deinitBatch(ctx: *anyopaque, gpa: Allocator) void {
        gpa.destroy(@as(*u32, @ptrCast(@alignCast(ctx))));
    }

    /// `freeze` of one fake proto over nodes "a" and "b", checked against
    /// the hand-built pattern.
    fn freezeCase(gpa: Allocator) !void {
        var fake: FakeProto = .{ .row = 0, .col = 1 };
        const protos = [_]abi.Proto{fake.proto()};
        const bytes = try gpa.dupe(u8, "ab");
        const offs = gpa.dupe(u32, &.{ 0, 1, 2 }) catch |err| {
            gpa.free(bytes);
            return err;
        };
        // On failure the caller keeps the names.
        var ckt = freeze(gpa, 2, bytes, offs, &protos, &.{@fromBackingInt(0)}, null, .{}) catch |err| {
            gpa.free(bytes);
            gpa.free(offs);
            return err;
        };
        defer ckt.deinit();

        const t = std.testing;
        // The diagonals plus (0, 1): column 0 {0}, column 1 {0, 1}.
        try t.expectEqual(@as(u32, 3), ckt.nnz);
        try t.expectEqualSlices(u32, &.{ 0, 1, 3 }, ckt.col_ptr);
        try t.expectEqualSlices(u32, &.{ 0, 0, 1 }, ckt.row_idx);
        try t.expectEqualSlices(u32, &.{ 0, 2 }, ckt.diag_slots);
        try t.expectEqualSlices(bool, &.{ false, true }, ckt.current_row);
        try t.expectEqualSlices(u32, &.{1}, ckt.ac_dyn_slots);
        try t.expectEqual(@as(?u32, 1), ckt.findSlot(0, 1));
        try t.expectEqual(@as(?u32, null), ckt.findSlot(1, 0));
        try t.expectEqual(@as(?u32, null), ckt.findSlot(2, 0));
        try t.expectEqual(@as(?u32, null), ckt.findSlot(0, 2));
        try t.expectEqualStrings("b", ckt.nodeName(1));
        try t.expectEqualStrings("", ckt.nodeName(2));
        try t.expect(!ckt.has_charge);
        try t.expect(!ckt.has_state_q);
    }
};

/// Builds the pattern from every proto and finalizes each into a batch
/// allocated with `allocator`. On success it destroys the protos and takes
/// ownership of `intern_bytes`, `intern_offs` and `bbd`. `facts` carries
/// what the netlist decided beside the protos, so no caller sets it
/// afterwards (#13). Cost: one sort of every structural entry, O(entries).
pub fn freeze(
    allocator: Allocator,
    n: u32,
    intern_bytes: []u8,
    intern_offs: []u32,
    protos: []const abi.Proto,
    types: []const abi.DeviceType,
    bbd: ?numerics.BbdInfo,
    facts: Facts,
) FreezeError!Circuit {
    // The trust boundary: every table the builder hands over is checked
    // here, before any proto is read.
    if (n == 0) return error.InvalidCircuit;
    if (intern_offs.len != @as(usize, n) + 1) return error.InvalidCircuit;
    if (types.len != protos.len) return error.InvalidCircuit;
    if (intern_offs[n] != intern_bytes.len) return error.InvalidCircuit;
    if (!std.sort.isSorted(u32, intern_offs, {}, std.sort.asc(u32))) return error.InvalidCircuit;

    var col_ptr: []u32 = undefined;
    var row_idx: []u32 = undefined;
    const nnz = try buildPattern(allocator, n, protos, &col_ptr, &row_idx);
    errdefer allocator.free(col_ptr);
    errdefer allocator.free(row_idx);
    const view: abi.PatternView = .{ .col_ptr = col_ptr, .row_idx = row_idx, .n = n, .trash_slot = nnz };

    const diag_slots = try allocator.alloc(u32, n);
    errdefer allocator.free(diag_slots);
    // `buildPattern` added every diagonal.
    for (diag_slots, 0..) |*slot, i| slot.* = view.findSlot(@intCast(i), @intCast(i)).?;

    const batches = try allocator.alloc(abi.Batch, protos.len);
    errdefer allocator.free(batches);
    var finalized: usize = 0;
    errdefer for (batches[0..finalized]) |batch| batch.hooks.deinit(batch.ctx, allocator);
    for (protos, batches) |proto, *batch| {
        batch.* = try proto.finalize(proto.ctx, allocator, view).unwrap();
        finalized += 1;
    }

    const batch_types = try allocator.dupe(abi.DeviceType, types);
    errdefer allocator.free(batch_types);
    const column_order = try allocator.create(ColumnOrder);
    errdefer allocator.destroy(column_order);
    column_order.* = .{ .gpa = allocator };
    const current_row = try allocator.alloc(bool, n);
    errdefer allocator.free(current_row);
    @memset(current_row, false);
    for (batches) |batch| if (batch.hooks.mark_current_rows) |mark| mark(batch.ctx, current_row);
    var ac_dyn: std.ArrayList(u32) = .empty;
    errdefer ac_dyn.deinit(allocator);
    for (batches) |batch| if (batch.hooks.collect_ac_dyn) |collect| try collect(batch.ctx, allocator, &ac_dyn).unwrap();
    const ac_dyn_slots = try ac_dyn.toOwnedSlice(allocator);

    // Nothing fails past this line: the protos die and the batches commit.
    assert(finalized == protos.len);
    for (protos) |proto| proto.destroy(proto.ctx, allocator);
    var has_charge = false;
    var has_state_q = false;
    for (batches, batch_types) |*b, t| {
        const i: usize = @backingInt(t);
        b.digital = t != .unset and i < facts.digital.len and facts.digital[i];
        has_charge = has_charge or b.has_charge;
        has_state_q = has_state_q or (b.has_charge and
            (b.hooks.update_state != null or b.hooks.commit_state != null));
    }
    if (has_state_q) assert(has_charge);
    return .{
        .needs_tran_op = facts.needs_tran_op,
        .vpi = facts.vpi,
        .allocator = allocator,
        .n = n,
        .nnz = nnz,
        .col_ptr = col_ptr,
        .row_idx = row_idx,
        .diag_slots = diag_slots,
        .batches = batches,
        .batch_types = batch_types,
        .current_row = current_row,
        .intern_bytes = intern_bytes,
        .intern_offs = intern_offs,
        .bbd = bbd,
        .ac_dyn_slots = ac_dyn_slots,
        .has_charge = has_charge,
        .has_state_q = has_state_q,
        .column_order = column_order,
    };
}

/// Frees every batch and table, the adopted node names and `bbd` included.
/// Asserts that the tables still have their frozen lengths.
pub fn deinit(self: *Circuit) void {
    assert(self.batch_types.len == self.batches.len);
    assert(self.intern_offs.len == @as(usize, self.n) + 1);
    const allocator = self.allocator;
    for (self.batches) |batch| batch.hooks.deinit(batch.ctx, allocator);
    allocator.free(self.batches);
    allocator.free(self.batch_types);
    allocator.free(self.col_ptr);
    allocator.free(self.row_idx);
    allocator.free(self.diag_slots);
    allocator.free(self.current_row);
    allocator.free(self.ac_dyn_slots);
    allocator.free(self.intern_bytes);
    allocator.free(self.intern_offs);
    if (self.bbd) |bbd| allocator.free(bbd.blocks);
    allocator.free(self.param_starts);
    if (self.column_order) |o| o.destroy(self.n);
    self.* = undefined;
}

/// Fills `param_starts` once; a second call does nothing. A batch with
/// `param_layout` keeps its Model rows shared; the others are collected,
/// which unshares theirs. On error `param_starts` stays empty.
pub fn countParams(self: *Circuit) error{ OutOfMemory, TooManyInstances }!void {
    if (self.param_starts.len != 0) return;
    const starts = try self.allocator.alloc(u32, self.batches.len + 1);
    errdefer self.allocator.free(starts);
    var refs: std.ArrayList(abi.ParamRef) = .empty;
    defer refs.deinit(self.allocator);
    starts[0] = 0;
    for (self.batches, starts[0..self.batches.len], starts[1..]) |b, start, *next| {
        const count: usize = if (b.hooks.param_layout) |layout| blk: {
            const l = try layout(b.ctx).unwrap();
            break :blk @as(usize, l.count) * l.fields.len;
        } else blk: {
            refs.clearRetainingCapacity();
            try b.hooks.collect_params(b.ctx, self.allocator, &refs).unwrap();
            break :blk refs.items.len;
        };
        next.* = start + @as(u32, @intCast(count));
    }
    assert(starts[0] == 0);
    self.param_starts = starts;
}

/// Returns the name of `node`, or "" when it is out of range.
pub fn nodeName(self: Circuit, node: u32) []const u8 {
    if (node >= self.n) return "";
    assert(self.intern_offs[node] <= self.intern_offs[node + 1]);
    assert(self.intern_offs[node + 1] <= self.intern_bytes.len);
    return self.intern_bytes[self.intern_offs[node]..self.intern_offs[node + 1]];
}

/// Returns the CSC slot of (row, col), or null when it is out of range or
/// not in the pattern. O(log) in the column's length.
pub fn findSlot(self: Circuit, row: u32, col: u32) ?u32 {
    if (row >= self.n or col >= self.n) return null;
    const view: abi.PatternView = .{ .col_ptr = self.col_ptr, .row_idx = self.row_idx, .n = self.n, .trash_slot = self.nnz };
    return view.findSlot(row, col);
}

// ---- private ----

/// The CSC pattern of every diagonal plus each proto's entries, into
/// `col_ptr_out` and `row_idx_out` (owned by `allocator`); returns nnz. The
/// keys live on core's process allocator, never the circuit's: an arena
/// would keep them (`PatternBuilder`).
fn buildPattern(allocator: Allocator, n: u32, protos: []const abi.Proto, col_ptr_out: *[]u32, row_idx_out: *[]u32) error{ OutOfMemory, TooManyInstances }!u32 {
    const scratch = core.gpa;
    var pattern: abi.PatternBuilder = .{};
    defer pattern.deinit(scratch);
    try pattern.reserve(scratch, n);
    for (0..n) |i| try pattern.add(scratch, @intCast(i), @intCast(i));
    for (protos) |proto| try proto.pattern(proto.ctx, scratch, &pattern).unwrap();
    const nnz = try pattern.toCsc(allocator, scratch, n, col_ptr_out, row_idx_out);
    // Every diagonal is in the pattern, so it holds at least n entries.
    assert(nnz >= n);
    assert(row_idx_out.len == nnz);
    return nnz;
}

test freeze {
    try FakeProto.freezeCase(std.testing.allocator);
    try std.testing.checkAllAllocationFailures(core.testing.fixed_growth, FakeProto.freezeCase, .{});
}

test "freeze rejects inconsistent tables before touching a proto" {
    const t = std.testing;
    var fake: FakeProto = .{ .row = 0, .col = 0 };
    const protos = [_]abi.Proto{fake.proto()};
    const one: []const abi.DeviceType = &.{@fromBackingInt(0)};
    var bytes = "ab".*;
    var short = [_]u32{0};
    var offs = [_]u32{ 0, 1, 2 };
    var past_end = [_]u32{ 0, 1, 3 };
    var descending = [_]u32{ 0, 3, 2 };
    try t.expectError(error.InvalidCircuit, freeze(t.allocator, 0, &bytes, &short, &.{}, &.{}, null, .{}));
    try t.expectError(error.InvalidCircuit, freeze(t.allocator, 2, &bytes, &short, &protos, one, null, .{}));
    try t.expectError(error.InvalidCircuit, freeze(t.allocator, 2, &bytes, &offs, &protos, &.{}, null, .{}));
    try t.expectError(error.InvalidCircuit, freeze(t.allocator, 2, &bytes, &past_end, &protos, one, null, .{}));
    try t.expectError(error.InvalidCircuit, freeze(t.allocator, 2, &bytes, &descending, &protos, one, null, .{}));
}
