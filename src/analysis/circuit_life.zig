//! A circuit's lifetime: building one over a prepared template (copying or
//! moving its device state), freeing it, and the device-state snapshots an
//! analysis that rewinds time saves and restores. `Circuit` exposes each
//! function here as a method or a namespace member.
//! In: a `device.Circuit` template or a live circuit. Out: a circuit that
//! owns its planes and device state.
//! Cites no outside source.
//!
//! Contents:
//! - `pub const SavedState = struct` (`deinit`)
//! - `pub fn instantiate(template: *const Prepared, gpa: Allocator) !Circuit`
//! - `pub fn instantiateMove(template: *const Prepared, gpa: Allocator) !Circuit`
//! - `pub fn fromSnapshot(template: *const Prepared, source: *const Circuit, gpa: Allocator) !Circuit`
//! - `pub fn fromSnapshotMove(template: *const Prepared, source: *Circuit, gpa: Allocator) !Circuit`
//! - `pub fn freeze(gpa, n, intern_bytes, intern_offs, protos, types, bbd) !Circuit`
//! - `pub fn deinit(self: *Circuit) void`
//! - `pub fn release(self: *Circuit) void`
//! - `pub fn saveState(self: *const Circuit, gpa: Allocator) !SavedState`
//! - `pub fn storeState(self: *const Circuit, saved: SavedState) void`
//! - `pub fn restoreState(self: *Circuit, saved: SavedState) void`
//! - `pub fn refuseDigital(self: *const Circuit, analysis: []const u8) error{DigitalDeviceUnsupported}!void`

const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;
const abi = @import("device").abi;
const Prepared = @import("device").Circuit;
const BbdInfo = @import("core").numerics.BbdInfo;
const circuit_mod = @import("Circuit.zig");
const Circuit = circuit_mod.Circuit;
const certify = @import("certify.zig");
const Batch = abi.Batch;

/// Device state held outside the circuit, one `snapshot` per batch, so an
/// analysis that must rewind time can restore what it committed.
pub const SavedState = struct {
    batches: []Batch,

    /// Frees every snapshot; `gpa` must be the allocator `saveState` took.
    pub fn deinit(self: SavedState, gpa: Allocator) void {
        freeBatches(gpa, self.batches);
    }
};

/// Whether a new circuit frees the template's pattern, tapes and names.
const Topology = enum { borrowed, owned };

/// Returns a circuit that shares `template`'s immutable topology and owns
/// fresh device and solver state. `template` must outlive it and must never
/// have been evaluated. Free it with `deinit`.
pub fn instantiate(template: *const Prepared, gpa: Allocator) !Circuit {
    const batches = try copyBatches(gpa, template.batches, "instantiate");
    errdefer freeBatches(gpa, batches);

    return allocate(template.*, gpa, batches, .borrowed);
}

/// `instantiate` for the last query that will ever read `template`'s device
/// state: the circuit evaluates the template's own batches in place instead
/// of copying them. A never-evaluated template already holds the state
/// `instantiate` would copy, and it keeps ownership.
pub fn instantiateMove(template: *const Prepared, gpa: Allocator) !Circuit {
    var ckt = try allocate(template.*, gpa, template.batches, .borrowed);
    ckt.owns_batches = false;
    return ckt;
}

/// Returns a copy of `source`'s device state over `template`'s topology, for
/// a query that starts from a completed operating point. Fails with
/// `error.IncompatibleDependency` when `source` was not built from
/// `template`. Free it with `deinit`.
pub fn fromSnapshot(template: *const Prepared, source: *const Circuit, gpa: Allocator) !Circuit {
    if (source.n != template.n or source.col_ptr.ptr != template.col_ptr.ptr) return error.IncompatibleDependency;
    const batches = try copyBatches(gpa, source.batches, "snapshot");
    errdefer freeBatches(gpa, batches);

    var ckt = try allocate(template.*, gpa, batches, .borrowed);
    ckt.sim = source.sim;
    ckt.temp_c = source.temp_c;
    return ckt;
}

/// `fromSnapshot` for the last query that will ever read `source`: the
/// device state moves instead of being copied, and `source` keeps no
/// batches and frees the rest of its state at once (its `deinit` is then a
/// no-op). Copies when the two circuits do not share an allocator.
pub fn fromSnapshotMove(template: *const Prepared, source: *Circuit, gpa: Allocator) !Circuit {
    if (source.n != template.n or source.col_ptr.ptr != template.col_ptr.ptr) return error.IncompatibleDependency;
    if (source.owns_batches and (source.gpa.ptr != gpa.ptr or source.gpa.vtable != gpa.vtable))
        return fromSnapshot(template, source, gpa);
    var ckt = try allocate(template.*, gpa, source.batches, .borrowed);
    ckt.owns_batches = source.owns_batches;
    ckt.sim = source.sim;
    ckt.temp_c = source.temp_c;
    source.batches = &.{};
    source.release();
    assert(source.batches.len == 0);
    return ckt;
}

/// Freezes `protos` into an owning Circuit (test construction). Takes
/// ownership of the intern table and consumes the protos.
pub fn freeze(
    gpa: Allocator,
    n: u32,
    intern_bytes: []u8,
    intern_offs: []u32,
    protos: []const abi.Proto,
    types: []const abi.DeviceType,
    bbd: ?BbdInfo,
) !Circuit {
    var prepared = try Prepared.freeze(gpa, n, intern_bytes, intern_offs, protos, types, bbd, .{});
    errdefer prepared.deinit();

    return allocate(prepared, prepared.allocator, prepared.batches, .owned);
}

/// `release`, then poisons the struct.
pub fn deinit(self: *Circuit) void {
    self.release();
    self.* = undefined;
}

/// Frees all state the circuit owns and leaves an empty one whose `release`
/// and `deinit` free nothing more. Nothing may evaluate or solve it
/// afterwards.
pub fn release(self: *Circuit) void {
    const gpa = self.gpa;
    if (self.knobs.tran_stats) for (self.batches) |b| {
        const count = b.hooks.derivative_counts orelse continue;
        const counts = count(b.ctx);
        std.debug.print("derivative-stats: model={s} wide={d} narrow={d} wide_lanes={d} narrow_lanes={d}\n", .{ b.type_name, counts.wide, counts.narrow, counts.wide_lanes, counts.narrow_lanes });
    };
    if (certify.enabled) self.cert.deinit(gpa, self.nodeName(self.cert.worst_row));
    if (self.ws) |*w| {
        if (self.pooled()) |p| p.give(@intFromPtr(self.col_ptr.ptr), self.knobs.newton, self.knobs.lu, w.*) else w.deinit(gpa);
    }
    self.ws = null;
    if (self.load_check) |*lc| lc.deinit(gpa);
    self.load_check = null;
    if (self.param_refs) |refs| gpa.free(refs);
    self.param_refs = null;
    if (self.owns_batches) freeBatches(gpa, self.batches);
    self.batches = &.{};
    // g_base/c_base unconditionally: a failed computeBaseline can leave them
    // allocated with has_baseline false.
    inline for (.{ "g_vals", "c_vals", "g_base", "c_base", "rhs", "q_vec" }) |name| {
        gpa.free(@field(self, name));
        @field(self, name) = &.{};
    }
    self.has_baseline = false;
    if (!self.owns_topology) return;
    gpa.free(self.col_ptr);
    gpa.free(self.row_idx);
    gpa.free(self.diag_slots);
    gpa.free(self.current_row);
    gpa.free(self.batch_types);
    if (self.bbd) |bbd| gpa.free(bbd.blocks);
    gpa.free(self.ac_dyn_slots);
    gpa.free(self.intern_bytes);
    gpa.free(self.intern_offs);
    gpa.free(self.param_starts);
    if (self.column_order) |o| o.destroy(self.n);
    self.column_order = null;
    self.owns_topology = false;
}

/// Returns a copy of every batch's device state. The caller frees it with
/// `SavedState.deinit`.
pub fn saveState(self: *const Circuit, gpa: Allocator) !SavedState {
    return .{ .batches = try copyBatches(gpa, self.batches, "snapshot") };
}

/// Overwrites `saved` with the current device state, allocating nothing.
/// Asserts that `saved` came from this circuit's `saveState`.
pub fn storeState(self: *const Circuit, saved: SavedState) void {
    assert(saved.batches.len == self.batches.len);
    for (saved.batches, self.batches) |dst, src| dst.hooks.copy_state(dst.ctx, src.ctx);
}

/// Rewinds every batch to `saved`, which becomes the accepted state. Batch
/// storage and every `ParamRef` into it stay valid. Does nothing while a GPU
/// context is attached.
// ponytail: host batches only. Under a GPU hook the resident state is left
// where it is; download it (`syncHostState`) and re-upload states too when a
// GPU-resident device carries history that must rewind.
pub fn restoreState(self: *Circuit, saved: SavedState) void {
    if (self.gpuContext() != null) return;
    assert(saved.batches.len == self.batches.len);
    for (self.batches, saved.batches) |dst, src| dst.hooks.copy_state(dst.ctx, src.ctx);
    self.state_staged = false;
    self.lin.valid = false;
}

/// Fails with `error.DigitalDeviceUnsupported` when a device type runs a
/// digital engine (`Batch.digital`), for an analysis that rewinds time (pss,
/// qpss, hb, envelope): a committed digital time cannot move back. A warning
/// names the analysis and the type.
pub fn refuseDigital(self: *const Circuit, analysis: []const u8) error{DigitalDeviceUnsupported}!void {
    for (self.batches) |b| if (b.digital) {
        std.log.warn("{s}: device type '{s}' holds digital state, whose time cannot rewind", .{ analysis, b.type_name });
        return error.DigitalDeviceUnsupported;
    };
}

// ---- private ----

/// The circuit over `data` with fresh planes and `batches`.
fn allocate(data: Prepared, gpa: Allocator, batches: []Batch, topology: Topology) !Circuit {
    assert(batches.len == data.batches.len);
    const g_vals = try gpa.alloc(f64, @as(usize, data.nnz) + 1);
    errdefer gpa.free(g_vals);

    const c_vals = try gpa.alloc(f64, @as(usize, data.nnz) + 1);
    errdefer gpa.free(c_vals);

    const rhs = try gpa.alloc(f64, @as(usize, data.n) + 1);
    errdefer gpa.free(rhs);

    const q_vec = try gpa.alloc(f64, @as(usize, data.n) + 1);
    @memset(c_vals, 0);
    @memset(q_vec, 0);
    // `Batch.digital` comes from the Library, not the device, so a batch
    // rebuilt by `instantiate`/`snapshot` takes it from the template.
    for (batches, data.batches) |*b, t| b.digital = t.digital;
    return .{
        .gpa = gpa,
        .col_ptr = data.col_ptr,
        .row_idx = data.row_idx,
        .nnz = data.nnz,
        .trash_slot = data.nnz,
        .n = data.n,
        .g_vals = g_vals,
        .c_vals = c_vals,
        .rhs = rhs,
        .q_vec = q_vec,
        .diag_slots = data.diag_slots,
        .batches = batches,
        .batch_types = data.batch_types,
        .current_row = data.current_row,
        .has_charge = data.has_charge,
        .has_state_q = data.has_state_q,
        .has_baseline = false,
        .g_base = &.{},
        .c_base = &.{},
        .needs_tran_op = data.needs_tran_op,
        .vpi = data.vpi,
        .intern_bytes = data.intern_bytes,
        .intern_offs = data.intern_offs,
        .bbd = data.bbd,
        .ac_dyn_slots = data.ac_dyn_slots,
        .param_starts = data.param_starts,
        .column_order = data.column_order,
        .owns_topology = topology == .owned,
    };
}

/// One copy of each of `sources` made by `Hooks.<hook>` (`instantiate` or
/// `snapshot`). Caller frees it with `freeBatches`. Fails with the hook's
/// error, leaving nothing allocated.
fn copyBatches(gpa: Allocator, sources: []const Batch, comptime hook: []const u8) ![]Batch {
    const batches = try gpa.alloc(Batch, sources.len);
    errdefer gpa.free(batches);

    var count: usize = 0;
    errdefer for (batches[0..count]) |b| b.hooks.deinit(b.ctx, gpa);
    for (sources, batches) |source, *copy| {
        copy.* = try @field(source.hooks, hook)(source.ctx, gpa).unwrap();
        count += 1;
    }
    assert(count == sources.len);
    return batches;
}

fn freeBatches(gpa: Allocator, batches: []const Batch) void {
    for (batches) |b| b.hooks.deinit(b.ctx, gpa);
    gpa.free(batches);
}
