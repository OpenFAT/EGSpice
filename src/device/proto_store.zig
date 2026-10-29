//! Construction-time rows for one device type: each staged instance's
//! Model, Instance and nodes in, a frozen `DeviceBatch(D)` out
//! (`finalize`), reached through the type-erased `Proto`. Equal Models
//! share one staged row. Cites VerA device ABI 6 for `setup` before
//! `initState`, and #142 for the permutation check.
//!
//! Contents:
//! - `pub const staging_gpa: Allocator`
//! - `pub fn permuteInPlace(comptime T: type, items: []T, dst: []const u32, seen: []bool) void`
//! - `pub fn ProtoStore(comptime D: type) type`, with fields `models`,
//!   `model_ids`, `seen`, `instances`, `nodes`, and `shares`, `append`,
//!   `appendRow`, `addPattern`, `finalize`, `applyPerm`, `destroy`; its
//!   `RowCtx` adapter's `hash` and `eql`

const std = @import("std");
const builtin = @import("builtin");
const assert = std.debug.assert;
const contract = @import("contract");
const abi = @import("device_abi");
const breakpoints = @import("breakpoints.zig");
const dual = @import("dual.zig");
const layout = @import("layout.zig");
const traits = @import("traits.zig");
const DeviceBatch = @import("batch.zig").DeviceBatch;
const GROUND = abi.GROUND;
const Batch = abi.Batch;
const PatternBuilder = abi.PatternBuilder;
const PatternView = abi.PatternView;
const LimMap = layout.LimMap;
const Mask = layout.Mask;
const SlotMap = layout.SlotMap;
const SrcBrk = breakpoints.SrcBrk;
const Real = dual.Real;

/// Allocator for `ProtoStore` staging columns. Not the caller's allocator,
/// which is usually an arena: an arena keeps every abandoned list capacity
/// alive, measured at 73 MB on a 25,000-MOSFET deck. Separate per-column
/// lists grow by remap, never by copy, and `finalize` moves one column at a
/// time into the caller's allocator, freeing each staging column at once.
/// `core.gpa`, which this file cannot import.
pub const staging_gpa = if (builtin.single_threaded) std.heap.page_allocator else std.heap.smp_allocator;

/// Moves `items[k]` to `items[dst[k]]` for every k by following each cycle
/// of the permutation, one element in hand at a time. `seen` is scratch of
/// the same length; its contents on entry do not matter. O(n), no
/// allocation. Asserts that `dst` and `seen` have `items.len` entries and
/// that `dst` is a permutation of `0..items.len` (every target in range,
/// none repeated): a repeated target would otherwise walk a cycle forever
/// (#142).
pub fn permuteInPlace(comptime T: type, items: []T, dst: []const u32, seen: []bool) void {
    assert(dst.len == items.len);
    assert(seen.len == items.len);
    // The permutation check costs one flag write per row, against the walk's
    // one `T` swap per row, and runs once per column at freeze time.
    @memset(seen, false);
    for (dst) |d| {
        assert(d < items.len);
        assert(!seen[d]);
        seen[d] = true;
    }
    @memset(seen, false);
    var moved: usize = 0;
    for (0..items.len) |start| {
        if (seen[start]) continue;
        var held = items[start];
        var j = start;
        // A cycle visits each row at most once, so it closes within len steps.
        for (0..items.len) |_| {
            seen[j] = true;
            const d = dst[j];
            std.mem.swap(T, &held, &items[d]);
            moved += 1;
            j = d;
            if (j == start) break;
        } else unreachable;
    }
    assert(moved == items.len);
}

/// Construction-time rows for one device type: each instance's Model,
/// Instance and nodes, staged until `finalize` freezes them into a
/// `DeviceBatch(D)`. Reached through the type-erased `Proto`.
pub fn ProtoStore(comptime D: type) type {
    const n_u: usize = comptime contract.nU(D);
    return struct {
        /// One entry per instance in `instances` and `nodes`, owned by
        /// `staging_gpa`. `models` holds one row per distinct (canonical)
        /// Model when the batch shares rows, `model_ids` naming each
        /// instance's; otherwise one row per instance and no ids. A BSIM4
        /// row is ~10 KB, so staging a copy per instance held most of the
        /// 1.8 GB a 1M-device deck peaked at during preparation.
        models: std.ArrayList(D.Model) = .empty,
        model_ids: std.ArrayList(u32) = .empty,
        /// `models` rows by content, for `append`'s lookup.
        seen: std.ArrayHashMapUnmanaged(u32, void, void, true) = .empty,
        instances: std.ArrayList(D.Instance) = .empty,
        nodes: std.ArrayList([n_u]u32) = .empty,

        const Self = @This();
        const shares = DeviceBatch(D).shares_models;

        /// Hashes and compares Model rows by bytes, padding included
        /// (rows are canonical: `canonicalize` zeroes the padding).
        const RowCtx = struct {
            rows: []const D.Model,
            pub fn hash(_: @This(), m: *const D.Model) u32 {
                return @truncate(std.hash.Wyhash.hash(0, std.mem.asBytes(m)));
            }
            pub fn eql(c: @This(), m: *const D.Model, row: u32, _: usize) bool {
                return std.mem.eql(u8, std.mem.asBytes(m), std.mem.asBytes(&c.rows[row]));
            }
        };

        /// Instance k's staged Model.
        fn stagedModel(self: *Self, k: usize) *D.Model {
            return &self.models.items[if (shares) self.model_ids.items[k] else k];
        }

        /// Stages one instance. The rows live on `staging_gpa`, never the
        /// caller's allocator. On failure no column grows. Asserts that a
        /// device calling a §12.32 `$name` has its application bound
        /// (`DeviceVtable.bind_systf`): `contract.validateHost` checks the
        /// binding only in Debug, and generated code unwraps it in every mode.
        /// Reads `model` and `instance` through pointers, copied straight into
        /// the rows: a delay line's Instance carries its history (wline_4:
        /// 2 MB), and by-value temporaries of it overflowed a Debug stack.
        pub fn append(self: *Self, model: *const D.Model, instance: *const D.Instance, nodes: [n_u]u32) !void {
            if (comptime @hasDecl(D, "systf_calls") and D.systf_calls.len != 0)
                assert(instance.systf != null);
            try self.models.ensureUnusedCapacity(staging_gpa, 1);
            try self.instances.ensureUnusedCapacity(staging_gpa, 1);
            try self.nodes.ensureUnusedCapacity(staging_gpa, 1);
            if (comptime shares) {
                try self.model_ids.ensureUnusedCapacity(staging_gpa, 1);
                try self.seen.ensureUnusedCapacity(staging_gpa, 1);
                const n: u32 = @intCast(self.models.items.len);
                const row = self.models.addOneAssumeCapacity();
                row.* = model.*;
                canonicalize(row);
                const gop = self.seen.getOrPutAssumeCapacityAdapted(row, RowCtx{ .rows = self.models.items[0..n] });
                if (gop.found_existing) self.models.items.len -= 1 else gop.key_ptr.* = n;
                self.model_ids.appendAssumeCapacity(gop.key_ptr.*);
            } else self.models.addOneAssumeCapacity().* = model.*;
            self.nodes.appendAssumeCapacity(nodes);
            self.instances.addOneAssumeCapacity().* = instance.*;
        }

        /// `append` of an instance whose Model is staged row `row` already.
        pub fn appendRow(self: *Self, row: u32, instance: *const D.Instance, nodes: [n_u]u32) !void {
            comptime assert(shares);
            assert(row < self.models.items.len);
            try self.model_ids.ensureUnusedCapacity(staging_gpa, 1);
            try self.instances.ensureUnusedCapacity(staging_gpa, 1);
            try self.nodes.ensureUnusedCapacity(staging_gpa, 1);
            self.model_ids.appendAssumeCapacity(row);
            self.nodes.appendAssumeCapacity(nodes);
            self.instances.addOneAssumeCapacity().* = instance.*;
        }

        /// `Proto.pattern`: adds every (row, col) the device's structural
        /// Jacobian can fill, ground excluded. Reserving only the structural
        /// entries (25 of 64 for mos1) keeps fill-in out of every
        /// factorization.
        pub fn addPattern(ctx: *anyopaque, gpa: std.mem.Allocator, pb: *PatternBuilder) abi.DeviceResult(void) {
            return abi.DeviceResult(void).fromLocal(addPatternLocal(ctx, gpa, pb));
        }

        fn addPatternLocal(ctx: *anyopaque, gpa: std.mem.Allocator, pb: *PatternBuilder) error{OutOfMemory}!void {
            const self: *Self = @ptrCast(@alignCast(ctx));
            const pat = comptime layout.jacPattern(D);
            try pb.reserve(gpa, self.nodes.items.len * SlotMap(D).k);
            // Runtime loops: this runs once per batch, and unrolling n_u^2
            // for every device only costs comptime quota.
            for (self.nodes.items) |nd| {
                for (0..n_u) |ru| for (0..n_u) |cu| {
                    if (!layout.has(pat[ru], cu)) continue;
                    if (nd[ru] != GROUND and nd[cu] != GROUND)
                        try pb.add(gpa, nd[ru], nd[cu]);
                };
            }
        }

        /// `Proto.finalize`: copies the staged rows into a new batch owned by
        /// `gpa`, builds its tapes against `pv`, and frees the staging rows.
        /// Fails with `TooManyInstances` past `maxInt(u32)` rows. On
        /// `OutOfMemory` nothing of `gpa` is left allocated, but the staging
        /// columns already moved are gone, so the store is only fit for
        /// `destroy`. Asserts that `pv` holds every non-ground (row, col)
        /// the device's structural Jacobian can fill and that `pv.col_ptr`
        /// has `pv.n + 1` entries.
        pub fn finalize(ctx: *anyopaque, gpa: std.mem.Allocator, pv: PatternView) abi.DeviceResult(Batch) {
            return abi.DeviceResult(Batch).fromLocal(finalizeLocal(ctx, gpa, pv));
        }

        fn finalizeLocal(ctx: *anyopaque, gpa: std.mem.Allocator, pv: PatternView) error{ OutOfMemory, TooManyInstances }!Batch {
            const has_q = @hasDecl(D, "q");
            const self: *Self = @ptrCast(@alignCast(ctx));
            // The count guard runs first: a row count past u32 fails before any
            // column or the pattern is read.
            const count = std.math.cast(u32, self.nodes.items.len) orelse return error.TooManyInstances;
            // `append` grows the per-instance columns together.
            assert((if (shares) self.model_ids.items.len else self.models.items.len) == self.nodes.items.len);
            assert(self.instances.items.len == self.nodes.items.len);
            assert(pv.col_ptr.len == @as(usize, pv.n) + 1);
            const store = try gpa.create(DeviceBatch(D));

            // `create` returns undefined bytes, so every field is written here,
            // the ones with struct defaults included.
            store.count = count;
            store.sim = .{};
            store.held_initial = false;
            store.committed_initial = false;
            store.owns_tapes = true;
            store.unshared = false;
            store.cowed = false;
            store.models_shared = &.{};
            store.models = &.{};
            store.model_of = &.{};
            store.instances = &.{};
            store.gath = &.{};
            store.rhs_idx = &.{};
            store.slots = &.{};
            store.original_ids = &.{};
            store.bypass = null;
            if (comptime has_q) store.q_tape = &.{};
            if (comptime traits.hasSpiceBreaks(D)) store.src_brk = &.{};
            if (comptime @hasDecl(D, "limit")) store.lim_x = &.{};
            if (comptime @hasDecl(D, "State")) store.states = &.{};
            errdefer DeviceBatch(D).hooks.deinit(store, gpa);
            // The staged rows can only be reordered before the copies below.
            if (comptime layout.canNarrow(D)) store.narrow_count = try self.partitionCollapsed(gpa, &store.original_ids);

            // Each column moves to `gpa` at exact size and its staging copy
            // dies right after, so at most one column exists twice.
            store.model_of = try gpa.alloc(u32, count);
            if (comptime shares) {
                // Rows numbered by first use in instance order, as a dedup
                // over one row per instance numbered them.
                const remap = try staging_gpa.alloc(u32, self.models.items.len);
                defer staging_gpa.free(remap);
                @memset(remap, std.math.maxInt(u32));
                var n: u32 = 0;
                for (self.model_ids.items, store.model_of) |old, *row| {
                    if (remap[old] == std.math.maxInt(u32)) {
                        remap[old] = n;
                        n += 1;
                    }
                    row.* = remap[old];
                }
                store.models = try gpa.alloc(D.Model, n);
                for (self.models.items, remap) |*m, to| if (to != std.math.maxInt(u32)) @memcpy(std.mem.asBytes(&store.models[to]), std.mem.asBytes(m));
                self.model_ids.deinit(staging_gpa);
                self.model_ids = .empty;
                self.seen.deinit(staging_gpa);
                self.seen = .empty;
            } else {
                for (store.model_of, 0..) |*row, id| row.* = @intCast(id);
                store.models = try gpa.dupe(D.Model, self.models.items);
            }
            self.models.deinit(staging_gpa);
            self.models = .empty;
            if (comptime @hasDecl(D, "nextBreakpoint") and traits.skipsTimerState(D)) store.bp = .{ .lo = std.math.inf(f64), .hi = std.math.inf(f64) };
            if (comptime traits.hasSpiceBreaks(D)) {
                store.src_brk = try gpa.alloc(SrcBrk, count);
                @memset(store.src_brk, .{});
            }
            if (comptime @hasDecl(D, "limit")) {
                store.lim_x = try gpa.alloc(f64, count * LimMap(D).k);
                @memset(store.lim_x, 0);
                store.lim_active = false;
            }
            store.instances = try gpa.dupe(D.Instance, self.instances.items);
            self.instances.deinit(staging_gpa);
            self.instances = .empty;

            store.gath = try gpa.alloc(u32, count * n_u);
            store.rhs_idx = try gpa.alloc(u32, count * n_u);
            store.slots = try gpa.alloc(u32, count * SlotMap(D).k);
            const flat_nodes = @as([*]const u32, @ptrCast(self.nodes.items.ptr))[0 .. count * n_u];
            layout.buildTapes(Mask(n_u), flat_nodes, n_u, &layout.jacPattern(D), pv, store.gath, store.rhs_idx, store.slots);
            if (comptime has_q) {
                store.q_tape = try gpa.alloc(f64, count * comptime layout.lteSites(D).len);
                @memset(store.q_tape, 0);
            }
            self.nodes.deinit(staging_gpa);
            self.nodes = .empty;

            // `setup` before `initState`: both read the card, and only the
            // former fills the row's `Model.su` (VerA device ABI 6).
            store.setupRows();
            if (comptime @hasDecl(D, "State")) {
                store.states = try gpa.alloc(D.State, count);
                for (0..count) |i| store.states[i] = D.initState(store.model(i), &store.instances[i]);
                store.commitBirth();
            }
            if (comptime @hasDecl(D, "precompute")) {
                for (0..count) |i| D.precompute(&store.instances[i], store.model(i));
            }

            // Shared rows never outnumber the instances, and any instance has one.
            assert(store.models.len <= count);
            if (count > 0) assert(store.models.len > 0);
            assert(store.model_of.len == count);
            return DeviceBatch(D).binding(store);
        }

        /// Rewrites `m` field by field over zeroed bytes.
        fn canonicalize(m: *D.Model) void {
            @setEvalBranchQuota(10 * @typeInfo(D.Model).@"struct".field_names.len + 1000);
            const copy = m.*;
            @memset(std.mem.asBytes(m), 0);
            inline for (@typeInfo(D.Model).@"struct".field_names) |name| @field(m, name) = @field(copy, name);
        }

        /// Stable-partitions the rows so maximally collapsed instances come
        /// first, and returns their count; the narrow basis is a comptime
        /// width and needs a uniform range. A uniform batch is left in place,
        /// so instance order (and slot summation order) only changes in a
        /// genuinely mixed batch.
        fn partitionCollapsed(self: *Self, gpa: std.mem.Allocator, original_ids: *[]u32) !u32 {
            const len = self.nodes.items.len;
            const flags = try staging_gpa.alloc(bool, len);
            defer staging_gpa.free(flags);
            var n: usize = 0;
            // `collapse` is pure in (Model, Instance): an instance that reads
            // the row an earlier one read, with equal Instance bytes, takes
            // that one's answer (a flattened deck repeats a few cells).
            const first = try staging_gpa.alloc(u32, if (shares) self.models.items.len else 0);
            defer staging_gpa.free(first);
            @memset(first, std.math.maxInt(u32));
            for (self.instances.items, flags, 0..) |*i, *f, k| {
                if (comptime shares) {
                    const seen = &first[self.model_ids.items[k]];
                    if (seen.* != std.math.maxInt(u32) and std.mem.eql(u8, std.mem.asBytes(&self.instances.items[seen.*]), std.mem.asBytes(i))) {
                        f.* = flags[seen.*];
                        if (f.*) n += 1;
                        continue;
                    }
                    if (seen.* == std.math.maxInt(u32)) seen.* = @intCast(k);
                }
                f.* = std.meta.eql(D.collapse(Real, self.stagedModel(k), i), D.collapse_full);
                if (f.*) n += 1;
            }
            if (n == 0 or n == len) return @intCast(n);
            // Row k moves to dst[k]; each column is permuted in place, so no
            // row is ever held twice.
            const dst = try staging_gpa.alloc(u32, len);
            defer staging_gpa.free(dst);
            var lo: u32 = 0;
            var hi: u32 = @intCast(n);
            for (flags, dst) |f, *d| {
                const next = if (f) &lo else &hi;
                d.* = next.*;
                next.* += 1;
            }
            // Allocate before mutating any column. Uniform batches need no
            // mapping; mixed batches retain the source identity once, shared
            // immutably by query clones alongside the other tapes.
            original_ids.* = try gpa.alloc(u32, len);
            for (dst, 0..) |d, id| original_ids.*[d] = @intCast(id);
            if (comptime shares) permuteInPlace(u32, self.model_ids.items, dst, flags) else permuteInPlace(D.Model, self.models.items, dst, flags);
            permuteInPlace(D.Instance, self.instances.items, dst, flags);
            permuteInPlace([n_u]u32, self.nodes.items, dst, flags);
            return @intCast(n);
        }

        /// `Proto.apply_perm`: renumbers staged nodes through `perm`; nodes
        /// past its end are left alone.
        pub fn applyPerm(ctx: *anyopaque, perm: []const u32) void {
            const self: *Self = @ptrCast(@alignCast(ctx));
            for (self.nodes.items) |*nd| {
                inline for (0..n_u) |u| {
                    if (nd[u] < perm.len) nd[u] = perm[nd[u]];
                }
            }
        }

        /// `Proto.destroy`: frees the staging rows and the store itself.
        pub fn destroy(ctx: *anyopaque, gpa: std.mem.Allocator) void {
            const self: *Self = @ptrCast(@alignCast(ctx));
            self.models.deinit(staging_gpa);
            self.model_ids.deinit(staging_gpa);
            self.seen.deinit(staging_gpa);
            self.instances.deinit(staging_gpa);
            self.nodes.deinit(staging_gpa);
            gpa.destroy(self);
        }
    };
}
