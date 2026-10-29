//! The parameter hooks of one device batch: sweep, Monte Carlo and
//! variant writes in, the batch's Model and Instance rows changed out,
//! sharing Model rows between instances wherever their writes agree.
//! Cites VerA device ABI 6 (temperature on the Model row) and `.options
//! tnom`'s `nom_temp__` for the fields that are not parameters.
//!
//! Contents:
//! - `pub fn Params(comptime D: type) type`, with `unshareModels`,
//!   `growModels`, `collectParams`, `collectParamsLocal`, `paramLayout`,
//!   `writeParams`, `fieldOf`, `writeParamsLocal`, `isolateRows`,
//!   `isolateRowsLocal`, `appendParams`, `paramField`; `RowWrite`'s
//!   `before` and `same`

const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;
const contract = @import("contract");
const abi = @import("device_abi");
const layout = @import("layout.zig");
const stamp = @import("stamp.zig");
const traits = @import("traits.zig");
const DeviceBatch = @import("batch.zig").DeviceBatch;
const staging_gpa = @import("proto_store.zig").staging_gpa;
const ParamRef = abi.ParamRef;

/// The parameter hooks of `DeviceBatch(D)`, each over a type-erased batch.
pub fn Params(comptime D: type) type {
    return struct {
        const Batch = DeviceBatch(D);
        const n_u: usize = contract.nU(D);
        const has_state = @hasDecl(D, "State");
        const has_q = @hasDecl(D, "q");
        const has_limit = @hasDecl(D, "limit");
        // Only timer-only models: their breakpoints are pure in (model, t).
        const has_bp = @hasDecl(D, "nextBreakpoint") and traits.skipsTimerState(D);
        const has_src_brk = traits.hasSpiceBreaks(D);
        const Ac = traits.AcTerms(D);

        /// Gives every instance its own Model row, so a write through one
        /// instance's `ParamRef` reaches that instance alone.
        pub fn unshareModels(self: *Batch) error{OutOfMemory}!void {
            if (self.ownsRows()) return;
            const own = try staging_gpa.alloc(D.Model, self.count);
            for (own, self.model_of, 0..) |*m, row, id| {
                m.* = self.models[row];
                self.model_of[id] = @intCast(id);
            }
            if (self.unshared) staging_gpa.free(self.models) else self.models_shared = self.models;
            self.models = own;
            self.unshared = true;
            self.cowed = false;
        }

        /// Appends `extra` rows to `models` (`staging_gpa`, as `unshareModels`).
        pub fn growModels(self: *Batch, extra: usize) error{OutOfMemory}!void {
            const len = self.models.len;
            if (self.unshared) {
                self.models = try staging_gpa.realloc(self.models, len + extra);
            } else {
                const own = try staging_gpa.alloc(D.Model, len + extra);
                @memcpy(own[0..len], self.models);
                self.models_shared = self.models;
                self.models = own;
                self.unshared = true;
            }
            self.cowed = true;
        }

        pub fn collectParams(ctx: *anyopaque, gpa: Allocator, list: *std.ArrayList(ParamRef)) abi.DeviceResult(void) {
            return abi.DeviceResult(void).fromLocal(collectParamsLocal(ctx, gpa, list));
        }

        pub fn collectParamsLocal(ctx: *anyopaque, gpa: Allocator, list: *std.ArrayList(ParamRef)) error{OutOfMemory}!void {
            // `paramField` runs per field; txl's history has ~10k fields.
            @setEvalBranchQuota(100_000 + 100 * (@typeInfo(D.Instance).@"struct".field_names.len + @typeInfo(D.Model).@"struct".field_names.len));
            const self = Batch.of(ctx);
            try unshareModels(self);
            try appendParams(D.Instance, self.instances, true, gpa, list);
            try appendParams(D.Model, self.models, false, gpa, list);
        }

        /// `appendParams`' fields for the Instance then the Model, with
        /// each field's byte offset in its struct.
        const param_table = blk: {
            @setEvalBranchQuota(100_000 + 100 * (@typeInfo(D.Instance).@"struct".field_names.len + @typeInfo(D.Model).@"struct".field_names.len));
            var fs: []const abi.ParamField = &.{};
            var os: []const u32 = &.{};
            for (.{ D.Instance, D.Model }, .{ true, false }) |T, is_instance| {
                var field_idx: usize = 0;
                for (@typeInfo(T).@"struct".field_names) |field_name| {
                    if (!paramField(T, field_name)) continue;
                    fs = fs ++ [_]abi.ParamField{.{ .name = field_name, .offset = @offsetOf(T, field_name), .is_instance = is_instance, .primary = !is_instance and field_idx == 0 }};
                    os = os ++ [_]u32{@offsetOf(T, field_name)};
                    field_idx += 1;
                }
            }
            const f = fs[0..fs.len].*;
            const o = os[0..os.len].*;
            break :blk struct {
                const fields: [f.len]abi.ParamField = f;
                const offsets: [o.len]u32 = o;
            };
        };

        pub fn paramLayout(ctx: *anyopaque) abi.DeviceResult(abi.ParamLayout) {
            const self = Batch.of(ctx);
            return .{ .ok = .{
                .count = @intCast(self.instances.len),
                .fields = &param_table.fields,
                .instances = @ptrCast(self.instances.ptr),
                .instance_stride = @sizeOf(D.Instance),
                .models = @ptrCast(self.models.ptr),
                .model_stride = @sizeOf(D.Model),
                .model_of = self.model_of.ptr,
            } };
        }

        pub fn writeParams(ctx: *anyopaque, refs: []const u32, values: []const f64, lo: u32, hi: u32) abi.DeviceResult(void) {
            return abi.DeviceResult(void).fromLocal(writeParamsLocal(ctx, refs, values, lo, hi));
        }

        /// One Model-field write through instance `id`.
        const RowWrite = struct {
            id: u32,
            offset: u32,
            value: f64,

            fn before(_: void, a: RowWrite, b: RowWrite) bool {
                return a.id < b.id or (a.id == b.id and a.offset < b.offset);
            }

            fn same(a: []const RowWrite, b: []const RowWrite) bool {
                if (a.len != b.len) return false;
                for (a, b) |x, y| if (x.offset != y.offset or @as(u64, @bitCast(x.value)) != @as(u64, @bitCast(y.value))) return false;
                return true;
            }
        };

        pub fn fieldOf(row: anytype, offset: u32) *f64 {
            return @ptrCast(@alignCast(@as([*]u8, @ptrCast(row)) + offset));
        }

        /// `Hooks.write_params`. A shared row is written in place when
        /// every instance reading it receives the same writes (a `.step`
        /// of a model parameter); otherwise each written instance gets a
        /// copy of the row, and the others keep it.
        pub fn writeParamsLocal(ctx: *anyopaque, refs: []const u32, values: []const f64, lo: u32, hi: u32) error{OutOfMemory}!void {
            if (comptime param_table.fields.len == 0) return;
            const self = Batch.of(ctx);
            const fields = &param_table.fields;
            const n = self.count;
            assert(hi - lo == n * fields.len);
            const owns = self.ownsRows();
            var shared: usize = 0;
            for (refs, values) |r, v| {
                if (r < lo or r >= hi) continue;
                const f = fields[(r - lo) / n];
                const id = (r - lo) % n;
                if (f.is_instance) {
                    fieldOf(&self.instances[id], f.offset).* = v;
                } else if (owns) {
                    fieldOf(self.model(id), f.offset).* = v;
                } else shared += 1;
            }
            if (shared == 0) return;
            const ws = try staging_gpa.alloc(RowWrite, shared);
            defer staging_gpa.free(ws);
            var k: usize = 0;
            for (refs, values) |r, v| {
                if (r < lo or r >= hi) continue;
                const f = fields[(r - lo) / n];
                if (f.is_instance) continue;
                ws[k] = .{ .id = (r - lo) % n, .offset = f.offset, .value = v };
                k += 1;
            }
            // Stable, so a repeated write keeps the last value.
            std.mem.sort(RowWrite, ws, {}, RowWrite.before);
            const Row = struct { first: u32 = 0, len: u32 = 0, written: u32 = 0, readers: u32 = 0, same: bool = true };
            const rows = try staging_gpa.alloc(Row, self.models.len);
            defer staging_gpa.free(rows);
            @memset(rows, .{});
            for (self.model_of) |row| rows[row].readers += 1;
            // Per row: how many of its readers are written, and whether
            // all with the same writes.
            var a: usize = 0;
            while (a < ws.len) {
                var b = a + 1;
                while (b < ws.len and ws[b].id == ws[a].id) b += 1;
                const row = &rows[self.model_of[ws[a].id]];
                if (row.written == 0) {
                    row.first = @intCast(a);
                    row.len = @intCast(b - a);
                } else row.same = row.same and RowWrite.same(ws[row.first..][0..row.len], ws[a..b]);
                row.written += 1;
                a = b;
            }
            var copies: usize = 0;
            for (rows) |row| {
                if (row.written != 0 and !(row.same and row.written == row.readers)) copies += row.written;
            }
            var next = self.models.len;
            if (copies != 0) try growModels(self, copies);
            a = 0;
            while (a < ws.len) {
                var b = a + 1;
                while (b < ws.len and ws[b].id == ws[a].id) b += 1;
                const id = ws[a].id;
                const row = rows[self.model_of[id]];
                if (row.same and row.written == row.readers) {
                    // The row's other readers write the same values.
                    if (row.first != a) {
                        a = b;
                        continue;
                    }
                } else {
                    self.models[next] = self.models[self.model_of[id]];
                    self.model_of[id] = @intCast(next);
                    next += 1;
                }
                for (ws[a..b]) |w| fieldOf(self.model(id), w.offset).* = w.value;
                a = b;
            }
            assert(next == self.models.len);
        }

        pub fn isolateRows(ctx: *anyopaque, ids: []const u32) abi.DeviceResult(void) {
            return abi.DeviceResult(void).fromLocal(isolateRowsLocal(ctx, ids));
        }

        /// `Hooks.isolate_rows`: every reader in `ids` of a row with other
        /// readers moves to a copy, until the row has one reader left.
        pub fn isolateRowsLocal(ctx: *anyopaque, ids: []const u32) error{OutOfMemory}!void {
            if (comptime param_table.fields.len == 0) return;
            const self = Batch.of(ctx);
            if (self.ownsRows()) return;
            const readers = try staging_gpa.alloc(u32, self.models.len);
            defer staging_gpa.free(readers);
            // Twice over the same counts: once to size the copies, once to
            // make them.
            var next = self.models.len;
            for (0..2) |pass| {
                @memset(readers, 0);
                for (self.model_of) |row| readers[row] += 1;
                var copies: usize = 0;
                for (ids, 0..) |id, k| {
                    assert(k == 0 or ids[k - 1] < id);
                    const row = self.model_of[id];
                    if (readers[row] < 2) continue;
                    readers[row] -= 1;
                    if (pass == 1) {
                        self.models[next] = self.models[row];
                        self.model_of[id] = @intCast(next);
                        next += 1;
                    }
                    copies += 1;
                }
                if (copies == 0) return;
                if (pass == 0) try growModels(self, copies);
            }
            assert(next == self.models.len);
        }

        pub fn appendParams(comptime T: type, items: anytype, comptime is_instance: bool, gpa: Allocator, list: *std.ArrayList(ParamRef)) error{OutOfMemory}!void {
            comptime var field_idx: usize = 0;
            inline for (@typeInfo(T).@"struct".field_names) |field_name| {
                if (comptime paramField(T, field_name)) {
                    const primary = !is_instance and field_idx == 0;
                    for (items, 0..) |*it, idx| {
                        try list.append(gpa, .{
                            .ptr = &@field(it, field_name),
                            .param_name = field_name,
                            .index = @intCast(idx),
                            .is_instance = is_instance,
                            .primary = primary,
                        });
                    }
                    field_idx += 1;
                }
            }
        }

        /// Whether field `field_name` of T is a sweepable parameter: an f64 with a
        /// finite default that is not a VerA-internal or runtime-state field.
        pub fn paramField(comptime T: type, comptime field_name: []const u8) bool {
            if (@FieldType(T, field_name) != f64) return false;
            // A trailing `__` is VerA's own namespace (no escaped Verilog-A
            // name ends in `_`), e.g. `nom_temp__` from `.options tnom`,
            // which `.mc`/`.sens` must not perturb. So is `<flow>__retained`,
            // the retention flag `derive` writes (contract `JacWhen`).
            if (comptime std.mem.endsWith(u8, field_name, "__") or
                std.mem.endsWith(u8, field_name, "__retained")) return false;
            if (T == D.Instance) {
                // A VerA Instance holds runtime state; only `mfactor` is a
                // parameter (the temperature is the Model row's
                // `temperature__`, VerA device ABI 6).
                if (!std.mem.eql(u8, field_name, "mfactor")) return false;
            }
            const dflt = @field(T{}, field_name);
            return dflt > -1e30 and dflt < 1e30;
        }
    };
}
