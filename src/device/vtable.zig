//! A device type's construction entry points: the generated device `D` in,
//! its `DeviceVtable` out, either exported from a runtime `.so` under the
//! `arp_*` symbols or returned in-process. Also the host check every device
//! must pass. Cites LRM §12.32 for `$systf` binding and VerA's
//! `contract.validateHost`.
//!
//! Contents:
//! - `pub fn checkHost(comptime D: type) void`
//! - `pub fn exportDevice(comptime D: type, comptime device_name: []const u8) void`
//! - `pub fn deviceVtable(comptime D: type, comptime device_name: []const u8) *const DeviceVtable`
//! - the host facts `VpiHost` declares for `validateHost`: `iteration_hooks`,
//!   `calls_setup`, `mutable_eval`, `calls_ac_dyn`, `noise_table_points`, `systf`

const std = @import("std");
const contract = @import("contract");
const abi = @import("device_abi");
const dual = @import("dual.zig");
const ProtoStore = @import("proto_store.zig").ProtoStore;
const DeviceVtable = abi.DeviceVtable;
const Proto = abi.Proto;
const Real = dual.Real;

/// Compile error unless this host provides everything D requires. The check
/// lives here rather than in `contract.validate` because a device compiled to
/// a `.so` cannot know which simulator will load it.
pub fn checkHost(comptime D: type) void {
    contract.validateHost(VpiHost(D), D);
}

/// Exports D under the runtime device ABI (`arp_abi_version`,
/// `arp_layout_hash`, `arp_device`). The generated `.so` shim calls this:
/// `comptime { @import("dyn").exportDevice(@import("device"), "name"); }`.
pub fn exportDevice(comptime D: type, comptime device_name: []const u8) void {
    comptime checkHost(D);
    const impl = Impl(D, device_name);
    @export(&impl.abiVersion, .{ .name = "arp_abi_version" });
    @export(&impl.layoutHashC, .{ .name = "arp_layout_hash" });
    @export(&impl.getVtable, .{ .name = "arp_device" });
}

/// D's vtable without going through a shared library.
pub fn deviceVtable(comptime D: type, comptime device_name: []const u8) *const DeviceVtable {
    return &Impl(D, device_name).vtable;
}

// ---- private ----

/// What this simulator offers D, checked by `contract.validateHost`: Newton
/// iteration hooks, mutable evaluation, and the VPI application behind D's
/// `$systf` calls (LRM §12.32).
fn VpiHost(comptime D: type) type {
    return struct {
        pub const iteration_hooks = true;
        /// The batch runs `setup` once before `initState` (after `.options tnom`
        /// is on the card), and `reprep` reruns it after every parameter or
        /// temperature write.
        pub const calls_setup = true;
        pub const mutable_eval = true;
        /// Every small-signal matrix adds the device's `acDyn` terms
        /// (`Hooks.ac_dyn`).
        pub const calls_ac_dyn = true;
        /// A table row's knots come from its Model row
        /// (`contract.noiseTable`), so a card's §4.6.4.3 array overrides
        /// the declared defaults.
        pub const noise_table_points = true;
        /// Null: ESPice binds the application per instance instead, through
        /// `DeviceVtable.bind_systf`, because it is chosen at run time (a
        /// `--vpi` application, or a no-op for task calls only) and refused
        /// by name before staging when there is none (`DeviceVtable.systf_calls`).
        pub fn systf(_: *const D.Model) ?*const contract.SystfHost {
            return null;
        }
    };
}

fn Impl(comptime D: type, comptime device_name: []const u8) type {
    return struct {
        const Store = ProtoStore(D);
        const n_u: usize = contract.nU(D);

        fn abiVersion() callconv(.c) u32 {
            return abi.abi_version;
        }
        fn layoutHashC() callconv(.c) u64 {
            return abi.layoutHash();
        }
        fn getVtable() callconv(.c) *const DeviceVtable {
            return &vtable;
        }

        const vtable: DeviceVtable = .{
            .name = device_name,
            .n_u = n_u,
            .num_ports = D.num_ports,
            .model_size = @sizeOf(D.Model),
            .instance_size = @sizeOf(D.Instance),
            .init_model = initBlob(D.Model),
            .init_instance = initBlob(D.Instance),
            .bind_model = bindFn(D.Model),
            .bind_instance = bindFn(D.Instance),
            .derive = if (@hasDecl(D, "derive")) deriveFn else null,
            .collapse = if (@hasDecl(D, "collapse")) collapseFn else null,
            .proto_create = protoCreate,
            .proto_add = protoAdd,
            .proto_last_row = if (Store.shares) protoLastRow else null,
            .proto_add_row = if (Store.shares) protoAddRow else null,
            .unknown_names = &unknown_names,
            .systf_calls = if (has_systf) &D.systf_calls else &.{},
            .bind_systf = if (has_systf) bindSystf else null,
        };

        const has_systf = @hasDecl(D, "systf_calls") and D.systf_calls.len != 0;

        fn bindSystf(instance: [*]u8, host: *const contract.SystfHost) void {
            const i: *D.Instance = @ptrCast(@alignCast(instance));
            i.systf = host;
        }

        const unknown_names: [n_u][]const u8 = blk: {
            var out: [n_u][]const u8 = undefined;
            for (&out, @typeInfo(D.U).@"enum".field_names) |*o, f| o.* = f;
            break :blk out;
        };

        fn initBlob(comptime T: type) *const fn ([*]u8) void {
            return struct {
                fn f(dest: [*]u8) void {
                    const p: *T = @ptrCast(@alignCast(dest));
                    p.* = .{};
                }
            }.f;
        }

        fn bindFn(comptime T: type) *const fn ([*]u8, []const abi.Param) abi.BindStatus {
            return struct {
                fn f(dest: [*]u8, params: []const abi.Param) abi.BindStatus {
                    return abi.bind.apply(@as(*T, @ptrCast(@alignCast(dest))), params);
                }
            }.f;
        }

        fn deriveFn(model: [*]u8) void {
            const m: *D.Model = @ptrCast(@alignCast(model));
            D.derive(Real, m);
        }

        fn collapseFn(model: [*]const u8, instance: [*]const u8, out: [*]i32) void {
            const m: *const D.Model = @ptrCast(@alignCast(model));
            const i: *const D.Instance = @ptrCast(@alignCast(instance));
            const col = D.collapse(Real, m, i);
            inline for (D.num_ports..n_u) |u|
                out[u] = if (col[u]) |p| @intCast(p) else -1;
        }

        fn protoCreate(gpa: std.mem.Allocator) abi.DeviceResult(Proto) {
            const store = gpa.create(Store) catch return .out_of_memory;
            store.* = .{};
            return .{ .ok = .{
                .ctx = store,
                .type_name = device_name,
                .pattern = Store.addPattern,
                .finalize = Store.finalize,
                .destroy = Store.destroy,
                .apply_perm = Store.applyPerm,
            } };
        }

        // The rows live on `staging_gpa`, not the ABI's allocator.
        fn protoAdd(ctx: *anyopaque, _: std.mem.Allocator, model: [*]const u8, instance: [*]const u8, nodes: [*]const u32) abi.DeviceResult(void) {
            const store: *Store = @ptrCast(@alignCast(ctx));
            const m: *const D.Model = @ptrCast(@alignCast(model));
            const i: *const D.Instance = @ptrCast(@alignCast(instance));
            return abi.DeviceResult(void).fromLocal(store.append(m, i, nodes[0..n_u].*));
        }

        fn protoLastRow(ctx: *anyopaque) u32 {
            const store: *Store = @ptrCast(@alignCast(ctx));
            return store.model_ids.items[store.model_ids.items.len - 1];
        }

        fn protoAddRow(ctx: *anyopaque, row: u32, instance: [*]const u8, nodes: [*]const u32) abi.DeviceResult(void) {
            const store: *Store = @ptrCast(@alignCast(ctx));
            const i: *const D.Instance = @ptrCast(@alignCast(instance));
            return abi.DeviceResult(void).fromLocal(store.appendRow(row, i, nodes[0..n_u].*));
        }
    };
}
