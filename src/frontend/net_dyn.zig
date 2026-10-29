//! Runtime-loaded (`.hdl`) devices. In: the cards whose first positional
//! names a Verilog-A module the Library loaded. Out: their instances, with
//! the module's parameters, ports and `$systf` hosts bound.
//! Cites: Verilog-A LRM 12.32 (VPI system tasks), §9.19 ($port_connected).
//!
//! Contents:
//! - `pub fn systfHostFor(lib: *const Library, t: DeviceType, instance: []const u8, vt: *const batch.DeviceVtable) error{UnregisteredSystf}!*const batch.SystfHost`
//! - `pub fn addDynDevices(self: *NetBuilder) !void`

const std = @import("std");
const core = @import("core");
const requests = core.query;
const numerics = core.numerics;
const devices = @import("spice.zig");
const sparam = @import("sparam.zig");
const wfit = @import("wfit.zig");
const device = @import("device");
const netlist = @import("netlist");
const Netlist = netlist.Netlist;
const Device = Netlist.View;
const Model = netlist.Model;
const Kv = netlist.Kv;
const Value = netlist.Value;
const Op = netlist.expr.Op;
const batch = device.abi;
const Library = device.Library;
const DeviceType = batch.DeviceType;
const castField = batch.bind.castField;
const markGiven = batch.bind.markGiven;

const GROUND = core.GROUND;
const Circuit = device.Circuit;
const Proto = batch.Proto;

const net_builder = @import("net_builder.zig");
const net_devices = @import("net_devices.zig");
const net_cards = @import("net_cards.zig");
const NetBuilder = net_builder.NetBuilder;
const builder = @import("builder.zig");
const Builder = builder.Builder;

/// The application an instance of type `t` calls its LRM 12.32 `$name`s
/// through: the loaded one (`Library.systfHost`), or `no_vpi_app` when
/// every call no application registered is a task, whose value nothing
/// reads (`Library.loadWith` warned for each). A call whose value is used
/// and that nothing registered is refused by name, `UnregisteredSystf`:
/// the LRM fixes no value for it. While VerA's `contract.Systf` does not
/// mark tasks (VerA#83), every unregistered call is refused.
pub fn systfHostFor(lib: *const Library, t: DeviceType, instance: []const u8, vt: *const batch.DeviceVtable) error{UnregisteredSystf}!*const batch.SystfHost {
    for (vt.systf_calls) |c| {
        if (lib.systfRegistered(c.name) or Library.isSystfTask(c)) continue;
        if (!@import("builtin").is_test) std.log.err("{s}: module '{s}' calls `{s}`: no VPI application registered (LRM 12.32), so its value is unknown; see #198", .{ instance, vt.name, c.name });
        return error.UnregisteredSystf;
    }
    return lib.systfHost(t) orelse &Library.no_vpi_app;
}

/// Adds the runtime-loaded (.hdl) devices, cards shaped `<name> node...
/// <model>`, through their vtables, after every built-in card.
/// UnknownParameter for a card key neither blob declares, WrongNodeCount
/// for more nodes than the module's ports. Omitted trailing ports float,
/// as ngspice leaves an OSDI card's (BSIM-CMG's `t`).
pub fn addDynDevices(self: *NetBuilder) !void {
    const b = self.b;
    const arena = self.arena;
    if (b.lib.names.items.len == Library.builtin_count) return;
    for (self.nl.order) |e| {
        const dev = self.nl.device(e);
        const t = net_builder.loadedType(b.lib, dev) orelse continue;
        const vt = b.lib.vtable(t);
        const systf_host = if (vt.systf_calls.len != 0) try systfHostFor(b.lib, t, dev.name, vt) else null;

        const mblob = try arena.alignedAlloc(u8, .@"16", vt.model_size);
        vt.init_model(mblob.ptr);
        if (dev.model) |m| try net_cards.bindKv(vt.bind_model, mblob.ptr, m.kv);
        // Card pairs override the .model card. VA parameters are Model
        // fields, so they go to the model blob too.
        try net_cards.bindKv(vt.bind_model, mblob.ptr, dev.kv);
        // The §9.15 `$simparam` host fields, as `deriveModel` writes them
        // for built-in devices; a model without them ignores the keys.
        var host: [net_devices.simparam_fields.len]batch.Param = undefined;
        for (net_devices.simparam_fields, net_devices.simparamValues(b), &host) |f, v, *h| h.* = .{ .key = f, .value = v };
        try vt.bind_model(mblob.ptr, &host).unwrap();
        // LRM 6.3.4/3.4.5: derived parameters after the last write, before
        // `collapse`/`proto_add` read the blob.
        if (vt.derive) |df| df(mblob.ptr);
        const iblob = try arena.alignedAlloc(u8, .@"16", vt.instance_size);
        vt.init_instance(iblob.ptr);
        // A key neither blob declares would be dropped, and the model
        // default used in its place. The binder rejects a null value for
        // a key it knows and ignores one it does not, so a null probe
        // tells the two apart without writing either blob.
        for (dev.kv) |item| {
            const probe = [_]batch.Param{.{ .key = item.key, .value = null }};
            if (vt.bind_model(mblob.ptr, &probe) == .ok and vt.bind_instance(iblob.ptr, &probe) == .ok) {
                if (!@import("builtin").is_test) std.log.err("{s}: module '{s}' has no parameter '{s}'", .{ dev.name, vt.name, item.key });
                return error.UnknownParameter;
            }
        }
        try net_cards.bindKv(vt.bind_instance, iblob.ptr, dev.kv);
        if (dev.pins.len == 0 or dev.pins.len > vt.num_ports) {
            if (!@import("builtin").is_test) std.log.err("{s}: module '{s}' has {d} ports, the card connects {d} nodes", .{ dev.name, vt.name, vt.num_ports, dev.pins.len });
            return error.WrongNodeCount;
        }

        // Same port/internal-node policy as Builder.addDevice.
        const nodes = try arena.alloc(u32, vt.n_u);
        for (dev.pins, 0..) |pin, p| nodes[p] = try self.rowOf(pin);
        for (dev.pins.len..vt.num_ports) |p| {
            nodes[p] = try b.addNode();
            if (p < vt.unknown_names.len)
                b.node_labels.items[nodes[p]] = try std.fmt.allocPrint(arena, "{s}#{s}", .{ dev.name, vt.unknown_names[p] });
        }
        if (vt.n_u > vt.num_ports) {
            const col = try arena.alloc(i32, vt.n_u);
            @memset(col, -1);
            if (vt.collapse) |cf| cf(mblob.ptr, iblob.ptr, col.ptr);
            for (vt.num_ports..vt.n_u) |u| {
                if (col[u] >= 0) {
                    nodes[u] = nodes[@intCast(col[u])];
                    continue;
                }
                nodes[u] = try b.addNode();
                // An internal net publishes as ngspice names an OSDI
                // device's: `v(<instance>#<net>)`. Branch flows are not
                // node voltages.
                if (u < vt.unknown_names.len and !std.mem.startsWith(u8, vt.unknown_names[u], "flowZ28"))
                    b.node_labels.items[nodes[u]] = try std.fmt.allocPrint(arena, "{s}#{s}", .{ dev.name, vt.unknown_names[u] });
            }
        }

        if (systf_host) |h| {
            vt.bind_systf.?(iblob.ptr, h);
            if (h != &Library.no_vpi_app) b.calls_vpi = true;
        }
        const proto = try b.protoOf(t);
        try vt.proto_add(proto.ctx, b.gpa, mblob.ptr, iblob.ptr, nodes.ptr).unwrap();
        // `%m` in a display task and a `$fatal` refusal name the
        // instance by its card, as for a built-in device.
        try b.countCard(t, dev.name);
    }
}

// ---- private ----
