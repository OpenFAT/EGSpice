//! Netlist hypergraph to frozen `device.Circuit`: the builder module root.
//! In: a parsed `Netlist`. Out: a frozen `Circuit` and the deck tables.
//! `NetBuilder` maps nets to rows and turns each card into device instances;
//! `Builder` accumulates them per `device.Library` type, computes the BBD
//! node permutation and freezes the pattern.
//!
//! Cites ngspice's setup, load and parameter sources (file:line at each use:
//! vsrcload.c, vsrctemp.c, resparam.c, asrcset.c, b4set.c, ...) for what a
//! card means, the HSPICE manuals ([SA], [SI]) for the HSPICE elements, and
//! the Verilog-A LRM (§) for the `$simparam`, `derive` and `$port_connected`
//! hooks the generated models expose.
//!
//! Contents:
//! - `pub const spice`
//! - `pub const NetBuilder`
//! - `pub const test_access`
//! - `pub const Builder`
//! - `pub fn init(gpa: std.mem.Allocator, lib: *const Library) !Builder`
//! - `pub fn deinit(self: *Builder) void`
//! - `pub fn addNode(self: *Builder) !u32`
//! - `pub fn addDevice(self: *Builder, comptime D: type, card: []const u8, model: *const D.Model, instance: *const D.Instance, nodes: anytype) !void`
//! - `pub fn protoOf(self: *Builder, t: DeviceType) !Proto`
//! - `pub fn compile(self: *Builder) !Circuit`
//! - `pub fn compilePerm(self: *Builder, perm_out: *?[]const u32) !Circuit`

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
const net_cards = @import("net_cards.zig");
const net_waves = @import("net_waves.zig");
const net_filters = @import("net_filters.zig");
const net_tape = @import("net_tape.zig");
const net_devices = @import("net_devices.zig");
const net_sources = @import("net_sources.zig");
const net_passive = @import("net_passive.zig");
const net_lines = @import("net_lines.zig");
const net_topology = @import("net_topology.zig");
const net_dyn = @import("net_dyn.zig");
const net_publish = @import("net_publish.zig");
const net_card = @import("net_card.zig");
const net_branches = @import("net_branches.zig");

/// SPICE letter/LEVEL dispatch, re-exported for the frontend tests.
pub const spice = devices;
/// The card pass (net_builder.zig).
pub const NetBuilder = net_builder.NetBuilder;

/// Private helpers the frontend tests reach; empty outside tests.
pub const test_access = if (@import("builtin").is_test) .{
    .applySourceWaveform = net_waves.applySourceWaveform,
    .pwlSlot = net_waves.pwlSlot,
    .pwlCapacity = net_waves.pwlCapacity,
    .Wave = net_waves.Wave,
    .bindKv = net_cards.bindKv,
    .waveKind = net_waves.waveKind,
    .sourceDc = net_cards.sourceDc,
    .sourceAc = net_cards.sourceAc,
    .sourceDisto = net_cards.sourceDisto,
    .sourcePort = net_cards.sourcePort,
    .isPortCard = net_cards.isPortCard,
    .rootProduct = net_filters.rootProduct,
    .cplVector = net_cards.cplVector,
    .branchRow = net_branches.branchRow,
    .vRow = net_branches.vRow,
    .lRow = net_branches.lRow,
    .indexBranchNames = net_branches.indexBranchNames,
    .tagNodeInstance = Builder.tagNodeInstance,
    .computeBbd = Builder.computeBbd,
} else {};

/// Mutable circuit under construction: rows, device protos and card
/// identities. `compilePerm` consumes it into a `Circuit`.
pub const Builder = struct {
    /// Allocates the protos, the BBD permutation and the frozen Circuit.
    gpa: std.mem.Allocator,
    /// Device types and their vtables; outlives the Builder.
    lib: *const Library,
    /// Row count, ground included.
    n: u32,
    /// Per row, the net name it came from ("" for internal unknowns).
    /// Borrowed: copied into the circuit's intern table at the freeze.
    node_labels: std.ArrayList([]const u8),
    /// One proto (future batch) per device type, in first-use order.
    protos: std.ArrayList(Proto),
    /// Library type of each proto, parallel to `protos`.
    proto_types: std.ArrayList(DeviceType) = .empty,
    /// Per-row subcircuit instance: 0 is top level, MULTI_INSTANCE a
    /// coupling node. Empty when the deck has no subcircuits.
    node_instance: std.ArrayList(u32) = .empty,
    /// Some node has no DC path, so the operating point needs the transient fallback.
    needs_tran_op: bool = false,
    /// An instance calls a `$systf` through a loaded VPI application.
    calls_vpi: bool = false,
    /// `.options tnom` in degrees Celsius, ngspice's `CKTnomTemp`
    /// (cktsopt.c:71-73, default 27 degC from cktntask.c:127): the temperature
    /// a model card without its own TNOM/TREF was extracted at. `deriveModel`
    /// copies it into the models that read it.
    nom_temp_c: f64 = 27.0,
    /// The deck's `.options`: its reltol, abstol (A), vntol (V) and gmin (S)
    /// are §9.15 `$simparam` values, copied like `nom_temp_c`. A model reads
    /// this gmin outside the operating point's gmin-stepping rung, which
    /// writes its own.
    tol: numerics.Tolerances = .{},

    /// (device type, instance ordinal) to card name, for `.sens` columns: a
    /// `ParamRef` only knows its ordinal (`resistor#0`). Names borrow the
    /// parse arena; copy them before it dies.
    cards: std.ArrayList(requests.CardRef) = .empty,
    /// Per-device-type instance count, the ordinal `ParamRef.index` carries.
    /// Counted here because a generated device's store lives behind its
    /// vtable. Every add counts, card or not, so it stays in step with the store.
    card_counts: std.ArrayList(u32) = .empty,

    /// An empty circuit with only the ground row. `lib` must outlive the Builder.
    pub fn init(gpa: std.mem.Allocator, lib: *const Library) !Builder {
        var labels: std.ArrayList([]const u8) = .empty;
        try labels.append(staging, "0");
        return .{
            .gpa = gpa,
            .lib = lib,
            .n = 1,
            .node_labels = labels,
            .protos = .empty,
        };
    }

    /// Frees the protos and tables. Only for a Builder that was never compiled.
    pub fn deinit(self: *Builder) void {
        for (self.protos.items) |p| p.destroy(p.ctx, self.gpa);
        self.deinitStorage();
    }

    inline fn deinitStorage(self: *Builder) void {
        self.protos.deinit(staging);
        self.proto_types.deinit(staging);
        self.cards.deinit(staging);
        self.card_counts.deinit(staging);
        self.node_labels.deinit(staging);
        self.node_instance.deinit(staging);
        self.* = undefined;
    }

    /// Tags a node with a subcircuit instance; a second instance makes it a
    /// coupling node.
    fn tagNodeInstance(self: *Builder, node: u32, subckt_instance: u32) !void {
        if (node == GROUND) return;
        if (self.node_instance.items.len <= node) {
            const grow = node + 1 - self.node_instance.items.len;
            try self.node_instance.appendNTimes(staging, 0, grow);
        }
        const cur = &self.node_instance.items[node];
        if (subckt_instance != 0 and cur.* != subckt_instance) cur.* = if (cur.* == 0) subckt_instance else MULTI_INSTANCE;
    }

    /// The BBD permutation (old row to new row) and its blocks, both in `gpa`.
    const Bbd = struct { perm: []u32, info: numerics.BbdInfo };

    /// Null when fewer than two subcircuit instances own a node: one block
    /// orders nothing.
    fn computeBbd(self: *Builder) !?Bbd {
        if (self.node_instance.items.len == 0) return null;

        const gpa = self.gpa;
        const n: u32 = self.n;
        const ni = self.node_instance.items;

        // One block per instance, in first-seen node order. `at` counts the
        // block's nodes here and becomes its write cursor below.
        var instance_list: std.ArrayList(struct { inst: u32, at: u32 }) = .empty;
        defer instance_list.deinit(staging);
        const tagged = @min(n, @as(u32, @intCast(ni.len)));
        std.debug.assert(tagged >= 1);
        // Instance to block index, `unseen` until its first node. Instance
        // ordinals count from 1 (`Reader.instances`), so the table runs to
        // the largest one.
        const unseen = std.math.maxInt(u32);
        var max_inst: u32 = 0;
        for (ni[1..tagged]) |inst| if (inst != MULTI_INSTANCE) {
            max_inst = @max(max_inst, inst);
        };
        const index = try staging.alloc(u32, @as(usize, max_inst) + 1);
        defer staging.free(index);
        @memset(index, unseen);
        for (ni[1..tagged]) |inst| {
            if (inst == 0 or inst == MULTI_INSTANCE) continue;
            if (index[inst] == unseen) {
                index[inst] = @intCast(instance_list.items.len);
                try instance_list.append(staging, .{ .inst = inst, .at = 0 });
            }
            instance_list.items[index[inst]].at += 1;
        }
        std.debug.assert(index[0] == unseen);

        if (instance_list.items.len < 2) return null;

        // Internal nodes grouped by instance, then the coupling block; ground stays 0.
        const perm = try gpa.alloc(u32, n);
        errdefer gpa.free(perm);
        perm[0] = 0;
        var pos: u32 = 1;

        // Prefix-sum the counts into block starts, then scatter every tagged
        // node in one ascending pass.
        const blocks = try gpa.alloc(numerics.BbdBlock, instance_list.items.len);
        for (instance_list.items, blocks) |*entry, *blk| {
            blk.* = .{ .start = pos, .size = entry.at };
            entry.at = pos;
            pos += blk.size;
        }
        for (1..tagged) |i| {
            if (ni[i] == 0 or ni[i] == MULTI_INSTANCE) continue;
            const cursor = &instance_list.items[index[ni[i]]].at;
            perm[i] = cursor.*;
            cursor.* += 1;
        }

        // Coupling, top-level and untagged nodes.
        const coupling_start = pos;
        for (1..n) |i| {
            if (i >= ni.len or ni[i] == 0 or ni[i] == MULTI_INSTANCE) {
                perm[i] = pos;
                pos += 1;
            }
        }

        return .{
            .perm = perm,
            .info = .{
                .blocks = blocks,
                .coupling_start = coupling_start,
                .coupling_size = pos - coupling_start,
            },
        };
    }

    /// Appends an unlabelled row and returns it; TooManyNodes past the
    /// u32 row space.
    pub fn addNode(self: *Builder) !u32 {
        if (self.n == std.math.maxInt(u32)) return error.TooManyNodes;
        const id = self.n;
        try self.node_labels.append(staging, "");
        self.n += 1;
        return id;
    }

    /// Reserves label capacity for `expected` rows past ground.
    fn reserveNodes(self: *Builder, expected: u32) !void {
        if (expected == std.math.maxInt(u32)) return error.TooManyNodes;
        try self.node_labels.ensureTotalCapacity(staging, expected + 1);
    }

    /// Adds one instance of built-in `D` on `nodes` (one row per port),
    /// attributed to netlist card `card` in `cards`; "" records no card.
    /// `card` is borrowed like the card table. `model` and `instance` are
    /// read through pointers: a delay line's Instance carries its history
    /// inline (wline_4: 2 MB). Instances of one type share a
    /// batch whatever the call order. Internal unknowns get fresh rows here,
    /// in `D.U` order, unless `collapse` maps them onto a port.
    pub fn addDevice(
        self: *Builder,
        comptime D: type,
        card: []const u8,
        model: *const D.Model,
        instance: *const D.Instance,
        nodes: anytype,
    ) !void {
        const n_u = comptime @typeInfo(D.U).@"enum".field_names.len;
        var all: [n_u]u32 = undefined;
        inline for (0..D.num_ports) |p| all[p] = nodes[p];

        // Reach the device through its own object's vtable: naming
        // `D.collapse` or `ProtoStore(D)` here would compile the device body
        // into the executable a second time (device/eval.zig).
        const t = comptime Library.builtin(device.modelName(D) orelse
            @compileError(@typeName(D) ++ " is not a catalog device"));
        const vt = self.lib.vtable(t);
        try self.countCard(t, card);
        if (comptime n_u > D.num_ports) {
            var col: [n_u]i32 = @splat(-1);
            if (vt.collapse) |collapse| collapse(@ptrCast(model), @ptrCast(instance), &col);
            for (D.num_ports..n_u) |u|
                all[u] = if (col[u] >= 0) all[@intCast(col[u])] else try self.addNode();
        }
        const proto = try self.protoOf(t);
        try vt.proto_add(proto.ctx, self.gpa, @ptrCast(model), @ptrCast(instance), &all).unwrap();
    }

    /// `addDevice` of an instance whose Model is row `row` of its type's
    /// store and whose collapse map is `col`, both what an earlier
    /// `addDevice` with the same card inputs gave (`NetBuilder.memo`).
    fn addDeviceRow(
        self: *Builder,
        comptime D: type,
        card: []const u8,
        row: u32,
        instance: *const D.Instance,
        col: []const i32,
        nodes: anytype,
    ) !void {
        const n_u = comptime @typeInfo(D.U).@"enum".field_names.len;
        std.debug.assert(col.len == n_u);
        var all: [n_u]u32 = undefined;
        inline for (0..D.num_ports) |p| all[p] = nodes[p];
        const t = comptime Library.builtin(device.modelName(D).?);
        const vt = self.lib.vtable(t);
        try self.countCard(t, card);
        for (D.num_ports..n_u) |u|
            all[u] = if (col[u] >= 0) all[@intCast(col[u])] else try self.addNode();
        const proto = try self.protoOf(t);
        try vt.proto_add_row.?(proto.ctx, row, @ptrCast(instance), &all).unwrap();
    }

    /// Counts one instance of `t` and records its card name, if any.
    fn countCard(self: *Builder, t: DeviceType, card: []const u8) !void {
        const i = @backingInt(t);
        if (self.card_counts.items.len <= i)
            try self.card_counts.appendNTimes(staging, 0, i + 1 - self.card_counts.items.len);
        const ordinal = &self.card_counts.items[i];
        if (card.len != 0) try self.cards.append(staging, .{ .type = t, .index = ordinal.*, .name = card });
        ordinal.* += 1;
    }

    /// Returns the proto of device type `t`, creating it on first use.
    pub fn protoOf(self: *Builder, t: DeviceType) !Proto {
        for (self.protos.items, self.proto_types.items) |p, pt| {
            if (pt == t) return p;
        }
        try self.protos.ensureUnusedCapacity(staging, 1);
        try self.proto_types.ensureUnusedCapacity(staging, 1);
        const p = try self.lib.vtable(t).proto_create(self.gpa).unwrap();
        self.protos.appendAssumeCapacity(p);
        self.proto_types.appendAssumeCapacity(t);
        return p;
    }

    /// `compilePerm` for callers that recorded no node or branch row; the
    /// permutation is freed here. Consumes the Builder on success.
    pub fn compile(self: *Builder) !Circuit {
        const gpa = self.gpa;
        var perm: ?[]const u32 = null;
        const circuit = try self.compilePerm(&perm);
        if (perm) |p| gpa.free(p);
        return circuit;
    }

    /// Freezes into a Circuit, consuming the Builder. Protos and labels are
    /// renumbered by the BBD permutation (old row to frozen row), returned in
    /// `perm_out` (allocated with the Builder's allocator), or null when there
    /// is none, and on failure. Any row the caller recorded before the call
    /// must be mapped through it.
    pub fn compilePerm(self: *Builder, perm_out: *?[]const u32) !Circuit {
        const gpa = self.gpa;
        perm_out.* = null;
        const n: usize = self.n;

        const bbd = try self.computeBbd();
        errdefer if (bbd) |o| {
            gpa.free(o.perm);
            gpa.free(o.info.blocks);
        };
        if (bbd) |o| {
            const perm = o.perm;
            for (self.protos.items) |p| p.apply_perm(p.ctx, perm);
            const old_labels = try staging.alloc([]const u8, n);
            defer staging.free(old_labels);
            @memcpy(old_labels, self.node_labels.items);
            for (old_labels, perm) |label, new_i| self.node_labels.items[new_i] = label;
        }

        // Frozen intern table: one byte blob and n+1 offsets.
        const labels = self.node_labels.items;
        var total: usize = 0;
        for (labels) |label| total += label.len;
        const intern_bytes = try gpa.alloc(u8, total);
        errdefer gpa.free(intern_bytes);
        const intern_offs = try gpa.alloc(u32, n + 1);
        errdefer gpa.free(intern_offs);
        var off: u32 = 0;
        for (labels, 0..) |label, i| {
            intern_offs[i] = off;
            @memcpy(intern_bytes[off..][0..label.len], label);
            off += @intCast(label.len);
        }
        intern_offs[n] = off;

        const ckt = try Circuit.freeze(gpa, self.n, intern_bytes, intern_offs, self.protos.items, self.proto_types.items, if (bbd) |o| o.info else null, .{
            .needs_tran_op = self.needs_tran_op,
            .vpi = if (self.calls_vpi) self.lib.vpi else null,
            .digital = self.lib.digital.items,
        });

        self.deinitStorage(); // Circuit.freeze consumed the protos
        // Handed over only now, so the errdefer above frees it on any failure.
        perm_out.* = if (bbd) |o| o.perm else null;
        return ckt;
    }
};

/// `node_instance` tag of a node shared by two subcircuit instances.
const MULTI_INSTANCE: u32 = std.math.maxInt(u32);

/// Allocator for the Builder's own growing tables, which die at the freeze.
/// Not `Builder.gpa`, usually the Problem's arena: an arena keeps every
/// abandoned capacity alive for the run, 30 MB of card and label lists on
/// the 200,000-device RC ladder. Tests use the leak-checked
/// `std.testing.allocator`, so a Builder neither compiled nor deinited fails
/// its test.
/// ponytail: no allocation-failure sweep reaches these tables; a `staging`
/// field set by an `init` variant is the upgrade if one must.
const staging = if (@import("builtin").is_test) std.testing.allocator else core.gpa;

// ---- private ----

test {
    // spice.zig lives only in this module; its tests run from here (sparam and wfit run under wfit_mod).
    _ = devices;
    _ = net_builder;
    _ = net_branches;
    _ = net_card;
    _ = net_publish;
    _ = net_dyn;
    _ = net_topology;
    _ = net_lines;
    _ = net_passive;
    _ = net_sources;
    _ = net_devices;
    _ = net_tape;
    _ = net_filters;
    _ = net_waves;
    _ = net_cards;
}
