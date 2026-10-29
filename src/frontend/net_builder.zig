//! Netlist to Builder: the card pass. In: a parsed `Netlist` and a
//! `Builder`. Out: every card's device instances on the Builder, plus the
//! source, branch, port and probe tables the deck's queries resolve
//! against (`publish`), on circuit rows after the freeze (`frozenRow`).
//! Cites: ngspice CKTsetup, vsrcload.c, vsrcacld.c, acan.c, cccsset.c.
//!
//! Contents:
//! - `pub const NetBuilder`
//! - `pub const VCard`
//! - `pub fn init(arena: std.mem.Allocator, b: *builder.Builder, nl: Netlist) !NetBuilder`
//! - `pub fn rowOf(self: *NetBuilder, v: netlist.VertexId) !u32`
//! - `pub fn frozenRow(self: *const NetBuilder, net: u32) u32`
//! - `pub fn tagSubcircuitNodes(self: *NetBuilder) !void`
//! - `pub const Published`
//! - `pub fn publish(self: *NetBuilder, arena: std.mem.Allocator, circuit: *const Circuit, perm: ?[]const u32) !Published`
//! - `pub fn build(self: *NetBuilder) !void`
//! - `pub fn loadedType(lib: *const Library, dev: Device) ?DeviceType`
//! - `pub fn setParam(comptime D: type, model: *D.Model, instance: *D.Instance, comptime field: []const u8, value: f64) !bool`

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

const net_branches = @import("net_branches.zig");
const net_card = @import("net_card.zig");
const net_publish = @import("net_publish.zig");
const net_dyn = @import("net_dyn.zig");
const net_topology = @import("net_topology.zig");
const net_cards = @import("net_cards.zig");
const builder = @import("builder.zig");
const Builder = builder.Builder;

/// Netlist to Builder: one pass over the cards, recording the source,
/// branch and port tables the deck's queries resolve against. Rows it
/// records are pre-permutation; `prepare.build` maps them after the freeze.
pub const NetBuilder = struct {
    /// Scratch for everything recorded here; dies after `prepare.build`.
    arena: std.mem.Allocator,
    /// The circuit under construction; every row recorded here is one of its.
    /// Dangles once `Builder.compile` consumes it: `publish` and `frozenRow`
    /// never read it.
    b: *builder.Builder,
    /// The parsed deck. Names recorded here borrow its arena.
    nl: Netlist,
    /// Circuit row of each net, 0 until a device touches it (ground is 0
    /// anyway). Rows follow stamping order, not net order.
    rows: []u32,
    /// Per control name, how many F/H/W cards sense it. The first V card of
    /// that name takes the count, as `resolveDeferred` binds the first.
    sensors: std.StringHashMapUnmanaged(u32),

    /// One row per V card.
    v: std.MultiArrayList(VCard) = .empty,
    /// One row per V card with a `portnum` (vsrcdefs.h:104-105), in card
    /// order: `num` is the 1-based `.sp` index. A P card's port node is n+
    /// in front of its z0, and a mixed-mode one's branch is its + leg's.
    /// Its own table, not a `VCard` column, as almost no V card is a port.
    ports: std.ArrayList(struct { num: u16, port: requests.Port }) = .empty,
    /// Branch-current probes other than V and L cards. ngspice gives every
    /// MNA branch unknown an `i(<card>)` column (vcvsset.c:41-46,
    /// ccvsset.c:41-46, asrcsetup.c:78-83 for a V-mode B); F, G, S and an
    /// I-mode B have no branch.
    br: std.MultiArrayList(struct { name: []const u8, row: u32 }) = .empty,
    /// `R2 2 0 5K ac=15k`: an AC-only resistance (ngspice restemp.c:112-118),
    /// keyed by card because instance ordinals only exist after the freeze.
    ac_res: std.MultiArrayList(struct { name: []const u8, value: f64 }) = .empty,
    /// One row per I card. An I source has no branch row, so a `.tf` driven
    /// from one excites the node pair.
    i: std.MultiArrayList(struct { name: []const u8, pos: u32, neg: u32 }) = .empty,
    /// One row per source with an `AC` spec; all rows drive one rhs (ngspice
    /// CKTacLoad, acan.c:471-490). A row subtracts at `pos` and adds at `neg`
    /// under Circuit.rhs's residual sign: an I card is (n+, n-), a V card only
    /// drives its branch row (vsrcacld.c:175), so (GROUND, branch).
    ac: std.MultiArrayList(struct { pos: u32, neg: u32, re: f64, im: f64 }) = .empty,
    /// One row per L card; `value` is the inductance, for a K card's M = k*sqrt(L1*L2).
    l: std.MultiArrayList(struct { name: []const u8, branch: u32, value: f64 }) = .empty,
    /// C and L `ic=` (ngspice CAPinitCond, INDinitCond) as rows for a `uic`
    /// start: a C's n+ row (n- negated when n+ is ground), an L's branch row.
    dev_ic: std.ArrayList(core.Ic) = .empty,
    /// F/H/W/K cards, added after every other card so V and L rows exist.
    deferred: std.ArrayList(Device) = .empty,
    /// Card name to its first branch row, V rows before L rows before `br`
    /// rows: a value `k` below `v.len` is V row `k`, then L row
    /// `k - v.len`, then `br` row `k - v.len - l.len`. `resolveDeferred`
    /// builds it once the V and L tables are complete; the first
    /// `branch_indexed` `br` rows are in it.
    branch_index: std.StringHashMapUnmanaged(u32) = .empty,
    branch_indexed: usize = 0,
    /// Per `Netlist.models` row, the Model its card binds to (`.{}`, the
    /// card's pairs, its polarity), bound on first use and copied per
    /// instance: a BSIM4 bin card is ~270 pairs against ~1000 fields.
    bound_cards: []?BoundCard,
    /// A device card's bound result by its inputs (`memoKey`): the Model
    /// row its type's store holds, its Instance and collapse map. A later
    /// card with the same inputs skips binding, `derive`, `collapse` and
    /// the store's row hash and compare (7.5).
    memo: std.StringHashMapUnmanaged(Memo) = .empty,
    memo_key: std.ArrayList(u8) = .empty,

    /// The first `.tran` card's TSTEP and TSTOP, ngspice's TRANinit values
    /// for `resolvePulseDefaults`; 1 ns and 1e30 s without one.
    tran_step: f64 = 1e-9,
    tran_stop: f64 = 1e30,

    /// First stamped V card's positive node and branch row, the `.op` ladder's anchor.
    source_node: u32 = GROUND,
    source_branch: u32 = GROUND,

    /// Topology check (ngspice CKTsetup-class), indexed by row; only rows of
    /// netlist nets are `seen`. `comp` is a union-find over the DC paths
    /// (every element but capacitors and current sources joins its pins), so
    /// a node whose component lacks ground has no DC path, however many
    /// elements touch it (#90). V and L cards are DC shorts: `uf`/`pot` form
    /// a weighted union-find with `pot` = v(x) - v(parent), and a loop of
    /// shorts is an error when its KVL sum is inconsistent (V1=5 parallel to
    /// V2=3) or when V sources alone close it (#70).
    topo: std.MultiArrayList(struct { seen: bool, comp: u32, uf: u32, pot: f64 }) = .empty,
    /// I cards as rows and DC value, for the current balance of each
    /// component without a DC path.
    topo_isrc: std.ArrayList(struct { pos: u32, neg: u32, dc: f64 }) = .empty,

    /// A V card as the queries and the F/H/W cards see it.
    pub const VCard = struct {
        name: []const u8,
        /// `pos`/`neg`: the node pair, which is the control port of an F/H/W sensing it.
        pos: u32,
        neg: u32,
        /// Branch-current row.
        branch: u32,
        /// A sensed source drives `pos` to `tap`, and its sensors' 0 V
        /// control branches chain from `tap` to `neg`, so each carries the
        /// source's whole current (ngspice cccsset.c reads that branch).
        /// `tap` is the next sensor's cp; `sensors` how many are still to come.
        tap: u32,
        sensors: u32,
        /// `DISTOF1 [mag [phase]]`, `.disto`'s F1 drive (ngspice
        /// cktdisto.c:100-117), phase in degrees. `{0, 0}` when not given.
        distof1: [2]f64,
        /// `DISTOF2 [mag [phase]]`, the F2 drive of a two-tone `.disto`.
        distof2: [2]f64,

        // A SoA row: 68 bytes over its columns (it held a 42-byte port too).
        comptime {
            if (@sizeOf(usize) == 8) std.debug.assert(@sizeOf(VCard) == 72);
        }
    };

    /// A builder over `nl` writing into `b`; both must outlive it.
    /// Reserves `b`'s label capacity for every net.
    pub fn init(arena: std.mem.Allocator, b: *builder.Builder, nl: Netlist) !NetBuilder {
        try b.reserveNodes(nl.graph.vertexCount());
        // The cards `resolveDeferred` hands to addBranchRef, and no others:
        // a sensor counted here and never added would leave its chain open.
        var sensors: std.StringHashMapUnmanaged(u32) = .empty;
        for ("fhw") |letter| for (nl.bucket(letter)) |e| {
            const d = nl.device(e);
            // An HSPICE W card is a transmission line, and an HDL card is not deferred.
            if (letter == 'w' and nl.deck.dialect == .hspice) continue;
            if (loadedType(b.lib, d) != null) continue;
            const name = Value.nameAt(d.positional, 0) orelse continue;
            const entry = try sensors.getOrPut(arena, name);
            entry.value_ptr.* = if (entry.found_existing) entry.value_ptr.* + 1 else 1;
        };
        const rows = try arena.alloc(u32, nl.graph.vertexCount());
        @memset(rows, 0);
        const cards = try arena.alloc(?BoundCard, nl.models.len);
        @memset(cards, null);
        var nb: NetBuilder = .{ .arena = arena, .b = b, .nl = nl, .rows = rows, .sensors = sensors, .bound_cards = cards };
        for (nl.deck.analyses) |dir| {
            if (dir.kind != .tran) continue;
            const a0 = Value.numberAt(dir.args, 0);
            const a1 = Value.numberAt(dir.args, 1);
            if (a1 orelse a0) |ts| nb.tran_stop = ts;
            // A malformed card (`.tran xyz 1u`) keeps the default step; the
            // query check rejects it after the build.
            nb.tran_step = if (a1 != null) a0 orelse nb.tran_step else nb.tran_stop / 100.0;
            break;
        }
        return nb;
    }

    /// Row of net `v`, allocated on first touch.
    pub fn rowOf(self: *NetBuilder, v: netlist.VertexId) !u32 {
        const row = &self.rows[v.index()];
        if (row.* != 0 or v == netlist.ground) return row.*;
        row.* = try self.b.addNode();
        self.b.node_labels.items[row.*] = self.nl.netName(v);
        return row.*;
    }

    /// Row of net index `net`, or `netlist.none` for a net no device touched
    /// and for `netlist.none` itself.
    pub fn frozenRow(self: *const NetBuilder, net: u32) u32 {
        if (net == netlist.none) return netlist.none;
        if (net == 0) return GROUND;
        const r = self.rows[net];
        return if (r == 0) netlist.none else r;
    }

    /// Tags every row a subcircuit device touches with its instance, for the
    /// BBD permutation.
    pub fn tagSubcircuitNodes(self: *NetBuilder) !void {
        const nl = &self.nl;
        for (nl.order) |e| {
            const dev = nl.device(e);
            if (dev.subckt_instance == 0) continue;
            for (dev.pins) |pin| {
                const r = self.rows[pin.index()];
                if (r != 0) try self.b.tagNodeInstance(r, dev.subckt_instance);
            }
        }
    }

    /// What the deck keeps of the cards, in frozen rows.
    pub const Published = struct {
        /// Source and `.sp` port tables, copied into the
        /// publish arena.
        bindings: core.QueryBindings,
        /// Branch currents first, then every named node; parallel to
        /// `probe_labels`. Mutable so `.save` can narrow them in place.
        probes: []u32,
        /// `i(card)` and `v(net)` names, parallel to `probes`.
        probe_labels: [][]const u8,
        /// `acExcitation` over the frozen rows: `[re(0..n), im(0..n)]`.
        ac_drive: []const f64,
        /// The first stamped V card's positive node, the `.op` ladder's anchor.
        source_node: u32,
        /// That card's branch row; GROUND when the deck stamps no V card.
        source_branch: u32,
        /// Row of the last net the deck introduces.
        output_node: u32,
        /// `dev_ic` over the frozen rows.
        uic_ic: []const core.Ic,
    };

    /// Maps every row recorded so far through `perm`, the BBD permutation
    /// `Builder.compilePerm` returned (null: none), then copies the deck's
    /// tables into `arena`. Call once, after the freeze: `frozenRow` answers
    /// in frozen rows from then on. `circuit` is the frozen result.
    /// DuplicatePortNumber or MissingPortNumber for a gapped `.sp` port set.
    pub fn publish(self: *NetBuilder, arena: std.mem.Allocator, circuit: *const Circuit, perm: ?[]const u32) !Published {
        if (perm) |p| for ([_][]u32{
            self.v.items(.branch),       self.v.items(.pos),    self.v.items(.neg),  self.i.items(.pos),  self.i.items(.neg),
            self.br.items(.row),         self.l.items(.branch), self.ac.items(.pos), self.ac.items(.neg), (&self.source_node)[0..1],
            (&self.source_branch)[0..1], self.rows,
        }) |rows| for (rows) |*row| {
            if (row.* < p.len) row.* = p[row.*];
        };
        if (perm) |p| for (self.dev_ic.items) |*ic| {
            if (ic.node < p.len) ic.node = p[ic.node];
        };
        // A port carries its own copy of the card's rows.
        if (perm) |p| for (self.ports.items) |*entry| {
            const port = &entry.port;
            for ([_]*u32{ &port.node, &port.neg, &port.branch }) |row| {
                if (row.* < p.len) row.* = p[row.*];
            }
            if (port.balanced) |*leg| for ([_]*u32{ &leg.node, &leg.branch }) |row| {
                if (row.* < p.len) row.* = p[row.*];
            };
        };

        // Probes: branch currents first, then every named node. ngspice gives every
        // MNA branch-current unknown an `i(<card>)` column (V, L, E, H, V-mode B);
        // F, G, S and I-mode B stamp no branch. Branch-first, unlike ngspice.
        // Named nodes and branch rows are disjoint, so circuit.n bounds the total.
        const probe_buf = try arena.alloc(u32, circuit.n);
        const label_buf = try arena.alloc([]const u8, circuit.n);
        var n_probes: u32 = 0;
        for ([_][]const []const u8{ self.v.items(.name), self.l.items(.name), self.br.items(.name) }, [_][]const u32{ self.v.items(.branch), self.l.items(.branch), self.br.items(.row) }) |names, rows| {
            for (names, rows) |name, br| {
                probe_buf[n_probes] = br;
                label_buf[n_probes] = try std.mem.concat(arena, u8, &.{ "i(", name, ")" });
                n_probes += 1;
            }
        }
        for (1..circuit.n) |i| {
            const label = circuit.nodeName(@intCast(i));
            if (label.len != 0) {
                probe_buf[n_probes] = @intCast(i);
                label_buf[n_probes] = try std.mem.concat(arena, u8, &.{ "v(", label, ")" });
                n_probes += 1;
            }
        }

        // The last probe will not do for `output_node`, since the BBD
        // permutation reorders node rows.
        var last_net: u32 = 0;
        for (self.rows, 0..) |row, net| if (row != GROUND) {
            last_net = @intCast(net);
        };

        return .{
            .bindings = .{
                .v_names = try net_cards.copyNames(arena, self.v.items(.name)),
                .i_names = try net_cards.copyNames(arena, self.i.items(.name)),
                .v_branches = try arena.dupe(u32, self.v.items(.branch)),
                .v_pos = try arena.dupe(u32, self.v.items(.pos)),
                .v_neg = try arena.dupe(u32, self.v.items(.neg)),
                .i_pos = try arena.dupe(u32, self.i.items(.pos)),
                .i_neg = try arena.dupe(u32, self.i.items(.neg)),
                .v_distof1 = try arena.dupe([2]f64, self.v.items(.distof1)),
                .v_distof2 = try arena.dupe([2]f64, self.v.items(.distof2)),
                .ports = try net_publish.portList(self, arena),
            },
            .probes = probe_buf[0..n_probes],
            .probe_labels = label_buf[0..n_probes],
            .ac_drive = try net_publish.acExcitation(self, arena, circuit.n),
            .source_node = self.source_node,
            .source_branch = self.source_branch,
            .output_node = self.frozenRow(last_net),
            .uic_ic = try arena.dupe(core.Ic, self.dev_ic.items),
        };
    }

    /// Adds every card to the Builder, ready for `Builder.compilePerm`:
    /// the built-in ones, a topology check, the subcircuit tags of the BBD
    /// permutation, then the runtime-loaded (.hdl) ones. V and L go first;
    /// F/H/W/K wait until the end so the rows they name exist. Refuses a
    /// card it cannot bind, a loop of V/L shorts whose KVL sum disagrees
    /// (VoltageSourceLoop) and a node only current sources reach
    /// (CurrentSourceCutset); a capacitor-only node sets `needs_tran_op`.
    pub fn build(self: *NetBuilder) !void {
        for ("vlifhwkabcdegjmnopqrstuxyz") |c| for (self.nl.bucket(c)) |e| try net_card.addDevice(self, self.nl.device(e));
        try net_branches.resolveDeferred(self);
        try net_topology.topoCheck(self);
        try self.tagSubcircuitNodes();
        try net_dyn.addDynDevices(self);
    }
};

/// A model card bound once for device type `type`; `model` points at its `D.Model`.
const BoundCard = struct { type: DeviceType, model: *const anyopaque };

/// `NetBuilder.memo`'s value: Model row `row`, a `D.Instance` and the
/// `n_u` entries of its collapse map, for the type the key names.
const Memo = struct { row: u32, instance: *const anyopaque, col: []const i32 };

/// The runtime-loaded (.hdl) module a card's first positional names, directly
/// or through its `.model` card's kind (`.model psp103n psp103va ...`).
pub fn loadedType(lib: *const Library, dev: Device) ?DeviceType {
    const name = Value.nameAt(dev.positional, 0) orelse return null;
    return lib.find(name) orelse if (dev.model) |m| lib.find(m.kind) else null;
}

/// Writes a positional card value (`R1 a b 1k`, `F1 ... 2.0`) to `field` on
/// whichever of Model/Instance declares it; `applyKv` covers `name=value`.
/// Checking both keeps the binding if a model moves the parameter. Returns
/// false when neither declares it.
pub fn setParam(comptime D: type, model: *D.Model, instance: *D.Instance, comptime field: []const u8, value: f64) !bool {
    if (comptime @hasField(D.Model, field)) {
        @field(model.*, field) = try castField(@TypeOf(@field(model.*, field)), value);
        markGiven(model, field);
    } else if (comptime @hasField(D.Instance, field)) {
        @field(instance.*, field) = try castField(@TypeOf(@field(instance.*, field)), value);
        markGiven(instance, field);
    } else return false;
    return true;
}

// ---- private ----
