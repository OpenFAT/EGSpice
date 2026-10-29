//! The DC topology check. In: each card's pins and kind as it is added.
//! Out: refusals for a node with no DC path to ground, an inconsistent
//! loop of voltage sources or inductors, and a cut set of current sources.
//! Cites: ngspice CKTsetup-class checks; #70, #90.
//!
//! Contents:
//! - `pub fn topoEnsure(self: *NetBuilder, id: u32) !void`
//! - `pub fn topoRoot(self: *NetBuilder, id0: u32) struct { root: u32, pot: f64 }`
//! - `pub fn compRoot(self: *NetBuilder, id0: u32) u32`
//! - `pub fn topoMark(self: *NetBuilder, dev: Device, kind: enum { dc, cap, cur, short }, vshort: f64) !void`
//! - `pub fn topoCheck(self: *NetBuilder) !void`

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
const NetBuilder = net_builder.NetBuilder;

pub fn topoEnsure(self: *NetBuilder, id: u32) !void {
    while (self.topo.len <= id) {
        const next: u32 = @intCast(self.topo.len);
        try self.topo.append(self.arena, .{ .seen = false, .comp = next, .uf = next, .pot = 0 });
    }
}

/// Root and potential-to-root of `id0` in the weighted forest. Points
/// the whole path at the root, so a chain of shorts listed against its
/// order cannot make every later lookup walk it again.
pub fn topoRoot(self: *NetBuilder, id0: u32) struct { root: u32, pot: f64 } {
    const uf = self.topo.items(.uf);
    const pots = self.topo.items(.pot);
    var root = id0;
    var total: f64 = 0;
    while (uf[root] != root) {
        total += pots[root];
        root = uf[root];
    }
    // Second pass: each node's potential to the root is the total less
    // what lies between id0 and it.
    var id = id0;
    var walked: f64 = 0;
    while (uf[id] != root) {
        const next = uf[id];
        const step = pots[id];
        uf[id] = root;
        pots[id] = total - walked;
        walked += step;
        id = next;
    }
    return .{ .root = root, .pot = total };
}

/// Root of `id` in the DC-path forest, halving the path as it walks.
pub fn compRoot(self: *NetBuilder, id0: u32) u32 {
    const comp = self.topo.items(.comp);
    var id = id0;
    while (comp[id] != id) {
        comp[id] = comp[comp[id]];
        id = comp[id];
    }
    return id;
}

/// Classifies one card's nodes. `.dc` and `.short` join the pins into one
/// DC-path component, `.cap` and `.cur` mark presence only; `.short` also
/// joins the first two nodes as a DC short with v(node0) - v(node1) =
/// `vshort` and rejects a loop whose KVL sum disagrees, or which V
/// sources alone close: V cards are marked before any L, so a V that
/// finds its pins already shorted closes a loop of V sources, whose
/// current no equation fixes (ngspice: singular matrix).
pub fn topoMark(self: *NetBuilder, dev: Device, kind: enum { dc, cap, cur, short }, vshort: f64) !void {
    var first_two: [2]u32 = .{ GROUND, GROUND };
    for (dev.pins, 0..) |pin, i| {
        const id = try self.rowOf(pin);
        try topoEnsure(self, id);
        self.topo.items(.seen)[id] = true;
        if (i < 2) first_two[i] = id;
        if (i > 0 and (kind == .dc or kind == .short))
            self.topo.items(.comp)[compRoot(self, id)] = compRoot(self, first_two[0]);
    }
    if (kind == .short and dev.pins.len >= 2) {
        const a = topoRoot(self, first_two[0]);
        const b_ = topoRoot(self, first_two[1]);
        if (a.root == b_.root) {
            const gap = (a.pot - b_.pot) - vshort;
            // The test runner fails any test that logs an error.
            if (@abs(gap) > 1e-9 * @max(1.0, @abs(vshort))) {
                if (!@import("builtin").is_test) std.log.err(
                    "topology: '{s}' closes an inconsistent voltage-source/inductor loop ({d} V of KVL violation)",
                    .{ dev.name, gap },
                );
                return error.VoltageSourceLoop;
            }
            if (dev.kind == 'v') {
                if (!@import("builtin").is_test) std.log.err(
                    "topology: '{s}' closes a loop of voltage sources — the loop current is not unique",
                    .{dev.name},
                );
                return error.NonuniqueOperatingPoint;
            }
        } else {
            // Attach so every member's potential stays consistent:
            // v(b) = pot_b + pot[rb] must equal v(a) - vshort.
            self.topo.items(.uf)[b_.root] = a.root;
            self.topo.items(.pot)[b_.root] = a.pot - vshort - b_.pot;
        }
    }
}

/// Rejects a net no element touches (one only an expression reads, #92)
/// and a component without a DC path to ground whose current sources do
/// not balance, which static KCL cannot satisfy. A component that
/// balances (capacitors only, or equal sources in and out, #70) sets
/// `needs_tran_op`, ngspice's OPtran fallback.
pub fn topoCheck(self: *NetBuilder) !void {
    const quiet = @import("builtin").is_test; // the test runner fails any test that logs an error
    for (self.rows) |row| {
        if (row == 0 or (row < self.topo.len and self.topo.items(.seen)[row])) continue;
        if (!quiet) std.log.err("topology: node '{s}' is read but no element connects to it", .{self.b.node_labels.items[row]});
        return error.FloatingNode;
    }
    try topoEnsure(self, GROUND);
    // Net current into, and the magnitude through, each component root.
    const net = try self.arena.alloc(f64, 2 * self.topo.len);
    @memset(net, 0);
    const mag = net[self.topo.len..];
    for (self.topo_isrc.items) |src| {
        // n+ to n- through the source: out of n+, into n-.
        for ([2]u32{ src.pos, src.neg }, [2]f64{ -src.dc, src.dc }) |node, i| {
            const r = compRoot(self, node);
            net[r] += i;
            mag[r] += @abs(i);
        }
    }
    const ground = compRoot(self, GROUND);
    for (0..self.topo.len) |id| {
        if (!self.topo.items(.seen)[id]) continue;
        const r = compRoot(self, @intCast(id));
        if (r == ground) continue;
        if (!(@abs(net[r]) <= 1e-9 * mag[r])) {
            if (!quiet) std.log.err("topology: node '{s}' is a current-source cutset — its current sources do not balance and KCL has no DC path to satisfy it", .{self.b.node_labels.items[id]});
            return error.CurrentSourceCutset;
        }
        self.b.needs_tran_op = true;
    }
}

// ---- private ----
