//! One card to its devices: the dispatch over the card letter. In: one
//! netlist card. Out: the device instances its letter (or loaded module)
//! selects, on the Builder.
//! Cites: ngspice inp2*.c (one reader per letter).
//!
//! Contents:
//! - `pub fn addDevice(self: *NetBuilder, card: Device) !void`
//! - `pub fn addByLetter(self: *NetBuilder, letter: u8, dev: Device) !void`
//! - `pub fn addB(self: *NetBuilder, dev: Device) !void`

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
const net_branches = @import("net_branches.zig");
const net_topology = @import("net_topology.zig");
const net_lines = @import("net_lines.zig");
const net_passive = @import("net_passive.zig");
const net_sources = @import("net_sources.zig");
const net_devices = @import("net_devices.zig");
const net_tape = @import("net_tape.zig");
const net_filters = @import("net_filters.zig");
const net_waves = @import("net_waves.zig");
const net_cards = @import("net_cards.zig");
const NetBuilder = net_builder.NetBuilder;

pub fn addDevice(self: *NetBuilder, card: Device) !void {
    // A source's waveform in the model's argument order (ngspiceWaveform).
    const dev = if (card.kind == 'v' or card.kind == 'i') try net_sources.ngspiceWaveform(self, card) else card;
    // HDL devices are added by addDynDevices; their opaque nodes count as
    // a DC path for the topology check.
    if (net_builder.loadedType(self.b.lib, dev) != null) {
        try net_topology.topoMark(self, dev, .dc, 0);
        return;
    }
    const letter = dev.kind;
    // 'u' (URC) has no DeviceId behind it: the card expands into
    // resistor/capacitor/diode lumps in addUrc, like ngspice's URCsetup.
    if (letter != 'u' and devices.letter_map.get(&.{letter}) == null) {
        if (try net_devices.inferDeviceFromModel(dev) == null)
            return net_devices.named(dev, error.UnsupportedDevice);
    }

    switch (letter) {
        'c' => try net_topology.topoMark(self, dev, .cap, 0),
        'i' => try net_topology.topoMark(self, dev, .cur, 0),
        // A V source shorts its pair at its DC value; an inductor at 0 V.
        // A P card's series z0 keeps n+ off the short.
        'v' => if (net_cards.isPortCard(dev)) try net_topology.topoMark(self, dev, .dc, 0) else try net_topology.topoMark(self, dev, .short, net_cards.sourceDc(dev) orelse 0),
        'l' => try net_topology.topoMark(self, dev, .short, 0),
        else => try net_topology.topoMark(self, dev, .dc, 0),
    }

    switch (letter) {
        'r' => _ = try net_passive.addPassive(self, devices.resistor, dev, "r", "resist"),
        'c' => {
            _ = try net_passive.addPassive(self, devices.capacitor, dev, "c", "cap");
            // ponytail: a floating C sets its n+ node, not only its own
            // charge as ngspice's CAPload under MODEUIC does, so another
            // charge on that node starts charged too. Exact for a grounded
            // C; a per-device charge seed in the uic start is the upgrade.
            if (try net_cards.numericParameter(dev.kv, "ic")) |ic| {
                const p = try self.rowOf(dev.pins[0]);
                try self.dev_ic.append(self.arena, if (p != GROUND) .{ .node = p, .value = ic } else .{ .node = try self.rowOf(dev.pins[1]), .value = -ic });
            }
        },
        'l' => {
            const br = try net_passive.addPassive(self, devices.inductor, dev, "l", "inductance");
            if (try net_cards.numericParameter(dev.kv, "ic")) |ic| try self.dev_ic.append(self.arena, .{ .node = br, .value = ic });
            try self.l.append(self.arena, .{ .name = dev.name, .branch = br, .value = Value.numberAt(dev.positional, 0) orelse
                net_cards.kvNumber(dev.kv, "inductance") orelse net_cards.kvNumber(dev.kv, "l") orelse 0 });
        },
        'v' => {
            if (comptime !@hasDecl(devices.vsource, "eval")) return error.UnsupportedDevice;
            const bound = try net_sources.bindSource(self, devices.vsource, dev);
            var nodes = try net_devices.deviceNodes(self, devices.vsource, dev);
            const pos = nodes[0];
            const sensors: u32 = if (self.sensors.fetchRemove(dev.name)) |kv| kv.value else 0;
            const port = if (sensors != 0) null else try net_cards.sourcePort(dev);
            // An HSPICE P card keeps its z0 in series in every analysis
            // [SA Ch.17]: a noiseless resistor from n+ to a hidden node the
            // source drives. ngspice's portnum V stays ideal.
            const series = port != null and net_cards.isPortCard(dev);
            const neg = nodes[1];
            var port_branch: u32 = undefined;
            var balanced: ?requests.Port.Leg = null;
            if (series and dev.pins.len == 3) {
                // Mixed mode [SA Ch.17]: the card's value V drives a hidden
                // node m against ref, and each leg is V/2 or -V/2 against
                // ref (a VCVS on m) behind its own z0.
                const ref = try self.rowOf(dev.pins[2]);
                nodes = .{ try self.b.addNode(), ref };
                var legs: [2]requests.Port.Leg = undefined;
                for (&legs, [2]u32{ pos, neg }, [2]f64{ 0.5, -0.5 }) |*leg, pin, gain| {
                    const h = try self.b.addNode();
                    try self.b.addDevice(devices.resistor, dev.name, &.{ .r = port.?.z0, .noisy = 0 }, &.{}, [2]u32{ pin, h });
                    try self.b.addDevice(devices.vcvs, dev.name, &.{ .gain = gain }, &.{}, [4]u32{ h, ref, nodes[0], ref });
                    leg.* = .{ .node = pin, .branch = net_devices.internalRow(devices.vcvs, "flowZ28pZ2cnZ29", self.b.n) };
                }
                port_branch = legs[0].branch;
                balanced = legs[1];
            } else if (series) {
                nodes[0] = try self.b.addNode();
                try self.b.addDevice(devices.resistor, dev.name, &.{ .r = port.?.z0, .noisy = 0 }, &.{}, [2]u32{ pos, nodes[0] });
            }
            // A PWL past the table's capacity continues in sources of its
            // later segments, in series between the card's source and n-.
            const tail = try net_waves.pwlTail(devices.vsource, self.arena, dev);
            if (tail.len != 0 and series) {
                if (!@import("builtin").is_test) std.log.err("{s}: a PWL past {d} points on a port source", .{ dev.name, comptime net_waves.pwlCapacity(devices.vsource.Model) });
                return error.UnsupportedCard;
            }
            // A sensed source ends at a hidden tap; addBranchRef chains
            // its sensors from there to n- (see `VCard.tap`).
            if (sensors != 0) nodes[1] = try self.b.addNode();
            const chain_end = nodes[1];
            if (tail.len != 0) nodes[1] = try self.b.addNode();
            const br = self.b.n;
            if (balanced == null) port_branch = br;
            try self.b.addDevice(devices.vsource, dev.name, &bound[0], &bound[1], nodes);
            var from = nodes[1];
            for (tail, 0..) |seg, k| {
                const to = if (k + 1 == tail.len) chain_end else try self.b.addNode();
                try self.b.addDevice(devices.vsource, dev.name, &seg, &.{}, [2]u32{ from, to });
                from = to;
            }
            try self.v.append(self.arena, .{
                .name = dev.name,
                .pos = pos,
                .neg = neg,
                .branch = br,
                .tap = chain_end,
                .sensors = sensors,
                .distof1 = net_cards.sourceDisto(dev, "distof1"),
                .distof2 = net_cards.sourceDisto(dev, "distof2"),
            });
            if (port) |p| try self.ports.append(self.arena, .{ .num = p.num, .port = .{
                .node = pos,
                .neg = chain_end,
                .branch = port_branch,
                .z0 = p.z0,
                .band = p.band,
                .series_z0 = series,
                .balanced = balanced,
            } });
            if (self.source_branch == GROUND) {
                self.source_node = if (series) pos else nodes[0];
                self.source_branch = br;
            }
            if (net_cards.sourceAc(dev)) |ac| try self.ac.append(self.arena, .{ .pos = GROUND, .neg = br, .re = ac.re, .im = ac.im });
        },
        'i' => {
            if (comptime !@hasDecl(devices.isource, "eval")) return error.UnsupportedDevice;
            var bound = try net_sources.bindSource(self, devices.isource, dev);
            // isrcload.c/isrcacld.c scale every value by `m=`.
            const m = net_cards.kvNumber(dev.kv, "m") orelse 1;
            bound[1].mfactor = m;
            const nodes = try net_devices.deviceNodes(self, devices.isource, dev);
            try self.b.addDevice(devices.isource, dev.name, &bound[0], &bound[1], nodes);
            // A PWL past the table's capacity: its later segments in parallel.
            for (try net_waves.pwlTail(devices.isource, self.arena, dev)) |seg| try self.b.addDevice(devices.isource, dev.name, &seg, &.{ .mfactor = m }, nodes);
            if (net_cards.sourceAc(dev)) |ac| try self.ac.append(self.arena, .{ .pos = nodes[0], .neg = nodes[1], .re = m * ac.re, .im = m * ac.im });
            try self.i.append(self.arena, .{ .name = dev.name, .pos = nodes[0], .neg = if (nodes.len > 1) nodes[1] else GROUND });
            try self.topo_isrc.append(self.arena, .{ .pos = nodes[0], .neg = if (nodes.len > 1) nodes[1] else GROUND, .dc = bound[0].dc });
        },
        'w' => if (self.nl.deck.dialect == .hspice) try net_lines.addWline(self, dev) else try self.deferred.append(self.arena, dev),
        's' => if (self.nl.deck.dialect == .hspice) try net_lines.addSparam(self, dev) else try addByLetter(self, letter, dev),
        'f', 'h', 'k' => try self.deferred.append(self.arena, dev),
        // A tape that reads i() waits until every branch row exists.
        'b' => if (net_tape.readsCurrent(self.nl, dev)) try self.deferred.append(self.arena, dev) else try addB(self, dev),
        'p' => try net_lines.addCpl(self, dev),
        'o' => try net_lines.addLossyLine(self, dev),
        'y' => try net_lines.addTxl(self, dev),
        'u' => try net_lines.addUrc(self, dev),
        'e', 'g' => if (net_cards.kvNumber(dev.kv, "laplace") != null) {
            try net_filters.addLaplace(self, dev, letter);
        } else if (net_cards.kvNumber(dev.kv, "pole") != null) {
            try net_filters.addPole(self, dev, letter);
        } else if (net_cards.kvNumber(dev.kv, "td") != null) {
            try net_filters.addDelay(self, dev, letter);
        } else {
            try addByLetter(self, letter, dev);
            if (letter == 'e') try net_branches.addBranchProbe(self, dev.name, net_devices.internalRow(devices.vcvs, "flowZ28pZ2cnZ29", self.b.n));
        },
        else => try addByLetter(self, letter, dev),
    }
}

pub fn addByLetter(self: *NetBuilder, letter: u8, dev: Device) !void {
    // The netlist moved a Q/M card's model name to positional[0] whatever
    // its terminal count.
    switch (net_devices.resolveDeviceId(letter, dev) catch |err| return net_devices.named(dev, err)) {
        inline else => |comptime_id| {
            const D = devices.DeviceId.Type(comptime_id);
            net_devices.addSingleDevice(self, D, dev) catch |err| return net_devices.named(dev, err);
            // ngspice's VBIC `i(q1)` is its excess-phase branch, made only
            // when TD > 0 (vbicsetup.c:510-525). Node xf2 carries only a
            // 1 ohm load, so that current equals v(xf2): alias the column.
            if (comptime_id == .vbic13_4t) {
                const td = net_cards.kvNumber(dev.kv, "td") orelse if (dev.model) |m| net_cards.kvNumber(m.kv, "td") orelse 0 else 0;
                if (td > 0) try net_branches.addBranchProbe(self, dev.name, net_devices.internalRow(D, "xf2", self.b.n));
            }
        },
    }
}

/// B card. Only a V-mode B gets an i() column: ngspice makes the branch
/// only for ASRC_VOLTAGE (asrcset.c:81-88). Our bsource declares the
/// unknown either way; the I-mode one stays unprobed rather than
/// published as a column ngspice never writes.
pub fn addB(self: *NetBuilder, dev: Device) !void {
    if (try net_tape.addBsource(self, dev))
        try net_branches.addBranchProbe(self, dev.name, net_devices.internalRow(devices.bsource, "flowZ28pZ2cnZ29", self.b.n));
}

// ---- private ----
