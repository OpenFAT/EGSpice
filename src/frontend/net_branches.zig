//! Branch rows by card name. In: the V, L and branch-probe tables as cards
//! are added. Out: each card's branch row, and the F/H/W/K cards bound to
//! the branch they sense once every V and L card exists.
//! Cites: ngspice cccsset.c, ccvsset.c, mutsetup.c (sensed branches).
//!
//! Contents:
//! - `pub fn addBranchProbe(self: *NetBuilder, name: []const u8, row: u32) !void`
//! - `pub fn branchRow(self: *const NetBuilder, name: []const u8) ?u32`
//! - `pub fn vRow(self: *const NetBuilder, name: []const u8) ?u32`
//! - `pub fn lRow(self: *const NetBuilder, name: []const u8) ?usize`
//! - `pub fn indexBranches(self: *NetBuilder) !void`
//! - `pub fn indexBranchNames(self: *NetBuilder) !void`
//! - `pub fn resolveDeferred(self: *NetBuilder) !void`

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
const net_card = @import("net_card.zig");
const net_passive = @import("net_passive.zig");
const NetBuilder = net_builder.NetBuilder;

pub fn addBranchProbe(self: *NetBuilder, name: []const u8, row: u32) !void {
    try self.br.append(self.arena, .{ .name = name, .row = row });
}

/// The branch-current row of card `name`: a V card's (its sensing
/// model's control branch once an F/H/W takes it over), an L card's, or
/// any other branch `br` lists, the first of each in that order. Null
/// for a card without one. Asserts that `branch_index` is current.
pub fn branchRow(self: *const NetBuilder, name: []const u8) ?u32 {
    std.debug.assert(self.branch_indexed == self.br.len);
    const k = self.branch_index.get(name) orelse return null;
    const nv = self.v.len;
    const nl = self.l.len;
    if (k < nv) return self.v.items(.branch)[k];
    if (k < nv + nl) return self.l.items(.branch)[k - nv];
    std.debug.assert(k - nv - nl < self.br.len);
    return self.br.items(.row)[k - nv - nl];
}

/// The first V row named `name`.
pub fn vRow(self: *const NetBuilder, name: []const u8) ?u32 {
    const k = self.branch_index.get(name) orelse return null;
    return if (k < self.v.len) k else null;
}

/// The first L row named `name`.
pub fn lRow(self: *const NetBuilder, name: []const u8) ?usize {
    const k = self.branch_index.get(name) orelse return null;
    const nv = self.v.len;
    if (k >= nv) return if (k < nv + self.l.len) k - nv else null;
    // ponytail: a V card of the same name (Spectre names carry no
    // letter) hides the L in the index; scan the L table for that case.
    return netlist.nameIndex(self.l.items(.name), name);
}

/// Adds the `br` rows past `branch_indexed` to `branch_index`, the
/// first of each name winning.
pub fn indexBranches(self: *NetBuilder) !void {
    const nv: u32 = @intCast(self.v.len);
    const nl: u32 = @intCast(self.l.len);
    const names = self.br.items(.name);
    for (names[self.branch_indexed..], self.branch_indexed..) |name, k| {
        const entry = try self.branch_index.getOrPut(self.arena, name);
        if (!entry.found_existing) entry.value_ptr.* = nv + nl + @as(u32, @intCast(k));
    }
    self.branch_indexed = names.len;
}

/// Builds `branch_index` afresh over the V and L tables, which must be
/// complete, and the `br` rows so far.
pub fn indexBranchNames(self: *NetBuilder) !void {
    self.branch_index.clearRetainingCapacity();
    self.branch_indexed = 0;
    try self.branch_index.ensureTotalCapacity(self.arena, @intCast(self.v.len + self.l.len + self.br.len));
    for ([_][]const []const u8{ self.v.items(.name), self.l.items(.name) }, [_]usize{ 0, self.v.len }) |names, base| {
        for (names, base..) |name, k| {
            const entry = self.branch_index.getOrPutAssumeCapacity(name);
            if (!entry.found_existing) entry.value_ptr.* = @intCast(k);
        }
    }
    try indexBranches(self);
    std.debug.assert(self.branch_indexed == self.br.len);
}

/// Adds the F/H/W/K cards once every V and L row exists, then the B
/// cards that read i(), once F/H/W have claimed their sensed branches.
pub fn resolveDeferred(self: *NetBuilder) !void {
    if (self.deferred.items.len == 0) return;
    try indexBranchNames(self);
    for (self.deferred.items) |dev| {
        switch (dev.kind) {
            'f' => try net_passive.addBranchRef(self, devices.cccs, dev, 1.0),
            'h' => try net_passive.addBranchRef(self, devices.ccvs, dev, 0.0),
            'w' => try net_passive.addBranchRef(self, devices.cswitch, dev, null),
            'k' => try net_passive.addKinduc(self, dev),
            'b' => {},
            else => unreachable,
        }
    }
    // H output branches, then each B's own as it lands, so a later B
    // reads an earlier one's i() as the scan did.
    try indexBranches(self);
    for (self.deferred.items) |dev| if (dev.kind == 'b') {
        try net_card.addB(self, dev);
        try indexBranches(self);
    };
}

// ---- private ----
