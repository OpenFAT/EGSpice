//! Behavioural B sources as expression tapes. In: a B card's `v=`, `i=` or
//! `q=` postfix. Out: the opcode tape models/bsource.va interprets, with
//! constants folded and probes mapped to rows.
//! Cites: ngspice asrcset.c; models/bsource.va (opcodes).
//!
//! Contents:
//! - `pub fn addBsource(self: *NetBuilder, dev: Device) !bool`
//! - `pub fn controlledNodes(self: *NetBuilder, dev: Device) ![4]u32`
//! - `pub fn cardScale(dev: Device, letter: u8) f64`
//! - `pub fn readsCurrent(nl: Netlist, dev: Device) bool`

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
const net_waves = @import("net_waves.zig");
const net_cards = @import("net_cards.zig");
const NetBuilder = net_builder.NetBuilder;

/// A B card's output: V(p,n), I(p,n), or the charge whose ddt is I(p,n).
const BMode = enum { v, i, q };
const b_modes = std.StaticStringMap(BMode).initComptime(.{ .{ "v", .v }, .{ "i", .i }, .{ "q", .q } });

/// The tape's opcodes and capacities. The opcode numbers and capacities are
/// the ones models/bsource.va interprets; `Tape.store` checks the capacities
/// against the generated Model at compile time.
const tape = struct {
    /// The interpreter's stack, `st[0:15]` in models/bsource.va: zeroed on
    /// every evaluation (vera_scratch), so it is kept short.
    const max_depth = 16;
    const max_ops = 64;
    const max_consts = 32;
    const max_probes = 8;
    const Code = enum(u8) {
        num,
        v,
        vd,
        neg,
        not,
        add,
        sub,
        mul,
        div,
        powi,
        powc,
        pow,
        lt,
        gt,
        le,
        ge,
        eq,
        ne,
        @"and",
        @"or",
        sel,
        sqrt,
        abs,
        min,
        max,
        exp,
        ln,
        log10,
        sin,
        cos,
        tan,
        atan,
        tanh,
        floor,
        ceil,
        time,
        temper,
        pwl,
    };
};

/// A compiled expression: postfix opcodes with two u8 operands each, and a
/// constant pool. Built here, then copied into the device's parameter arrays.
const Tape = struct {
    n_ops: u8 = 0,
    op_code: [tape.max_ops]tape.Code = @splat(.num),
    op_a: [tape.max_ops]u8 = @splat(0),
    op_b: [tape.max_ops]u8 = @splat(0),
    consts: [tape.max_consts]f64 = @splat(0.0),

    /// Writes the tape into a bsource Model, whose arrays VerA flattens into
    /// one field per slot (see `pwlSlot`).
    fn store(t: Tape, comptime M: type, model: *M) void {
        comptime std.debug.assert(net_waves.slotCount(M, "op_code") == tape.max_ops and net_waves.slotCount(M, "consts") == tape.max_consts);
        model.n_ops = t.n_ops;
        inline for (0..tape.max_ops) |k| {
            @field(model, net_waves.pwlSlot("op_code", k)) = @backingInt(t.op_code[k]);
            @field(model, net_waves.pwlSlot("op_a", k)) = t.op_a[k];
            @field(model, net_waves.pwlSlot("op_b", k)) = t.op_b[k];
        }
        inline for (0..tape.max_consts) |k| @field(model, net_waves.pwlSlot("consts", k)) = t.consts[k];
    }
};

/// B card as an expression tape, run by models/bsource.va (V output),
/// bsource_i.va (I output) or bsource_q.va (the `q=` charge of a behavioural
/// VCCAP). Returns true for a voltage-mode B, the only kind with a branch
/// current worth probing (ngspice asrcset.c:81-88).
pub fn addBsource(self: *NetBuilder, dev: Device) !bool {
    // The last v=/i=/q= wins, as ngspice reads the card.
    var mode: BMode = .v;
    for (dev.kv) |item| mode = b_modes.get(item.key) orelse mode;
    switch (mode) {
        .v => try addTape(self, devices.bsource, dev),
        .i => try addTape(self, devices.bsource_i, dev),
        .q => try addTape(self, devices.bsource_q, dev),
    }
    return mode == .v;
}

/// Four rows: the output pair, then the control pair.
pub fn controlledNodes(self: *NetBuilder, dev: Device) ![4]u32 {
    if (dev.pins.len != 4) return error.InvalidControlledSourceNodes;
    var nodes: [4]u32 = undefined;
    for (dev.pins, &nodes) |pin, *row| row.* = try self.rowOf(pin);
    return nodes;
}

/// HSPICE `SCALE` times, for a G card, `M`; both default to 1.
pub fn cardScale(dev: Device, letter: u8) f64 {
    const m = if (letter == 'g') net_cards.kvNumber(dev.kv, "m") orelse 1 else 1;
    return (net_cards.kvNumber(dev.kv, "scale") orelse 1) * m;
}

/// True when a B card's expression reads a branch current.
pub fn readsCurrent(nl: Netlist, dev: Device) bool {
    for (dev.kv) |item| switch (item.value) {
        .expr => |span| for (nl.exprOps(span)) |op| {
            if (op.code == .iprobe) return true;
        },
        else => {},
    };
    return false;
}

// ---- private ----

fn addTape(self: *NetBuilder, comptime B: type, dev: Device) !void {
    if (comptime !@hasDecl(B, "eval")) return error.UnsupportedDevice;
    var model: B.Model = .{};
    var instance: B.Instance = .{};
    if (dev.model) |m| try net_cards.applyKv(&model, m.kv);
    try net_cards.applyKv(&model, dev.kv);
    try net_cards.applyKv(&instance, dev.kv);

    // Output rows before control rows, so node numbering follows the card.
    var nodes: [B.num_ports]u32 = @splat(GROUND);
    for (dev.pins[0..@min(dev.pins.len, 2)], 0..) |pin, k| nodes[k] = try self.rowOf(pin);
    var t: Tape = .{};
    for (dev.kv) |item| {
        _ = b_modes.get(item.key) orelse continue;
        t = .{};
        switch (item.value) {
            .num => |value| {
                t.consts[0] = value;
                t.n_ops = 1;
            },
            .expr => |span| {
                const folded = try foldConstants(self.arena, self.nl.exprOps(span), self.nl.consts);
                compileTape(self, folded.ops, folded.consts, &t, &nodes) catch |err| switch (err) {
                    error.OutOfMemory => return err,
                    else => {
                        // The test runner fails any test that logs an error.
                        if (!@import("builtin").is_test) std.log.err("B-source '{s}': the {s}= expression is not supported ({s}); the limits are i() of branch cards only, no unknown names, known functions only, {d} probed nets and currents, {d} ops, {d} constants, stack depth {d}", .{ dev.name, item.key, @errorName(err), tape.max_probes, tape.max_ops, tape.max_consts, tape.max_depth });
                        return error.UnsupportedBsourceExpression;
                    },
                };
            },
            else => return error.UnresolvedParameter,
        }
    }
    t.store(B.Model, &model);
    try self.b.addDevice(B, dev.name, &model, &instance, nodes);
}

/// `ops` with every constant subtree folded to one `.num`, the consts it
/// reads renumbered into a fresh pool. A parameter spliced into a B
/// expression arrives as its own postfix (sky130's `rbody` carries
/// `mc_mm_switch*agauss(...)`), so without this the tape would hold
/// distribution calls and spend its op budget on arithmetic of constants.
/// Distributions fold to their nominal, as `expr.fold` does.
fn foldConstants(arena: std.mem.Allocator, ops: []const Op, consts: []const f64) !struct { ops: []const Op, consts: []const f64 } {
    var out: std.ArrayList(Op) = .empty;
    var pool: std.ArrayList(f64) = .empty;
    var stack: std.ArrayList(netlist.expr.Val) = .empty;
    for (ops) |op| {
        if (op.code == .num) {
            try out.append(arena, .{ .code = .num, .a = @intCast(pool.items.len) });
            try pool.append(arena, consts[op.a]);
            continue;
        }
        const k = netlist.expr.arity(op);
        const table = op.code == .call and @as(netlist.expr.Fn, @fromBackingInt(@intCast(op.a))) == .table;
        const foldable = k > 0 and k <= out.items.len and !table and op.code != .live and
            for (out.items[out.items.len - k ..]) |o| {
                if (o.code != .num) break false;
            } else true;
        if (!foldable) {
            try out.append(arena, op);
            continue;
        }
        // The operands are the last k consts, in order.
        var mini: [9]Op = undefined;
        if (k + 1 > mini.len) {
            try out.append(arena, op);
            continue;
        }
        for (0..k) |i| mini[i] = .{ .code = .num, .a = @intCast(i) };
        mini[k] = op;
        const args = pool.items[pool.items.len - k ..];
        const v = try netlist.expr.eval(arena, &stack, mini[0 .. k + 1], args, &.{}, null);
        if (std.math.isNan(v)) {
            try out.append(arena, op);
            continue;
        }
        out.shrinkRetainingCapacity(out.items.len - k);
        pool.shrinkRetainingCapacity(pool.items.len - k);
        try out.append(arena, .{ .code = .num, .a = @intCast(pool.items.len) });
        try pool.append(arena, v);
    }
    return .{ .ops = out.items, .consts = pool.items };
}

/// Translates a postfix card expression into the bsource tape, giving each
/// distinct probed row (a net, or the branch an i() reads) a control port in
/// `nodes`. Fails on anything the tape cannot express or hold.
fn compileTape(self: *NetBuilder, ops: []const Op, consts: []const f64, model: *Tape, nodes: []u32) !void {
    var rows: [tape.max_probes]u32 = undefined;
    var n_rows: usize = 0;
    var n_consts: usize = 0;
    var n: usize = 0;
    for (ops) |op| {
        if (n == tape.max_ops) return error.TooManyOps;
        const code: tape.Code = switch (op.code) {
            .num => blk: {
                if (n_consts == tape.max_consts) return error.TooManyConstants;
                model.consts[n_consts] = consts[op.a];
                model.op_a[n] = @intCast(n_consts);
                n_consts += 1;
                break :blk .num;
            },
            .vprobe, .iprobe => blk: {
                if (op.a == netlist.none) return error.EmptyProbe;
                var port: [2]u8 = .{ 0, 0 };
                for ([2]u32{ op.a, op.b }, &port) |id, *slot| {
                    if (id == netlist.none) break;
                    const row = if (op.code == .vprobe)
                        try self.rowOf(.from(id))
                    else
                        net_branches.branchRow(self, self.nl.pool.str(@fromBackingInt(@intCast(id)))) orelse return error.UnknownCurrentProbe;
                    const k = std.mem.indexOfScalar(u32, rows[0..n_rows], row) orelse k: {
                        if (n_rows == tape.max_probes) return error.TooManyProbes;
                        rows[n_rows] = row;
                        nodes[2 + n_rows] = row;
                        n_rows += 1;
                        break :k n_rows - 1;
                    };
                    slot.* = @intCast(k);
                }
                model.op_a[n] = port[0];
                model.op_b[n] = port[1];
                break :blk if (op.b == netlist.none) .v else .vd;
            },
            .ident => switch (op.a) {
                2 => .time,
                3 => .temper,
                else => return error.UnsupportedOperand,
            },
            .call => blk: {
                if (@as(netlist.expr.Fn, @fromBackingInt(@intCast(op.a))) != .table) break :blk try callCode(@fromBackingInt(@intCast(op.a)), op.b);
                // table(x, d, x1, y1, ...): the constants after x become the
                // op's table, already consecutive in the pool.
                const k = op.b - 1;
                if (op.b < 6 or k % 2 != 1 or n < k) return error.WrongArity;
                for (model.op_code[n - k .. n]) |c| if (c != .num) return error.NonConstantTable;
                n -= k;
                const at = model.op_a[n];
                const pts = model.consts[at + 1 .. at + k];
                var i: usize = 2;
                while (i < pts.len) : (i += 2) if (!(pts[i] > pts[i - 2])) return error.TableNotAscending;
                model.op_b[n] = @intCast(k / 2);
                break :blk .pwl;
            },
            // The tape has no opcode for numparam's `%` and `\`; they fold
            // when their operands are constant.
            .live, .mod, .idiv => return error.UnsupportedOperand,
            // `subst` resolves every one; none reaches a tape.
            .ucall => return error.UnsupportedFunction,
            inline else => |c| @field(tape.Code, @tagName(c)),
        };
        model.op_code[n] = code;
        // A constant exponent folds into the power op: that constant is the
        // previous op and the last pool entry (`powc` keeps its index).
        if (code == .pow and model.op_code[n - 1] == .num) {
            const e = model.consts[n_consts - 1];
            n -= 1;
            if (e >= 0 and e <= 255 and e == @round(e)) {
                n_consts -= 1;
                // ngspice 45's default PTpower AND PTpowerH use |x|^y.
                // Keep constant-integer folding consistent with the general
                // power and the frontend's constant evaluator. HSPICE keeps
                // its signed integer power.
                if (self.nl.deck.dialect == .ngspice) {
                    if (n + 1 == tape.max_ops) return error.TooManyOps;
                    model.op_code[n] = .abs;
                    n += 1;
                }
                model.op_code[n] = .powi;
                model.op_a[n] = @intFromFloat(e);
            } else model.op_code[n] = .powc;
        }
        n += 1;
    }
    model.n_ops = @intCast(n);
    if (stackDepth(model.*) > tape.max_depth) return error.TooDeep;
}

/// The deepest the interpreter's stack gets running `t`.
fn stackDepth(t: Tape) usize {
    var sp: isize = 0;
    var deepest: isize = 0;
    for (t.op_code[0..t.n_ops]) |code| {
        sp += switch (code) {
            .num, .v, .vd, .time, .temper => 1,
            .sel => -2,
            .add, .sub, .mul, .div, .pow, .lt, .gt, .le, .ge, .eq, .ne, .@"and", .@"or", .min, .max => -1,
            else => 0,
        };
        deepest = @max(deepest, sp);
    }
    return @intCast(deepest);
}

/// The tape op for a call of `f` with `argc` arguments.
fn callCode(f: netlist.expr.Fn, argc: u32) !tape.Code {
    const want: u32 = switch (f) {
        .min, .max, .pow, .pwr => 2,
        .ternary => 3,
        else => 1,
    };
    if (argc != want) return error.WrongArity;
    return switch (f) {
        .ternary => .sel,
        .ln, .log => .ln,
        // ponytail: pwr folds when its arguments are constant (sky130's
        // pwr(l,0.5)); a tape opcode is needed only if one reads a probe.
        .pwr, .agauss, .gauss, .unif, .aunif, .limit, .table, .other => error.UnsupportedFunction,
        // numparam built-ins with no tape opcode: they fold when their
        // arguments are constant, and a B source reading a probe refuses them.
        .sqr, .int, .nint, .sgn, .sinh, .cosh, .asin, .acos, .asinh, .acosh, .atanh => error.UnsupportedFunction,
        // ponytail: HSPICE's signed sqrt/log and integer-exponent pow fold
        // on constants; a tape opcode is needed only if one reads a probe.
        .hsqrt, .hlog, .hlog10, .hpow => error.UnsupportedFunction,
        inline else => |g| @field(tape.Code, @tagName(g)),
    };
}
