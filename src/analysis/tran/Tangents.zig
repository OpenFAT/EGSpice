//! Direct sensitivities: the first-order trajectories of `Options.tangents`,
//! one per variant row, on the base run's own time grid. The base solution,
//! its step's matrix and companion terms in; each row's x, charge history
//! and device state, moved by -A⁻¹ (R(x, row) - R(x)) at the operating point
//! and at every accepted step, and its waveform out. Each row keeps its
//! device state, with its parameter values, as a `Circuit.SavedState`
//! swapped in for its turn, so a GPU-resident circuit, which `restoreState`
//! leaves alone, turns every row to NaN, as do a singular A and a row that
//! re-wires a device.
//! Cites: none (dev/analysis/optimize.md, "Direct sensitivities").
//!
//! Contents:
//! - `pub fn init(ctx: *const root.RunCtx, opts: Options, n_probes: u32, gpa: std.mem.Allocator) !Tangents`
//! - `pub fn deinit(tg: *Tangents, gpa: std.mem.Allocator) void`
//! - `pub fn advance(tg: *Tangents, ckt: *root.Circuit, ws: *converger.Workspace, a_vals: []f64, x: []const f64, t: f64, step: ?Step) !void`
//! - `pub fn record(tg: *Tangents, t: f64, probes: []const u32) !void`
//! - `pub fn open(tg: Tangents) usize`

const std = @import("std");
const assert = std.debug.assert;
const num = @import("core").numerics;
const converger = @import("solver").converger;
const Stream = @import("measure").Stream;
const root = @import("../types.zig");
const integrator = @import("integrator.zig");
const tran_types = @import("types.zig");

const Method = tran_types.Method;
const Options = tran_types.Options;
const Waveform = tran_types.Waveform;
const initialCapacity = tran_types.initialCapacity;

rows: []const u32,
variants: @import("core").Variants,
params: []const root.ParamRef,
/// Per row, 5 n values: x, then the charge ring q0, q1, q2, then the
/// dynamic current, as the base run's `q_hist` and `i_prev`.
state: []f64,
dead: []bool,
wf: []Waveform,
/// Each row's device state between turns, then the base run's.
devices: []root.Circuit.SavedState,
/// The base residual, n values.
r0: []f64,
/// Each row's `.meas` cards under early stop, else empty.
streams: []Stream = &.{},

/// The companion terms of the step the base run just accepted.
pub const Step = struct { method: Method, c: integrator.Coeffs, q1: []const f64, q2: []const f64, i: []const f64 };

const Tangents = @This();

pub fn init(ctx: *const root.RunCtx, opts: Options, n_probes: u32, gpa: std.mem.Allocator) !Tangents {
    const ckt = ctx.circuit;
    const k = opts.tangents.len;
    var tg: Tangents = .{
        .rows = opts.tangents,
        .variants = ctx.variants,
        .params = try ckt.collectParams(),
        .state = &.{},
        .dead = &.{},
        .wf = &.{},
        .devices = &.{},
        .r0 = &.{},
    };
    errdefer tg.deinit(gpa);
    tg.state = try gpa.alloc(f64, 5 * ckt.n * k);
    tg.dead = try gpa.alloc(bool, k);
    @memset(tg.dead, ckt.gpu != null);
    tg.r0 = try gpa.alloc(f64, ckt.n);
    tg.wf = try gpa.alloc(Waveform, k);
    for (tg.wf, 0..) |*w, i| w.* = Waveform.init(gpa, n_probes, initialCapacity(opts)) catch |e| {
        tg.wf = tg.wf[0..i];
        return e;
    };
    tg.devices = try gpa.alloc(root.Circuit.SavedState, k + 1);
    for (tg.devices, 0..) |*d, i| d.* = ckt.saveState(gpa) catch |e| {
        tg.devices = tg.devices[0..i];
        return e;
    };
    return tg;
}

pub fn deinit(tg: *Tangents, gpa: std.mem.Allocator) void {
    for (tg.streams) |*s| s.deinit();
    gpa.free(tg.streams);
    for (tg.wf) |*w| w.deinit();
    for (tg.devices) |d| d.deinit(gpa);
    gpa.free(tg.wf);
    gpa.free(tg.devices);
    gpa.free(tg.dead);
    gpa.free(tg.state);
    gpa.free(tg.r0);
}

/// Moves every row to the base solution `x` at `t`: after a static
/// solve when `step` is null (the operating point, where each row takes
/// its parameter values and seeds its charge history), else after that
/// companion step, its device state staged but not committed. Leaves the
/// base run's device state and parameters as it found them; the first
/// call drops the constant-Jacobian baseline (`Circuit.recompute`).
pub fn advance(tg: *Tangents, ckt: *root.Circuit, ws: *converger.Workspace, a_vals: []f64, x: []const f64, t: f64, step: ?Step) !void {
    const n = ckt.n;
    const charge = ckt.has_charge;
    const base = tg.devices[tg.rows.len];
    // The rows' accepted points print nothing.
    const display = ckt.display;
    ckt.display = null;
    defer ckt.display = display;
    ckt.storeState(base);
    defer ckt.restoreState(base);
    // The device state the step's Newton solved under.
    _ = ckt.stateCtl(.revert);
    // A full stamp: the baseline holds the const-Jacobian devices at
    // their nominal values (dc.zig `tangentRows`).
    ckt.eval(x, t);
    num.copySimd(tg.r0, ckt.rhs[0..n]);
    if (step) |s| if (charge) companion(s, tg.r0, ckt.q_vec[0..n], s.q1, s.q2, s.i);
    const vals = if (step != null and charge) blk: {
        ckt.combineGC(step.?.c.ag0, a_vals);
        break :blk a_vals;
    } else ckt.g_vals;
    const factored = if (ws.slv.factor(vals, ckt.solver_execution)) true else |e| switch (e) {
        error.SingularMatrix => false,
        else => return e, // else: only a singular A has an answer (NaN rows)
    };
    for (tg.rows, tg.dead, tg.devices[0..tg.rows.len], 0..) |row, *dead, devices, k| {
        const st = tg.state[5 * n * k ..][0 .. 5 * n];
        const xk = st[0..n];
        const q0 = st[n..][0..n];
        const q1 = st[2 * n ..][0..n];
        const q2 = st[3 * n ..][0..n];
        const ik = st[4 * n ..][0..n];
        if (!factored) dead.* = true;
        if (dead.*) {
            @memset(xk, std.math.nan(f64));
            continue;
        }
        ckt.restoreState(devices);
        if (step == null) {
            const refs, const values = tg.variants.writes(row);
            for (refs, values) |r, v| tg.params[r].set(v);
            ckt.recompute() catch |e| switch (e) {
                // No first-order answer on this pattern.
                error.TopologyChanged => {
                    dead.* = true;
                    @memset(xk, std.math.nan(f64));
                    continue;
                },
            };
        }
        ckt.eval(x, t);
        // xk holds dx until the solve is added to x.
        num.copySimd(xk, ckt.rhs[0..n]);
        if (step) |s| if (charge) companion(s, xk, ckt.q_vec[0..n], q1, q2, ik);
        for (xk, tg.r0) |*d, r| d.* -= r;
        ws.slv.solveNeg(xk, xk);
        for (xk, x) |*xi, xb| xi.* += xb;
        if (step) |s| if (charge) {
            ckt.evalQ(xk, t);
            num.copySimd(q0, ckt.q_vec[0..n]);
            integrator.advanceCurrent(s.method, ik, q0, q1, q2, s.c);
            num.copySimd(q2, q1);
            num.copySimd(q1, q0);
        };
        // The row's accepted point, as the base run commits its own.
        _ = ckt.updateStates(xk);
        _ = ckt.commitStates(xk);
        _ = ckt.stateCtl(.commit);
        if (charge) if (step) |s| {
            if (ckt.has_state_q) {
                ckt.evalQ(xk, t);
                integrator.rebaseCurrent(ik, ckt.q_vec[0..n], q1, s.c.ag0);
                num.copySimd(q1, ckt.q_vec[0..n]);
            }
        } else {
            ckt.evalQ(xk, t);
            num.copySimd(q1, ckt.q_vec[0..n]);
            num.copySimd(q2, q1);
            num.zeroSimd(ik);
        };
        ckt.storeState(devices);
    }
}

pub fn record(tg: *Tangents, t: f64, probes: []const u32) !void {
    const n = tg.r0.len;
    for (tg.wf, 0..) |*w, k| {
        const xk = tg.state[5 * n * k ..][0..n];
        try w.record(t, xk, probes);
        if (tg.streams.len != 0) tg.streams[k].feed(t, xk, w.data());
    }
}

/// `.meas` cards still open over the rows that have values.
pub fn open(tg: Tangents) usize {
    var n: usize = 0;
    for (tg.streams, tg.dead) |s, dead| n += if (dead) 0 else s.open;
    return n;
}

// ---- private ----

/// Adds to `out` the step's dynamic current over the history `q1`,
/// `q2`, `i`, as `TranHook.assemble` adds it to the residual.
fn companion(s: Step, out: []f64, q: []const f64, q1: []const f64, q2: []const f64, i: []const f64) void {
    switch (s.method) {
        inline else => |m| integrator.companionAt(m, true, out, q, q1, q2, i, s.c),
    }
}
