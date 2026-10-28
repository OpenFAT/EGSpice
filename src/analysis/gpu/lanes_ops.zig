//! `tran.Sim`'s numerics backend for one lane of a GPU lane round. In: the
//! step machine's calls. Out: answers from the numbers `GpuLanes.probe`
//! computed for every lane at once, and lane l's mutations queued for
//! `GpuLanes.apply`.
//! Cites: phase4-gpu-lanes.md G1 (the `Ops` surface).
//!
//! Contents:
//! - `pub const Ops = struct (`attempt`, `revert`, `rejectAt`, `flipped`, `boundStep`, `charges`, `bound`, `boundGear`, `advance`, `commit`, `rebase`, `accepted`)`

const std = @import("std");
const assert = std.debug.assert;
const device_ir = @import("device").abi;
const devices = @import("devices.zig");
const lanes = @import("lanes.zig");
const lanes_round = @import("lanes_round.zig");
const SimState = device_ir.SimState;
const StateCtlOp = device_ir.StateCtlOp;
const backend = devices.backend;
const GpuLanes = lanes.GpuLanes;
const LaneNewton = lanes.LaneNewton;

/// `tran.Sim`'s numerics backend for lane l of a `GpuLanes` round (the
/// phase 4 G1 surface): the device did the numerics for every lane at once
/// (`GpuLanes.probe`); this answers `Sim.end`'s calls from those numbers
/// and queues lane l's mutations for `GpuLanes.apply`.
pub const Ops = struct {
    g: *GpuLanes,
    l: u32,

    fn methodCode(m: anytype) u32 {
        return switch (m) {
            .backward_euler => 0,
            .trapezoidal => 1,
            .gear_2 => 2,
            else => unreachable, // gear-3..6 are not lane-eligible
        };
    }

    /// Queues lane l into the next round with the step `a` that `Sim` `s`
    /// attempts: its scalars, `SimState` and LTE inputs. Gear-3..6 is
    /// unreachable.
    pub fn attempt(o: Ops, s: anytype, a: anytype) void {
        const g = o.g;
        const promote = methodCode(s.promoteMethod());
        g.sc[o.l] = .{
            .t_dev = a.t_dev,
            .dt = s.dt,
            .xfact = s.dt / s.dt_old[0],
            .ag0 = a.c.ag0,
            .ag1 = a.c.ag1,
            .ag2 = a.c.ag2,
            .method = methodCode(a.method),
            .flags = 1 | (promote << 4),
            .sq = @splat(0),
            .sqt = @splat(0),
        };
        g.sims[o.l] = s.ckt.sim;
        g.lte_in[o.l] = .{ .dt = s.dt, .dt1 = s.dt_old[0], .dt2 = s.dt_old[1], .reltol = s.options.tol.reltol, .abstol = s.options.tol.abstol, .chgtol = s.options.tol.chgtol, .trtol = s.options.tol.trtol, .ag0 = a.c.ag0, .ag1 = a.c.ag1, .ag2 = a.c.ag2 };
        g.round[o.l] = 1;
    }
    /// Queues the latch revert of a rejected step for `GpuLanes.apply`.
    pub fn revert(o: Ops, _: anytype) void {
        o.g.ops[o.l] = 1 + @as(u8, @backingInt(StateCtlOp.revert));
    }
    /// Always null: a reject request fails the lane instead
    /// (`tran.runGpuLanes`).
    pub fn rejectAt(_: Ops, _: anytype) ?f64 {
        return null;
    }
    /// The device state-flip query `probe` ran, when `ask`.
    pub fn flipped(o: Ops, _: anytype, _: anytype, ask: bool) bool {
        return ask and o.g.query[o.l] & 1 != 0;
    }
    /// Always null: a finite bound hands the lane back (`LaneNewton.reject`).
    pub fn boundStep(_: Ops, _: anytype) ?f64 {
        return null;
    }
    /// A no-op: `probe` evaluated the round's charges.
    pub fn charges(_: Ops, _: anytype, _: anytype) void {}
    /// The LTE bound `probe` computed for `method`.
    pub fn bound(o: Ops, _: anytype, a: anytype, method: anytype, _: anytype) f64 {
        const out = o.g.lte_out[o.l];
        // `end` asks for the step's own method, or the order-2 promotion
        // probe (the only other order <= 2 call).
        return if (method == a.method) out[0] else out[1];
    }
    /// Unreachable: gear-3..6 are not lane-eligible.
    pub fn boundGear(_: Ops, _: anytype, _: anytype, _: anytype, _: anytype) f64 {
        unreachable;
    }
    /// Rotates lane l's history permutations (`queueAccept`).
    pub fn advance(o: Ops, _: anytype, _: anytype) void {
        lanes_round.queueAccept(o.g, o.l);
    }
    /// Every accepted step: the commit, and `accept`'s `prev <- cur <- x`.
    pub fn commit(o: Ops, _: anytype) void {
        o.g.ops[o.l] = 1 + @as(u8, @backingInt(StateCtlOp.commit));
        o.g.accept_[o.l] = 1;
    }
    /// Records the tape's logical slot 0 for `rebaseRound` and swaps the
    /// tape's slots 0 and 1, as `std.mem.swap` moves the host ring.
    pub fn rebase(o: Ops, s: anytype, _: anytype) void {
        if (!(s.has_charge and s.ckt.has_state_q)) return;
        const nh: usize = o.g.n_hist;
        const p = o.g.perm_qt[@as(usize, o.l) * nh ..][0..2];
        o.g.sc[o.l].sqt[3] = p[0];
        std.mem.swap(u8, &p[0], &p[1]);
    }
    /// Lane l's n-long row: only its probe positions hold the solution.
    pub fn accepted(o: Ops, _: anytype) []const f64 {
        const n: usize = o.g.n;
        return o.g.rows[@as(usize, o.l) * n ..][0..n];
    }
};

// ---- private ----
