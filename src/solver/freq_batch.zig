//! `FreqSolver.solveBatch` and its chunk providers: a batch of ω, an
//! optional `Dyn` and right-hand sides in, stacked-real solutions out. W
//! frequencies run as lanes on one pivot tape (`LaneLu` on the stacked 2n
//! pattern, or `ComplexLaneLu` on the n x n one); a lane that fails its
//! pivot test peels to the scalar path, so every lane is bitwise its own
//! scalar solve. The `begin*Chunk`/`finish*Chunk` pairs hand one chunk's
//! tape to a device provider and take its failures back.
//! Cites: dev/solvers/overview.md; #44 (the replay threshold).
//!
//! Contents:
//! - `pub const ComplexChunk = struct`
//!   - `pub fn sameTape(a: ComplexChunk, b: ComplexChunk) bool`
//! - `pub const LaneChunk = struct`
//!   - `pub fn sameTape(a: LaneChunk, b: LaneChunk) bool`
//! - `pub fn solveBatch(self: *FreqSolver, omegas: []const f64, dyn: freq_solve.Dyn, rhs: []const f64, x_out: []f64, adjoint: bool) !void`
//! - `pub fn complexBatch(self: *FreqSolver, sp: *FreqSolver.Sparse, omegas: []const f64, dyn: freq_solve.Dyn, rhs: []const f64, x_out: []f64, adjoint: bool) !usize`
//! - `pub fn complexChunkTape(self: *FreqSolver, sp: *FreqSolver.Sparse, omegas: []const f64, dyn: freq_solve.Dyn, base: usize, cnt: usize, rhs: []const f64, x_out: []f64, adjoint: bool) !?bool`
//! - `pub fn complexOwnFailure(self: *FreqSolver, fresh: bool, cnt: usize, bad: u64) bool`
//! - `pub fn complexChunkPeel(self: *FreqSolver, omegas: []const f64, dyn: freq_solve.Dyn, base: usize, cnt: usize, bad: u64, rhs: []const f64, x_out: []f64, adjoint: bool) !void`
//! - `pub fn beginComplexChunk(self: *FreqSolver, omegas: []const f64, base: usize, cnt: usize, rhs: []const f64, x_out: []f64, adjoint: bool) !?ComplexChunk`
//! - `pub fn finishComplexChunk(self: *FreqSolver, omegas: []const f64, base: usize, cnt: usize, bad: u64, fresh: bool, rhs: []const f64, x_out: []f64, adjoint: bool) !bool`
//! - `pub fn chunkTape(self: *FreqSolver, sp: *FreqSolver.Sparse, omegas: []const f64, dyn: freq_solve.Dyn, base: usize, cnt: usize, rhs: []const f64, x_out: []f64, adjoint: bool) !?*SparseLu`
//! - `pub fn chunkPeel(self: *FreqSolver, omegas: []const f64, dyn: freq_solve.Dyn, base: usize, cnt: usize, bad: u64, rhs: []const f64, x_out: []f64, adjoint: bool) !void`
//! - `pub fn beginLaneChunk(self: *FreqSolver, omegas: []const f64, base: usize, cnt: usize, rhs: []const f64, x_out: []f64, adjoint: bool) !?LaneChunk`
//! - `pub fn laneChunk(sp: *const FreqSolver.Sparse, lu: *const SparseLu) LaneChunk`
//! - `pub fn finishLaneChunk(self: *FreqSolver, omegas: []const f64, base: usize, cnt: usize, bad: u64, rhs: []const f64, x_out: []f64, adjoint: bool) !void`
//! - `pub fn chunkOmegas(omegas: []const f64, base: usize, cnt: usize) [FreqSolver.W]f64`
//! - `pub fn ensureComplex(self: *const FreqSolver, gpa: Allocator, sp: *FreqSolver.Sparse) !bool`
//! - `pub fn solveBatchSerial(self: *FreqSolver, omegas: []const f64, dyn: freq_solve.Dyn, first: usize, count: usize, rhs: []const f64, x_out: []f64, adjoint: bool) !void`

const std = @import("std");
const assert = std.debug.assert;
const dense_lu = @import("dense_lu.zig");
const direct = @import("direct.zig");
const lane_lu = @import("lane/lu.zig");
const ComplexLaneLu = @import("lane/clu.zig").ComplexLaneLu;
const order_mod = @import("order.zig");
const SparseLu = @import("sparse_lu.zig").SparseLu;
const num = @import("core").numerics;
const Allocator = std.mem.Allocator;
const freq_solve = @import("freq_solve.zig");
const FreqSolver = freq_solve.FreqSolver;
const freq_held = @import("freq_held.zig");
const freq_fill = @import("freq_fill.zig");

/// Borrowed inputs to a complex frequency lane replay (`ComplexLaneLu`
/// on the n x n proxy tape): valid until the next factor or peel on
/// this solver.
pub const ComplexChunk = struct {
    lu: *const SparseLu,
    src_col_ptr: []const u32,
    g: []const f64,
    c: []const f64,
    growth: f64,
    /// The chunk laid this tape (`finishComplexChunk` needs it).
    fresh: bool,

    /// Equal layouts and pivot operations.
    pub fn sameTape(a: ComplexChunk, b: ComplexChunk) bool {
        const x = a.lu;
        const y = b.lu;
        return a.growth == b.growth and
            std.mem.eql(u32, a.src_col_ptr, b.src_col_ptr) and
            std.mem.eql(u32, x.prow, y.prow) and
            std.mem.eql(u32, x.q, y.q) and std.mem.eql(u32, x.pinv, y.pinv) and
            std.mem.eql(u32, x.lp, y.lp) and std.mem.eql(u32, x.li.items, y.li.items) and
            std.mem.eql(u32, x.up, y.up) and std.mem.eql(u32, x.ui.items, y.ui.items) and
            std.mem.eql(bool, x.scaled_pivot, y.scaled_pivot);
    }
};

/// Borrowed inputs to a frequency lane replay. Valid until the next
/// factor/peel on this solver. Device providers retain one solver per
/// point, so grouping equal tapes never changes a point's pivot history.
pub const LaneChunk = struct {
    lu: *const SparseLu,
    src_col_ptr: []const u32,
    col_ptr: []const u32,
    row_idx: []const u32,
    g: []const f64,
    c: []const f64,
    growth: f64,

    /// Equal layouts and pivot operations, including fabricated pivots.
    pub fn sameTape(a: LaneChunk, b: LaneChunk) bool {
        const x = a.lu;
        const y = b.lu;
        return a.growth == b.growth and
            std.mem.eql(u32, a.col_ptr, b.col_ptr) and
            std.mem.eql(u32, a.row_idx, b.row_idx) and
            std.mem.eql(u32, a.src_col_ptr, b.src_col_ptr) and
            std.mem.eql(u32, x.q, y.q) and std.mem.eql(u32, x.pinv, y.pinv) and
            std.mem.eql(u32, x.lp, y.lp) and std.mem.eql(u32, x.li.items, y.li.items) and
            std.mem.eql(u32, x.up, y.up) and std.mem.eql(u32, x.ui.items, y.ui.items) and
            std.mem.eql(bool, x.void_col, y.void_col) and
            std.mem.eql(bool, x.scaled_pivot, y.scaled_pivot) and
            std.mem.eql(u32, x.void_slots.items, y.void_slots.items);
    }
};

/// Solves every ω in `omegas` against each right-hand side in `rhs`
/// (nr stacked 2n vectors), with `dyn`'s terms added to each A(ω), W
/// frequencies per LaneLu pass;
/// `x_out[(k*nr + r)*2n..][0..2n]` receives ω_k for rhs r, so one
/// factorization serves every rhs. `adjoint` solves the transpose of the
/// stacked-real system, A^H = G^T − jωC^T, not A^T: conjugate the
/// solution for A^T's (|x| is the same either way). The dense
/// strategy, a non-LU engine and any lane whose refactor fails take the
/// per-ω scalar path. The scalar factorization afterwards holds some ω
/// of the batch. Allocates the lane planes on first use. Fails with
/// `error.Singular` or `error.SingularMatrix` when an ω the scalar path
/// takes is singular, with some ω's rows of `x_out` already written.
/// Asserts that `rhs` is a positive multiple of 2n long, that `x_out`
/// holds `omegas.len` times that, and that `dyn` has one term per entry
/// per ω.
pub fn solveBatch(self: *FreqSolver, omegas: []const f64, dyn: freq_solve.Dyn, rhs: []const f64, x_out: []f64, adjoint: bool) !void {
    self.held_location = .invalid;
    const nn: usize = self.nn;
    const m = rhs.len;
    assert(m > 0);
    assert(m % nn == 0);
    assert(x_out.len == omegas.len * m);
    if (omegas.len == 0) return;
    assert(dyn.re.len == dyn.slots.len * omegas.len);
    assert(dyn.im.len == dyn.re.len);
    const sp: *FreqSolver.Sparse = switch (self.strategy) {
        .sp => |*s| s,
        .dense => return solveBatchSerial(self, omegas, dyn, 0, omegas.len, rhs, x_out, adjoint),
    };

    const gpa = sp.slv.gpa;
    const LL = lane_lu.LaneLu(FreqSolver.W);
    try freq_fill.indexDyn(gpa, sp, dyn);
    var base: usize = if (self.complex_lanes) try complexBatch(self, sp, omegas, dyn, rhs, x_out, adjoint) else 0;
    if (base == omegas.len) return;
    try freq_held.ensureLaneWork(gpa, sp, nn);
    const b_plane = sp.lane_work[0..nn];
    const x_plane = sp.lane_work[nn..];

    while (base < omegas.len) : (base += FreqSolver.W) {
        const cnt = @min(FreqSolver.W, omegas.len - base);
        const omega_vec: @Vector(FreqSolver.W, f64) = chunkOmegas(omegas, base, cnt);

        // The lanes replay the current pivot sequence; only the first
        // chunk, or one after a failed factor, pays a scalar factor.
        // A lane whose pivots decay fails the scalar growth monitor,
        // and its serial full factor repivots for the next chunk.
        // A singular centre point falls back to the serial path; an
        // allocation failure is the caller's, not a numerical one (#83).
        const lu = (try chunkTape(self, sp, omegas, dyn, base, cnt, rhs, x_out, adjoint)) orelse continue;

        // A full factor can change the L/U lengths; rebuild then.
        if (sp.lanes) |*l| {
            if (l.lx.len != lu.lx.items.len or l.ux.len != lu.ux.items.len) {
                l.deinit(gpa);
                sp.lanes = null;
            }
        }
        if (sp.lanes == null) sp.lanes = try LL.init(gpa, lu);
        sp.lanes.?.base = lu;

        const src = freq_fill.Stacked(FreqSolver.W).of(sp, omega_vec, dyn, base, cnt);
        const bad = sp.lanes.?.refactor(src, sp.slv.params.refactor_growth_limit);

        var r: usize = 0;
        while (r < m) : (r += nn) {
            for (rhs[r..][0..nn], b_plane) |value, *lane| lane.* = @splat(value);
            if (adjoint) sp.lanes.?.solveT(b_plane, x_plane) else sp.lanes.?.solve(b_plane, x_plane);
            for (0..cnt) |l| {
                if ((bad >> @intCast(l)) & 1 != 0) continue;
                const dst = x_out[(base + l) * m + r ..][0..nn];
                for (0..nn) |i| {
                    const row: [FreqSolver.W]f64 = x_plane[i];
                    dst[i] = row[l];
                }
            }
        }
        // After every lane solve: a failed lane's full factor may
        // repivot the tape the lanes replay.
        try chunkPeel(self, omegas, dyn, base, cnt, bad, rhs, x_out, adjoint);
    }
}

/// `solveBatch` on the complex lanes, chunk by chunk from the first ω.
/// Returns where the stacked lanes take over: `omegas.len` when every
/// chunk is solved, earlier once the proxy tape fails at its own ω,
/// which turns the complex lanes off for good (`complex_lanes`).
pub fn complexBatch(self: *FreqSolver, sp: *FreqSolver.Sparse, omegas: []const f64, dyn: freq_solve.Dyn, rhs: []const f64, x_out: []f64, adjoint: bool) !usize {
    const gpa = sp.slv.gpa;
    const n: usize = self.n;
    const nn: usize = self.nn;
    const m = rhs.len;
    if (!try ensureComplex(self, gpa, sp)) {
        self.complex_lanes = false;
        return 0;
    }
    const cx = &sp.cx.?;
    const br = cx.work[0..n];
    const bi = cx.work[n..][0..n];
    const xr = cx.work[2 * n ..][0..n];
    const xi = cx.work[3 * n ..][0..n];
    var base: usize = 0;
    while (base < omegas.len) : (base += FreqSolver.W) {
        const cnt = @min(FreqSolver.W, omegas.len - base);
        const fresh = (try complexChunkTape(self, sp, omegas, dyn, base, cnt, rhs, x_out, adjoint)) orelse continue;
        if (cx.lanes) |*l| if (!l.fits()) {
            l.deinit(gpa);
            cx.lanes = null;
        };
        if (cx.lanes == null) cx.lanes = try ComplexLaneLu(FreqSolver.W).init(gpa, &cx.lu);
        const lanes = &cx.lanes.?;
        const bad = lanes.refactor(freq_fill.ComplexFill(FreqSolver.W).of(sp, chunkOmegas(omegas, base, cnt), dyn, base, cnt), sp.slv.params.refactor_growth_limit);
        if (complexOwnFailure(self, fresh, cnt, bad)) return base;
        var r: usize = 0;
        while (r < m) : (r += nn) {
            for (rhs[r..][0..n], rhs[r + n ..][0..n], br, bi) |a, b, *va, *vb| {
                va.* = @splat(a);
                vb.* = @splat(b);
            }
            if (adjoint) lanes.solveH(br, bi, xr, xi) else lanes.solve(br, bi, xr, xi);
            for (xr, xi, 0..) |vr, vi, i| {
                const ar: [FreqSolver.W]f64 = vr;
                const ai: [FreqSolver.W]f64 = vi;
                for (0..cnt) |l| {
                    if ((bad >> @intCast(l)) & 1 != 0) continue;
                    const dst = x_out[(base + l) * m + r ..][0..nn];
                    dst[i] = ar[l];
                    dst[n + i] = ai[l];
                }
            }
        }
        try complexChunkPeel(self, omegas, dyn, base, cnt, bad, rhs, x_out, adjoint);
    }
    return omegas.len;
}

/// `complexBatch`'s chunk head: lays the proxy tape at the chunk's
/// centre ω when none is current. Returns whether it is fresh, or null
/// when the proxy is singular and the chunk took the serial path, its
/// rows of `x_out` written. Asserts that `sp.cx` exists.
pub fn complexChunkTape(self: *FreqSolver, sp: *FreqSolver.Sparse, omegas: []const f64, dyn: freq_solve.Dyn, base: usize, cnt: usize, rhs: []const f64, x_out: []f64, adjoint: bool) !?bool {
    const cx = &sp.cx.?;
    if (cx.tape) return false;
    // The tape: a real factor of the moduli at the centre ω.
    const centre = base + cnt / 2;
    const g = sp.g_vals;
    const c = sp.c_vals;
    for (cx.proxy, g, c) |*v, gv, cv| v.* = std.math.hypot(gv, omegas[centre] * cv);
    if (dyn.slots.len != 0) {
        // Add dyn as complex terms, then take the modulus again.
        for (dyn.slots, 0..) |slot, e| if (slot < g.len) {
            const t = dyn.at(e, centre);
            cx.proxy[slot] = std.math.hypot(g[slot] + t.re, omegas[centre] * c[slot] + t.im);
        };
    }
    cx.lu.factor(sp.slv.gpa, sp.src_col_ptr, sp.src_row_idx, cx.proxy, direct.pivot_tol) catch |err| switch (err) {
        error.OutOfMemory => return err,
        error.SingularMatrix => {
            try solveBatchSerial(self, omegas, dyn, base, cnt, rhs, x_out, adjoint);
            return null;
        },
    };
    cx.tape = true;
    return true;
}

/// True when a fresh tape's lanes fail at its own centre ω: a matrix
/// whose pivoting the moduli do not predict. The complex lanes turn off
/// for good and the stacked lanes redo the chunk from `base`.
pub fn complexOwnFailure(self: *FreqSolver, fresh: bool, cnt: usize, bad: u64) bool {
    if (!fresh or (bad >> @intCast(cnt / 2)) & 1 == 0) return false;
    self.strategy.sp.cx.?.tape = false;
    self.complex_lanes = false;
    return true;
}

/// `complexBatch`'s chunk tail: the lanes `bad` marks solve on the
/// serial path in lane order, and the next chunk lays a fresh tape.
pub fn complexChunkPeel(self: *FreqSolver, omegas: []const f64, dyn: freq_solve.Dyn, base: usize, cnt: usize, bad: u64, rhs: []const f64, x_out: []f64, adjoint: bool) !void {
    if (bad == 0) return;
    self.strategy.sp.cx.?.tape = false;
    try chunkPeel(self, omegas, dyn, base, cnt, bad, rhs, x_out, adjoint);
}

/// Starts one `complexBatch` chunk without `Dyn` terms for a device
/// provider: the same tape decisions as the host's complex lanes, so a
/// replay that matches `ComplexLaneLu`'s arithmetic is bitwise the
/// host's. Null when the chunk is not the complex lanes': it was solved
/// serially (a singular proxy) when `complex_lanes` is still set, else
/// the complex lanes are off and the caller takes `beginLaneChunk`.
/// Asserts that `complex_lanes` is set. The caller solves every lane,
/// then calls `finishComplexChunk` before this point's next chunk.
pub fn beginComplexChunk(self: *FreqSolver, omegas: []const f64, base: usize, cnt: usize, rhs: []const f64, x_out: []f64, adjoint: bool) !?ComplexChunk {
    self.held_location = .invalid;
    assert(self.complex_lanes);
    assert(cnt > 0 and cnt <= FreqSolver.W and base + cnt <= omegas.len);
    assert(rhs.len > 0 and rhs.len % self.nn == 0);
    assert(x_out.len == omegas.len * rhs.len);
    const sp = switch (self.strategy) {
        .sp => |*s| s,
        .dense => {
            self.complex_lanes = false;
            return null;
        },
    };
    if (!try ensureComplex(self, sp.slv.gpa, sp)) {
        self.complex_lanes = false;
        return null;
    }
    const fresh = (try complexChunkTape(self, sp, omegas, .{}, base, cnt, rhs, x_out, adjoint)) orelse return null;
    const lu = &sp.cx.?.lu;
    // The host lanes fail every lane on a fabricated pivot, the
    // centre's included, and only a fresh tape can carry one.
    if (std.mem.indexOfScalar(bool, lu.void_col, true) != null) {
        assert(fresh);
        const off = complexOwnFailure(self, fresh, cnt, std.math.maxInt(u64));
        assert(off);
        return null;
    }
    return .{ .lu = lu, .src_col_ptr = sp.src_col_ptr, .g = sp.g_vals, .c = sp.c_vals, .growth = sp.slv.params.refactor_growth_limit, .fresh = fresh };
}

/// Ends a device provider's complex chunk as `complexBatch` does.
/// False when the fresh tape failed at its own centre ω: the complex
/// lanes are off and the chunk's rows are void; the caller redoes it
/// from `base` with `beginLaneChunk`. Else the failed lanes peel.
pub fn finishComplexChunk(self: *FreqSolver, omegas: []const f64, base: usize, cnt: usize, bad: u64, fresh: bool, rhs: []const f64, x_out: []f64, adjoint: bool) !bool {
    assert(cnt > 0 and cnt <= FreqSolver.W and base + cnt <= omegas.len);
    if (complexOwnFailure(self, fresh, cnt, bad)) return false;
    try complexChunkPeel(self, omegas, .{}, base, cnt, bad, rhs, x_out, adjoint);
    return true;
}

/// `solveBatch`'s chunk head: the pivot tape lanes `[base, base + cnt)`
/// replay, laid by a scalar factor at the chunk's centre ω when none is
/// current. Null when the chunk took the serial path instead (a
/// singular centre or a non-LU engine), its rows of `x_out` written. An
/// allocation failure is the caller's, not a numerical one (#83). The
/// GPU lane backend runs it per point per round.
pub fn chunkTape(self: *FreqSolver, sp: *FreqSolver.Sparse, omegas: []const f64, dyn: freq_solve.Dyn, base: usize, cnt: usize, rhs: []const f64, x_out: []f64, adjoint: bool) !?*SparseLu {
    const centre = base + cnt / 2;
    if (!sp.slv.factored) freq_fill.setOmegaSparse(self.n, sp, omegas[centre], dyn, centre) catch |err| switch (err) {
        error.OutOfMemory => return err,
        else => { // else: a numerical factor failure takes the serial path
            try solveBatchSerial(self, omegas, dyn, base, cnt, rhs, x_out, adjoint);
            return null;
        },
    };
    // LaneLu replays SparseLu only, not the tridiagonal engine.
    return sp.slv.sparse() orelse {
        try solveBatchSerial(self, omegas, dyn, base, cnt, rhs, x_out, adjoint);
        return null;
    };
}

/// `solveBatch`'s chunk tail: each lane `bad` marks solves on the serial
/// path, in lane order, after every lane solve: a failed lane's full
/// factor may repivot the tape the next chunk replays.
pub fn chunkPeel(self: *FreqSolver, omegas: []const f64, dyn: freq_solve.Dyn, base: usize, cnt: usize, bad: u64, rhs: []const f64, x_out: []f64, adjoint: bool) !void {
    for (0..cnt) |l| if ((bad >> @intCast(l)) & 1 != 0)
        try solveBatchSerial(self, omegas, dyn, base + l, 1, rhs, x_out, adjoint);
}

/// Starts exactly one `solveBatch` chunk without frequency-dependent
/// `Dyn` terms. Null means the CPU already solved it (dense strategy,
/// unsupported replay engine, or singular centre). The caller solves
/// every returned lane, then calls `finishLaneChunk` before starting
/// this point's next chunk. The view must not outlive that call.
pub fn beginLaneChunk(self: *FreqSolver, omegas: []const f64, base: usize, cnt: usize, rhs: []const f64, x_out: []f64, adjoint: bool) !?LaneChunk {
    self.held_location = .invalid;
    assert(cnt > 0 and cnt <= FreqSolver.W and base + cnt <= omegas.len);
    assert(rhs.len > 0 and rhs.len % self.nn == 0);
    assert(x_out.len == omegas.len * rhs.len);
    const sp = switch (self.strategy) {
        .sp => |*s| s,
        .dense => {
            try solveBatchSerial(self, omegas, .{}, base, cnt, rhs, x_out, adjoint);
            return null;
        },
    };
    const lu = (try chunkTape(self, sp, omegas, .{}, base, cnt, rhs, x_out, adjoint)) orelse return null;
    // LaneLu peels every fabricated pivot; the device must make the
    // same per-point decision before it groups tapes.
    if (std.mem.indexOfScalar(bool, lu.void_col, true) != null) {
        try solveBatchSerial(self, omegas, .{}, base, cnt, rhs, x_out, adjoint);
        return null;
    }
    return laneChunk(sp, lu);
}

/// Returns the `LaneChunk` view of `sp`'s lane planes on `lu`'s tape.
pub fn laneChunk(sp: *const FreqSolver.Sparse, lu: *const SparseLu) LaneChunk {
    return .{
        .lu = lu,
        .src_col_ptr = sp.src_col_ptr,
        .col_ptr = sp.col_ptr,
        .row_idx = sp.row_idx,
        .g = sp.g_vals,
        .c = sp.c_vals,
        .growth = sp.slv.params.refactor_growth_limit,
    };
}

/// Ends a provider's chunk in the scalar oracle's order: failed
/// frequencies solve and repivot one at a time, in ascending lane order.
pub fn finishLaneChunk(self: *FreqSolver, omegas: []const f64, base: usize, cnt: usize, bad: u64, rhs: []const f64, x_out: []f64, adjoint: bool) !void {
    assert(cnt > 0 and cnt <= FreqSolver.W and base + cnt <= omegas.len);
    try chunkPeel(self, omegas, .{}, base, cnt, bad, rhs, x_out, adjoint);
}

/// The ω of lanes `[base, base + cnt)`, a ragged tail padded by repeating
/// its last ω.
pub fn chunkOmegas(omegas: []const f64, base: usize, cnt: usize) [FreqSolver.W]f64 {
    var ow: [FreqSolver.W]f64 = undefined;
    for (0..FreqSolver.W) |l| ow[l] = omegas[base + @min(l, cnt - 1)];
    return ow;
}

/// Builds `sp.cx` on first use; false when the ordering's fill cap
/// refuses it, which leaves the stacked lanes as the only path.
pub fn ensureComplex(self: *const FreqSolver, gpa: Allocator, sp: *FreqSolver.Sparse) !bool {
    if (sp.cx == null) sp.cx = try FreqSolver.Complex.init(gpa, self.n, sp.src_col_ptr, sp.src_row_idx);
    return sp.cx != null;
}

/// The lane path's oracle: `setOmega` with `dyn` and a solve per ω, over
/// `omegas[first..][0..count]`, into the `x_out` rows `solveBatch` uses.
pub fn solveBatchSerial(self: *FreqSolver, omegas: []const f64, dyn: freq_solve.Dyn, first: usize, count: usize, rhs: []const f64, x_out: []f64, adjoint: bool) !void {
    const nn: usize = self.nn;
    const m = rhs.len;
    assert(first + count <= omegas.len);
    assert(m % nn == 0);
    for (first..first + count) |k| {
        try self.setOmegaDyn(omegas[k], dyn, k);
        var r: usize = 0;
        while (r < m) : (r += nn) {
            const b = rhs[r..][0..nn];
            const x = x_out[k * m + r ..][0..nn];
            if (adjoint) self.solveRhsT(b, x) else self.solveRhs(b, x);
        }
    }
}
