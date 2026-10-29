//! The A(ω) fills of the frequency solves: ω, the G and C snapshots and a
//! `Dyn` in, A(ω)'s values out. `setOmegaDense` and `setOmegaSparse` write
//! one scalar fill; the lane sources compute each column as a lane LU's
//! refactor reads it, so no lane value plane exists. `Stacked` builds
//! the stacked-real [G, -ωC; ωC, G] columns, `ComplexFill` the n x n complex
//! ones, `StackedPoints` per-point G and C; each `(1)` instantiation is the
//! scalar oracle of its fill. Cites nothing beyond `freq_solve.zig`.
//!
//! Contents:
//! - `pub fn indexDyn(gpa: Allocator, s: *FreqSolver.Sparse, dyn: freq_solve.Dyn) !void`
//! - `pub fn setOmegaDense(n: u32, nn: u32, d: *FreqSolver.Dense, omega: f64, dyn: freq_solve.Dyn, k: usize) !void`
//! - `pub fn setOmegaSparse(n: u32, s: *FreqSolver.Sparse, omega: f64, dyn: freq_solve.Dyn, k: usize) !void`
//! - `pub fn Stacked(comptime L: usize) type`
//!     - `pub fn of(sp: anytype, omega: V, dyn: freq_solve.Dyn, base: usize, cnt: usize) @This()`
//!     - `pub inline fn load(s: @This(), col: usize, w: []V, prow: []const u32) void`
//! - `pub fn ComplexFill(comptime L: usize) type`
//!     - `pub fn of(sp: anytype, omega: V, dyn: freq_solve.Dyn, base: usize, cnt: usize) @This()`
//!     - `pub inline fn load(s: @This(), col: usize, wr: []V, wi: []V, prow: []const u32) void`
//! - `pub fn StackedPoints(comptime L: usize) type`
//!     - `pub inline fn load(s: @This(), col: usize, w: []V, prow: []const u32) void`

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

/// Groups `dyn`'s entries by source column into `dyn_ptr`/`dyn_ent`
/// with a counting sort, so each column keeps entry order. An empty
/// `dyn` leaves them alone: `Stacked.of` ignores them then.
pub fn indexDyn(gpa: Allocator, s: *FreqSolver.Sparse, dyn: freq_solve.Dyn) !void {
    if (dyn.slots.len == 0) return;
    const n = s.src_col_ptr.len - 1;
    const nnz = s.src_col_ptr[n];
    if (s.dyn_ptr.len != n + 1) {
        gpa.free(s.dyn_ptr);
        s.dyn_ptr = &.{};
        s.dyn_ptr = try gpa.alloc(u32, n + 1);
    }
    const ptr = s.dyn_ptr;
    @memset(ptr, 0);
    for (dyn.slots) |slot| if (slot < nnz) {
        ptr[freq_solve.slotCol(s.src_col_ptr, slot) + 1] += 1;
    };
    for (1..n + 1) |j| ptr[j] += ptr[j - 1];
    if (s.dyn_ent.len != ptr[n]) {
        gpa.free(s.dyn_ent);
        s.dyn_ent = &.{};
        s.dyn_ent = try gpa.alloc(u32, ptr[n]);
    }
    // Backwards from each column's end leaves ptr[j + 1] at column j's
    // start; one shift down restores the starts.
    var e = dyn.slots.len;
    while (e > 0) {
        e -= 1;
        const slot = dyn.slots[e];
        if (slot >= nnz) continue;
        const j = freq_solve.slotCol(s.src_col_ptr, slot);
        ptr[j + 1] -= 1;
        s.dyn_ent[ptr[j + 1]] = @intCast(e);
    }
    std.mem.copyForwards(u32, ptr[0..n], ptr[1..]);
    ptr[n] = @intCast(s.dyn_ent.len);
    assert(ptr[0] == 0);
}

/// Writes the dense 2n x 2n stacked-real A(ω) (plus `dyn` at ω index `k`)
/// into `d.a_lu`.
pub fn setOmegaDense(n: u32, nn: u32, d: *FreqSolver.Dense, omega: f64, dyn: freq_solve.Dyn, k: usize) !void {
    dense_lu.buildComplexAdmittance(n, nn, d.g_dense, d.c_mat, omega, d.a_lu);
    for (dyn.slots, 0..) |slot, e| {
        if (slot >= d.src_row_idx.len) continue;
        const t = dyn.at(e, k);
        freq_solve.addDense(n, nn, d.a_lu, d.src_row_idx[slot], freq_solve.slotCol(d.src_col_ptr, slot), t.re, t.im);
    }
    dense_lu.pinVoid(nn, d.a_lu);
    try dense_lu.factorize(nn, d.a_lu, d.piv);
}

/// Fills the 2n stacked-real values, column j < n as [G | ωC] and
/// column n + j as [-ωC | G], adds `dyn` at ω index `k`, then factors.
pub fn setOmegaSparse(n: u32, s: *FreqSolver.Sparse, omega: f64, dyn: freq_solve.Dyn, k: usize) !void {
    const nu: usize = n;
    const neg_omega = -omega;
    var p: usize = 0;

    for (0..nu) |j| {
        const cs = s.src_col_ptr[j];
        const len = s.src_col_ptr[j + 1] - cs;
        const lenu: usize = len;
        @memcpy(s.vals[p..][0..lenu], s.g_vals[cs..][0..lenu]);
        p += lenu;
        num.scale(s.vals[p..][0..lenu], omega, s.c_vals[cs..][0..lenu]);
        p += lenu;
    }
    for (0..nu) |j| {
        const cs = s.src_col_ptr[j];
        const len = s.src_col_ptr[j + 1] - cs;
        const lenu: usize = len;
        num.scale(s.vals[p..][0..lenu], neg_omega, s.c_vals[cs..][0..lenu]);
        p += lenu;
        @memcpy(s.vals[p..][0..lenu], s.g_vals[cs..][0..lenu]);
        p += lenu;
    }
    for (dyn.slots, 0..) |slot, e| {
        if (slot >= s.g_vals.len) continue;
        const q = freq_solve.stackedPos(s.src_col_ptr, slot);
        const t = dyn.at(e, k);
        s.vals[q[0]] += t.re;
        s.vals[q[1]] += t.im;
        s.vals[q[2]] -= t.im;
        s.vals[q[3]] += t.re;
    }
    assert(p == s.vals.len);

    try s.slv.factor(s.vals, .{});
}

/// The stacked-real A(ω) = [G, -ωC; ωC, G] (plus `Dyn`) for L frequencies,
/// as a `LaneLu(L).refactor` source: each column is computed from the G and
/// C snapshots as the refactor reads it, so no lane value plane exists. Same
/// products and adds, in the same order per entry, as `setOmegaSparse`, so
/// lane l is bitwise that scalar fill at ω index `base + min(l, cnt - 1)`.
/// `Stacked(1)` is the scalar oracle of the fill.
pub fn Stacked(comptime L: usize) type {
    return struct {
        const V = @Vector(L, f64);
        src_col_ptr: []const u32,
        col_ptr: []const u32,
        g: []const f64,
        c: []const f64,
        omega: V,
        dyn: freq_solve.Dyn,
        dyn_ptr: []const u32,
        dyn_ent: []const u32,
        base: usize,
        cnt: usize,

        /// The fill of `sp` at `omega`; `dyn` must be the one `indexDyn`
        /// grouped into `sp`, with lane l at ω index `base + min(l, cnt - 1)`.
        /// Asserts that 1 <= cnt <= L.
        pub fn of(sp: anytype, omega: V, dyn: freq_solve.Dyn, base: usize, cnt: usize) @This() {
            assert(cnt >= 1);
            assert(cnt <= L);
            return .{
                .src_col_ptr = sp.src_col_ptr,
                .col_ptr = sp.col_ptr,
                .g = sp.g_vals,
                .c = sp.c_vals,
                .omega = omega,
                .dyn = dyn,
                .dyn_ptr = sp.dyn_ptr,
                .dyn_ent = if (dyn.slots.len == 0) &.{} else sp.dyn_ent,
                .base = base,
                .cnt = cnt,
            };
        }

        /// Stacked column `col`: for col = j < n, G then ωC of source column
        /// j; for col = n + j, -ωC then G. Entry p lands at `w[prow[p]]`.
        pub inline fn load(s: @This(), col: usize, w: []V, prow: []const u32) void {
            const n = s.src_col_ptr.len - 1;
            const top = col < n;
            const j = if (top) col else col - n;
            const cs = s.src_col_ptr[j];
            const len: usize = s.src_col_ptr[j + 1] - cs;
            const rows = prow[s.col_ptr[col]..][0 .. 2 * len];
            const g = s.g[cs..][0..len];
            const c = s.c[cs..][0..len];
            if (top) {
                for (rows[0..len], g) |r, v| w[r] = @splat(v);
                for (rows[len..], c) |r, v| w[r] = s.omega * @as(V, @splat(v));
            } else {
                const neg = -s.omega;
                for (rows[0..len], c) |r, v| w[r] = neg * @as(V, @splat(v));
                for (rows[len..], g) |r, v| w[r] = @splat(v);
            }
            if (s.dyn_ent.len == 0) return;
            for (s.dyn_ent[s.dyn_ptr[j]..s.dyn_ptr[j + 1]]) |e| {
                var re: [L]f64 = undefined;
                var im: [L]f64 = undefined;
                for (0..L) |l| {
                    const t = s.dyn.at(e, s.base + @min(l, s.cnt - 1));
                    re[l] = t.re;
                    im[l] = t.im;
                }
                const at = s.dyn.slots[e] - cs;
                const rv: V = re;
                const iv: V = im;
                if (top) {
                    w[rows[at]] += rv;
                    w[rows[len + at]] += iv;
                } else {
                    w[rows[at]] -= iv;
                    w[rows[len + at]] += rv;
                }
            }
        }
    };
}

/// A(ω) = G + jωC (plus `Dyn`) for L frequencies on the n x n source
/// pattern, as a `ComplexLaneLu(L).refactor` source: each column is
/// computed from the G and C snapshots as the refactor reads it.
pub fn ComplexFill(comptime L: usize) type {
    return struct {
        const V = @Vector(L, f64);
        src_col_ptr: []const u32,
        g: []const f64,
        c: []const f64,
        omega: V,
        dyn: freq_solve.Dyn,
        dyn_ptr: []const u32,
        dyn_ent: []const u32,
        base: usize,
        cnt: usize,

        /// As `Stacked.of`.
        pub fn of(sp: anytype, omega: V, dyn: freq_solve.Dyn, base: usize, cnt: usize) @This() {
            assert(cnt >= 1);
            assert(cnt <= L);
            return .{
                .src_col_ptr = sp.src_col_ptr,
                .g = sp.g_vals,
                .c = sp.c_vals,
                .omega = omega,
                .dyn = dyn,
                .dyn_ptr = sp.dyn_ptr,
                .dyn_ent = if (dyn.slots.len == 0) &.{} else sp.dyn_ent,
                .base = base,
                .cnt = cnt,
            };
        }

        /// Source column `col`, entry p at row `prow[p]`.
        pub inline fn load(s: @This(), col: usize, wr: []V, wi: []V, prow: []const u32) void {
            const cs = s.src_col_ptr[col];
            const ce = s.src_col_ptr[col + 1];
            for (prow[cs..ce], s.g[cs..ce], s.c[cs..ce]) |r, gv, cv| {
                wr[r] = @splat(gv);
                wi[r] = s.omega * @as(V, @splat(cv));
            }
            if (s.dyn_ent.len == 0) return;
            for (s.dyn_ent[s.dyn_ptr[col]..s.dyn_ptr[col + 1]]) |e| {
                var re: [L]f64 = undefined;
                var im: [L]f64 = undefined;
                for (0..L) |l| {
                    const t = s.dyn.at(e, s.base + @min(l, s.cnt - 1));
                    re[l] = t.re;
                    im[l] = t.im;
                }
                const r = prow[s.dyn.slots[e]];
                wr[r] += @as(V, re);
                wi[r] += @as(V, im);
            }
        }
    };
}

/// `Stacked` over lanes from several points on one stacked pattern: lane l
/// fills from `g[l]`/`c[l]`, one point's source-slot G and C snapshot, at
/// `omega[l]`. Same products, in the same order per entry, as `Stacked`,
/// so lane l is bitwise `Stacked(1)` of its own point. No `Dyn`: a deck
/// with `acDyn` terms solves per point. The host oracle of the GPU lane
/// backend's (point, ω) fill (phase 4); every point must share the pattern
/// `src_col_ptr`/`col_ptr`.
pub fn StackedPoints(comptime L: usize) type {
    return struct {
        const V = @Vector(L, f64);
        src_col_ptr: []const u32,
        col_ptr: []const u32,
        g: [L][]const f64,
        c: [L][]const f64,
        omega: V,

        /// Stacked column `col` per lane, as `Stacked.load`.
        pub inline fn load(s: @This(), col: usize, w: []V, prow: []const u32) void {
            const n = s.src_col_ptr.len - 1;
            const top = col < n;
            const j = if (top) col else col - n;
            const cs = s.src_col_ptr[j];
            const len: usize = s.src_col_ptr[j + 1] - cs;
            const rows = prow[s.col_ptr[col]..][0 .. 2 * len];
            for (0..len) |i| {
                var ga: [L]f64 = undefined;
                var ca: [L]f64 = undefined;
                for (0..L) |l| {
                    ga[l] = s.g[l][cs + i];
                    ca[l] = s.c[l][cs + i];
                }
                const gv: V = ga;
                const cv: V = ca;
                if (top) {
                    w[rows[i]] = gv;
                    w[rows[len + i]] = s.omega * cv;
                } else {
                    w[rows[i]] = -s.omega * cv;
                    w[rows[len + i]] = gv;
                }
            }
        }
    };
}
