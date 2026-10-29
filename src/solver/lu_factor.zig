//! The full Gilbert-Peierls factor of `SparseLu`: a CSC pattern, its values
//! and a column order in, L, U, the pivot sequence and the replay tape out.
//! Per column: DFS reach (supernodal once panels dominate), the sparse
//! triangular solve, the threshold pivot (diagonal, then the wanted row,
//! then row-scaled), and the L/U stores.
//! Cites: Gilbert and Peierls, "Sparse partial pivoting in time proportional
//! to arithmetic operations", SIAM J. Sci. Stat. Comput. 9(5), 1988; KLU's
//! threshold pivoting; ngspice Sparse 1.3 spfactor.c (ExchangeRowsAndCols);
//! dev/solvers/gilbert-peierls-lu.md; #45.
//!
//! Contents:
//! - `pub const scratch_u32 = .{ "flag", "topo", "stack", "pstack", "want", "want_of",...`
//! - `pub fn holdScratch(self: *SparseLu, gpa: Allocator) Allocator.Error!void`
//! - `pub fn dropScratch(self: *SparseLu, gpa: Allocator) void`
//! - `pub fn factorPass(...)`
//! - `pub fn voidUnknown(...)`

const std = @import("std");
const assert = std.debug.assert;
const num = @import("core").numerics;
const Allocator = std.mem.Allocator;
const sparse_lu = @import("sparse_lu.zig");
const SparseLu = sparse_lu.SparseLu;
const instrument = sparse_lu.instrument;
const lu_panel = @import("lu_panel.zig");
const lu_replay = @import("lu_replay.zig");
const lu_census = @import("lu_census.zig");

/// The full factor's own scratch, `n` per array, alive only inside
/// `factor`: a refactor or a solve never reads it (n * 44 bytes, 4.4 MB
/// at the 100k-node RC ladder).
pub const scratch_u32 = .{ "flag", "topo", "stack", "pstack", "want", "want_of", "lend", "sn_of", "sn_slot", "sn_end" };

const NONE: u32 = std.math.maxInt(u32);

/// Allocates the full factor's scratch (`scratch_u32` and `rscale`), freed
/// by `dropScratch`. Asserts that none is held; on failure nothing is.
pub fn holdScratch(self: *SparseLu, gpa: Allocator) Allocator.Error!void {
    assert(self.flag.len == 0);
    errdefer dropScratch(self, gpa);
    inline for (scratch_u32) |name| @field(self, name) = try gpa.alloc(u32, self.n);
    self.rscale = try gpa.alloc(f64, self.n);
}

/// Frees the full factor's scratch and leaves every slice empty.
pub fn dropScratch(self: *SparseLu, gpa: Allocator) void {
    inline for (scratch_u32) |name| {
        gpa.free(@field(self, name));
        @field(self, name) = &.{};
    }
    gpa.free(self.rscale);
    self.rscale = &.{};
}

/// One full factor. `keep_prev` starts each column from `prev` and
/// takes it while it passes `keep_min`; otherwise from the diagonal under the
/// threshold test. Returns true, unfactored, once L plus U exceeds `cap`.
pub fn factorPass(
    self: *SparseLu,
    gpa: Allocator,
    col_ptr: []const u32,
    row_idx: []const u32,
    vals: []const f64,
    pivot_tol: f64,
    cap: usize,
    keep_prev: bool,
) SparseLu.FactorError!bool {
    const n = self.n;
    assert(self.flag.len == n);
    assert(self.rscale.len == n);
    self.factored = false;
    self.replay_mult = null;
    @memset(self.pinv, NONE);
    @memset(self.flag, 0);
    @memset(self.void_col, false);
    @memset(self.scaled_pivot, false);
    self.tv.clearRetainingCapacity();
    self.panels.clearRetainingCapacity();
    self.panel_rows.clearRetainingCapacity();
    self.panel_vals.clearRetainingCapacity();
    self.void_slots.clearRetainingCapacity();
    var has_void = false;
    self.li.clearRetainingCapacity();
    self.lx.clearRetainingCapacity();
    self.ui.clearRetainingCapacity();
    self.ux.clearRetainingCapacity();

    // Threshold pivoting compares candidates in whatever units each
    // device wrote its row in. Record 1/max|A[r,:]| so the diagonal
    // test below can retry in a scale-free metric.
    @memset(self.rscale, 0);
    // A NaN or inf entry fails the factor: the pivot search skips NaN
    // and the solve skips zero multipliers, so one left in L or U could
    // reach x unseen. One vectorized pass, ~0.4% of a circuit factor.
    // An overflow inside the elimination is caught where L and U are
    // stored (`nonfinite`).
    if (!(num.normInf(vals[0..self.prow.len]) <= std.math.floatMax(f64))) return error.SingularMatrix;
    for (0..n) |j| {
        for (col_ptr[j]..col_ptr[j + 1]) |p|
            self.rscale[row_idx[p]] = @max(self.rscale[row_idx[p]], @abs(vals[p]));
    }
    for (self.rscale) |*s|
        s.* = if (s.* > 0 and std.math.isFinite(s.*)) 1 / s.* else 1;

    // Locals, not fields: a store through any slice may alias `self`,
    // and field access would reload the slice pointers per element.
    const pinv = self.pinv;
    const flag = self.flag;
    const stack = self.stack;
    const pstack = self.pstack;
    const topo = self.topo;
    const lp = self.lp;
    const w = self.w;
    const rscale = self.rscale;
    const lend = self.lend;
    const want = self.want;
    const want_of = self.want_of;
    // Nonzero once an L or U entry stored is NaN or inf; tested once
    // after the column loop. x - x is +0 for a finite x and NaN
    // otherwise, so OR-ing its bits costs two ops and no compare.
    var nonfinite: u64 = 0;
    for (want, self.prev, 0..) |*a, p, i| a.* = if (keep_prev) p else @intCast(i);
    for (want, 0..) |r, c| want_of[r] = @intCast(c);
    // Two copies of the column loop. The plain one runs until an L
    // column reaches lu_panel.panel_min_rows, which typical circuit matrices
    // never do. The supernodal copy adds the lend shortcut and panels.
    var k: usize = 0;
    inline for (.{ false, true }) |sn| {
        if (sn) {
            lp[k] = @intCast(self.li.items.len);
            for (lend[0..k], lp[1 .. k + 1]) |*e, end| e.* = end;
            @memset(self.sn_of, NONE);
            self.sn_dfs = false;
        }
        while (k < n) : (k += 1) {
            const c = self.q[k];
            lp[k] = @intCast(self.li.items.len);
            self.up[k] = @intCast(self.ui.items.len);
            const mark: u32 = @intCast(k + 1);
            // L may move at the reservation below; re-take it after.
            const li = self.li.items;

            // Reach of A[:,c] through the graph of L.
            var nt: u32 = 0;
            for (col_ptr[c]..col_ptr[c + 1]) |p| {
                var r = row_idx[p];
                if (flag[r] == mark) continue;
                var sp: u32 = 0;
                stack[sp] = r;
                flag[r] = mark;
                // An unpivoted row has no L column: the empty range [0, 0).
                pstack[sp] = if (sn) lu_panel.enter(self, r, sp, mark) else if (pinv[r] == NONE) 0 else lp[pinv[r]];
                // Terminates: each pass pushes a row not yet marked for
                // this column (at most n of them) or finishes the top
                // one, so it runs at most 2n times and sp stays below n.
                while (true) {
                    r = stack[sp];
                    const kc = pinv[r];
                    const in_panel = sn and kc != NONE and self.sn_end[sp] != NONE;
                    // A panel row scans the panel's rows below its block.
                    const adj = if (in_panel) self.panel_rows.items else li;
                    const end = if (kc == NONE) 0 else if (in_panel) lu_panel.panelEnd(self, kc) else if (sn) lend[kc] else lp[kc + 1];
                    // Keep the cursor in a register; spill it only on descent.
                    var pos = pstack[sp];
                    var descended = false;
                    while (pos < end) {
                        const child = adj[pos];
                        pos += 1;
                        if (flag[child] != mark) {
                            pstack[sp] = pos;
                            flag[child] = mark;
                            sp += 1;
                            stack[sp] = child;
                            pstack[sp] = if (sn) lu_panel.enter(self, child, sp, mark) else if (pinv[child] == NONE) 0 else lp[pinv[child]];
                            descended = true;
                            break;
                        }
                    }
                    if (descended) continue;
                    if (in_panel) {
                        // The covered steps finish deepest first, so
                        // the solve meets kc..e in step order.
                        const pan = self.panels.items[self.sn_of[kc]];
                        const rows = self.panel_rows.items[pan.row_off..];
                        var t = self.sn_end[sp];
                        while (t > kc) : (t -= 1) {
                            topo[nt] = rows[t - pan.first - 1];
                            nt += 1;
                        }
                    }
                    topo[nt] = r;
                    nt += 1;
                    if (sp == 0) break;
                    sp -= 1;
                }
            }

            // The reach bounds this column's U and L entries. Reserve
            // before the scatter so OutOfMemory leaves `w` zero.
            try self.ui.ensureUnusedCapacity(gpa, nt);
            try self.ux.ensureUnusedCapacity(gpa, nt);
            try self.li.ensureUnusedCapacity(gpa, nt);
            try self.lx.ensureUnusedCapacity(gpa, nt);
            const lcol = self.li.items;
            const lval = self.lx.items;

            for (col_ptr[c]..col_ptr[c + 1]) |p| w[row_idx[p]] = vals[p];

            // Triangular solve in reverse finish order.
            var idx: u32 = nt;
            while (idx > 0) {
                idx -= 1;
                const r = topo[idx];
                const kc = pinv[r];
                if (kc == NONE) continue;
                if (sn and self.panels.items.len != 0 and self.sn_of[kc] != NONE) {
                    idx = lu_panel.panelRun(self, w, topo, idx, kc);
                    continue;
                }
                const ukr = w[r];
                self.ui.appendAssumeCapacity(kc);
                self.ux.appendAssumeCapacity(ukr);
                lu_replay.scatterAxpy(w, lcol, lval, lp[kc], lp[kc + 1], ukr);
            }

            // `amax`, the raw column max, picks the off-diagonal
            // fallback and gates singularity. `smax` is the same max
            // under row scaling, the diagonal test's second chance: an
            // equation decades below the rest may still own its unknown.
            var amax: f64 = 0;
            var smax: f64 = 0;
            var piv: u32 = NONE;
            for (topo[0..nt]) |r| {
                if (pinv[r] != NONE) continue;
                const a = @abs(w[r]);
                if (a > amax) {
                    amax = a;
                    piv = r;
                }
                smax = @max(smax, a * rscale[r]);
            }
            if (piv == NONE or amax == 0 or !std.math.isFinite(amax)) {
                // Usually a singular circuit, but a compact model can
                // disable part of itself (HICUM's `V(br_sht) <+ 0` at
                // flsh = 0, the BSIM4/BSIMSOI/HiSIM self-heating nodes),
                // and Verilog-A cannot delete the node: its unknown keeps
                // an all-zero row and column. That is a free variable,
                // not a singular system. A unit pivot is sound only when
                // row c is void too: then x_c = b_c, and a nonzero b_c
                // fails the Newton residual gate instead of solving
                // silently. Without this, devices/hicum2_output paid the
                // whole continuation ladder at each of 1809 DC points.
                if (!voidUnknown(self, col_ptr, row_idx, vals, c)) {
                    // `w` still holds this column's values; restore
                    // the all-zero invariant.
                    for (topo[0..nt]) |r| w[r] = 0;
                    return error.SingularMatrix;
                }
                self.void_col[k] = true;
                has_void = true;
                self.udiag[k] = 1;
                pinv[c] = @intCast(k);
                if (sn) lend[k] = lp[k]; // empty L column
                for (topo[0..nt]) |r| w[r] = 0;
                continue;
            }
            const pr = want[c];
            if (pinv[pr] == NONE) {
                const dmag = @abs(w[pr]);
                // `dmag > 0`: a zero tolerance must not accept a zero pivot.
                if (keep_prev) {
                    if (dmag > 0 and dmag >= self.keep_min * amax) piv = pr;
                } else if (dmag > 0 and dmag >= pivot_tol * amax) {
                    piv = pr;
                } else if (dmag > 0 and dmag * rscale[pr] >= pivot_tol * smax) {
                    // The diagonal is the largest entry measured against
                    // its own equation. A BSIMSOI floating body at default
                    // junction params: its KCL row is ~1e-18 S while Gmbs
                    // ~1e-4 S sits in its column on the drain row, and
                    // pivoting there makes dx_body drain-row rounding noise
                    // (observed -13.9 V). ngspice avoids it by permuting
                    // rows and columns together (Sparse 1.3 spfactor.c
                    // ExchangeRowsAndCols), which a fixed column order
                    // cannot do.
                    // ponytail: scales the pivot choice only; full row
                    // equilibration (KLU scale=2) if the arithmetic needs it.
                    piv = pr;
                    self.scaled_pivot[k] = true;
                }
            }
            if (piv != pr) {
                // Row piv was column want_of[piv]'s first choice; hand
                // it the row c gave up. Without this, each off-diagonal
                // pivot also cost the column whose diagonal it took, and
                // that one another: 1,811 of the 2,654 off-diagonal
                // pivots in one chain_bsim4_10k re-pivot were such
                // cascades. The same 2x2 exchange ngspice's Sparse gets
                // from ExchangeRowsAndCols; KLU has no equivalent.
                const c2 = want_of[piv];
                want[c2] = pr;
                want_of[pr] = c2;
            }
            const d = w[piv];
            self.udiag[k] = d;
            pinv[piv] = @intCast(k);

            // L[:,k] is the unpivoted reach over the pivot; clear w. A
            // pivoted row's w still holds its U entry (`panelRun`
            // writes its rows back), so this one pass flags U and L.
            for (topo[0..nt]) |r| {
                const v = w[r];
                nonfinite |= @as(u64, @bitCast(v - v));
                if (pinv[r] == NONE) {
                    const l = v / d;
                    nonfinite |= @as(u64, @bitCast(l - l));
                    self.li.appendAssumeCapacity(r);
                    self.lx.appendAssumeCapacity(l);
                }
                w[r] = 0;
            }
            if (self.li.items.len + self.ui.items.len > cap) {
                assert(!self.factored);
                return true;
            }
            if (!sn) {
                if (self.li.items.len - lp[k] >= lu_panel.panel_min_rows) {
                    k += 1;
                    break;
                }
                continue;
            }
            lend[k] = @intCast(self.li.items.len);
            // Short columns gain nothing from either shortcut.
            if (k > 0 and (lp[k] - lp[k - 1] >= lu_panel.panel_min_rows or self.sn_of[k - 1] != NONE))
                try lu_panel.linkStep(self, gpa, @intCast(k), piv, mark);
            // Panels hold most of L so far: a mesh. From here on the
            // walk visits each panel run once (`enter`).
            if (self.panel_vals.items.len >= self.li.items.len / 5 * 4) self.sn_dfs = true;
        }
    }
    self.lp[n] = @intCast(self.li.items.len);
    self.up[n] = @intCast(self.ui.items.len);
    // Every row pivoted exactly once: k ran to n and each step set one.
    assert(k == n);
    if (nonfinite != 0) return error.SingularMatrix;

    for (self.li.items) |*r| r.* = self.pinv[r.*];
    for (row_idx[0..self.prow.len], self.prow) |r, *pr| pr.* = self.pinv[r];

    if (has_void) {
        var count: usize = 0;
        for (self.q, 0..) |c, j| {
            for (col_ptr[c]..col_ptr[c + 1]) |p|
                count += @intFromBool(self.void_col[j] or self.void_col[self.prow[p]]);
        }
        try self.void_slots.ensureTotalCapacityPrecise(gpa, count);
        for (self.q, 0..) |c, j| {
            for (col_ptr[c]..col_ptr[c + 1]) |p| {
                if (self.void_col[j] or self.void_col[self.prow[p]])
                    self.void_slots.appendAssumeCapacity(@intCast(p));
            }
        }
    }

    try lu_replay.buildTape(self, gpa, col_ptr);

    // ZP_LU_STATS prints n, nnz and fill per full factor, then the
    // structural census of dev/solvers/gpu-lu.md E1.
    if (instrument and self.stats) {
        std.debug.print("lu-stats: n={d} nnz={d} L={d} U={d} fill={d:.1}x\n", .{
            n,                 col_ptr[n],
            self.li.items.len, self.ui.items.len,
            @as(f64, @floatFromInt(self.li.items.len + self.ui.items.len)) /
                @as(f64, @floatFromInt(col_ptr[n])),
        });
        lu_census.census(self, gpa, col_ptr, row_idx) catch |e| std.debug.print("lu-census: {s}\n", .{@errorName(e)});
    }

    assert(self.lx.items.len == self.li.items.len);
    assert(self.ux.items.len == self.ui.items.len);
    self.pattern_epoch +%= 1;
    self.factored = true;
    return false;
}

/// True when unpivoted unknown `c` appears in no equation and
/// equation `c` holds no unknown. Zero-valued structural entries count
/// as absent: a disabled branch keeps its slots, only its values
/// vanish. O(nnz); runs only on a failed pivot.
pub fn voidUnknown(
    self: *const SparseLu,
    col_ptr: []const u32,
    row_idx: []const u32,
    vals: []const f64,
    c: u32,
) bool {
    assert(c < self.n);
    if (self.pinv[c] != NONE) return false;
    for (col_ptr[c]..col_ptr[c + 1]) |p| {
        if (vals[p] != 0) return false;
    }
    for (0..self.n) |j| {
        if (j == c) continue;
        for (col_ptr[j]..col_ptr[j + 1]) |p| {
            if (row_idx[p] == c and vals[p] != 0) return false;
        }
    }
    return true;
}
