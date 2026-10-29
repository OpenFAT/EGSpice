//! The full factor's supernode panels: a column's L pattern and values in,
//! dense column-major panels and their vector updates out. Every update
//! subtracts per row in step order, so the factor's bits are the plain
//! column loop's. Cites: Demmel, Eisenstat, Gilbert, Li and Liu, "A
//! supernodal approach to sparse partial pivoting", SIAM J. Matrix Anal.
//! Appl. 20(3), 1999 (SuperLU's supernodes); dev/solvers/gilbert-peierls-lu.md.
//!
//! Contents:
//! - `pub const panel_min_rows = 32`
//! - `pub fn enter(self: *SparseLu, r: u32, sp: u32, mark: u32) u32`
//! - `pub fn panelEnd(self: *const SparseLu, kc: u32) u32`
//! - `pub fn linkStep(self: *SparseLu, gpa: Allocator, k: u32, piv: u32, mark: u32) Allocator.Error!void`
//! - `pub fn growPanel(self: *SparseLu, gpa: Allocator, k: u32, piv: u32) Allocator.Error!void`
//! - `pub fn panelRun(self: *SparseLu, w: []f64, topo: []const u32, idx0: u32, kc: u32) u32`
//! - `pub inline fn subScaled(x: []f64, a: []const f64, f: f64) void`
//! - `pub inline fn panelRows(comptime W: usize, w: []f64, rows: []const u32, vals: []const f64, nrows: u32, i: u32, u: []const f64) void`

const std = @import("std");
const assert = std.debug.assert;
const num = @import("core").numerics;
const Allocator = std.mem.Allocator;
const sparse_lu = @import("sparse_lu.zig");
const SparseLu = sparse_lu.SparseLu;
const instrument = sparse_lu.instrument;

/// A panel opens only on an L column at least this long, and the
/// factor switches to its supernodal loop only after one appears.
/// Short supernodes (a complex AC entry's 2x2 real block) lose to
/// the scalar path: 16 and 8 cost 0.5% and 13% more Ir than 32 on
/// the 100x100 grid and the sweep_opamp AC matrix respectively.
pub const panel_min_rows = 32;

const NONE: u32 = std.math.maxInt(u32);

/// The supernodal DFS's entry to row `r`, stacked at `sp`: returns its
/// scan start. A row whose step kc lies in a panel [first, last]
/// reaches every later panel step, whose rows are the block rows, and
/// through `last` the rows below the block (L[:,first] is the block
/// rows plus L[:,last]). The visit marks the steps kc+1..e up to the
/// first already reached (that step was finished with its own reach:
/// it is no ancestor, its step being above kc), records e in
/// `sn_end`, and scans the rows below the block only when e is
/// `last`. One visit then covers the run instead of a walk per step.
/// Not the plain walk's order, so the factor's rounding can move: only
/// a factor whose panels hold most of L takes it (`sn_dfs`).
pub fn enter(self: *SparseLu, r: u32, sp: u32, mark: u32) u32 {
    const kc = self.pinv[r];
    self.sn_end[sp] = NONE;
    if (kc == NONE) return 0;
    const id = self.sn_of[kc];
    if (id == NONE or !self.sn_dfs) return self.lp[kc];
    const pan = self.panels.items[id];
    const rows = self.panel_rows.items[pan.row_off..];
    var e = kc;
    while (e < pan.last and self.flag[rows[e - pan.first]] != mark) : (e += 1)
        self.flag[rows[e - pan.first]] = mark;
    self.sn_end[sp] = e;
    const m = pan.last - pan.first + 1;
    return if (e == pan.last) pan.row_off + m - 1 else panelEnd(self, kc);
}

/// End of the rows below the block of step kc's panel, in `panel_rows`.
pub fn panelEnd(self: *const SparseLu, kc: u32) u32 {
    const pan = self.panels.items[self.sn_of[kc]];
    return pan.row_off + self.lp[pan.first + 1] - self.lp[pan.first];
}

/// Step k just stored L[:,k] and pivoted row `piv`. Compares L[:,k-1]
/// with it: sets lend[k-1], and grows a panel when the two columns
/// form a supernode. A row of L[:,k] is exactly a row this column's
/// walk reached (flag == mark) that is still unpivoted.
pub fn linkStep(self: *SparseLu, gpa: Allocator, k: u32, piv: u32, mark: u32) Allocator.Error!void {
    assert(k > 0);
    assert(k < self.n);
    const s = k - 1;
    const col = self.li.items[self.lp[s]..self.lp[k]];
    var at: u32 = NONE; // piv's offset in L[:,s]
    var last_out: u32 = NONE; // last offset whose row is not in L[:,k]
    for (col, 0..) |r, i| {
        if (r == piv) {
            at = @intCast(i);
        } else if (self.flag[r] != mark or self.pinv[r] != NONE) {
            last_out = @intCast(i);
        }
    }
    if (at == NONE or (last_out != NONE and last_out > at)) return;
    self.lend[s] = self.lp[s] + at + 1;
    const joins = last_out == NONE and col.len == self.li.items.len - self.lp[k] + 1;
    if (joins) try growPanel(self, gpa, k, piv);
}

/// Step k joined step k-1's supernode; `piv` is the row it pivoted.
/// Opens a panel at k-1 if none is open, moves `piv` to block slot
/// k-first-1 and appends column k.
pub fn growPanel(self: *SparseLu, gpa: Allocator, k: u32, piv: u32) Allocator.Error!void {
    assert(k > 0);
    assert(self.sn_of[k] == NONE);
    const lp = self.lp;
    const li = self.li.items;
    const lx = self.lx.items;
    const slot = self.sn_slot;
    if (self.sn_of[k - 1] == NONE) {
        const first = k - 1;
        const nrows = lp[k] - lp[first];
        try self.panels.ensureUnusedCapacity(gpa, 1);
        try self.panel_rows.ensureUnusedCapacity(gpa, nrows);
        try self.panel_vals.ensureUnusedCapacity(gpa, nrows);
        self.sn_of[first] = @intCast(self.panels.items.len);
        self.panels.appendAssumeCapacity(.{
            .first = first,
            .last = first,
            .row_off = @intCast(self.panel_rows.items.len),
            .val_off = @intCast(self.panel_vals.items.len),
        });
        for (li[lp[first]..lp[k]], lx[lp[first]..lp[k]], 0..) |r, v, i| {
            slot[r] = @intCast(i);
            self.panel_rows.appendAssumeCapacity(r);
            self.panel_vals.appendAssumeCapacity(v);
        }
    }
    const pid = self.sn_of[k - 1];
    const pan = &self.panels.items[pid];
    const nrows = lp[pan.first + 1] - lp[pan.first];
    try self.panel_vals.ensureUnusedCapacity(gpa, nrows);
    const rows = self.panel_rows.items[pan.row_off..][0..nrows];
    const b = k - pan.first - 1; // piv's block slot
    const a = slot[piv];
    if (a != b) {
        const vals = self.panel_vals.items[pan.val_off..];
        for (0..k - pan.first) |col| std.mem.swap(f64, &vals[col * nrows + a], &vals[col * nrows + b]);
        slot[rows[b]] = a;
        slot[piv] = b;
        std.mem.swap(u32, &rows[a], &rows[b]);
    }
    const dst = self.panel_vals.addManyAsSliceAssumeCapacity(nrows);
    @memset(dst, 0);
    // lp[k + 1] is not written yet: column k ends at the list's end.
    for (li[lp[k]..], lx[lp[k]..]) |r, v| dst[slot[r]] = v;
    assert(pan.last + 1 == k);
    pan.last = k;
    self.sn_of[k] = pid;
}

/// Applies the run of panel steps that starts at topo[idx0] (step kc)
/// and continues while the next topo entry is the panel's next step.
/// Returns the topo index of the run's last entry. Per row the
/// subtractions happen in step order, so the result is bitwise the
/// column-at-a-time loop's.
pub fn panelRun(self: *SparseLu, w: []f64, topo: []const u32, idx0: u32, kc: u32) u32 {
    assert(self.sn_of[kc] != NONE);
    const pan = self.panels.items[self.sn_of[kc]];
    const nrows = self.lp[pan.first + 1] - self.lp[pan.first];
    const m = pan.last - pan.first + 1;
    const rows = self.panel_rows.items[pan.row_off..][0..nrows];
    const vals = self.panel_vals.items[pan.val_off..];
    var len: u32 = 1;
    while (kc + len <= pan.last and len <= idx0 and self.pinv[topo[idx0 - len]] == kc + len) len += 1;

    // Block rows of steps kc+1..last sit in slots b0..m-2. Copy them
    // to a dense scratch (`y` is free during factor) so the triangular
    // part runs as contiguous axpys.
    const b0 = kc - pan.first;
    const x = self.y[0 .. m - 1 - b0];
    for (x, rows[b0 .. m - 1]) |*xi, r| xi.* = w[r];
    const run0 = self.ux.items.len;
    for (0..len) |j| {
        const t: u32 = kc + @as(u32, @intCast(j));
        const ut = if (j == 0) w[topo[idx0]] else x[j - 1];
        self.ui.appendAssumeCapacity(t);
        self.ux.appendAssumeCapacity(ut);
        const col = vals[(t - pan.first) * nrows ..];
        subScaled(x[j..], col[b0 + j .. m - 1], ut);
    }
    for (x, rows[b0 .. m - 1]) |xi, r| w[r] = xi;

    const u = self.ux.items[run0..];
    const c0 = b0 * nrows;
    const W = comptime std.simd.suggestVectorLength(f64) orelse 1;
    var i: u32 = m - 1;
    while (i + W <= nrows) : (i += W) panelRows(W, w, rows, vals[c0..], nrows, i, u);
    while (i < nrows) : (i += 1) panelRows(1, w, rows, vals[c0..], nrows, i, u);
    assert(len <= m - b0);
    return idx0 - (len - 1);
}

/// x[i] -= a[i] * f; LLVM vectorizes it behind a runtime alias check.
pub inline fn subScaled(x: []f64, a: []const f64, f: f64) void {
    for (x, a[0..x.len]) |*xi, ai| xi.* -= ai * f;
}

/// w[rows[i..i+W]] -= sum over j of vals[j*nrows + i..][0..W] * u[j],
/// subtracted one j at a time. W == 1 is the scalar oracle.
pub inline fn panelRows(comptime W: usize, w: []f64, rows: []const u32, vals: []const f64, nrows: u32, i: u32, u: []const f64) void {
    const V = @Vector(W, f64);
    var acc: V = undefined;
    inline for (0..W) |l| acc[l] = w[rows[i + l]];
    for (u, 0..) |uj, j| {
        const lv: V = vals[j * nrows + i ..][0..W].*;
        acc -= lv * @as(V, @splat(uj));
    }
    inline for (0..W) |l| w[rows[i + l]] = acc[l];
}
