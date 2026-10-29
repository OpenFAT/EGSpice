//! ngspice's in-load current test for the models that keep it
//! (hfetload.c:389): each current the model drives into one of its internal
//! nodes must land within reltol m + abstol of its linear prediction from
//! the previous iterate, `i_old + J_old (x - x_old)`, where m is the
//! largest branch flow into the node (ngspice's cg or cd). The rows are the
//! ones a batch marks (`Hooks.mark_load_check_rows`), their CSC entries
//! gathered row by row once, with the workspace.
//! In: the pattern, the marked rows, and each iterate's matrix and
//! residual. Out: pass or fail per iterate.
//! Cites ngspice hfetload.c (:211-216, :389); dev/devices/models.md.
//!
//! Contents:
//! - `pub const LoadCheck = struct` (`init`, `deinit`, `check`)

const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;
const Batch = @import("device").abi.Batch;

/// The rows the test covers, their CSC entries row by row, and the last
/// iterate's current, entries and x. One allocation per slice, freed with
/// the workspace. Invariant: `ptr` has `rows.len + 1` ascending offsets
/// into `slot`, `col`, `v_old` and `x_old`.
pub const LoadCheck = struct {
    rows: []u32,
    /// Entries of row k are `ptr[k]..ptr[k + 1]`.
    ptr: []u32,
    slot: []u32,
    col: []u32,
    i_old: []f64,
    v_old: []f64,
    x_old: []f64,
    /// The previous iterate is recorded; false on a solve's first iterate.
    valid: bool = false,

    /// Returns the rows the batches mark, or null when no batch marks any.
    /// Free with `deinit` and the same allocator.
    pub fn init(gpa: Allocator, batches: []const Batch, col_ptr: []const u32, row_idx: []const u32) Allocator.Error!?LoadCheck {
        assert(col_ptr.len >= 1);
        const n = col_ptr.len - 1;
        const mask = try gpa.alloc(bool, n);
        defer gpa.free(mask);

        @memset(mask, false);
        var any = false;
        for (batches) |b| if (b.hooks.mark_load_check_rows) |mark| {
            mark(b.ctx, mask);
            any = true;
        };
        if (!any) return null;
        // Row position of each marked row, then the entries counted per row.
        const pos = try gpa.alloc(u32, n);
        defer gpa.free(pos);

        var n_rows: u32 = 0;
        for (mask, pos) |m, *p| {
            p.* = n_rows;
            n_rows += @intFromBool(m);
        }
        var lc: LoadCheck = .{ .rows = &.{}, .ptr = &.{}, .slot = &.{}, .col = &.{}, .i_old = &.{}, .v_old = &.{}, .x_old = &.{} };
        errdefer lc.deinit(gpa);

        lc.rows = try gpa.alloc(u32, n_rows);
        lc.ptr = try gpa.alloc(u32, n_rows + 1);
        for (mask, 0..) |m, r| if (m) {
            lc.rows[pos[r]] = @intCast(r);
        };
        @memset(lc.ptr, 0);
        for (row_idx) |r| if (mask[r]) {
            lc.ptr[pos[r] + 1] += 1;
        };
        for (1..lc.ptr.len) |k| lc.ptr[k] += lc.ptr[k - 1];
        const n_e = lc.ptr[n_rows];
        lc.slot = try gpa.alloc(u32, n_e);
        lc.col = try gpa.alloc(u32, n_e);
        const fill = try gpa.dupe(u32, lc.ptr[0..n_rows]);
        defer gpa.free(fill);

        for (0..n) |c| for (col_ptr[c]..col_ptr[c + 1]) |slot| {
            const r = row_idx[slot];
            if (!mask[r]) continue;
            const e = fill[pos[r]];
            fill[pos[r]] += 1;
            lc.slot[e] = @intCast(slot);
            lc.col[e] = @intCast(c);
        };
        lc.i_old = try gpa.alloc(f64, n_rows);
        lc.v_old = try gpa.alloc(f64, n_e);
        lc.x_old = try gpa.alloc(f64, n_e);
        lc.valid = false;
        return lc;
    }

    /// Frees every slice; `gpa` is the allocator `init` took.
    pub fn deinit(self: *LoadCheck, gpa: Allocator) void {
        for ([_][]u32{ self.rows, self.ptr, self.slot, self.col }) |s| gpa.free(s);
        for ([_][]f64{ self.i_old, self.v_old, self.x_old }) |s| gpa.free(s);
    }

    /// Returns true when every row's current at `x` lies within its bound.
    /// `vals`/`rhs` are the Newton matrix and residual assembled at `x`,
    /// companion terms included, so the row test sees the same currents
    /// ngspice's cg/cd carry; the flow and resistor terms are linear and
    /// cancel. False on the first iterate of a solve, which has no previous
    /// one. Records `x` as the previous iterate either way. `current_row`
    /// marks the branch unknowns.
    ///
    /// ponytail: hfet1's cdhat leaves out the -ggdpp delvgdpp its cd carries
    /// (hfetload.c:211-216), a first-order miss on every gate-drain swing
    /// that makes ngspice iterate more; a row test cannot see branches, so
    /// that quirk is not reproduced. dev/devices/models.md.
    pub fn check(self: *LoadCheck, current_row: []const bool, x: []const f64, vals: []const f64, rhs: []const f64, reltol: f64, abstol: f64) bool {
        assert(self.ptr.len == self.rows.len + 1);
        assert(x.len >= current_row.len);
        var ok = self.valid;
        for (self.rows, 0..) |r, k| {
            var pred = self.i_old[k];
            var mag: f64 = 0;
            for (self.ptr[k]..self.ptr[k + 1]) |e| {
                const c = self.col[e];
                pred += self.v_old[e] * (x[c] - self.x_old[e]);
                if (current_row[c]) mag = @max(mag, @abs(x[c]));
                self.v_old[e] = vals[self.slot[e]];
                self.x_old[e] = x[c];
            }
            const i = rhs[r];
            if (mag == 0) mag = @max(@abs(i), @abs(pred));
            if (!(@abs(i - pred) <= reltol * mag + abstol)) ok = false;
            self.i_old[k] = i;
        }
        self.valid = true;
        return ok;
    }
};
