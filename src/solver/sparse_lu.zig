//! Left-looking Gilbert-Peierls sparse LU with threshold partial pivoting,
//! a numeric refactor that replays the frozen pattern and pivot sequence,
//! and forward/transpose solves. The caller supplies the column ordering.
//! In: a CSC pattern, its values and a column order. Out: L, U and the
//! pivot sequence, and solutions of A x = b and A^T x = b.
//! Background: dev/solvers/gilbert-peierls-lu.md, klu-pipeline.md,
//! refactor-tape-2026-09.md, gpu-lu.md (the census); KLU's threshold
//! pivoting; ngspice Sparse 1.3 spfactor.c (ExchangeRowsAndCols, the row
//! exchange the scaled pivot and the want hand-off stand in for); #45.
//!
//! Contents:
//! - `pub const instrument = @import("build_options").lu_instrument`
//! - `pub const SparseLu = struct`
//!   - `pub const FactorError = error`
//!   - `pub fn init(...)`
//!   - `pub fn deinit(self: *Self, gpa: Allocator) void`
//!   - `pub fn factor(...)`
//!   - `pub const test_access = if (@import("builtin").is_test) .{`
//!   - `pub fn refactor(...)`
//!   - `pub const Replay = lu_replay.Replay`
//!   - `pub fn replay(self: *Self, col_ptr: []const u32, vals: []const f64, growth_limit: f64) Replay`
//!   - `pub fn solve(self: *Self, b: []const f64, x: []f64) void`
//!   - `pub fn solveT(self: *Self, b: []const f64, x: []f64) void`
//!   - `pub fn init(gpa: Allocator, r: std.Random, n: u32, density: f64) !TestSystem`
//!   - `pub fn deinit(self: TestSystem, gpa: Allocator) void`
//!   - `pub fn residual(self: TestSystem, vals: []const f64, x: []const f64, b: []const f64, comptime transpose: bool) f64`
//!   - `pub const col_ptr = [_]u32{ 0, 2, 3, 6 }`
//!   - `pub const row_idx = [_]u32{ 0, 1, 1, 0, 1, 2 }`
//!   - `pub const q = [_]u32{ 0, 1, 2 }`
//!   - `pub const benign_vals = [_]f64{ 1, 1, 1, 1, 1, 1 }`
//!   - `pub const overflow_vals = [_]f64{ 1, 1, 1, 1.5e308, -1.5e308, 1 }`

const std = @import("std");
const assert = std.debug.assert;
const num = @import("core").numerics;
const lu_factor = @import("lu_factor.zig");
const lu_replay = @import("lu_replay.zig");
/// `-Dlu-instrument`: compiles the `ZP_LU_STATS` fill print and census.
pub const instrument = @import("build_options").lu_instrument;
const Allocator = std.mem.Allocator;
const testing = std.testing;

/// Sparse LU. Arrays are SoA; the DFS workspace is touched
/// only by a full `factor`.
pub const SparseLu = struct {
    const Self = @This();
    /// Private kernels for src/solver/tests.zig; empty outside test builds.
    // The column replay, for this file's tests (`lu_replay.zig`).
    const refactorColumns = lu_replay.refactorColumns;

    n: u32,
    /// Borrowed column ordering: pivot step k factors column q[k].
    q: []const u32,
    /// Original row -> pivot step.
    pinv: []u32,
    /// L, strictly lower with unit diagonal, CSC over pivot steps. Row
    /// indices are pivot steps once `factor` returns.
    lp: []u32,
    li: std.ArrayList(u32) = .empty,
    lx: std.ArrayList(f64) = .empty,
    /// U, strictly upper, CSC over pivot steps. Each column's rows are in
    /// the topological order the triangular solve visited them, which is
    /// the order `refactor` replays.
    up: []u32,
    ui: std.ArrayList(u32) = .empty,
    ux: std.ArrayList(f64) = .empty,
    udiag: []f64,
    /// prow[p] = pinv[row_idx[p]]: A entry p -> its permuted row.
    prow: []u32,
    /// Pivot steps that got a fabricated unit pivot (see `voidUnknown`).
    void_col: []bool,
    /// CSC entries in a fabricated pivot's row or column. A replay is
    /// valid only while all of them stay zero.
    void_slots: std.ArrayList(u32) = .empty,
    /// Pivot steps whose diagonal passed only the row-scaled threshold
    /// test. Such a pivot sits far below its raw column max, so the
    /// refactor growth monitor skips these steps.
    scaled_pivot: []bool,
    /// Full-factor DFS scan end of L[:,s]: lp[s+1], or just past the row
    /// pivoted at step s+1 when every later entry of L[:,s] is in
    /// L[:,s+1] (that child's walk has marked them all).
    lend: []u32,
    /// Full-factor scratch: a dense copy of each supernode that starts
    /// on an L column of at least `panel_min_rows`, so a later column
    /// applies a run of its steps as vector updates. `sn_of[s]` is the
    /// panel holding step s, or NONE.
    sn_of: []u32,
    /// Row -> slot in the open panel.
    sn_slot: []u32,
    /// DFS stack companion of `pstack`: for a row whose step lies in a
    /// panel, the last panel step its visit covered (`enterPanel`); NONE
    /// for any other row.
    sn_end: []u32,
    /// Full-factor scratch: the walk takes panel runs whole (`enter`).
    /// Set, for the rest of the factor, once panels hold 4/5 of the L
    /// entries stored so far: 0.86-0.98 on the resistor grids, at most
    /// 0.68 on the chain/ring_bsim4_10k re-pivots, whose factors keep the
    /// plain walk's order, bitwise.
    sn_dfs: bool = false,
    panels: std.ArrayList(Panel) = .empty,
    panel_rows: std.ArrayList(u32) = .empty,
    panel_vals: std.ArrayList(f64) = .empty,
    /// Small-matrix refactor tape: one {dst, l, u} slot triple per flop,
    /// `tv[dst] -= tv[l] * tv[u]`, in refactor order. Empty above
    /// `tape_max_flops`, where `refactor` runs the column loop instead.
    tape: std.ArrayList([3]u32) = .empty,
    /// Column k's flops are tape[tape_col[k]..tape_col[k+1]].
    tape_col: []u32,
    /// A entry p -> its slot in `tv`.
    amap: []u32,
    /// L entry -> its column; built only with a tape.
    lsrc: std.ArrayList(u32) = .empty,
    /// Tape slot values: ux ++ udiag ++ lx ++ one discard slot.
    tv: std.ArrayList(f64) = .empty,
    /// Dense accumulator, all-zero on entry to and exit from every public
    /// call, errors included. `factor` reads unscattered fill rows as 0,
    /// and `refactor` zeroes each slot as it consumes it rather than
    /// clearing the pattern up front.
    w: []f64,
    /// Solve workspace; also panel scratch during `factor`.
    y: []f64,
    /// rscale[r] = 1 / max_j |A[r][j]| (1 for an all-zero or non-finite
    /// row), by original row. Recomputed by each `factor`.
    rscale: []f64,
    /// Full-factor scratch: `want[c]` is the row column c tries first,
    /// `want_of` its inverse. When column c pivots on row r != want[c],
    /// the column that wanted r inherits want[c] (see `factor`).
    want: []u32,
    want_of: []u32,
    /// prev[c]: the row the last successful factor pivoted column c on,
    /// saved when a re-pivot starts.
    prev: []u32,
    /// L plus U entries of the last factor that started without a pivot
    /// sequence: the fill the column ordering was chosen for.
    base_lu: usize = 0,
    /// A bound on L plus U entries under diagonal pivots in this column
    /// order (`order.diagFill`), set by the owner: a factor that starts
    /// without a pivot sequence and grows past `fill_cap * diag_fill`
    /// restarts on the diagonal pivots. 0 leaves it uncapped.
    diag_fill: u64 = 0,
    /// A re-pivot whose L plus U grows past `fill_cap * base_lu` is
    /// abandoned for the previous pivot sequence. 0 disables the cap.
    fill_cap: f64 = 0,
    /// The capped re-pivot keeps a previous pivot only while it is at least
    /// this fraction of its column max, the test a refactor applies
    /// (`direct` sets its `refactor_growth_limit`); a smaller one is
    /// threshold-pivoted like a zero one (#45).
    keep_min: f64 = 0,
    /// Print every full factor's fill and census (`direct.Knobs.stats`);
    /// read only under `instrument`.
    stats: bool = false,
    /// DFS workspace, alive only inside `factor` (`scratch_u32`): the
    /// per-column visit mark (pivot step + 1), the reach in finish order,
    /// the DFS stack, and each stacked row's resume offset into its L
    /// column.
    flag: []u32,
    topo: []u32,
    stack: []u32,
    pstack: []u32,
    factored: bool = false,
    /// The largest |L| entry over the monitored steps (not `scaled_pivot`)
    /// of the last successful column replay that ran the growth monitor,
    /// or null when the last kernel to write L did not record it. A step's
    /// entries are v / d for its column's values v, so their max is the
    /// rounded max|v| / |d|, rounding being monotonic; the step's own
    /// pivot joins that max, so the value is at least 1. Lets `direct`'s
    /// replay check skip its O(|L|) scan.
    replay_mult: ?f64 = null,
    /// Bumped by every successful `factor`: a new pattern and pivot
    /// sequence, so device tables built from the old ones are stale.
    pattern_epoch: u32 = 0,

    /// `SingularMatrix`: some column has no finite nonzero pivot and is
    /// not a structurally void unknown.
    pub const FactorError = error{ OutOfMemory, SingularMatrix };

    /// Allocates every workspace. `q` is borrowed and must outlive the
    /// SparseLu; the row indices arrive with `factor`. Asserts that `q` has
    /// n entries and `col_ptr` n + 1.
    pub fn init(
        gpa: Allocator,
        n: u32,
        col_ptr: []const u32,
        q: []const u32,
    ) !Self {
        assert(q.len == n);
        assert(col_ptr.len == @as(usize, n) + 1);
        const nnz = col_ptr[n];
        // ponytail: pre-size L and U to max(nnz, 2n), typical circuit
        // fill; tune from a fill profile if first factors reallocate.
        const est_lu: usize = @max(nnz, 2 * @as(usize, n));

        var self = Self{
            .n = n,
            .q = q,
            .pinv = &.{},
            .lp = &.{},
            .up = &.{},
            .udiag = &.{},
            .prow = &.{},
            .void_col = &.{},
            .scaled_pivot = &.{},
            .lend = &.{},
            .sn_of = &.{},
            .sn_slot = &.{},
            .sn_end = &.{},
            .tape_col = &.{},
            .amap = &.{},
            .w = &.{},
            .y = &.{},
            .rscale = &.{},
            .flag = &.{},
            .topo = &.{},
            .stack = &.{},
            .pstack = &.{},
            .want = &.{},
            .want_of = &.{},
            .prev = &.{},
        };
        errdefer self.deinit(gpa);
        self.pinv = try gpa.alloc(u32, n);
        self.lp = try gpa.alloc(u32, @as(usize, n) + 1);
        self.up = try gpa.alloc(u32, @as(usize, n) + 1);
        self.udiag = try gpa.alloc(f64, n);
        self.prow = try gpa.alloc(u32, nnz);
        self.void_col = try gpa.alloc(bool, n);
        self.scaled_pivot = try gpa.alloc(bool, n);
        self.tape_col = try gpa.alloc(u32, @as(usize, n) + 1);
        self.amap = try gpa.alloc(u32, nnz);
        self.w = try gpa.alloc(f64, n);
        self.y = try gpa.alloc(f64, n);
        self.prev = try gpa.alloc(u32, n);
        try self.li.ensureTotalCapacity(gpa, est_lu);
        try self.lx.ensureTotalCapacity(gpa, est_lu);
        try self.ui.ensureTotalCapacity(gpa, est_lu);
        try self.ux.ensureTotalCapacity(gpa, est_lu);
        @memset(self.w, 0);
        return self;
    }

    /// Frees every buffer `init` and `factor` allocated; `q` stays the
    /// caller's.
    pub fn deinit(self: *Self, gpa: Allocator) void {
        lu_factor.dropScratch(self, gpa);
        inline for (.{ self.pinv, self.lp, self.up, self.prow, self.prev, self.void_col, self.scaled_pivot, self.tape_col, self.amap, self.udiag, self.w, self.y }) |s|
            gpa.free(s);
        inline for (.{ &self.tape, &self.lsrc, &self.tv, &self.panels, &self.panel_rows, &self.panel_vals, &self.void_slots, &self.li, &self.lx, &self.ui, &self.ux }) |l|
            l.deinit(gpa);
    }

    /// Full symbolic and numeric factorization. Per column: DFS reach,
    /// sparse triangular solve, threshold pivot (diagonal preferred),
    /// store L and U. Rebuilds the pattern, pivot sequence and replay
    /// tapes. `SingularMatrix` also when a value is NaN or inf.
    /// On error the factorization is unusable (`factored` false).
    /// Invalidates slices of L and U, which may reallocate, and bumps
    /// `pattern_epoch` on success: a LaneLu or device table built on the
    /// old factor must be rebuilt.
    ///
    /// A re-pivot (a factor after a successful one, which the caller runs
    /// when `refactor` fails) that fills past `fill_cap * base_lu` restarts
    /// on the previous pivot sequence, keeping every old pivot that still
    /// passes `keep_min`. Measured on the chain_bsim4_10k operating point's
    /// diverging Newton steps (values up to 1e60): threshold pivoting
    /// sent 4,503 of 44,925 columns off the diagonal and filled 34x
    /// against 2.6x (dev/solvers/gilbert-peierls-lu.md, "Re-pivoting").
    /// Any other factor that fills past `fill_cap * diag_fill` restarts the
    /// same way on the diagonal, and fails as singular when that passes the
    /// cap too: the cold operating point of chain_bsim4_100k (446k
    /// unknowns) pivoted off it so often that each factor took 400-570 s,
    /// and a capped re-pivot inherits that fill.
    ///
    /// Asserts that the pattern is the one `init` was sized for and that
    /// `pivot_tol` lies in [0, 1].
    pub fn factor(
        self: *Self,
        gpa: Allocator,
        col_ptr: []const u32,
        row_idx: []const u32,
        vals: []const f64,
        pivot_tol: f64,
    ) FactorError!void {
        assert(col_ptr.len == @as(usize, self.n) + 1);
        assert(col_ptr[self.n] == self.prow.len);
        assert(row_idx.len >= self.prow.len);
        assert(vals.len >= self.prow.len);
        assert(pivot_tol >= 0);
        assert(pivot_tol <= 1);
        try lu_factor.holdScratch(self, gpa);
        defer lu_factor.dropScratch(self, gpa);
        const repivot = self.factored and self.fill_cap > 0 and self.base_lu > 0;
        var cap: usize = std.math.maxInt(usize);
        if (repivot) {
            for (self.pinv, 0..) |step, r| self.prev[self.q[step]] = @intCast(r);
            cap = @intFromFloat(@min(self.fill_cap * @as(f64, @floatFromInt(self.base_lu)), 1e18));
        } else if (self.fill_cap > 0 and self.diag_fill > 0) {
            for (self.prev, 0..) |*r, c| r.* = @intCast(c);
            cap = @intFromFloat(@min(self.fill_cap * @as(f64, @floatFromInt(self.diag_fill)), 1e18));
        }
        if (try lu_factor.factorPass(self, gpa, col_ptr, row_idx, vals, pivot_tol, cap, false)) {
            if (instrument and self.stats)
                std.debug.print("lu-stats: factor passed {d} L+U entries; keeping the previous (or diagonal) pivots\n", .{cap});
            // The diagonal retry is held to the same cap: values that fill
            // past it even on the diagonal (the cold start above, 48M
            // entries) are no factor worth having, and the caller's
            // continuation (gmin stepping, a smaller step) does better.
            if (try lu_factor.factorPass(self, gpa, col_ptr, row_idx, vals, pivot_tol, if (repivot) std.math.maxInt(usize) else cap, true))
                return error.SingularMatrix;
        }
        assert(self.factored);
        if (!repivot) self.base_lu = self.li.items.len + self.ui.items.len;
    }

    pub const test_access = if (@import("builtin").is_test) .{
        .scatterAxpy = lu_replay.scatterAxpy,
        .refactorColumns = lu_replay.refactorColumns,
        .TestSystem = TestSystem,
        .overflow_case = overflow_case,
    } else {};

    /// Numeric refactor: the last factor's pattern and pivot sequence,
    /// new values, no allocation. Fails when a pivot is zero or
    /// non-finite or falls below `growth_limit` times its column max (steps
    /// in `scaled_pivot` excepted; 0 disables the test), an L or U entry is
    /// non-finite, or a fabricated pivot's row or column turned nonzero;
    /// the caller then runs a full `factor`, and the old factors are
    /// unusable until it does. Asserts that a `factor` has succeeded and that `col_ptr` and
    /// `vals` cover its pattern. Allocates nothing and leaves `w` all-zero
    /// on every path.
    pub fn refactor(
        self: *Self,
        col_ptr: []const u32,
        vals: []const f64,
        growth_limit: f64,
    ) error{SingularMatrix}!void {
        assert(self.factored);
        assert(col_ptr.len == @as(usize, self.n) + 1);
        assert(vals.len >= self.prow.len);
        for (self.void_slots.items) |p| {
            if (vals[p] != 0) return error.SingularMatrix;
        }
        if (self.tv.items.len != 0) return lu_replay.refactorTape(self, vals, growth_limit);
        return lu_replay.refactorColumns(self, col_ptr, vals, growth_limit);
    }

    /// One factor's slices for a column replay (`lu_replay.zig`).
    pub const Replay = lu_replay.Replay;

    /// The column replay's inputs for new `vals`, for `refactorColumns` and
    /// for callers that replay columns in their own schedule
    /// (`subtree.Plan`). Asserts as `refactor` does.
    pub fn replay(self: *Self, col_ptr: []const u32, vals: []const f64, growth_limit: f64) Replay {
        assert(self.factored);
        assert(col_ptr.len == @as(usize, self.n) + 1);
        assert(vals.len >= self.prow.len);
        return .{
            .li = self.li.items,
            .lx = self.lx.items,
            .ui = self.ui.items,
            .ux = self.ux.items,
            .lp = self.lp,
            .up = self.up,
            .prow = self.prow,
            .udiag = self.udiag,
            .q = self.q,
            .void_col = self.void_col,
            .scaled_pivot = self.scaled_pivot,
            .col_ptr = col_ptr,
            .vals = vals,
            .growth_limit = growth_limit,
        };
    }

    /// x = A^-1 b. `b` and `x` may alias. Requires the last `factor` or
    /// `refactor` to have succeeded; garbage otherwise. O(nnz(L+U)).
    /// Asserts that a `factor` has succeeded, that `b` has n entries and
    /// `x` at least n.
    pub fn solve(self: *Self, b: []const f64, x: []f64) void {
        assert(self.factored);
        assert(b.len == self.n);
        assert(x.len >= self.n);
        const li = self.li.items;
        const lx = self.lx.items;
        const ui = self.ui.items;
        const ux = self.ux.items;
        // Locals for the same aliasing reason as `refactorColumns`.
        const y = self.y;
        const lp = self.lp;
        const up = self.up;
        const pinv = self.pinv;
        const udiag = self.udiag;

        // y = P b
        for (b, 0..) |bi, r| y[pinv[r]] = bi;

        // Forward through unit-lower L.
        if (self.tv.items.len != 0) {
            // Tape matrices: one flat pass over L. Skipping a zero y[k]
            // matches the column loop, signed zeros included.
            for (li, lx, self.lsrc.items) |r, l, k| {
                const yk = y[k];
                if (yk != 0) y[r] -= l * yk;
            }
        } else for (0..self.n) |k| {
            const yk = y[k];
            if (yk == 0) continue;
            lu_replay.scatterAxpy(y, li, lx, lp[k], lp[k + 1], yk);
        }

        // Back through U.
        var k = self.n;
        while (k > 0) {
            k -= 1;
            const zk = y[k] / udiag[k];
            y[k] = zk;
            if (zk == 0) continue;
            lu_replay.scatterAxpy(y, ui, ux, up[k], up[k + 1], zk);
        }

        // x = Q^-1 z
        for (self.q, 0..) |c, j| x[c] = y[j];
    }

    /// x = A^-T b, for adjoint analyses. A = P^-1 L U Q^-1, so
    /// A^-T = P^T L^-T U^-T Q^T. `b` and `x` may alias. Same preconditions
    /// and asserts as `solve`.
    pub fn solveT(self: *Self, b: []const f64, x: []f64) void {
        assert(self.factored);
        assert(b.len == self.n);
        assert(x.len >= self.n);
        const li = self.li.items;
        const lx = self.lx.items;
        const ui = self.ui.items;
        const ux = self.ux.items;
        // Locals for the same aliasing reason as `refactorColumns`; the
        // running `yk` does the same subtractions in the same order.
        const y = self.y;
        const lp = self.lp;
        const up = self.up;
        const pinv = self.pinv;
        const udiag = self.udiag;

        for (self.q, 0..) |c, k| y[k] = b[c];

        // Forward through U^T, gathering.
        for (0..self.n) |k| {
            var yk = y[k];
            for (up[k]..up[k + 1]) |p| yk -= ux[p] * y[ui[p]];
            y[k] = yk / udiag[k];
        }

        // Back through unit-upper L^T, gathering.
        var k = self.n;
        while (k > 0) {
            k -= 1;
            var yk = y[k];
            for (lp[k]..lp[k + 1]) |p| yk -= lx[p] * y[li[p]];
            y[k] = yk;
        }

        for (0..self.n) |r| x[r] = y[pinv[r]];
    }

    // ---- private ----

    /// Supernode steps [first, last]. Its rows are L[:,first]'s: first
    /// the block rows of steps first+1..last (slot t-first-1), then the
    /// rest. Values are column-major, `nrows` per member step.
    const Panel = struct { first: u32, last: u32, row_off: u32, val_off: u32 };

    comptime {
        assert(@sizeOf(Panel) == 16);
    }
};

const NONE: u32 = std.math.maxInt(u32);

/// Test-only random system: an n x n CSC pattern (n <= 64) with a full
/// structural diagonal, as the pattern merge guarantees, and off-diagonals
/// at `density`. Values lie in [-1, 1] with the diagonal pushed past its
/// column's off-diagonal sum, except that a column with another entry gets
/// a zero diagonal (an MNA branch row) 15% of the time. No unknown is void.
const TestSystem = struct {
    col_ptr: []u32,
    row_idx: []u32,
    vals: []f64,

    /// Caller owns the result; free with `deinit`. Asserts n <= 64.
    pub fn init(gpa: Allocator, r: std.Random, n: u32, density: f64) !TestSystem {
        assert(n <= 64);
        var mask: [64]u64 = @splat(0);
        var nnz: u32 = 0;
        for (0..n) |j| {
            mask[j] = @as(u64, 1) << @intCast(j);
            for (0..n) |i| {
                if (i != j and r.float(f64) < density) mask[j] |= @as(u64, 1) << @intCast(i);
            }
            nnz += @popCount(mask[j]);
        }
        const col_ptr = try gpa.alloc(u32, n + 1);
        errdefer gpa.free(col_ptr);
        const row_idx = try gpa.alloc(u32, nnz);
        errdefer gpa.free(row_idx);
        const vals = try gpa.alloc(f64, nnz);
        var p: u32 = 0;
        col_ptr[0] = 0;
        for (0..n) |j| {
            var sum: f64 = 0;
            var diag: u32 = 0;
            for (0..n) |i| {
                if ((mask[j] >> @intCast(i)) & 1 == 0) continue;
                row_idx[p] = @intCast(i);
                vals[p] = 2 * r.float(f64) - 1;
                if (i == j) diag = p else sum += @abs(vals[p]);
                p += 1;
            }
            const d = vals[diag];
            vals[diag] = if (sum > 0 and r.float(f64) < 0.15) 0 else d + std.math.copysign(sum, d);
            col_ptr[j + 1] = p;
        }
        return .{ .col_ptr = col_ptr, .row_idx = row_idx, .vals = vals };
    }

    /// Frees the three arrays.
    pub fn deinit(self: TestSystem, gpa: Allocator) void {
        gpa.free(self.col_ptr);
        gpa.free(self.row_idx);
        gpa.free(self.vals);
    }

    /// LAPACK's scaled residual ||A x - b|| / (||A|| ||x|| n eps), max norms,
    /// with n * max|a_ij| bounding ||A||; `transpose` checks A^T x = b.
    pub fn residual(self: TestSystem, vals: []const f64, x: []const f64, b: []const f64, comptime transpose: bool) f64 {
        const n = self.col_ptr.len - 1;
        var r: [64]f64 = undefined;
        @memcpy(r[0..n], b);
        var amax: f64 = 0;
        for (0..n) |j| for (self.col_ptr[j]..self.col_ptr[j + 1]) |p| {
            const i = self.row_idx[p];
            if (transpose) r[j] -= vals[p] * x[i] else r[i] -= vals[p] * x[j];
            amax = @max(amax, @abs(vals[p]));
        };
        var rmax: f64 = 0;
        var xmax: f64 = 0;
        for (r[0..n], x) |ri, xi| {
            rmax = @max(rmax, @abs(ri));
            xmax = @max(xmax, @abs(xi));
        }
        const nf: f64 = @floatFromInt(n);
        return rmax / (nf * amax * xmax * nf * std.math.floatEps(f64));
    }
};

/// U[1,2] = A[1,2] - L[1,0] U[0,2] = -1.5e308 - 1.5e308 overflows, and no
/// L entry carries it to a pivot: L[:,1] is empty. `overflow_vals` are
/// finite; `benign_vals` give the same pattern and pivots.
const overflow_case = struct {
    pub const col_ptr = [_]u32{ 0, 2, 3, 6 };
    pub const row_idx = [_]u32{ 0, 1, 1, 0, 1, 2 };
    pub const q = [_]u32{ 0, 1, 2 };
    pub const benign_vals = [_]f64{ 1, 1, 1, 1, 1, 1 };
    pub const overflow_vals = [_]f64{ 1, 1, 1, 1.5e308, -1.5e308, 1 };
};

// ---- private ----

fn expectAllZero(w: []const f64) !void {
    for (w) |v| try testing.expect(v == 0);
}

fn expectBitwise(a: []const f64, b: []const f64) !void {
    try testing.expectEqualSlices(u64, @ptrCast(a), @ptrCast(b));
}

/// One `init`, a full factor, a capped re-pivot that overflows its cap and
/// restarts on the old pivots, a replay and a solve: every allocating path
/// of the factor. Any error must hand `w` back all-zero. Asserts n > 3.
fn factorLifecycle(gpa: Allocator, sys: TestSystem, q: []const u32) !void {
    const n: u32 = @intCast(q.len);
    assert(n > 3);
    var lu = try SparseLu.init(gpa, n, sys.col_ptr, q);
    defer lu.deinit(gpa);
    errdefer expectAllZero(lu.w) catch @panic("w not zero after an error");
    try lu.factor(gpa, sys.col_ptr, sys.row_idx, sys.vals, 1e-3);
    lu.fill_cap = 0.5;
    lu.keep_min = 1e-12;
    try lu.factor(gpa, sys.col_ptr, sys.row_idx, sys.vals, 1e-3);
    try lu.refactor(sys.col_ptr, sys.vals, 1e-12);
    // b_3 = 0: the void unknown's equation is 0 = b_3.
    var b: [64]f64 = @splat(1);
    b[3] = 0;
    var x: [64]f64 = undefined;
    lu.solve(b[0..n], x[0..n]);
    try testing.expect(sys.residual(sys.vals, x[0..n], b[0..n], false) <= 1e6);
}

test "factor, refactor and both solves leave a small scaled residual on random systems" {
    // A wrong factor scores ~1/(n eps), 1e13 and up; threshold pivoting at
    // 1e-3 can grow entries, so the bar is far above LAPACK's 30.
    const thresh = 1e6;
    const gpa = testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x5ba75e);
    const r = prng.random();
    const trials = 256;
    var solved: u32 = 0;
    for (0..trials) |t| {
        // Every order 1..64 once, then random ones; dense enough past 32
        // that L columns reach `panel_min_rows` and panels open.
        const n: u32 = if (t < 64) @intCast(t + 1) else r.intRangeAtMost(u32, 1, 64);
        const density = r.float(f64) * @as(f64, if (n > 32) 0.6 else 0.3);
        const sys = try TestSystem.init(gpa, r, n, density);
        defer sys.deinit(gpa);
        const q = try gpa.alloc(u32, n);
        defer gpa.free(q);
        for (q, 0..) |*c, i| c.* = @intCast(i);
        r.shuffle(u32, q);
        var lu = try SparseLu.init(gpa, n, sys.col_ptr, q);
        defer lu.deinit(gpa);
        lu.factor(gpa, sys.col_ptr, sys.row_idx, sys.vals, 1e-3) catch |err| {
            try testing.expectEqual(error.SingularMatrix, err);
            try expectAllZero(lu.w);
            continue;
        };
        try expectAllZero(lu.w);
        solved += 1;

        const b = try gpa.alloc(f64, 3 * @as(usize, n));
        defer gpa.free(b);
        const rhs = b[0..n];
        const x = b[n .. 2 * n];
        const x2 = b[2 * n ..];
        for (rhs) |*v| v.* = 2 * r.float(f64) - 1;
        inline for (.{ false, true }) |transpose| {
            if (transpose) lu.solveT(rhs, x) else lu.solve(rhs, x);
            try testing.expect(sys.residual(sys.vals, x, rhs, transpose) <= thresh);
            // In place (b aliases x), bitwise.
            @memcpy(x2, rhs);
            if (transpose) lu.solveT(x2, x2) else lu.solve(x2, x2);
            try expectBitwise(x, x2);
        }

        // The full factor (panels included) is bitwise a column replay of
        // the same values.
        const f_l = try gpa.dupe(f64, lu.lx.items);
        defer gpa.free(f_l);
        const f_u = try gpa.dupe(f64, lu.ux.items);
        defer gpa.free(f_u);
        const f_d = try gpa.dupe(f64, lu.udiag);
        defer gpa.free(f_d);
        try lu.refactorColumns(sys.col_ptr, sys.vals, 1e-12);
        try expectBitwise(f_l, lu.lx.items);
        try expectBitwise(f_u, lu.ux.items);
        try expectBitwise(f_d, lu.udiag);

        // New values on the frozen pivots: `refactor` (the tape on small
        // factors) is bitwise the column replay, failures included.
        const pert = try gpa.alloc(f64, sys.vals.len);
        defer gpa.free(pert);
        for (pert, sys.vals) |*o, v| o.* = v * (0.9 + 0.2 * r.float(f64));
        const ok = if (lu.refactor(sys.col_ptr, pert, 1e-12)) |_| true else |_| false;
        try expectAllZero(lu.w);
        if (ok) lu.solve(rhs, x);
        const ok2 = if (lu.refactorColumns(sys.col_ptr, pert, 1e-12)) |_| true else |_| false;
        try expectAllZero(lu.w);
        try testing.expectEqual(ok, ok2);
        if (!ok) continue;
        lu.solve(rhs, x2);
        try expectBitwise(x, x2);
        try testing.expect(sys.residual(pert, x, rhs, false) <= thresh);
        lu.solveT(rhs, x);
        try testing.expect(sys.residual(pert, x, rhs, true) <= thresh);
    }
    try testing.expect(solved >= trials / 2);
}

test "a failed replay after a void column with a NaN multiplier leaves w zero" {
    // Unknown 1 is void (zero-valued column and row), and its U entry
    // reaches L[:,0], whose row 3 lies below it. Step 2 then fails, so
    // nothing after the void column would clear row 3.
    const gpa = testing.allocator;
    const col_ptr = [_]u32{ 0, 2, 4, 5, 6 };
    const row_idx = [_]u32{ 0, 3, 0, 1, 2, 3 };
    const vals = [_]f64{ 1, 0.5, 0, 0, 1, 1 };
    const q = [_]u32{ 0, 1, 2, 3 };
    var lu = try SparseLu.init(gpa, 4, &col_ptr, &q);
    defer lu.deinit(gpa);
    try lu.factor(gpa, &col_ptr, &row_idx, &vals, 1e-3);
    try testing.expect(lu.void_col[1]);
    const bad = [_]f64{ 1, std.math.nan(f64), 0, 0, 0, 1 };
    try testing.expectError(error.SingularMatrix, lu.refactorColumns(&col_ptr, &bad, 1e-12));
    try expectAllZero(lu.w);
    try testing.expectError(error.SingularMatrix, lu.refactor(&col_ptr, &bad, 1e-12));
    try expectAllZero(lu.w);
}

test "a factor whose finite values overflow in U off the pivot fails" {
    const gpa = testing.allocator;
    const c = overflow_case;
    var lu = try SparseLu.init(gpa, 3, &c.col_ptr, &c.q);
    defer lu.deinit(gpa);
    try testing.expectError(error.SingularMatrix, lu.factor(gpa, &c.col_ptr, &c.row_idx, &c.overflow_vals, 1e-3));
    try expectAllZero(lu.w);
}

test "a replay whose finite values overflow in U off the pivot fails, tape and columns" {
    const gpa = testing.allocator;
    const c = overflow_case;
    var lu = try SparseLu.init(gpa, 3, &c.col_ptr, &c.q);
    defer lu.deinit(gpa);
    try lu.factor(gpa, &c.col_ptr, &c.row_idx, &c.benign_vals, 1e-3);
    try testing.expect(lu.tv.items.len != 0);
    try testing.expectError(error.SingularMatrix, lu.refactorColumns(&c.col_ptr, &c.overflow_vals, 1e-12));
    try expectAllZero(lu.w);
    try testing.expectError(error.SingularMatrix, lu.refactor(&c.col_ptr, &c.overflow_vals, 1e-12));
    // The growth monitor off: the store flag alone must catch it.
    try testing.expectError(error.SingularMatrix, lu.refactorColumns(&c.col_ptr, &c.overflow_vals, 0));
    try testing.expectError(error.SingularMatrix, lu.refactor(&c.col_ptr, &c.overflow_vals, 0));
    try lu.refactor(&c.col_ptr, &c.benign_vals, 1e-12);
}

test "a dense block factors through the supernodal walk and solves" {
    const gpa = testing.allocator;
    // Dense and diagonally dominant: every L column nests in the one
    // before, so panels hold nearly all of L and the walk takes their
    // runs whole (`sn_dfs`), the mesh case.
    const n = 96;
    var col_ptr: [n + 1]u32 = undefined;
    var row_idx: [n * n]u32 = undefined;
    var vals: [n * n]f64 = undefined;
    for (0..n) |j| {
        col_ptr[j] = @intCast(j * n);
        for (0..n) |i| {
            row_idx[j * n + i] = @intCast(i);
            vals[j * n + i] = if (i == j) 2 * n else 1 / @as(f64, @floatFromInt(1 + i + 2 * j));
        }
    }
    col_ptr[n] = n * n;
    var q: [n]u32 = undefined;
    for (&q, 0..) |*c, i| c.* = @intCast(n - 1 - i);
    var lu = try SparseLu.init(gpa, n, &col_ptr, &q);
    defer lu.deinit(gpa);
    try lu.factor(gpa, &col_ptr, &row_idx, &vals, 1e-3);
    try testing.expect(lu.sn_dfs);
    try expectAllZero(lu.w);
    // b = A * 1, so x is all ones; then again through the replay.
    var b: [n]f64 = @splat(0);
    for (0..n) |j| for (col_ptr[j]..col_ptr[j + 1]) |p| {
        b[row_idx[p]] += vals[p];
    };
    var x: [n]f64 = undefined;
    for (0..2) |pass| {
        if (pass == 1) try lu.refactor(&col_ptr, &vals, 1e-12);
        lu.solve(&b, &x);
        for (x) |xi| try testing.expectApproxEqAbs(@as(f64, 1), xi, 1e-12);
    }
}

test "a first factor past fill_cap * diag_fill retries on the diagonal, and fails past it there too" {
    const gpa = testing.allocator;
    // An arrow pointing at the last row and column, with a tiny first
    // pivot: threshold pivoting takes row 5 at step 0 and fills, the
    // diagonal does not (L and U hold the arrow's 5 + 5 entries).
    const n = 6;
    var col_ptr: [n + 1]u32 = undefined;
    var row_idx: [3 * n]u32 = undefined;
    var vals: [3 * n]f64 = undefined;
    var nnz: u32 = 0;
    for (0..n) |j| {
        col_ptr[j] = nnz;
        for (0..n) |i| {
            if (!(i == j or i == n - 1 or j == n - 1)) continue;
            row_idx[nnz] = @intCast(i);
            vals[nnz] = if (i == j) (if (i == 0) 1e-6 else 4) else 1;
            nnz += 1;
        }
    }
    col_ptr[n] = nnz;
    var q: [n]u32 = undefined;
    for (&q, 0..) |*c, i| c.* = @intCast(i);
    for ([_]u64{ 10, 9 }) |diag_fill| {
        var lu = try SparseLu.init(gpa, n, col_ptr[0 .. n + 1], &q);
        defer lu.deinit(gpa);
        lu.fill_cap = 1;
        lu.keep_min = 1e-12;
        lu.diag_fill = diag_fill;
        const got = lu.factor(gpa, col_ptr[0 .. n + 1], row_idx[0..nnz], vals[0..nnz], 0.5);
        try expectAllZero(lu.w);
        if (diag_fill == 9) {
            try testing.expectError(error.SingularMatrix, got);
            continue;
        }
        try got;
        for (lu.pinv, 0..) |step, r| try testing.expectEqual(@as(u32, @intCast(r)), step);
        try testing.expectEqual(@as(usize, 10), lu.li.items.len + lu.ui.items.len);
    }
}

test "the empty and 1x1 systems factor, replay and solve" {
    const gpa = testing.allocator;
    var empty = try SparseLu.init(gpa, 0, &.{0}, &.{});
    defer empty.deinit(gpa);
    try empty.factor(gpa, &.{0}, &.{}, &.{}, 1e-3);
    try empty.refactor(&.{0}, &.{}, 1e-12);
    empty.solve(&.{}, &.{});
    empty.solveT(&.{}, &.{});

    var one = try SparseLu.init(gpa, 1, &.{ 0, 1 }, &.{0});
    defer one.deinit(gpa);
    try one.factor(gpa, &.{ 0, 1 }, &.{0}, &.{4}, 1e-3);
    var x: [1]f64 = undefined;
    one.solve(&.{2}, &x);
    try testing.expectEqual(@as(f64, 0.5), x[0]);
    try testing.expectError(error.SingularMatrix, one.refactor(&.{ 0, 1 }, &.{std.math.inf(f64)}, 1e-12));
    try testing.expectError(error.SingularMatrix, one.factor(gpa, &.{ 0, 1 }, &.{0}, &.{std.math.nan(f64)}, 1e-3));
    try expectAllZero(one.w);
}

test "solutions scale exactly with A and b by powers of two, factor and replay" {
    // Every pivot test is homogeneous and every product or quotient of
    // binary-scaled operands rounds identically, so x(2^-20 A, 2^7 b) is
    // 2^27 x(A, b) bitwise: the same pivots, the same roundings.
    const gpa = testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x2b17);
    const r = prng.random();
    const sa = std.math.ldexp(@as(f64, 1), -20);
    const sb = std.math.ldexp(@as(f64, 1), 7);
    const sx = std.math.ldexp(@as(f64, 1), 27);
    var checked: u32 = 0;
    for (0..64) |t| {
        const n: u32 = @intCast(1 + t % 48);
        const sys = try TestSystem.init(gpa, r, n, if (n > 32) 0.5 else 0.25);
        defer sys.deinit(gpa);
        const q = try gpa.alloc(u32, n);
        defer gpa.free(q);
        for (q, 0..) |*c, i| c.* = @intCast(i);
        r.shuffle(u32, q);
        const scaled = try gpa.alloc(f64, sys.vals.len);
        defer gpa.free(scaled);
        for (scaled, sys.vals) |*o, v| o.* = v * sa;
        const buf = try gpa.alloc(f64, 4 * @as(usize, n));
        defer gpa.free(buf);
        const b = buf[0..n];
        const b2 = buf[n .. 2 * n];
        const x = buf[2 * n .. 3 * n];
        const x2 = buf[3 * n ..];
        for (b, b2) |*v, *v2| {
            v.* = 2 * r.float(f64) - 1;
            v2.* = v.* * sb;
        }

        var lu = try SparseLu.init(gpa, n, sys.col_ptr, q);
        defer lu.deinit(gpa);
        var lu2 = try SparseLu.init(gpa, n, sys.col_ptr, q);
        defer lu2.deinit(gpa);
        const ok = if (lu.factor(gpa, sys.col_ptr, sys.row_idx, sys.vals, 1e-3)) |_| true else |_| false;
        const ok2 = if (lu2.factor(gpa, sys.col_ptr, sys.row_idx, scaled, 1e-3)) |_| true else |_| false;
        try testing.expectEqual(ok, ok2);
        if (!ok) continue;
        try testing.expectEqualSlices(u32, lu.pinv, lu2.pinv);
        inline for (.{ false, true }) |transpose| {
            if (transpose) lu.solveT(b, x) else lu.solve(b, x);
            if (transpose) lu2.solveT(b2, x2) else lu2.solve(b2, x2);
            for (x) |*v| v.* *= sx;
            try expectBitwise(x, x2);
        }
        // The replay of the same values keeps the relation.
        try lu2.refactor(sys.col_ptr, scaled, 1e-12);
        try lu.refactor(sys.col_ptr, sys.vals, 1e-12);
        lu.solve(b, x);
        lu2.solve(b2, x2);
        for (x) |*v| v.* *= sx;
        try expectBitwise(x, x2);
        checked += 1;
    }
    try testing.expect(checked >= 32);
}

test "factor, a capped re-pivot and a replay release storage and leave w zero on every allocation failure" {
    const gpa = testing.allocator;
    var prng = std.Random.DefaultPrng.init(0xa110c);
    const r = prng.random();
    // A small factor (the replay tape) with a void unknown, and a dense
    // one (panels, L and U past their initial capacity).
    for ([_]struct { n: u32, density: f64 }{ .{ .n = 9, .density = 0.3 }, .{ .n = 40, .density = 0.7 } }) |c| {
        const sys = try TestSystem.init(gpa, r, c.n, c.density);
        defer sys.deinit(gpa);
        if (c.n < 32) {
            // Unknown 3: every entry of its row and column becomes zero.
            for (0..c.n) |j| for (sys.col_ptr[j]..sys.col_ptr[j + 1]) |p| {
                if (j == 3 or sys.row_idx[p] == 3) sys.vals[p] = 0;
            };
        }
        var q: [64]u32 = undefined;
        for (q[0..c.n], 0..) |*v, i| v.* = @intCast(i);
        // The case must succeed without injected failures.
        try factorLifecycle(gpa, sys, q[0..c.n]);
        try testing.checkAllAllocationFailures(@import("core").testing.fixed_growth, factorLifecycle, .{ sys, q[0..c.n] });
    }
}
