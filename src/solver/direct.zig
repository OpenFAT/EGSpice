//! Fixed-pattern sparse direct solver: the facade every Newton caller uses.
//! Picks an engine from the pattern (tridiagonal, bordered block diagonal or
//! general sparse LU), skips refactoring when the values did not change, and
//! demotes to the general LU when a structured engine meets a singular pivot
//! (or, for BBD, a Schur reduction too unstable to refine). A pivot replay
//! that kept a sub-threshold pivot is probed and re-factored when its solves
//! miss the threshold-pivoting bar.
//!
//! In: a frozen CSC pattern, then values per factor and right-hand sides per
//! solve. Out: solutions of A x = b and A^T x = b. Cites KLU's threshold
//! pivoting (`pivot_tol`), LAPACK dgerfs and LAWN 41 (BBD refinement and the
//! residual bars), dev/solvers/gilbert-peierls-lu.md ("Two replay
//! thresholds"), dev/solvers/gpu-lu.md ("Multicore host refactor"); #44,
//! #45, #136.

const std = @import("std");
const assert = std.debug.assert;
const sparse_lu = @import("sparse_lu.zig");
const tridiag_mod = @import("tridiag.zig");
const order_mod = @import("order.zig");
const bbd_mod = @import("bbd.zig");
const lu_kernels = @import("kernels/lu.zig");
const subtree = @import("subtree.zig");
const num = @import("core").numerics;

const Allocator = std.mem.Allocator;

/// `-Dlu-instrument`: the `ZP_LU_STATS` census is compiled in.
pub const instrument = sparse_lu.instrument;

/// Threshold partial pivoting keeps the diagonal while
/// |diag| >= pivot_tol * column max (KLU's default 0.001).
pub const pivot_tol: f64 = 1e-3;
/// A re-pivot whose L plus U outgrows this multiple of the fresh factor's
/// restarts on the previous pivots (`SparseLu.fill_cap`).
const repivot_fill_cap: f64 = 3;
/// A BBD factor whose `Bbd.growth` exceeds this is probed: a refined solve
/// each way (A and A^T) whose scaled residual misses `probe_ratio_max`
/// demotes it to the general LU, else every solve on it is refined (#136).
/// Below it the certificate's ratio stayed under 0.04 (`residual_tests.zig`).
const bbd_refine_growth: f64 = 1e3;

/// Pivoting knobs. The defaults are the production settings.
pub const Params = struct {
    /// A refactor whose reused pivot falls below this fraction of its column
    /// max fails, forcing a full re-pivoting factor. 0 disables the monitor.
    /// Permissive for Newton, whose residual gate absorbs an inexact step
    /// and whose diverging iterates (values to 1e60) would otherwise
    /// re-pivot and fill every iteration; `FreqSolver`, whose solves are
    /// final, raises it to `pivot_tol` (#44). A replay that took a pivot
    /// below `pivot_tol` is probed and re-factored when its solves miss the
    /// threshold-pivoting bar (`replayHolds`). dev/solvers/gilbert-peierls-lu.md,
    /// "Two replay thresholds".
    refactor_growth_limit: f64 = 1e-12,
};

/// Refinement steps after a BBD solve at most (LAPACK dgerfs' ITMAX).
const refine_max = 5;
/// The probe's bar on ||b - A x|| / (||A|| ||x|| n eps): LAWN 41's 30
/// over 30, since one right-hand side only samples the error. On the
/// certificate's BBD family a probe at 27.7 let a solve through at 67;
/// refined solves that keep BBD probe at 0.2 or less (#136).
const probe_ratio_max = 1;

/// Developer switches. The facade reads them from the environment once
/// (`espice.zig`); the solver reads no environment.
pub const Knobs = struct {
    /// `ZP_LU_STATS`: print the BBD census and every full factor's fill.
    /// Read only under `-Dlu-instrument` (`instrument`).
    stats: bool = false,
    /// `ESPICE_NO_BBD`: force the flat LU, for A/B comparisons.
    no_bbd: bool = false,
    /// `ESPICE_LU_PAR=0/1`: force the multicore refactor off or on.
    par: ?bool = null,
    /// `ESPICE_LU_PAR_STATS`: print the multicore refactor's admission.
    par_stats: bool = false,
};

/// One linear engine, picked from the pattern by `Solver.init`.
pub const Engine = union(enum) {
    lu: sparse_lu.SparseLu,
    tri: tridiag_mod.TriDiag,
    bbd: bbd_mod.Bbd,
};

/// Sparse direct solver on one frozen CSC pattern.
pub const Solver = struct {
    const Self = @This();
    const SparseLu = sparse_lu.SparseLu;
    const TriDiag = tridiag_mod.TriDiag;
    const BbdEng = bbd_mod.Bbd;

    n: u32,
    /// Borrowed CSC pattern; must outlive the solver.
    col_ptr: []const u32,
    row_idx: []const u32,
    /// The live engine. The general LU is built only when a structured
    /// engine is demoted, so BBD and tridiagonal systems never pay for it.
    engine: Engine,
    /// The engine `init` picked, which `reset` restores after a demotion.
    init_engine: std.meta.Tag(Engine) = .lu,
    gpa: Allocator,
    /// True once a factorization matches `vcopy`.
    factored: bool = false,
    params: Params = .{},
    knobs: Knobs,
    /// Owned column ordering handed to the LU; empty until the LU exists.
    q: []u32 = &.{},
    /// Values of the last factorization, for the unchanged-matrix bypass.
    vcopy: []f64 = &.{},
    /// Bumped whenever `vcopy` changes, by `factor` or by a device factor,
    /// so a device copy of the factors knows whether it is current.
    gen: u32 = 0,
    /// The multicore refactor of the current pivot epoch, when the cost
    /// model admitted it (`refactorLu`).
    par: ?lu_kernels.HostRefactor = null,
    /// The subtree refactor of the current pivot epoch, for a thin factor
    /// the ticketed one declined (`refactorLu`).
    sub: ?subtree.Plan = null,
    /// Pivot epoch the cost model last judged.
    par_epoch: u32 = 0,
    /// A device factored `vcopy` and the LU still holds older values. The
    /// next host use refactors from `vcopy` first (bitwise the device's).
    host_stale: bool = false,
    /// BBD refinement scratch: the right-hand side, the residual,
    /// |A||x| + |b| and the probe's solution, n entries each. Empty on the
    /// other engines.
    refine_buf: []f64 = &.{},
    /// `replayHolds` scratch, 3n: the probe solution, its residual and the
    /// row sums of |A|. Empty until a replay first needs the probe.
    probe: []f64 = &.{},

    /// Chooses the engine from the pattern alone: tridiagonal, then BBD
    /// when `bbd` describes a profitable split, else sparse LU (whose AMD
    /// ordering fails with `OutOfWorkspace` on pathological fill). Borrows
    /// `col_ptr` and `row_idx`, which must outlive the solver; free with
    /// `deinit`. Asserts that `col_ptr` has n + 1 entries and `row_idx`
    /// covers them.
    pub fn init(gpa: Allocator, n: u32, col_ptr: []const u32, row_idx: []const u32, bbd: ?num.BbdInfo, knobs: Knobs) !Self {
        return initOrdered(gpa, n, col_ptr, row_idx, bbd, knobs, null);
    }

    /// `init`, reusing `order` (the `q` another solver's LU computed on this
    /// same pattern, copied) when the engine is the sparse LU: a sweep's
    /// queries share one pattern, and BTF + AMD cost 95M instructions a
    /// call on a 100k-node ladder.
    pub fn initOrdered(gpa: Allocator, n: u32, col_ptr: []const u32, row_idx: []const u32, bbd: ?num.BbdInfo, knobs: Knobs, order: ?[]const u32) !Self {
        assert(col_ptr.len == @as(usize, n) + 1);
        assert(row_idx.len >= col_ptr[n]);
        var self: Self = .{ .n = n, .col_ptr = col_ptr, .row_idx = row_idx, .engine = undefined, .gpa = gpa, .knobs = knobs };
        if (order) |o| {
            assert(o.len == n);
            self.q = try gpa.dupe(u32, o);
        }
        // `q` is freed here on failure until an engine holds; after that,
        // `deinit` frees it.
        var engine_set = false;
        errdefer if (!engine_set) gpa.free(self.q);
        if (tridiag_mod.isTridiag(n, col_ptr, row_idx)) {
            self.engine = .{ .tri = try TriDiag.init(gpa, n, col_ptr, row_idx) };
        } else if (try initBbd(gpa, n, col_ptr, row_idx, bbd, knobs)) |eng| {
            self.engine = .{ .bbd = eng };
        } else {
            self.engine = .{ .lu = try self.initLu() };
        }
        engine_set = true;
        self.init_engine = self.engine;
        errdefer self.deinit();
        self.vcopy = try gpa.alloc(f64, col_ptr[n]);
        if (self.engine == .bbd) self.refine_buf = try gpa.alloc(f64, 4 * @as(usize, n));
        assert(!self.factored);
        return self;
    }

    /// Returns the solver to its state right after `initOrdered`, keeping
    /// its allocations: the next `factor` pivots from scratch as a fresh
    /// solver's would, so everything after it is bitwise a fresh solver's.
    /// A structured engine demoted since `init` is rebuilt, which is why
    /// `bbd` must be what `init` got. `gen` moves on, so a device copy of
    /// the factors goes stale. Fails only with `OutOfMemory`, rebuilding a
    /// demoted engine or the bypass copy `dropBypass` freed, and leaves the
    /// solver usable (unfactored) then.
    pub fn reset(self: *Self, bbd: ?num.BbdInfo) Allocator.Error!void {
        self.factored = false;
        self.host_stale = false;
        self.params = .{};
        self.gen +%= 1;
        self.dropPlans();
        if (self.sparse()) |lu| {
            lu.factored = false;
            self.par_epoch = lu.pattern_epoch;
        }
        if (self.vcopy.len == 0) self.vcopy = try self.gpa.alloc(f64, self.col_ptr[self.n]);
        if (self.engine == self.init_engine) return;
        const eng: Engine = switch (self.init_engine) {
            .lu => unreachable, // the general LU is never demoted
            .tri => .{ .tri = try TriDiag.init(self.gpa, self.n, self.col_ptr, self.row_idx) },
            // `init` accepted this pattern and info, and `Bbd.init` is a
            // function of them.
            .bbd => .{ .bbd = (initBbd(self.gpa, self.n, self.col_ptr, self.row_idx, bbd, self.knobs) catch |err| switch (err) {
                error.OutOfMemory => return error.OutOfMemory,
            }).? },
        };
        self.engine.lu.deinit(self.gpa);
        self.engine = eng;
        assert(self.engine == self.init_engine);
    }

    /// The general LU, or null while a structured engine is live.
    pub fn sparse(self: *Self) ?*SparseLu {
        return switch (self.engine) {
            .lu => |*lu| lu,
            .tri, .bbd => null,
        };
    }

    /// Orders the pattern (once) and builds the general LU on it.
    fn initLu(self: *Self) !SparseLu {
        const gpa = self.gpa;
        const n = self.n;
        if (instrument and self.knobs.stats) btfCensus(gpa, n, self.col_ptr, self.row_idx) catch {};
        // One slab for the ordering and the fill bound; both restore it.
        const ws_buf = try gpa.alloc(u32, order_mod.wsSize(n, self.col_ptr[n]));
        defer gpa.free(ws_buf);
        var ws = order_mod.Ws.init(ws_buf);
        if (self.q.len == 0) {
            const q = try gpa.alloc(u32, n);
            errdefer gpa.free(q);
            try order_mod.order(n, self.col_ptr, self.row_idx, q, &ws);
            self.q = q;
        }
        assert(self.q.len == n);
        // At least nnz(A): a diagonal matrix bounds nothing.
        const diag_fill = @max(try order_mod.diagFill(n, self.col_ptr, self.row_idx, self.q, &ws), self.col_ptr[n]);
        assert(ws.used == 0);
        var lu = try SparseLu.init(gpa, n, self.col_ptr, self.q);
        lu.stats = self.knobs.stats;
        lu.diag_fill = diag_fill;
        return lu;
    }

    /// Swaps a singular structured engine for the general LU, for good. The
    /// LU is built first, so out of memory leaves the old engine live.
    fn demote(self: *Self) !void {
        const lu = try self.initLu();
        switch (self.engine) {
            .lu => unreachable,
            .tri => |*tri| tri.deinit(self.gpa),
            .bbd => |*eng| eng.deinit(),
        }
        self.engine = .{ .lu = lu };
        self.factored = false;
        assert(self.sparse() != null);
    }

    fn initBbd(gpa: Allocator, n: u32, col_ptr: []const u32, row_idx: []const u32, bbd: ?num.BbdInfo, knobs: Knobs) !?BbdEng {
        const info = bbd orelse {
            if (instrument and knobs.stats) std.debug.print("lu-census: no BbdInfo\n", .{});
            return null;
        };
        if (knobs.no_bbd) return null;
        const eng = BbdEng.init(gpa, n, col_ptr, row_idx, info, .{}) catch |err| switch (err) {
            error.NotApplicable => {
                if (instrument and knobs.stats) std.debug.print("lu-census: BbdInfo emitted; Bbd.init declined (NotApplicable)\n", .{});
                return null;
            },
            error.OutOfMemory => return error.OutOfMemory,
        };
        if (instrument and knobs.stats) std.debug.print("lu-census: BbdInfo emitted; Bbd.init accepted\n", .{});
        return eng;
    }

    /// Frees the multicore and subtree refactor plans of the current pivot
    /// epoch; the next `refactorLu` on a new epoch prices them again.
    fn dropPlans(self: *Self) void {
        if (self.par) |*p| p.deinit(self.gpa);
        self.par = null;
        if (self.sub) |*p| p.deinit(self.gpa);
        self.sub = null;
    }

    /// Frees every engine and copy; the borrowed pattern stays the caller's.
    pub fn deinit(self: *Self) void {
        self.dropPlans();
        switch (self.engine) {
            .lu => |*lu| lu.deinit(self.gpa),
            .tri => |*tri| tri.deinit(self.gpa),
            .bbd => |*eng| eng.deinit(),
        }
        self.gpa.free(self.q);
        self.gpa.free(self.vcopy);
        self.gpa.free(self.refine_buf);
        self.gpa.free(self.probe);
        self.* = undefined;
    }

    /// Factors `vals` (length nnz, pattern order). Returns at once when
    /// the values equal the last factored ones, which is common on linear
    /// circuits between timestep changes. Otherwise refactors on the
    /// existing pivot sequence and falls back to a full factor if the
    /// replay fails or its probe misses (`replayHolds`). The solver keeps
    /// its own copy of `vals` unless `dropBypass` ran. `execution`
    /// schedules the BBD block factors and the multicore refactor; `.{}`
    /// runs serially. A structured engine that meets a singular pivot is
    /// freed for good and the call retries on the general LU, as is a BBD
    /// factor that fails its refinement probe (`bbd_refine_growth`).
    /// `SingularMatrix` leaves no usable factorization, and neither does
    /// `OutOfMemory`: solve only after a successful `factor`. Asserts that
    /// `vals` holds at least the pattern's nnz entries.
    pub fn factor(self: *Self, vals: []const f64, execution: num.Execution) !void {
        assert(vals.len >= self.col_ptr[self.n]);
        const nnz = self.vcopy.len;
        if (nnz != 0 and self.factored and !self.host_stale and simdEql(self.vcopy, vals[0..nnz])) return;
        if (instrument and self.knobs.stats and self.factored and nnz != 0) if (self.sparse()) |lu| reachCensus(self.gpa, lu, self.col_ptr, self.vcopy, vals) catch {};
        try self.factorInner(vals, execution);
        assert(self.factored);
        self.host_stale = false;
        num.copySimd(self.vcopy, vals[0..nnz]);
        self.gen +%= 1;
    }

    /// True when `vals` equal the last factored values (the bypass test).
    pub fn unchanged(self: *const Self, vals: []const f64) bool {
        return self.vcopy.len != 0 and self.factored and simdEql(self.vcopy, vals[0..self.vcopy.len]);
    }

    /// Frees the copy behind the unchanged-matrix bypass, for a caller whose
    /// values never repeat (a frequency sweep): every `factor` then factors.
    /// Not for a solver a device LU (analysis/gpu/lu.zig) shares, which syncs
    /// through the copy, nor for a BBD engine, whose solves refine against
    /// the copy and go unrefined without it.
    pub fn dropBypass(self: *Self) void {
        self.gpa.free(self.vcopy);
        self.vcopy = &.{};
    }

    /// Brings the LU up to `vcopy` after a device factor. The device ran
    /// this refactor's exact operations and passed its tests, so it cannot
    /// fail here.
    fn syncHost(self: *Self) void {
        if (!self.host_stale) return;
        assert(self.factored);
        self.sparse().?.refactor(self.col_ptr, self.vcopy, self.params.refactor_growth_limit) catch
            @panic("device LU passed a refactor the host fails");
        self.host_stale = false;
    }

    fn factorInner(self: *Self, vals: []const f64, execution: num.Execution) !void {
        // A refactor that fails after overwriting the LU (the probe's OOM)
        // must not leave the old `vcopy` bypassing onto it.
        errdefer self.factored = false;
        // BBD wants one value per pattern entry: drop the trash slot past it.
        const pattern_vals = vals[0..self.col_ptr[self.n]];
        switch (self.engine) {
            .lu => {},
            // Thomas has no pivoting, so an MNA branch row's zero diagonal
            // fails here. ponytail: vsource-bearing ladders lose the
            // tridiagonal speed; a 2x2-block-pivot Thomas would recover it.
            .tri => |*tri| if (tri.factor(vals, pivot_tol)) |_| {
                self.factored = true;
                return;
            } else |_| try self.demote(),
            // A singular block, or a Schur reduction too unstable to refine.
            .bbd => |*eng| if (eng.factorWithExecution(pattern_vals, execution)) |_| {
                if (!(eng.growth > bbd_refine_growth) or (self.probeOne(eng, false, pattern_vals) and self.probeOne(eng, true, pattern_vals))) {
                    self.factored = true;
                    return;
                }
                try self.demote();
            } else |_| try self.demote(),
        }
        const lu = &self.engine.lu;
        lu.fill_cap = repivot_fill_cap;
        lu.keep_min = self.params.refactor_growth_limit;
        if (self.factored) {
            if (self.refactorLu(vals, execution)) |_| {
                if (try self.replayHolds(vals)) return;
            } else |_| {}
            self.factored = false;
        }
        try lu.factor(self.gpa, self.col_ptr, self.row_idx, vals, pivot_tol);
        self.factored = true;
    }

    /// Whether the replay that just refactored `vals` solves as threshold
    /// pivoting promises. The growth monitor admits a pivot down to
    /// `refactor_growth_limit` (1e-12) of its column, for Newton's sake, and
    /// on an ill-conditioned matrix such a replay reached a LAWN 41 ratio
    /// of 1.5e6 where a fresh factor gives 0.019 (#45, #136). A replay
    /// whose multipliers all stay within 1/pivot_tol is one the full factor
    /// could have laid down and passes at once. Otherwise a probe
    /// right-hand side is solved both ways and each scaled residual must
    /// stay within `replay_ratio_max / pivot_tol`, the residual
    /// certificate's bar for threshold pivoting; a miss sends the caller to
    /// a full factor. The check sits here, not in `solve`, so one probe
    /// covers every later `solve`, `solveNeg` and `solveT`, and only a
    /// fallible call re-factors. O(|L|) per replay for the scan; the probe
    /// (two solves, two passes over A) runs only after a marginal replay.
    /// `FreqSolver` replays at pivot_tol, so it never probes. ponytail: a
    /// device factor (`syncHost`, the device LU's own solve) is not
    /// probed; check on the host after the sync if that path needs it.
    fn replayHolds(self: *Self, vals: []const f64) !bool {
        const lu = &self.engine.lu;
        comptime assert(pivot_tol > 0);
        // A replay checks only its pivots. A NaN in L fails the bound below
        // and then the probe, and the full factor refuses the matrix.
        // ponytail: a NaN U entry from a replay reaches the solve, and shows
        // in x unless its column's x is exactly 0; scanning U here cost
        // 0.4-3.6% of a replay. Fold an OR into each refactor kernel's U
        // store if that hole matters.
        if (maxMultiplier(lu) <= 1 / pivot_tol) return true;
        const n: usize = self.n;
        if (self.probe.len == 0) self.probe = try self.gpa.alloc(f64, 3 * n);
        assert(self.probe.len == 3 * n);
        const x = self.probe[0..n];
        const r = self.probe[n..][0..n];
        const rows = self.probe[2 * n ..][0..n];
        // ||A||_inf from the row sums, ||A||_1 = ||A^T||_inf from the columns.
        @memset(rows, 0);
        var norm_1: f64 = 0;
        for (0..n) |j| {
            var col: f64 = 0;
            for (self.col_ptr[j]..self.col_ptr[j + 1]) |p| {
                const a = @abs(vals[p]);
                rows[self.row_idx[p]] += a;
                col += a;
            }
            norm_1 = @max(norm_1, col);
        }
        const norm_inf = num.normInf(rows);
        const nf: f64 = @floatFromInt(n);
        inline for (.{ false, true }) |transpose| {
            probeRhs(r);
            if (transpose) lu.solveT(r, x) else lu.solve(r, x);
            probeRhs(r);
            // r = b - A x, or b - A^T x.
            for (0..n) |j| for (self.col_ptr[j]..self.col_ptr[j + 1]) |p| {
                const i = self.row_idx[p];
                if (transpose) r[j] -= vals[p] * x[i] else r[i] -= vals[p] * x[j];
            };
            const a_norm = if (transpose) norm_1 else norm_inf;
            const ratio = num.normInf(r) / (a_norm * num.normInf(x) * nf * std.math.floatEps(f64));
            // Negated, so a NaN ratio fails too.
            if (!(ratio <= replay_ratio_max / pivot_tol)) return false;
        }
        return true;
    }

    /// `lu.refactor`, on `execution.lu_threads` workers when the flop-count
    /// model below prices that faster. Bitwise the same either way.
    fn refactorLu(self: *Self, vals: []const f64, execution: num.Execution) error{SingularMatrix}!void {
        assert(self.factored);
        const lu = &self.engine.lu;
        const growth = self.params.refactor_growth_limit;
        const threads = execution.lu_threads;
        const io = execution.io orelse return lu.refactor(self.col_ptr, vals, growth);
        if (threads < 2) return lu.refactor(self.col_ptr, vals, growth);
        if (self.par_epoch != lu.pattern_epoch) {
            self.par_epoch = lu.pattern_epoch;
            self.dropPlans();
            // Out of memory for the tables: stay serial this epoch.
            if (parWorthIt(lu, self.knobs)) {
                self.par = lu_kernels.HostRefactor.init(self.gpa, lu, self.col_ptr, growth) catch null;
            } else {
                // Thin factors: lanes over independent subtrees when the
                // model prices them at least `min_speedup` faster.
                self.sub = subtree.Plan.init(self.gpa, lu, self.col_ptr, @min(threads, subtree.max_lanes), .{}) catch null;
                if (self.knobs.par_stats) if (self.sub) |*p| {
                    std.debug.print("lu-subtree: n={d} lanes={d} phases={d} cut={d} model={d:.2}x tail={d:.2}\n", .{ lu.n, p.width, p.phases(), p.cut, p.speedup, p.tail_share });
                    if (instrument) benchSubtree(p, lu, self.col_ptr, vals, growth, io);
                } else std.debug.print("lu-subtree: n={d} declined\n", .{lu.n});
            }
        }
        if (self.par) |*p| {
            lu.replay_mult = null; // its copy-out writes L behind `lu`'s back
            return p.run(lu, vals, io, threads);
        }
        if (self.sub) |*p| return p.run(lu, self.col_ptr, vals, growth, io);
        return lu.refactor(self.col_ptr, vals, growth);
    }

    /// Solves in place with whichever engine is live.
    fn solveInPlace(self: *Self, x: []f64) void {
        assert(self.factored);
        assert(x.len >= self.n);
        self.syncHost();
        switch (self.engine) {
            .lu => |*lu| lu.solve(x[0..self.n], x[0..self.n]),
            .tri => |*tri| tri.solve(x[0..self.n]),
            .bbd => |*eng| self.solveBbd(eng, false, x[0..self.n]),
        }
    }

    /// The BBD solve, refined against `vcopy` when the factor's Schur
    /// growth says it may not be backward stable (#136).
    fn solveBbd(self: *Self, eng: *BbdEng, comptime transpose: bool, x: []f64) void {
        if (!(eng.growth > bbd_refine_growth) or self.vcopy.len == 0) {
            eng.solve(transpose, x);
            return;
        }
        const b = self.refine_buf[0..self.n];
        num.copySimd(b, x);
        eng.solve(transpose, x);
        self.refine(eng, transpose, self.vcopy, b, x);
    }

    /// Solves a fixed pseudo-random right-hand side on the fresh factor of
    /// `vals` (A, or A^T with `transpose`), refines it, and reports whether
    /// the scaled residual meets `probe_ratio_max`. Refinement in working
    /// precision cannot recover a Schur reduction whose error outgrows the
    /// solution, which this catches before any caller solves.
    fn probeOne(self: *Self, eng: *BbdEng, comptime transpose: bool, vals: []const f64) bool {
        const n = self.n;
        const b = self.refine_buf[0..n];
        const r = self.refine_buf[n..][0..n];
        const s = self.refine_buf[2 * n ..][0..n];
        const x = self.refine_buf[3 * n ..][0..n];
        for (b, 0..) |*bi, i| {
            const h = (@as(u64, i) +% 1) *% 0x9E3779B97F4A7C15;
            bi.* = @as(f64, @floatFromInt(h >> 11)) / 0x1p53 * 2 - 1;
        }
        num.copySimd(x, b);
        eng.solve(transpose, x);
        self.refine(eng, transpose, vals, b, x);
        // r = b - op(A) x and s = the row sums of |op(A)|.
        num.copySimd(r, b);
        @memset(s, 0);
        for (0..n) |j| for (self.col_ptr[j]..self.col_ptr[j + 1]) |p| {
            const i = self.row_idx[p];
            const row = if (transpose) j else i;
            r[row] -= vals[p] * x[if (transpose) i else j];
            s[row] += @abs(vals[p]);
        };
        const nf: f64 = @floatFromInt(n);
        // A NaN in the refined solve or its residual fails the probe.
        const ratio = num.normInf(r) / (num.normInf(s) * num.normInf(x) * nf * std.math.floatEps(f64));
        return ratio <= probe_ratio_max;
    }

    /// Fixed-precision iterative refinement of `x` against `vals`, as
    /// LAPACK's dgerfs: stop once the componentwise backward error
    /// max |r_i| / (|A||x| + |b|)_i reaches eps or stops halving.
    fn refine(self: *Self, eng: *BbdEng, comptime transpose: bool, vals: []const f64, b: []const f64, x: []f64) void {
        const n = self.n;
        const r = self.refine_buf[n..][0..n];
        const s = self.refine_buf[2 * n ..][0..n];
        assert(self.refine_buf.len == 4 * @as(usize, n));
        var last: f64 = 3;
        for (0..refine_max) |_| {
            num.copySimd(r, b);
            // ponytail: plain loops on a path only a high-growth BBD factor
            // takes; the sparse walk could not vectorize anyway.
            for (s, b) |*si, bi| si.* = @abs(bi);
            for (0..n) |j| for (self.col_ptr[j]..self.col_ptr[j + 1]) |p| {
                const i = self.row_idx[p];
                const ax = vals[p] * x[if (transpose) i else j];
                const row = if (transpose) j else i;
                r[row] -= ax;
                s[row] += @abs(ax);
            };
            var berr: f64 = 0;
            for (r, s) |ri, si| if (si > 0) {
                berr = @max(berr, @abs(ri) / si);
            };
            if (!(berr > std.math.floatEps(f64) and 2 * berr <= last)) break;
            last = berr;
            eng.solve(transpose, r);
            num.axpy(x, 1, r);
        }
    }

    /// x = -A^-1 rhs, the Newton step. `rhs` and `x` may alias. Asserts that
    /// the last `factor` succeeded and that both slices hold n entries.
    pub fn solveNeg(self: *Self, rhs: []const f64, x: []f64) void {
        assert(rhs.len >= self.n);
        assert(x.len >= self.n);
        num.scale(x[0..self.n], -1, rhs[0..self.n]);
        self.solveInPlace(x);
    }

    /// x = A^-1 rhs. `rhs` and `x` may alias. Asserts as `solveNeg` does.
    pub fn solve(self: *Self, rhs: []const f64, x: []f64) void {
        assert(rhs.len >= self.n);
        assert(x.len >= self.n);
        if (rhs.ptr != x.ptr) @memcpy(x[0..self.n], rhs[0..self.n]);
        self.solveInPlace(x);
    }

    /// x = A^-T rhs, for adjoint analyses. `rhs` and `x` may alias. Asserts
    /// as `solveNeg` does.
    pub fn solveT(self: *Self, rhs: []const f64, x: []f64) void {
        assert(self.factored);
        assert(rhs.len >= self.n);
        assert(x.len >= self.n);
        if (rhs.ptr != x.ptr) @memcpy(x[0..self.n], rhs[0..self.n]);
        self.syncHost();
        switch (self.engine) {
            .lu => |*lu| lu.solveT(x[0..self.n], x[0..self.n]),
            .tri => |*tri| tri.solveT(x[0..self.n]),
            .bbd => |*eng| self.solveBbd(eng, true, x[0..self.n]),
        }
    }
};

/// `-Dlu-instrument` with `ESPICE_LU_PAR_STATS`: the subtree refactor, and
/// the same with its tail run serially, against the serial one on this
/// epoch's first values, medians of 15 interleaved runs each. Leaves `lu`
/// refactored on `vals` (or failed, as the caller's own refactor would).
fn benchSubtree(p: *subtree.Plan, lu: *sparse_lu.SparseLu, col_ptr: []const u32, vals: []const f64, growth: f64, io: std.Io) void {
    const reps = 15;
    var ser: [reps]u64 = undefined;
    var par: [reps]u64 = undefined;
    var scan: [reps]u64 = undefined;
    var stail: [reps]u64 = undefined;
    var sink: f64 = 0;
    const done = p.done;
    defer p.done = done;
    for (0..reps) |i| {
        const t0 = std.Io.Timestamp.now(io, .awake);
        lu.refactor(col_ptr, vals, growth) catch |e| {
            std.debug.print("lu-subtree-bench: refactor {s}\n", .{@errorName(e)});
            return;
        };
        const t1 = std.Io.Timestamp.now(io, .awake);
        p.run(lu, col_ptr, vals, growth, io) catch return;
        const t2 = std.Io.Timestamp.now(io, .awake);
        lu.replay_mult = null;
        sink += maxMultiplier(lu);
        const t3 = std.Io.Timestamp.now(io, .awake);
        p.done = &.{};
        p.run(lu, col_ptr, vals, growth, io) catch return;
        p.done = done;
        const t4 = std.Io.Timestamp.now(io, .awake);
        ser[i] = @intCast(t0.durationTo(t1).nanoseconds);
        par[i] = @intCast(t1.durationTo(t2).nanoseconds);
        scan[i] = @intCast(t2.durationTo(t3).nanoseconds);
        stail[i] = @intCast(t3.durationTo(t4).nanoseconds);
    }
    std.sort.pdq(u64, &stail, {}, std.sort.asc(u64));
    std.debug.print("lu-subtree-bench: serial-tail lanes={d}us\n", .{stail[reps / 2] / 1000});
    std.mem.doNotOptimizeAway(sink);
    std.sort.pdq(u64, &scan, {}, std.sort.asc(u64));
    std.sort.pdq(u64, &ser, {}, std.sort.asc(u64));
    std.sort.pdq(u64, &par, {}, std.sort.asc(u64));
    std.debug.print("lu-subtree-bench: serial={d}us lanes={d}us L-scan={d}us speedup={d:.2}x\n", .{ ser[reps / 2] / 1000, par[reps / 2] / 1000, scan[reps / 2] / 1000, @as(f64, @floatFromInt(ser[reps / 2])) / @as(f64, @floatFromInt(par[reps / 2])) });
}

/// The multicore refactor's admission: enough flops per refactor to pay
/// for the handoffs, and enough per column that the columns are not a
/// dependent chain. `Knobs.par` forces it off or on.
fn parWorthIt(lu: *const sparse_lu.SparseLu, knobs: Knobs) bool {
    assert(lu.factored);
    var flops: u64 = 0;
    for (lu.ui.items) |i| flops += lu.lp[i + 1] - lu.lp[i];
    // Tape-sized factors refactor in microseconds.
    var ok = lu.tv.items.len == 0 and flops >= par_min_flops and flops >= par_min_per_col * @as(u64, lu.n) and flops <= par_max_flops;
    if (knobs.par) |forced| ok = forced;
    if (knobs.par_stats)
        std.debug.print("lu-par: n={d} F={d} F/n={d} admit={}\n", .{ lu.n, flops, flops / @max(lu.n, 1), ok });
    return ok;
}

/// LAPACK's LAWN 41 bar for a backward-stable solve. `replayHolds` divides
/// it by `pivot_tol`, as the residual certificate does for threshold
/// pivoting (residual_tests.zig `ratio_max_pivoting`): multipliers up to
/// 1/pivot_tol are the full factor's own.
const replay_ratio_max = 30.0;

/// The largest |L| entry over the steps the refactor's growth monitor
/// covers, NaN when any L entry is NaN. A replay writes L as v / d, so
/// its smallest accepted pivot ratio is 1 / max(1, this), whichever kernel
/// ran it (tape, columns, threads). Scale-accepted steps are left out of
/// the max: their pivots sit decades below the raw column by design.
fn maxMultiplier(lu: *const sparse_lu.SparseLu) f64 {
    // A successful monitored replay leaves no NaN in L, and its recorded
    // max is the scan's (at least 1, which the replay bar never reaches).
    if (lu.replay_mult) |m| {
        // Safe builds check the recorded max against the scan it replaces.
        if (std.debug.runtime_safety) {
            var scan: f64 = 1;
            for (0..lu.n) |k| if (!lu.scaled_pivot[k]) {
                scan = @max(scan, num.normInf(lu.lx.items[lu.lp[k]..lu.lp[k + 1]]));
            };
            assert(scan == m);
        }
        return m;
    }
    const lx = lu.lx.items;
    const all = num.normInf(lx);
    if (std.math.isNan(all) or std.mem.indexOfScalar(bool, lu.scaled_pivot, true) == null) return all;
    var m: f64 = 0;
    for (0..lu.n) |k| {
        if (lu.scaled_pivot[k]) continue;
        m = @max(m, num.normInf(lx[lu.lp[k]..lu.lp[k + 1]]));
    }
    return m;
}

/// Higham's alternating probe, b[i] = (-1)^i (1 + i / (n - 1)): no
/// structure a circuit matrix shares, the right-hand side LAPACK's
/// condition estimators use (dlacn2's last vector).
fn probeRhs(b: []f64) void {
    const den: f64 = @floatFromInt(@max(b.len, 2) - 1);
    for (b, 0..) |*v, i| {
        const mag = 1 + @as(f64, @floatFromInt(i)) / den;
        v.* = if (i % 2 == 0) mag else -mag;
    }
}

/// Measured on the post-layout and stress decks, 2 to 8 threads
/// (dev/solvers/gpu-lu.md, "Multicore host refactor"): F/n = 445 and up
/// wins at every thread count from F = 0.78M (sram_bsim4_1k); F/n = 152
/// breaks even at 4 and 8 threads and loses at 2; F/n <= 95 loses up to 12x
/// (the columns form a chain, and every handoff is a wait). ponytail: one
/// F/n bar for every thread count; fit per count if a deck lands between.
const par_min_flops = 500_000;
const par_min_per_col = 300;
/// 4 bytes of tables per flop.
const par_max_flops = 400_000_000;
comptime {
    assert(par_min_flops <= par_max_flops);
    // The multicore tables hold u32 offsets.
    assert(par_max_flops <= std.math.maxInt(u32));
}

/// The census for a partial refactor (TODO 2.8): the share of A entries
/// that changed since the last factor, and of pivot steps a replay would
/// have to redo, those whose column changed or whose U names a redone step.
fn reachCensus(gpa: Allocator, lu: *const sparse_lu.SparseLu, col_ptr: []const u32, old: []const f64, vals: []const f64) !void {
    const n = lu.n;
    const redo = try gpa.alloc(bool, n);
    defer gpa.free(redo);
    var changed: u64 = 0;
    var steps: u64 = 0;
    var flops: u64 = 0;
    var redo_flops: u64 = 0;
    for (0..n) |k| {
        const c = lu.q[k];
        var r = false;
        for (col_ptr[c]..col_ptr[c + 1]) |p| if (!(old[p] == vals[p])) {
            changed += 1;
            r = true;
        };
        var f: u64 = 0;
        for (lu.ui.items[lu.up[k]..lu.up[k + 1]]) |i| {
            r = r or redo[i];
            f += lu.lp[i + 1] - lu.lp[i];
        }
        redo[k] = r;
        flops += f;
        if (r) {
            steps += 1;
            redo_flops += f;
        }
    }
    const pct = struct {
        fn f(a: u64, b: u64) f64 {
            return 100 * @as(f64, @floatFromInt(a)) / @as(f64, @floatFromInt(@max(b, 1)));
        }
    }.f;
    std.debug.print("lu-reach: changed={d:.2}% steps={d:.2}% flops={d:.2}%\n", .{ pct(changed, col_ptr[n]), pct(steps, n), pct(redo_flops, flops) });
}

/// The census for keeping the BTF blocks through the factor (TODO 2.5):
/// the block count, singletons, the largest block and the entries outside
/// every diagonal block, which a block factor would leave unfactored.
fn btfCensus(gpa: Allocator, n: u32, col_ptr: []const u32, row_idx: []const u32) !void {
    const buf = try gpa.alloc(u32, 3 * @as(usize, n) + 1 + order_mod.wsSize(n, col_ptr[n]));
    defer gpa.free(buf);
    const emit = buf[0..n];
    const block_ptr = buf[n..][0 .. n + 1];
    const blk_of = buf[2 * n + 1 ..][0..n];
    var ws = order_mod.Ws.init(buf[3 * n + 1 ..]);
    const nb = try order_mod.blocks(n, col_ptr, row_idx, emit, block_ptr, &ws);
    var largest: u32 = 0;
    var singletons: u32 = 0;
    for (0..nb) |b| {
        const size = block_ptr[b + 1] - block_ptr[b];
        largest = @max(largest, size);
        singletons += @intFromBool(size == 1);
        for (emit[block_ptr[b]..block_ptr[b + 1]]) |v| blk_of[v] = @intCast(b);
    }
    var off: u64 = 0;
    for (0..n) |j| for (row_idx[col_ptr[j]..col_ptr[j + 1]]) |r| {
        off += @intFromBool(blk_of[r] != blk_of[j]);
    };
    std.debug.print("lu-btf: n={d} blocks={d} singletons={d} largest={d} ({d:.1}% of n) off_block_nnz={d} ({d:.1}% of nnz)\n", .{
        n,                                                                            nb,  singletons,                                                                        largest,
        100 * @as(f64, @floatFromInt(largest)) / @as(f64, @floatFromInt(@max(n, 1))), off, 100 * @as(f64, @floatFromInt(off)) / @as(f64, @floatFromInt(@max(col_ptr[n], 1))),
    });
}

/// IEEE equality per element (-0 == +0, NaN != NaN) and equal lengths,
/// not `std.mem.eql`'s bytes. Chunked: each 64-entry block is a branchless
/// OR the loop vectorizer takes, one branch per block: 0.08-0.10 ns per
/// entry against 0.12-0.17 for the iterator-library `eq` it replaced
/// (ReleaseFast, n = 64 to 16384).
fn simdEql(a: []const f64, b: []const f64) bool {
    if (a.len != b.len) return false;
    const block = 64;
    var i: usize = 0;
    while (i + block <= a.len) : (i += block) {
        var ne: u64 = 0;
        for (a[i..][0..block], b[i..][0..block]) |x, y| ne |= @intFromBool(!(x == y));
        if (ne != 0) return false;
    }
    var ne: u64 = 0;
    for (a[i..], b[i..]) |x, y| ne |= @intFromBool(!(x == y));
    return ne == 0;
}

const testing = std.testing;

test simdEql {
    const Ref = struct {
        fn scalar(a: []const f64, b: []const f64) bool {
            if (a.len != b.len) return false;
            for (a, b) |x, y| if (x != y) return false;
            return true;
        }
        fn check(a: []const f64, b: []const f64) !void {
            const want = scalar(a, b);
            try testing.expectEqual(want, simdEql(a, b));
        }
    };
    var buf_a: [208]f64 = undefined;
    var buf_b: [208]f64 = undefined;
    var prng = std.Random.DefaultPrng.init(0xe91);
    const r = prng.random();
    for (&buf_a, &buf_b) |*a, *b| {
        a.* = r.float(f64) - 0.5;
        b.* = a.*;
    }
    // Past 3x the widest f64 lane count, at several misalignments, with
    // one planted change at a spread of positions: a different value, a
    // NaN on both sides (unequal), zeros of opposite sign (equal).
    for (0..201) |n| for (0..3) |off| {
        const a = buf_a[off..][0..n];
        const b = buf_b[off..][0..n];
        try Ref.check(a, b);
        if (n > 0) try Ref.check(a, b[0 .. n - 1]);
        const step = @max(1, n / 16);
        for (0..n) |i| {
            if (i % step != 0 and i != n - 1) continue;
            const ka = a[i];
            const kb = b[i];
            defer {
                a[i] = ka;
                b[i] = kb;
            }
            b[i] = kb + 1;
            try Ref.check(a, b);
            a[i] = std.math.nan(f64);
            b[i] = a[i];
            try Ref.check(a, b);
            a[i] = 0;
            b[i] = -0.0;
            try Ref.check(a, b);
        }
    };
}

test "Solver: a tridiagonal pattern with a zero diagonal demotes to the pivoting LU" {
    // A = [0 1 0; 1 2 4; 0 1 3]: row 0 is an MNA branch row with no
    // diagonal value, so Thomas fails at its first pivot.
    const gpa = testing.allocator;
    const col_ptr = [_]u32{ 0, 2, 5, 7 };
    const row_idx = [_]u32{ 0, 1, 0, 1, 2, 1, 2 };
    var vals = [_]f64{ 0, 1, 1, 2, 1, 4, 3 };
    var s = try Solver.init(gpa, 3, &col_ptr, &row_idx, null, .{});
    defer s.deinit();
    try testing.expect(s.engine == .tri);
    try s.factor(&vals, .{});
    try testing.expect(s.engine == .lu);

    // A [1 2 3] = [2 17 11], A^T [1 2 3] = [2 8 17].
    var x: [3]f64 = undefined;
    s.solve(&.{ 2, 17, 11 }, &x);
    for (x, [_]f64{ 1, 2, 3 }) |xi, want| try testing.expectApproxEqAbs(want, xi, 1e-12);
    s.solveT(&.{ 2, 8, 17 }, &x);
    for (x, [_]f64{ 1, 2, 3 }) |xi, want| try testing.expectApproxEqAbs(want, xi, 1e-12);

    // The same values again are the bypass: no new generation. A NaN never
    // matches, so it always refactors.
    const gen = s.gen;
    try s.factor(&vals, .{});
    try testing.expectEqual(gen, s.gen);
    try testing.expect(s.unchanged(&vals));
    vals[3] = std.math.nan(f64);
    try testing.expect(!s.unchanged(&vals));
}

test "Solver: dropBypass makes every factor refactor; solveNeg is -solve" {
    // A = [4 1 0 1; 1 4 1 0; 0 1 4 1; 1 0 1 4], a ring: not tridiagonal, so
    // the general LU.
    const gpa = testing.allocator;
    const col_ptr = [_]u32{ 0, 3, 6, 9, 12 };
    const row_idx = [_]u32{ 0, 1, 3, 0, 1, 2, 1, 2, 3, 0, 2, 3 };
    const vals = [_]f64{ 4, 1, 1, 1, 4, 1, 1, 4, 1, 1, 1, 4 };
    var s = try Solver.init(gpa, 4, &col_ptr, &row_idx, null, .{});
    defer s.deinit();
    try testing.expect(s.engine == .lu);
    try s.factor(&vals, .{});
    const gen = s.gen;
    try s.factor(&vals, .{});
    try testing.expectEqual(gen, s.gen);
    s.dropBypass();
    try testing.expect(!s.unchanged(&vals));
    try s.factor(&vals, .{});
    try testing.expectEqual(gen +% 1, s.gen);
    // A [1 2 3 4] = [10 12 18 20].
    const b = [_]f64{ 10, 12, 18, 20 };
    var x: [4]f64 = undefined;
    s.solve(&b, &x);
    for (x, [_]f64{ 1, 2, 3, 4 }) |xi, want| try testing.expectApproxEqAbs(want, xi, 1e-13);
    var y = b;
    s.solveNeg(&y, &y); // aliased
    for (y, [_]f64{ 1, 2, 3, 4 }) |yi, want| try testing.expectApproxEqAbs(-want, yi, 1e-13);
}

test "Solver: a singular matrix fails factor with SingularMatrix and leaves no factor" {
    // The ring pattern above with columns 0 and 1 both [1 1 0 0]^T: rank 3.
    const gpa = testing.allocator;
    const col_ptr = [_]u32{ 0, 3, 6, 9, 12 };
    const row_idx = [_]u32{ 0, 1, 3, 0, 1, 2, 1, 2, 3, 0, 2, 3 };
    const vals = [_]f64{ 1, 1, 0, 1, 1, 0, 0, 2, 3, 1, 5, 7 };
    var s = try Solver.init(gpa, 4, &col_ptr, &row_idx, null, .{});
    defer s.deinit();
    try testing.expectError(error.SingularMatrix, s.factor(&vals, .{}));
    try testing.expect(!s.factored);
    try testing.expect(!s.unchanged(&vals));
}

test "maxMultiplier is NaN when an L entry is NaN, scale-accepted steps included" {
    const gpa = testing.allocator;
    // diag(4) plus A[2][0]: L[2][0] is the only multiplier. A replay
    // fails on a NaN there, so the test plants it in the factors.
    const col_ptr = [_]u32{ 0, 2, 3, 4 };
    const row_idx = [_]u32{ 0, 2, 1, 2 };
    const q = [_]u32{ 0, 1, 2 };
    var lu = try sparse_lu.SparseLu.init(gpa, 3, &col_ptr, &q);
    defer lu.deinit(gpa);
    try lu.factor(gpa, &col_ptr, &row_idx, &.{ 4, 1, 4, 4 }, pivot_tol);
    try testing.expectError(error.SingularMatrix, lu.refactor(&col_ptr, &.{ 4, std.math.nan(f64), 4, 4 }, 1e-12));
    try lu.refactor(&col_ptr, &.{ 4, 1, 4, 4 }, 1e-12);
    lu.lx.items[0] = std.math.nan(f64);
    try testing.expect(std.math.isNan(maxMultiplier(&lu)));
    lu.scaled_pivot[1] = true;
    try testing.expect(std.math.isNan(maxMultiplier(&lu)));
}

test "Solver.reset: a reset solver factors and solves bitwise as a fresh one" {
    const gpa = testing.allocator;
    // The ring of the dropBypass test, and the tridiagonal pattern whose
    // zero diagonal demotes Thomas to the LU.
    const Case = struct { n: u32, cp: []const u32, ri: []const u32, a: []const f64, b: []const f64 };
    const cases = [_]Case{
        .{ .n = 4, .cp = &.{ 0, 3, 6, 9, 12 }, .ri = &.{ 0, 1, 3, 0, 1, 2, 1, 2, 3, 0, 2, 3 }, .a = &.{ 1e-6, 1, 1, 1, 4, 1, 1, 4, 1, 1, 1, 4 }, .b = &.{ 4, 1, 1, 1, 1e-9, 1, 1, 4, 1, 1, 1, 4 } },
        .{ .n = 3, .cp = &.{ 0, 2, 5, 7 }, .ri = &.{ 0, 1, 0, 1, 2, 1, 2 }, .a = &.{ 0, 1, 1, 2, 1, 4, 3 }, .b = &.{ 2, 1, 1, 2, 1, 4, 3 } },
    };
    for (cases) |c| {
        var a = try Solver.init(gpa, c.n, c.cp, c.ri, null, .{});
        defer a.deinit();
        var b = try Solver.init(gpa, c.n, c.cp, c.ri, null, .{});
        defer b.deinit();
        try a.factor(c.a, .{});
        a.dropBypass();
        try a.reset(null);
        try testing.expect(std.meta.activeTag(a.engine) == std.meta.activeTag(b.engine) and !a.factored);
        try a.factor(c.b, .{});
        try b.factor(c.b, .{});
        try testing.expect(std.meta.activeTag(a.engine) == std.meta.activeTag(b.engine));
        if (a.sparse()) |la| {
            const lb = b.sparse().?;
            try testing.expectEqualSlices(u32, lb.pinv, la.pinv);
            try testing.expectEqualSlices(u8, std.mem.sliceAsBytes(lb.lx.items), std.mem.sliceAsBytes(la.lx.items));
            try testing.expectEqualSlices(u8, std.mem.sliceAsBytes(lb.ux.items), std.mem.sliceAsBytes(la.ux.items));
            try testing.expectEqualSlices(u8, std.mem.sliceAsBytes(lb.udiag), std.mem.sliceAsBytes(la.udiag));
        }
        var xa: [4]f64 = undefined;
        var xb: [4]f64 = undefined;
        const rhs = [_]f64{ 1, -2, 3, -4 };
        a.solve(rhs[0..c.n], xa[0..c.n]);
        b.solve(rhs[0..c.n], xb[0..c.n]);
        try testing.expectEqualSlices(u8, std.mem.sliceAsBytes(xb[0..c.n]), std.mem.sliceAsBytes(xa[0..c.n]));
        // The bypass copy is back.
        try testing.expect(a.unchanged(c.b));
    }
}
