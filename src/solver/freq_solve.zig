//! (G + jωC) x = b solves in the stacked-real form [G, -ωC; ωC, G], dense
//! for small circuits and sparse above. The sparse 2n pattern is derived once
//! from the circuit CSC, so each frequency's fill is a streamed copy of the G
//! and C planes. Batches of frequencies run as SIMD lanes through LaneLu.
//!
//! In: a circuit's G and C planes (or a dense G and C) and right-hand sides
//! per ω. Out: stacked-real solutions, real parts then imaginary. Cites
//! dev/solvers/overview.md and dev/solvers/gilbert-peierls-lu.md ("Two
//! replay thresholds", #44); #43 (held singular ω in `factorEach`), #83
//! (allocation failure is not a numerical one).
//!
//! Contents:
//! - `pub const Dyn = struct`
//!   - `pub fn at(self: Dyn, e: usize, k: usize) struct { re: f64, im: f64 }`
//! - `pub const Stacked = freq_fill.Stacked`
//! - `pub const ComplexFill = freq_fill.ComplexFill`
//! - `pub const StackedPoints = freq_fill.StackedPoints`
//! - `pub const FreqSolver = struct`
//!   - `pub const W = std.simd.suggestVectorLength(f64) orelse 1`
//!   - `pub const Dense = struct`
//!   - `pub const Sparse = struct`
//!   - `pub const Complex = struct`
//!     - `pub fn init(gpa: Allocator, n: u32, col_ptr: []const u32, row_idx: []const u32) !?Complex`
//!     - `pub fn deinit(self: *Complex, gpa: Allocator) void`
//!   - `pub fn fromPlanes(allocator: Allocator, n: u32, col_ptr: []const u32, row_idx: []const u32, g: []const f64, c: []const f64) !Self`
//!   - `pub fn fromCircuit(allocator: Allocator, ckt: anytype, x_op: []const f64) !Self`
//!   - `pub fn initDense(allocator: Allocator, n: u32, g: []f64, c: []f64) !Self`
//!   - `pub fn deinit(self: *Self) void`
//!   - `pub fn solve(self: *Self, omega: f64, rhs: []const f64, x_out: []f64) !void`
//!   - `pub fn setOmegaDyn(self: *Self, omega: f64, dyn: Dyn, k: usize) !void`
//!   - `pub fn solveRhs(self: *Self, rhs: []const f64, x_out: []f64) void`
//!   - `pub fn solveRhsT(self: *Self, rhs: []const f64, x_out: []f64) void`
//!   - `pub fn addDiagG(self: *Self, i: u32, v: f64) void`
//!   - `pub const solveBatch = freq_batch.solveBatch`
//!   - `pub const ComplexChunk = freq_batch.ComplexChunk`
//!   - `pub const beginComplexChunk = freq_batch.beginComplexChunk`
//!   - `pub const finishComplexChunk = freq_batch.finishComplexChunk`
//!   - `pub const LaneChunk = freq_batch.LaneChunk`
//!   - `pub const beginLaneChunk = freq_batch.beginLaneChunk`
//!   - `pub const finishLaneChunk = freq_batch.finishLaneChunk`
//!   - `pub const heldTape = freq_held.heldTape`
//!   - `pub const beginHeldTape = freq_held.beginHeldTape`
//!   - `pub const finishHeldTape = freq_held.finishHeldTape`
//!   - `pub const replayCurrentHeld = freq_held.replayCurrentHeld`
//!   - `pub const completeHeldPolicy = freq_held.completeHeldPolicy`
//!   - `pub const ensureHeldOnHost = freq_held.ensureHeldOnHost`
//!   - `pub const packHeld = freq_held.packHeld`
//!   - `pub fn setPlanes(self: *Self, g: []const f64, c: []const f64) void`
//!   - `pub const factorEach = freq_held.factorEach`
//!   - `pub const solveEach = freq_held.solveEach`
//!   - `pub const test_access = if (@import("builtin").is_test) .{`
//! - `pub fn slotCol(col_ptr: []const u32, slot: u32) usize`
//! - `pub fn stackedPos(src_col_ptr: []const u32, slot: u32) [4]usize`
//! - `pub fn addDense(n: usize, nn: usize, a: []f64, row: usize, col: usize, re: f64, im: f64) void`

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
const freq_batch = @import("freq_batch.zig");
const freq_held = @import("freq_held.zig");
const freq_fill = @import("freq_fill.zig");
const testing = std.testing;

/// The part of A(ω) beyond G + jωC over the ω of one `solveBatch` call:
/// entry e adds re + j·im at source CSC slot `slots[e]`, its term at
/// `omegas[k]` being `re[e * omegas.len + k]` and `im[...]`. A slot at or past
/// the pattern's nnz (a ground entry) adds nothing. The default is empty.
pub const Dyn = struct {
    slots: []const u32 = &.{},
    re: []const f64 = &.{},
    im: []const f64 = &.{},

    /// Entry e's term at ω index k.
    pub fn at(self: Dyn, e: usize, k: usize) struct { re: f64, im: f64 } {
        assert(e < self.slots.len);
        assert(k < self.re.len / self.slots.len);
        const i = e * (self.re.len / self.slots.len) + k;
        return .{ .re = self.re[i], .im = self.im[i] };
    }
};

/// The stacked-real lane source of A(ω) (`freq_fill.zig`).
pub const Stacked = freq_fill.Stacked;

/// The complex n x n lane source of A(ω) (`freq_fill.zig`).
pub const ComplexFill = freq_fill.ComplexFill;

/// The per-point lane source of A(ω) (`freq_fill.zig`).
pub const StackedPoints = freq_fill.StackedPoints;

/// Frequency-domain solver. Solution and right-hand side
/// vectors are length 2n: real parts, then imaginary parts.
pub const FreqSolver = struct {
    const Self = @This();

    /// Every allocation the solver owns or makes later (`factorEach`,
    /// `deinit`) uses this, the allocator it was built with (#13).
    gpa: Allocator,
    n: u32,
    nn: u32,
    strategy: Strategy,
    /// `factorEach`'s ω indices held as the identity: a singular block, or
    /// a lane no pivot tape it tried could replay.
    held_id: std.ArrayList(u32) = .empty,
    /// A completed factorEach policy can have numeric factors only on the
    /// device. Recovery materializes that exact final tape on the host;
    /// restarting the policy would select extra pivots.
    held_location: enum { invalid, host, device, recovery_failed } = .invalid,
    held_count: usize = 0,
    /// `solveBatch` runs its sparse lanes in complex arithmetic on an n x n
    /// tape (`Complex`); false keeps the stacked-real LaneLu, whose lanes
    /// are bitwise the scalar path's. The device sweep replays the same
    /// complex tapes (`beginComplexChunk`); the live round driver clears it
    /// so its host fallback stays bitwise its stacked replay.
    complex_lanes: bool = true,
    /// `factorEach` holds complex factors on one n x n tape (`Complex`)
    /// when every ω replays it, else its stacked-real policy. Off by
    /// default: a device provider reads stacked held tapes (`heldTape`).
    complex_held: bool = false,

    /// Frequencies per LaneLu pass; `solveBatch` groups its ω from the
    /// first in runs of W.
    pub const W = std.simd.suggestVectorLength(f64) orelse 1;

    /// n <= dense_threshold: owned n x n row-major G and C, assembled per
    /// frequency into the reused 2n x 2n LU slab. The borrowed source
    /// pattern places `Dyn` entries; empty from `initDense`.
    pub const Dense = struct {
        src_col_ptr: []const u32 = &.{},
        src_row_idx: []const u32 = &.{},
        g_dense: []f64,
        c_mat: []f64,
        a_lu: []f64,
        piv: []u32,
        /// `factorEach`'s factors, one 2n x 2n LU and pivot row per ω.
        held: []f64 = &.{},
        held_piv: []u32 = &.{},
    };

    /// Owns the 2n stacked-real CSC, its solver and a snapshot of the
    /// circuit's G and C planes; borrows only the frozen source pattern.
    pub const Sparse = struct {
        g_vals: []f64,
        c_vals: []f64,
        src_col_ptr: []const u32,
        col_ptr: []u32,
        row_idx: []u32,
        vals: []f64,
        slv: direct.Solver,
        /// Lane scratch: the RHS and solution planes. Allocated by the
        /// first `solveBatch` or `factorEach`; the matrix values never exist
        /// as a lane plane (`Stacked` computes each column as it is read).
        lane_work: []@Vector(W, f64) = &.{},
        /// The current `Dyn`'s entries grouped by source column: column j's
        /// are `dyn_ent[dyn_ptr[j]..dyn_ptr[j + 1]]`, in entry order, ground
        /// entries left out. Rebuilt by every `solveBatch` with a `Dyn`.
        dyn_ptr: []u32 = &.{},
        dyn_ent: []u32 = &.{},
        lanes: ?lane_lu.LaneLu(W) = null,
        /// `factorEach`'s factors: W ω per group, all replaying `slv.lu`'s
        /// pivot tape.
        held: []lane_lu.LaneLu(W) = &.{},
        /// The circuit's source row indices, borrowed like `src_col_ptr`.
        src_row_idx: []const u32,
        /// `solveBatch`'s complex lanes, built on first use.
        cx: ?Complex = null,
        /// `factorEach`'s complex factors: W ω per group on `cx.lu`. Empty
        /// whenever `held` is in use.
        cx_held: []ComplexLaneLu(W) = &.{},
    };

    /// The complex frequency lanes: a real SparseLu laid on
    /// the n x n source pattern from the moduli |A(ω)_ij| at one chunk's
    /// centre is the pivot tape, and `ComplexLaneLu` replays W frequencies
    /// of A(ω) on it with its own growth test. A lane that fails takes the
    /// stacked-real scalar path and the next chunk lays a fresh tape.
    pub const Complex = struct {
        q: []u32,
        proxy: []f64,
        lu: SparseLu,
        lanes: ?ComplexLaneLu(W) = null,
        /// `lu` holds a tape the lanes may replay.
        tape: bool = false,
        /// b re, b im, x re, x im: n lanes each.
        work: []@Vector(W, f64),

        /// Orders the source pattern and lays the proxy tape; null when the
        /// ordering's fill cap refuses it. Free with `deinit`.
        pub fn init(gpa: Allocator, n: u32, col_ptr: []const u32, row_idx: []const u32) !?Complex {
            const q = try gpa.alloc(u32, n);
            errdefer gpa.free(q);
            {
                const ws_buf = try gpa.alloc(u32, order_mod.wsSize(n, col_ptr[n]));
                defer gpa.free(ws_buf);
                var ws = order_mod.Ws.init(ws_buf);
                order_mod.order(n, col_ptr, row_idx, q, &ws) catch |err| switch (err) {
                    // Pathological fill: the stacked lanes stay.
                    error.OutOfWorkspace => {
                        gpa.free(q);
                        return null;
                    },
                };
            }
            const proxy = try gpa.alloc(f64, col_ptr[n]);
            errdefer gpa.free(proxy);
            const work = try gpa.alloc(@Vector(W, f64), 4 * @as(usize, n));
            errdefer gpa.free(work);
            const lu = try SparseLu.init(gpa, n, col_ptr, q);
            return .{ .q = q, .proxy = proxy, .lu = lu, .work = work };
        }

        /// Frees the tape, the lanes and the work planes.
        pub fn deinit(self: *Complex, gpa: Allocator) void {
            if (self.lanes) |*l| l.deinit(gpa);
            self.lu.deinit(gpa);
            gpa.free(self.q);
            gpa.free(self.proxy);
            gpa.free(self.work);
        }
    };

    /// A solver over the CSC pattern `col_ptr`/`row_idx` (n columns,
    /// borrowed for the solver's lifetime) with G and C given per slot, for
    /// callers whose planes are not a circuit evaluation. Nothing is
    /// evaluated. Caller frees with `deinit` and the same allocator. Asserts
    /// that `col_ptr` has n + 1 entries and `g` and `c` one value per slot.
    pub fn fromPlanes(allocator: Allocator, n: u32, col_ptr: []const u32, row_idx: []const u32, g: []const f64, c: []const f64) !Self {
        assert(col_ptr.len == @as(usize, n) + 1);
        assert(g.len >= col_ptr[n]);
        assert(c.len >= col_ptr[n]);
        const Planes = struct {
            n: usize,
            nnz: usize,
            col_ptr: []const u32,
            row_idx: []const u32,
            g_vals: []const f64,
            c_vals: []const f64,
            fn linearizeAc(_: @This(), _: []const f64) !void {}
            fn denseG(p: @This(), out: []f64) void {
                p.dense(p.g_vals, out);
            }
            fn denseC(p: @This(), out: []f64) void {
                p.dense(p.c_vals, out);
            }
            fn dense(p: @This(), vals: []const f64, out: []f64) void {
                @memset(out[0 .. p.n * p.n], 0);
                for (0..p.n) |j| for (p.col_ptr[j]..p.col_ptr[j + 1]) |s| {
                    out[@as(usize, p.row_idx[s]) * p.n + j] += vals[s];
                };
            }
        };
        return fromCircuit(allocator, Planes{ .n = n, .nnz = col_ptr[n], .col_ptr = col_ptr, .row_idx = row_idx, .g_vals = g, .c_vals = c }, &.{});
    }

    /// Linearizes `ckt` at `x_op` (one eval; its G and C planes are the
    /// linearization) and builds the solver from copies of them, so the
    /// circuit may be re-evaluated afterwards. The sparse path borrows
    /// the circuit's CSC pattern for the solver's lifetime, and so does the
    /// dense one, to place `Dyn` entries and `setPlanes`. Caller frees with
    /// `deinit` and the same allocator.
    pub fn fromCircuit(allocator: Allocator, ckt: anytype, x_op: []const f64) !Self {
        try ckt.linearizeAc(x_op);
        const n: u32 = @intCast(ckt.n);

        if (n <= dense_threshold) {
            const nu: usize = n;
            const g = try allocator.alloc(f64, nu * nu);
            ckt.denseG(g);
            const c = allocator.alloc(f64, nu * nu) catch |err| {
                allocator.free(g);
                return err;
            };
            ckt.denseC(c);
            var fs = try initDense(allocator, n, g, c);
            fs.strategy.dense.src_col_ptr = ckt.col_ptr;
            fs.strategy.dense.src_row_idx = ckt.row_idx;
            return fs;
        }

        return initSparse(allocator, n, ckt);
    }

    /// Dense solver over row-major n x n `g` and `c`, allocated with
    /// `allocator`; takes ownership of both, also on error. Asserts that `g`
    /// and `c` are n x n.
    pub fn initDense(allocator: Allocator, n: u32, g: []f64, c: []f64) !Self {
        assert(g.len == @as(usize, n) * n);
        assert(c.len == g.len);
        errdefer allocator.free(g);
        errdefer allocator.free(c);
        const nn: u32 = 2 * n;
        const nnu: usize = nn;
        const a_lu = try allocator.alloc(f64, nnu * nnu);
        errdefer allocator.free(a_lu);
        const piv = try allocator.alloc(u32, nnu);

        return .{
            .gpa = allocator,
            .n = n,
            .nn = nn,
            .strategy = .{ .dense = .{
                .g_dense = g,
                .c_mat = c,
                .a_lu = a_lu,
                .piv = piv,
            } },
        };
    }

    /// Frees everything the solver owns.
    pub fn deinit(self: *Self) void {
        const allocator = self.gpa;
        self.held_id.deinit(allocator);
        switch (self.strategy) {
            .dense => |*d| {
                allocator.free(d.held);
                allocator.free(d.held_piv);
                allocator.free(d.g_dense);
                allocator.free(d.c_mat);
                allocator.free(d.a_lu);
                allocator.free(d.piv);
            },
            .sp => |*s| {
                freq_held.freeHeld(allocator, s);
                freq_held.freeComplexHeld(allocator, s);
                if (s.cx) |*c| c.deinit(allocator);
                if (s.lanes) |*l| l.deinit(allocator);
                allocator.free(s.lane_work);
                allocator.free(s.dyn_ptr);
                allocator.free(s.dyn_ent);
                s.slv.deinit();
                allocator.free(s.col_ptr);
                allocator.free(s.row_idx);
                allocator.free(s.vals);
                allocator.free(s.g_vals);
                allocator.free(s.c_vals);
            },
        }
    }

    /// Factors A(ω) = G + jωC and solves one stacked right-hand side; `rhs`
    /// and `x_out` are 2n long and may alias. Fails with `error.Singular`
    /// (dense) or `error.SingularMatrix` (sparse) on a singular A(ω), or
    /// `OutOfMemory`, leaving `x_out` untouched and no usable factor. Asserts
    /// that both slices hold 2n entries.
    pub fn solve(self: *Self, omega: f64, rhs: []const f64, x_out: []f64) !void {
        assert(rhs.len >= self.nn);
        assert(x_out.len >= self.nn);
        try self.setOmega(omega);
        self.solveRhs(rhs, x_out);
    }

    /// `setOmega` plus `dyn`'s terms at ω index `k`.
    pub fn setOmegaDyn(self: *Self, omega: f64, dyn: Dyn, k: usize) !void {
        self.held_location = .invalid;
        if (dyn.slots.len != 0) assert(k < dyn.re.len / dyn.slots.len);
        switch (self.strategy) {
            .dense => |*d| try freq_fill.setOmegaDense(self.n, self.nn, d, omega, dyn, k),
            .sp => |*s| try freq_fill.setOmegaSparse(self.n, s, omega, dyn, k),
        }
    }

    // Solves with the current factorization, which `setOmega` must have
    // left; both slices are 2n long and may alias.
    /// Solves A(ω) x = rhs for the ω the last `setOmega` filled.
    pub fn solveRhs(self: *Self, rhs: []const f64, x_out: []f64) void {
        self.held_location = .invalid;
        switch (self.strategy) {
            .dense => |*d| dense_lu.solveFactored(self.nn, d.a_lu, d.piv, rhs, x_out),
            .sp => |*s| s.slv.solve(rhs, x_out),
        }
    }

    // Adjoint solve with the current factorization: the transpose of the
    // stacked-real system, which as a complex system is A^H x = rhs
    // (A = G + jωC, so A^H = G^T − jωC^T). A caller that needs A^T's
    // solution conjugates this one; |x| is the same either way.
    /// Solves A(ω)^T x = rhs for the ω the last `setOmega` filled.
    pub fn solveRhsT(self: *Self, rhs: []const f64, x_out: []f64) void {
        self.held_location = .invalid;
        switch (self.strategy) {
            .dense => |*d| dense_lu.solveFactoredT(self.nn, d.a_lu, d.piv, rhs, x_out),
            .sp => |*s| s.slv.solveT(rhs, x_out),
        }
    }

    /// Adds `v` to entry (i, i) of the solver's copy of G; the circuit is
    /// untouched. Call it before the first solve: a factorization already
    /// held does not see it. Asserts that `i < n`.
    pub fn addDiagG(self: *Self, i: u32, v: f64) void {
        self.held_location = .invalid;
        assert(i < self.n);
        switch (self.strategy) {
            .dense => |*d| d.g_dense[@as(usize, i) * self.n + i] += v,
            .sp => |*s| {
                // Stacked column i opens with source column i's rows.
                const cs = s.src_col_ptr[i];
                const rows = s.row_idx[s.col_ptr[i]..][0 .. s.src_col_ptr[i + 1] - cs];
                // Circuit.freeze guarantees every diagonal is in the pattern.
                s.g_vals[cs + std.mem.indexOfScalar(u32, rows, i).?] += v;
            },
        }
    }

    /// See `freq_batch.solveBatch`.
    pub const solveBatch = freq_batch.solveBatch;

    /// See `freq_batch.ComplexChunk`.
    pub const ComplexChunk = freq_batch.ComplexChunk;

    /// See `freq_batch.beginComplexChunk`.
    pub const beginComplexChunk = freq_batch.beginComplexChunk;

    /// See `freq_batch.finishComplexChunk`.
    pub const finishComplexChunk = freq_batch.finishComplexChunk;

    /// See `freq_batch.LaneChunk`.
    pub const LaneChunk = freq_batch.LaneChunk;

    /// See `freq_batch.beginLaneChunk`.
    pub const beginLaneChunk = freq_batch.beginLaneChunk;

    /// See `freq_batch.finishLaneChunk`.
    pub const finishLaneChunk = freq_batch.finishLaneChunk;

    /// See `freq_held.heldTape`.
    pub const heldTape = freq_held.heldTape;

    /// See `freq_held.beginHeldTape`.
    pub const beginHeldTape = freq_held.beginHeldTape;

    /// See `freq_held.finishHeldTape`.
    pub const finishHeldTape = freq_held.finishHeldTape;

    /// See `freq_held.replayCurrentHeld`.
    pub const replayCurrentHeld = freq_held.replayCurrentHeld;

    /// See `freq_held.completeHeldPolicy`.
    pub const completeHeldPolicy = freq_held.completeHeldPolicy;

    /// See `freq_held.ensureHeldOnHost`.
    pub const ensureHeldOnHost = freq_held.ensureHeldOnHost;

    /// See `freq_held.packHeld`.
    pub const packHeld = freq_held.packHeld;

    /// Replaces the solver's G and C with `g` and `c`, indexed by CSC slot
    /// of the source pattern (a circuit's `g_vals`/`c_vals` layout). The
    /// dense strategy places them through the pattern `fromCircuit`
    /// borrowed. Asserts that the solver has that pattern (not
    /// `initDense`'s) and that `g` and `c` cover it.
    pub fn setPlanes(self: *Self, g: []const f64, c: []const f64) void {
        self.held_location = .invalid;
        assert(g.len == c.len);
        switch (self.strategy) {
            .dense => |*d| {
                const n: usize = self.n;
                assert(d.src_col_ptr.len == n + 1);
                assert(g.len >= d.src_col_ptr[n]);
                @memset(d.g_dense, 0);
                @memset(d.c_mat, 0);
                for (0..n) |j| for (d.src_col_ptr[j]..d.src_col_ptr[j + 1]) |p| {
                    const at = @as(usize, d.src_row_idx[p]) * n + j;
                    d.g_dense[at] += g[p];
                    d.c_mat[at] += c[p];
                };
            },
            .sp => |*s| {
                assert(g.len >= s.g_vals.len);
                @memcpy(s.g_vals, g[0..s.g_vals.len]);
                @memcpy(s.c_vals, c[0..s.c_vals.len]);
            },
        }
    }

    /// See `freq_held.factorEach`.
    pub const factorEach = freq_held.factorEach;

    /// See `freq_held.solveEach`.
    pub const solveEach = freq_held.solveEach;

    /// Private kernels and types for tests.zig; void outside test builds.
    pub const test_access = if (@import("builtin").is_test) .{
        .W = W,
        .setOmega = setOmega,
        .solveRhs = solveRhs,
        .solveRhsT = solveRhsT,
        .solveBatchSerial = freq_batch.solveBatchSerial,
        .setOmegaSparse = freq_fill.setOmegaSparse,
        .indexDyn = freq_fill.indexDyn,
        .Sparse = Sparse,
    } else {};

    // ---- private ----

    const Strategy = union(enum) {
        dense: Dense,
        sp: Sparse,
    };

    fn initSparse(allocator: Allocator, n: u32, ckt: anytype) !Self {
        const nn: u32 = 2 * n;
        const src_nnz: usize = ckt.nnz;
        const total_nnz: usize = 4 * src_nnz;

        const col_ptr = try allocator.alloc(u32, @as(usize, nn) + 1);
        errdefer allocator.free(col_ptr);
        const row_idx = try allocator.alloc(u32, total_nnz);
        errdefer allocator.free(row_idx);
        const vals = try allocator.alloc(f64, total_nnz);
        errdefer allocator.free(vals);
        const g_vals = try allocator.dupe(f64, ckt.g_vals[0..src_nnz]);
        errdefer allocator.free(g_vals);
        const c_vals = try allocator.dupe(f64, ckt.c_vals[0..src_nnz]);
        errdefer allocator.free(c_vals);

        buildStackedRealPattern(n, ckt.col_ptr, ckt.row_idx, col_ptr, row_idx);
        assert(col_ptr[nn] == total_nnz);

        // The analysis Circuit carries the facade's switches; a bare planes
        // view takes the defaults.
        const T = @TypeOf(ckt);
        const C = if (@typeInfo(T) == .pointer) @typeInfo(T).pointer.child else T;
        const knobs: direct.Knobs = if (comptime @hasField(C, "knobs")) ckt.knobs.lu else .{};
        var slv = try direct.Solver.init(allocator, nn, col_ptr, row_idx, null, knobs);
        errdefer slv.deinit();
        // A frequency solve is used as is, with no Newton step to correct
        // it, so a replayed pivot (the lanes', or the scalar refactor's
        // from the last ω) must pass the full factor's own threshold; at
        // the Newton default of 1e-12 a tape laid at ω = 0 replayed at
        // 1e6 left scaled residuals of 8e3 (#44).
        slv.params.refactor_growth_limit = direct.pivot_tol;
        // Each ω's values are new; the bypass copy would be nnz2 f64s
        // (3.7 MB on sweep_opamp_wl_5000) that never match.
        slv.dropBypass();

        return .{
            .gpa = allocator,
            .n = n,
            .nn = nn,
            .strategy = .{ .sp = .{
                .g_vals = g_vals,
                .c_vals = c_vals,
                .src_col_ptr = ckt.col_ptr,
                .src_row_idx = ckt.row_idx,
                .col_ptr = col_ptr,
                .row_idx = row_idx,
                .vals = vals,
                .slv = slv,
            } },
        };
    }

    // Assembles and factors G + jωC. Right-hand sides at this ω then
    // need only `solveRhs`/`solveRhsT`. Fails with `error.Singular`
    // (dense) or `error.SingularMatrix` (sparse) on a singular A(ω), after
    // which `solveRhs` must not run until a `setOmega` succeeds.
    fn setOmega(self: *Self, omega: f64) !void {
        try self.setOmegaDyn(omega, .{}, 0);
    }
};

// Dense only wins for tiny systems. medium/ladder_filter (n = 41) spent 90%
// of its run in dense 82^3/3 factors per omega; the sparse refactor is O(nnz).
const dense_threshold: u32 = 16;

/// G = 2 on the diagonal and -1 between ring neighbours (0 and n - 1
/// included), C = `c_diag` on the diagonal: a pattern no structured engine
/// takes, so the stacked 2n system gets the sparse LU.
const Ring = struct {
    const n_max = 24;
    col_ptr: [n_max + 1]u32 = undefined,
    row_idx: [3 * n_max]u32 = undefined,
    g: [3 * n_max]f64 = undefined,
    c: [3 * n_max]f64 = undefined,

    fn init(n: u32, c_diag: f64) Ring {
        assert(n >= 3);
        assert(n <= n_max);
        var r: Ring = .{};
        var p: u32 = 0;
        for (0..n) |j| {
            r.col_ptr[j] = p;
            var rows = [3]u32{ @intCast((j + n - 1) % n), @intCast(j), @intCast((j + 1) % n) };
            std.mem.sort(u32, &rows, {}, std.sort.asc(u32));
            for (rows) |i| {
                r.row_idx[p] = i;
                r.g[p] = if (i == j) 2 else -1;
                r.c[p] = if (i == j) c_diag else 0;
                p += 1;
            }
        }
        r.col_ptr[n] = p;
        return r;
    }
};

/// The column of CSC slot `slot` in the pattern `col_ptr` describes, by
/// binary search (O(log n)); empty columns are skipped. A slot at or past
/// nnz returns n, one past the last column, so callers filter ground slots
/// first. Asserts that `col_ptr` opens at 0.
pub fn slotCol(col_ptr: []const u32, slot: u32) usize {
    assert(col_ptr[0] == 0);
    const j = std.sort.upperBound(u32, col_ptr, slot, struct {
        fn order(key: u32, item: u32) std.math.Order {
            return std.math.order(key, item);
        }
    }.order) - 1;
    assert(j < col_ptr.len);
    return j;
}

/// Source slot `slot`'s four stacked-real positions: column j's G and ωC
/// entries, then column n + j's -ωC and G entries.
pub fn stackedPos(src_col_ptr: []const u32, slot: u32) [4]usize {
    assert(slot < src_col_ptr[src_col_ptr.len - 1]);
    const j = slotCol(src_col_ptr, slot);
    const cs: usize = src_col_ptr[j];
    const len: usize = src_col_ptr[j + 1] - cs;
    const top = 2 * cs + (slot - cs);
    const right = 2 * @as(usize, src_col_ptr[src_col_ptr.len - 1]) + top;
    return .{ top, top + len, right, right + len };
}

/// Adds re + j·im at entry (row, col) of the row-major 2n x 2n stacked-real
/// `a`, [Re -Im; Im Re], the layout `buildComplexAdmittance` writes.
/// Asserts that nn = 2n and that (row, col) lies in the n x n block.
pub fn addDense(n: usize, nn: usize, a: []f64, row: usize, col: usize, re: f64, im: f64) void {
    assert(nn == 2 * n);
    assert(row < n);
    assert(col < n);
    a[row * nn + col] += re;
    a[(n + row) * nn + col] += im;
    a[row * nn + n + col] -= im;
    a[(n + row) * nn + n + col] += re;
}

// ---- private ----

/// Writes the 2n x 2n stacked-real CSC pattern: each column j and n + j gets
/// column j's rows r followed by r + n. `sr_col_ptr` has 2n + 1 entries,
/// `sr_row_idx` 4 * nnz.
inline fn buildStackedRealPattern(
    n: u32,
    col_ptr: []const u32,
    row_idx: []const u32,
    sr_col_ptr: []u32,
    sr_row_idx: []u32,
) void {
    const nu: usize = n;
    assert(sr_col_ptr.len == 2 * nu + 1);
    assert(sr_row_idx.len == 4 * @as(usize, col_ptr[n]));
    var p: u32 = 0;
    sr_col_ptr[0] = 0;

    for (0..2) |half| {
        for (0..nu) |j| {
            const s = col_ptr[j];
            const e = col_ptr[j + 1];
            @memcpy(sr_row_idx[p..][0 .. e - s], row_idx[s..e]);
            p += e - s;
            for (row_idx[s..e]) |r| {
                sr_row_idx[p] = r + n;
                p += 1;
            }
            sr_col_ptr[half * nu + j + 1] = p;
        }
    }
    assert(p == sr_row_idx.len);
}

fn everyEntryPoint(gpa: Allocator, n: u32) !void {
    const ring: Ring = .init(n, 1e-3);
    const nnz = ring.col_ptr[n];
    var fs = try FreqSolver.fromPlanes(gpa, n, ring.col_ptr[0 .. n + 1], ring.row_idx[0..nnz], ring.g[0..nnz], ring.c[0..nnz]);
    defer fs.deinit();
    const nn: usize = 2 * @as(usize, n);
    var rhs: [2 * Ring.n_max]f64 = @splat(0);
    rhs[0] = 1;
    var x: [2 * Ring.n_max * 5]f64 = undefined;
    try fs.solve(1e3, rhs[0..nn], x[0..nn]);
    const omegas = [_]f64{ 1, 1e2, 1e4, 1e6, 1e8 };
    const dyn: Dyn = .{ .slots = &.{ 0, 1, 1_000_000 }, .re = &(@as([15]f64, @splat(0.5))), .im = &(@as([15]f64, @splat(0.25))) };
    try fs.solveBatch(&omegas, dyn, rhs[0..nn], x[0 .. omegas.len * nn], false);
    try fs.solveBatch(&omegas, .{}, rhs[0..nn], x[0 .. omegas.len * nn], true);
    try fs.factorEach(&omegas);
    var each: [2 * Ring.n_max * 5]f64 = @splat(1);
    fs.solveEach(each[0 .. omegas.len * nn], x[0 .. omegas.len * nn], false);
}

test "FreqSolver: every entry point frees all it holds on every allocation failure (#83)" {
    // n = 4 takes the dense strategy, n = 20 the sparse lanes. Any error but
    // OutOfMemory would be an allocation failure reported as a numerical one.
    inline for (.{ 4, 20 }) |n|
        try testing.checkAllAllocationFailures(@import("core").testing.fixed_growth, everyEntryPoint, .{@as(u32, n)});
}

test "FreqSolver: a dense factorEach that fails to grow keeps its held factors consistent" {
    // Grows from one held ω to three; the pivot rows' allocation fails. The
    // next factorEach of the same count must still allocate both.
    const ring: Ring = .init(4, 1e-3);
    const nnz = ring.col_ptr[4];
    for (0..2) |fail_at| {
        var failing = testing.FailingAllocator.init(testing.allocator, .{});
        var fs = try FreqSolver.fromPlanes(failing.allocator(), 4, ring.col_ptr[0..5], ring.row_idx[0..nnz], ring.g[0..nnz], ring.c[0..nnz]);
        defer fs.deinit();
        try testing.expect(fs.strategy == .dense);
        try fs.factorEach(&.{1});
        failing.fail_index = failing.alloc_index + fail_at;
        const omegas = [_]f64{ 1, 10, 100 };
        try testing.expectError(error.OutOfMemory, fs.factorEach(&omegas));
        failing.fail_index = std.math.maxInt(usize);
        try fs.factorEach(&omegas);
        var rhs: [3 * 8]f64 = @splat(0);
        for (0..3) |k| rhs[k * 8] = 1;
        var x: [3 * 8]f64 = undefined;
        fs.solveEach(&rhs, &x, false);
        // Each held factor solves its own ω: compare with a fresh solve.
        for (omegas, 0..) |w, k| {
            var want: [8]f64 = undefined;
            try fs.solve(w, rhs[k * 8 ..][0..8], &want);
            for (want, x[k * 8 ..][0..8]) |a, b| try testing.expectApproxEqAbs(a, b, 1e-12);
        }
    }
}

test "complex held factors solve each ω as a fresh solve does, both ways, and fall back when an ω cannot replay" {
    const gpa = testing.allocator;
    const n: u32 = 20;
    const nn = 2 * n;
    var ring: Ring = .init(n, 1e-3);
    // Diagonally dominant, so ω = 0 factors too.
    for (0..n) |j| for (ring.col_ptr[j]..ring.col_ptr[j + 1]) |p| {
        if (ring.row_idx[p] == j) ring.g[p] = 3;
    };
    const nz = ring.col_ptr[n];
    var fs = try FreqSolver.fromPlanes(gpa, n, ring.col_ptr[0 .. n + 1], ring.row_idx[0..nz], ring.g[0..nz], ring.c[0..nz]);
    defer fs.deinit();
    fs.complex_held = true;
    const omegas = [_]f64{ 0, 1, 3, 10, 30, 100, 300, 1000, 3000 };
    try fs.factorEach(&omegas);
    try testing.expect(fs.strategy.sp.cx_held.len != 0 and fs.strategy.sp.held.len == 0);
    var rhs: [omegas.len * nn]f64 = undefined;
    for (&rhs, 0..) |*v, i| v.* = @as(f64, @floatFromInt(i % 7)) - 3;
    var got: [rhs.len]f64 = undefined;
    for ([_]bool{ false, true }) |adjoint| {
        fs.solveEach(&rhs, &got, adjoint);
        var want: [rhs.len]f64 = undefined;
        for (omegas, 0..) |_, k| {
            try FreqSolver.test_access.solveBatchSerial(&fs, &omegas, .{}, k, 1, rhs[k * nn ..][0..nn], want[0 .. omegas.len * nn], adjoint);
            for (want[k * nn ..][0..nn], got[k * nn ..][0..nn]) |a, b| try testing.expectApproxEqAbs(a, b, 1e-12);
        }
    }
    // A singular ω (column 0 charge-only at ω = 0) cannot replay the tape:
    // the stacked policy takes over and holds it as the identity.
    var bad = ring;
    for (0..n) |j| for (bad.col_ptr[j]..bad.col_ptr[j + 1]) |p| {
        if (j == 0) {
            bad.g[p] = 0;
            if (bad.row_idx[p] == 0) bad.c[p] = 0.02;
        }
    };
    fs.setPlanes(bad.g[0..nz], bad.c[0..nz]);
    try fs.factorEach(&omegas);
    try testing.expect(fs.strategy.sp.cx_held.len == 0);
    try testing.expectEqualSlices(u32, &.{0}, fs.held_id.items);
}

test "slotCol and stackedPos place every slot of a pattern with empty columns" {
    // Columns 1 and 3 are empty: col_ptr = 0 2 2 5 5 6.
    const col_ptr = [_]u32{ 0, 2, 2, 5, 5, 6 };
    const want_col = [_]usize{ 0, 0, 2, 2, 2, 4 };
    for (want_col, 0..) |j, slot| try testing.expectEqual(j, slotCol(&col_ptr, @intCast(slot)));
    // A slot at nnz (a ground entry) maps one past the last column.
    try testing.expectEqual(@as(usize, 5), slotCol(&col_ptr, 6));
    // Stacked column j holds column j's G rows then its ωC rows; column
    // n + j starts at 2 nnz with -ωC then G.
    for (0..6) |slot| {
        const j = want_col[slot];
        const cs = col_ptr[j];
        const len = col_ptr[j + 1] - cs;
        const q = stackedPos(&col_ptr, @intCast(slot));
        try testing.expectEqual(2 * cs + (slot - cs), q[0]);
        try testing.expectEqual(q[0] + len, q[1]);
        try testing.expectEqual(12 + q[0], q[2]);
        try testing.expectEqual(q[2] + len, q[3]);
    }
}

test "addDense writes the stacked-real [Re -Im; Im Re] image of one complex entry" {
    var a: [16]f64 = @splat(0);
    addDense(2, 4, &a, 1, 0, 3, 5);
    // Row-major 4 x 4: (1,0) += 3, (3,0) += 5, (1,2) -= 5, (3,2) += 3.
    const want = [16]f64{ 0, 0, 0, 0, 3, 0, -5, 0, 0, 0, 0, 0, 5, 0, 3, 0 };
    try testing.expectEqualSlices(f64, &want, &a);
}

test "StackedPoints: lane (point, ω) is bitwise Stacked(1) of its own point on the shared tape" {
    const gpa = testing.allocator;
    const n: u32 = 20; // above dense_threshold: the sparse strategy
    const a: Ring = .init(n, 1e-3);
    var b: Ring = .init(n, 3e-3);
    for (b.g[0..a.col_ptr[n]]) |*v| v.* *= 1.05;
    const nnz = a.col_ptr[n];
    var fa = try FreqSolver.fromPlanes(gpa, n, a.col_ptr[0 .. n + 1], a.row_idx[0..nnz], a.g[0..nnz], a.c[0..nnz]);
    defer fa.deinit();
    var fb = try FreqSolver.fromPlanes(gpa, n, b.col_ptr[0 .. n + 1], b.row_idx[0..nnz], b.g[0..nnz], b.c[0..nnz]);
    defer fb.deinit();
    // Lay point a's tape at 1e4 rad/s; point b replays it.
    try FreqSolver.test_access.setOmega(&fa, 1e4);
    const spa = &fa.strategy.sp;
    const spb = &fb.strategy.sp;
    const lu = spa.slv.sparse().?;
    const L = 4;
    var lanes = try lane_lu.LaneLu(L).init(gpa, lu);
    defer lanes.deinit(gpa);
    const omegas = [L]f64{ 1e2, 1e2, 1e6, 1e6 };
    const src: StackedPoints(L) = .{
        .src_col_ptr = spa.src_col_ptr,
        .col_ptr = spa.col_ptr,
        .g = .{ spa.g_vals, spb.g_vals, spa.g_vals, spb.g_vals },
        .c = .{ spa.c_vals, spb.c_vals, spa.c_vals, spb.c_vals },
        .omega = omegas,
    };
    const bad = lanes.refactor(src, spa.slv.params.refactor_growth_limit);
    const nn: usize = 2 * n;
    const bv = try gpa.alloc(@Vector(L, f64), 2 * nn);
    defer gpa.free(bv);
    for (bv[0..nn], 0..) |*v, i| v.* = @splat(@as(f64, @floatFromInt(i % 3)) - 1);
    lanes.solve(bv[0..nn], bv[nn..]);
    var one = try lane_lu.LaneLu(1).init(gpa, lu);
    defer one.deinit(gpa);
    const b1 = try gpa.alloc(@Vector(1, f64), 2 * nn);
    defer gpa.free(b1);
    for (0..L) |l| {
        const sp = if (l % 2 == 0) spa else spb;
        const want_bad = one.refactor(Stacked(1).of(sp, @as(@Vector(1, f64), @splat(omegas[l])), .{}, 0, 1), spa.slv.params.refactor_growth_limit);
        try testing.expectEqual(want_bad != 0, (bad >> @intCast(l)) & 1 != 0);
        if (want_bad != 0) continue;
        for (b1[0..nn], 0..) |*v, i| v.* = @splat(@as(f64, @floatFromInt(i % 3)) - 1);
        one.solve(b1[0..nn], b1[nn..]);
        for (bv[nn..], b1[nn..]) |got, want| {
            const g: [L]f64 = got;
            const w: [1]f64 = want;
            try testing.expectEqual(@as(u64, @bitCast(w[0])), @as(u64, @bitCast(g[l])));
        }
    }
}

test "provider chunks keep the scalar frequency history across tails and adjoint multiple right-hand sides" {
    const gpa = testing.allocator;
    const W = FreqSolver.W;
    const n: u32 = 20;
    const nn = 2 * n;
    const ring: Ring = .init(n, 1e-3);
    const nz = ring.col_ptr[n];
    const omegas = [_]f64{ 1, 3, 1e2, 1e3, 1e4, 3e4, 1e5, 1e6, 1e9 };
    var rhs: [2 * nn]f64 = @splat(0);
    rhs[0] = 1;
    rhs[nn + 3] = -2;
    rhs[nn + n + 4] = 0.25;
    for ([_]bool{ false, true }) |adjoint| {
        var solo = try FreqSolver.fromPlanes(gpa, n, ring.col_ptr[0 .. n + 1], ring.row_idx[0..nz], ring.g[0..nz], ring.c[0..nz]);
        defer solo.deinit();
        // The provider replays the stacked tape; so must its oracle.
        solo.complex_lanes = false;
        var provider = try FreqSolver.fromPlanes(gpa, n, ring.col_ptr[0 .. n + 1], ring.row_idx[0..nz], ring.g[0..nz], ring.c[0..nz]);
        defer provider.deinit();
        var want: [omegas.len * rhs.len]f64 = undefined;
        var got: [omegas.len * rhs.len]f64 = undefined;
        try solo.solveBatch(&omegas, .{}, &rhs, &want, adjoint);
        var base: usize = 0;
        while (base < omegas.len) : (base += W) {
            const count = @min(W, omegas.len - base);
            const chunk = (try provider.beginLaneChunk(&omegas, base, count, &rhs, &got, adjoint)) orelse continue;
            var ll = try lane_lu.LaneLu(W).init(gpa, @constCast(chunk.lu));
            defer ll.deinit(gpa);
            var ow: [W]f64 = undefined;
            for (&ow, 0..) |*o, l| o.* = omegas[base + @min(l, count - 1)];
            const bad = ll.refactor(StackedPoints(W){
                .src_col_ptr = chunk.src_col_ptr,
                .col_ptr = chunk.col_ptr,
                .g = @splat(chunk.g),
                .c = @splat(chunk.c),
                .omega = ow,
            }, chunk.growth);
            var b: [nn]@Vector(W, f64) = undefined;
            var x: [nn]@Vector(W, f64) = undefined;
            var r: usize = 0;
            while (r < rhs.len) : (r += nn) {
                for (&b, rhs[r..][0..nn]) |*v, scalar| v.* = @splat(scalar);
                if (adjoint) ll.solveT(&b, &x) else ll.solve(&b, &x);
                for (0..count) |l| {
                    if (bad & (@as(u64, 1) << @intCast(l)) != 0) continue;
                    for (x, 0..) |v, i| {
                        const row: [W]f64 = v;
                        got[(base + l) * rhs.len + r + i] = row[l];
                    }
                }
            }
            try provider.finishLaneChunk(&omegas, base, count, bad, &rhs, &got, adjoint);
        }
        try testing.expectEqualSlices(u64, @ptrCast(&want), @ptrCast(&got));
    }
}

test "packed held factors preserve ragged harmonics, transpose solves and singular identity blocks" {
    const gpa = testing.allocator;
    const lk = @import("kernels/lane.zig");
    const B = @import("kernels/frequency.zig").Bodies(lk.HostP);
    const n: u32 = 20;
    const nn = 2 * n;
    var ring: Ring = .init(n, 0);
    const nz = ring.col_ptr[n];
    // Column zero has only charge. Its zero-frequency block is singular,
    // while nonzero frequencies retain the same sparse pivot tape.
    for (0..n) |j| for (ring.col_ptr[j]..ring.col_ptr[j + 1]) |p| {
        if (ring.row_idx[p] == j) ring.g[p] = 3;
        if (j == 0) {
            ring.g[p] = 0;
            if (ring.row_idx[p] == 0) ring.c[p] = 0.02;
        }
    };
    var fs = try FreqSolver.fromPlanes(gpa, n, ring.col_ptr[0 .. n + 1], ring.row_idx[0..nz], ring.g[0..nz], ring.c[0..nz]);
    defer fs.deinit();
    const omegas = [_]f64{ -1, 0, 2, 0, 3, -3, 10, 0, 30 };
    const N = omegas.len;
    try fs.factorEach(&omegas);
    try testing.expectEqualSlices(u32, &.{ 1, 3, 7 }, fs.held_id.items);
    const tape = fs.heldTape().?;
    const built = try lk.buildTab(gpa, tape.lu, tape.col_ptr, tape.growth, direct.pivot_tol);
    defer gpa.free(built.idx);
    const lower = try gpa.alloc(f64, tape.lu.lx.items.len * N);
    defer gpa.free(lower);
    const upper = try gpa.alloc(f64, tape.lu.ux.items.len * N);
    defer gpa.free(upper);
    var diag: [nn * N]f64 = undefined;
    fs.packHeld(N, lower, upper, &diag);
    for (lower) |v| try testing.expect(std.math.isFinite(v));
    for (upper) |v| try testing.expect(std.math.isFinite(v));
    for (diag) |v| try testing.expect(std.math.isFinite(v));
    var rhs: [nn * N]f64 = undefined;
    var rhs_plane: [nn * N]f64 = undefined;
    for (0..N) |k| for (0..nn) |i| {
        const value = if (k == 6) 0 else @as(f64, @floatFromInt((k + 1) * (i + 1))) / 11;
        rhs[k * nn + i] = value;
        rhs_plane[i * N + k] = value;
    };
    var want: [nn * N]f64 = undefined;
    var got: [nn * N]f64 = undefined;
    var work: [nn * N]f64 = undefined;
    for ([_]bool{ false, true }) |adjoint| {
        fs.solveEach(&rhs, &want, adjoint);
        for (0..N) |k| {
            if (std.mem.indexOfScalar(u32, fs.held_id.items, @intCast(k)) != null) {
                for (0..nn) |i| got[i * N + k] = rhs[k * nn + i];
            } else {
                B.solve(built.tab, built.idx.ptr, lower.ptr, upper.ptr, &diag, &rhs_plane, &work, &got, N, k, adjoint);
            }
        }
        for (0..N) |k| for (0..nn) |i|
            try testing.expectEqual(@as(u64, @bitCast(want[k * nn + i])), @as(u64, @bitCast(got[i * N + k])));
    }
}

test "resident held recovery replays the selected tape without another pivot generation" {
    const gpa = testing.allocator;
    const n: u32 = 20;
    const nn = 2 * n;
    var ring: Ring = .init(n, 1e-3);
    for (0..n) |j| for (ring.col_ptr[j]..ring.col_ptr[j + 1]) |p| {
        if (ring.row_idx[p] == j) ring.g[p] = 3;
    };
    const nz = ring.col_ptr[n];
    const omegas = [_]f64{ 1, 3, 9 };
    var oracle = try FreqSolver.fromPlanes(gpa, n, ring.col_ptr[0 .. n + 1], ring.row_idx[0..nz], ring.g[0..nz], ring.c[0..nz]);
    defer oracle.deinit();
    var resumed = try FreqSolver.fromPlanes(gpa, n, ring.col_ptr[0 .. n + 1], ring.row_idx[0..nz], ring.g[0..nz], ring.c[0..nz]);
    defer resumed.deinit();
    try oracle.factorEach(&omegas);
    try testing.expect((try resumed.beginHeldTape(&omegas, omegas.len / 2)) != null);
    const generation = resumed.strategy.sp.slv.gen;
    try testing.expectEqual(@as(usize, 0), try resumed.replayCurrentHeld(&omegas));
    try testing.expectEqual(generation, resumed.strategy.sp.slv.gen);
    // An earlier point in a failed device batch may have finished its
    // policy already. Materializing its selected tape must also be free
    // of additional full factors.
    resumed.completeHeldPolicy(omegas.len, false);
    try resumed.ensureHeldOnHost(&omegas);
    try testing.expectEqual(generation, resumed.strategy.sp.slv.gen);
    try testing.expectEqual(oracle.strategy.sp.slv.gen, generation);
    var rhs: [omegas.len * nn]f64 = undefined;
    for (&rhs, 0..) |*v, i| v.* = @as(f64, @floatFromInt(i % 11)) - 5;
    var want: [rhs.len]f64 = undefined;
    var got: [rhs.len]f64 = undefined;
    for ([_]bool{ false, true }) |adjoint| {
        oracle.solveEach(&rhs, &want, adjoint);
        resumed.solveEach(&rhs, &got, adjoint);
        try testing.expectEqualSlices(u64, @ptrCast(&want), @ptrCast(&got));
    }
    // New planes invalidate readiness and resume the ordinary policy.
    resumed.setPlanes(ring.g[0..nz], ring.c[0..nz]);
    try resumed.ensureHeldOnHost(&omegas);
    try testing.expectEqual(generation + 1, resumed.strategy.sp.slv.gen);
}

test "failed held recovery cannot silently restart the candidate policy" {
    const n: u32 = 20;
    var ring: Ring = .init(n, 1e-3);
    for (0..n) |j| for (ring.col_ptr[j]..ring.col_ptr[j + 1]) |p| {
        if (ring.row_idx[p] == j) ring.g[p] = 3;
    };
    const nz = ring.col_ptr[n];
    const omegas = [_]f64{ 1, 3, 9 };
    var failing = testing.FailingAllocator.init(testing.allocator, .{});
    var fs = try FreqSolver.fromPlanes(failing.allocator(), n, ring.col_ptr[0 .. n + 1], ring.row_idx[0..nz], ring.g[0..nz], ring.c[0..nz]);
    defer fs.deinit();
    try testing.expect((try fs.beginHeldTape(&omegas, 1)) != null);
    const generation = fs.strategy.sp.slv.gen;
    failing.fail_index = failing.alloc_index;
    try testing.expectError(error.OutOfMemory, fs.replayCurrentHeld(&omegas));
    failing.fail_index = std.math.maxInt(usize);
    try testing.expectError(error.OutOfMemory, fs.ensureHeldOnHost(&omegas));
    try testing.expectEqual(generation, fs.strategy.sp.slv.gen);

    // Fail inside the host candidate's first full factor. This precedes
    // any device launch and must still preserve the original point's
    // allocation failure without a later successful retry.
    var candidate_alloc = testing.FailingAllocator.init(testing.allocator, .{});
    var candidate = try FreqSolver.fromPlanes(candidate_alloc.allocator(), n, ring.col_ptr[0 .. n + 1], ring.row_idx[0..nz], ring.g[0..nz], ring.c[0..nz]);
    defer candidate.deinit();
    try candidate.held_id.ensureTotalCapacity(candidate_alloc.allocator(), omegas.len);
    candidate_alloc.fail_index = candidate_alloc.alloc_index;
    try testing.expectError(error.OutOfMemory, candidate.beginHeldTape(&omegas, 1));
    const failed_generation = candidate.strategy.sp.slv.gen;
    try testing.expect(!candidate.strategy.sp.slv.factored);
    candidate_alloc.fail_index = std.math.maxInt(usize);
    try testing.expectError(error.OutOfMemory, candidate.ensureHeldOnHost(&omegas));
    try testing.expectEqual(failed_generation, candidate.strategy.sp.slv.gen);
}
