//! Newton's per-circuit scratch: a CSC pattern in, the direct solver on it
//! and the step vectors out, plus the W-lane workspace that borrows W of
//! them and the `--timing-in-depth` phase clock. Cites nothing.
//!
//! Contents:
//! - `pub const Prof = struct`
//!   - `pub const Phase = enum`
//!   - `pub fn start(p: *Prof) void`
//!   - `pub fn lap(p: *Prof, phase: Phase) void`
//! - `pub const Workspace = struct`
//!   - `pub fn init(gpa: std.mem.Allocator, n: u32, col_ptr: []const u32, row_idx: []const u32, bbd: ?BbdInfo, knobs: Knobs, lu_knobs: direct.Knobs) direct.InitError!Workspace`
//!   - `pub fn initOrdered(gpa: std.mem.Allocator, n: u32, col_ptr: []const u32, row_idx: []const u32, bbd: ?BbdInfo, knobs: Knobs, lu_knobs: direct.Knobs, order: ?[]const u32) direct.InitError!Workspace`
//!   - `pub fn reset(self: *Workspace, bbd: ?BbdInfo) std.mem.Allocator.Error!void`
//!   - `pub fn ensureAVals(self: *Workspace, nnz: u32) std.mem.Allocator.Error![]f64`
//!   - `pub fn deinit(self: *Workspace, gpa: std.mem.Allocator) void`
//!   - `pub fn ensureKrylov(self: *Workspace) std.mem.Allocator.Error!*Gmres`
//! - `pub fn LaneWorkspace(comptime W: usize) type`
//!     - `pub fn init(gpa: std.mem.Allocator, ws: [W]*Workspace) !Self`
//!     - `pub fn deinit(self: *Self, gpa: std.mem.Allocator) void`

const std = @import("std");
const assert = std.debug.assert;
const num = @import("core").numerics;
const BbdInfo = num.BbdInfo;
const direct = @import("direct.zig");
const Gmres = @import("gmres.zig").Gmres;
const converger = @import("converger.zig");
const Knobs = converger.Knobs;
const Lanes = @import("lane/lanes.zig").Lanes;
const LaneSolver = @import("lane/solver.zig").LaneSolver;

/// `--timing-in-depth` wall-time split of `newton`, summed over every solve
/// on one workspace. Off (no clock reads) until `io` is set.
pub const Prof = struct {
    io: ?std.Io = null,
    /// Nanoseconds per phase, indexed by `Phase`.
    ns: [5]u64 = @splat(0),
    /// Newton iterations and the factorizations among them.
    counts: [2]u64 = @splat(0),
    last: std.Io.Timestamp = .zero,

    /// eval: device evaluation plus companion stamps (`hook.assemble`).
    /// load: the combined G + ag0*C matrix and gmin. update: the step,
    /// limiting and the convergence gates.
    pub const Phase = enum(u3) { eval, load, factor, solve, update };

    /// Opens one Newton iteration: stamps the clock and counts it. A no-op
    /// while `io` is null.
    pub fn start(p: *Prof) void {
        const io = p.io orelse return;
        p.last = .now(io, .awake);
        p.counts[0] += 1;
    }

    /// Charges the time since the last `start` or `lap` to `phase`; a
    /// `.factor` lap also counts a factorization. A no-op while `io` is null.
    pub fn lap(p: *Prof, phase: Phase) void {
        comptime assert(@typeInfo(Phase).@"enum".field_names.len == @typeInfo(@FieldType(Prof, "ns")).array.len);
        const io = p.io orelse return;
        const now: std.Io.Timestamp = .now(io, .awake);
        p.ns[@backingInt(phase)] += @intCast(p.last.durationTo(now).nanoseconds);
        p.last = now;
        if (phase == .factor) p.counts[1] += 1;
    }
};

/// Per-circuit Newton scratch: the direct solver on the circuit pattern plus
/// step vectors. GMRES and combined-matrix buffers grow on first use.
pub const Workspace = struct {
    slv: direct.Solver,
    dx: []f64,
    x_old: []f64,
    /// `jfnk`'s GMRES and its four n-vectors (x_pert, f0, b, diag), made
    /// on its first call.
    krylov: ?Gmres = null,
    jf: []f64 = &.{},
    /// tran's G + alpha*C values, grown by `ensureAVals`.
    a_vals: []f64 = &.{},
    prof: Prof = .{},
    knobs: Knobs,

    /// Borrows `col_ptr` and `row_idx` for the workspace's lifetime. Caller
    /// frees with `deinit` and the same allocator. Fails as
    /// `direct.Solver.init` does (out of memory, or the ordering's
    /// `OutOfWorkspace`), leaving nothing allocated. Asserts that `col_ptr`
    /// has n + 1 entries.
    pub fn init(gpa: std.mem.Allocator, n: u32, col_ptr: []const u32, row_idx: []const u32, bbd: ?BbdInfo, knobs: Knobs, lu_knobs: direct.Knobs) direct.InitError!Workspace {
        return initOrdered(gpa, n, col_ptr, row_idx, bbd, knobs, lu_knobs, null);
    }

    /// `init` with a column order to reuse (`direct.Solver.initOrdered`).
    pub fn initOrdered(gpa: std.mem.Allocator, n: u32, col_ptr: []const u32, row_idx: []const u32, bbd: ?BbdInfo, knobs: Knobs, lu_knobs: direct.Knobs, order: ?[]const u32) direct.InitError!Workspace {
        assert(col_ptr.len == @as(usize, n) + 1);
        const dx = try gpa.alloc(f64, n);
        errdefer gpa.free(dx);
        const x_old = try gpa.alloc(f64, n);
        errdefer gpa.free(x_old);
        return .{
            .slv = try direct.Solver.initOrdered(gpa, n, col_ptr, row_idx, bbd, lu_knobs, order),
            .dx = dx,
            .x_old = x_old,
            .knobs = knobs,
        };
    }

    /// Returns the workspace to its state right after `initOrdered`,
    /// keeping its allocations (`direct.Solver.reset`; `bbd` must be what
    /// `init` got). The GMRES and combined-value buffers are scratch every
    /// call rewrites, so they stay.
    pub fn reset(self: *Workspace, bbd: ?BbdInfo) std.mem.Allocator.Error!void {
        try self.slv.reset(bbd);
        self.prof = .{};
    }

    /// Scratch of length `nnz` for tran's combined G + alpha*C values, reused
    /// across runs; the slice is invalidated by the next call that grows it.
    pub fn ensureAVals(self: *Workspace, nnz: u32) std.mem.Allocator.Error![]f64 {
        if (self.a_vals.len < nnz) {
            self.slv.gpa.free(self.a_vals);
            // Empty before the alloc so a failure cannot double-free in deinit.
            self.a_vals = &.{};
            self.a_vals = try self.slv.gpa.alloc(f64, nnz);
        }
        return self.a_vals[0..nnz];
    }

    /// Frees every buffer, the lazily grown ones included; `gpa` must be the
    /// allocator `init` got.
    pub fn deinit(self: *Workspace, gpa: std.mem.Allocator) void {
        self.slv.deinit();
        gpa.free(self.dx);
        gpa.free(self.x_old);
        if (self.krylov) |*k| k.deinit(gpa);
        gpa.free(self.jf);
        gpa.free(self.a_vals);
        self.* = undefined;
    }

    /// Returns `jfnk`'s GMRES workspace, built with its four n-vectors on
    /// the first call.
    pub fn ensureKrylov(self: *Workspace) std.mem.Allocator.Error!*Gmres {
        if (self.krylov == null) {
            const gpa = self.slv.gpa;
            const n = self.slv.n;
            const jf = try gpa.alloc(f64, 4 * @as(usize, n));
            errdefer gpa.free(jf);
            self.krylov = try Gmres.init(gpa, n, @min(gmres_restart, n));
            self.jf = jf;
        }
        return &self.krylov.?;
    }
};

const gmres_restart = 30;

/// Newton scratch for W lanes: lane l's own `Workspace` (its solver,
/// borrowed into `slv`), plus lane-major step vectors. Lane l's scalar
/// workspace stays usable between lane solves: peeling a lane to W = 1 runs
/// `newton` on it directly.
pub fn LaneWorkspace(comptime W: usize) type {
    return struct {
        const Self = @This();

        /// Lane l's scalar workspace, borrowed; must outlive self.
        lanes: [W]*Workspace,
        slv: LaneSolver(W),
        dx: []L.V,
        x_old: []L.V,
        prof: Prof = .{},
        knobs: Knobs,

        /// Fails only on allocation. Asserts that the lanes share n.
        pub fn init(gpa: std.mem.Allocator, ws: [W]*Workspace) !Self {
            const n = ws[0].slv.n;
            for (ws) |w| assert(w.slv.n == n);
            var solvers: [W]*direct.Solver = undefined;
            for (&solvers, ws) |*s, w| s.* = &w.slv;
            var slv = try LaneSolver(W).init(gpa, solvers);
            errdefer slv.deinit();
            const dx = try gpa.alloc(L.V, n);
            errdefer gpa.free(dx);
            return .{ .lanes = ws, .slv = slv, .dx = dx, .x_old = try gpa.alloc(L.V, n), .knobs = ws[0].knobs };
        }

        /// Frees what `init` allocated; the lanes' workspaces stay theirs.
        pub fn deinit(self: *Self, gpa: std.mem.Allocator) void {
            self.slv.deinit();
            gpa.free(self.dx);
            gpa.free(self.x_old);
            self.* = undefined;
        }

        // ---- private ----

        const L = Lanes(W);
    };
}
