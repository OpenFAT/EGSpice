//! Thomas solver for tridiagonal patterns (RC ladders, chains): a CSC
//! pattern and its values in, solutions of A x = b and A^T x = b out, O(n).
//!
//! No pivoting: a pivot that threshold partial pivoting would not keep fails
//! the factor, and `direct` demotes the system to the pivoting sparse LU.
//! Cites: Thomas (1949) via Golub and Van Loan, Matrix Computations §4.3;
//! the demotion rule #45.
//!
//! Contents:
//! - `pub fn isTridiag(n: u32, col_ptr: []const u32, row_idx: []const u32) bool`
//! - `pub const TriDiag = struct`
//! - `pub const FactorError = error{SingularMatrix}` (as `TriDiag.FactorError`)
//! - `pub fn init(gpa: Allocator, n: u32, col_ptr: []const u32, row_idx: []const u32) Allocator.Error!TriDiag`
//! - `pub fn deinit(self: *TriDiag, gpa: Allocator) void`
//! - `pub fn factor(self: *TriDiag, vals: []const f64, pivot_tol: f64) FactorError!void`
//! - `pub fn solve(self: *TriDiag, x: []f64) void`
//! - `pub fn solveT(self: *TriDiag, x: []f64) void`

const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;

/// Factors of a tridiagonal matrix whose CSC slots were resolved once at
/// `init`; `factor`, `solve` and `solveT` allocate nothing.
pub const TriDiag = struct {
    n: u32,
    /// The CSC slots of row i's three band entries, read together by
    /// `factor`, so one row is one 12-byte record.
    rows: []Row,
    /// Modified diagonal `bp`, elimination multiplier `mul` (`mul[0]`
    /// unused) and super-diagonal `cv` (`cv[n-1]` unused), as three
    /// n-long planes in that order. Valid only after a successful `factor`.
    planes: []f64,

    /// `SingularMatrix`: a modified pivot is zero, non-finite or too small
    /// against the entry below it. Thomas does not pivot, so this covers
    /// nonsingular matrices too. The factors are unusable after it.
    pub const FactorError = error{SingularMatrix};

    /// Resolves the band's CSC slots, O(nnz). Borrows nothing.
    /// Caller owns the result; free it with `deinit` and the same `gpa`.
    /// Asserts that the pattern passes `isTridiag`, which needs n >= 3.
    pub fn init(gpa: Allocator, n: u32, col_ptr: []const u32, row_idx: []const u32) Allocator.Error!TriDiag {
        assert(isTridiag(n, col_ptr, row_idx));
        const rows = try gpa.alloc(Row, n);
        errdefer gpa.free(rows);

        const planes = try gpa.alloc(f64, 3 * @as(usize, n));

        @memset(rows, .{ .sub = .none, .diag = .none, .sup = .none });
        for (0..n) |j| {
            for (col_ptr[j]..col_ptr[j + 1]) |p| {
                const r = row_idx[p];
                const slot: Slot = @fromBackingInt(@as(u32, @intCast(p)));
                if (r == j) {
                    rows[j].diag = slot;
                } else if (r == j + 1) {
                    rows[j + 1].sub = slot;
                } else {
                    assert(r + 1 == j); // isTridiag leaves only the super-diagonal.
                    rows[r].sup = slot;
                }
            }
        }
        assert(rows[0].sub == .none);
        assert(rows[n - 1].sup == .none);
        return .{ .n = n, .rows = rows, .planes = planes };
    }

    /// Frees the slot table and the factors.
    pub fn deinit(self: *TriDiag, gpa: Allocator) void {
        assert(self.rows.len == self.n);
        assert(self.planes.len == 3 * @as(usize, self.n));
        gpa.free(self.rows);
        gpa.free(self.planes);
        self.* = undefined;
    }

    /// Runs the Thomas forward sweep over `vals`, O(n). Fails when a
    /// modified pivot is zero, non-finite, or below `pivot_tol` times the
    /// entry under it: the test threshold partial pivoting keeps the
    /// diagonal by, so the sparse LU `direct` demotes to pivots where Thomas
    /// cannot (#45). Asserts that `pivot_tol` is not negative and `vals`
    /// covers every slot.
    pub fn factor(self: *TriDiag, vals: []const f64, pivot_tol: f64) FactorError!void {
        const n = self.n;
        assert(n >= 3);
        assert(pivot_tol >= 0);
        const bp, const mul, const cv = self.split();
        bp[0] = value(self.rows[0].diag, vals);
        cv[0] = value(self.rows[0].sup, vals);
        for (self.rows, 0..) |row, i| {
            if (i > 0) {
                const m = value(row.sub, vals) / bp[i - 1];
                mul[i] = m;
                bp[i] = value(row.diag, vals) - m * cv[i - 1];
                if (i + 1 < n) cv[i] = value(row.sup, vals);
            }
            const d = @abs(bp[i]);
            const below = if (i + 1 < n) @abs(value(self.rows[i + 1].sub, vals)) else 0;
            // Negated, so a NaN pivot fails too.
            if (!(d > 0 and d < std.math.inf(f64) and d >= pivot_tol * below)) return error.SingularMatrix;
        }
    }

    /// Replaces `x` with A^-1 x after a successful `factor`, O(n).
    /// Asserts that `x` has n entries.
    pub fn solve(self: *TriDiag, x: []f64) void {
        const n = self.n;
        assert(n >= 3);
        assert(x.len == n);
        const bp, const mul, const cv = self.split();
        for (1..n) |i| x[i] -= mul[i] * x[i - 1];
        x[n - 1] /= bp[n - 1];
        var i: usize = n - 1;
        while (i > 0) {
            i -= 1;
            x[i] = (x[i] - cv[i] * x[i + 1]) / bp[i];
        }
    }

    /// Replaces `x` with A^-T x after a successful `factor`, O(n): forward
    /// through U^T (diagonal bp, sub-diagonal cv), then back through unit
    /// L^T. Asserts that `x` has n entries.
    pub fn solveT(self: *TriDiag, x: []f64) void {
        const n = self.n;
        assert(n >= 3);
        assert(x.len == n);
        const bp, const mul, const cv = self.split();
        x[0] /= bp[0];
        for (1..n) |i| x[i] = (x[i] - cv[i - 1] * x[i - 1]) / bp[i];
        var i: usize = n - 1;
        while (i > 0) {
            i -= 1;
            x[i] -= mul[i + 1] * x[i + 1];
        }
    }

    // ---- private ----

    fn split(self: *TriDiag) struct { []f64, []f64, []f64 } {
        const n = self.n;
        assert(self.planes.len == 3 * @as(usize, n));
        return .{ self.planes[0..n], self.planes[n .. 2 * n], self.planes[2 * n ..] };
    }
};

// A CSC position, or `none` for an entry the pattern lacks (read as 0).
const Slot = enum(u32) { none = std.math.maxInt(u32), _ };

const Row = struct { sub: Slot, diag: Slot, sup: Slot };

comptime {
    assert(@sizeOf(Row) == 12);
}

/// Returns true when every entry has |row - col| <= 1 and n >= 3, O(nnz).
/// Asserts that `col_ptr` has n + 1 entries.
pub fn isTridiag(n: u32, col_ptr: []const u32, row_idx: []const u32) bool {
    assert(col_ptr.len == @as(usize, n) + 1);
    if (n < 3) return false;
    assert(col_ptr[n] <= row_idx.len);
    for (0..n) |j| {
        for (col_ptr[j]..col_ptr[j + 1]) |p| {
            const r = row_idx[p];
            const diff = if (r >= j) r - j else j - r;
            if (diff > 1) return false;
        }
    }
    return true;
}

// ---- private ----

inline fn value(slot: Slot, vals: []const f64) f64 {
    return if (slot != .none) vals[@backingInt(slot)] else 0;
}

test "TriDiag: random diagonally dominant ladders solve both ways; NaN, Inf and absent entries" {
    const testing = std.testing;
    const gpa = testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x7d1a);
    const r = prng.random();
    var col_ptr: [65]u32 = undefined;
    var row_idx: [3 * 64]u32 = undefined;
    var vals: [3 * 64]f64 = undefined;
    var dense: [64][64]f64 = undefined;
    for (3..65) |n| {
        // Band entries, each off-diagonal absent one time in five.
        for (0..n) |i| @memset(dense[i][0..n], 0);
        for (0..n) |i| {
            for ([_]usize{ i -| 1, i + 1 }) |j| {
                if (j == i or j >= n or r.uintLessThan(u8, 5) == 0) continue;
                dense[i][j] = 2 * r.float(f64) - 1;
            }
            dense[i][i] = 3 + r.float(f64);
        }
        var p: u32 = 0;
        col_ptr[0] = 0;
        for (0..n) |j| {
            for (j -| 1..@min(j + 2, n)) |i| {
                if (i != j and dense[i][j] == 0) continue;
                row_idx[p] = @intCast(i);
                vals[p] = dense[i][j];
                p += 1;
            }
            col_ptr[j + 1] = p;
        }
        const nn: u32 = @intCast(n);
        try testing.expect(isTridiag(nn, col_ptr[0 .. n + 1], row_idx[0..p]));
        var td = try TriDiag.init(gpa, nn, col_ptr[0 .. n + 1], row_idx[0..p]);
        defer td.deinit(gpa);
        try td.factor(vals[0..p], 1e-3);
        var x: [64]f64 = undefined;
        var b: [64]f64 = undefined;
        for (b[0..n]) |*v| v.* = 2 * r.float(f64) - 1;
        inline for (.{ false, true }) |transpose| {
            @memcpy(x[0..n], b[0..n]);
            if (transpose) td.solveT(x[0..n]) else td.solve(x[0..n]);
            for (0..n) |i| {
                var ri = b[i];
                for (0..n) |j| ri -= (if (transpose) dense[j][i] else dense[i][j]) * x[j];
                try testing.expect(@abs(ri) < 1e-13);
            }
        }
        // A non-finite entry anywhere in the band poisons a pivot.
        const bad = r.uintLessThan(u32, p);
        const keep = vals[bad];
        vals[bad] = if (r.boolean()) std.math.nan(f64) else std.math.inf(f64);
        try testing.expectError(error.SingularMatrix, td.factor(vals[0..p], 1e-3));
        vals[bad] = keep;
    }
}

test "TriDiag: the pivot test passes at exactly pivot_tol times the entry below and fails one ulp under" {
    const testing = std.testing;
    const gpa = testing.allocator;
    // [b0 0 0; 1 1 0; 0 0.5 1], CSC. Step 0's pivot b0 meets 1e-3 * |1|.
    const col_ptr = [_]u32{ 0, 2, 4, 5 };
    const row_idx = [_]u32{ 0, 1, 1, 2, 2 };
    var vals = [_]f64{ 1e-3, 1, 1, 0.5, 1 };
    var td = try TriDiag.init(gpa, 3, &col_ptr, &row_idx);
    defer td.deinit(gpa);
    try td.factor(&vals, 1e-3);
    vals[0] = std.math.nextAfter(f64, 1e-3, 0);
    try testing.expectError(error.SingularMatrix, td.factor(&vals, 1e-3));
    // A zero tolerance still refuses a zero pivot.
    vals[0] = 0;
    try testing.expectError(error.SingularMatrix, td.factor(&vals, 0));
    vals[0] = -0.0;
    try testing.expectError(error.SingularMatrix, td.factor(&vals, 0));
}

test "TriDiag: LAPACK scaled residual on wide-range ladders, and init under every allocation failure" {
    const testing = std.testing;
    const gpa = testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x7d1b);
    const r = prng.random();
    const n = 40;
    var col_ptr: [n + 1]u32 = undefined;
    var row_idx: [3 * n]u32 = undefined;
    var vals: [3 * n]f64 = undefined;
    var p: u32 = 0;
    col_ptr[0] = 0;
    for (0..n) |j| {
        for (j -| 1..@min(j + 2, n)) |i| {
            row_idx[p] = @intCast(i);
            p += 1;
        }
        col_ptr[j + 1] = p;
    }
    try testing.checkAllAllocationFailures(@import("core").testing.fixed_growth, struct {
        fn run(a: Allocator, cp: []const u32, ri: []const u32) !void {
            var t = try TriDiag.init(a, n, cp, ri);
            t.deinit(a);
        }
    }.run, .{ col_ptr[0..], row_idx[0..p] });
    var td = try TriDiag.init(gpa, n, &col_ptr, row_idx[0..p]);
    defer td.deinit(gpa);
    for (0..50) |_| {
        // Column j scaled by 10^e, e in [-12, 12]: an RC ladder with
        // conductances across decades. Column dominance keeps every pivot
        // above its column, so Thomas factors without pivoting.
        for (0..n) |j| {
            const s = std.math.pow(f64, 10, @floatFromInt(r.intRangeAtMost(i32, -12, 12)));
            var off: f64 = 0;
            var diag: u32 = 0;
            for (col_ptr[j]..col_ptr[j + 1]) |q| {
                if (row_idx[q] == j) {
                    diag = @intCast(q);
                } else {
                    vals[q] = s * (2 * r.float(f64) - 1);
                    off += @abs(vals[q]);
                }
            }
            vals[diag] = off + s * (0.1 + r.float(f64));
        }
        try td.factor(vals[0..p], 1e-3);
        var b: [n]f64 = undefined;
        for (&b) |*v| v.* = 2 * r.float(f64) - 1;
        inline for (.{ false, true }) |transpose| {
            var x = b;
            if (transpose) td.solveT(&x) else td.solve(&x);
            // LAPACK's ||A x - b|| / (||A|| ||x|| n eps) in max norms, with
            // n max|a_ij| bounding ||A||: under 30 for a backward-stable
            // solve, ~1/(n eps) for a wrong one.
            var rmax: f64 = 0;
            var amax: f64 = 0;
            var xmax: f64 = 0;
            var res: [n]f64 = b;
            for (0..n) |j| for (col_ptr[j]..col_ptr[j + 1]) |q| {
                const i = row_idx[q];
                if (transpose) res[j] -= vals[q] * x[i] else res[i] -= vals[q] * x[j];
                amax = @max(amax, @abs(vals[q]));
            };
            for (res, x) |ri, xi| {
                rmax = @max(rmax, @abs(ri));
                xmax = @max(xmax, @abs(xi));
            }
            try testing.expect(rmax / (n * amax * xmax * n * std.math.floatEps(f64)) <= 30);
        }
    }
}
