//! The sparse `.pz` path above `pz.dense_max_n` unknowns: the `sparse_k`
//! poles and zeros nearest a shift, by shift-invert Arnoldi on one sparse LU
//! of G + sigma C. The zeros run the numerator pencil of
//! `pz.numeratorPencil` through the same factor, by bordering (`Op`).
//! In: the linearized pencil on the circuit's CSC pattern. Out: the roots
//! nearest the shift, sorted by |s|.
//! Cites dev/analysis/pole-zero.md.
//!
//! Contents:
//! - `pub const sparse_k`
//! - `pub const Pencil = struct`
//! - `Op` (private) has `pub fn apply`, the `arnoldi.dominant` operator hook
//! - `pub fn sparseRoots(p: Pencil, options: Options, gpa: Allocator) !pz.Roots`

const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;
const direct = @import("solver").direct;
const numerics = @import("core").numerics;
const Complex = numerics.Complex;
const root = @import("../types.zig");
const combinePlanes = @import("../Circuit.zig").combinePlanes;
const arnoldi = @import("arnoldi.zig");
const pz = @import("pz.zig");
const Options = pz.Options;
const GROUND = root.GROUND;

/// Roots of each kind the sparse path wants: the dominant ones, which an
/// engineer reads off a large circuit, and few enough that the Krylov basis
/// (2k + 20 columns of n) stays small.
pub const sparse_k = 20;

/// A pencil G + sC on the circuit's CSC pattern, for the sparse path. `g`
/// and `c` hold nnz + 1 values, the trash slot last.
pub const Pencil = struct {
    n: u32,
    col_ptr: []const u32,
    row_idx: []const u32,
    g: []const f64,
    c: []const f64,
    bbd: ?numerics.BbdInfo = null,
    lu: direct.Knobs = .{},
    execution: numerics.Execution = .{},
};

/// x ↦ −M⁻¹C'x at σ, for the poles' pencil (M = G + σC, C' = C) or, with
/// `numerator`, the zeros' one of `numeratorPencil`: G' = G·E + d·e_pᵀ and
/// C' = C·E, where E copies x with x_p replaced by x_{out−} (0 for a
/// grounded out−). M x = b then needs only the factor of G + σC: with
/// u = (G + σC)⁻¹b and w = (G + σC)⁻¹d, Ex = u − x_p·w, so
/// x_p = (u_p − u_{out−}) / h for h = w_p − w_{out−} = H(σ), and
/// x_j = u_j − x_p·w_j elsewhere.
///
/// The numerator pencil's infinite roots (the transfer's relative degree of
/// them) form one defective block of high index, which Arnoldi sees as a
/// ring of roundoff Ritz values; `arnoldi.dominant` drops those for the
/// zeros (`stable_only`). The poles' infinite roots are of index 2 at most
/// (MNA), a ring of radius eps^(1/2) at θ = 0 under the cutoff, so their
/// genuine defective clusters (a cascade of identical stages) come back as
/// the dense QR gives them.
const Op = struct {
    p: *const Pencil,
    slv: *direct.Solver,
    /// Scratch, n each.
    t: []f64,
    ex: []f64,
    numerator: ?struct { pos: u32, neg: u32, w: []const f64, h: f64 } = null,

    pub fn apply(self: *const Op, x: []const f64, y: []f64) void {
        var src = x;
        if (self.numerator) |z| {
            @memcpy(self.ex, x);
            self.ex[z.pos] = if (z.neg != GROUND) x[z.neg] else 0;
            src = self.ex;
        }
        // t = C'x, by columns.
        @memset(self.t, 0);
        for (0..self.p.n) |j| {
            if (src[j] == 0) continue;
            for (self.p.col_ptr[j]..self.p.col_ptr[j + 1]) |k| self.t[self.p.row_idx[k]] += self.p.c[k] * src[j];
        }
        self.slv.solve(self.t, y);
        if (self.numerator) |z| {
            const xp = (y[z.pos] - if (z.neg != GROUND) y[z.neg] else 0) / z.h;
            numerics.axpy(y, -xp, z.w);
            y[z.pos] = xp;
        }
        for (y) |*v| v.* = -v.*;
    }
};

/// The `sparse_k` poles and zeros nearest the shift σ, each list sorted by
/// |s|. Shift-invert: with G + σC regular, θ = 1/(s − σ) runs over the
/// eigenvalues of −(G + σC)⁻¹C, and its largest are the poles nearest σ;
/// the zeros solve the numerator pencil through the same factor (`Op`).
/// σ = 0 unless the zeros need another: G singular with only zeros wanted,
/// or a transfer zero at the origin, where `Op`'s bordering would divide by
/// H(σ) = 0. error.Singular as `solve` (G singular with poles wanted), or
/// when the transfer vanishes at the fallback shift too.
pub fn sparseRoots(p: Pencil, options: Options, allocator: Allocator) !pz.Roots {
    assert(p.g.len == p.col_ptr[p.n] + 1);
    assert(p.c.len == p.g.len);
    const n: usize = p.n;
    const nnz: usize = p.col_ptr[n];
    var slv = try direct.Solver.init(allocator, p.n, p.col_ptr, p.row_idx, p.bbd, p.lu);
    defer slv.deinit();
    const work = try allocator.alloc(f64, nnz + 1 + 4 * n);
    defer allocator.free(work);
    const y = work[0 .. nnz + 1];
    const d = work[nnz + 1 ..][0..n];
    const w = work[nnz + 1 + n ..][0..n];
    const t = work[nnz + 1 + 2 * n ..][0..n];
    const ex = work[nnz + 1 + 3 * n ..][0..n];

    // The drive column of `numeratorPencil`.
    @memset(d, 0);
    if (options.drive_branch != GROUND) {
        d[options.drive_branch] = 1;
    } else {
        if (options.in_pos != GROUND) d[options.in_pos] = 1;
        if (options.in_neg != GROUND) d[options.in_neg] = -1;
    }

    var shift: f64 = 0;
    var h: f64 = 0;
    for (0..2) |attempt| {
        combinePlanes(y, p.g, p.c, shift);
        slv.factor(y, p.execution) catch |err| switch (err) {
            error.SingularMatrix => {
                if (options.want != .zeros or attempt == 1) return error.Singular;
                shift = fallbackShift(p);
                continue;
            },
            else => return err,
        };
        if (options.want == .poles) break;
        // w = (G + σC)⁻¹d, and h = H(σ).
        slv.solve(d, w);
        h = w[options.out_pos] - if (options.out_neg != GROUND) w[options.out_neg] else 0;
        if (h != 0 and std.math.isFinite(h)) break;
        if (attempt == 1) return error.Singular;
        shift = fallbackShift(p);
    }

    var op: Op = .{ .p = &p, .slv = &slv, .t = t, .ex = ex };
    var poles: []Complex = &.{};
    errdefer allocator.free(poles);
    if (options.want != .zeros) poles = try nearest(allocator, n, &op, shift);
    var zeros: []Complex = &.{};
    if (options.want != .poles) {
        op.numerator = .{ .pos = options.out_pos, .neg = options.out_neg, .w = w, .h = h };
        zeros = try nearest(allocator, n, &op, shift);
    }
    return .{ .poles = poles, .zeros = zeros, .allocator = allocator, .requested = sparse_k, .shift = shift };
}

// ---- private ----

/// A shift off the origin when the pencil is singular there: 1e-6 of the
/// slowest rate |g/c| over the slots that hold both, so it sits well below
/// every root but one at the origin.
/// ponytail: a guess from single entries; a root closer to the origin than
/// it can rank out of order against ones near σ. Estimate the nearest pole
/// by a few power steps if that ever matters.
fn fallbackShift(p: Pencil) f64 {
    var rate = std.math.inf(f64);
    for (p.g, p.c) |g, c| {
        if (g != 0 and c != 0) rate = @min(rate, @abs(g / c));
    }
    return if (std.math.isFinite(rate)) 1e-6 * rate else 1;
}

/// The roots s = σ + 1/θ of `op`'s `sparse_k` largest θ, sorted by |s|, in a
/// list owned by `allocator`. Fewer when Arnoldi ran out of restarts or the
/// pencil has fewer finite roots.
fn nearest(allocator: Allocator, n: usize, op: *const Op, shift: f64) ![]Complex {
    var theta: [sparse_k + 1]Complex = undefined;
    const e = try arnoldi.dominant(allocator, n, sparse_k, op, &theta, op.numerator != null);
    const roots = try allocator.alloc(Complex, e.count);
    for (roots, theta[0..e.count]) |*r, l| {
        const m2 = l.magSq();
        r.* = .{ .re = shift + l.re / m2, .im = -l.im / m2 };
    }
    std.mem.sort(Complex, roots, {}, struct {
        fn before(_: void, l: Complex, r: Complex) bool {
            if (l.magSq() != r.magSq()) return l.magSq() < r.magSq();
            return l.im > r.im;
        }
    }.before);
    return roots;
}
