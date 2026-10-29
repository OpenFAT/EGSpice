//! Pole-zero analysis (`.pz`): the finite roots of det(G + sC) = 0 for two
//! pencils built from one linearization at the operating point.
//!
//! Poles use the circuit's own MNA. A linearization nulls the independent
//! sources, so the input voltage source is already the short a `vol`
//! denominator wants and a current source the open a `cur` one wants; the
//! mode word changes nothing for poles.
//!
//! Zeros use the same pencil with the output column replaced by the input
//! drive. H(s) = e_outᵀ Y⁻¹ d, so by Cramer's rule that determinant is H's
//! numerator and its roots are the transfer zeros. It equals the bordered
//! determinant [[Y, d], [e_outᵀ, 0]], but as a column swap the C plane stays
//! emptier: a grounded output capacitor drops out of the pencil instead of
//! surviving as a spurious root.
//!
//! Poles are s = 1/λ for the eigenvalues λ of A = −G⁻¹C, found by dense
//! Hessenberg reduction and Francis double-shift QR (core/eigen.zig). Zeros are the
//! finite generalized eigenvalues of the numerator pencil itself, by QZ
//! (qz.zig). That pencil is singular at s = 0 whenever the transfer has a
//! zero there, and its roots at infinity form a defective block. Through A
//! that block comes back as roundoff eigenvalues scattered as far from zero
//! as genuine ones; the QZ sees each of its roots as a vanishing diagonal of
//! C's triangular factor instead.
//!
//! Above `dense_max_n` unknowns the dense pencil is out of reach (O(n³)
//! time, 4n² f64), and `.pz` reports only the `sparse_k` roots of each kind
//! nearest a shift, by shift-invert Arnoldi on the sparse LU (`sparseRoots`);
//! the plot name says so.
//!
//! In: a circuit linearized at its operating point. Out: the poles and
//! zeros as one complex plot.
//! Cites ngspice cktpzstr.c (where the iteration limit diverges) and
//! dev/analysis/pole-zero.md.
//!
//! Contents:
//! - `pub const Options = core.query.Pz`
//! - `pub const Roots = struct { poles, zeros, allocator, requested, shift }`
//! - `pub fn run(ctx: *const RunCtx, opts: Options) !Result`

const std = @import("std");
const assert = std.debug.assert;
const root = @import("../types.zig");
const dense_lu = @import("solver").dense_lu;
const freq_solve = @import("solver").freq_solve;
const qr = @import("core").eigen;
const qz = @import("qz.zig");
const pz_sparse = @import("pz_sparse.zig");
const sparseRoots = pz_sparse.sparseRoots;
const Pencil = pz_sparse.Pencil;
const numerics = @import("core").numerics;
const Complex = numerics.Complex;
const GROUND = root.GROUND;

/// Query options, defined in core/query.zig.
pub const Options = @import("core").query.Pz;

/// Poles and zeros in rad/s, each owned by `allocator`.
pub const Roots = struct {
    poles: []Complex,
    zeros: []Complex,
    allocator: std.mem.Allocator,
    /// 0 when the lists hold every finite root (the dense path); else how
    /// many of each kind the sparse path wanted, nearest `shift` first.
    requested: u32 = 0,
    /// rad/s, the sparse path's shift.
    shift: f64 = 0,

    /// Frees both root lists.
    fn deinit(self: *Roots) void {
        self.allocator.free(self.poles);
        self.allocator.free(self.zeros);
    }
};

/// Unknowns above which `.pz` takes the sparse path. The dense one costs
/// 0.26 s on a 500-stage RC ladder and 2.1 s on a 1000-stage one
/// (ReleaseFast), growing as n³.
const dense_max_n = 1000;

/// One dense pencil solve's working set. `g` and `c` are the pencil; the
/// zeros pass rewrites both in place as the numerator.
const Work = struct {
    n: usize,
    g: []f64,
    c: []f64,
    /// G, overwritten with its LU.
    m: []f64,
    /// A = −G⁻¹C, destroyed by the QR.
    a: []f64,
    /// Column of C and its solve, n each.
    col: []f64,
    sol: []f64,
    piv: []u32,
    eigs: []Complex,
};

/// Test fixture (`sparse pz` test): a bridged RC ladder of `stages` nodes, a gyrator every 40th node (an
/// inductor out of the next node's capacitor, so complex poles), loaded at
/// the far end and driven by a voltage source: row-major dense G and C plus
/// the same pencil on a CSC pattern. Unknowns: GROUND, the nodes 1..stages,
/// the source node, its branch row. `ac_coupled` feeds node 1 through a
/// series capacitor, a transfer zero at the origin.
const TestLadder = struct {
    n: usize,
    g: []f64,
    c: []f64,
    col_ptr: []u32,
    row_idx: []u32,
    gv: []f64,
    cv: []f64,

    fn init(gpa: std.mem.Allocator, stages: usize, ac_coupled: bool) !TestLadder {
        const n = stages + 3;
        const src = stages + 1;
        const br = stages + 2;
        const g = try gpa.alloc(f64, n * n);
        const c = try gpa.alloc(f64, n * n);
        @memset(g, 0);
        @memset(c, 0);
        const two = struct {
            fn add(m: []f64, nn: usize, a: usize, b: usize, v: f64) void {
                m[a * nn + a] += v;
                m[b * nn + b] += v;
                m[a * nn + b] -= v;
                m[b * nn + a] -= v;
            }
        };
        g[0] = 1;
        g[src * n + br] = 1;
        g[br * n + src] = 1;
        if (ac_coupled) two.add(c, n, src, 1, 2) else two.add(g, n, src, 1, 1);
        g[stages * n + stages] += 0.5;
        for (1..stages + 1) |k| {
            const kf: f64 = @floatFromInt(k);
            c[k * n + k] += 1 + @mod(kf * 37, 11) / 10;
            if (k < stages) two.add(g, n, k, k + 1, 1 + @mod(kf * 13, 7) / 5);
            if (k % 7 == 0 and k + 3 <= stages) two.add(c, n, k, k + 3, 0.05 + @mod(kf * 29, 13) / 50);
            if (k % 40 == 0 and k < stages) {
                g[k * n + k + 1] += 0.7;
                g[(k + 1) * n + k] -= 0.7;
            }
        }
        var nnz: usize = 0;
        for (0..n) |j| for (0..n) |i| {
            if (g[i * n + j] != 0 or c[i * n + j] != 0) nnz += 1;
        };
        const col_ptr = try gpa.alloc(u32, n + 1);
        const row_idx = try gpa.alloc(u32, nnz);
        const gv = try gpa.alloc(f64, nnz + 1);
        const cv = try gpa.alloc(f64, nnz + 1);
        var k: usize = 0;
        for (0..n) |j| {
            col_ptr[j] = @intCast(k);
            for (0..n) |i| if (g[i * n + j] != 0 or c[i * n + j] != 0) {
                row_idx[k] = @intCast(i);
                gv[k] = g[i * n + j];
                cv[k] = c[i * n + j];
                k += 1;
            };
        }
        col_ptr[n] = @intCast(k);
        gv[nnz] = 0;
        cv[nnz] = 0;
        return .{ .n = n, .g = g, .c = c, .col_ptr = col_ptr, .row_idx = row_idx, .gv = gv, .cv = cv };
    }

    fn deinit(self: TestLadder, gpa: std.mem.Allocator) void {
        gpa.free(self.g);
        gpa.free(self.c);
        gpa.free(self.col_ptr);
        gpa.free(self.row_idx);
        gpa.free(self.gv);
        gpa.free(self.cv);
    }
};

/// Contract entry: the roots at ctx.x_op, complex, in one of two layouts.
/// Poles only: a list, one row per pole under (index, pole), which is what
/// decks written against bare `.pz` read. With zeros the two counts need not
/// match, so the layout is ngspice's: one row of `pole(k)` and `zero(k)`
/// columns.
pub fn run(ctx: *const root.RunCtx, opts: Options) !root.Result {
    const a = ctx.allocator;
    var res = try solve(ctx.circuit, ctx.x_op, opts, ctx.scratch_allocator);

    defer res.deinit();
    const plotname = try plotName(a, res, opts.want);

    if (opts.want == .poles) {
        const n = res.poles.len;
        const names = try a.dupe([]const u8, &.{ "index", "pole" });
        const data = try a.alloc(f64, n * 4);
        for (res.poles, 0..) |pole, i| {
            data[i * 4] = @floatFromInt(i);
            data[i * 4 + 1] = 0;
            data[i * 4 + 2] = pole.re;
            data[i * 4 + 3] = pole.im;
        }
        return .{ .plotname = plotname, .varnames = names, .is_complex = true, .npoints = n, .data = data };
    }

    const nvars = 1 + res.poles.len + res.zeros.len;
    const names = try a.alloc([]const u8, nvars);
    const data = try a.alloc(f64, nvars * 2);
    names[0] = "index";
    data[0] = 0;
    data[1] = 0;
    var col: usize = 1;
    for ([_][]const u8{ "pole", "zero" }, [_][]const Complex{ res.poles, res.zeros }) |label, roots| {
        for (roots, 1..) |r, k| {
            names[col] = try std.fmt.allocPrint(a, "{s}({d})", .{ label, k });
            data[col * 2] = r.re;
            data[col * 2 + 1] = r.im;
            col += 1;
        }
    }
    return .{ .plotname = plotname, .varnames = names, .is_complex = true, .npoints = 1, .data = data };
}

// ---- private ----

/// Poles and, when the directive asks for them, transfer zeros at the
/// operating point `x_op`. Returns error.Singular when poles are wanted and G
/// itself is singular (a node reached only through capacitors), and
/// error.PzDidNotConverge when the QR or QZ runs out of `qr_max_iter` with
/// roots still missing: the dense path never returns a partial root set.
/// Above `dense_max_n` unknowns the sparse path returns the roots nearest
/// its shift and says how many it wanted (`Roots.requested`).
///
/// Diverges from ngspice, which warns at its iteration limit and publishes
/// the roots it found (cktpzstr.c:225).
fn solve(
    ckt: *root.Circuit,
    x_op: []const f64,
    options: Options,
    allocator: std.mem.Allocator,
) !Roots {
    const n: usize = ckt.n;

    try ckt.linearizeAc(x_op);

    // The frequency-dependent entries at their DC gain, acDyn(0).re (absdelay
    // 1, a laplace H(0)): what a .dc eval stamps in G, bit for bit. e^(-sτ)
    // has no finite pole-zero form, so a delay stays transparent.
    const dyn = try allocator.alloc(f64, 2 * ckt.ac_dyn_slots.len);
    defer allocator.free(dyn);
    const dyn_re = dyn[0..ckt.ac_dyn_slots.len];
    if (ckt.ac_dyn_slots.len != 0) ckt.acDyn(x_op, &.{0}, dyn_re, dyn[dyn_re.len..]);

    if (n > dense_max_n) {
        const nnz: usize = ckt.nnz;
        const planes = try allocator.alloc(f64, 2 * (nnz + 1));
        defer allocator.free(planes);
        const g = planes[0 .. nnz + 1];
        const c = planes[nnz + 1 ..];
        @memcpy(g[0..nnz], ckt.g_vals[0..nnz]);
        @memcpy(c[0..nnz], ckt.c_vals[0..nnz]);
        g[nnz] = 0;
        c[nnz] = 0;
        for (ckt.ac_dyn_slots, dyn_re) |slot, v| {
            if (slot < nnz) g[slot] += v;
        }
        return sparseRoots(.{
            .n = ckt.n,
            .col_ptr = ckt.col_ptr,
            .row_idx = ckt.row_idx,
            .g = g,
            .c = c,
            .bbd = ckt.bbd,
            .lu = ckt.knobs.lu,
            .execution = ckt.solver_execution,
        }, options, allocator);
    }

    const planes = try allocator.alloc(f64, 2 * n * n);
    defer allocator.free(planes);
    ckt.denseG(planes[0 .. n * n]);
    ckt.denseC(planes[n * n ..]);
    for (ckt.ac_dyn_slots, dyn_re) |slot, v| {
        if (slot < ckt.nnz) planes[@as(usize, ckt.row_idx[slot]) * n + freq_solve.slotCol(ckt.col_ptr, slot)] += v;
    }
    return denseRoots(n, planes[0 .. n * n], planes[n * n ..], options, allocator);
}

/// Every finite root of the dense row-major pencil (`g`, `c`), both n×n and
/// destroyed. Errors as `solve`.
fn denseRoots(n: usize, g: []f64, c: []f64, options: Options, allocator: std.mem.Allocator) !Roots {
    const arena = try allocator.alloc(f64, 2 * n * n + 2 * n);
    defer allocator.free(arena);
    const piv = try allocator.alloc(u32, n);
    defer allocator.free(piv);
    // At most n eigenvalues: an exact bound.
    const eigs = try allocator.alloc(Complex, n);
    defer allocator.free(eigs);

    var w: Work = .{
        .n = n,
        .g = g,
        .c = c,
        .m = arena[0 .. n * n],
        .a = arena[n * n .. 2 * n * n],
        .col = arena[2 * n * n ..][0..n],
        .sol = arena[2 * n * n + n ..][0..n],
        .piv = piv,
        .eigs = eigs,
    };

    var poles: []Complex = &.{};
    errdefer allocator.free(poles);
    if (options.want != .zeros) poles = try poleRoots(&w, options, allocator);

    var zeros: []Complex = &.{};
    if (options.want != .poles) {
        numeratorPencil(&w, options);
        // det(G' + sC') = det(G' − s(−C')).
        for (w.c) |*v| v.* = -v.*;
        const num = qz.roots(n, w.g, w.c, eigs, options.qr_tol, options.qr_max_iter);
        if (!num.converged) return error.PzDidNotConverge;
        zeros = try allocator.dupe(Complex, eigs[0..num.count]);
    }

    return .{
        .poles = poles,
        .zeros = zeros,
        .allocator = allocator,
    };
}

/// Poles from A = −G⁻¹C, in a list owned by `allocator`.
fn poleRoots(w: *Work, options: Options, allocator: std.mem.Allocator) ![]Complex {
    @memcpy(w.m, w.g);
    try dense_lu.factorize(w.n, w.m, w.piv);
    buildA(w);
    const eigs = w.eigs;
    const den = qr.eigenvalues(w.n, w.a, eigs, options.qr_tol, options.qr_max_iter);
    if (!den.converged) return error.PzDidNotConverge;

    // s = 1/λ = conj(λ)/|λ|². |λ| ≈ 0 is a row with no dynamics (a resistive
    // node, a branch row), not a pole at the origin, and it is zero to within
    // the QR's backward error, O(n·eps·‖A‖). Anything above that is a real
    // time constant however small next to the slowest one, so the cutoff is
    // absolute: `pz/widely_separated_modes` has τ = 1 ns next to τ = 1000 s,
    // twelve decades apart.
    var max_abs: f64 = 0;
    for (eigs[0..den.count]) |l| max_abs = @max(max_abs, l.mag());
    const cutoff = @as(f64, @floatFromInt(w.n)) * std.math.floatEps(f64) * max_abs;
    var kept: usize = 0;
    for (eigs[0..den.count]) |l| {
        const m2 = l.magSq();
        if (l.mag() <= cutoff or m2 == 0) continue;
        eigs[kept] = .{ .re = l.re / m2, .im = -l.im / m2 };
        kept += 1;
    }
    return allocator.dupe(Complex, eigs[0..kept]);
}

/// A = −G⁻¹C into `w.a`, one back-substitution per column of C against the
/// LU in `w.m`.
fn buildA(w: *Work) void {
    const n = w.n;
    for (0..n) |j| {
        for (0..n) |r| w.col[r] = w.c[r * n + j];
        dense_lu.solveFactored(n, w.m, w.piv, w.col, w.sol);
        for (0..n) |r| w.a[r * n + j] = -w.sol[r];
    }
}

/// Rewrites (G, C) in place as the transfer numerator's pencil: the MNA with
/// the output column replaced by the input drive.
///
/// A differential output is one column operation away from a single column:
/// in det [[Y, d], [e_out⁺ᵀ − e_out⁻ᵀ, 0]], adding column out⁺ into column
/// out⁻ cancels the −1 the border row carries there and leaves the +1 at out⁺
/// to expand along, which is the column swap below. GROUND is row 0, the unit
/// clamp `v(0) = 0`, and never takes a stamp.
fn numeratorPencil(w: *Work, o: Options) void {
    const n = w.n;
    if (o.out_neg != GROUND) {
        for (0..n) |r| {
            w.g[r * n + o.out_neg] += w.g[r * n + o.out_pos];
            w.c[r * n + o.out_neg] += w.c[r * n + o.out_pos];
        }
    }
    for (0..n) |r| {
        w.g[r * n + o.out_pos] = 0;
        w.c[r * n + o.out_pos] = 0;
    }
    if (o.drive_branch != GROUND) {
        // `vol`: the input source's branch row is the only row its rhs enters.
        w.g[o.drive_branch * n + o.out_pos] = 1;
    } else {
        // `cur`: a current injected across the input node pair.
        if (o.in_pos != GROUND) w.g[o.in_pos * n + o.out_pos] = 1;
        if (o.in_neg != GROUND) w.g[o.in_neg * n + o.out_pos] = -1;
    }
}

/// "Pole-Zero Analysis", plus, for the sparse path's partial lists, how
/// many roots of each kind it wanted, the shift they are nearest, and how
/// many it found.
fn plotName(a: std.mem.Allocator, res: Roots, want: Options.Want) ![]const u8 {
    const base = "Pole-Zero Analysis";
    if (res.requested == 0) return base;
    const head = "{s} (partial: the {d} roots of each kind nearest s = {d} rad/s; found ";
    return switch (want) {
        .poles => std.fmt.allocPrint(a, head ++ "{d} poles)", .{ base, res.requested, res.shift, res.poles.len }),
        .zeros => std.fmt.allocPrint(a, head ++ "{d} zeros)", .{ base, res.requested, res.shift, res.zeros.len }),
        .both => std.fmt.allocPrint(a, head ++ "{d} poles, {d} zeros)", .{ base, res.requested, res.shift, res.poles.len, res.zeros.len }),
    };
}

/// det of a dense row-major n×n copy, by elimination with partial pivoting.
fn testDet(comptime n: usize, m0: [n * n]f64) f64 {
    var m = m0;
    var det: f64 = 1;
    for (0..n) |c| {
        var p = c;
        for (c + 1..n) |r| if (@abs(m[r * n + c]) > @abs(m[p * n + c])) {
            p = r;
        };
        if (p != c) {
            for (0..n) |j| std.mem.swap(f64, &m[c * n + j], &m[p * n + j]);
            det = -det;
        }
        det *= m[c * n + c];
        if (m[c * n + c] == 0) return 0;
        for (c + 1..n) |r| {
            const f = m[r * n + c] / m[c * n + c];
            for (c..n) |j| m[r * n + j] -= f * m[c * n + j];
        }
    }
    return det;
}

/// Asserts that `part` is the |s|-nearest prefix of `all`: each root within
/// 1e-8 relative of one in `all` (plus 1e-12 absolute, for a root at the
/// origin of these unit-scale circuits), and none in `all` nearer the
/// origin left out.
fn expectNearest(all: []const Complex, part: []const Complex, want: usize) !void {
    const gpa = std.testing.allocator;
    const sorted = try gpa.dupe(Complex, all);
    defer gpa.free(sorted);
    std.mem.sort(Complex, sorted, {}, struct {
        fn before(_: void, l: Complex, r: Complex) bool {
            return l.magSq() < r.magSq();
        }
    }.before);
    try std.testing.expect(part.len >= @min(want, sorted.len));
    try std.testing.expect(part.len <= sorted.len);
    if (part.len == 0) return;
    const big = part[part.len - 1].mag();
    const abs_tol = 1e-12 * @max(big, 1);
    for (part) |r| {
        var best = std.math.inf(f64);
        for (sorted) |d| best = @min(best, d.sub(r).mag());
        try std.testing.expect(best <= 1e-8 * r.mag() + abs_tol);
    }
    try std.testing.expect(big <= sorted[part.len - 1].mag() * (1 + 1e-8) + abs_tol);
}

test "sparse pz: the roots nearest the shift are the dense path's nearest" {
    const gpa = std.testing.allocator;
    for ([_]usize{ 5, 300, 600 }) |stages| for ([_]bool{ false, true }) |ac_coupled| {
        const lad = try TestLadder.init(gpa, stages, ac_coupled);
        defer lad.deinit(gpa);
        const pencil: Pencil = .{
            .n = @intCast(lad.n),
            .col_ptr = lad.col_ptr,
            .row_idx = lad.row_idx,
            .g = lad.gv,
            .c = lad.cv,
        };
        // The output a dozen stages in: the zeros are then the natural
        // frequencies of the ladder beyond it, well conditioned. At the far
        // end of a long ladder they are a cluster no method resolves (the
        // dense QZ's own are a ring of radius eps^(1/r)).
        const transfer: Options = .{ .in_pos = @intCast(stages + 1), .out_pos = @intCast(@min(stages, 12)), .drive_branch = @intCast(stages + 2), .want = .both };
        for ([_]Options{ .{}, transfer }) |o| {
            const g = try gpa.dupe(f64, lad.g);
            defer gpa.free(g);
            const c = try gpa.dupe(f64, lad.c);
            defer gpa.free(c);
            var dense = try denseRoots(lad.n, g, c, o, gpa);
            defer dense.deinit();
            var sparse = try sparseRoots(pencil, o, gpa);
            defer sparse.deinit();
            try std.testing.expectEqual(@as(u32, pz_sparse.sparse_k), sparse.requested);
            try std.testing.expect(sparse.poles.len + sparse.zeros.len > 0);
            if (o.want != .zeros) try expectNearest(dense.poles, sparse.poles, pz_sparse.sparse_k);
            if (o.want != .poles) try expectNearest(dense.zeros, sparse.zeros, pz_sparse.sparse_k);
            // A zero at the origin moves the shift off it.
            try std.testing.expectEqual(ac_coupled and o.want == .both, sparse.shift != 0);
        }
    };
}

test numeratorPencil {
    // Cramer's rule is the oracle: with y = G⁻¹d, the pencil's determinant
    // at s = 0 is (y[out+] − y[out−])·det G, and each y[i] is det G with
    // column i replaced by d. A `cur` drive into row 3 and a `vol` drive on
    // branch row 3 are the same d = e₃.
    const n = 4;
    const g0 = [n * n]f64{ 1, 0, 0, 0, 0.5, 2, -1, 0.25, 0.3, -1, 3, 0.75, 0, 0.75, -0.25, 1.5 };
    const c0 = [n * n]f64{ 0, 0, 0, 0, 0, 1, 0, 0.5, 0, 0, 2, 0, 0, 0.25, 0, 3 };
    var with_d: [2][n * n]f64 = .{ g0, g0 };
    for (with_d[0..], [_]usize{ 2, 1 }) |*m, col| {
        for (0..n) |r| m[r * n + col] = if (r == 3) 1 else 0;
    }
    const want = testDet(n, with_d[0]) - testDet(n, with_d[1]);
    for ([_]Options{
        .{ .out_pos = 2, .out_neg = 1, .in_pos = 3, .want = .zeros },
        .{ .out_pos = 2, .out_neg = 1, .drive_branch = 3, .want = .zeros },
    }) |o| {
        var g = g0;
        var c = c0;
        var w: Work = undefined;
        w.n = n;
        w.g = &g;
        w.c = &c;
        numeratorPencil(&w, o);
        try std.testing.expectApproxEqAbs(want, testDet(n, g), 1e-13);
        try std.testing.expectApproxEqAbs(@as(f64, -0.25), want, 1e-13);
        // C: column out− absorbed column out+, which then holds nothing.
        for (0..n) |r| {
            try std.testing.expectEqual(c0[r * n + 1] + c0[r * n + 2], c[r * n + 1]);
            try std.testing.expectEqual(@as(f64, 0), c[r * n + 2]);
        }
    }
}
