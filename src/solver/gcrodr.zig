//! GCRO-DR(m, k): restarted GMRES that keeps k harmonic Ritz vectors, the
//! slow directions a restart throws away, and carries them across restarts
//! and across solves of a slowly changing sequence (HB Newton steps, PAC
//! input frequencies, shooting iterations). Right preconditioned like
//! `Gmres`, with the same operator contract and result.
//!
//! The iteration runs on Â = A M^-1 and accumulates a correction d, so
//! x = x0 + M^-1 d. With U the recycled directions and C = Â U orthonormal,
//! each cycle's Arnoldi basis V is kept orthogonal to C (B = C^T Â V), so
//! the small least-squares problem over [U, V] splits: the V coefficients
//! are GMRES's own on H̄ and the U ones are -B times them. After a cycle
//! that restarts, the k harmonic Ritz vectors of the whole space replace U;
//! a solve that converges in one cycle neither pays for nor changes them.
//!
//! In: an operator `op.matvec` (and `op.precond`), b, a start x. Out: x with
//! ||b - A x|| <= tol ||b||, or the best iterate and its true residual. All
//! workspace is allocated by `init`. Cites Parks, de Sturler, Mackey,
//! Johnson and Maiti, "Recycling Krylov subspaces for sequences of linear
//! systems", SIAM J. Sci. Comput. 28(5), 2006 (Algorithm 2), and
//! dev/solvers/monodromy-krylov.md.
//!
//! Contents:
//! - `pub const Gcrodr = struct`
//!   - `pub fn init(gpa: Allocator, n: u32, m: u32, k: u32) Allocator.Error!Self`
//!   - `pub fn deinit(self: *Self, gpa: Allocator) void`
//!   - `pub fn forget(self: *Self) void`
//!   - `pub fn solve(self: *Self, op: anytype, b: []const f64, x: []f64, tol: f64, max_restarts: u32) SolveResult`
//! - `const TestOp = struct` (test fixture, private)
//!   - `pub fn matvec(self: *TestOp, v: []const f64, w: []f64) void`
//!   - `pub fn precond(self: *TestOp, r: []f64) void`

const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;
const core = @import("core");
const num = core.numerics;
const dense_lu = @import("dense_lu.zig");
const SolveResult = @import("gmres.zig").Gmres.SolveResult;
const testing = std.testing;

/// GCRO-DR(m, k) workspace for n-dimensional systems. The recycled space
/// survives `solve` calls; `forget` drops it.
pub const Gcrodr = struct {
    const Self = @This();

    n: u32,
    /// Basis size per cycle, recycled vectors included.
    m: u32,
    /// Recycled vectors kept.
    k: u32,
    /// Valid recycled vectors in `u`/`c`, at most k.
    have: u32 = 0,
    slab: []f64,
    /// Arnoldi basis, m + 1 vectors of n.
    v: []f64,
    /// Recycled directions U and C = Â U (orthonormal), k vectors each.
    u: []f64,
    c: []f64,
    /// The next U and C while they are built, k vectors each.
    tu: []f64,
    tc: []f64,
    // Scratch, n each: M^-1 v, Â v, the residual and the correction.
    w: []f64,
    t: []f64,
    r: []f64,
    d: []f64,
    /// H̄ unrotated and Givens-rotated, (m + 1) x m row-major.
    h: []f64,
    q: []f64,
    /// B = C^T Â V, k x m row-major.
    b: []f64,
    cs: []f64,
    sn: []f64,
    g: []f64,
    y: []f64,
    /// Harmonic Ritz scratch: Ḡ ((m + 1) x m), A_e, B_e, the eigen input
    /// (m x m each), V̂^T Ŵ ((m + 1) x m), the stacked shifted system
    /// (2m x 2m), its two vectors (2m each), P and Ḡ P (m + 1 by k).
    gbar: []f64,
    ae: []f64,
    be: []f64,
    em: []f64,
    vw: []f64,
    lu2: []f64,
    z2: []f64,
    b2: []f64,
    p: []f64,
    gp: []f64,
    piv: []u32,
    theta: []num.Complex,

    /// Asserts n > 0 and k < m; k = 0 is plain GMRES(m). Caller frees with `deinit` and the same
    /// allocator.
    pub fn init(gpa: Allocator, n: u32, m: u32, k: u32) Allocator.Error!Self {
        assert(n > 0);
        assert(k < m);
        // The Ritz selection keeps fixed 64-entry tables.
        assert(m <= 64);
        const nu: usize = n;
        const mu: usize = m;
        const ku: usize = k;
        const len = (mu + 1) * nu + 4 * ku * nu + 4 * nu + 2 * (mu + 1) * mu + ku * mu + 2 * mu + 2 * (mu + 1) +
            (mu + 1) * mu + 3 * mu * mu + (mu + 1) * mu + 4 * mu * mu + 4 * mu + 2 * (mu + 1) * ku;
        const slab = try gpa.alloc(f64, len);
        errdefer gpa.free(slab);
        const piv = try gpa.alloc(u32, 2 * mu);
        errdefer gpa.free(piv);
        const theta = try gpa.alloc(num.Complex, mu);
        var rest = slab;
        const take = struct {
            fn f(buf: *[]f64, l: usize) []f64 {
                const head = buf.*[0..l];
                buf.* = buf.*[l..];
                return head;
            }
        }.f;
        const self: Self = .{
            .n = n,
            .m = m,
            .k = k,
            .slab = slab,
            .v = take(&rest, (mu + 1) * nu),
            .u = take(&rest, ku * nu),
            .c = take(&rest, ku * nu),
            .tu = take(&rest, ku * nu),
            .tc = take(&rest, ku * nu),
            .w = take(&rest, nu),
            .t = take(&rest, nu),
            .r = take(&rest, nu),
            .d = take(&rest, nu),
            .h = take(&rest, (mu + 1) * mu),
            .q = take(&rest, (mu + 1) * mu),
            .b = take(&rest, ku * mu),
            .cs = take(&rest, mu),
            .sn = take(&rest, mu),
            .g = take(&rest, mu + 1),
            .y = take(&rest, mu + 1),
            .gbar = take(&rest, (mu + 1) * mu),
            .ae = take(&rest, mu * mu),
            .be = take(&rest, mu * mu),
            .em = take(&rest, mu * mu),
            .vw = take(&rest, (mu + 1) * mu),
            .lu2 = take(&rest, 4 * mu * mu),
            .z2 = take(&rest, 2 * mu),
            .b2 = take(&rest, 2 * mu),
            .p = take(&rest, (mu + 1) * ku),
            .gp = take(&rest, (mu + 1) * ku),
            .piv = piv,
            .theta = theta,
        };
        assert(rest.len == 0);
        return self;
    }

    /// Frees the workspace, the recycled space included; `gpa` must be the
    /// allocator `init` got.
    pub fn deinit(self: *Self, gpa: Allocator) void {
        gpa.free(self.slab);
        gpa.free(self.piv);
        gpa.free(self.theta);
        self.* = undefined;
    }

    /// Drops the recycled space: the next solve starts as plain GMRES.
    pub fn forget(self: *Self) void {
        self.have = 0;
    }

    /// Solves A x = b as `Gmres.solve` does, `x` holding the start; at most
    /// `max_restarts + 1` cycles of up to m - (recycled) Arnoldi steps.
    /// `iterations` counts operator applications in the cycles and in
    /// rebuilding C for this A. Asserts that `b` and `x` hold n entries.
    pub fn solve(self: *Self, op: anytype, b: []const f64, x: []f64, tol: f64, max_restarts: u32) SolveResult {
        const n: usize = self.n;
        assert(b.len >= n);
        assert(x.len >= n);
        const b_norm = num.norm2(b[0..n]);
        if (b_norm == 0) {
            @memset(x[0..n], 0);
            return .{ .iterations = 0, .residual = 0, .converged = true };
        }
        if (!std.math.isFinite(b_norm)) return .{ .iterations = 0, .residual = b_norm, .converged = false };
        const abs_tol = tol * b_norm;
        var iters: u32 = 0;

        op.matvec(x[0..n], self.r[0..n]);
        num.sub(self.r[0..n], b[0..n], self.r[0..n]);
        const r0 = num.norm2(self.r[0..n]);
        if (r0 <= abs_tol) return .{ .iterations = 0, .residual = r0 / b_norm, .converged = true };
        @memset(self.d[0..n], 0);

        // C = Â U for this operator, then U·R^-1 keeps C = Â U.
        if (self.have != 0) {
            for (0..self.have) |i| self.apply(op, vec(self.u, n, i), vec(self.c, n, i));
            iters += self.have;
            self.have = @intCast(orthonormalize(n, self.c, self.u, self.have));
        }

        var cycle: u64 = 0;
        while (cycle <= max_restarts) : (cycle += 1) {
            const steps = self.cycleOnce(op, abs_tol);
            iters += steps;
            if (num.norm2(self.r[0..n]) > abs_tol and cycle < max_restarts) {
                self.recycle(steps);
                continue;
            }
            // Flush d into x and measure the true residual.
            if (self.flush(op, b[0..n], x[0..n]) <= abs_tol) break;
        } else {
            return .{ .iterations = iters, .residual = num.norm2(self.r[0..n]) / b_norm, .converged = false };
        }
        const res = num.norm2(self.r[0..n]);
        return .{ .iterations = iters, .residual = res / b_norm, .converged = res <= abs_tol };
    }

    // ---- private ----

    inline fn vec(buf: []f64, n: usize, i: usize) []f64 {
        return buf[i * n ..][0..n];
    }

    /// out = Â v = A M^-1 v.
    inline fn apply(self: *Self, op: anytype, v: []const f64, out: []f64) void {
        const n: usize = self.n;
        if (comptime @hasDecl(std.meta.Child(@TypeOf(op)), "precond")) {
            @memcpy(self.w[0..n], v);
            op.precond(self.w[0..n]);
            op.matvec(self.w[0..n], out);
        } else op.matvec(v, out);
    }

    /// x += M^-1 d, d = 0, r = b - A x; returns ||r||.
    fn flush(self: *Self, op: anytype, b: []const f64, x: []f64) f64 {
        const n: usize = self.n;
        if (comptime @hasDecl(std.meta.Child(@TypeOf(op)), "precond")) op.precond(self.d[0..n]);
        num.axpy(x, 1, self.d[0..n]);
        @memset(self.d[0..n], 0);
        op.matvec(x, self.r[0..n]);
        num.sub(self.r[0..n], b, self.r[0..n]);
        return num.norm2(self.r[0..n]);
    }

    /// One cycle from the current r: project r off C, Arnoldi on
    /// (I - C C^T) Â, update d and r. Returns the Arnoldi steps, the
    /// operator applications; `recycle` may read the cycle's basis next.
    fn cycleOnce(self: *Self, op: anytype, abs_tol: f64) u32 {
        const n: usize = self.n;
        const m: usize = self.m;
        const kk: usize = self.have;
        const r = self.r[0..n];
        // d += U C^T r, r -= C C^T r: exact after a converged cycle, and a
        // repair after a rank-deficient least squares.
        for (0..kk) |i| {
            const ci = vec(self.c, n, i);
            const s = num.dot(r, ci);
            num.axpy(r, -s, ci);
            num.axpy(self.d[0..n], s, vec(self.u, n, i));
        }
        const beta = num.norm2(r);
        if (beta == 0) return 0;
        const mm = m - kk;
        num.scale(vec(self.v, n, 0), 1 / beta, r);
        @memset(self.g[0 .. mm + 1], 0);
        self.g[0] = beta;
        var steps: usize = 0;
        for (0..mm) |j| {
            steps = j + 1;
            const t = self.t[0..n];
            self.apply(op, vec(self.v, n, j), t);
            for (0..kk) |i| {
                const ci = vec(self.c, n, i);
                const bij = num.dot(t, ci);
                self.b[i * m + j] = bij;
                num.axpy(t, -bij, ci);
            }
            for (0..j + 1) |i| {
                const vi = vec(self.v, n, i);
                const hij = num.dot(t, vi);
                self.h[i * m + j] = hij;
                num.axpy(t, -hij, vi);
            }
            const hn = num.norm2(t);
            self.h[(j + 1) * m + j] = hn;
            if (hn != 0) num.scale(vec(self.v, n, j + 1), 1 / hn, t) else @memset(vec(self.v, n, j + 1), 0);
            // Rotated copy of column j.
            for (0..j + 2) |i| self.q[i * m + j] = self.h[i * m + j];
            for (0..j) |i| {
                const a = self.q[i * m + j];
                const bb = self.q[(i + 1) * m + j];
                self.q[i * m + j] = self.cs[i] * a + self.sn[i] * bb;
                self.q[(i + 1) * m + j] = -self.sn[i] * a + self.cs[i] * bb;
            }
            const a = self.q[j * m + j];
            const rho = std.math.hypot(a, hn);
            self.cs[j] = if (rho == 0) 1 else a / rho;
            self.sn[j] = if (rho == 0) 0 else hn / rho;
            self.q[j * m + j] = rho;
            self.q[(j + 1) * m + j] = 0;
            const gj = self.g[j];
            self.g[j] = self.cs[j] * gj;
            self.g[j + 1] = -self.sn[j] * gj;
            if (hn == 0 or @abs(self.g[j + 1]) <= abs_tol) break;
        }
        // y2 = R^-1 g; a tiny diagonal zeroes its component as Gmres does.
        var d_max: f64 = 0;
        for (0..steps) |i| d_max = @max(d_max, @abs(self.q[i * m + i]));
        const floor = std.math.floatEps(f64) * d_max;
        var i: usize = steps;
        while (i > 0) {
            i -= 1;
            var s = self.g[i];
            for (i + 1..steps) |jj| s -= self.q[i * m + jj] * self.y[jj];
            const diag = self.q[i * m + i];
            self.y[i] = if (@abs(diag) <= floor) 0 else s / diag;
        }
        const y2 = self.y[0..steps];
        // d += V y2 - U B y2.
        for (0..steps) |j| num.axpy(self.d[0..n], y2[j], vec(self.v, n, j));
        for (0..kk) |ii| {
            var s: f64 = 0;
            for (0..steps) |j| s += self.b[ii * m + j] * y2[j];
            num.axpy(self.d[0..n], -s, vec(self.u, n, ii));
        }
        // r -= V_{s+1} H̄ y2 (C's part cancels against U's).
        // H̄ is Hessenberg: row `row` starts at column row - 1.
        for (0..steps + 1) |row| {
            var s: f64 = 0;
            for (row -| 1..steps) |j| s += self.h[row * m + j] * y2[j];
            num.axpy(r, -s, vec(self.v, n, row));
        }
        return @intCast(steps);
    }

    /// Replaces U and C with the k harmonic Ritz vectors of smallest
    /// modulus over the cycle's space [U, V_steps]. Leaves them alone when
    /// the space has no more than k directions or a small solve fails.
    fn recycle(self: *Self, steps: usize) void {
        const n: usize = self.n;
        const m: usize = self.m;
        const kk: usize = self.have;
        const mc = kk + steps;
        if (mc <= self.k) return;
        const rows = mc + 1;
        // Ḡ = [D B; 0 H̄], rows x mc, stride mc; D = diag(1/||u_i||) scales U.
        const gb = self.gbar[0 .. rows * mc];
        @memset(gb, 0);
        var dscale: [64]f64 = undefined;
        assert(kk <= dscale.len);
        for (0..kk) |ii| {
            const un = num.norm2(vec(self.u, n, ii));
            if (un == 0) return;
            dscale[ii] = 1 / un;
            gb[ii * mc + ii] = dscale[ii];
            for (0..steps) |j| gb[ii * mc + kk + j] = self.b[ii * m + j];
        }
        for (0..steps + 1) |row| for (row -| 1..steps) |j| {
            gb[(kk + row) * mc + kk + j] = self.h[row * m + j];
        };
        const em = self.em[0 .. mc * mc];
        if (kk == 0) {
            // H_m + h² H_m^-T e_m e_m^T, h = H̄[m, m-1].
            const ht = self.lu2[0 .. mc * mc];
            for (0..mc) |row| for (0..mc) |col| {
                ht[col * mc + row] = gb[row * mc + col];
            };
            const f = self.z2[0..mc];
            @memset(self.b2[0..mc], 0);
            self.b2[mc - 1] = 1;
            dense_lu.factorize(mc, ht, self.piv) catch return;
            dense_lu.solveFactored(mc, ht, self.piv, self.b2[0..mc], f);
            const hl = gb[mc * mc + mc - 1];
            @memcpy(em, gb[0 .. mc * mc]);
            for (0..mc) |row| em[row * mc + mc - 1] += hl * hl * f[row];
        } else {
            // A_e = Ḡ^T Ḡ, B_e = Ḡ^T (V̂^T Ŵ), M = B_e^-1 A_e;
            // V̂^T Ŵ = [C^T U D, 0; V^T U D, I] with C^T V = 0.
            const vw = self.vw[0 .. rows * mc];
            @memset(vw, 0);
            for (0..kk) |col| {
                const uc = vec(self.u, n, col);
                for (0..kk) |row| vw[row * mc + col] = num.dot(vec(self.c, n, row), uc) * dscale[col];
                for (0..steps + 1) |row| vw[(kk + row) * mc + col] = num.dot(vec(self.v, n, row), uc) * dscale[col];
            }
            for (0..steps) |j| vw[(kk + j) * mc + kk + j] = 1;
            const ae = self.ae[0 .. mc * mc];
            const be = self.be[0 .. mc * mc];
            for (0..mc) |row| for (0..mc) |col| {
                var sa: f64 = 0;
                var sb: f64 = 0;
                for (0..rows) |l| {
                    sa += gb[l * mc + row] * gb[l * mc + col];
                    sb += gb[l * mc + row] * vw[l * mc + col];
                }
                ae[row * mc + col] = sa;
                be[row * mc + col] = sb;
            };
            dense_lu.factorize(mc, be, self.piv) catch return;
            const col_in = self.b2[0..mc];
            const col_out = self.z2[0..mc];
            for (0..mc) |col| {
                for (0..mc) |row| col_in[row] = ae[row * mc + col];
                dense_lu.solveFactored(mc, be, self.piv, col_in, col_out);
                for (0..mc) |row| em[row * mc + col] = col_out[row];
            }
        }
        // P: real basis of the k smallest-modulus eigenvectors of em.
        const kept = self.ritz(mc, em) orelse return;
        const p = self.p[0 .. mc * kept];
        // Ỹ = Ŵ P, C̃ = V̂ (Ḡ P).
        const gp = self.gp[0 .. rows * kept];
        for (0..rows) |row| for (0..kept) |l| {
            var s: f64 = 0;
            for (0..mc) |col| s += gb[row * mc + col] * p[col * kept + l];
            gp[row * kept + l] = s;
        };
        for (0..kept) |l| {
            const yl = vec(self.tu, n, l);
            const cl = vec(self.tc, n, l);
            @memset(yl, 0);
            @memset(cl, 0);
            for (0..kk) |ii| {
                num.axpy(yl, p[ii * kept + l] * dscale[ii], vec(self.u, n, ii));
                num.axpy(cl, gp[ii * kept + l], vec(self.c, n, ii));
            }
            for (0..steps) |j| num.axpy(yl, p[(kk + j) * kept + l], vec(self.v, n, j));
            for (0..steps + 1) |j| num.axpy(cl, gp[(kk + j) * kept + l], vec(self.v, n, j));
        }
        const got = orthonormalize(n, self.tc, self.tu, kept);
        if (got == 0) return;
        std.mem.swap([]f64, &self.u, &self.tu);
        std.mem.swap([]f64, &self.c, &self.tc);
        self.have = @intCast(got);
    }

    /// Writes into `p` (mc x kept, stride kept) a real basis of the
    /// eigenvectors of `em` (mc x mc, destroyed) for its smallest-modulus
    /// eigenvalues, a complex pair as its real and imaginary parts, kept
    /// <= k. Null when the eigenvalues do not converge or none is usable.
    fn ritz(self: *Self, mc: usize, em: []f64) ?usize {
        const k: usize = self.k;
        // Inverse iteration needs em after `eigenvalues` destroys it.
        const keep = self.ae[0 .. mc * mc];
        @memcpy(keep, em);
        var norm: f64 = 0;
        for (em) |e| norm = @max(norm, @abs(e));
        if (!(norm > 0) or !std.math.isFinite(norm)) return null;
        const eig = core.eigen.eigenvalues(mc, em, self.theta[0..mc], 1e-14, 60 * @as(u32, @intCast(mc)));
        if (!eig.converged) return null;
        const theta = self.theta[0..mc];
        // Selection sort by modulus over the eigenvalues still wanted.
        var used: [64]bool = @splat(false);
        assert(mc <= used.len);
        var picked: [64]num.Complex = undefined;
        var slots: usize = 0;
        var count: usize = 0;
        while (true) {
            var best: ?usize = null;
            for (theta, 0..) |th, i| {
                // One of each conjugate pair: the one with im >= 0.
                if (used[i] or th.im < 0) continue;
                if (best == null or th.mag() < theta[best.?].mag()) best = i;
            }
            const bi = best orelse break;
            used[bi] = true;
            const real = @abs(theta[bi].im) <= 1e-12 * theta[bi].mag();
            const need: usize = if (real) 1 else 2;
            if (slots + need > k) break;
            picked[count] = if (real) .{ .re = theta[bi].re, .im = 0 } else theta[bi];
            count += 1;
            slots += need;
        }
        if (slots == 0) return null;
        const p = self.p[0 .. mc * slots];
        var col: usize = 0;
        const nn = 2 * mc;
        for (picked[0..count]) |th| {
            // (em - θ' I) z = b in stacked real form, θ' nudged off θ.
            const shift: num.Complex = .{ .re = th.re + 1e-10 * @max(th.mag(), 1e-6 * norm), .im = th.im };
            const a = self.lu2[0 .. nn * nn];
            @memset(a, 0);
            for (0..mc) |row| for (0..mc) |cc| {
                const v = keep[row * mc + cc] - (if (row == cc) shift.re else 0);
                a[row * nn + cc] = v;
                a[(mc + row) * nn + mc + cc] = v;
            };
            for (0..mc) |ii| {
                a[ii * nn + mc + ii] = shift.im;
                a[(mc + ii) * nn + ii] = -shift.im;
            }
            dense_lu.factorize(nn, a, self.piv) catch return null;
            const z = self.z2[0..nn];
            const rhs = self.b2[0..nn];
            for (rhs[0..mc]) |*e| e.* = 1;
            @memset(rhs[mc..], 0);
            for (0..2) |_| {
                dense_lu.solveFactored(nn, a, self.piv, rhs, z);
                const zn = num.norm2(z);
                if (!(zn > 0) or !std.math.isFinite(zn)) return null;
                num.scale(rhs, 1 / zn, z);
            }
            for (0..mc) |row| p[row * slots + col] = rhs[row];
            col += 1;
            if (th.im != 0) {
                for (0..mc) |row| p[row * slots + col] = rhs[mc + row];
                col += 1;
            }
        }
        assert(col == slots);
        return slots;
    }
};

/// A dense nonsymmetric test operator with a cluster of small eigenvalues,
/// the shape restarted GMRES stalls on, and a Jacobi preconditioner.
const TestOp = struct {
    n: usize,
    a: []f64,
    diag: []f64,
    pub fn matvec(self: *TestOp, v: []const f64, w: []f64) void {
        for (0..self.n) |i| {
            var s: f64 = 0;
            for (0..self.n) |j| s += self.a[i * self.n + j] * v[j];
            w[i] = s;
        }
    }
    pub fn precond(self: *TestOp, r: []f64) void {
        for (r, self.diag) |*e, dg| e.* /= dg;
    }
};

// ---- private ----

/// Modified Gram-Schmidt QR of the `cnt` vectors in `c` (n each), applying
/// R^-1 to the matching vectors of `u` so that C = Q R keeps C U^-1 fixed:
/// afterwards c is orthonormal and Â u_i = c_i still holds. A column whose
/// norm falls under 1e-10 of its norm before orthogonalization is
/// dropped and the rest compacted. Returns the columns kept.
fn orthonormalize(n: usize, c: []f64, u: []f64, cnt: usize) usize {
    var kept: usize = 0;
    for (0..cnt) |l| {
        const cl = c[l * n ..][0..n];
        const ul = u[l * n ..][0..n];
        const before = num.norm2(cl);
        for (0..kept) |i| {
            const ci = c[i * n ..][0..n];
            const s = num.dot(cl, ci);
            num.axpy(cl, -s, ci);
            num.axpy(ul, -s, u[i * n ..][0..n]);
        }
        const after = num.norm2(cl);
        if (!(after > 1e-10 * before) or !std.math.isFinite(after)) continue;
        num.scale(cl, 1 / after, cl);
        num.scale(ul, 1 / after, ul);
        if (kept != l) {
            @memcpy(c[kept * n ..][0..n], cl);
            @memcpy(u[kept * n ..][0..n], ul);
        }
        kept += 1;
    }
    return kept;
}

test "Gcrodr solves a sequence of drifting systems to tolerance, recycling cuts the operator applications" {
    const gpa = testing.allocator;
    const n = 120;
    var prng = std.Random.DefaultPrng.init(0x9c0d);
    const rnd = prng.random();
    const a = try gpa.alloc(f64, n * n);
    defer gpa.free(a);
    const base = try gpa.alloc(f64, n * n);
    defer gpa.free(base);
    const diag = try gpa.alloc(f64, n);
    defer gpa.free(diag);
    for (0..n) |i| for (0..n) |j| {
        base[i * n + j] = if (i == j) (if (i < 8) 1e-3 * @as(f64, @floatFromInt(i + 1)) else 1 + @as(f64, @floatFromInt(i)) / n) else 0.3 * (rnd.float(f64) - 0.5) / @sqrt(@as(f64, n));
    };
    const b = try gpa.alloc(f64, n);
    defer gpa.free(b);
    const x = try gpa.alloc(f64, n);
    defer gpa.free(x);
    const ax = try gpa.alloc(f64, n);
    defer gpa.free(ax);
    for (b) |*e| e.* = rnd.float(f64) - 0.5;
    var rec = try Gcrodr.init(gpa, n, 20, 8);
    defer rec.deinit(gpa);
    var plain = try @import("gmres.zig").Gmres.init(gpa, n, 20);
    defer plain.deinit(gpa);
    var total_rec: u32 = 0;
    var total_plain: u32 = 0;
    for (0..6) |step| {
        const drift = 1 + 0.01 * @as(f64, @floatFromInt(step));
        for (a, base) |*e, bv| e.* = bv * drift;
        for (diag, 0..) |*dg, i| dg.* = a[i * n + i] + 0.5;
        var op: TestOp = .{ .n = n, .a = a, .diag = diag };
        @memset(x, 0);
        const res = rec.solve(&op, b, x, 1e-10, 50);
        try testing.expect(res.converged);
        op.matvec(x, ax);
        for (ax, b) |*e, bv| e.* -= bv;
        try testing.expect(num.norm2(ax) <= 1e-9 * num.norm2(b));
        total_rec += res.iterations;
        @memset(x, 0);
        const pres = plain.solve(&op, b, x, 1e-10, 50);
        try testing.expect(pres.converged);
        total_plain += pres.iterations;
    }
    try testing.expect(rec.have > 0);
    try testing.expect(total_rec < total_plain);
}

test "Gcrodr returns at once on a zero right-hand side and on a start that already solves" {
    const gpa = testing.allocator;
    var rec = try Gcrodr.init(gpa, 3, 3, 1);
    defer rec.deinit(gpa);
    var a = [_]f64{ 2, 0, 0, 0, 3, 0, 0, 0, 4 };
    var diag = [_]f64{ 1, 1, 1 };
    var op: TestOp = .{ .n = 3, .a = &a, .diag = &diag };
    var x = [_]f64{ 1, 1, 1 };
    const zero = [_]f64{ 0, 0, 0 };
    try testing.expect(rec.solve(&op, &zero, &x, 1e-12, 3).converged);
    try testing.expectEqualSlices(f64, &zero, &x);
    const b = [_]f64{ 2, 3, 4 };
    x = .{ 1, 1, 1 };
    const res = rec.solve(&op, &b, &x, 1e-12, 3);
    try testing.expect(res.converged);
    try testing.expectEqual(@as(u32, 0), res.iterations);
}

test "Gcrodr.init frees what it took on every allocation failure" {
    // Oracle: std.testing.allocator's leak and double-free detection, each of
    // init's three allocations failing once.
    try testing.checkAllAllocationFailures(testing.allocator, struct {
        fn run(gpa: Allocator) !void {
            var g = try Gcrodr.init(gpa, 9, 6, 2);
            g.deinit(gpa);
        }
    }.run, .{});
}
