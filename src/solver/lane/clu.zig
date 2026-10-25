//! W complex sparse LU factorizations at once on one n x n pivot tape: the
//! AC family's A(ω) = G + jωC solved in complex arithmetic instead of the
//! stacked-real 2n form.
//! One index per complex entry where the stacked form reads four, and a
//! complex multiply-add is 4 real multiplies against the 2x2 block's 8.
//!
//! The tape is a real SparseLu laid on the source pattern (G ∪ C) by
//! `freq_solve`, so its walks, pivot sequence and L/U patterns are
//! SparseLu's; only the values are complex, held as separate re and im
//! planes of `@Vector(W, f64)`. The pivot sequence was chosen on a real
//! proxy of the moduli, not on the complex values, so every replay checks
//! each pivot against its column under the ∞-norm |re| ∨ |im| (within √2
//! of the modulus) and reports the lanes that fail. Pivots are stored as
//! reciprocals. In: a factored SparseLu and a column source. Out: W lanes
//! of L, U and 1/diag, and W-lane solves of A x = b and A^H x = b.

const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;
const SparseLu = @import("../sparse_lu.zig").SparseLu;

/// W complex factorizations sharing one SparseLu tape. W is at most 64,
/// the width of the failure mask.
pub fn ComplexLaneLu(comptime W: usize) type {
    comptime {
        assert(W >= 1);
        assert(W <= 64);
    }
    return struct {
        const Self = @This();
        /// One entry's real or imaginary part across the lanes (lane l in
        /// element l).
        pub const V = @Vector(W, f64);
        const I = @Vector(W, i64);
        const U = @Vector(W, u64);

        /// Borrowed factored tape; must outlive self and keep its L/U
        /// lengths (a new `factor` of it means a new ComplexLaneLu).
        base: *const SparseLu,
        n: u32,

        // Real then imaginary part of every L entry, U entry and pivot
        // reciprocal.
        lr: []V,
        li: []V,
        ur: []V,
        ui: []V,
        dr: []V,
        di: []V,

        /// Accumulator, all-zero between calls.
        wr: []V,
        wi: []V,
        /// Solve workspace.
        yr: []V,
        yi: []V,

        /// Sizes the planes from `base`. Asserts that `base` is factored.
        pub fn init(gpa: Allocator, base: *const SparseLu) !Self {
            assert(base.factored);
            const n: usize = base.n;
            const nl = base.lx.items.len;
            const nu = base.ux.items.len;
            // One block: L, U, 1/diag, the accumulator and the solve planes.
            const block = try gpa.alloc(V, 2 * (nl + nu) + 6 * n);
            @memset(block[2 * (nl + nu) + 2 * n ..][0 .. 2 * n], @as(V, @splat(0)));
            var at: usize = 0;
            const take = struct {
                fn f(b: []V, p: *usize, len: usize) []V {
                    defer p.* += len;
                    return b[p.*..][0..len];
                }
            }.f;
            return .{
                .base = base,
                .n = base.n,
                .lr = take(block, &at, nl),
                .li = take(block, &at, nl),
                .ur = take(block, &at, nu),
                .ui = take(block, &at, nu),
                .dr = take(block, &at, n),
                .di = take(block, &at, n),
                .wr = take(block, &at, n),
                .wi = take(block, &at, n),
                .yr = take(block, &at, n),
                .yi = take(block, &at, n),
            };
        }

        /// Frees the one value block; `base` stays the caller's.
        pub fn deinit(self: *Self, gpa: Allocator) void {
            const len = 2 * (self.lr.len + self.ur.len) + 6 * @as(usize, self.n);
            gpa.free(self.lr.ptr[0..len]);
            self.* = undefined;
        }

        /// True while `base` still has the L/U lengths the planes were sized for.
        pub fn fits(self: *const Self) bool {
            return self.base.factored and self.lr.len == self.base.lx.items.len and self.ur.len == self.base.ux.items.len;
        }

        /// The `refactor` source for callers that hold the values: entry p
        /// of lane l is `re[p][l] + j im[p][l]`.
        pub const Plane = struct {
            col_ptr: []const u32,
            re: []const V,
            im: []const V,

            /// Writes column `c`'s entries to `wr`/`wi`, entry p at row `prow[p]`.
            pub inline fn load(self: Plane, c: usize, wr: []V, wi: []V, prow: []const u32) void {
                for (self.col_ptr[c]..self.col_ptr[c + 1]) |p| {
                    wr[prow[p]] = self.re[p];
                    wi[prow[p]] = self.im[p];
                }
            }
        };

        /// Replays the tape on `src`'s values: `src.load(c, wr, wi, prow)`
        /// writes source column c, entry p at row `prow[p]`, as
        /// `Plane.load`. Returns the mask of lanes whose pivot was zero or
        /// non-finite or fell below `growth_limit` times its column's
        /// ∞-norm max, or that stored a non-finite L or U entry; their
        /// factors are garbage. A fabricated (void) pivot in the tape fails
        /// every lane. Asserts that the planes fit `base`.
        pub fn refactor(self: *Self, src: anytype, growth_limit: f64) u64 {
            const b = self.base;
            assert(self.fits());
            const lidx = b.li.items;
            const uidx = b.ui.items;
            const up = b.up;
            const lp = b.lp;
            const prow = b.prow;
            const q = b.q;
            const lr = self.lr;
            const li = self.li;
            const ur = self.ur;
            const ui = self.ui;
            const wr = self.wr;
            const wi = self.wi;
            const zero: V = @splat(0);
            const one: V = @splat(1);
            const inf: V = @splat(std.math.inf(f64));
            const all_lanes: u64 = std.math.maxInt(@Int(.unsigned, W));
            if (std.mem.indexOfScalar(bool, b.void_col, true) != null) return all_lanes;
            var bad: u64 = 0;
            var nonfinite: U = @splat(0);

            for (0..self.n) |k| {
                src.load(q[k], wr, wi, prow);

                for (up[k]..up[k + 1]) |p| {
                    const i = uidx[p];
                    const xr = wr[i];
                    const xi = wi[i];
                    wr[i] = zero;
                    wi[i] = zero;
                    ur[p] = xr;
                    ui[p] = xi;
                    nonfinite |= @as(U, @bitCast(xr - xr)) | @as(U, @bitCast(xi - xi));
                    for (lp[i]..lp[i + 1]) |pl| {
                        const r = lidx[pl];
                        wr[r] -= lr[pl] * xr - li[pl] * xi;
                        wi[r] -= lr[pl] * xi + li[pl] * xr;
                    }
                }

                var pr = wr[k];
                var pi = wi[k];
                wr[k] = zero;
                wi[k] = zero;
                const pmax = norm(pr, pi);
                const dead = (pmax == zero) | !(pmax < inf);
                bad |= maskBits(dead);
                pr = @select(f64, dead, one, pr);
                pi = @select(f64, dead, zero, pi);
                // 1/d by Smith's scaling: no overflow or underflow in |d|^2.
                const s = @max(@abs(pr), @abs(pi));
                const a = pr / s;
                const c = pi / s;
                const den = s * (a * a + c * c);
                const rr = a / den;
                const ri = -c / den;
                self.dr[k] = rr;
                self.di[k] = ri;

                const check = growth_limit > 0 and !b.scaled_pivot[k];
                // Bit-pattern max: NaN sorts above inf and survives it.
                var cbits: I = @bitCast(@select(f64, dead, zero, pmax));
                for (lp[k]..lp[k + 1]) |p| {
                    const r = lidx[p];
                    const vr = wr[r];
                    const vi = wi[r];
                    wr[r] = zero;
                    wi[r] = zero;
                    cbits = @max(cbits, @as(I, @bitCast(norm(vr, vi))));
                    const xr = vr * rr - vi * ri;
                    const xi = vr * ri + vi * rr;
                    lr[p] = xr;
                    li[p] = xi;
                    nonfinite |= @as(U, @bitCast(xr - xr)) | @as(U, @bitCast(xi - xi));
                }
                if (check) {
                    const cmax: V = @bitCast(cbits);
                    bad |= maskBits(!(pmax >= @as(V, @splat(growth_limit)) * cmax));
                }
            }
            bad |= maskBits(nonfinite != @as(U, @splat(0)));
            assert(bad & ~all_lanes == 0);
            return bad;
        }

        /// x = A^-1 b per lane, b and x split into re and im planes of n
        /// entries; inputs and outputs may alias. A lane `refactor` masked
        /// holds garbage.
        pub fn solve(self: *Self, br: []const V, bi: []const V, xr: []V, xi: []V) void {
            const b = self.base;
            assert(br.len == self.n and bi.len == self.n);
            assert(xr.len >= self.n and xi.len >= self.n);
            const lidx = b.li.items;
            const uidx = b.ui.items;
            const lp = b.lp;
            const up = b.up;
            const yr = self.yr;
            const yi = self.yi;
            for (b.pinv, 0..) |k, r| {
                yr[k] = br[r];
                yi[k] = bi[r];
            }
            for (0..self.n) |k| {
                const sr = yr[k];
                const si = yi[k];
                for (lp[k]..lp[k + 1]) |p| {
                    const r = lidx[p];
                    yr[r] -= self.lr[p] * sr - self.li[p] * si;
                    yi[r] -= self.lr[p] * si + self.li[p] * sr;
                }
            }
            var k = self.n;
            while (k > 0) {
                k -= 1;
                const sr = yr[k] * self.dr[k] - yi[k] * self.di[k];
                const si = yr[k] * self.di[k] + yi[k] * self.dr[k];
                yr[k] = sr;
                yi[k] = si;
                for (up[k]..up[k + 1]) |p| {
                    const r = uidx[p];
                    yr[r] -= self.ur[p] * sr - self.ui[p] * si;
                    yi[r] -= self.ur[p] * si + self.ui[p] * sr;
                }
            }
            for (b.q, 0..) |c, j| {
                xr[c] = yr[j];
                xi[c] = yi[j];
            }
        }

        /// x = A^-H b per lane (conjugate transpose: the stacked-real
        /// transpose's complex meaning). Same layout and aliasing as `solve`.
        pub fn solveH(self: *Self, br: []const V, bi: []const V, xr: []V, xi: []V) void {
            const b = self.base;
            assert(br.len == self.n and bi.len == self.n);
            assert(xr.len >= self.n and xi.len >= self.n);
            const lidx = b.li.items;
            const uidx = b.ui.items;
            const lp = b.lp;
            const up = b.up;
            const yr = self.yr;
            const yi = self.yi;
            for (b.q, 0..) |c, k| {
                yr[k] = br[c];
                yi[k] = bi[c];
            }
            // U^H z = b: z_k = (b_k - Σ conj(u_ik) z_i) / conj(d_k).
            for (0..self.n) |k| {
                var sr = yr[k];
                var si = yi[k];
                for (up[k]..up[k + 1]) |p| {
                    const r = uidx[p];
                    sr -= self.ur[p] * yr[r] + self.ui[p] * yi[r];
                    si -= self.ur[p] * yi[r] - self.ui[p] * yr[r];
                }
                yr[k] = sr * self.dr[k] + si * self.di[k];
                yi[k] = si * self.dr[k] - sr * self.di[k];
            }
            // L^H y = z, unit diagonal.
            var k = self.n;
            while (k > 0) {
                k -= 1;
                var sr = yr[k];
                var si = yi[k];
                for (lp[k]..lp[k + 1]) |p| {
                    const r = lidx[p];
                    sr -= self.lr[p] * yr[r] + self.li[p] * yi[r];
                    si -= self.lr[p] * yi[r] - self.li[p] * yr[r];
                }
                yr[k] = sr;
                yi[k] = si;
            }
            for (b.pinv, 0..) |k2, r| {
                xr[r] = yr[k2];
                xi[r] = yi[k2];
            }
        }

        /// max(|re|, |im|) by bit pattern, so a NaN in either part survives
        /// (a float @max drops it).
        inline fn norm(re: V, im: V) V {
            return @bitCast(@max(@as(I, @bitCast(@abs(re))), @as(I, @bitCast(@abs(im)))));
        }

        inline fn maskBits(m: @Vector(W, bool)) u64 {
            const bits: @Int(.unsigned, W) = @bitCast(m);
            return @as(u64, bits) & std.math.maxInt(@Int(.unsigned, W));
        }
    };
}

const testing = std.testing;

test "ComplexLaneLu solves A x = b and A^H x = b on random complex systems; bad lanes are masked" {
    // W = 8 is the AVX-512 width (TODO 9.7): built and checked here even
    // on AVX2 hosts, where it lowers to two registers per vector.
    inline for (.{ 1, 4, 8 }) |w| try expectComplexLanes(w);
}

fn expectComplexLanes(comptime W: usize) !void {
    const gpa = testing.allocator;
    const L = ComplexLaneLu(W);
    const Sys = SparseLu.test_access.TestSystem;
    var prng = std.Random.DefaultPrng.init(0xc0de);
    const r = prng.random();
    var solved: usize = 0;
    for (0..64) |t| {
        const n: u32 = @intCast(2 + t % 20);
        const sys = try Sys.init(gpa, r, n, 0.3);
        defer sys.deinit(gpa);
        const nnz = sys.vals.len;
        const q = try gpa.alloc(u32, n);
        defer gpa.free(q);
        for (q, 0..) |*c, i| c.* = @intCast(i);
        var base = try SparseLu.init(gpa, n, sys.col_ptr, q);
        defer base.deinit(gpa);
        // The tape: the real proxy of moduli, as freq_solve lays it.
        const im = try gpa.alloc([W]f64, nnz);
        defer gpa.free(im);
        const re = try gpa.alloc([W]f64, nnz);
        defer gpa.free(re);
        for (re, im, sys.vals) |*a, *b, v| for (0..W) |l| {
            a[l] = v * (0.9 + 0.2 * r.float(f64));
            b[l] = (2 * r.float(f64) - 1) * @abs(v);
        };
        const proxy = try gpa.alloc(f64, nnz);
        defer gpa.free(proxy);
        for (proxy, re, im) |*p, a, b| p.* = std.math.hypot(a[0], b[0]);
        base.factor(gpa, sys.col_ptr, sys.row_idx, proxy, 1e-3) catch continue;
        var ll = try L.init(gpa, &base);
        defer ll.deinit(gpa);
        // The last lane gets a NaN (W > 1): it must fail.
        if (W > 1) im[r.uintLessThan(usize, nnz)][W - 1] = std.math.nan(f64);
        const pr = try gpa.alloc(L.V, nnz);
        defer gpa.free(pr);
        const pi = try gpa.alloc(L.V, nnz);
        defer gpa.free(pi);
        for (pr, pi, re, im) |*a, *b, x, y| {
            a.* = x;
            b.* = y;
        }
        const bad = ll.refactor(L.Plane{ .col_ptr = sys.col_ptr, .re = pr, .im = pi }, 1e-3);
        if (W > 1) try testing.expect((bad >> (W - 1)) & 1 != 0);
        for (ll.wr, ll.wi) |a, b| try testing.expect(@reduce(.And, a == @as(L.V, @splat(0))) and @reduce(.And, b == @as(L.V, @splat(0))));
        const v = try gpa.alloc(L.V, 4 * @as(usize, n));
        defer gpa.free(v);
        const br = v[0..n];
        const bi = v[n .. 2 * n];
        const xr = v[2 * n .. 3 * n];
        const xi = v[3 * n ..];
        for (br, bi) |*a, *b| {
            a.* = @splat(2 * r.float(f64) - 1);
            b.* = @splat(2 * r.float(f64) - 1);
        }
        const A = struct {
            fn at(vec: L.V, l: usize) f64 {
                const arr: [W]f64 = vec;
                return arr[l];
            }
        }.at;
        for ([_]bool{ false, true }) |herm| {
            if (herm) ll.solveH(br, bi, xr, xi) else ll.solve(br, bi, xr, xi);
            for (0..W) |l| {
                if ((bad >> @intCast(l)) & 1 != 0) continue;
                solved += 1;
                // Residual b - op(A) x, scaled by |A||x| + |b|.
                var worst: f64 = 0;
                const res = try gpa.alloc(f64, 3 * @as(usize, n));
                defer gpa.free(res);
                const rr = res[0..n];
                const ri = res[n .. 2 * n];
                const sc = res[2 * n ..];
                for (0..n) |i| {
                    rr[i] = A(br[i], l);
                    ri[i] = A(bi[i], l);
                    sc[i] = @abs(A(br[i], l)) + @abs(A(bi[i], l));
                }
                for (0..n) |j| for (sys.col_ptr[j]..sys.col_ptr[j + 1]) |p| {
                    const i = sys.row_idx[p];
                    const ar = re[p][l];
                    const ai = if (herm) -im[p][l] else im[p][l];
                    const row = if (herm) j else i;
                    const col = if (herm) i else j;
                    rr[row] -= ar * A(xr[col], l) - ai * A(xi[col], l);
                    ri[row] -= ar * A(xi[col], l) + ai * A(xr[col], l);
                    sc[row] += (@abs(ar) + @abs(ai)) * (@abs(A(xr[col], l)) + @abs(A(xi[col], l)));
                };
                for (0..n) |i| worst = @max(worst, (@abs(rr[i]) + @abs(ri[i])) / sc[i]);
                try testing.expect(worst < 1e-12);
            }
        }
    }
    try testing.expect(solved > 20 * @as(usize, @max(W - 1, 1)));
}
