//! Frequency lane bodies shared by the device exports and host oracle.
//! A lane is (point, frequency). Matrices use stacked-real [G -ωC; ωC G];
//! each point has its own G/C, RHS and pivot history. No reduction mixes
//! lanes. The positive and transpose solves follow LaneLu's operation order.
//! The complex bodies replay `ComplexLaneLu` on the n x n proxy tape in its
//! operation order, re and im planes stacked per buffer.
const std = @import("std");
const lk = @import("lane.zig");

pub fn Bodies(comptime P: fn (type) type) type {
    return struct {
        /// One source CSC slot of one lane. `positions[4*slot..][0..4]`
        /// are its G, +ωC, -ωC, G stacked positions. G/C are slot-major
        /// across points, and `point[l]` chooses this frequency's point.
        pub fn fill(slot: usize, l: usize, n_lanes: usize, n_points: usize, offsets: P(u32), g: P(f64), c: P(f64), point: P(u32), omega: P(f64), a: P(f64)) void {
            const p = point[l];
            const gv = g[slot * n_points + p];
            const cv = c[slot * n_points + p];
            a[@as(usize, offsets[4 * slot]) * n_lanes + l] = gv;
            a[@as(usize, offsets[4 * slot + 1]) * n_lanes + l] = omega[l] * cv;
            a[@as(usize, offsets[4 * slot + 2]) * n_lanes + l] = -omega[l] * cv;
            a[@as(usize, offsets[4 * slot + 3]) * n_lanes + l] = gv;
        }

        /// Positive or transpose solve, with one RHS per lane. The
        /// transpose of the stacked-real matrix is the complex adjoint;
        /// the analysis readout applies its existing conjugation.
        pub fn solve(t: lk.LaneTab, indices: P(u32), lx: P(f64), ux: P(f64), diag: P(f64), rhs: P(f64), y: P(f64), out: P(f64), n_lanes: usize, l: usize, adjoint: bool) void {
            const N = n_lanes;
            if (adjoint) {
                for (0..t.n) |k| y[k * N + l] = rhs[@as(usize, indices[t.q + k]) * N + l];
                for (0..t.n) |k| {
                    var v = y[k * N + l];
                    for (indices[t.up + k]..indices[t.up + k + 1]) |p|
                        v -= ux[p * N + l] * y[@as(usize, indices[t.ui + p]) * N + l];
                    y[k * N + l] = v / diag[k * N + l];
                }
                var k: usize = t.n;
                while (k > 0) {
                    k -= 1;
                    var v = y[k * N + l];
                    for (indices[t.lp + k]..indices[t.lp + k + 1]) |p|
                        v -= lx[p * N + l] * y[@as(usize, indices[t.li + p]) * N + l];
                    y[k * N + l] = v;
                }
                for (0..t.n) |r| out[r * N + l] = y[@as(usize, indices[t.pinv + r]) * N + l];
                return;
            }
            for (0..t.n) |r| y[@as(usize, indices[t.pinv + r]) * N + l] = rhs[r * N + l];
            for (0..t.n) |k| {
                const v = y[k * N + l];
                if (v != 0) for (indices[t.lp + k]..indices[t.lp + k + 1]) |p| {
                    y[@as(usize, indices[t.li + p]) * N + l] -= lx[p * N + l] * v;
                };
            }
            var k: usize = t.n;
            while (k > 0) {
                k -= 1;
                const v = y[k * N + l] / diag[k * N + l];
                y[k * N + l] = v;
                if (v != 0) for (indices[t.up + k]..indices[t.up + k + 1]) |p| {
                    y[@as(usize, indices[t.ui + p]) * N + l] -= ux[p * N + l] * v;
                };
            }
            for (0..t.n) |j| out[@as(usize, indices[t.q + j]) * N + l] = y[j * N + l];
        }

        /// The complex lanes' A(ω) = G + jωC on the n x n source pattern,
        /// one source slot of one lane: `a` holds the real plane in rows
        /// 0..nnz and the imaginary plane in rows nnz..2nnz, the products
        /// of `freq_solve.ComplexFill`.
        pub fn complexFill(slot: usize, l: usize, n_lanes: usize, n_points: usize, nnz: usize, g: P(f64), c: P(f64), point: P(u32), omega: P(f64), a: P(f64)) void {
            const p = point[l];
            a[slot * n_lanes + l] = g[slot * n_points + p];
            a[(nnz + slot) * n_lanes + l] = omega[l] * c[slot * n_points + p];
        }

        /// `ComplexLaneLu.refactor` for lane l on the source-pattern tape
        /// `t` (`lk.buildTab` of the proxy factor): true where its mask
        /// bit would be set. Each buffer holds its real plane, then its
        /// imaginary one: `a` 2·nnz rows (`complexFill`), `lx` 2·nnz_l,
        /// `ux` 2·nnz_u, `d` (pivot reciprocals) and `w` 2n; `w` is zero
        /// on entry and on exit.
        pub fn complexRefactor(t: lk.LaneTab, x: P(u32), a: P(f64), lx: P(f64), ux: P(f64), d: P(f64), w: P(f64), n_lanes: u32, l: u32) bool {
            const N: usize = n_lanes;
            const L: usize = l;
            const n: usize = t.n;
            const ai = @as(usize, t.nnz) * N;
            const li = @as(usize, t.nnz_l) * N;
            const ui = @as(usize, t.nnz_u) * N;
            const wi = n * N;
            const inf = std.math.inf(f64);
            for (0..n) |k| if (x[t.flags + k] & lk.void_bit != 0) return true;
            var bad = false;
            var nonfinite: u64 = 0;
            for (0..n) |k| {
                const col = x[t.q + k];
                for (x[t.col_ptr + col]..x[t.col_ptr + col + 1]) |p| {
                    const r: usize = x[t.prow + p];
                    w[r * N + L] = a[p * N + L];
                    w[wi + r * N + L] = a[ai + p * N + L];
                }
                for (x[t.up + k]..x[t.up + k + 1]) |p| {
                    const i: usize = x[t.ui + p];
                    const xr = w[i * N + L];
                    const xi = w[wi + i * N + L];
                    w[i * N + L] = 0;
                    w[wi + i * N + L] = 0;
                    ux[p * N + L] = xr;
                    ux[ui + p * N + L] = xi;
                    nonfinite |= @as(u64, @bitCast(xr - xr)) | @as(u64, @bitCast(xi - xi));
                    for (x[t.lp + i]..x[t.lp + i + 1]) |pl| {
                        const r: usize = x[t.li + pl];
                        const vr = lx[pl * N + L];
                        const vi = lx[li + pl * N + L];
                        w[r * N + L] -= vr * xr - vi * xi;
                        w[wi + r * N + L] -= vr * xi + vi * xr;
                    }
                }
                var pr = w[k * N + L];
                var pi = w[wi + k * N + L];
                w[k * N + L] = 0;
                w[wi + k * N + L] = 0;
                const pmax = norm(pr, pi);
                const dead = pmax == 0 or !(pmax < inf);
                bad = bad or dead;
                if (dead) {
                    pr = 1;
                    pi = 0;
                }
                // 1/d by Smith's scaling, as the host lanes.
                const s = @max(@abs(pr), @abs(pi));
                const sa = pr / s;
                const sc = pi / s;
                const den = s * (sa * sa + sc * sc);
                const rr = sa / den;
                const ri = -sc / den;
                d[k * N + L] = rr;
                d[wi + k * N + L] = ri;
                const check = t.growth > 0 and x[t.flags + k] & lk.scaled_bit == 0;
                var cbits: i64 = @bitCast(if (dead) @as(f64, 0) else pmax);
                for (x[t.lp + k]..x[t.lp + k + 1]) |p| {
                    const r: usize = x[t.li + p];
                    const vr = w[r * N + L];
                    const vi = w[wi + r * N + L];
                    w[r * N + L] = 0;
                    w[wi + r * N + L] = 0;
                    cbits = @max(cbits, @as(i64, @bitCast(norm(vr, vi))));
                    const xr = vr * rr - vi * ri;
                    const xi = vr * ri + vi * rr;
                    lx[p * N + L] = xr;
                    lx[li + p * N + L] = xi;
                    nonfinite |= @as(u64, @bitCast(xr - xr)) | @as(u64, @bitCast(xi - xi));
                }
                if (check) bad = bad or !(pmax >= t.growth * @as(f64, @bitCast(cbits)));
            }
            return bad or nonfinite != 0;
        }

        /// `ComplexLaneLu.solve` (or `solveH` when `adjoint`) for lane l:
        /// `rhs` and `out` hold the stacked 2n vector (real rows, then
        /// imaginary), `y` 2n rows of scratch; buffers as `complexRefactor`.
        pub fn complexSolve(t: lk.LaneTab, x: P(u32), lx: P(f64), ux: P(f64), d: P(f64), rhs: P(f64), y: P(f64), out: P(f64), n_lanes: usize, l: usize, adjoint: bool) void {
            const N = n_lanes;
            const n: usize = t.n;
            const li = @as(usize, t.nnz_l) * N;
            const ui = @as(usize, t.nnz_u) * N;
            const im = n * N;
            if (adjoint) {
                for (0..n) |k| {
                    const c: usize = x[t.q + k];
                    y[k * N + l] = rhs[c * N + l];
                    y[im + k * N + l] = rhs[im + c * N + l];
                }
                for (0..n) |k| {
                    var sr = y[k * N + l];
                    var si = y[im + k * N + l];
                    for (x[t.up + k]..x[t.up + k + 1]) |p| {
                        const r: usize = x[t.ui + p];
                        const vr = ux[p * N + l];
                        const vi = ux[ui + p * N + l];
                        sr -= vr * y[r * N + l] + vi * y[im + r * N + l];
                        si -= vr * y[im + r * N + l] - vi * y[r * N + l];
                    }
                    const dr = d[k * N + l];
                    const di = d[im + k * N + l];
                    y[k * N + l] = sr * dr + si * di;
                    y[im + k * N + l] = si * dr - sr * di;
                }
                var k: usize = n;
                while (k > 0) {
                    k -= 1;
                    var sr = y[k * N + l];
                    var si = y[im + k * N + l];
                    for (x[t.lp + k]..x[t.lp + k + 1]) |p| {
                        const r: usize = x[t.li + p];
                        const vr = lx[p * N + l];
                        const vi = lx[li + p * N + l];
                        sr -= vr * y[r * N + l] + vi * y[im + r * N + l];
                        si -= vr * y[im + r * N + l] - vi * y[r * N + l];
                    }
                    y[k * N + l] = sr;
                    y[im + k * N + l] = si;
                }
                for (0..n) |r| {
                    const k2: usize = x[t.pinv + r];
                    out[r * N + l] = y[k2 * N + l];
                    out[im + r * N + l] = y[im + k2 * N + l];
                }
                return;
            }
            for (0..n) |r| {
                const k: usize = x[t.pinv + r];
                y[k * N + l] = rhs[r * N + l];
                y[im + k * N + l] = rhs[im + r * N + l];
            }
            for (0..n) |k| {
                const sr = y[k * N + l];
                const si = y[im + k * N + l];
                for (x[t.lp + k]..x[t.lp + k + 1]) |p| {
                    const r: usize = x[t.li + p];
                    const vr = lx[p * N + l];
                    const vi = lx[li + p * N + l];
                    y[r * N + l] -= vr * sr - vi * si;
                    y[im + r * N + l] -= vr * si + vi * sr;
                }
            }
            var k: usize = n;
            while (k > 0) {
                k -= 1;
                const yr = y[k * N + l];
                const yi = y[im + k * N + l];
                const dr = d[k * N + l];
                const di = d[im + k * N + l];
                const sr = yr * dr - yi * di;
                const si = yr * di + yi * dr;
                y[k * N + l] = sr;
                y[im + k * N + l] = si;
                for (x[t.up + k]..x[t.up + k + 1]) |p| {
                    const r: usize = x[t.ui + p];
                    const vr = ux[p * N + l];
                    const vi = ux[ui + p * N + l];
                    y[r * N + l] -= vr * sr - vi * si;
                    y[im + r * N + l] -= vr * si + vi * sr;
                }
            }
            for (0..n) |j| {
                const c: usize = x[t.q + j];
                out[c * N + l] = y[j * N + l];
                out[im + c * N + l] = y[im + j * N + l];
            }
        }

        /// max(|re|, |im|) by bit pattern, so a NaN in either part survives.
        inline fn norm(re: f64, im: f64) f64 {
            return @bitCast(@max(@as(i64, @bitCast(@abs(re))), @as(i64, @bitCast(@abs(im)))));
        }
    };
}

/// Maps each source slot to its four positions in the stacked CSC.
/// Caller supplies 4*nnz indices. Mirrors freq_solve.StackedPoints.
pub fn positions(src_col_ptr: []const u32, out: []u32) void {
    const n = src_col_ptr.len - 1;
    const nnz = src_col_ptr[n];
    std.debug.assert(out.len == 4 * nnz);
    for (0..n) |j| {
        const start = src_col_ptr[j];
        const len = src_col_ptr[j + 1] - start;
        for (start..src_col_ptr[j + 1]) |p| {
            const top: u32 = @intCast(2 * start + p - start);
            const right = 2 * nnz + top;
            out[4 * p ..][0..4].* = .{ top, top + len, right, right + len };
        }
    }
}

test "frequency fill keeps each point's planes and ragged frequency lanes independent" {
    const t = std.testing;
    const B = Bodies(lk.HostP);
    const N = 5; // deliberately not a SIMD or warp multiple
    const np = 2;
    const nnz = 4;
    var pos: [4 * nnz]u32 = undefined;
    positions(&.{ 0, 2, 4 }, &pos);
    var g = [_]f64{ 4, 9, -1, -2, -3, -4, 5, 7 };
    var c = [_]f64{ 0.01, 0.02, 0, 0.03, 0.04, 0, 0.05, 0.06 };
    var point = [_]u32{ 0, 1, 0, 1, 1 };
    var omega = [_]f64{ 0, 2, 100, -7, 1e6 };
    var a: [4 * nnz * N]f64 = @splat(std.math.nan(f64));
    for (0..nnz) |s| for (0..N) |l| B.fill(s, l, N, np, &pos, &g, &c, &point, &omega, &a);
    for (0..nnz) |s| for (0..N) |l| {
        const gv = g[s * np + point[l]];
        const cv = c[s * np + point[l]];
        const want = [_]f64{ gv, omega[l] * cv, -omega[l] * cv, gv };
        for (0..4) |q| try t.expectEqual(@as(u64, @bitCast(want[q])), @as(u64, @bitCast(a[pos[4 * s + q] * N + l])));
    };
}

test "frequency forward and adjoint lane bodies match LaneLu with mixed and zero right-hand sides" {
    const gpa = std.testing.allocator;
    const SparseLu = @import("../sparse_lu.zig").SparseLu;
    const LL = @import("../lane/lu.zig").LaneLu(4);
    const B = Bodies(lk.HostP);
    var prng = std.Random.DefaultPrng.init(0xf2e9a);
    const random = prng.random();
    var compared: usize = 0;
    for (0..24) |trial| {
        const n: u32 = @intCast(4 + trial);
        const sys = try SparseLu.test_access.TestSystem.init(gpa, random, n, 0.3);
        defer sys.deinit(gpa);
        const q = try gpa.alloc(u32, n);
        defer gpa.free(q);
        for (q, 0..) |*v, i| v.* = @intCast(i);
        random.shuffle(u32, q);
        var base = try SparseLu.init(gpa, n, sys.col_ptr, q);
        defer base.deinit(gpa);
        base.factor(gpa, sys.col_ptr, sys.row_idx, sys.vals, 1e-3) catch continue;
        if (std.mem.indexOfScalar(bool, base.void_col, true) != null) continue;
        var ll = try LL.init(gpa, &base);
        defer ll.deinit(gpa);
        const built = try lk.buildTab(gpa, &base, sys.col_ptr, 1e-12, 1e-3);
        defer gpa.free(built.idx);
        const vals = try gpa.alloc(LL.V, sys.vals.len);
        defer gpa.free(vals);
        for (vals, sys.vals) |*v, x| v.* = .{ x, x * 1.1, x * 0.9, x };
        const bad = ll.refactor(LL.Plane{ .col_ptr = sys.col_ptr, .vals = vals }, 1e-12);
        const work = try gpa.alloc(LL.V, 4 * n);
        defer gpa.free(work);
        const rhs = work[0..n];
        const want = work[n..][0..n];
        const got = work[2 * n ..][0..n];
        const y = work[3 * n ..][0..n];
        for (rhs) |*v| v.* = .{ 0, random.float(f64) - 0.5, random.float(f64), -1 };
        for ([_]bool{ false, true }) |adjoint| {
            if (adjoint) ll.solveT(rhs, want) else ll.solve(rhs, want);
            for (0..4) |l| {
                if (bad & (@as(u64, 1) << @intCast(l)) != 0) continue;
                B.solve(built.tab, built.idx.ptr, @ptrCast(ll.lx.ptr), @ptrCast(ll.ux.ptr), @ptrCast(ll.udiag.ptr), @ptrCast(rhs.ptr), @ptrCast(y.ptr), @ptrCast(got.ptr), 4, l, adjoint);
                for (want, got) |wv, gv| {
                    const wa: [4]f64 = wv;
                    const ga: [4]f64 = gv;
                    try std.testing.expectEqual(@as(u64, @bitCast(wa[l])), @as(u64, @bitCast(ga[l])));
                }
                compared += 1;
            }
        }
    }
    try std.testing.expect(compared > 0);
}

test "complex lane bodies are bitwise ComplexLaneLu's lanes: fill, refactor mask, forward and adjoint solves" {
    const gpa = std.testing.allocator;
    const SparseLu = @import("../sparse_lu.zig").SparseLu;
    const W = 4;
    const CL = @import("../lane/clu.zig").ComplexLaneLu(W);
    const B = Bodies(lk.HostP);
    var prng = std.Random.DefaultPrng.init(0xc0f1);
    const random = prng.random();
    var compared: usize = 0;
    for (0..32) |trial| {
        const n: u32 = @intCast(3 + trial);
        const sys = try SparseLu.test_access.TestSystem.init(gpa, random, n, 0.3);
        defer sys.deinit(gpa);
        const nnz = sys.vals.len;
        const q = try gpa.alloc(u32, n);
        defer gpa.free(q);
        for (q, 0..) |*v, i| v.* = @intCast(i);
        random.shuffle(u32, q);
        var base = try SparseLu.init(gpa, n, sys.col_ptr, q);
        defer base.deinit(gpa);
        // G and C per point; lane l reads point l % 2 at its own ω.
        const g = try gpa.alloc(f64, 2 * nnz);
        defer gpa.free(g);
        const c = try gpa.alloc(f64, 2 * nnz);
        defer gpa.free(c);
        for (0..nnz) |s| for (0..2) |p| {
            g[s * 2 + p] = sys.vals[s] * (1 + 0.1 * @as(f64, @floatFromInt(p)));
            c[s * 2 + p] = (2 * random.float(f64) - 1) * @abs(sys.vals[s]);
        };
        var point = [W]u32{ 0, 1, 0, 1 };
        var omega = [W]f64{ 0.5, 1, 2, 1e3 };
        const proxy = try gpa.alloc(f64, nnz);
        defer gpa.free(proxy);
        for (proxy, 0..) |*v, s| v.* = std.math.hypot(g[s * 2], omega[0] * c[s * 2]);
        base.factor(gpa, sys.col_ptr, sys.row_idx, proxy, 1e-3) catch continue;
        if (std.mem.indexOfScalar(bool, base.void_col, true) != null) continue;
        var ll = try CL.init(gpa, &base);
        defer ll.deinit(gpa);
        const re = try gpa.alloc(CL.V, nnz);
        defer gpa.free(re);
        const im = try gpa.alloc(CL.V, nnz);
        defer gpa.free(im);
        const a = try gpa.alloc(f64, 2 * nnz * W);
        defer gpa.free(a);
        for (0..nnz) |s| for (0..W) |l| B.complexFill(s, l, W, 2, nnz, g.ptr, c.ptr, &point, &omega, a.ptr);
        for (re, im, 0..) |*vr, *vi, s| {
            var ar: [W]f64 = undefined;
            var ai: [W]f64 = undefined;
            for (0..W) |l| {
                ar[l] = g[s * 2 + point[l]];
                ai[l] = omega[l] * c[s * 2 + point[l]];
                try std.testing.expectEqual(@as(u64, @bitCast(ar[l])), @as(u64, @bitCast(a[s * W + l])));
                try std.testing.expectEqual(@as(u64, @bitCast(ai[l])), @as(u64, @bitCast(a[(nnz + s) * W + l])));
            }
            vr.* = ar;
            vi.* = ai;
        }
        const growth = 1e-3;
        const bad = ll.refactor(CL.Plane{ .col_ptr = sys.col_ptr, .re = re, .im = im }, growth);
        var built = try lk.buildTab(gpa, &base, sys.col_ptr, growth, 1e-3);
        defer gpa.free(built.idx);
        built.tab.wide = std.math.inf(f64);
        const nl = base.lx.items.len;
        const nu = base.ux.items.len;
        const buf = try gpa.alloc(f64, (2 * nl + 2 * nu + 2 * n + 2 * n + 2 * n + 2 * n + 2 * n) * W);
        defer gpa.free(buf);
        const lx = buf[0 .. 2 * nl * W];
        const ux = buf[2 * nl * W ..][0 .. 2 * nu * W];
        const d = buf[2 * (nl + nu) * W ..][0 .. 2 * n * W];
        const w = buf[2 * (nl + nu + n) * W ..][0 .. 2 * n * W];
        const rhs = buf[2 * (nl + nu + 2 * n) * W ..][0 .. 2 * n * W];
        const y = buf[2 * (nl + nu + 3 * n) * W ..][0 .. 2 * n * W];
        const out = buf[2 * (nl + nu + 4 * n) * W ..][0 .. 2 * n * W];
        @memset(w, 0);
        const work = try gpa.alloc(CL.V, 4 * n);
        defer gpa.free(work);
        const br = work[0..n];
        const bi = work[n..][0..n];
        const xr = work[2 * n ..][0..n];
        const xi = work[3 * n ..][0..n];
        for (0..n) |i| {
            var vr: [W]f64 = undefined;
            var vi: [W]f64 = undefined;
            for (0..W) |l| {
                vr[l] = if (l == 1) 0 else 2 * random.float(f64) - 1;
                vi[l] = 2 * random.float(f64) - 1;
                rhs[i * W + l] = vr[l];
                rhs[(n + i) * W + l] = vi[l];
            }
            br[i] = vr;
            bi[i] = vi;
        }
        for (0..W) |l| {
            const lane_bad = B.complexRefactor(built.tab, built.idx.ptr, a.ptr, lx.ptr, ux.ptr, d.ptr, w.ptr, W, @intCast(l));
            try std.testing.expectEqual((bad >> @intCast(l)) & 1 != 0, lane_bad);
        }
        for (w) |v| try std.testing.expectEqual(@as(f64, 0), v);
        for ([_]bool{ false, true }) |adjoint| {
            if (adjoint) ll.solveH(br, bi, xr, xi) else ll.solve(br, bi, xr, xi);
            for (0..W) |l| {
                if ((bad >> @intCast(l)) & 1 != 0) continue;
                B.complexSolve(built.tab, built.idx.ptr, lx.ptr, ux.ptr, d.ptr, rhs.ptr, y.ptr, out.ptr, W, l, adjoint);
                for (0..n) |i| {
                    const wr: [W]f64 = xr[i];
                    const wi: [W]f64 = xi[i];
                    try std.testing.expectEqual(@as(u64, @bitCast(wr[l])), @as(u64, @bitCast(out[i * W + l])));
                    try std.testing.expectEqual(@as(u64, @bitCast(wi[l])), @as(u64, @bitCast(out[(n + i) * W + l])));
                }
                compared += 1;
            }
        }
    }
    try std.testing.expect(compared > 40);
}
