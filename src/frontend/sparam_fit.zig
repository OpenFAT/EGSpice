//! Vector fitting. In: admittance samples (`Network`). Out: a rational
//! model with poles common to every entry (`Fit`), and its real
//! second-order sections (`Sections`), which models/sparam_N.va and
//! wline_N.va run through `laplace_nd`. HSPICE's RATIONAL_FUNC=1; its
//! default is IFFT convolution.
//! Cites: Gustavsen and Semlyen, IEEE Trans. Power Delivery 14(3), 1999;
//! Deschrijver et al., IEEE MWCL 18(6), 2008 (fast column-wise pole
//! identification).
//!
//! Contents:
//! - `pub const Fit = struct`
//! - `pub fn eval(self: Fit, e: usize, omega: f64) Cx`
//! - `pub const Sections = struct`
//! - `pub const FitOptions = struct`
//! - `pub fn fit(arena: Allocator, net: Network, opt: FitOptions) Error!Fit`
//! - `pub fn sections(arena: Allocator, f: Fit) Error!Sections`
//! - `pub fn fillBasis(poles: []const Cx, s: []const f64, a: []f64) void`
//! - `pub fn realOrder(poles: []const Cx) usize`
//! - `pub fn phi(p: Cx, w: f64) [2]Cx`

const std = @import("std");
const assert = std.debug.assert;
const testing = std.testing;
const Allocator = std.mem.Allocator;
const core = @import("core");
const eigen = core.eigen;
const numerics = core.numerics;
const sparam_touchstone = @import("sparam_touchstone.zig");
const sparam_lsq = @import("sparam_lsq.zig");
const Network = sparam_touchstone.Network;
const Error = sparam_touchstone.Error;
/// Y(s) ≈ d_e + Σ_k r_ek/(s − p_k) (+ the conjugate term for a complex
/// pole), common poles. A complex pole appears once, with Im p > 0.
pub const Fit = struct {
    /// rad/s, all with Re p < 0 (a pole is reflected into the left half
    /// plane when relocation lands it right of the axis).
    poles: []const Cx,
    /// `res[e * poles.len + k]`; real for a real pole.
    res: []Cx,
    /// Direct term per entry, `d[e]`.
    d: []f64,
    /// RMS error over the data relative to max |Y|.
    rel_err: f64 = 0,

    /// The fitted response of entry `e` at s = jω.
    pub fn eval(self: Fit, e: usize, omega: f64) Cx {
        const s: Cx = .{ .re = 0, .im = omega };
        var y: Cx = .{ .re = self.d[e], .im = 0 };
        const kk = self.poles.len;
        for (self.poles, self.res[e * kk ..][0..kk]) |p, r| {
            y = y.add(r.div(s.sub(p)));
            if (p.im != 0) y = y.add(r.conjugate().div(s.sub(p.conjugate())));
        }
        return y;
    }
};

/// A fit as real sections Σ_k (b1·s + b0)/(a2·s² + a1·s + a0): each
/// complex pair, or two real poles, is one second-order section (one odd
/// real pole is first order). Poles are never multiplied into one
/// polynomial, whose coefficients would be ill-conditioned.
pub const Sections = struct {
    /// Section k's denominator `.{ a0, a1, a2 }`, shared by every entry.
    den: []const [3]f64,
    /// Entry e's numerator for section k, `.{ b0, b1 }` at `num[e * den.len + k]`.
    num: []const [2]f64,
};

/// The fit's options: its accuracy target and pole budget.
pub const FitOptions = struct {
    /// Target RMS error relative to max |Y| (to |Y| itself with `relative`).
    tol: f64 = 1e-6,
    /// Largest real order tried (a pair counts two). 32 fills the 16
    /// second-order sections of models/sparam_1.va and wline_1.va.
    max_order: usize = 32,
    /// Pole-relocation passes per order.
    iterations: usize = 12,
    /// Weight each sample by 1/|Y| instead of each entry by 1/max |Y|, for
    /// a response that spans decades (a line's Yc).
    relative: bool = false,
};

const Cx = std.math.Complex(f64);

/// The rational fit of `net` over its data frequencies, orders 4, 8, ...
/// until `opt.tol` or `opt.max_order`, whichever comes first; the best one
/// tried is returned. An order is tried only below the frequency count,
/// so a short file gets the fit its data can determine.
/// Fails with InvalidTouchstone when `net` has no frequency above 0.
/// Allocates every table of the result, and its scratch, in `arena`.
pub fn fit(arena: Allocator, net: Network, opt: FitOptions) Error!Fit {
    if (net.freq.len == 0) return error.InvalidTouchstone;
    const w_max = 2 * std.math.pi * net.freq[net.freq.len - 1];
    if (!(w_max > 0)) return error.InvalidTouchstone;
    const weight = try weights(arena, net, opt.relative);
    const s = try arena.alloc(f64, net.freq.len);
    for (s, net.freq) |*o, f| o.* = 2 * std.math.pi * f / w_max;
    // Order 0 first: a constant (a lossless line's Yc) needs no poles.
    var best = try residues(arena, net, &.{}, s, w_max, weight);
    var n: usize = 4;
    // Order K's σ block is rows K+1..2K of a 2·nf-row QR, so K < nf.
    while (best.rel_err > opt.tol and n <= opt.max_order and n < net.freq.len) : (n += 4) {
        const f = try fitOrder(arena, net, n / 2, opt.iterations, s, w_max, weight);
        if (f.rel_err < best.rel_err) best = f;
    }
    return best;
}

/// `f`'s poles and residues as `Sections`; allocates both tables in `arena`.
pub fn sections(arena: Allocator, f: Fit) Error!Sections {
    const kk = f.poles.len;
    const ne = f.d.len;
    var reals: std.ArrayList(usize) = .empty;
    var pairs: std.ArrayList(usize) = .empty;
    for (f.poles, 0..) |p, k| try (if (p.im != 0) &pairs else &reals).append(arena, k);
    const ns = pairs.items.len + (reals.items.len + 1) / 2;
    const den = try arena.alloc([3]f64, ns);
    const num = try arena.alloc([2]f64, ne * ns);
    for (pairs.items, 0..) |k, sec| {
        const p = f.poles[k];
        den[sec] = .{ p.re * p.re + p.im * p.im, -2 * p.re, 1 };
        // r/(s − p) + r̄/(s − p̄) = (2 Re r·s − 2 Re(r·p̄)) / (s² − 2 Re p·s + |p|²).
        for (0..ne) |e| {
            const r = f.res[e * kk + k];
            num[e * ns + sec] = .{ -2 * (r.re * p.re + r.im * p.im), 2 * r.re };
        }
    }
    var sec = pairs.items.len;
    var at: usize = 0;
    while (at < reals.items.len) : ({
        at += 2;
        sec += 1;
    }) {
        const k1 = reals.items[at];
        const p1 = f.poles[k1].re;
        if (at + 1 == reals.items.len) {
            den[sec] = .{ -p1, 1, 0 };
            for (0..ne) |e| num[e * ns + sec] = .{ f.res[e * kk + k1].re, 0 };
            break;
        }
        const k2 = reals.items[at + 1];
        const p2 = f.poles[k2].re;
        den[sec] = .{ p1 * p2, -(p1 + p2), 1 };
        // r1/(s − p1) + r2/(s − p2) = ((r1 + r2)·s − (r1·p2 + r2·p1)) / ((s − p1)(s − p2)).
        for (0..ne) |e| {
            const r1 = f.res[e * kk + k1].re;
            const r2 = f.res[e * kk + k2].re;
            num[e * ns + sec] = .{ -(r1 * p2 + r2 * p1), r1 + r2 };
        }
    }
    return .{ .den = den, .num = num };
}

/// Column-major real LS matrix [basis | 1] over the scaled frequencies,
/// 2·s.len rows (re, im per frequency).
pub fn fillBasis(poles: []const Cx, s: []const f64, a: []f64) void {
    const m = 2 * s.len;
    std.debug.assert(a.len == m * (realOrder(poles) + 1));
    for (s, 0..) |sk, k| {
        var col: usize = 0;
        for (poles) |p| {
            const bv = phi(p, sk);
            for (bv[0..if (p.im != 0) 2 else 1]) |v| {
                a[col * m + 2 * k] = v.re;
                a[col * m + 2 * k + 1] = v.im;
                col += 1;
            }
        }
        a[col * m + 2 * k] = 1;
        a[col * m + 2 * k + 1] = 0;
    }
}

/// Real state count of the realization: one per real pole, two per pair.
pub fn realOrder(poles: []const Cx) usize {
    var n: usize = 0;
    for (poles) |p| n += if (p.im != 0) 2 else 1;
    return n;
}

/// The real-coefficient basis of pole `p` at s = j·w: 1/(s − p) for a real
/// pole; 1/(s − p) + 1/(s − p*) and j/(s − p) − j/(s − p*) for a pair.
pub fn phi(p: Cx, w: f64) [2]Cx {
    const s: Cx = .{ .re = 0, .im = w };
    const one: Cx = .{ .re = 1, .im = 0 };
    const a = one.div(s.sub(p));
    if (p.im == 0) return .{ a, .{ .re = 0, .im = 0 } };
    const b = one.div(s.sub(p.conjugate()));
    const j: Cx = .{ .re = 0, .im = 1 };
    return .{ a.add(b), j.mul(a.sub(b)) };
}

// ---- private ----

/// Per-sample LS weights, `[e * freq.len + k]`: 1/max_k |Y_e| per entry, or
/// 1/|Y_e(k)| with `relative` (0 for a zero sample or entry).
fn weights(arena: Allocator, net: Network, relative: bool) Error![]f64 {
    const nf = net.freq.len;
    const w = try arena.alloc(f64, net.ports * net.ports * nf);
    for (0..net.ports * net.ports) |e| {
        var m: f64 = 0;
        for (0..nf) |k| m = @max(m, net.at(e, k).magnitude());
        for (0..nf) |k| {
            const a = if (relative) net.at(e, k).magnitude() else m;
            w[e * nf + k] = if (a > 0) 1 / a else 0;
        }
    }
    return w;
}

/// Vector fitting with `pairs` starting complex pairs, at `s`, the data
/// frequencies in rad/s scaled by `w0` so the basis is O(1); `weight` as
/// `weights` returns it.
fn fitOrder(arena: Allocator, net: Network, pairs: usize, iterations: usize, s: []const f64, w0: f64, weight: []const f64) Error!Fit {
    const nf = net.freq.len;
    const ne = net.ports * net.ports;
    var poles = try arena.alloc(Cx, pairs);
    // Starting poles: complex pairs log-spaced over the band, lightly damped.
    var f_min = net.freq[nf - 1];
    for (net.freq) |f| if (f > 0) {
        f_min = @min(f_min, f);
    };
    const lo = @max(f_min * 2 * std.math.pi / w0, 1e-12);
    for (poles, 0..) |*p, i| {
        const t = if (pairs == 1) 1.0 else @as(f64, @floatFromInt(i)) / @as(f64, @floatFromInt(pairs - 1));
        const b = lo * std.math.pow(f64, 1 / lo, t);
        p.* = .{ .re = -b / 100, .im = b };
    }
    var n_used: usize = 0;
    for (0..ne) |e| n_used += @intFromBool(!std.mem.allEqual(f64, weight[e * nf ..][0..nf], 0));
    if (n_used == 0) return error.InvalidTouchstone;

    var it: usize = 0;
    while (it < iterations) : (it += 1) {
        const kk = realOrder(poles);
        // Per entry: QR of [Φ 1 | −HΦ], keep the σ block (rows past K+1).
        const rows = 2 * nf;
        const cols = 2 * kk + 1;
        const a = try arena.alloc(f64, rows * cols);
        const rhs = try arena.alloc(f64, rows);
        // The σ blocks stacked, n_used·K rows by K columns.
        const sr = n_used * kk;
        const stack = try arena.alloc(f64, sr * kk);
        const stack_b = try arena.alloc(f64, sr);
        var used: usize = 0;
        for (0..ne) |e| {
            const we = weight[e * nf ..][0..nf];
            if (std.mem.allEqual(f64, we, 0)) continue;
            for (0..nf) |k| {
                const h = net.at(e, k).mul(.{ .re = we[k], .im = 0 });
                var col: usize = 0;
                for (poles) |p| {
                    const basis = phi(p, s[k]);
                    for (basis[0..if (p.im != 0) 2 else 1]) |bv| {
                        a[col * rows + 2 * k] = bv.re * we[k];
                        a[col * rows + 2 * k + 1] = bv.im * we[k];
                        const hb = h.mul(bv).neg();
                        a[(kk + 1 + col) * rows + 2 * k] = hb.re;
                        a[(kk + 1 + col) * rows + 2 * k + 1] = hb.im;
                        col += 1;
                    }
                }
                a[kk * rows + 2 * k] = we[k];
                a[kk * rows + 2 * k + 1] = 0;
                rhs[2 * k] = h.re;
                rhs[2 * k + 1] = h.im;
            }
            sparam_lsq.householder(rows, cols, a, rhs);
            for (0..kk) |c| @memcpy(stack[c * sr + used * kk ..][0..kk], a[(kk + 1 + c) * rows + kk + 1 ..][0..kk]);
            @memcpy(stack_b[used * kk ..][0..kk], rhs[kk + 1 ..][0..kk]);
            used += 1;
        }
        const ct = try arena.alloc(f64, kk);
        sparam_lsq.leastSquares(sr, kk, stack, stack_b, ct);
        // New poles: the zeros of σ = eig(A − b·c̃ᵀ).
        const h = try arena.alloc(f64, kk * kk);
        @memset(h, 0);
        var at: usize = 0;
        for (poles) |p| {
            if (p.im != 0) {
                h[at * kk + at] = p.re;
                h[at * kk + at + 1] = p.im;
                h[(at + 1) * kk + at] = -p.im;
                h[(at + 1) * kk + at + 1] = p.re;
                for (0..kk) |c| h[at * kk + c] -= 2 * ct[c];
                at += 2;
            } else {
                h[at * kk + at] = p.re;
                for (0..kk) |c| h[at * kk + c] -= ct[c];
                at += 1;
            }
        }
        const ev = try arena.alloc(numerics.Complex, kk);
        const got = eigen.eigenvalues(kk, h, ev, 1e-14, 60 * @as(u32, @intCast(kk)));
        if (!got.converged or got.count != kk) break;
        poles = try arena.alloc(Cx, kk);
        var n: usize = 0;
        for (ev[0..got.count]) |z| {
            if (z.im < 0) continue; // the conjugate of a listed pole
            // Stable, and not so close to the axis that it rings forever.
            var p: Cx = .{ .re = -@abs(z.re), .im = z.im };
            if (p.re == 0) p.re = -1e-6 * @max(lo, @abs(p.im));
            poles[n] = p;
            n += 1;
        }
        poles = poles[0..n];
    }
    return residues(arena, net, poles, s, w0, weight);
}

/// The residues and direct terms for fixed `poles` (scaled by `w0`), by
/// weighted least squares over the data, then scaled back to rad/s.
/// `rel_err` is the weighted RMS error.
fn residues(arena: Allocator, net: Network, poles_s: []const Cx, s: []const f64, w0: f64, weight: []const f64) Error!Fit {
    const nf = net.freq.len;
    const ne = net.ports * net.ports;
    const kk = realOrder(poles_s);
    const rows = 2 * nf;
    const cols = kk + 1;
    const basis = try arena.alloc(f64, rows * cols);
    fillBasis(poles_s, s, basis);
    const res = try arena.alloc(Cx, ne * poles_s.len);
    const d = try arena.alloc(f64, ne);
    const a = try arena.alloc(f64, rows * cols);
    const b = try arena.alloc(f64, rows);
    const x = try arena.alloc(f64, cols);
    // ponytail: one QR per entry. Without `relative` every entry's weight is
    // one constant, so a single QR of the basis would serve them all.
    for (0..ne) |e| {
        const we = weight[e * nf ..][0..nf];
        for (0..cols) |c| for (0..nf) |k| {
            a[c * rows + 2 * k] = basis[c * rows + 2 * k] * we[k];
            a[c * rows + 2 * k + 1] = basis[c * rows + 2 * k + 1] * we[k];
        };
        for (0..nf) |k| {
            b[2 * k] = net.at(e, k).re * we[k];
            b[2 * k + 1] = net.at(e, k).im * we[k];
        }
        sparam_lsq.leastSquares(rows, cols, a, b, x);
        unpack(poles_s, x[0..kk], res[e * poles_s.len ..][0..poles_s.len]);
        d[e] = x[kk];
    }
    const poles = try arena.alloc(Cx, poles_s.len);
    for (poles, poles_s) |*p, q| p.* = q.mul(.{ .re = w0, .im = 0 });
    for (res) |*r| r.* = r.mul(.{ .re = w0, .im = 0 });
    var out: Fit = .{ .poles = poles, .res = res, .d = d };
    var err2: f64 = 0;
    for (0..ne) |e| for (0..nf) |k| {
        const err = out.eval(e, 2 * std.math.pi * net.freq[k]).sub(net.at(e, k)).magnitude() * weight[e * nf + k];
        err2 += err * err;
    };
    out.rel_err = @sqrt(err2 / @as(f64, @floatFromInt(ne * nf)));
    return out;
}

/// Real basis coefficients to complex residues: a pair's (c1, c2) is
/// c1 + j·c2 at the pole with Im p > 0.
fn unpack(poles: []const Cx, x: []const f64, res: []Cx) void {
    var at: usize = 0;
    for (poles, res) |p, *r| {
        if (p.im != 0) {
            r.* = .{ .re = x[at], .im = x[at + 1] };
            at += 2;
        } else {
            r.* = .{ .re = x[at], .im = 0 };
            at += 1;
        }
    }
}

test sections {
    // Three real poles (one pair of sections and one first-order) and a
    // complex pair: the sections sum to the pole-residue form at any ω.
    var arena_state = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var res = [_]Cx{ .{ .re = 1, .im = 0 }, .{ .re = 2, .im = 0 }, .{ .re = -0.5, .im = 0.25 }, .{ .re = 3, .im = 0 } };
    var d = [_]f64{0.1};
    const f: Fit = .{
        .poles = &.{ .{ .re = -1, .im = 0 }, .{ .re = -2, .im = 0 }, .{ .re = -1, .im = 2 }, .{ .re = -3, .im = 0 } },
        .res = &res,
        .d = &d,
    };
    try testing.expectEqual(@as(usize, 5), realOrder(f.poles));
    const sec = try sections(arena, f);
    try testing.expectEqual(@as(usize, 3), sec.den.len);
    for ([_]f64{ 0, 0.3, 1, 7 }) |w| {
        const s: Cx = .{ .re = 0, .im = w };
        var y: Cx = .{ .re = f.d[0], .im = 0 };
        for (sec.num, sec.den) |nm, dn| {
            const num: Cx = .{ .re = nm[0], .im = w * nm[1] };
            const den = (Cx{ .re = dn[0], .im = w * dn[1] }).add(s.mul(s).mul(.{ .re = dn[2], .im = 0 }));
            y = y.add(num.div(den));
        }
        try testing.expect(y.sub(f.eval(0, w)).magnitude() <= 1e-12 * f.eval(0, w).magnitude());
    }
}

test fit {
    var arena_state = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    try testing.expectError(error.InvalidTouchstone, fit(arena, .{ .ports = 1, .freq = &.{}, .re = &.{}, .im = &.{} }, .{}));
    try testing.expectError(error.InvalidTouchstone, fit(arena, .{ .ports = 1, .freq = &.{0}, .re = &.{1}, .im = &.{0} }, .{}));
    // Four points of a one-pole response: no order-4 fit fits in four
    // points, so the constant fit comes back instead of a σ block read
    // past the data.
    var freq: [4]f64 = undefined;
    var re: [4]f64 = undefined;
    var im: [4]f64 = undefined;
    for (&freq, &re, &im, 0..) |*f, *r, *i, k| {
        f.* = @floatFromInt(k + 1);
        const y = (Cx{ .re = 1, .im = 0 }).div(.{ .re = 1, .im = f.* });
        r.* = y.re;
        i.* = y.im;
    }
    const short = try fit(arena, .{ .ports = 1, .freq = &freq, .re = &re, .im = &im }, .{});
    try testing.expectEqual(@as(usize, 0), short.poles.len);
    try testing.expect(short.rel_err > 0);
}

test "weights: per entry, or per sample with `relative`, zero for a zero sample" {
    var arena_state = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const net: Network = .{ .ports = 1, .freq = &.{ 1, 2 }, .re = &.{ 0, 2 }, .im = &.{ 0, 0 } };
    try testing.expectEqualSlices(f64, &.{ 0.5, 0.5 }, try weights(arena, net, false));
    try testing.expectEqualSlices(f64, &.{ 0, 0.5 }, try weights(arena, net, true));
}
