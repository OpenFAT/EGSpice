//! The IDFT/DFT pair between a spectrum block and its time samples: a
//! `Spectrum` and its tones in, the sample instants and both dense
//! matrices (or, for one tone, their symmetry fold) out. One tone samples a
//! uniform period 2x oversampled, as `hb.zig` does; several tones take
//! 2m-1 collocation instants picked greedily from an oversampled pool for
//! the best-conditioned transform, and invert it (the APFT).
//! Cites: VACASK `lib/corehbcoloc.cpp`, after Kundert, White and
//! Sangiovanni-Vincentelli, 1990, App. B.
//!
//! Contents:
//! - `pub const Transform = struct`
//! - `pub const Fold = struct`
//! - `Fold`: `init`, `deinit`, `inverse`, `forward`
//! - `pub const pool_factor = 5`
//! - `pub fn transform(spec: Spectrum, tones: []const f64, buf: []f64, allocator: std.mem.Allocator) !Transform`
//! - `pub fn transformSamples(spec: Spectrum) usize`
//! - `pub fn basisRow(spec: Spectrum, tones: []const f64, t: f64, row: []f64) void`
//! - `pub fn idft(tr: Transform, nf: usize, x_hat: []const f64, x_td: []f64) void`
//! - `pub fn dft(tr: Transform, nf: usize, x_td: []const f64, x_hat: []f64) void`

const std = @import("std");
const root = @import("../types.zig");
const take = root.take;
const pss = @import("pss.zig");
const num = @import("core").numerics;
const solvers = @import("solver");
const converger = solvers.converger;
const dense_lu = solvers.dense_lu;
const FreqSolver = solvers.freq_solve.FreqSolver;
const Gmres = solvers.gmres.Gmres;
const Gcrodr = solvers.gcrodr.Gcrodr;
const SolveResult = pss.SolveResult;
const mhb_spectrum = @import("mhb_spectrum.zig");
const Spectrum = mhb_spectrum.Spectrum;
const max_tones = mhb_spectrum.max_tones;

/// The IDFT/DFT pair between a spectrum block (nf = 2m-1 reals) and nt
/// time samples: x(t_s) = Σ_c inv[s*nf + c]·X_c and
/// X_c = Σ_s fwd[c*nt + s]·x(t_s). Rows are contiguous for `num.dot`.
pub const Transform = struct {
    nt: usize,
    times: []f64,
    inv: []f64,
    fwd: []f64,
    /// One tone: the same pair folded by symmetry (`Fold`), which `idft`
    /// and `dft` take instead of `inv` and `fwd`. Owned by the `Newton`
    /// that set it; the device operator keeps the dense pair.
    fold: ?*const Fold = null,
};

/// The one-tone transform pair at a quarter of its multiply-adds. On the
/// uniform grid nt = 2·nf, so H = nt/2 is an integer and harmonic h obeys
///   x(s + H) = Σ_even - Σ_odd       (a half-period shift flips odd h)
///   x(H - s): cos even, sin odd for even h; cos odd, sin even for odd h
/// so every sample comes from the m bases s = 0..m-1 (H = 2m - 1), each
/// split into four partial sums: even/odd lines times cos/sin. `dft` folds
/// the samples the same way before m-long dots. Values match the dense pair
/// up to summation order (the angles are the same table entries).
pub const Fold = struct {
    /// Base samples: 0..m-1.
    m: usize,
    /// Coefficient indices in gather order: even-line cos (DC first),
    /// even-line sin, odd-line cos, odd-line sin; group ends in `ends`.
    perm: []u32,
    ends: [4]usize,
    /// m rows of nf: row s holds the gathered basis values at sample s.
    inv: []f64,
    /// nf rows of m, natural coefficient order: the projection weights
    /// over the folded samples of group `group[c]`.
    fwd: []f64,
    group: []u8,
    /// Scratch: the gathered coefficients (nf), the half-period sums and
    /// differences (2·H) and the four folded sample arrays (4·m).
    work: []f64,

    /// Folds the dense one-tone pair `tr` of `spec`. Asserts one tone on the
    /// uniform nt = 2·nf grid. Free with `deinit` on `gpa`.
    pub fn init(gpa: std.mem.Allocator, spec: Spectrum, tr: Transform) !*Fold {
        const m = spec.freqs.len;
        const nf = 2 * m - 1;
        const nt = tr.nt;
        std.debug.assert(spec.tones == 1 and nt == 2 * nf);
        const h_half = nt / 2;
        const self = try gpa.create(Fold);
        errdefer gpa.destroy(self);
        const perm = try gpa.alloc(u32, nf);
        errdefer gpa.free(perm);
        const group = try gpa.alloc(u8, nf);
        errdefer gpa.free(group);
        const vals = try gpa.alloc(f64, 2 * m * nf + nf + 2 * h_half + 4 * m);
        // Group of coefficient c: 0 even cos, 1 even sin, 2 odd cos, 3 odd sin.
        group[0] = 0;
        for (1..m) |j| {
            const odd: u8 = @intCast(spec.mix[j] & 1);
            group[2 * j - 1] = 2 * odd;
            group[2 * j] = 2 * odd + 1;
        }
        var at: usize = 0;
        var ends: [4]usize = undefined;
        for (0..4) |g| {
            for (group, 0..) |gc, c| if (gc == g) {
                perm[at] = @intCast(c);
                at += 1;
            };
            ends[g] = at;
        }
        const inv = vals[0 .. m * nf];
        const fwd = vals[m * nf ..][0 .. nf * m];
        // The dense pair's own entries at the base samples.
        for (0..m) |sb| for (perm, 0..) |c, k| {
            inv[sb * nf + k] = tr.inv[sb * nf + c];
        };
        for (0..nf) |c| for (0..m) |sb| {
            fwd[c * m + sb] = tr.fwd[c * nt + sb];
        };
        self.* = .{ .m = m, .perm = perm, .ends = ends, .inv = inv, .fwd = fwd, .group = group, .work = vals[2 * m * nf ..] };
        return self;
    }

    /// Frees the fold and every table `init` allocated on `gpa`.
    pub fn deinit(self: *const Fold, gpa: std.mem.Allocator) void {
        const nf = self.perm.len;
        gpa.free(self.inv.ptr[0 .. 2 * self.m * nf + self.work.len]);
        gpa.free(self.perm);
        gpa.free(self.group);
        gpa.destroy(self);
    }

    /// `idft` of one node: x (nt) from its coefficient block (nf).
    pub fn inverse(f: *const Fold, coef: []const f64, x: []f64) void {
        const nf = f.perm.len;
        const m = f.m;
        const hh = 2 * m - 1;
        const g = f.work[0..nf];
        for (g, f.perm) |*v, c| v.* = coef[c];
        const e = f.ends;
        for (0..m) |sb| {
            const row = f.inv[sb * nf ..][0..nf];
            const pc = num.dot(g[0..e[0]], row[0..e[0]]);
            const ps = num.dot(g[e[0]..e[1]], row[e[0]..e[1]]);
            const qc = num.dot(g[e[1]..e[2]], row[e[1]..e[2]]);
            const qs = num.dot(g[e[2]..e[3]], row[e[2]..e[3]]);
            const p = pc + ps;
            const q = qc + qs;
            x[sb] = p + q;
            x[sb + hh] = p - q;
            if (sb == 0) continue;
            const p2 = pc - ps;
            const q2 = qs - qc;
            x[hh - sb] = p2 + q2;
            x[2 * hh - sb] = p2 - q2;
        }
    }

    /// `dft` of one node: its coefficient block (nf) from x (nt).
    pub fn forward(f: *const Fold, x: []const f64, coef: []f64) void {
        const nf = f.perm.len;
        const m = f.m;
        const hh = 2 * m - 1;
        const u = f.work[nf..][0..hh];
        const v = f.work[nf + hh ..][0..hh];
        const folded = f.work[nf + 2 * hh ..][0 .. 4 * m];
        for (u, v, x[0..hh], x[hh..][0..hh]) |*a, *b, lo, hi| {
            a.* = lo + hi;
            b.* = lo - hi;
        }
        const ec = folded[0..m];
        const es = folded[m..][0..m];
        const oc = folded[2 * m ..][0..m];
        const os = folded[3 * m ..][0..m];
        ec[0] = u[0];
        es[0] = 0;
        oc[0] = v[0];
        os[0] = 0;
        for (1..m) |sb| {
            ec[sb] = u[sb] + u[hh - sb];
            es[sb] = u[sb] - u[hh - sb];
            oc[sb] = v[sb] - v[hh - sb];
            os[sb] = v[sb] + v[hh - sb];
        }
        for (coef, f.group, 0..) |*out, g, c| out.* = num.dot(folded[@as(usize, g) * m ..][0..m], f.fwd[c * m ..][0..m]);
    }
};

/// Pool size per collocation point (VACASK's default `samplefac`).
pub const pool_factor: usize = 5;

/// Builds the transform for `spec` into `buf`: nt + 2·nt·nf values.
pub fn transform(spec: Spectrum, tones: []const f64, buf: []f64, allocator: std.mem.Allocator) !Transform {
    const m = spec.freqs.len;
    const nf = 2 * m - 1;
    const nt = transformSamples(spec);
    const tr: Transform = .{ .nt = nt, .times = buf[0..nt], .inv = buf[nt..][0 .. nt * nf], .fwd = buf[nt + nt * nf ..][0 .. nt * nf] };
    if (spec.tones == 1) {
        // One tone: hb.zig's uniform, oversampled grid and its Galerkin
        // projection, harmonic h at sample s read at (h*s) mod nt so every
        // sample sees the same rounded angles.
        const nt_f: f64 = @floatFromInt(nt);
        for (0..nt) |s| {
            tr.times[s] = @as(f64, @floatFromInt(s)) / nt_f / tones[0];
            tr.inv[s * nf] = 1;
            tr.fwd[s] = 1 / nt_f;
            for (1..m) |j| {
                const h: usize = @intCast(spec.mix[j]);
                const angle = 2 * std.math.pi * @as(f64, @floatFromInt((h * s) % nt)) / nt_f;
                tr.inv[s * nf + 2 * j - 1] = @cos(angle);
                tr.inv[s * nf + 2 * j] = @sin(angle);
                tr.fwd[(2 * j - 1) * nt + s] = 2 * @cos(angle) / nt_f;
                tr.fwd[(2 * j) * nt + s] = 2 * @sin(angle) / nt_f;
            }
        }
        return tr;
    }

    // Several tones: a pool of pool_factor·nt uniform instants, shifted by
    // 0.2 step, then greedy selection of the rows farthest from the span
    // of those already kept (modified Gram-Schmidt), as VACASK
    // `buildColocation`. The pool spans 1/(closest line spacing), not
    // VACASK's one period of the lowest line: two lines Δf apart only
    // separate over a window near 1/Δf, and on 1k/1.3k tones at order 5
    // (lines every 100 Hz, lowest at 300 Hz) the shorter window left the
    // transform singular.
    const pool = pool_factor * nt;
    const rows = try allocator.alloc(f64, pool * nf);
    defer allocator.free(rows);
    const pool_t = try allocator.alloc(f64, pool);
    defer allocator.free(pool_t);
    var spacing = spec.freqs[1];
    for (spec.freqs[1 .. m - 1], spec.freqs[2..]) |lo, hi| spacing = @min(spacing, hi - lo);
    const step = 1 / spacing / @as(f64, @floatFromInt(pool));
    for (pool_t, 0..) |*t, i| t.* = (@as(f64, @floatFromInt(i)) + 0.2) * step;
    for (pool_t, 0..) |t, i| basisRow(spec, tones, t, rows[i * nf ..][0..nf]);
    for (0..nt) |i| {
        if (i > 0) {
            const wrt = rows[(i - 1) * nf ..][0..nf];
            const ww = num.dot(wrt, wrt);
            var best = i;
            var best_norm: f64 = -1;
            for (i..pool) |r| {
                const row = rows[r * nf ..][0..nf];
                num.axpy(row, -num.dot(row, wrt) / ww, wrt);
                const nrm = num.dot(row, row);
                if (nrm > best_norm) {
                    best_norm = nrm;
                    best = r;
                }
            }
            if (!(best_norm > 0)) return error.HbDidNotConverge;
            if (best != i) {
                for (rows[i * nf ..][0..nf], rows[best * nf ..][0..nf]) |*a, *b| std.mem.swap(f64, a, b);
                std.mem.swap(f64, &pool_t[i], &pool_t[best]);
            }
        }
    }
    @memcpy(tr.times, pool_t[0..nt]);
    std.sort.pdq(f64, tr.times, {}, std.sort.asc(f64));
    for (tr.times, 0..) |t, s| basisRow(spec, tones, t, tr.inv[s * nf ..][0..nf]);

    // fwd = inv^-1, one unit column at a time.
    const lu = try allocator.dupe(f64, tr.inv);
    defer allocator.free(lu);
    const piv = try allocator.alloc(u32, nt);
    defer allocator.free(piv);
    const e = try allocator.alloc(f64, 2 * nt);
    defer allocator.free(e);
    dense_lu.factorize(nt, lu, piv) catch return error.HbDidNotConverge;
    for (0..nt) |s| {
        @memset(e[0..nt], 0);
        e[s] = 1;
        dense_lu.solveFactored(nt, lu, piv, e[0..nt], e[nt..]);
        for (0..nf) |c| tr.fwd[c * nt + s] = e[nt + c];
    }
    return tr;
}

/// Time samples `transform` uses for `spec`.
pub fn transformSamples(spec: Spectrum) usize {
    const nf = 2 * spec.freqs.len - 1;
    return if (spec.tones == 1) 2 * nf else nf;
}

/// Row [1, cos ω_1 t, sin ω_1 t, ...] of the inverse transform at `t`.
/// Each tone's phase is reduced to [0, 1) cycles first, with the product's
/// rounding error recovered by fma, so a late instant keeps its phase.
pub fn basisRow(spec: Spectrum, tones: []const f64, t: f64, row: []f64) void {
    var cyc: [max_tones]f64 = undefined;
    for (tones, 0..) |f, i| {
        const p = f * t;
        cyc[i] = (p - @trunc(p)) + @mulAdd(f64, f, t, -p);
    }
    row[0] = 1;
    for (1..spec.freqs.len) |j| {
        var c: f64 = 0;
        for (spec.mix[j * tones.len ..][0..tones.len], cyc[0..tones.len]) |k, y| c += @as(f64, @floatFromInt(k)) * y;
        const angle = 2 * std.math.pi * c;
        row[2 * j - 1] = @cos(angle);
        row[2 * j] = @sin(angle);
    }
}

/// x_td[node*nt + s] = Σ_c inv[s, c]·x_hat[node*nf + c].
pub fn idft(tr: Transform, nf: usize, x_hat: []const f64, x_td: []f64) void {
    const n = x_hat.len / nf;
    if (tr.fold) |f| {
        for (0..n) |node| f.inverse(x_hat[node * nf ..][0..nf], x_td[node * tr.nt ..][0..tr.nt]);
        return;
    }
    for (0..n) |node| {
        const x = x_hat[node * nf ..][0..nf];
        for (x_td[node * tr.nt ..][0..tr.nt], 0..) |*out, s| out.* = num.dot(x, tr.inv[s * nf ..][0..nf]);
    }
}

/// x_hat[node*nf + c] = Σ_s fwd[c, s]·x_td[node*nt + s].
pub fn dft(tr: Transform, nf: usize, x_td: []const f64, x_hat: []f64) void {
    const n = x_hat.len / nf;
    if (tr.fold) |f| {
        for (0..n) |node| f.forward(x_td[node * tr.nt ..][0..tr.nt], x_hat[node * nf ..][0..nf]);
        return;
    }
    for (0..n) |node| {
        const x = x_td[node * tr.nt ..][0..tr.nt];
        for (x_hat[node * nf ..][0..nf], 0..) |*out, c| out.* = num.dot(x, tr.fwd[c * tr.nt ..][0..tr.nt]);
    }
}
