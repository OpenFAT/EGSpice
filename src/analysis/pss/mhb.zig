//! Harmonic balance over any number of tones: Newton on the spectral
//! residual, each step solved by GMRES preconditioned with the block
//! diagonal G0 + jω_j·C0, one block per spectral line. The blocks share one
//! sparsity pattern and differ only in ω, so they factor as `FreqSolver`
//! frequency lanes (`factorEach`) once per Newton step, and every GMRES
//! iteration solves all of them at once (`solveEach`). The Jacobian itself
//! is never formed: a product is IDFT, the sampled G(t) and C(t) planes,
//! and DFT back, the sparse block Jacobian applied matrix-free.
//!
//! The spectrum is the box of mixing products k·f with |k_i| <= H_i,
//! optionally cut to a diamond Σ|k_i| <= INTMODMAX that keeps each tone's
//! own harmonics (HSPICE `.hb` [CR .HB], VACASK `truncate="hybrid"`).
//! Unknowns are node-major real blocks [dc, c_1, s_1, ..., c_{m-1}, s_{m-1}]
//! with x(t) = dc + Σ c_j cos(ω_j t) + s_j sin(ω_j t), the layout
//! `hb.solveSpectrum` uses for one tone, so one tone is the lines 0..K.
//!
//! One tone samples a uniform period 2x oversampled, as `hb.zig` does, so
//! the fixed point is the dense solver's. Several tones have no common
//! period: they take 2m-1 collocation instants picked greedily from an
//! oversampled pool for the best-conditioned transform, and invert it
//! (the APFT; VACASK `lib/corehbcoloc.cpp`, after Kundert, White and
//! Sangiovanni-Vincentelli, 1990, App. B).
//!
//! Cites: VACASK `lib/spurs.cpp` (line
//! merging in `spectrum`), Verilog-A LRM §4.6.1 (sources follow their
//! waveform only in the transient phase), #83 (NaN residuals).
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

/// The `.hb` query, read for its tone list (`f0` then `extra_tones`).
pub const Options = @import("core").query.Hb;

const SolveResult = pss.SolveResult;

/// The kept spectral lines, DC first, then by ascending frequency. Line j
/// is `freqs[j]` Hz, the mixing product Σ_i mix[j*tones + i]·f_i.
pub const Spectrum = struct {
    /// At most `max_tones`.
    tones: u8,
    freqs: []f64,
    /// Line-major, `tones` coefficients per line; |k_i| <= NHARMS_i <= i16 max.
    mix: []i16,

    /// Frees both tables; `gpa` is the allocator `spectrum` took.
    pub fn deinit(self: Spectrum, gpa: std.mem.Allocator) void {
        gpa.free(self.freqs);
        gpa.free(self.mix);
    }
};

/// The most tones a spectrum takes.
pub const max_tones = @import("core").validate.mhb_max_tones;

/// The largest box a spectrum enumerates, Π(2H_i + 1).
const max_box = @import("core").validate.mhb_max_box;

/// The spectrum of `tones` Hz with `nharms[i]` harmonics each and
/// INTMODMAX `intmodmax` (0: the whole box), where `sub` steps of the
/// first tone count as one order (HSPICE SUBHARMS; 1 for none). A product whose frequency
/// matches a kept one to 1e-14 relative is the same line: the lower order
/// wins, then a single tone's harmonic, then the earlier product, as in
/// VACASK `lib/spurs.cpp`. Asserts tones.len == nharms.len; `sub` must be
/// at least 1 (the query validator refuses 0). error.InvalidQueryOptions
/// for no tones, more than `max_tones`, a box past `max_box`, or a harmonic
/// count past i16 max. The caller frees the result with `Spectrum.deinit`
/// on `gpa`.
pub fn spectrum(gpa: std.mem.Allocator, tones: []const f64, nharms: []const u16, intmodmax: u16, sub: u16) !Spectrum {
    std.debug.assert(tones.len == nharms.len);
    const nt = tones.len;
    if (nt == 0 or nt > max_tones) return error.InvalidQueryOptions;
    var box: usize = 1;
    for (nharms) |h| {
        box = std.math.mul(usize, box, 2 * @as(usize, h) + 1) catch return error.InvalidQueryOptions;
        if (box > max_box or h > std.math.maxInt(i16)) return error.InvalidQueryOptions;
    }

    // Candidates on the positive half: the negative one is the same line.
    const Cand = struct { f: f64, order: u32, mixed: bool, idx: u32 };
    var cands: std.ArrayList(Cand) = .empty;
    defer cands.deinit(gpa);
    var mixes: std.ArrayList(i16) = .empty;
    defer mixes.deinit(gpa);
    var k: [max_tones]i32 = undefined;
    for (0..nt) |i| k[i] = -@as(i32, nharms[i]);
    for (0..box) |_| {
        var order: u32 = 0;
        var nnz: u32 = 0;
        var f: f64 = 0;
        var scale: f64 = 0;
        for (0..nt) |i| {
            order += if (i == 0) std.math.divCeil(u32, @abs(k[i]), sub) catch unreachable else @abs(k[i]);
            nnz += @intFromBool(k[i] != 0);
            const term = @as(f64, @floatFromInt(k[i])) * tones[i];
            f += term;
            scale += @abs(term);
        }
        const kept = intmodmax == 0 or order <= intmodmax or nnz <= 1;
        // Zero only for k = 0; any other product at 0 Hz is DC's duplicate.
        const tol = 1e-14 * scale;
        if (kept and ((order == 0) or f > tol)) {
            try cands.append(gpa, .{ .f = if (order == 0) 0 else f, .order = order, .mixed = nnz > 1, .idx = @intCast(cands.items.len) });
            for (0..nt) |i| try mixes.append(gpa, @intCast(k[i]));
        }
        // Odometer, last tone fastest.
        var i = nt;
        while (i > 0) {
            i -= 1;
            k[i] += 1;
            if (k[i] <= nharms[i]) break;
            k[i] = -@as(i32, nharms[i]);
        }
    }

    const c = cands.items;
    std.sort.pdq(Cand, c, {}, struct {
        fn lt(_: void, a: Cand, b: Cand) bool {
            return a.f < b.f or (a.f == b.f and a.idx < b.idx);
        }
    }.lt);
    // A run of candidates within 1e-14 relative of its head is one line.
    const runEnd = struct {
        fn f(cs: []const Cand, head: usize) usize {
            var end = head + 1;
            while (end < cs.len and cs[end].f - cs[head].f <= 1e-14 * cs[end].f) end += 1;
            return end;
        }
    }.f;
    // Counted first, so both tables are allocated at their final size: two
    // shrinking reallocs could not both be undone when the second failed
    // after the first had moved (#202).
    var lines: usize = 0;
    var head: usize = 0;
    while (head < c.len) : (lines += 1) head = runEnd(c, head);
    const freqs = try gpa.alloc(f64, lines);
    errdefer gpa.free(freqs);
    const mix = try gpa.alloc(i16, lines * nt);
    errdefer gpa.free(mix);
    var m: usize = 0;
    head = 0;
    while (head < c.len) : (m += 1) {
        const end = runEnd(c, head);
        var best = head;
        for (head + 1..end) |j| {
            const a = c[j];
            const b = c[best];
            if (a.order < b.order or (a.order == b.order and (!a.mixed and b.mixed or (a.mixed == b.mixed and a.idx < b.idx)))) best = j;
        }
        freqs[m] = c[best].f;
        @memcpy(mix[m * nt ..][0..nt], mixes.items[c[best].idx * nt ..][0..nt]);
        head = end;
    }
    std.debug.assert(m == lines);
    return .{ .tones = @intCast(nt), .freqs = freqs, .mix = mix };
}

/// The IDFT/DFT pair between a spectrum block (nf = 2m-1 reals) and nt
/// time samples: x(t_s) = Σ_c inv[s*nf + c]·X_c and
/// X_c = Σ_s fwd[c*nt + s]·x(t_s). Rows are contiguous for `num.dot`.
const Transform = struct {
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
const Fold = struct {
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
    fn inverse(f: *const Fold, coef: []const f64, x: []f64) void {
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
    fn forward(f: *const Fold, x: []const f64, coef: []f64) void {
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
const pool_factor: usize = 5;

/// Builds the transform for `spec` into `buf`: nt + 2·nt·nf values.
fn transform(spec: Spectrum, tones: []const f64, buf: []f64, allocator: std.mem.Allocator) !Transform {
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
fn transformSamples(spec: Spectrum) usize {
    const nf = 2 * spec.freqs.len - 1;
    return if (spec.tones == 1) 2 * nf else nf;
}

/// Row [1, cos ω_1 t, sin ω_1 t, ...] of the inverse transform at `t`.
/// Each tone's phase is reduced to [0, 1) cycles first, with the product's
/// rounding error recovered by fma, so a late instant keeps its phase.
fn basisRow(spec: Spectrum, tones: []const f64, t: f64, row: []f64) void {
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
fn idft(tr: Transform, nf: usize, x_hat: []const f64, x_td: []f64) void {
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
fn dft(tr: Transform, nf: usize, x_td: []const f64, x_hat: []f64) void {
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

/// dst += Ω·q_hat: dq/dt of q(t) = a cos(ω t) + b sin(ω t) is
/// ω b cos - ω a sin, so line j's cos row takes +ω_j·b and its sin row
/// -ω_j·a.
fn addOmega(omegas: []const f64, q_hat: []const f64, dst: []f64) void {
    const nf = 2 * omegas.len - 1;
    const n = q_hat.len / nf;
    for (0..n) |node| {
        const q = q_hat[node * nf ..][0..nf];
        const d = dst[node * nf ..][0..nf];
        for (omegas[1..], 1..) |w, j| {
            d[2 * j - 1] += w * q[2 * j];
            d[2 * j] -= w * q[2 * j - 1];
        }
    }
}

/// Time-domain Jacobian-vector product w_td[., s] = G(t_s) * v_td[., s] over
/// the CSC pattern; g_td is slot-major, samples contiguous. Each structural
/// entry is one axpy over its nt samples, W at a time, with the scalar tail
/// as the width-one oracle. Each row still sums its columns in ascending
/// order from +0, and an entry outside the pattern could only add ±0, so for
/// finite v this is bitwise the dense product. Overwrites every w_td slot.
pub fn gvProduct(w_td: []f64, g_td: []const f64, v_td: []const f64, col_ptr: []const u32, row_idx: []const u32, nt: usize) void {
    const W = std.simd.suggestVectorLength(f64) orelse 8;
    const V = @Vector(W, f64);
    @memset(w_td, 0);
    for (0..col_ptr.len - 1) |col| {
        const vc = v_td[col * nt ..][0..nt];
        for (col_ptr[col]..col_ptr[col + 1]) |slot| {
            const g = g_td[slot * nt ..][0..nt];
            const w = w_td[@as(usize, row_idx[slot]) * nt ..][0..nt];
            var s: usize = 0;
            while (s + W <= nt) : (s += W) {
                const gv: V = g[s..][0..W].*;
                const vv: V = vc[s..][0..W].*;
                const wv: V = w[s..][0..W].*;
                w[s..][0..W].* = wv + gv * vv;
            }
            while (s < nt) : (s += 1) w[s] += g[s] * vc[s];
        }
    }
}

/// The Newton linear system for GMRES: `matvec` is the HB Jacobian at the
/// last residual's samples, `precond` the held G0 + jω_j·C0 blocks.
const Operator = struct {
    ckt: *const root.Circuit,
    tr: Transform,
    omegas: []const f64,
    nf: usize,
    /// G and C at every sample, slot-major, samples contiguous.
    g_td: []const f64,
    c_td: []const f64,
    /// Matvec scratch: two n·nt time planes and one n·nf spectrum.
    v_td: []f64,
    w_td: []f64,
    u_hat: []f64,
    fs: *FreqSolver,
    gpu_held: ?*@import("../gpu/freq.zig").Held = null,
    gpu_operator: ?*@import("../gpu/periodic.zig").Operator = null,
    /// Preconditioner right-hand sides and solutions, m stacked 2n blocks.
    p_rhs: []f64,
    p_x: []f64,

    /// w = J v: IDFT, G(t)·v and C(t)·v per sample, DFT, plus Ω on the charge.
    pub fn matvec(self: *Operator, v: []const f64, w: []f64) void {
        if (self.gpu_operator) |device| if (device.matvec(v, w)) return;
        const nf = self.nf;
        const ckt = self.ckt;
        idft(self.tr, nf, v, self.v_td);
        gvProduct(self.w_td, self.g_td, self.v_td, ckt.col_ptr, ckt.row_idx, self.tr.nt);
        dft(self.tr, nf, self.w_td, w);
        if (!ckt.has_charge) return;
        gvProduct(self.w_td, self.c_td, self.v_td, ckt.col_ptr, ckt.row_idx, self.tr.nt);
        dft(self.tr, nf, self.w_td, self.u_hat);
        addOmega(self.omegas, self.u_hat, w);
    }

    /// r := M^-1 r. Line j's (c, s) rows are the phasor X = c - js of
    /// (G0 + jω_j C0)·X = r_c - j·r_s.
    pub fn precond(self: *Operator, r: []f64) void {
        if (self.gpu_operator) |device| if (device.precond(r, r)) return;
        const nf = self.nf;
        const n = r.len / nf;
        const nn = 2 * n;
        for (0..self.omegas.len) |j| {
            const b = self.p_rhs[j * nn ..][0..nn];
            for (0..n) |node| {
                b[node] = r[node * nf + ((2 * j) -| 1)];
                b[n + node] = if (j == 0) 0 else -r[node * nf + 2 * j];
            }
        }
        const device_solved = if (self.gpu_held) |held| held.solve(self.p_rhs, self.p_x, false) else false;
        if (!device_solved) self.fs.solveEach(self.p_rhs, self.p_x, false);
        for (0..self.omegas.len) |j| {
            const x = self.p_x[j * nn ..][0..nn];
            for (0..n) |node| {
                if (j == 0) {
                    r[node * nf] = x[node];
                } else {
                    r[node * nf + 2 * j - 1] = x[node];
                    r[node * nf + 2 * j] = -x[n + node];
                }
            }
        }
    }
};

/// The sampled planes: x, residual f and charge q node-major, G and C
/// slot-major, `nt` samples contiguous in each.
const Planes = struct { x: []f64, f: []f64, q: []f64, g: []f64, c: []f64 };

/// Carves `Planes` out of `slab` at `off`: x, f and q `node_len` each, then
/// g and c `slot_len` each.
fn takePlanes(slab: []f64, off: *usize, node_len: usize, slot_len: usize) Planes {
    return .{
        .x = take(slab, off, node_len),
        .f = take(slab, off, node_len),
        .q = take(slab, off, node_len),
        .g = take(slab, off, slot_len),
        .c = take(slab, off, slot_len),
    };
}

/// The step `sample` declares to the devices, 1/(nt·f_1) with f_1 the
/// lowest line.
fn sampleDt(spec: Spectrum) f64 {
    return 1 / (spec.freqs[1] * @as(f64, @floatFromInt(transformSamples(spec))));
}

/// Evaluates the circuit at every instant of `tr` on `x_hat`'s waveform
/// into `p`; `p.c` is left alone without charge. `x_sample` (n) is scratch.
fn sample(ckt: *root.Circuit, tr: Transform, nf: usize, dt: f64, x_hat: []const f64, x_sample: []f64, p: Planes) void {
    const n: usize = ckt.n;
    const nnz: usize = ckt.nnz;
    const nt = tr.nt;
    idft(tr, nf, x_hat, p.x);
    // Sample s is the circuit at t_s in the transient phase, the only
    // phase in which a source follows its waveform (§4.6.1).
    // The grouped driver evaluates these instants through gpu_lanes.Samples
    // when every model supports independent sampled evaluation.
    for (0..nt) |s| {
        const t = tr.times[s];
        for (0..n) |node| x_sample[node] = p.x[node * nt + s];
        ckt.setSimState(.{ .t = t, .dt = dt, .kind = .tran });
        ckt.eval(x_sample, t);
        for (0..n) |node| p.f[node * nt + s] = ckt.rhs[node];
        for (0..n) |node| p.q[node * nt + s] = ckt.q_vec[node];
        for (ckt.g_vals[0..nnz], 0..) |g, slot| p.g[slot * nt + s] = g;
        if (ckt.has_charge) for (ckt.c_vals[0..nnz], 0..) |c, slot| {
            p.c[slot * nt + s] = c;
        };
    }
}

/// The backtracking line search both HB Newtons take on their own residual
/// evaluation: a step that made the residual worse (or non-finite, hence
/// the negated comparison) is undone and retaken at half length, and a step
/// that helped earns the length back. A full Newton step can hand a diode
/// the whole source swing at once.
pub const LineSearch = struct {
    step: f64 = 1,
    prev_residual: f64 = std.math.inf(f64),
    retrying: bool = false,

    /// The shortest step before the search stops shortening.
    const min_step: f64 = 1.0 / 1024.0;

    /// Judges the step that produced `residual`. True: rewind it and retake
    /// it at the halved `step`. False: the step stands and `step` is the
    /// next one's length. `first` (the starting point, or a judgement that
    /// must not rewind) never backtracks.
    pub fn backtrack(ls: *LineSearch, first: bool, residual: f64) bool {
        if (!first and !(residual <= ls.prev_residual) and ls.step > min_step) {
            ls.step *= 0.5;
            ls.retrying = true;
            return true;
        }
        ls.prev_residual = residual;
        if (!ls.retrying) ls.step = @min(ls.step * 2, 1);
        ls.retrying = false;
        return false;
    }
};

/// GMRES stopping tolerance relative to the Newton residual.
const gmres_tol: f64 = 1e-7;
/// GMRES restart depth and restarts per Newton step.
const gmres_restart: u32 = 40;
const gmres_max_restarts: u32 = 20;
/// Harmonic Ritz vectors the host solve recycles across restarts and
/// Newton steps (`Gcrodr`), out of `gmres_restart`.
const gmres_recycle: u32 = 10;

/// Zeroes `x_hat` (node-major, `nf` coefficients per node) and seeds each
/// node's DC coefficient with the DC operating point, solved into
/// `x_sample`, which cuts the iteration count on biased circuits. Best
/// effort: a failed solve seeds its last iterate; only cancellation and out
/// of memory propagate. Asserts that `x_hat` holds `nf` per unknown.
pub fn dcSeed(ckt: *root.Circuit, x_hat: []f64, x_sample: []f64, nf: usize, tol: converger.Tolerances) !void {
    const n: usize = ckt.n;
    std.debug.assert(x_hat.len == n * nf);
    std.debug.assert(x_sample.len >= n);
    @memset(x_hat, 0);
    const ws = try ckt.workspace();
    ckt.setSimState(.{ .kind = .dc });
    @memset(x_sample, 0);
    _ = converger.run(ckt, ws, x_sample, 0, converger.optionsFromTolerances(tol, null), root.EvalHook{}) catch |err| switch (err) {
        error.QueryCancelled, error.OutOfMemory => return err,
        else => {}, // else: the DC seed is best effort
    };
    for (0..n) |node| x_hat[node * nf] = x_sample[node];
}

/// Driven harmonic balance on `spec` (tones `tones` Hz) from the DC
/// operating point. `x_hat` (n·(2m-1), caller-owned) receives node-major
/// blocks [dc, c_1, s_1, ...], the last iterate when Newton does not
/// converge. Newton takes the `LineSearch`; each step
/// is GMRES on the matrix-free Jacobian with the block-diagonal
/// preconditioner. Memory is O(nnz·nt + m·LU) for the sampled planes and
/// the held factors. Asserts `spec` has two or more lines and x_hat.len is
/// n·(2m-1). Non-convergence is `converged = false`, not an error; a GMRES
/// step that stops short is taken as it stands. Errors: OutOfMemory,
/// QueryCancelled, error.HbDidNotConverge when the collocation transform
/// of several tones is singular, and `FreqSolver.factorEach`'s.
pub fn solve(ckt: *root.Circuit, x_hat: []f64, spec: Spectrum, tones: []const f64, options: Options, allocator: std.mem.Allocator) !SolveResult {
    var state = try Newton.init(ckt, x_hat, spec, tones, options, allocator, true);
    defer state.deinit();
    while (state.phase != .done) switch (state.phase) {
        .sample => {
            try state.sampleHost();
            try state.sampled();
        },
        .linear => state.solveAlone(),
        .done => unreachable,
    };
    return state.result;
}

/// One point's Newton and backtracking continuation. Sampling and linear
/// solves are separate requests so independent points can share device
/// work without sharing a convergence or damping decision.
const Newton = struct {
    const Phase = enum { sample, linear, done };
    allocator: std.mem.Allocator,
    ckt: *root.Circuit,
    options: Options,
    phase: Phase = .sample,
    result: SolveResult = undefined,
    slab: []f64,
    x_hat: []f64,
    tr: Transform,
    nf: usize,
    dt: f64,
    f_hat: []f64,
    q_hat: []f64,
    dx: []f64,
    x_prev: []f64,
    planes: Planes,
    g0: []f64,
    c0: []f64,
    p_rhs: []f64,
    p_x: []f64,
    omegas: []f64,
    x_sample: []f64,
    fs: FreqSolver,
    factor_on_host: bool,
    factors_ready: bool = false,
    gmres: Gcrodr,
    gpu_held: ?@import("../gpu/freq.zig").Held = null,
    gpu_held_disabled: bool = false,
    gpu_operator: ?@import("../gpu/periodic.zig").Operator = null,
    ls: LineSearch = .{},
    iter: u16 = 0,
    backtracks: usize = 0,

    fn init(ckt: *root.Circuit, x_hat: []f64, spec: Spectrum, tones: []const f64, options: Options, allocator: std.mem.Allocator, single_gpu: bool) !Newton {
        const n: usize = ckt.n;
        const nnz: usize = ckt.nnz;
        const m = spec.freqs.len;
        const nf = 2 * m - 1;
        const nt = transformSamples(spec);
        const total = n * nf;
        std.debug.assert(x_hat.len == total and m >= 2);
        const len = nt + 2 * nt * nf + 4 * total + 3 * n * nt + 2 * nnz * nt + 2 * nnz + 4 * m * n + m + n;
        const slab = try allocator.alloc(f64, len);
        errdefer allocator.free(slab);
        var off: usize = 0;
        var tr = try transform(spec, tones, take(slab, &off, nt + 2 * nt * nf), allocator);
        if (spec.tones == 1) tr.fold = try Fold.init(allocator, spec, tr);
        errdefer if (tr.fold) |f| f.deinit(allocator);
        const f_hat = take(slab, &off, total);
        const q_hat = take(slab, &off, total);
        const dx = take(slab, &off, total);
        const x_prev = take(slab, &off, total);
        const planes = takePlanes(slab, &off, n * nt, nnz * nt);
        const g0 = take(slab, &off, nnz);
        const c0 = take(slab, &off, nnz);
        const p_rhs = take(slab, &off, 2 * m * n);
        const p_x = take(slab, &off, 2 * m * n);
        const omegas = take(slab, &off, m);
        const x_sample = take(slab, &off, n);
        std.debug.assert(off == len);
        for (omegas, spec.freqs) |*w, f| w.* = 2 * std.math.pi * f;
        if (!ckt.has_charge) @memset(planes.c, 0);
        try dcSeed(ckt, x_hat, x_sample, nf, options.tol);
        var fs = try FreqSolver.fromCircuit(allocator, ckt, x_sample);
        errdefer fs.deinit();
        // Complex held factors for the single driver, host or device; the
        // grouped driver factors stacked tapes on the device.
        fs.complex_held = fs.complex_lanes and single_gpu;
        const restart: u32 = @intCast(@min(total, gmres_restart));
        var gmres = try Gcrodr.init(allocator, @intCast(total), restart, @min(gmres_recycle, restart / 4));
        errdefer gmres.deinit(allocator);
        const device: ?@import("../gpu/periodic.zig").Operator = if (single_gpu and ckt.gpu != null)
            @import("../gpu/periodic.zig").Operator.init(allocator, .{
                .points = 1,
                .n = @intCast(n),
                .nf = @intCast(nf),
                .nt = @intCast(nt),
                .charge = ckt.has_charge,
                .col_ptr = ckt.col_ptr,
                .row_idx = ckt.row_idx,
                .inv = tr.inv,
                .fwd = tr.fwd,
                .omegas = omegas,
            }, ckt.knobs.gpu_stats != .off) catch null
        else
            null;
        return .{
            .allocator = allocator,
            .ckt = ckt,
            .options = options,
            .slab = slab,
            .x_hat = x_hat,
            .tr = tr,
            .nf = nf,
            .dt = sampleDt(spec),
            .f_hat = f_hat,
            .q_hat = q_hat,
            .dx = dx,
            .x_prev = x_prev,
            .planes = planes,
            .g0 = g0,
            .c0 = c0,
            .p_rhs = p_rhs,
            .p_x = p_x,
            .omegas = omegas,
            .x_sample = x_sample,
            .fs = fs,
            .gmres = gmres,
            .gpu_operator = device,
            .factor_on_host = single_gpu,
        };
    }

    fn deinit(self: *Newton) void {
        if (self.tr.fold) |f| f.deinit(self.allocator);
        if (self.gpu_operator) |*device| device.deinit();
        if (self.gpu_held) |*held| held.deinit();
        self.gmres.deinit(self.allocator);
        self.fs.deinit();
        self.allocator.free(self.slab);
        self.* = undefined;
    }

    fn checkpoint(self: *Newton) !void {
        if (self.iter != 0) try self.ckt.checkpoint(.{ .phase = .harmonic, .completed = self.iter });
    }

    fn sampleHost(self: *Newton) !void {
        try self.checkpoint();
        sample(self.ckt, self.tr, self.nf, self.dt, self.x_hat, self.x_sample, self.planes);
    }

    fn sampled(self: *Newton) !void {
        std.debug.assert(self.phase == .sample);
        dft(self.tr, self.nf, self.planes.f, self.f_hat);
        if (self.ckt.has_charge) {
            dft(self.tr, self.nf, self.planes.q, self.q_hat);
            addOmega(self.omegas, self.q_hat, self.f_hat);
        }
        const residual = num.normInf(self.f_hat);
        if (self.ckt.knobs.hb_trace) std.debug.print("MHB iter={d} res={e} step={e}\n", .{ self.iter, residual, self.ls.step });
        if (residual < self.options.hb_tol) {
            self.result = .{ .converged = true, .iterations = self.iter + 1, .residual_norm = residual };
            self.phase = .done;
            return;
        }
        if (self.ls.backtrack(self.iter == 0, residual)) {
            self.backtracks += 1;
            num.copySimd(self.x_hat, self.x_prev);
            num.axpy(self.x_hat, self.ls.step, self.dx);
            self.iterationComplete();
            return;
        }
        const nt = self.tr.nt;
        const inv_nt = 1 / @as(f64, @floatFromInt(nt));
        for (0..self.ckt.nnz) |slot| {
            self.g0[slot] = num.sum(self.planes.g[slot * nt ..][0..nt]) * inv_nt;
            self.c0[slot] = num.sum(self.planes.c[slot * nt ..][0..nt]) * inv_nt;
        }
        self.fs.setPlanes(self.g0, self.c0);
        self.factors_ready = false;
        if (self.factor_on_host) try self.factorHost();
        num.scale(self.f_hat, -1, self.f_hat);
        @memset(self.dx, 0);
        self.phase = .linear;
    }

    fn iterationComplete(self: *Newton) void {
        self.iter += 1;
        if (self.iter == self.options.max_iter) {
            self.result = .{ .converged = false, .iterations = self.options.max_iter, .residual_norm = num.normInf(self.f_hat) };
            self.phase = .done;
        } else self.phase = .sample;
    }

    fn solved(self: *Newton, result: Gmres.SolveResult) void {
        std.debug.assert(self.phase == .linear);
        if (self.ckt.knobs.hb_trace) std.debug.print("  gmres it={d} res={e} conv={}\n", .{ result.iterations, result.residual, result.converged });
        num.copySimd(self.x_prev, self.x_hat);
        num.axpy(self.x_hat, self.ls.step, self.dx);
        self.iterationComplete();
    }

    fn solveAlone(self: *Newton) void {
        std.debug.assert(self.factors_ready);
        if (self.gpu_operator) |*device| {
            device.refresh(self.planes.g, self.planes.c, &self.fs) catch {
                device.deinit();
                self.gpu_operator = null;
            };
        }
        if (self.gpu_operator == null)
            @import("../gpu/freq.zig").prepareHeld(&self.gpu_held, &self.gpu_held_disabled, self.ckt, &self.fs, self.omegas.len, self.allocator);
        var op: Operator = .{
            .ckt = self.ckt,
            .tr = self.tr,
            .omegas = self.omegas,
            .nf = self.nf,
            .g_td = self.planes.g,
            .c_td = self.planes.c,
            .v_td = self.planes.x,
            .w_td = self.planes.f,
            .u_hat = self.q_hat,
            .fs = &self.fs,
            .p_rhs = self.p_rhs,
            .p_x = self.p_x,
            .gpu_operator = if (self.gpu_operator) |*device| device else null,
            .gpu_held = if (self.gpu_held) |*held| held else null,
        };
        const total = self.f_hat.len;
        const result = (if (self.gpu_operator) |*device|
            device.solve(self.f_hat, self.dx, @intCast(@min(total, gmres_restart)), gmres_tol, gmres_max_restarts)
        else
            null) orelse self.gmres.solve(&op, self.f_hat, self.dx, gmres_tol, gmres_max_restarts);
        self.solved(result);
    }

    fn factorHost(self: *Newton) !void {
        if (self.factors_ready) return;
        try self.fs.ensureHeldOnHost(self.omegas);
        self.factors_ready = true;
    }
};

/// One point's `.hb` outcome, as `hb.run` returns it.
pub const LaneOutcome = @import("hb.zig").LaneOutcome;
/// Test-only counters of points the device batch admitted and of their line
/// search backtracks; empty outside tests.
pub const gpu_test_access = if (@import("builtin").is_test) struct {
    pub var admitted = std.atomic.Value(usize).init(0);
    pub var backtracks = std.atomic.Value(usize).init(0);
} else struct {};

/// Driven sparse HB/MHB points share sampled evaluation and resident
/// Krylov work. Every point keeps its own Newton iteration and line search.
pub fn runGpuLanes(ctxs: []const root.RunCtx, options: Options, out: []LaneOutcome) void {
    const artifacts = @import("gompute_kernels");
    std.debug.assert(ctxs.len == out.len and ctxs.len != 0);
    if (comptime artifacts.has_cuda) return dispatchGpuBatch(.cuda, ctxs, options, out);
    if (comptime artifacts.has_hip) return dispatchGpuBatch(.hip, ctxs, options, out);
    for (ctxs, out) |*ctx, *result| result.* = @import("hb.zig").run(ctx, options);
}

fn dispatchGpuBatch(comptime backend: @import("gompute").Backend, ctxs: []const root.RunCtx, options: Options, out: []LaneOutcome) void {
    runGpuBatch(backend, ctxs, options, out) catch |err| {
        for (out) |*o| o.* = err;
    };
}

/// Writes `input` as point `point`'s column of the point-minor batch `out`:
/// out[row * points + point] = input[row].
fn interleave(out: []f64, input: []const f64, point: usize, points: usize) void {
    std.debug.assert(out.len == input.len * points);
    for (input, 0..) |value, row| out[row * points + point] = value;
}

/// `interleave`'s inverse: point `point`'s column of `input` into `out`.
fn scatter(out: []f64, input: []const f64, point: usize, points: usize) void {
    std.debug.assert(input.len == out.len * points);
    for (out, 0..) |*value, row| value.* = input[row * points + point];
}

fn runGpuBatch(comptime backend: @import("gompute").Backend, ctxs: []const root.RunCtx, options: Options, out: []LaneOutcome) !void {
    const Device = @import("../gpu/periodic.zig").Engine(backend);
    const Samples = @import("../gpu/lanes.zig").Samples;
    const a = ctxs[0].scratch_allocator;
    const N = ctxs.len;
    const n: usize = ctxs[0].circuit.n;
    const nnz: usize = ctxs[0].circuit.nnz;
    var tones: [max_tones]f64 = undefined;
    const spec = try querySpectrum(a, options, &tones);
    defer spec.deinit(a);
    if (spec.freqs.len < 2) return error.InvalidQueryOptions;
    const nf = 2 * spec.freqs.len - 1;
    const nt = transformSamples(spec);
    const states = try a.alloc(?Newton, N);
    defer a.free(states);
    @memset(states, null);
    const spectra = try a.alloc([]f64, N);
    defer a.free(spectra);
    @memset(spectra, &.{});
    defer for (states, spectra, ctxs) |*state, x, ctx| {
        if (state.*) |*s| s.deinit();
        ctx.scratch_allocator.free(x);
    };
    @memset(out, error.QueryCancelled);
    var first: ?usize = null;
    for (ctxs, 0..) |ctx, p| {
        ctx.circuit.refuseDigital("hb") catch |err| {
            out[p] = err;
            continue;
        };
        if (ctx.circuit.n != n or ctx.circuit.nnz != nnz or ctx.circuit.has_charge != ctxs[0].circuit.has_charge or
            !std.mem.eql(u32, ctx.circuit.col_ptr, ctxs[0].circuit.col_ptr) or
            !std.mem.eql(u32, ctx.circuit.row_idx, ctxs[0].circuit.row_idx))
        {
            out[p] = @import("hb.zig").run(&ctx, options);
            continue;
        }
        spectra[p] = ctx.scratch_allocator.alloc(f64, n * nf) catch |err| {
            out[p] = err;
            continue;
        };
        states[p] = Newton.init(ctx.circuit, spectra[p], spec, tones[0..spec.tones], options, ctx.scratch_allocator, false) catch |err| {
            out[p] = err;
            continue;
        };
        if (first == null) first = p;
    }
    const first_point = first orelse return;
    const head = &states[first_point].?;
    const values = std.math.mul(usize, n * nf, N) catch return error.OutOfMemory;
    const sample_values = std.math.mul(usize, n * nt, N) catch return error.OutOfMemory;
    const plane_values = std.math.mul(usize, nnz * nt, N) catch return error.OutOfMemory;
    const sample_size = std.math.add(
        usize,
        std.math.mul(usize, sample_values, 3) catch return error.OutOfMemory,
        std.math.mul(usize, plane_values, 2) catch return error.OutOfMemory,
    ) catch return error.OutOfMemory;
    const sample_slab = try a.alloc(f64, sample_size);
    defer a.free(sample_slab);
    var sample_offset: usize = 0;
    const sampled = takePlanes(sample_slab, &sample_offset, sample_values, plane_values);
    const rhs = try a.alloc(f64, values);
    defer a.free(rhs);
    const dx = try a.alloc(f64, values);
    defer a.free(dx);
    const omegas = try a.alloc(f64, head.omegas.len * N);
    defer a.free(omegas);
    for (0..N) |p| interleave(omegas, head.omegas, p, N);
    const frequencies = try a.dupe(f64, head.omegas);
    defer a.free(frequencies);
    const factors = try a.alloc(*FreqSolver, N);
    defer a.free(factors);
    const results = try a.alloc(Gmres.SolveResult, N);
    defer a.free(results);
    const active = try a.alloc(u8, N);
    defer a.free(active);
    const ckts = try a.alloc(*root.Circuit, N);
    defer a.free(ckts);
    for (states, ckts) |*state, *ckt| ckt.* = if (state.*) |*s| s.ckt else head.ckt;
    var sampler: ?*Samples = Samples.init(a, ckts, head.tr.times, head.dt) catch |err| blk: {
        if (head.ckt.knobs.gpu_stats != .off) std.debug.print("gpu-hb: sample provider declined ({s})\n", .{@errorName(err)});
        break :blk null;
    };
    defer if (sampler) |s| s.deinit();
    var device: ?Device = Device.init(a, .{
        .points = N,
        .n = @intCast(n),
        .nf = @intCast(nf),
        .nt = @intCast(nt),
        .charge = head.ckt.has_charge,
        .col_ptr = head.ckt.col_ptr,
        .row_idx = head.ckt.row_idx,
        .inv = head.tr.inv,
        .fwd = head.tr.fwd,
        .omegas = omegas,
    }, head.ckt.knobs.gpu_stats != .off) catch null;
    defer if (device) |*d| d.deinit();
    var device_samples: usize = 0;
    var device_solves: usize = 0;
    while (true) {
        @memset(active, 0);
        var any = false;
        for (states, 0..) |*state, p| if (state.*) |*s| {
            if (s.phase == .sample) {
                s.checkpoint() catch |err| {
                    out[p] = err;
                    s.deinit();
                    state.* = null;
                    continue;
                };
                active[p] = 1;
                any = true;
            }
            idft(s.tr, s.nf, s.x_hat, s.planes.x);
            interleave(sampled.x, s.planes.x, p, N);
        } else {
            for (0..n * nt) |row| sampled.x[row * N + p] = 0;
        };
        if (!any) break;
        const used_device = if (sampler) |s| blk: {
            s.evaluate(sampled.x, sampled.f, sampled.q, sampled.g, sampled.c, active) catch {
                s.deinit();
                sampler = null;
                break :blk false;
            };
            device_samples += 1;
            break :blk true;
        } else false;
        for (active, states, 0..) |enabled, *state, p| {
            if (enabled == 0) continue;
            const s = &state.*.?;
            if (used_device) {
                scatter(s.planes.f, sampled.f, p, N);
                scatter(s.planes.q, sampled.q, p, N);
                scatter(s.planes.g, sampled.g, p, N);
                scatter(s.planes.c, sampled.c, p, N);
            } else {
                // The checkpoint already ran for this iteration.
                sample(s.ckt, s.tr, s.nf, s.dt, s.x_hat, s.x_sample, s.planes);
            }
            s.sampled() catch |err| {
                out[p] = err;
                s.deinit();
                state.* = null;
            };
        }
        const leader = for (states) |*state| {
            if (state.*) |*s| if (s.phase == .linear) break s;
        } else continue;
        for (states, 0..) |*state, p| {
            const s = if (state.*) |*live| if (live.phase == .linear) live else leader else leader;
            interleave(sampled.g, s.planes.g, p, N);
            interleave(sampled.c, s.planes.c, p, N);
            factors[p] = &s.fs;
            const solving = if (state.*) |*live| live.phase == .linear else false;
            active[p] = @intFromBool(solving);
            if (solving) {
                interleave(rhs, s.f_hat, p, N);
            } else {
                for (0..n * nf) |row| rhs[row * N + p] = 0;
            }
        }
        @memset(dx, 0);
        const solved_device = if (device) |*d| blk: {
            d.factor(sampled.g, sampled.c, factors, frequencies, active) catch {
                d.deinit();
                device = null;
                break :blk false;
            };
            d.solveHost(rhs, dx, @intCast(@min(n * nf, gmres_restart)), gmres_tol, gmres_max_restarts, results) catch {
                d.deinit();
                device = null;
                break :blk false;
            };
            device_solves += 1;
            break :blk true;
        } else false;
        for (states, 0..) |*state, p| if (state.*) |*s| {
            if (s.phase != .linear) continue;
            if (solved_device) {
                scatter(s.dx, dx, p, N);
                s.solved(results[p]);
            } else {
                s.factorHost() catch |err| {
                    out[p] = err;
                    s.deinit();
                    state.* = null;
                    continue;
                };
                s.solveAlone();
            }
        };
    }
    for (states, 0..) |state, p| if (state) |s| {
        out[p] = if (s.result.converged) publish(&ctxs[p], options, spec, s.x_hat) else error.HbDidNotConverge;
        if (ctxs[p].circuit.knobs.gpu_stats != .off and device_solves != 0)
            std.debug.print("gpu-hb-point: {d}, {d} iterations, {d} backtracks, converged={}\n", .{ p, s.result.iterations, s.backtracks, s.result.converged });
    };
    if (ctxs[0].circuit.knobs.gpu_stats != .off and device_solves != 0)
        std.debug.print("gpu-hb: {d} points, {d} sampled device rounds, {d} Krylov rounds\n", .{ N, device_samples, device_solves });
    if (@import("builtin").is_test and device_samples != 0 and device_solves != 0) {
        _ = gpu_test_access.admitted.fetchAdd(N, .monotonic);
        for (states) |state| if (state) |s| {
            _ = gpu_test_access.backtracks.fetchAdd(s.backtracks, .monotonic);
        };
    }
}

/// `spectrum` of `opts`' tones, f0 first, which land in `tones[0..spec.tones]`.
/// error.InvalidQueryOptions for more than `max_tones` tones or a harmonic
/// list that does not match them, and as `spectrum` otherwise.
fn querySpectrum(gpa: std.mem.Allocator, opts: Options, tones: *[max_tones]f64) !Spectrum {
    const count = 1 + opts.extra_tones.len;
    if (count > max_tones or opts.extra_harmonics.len != opts.extra_tones.len) return error.InvalidQueryOptions;
    var nharms: [max_tones]u16 = undefined;
    tones[0] = opts.f0;
    nharms[0] = opts.n_harmonics;
    @memcpy(tones[1..count], opts.extra_tones);
    @memcpy(nharms[1..count], opts.extra_harmonics);
    return spectrum(gpa, tones[0..count], nharms[0..count], opts.intmodmax, opts.subharms);
}

/// Contract entry for `.hb TONES=` (any tone count) and the positional
/// card with several tones: one row per spectral line, DC first then by
/// frequency. With `opts.phasors` the rows are complex, (frequency, each
/// probe's phasor c - j·s), x(t) = Re{X·e^(jωt)}, DC real; otherwise
/// (frequency, each probe's amplitude sqrt(c² + s²)) with a signed DC.
/// Non-convergence is error.HbDidNotConverge.
pub fn run(ctx: *const root.RunCtx, opts: Options) !root.Result {
    const scratch = ctx.scratch_allocator;
    var tones: [max_tones]f64 = undefined;
    const spec = try querySpectrum(scratch, opts, &tones);
    defer spec.deinit(scratch);
    if (spec.freqs.len < 2) return error.InvalidQueryOptions;
    const nf = 2 * spec.freqs.len - 1;
    const x_hat = try scratch.alloc(f64, @as(usize, ctx.circuit.n) * nf);
    defer scratch.free(x_hat);
    const st = try solve(ctx.circuit, x_hat, spec, tones[0..spec.tones], opts, scratch);
    if (!st.converged) return error.HbDidNotConverge;

    return publish(ctx, opts, spec, x_hat);
}

/// `run`'s published rows from the converged `x_hat`, in `ctx.allocator`.
fn publish(ctx: *const root.RunCtx, opts: Options, spec: Spectrum, x_hat: []const f64) !root.Result {
    const a = ctx.allocator;
    const nf = 2 * spec.freqs.len - 1;
    const names = try root.probeNames(ctx, "frequency");
    const width: usize = if (opts.phasors) 2 else 1;
    const ncols = names.len * width;
    const data = try a.alloc(f64, spec.freqs.len * ncols);
    @memset(data, 0);
    for (spec.freqs, 0..) |f, j| {
        const row = data[j * ncols ..][0..ncols];
        row[0] = f;
        for (ctx.probes, 1..) |node, p| {
            const x = x_hat[node * nf ..][0..nf];
            const v = row[p * width ..][0..width];
            if (j == 0) {
                v[0] = x[0];
            } else if (opts.phasors) {
                v[0] = x[2 * j - 1];
                v[1] = -x[2 * j];
            } else v[0] = std.math.hypot(x[2 * j - 1], x[2 * j]);
        }
    }
    return .{
        .plotname = "Harmonic Balance",
        .varnames = names,
        .is_complex = opts.phasors,
        .npoints = spec.freqs.len,
        .data = data,
    };
}

/// A converged multi-tone solution sampled for linearization: the
/// spectrum, its transform, and x, G and C at every collocation instant.
pub const Orbit = struct {
    spec: Spectrum,
    tr: Transform,
    /// Node-major, `tr.nt` samples each.
    x_td: []const f64,
    /// Slot-major over the circuit's pattern, `tr.nt` samples each.
    g_td: []const f64,
    c_td: []const f64,
    buf: []f64,

    /// Frees the orbit; `gpa` is the allocator `orbit` took.
    pub fn deinit(self: Orbit, gpa: std.mem.Allocator) void {
        self.spec.deinit(gpa);
        gpa.free(self.buf);
    }
};

/// Solves `opts`' harmonic balance and samples the solution for
/// linearization. error.HbDidNotConverge when Newton stops short,
/// error.InvalidQueryOptions when the spectrum has fewer than two lines,
/// and as `solve` and `spectrum` otherwise. Free with `Orbit.deinit` on
/// `allocator`.
pub fn orbit(ckt: *root.Circuit, opts: Options, allocator: std.mem.Allocator) !Orbit {
    var tones: [max_tones]f64 = undefined;
    const spec = try querySpectrum(allocator, opts, &tones);
    errdefer spec.deinit(allocator);
    const tone_list = tones[0..spec.tones];
    if (spec.freqs.len < 2) return error.InvalidQueryOptions;
    const n: usize = ckt.n;
    const nnz: usize = ckt.nnz;
    const nf = 2 * spec.freqs.len - 1;
    const nt = transformSamples(spec);
    const len = n * nf + nt + 2 * nt * nf + 3 * n * nt + 2 * nnz * nt + n;
    const buf = try allocator.alloc(f64, len);
    errdefer allocator.free(buf);
    var off: usize = 0;
    const x_hat = take(buf, &off, n * nf);
    const tr_buf = take(buf, &off, nt + 2 * nt * nf);
    const p = takePlanes(buf, &off, n * nt, nnz * nt);
    const x_sample = take(buf, &off, n);
    std.debug.assert(off == len);
    const st = try solve(ckt, x_hat, spec, tone_list, opts, allocator);
    if (!st.converged) return error.HbDidNotConverge;
    const tr = try transform(spec, tone_list, tr_buf, allocator);
    @memset(p.c, 0);
    sample(ckt, tr, nf, sampleDt(spec), x_hat, x_sample, p);
    return .{ .spec = spec, .tr = tr, .x_td = p.x, .g_td = p.g, .c_td = p.c, .buf = buf };
}

/// Spectral lines of `opts`' spectrum, DC included. Fails as `querySpectrum`.
pub fn lineCount(gpa: std.mem.Allocator, opts: Options) !usize {
    var tones: [max_tones]f64 = undefined;
    const spec = try querySpectrum(gpa, opts, &tones);
    defer spec.deinit(gpa);
    return spec.freqs.len;
}

/// Solves one tone at f0 with K = `options.n_harmonics` through `solve`;
/// `x_hat` in `hb.solveSpectrum`'s layout. Asserts K >= 1.
pub fn solveOneTone(ckt: *root.Circuit, x_hat: []f64, options: Options, allocator: std.mem.Allocator) !SolveResult {
    const tones = [_]f64{options.f0};
    const spec = try spectrum(allocator, &tones, &.{options.n_harmonics}, 0, 1);
    defer spec.deinit(allocator);
    return solve(ckt, x_hat, spec, &tones, options, allocator);
}

/// Private implementation access for the analysis test suite; void outside tests.
pub const test_access = if (@import("builtin").is_test) .{
    .transform = transform,
    .transformSamples = transformSamples,
    .Fold = Fold,
    .idft = idft,
    .dft = dft,
} else {};
