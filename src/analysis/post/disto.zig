//! Distortion analysis (`.disto`) by a simplified Volterra series. One eval
//! at the operating point gives the analytic G and C planes. The
//! second-order kernel is a central difference of the analytic Jacobians
//! (d2F = dG/dx, d2Q = dC/dx), two evals per unknown, and each frequency is
//! a few dense solves.
//!
//! The third-order kernel is never stored. d3 is O(n^4) and its only uses
//! are the contractions d3(V1,V1,V1) and, for two tones, d3(V1,V1,V2*), so
//! each is taken as directional second differences of the Jacobians along
//! the real directions V1 spans: eight or twelve evals per frequency point,
//! no tensor (see `cubicForms`, `mixedCubic`).
//!
//! Every kernel has a resistive and a charge part. A product at ω_out
//! sees K = F⁽ᵏ⁾ + jω_out·Q⁽ᵏ⁾, as ngspice's device DISTO sections stamp the
//! charge Taylor coefficients times jω of the mixing frequency.
//!
//! With a second tone (the card's `f2overf1`, a `DISTOF2` source) the
//! products are ngspice's intermodulation set: f1+f2 and f1-f2 from the
//! quadratic kernel, 2f1-f2 from the cubic one and the f1-f2 and 2f1
//! responses beating against the tones.
//!
//! ponytail: d2 kept as its nonzero terms (at most nnz(G) per unknown) and
//! O(n^3) dense solves per frequency; device-side analytic F''/F''' stamps and
//! a sparse frequency solve are the upgrade for large n.
//!
//! The kernels themselves are `disto_kernels.zig`; this file is the sweep
//! and the plots.
//! In: a run context and the `.disto` card. Out: one of five plots.
//! Cites ngspice distoan.c (:143), cktdisto.c (:115-158), dkerproc.c,
//! dloadfns.c (:545) and diodset.c (:78).
//!
//! Contents:
//! - `pub const Options = core.query.Disto`
//! - `pub fn run(ctx: *const RunCtx, opts: Options) !Result`

const std = @import("std");
const assert = std.debug.assert;
const dense_lu = @import("solver").dense_lu;
const num = @import("core").numerics;
const root = @import("../types.zig");
const kernels_mod = @import("disto_kernels.zig");
const bilinear = kernels_mod.bilinear;
const scatter = kernels_mod.scatter;
const cubic_step = kernels_mod.cubic_step;

/// Query options, defined in core/query.zig.
pub const Options = @import("core").query.Disto;

/// The caller's output buffers for one sweep. The four summary columns are
/// always written, one value per frequency point. `h2`/`h3` are ngspice's
/// own product, the second- and third-harmonic solution sampled at `probes`,
/// point-major with (re, im) adjacent; left empty, that order is not
/// computed at all.
const Out = struct {
    freqs: []f64,
    hd2: []f64,
    v1_mag: []f64,
    v2_mag: []f64,
    probes: []const u32 = &.{},
    h2: []f64 = &.{},
    h3: []f64 = &.{},
    /// Two-tone products (`Options.f2_ratio` > 0) at `probes`, laid out as
    /// `h2`: the f1+f2, f1-f2 and 2f1-f2 vectors. Each one left empty is not
    /// computed.
    f1pf2: []f64 = &.{},
    f1mf2: []f64 = &.{},
    twof1mf2: []f64 = &.{},
};

/// Contract entry: drive the `DISTOF1` card's branch (opts.drive_branch,
/// resolved from the deck) and measure at opts.output_node.
///
/// One `.disto` card publishes three plots, one query each (`opts.plot`):
/// ngspice's `DISTORTION - 2nd harmonic` and `- 3rd harmonic`, complex and
/// point-major (frequency, probes...), and espice's `Distortion Analysis`
/// summary, real and point-major (frequency, hd2, v1_mag, v2_mag).
pub fn run(ctx: *const root.RunCtx, opts: Options) !root.Result {
    const a = ctx.allocator;
    const x_op = ctx.x_op;

    var o = opts;
    // ponytail: with no DISTOF1 in the deck ngspice solves an unexcited system
    // and prints zeros; this falls back to the deck's drive source so a
    // card-less `.disto` still reports something. Drop the fallback when a
    // fixture wants ngspice's zeros.
    if (o.drive_branch == root.GROUND and o.ac_source_node == root.GROUND)
        o.drive_branch = ctx.source_branch;

    const n_points: usize = o.sweep.count();
    const scratch = ctx.scratch_allocator;
    const cols = try scratch.alloc(f64, n_points * 4);
    defer scratch.free(cols);
    const freqs = cols[0..n_points];
    const hd2_buf = cols[n_points .. 2 * n_points];
    const v1_buf = cols[2 * n_points .. 3 * n_points];
    const v2_buf = cols[3 * n_points ..];

    var out: Out = .{ .freqs = freqs, .hd2 = hd2_buf, .v1_mag = v1_buf, .v2_mag = v2_buf };
    const harmonic = o.plot != .summary;

    const harm_buf = try scratch.alloc(f64, n_points * ctx.probes.len * 2);
    defer scratch.free(harm_buf);
    if (harmonic) {
        out.probes = ctx.probes;
        switch (o.plot) {
            .second => out.h2 = harm_buf,
            .third => out.h3 = harm_buf,
            .f1pf2 => out.f1pf2 = harm_buf,
            .f1mf2 => out.f1mf2 = harm_buf,
            .twof1mf2 => out.twof1mf2 = harm_buf,
            .summary => unreachable,
        }
    }

    try sweep(ctx.circuit, x_op, out, o, scratch);

    if (harmonic) {
        const names = try root.probeNames(ctx, "frequency");
        const ncols = names.len;
        const data = try a.alloc(f64, n_points * ncols * 2);
        for (0..n_points) |i| {
            const row = data[i * ncols * 2 ..][0 .. ncols * 2];
            row[0] = freqs[i];
            row[1] = 0;
            @memcpy(row[2..], harm_buf[i * ctx.probes.len * 2 ..][0 .. ctx.probes.len * 2]);
        }
        return .{
            .plotname = switch (o.plot) {
                .second => "DISTORTION - 2nd harmonic",
                .third => "DISTORTION - 3rd harmonic",
                .f1pf2 => "DISTORTION - IM: f1+f2",
                .f1mf2 => "DISTORTION - IM: f1-f2",
                .twof1mf2 => "DISTORTION - IM: 2f1-f2",
                .summary => unreachable,
            },
            .varnames = names,
            .is_complex = true,
            .npoints = n_points,
            .data = data,
        };
    }

    const names = try a.dupe([]const u8, &.{ "frequency", "hd2", "v1_mag", "v2_mag" });
    const ncols = names.len;
    const data = try a.alloc(f64, n_points * ncols);
    for (0..n_points) |i| {
        const row = data[i * ncols ..][0..ncols];
        row[0] = freqs[i];
        row[1] = hd2_buf[i];
        row[2] = v1_buf[i];
        row[3] = v2_buf[i];
    }

    return .{
        .plotname = "Distortion Analysis",
        .varnames = names,
        .is_complex = false,
        .npoints = n_points,
        .data = data,
    };
}

// ---- private ----

/// Harmonic responses across the frequency sweep, into `out`:
///   1. Linearize at the operating point: one eval(), dense G and C.
///   2. Second-order kernels d2F/dxa dxb = dG[.,a]/dx_b and d2Q = dC/dx, a
///      central difference of the analytic Jacobians at 2n perturbed points
///      (only the extra order is FD).
///   3. For each frequency f, with K(ω) = F + jω·Q for each order's kernel:
///      a. Solve first-order: (G + jwC) * V1 = excitation
///      b. Second-order nonlinear current ½ K''(2w)(V1, V1)
///      c. Solve second-order: (G + j*2w*C) * V2 = -½ K''(2w)(V1, V1)
///      d. HD2 = |V2[output]| / |V1[output]|
///      e. Third order, only when asked for: the 3f1 current is
///         K''(3w)(V1, V2) + ⅙·K'''(3w)(V1, V1, V1), and
///         (G + j*3w*C) * V3 = -that.
///
/// The phasor convention is ngspice's throughout: the drive is half the
/// sinusoid amplitude (cktdisto.c:115) and every kernel is a one-sided
/// phasor, so the 2f1 source term is ½·F''·V1². ngspice spells the ½ into the
/// device coefficient (`g2 = 0.5 * gd / vte`, diodset.c:78) and contributes
/// `g2 * V1²` (dloadfns.c:545 D1n2F1). The harmonic vectors are reported as
/// sinusoid amplitudes, ×2 (DkerProc, dkerproc.c:43-52, `harmonic_scale`);
/// the summary magnitudes are not rescaled. The planes are left at the
/// operating point. Asserts that `hd2`, `v1_mag` and `v2_mag` match
/// `freqs` in length, that each non-empty `h2`/`h3` holds
/// `freqs.len * probes.len * 2` values, and that a two-tone output comes
/// with `options.f2_ratio > 0`.
fn sweep(
    ckt: *root.Circuit,
    x_op: []const f64,
    out: Out,
    options: Options,
    allocator: std.mem.Allocator,
) !void {
    const n: usize = ckt.n;
    const freqs = out.freqs;
    const hd2 = out.hd2;
    const v1_mag = out.v1_mag;
    const v2_mag = out.v2_mag;
    assert(hd2.len == freqs.len);
    assert(v1_mag.len == freqs.len);
    assert(v2_mag.len == freqs.len);
    const want_h2 = out.h2.len != 0;
    const want_h3 = out.h3.len != 0;
    assert(!want_h2 or out.h2.len == freqs.len * out.probes.len * 2);
    assert(!want_h3 or out.h3.len == freqs.len * out.probes.len * 2);
    const want_im3 = out.twof1mf2.len != 0;
    const two_tone = out.f1pf2.len != 0 or out.f1mf2.len != 0 or want_im3;
    assert(!two_tone or options.f2_ratio > 0);

    ckt.eval(x_op, 0);

    // The dense G and C planes, adjacent: the third-order stencils
    // difference both halves as one 2n·n slab.
    const n2 = n * n;
    const gc0 = try allocator.alloc(f64, 2 * n2);
    defer allocator.free(gc0);
    const g_dense = gc0[0..n2];
    const c_mat = gc0[n2..];
    ckt.denseG(g_dense);
    ckt.denseC(c_mat);

    // The second-order kernels, then the planes back at the operating point.
    const g_pert = try allocator.alloc(f64, 2 * n2);
    defer allocator.free(g_pert);

    const x_pert = try allocator.alloc(f64, n);
    defer allocator.free(x_pert);

    var kernels = try kernels_mod.Kernels.build(ckt, x_op, options.fd_eps, allocator);
    defer kernels.deinit(allocator);

    const terms = kernels.terms;
    const row_start = kernels.row_start;
    const nn = 2 * n;
    const a_work = try allocator.alloc(f64, nn * nn);
    defer allocator.free(a_work);
    const rhs_work = try allocator.alloc(f64, nn);
    defer allocator.free(rhs_work);
    const x_work = try allocator.alloc(f64, nn);
    defer allocator.free(x_work);

    const x_work2 = try allocator.alloc(f64, nn);
    defer allocator.free(x_work2);
    const x_work3 = try allocator.alloc(f64, nn);
    defer allocator.free(x_work3);
    // Each A(kω) adds the frequency-dependent entries (`Circuit.acDyn`). The
    // kernels are differences of G, which leaves those entries out, so a
    // nonlinearity behind a delay or filter reaches no harmonic.
    // ponytail: exact for the linear lines in models/; a nonlinear
    // operator input needs acDyn's own second derivative.
    const dyn_work = try allocator.alloc(f64, 2 * ckt.ac_dyn_slots.len);
    defer allocator.free(dyn_work);

    // Third-order scratch: one more dense G/C pair and the eight real cubic
    // forms of `cubicForms`. Both are the size of the (2n)^2 LU slab or
    // less, so they are allocated even when h3 is not wanted.
    const g_minus = try allocator.alloc(f64, 2 * n2);
    defer allocator.free(g_minus);
    const cubic = try allocator.alloc(f64, 8 * n);
    defer allocator.free(cubic);
    // Two-tone scratch: the F2 response V1b (and its conjugate's imaginary
    // part), the f1-f2 response, the IM right-hand side and solution, and
    // the two 2n·n G/C slabs `mixedCubic` accumulates.
    const t2: usize = if (two_tone) nn else 0;
    const im_work = try allocator.alloc(f64, 5 * t2 + (if (want_im3) 4 * n2 else 0));
    defer allocator.free(im_work);
    const v1b = im_work[0..t2];
    const v1b_cim = im_work[t2..][0 .. t2 / 2];
    const v2m = im_work[2 * t2 ..][0..t2];
    const x_im = im_work[3 * t2 ..][0..t2];
    const t3 = im_work[4 * t2 ..][0..t2];
    const m_planes = im_work[5 * t2 ..];

    const v1_re = x_work[0..n];
    const v1_im = x_work[n..nn];
    const v2_re = x_work2[0..n];
    const v2_im = x_work2[n..nn];

    // Half amplitude (cktdisto.c:115-116): the F1 drive is the one-sided
    // phasor of a cosine of amplitude `ac_magnitude`.
    const phase_rad = options.ac_phase * std.math.pi / 180.0;
    const drive_re = 0.5 * options.ac_magnitude * @cos(phase_rad);
    const drive_im = 0.5 * options.ac_magnitude * @sin(phase_rad);

    // F2, held at f2_ratio·f_start over the sweep (distoan.c:143): its
    // first-order response V1b is one solve for every point.
    const omega_f2 = 2.0 * std.math.pi * options.f2_ratio * options.sweep.f_start;
    if (two_tone) {
        const ph2 = options.ac2_phase * std.math.pi / 180.0;
        @memset(rhs_work, 0);
        rhs_work[options.drive2_branch] = 0.5 * options.ac2_magnitude * @cos(ph2);
        rhs_work[n + options.drive2_branch] = 0.5 * options.ac2_magnitude * @sin(ph2);
        try solveAt(ckt, x_op, g_dense, c_mat, omega_f2, a_work, dyn_work, rhs_work, v1b);
        num.scale(v1b_cim, -1, v1b[n..]);
    }

    var sw = options.sweep.iter();
    var k: usize = 0;
    while (sw.next()) |f| : (k += 1) {
        if (k != 0) try ckt.checkpoint(.{ .phase = .frequency, .completed = k, .total = freqs.len });
        const omega = 2.0 * std.math.pi * f;

        // 3a. First order: (G + jwC) V1 = ½ mag · e[drive row].
        @memset(rhs_work, 0);
        if (options.drive_branch != root.GROUND) {
            // V card: its own branch row (cktdisto.c:115-116).
            rhs_work[options.drive_branch] = drive_re;
            rhs_work[n + options.drive_branch] = drive_im;
        } else {
            // I card: current into the node, so the row is negated
            // (cktdisto.c:151-158, ISRCposNode gets −0.5·mag).
            rhs_work[options.ac_source_node] = -drive_re;
            rhs_work[n + options.ac_source_node] = -drive_im;
        }

        try solveAt(ckt, x_op, g_dense, c_mat, omega, a_work, dyn_work, rhs_work, x_work);

        // 3b. Second-order rhs, -½ K''(2w)[V1, V1]. The ½ is ngspice's:
        // `g2 = 0.5 * gd / vte` is ½·d²I/dV² (diodset.c:78) and D1n2F1
        // contributes `g2 * V1²` (dloadfns.c:545). The full double sum
        // already carries both (a,b) and (b,a), so the factor appears once.
        const omega2 = 2.0 * omega;
        bilinear(terms, row_start, v1_re, v1_im, v1_re, v1_im, -0.5, omega2, rhs_work);

        // 3c. Second order: (G + j·2w·C) V2 = -½ K''(V1,V1).
        try solveAt(ckt, x_op, g_dense, c_mat, omega2, a_work, dyn_work, rhs_work, x_work2);

        // 3d. Third order, only when h3 is wanted. The 3f1 current is
        // K''(V1,V2) + ⅙·K'''(V1,V1,V1) at 3w: the 2f1·f1 beat through the
        // quadratic kernel plus the direct cube.
        if (want_h3) {
            const d3v = kernels_mod.cubicForms(ckt, x_op, v1_re, v1_im, cubic_step, 3.0 * omega, gc0, g_pert, g_minus, x_pert, cubic);
            bilinear(terms, row_start, v1_re, v1_im, v2_re, v2_im, -1, 3.0 * omega, rhs_work);
            for (rhs_work[0..n], rhs_work[n..], d3v[0], d3v[1]) |*re, *im, d_re, d_im| {
                re.* -= d_re / 6.0;
                im.* -= d_im / 6.0;
            }
            try solveAt(ckt, x_op, g_dense, c_mat, 3.0 * omega, a_work, dyn_work, rhs_work, x_work3);
        }

        // 3e. Two tones. The one-sided phasor of ½K''(x,x) at w1+w2 is
        // K''(V1a, V1b), at w1-w2 K''(V1a, V1b*): the cross term of the
        // square counts both orders. At 2w1-w2 the third-order current is
        // K''(V1a, V2(w1-w2)) + K''(V1b*, V2(2w1)) + ½·K'''(V1a, V1a, V1b*),
        // the multinomial 3 of (w1, w1, -w2) over ⅙, every K at 2w1-w2.
        if (two_tone) {
            const w1pw2 = omega + omega_f2;
            const w1mw2 = omega - omega_f2;
            const v1b_re = v1b[0..n];
            if (out.f1pf2.len != 0) {
                bilinear(terms, row_start, v1_re, v1_im, v1b_re, v1b[n..], -1, w1pw2, rhs_work);
                try solveAt(ckt, x_op, g_dense, c_mat, w1pw2, a_work, dyn_work, rhs_work, x_im);
                scatter(out.f1pf2, k, out.probes, x_im);
            }
            if (out.f1mf2.len != 0 or want_im3) {
                bilinear(terms, row_start, v1_re, v1_im, v1b_re, v1b_cim, -1, w1mw2, rhs_work);
                try solveAt(ckt, x_op, g_dense, c_mat, w1mw2, a_work, dyn_work, rhs_work, v2m);
                if (out.f1mf2.len != 0) scatter(out.f1mf2, k, out.probes, v2m);
            }
            if (want_im3) {
                const w_im3 = 2.0 * omega - omega_f2;
                bilinear(terms, row_start, v1_re, v1_im, v2m[0..n], v2m[n..], -1, w_im3, rhs_work);
                bilinear(terms, row_start, v1b_re, v1b_cim, v2_re, v2_im, -1, w_im3, t3);
                num.axpy(rhs_work, 1, t3);
                kernels_mod.mixedCubic(ckt, x_op, v1_re, v1_im, v1b_re, v1b_cim, cubic_step, w_im3, gc0, g_pert, g_minus, m_planes, x_pert, t3);
                num.axpy(rhs_work, -0.5, t3);
                try solveAt(ckt, x_op, g_dense, c_mat, w_im3, a_work, dyn_work, rhs_work, x_im);
                scatter(out.twof1mf2, k, out.probes, x_im);
            }
        }

        // 3f. HD2 = |V2[output]| / |V1[output]|.
        const out_row = options.output_node;
        const v1_out_re = v1_re[out_row];
        const v1_out_im = v1_im[out_row];
        const v1_out_mag = @sqrt(v1_out_re * v1_out_re + v1_out_im * v1_out_im);

        const v2_out_re = v2_re[out_row];
        const v2_out_im = v2_im[out_row];
        const v2_out_mag = @sqrt(v2_out_re * v2_out_re + v2_out_im * v2_out_im);

        const hd2_val = if (v1_out_mag > 1e-30) v2_out_mag / v1_out_mag else 0;

        freqs[k] = f;
        hd2[k] = hd2_val;
        // The summary magnitudes are the half-amplitude kernels as solved,
        // with no DkerProc ×2: `disto/linear_divider_0p01`, a 0.75 divider on
        // `DISTOF1 0.01`, wants 3.75e-3 = ½·0.01·0.75.
        v1_mag[k] = v1_out_mag;
        v2_mag[k] = v2_out_mag;

        // ngspice's own product: the harmonic vector at every probe, as a
        // sinusoid amplitude (DkerProc).
        if (want_h2) scatter(out.h2, k, out.probes, x_work2);
        if (want_h3) scatter(out.h3, k, out.probes, x_work3);
    }

    // The third-order kernels perturb the planes at every point.
    if (want_h3 or want_im3) ckt.eval(x_op, 0);
}

/// Solves A(ω)·x = rhs, A = G + jωC plus the circuit's frequency-dependent
/// entries at x_op, in the stacked-real form (`a_work` is its 2n·2n slab).
fn solveAt(ckt: *root.Circuit, x_op: []const f64, g_dense: []const f64, c_mat: []const f64, omega: f64, a_work: []f64, dyn_work: []f64, rhs: []const f64, x: []f64) !void {
    const n: usize = ckt.n;
    dense_lu.buildComplexAdmittance(n, 2 * n, g_dense, c_mat, omega, a_work);
    ckt.addAcDynDense(x_op, omega, a_work, dyn_work);
    try dense_lu.factorizeSolve(2 * n, a_work, rhs, x);
}

test {
    _ = kernels_mod;
}
