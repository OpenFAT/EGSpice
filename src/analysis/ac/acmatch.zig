//! AC mismatch (HSPICE `.acmatch`): the 1-sigma spread of one AC output over
//! the `.ac` sweep from the parameter variations `.dcmatch` uses, by adjoint
//! sensitivity per frequency lane. Each parameter costs one solve on the DC
//! factor (its operating-point shift, dx = -J^-1 dF/dp) and one eval at the
//! shifted point; each frequency then costs one pass over the pattern:
//!   dy/dp = -lambda(w)^T (dG/dp + jw dC/dp) x(w),
//! with A x = b the AC solution and A^T lambda = e_out. The magnitude, phase,
//! real and imaginary spreads sum each group's linear change in quadrature.
//! In: a run context and the variation table. Out: complex point-major rows
//! (frequency, four spreads, then each group's dy/dsigma).
//! Cites the HSPICE `.acmatch` manual; dev/analysis/dcmatch.md.
//!
//! Contents:
//! - `pub const Options = core.query.Acmatch`
//! - `pub fn run(ctx: *const RunCtx, opts: Options) !Result`

const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;
const num = @import("core").numerics;
const Complex = num.Complex;
const FreqSolver = @import("solver").freq_solve.FreqSolver;
const root = @import("../types.zig");
const dcmatch = @import("../dc/dcmatch.zig");
const dcxf = @import("../dc/xf.zig");
const fdStep = @import("../sweep/sens.zig").fdStep;
const freq = @import("freq.zig");

/// Query options, defined in core/query.zig.
pub const Options = @import("core").query.Acmatch;

/// The spread columns ahead of the per-group sensitivities: 1-sigma of the
/// magnitude, of the phase in degrees, and of the real and imaginary parts.
const spread_names = [_][]const u8{ "acm_mag", "acm_phase", "acm_re", "acm_im" };

/// Every point's AC solution x(w), then its adjoint z(w) = A^-H e_out (the
/// stacked-real transpose solve, so lambda = conj(z)), each 2n per point.
const Solutions = struct {
    all: []f64,
    n_points: usize,
    nn: usize,

    fn x(s: Solutions, k: usize) []const f64 {
        return s.all[k * s.nn ..][0..s.nn];
    }
    fn adjoint(s: Solutions, k: usize) []const f64 {
        return s.all[(s.n_points + k) * s.nn ..][0..s.nn];
    }
};

/// Contract entry: complex point-major rows (frequency, the four spreads as
/// real values, then each group's dy/dsigma, or dy/dp per parameter without
/// a variation block, labelled as `.dcmatch` labels them).
pub fn run(ctx: *const root.RunCtx, opts: Options) !root.Result {
    const scratch = ctx.scratch_allocator;
    const ckt = ctx.circuit;
    const sel = try dcmatch.Refs.init(scratch, ckt, opts.variations);
    defer sel.deinit(scratch);

    const groups = try dcmatch.Groups.init(scratch, sel.refs, sel.vars);
    defer groups.deinit(scratch);

    const freqs, const omegas = try freq.axis(scratch, opts.sweep);
    defer freq.freeAxis(scratch, freqs);

    const sol = try solveSweep(ctx, opts, omegas);
    defer scratch.free(sol.all);

    // lambda(w)[row] x(w)[col] per slot and point, once: every parameter
    // then reads its dy/dp off four contiguous dot products per point.
    // ponytail: holds 16 points nnz bytes; chunk the points when a long
    // sweep meets a large pattern.
    const nnz: usize = ckt.nnz;
    const prod = try scratch.alloc(f64, 2 * omegas.len * nnz);
    defer scratch.free(prod);

    const prod_re = prod[0 .. omegas.len * nnz];
    const prod_im = prod[omegas.len * nnz ..];
    for (0..omegas.len) |k| adjointProducts(ckt.col_ptr, ckt.row_idx, ckt.n, sol.x(k), sol.adjoint(k), prod_re[k * nnz ..][0..nnz], prod_im[k * nnz ..][0..nnz]);

    const dy = try scratch.alloc(Complex, groups.count() * omegas.len);
    defer scratch.free(dy);

    try groupShifts(ctx, sel.refs, groups, omegas, prod_re, prod_im, dy);
    return publish(ctx, opts, groups, sel.refs, freqs, sol, dy);
}

// ---- private ----

/// Solves the forward and adjoint sweeps on one `FreqSolver`.
fn solveSweep(ctx: *const root.RunCtx, opts: Options, omegas: []const f64) !Solutions {
    const scratch = ctx.scratch_allocator;
    const ckt = ctx.circuit;
    const nn = 2 * @as(usize, ckt.n);
    const n_points = omegas.len;
    const all = try scratch.alloc(f64, 2 * n_points * nn + 2 * nn);
    errdefer scratch.free(all);

    const drive = all[2 * n_points * nn ..][0..nn];
    const e_out = all[2 * n_points * nn + nn ..][0..nn];
    if (ctx.ac_drive.len == nn) @memcpy(drive, ctx.ac_drive) else @memset(drive, 0);
    root.outputDrive(e_out, opts.output_node, opts.output_neg);
    var fs = try FreqSolver.fromCircuit(scratch, ckt, ctx.x_op);
    defer fs.deinit();

    for ([_]bool{ false, true }, [_][]const f64{ drive, e_out }) |adjoint, rhs| {
        var stream = try freq.Stream.init(scratch, &fs, ckt, ctx.x_op, omegas, rhs, adjoint);
        defer stream.deinit(scratch);

        const base: usize = if (adjoint) n_points * nn else 0;
        while (try stream.next(ckt)) |pt| @memcpy(all[base + pt.k * nn ..][0..nn], pt.x);
    }
    return .{ .all = all, .n_points = n_points, .nn = nn };
}

/// dy per group and point: each member parameter's operating-point shift
/// and plane derivatives, folded against the adjoint products. Leaves the
/// planes at the nominal point, as the sweep found them.
// ponytail: dG/dp and dC/dp come from the plain eval, so an `ac=` value
// (Circuit.linearizeAc) and the frequency-dependent `acDyn` terms carry no
// parameter derivative; add them when a varied device has either.
fn groupShifts(ctx: *const root.RunCtx, refs: []const root.ParamRef, groups: dcmatch.Groups, omegas: []const f64, prod_re: []const f64, prod_im: []const f64, dy: []Complex) !void {
    const scratch = ctx.scratch_allocator;
    const ckt = ctx.circuit;
    const n: usize = ckt.n;
    const nnz: usize = ckt.nnz;
    const n_points = omegas.len;
    assert(dy.len == groups.count() * n_points);
    const planes = try scratch.alloc(f64, 4 * nnz + 3 * n);
    defer scratch.free(planes);

    const g0 = planes[0..nnz];
    const c0 = planes[nnz..][0..nnz];
    const dval = planes[2 * nnz ..][0..nnz];
    const dg = planes[3 * nnz ..][0..nnz];
    const rhs0 = planes[4 * nnz ..][0..n];
    const dx = planes[4 * nnz + n ..][0..n];
    const xs = planes[4 * nnz + 2 * n ..][0..n];
    const ws = try ckt.workspace();
    ckt.eval(ctx.x_op, 0);
    @memcpy(g0, ckt.g_vals[0..nnz]);
    @memcpy(c0, ckt.c_vals[0..nnz]);
    @memcpy(rhs0, ckt.rhs[0..n]);
    try ws.slv.factor(ckt.g_vals, ckt.solver_execution);

    @memset(dy, Complex.zero);
    for (0..groups.count()) |g| {
        try ckt.checkpoint(.{ .phase = .sweep, .completed = g, .total = groups.count() });
        for (groups.members(g)) |m| {
            const p = refs[m.param];
            const orig = p.get();
            p.set(orig + fdStep(orig));
            const delta = p.get() - orig;
            defer {
                p.set(orig);
                // Restores the value this circuit was built at, so it cannot re-wire it.
                ckt.recomputeType(p.type) catch unreachable;
            }
            ckt.recomputeType(p.type) catch |e| switch (e) {
                error.TopologyChanged => continue,
            };
            if (delta == 0) continue;
            // dx = -J^-1 dF/dp, then the planes at (p + delta, x_op + delta dx).
            ckt.eval(ctx.x_op, 0);
            diffQuot(dx, ckt.rhs[0..n], rhs0, delta);
            ws.slv.solve(dx, dx);
            @memcpy(xs, ctx.x_op[0..n]);
            num.axpy(xs, -delta, dx);
            ckt.eval(xs, 0);
            diffQuot(dg, ckt.g_vals[0..nnz], g0, delta);
            diffQuot(dval, ckt.c_vals[0..nnz], c0, delta);
            for (0..n_points) |k| {
                const acc = sensitivity(prod_re[k * nnz ..][0..nnz], prod_im[k * nnz ..][0..nnz], dg, dval, omegas[k]);
                dy[g * n_points + k] = dy[g * n_points + k].sub(acc.scale(m.scale));
            }
        }
    }
    ckt.eval(ctx.x_op, 0);
}

/// The result rows: the spreads summed in quadrature per point, then each
/// group's column.
fn publish(ctx: *const root.RunCtx, opts: Options, groups: dcmatch.Groups, refs: []const root.ParamRef, freqs: []const f64, sol: Solutions, dy: []const Complex) !root.Result {
    const a = ctx.allocator;
    const n: usize = ctx.circuit.n;
    const n_groups = groups.count();
    const n_points = freqs.len;
    const out: dcxf.Output = .{ .node = opts.output_node, .neg = opts.output_neg, .branch = null };
    // The results arena owns every allocation here, freed with the Result.
    const names = try a.alloc([]const u8, 1 + spread_names.len + n_groups);
    names[0] = "frequency";
    @memcpy(names[1..][0..spread_names.len], &spread_names);
    for (names[1 + spread_names.len ..], 0..) |*name, g| name.* = try groups.label(a, ctx.circuit, refs, g);
    const row_len = 2 * names.len;
    const data = try a.alloc(f64, n_points * row_len);
    const deg2 = std.math.radiansToDegrees(1) * std.math.radiansToDegrees(1);
    for (0..n_points) |k| {
        const row = data[k * row_len ..][0..row_len];
        const y = out.read(sol.x(k), n);
        row[0] = freqs[k];
        row[1] = 0;
        var var_m: f64 = 0;
        var var_p: f64 = 0;
        var var_r: f64 = 0;
        var var_i: f64 = 0;
        const mag = y.mag();
        for (0..n_groups) |g| {
            const d = dy[g * n_points + k];
            // One sigma of the group: its dy/dsigma; without a variation
            // block the column is dy/dp and the sigma scales it here.
            const s = d.scale(groups.sigmas[g]);
            const dm = (y.re * s.re + y.im * s.im) / mag;
            const dp = (y.re * s.im - y.im * s.re) / (mag * mag);
            var_m += dm * dm;
            var_p += dp * dp;
            var_r += s.re * s.re;
            var_i += s.im * s.im;
            row[2 * (1 + spread_names.len + g)] = d.re;
            row[2 * (1 + spread_names.len + g) + 1] = d.im;
        }
        for ([_]f64{ var_m, deg2 * var_p, var_r, var_i }, 0..) |v, i| {
            row[2 * (1 + i)] = @sqrt(v);
            row[2 * (1 + i) + 1] = 0;
        }
    }
    return .{
        .plotname = "AC Mismatch",
        .varnames = names,
        .is_complex = true,
        .npoints = n_points,
        .data = data,
    };
}

/// u[slot] = lambda[row] x[col] over a CSC pattern, with lambda = conj(adj):
/// `adj` and `x` are stacked-real 2n solutions, `re` and `im` hold the
/// pattern's nnz.
fn adjointProducts(col_ptr: []const u32, row_idx: []const u32, n: usize, x: []const f64, adj: []const f64, re: []f64, im: []f64) void {
    assert(x.len >= 2 * n);
    assert(adj.len >= 2 * n);
    for (0..n) |col| {
        const xc = Complex{ .re = x[col], .im = x[n + col] };
        for (col_ptr[col]..col_ptr[col + 1]) |slot| {
            const row = row_idx[slot];
            const u = (Complex{ .re = adj[row], .im = -adj[n + row] }).mul(xc);
            re[slot] = u.re;
            im[slot] = u.im;
        }
    }
}

/// Sum over slots of u (dg + jw dc), u = re + j im: lambda^T dA(w) x.
// fast-math: the four sums in one reassociated pass (2x the four strict dots
// at nnz = 16384, from reading each stream once). The gate falls back to the
// strict dots when the result is not finite, so a NaN or Inf from an input
// reaches the column; nothing downstream needs it detected.
fn sensitivity(re: []const f64, im: []const f64, dg: []const f64, dc: []const f64, omega: f64) Complex {
    assert(im.len == re.len);
    assert(dg.len == re.len);
    const fast = sensitivityFast(re, im, dg, dc, omega);
    if (std.math.isFinite(fast.re) and std.math.isFinite(fast.im)) return fast;
    return .{
        .re = num.dot(re, dg) - omega * num.dot(im, dc),
        .im = num.dot(im, dg) + omega * num.dot(re, dc),
    };
}

fn sensitivityFast(re: []const f64, im: []const f64, dg: []const f64, dc: []const f64, omega: f64) Complex {
    @setFloatMode(.optimized);
    var rg: f64 = 0;
    var id: f64 = 0;
    var ig: f64 = 0;
    var rd: f64 = 0;
    for (re, im[0..re.len], dg[0..re.len], dc[0..re.len]) |r, i, g, c| {
        rg += r * g;
        id += i * c;
        ig += i * g;
        rd += r * c;
    }
    return .{ .re = rg - omega * id, .im = ig + omega * rd };
}

/// dst[i] = (v[i] - v0[i]) / delta over dst.len.
fn diffQuot(dst: []f64, v: []const f64, v0: []const f64, delta: f64) void {
    assert(v.len >= dst.len);
    assert(v0.len >= dst.len);
    for (dst, v[0..dst.len], v0[0..dst.len]) |*d, a, b| d.* = (a - b) / delta;
}

test sensitivity {
    // Oracle: the per-slot complex fold it replaces, lambda^T (dG + jw dC) x,
    // on a two-column pattern whose nnz sweeps past dot's 16-accumulator
    // blocks.
    var prng = std.Random.DefaultPrng.init(0x5eed);
    const rnd = prng.random();
    const n = 96;
    var x: [2 * n]f64 = undefined;
    var adj: [2 * n]f64 = undefined;
    var dg: [n]f64 = undefined;
    var dc: [n]f64 = undefined;
    for (&x, &adj) |*a, *b| {
        a.* = rnd.float(f64) * 2 - 1;
        b.* = rnd.float(f64) * 2 - 1;
    }
    for (&dg, &dc) |*a, *b| {
        a.* = rnd.float(f64) * 2 - 1;
        b.* = rnd.float(f64) * 1e-9;
    }
    var row_idx: [n]u32 = undefined;
    for (&row_idx, 0..) |*r, i| r.* = @intCast((i * 37) % n);
    var col_ptr: [n + 1]u32 = undefined;
    var re: [n]f64 = undefined;
    var im: [n]f64 = undefined;
    const omega = 2 * std.math.pi * 1e6;
    for (0..n + 1) |nnz| {
        // Column 0 takes the first half of the slots, column 1 the rest.
        @memset(&col_ptr, @intCast(nnz));
        col_ptr[0] = 0;
        col_ptr[1] = @intCast(nnz / 2);
        adjointProducts(&col_ptr, &row_idx, n, &x, &adj, re[0..nnz], im[0..nnz]);
        const got = sensitivity(re[0..nnz], im[0..nnz], dg[0..nnz], dc[0..nnz], omega);
        var want = Complex.zero;
        var mass: f64 = 0;
        for (0..2) |col| {
            const xc = Complex{ .re = x[col], .im = x[n + col] };
            for (col_ptr[col]..col_ptr[col + 1]) |slot| {
                const row = row_idx[slot];
                const lam = Complex{ .re = adj[row], .im = -adj[n + row] };
                const term = lam.mul(.{ .re = dg[slot], .im = omega * dc[slot] }).mul(xc);
                want = want.add(term);
                mass += @abs(term.re) + @abs(term.im);
            }
        }
        // dot reassociates and the product is grouped differently: the
        // forward bound of a length-nnz fold, a few eps per term of mass.
        const tol = 8 * @as(f64, @floatFromInt(nnz + 2)) * std.math.floatEps(f64) * mass;
        try std.testing.expectApproxEqAbs(want.re, got.re, tol);
        try std.testing.expectApproxEqAbs(want.im, got.im, tol);
    }
    // A NaN or Inf input reaches the result through the strict fallback.
    for ([_]f64{ std.math.nan(f64), std.math.inf(f64) }) |bad| {
        for ([_]usize{ 0, 17, 95 }) |at| {
            const saved = dg[at];
            dg[at] = bad;
            const got = sensitivity(re[0..n], im[0..n], dg[0..n], dc[0..n], omega);
            try std.testing.expect(!std.math.isFinite(got.re) or !std.math.isFinite(got.im));
            dg[at] = saved;
        }
    }
}
