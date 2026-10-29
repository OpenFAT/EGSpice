//! HSPICE `.lin` network parameters after the S block of each `.sp` row:
//! Y, Z and H per frequency, group delays from a central difference at
//! omega (1 +- gd_step) (two more lanes per point, so frequency-dependent
//! elements are differentiated too), the stability factors, and the
//! two-port noise parameters off one adjoint solve per two-port mode.
//! In: the `.sp` rows with S filled, the port drives and the terminated
//! `FreqSolver`. Out: the same rows with every `.lin` column filled.
//! Cites HSPICE [SA Ch.17]; dev/analysis/s-parameters.md.
//!
//! Contents:
//! - `pub fn columns(n_ports: usize, lin: ?Lin) usize`
//! - `pub fn names(a: Allocator, n_ports: usize, lin: Lin) Allocator.Error![]const []const u8`
//! - `pub fn fill(ctx: *const RunCtx, fs: *FreqSolver, basis: *const Basis, omegas: []const f64, rhs: []const f64, data: []f64, row_len: usize, lin: Lin) !void`

const std = @import("std");
const assert = std.debug.assert;
const testing = std.testing;
const Allocator = std.mem.Allocator;
const Complex = @import("core").numerics.Complex;
const FreqSolver = @import("solver").freq_solve.FreqSolver;
const Lin = @import("core").query.Sp.Lin;
const root = @import("../types.zig");
const freq = @import("freq.zig");
const noise = @import("noise.zig");
const net = @import("sp_network.zig");
const Basis = @import("sp_basis.zig").Basis;

/// The two-port noise columns, in `net.noiseParams` order.
const noise_names = [_][]const u8{ "NFMIN", "NF", "RN", "YOPT", "GAMMA_OPT" };
/// The two-port stability columns (HSPICE's names), in `net.stability` order.
const stab_names = [_][]const u8{ "K_STABILITY_FACTOR", "MU_STABILITY_FACTOR" };
/// Relative half-width of the group-delay central difference. Truncation is
/// O(gd_step^2) and roundoff O(1e-16 / gd_step), both near 1e-10 relative.
const gd_step = 1e-5;

comptime {
    assert(noise_names.len == net.noise_count);
    assert(stab_names.len == net.stability_count);
}

/// Returns the columns `.sp` publishes for `n_ports` modes, the frequency
/// included; `session.schemaOf` sizes the result with it before the run.
pub fn columns(n_ports: usize, lin: ?Lin) usize {
    const l = lin orelse return 1 + n_ports * n_ports;
    const m = net.paramCount(n_ports);
    const two_port = stab_names.len + if (l.noise) noise_names.len else 0;
    return 1 + m * (1 + @as(usize, @intFromBool(l.group_delay))) + if (n_ports >= 2) two_port else 0;
}

/// Returns the `.lin` column names: `frequency`, then `X(i,j)` for X = S, Y,
/// Z, H (H over ports 1-2), then the group delays `TD(X(i,j))` in seconds
/// (HSPICE probes them as `X(i,j)(TD)`), the stability factors of ports 1-2
/// and the two-port noise parameters when asked. Real quantities have a zero
/// imaginary part. Allocates from the results arena `a`, which owns them.
pub fn names(a: Allocator, n_ports: usize, lin: Lin) Allocator.Error![]const []const u8 {
    const m = net.paramCount(n_ports);
    const noisy = lin.noise and n_ports >= 2;
    const out = try a.alloc([]const u8, columns(n_ports, lin));
    out[0] = "frequency";
    var col: usize = 1;
    for ("SYZH") |x| {
        const w = if (x == 'H') @min(n_ports, 2) else n_ports;
        if (w < 2 and x == 'H') continue;
        for (0..w) |i| for (0..w) |j| {
            out[col] = try a.print("{c}({d},{d})", .{ x, i + 1, j + 1 });
            col += 1;
        };
    }
    if (lin.group_delay) for (out[1..][0..m]) |name| {
        out[col] = try a.print("TD({s})", .{name});
        col += 1;
    };
    if (n_ports >= 2) {
        @memcpy(out[col..][0..stab_names.len], &stab_names);
        col += stab_names.len;
    }
    if (noisy) {
        @memcpy(out[col..][0..noise_names.len], &noise_names);
        col += noise_names.len;
    }
    assert(col == out.len);
    return out;
}

/// Fills every `.lin` column after S from the S block of each row: Y, Z, H;
/// then the group delays (a forward sweep at omega (1 +- gd_step)); then the
/// noise parameters (an adjoint sweep with one right-hand side per two-port
/// mode). `fs` holds the port terminations and `rhs` the unit port drives.
pub fn fill(ctx: *const root.RunCtx, fs: *FreqSolver, basis: *const Basis, omegas: []const f64, rhs: []const f64, data: []f64, row_len: usize, lin: Lin) !void {
    const np = basis.legs.len;
    const m = net.paramCount(np);
    assert(data.len == omegas.len * row_len);
    const work = try ctx.scratch_allocator.alloc(Complex, 2 * m + np * np);
    defer ctx.scratch_allocator.free(work);

    const lo = work[0..m];
    const hi = work[m..][0..m];
    const tmp = work[2 * m ..];
    for (0..omegas.len) |k| {
        const row = data[k * row_len ..][0..row_len];
        net.readComplex(row[2..][0 .. 2 * np * np], lo[0 .. np * np]);
        net.params(basis.z0, basis.pair, lo, tmp);
        net.writeComplex(lo[np * np ..], row[2 + 2 * np * np ..][0 .. 2 * (m - np * np)]);
    }
    var col: usize = 1 + m;
    if (lin.group_delay) {
        try groupDelays(ctx, fs, basis, omegas, rhs, data, row_len, col, lo, hi, tmp);
        col += m;
    }
    if (np < 2) return;
    for (0..omegas.len) |k| {
        const row = data[k * row_len ..][0..row_len];
        net.readComplex(row[2..][0 .. 2 * np * np], lo[0 .. np * np]);
        const ks = net.stability(net.block(lo, np, basis.pair));
        for (ks, 0..) |v, i| {
            row[2 * (col + i)] = v;
            row[2 * (col + i) + 1] = 0;
        }
    }
    col += stab_names.len;
    if (lin.noise) try noiseColumns(ctx, fs, basis, omegas, data, row_len, col, lo);
}

// ---- private ----

/// The group delays -dphi/domega of every parameter, from the phase of the
/// ratio of the two sides so a wrap cannot enter.
fn groupDelays(ctx: *const root.RunCtx, fs: *FreqSolver, basis: *const Basis, omegas: []const f64, rhs: []const f64, data: []f64, row_len: usize, col: usize, lo: []Complex, hi: []Complex, tmp: []Complex) !void {
    const scratch = ctx.scratch_allocator;
    const ckt = ctx.circuit;
    const n: usize = ckt.n;
    const nn = 2 * n;
    const np = basis.legs.len;
    const n_points = omegas.len;
    const pm = try scratch.alloc(f64, 2 * n_points + 2 * np * np);
    defer scratch.free(pm);

    const s_row = pm[2 * n_points ..];
    for (omegas, 0..) |w, k| {
        const dw = gd_step * @max(w, 1.0);
        pm[2 * k] = w - dw;
        pm[2 * k + 1] = w + dw;
    }
    var stream = try freq.Stream.init(scratch, fs, ckt, ctx.x_op, pm[0 .. 2 * n_points], rhs, false);
    defer stream.deinit(scratch);

    while (try stream.next(ckt)) |pt| {
        for (0..np) |p| net.writeColumn(n, basis.legs, pt.x[p * nn ..][0..nn], s_row, p);
        try basis.toModes(s_row);
        const side = if (pt.k % 2 == 0) lo else hi;
        net.readComplex(s_row, side[0 .. np * np]);
        net.params(basis.z0, basis.pair, side, tmp);
        if (pt.k % 2 == 0) continue;
        const row = data[(pt.k / 2) * row_len ..][0..row_len];
        const span = pm[pt.k] - pm[pt.k - 1];
        for (lo, hi, 0..) |l, h, e| {
            const r = h.div(l);
            row[2 * (col + e)] = -std.math.atan2(r.im, r.re) / span;
            row[2 * (col + e) + 1] = 0;
        }
    }
}

/// The two-port noise columns: the port voltage noise correlation with
/// every port terminated in its (noiseless) z0, then `net.noiseParams`.
fn noiseColumns(ctx: *const root.RunCtx, fs: *FreqSolver, basis: *const Basis, omegas: []const f64, data: []f64, row_len: usize, col: usize, lo: []Complex) !void {
    const scratch = ctx.scratch_allocator;
    const ckt = ctx.circuit;
    const n: usize = ckt.n;
    const nn = 2 * n;
    const np = basis.legs.len;
    const pq = basis.pair;
    const srcs = try ckt.collectNoiseSources(ctx.x_op, scratch);
    defer scratch.free(srcs);

    // Adjoint of each two-port mode's voltage, the sum over its legs of
    // w (v(node) - v(neg)) (`Basis.voltageWeight`).
    const e = try scratch.alloc(f64, 2 * nn);
    defer scratch.free(e);

    @memset(e, 0);
    for (pq, 0..) |q, i| for (basis.legs, 0..) |port, l| {
        const w = basis.voltageWeight(q, l);
        if (w == 0) continue;
        if (port.node != root.GROUND) e[i * nn + port.node] += w;
        if (port.neg != root.GROUND) e[i * nn + port.neg] -= w;
    };
    var stream = try freq.Stream.init(scratch, fs, ckt, ctx.x_op, omegas, e, true);
    defer stream.deinit(scratch);

    while (try stream.next(ckt)) |pt| {
        const row = data[pt.k * row_len ..][0..row_len];
        const f = row[0];
        // C_ij = sum over sources of PSD h_i conj(h_j). The stacked-real
        // transpose solve is A^H y = e, so the transfer h = conj(y_p - y_n),
        // and h1 conj(h2) = conj(y1) y2.
        var c11: f64 = 0;
        var c22: f64 = 0;
        var c12 = Complex.zero;
        var lead: usize = 0;
        while (lead < srcs.len) {
            const end = root.NoiseSource.groupEnd(srcs, lead);
            assert(end > lead);
            defer lead = end;
            // One correlated group: its rows' transfers add as phasors.
            const psd = noise.sourcePsd(srcs[lead], f);
            const h1 = noise.transfer(srcs[lead..end], pt.x[0..nn], n);
            const h2 = noise.transfer(srcs[lead..end], pt.x[nn..], n);
            const y1: Complex = .{ .re = h1[0], .im = h1[1] };
            const y2: Complex = .{ .re = h2[0], .im = h2[1] };
            c11 += psd * y1.magSq();
            c22 += psd * y2.magSq();
            c12 = c12.add(y2.mul(.{ .re = y1.re, .im = -y1.im }).scale(psd));
        }
        net.readComplex(row[2..][0 .. 2 * np * np], lo[0 .. np * np]);
        const out = net.noiseParams(net.block(lo, np, pq), .{ basis.z0[pq[0]], basis.z0[pq[1]] }, .{ c11, c22 }, c12);
        for (out, 0..) |v, i| {
            row[2 * (col + i)] = v.re;
            row[2 * (col + i) + 1] = v.im;
        }
    }
}

test names {
    // Every slot `columns` sizes is written, for every port count and flag.
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    for (1..5) |np| for ([_]bool{ false, true }) |gd| for ([_]bool{ false, true }) |nz| {
        const lin: Lin = .{ .group_delay = gd, .noise = nz };
        const got = try names(arena.allocator(), np, lin);
        try testing.expectEqual(columns(np, lin), got.len);
        const last = if (np >= 2 and nz) "GAMMA_OPT" else if (np >= 2) "MU_STABILITY_FACTOR" else if (gd) "TD(Z(1,1))" else "Z(1,1)";
        try testing.expectEqualStrings(last, got[got.len - 1]);
    };
}
