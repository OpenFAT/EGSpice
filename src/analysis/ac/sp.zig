//! S-parameter sweep (`.sp`) and HSPICE `.lin` network parameters. One
//! linearization gives G and C, the port z0 terminations go into the
//! frequency solver's copy of G, and each frequency is one lane of
//! `freq.Stream` with one right-hand side per port leg. The wave math is
//! `sp_network.zig`, the mixed-mode basis `sp_basis.zig`, and `.lin`'s
//! columns after S `sp_lin.zig`.
//! In: a run context and the query's ports. Out: complex point-major rows
//! (frequency, S11, S12, ..., Snn, then `.lin`'s columns).
//! Cites ngspice span.c (column and plot names, :544-551, :599-601);
//! HSPICE [SA Ch.17]; dev/analysis/s-parameters.md.
//!
//! Contents:
//! - `pub const Options = core.query.Sp`
//! - `pub const k_boltzmann`, `pub const t0_kelvin`
//! - `pub const columns = sp_lin.columns`
//! - `pub fn plan(ctx: *const RunCtx, opts: Options) !?freq.Plan`
//! - `pub fn run(ctx: *const RunCtx, opts: Options) !Result`
//! - `pub fn modeCount(ports: []const Port) usize`

const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;
const Port = @import("core").query.Port;
const FreqSolver = @import("solver").freq_solve.FreqSolver;
const root = @import("../types.zig");
const freq = @import("freq.zig");
const net = @import("sp_network.zig");
const sp_lin = @import("sp_lin.zig");
const Basis = @import("sp_basis.zig").Basis;

/// Query options, defined in core/query.zig.
pub const Options = @import("core").query.Sp;

/// ngspice CONSTboltz (const.h), the constant the device noise models use.
pub const k_boltzmann = net.k_boltzmann;
/// Noise figure reference temperature, K (IEEE; HSPICE's RN and GN too).
pub const t0_kelvin = net.t0_kelvin;
/// The columns `run` publishes for `n_ports` modes (`modeCount`), the
/// frequency included; `session.schemaOf` sizes the result with it.
pub const columns = sp_lin.columns;

/// Returns the leading multi-RHS sweep, including each ideal port's series
/// termination; null for `.lin`, which keeps its sweep live because later
/// delay and noise solves continue that solver's pivot history. Fails with
/// error.InvalidQueryOptions for a `.net` query past two legs.
pub fn plan(ctx: *const root.RunCtx, opts: Options) !?freq.Plan {
    if (opts.lin != null) return null;
    const scratch = ctx.scratch_allocator;
    const one_port = [_]Port{.{ .node = ctx.source_node, .branch = ctx.source_branch }};
    var basis = try Basis.init(scratch, if (opts.ports.len > 0) opts.ports else &one_port, .{ false, false });
    defer basis.deinit(scratch);

    if (opts.net and basis.legs.len > 2) return error.InvalidQueryOptions;
    return try sweepPlan(scratch, basis.legs, opts, ctx.circuit.n);
}

/// Contract entry: `opts.ports` (or the deck's drive source as port 1), the
/// full S matrix per frequency, then `.lin`'s columns (`sp_lin.names`).
/// Fails with error.InvalidQueryOptions for a `.net` query past two legs.
pub fn run(ctx: *const root.RunCtx, opts: Options) !root.Result {
    const a = ctx.allocator;
    const scratch = ctx.scratch_allocator;
    const ckt = ctx.circuit;
    const n: usize = ckt.n;
    const nn = 2 * n;
    const one_port = [_]Port{.{ .node = ctx.source_node, .branch = ctx.source_branch }};
    var basis = try Basis.init(scratch, if (opts.ports.len > 0) opts.ports else &one_port, if (opts.lin) |l| l.common else .{ false, false });
    defer basis.deinit(scratch);

    // Solves run over the legs; every published matrix is over the modes.
    const ports = basis.legs;
    const n_ports = ports.len;
    // `netRow`'s fixed 2x2 scratch, over legs: a balanced port is two.
    // core/validate.zig does not bound it.
    if (opts.net and n_ports > 2) return error.InvalidQueryOptions;
    const n_points: usize = opts.sweep.count();
    const names = if (opts.lin) |lin| try sp_lin.names(a, n_ports, lin) else try sNames(a, n_ports);
    const row_len = names.len * 2;
    const data = try a.alloc(f64, n_points * row_len);
    const sw = try sweepPlan(scratch, ports, opts, n);
    defer sw.deinit(scratch);

    var fs = try FreqSolver.fromCircuit(scratch, ckt, ctx.x_op);
    defer fs.deinit();

    // Series z0 inside each ideal port source: the branch row gains -z0 i_br.
    sw.configure(&fs);
    var stream = try freq.Stream.init(scratch, &fs, ckt, ctx.x_op, sw.omegas, sw.rhs, false);
    defer stream.deinit(scratch);

    stream.pre = ctx.freq_pre;
    while (try stream.next(ckt)) |pt| {
        const row = data[pt.k * row_len ..][0..row_len];
        row[0] = sw.freqs[pt.k];
        row[1] = 0;
        if (opts.net) {
            net.netRow(n, ports, pt.x, row[2..]);
        } else for (0..n_ports) |p| net.writeColumn(n, ports, pt.x[p * nn ..][0..nn], row[2..], p);
        try basis.toModes(row[2..][0 .. 2 * n_ports * n_ports]);
    }
    if (opts.lin) |lin| try sp_lin.fill(ctx, &fs, &basis, sw.omegas, sw.rhs, data, row_len, lin);
    return .{
        // The job name ngspice opens the plot under (span.c:599-601).
        .plotname = if (opts.lin != null) "LIN Analysis" else "SP Analysis",
        .varnames = names,
        .is_complex = true,
        .npoints = n_points,
        .data = data,
    };
}

/// Returns the modes (published ports) of `ports`: a balanced port counts
/// twice. At least 1, because an empty list means `run`'s one port at the
/// deck's drive source.
pub fn modeCount(ports: []const Port) usize {
    var n: usize = @max(ports.len, 1);
    for (ports) |p| n += @intFromBool(p.balanced != null);
    return n;
}

// ---- private ----

/// One unit drive per leg, and each ideal leg's series z0 termination (none
/// for `.net`): the sweep of `plan` and `run`.
fn sweepPlan(scratch: Allocator, legs: []const Port, opts: Options, n: usize) !freq.Plan {
    const rhs = try scratch.alloc(f64, legs.len * 2 * n);
    errdefer scratch.free(rhs);

    fillRhs(legs, n, rhs);
    var n_diag: usize = 0;
    if (!opts.net) {
        for (legs) |port| n_diag += @intFromBool(!port.series_z0);
    }
    const diag = try scratch.alloc(freq.Plan.Diagonal, n_diag);
    errdefer scratch.free(diag);

    var k: usize = 0;
    if (!opts.net) for (legs) |port| {
        if (port.series_z0) continue;
        diag[k] = .{ .row = port.branch, .value = -port.z0 };
        k += 1;
    };
    assert(k == n_diag);
    const freqs, const omegas = try freq.axis(scratch, opts.sweep);
    return .{ .freqs = freqs, .omegas = omegas, .rhs = rhs, .adjoint = false, .g_diagonal = diag };
}

/// The unit drive of each leg: its source voltage on a branch port, or for
/// `.net` a unit current into `node` and out of `neg`.
fn fillRhs(ports: []const Port, n: usize, rhs: []f64) void {
    assert(rhs.len == ports.len * 2 * n);
    @memset(rhs, 0);
    for (ports, 0..) |port, p| {
        if (port.branch != Port.no_branch) {
            rhs[p * 2 * n + port.branch] = 1.0;
            continue;
        }
        if (port.node != root.GROUND) rhs[p * 2 * n + port.node] = 1.0;
        if (port.neg != root.GROUND) rhs[p * 2 * n + port.neg] = -1.0;
    }
}

/// `frequency` then the S columns. ngspice names them `S_<row>_<col>`
/// (1-based) as UID_OTHER (span.c:544-551), and its raw writer types every
/// non-current UID as a voltage, so the file spells them `v(S_1_1)`.
fn sNames(a: Allocator, n_ports: usize) Allocator.Error![]const []const u8 {
    const names = try a.alloc([]const u8, 1 + n_ports * n_ports);
    names[0] = "frequency";
    for (0..n_ports) |i| for (0..n_ports) |j| {
        names[1 + i * n_ports + j] = try a.print("v(S_{d}_{d})", .{ i + 1, j + 1 });
    };
    return names;
}

test {
    _ = net;
    _ = sp_lin;
    _ = @import("sp_basis.zig");
}

test modeCount {
    // Oracle: the port list itself; a balanced port is two modes.
    try std.testing.expectEqual(@as(usize, 1), modeCount(&.{}));
    try std.testing.expectEqual(@as(usize, 3), modeCount(&.{
        .{ .node = 1, .branch = 4, .balanced = .{ .node = 2, .branch = 5 } },
        .{ .node = 3, .branch = 6 },
    }));
}
