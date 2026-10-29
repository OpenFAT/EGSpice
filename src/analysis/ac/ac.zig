//! AC small-signal sweep (`.ac`). One linearization at the operating point
//! gives the G and C planes; each frequency is then one lane of
//! `freq.Stream`, with no circuit evaluation inside the sweep.
//! In: a run context with the deck's composite AC drive. Out: complex
//! point-major rows (frequency, probes...).
//! Cites dev/analysis/ac-small-signal-noise.md.
//!
//! Contents:
//! - `pub const plotname`, `pub const scale`
//! - `pub const Options = core.query.Ac`
//! - `pub fn plan(ctx: *const RunCtx, opts: Options) !freq.Plan`
//! - `pub fn run(ctx: *const RunCtx, opts: Options) !Result`

const std = @import("std");
const assert = std.debug.assert;
const FreqSolver = @import("solver").freq_solve.FreqSolver;
const root = @import("../types.zig");
const freq = @import("freq.zig");

/// Query options, defined in core/query.zig.
pub const Options = @import("core").query.Ac;

/// The plot's name (`session.schemaOf` reports it).
pub const plotname = "AC Analysis";
/// The plot's scale column (`session.schemaOf` reports it).
pub const scale = "frequency";

/// Returns the sweep `run` solves: `ctx.ac_drive` (or zero) at every
/// omega of `opts`. Free it with `Plan.deinit` and the scratch allocator.
/// Asserts that `ctx.ac_drive` is 2n long or empty.
pub fn plan(ctx: *const root.RunCtx, opts: Options) !freq.Plan {
    const scratch = ctx.scratch_allocator;
    const nn = 2 * @as(usize, ctx.circuit.n);
    assert(ctx.ac_drive.len == nn or ctx.ac_drive.len == 0);
    const rhs = try scratch.alloc(f64, nn);
    errdefer scratch.free(rhs);

    if (ctx.ac_drive.len == nn) @memcpy(rhs, ctx.ac_drive) else @memset(rhs, 0);
    const freqs, const omegas = try freq.axis(scratch, opts.sweep);
    return .{ .freqs = freqs, .omegas = omegas, .rhs = rhs, .adjoint = false };
}

/// Contract entry: the response to `ctx.ac_drive`, the whole deck's AC
/// excitation. A V card drives its branch row, because its branch equation
/// is v_p - v_n - V = 0; driving the clamped node row instead would give an
/// identically zero response. An I card drives its two node rows. An empty
/// drive is a deck with no AC source and yields a zero response. A streamed
/// run (`ctx.stream`) writes each row there and returns empty `data`.
pub fn run(ctx: *const root.RunCtx, opts: Options) !root.Result {
    const a = ctx.allocator;
    const scratch = ctx.scratch_allocator;
    const ckt = ctx.circuit;
    const n: usize = ckt.n;
    const n_points: usize = opts.sweep.count();
    var fs = try FreqSolver.fromCircuit(scratch, ckt, ctx.x_op);
    defer fs.deinit();

    const p = try plan(ctx, opts);
    defer p.deinit(scratch);

    const names = try root.probeNames(ctx, scale);
    const row_len = names.len * 2;
    // A streamed run keeps only the row being built, in scratch; otherwise
    // the rows are the result.
    const streamed = ctx.stream != null;
    const data = if (streamed) try scratch.alloc(f64, row_len) else try a.alloc(f64, n_points * row_len);
    defer if (streamed) scratch.free(data);
    errdefer if (!streamed) a.free(data);

    var stream = try freq.Stream.init(scratch, &fs, ckt, ctx.x_op, p.omegas, p.rhs, false);
    defer stream.deinit(scratch);

    stream.pre = ctx.freq_pre;
    while (try stream.next(ckt)) |pt| {
        const row = if (streamed) data else data[pt.k * row_len ..][0..row_len];
        row[0] = p.freqs[pt.k];
        row[1] = 0;
        for (ctx.probes, 1..) |node, col| {
            row[col * 2] = pt.x[node];
            row[col * 2 + 1] = pt.x[n + node];
        }
        if (ctx.stream) |w| try w.writeAll(std.mem.sliceAsBytes(row));
    }
    assert(stream.k == n_points);
    return .{
        .plotname = plotname,
        .varnames = names,
        .is_complex = true,
        .npoints = n_points,
        .data = if (streamed) &.{} else data,
    };
}
