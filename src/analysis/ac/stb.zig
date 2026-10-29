//! Loop-gain stability (`.stb`): drive the deck's own 0 V probe source with
//! a unit injection on its branch row and sweep T(omega) = -V(+)/V(-).
//! In: a run context and the probe's rows. Out: complex point-major rows
//! (frequency, loop_gain).
//! Cites ngspice's probe orientation; dev/analysis/stability.md.
//!
//! Contents:
//! - `pub const Options = core.query.Stb`
//! - `pub fn plan(ctx: *const RunCtx, opts: Options) !freq.Plan`
//! - `pub fn run(ctx: *const RunCtx, opts: Options) !Result`

const std = @import("std");
const assert = std.debug.assert;
const Complex = @import("core").numerics.Complex;
const FreqSolver = @import("solver").freq_solve.FreqSolver;
const root = @import("../types.zig");
const freq = @import("freq.zig");

/// Query options, defined in core/query.zig.
pub const Options = @import("core").query.Stb;

/// Returns the sweep `run` solves: a unit injection on the probe's branch
/// row. Fails with error.InvalidProbe, before allocating, when a probe
/// index is out of range or the `-` node is ground (no returned voltage to
/// divide by).
pub fn plan(ctx: *const root.RunCtx, opts: Options) !freq.Plan {
    const scratch = ctx.scratch_allocator;
    const n: usize = ctx.circuit.n;
    if (opts.probe_branch >= n or opts.probe_p >= n or opts.probe_n >= n) return error.InvalidProbe;
    if (opts.probe_n == root.GROUND) return error.InvalidProbe;
    const rhs = try scratch.alloc(f64, 2 * n);
    errdefer scratch.free(rhs);

    @memset(rhs, 0);
    rhs[opts.probe_branch] = 1.0;
    const freqs, const omegas = try freq.axis(scratch, opts.sweep);
    return .{ .freqs = freqs, .omegas = omegas, .rhs = rhs, .adjoint = false };
}

/// Contract entry: the loop gain at every frequency. The probe is the
/// deck's own 0 V source (`.stb Vprobe ...`): its branch equation is already
/// `v_p - v_n - V = 0`, so `rhs[branch] = 1` makes it the 1 V loop injection
/// and leaves every other stamp alone. The orientation is ngspice's: `+` is
/// where the signal arrives (the driven side of the break) and `-` where it
/// leaves into the rest of the loop, so T = -V(+)/V(-). Fails as `plan` does.
pub fn run(ctx: *const root.RunCtx, opts: Options) !root.Result {
    const a = ctx.allocator;
    const scratch = ctx.scratch_allocator;
    const ckt = ctx.circuit;
    const n: usize = ckt.n;
    const p = try plan(ctx, opts);
    defer p.deinit(scratch);

    var fs = try FreqSolver.fromCircuit(scratch, ckt, ctx.x_op);
    defer fs.deinit();

    const n_points: usize = opts.sweep.count();
    const names = try a.dupe([]const u8, &.{ "frequency", "loop_gain" });
    errdefer a.free(names); // The entries are literals.

    const data = try a.alloc(f64, n_points * 4);
    errdefer a.free(data);

    var stream = try freq.Stream.init(scratch, &fs, ckt, ctx.x_op, p.omegas, p.rhs, false);
    defer stream.deinit(scratch);

    stream.pre = ctx.freq_pre;
    while (try stream.next(ckt)) |pt| {
        const v_p: Complex = .{ .re = pt.x[opts.probe_p], .im = pt.x[n + opts.probe_p] };
        const v_n: Complex = .{ .re = pt.x[opts.probe_n], .im = pt.x[n + opts.probe_n] };
        const t = v_p.div(v_n).scale(-1);
        data[pt.k * 4 ..][0..4].* = .{ p.freqs[pt.k], 0, t.re, t.im };
    }
    assert(stream.k == n_points);
    return .{
        .plotname = "Stability Analysis",
        .varnames = names,
        .is_complex = true,
        .npoints = n_points,
        .data = data,
    };
}
