//! DC transfer function (`.tf`): `.dcxf`'s readout for one source. One
//! factorization of G at the operating point serves a forward solve with the
//! input's unit drive (gain, Rin) and an adjoint solve with e_out (Rout).
//! In: a circuit and its operating point. Out: three reals.
//! Cites ngspice tfanal.c; dev/analysis/tf.md (where a current output's Rout
//! departs from ngspice's 1e20 clamp).
//!
//! Contents:
//! - `pub const Options = core.query.Tf`
//! - `pub const Values = struct { gain, input_resistance, output_resistance }`
//! - `pub fn run(ctx: *const RunCtx, opts: Options) !Result`
//! - `pub fn solve(ckt: *Circuit, x_op: []const f64, options: Options, input_branch: u32, scratch: Allocator) !Values`

const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;
const XfSource = @import("core").query.XfSource;
const root = @import("../types.zig");
const xf = @import("xf.zig");

/// Query options, defined in core/query.zig.
pub const Options = @import("core").query.Tf;

/// The three `.tf` results, in the output's units (V/V, A/V, V/A or A/A for
/// the gain; ohms for both resistances).
pub const Values = struct {
    gain: f64,
    input_resistance: f64,
    output_resistance: f64,
};

/// Contract entry: one real point (transfer_function, input_resistance,
/// output_resistance) at `ctx.x_op`.
pub fn run(ctx: *const root.RunCtx, opts: Options) !root.Result {
    const a = ctx.allocator;
    const res = try solve(ctx.circuit, ctx.x_op, opts, opts.input_branch orelse ctx.source_branch, ctx.scratch_allocator);
    const names = try a.dupe([]const u8, &.{ "transfer_function", "input_resistance", "output_resistance" });
    errdefer a.free(names); // The entries are literals.

    const data = try a.dupe(f64, &.{ res.gain, res.input_resistance, res.output_resistance });
    return .{
        .plotname = "Transfer Function",
        .varnames = names,
        .is_complex = false,
        .npoints = 1,
        .data = data,
    };
}

/// Returns the DC transfer function at `x_op` from the source on
/// `input_branch` (unused when `options.input_nodes` names an I card) to
/// `options`' output. Rin, and Rout for a current output, read `xf.infinite`
/// (1e20, tfanal.c) when the unit drive moves less than 1e-20 A of branch
/// current. Overwrites the planes with the eval at `x_op` and the workspace
/// with its factorization. Fails with error.SingularMatrix when G is
/// singular once SparseLu has pinned its void unknowns (#160).
/// Asserts that `x_op` holds the circuit's `n` unknowns.
pub fn solve(ckt: *root.Circuit, x_op: []const f64, options: Options, input_branch: u32, scratch: Allocator) !Values {
    const n: usize = ckt.n;
    assert(x_op.len >= n);
    assert(input_branch < n or options.input_nodes != null);
    const ws = try ckt.workspace();
    ckt.eval(x_op, 0);
    try ws.slv.factor(ckt.g_vals, ckt.solver_execution);

    const buf = try scratch.alloc(f64, 2 * n);
    defer scratch.free(buf);

    const rhs = buf[0..n];
    const x = buf[n..];
    const out: xf.Output = .{ .node = options.output_node, .neg = options.output_neg, .branch = options.output_branch };
    // A V card drives its own branch row; an I card has none, so the unit
    // current enters its node pair (`.tf v(out) Iin`).
    const src: XfSource = if (options.input_nodes) |pair|
        .{ .name = "", .branch = null, .nodes = pair }
    else
        .{ .name = "", .branch = input_branch };

    @memset(rhs, 0);
    xf.excite(src, rhs);
    ws.slv.solve(rhs, x);
    const gain = out.read(x, n).re;
    const input_resistance = impedance(src, x, n);

    // Adjoint: J^T y = e_out. A voltage output reads Rout back on itself; a
    // current output reads the resistance its sensing source sees, as Rin
    // does for a V card: y[br] = (J^-1)[br][br] is the branch current a unit
    // volt on that source moves. tfanal.c takes 1/MAX(1e-20, y[br]), 1e20 on
    // every passive circuit (dev/analysis/tf.md).
    @memset(rhs, 0);
    out.seed(rhs);
    ws.slv.solveT(rhs, x);
    const output_resistance = if (out.branch) |br|
        impedance(.{ .name = "", .branch = br }, x, n)
    else
        out.read(x, n).re;
    return .{ .gain = gain, .input_resistance = input_resistance, .output_resistance = output_resistance };
}

// ---- private ----

/// `xf.immittance`'s impedance with tfanal.c's floor on a V card: a branch
/// current under 1e-20 A per volt reads `xf.infinite`, not its reciprocal
/// (#199). `.dcxf` follows VACASK, which clamps only an exact zero.
fn impedance(src: XfSource, x: []const f64, n: usize) f64 {
    assert(x.len == n);
    if (src.branch) |br| if (@abs(x[br]) < 1 / xf.infinite) return xf.infinite;
    return xf.immittance(src, x, n)[0].re;
}
