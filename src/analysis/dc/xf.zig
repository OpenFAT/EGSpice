//! All-source DC transfer (`.dcxf`, VACASK `dcxf`) and the incremental
//! solve (`.dcinc`): G at the operating point, factored once, against every
//! source's unit drive or the deck's AC excitation. In: a linearized circuit
//! and the query's sources. Out: one real point per query. `ac/xf.zig` runs
//! the same readout over a frequency sweep and borrows the helpers here.
//! Cites VACASK coredcxf.cpp:268-282 and coreacxf.cpp:484-497.
//!
//! Contents:
//! - `pub const Options = core.query.Dcxf`
//! - `pub const infinite: f64`
//! - `pub const Output = struct` (`seed`, `read`)
//! - `pub const Inc = struct` (`Options`, `run`)
//! - `pub fn run(ctx: *const RunCtx, opts: Options) !Result`
//! - `pub fn at(x: []const f64, n: usize, r: u32) Complex`
//! - `pub fn excite(src: XfSource, rhs: []f64) void`
//! - `pub fn adjointTf(src: XfSource, y: []const f64, n: usize) Complex`
//! - `pub fn immittance(src: XfSource, x: []const f64, n: usize) [2]Complex`
//! - `pub fn names(a: Allocator, first: ?[]const u8, sources: []const XfSource, tf_only: bool) Allocator.Error![]const []const u8`

const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;
const core = @import("core");
const Complex = core.numerics.Complex;
const XfSource = core.query.XfSource;
const root = @import("../types.zig");

/// Query options, defined in core/query.zig.
pub const Options = core.query.Dcxf;

/// The finite stand-in for an infinite immittance (a source that moves no
/// current, or a current source across a short), as VACASK writes it.
pub const infinite: f64 = 1e20;

/// Where an all-source transfer reads its output: a node pair or a branch.
pub const Output = struct {
    /// Output row; unused when `branch` is set.
    node: u32,
    /// `v(a,b)` reference row; GROUND is single-ended.
    neg: u32,
    /// `i(Vmeasure)`: that source's branch row.
    branch: ?u32,

    /// Writes the adjoint seed e_out into the output's own rows of `rhs`,
    /// which the caller zeroed. Asserts that those rows lie in `rhs`.
    pub fn seed(self: Output, rhs: []f64) void {
        if (self.branch) |br| {
            assert(br < rhs.len);
            rhs[br] = 1;
            return;
        }
        assert(self.node < rhs.len);
        assert(self.neg < rhs.len);
        rhs[self.node] = 1;
        if (self.neg != root.GROUND) rhs[self.neg] = -1;
    }

    /// Returns the output read from solution `x` (see `at`).
    pub fn read(self: Output, x: []const f64, n: usize) Complex {
        if (self.branch) |br| return at(x, n, br);
        return at(x, n, self.node).sub(at(x, n, self.neg));
    }
};

/// `.dcinc` (VACASK `dcinc`): the incremental solution G dx = du, where du
/// is every source's AC excitation (its real part: VACASK's `mag`, with the
/// phase applied), published on the deck's probes as one real point.
pub const Inc = struct {
    /// Query options, defined in core/query.zig.
    pub const Options = core.query.Dcinc;

    /// Contract entry: one real point, one column per deck probe.
    pub fn run(ctx: *const root.RunCtx, _: Inc.Options) !root.Result {
        const n: usize = ctx.circuit.n;
        const ws = try factorAtOp(ctx);
        const buf = try ctx.scratch_allocator.alloc(f64, 2 * n);
        defer ctx.scratch_allocator.free(buf);

        const rhs = buf[0..n];
        const x = buf[n..];
        if (ctx.ac_drive.len == 0) @memset(rhs, 0) else @memcpy(rhs, ctx.ac_drive[0..n]);
        ws.slv.solve(rhs, x);

        const data = try ctx.allocator.alloc(f64, ctx.probes.len);
        errdefer ctx.allocator.free(data);

        for (ctx.probes, data) |row, *d| d.* = at(x, n, row).re;
        return .{
            .plotname = "DC Incremental Analysis",
            .varnames = try root.probeNames(ctx, null),
            .is_complex = false,
            .npoints = 1,
            .data = data,
        };
    }
};

/// Contract entry: one real point with `tf(src)` [, `zin(src)`, `yin(src)`]
/// per source. `tf_only` reads every transfer off one adjoint solve;
/// otherwise each source's own forward solve gives its three values.
pub fn run(ctx: *const root.RunCtx, opts: Options) !root.Result {
    const n: usize = ctx.circuit.n;
    const out: Output = .{ .node = opts.output_node, .neg = opts.output_neg, .branch = opts.output_branch };
    const ws = try factorAtOp(ctx);
    const buf = try ctx.scratch_allocator.alloc(f64, 2 * n);
    defer ctx.scratch_allocator.free(buf);

    const rhs = buf[0..n];
    const x = buf[n..];
    const per: usize = if (opts.tf_only) 1 else 3;
    const data = try ctx.allocator.alloc(f64, per * opts.sources.len);
    errdefer ctx.allocator.free(data);

    if (opts.tf_only) {
        @memset(rhs, 0);
        out.seed(rhs);
        ws.slv.solveT(rhs, x);
        for (opts.sources, data) |s, *d| d.* = adjointTf(s, x, n).re;
    } else for (opts.sources, 0..) |s, i| {
        @memset(rhs, 0);
        excite(s, rhs);
        ws.slv.solve(rhs, x);
        const zy = immittance(s, x, n);
        data[3 * i ..][0..3].* = .{ out.read(x, n).re, zy[0].re, zy[1].re };
    }
    return .{
        .plotname = "DC Transfer Functions",
        .varnames = try names(ctx.allocator, null, opts.sources, opts.tf_only),
        .is_complex = false,
        .npoints = 1,
        .data = data,
    };
}

/// Returns row `r` of a real length-n or stacked-real length-2n solution;
/// ground reads zero. Asserts that `x` holds at least `n` values and `r < n`.
pub fn at(x: []const f64, n: usize, r: u32) Complex {
    assert(x.len >= n);
    assert(r < n);
    if (r == root.GROUND) return .zero;
    return .{ .re = x[r], .im = if (x.len > n) x[n + r] else 0 };
}

/// Adds the unit excitation of `src` into `rhs`: 1 V on a V card's branch
/// row, 1 A through an I card from `nodes[0]` to `nodes[1]`, the stamps its
/// AC magnitude makes. Asserts that the rows lie in `rhs`.
pub fn excite(src: XfSource, rhs: []f64) void {
    if (src.branch) |br| {
        assert(br < rhs.len);
        rhs[br] = 1;
        return;
    }
    assert(src.nodes[0] < rhs.len);
    assert(src.nodes[1] < rhs.len);
    if (src.nodes[0] != root.GROUND) rhs[src.nodes[0]] -= 1;
    if (src.nodes[1] != root.GROUND) rhs[src.nodes[1]] += 1;
}

/// Returns the transfer of `src` from the adjoint solution `y` = A^-T e_out.
pub fn adjointTf(src: XfSource, y: []const f64, n: usize) Complex {
    if (src.branch) |br| return at(y, n, br);
    return at(y, n, src.nodes[1]).sub(at(y, n, src.nodes[0]));
}

/// Returns `src`'s input impedance and admittance from its own forward
/// response `x`: a V card delivers -i_branch per volt, an I card sees
/// v(nodes[1]) - v(nodes[0]) per ampere. An exact zero inverts to
/// `infinite`, as VACASK does; any other value inverts.
pub fn immittance(src: XfSource, x: []const f64, n: usize) [2]Complex {
    if (src.branch) |br| {
        const y = at(x, n, br).scale(-1);
        return .{ reciprocal(y), y };
    }
    const z = at(x, n, src.nodes[1]).sub(at(x, n, src.nodes[0]));
    return .{ z, reciprocal(z) };
}

/// Returns the column labels `tf(name)`, then `zin(name)` and `yin(name)`
/// unless `tf_only`, per source, after `first` when given. Caller owns the
/// slice and every label but `first` (borrowed), all allocated with `a`;
/// on error nothing stays allocated.
pub fn names(a: Allocator, first: ?[]const u8, sources: []const XfSource, tf_only: bool) Allocator.Error![]const []const u8 {
    const kinds: []const []const u8 = if (tf_only) &.{"tf"} else &.{ "tf", "zin", "yin" };
    const extra: usize = @intFromBool(first != null);
    const out = try a.alloc([]const u8, extra + kinds.len * sources.len);
    errdefer a.free(out);

    if (first) |f| out[0] = f;
    var done: usize = 0;
    errdefer for (out[extra..][0..done]) |l| a.free(l);
    for (sources) |s| for (kinds) |k| {
        const l = try a.print("{s}({s})", .{ k, s.name });
        out[extra + done] = std.ascii.lowerString(l, l);
        done += 1;
    };
    assert(extra + done == out.len);
    return out;
}

// ---- private ----

/// Evaluates at `ctx.x_op` and factors G in the circuit's workspace.
fn factorAtOp(ctx: *const root.RunCtx) !*@import("solver").converger.Workspace {
    const ckt = ctx.circuit;
    assert(ctx.x_op.len >= ckt.n);
    const ws = try ckt.workspace();
    ckt.eval(ctx.x_op, 0);
    try ws.slv.factor(ckt.g_vals, ckt.solver_execution);
    return ws;
}

/// 1/v, or `infinite` at v = 0. Smith's division never squares |v|, so a
/// tiny v inverts to a huge value instead of `Complex.div`'s |v|^2
/// underflowing to 0 and returning inf below |v| ~ 1e-154 (#199). A real v
/// gives exactly 1/v.re, VACASK's DC `1.0/yin`.
fn reciprocal(v: Complex) Complex {
    if (v.re == 0 and v.im == 0) return .{ .re = infinite, .im = 0 };
    if (@abs(v.re) >= @abs(v.im)) {
        const r = v.im / v.re;
        const d = v.re + v.im * r;
        return .{ .re = 1 / d, .im = -r / d };
    }
    const r = v.re / v.im;
    const d = v.im + v.re * r;
    return .{ .re = r / d, .im = -1 / d };
}

test reciprocal {
    // Oracle: exact reals and the hand value 1/(3+4i) = (3-4i)/25.
    try std.testing.expectEqual(Complex{ .re = 0.5, .im = 0 }, reciprocal(.{ .re = 2, .im = 0 }));
    const z = reciprocal(.{ .re = 3, .im = 4 });
    try std.testing.expectApproxEqRel(0.12, z.re, 1e-15);
    try std.testing.expectApproxEqRel(-0.16, z.im, 1e-15);
    // A |v| far below 1e-154 still inverts to a finite value (#199).
    const tiny = reciprocal(.{ .re = 1e-200, .im = 1e-200 });
    try std.testing.expect(std.math.isFinite(tiny.re));
    try std.testing.expectEqual(infinite, reciprocal(.zero).re);
}
