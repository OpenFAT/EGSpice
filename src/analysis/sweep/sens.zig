//! Adjoint DC sensitivity (`.sens`): one factorization at the operating
//! point and one transpose solve give lambda (J^T lambda = e_out). Each
//! parameter then costs one re-eval of F(x_op) and a dot product,
//! dy/dp = -lambda^T dF/dp with dF/dp a forward difference, instead of a
//! Newton solve per parameter. `Adjoint` is shared with `.dcmatch` and
//! `.dcsens`.
//! In: the circuit at its operating point and the output node pair. Out:
//! one real point of dVout/dp per parameter, named as ngspice names it.
//! Cites ngspice cktsens.c (Sens_Delta, names at :224-238) and the
//! IF_PRINCIPAL tables of res.c, cap.c, ind.c, vsrc.c and isrc.c; #71, #85,
//! #91, #154.
//!
//! Contents:
//! - `pub const Options = core.query.Sens`
//! - `pub const Adjoint = struct` (`init`, `deinit`, `fd`)
//! - `pub fn run(ctx: *const RunCtx, opts: Options) !Result`
//! - `pub fn solve(ckt, x_op, params, output_node, output_neg, gpa) ![]f64`
//! - `pub fn fdStep(orig: f64) f64`
//! - `pub fn isSensParam(type_name: []const u8, name: []const u8) bool`

const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;
const num = @import("core").numerics;
const root = @import("../types.zig");
const xf = @import("../dc/xf.zig");

/// Query options, defined in core/query.zig.
pub const Options = @import("core").query.Sens;

/// The adjoint of one output at an operating point: lambda from
/// J^T lambda = e_out, and the nominal F(x_op) each forward difference
/// subtracts. Both slices are n long.
pub const Adjoint = struct {
    lambda: []f64,
    rhs_nom: []f64,

    /// Evals F at `x_op`, factors J on the circuit workspace and solves for
    /// lambda. Overwrites the planes and the workspace factorization. Fails
    /// with OutOfMemory, or SingularMatrix from the factor. Free with
    /// `deinit` on `gpa`. Asserts that `x_op` holds `n` unknowns.
    pub fn init(ckt: *root.Circuit, x_op: []const f64, out: xf.Output, gpa: Allocator) !Adjoint {
        const n: usize = ckt.n;
        assert(x_op.len >= n);
        const ws = try ckt.workspace();
        const lambda = try gpa.alloc(f64, n);
        errdefer gpa.free(lambda);

        const rhs_nom = try gpa.alloc(f64, n);
        errdefer gpa.free(rhs_nom);

        // The full stamp, as every perturbed eval is: `recomputeType` drops
        // the constant-Jacobian baseline, so an `evalNewton` nominal would
        // difference two different summations.
        ckt.eval(x_op, 0);
        @memcpy(rhs_nom, ckt.rhs[0..n]);
        try ws.slv.factor(ckt.g_vals, ckt.solver_execution);
        @memset(lambda, 0);
        out.seed(lambda);
        ws.slv.solveT(lambda, lambda);
        return .{ .lambda = lambda, .rhs_nom = rhs_nom };
    }

    /// Frees both slices; `gpa` is the one `init` took.
    pub fn deinit(self: Adjoint, gpa: Allocator) void {
        gpa.free(self.lambda);
        gpa.free(self.rhs_nom);
    }

    /// Returns dy/dp = -lambda^T dF/dp by a forward difference that moves
    /// `p` to `to` and back: 0 when the move would change the topology,
    /// error.ZeroDelta when it rounds away. Overwrites the planes.
    pub fn fd(self: Adjoint, ckt: *root.Circuit, x_op: []const f64, p: root.ParamRef, to: f64) error{ZeroDelta}!f64 {
        assert(self.lambda.len == ckt.n);
        assert(self.rhs_nom.len == ckt.n);
        const orig = p.get();
        // Only this parameter moves and temperature does not, so re-deriving
        // its own device type is the whole recompute.
        p.set(to);
        defer {
            p.set(orig);
            // Restores the value this circuit was built at, so it cannot re-wire it.
            ckt.recomputeType(p.type) catch unreachable;
        }
        // A parameter whose nominal value collapses an internal node
        // (gummel_poon RC/RE = 0, mos1 RD/RS = 0, ...) is re-wired by the
        // absolute step at 0 (`fdStep`): the perturbed circuit has a node the
        // frozen pattern lacks, so the derivative is not representable;
        // report 0 rather than fail. ngspice never perturbs a topology
        // parameter (cktsens.c uses per-device analytic sensitivities).
        ckt.recomputeType(p.type) catch |e| switch (e) {
            error.TopologyChanged => return 0,
        };
        // Difference against the step actually stored: an f32 parameter
        // rounds it, and the requested delta would give a wrong derivative.
        const delta = p.get() - orig;
        if (delta == 0) return error.ZeroDelta;
        ckt.eval(x_op, 0);
        // dF = F(p + delta) - F(p) in the plane itself; `num.dot`'s fixed
        // fold keeps `.sens` and `.dcmatch` bit-reproducible.
        const df = ckt.rhs[0..self.lambda.len];
        num.sub(df, df, self.rhs_nom);
        return -num.dot(self.lambda, df) / delta;
    }
};

/// ngspice's IF_PRINCIPAL parameter (IOPP/IOPAP in res.c, cap.c, ind.c,
/// vsrc.c, isrc.c) by device type; every other device has none, so all its
/// columns are `<card>:<param>` (`v(d1:is)`, #71).
const principal = std.StaticStringMap([]const u8).initComptime(.{
    .{ "resistor", "r" }, .{ "capacitor", "c" }, .{ "inductor", "l" },
    .{ "vsource", "dc" }, .{ "isource", "dc" },
});

/// Contract entry: one real point, one column per collected device
/// parameter holding dVout/dp. Columns carry ngspice's names:
///   model parameter                  -> `<card>:<param>`   (r1:tc1)
///   principal instance parameter     -> `<card>`           (r1)
///   any other instance parameter     -> `<card>_<param>`   (r1_scale)
pub fn run(ctx: *const root.RunCtx, opts: Options) !root.Result {
    const a = ctx.allocator;
    const scratch = ctx.scratch_allocator;
    const all = try ctx.circuit.collectParams();
    var kept: std.ArrayList(root.ParamRef) = .empty;
    defer kept.deinit(scratch);

    for (all) |ref| if (isSensParam(ctx.circuit.typeName(ref.type), ref.param_name)) try kept.append(scratch, ref);
    const refs = kept.items;
    const sens = try solve(ctx.circuit, ctx.x_op, refs, opts.output_node, opts.output_neg, scratch);
    defer scratch.free(sens);

    const names = try a.alloc([]const u8, refs.len);
    errdefer a.free(names);

    var done: usize = 0;
    errdefer for (names[0..done]) |s| a.free(s);
    for (refs, names) |ref, *name| {
        name.* = try columnName(a, scratch, ctx, opts, ref);
        done += 1;
    }
    return .{
        .plotname = "Sensitivity Analysis",
        .varnames = names,
        .is_complex = false,
        .npoints = 1,
        .data = try a.dupe(f64, sens),
    };
}

/// Returns dV(output_node, output_neg)/dp for each of `params` at `x_op`,
/// parallel to `params` and owned by `gpa`. `x_op` is the executor's
/// operating point (the full OP ladder; a plain Newton here fails on
/// circuits that need stepping). Each parameter is restored before the
/// next. A parameter whose step would change the topology reads 0, and so
/// does one at its "infinite" default 0 whose quotient does not survive a
/// doubled step (#91). Overwrites the planes and the workspace
/// factorization. Fails with error.ZeroDelta when a step rounds away.
pub fn solve(ckt: *root.Circuit, x_op: []const f64, params: []const root.ParamRef, output_node: u32, output_neg: u32, gpa: Allocator) ![]f64 {
    // No `computeBaseline()`: the baseline would freeze the const-Jacobian
    // stamps (resistors) and mask the very perturbations being measured.
    const adj = try Adjoint.init(ckt, x_op, .{ .node = output_node, .neg = output_neg, .branch = null }, gpa);
    defer adj.deinit(gpa);

    const sens = try gpa.alloc(f64, params.len);
    errdefer gpa.free(sens);

    for (params, sens, 0..) |p, *out, index| {
        if (index != 0) try ckt.checkpoint(.{ .phase = .sweep, .completed = index, .total = params.len });
        const orig: f64 = p.get();
        out.* = try adj.fd(ckt, x_op, p, orig + fdStep(orig));
        // 0 is the "infinite" default of a 1/p parameter (gummel_poon VAF,
        // IKF), where dV/dp tends to 0, while the step lands on a finite p
        // and swaps the term in whole: the quotient is roundoff-dominated
        // and flips sign with line order (#91). A smooth parameter's
        // quotient barely moves when the step doubles; this one's halves or
        // worse, so it reports 0 as ngspice does.
        if (orig == 0 and out.* != 0) {
            const twice = try adj.fd(ckt, x_op, p, 2 * fdStep(0));
            if (@abs(twice - out.*) > 0.25 * @abs(out.*)) out.* = 0;
        }
    }
    return sens;
}

/// Returns the forward-difference step for a parameter at `orig`: ngspice's
/// relative 1e-6 (cktsens.c `Sens_Delta`), absolute 1e-12 only at 0. An
/// absolute floor added to the relative step would move a diode IS = 1e-14
/// by 100x (#85). `.sens`, `.dcmatch` and `.acmatch` share it.
pub fn fdStep(orig: f64) f64 {
    return if (orig == 0) 1e-12 else 1e-6 * @abs(orig);
}

/// Returns whether `.sens` lists parameter `name` of a `type_name` device.
/// ngspice lists device and model parameters but no source waveform
/// (vsrc.c's table is `dc` and the port parameters) and no table entry,
/// which VerA flattens into one field per slot (`pwl_timesZ5b0Z5d`, #154).
pub fn isSensParam(type_name: []const u8, name: []const u8) bool {
    if (std.mem.indexOf(u8, name, "Z5b") != null) return false;
    const src = std.mem.eql(u8, type_name, "vsource") or std.mem.eql(u8, type_name, "isource");
    return !src or std.mem.eql(u8, name, "dc");
}

// ---- private ----

/// One `.sens` column name in the results arena `a`. The table rather than
/// `is_instance`: ngspice's IF_PRINCIPAL flag lives on the instance
/// parameter table, but VerA puts every Verilog-A `parameter` on Model. The
/// `v(...)` wrapper is ngspice's too: cktsens.c hands the raw writer a
/// UID_OTHER name, which types as a voltage. The instance's $mfactor is
/// ngspice's instance parameter `m`.
fn columnName(a: Allocator, scratch: Allocator, ctx: *const root.RunCtx, opts: Options, ref: root.ParamRef) Allocator.Error![]const u8 {
    const type_name = ctx.circuit.typeName(ref.type);
    const card = opts.card_names.get(ref.type, ref.index);
    const dev = card orelse try scratch.print("{s}#{d}", .{ type_name, ref.index });
    defer if (card == null) scratch.free(dev);

    const pname = if (std.mem.eql(u8, ref.param_name, "mfactor")) "m" else ref.param_name;
    const is_principal = if (principal.get(type_name)) |pp| std.mem.eql(u8, ref.param_name, pp) else false;
    if (is_principal) return a.print("v({s})", .{dev});
    // ngspice prints keywords in lower case; gummel_poon spells `VAR` upper
    // case because `var` is a Zig keyword.
    const s = try a.print("v({s}{s}{s})", .{ dev, if (ref.is_instance) "_" else ":", pname });
    return std.ascii.lowerString(s, s);
}

test fdStep {
    // Oracle: cktsens.c's Sens_Delta, relative 1e-6 and 1e-12 only at 0.
    try std.testing.expectEqual(@as(f64, 1e-12), fdStep(0));
    try std.testing.expectEqual(@as(f64, 1e-6 * 1e-14), fdStep(-1e-14));
}
