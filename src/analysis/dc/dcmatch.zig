//! DC mismatch (Spectre `dcmatch`, HSPICE `.dcmatch`) and DC variation
//! sensitivity (HSPICE `.dcsens`) of one output at the operating point, by
//! adjoint sensitivity on `.sens`'s `Adjoint`: one factorization and one
//! transpose solve give lambda (J^T lambda = e_out); each parameter then costs
//! one finite-difference re-eval of F(x_op) and a dot with lambda. The
//! parameters come in `Groups`: the deck's variation block, or every
//! parameter alone at its Pelgrom sigma, sigma^2(p) = A_P^2 / (W L). The
//! offset is sigma^2(y) = sum over groups of (dy/dsigma_g)^2.
//! In: a circuit, its operating point, the variation table. Out: one real
//! point per query.
//! Cites the Spectre and HSPICE `.dcmatch`/`.dcsens` manuals;
//! dev/analysis/dcmatch.md.
//!
//! Contents:
//! - `pub const Options = core.query.Dcmatch`
//! - `pub const Contribution = struct { group, sensitivity, variance_contrib }`
//! - `pub const MismatchResult = struct { contributions, total_sigma }`
//! - `pub const Groups = struct` (`Member`, `Error`, `init`, `deinit`, `count`, `members`, `label`)
//! - `pub const Refs = struct` (`init`, `deinit`)
//! - `pub const Sens = struct` (`Options`, `run`)
//! - `pub const test_access`
//! - `pub fn run(ctx: *const RunCtx, opts: Options) !Result`
//! - `pub fn solve(ckt, x_op, output_node, output_neg, refs, groups, gpa) !MismatchResult`

const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;
const Variations = @import("core").query.Variations;
const root = @import("../types.zig");
const sens = @import("../sweep/sens.zig");

/// Query options, defined in core/query.zig.
pub const Options = @import("core").query.Dcmatch;

/// One group's share of the output variance.
pub const Contribution = struct {
    /// Index into the `Groups` the result was solved over.
    group: u32,
    /// dy/dsigma with a variation block; dy/dp per parameter without one.
    sensitivity: f64,
    /// Its square times the group's sigma squared.
    variance_contrib: f64,
};

/// The mismatch of one output.
pub const MismatchResult = struct {
    /// In group order; owned by the `solve` allocator.
    contributions: []Contribution,
    /// The one-sigma output spread: the square root of the summed
    /// `variance_contrib`.
    total_sigma: f64,
};

/// The parameter groups a mismatch or sensitivity analysis sums over:
/// `Variations` from the deck's variation block, or, without one, every
/// parameter of `Circuit.collectParams` alone, its column dy/dp and its
/// sigma the Pelgrom one. Invariant: `starts` ascends from 0 to `list.len`
/// and `sigmas.len == count()`.
pub const Groups = struct {
    /// `count() + 1` offsets into `list`: group g is `list[starts[g]..starts[g + 1]]`.
    starts: []const u32,
    list: []Member,
    /// Per group, the sigma its column is multiplied by for its spread: 1
    /// when the column is already per sigma.
    sigmas: []f64,
    /// Per group, borrowed from `Variations`; empty without a variation block.
    labels: []const []const u8,

    /// A parameter ordinal and the factor its dy/dp enters its group with:
    /// its one-sigma step, or 1 without a variation block. Read together, so
    /// one row.
    pub const Member = struct { param: u32, scale: f64 };

    comptime {
        assert(@sizeOf(Member) == 16); // 4 bytes of padding after `param`.
    }

    /// Every way `init` fails; nothing stays allocated.
    pub const Error = Allocator.Error || error{
        /// A group names a parameter past `refs`, or `vars` is no
        /// well-formed group table (offsets that start at 0, never
        /// decrease and end at `params.len`, one sigma per parameter, one
        /// label per group when labelled).
        InvalidVariation,
    };

    /// Builds the groups over `refs`; free them with `deinit` and the same
    /// allocator. `labels` stays borrowed from `vars`.
    pub fn init(a: Allocator, refs: []const root.ParamRef, vars: Variations) Error!Groups {
        if (vars.starts.len == 0) return pelgromGroups(a, refs);
        // The table arrives from the query, so a bad shape is an error here
        // rather than an out-of-bounds slice in `members`.
        const n_groups = vars.starts.len - 1;
        if (vars.starts[0] != 0) return error.InvalidVariation;
        if (vars.starts[n_groups] != vars.params.len) return error.InvalidVariation;
        if (vars.sigmas.len != vars.params.len) return error.InvalidVariation;
        if (vars.labels.len != 0 and vars.labels.len != n_groups) return error.InvalidVariation;
        for (vars.starts[0..n_groups], vars.starts[1..]) |lo, hi| if (lo > hi) return error.InvalidVariation;

        const list = try a.alloc(Member, vars.params.len);
        errdefer a.free(list);

        for (list, vars.params, vars.sigmas) |*m, p, sg| {
            if (p >= refs.len) return error.InvalidVariation;
            m.* = .{ .param = p, .scale = sg };
        }
        const starts = try a.dupe(u32, vars.starts);
        errdefer a.free(starts);

        const sigmas = try a.alloc(f64, n_groups);
        @memset(sigmas, 1);
        const g: Groups = .{ .starts = starts, .list = list, .sigmas = sigmas, .labels = vars.labels };
        assert(g.count() == n_groups);
        return g;
    }

    /// Frees what `init` allocated; `a` must be the allocator `init` took.
    pub fn deinit(self: Groups, a: Allocator) void {
        a.free(self.starts);
        a.free(self.list);
        a.free(self.sigmas);
    }

    /// Returns the number of groups, 0 for an empty block or no parameters.
    pub fn count(self: Groups) usize {
        assert(self.starts.len >= 1);
        return self.starts.len - 1;
    }

    /// Returns the parameters group `g` moves together. Asserts `g < count()`.
    pub fn members(self: Groups, g: usize) []const Member {
        assert(g < self.count());
        return self.list[self.starts[g]..self.starts[g + 1]];
    }

    /// Returns group `g`'s column label: the variation label, or,
    /// unlabelled, `<type>#<index>.<param>` of its first member. Caller owns
    /// the string and frees it with `a`.
    pub fn label(self: Groups, a: Allocator, ckt: *const root.Circuit, refs: []const root.ParamRef, g: usize) Allocator.Error![]const u8 {
        if (self.labels.len != 0) return a.dupe(u8, self.labels[g]);
        const r = refs[self.members(g)[0].param];
        return a.print("{s}#{d}.{s}", .{ ckt.typeName(r.type), r.index, r.param_name });
    }
};

/// The refs a mismatch or sensitivity solves over, and `vars` numbered
/// into them: with a variation block only its parameters, in block order
/// (`Circuit.isolatedParams`, so the circuit's other instances keep sharing
/// their Model rows), else every parameter (`Circuit.collectParams`).
/// Free with `deinit` and the allocator `init` took.
pub const Refs = struct {
    refs: []const root.ParamRef,
    vars: Variations,
    owned: bool,

    /// Without a variation block `refs` borrows `Circuit.collectParams`'
    /// cached list; with one both slices are allocated with `a`.
    pub fn init(a: Allocator, ckt: *root.Circuit, vars: Variations) !Refs {
        if (vars.starts.len == 0) return .{ .refs = try ckt.collectParams(), .vars = vars, .owned = false };
        const refs = try ckt.isolatedParams(a, vars.params);
        errdefer a.free(refs);

        const params = try a.alloc(u32, vars.params.len);
        for (params, 0..) |*p, k| p.* = @intCast(k);
        var v = vars;
        v.params = params;
        assert(refs.len == params.len);
        return .{ .refs = refs, .vars = v, .owned = true };
    }

    /// Frees what `init` allocated; `a` must be the allocator `init` took.
    pub fn deinit(self: Refs, a: Allocator) void {
        if (!self.owned) return;
        a.free(self.refs);
        a.free(self.vars.params);
    }
};

/// `.dcsens`: DC sensitivity to the variation block's parameters.
pub const Sens = struct {
    /// Query options, defined in core/query.zig.
    pub const Options = @import("core").query.Dcsens;

    /// Contract entry: one real point, each group's dy/dsigma (dy/dp per
    /// parameter without a variation block) in group order.
    pub fn run(ctx: *const root.RunCtx, opts: Sens.Options) !root.Result {
        return publish(ctx, opts.output_node, opts.output_neg, opts.variations, .sensitivity);
    }
};

/// Private implementation access for the analysis test suite.
pub const test_access = if (@import("builtin").is_test) .{
    .pelgromSigma = pelgromSigma,
} else {};

const Readout = enum { mismatch, sensitivity };

/// Contract entry: one real point, `total_3sigma` followed by each group's
/// sensitivity (`<type>#<index>.<param>`, or the variation label) in
/// descending variance order.
pub fn run(ctx: *const root.RunCtx, opts: Options) !root.Result {
    return publish(ctx, opts.output_node, opts.output_neg, opts.variations, .mismatch);
}

/// Returns each group's contribution to the mismatch of v(output_node) -
/// v(output_neg) at `x_op`, over `refs` (`Circuit.collectParams`). Leaves
/// the planes at the last perturbed eval and the workspace factored at
/// J(x_op). A parameter whose step rounds away contributes 0.
/// `contributions` is owned by `gpa`. Asserts that every member indexes `refs`.
pub fn solve(
    ckt: *root.Circuit,
    x_op: []const f64,
    output_node: u32,
    output_neg: u32,
    refs: []const root.ParamRef,
    groups: Groups,
    gpa: Allocator,
) !MismatchResult {
    if (groups.count() == 0) return .{ .contributions = &.{}, .total_sigma = 0 };
    const adj = try sens.Adjoint.init(ckt, x_op, .{ .node = output_node, .neg = output_neg, .branch = null }, gpa);
    defer adj.deinit(gpa);

    const contributions = try gpa.alloc(Contribution, groups.count());
    errdefer gpa.free(contributions);

    var total_var: f64 = 0;
    for (contributions, 0..) |*contrib, index| {
        if (index != 0) try ckt.checkpoint(.{ .phase = .sweep, .completed = index, .total = contributions.len });
        // The first member is not added to 0, which would turn a -0 into +0.
        var s: f64 = 0;
        for (groups.members(index), 0..) |m, k| {
            assert(m.param < refs.len);
            const p = refs[m.param];
            const orig = p.get();
            const dydp = adj.fd(ckt, x_op, p, orig + sens.fdStep(orig)) catch |e| switch (e) {
                error.ZeroDelta => 0,
            };
            const d = dydp * m.scale;
            s = if (k == 0) d else s + d;
        }
        const sigma_p = groups.sigmas[index];
        const var_contrib = s * s * sigma_p * sigma_p;
        total_var += var_contrib;
        contrib.* = .{ .group = @intCast(index), .sensitivity = s, .variance_contrib = var_contrib };
    }
    assert(!(total_var < 0));
    return .{ .contributions = contributions, .total_sigma = @sqrt(total_var) };
}

// ---- private ----

/// `Groups.init` without a variation block: every parameter alone.
fn pelgromGroups(a: Allocator, refs: []const root.ParamRef) Allocator.Error!Groups {
    const starts = try a.alloc(u32, refs.len + 1);
    errdefer a.free(starts);

    const list = try a.alloc(Groups.Member, refs.len);
    errdefer a.free(list);

    const sigmas = try a.alloc(f64, refs.len);
    for (refs, starts[0..refs.len], list, sigmas, 0..) |r, *st, *m, *sg, i| {
        st.* = @intCast(i);
        m.* = .{ .param = @intCast(i), .scale = 1 };
        sg.* = pelgromSigma(r);
    }
    starts[refs.len] = @intCast(refs.len);
    return .{ .starts = starts, .list = list, .sigmas = sigmas, .labels = &.{} };
}

/// The parameter's mismatch sigma (not sigma^2): A_P / sqrt(W L).
fn pelgromSigma(ref: root.ParamRef) f64 {
    if (ref.pelgrom_ap > 0 and ref.area_wl > 0) return ref.pelgrom_ap / @sqrt(ref.area_wl);
    // The device count is drawn, not a process variable.
    if (std.mem.eql(u8, ref.param_name, "mfactor")) return 0.0;
    // ponytail: unit sigma when the parameter carries no Pelgrom data; wire
    // a per-model coefficient when a deck needs a real one.
    return 1.0;
}

/// Solves the groups and publishes them, as `.dcmatch` (the 3-sigma total
/// first, variance-sorted) or `.dcsens`.
fn publish(ctx: *const root.RunCtx, output_node: u32, output_neg: u32, vars: Variations, readout: Readout) !root.Result {
    const a = ctx.allocator;
    const ckt = ctx.circuit;
    const scratch = ctx.scratch_allocator;
    const sel = try Refs.init(scratch, ckt, vars);
    defer sel.deinit(scratch);

    const groups = try Groups.init(scratch, sel.refs, sel.vars);
    defer groups.deinit(scratch);

    const res = try solve(ckt, ctx.x_op, output_node, output_neg, sel.refs, groups, scratch);
    defer scratch.free(res.contributions);

    const mismatch = readout == .mismatch;
    if (mismatch) std.mem.sort(Contribution, res.contributions, {}, moreVariance);
    const head: usize = @intFromBool(mismatch);
    const ncols = head + res.contributions.len;
    const names = try a.alloc([]const u8, ncols);
    errdefer a.free(names);

    if (mismatch) names[0] = "total_3sigma";
    var done: usize = 0;
    errdefer for (names[head..][0..done]) |s| a.free(s);
    for (res.contributions, names[head..]) |c, *name| {
        name.* = try groups.label(a, ckt, sel.refs, c.group);
        done += 1;
    }
    const data = try a.alloc(f64, ncols);
    if (mismatch) data[0] = 3.0 * res.total_sigma;
    for (res.contributions, data[head..]) |c, *out| out.* = c.sensitivity;
    return .{
        .plotname = if (mismatch) "DC Mismatch" else "DC Sensitivity",
        .varnames = names,
        .is_complex = false,
        .npoints = 1,
        .data = data,
    };
}

fn moreVariance(_: void, x: Contribution, y: Contribution) bool {
    return y.variance_contrib < x.variance_contrib;
}
