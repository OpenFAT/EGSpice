//! Monte Carlo (`.mc`): gaussian variation of every device's primary
//! parameter, one cold DC solve per trial. Each trial is a structural sweep
//! lane (`lanes.solveLanes`) sharing one Newton workspace. Each draw is
//! keyed by (Options.seed, trial, variable), so a run is reproducible and
//! any trial can be computed alone. With `Options.variants` the lanes are
//! deck variants instead, each warm from the nominal OP.
//! In: a run context and the trial or variant range. Out: real point-major
//! rows (run or axis value, probes...), one per converged lane.
//! Cites no outside source.
//!
//! Contents:
//! - `pub const Options = core.query.Mc`
//! - `pub const Setup = union(enum)` (`apply`, `restore`, `deinit`)
//! - `pub fn run(ctx: *const RunCtx, opts: Options) !Result`
//! - `pub fn setup(ctx: *const RunCtx, opts: Options) !Setup`
//! - `pub fn range(opts: Options) lanes.Range`
//! - `pub fn warm(ctx: *const RunCtx, opts: Options) ?[]const f64`
//! - `pub fn publish(ctx, opts, first: usize, probe_rows: []const f64, results: []const converger.Result) !Result`

const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;
const converger = @import("solver").converger;
const root = @import("../types.zig");
const lanes = @import("lanes.zig");

/// Query options, defined in core/query.zig.
pub const Options = @import("core").query.Mc;

/// The lanes' installer: random trials, or deck variants.
pub const Setup = union(enum) {
    trials: Trials,
    variants: VariantRows,

    /// Installs lane `k`.
    pub fn apply(self: *Setup, k: usize) !void {
        switch (self.*) {
            inline else => |*s| try s.apply(k),
        }
    }

    /// Puts the nominal parameters back.
    pub fn restore(self: *Setup) !void {
        switch (self.*) {
            inline else => |*s| try s.restore(),
        }
    }

    /// Frees the installer's table; `scratch` is the allocator `setup` took.
    pub fn deinit(self: *Setup, scratch: Allocator) void {
        switch (self.*) {
            .trials => |s| scratch.free(s.vars),
            .variants => |s| scratch.free(s.saved),
        }
    }
};

/// One varied parameter and the nominal value its trials perturb.
const ParamVar = struct {
    ref: root.ParamRef,
    nominal: f64,
};

/// Trial k draws `nominal (1 + variation N(0,1))` for every parameter.
const Trials = struct {
    vars: []const ParamVar,
    variation: f64,
    seed: u64,

    fn apply(self: *Trials, k: usize) !void {
        for (self.vars, 0..) |pv, v| pv.ref.set(pv.nominal + pv.nominal * self.variation * draw(self.seed, k, v));
    }

    fn restore(self: *Trials) !void {
        for (self.vars) |pv| pv.ref.set(pv.nominal);
    }
};

/// Lane k installs deck variant `first + k`, after putting back the
/// parameters the previous lane wrote. The writes go through
/// `Circuit.setParams`, as a `.step` point's do, so instances keep sharing
/// their Model rows where the variant writes them alike.
const VariantRows = struct {
    ckt: *root.Circuit,
    variants: @import("core").Variants,
    first: u32,
    /// The refs the installed lane wrote and their values before it.
    written: []const u32 = &.{},
    saved: []f64,

    fn apply(self: *VariantRows, k: usize) !void {
        try self.restore();
        const refs, const values = self.variants.writes(self.first + @as(u32, @intCast(k)));
        assert(refs.len <= self.saved.len);
        // All read before any is written, so a ref written twice saves its
        // nominal both times.
        for (refs, self.saved[0..refs.len]) |r, *old| old.* = (try self.ckt.paramAt(r)).value;
        try self.ckt.setParams(refs, values, false);
        self.written = refs;
    }

    fn restore(self: *VariantRows) !void {
        try self.ckt.setParams(self.written, self.saved[0..self.written.len], false);
        self.written = &.{};
    }
};

/// Contract entry: real, point-major (run, probes...), one row per
/// converged trial and no rows when there are no probes. Varies each
/// device's primary parameter, skipping those that are zero, or with
/// `opts.variants` solves those deck variants, each warm-started from the
/// nominal OP (`ctx.x_op`), the first column their axis value. With
/// `opts.lane` only that trial or variant runs (a row the session split
/// off), its run column 0. Parameters are back at their nominals afterwards
/// so later jobs see the netlist's circuit.
pub fn run(ctx: *const root.RunCtx, opts: Options) !root.Result {
    return lanes.run(@This(), ctx, opts);
}

/// Returns the installer for `ctx`'s circuit: each device's nonzero primary
/// parameter at its nominal, or the variants' widest write's save slots.
/// Free it with `Setup.deinit` and the scratch allocator.
pub fn setup(ctx: *const root.RunCtx, opts: Options) !Setup {
    const scratch = ctx.scratch_allocator;
    if (opts.variants.count != 0) {
        const rows = opts.variants;
        var widest: usize = 0;
        for (rows.first..rows.first + rows.count) |v| widest = @max(widest, ctx.variants.writes(@intCast(v))[0].len);
        return .{ .variants = .{ .ckt = ctx.circuit, .variants = ctx.variants, .first = rows.first, .saved = try scratch.alloc(f64, widest) } };
    }
    const refs = try ctx.circuit.collectParams();
    var n_vars: usize = 0;
    for (refs) |ref| n_vars += @intFromBool(ref.primary and ref.get() != 0);
    const vars = try scratch.alloc(ParamVar, n_vars);
    var i: usize = 0;
    for (refs) |ref| {
        if (!ref.primary or ref.get() == 0) continue;
        vars[i] = .{ .ref = ref, .nominal = ref.get() };
        i += 1;
    }
    assert(i == n_vars);
    return .{ .trials = .{ .vars = vars, .variation = opts.variation, .seed = opts.seed } };
}

/// Returns the trials or variants `opts` solves.
pub fn range(opts: Options) lanes.Range {
    return .of(opts.lane, if (opts.variants.count != 0) opts.variants.count else opts.n_trials);
}

/// Returns the nominal OP for variants, which warm-start from it; trials
/// start cold.
pub fn warm(ctx: *const root.RunCtx, opts: Options) ?[]const f64 {
    return if (opts.variants.count != 0) ctx.x_op else null;
}

/// Returns one row per converged lane of those from `first` (`probe_rows`
/// holds each lane's probe values): its axis value, the variant's or the
/// converged ordinal, then the probes. Asserts one probe row per result.
pub fn publish(ctx: *const root.RunCtx, opts: Options, first: usize, probe_rows: []const f64, results: []const converger.Result) !root.Result {
    const np = ctx.probes.len;
    assert(probe_rows.len == results.len * np);
    var n_conv: usize = 0;
    if (np > 0) {
        for (results) |r| n_conv += @intFromBool(r.converged);
    }
    const variants = opts.variants.count != 0;
    const names = try root.probeNames(ctx, if (variants) opts.axis else "run");
    const ncols = names.len;
    const data = try ctx.allocator.alloc(f64, n_conv * ncols);
    var k: usize = 0;
    if (n_conv > 0) for (results, 0..) |r, t| {
        if (!r.converged) continue;
        const row = data[k * ncols ..][0..ncols];
        row[0] = if (variants) ctx.variants.axis[opts.variants.first + first + t] else @floatFromInt(k);
        @memcpy(row[1..], probe_rows[t * np ..][0..np]);
        k += 1;
    };
    assert(k == n_conv);
    return .{
        .plotname = if (variants and opts.dc_plot) "DC transfer characteristic" else "Monte Carlo",
        .varnames = names,
        .is_complex = false,
        .npoints = n_conv,
        .data = data,
    };
}

// ---- private ----

/// N(0,1) for variable `variable` of trial `trial`: a counter-based draw, a
/// fresh generator seeded by the hash of the three, so it depends on no
/// other draw.
fn draw(seed: u64, trial: usize, variable: usize) f64 {
    const key = [2]u64{ trial, variable };
    var prng = std.Random.DefaultPrng.init(std.hash.Wyhash.hash(seed, std.mem.asBytes(&key)));
    return prng.random().floatNorm(f64);
}

test draw {
    // Oracle: the counter-based contract itself; a draw depends only on its
    // key, so recomputing one alone or out of order gives the same bits.
    const a = draw(7, 3, 1);
    _ = draw(7, 0, 0);
    try std.testing.expectEqual(@as(u64, @bitCast(a)), @as(u64, @bitCast(draw(7, 3, 1))));
    try std.testing.expect(a != draw(7, 3, 2));
    try std.testing.expect(a != draw(8, 3, 1));
}
