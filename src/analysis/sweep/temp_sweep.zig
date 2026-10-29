//! Temperature sweep (`.temp`): one structural lane per temperature, using
//! the devices' own temperature physics (`Circuit.setCircuitTemp`), no
//! external coefficients.
//! In: a run context and the temperature range. Out: real point-major rows
//! (temp, probes...), one per converged temperature.
//! Cites no outside source.
//!
//! Contents:
//! - `pub const Options = core.query.Temp`
//! - `pub const Setup = struct` (`apply`, `restore`, `deinit`)
//! - `pub fn run(ctx: *const RunCtx, opts: Options) !Result`
//! - `pub fn numPoints(options: Options) u32`
//! - `pub fn setup(ctx: *const RunCtx, opts: Options) !Setup`
//! - `pub fn range(opts: Options) lanes.Range`
//! - `pub fn warm(_: *const RunCtx, _: Options) ?[]const f64`
//! - `pub fn publish(ctx, opts, first: usize, probe_rows: []const f64, results: []const converger.Result) !Result`

const std = @import("std");
const assert = std.debug.assert;
const converger = @import("solver").converger;
const root = @import("../types.zig");
const lanes = @import("lanes.zig");

/// Query options, defined in core/query.zig.
pub const Options = @import("core").query.Temp;

/// Lane k runs at t_start + k t_step; `restore` reinstalls the temperature
/// the circuit had before the sweep (`Circuit.temp_c`).
pub const Setup = struct {
    ckt: *root.Circuit,
    t_start: f64,
    t_step: f64,
    temp_c: f32,

    /// Installs lane `k`'s temperature.
    pub fn apply(self: *Setup, k: usize) !void {
        self.ckt.setCircuitTemp(@floatCast(self.t_start + @as(f64, @floatFromInt(k)) * self.t_step));
    }

    /// Reinstalls the temperature from before the sweep.
    pub fn restore(self: *Setup) !void {
        self.ckt.setCircuitTemp(self.temp_c);
    }

    /// Frees nothing: the `lanes.run` protocol's hook.
    pub fn deinit(_: *Setup, _: std.mem.Allocator) void {}
};

/// Contract entry: real, point-major (temp, probes...), one row per
/// converged temperature, each solved cold; with `opts.lane` only that
/// temperature (a row the session split off). The circuit is back at the
/// temperature it ran at before the sweep, so later jobs see the netlist's
/// (or their query's) temperature.
pub fn run(ctx: *const root.RunCtx, opts: Options) !root.Result {
    return lanes.run(@This(), ctx, opts);
}

/// Returns the points from t_start to t_stop by t_step, endpoint inclusive;
/// 1 for a non-positive step, 0 for a descending span.
pub fn numPoints(options: Options) u32 {
    if (options.t_step <= 0) return 1;
    const span = options.t_stop - options.t_start;
    if (span < 0) return 0;
    // The 1e-6-step nudge of `.dc`'s sweepCount: an exact-integer ratio
    // arrives just under itself in f64 and a bare floor drops the endpoint.
    return @as(u32, @intFromFloat(@floor(span / options.t_step + 1e-6))) + 1;
}

/// Returns the installer for `ctx`'s circuit, saving its current
/// temperature for `restore`. Never fails; the error union is the
/// `lanes.run` protocol's.
pub fn setup(ctx: *const root.RunCtx, opts: Options) !Setup {
    return .{ .ckt = ctx.circuit, .t_start = opts.t_start, .t_step = opts.t_step, .temp_c = ctx.circuit.temp_c };
}

/// Returns the temperatures `opts` solves.
pub fn range(opts: Options) lanes.Range {
    return .of(opts.lane, numPoints(opts));
}

/// Returns null: every temperature starts cold.
pub fn warm(_: *const root.RunCtx, _: Options) ?[]const f64 {
    return null;
}

/// Returns one row per converged temperature of those from lane `first`.
/// Asserts one probe row per result.
pub fn publish(ctx: *const root.RunCtx, opts: Options, first: usize, probe_rows: []const f64, results: []const converger.Result) !root.Result {
    const np = ctx.probes.len;
    assert(probe_rows.len == results.len * np);
    const ncols = np + 1;
    var npoints: usize = 0;
    for (results) |r| npoints += @intFromBool(r.converged);
    const data = try ctx.allocator.alloc(f64, npoints * ncols);
    var pt: usize = 0;
    for (results, 0..) |r, i| {
        if (!r.converged) continue;
        const row = data[pt * ncols ..][0..ncols];
        row[0] = opts.t_start + @as(f64, @floatFromInt(first + i)) * opts.t_step;
        @memcpy(row[1..], probe_rows[i * np ..][0..np]);
        pt += 1;
    }
    assert(pt == npoints);
    return .{
        .plotname = "Temperature Sweep",
        .varnames = try root.probeNames(ctx, "temp"),
        .is_complex = false,
        .npoints = npoints,
        .data = data,
    };
}
