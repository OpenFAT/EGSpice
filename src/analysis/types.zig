//! Shared analysis context: the circuit, the run context and the result type
//! every analysis leaf needs. Leaves import this file, never ../root.zig,
//! which imports the leaves for dispatch: Circuit.zig (with gpu/) ->
//! types.zig -> leaves -> executor.zig -> session.zig -> root.zig.
//! In: nothing (declarations). Out: the names every leaf shares.
//! Cites Verilog-AMS LRM §4.6.1 (`analysis()`).
//!
//! Contents:
//! - `pub const Circuit`, `Knobs`, `EvalHook`, `Display`, `AcParam`, `GROUND`, `freeze`
//! - `pub const CardNames`, `ParamRef`, `NoiseSource`, `AnalysisKind`, `Result`
//! - `pub const RunCtx = struct`
//! - `pub fn ErrorOf(comptime f: anytype) type`
//! - `pub fn outputDrive(e: []f64, pos: u32, neg: u32) void`
//! - `pub fn take(arena: []f64, off: *usize, len: usize) []f64`
//! - `pub fn probeNames(ctx: *const RunCtx, first: ?[]const u8) Allocator.Error![]const []const u8`

const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;
const core = @import("core");
const abi = @import("device").abi;
const circuit_mod = @import("Circuit.zig");

pub const Circuit = circuit_mod.Circuit;
pub const Knobs = circuit_mod.Knobs;
pub const EvalHook = circuit_mod.EvalHook;
pub const Display = circuit_mod.Display;
pub const AcParam = circuit_mod.AcParam;
pub const GROUND = circuit_mod.GROUND;
/// Builds an owning Circuit from protos, for tests (`circuit_life.freeze`).
pub const freeze = circuit_mod.init;
pub const CardNames = core.query.CardNames;
pub const ParamRef = abi.ParamRef;
pub const NoiseSource = abi.NoiseSource;
/// §4.6.1 `analysis()`: the analysis a `Circuit.setSimState` publishes.
pub const AnalysisKind = @FieldType(abi.SimState, "kind");
pub const Result = core.Result;

/// Everything an analysis needs, resolved before dispatch.
pub const RunCtx = struct {
    circuit: *Circuit,
    /// The operating point. The session solves one ahead of every query
    /// except `.op` (the executor solves it) and a `uic` start (the executor
    /// seeds ICs).
    x_op: []f64,
    probes: []const u32,
    /// Raw column label per probe, parallel to `probes`. May be empty in
    /// hand-built contexts; see `probeNames`.
    probe_labels: []const []const u8 = &.{},
    source_node: u32,
    source_branch: u32,
    /// The deck's `.nodeset` rows; the DC sweep forces them at each cold
    /// point as the operating point does.
    nodeset: []const core.Ic = &.{},
    /// Composite AC excitation `[re(0..n), im(0..n)]`, length `2 * circuit.n`,
    /// summed over every source card with an `AC mag [phase]`. An .ac sweep
    /// is one solve per frequency against this drive. Empty means no AC
    /// source, which gives a zero response, as in ngspice.
    ac_drive: []const f64 = &.{},
    /// The deck's circuit variants, for queries that sweep them as lanes.
    variants: core.Variants = .{},
    /// Results arena: everything in the returned Result lives here.
    allocator: Allocator,
    /// Reclaimable work storage.
    scratch_allocator: Allocator,
    /// Where a transient or AC sweep writes its rows instead of keeping them
    /// (`Session.stream`); its Result then has empty `data`.
    stream: ?*std.Io.Writer = null,
    /// The deck's `.meas` cards: a transient's `point` options read them.
    measures: []const core.Measure = &.{},
    /// The leading frequency sweep's solutions, solved up front by a lane
    /// batch (`Executor.runFreqLanes`) as `freq.solveAll` lays them out;
    /// `.ac`, `.acxf`, `.stb` and `.noise` replay them (`freq.Stream.pre`).
    /// Empty: solve live.
    freq_pre: []const f64 = &.{},
};

/// Returns the error set `f` returns, so a module boundary can name an
/// inferred set.
pub fn ErrorOf(comptime f: anytype) type {
    return @typeInfo(@typeInfo(@TypeOf(f)).@"fn".return_type.?).error_union.error_set;
}

/// Writes `e` := the adjoint excitation of the output v(pos) - v(neg): +1
/// at `pos`, -1 at `neg`, 0 elsewhere. The ground row is the v(0) = 0 clamp
/// and never takes one. Asserts that both rows lie in `e`.
pub fn outputDrive(e: []f64, pos: u32, neg: u32) void {
    assert(pos < e.len);
    assert(neg < e.len);
    @memset(e, 0);
    if (pos != GROUND) e[pos] = 1;
    if (neg != GROUND) e[neg] = -1;
}

/// Returns the next `len` values of `arena` at `off.*` and advances `off`:
/// one scratch block split into named slices. Asserts that they fit.
pub fn take(arena: []f64, off: *usize, len: usize) []f64 {
    assert(off.* + len <= arena.len);
    defer off.* += len;
    return arena[off.*..][0..len];
}

/// Returns one column name per probe, preceded by `first` when given.
/// Labeled names borrow `ctx.probe_labels` (deck storage that outlives every
/// result), and without `first` the slice is `probe_labels` itself.
/// Unlabeled contexts fall back to `v(<node name>)`, or `v(<row>)` for an
/// unnamed branch row so two such columns never share a name, allocated
/// from `ctx.allocator`. Callers free nothing: the results arena owns the
/// rest.
pub fn probeNames(ctx: *const RunCtx, first: ?[]const u8) Allocator.Error![]const []const u8 {
    const a = ctx.allocator;
    const labeled = ctx.probe_labels.len == ctx.probes.len;
    if (labeled and first == null) return ctx.probe_labels;
    const extra: usize = if (first == null) 0 else 1;
    const names = try a.alloc([]const u8, ctx.probes.len + extra);
    if (first) |name| names[0] = name;
    if (labeled) {
        @memcpy(names[extra..], ctx.probe_labels);
        return names;
    }
    for (ctx.probes, names[extra..]) |row, *out| {
        const label = ctx.circuit.nodeName(row);
        out.* = if (label.len == 0)
            try a.print("v({d})", .{row})
        else
            try a.print("v({s})", .{label});
    }
    assert(names.len == ctx.probes.len + extra);
    return names;
}
