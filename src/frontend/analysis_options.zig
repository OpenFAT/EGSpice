//! `.options` and `.temp` cards to deck options. In: the deck's config
//! cards in order. Out: one `DeckOptions` (tolerances, method, temperatures,
//! limits), and its copy into each query.
//! Cites: ngspice cktsopt.c, cktntask.c, traninit.c; HSPICE [SA] Ch.20
//! (TNOM), [CR] .OPTION RUNLVL.
//!
//! Contents:
//! - `pub const DeckOptions`
//! - `pub fn deckOptions(config: []const netlist.Config, dialect: netlist.Dialect) !DeckOptions`
//! - `pub fn applyDeckOptions(job: *Job, o: DeckOptions) void`

const std = @import("std");
const assert = std.debug.assert;
const testing = std.testing;
const Allocator = std.mem.Allocator;
const core = @import("core");
const requests = core.query;
const numerics = core.numerics;
const netlist = @import("netlist");
const Value = netlist.Value;
const nameAt = Value.nameAt;
const numberAt = Value.numberAt;
const Job = requests.Query;
const GROUND = core.GROUND;
const Library = @import("device").Library;
const analysis_args = @import("analysis_args.zig");

/// Parsed `.options` overrides, in deck order: a later card wins.
pub const DeckOptions = struct {
    /// The `.options` tolerances and iteration limits, copied into every
    /// query by `applyDeckOptions`.
    tol: numerics.Tolerances = .{},
    /// `.options method`; null keeps each query's default.
    method: ?requests.Method = null,
    /// `.temp` or `.options temp` in degrees Celsius; null when not given.
    temp_c: ?f64 = null,
    /// An HSPICE `.temp t1 t2 ...` list: every query runs at each.
    temp_list: []const Value = &.{},
    /// `.options delmax`: the transient step cap when the `.tran` card sets
    /// none.
    delmax: ?f64 = null,
    /// `.options tran_ceiling=duration|compatibility`; null keeps the
    /// `.tran` query's own policy.
    tran_ceiling: ?@FieldType(requests.Tran, "ceiling_policy") = null,
    /// `.options xmu`: the trapezoidal damping weight; null keeps 0.5.
    xmu: ?f64 = null,
    /// `.options tnom=<degC>`: the temperature model cards were extracted at
    /// (ngspice cktsopt.c:71-73 takes the card in Celsius; default 27 degC
    /// from cktntask.c:127). Unlike `temp`, where the circuit runs, it reaches
    /// the devices at build time, so no sweep can move it.
    tnom_c: f64 = 27.0,
    /// HSPICE `.options runlvl`, `accurate` or `fast`: the transient's LTE
    /// factor `trtol` (see `runlvl_trtol`); null when none is set or
    /// `runlvl=0`.
    trtol: ?f64 = null,
    /// The variation block as `.dcmatch`, `.acmatch` and `.dcsens` groups.
    variations: requests.Variations = .{},
    /// HSPICE `.sample`, set on every `.noise` spectrum.
    sample: ?requests.NoiseSample = null,
    /// The deck's card names, for `.noise` and `.sens` labels.
    card_names: requests.CardNames = .empty,
};

/// `trtol` at HSPICE RUNLVL 1..6 [CR .OPTION RUNLVL]. HSPICE scales its
/// tolerances by level without publishing the factors, so these are
/// ESPice's: level 3 is the default 7 and each level halves or doubles the
/// LTE bound (dev/analysis/tolerance-system.md has the measurements).
const runlvl_trtol = [6]f64{ 28, 14, 7, 3.5, 1.75, 0.875 };

/// Option names outside this list (and the ones `foldOptions` passes on to
/// other readers) are not simulated, and each draws a warning in every
/// dialect. `gshunt`, `cshunt` and `rshunt` are read by the netlist.
const Option = enum(u8) { method, tran_ceiling, reltol, abstol, vntol, gmin, trtol, chgtol, itl1, itl2, itl4, maxord, temp, tnom, delmax, xmu, gshunt, cshunt, rshunt };

/// Folds the deck's `.options` and `.temp` cards; `InvalidAnalysisArguments`
/// on a value out of range, logged with its card. The last `.temp` card
/// wins, whether it is one value or a list. The HSPICE dialect defaults TNOM to 25 degC and
/// runs the circuit at TNOM [SA Ch.20]; ngspice defaults both to 27 degC.
pub fn deckOptions(config: []const netlist.Config, dialect: netlist.Dialect) !DeckOptions {
    // The card being folded, named by any error it raises.
    var line: []const u8 = "";
    return foldOptions(config, dialect, &line) catch |err| {
        // The test runner fails any test that logs an error.
        if (!@import("builtin").is_test) std.log.err("options: {s}: {s}", .{ line, @errorName(err) });
        return err;
    };
}

/// Copies the deck tolerances into `job`, and into a `.tran` also the
/// integration method, XMU and RUNLVL's TRTOL, each only when set. A
/// `.tran` keeps a `dt_max` it already has; otherwise it takes DELMAX,
/// then ngspice's min(tstep, (tstop - tstart) / 50).
pub fn applyDeckOptions(job: *Job, o: DeckOptions) void {
    switch (job.*) {
        inline else => |*opts| {
            if (comptime @hasField(@TypeOf(opts.*), "tol")) opts.tol = o.tol;
            if (comptime @hasField(@TypeOf(opts.*), "dc_options")) opts.dc_options.tol = o.tol;
        },
    }
    if (job.* == .tran) {
        const t = &job.tran;
        if (o.method) |m| t.method = m;
        if (o.xmu) |x| t.xmu = x;
        // RUNLVL overrides TRTOL wherever it sits [CR .OPTION RUNLVL].
        if (o.trtol) |trtol| t.tol.trtol = trtol;
        // ngspice's default tmax, min(tstep, (tstop - tstart) / 50)
        // (traninit.c:29-34).
        if (o.tran_ceiling) |policy| t.ceiling_policy = policy;
        t.dt_max = t.dt_max orelse o.delmax;
        t.dt_max = t.requestedCeiling(.deck);
    }
}

// ---- private ----

fn foldOptions(config: []const netlist.Config, dialect: netlist.Dialect, line: *[]const u8) !DeckOptions {
    // HSPICE spellings: ABSV/RELV/ABSI are VNTOL/RELTOL/ABSTOL, GMINDC the DC
    // gmin (ESPice has one gmin), METHOD=BDF its Gear.
    const names = std.StaticStringMap(Option).initComptime(.{
        .{ "method", .method }, .{ "reltol", .reltol },             .{ "abstol", .abstol },
        .{ "vntol", .vntol },   .{ "gmin", .gmin },                 .{ "trtol", .trtol },
        .{ "chgtol", .chgtol }, .{ "itl1", .itl1 },                 .{ "itl2", .itl2 },
        .{ "itl4", .itl4 },     .{ "maxord", .maxord },             .{ "temp", .temp },
        .{ "tnom", .tnom },     .{ "absv", .vntol },                .{ "relv", .reltol },
        .{ "absi", .abstol },   .{ "gmindc", .gmin },               .{ "delmax", .delmax },
        .{ "gshunt", .gshunt }, .{ "cshunt", .cshunt },             .{ "xmu", .xmu },
        .{ "rshunt", .rshunt }, .{ "tran_ceiling", .tran_ceiling },
    });
    const methods = std.StaticStringMap(requests.Method).initComptime(.{
        .{ "gear", .gear_2 }, .{ "trap", .trapezoidal }, .{ "trapezoidal", .trapezoidal }, .{ "bdf", .gear_2 },
    });
    var o: DeckOptions = .{ .tnom_c = if (dialect == .hspice) 25 else 27 };
    var maxord: ?f64 = null;
    var runlvl: ?f64 = null;
    var accurate = false;
    var fast = false;
    for (config) |card| {
        line.* = card.line;
        const args = card.args;
        if (card.temp) {
            // The last `.temp` card wins, list or single value.
            if (args.len > 1) {
                o.temp_list = args;
                o.temp_c = null;
                continue;
            }
            o.temp_c = try analysis_args.number(args, 0);
            o.temp_list = &.{};
            if (o.temp_c.? <= -273.15) return error.InvalidAnalysisArguments;
            continue;
        }
        var lower: [16]u8 = undefined;
        var i: usize = 0;
        while (i < args.len) : (i += 1) {
            const key = nameAt(args, i) orelse continue;
            // Monte Carlo draws read these (frontend/variants.zig), EM_AVG
            // measures the third (frontend/measure.zig), frontend/source.zig
            // `search`, and model binning `scale` and `wnflag` (netlist.zig).
            if (std.ascii.eqlIgnoreCase(key, "seed") or std.ascii.eqlIgnoreCase(key, "sampling_method") or std.ascii.eqlIgnoreCase(key, "em_recovery") or
                std.ascii.eqlIgnoreCase(key, "search") or std.ascii.eqlIgnoreCase(key, "scale") or std.ascii.eqlIgnoreCase(key, "wnflag"))
            {
                i += 1;
                continue;
            }
            // RUNLVL alone is level 3; ACCURATE and FAST are flags with an
            // optional 0|1.
            const preset = std.ascii.eqlIgnoreCase(key, "runlvl") or std.ascii.eqlIgnoreCase(key, "accurate") or std.ascii.eqlIgnoreCase(key, "fast");
            if (preset) {
                const given = numberAt(args, i + 1);
                if (given != null) i += 1;
                const on = (given orelse 1) != 0;
                if (std.ascii.eqlIgnoreCase(key, "accurate")) accurate = on else if (std.ascii.eqlIgnoreCase(key, "fast")) fast = on else {
                    const level = @floor(0.5 + (given orelse 3));
                    // Negated so a NaN level fails here, not in @intFromFloat.
                    if (!(level >= 0 and level <= 6)) return error.InvalidAnalysisArguments;
                    runlvl = level;
                }
                continue;
            }
            const known = if (key.len <= lower.len) names.get(std.ascii.lowerString(lower[0..key.len], key)) else null;
            const option = known orelse {
                std.log.warn("options: ignoring unsupported option '{s}'", .{key});
                continue;
            };
            i += 1;
            if (option == .tran_ceiling) {
                const value = nameAt(args, i) orelse return error.InvalidAnalysisArguments;
                if (std.ascii.eqlIgnoreCase(value, "duration")) {
                    o.tran_ceiling = .duration;
                } else if (std.ascii.eqlIgnoreCase(value, "compatibility")) {
                    o.tran_ceiling = .compatibility;
                } else return error.InvalidAnalysisArguments;
                continue;
            }
            if (option == .method) {
                const method = nameAt(args, i) orelse return error.InvalidAnalysisArguments;
                if (method.len > lower.len) return error.InvalidAnalysisArguments;
                o.method = methods.get(std.ascii.lowerString(lower[0..method.len], method)) orelse return error.InvalidAnalysisArguments;
                continue;
            }
            const raw = try analysis_args.number(args, i);
            // ngspice reads an IF_INTEGER option as floor(0.5 + x)
            // (spiceif.c:501, inpgval.c:31), so itl4=1.5 is 2.
            const value = switch (option) {
                .itl1, .itl2, .itl4, .maxord => @floor(0.5 + raw),
                else => raw,
            };
            switch (option) {
                .method, .tran_ceiling => unreachable,
                .temp, .tnom => {
                    if (value <= -273.15) return error.InvalidAnalysisArguments;
                    if (option == .temp) o.temp_c = value else o.tnom_c = value;
                },
                .itl1, .itl2, .itl4 => {
                    if (value < 1 or value > std.math.maxInt(u16)) return error.InvalidAnalysisArguments;
                    const iterations: u16 = @intFromFloat(value);
                    switch (option) {
                        .itl1 => o.tol.itl1 = iterations,
                        .itl2 => o.tol.itl2 = iterations,
                        .itl4 => o.tol.itl4 = iterations,
                        else => unreachable,
                    }
                },
                .maxord => {
                    if (value < 1) return error.InvalidAnalysisArguments;
                    maxord = value;
                },
                .delmax => {
                    if (!(value > 0)) return error.InvalidAnalysisArguments;
                    o.delmax = value;
                },
                .xmu => {
                    if (!(value >= 0 and value <= 0.5)) return error.InvalidAnalysisArguments;
                    o.xmu = value;
                },
                .gshunt, .cshunt, .rshunt => if (value < 0) return error.InvalidAnalysisArguments,
                // A zero reltol or trtol leaves no step a transient can take
                // (ngspice aborts "timestep too small"), so the card fails here.
                .reltol, .trtol => {
                    if (!(value > 0)) return error.InvalidAnalysisArguments;
                    if (option == .reltol) o.tol.reltol = value else o.tol.trtol = value;
                },
                inline else => |field| {
                    if (value < 0) return error.InvalidAnalysisArguments;
                    @field(o.tol, @tagName(field)) = value;
                },
            }
        }
    }
    // ngspice raises the order only from 1 to 2 and only when maxord > 1
    // (dctran.c:900), so maxord=1 is backward Euler under trap as under gear,
    // and gear with maxord 3..6 stays Gear-2.
    if (maxord != null and maxord.? < 2) o.method = .backward_euler;
    // ACCURATE lifts any level to at least 5; FAST is level 1 when no
    // RUNLVL is given; RUNLVL=0 turns the presets off.
    var level = runlvl orelse if (fast and !accurate) @as(f64, 1) else 0;
    if (accurate and runlvl != 0) level = @max(level, 5);
    if (level != 0) o.trtol = runlvl_trtol[@as(usize, @intFromFloat(level)) - 1];
    if (dialect == .hspice and o.temp_c == null and o.temp_list.len == 0) o.temp_c = o.tnom_c;
    return o;
}
