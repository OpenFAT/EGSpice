//! Plots and the checks that decide whether a format can hold one: a deck
//! title over a `core.Result` in, nothing or a `ValidationError` out. Every
//! check is pure, so a caller can refuse a selection before any I/O. Writers
//! and `Session` import this leaf, never root.zig.
//! Cites: Touchstone 2.0 (complete S matrix, one reference per port),
//! CITIfile A.01.01 (`S[m,n]` data), and the fixed fields of fsdb.zig and
//! sst2.zig.
//!
//! Contents:
//! - `pub const Format = enum(u8) { binary, ascii, csv, touchstone, psf, fsdb, sst2, citi, print }`
//! - `pub const Selection = struct { format: Format, path: ?[]const u8 }`
//! - `pub const Schema = core.Schema`
//! - `pub const Result = core.Result`
//! - `pub const sst2_names_max: u32`
//! - `pub const Plot = struct` with `pub fn schema(Plot) Schema`, `pub fn stride(Plot) usize`,
//!   `pub fn point(Plot, usize) []const f64`
//! - `pub const ValidationError`
//! - `pub const QueryError`
//! - `pub fn validatePlot(format: Format, plot: Plot) ValidationError!void`
//! - `pub fn validateSchema(format: Format, schema: Schema) ValidationError!void`
//! - `pub fn validateQuery(format: Format, query: core.QuerySchema, title: []const u8, probe_labels: []const []const u8) QueryError!void`
//! - `pub fn portCount(schema: Schema) ValidationError!u32`
//! - `pub fn sParameter(name: []const u8) ?[2]u32`

const std = @import("std");
const assert = std.debug.assert;
const core = @import("core");

/// An output file format. `root.parseFormat` maps user names and aliases
/// to it; `write.concatenates` says which formats hold several plots.
pub const Format = enum(u8) { binary, ascii, csv, touchstone, psf, fsdb, sst2, citi, print };

/// Where and how a session writes its plots. Fixed for the session's life.
pub const Selection = struct {
    format: Format = .binary,
    /// Null keeps results in memory and writes no file.
    path: ?[]const u8 = null,
};

pub const Schema = core.Schema;
pub const Result = core.Result;

/// Name slots in an sst2 names record; a plot with more variables is refused.
pub const sst2_names_max: u32 = 64;

/// A result as one file shows it: the deck title over the analysis payload.
/// Borrows every slice.
pub const Plot = struct {
    title: []const u8,
    result: Result,
    /// Touchstone port reference impedances in ohms, one per port; empty is
    /// 50 ohm on every port. Other formats ignore it.
    z0: []const f64 = &.{},

    /// Returns the plot's column layout with its point count known, as
    /// `validateSchema` takes it.
    pub fn schema(self: Plot) Schema {
        return .{ .varnames = self.result.varnames, .is_complex = self.result.is_complex, .npoints = self.result.npoints };
    }

    /// Returns the f64s per point: one per variable, two when complex.
    pub fn stride(self: Plot) usize {
        return self.result.varnames.len * @as(usize, if (self.result.is_complex) 2 else 1);
    }

    /// Returns the samples of point `pt`: one f64 per variable, or a
    /// (real, imaginary) pair per variable when complex. Borrows
    /// `result.data`. Asserts that `pt < result.npoints` and that the data
    /// holds the point, as it does for a plot that passed `validatePlot`.
    pub fn point(self: Plot, pt: usize) []const f64 {
        assert(pt < self.result.npoints);
        const n = self.stride();
        assert(self.result.data.len >= (pt + 1) * n);
        return self.result.data[pt * n ..][0..n];
    }
};

/// Why a format cannot hold a plot. Nothing is written before one is found.
pub const ValidationError = error{
    /// No variables, a sample count that overflows, data whose length
    /// disagrees with the shape, or Touchstone `z0` neither empty nor one
    /// impedance per port.
    DataLengthMismatch,
    /// Touchstone or CITI given anything but the S-matrix layout it needs.
    NotSParameterData,
    /// Past fsdb's u32 counts and u16 string lengths, or sst2's name slots.
    FormatLimitExceeded,
};

/// Why `validateQuery` refuses a query before it runs.
pub const QueryError = error{
    /// Touchstone or CITI for anything but `.sp`.
    NotSParameterData,
    /// Touchstone or CITI for an `.sp` with no ports.
    NoPorts,
    /// sst2 or fsdb for a result with no columns.
    DataLengthMismatch,
    /// sst2 or fsdb past its fixed limits.
    FormatLimitExceeded,
};

/// Checks that `format` can hold `plot`: `validateSchema`, then the data
/// length, fsdb's string limits and Touchstone's `z0`. Every encoder assumes
/// a plot that passed this.
pub fn validatePlot(format: Format, plot: Plot) ValidationError!void {
    try validateSchema(format, plot.schema());
    if (plot.result.data.len != try sampleCount(plot.schema(), plot.result.npoints)) return error.DataLengthMismatch;
    switch (format) {
        .fsdb => if (plot.title.len > std.math.maxInt(u16) or plot.result.plotname.len > std.math.maxInt(u16))
            return error.FormatLimitExceeded,
        .touchstone => if (plot.z0.len != 0 and plot.z0.len != try portCount(plot.schema()))
            return error.DataLengthMismatch,
        .binary, .ascii, .csv, .psf, .sst2, .citi, .print => {},
    }
}

/// Checks that `format` can hold a result of this shape. Pure, so
/// construction can reject a bad selection before any I/O.
pub fn validateSchema(format: Format, schema: Schema) ValidationError!void {
    if (schema.varnames.len == 0) return error.DataLengthMismatch;
    if (schema.npoints) |n| _ = try sampleCount(schema, n);
    switch (format) {
        .touchstone => _ = try portCount(schema),
        .citi => try citiLayout(schema),
        .fsdb => {
            if (schema.varnames.len > std.math.maxInt(u32)) return error.FormatLimitExceeded;
            if (schema.npoints) |n| if (n > std.math.maxInt(u32)) return error.FormatLimitExceeded;
            for (schema.varnames) |name| if (name.len > std.math.maxInt(u16)) return error.FormatLimitExceeded;
        },
        .sst2 => if (schema.varnames.len > sst2_names_max) return error.FormatLimitExceeded,
        .binary, .ascii, .csv, .psf, .print => {},
    }
}

/// Refuses, before it runs, a query whose result `format` cannot encode
/// under the deck's `title` and `probe_labels`. Formats other than
/// Touchstone, CITI, sst2 and fsdb are not checked until `validatePlot`.
pub fn validateQuery(format: Format, query: core.QuerySchema, title: []const u8, probe_labels: []const []const u8) QueryError!void {
    switch (format) {
        .touchstone, .citi => {
            if (query.kind != .sp) return error.NotSParameterData;
            if (query.portless) return error.NoPorts;
        },
        .sst2 => {
            if (query.columns == 0) return error.DataLengthMismatch;
            if (query.columns > sst2_names_max) return error.FormatLimitExceeded;
        },
        .fsdb => {
            if (query.columns == 0) return error.DataLengthMismatch;
            if (query.columns > std.math.maxInt(u32) or title.len > std.math.maxInt(u16)) return error.FormatLimitExceeded;
            for (probe_labels) |label| if (label.len > std.math.maxInt(u16)) return error.FormatLimitExceeded;
        },
        .binary, .ascii, .csv, .psf, .print => {},
    }
}

/// Returns the port count of a complete S matrix: complex data, `frequency`
/// first, then every `S(m,n)` in row-major order. Touchstone needs this
/// layout; columns after the S block (`.lin`'s) are allowed.
pub fn portCount(schema: Schema) ValidationError!u32 {
    const names = schema.varnames;
    if (!schema.is_complex or names.len < 2 or !std.mem.eql(u8, names[0], "frequency")) return error.NotSParameterData;
    var end: usize = 1;
    while (end < names.len and sParameter(names[end]) != null) end += 1;
    const last = sParameter(names[end - 1]) orelse return error.NotSParameterData;
    const n: usize = last[0];
    if (last[1] != n) return error.NotSParameterData;
    const cells = std.math.mul(usize, n, n) catch return error.NotSParameterData;
    if (cells != end - 1) return error.NotSParameterData;
    for (names[1..end], 0..) |name, i| {
        const pair = sParameter(name).?; // The scan above stopped at the first non-S name.
        if (pair[0] != i / n + 1 or pair[1] != i % n + 1) return error.NotSParameterData;
    }
    assert(last[0] > 0);
    return last[0];
}

/// Returns the one-based port pair of an `S(m,n)` or ngspice `v(S_m_n)`
/// label, or null if `name` is neither.
pub fn sParameter(name: []const u8) ?[2]u32 {
    if (!std.mem.endsWith(u8, name, ")")) return null;
    const ngspice = std.mem.startsWith(u8, name, "v(S_");
    if (!ngspice and !std.mem.startsWith(u8, name, "S(")) return null;
    const inner = name[(if (ngspice) @as(usize, 4) else 2) .. name.len - 1];
    const separator = std.mem.indexOfScalar(u8, inner, if (ngspice) '_' else ',') orelse return null;
    for (inner, 0..) |char, i| if (i != separator and !std.ascii.isDigit(char)) return null;
    const m = std.fmt.parseInt(u32, inner[0..separator], 10) catch return null;
    const n = std.fmt.parseInt(u32, inner[separator + 1 ..], 10) catch return null;
    if (m == 0 or n == 0) return null;
    return .{ m, n };
}

// ---- private ----

fn sampleCount(schema: Schema, npoints: usize) ValidationError!usize {
    const cells = std.math.mul(usize, npoints, schema.varnames.len) catch return error.DataLengthMismatch;
    return std.math.mul(usize, cells, if (schema.is_complex) 2 else 1) catch error.DataLengthMismatch;
}

/// CITI takes complex data on `frequency` with at least one S parameter;
/// every column labelled like one must parse as one.
fn citiLayout(schema: Schema) ValidationError!void {
    assert(schema.varnames.len > 0);
    if (!schema.is_complex or !std.mem.eql(u8, schema.varnames[0], "frequency")) return error.NotSParameterData;
    var found = false;
    for (schema.varnames[1..]) |name| {
        if (!std.mem.startsWith(u8, name, "S(") and !std.mem.startsWith(u8, name, "v(S_")) continue;
        _ = sParameter(name) orelse return error.NotSParameterData;
        found = true;
    }
    if (!found) return error.NotSParameterData;
}

test "output schemas validate unknown sizes, checked dimensions and S-parameter layout" {
    const t = std.testing;
    const real: Schema = .{ .varnames = &.{ "time", "v(out)" }, .is_complex = false };
    try validateSchema(.csv, real);
    try t.expectError(error.NotSParameterData, validateSchema(.touchstone, real));
    try t.expectError(error.NotSParameterData, validateSchema(.citi, real));
    var huge = real;
    huge.npoints = std.math.maxInt(usize);
    try t.expectError(error.DataLengthMismatch, validateSchema(.binary, huge));

    const partial: Schema = .{ .varnames = &.{ "frequency", "S(2,2)" }, .is_complex = true };
    try validateSchema(.citi, partial);
    try t.expectError(error.NotSParameterData, validateSchema(.touchstone, partial));
    const complete: Schema = .{
        .varnames = &.{ "frequency", "S(1,1)", "S(1,2)", "S(2,1)", "S(2,2)" },
        .is_complex = true,
    };
    try t.expectEqual(@as(u32, 2), try portCount(complete));
    try validateSchema(.touchstone, complete);

    const too_many_names: [65][]const u8 = @splat("v(out)");
    try t.expectError(error.FormatLimitExceeded, validateSchema(.sst2, .{
        .varnames = &too_many_names,
        .is_complex = false,
    }));
    const mislabeled: Schema = .{ .varnames = &.{ "time", "S(1,1)" }, .is_complex = true };
    try t.expectError(error.NotSParameterData, validateSchema(.citi, mislabeled));
}

test "S-parameter labels accept both producers and reject malformed port identities" {
    try std.testing.expectEqual([2]u32{ 12, 3 }, sParameter("S(12,3)").?);
    try std.testing.expectEqual([2]u32{ 12, 3 }, sParameter("v(S_12_3)").?);
    for ([_][]const u8{ "S()", "S(0,1)", "S(1,)", "S(1,2,3)", "v(S_)", "v(S_1_0)", "v(S_1_2_3)", "v(S_+1_2)", "v(S_4294967296_1)" }) |name| {
        try std.testing.expect(sParameter(name) == null);
        try std.testing.expectError(error.NotSParameterData, validateSchema(.citi, .{
            .varnames = &.{ "frequency", "S(1,1)", name },
            .is_complex = true,
        }));
    }
}

test validateQuery {
    const t = std.testing;
    const sp: core.QuerySchema = .{ .kind = .sp, .columns = 5, .portless = false };
    var tran = sp;
    tran.kind = .tran;
    try validateQuery(.touchstone, sp, "t", &.{});
    try t.expectError(error.NotSParameterData, validateQuery(.citi, tran, "t", &.{}));
    var portless = sp;
    portless.portless = true;
    try t.expectError(error.NoPorts, validateQuery(.touchstone, portless, "t", &.{}));
    var none = tran;
    none.columns = 0;
    try validateQuery(.csv, none, "t", &.{});
    try t.expectError(error.DataLengthMismatch, validateQuery(.sst2, none, "t", &.{}));
    var wide = tran;
    wide.columns = 65;
    try t.expectError(error.FormatLimitExceeded, validateQuery(.sst2, wide, "t", &.{}));
    try validateQuery(.fsdb, wide, "t", &.{});
    const long: [std.math.maxInt(u16) + 1]u8 = @splat('x');
    try t.expectError(error.FormatLimitExceeded, validateQuery(.fsdb, tran, &long, &.{}));
    try t.expectError(error.FormatLimitExceeded, validateQuery(.fsdb, tran, "t", &.{&long}));
    try validateQuery(.sst2, tran, &long, &.{&long});
}

test portCount {
    const t = std.testing;
    try t.expectEqual(@as(u32, 1), try portCount(.{ .varnames = &.{ "frequency", "S(1,1)" }, .is_complex = true }));
    // Columns after the S block (`.lin`'s noise figures) are allowed.
    try t.expectEqual(@as(u32, 1), try portCount(.{ .varnames = &.{ "frequency", "v(S_1_1)", "NFMIN" }, .is_complex = true }));
    for ([_]Schema{
        .{ .varnames = &.{ "frequency", "S(1,1)" }, .is_complex = false },
        .{ .varnames = &.{"frequency"}, .is_complex = true },
        .{ .varnames = &.{ "freq", "S(1,1)" }, .is_complex = true },
        .{ .varnames = &.{ "frequency", "NFMIN" }, .is_complex = true },
        .{ .varnames = &.{ "frequency", "S(1,1)", "S(2,1)", "S(1,2)", "S(2,2)" }, .is_complex = true },
        .{ .varnames = &.{ "frequency", "S(1,1)", "S(1,2)", "S(2,2)" }, .is_complex = true },
        .{ .varnames = &.{ "frequency", "S(1,2)" }, .is_complex = true },
    }) |schema| try t.expectError(error.NotSParameterData, portCount(schema));
}

test validatePlot {
    const t = std.testing;
    var plot: Plot = .{ .title = "t", .result = .{ .plotname = "p", .varnames = &.{ "time", "v(a)" }, .is_complex = true, .npoints = 2, .data = &.{ 0, 0, 1, 2, 3, 0, 4, 5 } } };
    try validatePlot(.binary, plot);
    try t.expectEqualSlices(f64, &.{ 3, 0, 4, 5 }, plot.point(1));
    try t.expectError(error.DataLengthMismatch, validatePlot(.binary, .{ .title = "t", .result = .{ .plotname = "p", .varnames = &.{}, .is_complex = false, .npoints = 0, .data = &.{} } }));
    const long: [std.math.maxInt(u16) + 1]u8 = @splat('x');
    plot.title = &long;
    try validatePlot(.csv, plot);
    try t.expectError(error.FormatLimitExceeded, validatePlot(.fsdb, plot));
    plot.title = "t";
    plot.result.plotname = &long;
    try t.expectError(error.FormatLimitExceeded, validatePlot(.fsdb, plot));
}

test "Touchstone z0 is empty or one impedance per port" {
    // Oracle: the Touchstone 2.0 `[Reference]` line lists one value per port.
    var plot: Plot = .{ .title = "s1p", .result = .{ .plotname = "sp", .varnames = &.{ "frequency", "S(1,1)" }, .is_complex = true, .npoints = 1, .data = &.{ 1, 0, 0.5, -0.5 } } };
    try validatePlot(.touchstone, plot);
    plot.z0 = &.{75};
    try validatePlot(.touchstone, plot);
    plot.z0 = &.{ 50, 75 };
    try std.testing.expectError(error.DataLengthMismatch, validatePlot(.touchstone, plot));
    // Other formats ignore z0.
    try validatePlot(.csv, plot);
}
