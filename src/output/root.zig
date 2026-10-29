//! Waveform writers: plots in, files and `.meas` listings out. Every format
//! is an `encode(w, plot)` onto a `std.Io.Writer`; write.zig validates,
//! encodes and atomically replaces the file, and `Session` publishes whole
//! plots in order. Aggregation only: leaves import plot.zig, never this file.
//! Cites: none (each writer cites its format).
//!
//! Contents:
//! - `pub const Session`, `pub const Selection`, `pub const Format`, `pub const Result`, `pub const Plot`
//! - `pub const validateQuery`, `pub const write`
//! - `pub const printMeasures`, `pub const printFailedMeasures`, `pub const MeasurePoint`,
//!   `pub const printMeasureStatistics`, `pub const measureValues`, `pub const measureOutcomes`,
//!   `pub const sampleMeasure`
//! - `pub fn parseFormat(s: []const u8) ?Format`

const std = @import("std");
const measure = @import("measure");
pub const Session = @import("session.zig").Session;
/// Validates, encodes and atomically replaces one file with one plot.
pub const write = @import("write.zig").write;
const plots = @import("plot.zig");

pub const Selection = plots.Selection;
pub const Format = plots.Format;
pub const Result = plots.Result;
pub const Plot = plots.Plot;
pub const validateQuery = plots.validateQuery;
/// `.meas` results in ngspice's print format.
pub const printMeasures = measure.print;
/// A failed sweep point's `.meas` heading and `name = NaN` lines.
pub const printFailedMeasures = measure.printFailed;
/// The sweep point a `.meas` block names in its heading.
pub const MeasurePoint = measure.Point;
/// `.meas` statistics over Monte Carlo trials.
pub const printMeasureStatistics = measure.printStatistics;
/// Every `.meas` card's value over one result; see `measure.evaluateAll`.
pub const measureValues = measure.evaluateAll;
/// `measureValues`, also flagging cards that name a missing vector.
pub const measureOutcomes = measure.evaluateOutcomes;
/// A measure clause's transient waveform at given times (HSPICE `.stim`).
pub const sampleMeasure = measure.sample;

const format_names: std.StaticStringMap(Format) = .initComptime(.{
    .{ "binary", .binary },         .{ "raw", .binary },
    .{ "ascii", .ascii },           .{ "csv", .csv },
    .{ "touchstone", .touchstone }, .{ "snp", .touchstone },
    .{ "s2p", .touchstone },        .{ "psf", .psf },
    .{ "fsdb", .fsdb },             .{ "sst2", .sst2 },
    .{ "hspice", .sst2 },           .{ "citi", .citi },
    .{ "citifile", .citi },         .{ "print", .print },
    .{ "text", .print },
});

/// Returns the `Format` a user-facing format name or alias names, or null
/// for an unknown name so the caller owns the diagnostic.
pub fn parseFormat(s: []const u8) ?Format {
    return format_names.get(s);
}

test "every Format arm is reachable by its own name" {
    const info = @typeInfo(Format).@"enum";
    inline for (info.field_names, info.field_values) |name, value| {
        try std.testing.expectEqual(@as(?Format, @fromBackingInt(@intCast(value))), parseFormat(name));
    }
}

test {
    _ = plots;
    _ = @import("write.zig");
    _ = @import("session.zig");
    _ = @import("rawfile.zig");
    _ = @import("ascii_raw.zig");
    _ = @import("csv.zig");
    _ = @import("spice_print.zig");
    _ = @import("touchstone.zig");
    _ = @import("citifile.zig");
    _ = @import("psf.zig");
    _ = @import("sst2.zig");
    _ = @import("fsdb.zig");
}
