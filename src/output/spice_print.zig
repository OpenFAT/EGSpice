//! SPICE3 `.print`-style text table: a plot line, a header row, a dashed
//! rule, then one tab-separated row per point.
const std = @import("std");
const Io = std.Io;
const assert = std.debug.assert;
const Plot = @import("types.zig").Plot;

/// Writes `plot`, which must have passed `types.validatePlot(.print, ...)`.
/// Asserts that it has variables and one sample row per point.
pub fn encode(w: *Io.Writer, plot: Plot) !void {
    assert(plot.result.data.len == plot.result.npoints * plot.stride());
    try writeHeader(w, plot);
    try writeRows(w, plot.stride(), 0, plot.result.data);
}

/// The plot line, header row and rule: what `encode` writes before the
/// points.
pub fn writeHeader(w: *Io.Writer, plot: Plot) !void {
    assert(plot.result.varnames.len > 0);
    try w.print("{s}: {s}\n", .{ plot.result.plotname, plot.title });
    try w.writeAll("Index");
    for (plot.result.varnames) |name| {
        try w.writeByte('\t');
        try w.writeAll(name);
        if (plot.result.is_complex) try w.writeAll("\t(imag)");
    }
    try w.writeAll("\n-----");
    for (0..plot.stride()) |_| try w.writeAll("\t---------------");
    try w.writeByte('\n');
}

/// The rows of the points in `rows` (`stride` f64s each), the first
/// numbered `first`: what `encode` writes after the header.
pub fn writeRows(w: *Io.Writer, stride: usize, first: usize, rows: []const f64) !void {
    assert(rows.len % stride == 0);
    for (0..rows.len / stride) |k| {
        try w.print("{d}", .{first + k});
        for (rows[k * stride ..][0..stride]) |x| try w.print("\t{e}", .{x});
        try w.writeByte('\n');
    }
}

test "SPICE3 print table" {
    var buf: [256]u8 = undefined;
    var w: Io.Writer = .fixed(&buf);
    try encode(&w, .{ .title = "ac", .result = .{ .plotname = "AC Analysis", .varnames = &.{ "frequency", "v(out)" }, .is_complex = true, .npoints = 1, .data = &.{ 1.0, 0.0, 0.5, -0.5 } } });
    try std.testing.expectEqualStrings("AC Analysis: ac\nIndex\tfrequency\t(imag)\tv(out)\t(imag)\n" ++
        "-----\t---------------\t---------------\t---------------\t---------------\n0\t1e0\t0e0\t5e-1\t-5e-1\n", w.buffered());
}
