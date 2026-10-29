//! ngspice ASCII raw file: a validated plot in, the raw header then one text
//! line per variable out, each point's first line led by the point index.
//! `Session` writes the rows of a side file through `writeRows`.
//! Cites: ngspice 44 manual §17.5 (raw file format, `Values:`).
//!
//! Contents:
//! - `pub fn encode(w: *Io.Writer, plot: Plot) Io.Writer.Error!void`
//! - `pub fn writeRows(w: *Io.Writer, complex: bool, stride: usize, first: usize, rows: []const f64) Io.Writer.Error!void`

const std = @import("std");
const Io = std.Io;
const assert = std.debug.assert;
const rawfile = @import("rawfile.zig");
const Plot = @import("plot.zig").Plot;

/// Writes `plot`, which must have passed `plot.validatePlot(.ascii, ...)`.
/// Asserts that it has variables and one sample row per point.
pub fn encode(w: *Io.Writer, plot: Plot) Io.Writer.Error!void {
    assert(plot.result.varnames.len > 0);
    assert(plot.result.data.len == plot.result.npoints * plot.stride());
    try rawfile.writeHeader(w, plot, .ascii);
    try writeRows(w, plot.result.is_complex, plot.stride(), 0, plot.result.data);
}

/// Writes the value lines of the points in `rows` (`stride` f64s each, a
/// (real, imaginary) pair per variable when `complex`), the first numbered
/// `first`: what `encode` writes after the header. Asserts that `rows` holds
/// whole points and a complex stride is even.
pub fn writeRows(w: *Io.Writer, complex: bool, stride: usize, first: usize, rows: []const f64) Io.Writer.Error!void {
    assert(stride > 0);
    assert(rows.len % stride == 0);
    if (complex) assert(stride % 2 == 0);
    for (0..rows.len / stride) |k| {
        const row = rows[k * stride ..][0..stride];
        try w.print("{d}", .{first + k});
        if (complex) {
            for (0..stride / 2) |v| try w.print("\t{e},{e}\n", .{ row[2 * v], row[2 * v + 1] });
        } else for (row) |x| try w.print("\t{e}\n", .{x});
    }
}

test "ASCII raw values follow the Values: header" {
    var buf: [512]u8 = undefined;
    var w: Io.Writer = .fixed(&buf);
    try encode(&w, .{ .title = "ac", .result = .{
        .plotname = "AC Analysis",
        .varnames = &.{ "frequency", "v(out)" },
        .is_complex = true,
        .npoints = 1,
        .data = &.{ 1.0, 0.0, 0.5, -0.5 },
    } });
    try std.testing.expect(std.mem.endsWith(u8, w.buffered(), "Flags: complex\nNo. Variables: 2\nNo. Points: 1\n" ++
        "Variables:\n\t0\tfrequency\tfrequency\n\t1\tv(out)\tvoltage\nValues:\n0\t1e0,0e0\n\t5e-1,-5e-1\n"));
}

test "ASCII raw writes one line per real variable" {
    var buf: [512]u8 = undefined;
    var w: Io.Writer = .fixed(&buf);
    try encode(&w, .{ .title = "t", .result = .{ .plotname = "Transient", .varnames = &.{ "time", "v(a)" }, .is_complex = false, .npoints = 2, .data = &.{ 0, 1, 0.5, -2 } } });
    try std.testing.expect(std.mem.endsWith(u8, w.buffered(), "Values:\n0\t0e0\n\t1e0\n1\t5e-1\n\t-2e0\n"));
}

test "ASCII raw round-trips every complex sample through parseFloat" {
    const a = std.testing.allocator;
    var data: [2 * 2 * 25]f64 = undefined;
    var prng: std.Random.DefaultPrng = .init(0xa5c);
    for (&data, 0..) |*x, i| x.* = if (i == 0) -0.0 else if (i == 1) 5e-324 else @bitCast(prng.random().int(u64) & ~@as(u64, 0x4000_0000_0000_0000));
    var out: Io.Writer.Allocating = .init(a);
    defer out.deinit();
    try encode(&out.writer, .{ .title = "t", .result = .{ .plotname = "AC", .varnames = &.{ "frequency", "v(o)" }, .is_complex = true, .npoints = 25, .data = &data } });
    const marker = "Values:\n";
    var lines = std.mem.splitScalar(u8, out.written()[std.mem.indexOf(u8, out.written(), marker).? + marker.len ..], '\n');
    for (0..25) |pt| for (0..2) |v| {
        const line = lines.next().?;
        // Point pt's first line is led by its index, the rest by a tab.
        const tab = std.mem.indexOfScalar(u8, line, '\t').?;
        if (v == 0) try std.testing.expectEqual(pt, try std.fmt.parseInt(usize, line[0..tab], 10)) else try std.testing.expectEqual(0, tab);
        const comma = std.mem.indexOfScalar(u8, line, ',').?;
        for ([_][]const u8{ line[tab + 1 .. comma], line[comma + 1 ..] }, 0..) |f, part| {
            const want: u64 = @bitCast(data[(pt * 2 + v) * 2 + part]);
            try std.testing.expectEqual(want, @as(u64, @bitCast(try std.fmt.parseFloat(f64, f))));
        }
    };
    try std.testing.expectEqualStrings("", lines.next().?);
    try std.testing.expectEqual(null, lines.next());
}
