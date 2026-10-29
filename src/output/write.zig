//! The one-plot file write every format goes through: a path, format and
//! plot in, the destination replaced atomically out, so it holds the old
//! bytes or the whole new plot, never a torn one. A destination that is not a
//! regular file (a device or FIFO, `-r /dev/null`) cannot be renamed over and
//! is written in place. Several plots in one file (`concatenates`) are
//! `Session`'s: it encodes each into one replacement and renames that once,
//! at `finish`, so a sweep of P plots writes each plot once (#179).
//! Cites: none (POSIX rename(2) atomicity, through `Io.File.Atomic`).
//!
//! Contents:
//! - `pub const Error`
//! - `pub fn write(io: Io, path: []const u8, format: Format, plot: Plot) Error!void`
//! - `pub fn isDevice(io: Io, path: []const u8) bool`
//! - `pub fn concatenates(format: Format) bool`
//! - `pub fn encode(w: *Io.Writer, format: Format, plot: Plot) Io.Writer.Error!void`

const std = @import("std");
const Io = std.Io;
const assert = std.debug.assert;
const plots = @import("plot.zig");
const Format = plots.Format;
const Plot = plots.Plot;

/// Every way `write` fails. A validation error leaves every destination
/// untouched; any other error leaves a regular file untouched and a device
/// with what it was sent.
pub const Error = plots.ValidationError || Io.File.OpenError || Io.Dir.CreateFileAtomicError ||
    Io.File.Atomic.ReplaceError || Io.Writer.Error;

/// Replaces `path` with `plot` encoded as `format`.
pub fn write(io: Io, path: []const u8, format: Format, plot: Plot) Error!void {
    try plots.validatePlot(format, plot);
    var buf: [8192]u8 = undefined;
    if (isDevice(io, path)) {
        const file = try Io.Dir.cwd().openFile(io, path, .{ .mode = .write_only });
        defer file.close(io);
        var fw = file.writer(io, &buf);
        try encode(&fw.interface, format, plot);
        return fw.interface.flush();
    }
    var atomic = try Io.Dir.cwd().createFileAtomic(io, path, .{ .replace = true });
    defer atomic.deinit(io);
    var fw = atomic.file.writer(io, &buf);
    try encode(&fw.interface, format, plot);
    try fw.interface.flush();
    try atomic.replace(io);
}

/// Whether `path` exists and is not a regular file: a device or FIFO
/// (`-r /dev/null`), which cannot be renamed over and is written in place.
/// A path that cannot be stat'ed is not a device.
pub fn isDevice(io: Io, path: []const u8) bool {
    const stat = Io.Dir.cwd().statFile(io, path, .{}) catch return false;
    return stat.kind != .file;
}

/// Whether one file holds several plots back to back. ngspice writes every
/// plot of a deck into one raw file, binary or ASCII, and readers take the
/// concatenation; a `.print` listing is a sequence of tables. The other
/// formats hold one dataset per file.
pub fn concatenates(format: Format) bool {
    return switch (format) {
        .binary, .ascii, .print => true,
        .csv, .touchstone, .psf, .fsdb, .sst2, .citi => false,
    };
}

/// Writes `plot` as `format` to `w`, without validating or flushing.
/// Asserts what the format's `encode` asserts: a plot that passed
/// `validatePlot(format, plot)`.
pub fn encode(w: *Io.Writer, format: Format, plot: Plot) Io.Writer.Error!void {
    assert(plot.result.varnames.len > 0);
    return switch (format) {
        .binary => @import("rawfile.zig").encode(w, plot),
        .ascii => @import("ascii_raw.zig").encode(w, plot),
        .csv => @import("csv.zig").encode(w, plot),
        .touchstone => @import("touchstone.zig").encode(w, plot),
        .psf => @import("psf.zig").encode(w, plot),
        .fsdb => @import("fsdb.zig").encode(w, plot),
        .sst2 => @import("sst2.zig").encode(w, plot),
        .citi => @import("citifile.zig").encode(w, plot),
        .print => @import("spice_print.zig").encode(w, plot),
    };
}

test "a character device is written in place, not renamed over" {
    const plot: Plot = .{ .title = "t", .result = .{
        .plotname = "p",
        .varnames = &.{"v(a)"},
        .is_complex = false,
        .npoints = 1,
        .data = &.{1.0},
    } };
    try write(std.testing.io, "/dev/null", .binary, plot);
}

test "every format refuses a plot whose data length disagrees with its shape" {
    const plot: Plot = .{ .title = "bad", .result = .{
        .plotname = "bad",
        .varnames = &.{ "v(a)", "v(b)" },
        .is_complex = false,
        .npoints = 2,
        .data = &.{ 1.0, 2.0, 3.0 },
    } };
    inline for (@typeInfo(Format).@"enum".field_values) |value| {
        const format: Format = @fromBackingInt(@intCast(value));
        const expected = if (format == .touchstone or format == .citi) error.NotSParameterData else error.DataLengthMismatch;
        try std.testing.expectError(expected, write(std.testing.io, "zig-out/should_not_exist", format, plot));
    }
}

test "write replaces the file" {
    const io = std.testing.io;
    const a = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try a.print(".zig-cache/tmp/{s}/out.txt", .{tmp.sub_path});
    defer a.free(path);
    const plot: Plot = .{ .title = "t", .result = .{ .plotname = "p", .varnames = &.{"v(a)"}, .is_complex = false, .npoints = 1, .data = &.{1.0} } };
    var one: Io.Writer.Allocating = .init(a);
    defer one.deinit();
    try encode(&one.writer, .print, plot);
    try tmp.dir.writeFile(io, .{ .sub_path = "out.txt", .data = "old" });
    try write(io, path, .print, plot);
    const got = try tmp.dir.readFileAlloc(io, "out.txt", a, .unlimited);
    defer a.free(got);
    try std.testing.expectEqualStrings(one.written(), got);
}
