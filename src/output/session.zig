//! Delivery of plots to one output selection, in publish order. The caller
//! keeps each result alive until `publish` returns, or streams a binary raw
//! plot's rows as they are computed (`beginStream`), or any number of plots'
//! rows to temporary files from any thread (`beginSide`). Plots that share one
//! file (binary and ASCII raw, `.print`) are appended to one replacement,
//! each written once, and `finish` renames it over the destination (#179).
const std = @import("std");
const Io = std.Io;
const assert = std.debug.assert;
const types = @import("types.zig");
const dispatch = @import("write.zig");

/// One output destination and the count of plots delivered to it.
pub const Session = struct {
    /// Owns `selection.path` and the open `stream`.
    allocator: std.mem.Allocator,
    /// Fixed at `init`; the path is the session's own copy.
    selection: types.Selection,
    /// Plots delivered so far; the next one is plot number `published`.
    /// At most maxInt(u32) plots, numbered 0 through maxInt(u32) - 1.
    published: u32 = 0,
    /// `failed` after a writer error; every later call reports it.
    state: enum(u8) { ready, failed } = .ready,
    /// The plot `beginStream` opened, until `endStream` or `abortStream`.
    stream: ?*Stream = null,
    /// The replacement of a file that holds several plots, from the first
    /// plot published since the last `finish` until the next one, which
    /// renames it over the destination; until then the destination keeps its
    /// old bytes. A batch after a `finish` starts with a copy of the file.
    pending: ?*Stream = null,
    /// The open `beginSide` files, dropped by `deinit`.
    sides: std.ArrayList(*Stream) = .empty,
    /// The next side file's number.
    side_serial: u32 = 0,

    /// Plot 0 in binary raw, written while its query runs: the header with
    /// the point count left blank, then rows as they arrive; `endStream`
    /// keeps it as `pending`. Without a path the rows are counted and
    /// dropped. A `pending` replacement opened by `publish` uses only
    /// `atomic`, `file` and `buffer`.
    pub const Stream = struct {
        atomic: ?Io.File.Atomic,
        file: Io.File.Writer,
        discard: Io.Writer.Discarding,
        /// File offsets of the blank point count and of the first row.
        count_at: u64,
        rows_at: u64,
        complex: bool,
        /// A `beginSide` file, read back by `endSide`, and its name, which
        /// `drop` deletes. Null for every other stream.
        side: ?Io.File,
        side_name: []u8,
        buffer: [64 * 1024]u8,

        /// Where rows go: the replacement or side file, else a discard that
        /// only counts them.
        pub fn writer(s: *Stream) *Io.Writer {
            return if (s.atomic != null or s.side != null) &s.file.interface else &s.discard.writer;
        }
    };

    /// Copies `selection.path`; performs no I/O. Free with `deinit`.
    pub fn init(allocator: std.mem.Allocator, selection: types.Selection) !Session {
        return .{
            .allocator = allocator,
            .selection = .{
                .format = selection.format,
                .path = if (selection.path) |path| try allocator.dupe(u8, path) else null,
            },
        };
    }

    /// Drops an open stream and any plots not yet `finish`ed, leaving the
    /// destination as it was, and frees the path copy.
    pub fn deinit(self: *Session) void {
        self.abortStream();
        for (self.sides.items) |s| self.drop(s);
        self.sides.deinit(self.allocator);
        if (self.pending) |s| self.drop(s);
        if (self.selection.path) |path| self.allocator.free(path);
        self.* = undefined;
    }

    /// Writes the next plot. For binary and ASCII raw and `.print`
    /// listings every plot goes into one file, `path`, which `finish`
    /// replaces; for every other format plot 0 replaces `path` and later
    /// ones `path.<number + 1>`, each at once. Without a path the plot is
    /// only counted. An invalid plot (`types.ValidationError`) fails
    /// before any I/O and is not counted; so does `error.TooManyPlots`
    /// past maxInt(u32) - 1 plots. A writer error is terminal
    /// (`DeliveryFailed` from then on): a partial append cannot be replayed.
    pub fn publish(self: *Session, io: std.Io, plot: types.Plot) !void {
        if (self.state == .failed) return error.DeliveryFailed;
        if (self.published == std.math.maxInt(u32)) return error.TooManyPlots;
        try types.validatePlot(self.selection.format, plot);
        if (self.selection.path) |path| self.deliver(io, path, plot) catch |err| {
            self.state = .failed;
            return err;
        };
        self.published += 1;
    }

    fn deliver(self: *Session, io: Io, path: []const u8, plot: types.Plot) !void {
        const format = self.selection.format;
        if (!dispatch.concatenates(format)) {
            const numbered = try self.plotPath(path);
            defer if (numbered) |n| self.allocator.free(n);
            return dispatch.write(io, numbered orelse path, format, plot);
        }
        // A device or FIFO cannot be renamed over: it takes each plot as it comes.
        const s = self.pending orelse try self.open(io, path, self.published != 0) orelse
            return dispatch.write(io, path, format, plot);
        self.pending = s;
        try dispatch.encode(&s.file.interface, format, plot);
    }

    /// The next plot's own file in a format that holds one plot: `path`
    /// for plot 0 (null), else `path.<number + 1>`, which the caller frees.
    fn plotPath(self: *Session, path: []const u8) !?[]u8 {
        if (self.published == 0) return null;
        return try self.allocator.print("{s}.{d}", .{ path, self.published + 1 });
    }

    /// A replacement for the file at `path`, holding the file's bytes when
    /// `keep`; null when `path` is a device or FIFO.
    fn open(self: *Session, io: Io, path: []const u8, keep: bool) !?*Stream {
        if (dispatch.isDevice(io, path)) return null;
        const s = try self.create();
        errdefer self.allocator.destroy(s);
        s.atomic = try Io.Dir.cwd().createFileAtomic(io, path, .{ .replace = true });
        errdefer s.atomic.?.deinit(io);
        s.file = s.atomic.?.file.writer(io, &s.buffer);
        if (keep) copy: {
            const old = Io.Dir.cwd().openFile(io, path, .{}) catch |err| switch (err) {
                error.FileNotFound => break :copy,
                else => return err, // else: any other open failure is the caller's
            };
            defer old.close(io);
            var rbuf: [8192]u8 = undefined;
            var reader = old.reader(io, &rbuf);
            _ = try reader.interface.streamRemaining(&s.file.interface);
        }
        return s;
    }

    /// A stream with no file behind it: no replacement, no side file, rows
    /// counted and dropped. `file` and `buffer` are left undefined.
    fn create(self: *Session) !*Stream {
        const s = try self.allocator.create(Stream);
        s.atomic = null;
        s.discard = .init(&.{});
        s.count_at = 0;
        s.rows_at = 0;
        s.complex = false;
        s.side = null;
        s.side_name = &.{};
        return s;
    }

    /// Closes `s` and deletes its replacement, if any, leaving the
    /// destination as it was.
    fn drop(self: *Session, s: *Stream) void {
        if (s.atomic) |*atomic| atomic.deinit(s.file.io);
        if (s.side) |file| {
            file.close(s.file.io);
            Io.Dir.cwd().deleteFile(s.file.io, s.side_name) catch {};
            self.allocator.free(s.side_name);
        }
        self.allocator.destroy(s);
    }

    /// Opens plot 0 in binary raw for rows the caller writes to the returned
    /// writer as native-endian f64s, `plot.result.data` being empty. Null when
    /// the selection cannot stream (another format, a plot already
    /// delivered, a failed session, a destination that is not a regular
    /// file); publish the whole plot then. The header leaves `No. Points:` blank, as ngspice's
    /// batch raw file does, for `endStream` to fill in. The writer is valid
    /// until `endStream`, `abortStream` or `deinit`.
    pub fn beginStream(self: *Session, io: Io, plot: types.Plot) !?*Io.Writer {
        if (self.selection.format != .binary or self.published != 0 or self.state == .failed) return null;
        assert(self.pending == null);
        try types.validatePlot(.binary, plot);
        const path = self.selection.path orelse {
            const s = try self.create();
            s.complex = plot.result.is_complex;
            self.stream = s;
            return s.writer();
        };
        const s = try self.open(io, path, false) orelse return null;
        errdefer self.drop(s);
        s.complex = plot.result.is_complex;
        var header: Io.Writer.Allocating = .init(self.allocator);
        defer header.deinit();
        // An Allocating writer fails only when it cannot grow.
        @import("rawfile.zig").writeHeader(&header.writer, plot, true) catch return error.OutOfMemory;
        const field = "No. Points: 0\n";
        const at = std.mem.indexOf(u8, header.written(), field).? + field.len - 2;
        const w = &s.file.interface;
        try w.writeAll(header.written()[0..at]);
        try w.splatByteAll(' ', count_width);
        try w.writeAll(header.written()[at + 1 ..]);
        s.count_at = at;
        s.rows_at = s.file.logicalPos();
        self.stream = s;
        return w;
    }

    /// Room for the point count of a streamed plot, as many digits as u64 has.
    const count_width = 20;

    /// Finishes the streamed plot: checks that `npoints` rows of
    /// `varnames` columns arrived and fills in the point count; the plots
    /// published after it append to the same replacement, which `finish`
    /// renames over the destination. Counts as one `publish`. The stream is closed whether
    /// or not it succeeds. With a path, any error (a wrong row count
    /// included, `error.DataLengthMismatch`) fails the session; without
    /// one, a wrong count is returned and the session stays usable. Asserts
    /// that a stream is open.
    pub fn endStream(self: *Session, io: Io, npoints: usize, columns: usize) !void {
        assert(self.stream != null);
        const s = self.stream.?;
        defer self.abortStream();
        const want = @as(u64, npoints) * columns * @sizeOf(f64) * @as(u64, if (s.complex) 2 else 1);
        // A pathless stream counts what it dropped (#88).
        if (s.atomic == null and s.discard.fullCount() != want) return error.DataLengthMismatch;
        if (s.atomic) |*atomic| {
            errdefer self.state = .failed;
            const w = &s.file.interface;
            try w.flush();
            if (s.file.logicalPos() - s.rows_at != want) return error.DataLengthMismatch;
            var digits: [count_width]u8 = undefined;
            try atomic.file.writePositionalAll(io, try std.fmt.bufPrint(&digits, "{d}", .{npoints}), s.count_at);
            self.stream = null;
            self.pending = s;
        }
        self.published += 1;
    }

    /// Drops the streamed plot, if any, leaving the destination as it was.
    pub fn abortStream(self: *Session) void {
        const s = self.stream orelse return;
        self.drop(s);
        self.stream = null;
    }

    /// A side plot: rows written to the returned stream's `writer` (native-
    /// endian f64s, as `beginStream` takes them) land in a temporary file
    /// beside the destination, or are only counted without a path, and
    /// `endSide` later publishes them as the next plot. Any thread may write
    /// one side while others write theirs, so a sweep's points stream as
    /// they run and publish in request order. Null when the selection is
    /// not a row format (binary or ASCII raw, CSV, `.print`), the session
    /// has failed, or the destination is a device or FIFO; publish the
    /// whole plot then. The stream is valid until `endSide`, `dropSide` or
    /// `deinit`.
    pub fn beginSide(self: *Session, io: Io) !?*Stream {
        const format = self.selection.format;
        if (!sideFormat(format) or self.state == .failed) return null;
        if (self.selection.path) |path| if (dispatch.isDevice(io, path)) return null;
        try self.sides.ensureUnusedCapacity(self.allocator, 1);
        const s = try self.create();
        errdefer self.allocator.destroy(s);
        if (self.selection.path) |path| {
            // Beside the destination, readable for `endSide`, named
            // `<path>.<n>.side` with the first `n` free; `drop` deletes it.
            while (true) : (self.side_serial +%= 1) {
                const name = try self.allocator.print("{s}.{d}.side", .{ path, self.side_serial });
                const file = Io.Dir.cwd().createFile(io, name, .{ .read = true, .exclusive = true }) catch |err| {
                    self.allocator.free(name);
                    if (err == error.PathAlreadyExists) continue;
                    return err;
                };
                s.side = file;
                s.side_name = name;
                self.side_serial +%= 1;
                break;
            }
            s.file = s.side.?.writer(io, &s.buffer);
        }
        self.sides.appendAssumeCapacity(s);
        return s;
    }

    /// Publishes side `s` as the next plot: `plot`'s header (its `data`
    /// empty, `npoints` the rows written) then the side's rows, appended
    /// to the replacement, or written to the plot's own file, as `publish`
    /// writes them. The bytes are those `publish` writes for the whole
    /// plot. Closes `s` whether or not it succeeds; errors are
    /// `endStream`'s, and with a path any of them fails the session.
    pub fn endSide(self: *Session, io: Io, s: *Stream, plot: types.Plot) !void {
        defer self.dropSide(s);
        if (self.state == .failed) return error.DeliveryFailed;
        if (self.published == std.math.maxInt(u32)) return error.TooManyPlots;
        try types.validateSchema(self.selection.format, plot.schema());
        const want = @as(u64, plot.result.npoints) * plot.stride() * @sizeOf(f64);
        if (s.side == null) {
            if (s.discard.fullCount() != want) return error.DataLengthMismatch;
            self.published += 1;
            return;
        }
        errdefer self.state = .failed;
        try s.file.interface.flush();
        if (s.file.logicalPos() != want) return error.DataLengthMismatch;
        const path = self.selection.path.?;
        if (dispatch.concatenates(self.selection.format)) {
            const out = self.pending orelse try self.open(io, path, self.published != 0) orelse return error.DeliveryFailed;
            self.pending = out;
            try self.copySide(io, &out.file.interface, s, plot);
        } else {
            const numbered = try self.plotPath(path);
            defer if (numbered) |n| self.allocator.free(n);
            var atomic = try Io.Dir.cwd().createFileAtomic(io, numbered orelse path, .{ .replace = true });
            defer atomic.deinit(io);
            var buf: [64 * 1024]u8 = undefined;
            var fw = atomic.file.writer(io, &buf);
            try self.copySide(io, &fw.interface, s, plot);
            try fw.interface.flush();
            try atomic.replace(io);
        }
        self.published += 1;
    }

    /// Writes `plot`'s header then side `s`'s rows to `w` in the
    /// selection's format.
    fn copySide(self: *Session, io: Io, w: *Io.Writer, s: *Stream, plot: types.Plot) !void {
        const format = self.selection.format;
        var rbuf: [64 * 1024]u8 = undefined;
        var reader = s.side.?.reader(io, &rbuf);
        switch (format) {
            .binary => try @import("rawfile.zig").writeHeader(w, plot, true),
            .ascii => try @import("rawfile.zig").writeHeader(w, plot, false),
            .csv => try @import("csv.zig").writeHeader(w, plot),
            .print => try @import("spice_print.zig").writeHeader(w, plot),
            else => unreachable, // else: `beginSide` opens only `sideFormat`s
        }
        if (format == .binary) {
            _ = try reader.interface.streamRemaining(w);
            return;
        }
        // Whole rows at a time, about 64 KiB of them.
        const stride = plot.stride();
        const per = @max(1, 8192 / stride);
        const rows = try self.allocator.alloc(f64, per * stride);
        defer self.allocator.free(rows);
        var pt: usize = 0;
        while (pt < plot.result.npoints) {
            const k = @min(per, plot.result.npoints - pt);
            const chunk = rows[0 .. k * stride];
            try reader.interface.readSliceAll(std.mem.sliceAsBytes(chunk));
            switch (format) {
                .ascii => try @import("ascii_raw.zig").writeRows(w, plot.result.is_complex, stride, pt, chunk),
                .csv => try @import("csv.zig").writeRows(w, stride, chunk),
                .print => try @import("spice_print.zig").writeRows(w, stride, pt, chunk),
                else => unreachable, // else: binary copied its bytes above
            }
            pt += k;
        }
    }

    /// The formats a side stream writes: a header, then rows.
    fn sideFormat(format: types.Format) bool {
        return switch (format) {
            .binary, .ascii, .csv, .print => true,
            .touchstone, .psf, .fsdb, .sst2, .citi => false,
        };
    }

    /// Closes side `s` without publishing it.
    pub fn dropSide(self: *Session, s: *Stream) void {
        const at = std.mem.indexOfScalar(*Stream, self.sides.items, s).?;
        _ = self.sides.swapRemove(at);
        self.drop(s);
    }

    /// Publishes the plots appended since the last `finish` with one rename
    /// over the destination, so a reader sees the old file or every plot.
    /// `DeliveryFailed` once delivery has failed, leaving the destination
    /// as it was; a flush or rename error fails the session the same way.
    /// Does not seal the session: queries appended later may publish more
    /// plots, and the next `finish` adds them to the file.
    pub fn finish(self: *Session, io: Io) !void {
        if (self.state == .failed) return error.DeliveryFailed;
        const s = self.pending orelse return;
        self.pending = null;
        defer self.drop(s);
        errdefer self.state = .failed;
        try s.file.interface.flush();
        try s.atomic.?.replace(io);
    }
};

test "session: finish and later publication preserve raw append order" {
    for ([_]types.Format{ .binary, .ascii }) |format| {
        const io = std.testing.io;
        const a = std.testing.allocator;
        var tmp = std.testing.tmpDir(.{});
        defer tmp.cleanup();
        const path = try a.print(".zig-cache/tmp/{s}/plots.raw", .{tmp.sub_path});
        defer a.free(path);
        var session = try Session.init(a, .{ .format = format, .path = path });
        defer session.deinit();
        const first: types.Plot = .{
            .title = "session",
            .result = .{
                .plotname = "first",
                .varnames = &.{"v(out)"},
                .is_complex = false,
                .npoints = 1,
                .data = &.{1},
            },
        };
        try session.publish(io, first);
        try session.finish(io);
        const initial = try tmp.dir.readFileAlloc(io, "plots.raw", a, .unlimited);
        defer a.free(initial);
        var second = first;
        second.result.plotname = "second";
        try session.publish(io, second);
        try session.finish(io);
        const complete = try tmp.dir.readFileAlloc(io, "plots.raw", a, .unlimited);
        defer a.free(complete);
        try std.testing.expect(std.mem.startsWith(u8, complete, initial));
        try std.testing.expectEqual(@as(usize, 2), std.mem.count(u8, complete, "Title: session\n"));
        try std.testing.expect(std.mem.indexOf(u8, complete[initial.len..], "Plotname: second\n") != null);
    }
}

test "session: appended plots are each written once and published at finish (#179)" {
    for ([_]types.Format{ .binary, .ascii, .print }) |format| {
        const io = std.testing.io;
        const a = std.testing.allocator;
        var tmp = std.testing.tmpDir(.{});
        defer tmp.cleanup();
        const path = try a.print(".zig-cache/tmp/{s}/sweep.raw", .{tmp.sub_path});
        defer a.free(path);
        try tmp.dir.writeFile(io, .{ .sub_path = "sweep.raw", .data = "old" });
        // The oracle: each plot encoded alone, back to back.
        var want: Io.Writer.Allocating = .init(a);
        defer want.deinit();
        var session = try Session.init(a, .{ .format = format, .path = path });
        defer session.deinit();
        for ([_][]const u8{ "step 1", "step 2", "step 3" }, 0..) |name, k| {
            const value: [1]f64 = .{@floatFromInt(k)};
            const plot: types.Plot = .{ .title = "sweep", .result = .{
                .plotname = name,
                .varnames = &.{"v(out)"},
                .is_complex = false,
                .npoints = 1,
                .data = &value,
            } };
            try session.publish(io, plot);
            try dispatch.encode(&want.writer, format, plot);
            const during = try tmp.dir.readFileAlloc(io, "sweep.raw", a, .unlimited);
            defer a.free(during);
            try std.testing.expectEqualStrings("old", during);
        }
        try session.finish(io);
        const got = try tmp.dir.readFileAlloc(io, "sweep.raw", a, .unlimited);
        defer a.free(got);
        try std.testing.expectEqualStrings(want.written(), got);
    }
}

test "session: plots never finished leave the previous file intact (#179)" {
    const io = std.testing.io;
    const a = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try a.print(".zig-cache/tmp/{s}/plots.raw", .{tmp.sub_path});
    defer a.free(path);
    try tmp.dir.writeFile(io, .{ .sub_path = "plots.raw", .data = "previous run" });
    const plot: types.Plot = .{ .title = "t", .result = .{ .plotname = "op", .varnames = &.{"v(out)"}, .is_complex = false, .npoints = 1, .data = &.{1} } };
    {
        var session = try Session.init(a, .{ .path = path });
        defer session.deinit();
        try session.publish(io, plot);
        try session.publish(io, plot);
    }
    const kept = try tmp.dir.readFileAlloc(io, "plots.raw", a, .unlimited);
    defer a.free(kept);
    try std.testing.expectEqualStrings("previous run", kept);
    // A device takes each plot in place; there is nothing to rename.
    var device = try Session.init(a, .{ .path = "/dev/null" });
    defer device.deinit();
    try device.publish(io, plot);
    try device.publish(io, plot);
    try std.testing.expectEqual(null, device.pending);
    try device.finish(io);
}

test "session: owned destination and numbered files" {
    const io = std.testing.io;
    const a = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try a.print(".zig-cache/tmp/{s}/plots.csv", .{tmp.sub_path});
    defer a.free(path);
    var session = try Session.init(a, .{ .format = .csv, .path = path });
    defer session.deinit();
    @memset(path, 'x');
    const plot: types.Plot = .{
        .title = "csv",
        .result = .{
            .plotname = "op",
            .varnames = &.{"v(out)"},
            .is_complex = false,
            .npoints = 1,
            .data = &.{1},
        },
    };
    try session.publish(io, plot);
    try session.finish(io);
    try session.publish(io, plot);
    try session.finish(io);
    const first = try tmp.dir.readFileAlloc(io, "plots.csv", a, .unlimited);
    defer a.free(first);
    const second = try tmp.dir.readFileAlloc(io, "plots.csv.2", a, .unlimited);
    defer a.free(second);
    try std.testing.expectEqualStrings(first, second);
}

test "session: validation preserves destination and permits a corrected publication" {
    const io = std.testing.io;
    const a = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try a.print(".zig-cache/tmp/{s}/plot.s1p", .{tmp.sub_path});
    defer a.free(path);
    try tmp.dir.writeFile(io, .{ .sub_path = "plot.s1p", .data = "existing" });
    var session = try Session.init(a, .{ .format = .touchstone, .path = path });
    defer session.deinit();
    var plot: types.Plot = .{
        .title = "s1p",
        .result = .{
            .plotname = "sp",
            .varnames = &.{ "frequency", "S(2,2)" },
            .is_complex = true,
            .npoints = 1,
            .data = &.{ 1e9, 0, 1, 0 },
        },
    };
    try std.testing.expectError(error.NotSParameterData, session.publish(io, plot));
    const old = try tmp.dir.readFileAlloc(io, "plot.s1p", a, .unlimited);
    defer a.free(old);
    try std.testing.expectEqualStrings("existing", old);
    plot.result.varnames = &.{ "frequency", "S(1,1)" };
    try session.publish(io, plot);
    try session.finish(io);
}

test "session: writer failure is terminal and does not acknowledge output" {
    const io = std.testing.io;
    const a = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try a.print(".zig-cache/tmp/{s}/missing/plot.raw", .{tmp.sub_path});
    defer a.free(path);
    var session = try Session.init(a, .{ .path = path });
    defer session.deinit();
    const plot: types.Plot = .{
        .title = "bad path",
        .result = .{
            .plotname = "op",
            .varnames = &.{"v(out)"},
            .is_complex = false,
            .npoints = 1,
            .data = &.{1},
        },
    };
    try std.testing.expectError(error.FileNotFound, session.publish(io, plot));
    try std.testing.expectEqual(@as(u32, 0), session.published);
    try std.testing.expectError(error.DeliveryFailed, session.publish(io, plot));
    try std.testing.expectError(error.DeliveryFailed, session.finish(io));
}

test "session: a streamed plot matches the whole-plot encoding but for the padded count" {
    const io = std.testing.io;
    const a = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try a.print(".zig-cache/tmp/{s}/stream.raw", .{tmp.sub_path});
    defer a.free(path);
    const rows = [_]f64{ 0, 1.5, 1e-9, 2.5 };
    var plot: types.Plot = .{ .title = "t", .result = .{
        .plotname = "Transient Analysis",
        .varnames = &.{ "time", "v(out)" },
        .is_complex = false,
        .npoints = 0,
        .data = &.{},
    } };
    var session = try Session.init(a, .{ .path = path });
    defer session.deinit();
    const w = (try session.beginStream(io, plot)).?;
    try w.writeAll(std.mem.sliceAsBytes(&rows));
    try std.testing.expectError(error.DataLengthMismatch, session.endStream(io, 3, 2));
    try std.testing.expectError(error.DeliveryFailed, session.publish(io, plot));

    var again = try Session.init(a, .{ .path = path });
    defer again.deinit();
    try (try again.beginStream(io, plot)).?.writeAll(std.mem.sliceAsBytes(&rows));
    try again.endStream(io, 2, 2);
    try std.testing.expectEqual(@as(u32, 1), again.published);
    try again.finish(io);
    const streamed = try tmp.dir.readFileAlloc(io, "stream.raw", a, .unlimited);
    defer a.free(streamed);
    plot.result.npoints = 2;
    plot.result.data = &rows;
    var whole: Io.Writer.Allocating = .init(a);
    defer whole.deinit();
    try @import("rawfile.zig").encode(&whole.writer, plot);
    const field = "No. Points: 2";
    const at = std.mem.indexOf(u8, streamed, field).? + field.len;
    try std.testing.expectEqualStrings(whole.written()[0..at], streamed[0..at]);
    try std.testing.expectEqualStrings(&@as([Session.count_width - 1]u8, @splat(' ')), streamed[at..][0 .. Session.count_width - 1]);
    try std.testing.expectEqualSlices(u8, whole.written()[at..], streamed[at + Session.count_width - 1 ..]);
}

test "session: side plots publish in order the bytes of whole plots" {
    for ([_]types.Format{ .binary, .ascii, .print, .csv }) |format| {
        const io = std.testing.io;
        const a = std.testing.allocator;
        var tmp = std.testing.tmpDir(.{});
        defer tmp.cleanup();
        const path = try a.print(".zig-cache/tmp/{s}/sweep.raw", .{tmp.sub_path});
        defer a.free(path);
        const rows = [_][4]f64{ .{ 0, 1.5, 1e-9, 2.5 }, .{ 0, -3, 2e-9, 0.25 } };
        // One file for the concatenating formats, else one per plot.
        var want: Io.Writer.Allocating = .init(a);
        defer want.deinit();
        var want2: Io.Writer.Allocating = .init(a);
        defer want2.deinit();
        var session = try Session.init(a, .{ .format = format, .path = path });
        defer session.deinit();
        // The second point finishes first; a third is dropped.
        const first = (try session.beginSide(io)).?;
        const second = (try session.beginSide(io)).?;
        const dropped = (try session.beginSide(io)).?;
        try dropped.writer().writeAll(std.mem.sliceAsBytes(&rows[0]));
        session.dropSide(dropped);
        try second.writer().writeAll(std.mem.sliceAsBytes(&rows[1]));
        try first.writer().writeAll(std.mem.sliceAsBytes(&rows[0]));
        for ([_]*Session.Stream{ first, second }, [_][]const u8{ "p (x=1)", "p (x=2)" }, &rows) |side, name, *data| {
            var plot: types.Plot = .{ .title = "t", .result = .{ .plotname = name, .varnames = &.{ "time", "v(out)" }, .is_complex = false, .npoints = 2, .data = &.{} } };
            try session.endSide(io, side, plot);
            plot.result.data = data;
            try dispatch.encode(if (dispatch.concatenates(format) or side == first) &want.writer else &want2.writer, format, plot);
        }
        try std.testing.expectEqual(@as(usize, 0), session.sides.items.len);
        try session.finish(io);
        const got = try tmp.dir.readFileAlloc(io, "sweep.raw", a, .unlimited);
        defer a.free(got);
        try std.testing.expectEqualStrings(want.written(), got);
        if (dispatch.concatenates(format)) continue;
        const got2 = try tmp.dir.readFileAlloc(io, "sweep.raw.2", a, .unlimited);
        defer a.free(got2);
        try std.testing.expectEqualStrings(want2.written(), got2);
    }
    // Without a path the rows are counted, and checked.
    var none = try Session.init(std.testing.allocator, .{});
    defer none.deinit();
    const side = (try none.beginSide(std.testing.io)).?;
    try side.writer().writeAll(std.mem.sliceAsBytes(&[_]f64{ 0, 1, 2 }));
    const plot: types.Plot = .{ .title = "t", .result = .{ .plotname = "p", .varnames = &.{ "time", "v(out)" }, .is_complex = false, .npoints = 1, .data = &.{} } };
    try std.testing.expectError(error.DataLengthMismatch, none.endSide(std.testing.io, side, plot));
    try std.testing.expectEqual(@as(u32, 0), none.published);
    _ = try none.beginSide(std.testing.io);
}

test "session: without a path plots are counted, not written" {
    var session = try Session.init(std.testing.allocator, .{ .format = .csv });
    defer session.deinit();
    const plot: types.Plot = .{ .title = "t", .result = .{ .plotname = "op", .varnames = &.{"v(out)"}, .is_complex = false, .npoints = 1, .data = &.{1} } };
    try session.publish(std.testing.io, plot);
    try session.publish(std.testing.io, plot);
    try std.testing.expectEqual(@as(u32, 2), session.published);
    try std.testing.expectEqual(null, try session.beginStream(std.testing.io, plot));
    session.published = std.math.maxInt(u32);
    try std.testing.expectError(error.TooManyPlots, session.publish(std.testing.io, plot));
}

test "session: a stream without a path drops its rows and counts one plot" {
    var session = try Session.init(std.testing.allocator, .{});
    defer session.deinit();
    const plot: types.Plot = .{ .title = "t", .result = .{ .plotname = "Transient Analysis", .varnames = &.{ "time", "v(out)" }, .is_complex = false, .npoints = 0, .data = &.{} } };
    const w = (try session.beginStream(std.testing.io, plot)).?;
    try w.writeAll(std.mem.sliceAsBytes(&[_]f64{ 0, 1 }));
    try session.endStream(std.testing.io, 1, 2);
    try std.testing.expectEqual(@as(u32, 1), session.published);
    try std.testing.expectEqual(null, session.stream);
    try std.testing.expectEqual(null, try session.beginStream(std.testing.io, plot));
}

test "session: a stream without a path checks its row count too" {
    var session = try Session.init(std.testing.allocator, .{});
    defer session.deinit();
    const plot: types.Plot = .{ .title = "t", .result = .{ .plotname = "Transient Analysis", .varnames = &.{ "time", "v(out)" }, .is_complex = false, .npoints = 0, .data = &.{} } };
    const w = (try session.beginStream(std.testing.io, plot)).?;
    try w.writeAll(std.mem.sliceAsBytes(&[_]f64{ 0, 1, 2 }));
    try std.testing.expectError(error.DataLengthMismatch, session.endStream(std.testing.io, 2, 2));
    try std.testing.expectEqual(@as(u32, 0), session.published);
    try std.testing.expectEqual(null, session.stream);
}

test "session: an aborted stream leaves the destination as it was" {
    const io = std.testing.io;
    const a = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try a.print(".zig-cache/tmp/{s}/plot.raw", .{tmp.sub_path});
    defer a.free(path);
    try tmp.dir.writeFile(io, .{ .sub_path = "plot.raw", .data = "existing" });
    var session = try Session.init(a, .{ .path = path });
    defer session.deinit();
    const plot: types.Plot = .{ .title = "t", .result = .{ .plotname = "Transient Analysis", .varnames = &.{"time"}, .is_complex = false, .npoints = 0, .data = &.{} } };
    try (try session.beginStream(io, plot)).?.writeAll(std.mem.sliceAsBytes(&[_]f64{1}));
    session.abortStream();
    try std.testing.expectEqual(@as(u32, 0), session.published);
    const old = try tmp.dir.readFileAlloc(io, "plot.raw", a, .unlimited);
    defer a.free(old);
    try std.testing.expectEqualStrings("existing", old);
}

test "session: every allocation failure is reported and leaks nothing" {
    const a = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try a.print(".zig-cache/tmp/{s}", .{tmp.sub_path});
    defer a.free(dir);
    const S = struct {
        fn run(gpa: std.mem.Allocator, base: []const u8) !void {
            const io = std.testing.io;
            var path_buf: [256]u8 = undefined;
            const rows = [_]f64{ 0, 1 };
            const stream: types.Plot = .{ .title = "t", .result = .{ .plotname = "tran", .varnames = &.{ "time", "v(a)" }, .is_complex = false, .npoints = 0, .data = &.{} } };
            const op: types.Plot = .{ .title = "t", .result = .{ .plotname = "op", .varnames = &.{"v(a)"}, .is_complex = false, .npoints = 1, .data = &.{1} } };
            // Streamed binary raw, then an appended plot, then the rename.
            var raw = try Session.init(gpa, .{ .path = try std.fmt.bufPrint(&path_buf, "{s}/a.raw", .{base}) });
            defer raw.deinit();
            try (try raw.beginStream(io, stream)).?.writeAll(std.mem.sliceAsBytes(&rows));
            try raw.endStream(io, 1, 2);
            try raw.publish(io, op);
            try raw.finish(io);
            // Numbered one-plot files, and a pathless stream.
            var csv = try Session.init(gpa, .{ .format = .csv, .path = try std.fmt.bufPrint(&path_buf, "{s}/a.csv", .{base}) });
            defer csv.deinit();
            try csv.publish(io, op);
            try csv.publish(io, op);
            var none = try Session.init(gpa, .{});
            defer none.deinit();
            try (try none.beginStream(io, stream)).?.writeAll(std.mem.sliceAsBytes(&rows));
            try none.endStream(io, 1, 2);
        }
    };
    try std.testing.checkAllAllocationFailures(a, S.run, .{dir});
}
