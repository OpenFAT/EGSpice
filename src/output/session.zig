//! Delivery of plots to one output selection, in publish order: plots in,
//! files out. The caller keeps each result alive until `publish` returns, or
//! streams a binary raw plot's rows as they are computed (`beginStream`), or
//! any number of plots' rows to temporary files from any thread
//! (`beginSide`). Plots that share one file (binary and ASCII raw, `.print`)
//! are appended to one replacement, each written once, and `finish` renames
//! it over the destination (#179).
//! Cites: ngspice 44 manual §17.5 (batch raw files leave `No. Points:` to be
//! filled in).
//!
//! Three lifetimes, three records: a `Replacement` lives from the first
//! plot after a `finish` to the next `finish`; the streamed plot 0 from
//! `beginStream` to `endStream`, which hands its replacement over; a side
//! (`Stream`) from `beginSide` to `endSide` or `dropSide`.
//!
//! Contents:
//! - `pub const Session = struct` with `pub const Stream`, `pub fn writer(*Stream) *Io.Writer`,
//!   `pub const Error`, `pub fn init(Allocator, Selection) Allocator.Error!Session`,
//!   `pub fn deinit(*Session) void`, `pub fn publish(*Session, Io, Plot) Error!void`,
//!   `pub fn beginStream(*Session, Io, Plot) Error!?*Io.Writer`,
//!   `pub fn endStream(*Session, Io, usize, usize) Error!void`, `pub fn abortStream(*Session) void`,
//!   `pub fn beginSide(*Session, Io) Error!?*Stream`, `pub fn endSide(*Session, Io, *Stream, Plot) Error!void`,
//!   `pub fn dropSide(*Session, *Stream) void`, `pub fn finish(*Session, Io) Error!void`

const std = @import("std");
const Io = std.Io;
const Allocator = std.mem.Allocator;
const assert = std.debug.assert;
const plots = @import("plot.zig");
const files = @import("write.zig");
const rawfile = @import("rawfile.zig");
const Plot = plots.Plot;
const Format = plots.Format;

/// One output destination and the count of plots delivered to it. Not
/// thread-safe, except that distinct side streams' writers may be written
/// from different threads.
pub const Session = struct {
    /// Owns `selection.path` and every record below.
    gpa: Allocator,
    /// Fixed at `init`; the path is the session's own copy.
    selection: plots.Selection,
    /// Plots delivered so far; the next one is plot number `published`.
    /// At most maxInt(u32) plots, numbered 0 through maxInt(u32) - 1.
    published: u32 = 0,
    /// `failed` after a writer error; every later call reports it.
    state: enum(u8) { ready, failed } = .ready,
    /// The plot `beginStream` opened, until `endStream` or `abortStream`.
    stream: ?*Streamed = null,
    /// The replacement of a file that holds several plots, from the first
    /// plot published since the last `finish` until the next one, which
    /// renames it over the destination; until then the destination keeps its
    /// old bytes. A batch after a `finish` starts with a copy of the file.
    pending: ?*Replacement = null,
    /// The open `beginSide` files, dropped by `deinit`.
    sides: std.ArrayList(*Stream) = .empty,
    /// The next side file's number.
    side_serial: u32 = 0,

    /// A side plot's rows: a temporary file beside the destination, or a
    /// discard that counts them without a path.
    pub const Stream = struct {
        /// The side file and its name, which `drop` deletes; null without a path.
        file: ?Io.File,
        name: []u8,
        file_writer: Io.File.Writer,
        discard: Io.Writer.Discarding,
        buffer: [buffer_len]u8,

        /// Returns where rows go: the side file, else a discard that only
        /// counts them. Valid until `endSide`, `dropSide` or `deinit`.
        pub fn writer(s: *Stream) *Io.Writer {
            return if (s.file != null) &s.file_writer.interface else &s.discard.writer;
        }
    };

    /// Every way a session call fails. A validation error, `TooManyPlots`
    /// and `DeliveryFailed` write nothing; any other error from a call with
    /// a path fails the session (`DeliveryFailed` from then on), leaving the
    /// destination as it was. `DataLengthMismatch` also reports a streamed or
    /// side plot whose row count disagrees with its shape.
    pub const Error = plots.ValidationError || files.Error || Allocator.Error ||
        Io.Reader.Error || Io.Reader.StreamRemainingError || Io.File.WritePositionalError || error{
        /// The session failed earlier, or a side's destination stopped
        /// being a regular file.
        DeliveryFailed,
        /// maxInt(u32) plots were delivered already.
        TooManyPlots,
    };

    /// Copies `selection.path`; performs no I/O. Free with `deinit`.
    pub fn init(gpa: Allocator, selection: plots.Selection) Allocator.Error!Session {
        const path = if (selection.path) |path| try gpa.dupe(u8, path) else null;
        return .{ .gpa = gpa, .selection = .{ .format = selection.format, .path = path } };
    }

    /// Drops an open stream, every side and any plots not yet `finish`ed,
    /// leaving the destination as it was, and frees the path copy.
    pub fn deinit(self: *Session) void {
        self.abortStream();
        for (self.sides.items) |s| self.dropStream(s);
        self.sides.deinit(self.gpa);
        if (self.pending) |r| self.dropReplacement(r);
        if (self.selection.path) |path| self.gpa.free(path);
        self.* = undefined;
    }

    /// Writes the next plot. For binary and ASCII raw and `.print` listings
    /// every plot goes into one file, `path`, which `finish` replaces; for
    /// every other format plot 0 replaces `path` and later ones
    /// `path.<number + 1>`, each at once. Without a path the plot is only
    /// counted.
    pub fn publish(self: *Session, io: Io, plot: Plot) Error!void {
        if (self.state == .failed) return error.DeliveryFailed;
        if (self.published == std.math.maxInt(u32)) return error.TooManyPlots;
        try plots.validatePlot(self.selection.format, plot);
        if (self.selection.path) |path| self.deliver(io, path, plot) catch |err| {
            self.state = .failed;
            return err;
        };
        self.published += 1;
    }

    /// Opens plot 0 in binary raw for rows the caller writes to the returned
    /// writer as native-endian f64s, `plot.result.data` being empty. Null
    /// when the selection cannot stream (another format, a plot already
    /// delivered, a failed session, a destination that is not a regular
    /// file); publish the whole plot then. The header leaves `No. Points:`
    /// blank, as ngspice's batch raw file does, for `endStream` to fill in.
    /// The writer is valid until `endStream`, `abortStream` or `deinit`.
    /// Asserts that no plot is pending and no stream is open.
    pub fn beginStream(self: *Session, io: Io, plot: Plot) Error!?*Io.Writer {
        if (self.selection.format != .binary or self.published != 0 or self.state == .failed) return null;
        assert(self.pending == null);
        assert(self.stream == null);
        try plots.validatePlot(.binary, plot);
        const s = try self.gpa.create(Streamed);
        errdefer self.gpa.destroy(s);
        s.* = .{ .replacement = null, .discard = .init(&.{}), .complex = plot.result.is_complex };
        const path = self.selection.path orelse {
            self.stream = s;
            return &s.discard.writer;
        };
        const r = try self.openReplacement(io, path, .fresh) orelse {
            self.gpa.destroy(s);
            return null;
        };
        errdefer self.dropReplacement(r);
        const w = &r.file.interface;
        try rawfile.writeHeaderHead(w, plot);
        s.count_at = r.file.logicalPos();
        try w.splatByteAll(' ', count_width);
        try rawfile.writeHeaderTail(w, plot, .binary);
        s.rows_at = r.file.logicalPos();
        s.replacement = r;
        self.stream = s;
        return w;
    }

    /// Finishes the streamed plot: checks that `npoints` rows of `columns`
    /// columns arrived and fills in the point count; the plots published
    /// after it append to the same replacement, which `finish` renames over
    /// the destination. Counts as one `publish`. The stream is closed whether
    /// or not it succeeds. Without a path a wrong row count is returned and
    /// the session stays usable. Asserts that a stream is open.
    pub fn endStream(self: *Session, io: Io, npoints: usize, columns: usize) Error!void {
        const s = self.stream.?;
        defer self.abortStream();
        const want = @as(u64, npoints) * columns * @sizeOf(f64) * @as(u64, if (s.complex) 2 else 1);
        const r = s.replacement orelse {
            // A pathless stream counts what it dropped (#88).
            if (s.discard.fullCount() != want) return error.DataLengthMismatch;
            self.published += 1;
            return;
        };
        errdefer self.state = .failed;
        try r.file.interface.flush();
        if (r.file.logicalPos() - s.rows_at != want) return error.DataLengthMismatch;
        var digits: [count_width]u8 = undefined;
        // count_width digits hold any u64.
        const count = std.fmt.bufPrint(&digits, "{d}", .{npoints}) catch unreachable;
        try r.file.file.writePositionalAll(io, count, s.count_at);
        s.replacement = null;
        self.pending = r;
        self.published += 1;
        assert(self.pending != null);
    }

    /// Drops the streamed plot, if any, leaving the destination as it was.
    pub fn abortStream(self: *Session) void {
        const s = self.stream orelse return;
        if (s.replacement) |r| self.dropReplacement(r);
        self.gpa.destroy(s);
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
    /// `deinit`. Invalidates nothing; `sides` may reallocate.
    pub fn beginSide(self: *Session, io: Io) Error!?*Stream {
        if (!sideFormat(self.selection.format) or self.state == .failed) return null;
        if (self.selection.path) |path| if (files.isDevice(io, path)) return null;
        try self.sides.ensureUnusedCapacity(self.gpa, 1);
        const s = try self.gpa.create(Stream);
        errdefer self.gpa.destroy(s);
        s.file = null;
        s.name = &.{};
        s.discard = .init(&.{});
        if (self.selection.path) |path| {
            s.name, s.file = try self.createSideFile(io, path);
            s.file_writer = s.file.?.writer(io, &s.buffer);
        }
        self.sides.appendAssumeCapacity(s);
        return s;
    }

    /// Publishes side `s` as the next plot: `plot`'s header (its `data`
    /// empty, `npoints` the rows written) then the side's rows, appended
    /// to the replacement, or written to the plot's own file, as `publish`
    /// writes them. The bytes are those `publish` writes for the whole
    /// plot. Closes `s` whether or not it succeeds. Asserts that `s` is one
    /// of this session's open sides.
    pub fn endSide(self: *Session, io: Io, s: *Stream, plot: Plot) Error!void {
        defer self.dropSide(s);
        if (self.state == .failed) return error.DeliveryFailed;
        if (self.published == std.math.maxInt(u32)) return error.TooManyPlots;
        try plots.validateSchema(self.selection.format, plot.schema());
        const want = @as(u64, plot.result.npoints) * plot.stride() * @sizeOf(f64);
        if (s.file == null) {
            if (s.discard.fullCount() != want) return error.DataLengthMismatch;
            self.published += 1;
            return;
        }
        errdefer self.state = .failed;
        try s.file_writer.interface.flush();
        if (s.file_writer.logicalPos() != want) return error.DataLengthMismatch;
        const path = self.selection.path.?;
        if (files.concatenates(self.selection.format)) {
            const r = self.pending orelse try self.openReplacement(io, path, self.keepOld()) orelse
                return error.DeliveryFailed;
            self.pending = r;
            try self.copySide(io, &r.file.interface, s, plot);
        } else {
            const numbered = try self.plotPath(path);
            defer if (numbered) |n| self.gpa.free(n);
            var atomic = try Io.Dir.cwd().createFileAtomic(io, numbered orelse path, .{ .replace = true });
            defer atomic.deinit(io);
            var buf: [buffer_len]u8 = undefined;
            var fw = atomic.file.writer(io, &buf);
            try self.copySide(io, &fw.interface, s, plot);
            try fw.interface.flush();
            try atomic.replace(io);
        }
        self.published += 1;
    }

    /// Closes side `s` without publishing it. Asserts that `s` is one of
    /// this session's open sides.
    pub fn dropSide(self: *Session, s: *Stream) void {
        const at = std.mem.indexOfScalar(*Stream, self.sides.items, s).?;
        _ = self.sides.swapRemove(at);
        self.dropStream(s);
    }

    /// Publishes the plots appended since the last `finish` with one rename
    /// over the destination, so a reader sees the old file or every plot.
    /// Does not seal the session: queries appended later may publish more
    /// plots, and the next `finish` adds them to the file.
    pub fn finish(self: *Session, io: Io) Error!void {
        if (self.state == .failed) return error.DeliveryFailed;
        const r = self.pending orelse return;
        self.pending = null;
        defer self.dropReplacement(r);
        errdefer self.state = .failed;
        try r.file.interface.flush();
        try r.atomic.replace(io);
    }

    // ---- private ----

    /// Room for the point count of a streamed plot, as many digits as u64 has.
    const count_width = 20;
    const buffer_len = 64 * 1024;
    /// Side file names tried before `beginSide` gives up.
    const side_attempts_max = 1 << 16;

    comptime {
        assert(std.fmt.count("{d}", .{std.math.maxInt(u64)}) == count_width);
    }

    /// The new contents of a destination, renamed over it at `finish`.
    const Replacement = struct {
        atomic: Io.File.Atomic,
        file: Io.File.Writer,
        buffer: [buffer_len]u8,
    };

    /// Plot 0 in binary raw, written while its query runs: the header with
    /// the point count left blank, then rows as they arrive. Without a path
    /// the rows are counted by `discard` and dropped.
    const Streamed = struct {
        replacement: ?*Replacement,
        discard: Io.Writer.Discarding,
        complex: bool,
        /// File offsets of the blank point count and of the first row.
        count_at: u64 = 0,
        rows_at: u64 = 0,
    };

    /// What a replacement starts with: the destination's bytes, or nothing.
    const Start = enum { fresh, keep_old };

    fn deliver(self: *Session, io: Io, path: []const u8, plot: Plot) Error!void {
        const format = self.selection.format;
        if (!files.concatenates(format)) {
            const numbered = try self.plotPath(path);
            defer if (numbered) |n| self.gpa.free(n);
            return files.write(io, numbered orelse path, format, plot);
        }
        // A device or FIFO cannot be renamed over: it takes each plot as it comes.
        const r = self.pending orelse try self.openReplacement(io, path, self.keepOld()) orelse
            return files.write(io, path, format, plot);
        self.pending = r;
        try files.encode(&r.file.interface, format, plot);
    }

    /// A batch after a `finish` appends to what that `finish` published.
    fn keepOld(self: *const Session) Start {
        return if (self.published != 0) .keep_old else .fresh;
    }

    /// The next plot's own file in a format that holds one plot: `path`
    /// for plot 0 (null), else `path.<number + 1>`, which the caller frees.
    fn plotPath(self: *Session, path: []const u8) Allocator.Error!?[]u8 {
        if (self.published == 0) return null;
        return try self.gpa.print("{s}.{d}", .{ path, self.published + 1 });
    }

    /// A replacement for the file at `path`, holding the file's bytes with
    /// `.keep_old`; null when `path` is a device or FIFO.
    fn openReplacement(self: *Session, io: Io, path: []const u8, start: Start) Error!?*Replacement {
        if (files.isDevice(io, path)) return null;
        const r = try self.gpa.create(Replacement);
        errdefer self.gpa.destroy(r);
        r.atomic = try Io.Dir.cwd().createFileAtomic(io, path, .{ .replace = true });
        errdefer r.atomic.deinit(io);
        r.file = r.atomic.file.writer(io, &r.buffer);
        if (start == .fresh) return r;
        const old = Io.Dir.cwd().openFile(io, path, .{}) catch |err| switch (err) {
            error.FileNotFound => return r,
            else => return err, // else: any other open failure is the caller's
        };
        defer old.close(io);
        var rbuf: [8192]u8 = undefined;
        var reader = old.reader(io, &rbuf);
        _ = try reader.interface.streamRemaining(&r.file.interface);
        return r;
    }

    /// Closes `r` and deletes its temporary file, leaving the destination
    /// as it was.
    fn dropReplacement(self: *Session, r: *Replacement) void {
        r.atomic.deinit(r.file.io);
        self.gpa.destroy(r);
    }

    /// `<path>.<n>.side`, readable for `endSide`, with the first `n` free.
    /// Caller owns the name and the file.
    fn createSideFile(self: *Session, io: Io, path: []const u8) Error!struct { []u8, Io.File } {
        for (0..side_attempts_max) |_| {
            const name = try self.gpa.print("{s}.{d}.side", .{ path, self.side_serial });
            errdefer self.gpa.free(name);
            const file = Io.Dir.cwd().createFile(io, name, .{ .read = true, .exclusive = true }) catch |err| switch (err) {
                error.PathAlreadyExists => {
                    self.side_serial +%= 1;
                    self.gpa.free(name);
                    continue;
                },
                else => return err, // else: any other create failure is the caller's
            };
            self.side_serial +%= 1;
            return .{ name, file };
        }
        return error.PathAlreadyExists;
    }

    /// Closes side `s`, deletes its file and frees it.
    fn dropStream(self: *Session, s: *Stream) void {
        if (s.file) |file| {
            const io = s.file_writer.io;
            file.close(io);
            // A side file left behind is litter, not lost output: say so and go on.
            Io.Dir.cwd().deleteFile(io, s.name) catch |err| std.log.warn("cannot delete {s}: {t}", .{ s.name, err });
            self.gpa.free(s.name);
        }
        self.gpa.destroy(s);
    }

    /// Writes `plot`'s header then side `s`'s rows to `w` in the
    /// selection's format. Asserts a side format and a side with a file.
    fn copySide(self: *Session, io: Io, w: *Io.Writer, s: *Stream, plot: Plot) Error!void {
        const format = self.selection.format;
        assert(sideFormat(format));
        var rbuf: [buffer_len]u8 = undefined;
        var reader = s.file.?.reader(io, &rbuf);
        switch (format) {
            .binary => {
                try rawfile.writeHeader(w, plot, .binary);
                _ = try reader.interface.streamRemaining(w);
                return;
            },
            .ascii => try rawfile.writeHeader(w, plot, .ascii),
            .csv => try @import("csv.zig").writeHeader(w, plot),
            .print => try @import("spice_print.zig").writeHeader(w, plot),
            .touchstone, .psf, .fsdb, .sst2, .citi => unreachable, // `beginSide` opens only `sideFormat`s
        }
        // Whole rows at a time, about 64 KiB of them.
        const stride = plot.stride();
        const per = @max(1, 8192 / stride);
        const rows = try self.gpa.alloc(f64, per * stride);
        defer self.gpa.free(rows);
        var pt: usize = 0;
        while (pt < plot.result.npoints) : (pt += per) {
            const k = @min(per, plot.result.npoints - pt);
            const chunk = rows[0 .. k * stride];
            try reader.interface.readSliceAll(std.mem.sliceAsBytes(chunk));
            switch (format) {
                .ascii => try @import("ascii_raw.zig").writeRows(w, plot.result.is_complex, stride, pt, chunk),
                .csv => try @import("csv.zig").writeRows(w, stride, chunk),
                .print => try @import("spice_print.zig").writeRows(w, stride, pt, chunk),
                .binary, .touchstone, .psf, .fsdb, .sst2, .citi => unreachable, // binary returned above
            }
        }
    }

    /// The formats a side stream writes: a header, then rows.
    fn sideFormat(format: Format) bool {
        return switch (format) {
            .binary, .ascii, .csv, .print => true,
            .touchstone, .psf, .fsdb, .sst2, .citi => false,
        };
    }
};

test "session: finish and later publication preserve raw append order" {
    for ([_]Format{ .binary, .ascii }) |format| {
        const io = std.testing.io;
        const a = std.testing.allocator;
        var tmp = std.testing.tmpDir(.{});
        defer tmp.cleanup();
        const path = try a.print(".zig-cache/tmp/{s}/plots.raw", .{tmp.sub_path});
        defer a.free(path);
        var session = try Session.init(a, .{ .format = format, .path = path });
        defer session.deinit();
        const first: Plot = .{
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
    for ([_]Format{ .binary, .ascii, .print }) |format| {
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
            const plot: Plot = .{ .title = "sweep", .result = .{
                .plotname = name,
                .varnames = &.{"v(out)"},
                .is_complex = false,
                .npoints = 1,
                .data = &value,
            } };
            try session.publish(io, plot);
            try files.encode(&want.writer, format, plot);
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
    const plot: Plot = .{ .title = "t", .result = .{ .plotname = "op", .varnames = &.{"v(out)"}, .is_complex = false, .npoints = 1, .data = &.{1} } };
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
    const plot: Plot = .{
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
    var plot: Plot = .{
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
    const plot: Plot = .{
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
    var plot: Plot = .{ .title = "t", .result = .{
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
    try rawfile.encode(&whole.writer, plot);
    const field = "No. Points: 2";
    const at = std.mem.indexOf(u8, streamed, field).? + field.len;
    try std.testing.expectEqualStrings(whole.written()[0..at], streamed[0..at]);
    try std.testing.expectEqualStrings(&@as([Session.count_width - 1]u8, @splat(' ')), streamed[at..][0 .. Session.count_width - 1]);
    try std.testing.expectEqualSlices(u8, whole.written()[at..], streamed[at + Session.count_width - 1 ..]);
}

test "session: side plots publish in order the bytes of whole plots" {
    for ([_]Format{ .binary, .ascii, .print, .csv }) |format| {
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
            var plot: Plot = .{ .title = "t", .result = .{ .plotname = name, .varnames = &.{ "time", "v(out)" }, .is_complex = false, .npoints = 2, .data = &.{} } };
            try session.endSide(io, side, plot);
            plot.result.data = data;
            try files.encode(if (files.concatenates(format) or side == first) &want.writer else &want2.writer, format, plot);
        }
        try std.testing.expectEqual(@as(usize, 0), session.sides.items.len);
        try session.finish(io);
        const got = try tmp.dir.readFileAlloc(io, "sweep.raw", a, .unlimited);
        defer a.free(got);
        try std.testing.expectEqualStrings(want.written(), got);
        if (files.concatenates(format)) continue;
        const got2 = try tmp.dir.readFileAlloc(io, "sweep.raw.2", a, .unlimited);
        defer a.free(got2);
        try std.testing.expectEqualStrings(want2.written(), got2);
    }
    // Without a path the rows are counted, and checked.
    var none = try Session.init(std.testing.allocator, .{});
    defer none.deinit();
    const side = (try none.beginSide(std.testing.io)).?;
    try side.writer().writeAll(std.mem.sliceAsBytes(&[_]f64{ 0, 1, 2 }));
    const plot: Plot = .{ .title = "t", .result = .{ .plotname = "p", .varnames = &.{ "time", "v(out)" }, .is_complex = false, .npoints = 1, .data = &.{} } };
    try std.testing.expectError(error.DataLengthMismatch, none.endSide(std.testing.io, side, plot));
    try std.testing.expectEqual(@as(u32, 0), none.published);
    _ = try none.beginSide(std.testing.io);
}

test "session: without a path plots are counted, not written" {
    var session = try Session.init(std.testing.allocator, .{ .format = .csv });
    defer session.deinit();
    const plot: Plot = .{ .title = "t", .result = .{ .plotname = "op", .varnames = &.{"v(out)"}, .is_complex = false, .npoints = 1, .data = &.{1} } };
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
    const plot: Plot = .{ .title = "t", .result = .{ .plotname = "Transient Analysis", .varnames = &.{ "time", "v(out)" }, .is_complex = false, .npoints = 0, .data = &.{} } };
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
    const plot: Plot = .{ .title = "t", .result = .{ .plotname = "Transient Analysis", .varnames = &.{ "time", "v(out)" }, .is_complex = false, .npoints = 0, .data = &.{} } };
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
    const plot: Plot = .{ .title = "t", .result = .{ .plotname = "Transient Analysis", .varnames = &.{"time"}, .is_complex = false, .npoints = 0, .data = &.{} } };
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
            const stream: Plot = .{ .title = "t", .result = .{ .plotname = "tran", .varnames = &.{ "time", "v(a)" }, .is_complex = false, .npoints = 0, .data = &.{} } };
            const op: Plot = .{ .title = "t", .result = .{ .plotname = "op", .varnames = &.{"v(a)"}, .is_complex = false, .npoints = 1, .data = &.{1} } };
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
