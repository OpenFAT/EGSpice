//! A finished main session in, the files an HSPICE deck asks for out: the
//! `.save` operating point as `.nodeset`/`.ic` cards, and each `.stim`
//! card's PWL sources or `.data` table. Both land beside the output file,
//! as HSPICE writes them beside its listing.
//! Cites: HSPICE Reference Manual, `.SAVE` and `.STIM` (behaviour as the
//! deck cards in src/frontend document it).
//!
//! Contents:
//! - `pub fn savePath(arena: Allocator, save_op: ?core.SaveOp, out_dir: []const u8, origin: []const u8) Allocator.Error!?[]const u8`
//! - `pub fn writeSave(gpa: Allocator, io: std.Io, session: *const Session, title: []const u8, path: []const u8, save: core.SaveOp) !void`
//! - `pub fn writeStim(gpa: Allocator, io: std.Io, session: *const Session, title: []const u8, stim: core.Stim, origin: []const u8, out_dir: []const u8) !void`

const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;
const analysis = @import("analysis");
const core = @import("core");
const output = @import("output");
const Session = analysis.session.Session;

/// A result row to save, and the fraction of the way to the next row.
const SavedPoint = struct { result: output.Result, row: usize, frac: f64 };

/// Where `.save` writes, resolved against `out_dir`; null without the card.
/// `FILE=x.ic` names the `.ic#` family (one file per `.alter` run in
/// HSPICE); ESPice writes `#` = 0, which `.load FILE=x.ic` reads. The path
/// lives in `arena`.
pub fn savePath(arena: Allocator, save_op: ?core.SaveOp, out_dir: []const u8, origin: []const u8) Allocator.Error!?[]const u8 {
    const save = save_op orelse return null;
    const name = if (save.file) |f|
        try std.mem.concat(arena, u8, &.{ f, if (std.ascii.eqlIgnoreCase(std.fs.path.extension(f), ".ic")) "0" else "" })
    else
        try std.mem.concat(arena, u8, &.{ std.fs.path.stem(origin), ".ic0" });
    const path = try std.fs.path.resolve(arena, &.{ out_dir, name });
    assert(path.len != 0);
    return path;
}

/// Writes the node voltages of `session`'s operating point to `path` as
/// `.nodeset` or `.ic` cards: the first `.op` result, else row 0 of the
/// first transient (its operating point) or DC sweep (its first point, as
/// HSPICE saves only a sweep's first). `save.time` > 0 reads the first
/// transient at that time, interpolated. Nothing to save only warns.
/// Fails on a session lookup, out of memory (`error.WriteFailed` from the
/// text buffer), or the file write.
pub fn writeSave(gpa: Allocator, io: std.Io, session: *const Session, title: []const u8, path: []const u8, save: core.SaveOp) !void {
    assert(path.len != 0);
    const op = try savedPoint(session, save) orelse {
        std.log.warn(".save: no {s} result to save", .{if (save.time > 0) "transient" else "operating point"});
        return;
    };
    assert(op.row < op.result.npoints);
    var text: std.Io.Writer.Allocating = .init(gpa);
    defer text.deinit();
    const w = &text.writer;
    try w.print("* {s}\n* operating point written by .save\n", .{title});
    const width = op.result.varnames.len;
    for (op.result.varnames, 0..) |name, k| {
        if (!std.mem.startsWith(u8, name, "v(") or std.mem.indexOfScalar(u8, name, ',') != null) continue;
        if (save.top_only and std.mem.indexOfScalar(u8, name, '.') != null) continue;
        const v0 = op.result.data[op.row * width + k];
        const v = if (op.frac == 0) v0 else v0 + op.frac * (op.result.data[(op.row + 1) * width + k] - v0);
        try w.print("{s} {s}={e}\n", .{ if (save.ic) ".ic" else ".nodeset", name, v });
    }
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = text.written() });
}

/// Writes `stim`'s file from `session`'s first non-empty transient: one PWL
/// source per signal, or one `.data` table with a row per time. No
/// transient, or a signal with no vector, only warns. The file is named
/// `<file or deck stem>.<pwl|dat><serial>_tr0` under `out_dir`.
/// Fails as `writeSave` does.
pub fn writeStim(gpa: Allocator, io: std.Io, session: *const Session, title: []const u8, stim: core.Stim, origin: []const u8, out_dir: []const u8) !void {
    if (!stim.data) assert(stim.signals.len == stim.nodes.len);
    const res = try firstTransient(session) orelse {
        std.log.warn(".stim: no transient result to write", .{});
        return;
    };
    assert(res.npoints != 0);
    var times: std.ArrayList(f64) = .empty;
    defer times.deinit(gpa);
    try stimTimes(gpa, stim, res, &times);
    // Column-major: signal k's samples at values[k * n ..].
    const n = times.items.len;
    const values = try gpa.alloc(f64, n * stim.signals.len);
    defer gpa.free(values);
    for (stim.signals, 0..) |m, k| output.sampleMeasure(m.first, res, times.items, values[k * n ..][0..n]) catch {
        std.log.warn(".stim: no such vector for {s}", .{m.name});
        return;
    };
    var text: std.Io.Writer.Allocating = .init(gpa);
    defer text.deinit();
    const w = &text.writer;
    try w.print("* {s}\n* written by .stim\n", .{title});
    if (stim.data) {
        try w.print(".data {s}{s}", .{ stim.dataname, if (stim.indepout) " time" else "" });
        for (stim.signals) |m| try w.print(" {s}", .{m.name});
        for (times.items, 0..) |t, i| {
            try w.writeAll("\n+");
            if (stim.indepout) try w.print(" {e}", .{t});
            for (0..stim.signals.len) |k| try w.print(" {e}", .{values[k * n + i]});
        }
        try w.writeAll("\n.enddata\n");
    } else for (stim.signals, stim.nodes, 0..) |m, pair, k| {
        try w.print("{s} {s} {s} pwl(", .{ m.name, pair[0], pair[1] });
        for (times.items, 0..) |t, i| try w.print("\n+ {e} {e}", .{ t, values[k * n + i] });
        try w.writeAll(")\n");
    }
    const name = try std.fmt.allocPrint(gpa, "{s}.{s}{d}_tr0", .{ stim.file orelse std.fs.path.stem(origin), if (stim.data) "dat" else "pwl", stim.serial });
    defer gpa.free(name);
    const path = try std.fs.path.resolve(gpa, &.{ out_dir, name });
    defer gpa.free(path);
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = text.written() });
}

// ---- private ----

/// The row `writeSave` reads, null when no completed result has one.
fn savedPoint(session: *const Session, save: core.SaveOp) !?SavedPoint {
    for (session.outputs.items) |id| {
        const info = try session.info(id);
        if (info.status != .complete) continue;
        const res = try session.result(id);
        if (res.is_complex or res.npoints == 0) continue;
        if (save.time > 0) {
            if (info.kind != .tran) continue;
            const width = res.varnames.len;
            var i: usize = 0;
            while (i + 1 < res.npoints and res.data[(i + 1) * width] < save.time) i += 1;
            if (i + 1 >= res.npoints) continue;
            const t0 = res.data[i * width];
            const t1 = res.data[(i + 1) * width];
            return .{ .result = res, .row = i, .frac = if (t1 > t0) (save.time - t0) / (t1 - t0) else 0 };
        }
        switch (info.kind) {
            .op, .tran, .dc => return .{ .result = res, .row = 0, .frac = 0 },
            else => {},
        }
    }
    return null;
}

/// The first completed transient with points, null when there is none.
fn firstTransient(session: *const Session) !?output.Result {
    for (session.outputs.items) |id| {
        const info = try session.info(id);
        if (info.status != .complete or info.kind != .tran) continue;
        const r = try session.result(id);
        if (r.npoints != 0) return r;
    }
    return null;
}

/// Appends the times `stim` samples `res` at to `times`: its own list, else
/// `npoints` even steps from `from` to `min(to, the last time)`, else every
/// time point of `res` inside `[from, to]`.
fn stimTimes(gpa: Allocator, stim: core.Stim, res: output.Result, times: *std.ArrayList(f64)) Allocator.Error!void {
    const width = res.varnames.len;
    try times.ensureTotalCapacity(gpa, if (stim.times.len != 0) stim.times.len else if (stim.npoints > 1) stim.npoints else res.npoints);
    if (stim.times.len != 0) {
        times.appendSliceAssumeCapacity(stim.times);
    } else if (stim.npoints > 1) {
        const to = @min(stim.to, res.data[(res.npoints - 1) * width]);
        const steps: f64 = @floatFromInt(stim.npoints - 1);
        for (0..stim.npoints) |k| times.appendAssumeCapacity(stim.from + (to - stim.from) * @as(f64, @floatFromInt(k)) / steps);
    } else for (0..res.npoints) |k| {
        const t = res.data[k * width];
        if (t >= stim.from and t <= stim.to) times.appendAssumeCapacity(t);
    }
    assert(times.items.len <= @max(stim.times.len, stim.npoints, res.npoints));
}
