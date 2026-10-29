//! A `Stream`'s passes handed to the GPU lanes and taken back: the open
//! cards' host passes in, `scan.Dev` passes and the unknowns or programs
//! each reads (`scan.DevColumns`) out, and after the device ran them, the
//! cards they finished resolved on the host.
//! Cites: performance TODO 06 6.8 (`.meas` on the device).
//!
//! Contents:
//! - `pub fn handOut(s: *const Stream, gpa: Allocator) Allocator.Error!?Stream.DeviceScans`
//! - `pub fn takeBack(s: *Stream, scans: []const scan.Dev) void`

const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;
const scan = @import("scan.zig");
const columns = @import("column.zig");
const Stream = @import("Stream.zig");
const Scan = @import("Scan.zig");

/// A column as the device reads it: the unknown behind it, or `none` and
/// a program for the scale or a `par()` waveform.
const Source = struct { unknown: u32, program: scan.Program };

const nan = std.math.nan(f64);

/// Returns `Stream.deviceScans`: every open card's passes as device
/// passes, or null when one cannot run on the device. Caller owns both
/// slices of the result.
pub fn handOut(s: *const Stream, gpa: Allocator) Allocator.Error!?Stream.DeviceScans {
    var n: usize = 0;
    for (s.cards) |*c| {
        if (c.done) continue;
        if (c.rule != .scan or !c.started or c.stage == .targ) return null;
        if (c.stage == .find) if (c.ring_y == null or source(s, c.ring_y.?) == null) return null;
        for (c.scans[0..c.n]) |*sc| {
            if (sc.missing or source(s, sc.y) == null) return null;
            if (sc.y2) |y2| if (source(s, y2) == null) return null;
            switch (sc.pass) {
                .cross, .at, .extremum, .integ => {},
                .em_avg, .rel_error => return null,
            }
        }
        n += c.n + @intFromBool(c.stage == .find);
    }
    const scans = try gpa.alloc(scan.Dev, n);
    errdefer gpa.free(scans);
    const cols = try gpa.alloc(scan.DevColumns, n);
    var k: usize = 0;
    for (s.cards) |*c| if (!c.done) for (c.scans[0..c.n], 0..) |*sc, j| {
        scans[k] = switch (sc.pass) {
            inline .cross, .at, .extremum, .integ => |v, tag| .{
                .kind = @field(scan.Dev.Kind, @tagName(tag)),
                .over = sc.over,
                .time = sc.time[0],
                .pass = @unionInit(@FieldType(scan.Dev, "pass"), @tagName(tag), v),
            },
            .em_avg, .rel_error => unreachable, // refused above
        };
        cols[k] = devColumns(s, sc.y, sc.y2);
        k += 1;
        if (c.stage == .find and j + 1 == c.n) {
            scans[k] = .{ .kind = .find, .pass = .{ .find = .{
                .at = .{ .at = nan, .slope = c.slope },
                .ring = c.ring,
                .ring_len = c.ring_len,
                .rows = s.rows,
            } } };
            cols[k] = devColumns(s, c.ring_y.?, null);
            k += 1;
        }
    };
    assert(k == n);
    return .{ .scans = scans, .cols = cols };
}

/// `Stream.adoptScans` without the live-list upkeep: copies each pass back
/// into its card and closes every card that resolved. Asserts that `scans`
/// is what `handOut` gave for the cards still open.
pub fn takeBack(s: *Stream, scans: []const scan.Dev) void {
    var k: usize = 0;
    for (s.cards, 0..) |*c, i| if (!c.done) {
        for (c.scans[0..c.n]) |*sc| {
            const d = scans[k];
            k += 1;
            sc.over = d.over;
            sc.time[0] = d.time;
            sc.pass = switch (d.kind) {
                inline .cross, .at, .extremum, .integ => |tag| @unionInit(Scan.Pass, @tagName(tag), @field(d.pass, @tagName(tag))),
                .find => unreachable, // a card's `.find` pass comes after its own
            };
        }
        if (c.stage == .find) {
            const d = scans[k];
            k += 1;
            const f = d.pass.find;
            c.ring = f.ring;
            c.ring_len = f.ring_len;
            switch (f.state) {
                .waiting => {},
                .none => c.stage = .none,
                .armed => {
                    c.stage = .none;
                    c.n = 2;
                    c.scans[1] = .{ .y = c.ring_y.?, .over = d.over, .pass = .{ .at = f.at } };
                },
            }
        }
        if (c.resolved()) s.close(i);
    };
    assert(k == scans.len);
}

// ---- private ----

/// `col` as the device reads it; null for a complex column, a power, or a
/// program past `scan.Program.max` ops.
fn source(s: *const Stream, col: columns.Waveform) ?Source {
    var p: scan.Program = .{};
    switch (col) {
        .column => |c| {
            if (c.complex) return null;
            if (c.offset != 0) return .{ .unknown = s.nodes[c.offset - 1], .program = p };
            p.ops[0] = .{ .code = .scale };
            p.len = 1;
            return .{ .unknown = scan.DevColumns.none, .program = p };
        },
        .par => |par| {
            if (par.base.complex) return null;
            if (par.ops.len > scan.Program.max) return null;
            for (par.ops, p.ops[0..par.ops.len]) |op, *o| o.* = switch (op) {
                .num => |x| .{ .code = .num, .num = x },
                .measure => .{ .code = .nan },
                .vector => |name| blk: {
                    // `Sample.leaf`'s column: every name resolved when the
                    // waveform was made.
                    const k = columns.columnIndex(par.names, name) orelse return null;
                    break :blk if (k == 0) .{ .code = .scale } else .{ .code = .unknown, .unknown = s.nodes[k - 1] };
                },
                .neg => .{ .code = .neg },
                .add => .{ .code = .add },
                .sub => .{ .code = .sub },
                .mul => .{ .code = .mul },
                .div => .{ .code = .div },
                .pow => return null, // `scan.Program` has no power
            };
            p.len = @intCast(par.ops.len);
            return .{ .unknown = scan.DevColumns.none, .program = p };
        },
    }
}

/// `source` of a pass's two columns, which `handOut` has checked.
fn devColumns(s: *const Stream, y: columns.Waveform, y2: ?columns.Waveform) scan.DevColumns {
    const a = source(s, y).?;
    const b: Source = if (y2) |c| source(s, c).? else .{ .unknown = scan.DevColumns.none, .program = .{} };
    return .{ .y = a.unknown, .y2 = b.unknown, .prog = a.program, .prog2 = b.program };
}
