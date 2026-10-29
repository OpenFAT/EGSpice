//! HSPICE external `.data`. In: a `.data name MER|LAM FILE=f p=col ...`
//! block and the files it names. Out: the inline `.data` table it stands
//! for, appended to the expanded deck.
//! Cites: HSPICE [CR] .DATA (external file forms).
//!
//! Contents:
//! - `pub const Error = Io.Dir.ReadFileAllocError || error{ InvalidData, InvalidInclude }`
//! - `pub fn appendData(io: Io, gpa: Allocator, search: Search, path: []const u8, name: []const u8, lam: bool, header: Fields, it: *std.mem.SplitIterator(u8, .scalar), out: *std.ArrayList(u8)) Error!void`

const std = @import("std");
const assert = std.debug.assert;
const mem = std.mem;
const Io = std.Io;
const Allocator = std.mem.Allocator;
const lines = @import("lines.zig");
const source_search = @import("source_search.zig");
const Fields = source_search.Fields;
const Search = source_search.Search;

/// Every way `appendData` fails; `out` may hold part of the table.
pub const Error = Io.Dir.ReadFileAllocError || error{
    /// A malformed block: a key with no `=`, no FILE= before a column, a
    /// column of 0, no column at all, OUT=, or a cell that is no number.
    InvalidData,
    /// A FILE= value that is no name.
    InvalidInclude,
};

/// One data file: its rows of cells, and the column (1-based, 0 for none)
/// each name reads.
const File = struct { rows: []const []const []const u8, cols: std.ArrayList(u32) };

/// Writes the inline table for one external `.data` block. Each `FILE=` is
/// followed by `name=column` pairs. MER stacks the files' rows, and a file
/// inherits the previous file's columns for names it does not give; LAM
/// puts the files side by side, row for row, each name from the file that
/// gives it. A missing value is 0. Data files hold blank- or
/// comma-separated numbers, one row per line; a line starting with `*` or
/// `$` is a comment. FILE= resolves as `.include` does. `header` holds the
/// rest of the `.data` line; `it` is left after `.enddata`.
pub fn appendData(
    io: Io,
    gpa: Allocator,
    search: Search,
    path: []const u8,
    name: []const u8,
    lam: bool,
    header: Fields,
    it: *mem.SplitIterator(u8, .scalar),
    out: *std.ArrayList(u8),
) Error!void {
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    var names: std.ArrayList([]const u8) = .empty;
    var files: std.ArrayList(File) = .empty;
    var tokens = header;
    while (true) {
        while (tokens.next()) |t| {
            if (t[0] == '$' or t[0] == ';') break;
            const key = mem.trimStart(u8, t, "+");
            if (key.len == 0) continue;
            if (!Fields.isWord(key) or !tokens.takeEq()) return error.InvalidData;
            const value = (try source_search.word(&tokens)) orelse return error.InvalidData;
            if (std.ascii.eqlIgnoreCase(key, "file")) {
                const resolved = try source_search.locate(io, gpa, search, path, value);
                defer gpa.free(resolved);
                const rows = try readRows(io, a, name, resolved);
                var cols: std.ArrayList(u32) = .empty;
                if (!lam and files.items.len != 0) try cols.appendSlice(a, files.items[files.items.len - 1].cols.items);
                try files.append(a, .{ .rows = rows, .cols = cols });
            } else if (std.ascii.eqlIgnoreCase(key, "out")) {
                // ponytail: OUT= (write the merged table back) is refused; add when a flow needs the file.
                if (!@import("builtin").is_test) std.log.err(".data {s}: OUT= is not supported", .{name});
                return error.InvalidData;
            } else {
                if (files.items.len == 0) return error.InvalidData;
                const col = std.fmt.parseInt(u32, value, 10) catch return error.InvalidData;
                if (col == 0) return error.InvalidData;
                const k = for (names.items, 0..) |n, i| {
                    if (std.ascii.eqlIgnoreCase(n, key)) break i;
                } else blk: {
                    try names.append(a, key);
                    break :blk names.items.len - 1;
                };
                const cols = &files.items[files.items.len - 1].cols;
                if (cols.items.len <= k) try cols.appendNTimes(a, 0, k + 1 - cols.items.len);
                cols.items[k] = col;
            }
        }
        const raw = it.next() orelse break;
        const line = mem.trim(u8, raw, " \t\r");
        if (line.len != 0 and std.ascii.eqlIgnoreCase(source_search.firstWord(line), ".enddata")) break;
        tokens = Fields.init(if (line.len != 0 and line[0] == '*') "" else line);
    }
    if (names.items.len == 0) return error.InvalidData;
    try writeTable(gpa, a, name, names.items, files.items, lam, out);
}

// ---- private ----

/// A data file's rows of number cells, comment lines and empty rows dropped.
fn readRows(io: Io, a: Allocator, name: []const u8, resolved: []const u8) Error![]const []const []const u8 {
    const text = try Io.Dir.cwd().readFileAlloc(io, resolved, a, .unlimited);
    var rows: std.ArrayList([]const []const u8) = .empty;
    var it = mem.splitScalar(u8, text, '\n');
    var line_no: u32 = 0;
    while (it.next()) |row| {
        line_no += 1;
        // `*` and `$` lines are comments, as in a netlist.
        const body = mem.trimStart(u8, row, " \t");
        if (body.len > 0 and (body[0] == '*' or body[0] == '$')) continue;
        var cells: std.ArrayList([]const u8) = .empty;
        var c = mem.tokenizeAny(u8, row, " \t,\r");
        while (c.next()) |v| {
            if (lines.hspice.parseNum(v) == null) {
                if (!@import("builtin").is_test) std.log.err(".data {s}: {s}:{d}: '{s}' is not a number", .{ name, resolved, line_no, v });
                return error.InvalidData;
            }
            try cells.append(a, v);
        }
        if (cells.items.len != 0) try rows.append(a, cells.items);
    }
    return rows.items;
}

/// The inline `.data name n1 n2 ... + row ... .enddata` table.
fn writeTable(gpa: Allocator, scratch: Allocator, name: []const u8, names: []const []const u8, files: []const File, lam: bool, out: *std.ArrayList(u8)) Allocator.Error!void {
    assert(names.len > 0);
    try out.print(gpa, ".data {s}", .{name});
    for (names) |n| try out.print(gpa, " {s}", .{n});
    try out.append(gpa, '\n');
    if (lam) {
        // Each name's file: the last that gives it a column.
        const owner = try scratch.alloc(?usize, names.len);
        @memset(owner, null);
        var rows: usize = 0;
        for (files, 0..) |f, fi| {
            rows = @max(rows, f.rows.len);
            for (f.cols.items, 0..) |col, k| if (col != 0) {
                owner[k] = fi;
            };
        }
        for (0..rows) |r| {
            try out.append(gpa, '+');
            for (owner, 0..) |fi, k| try out.print(gpa, " {s}", .{if (fi) |i| cell(files[i], r, k) else "0"});
            try out.append(gpa, '\n');
        }
    } else for (files) |f| for (0..f.rows.len) |r| {
        try out.append(gpa, '+');
        for (0..names.len) |k| try out.print(gpa, " {s}", .{cell(f, r, k)});
        try out.append(gpa, '\n');
    };
    try out.appendSlice(gpa, ".enddata\n");
}

/// Row `r`, name `k` of a data file; "0" where it has none.
fn cell(f: File, r: usize, k: usize) []const u8 {
    if (r >= f.rows.len or k >= f.cols.items.len or f.cols.items[k] == 0) return "0";
    const row = f.rows[r];
    const col = f.cols.items[k];
    assert(col >= 1);
    return if (col <= row.len) row[col - 1] else "0";
}
