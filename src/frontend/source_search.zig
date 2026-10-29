//! Where a file a deck names is found and how its name is read. In: the
//! naming file's path, a name as written, the `.option search=` directories.
//! Out: a resolved path, and the file's bytes read once per expansion.
//! Cites: HSPICE [CR] .OPTION SEARCH; ngspice inpcom.c (paths relative to
//! the including file).
//!
//! Contents:
//! - `pub const Fields = lines.Fields("\"'", false)`
//! - `pub const Search = struct`
//! - `pub const Files = struct`
//! - `pub fn get(self: *Files, io: Io, gpa: Allocator, path: []const u8) Io.Dir.ReadFileAllocError![]const u8`
//! - `pub fn deinit(self: *Files, gpa: Allocator) void`
//! - `pub fn locate(io: Io, gpa: Allocator, search: Search, path: []const u8, name: []const u8) Allocator.Error![]u8`
//! - `pub fn word(fields: *Fields) error{InvalidInclude}!?[]const u8`
//! - `pub fn firstWord(t: []const u8) []const u8`

const std = @import("std");
const assert = std.debug.assert;
const Io = std.Io;
const Allocator = std.mem.Allocator;
const lines = @import("lines.zig");

/// A card's fields with `"` and `'` quoting, no braces.
pub const Fields = lines.Fields("\"'", false);

/// Where a file a deck names is looked for: beside the file naming it, then
/// in `dirs`; `deck` is the top-level deck, whose stem names `.load`'s
/// default file. Borrows every slice.
pub const Search = struct {
    dirs: []const []const u8,
    deck: []const u8,
    /// The files read so far, so a library named by several `.lib` cards
    /// (one per section) is read once; null reads each time.
    files: ?*Files = null,
};

/// File bytes by resolved path, owned by the expansion's scratch allocator.
pub const Files = struct {
    map: std.StringHashMapUnmanaged([]u8) = .empty,

    /// `path`'s bytes, read on first use. They live until `deinit`.
    pub fn get(self: *Files, io: Io, gpa: Allocator, path: []const u8) Io.Dir.ReadFileAllocError![]const u8 {
        const got = try self.map.getOrPut(gpa, path);
        if (got.found_existing) return got.value_ptr.*;
        errdefer self.map.removeByPtr(got.key_ptr);
        const key = try gpa.dupe(u8, path);
        errdefer gpa.free(key);
        const bytes = try Io.Dir.cwd().readFileAlloc(io, path, gpa, .unlimited);
        got.key_ptr.* = key;
        got.value_ptr.* = bytes;
        assert(self.map.contains(path));
        return bytes;
    }

    /// Frees every path and every file's bytes, with the `gpa` that read them.
    pub fn deinit(self: *Files, gpa: Allocator) void {
        var it = self.map.iterator();
        while (it.next()) |e| {
            gpa.free(e.key_ptr.*);
            gpa.free(e.value_ptr.*);
        }
        self.map.deinit(gpa);
        self.* = undefined;
    }
};

/// `name` beside `path`, else in the first search directory holding it;
/// beside `path` when none does, so the open reports the usual error.
/// Caller owns returned string and must free with `gpa`.
pub fn locate(io: Io, gpa: Allocator, search: Search, path: []const u8, name: []const u8) Allocator.Error![]u8 {
    const here = try std.fs.path.resolve(gpa, &.{ std.fs.path.dirname(path) orelse ".", name });
    errdefer gpa.free(here);
    if (search.dirs.len == 0 or std.fs.path.isAbsolute(name)) return here;
    if (Io.Dir.cwd().access(io, here, .{})) |_| return here else |_| {}
    for (search.dirs) |dir| {
        const there = try std.fs.path.resolve(gpa, &.{ dir, name });
        if (Io.Dir.cwd().access(io, there, .{})) |_| {
            gpa.free(here);
            return there;
        } else |_| gpa.free(there);
    }
    return here;
}

/// The next field as a name: a quoted body or a bare word; null at the end
/// of the card or at a `$`/`;` comment. InvalidInclude for any other field.
pub fn word(fields: *Fields) error{InvalidInclude}!?[]const u8 {
    const rest = fields.rest();
    if (rest.len == 0 or rest[0] == '$' or rest[0] == ';') return null;
    const f = fields.next() orelse return null;
    assert(f.len > 0);
    if (Fields.isQuote(f[0])) return Fields.body(f);
    return if (Fields.isWord(f)) f else error.InvalidInclude;
}

/// The text before the first blank or tab.
pub fn firstWord(t: []const u8) []const u8 {
    return t[0 .. std.mem.indexOfAny(u8, t, " \t") orelse t.len];
}

test word {
    var f = Fields.init("'a b' c $ d");
    try std.testing.expectEqualStrings("a b", (try word(&f)).?);
    try std.testing.expectEqualStrings("c", (try word(&f)).?);
    try std.testing.expectEqual(null, try word(&f));
    try std.testing.expectEqualStrings(".lib", firstWord(".lib x"));
}
