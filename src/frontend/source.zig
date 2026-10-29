//! Include expansion, before parsing. In: a deck's text and the path it
//! came from. Out: one text with every `.include`, selected `.lib` section,
//! HSPICE `.load`, `.vec` and external `.data` inlined (a Spectre deck's
//! `include` too); an HDL include stays an `.include` of its resolved path.
//! Paths resolve against the directory of the file that names them, then
//! against each HSPICE `.option search=` directory (source_search.zig).
//! Cites: ngspice inpcom.c (.include, .lib sections), HSPICE [CR] .LIB,
//! .LOAD, .VEC, .DATA, .ALTER, .OPTION SEARCH;
//! dev/research/spectre-syntax.md §1.4, §3.3, §3.4.
//!
//! Contents:
//! - `pub const Error = Io.Dir.ReadFileAllocError || error{...}`
//! - `pub const ForeignKind = enum`
//! - `pub const splitAlters = source_alter.splitAlters`
//! - `pub fn load(io: Io, arena: Allocator, path: []const u8) Error![]const u8`
//! - `pub fn expand(io: Io, gpa: Allocator, arena: Allocator, origin: []const u8, src: []const u8) Error![]const u8`
//! - `pub fn expandSpectre(io: Io, arena: Allocator, origin: []const u8, src: []const u8) Error![]const u8`
//! - `pub fn foreignKindForPath(path: []const u8) ?ForeignKind`

const std = @import("std");
const assert = std.debug.assert;
const Io = std.Io;
const Allocator = std.mem.Allocator;
const vec = @import("vec.zig");
const switchesTo = @import("spectre.zig").switchesTo;
const source_search = @import("source_search.zig");
const source_alter = @import("source_alter.zig");
const source_data = @import("source_data.zig");
const Fields = source_search.Fields;
const Search = source_search.Search;
const Files = source_search.Files;

/// Every way an expansion fails; nothing is returned, the scratch is freed.
pub const Error = Io.Dir.ReadFileAllocError || error{
    /// A malformed `.include`, `.lib`, `.load` or Spectre `include` card.
    InvalidInclude,
    /// An unbalanced `.lib`/`.endl` or `section`/`endsection`.
    InvalidLibrarySection,
    /// The selected section appears twice in one library.
    DuplicateLibrarySection,
    /// No section of the library has the selected name.
    LibrarySectionNotFound,
    /// A malformed external `.data` block (source_data.zig).
    InvalidData,
    /// A `.vec` file vec.zig cannot read.
    InvalidVector,
    /// Includes nested past `include_depth_max` levels.
    IncludeDepthExceeded,
};

/// What a non-netlist file a deck names holds: a compiled OSDI model
/// (`osdi_include`, `pre_osdi`), HDL source (`verilog_a`, `verilog`), or
/// `data` that a card reads (an HSPICE RLGC or Touchstone file).
pub const ForeignKind = enum { osdi_include, pre_osdi, verilog_a, verilog, data };

/// HSPICE `.alter` runs (source_alter.zig).
pub const splitAlters = source_alter.splitAlters;

const Directive = enum { include, lib, endl, load, data, vec };
const directives = std.StaticStringMap(Directive).initComptime(.{
    .{ ".include", .include }, .{ ".inc", .include }, .{ ".lib", .lib }, .{ ".endl", .endl },
    .{ ".load", .load },       .{ ".data", .data },   .{ ".vec", .vec },
});

/// Includes past this depth are refused; the recursion stays bounded.
const include_depth_max = 32;

const SpectreInclude = struct { file: []const u8, section: ?[]const u8 };

/// Reads `path` and expands it (see `expand`). The result is allocated in
/// `arena`. Fails per `Error`.
pub fn load(io: Io, arena: Allocator, path: []const u8) Error![]const u8 {
    const src = try Io.Dir.cwd().readFileAlloc(io, path, arena, .unlimited);
    errdefer arena.free(src);
    const out = try expand(io, std.heap.page_allocator, arena, path, src);
    if (out.ptr != src.ptr) arena.free(src);
    return out;
}

/// Inlines includes and selected .lib sections, resolving paths against
/// `origin`'s directory; an HDL include stays an `.include` of its resolved
/// path. Returns `src` itself when it names none, else a copy in `arena`;
/// `gpa` holds the scratch (file buffers, the text being built), all freed
/// before return. Fails per `Error`.
pub fn expand(io: Io, gpa: Allocator, arena: Allocator, origin: []const u8, src: []const u8) Error![]const u8 {
    var lines = std.mem.splitScalar(u8, src, '\n');
    _ = lines.next(); // The title is opaque even when it starts with .include.
    while (lines.next()) |line| {
        if (directiveOf(std.mem.trim(u8, line, " \t\r")) != null) break;
    } else return src;

    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(gpa);
    var files: Files = .{};
    defer files.deinit(gpa);
    try appendContents(io, gpa, .{ .dirs = try searchDirs(arena, origin, src), .deck = origin, .files = &files }, origin, src, null, 0, &out);
    return try arena.dupe(u8, out.items);
}

/// `expand` for a Spectre deck: inlines `include "file" [section=name]`
/// (dev/research/spectre-syntax.md §3.3, §3.4) and expands a `simulator
/// lang=spice` region's SPICE includes as `expand` does. An included file
/// reads in Spectre mode when it ends in `.scs` or opens with `simulator
/// lang=spectre`, in SPICE mode otherwise (its `section=` then selects a
/// `.lib` section), and the includer's mode resumes after it. Returns `src`
/// itself when it names nothing, else a copy in `arena`. Fails per `Error`.
pub fn expandSpectre(io: Io, arena: Allocator, origin: []const u8, src: []const u8) Error![]const u8 {
    const title_end = std.mem.indexOfScalar(u8, src, '\n') orelse return src;
    var lines = std.mem.splitScalar(u8, src[title_end + 1 ..], '\n');
    while (lines.next()) |line| {
        const t = std.mem.trim(u8, line, " \t\r");
        // Any line that may name a file; `appendSpectre` reads it in its mode
        // (`include a 0 1m` in a SPICE region is a current source).
        if (directiveOf(t) != null or std.mem.startsWith(u8, t, "include")) break;
    } else return src;

    // ponytail: page_allocator scratch, as `load` gives `expand`; take it as a parameter when a test needs to fail it.
    const gpa = std.heap.page_allocator;
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(gpa);
    try out.appendSlice(gpa, src[0 .. title_end + 1]);
    _ = try appendSpectre(io, gpa, .{ .dirs = &.{}, .deck = origin }, origin, src[title_end + 1 ..], null, 0, &out);
    return try arena.dupe(u8, out.items);
}

/// The HDL kind a path's extension names; null for a netlist include.
pub fn foreignKindForPath(path: []const u8) ?ForeignKind {
    const ext = std.fs.path.extension(path);
    for ([_][]const u8{ ".va", ".vams", ".veriloga" }) |e| if (std.ascii.eqlIgnoreCase(ext, e)) return .verilog_a;
    if (std.ascii.eqlIgnoreCase(ext, ".v") or std.ascii.eqlIgnoreCase(ext, ".sv")) return .verilog;
    return null;
}

// ---- private ----

/// `include "file" [section = name] [// comment]`; null for any other line.
fn spectreInclude(line: []const u8) error{InvalidInclude}!?SpectreInclude {
    const blanks = " \t";
    if (!std.mem.startsWith(u8, line, "include")) return null;
    var rest = line["include".len..];
    if (rest.len == 0 or std.mem.indexOfScalar(u8, blanks, rest[0]) == null) return null;
    rest = std.mem.trimStart(u8, rest, blanks);
    if (rest.len == 0 or rest[0] != '"') return error.InvalidInclude;
    const close = std.mem.indexOfScalarPos(u8, rest, 1, '"') orelse return error.InvalidInclude;
    const file = rest[1..close];
    rest = std.mem.trim(u8, rest[close + 1 ..], blanks);
    if (rest.len == 0 or std.mem.startsWith(u8, rest, "//")) return .{ .file = file, .section = null };
    if (!std.mem.startsWith(u8, rest, "section")) return error.InvalidInclude;
    rest = std.mem.trimStart(u8, rest["section".len..], blanks);
    if (rest.len == 0 or rest[0] != '=') return error.InvalidInclude;
    rest = std.mem.trimStart(u8, rest[1..], blanks);
    const end = std.mem.indexOfAny(u8, rest, " \t/") orelse rest.len;
    const tail = std.mem.trim(u8, rest[end..], blanks);
    if (end == 0 or (tail.len > 0 and !std.mem.startsWith(u8, tail, "//"))) return error.InvalidInclude;
    return .{ .file = file, .section = rest[0..end] };
}

/// One file's lines in Spectre mode: `include` inlined, `section` selected
/// (case kept, as Spectre compares names), a SPICE region handed to
/// `appendContents` whole. Returns whether the file ends in SPICE mode.
// ponytail: `ahdl_include` paths stay as written, so one in an included
// file elsewhere resolves against the deck; rewrite them here when a PDK
// needs it.
fn appendSpectre(io: Io, gpa: Allocator, search: Search, path: []const u8, src: []const u8, section: ?[]const u8, depth: u8, out: *std.ArrayList(u8)) Error!bool {
    var region: std.ArrayList(u8) = .empty;
    defer region.deinit(gpa);
    var spice = false;
    var selected = section == null;
    var found = selected;
    var open = false;
    var lines = std.mem.splitScalar(u8, chomp(src), '\n');
    while (lines.next()) |line| {
        const t = std.mem.trim(u8, line, " \t\r");
        if (spice and switchesTo(t, "spectre")) {
            spice = false;
            if (!selected) continue;
            try appendContents(io, gpa, search, path, chomp(region.items), null, depth, out);
            region.clearRetainingCapacity();
        } else if (spice) {
            if (selected) {
                try region.appendSlice(gpa, line);
                try region.append(gpa, '\n');
            }
            continue;
        } else if (switchesTo(t, "spice")) {
            spice = true;
        } else if (section) |wanted| {
            const w = source_search.firstWord(t);
            if (std.mem.eql(u8, w, "library") or std.mem.eql(u8, w, "endlibrary")) continue;
            if (std.mem.eql(u8, w, "section")) {
                if (open) return error.InvalidLibrarySection;
                open = true;
                selected = std.mem.eql(u8, source_search.firstWord(std.mem.trimStart(u8, t[w.len..], " \t")), wanted);
                if (selected and found) return error.DuplicateLibrarySection;
                found = found or selected;
                continue;
            }
            if (std.mem.eql(u8, w, "endsection")) {
                if (!open) return error.InvalidLibrarySection;
                open = false;
                selected = false;
                continue;
            }
        }
        if (!selected) continue;
        // An `include` continued with `\` (§1.4) reads as its joined lines;
        // they pass through as written when they are not one.
        var stmt = t;
        var joined: std.ArrayList(u8) = .empty;
        defer joined.deinit(gpa);
        var raw: std.ArrayList(u8) = .empty;
        defer raw.deinit(gpa);
        if (std.mem.startsWith(u8, t, "include") and continues(t)) {
            try raw.appendSlice(gpa, line);
            try joined.appendSlice(gpa, t);
            while (continues(joined.items)) {
                const more = lines.next() orelse break;
                try raw.append(gpa, '\n');
                try raw.appendSlice(gpa, more);
                joined.items[joined.items.len - 1] = ' ';
                try joined.appendSlice(gpa, std.mem.trim(u8, more, " \t\r"));
            }
            stmt = std.mem.trim(u8, joined.items, " \t");
        }
        if (try spectreInclude(stmt)) |inc| {
            if (depth == include_depth_max - 1) return error.IncludeDepthExceeded;
            const resolved = try source_search.locate(io, gpa, search, path, inc.file);
            defer gpa.free(resolved);
            const text = try Io.Dir.cwd().readFileAlloc(io, resolved, gpa, .unlimited);
            defer gpa.free(text);
            const first = std.mem.trim(u8, firstLine(text), " \t\r");
            if (std.ascii.eqlIgnoreCase(std.fs.path.extension(resolved), ".scs") or switchesTo(first, "spectre")) {
                if (try appendSpectre(io, gpa, search, resolved, text, inc.section, depth + 1, out)) try out.appendSlice(gpa, "simulator lang=spectre\n");
            } else {
                try out.appendSlice(gpa, "simulator lang=spice\n");
                try appendContents(io, gpa, search, resolved, chomp(text), inc.section, depth + 1, out);
                try out.appendSlice(gpa, "simulator lang=spectre\n");
            }
            continue;
        }
        try out.appendSlice(gpa, if (raw.items.len > 0) raw.items else line);
        try out.append(gpa, '\n');
    }
    if (spice and selected) try appendContents(io, gpa, search, path, chomp(region.items), null, depth, out);
    if (open) return error.InvalidLibrarySection;
    if (!found) return error.LibrarySectionNotFound;
    return spice;
}

/// Whether a trimmed Spectre line continues on the next (§1.4): it ends
/// in an odd run of backslashes and holds no `//` comment outside quotes.
fn continues(t: []const u8) bool {
    var quoted = false;
    for (t, 0..) |c, i| {
        if (c == '"') quoted = !quoted;
        if (!quoted and std.mem.startsWith(u8, t[i..], "//")) return false;
    }
    var run: usize = 0;
    while (run < t.len and t[t.len - 1 - run] == '\\') run += 1;
    return run % 2 == 1;
}

/// `text` without its final newline, so splitting it yields no empty last
/// line.
fn chomp(text: []const u8) []const u8 {
    return if (std.mem.endsWith(u8, text, "\n")) text[0 .. text.len - 1] else text;
}

/// The first non-blank line of `text`.
fn firstLine(text: []const u8) []const u8 {
    var lines = std.mem.tokenizeAny(u8, text, "\r\n");
    while (lines.next()) |line| if (std.mem.trim(u8, line, " \t").len > 0) return line;
    return "";
}

/// HSPICE `.option search='dir'` [CR .OPTION SEARCH]: where `.lib`,
/// `.include` and `.load` look for a file not found beside the file naming
/// it. Read off the top-level deck only, so a search path set inside an
/// included file is not seen; a relative directory resolves against the
/// deck's.
fn searchDirs(arena: Allocator, origin: []const u8, src: []const u8) Allocator.Error![]const []const u8 {
    var dirs: std.ArrayList([]const u8) = .empty;
    var lines = std.mem.splitScalar(u8, src, '\n');
    _ = lines.next();
    while (lines.next()) |line| {
        var tokens = Fields.init(std.mem.trim(u8, line, " \t\r"));
        const head = tokens.next() orelse continue;
        if (!std.ascii.eqlIgnoreCase(head, ".option") and !std.ascii.eqlIgnoreCase(head, ".options") and !std.ascii.eqlIgnoreCase(head, ".opt")) continue;
        while (tokens.next()) |key| {
            if (!Fields.isWord(key) or !tokens.takeEq()) continue;
            const value = source_search.word(&tokens) catch continue orelse continue;
            if (std.ascii.eqlIgnoreCase(key, "search")) try dirs.append(arena, try std.fs.path.resolve(arena, &.{ std.fs.path.dirname(origin) orelse ".", value }));
        }
    }
    return dirs.items;
}

fn directiveOf(line: []const u8) ?Directive {
    if (line.len == 0 or line[0] != '.') return null;
    var key_buf: [8]u8 = undefined;
    const end = std.mem.indexOfAny(u8, line, " \t") orelse line.len;
    return if (end <= key_buf.len) directives.get(std.ascii.lowerString(key_buf[0..end], line[0..end])) else null;
}

fn appendFile(io: Io, gpa: Allocator, search: Search, path: []const u8, section: ?[]const u8, depth: u8, out: *std.ArrayList(u8)) Error!void {
    // ponytail: bounded recursion; iterative include stack if real PDKs exceed 32.
    if (depth == include_depth_max) return error.IncludeDepthExceeded;
    if (search.files) |files| return appendContents(io, gpa, search, path, try files.get(io, gpa, path), section, depth, out);
    const src = try Io.Dir.cwd().readFileAlloc(io, path, gpa, .unlimited);
    defer gpa.free(src);
    try appendContents(io, gpa, search, path, src, section, depth, out);
}

/// HSPICE `.load [FILE=f] [RUN=...]` [CR .LOAD]: inlines the `.nodeset`/`.ic`
/// cards an earlier `.save` wrote: `<top deck's stem>.ic0` by default, and
/// `x.ic0` for `FILE=x.ic` (the `.ic#` family). A file not there yet (the
/// run that first saves it) is skipped with a warning. RUN= only picks
/// between `.alter` runs' files, which ESPice does not number; it is
/// accepted and unused.
fn appendLoad(io: Io, gpa: Allocator, search: Search, path: []const u8, tokens: *Fields, depth: u8, out: *std.ArrayList(u8)) Error!void {
    var file: ?[]const u8 = null;
    while (tokens.next()) |key| {
        if (!Fields.isWord(key) or !tokens.takeEq()) return error.InvalidInclude;
        const value = (try source_search.word(tokens)) orelse return error.InvalidInclude;
        if (std.ascii.eqlIgnoreCase(key, "file")) file = value else if (!std.ascii.eqlIgnoreCase(key, "run")) return error.InvalidInclude;
    }
    // `FILE=x.ic` names the `.ic#` family; ESPice writes and reads `#` = 0.
    const name = if (file) |f|
        try std.mem.concat(gpa, u8, &.{ f, if (std.ascii.eqlIgnoreCase(std.fs.path.extension(f), ".ic")) "0" else "" })
    else
        try std.mem.concat(gpa, u8, &.{ std.fs.path.stem(search.deck), ".ic0" });
    defer gpa.free(name);
    const resolved = try source_search.locate(io, gpa, search, path, name);
    defer gpa.free(resolved);
    appendFile(io, gpa, search, resolved, null, depth + 1, out) catch |err| switch (err) {
        error.FileNotFound => std.log.warn(".load: {s} not found; no saved operating point", .{resolved}),
        else => return err,
    };
}

/// HSPICE `.vec 'file'`: the sources and `.dout` cards the vector file
/// stands for (frontend/vec.zig).
fn appendVec(io: Io, gpa: Allocator, path: []const u8, out: *std.ArrayList(u8)) Error!void {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const src = try Io.Dir.cwd().readFileAlloc(io, path, arena.allocator(), .unlimited);
    const text = vec.expand(arena.allocator(), src) catch |err| {
        if (err == error.InvalidVector) std.log.warn(".vec: cannot read {s}", .{path});
        return err;
    };
    try out.appendSlice(gpa, text);
}

fn appendContents(io: Io, gpa: Allocator, search: Search, path: []const u8, src: []const u8, section: ?[]const u8, depth: u8, out: *std.ArrayList(u8)) Error!void {
    var lines = std.mem.splitScalar(u8, src, '\n');
    var selected = section == null;
    var found = selected;
    var current_section: ?[]const u8 = null;
    while (lines.next()) |line| {
        const trimmed = std.mem.trim(u8, line, " \t\r");
        var tokens = Fields.init(trimmed);
        // Title and comments are opaque, including unmatched quotes.
        if (depth == 0 and out.items.len == 0) {
            try out.appendSlice(gpa, line);
            try out.append(gpa, '\n');
            continue;
        }
        if (directiveOf(trimmed)) |kind| {
            _ = try source_search.word(&tokens);
            if (kind == .data) {
                // An inline table passes through; MER and LAM read files.
                const name = tokens.next() orelse "";
                const mode = tokens.next() orelse "";
                const lam = std.ascii.eqlIgnoreCase(mode, "lam");
                if (selected and (lam or std.ascii.eqlIgnoreCase(mode, "mer"))) {
                    try source_data.appendData(io, gpa, search, path, name, lam, tokens, &lines, out);
                } else if (selected) {
                    try out.appendSlice(gpa, line);
                    try out.append(gpa, '\n');
                }
                continue;
            }
            if (kind == .load) {
                if (selected) try appendLoad(io, gpa, search, path, &tokens, depth, out);
                continue;
            }
            if (kind == .endl) {
                // The name after `.endl` is not checked: ngspice ignores it
                // (inpcom.c), and GF180 closes `.lib dio` with `.endl diode`.
                if (current_section == null) return error.InvalidLibrarySection;
                current_section = null;
                selected = section == null;
                continue;
            }
            const file_or_section = (try source_search.word(&tokens)) orelse return error.InvalidInclude;
            const corner = if (kind == .lib) try source_search.word(&tokens) else null;
            if (kind == .lib and corner == null) {
                if (current_section != null) return error.InvalidLibrarySection;
                current_section = file_or_section;
                selected = if (section) |wanted| std.ascii.eqlIgnoreCase(wanted, file_or_section) else false;
                if (selected and found) return error.DuplicateLibrarySection;
                found = found or selected;
                continue;
            }
            if (!selected) continue;
            const resolved = try source_search.locate(io, gpa, search, path, file_or_section);
            defer gpa.free(resolved);
            // HDL includes are left for the runtime loader.
            if (foreignKindForPath(file_or_section) != null) {
                try out.appendSlice(gpa, ".include \"");
                try out.appendSlice(gpa, resolved);
                try out.appendSlice(gpa, "\"\n");
                continue;
            }
            if (kind == .vec) {
                try appendVec(io, gpa, resolved, out);
                continue;
            }
            try appendFile(io, gpa, search, resolved, corner, depth + 1, out);
        } else if (selected) {
            try out.appendSlice(gpa, line);
            try out.append(gpa, '\n');
        }
    }
    if (current_section != null) return error.InvalidLibrarySection;
    if (!found) return error.LibrarySectionNotFound;
}

test "directives and foreign kinds" {
    try std.testing.expectEqual(.include, directiveOf(".INC x").?);
    try std.testing.expectEqual(.endl, directiveOf(".endl\tx").?);
    try std.testing.expectEqual(null, directiveOf(".includes x"));
    try std.testing.expectEqual(null, directiveOf("r1 a b 1"));
    try std.testing.expectEqual(null, directiveOf(""));
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    try std.testing.expectEqual(.verilog_a, foreignKindForPath("a/B.VA").?);
    try std.testing.expectEqual(.verilog, foreignKindForPath("x.sv").?);
    try std.testing.expectEqual(null, foreignKindForPath("x.sp"));
    // A deck naming nothing is returned as itself, even with a `.include` title.
    const plain = ".include x\nr1 a 0 1\n";
    try std.testing.expectEqual(plain.ptr, (try expand(std.testing.io, std.testing.allocator, a, "deck.sp", plain)).ptr);
}

test ".load: the top deck names the default file, FILE=x.ic reads x.ic0" {
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    try tmp.dir.writeFile(io, .{ .sub_path = "deck.ic0", .data = ".nodeset\n+ a= 1.0000\n" });
    try tmp.dir.writeFile(io, .{ .sub_path = "run.ic0", .data = ".ic v(b)=2\n" });
    try tmp.dir.writeFile(io, .{ .sub_path = "inc.sp", .data = ".load\n" });
    const path = try a.print(".zig-cache/tmp/{s}/deck.sp", .{tmp.sub_path});
    // Each inlined file ends in blank lines; only the cards matter here.
    try std.testing.expectEqualStrings("t\n.nodeset\n+ a= 1.0000", std.mem.trimEnd(u8, try expand(io, std.testing.allocator, a, path, "t\n.include 'inc.sp'\n"), "\n"));
    try std.testing.expectEqualStrings("t\n.ic v(b)=2", std.mem.trimEnd(u8, try expand(io, std.testing.allocator, a, path, "t\n.load file='run.ic'\n"), "\n"));
}

test "Spectre include: sections, SPICE files, and the includer's mode after" {
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    try tmp.dir.writeFile(io, .{ .sub_path = "lib.scs", .data = "library l\nsection tt\nr1 (a 0) resistor r=1\ninclude \"r.sp\"\nendsection tt\nsection ss\nr1 (a 0) resistor r=2\nendsection\nendlibrary l\n" });
    try tmp.dir.writeFile(io, .{ .sub_path = "r.sp", .data = "r2 a 0 3\n" });
    try tmp.dir.writeFile(io, .{ .sub_path = "c.inc", .data = "simulator lang=spectre\nc1 (a 0) capacitor c=1p\nsimulator lang=spice\nc2 a 0 2p\n" });
    const path = try a.print(".zig-cache/tmp/{s}/deck.scs", .{tmp.sub_path});
    const Case = struct { []const u8, anyerror![]const u8 };
    for ([_]Case{
        .{ "t\ninclude \"lib.scs\" section=tt // corner\n", "t\nr1 (a 0) resistor r=1\nsimulator lang=spice\nr2 a 0 3\nsimulator lang=spectre\n" },
        .{ "t\ninclude \"c.inc\"\nv1 (a 0) vsource dc=1\n", "t\nsimulator lang=spectre\nc1 (a 0) capacitor c=1p\nsimulator lang=spice\nc2 a 0 2p\nsimulator lang=spectre\nv1 (a 0) vsource dc=1\n" },
        .{ "t\nsimulator lang=spice\n.include 'r.sp'\nsimulator lang=spectre\n", "t\nsimulator lang=spice\nr2 a 0 3\n\nsimulator lang=spectre\n" },
        .{ "t\ninclude \"lib.scs\" section=TT\n", error.LibrarySectionNotFound },
        .{ "t\ninclude lib.scs\n", error.InvalidInclude },
        .{ "t\ninclude \"lib.scs\" corner=tt\n", error.InvalidInclude },
    }) |case| {
        const got = expandSpectre(io, a, path, case[0]);
        if (case[1]) |want| try std.testing.expectEqualStrings(want, try got) else |err| try std.testing.expectError(err, got);
    }
    const plain = "t\nr1 (a 0) resistor r=1\n";
    try std.testing.expectEqual(plain.ptr, (try expandSpectre(io, a, path, plain)).ptr);
}

test "external .data: MER stacks files, LAM sets them side by side" {
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    try tmp.dir.writeFile(io, .{ .sub_path = "a.txt", .data = "1 2\n3,4\n" });
    try tmp.dir.writeFile(io, .{ .sub_path = "b.txt", .data = "5 6\n" });
    try tmp.dir.writeFile(io, .{ .sub_path = "c.txt", .data = "* sweep points\n$ x\n-1.5k\n" });
    try tmp.dir.writeFile(io, .{ .sub_path = "h.txt", .data = "x\n1\n" });
    const path = try a.print(".zig-cache/tmp/{s}/deck.sp", .{tmp.sub_path});
    const Case = struct { []const u8, anyerror![]const u8 };
    for ([_]Case{
        .{ "t\n.data d MER\n+ FILE='a.txt' x=1 y=2\n+ FILE='b.txt' y=1\n.enddata\n.end\n", "t\n.data d x y\n+ 1 2\n+ 3 4\n+ 5 5\n.enddata\n.end\n\n" },
        .{ "t\n.data d LAM\n+ FILE='a.txt' x=1\n+ FILE='b.txt' y=2\n.enddata\n", "t\n.data d x y\n+ 1 6\n+ 3 0\n.enddata\n\n" },
        .{ "t\n.data d MER FILE='a.txt' x=0\n.enddata\n", error.InvalidData },
        .{ "t\n.data d MER x=1\n.enddata\n", error.InvalidData },
        .{ "t\n.data d MER FILE='a.txt'\n.enddata\n", error.InvalidData },
        .{ "t\n.data d MER FILE='c.txt' x=1\n.enddata\n", "t\n.data d x\n+ -1.5k\n.enddata\n\n" },
        .{ "t\n.data d MER FILE='h.txt' x=1\n.enddata\n", error.InvalidData },
    }) |case| {
        try tmp.dir.writeFile(io, .{ .sub_path = "deck.sp", .data = case[0] });
        const got = load(io, a, path);
        if (case[1]) |want| try std.testing.expectEqualStrings(want, try got) else |err| try std.testing.expectError(err, got);
    }
}

test {
    _ = source_search;
    _ = source_alter;
    _ = source_data;
    _ = @import("tests/source.zig");
}
