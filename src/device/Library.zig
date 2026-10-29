//! Every device type one Problem can instantiate, under one dense
//! `DeviceType` id: the catalog's built-ins and HDL sources in, one
//! (name, vtable, digital) row per type out. Built-ins hold ids
//! [0, builtin_count) in catalog order, so their ids are comptime-known
//! (`builtin`); each loaded HDL module appends one row. The compiling itself
//! is loader.zig's. Cites LRM 12.31-12.33 for the VPI host, and #13 for the
//! embedder's toolchain paths.
//!
//! Contents:
//! - `gpa`, `names`, `vtables`, `digital`, `sources`, `toolchain`, `vpi`,
//!   `vpi_loaded`, `systf_hosts` (fields)
//! - `pub const Vpi = struct`
//! - `pub const AnalogPoint = struct`
//! - `pub const Verdict = enum`
//! - `pub const VpiError = error{...}`
//! - `pub const Toolchain = struct`
//! - `pub const LoadError = error{...}`
//! - `pub const RegisterError = error{...}`
//! - `pub const builtin_count: u16`
//! - `pub const no_vpi_app: abi.SystfHost`
//! - `pub fn init(gpa: Allocator) Allocator.Error!Library`
//! - `pub fn deinit(self: *Library) void`
//! - `pub fn builtin(comptime name: []const u8) DeviceType`
//! - `pub fn vtable(self: *const Library, t: DeviceType) *const abi.DeviceVtable`
//! - `pub fn find(self: *const Library, module: []const u8) ?DeviceType`
//! - `pub fn register(self: *Library, module: []const u8, vt: *const abi.DeviceVtable, is_digital: bool) RegisterError!DeviceType`
//! - `pub fn loadVpiApp(self: *Library, path: []const u8) (error{VpiUnavailable} || VpiError)!void`
//! - `pub fn systfHost(self: *const Library, t: DeviceType) ?*const abi.SystfHost`
//! - `pub fn isSystfTask(call: abi.Systf) bool`
//! - `pub fn systfRegistered(self: *const Library, name: []const u8) bool`
//! - `pub fn load(self: *Library, io: std.Io, files: []const []const u8) LoadError!void`
//! - `pub fn loadWith(self: *Library, io: std.Io, files: []const []const u8, paths: loader.BuildPaths) LoadError!void`

const std = @import("std");
const builtin_info = @import("builtin");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;
const abi = @import("device_abi");
const build_options = @import("build_options");
const catalog = @import("catalog.zig");
const loader = @import("loader.zig");
const DeviceType = abi.DeviceType;

/// Owns every loaded name, `sources` and the tables.
gpa: Allocator,
/// Type names by id. Loaded names are owned and lowercased; built-in names
/// are static.
names: std.ArrayList([]const u8) = .empty,
/// Vtables by id, parallel to `names`.
vtables: std.ArrayList(*const abi.DeviceVtable) = .empty,
/// Whether each type runs a digital engine (a `.v` design), parallel to
/// `names`; it becomes `Batch.digital` at the freeze.
digital: std.ArrayList(bool) = .empty,
/// Every HDL source `load` handled, as resolved paths (owned), so one named
/// again, by another `.lib` section or a later `load`, compiles once.
sources: std.ArrayList([]const u8) = .empty,
/// Where `load` finds the Zig compiler and the evaluator sources. Borrowed;
/// it must outlive every `load`.
toolchain: Toolchain = .{},
/// The VPI host (LRM 12.32/12.33) behind `--vpi` and `.options vpi=`, which
/// the espice executable supplies; null in a library build. Borrowed.
vpi: ?*const Vpi = null,
/// Whether `loadVpiApp` loaded at least one application.
vpi_loaded: bool = false,
/// The application each loaded type with `systf_calls` calls through, once
/// `load` attached it (`Vpi.attach`). A handful at most: VerA's VPI holds
/// one design (VerA#84).
systf_hosts: std.ArrayList(SystfBinding) = .empty,

/// The VPI application host. The executable implements it over VerA's `vpi`
/// module, so the `vpi_*` symbols an application calls are exported from
/// the executable alone, never from a library that embeds ESPice. The
/// implementation serializes every call.
pub const Vpi = struct {
    ctx: *anyopaque,
    /// Opens the application at `path` and runs its `vlog_startup_routines`
    /// (LRM 12.33.2). Loading one path twice is a no-op.
    load_app: *const fn (ctx: *anyopaque, path: []const u8) VpiError!void,
    /// The analog functions registered as vpiIntFunc (LRM 12.32.1), which
    /// compile as integers; borrowed until the next `load_app`.
    int_systfs: *const fn (ctx: *anyopaque) []const []const u8,
    /// Whether an application registered an analog system task or function
    /// named `name` (with its `$`).
    registered: *const fn (ctx: *anyopaque, name: []const u8) bool,
    /// Builds VerA's design of the module at `source_path` and binds `calls`
    /// (a loaded device's `systf_calls`) to it; returns what its instances
    /// call through, valid for the life of the process.
    attach: *const fn (ctx: *anyopaque, source_path: []const u8, calls: []const abi.Systf) VpiError!*const abi.SystfHost,
    /// LRM 12.31.3, driven by `.op` and `.tran` once a module is attached:
    /// `point` publishes the solution about to be attempted, `verdict` asks
    /// whether it stands (acbConvergenceTest, 12.36), `accept` delivers
    /// acbInitialStep/acbAbsTime/acbElapsedTime/acbAcceptedPoint/acbFinalStep
    /// for the solution last published, and `next_forced` is the earliest
    /// acbAbsTime/acbElapsedTime strictly after `after`, a time the transient
    /// lands on.
    point: *const fn (ctx: *anyopaque, p: AnalogPoint) void,
    verdict: *const fn (ctx: *anyopaque) Verdict,
    accept: *const fn (ctx: *anyopaque, first: bool, last: bool) void,
    next_forced: *const fn (ctx: *anyopaque, after: f64) ?f64,
};

/// The solution an analysis is attempting: LRM 12.18's bounds and 12.9's time.
pub const AnalogPoint = struct {
    kind: enum { op, tran },
    start: f64 = 0,
    stop: f64 = 0,
    max_step: f64 = 0,
    t: f64 = 0,
    dt: f64 = 0,
};

/// `reject`: an acbConvergenceTest returned non-zero or
/// vpiRejectTransientStep ran, so back up; `again`:
/// vpiTransientFailConverge, so solve the same time again.
pub const Verdict = enum { accept, reject, again };

/// Every way a VPI step fails; each is logged with what it names.
pub const VpiError = error{
    /// dlopen refused the application.
    VpiApplicationUnloadable,
    /// The library exports no `vlog_startup_routines`.
    VpiNoStartupRoutines,
    /// A second module calls a `$systf`; VerA's VPI holds one design (VerA#84).
    VpiSecondSystfModule,
    /// VerA could not build the module's design.
    VpiDesignFailed,
    OutOfMemory,
};

/// Where `load` finds its compiler, sources and build cache, so a host that
/// ships ESPice in its own bundle can point at the files inside it (#13). A
/// null field keeps the default. The library reads no environment: the
/// facade fills `zig` and `cache_dir` from it when the embedder leaves them
/// null (`espice.zig` `toolchainFromEnv`).
pub const Toolchain = struct {
    /// The Zig compiler that builds a loaded device; null runs `zig` from
    /// PATH. It must be the version this espice was built with.
    zig: ?[]const u8 = null,
    /// The `share/espice` directory `zig build` installs, which holds
    /// `device/eval.zig`; null looks in `<exe>/../share/espice`, then falls
    /// back to the source tree this espice was built from.
    share: ?[]const u8 = null,
    /// The directory loaded devices' builds are cached in; null uses
    /// `/tmp/espice-cache/hdl`.
    cache_dir: ?[]const u8 = null,
    /// Searched for a source's `` `include``s after its own directory, in
    /// order (VerA's `-I`).
    include_dirs: []const []const u8 = &.{},
};

/// Every way `load` fails. Files before the failing one stay registered;
/// none after it are.
pub const LoadError = error{
    /// The target is wasm, which has no compiler to run and no dlopen.
    RuntimeHdlUnsupported,
    /// `toolchain.share` holds no `device/eval.zig`.
    HdlShareNotFound,
} || RegisterError || std.Io.Cancelable || loader.Error || VpiError;

/// Every way `register` fails. On any error nothing is added.
pub const RegisterError = error{
    /// The id space below `DeviceType.unset` is full.
    TooManyDeviceTypes,
} || Allocator.Error;

/// Built-in types, which hold ids [0, builtin_count); loaded ones follow.
pub const builtin_count: u16 = builtins.len;

/// What an instance whose every unregistered `$systf` call is a task calls
/// through: nothing reads a task's value, so 0 with zero partials, and its
/// arguments keep their values (`out` null). Never bound for a used value.
pub const no_vpi_app: abi.SystfHost = .{ .ctx = undefined, .call = noVpiApp };

/// Catalog entries with an evaluator, in catalog order.
const builtins = blk: {
    var list: []const []const u8 = &.{};
    for (catalog.catalog) |e| {
        if (@hasDecl(e.type, "eval")) list = list ++ .{e.name};
    }
    break :blk list;
};

comptime {
    // A built-in never takes the sentinel id.
    assert(builtin_count < @backingInt(DeviceType.unset));
}

const SystfBinding = struct { t: DeviceType, host: *const abi.SystfHost };
const Library = @This();

/// Returns a library holding the built-ins, each resolved through its linked
/// object. Caller owns it and must `deinit` it.
pub fn init(gpa: Allocator) Allocator.Error!Library {
    var lib: Library = .{ .gpa = gpa };
    errdefer lib.names.deinit(gpa);
    errdefer lib.vtables.deinit(gpa);
    errdefer lib.digital.deinit(gpa);
    try lib.names.ensureTotalCapacity(gpa, builtin_count);
    try lib.vtables.ensureTotalCapacity(gpa, builtin_count);
    try lib.digital.appendNTimes(gpa, false, builtin_count);
    inline for (builtins) |name| {
        lib.names.appendAssumeCapacity(name);
        lib.vtables.appendAssumeCapacity(catalog.vtable(name));
    }
    assert(lib.names.items.len == builtin_count);
    assert(lib.vtables.items.len == builtin_count);
    return lib;
}

/// Frees the loaded names and the tables. Loaded libraries stay mapped, so
/// vtables handed out stay valid. Asserts that the tables are parallel.
pub fn deinit(self: *Library) void {
    assert(self.names.items.len >= builtin_count);
    assert(self.vtables.items.len == self.names.items.len);
    for (self.names.items[builtin_count..]) |name| self.gpa.free(name);
    for (self.sources.items) |path| self.gpa.free(path);
    self.sources.deinit(self.gpa);
    self.systf_hosts.deinit(self.gpa);
    self.names.deinit(self.gpa);
    self.vtables.deinit(self.gpa);
    self.digital.deinit(self.gpa);
    self.* = undefined;
}

/// Returns the id of built-in `name`; a compile error when there is none.
pub fn builtin(comptime name: []const u8) DeviceType {
    inline for (builtins, 0..) |b, i| {
        if (comptime std.mem.eql(u8, b, name)) return @fromBackingInt(@intCast(i));
    }
    @compileError("device: no built-in model named '" ++ name ++ "'");
}

/// Returns the vtable of type `t`, valid for the life of the process.
/// Asserts that `t` is registered (`.unset` never is).
pub fn vtable(self: *const Library, t: DeviceType) *const abi.DeviceVtable {
    assert(t != .unset);
    assert(@backingInt(t) < self.vtables.items.len);
    return self.vtables.items[@backingInt(t)];
}

/// Returns the runtime-loaded type named `module`, case-insensitively, or
/// null. Built-ins are not searched: a netlist reaches them through SPICE
/// letters and LEVELs.
// ponytail: linear scan; a deck loads a handful of HDL modules.
pub fn find(self: *const Library, module: []const u8) ?DeviceType {
    assert(self.names.items.len >= builtin_count);
    assert(self.names.items.len <= @backingInt(DeviceType.unset));
    for (self.names.items[builtin_count..], builtin_count..) |n, i| {
        if (std.ascii.eqlIgnoreCase(n, module)) return @fromBackingInt(@intCast(i));
    }
    return null;
}

/// Adds a device type under `module` (copied and lowercased) and returns its
/// id; `is_digital` marks a `.v` design (`digital`). A name already loaded
/// keeps its first vtable, flag and id.
pub fn register(self: *Library, module: []const u8, vt: *const abi.DeviceVtable, is_digital: bool) RegisterError!DeviceType {
    assert(self.vtables.items.len == self.names.items.len);
    assert(self.digital.items.len == self.names.items.len);
    if (self.find(module)) |t| return t;
    if (self.names.items.len >= @backingInt(DeviceType.unset)) return error.TooManyDeviceTypes;
    const owned = try std.ascii.allocLowerString(self.gpa, module);
    errdefer self.gpa.free(owned);
    try self.vtables.ensureUnusedCapacity(self.gpa, 1);
    try self.digital.ensureUnusedCapacity(self.gpa, 1);
    try self.names.append(self.gpa, owned);
    self.vtables.appendAssumeCapacity(vt);
    self.digital.appendAssumeCapacity(is_digital);
    assert(self.vtables.items.len == self.names.items.len);
    assert(self.names.items.len > builtin_count);
    return @fromBackingInt(@intCast(self.names.items.len - 1));
}

/// Loads the VPI application at `path` (`--vpi`, `.options vpi=`) and runs
/// its startup routines. Call before `load`: a registration decides how a
/// module compiles (LRM 12.32.1). Fails with `VpiUnavailable` when this
/// build has no VPI host (`vpi` null: an embedding library), else with the
/// host's `VpiError`.
pub fn loadVpiApp(self: *Library, path: []const u8) (error{VpiUnavailable} || VpiError)!void {
    const v = self.vpi orelse {
        if (!builtin_info.is_test) std.log.err("vpi: {s}: this build loads no VPI applications; run the espice executable", .{path});
        return error.VpiUnavailable;
    };
    try v.load_app(v.ctx, path);
    self.vpi_loaded = true;
}

/// Returns the application type `t` calls its `$systf`s through; null when
/// no application is loaded, or `t` calls none.
pub fn systfHost(self: *const Library, t: DeviceType) ?*const abi.SystfHost {
    assert(t != .unset);
    for (self.systf_hosts.items) |b| if (b.t == t) return b.host;
    return null;
}

/// Returns whether `call` is a system task (its value unused), when VerA
/// says; false (a used value) on a contract that does not mark tasks
/// (VerA#83).
pub fn isSystfTask(call: abi.Systf) bool {
    return if (@hasField(abi.Systf, "task")) call.task else false;
}

/// Returns whether a loaded application registered `name` (LRM 12.32).
pub fn systfRegistered(self: *const Library, name: []const u8) bool {
    const v = self.vpi orelse return false;
    return self.vpi_loaded and v.registered(v.ctx, name);
}

/// Compiles, opens and registers each HDL source not loaded yet. The
/// generated devices build against `toolchain.share`, else the sources
/// `zig build` installs in `<exe>/../share/espice`, else the source tree
/// this espice was built from, with `toolchain.zig`, else `zig` on PATH.
/// Builds are cached under `toolchain.cache_dir`. Fails as `LoadError`
/// says; VerA and the loader compile out on wasm.
pub fn load(self: *Library, io: std.Io, files: []const []const u8) LoadError!void {
    if (comptime builtin_info.cpu.arch.isWasm()) return error.RuntimeHdlUnsupported;
    var arena_state: std.heap.ArenaAllocator = .init(self.gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const share = self.toolchain.share orelse
        try std.fs.path.join(arena, &.{ std.process.executableDirPathAlloc(io, arena) catch "", "..", "share", "espice" });
    const eval_path = try std.fs.path.join(arena, &.{ share, "device", "eval.zig" });
    const installed = if (std.Io.Dir.cwd().access(io, eval_path, .{})) |_| true else |_| false;
    // An explicit share without the sources is the caller's mistake;
    // building against the source tree instead would hide it.
    if (self.toolchain.share != null and !installed) {
        if (!builtin_info.is_test) std.log.warn("hdl: {s} holds no device/eval.zig; point the toolchain share path at the share/espice directory zig build installs", .{share});
        return error.HdlShareNotFound;
    }
    var paths: loader.BuildPaths = .{
        .work_dir = self.toolchain.cache_dir orelse "/tmp/espice-cache/hdl",
        .contract = build_options.contract_path,
        .dyn = build_options.dyn_path,
        .gompute = build_options.gompute_path,
        .device_abi = build_options.device_abi_path,
        .core = build_options.core_path,
        .zig = self.toolchain.zig orelse "zig",
        .int_systfs = if (self.vpi) |v| (if (self.vpi_loaded) v.int_systfs(v.ctx) else &.{}) else &.{},
        .include_dirs = self.toolchain.include_dirs,
    };
    if (installed) {
        paths.contract = try std.fs.path.join(arena, &.{ share, "vera", "lib", "contract.zig" });
        paths.dyn = eval_path;
        paths.gompute = try std.fs.path.join(arena, &.{ share, "gompute", "root.zig" });
        paths.device_abi = try std.fs.path.join(arena, &.{ share, "device", "abi.zig" });
        paths.core = try std.fs.path.join(arena, &.{ share, "core", "root.zig" });
    }
    try self.loadWith(io, files, paths);
}

/// `load` against explicit build paths: compiles and opens every file not
/// loaded yet in parallel (`loader.prepareAll`), then registers the results
/// in `files` order and attaches a loaded VPI application to each module
/// that calls a `$systf`. Fails as `load` does, wasm aside.
pub fn loadWith(self: *Library, io: std.Io, files: []const []const u8, paths: loader.BuildPaths) LoadError!void {
    var arena_state: std.heap.ArenaAllocator = .init(self.gpa);
    defer arena_state.deinit();
    const todo = try loader.pending(self.sources.items, arena_state.allocator(), files);
    assert(todo.files.len == todo.resolved.len);
    const results = try self.gpa.alloc(loader.Result, todo.files.len);
    defer self.gpa.free(results);
    try loader.prepareAll(self.gpa, io, todo.files, self.names.items[builtin_count..], paths, results);
    try self.sources.ensureUnusedCapacity(self.gpa, todo.resolved.len);
    for (results, todo.resolved) |result, resolved| {
        if (try result) |prepared| {
            const t = try self.register(prepared.name(), prepared.vtable, prepared.digital);
            const calls = prepared.vtable.systf_calls;
            for (calls) |c| if (isSystfTask(c) and !self.systfRegistered(c.name))
                std.log.warn("{s}: no VPI application registered (LRM 12.32); the call does nothing", .{c.name});
            // LRM 12.32: bind the module's calls to the loaded application.
            // Without one, `NetBuilder` refuses or no-ops each call by name.
            if (calls.len != 0 and self.vpi_loaded) {
                const v = self.vpi.?; // `vpi_loaded` is only ever set through `vpi`.
                try self.systf_hosts.ensureUnusedCapacity(self.gpa, 1);
                self.systf_hosts.appendAssumeCapacity(.{ .t = t, .host = try v.attach(v.ctx, resolved, calls) });
            }
        }
        self.sources.appendAssumeCapacity(try self.gpa.dupe(u8, resolved));
    }
    assert(self.vtables.items.len == self.names.items.len);
}

// ---- private ----

fn noVpiApp(_: *anyopaque, _: usize, _: []const f64, partials: []f64) f64 {
    @memset(partials, 0);
    return 0;
}

test register {
    var lib = try Library.init(std.testing.allocator);
    defer lib.deinit();
    var vts: [2]abi.DeviceVtable = undefined;
    _ = &vts;
    const a = try lib.register("MyRes", &vts[0], false);
    try std.testing.expectEqual(builtin_count, @backingInt(a));
    try std.testing.expectEqualStrings("myres", lib.names.items[@backingInt(a)]);
    // A second spelling of a loaded name keeps the first registration.
    try std.testing.expectEqual(a, try lib.register("myRES", &vts[1], true));
    try std.testing.expect(lib.vtable(a) == &vts[0]);
    try std.testing.expect(!lib.digital.items[@backingInt(a)]);
    try std.testing.expectEqual(@as(?DeviceType, a), lib.find("MYRES"));
    const d = try lib.register("logic", &vts[1], true);
    try std.testing.expectEqual(builtin_count + 1, @backingInt(d));
    try std.testing.expect(lib.digital.items[@backingInt(d)]);
    try std.testing.expectEqual(lib.names.items.len, lib.vtables.items.len);
    try std.testing.expectEqual(lib.names.items.len, lib.digital.items.len);
}

test "an explicit share path must hold the evaluator sources" {
    if (comptime builtin_info.cpu.arch.isWasm()) return error.SkipZigTest;
    var lib = try Library.init(std.testing.allocator);
    defer lib.deinit();
    lib.toolchain = .{ .share = "/nonexistent-espice-test-dir/share/espice", .zig = "/nonexistent/zig" };
    try std.testing.expectError(error.HdlShareNotFound, lib.load(std.testing.io, &.{"m.va"}));
    try std.testing.expectEqual(@as(usize, builtin_count), lib.names.items.len);
}

test "find searches loaded types only" {
    var lib = try Library.init(std.testing.allocator);
    defer lib.deinit();
    try std.testing.expectEqualStrings("resistor", lib.names.items[@backingInt(builtin("resistor"))]);
    try std.testing.expectEqual(@as(?DeviceType, null), lib.find("resistor"));
    try std.testing.expectEqual(@as(?DeviceType, null), lib.find(""));
}

test "init and register free everything when an allocation fails" {
    const case = struct {
        fn run(gpa: Allocator) !void {
            var lib = try Library.init(gpa);
            defer lib.deinit();
            var vt: abi.DeviceVtable = undefined;
            _ = &vt;
            _ = try lib.register("a", &vt, false);
            _ = try lib.register("b", &vt, true);
            try std.testing.expectEqual(lib.names.items.len, lib.digital.items.len);
        }
    }.run;
    try std.testing.checkAllAllocationFailures(@import("core").testing.fixed_growth, case, .{});
}

test no_vpi_app {
    var partials = [_]f64{ 1, 2 };
    try std.testing.expectEqual(@as(f64, 0), no_vpi_app.call(no_vpi_app.ctx, 0, &.{3}, &partials));
    try std.testing.expectEqualSlices(f64, &.{ 0, 0 }, &partials);
}
