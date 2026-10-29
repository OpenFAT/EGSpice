//! What an `espice_problem` token names: one `Problem`, the `Io` it runs
//! on, the caller's display callback, and the last error for
//! `espice_error_message`. `create` builds it from facade options; the
//! token table in handles.zig owns the only references to it.
//! Cites: include/espice.h (`espice_create`, `espice_destroy`,
//! `espice_error_message`).
//!
//! Contents:
//! - `pub const Display = struct { callback: *const records.DisplayFn, user: ?*anyopaque }`
//! - `pub fn create(gpa: Allocator, config: api.Options, display: ?Display) !*Handle`
//! - `pub fn destroy(self: *Handle) void`
//! - `pub fn fail(self: *Handle, err: anyerror) u32`

const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;
const api = @import("espice");
const records = @import("records.zig");

/// The caller's `espice_display_fn` and its user pointer.
pub const Display = struct { callback: *const records.DisplayFn, user: ?*anyopaque };

/// What `create` allocated the handle and its Problem from.
gpa: Allocator,
/// The Problem's Io points into it, so the handle keeps a stable address.
threaded: std.Io.Threaded,
problem: *api.Problem,
/// Read by `espice_error_message`; set by every failing call that got past
/// the handle lookup, and by an advance event's output error.
last_error: ?anyerror,
/// `Options.display` when the caller set a display callback. Unbuffered,
/// so concurrent queries never touch its state.
display: std.Io.Writer,
display_to: ?Display,

const Handle = @This();

/// Allocates a handle from `gpa` and builds its Problem from `config` on a
/// single-threaded Io. With `display`, the Problem's display-task text goes
/// to that callback. Caller owns the handle and frees it with `destroy`.
/// Fails as `Problem.init` does, or with `error.OutOfMemory`; nothing is
/// left allocated then.
pub fn create(gpa: Allocator, config: api.Options, display: ?Display) !*Handle {
    assert(config.display == null);
    const self = try gpa.create(Handle);
    errdefer gpa.destroy(self);
    // Query workers use std.Thread directly. The single-threaded Io still
    // supplies real futexes, and unlike Threaded.init it installs no
    // process-wide signal handlers inside the host application.
    self.gpa = gpa;
    self.threaded = .init_single_threaded;
    errdefer self.threaded.deinit();
    // Spawning a `.hdl` build allocates its argv and hands the child an
    // environment (the compiler finds its cache through HOME); the static
    // instance has neither.
    self.threaded.allocator = gpa;
    const environ = try copyEnviron(gpa);
    errdefer environ.deinit(gpa);
    self.threaded.environ = .{ .process_environ = .{ .block = environ } };
    self.threaded.environ_initialized = false;
    self.last_error = null;
    self.display = .{ .vtable = &.{ .drain = drainDisplay }, .buffer = &.{} };
    self.display_to = display;
    var options = config;
    if (display != null) options.display = &self.display;
    self.problem = try api.Problem.init(gpa, self.threaded.io(), options);
    assert(self.last_error == null);
    return self;
}

/// Frees the Problem, the environment copy and the handle itself.
/// Invalidates every result view the Problem handed out.
pub fn destroy(self: *Handle) void {
    const gpa = self.gpa;
    self.problem.deinit();
    const environ = self.threaded.environ.process_environ.block;
    self.threaded.deinit();
    environ.deinit(gpa);
    gpa.destroy(self);
}

/// Records `err` for `espice_error_message` and returns its status code.
pub fn fail(self: *Handle, err: anyerror) u32 {
    self.last_error = err;
    const code = records.status(err);
    assert(code != 0);
    return code;
}

// ---- private ----

/// A copy of the process environment at create time: libc's `environ` can
/// move under a later `setenv`. Caller owns it; free with `deinit(gpa)`.
fn copyEnviron(gpa: Allocator) !std.process.Environ.Block {
    const Environ = std.process.Environ;
    if (comptime Environ.Block != Environ.PosixBlock) return .global;
    const env = std.mem.span(std.c.environ);
    const block = try gpa.allocSentinel(?[*:0]const u8, env.len, null);
    var copied: usize = 0;
    errdefer {
        for (block[0..copied]) |entry| gpa.free(std.mem.span(entry.?));
        gpa.free(block);
    }
    for (env, block) |entry, *dst| {
        dst.* = (try gpa.dupeSentinel(u8, std.mem.span(entry.?), 0)).ptr;
        copied += 1;
    }
    assert(copied == env.len);
    return .{ .slice = block };
}

/// `display`'s drain: hands each buffered slice to the caller's callback,
/// the last one `splat` times.
fn drainDisplay(w: *std.Io.Writer, data: []const []const u8, splat: usize) std.Io.Writer.Error!usize {
    assert(data.len != 0);
    const self: *Handle = @alignCast(@fieldParentPtr("display", w));
    const to = self.display_to.?; // `create` installs the writer only with a display.
    var n: usize = 0;
    for (data[0 .. data.len - 1]) |d| {
        if (d.len != 0) to.callback(d.ptr, d.len, to.user);
        n += d.len;
    }
    const last = data[data.len - 1];
    if (last.len != 0) for (0..splat) |_| to.callback(last.ptr, last.len, to.user);
    return n + last.len * splat;
}
