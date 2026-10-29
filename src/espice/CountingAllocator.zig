//! A thread-safe allocator over a caller's one: allocation calls in, the same
//! calls on `backing` out under one mutex, with the bytes held now and the
//! most held at once counted on the way. The query workers of a `Problem`
//! allocate through it, so a caller's allocator need not be thread-safe.
//! Cites: none.
//!
//! Contents:
//! - `pub const Usage = struct { live: usize, peak: usize }`
//! - `pub fn init(backing: Allocator, io: std.Io) CountingAllocator`
//! - `pub fn allocator(self: *CountingAllocator) Allocator`
//! - `pub fn usage(self: *CountingAllocator) Usage`

const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;

/// Bytes held now and the most held at once, in bytes requested.
pub const Usage = struct { live: usize, peak: usize };

/// Every call reaches it under `mutex`. Borrowed; it must outlive this.
backing: Allocator,
io: std.Io,
mutex: std.Io.Mutex = .init,
/// Bytes requested and not yet freed, and the most at once. Read and
/// written under `mutex`; `live_bytes <= peak_bytes` always holds.
live_bytes: usize = 0,
peak_bytes: usize = 0,

const CountingAllocator = @This();

/// Counts nothing yet. Allocates nothing.
pub fn init(backing: Allocator, io: std.Io) CountingAllocator {
    return .{ .backing = backing, .io = io };
}

/// The serialized, counted allocator. Its address must stay put while any
/// allocation through it lives: it points at `self`.
pub fn allocator(self: *CountingAllocator) Allocator {
    return .{ .ptr = self, .vtable = &.{ .alloc = alloc, .resize = resize, .remap = remap, .free = free } };
}

/// A snapshot of the counters.
pub fn usage(self: *CountingAllocator) Usage {
    self.mutex.lockUncancelable(self.io);
    defer self.mutex.unlock(self.io);
    assert(self.live_bytes <= self.peak_bytes);
    return .{ .live = self.live_bytes, .peak = self.peak_bytes };
}

// ---- private ----

fn alloc(ctx: *anyopaque, n: usize, alignment: std.mem.Alignment, ret: usize) ?[*]u8 {
    const self: *CountingAllocator = @ptrCast(@alignCast(ctx));
    self.mutex.lockUncancelable(self.io);
    defer self.mutex.unlock(self.io);
    const p = self.backing.rawAlloc(n, alignment, ret) orelse return null;
    self.account(0, n);
    return p;
}

fn resize(ctx: *anyopaque, memory: []u8, alignment: std.mem.Alignment, n: usize, ret: usize) bool {
    const self: *CountingAllocator = @ptrCast(@alignCast(ctx));
    self.mutex.lockUncancelable(self.io);
    defer self.mutex.unlock(self.io);
    if (!self.backing.rawResize(memory, alignment, n, ret)) return false;
    self.account(memory.len, n);
    return true;
}

fn remap(ctx: *anyopaque, memory: []u8, alignment: std.mem.Alignment, n: usize, ret: usize) ?[*]u8 {
    const self: *CountingAllocator = @ptrCast(@alignCast(ctx));
    self.mutex.lockUncancelable(self.io);
    defer self.mutex.unlock(self.io);
    const p = self.backing.rawRemap(memory, alignment, n, ret) orelse return null;
    self.account(memory.len, n);
    return p;
}

fn free(ctx: *anyopaque, memory: []u8, alignment: std.mem.Alignment, ret: usize) void {
    const self: *CountingAllocator = @ptrCast(@alignCast(ctx));
    self.mutex.lockUncancelable(self.io);
    defer self.mutex.unlock(self.io);
    self.backing.rawFree(memory, alignment, ret);
    self.account(memory.len, 0);
}

/// Moves `old` counted bytes to `new`. Caller holds `mutex`.
fn account(self: *CountingAllocator, old: usize, new: usize) void {
    assert(old <= self.live_bytes); // Only memory this allocator handed out comes back.
    self.live_bytes = self.live_bytes - old + new;
    self.peak_bytes = @max(self.peak_bytes, self.live_bytes);
    assert(self.live_bytes <= self.peak_bytes);
}
