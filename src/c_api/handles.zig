//! The table of live problems behind `espice_problem`: a `*Handle` in, a
//! token out, and a token back to its handle only while that handle lives,
//! so a destroyed or foreign handle is an error code rather than a
//! use-after-free (#143). Process-global by necessity: a C caller holds only
//! the token, and every entry point must judge it with no other context.
//! Cites: include/espice.h, `espice_problem` and ESPICE_INVALID_HANDLE.
//!
//! A token packs a slot index into its low `index_bits` and the slot's
//! generation above them. Slot 0 is never handed out, so no token is NULL.
//! `remove` bumps the slot's generation, so every copy of the old token
//! stops matching; a slot whose generation would wrap is retired instead of
//! reused. One lock guards the table, and every critical section is O(1):
//! it never allocates, calls out or waits on anything but the lock.
//!
//! Contents:
//! - `pub const Token = opaque {}`
//! - `pub const index_bits: u16`
//! - `pub const capacity: u32`
//! - `pub const Index = @Int(.unsigned, index_bits)`
//! - `pub const Generation = @Int(.unsigned, @bitSizeOf(usize) - index_bits)`
//! - `pub var generation: [capacity]Generation`
//! - `pub var free_len: u32`
//! - `pub fn insert(h: *Handle) error{TooManyProblems}!*Token`
//! - `pub fn get(token: ?*Token) error{ InvalidArgument, InvalidHandle }!*Handle`
//! - `pub fn remove(token: *Token) error{InvalidHandle}!*Handle`

const std = @import("std");
const assert = std.debug.assert;
const Handle = @import("Handle.zig");

/// `espice_problem`: a C handle is a token, never an address.
pub const Token = opaque {};

/// Bits of a token that index the table.
pub const index_bits = 12;
/// Slots, slot 0 included; `capacity - 1` problems can be live at once.
pub const capacity = 1 << index_bits;
pub const Index = @Int(.unsigned, index_bits);
pub const Generation = @Int(.unsigned, @bitSizeOf(usize) - index_bits);

/// Each slot's current generation; a token matches only at it. Written
/// under `mutex` (tests set it directly, single-threaded).
pub var generation: [capacity]Generation = @splat(0);
/// The length of `free`. Under `mutex`.
pub var free_len: u32 = 0;

comptime {
    assert(capacity - 1 <= std.math.maxInt(Index));
    assert(index_bits + @bitSizeOf(Generation) == @bitSizeOf(usize));
}

var mutex: std.Io.Mutex = .init;
/// The handle each slot names, null when free or retired.
var live: [capacity]?*Handle = @splat(null);
/// Slots `remove` released, reused before `fresh` advances.
// SAFETY: only `free[0..free_len]` is ever read, and `remove` writes each
// entry before raising `free_len` over it.
var free: [capacity]Index = undefined;
/// The lowest slot never handed out.
var fresh: u32 = 1;

/// Registers `h` and returns its token. `error.TooManyProblems` when every
/// slot is live or retired. Asserts that `h` is not registered already in
/// the slot it gets.
pub fn insert(h: *Handle) error{TooManyProblems}!*Token {
    mutex.lockUncancelable(io());
    defer mutex.unlock(io());
    const index: Index = if (free_len > 0) reuse: {
        free_len -= 1;
        break :reuse free[free_len];
    } else if (fresh < capacity) next: {
        fresh += 1;
        break :next @intCast(fresh - 1);
    } else return error.TooManyProblems;
    assert(index != 0);
    assert(live[index] == null);
    live[index] = h;
    // SAFETY: the token is a packed (generation, index) pair, never
    // dereferenced; index != 0 keeps it non-NULL.
    return @ptrFromInt(@as(usize, generation[index]) << index_bits | index);
}

/// The handle `token` names. NULL is `error.InvalidArgument`; a destroyed
/// token, or any value whose bits name no live slot at its current
/// generation, is `error.InvalidHandle`.
pub fn get(token: ?*Token) error{ InvalidArgument, InvalidHandle }!*Handle {
    const bits = @intFromPtr(token orelse return error.InvalidArgument);
    mutex.lockUncancelable(io());
    defer mutex.unlock(io());
    return live[try find(bits)].?;
}

/// Unregisters `token`'s handle and returns it; the token goes stale.
pub fn remove(token: *Token) error{InvalidHandle}!*Handle {
    mutex.lockUncancelable(io());
    defer mutex.unlock(io());
    const index = try find(@intFromPtr(token));
    const h = live[index].?;
    live[index] = null;
    // Retired: its last token keeps failing `find` on the null slot.
    if (generation[index] == std.math.maxInt(Generation)) return h;
    generation[index] += 1;
    assert(free_len < capacity);
    free[free_len] = index;
    free_len += 1;
    assert(live[index] == null);
    return h;
}

// ---- private ----

/// The futex behind `mutex`; the single-threaded Io still supplies real
/// futexes (see `Handle.create`).
fn io() std.Io {
    return std.Io.Threaded.global_single_threaded.io();
}

/// The live slot `bits` names. Caller holds `mutex`.
fn find(bits: usize) error{InvalidHandle}!Index {
    const index: Index = @truncate(bits);
    if (live[index] == null) return error.InvalidHandle;
    if (generation[index] != @as(Generation, @truncate(bits >> index_bits))) return error.InvalidHandle;
    assert(index != 0); // Slot 0 is never live.
    return index;
}
