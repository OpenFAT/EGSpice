//! The scratch slab every ordering pass draws from: a caller-owned u32
//! slab in, bump-allocated sub-slices out, released to a mark in LIFO order.
//! No allocator is involved, so the orderings also run at comptime.
//! Cites nothing.
//!
//! Contents:
//! - `pub const Error = error{OutOfWorkspace}`
//! - `pub const Ws = struct`
//!   - `pub fn init(buf: []u32) Ws`
//!   - `pub fn alloc(w: *Ws, m: usize) Error![]u32`
//!   - `pub fn allocSet(w: *Ws, m: usize, fill: u32) Error![]u32`
//!   - `pub fn mark(w: *const Ws) usize`
//!   - `pub fn release(w: *Ws, m: usize) void`

const std = @import("std");
const assert = std.debug.assert;

/// Every way a slab allocation fails; the slab is unchanged after it.
pub const Error = error{
    /// The slab from `order.wsSize` was too small for this pattern's fill.
    OutOfWorkspace,
};

/// Bump allocator over a caller-owned u32 slab, with mark/release.
pub const Ws = struct {
    /// The borrowed slab.
    buf: []u32,
    /// Words handed out, from the front of `buf`.
    used: usize = 0,

    /// Borrows `buf`, which must outlive every slice handed out.
    pub fn init(buf: []u32) Ws {
        return .{ .buf = buf };
    }

    /// The next `m` words, uninitialized; `OutOfWorkspace` when they do
    /// not fit. Valid until a `release` to a mark at or before them.
    pub fn alloc(w: *Ws, m: usize) Error![]u32 {
        if (w.used + m > w.buf.len) return error.OutOfWorkspace;
        const s = w.buf[w.used .. w.used + m];
        w.used += m;
        assert(w.used <= w.buf.len);
        return s;
    }

    /// Returns the next `m` words of the slab, every word set to `fill`;
    /// `OutOfWorkspace` when they do not fit.
    pub fn allocSet(w: *Ws, m: usize, fill: u32) Error![]u32 {
        const s = try w.alloc(m);
        @memset(s, fill);
        return s;
    }

    /// The current top, for a later `release`.
    pub fn mark(w: *const Ws) usize {
        return w.used;
    }

    /// Frees everything allocated since `mark` returned `m`. Asserts that
    /// `m` is at or below the current top.
    pub fn release(w: *Ws, m: usize) void {
        assert(m <= w.used);
        w.used = m;
    }
};
