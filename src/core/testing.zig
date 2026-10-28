//! Test support: `std.testing.allocator` behind a vtable that refuses every
//! in-place resize and remap, for `std.testing.checkAllAllocationFailures`
//! sweeps (#128). Allocation requests in, the testing allocator's blocks out.
//! Cites: none.
//!
//! Contents:
//! - `pub const fixed_growth: std.mem.Allocator`

const std = @import("std");
const Allocator = std.mem.Allocator;

/// `std.testing.allocator` that never grows or shrinks a block in place.
/// Whether a large block grows in place depends on the pages around it, so
/// through the plain testing allocator the number of allocations a sweep
/// counts can change between runs (`error.NondeterministicMemoryUsage`).
/// Here every growth is a counted allocation. Test code only: it reaches
/// `std.testing.allocator`, which exists only in a test build.
pub const fixed_growth: Allocator = .{
    // SAFETY: the vtable below never reads `ptr`; the state lives in std.testing.
    .ptr = undefined,
    .vtable = &.{
        .alloc = alloc,
        .resize = Allocator.noResize,
        .remap = Allocator.noRemap,
        .free = free,
    },
};

// ---- private ----

fn alloc(_: *anyopaque, len: usize, alignment: std.mem.Alignment, ret_addr: usize) ?[*]u8 {
    return std.testing.allocator.rawAlloc(len, alignment, ret_addr);
}

fn free(_: *anyopaque, memory: []u8, alignment: std.mem.Alignment, ret_addr: usize) void {
    std.testing.allocator.rawFree(memory, alignment, ret_addr);
}
