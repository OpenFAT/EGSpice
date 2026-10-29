//! ngspice's diagonal-gmin placement for the stepping rungs. LoadGmin adds
//! gmin to the diagonal Sparse holds after spMNA_Preorder, which swaps each
//! column with no diagonal element for a symmetric pair of +-1 entries: a
//! grounded source's gmin lands on its two incidence entries, turning
//! V = E into (1 + gmin) V = E, and not on its node's diagonal.
//! In: the circuit's pattern and its last assembled g plane. Out: one
//! `GminStamp` per row that takes a gmin.
//! Cites ngspice spsmp.c LoadGmin and sputils.c spMNA_Preorder
//! (CountTwins, SwapCols).
//!
//! Contents:
//! - `pub fn build(ckt: *const Circuit, gpa: Allocator) Allocator.Error![]converger.GminStamp`

const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;
const converger = @import("solver").converger;
const root = @import("../types.zig");

/// The first +-1 entry of a column position whose transpose is +-1 too, and
/// how many there are (counting stops at 2).
const Twin = struct { count: u32, row: u32 };

/// Sparse's `Diag == NULL`: a column position with no diagonal element.
const no_diagonal = std.math.maxInt(u32);

/// Returns the gmin stamps of `ckt`, whose planes must hold an assembled
/// iterate. Row 0 (ground) takes none. Caller owns the result and frees it
/// with `gpa`.
pub fn build(ckt: *const root.Circuit, gpa: Allocator) Allocator.Error![]converger.GminStamp {
    const n = ckt.n;
    assert(n >= 1);
    // perm: column position -> original column. dcol: original column of the
    // diagonal element at each position, `no_diagonal` for a branch row
    // nothing stamps a diagonal into.
    const perm = try gpa.alloc(u32, n);
    defer gpa.free(perm);

    const dcol = try gpa.alloc(u32, n);
    defer gpa.free(dcol);

    for (perm, dcol, 0..) |*p, *d, i| {
        const slot = ckt.diag_slots[i];
        p.* = @intCast(i);
        d.* = if (ckt.current_row[i] and ckt.g_vals[slot] == 0 and ckt.c_vals[slot] == 0) no_diagonal else @intCast(i);
    }
    preorder(ckt, perm, dcol);

    var stamps: std.ArrayList(converger.GminStamp) = .empty;
    errdefer stamps.deinit(gpa);

    try stamps.ensureTotalCapacityPrecise(gpa, n);
    for (1..n) |i| if (dcol[i] != no_diagonal) stamps.appendAssumeCapacity(.{
        .slot = ckt.findSlot(@intCast(i), dcol[i]).?,
        .row = @intCast(i),
        .col = dcol[i],
    });
    assert(stamps.items.len < n);
    return stamps.toOwnedSlice(gpa);
}

// ---- private ----

/// spMNA_Preorder: lone twins first, then the first multi-twin column, until
/// no swap happens. Each pass swaps at least once or ends, and a swap fills
/// a missing diagonal, so at most `n` passes run.
fn preorder(ckt: *const root.Circuit, perm: []u32, dcol: []u32) void {
    const n: u32 = ckt.n;
    var start: u32 = 1;
    var passes: u32 = 0;
    while (true) : (passes += 1) {
        assert(passes <= n);
        var swapped = false;
        var again = false;
        var j = start;
        while (j < n) : (j += 1) if (dcol[j] == no_diagonal) {
            const t = countTwins(ckt, perm, j);
            if (t.count == 1) {
                swapCols(perm, dcol, j, t.row);
                swapped = true;
            } else if (t.count > 1 and !again) {
                again = true;
                start = j;
            }
        };
        if (!again) break;
        j = start;
        while (!swapped and j < n) : (j += 1) if (dcol[j] == no_diagonal) {
            const t = countTwins(ckt, perm, j);
            if (t.count > 0) {
                swapCols(perm, dcol, j, t.row);
                swapped = true;
            }
        };
        if (!swapped) break;
    }
}

/// CountTwins over column position `j`.
fn countTwins(c: *const root.Circuit, perm: []const u32, j: u32) Twin {
    var t: Twin = .{ .count = 0, .row = 0 };
    const col = perm[j];
    for (c.col_ptr[col]..c.col_ptr[col + 1]) |p| {
        const r = c.row_idx[p];
        if (r == root.GROUND or @abs(c.g_vals[p]) != 1.0) continue;
        const q = c.findSlot(j, perm[r]) orelse continue;
        if (@abs(c.g_vals[q]) != 1.0) continue;
        t.count += 1;
        if (t.count == 1) t.row = r else break;
    }
    assert(t.count <= 2);
    return t;
}

/// SwapCols: exchange positions `j` and `r`; each diagonal becomes the twin.
fn swapCols(perm: []u32, dcol: []u32, j: u32, r: u32) void {
    assert(j != r);
    dcol[j] = perm[r];
    dcol[r] = perm[j];
    std.mem.swap(u32, &perm[j], &perm[r]);
}
