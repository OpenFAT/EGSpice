//! Fill-reducing column ordering for the sparse LU: a CSC pattern in, the
//! column order q out. Block triangular form by Tarjan SCC, then AMD
//! (`amd.zig`) inside each block, or the natural order where that fills less
//! (#46). All scratch comes from one caller-owned u32 slab (`order_ws.zig`),
//! with no allocator, so the same code also runs at comptime.
//! Cites: Tarjan, "Depth-first search and linear graph algorithms", SIAM J.
//! Comput. 1(2), 1972; Liu, "The role of elimination trees in sparse
//! factorization", SIAM J. Matrix Anal. Appl. 11(1), 1990 (`cholCount`);
//! dev/solvers/btf-permutation.md.
//!
//! Contents:
//! - `pub const Error = order_ws.Error`
//! - `pub const Ws = order_ws.Ws`
//! - `pub const test_access = ...` (test builds only)
//! - `pub fn wsSize(n: usize, nnz: usize) usize`
//! - `pub fn order(n: u32, col_ptr: []const u32, row_idx: []const u32, q: []u32, ws: *Ws) Error!void`
//! - `pub fn blocks(n: u32, col_ptr: []const u32, row_idx: []const u32, emit: []u32, block_ptr: []u32, ws: *Ws) Error!u32`
//! - `pub fn diagFill(n: u32, col_ptr: []const u32, row_idx: []const u32, q: []const u32, ws: *Ws) Error!u64`

const std = @import("std");
const assert = std.debug.assert;
const order_ws = @import("order_ws.zig");
const amd = @import("amd.zig").amd;

/// Every way an ordering fails; `ws` is restored after it.
pub const Error = order_ws.Error;

/// The slab allocator every ordering pass takes.
pub const Ws = order_ws.Ws;

/// Private functions for tests.zig; void outside test builds.
pub const test_access = if (@import("builtin").is_test) .{
    .amd = amd,
    .allocSet = Ws.allocSet,
} else {};

const NONE: u32 = std.math.maxInt(u32);

/// Slab length for `order` on an n x n pattern with nnz entries.
/// A pathological fill can still exceed it (`error.OutOfWorkspace`).
pub fn wsSize(n: usize, nnz: usize) usize {
    return 48 * n + 8 * nnz + 64;
}

/// Writes q[k] = the column factored at step k: strongly connected blocks of
/// the graph j -> i for A(i,j) != 0 (`blocks`), sink block first, each
/// block's columns in AMD order or in natural order, whichever gives the
/// smaller Cholesky factor of the block's A + A^T (#46). Requires a
/// structurally full diagonal (the pattern merge guarantees it), so no
/// matching step is needed. Restores `ws` on return.
/// Asserts that `q` has n entries and `col_ptr` n + 1.
pub fn order(n: u32, col_ptr: []const u32, row_idx: []const u32, q: []u32, ws: *Ws) Error!void {
    if (n == 0) return;
    assert(q.len == n);
    assert(col_ptr.len == @as(usize, n) + 1);
    const base = ws.mark();
    defer ws.release(base);

    const emit = try ws.alloc(n); // vertices grouped by block
    const block_ptr = try ws.alloc(n + 1); // block boundaries in `emit`
    const nb = try blocks(n, col_ptr, row_idx, emit, block_ptr, ws);

    // AMD inside each non-singleton block, on the block's sub-pattern.
    const blk_of = try ws.alloc(n);
    const loc = try ws.alloc(n);
    for (0..nb) |b| {
        for (block_ptr[b]..block_ptr[b + 1], 0..) |i, li| {
            blk_of[emit[i]] = @intCast(b);
            loc[emit[i]] = @intCast(li);
        }
    }

    for (0..nb) |b| {
        const lo = block_ptr[b];
        const hi = block_ptr[b + 1];
        const bs = hi - lo;
        const verts = emit[lo..hi];

        if (bs == 1) {
            q[lo] = verts[0];
            continue;
        }

        const blk_mark = ws.mark();
        defer ws.release(blk_mark);

        const sub_cp = try ws.alloc(bs + 1);
        var sub_nnz: u32 = 0;
        sub_cp[0] = 0;
        for (verts, 0..) |v, j| {
            for (row_idx[col_ptr[v]..col_ptr[v + 1]]) |r| sub_nnz += @intFromBool(blk_of[r] == b);
            sub_cp[j + 1] = sub_nnz;
        }
        const sub_ri = try ws.alloc(sub_nnz);
        var w: u32 = 0;
        for (verts) |v| {
            for (col_ptr[v]..col_ptr[v + 1]) |p| {
                const r = row_idx[p];
                if (blk_of[r] == b) {
                    sub_ri[w] = loc[r];
                    w += 1;
                }
            }
        }
        const sub_q = try ws.alloc(bs);
        try amd(@intCast(bs), sub_cp, sub_ri, sub_q, ws);
        // AMD is a heuristic: on a rail-last inverter chain it fills more
        // than the natural order (50 against 48 at n = 10), so the block
        // keeps whichever of the two fills less.
        const nat = try ws.alloc(bs);
        @memcpy(nat, verts);
        std.sort.pdq(u32, nat, {}, std.sort.asc(u32));
        for (nat) |*v| v.* = loc[v.*];
        // The natural count stops at AMD's: a block whose natural order
        // fills densely (chain_bsim4_100k: tens of seconds) costs no more
        // than AMD's own count.
        const amd_count = try cholCount(sub_cp, sub_ri, sub_q, ws, std.math.maxInt(u64));
        const ord = if (try cholCount(sub_cp, sub_ri, nat, ws, amd_count) < amd_count) nat else sub_q;
        for (ord, 0..) |lq, i| q[lo + i] = verts[lq];
    }
}

/// Block triangular form: the strongly connected blocks of the graph
/// j -> i for A(i,j) != 0 by iterative Tarjan, sink block first. Writes
/// each block's vertices to `emit[block_ptr[b]..block_ptr[b + 1]]` and
/// returns the block count. Restores `ws` on return. Asserts that `emit`
/// has n entries and `block_ptr` n + 1.
pub fn blocks(n: u32, col_ptr: []const u32, row_idx: []const u32, emit: []u32, block_ptr: []u32, ws: *Ws) Error!u32 {
    assert(emit.len == n);
    assert(block_ptr.len == @as(usize, n) + 1);
    const base = ws.mark();
    defer ws.release(base);
    const num = try ws.allocSet(n, NONE); // DFS number
    const low = try ws.alloc(n); // low-link
    const on = try ws.allocSet(n, 0); // on the SCC stack
    const scc_stack = try ws.alloc(n);
    const frame_v = try ws.alloc(n); // DFS frame: vertex
    const frame_p = try ws.alloc(n); // DFS frame: next edge

    var sp: u32 = 0;
    var fp: u32 = 0;
    var counter: u32 = 0;
    var no: u32 = 0;
    var nb: u32 = 0;
    block_ptr[0] = 0;

    for (0..n) |root| {
        if (num[root] != NONE) continue;
        num[root] = counter;
        low[root] = counter;
        counter += 1;
        scc_stack[sp] = @intCast(root);
        sp += 1;
        on[root] = 1;
        frame_v[fp] = @intCast(root);
        frame_p[fp] = col_ptr[root];
        fp += 1;

        while (fp > 0) {
            const v = frame_v[fp - 1];
            if (frame_p[fp - 1] < col_ptr[v + 1]) {
                const child = row_idx[frame_p[fp - 1]];
                frame_p[fp - 1] += 1;
                if (child == v) continue;
                if (num[child] == NONE) {
                    num[child] = counter;
                    low[child] = counter;
                    counter += 1;
                    scc_stack[sp] = child;
                    sp += 1;
                    on[child] = 1;
                    frame_v[fp] = child;
                    frame_p[fp] = col_ptr[child];
                    fp += 1;
                } else if (on[child] != 0) {
                    low[v] = @min(low[v], num[child]);
                }
                continue;
            }
            // v is finished; if it roots an SCC, pop that block.
            if (low[v] == num[v]) {
                // Terminates: v is still on the SCC stack, below everything
                // pushed after it.
                while (true) {
                    assert(sp > 0);
                    sp -= 1;
                    const m = scc_stack[sp];
                    on[m] = 0;
                    emit[no] = m;
                    no += 1;
                    if (m == v) break;
                }
                nb += 1;
                block_ptr[nb] = no;
            }
            fp -= 1;
            if (fp > 0) {
                low[frame_v[fp - 1]] = @min(low[frame_v[fp - 1]], low[v]);
            }
        }
    }

    // Every vertex landed in exactly one block.
    assert(no == n);
    assert(sp == 0);
    assert(block_ptr[nb] == n);

    return nb;
}

/// A bound on the strictly lower plus strictly upper entries of an LU of
/// the pattern under diagonal pivots in column order `q` (q[k] = column at
/// step k): twice the off-diagonal Cholesky count of A + A^T. O(|L|).
/// Restores `ws` on return; `wsSize(n, nnz)` words suffice.
pub fn diagFill(n: u32, col_ptr: []const u32, row_idx: []const u32, q: []const u32, ws: *Ws) Error!u64 {
    assert(q.len == n);
    return 2 * (try cholCount(col_ptr, row_idx, q, ws, std.math.maxInt(u64)) - n);
}

// ---- private ----

// Entries of the Cholesky factor L, diagonal included, of the symmetrized
// pattern A + A^T eliminated in the order `perm` (perm[k] = column at step
// k): Liu's elimination tree with path compression, then each row's
// subtree walked once, so O(|L|). Stops once the count reaches `cap`
// and returns a value of at least `cap`. Restores `ws` on return.
fn cholCount(col_ptr: []const u32, row_idx: []const u32, perm: []const u32, ws: *Ws, cap: u64) Error!u64 {
    const n = perm.len;
    const nnz = col_ptr[n];
    const base = ws.mark();
    defer ws.release(base);
    const pinv = try ws.alloc(n);
    for (perm, 0..) |c, k| pinv[c] = @intCast(k);
    // Each off-diagonal entry, as the earlier step under the later one.
    const ptr = try ws.allocSet(n + 1, 0);
    const lst = try ws.alloc(nnz);
    for (0..n) |j| for (col_ptr[j]..col_ptr[j + 1]) |p| {
        const a: usize = pinv[row_idx[p]];
        const b: usize = pinv[j];
        // Not @max: at comptime its result narrows to the operands' range.
        if (a != b) ptr[(if (a > b) a else b) + 1] += 1;
    };
    for (1..n + 1) |k| ptr[k] += ptr[k - 1];
    const fill = try ws.alloc(n);
    @memcpy(fill, ptr[0..n]);
    for (0..n) |j| for (col_ptr[j]..col_ptr[j + 1]) |p| {
        const a: usize = pinv[row_idx[p]];
        const b: usize = pinv[j];
        if (a == b) continue;
        const hi = if (a > b) a else b;
        lst[fill[hi]] = @intCast(if (a > b) b else a);
        fill[hi] += 1;
    };
    const parent = try ws.alloc(n);
    const anc = fill; // reused: the counting sort is done
    for (0..n) |k| {
        parent[k] = NONE;
        anc[k] = NONE;
        for (lst[ptr[k]..ptr[k + 1]]) |first| {
            var i = first;
            while (i != NONE and i < k) {
                const next = anc[i];
                anc[i] = @intCast(k);
                if (next == NONE) parent[i] = @intCast(k);
                i = next;
            }
        }
    }
    const flag = anc;
    var count: u64 = 0;
    for (0..n) |k| {
        flag[k] = @intCast(k);
        count += 1;
        for (lst[ptr[k]..ptr[k + 1]]) |first| {
            var i = first;
            // k is an ancestor of every i < k with A(i, k) != 0, so the
            // walk stops at a flagged node.
            while (flag[i] != k) : (i = parent[i]) {
                flag[i] = @intCast(k);
                count += 1;
            }
        }
        if (count >= cap) return count;
    }
    assert(count >= n);
    return count;
}

test "order and amd return permutations and restore the workspace on random patterns" {
    var prng = std.Random.DefaultPrng.init(0x0de7);
    const r = prng.random();
    var col_ptr: [65]u32 = undefined;
    var row_idx: [64 * 64]u32 = undefined;
    var q: [64]u32 = undefined;
    var buf: [wsSize(64, 64 * 64)]u32 = undefined;
    for (0..300) |t| {
        const n: u32 = if (t < 64) @intCast(t + 1) else r.intRangeAtMost(u32, 1, 64);
        // Full diagonal, as `order` requires. Circuit-like sparsity, or
        // dense.
        const density = if (t % 4 == 0) 0.3 + 0.6 * r.float(f64) else 0.05 * r.float(f64);
        var nnz: u32 = 0;
        col_ptr[0] = 0;
        for (0..n) |j| {
            for (0..n) |i| {
                if (i != j and r.float(f64) >= density) continue;
                row_idx[nnz] = @intCast(i);
                nnz += 1;
            }
            col_ptr[j + 1] = nnz;
        }
        inline for (.{ order, amd }) |f| {
            var ws = Ws.init(buf[0..wsSize(n, nnz)]);
            try f(n, col_ptr[0 .. n + 1], row_idx[0..nnz], q[0..n], &ws);
            try std.testing.expectEqual(@as(usize, 0), ws.used);
            var seen: [64]bool = @splat(false);
            for (q[0..n]) |c| {
                try std.testing.expect(c < n and !seen[c]);
                seen[c] = true;
            }
        }
    }
}

test "cholCount under a cap: the full count below it, at least the cap otherwise" {
    var prng = std.Random.DefaultPrng.init(0x0de9);
    const r = prng.random();
    const n = 40;
    var col_ptr: [n + 1]u32 = undefined;
    var row_idx: [n * n]u32 = undefined;
    var perm: [n]u32 = undefined;
    var buf: [wsSize(n, n * n)]u32 = undefined;
    for (0..50) |_| {
        var nnz: u32 = 0;
        col_ptr[0] = 0;
        for (0..n) |j| {
            for (0..n) |i| {
                if (i != j and r.float(f64) >= 0.08) continue;
                row_idx[nnz] = @intCast(i);
                nnz += 1;
            }
            col_ptr[j + 1] = nnz;
        }
        for (&perm, 0..) |*c, i| c.* = @intCast(i);
        r.shuffle(u32, &perm);
        var ws = Ws.init(buf[0..wsSize(n, nnz)]);
        const full = try cholCount(col_ptr[0 .. n + 1], row_idx[0..nnz], &perm, &ws, std.math.maxInt(u64));
        for ([_]u64{ n, full - 1, full, full + 1 }) |cap| {
            const got = try cholCount(col_ptr[0 .. n + 1], row_idx[0..nnz], &perm, &ws, cap);
            if (full < cap) try std.testing.expectEqual(full, got) else try std.testing.expect(got >= cap);
        }
        try std.testing.expectEqual(@as(usize, 0), ws.used);
    }
}

test "order and amd on every slab too short fail with OutOfWorkspace, restore the workspace, and never write past it" {
    // The analogue of an allocation-failure sweep for the slab allocator:
    // each length from 0 up either fails cleanly or returns the same
    // order as the full slab.
    var prng = std.Random.DefaultPrng.init(0x0de8);
    const r = prng.random();
    const n = 24;
    var col_ptr: [n + 1]u32 = undefined;
    var row_idx: [n * n]u32 = undefined;
    var nnz: u32 = 0;
    col_ptr[0] = 0;
    for (0..n) |j| {
        for (0..n) |i| {
            if (i != j and r.float(f64) >= 0.2) continue;
            row_idx[nnz] = @intCast(i);
            nnz += 1;
        }
        col_ptr[j + 1] = nnz;
    }
    const canary: u32 = 0xdead_beef;
    var buf: [wsSize(n, n * n) + 1]u32 = undefined;
    inline for (.{ order, amd }) |f| {
        const full = wsSize(n, nnz);
        var want: [n]u32 = undefined;
        var ws = Ws.init(buf[0..full]);
        try f(n, &col_ptr, row_idx[0..nnz], &want, &ws);
        var passed = false;
        for (0..full + 1) |len| {
            buf[len] = canary;
            var q: [n]u32 = undefined;
            ws = Ws.init(buf[0..len]);
            if (f(n, &col_ptr, row_idx[0..nnz], &q, &ws)) |_| {
                try std.testing.expectEqualSlices(u32, &want, &q);
                passed = true;
            } else |err| try std.testing.expectEqual(error.OutOfWorkspace, err);
            try std.testing.expectEqual(@as(usize, 0), ws.used);
            try std.testing.expectEqual(canary, buf[len]);
        }
        try std.testing.expect(passed);
    }
}
