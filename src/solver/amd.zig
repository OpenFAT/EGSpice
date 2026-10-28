//! Approximate minimum degree on the symmetrized pattern of one block:
//! a CSC pattern in, the elimination order q (q[k] = column eliminated at
//! step k) out, all scratch from the caller's `Ws` slab.
//!
//! Quotient graph with approximate external degrees, aggressive absorption,
//! hashed supervariable detection and dense-row deferral (rows of degree
//! >= max(16, 10 sqrt(n)) go last).
//! Cites: Amestoy, Davis and Duff, "An approximate minimum degree ordering
//! algorithm", SIAM J. Matrix Anal. Appl. 17(4), 1996; dev/solvers/amd-ordering.md.
//!
//! Contents:
//! - `pub fn amd(n: u32, col_ptr: []const u32, row_idx: []const u32, q: []u32, ws: *Ws) Error!void`

const std = @import("std");
const assert = std.debug.assert;
const order_ws = @import("order_ws.zig");
const Ws = order_ws.Ws;
const Error = order_ws.Error;

const NONE: u32 = std.math.maxInt(u32);

// Tags a span's first word in place of its value while `compact` sweeps.
const span_tag: u32 = 1 << 31;

// Doubly-linked degree buckets: O(1) insert and remove.
const DegLists = struct {
    head: []u32, // first variable of each degree, or NONE
    next: []u32,
    prev: []u32,

    fn insert(s: *DegLists, d: u32, i: u32) void {
        assert(s.head[d] != i);
        s.next[i] = s.head[d];
        s.prev[i] = NONE;
        if (s.head[d] != NONE) s.prev[s.head[d]] = i;
        s.head[d] = i;
    }

    fn remove(s: *DegLists, d: u32, i: u32) void {
        if (s.prev[i] == NONE) assert(s.head[d] == i);
        if (s.prev[i] != NONE)
            s.next[s.prev[i]] = s.next[i]
        else
            s.head[d] = s.next[i];
        if (s.next[i] != NONE)
            s.prev[s.next[i]] = s.prev[i];
    }
};

/// Writes q[k] = the column eliminated at step k of AMD on the pattern's
/// A + A^T. Restores `ws` on return; `OutOfWorkspace` when the slab cannot
/// hold the quotient graph's growth. Asserts that `q` has n entries,
/// `col_ptr` n + 1, and that n is below 2^31.
pub fn amd(n: u32, col_ptr: []const u32, row_idx: []const u32, q: []u32, ws: *Ws) Error!void {
    if (n == 0) return;
    assert(q.len == n);
    assert(col_ptr.len == @as(usize, n) + 1);
    assert(n < span_tag); // `compact` tags with the top bit
    const base = ws.mark();
    defer ws.release(base);

    const nnz = col_ptr[n];

    // Variable adjacency: packed spans in `va`; eliminated elements append
    // their member lists at its tail.
    const va = try ws.alloc(4 * @as(usize, nnz) + 4 * @as(usize, n) + 16);
    const va_pe = try ws.alloc(n);
    const va_len = try ws.alloc(n);

    // Element adjacency per variable, bump-allocated with 2x growth.
    const ea = try ws.alloc(8 * @as(usize, n) + @as(usize, nnz));
    const ea_pe = try ws.alloc(n);
    const ea_len = try ws.allocSet(n, 0);
    const ea_lim = try ws.alloc(n);

    const nv = try ws.allocSet(n, 1); // supervariable count (0 = dead/absorbed)
    const elem_alive = try ws.allocSet(n, 0); // 1 = live element
    const esize = try ws.alloc(n); // esize[e] = Σnv over Le members
    const deg = try ws.alloc(n); // approximate external degree
    const w_buf = try ws.alloc(n); // |Le \ Lp| bound, per pivot step

    // Epoch marks, so no clearing between steps.
    const mark_arr = try ws.allocSet(n, 0);
    const wmark_arr = try ws.allocSet(n, 0);
    var era: u32 = 0; // epoch for mark_arr
    var erw: u32 = 0; // epoch for wmark_arr

    const hkey = try ws.alloc(n);
    const hhead = try ws.allocSet(n, NONE);
    const hnext = try ws.alloc(n);
    const hbuckets = try ws.alloc(n); // buckets in use, for reset
    const mlink = try ws.allocSet(n, NONE); // supervariable merge chains
    const lp_buf = try ws.alloc(n);
    const touched = try ws.alloc(n);
    const first = try ws.alloc(n); // `compact` scratch

    var dl = DegLists{
        .head = try ws.allocSet(n + 1, NONE),
        .next = try ws.alloc(n),
        .prev = try ws.alloc(n),
    };

    // Symmetrized adjacency: count, place spans, scatter, dedup.
    @memset(deg, 0);
    for (0..n) |j| {
        for (col_ptr[j]..col_ptr[j + 1]) |pi| {
            const r = row_idx[pi];
            if (r == j) continue;
            deg[j] += 1;
            deg[r] += 1;
        }
    }
    var va_free: usize = 0;
    for (0..n) |i| {
        va_pe[i] = @intCast(va_free);
        va_free += deg[i];
    }
    @memset(deg, 0);
    for (0..n) |j| {
        for (col_ptr[j]..col_ptr[j + 1]) |pi| {
            const r = row_idx[pi];
            if (r == j) continue;
            va[va_pe[j] + deg[j]] = r;
            deg[j] += 1;
            va[va_pe[r] + deg[r]] = @intCast(j);
            deg[r] += 1;
        }
    }
    @memcpy(va_len, deg);
    for (0..n) |i| {
        era += 1;
        var wp: u32 = 0;
        const s = va_pe[i];
        for (0..va_len[i]) |ki| {
            const v = va[s + ki];
            if (mark_arr[v] != era) {
                mark_arr[v] = era;
                va[s + wp] = v;
                wp += 1;
            }
        }
        va_len[i] = wp;
    }

    // Dense rows leave the quotient graph and are ordered last.
    const isq: u32 = std.math.sqrt(n);
    const dense_thresh: u32 = @max(@as(u32, 16), 10 * isq);
    var ndense: u32 = 0;
    for (va_len) |l| ndense += @intFromBool(l >= dense_thresh);
    const dense_start = n - ndense;
    {
        var di: u32 = 0;
        for (0..n) |i| {
            if (va_len[i] >= dense_thresh) {
                nv[i] = 0;
                lp_buf[dense_start + di] = @intCast(i);
                di += 1;
            }
        }
    }
    if (ndense > 0) {
        for (0..n) |i| {
            if (nv[i] == 0) continue;
            var wp: u32 = 0;
            const s = va_pe[i];
            for (0..va_len[i]) |ki| {
                const v = va[s + ki];
                if (nv[v] != 0) {
                    va[s + wp] = v;
                    wp += 1;
                }
            }
            va_len[i] = wp;
        }
    }

    var ea_free: usize = 0;
    for (0..n) |i| {
        ea_pe[i] = @intCast(ea_free);
        ea_lim[i] = 4;
        ea_free += 4;
    }

    for (0..n) |i| {
        if (nv[i] == 0) continue;
        deg[i] = va_len[i];
        dl.insert(deg[i], @intCast(i));
    }

    const n_amd: u32 = n - ndense;
    var mindeg: u32 = 0;
    var k: u32 = 0;

    // Terminates: each pass eliminates p, so k grows by at least one.
    while (k < n_amd) {
        // Some live variable remains, so a bucket at or above mindeg holds it.
        while (dl.head[mindeg] == NONE) {
            mindeg += 1;
            assert(mindeg <= n);
        }
        const p = dl.head[mindeg];
        dl.remove(deg[p], p);

        // Lp = (A_p union every Le for e in E_p) minus p.
        era += 1;
        mark_arr[p] = era;
        var nlp: u32 = 0;

        for (va[va_pe[p]..][0..va_len[p]]) |v| {
            if (nv[v] != 0 and mark_arr[v] != era) {
                mark_arr[v] = era;
                lp_buf[nlp] = v;
                nlp += 1;
            }
        }
        for (ea[ea_pe[p]..][0..ea_len[p]]) |e| {
            if (elem_alive[e] == 0) continue;
            for (va[va_pe[e]..][0..va_len[e]]) |v| {
                if (nv[v] != 0 and mark_arr[v] != era) {
                    mark_arr[v] = era;
                    lp_buf[nlp] = v;
                    nlp += 1;
                }
            }
            elem_alive[e] = 0; // absorbed into p
        }

        // p becomes an element whose member list is Lp.
        const nvpiv = nv[p];
        nv[p] = 0;
        elem_alive[p] = 1;

        if (va_free + nlp > va.len) {
            va_len[p] = 0; // p's variable list is spent: Lp is in lp_buf
            va_free = compact(va, va_free, va_pe, va_len, nv, elem_alive, first);
            if (va_free + nlp > va.len) return error.OutOfWorkspace;
        }
        va_pe[p] = @intCast(va_free);
        va_len[p] = nlp;
        @memcpy(va[va_free..][0..nlp], lp_buf[0..nlp]);
        va_free += nlp;

        var lpsize: u32 = 0;
        for (lp_buf[0..nlp]) |i| lpsize += nv[i];
        esize[p] = lpsize;

        for (lp_buf[0..nlp]) |i| dl.remove(deg[i], i);

        // w[e] = |Le minus Lp|, in one scan of Lp's element lists.
        erw += 1;
        var ntouched: u32 = 0;
        for (lp_buf[0..nlp]) |i| {
            for (ea[ea_pe[i]..][0..ea_len[i]]) |e| {
                if (elem_alive[e] == 0) continue;
                if (wmark_arr[e] != erw) {
                    wmark_arr[e] = erw;
                    w_buf[e] = esize[e];
                    touched[ntouched] = e;
                    ntouched += 1;
                }
                w_buf[e] -= @min(w_buf[e], nv[i]);
            }
        }
        // Aggressive absorption.
        for (touched[0..ntouched]) |e| {
            if (w_buf[e] == 0) elem_alive[e] = 0;
        }
        // Per member: prune, then the approximate external degree.
        for (lp_buf[0..nlp]) |i| {
            // E_i = live(E_i) + p, esum = sum of w[e].
            var esum: u32 = 0;
            var ewp: u32 = 0;
            for (ea[ea_pe[i]..][0..ea_len[i]]) |e| {
                if (elem_alive[e] != 0 and e != p) {
                    ea[ea_pe[i] + ewp] = e;
                    ewp += 1;
                    esum += w_buf[e];
                }
            }
            if (ewp < ea_lim[i]) {
                ea[ea_pe[i] + ewp] = p;
            } else {
                const nc: u32 = (ewp + 1) * 2;
                if (ea_free + nc > ea.len) {
                    ea_len[i] = ewp;
                    ea_free = compact(ea, ea_free, ea_pe, ea_len, nv, nv, first);
                    // Compacted spans have no slack left.
                    for (0..n) |o| ea_lim[o] = ea_len[o];
                    if (ea_free + nc > ea.len) return error.OutOfWorkspace;
                }
                @memcpy(ea[ea_free..][0..ewp], ea[ea_pe[i]..][0..ewp]);
                ea[ea_free + ewp] = p;
                ea_pe[i] = @intCast(ea_free);
                ea_lim[i] = nc;
                ea_free += nc;
            }
            ea_len[i] = ewp + 1;

            // A_i = A_i minus Lp and dead variables, asum = sum of nv.
            var asum: u32 = 0;
            var vwp: u32 = 0;
            for (va[va_pe[i]..][0..va_len[i]]) |v| {
                if (nv[v] != 0 and mark_arr[v] != era) {
                    va[va_pe[i] + vwp] = v;
                    vwp += 1;
                    asum += nv[v];
                }
            }
            va_len[i] = vwp;

            const cap: u32 = n_amd - k - nvpiv;
            deg[i] = @min(asum + (lpsize - nv[i]) + esum, cap);
        }

        // Supervariables: hash, then compare within each bucket.
        var nhb: u32 = 0;
        for (lp_buf[0..nlp]) |i| {
            if (nv[i] == 0) continue;
            var h: u32 = 0;
            for (va[va_pe[i]..][0..va_len[i]]) |v| h +%= v;
            for (ea[ea_pe[i]..][0..ea_len[i]]) |v| h +%= v;
            hkey[i] = h;
            const b = h % n;
            if (hhead[b] == NONE) {
                hbuckets[nhb] = b;
                nhb += 1;
            }
            hnext[i] = hhead[b];
            hhead[b] = i;
        }
        for (hbuckets[0..nhb]) |b| {
            var i = hhead[b];
            while (i != NONE) : (i = hnext[i]) {
                if (nv[i] == 0) continue;
                var j = hnext[i];
                while (j != NONE) : (j = hnext[j]) {
                    if (nv[j] == 0 or hkey[j] != hkey[i]) continue;
                    if (!sameAdj(va, va_pe, va_len, ea, ea_pe, ea_len, mark_arr, &era, i, j))
                        continue;
                    deg[i] -= @min(deg[i], nv[j]);
                    nv[i] += nv[j];
                    nv[j] = 0;
                    var tail = j;
                    // Merge chains are acyclic: each variable joins one once.
                    while (mlink[tail] != NONE) tail = mlink[tail];
                    mlink[tail] = mlink[i];
                    mlink[i] = j;
                }
            }
            hhead[b] = NONE;
        }

        for (lp_buf[0..nlp]) |i| {
            if (nv[i] == 0) continue;
            dl.insert(deg[i], i);
            if (deg[i] < mindeg) mindeg = deg[i];
        }

        // Mass elimination: p and every variable merged into it.
        q[k] = p;
        k += 1;
        var mb = mlink[p];
        while (mb != NONE) : (mb = mlink[mb]) {
            q[k] = mb;
            k += 1;
        }
    }

    assert(k == n_amd);
    @memcpy(q[k..][0..ndense], lp_buf[dense_start..][0..ndense]);
}

// ---- private ----

// Slides every live span of `adj` below `top` to the front, keeping their
// order, and returns the new top. Owner o's span is `adj[pe[o]..][0..len[o]]`,
// live when `live_a[o] != 0` or `live_b[o] != 0`. Each live span's first
// word is parked in `first` and its slot tagged with o, so one forward
// sweep finds every span start among the dead words. O(top + n).
fn compact(adj: []u32, top: usize, pe: []u32, len: []const u32, live_a: []const u32, live_b: []const u32, first: []u32) usize {
    for (pe, len, 0..) |at, l, o| {
        if (l == 0 or (live_a[o] == 0 and live_b[o] == 0)) continue;
        first[o] = adj[at];
        adj[at] = span_tag | @as(u32, @intCast(o));
    }
    var dst: usize = 0;
    var src: usize = 0;
    while (src < top) {
        if (adj[src] & span_tag == 0) {
            src += 1; // dead word
            continue;
        }
        const o = adj[src] & ~span_tag;
        const l = len[o];
        adj[dst] = first[o];
        @memmove(adj[dst + 1 ..][0 .. l - 1], adj[src + 1 ..][0 .. l - 1]);
        pe[o] = @intCast(dst);
        dst += l;
        src += l;
    }
    assert(dst <= top);
    return dst;
}

// True when i and j have equal variable and element adjacency sets (exact
// equality: Lp members are already pruned from both).
fn sameAdj(
    va: []const u32,
    va_pe: []const u32,
    va_len: []const u32,
    ea: []const u32,
    ea_pe: []const u32,
    ea_len: []const u32,
    mark_buf: []u32,
    era: *u32,
    i: u32,
    j: u32,
) bool {
    assert(i != j);
    if (va_len[i] != va_len[j] or ea_len[i] != ea_len[j]) return false;
    inline for (.{ va, ea }, .{ va_pe, ea_pe }, .{ va_len, ea_len }) |adj, pos, len| {
        era.* += 1;
        const m = era.*;
        for (adj[pos[i]..][0..len[i]]) |v| mark_buf[v] = m;
        for (adj[pos[j]..][0..len[j]]) |v| {
            if (mark_buf[v] != m) return false;
        }
    }
    return true;
}
