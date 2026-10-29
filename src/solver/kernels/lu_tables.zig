//! The device LU's tables, built on the host from a factored `SparseLu`:
//! its L/U pattern, pivot sequence and the source CSC in, one u32 index
//! buffer with every table the kernels read (`Tab` offsets) out. Picks per
//! column between the stored-order and gather forms and the solves' tail
//! cut by a cost model over the factor's structure.
//! Cites: dev/solvers/gpu-lu.md §4 (the tables, the tail panels, E2).
//!
//! Contents:
//! - `pub const Options = struct`
//! - `pub const Tables = struct`
//!   - `pub fn deinit(self: *Tables, gpa: std.mem.Allocator) void`
//! - `pub fn build(gpa: std.mem.Allocator, lu: *const SparseLu, col_ptr: []const u32, growth: f64, opt: Options) !Tables`

const std = @import("std");
const assert = std.debug.assert;
const klu = @import("lu.zig");
const SparseLu = @import("../sparse_lu.zig").SparseLu;

/// Overrides of `build`'s own choices, for tests and E2 calibration.
pub const Options = struct {
    /// The solves' tail cut t0.
    tail: ?u32 = null,
    /// False keeps every column in stored order.
    gather: bool = true,
    /// False keeps `dmap` at u32 even where u16 fits.
    dmap16: bool = true,
};

/// The device tables of one pivot epoch.
pub const Tables = struct {
    tab: klu.Tab,
    /// Every u32 table, at the offsets in `tab`. Owned; free with `deinit`.
    idx: []u32,

    /// Frees `idx`; `self` is undefined afterwards.
    pub fn deinit(self: *Tables, gpa: std.mem.Allocator) void {
        gpa.free(self.idx);
        self.* = undefined;
    }
};

/// Flop-latency units per barrier (plus the done wait) in the per-column
/// cost that picks the gather form: a U entry of the stored-order form
/// costs `step_cost + ceil(len / lanes)`, a gather level `level_cost` plus
/// its longest contribution chain. ponytail: fixed guesses; fit them from
/// E2's per-column timings if the choice matters.
const step_cost = 4;

const level_cost = 2;

/// Columns with fewer U entries skip the gather analysis.
const gather_min_u = 32;

/// Solve-cost units (one dependent load or flop on the device) per head
/// level (a barrier and the index loads before the first y load) and per
/// tail column (its share of a 64-row panel's handoff). Fit on an RTX 4060
/// Laptop against the measured best cut (`ESPICE_GPU_LU_TAIL`): the cut
/// this picks is the fastest of 512-5120 on logic_bsim4_2k/3k/10k,
/// sram_bsim4_10k/30k and within 2% on logic_psp103_10k; the one-block
/// sweep's 3 per column kept tails half as long, with solves 10-34% slower
/// (sram_bsim4_30k 18.2 against 12.0 ms).
const head_level_cost = 4;

const tail_step_cost = 1;

/// Builds the tables for `lu`'s current factor. O(F + nnz(L+U)) time; the
/// narrow columns' `dmap` is 2 bytes per flop when every column has fewer
/// than 65,535 slots, else 4. Asserts a successful
/// `factor` and a tail of at most `tail_max` steps. `OutOfMemory` also
/// means the tables passed u32 offsets. Caller owns the result; free with
/// `Tables.deinit`. A later `factor` of `lu` (new `pattern_epoch`) makes
/// them stale.
pub fn build(gpa: std.mem.Allocator, lu: *const SparseLu, col_ptr: []const u32, growth: f64, opt: Options) !Tables {
    const lanes = klu.refactor_block;
    assert(lu.factored);
    assert(col_ptr.len == @as(usize, lu.n) + 1);
    const n = lu.n;
    const lp = lu.lp;
    const up = lu.up;
    const li = lu.li.items;
    const ui = lu.ui.items;
    const nu: u32 = @intCast(ui.len);
    const nnz = col_ptr[n];

    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const ar = arena_state.allocator();

    const coff = try ar.alloc(u32, n + 1);
    for (0..n + 1) |k| coff[k] = up[k] + lp[k] + @as(u32, @intCast(k));
    const n_val = coff[n] + 1;
    const discard = coff[n];

    const ul0 = try ar.alloc(u32, nu);
    const ulen = try ar.alloc(u32, nu);
    for (ui, ul0, ulen) |i, *l0, *ln| {
        l0.* = coff[i] + up[i + 1] - up[i] + 1;
        ln.* = lp[i + 1] - lp[i];
    }
    const flags = try ar.alloc(u32, n);
    for (flags, 0..) |*f, k| f.* = (if (lu.void_col[k]) klu.void_bit else 0) | (if (lu.scaled_pivot[k]) klu.scaled_bit else 0);

    // Row -> slot of the current column, as in `buildTape`.
    const pos = try ar.alloc(u32, n);
    const amap = try ar.alloc(u32, nnz);
    const dcol = try ar.alloc(u32, n + 1);
    const wlev = try ar.alloc(u32, n + 1);
    // `dmap` is most of the tables (4 bytes per flop), so it is built in
    // place at offset 0 of `idx`, never copied: reserved at F, its bound.
    var flops: usize = 0;
    for (ulen) |l| flops += l;
    // u16 column-relative slots when every column fits: `dmap` is the
    // largest table and, with the L values, the refactor's streamed bytes.
    // ponytail: one width per epoch; a column of 65,535 slots or more (a
    // real deck's supply rail) keeps the whole epoch at u32. Split the
    // columns into a u16 and a u32 list if such a deck ever has a thick factor.
    var widest: u32 = 0;
    for (0..n) |k| widest = @max(widest, coff[k + 1] - coff[k]);
    const dmap16 = opt.dmap16 and widest < klu.narrow_discard;
    var dmap: std.ArrayList(u32) = .empty;
    errdefer dmap.deinit(gpa);
    try dmap.ensureTotalCapacityPrecise(gpa, if (dmap16) (flops + 1) / 2 else flops);
    // Entries written; the words behind them are `dmap`'s capacity.
    var nd: usize = 0;
    const d16: [*]u16 = @ptrCast(dmap.items.ptr);
    const d32: [*]u32 = dmap.items.ptr;
    var lev_ptr: std.ArrayList(u32) = .empty;
    try lev_ptr.append(ar, 0);
    var g_slot: std.ArrayList(u32) = .empty;
    var g_cptr: std.ArrayList(u32) = .empty;
    try g_cptr.append(ar, 0);
    var g_l: std.ArrayList(u32) = .empty;
    var g_u: std.ArrayList(u32) = .empty;
    // Gather scratch, per column slot (relative to coff[k]).
    const lvl = try ar.alloc(u32, n_val);
    const cnt = try ar.alloc(u32, n_val);
    for (0..n) |kk| {
        const k: u32 = @intCast(kk);
        const void_k = lu.void_col[k];
        const base = coff[k];
        const nuk = up[k + 1] - up[k];
        for (up[k]..up[k + 1]) |p| pos[ui[p]] = base + @as(u32, @intCast(p)) - up[k];
        pos[k] = base + nuk;
        for (lp[k]..lp[k + 1]) |q| pos[li[q]] = base + nuk + 1 + @as(u32, @intCast(q)) - lp[k];
        const c = lu.q[k];
        for (col_ptr[c]..col_ptr[c + 1]) |e| {
            const r = lu.prow[e];
            amap[e] = if (void_k and r > k) discard else pos[r];
        }
        dcol[k] = @intCast(nd);
        wlev[k] = @intCast(lev_ptr.items.len - 1);

        var col_flops: u64 = 0;
        var pseq: u64 = 0;
        for (up[k]..up[k + 1]) |p| {
            const len = lp[ui[p] + 1] - lp[ui[p]];
            col_flops += len;
            pseq += step_cost + (len + lanes - 1) / lanes;
        }

        // Only a column whose U entries average under half a block of L
        // rows can gain from the gather form (a stored-order step leaves
        // lanes idle): the analysis walks every flop of the column again,
        // which on a dense tail (logic_bsim4_10k) doubled the table build.
        const sparse_steps = col_flops * 2 < @as(u64, nuk) * lanes;
        const gathered = opt.gather and nuk >= gather_min_u and sparse_steps and try gatherColumn(ar, lu, k, base, coff[k + 1] - base, pos, lvl, cnt, pseq, &lev_ptr, &g_slot, &g_cptr, &g_l, &g_u);
        if (!gathered) {
            assert(nd + col_flops <= flops);
            for (up[k]..up[k + 1]) |p| {
                const i = ui[p];
                const rows = li[lp[i]..lp[i + 1]];
                if (dmap16) {
                    for (rows, d16[nd..][0..rows.len]) |r, *o| o.* = if (void_k and r > k) klu.narrow_discard else @intCast(pos[r] - base);
                } else for (rows, d32[nd..][0..rows.len]) |r, *o| {
                    o.* = if (void_k and r > k) discard else pos[r];
                }
                nd += rows.len;
            }
        }
    }
    dcol[n] = @intCast(nd);
    wlev[n] = @intCast(lev_ptr.items.len - 1);
    // An odd u16 count leaves half a word: zero it, so the tables are a
    // function of the factor.
    if (dmap16 and nd % 2 == 1) d16[nd] = 0;
    const dwords = if (dmap16) (nd + 1) / 2 else nd;
    dmap.items.len = dwords;

    // Solve row lists: L by rows with k ascending, U by rows with k
    // descending (the host's per-row orders).
    const perm_in = try ar.alloc(u32, n);
    for (lu.pinv, 0..) |r, o| perm_in[r] = @intCast(o);
    const u_ptr = try ar.alloc(u32, n + 1);
    const u_col = try ar.alloc(u32, nu);
    const u_slot = try ar.alloc(u32, nu);
    @memset(u_ptr, 0);
    for (ui) |i| u_ptr[i + 1] += 1;
    for (0..n) |i| u_ptr[i + 1] += u_ptr[i];
    @memset(pos, 0);
    var kd = n;
    while (kd > 0) {
        kd -= 1;
        for (up[kd]..up[kd + 1]) |p| {
            const i = ui[p];
            const at = u_ptr[i] + pos[i];
            pos[i] += 1;
            u_col[at] = kd;
            u_slot[at] = coff[kd] + @as(u32, @intCast(p - up[kd]));
        }
    }

    // The tail cut: the cheapest of t0 = n - m for m = 0, then 16, 24, 32,
    // 48, ... (powers of two and the halfway steps), ending at
    // min(n, tail_max).
    const level = try ar.alloc(u32, n);
    const len = try ar.alloc(u32, n);
    const scratch = try ar.alloc(u32, 2 * @as(usize, n) + 2);
    var t0 = n;
    if (opt.tail) |cut| t0 = cut else {
        const top = @min(n, klu.tail_max);
        var best = solveCost(lu, n, level, len, scratch);
        var m: u32 = 16;
        while (true) : (m = if (m & (m - 1) == 0) m + m / 2 else m / 3 * 4) {
            const cut = @min(m, top);
            const c = solveCost(lu, n - cut, level, len, scratch);
            if (c < best) {
                best = c;
                t0 = n - cut;
            }
            if (m >= top) break;
        }
    }
    assert(t0 <= n);
    assert(n - t0 <= klu.tail_max);
    const l_levels = headLevels(lu, t0, false, level, len);
    const l_sched = try bucket(ar, level[0..n], l_levels);
    const u_levels = headLevels(lu, t0, true, level, len);
    const u_sched = try bucket(ar, level[0..t0], u_levels);
    const l_chunk = try runs(ar, l_sched.lev);
    const u_chunk = try runs(ar, u_sched.lev);
    // L rows keep their head entries only (k < t0); the sweep does the rest.
    const nl_head = lp[t0];
    const l_ptr = try ar.alloc(u32, n + 1);
    const l_col = try ar.alloc(u32, nl_head);
    const l_slot = try ar.alloc(u32, nl_head);
    @memset(l_ptr, 0);
    for (li[0..nl_head]) |r| l_ptr[r + 1] += 1;
    for (0..n) |r| l_ptr[r + 1] += l_ptr[r];
    @memset(pos, 0);
    for (0..t0) |k| for (lp[k]..lp[k + 1]) |q| {
        const r = li[q];
        const at = l_ptr[r] + pos[r];
        pos[r] += 1;
        l_col[at] = @intCast(k);
        l_slot[at] = coff[k] + up[k + 1] - up[k] + 1 + @as(u32, @intCast(q - lp[k]));
    };
    const tn = n - t0;
    const t_diag = try ar.alloc(u32, tn);
    for (t0..n, 0..) |k, c| t_diag[c] = coff[k] + up[k + 1] - up[k];

    // Tail rows for the panel solves: L rows ascending by column (columns
    // visited ascending), U rows descending (columns visited descending).
    const tl_ptr = try ar.alloc(u32, tn + 1);
    const tu_ptr = try ar.alloc(u32, tn + 1);
    @memset(tl_ptr, 0);
    @memset(tu_ptr, 0);
    for (t0..n) |k| {
        for (li[lp[k]..lp[k + 1]]) |r| tl_ptr[r - t0 + 1] += 1;
        for (ui[up[k]..up[k + 1]]) |i| if (i >= t0) {
            tu_ptr[i - t0 + 1] += 1;
        };
    }
    for (0..tn) |i| {
        tl_ptr[i + 1] += tl_ptr[i];
        tu_ptr[i + 1] += tu_ptr[i];
    }
    const tl_col = try ar.alloc(u32, tl_ptr[tn]);
    const tl_slot = try ar.alloc(u32, tl_ptr[tn]);
    const tu_col = try ar.alloc(u32, tu_ptr[tn]);
    const tu_slot = try ar.alloc(u32, tu_ptr[tn]);
    @memset(len[0..tn], 0);
    for (t0..n) |k| for (lp[k]..lp[k + 1]) |q| {
        const r = li[q] - t0;
        const at = tl_ptr[r] + len[r];
        len[r] += 1;
        tl_col[at] = @intCast(k - t0);
        tl_slot[at] = coff[k] + up[k + 1] - up[k] + 1 + @as(u32, @intCast(q - lp[k]));
    };
    @memset(len[0..tn], 0);
    var kt = n;
    while (kt > t0) {
        kt -= 1;
        for (up[kt]..up[kt + 1]) |p| {
            const i = ui[p];
            if (i < t0) continue;
            const r = i - t0;
            const at = tu_ptr[r] + len[r];
            len[r] += 1;
            tu_col[at] = kt - t0;
            tu_slot[at] = coff[kt] + @as(u32, @intCast(p - up[kt]));
        }
    }

    // Concatenate, each table at the Tab field of its name.
    const parts = .{
        .{ "dmap", dmap.items[0..dwords] }, .{ "coff", coff },                   .{ "up", up },               .{ "lp", lp },
        .{ "ui", ui },                      .{ "ul0", ul0 },                     .{ "ulen", ulen },           .{ "flags", flags },
        .{ "q", lu.q },                     .{ "col_ptr", col_ptr[0 .. n + 1] }, .{ "amap", amap },           .{ "dcol", dcol },
        .{ "wlev", wlev },                  .{ "lev_ptr", lev_ptr.items },       .{ "g_slot", g_slot.items }, .{ "g_cptr", g_cptr.items },
        .{ "g_l", g_l.items },              .{ "g_u", g_u.items },               .{ "li", li },               .{ "t_diag", t_diag },
        .{ "l_lev", l_sched.lev },          .{ "l_rows", l_sched.rows },         .{ "l_ptr", l_ptr },         .{ "l_col", l_col },
        .{ "l_slot", l_slot },              .{ "perm_in", perm_in },             .{ "u_lev", u_sched.lev },   .{ "u_rows", u_sched.rows },
        .{ "u_ptr", u_ptr },                .{ "u_col", u_col },                 .{ "u_slot", u_slot },       .{ "l_chunk", l_chunk },
        .{ "u_chunk", u_chunk },            .{ "tl_ptr", tl_ptr },               .{ "tl_col", tl_col },       .{ "tl_slot", tl_slot },
        .{ "tu_ptr", tu_ptr },              .{ "tu_col", tu_col },               .{ "tu_slot", tu_slot },
    };
    var tab: klu.Tab = undefined;
    tab.n = n;
    tab.discard = discard;
    tab.growth = growth;
    tab.t0 = t0;
    tab.l_levels = l_levels;
    tab.u_levels = u_levels;
    tab.l_chunks = @intCast(l_chunk.len - 1);
    tab.u_chunks = @intCast(u_chunk.len - 1);
    tab.n_val = n_val;
    tab.dmap16 = @intFromBool(dmap16);
    var total: usize = 0;
    inline for (parts) |part| {
        @field(tab, part[0]) = std.math.cast(u32, total) orelse return error.OutOfMemory;
        total += part[1].len;
    }
    tab.n_idx = std.math.cast(u32, total) orelse return error.OutOfMemory;
    assert(tab.dmap == 0);
    try dmap.ensureTotalCapacityPrecise(gpa, total);
    dmap.items.len = total;
    inline for (parts, 0..) |part, i| {
        if (i != 0) @memcpy(dmap.items[@field(tab, part[0])..][0..part[1].len], part[1]);
    }
    const idx = try dmap.toOwnedSlice(gpa);
    assert(idx.len == tab.n_idx);
    assert(tab.coff == dwords);
    return .{ .idx = idx, .tab = tab };
}

// ---- private ----

/// Levels of the head schedule for tail cut t0 into `level` (rows below
/// t0 for U, all rows for L, whose tail rows run their head part) and the
/// head entries per row into `len`. Returns the level count.
fn headLevels(lu: *const SparseLu, t0: u32, comptime upper: bool, level: []u32, len: []u32) u32 {
    assert(t0 <= lu.n);
    const rows: u32 = if (upper) t0 else lu.n;
    assert(level.len >= rows);
    assert(len.len >= rows);
    @memset(level[0..rows], 0);
    @memset(len[0..rows], 0);
    var levels: u32 = @intFromBool(rows > 0);
    if (upper) {
        var k = t0;
        while (k > 0) {
            k -= 1;
            for (lu.ui.items[lu.up[k]..lu.up[k + 1]]) |i| {
                level[i] = @max(level[i], level[k] + 1);
                len[i] += 1;
                levels = @max(levels, level[i] + 1);
            }
        }
    } else for (0..t0) |k| for (lu.li.items[lu.lp[k]..lu.lp[k + 1]]) |r| {
        level[r] = @max(level[r], level[k] + 1);
        len[r] += 1;
        levels = @max(levels, level[r] + 1);
    };
    // A level holds at least one row.
    assert(levels <= rows);
    return levels;
}

/// Estimated span of both solves for tail cut t0: per head level its
/// barrier, its longest row and its rows over the lanes; per tail column
/// its step and its entries over the lanes.
fn solveCost(lu: *const SparseLu, t0: u32, level: []u32, len: []u32, scratch: []u32) u64 {
    const lanes = klu.solve_block;
    var cost: u64 = 0;
    inline for (.{ false, true }) |upper| {
        const levels = headLevels(lu, t0, upper, level, len);
        assert(2 * levels <= scratch.len);
        const width = scratch[0..levels];
        const chain = scratch[levels..][0..levels];
        @memset(width, 0);
        @memset(chain, 0);
        const rows: u32 = if (upper) t0 else lu.n;
        for (level[0..rows], len[0..rows]) |v, l| {
            width[v] += 1;
            chain[v] = @max(chain[v], l);
        }
        for (width, chain) |w, c| cost += head_level_cost + c + (w + lanes - 1) / lanes;
        for (t0..lu.n) |k| {
            const cnt = if (upper) lu.up[k + 1] - lu.up[k] else lu.lp[k + 1] - lu.lp[k];
            cost += tail_step_cost + (cnt + lanes - 1) / lanes;
        }
    }
    return cost;
}

/// Head block runs over level-grouped rows: each level cut every
/// `solve_block` rows, so a run's rows never read each other.
fn runs(ar: std.mem.Allocator, lev: []const u32) ![]u32 {
    var out: std.ArrayList(u32) = .empty;
    for (lev[0 .. lev.len - 1], lev[1..]) |a, b| {
        var s = a;
        while (s < b) : (s += klu.solve_block) try out.append(ar, s);
    }
    try out.append(ar, lev[lev.len - 1]);
    assert(out.items[0] == 0);
    assert(out.items.len >= lev.len);
    return out.items;
}

/// Rows grouped by level: `rows[lev[v]..lev[v+1]]` is level v.
fn bucket(ar: std.mem.Allocator, level: []const u32, levels: u32) !struct { rows: []u32, lev: []u32 } {
    const lev = try ar.alloc(u32, levels + 1);
    @memset(lev, 0);
    for (level) |v| lev[v + 1] += 1;
    for (0..levels) |v| lev[v + 1] += lev[v];
    const rows = try ar.alloc(u32, level.len);
    const cur = try ar.dupe(u32, lev[0..levels]);
    for (level, 0..) |v, r| {
        rows[cur[v]] = @intCast(r);
        cur[v] += 1;
    }
    assert(lev[levels] == level.len);
    return .{ .rows = rows, .lev = lev };
}

/// Builds column k's gather form when its span estimate beats the stored
/// order's (`pseq`); returns false, appending nothing, otherwise. A slot's
/// level is one past the deepest U slot it reads; slots with no
/// contribution and a void column's flops below the pivot are left out.
fn gatherColumn(
    ar: std.mem.Allocator,
    lu: *const SparseLu,
    k: u32,
    base: u32,
    width: u32,
    pos: []const u32,
    lvl: []u32,
    cnt: []u32,
    pseq: u64,
    lev_ptr: *std.ArrayList(u32),
    g_slot: *std.ArrayList(u32),
    g_cptr: *std.ArrayList(u32),
    g_l: *std.ArrayList(u32),
    g_u: *std.ArrayList(u32),
) !bool {
    const up = lu.up;
    const lp = lu.lp;
    const li = lu.li.items;
    const ui = lu.ui.items;
    const void_k = lu.void_col[k];
    assert(k < lu.n);
    assert(lvl.len >= width);
    assert(cnt.len >= width);
    const L = lvl[0..width];
    const C = cnt[0..width];
    @memset(L, 0);
    @memset(C, 0);
    const lstart = struct {
        fn f(l: *const SparseLu, i: u32) u32 {
            return l.up[i] + l.lp[i] + i + l.up[i + 1] - l.up[i] + 1;
        }
    }.f;
    // Levels and chain lengths, in stored U order.
    var depth: u32 = 0;
    for (up[k]..up[k + 1]) |p| {
        const i = ui[p];
        const lu_p = L[@as(u32, @intCast(p)) - up[k]];
        for (lp[i]..lp[i + 1]) |q| {
            const r = li[q];
            if (void_k and r > k) continue;
            const s = pos[r] - base;
            L[s] = @max(L[s], lu_p + 1);
            C[s] += 1;
            depth = @max(depth, L[s]);
        }
    }
    // Per level: the longest chain.
    const chain = try ar.alloc(u32, depth + 1);
    defer ar.free(chain);
    @memset(chain, 0);
    const per = try ar.alloc(u32, depth + 2);
    defer ar.free(per);
    @memset(per, 0);
    for (L, C) |l, cn| if (cn > 0) {
        chain[l] = @max(chain[l], cn);
        per[l + 1] += 1;
    };
    var gcost: u64 = level_cost;
    for (chain[1..]) |ch| gcost += level_cost + ch;
    // depth 0: every flop hits the discard slot; the stored-order form
    // (whose dmap must cover them) is as cheap.
    if (depth == 0 or gcost >= pseq) return false;

    // Items grouped by level (1..depth); contributions per item in stored
    // U order. `per` becomes each level's item cursor, C each item's.
    for (1..depth + 1) |l| per[l + 1] += per[l];
    const item0: u32 = @intCast(g_slot.items.len);
    const n_items = per[depth + 1];
    const slot_item = try ar.alloc(u32, width);
    defer ar.free(slot_item);
    try g_slot.resize(ar, item0 + n_items);
    for (L, C, 0..) |l, cn, s| if (cn > 0) {
        const it = per[l];
        per[l] += 1;
        slot_item[s] = it;
        g_slot.items[item0 + it] = base + @as(u32, @intCast(s));
    };
    // Levels' item ranges: per[l] now ends level l, i.e. starts level l+1.
    for (1..depth + 1) |l| try lev_ptr.append(ar, item0 + per[l]);
    // Contribution offsets per item.
    const c0: u32 = @intCast(g_l.items.len);
    const cptr0 = g_cptr.items.len - 1; // g_cptr[item0] exists
    try g_cptr.resize(ar, cptr0 + 1 + n_items);
    const cp = g_cptr.items[cptr0..];
    for (C, 0..) |cn, s| if (cn > 0) {
        cp[slot_item[s] + 1] = cn;
    };
    cp[0] = c0;
    for (0..n_items) |it| cp[it + 1] += cp[it];
    try g_l.resize(ar, cp[n_items]);
    try g_u.resize(ar, cp[n_items]);
    @memset(C, 0);
    for (up[k]..up[k + 1]) |p| {
        const i = ui[p];
        const l0 = lstart(lu, i);
        for (lp[i]..lp[i + 1]) |q| {
            const r = li[q];
            if (void_k and r > k) continue;
            const s = pos[r] - base;
            const at = cp[slot_item[s]] + C[s];
            C[s] += 1;
            g_l.items[at] = l0 + @as(u32, @intCast(q - lp[i]));
            g_u.items[at] = base + @as(u32, @intCast(p)) - up[k];
        }
    }
    return true;
}
