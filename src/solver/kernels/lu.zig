//! The device LU of dev/solvers/gpu-lu.md, stage 1: a replay of
//! `SparseLu`'s refactor and both triangular solves, written once for the
//! GPU (`lu_device.zig`) and for host threads (`runHost`). Each value slot
//! receives its subtractions in the host's order and no expression is
//! contracted, so the result is bitwise `SparseLu.refactorColumns` followed
//! by `solve`.
//!
//! The refactor is sync-free: one block per pivot step, taken by a u32
//! ticket in pivot order, waiting on the done stamps of the columns it
//! reads (release stores, acquire loads). The only read-modify-writes are
//! the relaxed ticket and the fail word's max, whose result does not depend
//! on order. The solves' sparse head is sync-free the same way: one block
//! per run of at most `solve_block` rows of one level, by ticket in level
//! order, each lane gathering one row after the done stamps of the rows it
//! reads. The dense tail, where one row per level would leave every lane
//! but one idle, runs as panels of `panel` rows, one block each by ticket:
//! a panel's rows gather each earlier panel's columns once it publishes,
//! then sweep their own diagonal block from shared memory.
//! std only: the kernel root compiles this file for NVPTX and AMDGCN.

const std = @import("std");
const assert = std.debug.assert;

/// Lanes per device refactor block. E2 on logic_bsim4_10k and
/// chain_bsim4_10k: 128 or 256 lanes were 1.0-1.8x slower
/// (dev/solvers/gpu-lu.md).
pub const refactor_block = 64;
/// The widest column a refactor block accumulates in shared memory (8
/// bytes a slot); wider columns work in `val`. A 2048-slot scratch (fewer
/// resident blocks) measured slower in the same E2 runs.
pub const col_max = 256;
/// Lanes of a device solve head block, one row each.
pub const solve_block = 256;
/// Tail rows per panel block of the tail solves (`lpanel`, `upanel`); a
/// block's lanes stride its rows, and its diagonal block (32 KB) sits in
/// shared memory, one u64 presence mask per row. At most 64: the mask.
/// logic_bsim4_10k, tail 2,048: solves 7.04 -> 5.9 ms a call against the
/// one-block column sweep this replaced; sram_bsim4_10k 5.55 -> 5.26,
/// chain_bsim4_30k 5.17 -> 4.88.
pub const panel = 64;
/// Longest solve tail the cut search (`solveCost`) tries.
pub const tail_max = 5120;
/// `Tab.flags` bit: the step has a fabricated unit pivot
/// (`SparseLu.void_col`).
pub const void_bit: u32 = 1;
/// `Tab.flags` bit: the pivot passed only the row-scaled test
/// (`SparseLu.scaled_pivot`), so the growth monitor skips it.
pub const scaled_bit: u32 = 2;
/// A narrow `dmap` entry (`Tab.dmap16`) naming the discard slot.
pub const narrow_discard: u16 = 0xFFFF;
/// u32 words ahead of the done stamps in the sync buffer: the refactor
/// ticket, the fail word (`n - k` for the lowest failing pivot step k, 0
/// when every step passed), and the forward and back solves' head tickets,
/// padded to one 64-byte line so the per-column ticket RMW does not
/// invalidate the first done stamps' line (#173). A literal, not
/// `std.atomic.cache_line`: host and device must agree on it.
pub const sync_header = 16;
comptime {
    assert(sync_header * @sizeOf(u32) == 64);
    // The column-max reduction halves the lane count down to one.
    assert(std.math.isPowerOfTwo(refactor_block));
    assert(std.math.isPowerOfTwo(solve_block));
    // A refactor block's `Shared` and a panel block's `PanelShared` fit the
    // 48 KB a block may declare.
    assert((col_max + 1 + refactor_block) * @sizeOf(f64) <= 48 * 1024);
    assert(panel <= 64 and panel * (panel + 3) * @sizeOf(f64) <= 48 * 1024);
}

/// Offsets (in u32 words) of every table inside the one index buffer, plus
/// the scalars the kernels read. Passed to each kernel by value.
///
/// Values live column-contiguous in `val`: column k is `[U_k, d_k, L_k]`
/// at `coff[k]`, and the last slot (`discard`) absorbs the flops of a void
/// column's rows below its pivot.
pub const Tab = extern struct {
    n: u32,
    discard: u32,
    growth: f64,
    // Refactor.
    coff: u32,
    up: u32,
    lp: u32,
    ui: u32,
    /// Per U entry p (source column i = ui[p]): the first value slot of
    /// L[:,i] and its length.
    ul0: u32,
    ulen: u32,
    flags: u32,
    q: u32,
    col_ptr: u32,
    amap: u32,
    /// Narrow columns: `dmap[dcol[k]..dcol[k+1]]`, the destination slot of
    /// each flop, U entries in stored order.
    dcol: u32,
    dmap: u32,
    /// Wide columns (gather form): levels `wlev[k]..wlev[k+1]`, each a
    /// range of items `lev_ptr`; item i updates slot `g_slot[i]` with the
    /// contributions `g_l[c] * g_u[c]`, c in `g_cptr[i]..g_cptr[i+1]`, in
    /// the host's order. A narrow column has an empty level range.
    wlev: u32,
    lev_ptr: u32,
    g_slot: u32,
    g_cptr: u32,
    g_l: u32,
    g_u: u32,
    /// Solves: pivot steps from `t0` on are the tail, run in panels by
    /// rows (`tl_*`, `tu_*` below); tail row c (step t0 + c) has its pivot
    /// at slot `t_diag[c]`. The head runs level by
    /// level: `l_rows[l_lev[v]..l_lev[v+1]]` are the rows of forward level
    /// v (tail rows too, for their head part), `u_rows`/`u_lev` the head
    /// rows of the back solve. Row r's head L entries (k < t0, ascending)
    /// are `l_col`/`l_slot` over `l_ptr[r]..`; its U entries (k
    /// descending) `u_col`/`u_slot`. A head block takes the rows
    /// `l_rows[l_chunk[c]..l_chunk[c+1]]` (`u_rows`/`u_chunk` going back):
    /// at most `solve_block` of them, all of one level.
    t0: u32,
    l_levels: u32,
    u_levels: u32,
    l_chunks: u32,
    u_chunks: u32,
    l_chunk: u32,
    u_chunk: u32,
    li: u32,
    t_diag: u32,
    l_lev: u32,
    l_rows: u32,
    l_ptr: u32,
    l_col: u32,
    l_slot: u32,
    /// perm_in[r] = the original row pivoted at step r.
    perm_in: u32,
    u_lev: u32,
    u_rows: u32,
    u_ptr: u32,
    u_col: u32,
    u_slot: u32,
    /// Length of `val`, `discard` included.
    n_val: u32,
    /// Length of the index buffer.
    n_idx: u32,
    /// Tail rows for the panel solves, tail-relative: row i's L entries
    /// left of it (columns ascending) are `tl_col`/`tl_slot` over
    /// `tl_ptr[i]..tl_ptr[i+1]`; its U entries right of it (columns
    /// descending) `tu_col`/`tu_slot` over `tu_ptr`.
    tl_ptr: u32,
    tl_col: u32,
    tl_slot: u32,
    tu_ptr: u32,
    tu_col: u32,
    tu_slot: u32,
    /// Nonzero: `dmap` holds u16 slots relative to the column's first,
    /// `narrow_discard` for the discard slot (`refactor(u16, ...)`); zero:
    /// u32 absolute slots.
    dmap16: u32,

    /// u32 words in the sync buffer: header plus three done stamps per
    /// pivot step (refactor, forward row, back row), then one per tail
    /// panel and direction. Zero it once per table upload.
    pub fn syncLen(t: Tab) u32 {
        return sync_header + 3 * t.n + 2 * panels(t);
    }

    /// Panel blocks of each tail solve.
    pub fn panels(t: Tab) u32 {
        return (t.n - t.t0 + panel - 1) / panel;
    }
};
comptime {
    // Passed to every kernel by value: the layout is the device ABI.
    assert(@sizeOf(Tab) == 208);
}

/// The kernel bodies over the pointer type `Sy.P` (a global pointer on the
/// device, a plain one on the host), `Sy.barrier`/`Sy.pause(spins)` and the done
/// stamps' `Sy.acquire`/`Sy.release`. `sh` is
/// the block's shared scratch (a `*Shared`, in the shared address space on
/// the device); `tid`/`nt` are the lane and lane count (0 and 1 on the
/// host), `nt` at most `lanes`.
pub fn Kernels(comptime Sy: type, comptime cap: u32, comptime lanes: u32) type {
    return struct {
        const P = Sy.P;

        /// Refactor block scratch: the ticket, the poll windows' first
        /// unfinished step (two, alternating), the column max per lane, and
        /// the column's working slots (plus a discard slot) when it has at
        /// most `cap`.
        pub const Shared = struct { k: u32, first: [2]u32, red: [lanes]f64, w: [cap + 1]f64 };
        /// Head block scratch: the ticket.
        pub const SolveShared = struct { k: u32 };
        /// Panel block scratch: the ticket, each row's running value and
        /// its next entry.
        pub const PanelShared = struct { k: u32, a: [panel]f64, e: [panel]u32, mask: [panel]u64, d: [panel]f64, blk: [panel][panel]f64 };

        inline fn wait(p: anytype, stamp: u32) void {
            // Ends: the awaited column belongs to an earlier ticket, and a
            // step waits only on earlier ones, so the earliest unfinished
            // step never waits and always publishes.
            var spins: u32 = 0;
            while (Sy.acquire(p) != stamp) : (spins +%= 1) Sy.pause(spins);
        }

        inline fn takeTicket(sy: P(u32), which: u32, sh: anytype, tid: u32) u32 {
            if (tid == 0) sh.k = @atomicRmw(u32, &sy[which], .Add, 1, .monotonic);
            Sy.barrier();
            return sh.k;
        }

        /// Refactors the pivot step its ticket names: zero its slots, scatter
        /// A's column, replay the flops, test and scale the pivot, publish.
        /// `D` is the `dmap` entry type, u16 when `t.dmap16` is set.
        /// Returns false once the tickets run out, so a host thread loops on
        /// it and a device block calls it once.
        pub fn refactor(comptime D: type, t: Tab, x: P(u32), val: P(f64), a: P(f64), sy: P(u32), stamp: u32, sh: anytype, tid: u32, nt: u32) bool {
            comptime assert(D == u16 or D == u32);
            const k = takeTicket(sy, 0, sh, tid);
            if (k >= t.n) return false;
            const width = x[t.coff + k + 1] - x[t.coff + k];
            const bad = if (width <= cap)
                column(D, true, &sh.w, t, x, val, a, sy, stamp, sh, tid, nt, k)
            else
                column(D, false, val, t, x, val, a, sy, stamp, sh, tid, nt, k);
            if (bad) _ = @atomicRmw(u32, &sy[1], .Max, t.n - k, .monotonic);
            Sy.barrier();
            if (tid == 0) Sy.release(&(sy + sync_header)[k], stamp);
            return true;
        }

        /// Lane 0 waits for step p0's source; lane t > 0 checks step
        /// p0 + t's once and records the first unfinished one in
        /// `sh.first[par]`. After the caller's barrier, every source that
        /// read done is visible to the whole block.
        inline fn pollWindow(t: Tab, x: P(u32), done: P(u32), stamp: u32, sh: anytype, par: u1, p0: u32, up1: u32, tid: u32) void {
            const p = p0 + tid;
            if (p >= up1) return;
            const src = &done[x[t.ui + p]];
            if (tid == 0) {
                wait(src, stamp);
            } else if (Sy.acquire(src) != stamp) {
                _ = @atomicRmw(u32, &sh.first[par], .Min, p, .monotonic);
            }
        }

        /// Flop i's destination: an absolute value slot or `t.discard`.
        inline fn slotAt(comptime D: type, x: P(u32), t: Tab, i: u32, base: u32) u32 {
            if (D == u32) return x[t.dmap + i];
            // A narrow dmap starts the index buffer (`t.dmap` is 0).
            const rel = @as(P(u16), @ptrCast(x))[i];
            return if (rel == narrow_discard) t.discard else base + rel;
        }

        /// Column k's replay on `w`: the block's scratch (`local`, indexed
        /// from the column's first slot, discard slot last) or `val` itself.
        /// Returns whether a pivot test failed or a slot this lane checked
        /// holds a NaN or inf.
        inline fn column(comptime D: type, comptime local: bool, w: anytype, t: Tab, x: P(u32), val: P(f64), a: P(f64), sy: P(u32), stamp: u32, sh: anytype, tid: u32, nt: u32, k: u32) bool {
            const done = sy + sync_header;
            const base = x[t.coff + k];
            const width = x[t.coff + k + 1] - base;
            // Column slot s lives at w[o + s]; `at` maps any value slot of
            // this column (or the discard slot) into w.
            const o: u32 = if (local) 0 else base;
            const S = struct {
                inline fn at(dst: u32, b: u32, wd: u32) u32 {
                    return if (local) @min(dst -% b, wd) else dst;
                }
            };
            // No window is open yet: every window slot says "none unfinished".
            if (tid == 0) sh.first = .{ ~@as(u32, 0), ~@as(u32, 0) };
            var s = tid;
            while (s < width) : (s += nt) w[o + s] = 0;
            Sy.barrier();
            const c = x[t.q + k];
            var e = x[t.col_ptr + c] + tid;
            while (e < x[t.col_ptr + c + 1]) : (e += nt) w[S.at(x[t.amap + e], base, width)] = a[e];

            const up0 = x[t.up + k];
            const up1 = x[t.up + k + 1];
            const w0 = x[t.wlev + k];
            const w1 = x[t.wlev + k + 1];
            if (w0 == w1) {
                // Stored U order; the L rows of one U entry are distinct. A
                // step's end barrier doubles as the next one's wait barrier.
                // Sources are polled a window at a time, one lane per step:
                // steps before `ready` need no poll of their own.
                var ready = up0;
                var par: u1 = 0;
                if (up0 < up1) pollWindow(t, x, done, stamp, sh, par, up0, up1, tid);
                Sy.barrier();
                if (up0 < up1) {
                    ready = @max(@min(sh.first[par], up0 + nt), up0 + 1);
                    par ^= 1;
                }
                var d = x[t.dcol + k];
                var p = up0;
                var l0 = if (up0 < up1) x[t.ul0 + up0] else 0;
                var len = if (up0 < up1) x[t.ulen + up0] else 0;
                while (p < up1) : (p += 1) {
                    const u = w[o + p - up0];
                    const more = p + 1 < up1;
                    const nl0 = if (more) x[t.ul0 + p + 1] else 0;
                    const nlen = if (more) x[t.ulen + p + 1] else 0;
                    // Four flops' loads go out before their stores: the
                    // destinations are distinct, but only we know that.
                    const U = 4;
                    var j = tid;
                    while (j + (U - 1) * nt < len) : (j += U * nt) {
                        var dst: [U]u32 = undefined;
                        var lv: [U]f64 = undefined;
                        var wv: [U]f64 = undefined;
                        inline for (0..U) |m| {
                            dst[m] = S.at(slotAt(D, x, t, d + j + @as(u32, m) * nt, base), base, width);
                            lv[m] = val[l0 + j + m * nt];
                        }
                        inline for (0..U) |m| wv[m] = w[dst[m]];
                        inline for (0..U) |m| w[dst[m]] = wv[m] - lv[m] * u;
                    }
                    while (j < len) : (j += nt) {
                        const dst = S.at(slotAt(D, x, t, d + j, base), base, width);
                        w[dst] -= val[l0 + j] * u;
                    }
                    d += len;
                    const poll = more and p + 1 >= ready;
                    if (poll) pollWindow(t, x, done, stamp, sh, par, p + 1, up1, tid);
                    Sy.barrier();
                    if (poll) {
                        ready = @max(@min(sh.first[par], p + 1 + nt), p + 2);
                        // The other slot was last read before this barrier.
                        if (tid == 0) sh.first[par ^ 1] = ~@as(u32, 0);
                        par ^= 1;
                    }
                    l0 = nl0;
                    len = nlen;
                }
            } else {
                // Gather form: wait for every source once, then per level
                // each item runs its slot's contributions in stored order.
                var p = up0 + tid;
                while (p < up1) : (p += nt) wait(&done[x[t.ui + p]], stamp);
                Sy.barrier();
                var lev = w0;
                while (lev < w1) : (lev += 1) {
                    var it = x[t.lev_ptr + lev] + tid;
                    while (it < x[t.lev_ptr + lev + 1]) : (it += nt) {
                        const slot = S.at(x[t.g_slot + it], base, width);
                        var acc = w[slot];
                        var g = x[t.g_cptr + it];
                        const g1 = x[t.g_cptr + it + 1];
                        // Eight contributions' loads, then their chain.
                        const U = 8;
                        while (g + U <= g1) : (g += U) {
                            var lv: [U]f64 = undefined;
                            var uv: [U]f64 = undefined;
                            inline for (0..U) |m| {
                                lv[m] = val[x[t.g_l + g + m]];
                                uv[m] = w[S.at(x[t.g_u + g + m], base, width)];
                            }
                            inline for (0..U) |m| acc -= lv[m] * uv[m];
                        }
                        while (g < g1) : (g += 1) acc -= val[x[t.g_l + g]] * w[S.at(x[t.g_u + g], base, width)];
                        w[slot] = acc;
                    }
                    Sy.barrier();
                }
            }

            // The pivot tests of `refactorColumns`, in its order.
            const diag = o + up1 - up0;
            const end = o + width;
            const flags = x[t.flags + k];
            var bad = false;
            if (flags & void_bit != 0) {
                if (tid == 0) w[diag] = 1;
                var l = diag + 1 + tid;
                while (l < end) : (l += nt) w[l] = 0;
            } else {
                const dv = w[diag];
                if (dv == 0 or !std.math.isFinite(dv)) {
                    bad = true;
                } else {
                    // cmax is a max: exact in any order. It drops a NaN
                    // the host's bit-pattern max keeps; the slot scan below
                    // fails that step instead.
                    var cm: f64 = @abs(dv);
                    var l = diag + 1 + tid;
                    // IEEE `/`. A Markstein division (one reciprocal, five
                    // FMAs a slot, same bits) measured 5-7% slower on the
                    // refactor (logic_bsim4_10k 11.04 -> 11.61 ms,
                    // sram_bsim4_10k 6.55 -> 7.05), so ptxas's own
                    // `div.rn.f64` stays.
                    while (l < end) : (l += nt) {
                        const v = w[l];
                        cm = @max(cm, @abs(v));
                        w[l] = v / dv;
                    }
                    if (t.growth > 0 and flags & scaled_bit == 0) {
                        // A tree over the lanes (nt is a power of two); only
                        // lane 0 reads the verdict.
                        sh.red[tid] = cm;
                        Sy.barrier();
                        var h = nt / 2;
                        while (h > 0) : (h /= 2) {
                            if (tid < h) sh.red[tid] = @max(sh.red[tid], sh.red[tid + h]);
                            Sy.barrier();
                        }
                        bad = @abs(dv) < t.growth * sh.red[0];
                    }
                }
            }
            // Every slot is final. A NaN or inf L or U entry fails the step,
            // as `refactorColumns`' stores flag it; the barrier because
            // other lanes wrote this lane's slots.
            Sy.barrier();
            var nonfinite: u1 = 0;
            s = tid;
            while (s < width) : (s += nt) {
                const v = w[o + s];
                nonfinite |= @intFromBool(!std.math.isFinite(v));
                if (local) val[base + s] = v;
            }
            return bad or nonfinite != 0;
        }

        /// Forward solve: y = L^-1 P (-rhs), each row's subtractions in
        /// ascending k, skipping y[k] == 0 as `SparseLu.solve` does. The
        /// host form: every head ticket, then every panel. `sh` is a
        /// `*SolveShared`, `psh` a `*PanelShared`. Does nothing when the
        /// refactor failed.
        pub fn lsolve(t: Tab, x: P(u32), val: P(f64), rhs: P(f64), y: P(f64), sy: P(u32), stamp: u32, sh: anytype, psh: anytype, tid: u32, nt: u32) void {
            while (lhead(t, x, val, rhs, y, sy, stamp, sh, tid, nt)) {}
            while (lpanel(t, x, val, y, sy, stamp, psh, tid, nt)) {}
        }

        /// The forward head for the run its ticket names: each lane its
        /// rows' head entries (k < t0), after their rows' done stamps,
        /// then its own stamp. Returns false once the tickets run out, so
        /// a host thread loops on it and a device block calls it once.
        pub fn lhead(t: Tab, x: P(u32), val: P(f64), rhs: P(f64), y: P(f64), sy: P(u32), stamp: u32, sh: anytype, tid: u32, nt: u32) bool {
            const c = takeTicket(sy, 2, sh, tid);
            if (c >= t.l_chunks) return false;
            if (sy[1] != 0) return true;
            const done = sy + sync_header + t.n;
            var j = x[t.l_chunk + c] + tid;
            while (j < x[t.l_chunk + c + 1]) : (j += nt) {
                const r = x[t.l_rows + j];
                const e0 = x[t.l_ptr + r];
                const e1 = x[t.l_ptr + r + 1];
                var e = e0;
                while (e < e1) : (e += 1) wait(&done[x[t.l_col + e]], stamp);
                y[r] = gather(-rhs[x[t.perm_in + r]], x, val, y, t.l_col, t.l_slot, e0, e1);
                Sy.release(&done[r], stamp);
            }
            return true;
        }

        /// Back solve: z = U^-1 y, each row's subtractions in descending
        /// k, and dx[q[i]] = z[i]; y ends holding z. The tail panels run
        /// first; head rows gather their tail entries afterwards, which is
        /// still the host's descending order. The host form: every panel,
        /// then every head ticket.
        pub fn usolve(t: Tab, x: P(u32), val: P(f64), y: P(f64), dx: P(f64), sy: P(u32), stamp: u32, sh: anytype, psh: anytype, tid: u32, nt: u32) void {
            while (upanel(t, x, val, y, dx, sy, stamp, psh, tid, nt)) {}
            while (uhead(t, x, val, y, dx, sy, stamp, sh, tid, nt)) {}
        }

        /// The back head for the run its ticket names, as `lhead`: a head
        /// row waits only on head rows; the tail is final before it runs.
        pub fn uhead(t: Tab, x: P(u32), val: P(f64), y: P(f64), dx: P(f64), sy: P(u32), stamp: u32, sh: anytype, tid: u32, nt: u32) bool {
            const c = takeTicket(sy, 3, sh, tid);
            if (c >= t.u_chunks) return false;
            if (sy[1] != 0) return true;
            const done = sy + sync_header + 2 * t.n;
            var j = x[t.u_chunk + c] + tid;
            while (j < x[t.u_chunk + c + 1]) : (j += nt) {
                const i = x[t.u_rows + j];
                const e0 = x[t.u_ptr + i];
                const e1 = x[t.u_ptr + i + 1];
                var e = e0;
                while (e < e1) : (e += 1) {
                    const k = x[t.u_col + e];
                    if (k < t.t0) wait(&done[k], stamp);
                }
                const acc = gather(y[i], x, val, y, t.u_col, t.u_slot, e0, e1);
                const z = acc / val[x[t.coff + i] + x[t.up + i + 1] - x[t.up + i]];
                y[i] = z;
                dx[x[t.q + i]] = z;
                Sy.release(&done[i], stamp);
            }
            return true;
        }

        /// The forward tail for the panel its ticket names: its rows take
        /// every earlier panel's columns as each publishes (a row's
        /// subtractions stay in ascending column order), then sweep their
        /// own panel's columns from a dense copy in shared memory, and
        /// publish. The rows leave `y` holding L^-1 for the tail. `sh` is a
        /// `*PanelShared`. Returns false once the tickets run out.
        pub fn lpanel(t: Tab, x: P(u32), val: P(f64), y: P(f64), sy: P(u32), stamp: u32, sh: anytype, tid: u32, nt: u32) bool {
            const b = takeTicket(sy, 4, sh, tid);
            const np = t.panels();
            if (b >= np) return false;
            if (sy[1] != 0) return true;
            const done = sy + sync_header + 3 * t.n;
            const r0 = b * panel;
            const r1 = @min(r0 + panel, t.n - t.t0);
            var i = r0 + tid;
            while (i < r1) : (i += nt) {
                sh.a[i - r0] = y[t.t0 + i];
                sh.e[i - r0] = x[t.tl_ptr + i];
            }
            var j: u32 = 0;
            while (j < b) : (j += 1) {
                if (tid == 0) wait(&done[j], stamp);
                Sy.barrier();
                i = r0 + tid;
                while (i < r1) : (i += nt) {
                    const ga = gatherBelow(sh.a[i - r0], x, val, y + t.t0, t.tl_col, t.tl_slot, sh.e[i - r0], x[t.tl_ptr + i + 1], (j + 1) * panel);
                    sh.a[i - r0] = ga.acc;
                    sh.e[i - r0] = ga.e;
                }
            }
            loadBlock(x, val, t.tl_ptr, t.tl_col, t.tl_slot, r0, r1, sh, tid, nt);
            var c = r0;
            while (c < r1) : (c += 1) {
                // Row c holds every subtraction from columns before it.
                Sy.barrier();
                const yc = sh.a[c - r0];
                if (yc == 0) continue;
                const bit = @as(u64, 1) << @intCast(c - r0);
                i = r0 + tid;
                while (i < r1) : (i += nt) {
                    if (sh.mask[i - r0] & bit != 0) sh.a[i - r0] -= sh.blk[i - r0][c - r0] * yc;
                }
            }
            Sy.barrier();
            i = r0 + tid;
            while (i < r1) : (i += nt) y[t.t0 + i] = sh.a[i - r0];
            Sy.barrier();
            if (tid == 0) Sy.release(&done[b], stamp);
            return true;
        }

        /// The back tail for the panel its ticket names, last panel first:
        /// its rows take every later panel's columns as each publishes
        /// (descending), then sweep their own panel from its last row, each
        /// row dividing by its pivot when its turn comes, and publish z into
        /// `y` and `dx`.
        pub fn upanel(t: Tab, x: P(u32), val: P(f64), y: P(f64), dx: P(f64), sy: P(u32), stamp: u32, sh: anytype, tid: u32, nt: u32) bool {
            const b = takeTicket(sy, 5, sh, tid);
            const np = t.panels();
            if (b >= np) return false;
            if (sy[1] != 0) return true;
            const done = sy + sync_header + 3 * t.n + np;
            const pb = np - 1 - b;
            const r0 = pb * panel;
            const r1 = @min(r0 + panel, t.n - t.t0);
            var i = r0 + tid;
            while (i < r1) : (i += nt) {
                sh.a[i - r0] = y[t.t0 + i];
                sh.e[i - r0] = x[t.tu_ptr + i];
            }
            var j = np - 1;
            while (j > pb) : (j -= 1) {
                if (tid == 0) wait(&done[j], stamp);
                Sy.barrier();
                i = r0 + tid;
                while (i < r1) : (i += nt) {
                    const ga = gatherAtLeast(sh.a[i - r0], x, val, y + t.t0, t.tu_col, t.tu_slot, sh.e[i - r0], x[t.tu_ptr + i + 1], j * panel);
                    sh.a[i - r0] = ga.acc;
                    sh.e[i - r0] = ga.e;
                }
            }
            loadBlock(x, val, t.tu_ptr, t.tu_col, t.tu_slot, r0, r1, sh, tid, nt);
            // The panel's pivots, staged with the block.
            i = r0 + tid;
            while (i < r1) : (i += nt) sh.d[i - r0] = val[x[t.t_diag + i]];
            var c = r1;
            while (c > r0) {
                c -= 1;
                Sy.barrier();
                if (tid == (c - r0) % nt) {
                    const z = sh.a[c - r0] / sh.d[c - r0];
                    sh.a[c - r0] = z;
                    y[t.t0 + c] = z;
                    dx[x[t.q + t.t0 + c]] = z;
                }
                Sy.barrier();
                const zc = sh.a[c - r0];
                if (zc == 0) continue;
                const bit = @as(u64, 1) << @intCast(c - r0);
                i = r0 + tid;
                while (i < c) : (i += nt) {
                    if (sh.mask[i - r0] & bit != 0) sh.a[i - r0] -= sh.blk[i - r0][c - r0] * zc;
                }
            }
            Sy.barrier();
            if (tid == 0) Sy.release(&done[pb], stamp);
            return true;
        }

        /// Each row's remaining entries (all inside rows r0..r1's panel)
        /// into `sh.blk`, flagged in `sh.mask`; the barrier after it is the
        /// caller's first step.
        inline fn loadBlock(x: P(u32), val: P(f64), ptr: u32, col: u32, slot: u32, r0: u32, r1: u32, sh: anytype, tid: u32, nt: u32) void {
            var i = r0 + tid;
            while (i < r1) : (i += nt) {
                var m: u64 = 0;
                var e = sh.e[i - r0];
                const e1 = x[ptr + i + 1];
                while (e < e1) : (e += 1) {
                    const c = x[col + e] - r0;
                    sh.blk[i - r0][c] = val[x[slot + e]];
                    m |= @as(u64, 1) << @intCast(c);
                }
                sh.mask[i - r0] = m;
            }
        }

        const Ga = struct { acc: f64, e: u32 };

        /// `acc` minus val[slot[e]] * yt[col[e]] from entry e on while
        /// col[e] < hi (columns ascending), skipping yt == 0; eight
        /// entries' loads go out before their subtractions.
        inline fn gatherBelow(acc0: f64, x: P(u32), val: P(f64), yt: P(f64), col: u32, slot: u32, e0: u32, e1: u32, hi: u32) Ga {
            const U = 8;
            var acc = acc0;
            var e = e0;
            while (e + U <= e1 and x[col + e + U - 1] < hi) : (e += U) {
                var yv: [U]f64 = undefined;
                var lv: [U]f64 = undefined;
                inline for (0..U) |u| {
                    yv[u] = yt[x[col + e + u]];
                    lv[u] = val[x[slot + e + u]];
                }
                inline for (0..U) |u| acc = if (yv[u] != 0) acc - lv[u] * yv[u] else acc;
            }
            while (e < e1 and x[col + e] < hi) : (e += 1) {
                const yk = yt[x[col + e]];
                if (yk != 0) acc -= val[x[slot + e]] * yk;
            }
            return .{ .acc = acc, .e = e };
        }

        /// `gatherBelow` for columns descending, while col[e] >= lo.
        inline fn gatherAtLeast(acc0: f64, x: P(u32), val: P(f64), yt: P(f64), col: u32, slot: u32, e0: u32, e1: u32, lo: u32) Ga {
            const U = 8;
            var acc = acc0;
            var e = e0;
            while (e + U <= e1 and x[col + e + U - 1] >= lo) : (e += U) {
                var yv: [U]f64 = undefined;
                var lv: [U]f64 = undefined;
                inline for (0..U) |u| {
                    yv[u] = yt[x[col + e + u]];
                    lv[u] = val[x[slot + e + u]];
                }
                inline for (0..U) |u| acc = if (yv[u] != 0) acc - lv[u] * yv[u] else acc;
            }
            while (e < e1 and x[col + e] >= lo) : (e += 1) {
                const yk = yt[x[col + e]];
                if (yk != 0) acc -= val[x[slot + e]] * yk;
            }
            return .{ .acc = acc, .e = e };
        }

        /// acc minus val[slot[e]] * y[col[e]] for e in e0..e1 in order,
        /// skipping y == 0. Eight entries' loads go out before their
        /// subtractions, so a long row waits on memory once per eight.
        inline fn gather(acc0: f64, x: P(u32), val: P(f64), y: P(f64), col: u32, slot: u32, e0: u32, e1: u32) f64 {
            const U = 8;
            var acc = acc0;
            var e = e0;
            while (e + U <= e1) : (e += U) {
                var yv: [U]f64 = undefined;
                var lv: [U]f64 = undefined;
                inline for (0..U) |u| {
                    yv[u] = y[x[col + e + u]];
                    lv[u] = val[x[slot + e + u]];
                }
                inline for (0..U) |u| acc = if (yv[u] != 0) acc - lv[u] * yv[u] else acc;
            }
            while (e < e1) : (e += 1) {
                const yk = y[x[col + e]];
                if (yk != 0) acc -= val[x[slot + e]] * yk;
            }
            return acc;
        }
    };
}

/// The host instance: plain pointers, a block of one lane.
pub const HostSy = struct {
    /// A plain many-item pointer: host memory has one address space.
    pub fn P(comptime T: type) type {
        return [*]T;
    }
    /// A no-op: a host block is one lane.
    pub inline fn barrier() void {}
    /// Spins briefly, then yields: on a loaded machine the thread holding
    /// the awaited column may be the one descheduled (chain_psp103_10k at
    /// load 98: 16.7 s of pure spinning became 149 s).
    pub inline fn pause(spins: u32) void {
        // Single-threaded never waits (wasm32-wasi has no sched_yield).
        if (spins < 256 or @import("builtin").single_threaded) std.atomic.spinLoopHint() else std.Thread.yield() catch {};
    }
    /// Reads a done stamp; pairs with `release`.
    pub inline fn acquire(p: *const u32) u32 {
        return @atomicLoad(u32, p, .acquire);
    }
    /// Publishes a done stamp after the column's stores.
    pub inline fn release(p: *u32, v: u32) void {
        @atomicStore(u32, p, v, .release);
    }
};

/// Host buffers for one `runHost` call; `sync` must be zeroed once after
/// the tables are built, and `stamp` must grow by one per call.
pub const HostBufs = struct {
    /// The index buffer `build` returned (`Tables.idx`), `n_idx` words.
    idx: []const u32,
    /// Factor values, `n_val` slots laid out as `Tab.coff` says.
    val: []f64,
    /// A's values in CSC order.
    a: []const f64,
    /// The residual; the solve returns dx = -A^-1 rhs.
    rhs: []const f64,
    /// Solve scratch, n entries; holds z = U^-1 L^-1 P (-rhs) afterwards.
    y: []f64,
    /// The Newton step, n entries.
    dx: []f64,
    /// `Tab.syncLen()` words of tickets and done stamps.
    sync: []u32,
};

/// Refactors on `threads` host threads (at most 64) and solves on this one
/// with the kernel bodies: bitwise `refactorColumns` plus `solve` (dx =
/// -A^-1 rhs). Returns the lowest failing pivot step, or null; the solves
/// then leave `y` and `dx` untouched. One thread is the scalar oracle: it
/// takes every ticket in order, so no wait ever spins. A thread that fails
/// to spawn leaves its tickets to the others. Fails only when the solve
/// scratch (40 KB) cannot be allocated. Asserts that `b` holds the index,
/// value and sync lengths `t` names.
pub fn runHost(t: Tab, b: HostBufs, stamp: u32, threads: u32) !?u32 {
    return runHostCap(col_max, t, b, stamp, threads);
}

/// Private functions for tests.zig; void outside test builds.
pub const test_access = if (@import("builtin").is_test) .{ .runHostCap = runHostCap } else {};

// `runHost` with the refactor's scratch capped at `cap` slots; 0 runs
// every column in `val`, the path the device takes past `col_max`.
fn runHostCap(comptime cap: u32, t: Tab, b: HostBufs, stamp: u32, threads: u32) !?u32 {
    assert(b.idx.len >= t.n_idx);
    assert(b.val.len >= t.n_val);
    assert(b.sync.len >= t.syncLen());
    const H = Kernels(HostSy, cap, 1);
    @memset(b.sync[0..sync_header], 0);
    const x = @constCast(b.idx.ptr);
    const Worker = struct {
        fn work(tt: Tab, bb: HostBufs, st: u32) void {
            var sh: H.Shared = undefined;
            const ix = @constCast(bb.idx.ptr);
            const av = @constCast(bb.a.ptr);
            if (tt.dmap16 != 0) {
                while (H.refactor(u16, tt, ix, bb.val.ptr, av, bb.sync.ptr, st, &sh, 0, 1)) {}
            } else while (H.refactor(u32, tt, ix, bb.val.ptr, av, bb.sync.ptr, st, &sh, 0, 1)) {}
        }
    };
    var pool: [63]std.Thread = undefined;
    const extra = @min(@max(threads, 1) - 1, pool.len);
    // Returning on a failed spawn would free `b` under the running threads.
    var spawned: usize = 0;
    // Single-threaded (wasm) never names `spawn`: one thread, the oracle.
    if (!@import("builtin").single_threaded) {
        for (pool[0..extra]) |*th| {
            th.* = std.Thread.spawn(.{}, Worker.work, .{ t, b, stamp }) catch break;
            spawned += 1;
        }
    }
    Worker.work(t, b, stamp);
    for (pool[0..spawned]) |th| th.join();
    var ssh: H.SolveShared = undefined;
    // 34 KB, too much for a worker thread's stack.
    const psh = try std.heap.page_allocator.create(H.PanelShared);
    defer std.heap.page_allocator.destroy(psh);
    H.lsolve(t, x, b.val.ptr, @constCast(b.rhs.ptr), b.y.ptr, b.sync.ptr, stamp, &ssh, psh, 0, 1);
    H.usolve(t, x, b.val.ptr, b.y.ptr, b.dx.ptr, b.sync.ptr, stamp, &ssh, psh, 0, 1);
    const f = b.sync[1];
    assert(f <= t.n);
    return if (f == 0) null else t.n - f;
}

// ---------------------------------------------------------------------------
// Tables, built on the host from a factored SparseLu.

const SparseLu = @import("../sparse_lu.zig").SparseLu;

/// The refactor kernel body as a multicore host refactor (dev/solvers/
/// gpu-lu.md option 3): one pivot epoch's tables and buffers. Its factors
/// are bitwise `SparseLu.refactor`'s, so callers may pick it by speed.
pub const HostRefactor = struct {
    const H = Kernels(HostSy, col_max, 1);

    /// `SparseLu.pattern_epoch` the tables were built from.
    epoch: u32,
    tb: Tables,
    val: []f64,
    /// Line-aligned, so the header fills exactly the first line.
    sync: []align(64) u32,
    stamp: u32 = 0,

    /// Builds the tables for `lu`'s current factor (4 bytes per flop).
    /// Caller owns the result; free with `deinit`.
    pub fn init(gpa: std.mem.Allocator, lu: *const SparseLu, col_ptr: []const u32, growth: f64) !HostRefactor {
        // No solve runs from these tables: skip the tail cut search.
        var tb = try build(gpa, lu, col_ptr, growth, .{ .tail = lu.n });
        errdefer tb.deinit(gpa);
        const val = try gpa.alloc(f64, tb.tab.n_val);
        errdefer gpa.free(val);
        const sync = try gpa.alignedAlloc(u32, .@"64", tb.tab.syncLen());
        @memset(sync, 0);
        assert(tb.idx.len == tb.tab.n_idx);
        return .{ .epoch = lu.pattern_epoch, .tb = tb, .val = val, .sync = sync };
    }

    /// Frees the tables and buffers; `lu` stays the caller's.
    pub fn deinit(self: *HostRefactor, gpa: std.mem.Allocator) void {
        self.tb.deinit(gpa);
        gpa.free(self.val);
        gpa.free(self.sync);
        self.* = undefined;
    }

    /// `lu.refactor(col_ptr, vals, growth)`, with the `growth` `init` got,
    /// on the caller plus `threads - 1` tasks of `io` (at most 15), then the
    /// factors copied into `lu`.
    /// Fails exactly when `refactor` would, and `lu`'s factors are then
    /// unusable, as after a failed `refactor`. Asserts that `lu` is factored
    /// on the epoch the tables were built from.
    pub fn run(self: *HostRefactor, lu: *SparseLu, vals: []const f64, io: std.Io, threads: u32) error{SingularMatrix}!void {
        assert(lu.factored);
        assert(lu.pattern_epoch == self.epoch);
        assert(self.sync.len == self.tb.tab.syncLen());
        assert(self.val.len == self.tb.tab.n_val);
        for (lu.void_slots.items) |p| {
            if (vals[p] != 0) return error.SingularMatrix;
        }
        const Work = struct {
            fn work(t: Tab, x: [*]u32, val: [*]f64, av: [*]f64, sy: [*]u32, stamp: u32) void {
                var sh: H.Shared = undefined;
                if (t.dmap16 != 0) {
                    while (H.refactor(u16, t, x, val, av, sy, stamp, &sh, 0, 1)) {}
                } else while (H.refactor(u32, t, x, val, av, sy, stamp, &sh, 0, 1)) {}
            }
        };
        self.stamp +%= 1;
        @memset(self.sync[0..sync_header], 0);
        const args = .{ self.tb.tab, self.tb.idx.ptr, self.val.ptr, @constCast(vals.ptr), self.sync.ptr, self.stamp };
        var futures: [15]std.Io.Future(void) = undefined;
        const extra = @min(@max(threads, 1) - 1, futures.len);
        for (futures[0..extra]) |*f| f.* = io.async(Work.work, args);
        @call(.auto, Work.work, args);
        for (futures[0..extra]) |*f| f.await(io);
        const t = self.tb.tab;
        assert(self.sync[1] <= t.n);
        if (self.sync[1] != 0) return error.SingularMatrix;
        const x = self.tb.idx;
        for (0..t.n) |k| {
            const base = x[t.coff + k];
            const nuk = lu.up[k + 1] - lu.up[k];
            const nlk = lu.lp[k + 1] - lu.lp[k];
            @memcpy(lu.ux.items[lu.up[k]..][0..nuk], self.val[base..][0..nuk]);
            lu.udiag[k] = self.val[base + nuk];
            @memcpy(lu.lx.items[lu.lp[k]..][0..nlk], self.val[base + nuk + 1 ..][0..nlk]);
        }
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
    tab: Tab,
    /// Every u32 table, at the offsets in `tab`. Owned; free with `deinit`.
    idx: []u32,

    /// Frees `idx`; `self` is undefined afterwards.
    pub fn deinit(self: *Tables, gpa: std.mem.Allocator) void {
        gpa.free(self.idx);
        self.* = undefined;
    }
};

/// Builds the tables for `lu`'s current factor. O(F + nnz(L+U)) time; the
/// narrow columns' `dmap` is 2 bytes per flop when every column has fewer
/// than 65,535 slots, else 4. Asserts a successful
/// `factor` and a tail of at most `tail_max` steps. `OutOfMemory` also
/// means the tables passed u32 offsets. Caller owns the result; free with
/// `Tables.deinit`. A later `factor` of `lu` (new `pattern_epoch`) makes
/// them stale.
pub fn build(gpa: std.mem.Allocator, lu: *const SparseLu, col_ptr: []const u32, growth: f64, opt: Options) !Tables {
    const lanes = refactor_block;
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
    for (flags, 0..) |*f, k| f.* = (if (lu.void_col[k]) void_bit else 0) | (if (lu.scaled_pivot[k]) scaled_bit else 0);

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
    const dmap16 = opt.dmap16 and widest < narrow_discard;
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
                    for (rows, d16[nd..][0..rows.len]) |r, *o| o.* = if (void_k and r > k) narrow_discard else @intCast(pos[r] - base);
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
        const top = @min(n, tail_max);
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
    assert(n - t0 <= tail_max);
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
        .{ "ui", ui },             .{ "ul0", ul0 },                     .{ "ulen", ulen },           .{ "flags", flags },
        .{ "q", lu.q },            .{ "col_ptr", col_ptr[0 .. n + 1] }, .{ "amap", amap },           .{ "dcol", dcol },
        .{ "wlev", wlev },         .{ "lev_ptr", lev_ptr.items },       .{ "g_slot", g_slot.items }, .{ "g_cptr", g_cptr.items },
        .{ "g_l", g_l.items },     .{ "g_u", g_u.items },               .{ "li", li }, .{ "t_diag", t_diag },               .{ "l_lev", l_sched.lev },   .{ "l_rows", l_sched.rows },
        .{ "l_ptr", l_ptr },       .{ "l_col", l_col },                 .{ "l_slot", l_slot },       .{ "perm_in", perm_in },
        .{ "u_lev", u_sched.lev }, .{ "u_rows", u_sched.rows },         .{ "u_ptr", u_ptr },         .{ "u_col", u_col },
        .{ "u_slot", u_slot },     .{ "l_chunk", l_chunk },             .{ "u_chunk", u_chunk },
        .{ "tl_ptr", tl_ptr },     .{ "tl_col", tl_col },               .{ "tl_slot", tl_slot },
        .{ "tu_ptr", tu_ptr },     .{ "tu_col", tu_col },               .{ "tu_slot", tu_slot },
    };
    var tab: Tab = undefined;
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
    const lanes = solve_block;
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
        while (s < b) : (s += solve_block) try out.append(ar, s);
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
    for (L, C, 0..) |_, cn, s| if (cn > 0) {
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

