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
//!
//! Contents:
//! - `pub const void_bit = @import("lane.zig").void_bit`
//! - `pub const scaled_bit = @import("lane.zig").scaled_bit`
//! - `pub const refactor_block = 64`
//! - `pub const col_max = 256`
//! - `pub const solve_block = 256`
//! - `pub const panel = 64`
//! - `pub const tail_max = 5120`
//! - `pub const narrow_discard: u16 = 0xFFFF`
//! - `pub const sync_header = 16`
//! - `pub const Tab = extern struct`
//!   - `pub fn syncLen(t: Tab) u32`
//!   - `pub fn panels(t: Tab) u32`
//! - `pub const HostBufs = lu_host.HostBufs`
//! - `pub const runHost = lu_host.runHost`
//! - `pub const HostRefactor = lu_host.HostRefactor`
//! - `pub const test_access = lu_host.test_access`
//! - `pub const Options = lu_tables.Options`
//! - `pub const Tables = lu_tables.Tables`
//! - `pub const build = lu_tables.build`
//! - `pub fn Kernels(comptime Sy: type, comptime cap: u32, comptime lanes: u32) type`
//!     - `pub const Shared = struct`
//!     - `pub const SolveShared = struct`
//!     - `pub const PanelShared = struct`
//!     - `pub fn refactor(comptime D: type, t: Tab, x: P(u32), val: P(f64), a: P(f64), sy: P(u32), stamp: u32, sh: anytype, tid: u32, nt: u32) bool`
//!     - `pub fn lhead(t: Tab, x: P(u32), val: P(f64), rhs: P(f64), y: P(f64), sy: P(u32), stamp: u32, sh: anytype, tid: u32, nt: u32) bool`
//!     - `pub fn uhead(t: Tab, x: P(u32), val: P(f64), y: P(f64), dx: P(f64), sy: P(u32), stamp: u32, sh: anytype, tid: u32, nt: u32) bool`
//!     - `pub fn lpanel(t: Tab, x: P(u32), val: P(f64), y: P(f64), sy: P(u32), stamp: u32, sh: anytype, tid: u32, nt: u32) bool`
//!     - `pub fn upanel(t: Tab, x: P(u32), val: P(f64), y: P(f64), dx: P(f64), sy: P(u32), stamp: u32, sh: anytype, tid: u32, nt: u32) bool`

const std = @import("std");
const assert = std.debug.assert;
const lu_host = @import("lu_host.zig");
const lu_tables = @import("lu_tables.zig");
// `Tab.flags` bits, the encoding `LaneTab.flags` uses.
/// `Tab.flags` bit of a fabricated (void) pivot, `LaneTab`'s encoding.
pub const void_bit = @import("lane.zig").void_bit;
/// `Tab.flags` bit of a row-scale-accepted pivot.
pub const scaled_bit = @import("lane.zig").scaled_bit;

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

/// A narrow `dmap` entry (`Tab.dmap16`) naming the discard slot.
pub const narrow_discard: u16 = 0xFFFF;

/// u32 words ahead of the done stamps in the sync buffer: the refactor
/// ticket, the fail word (`n - k` for the lowest failing pivot step k, 0
/// when every step passed), and the forward and back solves' head tickets,
/// padded to one 64-byte line so the per-column ticket RMW does not
/// invalidate the first done stamps' line (#173). A literal, not
/// `std.atomic.cache_line`: host and device must agree on it.
pub const sync_header = 16;

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

/// See `lu_host.HostBufs`.
pub const HostBufs = lu_host.HostBufs;

/// See `lu_host.runHost`.
pub const runHost = lu_host.runHost;

/// See `lu_host.HostRefactor`.
pub const HostRefactor = lu_host.HostRefactor;

/// Private kernels for src/solver/tests.zig; empty outside test builds.
pub const test_access = lu_host.test_access;

/// See `lu_tables.Options`.
pub const Options = lu_tables.Options;

/// See `lu_tables.Tables`.
pub const Tables = lu_tables.Tables;

/// See `lu_tables.build`.
pub const build = lu_tables.build;

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

comptime {
    // Passed to every kernel by value: the layout is the device ABI.
    assert(@sizeOf(Tab) == 208);
}

/// The kernel bodies over the pointer type `Sy.P` (a global pointer on the
/// device, a plain one on the host), `Sy.barrier`/`Sy.pause(spins)` and the
/// done stamps' `Sy.acquire`/`Sy.release`. `sh` is the block's shared
/// scratch (a `*Shared`, in the shared address space on the device);
/// `tid`/`nt` are the lane and lane count (0 and 1 on the host), `nt` at
/// most `lanes`, a power of two for `refactor`'s column-max tree.
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

        /// The back head for the run its ticket names, as `lhead`: a head
        /// row waits only on head rows, because the tail panels (`upanel`)
        /// finish before any head ticket; each row's subtractions still run
        /// in the host's descending k. Sets dx[q[i]] = z[i]; y ends holding
        /// z.
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
                    const ga = gatherUntil(true, sh.a[i - r0], x, val, y + t.t0, t.tl_col, t.tl_slot, sh.e[i - r0], x[t.tl_ptr + i + 1], (j + 1) * panel);
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
                    const ga = gatherUntil(false, sh.a[i - r0], x, val, y + t.t0, t.tu_col, t.tu_slot, sh.e[i - r0], x[t.tu_ptr + i + 1], j * panel);
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

        // ---- private ----

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

        /// `acc` minus val[slot[e]] * yt[col[e]] from entry e on while the
        /// column is on the near side of `bound`: col[e] < bound with the
        /// columns ascending (`below`), col[e] >= bound with them
        /// descending. Skips yt == 0; eight entries' loads go out before
        /// their subtractions.
        inline fn gatherUntil(comptime below: bool, acc0: f64, x: P(u32), val: P(f64), yt: P(f64), col: u32, slot: u32, e0: u32, e1: u32, bound: u32) Ga {
            const S = struct {
                inline fn in(c: u32, b: u32) bool {
                    return if (below) c < b else c >= b;
                }
            };
            const U = 8;
            var acc = acc0;
            var e = e0;
            while (e + U <= e1 and S.in(x[col + e + U - 1], bound)) : (e += U) {
                var yv: [U]f64 = undefined;
                var lv: [U]f64 = undefined;
                inline for (0..U) |u| {
                    yv[u] = yt[x[col + e + u]];
                    lv[u] = val[x[slot + e + u]];
                }
                inline for (0..U) |u| acc = if (yv[u] != 0) acc - lv[u] * yv[u] else acc;
            }
            while (e < e1 and S.in(x[col + e], bound)) : (e += 1) {
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

// The host side lives in lu_host.zig and lu_tables.zig; these re-exports keep
// `solver.lu_kernels` and tests.zig's file import whole. Neither file is
// analyzed by the kernel root, which references none of them.
