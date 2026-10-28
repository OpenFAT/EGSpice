//! GPU kernel root for the device LU: the tables and values the host
//! uploads in, the factors and solutions in device memory out.
//!
//! Exports `arp_lu_refactor` (one block per pivot step, any grid size:
//! blocks take their step by ticket, so a block that starts late never holds
//! a step an earlier one waits on) and `arp_lu_refactor16` (the same over a
//! u16 `dmap`, `Tab.dmap16`), the solves' heads `arp_lu_lhead`/`arp_lu_uhead`
//! (one block per run of rows, by ticket the same way) and their tails
//! `arp_lu_lpanel`/`arp_lu_upanel` (one block of `panel` lanes per panel, by
//! ticket in solve order), plus the resident Newton matrix's
//! `arp_gc_combine`/`arp_gc_gather`/`arp_gc_scatter`. The LU bodies are
//! `kernels/lu.zig`'s, shared with the host threads, so this file holds only
//! the device's synchronization and the export table. It is a kernel image
//! root: no host module imports it.
//! Cites: dev/solvers/gpu-lu.md.
//!
//! Contents: the nine `arp_*` kernel symbols above, and the private `Sy`
//! synchronization interface `kernels/lu.zig` checks by `@hasDecl`:
//! - `pub const P = gompute.GlobalPtr`
//! - `pub inline fn barrier() void`
//! - `pub inline fn pause(_: u32) void`
//! - `pub const acquire = gompute.builtins.loadAcquireDevice`
//! - `pub const release = gompute.builtins.storeReleaseDevice`

const gompute = @import("gompute");
const lu = @import("kernels/lu.zig");
const P = gompute.GlobalPtr;

const Sy = struct {
    /// Kernel arguments live in the global address space.
    pub const P = gompute.GlobalPtr;
    /// `__syncthreads`; every lane of the block must reach it.
    pub inline fn barrier() void {
        gompute.builtins.barrier();
    }
    /// A ~64 ns backoff, whatever the spin count.
    pub inline fn pause(_: u32) void {
        gompute.builtins.spinPause();
    }
    /// Device scope: the done stamps never leave the device, and Zig's own
    /// atomics lower to system scope on NVPTX (7.1 vs 3.7 ms on a refactor).
    pub const acquire = gompute.builtins.loadAcquireDevice;
    /// Device-scope release store, paired with `acquire`.
    pub const release = gompute.builtins.storeReleaseDevice;
};

const K = lu.Kernels(Sy, lu.col_max, lu.refactor_block);
const Solve = lu.Kernels(Sy, 0, 1);

// Block width of the resident Newton matrix kernels below, one thread per
// entry; `analysis/gpu/context.zig` and `gpu/lu.zig` launch at this width.
const gc_block = 256;

// The per-block shared-memory tiles. Module-level `var`s because that is how
// an `addrspace(.shared)` allocation is declared; each block sees its own.
// SAFETY: every kernel writes a tile before any lane of its block reads it.
var sh: K.Shared addrspace(.shared) = undefined;
var ssh: Solve.SolveShared addrspace(.shared) = undefined;
var psh: Solve.PanelShared addrspace(.shared) = undefined;

comptime {
    if (gompute.is_device) {
        gompute.exportRaw("arp_gc_combine", &gcCombine);
        gompute.exportRaw("arp_gc_gather", &gcGather);
        gompute.exportRaw("arp_gc_scatter", &gcScatter);
        gompute.exportRaw("arp_lu_refactor", &refactor);
        gompute.exportRaw("arp_lu_refactor16", &refactor16);
        gompute.exportRaw("arp_lu_lhead", &lhead);
        gompute.exportRaw("arp_lu_uhead", &uhead);
        gompute.exportRaw("arp_lu_lpanel", &lpanel);
        gompute.exportRaw("arp_lu_upanel", &upanel);
    }
}

// ---- private ----

fn refactor(t: lu.Tab, x: P(u32), val: P(f64), a: P(f64), sy: P(u32), stamp: u32) callconv(gompute.kernel_callconv) void {
    _ = K.refactor(u32, t, x, val, a, sy, stamp, &sh, gompute.builtins.localIdX(), lu.refactor_block);
}

fn refactor16(t: lu.Tab, x: P(u32), val: P(f64), a: P(f64), sy: P(u32), stamp: u32) callconv(gompute.kernel_callconv) void {
    _ = K.refactor(u16, t, x, val, a, sy, stamp, &sh, gompute.builtins.localIdX(), lu.refactor_block);
}

fn lhead(t: lu.Tab, x: P(u32), val: P(f64), rhs: P(f64), y: P(f64), sy: P(u32), stamp: u32) callconv(gompute.kernel_callconv) void {
    _ = Solve.lhead(t, x, val, rhs, y, sy, stamp, &ssh, gompute.builtins.localIdX(), lu.solve_block);
}

fn lpanel(t: lu.Tab, x: P(u32), val: P(f64), y: P(f64), sy: P(u32), stamp: u32) callconv(gompute.kernel_callconv) void {
    _ = Solve.lpanel(t, x, val, y, sy, stamp, &psh, gompute.builtins.localIdX(), lu.panel);
}

fn upanel(t: lu.Tab, x: P(u32), val: P(f64), y: P(f64), dx: P(f64), sy: P(u32), stamp: u32) callconv(gompute.kernel_callconv) void {
    _ = Solve.upanel(t, x, val, y, dx, sy, stamp, &psh, gompute.builtins.localIdX(), lu.panel);
}

fn uhead(t: lu.Tab, x: P(u32), val: P(f64), y: P(f64), dx: P(f64), sy: P(u32), stamp: u32) callconv(gompute.kernel_callconv) void {
    _ = Solve.uhead(t, x, val, y, dx, sy, stamp, &ssh, gompute.builtins.localIdX(), lu.solve_block);
}

// `a[i] = g[i] + alpha * c[i]`, uncontracted, as `Circuit.combinePlanes`.
fn gcCombine(n: u64, alpha: f64, g: P(f64), c: P(f64), a: P(f64)) callconv(gompute.kernel_callconv) void {
    const i = gompute.globalIdX(gc_block);
    if (i < n) a[i] = g[i] + alpha * c[i];
}

// `out[i] = g[k[i]]` and `out[n + i] = c[k[i]]`.
fn gcGather(n: u64, k: P(u32), g: P(f64), c: P(f64), out: P(f64)) callconv(gompute.kernel_callconv) void {
    const i = gompute.globalIdX(gc_block);
    if (i >= n) return;
    const s = k[i];
    out[i] = g[s];
    out[n + i] = c[s];
}

// `a[k[i]] = v[i]`: the host's values replace the device's.
fn gcScatter(n: u64, k: P(u32), v: P(f64), a: P(f64)) callconv(gompute.kernel_callconv) void {
    const i = gompute.globalIdX(gc_block);
    if (i < n) a[k[i]] = v[i];
}
