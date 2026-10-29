//! Developer switches, read from the environment once by the facade
//! (`espice.zig` `knobsFromEnv`) and carried by `Config.knobs` into
//! `Circuit.knobs`, so nothing below the facade reads the environment. Each
//! field names its variable; dev/ notes cite them.
//! In: nothing at run time (plain data). Out: the switches.
//! Cites no outside source.
//!
//! Contents:
//! - `pub const Knobs = struct` (`instrument`, `parseBlock`)

const std = @import("std");
const assert = std.debug.assert;
const converger = @import("solver").converger;
const direct = @import("solver").direct;

/// Every developer switch; the defaults are the shipped behaviour.
pub const Knobs = struct {
    newton: converger.Knobs = .{},
    lu: direct.Knobs = .{},
    /// `ESPICE_LANES`: `1` groups ready operating points and `.dc` sweeps
    /// into lane batches (`Session.gather`), `0` runs every query alone;
    /// null leaves it to `executor.lanesOn`.
    lanes: ?bool = null,
    /// `ESPICE_HB_TRACE`: trace harmonic-balance iterates.
    hb_trace: bool = false,
    /// `ZP_NO_QTAPE`: run the LTE over the row plane, not the charge tape.
    no_qtape: bool = false,
    /// `ZP_TRAN_STATS`: print the transient step statistics.
    tran_stats: bool = false,
    /// The device latency bypass (`Hooks.set_bypass`) on host Newton
    /// evaluations; each batch drops it once its hit rate is too low
    /// (`Bypass.review`). Changes values within the Newton tolerance, so a
    /// lane (which never bypasses) equals its scalar run within tolerance,
    /// not bitwise. Off only in tests whose oracle is the plain scalar run.
    bypass: bool = true,
    /// `ZP_TRAN_LTE_TRACE`: diagnose the first 16 rejected LTE estimates
    /// from captured history, without evaluating devices or changing steps.
    tran_lte_trace: bool = false,
    /// `ZP_NO_TANGENTS`: an optimization takes every Jacobian by finite
    /// differences, not from `.dc`/`.tran` tangents, for A/B runs.
    no_tangents: bool = false,
    /// `ESPICE_GPU_LU`: `1` forces the device LU on, any other value off;
    /// null leaves it to the size gate.
    gpu_lu: ?bool = null,
    /// `ESPICE_GPU_LU_CHECK`: check every device solve against the host.
    gpu_lu_check: bool = false,
    /// `ESPICE_GPU_LU_NOGATHER`: refactor without the gather kernel.
    gpu_lu_nogather: bool = false,
    /// `ESPICE_GPU_LU_TAIL`: dense-tail columns (0 when unparsable).
    gpu_lu_tail: ?u32 = null,
    /// `ESPICE_GPU_LU_BENCH` (`-Dlu-instrument`): time the device LU,
    /// stopping after this many samples (0, or unparsable: never).
    gpu_lu_bench: ?usize = null,
    /// `ESPICE_GPU_LU_THREADS`: host refactor threads the bench compares.
    gpu_lu_threads: u32 = 8,
    /// `ESPICE_GPU_STATS`: set, or `phases` for the per-phase wall split.
    gpu_stats: enum { off, on, phases } = .off,
    /// `ESPICE_GPU_BLOCK`: the CUDA block width, 1..1024.
    gpu_block: u32 = 64,
    /// `ESPICE_GPU_EVAL_CHECK`: replay every device eval at the same x and
    /// report the worst plane cell that moved (`GpuContext.evalCheck`).
    gpu_eval_check: bool = false,
    /// `ESPICE_GPU_NOFUSE`: give every device eval its own host wait instead
    /// of sharing the limit pass's (`GpuContext.evalFollows`), for A/B runs.
    gpu_nofuse: bool = false,
    /// `ESPICE_GPU_NOGRAPHS`: use direct kernel launches for graph A/B runs.
    gpu_nographs: bool = false,
    /// `ESPICE_GPU_HOST_STEP`: GPU transient lanes keep step control on the
    /// host (`tran.Sim` per lane), the bitwise oracle for device step control.
    gpu_host_step: bool = false,
    /// `XDG_CACHE_HOME` and `HOME`, where the GPU JIT warm markers live.
    xdg_cache_home: ?[]const u8 = null,
    home: ?[]const u8 = null,

    /// `-Dlu-instrument`: `lu.stats`, `gpu_lu_bench` and `gpu_lu_threads`
    /// are read; otherwise the facade leaves them unset.
    pub const instrument = direct.instrument;

    /// Returns `ESPICE_GPU_BLOCK`'s width, or 64 when unset, unparsable or
    /// outside CUDA's 1..1024: a zero width would divide by zero in
    /// `Dim3.linear`.
    pub fn parseBlock(env: ?[]const u8) u32 {
        const v = std.fmt.parseInt(u32, env orelse return block_default, 10) catch return block_default;
        const w = if (v >= 1 and v <= 1024) v else block_default;
        assert(w >= 1);
        return w;
    }

    const block_default: u32 = 64;
};

test "Knobs.parseBlock" {
    const parseBlock = Knobs.parseBlock;
    try std.testing.expectEqual(@as(u32, 64), parseBlock(null));
    try std.testing.expectEqual(@as(u32, 32), parseBlock("32"));
    try std.testing.expectEqual(@as(u32, 1), parseBlock("1"));
    try std.testing.expectEqual(@as(u32, 1024), parseBlock("1024"));
    for ([_][]const u8{ "0", "1025", "", "-8", "64x", "99999999999" }) |bad|
        try std.testing.expectEqual(@as(u32, 64), parseBlock(bad));
}
