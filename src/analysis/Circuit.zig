//! The analysis-side circuit: a frozen CSC pattern plus four value planes
//! (g = dI/dx, c = dQ/dx, rhs = I, q = Q) that one device pass fills. Every
//! analysis is an affine consumer of the planes: DC solves G, transient
//! G + a*C, AC G + jwC. In: the prepared `device.Circuit` template and an
//! x. Out: the planes, the device state walks the solvers drive, and the
//! routing to an attached ParEval, GPU context or device LU.
//!
//! Cites: ngspice resdefs.h/resload.c (`ac=`), hfetload.c (the in-load
//! current test), CKTacLoad; VAMS LRM §9.4 (display tasks), §9.15
//! (`$simparam`), §9.17.2 (`$bound_step`); dev/devices/models.md;
//! analysis/certify.zig (#53).

const std = @import("std");
const device_ir = @import("device").abi;
const Prepared = @import("device").Circuit;
const progress_api = @import("worker.zig");
const par_eval = @import("par_eval.zig");
const gpu_mod = @import("gpu/context.zig");
const GpuContext = gpu_mod.GpuContext;
const GpuLu = @import("gpu/lu.zig").GpuLu;
const converger = @import("solver").converger;
const WsPool = @import("ws_pool.zig").WsPool;
const freq_solve = @import("solver").freq_solve;
const direct = @import("solver").direct;
const numerics = @import("core").numerics;

const Batch = device_ir.Batch;
const Planes = device_ir.Planes;
const ParamRef = device_ir.ParamRef;
const NoiseSource = device_ir.NoiseSource;
const StateCtlOp = device_ir.StateCtlOp;
const ParEval = par_eval.ParEval;
const BbdInfo = numerics.BbdInfo;

const zeroSimd = numerics.zeroSimd;
const copySimd = numerics.copySimd;

/// Developer switches, read from the environment once by the facade
/// (`espice.zig` `knobsFromEnv`) and carried by `Config.knobs` into
/// `Circuit.knobs`, so nothing below the facade reads the environment. Each
/// field names its variable; dev/ notes cite them.
pub const Knobs = struct {
    /// `-Dlu-instrument`: `lu.stats`, `gpu_lu_bench` and `gpu_lu_threads`
    /// are read; otherwise the facade leaves them unset.
    pub const instrument = direct.instrument;

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

    /// `ESPICE_GPU_BLOCK`'s width, or 64 when unset, unparsable or outside
    /// CUDA's 1..1024: a zero width would divide by zero in `Dim3.linear`.
    pub fn parseBlock(env: ?[]const u8) u32 {
        const v = std.fmt.parseInt(u32, env orelse return 64, 10) catch return 64;
        return if (v >= 1 and v <= 1024) v else 64;
    }
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

/// The KCL certificate (`zig build -Dcertify`, #53): at every point direct
/// Newton converges on and stages, re-evaluate the devices at the
/// published x with the limiters cleared and check the result against the
/// solver's own promise.
///
/// Asserted: the converger's per-row residual gate
/// (`converger.residualGate`, with the same gmin and force loads). A row
/// over it panics, naming the row, so a corpus run fails that deck.
/// Reported: the current bound ||F_kcl||_inf <= abstol + reltol * max|I|,
/// where F_kcl is every non-branch row and max|I| the largest branch
/// current in x. The circuit keeps the worst ratio and prints it once, on
/// release. The gate is looser than this bound by design (dev/analysis/
/// tolerance-system.md, "KCL certificate"), so the bound only reports.
///
/// Off by default, and comptime-gated: without `-Dcertify` the `Tally`
/// field is void and `newton` never calls `check`.
pub const kcl_cert = struct {
    /// Whether this build certifies, from `-Dcertify`.
    pub const enabled = @import("build_options").certify;

    /// One circuit's certificate record: the worst current-bound ratio, where
    /// it fell, and the point counts. `scratch` holds the planes `check`
    /// restores, sized on first use and freed by `deinit`.
    pub const Tally = struct {
        points: u64 = 0,
        /// Points whose ratio exceeded 1.
        over: u64 = 0,
        worst: f64 = 0,
        worst_t: f64 = 0,
        worst_row: u32 = 0,
        worst_kind: []const u8 = "",
        scratch: []f64 = &.{},

        /// Prints the record when any point was certified, then frees `scratch`.
        pub fn deinit(self: *Tally, gpa: std.mem.Allocator, node_name: []const u8) void {
            if (self.points > 0) std.debug.print(
                "certify: {d} points, {d} over abstol+reltol*max|I|, worst |F|/bound = {e:.3} ({s} t={e:.3} row {d} {s})\n",
                .{ self.points, self.over, self.worst, self.worst_kind, self.worst_t, self.worst_row, node_name },
            );
            gpa.free(self.scratch);
            self.* = .{};
        }
    };

    /// What `measure` found at one point.
    pub const Point = struct {
        /// The first row over the converger's gate, or null when every row
        /// passes it.
        gate_row: ?u32,
        /// |F| and the gate on `gate_row`; zero when it is null.
        gate_f: f64 = 0,
        gate_tol: f64 = 0,
        /// ||F_kcl||_inf / (abstol + reltol * max|I|), and its row.
        ratio: f64,
        kcl_row: u32,
    };

    /// The f64 count of `scratch` that `measure` needs for `ckt`.
    pub fn scratchLen(ckt: anytype) usize {
        return ckt.g_vals.len + ckt.c_vals.len + ckt.rhs.len + ckt.q_vec.len;
    }

    /// Re-evaluates `ckt` at the converged `x` through `hook`, with the
    /// limiters cleared and `opts`' gmin and force loads, and measures it
    /// against the gate and the current bound (see the file comment). Leaves
    /// the planes and `q_fresh` as it found them, but the limiters cleared: a
    /// transient carries limiter history from point to point, so a certified
    /// run can take other steps than an uncertified one.
    /// `scratch.len` must be `scratchLen(ckt)`.
    pub fn measure(ckt: anytype, hook: anytype, x: []const f64, t: f64, opts: converger.Options, scratch: []f64) Point {
        const n: usize = ckt.n;
        std.debug.assert(x.len == n);
        std.debug.assert(scratch.len == scratchLen(ckt));
        const planes = .{ "g_vals", "c_vals", "rhs", "q_vec" };
        var off: usize = 0;
        inline for (planes) |p| {
            const src = @field(ckt, p);
            @memcpy(scratch[off..][0..src.len], src);
            off += src.len;
        }
        const q_fresh = ckt.q_fresh;

        ckt.clearLimits();
        hook.assemble(ckt, x, t);
        const v = hook.vals(ckt);
        converger.loadExtras(ckt, v, x, opts);
        var pt: Point = .{ .gate_row = null, .ratio = 0, .kcl_row = 0 };
        var kcl: f64 = 0;
        var i_max: f64 = 0;
        for (0..n) |i| {
            const f = @abs(ckt.rhs[i]);
            const gate = converger.residualGate(opts, v[ckt.diag_slots[i]], x[i]);
            // Negated, so a NaN residual fails too.
            if (!(f <= gate) and pt.gate_row == null) {
                pt.gate_row = @intCast(i);
                pt.gate_f = f;
                pt.gate_tol = gate;
            }
            if (ckt.current_row[i]) {
                i_max = @max(i_max, @abs(x[i]));
            } else if (!(f <= kcl)) {
                kcl = f;
                pt.kcl_row = @intCast(i);
            }
        }
        pt.ratio = kcl / (opts.abstol + opts.reltol * i_max);

        ckt.clearLimits();
        off = 0;
        inline for (planes) |p| {
            const dst = @field(ckt, p);
            @memcpy(dst, scratch[off..][0..dst.len]);
            off += dst.len;
        }
        ckt.q_fresh = q_fresh;
        std.debug.assert(off == scratch.len);
        return pt;
    }

    /// Sizes `ckt.cert.scratch` for `check`. `newton` calls it, through
    /// `Circuit.certifyReserve`, on entry under `-Dcertify`, where an
    /// allocation failure still returns to its caller.
    pub fn reserve(ckt: anytype) error{OutOfMemory}!void {
        comptime std.debug.assert(enabled);
        const tally = &ckt.cert;
        const need = scratchLen(ckt);
        if (tally.scratch.len == need) return;
        ckt.gpa.free(tally.scratch);
        tally.scratch = &.{};
        tally.scratch = try ckt.gpa.alloc(f64, need);
        std.debug.assert(tally.scratch.len == need);
    }

    /// `measure`, then panics when a row is over the converger's gate, and
    /// records the ratio in `ckt.cert`. `newton` calls it, through
    /// `Circuit.certifyPoint`, on every converged x it stages under
    /// `-Dcertify`, after every gate and before `updateStates` stages the
    /// point: a device whose eval reads its staged history would otherwise see
    /// x twice. Asserts that `reserve` sized `ckt.cert.scratch`.
    pub fn check(ckt: anytype, hook: anytype, x: []const f64, t: f64, opts: converger.Options) void {
        comptime std.debug.assert(enabled);
        const tally = &ckt.cert;
        std.debug.assert(tally.scratch.len == scratchLen(ckt));
        const pt = measure(ckt, hook, x, t, opts, tally.scratch);
        if (pt.gate_row) |row| std.debug.panic(
            "certify: row {d} ({s}) |F| = {e:.3} exceeds the converger's gate {e:.3} at {s} t={e:.6}, x = {e:.6}",
            .{ row, ckt.nodeName(row), pt.gate_f, pt.gate_tol, @tagName(ckt.sim.kind), t, x[row] },
        );
        tally.points += 1;
        if (pt.ratio > 1) tally.over += 1;
        if (!(pt.ratio <= tally.worst)) {
            tally.worst = pt.ratio;
            tally.worst_t = t;
            tally.worst_row = pt.kcl_row;
            tally.worst_kind = @tagName(ckt.sim.kind);
        }
        std.debug.assert(tally.over <= tally.points);
    }
};

/// Row and column index of the ground node in every pattern.
pub const GROUND: u32 = 0;

/// A card parameter with a separate frequency-domain value, such as a
/// resistor's `ac=`. Matches ngspice's RESacResist/RESacConduct
/// (resdefs.h:48-49, stamped by resload.c:60-62): it moves .ac/.sp/.noise/.pz
/// and leaves .op/.dc/.tran alone.
pub const AcParam = struct {
    ptr: ParamRef,
    ac_value: f64,
    /// The DC value, saved across one AC linearization.
    saved: f64 = 0,
};

/// Newton hook for a plain DC solve: assemble is one eval, the matrix is G.
pub const EvalHook = struct {
    /// Stamps the Newton planes at `x`, seeded from the baseline when one exists.
    pub fn assemble(_: EvalHook, ckt: *Circuit, x: []const f64, t: f64) void {
        ckt.evalNewton(x, t);
    }
    /// Returns G, the matrix `assemble` just filled. Borrowed from `ckt`.
    pub fn vals(_: EvalHook, ckt: *Circuit) []f64 {
        return ckt.g_vals;
    }
    /// Returns G at plane index `slot`, which must be below `nnz + 1`.
    pub fn diagAt(_: EvalHook, ckt: *Circuit, slot: u32) f64 {
        return ckt.g_vals[slot];
    }
};

/// The LRM §9.4 display output of the devices with `Hooks.say` (#102),
/// executor-owned. `updateStates` renders each converged point's lines into
/// `staged`; `stateCtl(.commit)` writes them to `out` and `.revert` drops
/// them, so only accepted points print, each once.
pub const Display = struct {
    /// Card name of each instance, by batch; empty for a batch without `say`.
    names: []const []const []const u8,
    staged: std.Io.Writer.Allocating,
    out: *std.Io.Writer,
};

/// Frozen topology plus the mutable device and solver state of one query.
///
/// One per query, read through a pointer once per eval or solve, never per
/// entry: the hot data is the slices it points at (pattern, planes,
/// batches), each a contiguous SoA column. Zig reorders an auto-layout
/// struct's fields, so declaration order here is for the reader only.
pub const Circuit = struct {
    col_ptr: []u32,
    row_idx: []u32,
    nnz: u32,
    /// Plane index that absorbs ground rows and structurally dead entries.
    /// Equals `nnz`, so every plane has `nnz + 1` entries.
    trash_slot: u32,
    /// Unknown count; `rhs` and `q_vec` have `n + 1` entries, `[n]` is trash.
    n: u32,

    g_vals: []f64,
    c_vals: []f64,
    rhs: []f64,
    q_vec: []f64,

    diag_slots: []u32,
    batches: []Batch,
    /// Library type of each batch, parallel to `batches`. Borrowed topology.
    batch_types: []const device_ir.DeviceType,
    /// True where unknown i is an MNA branch current (KVL row), false for a
    /// node voltage. Read by the converger's per-row tolerance.
    current_row: []bool,

    has_charge: bool,
    /// Some charge-carrying batch advances device state at the accepted step
    /// (freeze_grad latches, absdelay rings). The transient must then re-read q
    /// under the committed state before recording it, or the next step opens
    /// on a residual a*dq that doubles every time dt halves.
    has_state_q: bool,
    /// `g_base`/`c_base` hold the constant-Jacobian contribution and
    /// `evalNewton` seeds from them.
    has_baseline: bool,

    g_base: []f64,
    c_base: []f64,

    /// Capacitor-only nodes have a charge-defined startup, not a unique static OP.
    needs_tran_op: bool = false,
    /// See `device.Circuit.vpi`.
    vpi: ?*const @import("device").Library.Vpi = null,

    /// Node names as one flat table: `nodeName(i)` is
    /// `intern_bytes[intern_offs[i]..intern_offs[i+1]]`, empty for a branch
    /// unknown. `intern_offs` has `n + 1` entries.
    intern_bytes: []u8,
    intern_offs: []u32,

    bbd: ?BbdInfo = null,
    /// The template's shared column-order slot (`device.Circuit.column_order`),
    /// borrowed; the pattern is the template's.
    column_order: ?*Prepared.ColumnOrder = null,
    /// CSC slot of each frequency-dependent small-signal entry, the trash
    /// slot for a ground one (`Prepared.ac_dyn_slots`); `acDyn` fills their
    /// terms. Borrowed topology.
    ac_dyn_slots: []const u32 = &.{},
    solver_execution: numerics.Execution = .{},
    /// The facade's developer switches (`Knobs`), set by the executor.
    knobs: Knobs = .{},
    /// Executor-owned threaded stamp, borrowed: set it to null before
    /// freeing it. Null means serial eval.
    par_eval: ?*ParEval = null,
    /// `setBypass`'s tolerances while the bypass is on.
    bypass_tol: ?device_ir.BypassTol = null,
    /// The executor's GPU evaluation context, borrowed while attached
    /// (`attachGpu`); null means every batch stamps on the host.
    gpu: ?*GpuContext = null,
    /// The executor's device LU that `deviceSolve` routes through, borrowed:
    /// set it to null before freeing it. Independent of `gpu`: a deck whose
    /// devices all evaluate on the host can still factor on the device. Null
    /// means the host factors.
    lu: ?*GpuLu = null,
    /// Where `newtonGC` finds A after the last eval (`evalNewtonGC`, 02
    /// 2.3), and the alpha that eval formed it at.
    gc: gpu_mod.GcServed = .full,
    gc_alpha: f64 = 0,
    /// Executor-owned display output; null when no batch has `Hooks.say`.
    /// Borrowed: set it to null before freeing it.
    display: ?*Display = null,
    /// The temperature `setCircuitTemp` last installed, in degrees Celsius;
    /// the devices' own 27 until then. A sweep restores it from here.
    temp_c: f32 = 27,
    /// The analysis state every device call receives (`setSimState`), with
    /// the Newton iteration `beginSolve`/`advanceIteration` count.
    sim: device_ir.SimState = .{},
    gpa: std.mem.Allocator,
    /// False when the pattern, tapes and names are borrowed from a template.
    owns_topology: bool = true,
    /// False when `batches` are the template's own (`instantiateMove`): the
    /// template frees them.
    owns_batches: bool = true,
    /// Where `checkpoint` reports; null reports nowhere.
    progress: ?progress_api.Callback = null,

    /// "The planes hold the linearization at this x_op." Set by `linearize`,
    /// cleared by every plane or device-parameter writer. `x_ptr` is an
    /// identity key and is never dereferenced: x_op is one arena slice that
    /// the executor hands read-only to each consumer.
    lin: struct { x_ptr: [*]const f64 = undefined, len: u32 = 0, valid: bool = false } = .{},
    /// `updateStates` ran since the last `stateCtl(.commit)` or `.revert`.
    state_staged: bool = false,
    /// Set by the transient. `updateStates` then records a device's
    /// `request_reject_at` in `reject_at`, for the step to be retried
    /// ending there, instead of returning it as a flip that forces another
    /// iterate at the same time.
    land_rejects: bool = false,
    /// The earliest rejection time the last `updateStates` recorded under
    /// `land_rejects`, or null. The transient clears it before an attempt
    /// that may fail without staging any.
    reject_at: ?f64 = null,
    /// `q_vec` and the q tapes hold the charges at the x the last
    /// `updateStates` staged, which computed them on the way (`acceptStates`).
    /// Cleared by every plane reset.
    q_fresh: bool = false,

    /// Symbolic LU and Newton scratch, built on first use and shared by every
    /// analysis. The pattern is frozen, so it stays valid for the lifetime.
    ws: ?converger.Workspace = null,
    /// Where `release` returns `ws` and `workspace` takes one from, shared
    /// by the Problem's queries; borrowed. Null builds and frees its own.
    ws_pool: ?*WsPool = null,
    /// `loadCheck`'s rows and last iterate, built with `ws`; null when no
    /// batch marks rows (`Hooks.mark_load_check_rows`).
    load_check: ?LoadCheck = null,
    /// `collectParams` memo. The refs point into frozen batch storage.
    param_refs: ?[]ParamRef = null,
    /// The template's `Prepared.param_starts`, borrowed; empty when it
    /// has none.
    param_starts: []const u32 = &.{},
    /// Parameters with a frequency-domain value, which `linearizeAc` swaps
    /// to their AC values. Set by the executor; owned by the query work
    /// arena, which outlives every `linearizeAc`. Empty on every deck
    /// without one.
    ac_params: []AcParam = &.{},
    /// The KCL certificate's record under `-Dcertify`; void otherwise.
    cert: if (kcl_cert.enabled) kcl_cert.Tally else void = if (kcl_cert.enabled) .{} else {},

    /// `newton` calls `certifyPoint` on every converged x it stages when true.
    pub const certify = kcl_cert.enabled;

    /// Sizes the certificate's scratch (`kcl_cert.reserve`). Only `newton`
    /// calls it, on entry, and only under `-Dcertify`.
    pub fn certifyReserve(self: *Circuit) error{OutOfMemory}!void {
        return kcl_cert.reserve(self);
    }

    /// Certifies the converged `x` (analysis/certify.zig). Only `newton`
    /// calls it, and only under `-Dcertify`.
    pub fn certifyPoint(self: *Circuit, hook: anytype, x: []const f64, t: f64, opts: converger.Options) void {
        kcl_cert.check(self, hook, x, t, opts);
    }

    /// `newton`'s device path: factors `vals` and solves dx = -A^-1 rhs on
    /// the device. Null without a device LU; see `GpuLu.solve` for true and
    /// false.
    pub fn deviceSolve(self: *Circuit, slv: *direct.Solver, vals: []f64, dx: []f64) ?bool {
        const lu = self.lu orelse return null;
        if (self.gc == .device) if (self.gpuContext()) |g| return g.gcSolve(lu, slv, vals, self.rhs[0..self.n], dx, self.gc_alpha);
        return lu.solve(slv, vals, self.rhs[0..self.n], dx);
    }

    /// Returns the shared Newton/JFNK workspace, building it (and the
    /// `loadCheck` rows) on first use. The pointer stays valid until
    /// `release`; the circuit owns the workspace.
    pub fn workspace(self: *Circuit) !*converger.Workspace {
        if (self.ws == null) {
            self.load_check = try LoadCheck.init(self.gpa, self.batches, self.col_ptr, self.row_idx[0..self.nnz]);
            errdefer if (self.load_check) |*lc| {
                lc.deinit(self.gpa);
                self.load_check = null;
            };
            if (self.pooled()) |p| if (p.take(@intFromPtr(self.col_ptr.ptr), self.knobs.newton, self.knobs.lu)) |w| {
                var ws = w;
                ws.reset(self.bbd) catch |e| {
                    ws.deinit(self.gpa);
                    return e;
                };
                self.ws = ws;
                return &self.ws.?;
            };
            const shared = if (self.column_order) |o| o.get(self.n) else null;
            var ws = try converger.Workspace.initOrdered(self.gpa, self.n, self.col_ptr, self.row_idx, self.bbd, self.knobs.newton, self.knobs.lu, shared);
            errdefer ws.deinit(self.gpa);
            if (shared == null) if (self.column_order) |o| if (ws.slv.q.len == self.n) try o.share(ws.slv.q);
            self.ws = ws;
        }
        return &self.ws.?;
    }

    /// ngspice's in-load current test for the models that keep it
    /// (hfetload.c:389): each current the model drives into one of its
    /// internal nodes must land within reltol·m + abstol of its linear
    /// prediction from the previous iterate, `i_old + J_old·(x − x_old)`,
    /// where m is the largest branch flow into the node (ngspice's cg or
    /// cd). `vals`/`rhs` are the Newton matrix and residual assembled
    /// at `x`, companion terms included, so the row test sees the same
    /// currents ngspice's cg/cd carry; the flow and resistor terms are
    /// linear and cancel. True when every row passes; false on the first
    /// iterate of a solve, which has no previous one. Records `x` as the
    /// previous iterate either way.
    ///
    /// ponytail: hfet1's cdhat leaves out the -ggdpp*delvgdpp its cd
    /// carries (hfetload.c:211-216), a first-order miss on every gate-drain
    /// swing that makes ngspice iterate more; a row test cannot see
    /// branches, so that quirk is not reproduced. dev/devices/models.md.
    pub fn loadCheck(self: *Circuit, x: []const f64, vals: []const f64, rhs: []const f64, reltol: f64, abstol: f64) bool {
        const lc = if (self.load_check) |*l| l else return true;
        var ok = lc.valid;
        for (lc.rows, 0..) |r, k| {
            var pred = lc.i_old[k];
            var mag: f64 = 0;
            for (lc.ptr[k]..lc.ptr[k + 1]) |e| {
                const c = lc.col[e];
                pred += lc.v_old[e] * (x[c] - lc.x_old[e]);
                if (self.current_row[c]) mag = @max(mag, @abs(x[c]));
                lc.v_old[e] = vals[lc.slot[e]];
                lc.x_old[e] = x[c];
            }
            const i = rhs[r];
            if (mag == 0) mag = @max(@abs(i), @abs(pred));
            if (!(@abs(i - pred) <= reltol * mag + abstol)) ok = false;
            lc.i_old[k] = i;
        }
        lc.valid = true;
        return ok;
    }

    /// `ws_pool` when this circuit's workspaces come from the same allocator.
    fn pooled(self: *const Circuit) ?*WsPool {
        const p = self.ws_pool orelse return null;
        return if (p.gpa.ptr == self.gpa.ptr and p.gpa.vtable == self.gpa.vtable) p else null;
    }

    // The executor's setup. None touches the planes or a device parameter,
    // so none clears `lin`.

    /// Installs the query's solver threading and the facade's developer
    /// switches. Asserts that `workspace` has not run yet, since it reads
    /// `knobs`.
    pub fn configure(self: *Circuit, execution: numerics.Execution, knobs: Knobs) void {
        std.debug.assert(self.ws == null);
        self.solver_execution = execution;
        self.knobs = knobs;
    }

    /// Evaluates the resident batches through `g` from now on and moves the
    /// planes into its page-locked buffers, which `GpuContext.init` filled
    /// with their values. `g` is borrowed: `detachGpu` before freeing it.
    /// Asserts that no context is attached.
    pub fn attachGpu(self: *Circuit, g: *GpuContext) void {
        std.debug.assert(self.gpu == null);
        self.gpu = g;
        self.bindPlanes(g.planes());
    }

    /// The attached GPU context, or null; comptime null in a build without
    /// kernel images (`gpu/context.zig` `available`), so every GPU branch below
    /// folds away there.
    inline fn gpuContext(self: *const Circuit) ?*GpuContext {
        return if (comptime gpu_mod.available) self.gpu else null;
    }

    /// Returns the planes to the circuit's own allocations, carrying their
    /// last values, and evaluates on the host again. No-op when detached.
    pub fn detachGpu(self: *Circuit) void {
        const g = self.gpu orelse return;
        self.gpu = null;
        self.bindPlanes(g.restorePlanes());
    }

    /// Reports progress to the owning worker. Fails only when the query was
    /// cancelled.
    pub fn checkpoint(self: *Circuit, event: progress_api.Event) error{QueryCancelled}!void {
        if (self.progress) |callback| try callback.checkpoint(event);
    }

    /// Returns a circuit that shares `template`'s immutable topology and owns
    /// fresh device and solver state. `template` must outlive it and must never
    /// have been evaluated. Free it with `deinit`.
    pub fn instantiate(template: *const Prepared, allocator: std.mem.Allocator) !Circuit {
        const batches = try copyBatches(allocator, template.batches, "instantiate");
        errdefer freeBatches(allocator, batches);
        return allocate(template.*, allocator, batches, false);
    }

    /// `instantiate` for the last query that will ever read `template`'s
    /// device state: the circuit evaluates the template's own batches in
    /// place instead of copying them. A never-evaluated template already
    /// holds the state `instantiate` would copy, and it keeps ownership.
    pub fn instantiateMove(template: *const Prepared, allocator: std.mem.Allocator) !Circuit {
        var ckt = try allocate(template.*, allocator, template.batches, false);
        ckt.owns_batches = false;
        return ckt;
    }

    /// Returns a copy of `source`'s device state over `template`'s topology,
    /// for a query that starts from a completed operating point. Fails with
    /// `error.IncompatibleDependency` when `source` was not built from
    /// `template`. Free it with `deinit`.
    pub fn fromSnapshot(template: *const Prepared, source: *const Circuit, allocator: std.mem.Allocator) !Circuit {
        if (source.n != template.n or source.col_ptr.ptr != template.col_ptr.ptr)
            return error.IncompatibleDependency;
        const batches = try copyBatches(allocator, source.batches, "snapshot");
        errdefer freeBatches(allocator, batches);
        var ckt = try allocate(template.*, allocator, batches, false);
        ckt.sim = source.sim;
        ckt.temp_c = source.temp_c;
        return ckt;
    }

    /// `fromSnapshot` for the last query that will ever read `source`: the
    /// device state moves instead of being copied, and `source` keeps no
    /// batches and frees the rest of its state at once (its `deinit` is then
    /// a no-op). Copies when the two circuits do not share an allocator.
    pub fn fromSnapshotMove(template: *const Prepared, source: *Circuit, allocator: std.mem.Allocator) !Circuit {
        if (source.n != template.n or source.col_ptr.ptr != template.col_ptr.ptr)
            return error.IncompatibleDependency;
        if (source.owns_batches and (source.gpa.ptr != allocator.ptr or source.gpa.vtable != allocator.vtable))
            return fromSnapshot(template, source, allocator);
        var ckt = try allocate(template.*, allocator, source.batches, false);
        ckt.owns_batches = source.owns_batches;
        ckt.sim = source.sim;
        ckt.temp_c = source.temp_c;
        source.batches = &.{};
        source.release();
        return ckt;
    }

    fn allocate(data: Prepared, allocator: std.mem.Allocator, batches: []Batch, owns_topology: bool) !Circuit {
        const g_vals = try allocator.alloc(f64, @as(usize, data.nnz) + 1);
        errdefer allocator.free(g_vals);
        const c_vals = try allocator.alloc(f64, @as(usize, data.nnz) + 1);
        errdefer allocator.free(c_vals);
        const rhs = try allocator.alloc(f64, @as(usize, data.n) + 1);
        errdefer allocator.free(rhs);
        const q_vec = try allocator.alloc(f64, @as(usize, data.n) + 1);
        @memset(c_vals, 0);
        @memset(q_vec, 0);
        // `Batch.digital` comes from the Library, not the device, so a batch
        // rebuilt by `instantiate`/`snapshot` takes it from the template.
        for (batches, data.batches) |*b, t| b.digital = t.digital;
        return .{
            .gpa = allocator,
            .col_ptr = data.col_ptr,
            .row_idx = data.row_idx,
            .nnz = data.nnz,
            .trash_slot = data.nnz,
            .n = data.n,
            .g_vals = g_vals,
            .c_vals = c_vals,
            .rhs = rhs,
            .q_vec = q_vec,
            .diag_slots = data.diag_slots,
            .batches = batches,
            .batch_types = data.batch_types,
            .current_row = data.current_row,
            .has_charge = data.has_charge,
            .has_state_q = data.has_state_q,
            .has_baseline = false,
            .g_base = &.{},
            .c_base = &.{},
            .needs_tran_op = data.needs_tran_op,
            .vpi = data.vpi,
            .intern_bytes = data.intern_bytes,
            .intern_offs = data.intern_offs,
            .bbd = data.bbd,
            .ac_dyn_slots = data.ac_dyn_slots,
            .param_starts = data.param_starts,
            .column_order = data.column_order,
            .owns_topology = owns_topology,
        };
    }

    /// `release`, then poisons the struct.
    pub fn deinit(self: *Circuit) void {
        self.release();
        self.* = undefined;
    }

    /// Frees all state the circuit owns and leaves an empty one whose
    /// `release` and `deinit` free nothing more. Nothing may evaluate or
    /// solve it afterwards.
    pub fn release(self: *Circuit) void {
        const gpa = self.gpa;
        if (self.knobs.tran_stats) for (self.batches) |b| {
            const count = b.hooks.derivative_counts orelse continue;
            const counts = count(b.ctx);
            std.debug.print("derivative-stats: model={s} wide={d} narrow={d} wide_lanes={d} narrow_lanes={d}\n", .{ b.type_name, counts.wide, counts.narrow, counts.wide_lanes, counts.narrow_lanes });
        };
        if (kcl_cert.enabled) self.cert.deinit(gpa, self.nodeName(self.cert.worst_row));
        if (self.ws) |*w| {
            if (self.pooled()) |p| p.give(@intFromPtr(self.col_ptr.ptr), self.knobs.newton, self.knobs.lu, w.*) else w.deinit(gpa);
        }
        self.ws = null;
        if (self.load_check) |*lc| lc.deinit(gpa);
        self.load_check = null;
        if (self.param_refs) |refs| gpa.free(refs);
        self.param_refs = null;
        if (self.owns_batches) freeBatches(gpa, self.batches);
        self.batches = &.{};
        // g_base/c_base unconditionally: a failed computeBaseline can leave
        // them allocated with has_baseline false.
        inline for (.{ "g_vals", "c_vals", "g_base", "c_base", "rhs", "q_vec" }) |name| {
            gpa.free(@field(self, name));
            @field(self, name) = &.{};
        }
        self.has_baseline = false;
        if (self.owns_topology) {
            gpa.free(self.col_ptr);
            gpa.free(self.row_idx);
            gpa.free(self.diag_slots);
            gpa.free(self.current_row);
            gpa.free(self.batch_types);
            if (self.bbd) |bbd| gpa.free(bbd.blocks);
            gpa.free(self.ac_dyn_slots);
            gpa.free(self.intern_bytes);
            gpa.free(self.intern_offs);
            gpa.free(self.param_starts);
            if (self.column_order) |o| o.destroy(self.n);
            self.column_order = null;
            self.owns_topology = false;
        }
    }

    /// Device state held outside the circuit, one `snapshot` per batch, so
    /// an analysis that must rewind time can restore what it committed.
    pub const SavedState = struct {
        batches: []Batch,

        /// Frees every snapshot; `gpa` must be the allocator `saveState` took.
        pub fn deinit(self: SavedState, gpa: std.mem.Allocator) void {
            freeBatches(gpa, self.batches);
        }
    };

    /// Returns a copy of every batch's device state. The caller frees it
    /// with `SavedState.deinit`.
    pub fn saveState(self: *const Circuit, gpa: std.mem.Allocator) !SavedState {
        return .{ .batches = try copyBatches(gpa, self.batches, "snapshot") };
    }

    /// Overwrites `saved` with the current device state, allocating nothing.
    pub fn storeState(self: *const Circuit, saved: SavedState) void {
        for (saved.batches, self.batches) |dst, src| dst.hooks.copy_state(dst.ctx, src.ctx);
    }

    /// Rewinds every batch to `saved`, which becomes the accepted state.
    /// Batch storage and every `ParamRef` into it stay valid. Does nothing
    /// while a GPU context is attached.
    // ponytail: host batches only. Under a GPU hook the resident state is
    // left where it is; download it (`syncHostState`) and re-upload states
    // too when a GPU-resident device carries history that must rewind.
    pub fn restoreState(self: *Circuit, saved: SavedState) void {
        if (self.gpuContext() != null) return;
        for (self.batches, saved.batches) |dst, src| dst.hooks.copy_state(dst.ctx, src.ctx);
        self.state_staged = false;
        self.lin.valid = false;
    }

    /// Fails with `error.DigitalDeviceUnsupported` when a device type runs
    /// a digital engine (`Batch.digital`), for an analysis that rewinds
    /// time (pss, qpss, hb, envelope): a committed digital time cannot move
    /// back. A warning names the analysis and the type.
    pub fn refuseDigital(self: *const Circuit, analysis: []const u8) error{DigitalDeviceUnsupported}!void {
        for (self.batches) |b| if (b.digital) {
            std.log.warn("{s}: device type '{s}' holds digital state, whose time cannot rewind", .{ analysis, b.type_name });
            return error.DigitalDeviceUnsupported;
        };
    }

    /// Points the four planes at `planes`, values as they stand there: the
    /// GPU context's page-locked buffers while it lives, the circuit's own
    /// allocations again when it goes. Asserts the lengths match.
    fn bindPlanes(self: *Circuit, planes: Planes) void {
        std.debug.assert(planes.g_vals.len == self.g_vals.len);
        std.debug.assert(planes.c_vals.len == self.c_vals.len);
        std.debug.assert(planes.rhs.len == self.rhs.len);
        std.debug.assert(planes.q_vec.len == self.q_vec.len);
        self.g_vals = planes.g_vals;
        self.c_vals = planes.c_vals;
        self.rhs = planes.rhs;
        self.q_vec = planes.q_vec;
    }

    /// Returns a view of this circuit's own value planes.
    pub fn ownPlanes(self: *Circuit) Planes {
        return .{ .g_vals = self.g_vals, .c_vals = self.c_vals, .rhs = self.rhs, .q_vec = self.q_vec };
    }

    /// Returns the length of the per-device-state charge tape: one entry per
    /// (instance, LTE charge site) of every charge-carrying batch,
    /// batch-major.
    ///
    /// Zero when nothing carries charge, and zero while an attached GPU
    /// context cannot fill the resident batches' tapes
    /// (`GpuContext.tape_ok`). The transient reads zero as "use per-row LTE
    /// on the summed q plane".
    pub fn qTapeLen(self: *const Circuit) u32 {
        if (self.gpuContext()) |g| if (!g.tape_ok) return 0;
        var total: u32 = 0;
        for (self.batches) |b| {
            if (b.hooks.q_tape) |f| total += @intCast(f(b.ctx).len);
        }
        return total;
    }

    /// Sets `mask[row]` for each branch-current row of the batches named
    /// `type_name`, a subset of `current_row`. The transient takes the
    /// inductor rows from it on a coupled-inductor deck. Asserts that `mask`
    /// holds `n` entries.
    pub fn markCurrentRowsOf(self: *const Circuit, type_name: []const u8, mask: []bool) void {
        std.debug.assert(mask.len == self.n);
        for (self.batches) |b| {
            const mark = b.hooks.mark_current_rows orelse continue;
            if (std.mem.eql(u8, b.type_name, type_name)) mark(b.ctx, mask);
        }
    }

    /// Concatenates every batch's live charge tape into `dst`. Asserts `dst`
    /// holds exactly `qTapeLen()` entries.
    pub fn snapshotQTape(self: *const Circuit, dst: []f64) void {
        if (self.gpuContext()) |g| g.syncQTape(self.sim);
        var off: usize = 0;
        for (self.batches) |b| {
            const f = b.hooks.q_tape orelse continue;
            const src = f(b.ctx);
            @memcpy(dst[off..][0..src.len], src);
            off += src.len;
        }
        std.debug.assert(off == dst.len);
    }

    /// Adds the ground pin. Runs once, after every batch stamp and reduction.
    fn groundStamp(self: *Circuit, x: []const f64) void {
        self.g_vals[self.diag_slots[0]] += 1.0;
        self.rhs[0] += x[0];
    }

    /// Stamps all four planes for state `x` at time `t`, on the attached GPU
    /// context when there is one. Invalidates the `linearize` memo.
    ///
    /// The GPU replaces the stamp, not the analysis: every host step layered
    /// on a stamp (companion RHS, `combineGC`, limiting) composes with it.
    pub fn eval(self: *Circuit, x: []const f64, t: f64) void {
        // Callers also eval at points other than x_op (disto, pss, pac, ...),
        // so the linearization memo cannot survive this.
        self.lin.valid = false;
        self.gc = .full;
        if (self.gpuContext()) |g| return self.evalGpu(g, x, t, false);
        self.stamp(x, t, .full);
    }

    /// The device stamp at `x` (`GpuContext.eval`), then the ground pin, or
    /// the host stamp after a driver fault `g` has already reported. The
    /// planes follow the buffer the device wrote, which a prefetch moves.
    fn evalGpu(self: *Circuit, g: *GpuContext, x: []const f64, t: f64, charge: bool) void {
        const ok = g.eval(x, t, self.sim, charge);
        self.bindPlanes(g.planes());
        if (ok) self.groundStamp(x) else self.evalNewtonCpu(x, t);
    }

    /// Clears the planes `mode` writes, or seeds g/c from the baseline for
    /// `.newton`. The one plane reset; every eval path in this file runs it.
    fn clearPlanes(self: *Circuit, comptime mode: par_eval.Mode) void {
        self.q_fresh = false;
        switch (mode) {
            .charge => zeroSimd(self.q_vec),
            .newton => {
                copySimd(self.g_vals, self.g_base);
                if (self.has_charge) {
                    copySimd(self.c_vals, self.c_base);
                    zeroSimd(self.q_vec);
                }
                zeroSimd(self.rhs);
            },
            .full => {
                zeroSimd(self.g_vals);
                if (self.has_charge) {
                    zeroSimd(self.c_vals);
                    zeroSimd(self.q_vec);
                }
                zeroSimd(self.rhs);
            },
        }
    }

    /// Host stamp: clear, run every batch serially or threaded, then add the
    /// ground pin (which `.charge` does not touch). ParEval sets every cell
    /// itself, baseline included, so it skips the clear.
    fn stamp(self: *Circuit, x: []const f64, t: f64, comptime mode: par_eval.Mode) void {
        const pl = self.ownPlanes();
        if (self.par_eval) |p| {
            self.q_fresh = false;
            const base: ParEval.Base = if (mode == .newton) .{ .g = self.g_base, .c = if (self.has_charge) self.c_base else &.{} } else .{};
            p.run(self.batches, pl, base, x, t, mode);
        } else {
            self.clearPlanes(mode);
            for (self.batches) |b| par_eval.stampRange(b, &pl, 0, b.count, x, t, mode);
        }
        if (mode != .charge) self.groundStamp(x);
    }

    /// Stamps only the charges at `x`: `q_vec` and every batch's `q_tape`,
    /// bit-for-bit what `eval` would leave there, with g/c/rhs untouched.
    /// ParEval reuses the lane cuts and reduce order of `.full`, so the
    /// promise holds threaded too. With a context attached the device runs
    /// its charge-only kernels (`GpuContext.evalQ`), or the full pass when
    /// one is missing, which changes g, c and rhs too; the resident
    /// batches' tapes reach the host at `snapshotQTape`.
    pub fn evalQ(self: *Circuit, x: []const f64, t: f64) void {
        self.lin.valid = false;
        if (self.gpuContext()) |g| {
            if (g.evalQ(x, t, self.sim)) return self.bindPlanes(g.planes());
            return self.evalGpu(g, x, t, true);
        }
        self.stamp(x, t, .charge);
    }

    /// Ensures the planes hold the linearization at `x_op`, reusing them when
    /// they already do. The AC-family analyses all linearize at the same
    /// operating point, so an op+ac+noise+pz deck pays for one device eval.
    pub fn linearize(self: *Circuit, x_op: []const f64) void {
        if (self.lin.valid and self.lin.x_ptr == x_op.ptr and self.lin.len == x_op.len) return;
        self.eval(x_op, 0);
        self.lin = .{ .x_ptr = x_op.ptr, .len = @intCast(x_op.len), .valid = true };
    }

    /// `linearize` for a frequency-domain analysis (ngspice CKTacLoad rather
    /// than CKTload): swaps in each `AcParam` value, stamps, and restores the
    /// DC values. Leaves the memo invalid, because the planes then hold the AC
    /// linearization that a later DC eval must not reuse. Fails with
    /// `error.TopologyChanged` when an AC value re-wires a device; the DC
    /// values are back in place either way.
    pub fn linearizeAc(self: *Circuit, x_op: []const f64) !void {
        if (self.ac_params.len == 0) return self.linearize(x_op);
        for (self.ac_params) |*p| {
            p.saved = p.ptr.get();
            p.ptr.set(p.ac_value);
        }
        defer {
            for (self.ac_params) |*p| p.ptr.set(p.saved);
            self.recompute() catch {};
            self.lin.valid = false;
        }
        try self.recompute();
        self.lin.valid = false;
        self.eval(x_op, 0);
    }

    /// Writes the frequency-dependent term of every `ac_dyn_slots` entry e at
    /// `x` and each ω in `omegas` into `re[e * omegas.len + k]` and `im[...]`,
    /// under the stored sim state: A(ω) = G + jωC + (re + j·im) with G and C
    /// from a `.ac`/`.noise` eval at the same x. Host-side, GPU context or not.
    /// Asserts `re` and `im` hold `ac_dyn_slots.len * omegas.len` entries.
    pub fn acDyn(self: *const Circuit, x: []const f64, omegas: []const f64, re: []f64, im: []f64) void {
        std.debug.assert(re.len == self.ac_dyn_slots.len * omegas.len and im.len == re.len);
        var off: usize = 0;
        for (self.batches) |b| if (b.hooks.ac_dyn) |f| {
            off += omegas.len * f(b.ctx, x, omegas, re[off..], im[off..]);
        };
        std.debug.assert(off == re.len);
    }

    /// Adds `acDyn` at `x` and one ω into the row-major 2n x 2n stacked-real
    /// `a` ([Re -Im; Im Re], `dense_lu.buildComplexAdmittance`'s layout).
    /// `scratch` holds `2 * ac_dyn_slots.len` values.
    pub fn addAcDynDense(self: *const Circuit, x: []const f64, omega: f64, a: []f64, scratch: []f64) void {
        const e = self.ac_dyn_slots.len;
        if (e == 0) return;
        const re = scratch[0..e];
        const im = scratch[e..][0..e];
        self.acDyn(x, &.{omega}, re, im);
        const n: usize = self.n;
        for (self.ac_dyn_slots, re, im) |slot, r, i| {
            if (slot >= self.nnz) continue;
            freq_solve.addDense(n, 2 * n, a, self.row_idx[slot], freq_solve.slotCol(self.col_ptr, slot), r, i);
        }
    }

    /// Stamps the planes for a Newton iterate, seeding from the constant
    /// baseline when one exists.
    pub fn evalNewton(self: *Circuit, x: []const f64, t: f64) void {
        self.lin.valid = false;
        self.gc = .full;
        // The device path ignores the baseline: `g_base` includes the resident
        // batches too, and the GPU sink cannot skip them, so it zeroes and
        // restamps instead.
        if (self.gpuContext()) |g| return self.evalGpu(g, x, t, false);
        self.evalNewtonCpu(x, t);
    }

    /// `evalNewton` for a companion step whose matrix is G + alpha·C, read
    /// back through `newtonGC`: with a context attached the device forms A
    /// (02 2.3), which then comes down in place of g and c, or stays on the
    /// device for the device LU (`deviceSolve`). The planes' g and c then
    /// hold only the diagonal until the next eval.
    pub fn evalNewtonGC(self: *Circuit, x: []const f64, t: f64, alpha: f64) void {
        if (self.gpuContext()) |g| if (self.gcDevice()) |device| {
            self.lin.valid = false;
            self.gc = g.evalGc(x, t, self.sim, .{ .alpha = alpha, .device = device }, self.diag_slots[0..self.n], self.nnz);
            self.gc_alpha = alpha;
            self.bindPlanes(g.planes());
            if (self.gc != .failed) return self.groundStamp(x);
            self.gc = .full;
            return self.evalNewtonCpu(x, t);
        };
        self.evalNewton(x, t);
    }

    /// Whether `evalNewtonGC` may hand A to the device, and whether the
    /// device LU takes it there (`GpuLu.residentReady`). Null under a JFNK
    /// pin, which reads g and c apart, with load-check rows, which read A
    /// off the diagonal, and under `-Dcertify`, which restamps.
    fn gcDevice(self: *Circuit) ?bool {
        if (kcl_cert.enabled or self.load_check != null) return null;
        switch (self.knobs.newton.pin) {
            .auto, .direct => {},
            else => return null,
        }
        const lu = self.lu orelse return false;
        const ws = if (self.ws) |*w| w else return false;
        return lu.residentReady(&ws.slv);
    }

    /// The Newton matrix G + alpha·C after `evalNewtonGC` at the same
    /// alpha: `combineGC` into `out`, the A the device brought down, or,
    /// for the device LU, `out` with only the diagonal written, which
    /// `deviceSolve` completes.
    pub fn newtonGC(self: *Circuit, alpha: f64, out: []f64) []f64 {
        if (self.gc != .full) if (self.gpuContext()) |g| {
            std.debug.assert(@as(u64, @bitCast(alpha)) == @as(u64, @bitCast(self.gc_alpha)));
            return g.gcVals(self.gc, alpha, out, self.diag_slots[0..self.n], self.nnz);
        };
        self.combineGC(alpha, out);
        return out;
    }

    /// Host half of `evalNewton`, also the GPU context's fault fallback.
    fn evalNewtonCpu(self: *Circuit, x: []const f64, t: f64) void {
        if (self.has_baseline) self.stamp(x, t, .newton) else self.stamp(x, t, .full);
    }

    /// Precomputes the constant-Jacobian contribution into `g_base`/`c_base`.
    /// A no-op when no batch has a constant Jacobian or the baseline is
    /// already current. Clobbers `rhs` and `q_vec`. Fails only on
    /// allocation, leaving no baseline and a retry safe.
    pub fn computeBaseline(self: *Circuit) !void {
        if (self.has_baseline) return;
        for (self.batches) |b| {
            if (b.has_const_jacobian) break;
        } else return;

        if (self.g_base.len == 0) {
            // Both or neither: a half-built pair would pass the length test
            // on the retry and stamp into an empty `c_base`.
            const g_base = try self.gpa.alloc(f64, self.nnz + 1);
            errdefer self.gpa.free(g_base);
            self.c_base = try self.gpa.alloc(f64, self.nnz + 1);
            self.g_base = g_base;
        }
        @memset(self.g_base, 0);
        @memset(self.c_base, 0);

        const x_zero = try self.gpa.alloc(f64, self.n + 1);
        defer self.gpa.free(x_zero);
        @memset(x_zero, 0);

        self.lin.valid = false;
        self.q_fresh = false;
        zeroSimd(self.rhs);
        if (self.has_charge) zeroSimd(self.q_vec);
        const pl: Planes = .{ .g_vals = self.g_base, .c_vals = self.c_base, .rhs = self.rhs, .q_vec = self.q_vec };
        for (self.batches) |b| {
            if (b.has_const_jacobian) b.eval(b.ctx, &pl, 0, b.count, x_zero, 0);
        }
        // No ground pin here: `stamp` adds it after copying the baseline.
        self.has_baseline = true;
    }

    /// Drops the baseline and computes it again: for a caller that moved
    /// device parameters the baseline reads (a homotopy rung, a restored
    /// state). Fails as `computeBaseline` does.
    pub fn rebuildBaseline(self: *Circuit) !void {
        self.has_baseline = false;
        try self.computeBaseline();
    }

    /// Returns one `G + alpha*C` entry, the same expression `combineGC` uses,
    /// so the Newton residual gate can read a diagonal in O(1) instead of
    /// rebuilding the whole plane.
    pub fn gcAt(self: *const Circuit, alpha: f64, slot: u32) f64 {
        return self.g_vals[slot] + alpha * self.c_vals[slot];
    }

    /// Writes `G + alpha*C` into `out[0..nnz]`, leaving the trash entry
    /// alone. Asserts `out` holds at least `nnz` entries.
    pub fn combineGC(self: *const Circuit, alpha: f64, out: []f64) void {
        std.debug.assert(out.len >= self.nnz);
        combinePlanes(out[0..self.nnz], self.g_vals[0..self.nnz], self.c_vals[0..self.nnz], alpha);
    }

    /// Writes G as a dense row-major `n*n` matrix into `out`. O(n^2).
    /// Asserts `out` holds at least `n*n` entries.
    pub fn denseG(self: *const Circuit, out: []f64) void {
        self.denseFrom(self.g_vals, out);
    }
    /// `denseG` for C.
    pub fn denseC(self: *const Circuit, out: []f64) void {
        self.denseFrom(self.c_vals, out);
    }

    fn denseFrom(self: *const Circuit, vals: []const f64, out: []f64) void {
        const n: usize = self.n;
        std.debug.assert(out.len >= n * n);
        @memset(out[0 .. n * n], 0);
        for (0..n) |j| {
            for (self.col_ptr[j]..self.col_ptr[j + 1]) |p| {
                out[@as(usize, self.row_idx[p]) * n + j] = vals[p];
            }
        }
    }

    /// Returns the plane index of (row, col), or null when the pattern has no
    /// such entry. O(log) in the column's length. Asserts `col < n`.
    pub fn findSlot(self: *const Circuit, row: u32, col: u32) ?u32 {
        const pattern: device_ir.PatternView = .{
            .col_ptr = self.col_ptr,
            .row_idx = self.row_idx,
            .n = self.n,
            .trash_slot = self.trash_slot,
        };
        return pattern.findSlot(row, col);
    }

    /// Tells the circuit that the next `applyLimits` or `stateCtl(.query)`
    /// is followed by `eval` (or, with `charge`, `evalQ`) at `x` and `t`,
    /// with nothing in between that changes device state. `x` is read when
    /// that call runs, so for `applyLimits` it is the limited iterate. The
    /// GPU path then runs both on one host wait; the host path ignores it. A
    /// wrong promise costs one wasted device eval, never a wrong stamp: the
    /// eval checks its x and its whole `SimState`. Without `charge` the eval
    /// is the next Newton iterate, one `advanceIteration` on.
    pub fn evalFollows(self: *const Circuit, x: []const f64, t: f64, charge: bool) void {
        var sim = self.sim;
        sim.t = t;
        if (!charge) sim.iteration +|= 1;
        if (self.gpuContext()) |g| g.evalFollows(x, sim, charge);
    }

    /// `evalFollows` for the first iterate of the next solve, which
    /// `beginSolve` numbers 1: the predictor's and a sweep's warm start.
    pub fn evalFollowsFirst(self: *const Circuit, x: []const f64, t: f64) void {
        var sim = self.sim;
        sim.t = t;
        sim.iteration = 1;
        if (self.gpuContext()) |g| g.evalFollows(x, sim, false);
    }

    /// Limits the Newton update in place (pnjlim/fetlim). Returns true when any
    /// device clamped `x`.
    pub fn applyLimits(self: *const Circuit, x: []f64, x_old: []const f64) bool {
        if (self.gpuContext()) |g| return g.applyLimits(x, x_old, self.sim);
        if (self.par_eval) |p| return p.limit(self.batches, x, x_old);
        return par_eval.limitBatches(self.batches, x, x_old);
    }

    /// Sets every `idt` state unknown with an ic to that ic (`Hooks.seed_ic`):
    /// the uic start's stand-in for the operating point, which solves them.
    /// Needs a static sim state.
    pub fn seedIc(self: *const Circuit, x: []f64) void {
        for (self.batches) |b| if (b.hooks.seed_ic) |f| f(b.ctx, x);
    }

    /// Writes the devices' §3.6.3.2 net initializers into `x`
    /// (`Hooks.seed_nodeset`). Cold starts only.
    pub fn seedNodesets(self: *const Circuit, x: []f64) void {
        for (self.batches) |b| if (b.hooks.seed_nodeset) |f| f(b.ctx, x);
    }

    /// SPICE MODEINITJCT: devices write junction seed voltages into a zeroed
    /// `x`, so iteration 1 linearizes at vcrit/vto instead of 0 and the
    /// limiters clamp against the seed. Cold starts only.
    pub fn seedJunctions(self: *const Circuit, x: []f64) void {
        if (self.gpuContext()) |g| return g.seedJunctions(x);
        par_eval.seedBatches(self.batches, x);
    }

    /// Resets device-private limiting state after a Newton solve, so later
    /// evals (waveform, AC, noise) see the node vector itself.
    pub fn clearLimits(self: *const Circuit) void {
        if (self.gpuContext()) |g| return g.clearLimits();
        par_eval.clearLimitBatches(self.batches);
    }

    /// Starts a nonlinear solve: `$simparam("iteration")` reads 1.
    pub fn beginSolve(self: *Circuit) void {
        self.lin.valid = false;
        if (self.load_check) |*lc| lc.valid = false;
        self.sim.iteration = 1;
        self.publishSim();
    }

    /// Hands every device the previous Newton iterate, then counts the
    /// iteration.
    pub fn advanceIteration(self: *Circuit, previous_x: []const f64) void {
        self.lin.valid = false;
        for (self.batches) |b| if (b.hooks.advance_iteration) |f| f(b.ctx, previous_x);
        self.sim.iteration +|= 1;
        self.publishSim();
    }

    /// Lets each device that predicts its first transient iterate over the
    /// step two back (`Hooks.predict_first_iterate`) overwrite its internal
    /// unknowns in `trial` with `cur + xfact2 * (cur - prev)`.
    pub fn predictFirstIterate(self: *const Circuit, trial: []f64, cur: []const f64, prev: []const f64, xfact2: f64) void {
        for (self.batches) |b| if (b.hooks.predict_first_iterate) |f| f(b.ctx, trial, cur, prev, xfact2);
    }

    /// Returns false when any device vetoes convergence at `x`.
    pub fn checkConvergence(self: *const Circuit, x: []const f64) bool {
        for (self.batches) |b| if (b.hooks.check_convergence) |f| {
            if (!f(b.ctx, x)) return false;
        };
        return true;
    }

    /// Advances device state (operator history, held variables, latches)
    /// from the last accepted point to `x`. Returns the earliest time a
    /// device asks the step to be rejected at, or null. A second call
    /// before the next commit or revert (a flip that forced another
    /// iterate, a continuation rung) reverts first, since `stateCtl(.revert)`
    /// is exact only across one `updateState`. Under `land_rejects` the
    /// reject time goes to `reject_at` and this returns null. Stages the
    /// attached `Display` lines for the next `stateCtl`.
    pub fn updateStates(self: *Circuit, x: []const f64) ?f64 {
        if (self.state_staged) _ = self.stateCtl(.revert);
        self.state_staged = true;
        if (self.display) |d| self.say(d, x);
        // The transient reads q at the converged point next, and it must be
        // the charge before `updateState` latches the `$prev` values `q`
        // reads (the Meyer caps average against the last accepted point).
        const accept = self.land_rejects and self.has_charge;
        const tr = if (accept) self.acceptStates(x) else if (self.gpuContext()) |g| g.updateStates(x, self.sim) else self.hostUpdate(x);
        if (!self.land_rejects) return tr;
        self.reject_at = tr;
        return null;
    }

    /// `updateBatches` on the host, over ParEval's lanes when it exists.
    fn hostUpdate(self: *const Circuit, x: []const f64) ?f64 {
        if (self.par_eval) |p| return p.update(self.batches, x);
        return par_eval.updateBatches(self.batches, x);
    }

    /// Stages every `Hooks.say` line at x, before `updateState` moves the
    /// instances on from the state eval saw.
    fn say(self: *const Circuit, d: *Display, x: []const f64) void {
        std.debug.assert(d.names.len == self.batches.len);
        for (self.batches, d.names) |b, names| {
            const f = b.hooks.say orelse continue;
            std.debug.assert(names.len == b.count);
            for (names, 0..) |name, i| f(b.ctx, x, @intCast(i), name, &d.staged.writer);
        }
    }

    /// `evalQ` at x and the published time, then `updateBatches`. Serially
    /// through `accept_q` where the batch has it, so its model core runs
    /// once for both.
    // ponytail: the GPU and ParEval paths run two passes; a ParEval
    // `.accept` mode would fuse them when lanes are on.
    fn acceptStates(self: *Circuit, x: []const f64) ?f64 {
        // The device runs the charges, the state halves and the query on
        // one wait (`GpuContext.acceptStates`).
        if (self.gpuContext()) |g| if (g.acceptStates(x, self.sim.t, self.sim)) |tr| {
            self.lin.valid = false;
            self.bindPlanes(g.planes());
            self.q_fresh = true;
            return tr;
        };
        if (self.gpuContext() != null or self.par_eval != null) {
            self.evalQ(x, self.sim.t);
            const tr = if (self.gpuContext()) |g| g.updateStates(x, self.sim) else self.hostUpdate(x);
            self.q_fresh = true;
            return tr;
        }
        self.lin.valid = false;
        self.clearPlanes(.charge);
        const pl = self.ownPlanes();
        var min_reject: ?f64 = null;
        for (self.batches) |b| {
            if (b.hooks.accept_q) |f| {
                f(b.ctx, &pl, 0, b.count, x);
                continue;
            }
            if (b.hooks.eval_q) |f| f(b.ctx, &pl, 0, b.count, x, self.sim.t);
            if (b.hooks.update_state) |f| if (f(b.ctx, x)) |tr| {
                min_reject = if (min_reject) |cur| @min(cur, tr) else tr;
            };
        }
        self.q_fresh = true;
        return min_reject;
    }

    /// Accepted-point half of `updateStates`, for delay-line history no
    /// static solve may push (`Hooks.commit_state`). The transient calls it
    /// once per accepted point, before `stateCtl(.commit)`.
    pub fn commitStates(self: *const Circuit, x: []const f64) ?f64 {
        return par_eval.minReject(self.batches, "commit_state", x);
    }

    /// Commits, reverts or queries the accepted device state. `.commit` at
    /// every accepted point, the operating point included; `.revert` after
    /// every rejected attempt. For `.query`, returns true when a cross/above
    /// flip moved the working state off the last accepted one.
    pub fn stateCtl(self: *Circuit, sop: StateCtlOp) bool {
        if (sop != .query) self.state_staged = false;
        if (self.display) |d| if (sop != .query) {
            // ponytail: a failed write to `out` drops the point's lines.
            if (sop == .commit) d.out.writeAll(d.staged.written()) catch {};
            d.staged.clearRetainingCapacity();
        };
        if (self.gpuContext()) |g| return g.stateCtl(sop, self.sim);
        return par_eval.stateCtlBatches(self.batches, sop);
    }

    /// Returns the smallest model delay (`D.delays`) in the circuit, or null.
    pub fn minDelay(self: *const Circuit) ?f64 {
        var min_td = std.math.inf(f64);
        for (self.batches) |b| if (b.hooks.min_delay) |f| {
            min_td = @min(min_td, f(b.ctx));
        };
        return if (min_td == std.math.inf(f64)) null else min_td;
    }

    /// Returns the tightest `$bound_step` (VAMS 9.17.2) any device asked for,
    /// or null. Only meaningful after an accepted step ran `updateStates`.
    pub fn boundStep(self: *const Circuit) ?f64 {
        var best = std.math.inf(f64);
        for (self.batches) |b| if (b.hooks.bound_step) |f| {
            best = @min(best, f(b.ctx));
        };
        return if (best == std.math.inf(f64)) null else best;
    }

    /// Returns the earliest device breakpoint or VPI forced time (LRM
    /// 12.31.3 acbAbsTime/acbElapsedTime) after `t`, or null.
    pub fn nextBreakpoint(self: *const Circuit, t: f64) ?f64 {
        var best = std.math.inf(f64);
        for (self.batches) |b| if (b.hooks.next_breakpoint) |f| {
            if (f(b.ctx, t)) |bp| best = @min(best, bp);
        };
        if (self.vpi) |v| if (v.next_forced(v.ctx, t)) |f| {
            best = @min(best, f);
        };
        return if (best == std.math.inf(f64)) null else best;
    }

    /// Installs a circuit temperature in degrees Celsius. Call `recompute`
    /// before solving, which also detects a temperature-driven topology change.
    pub fn setCircuitTemp(self: *Circuit, temp_c: f32) void {
        self.temp_c = temp_c;
        self.lin.valid = false;
        for (self.batches) |b| if (b.hooks.set_temp) |f| f(b.ctx, temp_c);
        self.markGpuDirty();
    }

    /// Publishes the stepping rungs' gmin (S) and source factor to the models
    /// that read §9.15 `$simparam("gmin")` or `("sourceScaleFactor")`. The
    /// caller restores the deck's gmin and 1 when stepping ends. A circuit
    /// without such a model is untouched; otherwise the constant-Jacobian
    /// baseline is dropped, so call `computeBaseline` before the next solve.
    pub fn setHomotopy(self: *Circuit, gmin: f64, source_scale: f64) void {
        var any = false;
        for (self.batches) |b| if (b.hooks.set_homotopy) |f| {
            f(b.ctx, gmin, source_scale);
            any = true;
        };
        if (any) self.paramsMoved();
    }

    /// Device parameters moved: drops the `linearize` memo and the
    /// constant-Jacobian baseline, and marks the resident GPU parameters and
    /// every bypass recording stale.
    fn paramsMoved(self: *Circuit) void {
        self.lin.valid = false;
        self.has_baseline = false;
        self.markGpuDirty();
    }

    fn markGpuDirty(self: *Circuit) void {
        if (self.gpuContext()) |g| g.markDirty();
        // Device parameters moved: every bypass recording is stale.
        if (self.bypass_tol) |tol| self.setBypass(tol);
    }

    /// Turns the device latency bypass on with `tol`, or off with null, on
    /// every batch that supports it; turning it on again drops every
    /// recording.
    pub fn setBypass(self: *Circuit, tol: ?device_ir.BypassTol) void {
        self.bypass_tol = tol;
        for (self.batches) |b| if (b.hooks.set_bypass) |f| f(b.ctx, tol);
    }

    /// Publishes the analysis state (`$abstime`, timestep, `analysis()`,
    /// `initial_step`/`final_step`) every later device call receives; `eval`
    /// takes `$abstime` from its own `t`. Keeps the Newton iteration count.
    /// Must run before the eval it describes.
    pub fn setSimState(self: *Circuit, st: device_ir.SimState) void {
        const iteration = self.sim.iteration;
        self.sim = st;
        self.sim.iteration = iteration;
        self.publishSim();
    }

    /// Copies `sim` into every batch: one store each.
    fn publishSim(self: *const Circuit) void {
        for (self.batches) |b| b.hooks.set_sim_state(b.ctx, self.sim);
    }

    /// Re-derives every batch's numeric parameters. Drops the constant-Jacobian
    /// baseline (`computeBaseline` rebuilds it) and marks the GPU context's
    /// resident parameters stale. `error.TopologyChanged` means internal
    /// wiring moved and the circuit must be rebuilt; restore the parameters
    /// and recompute before reusing it.
    pub fn recompute(self: *Circuit) error{TopologyChanged}!void {
        self.paramsMoved();
        for (self.batches) |b| if (b.hooks.recompute) |f| {
            if (!f(b.ctx)) return error.TopologyChanged;
        };
    }

    /// `recompute` restricted to batches of type `t` (`ParamRef.type`).
    ///
    /// A `.dc` point writes one parameter, and a full recompute re-runs every
    /// compact model's temperature preamble (1,811 BJT preambles per sweep on
    /// a one-instance deck). Sound only while nothing global has moved: a
    /// temperature change can re-wire a device of another type (a BJT whose
    /// RB(T) reaches zero). The caller owns that distinction; `.dc` takes the
    /// full walk on the first point of every inner sweep and narrows after.
    pub fn recomputeType(self: *Circuit, t: device_ir.DeviceType) error{TopologyChanged}!void {
        self.paramsMoved();
        for (self.batches, self.batch_types) |b, bt| if (bt == t) if (b.hooks.recompute) |f| {
            if (!f(b.ctx)) return error.TopologyChanged;
        };
    }

    /// Returns every device parameter, typed by batch. Built once and freed by
    /// `release`; the refs point into frozen batch storage, so they stay
    /// valid across `restoreState` and die with the batches.
    pub fn collectParams(self: *Circuit) ![]const ParamRef {
        if (self.param_refs) |refs| return refs;
        const gpa = self.gpa;
        var list: std.ArrayList(ParamRef) = .empty;
        errdefer list.deinit(gpa);
        try collectTyped(self.batches, self.batch_types, gpa, &list);
        self.param_refs = try list.toOwnedSlice(gpa);
        return self.param_refs.?;
    }

    /// Writes `values` to the parameters numbered `refs` (`collectParams`
    /// order). With `rederive`, re-derives the batches they reach, which
    /// is `recompute` when nothing else moved. With the template's
    /// `param_starts` only those batches' parameters are collected;
    /// without them, all of them.
    pub fn setParams(self: *Circuit, refs: []const u32, values: []const f64, rederive: bool) !void {
        std.debug.assert(refs.len == values.len);
        const starts = self.param_starts;
        if (starts.len == 0) {
            const params = try self.collectParams();
            for (refs, values) |r, v| params[r].set(v);
            if (rederive) try self.recompute();
            return;
        }
        std.debug.assert(starts.len == self.batches.len + 1);
        const gpa = self.gpa;
        const touched = try gpa.alloc(bool, self.batches.len);
        defer gpa.free(touched);
        @memset(touched, false);
        for (refs) |r| touched[std.sort.upperBound(u32, starts, r, orderU32) - 1] = true;
        var list: std.ArrayList(ParamRef) = .empty;
        defer list.deinit(gpa);
        for (self.batches, touched, starts[0..self.batches.len], starts[1..]) |b, t, lo, hi| {
            if (!t) continue;
            if (b.hooks.write_params) |write| {
                try write(b.ctx, refs, values, lo, hi).unwrap();
                continue;
            }
            list.clearRetainingCapacity();
            try b.hooks.collect_params(b.ctx, gpa, &list).unwrap();
            std.debug.assert(list.items.len == hi - lo);
            for (refs, values) |r, v| if (r >= lo and r < hi) list.items[r - lo].set(v);
        }
        if (!rederive) return;
        self.paramsMoved();
        for (self.batches, touched) |b, t| if (t) if (b.hooks.recompute) |f| {
            if (!f(b.ctx)) return error.TopologyChanged;
        };
    }

    /// Refs to the parameters numbered `ords` (`collectParams` numbering)
    /// that each write their own instance alone: an instance a Model-field
    /// ref reaches gets a private copy of its row when others read it
    /// (`Hooks.isolate_rows`), and the rest keep sharing, so a mismatch over
    /// a few devices of a large circuit copies a few rows. Without the
    /// template's `param_starts`, or on a batch without the hooks, every row
    /// is unshared (`collectParams`). Valid until the rows move again (a
    /// `setParams`, `collectParams` or another call). Fails with
    /// `error.InvalidVariation` for an ordinal past the circuit's
    /// parameters. Caller frees the slice with `gpa`.
    pub fn isolatedParams(self: *Circuit, gpa: std.mem.Allocator, ords: []const u32) ![]ParamRef {
        const out = try gpa.alloc(ParamRef, ords.len);
        errdefer gpa.free(out);
        const starts = self.param_starts;
        const layouts = starts.len != 0 and self.param_refs == null and for (ords) |o| {
            if (o >= starts[self.batches.len]) return error.InvalidVariation;
            const h = self.batches[std.sort.upperBound(u32, starts, o, orderU32) - 1].hooks;
            if (h.param_layout == null or h.isolate_rows == null) break false;
        } else true;
        if (!layouts) {
            const all = try self.collectParams();
            for (ords, out) |o, *ref| ref.* = if (o < all.len) all[o] else return error.InvalidVariation;
            return out;
        }
        var ids: std.ArrayList(u32) = .empty;
        defer ids.deinit(gpa);
        for (self.batches, starts[0..self.batches.len], starts[1..]) |b, lo, hi| {
            ids.clearRetainingCapacity();
            for (ords) |o| if (o >= lo and o < hi) {
                const l = try b.hooks.param_layout.?(b.ctx).unwrap();
                if (!l.fields[(o - lo) / l.count].is_instance) try ids.append(gpa, (o - lo) % l.count);
            };
            if (ids.items.len == 0) continue;
            std.mem.sort(u32, ids.items, {}, std.sort.asc(u32));
            var unique: usize = 0;
            for (ids.items) |id| if (unique == 0 or ids.items[unique - 1] != id) {
                ids.items[unique] = id;
                unique += 1;
            };
            try b.hooks.isolate_rows.?(b.ctx, ids.items[0..unique]).unwrap();
        }
        for (ords, out) |o, *ref| {
            const k = std.sort.upperBound(u32, starts, o, orderU32) - 1;
            const b = self.batches[k];
            const l = try b.hooks.param_layout.?(b.ctx).unwrap();
            const r = o - starts[k];
            const f = l.fields[r / l.count];
            ref.* = .{ .ptr = l.ptr(r), .type = self.batch_types[k], .param_name = f.name, .index = r % l.count, .is_instance = f.is_instance, .primary = f.primary };
        }
        return out;
    }

    /// Parameter `ord`'s (`collectParams` numbering) value and device type,
    /// read without unsharing any Model row when the template has
    /// `param_starts` and the batch a `param_layout`.
    pub fn paramAt(self: *Circuit, ord: u32) !struct { value: f64, type: device_ir.DeviceType } {
        const starts = self.param_starts;
        if (starts.len != 0) {
            const k = std.sort.upperBound(u32, starts, ord, orderU32) - 1;
            const b = self.batches[k];
            if (b.hooks.param_layout) |layout| return .{ .value = (try layout(b.ctx).unwrap()).ptr(ord - starts[k]).*, .type = self.batch_types[k] };
        }
        const ref = (try self.collectParams())[ord];
        return .{ .value = ref.get(), .type = ref.type };
    }

    /// Returns the short type name of device type `t`, or "" when absent.
    pub fn typeName(self: *const Circuit, t: device_ir.DeviceType) []const u8 {
        for (self.batches, self.batch_types) |b, bt| if (bt == t) return b.type_name;
        return "";
    }

    /// Appends every batch's parameters to `list`, each stamped with its
    /// batch's Library type. Invalidates pointers into `list` when it grows.
    fn collectTyped(batches: []const Batch, types: []const device_ir.DeviceType, gpa: std.mem.Allocator, list: *std.ArrayList(ParamRef)) !void {
        for (batches, types) |b, t| {
            const first = list.items.len;
            try b.hooks.collect_params(b.ctx, gpa, list).unwrap();
            for (list.items[first..]) |*ref| ref.type = t;
        }
    }

    /// Returns every device's noise generators at `x`, with each device's own
    /// PSDs (which already carry its temperature). Pure in `x`, so `.pnoise`
    /// calls it per PSS sample. Caller owns the slice and frees it with `gpa`.
    pub fn collectNoiseSources(self: *const Circuit, x: []const f64, gpa: std.mem.Allocator) ![]NoiseSource {
        var list: std.ArrayList(NoiseSource) = .empty;
        errdefer list.deinit(gpa);
        for (self.batches) |b| if (b.hooks.collect_noise) |f| try f(b.ctx, x, gpa, &list).unwrap();
        return try list.toOwnedSlice(gpa);
    }

    /// Returns the netlist name of unknown `node`, or "" for a branch unknown
    /// or a row past the circuit.
    pub fn nodeName(self: *const Circuit, node: u32) []const u8 {
        if (@as(u64, node) + 1 < self.intern_offs.len) // u64: maxInt(u32) + 1 wraps a 32-bit usize
            return self.intern_bytes[self.intern_offs[node]..self.intern_offs[node + 1]];
        return "";
    }
};

/// Returns one copy of each of `sources` made by `Hooks.<hook>`
/// (`instantiate` or `snapshot`). Caller frees it with `freeBatches`.
/// Fails with the hook's error, leaving nothing allocated.
fn copyBatches(gpa: std.mem.Allocator, sources: []const Batch, comptime hook: []const u8) ![]Batch {
    const batches = try gpa.alloc(Batch, sources.len);
    errdefer gpa.free(batches);
    var count: usize = 0;
    errdefer for (batches[0..count]) |b| b.hooks.deinit(b.ctx, gpa);
    for (sources, batches) |source, *copy| {
        copy.* = try @field(source.hooks, hook)(source.ctx, gpa).unwrap();
        count += 1;
    }
    return batches;
}

fn freeBatches(gpa: std.mem.Allocator, batches: []const Batch) void {
    for (batches) |b| b.hooks.deinit(b.ctx, gpa);
    gpa.free(batches);
}

/// The rows `Circuit.loadCheck` tests, their CSC entries row by row, and
/// the last iterate's current, entries and x. One allocation per slice,
/// freed with the workspace.
const LoadCheck = struct {
    rows: []u32,
    /// Entries of row k are `ptr[k]..ptr[k + 1]`.
    ptr: []u32,
    slot: []u32,
    col: []u32,
    i_old: []f64,
    v_old: []f64,
    x_old: []f64,
    valid: bool = false,

    /// Null when no batch marks a row.
    fn init(gpa: std.mem.Allocator, batches: []const Batch, col_ptr: []const u32, row_idx: []const u32) !?LoadCheck {
        const n = col_ptr.len - 1;
        const mask = try gpa.alloc(bool, n);
        defer gpa.free(mask);
        @memset(mask, false);
        var any = false;
        for (batches) |b| if (b.hooks.mark_load_check_rows) |mark| {
            mark(b.ctx, mask);
            any = true;
        };
        if (!any) return null;
        // Row position of each marked row, then the entries counted per row.
        const pos = try gpa.alloc(u32, n);
        defer gpa.free(pos);
        var n_rows: u32 = 0;
        for (mask, pos) |m, *p| {
            p.* = n_rows;
            n_rows += @intFromBool(m);
        }
        var lc: LoadCheck = undefined;
        lc.rows = try gpa.alloc(u32, n_rows);
        errdefer gpa.free(lc.rows);
        lc.ptr = try gpa.alloc(u32, n_rows + 1);
        errdefer gpa.free(lc.ptr);
        for (mask, 0..) |m, r| if (m) {
            lc.rows[pos[r]] = @intCast(r);
        };
        @memset(lc.ptr, 0);
        for (row_idx) |r| if (mask[r]) {
            lc.ptr[pos[r] + 1] += 1;
        };
        for (1..lc.ptr.len) |k| lc.ptr[k] += lc.ptr[k - 1];
        const n_e = lc.ptr[n_rows];
        lc.slot = try gpa.alloc(u32, n_e);
        errdefer gpa.free(lc.slot);
        lc.col = try gpa.alloc(u32, n_e);
        errdefer gpa.free(lc.col);
        const fill = try gpa.dupe(u32, lc.ptr[0..n_rows]);
        defer gpa.free(fill);
        for (0..n) |c| for (col_ptr[c]..col_ptr[c + 1]) |slot| {
            const r = row_idx[slot];
            if (!mask[r]) continue;
            const e = fill[pos[r]];
            fill[pos[r]] += 1;
            lc.slot[e] = @intCast(slot);
            lc.col[e] = @intCast(c);
        };
        lc.i_old = try gpa.alloc(f64, n_rows);
        errdefer gpa.free(lc.i_old);
        lc.v_old = try gpa.alloc(f64, n_e);
        errdefer gpa.free(lc.v_old);
        lc.x_old = try gpa.alloc(f64, n_e);
        lc.valid = false;
        return lc;
    }

    fn deinit(self: *LoadCheck, gpa: std.mem.Allocator) void {
        for ([_][]u32{ self.rows, self.ptr, self.slot, self.col }) |s| gpa.free(s);
        for ([_][]f64{ self.i_old, self.v_old, self.x_old }) |s| gpa.free(s);
    }
};

/// Freezes `protos` into an owning Circuit (test construction). Takes
/// ownership of the intern table and consumes the protos.
pub fn init(
    gpa: std.mem.Allocator,
    n: u32,
    intern_bytes: []u8,
    intern_offs: []u32,
    protos: []const device_ir.Proto,
    types: []const device_ir.DeviceType,
    bbd: ?BbdInfo,
) !Circuit {
    var prepared = try Prepared.freeze(gpa, n, intern_bytes, intern_offs, protos, types, bbd, .{});
    errdefer prepared.deinit();
    return Circuit.allocate(prepared, prepared.allocator, prepared.batches, true);
}

/// Writes `out = g + alpha*c` over independent CSC entries, rounded as the
/// scalar expression (no FMA contraction). `out` may alias `g`. Asserts
/// that the three slices have one length.
pub fn combinePlanes(out: []f64, g: []const f64, c: []const f64, alpha: f64) void {
    std.debug.assert(out.len == g.len);
    std.debug.assert(out.len == c.len);
    for (out, g, c) |*o, gv, cv| o.* = gv + alpha * cv;
}

fn orderU32(a: u32, b: u32) std.math.Order {
    return std.math.order(a, b);
}
