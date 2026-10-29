//! The analysis-side circuit: a frozen CSC pattern plus four value planes
//! (g = dI/dx, c = dQ/dx, rhs = I, q = Q) that one device pass fills. Every
//! analysis is an affine consumer of the planes: DC solves G, transient
//! G + a C, AC G + jwC. This file owns the planes and their evaluation; the
//! circuit's lifetime is `circuit_life.zig`, its device-state walks
//! `circuit_devices.zig`, its parameters `circuit_params.zig`, and each is a
//! method here.
//! In: the prepared `device.Circuit` template and an x. Out: the planes, the
//! device state the solvers drive, and the routing to an attached ParEval,
//! GPU context or device LU.
//! Cites ngspice resdefs.h:48-49 and resload.c:60-62 (`ac=`), CKTacLoad;
//! dev/devices/models.md; #53 (`certify.zig`).
//!
//! Contents:
//! - `pub const Knobs = knobs.Knobs`
//! - `pub const kcl_cert = @import("certify.zig")`
//! - `pub const GROUND: u32`
//! - `pub const AcParam = struct { ptr, ac_value, saved }`
//! - `pub const EvalHook = struct` (`assemble`, `vals`, `diagAt`)
//! - `pub const Display = struct { names, staged, out }`
//! - `pub const Circuit = struct` (fields, the plane methods below, and the
//!   methods of `circuit_life.zig`, `circuit_devices.zig`, `circuit_params.zig`);
//!   methods and members: `certify`, `SavedState`, `instantiate`,
//!   `instantiateMove`, `fromSnapshot`, `fromSnapshotMove`, `deinit`,
//!   `release`, `saveState`, `storeState`, `restoreState`, `refuseDigital`,
//!   `evalFollows`, `evalFollowsFirst`, `applyLimits`, `seedIc`,
//!   `seedNodesets`, `seedJunctions`, `clearLimits`, `beginSolve`,
//!   `advanceIteration`, `predictFirstIterate`, `checkConvergence`,
//!   `updateStates`, `commitStates`, `stateCtl`, `minDelay`, `boundStep`,
//!   `nextBreakpoint`, `setCircuitTemp`, `setHomotopy`, `paramsMoved`,
//!   `setBypass`, `setSimState`, `recompute`, `recomputeType`,
//!   `collectNoiseSources`, `collectParams`, `setParams`, `isolatedParams`,
//!   `paramAt`, `typeName`, `certifyPoint`, `deviceSolve`, `workspace`,
//!   `loadCheck`, `pooled`, `configure`, `attachGpu`, `gpuContext`,
//!   `detachGpu`, `checkpoint`, `bindPlanes`, `ownPlanes`, `qTapeLen`,
//!   `markCurrentRowsOf`, `snapshotQTape`, `eval`, `clearPlanes`, `evalQ`,
//!   `linearize`, `linearizeAc`, `acDyn`, `addAcDynDense`, `evalNewton`,
//!   `evalNewtonGC`, `newtonGC`, `computeBaseline`, `rebuildBaseline`,
//!   `gcAt`, `combineGC`, `denseG`, `denseC`, `findSlot`, `nodeName`
//! - `pub const init = circuit_life.freeze`
//! - `pub fn combinePlanes(out: []f64, g: []const f64, c: []const f64, alpha: f64) void`

const std = @import("std");
const assert = std.debug.assert;
const abi = @import("device").abi;
const Prepared = @import("device").Circuit;
const solver = @import("solver");
const converger = solver.converger;
const direct = solver.direct;
const freq_solve = solver.freq_solve;
const numerics = @import("core").numerics;
const progress_api = @import("worker.zig");
const par_eval = @import("par_eval.zig");
const gpu_mod = @import("gpu/context.zig");
const GpuContext = gpu_mod.GpuContext;
const GpuLu = @import("gpu/lu.zig").GpuLu;
const WsPool = @import("ws_pool.zig").WsPool;
const knobs = @import("knobs.zig");
const certify_mod = @import("certify.zig");
const LoadCheck = @import("load_check.zig").LoadCheck;
const circuit_life = @import("circuit_life.zig");
const circuit_devices = @import("circuit_devices.zig");
const circuit_params = @import("circuit_params.zig");
const Planes = abi.Planes;
const ParEval = par_eval.ParEval;

/// The facade's developer switches (`knobs.zig`).
pub const Knobs = knobs.Knobs;

/// The KCL certificate under `-Dcertify` (`certify.zig`, #53).
pub const kcl_cert = certify_mod;

/// Row and column index of the ground node in every pattern.
pub const GROUND: u32 = 0;

/// A card parameter with a separate frequency-domain value, such as a
/// resistor's `ac=`. Matches ngspice's RESacResist/RESacConduct
/// (resdefs.h:48-49, stamped by resload.c:60-62): it moves .ac/.sp/.noise/.pz
/// and leaves .op/.dc/.tran alone.
pub const AcParam = struct {
    ptr: abi.ParamRef,
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
        assert(slot <= ckt.nnz);
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
/// Invariant: every plane of `g_vals`/`c_vals` has `nnz + 1` entries (the
/// last is `trash_slot`), `rhs`/`q_vec` have `n + 1`.
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
    batches: []abi.Batch,
    /// Library type of each batch, parallel to `batches`. Borrowed topology.
    batch_types: []const abi.DeviceType,
    /// True where unknown i is an MNA branch current (KVL row), false for a
    /// node voltage. Read by the converger's per-row tolerance.
    current_row: []bool,
    has_charge: bool,
    /// Some charge-carrying batch advances device state at the accepted step
    /// (freeze_grad latches, absdelay rings). The transient must then re-read
    /// q under the committed state before recording it, or the next step
    /// opens on a residual a dq that doubles every time dt halves.
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
    bbd: ?numerics.BbdInfo = null,
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
    bypass_tol: ?abi.BypassTol = null,
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
    sim: abi.SimState = .{},
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
    /// `request_reject_at` in `reject_at`, for the step to be retried ending
    /// there, instead of returning it as a flip that forces another iterate
    /// at the same time.
    land_rejects: bool = false,
    /// The earliest rejection time the last `updateStates` recorded under
    /// `land_rejects`, or null. The transient clears it before an attempt
    /// that may fail without staging any.
    reject_at: ?f64 = null,
    /// `q_vec` and the q tapes hold the charges at the x the last
    /// `updateStates` staged, which computed them on the way. Cleared by
    /// every plane reset.
    q_fresh: bool = false,
    /// Symbolic LU and Newton scratch, built on first use and shared by every
    /// analysis. The pattern is frozen, so it stays valid for the lifetime.
    ws: ?converger.Workspace = null,
    /// Where `release` returns `ws` and `workspace` takes one from, shared by
    /// the Problem's queries; borrowed. Null builds and frees its own.
    ws_pool: ?*WsPool = null,
    /// `loadCheck`'s rows and last iterate, built with `ws`; null when no
    /// batch marks rows (`Hooks.mark_load_check_rows`).
    load_check: ?LoadCheck = null,
    /// `collectParams` memo. The refs point into frozen batch storage.
    param_refs: ?[]abi.ParamRef = null,
    /// The template's `Prepared.param_starts`, borrowed; empty when it has
    /// none.
    param_starts: []const u32 = &.{},
    /// Parameters with a frequency-domain value, which `linearizeAc` swaps to
    /// their AC values. Set by the executor; owned by the query work arena,
    /// which outlives every `linearizeAc`. Empty on every deck without one.
    ac_params: []AcParam = &.{},
    /// The KCL certificate's record under `-Dcertify`; void otherwise.
    cert: if (certify_mod.enabled) certify_mod.Tally else void = if (certify_mod.enabled) .{} else {},

    /// `newton` calls `certifyPoint` on every converged x it stages when true.
    pub const certify = certify_mod.enabled;

    /// Device state held outside the circuit (`circuit_life.SavedState`).
    pub const SavedState = circuit_life.SavedState;
    /// Lifetime (`circuit_life.zig`).
    pub const instantiate = circuit_life.instantiate;
    pub const instantiateMove = circuit_life.instantiateMove;
    pub const fromSnapshot = circuit_life.fromSnapshot;
    pub const fromSnapshotMove = circuit_life.fromSnapshotMove;
    pub const deinit = circuit_life.deinit;
    pub const release = circuit_life.release;
    pub const saveState = circuit_life.saveState;
    pub const storeState = circuit_life.storeState;
    pub const restoreState = circuit_life.restoreState;
    pub const refuseDigital = circuit_life.refuseDigital;
    /// Device-state walks (`circuit_devices.zig`).
    pub const evalFollows = circuit_devices.evalFollows;
    pub const evalFollowsFirst = circuit_devices.evalFollowsFirst;
    pub const applyLimits = circuit_devices.applyLimits;
    pub const seedIc = circuit_devices.seedIc;
    pub const seedNodesets = circuit_devices.seedNodesets;
    pub const seedJunctions = circuit_devices.seedJunctions;
    pub const clearLimits = circuit_devices.clearLimits;
    pub const beginSolve = circuit_devices.beginSolve;
    pub const advanceIteration = circuit_devices.advanceIteration;
    pub const predictFirstIterate = circuit_devices.predictFirstIterate;
    pub const checkConvergence = circuit_devices.checkConvergence;
    pub const updateStates = circuit_devices.updateStates;
    pub const commitStates = circuit_devices.commitStates;
    pub const stateCtl = circuit_devices.stateCtl;
    pub const minDelay = circuit_devices.minDelay;
    pub const boundStep = circuit_devices.boundStep;
    pub const nextBreakpoint = circuit_devices.nextBreakpoint;
    pub const setCircuitTemp = circuit_devices.setCircuitTemp;
    pub const setHomotopy = circuit_devices.setHomotopy;
    pub const paramsMoved = circuit_devices.paramsMoved;
    pub const setBypass = circuit_devices.setBypass;
    pub const setSimState = circuit_devices.setSimState;
    pub const recompute = circuit_devices.recompute;
    pub const recomputeType = circuit_devices.recomputeType;
    pub const collectNoiseSources = circuit_devices.collectNoiseSources;
    /// Parameters (`circuit_params.zig`).
    pub const collectParams = circuit_params.collectParams;
    pub const setParams = circuit_params.setParams;
    pub const isolatedParams = circuit_params.isolatedParams;
    pub const paramAt = circuit_params.paramAt;
    pub const typeName = circuit_params.typeName;

    /// Certifies the converged `x` (`certify.zig`). Only `newton` calls it,
    /// and only under `-Dcertify`.
    pub fn certifyPoint(self: *Circuit, hook: anytype, x: []const f64, t: f64, opts: converger.Options) void {
        certify_mod.check(self, hook, x, t, opts);
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
        if (self.ws != null) return &self.ws.?;
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
        return &self.ws.?;
    }

    /// ngspice's in-load current test (`LoadCheck.check`): true when every
    /// marked row passes, always true when no batch marks rows.
    pub fn loadCheck(self: *Circuit, x: []const f64, vals: []const f64, rhs: []const f64, reltol: f64, abstol: f64) bool {
        const lc = if (self.load_check) |*l| l else return true;
        return lc.check(self.current_row, x, vals, rhs, reltol, abstol);
    }

    /// Returns `ws_pool` when this circuit's workspaces come from the same
    /// allocator.
    pub fn pooled(self: *const Circuit) ?*WsPool {
        const p = self.ws_pool orelse return null;
        return if (p.gpa.ptr == self.gpa.ptr and p.gpa.vtable == self.gpa.vtable) p else null;
    }

    /// Installs the query's solver threading and the facade's developer
    /// switches. Touches no plane or parameter, so `lin` stays. Asserts that
    /// `workspace` has not run yet, since it reads `knobs`.
    pub fn configure(self: *Circuit, execution: numerics.Execution, k: Knobs) void {
        assert(self.ws == null);
        self.solver_execution = execution;
        self.knobs = k;
    }

    /// Evaluates the resident batches through `g` from now on and moves the
    /// planes into its page-locked buffers, which `GpuContext.init` filled
    /// with their values. `g` is borrowed: `detachGpu` before freeing it.
    /// Asserts that no context is attached.
    pub fn attachGpu(self: *Circuit, g: *GpuContext) void {
        assert(self.gpu == null);
        self.gpu = g;
        self.bindPlanes(g.planes());
    }

    /// Returns the attached GPU context, or null; comptime null in a build
    /// without kernel images (`gpu/context.zig` `available`), so every GPU
    /// branch folds away there.
    pub inline fn gpuContext(self: *const Circuit) ?*GpuContext {
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

    /// Points the four planes at `planes`, values as they stand there: the
    /// GPU context's page-locked buffers while it lives, the circuit's own
    /// allocations again when it goes. Asserts the lengths match.
    pub fn bindPlanes(self: *Circuit, planes: Planes) void {
        assert(planes.g_vals.len == self.g_vals.len);
        assert(planes.c_vals.len == self.c_vals.len);
        assert(planes.rhs.len == self.rhs.len);
        assert(planes.q_vec.len == self.q_vec.len);
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
    /// batch-major. Zero when nothing carries charge, and zero while an
    /// attached GPU context cannot fill the resident batches' tapes
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
        assert(mask.len == self.n);
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
        assert(off == dst.len);
    }

    /// Stamps all four planes for state `x` at time `t`, on the attached GPU
    /// context when there is one. Invalidates the `linearize` memo. The GPU
    /// replaces the stamp, not the analysis: every host step layered on a
    /// stamp (companion RHS, `combineGC`, limiting) composes with it.
    pub fn eval(self: *Circuit, x: []const f64, t: f64) void {
        // Callers also eval at points other than x_op (disto, pss, pac, ...),
        // so the linearization memo cannot survive this.
        self.lin.valid = false;
        self.gc = .full;
        if (self.gpuContext()) |g| return self.evalGpu(g, x, t, false);
        self.stamp(x, t, .full);
    }

    /// Clears the planes `mode` writes, or seeds g/c from the baseline for
    /// `.newton`. The one plane reset; every eval path runs it.
    pub fn clearPlanes(self: *Circuit, comptime mode: par_eval.Mode) void {
        self.q_fresh = false;
        switch (mode) {
            .charge => numerics.zeroSimd(self.q_vec),
            .newton => {
                numerics.copySimd(self.g_vals, self.g_base);
                if (self.has_charge) {
                    numerics.copySimd(self.c_vals, self.c_base);
                    numerics.zeroSimd(self.q_vec);
                }
                numerics.zeroSimd(self.rhs);
            },
            .full => {
                numerics.zeroSimd(self.g_vals);
                if (self.has_charge) {
                    numerics.zeroSimd(self.c_vals);
                    numerics.zeroSimd(self.q_vec);
                }
                numerics.zeroSimd(self.rhs);
            },
        }
    }

    /// Stamps only the charges at `x`: `q_vec` and every batch's `q_tape`,
    /// bit-for-bit what `eval` would leave there, with g/c/rhs untouched.
    /// ParEval reuses the lane cuts and reduce order of `.full`, so the
    /// promise holds threaded too. With a context attached the device runs
    /// its charge-only kernels (`GpuContext.evalQ`), or the full pass when
    /// one is missing, which changes g, c and rhs too; the resident batches'
    /// tapes reach the host at `snapshotQTape`.
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
    /// DC values. Leaves the memo invalid, because the planes then hold the
    /// AC linearization that a later DC eval must not reuse. Fails with
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
            // The DC values were installed by this circuit, so they cannot re-wire it.
            self.recompute() catch {};
            self.lin.valid = false;
        }
        try self.recompute();
        self.lin.valid = false;
        self.eval(x_op, 0);
    }

    /// Writes the frequency-dependent term of every `ac_dyn_slots` entry e at
    /// `x` and each omega in `omegas` into `re[e * omegas.len + k]` and
    /// `im[...]`, under the stored sim state: A(w) = G + jwC + (re + j im)
    /// with G and C from a `.ac`/`.noise` eval at the same x. Host-side, GPU
    /// context or not. Asserts `re` and `im` hold `ac_dyn_slots.len *
    /// omegas.len` entries.
    pub fn acDyn(self: *const Circuit, x: []const f64, omegas: []const f64, re: []f64, im: []f64) void {
        assert(re.len == self.ac_dyn_slots.len * omegas.len);
        assert(im.len == re.len);
        var off: usize = 0;
        for (self.batches) |b| if (b.hooks.ac_dyn) |f| {
            off += omegas.len * f(b.ctx, x, omegas, re[off..], im[off..]);
        };
        assert(off == re.len);
    }

    /// Adds `acDyn` at `x` and one omega into the row-major 2n x 2n
    /// stacked-real `a` ([Re -Im; Im Re], `dense_lu.buildComplexAdmittance`'s
    /// layout). Asserts that `scratch` holds `2 * ac_dyn_slots.len` values.
    pub fn addAcDynDense(self: *const Circuit, x: []const f64, omega: f64, a: []f64, scratch: []f64) void {
        const e = self.ac_dyn_slots.len;
        if (e == 0) return;
        assert(scratch.len >= 2 * e);
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
        // The device path ignores the baseline: `g_base` includes the
        // resident batches too, and the GPU sink cannot skip them, so it
        // zeroes and restamps instead.
        if (self.gpuContext()) |g| return self.evalGpu(g, x, t, false);
        self.evalNewtonCpu(x, t);
    }

    /// `evalNewton` for a companion step whose matrix is G + alpha C, read
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

    /// Returns the Newton matrix G + alpha C after `evalNewtonGC` at the same
    /// alpha: `combineGC` into `out`, the A the device brought down, or, for
    /// the device LU, `out` with only the diagonal written, which
    /// `deviceSolve` completes.
    pub fn newtonGC(self: *Circuit, alpha: f64, out: []f64) []f64 {
        if (self.gc != .full) if (self.gpuContext()) |g| {
            assert(@as(u64, @bitCast(alpha)) == @as(u64, @bitCast(self.gc_alpha)));
            return g.gcVals(self.gc, alpha, out, self.diag_slots[0..self.n], self.nnz);
        };
        self.combineGC(alpha, out);
        return out;
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
        assert(self.g_base.len == self.nnz + 1);
        @memset(self.g_base, 0);
        @memset(self.c_base, 0);
        const x_zero = try self.gpa.alloc(f64, self.n + 1);
        defer self.gpa.free(x_zero);

        @memset(x_zero, 0);
        self.lin.valid = false;
        self.q_fresh = false;
        numerics.zeroSimd(self.rhs);
        if (self.has_charge) numerics.zeroSimd(self.q_vec);
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

    /// Returns one `G + alpha C` entry, the same expression `combineGC` uses,
    /// so the Newton residual gate can read a diagonal in O(1) instead of
    /// rebuilding the whole plane.
    pub fn gcAt(self: *const Circuit, alpha: f64, slot: u32) f64 {
        return self.g_vals[slot] + alpha * self.c_vals[slot];
    }

    /// Writes `G + alpha C` into `out[0..nnz]`, leaving the trash entry
    /// alone. Asserts `out` holds at least `nnz` entries.
    pub fn combineGC(self: *const Circuit, alpha: f64, out: []f64) void {
        assert(out.len >= self.nnz);
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

    /// Returns the plane index of (row, col), or null when the pattern has
    /// no such entry. O(log) in the column's length. Asserts `col < n`.
    pub fn findSlot(self: *const Circuit, row: u32, col: u32) ?u32 {
        const pattern: abi.PatternView = .{
            .col_ptr = self.col_ptr,
            .row_idx = self.row_idx,
            .n = self.n,
            .trash_slot = self.trash_slot,
        };
        return pattern.findSlot(row, col);
    }

    /// Returns the netlist name of unknown `node`, or "" for a branch unknown
    /// or a row past the circuit.
    pub fn nodeName(self: *const Circuit, node: u32) []const u8 {
        // u64: maxInt(u32) + 1 wraps a 32-bit usize.
        if (@as(u64, node) + 1 < self.intern_offs.len)
            return self.intern_bytes[self.intern_offs[node]..self.intern_offs[node + 1]];
        return "";
    }

    // ---- private ----

    /// Adds the ground pin. Runs once, after every batch stamp and reduction.
    fn groundStamp(self: *Circuit, x: []const f64) void {
        self.g_vals[self.diag_slots[0]] += 1.0;
        self.rhs[0] += x[0];
    }

    /// The device stamp at `x` (`GpuContext.eval`), then the ground pin, or
    /// the host stamp after a driver fault `g` has already reported. The
    /// planes follow the buffer the device wrote, which a prefetch moves.
    fn evalGpu(self: *Circuit, g: *GpuContext, x: []const f64, t: f64, charge: bool) void {
        const ok = g.eval(x, t, self.sim, charge);
        self.bindPlanes(g.planes());
        if (ok) self.groundStamp(x) else self.evalNewtonCpu(x, t);
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

    /// Whether `evalNewtonGC` may hand A to the device, and whether the
    /// device LU takes it there (`GpuLu.residentReady`). Null under a JFNK
    /// pin, which reads g and c apart, with load-check rows, which read A
    /// off the diagonal, and under `-Dcertify`, which restamps.
    fn gcDevice(self: *Circuit) ?bool {
        if (certify_mod.enabled or self.load_check != null) return null;
        switch (self.knobs.newton.pin) {
            .auto, .direct => {},
            else => return null,
        }
        const lu = self.lu orelse return false;
        const ws = if (self.ws) |*w| w else return false;
        return lu.residentReady(&ws.slv);
    }

    /// Host half of `evalNewton`, also the GPU context's fault fallback.
    fn evalNewtonCpu(self: *Circuit, x: []const f64, t: f64) void {
        if (self.has_baseline) self.stamp(x, t, .newton) else self.stamp(x, t, .full);
    }

    fn denseFrom(self: *const Circuit, vals: []const f64, out: []f64) void {
        const n: usize = self.n;
        assert(out.len >= n * n);
        @memset(out[0 .. n * n], 0);
        for (0..n) |j| {
            for (self.col_ptr[j]..self.col_ptr[j + 1]) |p| {
                out[@as(usize, self.row_idx[p]) * n + j] = vals[p];
            }
        }
    }
};

/// Freezes `protos` into an owning Circuit (test construction); see
/// `circuit_life.freeze`.
pub const init = circuit_life.freeze;

/// Writes `out = g + alpha c` over independent CSC entries, rounded as the
/// scalar expression (no FMA contraction). `out` may alias `g`. Asserts
/// that the three slices have one length.
pub fn combinePlanes(out: []f64, g: []const f64, c: []const f64, alpha: f64) void {
    assert(out.len == g.len);
    assert(out.len == c.len);
    for (out, g, c) |*o, gv, cv| o.* = gv + alpha * cv;
}

test {
    _ = knobs;
    _ = @import("load_check.zig");
}
