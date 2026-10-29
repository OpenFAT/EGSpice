//! The neutral device ABI: the types every separately compiled side agrees
//! on, the host, the per-model device objects and runtime-loaded `.so`
//! devices. Construction (`Proto`, `PatternBuilder`) in, the frozen per-type
//! `Batch` with its `Hooks` and the `DeviceVtable` out. Everything that
//! crosses an object boundary is in `layoutHash`; a change no size, offset
//! or alignment shows bumps `abi_version`.
//! Cites VerA's device contract (`contract`: `SimState`, `StateCtlOp`,
//! noise terms), the LRM sections named on each hook, and ngspice's
//! nevalsrc.c for the noise row shape; `abi_version`'s comment records why
//! each ABI version moved.
//!
//! Contents:
//! - `pub const bind`, `Param`, `BindStatus` (bind.zig)
//! - `pub const GROUND: u32`, `pub const DeviceType`
//! - `pub const gpu_block_size: u32`, `pub const gpu_lane_block_size: u32`
//! - `pub const StateCtlOp`, `SimState`, `PsdTerm`, `Systf`, `SystfHost` (contract)
//! - `pub const BypassTol = struct`
//! - `pub const ParamField = struct`, `pub const ParamLayout = struct` (`ptr`)
//! - `pub const ParamRef = struct` (`get`, `set`)
//! - `pub const NoiseSource = struct` (`tableAt`, `groupEnd`)
//! - `pub const NoiseGenKind = enum`, `pub const NoiseGen = struct`
//! - `pub const Planes = struct`, `pub const StampMode = enum(u8)`
//! - `pub const ScatterTapes = struct`, `pub const ChargeIdentity = struct`
//! - `pub const DerivativeCounts = struct`, `pub const Batch = struct`
//! - `pub const StatusHit = struct`, `pub const DeviceStatus = enum(u8)`
//! - `pub const Hooks = struct`, `pub const Proto = struct`
//! - `pub const PatternView = struct` (`findSlot`)
//! - `pub const PatternBuilder = struct` (`add`, `reserve`, `deinit`, `toCsc`)
//! - `pub const lim_pass_limit: u64`, `pub const lim_pass_state: u64`
//! - `pub const GpuPayload = struct`
//! - `pub const abi_version: u32`
//! - `pub const DeviceVtable = struct`
//! - `pub fn DeviceResult(comptime T: type) type` (`fromLocal`, `unwrap`)
//! - `pub fn layoutHash() u64`

const std = @import("std");
const builtin = @import("builtin");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;
const contract = @import("contract");
const core = @import("core");
/// The parameter binder, here so eval.zig compiles it into every device object.
pub const bind = @import("bind.zig");

/// The ground node id, re-exported because device code imports this module
/// and not core.
pub const GROUND = core.GROUND;
/// A dense device-type id (`Library`), re-exported like `GROUND`.
pub const DeviceType = core.DeviceType;
/// One card `name=value` pair, as `DeviceVtable.bind_model` takes it.
pub const Param = bind.Param;
/// The binder's result, an enum because it crosses the object boundary.
pub const BindStatus = bind.BindStatus;

/// Threads per GPU block, shared by the kernel exports and the launcher.
pub const gpu_block_size: u32 = 256;
/// Threads per block of every lane kernel (phase 4), exported and launched
/// at this width: HIP bakes it into `globalIdX`.
pub const gpu_lane_block_size: u32 = 64;

/// VerA's accepted-step control operation (`Hooks.state_ctl`).
pub const StateCtlOp = contract.StateCtlOp;
/// VerA's analysis state, passed by value to every device call; in
/// `layoutHash` because GPU kernels take it too.
pub const SimState = contract.SimState;
/// A device's `noisePsd` result, one per `noise_gens` entry.
pub const PsdTerm = contract.PsdTerm;
/// VerA's LRM 12.32 call table entry, re-exported for hosts that do not
/// import the contract.
pub const Systf = contract.Systf;
/// The application a device's `$systf` calls go through.
pub const SystfHost = contract.SystfHost;

/// `Hooks.set_bypass`'s tolerances: the Newton test's.
pub const BypassTol = struct { reltol: f64, vntol: f64, abstol: f64 };

/// One sweepable field of a device type, as `collect_params` lists it.
pub const ParamField = struct {
    name: []const u8,
    /// Byte offset of the field in its Instance or Model struct.
    offset: u32,
    is_instance: bool,
    /// The device's principal parameter: its first Model parameter.
    primary: bool,
};

/// A batch's parameters as `fields.len` runs of `count` each, in
/// `collect_params` order: parameter `ord` is field `ord / count` of
/// instance `ord % count` (its own row, or the Model row it reads).
/// A read-only view: instances may share a Model row, so a write through
/// `ptr` can reach other instances; `Hooks.write_params` writes.
pub const ParamLayout = struct {
    count: u32,
    fields: []const ParamField,
    /// Instance and Model rows at their byte strides: parameter `ord`'s
    /// address is `base + row * stride + fields[ord / count].offset`, the
    /// row `ord % count` for an instance field and `model_of[ord % count]`
    /// for a Model field. Valid until the rows move.
    instances: [*]u8,
    instance_stride: u32,
    models: [*]u8,
    model_stride: u32,
    model_of: [*]const u32,

    /// Returns the address of parameter `ord`. Asserts that `ord` is below
    /// `count * fields.len`.
    pub fn ptr(l: ParamLayout, ord: u32) *f64 {
        assert(l.count > 0);
        assert(ord / l.count < l.fields.len);
        const f = l.fields[ord / l.count];
        const id = ord % l.count;
        const base = if (f.is_instance) l.instances + @as(usize, id) * l.instance_stride else l.models + @as(usize, l.model_of[id]) * l.model_stride;
        // SAFETY: `offset` is an f64 field's `@offsetOf` in a row the batch
        // allocated at its type's alignment.
        return @ptrCast(@alignCast(base + f.offset));
    }
};

/// A handle to one scalar parameter of one instance or model, for sweeps,
/// Monte Carlo and sensitivity. Points into the batch's Model/Instance
/// storage, which must outlive it.
pub const ParamRef = struct {
    /// The field's address; VerA emits every parameter as f64.
    ptr: *f64,
    /// Filled by the host when it collects the batch's parameters.
    type: DeviceType = .unset,
    param_name: []const u8,
    /// Instance index within the batch.
    index: u32,
    is_instance: bool,
    /// The device's principal parameter: its first Model parameter.
    primary: bool,
    pelgrom_ap: f64 = 0,
    area_wl: f64 = 0,

    /// Returns the parameter's value.
    pub fn get(self: ParamRef) f64 {
        return self.ptr.*;
    }

    /// Writes the parameter. The caller must run the batch's `recompute`
    /// hook before the next solve.
    pub fn set(self: ParamRef, v: f64) void {
        self.ptr.* = v;
    }
};

/// One noise generator row between two nodes. Its unit-coefficient shape is
/// `S(f) = white + flicker / f^ef + T(f)` in the contributed nature's units²
/// per Hz, `T` the tabulated PSD (`tableAt`, zero for a parametric row), and
/// the row injects `coeff` times that generator. A density, not a kind:
/// shot and thermal sources both arrive as a white density, as in ngspice's
/// NevalSrc (nevalsrc.c:105-113).
///
/// Rows of one correlated generator (LRM §4.6.4.6, contract
/// `NoiseGen.source`) are contiguous in a collected list and share `group`,
/// the list index of their first row, so a consumer sums their transfers
/// `Σ coeff·(y[node_p] − y[node_n])` as phasors before squaring and prices
/// the sum with the first row's shape (`groupEnd`). An independent row is a
/// group of one. Within a group only the first row's shape counts.
// ponytail: no §4.6.4.6 partial correlation between distinct generators
// (`PsdTerm.corr_with`/`corr`; `collect_noise` asserts it null): VerA never
// emits it, and shared-`source` groups cover fully correlated generators.
// Once a device publishes it: a partner index (the partner group's first
// row) and `corr` here, the cross term 2·corr·sqrt(S_a·S_b)·Re(H_a·conj(H_b))
// once per pair in `.noise` (ac/noise.zig), and a refusal in the other
// consumers (ac/sp, tran_noise, pnoise, phasenoise, mhb_lptv).
pub const NoiseSource = struct {
    node_p: u32,
    node_n: u32,
    white: f64 = 0,
    flicker: f64 = 0,
    ef: f64 = 1,
    /// The signed factor the contribution applies (contract `PsdTerm.coeff`).
    coeff: f64 = 1,
    /// List index of this row's group's first row; `maxInt` reads as the row
    /// itself, so a hand-built list of independent rows can leave it.
    group: u32 = std.math.maxInt(u32),
    /// The §4.6.4.3/.4 table of a table row; empty `points` otherwise. The
    /// slice is the device's static data, or the instance's Model row when
    /// the card sets the knots (`contract.noiseTable`): it lives while the
    /// batch's rows do.
    table: contract.NoiseTable = .{ .interp = .linear, .points = &.{} },

    /// Returns the tabulated PSD at `f` (contract `noiseTableAt`), or 0 for
    /// a parametric row.
    pub fn tableAt(self: NoiseSource, f: f64) f64 {
        return if (self.table.points.len == 0) 0 else contract.noiseTableAt(self.table, f);
    }

    /// Returns the end of the group starting at `srcs[lead]`: rows
    /// `[lead, end)` are one correlated generator. Asserts that `lead` is in
    /// range and starts a group. O(group length).
    pub fn groupEnd(srcs: []const NoiseSource, lead: usize) usize {
        assert(lead < srcs.len);
        assert(srcs[lead].group == lead or srcs[lead].group == std.math.maxInt(u32));
        var end = lead + 1;
        while (end < srcs.len and srcs[end].group == lead) end += 1;
        assert(end > lead);
        assert(end <= srcs.len);
        return end;
    }
};

/// The physical origin a device declares for a noise generator.
pub const NoiseGenKind = enum { thermal, shot, flicker };
/// A device's generator declaration: local rows `row`/`col` and its kind.
pub const NoiseGen = struct { row: usize, col: usize, kind: NoiseGenKind };

/// The value planes one eval pass stamps into. The serial path points them
/// at the circuit's own slices; ParEval points them at each batch's window
/// of its staging arrays and calls `Hooks.stage`.
pub const Planes = struct {
    g_vals: []f64,
    c_vals: []f64,
    rhs: []f64,
    q_vec: []f64,
};

/// Which planes a stamp writes. `.charge` is the transient's post-accept
/// re-read: `q_vec` and the per-batch `q_tape` only.
pub const StampMode = enum(u8) { full, newton, charge };

/// A batch's scatter tapes, `count * k` slot entries (g and c) and `count *
/// n_u` row entries (rhs and q), instance-major. Ground entries hold the
/// trash slot or row.
pub const ScatterTapes = struct {
    slots: []const u32,
    rows: []const u32,
};

/// Stable identity of a captured LTE charge: the original staged instance
/// index (before collapsed-batch partitioning), generated charge-site index,
/// and current circuit rows of its local unknowns. The rows are borrowed.
pub const ChargeIdentity = struct {
    instance: u32,
    site: u32,
    rows: []const u32,
};

/// Static evaluator partition counts; querying them performs no device work.
pub const DerivativeCounts = struct {
    wide: u32,
    narrow: u32,
    wide_lanes: u8,
    narrow_lanes: u8,
};

/// One device type's frozen batch, made by `Proto.finalize`. `eval` and
/// `eval_newton` add the stamps of instances `[first, last)` into the given
/// planes; `eval_newton` skips a constant Jacobian half. ParEval stamps
/// through `Hooks.stage` instead. Both assert `first <= last <= count`.
pub const Batch = struct {
    // hot
    ctx: *anyopaque,
    eval: *const fn (*anyopaque, *const Planes, first: u32, last: u32, []const f64, f64) void,
    eval_newton: *const fn (*anyopaque, *const Planes, first: u32, last: u32, []const f64, f64) void,
    count: u32,
    n_u: u32,
    has_charge: bool,
    has_const_jacobian: bool,

    // cold
    /// The device type's name without its namespace.
    type_name: []const u8,
    hooks: *const Hooks,
    /// The type runs a digital engine (a Verilog `.v` module) whose time
    /// only moves forward, so analyses that rewind time refuse it
    /// (`Circuit.refuseDigital`). The frontend copies it from
    /// `Library.digital` at the freeze.
    digital: bool = false,
};

/// Where `Hooks.status` found a latched status: the instance index within
/// the batch and the length of the message it wrote.
pub const StatusHit = struct { index: u32, len: u32 };

/// Callback status that is safe across separately compiled objects, unlike
/// Zig error values, whose numbering is per compilation.
pub const DeviceStatus = enum(u8) { ok = 0, out_of_memory = 1, too_many_instances = 2 };

/// Per-type hooks, one static table per device type. A null entry means the
/// device lacks the feature.
pub const Hooks = struct {
    /// A fresh mutable batch from an unevaluated template. The template and
    /// its tapes must outlive every instance made from it.
    instantiate: *const fn (*const anyopaque, Allocator) DeviceResult(Batch),
    /// A copy of an accepted batch, mutable history included.
    snapshot: *const fn (*const anyopaque, Allocator) DeviceResult(Batch),
    /// Overwrites the first batch's mutable state (models, instances,
    /// history, limiting state) with the second's. Both must be this type
    /// and one a `snapshot` of the other, or of a common batch. Allocates
    /// nothing, so `ParamRef`s into the first stay valid.
    copy_state: *const fn (*anyopaque, *const anyopaque) void,
    /// Syncs the host limiting flag after GPU state is downloaded.
    set_limit_active: ?*const fn (*anyopaque, bool) void = null,
    /// Stamps instances `[first, last)` in `mode` with every stamp stored at
    /// its own position, relative to the batch: g/c at its `slots` index,
    /// rhs/q at `id * n_u + ru`. A position the pass does not stamp is left
    /// alone. ParEval sums the positions into each plane cell in tape order,
    /// which is the order the serial `eval` adds them in.
    stage: *const fn (*anyopaque, *const Planes, first: u32, last: u32, []const f64, f64, StampMode) void,
    /// The batch's scatter tapes: plane cell of every staged position.
    scatter_tapes: *const fn (*anyopaque) ScatterTapes,
    /// Runs the device's `limit` on every instance at x, against x_old (the
    /// previous limited value once limiting is active), into the batch's lim
    /// plane, which later evals read; x itself is not written. Turns limiting
    /// on. Returns true when any instance reports its clamp not converged.
    apply_limits: ?*const fn (*anyopaque, []f64, []const f64) bool = null,
    /// Turns limiting off, so the next eval reads x unlimited.
    clear_limits: ?*const fn (*anyopaque) void = null,
    /// `apply_limits` over instances `[first, last)` without turning
    /// limiting on, so threads can split a batch; `arm_limits` turns it on
    /// once every range is done. Each instance writes only its own lim
    /// cells, so the split changes no value.
    limit_range: ?*const fn (*anyopaque, []f64, []const f64, u32, u32) bool = null,
    arm_limits: ?*const fn (*anyopaque) void = null,
    /// `update_state` over instances `[first, last)`, then `update_done`
    /// once every range is done.
    update_range: ?*const fn (*anyopaque, []const f64, u32, u32) ?f64 = null,
    update_done: ?*const fn (*anyopaque) void = null,
    /// Latency bypass for Newton evaluations (`StampMode.newton`), or null
    /// to turn it off: an instance whose limited unknowns all moved by at
    /// most `reltol * max(|v|, |v_ref|) + vntol` (`abstol` on a current)
    /// since its last full evaluation, with no live limiting correction,
    /// the same iteration class and unchanged state bytes, replays that
    /// evaluation's stamps instead of running its model core. Changes
    /// values within the Newton tolerance, so it is opt-in. Null for a
    /// device it cannot serve (sources, history, mutable evaluation).
    set_bypass: ?*const fn (*anyopaque, ?BypassTol) void = null,
    /// Advances Newton-history state between evaluated iterates, given the
    /// previous x.
    advance_iteration: ?*const fn (*anyopaque, []const f64) void = null,
    /// False when any instance's own convergence test fails at x.
    check_convergence: ?*const fn (*anyopaque, []const f64) bool = null,
    /// Overwrites the batch's internal voltage unknowns in `trial`, the
    /// transient's first Newton iterate, with `cur + xfact * (cur - prev)`:
    /// the last two accepted points extrapolated at the device's own ratio.
    /// Only models whose load predicts over another step than the host's
    /// dt/dt_prev have it (`predictsOverDeltaOld2` in eval.zig).
    predict_first_iterate: ?*const fn (*anyopaque, trial: []f64, cur: []const f64, prev: []const f64, xfact: f64) void = null,
    /// Sets `mask[row]` for each internal node voltage of a model whose
    /// ngspice load keeps its own current convergence test (the host's
    /// `Circuit.loadCheck`); every device current into those rows is the
    /// model's own.
    mark_load_check_rows: ?*const fn (*anyopaque, mask: []bool) void = null,
    /// Fills the lim plane from the device's `seed` values, x where it gives
    /// none, and turns limiting on. Reads x only.
    seed: ?*const fn (*anyopaque, []f64) void = null,
    /// Writes each LRM §4.5.4 `idt` initial condition into its operator
    /// unknown in `x`, for a transient that skips the operating point
    /// (`uic`), which would otherwise have solved it. Reads the device's
    /// static form, so the host sets a static sim state (dt = 0) first. An
    /// idt without ic keeps its value.
    seed_ic: ?*const fn (*anyopaque, []f64) void = null,
    /// Writes each §3.6.3.2 net initializer (contract `u_nodeset`) into its
    /// unknown in `x`, ground excepted: a cold start's Newton guess, never
    /// held (VerA VD-117).
    seed_nodeset: ?*const fn (*anyopaque, []f64) void = null,
    /// Sets `mask[row]` for each branch-current unknown, unless the same row
    /// is also one of the instance's voltage unknowns.
    mark_current_rows: ?*const fn (*anyopaque, []bool) void = null,
    /// Runs `updateState` at x once per converged solve; returns the earliest
    /// requested rejection time, if any. `state_ctl(.revert)` takes it back
    /// exactly only when it is the one `updateState` since the last commit
    /// or revert.
    update_state: ?*const fn (*anyopaque, []const f64) ?f64 = null,
    /// `update_state` for delay-line history (`absdelay`, the native lines):
    /// called once per accepted transient point instead, before
    /// `state_ctl(.commit)`, so no static solve pushes the ring.
    commit_state: ?*const fn (*anyopaque, []const f64) ?f64 = null,
    /// VerA `stateCtl`: `.commit` at every accepted point, the operating
    /// point included, `.revert` after a rejected attempt, `.query` for a
    /// cross/above flip since the last commit. Commit runs once at
    /// instantiation too, so a revert is always defined.
    state_ctl: ?*const fn (*anyopaque, StateCtlOp) bool = null,
    /// Sets every instance's temperature, in Celsius.
    set_temp: ?*const fn (*anyopaque, f32) void = null,
    /// Sets §9.15 `$simparam("gmin")` (S) and `("sourceScaleFactor")` on
    /// every Model row, for the operating point's stepping rungs. Null when
    /// the model reads neither.
    set_homotopy: ?*const fn (*anyopaque, gmin: f64, source_scale: f64) void = null,
    /// Stores the analysis state (`$abstime`, timestep, `analysis()`,
    /// initial/final step, Newton iteration) that every later device call
    /// of this batch receives. One store per batch, so it is cheap enough
    /// to call per Newton iteration.
    set_sim_state: *const fn (*anyopaque, SimState) void,
    /// The model's smallest static delay (`D.delays`).
    min_delay: ?*const fn (*anyopaque) f64 = null,
    /// The tightest LRM §9.17.2 `$bound_step` any instance requested for the
    /// next step, or `inf`. Valid after `updateState` has run for the
    /// accepted step.
    bound_step: ?*const fn (*anyopaque) f64 = null,
    /// The earliest model or instance breakpoint strictly after t, or null.
    next_breakpoint: ?*const fn (*anyopaque, f64) ?f64 = null,
    /// The first instance with a latched VerA `$fatal`/`$error`
    /// (`vera_status__`): writes its `contract.formatStatus` text into `msg`,
    /// truncated to fit. Null when the type has no status sites; those types
    /// are `mutable_eval`, so their instances live on the host.
    status: ?*const fn (*anyopaque, msg: []u8) ?StatusHit = null,
    /// Runs instance `index`'s LRM §9.4 display tasks at x (VerA's `say`)
    /// and writes the lines they print to `w`, `name` standing for `%m`.
    /// The host calls it at each converged point before `update_state`, and
    /// prints the text only once the point is accepted. Null when the
    /// device has no display tasks; the built-ins are generated without
    /// them, so it is host-only.
    say: ?*const fn (*anyopaque, x: []const f64, index: u32, name: []const u8, w: *std.Io.Writer) void = null,
    /// Charge per instance and LTE site (`ddt()` sites not marked
    /// `vera_lte = 0`) from the last eval, instance-major. Lets the transient
    /// run its LTE per charge state, as ngspice's CKTterr does, instead of
    /// per summed matrix row. Null when the device has no `q`.
    q_tape: ?*const fn (*anyopaque) []const f64 = null,
    /// Maps a batch-local charge tape index without evaluating the device.
    charge_identity: ?*const fn (*anyopaque, usize) ChargeIdentity = null,
    derivative_counts: ?*const fn (*anyopaque) DerivativeCounts = null,
    /// Restamps `q_vec` and `q_tape` for instances `[first, last)` at x,
    /// leaving the other planes alone. Null when the device has no `q`.
    eval_q: ?*const fn (*anyopaque, *const Planes, u32, u32, []const f64, f64) void = null,
    /// `eval_q` and `update_state` at a converged point from one model-core
    /// run per instance (VerA's `acceptQ`), over instances `[first, last)`.
    /// The time is the published sim state's. Null when the device has no
    /// `acceptQ`: no charge, or it calls `$vera_reject_step`, whose request
    /// a charge return cannot carry. The host then calls the two hooks.
    accept_q: ?*const fn (*anyopaque, *const Planes, u32, u32, []const f64) void = null,
    /// Appends a `ParamRef` for every sweepable field of every instance and
    /// Model row. First gives every instance its own Model row, so a
    /// `GpuPayload.models` slice taken before the call is stale.
    collect_params: *const fn (*anyopaque, Allocator, *std.ArrayList(ParamRef)) DeviceResult(void),
    /// `collect_params`' order without a `ParamRef` per parameter: one run
    /// of `count` parameters per field, instance fields first. Leaves the
    /// Model rows shared, so it is for reading. A sweep planner over a
    /// large circuit holds no table (8.7M refs, 555 MB, for 10k BSIM4
    /// devices). Null: the host falls back to `collect_params`.
    param_layout: ?*const fn (*anyopaque) DeviceResult(ParamLayout) = null,
    /// Writes `values` to the parameters numbered `refs` that fall in this
    /// batch's `[lo, hi)` of the circuit's `collect_params` numbering. An
    /// instance gets its own Model row only when the instances sharing its
    /// row receive different writes, so a variant over a few devices copies
    /// a few rows, not one per instance. New rows make a `GpuPayload.models`
    /// slice taken before the call stale. Null: the host writes through
    /// `collect_params`.
    write_params: ?*const fn (*anyopaque, refs: []const u32, values: []const f64, lo: u32, hi: u32) DeviceResult(void) = null,
    /// Gives each instance `ids` names (ascending, distinct) a Model row no
    /// other instance reads: a copy when others read its row, which keep
    /// it. A write through `param_layout`'s address then reaches that
    /// instance alone, as after `collect_params`, while the rest of the
    /// batch keeps sharing. New rows make a `GpuPayload.models` slice taken
    /// before the call stale. Null: the host falls back to `collect_params`.
    isolate_rows: ?*const fn (*anyopaque, ids: []const u32) DeviceResult(void) = null,
    /// Appends every declared noise generator with its PSD at x. Temperature
    /// is the instance's own, already applied by the device.
    collect_noise: ?*const fn (*anyopaque, []const f64, Allocator, *std.ArrayList(NoiseSource)) DeviceResult(void) = null,
    /// The model's name for each generator `collect_noise` appends per
    /// instance, in that order (LRM §4.6.4 `name` argument; "" when absent).
    /// Its length is the per-instance generator count.
    noise_names: []const []const u8 = &.{},
    /// Appends the global CSC slot of every frequency-dependent small-signal
    /// entry (VerA `ac_dyn_slots`: §4.5.7 absdelay, §4.5.11 laplace, §4.5.12
    /// zi), instance-major. A ground entry reads as the trash slot. Under
    /// kind `.ac`/`.noise` the G and C planes leave these partials out, and
    /// `ac_dyn` supplies them. Null when the device has none.
    collect_ac_dyn: ?*const fn (*anyopaque, Allocator, *std.ArrayList(u32)) DeviceResult(void) = null,
    /// Writes the small-signal term of each `collect_ac_dyn` entry e at x and
    /// at every ω in `omegas`: `re[e * omegas.len + k]` and `im[...]`, so
    /// A(ω_k) = G + jω_kC + (re + j·im) per entry. Reads the stored sim state
    /// (`set_sim_state`), its t included. Returns the number of entries
    /// written; `re`/`im` must hold at least that many times `omegas.len`.
    ac_dyn: ?*const fn (*anyopaque, x: []const f64, omegas: []const f64, re: []f64, im: []f64) usize = null,
    /// Reruns parameter-derived state; false means the new parameters need
    /// a different topology than the frozen one.
    recompute: ?*const fn (*anyopaque) bool = null,
    /// The batch's GPU working set, or null when the type is not
    /// GPU-eligible, in which case the launcher keeps the whole circuit on
    /// the CPU.
    gpu_payload: ?*const fn (*anyopaque) GpuPayload = null,
    deinit: *const fn (*anyopaque, Allocator) void,
};

/// A device type's construction-time store. `pattern` adds its matrix
/// entries, `finalize` freezes it into a `Batch`, `destroy` frees it.
pub const Proto = struct {
    ctx: *anyopaque,
    type_name: []const u8,
    /// Adds every staged instance's matrix entries; the allocator is the
    /// builder's.
    pattern: *const fn (*anyopaque, Allocator, *PatternBuilder) DeviceResult(void),
    /// Builds the frozen batch, allocated with the given allocator, against
    /// the final pattern. The proto still needs `destroy` afterwards.
    finalize: *const fn (*anyopaque, Allocator, PatternView) DeviceResult(Batch),
    destroy: *const fn (*anyopaque, Allocator) void,
    /// Renumbers staged nodes through `perm` (old id to new); ids at or past
    /// `perm.len`, ground among them, stay as they are.
    apply_perm: *const fn (*anyopaque, []const u32) void,
};

/// A read-only CSC sparsity pattern. `trash_slot` (= nnz) is the slot
/// ground entries scatter into.
pub const PatternView = struct {
    col_ptr: []const u32,
    row_idx: []const u32,
    n: u32,
    trash_slot: u32,

    /// Returns the slot of (`row`, `col`), or null when the pattern lacks
    /// it. Binary search, O(log) in the column's length. Asserts that
    /// `col < n`, that `col_ptr` has n + 1 entries, and that column `col`
    /// is a well-formed range.
    pub fn findSlot(self: PatternView, row: u32, col: u32) ?u32 {
        assert(col < self.n);
        assert(self.col_ptr.len == @as(usize, self.n) + 1);
        var lo = self.col_ptr[col];
        var hi = self.col_ptr[col + 1];
        assert(lo <= hi);
        while (lo < hi) {
            const mid = lo + (hi - lo) / 2;
            if (self.row_idx[mid] < row) lo = mid + 1 else hi = mid;
        }
        if (lo < self.col_ptr[col + 1] and self.row_idx[lo] == row) {
            // A real slot is never the trash slot ground entries scatter into.
            assert(lo < self.trash_slot);
            return lo;
        }
        return null;
    }
};

/// Accumulates `(col << 32 | row)` keys, duplicates included, and turns them
/// into CSC with `toCsc`. Pass a scratch allocator, not the circuit's arena:
/// the keys and sort buffers die in `Circuit.freeze`, and an arena would
/// keep them (measured 14.6 MB on a 25,000-MOSFET deck).
pub const PatternBuilder = struct {
    keys: std.ArrayList(u64) = .empty,

    /// Records entry (`row`, `col`); duplicates merge in `toCsc`.
    /// Invalidates nothing the caller holds; the keys are private.
    pub fn add(self: *PatternBuilder, gpa: Allocator, row: u32, col: u32) Allocator.Error!void {
        try self.keys.append(gpa, (@as(u64, col) << 32) | row); // col-major sort order
    }

    /// Makes room for `extra` more `add`s.
    pub fn reserve(self: *PatternBuilder, gpa: Allocator, extra: usize) Allocator.Error!void {
        try self.keys.ensureUnusedCapacity(gpa, extra);
        assert(self.keys.capacity - self.keys.items.len >= extra);
    }

    /// Frees the keys; `gpa` must be the allocator every `add` used.
    pub fn deinit(self: *PatternBuilder, gpa: Allocator) void {
        self.keys.deinit(gpa);
    }

    /// Sorts and dedups the keys into an `n`-column CSC pattern and returns
    /// nnz. Rows come out ascending within each column. O(keys): a counting
    /// sort by column, then each column's few rows sorted in place, two
    /// passes over the keys where a 16-bit radix sort makes four.
    /// Caller owns `col_ptr_out` (n + 1 entries) and `row_idx_out` (nnz) and
    /// must free them with `gpa`; `scratch` is only used during the call.
    /// On error neither output is written. Asserts that every row and column
    /// is below `n`.
    pub fn toCsc(self: *PatternBuilder, gpa: Allocator, scratch: Allocator, n: u32, col_ptr_out: *[]u32, row_idx_out: *[]u32) Allocator.Error!u32 {
        const all = self.keys.items;
        const col_ptr = try gpa.alloc(u32, n + 1);
        errdefer gpa.free(col_ptr);
        const rows = try scratch.alloc(u32, all.len);
        defer scratch.free(rows);
        const next = try scratch.alloc(u32, n);
        defer scratch.free(next);
        // Counting sort by column.
        @memset(col_ptr, 0);
        for (all) |k| {
            assert(k >> 32 < n);
            col_ptr[@intCast((k >> 32) + 1)] += 1;
        }
        for (0..n) |j| col_ptr[j + 1] += col_ptr[j];
        @memcpy(next, col_ptr[0..n]);
        for (all) |k| {
            const c: usize = @intCast(k >> 32);
            rows[next[c]] = @truncate(k);
            next[c] += 1;
        }
        // Each column sorted, then compacted toward the front: a write at
        // `m` never passes the read.
        var m: u32 = 0;
        var lo: u32 = 0;
        for (0..n) |j| {
            const hi = col_ptr[j + 1];
            const col = rows[lo..hi];
            std.mem.sortUnstable(u32, col, {}, std.sort.asc(u32));
            const first = m;
            for (col) |r| {
                assert(r < n);
                if (m != first and rows[m - 1] == r) continue;
                rows[m] = r;
                m += 1;
            }
            col_ptr[j] = first;
            lo = hi;
        }
        col_ptr[n] = m;
        const row_idx = try gpa.dupe(u32, rows[0..m]);
        assert(m <= all.len);
        col_ptr_out.* = col_ptr;
        row_idx_out.* = row_idx;
        return m;
    }
};

/// Bit of `arp_lim_<model>`'s `halves` argument that runs `limit` into the
/// lim plane. The launcher sets it once per Newton iterate.
pub const lim_pass_limit: u64 = 1;
/// Bit of `halves` that runs `updateState`. The launcher sets it once per
/// converged solve, at the converged x, never with `lim_pass_limit` (#15).
pub const lim_pass_state: u64 = 2;

/// One batch's GPU-resident working set, type-erased. Frozen for the solve:
/// the launcher uploads it once and afterwards moves only x in and the value
/// planes out per Newton iteration. `models` changes only when a sweep writes
/// a parameter; a host-stepped batch's `instances` also change at each host
/// state step, and the launcher re-uploads them (`GpuContext.pushStepped`).
/// The slices stay owned by the batch.
pub const GpuPayload = struct {
    /// `arp_eval_<model>`.
    kernel: []const u8,
    /// Instances in the batch, one GPU thread each.
    count: u32,
    /// Unknowns per instance; the tape stride.
    n_u: u32,
    /// `[]D.Model` and `[]D.Instance` as bytes. POD by contract, so a byte
    /// copy is the whole upload. `models` holds one row per distinct Model,
    /// fewer than `count` when instances share one.
    models: []const u8,
    /// count: instance i's row of `models`. Every kernel reads its Model
    /// through it.
    model_of: []const u32,
    instances: []const u8,
    /// count * n_u: the global row each local unknown gathers x from.
    gath: []const u32,
    /// count * n_u: the residual row each local unknown scatters to.
    rhs_idx: []const u32,
    /// The CSC slot each local Jacobian entry the device's jac/q pattern sets
    /// scatters to, row-major, the same number per instance.
    slots: []const u32,
    /// `arp_lim_<model>`, or "" when the device has no state kernel.
    lim_kernel: []const u8,
    /// `arp_ctl_<model>`, or "" when the device has no accepted-step latch.
    ctl_kernel: []const u8,
    /// `arp_reduce_<model>`, the segmented sum from staging cells to planes.
    reduce_kernel: []const u8,
    /// Host lim plane for the initial upload, one value per instance and
    /// unknown `limit` writes, empty without `limit`. Stale once the
    /// device-side state kernel has run.
    lim_x: []const f64,
    /// `[]D.State` as bytes, uploaded once; empty without `State`.
    states: []const u8,
    /// Host `lim_active`, used only until the device owns the lim plane.
    lim_active: bool,
    /// Per instance, the PULSE/PWL breakpoint cache (`SrcBrk`: req, at as
    /// f64 pairs) as bytes; empty for a model without spice breaks. Device
    /// step control starts each lane from it, so its corners round from the
    /// same request times as the host's.
    brk: []const u8 = &.{},
};

/// Runtime ABI version, mixed into `layoutHash`. `hashType` sees only sizes,
/// alignments and offsets, so a change that moves none of them (a tape's
/// meaning, a function signature) must bump this.
// 7: the slot tape's cleared entries are the device's structural Jacobian
//    zeros, not just ground; `addPattern` no longer reserves them.
// 8: `Hooks.eval_q` takes an instance range.
// 10: CPU callbacks return `DeviceResult`/`DeviceStatus` instead of Zig error
//    unions.
// 11: dropped `Batch.thread_safe`, `Hooks.record_history`/`inject_history`,
//    `DeviceVtable.gpu_kernel_name`/`gpu_ptx`/`gpu_amdgcn`; `Batch.type_name`
//    is the short type name.
// 12: `set_model_param`/`set_instance_param` became `bind_model`/
//    `bind_instance`; `ParamRef.type`.
// 13: dropped `ParamRef.device_type`.
// 14: VerA contract ABI 5. `SimState` (hashed) reaches every device call and
//    GPU kernel by value; `Hooks.begin_solve` is gone and `set_sim_state` is
//    required.
// 15: `Hooks.collect_ac_dyn`/`ac_dyn`, VerA's frequency-dependent entries.
// 16: `Hooks.copy_state`, `Batch.digital`.
// 17: `Hooks.predict_first_iterate`, `Hooks.mark_load_check_rows`.
// 18: `Hooks.accept_q`.
// GPU planes, Model/Instance PODs and scatter tapes are unchanged by 10 to 18.
// 19: instances share bit-identical Models: `GpuPayload.model_of`, and every
//    kernel takes `model_of` after `models`.
// 20: the slot tape holds only the device's pattern entries, not n_u^2.
// 21: the lim plane holds only the unknowns `limit` writes, not n_u.
// 22: `Hooks.status`, VerA's `$fatal`/`$error` channel.
// 23: `Hooks.set_homotopy`, VerA's host-written `gmin__`/`source_scale__`.
// 24: `NoiseSource.coeff` (no longer folded into white/flicker), `.group`
//    (correlated rows contiguous) and `.table`.
// 25: `DeviceVtable.unknown_names`.
// 26: `Hooks.say`, VerA's display tasks.
// 27: `GpuPayload.held`; the state kernel takes `halves` after `lim_active`,
//    so held-variable devices that limit (mos1/2/3/6/9, bsim4va,
//    vbic13_4t) are resident.
// 28: `Hooks.stage` and `Hooks.scatter_tapes` replace `scatter_bounds`;
//     `layoutHash` recurses into by-value struct fields (no layout moved);
//     the ngspice-port junction models' Model PODs gain `gmin__`, their gmin
//     default being `$simparam("gmin")` (#78).
// 29: dropped `GpuPayload.held` (dead since #15); `layoutHash` covers
//     `GpuPayload` and `ScatterTapes`, which cross through `Hooks` (#186).
// 30: `DeviceVtable.systf_calls`, the LRM 12.32 `$name` calls a device leaves
//     to a VPI application, so the host can refuse one by name, and
//     `bind_systf`, which binds the application into an Instance blob (#197).
// 31: lane kernels `arp_l{eval,lim,ctl,qtp}_<model>` beside each image's
//     `arp_eval_<model>` (phase 4), the host deriving their names; a host of
//     30 loads no lane kernel and a lane host refuses an image without them.
// 32: `Hooks.charge_identity`, `derivative_counts` and verified built-in `Hooks.linear_rc`.
// 33: `Hooks.collect_ac_dyn`/`ac_dyn` take ω vectors (gpu frequency lanes).
// 34: `ParamRef.ptr` is a bare `*f64`; dropped `Hooks.linear_rc` and `LinearRc`.
// 35: dropped `Hooks.apply_attempt`/`restore_models`: no VerA device declares
//     `attempt`; source stepping runs through §9.15 `sourceScaleFactor`.
// 36: `Hooks.param_layout`, `collect_params`' order without a `ParamRef`
//     per parameter; `limit_range`/`arm_limits` and `update_range`/
//     `update_done`, the limit and state passes split over threads;
//     `GpuPayload.brk`, the source breakpoint caches device step control
//     starts from; `set_bypass`, the opt-in latency bypass.
pub const abi_version: u32 = 36;

/// A device type's construction entry points, exported by each device object
/// and by runtime-loaded `.so` devices. Every blob pointer it takes must be
/// 16-byte aligned: the host aligns every Model and Instance blob to 16, and
/// no device field needs more.
pub const DeviceVtable = struct {
    name: []const u8,
    /// Unknowns per instance: the ports, then the internal nodes.
    n_u: u32,
    num_ports: u32,
    /// Bytes of the Model blob the model callbacks read and write.
    model_size: usize,
    /// Bytes of the Instance blob.
    instance_size: usize,
    /// Writes the declared defaults into a `model_size` blob.
    init_model: *const fn ([*]u8) void,
    init_instance: *const fn ([*]u8) void,
    /// Binds card pairs onto a Model / Instance blob (bind.zig).
    bind_model: *const fn ([*]u8, []const Param) BindStatus,
    bind_instance: *const fn ([*]u8, []const Param) BindStatus,
    /// Recomputes parameters declared as expressions of other parameters, and
    /// localparams (LRM 6.3.4, 3.4.5). The host must call it after the last
    /// `bind_model` and before anything reads the model. Null when the module
    /// has none.
    derive: ?*const fn (model: [*]u8) void,
    /// Writes, for each internal unknown in `[num_ports, n_u)`, the unknown
    /// it aliases, or -1.
    collapse: ?*const fn (model: [*]const u8, instance: [*]const u8, out: [*]i32) void,
    proto_create: *const fn (Allocator) DeviceResult(Proto),
    /// Stages one instance into a `proto_create` store, copying both blobs;
    /// `nodes` holds `n_u` global unknowns.
    proto_add: *const fn (ctx: *anyopaque, gpa: Allocator, model: [*]const u8, instance: [*]const u8, nodes: [*]const u32) DeviceResult(void),
    /// The Model row the last `proto_add` staged into a store that shares
    /// rows between equal Models. Null when the store keeps a row per
    /// instance.
    proto_last_row: ?*const fn (ctx: *anyopaque) u32 = null,
    /// `proto_add` of an instance whose Model is bit-for-bit row `row`'s,
    /// which `proto_last_row` returned: no copy, hash or compare of the
    /// Model. Non-null exactly when `proto_last_row` is.
    proto_add_row: ?*const fn (ctx: *anyopaque, row: u32, instance: [*]const u8, nodes: [*]const u32) DeviceResult(void) = null,
    /// Name of each unknown in `[0, n_u)`, VerA's `U`: the ports, the
    /// internal nets, then the branch flows (`flowZ28...`, mangled `flow(`).
    /// Empty when the device does not say.
    unknown_names: []const []const u8 = &.{},
    /// VerA's `systf_calls`: each `$name` call the device leaves to a VPI
    /// application (LRM 12.32), one per source call. Every one calls through
    /// `Instance.systf`, so the host must refuse an instance of a device
    /// whose names no application registered before staging it.
    systf_calls: []const contract.Systf = &.{},
    /// Writes `host` into an Instance blob's `systf`. Non-null exactly when
    /// `systf_calls` is non-empty; the host calls it before `proto_add`.
    /// `host` must outlive every batch the instance reaches.
    bind_systf: ?*const fn (instance: [*]u8, host: *const contract.SystfHost) void = null,
};

comptime {
    // Read whole per row and frequency; the 4 pad bytes are
    // `contract.NoiseTable`'s.
    if (@sizeOf(usize) == 8) assert(@sizeOf(NoiseSource) == 72);
    // Every eval call receives a `*const Planes`; nothing beyond the four
    // plane slices may ride along on that path.
    assert(@sizeOf(Planes) == 4 * @sizeOf([]f64));
    // Every Newton iteration walks the circuit's `[]Batch` and calls `eval`
    // through each; one cache line per batch (59 bytes of fields on a
    // 64-bit target, 64 with padding) keeps that walk one line per type.
    assert(@sizeOf(Batch) <= 64);
    // The two `halves` bits are distinct.
    assert(lim_pass_limit & lim_pass_state == 0);
    assert(gpu_block_size % gpu_lane_block_size == 0);
}

/// Returns a callback result type carried by value across the object
/// boundary. A successful payload keeps its existing owner.
pub fn DeviceResult(comptime T: type) type {
    return union(DeviceStatus) {
        ok: T,
        out_of_memory: void,
        too_many_instances: void,

        /// Encodes this compilation's error union for the trip across the
        /// object boundary.
        pub fn fromLocal(value: error{ OutOfMemory, TooManyInstances }!T) @This() {
            return .{ .ok = value catch |err| return switch (err) {
                error.OutOfMemory => .out_of_memory,
                error.TooManyInstances => .too_many_instances,
            } };
        }

        /// Decodes into the caller's own error values, whatever numbering the
        /// compilation that produced it used.
        pub fn unwrap(self: @This()) error{ OutOfMemory, TooManyInstances }!T {
            return switch (self) {
                .ok => |value| value,
                .out_of_memory => error.OutOfMemory,
                .too_many_instances => error.TooManyInstances,
            };
        }
    };
}

/// Returns the hash of every type that crosses the object boundary, the
/// compiler version, backend and mode, and `abi_version`. Both sides compile
/// this same source, so equal hashes mean compatible layouts. Comptime.
pub fn layoutHash() u64 {
    return comptime blk: {
        @setEvalBranchQuota(100_000);
        var h: u64 = 0xcbf29ce484222325;
        for (builtin.zig_version_string) |c| h = mix(h, c);
        h = mix(h, @backingInt(builtin.zig_backend));
        h = mix(h, @backingInt(builtin.mode));
        // Error tracing adds a hidden argument to every callconv(.auto) call.
        h = mix(h, @intFromBool(builtin.have_error_return_tracing));
        for ([_]type{
            DeviceVtable,        Proto,               Batch,
            Hooks,               Planes,              PatternView,
            PatternBuilder,      ParamRef,            NoiseSource,
            std.mem.Allocator,   DeviceStatus,        DeviceResult(void),
            DeviceResult(Batch), DeviceResult(Proto), Param,
            SimState,            StatusHit,           std.Io.Writer,
            GpuPayload,          ScatterTapes,        contract.Systf,
            contract.SystfHost,
        }) |T| h = hashType(h, T);
        h = mix(h, abi_version);
        break :blk h;
    };
}

// ---- private ----

/// One FNV-1a step.
fn mix(h: u64, v: u64) u64 {
    return (h ^ v) *% 0x100000001b3;
}

/// Folds `T`'s size, alignment and every field offset into `h0`, recursing
/// into by-value struct fields; a pointer's target is hashed only if listed.
fn hashType(h0: u64, comptime T: type) u64 {
    var h = mix(mix(h0, @sizeOf(T)), @alignOf(T));
    switch (@typeInfo(T)) {
        .@"struct" => |si| inline for (si.field_names, si.field_types, si.field_attrs) |name, FT, attrs| {
            if (attrs.@"comptime" or @sizeOf(FT) == 0) continue;
            h = mix(h, @offsetOf(T, name));
            // A by-value struct (`NoiseSource.table`) crosses with its parent.
            if (@typeInfo(FT) == .@"struct") h = hashType(h, FT);
        },
        else => {}, // else: a non-struct type hashes by size and alignment only
    }
    return h;
}

test {
    _ = bind;
}

test "hashType sees a field move inside a by-value nested struct" {
    const Outer = struct {
        fn of(comptime Inner: type) type {
            return extern struct { x: u32, inner: Inner };
        }
    }.of;
    const A = extern struct { a: u16, b: u8, c: u8 };
    const B = extern struct { a: u8, b: u8, c: u16 };
    comptime assert(@sizeOf(A) == @sizeOf(B) and @alignOf(A) == @alignOf(B));
    try std.testing.expect(hashType(0, Outer(A)) != hashType(0, Outer(B)));
}

test DeviceResult {
    const R = DeviceResult(u32);
    try std.testing.expectEqual(@as(u32, 7), try R.fromLocal(7).unwrap());
    try std.testing.expectError(error.OutOfMemory, R.fromLocal(error.OutOfMemory).unwrap());
    try std.testing.expectError(error.TooManyInstances, R.fromLocal(error.TooManyInstances).unwrap());
}

test ParamRef {
    var wide: f64 = 0;
    const b: ParamRef = .{ .ptr = &wide, .param_name = "l", .index = 0, .is_instance = true, .primary = true };
    b.set(0.1);
    try std.testing.expectEqual(@as(f64, 0.1), b.get());
}

test "ParamLayout.ptr reaches instance and shared Model fields" {
    const Inst = extern struct { pad: f64 = 0, m: f64 };
    const Model = extern struct { r: f64 };
    var insts = [_]Inst{ .{ .m = 1 }, .{ .m = 2 } };
    var models = [_]Model{.{ .r = 10 }};
    const model_of = [_]u32{ 0, 0 };
    const fields = [_]ParamField{
        .{ .name = "m", .offset = @offsetOf(Inst, "m"), .is_instance = true, .primary = false },
        .{ .name = "r", .offset = @offsetOf(Model, "r"), .is_instance = false, .primary = true },
    };
    const l: ParamLayout = .{
        .count = 2,
        .fields = &fields,
        .instances = @ptrCast(&insts),
        .instance_stride = @sizeOf(Inst),
        .models = @ptrCast(&models),
        .model_stride = @sizeOf(Model),
        .model_of = &model_of,
    };
    try std.testing.expectEqual(@as(f64, 2), l.ptr(1).*);
    // Both instances read the one Model row.
    try std.testing.expectEqual(&models[0].r, l.ptr(2));
    try std.testing.expectEqual(&models[0].r, l.ptr(3));
}

test "NoiseSource.groupEnd spans a correlated group and stops at the next" {
    const srcs = [_]NoiseSource{
        .{ .node_p = 1, .node_n = 0, .group = 0 },
        .{ .node_p = 2, .node_n = 0, .group = 0 },
        .{ .node_p = 3, .node_n = 0 },
        .{ .node_p = 4, .node_n = 0, .group = 3 },
    };
    try std.testing.expectEqual(@as(usize, 2), NoiseSource.groupEnd(&srcs, 0));
    try std.testing.expectEqual(@as(usize, 3), NoiseSource.groupEnd(&srcs, 2));
    try std.testing.expectEqual(@as(usize, 4), NoiseSource.groupEnd(&srcs, 3));
}

test "PatternView.findSlot: hits, misses and an empty column" {
    // Column 0 holds rows {0, 2}, column 1 nothing, column 2 row {1}.
    const v: PatternView = .{ .col_ptr = &.{ 0, 2, 2, 3 }, .row_idx = &.{ 0, 2, 1 }, .n = 3, .trash_slot = 3 };
    const expectSlot = struct {
        fn f(want: ?u32, got: ?u32) !void {
            try std.testing.expectEqual(want, got);
        }
    }.f;
    try expectSlot(0, v.findSlot(0, 0));
    try expectSlot(1, v.findSlot(2, 0));
    try expectSlot(null, v.findSlot(1, 0));
    try expectSlot(null, v.findSlot(3, 0));
    try expectSlot(null, v.findSlot(0, 1));
    try expectSlot(2, v.findSlot(1, 2));
    try expectSlot(null, v.findSlot(0, 2));
    try expectSlot(null, v.findSlot(std.math.maxInt(u32), 2));
}

test "PatternBuilder.toCsc matches a comparison sort of the keys" {
    const gpa = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x5eed);
    const r = prng.random();
    // Few columns (long ones, many duplicates) and many (short ones).
    for ([_]u32{ 1, 3, 70, 5000 }) |n| for ([_]usize{ 0, 1, 63, 1000 }) |len| {
        var pb: PatternBuilder = .{};
        defer pb.deinit(gpa);
        for (0..len) |_| try pb.add(gpa, r.uintLessThan(u32, n), r.uintLessThan(u32, n));
        const want = try gpa.dupe(u64, pb.keys.items);
        defer gpa.free(want);
        std.mem.sortUnstable(u64, want, {}, std.sort.asc(u64));
        var col_ptr: []u32 = undefined;
        var row_idx: []u32 = undefined;
        const nnz = try pb.toCsc(gpa, gpa, n, &col_ptr, &row_idx);
        defer gpa.free(col_ptr);
        defer gpa.free(row_idx);
        var k: usize = 0;
        for (want, 0..) |key, i| {
            if (i != 0 and want[i - 1] == key) continue;
            const c: u32 = @intCast(key >> 32);
            try std.testing.expect(k >= col_ptr[c] and k < col_ptr[c + 1]);
            try std.testing.expectEqual(@as(u32, @truncate(key)), row_idx[k]);
            k += 1;
        }
        try std.testing.expectEqual(k, nnz);
        try std.testing.expectEqual(@as(u32, nnz), col_ptr[n]);
    };
}

test "PatternBuilder.toCsc dedups, sorts rows and keeps empty columns" {
    const case = struct {
        fn run(gpa: Allocator) !void {
            var pb: PatternBuilder = .{};
            defer pb.deinit(gpa);
            // Column 1 stays empty; (2, 0) arrives twice and out of order.
            for ([_][2]u32{ .{ 2, 0 }, .{ 0, 0 }, .{ 2, 2 }, .{ 2, 0 }, .{ 1, 2 } }) |rc| try pb.add(gpa, rc[0], rc[1]);
            var col_ptr: []u32 = undefined;
            var row_idx: []u32 = undefined;
            const nnz = try pb.toCsc(gpa, gpa, 3, &col_ptr, &row_idx);
            defer gpa.free(col_ptr);
            defer gpa.free(row_idx);
            try std.testing.expectEqual(@as(u32, 4), nnz);
            try std.testing.expectEqualSlices(u32, &.{ 0, 2, 2, 4 }, col_ptr);
            try std.testing.expectEqualSlices(u32, &.{ 0, 2, 1, 2 }, row_idx);
        }
    }.run;
    try case(std.testing.allocator);
    try std.testing.checkAllAllocationFailures(core.testing.fixed_growth, case, .{});
}

test "PatternBuilder.toCsc of no keys is all-empty columns" {
    const gpa = std.testing.allocator;
    var pb: PatternBuilder = .{};
    defer pb.deinit(gpa);
    var col_ptr: []u32 = undefined;
    var row_idx: []u32 = undefined;
    try std.testing.expectEqual(@as(u32, 0), try pb.toCsc(gpa, gpa, 2, &col_ptr, &row_idx));
    defer gpa.free(col_ptr);
    defer gpa.free(row_idx);
    try std.testing.expectEqualSlices(u32, &.{ 0, 0, 0 }, col_ptr);
    try std.testing.expectEqual(@as(usize, 0), row_idx.len);
}

test "hashType sees a field's width" {
    const A = extern struct { a: u32, b: u32 };
    const B = extern struct { a: u32, b: u64 };
    try std.testing.expect(hashType(0, A) != hashType(0, B));
}
