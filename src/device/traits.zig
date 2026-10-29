//! What the host offers each device type, decided at comptime from its
//! declarations: a device type `D` in, its predicates out (timer-only
//! state, breakpoint caches, delay history, GPU eligibility, kernel names).
//! Each hand-kept name list says why it is opt-in beside it.
//! Cites LRM §5.10.3 (timers), §4.5 (`absdelay`), §4.5.4 (`idt`), §5.4.2
//! (branch flows), ngspice's hfetload.c, mesaload.c and hfet2load.c for the
//! hfet1 rules, and dev/devices/gpu-evaluation.md for the GPU rules.
//!
//! Contents:
//! - `pub const bypass_models`, `pub const load_check`
//! - `pub const GpuRefusal = enum`
//! - `pub fn skipsTimerState(comptime D: type) bool`, `hasSpiceBreaks`,
//!   `predictsOverDeltaOld2`, `hasAbsdelayState`, `walksPending`,
//!   `holdsOnlyHeld`, `hasInstanceField`, `gpuEligible`, `gpuHostStepped`,
//!   `statusOnlyMutable`, `hasStateKernel`, `hasCtlKernel`, `constJacobian`
//! - `pub fn noiseFlow(comptime D: type, comptime k: usize) ?usize`
//! - `pub fn noiseOrder(comptime D: type) struct { gen, lead }`
//! - `pub fn idtUnknowns(comptime D: type) []const u32`
//! - `pub fn AcTerms(comptime D: type) type`
//! - `pub fn gpuRefusal(comptime D: type) ?GpuRefusal`
//! - `pub fn baseName(comptime D: type) []const u8`
//! - `pub fn kernelSym(comptime D: type, comptime what: []const u8) [:0]const u8`

const std = @import("std");
const layout = @import("layout.zig");

/// Models whose evaluation reads no time (`sim.t`, `sim.dt`) beyond the
/// flags the bypass class keys on (the MOS levels test `t > 0` on the
/// initial step), so a recording stays valid across timepoints. An unlisted
/// model, a runtime `.hdl` one included, always evaluates.
// ponytail: hand-kept list (grep of the generated sources); a VerA flag for time reads would replace it.
pub const bypass_models = std.StaticStringMap(void).initComptime(.{
    .{"asmesd"},         .{"asmesd_dio"},   .{"asmhemt"},     .{"b3soidd"},        .{"b3soifd"},
    .{"bjt505"},         .{"bjt505t"},      .{"bsim1"},       .{"bsim2"},          .{"bsim3"},
    .{"bsim4va"},        .{"bsimbulk"},     .{"bsimcmg_std"}, .{"bsimimg"},        .{"bsimsoi"},
    .{"diode_cmc"},      .{"diode"},        .{"gummel_poon"}, .{"hfet1"},          .{"hfet2"},
    .{"hicumL0_v2p1p0"}, .{"hicumL2_v320"}, .{"hisim2_va"},   .{"hisimhv"},        .{"hisimsoi_std"},
    .{"hisimsotb_std"},  .{"jfet2"},        .{"jfet"},        .{"juncap200"},      .{"L_UTSOI_102"},
    .{"mesa"},           .{"mes"},          .{"mosvar"},      .{"mvsg_cmc_5.0.0"}, .{"psp103"},
    .{"psp103t"},        .{"psp104"},       .{"psp104t"},     .{"r2_cmc"},         .{"r3_cmc"},
    .{"vbic13_4t"},      .{"vdmos"},        .{"mos1"},        .{"mos2"},           .{"mos3"},
    .{"mos6"},           .{"mos9"},
});

/// Models whose ngspice load flags non-convergence itself when a branch
/// current misses its linear prediction from the previous iterate
/// (hfetload.c:389), checked host-side by `Circuit.loadCheck`. The MOS,
/// BJT and diode tests are dead in ngspice (NIiter overwrites CKTnoncon).
/// hfet2load.c:270 and mesaload.c:388 have the same test, but it moved
/// tran/device_mesa_oscillator away from ngspice (52.5x -> 65x: mesa.va's
/// series resistors are flow unknowns ngspice does not have, and their
/// delta test already adds an iteration) and op/device_hfet2 only by
/// roundoff, so both stay off.
pub const load_check = std.StaticStringMap(void).initComptime(.{.{"hfet1"}});

/// Why `gpuEligible` turns D down, one tag per rule above in order. The
/// inventory test in tests/eval.zig pins it per built-in model.
pub const GpuRefusal = enum { mutable_eval, newton_history, systf };

/// Models whose `updateState` only advances §5.10.3 timer bookkeeping
/// (`__analog_op__timer__*`). This host never reads it: breakpoints come from
/// the model-level `nextBreakpoint`, and the pass writes `bound_step = inf`
/// and `discontinuity_order = -1`, the Instance defaults. So the batch skips
/// `updateState`, the `bound_step` walk and `stateCtl`, whose twins then
/// never differ from the Instance. Opt-in by name because a
/// `$bound_step` or `$discontinuity` call leaves no trace in the Instance:
/// vsource.va and isource.va call neither (only `$abstime` and `@(timer)`);
/// re-check before adding one. `skipsTimerState` rejects any other state.
/// Measured on tran/rc_sinusoidal_startup: ~660 Ir per step.
const timer_only_state = std.StaticStringMap(void).initComptime(.{
    .{"vsource"},
    .{"isource"},
});

/// Models whose ngspice load extrapolates the first Newton iterate of a
/// transient step over CKTdeltaOld[2], the step two back, where the host
/// and the other models use the previous step (hfetload.c:129). Every
/// branch voltage hfet1 reads lies between its internal nodes gp/sp/dp, so
/// re-predicting only those reproduces ngspice's and leaves the ports'
/// prediction to the other devices on them. mesaload.c:150 and
/// hfet2load.c:98 predict the same way; mesa stays off because it moved
/// tran/device_mesa_oscillator away from ngspice (52.5x -> 57.7x), and
/// hfet2 reads its gate port directly.
const delta_old2_predict = std.StaticStringMap(void).initComptime(.{.{"hfet1"}});

pub fn skipsTimerState(comptime D: type) bool {
    if (timer_only_state.get(comptime baseName(D)) == null) return false;
    // Two substring scans per field name; VerA emits one field per timer.
    @setEvalBranchQuota(200_000 + 2_000 * @typeInfo(D.Instance).@"struct".field_names.len);
    for (@typeInfo(D.State).@"struct".field_names) |name| {
        if (!@hasField(D.Instance, name))
            @compileError(@typeName(D) ++ " carries State beyond stateCtl twins (" ++ name ++ "); drop it from timer_only_state");
    }
    for (@typeInfo(D.Instance).@"struct".field_names) |name| {
        if (std.mem.indexOf(u8, name, "__") != null and std.mem.indexOf(u8, name, "__analog_op__timer") == null)
            @compileError(@typeName(D) ++ " holds non-timer state (" ++ name ++ "); drop it from timer_only_state");
    }
    return true;
}

/// A source model whose PULSE and PWL corners follow ngspice's request-time
/// rounding (`spicePulseBreak`, `spicePwlBreak`) instead of VerA's timer
/// arithmetic.
pub fn hasSpiceBreaks(comptime D: type) bool {
    return @hasDecl(D, "nextBreakpoint") and skipsTimerState(D) and @hasField(D.Model, "pulse_td");
}

pub fn predictsOverDeltaOld2(comptime D: type) bool {
    return delta_old2_predict.get(comptime baseName(D)) != null;
}

/// Whether D's `updateState` pushes delay-line history, which only accepted
/// transient points may feed: LRM §4.5 `absdelay`. A VerA ring would
/// revert, but a push at the operating point seeds it with a static solve
/// (multi_analysis/bench_hb_tline_guard: 6.7e-8 to 0.66 of its tolerance).
/// VerA devices are recognized by the `__analog_op__absdelay__` infix its
/// naming scheme puts on an absdelay operator's Instance fields. A bare
/// `__absdelay__` also matches a field that merely carries the name (a
/// Verilog device's), and a device routed to `commit_state` never runs
/// `updateState` at the operating point, so its fixed point never forms.
// ponytail: field-name matching until VerA's contract marks delay history.
pub fn hasAbsdelayState(comptime D: type) bool {
    return hasInstanceField(D, "__analog_op__absdelay__");
}

/// The §5.4.2 branch-flow unknown of generator `noise_gens[k]`'s branch,
/// or null when the branch has none. A branch with a flow unknown is a
/// potential (or switch) branch, whose flow row VerA writes as
/// `V(p,n) − (contribution)` or `I − (contribution)`: a generator in the
/// contribution is a source in that row, not a current between the nodes
/// (`noise_gens` names only the nodes). VerA names the unknown
/// `flow(p,n)`, or `flow(p,gnd)` for a to-ground branch.
// ponytail: name matching, as `idtUnknowns`, until VerA's contract says
// which generators sit on a potential contribution.
pub fn noiseFlow(comptime D: type, comptime k: usize) ?usize {
    comptime {
        const names = @typeInfo(D.U).@"enum".field_names;
        const g = D.noise_gens[k];
        const col = if (g.row == g.col) "gnd" else names[g.col];
        const want = "flowZ28" ++ names[g.row] ++ "Z2c" ++ col ++ "Z29";
        for (names, 0..) |name, u| if (std.mem.eql(u8, name, want)) return u;
        return null;
    }
}

/// The order `collect_noise` emits D's generators in: `gen[j]` is the
/// `noise_gens` index of output row j, and `lead[j]` the row that starts
/// its group. Rows sharing a non-null `source` are contiguous and lead by
/// the first; the rest keep declaration order.
pub fn noiseOrder(comptime D: type) struct { gen: [D.noise_gens.len]usize, lead: [D.noise_gens.len]u32 } {
    const gens = D.noise_gens;
    const has_source = @hasField(@typeInfo(@TypeOf(gens)).array.child, "source");
    var gen: [gens.len]usize = undefined;
    var lead: [gens.len]u32 = undefined;
    var placed: [gens.len]bool = @splat(false);
    var j: u32 = 0;
    for (gens, 0..) |g, k| {
        if (placed[k]) continue;
        const first = j;
        const src: ?u16 = if (has_source) g.source else null;
        for (gens[k..], k..) |h, m| {
            if (m != k and (src == null or h.source != src)) continue;
            placed[m] = true;
            gen[j] = m;
            lead[j] = first;
            j += 1;
        }
    }
    return .{ .gen = gen, .lead = lead };
}

/// D's §4.5.4 `idt` operator unknowns, as local indices. VerA spells the
/// k-th one `idt$k`, escaped to `idtZ24k` in `D.U`.
// ponytail: name matching, as above, until VerA's contract marks operator
// unknowns.
pub fn idtUnknowns(comptime D: type) []const u32 {
    comptime {
        var out: []const u32 = &.{};
        if (@hasDecl(D, "U")) for (@typeInfo(D.U).@"enum".field_names, @typeInfo(D.U).@"enum".field_values) |name, value| {
            if (std.mem.startsWith(u8, name, "idtZ24")) out = out ++ [_]u32{value};
        };
        const final = out[0..out.len].*;
        return &final;
    }
}

/// Whether the host reads D's live per-instance timer schedule
/// (`pendingBreakpoint`, which holds §5.10.3.3 start times computed during
/// the solve). Not for timer-only models, whose `updateState` never runs so
/// `__next` is stale, nor for types whose state the GPU steps
/// (`hasCtlKernel`), whose host Instance is. A host-stepped type walks.
pub fn walksPending(comptime D: type) bool {
    return @hasDecl(D, "pendingBreakpoint") and !skipsTimerState(D) and !hasCtlKernel(D);
}

/// Whether D's history is §5.10 held variables and no analog operator (no
/// cross/above FSM, delay line or timer).
pub fn holdsOnlyHeld(comptime D: type) bool {
    return hasInstanceField(D, "__held__") and !hasInstanceField(D, "__analog_op__");
}

/// Whether some Instance field of D contains `infix`.
pub fn hasInstanceField(comptime D: type, comptime infix: []const u8) bool {
    if (!@hasDecl(D, "Instance")) return false;
    // hisim Instances have hundreds of long field names.
    @setEvalBranchQuota(2_000_000);
    inline for (@typeInfo(D.Instance).@"struct".field_names) |name| {
        if (std.mem.indexOf(u8, name, infix) != null) return true;
    }
    return false;
}

/// Where D's frequency-dependent terms come from: D itself, except ltra,
/// whose exact dispersive AC kernel no Verilog-A operator expresses
/// (models/ltra_ac.zig). build.zig puts it in ltra's own `models` wrapper,
/// so only ltra's device object (eval.zig as root) sees it.
pub fn AcTerms(comptime D: type) type {
    if (layout.object_root) {
        const models = @import("models");
        if (@hasDecl(models, "ltra_ac") and std.mem.eql(u8, baseName(D), "ltra")) return models.ltra_ac;
    }
    return D;
}

/// Whether D gets GPU kernels. Excluded, and kept on the host:
/// - `mutable_eval` devices, whose first-call snapshots need exclusive
///   evaluation, unless the status latch and timepoint cache are their only
///   eval writes (`statusOnlyMutable`);
/// - Newton-history devices (`advanceIteration`/`checkConvergence`), whose
///   `limiter_previous` the host advances on its own Instance copy;
/// - devices with `systf_calls`, whose `Instance.systf` is a host function
///   pointer (LRM §12.32) a GPU thread cannot call.
///
/// `State` without `limit` that VerA does not declare `.path_latch` (the
/// `.history` class: switches, delay lines, Laplace and pole filters) is
/// admitted host-stepped (`gpuHostStepped`).
/// A path latch (`state_class == .path_latch`, or `State` with `limit`) is
/// what `StateKernel` and `CtlKernel` run on the device: `updateState`
/// stages `wb__/wq__`, `stateCtl` commits them into the `pb__/pq__` latches
/// eval reads. A held device's `State` is that latch plus its held
/// variables, all in the resident Instance. Every `StateKernel` runs the
/// limit half per iterate and the state half once per converged solve
/// (#15), so no Newton iterate is latched. `SimState` is a
/// kernel argument, so a core that reads `analysis()` or `$abstime` (the
/// Meyer MOS models, jfet2) runs resident too.
// ponytail: the State-with-limit rule is decl correlation, not proof;
// `StateKernel` flags any non-`.ok` `updateState` so a device that breaks it
// falls back to the CPU instead of running wrong.
pub fn gpuEligible(comptime D: type) bool {
    return gpuRefusal(D) == null;
}

pub fn gpuRefusal(comptime D: type) ?GpuRefusal {
    if (@hasDecl(D, "mutable_eval") and D.mutable_eval and !statusOnlyMutable(D)) return .mutable_eval;
    if (@hasDecl(D, "advanceIteration") or @hasDecl(D, "checkConvergence")) return .newton_history;
    if (@hasDecl(D, "systf_calls") and D.systf_calls.len != 0) return .systf;
    return null;
}

/// Whether D evaluates on the device but steps its state on the host: its
/// `State` is `.history` (no `limit`, no held variables, not a path
/// latch), so `updateState`, `commit_state` and `stateCtl` walk the host
/// Instance, where step control reads breakpoints and `$bound_step`, and
/// the GPU context uploads the rows they change before the next eval
/// (`GpuContext.pushStepped`). No state or control kernel.
pub fn gpuHostStepped(comptime D: type) bool {
    if (!gpuEligible(D) or holdsOnlyHeld(D)) return false;
    if (@hasDecl(D, "limit") or !@hasDecl(D, "State")) return false;
    return !(@hasDecl(D, "state_class") and D.state_class == .path_latch);
}

/// Whether D's eval writes only its own instance: VerA's §9.7.3 status
/// latch (`status_sites`) and/or its `vera_timepoint` cache (`tp0_t`; ltra
/// has the cache and no latch), which a device thread writes on its
/// resident Instance as the host does. §9.21.1 table state (`table_ready`)
/// keeps the device on the host.
pub fn statusOnlyMutable(comptime D: type) bool {
    const latch_or_cache = @hasDecl(D, "status_sites") or @hasField(D.Instance, "tp0_t");
    return latch_or_cache and !@hasField(D.Instance, "table_ready");
}

/// Whether D's GPU eval kernel is paired with a `StateKernel`.
pub fn hasStateKernel(comptime D: type) bool {
    return gpuEligible(D) and (@hasDecl(D, "limit") or @hasDecl(D, "State")) and !gpuHostStepped(D);
}

/// Whether D also needs a `CtlKernel`: `stateCtl` mutates the resident
/// Instance/State blobs, which the host cannot reach.
pub fn hasCtlKernel(comptime D: type) bool {
    return hasStateKernel(D) and @hasDecl(D, "stateCtl");
}

/// D's type name without its namespace (`vsource.Vsource` gives `Vsource`).
/// Used as the batch `type_name` and the kernel symbol suffix.
pub fn baseName(comptime D: type) []const u8 {
    const full = @typeName(D);
    const dot = std.mem.lastIndexOfScalar(u8, full, '.') orelse return full;
    return full[dot + 1 ..];
}

/// Kernel symbol `arp_<what>_<model>`, derived from the type so the build
/// root that exports it and `gpuPayload` that names it for the launcher
/// cannot drift apart. `what`: `eval`, `evalc` (constant Jacobian), `lim`,
/// `ctl`, `qtp`, `qonly`, and `reduce`, whose body is device-independent but
/// whose name gompute needs unique across the per-device build roots. The
/// lane kernels (`l` ++ what) are not carried in the frozen `GpuPayload`:
/// the host derives them from the eval name (`gpu_lanes.laneName`), so the
/// prefix swap there must match this.
pub fn kernelSym(comptime D: type, comptime what: []const u8) [:0]const u8 {
    return "arp_" ++ what ++ "_" ++ comptime baseName(D);
}

/// D's whole Jacobian is constant (`Batch.has_const_jacobian`), so its g
/// and c stamps can live in a baseline.
pub fn constJacobian(comptime D: type) bool {
    const const_g = @hasDecl(D, "constant") and D.constant.g;
    const const_c = @hasDecl(D, "constant") and D.constant.c;
    return const_g and (!@hasDecl(D, "q") or const_c);
}
