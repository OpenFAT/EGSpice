//! The query kind to analysis leaf table, and the per-kind facts the
//! executor reads off a query without running it: the simulation state its
//! devices see, the temperature and variant it overrides, and a guess at
//! its device evaluations.
//! In: a query. Out: its leaf module, its run, its kind's facts.
//! Cites VAMS LRM Table 4-22; ngspice osdiload.c:165 (the small-signal
//! `analysis()` kind).
//!
//! Contents:
//! - `pub const SimKind`
//! - `pub fn run(ctx: *const RunCtx, job: requests.Query) !Result`
//! - `pub fn module(comptime kind: requests.Kind) type`
//! - `pub fn simKind(job: requests.Query) SimKind`
//! - `pub fn queryTemp(job: requests.Query) ?f64`
//! - `pub fn queryVariant(job: requests.Query) ?u32`
//! - `pub fn expectedEvals(job: requests.Query) f64`

const std = @import("std");
const requests = @import("core").query;
const types = @import("types.zig");
const op = @import("dc/op.zig");

/// The analysis kind a device's `analysis()` reads.
pub const SimKind = @FieldType(@import("device").abi.SimState, "kind");

/// Runs `job`'s analysis leaf on `ctx`.
pub fn run(ctx: *const types.RunCtx, job: requests.Query) !types.Result {
    switch (job) {
        inline else => |opts, tag| return module(tag).run(ctx, opts),
    }
}

/// Returns the analysis leaf that runs queries of `kind`.
pub fn module(comptime kind: requests.Kind) type {
    return switch (kind) {
        .op => op,
        .dc => @import("dc/dc.zig"),
        .tf => @import("dc/tf.zig"),
        .dcmatch => @import("dc/dcmatch.zig"),
        .dcsens => @import("dc/dcmatch.zig").Sens,
        .acmatch => @import("ac/acmatch.zig"),
        .ac => @import("ac/ac.zig"),
        .noise => @import("ac/noise.zig"),
        .sp => @import("ac/sp.zig"),
        .stb => @import("ac/stb.zig"),
        .lstb => @import("ac/lstb.zig"),
        .acxf => @import("ac/xf.zig"),
        .dcxf => @import("dc/xf.zig"),
        .dcinc => @import("dc/xf.zig").Inc,
        .tran => @import("tran/tran.zig"),
        .tran_noise => @import("tran/tran_noise.zig"),
        .envelope => @import("tran/envelope.zig"),
        .matex => @import("tran/matex.zig"),
        .pss => @import("pss/pss.zig"),
        .pac => @import("pss/pac.zig"),
        .pnoise => @import("pss/pnoise.zig"),
        .hb => @import("pss/hb.zig"),
        .pxf => @import("pss/pxf.zig"),
        .hbac => @import("pss/hb_lptv.zig").Ac,
        .hbxf => @import("pss/hb_lptv.zig").Xf,
        .hbnoise => @import("pss/hb_lptv.zig").Noise,
        .hblin => @import("pss/hb_lptv.zig").Lin,
        .phasenoise => @import("pss/phasenoise.zig"),
        .qpss => @import("pss/qpss.zig"),
        .four => @import("post/four.zig"),
        .disto => @import("post/disto.zig"),
        .fft => @import("post/fft.zig"),
        .sens => @import("sweep/sens.zig"),
        .mc => @import("sweep/mc.zig"),
        .temp => @import("sweep/temp_sweep.zig"),
        .pz => @import("eigen/pz.zig"),
    };
}

/// Returns the analysis state a query's devices see while it runs (`runQuery`).
/// Small-signal linearizations run as analysis("ac") or ("noise"), LRM
/// Table 4-22, as ngspice's MODEINITSMSIG load sets ANALYSIS_AC for ac,
/// noise, pz and disto (osdiload.c:165).
pub fn simKind(job: requests.Query) SimKind {
    const transient = if (job == .op) job.op.tran_op else @as(requests.Kind, job).transient();
    return switch (job) {
        .ac, .sp, .stb, .pz, .disto, .lstb, .acxf => .ac,
        .noise => .noise,
        else => if (transient) .ic else .dc,
    };
}

/// Returns the temperature `job` runs at when it overrides the deck's.
pub fn queryTemp(job: requests.Query) ?f64 {
    return switch (job) {
        inline else => |o| o.tol.temp_c,
    };
}

/// The variant row `job` runs, when it names one.
pub fn queryVariant(job: requests.Query) ?u32 {
    return switch (job) {
        inline else => |o| o.tol.variant,
    };
}

/// Returns a rough count of the device evals `job` runs, for the GPU cost model:
/// four per transient step (measured 4.3 on mos1_2000 and 5.2 on
/// vacask_ring) at the step the deck asks for, three per dc point, and a
/// Newton solve's worth for an operating point. Small-signal analyses
/// linearize once.
/// ponytail: a static guess; count evals as the query runs and move to the
/// GPU mid-run if a guess ever misprices a deck.
pub fn expectedEvals(job: requests.Query) f64 {
    return switch (job) {
        .op => 30,
        .dc => |d| blk: {
            const inner = @abs(d.stop - d.start) / @max(@abs(d.step), 1e-300) + 1;
            const outer = if (d.target2 != null) @abs(d.stop2 - d.start2) / @max(@abs(d.step2), 1e-300) + 1 else 1;
            break :blk 3 * inner * outer;
        },
        .tran => |t| 4 * t.t_stop / @min(t.dt_init, t.requestedCeiling(.api)),
        .ac, .noise, .sp, .stb, .tf, .pz, .disto, .four, .dcmatch, .sens, .lstb, .acxf, .dcxf, .dcinc, .acmatch, .dcsens => 1,
        else => 200,
    };
}
