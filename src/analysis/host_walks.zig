//! Host walks over a batch slice: each runs one device hook over every
//! batch in order, serially. `Circuit` walks all of its batches; the GPU
//! context walks the ones it keeps on the host, or all of them after a
//! fault. ParEval (`par_eval.zig`) is the threaded twin of the stamp and
//! of the limit and update walks.
//! In: batches and an iterate. Out: stamped planes, limited x, reject times.
//! Cites no outside source.
//!
//! Contents:
//! - `pub const Mode = abi.StampMode`
//! - `pub fn stampRange(b: Batch, pl: *const Planes, first: u32, last: u32, x: []const f64, t: f64, comptime mode: Mode) void`
//! - `pub fn limitBatches(batches: []const Batch, x: []f64, x_old: []const f64) bool`
//! - `pub fn seedBatches(batches: []const Batch, x: []f64) void`
//! - `pub fn clearLimitBatches(batches: []const Batch) void`
//! - `pub fn updateBatches(batches: []const Batch, x: []const f64) ?f64`
//! - `pub fn stateCtlBatches(batches: []const Batch, sop: StateCtlOp) bool`
//! - `pub fn minReject(batches: []const Batch, comptime hook: []const u8, x: []const f64) ?f64`

const std = @import("std");
const assert = std.debug.assert;
const abi = @import("device").abi;
const Batch = abi.Batch;
const Planes = abi.Planes;
const StateCtlOp = abi.StateCtlOp;

/// Which planes a stamp writes; see `abi.StampMode`. `.charge` uses the
/// same staging and reduce as `.full`, so its q plane is bit-for-bit what
/// `.full` leaves.
pub const Mode = abi.StampMode;

/// Stamps instances [first, last) of one batch in `mode`. The serial path's
/// per-mode dispatch.
pub fn stampRange(b: Batch, pl: *const Planes, first: u32, last: u32, x: []const f64, t: f64, comptime mode: Mode) void {
    assert(first <= last);
    assert(last <= b.count);
    switch (mode) {
        .full => b.eval(b.ctx, pl, first, last, x, t),
        .newton => b.eval_newton(b.ctx, pl, first, last, x, t),
        .charge => if (b.hooks.eval_q) |f| f(b.ctx, pl, first, last, x, t),
    }
}

/// `applyLimits` over `batches` on the host.
pub fn limitBatches(batches: []const Batch, x: []f64, x_old: []const f64) bool {
    var any_limited = false;
    for (batches) |b| if (b.hooks.apply_limits) |f| {
        if (f(b.ctx, x, x_old)) any_limited = true;
    };
    return any_limited;
}

/// `seedJunctions` over `batches` on the host.
pub fn seedBatches(batches: []const Batch, x: []f64) void {
    for (batches) |b| if (b.hooks.seed) |f| f(b.ctx, x);
}

/// `clearLimits` over `batches` on the host.
pub fn clearLimitBatches(batches: []const Batch) void {
    for (batches) |b| if (b.hooks.clear_limits) |f| f(b.ctx);
}

/// `updateStates`' device walk over `batches`: the earliest reject time,
/// or null. Stages state for the next `stateCtl`.
pub fn updateBatches(batches: []const Batch, x: []const f64) ?f64 {
    return minReject(batches, "update_state", x);
}

/// `stateCtl` over `batches` on the host.
pub fn stateCtlBatches(batches: []const Batch, sop: StateCtlOp) bool {
    var dirty = false;
    for (batches) |b| if (b.hooks.state_ctl) |f| {
        if (f(b.ctx, sop)) dirty = true;
    };
    return dirty;
}

/// Earliest step-reject time any batch's `hook` returns, or null.
pub fn minReject(batches: []const Batch, comptime hook: []const u8, x: []const f64) ?f64 {
    var min_reject: ?f64 = null;
    for (batches) |b| {
        if (@field(b.hooks, hook)) |f| if (f(b.ctx, x)) |tr| {
            min_reject = if (min_reject) |cur| @min(cur, tr) else tr;
        };
    }
    return min_reject;
}
