//! Source breakpoints the way ngspice rounds them: a PULSE or PWL Model and
//! the accepted and query times in, the next corner out, cached per
//! instance in a `SrcBrk` so host and device step control agree bit for
//! bit. Cites ngspice's vsrcacct.c:48-121 and 167-216 and isrcacct.c, and
//! txlload.c for why one ulp matters.
//!
//! Contents:
//! - `pub const SrcBrk = extern struct`
//! - `pub inline fn srcBreak(comptime D: type, c: *SrcBrk, m: *const D.Model, t_acc: f64, t: f64) f64`
//! - `pub fn spicePulseBreak(m: anytype, t_acc: f64, tq: f64) ?f64`
//! - `pub fn spicePwlBreak(m: anytype, t_acc: f64, tq: f64) ?f64`

const std = @import("std");
const traits = @import("traits.zig");

/// `extern`: `GpuPayload.brk` hands these to device step control as bytes.
pub const SrcBrk = extern struct { req: f64 = std.math.inf(f64), at: f64 = std.math.inf(f64) };

/// Model `m`'s next breakpoint after `t`, through its cache `c`: kept until a
/// query reaches it or goes back before the request that set it. Shared by
/// `nextBreakpointFn` and `LaneBreakKernel`, so host and device round alike.
/// isrcacct.c requests the raw PWL table times, so only the V source rounds
/// its PWL corners.
pub inline fn srcBreak(comptime D: type, c: *SrcBrk, m: *const D.Model, t_acc: f64, t: f64) f64 {
    if (t < c.req or t >= c.at)
        c.* = .{ .req = t, .at = spicePulseBreak(m, t_acc, t) orelse
            (if (comptime std.mem.eql(u8, traits.baseName(D), "vsource")) spicePwlBreak(m, t_acc, t) else null) orelse
            D.nextBreakpoint(m, t) orelse std.math.inf(f64) };
    return c.at;
}

/// ngspice's PULSE breakpoint (vsrcacct.c:48-121, isrcacct.c the same): the
/// corner after `tq`, requested at the accepted time `t_acc <= tq` and
/// rounded as ngspice rounds it, `t_acc + (corner - phase(t_acc))`. VerA's
/// `td + tr + pw + tf` can land one ulp away, which moves a breakpoint across
/// an integer picosecond and changes every TXL/CPL delayed read after it
/// (txlload.c truncates time to ps). Null when the result is not after `tq`,
/// i.e. `t_acc` is not a time ngspice would have asked at, and for any
/// waveform but PULSE. Nonzero PHASE is left to the timers too: ngspice gives
/// it a different meaning outside xs mode. The widths are vsource.va's
/// P_TR/P_TF/P_PW/P_PER.
pub fn spicePulseBreak(m: anytype, t_acc: f64, tq: f64) ?f64 {
    if (m.waveform != 1 or m.pulse_phase != 0) return null;
    const tr = @max(m.pulse_tr, 1.0e-12);
    const tf = @max(m.pulse_tf, 1.0e-12);
    const pw = @max(m.pulse_pw, 0.0) + @max(-m.pulse_pw, 0.0) * 1.0e30;
    const per = @max(m.pulse_per, tr + pw + tf + 1.0e-12);
    var time = t_acc - m.pulse_td;
    if (time >= per) time -= per * @floor(time / per);
    // ngspice's `atime = time + CKTminBreak`; the host asks at t + minBreak.
    const atime = time + (tq - t_acc);
    const corner: f64 = if (atime < 0.0) 0.0 else if (atime < tr) tr else if (atime < tr + pw) tr + pw else if (atime < tr + pw + tf) tr + pw + tf else per;
    const bp = t_acc + (corner - time);
    return if (bp > tq) bp else null;
}

/// ngspice's PWL breakpoint for a V source (vsrcacct.c:167-216): the first
/// table time `t_k` after the request, rounded from the accepted time
/// `t_acc <= tq` as `t_acc + (t_k - local)`, where `local` is `t_acc - td`
/// folded into `[r, t_last]` by `r=`. VerA's timers give `t_k + td`, one ulp
/// away at times, and add the PULSE timers' corners at TD and TD + TR, which
/// ngspice does not have. Past the last point of a non-repeating table there
/// is no breakpoint (inf). Null when the rounded time is not after `tq`
/// (an anchor ngspice would not ask at), and for any other waveform. The
/// repeat gate is vsource.va's own, so breakpoints follow the waveform it
/// computes.
pub fn spicePwlBreak(m: anytype, t_acc: f64, tq: f64) ?f64 {
    const T = @TypeOf(m.*);
    if (m.waveform != 4 or m.pwl_len <= 0) return null;
    // Slot `k` of the flattened `pwl_times[]`, named as in
    // frontend/builder.zig pwlSlot.
    @setEvalBranchQuota(200_000);
    const cap = comptime blk: {
        var c: usize = 0;
        while (@hasField(T, std.fmt.comptimePrint("pwl_timesZ5b{d}Z5d", .{c}))) c += 1;
        break :blk c;
    };
    var ts: [cap]f64 = undefined;
    inline for (0..cap) |k| ts[k] = @field(m.*, std.fmt.comptimePrint("pwl_timesZ5b{d}Z5d", .{k}));
    const n: usize = @min(@as(usize, @intCast(m.pwl_len)), cap);
    const t_last = ts[n - 1];
    const r = m.pwl_repeat;
    const repeats = r >= 0.0 and n > 1 and t_last > r;
    const period = t_last - r;
    var time = t_acc - m.pwl_td;
    if (time > t_last) {
        if (!repeats) return std.math.inf(f64);
        time -= r;
        time -= period * @floor(time / period);
        time += r;
    }
    // ngspice's `atime = time + CKTminBreak`; the host asks at t + minBreak.
    const atime = time + (tq - t_acc);
    const corner = for (ts[0..n]) |c| {
        if (c > atime) break c;
    } else if (!repeats) return std.math.inf(f64) else for (ts[0..n]) |c| {
        // At the end of a repeating table ngspice sets nothing and asks again
        // one step later, after the fold; this names the same corner now.
        if (c + period > atime) break c + period;
    } else return null;
    const bp = t_acc + (corner - time);
    return if (bp > tq) bp else null;
}
