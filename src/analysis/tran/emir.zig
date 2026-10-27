//! `--emir` (#237): every device terminal's current at each accepted
//! transient point, folded into its window average, EM_AVG halves, RMS and
//! peak. A terminal's current is what its instance stages at that port
//! (`Hooks.stage`, `.full`), an internal node collapsed onto the port
//! included: the static current, plus the dynamic current of the staged
//! charge under the accepted step's own method and coefficients
//! (`integrator.advanceCurrent`) over a charge history kept per terminal.
//! Both are what the transient sums into each KCL row, so a net's
//! terminals sum to its residual. The window reads samples as `.meas`
//! FROM=/TO= do, and the average and EM_AVG halves fold as `.meas AVG` and
//! `EM_AVG` fold them (output/scan.zig `Extremum`, output/measure.zig
//! `EmAvg`), to the bit; the RMS is the trapezoid of i². An operating or
//! DC point is one sample (`atPoint`): every statistic its |i|, the sign
//! kept in the average.
const std = @import("std");
const core = @import("core");
const root = @import("../types.zig");
const integrator = @import("integrator.zig");
const Method = @import("types.zig").Method;
const Planes = @import("device").abi.Planes;
const positiveArea = @import("measure").positiveArea;

const none = std.math.maxInt(u32);
const nan = std.math.nan(f64);

/// The accepted step a point closes: its attempt's method and coefficients.
pub const Step = struct { method: Method, c: integrator.Coeffs, ag: [7]f64, gear_hi: bool };

/// One transient's terminal currents and their window sums. Terminal
/// `tbase[k] + id * ports[k] + p` is port `p` of instance `id` of batch `k`.
pub const Emir = struct {
    gpa: std.mem.Allocator,
    /// `[from, to]` in seconds; `to` 0 runs to the end.
    window: [2]f64,
    /// Points before it only advance the charge histories (`Tran.t_start`).
    t_start: f64,
    ports: []const u32,
    /// Per batch, its first terminal, then the total.
    tbase: []u32,
    /// Per staged position, batch-major, the terminal it adds to, or `none`.
    owner: []u32,
    /// Staging planes, each as long as the longest batch's tape.
    g_stage: []f64,
    c_stage: []f64,
    rhs_stage: []f64,
    q_stage: []f64,
    /// Per terminal: the current at this point, its dynamic part, the
    /// charge ring [this point, prev, ...] and gear-3..6's history term.
    i: []f64,
    i_dyn: []f64,
    q_ring: [7][]f64,
    n_hist: usize,
    g_hist: []f64,
    /// Per terminal, over the window: the integrals of i and i², the
    /// EM_AVG halves, the largest |i| and the last sample.
    sum: []f64,
    sum2: []f64,
    pos: []f64,
    neg: []f64,
    peak: []f64,
    last: []f64,
    store: []f64,
    /// The window's first and last sample times, and the average's span.
    from: f64 = nan,
    to: f64 = nan,
    span: f64 = 0,
    over: bool = false,

    /// Sizes the tables for `ckt`, whose batch `k` has `ports[k]` ports, and
    /// a run whose top method is `method`. `ports` must outlive it.
    pub fn init(gpa: std.mem.Allocator, ckt: *const root.Circuit, ports: []const u32, method: Method, window: [2]f64, t_start: f64) !Emir {
        std.debug.assert(ports.len == ckt.batches.len);
        const tbase = try gpa.alloc(u32, ckt.batches.len + 1);
        errdefer gpa.free(tbase);
        tbase[0] = 0;
        var n_pos: usize = 0;
        var n_slot: usize = 0;
        var n_row: usize = 0;
        for (ckt.batches, ports, 0..) |b, p, k| {
            std.debug.assert(p <= b.n_u);
            const tapes = b.hooks.scatter_tapes(b.ctx);
            tbase[k + 1] = tbase[k] + b.count * p;
            n_pos += tapes.rows.len;
            n_slot = @max(n_slot, tapes.slots.len);
            n_row = @max(n_row, tapes.rows.len);
        }
        const owner = try gpa.alloc(u32, n_pos);
        errdefer gpa.free(owner);
        var at: usize = 0;
        for (ckt.batches, ports, tbase[0..ckt.batches.len]) |b, p, base| {
            const rows = b.hooks.scatter_tapes(b.ctx).rows;
            for (0..b.count) |id| {
                const r = rows[id * b.n_u ..][0..b.n_u];
                const first: u32 = base + @as(u32, @intCast(id)) * p;
                // ponytail: an internal node collapsed onto a port joins the
                // first port on its row, the wrong one only between two
                // ports tied to one net; the device's `collapse` map would
                // name the right one.
                for (r, 0..) |row, u| {
                    owner[at] = if (u < p) first + @as(u32, @intCast(u)) else for (r[0..p], 0..) |pr, j| {
                        if (pr == row) break first + @as(u32, @intCast(j));
                    } else none;
                    at += 1;
                }
            }
        }
        const nt: usize = tbase[ckt.batches.len];
        const n_hist: usize = @max(3, integrator.order(method) + 1);
        const store = try gpa.alloc(f64, (9 + n_hist) * nt + 2 * n_slot + 2 * n_row);
        @memset(store, 0);
        var rest = store;
        var e: Emir = .{
            .gpa = gpa,
            .window = window,
            .t_start = t_start,
            .ports = ports,
            .tbase = tbase,
            .owner = owner,
            .g_stage = cut(&rest, n_slot),
            .c_stage = cut(&rest, n_slot),
            .rhs_stage = cut(&rest, n_row),
            .q_stage = cut(&rest, n_row),
            .i = cut(&rest, nt),
            .i_dyn = cut(&rest, nt),
            .q_ring = @splat(store[0..0]),
            .n_hist = n_hist,
            .g_hist = cut(&rest, nt),
            .sum = cut(&rest, nt),
            .sum2 = cut(&rest, nt),
            .pos = cut(&rest, nt),
            .neg = cut(&rest, nt),
            .peak = cut(&rest, nt),
            .last = cut(&rest, nt),
            .store = store,
        };
        for (e.q_ring[0..n_hist]) |*h| h.* = cut(&rest, nt);
        std.debug.assert(rest.len == 0);
        return e;
    }

    /// Frees what `init` allocated.
    pub fn deinit(e: *Emir) void {
        e.gpa.free(e.store);
        e.gpa.free(e.owner);
        e.gpa.free(e.tbase);
        e.* = undefined;
    }

    /// The run's first point, `x` at t = 0: seeds every charge history, as
    /// the transient seeds its own, and with `sample` reads its currents,
    /// which hold no dynamic part.
    pub fn start(e: *Emir, ckt: *root.Circuit, x: []const f64, sample: bool) void {
        e.stamp(ckt, x, 0);
        for (e.q_ring[1..e.n_hist]) |h| @memcpy(h, e.q_ring[0]);
        if (sample) e.fold(0);
    }

    /// Accepted point `x` at `t`, the end of `step`: its terminal currents,
    /// folded in once `t` reaches `t_start`.
    pub fn point(e: *Emir, ckt: *root.Circuit, x: []const f64, t: f64, step: Step) void {
        e.stamp(ckt, x, t);
        if (ckt.has_charge) {
            const h = e.q_ring[0..e.n_hist];
            if (step.gear_hi) integrator.gearHistory(e.g_hist, h[0 .. integrator.order(step.method) + 1], step.ag);
            integrator.advanceCurrent(step.method, e.i_dyn, h[0], h[1], if (step.gear_hi) e.g_hist else h[2], step.c);
            std.mem.rotate([]f64, h, e.n_hist - 1);
            for (e.i, e.i_dyn) |*i, d| i.* += d;
        }
        if (t >= e.t_start) e.fold(t);
    }

    /// Each terminal's static current at `x` into `i`, its charge into
    /// ring slot 0.
    fn stamp(e: *Emir, ckt: *root.Circuit, x: []const f64, t: f64) void {
        @memset(e.i, 0);
        @memset(e.q_ring[0], 0);
        var at: usize = 0;
        for (ckt.batches) |b| {
            const tapes = b.hooks.scatter_tapes(b.ctx);
            const n = tapes.rows.len;
            const pl: Planes = .{
                .g_vals = e.g_stage[0..tapes.slots.len],
                .c_vals = if (ckt.has_charge) e.c_stage[0..tapes.slots.len] else &.{},
                .rhs = e.rhs_stage[0..n],
                .q_vec = if (ckt.has_charge) e.q_stage[0..n] else &.{},
            };
            @memset(pl.rhs, 0);
            @memset(pl.q_vec, 0);
            b.hooks.stage(b.ctx, &pl, 0, b.count, x, t, .full);
            for (e.owner[at..][0..n], pl.rhs, 0..) |o, f, k| if (o != none) {
                e.i[o] += f;
                if (ckt.has_charge) e.q_ring[0][o] += pl.q_vec[k];
            };
            at += n;
        }
    }

    /// Folds this point's currents, at `t`, into the window's sums.
    fn fold(e: *Emir, t: f64) void {
        if (e.over or t < e.window[0]) return;
        if (e.window[1] != 0 and t > e.window[1]) {
            e.over = true;
            return;
        }
        if (std.math.isNan(e.from)) {
            e.from = t;
            @memcpy(e.last, e.i);
            for (e.peak, e.i) |*pk, i| pk.* = @abs(i);
        } else {
            const dx = t - e.to;
            e.span += dx;
            for (e.i, e.last, e.sum, e.sum2, e.pos, e.neg, e.peak) |i, *l, *s, *s2, *up, *down, *pk| {
                s.* += 0.5 * (i + l.*) * dx;
                s2.* += 0.5 * (i * i + l.* * l.*) * dx;
                up.* += positiveArea(l.*, i, dx);
                down.* += positiveArea(-l.*, -i, dx);
                pk.* = @max(pk.*, @abs(i));
                l.* = i;
            }
        }
        e.to = t;
    }

    /// The window's statistics, every terminal's in terminal order, in `a`.
    /// A window of one sample publishes that sample.
    pub fn publish(e: *const Emir, a: std.mem.Allocator, ckt: *const root.Circuit) !core.Emir {
        const one = e.from == e.to;
        const out = try a.alloc(core.EmirTerminal, e.sum.len);
        for (ckt.batches, e.ports, e.tbase[0..ckt.batches.len], 0..) |b, p, base, k| {
            const rows = b.hooks.scatter_tapes(b.ctx).rows;
            for (0..b.count) |id| for (0..p) |u| {
                const j = base + id * p + u;
                out[j] = .{
                    .batch = @intCast(k),
                    .instance = @intCast(id),
                    .port = @intCast(u),
                    .row = rows[id * b.n_u + u],
                    .avg = if (one) e.last[j] else e.sum[j] / e.span,
                    .rms = if (one) e.peak[j] else @sqrt(e.sum2[j] / e.span),
                    .peak = e.peak[j],
                    .pos = e.pos[j],
                    .neg = e.neg[j],
                };
            };
        }
        return .{ .from = e.from, .to = e.to, .terminals = out };
    }
};

/// Every terminal's current at the static point `x` of `ctx.circuit` (an
/// `.op`, a `.dc` point), at t = 0; no terminals when `ctx` has no ports.
pub fn atPoint(ctx: *const root.RunCtx, x: []const f64) !core.Emir {
    if (ctx.emir_ports.len != ctx.circuit.batches.len) return .{};
    var e = try Emir.init(ctx.scratch_allocator, ctx.circuit, ctx.emir_ports, .backward_euler, .{ 0, 0 }, 0);
    defer e.deinit();
    e.start(ctx.circuit, x, true);
    return e.publish(ctx.allocator, ctx.circuit);
}

/// The first `len` of `rest`, which then starts past them.
fn cut(rest: *[]f64, len: usize) []f64 {
    const s = rest.*[0..len];
    rest.* = rest.*[len..];
    return s;
}
