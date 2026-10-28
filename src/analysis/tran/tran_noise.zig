//! Transient noise: a circuit at its operating point and its noise sources
//! in, the probes at every accepted point out (plus `onoise` under SDE). A
//! backward-Euler transient draws one Gaussian current sample per noise
//! source per step, sigma = sqrt(S * BW) with BW = 1/(2dt), so the sampled
//! sequence carries the source's white PSD. Flicker noise (K / f^ef) is a
//! sum of Ornstein-Uhlenbeck (Lorentzian) processes with corners log-spaced
//! over [f_min, 1/(2 dt_max)], each advanced exactly over the step. The PSDs
//! come from the devices (`Circuit.collectNoiseSources`). METHOD=SDE draws
//! nothing: the noiseless march carries the covariance of the noise that
//! sampled runs on the same steps would add (`Covariance`).
//! Cites: HSPICE `.trannoise` (SEED, TIME=, SAMPLES) and Verilog-AMS LRM
//! §9.10.
//!
//! Contents:
//! - `pub const Options = core.query.TranNoise`
//! - `pub const test_access`, over the private types below, whose `pub`
//!   members serve the tests and the converger:
//!   `Xorshift64.init(seed: u64) Xorshift64`, `Xorshift64.randn(self) f64`, `Xorshift64.next(self) u64`,
//!   `Flicker.init(gpa, sources: []const NoiseSource, options: Options) Allocator.Error!Flicker`,
//!   `Flicker.deinit(f: *Flicker, gpa) void`,
//!   `Covariance.stampVec(m: []f64, n: usize, nodes: []const u32, weights: []const f64, v: f64) void`,
//!   `Covariance.transpose(m: []f64, n: usize) void`,
//!   `NoiseHook.assemble(self, ckt, x, t) void`, `NoiseHook.vals(self, ckt) []f64`,
//!   `NoiseHook.diagAt(self, ckt, slot: u32) f64`
//! - `pub fn run(ctx: *const root.RunCtx, opts: Options) !root.Result`

const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;
const numerics = @import("core").numerics;
const converger = @import("solver").converger;
const direct = @import("solver").direct;
const root = @import("../types.zig");
const integrator = @import("integrator.zig");
const Waveform = @import("types.zig").Waveform;

const NoiseSource = root.NoiseSource;

/// `.trannoise` query options, defined in core/query.zig.
pub const Options = @import("core").query.TranNoise;

/// Private implementation access for the analysis test suite.
pub const test_access = if (@import("builtin").is_test) .{
    .Xorshift64 = Xorshift64,
    .Flicker = Flicker,
    .Covariance = Covariance,
} else {};

/// Rows `run` reserves up front: without Newton failures dt only grows from
/// dt_init, so t_stop/dt_init + 1 rows is the most a run records; 2x is
/// headroom and the waveform doubles past it, between these bounds.
// ponytail: capacity heuristic.
const rows_min: f64 = 1024;
const rows_max: f64 = 1 << 22;

/// Xorshift64 PRNG with a Box-Muller normal draw: deterministic per seed.
// ponytail: stays in-tree; std.Random's stream isn't stable across Zig
// releases and seeded output must be (#196).
const Xorshift64 = struct {
    /// Never 0: xorshift is stuck at zero.
    state: u64,

    /// Returns the generator for `seed`. The seed goes through SplitMix64
    /// first, so small seeds (HSPICE's SEED=2, 3, ...) start well mixed; a
    /// zero state maps to 1.
    pub fn init(seed: u64) Xorshift64 {
        var mix = std.Random.SplitMix64.init(seed);
        const state = mix.next();
        return .{ .state = if (state == 0) 1 else state };
    }

    /// Returns a standard normal draw via Box-Muller.
    pub fn randn(self: *Xorshift64) f64 {
        const r1 = @max(self.uniform(), 1e-300); // keeps log finite
        const r2 = self.uniform();
        return @sqrt(-2.0 * @log(r1)) * @cos(2.0 * std.math.pi * r2);
    }

    /// Returns the next state; never 0, since a nonzero state stays nonzero.
    pub fn next(self: *Xorshift64) u64 {
        assert(self.state != 0);
        var s = self.state;
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        self.state = s;
        assert(s != 0);
        return s;
    }

    // ---- private ----

    /// Uniform in [0, 1) from the top 53 bits.
    fn uniform(self: *Xorshift64) f64 {
        return @as(f64, @floatFromInt(self.next() >> 11)) / @as(f64, @floatFromInt(@as(u64, 1) << 53));
    }
};

/// Flicker generators, SoA over every pole of every source with flicker
/// noise. A pole is an Ornstein-Uhlenbeck process of corner f_k and
/// variance v_k, one-sided PSD 4 v_k tau_k / (1 + (2 pi f tau_k)^2) with
/// tau_k = 1 / (2 pi f_k). With `per_decade` corners log-spaced by ratio r,
/// v_k = K ln(r) sin(pi (2 - ef) / 2) f_k^(1 - ef) makes the sum K / f^ef
/// between the outer corners (the integral of the Lorentzian over log f),
/// so for ef = 1 the variance is K ln(f_max / f_min). The four columns have
/// one length.
const Flicker = struct {
    /// The owning noise source per pole.
    source: []u32,
    /// 1/tau per pole.
    rate: []f64,
    /// The stationary variance per pole.
    variance: []f64,
    /// The process value. Every attempt advances it; a rejected attempt's
    /// advance stands, which keeps each process stationary.
    y: []f64,
    started: bool = false,

    // ponytail: fixed density; the sum ripples about 0.1 dB around K/f^ef.
    const per_decade = 3.0;

    /// Returns `per_decade` poles per decade of [f_min, 1/(2 dt_max)] for
    /// every source with 0 < ef < 2; none when the band is empty. Caller
    /// owns the columns; free with `deinit` and the same `gpa`.
    pub fn init(gpa: Allocator, sources: []const NoiseSource, options: Options) Allocator.Error!Flicker {
        const f_max = 0.5 / options.dt_max;
        const f_min = options.f_min orelse 1.0 / options.t_stop;
        const decades = if (f_max > f_min) std.math.log10(f_max / f_min) else 0;
        const k: usize = @intFromFloat(@ceil(per_decade * decades));
        var count: usize = 0;
        for (sources) |s| count += if (hasFlicker(s)) k else 0;
        const source = try gpa.alloc(u32, count);
        errdefer gpa.free(source);
        const rate = try gpa.alloc(f64, count);
        errdefer gpa.free(rate);
        const variance = try gpa.alloc(f64, count);
        errdefer gpa.free(variance);
        var f: Flicker = .{ .source = source, .rate = rate, .variance = variance, .y = try gpa.alloc(f64, count) };
        if (count == 0) return f;
        const ln_r = @log(f_max / f_min) / @as(f64, @floatFromInt(k));
        var p: usize = 0;
        for (sources, 0..) |s, i| {
            if (!hasFlicker(s)) continue;
            const weight = s.flicker * options.scale * ln_r * @sin(std.math.pi * (2 - s.ef) / 2);
            for (0..k) |j| {
                const corner = f_min * @exp(ln_r * (@as(f64, @floatFromInt(j)) + 0.5));
                f.source[p] = @intCast(i);
                f.rate[p] = 2 * std.math.pi * corner;
                f.variance[p] = weight * std.math.pow(f64, corner, 1 - s.ef);
                p += 1;
            }
        }
        assert(p == count);
        return f;
    }

    /// Frees the columns `init` allocated on `gpa`.
    pub fn deinit(f: *Flicker, gpa: Allocator) void {
        gpa.free(f.source);
        gpa.free(f.rate);
        gpa.free(f.variance);
        gpa.free(f.y);
    }

    // ---- private ----

    // ponytail: ef outside (0, 2) has no finite Lorentzian sum; such a
    // source keeps its white half only.
    fn hasFlicker(s: NoiseSource) bool {
        return s.flicker > 0 and s.ef > 0 and s.ef < 2;
    }

    /// Advances every pole by `dt` and adds each to its source's current.
    /// The first call draws the stationary state.
    fn step(f: *Flicker, dt: f64, rng: *Xorshift64, currents: []f64) void {
        assert(dt > 0);
        for (f.source, f.rate, f.variance, f.y) |s, rate, v, *y| {
            const decay = @exp(-rate * dt);
            y.* = if (f.started) y.* * decay + @sqrt(v * (1 - decay * decay)) * rng.randn() else @sqrt(v) * rng.randn();
            currents[s] += y.*;
        }
        f.started = true;
    }
};

/// Where each noise group injects: group g's current enters node `node[e]`
/// with weight `weight[e]` (the row's signed `coeff`, negated on the n
/// side) for e in `ptr[g]..ptr[g + 1]`, ground entries dropped. A group is
/// one generator, every row sharing a `NoiseSource.group`, and `lead[g]`
/// is its first row, whose shape prices it. `ptr` and `node` share one
/// allocation. Free with `deinit` and the `init` allocator.
const Injection = struct {
    ptr: []u32,
    node: []u32,
    weight: []f64,
    lead: []NoiseSource,

    fn init(gpa: Allocator, srcs: []const NoiseSource) Allocator.Error!Injection {
        var groups: usize = 0;
        var i: usize = 0;
        while (i < srcs.len) : (groups += 1) i = NoiseSource.groupEnd(srcs, i);
        const ptr = try gpa.alloc(u32, groups + 1 + 2 * srcs.len);
        errdefer gpa.free(ptr);
        const weight = try gpa.alloc(f64, 2 * srcs.len);
        errdefer gpa.free(weight);
        const lead = try gpa.alloc(NoiseSource, groups);
        const inj: Injection = .{ .ptr = ptr[0 .. groups + 1], .node = ptr[groups + 1 ..], .weight = weight, .lead = lead };
        var e: u32 = 0;
        var g: usize = 0;
        i = 0;
        while (i < srcs.len) : (g += 1) {
            const end = NoiseSource.groupEnd(srcs, i);
            defer i = end;
            inj.ptr[g] = e;
            lead[g] = srcs[i];
            for (srcs[i..end]) |src| for ([2]u32{ src.node_p, src.node_n }, [2]f64{ src.coeff, -src.coeff }) |node, w| {
                if (node == root.GROUND) continue;
                inj.node[e] = node;
                inj.weight[e] = w;
                e += 1;
            };
        }
        inj.ptr[groups] = e;
        assert(g == groups);
        assert(e <= inj.node.len);
        return inj;
    }

    fn deinit(inj: Injection, gpa: Allocator) void {
        gpa.free(inj.ptr.ptr[0 .. inj.ptr.len + inj.node.len]);
        gpa.free(inj.weight);
        gpa.free(inj.lead);
    }

    /// Group g's entries: its nodes and their weights.
    fn of(inj: Injection, g: usize) struct { []const u32, []const f64 } {
        const lo = inj.ptr[g];
        const hi = inj.ptr[g + 1];
        assert(lo <= hi);
        return .{ inj.node[lo..hi], inj.weight[lo..hi] };
    }
};

/// METHOD=SDE: the covariance of the noise part of x through the same
/// backward-Euler steps the sampled run takes, exact for the linearized
/// circuit. With A = G + C/h and P = C/h at the step's end point, the white
/// draws w (covariance N, sigma^2 = S/(2h) per source) and the flicker
/// poles y (stationary variances V, decays D = e^(-h/tau), injected by J),
///   x_k = A^-1 (P x_{k-1} + J y_k + w_k)
///   K_k = A^-1 (P K P^T + P Kxy D J^T + J D Kxy^T P^T + J V J^T + N) A^-T
///   Kxy_k = A^-1 (P Kxy D + J V),
/// with K = E[x x^T] and Kxy = E[x y^T], both zero at t = 0. A is the
/// circuit's sparse LU, so a step costs 2n + poles sparse solves and the
/// O(n nnz) products, not a dense O(n^3) factor.
// ponytail: K itself stays dense, n^2 memory and O(n^2) a step. A
// low-rank factor of K is the upgrade past a few thousand unknowns.
const Covariance = struct {
    /// Row-major n x n: K, then the products that become the next K.
    k: []f64,
    m: []f64,
    /// Row-major n x n scratch for P K.
    a: []f64,
    /// Pole-major `poles x n`: row p is E[x y_p]. `k`, `m`, `a` and `kxy`
    /// share one allocation.
    kxy: []f64,
    /// G + C/h on the circuit pattern and its LU, made by the first `step`.
    vals: []f64 = &.{},
    slv: ?direct.Solver = null,
    /// Allocates `vals` and the solver on the first `step`.
    gpa: Allocator,
    /// Where each group injects; the white PSD is its lead's. Borrowed.
    inj: Injection,
    n: usize,

    /// Adds v j j^T to the dense matrix m, j the sparse vector with
    /// `weights` at `nodes` (ground already dropped).
    pub fn stampVec(m: []f64, n: usize, nodes: []const u32, weights: []const f64, v: f64) void {
        assert(nodes.len == weights.len);
        assert(m.len >= n * n);
        for (nodes, weights) |a, wa| for (nodes, weights) |b, wb| {
            m[a * n + b] += v * wa * wb;
        };
    }

    /// Transposes the row-major n x n matrix m in place.
    pub fn transpose(m: []f64, n: usize) void {
        assert(m.len >= n * n);
        for (0..n) |i| for (i + 1..n) |j| std.mem.swap(f64, &m[i * n + j], &m[j * n + i]);
    }

    // ---- private ----

    fn init(gpa: Allocator, n: usize, poles: usize, inj: Injection) Allocator.Error!Covariance {
        const buf = try gpa.alloc(f64, 3 * n * n + poles * n);
        @memset(buf, 0);
        return .{
            .k = buf[0 .. n * n],
            .m = buf[n * n ..][0 .. n * n],
            .a = buf[2 * n * n ..][0 .. n * n],
            .kxy = buf[3 * n * n ..],
            .gpa = gpa,
            .inj = inj,
            .n = n,
        };
    }

    fn deinit(c: *Covariance, gpa: Allocator) void {
        gpa.free(c.k.ptr[0 .. 3 * c.n * c.n + c.kxy.len]);
        gpa.free(c.vals);
        if (c.slv) |*slv| slv.deinit();
    }

    /// out[row] += alpha * C x for every row; C from the CSC plane.
    fn addCx(ckt: *const root.Circuit, alpha: f64, x: []const f64, out: []f64) void {
        assert(x.len >= ckt.n);
        for (0..ckt.n) |j| {
            if (x[j] == 0) continue;
            for (ckt.col_ptr[j]..ckt.col_ptr[j + 1]) |p| out[ckt.row_idx[p]] += alpha * ckt.c_vals[p] * x[j];
        }
    }

    /// Advances K and Kxy over one accepted step of length h ending at x,
    /// whose planes `ckt` holds. `flicker` gives the poles, `scale` the PSD
    /// factor of the white sources.
    fn step(c: *Covariance, ckt: *const root.Circuit, h: f64, flicker: *const Flicker, scale: f64) !void {
        const n = c.n;
        assert(n == ckt.n);
        assert(c.kxy.len == flicker.source.len * n);
        const alpha = 1 / h;
        // m = P K P^T, K symmetric: a gets the columns of P K by rows,
        // turns into P K, and row i of m (= column i) is P (P K)[i,:]^T.
        @memset(c.a, 0);
        for (0..n) |i| addCx(ckt, alpha, c.k[i * n ..][0..n], c.a[i * n ..][0..n]);
        transpose(c.a, n);
        @memset(c.m, 0);
        for (0..n) |i| addCx(ckt, alpha, c.a[i * n ..][0..n], c.m[i * n ..][0..n]);
        for (c.inj.lead, 0..) |src, g| {
            const nodes, const weights = c.inj.of(g);
            stampVec(c.m, n, nodes, weights, src.white * scale * 0.5 * alpha);
        }
        for (flicker.source, flicker.rate, flicker.variance, 0..) |g, rate, v, p| {
            const nodes, const weights = c.inj.of(g);
            // u = D P Kxy[p]: the cross term P Kxy D J^T is u j_p^T.
            const row = c.kxy[p * n ..][0..n];
            const u = c.a[0..n];
            @memset(u, 0);
            addCx(ckt, alpha * @exp(-rate * h), row, u);
            for (0..n) |i| for (nodes, weights) |node, w| {
                c.m[i * n + node] += w * u[i];
                c.m[node * n + i] += w * u[i];
            };
            stampVec(c.m, n, nodes, weights, v);
            // Kxy[p] <- u + v j_p, solved below.
            numerics.copySimd(row, u);
            for (nodes, weights) |node, w| row[node] += w * v;
        }
        if (c.slv == null) {
            c.vals = try c.gpa.alloc(f64, ckt.nnz);
            c.slv = try direct.Solver.init(c.gpa, ckt.n, ckt.col_ptr, ckt.row_idx, null, ckt.knobs.lu);
        }
        const slv = &c.slv.?;
        ckt.combineGC(alpha, c.vals);
        try slv.factor(c.vals, .{});
        for (0..flicker.source.len) |p| slv.solve(c.kxy[p * n ..][0..n], c.kxy[p * n ..][0..n]);
        // K = A^-1 (A^-1 m)^T, m symmetric: rows of m are its columns.
        for (0..n) |i| slv.solve(c.m[i * n ..][0..n], c.m[i * n ..][0..n]);
        transpose(c.m, n);
        for (0..n) |i| slv.solve(c.m[i * n ..][0..n], c.k[i * n ..][0..n]);
    }

    /// Variance of x[a] - x[b].
    fn variance(c: Covariance, a: u32, b: u32) f64 {
        const n = c.n;
        assert(a < n);
        assert(b < n);
        return c.k[a * n + a] + c.k[b * n + b] - c.k[a * n + b] - c.k[b * n + a];
    }
};

/// Newton hook: the backward-Euler companion from the q plane plus the
/// sampled noise currents on top of the device residual. Matrix G + C/dt.
const NoiseHook = struct {
    alpha: f64,
    q_prev: []const f64,
    a_vals: []f64,
    has_charge: bool,
    /// Where each group's current enters, `noise_currents` in group order.
    inj: Injection,
    noise_currents: []const f64,

    /// Stamps the planes at `x`, then the BE companion and this step's
    /// noise currents into the residual rows (converger hook).
    pub fn assemble(self: NoiseHook, ckt: *root.Circuit, x: []const f64, t: f64) void {
        assert(self.noise_currents.len == self.inj.lead.len);
        ckt.eval(x, t);
        if (self.has_charge) {
            const n: usize = ckt.n;
            integrator.companionAt(.backward_euler, true, ckt.rhs[0..n], ckt.q_vec[0..n], self.q_prev[0..n], &.{}, &.{}, .{ .ag0 = self.alpha, .ag2 = 0 });
        }
        for (self.noise_currents, 0..) |i_n, g| {
            const nodes, const weights = self.inj.of(g);
            for (nodes, weights) |node, w| ckt.rhs[node] += w * i_n;
        }
    }

    /// Returns the Newton matrix values: G itself without charge, else
    /// G + C/dt rebuilt into `a_vals`, overwriting the previous return.
    pub fn vals(self: NoiseHook, ckt: *root.Circuit) []f64 {
        if (!self.has_charge) return ckt.g_vals;
        ckt.combineGC(self.alpha, self.a_vals);
        return self.a_vals;
    }

    /// Returns one diagonal of the combined matrix without materializing it
    /// (`Circuit.gcAt`); the residual gate calls this per unknown.
    pub fn diagAt(self: NoiseHook, ckt: *root.Circuit, slot: u32) f64 {
        return if (self.has_charge) ckt.gcAt(self.alpha, slot) else ckt.g_vals[slot];
    }
};

/// Returns the noise sources at ctx.x_op integrated as point-major rows
/// (time, probes..., and `onoise` for SDE) in `ctx.allocator`. A run cut
/// short by dt_min says so in the plot name rather than failing, and a run
/// of a SAMPLES set names its index.
pub fn run(ctx: *const root.RunCtx, opts: Options) !root.Result {
    const a = ctx.allocator;
    const scratch = ctx.scratch_allocator;
    const x = try scratch.dupe(f64, ctx.x_op);
    defer scratch.free(x);
    const srcs = try ctx.circuit.collectNoiseSources(ctx.x_op, scratch);
    defer scratch.free(srcs);

    // Recorded straight into the results arena; the Result borrows it
    // unless SDE appends its onoise column.
    const est_rows = 2.0 * opts.t_stop / opts.dt_init;
    var wf = try Waveform.init(a, @intCast(ctx.probes.len), @intFromFloat(@min(@max(rows_min, est_rows), rows_max)));
    errdefer wf.deinit();
    var onoise: std.ArrayList(f64) = .empty;
    defer onoise.deinit(scratch);
    const completed = try simulate(ctx.circuit, x, ctx.probes, srcs, &wf, &onoise, opts, scratch);
    if (opts.sde) assert(onoise.items.len == wf.len);

    const probe_names = try root.probeNames(ctx, "time");
    var names = probe_names;
    var data = wf.data();
    if (opts.sde) {
        const s = wf.stride();
        const width = s + 1;
        const rows = try a.alloc(f64, @as(usize, wf.len) * width);
        for (onoise.items, 0..) |v, r| {
            @memcpy(rows[r * width ..][0..s], data[r * s ..][0..s]);
            rows[r * width + s] = v;
        }
        data = rows;
        const all = try a.realloc(@constCast(probe_names), width);
        all[s] = "onoise";
        names = all;
    }
    const base = if (completed) "Transient Noise Analysis" else "Transient Noise Analysis (stopped early)";
    return .{
        .plotname = if (opts.sample == 0) base else try a.print("{s} (sample={d})", .{ base, opts.sample }),
        .varnames = names,
        .is_complex = false,
        .npoints = wf.len,
        .data = data,
    };
}

// ---- private ----

/// Integrates from `x`, recording every accepted point into `waveform`.
/// Returns false when dt fell below dt_min or `max_steps` ran out before
/// t_stop. There is no LTE control: the noise dominates the local error,
/// so dt only shrinks (by half) on a Newton failure and otherwise grows
/// 1.5x up to dt_max. With `options.sde` nothing is drawn, and `onoise`
/// gets the rms noise of v(out_node, out_neg) at every recorded point,
/// grown on `scratch`.
fn simulate(
    ckt: *root.Circuit,
    x: []f64,
    probes: []const u32,
    noise_sources: []const NoiseSource,
    waveform: *Waveform,
    onoise: *std.ArrayList(f64),
    options: Options,
    scratch: Allocator,
) !bool {
    const n: usize = ckt.n;
    const has_charge = ckt.has_charge;
    assert(x.len == n);
    assert(options.dt_init > 0);

    const ws = try ckt.workspace();
    const x_try = try scratch.alloc(f64, n);
    defer scratch.free(x_try);

    // Split the source table for the run, one entry per correlated group:
    // sampling streams sqrt(white), the dt-independent factor of sigma, and
    // injection streams the weighted endpoints. Valid because
    // collectNoiseSources ran once on x_op.
    // ponytail: a table row's PSD (`NoiseSource.table`) is not sampled; it
    // needs a shaping filter per table. Add one when a transient-noise deck
    // carries a noise_table.
    const inj = try Injection.init(scratch, noise_sources);
    defer inj.deinit(scratch);
    const groups = inj.lead.len;
    const noise_currents = try scratch.alloc(f64, groups);
    defer scratch.free(noise_currents);
    const noise_prefix = try scratch.alloc(f64, groups);
    defer scratch.free(noise_prefix);
    for (inj.lead, noise_prefix) |src, *pfx| pfx.* = @sqrt(src.white * options.scale);
    var flicker = try Flicker.init(scratch, inj.lead, options);
    defer flicker.deinit(scratch);
    var cov: ?Covariance = if (options.sde) try Covariance.init(scratch, n, flicker.source.len, inj) else null;
    defer if (cov) |*c| c.deinit(scratch);
    @memset(noise_currents, 0);

    var a_vals: []f64 = &.{};
    var q_prev: []f64 = &.{};
    defer if (has_charge) scratch.free(q_prev);
    if (has_charge) {
        a_vals = try ws.ensureAVals(ckt.nnz);
        q_prev = try scratch.alloc(f64, n);
        ckt.eval(x, 0);
        numerics.copySimd(q_prev, ckt.q_vec[0..n]);
    }

    var rng = Xorshift64.init(options.seed);
    try waveform.record(0, x, probes);
    if (cov != null) try onoise.append(scratch, 0);

    var t: f64 = 0;
    var dt: f64 = options.dt_init;
    var steps: u32 = 0;
    const nr_opts = converger.optionsFromTolerances(options.tol, options.tol.itl4);
    var attempts: u64 = 0;
    while (t < options.t_stop and steps < options.max_steps) : (attempts += 1) {
        if (attempts != 0) try ckt.checkpoint(.{ .phase = .transient, .completed = attempts });
        // HSPICE TIME=: land on the time exactly, then march on.
        const land = if (options.t_break) |tb| t < tb and t + dt >= tb else false;
        if (land) dt = options.t_break.? - t;
        if (cov == null) {
            // sqrt(BW), BW = 1/(2dt).
            const bandwidth_scale = @sqrt(1.0 / (2.0 * dt));
            for (noise_prefix, noise_currents) |pfx, *i_n| i_n.* = pfx * bandwidth_scale * rng.randn();
            flicker.step(dt, &rng, noise_currents);
        }
        const hook: NoiseHook = .{
            .alpha = 1.0 / dt,
            .q_prev = q_prev,
            .a_vals = a_vals,
            .has_charge = has_charge,
            .inj = inj,
            .noise_currents = noise_currents,
        };

        // As in tran, devices must see the point this attempt targets
        // (§9.10 `$abstime`); a rejected step retries from here.
        ckt.setSimState(.{ .t = t + dt, .dt = dt, .kind = .tran, .initial_step = steps == 0 });
        numerics.copySimd(x_try, x);
        const nr = converger.run(ckt, ws, x_try, t + dt, nr_opts, hook) catch |err| switch (err) {
            error.QueryCancelled, error.OutOfMemory => return err,
            else => converger.Result{ .converged = false, .iterations = 0 }, // else: a numerical failure retries the step
        };
        if (!nr.converged) {
            _ = ckt.stateCtl(.revert);
            dt *= 0.5;
            if (dt < options.dt_min) return false;
            continue;
        }
        _ = ckt.stateCtl(.commit);

        if (has_charge) {
            // The planes are one iterate behind the converged point. Charges
            // only: the next assemble restamps the other planes.
            ckt.evalQ(x_try, t + dt);
            numerics.copySimd(q_prev, ckt.q_vec[0..n]);
        }
        if (cov) |*c| {
            ckt.eval(x_try, t + dt);
            try c.step(ckt, dt, &flicker, options.scale);
            // Clamps round-off below zero; not @max, which would turn NaN into 0.
            const v = c.variance(options.out_node, options.out_neg);
            try onoise.append(scratch, @sqrt(if (v < 0) 0 else v));
        }

        numerics.copySimd(x, x_try);
        t = if (land) options.t_break.? else t + dt;
        steps += 1;
        try waveform.record(t, x, probes);
        dt = @min(dt * 1.5, options.dt_max);
        if (t + dt > options.t_stop) dt = options.t_stop - t;
    }
    assert(waveform.len == steps + 1);
    return t >= options.t_stop;
}
