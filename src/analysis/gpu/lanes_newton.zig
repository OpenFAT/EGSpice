//! Newton over the lanes of a round. In: the round mask and the
//! lane-uniform tolerances. Out: each lane's converged, singular or reject
//! verdict, the ticks replayed from captured graphs where the driver has
//! them, a failed LU replay solved by the device dense peel or peeled to the
//! lane's own host solver, and the pivot tape rebuilt from a lane when most
//! lanes fail on the stale one.
//! Cites: phase4-gpu-lanes.md G6; `LaneSolver`'s peel and lead rules.
//!
//! Contents:
//! - `pub fn newton(self: *GpuLanes, opts: converger.Options, out: []LaneNewton) !void`
//! - `pub fn newtonOpts(opts: converger.Options) lk.NewtonOpts`
//! - `pub fn ticks(self: *GpuLanes, nopts: lk.NewtonOpts, max_iter: u16, active_in: u64) !void`
//! - `pub fn prepareGraphs(self: *GpuLanes, opts: lk.NewtonOpts) void`
//! - `pub fn capture(self: *GpuLanes, comptime body: anytype, args: anytype) !GraphExec`
//! - `pub fn clearGraphs(self: *GpuLanes) void`
//! - `pub fn retapeFrom(self: *GpuLanes, first: u32) !void`

const std = @import("std");
const assert = std.debug.assert;
const gompute = @import("gompute");
const solver = @import("solver");
const lanes = @import("lanes.zig");
const lanes_batch = @import("lanes_batch.zig");
const lanes_tick = @import("lanes_tick.zig");
const lk = solver.lane_kernels;
const direct = solver.direct;
const converger = solver.converger;
const GpuLanes = lanes.GpuLanes;
const LaneNewton = lanes.LaneNewton;
const Buffer = lanes_batch.Buffer;
const Stream = lanes_batch.Stream;
const GraphExec = lanes_batch.GraphExec;
const arg = gompute.interface.arg;

/// Newton for the round lanes, every lane its own `converger.newton`
/// at its own t (`sims`), with `opts` (lane-uniform). Fills `out[l]`
/// for the round lanes. `clearLimits` at the end.
pub fn newton(self: *GpuLanes, opts: converger.Options, out: []LaneNewton) !void {
    std.debug.assert(opts.gmin_stamps.len == 0 and opts.force.len == 0);
    const nopts = newtonOpts(opts);
    prepareGraphs(self, nopts);
    var active: u64 = 0;
    for (self.round) |r| active += @intFromBool(r != 0);
    self.prof.lane_rounds += active;
    // `predictRound`'s queued copies still read `sims`.
    try self.stream.synchronize();
    const fix: u32 = if (opts.init_fix) 2 else 0;
    for (self.gate, self.round, self.sims) |*g, r, *s| {
        g.* = .{ .limit = 0, .scaled = 0, .iterations = opts.max_iter, .bits = if (r != 0) r | fix else 0 };
        // `beginSolve`.
        s.iteration = 1;
    }
    try self.put(self.buf(.gate), self.gate);
    try self.put(self.buf(.sims), self.sims);
    try self.put(self.buf(.active), self.round);
    @memset(self.fail, 0);
    try self.put(self.buf(.fail), self.fail);
    try ticks(self, nopts, opts.max_iter, active);
    // `clearLimits`, every lane's word.
    for (self.batches) |*b| try b.d_limw.fillAsync(0, 4 * @as(usize, self.n_lanes), &self.stream);
    try self.get(self.gate, self.buf(.gate));
    try self.stream.synchronize();
    for (self.gate, self.round, out) |g, r, *o| {
        if (r == 0) continue;
        const singular = g.bits & 8 != 0;
        o.* = .{
            .result = if (singular) .{ .converged = false, .iterations = 0 } else .{ .converged = g.bits & 4 != 0, .iterations = @intCast(@min(g.iterations, opts.max_iter)) },
            .singular = singular,
            .reject = g.bits & 32 != 0,
        };
    }
}

/// The lane-uniform tolerances the tick kernels take, from `opts`.
pub fn newtonOpts(opts: converger.Options) lk.NewtonOpts {
    return .{ .reltol = opts.reltol, .abstol = opts.abstol, .vntol = opts.vntol, .gmin = opts.gmin, .residual_tol = opts.residual_tol, .max_iter = opts.max_iter };
}

/// Newton's ticks for the lanes in `d_active` (`active` of them, or an
/// upper bound), until every lane leaves or `max_iter`.
pub fn ticks(self: *GpuLanes, nopts: lk.NewtonOpts, max_iter: u16, active_in: u64) !void {
    var active = active_in;
    // Every lane has a dense slot: no failure can need the host this
    // tick, so its words come down once, after graph B.
    const device_only = self.dense_slots >= self.n_lanes;
    var iter: u16 = 0;
    while (iter < max_iter) : (iter += 1) {
        self.prof.lane_ticks += active;
        if (self.graph_a) |*graph| try graph.launch(&self.stream) else try lanes_tick.linearTick(self, nopts);
        if (!device_only) {
            try readWords(self);
            const fails = self.words[0];
            // Most lanes failing means the tape went stale, not a few
            // lanes going bad: refresh it rather than peel every tick.
            const stale = 2 * @as(u64, fails) >= active;
            const host = if (self.dense_slots > 0) self.words[3] else fails;
            if (host > 0) {
                self.prof.peels += host;
                try peel(self, stale);
            } else if (fails > 0 and stale) try retapeFrom(self, self.words[2]);
        }

        if (self.graph_b) |*graph| {
            try graph.launch(&self.stream);
            self.prof.graph_ticks += 1;
        } else try lanes_tick.updateTick(self, nopts);
        self.prof.ticks += 1;
        try readWords(self);
        const count = self.words[1];
        if (device_only) {
            const fails = self.words[0];
            self.prof.dense_peels += fails;
            if (fails > 0 and 2 * @as(u64, fails) >= active) try retapeFrom(self, self.words[2]);
        }
        if (count == 0) break;
        active = count;
    }
}

pub fn prepareGraphs(self: *GpuLanes, opts: lk.NewtonOpts) void {
    if (self.graph_opts) |saved| if (std.meta.eql(saved, opts)) return;
    clearGraphs(self);
    self.graph_opts = opts;
    if (self.prof.phases or self.ckts[0].knobs.gpu_nographs or !self.k.refactor.hasGraphs()) return;
    createGraphs(self, opts) catch |e| {
        std.log.warn("GPU lane graph capture failed ({s}); using direct launches", .{@errorName(e)});
    };
}

/// `body(self, args...)` captured as a graph. Capture on a temporary
/// nonblocking stream, then replay on the normal blocking stream.
/// Keeping that execution stream preserves ordering with the existing
/// synchronous uploads/downloads on the NULL stream.
pub fn capture(self: *GpuLanes, comptime body: anytype, args: anytype) !GraphExec {
    var capture_stream = try self.k.refactor.createStreamNonBlocking();
    defer capture_stream.deinit();
    std.mem.swap(Stream, &self.stream, &capture_stream);
    defer std.mem.swap(Stream, &self.stream, &capture_stream);
    try self.stream.beginCapture(.thread_local);
    var capturing = true;
    defer if (capturing) {
        // Even an invalidated capture must be ended before destroying
        // its stream. No captured work has executed.
        if (self.stream.endCapture()) |abandoned| {
            var graph = abandoned;
            graph.deinit();
        } else |_| {}
    };
    try @call(.auto, body, .{self} ++ args);
    const ended = self.stream.endCapture();
    capturing = false;
    var graph = try ended;
    defer graph.deinit();
    return graph.instantiate();
}

pub fn clearGraphs(self: *GpuLanes) void {
    inline for (.{ "graph_a", "graph_b", "graph_round" }) |f| if (@field(self, f)) |*g| {
        g.deinit();
        @field(self, f) = null;
    };
}

/// The device dense peel saw most lanes fail: factor lane `first`'s
/// matrix on its host solver and make it the tape's lead (`retape`).
/// A singular lead leaves the tape as it was. Called with the stream
/// idle.
pub fn retapeFrom(self: *GpuLanes, first: u32) !void {
    if (first >= self.n_lanes) return;
    const nnz: usize = self.nnz;
    const vals = try self.gpa.alloc(f64, nnz);
    defer self.gpa.free(vals);
    try ensurePeel(self, 1);
    try self.buf(.peel_idx).upload(&first, 4);
    try packLanes(self, 1, nnz, self.buf(.a), 0);
    try self.buf(.peel).download(vals.ptr, nnz * @sizeOf(f64));
    const l: usize = first;
    const slv = &self.ckts[l].ws.?.slv;
    slv.factor(vals, self.ckts[l].solver_execution) catch |e| switch (e) {
        error.SingularMatrix => return,
        else => |other| return other,
    };
    try retape(self, l);
}

// ---- private ----

fn createGraphs(self: *GpuLanes, opts: lk.NewtonOpts) !void {
    try self.stream.synchronize();
    var a = try capture(self, lanes_tick.linearTick, .{opts});
    errdefer a.deinit();
    const b = try capture(self, lanes_tick.updateTick, .{opts});
    self.graph_a = a;
    self.graph_b = b;
}

/// The lanes whose device refactor failed this tick: each factors and
/// solves on its own solver from its downloaded A and rhs, as
/// `LaneSolver` peels; a singular one stays failed (`finalize` marks it).
/// With `refresh`, the first lane that factors becomes the tape's lead
/// (`retape`), as `LaneSolver`'s lead follows its own solver.
fn peel(self: *GpuLanes, refresh: bool) !void {
    const gpa = self.gpa;
    const n: usize = self.n;
    const nnz: usize = self.nnz;
    try self.get(self.fail, self.buf(.fail));
    try self.stream.synchronize();
    // Only `fail == 1` lanes are the host's: 2 is the dense peel's
    // singular verdict, which stands.
    var k: usize = 0;
    for (self.fail) |f| k += @intFromBool(f == 1);
    if (k == 0) return;
    const idx = try gpa.alloc(u32, k);
    defer gpa.free(idx);
    k = 0;
    for (self.fail, 0..) |f, l| if (f == 1) {
        idx[k] = @intCast(l);
        k += 1;
    };
    // Per lane: A (nnz), then rhs (n), then dx (n).
    const per = nnz + 2 * n;
    const host = try gpa.alloc(f64, k * per);
    defer gpa.free(host);
    try ensurePeel(self, k);
    try self.buf(.peel_idx).upload(idx.ptr, k * 4);
    try packLanes(self, k, nnz, self.buf(.a), 0);
    try packLanes(self, k, n, self.buf(.rhs), k * nnz);
    try self.buf(.peel).download(host.ptr, k * (nnz + n) * @sizeOf(f64));
    const xs = host[k * (nnz + n) ..][0 .. k * n];
    var lead: ?usize = null;
    for (idx, 0..) |l32, j| {
        const l: usize = l32;
        const slv = &self.ckts[l].ws.?.slv;
        slv.factor(host[j * nnz ..][0..nnz], self.ckts[l].solver_execution) catch |e| switch (e) {
            error.SingularMatrix => {
                @memset(xs[j * n ..][0..n], 0);
                continue;
            },
            else => |other| return other,
        };
        slv.solveNeg(host[k * nnz + j * n ..][0..n], xs[j * n ..][0..n]);
        self.fail[l] = 0;
        if (lead == null) lead = l;
    }
    // Singular lanes keep fail = 1, so their scattered zeros are unread.
    try self.buf(.peel).uploadAt(xs.ptr, k * (nnz + n) * @sizeOf(f64), k * n * @sizeOf(f64));
    try unpackLanes(self, k, n, k * (nnz + n), self.buf(.dx));
    try self.put(self.buf(.fail), self.fail);
    try self.stream.synchronize();
    if (refresh) if (lead) |l| try retape(self, l);
}

/// The per-tick device words into `words`, one sync.
fn readWords(self: *GpuLanes) !void {
    try self.buf(.fails).downloadAtAsync(&self.words[0], 0, 4, &self.stream);
    try self.buf(.count).downloadAtAsync(&self.words[1], 0, 4, &self.stream);
    try self.buf(.first).downloadAtAsync(&self.words[2], 0, 4, &self.stream);
    try self.buf(.over).downloadAtAsync(&self.words[3], 0, 4, &self.stream);
    try self.stream.synchronize();
}

/// Room in `d_peel_idx`/`d_peel` for `k` packed lanes.
fn ensurePeel(self: *GpuLanes, k: usize) !void {
    if (k <= self.peel_cap) return;
    const k0 = &self.k.refactor;
    const per = (self.nnz + 2 * @as(usize, self.n)) * @sizeOf(f64);
    var idx = try k0.alloc(k * 4);
    errdefer idx.free();
    const buf = try k0.alloc(k * per);
    self.buf(.peel_idx).free();
    self.buf(.peel).free();
    self.buf(.peel_idx).* = idx;
    self.buf(.peel).* = buf;
    self.peel_cap = @intCast(k);
}

/// `arp_lane_pack` of `rows` rows of `src` for the `k` lanes in
/// `d_peel_idx`, into `d_peel` from entry `at`.
fn packLanes(self: *GpuLanes, k: usize, rows: usize, src: *Buffer, at: usize) !void {
    var nl: u64 = self.n_lanes;
    var r: u64 = rows;
    var kk: u64 = k;
    var off: u64 = at;
    try self.launch(&self.k.pack, rows * k, &.{ arg(&nl), arg(&r), arg(&kk), self.buf(.peel_idx).argPtr(), src.argPtr(), self.buf(.peel).argPtr(), arg(&off) });
}

fn unpackLanes(self: *GpuLanes, k: usize, rows: usize, at: usize, dst: *Buffer) !void {
    var nl: u64 = self.n_lanes;
    var r: u64 = rows;
    var kk: u64 = k;
    var off: u64 = at;
    try self.launch(&self.k.unpack, rows * k, &.{ arg(&nl), arg(&r), arg(&kk), self.buf(.peel_idx).argPtr(), self.buf(.peel).argPtr(), arg(&off), dst.argPtr() });
}

/// Rebuilds the replay tape from lane l's solver, just factored on its
/// own matrix, and uploads it. The lanes' L/U values are rewritten by
/// the next refactor, so only their capacity carries over. The captured
/// graphs hold the old tape, so they go; the next `newton` recaptures.
/// Called with the stream idle.
fn retape(self: *GpuLanes, l: usize) !void {
    const slv = &self.ckts[l].ws.?.slv;
    const lu = slv.sparse() orelse return;
    if (!lu.factored) return;
    const built = try lk.buildTab(self.gpa, lu, slv.col_ptr, slv.params.refactor_growth_limit, direct.pivot_tol);
    defer self.gpa.free(built.idx);
    const t = built.tab;
    const k0 = &self.k.refactor;
    const per_lane = @as(usize, self.n_lanes) * @sizeOf(f64);
    var idx = try lanes_batch.upload(k0, std.mem.sliceAsBytes(built.idx));
    errdefer idx.free();
    if (t.nnz_l > self.lx_cap) {
        const lx = try lanes_batch.zeros(k0, t.nnz_l * per_lane);
        self.buf(.lx).free();
        self.buf(.lx).* = lx;
        self.lx_cap = t.nnz_l;
    }
    if (t.nnz_u > self.ux_cap) {
        const ux = try lanes_batch.zeros(k0, t.nnz_u * per_lane);
        self.buf(.ux).free();
        self.buf(.ux).* = ux;
        self.ux_cap = t.nnz_u;
    }
    self.buf(.tab_idx).free();
    self.buf(.tab_idx).* = idx;
    self.tab = t;
    clearGraphs(self);
    self.graph_opts = null;
    self.prof.retapes += 1;
}
