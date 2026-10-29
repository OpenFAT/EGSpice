//! The latency bypass of one device batch: a Newton iterate's limited
//! unknowns in, either the instance's recorded stamps replayed to first
//! order or a fresh, recorded evaluation out. Opt-in (`Hooks.set_bypass`):
//! it moves values within the Newton tolerance. Cites SPICE's bypass test
//! (the unknowns and the predicted current both within tolerance) and the
//! hand-kept model list `traits.bypass_models`.
//!
//! Contents:
//! - `pub fn Bypass(comptime D: type) type`: the per-batch record (`tol`,
//!   `ref`, `meta`, `writes`, `hits`, `misses`, `off`) with `bypassable`,
//!   `max_writes`, `review`, `setBypass`, `freeBypass`, `evalBypass`

const std = @import("std");
const assert = std.debug.assert;
const contract = @import("contract");
const abi = @import("device_abi");
const dual = @import("dual.zig");
const layout = @import("layout.zig");
const stamp = @import("stamp.zig");
const traits = @import("traits.zig");
const sink_mod = @import("sink.zig");
const staging_gpa = @import("proto_store.zig").staging_gpa;
const DeviceBatch = @import("batch.zig").DeviceBatch;
const Planes = abi.Planes;
const SimState = abi.SimState;
const RecordWrite = sink_mod.RecordWrite;
const RecordSink = sink_mod.RecordSink;

/// Per instance of `DeviceBatch(D)`: the limited unknowns and the writes of
/// its last full Newton evaluation, and what that evaluation depended on.
/// Lives on `staging_gpa`, like every buffer a worker thread may create.
pub fn Bypass(comptime D: type) type {
    const n_u: usize = contract.nU(D);
    const has_q = @hasDecl(D, "q");
    const has_limit = @hasDecl(D, "limit");
    return struct {
        tol: abi.BypassTol,
        /// `count * n_u`: each instance's limited unknowns at its recording.
        ref: []f64,
        /// `count`: what each recording depends on; `len` 0 is none.
        meta: []Meta,
        /// `count * max_writes`: each recording's sink writes, in order.
        writes: []RecordWrite,
        /// Instance evaluations that replayed and that evaluated, summed
        /// over ParEval's lanes; read between solves (`setSimState`).
        /// usize: wasm32 has no 64-bit atomics.
        // ponytail: wraps after 2^32 evaluations on 32-bit targets, which
        // only skews `review`'s ratio there; u64 again if that matters.
        hits: std.atomic.Value(usize) = .init(0),
        misses: std.atomic.Value(usize) = .init(0),
        /// Set once the hit rate proved too low to pay for recording: the
        /// batch evaluates plainly for the rest of the run.
        off: bool = false,

        const Bp = @This();
        const Batch = DeviceBatch(D);
        const Meta = struct { class: u16 = 0, len: u16 = 0, key: u64 = 0 };

        /// Whether a Newton evaluation of D is a pure function of the
        /// instance's limited unknowns, its bytes and the iteration class:
        /// no time-dependent stimulus, no Newton history, no evaluation that
        /// writes the instance beyond a status latch, and not a constant
        /// Jacobian (whose evaluation costs less than a replay).
        pub const bypassable = !(@hasDecl(D, "nextBreakpoint") and traits.skipsTimerState(D)) and
            !traits.hasSpiceBreaks(D) and !traits.skipsTimerState(D) and
            !@hasDecl(D, "advanceIteration") and !@hasDecl(D, "checkConvergence") and
            !(@hasDecl(D, "mutable_eval") and D.mutable_eval and !traits.statusOnlyMutable(D)) and
            !(@hasDecl(D, "constant") and D.constant.g) and @hasDecl(D, "u_kinds") and
            @sizeOf(D.Instance) <= 4096 and traits.bypass_models.has(traits.baseName(D));

        /// Upper bound on one instance's sink writes in a Newton pass: each
        /// written row's residual and pattern entries, the same for charge,
        /// and the LTE sites.
        pub const max_writes: usize = blk: {
            @setEvalBranchQuota(100_000);
            const jp = layout.rowPattern(D, "jac_pattern");
            const jr = layout.writtenRows(D, "jac_rows");
            var w: usize = 0;
            for (0..n_u) |ru| if (jr[ru]) {
                w += 1 + @popCount(jp[ru]);
            };
            if (has_q) {
                const qp = layout.rowPattern(D, "q_pattern");
                const qr = layout.writtenRows(D, "q_rows");
                for (0..n_u) |ru| if (qr[ru]) {
                    w += 1 + @popCount(qp[ru]);
                };
                w += layout.lteSites(D).len;
            }
            break :blk w;
        };

        comptime {
            // `Meta.len` holds a recording's write count, the local row and
            // column ride in 8 bits each of `RecordWrite.tag`.
            if (bypassable) assert(max_writes <= std.math.maxInt(u16));
            if (bypassable) assert(n_u <= 256);
        }

        /// Turns the bypass off once, after four evaluations per instance, a
        /// batch replays under one in five: it pays the recording for
        /// nothing (a BSIM4 whose instance bytes move every step: 3%; a CMOS
        /// chain: 85%).
        pub fn review(bp: *Bp, count: u32) void {
            if (bp.off) return;
            const h = bp.hits.load(.monotonic);
            const total = h + bp.misses.load(.monotonic);
            if (total < 4 * @as(u64, count)) return;
            if (h * 5 < total) bp.off = true;
        }

        /// `Hooks.set_bypass`: sets the tolerances, clearing every
        /// recording, or frees the bypass when `tol` is null. Best effort:
        /// out of memory leaves the batch evaluating plainly.
        pub fn setBypass(ctx: *anyopaque, tol: ?abi.BypassTol) void {
            const self = Batch.of(ctx);
            if (comptime !bypassable) return;
            if (self.bypass) |bp| {
                bp.tol = tol orelse return freeBypass(self);
                if (!bp.off) @memset(bp.meta, .{});
                return;
            }
            const t = tol orelse return;
            const ref = staging_gpa.alloc(f64, self.count * n_u) catch return;
            const meta = staging_gpa.alloc(Meta, self.count) catch return staging_gpa.free(ref);
            const writes = staging_gpa.alloc(RecordWrite, self.count * max_writes) catch {
                staging_gpa.free(meta);
                return staging_gpa.free(ref);
            };
            const bp = staging_gpa.create(Bp) catch {
                staging_gpa.free(writes);
                staging_gpa.free(meta);
                return staging_gpa.free(ref);
            };
            @memset(meta, .{});
            bp.* = .{ .tol = t, .ref = ref, .meta = meta, .writes = writes };
            self.bypass = bp;
            assert(self.bypass != null);
        }

        /// Frees the batch's bypass records, if any.
        pub fn freeBypass(self: *Batch) void {
            const bp = self.bypass orelse return;
            staging_gpa.free(bp.writes);
            staging_gpa.free(bp.meta);
            staging_gpa.free(bp.ref);
            staging_gpa.destroy(bp);
            self.bypass = null;
        }

        /// Newton evaluation of `[first, last)` with the latency bypass: each
        /// instance replays its recorded writes when its inputs are within
        /// tolerance of the recording, else evaluates (recording afresh). The
        /// writes replay in their recorded order, so each plane cell sums as
        /// a fresh evaluation with those values would.
        pub fn evalBypass(self: *Batch, bp: *Bp, sink: anytype, first: u32, last: u32, sim: SimState, limiting: bool, pl: *const Planes, comptime staged: bool) void {
            assert(first <= last);
            assert(last <= self.count);
            const SinkT = @typeInfo(@TypeOf(sink)).pointer.child;
            // The first iterate neither replays nor records (Meyer models
            // branch on it), so a later iterate's recording reaches the next
            // timepoint.
            if (sim.iteration <= 1) return self.evalSplit(sink, first, last, sim, limiting);
            // Which planes the recorded pointers address.
            var key = std.hash.Wyhash.init(@intFromBool(staged));
            key.update(std.mem.asBytes(&[_]usize{ @intFromPtr(pl.g_vals.ptr), @intFromPtr(pl.c_vals.ptr), @intFromPtr(pl.rhs.ptr), @intFromPtr(pl.q_vec.ptr) }));
            const plane_key = key.final();
            const class = classOf(sim);
            const lim_writes = comptime if (has_limit) layout.limitWritesOf(D) else 0;
            var hits: usize = 0;
            var misses: usize = 0;
            defer {
                _ = bp.hits.fetchAdd(hits, .monotonic);
                _ = bp.misses.fetchAdd(misses, .monotonic);
            }
            var id = first;
            while (id < last) : (id += 1) {
                var lx: [n_u]f64 = undefined;
                var corr = false;
                inline for (0..n_u) |u| {
                    const xg = sink.x(sink.gath(id, u));
                    lx[u] = xg;
                    if (comptime layout.has(lim_writes, u)) if (limiting) {
                        const l = sink.lim(id, u);
                        lx[u] = l;
                        if (@as(u64, @bitCast(xg)) != @as(u64, @bitCast(l))) corr = true;
                    };
                }
                var h = std.hash.Wyhash.init(plane_key);
                // The instance's bytes hold latched and held values; `State`
                // is `updateState`'s staging, which `eval` never reads.
                h.update(std.mem.asBytes(&self.instances[id]));
                const state_key = h.final();
                const m = &bp.meta[id];
                const ref = bp.ref[@as(usize, id) * n_u ..][0..n_u];
                if (!corr and m.class == class and m.key == state_key and
                    replay(bp, bp.writes[@as(usize, id) * max_writes ..][0..m.len], &lx, ref))
                {
                    // Charges and the tape are re-read exactly (values only):
                    // a Meyer model's C is not dq/dx, and the transient
                    // multiplies a charge error by 2/dt.
                    if (comptime has_q) stamp.evalQRange(D, dual.RealFor(@hasDecl(D, "collapse")), sink, id, id + 1, sim, null);
                    hits += 1;
                    continue;
                }
                misses += 1;
                var rec: RecordSink(SinkT) = .{ .inner = sink, .id = id, .out = bp.writes[@as(usize, id) * max_writes ..][0..max_writes] };
                self.evalSplit(&rec, id, id + 1, sim, limiting);
                @memcpy(ref, &lx);
                m.* = if (corr or rec.overflow) .{} else .{ .class = class, .len = @intCast(rec.n), .key = state_key };
            }
            assert(id == last);
        }

        // ---- private ----

        /// The iteration class a recording is keyed on: the analysis kind and
        /// the flags a model branches on. Never 0, the empty `Meta`'s class.
        fn classOf(sim: SimState) u16 {
            const class: u16 = 1 + 2 * @as(u16, @intCast(@backingInt(sim.kind))) +
                64 * @as(u16, @intFromBool(sim.analog_initial)) +
                128 * @as(u16, @intFromBool(sim.initial_step)) +
                256 * @as(u16, @intFromBool(sim.final_step)) +
                512 * @as(u16, @intFromBool(sim.t > 0));
            assert(class & 1 == 1);
            return class;
        }

        /// Replays recording `ws` at `lx` in residual form, f(x) ~ f(ref) +
        /// J (x - ref), when every unknown and every predicted residual
        /// change is within tolerance; false, writing nothing, otherwise.
        fn replay(bp: *const Bp, ws: []const RecordWrite, lx: *const [n_u]f64, ref: []const f64) bool {
            assert(ref.len == n_u);
            assert(ws.len <= max_writes);
            inline for (0..n_u) |u| {
                const tol = bp.tol.reltol * @max(@abs(lx[u]), @abs(ref[u])) +
                    (if (comptime D.u_kinds[u] == .voltage) bp.tol.vntol else bp.tol.abstol);
                if (!(@abs(lx[u] - ref[u]) <= tol)) return false;
            }
            var jl = std.mem.zeroes([n_u][n_u]f64);
            var d: [n_u]f64 = undefined;
            inline for (0..n_u) |u| d[u] = lx[u] - ref[u];
            for (ws) |w| if (w.tag >> 16 == sink_mod.rec_jac) {
                jl[(w.tag >> 8) & 0xff][w.tag & 0xff] += w.val;
            };
            // SPICE's second test: the predicted change itself within
            // tolerance (an exponential junction moves far more current than
            // a vntol-sized step suggests).
            for (ws) |w| if (w.tag >> 16 == sink_mod.rec_res) {
                const dv = delta(w, &jl, &d);
                if (!(@abs(dv) <= bp.tol.reltol * @abs(w.val) + bp.tol.abstol)) return false;
            };
            for (ws) |w| switch (w.tag >> 16) {
                sink_mod.rec_q, 0 => {},
                else => {
                    // SAFETY: `RecordSink` stored a live plane cell's address
                    // with bit 0 marking a store, and the plane key proved the
                    // same planes are bound.
                    const p: *f64 = @ptrFromInt(w.ptr & ~@as(usize, 1));
                    const v = w.val + delta(w, &jl, &d);
                    if (w.ptr & 1 != 0) p.* = v else p.* += v;
                },
            };
            return true;
        }

        /// A recorded residual write's first-order move to `d`; 0 for any
        /// other write.
        fn delta(w: RecordWrite, jl: *const [n_u][n_u]f64, d: *const [n_u]f64) f64 {
            if (w.tag >> 16 != sink_mod.rec_res) return 0;
            var dv: f64 = 0;
            for (jl[(w.tag >> 8) & 0xff], d) |j, dx| dv += j * dx;
            return dv;
        }
    };
}
