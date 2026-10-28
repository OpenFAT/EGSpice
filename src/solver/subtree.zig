//! Multicore refactor for thin factors: independent subtrees of the column
//! dependency DAG on separate lanes, no handoffs inside a lane.
//!
//! The ticketed kernel (`kernels/lu.zig` `HostRefactor`) hands every column
//! to the next free thread and waits on each of its U entries' columns, so a
//! factor whose columns cost tens of flops (chains, rings, RC trees: F/n
//! under 100) spends its time in handoffs and runs at 0.39-0.57x of one
//! thread (dev/solvers/gpu-lu.md, "Multicore host refactor"). Here a cut
//! step K splits the pivot steps: below K, step k depends only on the steps
//! in U[:,k], all earlier, so the connected components of that DAG are
//! independent and each runs start to finish on one lane in ascending step
//! order, with its own dense accumulator; steps K and up (the separators
//! and supply columns, which depend on everything) run serially after the
//! join. Every column runs `SparseLu.Replay.column` on final inputs, so the
//! factors are bitwise `SparseLu.refactor`'s whatever the schedule.
//!
//! The tail itself runs pipelined when the model prices that faster: the
//! lanes take its steps in order by ticket, and a step waits only for the
//! tail steps in its U[:,k] before reading their L columns (the ticketed
//! kernel's schedule, `kernels/lu.zig`, on `Replay.column`). Supply
//! columns are thick (F/n about 1,700 in the tail of chain_bsim4_10k) and
//! most of their updates come from the finished phases.
//!
//! The cut and the lanes come from a cost model over the factor's structure
//! (`Plan.init`), priced once per pivot epoch. AMD's own elimination forest
//! is wide enough: on chain_bsim4_10k and ring_bsim4_10k the model finds
//! 2.1-2.6x at 8 lanes, the supply columns' tail holding the rest. A
//! level-structure nested dissection was tried for more and filled 116x
//! against AMD's 2.6x there.
//!
//! Contents:
//! - `pub const max_lanes = 16`
//! - `pub const Plan = struct`
//!   - `pub const Options = struct`
//!   - `pub fn phases(self: *const Plan) u32`
//!   - `pub fn init(gpa: Allocator, lu: *const SparseLu, col_ptr: []const u32, threads: u32, opt: Options) Allocator.Error!?Plan`
//!   - `pub fn deinit(self: *Plan, gpa: Allocator) void`
//!   - `pub fn run(self: *Plan, lu: *SparseLu, col_ptr: []const u32, vals: []const f64, growth: f64, io: std.Io) error{SingularMatrix}!void`
//!     - `pub inline fn wait(self: Tail, i: u32) void`

const std = @import("std");
const assert = std.debug.assert;
const SparseLu = @import("sparse_lu.zig").SparseLu;
const Allocator = std.mem.Allocator;
const testing = std.testing;

/// Lanes at most, the caller's included (`num.Execution.lu_threads` caps
/// at 16).
pub const max_lanes = 16;

/// One pivot epoch's schedule: phases of lanes, then a serial tail.
pub const Plan = struct {
    /// `SparseLu.pattern_epoch` the plan was built from.
    epoch: u32,
    /// Lanes per phase, the caller's included.
    width: u32,
    /// Lane l of phase p runs steps[lane_ptr[p * width + l]..][..next],
    /// ascending; phase p covers steps phase_end[p - 1]..phase_end[p]
    /// (from 0), and the phases run one after another.
    lane_ptr: []u32,
    steps: []u32,
    /// Steps cut..n-1 run after the last phase, serially or pipelined:
    /// the first step no phase covers.
    cut: u32,
    /// Done stamps of the pipelined tail, one per step cut..n-1; empty when
    /// the tail runs serially. Step k is final in this run once its stamp
    /// equals `stamp`.
    done: []std.atomic.Value(u32),
    stamp: u32 = 0,
    /// Private accumulators of lanes 1..width-1: n entries each, all-zero
    /// between runs. Lane 0 and the serial tail use `SparseLu.w`.
    w: []f64,
    /// The model's serial cost over its makespan, for the admission print.
    speedup: f64,
    /// The tail's share of the modeled makespan.
    tail_share: f64,

    /// Admission knobs; the defaults are production's.
    pub const Options = struct {
        /// The model's serial cost over its makespan must reach this.
        /// Measured runs reached 0.65-1.0x of the model at 8 lanes.
        min_speedup: f64 = 1.5,
        /// Cost units charged per phase for the lanes' spawn and join.
        spawn: u64 = spawn_cost,
        /// Cost units charged per step of the pipelined tail.
        hop: u64 = hop_cost,
    };

    /// Phases of lanes before the tail, 1 to `max_phases`.
    pub fn phases(self: *const Plan) u32 {
        return @intCast((self.lane_ptr.len - 1) / self.width);
    }

    /// Plans `lu`'s current factor for `threads` lanes. Each phase starts
    /// where the last ended, at s, and ends at the cut K that minimizes
    /// the model's phase time plus the serial cost after K: the phase is
    /// its largest component of the DAG restricted to [s, K) or its cost
    /// over the lanes, whichever is more. Null when the model's makespan
    /// is not `opt.min_speedup` times below the serial replay, or `lu` has
    /// a refactor tape (a factor that small replays in microseconds).
    /// O(phases * nnz(L + U)) time. Caller owns the result; free with
    /// `deinit`. Asserts that `lu` is factored and `threads` is in
    /// [2, max_lanes].
    pub fn init(gpa: Allocator, lu: *const SparseLu, col_ptr: []const u32, threads: u32, opt: Options) Allocator.Error!?Plan {
        assert(lu.factored);
        assert(threads >= 2 and threads <= max_lanes);
        if (lu.tv.items.len != 0) return null;
        const n = lu.n;
        const scratch = try gpa.alloc(u64, 2 * @as(usize, n));
        defer gpa.free(scratch);
        const cost = scratch[0..n];
        const comp = scratch[n..]; // per root: component cost, then its lane
        const parent = try gpa.alloc(u32, n);
        defer gpa.free(parent);

        var total: u64 = 0;
        for (0..n) |k| {
            var c: u64 = col_cost + (lu.up[k + 1] - lu.up[k]) + (lu.lp[k + 1] - lu.lp[k]);
            const col = lu.q[k];
            c += col_ptr[col + 1] - col_ptr[col];
            for (lu.ui.items[lu.up[k]..lu.up[k + 1]]) |i| c += lu.lp[i + 1] - lu.lp[i];
            cost[k] = c;
            total += c;
        }

        // Phase ends, chosen greedily; `done` is the modeled time so far.
        var ends: [max_phases]u32 = undefined;
        var nphases: usize = 0;
        var start: u32 = 0;
        var rest: u64 = total;
        var done: u64 = 0;
        const t: u64 = threads;
        while (nphases < max_phases and start < n) {
            var best_k = start;
            var best = rest;
            var best_phase: u64 = 0;
            var prefix: u64 = 0;
            var largest: u64 = 0;
            for (start..n) |k| {
                unite(lu, parent, comp, cost, start, k);
                prefix += cost[k];
                largest = @max(largest, comp[k]);
                const phase = @max(largest, (prefix + t - 1) / t) + opt.spawn;
                if (phase + (rest - prefix) < best) {
                    best = phase + (rest - prefix);
                    best_k = @intCast(k + 1);
                    best_phase = phase;
                }
            }
            if (best_k == start) break;
            var covered: u64 = 0;
            for (start..best_k) |k| covered += cost[k];
            done += best_phase;
            rest -= covered;
            ends[nphases] = best_k;
            nphases += 1;
            start = best_k;
        }
        if (nphases == 0) return null;
        // The tail pipelined, priced on `comp` (free until the lanes are
        // dealt below) as its finish times.
        const piped = pipeline(lu, start, threads, opt.hop, comp) + opt.spawn;
        const tail = @min(rest, piped);
        const speedup = share(total, done + tail);
        if (speedup < opt.min_speedup) return null;

        var self: Plan = .{ .epoch = lu.pattern_epoch, .width = threads, .lane_ptr = &.{}, .steps = &.{}, .cut = start, .done = &.{}, .w = &.{}, .speedup = speedup, .tail_share = share(tail, done + tail) };
        errdefer self.deinit(gpa);
        if (piped < rest) {
            self.done = try gpa.alloc(std.atomic.Value(u32), n - start);
            @memset(self.done, .init(0));
        }
        self.lane_ptr = try gpa.alloc(u32, nphases * threads + 1);
        self.steps = try gpa.alloc(u32, start);
        self.w = try gpa.alloc(f64, (threads - 1) * @as(usize, n));
        @memset(self.w, 0);
        self.lane_ptr[0] = 0;
        var lo: u32 = 0;
        for (ends[0..nphases], 0..) |hi, p| {
            // The phase's components, dealt largest first to the
            // least-loaded lane (LPT); a root's lane overwrites its cost.
            for (lo..hi) |k| unite(lu, parent, comp, cost, lo, k);
            const roots = cost[lo..hi]; // free once the components are summed
            var nroots: usize = 0;
            for (lo..hi) |k| if (parent[k] == k) {
                roots[nroots] = k;
                nroots += 1;
            };
            std.sort.pdq(u64, roots[0..nroots], @as([]const u64, comp), Desc.lt);
            var load: [max_lanes]u64 = @splat(0);
            for (roots[0..nroots]) |root| {
                var pick: usize = 0;
                for (1..threads) |l| if (load[l] < load[pick]) {
                    pick = l;
                };
                load[pick] += comp[root];
                comp[root] = pick;
            }
            const ptr = self.lane_ptr[p * threads ..][0 .. threads + 1];
            var count: [max_lanes]u32 = @splat(0);
            for (lo..hi) |k| count[comp[find(parent, @intCast(k))]] += 1;
            for (0..threads) |l| ptr[l + 1] = ptr[l] + count[l];
            var fill: [max_lanes]u32 = undefined;
            @memcpy(fill[0..threads], ptr[0..threads]);
            for (lo..hi) |k| {
                const l = comp[find(parent, @intCast(k))];
                self.steps[fill[l]] = @intCast(k);
                fill[l] += 1;
            }
            assert(ptr[threads] == hi);
            lo = hi;
        }
        return self;
    }

    /// Frees the schedule and lane accumulators; `lu` stays the caller's.
    pub fn deinit(self: *Plan, gpa: Allocator) void {
        gpa.free(self.lane_ptr);
        gpa.free(self.steps);
        gpa.free(self.done);
        gpa.free(self.w);
        self.* = undefined;
    }

    /// `lu.refactor(col_ptr, vals, growth)`: each phase's lanes on the
    /// caller plus `width - 1` tasks of `io`, then the tail on the caller.
    /// Fails exactly when `refactor` would, leaving `lu`'s factors unusable
    /// as a failed `refactor` does. Asserts that `lu` is factored on the
    /// plan's epoch.
    pub fn run(self: *Plan, lu: *SparseLu, col_ptr: []const u32, vals: []const f64, growth: f64, io: std.Io) error{SingularMatrix}!void {
        assert(lu.factored);
        assert(lu.pattern_epoch == self.epoch);
        assert(lu.tv.items.len == 0);
        for (lu.void_slots.items) |p| {
            if (vals[p] != 0) return error.SingularMatrix;
        }
        const r = lu.replay(col_ptr, vals, growth);
        const n = lu.n;
        const nl = self.width;
        lu.replay_mult = null;
        var failed = std.atomic.Value(bool).init(false);
        var nonfinite: [max_lanes]u64 = @splat(0);
        var mult: [max_lanes]f64 = @splat(0);
        var futures: [max_lanes - 1]std.Io.Future(void) = undefined;
        for (0..self.phases()) |p| {
            for (1..nl) |l| futures[l - 1] = io.async(lane, .{ r, self.w[(l - 1) * @as(usize, n) ..][0..n], self.laneSteps(p, l), &failed, &nonfinite[l], &mult[l] });
            lane(r, lu.w, self.laneSteps(p, 0), &failed, &nonfinite[0], &mult[0]);
            for (futures[0 .. nl - 1]) |*f| f.await(io);
            if (failed.load(.monotonic)) return error.SingularMatrix;
        }
        var bits: u64 = 0;
        var m: f64 = 0;
        if (self.done.len != 0) {
            self.stamp +%= 1;
            var ticket = std.atomic.Value(u32).init(self.cut);
            const tl: Tail = .{ .r = r, .ticket = &ticket, .failed = &failed, .done = self.done, .cut = self.cut, .stamp = self.stamp };
            for (1..nl) |l| futures[l - 1] = io.async(Tail.work, .{ tl, self.w[(l - 1) * @as(usize, n) ..][0..n], &nonfinite[l], &mult[l] });
            tl.work(lu.w, &nonfinite[0], &mult[0]);
            for (futures[0 .. nl - 1]) |*f| f.await(io);
            if (failed.load(.monotonic)) return error.SingularMatrix;
        } else for (self.cut..n) |k| {
            if (!r.column(lu.w, k, &bits, &m)) return error.SingularMatrix;
        }
        for (nonfinite[0..nl], mult[0..nl]) |b, lm| {
            bits |= b;
            m = @max(m, lm);
        }
        if (bits != 0) return error.SingularMatrix;
        if (growth > 0) lu.replay_mult = m;
    }

    // ---- private ----

    /// Phases at most: each costs a spawn and a join.
    const max_phases = 8;

    /// The pipelined tail's makespan in model units: `threads` workers
    /// take steps cut..n-1 in order, each when free, and a step's update
    /// from a tail step i starts once i is done. `fin` holds each step's
    /// finish time afterwards (n entries, those below `cut` untouched).
    fn pipeline(lu: *const SparseLu, cut: u32, threads: u32, hop: u64, fin: []u64) u64 {
        var free: [max_lanes]u64 = @splat(0);
        var span: u64 = 0;
        for (cut..lu.n) |k| {
            var pick: usize = 0;
            for (1..threads) |l| if (free[l] < free[pick]) {
                pick = l;
            };
            var t = free[pick] + hop + col_cost + (lu.lp[k + 1] - lu.lp[k]);
            for (lu.ui.items[lu.up[k]..lu.up[k + 1]]) |i| {
                if (i >= cut) t = @max(t, fin[i]);
                t += 1 + lu.lp[i + 1] - lu.lp[i];
            }
            fin[k] = t;
            free[pick] = t;
            span = @max(span, t);
        }
        return span;
    }

    const Desc = struct {
        fn lt(c: []const u64, a: u64, b: u64) bool {
            return c[a] > c[b] or (c[a] == c[b] and a < b);
        }
    };

    /// Adds step k to the union-find of the DAG over steps [start, k]:
    /// k joins the component of each U[:,k] step at or after `start`.
    fn unite(lu: *const SparseLu, parent: []u32, comp: []u64, cost: []const u64, start: usize, k: usize) void {
        parent[k] = @intCast(k);
        comp[k] = cost[k];
        for (lu.ui.items[lu.up[k]..lu.up[k + 1]]) |i| {
            if (i < start) continue;
            const ri = find(parent, i);
            if (ri == k) continue;
            parent[ri] = @intCast(k);
            comp[k] += comp[ri];
        }
    }

    /// The pipelined tail: one worker per lane, steps by ticket in order.
    /// The earliest unfinished step waits on nothing, so the tickets drain.
    const Tail = struct {
        r: SparseLu.Replay,
        ticket: *std.atomic.Value(u32),
        failed: *std.atomic.Value(bool),
        done: []std.atomic.Value(u32),
        cut: u32,
        stamp: u32,

        /// Blocks until tail step i, if i is one, is done this run.
        pub inline fn wait(self: Tail, i: u32) void {
            if (i < self.cut) return;
            var spins: u32 = 0;
            while (self.done[i - self.cut].load(.acquire) != self.stamp) : (spins +%= 1) {
                // A descheduled worker may hold the awaited step (HostSy.pause).
                if (spins < 256 or @import("builtin").single_threaded) std.atomic.spinLoopHint() else std.Thread.yield() catch {};
            }
        }

        /// Takes steps until the tickets run out or a step fails. A failed
        /// step still publishes, so no worker waits on it forever.
        fn work(self: Tail, w: []f64, nonfinite: *u64, mult: *f64) void {
            var bits: u64 = nonfinite.*;
            var m: f64 = mult.*;
            while (!self.failed.load(.monotonic)) {
                const k = self.ticket.fetchAdd(1, .monotonic);
                if (k >= self.r.lp.len - 1) break;
                if (!self.r.columnAfter(w, k, &bits, &m, self)) self.failed.store(true, .monotonic);
                self.done[k - self.cut].store(self.stamp, .release);
            }
            nonfinite.* = bits;
            mult.* = m;
        }
    };

    fn laneSteps(self: *const Plan, p: usize, l: usize) []const u32 {
        const i = p * self.width + l;
        return self.steps[self.lane_ptr[i]..self.lane_ptr[i + 1]];
    }

    /// One lane's steps; ORs its non-finite stores into `nonfinite` and
    /// maxes its multipliers into `mult`, both carried over from the
    /// lane's earlier phases.
    fn lane(r: SparseLu.Replay, w: []f64, steps: []const u32, failed: *std.atomic.Value(bool), nonfinite: *u64, mult: *f64) void {
        var bits: u64 = nonfinite.*;
        var m: f64 = mult.*;
        for (steps) |k| {
            if (!r.column(w, k, &bits, &m)) {
                failed.store(true, .monotonic);
                break;
            }
        }
        nonfinite.* = bits;
        mult.* = m;
    }
};

/// Model cost units per column on top of its flops and slots: the
/// scatter, the pivot tests and the loop overheads. ponytail: a guess in
/// flop units; fit it if the admission misjudges a deck.
const col_cost = 8;

/// Model cost of starting and joining one phase's lanes, in the same units
/// (about 1 ns each on the i9-14900HX): the `std.Io` task queue and futex
/// wake-ups, padded for a loaded machine, where the slowest lane decides.
const spawn_cost = 50_000;

/// Model cost per step of the pipelined tail on top of the column's own:
/// the ticket and the done stamp's cache line moving between cores.
/// ponytail: a guess, like `col_cost`.
const hop_cost = 200;

// ---- private ----

fn share(part: u64, total: u64) f64 {
    return @as(f64, @floatFromInt(part)) / @as(f64, @floatFromInt(@max(total, 1)));
}

/// Union-find root with path halving.
fn find(parent: []u32, x: u32) u32 {
    var i = x;
    while (parent[i] != i) {
        parent[i] = parent[parent[i]];
        i = parent[i];
    }
    return i;
}

test "Plan: lanes plus the serial tail are bitwise refactor, failures included" {
    const gpa = testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x5b7);
    const rnd = prng.random();
    // `blocks` random sparse blocks of `bs` rows plus a border of `nb` rows
    // and columns coupled to every block: the components the plan deals out
    // and the tail it runs last.
    const blocks = 6;
    const bs = 40;
    const nb = 3;
    const n: u32 = blocks * bs + nb;
    var col_ptr: std.ArrayList(u32) = .empty;
    defer col_ptr.deinit(gpa);
    var row_idx: std.ArrayList(u32) = .empty;
    defer row_idx.deinit(gpa);
    try col_ptr.append(gpa, 0);
    for (0..n) |j| {
        const blk = j / bs;
        for (0..n) |i| {
            const ib = i / bs;
            const border = i >= blocks * bs or j >= blocks * bs;
            const keep = i == j or (border and rnd.float(f64) < 0.1) or (!border and ib == blk and rnd.float(f64) < 0.12);
            if (keep) try row_idx.append(gpa, @intCast(i));
        }
        try col_ptr.append(gpa, @intCast(row_idx.items.len));
    }
    const nnz = row_idx.items.len;
    const vals = try gpa.alloc(f64, nnz);
    defer gpa.free(vals);
    const fill = struct {
        fn f(cp: []const u32, ri: []const u32, v: []f64, rr: std.Random) void {
            for (0..cp.len - 1) |j| for (cp[j]..cp[j + 1]) |p| {
                v[p] = if (ri[p] == j) 4 + rr.float(f64) else rr.float(f64) - 0.5;
            };
        }
    }.f;
    const q = try gpa.alloc(u32, n);
    defer gpa.free(q);
    for (q, 0..) |*v, i| v.* = @intCast(i);
    var a = try SparseLu.init(gpa, n, col_ptr.items, q);
    defer a.deinit(gpa);
    var b = try SparseLu.init(gpa, n, col_ptr.items, q);
    defer b.deinit(gpa);
    fill(col_ptr.items, row_idx.items, vals, rnd);
    try a.factor(gpa, col_ptr.items, row_idx.items, vals, 1e-3);
    try b.factor(gpa, col_ptr.items, row_idx.items, vals, 1e-3);
    try testing.expect(a.tv.items.len == 0);
    var tail_steps: [n]u32 = undefined;
    const seen = try gpa.alloc(bool, n);
    defer gpa.free(seen);
    // The tail pipelined (no hop cost), then serial (a prohibitive one);
    // inline tasks and four real ones each.
    for ([_]u64{ 0, 1 << 40 }) |hop| for ([_]std.Io.Limit{ .nothing, .limited(4) }) |limit| {
        var plan = (try Plan.init(gpa, &a, col_ptr.items, 4, .{ .min_speedup = 0, .spawn = 0, .hop = hop })).?;
        defer plan.deinit(gpa);
        try testing.expect(plan.cut > 0 and plan.cut < n and plan.phases() >= 1);
        try testing.expectEqual(hop == 0, plan.done.len != 0);
        @memset(seen, false);
        for (plan.steps) |k| {
            try testing.expect(k < plan.cut and !seen[k]);
            seen[k] = true;
        }
        for (tail_steps[0 .. n - plan.cut], plan.cut..) |*k, step| k.* = @intCast(step);
        var threaded = std.Io.Threaded.init(gpa, .{ .async_limit = limit });
        defer threaded.deinit();
        for (0..5) |round| {
            fill(col_ptr.items, row_idx.items, vals, rnd);
            // Rounds 2 and 3 grow an L entry 1e20 past its pivot in the
            // first column that has one: in the first lane, then in the tail.
            const grow: []const u32 = switch (round) {
                2 => plan.laneSteps(0, 0),
                3 => tail_steps[0 .. n - plan.cut],
                else => &.{},
            };
            for (grow) |k| {
                var hit = false;
                for (col_ptr.items[q[k]]..col_ptr.items[q[k] + 1]) |p| if (a.prow[p] > k) {
                    vals[p] *= 1e20;
                    hit = true;
                };
                if (hit) break;
            }
            const ref = b.refactor(col_ptr.items, vals, 1e-12);
            const got = plan.run(&a, col_ptr.items, vals, 1e-12, threaded.io());
            if (ref) |_| {
                try got;
                try testing.expectEqualSlices(u8, std.mem.sliceAsBytes(b.ux.items), std.mem.sliceAsBytes(a.ux.items));
                try testing.expectEqualSlices(u8, std.mem.sliceAsBytes(b.udiag), std.mem.sliceAsBytes(a.udiag));
                try testing.expectEqualSlices(u8, std.mem.sliceAsBytes(b.lx.items), std.mem.sliceAsBytes(a.lx.items));
                try testing.expectEqual(b.replay_mult.?, a.replay_mult.?);
                // The recorded max is the scan's, floored at 1.
                var scan: f64 = 1;
                for (0..n) |k| if (!b.scaled_pivot[k]) for (b.lx.items[b.lp[k]..b.lp[k + 1]]) |v| {
                    scan = @max(scan, @abs(v));
                };
                try testing.expectEqual(scan, b.replay_mult.?);
            } else |e| {
                try testing.expect(round == 2 or round == 3);
                try testing.expectError(e, got);
            }
            // The accumulators are clean after every run.
            for (plan.w) |v| try testing.expectEqual(0, v);
            for (a.w) |v| try testing.expectEqual(0, v);
        }
    };
}
