//! Frequency sweep in bounded lane chunks. Every point solves against the
//! same right-hand sides (one for ac, stb and noise, one per port for sp),
//! and the lane axis is frequency (`FreqSolver.solveBatch`). One lane pass,
//! `FreqSolver.W` points, is live at once, so a sweep holds W rhs.len values
//! instead of points rhs.len.
//! In: a linearized `FreqSolver`, the omegas and right-hand sides. Out: each
//! point's stacked-real solutions, in order.
//! Cites no outside source; dev/analysis/README.md describes the lane axis.
//!
//! Contents:
//! - `pub const quantum: usize`
//! - `pub const Plan = struct` (`Diagonal`, `configure`, `deinit`)
//! - `pub const Point = struct { k, x }`
//! - `pub const Stream = struct` (`init`, `deinit`, `next`)
//! - `pub fn axis(scratch: Allocator, sweep: FreqSweep) Allocator.Error!struct { []f64, []f64 }`
//! - `pub fn freeAxis(scratch: Allocator, freqs: []f64) void`
//! - `pub fn solveAll(gpa, fs, ckt, x_op, omegas, rhs, adjoint: bool, out: []f64) !void`

const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;
const FreqSweep = @import("core").numerics.FreqSweep;
const freq_solve = @import("solver").freq_solve;
const FreqSolver = freq_solve.FreqSolver;
const root = @import("../types.zig");
const lanes = FreqSolver.W;

// ponytail: fixed at 64; expose a work budget if a caller needs another
// latency/throughput tradeoff. One factorization is never split.
/// Frequencies per progress checkpoint; a multiple of every `FreqSolver.W`.
pub const quantum: usize = 64;

/// One frequency analysis' leading sweep: what its `Stream` solves. Built
/// by the analysis' `plan`; `deinit` returns the axis block and `rhs` to
/// the scratch allocator that built them.
pub const Plan = struct {
    freqs: []f64,
    omegas: []f64,
    rhs: []f64,
    adjoint: bool,
    /// Analysis-local additions after linearization, such as SP port
    /// terminations. The provider applies them only to its solver copy.
    g_diagonal: []Diagonal = &.{},

    /// `value` added at G's diagonal entry (`row`, `row`): siemens on a node
    /// row, ohms (-z0) on a branch row.
    pub const Diagonal = struct { row: u32, value: f64 };

    /// Adds `g_diagonal` to `fs`'s copy of G; call it before `fs`'s first
    /// solve (`FreqSolver.addDiagG`).
    pub fn configure(p: Plan, fs: *FreqSolver) void {
        for (p.g_diagonal) |d| fs.addDiagG(d.row, d.value);
    }

    /// Frees the axis block, `rhs` and `g_diagonal`; `scratch` must be the
    /// allocator `plan` used.
    pub fn deinit(p: Plan, scratch: Allocator) void {
        freeAxis(scratch, p.freqs);
        scratch.free(p.rhs);
        scratch.free(p.g_diagonal);
    }
};

/// Point k's stacked-real solutions `[re(0..n), im(0..n)]`, one per 2n block
/// of the stream's rhs, valid until the next `next` call.
pub const Point = struct { k: usize, x: []const f64 };

/// Iterator over the solutions of `omegas`, in order. Chunking cannot change
/// a bit of the result: lanes are independent, and the pivot tape carries
/// across `solveBatch` calls the same way it carries across its internal
/// W-chunks. Each chunk adds the circuit's frequency-dependent terms at
/// `x_op` (`Circuit.acDyn`). Borrows `fs`, `x_op`, `omegas` and `rhs` for
/// its whole life. Invariant: `base <= k <= end <= omegas.len`.
pub const Stream = struct {
    fs: *FreqSolver,
    x_op: []const f64,
    omegas: []const f64,
    rhs: []const f64,
    adjoint: bool,
    /// Every point's solutions solved up front by `solveAll` on these same
    /// inputs, `omegas.len * rhs.len` values point-major; `next` then reads
    /// them instead of solving. Empty: solve as it goes.
    pre: []const f64 = &.{},
    buf: []f64,
    /// A chunk's `acDyn` terms, re then im, each `ac_dyn_slots.len` rows of
    /// up to `lanes` points.
    dyn: []f64,
    /// The chunk in `buf` covers omegas[base..end]; `k` is the next point out.
    base: usize = 0,
    end: usize = 0,
    k: usize = 0,

    /// Allocates the chunk buffers for `ckt`, the circuit `fs` linearized;
    /// free them with `deinit` and the same allocator.
    pub fn init(gpa: Allocator, fs: *FreqSolver, ckt: *const root.Circuit, x_op: []const f64, omegas: []const f64, rhs: []const f64, adjoint: bool) Allocator.Error!Stream {
        const chunk = @min(lanes, omegas.len);
        const buf = try gpa.alloc(f64, chunk * rhs.len);
        errdefer gpa.free(buf);

        const dyn = try gpa.alloc(f64, 2 * ckt.ac_dyn_slots.len * chunk);
        return .{ .fs = fs, .x_op = x_op, .omegas = omegas, .rhs = rhs, .adjoint = adjoint, .buf = buf, .dyn = dyn };
    }

    /// Frees the chunk buffers; points returned so far become invalid.
    pub fn deinit(self: *Stream, gpa: Allocator) void {
        gpa.free(self.buf);
        gpa.free(self.dyn);
    }

    /// Returns the next point, or null past the last one. Solves a new chunk
    /// when the current one is spent, then checkpoints `ckt` (which may fail
    /// with error.QueryCancelled) every `quantum` points and at the end.
    /// Invalidates the previous point's `x`. Asserts that `pre`, when set,
    /// holds every point.
    pub fn next(self: *Stream, ckt: *root.Circuit) !?Point {
        const m = self.rhs.len;
        assert(self.pre.len == 0 or self.pre.len == self.omegas.len * m);
        if (self.k == self.end) {
            if (self.end == self.omegas.len) return null;
            self.base = self.end;
            self.end += @min(lanes, self.omegas.len - self.base);
            if (self.pre.len == 0) {
                const chunk = self.omegas[self.base..self.end];
                const terms = ckt.ac_dyn_slots.len * chunk.len;
                const re = self.dyn[0..terms];
                const im = self.dyn[self.dyn.len / 2 ..][0..terms];
                if (terms != 0) ckt.acDyn(self.x_op, chunk, re, im);
                const dyn: freq_solve.Dyn = .{ .slots = ckt.ac_dyn_slots, .re = re, .im = im };
                try self.fs.solveBatch(chunk, dyn, self.rhs, self.buf[0 .. chunk.len * m], self.adjoint);
            }
            if (self.end % quantum == 0 or self.end == self.omegas.len)
                try ckt.checkpoint(.{ .phase = .frequency, .completed = self.end, .total = self.omegas.len });
        }
        assert(self.k >= self.base);
        assert(self.k < self.end);
        defer self.k += 1;
        const x = if (self.pre.len != 0) self.pre[self.k * m ..][0..m] else self.buf[(self.k - self.base) * m ..][0..m];
        return .{ .k = self.k, .x = x };
    }
};

comptime {
    assert(quantum % lanes == 0);
}

/// Returns the frequencies (Hz) of `sweep` and their omega (rad/s), from one
/// block of `scratch` that `freeAxis` returns.
pub fn axis(scratch: Allocator, sweep: FreqSweep) Allocator.Error!struct { []f64, []f64 } {
    const n: usize = sweep.count();
    const block = try scratch.alloc(f64, 2 * n);
    sweep.fill(block[0..n], block[n..]);
    return .{ block[0..n], block[n..] };
}

/// Frees an `axis` block through its `freqs` half.
pub fn freeAxis(scratch: Allocator, freqs: []f64) void {
    scratch.free(freqs.ptr[0 .. 2 * freqs.len]);
}

/// Solves every point of a `Stream` over these inputs into `out`
/// (`omegas.len * rhs.len` values, point-major), bitwise what that stream
/// returns point by point: the precompute behind `Stream.pre`. Checkpoints
/// as the stream does. Asserts that `out` has that length.
pub fn solveAll(gpa: Allocator, fs: *FreqSolver, ckt: *root.Circuit, x_op: []const f64, omegas: []const f64, rhs: []const f64, adjoint: bool, out: []f64) !void {
    assert(out.len == omegas.len * rhs.len);
    var s = try Stream.init(gpa, fs, ckt, x_op, omegas, rhs, adjoint);
    defer s.deinit(gpa);

    while (try s.next(ckt)) |pt| @memcpy(out[pt.k * rhs.len ..][0..rhs.len], pt.x);
    assert(s.k == omegas.len);
}
