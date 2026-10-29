//! `FreqSolver.factorEach` and `solveEach`: a list of ω in, one held
//! factor per ω out (dense LUs, stacked-real `LaneLu` groups or complex
//! `ComplexLaneLu` groups on one tape), then solves against them for any
//! number of right-hand sides. An ω whose block is singular, or that no
//! tape could replay, is held as the identity (#43). The held tape can live
//! on a device; `ensureHeldOnHost` replays it on the host bitwise.
//! Cites: #43, #83 (allocation failure is not a numerical one).
//!
//! Contents:
//! - `pub fn heldTape(self: *const FreqSolver) ?freq_batch.LaneChunk`
//! - `pub fn beginHeldTape(self: *FreqSolver, omegas: []const f64, tape: usize) !?freq_batch.LaneChunk`
//! - `pub fn finishHeldTape(self: *FreqSolver, failed: []const u8) void`
//! - `pub fn replayCurrentHeld(self: *FreqSolver, omegas: []const f64) Allocator.Error!usize`
//! - `pub fn completeHeldPolicy(self: *FreqSolver, count: usize, on_host: bool) void`
//! - `pub fn ensureHeldOnHost(self: *FreqSolver, omegas: []const f64) !void`
//! - `pub fn packHeld(self: *const FreqSolver, count: usize, lower: []f64, upper: []f64, diag: []f64) void`
//! - `pub fn factorEach(self: *FreqSolver, omegas: []const f64) !void`
//! - `pub fn factorEachSparse(self: *FreqSolver, sp: *FreqSolver.Sparse, omegas: []const f64) !void`
//! - `pub fn replayTape(self: *FreqSolver, sp: *FreqSolver.Sparse, omegas: []const f64, tape: usize) !usize`
//! - `pub fn replayHeldNumbers(self: *FreqSolver, sp: *FreqSolver.Sparse, lu: *SparseLu, omegas: []const f64) Allocator.Error!usize`
//! - `pub fn solveEach(self: *FreqSolver, rhs: []const f64, x_out: []f64, adjoint: bool) void`
//! - `pub fn freeHeld(gpa: Allocator, sp: *FreqSolver.Sparse) void`
//! - `pub fn freeComplexHeld(gpa: Allocator, sp: *FreqSolver.Sparse) void`
//! - `pub fn ensureLaneWork(gpa: Allocator, sp: *FreqSolver.Sparse, nn: usize) Allocator.Error!void`
//! - `pub fn initGroups(comptime T: type, gpa: Allocator, lu: *SparseLu, groups: usize) Allocator.Error![]T`
//! - `pub fn factorEachComplex(self: *FreqSolver, sp: *FreqSolver.Sparse, omegas: []const f64) !bool`
//! - `pub fn solveEachComplex(self: *FreqSolver, sp: *FreqSolver.Sparse, rhs: []const f64, x_out: []f64, adjoint: bool) void`

const std = @import("std");
const assert = std.debug.assert;
const dense_lu = @import("dense_lu.zig");
const direct = @import("direct.zig");
const lane_lu = @import("lane/lu.zig");
const ComplexLaneLu = @import("lane/clu.zig").ComplexLaneLu;
const order_mod = @import("order.zig");
const SparseLu = @import("sparse_lu.zig").SparseLu;
const num = @import("core").numerics;
const Allocator = std.mem.Allocator;
const freq_solve = @import("freq_solve.zig");
const FreqSolver = freq_solve.FreqSolver;
const freq_batch = @import("freq_batch.zig");
const freq_fill = @import("freq_fill.zig");

/// The frozen tape of the most recent `factorEach`, for a device
/// preconditioner. Dense blocks and an all-identity factorization have
/// no sparse tape and stay on the host.
pub fn heldTape(self: *const FreqSolver) ?freq_batch.LaneChunk {
    const sp = switch (self.strategy) {
        .sp => |*s| s,
        .dense => return null,
    };
    if (sp.held.len == 0) return null;
    return freq_batch.laneChunk(sp, sp.held[0].base);
}

/// Host pivot-selection boundary for a resident factorEach provider.
/// It lays the same fresh candidate tape as replayTape, but leaves
/// numeric harmonic replay to the provider. Null means every harmonic
/// is identity because the candidate is singular or has a void pivot.
/// Until ordinary factorEach rebuilds host factors, solveEach is
/// unavailable.
pub fn beginHeldTape(self: *FreqSolver, omegas: []const f64, tape: usize) !?freq_batch.LaneChunk {
    self.held_location = .invalid;
    assert(omegas.len > 0 and tape < omegas.len);
    const sp = switch (self.strategy) {
        .sp => |*s| s,
        .dense => return error.UnsupportedPeriodicPreconditioner,
    };
    self.held_id.ensureTotalCapacity(self.gpa, omegas.len) catch |err| {
        self.held_location = .recovery_failed;
        return err;
    };
    freeHeld(self.gpa, sp);
    freeComplexHeld(self.gpa, sp);
    self.held_id.clearRetainingCapacity();
    sp.slv.factored = false;
    freq_fill.setOmegaSparse(self.n, sp, omegas[tape], .{}, 0) catch |err| {
        if (err != error.SingularMatrix) {
            // A failed full factor may already have changed its
            // prior pivots. Do not retry from that changed history.
            if (err == error.OutOfMemory) self.held_location = .recovery_failed;
            return err;
        }
        for (0..omegas.len) |k| self.held_id.appendAssumeCapacity(@intCast(k));
        return null;
    };
    const lu = sp.slv.sparse() orelse return error.UnsupportedEngine;
    // LaneLu rejects an entire replay when the base contains a
    // fabricated pivot. The Newton device kernel can replay those,
    // but factorEach must preserve LaneLu's identity decision.
    for (lu.void_col) |is_void| if (is_void) {
        for (0..omegas.len) |k| self.held_id.appendAssumeCapacity(@intCast(k));
        return null;
    };
    return freq_batch.laneChunk(sp, lu);
}

/// Publish one provider replay's failed harmonics in ascending order,
/// for factorEach's point-local choice of its next candidate tape.
pub fn finishHeldTape(self: *FreqSolver, failed: []const u8) void {
    assert(self.held_id.capacity >= failed.len);
    self.held_id.clearRetainingCapacity();
    for (failed, 0..) |bad, k| if (bad != 0)
        self.held_id.appendAssumeCapacity(@intCast(k));
}

/// Materializes numeric harmonic blocks from the current full-factor
/// tape without selecting or refactoring any pivot. A device provider
/// uses this after a failed launch, then continues its original
/// tried/best-candidate policy from the same position.
pub fn replayCurrentHeld(self: *FreqSolver, omegas: []const f64) Allocator.Error!usize {
    assert(omegas.len > 0 and self.strategy == .sp);
    const sp = &self.strategy.sp;
    assert(sp.slv.factored);
    const lu = sp.slv.sparse().?;
    errdefer self.held_location = .recovery_failed;
    try self.held_id.ensureTotalCapacity(self.gpa, omegas.len);
    try ensureLaneWork(self.gpa, sp, self.nn);
    freeHeld(self.gpa, sp);
    freeComplexHeld(self.gpa, sp);
    self.held_id.clearRetainingCapacity();
    return replayHeldNumbers(self, sp, lu, omegas);
}

/// Records only the final result of the complete candidate policy.
/// `on_host` distinguishes recovered numeric factors from resident
/// factors; neither path may run factorEach again for these planes.
pub fn completeHeldPolicy(self: *FreqSolver, count: usize, on_host: bool) void {
    assert(count > 0 and self.held_id.items.len <= count);
    self.held_count = count;
    self.held_location = if (on_host) .host else .device;
}

/// Makes the current completed policy available to solveEach. The
/// caller passes the same frequencies until setPlanes or another
/// factor/solve invalidates it. Untouched points follow factorEach.
pub fn ensureHeldOnHost(self: *FreqSolver, omegas: []const f64) !void {
    switch (self.held_location) {
        .invalid => return factorEach(self, omegas),
        .recovery_failed => return error.OutOfMemory,
        .host, .device => {
            assert(self.held_count == omegas.len);
            errdefer self.held_location = .recovery_failed;
            if (self.strategy == .sp) try ensureLaneWork(self.gpa, &self.strategy.sp, self.nn);
            // A singular final candidate may have no usable LU tape.
            // solveEach already copies every all-identity RHS.
            if (self.held_location == .device and self.held_id.items.len != omegas.len)
                _ = try replayCurrentHeld(self, omegas);
            self.held_location = .host;
        },
    }
}

/// Packs `factorEach`'s retained factors with harmonics innermost,
/// omitting SIMD padding. Identity harmonics get finite dummy factors;
/// consumers mask `held_id` and copy the original right-hand side.
/// The caller sized the slices from `heldTape` and `count`.
pub fn packHeld(self: *const FreqSolver, count: usize, lower: []f64, upper: []f64, diag: []f64) void {
    const sp = &self.strategy.sp;
    const tape = heldTape(self).?;
    assert(count > 0 and sp.held.len == (count + FreqSolver.W - 1) / FreqSolver.W);
    assert(lower.len == tape.lu.lx.items.len * count);
    assert(upper.len == tape.lu.ux.items.len * count);
    assert(diag.len == self.nn * count);
    var held: usize = 0;
    for (0..count) |k| {
        if (held < self.held_id.items.len and self.held_id.items[held] == k) {
            for (0..lower.len / count) |i| lower[i * count + k] = 0;
            for (0..upper.len / count) |i| upper[i * count + k] = 0;
            for (0..self.nn) |i| diag[i * count + k] = 1;
            held += 1;
            continue;
        }
        const group = sp.held[k / FreqSolver.W];
        const lane = k % FreqSolver.W;
        for (group.lx, 0..) |v, i| {
            const values: [FreqSolver.W]f64 = v;
            lower[i * count + k] = values[lane];
        }
        for (group.ux, 0..) |v, i| {
            const values: [FreqSolver.W]f64 = v;
            upper[i * count + k] = values[lane];
        }
        for (group.udiag, 0..) |v, i| {
            const values: [FreqSolver.W]f64 = v;
            diag[i * count + k] = values[lane];
        }
    }
}

/// Factors G + jω_k C for every ω_k in `omegas` and keeps all of them,
/// so `solveEach` can solve a different right-hand side per ω as often
/// as it likes: the block-diagonal preconditioner shape. The sparse
/// strategy holds ceil(len/W) LaneLu groups replaying one pivot tape,
/// taken at the middle ω. When lanes fail to replay it, up to two more
/// tapes are laid at the first failed ω not yet tried, and the tape
/// that held the fewest lanes is kept. A lane that still fails (every
/// lane when A at each tape ω is singular), or a singular dense block,
/// is held as the identity and listed in `held_id`, ascending. Any
/// other factor or solve on `self` invalidates the held
/// factors. Asserts that `omegas` is not empty; the sparse strategy
/// fails with `error.UnsupportedEngine` when the 2n system did not get
/// the sparse LU, the only engine LaneLu replays.
pub fn factorEach(self: *FreqSolver, omegas: []const f64) !void {
    self.held_location = .invalid;
    const allocator = self.gpa;
    assert(omegas.len > 0);
    self.held_id.clearRetainingCapacity();
    // Every ω at most is held, so no append below allocates.
    try self.held_id.ensureTotalCapacity(allocator, omegas.len);
    switch (self.strategy) {
        .dense => |*d| {
            const nn: usize = self.nn;
            if (d.held.len != omegas.len * nn * nn) {
                // Both before either is replaced: a held block without its
                // pivot rows passed this length check on the next call.
                const held = try allocator.alloc(f64, omegas.len * nn * nn);
                errdefer allocator.free(held);
                const held_piv = try allocator.alloc(u32, omegas.len * nn);
                allocator.free(d.held);
                allocator.free(d.held_piv);
                d.held = held;
                d.held_piv = held_piv;
            }
            assert(d.held_piv.len * nn == d.held.len);
            for (omegas, 0..) |omega, k| {
                const a = d.held[k * nn * nn ..][0 .. nn * nn];
                const piv = d.held_piv[k * nn ..][0..nn];
                dense_lu.buildComplexAdmittance(self.n, self.nn, d.g_dense, d.c_mat, omega, a);
                dense_lu.pinVoid(nn, a);
                dense_lu.factorize(nn, a, piv) catch {
                    // A failed factor leaves `piv` partial; the identity
                    // is a factor `solveEach` can read.
                    @memset(a, 0);
                    for (piv, 0..) |*p, i| {
                        p.* = @intCast(i);
                        a[i * nn + i] = 1;
                    }
                    self.held_id.appendAssumeCapacity(@intCast(k));
                };
            }
        },
        .sp => |*sp| try factorEachSparse(self, sp, omegas),
    }
    assert(self.held_id.items.len <= omegas.len);
    completeHeldPolicy(self, omegas.len, true);
}

/// Runs `factorEach` on the sparse strategy: stacked-real `LaneLu` groups
/// on one tape, an ω no tape replays held as the identity.
pub fn factorEachSparse(self: *FreqSolver, sp: *FreqSolver.Sparse, omegas: []const f64) !void {
    const gpa = self.gpa;
    // The sparse solver was built from the same gpa.
    assert(sp.slv.gpa.ptr == gpa.ptr);
    if (self.complex_held and try factorEachComplex(self, sp, omegas)) return;
    freeComplexHeld(gpa, sp);
    try ensureLaneWork(gpa, sp, self.nn);
    var tried: [3]usize = undefined;
    var tries: usize = 0;
    var best: usize = undefined;
    var best_held: usize = std.math.maxInt(usize);
    var tape = omegas.len / 2;
    while (tries < tried.len) {
        tried[tries] = tape;
        tries += 1;
        const held = try replayTape(self, sp, omegas, tape);
        if (held < best_held) {
            best = tape;
            best_held = held;
        }
        if (held == 0) return;
        tape = for (self.held_id.items) |k| {
            if (std.mem.indexOfScalar(usize, tried[0..tries], k) == null) break k;
        } else break;
    }
    // A full factor can depend on the previous pivots (the re-pivot fill
    // cap), so the replayed best may hold a different count.
    if (best != tried[tries - 1]) _ = try replayTape(self, sp, omegas, best);
}

/// Lays a fresh pivot tape by a full factor at `omegas[tape]` and
/// replays it into every LaneLu group. `held_id` receives the ω it
/// cannot replay, every ω when A(omegas[tape]) is singular (no group is
/// built then). Returns how many ω that holds. Asserts that `held_id`
/// has room for every ω (`factorEach` reserves it).
pub fn replayTape(self: *FreqSolver, sp: *FreqSolver.Sparse, omegas: []const f64, tape: usize) !usize {
    const gpa = self.gpa;
    // The groups stay allocated across tapes: `replayHeldNumbers`
    // reuses them while the L/U lengths hold (an HB Newton step or a
    // PAC frequency used to reallocate every one).
    self.held_id.clearRetainingCapacity();
    assert(self.held_id.capacity >= omegas.len);
    sp.slv.factored = false;
    freq_fill.setOmegaSparse(self.n, sp, omegas[tape], .{}, 0) catch |err| {
        freeHeld(gpa, sp);
        if (err != error.SingularMatrix) return err;
        for (0..omegas.len) |k| self.held_id.appendAssumeCapacity(@intCast(k));
        return omegas.len;
    };
    const lu = sp.slv.sparse() orelse return error.UnsupportedEngine;
    return replayHeldNumbers(self, sp, lu, omegas);
}

/// Replays every held group on `lu`'s current tape and returns the
/// groups replayed; the host copy of a device-held policy.
pub fn replayHeldNumbers(self: *FreqSolver, sp: *FreqSolver.Sparse, lu: *SparseLu, omegas: []const f64) Allocator.Error!usize {
    const gpa = self.gpa;
    const groups = (omegas.len + FreqSolver.W - 1) / FreqSolver.W;
    const reuse = sp.held.len == groups and for (sp.held) |h| {
        if (h.lx.len != lu.lx.items.len or h.ux.len != lu.ux.items.len) break false;
    } else true;
    if (reuse) {
        for (sp.held) |*h| h.base = lu;
    } else {
        freeHeld(gpa, sp);
        sp.held = try initGroups(lane_lu.LaneLu(FreqSolver.W), gpa, lu, groups);
    }
    for (sp.held, 0..) |*h, g| {
        const base = g * FreqSolver.W;
        const cnt = @min(FreqSolver.W, omegas.len - base);
        const bad = h.refactor(freq_fill.Stacked(FreqSolver.W).of(sp, freq_batch.chunkOmegas(omegas, base, cnt), .{}, base, cnt), sp.slv.params.refactor_growth_limit);
        for (0..cnt) |l| if ((bad >> @intCast(l)) & 1 != 0)
            self.held_id.appendAssumeCapacity(@intCast(base + l));
    }
    return self.held_id.items.len;
}

/// Solves right-hand side k (the stacked 2n vector at `rhs[k*2n..]`)
/// against `factorEach`'s factor k into the same rows of `x_out`, for
/// every held ω; `adjoint` solves the stacked-real transpose, which is
/// A^H. The identity lanes copy their right-hand side, so `rhs` and
/// `x_out` must not overlap. Asserts that `rhs` holds exactly one
/// vector per ω `factorEach` was given and `x_out` as many, and that
/// the two do not overlap.
pub fn solveEach(self: *FreqSolver, rhs: []const f64, x_out: []f64, adjoint: bool) void {
    const nn: usize = self.nn;
    if (nn == 0) return;
    assert(rhs.len % nn == 0);
    assert(x_out.len >= rhs.len);
    const r0 = @intFromPtr(rhs.ptr);
    const x0 = @intFromPtr(x_out.ptr);
    assert(r0 + rhs.len * @sizeOf(f64) <= x0 or x0 + rhs.len * @sizeOf(f64) <= r0);
    switch (self.strategy) {
        .dense => |d| assert(d.held_piv.len == rhs.len),
        // No groups when no tape could be laid; every ω is held then (#43).
        .sp => |sp| assert(sp.held.len == (rhs.len / nn + FreqSolver.W - 1) / FreqSolver.W or sp.cx_held.len == (rhs.len / nn + FreqSolver.W - 1) / FreqSolver.W or self.held_id.items.len == rhs.len / nn),
    }
    switch (self.strategy) {
        .dense => |*d| for (0..d.held_piv.len / nn) |k| {
            const lu = d.held[k * nn * nn ..][0 .. nn * nn];
            const piv = d.held_piv[k * nn ..][0..nn];
            const b = rhs[k * nn ..][0..nn];
            const x = x_out[k * nn ..][0..nn];
            if (adjoint) dense_lu.solveFactoredT(nn, lu, piv, b, x) else dense_lu.solveFactored(nn, lu, piv, b, x);
        },
        .sp => |*sp| if (sp.cx_held.len != 0) solveEachComplex(self, sp, rhs, x_out, adjoint) else {
            const b_plane = sp.lane_work[0..nn];
            const x_plane = sp.lane_work[nn..];
            const m = rhs.len / nn;
            for (sp.held, 0..) |*h, g| {
                const base = g * FreqSolver.W;
                const cnt = @min(FreqSolver.W, m - base);
                for (b_plane, 0..) |*lane, i| {
                    var row: [FreqSolver.W]f64 = undefined;
                    for (0..FreqSolver.W) |l| row[l] = rhs[(base + @min(l, cnt - 1)) * nn + i];
                    lane.* = row;
                }
                if (adjoint) h.solveT(b_plane, x_plane) else h.solve(b_plane, x_plane);
                for (x_plane, 0..) |lane, i| {
                    const row: [FreqSolver.W]f64 = lane;
                    for (0..cnt) |l| x_out[(base + l) * nn + i] = row[l];
                }
            }
        },
    }
    for (self.held_id.items) |k| @memcpy(x_out[k * nn ..][0..nn], rhs[k * nn ..][0..nn]);
}

/// Frees the stacked-real held groups and leaves `sp.held` empty.
pub fn freeHeld(gpa: Allocator, sp: *FreqSolver.Sparse) void {
    for (sp.held) |*h| h.deinit(gpa);
    gpa.free(sp.held);
    sp.held = &.{};
}

/// Frees the complex held groups and leaves `sp.cx_held` empty.
pub fn freeComplexHeld(gpa: Allocator, sp: *FreqSolver.Sparse) void {
    for (sp.cx_held) |*h| h.deinit(gpa);
    gpa.free(sp.cx_held);
    sp.cx_held = &.{};
}

/// The lane RHS and solution planes, 2 * nn lanes, on first use.
pub fn ensureLaneWork(gpa: Allocator, sp: *FreqSolver.Sparse, nn: usize) Allocator.Error!void {
    if (sp.lane_work.len == 0) sp.lane_work = try gpa.alloc(@Vector(FreqSolver.W, f64), 2 * nn);
}

/// `groups` lane factors on `lu`'s tape; on failure none is left.
pub fn initGroups(comptime T: type, gpa: Allocator, lu: *SparseLu, groups: usize) Allocator.Error![]T {
    const out = try gpa.alloc(T, groups);
    for (out, 0..) |*h, g| h.* = T.init(gpa, lu) catch |err| {
        for (out[0..g]) |*done| done.deinit(gpa);
        gpa.free(out);
        return err;
    };
    return out;
}

/// `factorEach` on the complex tape: one proxy factor at the middle ω,
/// every group replayed on it. False, with no complex groups left, when
/// the tape cannot be laid or some ω fails to replay it; the caller
/// then runs the stacked policy, so its tries and identity holds are
/// unchanged. On success the stacked groups are freed and no ω is held.
pub fn factorEachComplex(self: *FreqSolver, sp: *FreqSolver.Sparse, omegas: []const f64) !bool {
    const gpa = self.gpa;
    if (!try freq_batch.ensureComplex(self, gpa, sp)) {
        self.complex_held = false;
        return false;
    }
    const cx = &sp.cx.?;
    // `solveBatch`'s lanes replay this tape no more.
    cx.tape = false;
    const mid = omegas[omegas.len / 2];
    for (cx.proxy, sp.g_vals, sp.c_vals) |*v, gv, cv| v.* = std.math.hypot(gv, mid * cv);
    cx.lu.factor(gpa, sp.src_col_ptr, sp.src_row_idx, cx.proxy, direct.pivot_tol) catch |err| switch (err) {
        error.OutOfMemory => return err,
        error.SingularMatrix => {
            freeComplexHeld(gpa, sp);
            return false;
        },
    };
    const groups = (omegas.len + FreqSolver.W - 1) / FreqSolver.W;
    const reuse = sp.cx_held.len == groups and for (sp.cx_held) |h| {
        if (h.lr.len != cx.lu.lx.items.len or h.ur.len != cx.lu.ux.items.len) break false;
    } else true;
    if (!reuse) {
        freeComplexHeld(gpa, sp);
        sp.cx_held = try initGroups(ComplexLaneLu(FreqSolver.W), gpa, &cx.lu, groups);
    }
    for (sp.cx_held, 0..) |*h, g| {
        const base = g * FreqSolver.W;
        const cnt = @min(FreqSolver.W, omegas.len - base);
        const bad = h.refactor(freq_fill.ComplexFill(FreqSolver.W).of(sp, freq_batch.chunkOmegas(omegas, base, cnt), .{}, base, cnt), sp.slv.params.refactor_growth_limit);
        if (bad != 0) {
            freeComplexHeld(gpa, sp);
            return false;
        }
    }
    freeHeld(gpa, sp);
    self.held_id.clearRetainingCapacity();
    return true;
}

/// `solveEach` on the complex groups.
pub fn solveEachComplex(self: *FreqSolver, sp: *FreqSolver.Sparse, rhs: []const f64, x_out: []f64, adjoint: bool) void {
    const n: usize = self.n;
    const nn: usize = self.nn;
    const cx = &sp.cx.?;
    const br = cx.work[0..n];
    const bi = cx.work[n..][0..n];
    const xr = cx.work[2 * n ..][0..n];
    const xi = cx.work[3 * n ..][0..n];
    const m = rhs.len / nn;
    for (sp.cx_held, 0..) |*h, g| {
        const base = g * FreqSolver.W;
        const cnt = @min(FreqSolver.W, m - base);
        for (br, bi, 0..) |*vr, *vi, i| {
            var re: [FreqSolver.W]f64 = undefined;
            var im: [FreqSolver.W]f64 = undefined;
            for (0..FreqSolver.W) |l| {
                const b = rhs[(base + @min(l, cnt - 1)) * nn ..];
                re[l] = b[i];
                im[l] = b[n + i];
            }
            vr.* = re;
            vi.* = im;
        }
        if (adjoint) h.solveH(br, bi, xr, xi) else h.solve(br, bi, xr, xi);
        for (xr, xi, 0..) |vr, vi, i| {
            const re: [FreqSolver.W]f64 = vr;
            const im: [FreqSolver.W]f64 = vi;
            for (0..cnt) |l| {
                const x = x_out[(base + l) * nn ..];
                x[i] = re[l];
                x[n + i] = im[l];
            }
        }
    }
}
