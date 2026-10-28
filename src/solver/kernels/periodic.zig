//! Sampled HB Jacobian kernels: per-point time samples of G and C in, the
//! harmonic-balance Jacobian rows and products per point out. Points are the
//! innermost dimension; each point's reductions are local and inactive points
//! are never read. std only, so the same bodies compile for the host oracle
//! and the GPU lane kernels (`lane_device.zig`).
//! Cites: Kundert, Sorkin and Sangiovanni-Vincentelli, "Applying harmonic
//! balance to almost-periodic circuits", IEEE T-MTT 36(2), 1988 (sampled
//! Jacobian products).
//!
//! Contents:
//! - `pub const Shape = extern struct`
//! - `pub const Rows = struct`
//!   - `pub fn init(a: std.mem.Allocator, col_ptr: []const u32, row_idx: []const u32) !Rows`
//!   - `pub fn deinit(self: Rows, a: std.mem.Allocator) void`
//! - `pub fn Bodies(comptime P: fn (type) type) type`
//!     - `pub fn inverse(t: Shape, row: usize, p: usize, active: P(u8), inv: P(f64), x: P(f64), time: P(f64)) void`
//!     - `pub fn product(t: Shape, row: usize, p: usize, active: P(u8), row_ptr: P(u32), columns: P(u32), slots: P(u32), g: P(f64), c: P(f64), time: P(f64), gv: P(f64), cv: P(f64)) void`
//!     - `pub fn forward(t: Shape, row: usize, p: usize, active: P(u8), fwd: P(f64), omega: P(f64), gv: P(f64), cv: P(f64), out: P(f64)) void`
//!     - `pub fn pack(t: Shape, i: usize, p: usize, active: P(u8), input: P(f64), rhs: P(f64)) void`
//!     - `pub fn unpack(t: Shape, i: usize, p: usize, active: P(u8), identity: P(u8), rhs: P(f64), x: P(f64), out: P(f64)) void`
//!     - `pub fn copy(t: Shape, i: usize, p: usize, active: P(u8), input: P(f64), out: P(f64)) void`

const std = @import("std");
const lk = @import("lane.zig");

/// The batch's dimensions: `points` independent points, `n` nodes, `nf`
/// real spectral coefficients and `nt` time samples; `charge` nonzero
/// adds the C (omega) terms.
pub const Shape = extern struct {
    points: u64,
    n: u32,
    nf: u32,
    nt: u32,
    charge: u32,
};

/// Incoming CSC slots, grouped by output row and retaining column order.
/// The caller frees all three allocations with `deinit`.
pub const Rows = struct {
    ptr: []u32,
    columns: []u32,
    slots: []u32,

    /// Transposes the CSC pattern's slots into rows, O(n + nnz).
    pub fn init(a: std.mem.Allocator, col_ptr: []const u32, row_idx: []const u32) !Rows {
        const n = col_ptr.len - 1;
        const nz = col_ptr[n];
        const ptr = try a.alloc(u32, n + 1);
        errdefer a.free(ptr);
        const columns = try a.alloc(u32, nz);
        errdefer a.free(columns);
        const slots = try a.alloc(u32, nz);
        errdefer a.free(slots);
        const cursor = try a.alloc(u32, n);
        defer a.free(cursor);
        @memset(ptr, 0);
        for (row_idx[0..nz]) |row| ptr[row + 1] += 1;
        for (1..n + 1) |i| ptr[i] += ptr[i - 1];
        @memcpy(cursor, ptr[0..n]);
        for (0..n) |col| for (col_ptr[col]..col_ptr[col + 1]) |slot| {
            const row = row_idx[slot];
            const at = cursor[row];
            columns[at] = @intCast(col);
            slots[at] = @intCast(slot);
            cursor[row] += 1;
        };
        return .{ .ptr = ptr, .columns = columns, .slots = slots };
    }

    /// Frees the three arrays with the allocator `init` got.
    pub fn deinit(self: Rows, a: std.mem.Allocator) void {
        a.free(self.ptr);
        a.free(self.columns);
        a.free(self.slots);
    }
};

/// The per-point bodies over the pointer type `P`; each returns without
/// touching an inactive point (`active[p] == 0`), except `pack`.
pub fn Bodies(comptime P: fn (type) type) type {
    return struct {
        /// Row is (node, time sample). x is (node, coefficient, point).
        pub fn inverse(t: Shape, row: usize, p: usize, active: P(u8), inv: P(f64), x: P(f64), time: P(f64)) void {
            if (active[p] == 0) return;
            const N: usize = @intCast(t.points);
            const node = row / t.nt;
            const s = row % t.nt;
            time[row * N + p] = dot(x, node * t.nf * N + p, N, inv, s * t.nf, t.nf);
        }

        /// G(t)v(t), C(t)v(t), with incoming slots in CSC column order.
        /// No atomic sum is needed; one thread owns each output row.
        pub fn product(t: Shape, row: usize, p: usize, active: P(u8), row_ptr: P(u32), columns: P(u32), slots: P(u32), g: P(f64), c: P(f64), time: P(f64), gv: P(f64), cv: P(f64)) void {
            if (active[p] == 0) return;
            const N: usize = @intCast(t.points);
            const node = row / t.nt;
            const s = row % t.nt;
            var value_g: f64 = 0;
            var value_c: f64 = 0;
            for (row_ptr[node]..row_ptr[node + 1]) |i| {
                const v = time[(@as(usize, columns[i]) * t.nt + s) * N + p];
                const at = (@as(usize, slots[i]) * t.nt + s) * N + p;
                value_g += g[at] * v;
                if (t.charge != 0) value_c += c[at] * v;
            }
            gv[row * N + p] = value_g;
            cv[row * N + p] = value_c;
        }

        /// Row is (node, coefficient). DFT(Gv) + omega DFT(Cv);
        /// each frequency's cosine and sine charge rows cross here.
        pub fn forward(t: Shape, row: usize, p: usize, active: P(u8), fwd: P(f64), omega: P(f64), gv: P(f64), cv: P(f64), out: P(f64)) void {
            if (active[p] == 0) return;
            const N: usize = @intCast(t.points);
            const node = row / t.nf;
            const coeff = row % t.nf;
            var value = dot(gv, node * t.nt * N + p, N, fwd, coeff * t.nt, t.nt);
            if (t.charge != 0 and coeff != 0) {
                const odd = coeff % 2 != 0;
                const partner = if (odd) coeff + 1 else coeff - 1;
                const charge = dot(cv, node * t.nt * N + p, N, fwd, partner * t.nt, t.nt);
                const w = omega[((coeff + 1) / 2) * N + p];
                if (odd) value += w * charge else value -= w * charge;
            }
            out[row * N + p] = value;
        }

        /// One point's spectrum into the harmonic-inner stacked RHS
        /// consumed by frequencySolve. Harmonic h is one complex lane.
        pub fn pack(t: Shape, i: usize, p: usize, active: P(u8), input: P(f64), rhs: P(f64)) void {
            // The retained solve still launches its harmonic lanes for an
            // inactive point. Feed finite scratch without reading its x;
            // unpack leaves that point's output untouched.
            if (active[p] == 0) {
                rhs[i] = 0;
                return;
            }
            const N: usize = @intCast(t.points);
            const m = (t.nf + 1) / 2;
            const row = i / m;
            const h = i % m;
            const imag = row >= t.n;
            const node = row % t.n;
            const coeff = if (imag) 2 * h else (2 * h) -| 1;
            rhs[i] = if (imag and h == 0) 0 else input[(node * t.nf + coeff) * N + p] * (if (imag) @as(f64, -1) else 1);
        }

        /// Solved phasors back into node-major real spectral coefficients;
        /// singular held blocks read the original RHS, as solveEach does.
        pub fn unpack(t: Shape, i: usize, p: usize, active: P(u8), identity: P(u8), rhs: P(f64), x: P(f64), out: P(f64)) void {
            if (active[p] == 0) return;
            const N: usize = @intCast(t.points);
            const m = (t.nf + 1) / 2;
            const node = i / t.nf;
            const coeff = i % t.nf;
            const imag = coeff != 0 and coeff % 2 == 0;
            const h = (coeff + 1) / 2;
            const at = (node + (if (imag) @as(usize, t.n) else 0)) * m + h;
            const value = if (identity[h] != 0) rhs[at] else x[at];
            out[i * N + p] = value * (if (imag) @as(f64, -1) else 1);
        }

        /// out = input at (row i, point p), for an active point.
        pub fn copy(t: Shape, i: usize, p: usize, active: P(u8), input: P(f64), out: P(f64)) void {
            if (active[p] == 0) return;
            const at = i * @as(usize, @intCast(t.points)) + p;
            out[at] = input[at];
        }

        // ---- private ----

        /// core.numerics.dot's fixed sixteen-accumulator reduction, with
        /// the left vector strided through independent point lanes.
        fn dot(left: P(f64), base: usize, stride: usize, right: P(f64), offset: usize, count: usize) f64 {
            var acc: [16]f64 = @splat(0);
            var i: usize = 0;
            while (i + 16 <= count) : (i += 16) {
                for (&acc, 0..) |*v, j| v.* += left[base + (i + j) * stride] * right[offset + i + j];
            }
            var tail: f64 = 0;
            while (i < count) : (i += 1) tail += left[base + i * stride] * right[offset + i];
            var width: usize = 8;
            while (width != 0) : (width /= 2) for (0..width) |j| {
                acc[j] += acc[j + width];
            };
            return acc[0] + tail;
        }
    };
}

test "periodic harmonic packing retains each point and held identity blocks" {
    const B = Bodies(lk.HostP);
    const t: Shape = .{ .points = 3, .n = 2, .nf = 5, .nt = 12, .charge = 1 };
    var active = [_]u8{ 1, 0, 1 };
    var identity = [_]u8{ 1, 0, 1 };
    var input: [2 * 5 * 3]f64 = undefined;
    for (&input, 0..) |*v, i| v.* = @as(f64, @floatFromInt(i)) - 7;
    var rhs: [2 * 2 * 3]f64 = undefined;
    var x: [2 * 2 * 3]f64 = undefined;
    var out: [input.len]f64 = @splat(123);
    for (0..3) |p| {
        for (0..rhs.len) |i| B.pack(t, i, p, &active, &input, &rhs);
        for (&x, rhs) |*v, r| v.* = 2 * r;
        for (0..2 * 5) |i| B.unpack(t, i, p, &active, &identity, &rhs, &x, &out);
    }
    for (0..2 * 5) |i| for (0..3) |p| {
        const h = ((i % 5) + 1) / 2;
        const want = if (p == 1) 123 else input[i * 3 + p] * (if (identity[h] != 0) @as(f64, 1) else 2);
        try std.testing.expectEqual(want, out[i * 3 + p]);
    };
}

test "sampled harmonic operator masks poisoned points and matches a dense spectral Jacobian" {
    const a = std.testing.allocator;
    const B = Bodies(lk.HostP);
    const N = 5;
    const n = 3;
    const nf = 17;
    const nt = 36;
    const nz = 7;
    const t: Shape = .{ .points = N, .n = n, .nf = nf, .nt = nt, .charge = 1 };
    const col_ptr = [_]u32{ 0, 2, 5, 7 };
    const row_idx = [_]u32{ 0, 1, 0, 1, 2, 1, 2 };
    const rows = try Rows.init(a, &col_ptr, &row_idx);
    defer rows.deinit(a);
    var active = [_]u8{ 1, 0, 1, 1, 1 };
    var inv: [nt * nf]f64 = undefined;
    var fwd: [nt * nf]f64 = undefined;
    for (0..nt) |s| for (0..nf) |c| {
        const angle = 2 * std.math.pi * @as(f64, @floatFromInt(((c + 1) / 2) * s)) / nt;
        const v = if (c == 0) 1 else if (c % 2 != 0) @cos(angle) else @sin(angle);
        inv[s * nf + c] = v;
        fwd[c * nt + s] = v * (if (c == 0) @as(f64, 1) else 2) / nt;
    };
    var omega: [((nf + 1) / 2) * N]f64 = undefined;
    for (0..(nf + 1) / 2) |h| for (0..N) |p| {
        omega[h * N + p] = @as(f64, @floatFromInt(h)) * (if (p == 3) @as(f64, 0.7) else 0.5);
    };
    var g: [nz * nt * N]f64 = undefined;
    var c: [nz * nt * N]f64 = undefined;
    for (0..nz) |slot| for (0..nt) |s| for (0..N) |p| {
        const at = (slot * nt + s) * N + p;
        g[at] = if (p == 1) std.math.nan(f64) else @as(f64, @floatFromInt(slot + 1)) / 13 + @sin(@as(f64, @floatFromInt(s))) / (if (p == 3) @as(f64, 7) else 10);
        c[at] = if (p == 1) std.math.nan(f64) else @as(f64, @floatFromInt(1 + slot * s)) / 137;
    };
    var x: [n * nf * N]f64 = undefined;
    for (0..n * nf) |row| for (0..N) |p| {
        x[row * N + p] = if (p == 1) std.math.nan(f64) else @cos(@as(f64, @floatFromInt(row))) * (if (p == 3) @as(f64, 2) else 1);
    };
    var time: [n * nt * N]f64 = @splat(123);
    var gv: [n * nt * N]f64 = @splat(123);
    var cv: [n * nt * N]f64 = @splat(123);
    var out: [n * nf * N]f64 = @splat(123);
    for (0..n * nt) |row| for (0..N) |p| B.inverse(t, row, p, &active, &inv, &x, &time);
    for (0..n * nt) |row| for (0..N) |p| B.product(t, row, p, &active, rows.ptr.ptr, rows.columns.ptr, rows.slots.ptr, &g, &c, &time, &gv, &cv);
    for (0..n * nf) |row| for (0..N) |p| B.forward(t, row, p, &active, &fwd, &omega, &gv, &cv, &out);
    for (0..n * nf) |row| for (0..N) |p| {
        const got = out[row * N + p];
        if (p == 1) {
            try std.testing.expectEqual(@as(f64, 123), got);
            continue;
        }
        // Form the full spectral matrix independently, then multiply it
        // by x. This changes summation order and checks all coefficient,
        // source-slot, point and omega mappings.
        const node = row / nf;
        const coeff = row % nf;
        var want: f64 = 0;
        for (0..n) |col| for (0..nf) |in_coeff| {
            var entry: f64 = 0;
            for (col_ptr[col]..col_ptr[col + 1]) |slot| {
                if (row_idx[slot] != node) continue;
                for (0..nt) |s| {
                    var weight = fwd[coeff * nt + s] * g[(slot * nt + s) * N + p];
                    if (coeff != 0) {
                        const odd = coeff % 2 != 0;
                        const partner = if (odd) coeff + 1 else coeff - 1;
                        const term = omega[((coeff + 1) / 2) * N + p] * fwd[partner * nt + s] * c[(slot * nt + s) * N + p];
                        if (odd) weight += term else weight -= term;
                    }
                    entry += weight * inv[s * nf + in_coeff];
                }
            }
            want += entry * x[(col * nf + in_coeff) * N + p];
        };
        try std.testing.expectApproxEqAbs(want, got, 1e-12);
        if (p == 2 or p == 4) try std.testing.expectEqual(@as(u64, @bitCast(out[row * N])), @as(u64, @bitCast(got)));
    };
}
