//! Resumable GMRES(m), one independent system per point. Device callbacks
//! perform only the requested matrix products or preconditioner applications;
//! every norm, reduction, Arnoldi step and stopping decision stays within its
//! point. Arithmetic follows gmres.zig and core/numerics.zig, including the
//! fixed sixteen-accumulator dot product and singular true-residual check.
//! std only, so the same bodies compile for the host oracle and GPU kernels.
const std = @import("std");

pub const Request = enum(u8) { done = 0, matvec = 1, precond = 2 };
const Phase = enum(u32) { done, cycle_residual, arnoldi_precond, arnoldi_matvec, update_precond, final_residual };

pub const Shape = extern struct {
    points: u64,
    n: u32,
    m: u32,
    max_restarts: u32,
    preconditioned: u32,
    tolerance: f64,
};

/// Each workspace row has Shape.points adjacent values. Offsets are rows,
/// not bytes, and have the same ordering as Gmres's scalar slab.
pub const Layout = extern struct {
    v: u64,
    h: u64,
    cs: u64,
    sn: u64,
    g: u64,
    y: u64,
    w: u64,
    r: u64,
    rows: u64,

    pub fn init(n: u32, m: u32) !Layout {
        if (n == 0 or m == 0) return error.InvalidKrylovShape;
        var next: u64 = 0;
        const take = struct {
            fn rows(offset: *u64, count: u64) !u64 {
                const first = offset.*;
                offset.* = try std.math.add(u64, first, count);
                return first;
            }
        }.rows;
        const nu: u64 = n;
        const mu: u64 = m;
        return .{
            .v = try take(&next, try std.math.mul(u64, mu + 1, nu)),
            .h = try take(&next, try std.math.mul(u64, mu + 1, mu)),
            .cs = try take(&next, mu),
            .sn = try take(&next, mu),
            .g = try take(&next, mu + 1),
            .y = try take(&next, mu),
            .w = try take(&next, nu),
            .r = try take(&next, nu),
            .rows = next,
        };
    }
};

pub const State = extern struct {
    b_norm: f64 = 0,
    abs_tol: f64 = 0,
    residual: f64 = 0,
    cycle: u64 = 0,
    iterations: u32 = 0,
    j: u32 = 0,
    k: u32 = 0,
    singular: u32 = 0,
    converged: u32 = 0,
    phase: Phase = .done,
};

pub fn Bodies(comptime P: fn (type) type) type {
    return struct {
        const Self = @This();
        const Vec = struct { ptr: P(f64), start: u64 = 0 };

        fn get(v: Vec, row: u64, count: u64, point: u64) f64 {
            return v.ptr[(v.start + row) * count + point];
        }

        fn put(v: Vec, row: u64, count: u64, point: u64, value: f64) void {
            v.ptr[(v.start + row) * count + point] = value;
        }

        fn copy(dst: Vec, src: Vec, n: u64, count: u64, point: u64) void {
            for (0..n) |i| put(dst, i, count, point, get(src, i, count, point));
        }

        fn fill(dst: Vec, n: u64, count: u64, point: u64, value: f64) void {
            for (0..n) |i| put(dst, i, count, point, value);
        }

        fn scale(dst: Vec, value: f64, src: Vec, n: u64, count: u64, point: u64) void {
            for (0..n) |i| put(dst, i, count, point, value * get(src, i, count, point));
        }

        fn axpy(dst: Vec, value: f64, src: Vec, n: u64, count: u64, point: u64) void {
            for (0..n) |i| put(dst, i, count, point, get(dst, i, count, point) + value * get(src, i, count, point));
        }

        fn dot(a: Vec, b: Vec, n: u64, count: u64, point: u64) f64 {
            var acc: [16]f64 = @splat(0);
            var i: u64 = 0;
            while (i + 16 <= n) : (i += 16) {
                for (&acc, 0..) |*value, j| value.* += get(a, i + j, count, point) * get(b, i + j, count, point);
            }
            var tail: f64 = 0;
            while (i < n) : (i += 1) tail += get(a, i, count, point) * get(b, i, count, point);
            var width: usize = 8;
            while (width > 0) : (width /= 2) {
                for (0..width) |j| acc[j] += acc[width + j];
            }
            return acc[0] + tail;
        }

        fn norm(v: Vec, n: u64, count: u64, point: u64) f64 {
            const ss = dot(v, v, n, count, point);
            const tiny = std.math.floatMin(f64) / std.math.floatEps(f64);
            if (ss >= tiny and ss < std.math.inf(f64)) return @sqrt(ss);
            for (0..n) |i| {
                const value = get(v, i, count, point);
                if (@abs(value) == std.math.inf(f64)) return @abs(value);
            }
            if (std.math.isNan(ss)) return ss;
            var largest: i64 = 0;
            for (0..n) |i| largest = @max(largest, @as(i64, @bitCast(@abs(get(v, i, count, point)))));
            const maximum: f64 = @bitCast(largest);
            if (maximum == 0) return 0;
            var sum: f64 = 0;
            for (0..n) |i| {
                const t = get(v, i, count, point) / maximum;
                sum += t * t;
            }
            return maximum * @sqrt(sum);
        }

        fn residual(dst: Vec, b: Vec, product: Vec, s: Shape, point: u64) f64 {
            for (0..s.n) |i| put(dst, i, s.points, point, get(b, i, s.points, point) - get(product, i, s.points, point));
            return norm(dst, s.n, s.points, point);
        }

        fn finish(state: *State, value: f64, converged: bool) Request {
            state.residual = value;
            state.converged = @intFromBool(converged);
            state.phase = .done;
            return .done;
        }

        /// Initializes one point, writes its first operator input, and
        /// returns what the driver must execute before calling advance.
        pub fn start(s: Shape, states: P(State), b: P(f64), x: P(f64), input: P(f64), point: u64) Request {
            var state: State = .{};
            state.b_norm = norm(.{ .ptr = b }, s.n, s.points, point);
            const request: Request = blk: {
                if (state.b_norm == 0) {
                    fill(.{ .ptr = x }, s.n, s.points, point, 0);
                    break :blk finish(&state, 0, true);
                }
                if (!std.math.isFinite(state.b_norm)) break :blk finish(&state, state.b_norm, false);
                state.abs_tol = s.tolerance * state.b_norm;
                state.phase = .cycle_residual;
                copy(.{ .ptr = input }, .{ .ptr = x }, s.n, s.points, point);
                break :blk .matvec;
            };
            states[point] = state;
            return request;
        }

        fn arnoldi(s: Shape, layout: Layout, state: *State, work: P(f64), input: P(f64), point: u64) Request {
            state.iterations += 1;
            const v: Vec = .{ .ptr = work, .start = layout.v + @as(u64, state.j) * s.n };
            copy(.{ .ptr = input }, v, s.n, s.points, point);
            if (s.preconditioned != 0) {
                state.phase = .arnoldi_precond;
                return .precond;
            }
            state.phase = .arnoldi_matvec;
            return .matvec;
        }

        fn afterUpdate(s: Shape, layout: Layout, state: *State, work: P(f64), x: P(f64), input: P(f64), point: u64) Request {
            const estimate = @abs(get(.{ .ptr = work, .start = layout.g }, state.k, s.points, point));
            if (state.singular == 0 and estimate <= state.abs_tol) return finish(state, estimate / state.b_norm, true);
            state.cycle += 1;
            state.phase = if (state.cycle <= s.max_restarts) .cycle_residual else .final_residual;
            copy(.{ .ptr = input }, .{ .ptr = x }, s.n, s.points, point);
            return .matvec;
        }

        /// Consumes this point's last callback output. Inactive points
        /// remain untouched; an active point advances until its next request.
        pub fn advance(s: Shape, layout: Layout, states: P(State), b: P(f64), x: P(f64), work: P(f64), input: P(f64), output: P(f64), point: u64) Request {
            var state = states[point];
            const request = advanceState(s, layout, &state, b, x, work, input, output, point);
            states[point] = state;
            return request;
        }

        fn advanceState(s: Shape, layout: Layout, state: *State, b: P(f64), x: P(f64), work: P(f64), input: P(f64), output: P(f64), point: u64) Request {
            const n: u64 = s.n;
            const m: u64 = s.m;
            const count = s.points;
            const r: Vec = .{ .ptr = work, .start = layout.r };
            const h: Vec = .{ .ptr = work, .start = layout.h };
            const g: Vec = .{ .ptr = work, .start = layout.g };
            const cs: Vec = .{ .ptr = work, .start = layout.cs };
            const sn: Vec = .{ .ptr = work, .start = layout.sn };
            const y: Vec = .{ .ptr = work, .start = layout.y };
            switch (state.phase) {
                .done => return .done,
                .cycle_residual => {
                    const beta = residual(r, .{ .ptr = b }, .{ .ptr = output }, s, point);
                    if (beta <= state.abs_tol) return finish(state, beta / state.b_norm, true);
                    scale(.{ .ptr = work, .start = layout.v }, 1.0 / beta, r, n, count, point);
                    fill(g, m + 1, count, point, 0);
                    put(g, 0, count, point, beta);
                    state.j = 0;
                    return arnoldi(s, layout, state, work, input, point);
                },
                .arnoldi_precond => {
                    copy(.{ .ptr = input }, .{ .ptr = output }, n, count, point);
                    state.phase = .arnoldi_matvec;
                    return .matvec;
                },
                .update_precond => {
                    axpy(.{ .ptr = x }, 1, .{ .ptr = output }, n, count, point);
                    return afterUpdate(s, layout, state, work, x, input, point);
                },
                .final_residual => {
                    const value = residual(r, .{ .ptr = b }, .{ .ptr = output }, s, point);
                    return finish(state, value / state.b_norm, value <= state.abs_tol);
                },
                .arnoldi_matvec => {},
            }
            const j: u64 = state.j;
            copy(r, .{ .ptr = output }, n, count, point);
            for (0..j + 1) |i| {
                const vi: Vec = .{ .ptr = work, .start = layout.v + i * n };
                const hij = dot(r, vi, n, count, point);
                put(h, i * m + j, count, point, hij);
                axpy(r, -hij, vi, n, count, point);
            }
            const next_norm = norm(r, n, count, point);
            put(h, (j + 1) * m + j, count, point, next_norm);
            if (next_norm != 0) scale(.{ .ptr = work, .start = layout.v + (j + 1) * n }, 1.0 / next_norm, r, n, count, point);
            for (0..j) |i| {
                const c = get(cs, i, count, point);
                const sine = get(sn, i, count, point);
                const a = get(h, i * m + j, count, point);
                const v = get(h, (i + 1) * m + j, count, point);
                put(h, i * m + j, count, point, c * a + sine * v);
                put(h, (i + 1) * m + j, count, point, -sine * a + c * v);
            }
            const hjj = get(h, j * m + j, count, point);
            const hjp1j = get(h, (j + 1) * m + j, count, point);
            const rotation = givens(hjj, hjp1j);
            put(cs, j, count, point, rotation.c);
            put(sn, j, count, point, rotation.s);
            put(h, j * m + j, count, point, rotation.c * hjj + rotation.s * hjp1j);
            put(h, (j + 1) * m + j, count, point, 0);
            const gj = get(g, j, count, point);
            const gjp1 = get(g, j + 1, count, point);
            put(g, j, count, point, rotation.c * gj + rotation.s * gjp1);
            put(g, j + 1, count, point, -rotation.s * gj + rotation.c * gjp1);
            state.j += 1;
            if (state.j < s.m and next_norm != 0 and !(@abs(get(g, j + 1, count, point)) <= state.abs_tol))
                return arnoldi(s, layout, state, work, input, point);

            state.k = state.j;
            const k: u64 = state.k;
            var d_max: f64 = 0;
            for (0..k) |i| d_max = @max(d_max, @abs(get(h, i * m + i, count, point)));
            const floor = std.math.floatEps(f64) * d_max;
            state.singular = 0;
            copy(y, g, k, count, point);
            var i = k;
            while (i > 0) {
                i -= 1;
                for (i + 1..k) |jj| put(y, i, count, point, get(y, i, count, point) - get(h, i * m + jj, count, point) * get(y, jj, count, point));
                const diag = get(h, i * m + i, count, point);
                if (@abs(diag) <= floor) {
                    put(y, i, count, point, 0);
                    state.singular = 1;
                } else put(y, i, count, point, get(y, i, count, point) / diag);
            }
            if (s.preconditioned != 0) {
                const w: Vec = .{ .ptr = work, .start = layout.w };
                fill(w, n, count, point, 0);
                for (0..k) |v| axpy(w, get(y, v, count, point), .{ .ptr = work, .start = layout.v + v * n }, n, count, point);
                copy(.{ .ptr = input }, w, n, count, point);
                state.phase = .update_precond;
                return .precond;
            }
            for (0..k) |v| axpy(.{ .ptr = x }, get(y, v, count, point), .{ .ptr = work, .start = layout.v + v * n }, n, count, point);
            return afterUpdate(s, layout, state, work, x, input, point);
        }
    };
}

fn givens(a: f64, b: f64) struct { c: f64, s: f64 } {
    if (b == 0) return .{ .c = 1, .s = 0 };
    if (a == 0) return .{ .c = 0, .s = std.math.sign(b) };
    const r = std.math.hypot(a, b);
    return .{ .c = a / r, .s = b / r };
}

test "masked GMRES matches scalar histories for mixed zero, singular, NaN, restarted and preconditioned points" {
    const t = std.testing;
    const Gmres = @import("../gmres.zig").Gmres;
    const B = Bodies(@import("lane.zig").HostP);
    const N = 11;
    const n = 19; // both full dot accumulators and its scalar tail
    const depth = 4;
    const Op = struct {
        point: usize,
        pub fn matvec(self: *const @This(), v: []const f64, w: []f64) void {
            for (0..n) |i| {
                w[i] = switch (self.point) {
                    2 => 0, // singular, inconsistent
                    4 => std.math.nan(f64), // non-finite operator
                    0, 1, 3, 5, 9, 10 => v[i],
                    else => @as(f64, @floatFromInt(i + 2)) * v[i] + 0.125 * v[(i + 1) % n],
                };
            }
        }
        pub fn precond(_: *const @This(), v: []f64) void {
            for (v, 0..) |*x, i| x.* /= @as(f64, @floatFromInt(i + 2));
        }
    };
    const Bare = struct {
        op: Op,
        pub fn matvec(self: *const @This(), v: []const f64, w: []f64) void {
            self.op.matvec(v, w);
        }
    };
    const layout = try Layout.init(n, depth);
    const work = try t.allocator.alloc(f64, layout.rows * N);
    defer t.allocator.free(work);
    for ([_]bool{ false, true }) |preconditioned| {
        const shape: Shape = .{ .points = N, .n = n, .m = depth, .max_restarts = 3, .preconditioned = @intFromBool(preconditioned), .tolerance = 1e-10 };
        var states: [N]State = undefined;
        var requests: [N]Request = undefined;
        var b: [n * N]f64 = undefined;
        var x: [n * N]f64 = undefined;
        var input: [n * N]f64 = undefined;
        var output: [n * N]f64 = undefined;
        var expected: [N]Gmres.SolveResult = undefined;
        var want: [N][n]f64 = undefined;
        for (0..N) |p| {
            var rhs: [n]f64 = undefined;
            for (0..n) |i| {
                const magnitude: f64 = switch (p) {
                    0 => 0,
                    3 => std.math.nan(f64),
                    5 => 1e-200, // rescaled norm
                    9 => 1e200,
                    10 => std.math.inf(f64),
                    else => @as(f64, @floatFromInt(i + 1)) / 7,
                };
                rhs[i] = magnitude;
                b[i * N + p] = magnitude;
                const initial: f64 = if (p == 1 or p == 8) magnitude else 0;
                want[p][i] = initial;
                x[i * N + p] = initial;
            }
            var scalar = try Gmres.init(t.allocator, n, depth);
            defer scalar.deinit(t.allocator);
            const op: Op = .{ .point = p };
            expected[p] = if (preconditioned)
                scalar.solve(&op, &rhs, &want[p], shape.tolerance, shape.max_restarts)
            else
                scalar.solve(&Bare{ .op = op }, &rhs, &want[p], shape.tolerance, shape.max_restarts);
            requests[p] = B.start(shape, &states, &b, &x, &input, p);
        }
        var rounds: usize = 0;
        while (std.mem.indexOfNone(Request, &requests, &.{.done}) != null) {
            rounds += 1;
            try t.expect(rounds < 100);
            for (0..N) |p| {
                if (requests[p] == .done) continue;
                var v: [n]f64 = undefined;
                var out: [n]f64 = undefined;
                for (0..n) |i| v[i] = input[i * N + p];
                const op: Op = .{ .point = p };
                switch (requests[p]) {
                    .done => unreachable,
                    .matvec => op.matvec(&v, &out),
                    .precond => {
                        op.precond(&v);
                        out = v;
                    },
                }
                for (0..n) |i| output[i * N + p] = out[i];
            }
            // Reverse the control walk to ensure no neighbouring point
            // can affect a reduction, restart or stopping decision.
            var p: usize = N;
            while (p > 0) {
                p -= 1;
                requests[p] = B.advance(shape, layout, &states, &b, &x, work.ptr, &input, &output, p);
            }
        }
        var min_iters: u32 = std.math.maxInt(u32);
        var max_iters: u32 = 0;
        for (states, expected, 0..) |state, oracle, p| {
            try t.expectEqual(oracle.iterations, state.iterations);
            try t.expectEqual(oracle.converged, state.converged != 0);
            if (std.math.isNan(oracle.residual)) {
                try t.expect(std.math.isNan(state.residual));
            } else try t.expectEqual(@as(u64, @bitCast(oracle.residual)), @as(u64, @bitCast(state.residual)));
            for (0..n) |i| {
                if (std.math.isNan(want[p][i])) {
                    try t.expect(std.math.isNan(x[i * N + p]));
                } else try t.expectEqual(@as(u64, @bitCast(want[p][i])), @as(u64, @bitCast(x[i * N + p])));
            }
            min_iters = @min(min_iters, state.iterations);
            max_iters = @max(max_iters, state.iterations);
        }
        try t.expect(max_iters > min_iters);
    }
}
