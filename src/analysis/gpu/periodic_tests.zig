//! Device tests of the periodic engine and its masked GMRES. In: small
//! diagonal and structurally singular systems on the first backend this
//! build carries. Out: pass or fail against independent scalar solves
//! (`solver.gmres.Gmres` and `FreqSolver.solveEach`), bitwise, NaN and
//! infinity points included. Skipped without kernel images.
//! Cites: none.
//!
//! Contents: tests only; their local operators expose `matvec` and `precond`
//! as the Krylov engine requires.

const std = @import("std");
const gompute = @import("gompute");
const artifacts = @import("gompute_kernels");
const solver = @import("solver");
const periodic = @import("periodic.zig");

const FS = solver.freq_solve.FreqSolver;
const SolveResult = solver.gmres.Gmres.SolveResult;
const Setup = periodic.Setup;
const Engine = periodic.Engine;

// ---- private ----

fn testMaskedKrylov(comptime backend: gompute.Backend) !void {
    const t = std.testing;
    const Device = Engine(backend);
    const Krylov = @import("krylov.zig").Engine(backend);
    const Gmres = solver.gmres.Gmres;
    const n = 19; // a full sixteen-value reduction and its scalar tail
    const points = 11; // a partial GPU block with independent request masks
    const depth = 4;
    var columns: [n + 1]u32 = undefined;
    var rows: [n]u32 = undefined;
    for (&columns, 0..) |*v, i| v.* = @intCast(i);
    for (&rows, 0..) |*v, i| v.* = @intCast(i);
    const setup: Setup = .{
        .points = points,
        .n = n,
        .nf = 1,
        .nt = 1,
        .charge = false,
        .col_ptr = &columns,
        .row_idx = &rows,
        .inv = &.{1},
        .fwd = &.{1},
        .omegas = &@as([points]f64, @splat(0)),
    };
    var matrix = try Device.init(t.allocator, setup, false);
    defer matrix.deinit();
    var preconditioner = try Device.init(t.allocator, setup, false);
    defer preconditioner.deinit();
    var krylov = try Krylov.init(t.allocator);
    defer krylov.deinit();
    var diagonal: [n * points]f64 = undefined;
    var inverse: [n * points]f64 = undefined;
    const zero: [n * points]f64 = @splat(0);
    for (0..points) |p| for (0..n) |i| {
        diagonal[i * points + p] = switch (p) {
            2 => 0,
            4 => std.math.nan(f64),
            6, 7, 8 => @floatFromInt(i + 2),
            else => 1,
        };
        inverse[i * points + p] = 1 / @as(f64, @floatFromInt(i + 2));
    };
    try matrix.upload(.g, &diagonal);
    try matrix.upload(.c, &zero);
    try preconditioner.upload(.g, &inverse);
    try preconditioner.upload(.c, &zero);
    var factor_g: [n]f64 = undefined;
    const factor_c: [n]f64 = @splat(0);
    for (&factor_g, 0..) |*v, i| v.* = @floatFromInt(i + 2);
    var host_factor = try FS.fromPlanes(t.allocator, n, &columns, &rows, &factor_g, &factor_c);
    defer host_factor.deinit();
    try host_factor.factorEach(&.{0});
    var device_factors: [points]FS = undefined;
    var factor_ptrs: [points]*FS = undefined;
    var initialized: usize = 0;
    defer for (device_factors[0..initialized]) |*f| f.deinit();
    for (&device_factors, &factor_ptrs) |*f, *ptr| {
        f.* = try FS.fromPlanes(t.allocator, n, &columns, &rows, &factor_g, &factor_c);
        initialized += 1;
        ptr.* = f;
    }
    const DeviceOp = struct {
        matrix: *Device,
        preconditioner: *Device,
        factored: bool,
        pub fn matvec(self: *const @This(), input: *Device.Buffer, output: *Device.Buffer, mask: *Device.Buffer, stream: *Device.Stream) !void {
            try self.matrix.matvec(input, output, mask, stream);
        }
        pub fn precond(self: *const @This(), input: *Device.Buffer, output: *Device.Buffer, mask: *Device.Buffer, stream: *Device.Stream) !void {
            if (self.factored)
                try self.matrix.precond(input, output, mask, stream)
            else
                try self.preconditioner.matvec(input, output, mask, stream);
        }
    };
    const BareDeviceOp = struct {
        op: DeviceOp,
        pub fn matvec(self: *const @This(), input: *Device.Buffer, output: *Device.Buffer, mask: *Device.Buffer, stream: *Device.Stream) !void {
            try self.op.matvec(input, output, mask, stream);
        }
    };
    const HostOp = struct {
        diagonal: [n]f64,
        factor: ?*FS,
        pub fn matvec(self: *const @This(), input: []const f64, output: []f64) void {
            for (self.diagonal, input, output) |d, v, *w| w.* = 0 + d * v;
        }
        pub fn precond(self: *const @This(), values: []f64) void {
            if (self.factor) |factor| {
                var rhs: [2 * n]f64 = @splat(0);
                @memcpy(rhs[0..n], values);
                var solution: [2 * n]f64 = undefined;
                factor.solveEach(&rhs, &solution, false);
                @memcpy(values, solution[0..n]);
                return;
            }
            for (values, 0..) |*v, i| v.* = 0 + (1 / @as(f64, @floatFromInt(i + 2))) * v.*;
        }
    };
    const BareHostOp = struct {
        op: HostOp,
        pub fn matvec(self: *const @This(), input: []const f64, output: []f64) void {
            self.op.matvec(input, output);
        }
    };
    const modes = [_][2]bool{ .{ false, false }, .{ true, false }, .{ true, true } };
    for (modes) |mode| {
        const preconditioned = mode[0];
        const factored = mode[1];
        var b: [n * points]f64 = undefined;
        var x: [n * points]f64 = undefined;
        var expected: [points]SolveResult = undefined;
        var want: [points][n]f64 = undefined;
        for (0..points) |p| {
            var rhs: [n]f64 = undefined;
            var op: HostOp = undefined;
            op.factor = if (factored) &host_factor else null;
            for (0..n) |i| {
                const value: f64 = switch (p) {
                    0 => 0,
                    3 => std.math.nan(f64),
                    5 => 1e-200,
                    9 => 1e200,
                    10 => std.math.inf(f64),
                    else => @as(f64, @floatFromInt(i + 1)) / 7,
                };
                rhs[i] = value;
                b[i * points + p] = value;
                const initial = if (p == 1 or p == 8) value else 0;
                want[p][i] = initial;
                x[i * points + p] = initial;
                op.diagonal[i] = diagonal[i * points + p];
            }
            var scalar = try Gmres.init(t.allocator, n, depth);
            defer scalar.deinit(t.allocator);
            expected[p] = if (preconditioned)
                scalar.solve(&op, &rhs, &want[p], 1e-10, 3)
            else
                scalar.solve(&BareHostOp{ .op = op }, &rhs, &want[p], 1e-10, 3);
        }
        if (factored) {
            try matrix.factor(&diagonal, &zero, &factor_ptrs, &.{0}, &@as([points]u8, @splat(1)));
            for (matrix.factors) |factor| try t.expect(factor != null);
            for (&device_factors) |*factor| try t.expectEqualSlices(u32, host_factor.held_id.items, factor.held_id.items);
        }
        try matrix.upload(.input, &b);
        try matrix.upload(.output, &x);
        const op: DeviceOp = .{ .matrix = &matrix, .preconditioner = &preconditioner, .factored = factored };
        const stats = if (preconditioned)
            try krylov.solve(&op, matrix.buffer(.input), matrix.buffer(.output), points, n, depth, 1e-10, 3)
        else
            try krylov.solve(&BareDeviceOp{ .op = op }, matrix.buffer(.input), matrix.buffer(.output), points, n, depth, 1e-10, 3);
        try t.expect(stats.rounds > 0 and stats.matvec_rounds > 0);
        try t.expectEqual(preconditioned, stats.precond_rounds > 0);
        const got = try krylov.downloadResults(t.allocator);
        defer t.allocator.free(got);
        try matrix.buffer(.output).download(&x, @sizeOf(@TypeOf(x)));
        var min_iterations: u32 = std.math.maxInt(u32);
        var max_iterations: u32 = 0;
        for (got, expected, 0..) |actual, oracle, p| {
            try t.expectEqual(oracle.converged, actual.converged);
            try t.expectEqual(oracle.iterations, actual.iterations);
            try expectSameFloat(oracle.residual, actual.residual);
            for (want[p], 0..) |value, i| try expectSameFloat(value, x[i * points + p]);
            min_iterations = @min(min_iterations, actual.iterations);
            max_iterations = @max(max_iterations, actual.iterations);
        }
        try t.expect(max_iterations > min_iterations);
    }
}

fn expectSameFloat(expected: f64, actual: f64) !void {
    if (std.math.isNan(expected)) return std.testing.expect(std.math.isNan(actual));
    try std.testing.expectEqual(@as(u64, @bitCast(expected)), @as(u64, @bitCast(actual)));
}

fn testIdentityPreconditioner(comptime backend: gompute.Backend) !void {
    const t = std.testing;
    const Device = Engine(backend);
    var dense = try FS.fromPlanes(t.allocator, 2, &.{ 0, 1, 2 }, &.{ 0, 1 }, &.{ 2, 3 }, &.{ 0, 0 });
    defer dense.deinit();
    try dense.factorEach(&.{0});
    try t.expect(dense.heldTape() == null and dense.held_id.items.len == 0);
    var small = try Device.init(t.allocator, .{
        .points = 1,
        .n = 2,
        .nf = 1,
        .nt = 1,
        .charge = false,
        .col_ptr = &.{ 0, 1, 2 },
        .row_idx = &.{ 0, 1 },
        .inv = &.{1},
        .fwd = &.{1},
        .omegas = &.{0},
    }, false);
    defer small.deinit();
    try t.expectError(error.UnsupportedPeriodicPreconditioner, small.refresh(&.{ 2, 3 }, &.{ 0, 0 }, &.{&dense}));
    try t.expectError(error.UnsupportedPeriodicPreconditioner, small.factor(&.{ 2, 3 }, &.{ 0, 0 }, &.{&dense}, &.{0}, &.{1}));

    // Two identical columns make a genuinely singular non-void block.
    // Other rows are diagonal; n exceeds the dense/sparse crossover.
    const n = 19;
    var columns: [n + 1]u32 = undefined;
    var rows: [n + 2]u32 = undefined;
    const conductance: [n + 2]f64 = @splat(1);
    const capacitance: [n + 2]f64 = @splat(0);
    columns[0] = 0;
    columns[1] = 2;
    rows[0..4].* = .{ 0, 1, 0, 1 };
    for (2..n + 1) |i| columns[i] = @intCast(i + 2);
    for (2..n) |i| rows[i + 2] = @intCast(i);
    var sparse = try FS.fromPlanes(t.allocator, n, &columns, &rows, &conductance, &capacitance);
    defer sparse.deinit();
    try sparse.factorEach(&.{0});
    try t.expect(sparse.heldTape() == null);
    try t.expectEqualSlices(u32, &.{0}, sparse.held_id.items);
    var large = try Device.init(t.allocator, .{
        .points = 1,
        .n = n,
        .nf = 1,
        .nt = 1,
        .charge = false,
        .col_ptr = &columns,
        .row_idx = &rows,
        .inv = &.{1},
        .fwd = &.{1},
        .omegas = &.{0},
    }, false);
    defer large.deinit();
    try large.refresh(&conductance, &capacitance, &.{&sparse});
    try large.factor(&conductance, &capacitance, &.{&sparse}, &.{0}, &.{1});
    try t.expectEqualSlices(u32, &.{0}, sparse.held_id.items);
    var rhs: [n]f64 = undefined;
    for (&rhs, 0..) |*v, i| v.* = @floatFromInt(i);
    rhs[0] = -0.0;
    rhs[3] = std.math.nan(f64);
    rhs[5] = std.math.inf(f64);
    var result: [n]f64 = undefined;
    try large.applyHost(&rhs, &result, true);
    for (rhs, result) |expected, actual| try expectSameFloat(expected, actual);

    // A fabricated pivot is also identity under the host LaneLu policy,
    // even though the general transient GPU refactor supports that pivot.
    var void_g = conductance;
    void_g[0] = 3;
    void_g[3] = 3;
    void_g[n + 1] = 0;
    sparse.setPlanes(&void_g, &capacitance);
    try sparse.factorEach(&.{0});
    try t.expectEqualSlices(u32, &.{0}, sparse.held_id.items);
    try large.factor(&void_g, &capacitance, &.{&sparse}, &.{0}, &.{1});
    try t.expectEqualSlices(u32, &.{0}, sparse.held_id.items);
    try large.applyHost(&rhs, &result, true);
    for (rhs, result) |expected, actual| try expectSameFloat(expected, actual);
}

fn testFactorRecovery(comptime backend: gompute.Backend) !void {
    const t = std.testing;
    const Device = Engine(backend);
    const hooks = @import("freq.zig").test_hooks;
    const n = 19;
    const nn = 2 * n;
    const points = 3;
    const harmonics = 3;
    const nf = 2 * harmonics - 1;
    const nt = nf;
    const omegas = [harmonics]f64{ 0, 1, 2 };
    var columns: [n + 1]u32 = undefined;
    var rows: [n]u32 = undefined;
    var conductance: [points][n]f64 = undefined;
    var capacitance: [n]f64 = undefined;
    for (&columns, 0..) |*v, i| v.* = @intCast(i);
    for (&rows, &capacitance, 0..) |*row, *c, i| {
        row.* = @intCast(i);
        c.* = 1e-3 * @as(f64, @floatFromInt(i + 1));
        for (&conductance) |*g| g[i] = @floatFromInt(i + 2);
    }
    // The middle point holds DC as identity. Its zero-frequency pivot
    // candidate fabricates a pivot, so the policy tries another harmonic
    // and then replays its best candidate. Failure indices cover all
    // these non-void candidates and completed neighboring points.
    conductance[1][0] = 0;
    var packed_omegas: [harmonics * points]f64 = undefined;
    for (omegas, 0..) |w, h| {
        for (0..points) |p| packed_omegas[h * points + p] = w;
    }
    const sampled: [n * nt * points]f64 = @splat(0);
    const transform: [nt * nf]f64 = @splat(0);
    var rhs: [harmonics * nn]f64 = undefined;
    for (&rhs, 0..) |*v, i| v.* = @as(f64, @floatFromInt(i % 17)) - 8;
    defer {
        hooks.factor_failure_after = null;
        hooks.factor_candidates = 0;
    }
    for (0..5) |fail_after| {
        var oracle: [points]FS = undefined;
        var resumed: [points]FS = undefined;
        var factor_ptrs: [points]*FS = undefined;
        var oracle_count: usize = 0;
        var resumed_count: usize = 0;
        defer for (oracle[0..oracle_count]) |*f| f.deinit();
        defer for (resumed[0..resumed_count]) |*f| f.deinit();
        for (&oracle, &resumed, &factor_ptrs, &conductance) |*want, *got, *ptr, *g| {
            want.* = try FS.fromPlanes(t.allocator, n, &columns, &rows, g, &capacitance);
            oracle_count += 1;
            got.* = try FS.fromPlanes(t.allocator, n, &columns, &rows, g, &capacitance);
            resumed_count += 1;
            ptr.* = got;
            try want.factorEach(&omegas);
        }
        var device = try Device.init(t.allocator, .{
            .points = points,
            .n = n,
            .nf = nf,
            .nt = nt,
            .charge = true,
            .col_ptr = &columns,
            .row_idx = &rows,
            .inv = &transform,
            .fwd = &transform,
            .omegas = &packed_omegas,
        }, false);
        defer device.deinit();
        hooks.factor_candidates = 0;
        hooks.factor_failure_after = fail_after;
        try t.expectError(error.GpuBackendUnavailable, device.factor(&sampled, &sampled, &factor_ptrs, &omegas, &.{ 1, 1, 1 }));
        try t.expect(hooks.factor_failure_after == null);
        try t.expectEqual(fail_after + 1, hooks.factor_candidates);
        for (&oracle, &resumed) |*want, *got| {
            try got.ensureHeldOnHost(&omegas);
            try t.expectEqual(want.strategy.sp.slv.gen, got.strategy.sp.slv.gen);
            try t.expectEqualSlices(u32, want.held_id.items, got.held_id.items);
            var wanted: [rhs.len]f64 = undefined;
            var actual: [rhs.len]f64 = undefined;
            for ([_]bool{ false, true }) |adjoint| {
                want.solveEach(&rhs, &wanted, adjoint);
                got.solveEach(&rhs, &actual, adjoint);
                for (wanted, actual) |a, b| try expectSameFloat(a, b);
            }
            // A subsequent live sweep is sensitive to the final pivot
            // tape, even when the held preconditioner values happened to
            // agree. Its values and factor generation also stay equal.
            const next = [_]f64{ 5, 11 };
            try want.solveBatch(&next, .{}, rhs[0..nn], wanted[0 .. next.len * nn], false);
            try got.solveBatch(&next, .{}, rhs[0..nn], actual[0 .. next.len * nn], false);
            try t.expectEqual(want.strategy.sp.slv.gen, got.strategy.sp.slv.gen);
            for (wanted[0 .. next.len * nn], actual[0 .. next.len * nn]) |a, b| try expectSameFloat(a, b);
        }
    }
}

test "GPU masked GMRES matches independent scalar solves with mixed convergence and nonfinite points" {
    if (comptime artifacts.has_cuda) return testMaskedKrylov(.cuda);
    if (comptime artifacts.has_hip) return testMaskedKrylov(.hip);
    return error.SkipZigTest;
}

test "GPU periodic preconditioner declines dense factors and preserves explicit sparse identity blocks" {
    if (comptime artifacts.has_cuda) return testIdentityPreconditioner(.cuda);
    if (comptime artifacts.has_hip) return testIdentityPreconditioner(.hip);
    return error.SkipZigTest;
}

test "GPU periodic factor failure resumes current candidates and completed peers without extra pivots" {
    if (comptime artifacts.has_cuda) return testFactorRecovery(.cuda);
    if (comptime artifacts.has_hip) return testFactorRecovery(.hip);
    return error.SkipZigTest;
}
