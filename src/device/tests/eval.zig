//! Tests for the evaluator: AD scalars, the bit-trick helpers, the runtime
//! vtable and binder, and batch instantiation and hooks. Oracles, per test:
//! `std.math` and the float compare a bit trick replaces, a central
//! difference, the same batch forced wide or run serially, hand-derived
//! stamps of a toy device, and ngspice's breakpoint arithmetic.
const impl = @import("device_eval");
const contract = @import("contract");
const SimState = contract.SimState;
const Batch = impl.Batch;
const DeviceBatch = impl.DeviceBatch;
const Dual = impl.Dual;
const NoiseGen = impl.NoiseGen;
const NoiseSource = impl.NoiseSource;
const ProtoStore = impl.ProtoStore;
const PsdTerm = impl.PsdTerm;
const Real = impl.test_access.Real;
const deviceVtable = impl.deviceVtable;
const std = @import("std");

/// Writes integer card value `v` to `field`, which VerA may narrow to the
/// smallest type its range needs, and marks it given when `given`.
fn cardInt(model: anytype, comptime field: []const u8, v: i64, given: bool) void {
    @field(model.*, field) = @intCast(v);
    if (given) markGiven(model, field);
}

/// LRM 9.19: marks `field` given, through VerA's `setGiven` bitset or an
/// older device's `<field>__given` flag.
fn markGiven(model: anytype, comptime field: []const u8) void {
    const T = @TypeOf(model.*);
    if (comptime @hasDecl(T, "setGiven")) {
        model.setGiven(field);
    } else if (comptime @hasField(T, field ++ "__given")) {
        @field(model.*, field ++ "__given") = true;
    }
}

/// A dense 3x3 pattern over nodes 0..2; slot 9 and row 3 take ground writes.
const dense3: impl.test_access.PatternView = .{
    .col_ptr = &.{ 0, 3, 6, 9 },
    .row_idx = &.{ 0, 1, 2, 0, 1, 2, 0, 1, 2 },
    .n = 3,
    .trash_slot = 9,
};

test "narrow admission rejects a limiter write without a derivative lane" {
    const D = struct {
        pub const U = enum(u8) { a, b, c, d, e, missing };
        pub const deriv_reads: u64 = 0b011111;
        pub const collapse_full: [6]?u8 = .{ null, null, null, 0, 1, null };
        pub const limit_writes: u64 = 0b100000;
        pub fn limit() void {}
    };
    try std.testing.expect(!impl.test_access.narrowEligible(D));
}

test "mixed collapsed batches preserve identity and ranges crossing the narrow boundary" {
    const D = struct {
        pub const U = enum(u8) { p, n, a, b, c, d };
        pub const num_ports: usize = 2;
        pub const Model = struct { gain: f64 = 1, collapsed: bool = false };
        pub const Instance = struct { mfactor: f64 = 1 };
        pub const collapse_full: [6]?u8 = .{ null, null, 0, 1, 0, 1 };
        pub const n_q: usize = 1;
        pub const q_stamps = [_]contract.QStamp(U){
            .{ .site = 0, .row = .p, .sign = 1 },
            .{ .site = 0, .row = .n, .sign = -1 },
        };
        pub fn collapse(comptime _: type, m: *const Model, _: *const Instance) [6]?u8 {
            return if (m.collapsed) collapse_full else @splat(null);
        }
        fn current(comptime S: type, xv: *const [6]S.V, m: *const Model) S.Of(0x3f) {
            const x = contract.probes(@This(), S, xv);
            return x[0].sub(x[1]).add(x[2].sub(x[3]).scale(2)).add(x[4].sub(x[5]).scale(3)).scale(m.gain);
        }
        pub fn eval(comptime S: type, x: *const [6]S.V, m: *const Model, _: *const Instance, _: SimState) contract.Rows(@This(), S) {
            const i = current(S, x, m);
            const zero = S.con(0).to(0x3f);
            return contract.rows(@This(), S, .{ i, i.neg(), zero, zero, zero, zero });
        }
        pub fn q(comptime S: type, x: *const [6]S.V, m: *const Model, _: *const Instance, _: SimState) contract.Sites(@This(), S) {
            return contract.sites(@This(), S, .{current(S, x, m).scale(1e-12)});
        }
    };
    const allocator = std.testing.allocator;
    var proto: ProtoStore(D) = .{};
    for (0..4) |id| {
        const collapsed = id % 2 != 0;
        try proto.append(&.{ .gain = @floatFromInt(id + 1), .collapsed = collapsed }, &.{ .mfactor = @floatFromInt(2 * id + 1) }, if (collapsed) .{ 0, 1, 0, 1, 0, 1 } else .{ 0, 1, 2, 3, 4, 5 });
    }
    var rows: [36]u32 = undefined;
    for (&rows, 0..) |*row, i| row.* = @intCast(i % 6);
    const batch = try ProtoStore(D).finalize(&proto, allocator, .{
        .col_ptr = &.{ 0, 6, 12, 18, 24, 30, 36 },
        .row_idx = &rows,
        .n = 6,
        .trash_slot = 36,
    }).unwrap();
    defer batch.hooks.deinit(batch.ctx, allocator);
    const store: *DeviceBatch(D) = @ptrCast(@alignCast(batch.ctx));
    try std.testing.expectEqual(@as(u32, 2), store.narrow_count);
    const counts = batch.hooks.derivative_counts.?(batch.ctx);
    try std.testing.expectEqual(@as(u32, 2), counts.narrow);
    try std.testing.expectEqual(@as(u32, 2), counts.wide);
    for ([_]u32{ 1, 3, 0, 2 }, 0..) |original, id| {
        const identity = batch.hooks.charge_identity.?(batch.ctx, id);
        try std.testing.expectEqual(original, identity.instance);
        try std.testing.expectEqual(@as(u32, 0), identity.site);
        try std.testing.expectEqual(@as(f64, @floatFromInt(original + 1)), store.models[id].gain);
        try std.testing.expectEqual(@as(f64, @floatFromInt(2 * original + 1)), store.instances[id].mfactor);
    }
    const PlaneSet = struct {
        g: [37]f64 = @splat(0),
        c: [37]f64 = @splat(0),
        rhs: [7]f64 = @splat(0),
        q: [7]f64 = @splat(0),
        fn eval(self: *@This(), b: Batch, first: u32, last: u32) void {
            b.eval(b.ctx, &.{ .g_vals = &self.g, .c_vals = &self.c, .rhs = &self.rhs, .q_vec = &self.q }, first, last, &.{ 1, 0, 2, -1, 3, -2 }, 0);
        }
    };
    var full: PlaneSet = .{};
    full.eval(batch, 0, 4);
    var ranged: PlaneSet = .{};
    ranged.eval(batch, 0, 1);
    ranged.eval(batch, 1, 3); // Same range path used by a worker crossing 2.
    ranged.eval(batch, 3, 4);
    try std.testing.expectEqualDeep(full, ranged);
    store.narrow_count = 0; // Test-only forced-wide reference, identical order.
    var wide: PlaneSet = .{};
    wide.eval(batch, 0, 4);
    try std.testing.expectEqualDeep(full, wide);
    store.narrow_count = 2;

    if (!@import("builtin").single_threaded) {
        // The production ParEval hook writes distinct per-instance positions.
        // One worker deliberately straddles the narrow/wide boundary.
        const Planes = impl.Planes;
        const tapes = batch.hooks.scatter_tapes(batch.ctx);
        var arena = std.heap.ArenaAllocator.init(allocator);
        defer arena.deinit();
        const a = arena.allocator();
        const Staged = struct {
            fn init(gpa: std.mem.Allocator, slots: usize, rows_len: usize) !Planes {
                return .{
                    .g_vals = try gpa.alloc(f64, slots),
                    .c_vals = try gpa.alloc(f64, slots),
                    .rhs = try gpa.alloc(f64, rows_len),
                    .q_vec = try gpa.alloc(f64, rows_len),
                };
            }
            fn run(b: Batch, planes: *const Planes, first: u32, last: u32, mode: anytype) void {
                b.hooks.stage(b.ctx, planes, first, last, &.{ 1, 0, 2, -1, 3, -2 }, 0, mode);
            }
        };
        const serial = try Staged.init(a, tapes.slots.len, tapes.rows.len);
        const parallel = try Staged.init(a, tapes.slots.len, tapes.rows.len);
        inline for (.{ .full, .newton, .charge }) |mode| {
            inline for (.{ "g_vals", "c_vals", "rhs", "q_vec" }) |field| {
                @memset(@field(serial, field), -0.0);
                @memset(@field(parallel, field), -0.0);
            }
            Staged.run(batch, &serial, 0, 4, mode);
            {
                const first = try std.Thread.spawn(.{}, Staged.run, .{ batch, &parallel, 0, 1, mode });
                defer first.join();
                const crossing = try std.Thread.spawn(.{}, Staged.run, .{ batch, &parallel, 1, 3, mode });
                defer crossing.join();
                Staged.run(batch, &parallel, 3, 4, mode);
            }
            inline for (.{ "g_vals", "c_vals", "rhs", "q_vec" }) |field|
                try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(@field(serial, field)), std.mem.sliceAsBytes(@field(parallel, field)));
        }
    }
}

test "BSIM4 noise remains wide and source PSDs are finite nonnegative across modes" {
    const D = @import("models").bsim4va;
    const S = impl.test_access.NoiseFamily(D);
    for ([_]i64{ 1, -1 }) |polarity| {
        for ([_]i64{ 0, 1 }) |tnoi| {
            for ([_]i64{ 0, 1 }) |rds| {
                for ([_]i64{ 0, 1, 2, 3 }) |gate| {
                    for ([_]i64{ 0, 1 }) |body| {
                        var model: D.Model = .{ .l = 0.5e-6, .w = 10e-6, .nf = 2, .rnoib = 1.2, .vtl = 1e5 };
                        cardInt(&model, "typeZ", polarity, true);
                        cardInt(&model, "tnoimod", tnoi, false);
                        cardInt(&model, "rdsmod", rds, false);
                        cardInt(&model, "rgatemod", gate, false);
                        cardInt(&model, "rbodymod", body, false);
                        markGiven(&model, "vtl");
                        D.derive(Real, &model);
                        for ([_][4]f64{ .{ 0.8, 0.9, 0, -0.2 }, .{ 0, 0.9, 0.8, -0.2 }, .{ 1e-8, 0.9, 0, 0 } }) |ports| {
                            var x: [contract.nU(D)]f64 = @splat(0);
                            for (ports, 0..) |v, i| x[i] = v * @as(f64, @floatFromInt(polarity));
                            // Set internal nodes to the chosen intrinsic bias.
                            // They remain distinct derivative seeds in NoiseFamily.
                            for (D.collapse_full, 0..) |alias, i| {
                                if (alias) |a| x[i] = x[a];
                            }
                            var inst: D.Instance = .{};
                            const terms = D.noisePsd(S, x, &model, &inst, .{ .kind = .noise });
                            for (terms) |term| {
                                try std.testing.expect(std.math.isFinite(term.white));
                                try std.testing.expect(term.white >= 0);
                                try std.testing.expect(std.math.isFinite(term.flicker));
                                try std.testing.expect(term.flicker >= 0);
                            }
                        }
                    }
                }
            }
        }
    }
}

test "anyNonzero matches the float compare it replaces, ±0 and NaN included" {
    const specials = [_]f64{ 0.0, -0.0, std.math.nan(f64), -std.math.nan(f64), std.math.inf(f64), -std.math.inf(f64), std.math.floatTrueMin(f64), -std.math.floatTrueMin(f64), 1.0 };
    inline for (.{ 1, 2, 4, 8 }) |w| {
        const V = @Vector(w, f64);
        // Every lane drawn from the specials, lanes mostly ±0 so both answers occur.
        var prng = std.Random.DefaultPrng.init(0xc0de + w);
        for (0..4096) |_| {
            var a: [w]f64 = undefined;
            for (&a) |*x| x.* = if (prng.random().uintLessThan(u8, 4) != 0) specials[prng.random().uintLessThan(usize, 2)] else specials[prng.random().uintLessThan(usize, specials.len)];
            const v: V = a;
            try std.testing.expectEqual(@reduce(.Or, v != @as(V, @splat(0))), impl.test_access.anyNonzero(w, v));
        }
    }
}

test "Dual: expm1 and log1p retain finite range and IEEE endpoints" {
    const S = Dual(1);
    for ([_]f64{ -740, -1, -1e-17, -0.0, 0, 1e-17, 0.5, 704, 709 }) |x| {
        const y = S.probe(0, x).expm1();
        try std.testing.expect(std.math.isFinite(y.v));
        try std.testing.expectApproxEqRel(std.math.expm1(x), y.v, 3e-15);
        try std.testing.expectEqual(@exp(x), y.ddxAt(0));
    }
    for ([_]f64{ -1, -0.9999999999999999, -1e-17, -0.0, 0, 1e-17, 0.5, 1e308, std.math.inf(f64) }) |x| {
        const y = S.probe(0, x).log1p();
        try std.testing.expectApproxEqRel(std.math.log1p(x), y.v, 3e-15);
        try std.testing.expectEqual(1.0 / (1.0 + x), y.ddxAt(0));
    }
    try std.testing.expectEqual(std.math.inf(f64), S.probe(0, 710).expm1().v);
    try std.testing.expectEqual(std.math.inf(f64), S.probe(0, std.math.inf(f64)).expm1().v);
    try std.testing.expectEqual(@as(f64, -1), S.probe(0, -std.math.inf(f64)).expm1().v);
    try std.testing.expect(std.math.isNan(S.probe(0, -2).log1p().v));
    try std.testing.expect(std.math.isNan(S.probe(0, std.math.nan(f64)).expm1().v));
    try std.testing.expect(std.math.isNan(S.probe(0, std.math.nan(f64)).log1p().v));
    try std.testing.expect(std.math.signbit(S.probe(0, -0.0).expm1().v));
    try std.testing.expect(std.math.signbit(S.probe(0, -0.0).log1p().v));
}

test "Dual is the contract's arithmetic, exact and padded" {
    try contract.expectFamily(Dual(2));
    try contract.expectFamily(impl.test_access.Sparse);
    contract.checkFamily(Real);
}

test "Real: every primitive is Dual's value half, bit for bit" {
    // `evalQRange` must reproduce the Dual pass exactly: its charges feed the
    // next step's residual and LTE, so one ulp shifts the timestep sequence.
    const core = struct {
        fn f(comptime S: type, a: f64, b: f64) S.Of(0b11) {
            const x = S.probe(0, a);
            const y = S.probe(1, b);
            // One chain per primitive, so a desync anywhere lands in the result.
            var r = x.add(y).sub(y).neg().mul(x).div(y.addC(3.0)).scale(-0.5);
            // §4.3.1's abs and slew clamps, as generated devices spell them.
            r = S.sel(S.con(0.0).lt(r), r, r.neg());
            r = S.sel(S.con(4.0).lt(r), S.con(4.0).to(0b11), r);
            r = S.sel(r.lt(S.con(-4.0)), S.con(-4.0).to(0b11), r);
            r = r.add(x.exp().log().expm1().log1p());
            r = r.add(y.addC(9.0).sqrt().pow(1.5));
            r = r.add(x.sin().cos().tanh().sinh().cosh().atan());
            r = r.add(S.sel(x.lt(y), S.sel(y.lt(x), x, y), S.sel(x.lt(y), x, y)));
            return r.add(S.sel(x.le(y).add(x.eq(y)), x, y));
        }
    }.f;
    for (0..13) |i| {
        for (0..13) |j| {
            // Crosses zero exactly, where `abs`, min/max ties and `sel` would
            // differ.
            const a = @as(f64, @floatFromInt(i)) * 0.25 - 1.5;
            const b = @as(f64, @floatFromInt(j)) * 0.25 - 1.5;
            try std.testing.expectEqual(
                core(Dual(2), a, b).val(),
                core(Real, a, b).val(),
            );
        }
    }
}

test "dyn vtable: blob init, param set by name, proto add" {
    const R = struct {
        pub const U = enum(u8) { p, n };
        pub const num_ports: usize = 2;
        // Mixed scalar widths exercise the binder's conversion bounds.
        pub const Model = struct {
            r: f32 = 1000,
            mode: i8 = 0,
            mode__given: bool = false,
            count: u64 = 0,
            wide: i64 = 0,
        };
        pub const Instance = struct { temp: f32 = 300.15 };
        pub fn eval(comptime Sc: type, xv: *const [2]Sc.V, m: *const Model, _: *const Instance, _: SimState) contract.Rows(@This(), Sc) {
            const x = contract.probes(@This(), Sc, xv);
            const i = x[0].sub(x[1]).scale(1.0 / @as(f64, m.r));
            return contract.rows(@This(), Sc, .{ i, i.neg() });
        }
    };
    const testing = std.testing;
    const vt = deviceVtable(R, "tres");
    try testing.expectEqual(@as(u32, 2), vt.n_u);
    try testing.expectEqual(@sizeOf(R.Model), vt.model_size);

    var mblob: [@sizeOf(R.Model)]u8 align(16) = undefined;
    vt.init_model(&mblob);
    const Status = @TypeOf(vt.bind_model(&mblob, &.{}));
    const one = struct {
        fn f(bind: anytype, blob: [*]u8, key: []const u8, value: f64) Status {
            return bind(blob, &.{.{ .key = key, .value = value }});
        }
    }.f;
    try testing.expectEqual(.ok, one(vt.bind_model, &mblob, "r", 42));
    try testing.expectEqual(.ok, one(vt.bind_model, &mblob, "bogus", 1)); // unknown keys are ignored
    const m: *R.Model = @ptrCast(@alignCast(&mblob));
    try testing.expectEqual(@as(f32, 42), m.r);

    for ([_]f64{ std.math.nan(f64), std.math.inf(f64), -std.math.inf(f64) }) |invalid|
        try testing.expectEqual(.non_finite_parameter, one(vt.bind_model, &mblob, "r", invalid));
    for ([_]f64{ 1e300, -1e300 }) |invalid|
        try testing.expectEqual(.parameter_out_of_range, one(vt.bind_model, &mblob, "r", invalid));
    try testing.expectEqual(@as(f32, 42), m.r);
    for ([_]f64{ -129, 128, 127.5, -128.6 }) |invalid|
        try testing.expectEqual(.parameter_out_of_range, one(vt.bind_model, &mblob, "mode", invalid));
    try testing.expectEqual(.non_finite_parameter, one(vt.bind_model, &mblob, "mode", std.math.nan(f64)));
    try testing.expectEqual(@as(i8, 0), m.mode);
    try testing.expect(!m.mode__given);
    try testing.expectEqual(.ok, one(vt.bind_model, &mblob, "mode", -128));
    try testing.expectEqual(@as(i8, -128), m.mode);
    try testing.expect(m.mode__given);
    try testing.expectEqual(.ok, one(vt.bind_model, &mblob, "mode", 127));
    try testing.expectEqual(@as(i8, 127), m.mode);
    try testing.expectEqual(.ok, one(vt.bind_model, &mblob, "mode", -0.5));
    try testing.expectEqual(@as(i8, 0), m.mode);
    try testing.expectEqual(.parameter_out_of_range, one(vt.bind_model, &mblob, "count", -1));
    try testing.expectEqual(.parameter_out_of_range, one(vt.bind_model, &mblob, "count", 0x1p64));
    try testing.expectEqual(.ok, one(vt.bind_model, &mblob, "count", 0x1p64 - 2048));
    try testing.expectEqual(@as(u64, 18446744073709549568), m.count);
    try testing.expectEqual(.parameter_out_of_range, one(vt.bind_model, &mblob, "wide", 0x1p63));
    try testing.expectEqual(.parameter_out_of_range, one(vt.bind_model, &mblob, "wide", -0x1p63 - 2048));
    try testing.expectEqual(.ok, one(vt.bind_model, &mblob, "wide", -0x1p63));
    try testing.expectEqual(std.math.minInt(i64), m.wide);

    var iblob: [@sizeOf(R.Instance)]u8 align(16) = undefined;
    vt.init_instance(&iblob);
    try testing.expectEqual(.parameter_out_of_range, one(vt.bind_instance, &iblob, "temp", 1e300));
    const inst: *R.Instance = @ptrCast(@alignCast(&iblob));
    try testing.expectEqual(@as(f32, 300.15), inst.temp);

    const proto = try vt.proto_create(testing.allocator).unwrap();
    const nodes = [2]u32{ 1, 2 };
    try vt.proto_add(proto.ctx, testing.allocator, &mblob, &iblob, &nodes).unwrap();
    const store: *ProtoStore(R) = @ptrCast(@alignCast(proto.ctx));
    try testing.expectEqual(@as(usize, 1), store.nodes.items.len);
    try testing.expectEqual(@as(f32, 42), store.models.items[0].r);
    proto.destroy(proto.ctx, testing.allocator);
}

test "equal cards share one Model row until a parameter is collected" {
    const D = struct {
        pub const U = enum(u8) { p, n };
        pub const num_ports: usize = 2;
        pub const Model = struct { r: f64 = 1000, given: bool = false };
        pub const Instance = struct { temperature: f64 = 300.15, mfactor: f64 = 1 };
        pub fn eval(comptime S: type, xv: *const [2]S.V, m: *const Model, _: *const Instance, _: SimState) contract.Rows(@This(), S) {
            const x = contract.probes(@This(), S, xv);
            const current = x[0].sub(x[1]).scale(1 / m.r);
            return contract.rows(@This(), S, .{ current, current.neg() });
        }
    };
    comptime std.debug.assert(DeviceBatch(D).shares_models);
    const a = std.testing.allocator;
    var proto: ProtoStore(D) = .{};
    for ([_]f64{ 1000, 50, 1000 }, [_][2]u32{ .{ 1, 2 }, .{ 2, 0 }, .{ 1, 0 } }) |r, nodes|
        try proto.append(&.{ .r = r }, &.{}, nodes);
    const batch = try ProtoStore(D).finalize(&proto, a, dense3).unwrap();
    defer batch.hooks.deinit(batch.ctx, a);
    const copy = try batch.hooks.instantiate(batch.ctx, a).unwrap();
    defer copy.hooks.deinit(copy.ctx, a);
    const shared: *DeviceBatch(D) = @ptrCast(@alignCast(copy.ctx));
    try std.testing.expectEqual(@as(usize, 2), shared.models.len);
    try std.testing.expectEqualSlices(u32, &.{ 0, 1, 0 }, shared.model_of);

    // The shared rows stamp what one row per instance stamps.
    var g: [10]f64 = undefined;
    var rhs: [4]f64 = undefined;
    const x = [_]f64{ 0, 2, 0.5 };
    const stamp = struct {
        fn run(b: Batch, gv: *[10]f64, rv: *[4]f64, xs: []const f64) void {
            @memset(gv, 0);
            @memset(rv, 0);
            var c: [10]f64 = @splat(0);
            var q: [4]f64 = @splat(0);
            b.eval(b.ctx, &.{ .g_vals = gv, .c_vals = &c, .rhs = rv, .q_vec = &q }, 0, b.count, xs, 0);
        }
    }.run;
    stamp(copy, &g, &rhs, &x);
    const g_shared = g;
    const rhs_shared = rhs;

    // Collecting parameters gives each instance its own row, so a write
    // through instance 0 leaves instance 2, which shared its card, alone.
    var params: std.ArrayList(impl.test_access.ParamRef) = .empty;
    defer params.deinit(a);
    try copy.hooks.collect_params(copy.ctx, a, &params).unwrap();
    try std.testing.expectEqual(@as(usize, 3), shared.models.len);
    try std.testing.expectEqualSlices(u32, &.{ 0, 1, 2 }, shared.model_of);
    stamp(copy, &g, &rhs, &x);
    try std.testing.expectEqualSlices(f64, &g_shared, &g);
    try std.testing.expectEqualSlices(f64, &rhs_shared, &rhs);
    for (params.items) |ref| if (!ref.is_instance and ref.index == 0) ref.set(10);
    try std.testing.expectEqual(@as(f64, 10), shared.models[0].r);
    try std.testing.expectEqual(@as(f64, 1000), shared.models[2].r);
}

test "write_params copies a shared Model row only for an instance written apart" {
    const D = struct {
        pub const U = enum(u8) { p, n };
        pub const num_ports: usize = 2;
        pub const Model = struct { r: f64 = 1000 };
        pub const Instance = struct { mfactor: f64 = 1 };
        pub fn eval(comptime S: type, xv: *const [2]S.V, m: *const Model, _: *const Instance, _: SimState) contract.Rows(@This(), S) {
            const x = contract.probes(@This(), S, xv);
            const current = x[0].sub(x[1]).scale(1 / m.r);
            return contract.rows(@This(), S, .{ current, current.neg() });
        }
    };
    const a = std.testing.allocator;
    var proto: ProtoStore(D) = .{};
    for ([_]f64{ 1000, 50, 1000 }, [_][2]u32{ .{ 1, 2 }, .{ 2, 0 }, .{ 1, 0 } }) |r, nodes|
        try proto.append(&.{ .r = r }, &.{}, nodes);
    const batch = try ProtoStore(D).finalize(&proto, a, dense3).unwrap();
    defer batch.hooks.deinit(batch.ctx, a);
    // Ordinals: mfactor of instances 0..2, then r of instances 0..2.
    const layout = try batch.hooks.param_layout.?(batch.ctx).unwrap();
    try std.testing.expectEqual(@as(usize, 2), layout.fields.len);
    try std.testing.expectEqual(@as(f64, 1000), layout.ptr(5).*);

    // Instance 0 alone moves: it gets a row of its own, instance 2 keeps
    // the card's.
    {
        const copy = try batch.hooks.instantiate(batch.ctx, a).unwrap();
        defer copy.hooks.deinit(copy.ctx, a);
        const b: *DeviceBatch(D) = @ptrCast(@alignCast(copy.ctx));
        try copy.hooks.write_params.?(copy.ctx, &.{ 1, 3 }, &.{ 2, 10 }, 0, 6).unwrap();
        try std.testing.expectEqual(@as(usize, 3), b.models.len);
        try std.testing.expectEqualSlices(u32, &.{ 2, 1, 0 }, b.model_of);
        try std.testing.expectEqual(@as(f64, 10), b.models[2].r);
        try std.testing.expectEqual(@as(f64, 1000), b.models[0].r);
        try std.testing.expectEqual(@as(f64, 2), b.instances[1].mfactor);
        // A second write to the private row lands in place.
        try copy.hooks.write_params.?(copy.ctx, &.{3}, &.{11}, 0, 6).unwrap();
        try std.testing.expectEqual(@as(usize, 3), b.models.len);
        try std.testing.expectEqual(@as(f64, 11), b.models[2].r);
        // Collecting then gives every instance its own row, values kept.
        var params: std.ArrayList(impl.test_access.ParamRef) = .empty;
        defer params.deinit(a);
        try copy.hooks.collect_params(copy.ctx, a, &params).unwrap();
        try std.testing.expectEqualSlices(u32, &.{ 0, 1, 2 }, b.model_of);
        try std.testing.expectEqual(@as(f64, 11), b.models[0].r);
        try std.testing.expectEqual(@as(f64, 50), b.models[1].r);
        try std.testing.expectEqual(@as(f64, 1000), b.models[2].r);
    }
    // Every reader of the shared row moves alike: the row is written in
    // place.
    {
        const copy = try batch.hooks.instantiate(batch.ctx, a).unwrap();
        defer copy.hooks.deinit(copy.ctx, a);
        const b: *DeviceBatch(D) = @ptrCast(@alignCast(copy.ctx));
        try copy.hooks.write_params.?(copy.ctx, &.{ 3, 5 }, &.{ 7, 7 }, 0, 6).unwrap();
        try std.testing.expectEqual(@as(usize, 2), b.models.len);
        try std.testing.expectEqualSlices(u32, &.{ 0, 1, 0 }, b.model_of);
        try std.testing.expectEqual(@as(f64, 7), b.models[0].r);
    }
    // Isolating instance 2 copies the row it shares with instance 0; a
    // second call finds every named row private and copies nothing.
    {
        const copy = try batch.hooks.instantiate(batch.ctx, a).unwrap();
        defer copy.hooks.deinit(copy.ctx, a);
        const b: *DeviceBatch(D) = @ptrCast(@alignCast(copy.ctx));
        try copy.hooks.isolate_rows.?(copy.ctx, &.{ 1, 2 }).unwrap();
        try std.testing.expectEqualSlices(u32, &.{ 0, 1, 2 }, b.model_of);
        try copy.hooks.isolate_rows.?(copy.ctx, &.{ 0, 2 }).unwrap();
        try std.testing.expectEqual(@as(usize, 3), b.models.len);
        const l = try copy.hooks.param_layout.?(copy.ctx).unwrap();
        l.ptr(5).* = 9;
        try std.testing.expectEqual(@as(f64, 1000), b.models[0].r);
        try std.testing.expectEqual(@as(f64, 9), b.models[2].r);
    }
    // The template is untouched.
    const t: *DeviceBatch(D) = @ptrCast(@alignCast(batch.ctx));
    try std.testing.expectEqual(@as(usize, 2), t.models.len);
    try std.testing.expectEqual(@as(f64, 1000), t.models[0].r);
}

test "collect_noise emits a correlated source's rows contiguously, with coeff, group and table" {
    const D = struct {
        const Gen = struct { row: u8, col: u8, kind: enum { thermal, shot, flicker, table }, source: ?u16 = null, table: ?u16 = null, name: []const u8 = "" };
        pub const noise_tables = [_]contract.NoiseTable{.{ .interp = .linear, .points = &.{ .{ 1, 2 }, .{ 3, 4 } } }};
        // Generators 0 and 2 are one source; 1 is independent; 3 is a table.
        pub const noise_gens = [_]Gen{
            .{ .row = 0, .col = 1, .kind = .thermal, .source = 5, .name = "a" },
            .{ .row = 1, .col = 0, .kind = .shot, .name = "b" },
            .{ .row = 1, .col = 1, .kind = .thermal, .source = 5, .name = "c" },
            .{ .row = 0, .col = 0, .kind = .table, .table = 0, .name = "t" },
        };
        pub fn noisePsd(comptime _: type, _: [2]f64, _: *const Model, _: *const Instance, _: SimState) [4]PsdTerm {
            return .{ .{ .white = 1, .coeff = 2 }, .{ .white = 3 }, .{ .white = 1, .coeff = -2 }, .{ .white = 0, .coeff = 0.5 } };
        }
        pub const U = enum(u8) { p, n };
        pub const num_ports: usize = 2;
        // The card's knots (§4.6.4.3 array parameter), double the defaults.
        pub const Model = struct { r: f64 = 1000, noise_table_points__: [2][2]f64 = .{ .{ 1, 4 }, .{ 3, 8 } } };
        pub fn noiseTablePoints(m: *const Model) [2][2]f64 {
            return m.noise_table_points__;
        }
        pub const Instance = struct {};
        pub fn eval(comptime S: type, xv: *const [2]S.V, m: *const Model, _: *const Instance, _: SimState) contract.Rows(@This(), S) {
            const x = contract.probes(@This(), S, xv);
            const current = x[0].sub(x[1]).scale(1 / m.r);
            return contract.rows(@This(), S, .{ current, current.neg() });
        }
    };
    const a = std.testing.allocator;
    var proto: ProtoStore(D) = .{};
    try proto.append(&.{}, &.{}, .{ 1, 2 });
    try proto.append(&.{}, &.{}, .{ 2, 1 });
    const batch = try ProtoStore(D).finalize(&proto, a, dense3).unwrap();
    defer batch.hooks.deinit(batch.ctx, a);
    try std.testing.expectEqualDeep(&[_][]const u8{ "a", "c", "b", "t" }, batch.hooks.noise_names);
    var noise: std.ArrayList(NoiseSource) = .empty;
    defer noise.deinit(a);
    try noise.append(a, .{ .node_p = 0, .node_n = 0 }); // a row from an earlier batch
    try batch.hooks.collect_noise.?(batch.ctx, &.{ 0, 0, 0 }, a, &noise).unwrap();
    const s = noise.items[1..];
    try std.testing.expectEqual(@as(usize, 8), s.len);
    for (0..2) |id| {
        const r = s[4 * id ..][0..4];
        const base: u32 = @intCast(1 + 4 * id);
        try std.testing.expectEqualSlices(f64, &.{ 2, -2, 1, 0.5 }, &.{ r[0].coeff, r[1].coeff, r[2].coeff, r[3].coeff });
        try std.testing.expectEqualSlices(u32, &.{ base, base, base + 2, base + 3 }, &.{ r[0].group, r[1].group, r[2].group, r[3].group });
        // (u, u) is u to ground; the table row carries the card's knots.
        try std.testing.expectEqual(@as(u32, 0), r[1].node_n);
        try std.testing.expectEqual(@as(f64, 6), r[3].tableAt(2));
        try std.testing.expectEqual(@as(f64, 0), r[0].tableAt(2));
        try std.testing.expectEqual(base + 2, NoiseSource.groupEnd(noise.items, base));
    }
    try std.testing.expectEqual(@as(u32, 1), s[0].node_p);
    try std.testing.expectEqual(@as(u32, 2), s[4].node_p);
}

test "collect_noise puts a generator on a potential branch into its flow row" {
    const D = struct {
        // `V(p,n) <+ R*I(p,n) + white_noise` and `I(p) <+ white_noise`:
        // VerA names both by their nodes. The third is a generator VerA
        // states on the flow row itself (row = col = the flow unknown).
        pub const noise_gens = [_]NoiseGen{
            .{ .row = 0, .col = 1, .kind = .thermal },
            .{ .row = 0, .col = 0, .kind = .thermal },
            .{ .row = 2, .col = 2, .kind = .thermal },
        };
        pub fn noisePsd(comptime _: type, _: [3]f64, _: *const Model, _: *const Instance, _: SimState) [3]PsdTerm {
            return .{ .{ .white = 1 }, .{ .white = 2 }, .{ .white = 3, .coeff = -1 } };
        }
        pub const U = enum(u8) { p, n, flowZ28pZ2cnZ29 };
        pub const num_ports: usize = 2;
        pub const Model = struct { r: f64 = 1000 };
        pub const Instance = struct {};
        pub fn eval(comptime S: type, xv: *const [3]S.V, m: *const Model, _: *const Instance, _: SimState) contract.Rows(@This(), S) {
            const x = contract.probes(@This(), S, xv);
            return contract.rows(@This(), S, .{ x[2], x[2].neg(), x[0].sub(x[1]).sub(x[2].scale(m.r)) });
        }
    };
    const a = std.testing.allocator;
    var proto: ProtoStore(D) = .{};
    try proto.append(&.{}, &.{}, .{ 1, 2, 3 });
    // A switch branch in its flow arm: the flow collapsed onto p.
    try proto.append(&.{}, &.{}, .{ 1, 2, 1 });
    const batch = try ProtoStore(D).finalize(&proto, a, .{
        .col_ptr = &.{ 0, 4, 8, 12, 16 },
        .row_idx = &.{ 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3 },
        .n = 4,
        .trash_slot = 16,
    }).unwrap();
    defer batch.hooks.deinit(batch.ctx, a);
    var noise: std.ArrayList(NoiseSource) = .empty;
    defer noise.deinit(a);
    try batch.hooks.collect_noise.?(batch.ctx, &.{ 0, 0, 0, 0 }, a, &noise).unwrap();
    // The branch generator is −1 at the flow row (ground to row 3), not a
    // current from p to n; the to-ground one has no flow(p,gnd) and stays.
    try std.testing.expectEqual(@as(u32, 0), noise.items[0].node_p);
    try std.testing.expectEqual(@as(u32, 3), noise.items[0].node_n);
    try std.testing.expectEqual(@as(u32, 1), noise.items[1].node_p);
    try std.testing.expectEqual(@as(u32, 0), noise.items[1].node_n);
    // The flow-row form passes through as is, not retargeted twice.
    try std.testing.expectEqual(@as(u32, 3), noise.items[2].node_p);
    try std.testing.expectEqual(@as(u32, 0), noise.items[2].node_n);
    try std.testing.expectEqual(@as(f64, -1), noise.items[2].coeff);
    // Collapsed, the branch generator is the current from p to n again.
    try std.testing.expectEqual(@as(u32, 1), noise.items[3].node_p);
    try std.testing.expectEqual(@as(u32, 2), noise.items[3].node_n);
}

test "seed_nodeset writes each net initializer into its unknown, ground excepted" {
    const D = struct {
        // §3.6.3.2 `electrical p = 2.75;`: only p has an initializer.
        pub const u_nodeset = [2]?f64{ 2.75, null };
        pub const U = enum(u8) { p, n };
        pub const num_ports: usize = 2;
        pub const Model = struct { r: f64 = 1000 };
        pub const Instance = struct {};
        pub fn eval(comptime S: type, xv: *const [2]S.V, m: *const Model, _: *const Instance, _: SimState) contract.Rows(@This(), S) {
            const x = contract.probes(@This(), S, xv);
            const current = x[0].sub(x[1]).scale(1 / m.r);
            return contract.rows(@This(), S, .{ current, current.neg() });
        }
    };
    const a = std.testing.allocator;
    var proto: ProtoStore(D) = .{};
    try proto.append(&.{}, &.{}, .{ 1, 2 });
    try proto.append(&.{}, &.{}, .{ 0, 1 }); // p on ground: nothing written
    const batch = try ProtoStore(D).finalize(&proto, a, dense3).unwrap();
    defer batch.hooks.deinit(batch.ctx, a);
    var x = [_]f64{ -1, -1, -1 };
    batch.hooks.seed_nodeset.?(batch.ctx, &x);
    try std.testing.expectEqualSlices(f64, &.{ -1, 2.75, -1 }, &x);
}

test "prepared device instances share tapes and isolate parameters and accepted history" {
    const D = struct {
        pub const noise_gens = [_]NoiseGen{
            .{ .row = 0, .col = 1, .kind = .shot },
            .{ .row = 1, .col = 0, .kind = .thermal },
        };
        pub fn noisePsd(comptime _: type, x: [2]f64, _: *const Model, _: *const Instance, _: SimState) [2]PsdTerm {
            return .{ .{ .white = @abs(x[0] - x[1]) }, .{ .white = 7 } };
        }
        pub const U = enum(u8) { p, n };
        pub const num_ports: usize = 2;
        pub const Model = struct { r: f64 = 1000 };
        pub const Instance = struct {
            accepted: u8 = 0,
            history: [4]f64 = @splat(0),
        };
        pub const State = struct { bias: f64 = 0 };
        pub fn initState(_: *const Model, _: *const Instance) State {
            return .{};
        }
        pub fn eval(comptime S: type, xv: *const [2]S.V, m: *const Model, _: *const Instance, _: SimState) contract.Rows(@This(), S) {
            const x = contract.probes(@This(), S, xv);
            const current = x[0].sub(x[1]).scale(1 / m.r);
            return contract.rows(@This(), S, .{ current, current.neg() });
        }
    };
    const a = std.testing.allocator;
    var proto: ProtoStore(D) = .{};
    try proto.append(&.{}, &.{}, .{ 1, 2 });
    const batch = try ProtoStore(D).finalize(&proto, a, dense3).unwrap();
    defer batch.hooks.deinit(batch.ctx, a);
    var noise: std.ArrayList(NoiseSource) = .empty;
    defer noise.deinit(a);
    for ([_]f64{ 0, 3, 0 }) |bias| {
        noise.clearRetainingCapacity();
        try batch.hooks.collect_noise.?(batch.ctx, &.{ 0, bias, 0 }, a, &noise).unwrap();
        try std.testing.expectEqual(@as(usize, 2), noise.items.len);
        try std.testing.expectEqual(bias, noise.items[0].white);
        try std.testing.expectEqual(@as(f64, 7), noise.items[1].white);
        try std.testing.expectEqual(@as(u32, 1), noise.items[0].node_p);
        try std.testing.expectEqual(@as(u32, 2), noise.items[1].node_p);
    }
    const first = try batch.hooks.instantiate(batch.ctx, a).unwrap();
    defer first.hooks.deinit(first.ctx, a);
    const second = try batch.hooks.instantiate(batch.ctx, a).unwrap();
    defer second.hooks.deinit(second.ctx, a);
    const template: *DeviceBatch(D) = @ptrCast(@alignCast(batch.ctx));
    const one: *DeviceBatch(D) = @ptrCast(@alignCast(first.ctx));
    const two: *DeviceBatch(D) = @ptrCast(@alignCast(second.ctx));
    one.models[0].r = 25;
    one.instances[0].accepted = 1;
    one.instances[0].history[0] = 2;
    one.states[0].bias = 3;
    try std.testing.expectEqual(@as(f64, 1000), two.models[0].r);
    try std.testing.expectEqual(@as(u8, 0), two.instances[0].accepted);
    try std.testing.expectEqual(@as(f64, 0), two.instances[0].history[0]);
    try std.testing.expectEqual(@as(f64, 0), two.states[0].bias);
    try std.testing.expectEqual(@as(u8, 0), template.instances[0].accepted);
    try std.testing.expectEqual(template.slots.ptr, one.slots.ptr);
    try std.testing.expectEqual(template.gath.ptr, two.gath.ptr);
    const accepted = try first.hooks.snapshot(first.ctx, a).unwrap();
    defer accepted.hooks.deinit(accepted.ctx, a);
    const captured: *DeviceBatch(D) = @ptrCast(@alignCast(accepted.ctx));
    try std.testing.expectEqual(@as(f64, 25), captured.models[0].r);
    try std.testing.expectEqual(@as(u8, 1), captured.instances[0].accepted);
    try std.testing.expectEqual(@as(f64, 2), captured.instances[0].history[0]);
    try std.testing.expectEqual(@as(f64, 3), captured.states[0].bias);
    one.instances[0].history[0] = 99;
    try std.testing.expectEqual(@as(f64, 2), captured.instances[0].history[0]);
    // `copy_state` rewinds in place, so storage a ParamRef points into stays.
    one.states[0].bias = 42;
    const storage = one.instances.ptr;
    first.hooks.copy_state(first.ctx, accepted.ctx);
    try std.testing.expectEqual(@as(f64, 2), one.instances[0].history[0]);
    try std.testing.expectEqual(@as(f64, 3), one.states[0].bias);
    try std.testing.expectEqual(storage, one.instances.ptr);
    try std.testing.checkAllAllocationFailures(a, struct {
        fn run(allocator: std.mem.Allocator, prepared: Batch) !void {
            const instance = try prepared.hooks.instantiate(prepared.ctx, allocator).unwrap();
            defer instance.hooks.deinit(instance.ctx, allocator);
        }
    }.run, .{batch});
}

test "a live timer schedule reaches next_breakpoint" {
    const D = struct {
        pub const U = enum(u8) { p, n };
        pub const num_ports: usize = 2;
        pub const Model = struct {};
        pub const Instance = struct { next: f64 = std.math.inf(f64) };
        pub const State = struct {};
        pub fn initState(_: *const Model, _: *const Instance) State {
            return .{};
        }
        pub fn eval(comptime S: type, xv: *const [2]S.V, _: *const Model, _: *const Instance, _: SimState) contract.Rows(@This(), S) {
            const x = contract.probes(@This(), S, xv);
            const current = x[0].sub(x[1]);
            return contract.rows(@This(), S, .{ current, current.neg() });
        }
        pub fn updateState(comptime _: type, _: *const Model, inst: *Instance, _: [2]f64, _: *State, sim: SimState) contract.UpdateResult {
            // A timer re-armed during the solve: fires 1 ns after each update.
            inst.next = sim.t + 1e-9;
            return .ok;
        }
        pub fn pendingBreakpoint(inst: *const Instance, t: f64) ?f64 {
            return if (inst.next > t) inst.next else null;
        }
    };
    const a = std.testing.allocator;
    var proto: ProtoStore(D) = .{};
    try proto.append(&.{}, &.{}, .{ 1, 2 });
    try proto.append(&.{}, &.{}, .{ 2, 1 });
    const batch = try ProtoStore(D).finalize(&proto, a, dense3).unwrap();
    defer batch.hooks.deinit(batch.ctx, a);
    const next = batch.hooks.next_breakpoint.?;
    try std.testing.expectEqual(@as(?f64, null), next(batch.ctx, 0));
    const self: *DeviceBatch(D) = @ptrCast(@alignCast(batch.ctx));
    self.sim.t = 2e-9;
    _ = batch.hooks.update_state.?(batch.ctx, &.{ 0, 0, 0 });
    const fire = self.sim.t + 1e-9;
    try std.testing.expectEqual(@as(?f64, fire), next(batch.ctx, 2e-9));
    try std.testing.expectEqual(@as(?f64, null), next(batch.ctx, fire));
}

test "a field that only names absdelay keeps updateState on every solve" {
    const D = struct {
        pub const U = enum(u8) { p, n };
        pub const num_ports: usize = 2;
        pub const Model = struct {};
        // Named like a delay ring, but outside VerA's operator namespace
        // (`__analog_op__absdelay__`): routed to `commit_state`, the
        // operating point would never run updateState.
        pub const Instance = struct { sig__absdelay__x: f64 = 0 };
        pub const State = struct {};
        pub fn initState(_: *const Model, _: *const Instance) State {
            return .{};
        }
        pub fn eval(comptime S: type, xv: *const [2]S.V, _: *const Model, _: *const Instance, _: SimState) contract.Rows(@This(), S) {
            const x = contract.probes(@This(), S, xv);
            const current = x[0].sub(x[1]);
            return contract.rows(@This(), S, .{ current, current.neg() });
        }
        pub fn updateState(comptime _: type, _: *const Model, _: *Instance, _: [2]f64, _: *State, _: SimState) contract.UpdateResult {
            return .ok;
        }
    };
    try std.testing.expect(DeviceBatch(D).hooks.update_state != null);
    try std.testing.expect(DeviceBatch(D).hooks.commit_state == null);
}

test "iteration hooks gather each instance and preserve accepted-time state" {
    const D = struct {
        pub const U = enum(u8) { p, n };
        pub const num_ports: usize = 2;
        pub const Model = struct {};
        pub const Instance = struct {
            previous: f64 = 0,
            accepted_time: f64 = 17,
        };
        pub fn eval(comptime S: type, xv: *const [2]S.V, _: *const Model, _: *const Instance, _: SimState) contract.Rows(@This(), S) {
            const x = contract.probes(@This(), S, xv);
            const current = x[0].sub(x[1]);
            return contract.rows(@This(), S, .{ current, current.neg() });
        }
        pub fn advanceIteration(comptime _: type, _: *const Model, inst: *Instance, x: [2]f64, _: SimState) void {
            inst.previous = x[0] - x[1];
        }
        pub fn checkConvergence(comptime _: type, _: *const Model, inst: *const Instance, x: [2]f64, sim: SimState) bool {
            return sim.iteration > 1 and inst.previous == x[0] - x[1];
        }
    };
    const a = std.testing.allocator;
    var proto: ProtoStore(D) = .{};
    try proto.append(&.{}, &.{}, .{ 1, 2 });
    try proto.append(&.{}, &.{}, .{ 2, 1 });
    const batch = try ProtoStore(D).finalize(&proto, a, dense3).unwrap();
    defer batch.hooks.deinit(batch.ctx, a);
    const typed: *DeviceBatch(D) = @ptrCast(@alignCast(batch.ctx));
    try std.testing.expect(batch.hooks.gpu_payload == null);
    const x = [_]f64{ 0, 5, 2 };
    // The host counts iterations into the SimState it publishes.
    batch.hooks.set_sim_state(batch.ctx, .{ .iteration = 1 });
    try std.testing.expect(!batch.hooks.check_convergence.?(batch.ctx, &x));
    batch.hooks.advance_iteration.?(batch.ctx, &x);
    batch.hooks.set_sim_state(batch.ctx, .{ .iteration = 2 });
    try std.testing.expect(batch.hooks.check_convergence.?(batch.ctx, &x));
    try std.testing.expectEqual(@as(f64, 3), typed.instances[0].previous);
    try std.testing.expectEqual(@as(f64, -3), typed.instances[1].previous);
    for (typed.instances) |inst| try std.testing.expectEqual(@as(f64, 17), inst.accepted_time);
    batch.hooks.set_sim_state(batch.ctx, .{ .iteration = 1 });
    try std.testing.expect(!batch.hooks.check_convergence.?(batch.ctx, &x));
    try std.testing.expectEqual(@as(f64, 3), typed.instances[0].previous);
}

test "set_homotopy writes gmin__ and source_scale__ on every row and reruns setup" {
    const D = struct {
        pub const U = enum(u8) { p, n };
        pub const num_ports: usize = 2;
        pub const Setup = struct { g: f64 = 0 };
        pub const Model = struct { gmin__: f64 = 1e-12, source_scale__: f64 = 1, r: f64 = 1, su: Setup = .{} };
        pub const Instance = struct {};
        pub const setup_simparams = [_][]const u8{"gmin"};
        pub fn setup(comptime _: type, m: *Model) void {
            m.su.g = m.gmin__ * m.r;
        }
        pub fn eval(comptime S: type, xv: *const [2]S.V, m: *const Model, _: *const Instance, _: SimState) contract.Rows(@This(), S) {
            const x = contract.probes(@This(), S, xv);
            const current = x[0].sub(x[1]).scale(m.su.g + m.source_scale__);
            return contract.rows(@This(), S, .{ current, current.neg() });
        }
    };
    const a = std.testing.allocator;
    var proto: ProtoStore(D) = .{};
    try proto.append(&.{ .r = 1 }, &.{}, .{ 1, 2 });
    try proto.append(&.{ .r = 3 }, &.{}, .{ 2, 1 });
    const batch = try ProtoStore(D).finalize(&proto, a, dense3).unwrap();
    defer batch.hooks.deinit(batch.ctx, a);
    const typed: *DeviceBatch(D) = @ptrCast(@alignCast(batch.ctx));
    batch.hooks.set_homotopy.?(batch.ctx, 1e-3, 0.25);
    try std.testing.expectEqual(@as(usize, 2), typed.models.len);
    for (typed.models) |m| {
        try std.testing.expectEqual(@as(f64, 1e-3), m.gmin__);
        try std.testing.expectEqual(@as(f64, 0.25), m.source_scale__);
        try std.testing.expectEqual(1e-3 * m.r, m.su.g);
    }
    // A model that reads neither knob costs the host nothing.
    const Plain = struct {
        pub const U = enum(u8) { p, n };
        pub const num_ports: usize = 2;
        pub const Model = struct {};
        pub const Instance = struct {};
        pub fn eval(comptime S: type, xv: *const [2]S.V, _: *const Model, _: *const Instance, _: SimState) contract.Rows(@This(), S) {
            const x = contract.probes(@This(), S, xv);
            return contract.rows(@This(), S, .{ x[0], x[0].neg() });
        }
    };
    try std.testing.expect(DeviceBatch(Plain).hooks.set_homotopy == null);
}

test "a status-latch-only device stays GPU-eligible and reports its latched status" {
    const D = struct {
        pub const mutable_eval = true;
        pub const vera_version = contract.vera_version;
        pub const U = enum(u8) { p, n };
        pub const num_ports: usize = 2;
        pub const Model = struct { limit: f64 = 4 };
        pub const Instance = struct {
            vera_status__: u32 = 0,
            vera_status_args__: [4]f64 = @splat(0.0),
        };
        pub const status_sites = [_]contract.StatusSite{.{ .severity = .@"error", .fmt = "v = %g", .file = "m.va", .line = 7 }};
        pub fn eval(comptime S: type, xv: *const [2]S.V, m: *const Model, inst: *Instance, _: SimState) contract.Rows(@This(), S) {
            const x = contract.probes(@This(), S, xv);
            const v = x[0].sub(x[1]);
            if (inst.vera_status__ == 0 and v.val() > m.limit) {
                inst.vera_status__ = contract.statusCode(.@"error", 0);
                inst.vera_status_args__[0] = v.val();
            }
            return contract.rows(@This(), S, .{ v, v.neg() });
        }
    };
    const a = std.testing.allocator;
    var proto: ProtoStore(D) = .{};
    try proto.append(&.{}, &.{}, .{ 1, 2 });
    try proto.append(&.{}, &.{}, .{ 2, 1 });
    const batch = try ProtoStore(D).finalize(&proto, a, dense3).unwrap();
    defer batch.hooks.deinit(batch.ctx, a);
    // The latch is its only eval write, so the device may be resident.
    try std.testing.expect(batch.hooks.gpu_payload != null);
    var msg: [64]u8 = undefined;
    try std.testing.expectEqual(null, batch.hooks.status.?(batch.ctx, &msg));
    var g: [10]f64 = @splat(0);
    var c: [10]f64 = @splat(0);
    var rhs: [4]f64 = @splat(0);
    var q: [4]f64 = @splat(0);
    const planes: impl.Planes = .{ .g_vals = &g, .c_vals = &c, .rhs = &rhs, .q_vec = &q };
    // v(1) - v(2) = -5 for instance 0 and +5 for instance 1.
    batch.eval(batch.ctx, &planes, 0, 2, &.{ 0, 0, 5 }, 0);
    const hit = batch.hooks.status.?(batch.ctx, &msg).?;
    try std.testing.expectEqual(@as(u32, 1), hit.index);
    try std.testing.expectEqualStrings("m.va:7: error: v = 5", msg[0..hit.len]);
}

test "mutable evaluation captures per-instance data without GPU residency" {
    const D = struct {
        pub const mutable_eval = true;
        pub const vera_version = contract.vera_version;
        pub const U = enum(u8) { p, n };
        pub const num_ports: usize = 2;
        pub const Model = struct {};
        pub const Instance = struct {
            ready: bool = false,
            first_value: f64 = 0,
        };
        pub fn eval(comptime S: type, xv: *const [2]S.V, _: *const Model, inst: *Instance, _: SimState) contract.Rows(@This(), S) {
            const x = contract.probes(@This(), S, xv);
            const v = x[0].sub(x[1]);
            if (!inst.ready) {
                inst.first_value = v.val();
                inst.ready = true;
            }
            const current = v.sub(S.con(inst.first_value));
            return contract.rows(@This(), S, .{ current, current.neg() });
        }
    };
    comptime impl.checkHost(D);
    const a = std.testing.allocator;
    var proto: ProtoStore(D) = .{};
    try proto.append(&.{}, &.{}, .{ 1, 2 });
    try proto.append(&.{}, &.{}, .{ 2, 1 });
    const batch = try ProtoStore(D).finalize(&proto, a, dense3).unwrap();
    defer batch.hooks.deinit(batch.ctx, a);
    const typed: *DeviceBatch(D) = @ptrCast(@alignCast(batch.ctx));
    try std.testing.expect(batch.hooks.gpu_payload == null);
    for (typed.instances) |inst| try std.testing.expect(!inst.ready);
    var g: [10]f64 = @splat(0);
    var c: [10]f64 = @splat(0);
    var rhs: [4]f64 = @splat(0);
    var q: [4]f64 = @splat(0);
    const planes: impl.Planes = .{ .g_vals = &g, .c_vals = &c, .rhs = &rhs, .q_vec = &q };
    batch.eval(batch.ctx, &planes, 0, 2, &.{ 0, 5, 2 }, 0);
    try std.testing.expectEqual(@as(f64, 3), typed.instances[0].first_value);
    try std.testing.expectEqual(@as(f64, -3), typed.instances[1].first_value);
    batch.eval(batch.ctx, &planes, 0, 2, &.{ 0, 9, 1 }, 1);
    try std.testing.expectEqual(@as(f64, 3), typed.instances[0].first_value);
    try std.testing.expectEqual(@as(f64, -3), typed.instances[1].first_value);
}

test "systf: the vtable names every $systf call and binds the application per instance; such a device stays off the GPU" {
    // #197: before ABI 30 the host could neither see these calls nor bind
    // them, and a release build evaluated `inst.systf.?` on a null binding
    // (validateHost checks it only in Debug).
    const D = struct {
        pub const vera_version = contract.vera_version;
        pub const U = enum(u8) { p, n };
        pub const num_ports: usize = 2;
        pub const systf_calls = [_]contract.Systf{ .{ .name = "$probe" }, .{ .name = "$note", .tok = 7 } };
        pub const Model = struct {};
        pub const Instance = struct {
            systf: ?*const contract.SystfHost = null,
        };
        pub fn eval(comptime S: type, xv: *const [2]S.V, _: *const Model, _: *const Instance, _: SimState) contract.Rows(@This(), S) {
            const x = contract.probes(@This(), S, xv);
            const v = x[0].sub(x[1]);
            return contract.rows(@This(), S, .{ v, v.neg() });
        }
    };
    comptime impl.checkHost(D);
    const vt = deviceVtable(D, "m");
    try std.testing.expectEqual(@as(usize, 2), vt.systf_calls.len);
    try std.testing.expectEqualStrings("$probe", vt.systf_calls[0].name);
    try std.testing.expectEqualStrings("$note", vt.systf_calls[1].name);
    try std.testing.expectEqual(@as(u32, 7), vt.systf_calls[1].tok);
    try std.testing.expectEqual(impl.test_access.GpuRefusal.systf, comptime impl.test_access.gpuRefusal(D).?);
    // `bind_systf` writes the application into the blob `proto_add` copies.
    const Echo = struct {
        fn call(_: *anyopaque, _: usize, _: []const f64, partials: []f64) f64 {
            @memset(partials, 0);
            return 4;
        }
    };
    const app: contract.SystfHost = .{ .ctx = undefined, .call = Echo.call };
    var iblob: D.Instance align(16) = .{};
    vt.init_instance(@ptrCast(&iblob));
    try std.testing.expect(iblob.systf == null);
    vt.bind_systf.?(@ptrCast(&iblob), &app);
    try std.testing.expect(iblob.systf.? == &app);
    // A device that calls none lists none and binds nothing.
    const R = struct {
        pub const vera_version = contract.vera_version;
        pub const U = enum(u8) { p, n };
        pub const num_ports: usize = 2;
        pub const Model = struct {};
        pub const Instance = struct {};
        pub fn eval(comptime S: type, xv: *const [2]S.V, _: *const Model, _: *const Instance, _: SimState) contract.Rows(@This(), S) {
            const x = contract.probes(@This(), S, xv);
            const v = x[0].sub(x[1]);
            return contract.rows(@This(), S, .{ v, v.neg() });
        }
    };
    try std.testing.expectEqual(@as(usize, 0), deviceVtable(R, "r").systf_calls.len);
    try std.testing.expect(deviceVtable(R, "r").bind_systf == null);
}

test "permuteInPlace: moves every row to dst[k], cycles and fixed points alike" {
    var prng = std.Random.DefaultPrng.init(0x9e37);
    const r = prng.random();
    for (1..40) |len| {
        var dst: [40]u32 = undefined;
        for (dst[0..len], 0..) |*d, i| d.* = @intCast(i);
        r.shuffle(u32, dst[0..len]);
        var items: [40]u64 = undefined;
        for (items[0..len], 0..) |*v, i| v.* = i * 7 + 1;
        var want: [40]u64 = undefined;
        for (0..len) |k| want[dst[k]] = items[k];
        var seen: [40]bool = undefined;
        impl.permuteInPlace(u64, items[0..len], dst[0..len], seen[0..len]);
        try std.testing.expectEqualSlices(u64, want[0..len], items[0..len]);
    }
}

test "permuteInPlace: a permutation then its inverse restores every row (#142)" {
    // dst must be a permutation of 0..len; a repeated or out-of-range target
    // is a programmer error the function asserts on, so only the valid side
    // is testable here. The expectation is the identity: inv[dst[k]] = k
    // undoes the move of row k, whatever the cycle structure.
    var prng = std.Random.DefaultPrng.init(0x142);
    const r = prng.random();
    for (0..40) |len| {
        var dst: [40]u32 = undefined;
        for (dst[0..len], 0..) |*d, i| d.* = @intCast(i);
        r.shuffle(u32, dst[0..len]);
        var inv: [40]u32 = undefined;
        for (dst[0..len], 0..) |d, k| inv[d] = @intCast(k);
        var items: [40]u64 = undefined;
        for (items[0..len], 0..) |*v, i| v.* = i * 7 + 1;
        var seen: [40]bool = undefined;
        impl.permuteInPlace(u64, items[0..len], dst[0..len], seen[0..len]);
        impl.permuteInPlace(u64, items[0..len], inv[0..len], seen[0..len]);
        for (items[0..len], 0..) |v, i| try std.testing.expectEqual(@as(u64, i * 7 + 1), v);
    }
}

test "analog initial reruns after a revert rolls its held results back" {
    // Stands in for a VerA device: `updateState` holds what the §5.2.1 block
    // computes, `stateCtl` commits and reverts it like any held value.
    const D = struct {
        pub const U = enum(u8) { p, n };
        pub const num_ports: usize = 2;
        pub const Model = struct {};
        pub const Instance = struct { held: f64 = 0 };
        pub const State = struct { committed: f64 = 0 };
        pub fn initState(_: *const Model, _: *const Instance) State {
            return .{};
        }
        pub fn eval(comptime S: type, xv: *const [2]S.V, _: *const Model, _: *const Instance, _: SimState) contract.Rows(@This(), S) {
            const x = contract.probes(@This(), S, xv);
            const current = x[0].sub(x[1]);
            return contract.rows(@This(), S, .{ current, current.neg() });
        }
        pub fn updateState(comptime _: type, _: *const Model, inst: *Instance, _: [2]f64, _: *State, sim: SimState) contract.UpdateResult {
            if (sim.analog_initial) inst.held = 5;
            return .ok;
        }
        pub fn stateCtl(_: *const Model, inst: *Instance, st: *State, op: contract.StateCtlOp) bool {
            switch (op) {
                .commit => st.committed = inst.held,
                .revert => inst.held = st.committed,
                .query => {},
            }
            return false;
        }
    };
    const a = std.testing.allocator;
    var proto: ProtoStore(D) = .{};
    try proto.append(&.{}, &.{}, .{ 1, 2 });
    const template = try ProtoStore(D).finalize(&proto, a, dense3).unwrap();
    defer template.hooks.deinit(template.ctx, a);
    const batch = try template.hooks.instantiate(template.ctx, a).unwrap();
    defer batch.hooks.deinit(batch.ctx, a);
    const typed: *DeviceBatch(D) = @ptrCast(@alignCast(batch.ctx));
    const x = [_]f64{ 0, 1, 0 };
    const ctl = batch.hooks.state_ctl.?;
    const update = batch.hooks.update_state.?;

    _ = update(batch.ctx, &x);
    try std.testing.expectEqual(@as(f64, 5), typed.instances[0].held);
    try std.testing.expect(!typed.sim.analog_initial);
    // Back to the birth commit: the held result is gone, so the block reruns.
    _ = ctl(batch.ctx, .revert);
    try std.testing.expectEqual(@as(f64, 0), typed.instances[0].held);
    try std.testing.expect(typed.sim.analog_initial);
    batch.hooks.set_sim_state(batch.ctx, .{});
    try std.testing.expect(typed.sim.analog_initial);
    _ = update(batch.ctx, &x);
    try std.testing.expectEqual(@as(f64, 5), typed.instances[0].held);
    // Committed, a revert keeps the result and the block stays skipped.
    _ = ctl(batch.ctx, .commit);
    _ = ctl(batch.ctx, .revert);
    try std.testing.expectEqual(@as(f64, 5), typed.instances[0].held);
    try std.testing.expect(!typed.sim.analog_initial);
    batch.hooks.set_sim_state(batch.ctx, .{});
    try std.testing.expect(!typed.sim.analog_initial);
    // A fresh instance starts over; copy_state carries the flag with the values.
    const fresh = try template.hooks.instantiate(template.ctx, a).unwrap();
    defer fresh.hooks.deinit(fresh.ctx, a);
    const fresh_typed: *DeviceBatch(D) = @ptrCast(@alignCast(fresh.ctx));
    try std.testing.expect(fresh_typed.sim.analog_initial);
    fresh.hooks.copy_state(fresh.ctx, batch.ctx);
    try std.testing.expect(!fresh_typed.sim.analog_initial);
}

test "a held-variable device that limits is resident, its state half deferred to the converged solve" {
    // vbic13_4t and bsim4va: §5.10 held variables plus `$limit`.
    const D = struct {
        pub const U = enum(u8) { p, n };
        pub const num_ports: usize = 2;
        pub const Model = struct {};
        pub const Instance = struct { v__held__: f64 = 0 };
        pub const State = struct { committed: f64 = 0 };
        pub fn initState(_: *const Model, _: *const Instance) State {
            return .{};
        }
        pub fn eval(comptime S: type, xv: *const [2]S.V, _: *const Model, _: *const Instance, _: SimState) contract.Rows(@This(), S) {
            const x = contract.probes(@This(), S, xv);
            const current = x[0].sub(x[1]);
            return contract.rows(@This(), S, .{ current, current.neg() });
        }
        pub fn limit(comptime _: type, _: *const Model, _: *const Instance, cur: [2]f64, old: [2]f64, _: SimState) contract.LimitResult(2) {
            return .{ .x = .{ @min(cur[0], old[0] + 1), cur[1] }, .converged = cur[0] <= old[0] + 1 };
        }
        pub fn updateState(comptime _: type, _: *const Model, inst: *Instance, x: [2]f64, _: *State, _: SimState) contract.UpdateResult {
            inst.v__held__ = x[0] - x[1];
            return .ok;
        }
        pub fn stateCtl(_: *const Model, inst: *Instance, st: *State, op: contract.StateCtlOp) bool {
            switch (op) {
                .commit => st.committed = inst.v__held__,
                .revert => inst.v__held__ = st.committed,
                .query => {},
            }
            return false;
        }
    };
    const a = std.testing.allocator;
    var proto: ProtoStore(D) = .{};
    try proto.append(&.{}, &.{}, .{ 1, 2 });
    const batch = try ProtoStore(D).finalize(&proto, a, dense3).unwrap();
    defer batch.hooks.deinit(batch.ctx, a);
    const p = batch.hooks.gpu_payload.?(batch.ctx);
    try std.testing.expect(p.lim_kernel.len > 0);
    try std.testing.expect(p.lim_x.len > 0);
}

test "spicePwlBreak rounds from the request time as ngspice does" {
    const spicePwlBreak = impl.test_access.spicePwlBreak;
    const M = struct {
        waveform: i64 = 4,
        pwl_len: i64 = 3,
        pwl_repeat: f64 = -1,
        pwl_td: f64 = 0,
        pwl_timesZ5b0Z5d: f64 = 0,
        pwl_timesZ5b1Z5d: f64 = 1.0e-9,
        pwl_timesZ5b2Z5d: f64 = 3.1e-9,
        pwl_timesZ5b3Z5d: f64 = 0,
    };
    const mb = 1e-20;
    var m: M = .{};
    try std.testing.expectEqual(@as(?f64, 1.0e-9), spicePwlBreak(&m, 0, mb));
    // `t_acc + (t_k - t_acc)`, not the table time itself.
    const t_acc = 0.7e-9;
    try std.testing.expectEqual(@as(?f64, t_acc + (3.1e-9 - t_acc)), spicePwlBreak(&m, t_acc, 1.0e-9 + mb));
    try std.testing.expectEqual(@as(?f64, std.math.inf(f64)), spicePwlBreak(&m, 3.1e-9, 3.1e-9 + mb));
    // TD delays every corner; the accepted time rounds it.
    m.pwl_td = 0.3e-9;
    try std.testing.expectEqual(@as(?f64, 0.3e-9), spicePwlBreak(&m, 0, mb));
    try std.testing.expectEqual(@as(?f64, 0.3e-9 + 1.0e-9), spicePwlBreak(&m, 0.3e-9, 0.3e-9 + mb));
    // `r=1n`: past the end the table replays [1n, 3.1n], period 2.1n.
    m = .{ .pwl_repeat = 1.0e-9 };
    const bp = spicePwlBreak(&m, 3.1e-9, 3.1e-9 + mb).?;
    try std.testing.expectApproxEqRel(@as(f64, 5.2e-9), bp, 1e-12);
    try std.testing.expectApproxEqRel(@as(f64, 7.3e-9), spicePwlBreak(&m, bp, bp + mb).?, 1e-12);
    // `r=0` repeats the whole table (period 3.1n); a negative `r` is no repeat.
    m = .{ .pwl_repeat = 0 };
    try std.testing.expectApproxEqRel(@as(f64, 4.1e-9), spicePwlBreak(&m, 3.1e-9, 3.1e-9 + mb).?, 1e-12);
    // An empty table or another waveform: the caller falls back.
    m = .{ .pwl_len = 0 };
    try std.testing.expectEqual(@as(?f64, null), spicePwlBreak(&m, 0, mb));
    m = .{ .waveform = 1 };
    try std.testing.expectEqual(@as(?f64, null), spicePwlBreak(&m, 0, mb));
}

test "spicePulseBreak rounds from the request time as ngspice does" {
    const spicePulseBreak = impl.test_access.spicePulseBreak;
    // tran/bench_tline_txl1_1_line's VS: PULSE(0 5 15.9n 0.2n 0.2n 15.8n 32n).
    // Typed f64 fields, as on vsource's Model: comptime_float fields would
    // fold the corner sums at f128 and round them once.
    const P = struct {
        waveform: i64 = 1,
        pulse_phase: f64 = 0,
        pulse_td: f64 = 15.9e-9,
        pulse_tr: f64 = 0.2e-9,
        pulse_tf: f64 = 0.2e-9,
        pulse_pw: f64 = 15.8e-9,
        pulse_per: f64 = 32e-9,
    };
    var m: P = .{};
    const mb = 1e-20;
    try std.testing.expectEqual(@as(?f64, 15.9e-9), spicePulseBreak(m, 0, mb));
    try std.testing.expectEqual(@as(?f64, 1.61e-8), spicePulseBreak(m, 15.9e-9, 15.9e-9 + mb));
    // The fall's end: ngspice lands one ulp below td+tr+pw+tf = 3.21e-8,
    // at 32099.999... ps, which txlload.c truncates to 32099.
    try std.testing.expectEqual(@as(?f64, 3.2099999999999996e-08), spicePulseBreak(m, 3.19e-8, 3.19e-8 + mb));
    try std.testing.expectEqual(@as(?f64, 4.79e-8), spicePulseBreak(m, 3.2099999999999996e-08, 3.2099999999999996e-08 + mb));
    // An anchor ngspice would not ask at yields nothing (caller falls back).
    try std.testing.expectEqual(@as(?f64, null), spicePulseBreak(m, 0, 60e-9));
    // A nonzero PHASE and any other waveform are left to the timers.
    m.pulse_phase = 90;
    try std.testing.expectEqual(@as(?f64, null), spicePulseBreak(m, 0, mb));
    m = .{ .waveform = 4 };
    try std.testing.expectEqual(@as(?f64, null), spicePulseBreak(m, 0, mb));
}

test "Dual: every partial matches a central difference of the value" {
    // Independent of the contract's reference family: a wrong lane rule
    // there and here would still disagree with the difference quotient.
    const S = Dual(2);
    const core = struct {
        fn f(a: f64, b: f64) S.Of(0b11) {
            const x = S.probe(0, a);
            const y = S.probe(1, b);
            return x.mul(y).add(x.exp().scale(0.3)).sub(y.div(x.addC(2.5)))
                .add(y.mul(y).addC(1.0).sqrt().pow(1.5))
                .add(x.sin().mul(y.cos())).add(x.tanh().atan())
                .add(x.mul(x).addC(1.0).log()).add(y.sinh().scale(0.1))
                .add(x.cosh().scale(0.05)).add(x.expm1().scale(0.2))
                .add(y.mul(y).log1p()).neg();
        }
    }.f;
    for (0..9) |i| for (0..9) |j| {
        const a = @as(f64, @floatFromInt(i)) * 0.3 - 1.2;
        const b = @as(f64, @floatFromInt(j)) * 0.3 - 1.2;
        const r = core(a, b);
        const h = 1e-6;
        const fd = [2]f64{
            (core(a + h, b).val() - core(a - h, b).val()) / (2 * h),
            (core(a, b + h).val() - core(a, b - h).val()) / (2 * h),
        };
        inline for (0..2) |u| {
            const d = r.ddxAt(u);
            try std.testing.expect(@abs(d - fd[u]) <= 1e-6 * @max(1.0, @abs(d)));
        }
    };
}

test "Dual: the padded layout computes the exact layout's numbers" {
    // Three lanes pad to four, and the operands below carry {0,2}, {1,2}
    // and their join, so every spread shuffle and the pad lane are crossed.
    const chain = struct {
        fn f(comptime S: type, a: f64, b: f64, c: f64) S.Of(0b111) {
            const x = S.probe(0, a);
            const y = S.probe(1, b);
            const z = S.probe(2, c);
            const xz = x.mul(z).add(x.exp());
            const yz = y.div(z.addC(3.0)).sub(y.pow(2.0));
            const s = S.sel(x.lt(y), xz, yz);
            return s.mul(xz).add(yz.mul(yz).addC(1.0).sqrt()).add(x.mul(S.con(2.5))).to(0b111);
        }
    }.f;
    const Sparse3 = impl.test_access.Sparse3;
    const grid = [_]f64{ -1.0, -0.5, 0.0, 0.5, 1.0 };
    for (grid) |a| for (grid) |b| for (grid) |c| {
        const exact = chain(Dual(3), a, b, c);
        const padded = chain(Sparse3, a, b, c);
        try std.testing.expectEqual(exact.val(), padded.val());
        inline for (0..3) |u| try std.testing.expectEqual(exact.ddxAt(u), padded.ddxAt(u));
    };
}

test "segmentSum: the eight-wide loads keep the plain loop's order, bit for bit" {
    var prng = std.Random.DefaultPrng.init(0x5e65);
    const r = prng.random();
    var stage: [64]f64 = undefined;
    // Mixed signs and magnitudes, so any reassociation changes the bits.
    for (&stage) |*v| v.* = (r.float(f64) - 0.5) * std.math.pow(f64, 10, @floatFromInt(r.intRangeAtMost(i32, -8, 8)));
    for (0..4) |off| for (0..3 * 8 + 2) |n| {
        const first: u32 = @intCast(off);
        const end: u32 = @intCast(off + n);
        var want: f64 = 0;
        for (stage[first..end]) |v| want += v;
        const got = impl.test_access.segmentSum(&stage, first, end);
        try std.testing.expectEqual(@as(u64, @bitCast(want)), @as(u64, @bitCast(got)));
    };
}

test "repMask: one column per lane, an alias before its root, unlaned columns dropped" {
    const repMask = impl.test_access.repMask;
    const nl = contract.no_lane;
    // Columns 0 and 1 share lane 0; column 1 is the alias, so it stamps.
    try std.testing.expectEqual([3]u64{ 0b110, 0b010, 0 }, comptime repMask(3, .{ 0, 0, 1 }, .{ false, true, false }, .{ 0b111, 0b011, 0 }));
    // No alias: the first column of the lane wins (the identity on the wide basis).
    try std.testing.expectEqual([3]u64{ 0b101, 0b001, 0 }, comptime repMask(3, .{ 0, 0, 1 }, .{ false, false, false }, .{ 0b111, 0b011, 0 }));
    // A column without a lane takes its partial from `jac_const`, not a stamp.
    try std.testing.expectEqual([3]u64{ 0b101, 0b100, 0 }, comptime repMask(3, .{ 0, nl, 1 }, .{ false, false, false }, .{ 0b111, 0b110, 0 }));
}

/// A two-terminal conductance, the smallest device a batch accepts.
const Conductance = struct {
    pub const U = enum(u8) { p, n };
    pub const num_ports: usize = 2;
    pub const Model = struct { g: f64 = 1e-3 };
    pub const Instance = struct {};
    pub fn eval(comptime S: type, xv: *const [2]S.V, m: *const Model, _: *const Instance, _: SimState) contract.Rows(@This(), S) {
        const x = contract.probes(@This(), S, xv);
        const i = x[0].sub(x[1]).scale(m.g);
        return contract.rows(@This(), S, .{ i, i.neg() });
    }
};

test "ac_dyn: every frequency once, ragged tails included" {
    const D = struct {
        pub const U = Conductance.U;
        pub const num_ports = Conductance.num_ports;
        pub const Model = Conductance.Model;
        pub const Instance = Conductance.Instance;
        pub fn eval(comptime S: type, xv: *const [2]S.V, m: *const Model, _: *const Instance, _: SimState) contract.Rows(@This(), S) {
            const x = contract.probes(@This(), S, xv);
            const i = x[0].sub(x[1]).scale(m.g);
            return contract.rows(@This(), S, .{ i, i.neg() });
        }
        pub const ac_dyn_slots = [_]usize{0};
        pub fn acDyn(comptime V: type, _: *const Model, _: *const Instance, lx: *const [2]f64, _: SimState, ow: anytype, out: anytype) void {
            const w: V = ow;
            out[0] = .{ .re = w + @as(V, @splat(lx[0])), .im = -w };
        }
    };
    const a = std.testing.allocator;
    var proto: ProtoStore(D) = .{};
    try proto.append(&.{}, &.{}, .{ 1, 2 });
    try proto.append(&.{}, &.{}, .{ 2, 1 });
    const batch = try ProtoStore(D).finalize(&proto, a, dense3).unwrap();
    defer batch.hooks.deinit(batch.ctx, a);
    const lw = std.simd.suggestVectorLength(f64) orelse 1;
    const x = [_]f64{ 0, 10, 20 };
    var omegas: [3 * lw + 1]f64 = undefined;
    for (&omegas, 0..) |*w, i| w.* = @floatFromInt(i + 1);
    var re: [2 * omegas.len]f64 = undefined;
    var im: [2 * omegas.len]f64 = undefined;
    for (0..omegas.len + 1) |nw| {
        @memset(&re, std.math.nan(f64));
        @memset(&im, std.math.nan(f64));
        try std.testing.expectEqual(@as(usize, 2), batch.hooks.ac_dyn.?(batch.ctx, &x, omegas[0..nw], re[0 .. 2 * nw], im[0 .. 2 * nw]));
        for (0..2) |id| for (0..nw) |i| {
            try std.testing.expectEqual(omegas[i] + x[1 + id], re[id * nw + i]);
            try std.testing.expectEqual(-omegas[i], im[id * nw + i]);
        };
        // Nothing past the entries asked for.
        for (re[2 * nw ..]) |v| try std.testing.expect(std.math.isNan(v));
    }
}

test "finalize leaves nothing of its allocator behind when an allocation fails" {
    const D = struct {
        pub const U = Conductance.U;
        pub const num_ports = Conductance.num_ports;
        pub const Model = Conductance.Model;
        pub const Instance = Conductance.Instance;
        pub fn eval(comptime S: type, xv: *const [2]S.V, m: *const Model, _: *const Instance, _: SimState) contract.Rows(@This(), S) {
            const x = contract.probes(@This(), S, xv);
            const i = x[0].sub(x[1]).scale(m.g);
            return contract.rows(@This(), S, .{ i, i.neg() });
        }
        pub const State = struct { seen: f64 = 0 };
        pub fn initState(_: *const Model, _: *const Instance) State {
            return .{};
        }
    };
    try std.testing.checkAllAllocationFailures(std.testing.allocator, struct {
        fn run(gpa: std.mem.Allocator) !void {
            const proto = try gpa.create(ProtoStore(D));
            proto.* = .{};
            defer ProtoStore(D).destroy(proto, gpa);
            try proto.append(&.{}, &.{}, .{ 1, 2 });
            try proto.append(&.{}, &.{}, .{ 2, 0 });
            const batch = try ProtoStore(D).finalize(proto, gpa, dense3).unwrap();
            batch.hooks.deinit(batch.ctx, gpa);
        }
    }.run, .{});
}

test "finalize starts a batch before its analog initial block, whatever the allocator held" {
    // A fixed buffer of set bits stands in for reused memory: `create` hands
    // out the bytes as they are, so a field finalize skips reads as true.
    var buf: [16 * 1024]u8 align(16) = @splat(0xff);
    var fba = std.heap.FixedBufferAllocator.init(&buf);
    const a = fba.allocator();
    var proto: ProtoStore(Conductance) = .{};
    try proto.append(&.{}, &.{}, .{ 1, 2 });
    const batch = try ProtoStore(Conductance).finalize(&proto, a, dense3).unwrap();
    defer batch.hooks.deinit(batch.ctx, a);
    const typed: *DeviceBatch(Conductance) = @ptrCast(@alignCast(batch.ctx));
    batch.hooks.set_sim_state(batch.ctx, .{});
    try std.testing.expect(typed.sim.analog_initial);
    try std.testing.expect(!typed.held_initial);
    try std.testing.expect(!typed.committed_initial);
}

test "snapLimited returns the iterate for a few-ulp limited leg and keeps a real clamp" {
    // Unknowns 0 and 1 are read and written, 2 only read (it sets the scale).
    const cur = [3]f64{ 1.0, 2.0, 3.0 };
    var lx = [3]f64{ 1.0 + 4 * std.math.floatEps(f64), 2.1, 99.0 };
    impl.snapLimited(3, 0b111, 0b011, &lx, cur);
    try std.testing.expectEqual(@as(f64, 1.0), lx[0]);
    try std.testing.expectEqual(@as(f64, 2.1), lx[1]);
    try std.testing.expectEqual(@as(f64, 99.0), lx[2]);
}

test "GPU inventory: every built-in model evaluates on the GPU" {
    // dev/devices/gpu-evaluation.md, "Inventory" (#139): gpuEligible turns
    // down no built-in; a model that newly falls back fails here.
    const Why = impl.test_access.GpuRefusal;
    const models = @import("models");
    @setEvalBranchQuota(1_000_000);
    inline for (@typeInfo(models).@"struct".decl_names) |name| {
        const D = @field(models, name);
        if (comptime @hasDecl(D, "eval")) {
            const got: ?Why = comptime impl.test_access.gpuRefusal(D);
            std.testing.expectEqual(@as(?Why, null), got) catch |e| {
                std.debug.print("gpu inventory: {s}\n", .{name});
                return e;
            };
        }
    }
}

test "Dual: sel takes b at a ±0 indicator and a at any other, NaN included" {
    // `sel` computes `c.v != 0.0` on the bits (#25); this pins the bit trick
    // to the float compare. The miscompile it dodges needs a stripped object
    // (dev/zig-backend.md), as the device models are: the B source tests in
    // src/tests catch it.
    const S = impl.test_access.Sparse;
    const x = S.probe(1, 0.5);
    const ax = S.sel(x.lt(S.con(0.0)), x.neg(), x);
    try std.testing.expectEqual(@as(f64, 0.5), ax.v);
    try std.testing.expectEqual(@as(f64, 1), ax.ddxAt(1));
    for ([_]f64{ 0.0, -0.0 }) |c| try std.testing.expectEqual(@as(f64, 0.5), S.sel(S.con(c), x.neg(), x).v);
    for ([_]f64{ 1.0, -1.0, std.math.floatTrueMin(f64), std.math.inf(f64), std.math.nan(f64) }) |c| {
        try std.testing.expectEqual(@as(f64, -0.5), S.sel(S.con(c), x.neg(), x).v);
    }
}

test "Real: sel picks what Dual's .v picks, ±0 and NaN included" {
    // Oracle: `DualFor.sel`'s `.v`, which `RealFor` must repeat verbatim; it
    // used `c.v != 0.0`, the compare #25 miscompiles.
    const R = impl.test_access.Real;
    const S = impl.test_access.Sparse;
    for ([_]f64{ 0.0, -0.0, 1.0, -1.0, std.math.floatTrueMin(f64), std.math.inf(f64), -std.math.inf(f64), std.math.nan(f64) }) |c| {
        const want = S.sel(S.con(c), S.con(-0.5), S.con(0.5)).v;
        try std.testing.expectEqual(want, R.sel(R.con(c), R.con(-0.5), R.con(0.5)).v);
    }
}

test "BSIM4 linked admission and limited planes agree with forced wide across modes" {
    const D = @import("models").bsim4va;
    const n = contract.nU(D);
    const reference_vt = deviceVtable(D, "bsim4_forced_wide_test");
    const get = @extern(*const fn () callconv(.c) @TypeOf(reference_vt), .{ .name = "arp_device_bsim4va" });
    const actual_vt = get();
    const a = std.testing.allocator;
    var rows: [n * n]u32 = undefined;
    var columns: [n + 1]u32 = undefined;
    for (&rows, 0..) |*row, i| row.* = @intCast(i % n);
    for (&columns, 0..) |*column, i| column.* = @intCast(i * n);
    const pattern: impl.test_access.PatternView = .{ .col_ptr = &columns, .row_idx = &rows, .n = n, .trash_slot = n * n };
    const Helpers = struct {
        fn batch(gpa: std.mem.Allocator, vt: @TypeOf(reference_vt), model: *const D.Model, instance: *const D.Instance, nodes: *const [n]u32, pv: impl.test_access.PatternView) !Batch {
            const proto = try vt.proto_create(gpa).unwrap();
            defer proto.destroy(proto.ctx, gpa);
            try vt.proto_add(proto.ctx, gpa, @ptrCast(model), @ptrCast(instance), nodes).unwrap();
            // Evaluation is also valid on the finalized template; retain its
            // owned immutable tapes for the whole test batch lifetime.
            return proto.finalize(proto.ctx, gpa, pv).unwrap();
        }
        fn voltages(nodes: [n]u32, ports: [4]f64, sign: f64) [n + 1]f64 {
            var result: [n + 1]f64 = @splat(0);
            inline for (0..n) |u| {
                if (comptime D.u_kinds[u] == .voltage) {
                    const port = comptime D.collapse_full[u] orelse u;
                    result[nodes[u]] = sign * ports[port];
                }
            }
            return result;
        }
    };
    const Planes = struct {
        g: [n * n + 1]f64 = @splat(0),
        c: [n * n + 1]f64 = @splat(0),
        rhs: [n + 1]f64 = @splat(0),
        q: [n + 1]f64 = @splat(0),
        fn eval(self: *@This(), b: Batch, x: []const f64) void {
            self.* = .{};
            b.eval(b.ctx, &.{ .g_vals = &self.g, .c_vals = &self.c, .rhs = &self.rhs, .q_vec = &self.q }, 0, 1, x, 0);
        }
    };
    var checked: usize = 0;
    var admitted: usize = 0;
    for ([_]i64{ 1, -1 }) |polarity| {
        for ([_]i64{ 0, 1 }) |rds| for ([_]i64{ 0, 1, 2, 3 }) |gate| for ([_]i64{ 0, 1, 2 }) |body| for ([_]i64{ 0, 1 }) |tnoi| {
            var model: D.Model align(16) = .{ .temperature__ = if (polarity > 0) 250 else 400, .l = 0.5e-6, .w = 10e-6, .nf = 2 };
            cardInt(&model, "typeZ", polarity, true);
            cardInt(&model, "rdsmod", rds, false);
            cardInt(&model, "rgatemod", gate, false);
            cardInt(&model, "rbodymod", body, false);
            cardInt(&model, "tnoimod", tnoi, false);
            D.derive(Real, &model);
            const instance: D.Instance align(16) = .{ .mfactor = 3 };
            const collapse = D.collapse(Real, &model, &instance);
            const eligible = rds == 0 and gate == 0 and body == 0 and tnoi == 0;
            try std.testing.expectEqual(eligible, std.meta.eql(collapse, D.collapse_full));
            var nodes: [n]u32 = undefined;
            for (&nodes, collapse, 0..) |*node, alias, u| node.* = if (alias) |root_node| root_node else @intCast(u);
            const actual = try Helpers.batch(a, actual_vt, &model, &instance, &nodes, pattern);
            defer actual.hooks.deinit(actual.ctx, a);
            const reference = try Helpers.batch(a, reference_vt, &model, &instance, &nodes, pattern);
            defer reference.hooks.deinit(reference.ctx, a);
            const counts = actual.hooks.derivative_counts.?(actual.ctx);
            const reference_counts = reference.hooks.derivative_counts.?(reference.ctx);
            try std.testing.expectEqual(@as(u32, 0), reference_counts.narrow);
            try std.testing.expectEqual(@as(u8, 11), counts.wide_lanes);
            try std.testing.expect(counts.narrow_lanes == 0 or counts.narrow_lanes == 4);
            try std.testing.expectEqual(@as(u32, @intFromBool(eligible and counts.narrow_lanes == 4)), counts.narrow);
            admitted += counts.narrow;
            checked += 1;
            const sim: SimState = .{ .kind = .tran, .analog_initial = false, .iteration = 2 };
            actual.hooks.set_sim_state(actual.ctx, sim);
            reference.hooks.set_sim_state(reference.ctx, sim);
            for ([_]bool{ false, true }) |reverse| {
                actual.hooks.clear_limits.?(actual.ctx);
                reference.hooks.clear_limits.?(reference.ctx);
                const sign: f64 = @floatFromInt(polarity);
                const old = Helpers.voltages(nodes, if (reverse) .{ 0, 0.4, 0.1, -0.1 } else .{ 0.1, 0.4, 0, -0.1 }, sign);
                var x_actual = Helpers.voltages(nodes, if (reverse) .{ 0, 2.7, 1.8, 0.8 } else .{ 1.8, 2.7, 0, 0.8 }, sign);
                var x_reference = x_actual;
                for (0..3) |phase| {
                    if (phase != 0) {
                        const changed_a = actual.hooks.apply_limits.?(actual.ctx, &x_actual, &old);
                        const changed_r = reference.hooks.apply_limits.?(reference.ctx, &x_reference, &old);
                        try std.testing.expectEqual(changed_r, changed_a);
                        try std.testing.expectEqualSlices(f64, &x_reference, &x_actual);
                    }
                    var got: Planes = .{};
                    var want: Planes = .{};
                    got.eval(actual, &x_actual);
                    want.eval(reference, &x_reference);
                    inline for (.{ "g", "c", "rhs", "q" }, .{ 1e-12, 1e-24, 1e-12, 1e-24 }) |field, floor| {
                        for (@field(got, field), @field(want, field)) |v, expected| {
                            try std.testing.expect(std.math.isFinite(v) and std.math.isFinite(expected));
                            try std.testing.expectApproxEqAbs(expected, v, 1e-10 * @max(@abs(v), @abs(expected)) + floor);
                        }
                    }
                }
            }
        };
    }
    try std.testing.expectEqual(@as(usize, 96), checked);
    // The same test runs in default and opt-in builds. The real linked
    // vtable reports whether its BSIM4 object compiled the narrow family.
    try std.testing.expect(admitted == 0 or admitted == 2);
    std.debug.print("bsim4-mode-gate: checked={d} admitted={d}\n", .{ checked, admitted });
}
