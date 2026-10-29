//! Mixed-mode ports [SA Ch.17] as a change of basis. A balanced port has two
//! legs, each a single-ended port against the shared reference, and two
//! modes: differential, (a+ - a-)/sqrt 2 against 2 z0, and common,
//! (a+ + a-)/sqrt 2 against z0/2. These are the power waves of V+ - V- and
//! of (V+ + V-)/2, so S over the modes is M S_legs M^T for the orthogonal M,
//! and Y, Z, H and the noise parameters follow from it unchanged. Modes are
//! ordered as HSPICE maps them to Touchstone (S13 = SDC11 for two balanced
//! ports): each port's first mode (single-ended or differential) in port
//! order, then the common modes of the balanced ports in port order. Legs
//! follow the same order, so with no balanced port M is the identity and is
//! not stored.
//! In: the query's ports. Out: the legs the solves run over, M and each
//! mode's z0.
//! Cites HSPICE [SA Ch.17]; Bockelman and Eisenstadt, IEEE Trans. MTT 43(7),
//! 1995.
//!
//! Contents:
//! - `pub const legs_max = 16`
//! - `pub const Basis = struct` (`legs`, `m`, `z0`, `pair`, `Error`, `init`, `deinit`, `toModes`, `voltageWeight`)

const std = @import("std");
const assert = std.debug.assert;
const testing = std.testing;
const Allocator = std.mem.Allocator;
const Port = @import("core").query.Port;

/// Most legs `toModes` takes, the size of its fixed scratch.
pub const legs_max = 16;

/// The legs, M and the mode impedances of one query's ports.
pub const Basis = struct {
    legs: []Port,
    /// Mode-by-leg M, row-major; empty when no port is balanced.
    m: []f64,
    /// Reference impedance of each mode.
    z0: []f64,
    /// The modes `.lin` reads as its two-port (H, stability, noise).
    pair: [2]usize,

    /// Every way `init` fails; nothing stays allocated.
    pub const Error = Allocator.Error || error{
        /// `common[i]` asks a common mode of a port that is not balanced.
        InvalidQueryOptions,
    };

    /// `common[i]` picks port i's common mode for the two-port
    /// (`Lin.common`). The legs and tables are allocated from `gpa`; free
    /// them with `deinit`.
    pub fn init(gpa: Allocator, ports: []const Port, common: [2]bool) Error!Basis {
        var n_bal: usize = 0;
        for (ports) |p| n_bal += @intFromBool(p.balanced != null);
        const nl = ports.len + n_bal;
        const legs = try gpa.alloc(Port, nl);
        errdefer gpa.free(legs);

        const z0 = try gpa.alloc(f64, nl);
        errdefer gpa.free(z0);

        const m = try gpa.alloc(f64, if (n_bal == 0) 0 else nl * nl);
        errdefer gpa.free(m);

        @memset(m, 0);
        var b: Basis = .{ .legs = legs, .m = m, .z0 = z0, .pair = .{ 0, 1 } };
        var c = ports.len;
        const r: f64 = std.math.sqrt1_2;
        for (ports, 0..) |p, k| {
            legs[k] = p;
            legs[k].balanced = null;
            z0[k] = p.z0;
            const minus = p.balanced orelse {
                if (m.len != 0) m[k * nl + k] = 1;
                if (k < 2 and common[k]) return error.InvalidQueryOptions;
                continue;
            };
            legs[c] = legs[k];
            legs[c].node = minus.node;
            legs[c].branch = minus.branch;
            m[k * nl + k] = r;
            m[k * nl + c] = -r;
            m[c * nl + k] = r;
            m[c * nl + c] = r;
            z0[k] = 2 * p.z0;
            z0[c] = p.z0 / 2;
            if (k < 2 and common[k]) b.pair[k] = c;
            c += 1;
        }
        assert(c == nl);
        return b;
    }

    /// Frees what `init` allocated; `gpa` must be the allocator it took.
    pub fn deinit(b: *Basis, gpa: Allocator) void {
        gpa.free(b.legs);
        gpa.free(b.m);
        gpa.free(b.z0);
    }

    /// Turns the stacked (re, im) leg S matrix `s` into the mode one in
    /// place: M S M^T. A no-op with no balanced port. Fails with
    /// error.InvalidQueryOptions past `legs_max` legs.
    pub fn toModes(b: Basis, s: []f64) error{InvalidQueryOptions}!void {
        if (b.m.len == 0) return;
        const n = b.legs.len;
        if (n > legs_max) return error.InvalidQueryOptions;
        assert(s.len >= 2 * n * n);
        // SAFETY: every entry read below is written by the first pass.
        var tmp: [2 * legs_max * legs_max]f64 = undefined;
        // ponytail: dense O(n^3) on at most 16 legs; M has two nonzeros a row.
        for (0..n) |i| for (0..n) |j| for (0..2) |c| {
            var acc: f64 = 0;
            for (0..n) |k| acc += b.m[i * n + k] * s[(k * n + j) * 2 + c];
            tmp[(i * n + j) * 2 + c] = acc;
        };
        for (0..n) |i| for (0..n) |j| for (0..2) |c| {
            var acc: f64 = 0;
            for (0..n) |k| acc += tmp[(i * n + k) * 2 + c] * b.m[j * n + k];
            s[(i * n + j) * 2 + c] = acc;
        };
    }

    /// Returns the weight of leg `l`'s port voltage in mode `q`'s voltage:
    /// the mode wave's M entry scaled by sqrt(z0_mode / z0_leg), so the noise
    /// wave of mode q is the sum over legs of w v_l / sqrt z0_q.
    pub fn voltageWeight(b: Basis, q: usize, l: usize) f64 {
        assert(q < b.z0.len);
        assert(l < b.legs.len);
        const mql = if (b.m.len == 0) @as(f64, @floatFromInt(@intFromBool(q == l))) else b.m[q * b.legs.len + l];
        return mql * @sqrt(b.z0[q] / b.legs[l].z0);
    }
};

test Basis {
    // A balanced port and a single-ended one: legs (+, single, -), M
    // orthogonal, so the identity S stays the identity over the modes.
    const ports = [_]Port{
        .{ .node = 1, .branch = 4, .balanced = .{ .node = 2, .branch = 5 } },
        .{ .node = 3, .branch = 6 },
    };
    var b = try Basis.init(testing.allocator, &ports, .{ false, false });
    defer b.deinit(testing.allocator);
    try testing.expectEqual(@as(usize, 3), b.legs.len);
    try testing.expectEqual(@as(u32, 2), b.legs[2].node);
    try testing.expectEqualSlices(f64, &.{ 100, 50, 25 }, b.z0);
    var s: [2 * 9]f64 = @splat(0);
    for (0..3) |i| s[(i * 3 + i) * 2] = 1;
    try b.toModes(&s);
    for (0..3) |i| for (0..3) |j| {
        try testing.expectApproxEqAbs(@as(f64, @floatFromInt(@intFromBool(i == j))), s[(i * 3 + j) * 2], 1e-15);
        try testing.expectApproxEqAbs(0, s[(i * 3 + j) * 2 + 1], 1e-15);
    };
    // The common mode of a balanced port can be the two-port's first port;
    // asking it of a single-ended port is an error.
    var c = try Basis.init(testing.allocator, &ports, .{ true, false });
    defer c.deinit(testing.allocator);
    try testing.expectEqual([2]usize{ 2, 1 }, c.pair);
    try testing.expectError(error.InvalidQueryOptions, Basis.init(testing.allocator, &ports, .{ false, true }));
    // Every allocation failure, and the late option error, frees the rest.
    try testing.checkAllAllocationFailures(testing.allocator, struct {
        fn f(gpa: Allocator, ps: []const Port, common: [2]bool) !void {
            var basis = Basis.init(gpa, ps, common) catch |e| switch (e) {
                error.InvalidQueryOptions => return,
                else => return e,
            };
            basis.deinit(gpa);
        }
    }.f, .{ &ports, [2]bool{ false, true } });
    try testing.checkAllAllocationFailures(testing.allocator, struct {
        fn f(gpa: Allocator, ps: []const Port) !void {
            var basis = try Basis.init(gpa, ps, .{ false, false });
            basis.deinit(gpa);
        }
    }.f, .{&ports});
}
