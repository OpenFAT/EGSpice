//! The small-signal hooks of one device batch: an operating point (and a
//! frequency grid) in, the noise generators with their PSDs and the
//! frequency-dependent matrix terms out. Cites LRM §4.6.4 (noise sources,
//! §4.6.4.6 correlation), §4.5.7/§4.5.11/§4.5.12 (absdelay, laplace, zi),
//! ngspice's nevalsrc.c:106 for `@abs`, and #119 for the PSD family.
//!
//! Contents:
//! - `pub fn SmallSignal(comptime D: type) type`, with `noise_names`,
//!   `collectNoise`, `collectNoiseLocal`, `collectAcDyn`, `acDyn`

const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;
const contract = @import("contract");
const abi = @import("device_abi");
const layout = @import("layout.zig");
const stamp = @import("stamp.zig");
const traits = @import("traits.zig");
const DeviceBatch = @import("batch.zig").DeviceBatch;
const GROUND = abi.GROUND;
const NoiseSource = abi.NoiseSource;
const SlotMap = layout.SlotMap;

/// The small-signal hooks of `DeviceBatch(D)`, each over a type-erased batch.
pub fn SmallSignal(comptime D: type) type {
    return struct {
        const Batch = DeviceBatch(D);
        const n_u: usize = contract.nU(D);
        const has_state = @hasDecl(D, "State");
        const has_q = @hasDecl(D, "q");
        const has_limit = @hasDecl(D, "limit");
        // Only timer-only models: their breakpoints are pure in (model, t).
        const has_bp = @hasDecl(D, "nextBreakpoint") and traits.skipsTimerState(D);
        const has_src_brk = traits.hasSpiceBreaks(D);
        const Ac = traits.AcTerms(D);

        const noise_order = if (@hasDecl(D, "noise_gens")) traits.noiseOrder(D) else {};

        pub const noise_names = if (@hasDecl(D, "noise_gens")) blk: {
            var names: [D.noise_gens.len][]const u8 = undefined;
            for (noise_order.gen, &names) |k, *name| {
                const gen = D.noise_gens[k];
                name.* = if (@hasField(@TypeOf(gen), "name")) gen.name else "";
            }
            const frozen = names;
            break :blk frozen;
        } else {};

        /// Appends one `NoiseSource` per declared generator per instance, with
        /// the PSD `D.noisePsd` gives at `x`. Pure in `x`, so `.pnoise` can
        /// call it once per PSS sample.
        pub fn collectNoise(ctx: *anyopaque, x: []const f64, gpa: Allocator, list: *std.ArrayList(NoiseSource)) abi.DeviceResult(void) {
            return abi.DeviceResult(void).fromLocal(collectNoiseLocal(ctx, x, gpa, list));
        }

        pub fn collectNoiseLocal(ctx: *anyopaque, x: []const f64, gpa: Allocator, list: *std.ArrayList(NoiseSource)) error{OutOfMemory}!void {
            const self = Batch.of(ctx);
            try list.ensureUnusedCapacity(gpa, self.count * D.noise_gens.len);
            for (0..self.count) |id| {
                const base: u32 = @intCast(list.items.len);
                const terms = D.noisePsd(stamp.NoiseFamily(D), self.localX(x, id), self.model(id), &self.instances[id], self.sim);
                // m copies are m independent generators: m times the power.
                // A table row's PSD is the table, so its amplitude takes sqrt(m).
                const mf = stamp.mfactorOf(D, &self.instances[id]);
                // Rows go out in `noise_order`, so a correlated generator's
                // rows are contiguous. Term k belongs to generator k
                // (noise-contract.md §3). `@abs` matches ngspice
                // nevalsrc.c:106. Zero-power generators stay: their ordinal
                // identifies them across PSS samples. A (u, u) branch is u to
                // ground. A generator on a branch with a retained flow unknown
                // enters its flow row with a minus sign (`noiseFlow`):
                // ground to that row.
                inline for (noise_order.gen, noise_order.lead) |k, lead| {
                    const gen = D.noise_gens[k];
                    const t = terms[k];
                    // ponytail: no device publishes a §4.6.4.6 partial
                    // correlation (VerA correlates through `source`); one that
                    // does needs a NoiseSource field and the consumers' cross
                    // term 2·rho·sqrt(S_a·S_b)·Re(H_a·conj(H_b)).
                    assert(t.corr_with == null);
                    const table = comptime if (@hasField(@TypeOf(gen), "table")) gen.table else null;
                    // A collapsed flow unknown shares another unknown's row
                    // (a switch branch in its flow arm): the branch adds
                    // to KCL directly, so the generator stays a current.
                    const flow: ?u32 = if (comptime traits.noiseFlow(D, k)) |u| blk: {
                        const r = self.gath[id * n_u + u];
                        if (r == GROUND) break :blk null;
                        for (0..n_u) |v| if (v != u and self.gath[id * n_u + v] == r) break :blk null;
                        break :blk r;
                    } else null;
                    list.appendAssumeCapacity(.{
                        .node_p = if (flow != null) GROUND else self.gath[id * n_u + gen.row],
                        .node_n = flow orelse if (gen.row == gen.col) GROUND else self.gath[id * n_u + gen.col],
                        .white = @abs(t.white) * mf,
                        .flicker = @abs(t.flicker) * mf,
                        .ef = t.ef,
                        .coeff = if (table != null) t.coeff * @sqrt(mf) else t.coeff,
                        .group = base + lead,
                        .table = if (table) |i| contract.noiseTable(D, self.model(id), i) else .{ .interp = .linear, .points = &.{} },
                    });
                }
            }
        }

        pub fn collectAcDyn(ctx: *anyopaque, gpa: Allocator, list: *std.ArrayList(u32)) abi.DeviceResult(void) {
            const self = Batch.of(ctx);
            list.ensureUnusedCapacity(gpa, self.count * Ac.ac_dyn_slots.len) catch return .out_of_memory;
            const k = SlotMap(D).k;
            const pos = comptime blk: {
                @setEvalBranchQuota(10_000_000);
                var p: [Ac.ac_dyn_slots.len]u32 = undefined;
                for (&p, Ac.ac_dyn_slots) |*o, s| o.* = SlotMap(D).at(s / n_u, s % n_u);
                break :blk p;
            };
            for (0..self.count) |id| {
                for (pos) |p| list.appendAssumeCapacity(self.slots[id * k + p]);
            }
            return .{ .ok = {} };
        }

        /// `Hooks.ac_dyn`: one `Ac.acDyn` call per instance and lane-width
        /// chunk of `omegas`, a ragged tail padded with its last ω.
        pub fn acDyn(ctx: *anyopaque, x: []const f64, omegas: []const f64, re: []f64, im: []f64) usize {
            const self = Batch.of(ctx);
            const K = Ac.ac_dyn_slots.len;
            const LW = std.simd.suggestVectorLength(f64) orelse 1;
            const V = @Vector(LW, f64);
            const nw = omegas.len;
            const sim = self.sim;
            for (0..self.count) |id| {
                const lx = self.localX(x, id);
                var j: usize = 0;
                while (j < nw) : (j += LW) {
                    const cnt = @min(LW, nw - j);
                    var ow: [LW]f64 = undefined;
                    for (0..LW) |l| ow[l] = omegas[j + @min(l, cnt - 1)];
                    var out: [K]std.math.Complex(V) = undefined;
                    Ac.acDyn(V, self.model(id), &self.instances[id], &lx, sim, ow, &out);
                    const mf: V = @splat(stamp.mfactorOf(D, &self.instances[id]));
                    for (out, 0..) |o, k| {
                        const e = (id * K + k) * nw + j;
                        const r: [LW]f64 = o.re * mf;
                        const i: [LW]f64 = o.im * mf;
                        @memcpy(re[e..][0..cnt], r[0..cnt]);
                        @memcpy(im[e..][0..cnt], i[0..cnt]);
                    }
                }
            }
            return self.count * K;
        }
    };
}
