//! The `ESPICE_GPU_STATS` census of one GPU query. In: the entry points'
//! laps, waits and copies. Out: the per-eval split and the census lines
//! printed at the context's `deinit`. Diagnosis only: nothing reads it back.
//! Cites: none.
//!
//! Contents:
//! - `pub const Prof = struct (`start`, `lap`, `print`)`
//! - `pub const Wait = enum`

const std = @import("std");
const assert = std.debug.assert;
const context = @import("context.zig");
const Host = context.Host;
const nowNs = context.nowNs;

/// `ESPICE_GPU_STATS` wall-time split of the device path, printed at deinit.
pub const Prof = struct {
    on: bool = false,
    last: u64 = 0,
    evals: u64 = 0,
    /// Evals served by a prefetch, with no wait of their own.
    fused: u64 = 0,
    /// Host time to enqueue an eval, the wait for the device, and the host
    /// batches stamped after it.
    enqueue: u64 = 0,
    wait: u64 = 0,
    host: u64 = 0,
    lims: u64 = 0,
    lim: u64 = 0,
    /// `ESPICE_GPU_STATS=phases`: a sync after each eval stage, so the
    /// stage times below add up to the device time (upload+clear, eval
    /// kernels, reduce level 1, level 2, download). Serializes the
    /// pipeline; diagnosis only.
    phases: bool = false,
    phase: [5]u64 = @splat(0),
    /// Host waits on the stream, all causes, and by cause (`Wait`).
    syncs: u64 = 0,
    waits: [@typeInfo(Wait).@"enum".field_names.len]u64 = @splat(0),
    /// The census: evals that carried charge (`evalQ`), accepted
    /// steps (`stateCtl(.commit)`, the operating point included), and the
    /// bytes each way over the bus, every copy counted.
    charge_evals: u64 = 0,
    steps: u64 = 0,
    up: u64 = 0,
    down: u64 = 0,

    pub fn start(self: *Prof) void {
        if (self.on) self.last = nowNs();
    }

    /// Adds the time since the previous lap to `field`.
    pub fn lap(self: *Prof, comptime field: []const u8) void {
        if (!self.on) return;
        const t = nowNs();
        @field(self, field) += t - self.last;
        self.last = t;
    }

    pub fn print(self: Prof) void {
        if (!self.on) return;
        const us = struct {
            fn f(ns: u64, n: u64) f64 {
                return if (n == 0) 0 else @as(f64, @floatFromInt(ns)) * 1e-3 / @as(f64, @floatFromInt(n));
            }
        }.f;
        std.debug.print(
            "gpu-prof: syncs={d} evals={d} fused={d} per eval us: enqueue={d:.1} wait={d:.1} host={d:.1}; limits={d} {d:.1} us each\n",
            .{ self.syncs, self.evals, self.fused, us(self.enqueue, self.evals), us(self.wait, self.evals), us(self.host, self.evals), self.lims, us(self.lim, self.lims) },
        );
        const per = struct {
            fn f(v: u64, n: u64) f64 {
                return if (n == 0) 0 else @as(f64, @floatFromInt(v)) / @as(f64, @floatFromInt(n));
            }
        }.f;
        const newton = self.evals - self.charge_evals;
        std.debug.print(
            "gpu-census: newton evals={d} charge evals={d} steps={d}; syncs per newton eval={d:.2} per step={d:.2}; KB per newton eval up={d:.1} down={d:.1}; KB per step up={d:.1} down={d:.1}\n",
            .{ newton, self.charge_evals, self.steps, per(self.syncs, newton), per(self.syncs, self.steps), per(self.up, newton) / 1024, per(self.down, newton) / 1024, per(self.up, self.steps) / 1024, per(self.down, self.steps) / 1024 },
        );
        std.debug.print("gpu-census: waits", .{});
        inline for (@typeInfo(Wait).@"enum".field_names, self.waits) |name, w| std.debug.print(" {s}={d}", .{ name, w });
        std.debug.print("\n", .{});
        if (self.phases) std.debug.print(
            "gpu-prof: phases per eval us: upload+clear={d:.1} eval={d:.1} reduce1={d:.1} reduce2={d:.1} download={d:.1}\n",
            .{ us(self.phase[0], self.evals), us(self.phase[1], self.evals), us(self.phase[2], self.evals), us(self.phase[3], self.evals), us(self.phase[4], self.evals) },
        );
    }
};

/// Why the host waited on the stream, for the `ESPICE_GPU_STATS` census:
/// a Newton or charge eval nobody prefetched, a limit pass (with the eval
/// it prefetched), the state half, the accepted-state query (with the
/// charge eval it prefetched), a lazy tape pass, a host-stepped batch's
/// Instance merge, the eval check's replay, and everything else
/// (parameter repacks, `syncHostState`).
pub const Wait = enum { eval, charge, lim, state, query, tape, stepped, check, other };

// ---- private ----
