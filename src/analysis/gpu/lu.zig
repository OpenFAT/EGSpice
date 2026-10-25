//! The device LU, stage 1 of dev/solvers/gpu-lu.md: the host assembles A,
//! keeps every pivot decision and uploads the values; the device replays
//! the refactor and runs both triangular solves (`solver.lu_kernels`), and
//! dx comes back. Bitwise the host's `refactor` plus `solveNeg`, so a device
//! pivot failure peels to the host's full factor as if the host had run
//! the refactor itself. The admission bars cite the E2/E3 measurements in
//! dev/solvers/gpu-lu.md and #141.

const std = @import("std");
const gompute = @import("gompute");
const solver = @import("solver");
const numerics = @import("core").numerics;
const Knobs = @import("../Circuit.zig").Knobs;
const devices = @import("devices.zig");

const K = solver.lu_kernels;
const direct = solver.direct;
/// `-Dlu-instrument`: compiles the `ESPICE_GPU_LU_BENCH` harness (`Bench`).
const instrument = direct.instrument;

const backend = devices.backend;
const Raw = if (backend) |be| gompute.RawByName(be) else void;
const Buffer = if (backend != null) Raw.Buffer else void;
const Stream = if (backend != null) Raw.Stream else void;
const GraphExec = if (backend != null) Raw.GraphExec else void;
const Event = if (backend != null) Raw.Event else void;

const nowNs = @import("context.zig").nowNs;

/// A Newton matrix formed on the device (02 2.3, `GpuLu.solveResident`):
/// A = g + alpha·c over the GPU context's planes, then the host's values at
/// `k` (the diagonal: the ground pin and every host load land there)
/// written over it. Borrowed views of the context's buffers.
pub const Resident = struct {
    alpha: f64,
    g: Buffer,
    c: Buffer,
    /// The overlay slots on the device and on the host.
    k: Buffer,
    k_host: []const u32,
};

/// `lu_device.zig`'s `gc_block`.
const gc_block: u32 = 256;

/// Per-query device LU context: kernels, a stream, the per-iteration
/// buffers, and the tables of the current pivot epoch.
pub const GpuLu = struct {
    const Self = @This();

    gpa: std.mem.Allocator,
    n: u32,
    nnz: u32,
    k_ref: Raw,
    k_ref16: Raw,
    k_lh: Raw,
    k_uh: Raw,
    k_lp: Raw,
    k_up: Raw,
    stream: Stream,
    pin_a: []f64,
    pin_rhs: []f64,
    pin_dx: []f64,
    pin_fail: []u8,
    d_a: Buffer,
    d_rhs: Buffer,
    d_y: Buffer,
    d_dx: Buffer,

    /// `SparseLu.pattern_epoch` the tables below were built from; 0 = none.
    epoch: u32 = 0,
    /// This epoch's tables were too big or too thin (or failed to upload).
    declined: bool = true,
    /// `ESPICE_GPU_LU=1`: skip the `min_per_col` bar.
    forced: bool = false,
    tab: K.Tab = undefined,
    d_idx: Buffer = undefined,
    d_val: Buffer = undefined,
    d_sync: Buffer = undefined,
    /// Host copy of the tables, kept for `ESPICE_GPU_LU_BENCH`'s host runs.
    idx: []u32 = &.{},
    flops: u64 = 0,
    /// Done-stamp value of the next launch; grows by one per call.
    stamp: u32 = 0,
    /// The current epoch's call (`launchCall`) captured once, without and
    /// with the refactor (02 2.7): the sync buffer cleared and stamp 1, so
    /// one graph replays every call. Dropped with the tables.
    graphs: [2]?GraphExec = .{ null, null },
    /// `direct.Solver.gen` whose factors the device holds, if any.
    dev_gen: ?u32 = null,
    /// A driver call failed: the rest of the query solves on the host.
    poisoned: bool = false,
    knobs: Knobs,
    bench: ?*Bench = null,
    /// `solveResident`'s kernels and overlay buffers, made on its first call.
    res: ?Res = null,
    /// `slv` and its `gen` after a resident refactor, whose A lives only in
    /// `d_a`: `slv.vcopy` is stale while `slv.gen` still equals `gen`
    /// (`flushVcopy`).
    stale: ?struct { slv: *direct.Solver, gen: u32 } = null,
    /// `load_ns`: host time building and uploading epoch tables.
    st: struct { calls: u64 = 0, resident: u64 = 0, refactors: u64 = 0, peels: u64 = 0, epochs: u64 = 0, mismatches: u64 = 0, load_ns: u64 = 0 } = .{},

    const Res = struct { combine: Raw, scatter: Raw, pin_ov: []f64, d_ov: Buffer };

    /// Loads the five kernels and allocates the per-iteration buffers for an
    /// n-unknown, nnz-entry pattern. Fails with `error.NoGpuArtifacts` in a
    /// build without kernel images, or with the driver's error. Every solver
    /// it later serves must be n by n with nnz entries. Caller owns the
    /// result; free with `deinit`.
    pub fn init(gpa: std.mem.Allocator, n: u32, nnz: u32, knobs: Knobs) !*Self {
        if (comptime backend == null) return error.NoGpuArtifacts;
        const be = backend.?;
        // On this thread's device (`devices.place`), as the query's eval.
        const ord = devices.ordinal;
        var k_ref = try gompute.rawKernelByName(be, "arp_lu_refactor", ord);
        errdefer k_ref.deinit();
        var k_ref16 = try gompute.rawKernelByName(be, "arp_lu_refactor16", ord);
        errdefer k_ref16.deinit();
        var k_lh = try gompute.rawKernelByName(be, "arp_lu_lhead", ord);
        errdefer k_lh.deinit();
        var k_uh = try gompute.rawKernelByName(be, "arp_lu_uhead", ord);
        errdefer k_uh.deinit();
        var k_lp = try gompute.rawKernelByName(be, "arp_lu_lpanel", ord);
        errdefer k_lp.deinit();
        var k_up = try gompute.rawKernelByName(be, "arp_lu_upanel", ord);
        errdefer k_up.deinit();
        const pin_a = try pinnedF64(&k_ref, nnz);
        errdefer freePinnedF64(&k_ref, pin_a);
        const pin_rhs = try pinnedF64(&k_ref, n);
        errdefer freePinnedF64(&k_ref, pin_rhs);
        const pin_dx = try pinnedF64(&k_ref, n);
        errdefer freePinnedF64(&k_ref, pin_dx);
        const pin_fail = try k_ref.allocPinned(4);
        errdefer k_ref.freePinned(pin_fail);
        var d_a = try k_ref.alloc(@max(nnz, 1) * 8);
        errdefer d_a.free();
        var d_rhs = try k_ref.alloc(@max(n, 1) * 8);
        errdefer d_rhs.free();
        var d_y = try k_ref.alloc(@max(n, 1) * 8);
        errdefer d_y.free();
        var d_dx = try k_ref.alloc(@max(n, 1) * 8);
        errdefer d_dx.free();
        var stream = try k_ref.createStream();
        errdefer stream.deinit();
        const self = try gpa.create(Self);
        self.* = .{
            .gpa = gpa,
            .n = n,
            .nnz = nnz,
            .k_ref = k_ref,
            .k_ref16 = k_ref16,
            .k_lh = k_lh,
            .k_uh = k_uh,
            .k_lp = k_lp,
            .k_up = k_up,
            .stream = stream,
            .pin_a = pin_a,
            .pin_rhs = pin_rhs,
            .pin_dx = pin_dx,
            .pin_fail = pin_fail,
            .d_a = d_a,
            .d_rhs = d_rhs,
            .d_y = d_y,
            .d_dx = d_dx,
            .knobs = knobs,
        };
        if (instrument) {
            if (knobs.gpu_lu_bench) |limit| self.bench = Bench.init(gpa, &self.k_ref, limit, knobs.gpu_lu_threads) catch null;
        }
        return self;
    }

    /// `bench`, or a comptime null without `-Dlu-instrument`, so no `Bench`
    /// code is compiled.
    inline fn benchOf(self: *const Self) ?*Bench {
        return if (instrument) self.bench else null;
    }

    /// Page-locked `[]f64` of length `n`, backed by at least one element so
    /// even n = 0 has a pointer; release with `freePinnedF64`.
    fn pinnedF64(k: *Raw, n: usize) ![]f64 {
        const bytes = try k.allocPinned(@max(n, 1) * @sizeOf(f64));
        return @as([*]f64, @ptrCast(@alignCast(bytes.ptr)))[0..n];
    }

    /// Frees a `pinnedF64` slice. Not `sliceAsBytes`: it maps an empty
    /// slice to a pointerless `&.{}`, which `freePinned` skips, leaking the
    /// one-element backing.
    fn freePinnedF64(k: *Raw, p: []f64) void {
        k.freePinned(@as([*]u8, @ptrCast(p.ptr))[0 .. @max(p.len, 1) * @sizeOf(f64)]);
    }

    /// Prints the `ESPICE_GPU_STATS`/check/bench summary, then frees every
    /// device and pinned buffer and `self`. Detach it first
    /// (set `Circuit.lu` to null).
    pub fn deinit(self: *Self) void {
        if (comptime backend == null) return;
        if (self.knobs.gpu_stats != .off or self.knobs.gpu_lu_check or self.benchOf() != null) std.debug.print(
            "gpu-lu: calls={d} resident={d} refactors={d} peels={d} epochs={d} load_ms={d} F={d} check_mismatches={d}{s}\n",
            .{ self.st.calls, self.st.resident, self.st.refactors, self.st.peels, self.st.epochs, self.st.load_ns / std.time.ns_per_ms, self.flops, self.st.mismatches, if (self.knobs.gpu_lu_check) "" else " (unchecked)" },
        );
        if (self.benchOf()) |b| b.deinit(self);
        // The solver outlives this context; its copy must not stay stale.
        self.flushVcopy() catch {};
        self.dropTables();
        const k = &self.k_ref;
        if (self.res) |*r| {
            freePinnedF64(k, r.pin_ov);
            r.d_ov.free();
            r.combine.deinit();
            r.scatter.deinit();
        }
        for ([_][]f64{ self.pin_a, self.pin_rhs, self.pin_dx }) |p| freePinnedF64(k, p);
        k.freePinned(self.pin_fail);
        for ([_]*Buffer{ &self.d_a, &self.d_rhs, &self.d_y, &self.d_dx }) |b| b.free();
        self.stream.deinit();
        for ([_]*Raw{ &self.k_uh, &self.k_lp, &self.k_up, &self.k_lh, &self.k_ref16, &self.k_ref }) |kern| kern.deinit();
        self.gpa.destroy(self);
    }

    fn dropTables(self: *Self) void {
        if (comptime backend != null) for (&self.graphs) |*g| if (g.*) |*exec| {
            exec.deinit();
            g.* = null;
        };
        if (self.epoch != 0 and !self.declined) {
            self.d_idx.free();
            self.d_val.free();
            self.d_sync.free();
        }
        self.gpa.free(self.idx);
        self.idx = &.{};
    }

    /// `Circuit.deviceSolve`'s body: dx = -A^-1 rhs on the device. Null
    /// when the device did not try (no general LU yet, a declined epoch,
    /// factors it does not hold); false when it tried and the host's exact
    /// factor and solve must run: a device pivot failure, which clears
    /// `slv.factored`, so the host goes straight to the full factor its own
    /// refactor would have fallen back to. A driver error warns once and
    /// returns null for the rest of the query.
    pub fn solve(self: *Self, slv: *direct.Solver, vals: []f64, rhs: []const f64, dx: []f64) ?bool {
        if (self.poisoned) return null;
        return self.trySolve(slv, vals, rhs, dx) catch |e| {
            std.debug.print("warning: device LU off for this query ({s})\n", .{@errorName(e)});
            self.poisoned = true;
            self.dev_gen = null;
            return null;
        };
    }

    fn trySolve(self: *Self, slv: *direct.Solver, vals: []const f64, rhs: []const f64, dx: []f64) !?bool {
        if (comptime backend == null) return null;
        const lu = slv.sparse() orelse return null;
        if (!slv.factored or !lu.factored) return null;
        // Tape-sized factors refactor in microseconds on the host.
        if (lu.tv.items.len != 0) return null;
        try self.ensureEpoch(slv);
        if (self.declined) return null;
        // A stale `vcopy` (`stale`) says nothing about `vals`.
        const refactor = self.vcopyStale(slv) or !slv.unchanged(vals);
        if (!refactor and self.dev_gen != slv.gen) return null;
        if (refactor) for (lu.void_slots.items) |p| {
            if (vals[p] != 0) return null;
        };

        const n = self.n;
        self.st.calls += 1;
        self.stamp +%= 1;
        const t0 = if (self.benchOf() != null) nowNs() else 0;
        if (refactor) {
            @memcpy(self.pin_a, vals[0..self.nnz]);
            try self.d_a.uploadAtAsync(self.pin_a.ptr, 0, self.nnz * 8, &self.stream);
        }
        @memcpy(self.pin_rhs, rhs[0..n]);
        try self.d_rhs.uploadAtAsync(self.pin_rhs.ptr, 0, n * 8, &self.stream);
        try self.d_sync.fillAsync(0, K.sync_header * 4, &self.stream);
        if (self.benchOf()) |b| try b.ev[0].record(&self.stream);
        try self.launchCall(refactor);
        const t1 = if (self.benchOf() != null) nowNs() else 0;

        if (std.mem.bytesToValue(u32, self.pin_fail[0..4]) != 0) {
            // The lowest failing step: every step below it matched the host,
            // so the host refactor fails there too. Skip it.
            self.st.peels += 1;
            self.dev_gen = null;
            slv.factored = false;
            return false;
        }
        if (refactor) {
            self.st.refactors += 1;
            self.stale = null;
            @memcpy(slv.vcopy, vals[0..self.nnz]);
            slv.gen +%= 1;
            slv.host_stale = true;
            self.dev_gen = slv.gen;
        }
        @memcpy(dx[0..n], self.pin_dx);
        if (self.knobs.gpu_lu_check and refactor) try self.checkAgainstHost(slv, rhs, dx);
        if (self.benchOf()) |b| if (refactor) try b.sample(self, slv, vals, rhs, t1 - t0);
        return true;
    }

    /// The refactor (when `refactor`) from `d_a` and both solves from
    /// `d_rhs`, then dx and the fail word down and the wait. Replays the
    /// epoch's captured graph where the driver has graphs (not under
    /// `ESPICE_GPU_NOGRAPHS` or the bench, whose events sit between the
    /// launches): the same launches, so the same bits.
    fn launchCall(self: *Self, refactor: bool) !void {
        if (self.benchOf() == null and !self.knobs.gpu_nographs and self.k_ref.hasGraphs()) {
            const g = &self.graphs[@intFromBool(refactor)];
            if (g.* == null) g.* = try self.captureCall(refactor);
            try g.*.?.launch(&self.stream);
            // The graph left stamp 1 in the done words; a later direct call
            // must stamp past it.
            self.stamp = 1;
            return self.stream.synchronize();
        }
        try self.enqueueCall(refactor, self.stamp);
        try self.stream.synchronize();
    }

    /// `launchCall`'s graph, captured on a scratch stream: the whole sync
    /// buffer cleared, so stamp 1 is fresh on every replay, then
    /// `enqueueCall`.
    fn captureCall(self: *Self, refactor: bool) !GraphExec {
        var capture_stream = try self.k_ref.createStreamNonBlocking();
        defer capture_stream.deinit();
        std.mem.swap(Stream, &self.stream, &capture_stream);
        defer std.mem.swap(Stream, &self.stream, &capture_stream);
        try self.stream.beginCapture(.thread_local);
        var capturing = true;
        defer if (capturing) {
            // An invalidated capture still ends before its stream goes.
            if (self.stream.endCapture()) |abandoned| {
                var graph = abandoned;
                graph.deinit();
            } else |_| {}
        };
        try self.d_sync.fillAsync(0, @as(usize, self.tab.syncLen()) * 4, &self.stream);
        try self.enqueueCall(refactor, 1);
        const ended = self.stream.endCapture();
        capturing = false;
        var graph = try ended;
        defer graph.deinit();
        return graph.instantiate();
    }

    fn enqueueCall(self: *Self, refactor: bool, stamp_in: u32) !void {
        const n = self.n;
        var tab = self.tab;
        var stamp = stamp_in;
        const k_ref = if (tab.dmap16 != 0) &self.k_ref16 else &self.k_ref;
        if (refactor) try k_ref.launchOn(&self.stream, .{ .x = n }, .{ .x = K.refactor_block }, 0, &.{
            gompute.interface.arg(&tab), self.d_idx.argPtr(), self.d_val.argPtr(), self.d_a.argPtr(), self.d_sync.argPtr(), gompute.interface.arg(&stamp),
        });
        if (self.benchOf()) |b| try b.ev[1].record(&self.stream);
        // Heads: one block per run (an empty grid is not a launch).
        if (tab.l_chunks > 0) try self.k_lh.launchOn(&self.stream, .{ .x = tab.l_chunks }, .{ .x = K.solve_block }, 0, &.{
            gompute.interface.arg(&tab), self.d_idx.argPtr(), self.d_val.argPtr(), self.d_rhs.argPtr(), self.d_y.argPtr(), self.d_sync.argPtr(), gompute.interface.arg(&stamp),
        });
        if (tab.panels() > 0) try self.k_lp.launchOn(&self.stream, .{ .x = tab.panels() }, .{ .x = K.panel }, 0, &.{
            gompute.interface.arg(&tab), self.d_idx.argPtr(), self.d_val.argPtr(), self.d_y.argPtr(), self.d_sync.argPtr(), gompute.interface.arg(&stamp),
        });
        if (self.benchOf()) |b| try b.ev[2].record(&self.stream);
        if (tab.panels() > 0) try self.k_up.launchOn(&self.stream, .{ .x = tab.panels() }, .{ .x = K.panel }, 0, &.{
            gompute.interface.arg(&tab), self.d_idx.argPtr(), self.d_val.argPtr(), self.d_y.argPtr(), self.d_dx.argPtr(), self.d_sync.argPtr(), gompute.interface.arg(&stamp),
        });
        if (tab.u_chunks > 0) try self.k_uh.launchOn(&self.stream, .{ .x = tab.u_chunks }, .{ .x = K.solve_block }, 0, &.{
            gompute.interface.arg(&tab), self.d_idx.argPtr(), self.d_val.argPtr(), self.d_y.argPtr(), self.d_dx.argPtr(), self.d_sync.argPtr(), gompute.interface.arg(&stamp),
        });
        if (self.benchOf()) |b| try b.ev[3].record(&self.stream);
        try self.d_dx.downloadAtAsync(self.pin_dx.ptr, 0, n * 8, &self.stream);
        try self.d_sync.downloadAtAsync(self.pin_fail.ptr, 4, 4, &self.stream);
    }

    /// Whether the device holds the resident tables `solveResident` needs
    /// for `slv` now: the current epoch loaded and admitted, no tape, no
    /// void slot, no bench. Read before the eval, so the planes it asks for
    /// match; a wrong answer only costs `solveResident` a download of A.
    pub fn residentReady(self: *const Self, slv: *direct.Solver) bool {
        if (self.poisoned or self.declined or self.benchOf() != null) return false;
        const lu = slv.sparse() orelse return false;
        return slv.factored and lu.factored and lu.tv.items.len == 0 and lu.void_slots.items.len == 0 and self.epoch == lu.pattern_epoch;
    }

    /// `solve` for a Newton matrix the device forms (02 2.3): `r`'s A, the
    /// host's `vals` at `r.k_host` written over it, then the same refactor
    /// and solves, bitwise `solve` on the whole A. `vals` holds A only at
    /// `r.k_host` on entry; on a null or false return (the host factors)
    /// it holds all of A. Always refactors: an unchanged A refactors to the
    /// same bits. Fails only on a driver fault (warned once, and the rest
    /// of the query solves on the host), with `vals` unfilled.
    pub fn solveResident(self: *Self, slv: *direct.Solver, vals: []f64, rhs: []const f64, dx: []f64, r: Resident) !?bool {
        if (self.poisoned) return error.DeviceLuLost;
        return self.trySolveResident(slv, vals, rhs, dx, r) catch |e| {
            std.debug.print("warning: device LU off for this query ({s})\n", .{@errorName(e)});
            self.poisoned = true;
            self.dev_gen = null;
            return e;
        };
    }

    fn trySolveResident(self: *Self, slv: *direct.Solver, vals: []f64, rhs: []const f64, dx: []f64, r: Resident) !?bool {
        if (comptime backend == null) return error.NoGpuArtifacts;
        const lu = slv.sparse() orelse return self.fill(vals, r);
        if (!slv.factored or !lu.factored or lu.tv.items.len != 0) return self.fill(vals, r);
        try self.ensureEpoch(slv);
        if (self.declined or lu.void_slots.items.len != 0) return self.fill(vals, r);

        self.st.calls += 1;
        self.st.resident += 1;
        self.stamp +%= 1;
        try self.formA(vals, r);
        @memcpy(self.pin_rhs, rhs[0..self.n]);
        try self.d_rhs.uploadAtAsync(self.pin_rhs.ptr, 0, self.n * 8, &self.stream);
        try self.d_sync.fillAsync(0, K.sync_header * 4, &self.stream);
        try self.launchCall(true);
        if (std.mem.bytesToValue(u32, self.pin_fail[0..4]) != 0) {
            // As `trySolve`: the host's full factor takes the whole A.
            self.st.peels += 1;
            self.dev_gen = null;
            slv.factored = false;
            try self.downloadA(vals);
            return false;
        }
        self.st.refactors += 1;
        slv.gen +%= 1;
        slv.host_stale = true;
        self.dev_gen = slv.gen;
        self.stale = .{ .slv = slv, .gen = slv.gen };
        @memcpy(dx[0..self.n], self.pin_dx);
        if (self.knobs.gpu_lu_check) {
            try self.flushVcopy();
            try self.checkAgainstHost(slv, rhs, dx);
        }
        return true;
    }

    /// Queues A = g + alpha·c into `d_a` and `vals` at `r.k_host` over it.
    fn formA(self: *Self, vals: []const f64, r: Resident) !void {
        const res = try self.resKernels(r.k_host.len);
        var n: u64 = self.nnz;
        var alpha = r.alpha;
        var g = r.g;
        var c = r.c;
        if (self.nnz > 0) try res.combine.launchOn(&self.stream, gompute.Dim3.linear(self.nnz, gc_block), .{ .x = gc_block }, 0, &.{
            gompute.interface.arg(&n), gompute.interface.arg(&alpha), g.argPtr(), c.argPtr(), self.d_a.argPtr(),
        });
        if (r.k_host.len == 0) return;
        for (r.k_host, res.pin_ov[0..r.k_host.len]) |slot, *v| v.* = vals[slot];
        try res.d_ov.uploadAtAsync(res.pin_ov.ptr, 0, r.k_host.len * 8, &self.stream);
        var nk: u64 = r.k_host.len;
        var k = r.k;
        try res.scatter.launchOn(&self.stream, gompute.Dim3.linear(r.k_host.len, gc_block), .{ .x = gc_block }, 0, &.{
            gompute.interface.arg(&nk), k.argPtr(), res.d_ov.argPtr(), self.d_a.argPtr(),
        });
    }

    /// `res`, made for `nk` overlay slots on the first call.
    fn resKernels(self: *Self, nk: usize) !*Res {
        if (self.res) |*r| {
            std.debug.assert(r.pin_ov.len >= nk);
            return r;
        }
        const be = backend.?;
        var combine = try gompute.rawKernelByName(be, "arp_gc_combine", self.k_ref.context.ordinal);
        errdefer combine.deinit();
        var scatter = try gompute.rawKernelByName(be, "arp_gc_scatter", self.k_ref.context.ordinal);
        errdefer scatter.deinit();
        const pin_ov = try pinnedF64(&self.k_ref, nk);
        errdefer freePinnedF64(&self.k_ref, pin_ov);
        var d_ov = try self.k_ref.alloc(@max(nk, 1) * 8);
        errdefer d_ov.free();
        self.res = .{ .combine = combine, .scatter = scatter, .pin_ov = pin_ov, .d_ov = d_ov };
        return &self.res.?;
    }

    /// The host's way out of `solveResident`: A formed and brought down
    /// into `vals`; null.
    fn fill(self: *Self, vals: []f64, r: Resident) !?bool {
        try self.formA(vals, r);
        try self.downloadA(vals);
        return null;
    }

    /// `d_a` into `vals`, waited.
    fn downloadA(self: *Self, vals: []f64) !void {
        try self.d_a.downloadAtAsync(self.pin_a.ptr, 0, self.nnz * 8, &self.stream);
        try self.stream.synchronize();
        @memcpy(vals[0..self.nnz], self.pin_a);
    }

    fn vcopyStale(self: *const Self, slv: *const direct.Solver) bool {
        const st = self.stale orelse return false;
        return st.slv == slv and st.gen == slv.gen;
    }

    /// Brings `slv.vcopy` up to the resident A in `d_a` (`stale`), which a
    /// host refactor (`host_stale`) reads.
    fn flushVcopy(self: *Self) !void {
        const st = self.stale orelse return;
        self.stale = null;
        if (st.gen != st.slv.gen) return;
        try self.downloadA(st.slv.vcopy);
    }

    /// `load`s `slv`'s pivot epoch unless the tables already hold it,
    /// timed into `st.load_ns`.
    fn ensureEpoch(self: *Self, slv: *direct.Solver) !void {
        if (self.epoch == slv.sparse().?.pattern_epoch) return;
        const t = nowNs();
        defer self.st.load_ns += nowNs() - t;
        try self.load(slv);
    }

    /// Builds and uploads the tables of `slv`'s current pivot epoch, or
    /// declines it (`declined`) when they do not fit in host or device
    /// memory (or u32 offsets) or, unless `forced`, under `min_flops` and
    /// under either `min_per_col` flops per column or `min_call_flops`.
    fn load(self: *Self, slv: *direct.Solver) !void {
        const lu = slv.sparse().?;
        self.dropTables();
        self.epoch = lu.pattern_epoch;
        self.declined = true;
        self.dev_gen = null;
        var flops: u64 = 0;
        for (lu.ui.items) |i| flops += lu.lp[i + 1] - lu.lp[i];
        self.flops = flops;
        // The buffers and every copy are sized by `self.n`.
        if (lu.n != self.n or !admits(flops, lu.n, self.forced)) return;
        self.upload(slv) catch |e| switch (e) {
            // The device LU is an accelerator: the host LU keeps the epoch.
            error.OutOfMemory, error.AllocFailed => return,
            else => return e, // else: a driver failure poisons the query, as before
        };
        self.declined = false;
        self.st.epochs += 1;
    }

    /// `load`'s table build and upload; frees what it made on failure.
    fn upload(self: *Self, slv: *direct.Solver) !void {
        const lu = slv.sparse().?;
        // E2 calibration knobs: ESPICE_GPU_LU_TAIL=m fixes the solves' tail
        // at m steps, ESPICE_GPU_LU_NOGATHER keeps every column in stored
        // order.
        var opt: K.Options = .{ .gather = !self.knobs.gpu_lu_nogather };
        if (self.knobs.gpu_lu_tail) |m| opt.tail = lu.n - @min(lu.n, m);
        var tb = try K.build(self.gpa, lu, slv.col_ptr, slv.params.refactor_growth_limit, opt);
        defer tb.deinit(self.gpa);
        self.tab = tb.tab;
        self.d_idx = try self.k_ref.alloc(@as(usize, tb.tab.n_idx) * 4);
        errdefer self.d_idx.free();
        try self.d_idx.upload(tb.idx.ptr, @as(usize, tb.tab.n_idx) * 4);
        self.d_val = try self.k_ref.alloc(@as(usize, tb.tab.n_val) * 8);
        errdefer self.d_val.free();
        self.d_sync = try self.k_ref.alloc(@as(usize, tb.tab.syncLen()) * 4);
        errdefer self.d_sync.free();
        try self.d_sync.fillAsync(0, @as(usize, tb.tab.syncLen()) * 4, &self.stream);
        self.stamp = 0;
        if (self.benchOf() != null) self.idx = try self.gpa.dupe(u32, tb.idx);
    }

    /// Whether an epoch of `flops` over `n` columns goes to the device. An
    /// empty matrix never does: its refactor grid would have no blocks.
    /// There is no size cap: tables (4 bytes per flop, built in place on
    /// the host) that do not fit decline in `load`.
    fn admits(flops: u64, n: u32, forced: bool) bool {
        if (n == 0) return false;
        return forced or flops >= min_flops or (flops >= min_per_col * @as(u64, n) and flops >= min_call_flops);
    }

    /// Below this F/n the columns form a chain and every handoff is a
    /// device-wide wait. Wall time on an RTX 4060 Laptop (FP64 1/64),
    /// `cuda` with the device LU against the host LU on one thread:
    /// logic_bsim4_10k (F/n 4,400) 63 against 237 s, sram_bsim4_10k
    /// (2,400) 26 against 52 s, logic_psp103_10k (1,600) 86 against 357 s;
    /// chain_bsim4_10k (63) 11.6 against 9.9 s,
    /// ring_bsim4_10k (87) 24.4 against 20.1 s, and the E2 100k decks
    /// (about 160) lost too (dev/solvers/gpu-lu.md, "E3, host").
    /// logic_bsim4_2k (834) and sram_bsim4_5k (1,976) win 1.7x and 1.3x.
    /// ponytail: no deck of 5M flops or more measured between F/n 182 and
    /// 834; refit the bar if one lands there.
    const min_per_col = 500;

    /// The F/n bar admits nothing smaller: a device call costs about 1.3-1.5
    /// ms whatever its size (launches, uploads, the solve chain) and the host
    /// LU 0.4-0.8 ns per flop. LU ms per Newton iteration, host against
    /// device, RTX 4060 Laptop: sram_bsim4_1k (F 0.8M) 0.60 against 1.40,
    /// sram_bsim4_2k (1.7M, F/n 509) 1.35 against 1.80, sram_bsim4_3k
    /// (5.1M) a tie, logic_bsim4_2k (6.5M) 4.90 against 2.88.
    const min_call_flops = 5_000_000;

    /// At this many flops the device LU wins whatever F/n, since its solve
    /// heads run across the grid (#141). LU time over the transient, host
    /// against device, RTX 4060 Laptop: chain_bsim4_100k (F 74M, F/n 63)
    /// 33.0 against 16.4 s, ring_bsim4_100k (69M) 61.4 against 32.2 s,
    /// chain_psp103_100k (107M) 29.7 against 16.0-18.5 s,
    /// chain_psp103_300k (734M) 134.7 against 98.9 s. The 10k chains and
    /// rings (F 3-5M) still lose or tie, ring_bsim4_60k (34M, F/n 129) 30.2
    /// against 34.5 ms per iteration, chain_bsim4_60k (49M) 55.7 against
    /// 35.0 (host time under load).
    const min_flops = 50_000_000;

    /// `ESPICE_GPU_LU_CHECK`: refactors and solves on the host from the same
    /// values and compares the factors and dx bitwise, printing the first
    /// mismatch. Leaves the host factors current.
    fn checkAgainstHost(self: *Self, slv: *direct.Solver, rhs: []const f64, dx: []const f64) !void {
        const lu = slv.sparse().?;
        const t = self.tab;
        const val = try self.gpa.alloc(f64, t.n_val);
        defer self.gpa.free(val);
        try self.d_val.download(val.ptr, @as(usize, t.n_val) * 8);
        const ref = try self.gpa.alloc(f64, self.n);
        defer self.gpa.free(ref);
        // solveNeg syncs the host factors from vcopy first.
        slv.solveNeg(rhs, ref);
        var bad: ?[]const u8 = null;
        var where: usize = 0;
        var off: u32 = 0;
        for (0..self.n) |k| {
            const nuk = lu.up[k + 1] - lu.up[k];
            const nlk = lu.lp[k + 1] - lu.lp[k];
            const col = val[off..][0 .. nuk + 1 + nlk];
            off += nuk + 1 + nlk;
            if (!std.mem.eql(u8, std.mem.sliceAsBytes(col[0..nuk]), std.mem.sliceAsBytes(lu.ux.items[lu.up[k]..lu.up[k + 1]])) or
                @as(u64, @bitCast(col[nuk])) != @as(u64, @bitCast(lu.udiag[k])) or
                !std.mem.eql(u8, std.mem.sliceAsBytes(col[nuk + 1 ..]), std.mem.sliceAsBytes(lu.lx.items[lu.lp[k]..lu.lp[k + 1]])))
            {
                bad = "factor column";
                where = k;
                break;
            }
        }
        if (bad == null) for (dx, ref, 0..) |a, b, i| {
            if (@as(u64, @bitCast(a)) != @as(u64, @bitCast(b))) {
                bad = "dx";
                where = i;
                break;
            }
        };
        if (bad) |what| {
            self.st.mismatches += 1;
            if (self.st.mismatches <= 3) std.debug.print("gpu-lu check: {s} {d} differs from the host (refactor {d})\n", .{ what, where, self.st.refactors });
        }
    }
};

/// `ESPICE_GPU_LU_BENCH` (E2): after each device refactor, times the same
/// refactor and solve on the host (`SparseLu`, one thread) and with the
/// kernel bodies on 1 and `ESPICE_GPU_LU_THREADS` (default 8) host threads,
/// against the device kernels by event timing. Prints medians at teardown,
/// or after N samples for `ESPICE_GPU_LU_BENCH=N` (then stops sampling).
const Bench = struct {
    ev: [4]Event,
    threads: u32,
    limit: usize = 0,
    /// Microseconds per sample: device refactor kernel, device solve
    /// kernels, device call wall (upload to download), host SparseLu
    /// refactor, host solve, kernel bodies on 1 thread, on `threads`.
    s: [8]std.ArrayList(f64) = @splat(.empty),
    val: []f64 = &.{},
    y: []f64 = &.{},
    dx: []f64 = &.{},
    sync: []u32 = &.{},
    stamp: u32 = 0,
    /// `GpuLu.st.epochs` the buffers above are sized for.
    epoch: u64 = 0,

    fn init(gpa: std.mem.Allocator, k: *Raw, limit: usize, threads: u32) !*Bench {
        if (!k.hasEvents()) return error.Unsupported;
        const b = try gpa.create(Bench);
        errdefer gpa.destroy(b);
        b.* = .{ .ev = undefined, .threads = threads, .limit = limit };
        var made: usize = 0;
        errdefer for (b.ev[0..made]) |*e| e.deinit();
        for (&b.ev) |*e| {
            e.* = try k.createEvent(true);
            made += 1;
        }
        return b;
    }

    fn sample(b: *Bench, lu_ctx: *GpuLu, slv: *direct.Solver, vals: []const f64, rhs: []const f64, wall_ns: u64) !void {
        const gpa = lu_ctx.gpa;
        const t = lu_ctx.tab;
        if (b.limit != 0 and b.s[0].items.len >= b.limit) return;
        defer if (b.limit != 0 and b.s[0].items.len == b.limit) b.report(lu_ctx);
        if (b.epoch != lu_ctx.st.epochs) {
            for ([_][]f64{ b.val, b.y, b.dx }) |p| gpa.free(p);
            gpa.free(b.sync);
            b.val = try gpa.alloc(f64, t.n_val);
            b.y = try gpa.alloc(f64, t.n);
            b.dx = try gpa.alloc(f64, t.n);
            b.sync = try gpa.alloc(u32, t.syncLen());
            @memset(b.sync, 0);
            b.epoch = lu_ctx.st.epochs;
        }
        try b.s[0].append(gpa, try Event.elapsedUs(&b.ev[0], &b.ev[1]));
        try b.s[1].append(gpa, try Event.elapsedUs(&b.ev[1], &b.ev[3]));
        try b.s[7].append(gpa, try Event.elapsedUs(&b.ev[1], &b.ev[2]));
        try b.s[2].append(gpa, @as(f64, @floatFromInt(wall_ns)) * 1e-3);
        const lu = slv.sparse().?;
        var t0 = nowNs();
        try lu.refactor(slv.col_ptr, vals, slv.params.refactor_growth_limit);
        var t1 = nowNs();
        try b.s[3].append(gpa, @as(f64, @floatFromInt(t1 - t0)) * 1e-3);
        const neg = b.dx; // scratch until the kernel runs below
        numerics.scale(neg, -1, rhs[0..t.n]);
        t0 = nowNs();
        lu.solve(neg, neg);
        t1 = nowNs();
        try b.s[4].append(gpa, @as(f64, @floatFromInt(t1 - t0)) * 1e-3);
        slv.host_stale = false; // the refactor above brought lu to vcopy
        const bufs: K.HostBufs = .{ .idx = lu_ctx.idx, .val = b.val, .a = vals, .rhs = rhs, .y = b.y, .dx = b.dx, .sync = b.sync };
        for ([_]u32{ 1, b.threads }, 5..) |th, slot| {
            b.stamp += 1;
            t0 = nowNs();
            _ = try K.runHost(t, bufs, b.stamp, th);
            t1 = nowNs();
            try b.s[slot].append(gpa, @as(f64, @floatFromInt(t1 - t0)) * 1e-3);
        }
    }

    fn report(b: *Bench, lu_ctx: *GpuLu) void {
        const names = [_][]const u8{ "dev_refactor", "dev_solves", "dev_call_wall", "host_refactor", "host_solve", "kern_1thr", "kern_nthr", "dev_lsolve" };
        const t = lu_ctx.tab;
        const x = lu_ctx.idx;
        var wide: u32 = 0;
        for (0..t.n) |k| wide += @intFromBool(x[t.wlev + k] != x[t.wlev + k + 1]);
        // Head span: per level, the longest row list.
        var chains: [2]u64 = .{ 0, 0 };
        for ([_][3]u32{ .{ t.l_levels, t.l_lev, t.l_rows }, .{ t.u_levels, t.u_lev, t.u_rows } }, [_]u32{ t.l_ptr, t.u_ptr }, 0..) |lv, ptr, side| {
            for (0..lv[0]) |v| {
                var mx: u32 = 0;
                for (x[lv[1] + v]..x[lv[1] + v + 1]) |j| {
                    const r = x[lv[2] + j];
                    mx = @max(mx, x[ptr + r + 1] - x[ptr + r]);
                }
                chains[side] += mx;
            }
        }
        std.debug.print("gpu-lu-bench: n={d} F={d} l_levels={d} u_levels={d} head_chain={d}+{d} tail={d} gather_cols={d} samples={d} threads={d} checked={d} mismatches={d} (median us)", .{ lu_ctx.n, lu_ctx.flops, t.l_levels, t.u_levels, chains[0], chains[1], t.n - t.t0, wide, b.s[0].items.len, b.threads, if (lu_ctx.knobs.gpu_lu_check) lu_ctx.st.refactors else 0, lu_ctx.st.mismatches });
        for (&b.s, names) |*s, name| {
            std.mem.sort(f64, s.items, {}, std.sort.asc(f64));
            const m = if (s.items.len == 0) 0 else s.items[s.items.len / 2];
            std.debug.print(" {s}={d:.1}", .{ name, m });
        }
        std.debug.print("\n", .{});
    }

    fn deinit(b: *Bench, lu_ctx: *GpuLu) void {
        const gpa = lu_ctx.gpa;
        if (b.limit == 0 or b.s[0].items.len < b.limit) b.report(lu_ctx);
        for (&b.s) |*s| s.deinit(gpa);
        for ([_][]f64{ b.val, b.y, b.dx }) |p| gpa.free(p);
        gpa.free(b.sync);
        for (&b.ev) |*e| e.deinit();
        gpa.destroy(b);
    }
};

test "GpuLu.init without kernel images is NoGpuArtifacts" {
    if (comptime backend != null) return error.SkipZigTest;
    try std.testing.expectError(error.NoGpuArtifacts, GpuLu.init(std.testing.allocator, 4, 8, .{}));
}

test "GpuLu.admits: the flop bars and the empty matrix" {
    const admits = GpuLu.admits;
    const n: u32 = 20_000;
    try std.testing.expect(GpuLu.min_call_flops <= GpuLu.min_per_col * @as(u64, n));
    try std.testing.expect(admits(GpuLu.min_per_col * n, n, false));
    try std.testing.expect(!admits(GpuLu.min_per_col * n - 1, n, false));
    // A small dense refactor stays on the host whatever its F/n.
    try std.testing.expect(admits(GpuLu.min_call_flops, 1000, false));
    try std.testing.expect(!admits(GpuLu.min_call_flops - 1, 1000, false));
    // A big enough refactor goes whatever its F/n.
    const wide: u32 = 1_000_000;
    try std.testing.expect(GpuLu.min_flops < GpuLu.min_per_col * @as(u64, wide));
    try std.testing.expect(admits(GpuLu.min_flops, wide, false));
    try std.testing.expect(!admits(GpuLu.min_flops - 1, wide, false));
    // `ESPICE_GPU_LU=1` lifts the per-column bar.
    try std.testing.expect(admits(0, n, true));
    try std.testing.expect(admits(1 << 40, n, false));
    try std.testing.expect(!admits(0, 0, true));
}
