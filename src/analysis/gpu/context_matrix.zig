//! The resident Newton matrix (02 2.3). In: an eval's device planes and the
//! step's alpha. Out: A = G + alpha·C formed on the device, either brought
//! down in place of g and c (host LU) or left for the device LU
//! (`GpuLu.solveResident`), with only the diagonal's g and c gathered home.
//! Cites dev/solvers/gpu-lu.md (02 2.3).
//!
//! Contents:
//! - `pub const Gc = struct`
//! - `pub noinline fn evalGc(self: *GpuContext, x: []const f64, t: f64, sim: SimState, req: GcReq, diag: []const u32, nnz: u32) GcServed`
//! - `pub noinline fn gcSolve(self: *GpuContext, lu: *gpu_lu.GpuLu, slv: *direct.Solver, vals: []f64, rhs: []const f64, dx: []f64, alpha: f64) ?bool`
//! - `pub fn gcVals(self: *GpuContext, got: GcServed, alpha: f64, out: []f64, diag: []const u32, nnz: u32) []f64`
//! - `pub fn enqueueGcDown(self: *GpuContext, buf: u1, r: GcReq) !void`
//! - `pub fn enqueueCombine(self: *GpuContext, buf: u1, alpha: f64) !void`
//! - `pub fn gcScatter(self: *GpuContext, buf: u1, r: GcReq) void`

const std = @import("std");
const assert = std.debug.assert;
const gompute = @import("gompute");
const device_ir = @import("device").abi;
const direct = @import("solver").direct;
const walk = @import("../par_eval.zig");
const devices = @import("devices.zig");
const gpu_lu = @import("lu.zig");
const context = @import("context.zig");
const ctx_batch = @import("context_batch.zig");
const ctx_eval = @import("context_eval.zig");
const ctx_scatter = @import("context_scatter.zig");
const SimState = device_ir.SimState;
const backend = devices.backend;
const GpuContext = context.GpuContext;
const GcReq = context.GcReq;
const GcServed = context.GcServed;
const Raw = ctx_batch.Raw;
const Buffer = ctx_batch.Buffer;

/// The resident Newton matrix's buffers: the gather slots (the
/// diagonal, the ground pin's first), their g then c per plane buffer, and
/// A per plane buffer. Kernels from the LU image (`lu_device.zig`), launched
/// at `gc_block` threads.
pub const Gc = struct {
    combine: Raw,
    gather: Raw,
    d_k: Buffer,
    d_kv: Buffer,
    d_a: Buffer,
    /// Whole pinned allocations, sliced per plane buffer below.
    kv_all: []f64,
    a_all: []f64,
    pin_kv: [2][]f64,
    pin_a: [2][]f64,
    k: []const u32,
    nnz: u32,
    /// `ESPICE_GPU_LU_CHECK`: device-formed matrices compared with the
    /// host's, and how many differed.
    checks: u64 = 0,
    mismatches: u64 = 0,
};

/// `evalOnGpu` for a Newton iterate whose matrix is `G + req.alpha·C`
/// (`Circuit.evalNewtonGC`). Returns what the planes hold:
/// `full` when the query cannot take a resident matrix, `failed` as
/// `eval`'s false. `diag` (n slots) and `nnz` are the circuit's, the
/// same every call.
pub noinline fn evalGc(self: *GpuContext, x: []const f64, t: f64, sim: SimState, req: GcReq, diag: []const u32, nnz: u32) GcServed {
    self.query_hit = null;
    const t0 = ctx_eval.auditStart(self);
    defer ctx_eval.auditEnd(self, t0);
    self.sim = sim;
    const want: ?GcReq = if (gcInit(self, diag, nnz)) req else null;
    const got = ctx_eval.evalOnGpu(self, x, t, false, want) catch |e| {
        ctx_eval.evalFailed(self, e);
        return .failed;
    };
    const r = got orelse return .full;
    return if (r.device) .device else .host_a;
}

/// `Circuit.deviceSolve` after `evalGc` served `device`: the device LU
/// forms A from this eval's planes and `vals` on the diagonal
/// (`GpuLu.solveResident`). On a null or false return `vals` holds all
/// of A. A driver fault stamps g and c on the host at the eval's x and
/// combines them (`gcRescue`) before returning null.
pub noinline fn gcSolve(self: *GpuContext, lu: *gpu_lu.GpuLu, slv: *direct.Solver, vals: []f64, rhs: []const f64, dx: []f64, alpha: f64) ?bool {
    const gc = &self.gc.?;
    const c_plane = ctx_batch.view(self.d_planes, self.layout.off(.c) * @sizeOf(f64));
    const r: gpu_lu.Resident = .{ .alpha = alpha, .g = self.d_planes, .c = c_plane, .k = gc.d_k, .k_host = gc.k };
    const out = lu.solveResident(slv, vals, rhs, dx, r) catch {
        gcRescue(self, vals, alpha);
        return null;
    };
    if (self.lu_check and out == true) gcCheck(self, slv.vcopy, alpha, vals, gc.k);
    return out;
}

/// `lu_device.zig`'s `gc_block`.
const gc_block: u32 = 256;

/// `Circuit.newtonGC` after `evalGc` served `got` (`host_a` or
/// `device`), the ground pin added: under `host_a` the pinned A with
/// the ground pin's entry formed again from the planes; under `device`
/// `out` with the diagonal formed from the planes (the device LU forms
/// the rest, `gcSolve`). `diag` and `nnz` as `evalGc`'s.
pub fn gcVals(self: *GpuContext, got: GcServed, alpha: f64, out: []f64, diag: []const u32, nnz: u32) []f64 {
    const pl = self.planes();
    if (got == .host_a) {
        const a = self.gc.?.pin_a[self.cur];
        if (diag.len > 0) a[diag[0]] = pl.g_vals[diag[0]] + alpha * pl.c_vals[diag[0]];
        if (self.lu_check) gcCheck(self, a, alpha, &.{}, &.{});
        return a[0..nnz];
    }
    std.debug.assert(got == .device);
    for (diag) |s| out[s] = pl.g_vals[s] + alpha * pl.c_vals[s];
    return out;
}

/// Queues the planes a request needs into plane buffer `buf`: rhs and q
/// whole, g and c at the gather slots, and A unless the device LU forms
/// it. Under `ESPICE_GPU_LU_CHECK` the whole planes come down too.
pub fn enqueueGcDown(self: *GpuContext, buf: u1, r: GcReq) !void {
    const l = self.layout;
    const gc = &self.gc.?;
    const pb = self.pinnedBuf(buf);
    if (self.lu_check) {
        try ctx_eval.download(self, &self.d_planes, std.mem.sliceAsBytes(pb[0..self.n_cells]));
    } else {
        var rhs = ctx_batch.view(self.d_planes, l.off(.rhs) * @sizeOf(f64));
        try ctx_eval.download(self, &rhs, std.mem.sliceAsBytes(pb[l.off(.rhs)..][0..l.nr]));
        var q = ctx_batch.view(self.d_planes, l.off(.q) * @sizeOf(f64));
        try ctx_eval.download(self, &q, std.mem.sliceAsBytes(pb[l.off(.q)..][0..l.nr]));
    }
    const count = gcCount(self, r);
    if (count > 0) {
        var n_k: u64 = count;
        var c_plane = ctx_batch.view(self.d_planes, l.off(.c) * @sizeOf(f64));
        try gc.gather.launchOn(&self.stream, gompute.Dim3.linear(count, gc_block), .{ .x = gc_block }, 0, &.{
            gompute.interface.arg(&n_k), gc.d_k.argPtr(), self.d_planes.argPtr(), c_plane.argPtr(), gc.d_kv.argPtr(),
        });
        try ctx_eval.download(self, &gc.d_kv, std.mem.sliceAsBytes(gc.pin_kv[buf][0 .. 2 * count]));
    }
    if (!r.device) try enqueueCombine(self, buf, r.alpha);
}

/// Queues `A = g + alpha·c` over the last eval's planes into `gc.d_a`
/// and its download into plane buffer `buf`'s pinned A.
pub fn enqueueCombine(self: *GpuContext, buf: u1, alpha: f64) !void {
    const gc = &self.gc.?;
    if (gc.nnz == 0) return;
    var n: u64 = gc.nnz;
    var a = alpha;
    var c_plane = ctx_batch.view(self.d_planes, self.layout.off(.c) * @sizeOf(f64));
    try gc.combine.launchOn(&self.stream, gompute.Dim3.linear(gc.nnz, gc_block), .{ .x = gc_block }, 0, &.{
        gompute.interface.arg(&n), gompute.interface.arg(&a), self.d_planes.argPtr(), c_plane.argPtr(), gc.d_a.argPtr(),
    });
    try ctx_eval.download(self, &gc.d_a, std.mem.sliceAsBytes(gc.pin_a[buf]));
}

/// Writes the gathered g and c into plane buffer `buf`'s planes.
pub fn gcScatter(self: *GpuContext, buf: u1, r: GcReq) void {
    const gc = &self.gc.?;
    const pl = self.pinnedPlanes(buf);
    const count = gcCount(self, r);
    const kv = gc.pin_kv[buf];
    for (gc.k[0..count], kv[0..count], kv[count..][0..count]) |s, g, c| {
        pl.g_vals[s] = g;
        pl.c_vals[s] = c;
    }
}

// ---- private ----

/// Allocates `gc` on the first call; false for good when the query
/// cannot take a resident matrix: host batches stamp g and c anywhere,
/// the eval check compares whole planes, and without device charge
/// planes there is no c to combine.
fn gcInit(self: *GpuContext, diag: []const u32, nnz: u32) bool {
    if (self.gc != null) return !self.poisoned;
    if (self.gc_off) return false;
    self.gc_off = true;
    if (!self.fuse or self.poisoned or !self.resident_charge or self.cpu_batches.len != 0 or self.chk.len > 0) return false;
    self.gc = gcAlloc(self, diag, nnz) catch return false;
    self.gc_off = false;
    return true;
}

fn gcAlloc(self: *GpuContext, diag: []const u32, nnz: u32) !Gc {
    const be = backend.?;
    std.debug.assert(diag.len == self.n);
    var combine = try gompute.rawKernelByName(be, "arp_gc_combine", self.batches[0].kernel.context.ordinal);
    errdefer combine.deinit();
    var gather = try gompute.rawKernelByName(be, "arp_gc_gather", self.batches[0].kernel.context.ordinal);
    errdefer gather.deinit();
    const k0 = &self.batches[0].kernel;
    const nk: usize = @max(diag.len, 1);
    var d_k = try k0.alloc(nk * @sizeOf(u32));
    errdefer d_k.free();
    var d_kv = try k0.alloc(2 * nk * @sizeOf(f64));
    errdefer d_kv.free();
    var d_a = try k0.alloc(@max(nnz, 1) * @sizeOf(f64));
    errdefer d_a.free();
    const kv_all = try ctx_batch.pinnedF64(k0, 4 * nk);
    errdefer k0.freePinned(std.mem.sliceAsBytes(kv_all));
    const na: usize = @max(nnz, 1);
    const a_all = try ctx_batch.pinnedF64(k0, 2 * na);
    errdefer k0.freePinned(std.mem.sliceAsBytes(a_all));
    try ctx_eval.upload(self, &d_k, std.mem.sliceAsBytes(diag));
    try ctx_eval.sync(self, .other);
    return .{
        .combine = combine,
        .gather = gather,
        .d_k = d_k,
        .d_kv = d_kv,
        .d_a = d_a,
        .kv_all = kv_all,
        .a_all = a_all,
        .pin_kv = .{ kv_all[0 .. 2 * nk], kv_all[2 * nk ..] },
        .pin_a = .{ a_all[0..nnz], a_all[na..][0..nnz] },
        .k = diag,
        .nnz = nnz,
    };
}

/// Gather slots a request downloads: the whole diagonal under the device
/// LU, else only the ground pin's.
fn gcCount(self: *const GpuContext, r: GcReq) usize {
    return if (r.device) self.gc.?.k.len else @min(self.gc.?.k.len, 1);
}

/// `gcSolve`'s fault path: g and c stamped on the host at the last
/// eval's x into scratch, combined into `vals` off the diagonal, which
/// keeps the host's own entries. Tolerance class (the host stamp's sum
/// order), as every fault fallback.
fn gcRescue(self: *GpuContext, vals: []f64, alpha: f64) void {
    const l = self.layout;
    const scratch = self.gpa.alloc(f64, 2 * l.ng + 2 * l.nr) catch @panic("OOM in the device-LU fault fallback");
    defer self.gpa.free(scratch);
    @memset(scratch, 0);
    const pl = ctx_scatter.scratchPlanes(scratch, l.ng, l.nr);
    const x = self.pin_x[self.cur][0..self.x_len];
    for (self.all_batches) |b| walk.stampRange(b, &pl, 0, b.count, x, self.last_t, .full);
    const gc = &self.gc.?;
    const keep = self.gpa.alloc(f64, gc.k.len) catch @panic("OOM in the device-LU fault fallback");
    defer self.gpa.free(keep);
    for (gc.k, keep) |s, *v| v.* = vals[s];
    for (vals[0..gc.nnz], pl.g_vals[0..gc.nnz], pl.c_vals[0..gc.nnz]) |*v, g, c| v.* = g + alpha * c;
    for (gc.k, keep) |s, v| vals[s] = v;
}

/// `ESPICE_GPU_LU_CHECK` (the whole planes came down too): compares the
/// device-formed `a` with `G + alpha·C` from the planes, overlaid with
/// `over` at `slots`, bitwise.
fn gcCheck(self: *GpuContext, a: []const f64, alpha: f64, over: []const f64, slots: []const u32) void {
    const gc = &self.gc.?;
    const pl = self.planes();
    gc.checks += 1;
    const ref = self.gpa.alloc(f64, gc.nnz) catch return;
    defer self.gpa.free(ref);
    for (ref, pl.g_vals[0..gc.nnz], pl.c_vals[0..gc.nnz]) |*v, g, c| v.* = g + alpha * c;
    for (slots) |s| ref[s] = over[s];
    for (ref, a[0..gc.nnz], 0..) |u, v, i| if (@as(u64, @bitCast(u)) != @as(u64, @bitCast(v))) {
        gc.mismatches += 1;
        if (gc.mismatches <= 3) std.debug.print("gpu-gc check: A slot {d} differs from the host ({e} vs {e})\n", .{ i, v, u });
        return;
    };
}
