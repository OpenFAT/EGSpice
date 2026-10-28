//! GPU device evaluation feeding the host solver: the query's resident
//! device state and its entry points. In: the analysis circuit's batches
//! and planes (`Host`). Out: the planes stamped on the device, plus the
//! limit, state and accepted-step passes of the resident batches. Every
//! eligible batch's models, instances and tapes are uploaded once and stay
//! resident; per Newton iteration the bus carries `x` up and one contiguous
//! block of value planes down, and the Newton update and convergence test
//! stay on the CPU (the device LU is lu.zig's). That round trip is the
//! floor, so the GPU pays off on device work, not circuit size; `init`
//! prices both sides (context_cost.zig) before `auto` commits.
//!
//! The entry points live beside this file, one transform each:
//! context_eval.zig (the eval and the charge pass), context_matrix.zig (the
//! resident Newton matrix), context_state.zig (limit, state and latch
//! passes), context_sync.zig (Instance merges), context_scatter.zig (the
//! deterministic scatter), context_batch.zig (one resident batch).
//! Cites #15, #93, #126 and #139 for the decisions marked in the bodies.
//!
//! Contents:
//! - `pub const available: bool`
//! - `pub const Request = enum { cpu, auto, cuda, hip }`
//! - `pub const Decline = enum { policy, capability, machine }`
//! - `pub const InitError = Error || std.mem.Allocator.Error || gompute.Error`
//! - `pub const Error = error{ NoGpuArtifacts, CircuitNotEligible, NotEnoughGpuWork, ImageNotCompiled }`
//! - `pub const Driver = struct` (`prestart`, `started`, `deinit`)
//! - `pub const GcReq = struct { alpha, device }`
//! - `pub const GcServed = enum { failed, full, host_a, device }`
//! - `pub const Host = struct`
//! - `pub const GpuContext = struct` (`init`, `planes`, `restorePlanes`, `deinit`, `pinnedBuf`, `pinnedPlanes`, `simAt`,
//!   `Follow`, `Pre`, and the entry points `eval`, `evalQ`, `syncQTape`, `evalGc`, `gcVals`, `gcSolve`, `applyLimits`,
//!   `evalFollows`, `updateStates`, `acceptStates`, `clearLimits`, `stateCtl`, `seedJunctions`, `syncStatus`,
//!   `syncHostState`, `markDirty`)
//! - `pub const test_access`
//! - `pub fn declineKind(e: InitError) Decline`
//! - `pub fn fallsBack(req: Request, explicit: bool, e: InitError) bool`
//! - `pub fn detectedHardware() []const u8`
//! - `pub fn requestSupported(req: Request) bool`
//! - `pub fn nowNs() u64`

const std = @import("std");
const assert = std.debug.assert;
const gompute = @import("gompute");
const device_ir = @import("device").abi;
const Knobs = @import("../Circuit.zig").Knobs;
const devices = @import("devices.zig");
const hdl = @import("hdl.zig");
const ctx_batch = @import("context_batch.zig");
const ctx_cost = @import("context_cost.zig");
const ctx_eval = @import("context_eval.zig");
const ctx_matrix = @import("context_matrix.zig");
const ctx_scatter = @import("context_scatter.zig");
const ctx_state = @import("context_state.zig");
const ctx_stats = @import("context_stats.zig");
const ctx_sync = @import("context_sync.zig");

const Planes = device_ir.Planes;
const SimState = device_ir.SimState;
const backend = devices.backend;
const Buffer = ctx_batch.Buffer;
const Stream = ctx_batch.Stream;
const Raw = ctx_batch.Raw;
const BatchGpu = ctx_batch.BatchGpu;
const Layout = ctx_scatter.Layout;
const Order = ctx_scatter.Order;
const Gc = ctx_matrix.Gc;
const Audit = ctx_cost.Audit;
const Prof = ctx_stats.Prof;

/// This binary carries kernel images, so a context can exist at all.
pub const available = backend != null;

/// What `--backend` can ask for. `auto` may decline for any reason and fall
/// back to the CPU; `cuda`/`hip` are explicit and either run there or fail
/// saying why (see `fallsBack`).
pub const Request = enum { cpu, auto, cuda, hip };

/// Why `init` refused, in the three kinds a caller acts on differently.
pub const Decline = enum {
    /// A performance estimate: the CPU is priced faster, or a heavy image
    /// is not compiled yet. An explicit request overrides it, so only
    /// `auto` sees one.
    policy,
    /// Nothing in this circuit has a kernel image. A reported CPU fallback
    /// under `auto`; `cuda`/`hip` fail.
    capability,
    /// The driver, hardware or build said no (including a build with no
    /// images). An explicit request fails naming the detected hardware.
    machine,
};

/// Why `GpuContext.init` failed: one of `Error`, `error.OutOfMemory`, or a
/// driver error gompute reports (bringing up the device, loading a module,
/// allocating or copying device memory). `declineKind` sorts them.
pub const InitError = Error || std.mem.Allocator.Error || gompute.Error;

/// The refusals `GpuContext.init` makes on its own, before or instead of a
/// driver error; `declineKind` sorts them.
pub const Error = error{
    /// The build machine had no GPU, so no kernel images were emitted.
    NoGpuArtifacts,
    /// No device in this circuit has a kernel image (see `eval.gpuEligible`).
    CircuitNotEligible,
    /// The cost model priced the CPU faster.
    NotEnoughGpuWork,
    /// `auto` met a heavy image the driver has not compiled yet. The whole
    /// query runs on the CPU rather than leave that one model behind (#139).
    ImageNotCompiled,
};

/// Whether a context came up, so `auto`'s cost model stops pricing the
/// driver setup. The facade owns one per Problem and lends it to every query
/// (`Config.gpu_driver`). Queries on other threads read it while one sets
/// it: monotonic is enough, since a stale read only overprices one query
/// (#126).
pub const Driver = struct {
    up: std.atomic.Value(bool) = .init(false),
    /// `prestart`'s thread; `deinit` joins it.
    warm: ?std.Thread = null,

    /// Brings the driver and device 0's primary context up on a thread,
    /// so the ~350 ms start overlaps the frontend instead of the
    /// first query. The primary context stays retained for the process
    /// (gompute's `Compute.deinit` drops the handle only); a failure here
    /// leaves the first `GpuContext.init` to report it. Call once, before
    /// any query; pair with `deinit`.
    pub fn prestart(self: *Driver) void {
        if (comptime backend == null) return;
        std.debug.assert(self.warm == null);
        self.warm = std.Thread.spawn(.{}, warmUp, .{self}) catch null;
    }

    /// The driver is up, or `prestart` is bringing it up beside the
    /// frontend: either way a query no longer pays the start. `warm` is
    /// set before the first query and cleared after the last.
    pub fn started(self: *const Driver) bool {
        return self.warm != null or self.up.load(.monotonic);
    }

    /// Joins `prestart`'s thread, if any.
    pub fn deinit(self: *Driver) void {
        if (self.warm) |t| t.join();
        self.warm = null;
    }

    // ---- private ----

    fn warmUp(self: *Driver) void {
        var compute = gompute.runtime.dynamic.Compute.init(if (backend.? == .cuda) .cuda else .hip) catch return;
        compute.deinit();
        self.up.store(true, .monotonic);
        // Every other device's context too: the device table opens each one
        // (multi-GPU stage 1). ponytail: one after another; a thread per
        // device when a box has many.
        _ = devices.table();
    }
};

/// One Newton eval's matrix request: A = G + alpha·C. Under
/// `device` the device LU forms A from `d_planes` (`GpuLu.solveResident`)
/// and the host keeps g and c on the diagonal only; otherwise the device
/// forms A here and A comes down in place of g and c.
pub const GcReq = struct { alpha: f64, device: bool };

/// What `GpuContext.evalGc` left in the circuit's planes: `failed` as
/// `eval`'s false, `full` whole planes (the host combines), `host_a` A in
/// `gcVals`'s pinned buffer, `device` A on the device for `gcSolve`.
pub const GcServed = enum { failed, full, host_a, device };

/// What `GpuContext.init` reads off the analysis circuit. The context keeps
/// `batches` (borrowed: the circuit's own, which outlive it) and the plane
/// allocations, and copies the rest.
pub const Host = struct {
    /// Every batch, resident or not; the fallback walks run over all of them.
    batches: []const device_ir.Batch,
    /// The circuit's own planes, with their current values.
    planes: Planes,
    n: u32,
    trash_slot: u32,
    has_charge: bool,
    knobs: Knobs,
};

pub const GpuContext = struct {
    gpa: std.mem.Allocator,
    /// The circuit's batches (`Host.batches`), for the fallback walks.
    all_batches: []const device_ir.Batch,
    n: u32,
    /// Some batch, resident or not, carries charge (`Host.has_charge`).
    has_charge: bool,
    /// The analysis state of the entry point running now, for the launches.
    sim: SimState = .{},
    /// The eligible batches, resident on the device.
    batches: []BatchGpu,
    /// Everything else: whatever `eval.gpuEligible` turns down, plus eligible
    /// batches demoted for a missing or cold image. They stamp on the host.
    cpu_batches: []const device_ir.Batch,
    /// The resident host-stepped batches (`BatchGpu.stepped`), which the
    /// state walks visit beside `cpu_batches`. A view of `cpu_owned`'s tail.
    stepped_batches: []const device_ir.Batch,
    /// Full-length allocations behind `batches` and `cpu_batches`. Demotions
    /// are only known after loading, so both are sized for the worst case and
    /// narrowed; `deinit` frees these, not the narrowed views.
    batches_owned: []BatchGpu,
    cpu_owned: []device_ir.Batch,

    /// The circuit's own plane allocations, swapped out for `pin_planes` while
    /// the context lives and handed back (with the last values) by
    /// `restorePlanes`.
    host_planes: Planes,
    /// Two page-locked landing areas in `Layout` order, back to back. The
    /// circuit's planes point into buffer `cur`, so the one download per eval
    /// is the whole plane update and the host batches stamp on top after it
    /// lands: no host clear, no host add. A prefetched eval (`pre`) lands
    /// in the other buffer, because the converger still reads this one's
    /// residual after the limit pass that launched it.
    pin_planes: []f64,
    cur: u1 = 0,
    layout: Layout,
    /// Page-locked staging for the eval's `x` (one per plane buffer: buffer
    /// `cur`'s holds the x of the last eval served, the other a prefetch's
    /// key), the limit pass's iterate and `x_old`, the flag word and the
    /// concatenated charge tapes.
    pin_x: [2][]f64,
    pin_x2: []f64,
    pin_xold: []f64,
    pin_flags: []u8,
    pin_tape: []f64,

    /// The eval's x, paired with `pin_x`: `d_x[cur]` holds the x of the last
    /// eval served, which the lazy tape pass reuses. The limit pass takes its
    /// iterate in `d_x2` and `x_old` in `d_xold`.
    d_x: [2]Buffer,
    d_xold: Buffer,
    d_x2: Buffer,
    /// The flag words the state kernels OR into: word 0 bit 0 limited, bit 1
    /// reject requested; word 1 the accepted-state query `acceptStates`
    /// fuses.
    d_flags: Buffer,
    d_planes: Buffer,

    /// Deterministic scatter. Kernels write each contribution to its own
    /// staging cell, and `reduce` sums each plane cell's run of cells in the
    /// order the serial CPU stamp visits them.
    ///
    /// Atomic accumulation reorders the sum from pass to pass. On
    /// parallel_inverters_2000 the Vdd row takes 8000 contributions that
    /// cancel to 3.6e-9; two replays at the same x differed by 2.1e-10, the
    /// converger rejected every iterate and dt underflowed. In tape order each
    /// instance's +1.8 meets its own -1.8 two entries later and the sum is
    /// exact.
    ///
    /// Built beside the frozen tapes, not through them: `d_slots` and
    /// `d_rhs_idx` keep their `[id][ru][cu]` u32 layout and length; only their
    /// values change from plane index to staging index.
    reduce: Raw,
    /// Staging for all planes back to back (`Order`), and views of it at each
    /// plane's base, which the eval kernels take as their plane pointers.
    d_stage: Buffer,
    v_stage: [4]Buffer,
    d_seg1: Buffer,
    d_seg2: Buffer,
    d_mid: Buffer,
    n_stage: usize,
    n_pieces: usize,
    n_cells: usize,
    /// Slot cells the constant batches stamp (`BatchGpu.const_kernel`), in
    /// order; their g then c baseline values sit at stage `n_stage` on,
    /// each a level-1 piece of its own ahead of the cell's run.
    base_cells: []u32,
    /// The q plane's first level-1 piece; its pieces run to `n_pieces`.
    q_piece0: usize,
    /// Every resident batch with charge has `q_kernel` and the device has a
    /// q plane, so `evalQ` runs charges only.
    q_only: bool,

    /// `pin_tape` holds the tape for plane buffer `tape_buf`'s eval; null
    /// when it holds nothing current. Host tapes then match with a copy.
    tape_buf: ?u1 = null,
    /// The host tapes match `d_x[cur]`; cleared by every eval.
    tape_fresh: bool = false,
    /// Set by `evalFollows`, consumed by the next limit pass or state query,
    /// and dropped by every other hook.
    follow: ?Follow = null,
    /// A prefetched eval at `pin_x[cur ^ 1]` sits in plane buffer `cur ^ 1`.
    /// Any hook that changes device state drops it.
    pre: ?Pre = null,
    /// Every resident batch with a host `q_tape` has a tape kernel, so the
    /// transient can keep its per-device LTE (`Circuit.qTapeLen`).
    tape_ok: bool,
    last_t: f64 = 0,

    /// A resident device requested a step reject, which the GPU path cannot
    /// deliver, or a tape sync faulted. Every later eval, limit and state
    /// call takes the host path.
    poisoned: bool = false,
    /// Host parameters changed; re-upload models and instances before the
    /// next launch. Lazy, so a pair of writes costs one upload.
    params_dirty: bool = false,

    /// Every copy and launch goes on this stream, never the NULL stream,
    /// which would synchronize with the copies this overlaps. One stream is
    /// enough: an iteration is a strict chain.
    stream: Stream,

    /// One-shot fallback warnings, one per failure kind so neither silences
    /// the other.
    warned_eval: bool = false,
    warned_state: bool = false,

    /// `evalCheck` scratch (a copy of the device planes); non-empty only
    /// under `ESPICE_GPU_EVAL_CHECK`.
    chk: []f64 = &.{},
    /// Worst reproducibility gap printed so far.
    chk_worst: f64 = 0,

    /// Some resident batch produces charge. When none does, the device c/q
    /// planes would only carry zeros, so they are neither reduced nor
    /// downloaded; the host batches stamp them alone.
    resident_charge: bool,

    /// `auto` admitted this query on a price: `Audit` holds the device to it.
    audit: ?Audit = null,
    /// `Audit` found the device losing and moved the rest of the query to
    /// the host, state included: every entry point walks on the host,
    /// quietly, and `syncHostState` has nothing left to bring home.
    demoted: bool = false,

    /// The device half of the next `stateCtl(.query)`, which `acceptStates`
    /// ran on its wait; dropped by every other entry point.
    query_hit: ?bool = null,

    prof: Prof = .{},
    /// False under `ESPICE_GPU_NOFUSE`: the unfused sequence, for A/B runs.
    /// Read once at `init`, since `evalFollows` fires every iteration.
    fuse: bool = true,

    /// The resident Newton matrix (`evalGc`), allocated on the first
    /// request; null before, and for good once `gc_off`.
    gc: ?Gc = null,
    gc_off: bool = false,
    /// `ESPICE_GPU_LU_CHECK`, read once at `init`.
    lu_check: bool = false,
    /// Length of the last eval's x (`gcRescue` restamps at it).
    x_len: usize = 0,
    /// What each plane buffer's last eval left on the host: null for whole
    /// planes, else the request it served (rhs and q whole, g and c only at
    /// the gathered slots).
    gc_buf: [2]?GcReq = .{ null, null },
    /// The request of the last Newton eval, which a prefetch inherits.
    gc_last: ?GcReq = null,

    const Self = @This();

    pub const eval = ctx_eval.eval;
    pub const evalQ = ctx_eval.evalQ;
    pub const syncQTape = ctx_eval.syncQTape;
    pub const evalGc = ctx_matrix.evalGc;
    pub const gcVals = ctx_matrix.gcVals;
    pub const gcSolve = ctx_matrix.gcSolve;
    pub const applyLimits = ctx_state.applyLimits;
    pub const evalFollows = ctx_state.evalFollows;
    pub const updateStates = ctx_state.updateStates;
    pub const acceptStates = ctx_state.acceptStates;
    pub const clearLimits = ctx_state.clearLimits;
    pub const stateCtl = ctx_state.stateCtl;
    pub const seedJunctions = ctx_state.seedJunctions;
    pub const syncStatus = ctx_sync.syncStatus;
    pub const syncHostState = ctx_sync.syncHostState;
    pub const markDirty = ctx_sync.markDirty;

    /// `sim` is the analysis state the announced eval runs under: a
    /// model that reads `$simparam("iteration")` stamps differently at the
    /// next iterate (the Meyer caps at iteration 1).
    pub const Follow = struct { x: []const f64, sim: SimState, charge: bool };

    /// A prefetched eval's key (`pre`).
    pub const Pre = struct { sim: SimState, charge: bool, gc: ?GcReq };

    /// Uploads every eligible batch and keeps it resident. The caller
    /// classifies a failure with `declineKind`.
    ///
    /// `explicit` (the user named the GPU) skips the cost model and loads
    /// cold images; otherwise `init` prices the query both ways (`Cost`,
    /// from `evals`, the caller's estimate of its device evals, and
    /// `threads` host lanes) and declines before the first driver call when
    /// the CPU wins. The `auto` probe evaluates the eligible batches three
    /// times on the host, which rewrites their `q_tape`.
    ///
    /// On success the page-locked planes (`planes`) hold `host.planes`'
    /// values, ready for `Circuit.attachGpu`; `restorePlanes` copies them
    /// back. A success marks `driver` up, which later queries' cost models
    /// read; a null `driver` prices the setup every time. Caller owns the
    /// result and frees it with `deinit`.
    pub fn init(gpa: std.mem.Allocator, host: Host, driver: ?*Driver, explicit: bool, threads: u32, evals: f64) InitError!*Self {
        if (comptime backend == null) return Error.NoGpuArtifacts;
        const knobs = host.knobs;
        const report = explicit or knobs.gpu_stats != .off;

        var n_gpu: usize = 0;
        for (host.batches) |b| {
            if (b.hooks.gpu_payload != null) n_gpu += 1;
        }
        if (n_gpu == 0) return Error.CircuitNotEligible;
        const auto_price: ?ctx_cost.Price = if (explicit) null else try ctx_cost.admit(gpa, host, threads, evals, if (driver) |d| d.started() else false);

        const self = try gpa.create(Self);
        errdefer gpa.destroy(self);

        const batches = try gpa.alloc(BatchGpu, n_gpu);
        errdefer gpa.free(batches);
        // Sized for every batch: an eligible model with no image, or a cold
        // one under `auto`, demotes into this array below.
        const cpu_batches = try gpa.alloc(device_ir.Batch, host.batches.len);
        errdefer gpa.free(cpu_batches);

        var n_up: usize = 0;
        var n_cpu: usize = 0;
        var resident_charge = false;
        var tape_ok = true;
        var tape_len: usize = 0;
        errdefer for (batches[0..n_up]) |*bg| bg.deinit(gpa);

        var n_stepped: usize = 0;
        var q_only = true;
        for (host.batches) |b| {
            // Host-only batches first: once every eligible batch is up,
            // `batches[n_up]` is one past the end (#93).
            if (b.hooks.gpu_payload == null) {
                cpu_batches[n_cpu] = b;
                n_cpu += 1;
                continue;
            }
            const up = try ctx_batch.loadBatch(gpa, &batches[n_up], b, explicit, report, knobs);
            if (!up) {
                cpu_batches[n_cpu] = b;
                n_cpu += 1;
                continue;
            }
            const bg = &batches[n_up];
            if (bg.stepped) {
                n_stepped += 1;
                cpu_batches[cpu_batches.len - n_stepped] = b;
            }
            resident_charge = resident_charge or b.has_charge;
            if (b.hooks.q_tape != null and bg.qtp_kernel == null) tape_ok = false;
            if (bg.has_charge and bg.q_kernel == null) q_only = false;
            tape_len += bg.tape.len;
            n_up += 1;
        }

        // Every eligible batch demoted; `batches[0]` below needs one.
        if (n_up == 0) return Error.CircuitNotEligible;
        if (knobs.gpu_stats != .off) std.debug.print(
            "gpu-stats: resident batches={d} (host-stepped {d}) host batches={d} charge planes={s} lte tape={s}\n",
            .{ n_up, n_stepped, n_cpu, if (resident_charge) "device" else "host-only", if (tape_ok) "device" else "row-plane" },
        );

        // Every handle allocates from the same primary context, so buffers
        // made through batch 0 are valid in every launch.
        const k0 = &batches[0].kernel;
        const layout: Layout = .{ .ng = host.planes.g_vals.len, .nr = host.planes.rhs.len };
        const n_planes: usize = if (resident_charge) 4 else 2;
        const x_bytes = (host.n + 1) * @sizeOf(f64);

        const base_cells = try ctx_scatter.baseCells(gpa, batches[0..n_up], host.trash_slot, layout.ng);
        errdefer gpa.free(base_cells);
        const order = blk: {
            const tapes = try gpa.alloc([]const u32, 2 * n_up);
            defer gpa.free(tapes);
            for (batches[0..n_up], tapes[0..n_up], tapes[n_up..]) |*bg, *st, *rt| {
                const p = bg.payload(bg.ctx);
                // A constant batch's g/c live in the baseline, not the stage.
                st.* = if (bg.const_kernel != null) &.{} else p.slots;
                rt.* = p.rhs_idx;
            }
            break :blk try Order.build(gpa, tapes[0..n_up], tapes[n_up..], host.trash_slot, host.n, layout, n_planes, base_cells);
        };
        defer order.deinit(gpa);
        // Same image as the eval kernels, so a miss is a stale build: fatal.
        var reduce = try hdl.kernel(backend.?, @ptrCast(batches[0].payload), batches[0].payload(batches[0].ctx).reduce_kernel);
        errdefer reduce.deinit();
        var stream = try batches[0].kernel.createStream();
        errdefer stream.deinit();
        try order.upload(batches[0..n_up], &stream);

        // One errdefer per resource, so a failure part way through releases
        // exactly what was made. Pinned memory goes through batch 0's handle,
        // which the batch errdefer above releases after these run.
        const pin_planes = try ctx_batch.pinnedF64(k0, 2 * layout.cells(4));
        errdefer k0.freePinned(std.mem.sliceAsBytes(pin_planes));
        const pin_x0 = try ctx_batch.pinnedF64(k0, host.n + 1);
        errdefer k0.freePinned(std.mem.sliceAsBytes(pin_x0));
        const pin_x1 = try ctx_batch.pinnedF64(k0, host.n + 1);
        errdefer k0.freePinned(std.mem.sliceAsBytes(pin_x1));
        const pin_xold = try ctx_batch.pinnedF64(k0, host.n + 1);
        errdefer k0.freePinned(std.mem.sliceAsBytes(pin_xold));
        const pin_x2 = try ctx_batch.pinnedF64(k0, host.n + 1);
        errdefer k0.freePinned(std.mem.sliceAsBytes(pin_x2));
        // Word 0 the limit and state verdicts, word 1 a fused query's.
        const pin_flags = try k0.allocPinned(8);
        errdefer k0.freePinned(pin_flags);
        const pin_tape: []f64 = if (tape_len > 0) try ctx_batch.pinnedF64(k0, tape_len) else &.{};
        errdefer if (pin_tape.len > 0) k0.freePinned(std.mem.sliceAsBytes(pin_tape));
        var d_x0 = try k0.alloc(x_bytes);
        errdefer d_x0.free();
        var d_x1 = try k0.alloc(x_bytes);
        errdefer d_x1.free();
        var d_xold = try k0.alloc(x_bytes);
        errdefer d_xold.free();
        var d_x2 = try k0.alloc(x_bytes);
        errdefer d_x2.free();
        var d_flags = try k0.alloc(8);
        errdefer d_flags.free();
        var d_planes = try k0.alloc(layout.cells(n_planes) * @sizeOf(f64));
        errdefer d_planes.free();
        var d_stage = try k0.alloc(@max(order.n_stage + order.n_base, 1) * @sizeOf(f64));
        errdefer d_stage.free();
        var d_seg1 = try ctx_batch.uploadBytes(k0, std.mem.sliceAsBytes(order.seg1));
        errdefer d_seg1.free();
        var d_seg2 = try ctx_batch.uploadBytes(k0, std.mem.sliceAsBytes(order.seg2));
        errdefer d_seg2.free();
        var d_mid = try k0.alloc(@max(order.n_pieces, 1) * @sizeOf(f64));
        errdefer d_mid.free();
        const chk: []f64 = if (knobs.gpu_eval_check)
            try gpa.alloc(f64, layout.cells(n_planes))
        else
            &.{};

        if (knobs.gpu_stats != .off) std.debug.print("gpu-stats: stage={d} pieces={d} cells={d}\n", .{ order.n_stage, order.n_pieces, layout.cells(n_planes) });
        // The eval kernels store every staging cell they own each pass; the
        // cells no pass writes keep this zero.
        try d_stage.fillAsync(0, @max(order.n_stage, 1) * @sizeOf(f64), &stream);
        // Charge-free planes read zero until a host batch stamps them.
        @memset(pin_planes, 0);
        var v_stage: [4]Buffer = undefined;
        for (&v_stage, order.stage_base) |*v, base| v.* = ctx_batch.view(d_stage, base * @sizeOf(f64));

        self.* = .{
            .gpa = gpa,
            .all_batches = host.batches,
            .n = host.n,
            .has_charge = host.has_charge,
            .batches = batches[0..n_up],
            .cpu_batches = cpu_batches[0..n_cpu],
            .stepped_batches = cpu_batches[cpu_batches.len - n_stepped ..],
            .batches_owned = batches,
            .cpu_owned = cpu_batches,
            .host_planes = host.planes,
            .pin_planes = pin_planes,
            .layout = layout,
            .pin_x = .{ pin_x0, pin_x1 },
            .pin_xold = pin_xold,
            .pin_x2 = pin_x2,
            .pin_flags = pin_flags,
            .pin_tape = pin_tape,
            .d_x = .{ d_x0, d_x1 },
            .d_xold = d_xold,
            .d_x2 = d_x2,
            .d_flags = d_flags,
            .d_planes = d_planes,
            .reduce = reduce,
            .d_stage = d_stage,
            .v_stage = v_stage,
            .d_seg1 = d_seg1,
            .d_seg2 = d_seg2,
            .d_mid = d_mid,
            .n_stage = order.n_stage,
            .n_pieces = order.n_pieces,
            .n_cells = layout.cells(n_planes),
            .q_piece0 = order.q_piece0,
            .base_cells = base_cells,
            .q_only = q_only and resident_charge,
            .tape_ok = tape_ok,
            .stream = stream,
            .resident_charge = resident_charge,
            .chk = chk,
            .prof = .{ .on = knobs.gpu_stats != .off, .phases = knobs.gpu_stats == .phases },
            .fuse = !knobs.gpu_nofuse,
            .lu_check = knobs.gpu_lu_check,
            .audit = if (auto_price) |p| .{ .cpu_us = p.cpu_us, .key = p.key, .knobs = knobs } else ctx_cost.CostCache.learn(host, threads, evals, knobs),
        };
        if (driver) |d| d.up.store(true, .monotonic);
        // A failure here unwinds through the errdefers above.
        try ctx_eval.uploadBase(self);
        // The pinned planes start from the circuit's values.
        ctx_scatter.copyPlanes(self.pinnedPlanes(0), host.planes);
        return self;
    }

    /// Plane buffer `buf` (all four planes) within `pin_planes`.
    pub fn pinnedBuf(self: *const Self, buf: u1) []f64 {
        const n = self.layout.cells(4);
        return self.pin_planes[@as(usize, buf) * n ..][0..n];
    }

    /// The four planes as slices of plane buffer `buf`.
    pub fn pinnedPlanes(self: *const Self, buf: u1) Planes {
        const l = self.layout;
        const b = self.pinnedBuf(buf);
        return .{
            .g_vals = b[l.off(.g)..][0..l.ng],
            .rhs = b[l.off(.rhs)..][0..l.nr],
            .c_vals = b[l.off(.c)..][0..l.ng],
            .q_vec = b[l.off(.q)..][0..l.nr],
        };
    }

    /// The planes the last eval wrote, which the attached circuit points at
    /// (`Circuit.attachGpu`, `Circuit.evalGpu`).
    pub fn planes(self: *const Self) Planes {
        return self.pinnedPlanes(self.cur);
    }

    /// Drains the stream and copies the current planes into the circuit's
    /// own allocations (`Host.planes`), which it returns for
    /// `Circuit.detachGpu` to rebind. Call once, before `deinit`.
    pub fn restorePlanes(self: *Self) Planes {
        if (comptime backend == null) return self.host_planes;
        // `.commit`/`.revert` latches stay queued unwaited. A fault here was
        // already reported or is moot.
        self.stream.synchronize() catch {};
        // ponytail: after a resident Newton eval (`evalGc`) g and c hold the
        // diagonal only; every reader of the circuit's planes evaluates
        // first. A reader that does not would need the planes downloaded
        // again here, while `d_planes` still holds them.
        ctx_scatter.copyPlanes(self.host_planes, self.planes());
        return self.host_planes;
    }

    /// Frees every device and pinned buffer and `self`. Detach it first
    /// (`Circuit.detachGpu`, which takes the planes back). Device-side state
    /// is dropped; call `syncStatus` or `syncHostState` first to keep it.
    pub fn deinit(self: *Self) void {
        if (comptime backend == null) return;
        self.prof.print();
        // `.commit`/`.revert` latches stay queued unwaited; drain them before
        // their buffers go. A fault here was already reported or is moot.
        self.stream.synchronize() catch {};
        // Pinned memory first, while batch 0's handle that owns it is alive.
        const k0 = &self.batches[0].kernel;
        if (self.gc) |*gc| {
            if (self.lu_check) std.debug.print("gpu-gc: checked={d} mismatches={d}\n", .{ gc.checks, gc.mismatches });
            k0.freePinned(std.mem.sliceAsBytes(gc.kv_all));
            k0.freePinned(std.mem.sliceAsBytes(gc.a_all));
            for ([_]*Buffer{ &gc.d_k, &gc.d_kv, &gc.d_a }) |b| b.free();
            gc.combine.deinit();
            gc.gather.deinit();
        }
        for ([_][]f64{ self.pin_planes, self.pin_x[0], self.pin_x[1], self.pin_x2, self.pin_xold, self.pin_tape }) |p| {
            if (p.len > 0) k0.freePinned(std.mem.sliceAsBytes(p));
        }
        k0.freePinned(self.pin_flags);
        self.stream.deinit();
        for ([_]*Buffer{ &self.d_x[0], &self.d_x[1], &self.d_xold, &self.d_x2, &self.d_flags, &self.d_planes, &self.d_stage, &self.d_seg1, &self.d_seg2, &self.d_mid }) |b| b.free();
        self.reduce.deinit();
        for (self.batches) |*bg| bg.deinit(self.gpa);
        if (self.chk.len > 0) self.gpa.free(self.chk);
        self.gpa.free(self.base_cells);
        self.gpa.free(self.batches_owned);
        self.gpa.free(self.cpu_owned);
        self.gpa.destroy(self);
    }

    /// The circuit's analysis state at eval time `t`, as the host batches
    /// build it (`DeviceBatch.simAt`).
    pub fn simAt(self: *const Self, t: f64) SimState {
        var sim = self.sim;
        sim.t = t;
        return sim;
    }
};

/// Private access for the analysis test suite.
pub const test_access = if (@import("builtin").is_test) .{
    .backend = backend,
} else {};

/// Classifies an `init` error. Anything outside `Error`, a driver or
/// allocation failure included, is `.machine`.
pub fn declineKind(e: InitError) Decline {
    return switch (e) {
        Error.NotEnoughGpuWork, Error.ImageNotCompiled => .policy,
        Error.CircuitNotEligible => .capability,
        else => .machine, // else: every driver and allocator error is the machine saying no
    };
}

/// Whether a query runs on the CPU after `GpuContext.init` failed with
/// `e`. `cuda` and `hip` never fall back. `auto` does unless `explicit` (the
/// user named the GPU) and the machine said no.
pub fn fallsBack(req: Request, explicit: bool, e: InitError) bool {
    return switch (req) {
        .cpu => true,
        .auto => !explicit or declineKind(e) != .machine,
        .cuda, .hip => false,
    };
}

/// Returns "cuda", "hip" or "none": the first driver that brings up a device
/// on this machine, CUDA first, whatever this binary carries. Costs a driver
/// init and retains device 0's primary context, so call it when reporting a
/// failure.
pub fn detectedHardware() []const u8 {
    // wasm has no drivers to load (and gompute's loader lock needs sched_yield).
    if (comptime @import("builtin").cpu.arch.isWasm()) return "none";
    var compute = gompute.runtime.dynamic.Compute.init(null) catch return "none";
    defer compute.deinit();
    return switch (compute.backend) {
        .cpu => "none",
        .cuda, .hip => |b| @tagName(b),
    };
}

/// False for an explicit `cuda`/`hip` this binary cannot serve, so the
/// request fails before any simulation runs. `auto` and `cpu` always pass.
pub fn requestSupported(req: Request) bool {
    return switch (req) {
        .cpu, .auto => true,
        .cuda => backend == .cuda,
        .hip => backend == .hip,
    };
}

/// Monotonic nanoseconds, for the cost probe, `Prof` and `lu.zig`'s timers.
pub fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

// ---- private ----

test declineKind {
    try std.testing.expectEqual(Decline.policy, declineKind(Error.NotEnoughGpuWork));
    try std.testing.expectEqual(Decline.capability, declineKind(Error.CircuitNotEligible));
    try std.testing.expectEqual(Decline.machine, declineKind(Error.NoGpuArtifacts));
    // Driver and allocator errors never come from `Error`.
    try std.testing.expectEqual(Decline.machine, declineKind(error.InitFailed));
    try std.testing.expectEqual(Decline.machine, declineKind(error.OutOfMemory));
}

test requestSupported {
    try std.testing.expect(requestSupported(.cpu));
    try std.testing.expect(requestSupported(.auto));
    try std.testing.expectEqual(backend == .cuda, requestSupported(.cuda));
    try std.testing.expectEqual(backend == .hip, requestSupported(.hip));
}

test fallsBack {
    const errs = [_]InitError{ Error.NotEnoughGpuWork, Error.CircuitNotEligible, Error.NoGpuArtifacts, error.InitFailed };
    for (errs) |e| {
        // An explicit backend never runs on the CPU, nothing eligible included.
        try std.testing.expect(!fallsBack(.cuda, true, e));
        try std.testing.expect(!fallsBack(.hip, true, e));
        try std.testing.expect(fallsBack(.auto, false, e));
        try std.testing.expectEqual(declineKind(e) != .machine, fallsBack(.auto, true, e));
    }
}

test detectedHardware {
    // Whatever the machine has, the answer is one of the three names and
    // does not depend on the build's images.
    const name = detectedHardware();
    try std.testing.expect(std.mem.eql(u8, name, "none") or std.mem.eql(u8, name, "cuda") or std.mem.eql(u8, name, "hip"));
}

test "without kernel images init declines as the machine and auto falls back" {
    if (comptime backend != null) return error.SkipZigTest;
    const empty: Planes = .{ .g_vals = &.{}, .c_vals = &.{}, .rhs = &.{}, .q_vec = &.{} };
    const host: Host = .{ .batches = &.{}, .planes = empty, .n = 0, .trash_slot = 0, .has_charge = false, .knobs = .{} };
    const got = GpuContext.init(std.testing.allocator, host, null, true, 1, 1e6);
    try std.testing.expectError(Error.NoGpuArtifacts, got);
    try std.testing.expect(!available);
    try std.testing.expect(fallsBack(.auto, false, Error.NoGpuArtifacts));
}
