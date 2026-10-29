//! Memory access for `evalRange` and `limitRange`: a batch's tapes and
//! planes in, one inline accessor per gather or stamp out. `Sink` serves
//! the host (adds, or staged stores) and the per-instance GPU kernels
//! (staging stores); `RecordSink` wraps a host sink to record every write
//! for the latency bypass. Cites dev/devices/gpu-evaluation.md for the
//! staging and reduce order, and 01 1.10 for store-not-add on the GPU.
//!
//! Contents:
//! - `pub const RecordWrite = struct`
//! - `pub const rec_res`, `rec_jac`, `rec_q`, `rec_qjac`
//! - `pub fn RecordSink(comptime Inner: type) type`
//! - `pub fn Sink(comptime D: type, comptime device: bool, comptime skip_const: bool, comptime staged: bool) type`
//! - sink members: `skip_g`, `skip_c`, `on_device`, `skip_tape`, `stores`,
//!   `x`, `xOld`, `gath`, `rhsRow`, `lim`, `setLim`, `model`, `inst`,
//!   `scatterRes`, `scatterJac`, `scatterQ`, `tapeQ`, `scatterQJac`,
//!   `resPtr`, `jacPtr`, `qPtr`, `qJacPtr`, `tapePtr`, `host`

const std = @import("std");
const contract = @import("contract");
const gompute = @import("gompute");
const abi = @import("device_abi");
const layout = @import("layout.zig");
const traits = @import("traits.zig");
const DeviceBatch = @import("batch.zig").DeviceBatch;
const Planes = abi.Planes;
const LimMap = layout.LimMap;
const SlotMap = layout.SlotMap;

/// One write `RecordSink` keeps: the cell's address with bit 0 set for a
/// store (staging positions, the charge tape), clear for an add, and the
/// value. `tag`: kind << 16 | local row << 8 | local column (residual and
/// Jacobian writes); 0 for the charge tape.
pub const RecordWrite = struct { ptr: usize, val: f64, tag: u32 = 0 };

pub const rec_res = 1;

pub const rec_jac = 2;

pub const rec_q = 3;

pub const rec_qjac = 4;

/// A host `Sink` that also records every write into `out`, in order: the
/// latency bypass (`DeviceBatch.evalBypass`) replays them.
pub fn RecordSink(comptime Inner: type) type {
    return struct {
        inner: *const Inner,
        out: []RecordWrite,
        id: u32,
        n: usize = 0,
        overflow: bool = false,
        // evalRange writes residual rows in ascending local order, so the
        // next row at or after the last one is the writer.
        res_ru: u32 = 0,

        pub const skip_g = Inner.skip_g;
        pub const skip_c = Inner.skip_c;
        pub const on_device = false;
        pub const skip_tape = Inner.skip_tape;
        const Rs = @This();

        inline fn rec(s: *const Rs, p: *f64, val: f64, store: bool, tag: u32) void {
            const m: *Rs = @constCast(s);
            if (m.n == m.out.len) {
                m.overflow = true;
                return;
            }
            m.out[m.n] = .{ .ptr = @intFromPtr(p) | @intFromBool(store), .val = val, .tag = tag };
            m.n += 1;
        }
        fn rowOf(s: *const Rs, next: *u32, row: u32) u32 {
            while (s.inner.rhsRow(s.id, next.*) != row) next.* += 1;
            next.* += 1;
            return next.* - 1;
        }
        pub inline fn x(s: *const Rs, gi: u32) f64 {
            return s.inner.x(gi);
        }
        pub inline fn gath(s: *const Rs, id: u32, u: usize) u32 {
            return s.inner.gath(id, u);
        }
        pub inline fn rhsRow(s: *const Rs, id: u32, ru: usize) u32 {
            return s.inner.rhsRow(id, ru);
        }
        pub inline fn lim(s: *const Rs, id: u32, comptime u: usize) f64 {
            return s.inner.lim(id, u);
        }
        pub inline fn model(s: *const Rs, id: u32) @TypeOf(s.inner.model(id)) {
            return s.inner.model(id);
        }
        pub inline fn inst(s: *const Rs, id: u32) @TypeOf(s.inner.inst(id)) {
            return s.inner.inst(id);
        }
        pub inline fn scatterRes(s: *const Rs, row: u32, val: f64) void {
            s.inner.scatterRes(row, val);
            s.rec(s.inner.resPtr(row), val, Inner.stores, rec_res << 16 | s.rowOf(&@constCast(s).res_ru, row) << 8);
        }
        pub inline fn scatterJac(s: *const Rs, id: u32, comptime ru: usize, comptime cu: usize, val: f64) void {
            s.inner.scatterJac(id, ru, cu, val);
            s.rec(s.inner.jacPtr(id, ru, cu), val, Inner.stores, rec_jac << 16 | ru << 8 | cu);
        }
        pub inline fn scatterQ(s: *const Rs, row: u32, qv: f64) void {
            s.inner.scatterQ(row, qv);
            s.rec(s.inner.qPtr(row), qv, Inner.stores, rec_q << 16);
        }
        pub inline fn scatterQJac(s: *const Rs, id: u32, comptime ru: usize, comptime cu: usize, val: f64) void {
            s.inner.scatterQJac(id, ru, cu, val);
            s.rec(s.inner.qJacPtr(id, ru, cu), val, Inner.stores, rec_qjac << 16);
        }
        pub inline fn tapeQ(s: *const Rs, id: u32, j: usize, qv: f64) void {
            s.inner.tapeQ(id, j, qv);
            s.rec(s.inner.tapePtr(id, j), qv, true, 0);
        }
    };
}

/// Memory access for `evalRange`/`limitRange`. `device` changes only how
/// pointers are reached: the host indexes plain slices (`GlobalPtr(T)` is
/// `[*]T`), the GPU casts `.global` kernel arguments to the generic address
/// space the contract's `eval` takes. Host-only state is `void` under
/// `device`. `skip_const` drops stamps of a constant Jacobian half.
/// `staged` (host only) stores each stamp at its own tape position instead
/// of adding it into its plane cell: `rhs`/`q_vec` are indexed `id * n_u +
/// ru` and `g_vals`/`c_vals` like `slots`, both relative to the batch, and a
/// ground stamp lands on its own position too. ParEval then sums the
/// positions into the planes in tape order (`Hooks.stage`).
pub fn Sink(comptime D: type, comptime device: bool, comptime skip_const: bool, comptime staged: bool) type {
    if (device and staged) @compileError("staging is host only");
    const n_u: usize = comptime contract.nU(D);
    const BatchT = DeviceBatch(D);
    return struct {
        xs: gompute.GlobalPtr(f64),
        gath_: gompute.GlobalPtr(u32),
        rhs_idx_: gompute.GlobalPtr(u32),
        slots_: gompute.GlobalPtr(u32),
        models_: gompute.GlobalPtr(D.Model),
        model_of_: gompute.GlobalPtr(u32),
        instances_: gompute.GlobalPtr(D.Instance),
        g_vals: gompute.GlobalPtr(f64),
        c_vals: gompute.GlobalPtr(f64),
        rhs: gompute.GlobalPtr(f64),
        q_vec: gompute.GlobalPtr(f64),
        /// The lim plane (`count * LimMap(D).k`). Undefined, and never read,
        /// when D has no `limit`.
        lim_: gompute.GlobalPtr(f64),
        /// `x_old`; read only by `limitRange`.
        xo: gompute.GlobalPtr(f64),
        /// The batch's `q_tape`, host only. Held as a bare pointer: through
        /// the batch, every plane store might alias the slice header, and
        /// LLVM reloaded it before each charge store.
        q_tape: if (device or !@hasDecl(D, "q")) void else [*]f64,

        // `Circuit.computeBaseline` stamps only batches whose whole Jacobian
        // is constant (`Batch.has_const_jacobian`). A lone constant half has
        // no baseline copy, so skipping it would drop it from the Newton
        // matrix: an idt's charge (C = 1) beside a non-constant G, say.
        const in_baseline = traits.constJacobian(D);
        pub const skip_g = skip_const and in_baseline;
        pub const skip_c = skip_const and in_baseline;
        pub const on_device = device;
        /// Newton passes leave the LTE tape alone: the transient reads it
        /// only after a full, charge-only or accepted-point pass.
        pub const skip_tape = skip_const;

        const Sk = @This();

        pub inline fn x(s: *const Sk, gi: u32) f64 {
            return s.xs[gi];
        }
        pub inline fn xOld(s: *const Sk, gi: u32) f64 {
            return s.xo[gi];
        }
        pub inline fn gath(s: *const Sk, id: u32, u: usize) u32 {
            return s.gath_[@as(usize, id) * n_u + u];
        }
        pub inline fn rhsRow(s: *const Sk, id: u32, ru: usize) u32 {
            if (comptime staged) return @intCast(@as(usize, id) * n_u + ru);
            return s.rhs_idx_[@as(usize, id) * n_u + ru];
        }
        pub inline fn lim(s: *const Sk, id: u32, comptime u: usize) f64 {
            return s.lim_[@as(usize, id) * LimMap(D).k + comptime LimMap(D).at(u)];
        }
        pub inline fn setLim(s: *const Sk, id: u32, comptime u: usize, v: f64) void {
            s.lim_[@as(usize, id) * LimMap(D).k + comptime LimMap(D).at(u)] = v;
        }
        // An address-space cast, not a copy: a compact model's Model has
        // hundreds of fields and would not fit in GPU registers. A no-op on
        // the host.
        pub inline fn model(s: *const Sk, id: u32) *const D.Model {
            return @addrSpaceCast(&s.models_[s.model_of_[id]]);
        }
        pub inline fn inst(s: *const Sk, id: u32) if (@hasDecl(D, "mutable_eval") and D.mutable_eval) *D.Instance else *const D.Instance {
            return @addrSpaceCast(&s.instances_[id]);
        }
        inline fn slot(s: *const Sk, id: u32, comptime ru: usize, comptime cu: usize) u32 {
            if (comptime staged) return @intCast(@as(usize, id) * SlotMap(D).k + comptime SlotMap(D).at(ru, cu));
            return s.slots_[@as(usize, id) * SlotMap(D).k + comptime SlotMap(D).at(ru, cu)];
        }
        // A store on the GPU. There the tape indexes a staging cell only
        // this thread writes, once per pass (the host's staged positions
        // rely on the same), and `ReduceKernel` sums each plane cell's
        // cells in tape order from +0.0; an atomic add would make the order,
        // and so the rounding, vary between runs. A store needs no cleared
        // staging per pass (01 1.10), and `+0.0 + v` in the reduce gives the
        // bits `+0.0 + (0 + v)` did, signed zeros included. Cells no pass
        // writes (rows outside `jac_rows`/`q_rows`, an entry
        // `jacConstApplies` drops) hold the zero the launcher cleared once.
        inline fn add(p: gompute.GlobalPtr(f64), i: u32, v: f64) void {
            // A staged position takes one stamp per pass.
            if (comptime staged or device) p[i] = v else p[i] += v;
        }
        pub inline fn scatterRes(s: *const Sk, row: u32, val: f64) void {
            add(s.rhs, row, val);
        }
        pub inline fn scatterJac(s: *const Sk, id: u32, comptime ru: usize, comptime cu: usize, val: f64) void {
            add(s.g_vals, s.slot(id, ru, cu), val);
        }
        pub inline fn scatterQ(s: *const Sk, row: u32, qv: f64) void {
            add(s.q_vec, row, qv);
        }
        /// Records LTE site `j` of instance `id` in the host `q_tape`.
        pub inline fn tapeQ(s: *const Sk, id: u32, j: usize, qv: f64) void {
            s.q_tape[@as(usize, id) * (comptime layout.lteSites(D).len) + j] = qv;
        }
        pub inline fn scatterQJac(s: *const Sk, id: u32, comptime ru: usize, comptime cu: usize, val: f64) void {
            add(s.c_vals, s.slot(id, ru, cu), val);
        }
        /// The cell each scatter writes, and whether it stores (a staged
        /// position) rather than adds: what `RecordSink` keeps.
        pub const stores = staged;
        pub inline fn resPtr(s: *const Sk, row: u32) *f64 {
            return &s.rhs[row];
        }
        pub inline fn jacPtr(s: *const Sk, id: u32, comptime ru: usize, comptime cu: usize) *f64 {
            return &s.g_vals[s.slot(id, ru, cu)];
        }
        pub inline fn qPtr(s: *const Sk, row: u32) *f64 {
            return &s.q_vec[row];
        }
        pub inline fn qJacPtr(s: *const Sk, id: u32, comptime ru: usize, comptime cu: usize) *f64 {
            return &s.c_vals[s.slot(id, ru, cu)];
        }
        pub inline fn tapePtr(s: *const Sk, id: u32, j: usize) *f64 {
            return &s.q_tape[@as(usize, id) * (comptime layout.lteSites(D).len) + j];
        }

        /// Host sink over batch `b`, writing into `pl`. `xo` may be undefined
        /// outside `limitRange`.
        pub fn host(b: *BatchT, pl: *const Planes, xs: []const f64, xo: [*]const f64) Sk {
            return .{
                .xs = @constCast(xs.ptr), // GlobalPtr has no const form; never written
                .gath_ = b.gath.ptr,
                .rhs_idx_ = b.rhs_idx.ptr,
                .slots_ = b.slots.ptr,
                .models_ = b.models.ptr,
                .model_of_ = b.model_of.ptr,
                .instances_ = b.instances.ptr,
                .g_vals = pl.g_vals.ptr,
                .c_vals = pl.c_vals.ptr,
                .rhs = pl.rhs.ptr,
                .q_vec = pl.q_vec.ptr,
                .lim_ = if (comptime @hasDecl(D, "limit")) b.lim_x.ptr else undefined,
                .xo = @constCast(xo),
                .q_tape = if (comptime @hasDecl(D, "q")) b.q_tape.ptr else {},
            };
        }
    };
}
