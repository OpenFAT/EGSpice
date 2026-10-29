//! Comptime layout of one device type's tapes and derivative basis: the
//! VerA contract's pattern and limit declarations in, the slot, lim and
//! LTE-site maps and the gather/scatter tapes out. Nothing here runs per
//! Newton iteration except the `at` lookups, which fold to constants.
//! Cites VerA's contract (`jac_pattern`, `q_pattern`, `deriv_reads`,
//! `collapse_full`, `limit_writes`) and #117 for masks past 64 unknowns.
//!
//! Contents:
//! - `pub const object_root: bool`
//! - `pub const Mask`
//! - `pub const Basis = struct`
//! - `pub fn widen(comptime n: usize, m: anytype) Mask(n)`
//! - `pub fn lowBits(comptime n: usize) Mask(n)`
//! - `pub inline fn has(m: anytype, u: usize) bool`
//! - `pub fn buildTapes(comptime M: type, nodes: []const u32, n_u: usize, pat: []const M, pv: PatternView, gath: []u32, rhs_idx: []u32, slots: []u32) void`
//! - `pub fn SlotMap(comptime D: type) type` (`k`, `at`)
//! - `pub fn LimMap(comptime D: type) type` (`k`, `at`)
//! - `pub fn jacPattern(comptime D: type) [nU(D)]Mask(nU(D))`
//! - `pub fn rowPattern(comptime D: type, comptime name: []const u8) [nU(D)]Mask(nU(D))`
//! - `pub fn limitWritesOf(comptime D: type) Mask(nU(D))`, `limitReadsOf`
//! - `pub fn writtenRows(comptime D: type, comptime name: []const u8) [nU(D)]bool`
//! - `pub fn lteSites(comptime D: type) []const usize`
//! - `pub fn basisOf(comptime D: type, comptime narrow: bool) Basis`
//! - `pub fn repMask(comptime n_u: usize, comptime lane: [n_u]u8, comptime alias: [n_u]bool, comptime pat: [n_u]Mask(n_u)) [n_u]Mask(n_u)`
//! - `pub fn constEntry(comptime D: type, comptime ru: usize, comptime cu: usize) ?contract.JacConst(D.U)`
//! - `pub fn canNarrow(comptime D: type) bool`, `narrowEligible`

const std = @import("std");
const assert = std.debug.assert;
const contract = @import("contract");
const abi = @import("device_abi");
const GROUND = abi.GROUND;
const PatternView = abi.PatternView;

/// Whether eval.zig is the build root: a linked per-model object or GPU
/// image, which alone has the per-model `models` wrapper. eval.zig declares
/// `device_object_root`; a test root or a `.so` shim does not.
pub const object_root = @hasDecl(@import("root"), "device_object_root");

/// The host's mask over `n` unknowns or lanes: u64, wider past 64 (#117).
pub const Mask = contract.MaskOf;

/// The derivative basis `evalRange` seeds: `w` lanes, and the lane each
/// unknown seeds into.
///
/// Only the unknowns in `contract.derivReads` get a lane: the device reads
/// the others as constants and `jac_const` holds their exact partials. Wide:
/// one lane per read unknown. Narrow, for an instance whose collapse is
/// maximal (`collapse_full`): every merged set is one circuit node, its
/// columns share a matrix slot, and one shared lane computes the summed entry
/// the solver sees. mos1 goes from 6 lanes to 4, one ymm instead of two.
pub const Basis = struct {
    w: usize,
    /// `lane[u]` is the lane unknown `u` seeds, or `contract.no_lane`.
    lane: []const u8,
};

/// A `contract.Mask(D)` mask over `n` unknowns as `Mask(n)`. A device past
/// 64 unknowns without `deriv_reads` keeps u64 masks, all ones meaning every
/// unknown, so the bits u64 cannot hold are set too.
pub fn widen(comptime n: usize, m: anytype) Mask(n) {
    if (@TypeOf(m) == Mask(n) or n <= 64) return m;
    return if (m == std.math.maxInt(u64)) std.math.maxInt(Mask(n)) else m;
}

/// The low `n` bits of `Mask(n)`.
pub fn lowBits(comptime n: usize) Mask(n) {
    const M = Mask(n);
    return if (n == @bitSizeOf(M)) std.math.maxInt(M) else (@as(M, 1) << @intCast(n)) - 1;
}

/// Bit `u` of `m`.
pub inline fn has(m: anytype, u: usize) bool {
    return (m >> @intCast(u)) & 1 != 0;
}

/// Fills the gather, residual-row and Jacobian-slot tapes from the flat node
/// list (`[id * n_u + u]`). Ground rows go to row `pv.n`. The slot tape holds
/// only the entries `pat` sets, row-major, `slots.len / count` per instance
/// (`SlotMap`); a ground one goes to `pv.trash_slot`. Asserts every other
/// (row, col) is in the pattern, that `pat` has `n_u` rows, and that
/// `gath` and `rhs_idx` match `nodes` and `slots` fills exactly.
pub fn buildTapes(comptime M: type, nodes: []const u32, n_u: usize, pat: []const M, pv: PatternView, gath: []u32, rhs_idx: []u32, slots: []u32) void {
    assert(n_u > 0);
    assert(pat.len == n_u);
    assert(nodes.len % n_u == 0);
    assert(gath.len == nodes.len);
    assert(rhs_idx.len == nodes.len);
    const count = nodes.len / n_u;
    var k: usize = 0;
    for (0..count) |id| {
        const nd = nodes[id * n_u ..][0..n_u];
        for (nd, 0..) |node, u| {
            gath[id * n_u + u] = node;
            rhs_idx[id * n_u + u] = if (node == GROUND) pv.n else node;
        }
        for (nd, 0..) |r, ru| for (nd, 0..) |c, cu| {
            if (!has(pat[ru], cu)) continue;
            slots[k] = if (r == GROUND or c == GROUND) pv.trash_slot else pv.findSlot(r, c).?;
            k += 1;
        };
    }
    assert(k == slots.len);
}

/// D's slot tape layout: `k` entries per instance, one per set bit of
/// `jacPattern(D)` in row-major order, and `at[ru * n_u + cu]` the position of
/// entry (ru, cu) among them. A structural zero has no slot, so no stamp, no
/// tape entry and no GPU staging cell exists for it.
pub fn SlotMap(comptime D: type) type {
    const n_u = contract.nU(D);
    return struct {
        const pat = jacPattern(D);
        /// Row `ru`'s first position, so `at` is O(1): a dense device past
        /// 64 unknowns calls it n_u^2 times (#117).
        const row_start: [n_u + 1]usize = blk: {
            @setEvalBranchQuota(100_000);
            var s: [n_u + 1]usize = undefined;
            s[0] = 0;
            for (pat, 0..) |row, r| s[r + 1] = s[r] + @popCount(row & lowBits(n_u));
            break :blk s;
        };
        pub const k: usize = row_start[n_u];
        /// Entry (ru, cu)'s position; a compile error for a structural zero.
        pub fn at(comptime ru: usize, comptime cu: usize) usize {
            comptime {
                if (!has(pat[ru], cu)) @compileError(@typeName(D) ++ " stamps an entry outside its jac/q pattern");
                return row_start[ru] + @popCount(pat[ru] & ((@as(Mask(n_u), 1) << @intCast(cu)) - 1));
            }
        }
    };
}

/// D's lim plane layout: `k` values per instance, one per unknown `limit`
/// writes (`contract.limitWrites`), and `at(u)` unknown u's position among
/// them. Nothing reads a lim value `limit` does not write.
pub fn LimMap(comptime D: type) type {
    const n_u = contract.nU(D);
    const writes = limitWritesOf(D);
    return struct {
        pub const k: usize = @popCount(writes);
        pub fn at(comptime u: usize) usize {
            comptime {
                if (!has(writes, u)) @compileError(@typeName(D) ++ " reads a lim value its `limit` does not write");
                return @popCount(writes & ((@as(Mask(n_u), 1) << @intCast(u)) - 1));
            }
        }
    };
}

/// D's structural Jacobian, resistive and reactive combined: bit `cu` of row
/// `ru` is set when the device can write local entry (ru, cu).
pub fn jacPattern(comptime D: type) [contract.nU(D)]Mask(contract.nU(D)) {
    var out = rowPattern(D, "jac_pattern");
    // Without `q` there is no reactive half; `rowPattern` would call it dense.
    if (@hasDecl(D, "q")) {
        const q = rowPattern(D, "q_pattern");
        for (&out, q) |*o, qm| o.* |= qm;
    }
    return out;
}

/// The pattern decl `name` (`jac_pattern` or `q_pattern`). VerA omits it
/// past about 2048 unknowns; each row is then the lanes `deriv_reads` gives
/// it plus its `jac_const` columns, all ones (dense) without `deriv_reads`.
pub fn rowPattern(comptime D: type, comptime name: []const u8) [contract.nU(D)]Mask(contract.nU(D)) {
    const n_u = contract.nU(D);
    var out: [n_u]Mask(n_u) = undefined;
    if (!@hasDecl(D, name)) {
        @setEvalBranchQuota(100 * n_u + 1000);
        out = @splat(widen(n_u, contract.derivReads(D)) & lowBits(n_u));
        for (contract.jacConst(D)) |e| out[@backingInt(e.row)] |= @as(Mask(n_u), 1) << @intCast(@backingInt(e.col));
        return out;
    }
    for (&out, @field(D, name)) |*o, m| o.* = widen(n_u, m) & lowBits(n_u);
    return out;
}

/// The unknowns `limit` writes and reads, as host masks.
pub fn limitWritesOf(comptime D: type) Mask(contract.nU(D)) {
    return widen(contract.nU(D), contract.limitWrites(D)) & lowBits(contract.nU(D));
}

pub fn limitReadsOf(comptime D: type) Mask(contract.nU(D)) {
    return widen(contract.nU(D), contract.limitReads(D)) & lowBits(contract.nU(D));
}

/// Which residual rows D's `eval` (`jac_rows`) or `q` (`q_rows`) ever writes;
/// all rows when undeclared. The others are zero at every bias, so their
/// stamps are dropped at comptime.
///
/// Never derive this from `rowPattern(...)[ru] != 0`. A row can be written
/// with a value that depends on no unknown: `isource` has `jac_pattern =
/// {0, 0}` and still stamps its current into both rows.
pub fn writtenRows(comptime D: type, comptime name: []const u8) [contract.nU(D)]bool {
    if (!@hasDecl(D, name)) return @splat(true);
    const mask = widen(contract.nU(D), @field(D, name));
    var out: [contract.nU(D)]bool = @splat(true);
    for (&out, 0..) |*o, ru| o.* = has(mask, ru);
    return out;
}

/// The charge sites of D that join the transient LTE (`contract.qLte`), in
/// site order. `q_tape` holds one entry per instance and listed site,
/// indexed `id * lteSites(D).len + j`, the way ngspice's `*trunc.c` routines
/// hand CKTterr one charge state each.
pub fn lteSites(comptime D: type) []const usize {
    comptime {
        const on = contract.qLte(D);
        var out: [on.len]usize = undefined;
        var n: usize = 0;
        for (on, 0..) |o, k| if (o) {
            out[n] = k;
            n += 1;
        };
        const sites = out[0..n].*;
        return &sites;
    }
}

pub fn basisOf(comptime D: type, comptime narrow: bool) Basis {
    const n_u = contract.nU(D);
    @setEvalBranchQuota(100 * n_u + 1000);
    const reads = widen(n_u, contract.derivReads(D));
    var lane: [n_u]u8 = @splat(contract.no_lane);
    var of_root: [n_u]u8 = @splat(contract.no_lane);
    var w: usize = 0;
    for (0..n_u) |u| {
        if (!has(reads, u)) continue;
        // The contract guarantees `collapse_full` is resolved.
        const root = if (narrow) D.collapse_full[u] orelse u else u;
        if (of_root[root] == contract.no_lane) {
            of_root[root] = @intCast(w);
            w += 1;
        }
        lane[u] = of_root[root];
    }
    const l = lane;
    return .{ .w = w, .lane = &l };
}

/// One pattern half with a single representative column kept per lane, so a
/// shared lane is stamped once; columns without a lane are dropped (their
/// partials come from `jac_const`). Computed per half because `g_vals` and
/// `c_vals` are separate planes.
///
/// The representative is the lowest alias, not the root: that is the column
/// the wide kernel stamps, so a slot shared with an unrelated column (a gate
/// tied to its drain) sums in the same order and the output stays
/// byte-identical. `alias[cu]` is `collapse_full[cu] != null`; all-false makes
/// this the identity on the laned columns.
pub fn repMask(comptime n_u: usize, comptime lane: [n_u]u8, comptime alias: [n_u]bool, comptime pat: [n_u]Mask(n_u)) [n_u]Mask(n_u) {
    var out: [n_u]Mask(n_u) = @splat(0);
    for (&out, pat) |*m, row| {
        var rep: [n_u]?usize = @splat(null);
        for (0..n_u) |cu| {
            if (!has(row, cu) or lane[cu] == contract.no_lane) continue;
            const cur = rep[lane[cu]];
            // First column of the lane wins, then any alias beats a root.
            if (cur == null or (!alias[cur.?] and alias[cu])) rep[lane[cu]] = cu;
        }
        for (rep) |c| if (c) |cu| {
            m.* |= @as(Mask(n_u), 1) << @intCast(cu);
        };
    }
    return out;
}

/// The `jac_const` entry for local (row `ru`, column `cu`), if any.
pub fn constEntry(comptime D: type, comptime ru: usize, comptime cu: usize) ?contract.JacConst(D.U) {
    for (contract.jacConst(D)) |e| {
        if (@backingInt(e.row) == ru and @backingInt(e.col) == cu) return e;
    }
    return null;
}

/// Whether D gets a second `evalRange` instantiation on the narrow basis for
/// its fully collapsed instances. It pays only when it saves a register on
/// AVX2 (dev/devices/abi.md, "Derivative lanes"): the narrow dual spans
/// fewer ymm than the wide one. That admits mos1/2/3/6/9, bsim1, bsim3,
/// hfet2, jfet and mes, and BSIM4 (11 lanes, three ymm, down to 4, one).
///
/// Limiting adds a correctness gate: the correction is stored per lane, so no
/// two unknowns `limit` writes may share one.
///
/// A model past 8 unknowns (BSIM4) narrows only in its linked device object
/// (`eval.zig` as the build root), so a test-built vtable stays the forced
/// wide reference the narrow one is checked against.
pub fn canNarrow(comptime D: type) bool {
    return narrowEligible(D) and (contract.nU(D) <= 8 or object_root);
}

pub fn narrowEligible(comptime D: type) bool {
    if (!@hasDecl(D, "collapse_full")) return false;
    const n_u = contract.nU(D);
    if (n_u <= 4) return false;
    const b = comptime basisOf(D, true);
    if ((b.w + 3) / 4 >= (comptime basisOf(D, false).w + 3) / 4) return false;
    if (@hasDecl(D, "limit")) {
        var seen: [n_u]bool = @splat(false);
        const writes = limitWritesOf(D);
        for (0..n_u) |u| if (has(writes, u)) {
            if (b.lane[u] == contract.no_lane or seen[b.lane[u]]) return false;
            seen[b.lane[u]] = true;
        };
    }
    return true;
}
