//! The kept spectral lines of a multi-tone harmonic balance: tones, their
//! harmonic counts and INTMODMAX in, the box of mixing products k·f with
//! |k_i| <= H_i (optionally cut to the diamond Σ|k_i| <= INTMODMAX that
//! keeps each tone's own harmonics), merged by frequency, out.
//! Cites: HSPICE `.hb` [CR .HB] (INTMODMAX, SUBHARMS), VACASK
//! `truncate="hybrid"` and `lib/spurs.cpp` (line merging).
//!
//! Contents:
//! - `pub const max_tones = @import("core").validate.mhb_max_tones`
//! - `pub const max_box = @import("core").validate.mhb_max_box`
//! - `pub const Spectrum = struct`
//! - `Spectrum`: `deinit`
//! - `pub fn spectrum(gpa: std.mem.Allocator, tones: []const f64, nharms: []const u16, intmodmax: u16, sub: u16) !Spectrum`

const std = @import("std");
const root = @import("../types.zig");
const take = root.take;
const pss = @import("pss.zig");
const num = @import("core").numerics;
const solvers = @import("solver");
const converger = solvers.converger;
const dense_lu = solvers.dense_lu;
const FreqSolver = solvers.freq_solve.FreqSolver;
const Gmres = solvers.gmres.Gmres;
const Gcrodr = solvers.gcrodr.Gcrodr;
const SolveResult = pss.SolveResult;

/// The most tones a spectrum takes.
pub const max_tones = @import("core").validate.mhb_max_tones;

/// The largest box a spectrum enumerates, Π(2H_i + 1).
pub const max_box = @import("core").validate.mhb_max_box;

/// The kept spectral lines, DC first, then by ascending frequency. Line j
/// is `freqs[j]` Hz, the mixing product Σ_i mix[j*tones + i]·f_i.
pub const Spectrum = struct {
    /// At most `max_tones`.
    tones: u8,
    freqs: []f64,
    /// Line-major, `tones` coefficients per line; |k_i| <= NHARMS_i <= i16 max.
    mix: []i16,

    /// Frees both tables; `gpa` is the allocator `spectrum` took.
    pub fn deinit(self: Spectrum, gpa: std.mem.Allocator) void {
        gpa.free(self.freqs);
        gpa.free(self.mix);
    }
};

/// The spectrum of `tones` Hz with `nharms[i]` harmonics each and
/// INTMODMAX `intmodmax` (0: the whole box), where `sub` steps of the
/// first tone count as one order (HSPICE SUBHARMS; 1 for none). A product whose frequency
/// matches a kept one to 1e-14 relative is the same line: the lower order
/// wins, then a single tone's harmonic, then the earlier product, as in
/// VACASK `lib/spurs.cpp`. Asserts tones.len == nharms.len; `sub` must be
/// at least 1 (the query validator refuses 0). error.InvalidQueryOptions
/// for no tones, more than `max_tones`, a box past `max_box`, or a harmonic
/// count past i16 max. The caller frees the result with `Spectrum.deinit`
/// on `gpa`.
pub fn spectrum(gpa: std.mem.Allocator, tones: []const f64, nharms: []const u16, intmodmax: u16, sub: u16) !Spectrum {
    std.debug.assert(tones.len == nharms.len);
    const nt = tones.len;
    if (nt == 0 or nt > max_tones) return error.InvalidQueryOptions;
    var box: usize = 1;
    for (nharms) |h| {
        box = std.math.mul(usize, box, 2 * @as(usize, h) + 1) catch return error.InvalidQueryOptions;
        if (box > max_box or h > std.math.maxInt(i16)) return error.InvalidQueryOptions;
    }

    // Candidates on the positive half: the negative one is the same line.
    const Cand = struct { f: f64, order: u32, mixed: bool, idx: u32 };
    var cands: std.ArrayList(Cand) = .empty;
    defer cands.deinit(gpa);
    var mixes: std.ArrayList(i16) = .empty;
    defer mixes.deinit(gpa);
    var k: [max_tones]i32 = undefined;
    for (0..nt) |i| k[i] = -@as(i32, nharms[i]);
    for (0..box) |_| {
        var order: u32 = 0;
        var nnz: u32 = 0;
        var f: f64 = 0;
        var scale: f64 = 0;
        for (0..nt) |i| {
            order += if (i == 0) std.math.divCeil(u32, @abs(k[i]), sub) catch unreachable else @abs(k[i]);
            nnz += @intFromBool(k[i] != 0);
            const term = @as(f64, @floatFromInt(k[i])) * tones[i];
            f += term;
            scale += @abs(term);
        }
        const kept = intmodmax == 0 or order <= intmodmax or nnz <= 1;
        // Zero only for k = 0; any other product at 0 Hz is DC's duplicate.
        const tol = 1e-14 * scale;
        if (kept and ((order == 0) or f > tol)) {
            try cands.append(gpa, .{ .f = if (order == 0) 0 else f, .order = order, .mixed = nnz > 1, .idx = @intCast(cands.items.len) });
            for (0..nt) |i| try mixes.append(gpa, @intCast(k[i]));
        }
        // Odometer, last tone fastest.
        var i = nt;
        while (i > 0) {
            i -= 1;
            k[i] += 1;
            if (k[i] <= nharms[i]) break;
            k[i] = -@as(i32, nharms[i]);
        }
    }

    const c = cands.items;
    std.sort.pdq(Cand, c, {}, struct {
        fn lt(_: void, a: Cand, b: Cand) bool {
            return a.f < b.f or (a.f == b.f and a.idx < b.idx);
        }
    }.lt);
    // A run of candidates within 1e-14 relative of its head is one line.
    const runEnd = struct {
        fn f(cs: []const Cand, head: usize) usize {
            var end = head + 1;
            while (end < cs.len and cs[end].f - cs[head].f <= 1e-14 * cs[end].f) end += 1;
            return end;
        }
    }.f;
    // Counted first, so both tables are allocated at their final size: two
    // shrinking reallocs could not both be undone when the second failed
    // after the first had moved (#202).
    var lines: usize = 0;
    var head: usize = 0;
    while (head < c.len) : (lines += 1) head = runEnd(c, head);
    const freqs = try gpa.alloc(f64, lines);
    errdefer gpa.free(freqs);
    const mix = try gpa.alloc(i16, lines * nt);
    errdefer gpa.free(mix);
    var m: usize = 0;
    head = 0;
    while (head < c.len) : (m += 1) {
        const end = runEnd(c, head);
        var best = head;
        for (head + 1..end) |j| {
            const a = c[j];
            const b = c[best];
            if (a.order < b.order or (a.order == b.order and (!a.mixed and b.mixed or (a.mixed == b.mixed and a.idx < b.idx)))) best = j;
        }
        freqs[m] = c[best].f;
        @memcpy(mix[m * nt ..][0..nt], mixes.items[c[best].idx * nt ..][0..nt]);
        head = end;
    }
    std.debug.assert(m == lines);
    return .{ .tones = @intCast(nt), .freqs = freqs, .mix = mix };
}
