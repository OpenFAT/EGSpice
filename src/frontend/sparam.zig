//! HSPICE S-element data. In: a Touchstone 1.0 file. Out: its admittance
//! samples (sparam_touchstone.zig), the rational model every analysis runs
//! (sparam_fit.zig, vector fitting with common poles), made passive
//! (sparam_passivity.zig), as second-order sections. Every table lives in
//! the parse arena and dies after the build. The W element's fitter
//! (wfit.zig) reuses `fit`, `enforcePassivity`, `sections` and `symEig`.
//! Cites: HSPICE [SI] Ch.2; the papers each stage file cites.
//!
//! Contents:
//! - `pub const Network, Error, Raw, readTouchstone, parseTouchstone` (sparam_touchstone.zig)
//! - `pub const Fit, Sections, FitOptions, fit, sections` (sparam_fit.zig)
//! - `pub const enforcePassivity, symEig` (sparam_passivity.zig)

const std = @import("std");
const testing = std.testing;
const sparam_touchstone = @import("sparam_touchstone.zig");
const sparam_fit = @import("sparam_fit.zig");
const sparam_passivity = @import("sparam_passivity.zig");
const sparam_lsq = @import("sparam_lsq.zig");

/// Admittance samples of a P-port.
pub const Network = sparam_touchstone.Network;
/// Every way a read or fit fails.
pub const Error = sparam_touchstone.Error;
/// A Touchstone 1.0 file as written.
pub const Raw = sparam_touchstone.Raw;
/// Reads a Touchstone 1.0 file to its option line and records.
pub const readTouchstone = sparam_touchstone.readTouchstone;
/// Parses a Touchstone 1.0 file into admittance samples.
pub const parseTouchstone = sparam_touchstone.parseTouchstone;
/// A rational model with common poles.
pub const Fit = sparam_fit.Fit;
/// A fit as real second-order sections.
pub const Sections = sparam_fit.Sections;
/// The fit's accuracy target and pole budget.
pub const FitOptions = sparam_fit.FitOptions;
/// The rational fit of a network.
pub const fit = sparam_fit.fit;
/// A fit's poles and residues as sections.
pub const sections = sparam_fit.sections;
/// Makes a fit passive by residue perturbation.
pub const enforcePassivity = sparam_passivity.enforcePassivity;
/// Eigenvalues and vectors of a symmetric matrix.
pub const symEig = sparam_passivity.symEig;

const Cx = std.math.Complex(f64);

test "vector fit reproduces a rational two-port and stays passive" {
    var arena_state = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    // Series R-L between the ports, series R-C to ground at each:
    // Y = Ys·[[1,−1],[−1,1]] + Yp·I.
    const r = 10.0;
    const l = 5e-9;
    const c = 2e-12;
    const nf = 200;
    const freq = try arena.alloc(f64, nf);
    const re = try arena.alloc(f64, 4 * nf);
    const im = try arena.alloc(f64, 4 * nf);
    for (0..nf) |k| {
        freq[k] = 1e6 * std.math.pow(f64, 1e4, @as(f64, @floatFromInt(k)) / (nf - 1));
        const w = 2 * std.math.pi * freq[k];
        const ys = (Cx{ .re = 1, .im = 0 }).div(.{ .re = r, .im = w * l });
        for (0..4) |e| {
            const sgn: f64 = if (e == 0 or e == 3) 1 else -1;
            const yp = (Cx{ .re = 1, .im = 0 }).div(.{ .re = 25, .im = -1 / (w * c) });
            const y = ys.mul(.{ .re = sgn, .im = 0 }).add(if (sgn > 0) yp else .{ .re = 0, .im = 0 });
            re[e * nf + k] = y.re;
            im[e * nf + k] = y.im;
        }
    }
    const net: Network = .{ .ports = 2, .freq = freq, .re = re, .im = im };
    var f = try fit(arena, net, .{});
    try testing.expect(f.rel_err < 1e-6);
    try testing.expect(try enforcePassivity(arena, net, &f));
    for (0..nf) |k| try testing.expect(f.eval(1, 2 * std.math.pi * freq[k]).sub(net.at(1, k)).magnitude() < 1e-4 * net.at(1, k).magnitude() + 1e-9);
}

test {
    _ = sparam_touchstone;
    _ = sparam_fit;
    _ = sparam_passivity;
    _ = sparam_lsq;
    _ = @import("tests/sparam.zig");
}
