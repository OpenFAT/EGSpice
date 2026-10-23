//! Exact small-signal RLC/RC line, added to ltra.va's DC Jacobian through
//! ac_dyn. The one model term in Zig: Verilog-AMS gives a device no access to
//! the analysis frequency, so its AC response is always a ratio of sums of
//! polynomial(s)*exp(s*tau) terms (ddt, idt, laplace_*, absdelay, zi_*). This
//! kernel is not one for R > 0: Zc = sqrt((R+jwL)/(jwC)),
//! E = exp(-len*sqrt((R+jwL)*jwC)). R = 0 is an exact absdelay (tline.va).
//! The port equations are v1-Zc*i1 = E*(v2+Zc*i2) and its mirror.
//! See ngspice ltraacld.c; its admittance form is the same two-port.
const std = @import("std");
const C = std.math.Complex(f64);

// ltra.va: four ports followed by the two branch currents. Both branch
// rows already have all six columns in the transient Jacobian pattern.
pub const ac_dyn_slots = [_]u32{ 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35 };

/// The same entry point as a VerA-generated frequency-dependent device.
pub fn acDyn(comptime F: type, model: anytype, _: anytype, _: anytype, _: anytype, omega: F, out: *[ac_dyn_slots.len]std.math.Complex(F)) void {
    if (comptime @typeInfo(F) == .vector) {
        inline for (0..@typeInfo(F).vector.len) |lane| {
            const delta = correction(model.r, model.l, model.c, model.len, omega[lane]);
            for (out, delta) |*entry, value| {
                entry.re[lane] = value.re;
                entry.im[lane] = value.im;
            }
        }
    } else out.* = correction(model.r, model.l, model.c, model.len, omega);
}

fn correction(r: f64, l: f64, c: f64, len: f64, omega: f64) [12]C {
    const zero = C.init(0, 0);
    const one = C.init(1, 0);
    // At DC the existing rows are exact, including current continuity.
    if (omega == 0) return @splat(zero);

    const w = @abs(omega);
    const zmag = std.math.hypot(r, w * l);
    const theta = 0.5 * std.math.atan2(r, w * l);
    const zm = @sqrt(zmag / (w * c));
    const gm = @sqrt(zmag * (w * c));
    const sign: f64 = if (omega < 0) -1 else 1;
    const z = C.init(zm * @cos(theta), -sign * zm * @sin(theta));
    const attenuation = @exp(-len * gm * @sin(theta));
    const phase = -sign * len * gm * @cos(theta);
    const e = C.init(attenuation * @cos(phase), attenuation * @sin(phase));
    const ze = z.mul(e);

    // Subtract ltra.va's two DC rows before adding the AC rows:
    //   v1 - v2 - R*len*i1 = 0
    //   -i1 - i2 = 0
    // Port KCL rows remain untouched; the correction is independent of bias.
    return .{
        zero,    zero, one.sub(e), e.sub(one), C.init(r * len, 0).sub(z), ze.neg(),
        e.neg(), e,    one,        one.neg(),  one.sub(ze),               one.sub(z),
    };
}

test "LTRA frequency correction preserves DC and conjugates negative frequencies" {
    for ([_]f64{ 0, 250e-9 }) |l| {
        for (correction(5, l, 100e-12, 2, 0)) |entry|
            try std.testing.expectEqual(C.init(0, 0), entry);
        const positive = correction(5, l, 100e-12, 2, 2 * std.math.pi * 12.5e6);
        const negative = correction(5, l, 100e-12, 2, -2 * std.math.pi * 12.5e6);
        for (positive, negative) |p, n| {
            try std.testing.expectEqual(p.re, n.re);
            try std.testing.expectEqual(p.im, -n.im);
        }
    }
}
