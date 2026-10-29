//! Touchstone 1.0 files. In: a `.sNp` file's bytes and its port count.
//! Out: the option line and records as written (`Raw`), and the admittance
//! samples Y = Z0⁻¹·(I − S)(I + S)⁻¹ every fit reads (`Network`), SoA over
//! (entry, frequency), entry-major.
//! Cites: Touchstone 1.1 and 2.0 (dev/research/data-file-syntax.md §1);
//! tests/frontend/grammar/touchstone.peg; HSPICE [SI] Ch.2.
//!
//! Contents:
//! - `pub const Network = struct`
//! - `pub fn at(self: Network, e: usize, k: usize) Cx`
//! - `pub const Error = error{ OutOfMemory, InvalidTouchstone, SingularSParameters }`
//! - `pub const Raw = struct`
//! - `pub const Format = enum { ma, db, ri }`
//! - `pub fn readTouchstone(arena: Allocator, bytes: []const u8, ports: usize) Error!Raw`
//! - `pub fn parseTouchstone(arena: Allocator, bytes: []const u8, ports: usize) Error!Network`

const std = @import("std");
const assert = std.debug.assert;
const testing = std.testing;
const Allocator = std.mem.Allocator;
const plainNumber = @import("netlist").lines.plainNumber;
/// Admittance samples of a P-port.
pub const Network = struct {
    /// P, 1 to 64 from `parseTouchstone`; the entry count is P².
    ports: usize,
    /// Hz, strictly ascending.
    freq: []const f64,
    /// Y entry e at frequency k: `re[e * freq.len + k]`, `im[...]`.
    re: []const f64,
    /// Imaginary parts, laid out as `re`.
    im: []const f64,

    /// Entry `e` (= i·P + j) at frequency index `k`.
    pub fn at(self: Network, e: usize, k: usize) Cx {
        assert(k < self.freq.len);
        return .{ .re = self.re[e * self.freq.len + k], .im = self.im[e * self.freq.len + k] };
    }
};

/// Every way a read or fit fails; partial tables stay in the arena.
pub const Error = error{
    OutOfMemory,
    /// A malformed or Touchstone 2.0 file, or a fit with no usable data.
    InvalidTouchstone,
    /// I + S (or Z) is singular at a data point.
    SingularSParameters,
};

/// A Touchstone 1.0 file as written: the option line's settings and the
/// network records, before any conversion. tests/frontend/grammar/touchstone.peg is
/// the syntax; the rules cite dev/research/data-file-syntax.md §1 (TS1.1 =
/// Touchstone 1.1, TS2.0 = 2.0).
pub const Raw = struct {
    /// Hz per frequency unit: 1, 1e3, 1e6 or 1e9 (default GHz).
    unit: f64 = 1e9,
    /// `s`, `y` or `z` (default S).
    kind: u8 = 's',
    /// How each value pair reads: magnitude-angle, dB-angle (degrees) or
    /// real-imaginary (default MA).
    format: Format = .ma,
    /// The `R` reference, > 0 (default 50).
    z0: f64 = 50,
    /// Network records, each a frequency then N² value pairs: `nf` rows of
    /// 1 + 2N² numbers, frequencies strictly ascending.
    values: []const f64,
    /// Network records in `values`, at least 1.
    nf: usize,

    /// The data format the option line names.
    pub const Format = enum { ma, db, ri };
};

const Cx = std.math.Complex(f64);

/// Most ports a file may have; the conversion scratch is sized from it.
const ports_max = 64;

/// Reads a Touchstone 1.0 file of `ports` ports (the `.sNp` extension's N)
/// to its option line and records. The option line is the first non-comment
/// line and later ones are ignored (TS1.1 p.4); one after the data is
/// malformed. A frequency that does not ascend ends the network data, and
/// what follows must be a 2-port's noise block, five values a row (TS1.1
/// p.10-11). Lines end at LF, CR+LF or a lone CR (TS2.0 rule 2); numbers are
/// plain or scientific, with no scale suffix (TS2.0 rule 6). Fails with
/// `InvalidTouchstone` on anything else, H and G data included (legal, not
/// read).
pub fn readTouchstone(arena: Allocator, bytes: []const u8, ports: usize) Error!Raw {
    if (ports == 0 or ports > ports_max) return error.InvalidTouchstone;
    var raw: Raw = .{ .values = &.{}, .nf = 0 };
    var nums: std.ArrayList(f64) = .empty;
    var option_seen = false;
    var rest = bytes;
    while (rest.len > 0) {
        const nl = std.mem.indexOfAny(u8, rest, "\r\n") orelse rest.len;
        const raw_line = rest[0..nl];
        rest = rest[nl..];
        if (std.mem.startsWith(u8, rest, "\r\n")) rest = rest[2..] else if (rest.len > 0) rest = rest[1..];
        const line = std.mem.trim(u8, raw_line[0 .. std.mem.indexOfScalar(u8, raw_line, '!') orelse raw_line.len], " \t");
        if (line.len == 0) continue;
        if (line[0] == '[') return error.InvalidTouchstone; // Touchstone 2.0 keywords
        if (line[0] == '#') {
            // Only the first option line counts, and it precedes the data.
            if (option_seen) continue;
            if (nums.items.len > 0) return error.InvalidTouchstone;
            option_seen = true;
            var it = std.mem.tokenizeAny(u8, line[1..], " \t");
            while (it.next()) |t| {
                var buf: [8]u8 = undefined;
                if (t.len > buf.len) return error.InvalidTouchstone;
                const w = std.ascii.lowerString(&buf, t);
                const units = std.StaticStringMap(f64).initComptime(.{ .{ "hz", 1 }, .{ "khz", 1e3 }, .{ "mhz", 1e6 }, .{ "ghz", 1e9 } });
                const formats = std.StaticStringMap(Raw.Format).initComptime(.{ .{ "ma", .ma }, .{ "db", .db }, .{ "ri", .ri } });
                if (units.get(w)) |u| {
                    raw.unit = u;
                } else if (formats.get(w)) |f| {
                    raw.format = f;
                } else if (std.mem.eql(u8, w, "s") or std.mem.eql(u8, w, "y") or std.mem.eql(u8, w, "z")) {
                    raw.kind = w[0];
                } else if (std.mem.eql(u8, w, "r")) {
                    raw.z0 = plainNumber(it.next() orelse return error.InvalidTouchstone) orelse return error.InvalidTouchstone;
                } else return error.InvalidTouchstone;
            }
            continue;
        }
        var it = std.mem.tokenizeAny(u8, line, " \t");
        while (it.next()) |t| try nums.append(arena, plainNumber(t) orelse return error.InvalidTouchstone);
    }
    if (!(raw.z0 > 0) or !std.math.isFinite(raw.z0)) return error.InvalidTouchstone;
    const stride = 1 + 2 * ports * ports;
    const all = nums.items;
    var nf: usize = 0;
    while ((nf + 1) * stride <= all.len) : (nf += 1) {
        const f = all[nf * stride];
        if (nf > 0 and !(f > all[(nf - 1) * stride])) break;
    }
    if (nf == 0) return error.InvalidTouchstone;
    // After the network data: nothing, or a 2-port's noise rows starting at
    // or below the last network frequency, ascending (TS1.1 p.10-11).
    const tail = all[nf * stride ..];
    if (tail.len > 0) {
        if (ports != 2 or tail.len % 5 != 0) return error.InvalidTouchstone;
        if (!(tail[0] <= all[(nf - 1) * stride])) return error.InvalidTouchstone;
        var k: usize = 5;
        while (k < tail.len) : (k += 5) if (!(tail[k] > tail[k - 5])) return error.InvalidTouchstone;
    }
    raw.values = all[0 .. nf * stride];
    raw.nf = nf;
    assert(raw.values.len == raw.nf * stride);
    return raw;
}

/// Parses a Touchstone 1.0 file of `ports` ports (`readTouchstone`) into
/// admittance samples. Y and Z data are normalized to R, as version 1.0
/// writes them. A two-port's entries come in the column order N11 N21 N12
/// N22. The tables, and the parse scratch, live in `arena`. Every sample
/// is finite: a non-finite value is `InvalidTouchstone`, a non-finite
/// admittance `SingularSParameters`.
pub fn parseTouchstone(arena: Allocator, bytes: []const u8, ports: usize) Error!Network {
    const raw = try readTouchstone(arena, bytes, ports);
    const z0 = raw.z0;
    const pp = ports * ports;
    const stride = 1 + 2 * pp;
    const nf = raw.nf;
    const freq = try arena.alloc(f64, nf);
    const re = try arena.alloc(f64, pp * nf);
    const im = try arena.alloc(f64, pp * nf);
    const m = try arena.alloc(Cx, 2 * pp);
    for (0..nf) |k| {
        const row = raw.values[k * stride ..][0..stride];
        freq[k] = row[0] * raw.unit;
        if (!std.math.isFinite(freq[k]) or freq[k] < 0) return error.InvalidTouchstone;
        for (0..pp) |q| {
            const a = row[1 + 2 * q];
            const b = row[2 + 2 * q];
            const v: Cx = switch (raw.format) {
                .ri => .{ .re = a, .im = b },
                .ma, .db => blk: {
                    const mag = if (raw.format == .db) std.math.pow(f64, 10, a / 20) else a;
                    const ang = b * std.math.pi / 180;
                    break :blk .{ .re = mag * @cos(ang), .im = mag * @sin(ang) };
                },
            };
            // A dB magnitude past ~6000 overflows; no fit reads an infinity.
            if (!std.math.isFinite(v.re) or !std.math.isFinite(v.im)) return error.InvalidTouchstone;
            // Two-port files list N11 N21 N12 N22.
            const e = if (ports == 2) (q % 2) * 2 + q / 2 else q;
            m[e] = v;
        }
        const y = m[pp..];
        switch (raw.kind) {
            's' => try sToY(ports, m[0..pp], y, z0),
            'y' => for (y, m[0..pp]) |*o, v| {
                o.* = v.mul(.{ .re = 1 / z0, .im = 0 });
            },
            else => {
                for (m[0..pp]) |*v| v.* = v.mul(.{ .re = z0, .im = 0 });
                // Y = Z⁻¹: solve Z·Y = I.
                for (y, 0..) |*o, e| o.* = .{ .re = if (e % (ports + 1) == 0) 1 else 0, .im = 0 };
                if (!solve(ports, m[0..pp], y)) return error.SingularSParameters;
            },
        }
        for (0..pp) |e| {
            // A near-singular I + S or Z overflows the admittance.
            if (!std.math.isFinite(y[e].re) or !std.math.isFinite(y[e].im)) return error.SingularSParameters;
            re[e * nf + k] = y[e].re;
            im[e * nf + k] = y[e].im;
        }
    }
    assert(re.len == pp * freq.len);
    return .{ .ports = ports, .freq = freq, .re = re, .im = im };
}

// ---- private ----

/// Y = Z0⁻¹·(I − S)(I + S)⁻¹ with one real reference impedance.
fn sToY(p: usize, s: []const Cx, y: []Cx, z0: f64) Error!void {
    assert(p <= ports_max);
    assert(s.len == p * p);
    var buf_a: [ports_max * ports_max]Cx = undefined;
    var buf_b: [ports_max * ports_max]Cx = undefined;
    const a = buf_a[0 .. p * p];
    const b = buf_b[0 .. p * p];
    for (0..p) |i| for (0..p) |j| {
        const id: f64 = if (i == j) 1 else 0;
        a[i * p + j] = s[i * p + j].add(.{ .re = id, .im = 0 });
        b[i * p + j] = (Cx{ .re = id, .im = 0 }).sub(s[i * p + j]);
    };
    // (I − S) and (I + S)⁻¹ commute, so this is (I + S)⁻¹(I − S).
    if (!solve(p, a, b)) return error.SingularSParameters;
    for (y, b) |*o, v| o.* = v.mul(.{ .re = 1 / z0, .im = 0 });
}

/// Solves A·X = B in place (B becomes X), Gauss-Jordan with partial
/// pivoting; false when A is singular.
fn solve(p: usize, a: []Cx, b: []Cx) bool {
    assert(a.len == p * p);
    assert(b.len == p * p);
    for (0..p) |c| {
        var piv = c;
        for (c + 1..p) |r| if (a[r * p + c].magnitude() > a[piv * p + c].magnitude()) {
            piv = r;
        };
        if (a[piv * p + c].magnitude() == 0) return false;
        for (0..p) |j| {
            std.mem.swap(Cx, &a[piv * p + j], &a[c * p + j]);
            std.mem.swap(Cx, &b[piv * p + j], &b[c * p + j]);
        }
        const inv = (Cx{ .re = 1, .im = 0 }).div(a[c * p + c]);
        for (0..p) |j| {
            a[c * p + j] = a[c * p + j].mul(inv);
            b[c * p + j] = b[c * p + j].mul(inv);
        }
        for (0..p) |r| {
            if (r == c) continue;
            const f = a[r * p + c];
            for (0..p) |j| {
                a[r * p + j] = a[r * p + j].sub(f.mul(a[c * p + j]));
                b[r * p + j] = b[r * p + j].sub(f.mul(b[c * p + j]));
            }
        }
    }
    return true;
}

test "touchstone: a matched attenuator converts to its admittance" {
    // 6.02 dB matched pad: S11 = S22 = 0, S21 = S12 = 0.5, 50 Ω.
    var arena_state = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const net = try parseTouchstone(arena, "! pad\n# MHz S RI R 50\n1 0 0 0.5 0 0.5 0 0 0\n2 0 0 0.5 0 0.5 0 0 0\n", 2);
    try testing.expectEqual(@as(usize, 2), net.freq.len);
    try testing.expectEqual(@as(f64, 2e6), net.freq[1]);
    // Y = (1/50)·(I − S)(I + S)⁻¹ = (1/50)·[[5/3, −4/3], [−4/3, 5/3]].
    try testing.expectApproxEqRel(@as(f64, 1.0 / 30.0), net.at(0, 0).re, 1e-12);
    try testing.expectApproxEqRel(@as(f64, -1.0 / 37.5), net.at(1, 0).re, 1e-12);
}

test "touchstone: Y, Z and dB data convert, and the noise block is skipped" {
    var arena_state = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    // Normalized y = 1 and z = 1 are both 1/50 S; |S| = 0.5 (−6.02 dB) at 0°
    // is (1 − 0.5)/(1 + 0.5)/50 = 1/150 S.
    for ([_][]const u8{ "# Hz Y RI R 50\n1 1 0\n", "# Hz Z RI R 50\n1 1 0\n", "# Hz S DB R 50\n1 -6.020599913279624 0\n" }, [_]f64{ 1.0 / 50.0, 1.0 / 50.0, 1.0 / 150.0 }) |file, want| {
        const net = try parseTouchstone(arena, file, 1);
        try testing.expectEqual(@as(f64, 1), net.freq[0]);
        try testing.expectApproxEqRel(want, net.at(0, 0).re, 1e-12);
        try testing.expectApproxEqAbs(@as(f64, 0), net.at(0, 0).im, 1e-15);
    }
    // Two noise rows (5 numbers each) after the data: their first frequency
    // does not ascend, so the data ends at two points.
    const net = try parseTouchstone(arena, "# Hz S RI R 50\n1 0 0 0.5 0 0.5 0 0 0\n2 0 0 0.5 0 0.5 0 0 0\n1 2.0 0.5 0 0.1\n2 2.0 0.5 0 0.1\n", 2);
    try testing.expectEqual(@as(usize, 2), net.freq.len);
}

test "touchstone: malformed files are refused, never misread" {
    var arena_state = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    try testing.expectError(error.InvalidTouchstone, parseTouchstone(arena, "1 0 0\n", 0));
    try testing.expectError(error.InvalidTouchstone, parseTouchstone(arena, "1 0 0\n", 65));
    for ([_][]const u8{
        "", // no data
        "! only a comment\n",
        "[Version] 2.0\n1 0 0\n", // Touchstone 2.0
        "# Hz Q RI\n1 0 0\n", // unknown option
        "# frequency\n1 0 0\n", // option token too long to be one
        "# Hz S RI R 0\n1 0 0\n", // z0 must be positive
        "# Hz S RI R\n1 0 0\n", // R without a value
        "# Hz S RI R 50\n-1 0 0\n", // negative frequency
        "# Hz S RI R 50\n1 x 0\n", // not a number
        "# Hz S RI R 50\n1 0\n", // a short row
    }) |file| try testing.expectError(error.InvalidTouchstone, parseTouchstone(arena, file, 1));
    // S = −1 makes I + S singular.
    try testing.expectError(error.SingularSParameters, parseTouchstone(arena, "# Hz S RI R 50\n1 -1 0\n", 1));
}

test "touchstone: option-line and data rules follow the 1.1 spec (#121)" {
    var arena_state = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    // A unit glued to `#` still counts (it read as GHz, 1000x off).
    try testing.expectEqual(@as(f64, 1e6), (try readTouchstone(arena, "#MHz S RI R 50\n1 0 0\n", 1)).unit);
    // Only the first option line counts (TS1.1 p.4).
    const twice = try readTouchstone(arena, "# Hz S RI R 50\n# GHz Y MA R 75\n1 0 0\n", 1);
    try testing.expectEqual(@as(f64, 1), twice.unit);
    try testing.expectEqual(@as(u8, 's'), twice.kind);
    try testing.expectEqual(@as(f64, 50), twice.z0);
    // A lone CR ends a line (TS2.0 rule 2).
    try testing.expectEqual(@as(usize, 2), (try readTouchstone(arena, "# Hz S RI R 50\r1 0 0 ! c\r2 0 0\r", 1)).nf);
    for ([_][]const u8{
        "1 0 0\n# Hz S RI R 50\n2 0 0\n", // option line after the data
        "# Hz S RI R 50\n1 nan 0\n", // not a Touchstone number
        "# Hz S RI R 50\n1 0x1p3 0\n",
        "# Hz S RI R 50\n1 1k 0\n", // no scale suffixes
        "# Hz S RI R 50\n2 0 0\n1 0 0\n", // a 1-port's frequencies must ascend
        "# Hz S RI R 50\n1 0 0\n2 0\n", // a short trailing record
        "# Hz H RI R 50\n1 0 0\n", // H data is not read
    }) |file| try testing.expectError(error.InvalidTouchstone, readTouchstone(arena, file, 1));
    // A 2-port's noise rows: five values, starting at or below the last
    // network frequency, ascending.
    const two = "# Hz S RI R 50\n1 0 0 0.5 0 0.5 0 0 0\n2 0 0 0.5 0 0.5 0 0 0\n";
    try testing.expectEqual(@as(usize, 2), (try readTouchstone(arena, two ++ "2 2.0 0.5 0 0.1\n", 2)).nf);
    try testing.expectError(error.InvalidTouchstone, readTouchstone(arena, two ++ "1 2.0 0.5 0\n", 2));
    try testing.expectError(error.InvalidTouchstone, readTouchstone(arena, two ++ "1 2 0.5 0 0.1\n1 2 0.5 0 0.1\n", 2));
}
