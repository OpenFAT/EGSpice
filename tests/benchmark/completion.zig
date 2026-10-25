//! Independent output completion, using requests from the input deck rather
//! than the final point of a second simulator. Repeated times are valid.
const std = @import("std");
const compare = @import("compare.zig");
const spice = @import("vacask.zig");
const number = spice.parseSpice;
const Allocator = std.mem.Allocator;

/// The analyses a contract can request. `noise_integrated` is the one-point
/// total plot that every `.noise` request implies.
pub const Kind = enum { op, dc, ac, tran, noise, noise_integrated };
/// One analysis the deck asked for, and what its plot must cover. `stop` is
/// the last grid point the simulator generates, which for a log sweep can
/// fall short of `requested_stop`, the deck's literal FSTOP. `points` 0
/// leaves the point count unchecked; empty `signals` checks no names.
pub const Request = struct {
    kind: Kind,
    start: f64 = 0,
    stop: f64 = 0,
    requested_stop: ?f64 = null,
    points: usize = 0,
    signals: []const []const u8 = &.{},
};
/// Everything one run must write to count as complete. Serialized beside
/// the deck as `*.completion.json` and into the samples JSONL, so its field
/// names are a file format. A non-empty `unsupported` fails every run.
pub const Contract = struct {
    schema_version: u32 = 1,
    requests: []const Request = &.{},
    unsupported: []const []const u8 = &.{},
};

// Mirrors core/numerics.FreqSweep without importing the simulator into the
// independent benchmark runner. Its stopping tolerance selects grid points;
// output coverage below still uses rounding tolerance only.
fn frequencyGrid(request: *Request, mode: []const u8, points: f64) bool {
    const lin = std.mem.eql(u8, mode, "lin");
    const dec = std.mem.eql(u8, mode, "dec");
    const oct = std.mem.eql(u8, mode, "oct");
    if ((!lin and !dec and !oct) or !std.math.isFinite(points) or points < 1 or points != @floor(points) or points > std.math.maxInt(u32)) return false;
    const start = request.start;
    const stop = request.stop;
    if (!std.math.isFinite(start) or !std.math.isFinite(stop) or stop < start or start < 0 or (!lin and start == 0)) return false;
    request.requested_stop = stop;
    if (lin) {
        request.points = if (stop != start) @intFromFloat(points) else 1;
        if (request.points == 1) request.stop = start;
        return true;
    }
    const ratio = stop / start;
    const stretched = if (request.kind == .ac and dec and stop / 10 >= start) @floor(std.math.log10(ratio) * points) else 0;
    const cap: f64 = std.math.maxInt(u32) - 1;
    if (stretched >= 1 and std.math.isFinite(stretched)) {
        request.points = @as(usize, @intFromFloat(@min(stretched, cap))) + 1;
        request.stop = start * std.math.pow(f64, ratio, @as(f64, @floatFromInt(request.points - 1)) / stretched);
    } else {
        const base: f64 = if (oct) 2 else 10;
        const ln_step = @log(base) / points;
        const ln_span = if (std.math.isFinite(ratio)) @log(ratio) else @log(stop) - @log(start);
        const steps = (ln_span + std.math.log1p(@exp(ln_step) * 1e-3)) / ln_step;
        request.points = @as(usize, @intFromFloat(@min(@floor(steps), cap))) + 1;
        request.stop = start * std.math.pow(f64, base, @as(f64, @floatFromInt(request.points - 1)) / points);
    }
    return std.math.isFinite(request.stop);
}

/// The contract implied by deck `text`: one request per analysis card, with
/// the `.save`d signals, or else the `v(...)`/`i(...)` columns `oracle`
/// (the deck's `.expected.json`) lists for that analysis. A card this file
/// cannot parse lands in `unsupported` rather than failing the call. All
/// slices are allocated in `a`; pass an arena.
pub fn fromDeck(a: Allocator, text: []const u8, oracle: ?[]const u8) !Contract {
    var requests: std.ArrayList(Request) = .empty;
    var unsupported: std.ArrayList([]const u8) = .empty;
    var saves: std.ArrayList([]const u8) = .empty;
    var cards: std.ArrayList(spice.Card) = .empty;
    var title: []const u8 = "";
    try spice.lex(a, text, &title, &cards);
    for (cards.items) |logical| {
        var words = std.mem.tokenizeAny(u8, logical.source, " \t\r");
        const card = words.next() orelse continue;
        if (std.mem.eql(u8, card, ".save") or std.mem.eql(u8, card, ".probe")) {
            var index: usize = 1;
            while (index < logical.fields.len) {
                const field = logical.fields[index];
                if (std.mem.eql(u8, field, "all")) {
                    index += 1;
                    continue;
                }
                if ((std.mem.eql(u8, field, "v") or std.mem.eql(u8, field, "i")) and index + 1 < logical.fields.len) {
                    try saves.append(a, try std.fmt.allocPrint(a, "{s}({s})", .{ field, logical.fields[index + 1] }));
                    index += 2;
                } else {
                    try unsupported.append(a, "unsupported saved expression");
                    break;
                }
            }
            continue;
        }
        const kind: Kind = if (std.mem.eql(u8, card, ".tran")) .tran else if (std.mem.eql(u8, card, ".op")) .op else if (std.mem.eql(u8, card, ".dc")) .dc else if (std.mem.eql(u8, card, ".ac")) .ac else if (std.mem.eql(u8, card, ".noise")) .noise else {
            for ([_][]const u8{ ".pz", ".four", ".hb", ".pss", ".sp", ".sens", ".tf", ".step" }) |other| {
                if (std.mem.eql(u8, card, other)) try unsupported.append(a, card);
            }
            continue;
        };
        var request: Request = .{ .kind = kind };
        parse: {
            switch (kind) {
                .op => {},
                .tran => {
                    _ = words.next() orelse break :parse;
                    request.stop = number(words.next() orelse break :parse) orelse break :parse;
                    if (words.next()) |start| if (!std.mem.eql(u8, start, "uic")) {
                        request.start = number(start) orelse break :parse;
                    };
                },
                .noise => {
                    if (logical.fields.len < 8 or !std.mem.eql(u8, logical.fields[1], "v")) break :parse;
                    request.signals = &.{ "v(onoise_spectrum)", "v(inoise_spectrum)" };
                    request.start = number(logical.fields[6]) orelse break :parse;
                    request.stop = number(logical.fields[7]) orelse break :parse;
                    if (!frequencyGrid(&request, logical.fields[4], number(logical.fields[5]) orelse break :parse)) break :parse;
                },
                .noise_integrated => unreachable,
                .ac => {
                    const mode = words.next() orelse break :parse;
                    const points = number(words.next() orelse break :parse) orelse break :parse;
                    request.start = number(words.next() orelse break :parse) orelse break :parse;
                    request.stop = number(words.next() orelse break :parse) orelse break :parse;
                    if (!frequencyGrid(&request, mode, points)) break :parse;
                },
                .dc => {
                    request.points = 1;
                    var axes: usize = 0;
                    while (words.next()) |_| {
                        const start = number(words.next() orelse break :parse) orelse break :parse;
                        const stop = number(words.next() orelse break :parse) orelse break :parse;
                        const step = number(words.next() orelse break :parse) orelse break :parse;
                        const n = (stop - start) / step;
                        if (!std.math.isFinite(n) or n < 0 or n > 1e9) break :parse;
                        request.points = std.math.mul(usize, request.points, @as(usize, @intFromFloat(n + 1e-8)) + 1) catch break :parse;
                        axes += 1;
                    }
                    if (axes == 0) break :parse;
                },
            }
            try requests.append(a, request);
            if (kind == .noise) try requests.append(a, .{ .kind = .noise_integrated, .signals = &.{ "v(onoise_total)", "v(inoise_total)" } });
            continue;
        }
        try unsupported.append(a, card);
    }
    for (requests.items) |*request| {
        if (request.kind == .noise or request.kind == .noise_integrated) continue;
        if (saves.items.len > 0) {
            request.signals = saves.items;
        } else if (oracle) |bytes| {
            const parsed = std.json.parseFromSliceLeaky(std.json.Value, a, bytes, .{}) catch continue;
            if (parsed != .object) continue;
            const expect = parsed.object.get("expect") orelse continue;
            if (expect != .object) continue;
            const plots = expect.object.get("plots") orelse continue;
            if (plots != .array) continue;
            var signals: std.ArrayList([]const u8) = .empty;
            for (plots.array.items) |plot| {
                if (plot != .object) continue;
                const name = plot.object.get("name") orelse continue;
                if (name != .string or plotKind(name.string) != request.kind) continue;
                const columns = plot.object.get("columns") orelse continue;
                if (columns != .object) continue;
                for (columns.object.keys()) |key| {
                    if (std.mem.indexOfAny(u8, key, "#:!") != null) continue;
                    if (std.mem.startsWith(u8, key, "v(") or std.mem.startsWith(u8, key, "i(")) try signals.append(a, try std.ascii.allocLowerString(a, key));
                }
            }
            request.signals = signals.items;
        }
    }
    return .{ .requests = requests.items, .unsupported = unsupported.items };
}

fn plotKind(name: []const u8) ?Kind {
    var buf: [128]u8 = undefined;
    if (name.len > buf.len) return null;
    const lower = std.ascii.lowerString(&buf, name);
    if (std.mem.indexOf(u8, lower, "transient") != null) return .tran;
    if (std.mem.indexOf(u8, lower, "noise") != null) return if (std.mem.indexOf(u8, lower, "integrated") != null) .noise_integrated else .noise;
    if (std.mem.startsWith(u8, lower, "ac ")) return .ac;
    if (std.mem.indexOf(u8, lower, "operating point") != null) return .op;
    if (std.mem.eql(u8, lower, "dc transfer characteristic")) return .dc;
    return null;
}

/// Null means complete; otherwise an independent reason for rejecting the run.
pub fn check(a: Allocator, plots: []const compare.Plot, contract: Contract) !?[]const u8 {
    if (contract.schema_version != 1 or contract.unsupported.len > 0) return "UnsupportedCompletionContract";
    if (contract.requests.len == 0) return "MissingCompletionContract";
    const taken = try a.alloc(bool, plots.len);
    defer a.free(taken);
    @memset(taken, false);
    for (contract.requests) |request| {
        if (!std.math.isFinite(request.start) or !std.math.isFinite(request.stop) or request.start > request.stop) return "InvalidCompletionContract";
        const index = for (plots, 0..) |plot, i| {
            const kind = plotKind(plot.plotname);
            if (!taken[i] and (kind == request.kind or (request.kind == .dc and kind == .op) or (request.kind == .op and kind == .dc and plot.npoints == 1))) break i;
        } else return "MissingRequestedPlot";
        taken[index] = true;
        const plot = plots[index];
        if (plot.npoints == 0) return "EmptyRequestedPlot";
        for (plot.data) |value| if (!std.math.isFinite(value)) return "NonfiniteSamples";
        for (request.signals) |signal| {
            var found = false;
            for (plot.varnames) |name| {
                // VACASK's raw noise PSD/gain has not been converted yet.
                found = found or std.mem.eql(u8, signal, name) or (request.kind == .noise and
                    ((std.mem.eql(u8, signal, "v(onoise_spectrum)") and std.mem.eql(u8, name, "v(onoise)")) or
                        (std.mem.eql(u8, signal, "v(inoise_spectrum)") and std.mem.eql(u8, name, "v(gain)"))));
            }
            if (!found) return "MissingRequestedSignal";
        }
        if ((request.kind == .op or request.kind == .noise_integrated) and plot.npoints != 1) return "OperatingPointSampleCount";
        if (request.kind == .dc and plot.npoints != request.points) return "IncompleteDcSweep";
        if ((request.kind == .ac or request.kind == .noise) and request.points != 0 and plot.npoints != request.points) return "IncompleteFrequencySweep";
        if (request.kind == .tran or request.kind == .ac or request.kind == .noise) {
            const axis = if (request.kind == .tran) "time" else "frequency";
            const column = for (plot.varnames, 0..) |name, i| {
                if (std.mem.eql(u8, axis, name)) break i;
            } else return "MissingRequestedAxis";
            if (plot.nvars < 2) return "MissingWaveformSignals";
            const parts: usize = if (plot.is_complex) 2 else 1;
            const stride = parts * plot.nvars;
            const first = plot.data[column * parts];
            var previous = first;
            for (1..plot.npoints) |i| {
                const value = plot.data[i * stride + column * parts];
                if (value < previous) return "DecreasingRequestedAxis";
                previous = value;
            }
            const epsilon = @max(@max(@abs(request.start), @abs(request.stop)), 1e-300) * 64 * std.math.floatEps(f64);
            // TSTART suppresses earlier output; it does not force an accepted
            // point exactly at TSTART. At zero (including UIC) the initial
            // sample is part of the expected waveform.
            const require_start = request.kind != .tran or request.start <= 0;
            if ((require_start and first > request.start + epsilon) or previous < request.stop - epsilon) return "IncompleteRequestedInterval";
        }
    }
    return null;
}

test "literal requests retain stop time and saved signals" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const contract = try fromDeck(arena.allocator(), "title\n.save v(out)\n.tran 2p 200p 0 2p uic\n", null);
    try std.testing.expectEqual(@as(usize, 1), contract.requests.len);
    try std.testing.expectApproxEqRel(@as(f64, 200e-12), contract.requests[0].stop, 1e-15);
    try std.testing.expectEqualStrings("v(out)", contract.requests[0].signals[0]);
}

test "equally truncated runs do not establish completion" {
    const t = std.testing;
    const plots = [_]compare.Plot{.{ .plotname = "Transient Analysis", .varnames = &.{ "time", "v(out)" }, .is_complex = false, .npoints = 3, .nvars = 2, .data = &.{ 0, 0, 1, 1, 1, 2 } }};
    const contract: Contract = .{ .requests = &.{.{ .kind = .tran, .stop = 2, .signals = &.{"v(out)"} }} };
    try t.expectEqualStrings("IncompleteRequestedInterval", (try check(t.allocator, &plots, contract)).?);
    var complete = plots;
    complete[0].data = &.{ 0, 0, 1, 1, 2, 2 };
    try t.expectEqual(@as(?[]const u8, null), try check(t.allocator, &complete, contract));
    complete[0].varnames = &.{ "time", "v(other)" };
    try t.expectEqualStrings("MissingRequestedSignal", (try check(t.allocator, &complete, contract)).?);
    try t.expectEqualStrings("MissingRequestedPlot", (try check(t.allocator, &.{}, contract)).?);
}

test "TSTART suppresses output without requiring an accepted sample at its boundary" {
    const t = std.testing;
    const plots = [_]compare.Plot{.{ .plotname = "Transient Analysis", .varnames = &.{ "time", "v(out)" }, .is_complex = false, .npoints = 2, .nvars = 2, .data = &.{ 1.1, 0, 2, 1 } }};
    const window: Contract = .{ .requests = &.{.{ .kind = .tran, .start = 1, .stop = 2 }} };
    try t.expectEqual(@as(?[]const u8, null), try check(t.allocator, &plots, window));
    const uic: Contract = .{ .requests = &.{.{ .kind = .tran, .start = 0, .stop = 2 }} };
    try t.expectEqualStrings("IncompleteRequestedInterval", (try check(t.allocator, &plots, uic)).?);
}

test "logical saves include continuation and spaced probes but exclude comments" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const contract = try fromDeck(arena.allocator(), "title\n.save V ( q ) ; v(comment)\n+ v(qb) $ i(comment)\n.tran 1p 2p; .op\n", null);
    try std.testing.expectEqual(@as(usize, 0), contract.unsupported.len);
    try std.testing.expectEqual(@as(usize, 1), contract.requests.len);
    try std.testing.expectEqual(@as(usize, 2), contract.requests[0].signals.len);
    try std.testing.expectEqualStrings("v(q)", contract.requests[0].signals[0]);
    try std.testing.expectEqualStrings("v(qb)", contract.requests[0].signals[1]);
}

test "noise requires both spectrum and integrated output" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const contract = try fromDeck(arena.allocator(), "title\n.noise v(out) vin dec 10 1 1k\n", null);
    try std.testing.expectEqual(@as(usize, 2), contract.requests.len);
    try std.testing.expectEqual(Kind.noise_integrated, contract.requests[1].kind);
}

test "native hierarchical voltage outputs satisfy original SPICE save contracts" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const contract = try fromDeck(a, "title\n.save v(xb0_0.q) v(xbank.xcell.q)\n.op\n", null);
    const raw =
        \\Title: Hierarchical saves
        \\Plotname: Operating Point
        \\Flags: real
        \\No. Variables: 2
        \\No. Points: 1
        \\Variables:
        \\0 xb0_0:q voltage
        \\1 xbank:xcell:q voltage
        \\Values:
        \\0 0.5
        \\0.5
        \\
    ;
    const plots = compare.parseRawBlob(a, raw) orelse return error.InvalidRawFixture;
    try std.testing.expectEqualStrings("v(xb0_0.q)", plots[0].varnames[0]);
    try std.testing.expectEqualStrings("v(xbank.xcell.q)", plots[0].varnames[1]);
    try std.testing.expectEqual(@as(?[]const u8, null), try check(a, plots, contract));
}

test "frequency contract covers generated grid rather than literal upper bound" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const Case = struct { card: []const u8, points: usize, last: f64, requested_stop: f64 };
    for ([_]Case{
        .{ .card = "title\n.ac dec 3 10 730\n", .points = 6, .last = 730, .requested_stop = 730 },
        .{ .card = "title\n.ac dec 10 1 9.999\n", .points = 11, .last = 10, .requested_stop = 9.999 },
        .{ .card = "title\n.ac oct 3 10 100\n", .points = 10, .last = 80, .requested_stop = 100 },
        .{ .card = "title\n.ac lin 1 10 1000\n", .points = 1, .last = 10, .requested_stop = 1000 },
        .{ .card = "title\n.ac lin 9 0 1000\n", .points = 9, .last = 1000, .requested_stop = 1000 },
        .{ .card = "title\n.ac dec 10 100 100\n", .points = 1, .last = 100, .requested_stop = 100 },
        .{ .card = "title\n.noise v(out) vin dec 3 10 730\n", .points = 6, .last = 464.158883361278, .requested_stop = 730 },
    }) |case| {
        const contract = try fromDeck(arena.allocator(), case.card, null);
        try std.testing.expectEqual(@as(usize, 0), contract.unsupported.len);
        const request = contract.requests[0];
        try std.testing.expectEqual(case.points, request.points);
        try std.testing.expectApproxEqRel(case.last, request.stop, 1e-13);
        try std.testing.expectEqual(case.requested_stop, request.requested_stop.?);
    }
    for ([_][]const u8{ "title\n.ac dec 0 1 10\n", "title\n.ac oct 2 10 1\n", "title\n.ac lin 3 2 1\n", "title\n.ac dec 2 0 10\n" }) |card| {
        const contract = try fromDeck(arena.allocator(), card, null);
        try std.testing.expectEqual(@as(usize, 1), contract.unsupported.len);
    }
    const contract = try fromDeck(arena.allocator(), "title\n.ac oct 1 10 30\n", null);
    var plots = [_]compare.Plot{.{ .plotname = "AC Analysis", .varnames = &.{ "frequency", "v(out)" }, .is_complex = false, .npoints = 2, .nvars = 2, .data = &.{ 10, 1, 20, 2 } }};
    try std.testing.expectEqual(@as(?[]const u8, null), try check(arena.allocator(), &plots, contract));
    plots[0].npoints = 1;
    plots[0].data = &.{ 10, 1 };
    try std.testing.expectEqualStrings("IncompleteFrequencySweep", (try check(arena.allocator(), &plots, contract)).?);
}
