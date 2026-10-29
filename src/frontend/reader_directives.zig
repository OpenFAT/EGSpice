//! Directive cards: analyses, `.options`, `.ic`, `.save`, `.step`, `.mosra`,
//! `.sample`, and the global `.param` fold.
//! Cites: ngspice inp.c; HSPICE [CR] .SAVE, .SAMPLE, .MOSRA, [SA] Ch.29.
//!
//! Contents:
//! - `pub fn Methods(comptime S: type) type`
//! - `pub fn readArgs(r: *R, f: *F) Error![]const Value`
//! - `pub fn readVpiOptions(r: *R, line: []const u8) Error!void`
//! - `pub fn isStep(line: []const u8) bool`
//! - `pub fn readDirective(r: *R, line: []const u8) Error!void`
//! - `pub fn readMosra(r: *R, line: []const u8, args: []const Value) Error!Mosra`
//! - `pub fn readSample(args: []const Value) Error!core.query.NoiseSample`
//! - `pub fn readSaveOp(r: *R, line: []const u8, f: *F) Error!void`
//! - `pub fn written(r: *const R, line: []const u8) []const u8`
//! - `pub fn failed(r: *R, line: []const u8, err: Error) Error`
//! - `pub fn liveParam(r: *R, name: []const u8) Error!?u32`
//! - `pub fn foldGlobals(r: *R) Error!void`
//! - `pub fn stepTarget(r: *R, name: []const u8) Error!StepTarget`
//! - `pub fn readStep(r: *R, args: []const Value) Error!Step`

const std = @import("std");
const Allocator = std.mem.Allocator;
const lines = @import("lines.zig");
const spectre_rewrite = @import("spectre.zig");
const source = @import("source.zig");
const hypergraph = @import("hypergraph.zig");
const expr = @import("expr.zig");
const measure = @import("measure.zig");
const pattern = @import("pattern.zig");
const core = @import("core");
const Name = core.Name;
const InternPool = core.InternPool;
const assert = std.debug.assert;
const reader = @import("reader.zig");
const card_rows = @import("card_rows.zig");
const card_table = @import("card_table.zig");
const reader_frame = @import("reader_frame.zig");
const sweep_points = @import("sweep_points.zig");
const Error = card_rows.Error;
const Value = card_rows.Value;
const StepTarget = card_rows.StepTarget;
const Step = card_rows.Step;
const appendAnalysis = reader_frame.appendAnalysis;
const Mosra = card_rows.Mosra;
const cardOf = card_table.cardOf;
const isSn = card_table.isSn;
const nodeText = reader_frame.nodeText;
const Frame = reader_frame.Frame;
const setsKey = card_table.setsKey;
const Grid = sweep_points.Grid;
const grids = sweep_points.grids;
const points = sweep_points.points;

/// The stage's `Reader` methods over dialect syntax `S`; `Reader(S)`
/// aliases each one as its own decl.
pub fn Methods(comptime S: type) type {
    const F = S.Split;
    return struct {
        const R = reader.Reader(S);
        const max_trials = R.max_trials;

        pub fn readArgs(r: *R, f: *F) Error![]const Value {
            const top: Frame = .{ .scopes = &r.global_scopes };
            var args: std.ArrayList(Value) = .empty;
            while (f.next()) |t| {
                if (t[0] == ',' or t[0] == '=') continue;
                const last = if (args.items.len != 0) args.items[args.items.len - 1] else Value{ .num = 0 };
                if (last == .name and std.mem.eql(u8, last.name, "monte") and std.ascii.eqlIgnoreCase(t, "list")) {
                    // `MONTE=list(10 20:30)`, parentheses optional: the trial
                    // numbers follow `list`, each `a:b` range expanded.
                    try args.append(r.arena, .{ .name = "list" });
                    var trials: f64 = 0;
                    while (f.next()) |n| {
                        if (n[0] == ')') break;
                        if (n[0] == '(' or n[0] == ',') continue;
                        const colon = std.mem.indexOfScalar(u8, n, ':') orelse n.len;
                        const lo = S.parseNum(n[0..colon]) orelse return error.ParseError;
                        const hi = if (colon == n.len) lo else S.parseNum(n[colon + 1 ..]) orelse return error.ParseError;
                        if (!(lo >= 1 and hi >= lo and hi <= std.math.maxInt(u32)) or lo != @trunc(lo) or hi != @trunc(hi)) return error.ParseError;
                        trials += hi - lo + 1;
                        if (trials > max_trials) return error.ParseError;
                        var k = lo;
                        while (k <= hi) : (k += 1) try args.append(r.arena, .{ .num = k });
                    }
                    continue;
                }
                try args.append(r.arena, try r.valueAt(t, f, &top, false, false));
            }
            return args.items;
        }

        /// A `.options` card that sets `vpi=` (ESPice's, #197): each
        /// `vpi="app.so"` names a VPI application (LRM 12.33) for `vpi_apps`,
        /// path case kept; the card's other options read as any `.options`.
        pub fn readVpiOptions(r: *R, line: []const u8) Error!void {
            var f = F.init(line);
            _ = f.next(); // the card
            var rest: std.ArrayList(u8) = .empty;
            try rest.appendSlice(r.arena, ".options");
            while (f.next()) |t| {
                if (std.mem.eql(u8, t, "vpi")) {
                    const eq = f.next() orelse return r.failed(line, error.ParseError);
                    if (eq[0] != '=') return r.failed(line, error.ParseError);
                    try r.vpi_apps.append(r.arena, r.pathOf(&f) catch |err| return r.failed(line, err));
                    continue;
                }
                try rest.append(r.arena, ' ');
                try rest.appendSlice(r.arena, t);
            }
            var g = F.init(rest.items);
            _ = g.next();
            const args = r.readArgs(&g) catch |err| return r.failed(line, err);
            try r.config.append(r.arena, .{ .temp = false, .args = args, .line = r.written(line) });
        }

        /// True for a `.step` directive line.
        pub fn isStep(line: []const u8) bool {
            var f = F.init(line);
            const c = cardOf(f.next().?[1..]) orelse return false;
            return c == .step;
        }

        pub fn readDirective(r: *R, line: []const u8) Error!void {
            r.card = line;
            var f = F.init(line);
            const card = cardOf(f.next().?[1..]);
            // Read once every analysis card is: HSPICE's untyped `.meas`
            // takes the last one.
            if (card != null and card.? == .meas) return r.meas_lines.append(r.arena, line);
            const c = card orelse return r.unsupported(line, "unsupported card");
            if (c == .ignored) return;
            if (c == .save and r.dialect == .hspice) return r.readSaveOp(line, &f);
            // HSPICE `.store` checkpoints the process on a wall-clock
            // schedule for an OS-level restore [CR .STORE]; the results do not
            // depend on it.
            if (c == .store) return std.log.warn("netlist: .store: checkpoints are not written; the run is not restartable", .{});
            if (c == .options and setsKey(line, "vpi")) return r.readVpiOptions(line);
            const args = r.readArgs(&f) catch |err| return r.failed(line, err);
            switch (c) {
                .step => try r.steps.append(r.arena, r.readStep(args) catch |err| return r.failed(line, err)),
                .analysis => |kind| {
                    // HSPICE's `.temp t1 t2 ...` lists run temperatures; ngspice's
                    // three-number form is a sweep.
                    const temp_list = kind == .temp and args.len > 1 and r.dialect == .hspice;
                    const swept = r.readSweep(kind, args) catch |err| return r.failed(line, err);
                    if (!temp_list) try appendAnalysis(r.arena, &r.analyses, .{ .kind = kind, .args = swept.args, .sweep = swept.sweep, .line = r.written(line), .dialect = r.dialect, .sn = isSn(line) });
                    if (kind == .temp and (args.len == 1 or temp_list)) try r.config.append(r.arena, .{ .temp = true, .args = args, .line = r.written(line) });
                },
                .options => try r.config.append(r.arena, .{ .temp = false, .args = args, .line = r.written(line) }),
                .ic => try r.ic_cards.append(r.arena, args),
                .sample => r.sample = readSample(args) catch |err| return r.failed(line, err),
                .mosra => r.mosra = try r.readMosra(line, args),
                .appendmodel => {
                    // `.appendmodel src mosra dst nmos`; HSPICE's other
                    // appended kinds (a second model card) are not modeled.
                    if (args.len != 4 or args[0] != .name or args[2] != .name or args[1] != .name or !std.mem.eql(u8, args[1].name, "mosra"))
                        return r.unsupported(line, ".appendmodel other than `src mosra dst type`");
                    try r.appendmodels.append(r.arena, .{ .src = args[0].name, .dst = args[2].name });
                },
                .nodeset => try r.nodeset_cards.append(r.arena, args),
                .global => for (args) |a| {
                    var buf: [24]u8 = undefined;
                    const name = nodeText(a, &buf) orelse return error.ParseError;
                    try r.global_nets.put(r.arena, try r.arena.dupe(u8, name), {});
                },
                .connect => {
                    if (args.len != 2) return error.ParseError;
                    // ponytail: top level only; a subcircuit's `.connect` needs
                    // per-instance aliases.
                    for (r.subckts.items) |s| for (r.lines.items[s.first..s.end]) |l| if (l.ptr == line.ptr)
                        return r.unsupported(line, ".connect inside a subcircuit");
                    var bufs: [2][24]u8 = undefined;
                    try r.connect(nodeText(args[0], &bufs[0]) orelse return error.ParseError, nodeText(args[1], &bufs[1]) orelse return error.ParseError);
                },
                .save => for (args) |a| switch (a) {
                    .group => |g| if (g.args().len == 1 and (std.ascii.eqlIgnoreCase(g.name(), "v") or std.ascii.eqlIgnoreCase(g.name(), "i"))) {
                        var buf: [24]u8 = undefined;
                        const name = nodeText(g.args()[0], &buf) orelse continue;
                        const label = try r.arena.print("{c}({s})", .{ std.ascii.toLower(g.name()[0]), name });
                        try r.saves.append(r.arena, std.ascii.lowerString(label, label));
                    },
                    .name => |n| if (std.ascii.eqlIgnoreCase(n, "all")) {
                        r.save_all = true;
                    },
                    else => {},
                },
                else => {},
            }
        }

        /// HSPICE `.mosra k=v ...` [SA Ch.29]. Keys outside `Key` (DEC, LIN,
        /// AgingPeriod, AgingInst, ...) and SimMode 1 or 3 are refused.
        pub fn readMosra(r: *R, line: []const u8, args: []const Value) Error!Mosra {
            var m: Mosra = .{ .line = r.written(line) };
            var i: usize = 0;
            while (i + 1 < args.len) : (i += 2) {
                if (args[i] != .name or args[i + 1] != .num) return r.failed(line, error.ParseError);
                const v = args[i + 1].num;
                const Key = enum { reltotaltime, relstarttime, relstep, simmode, relmode, agingstart, agingstop, hcithreshold, nbtithreshold, degf };
                switch (std.meta.stringToEnum(Key, args[i].name) orelse return r.unsupported(line, ".mosra key other than RelTotalTime, RelStartTime, RelStep, SimMode, RelMode, AgingStart, AgingStop, HciThreshold, NbtiThreshold, DegF")) {
                    .reltotaltime => m.rel_total_time = v,
                    .relstarttime => m.rel_start_time = v,
                    .relstep => m.rel_step = v,
                    .simmode => m.aged_runs = if (v == 2) true else if (v == 0) false else return r.unsupported(line, ".mosra SimMode other than 0 or 2"),
                    .relmode => m.rel_mode = if (v == 0 or v == 1 or v == 2) @intFromFloat(v) else return r.failed(line, error.ParseError),
                    .agingstart => m.aging_start = v,
                    .agingstop => m.aging_stop = v,
                    .hcithreshold => m.hci_threshold = v,
                    .nbtithreshold => m.nbti_threshold = v,
                    .degf => m.deg_f = v,
                }
            }
            if (i != args.len or !(m.rel_total_time > 0) or !(m.rel_step >= 0) or !(m.rel_start_time >= 0) or !(m.aging_stop > m.aging_start))
                return r.failed(line, error.ParseError);
            return m;
        }

        /// HSPICE `.sample FS= [TOL=] [NUMF=] [MAXFLD=] [BETA=]` [CR .SAMPLE].
        /// TOL and NUMF size HSPICE's adaptive fold count; MAXFLD bounds it
        /// here, so they are checked and unused.
        pub fn readSample(args: []const Value) Error!core.query.NoiseSample {
            var s: core.query.NoiseSample = .{ .fs = 0 };
            var i: usize = 0;
            while (i + 1 < args.len) : (i += 2) {
                if (args[i] != .name or args[i + 1] != .num) return error.ParseError;
                const v = args[i + 1].num;
                const Key = enum { fs, tol, numf, maxfld, beta };
                switch (std.meta.stringToEnum(Key, args[i].name) orelse return error.ParseError) {
                    .fs => s.fs = v,
                    .maxfld => s.max_fold = v,
                    .beta => s.beta = v,
                    .tol, .numf => {},
                }
            }
            if (i != args.len or !(s.fs > 0) or !(s.max_fold >= 1) or !(s.beta >= 0 and s.beta <= 1)) return error.ParseError;
            return s;
        }

        /// HSPICE `.save [TYPE=NODESET|IC] [FILE=] [LEVEL=ALL|TOP|SELECT|NONE]
        /// [TIME=]` [CR .SAVE]. SELECT saves every node, as ALL; NONE saves
        /// nothing. The last card wins.
        pub fn readSaveOp(r: *R, line: []const u8, f: *F) Error!void {
            var save: core.SaveOp = .{};
            var none_level = false;
            while (f.next()) |key| {
                if (!F.isWord(key) or !f.takeEq()) return r.failed(line, error.ParseError);
                const Key = enum { type, file, level, time };
                switch (std.meta.stringToEnum(Key, key) orelse return r.failed(line, error.ParseError)) {
                    .file => save.file = try r.pathOf(f),
                    .time => save.time = S.parseNum(f.next() orelse "") orelse return r.failed(line, error.ParseError),
                    .type => {
                        const v = f.next() orelse "";
                        if (!std.mem.eql(u8, v, "ic") and !std.mem.eql(u8, v, "nodeset")) return r.failed(line, error.ParseError);
                        save.ic = std.mem.eql(u8, v, "ic");
                    },
                    .level => {
                        const v = f.next() orelse "";
                        const levels = std.StaticStringMap(void).initComptime(.{ .{"all"}, .{"top"}, .{"select"}, .{"none"} });
                        if (!levels.has(v)) return r.failed(line, error.ParseError);
                        save.top_only = std.mem.eql(u8, v, "top");
                        none_level = std.mem.eql(u8, v, "none");
                    },
                }
            }
            if (!(save.time >= 0)) return r.failed(line, error.ParseError);
            r.save_op = if (none_level) null else save;
        }

        /// `line` as the deck spells it: the original case when `line` is a
        /// slice of the parsed text, else the joined continuation as read.
        pub fn written(r: *const R, line: []const u8) []const u8 {
            const at = @intFromPtr(line.ptr);
            const base = @intFromPtr(r.text.ptr);
            return if (at >= base and at + line.len <= base + r.text.len) r.orig[at - base ..][0..line.len] else line;
        }

        /// Logs `line` with `err` and returns `err`; `ParseError` for a
        /// malformed card, `UnsupportedCard` for a form ESPice cannot run.
        pub fn failed(r: *R, line: []const u8, err: Error) Error {
            r.logged = true;
            if (!@import("builtin").is_test) std.log.err("netlist: {s}: {s}", .{ @errorName(err), r.written(line) });
            return err;
        }

        // Sweeps and variants.

        /// The live-table row of global `.param` `name`, registering it (and
        /// folding its nominal value) on first use; null for a name no
        /// `.param` defines.
        pub fn liveParam(r: *R, name: []const u8) Error!?u32 {
            if (r.live_names.getIndex(name)) |k| return @intCast(k);
            const entry = r.globals.get(name) orelse return null;
            const top: Frame = .{ .scopes = &r.global_scopes };
            r.live_ok = true;
            defer r.live_ok = false;
            const v = switch (entry) {
                .num => |n| n,
                .text => |text| switch (try r.exprValue(text, &top, false)) {
                    .num => |n| n,
                    else => return error.ParseError,
                },
            };
            try r.live_names.put(r.arena, name, {});
            try r.live_nominal.append(r.arena, v);
            return @intCast(r.live_names.count() - 1);
        }

        /// Folds each global `.param` to its number in deck order, once every
        /// live name is known, so each link of a chain reads its
        /// predecessor's number and `subst` recurses one level per read, not
        /// one per link (#110). A value that reads a live name, draws a
        /// distribution or does not fold keeps its text, and an error stays
        /// for the card that reads it.
        pub fn foldGlobals(r: *R) Error!void {
            const top: Frame = .{ .scopes = &r.global_scopes };
            const was_in_card = r.in_card;
            r.in_card = true;
            defer r.in_card = was_in_card;
            for (r.global_keys.items) |key| {
                const entry = r.globals.getPtr(key).?;
                if (entry.* != .text) continue;
                const ops_mark = r.ops.items.len;
                const consts_mark = r.consts.items.len;
                const sites_mark = r.site_ops.items.len;
                const v: ?Value = r.exprValue(entry.text, &top, false) catch |err| switch (err) {
                    error.OutOfMemory => return error.OutOfMemory,
                    else => null, // else: the reading card reports it
                };
                r.ops.shrinkRetainingCapacity(ops_mark);
                r.consts.shrinkRetainingCapacity(consts_mark);
                r.site_ops.shrinkRetainingCapacity(sites_mark);
                r.site_keys.shrinkRetainingCapacity(sites_mark);
                if (v) |x| if (x == .num) {
                    entry.* = .{ .num = x.num };
                };
            }
        }

        /// A swept name: a global `.param`, `temp`, or a card.
        pub fn stepTarget(r: *R, name: []const u8) Error!StepTarget {
            if (std.mem.eql(u8, name, "temp")) return .temp;
            if (try r.liveParam(name)) |k| return .{ .param = k };
            return .{ .card = name };
        }

        /// `.step [lin|dec|oct] [param] name start stop incr`, `.step
        /// [param] name list v...`, with `temp` or a source for the name
        /// (ngspice/LTspice form). `dec`/`oct` take points per decade or
        /// octave.
        pub fn readStep(r: *R, args: []const Value) Error!Step {
            var i: usize = 0;
            var grid: Grid = .lin;
            if (i < args.len and args[i] == .name) if (grids.get(args[i].name)) |g| {
                grid = g;
                i += 1;
            };
            const param = i < args.len and args[i] == .name and std.mem.eql(u8, args[i].name, "param");
            i += @intFromBool(param);
            if (i >= args.len or args[i] != .name) return error.ParseError;
            const target: StepTarget = if (param)
                .{ .param = try r.liveParam(args[i].name) orelse return error.ParseError }
            else
                try r.stepTarget(args[i].name);
            i += 1;
            if (i < args.len and args[i] == .name and std.mem.eql(u8, args[i].name, "list")) {
                // An empty list would plan no runs at all (#66).
                if (i + 1 == args.len) return error.ParseError;
                return .{ .target = target, .values = try r.numbers(args[i + 1 ..]) };
            }
            if (grid == .poi) return error.ParseError;
            const n = try r.numbers(args[i..]);
            if (n.len != 3) return error.ParseError;
            return .{ .target = target, .values = try points(r.arena, grid, n[0], n[1], n[2], .per_unit) };
        }
    };
}
