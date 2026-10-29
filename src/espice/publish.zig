//! Terminal queries in, plots in the output out, in request order: the main
//! session's results, then each variant run's. Under a final plan a lone
//! leading sweep streams straight to the output, other sweeps stream to side
//! files their workers fill, and each result is freed once written and
//! measured. Every output failure lands in `Problem.delivery_error` and
//! stops publication.
//! Cites: none.
//!
//! Contents:
//! - `pub const DeliveryError = error{...}`
//! - `pub fn openStream(self: *Problem) !void`
//! - `pub fn sidesAllowed(self: *const Problem) bool`
//! - `pub fn openSide(ctx: *anyopaque, id: QueryId, job: Query) ?*std.Io.Writer`
//! - `pub fn deliverRuns(self: *Problem) void`
//! - `pub fn deliverFrom(self: *Problem, session: *const Session, next_output: *usize) void`
//! - `pub fn finishWhenDone(self: *Problem) void`
//! - `pub fn releaseDelivered(self: *Problem) void`

const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;
const analysis = @import("analysis");
const core = @import("core");
const output = @import("output");
const Problem = @import("Problem.zig");
const report = @import("report.zig");
const requests = core.query;
const Query = requests.Query;
const QueryId = requests.QueryId;
const Session = analysis.session.Session;

/// Why publication stopped (`Problem.output_error`): an encoder or file
/// error from the output session (`publish`, `endStream`, `finish`) or the
/// Touchstone file writer (`output.write`), `error.OutOfMemory` included.
/// The session lookups' `error.InvalidQuery` and `error.ResultUnavailable`
/// mean the query bookkeeping broke (a bug), reported instead of a panic.
pub const DeliveryError = errorOf(output.Session.publish) || errorOf(output.Session.endStream) ||
    errorOf(output.Session.endSide) || errorOf(output.Session.finish) || errorOf(output.write) ||
    errorOf(Session.info) || errorOf(Session.result) || errorOf(Session.query);

/// Under a final plan, a transient or AC sweep that publishes first and that
/// nothing reads afterwards (no `.meas` over it, no `.save` or `.stim`, no
/// optimization) writes its rows straight to a binary raw output, or drops
/// them without one, instead of keeping them all in memory. Its Result then
/// has empty `data`. Plotname and scale column come from
/// `analysis.schemaOf`; `endStream` checks the row count.
pub fn openStream(self: *Problem) !void {
    const deck = &self.prepared.deck;
    if (!sidesAllowed(self) or self.next_output != 0 or self.streaming != null) return;
    if (self.session.outputs.items.len == 0) return;
    const id = self.session.outputs.items[0];
    if ((try self.session.info(id)).status != .pending) return;
    const job = try self.session.query(id);
    for (deck.measures) |m| if (m.analysis == job) return;
    const schema = try analysis.schemaOf(self.gpa, &self.prepared.circuit, deck, job);
    if (schema.plotname.len == 0) return;
    switch (job) {
        inline else => |o| if (o.tol.temp_c != null or o.tol.variant != null) return,
    }
    const names = try self.gpa.alloc([]const u8, deck.probe_labels.len + 1);
    defer self.gpa.free(names);
    names[0] = schema.scale;
    @memcpy(names[1..], deck.probe_labels);
    const writer = try self.delivery.beginStream(self.io, .{ .title = deck.title, .result = .{
        .plotname = schema.plotname,
        .varnames = names,
        .is_complex = job == .ac,
        .npoints = 0,
        .data = &.{},
    } }) orelse return;
    if (!try self.session.streamTo(id, writer)) return self.delivery.abortStream();
    self.streaming = id;
    assert(self.next_output == 0);
}

/// What `openStream` asks of the whole run before any plot streams: a final
/// plan, nothing that reads results after the run, and a probe label per
/// column.
pub fn sidesAllowed(self: *const Problem) bool {
    const deck = &self.prepared.deck;
    return self.session.config.final_plan and self.save_path == null and deck.stims.len == 0 and
        self.prepared.tuner == null and deck.probe_labels.len == deck.probes.len;
}

/// `Session.stream_opener`: a transient or AC plot that no `.meas` card and
/// no tangent row reads streams to a side file of the output from its
/// worker, and `deliverOne` appends it in request order, so `-r` over a
/// sweep holds one row buffer per running point instead of every waveform.
/// Null (keep the rows) for anything else, or when the side cannot open.
pub fn openSide(ctx: *anyopaque, id: QueryId, job: Query) ?*std.Io.Writer {
    const self: *Problem = @ptrCast(@alignCast(ctx));
    switch (job) {
        .tran => |o| if (o.snapshot or o.tangents.len != 0) return null,
        .ac => {},
        else => return null, // else: only a transient or AC sweep streams its rows
    }
    if (self.streaming == id or self.tangent_steps.contains(id)) return null;
    for (self.prepared.deck.measures) |m| if (m.analysis == job) return null;
    self.sides.ensureUnusedCapacity(self.gpa, 1) catch return null;
    const side = (self.delivery.beginSide(self.io) catch return null) orelse return null;
    self.sides.putAssumeCapacity(id, side);
    return side.writer();
}

/// Publishes what has completed, the main session's results first, then
/// each variant run's once every earlier one has published all of its.
pub fn deliverRuns(self: *Problem) void {
    deliverFrom(self, &self.session, &self.next_output);
    if (self.next_output != self.session.outputs.items.len) return;
    for (self.runs) |*run| {
        deliverFrom(self, &run.session, &run.next_output);
        if (run.next_output != run.session.outputs.items.len) return;
    }
}

/// Publishes `session`'s completed results in request order from
/// `next_output.*`, stopping at the first unfinished query or the first
/// output failure.
pub fn deliverFrom(self: *Problem, session: *const Session, next_output: *usize) void {
    if (self.delivery_error != null) return;
    assert(next_output.* <= session.outputs.items.len);
    while (next_output.* < session.outputs.items.len) : (next_output.* += 1) {
        const delivered = deliverOne(self, session, session.outputs.items[next_output.*]) catch |err| {
            self.delivery_error = err;
            return;
        };
        if (!delivered) return;
    }
}

/// Finishes the output once every query has ended and published, so a
/// caller that steps to the end gets the file `run_all` writes. `run_all`
/// alone runs the `.step` variant runs, and publishes their plots in a
/// later `finish`.
pub fn finishWhenDone(self: *Problem) void {
    if (self.delivery_error != null or !self.session.finished()) return;
    if (self.next_output != self.session.outputs.items.len) return;
    self.delivery.finish(self.io) catch |err| {
        self.delivery_error = err;
    };
}

/// Under a final plan, frees each main-session query whose plot is written
/// and whose `.meas` block is printed, so a `-r` sweep holds the points
/// still running rather than the whole run. Nothing with `.save` or
/// `.stim`, which read the results after the run. Keeps the Monte Carlo
/// trials (`print_measures` reads them at the end), the tangent runs, and a
/// requested OP that later queries still read. Running out of memory only
/// keeps the rest for later.
pub fn releaseDelivered(self: *Problem) void {
    if (!self.session.config.final_plan or self.save_path != null or self.prepared.deck.stims.len != 0) return;
    const outputs = self.session.outputs.items;
    const cursor = if (self.live_measures) |live| live.cursor else report.Cursor{};
    const printed = if (self.prepared.deck.measures.len == 0 or cursor.session > 0) outputs.len else cursor.next;
    const upto = @min(self.next_output, printed);
    while (self.released_upto < upto) : (self.released_upto += 1) {
        const id = outputs[self.released_upto];
        if (self.tangent_steps.contains(id)) continue;
        const s = self.summary(id) catch continue;
        if (std.mem.indexOf(u8, s.plotname, "(monte=") != null) continue;
        const kept: Problem.Summary = .{ .plotname = self.arena.allocator().dupe(u8, s.plotname) catch return, .npoints = s.npoints, .nvars = s.nvars };
        self.released.ensureUnusedCapacity(self.gpa, 1) catch return;
        self.session.release(id) catch continue;
        self.released.putAssumeCapacity(id, kept);
    }
    assert(self.released_upto >= upto);
}

// ---- private ----

fn errorOf(comptime f: anytype) type {
    return @typeInfo(@typeInfo(@TypeOf(f)).@"fn".return_type.?).error_union.error_set;
}

/// Publishes (or ends the stream of) `id` once it is terminal; false while
/// it still runs. `session.outputs` holds only appended ids, and a row reads
/// `.complete` only after its executor published, so the session lookups
/// fail only if Session breaks that bookkeeping; the error then stops
/// delivery like any output failure instead of aborting a C host.
fn deliverOne(self: *Problem, session: *const Session, id: QueryId) DeliveryError!bool {
    const status = (try session.info(id)).status;
    if (!status.terminal()) return false;
    if (session.released(id)) return true;
    const main = session == &self.session;
    if (main and self.streaming == id) {
        self.streaming = null;
        if (status != .complete) {
            self.delivery.abortStream();
            return true;
        }
        const res = try session.result(id);
        try self.delivery.endStream(self.io, res.npoints, res.varnames.len);
    } else if (main and self.sides.contains(id)) {
        const side = self.sides.fetchRemove(id).?.value;
        if (status != .complete) {
            self.delivery.dropSide(side);
            return true;
        }
        const res = try session.result(id);
        var lap = Problem.lapStart(self.io, self.session.config.timing_in_depth);
        defer Problem.timingLap(self.io, &lap, "output delivery");
        try self.delivery.endSide(self.io, side, .{ .title = session.deck.title, .result = res });
    } else if (status == .complete) {
        const res = try session.result(id);
        var lap = Problem.lapStart(self.io, self.session.config.timing_in_depth);
        defer Problem.timingLap(self.io, &lap, "output delivery");
        const job = try session.query(id);
        // An `.sp` carries its ports' z0, which a Touchstone output records (#60).
        var z0: std.ArrayList(f64) = .empty;
        defer z0.deinit(self.gpa);
        if (job == .sp) try modeZ0(self.gpa, job.sp.ports, &z0);
        const plot: output.Plot = .{ .title = session.deck.title, .result = res, .z0 = z0.items };
        try self.delivery.publish(self.io, plot);
        if (job == .sp) if (job.sp.lin) |lin| if (lin.touchstone) try writeTouchstone(self, lin.file, plot);
    }
    if (main) assert(self.streaming != id);
    return true;
}

/// Appends the reference impedance of each of `ports`' modes to `z0`, in
/// sp.zig's `Basis` order: each port's first mode (a balanced one's
/// differential, 2·z0), then the common modes (z0/2).
fn modeZ0(gpa: Allocator, ports: []const requests.Port, z0: *std.ArrayList(f64)) Allocator.Error!void {
    const before = z0.items.len;
    for (ports) |p| try z0.append(gpa, if (p.balanced != null) 2 * p.z0 else p.z0);
    for (ports) |p| if (p.balanced != null) try z0.append(gpa, p.z0 / 2);
    assert(z0.items.len - before >= ports.len);
}

/// Writes a `.lin format=touchstone` result as `<file>.s<N>p` beside the
/// deck, `file` defaulting to the deck's name without its extension (HSPICE
/// names it after the netlist), against the ports' own z0.
fn writeTouchstone(self: *Problem, file: []const u8, plot: output.Plot) !void {
    const stem = if (file.len > 0) file else std.fs.path.stem(self.origin);
    const name = try std.fmt.allocPrint(self.gpa, "{s}.s{d}p", .{ stem, @max(plot.z0.len, 1) });
    defer self.gpa.free(name);
    const path = try std.fs.path.join(self.gpa, &.{ std.fs.path.dirname(self.origin) orelse ".", name });
    defer self.gpa.free(path);
    try output.write(self.io, path, .touchstone, plot);
}
