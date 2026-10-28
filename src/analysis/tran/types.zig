//! Transient tables the driver, the integrator, MATEX and the
//! post-processors share, kept apart from the driver so they import no
//! solver: the recorded waveform (point-major f64 rows in, borrowed columns
//! out) and the `.tran` query types.
//! Cites: none (ESPice's own layouts; #52 and #123 fixed two of their edge
//! cases).
//!
//! Contents:
//! - `pub const Method = core.query.Method`
//! - `pub const Options = core.query.Tran`
//! - `pub fn initialCapacity(options: Options) u32`
//! - `pub const Column = struct`, `Column.of(s: []const f64) Column`,
//!   `Column.at(c: Column, i: usize) f64`, `Column.from(c: Column, i: usize) Column`,
//!   `Column.lowerBound(c: Column, v: f64) usize`
//! - `pub const Waveform = struct`, `Waveform.init(allocator, n_probes: u32, capacity: u32) Allocator.Error!Waveform`,
//!   `Waveform.initStream(allocator, n_probes: u32, sink: *std.Io.Writer) Allocator.Error!Waveform`,
//!   `Waveform.stride(self) usize`, `Waveform.record(self: *Waveform, t: f64, x: []const f64, probes: []const u32) RecordError!void`,
//!   `Waveform.data(self) []f64`, `Waveform.time(self, i: usize) f64`, `Waveform.column(self, c: usize) Column`,
//!   `Waveform.deinit(self: *Waveform) void`

const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;

/// The integration method a `.tran` card selects (core/query.zig).
pub const Method = @import("core").query.Method;

/// `.tran` query options, defined in core/query.zig.
pub const Options = @import("core").query.Tran;

/// A borrowed series inside a row-major table: element i is
/// `base[i * stride]`. Readers walk a waveform column in place instead of
/// gathering it.
pub const Column = struct {
    base: []const f64,
    stride: usize,
    len: usize,

    /// Returns a plain slice as a column of stride 1.
    pub fn of(s: []const f64) Column {
        return .{ .base = s, .stride = 1, .len = s.len };
    }

    /// Returns element `i`. Asserts (via the slice bounds check) that `i < len`.
    pub fn at(c: Column, i: usize) f64 {
        return c.base[i * c.stride];
    }

    /// Returns the column from element `i` on; `i == len` gives an empty
    /// column. Asserts that `i <= len`.
    pub fn from(c: Column, i: usize) Column {
        assert(i <= c.len);
        // A strided base ends at the last element, (len-1)*stride+1 values,
        // so the start of an empty tail is clamped to the base (#123).
        const tail: Column = .{ .base = c.base[@min(i * c.stride, c.base.len)..], .stride = c.stride, .len = c.len - i };
        assert(tail.len <= c.len);
        return tail;
    }

    /// Returns the first index whose value is >= `v`, or `len` when none is;
    /// the column must be ascending. O(log len).
    pub fn lowerBound(c: Column, v: f64) usize {
        var lo: usize = 0;
        var hi = c.len;
        while (lo < hi) {
            const mid = lo + (hi - lo) / 2;
            // !(x >= v), not x < v: no value is >= NaN, so NaN gives len (#52).
            if (!(c.at(mid) >= v)) lo = mid + 1 else hi = mid;
        }
        assert(lo <= c.len);
        return lo;
    }
};

/// The recorded transient waveform as point-major rows (time, probe 0, ...),
/// the exact layout a transient `Result` publishes, so the result borrows
/// the recording instead of transposing a copy of it. Untouched capacity is
/// virtual memory only: the estimate in `initialCapacity` costs no RSS.
/// Invariant: `len <= capacity`, and `rows.len == capacity * stride()`
/// unless `sink` is set, when `rows` holds the one row being built.
pub const Waveform = struct {
    /// `capacity` rows of `stride()` f64s; the first `len` rows are valid.
    rows: []f64,
    len: u32,
    capacity: u32,
    n_probes: u32,
    /// Frees `rows` in `deinit`, and grows them in `record`.
    allocator: Allocator,
    /// When set, every row goes here as native-endian f64s and only the row
    /// being built is kept: `data()` is empty and `column`/`time` are
    /// invalid, while `len` still counts the rows written.
    sink: ?*std.Io.Writer = null,

    /// Every way `record` fails. On any error the waveform is as it was
    /// except that a streamed row may be partly written to the sink.
    pub const RecordError = Allocator.Error || std.Io.Writer.Error;

    /// Returns room for `capacity` points (at least 1) of `n_probes` probes.
    /// Caller owns the rows; free with `deinit`.
    pub fn init(allocator: Allocator, n_probes: u32, capacity: u32) Allocator.Error!Waveform {
        const cap: u32 = @max(capacity, 1);
        const wf: Waveform = .{
            .rows = try allocator.alloc(f64, (@as(usize, n_probes) + 1) * cap),
            .len = 0,
            .capacity = cap,
            .n_probes = n_probes,
            .allocator = allocator,
        };
        assert(wf.rows.len == wf.stride() * cap);
        return wf;
    }

    /// Returns a waveform that writes each row to `sink` instead of keeping
    /// it. `sink` must outlive the waveform.
    pub fn initStream(allocator: Allocator, n_probes: u32, sink: *std.Io.Writer) Allocator.Error!Waveform {
        var wf = try init(allocator, n_probes, 1);
        wf.sink = sink;
        assert(wf.capacity == 1);
        return wf;
    }

    /// Returns the columns per row: time, then one per probe.
    pub fn stride(self: Waveform) usize {
        return @as(usize, self.n_probes) + 1;
    }

    /// Appends point `t` with `x[probes[k]]` as probe k. Asserts that
    /// `probes` has `n_probes` entries. Invalidates earlier `data` and
    /// `column` slices if additional memory is needed; on
    /// error.OutOfMemory nothing is appended.
    pub fn record(self: *Waveform, t: f64, x: []const f64, probes: []const u32) RecordError!void {
        assert(probes.len == self.n_probes);
        const len_before = self.len;
        const row = try self.next();
        row[0] = t;
        for (probes, row[1..]) |node, *v| v.* = x[node];
        if (self.sink) |w| try w.writeAll(std.mem.sliceAsBytes(row));
        assert(self.len == len_before + 1);
    }

    /// Returns the recorded rows, borrowed: valid until the next `record`
    /// or `deinit`. A transient `Result.data` is exactly this slice.
    pub fn data(self: Waveform) []f64 {
        if (self.sink != null) return &.{};
        assert(self.len <= self.capacity);
        return self.rows[0 .. @as(usize, self.len) * self.stride()];
    }

    /// Returns point `i`'s time. Asserts a kept (not streamed) waveform.
    pub fn time(self: Waveform, i: usize) f64 {
        assert(self.sink == null);
        return self.rows[i * self.stride()];
    }

    /// Returns column `c` (0 = time, k + 1 = probe k), borrowed in place;
    /// valid until the next `record` or `deinit`. Asserts that `c` is a
    /// column of a kept waveform.
    pub fn column(self: Waveform, c: usize) Column {
        assert(c < self.stride());
        assert(self.sink == null);
        return .{ .base = self.rows[c..], .stride = self.stride(), .len = self.len };
    }

    /// Frees the rows; every borrowed `data` or `column` slice dies with them.
    pub fn deinit(self: *Waveform) void {
        self.allocator.free(self.rows);
        self.* = undefined;
    }

    // ---- private ----

    /// The row the next record fills; the count moves only once the row
    /// exists, so a failed grow appends nothing.
    fn next(self: *Waveform) Allocator.Error![]f64 {
        const s = self.stride();
        if (self.sink != null) {
            self.len += 1;
            return self.rows[0..s];
        }
        if (self.len == self.capacity) try self.grow();
        assert(self.len < self.capacity);
        defer self.len += 1;
        return self.rows[@as(usize, self.len) * s ..][0..s];
    }

    // ponytail: doubling fallback; initialCapacity covers normal runs. On an
    // arena a growth that cannot resize in place leaves the old rows resident
    // until the arena dies; a reserve-and-commit buffer removes that.
    fn grow(self: *Waveform) Allocator.Error!void {
        const new_cap = @as(usize, self.capacity) * 2;
        self.rows = try self.allocator.realloc(self.rows, self.stride() * new_cap);
        self.capacity = @intCast(new_cap);
        assert(self.rows.len == self.stride() * self.capacity);
    }
};

/// Most rows `initialCapacity` reserves up front; `Waveform.record` doubles
/// past it.
const capacity_max: f64 = 1 << 22;
/// Fewest rows `initialCapacity` reserves.
const capacity_min: f64 = 64;

comptime {
    assert(capacity_min < capacity_max);
    assert(capacity_max <= std.math.maxInt(u32));
}

/// Returns the initial waveform capacity in points: twice the printed window
/// over the print step, clamped to [64, 2^22]. Adaptive dt makes the real
/// count unknown; 2x covers the LTE-refined tail on the fixture corpus.
/// Points before t_start are never stored.
pub fn initialCapacity(options: Options) u32 {
    const est = 2.0 * (options.t_stop - options.t_start) / options.dt_init;
    const capacity: u32 = @intFromFloat(@min(@max(capacity_min, est), capacity_max));
    assert(capacity >= capacity_min);
    return capacity;
}

test "initialCapacity clamps to its bounds" {
    const lo: Options = .{ .t_stop = 1e-9, .dt_init = 1e-6 };
    try std.testing.expectEqual(@as(u32, 64), initialCapacity(lo));
    const hi: Options = .{ .t_stop = 1, .dt_init = 1e-12 };
    try std.testing.expectEqual(@as(u32, 1 << 22), initialCapacity(hi));
}
