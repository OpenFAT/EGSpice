//! `.meas` evaluation, ported from ngspice 45 com_measure2.c: a deck's
//! cards and a finished result in, the same event counting, linear
//! interpolation and quadrature, printed in the same lines, out; `Stream`
//! measures the same cards row by row while a transient runs, and
//! `passes` (scan.zig) runs them on the GPU lanes. Aggregation only: the
//! files below never import this one.
//! Cites: ngspice 45 com_measure2.c, measure.c; HSPICE [CR .MEASURE].
//!
//! Contents:
//! - `pub const passes`, `pub const Stream`, `pub const Point`
//! - `pub const print`, `pub const printFailed`, `pub const printStatistics`
//! - `pub const evaluateAll`, `pub const evaluateOutcomes`, `pub const sample`, `pub const markColumns`

const std = @import("std");
const assert = std.debug.assert;
const core = @import("core");
/// The one-sample-at-a-time passes, shared with the GPU lanes.
pub const passes = @import("scan.zig");
pub const Stream = @import("Stream.zig");
pub const markColumns = @import("rule.zig").markColumns;
const listing = @import("listing.zig");
const card = @import("card.zig");

pub const Point = listing.Point;
pub const print = listing.print;
pub const printFailed = listing.printFailed;
pub const printStatistics = listing.printStatistics;
pub const evaluateAll = card.evaluateAll;
pub const evaluateOutcomes = card.evaluateOutcomes;
pub const sample = card.sample;

comptime {
    // scan.zig imports only std, so it restates the core sentinels.
    assert(passes.unset == core.measure_unset);
    assert(passes.last == core.measure_last);
}

test {
    _ = @import("sci.zig");
    _ = @import("column.zig");
    _ = @import("Wave.zig");
    _ = @import("checks.zig");
    _ = card;
    _ = listing;
    _ = @import("rule.zig");
    _ = @import("Scan.zig");
    _ = Stream;
    _ = @import("device_scans.zig");
    _ = passes;
}
