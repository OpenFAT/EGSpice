//! The `core` module: the data every layer speaks (numerics, query requests,
//! the deck, results, the name pool, the shared ids) and the small solvers
//! more than one layer drives (the `.model OPT` searches, dense
//! eigenvalues). Imports nothing but std. Re-exports only; the files below
//! never import this root, so `ids.zig` can stand alone as the device
//! objects' `core`.
//! Cites: none here; each file names its own.
//!
//! Contents:
//! - `pub const gpa: std.mem.Allocator`
//! - namespaces: `numerics`, `query`, `validate`, `lm`, `bisect`, `eigen`, `testing`
//! - deck (`deck.zig`): `Deck`, `Ic`, `AcOverride`, `Variants`, `QueryBindings`,
//!   `SaveOp`, `Mosra`, `MosraModel`, `Stim`
//! - measures (`measure_card.zig`): `Measure`, `MeasureClause`, `GoalBound`,
//!   `MeasureFunc`, `Check`, `MeasureOp`, `MeasureRef`, `measure_unset`,
//!   `measure_last`, `measure_no_at`
//! - results (`result.zig`): `Result`, `Measured`, `Schema`, `QuerySchema`
//! - names (`intern.zig`): `InternPool`, `Name`
//! - ids (`ids.zig`): `DeviceType`, `GROUND`

const std = @import("std");
const builtin = @import("builtin");
const deck = @import("deck.zig");
const measure_card = @import("measure_card.zig");
const result = @import("result.zig");
const intern = @import("intern.zig");
const ids = @import("ids.zig");

/// Vector kernels, complex numbers, the frequency grid and the solver
/// tolerances.
pub const numerics = @import("numerics.zig");
/// One options struct per analysis, and the `Query` union over them.
pub const query = @import("query.zig");
/// Query validation (`validate.validate`), shared by the analysis boundary
/// and the frontend.
pub const validate = @import("validate.zig");
/// The HSPICE `.model OPT` Levenberg-Marquardt optimizer, driven by its caller.
pub const lm = @import("lm.zig");
/// The HSPICE `.model OPT` bisection and pass/fail searches.
pub const bisect = @import("bisect.zig");
/// Dense real eigenvalues (balance, Hessenberg, Francis QR).
pub const eigen = @import("eigen.zig");
/// Test support: the allocator every allocation-failure sweep runs over.
pub const testing = @import("testing.zig");

/// The process-wide general-purpose allocator. SmpAllocator does not build
/// single-threaded (the wasm playground), so there it is the page allocator,
/// which on wasm is WasmAllocator.
pub const gpa: std.mem.Allocator = if (builtin.single_threaded) std.heap.page_allocator else std.heap.smp_allocator;

pub const Deck = deck.Deck;
pub const Ic = deck.Ic;
pub const AcOverride = deck.AcOverride;
pub const Variants = deck.Variants;
pub const QueryBindings = deck.QueryBindings;
pub const SaveOp = deck.SaveOp;
pub const Mosra = deck.Mosra;
pub const MosraModel = deck.MosraModel;
pub const Stim = deck.Stim;

pub const Measure = measure_card.Measure;
pub const MeasureClause = measure_card.MeasureClause;
pub const GoalBound = measure_card.GoalBound;
pub const MeasureFunc = measure_card.MeasureFunc;
pub const Check = measure_card.Check;
pub const MeasureOp = measure_card.MeasureOp;
pub const MeasureRef = measure_card.MeasureRef;
pub const measure_unset = measure_card.measure_unset;
pub const measure_last = measure_card.measure_last;
pub const measure_no_at = measure_card.measure_no_at;

pub const Result = result.Result;
pub const Measured = result.Measured;
pub const Schema = result.Schema;
pub const QuerySchema = result.QuerySchema;

pub const InternPool = intern.InternPool;
pub const Name = intern.Name;

pub const DeviceType = ids.DeviceType;
pub const GROUND = ids.GROUND;

test {
    _ = numerics;
    _ = @import("freq_sweep.zig");
    _ = intern;
    _ = lm;
    _ = bisect;
    _ = eigen;
    _ = query;
    _ = @import("query/card_names.zig");
    _ = @import("query/periodic.zig");
    _ = @import("query/transient.zig");
    _ = deck;
    _ = measure_card;
    _ = validate;
}
