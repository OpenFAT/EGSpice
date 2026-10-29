//! The C ABI's data at the boundary: include/espice.h's records and status
//! codes in Zig, each pinned to the header at compile time, and the
//! conversions between them and the facade's types (C pointer and length
//! pairs to slices, scopes, progress, advance events, Zig errors to
//! `espice_status`). A reordered tag, a moved field or a resized record
//! fails the build here.
//! Cites: include/espice.h (the published contract, ABI 1).
//!
//! Contents:
//! - `pub const abi_version: u32`
//! - `pub const no_query: u32`
//! - `pub const Code = enum(u32)`
//! - `pub const Bytes = extern struct { data: ?[*]const u8, len: usize }`
//! - `pub const DisplayFn = fn (text: [*]const u8, len: usize, user: ?*anyopaque) callconv(.c) void`
//! - `pub const CreateOptions = extern struct`
//! - `pub const PointsOptions = extern struct`
//! - `pub const Scope = extern struct { kind: u32, id: u32 }`
//! - `pub const Progress = extern struct`
//! - `pub const QueryInfo = extern struct`
//! - `pub const Advance = extern struct`
//! - `pub const ResultInfo = extern struct`
//! - `pub const PrintOptions = extern struct`
//! - `pub const InputError = error{ InvalidArgument, InvalidConcurrency }`
//! - `pub fn status(err: anyerror) u32`
//! - `pub fn constSlice(comptime T: type, ptr: ?[*]const T, len: usize) error{InvalidArgument}![]const T`
//! - `pub fn mutableSlice(comptime T: type, ptr: ?[*]T, len: usize) error{InvalidArgument}![]T`
//! - `pub fn strings(gpa: Allocator, bytes: []const Bytes) error{ OutOfMemory, InvalidArgument }![][]const u8`
//! - `pub fn concurrency(n: u32) error{InvalidConcurrency}!u16`
//! - `pub fn decodeScope(scope: Scope) error{InvalidArgument}!api.Scope`
//! - `pub fn queryInfo(q: api.QueryInfo) QueryInfo`
//! - `pub fn advance(event: api.Advance) Advance`
//! - `pub fn writeTruncated(buffer: []u8, text: []const u8) void`

const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;
const api = @import("espice");
const header = @import("espice_h");

/// Bumped on a change that breaks a caller built against an older header;
/// `espice_create` rejects a mismatch. Fields appended to
/// `espice_create_options` do not bump it: `struct_size` gates them.
pub const abi_version: u32 = 1;

/// ESPICE_NO_QUERY, "no dependency" in `QueryInfo`, and ESPICE_PLOT_TITLE,
/// "plot title" in `espice_copy_result_name`.
pub const no_query: u32 = std.math.maxInt(u32);

/// `espice_status` values; `status` maps Zig errors onto them.
pub const Code = enum(u32) {
    ok = 0,
    invalid_argument = 1,
    buffer_too_small = 2,
    out_of_memory = 3,
    invalid_query = 4,
    result_unavailable = 5,
    failed = 6,
    abi_mismatch = 7,
    invalid_handle = 8,
};

/// `espice_bytes`: a C (pointer, length) pair. NULL only at length 0.
pub const Bytes = extern struct { data: ?[*]const u8, len: usize };

/// `espice_display_fn`: receives `.hdl` display-task text, not NUL-terminated.
pub const DisplayFn = fn (text: [*]const u8, len: usize, user: ?*anyopaque) callconv(.c) void;

/// `espice_create_options`.
pub const CreateOptions = extern struct {
    abi_version: u32,
    struct_size: u32,
    source_kind: u32,
    dialect: u32,
    backend: u32,
    explicit_gpu: u32,
    output_format: u32,
    max_parallel: u32,
    source: Bytes,
    origin: Bytes,
    output_path: Bytes,
    /// Added after ABI 1 shipped: read only when `struct_size` covers them.
    zig_path: Bytes,
    share_path: Bytes,
    /// Added after the toolchain fields, read under the same rule.
    live_params: ?[*]const Bytes,
    live_param_count: usize,
    /// Added after `live_params`, read under the same rule.
    display: ?*const DisplayFn,
    display_user: ?*anyopaque,
};

/// `espice_points_options`: MATCHA's layout (MATCHA/SEAM.md row 1) through
/// `warm_start`, 40 bytes on LP64; `tangent_step` is ESPice's, read when
/// `struct_size` covers it.
pub const PointsOptions = extern struct {
    struct_size: usize,
    abort_measure: Bytes,
    abort_after: f64,
    measures_only: u8,
    early_stop: u8,
    warm_start: u8,
    tangent_step: f64,
};

/// `espice_scope`: kind 0 all, 1 one query's chain, 2 one component.
pub const Scope = extern struct { kind: u32, id: u32 };

/// `espice_progress`.
pub const Progress = extern struct { phase: u32, reserved: u32 = 0, completed: u64, total: u64 };

/// `espice_query_info`.
pub const QueryInfo = extern struct {
    id: u32,
    kind: u32,
    status: u32,
    dependency: u32,
    component: u32,
    requested: u32,
    has_progress: u32,
    failure_code: u32,
    progress: Progress,
};

/// `espice_advance_event`.
pub const Advance = extern struct {
    requested: u32,
    advanced: u32,
    status: u32,
    target_status: u32,
    has_progress: u32,
    failure_code: u32,
    output_error: u32,
    reserved: u32 = 0,
    progress: Progress,
    output_error_name: [96]u8,
};

/// `espice_result_info`.
pub const ResultInfo = extern struct { variable_count: u32, is_complex: u32, point_count: u64, value_count: u64 };

/// `espice_print_options`.
pub const PrintOptions = extern struct {
    scope: Scope,
    preview: u32,
    query: u32,
    ascii: u32,
    max_parallel: u32,
    ready_ids: ?[*]const u32,
    ready_count: usize,
};

/// The ways a value from the C side is malformed; both map to
/// ESPICE_INVALID_ARGUMENT.
pub const InputError = error{ InvalidArgument, InvalidConcurrency };

/// `QueryInfo.progress`'s phase enum.
const Phase = @FieldType(ProgressEvent, "phase");
const ProgressEvent = @typeInfo(@FieldType(api.QueryInfo, "progress")).optional.child;

comptime {
    if (header.ESPICE_NO_QUERY != no_query) @compileError("include/espice.h ESPICE_NO_QUERY drifted");
    if (header.ESPICE_PLOT_TITLE != no_query) @compileError("include/espice.h ESPICE_PLOT_TITLE drifted");
}

// Enums cross the boundary as tag values, so each is pinned to the header's
// ESPICE_<NAME> constants: a reordered or renamed tag fails the build.
comptime {
    pinTags(Code, .{});
    pinTags(api.Dialect, .{});
    pinTags(api.Request, .{});
    pinTags(api.Format, .{});
    pinTags(api.Status, .{ .failed = "QUERY_FAILED" });
    pinTags(api.requests.Kind, .{});
    pinTags(api.Problem.Reason, .{
        .ok = "REASON_OK",
        .no_convergence = "REASON_NO_CONVERGENCE",
        .timestep_too_small = "REASON_TIMESTEP_TOO_SMALL",
        .singular_matrix = "REASON_SINGULAR_MATRIX",
        .measure_not_found = "REASON_MEASURE_NOT_FOUND",
        .measure_unresolved = "REASON_MEASURE_UNRESOLVED",
        .aborted = "REASON_ABORTED",
        .model_bin_not_found = "REASON_MODEL_BIN_NOT_FOUND",
        .topology_changed = "REASON_TOPOLOGY_CHANGED",
        .invalid_parameter = "REASON_INVALID_PARAMETER",
        .device_refused = "REASON_DEVICE_REFUSED",
        .other = "REASON_OTHER",
    });
    pinTags(Phase, .{ .dc = "DC_PHASE" });
}

// The records are AoS on purpose: each crosses the boundary whole, once per
// call. Each is pinned to its header struct, field by field.
comptime {
    pinLayout(Bytes, header.espice_bytes);
    pinLayout(CreateOptions, header.espice_create_options);
    pinLayout(PointsOptions, header.espice_points_options);
    pinLayout(Scope, header.espice_scope);
    pinLayout(Progress, header.espice_progress);
    pinLayout(QueryInfo, header.espice_query_info);
    pinLayout(Advance, header.espice_advance_event);
    pinLayout(ResultInfo, header.espice_result_info);
    pinLayout(PrintOptions, header.espice_print_options);
    // MATCHA's 40 bytes end at `warm_start`, tail padding included.
    if (@sizeOf(usize) == 8) {
        assert(@sizeOf(PointsOptions) == 48);
        for (.{ "struct_size", "abort_measure", "abort_after", "measures_only", "early_stop", "warm_start", "tangent_step" }, .{ 0, 8, 24, 32, 33, 34, 40 }) |field, at|
            assert(@offsetOf(PointsOptions, field) == at);
    }
}

/// Collapses a Zig error into an `espice_status`; the caller keeps the name
/// for `espice_error_message`.
pub fn status(err: anyerror) u32 {
    const code: Code = switch (err) {
        error.InvalidArgument, error.InvalidConcurrency, error.InvalidComponent, error.QueryNotReady, error.DuplicateQuery => .invalid_argument,
        // Malformed caller input (analysis cards, query options, directive edits), not a numerical failure.
        error.InvalidAnalysisArguments, error.InvalidQueryOptions, error.UnsupportedDirectiveMutation => .invalid_argument,
        // A live parameter name the deck does not define or declare.
        error.UnknownParameter => .invalid_argument,
        // Points API: no live parameters, a deck that sweeps already, a
        // measure name no card has.
        error.NoLiveParameters, error.SweptTemplate, error.UnknownMeasure => .invalid_argument,
        // Points API: tangents on a transient that cannot integrate them.
        error.TangentsUnavailable => .invalid_argument,
        error.BufferTooSmall => .buffer_too_small,
        error.OutOfMemory => .out_of_memory,
        error.InvalidQuery => .invalid_query,
        error.ResultUnavailable => .result_unavailable,
        error.AbiMismatch => .abi_mismatch,
        error.InvalidHandle => .invalid_handle,
        // A full handle table is a capacity, reported like any allocation.
        error.TooManyProblems => .out_of_memory,
        // `err` is anyerror (a handle's last error keeps any name), so no
        // exhaustive switch exists: every error not named above is a
        // numerical, I/O or output failure.
        else => .failed, // else: FAILED in include/espice.h
    };
    assert(code != .ok);
    return @backingInt(code);
}

/// A C (pointer, length) pair as a slice. NULL is allowed only at length 0.
pub fn constSlice(comptime T: type, ptr: ?[*]const T, len: usize) error{InvalidArgument}![]const T {
    if (len == 0) return &.{};
    return (ptr orelse return error.InvalidArgument)[0..len];
}

/// `constSlice` for a buffer the caller lets the library write.
pub fn mutableSlice(comptime T: type, ptr: ?[*]T, len: usize) error{InvalidArgument}![]T {
    if (len == 0) return &.{};
    return (ptr orelse return error.InvalidArgument)[0..len];
}

/// Each C string of `bytes` as a slice, in a table the caller owns and
/// frees with `gpa`; the strings stay borrowed. Fails as `constSlice` does
/// on a NULL string of nonzero length, freeing the table.
pub fn strings(gpa: Allocator, bytes: []const Bytes) error{ OutOfMemory, InvalidArgument }![][]const u8 {
    const out = try gpa.alloc([]const u8, bytes.len);
    errdefer gpa.free(out);
    for (bytes, out) |b, *s| s.* = try constSlice(u8, b.data, b.len);
    assert(out.len == bytes.len);
    return out;
}

/// A C `max_parallel` as the facade's u16, rejecting 0 and overflow.
pub fn concurrency(n: u32) error{InvalidConcurrency}!u16 {
    if (n == 0) return error.InvalidConcurrency;
    return std.math.cast(u16, n) orelse error.InvalidConcurrency;
}

/// An `espice_scope` as the facade's, rejecting an unknown kind.
pub fn decodeScope(scope: Scope) error{InvalidArgument}!api.Scope {
    return switch (scope.kind) {
        0 => .all,
        1 => .{ .query = @fromBackingInt(scope.id) },
        2 => .{ .component = scope.id },
        else => error.InvalidArgument,
    };
}

/// One query's snapshot as its header record.
pub fn queryInfo(q: api.QueryInfo) QueryInfo {
    const info: QueryInfo = .{
        .id = @backingInt(q.id),
        .kind = @backingInt(q.kind),
        .status = @backingInt(q.status),
        .dependency = if (q.dependency) |dep| @backingInt(dep) else no_query,
        .component = q.component,
        .requested = @intFromBool(q.requested),
        .has_progress = @intFromBool(q.progress != null),
        .failure_code = if (q.failure) |err| status(err) else 0,
        .progress = progress(q.progress),
    };
    assert(info.dependency != info.id);
    return info;
}

/// One advance event as its header record, its output error's name
/// truncated into `output_error_name`.
pub fn advance(event: api.Advance) Advance {
    var out: Advance = .{
        .requested = @backingInt(event.requested),
        .advanced = @backingInt(event.advanced),
        .status = @backingInt(event.status),
        .target_status = @backingInt(event.target_status),
        .has_progress = @intFromBool(event.progress != null),
        .failure_code = if (event.failure) |err| status(err) else 0,
        .output_error = if (event.delivery_error) |err| status(err) else 0,
        .progress = progress(event.progress),
        .output_error_name = @splat(0),
    };
    if (event.delivery_error) |err| writeTruncated(&out.output_error_name, @errorName(err));
    assert(out.output_error_name[out.output_error_name.len - 1] == 0);
    return out;
}

/// Copies as much of `text` as fits, always NUL-terminated. No-op on an
/// empty buffer.
pub fn writeTruncated(buffer: []u8, text: []const u8) void {
    if (buffer.len == 0) return;
    const n = @min(buffer.len - 1, text.len);
    @memcpy(buffer[0..n], text[0..n]);
    buffer[n] = 0;
    assert(n <= text.len);
}

// ---- private ----

/// Checks every tag of `E` against `ESPICE_<TAG>` in the header. `rename`
/// maps a tag to its header name where the upper-cased tag would clash.
fn pinTags(comptime E: type, comptime rename: anytype) void {
    @setEvalBranchQuota(20_000);
    const info = @typeInfo(E).@"enum";
    for (info.field_names, info.field_values) |field, value| {
        var upper: [field.len]u8 = undefined;
        const name = if (@hasField(@TypeOf(rename), field)) @field(rename, field) else std.ascii.upperString(&upper, field);
        if (@field(header, "ESPICE_" ++ name) != value) @compileError("include/espice.h ESPICE_" ++ name ++ " != " ++ @typeName(E) ++ "." ++ field);
    }
}

/// Fails the build unless `C` (the header's struct) has `Z`'s size and
/// every field of `Z` at the same offset, under the same name.
fn pinLayout(comptime Z: type, comptime C: type) void {
    if (@sizeOf(Z) != @sizeOf(C)) @compileError(@typeName(C) ++ " drifted in size from " ++ @typeName(Z));
    for (@typeInfo(Z).@"struct".field_names) |name|
        if (@offsetOf(Z, name) != @offsetOf(C, name)) @compileError(@typeName(C) ++ "." ++ name ++ " drifted from " ++ @typeName(Z));
}

fn progress(p: ?ProgressEvent) Progress {
    const event = p orelse return .{ .phase = 0, .completed = 0, .total = 0 };
    return .{ .phase = @backingInt(event.phase), .completed = event.completed, .total = event.total };
}
