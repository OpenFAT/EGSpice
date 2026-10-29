//! The C ABI over `Problem`, built as libespice.a: C calls in, status codes
//! and caller-owned buffers out. Every entry point validates its arguments
//! at the boundary, resolves its token through the handle table, calls the
//! facade, and records any failure as the handle's last error. The caller
//! owns every buffer; a sizing call (NULL buffer, zero capacity) writes
//! only `required`.
//! Cites: include/espice.h (the published contract: every export's
//! behaviour and every record layout; records.zig pins the layouts).
//!
//! Contents:
//! - `pub const std_options: std.Options`
//! - `pub const vera_validate_contract: bool`
//! - `pub export fn espice_abi_version() u32`
//! - `pub export fn espice_default_options(out: ?*CreateOptions) void`
//! - `pub export fn espice_create(options: ?*const CreateOptions, out: ?*?*Token, diagnostic: ?[*]u8, diagnostic_capacity: usize) u32`
//! - `pub export fn espice_destroy(handle: ?*Token) u32`
//! - `pub export fn espice_query_count(handle: ?*Token, out: ?*u32) u32`
//! - `pub export fn espice_get_query_info(handle: ?*Token, id: u32, out: ?*QueryInfo) u32`
//! - `pub export fn espice_ready_queries(handle: ?*Token, scope: Scope, ids: ?[*]u32, capacity: usize, required: ?*usize) u32`
//! - `pub export fn espice_advance(handle: ?*Token, id: u32, out: ?*Advance) u32`
//! - `pub export fn espice_advance_ready(handle: ?*Token, ids: ?[*]const u32, count: usize, max_parallel: u32, events: ?[*]Advance, capacity: usize, required: ?*usize) u32`
//! - `pub export fn espice_run_all(handle: ?*Token) u32`
//! - `pub export fn espice_append_directives(handle: ?*Token, directives: Bytes, ids: ?[*]u32, capacity: usize, required: ?*usize) u32`
//! - `pub export fn espice_set_params(handle: ?*Token, names: ?[*]const Bytes, values: ?[*]const f64, count: usize) u32`
//! - `pub export fn espice_get_result_info(handle: ?*Token, id: u32, out: ?*ResultInfo) u32`
//! - `pub export fn espice_copy_result(handle: ?*Token, id: u32, values: ?[*]f64, capacity: usize, required: ?*usize) u32`
//! - `pub export fn espice_result_view(handle: ?*Token, id: u32, data: ?*?[*]const f64, len: ?*usize) u32`
//! - `pub export fn espice_copy_result_name(handle: ?*Token, id: u32, variable: u32, buffer: ?[*]u8, capacity: usize, required: ?*usize) u32`
//! - `pub export fn espice_print(handle: ?*Token, options: ?*const PrintOptions, buffer: ?[*]u8, capacity: usize, required: ?*usize) u32`
//! - `pub export fn espice_error_message(handle: ?*Token, buffer: ?[*]u8, capacity: usize, required: ?*usize) u32`
//! - `pub export fn espice_query_error_message(handle: ?*Token, id: u32, buffer: ?[*]u8, capacity: usize, required: ?*usize) u32`
//! - `pub export fn espice_append_points(handle: ?*Token, values: ?[*]const f64, count: usize, options: ?*const PointsOptions, first_id: ?*u32) u32`
//! - `pub export fn espice_copy_point_measures(handle: ?*Token, first_id: u32, count: usize, names: ?[*]const Bytes, measure_count: usize, out: ?[*]f64, reasons: ?[*]u32) u32`
//! - `pub export fn espice_copy_point_tangents(handle: ?*Token, first_id: u32, count: usize, names: ?[*]const Bytes, measure_count: usize, out: ?[*]f64) u32`
//! - `pub export fn espice_release_points(handle: ?*Token, first_id: u32, count: usize) u32`
//! - `pub export fn espice_lane_width(handle: ?*Token, out: ?*u32) u32`

const std = @import("std");
const builtin = @import("builtin");
const api = @import("espice");
const header = @import("espice_h");
const Handle = @import("c_api/Handle.zig");
const handles = @import("c_api/handles.zig");
const records = @import("c_api/records.zig");
const Token = handles.Token;
const Code = records.Code;
const Bytes = records.Bytes;
const CreateOptions = records.CreateOptions;
const PointsOptions = records.PointsOptions;
const Scope = records.Scope;
const QueryInfo = records.QueryInfo;
const Advance = records.Advance;
const ResultInfo = records.ResultInfo;
const PrintOptions = records.PrintOptions;
const abi_version = records.abi_version;
const no_query = records.no_query;
const status = records.status;
const constSlice = records.constSlice;
const mutableSlice = records.mutableSlice;
const strings = records.strings;
const concurrency = records.concurrency;
const decodeScope = records.decodeScope;
const writeTruncated = records.writeTruncated;

/// std reserves a 256 KB alternate signal stack per thread for stack-overflow
/// traces, as zero-filled TLS: every thread of every compilation unit pays it
/// (1.5 MB per thread across the device objects before this). Kept in Debug,
/// where the trace is worth it.
pub const std_options: std.Options = .{ .signal_stack_size = if (builtin.mode == .debug) 1 << 18 else null };

/// VerA's device/host contract checks (`contract.validating`), on in Debug
/// only: off, a device build spends 0.4-1.8% fewer instructions.
pub const vera_validate_contract = builtin.mode == .debug;

/// glibc malloc where libc is linked, else `core.gpa` (SmpAllocator, or the
/// page allocator single-threaded: wasm). A handle run batch after batch
/// starts a worker thread per query, and SmpAllocator kept what each freed:
/// 48 KB of RSS per released point, against 1.6 KB under malloc.
const allocator = if (builtin.link_libc) std.heap.c_allocator else if (builtin.single_threaded) std.heap.page_allocator else std.heap.smp_allocator;

pub export fn espice_abi_version() u32 {
    return abi_version;
}

pub export fn espice_default_options(out: ?*CreateOptions) void {
    const options = out orelse return;
    options.* = std.mem.zeroes(CreateOptions);
    options.abi_version = abi_version;
    options.struct_size = @sizeOf(CreateOptions);
    options.max_parallel = 1;
}

pub export fn espice_create(options: ?*const CreateOptions, out: ?*?*Token, diagnostic: ?[*]u8, diagnostic_capacity: usize) u32 {
    const destination = out orelse return status(error.InvalidArgument);
    destination.* = null;
    const buffer = mutableSlice(u8, diagnostic, diagnostic_capacity) catch return status(error.InvalidArgument);
    writeTruncated(buffer, "");
    destination.* = create(allocator, options orelse {
        writeTruncated(buffer, "InvalidArgument");
        return status(error.InvalidArgument);
    }) catch |err| {
        writeTruncated(buffer, @errorName(err));
        return status(err);
    };
    return 0;
}

pub export fn espice_destroy(handle: ?*Token) u32 {
    // NULL is a no-op, as for free().
    const h = handles.remove(handle orelse return 0) catch |err| return status(err);
    h.destroy();
    return 0;
}

pub export fn espice_query_count(handle: ?*Token, out: ?*u32) u32 {
    const h = handles.get(handle) catch |err| return status(err);
    const destination = out orelse return h.fail(error.InvalidArgument);
    destination.* = h.problem.query_count();
    return 0;
}

pub export fn espice_get_query_info(handle: ?*Token, id: u32, out: ?*QueryInfo) u32 {
    const h = handles.get(handle) catch |err| return status(err);
    const destination = out orelse return h.fail(error.InvalidArgument);
    const q = h.problem.query_info(@fromBackingInt(id)) catch |err| return h.fail(err);
    destination.* = records.queryInfo(q);
    return 0;
}

pub export fn espice_ready_queries(handle: ?*Token, scope: Scope, ids: ?[*]u32, capacity: usize, required: ?*usize) u32 {
    const h = handles.get(handle) catch |err| return status(err);
    const count = required orelse return h.fail(error.InvalidArgument);
    count.* = 0;
    const buffer = mutableSlice(u32, ids, capacity) catch |err| return h.fail(err);
    const selection = decodeScope(scope) catch |err| return h.fail(err);
    count.* = h.problem.ready_count(selection) catch |err| return h.fail(err);
    if (count.* > capacity) return h.fail(error.BufferTooSmall);
    // SAFETY: `QueryId` is `enum(u32)`, so a `[]u32` holds its values as is.
    _ = h.problem.ready_queries(selection, @ptrCast(buffer)) catch |err| return h.fail(err);
    return 0;
}

pub export fn espice_advance(handle: ?*Token, id: u32, out: ?*Advance) u32 {
    const h = handles.get(handle) catch |err| return status(err);
    const destination = out orelse return h.fail(error.InvalidArgument);
    const event = h.problem.advance(@fromBackingInt(id)) catch |err| return h.fail(err);
    destination.* = advanceRecord(h, event);
    return 0;
}

pub export fn espice_advance_ready(handle: ?*Token, ids: ?[*]const u32, count: usize, max_parallel: u32, events: ?[*]Advance, capacity: usize, required: ?*usize) u32 {
    const h = handles.get(handle) catch |err| return status(err);
    const needed = required orelse return h.fail(error.InvalidArgument);
    needed.* = count;
    const selected = constSlice(u32, ids, count) catch |err| return h.fail(err);
    const destination = mutableSlice(Advance, events, capacity) catch |err| return h.fail(err);
    const limit = concurrency(max_parallel) catch |err| return h.fail(err);
    if (capacity < count) return h.fail(error.BufferTooSmall);
    const scratch = allocator.alloc(api.Advance, count) catch |err| return h.fail(err);
    defer allocator.free(scratch);
    // SAFETY: `QueryId` is `enum(u32)`, so a `[]const u32` holds its values as is.
    needed.* = h.problem.advance_ready(@ptrCast(selected), .{ .max_parallel = limit }, scratch) catch |err| return h.fail(err);
    for (scratch[0..needed.*], destination[0..needed.*]) |event, *dest| dest.* = advanceRecord(h, event);
    return 0;
}

pub export fn espice_run_all(handle: ?*Token) u32 {
    const h = handles.get(handle) catch |err| return status(err);
    h.problem.run_all() catch |err| return h.fail(err);
    return 0;
}

pub export fn espice_append_directives(handle: ?*Token, directives: Bytes, ids: ?[*]u32, capacity: usize, required: ?*usize) u32 {
    const h = handles.get(handle) catch |err| return status(err);
    const count = required orelse return h.fail(error.InvalidArgument);
    count.* = 0;
    const text = constSlice(u8, directives.data, directives.len) catch |err| return h.fail(err);
    const buffer = mutableSlice(u32, ids, capacity) catch |err| return h.fail(err);
    // SAFETY: `QueryId` is `enum(u32)`, so a `[]u32` holds its values as is.
    count.* = h.problem.append_directives(text, @ptrCast(buffer)) catch |err| return h.fail(err);
    return if (count.* > capacity) h.fail(error.BufferTooSmall) else 0;
}

pub export fn espice_set_params(handle: ?*Token, names: ?[*]const Bytes, values: ?[*]const f64, count: usize) u32 {
    const h = handles.get(handle) catch |err| return status(err);
    const name_bytes = constSlice(Bytes, names, count) catch |err| return h.fail(err);
    const numbers = constSlice(f64, values, count) catch |err| return h.fail(err);
    const keys = strings(allocator, name_bytes) catch |err| return h.fail(err);
    defer allocator.free(keys);
    h.problem.set_params(keys, numbers) catch |err| return h.fail(err);
    return 0;
}

pub export fn espice_get_result_info(handle: ?*Token, id: u32, out: ?*ResultInfo) u32 {
    const h = handles.get(handle) catch |err| return status(err);
    const destination = out orelse return h.fail(error.InvalidArgument);
    const result = h.problem.result(@fromBackingInt(id)) catch |err| return h.fail(err);
    destination.* = .{
        .variable_count = std.math.cast(u32, result.varnames.len) orelse return h.fail(error.Overflow),
        .is_complex = @intFromBool(result.is_complex),
        .point_count = result.npoints,
        .value_count = result.data.len,
    };
    return 0;
}

pub export fn espice_copy_result(handle: ?*Token, id: u32, values: ?[*]f64, capacity: usize, required: ?*usize) u32 {
    const h = handles.get(handle) catch |err| return status(err);
    const count = required orelse return h.fail(error.InvalidArgument);
    count.* = 0;
    const buffer = mutableSlice(f64, values, capacity) catch |err| return h.fail(err);
    const query: api.QueryId = @fromBackingInt(id);
    count.* = (h.problem.result(query) catch |err| return h.fail(err)).data.len;
    if (count.* > capacity) return h.fail(error.BufferTooSmall);
    _ = h.problem.copy_result(query, buffer) catch |err| return h.fail(err);
    return 0;
}

pub export fn espice_result_view(handle: ?*Token, id: u32, data: ?*?[*]const f64, len: ?*usize) u32 {
    // The header promises *data NULL and *len 0 on any failure, a NULL or
    // stale handle included (#76).
    if (data) |view| view.* = null;
    if (len) |count| count.* = 0;
    const h = handles.get(handle) catch |err| return status(err);
    const view = data orelse return h.fail(error.InvalidArgument);
    const count = len orelse return h.fail(error.InvalidArgument);
    const values = (h.problem.result(@fromBackingInt(id)) catch |err| return h.fail(err)).data;
    view.* = values.ptr;
    count.* = values.len;
    return 0;
}

pub export fn espice_copy_result_name(handle: ?*Token, id: u32, variable: u32, buffer: ?[*]u8, capacity: usize, required: ?*usize) u32 {
    const h = handles.get(handle) catch |err| return status(err);
    const result = h.problem.result(@fromBackingInt(id)) catch |err| return h.fail(err);
    const name = if (variable == no_query) result.plotname else if (variable < result.varnames.len) result.varnames[variable] else return h.fail(error.InvalidArgument);
    return copyString(h, name, buffer, capacity, required);
}

pub export fn espice_print(handle: ?*Token, options: ?*const PrintOptions, buffer: ?[*]u8, capacity: usize, required: ?*usize) u32 {
    const h = handles.get(handle) catch |err| return status(err);
    if (required == null) return h.fail(error.InvalidArgument);
    if (buffer == null and capacity != 0) return h.fail(error.InvalidArgument);
    const config: api.PrintOptions = if (options) |o| printOptions(o.*) catch |err| return h.fail(err) else .{};
    var writer = std.Io.Writer.Allocating.init(allocator);
    defer writer.deinit();
    // `Writer.Allocating` reports running out of memory as WriteFailed.
    h.problem.print(&writer.writer, config) catch |err| return h.fail(if (err == error.WriteFailed) error.OutOfMemory else err);
    return copyString(h, writer.written(), buffer, capacity, required);
}

pub export fn espice_error_message(handle: ?*Token, buffer: ?[*]u8, capacity: usize, required: ?*usize) u32 {
    const h = handles.get(handle) catch |err| return status(err);
    // Querying an error must not replace it with the sizing call's error.
    const saved = h.last_error;
    defer h.last_error = saved;
    return copyString(h, if (saved) |err| @errorName(err) else "", buffer, capacity, required);
}

pub export fn espice_query_error_message(handle: ?*Token, id: u32, buffer: ?[*]u8, capacity: usize, required: ?*usize) u32 {
    const h = handles.get(handle) catch |err| return status(err);
    const q = h.problem.query_info(@fromBackingInt(id)) catch |err| return h.fail(err);
    return copyString(h, if (q.failure) |err| @errorName(err) else "", buffer, capacity, required);
}

pub export fn espice_append_points(handle: ?*Token, values: ?[*]const f64, count: usize, options: ?*const PointsOptions, first_id: ?*u32) u32 {
    const h = handles.get(handle) catch |err| return status(err);
    const first = first_id orelse return h.fail(error.InvalidArgument);
    const len = std.math.mul(usize, count, h.problem.live.len) catch |err| return h.fail(err);
    const rows = constSlice(f64, values, len) catch |err| return h.fail(err);
    const config: api.requests.PointOptions = if (options) |o| pointOptions(o) catch |err| return h.fail(err) else .{};
    const id = h.problem.append_points(rows, count, config) catch |err| return h.fail(err);
    first.* = @backingInt(id);
    return 0;
}

pub export fn espice_copy_point_measures(handle: ?*Token, first_id: u32, count: usize, names: ?[*]const Bytes, measure_count: usize, out: ?[*]f64, reasons: ?[*]u32) u32 {
    const h = handles.get(handle) catch |err| return status(err);
    const name_bytes = constSlice(Bytes, names, measure_count) catch |err| return h.fail(err);
    const len = std.math.mul(usize, count, measure_count) catch |err| return h.fail(err);
    const values = mutableSlice(f64, out, len) catch |err| return h.fail(err);
    const codes = mutableSlice(u32, reasons, count) catch |err| return h.fail(err);
    const keys = strings(allocator, name_bytes) catch |err| return h.fail(err);
    defer allocator.free(keys);
    // SAFETY: `Reason` is `enum(u32)`, so the caller's array holds the codes as is.
    h.problem.point_measures(@fromBackingInt(first_id), keys, values, @ptrCast(codes)) catch |err| return h.fail(err);
    return 0;
}

pub export fn espice_copy_point_tangents(handle: ?*Token, first_id: u32, count: usize, names: ?[*]const Bytes, measure_count: usize, out: ?[*]f64) u32 {
    const h = handles.get(handle) catch |err| return status(err);
    const name_bytes = constSlice(Bytes, names, measure_count) catch |err| return h.fail(err);
    const len = std.math.mul(usize, count, measure_count) catch |err| return h.fail(err);
    const total = std.math.mul(usize, len, h.problem.live.len) catch |err| return h.fail(err);
    const values = mutableSlice(f64, out, total) catch |err| return h.fail(err);
    const keys = strings(allocator, name_bytes) catch |err| return h.fail(err);
    defer allocator.free(keys);
    h.problem.point_tangents(@fromBackingInt(first_id), count, keys, values) catch |err| return h.fail(err);
    return 0;
}

pub export fn espice_release_points(handle: ?*Token, first_id: u32, count: usize) u32 {
    const h = handles.get(handle) catch |err| return status(err);
    h.problem.release_points(@fromBackingInt(first_id), count) catch |err| return h.fail(err);
    return 0;
}

pub export fn espice_lane_width(handle: ?*Token, out: ?*u32) u32 {
    const h = handles.get(handle) catch |err| return status(err);
    const destination = out orelse return h.fail(error.InvalidArgument);
    destination.* = h.problem.lane_width();
    return 0;
}

// ---- private ----

/// `espice_create` over `gpa`, which the handle keeps for its own free. Not
/// exported: every C entry point passes `allocator`; only the in-file
/// allocation sweep passes another. Validates every field the caller's
/// `struct_size` covers before allocating the handle.
fn create(gpa: std.mem.Allocator, options: *const CreateOptions) !*Token {
    // A caller built before the toolchain fields passes a struct ending at
    // `output_path`; the fields past its size are never read.
    if (options.abi_version != abi_version) return error.AbiMismatch;
    if (options.struct_size < @offsetOf(CreateOptions, "zig_path")) return error.AbiMismatch;
    const toolchain = options.struct_size >= @offsetOf(CreateOptions, "live_params");
    const live = options.struct_size >= @offsetOf(CreateOptions, "display");
    const covers_display = options.struct_size >= @sizeOf(CreateOptions);
    const source = try constSlice(u8, options.source.data, options.source.len);
    const origin = try constSlice(u8, options.origin.data, options.origin.len);
    const path = try constSlice(u8, options.output_path.data, options.output_path.len);
    const zig_path = if (toolchain) try constSlice(u8, options.zig_path.data, options.zig_path.len) else "";
    const share_path = if (toolchain) try constSlice(u8, options.share_path.data, options.share_path.len) else "";
    const live_bytes = if (live) try constSlice(Bytes, options.live_params, options.live_param_count) else &.{};
    const live_params = try strings(gpa, live_bytes);
    defer gpa.free(live_params);
    if (source.len == 0 or options.explicit_gpu > 1) return error.InvalidArgument;
    const config: api.Options = .{
        .source = switch (options.source_kind) {
            0 => .{ .file = source },
            1 => .{ .bytes = .{ .data = source, .origin = origin } },
            else => return error.InvalidArgument,
        },
        .dialect = std.enums.fromInt(api.Dialect, options.dialect) orelse return error.InvalidArgument,
        .backend = .{
            .backend = std.enums.fromInt(api.Request, options.backend) orelse return error.InvalidArgument,
            .gpu_explicit = options.explicit_gpu == 1,
        },
        .output = .{
            .format = std.enums.fromInt(api.Format, options.output_format) orelse return error.InvalidArgument,
            .path = if (path.len == 0) null else path,
        },
        .max_parallel = try concurrency(options.max_parallel),
        .hdl_toolchain = .{
            .zig = if (zig_path.len == 0) null else zig_path,
            .share = if (share_path.len == 0) null else share_path,
        },
        .live_params = live_params,
    };
    const display: ?Handle.Display = if (!covers_display) null else if (options.display) |callback| .{ .callback = callback, .user = options.display_user } else null;
    const h = try Handle.create(gpa, config, display);
    errdefer h.destroy();
    return handles.insert(h);
}

/// An event's header record; an output failure it carries also becomes the
/// handle's last error.
fn advanceRecord(h: *Handle, event: api.Advance) Advance {
    if (event.delivery_error) |err| h.last_error = err;
    return records.advance(event);
}

/// An `espice_print_options` as the facade's `PrintOptions`.
fn printOptions(o: PrintOptions) records.InputError!api.PrintOptions {
    if (o.ascii > 1) return error.InvalidArgument;
    var config: api.PrintOptions = .{ .scope = try decodeScope(o.scope), .ascii = o.ascii == 1 };
    if (o.max_parallel != 0) config.limits = .{ .max_parallel = try concurrency(o.max_parallel) };
    config.preview = switch (o.preview) {
        0 => .run_all,
        1 => .{ .advance = @fromBackingInt(o.query) },
        // SAFETY: `QueryId` is `enum(u32)`, so a `[]const u32` holds its values as is.
        2 => .{ .advance_ready = @ptrCast(try constSlice(u32, o.ready_ids, o.ready_count)) },
        else => return error.InvalidArgument,
    };
    return config;
}

/// An `espice_points_options` as the facade's `PointOptions`. Fields
/// appended later are read only when `struct_size` covers them; MATCHA's
/// 40 bytes end at `warm_start`, tail padding included.
fn pointOptions(o: *const PointsOptions) (records.InputError || error{AbiMismatch})!api.requests.PointOptions {
    if (o.struct_size < @offsetOf(PointsOptions, "tangent_step")) return error.AbiMismatch;
    if (o.measures_only > 1 or o.early_stop > 1 or o.warm_start > 1) return error.InvalidArgument;
    if (o.abort_after < 0 or std.math.isInf(o.abort_after)) return error.InvalidArgument;
    var config: api.requests.PointOptions = .{
        .measures_only = o.measures_only == 1,
        .early_stop = o.early_stop == 1,
        .warm_start = o.warm_start == 1,
        .abort_after = if (std.math.isNan(o.abort_after)) null else o.abort_after,
        .abort_measure = try constSlice(u8, o.abort_measure.data, o.abort_measure.len),
    };
    if (o.struct_size >= @sizeOf(PointsOptions)) {
        if (!(o.tangent_step >= 0) or std.math.isInf(o.tangent_step)) return error.InvalidArgument;
        config.tangent_step = o.tangent_step;
    }
    return config;
}

/// Copies `text` NUL-terminated into the caller's buffer. Sets `required`
/// first; a short buffer receives nothing and fails with BUFFER_TOO_SMALL.
fn copyString(h: *Handle, text: []const u8, ptr: ?[*]u8, capacity: usize, required: ?*usize) u32 {
    const count = required orelse return h.fail(error.InvalidArgument);
    count.* = std.math.add(usize, text.len, 1) catch |err| return h.fail(err);
    const buffer = mutableSlice(u8, ptr, capacity) catch |err| return h.fail(err);
    if (capacity < count.*) return h.fail(error.BufferTooSmall);
    @memcpy(buffer[0..text.len], text);
    buffer[text.len] = 0;
    return 0;
}

fn createAppendRun(gpa: std.mem.Allocator) !void {
    const deck = "C sweep\nV1 in 0 dc 1 ac 1\nR1 in out 1k\nC1 out 0 1u\n.op\n.end\n";
    var options: CreateOptions = undefined;
    espice_default_options(&options);
    options.source_kind = 1;
    options.source = .{ .data = deck.ptr, .len = deck.len };
    const token = try create(gpa, &options);
    defer std.debug.assert(espice_destroy(token) == 0);
    const ac = ".ac dec 1 1 10\n";
    var ids: [1]u32 = undefined;
    var n: usize = 0;
    try expectOk(espice_append_directives(token, .{ .data = ac, .len = ac.len }, &ids, ids.len, &n));
    try expectOk(espice_run_all(token));
}

/// OUT_OF_MEMORY as `error.OutOfMemory`, which the sweep expects at an
/// induced failure; any other failing code fails the test.
fn expectOk(code: u32) !void {
    if (code == @backingInt(Code.out_of_memory)) return error.OutOfMemory;
    if (code != 0) return error.UnexpectedStatus;
}

// Expected: every induced failure surfaces as OUT_OF_MEMORY from the call
// that hit it, and destroy then frees everything (the C contract for a
// failed call, include/espice.h); the testing allocator checks the frees.
test "create, append and run_all report OUT_OF_MEMORY at every allocation failure and leak nothing" {
    try std.testing.checkAllAllocationFailures(@import("core").testing.fixed_growth, createAppendRun, .{});
}

test "status: every documented error maps to its include/espice.h code" {
    const cases = [_]struct { anyerror, Code }{
        .{ error.InvalidArgument, .invalid_argument },     .{ error.InvalidConcurrency, .invalid_argument },
        .{ error.InvalidComponent, .invalid_argument },    .{ error.QueryNotReady, .invalid_argument },
        .{ error.DuplicateQuery, .invalid_argument },      .{ error.InvalidAnalysisArguments, .invalid_argument },
        .{ error.InvalidQueryOptions, .invalid_argument }, .{ error.UnsupportedDirectiveMutation, .invalid_argument },
        .{ error.UnknownParameter, .invalid_argument },    .{ error.BufferTooSmall, .buffer_too_small },
        .{ error.OutOfMemory, .out_of_memory },            .{ error.TooManyProblems, .out_of_memory },
        .{ error.InvalidQuery, .invalid_query },           .{ error.ResultUnavailable, .result_unavailable },
        .{ error.AbiMismatch, .abi_mismatch },             .{ error.InvalidHandle, .invalid_handle },
        // The header's FAILED: numerical, I/O and output failures, and
        // set_params' bin and topology refusals.
        .{ error.SingularMatrix, .failed },                .{ error.FileNotFound, .failed },
        .{ error.ModelBinNotFound, .failed },              .{ error.UnsupportedCard, .failed },
        .{ error.Overflow, .failed },
    };
    for (cases) |case| try std.testing.expectEqual(@backingInt(case[1]), status(case[0]));
}

test "writeTruncated: as much as fits, NUL-terminated; nothing past the buffer" {
    for ([_]struct { usize, []const u8 }{ .{ 0, "" }, .{ 1, "" }, .{ 3, "ab" }, .{ 4, "abc" }, .{ 7, "abc" } }) |case| {
        var buffer: [8]u8 = @splat('x');
        writeTruncated(buffer[0..case[0]], "abc");
        if (case[0] != 0) try std.testing.expectEqualStrings(case[1], std.mem.sliceTo(&buffer, 0));
        try std.testing.expect(std.mem.allEqual(u8, buffer[@min(case[0], case[1].len + 1)..], 'x'));
    }
}

// The table is process-global: this test leaves every slot it took free
// again, except the one it retires on purpose.
test "handles: a full table refuses create without leaking, and a retired slot never matches again" {
    const t = std.testing;
    var dummy: Handle = undefined;
    var tokens: [handles.capacity]*Token = undefined;
    var n: usize = 0;
    defer for (tokens[0..n]) |token| {
        _ = handles.remove(token) catch unreachable;
    };
    while (handles.insert(&dummy)) |token| : (n += 1) tokens[n] = token else |err| try t.expectEqual(error.TooManyProblems, err);
    try t.expectEqual(@as(usize, handles.capacity - 1), n);
    // A full table: create builds the Problem, then frees it (the testing
    // allocator checks) and reports OUT_OF_MEMORY through its diagnostic.
    const deck = "full\nV1 a 0 1\nR1 a 0 1k\n.op\n.end\n";
    var options: CreateOptions = undefined;
    espice_default_options(&options);
    options.source_kind = 1;
    options.source = .{ .data = deck.ptr, .len = deck.len };
    try t.expectError(error.TooManyProblems, create(t.allocator, &options));
    var out: ?*Token = @ptrFromInt(8);
    var diagnostic: [32]u8 = undefined;
    try t.expectEqual(@backingInt(Code.out_of_memory), espice_create(&options, &out, &diagnostic, diagnostic.len));
    try t.expect(out == null);
    try t.expectEqualStrings("TooManyProblems", std.mem.sliceTo(&diagnostic, 0));
    // Every token still names its own slot, and a slot-0 value names none.
    for (tokens[0..n]) |token| try t.expectEqual(&dummy, try handles.get(token));
    try t.expectError(error.InvalidHandle, handles.get(@ptrFromInt(@as(usize, 1) << handles.index_bits)));
    // Free one slot; its old token is stale, the slot's next token differs.
    const old = tokens[n - 1];
    n -= 1;
    _ = try handles.remove(old);
    try t.expectError(error.InvalidHandle, handles.get(old));
    try t.expectError(error.InvalidHandle, handles.remove(old));
    const next = try handles.insert(&dummy);
    try t.expect(next != old);
    try t.expectEqual(@as(handles.Index, @truncate(@intFromPtr(old))), @as(handles.Index, @truncate(@intFromPtr(next))));
    // A slot at the last generation is retired on remove, not reused.
    const index: handles.Index = @truncate(@intFromPtr(next));
    handles.generation[index] = std.math.maxInt(handles.Generation);
    const last: *Token = @ptrFromInt(@as(usize, handles.generation[index]) << handles.index_bits | index);
    const free_len = handles.free_len;
    try t.expectEqual(&dummy, try handles.remove(last));
    try t.expectEqual(free_len, handles.free_len);
    try t.expectError(error.InvalidHandle, handles.get(last));
    try t.expectError(error.InvalidHandle, handles.get(next));
}

// Stateful fuzz (TESTING.md, C ABI misuse): random espice_* calls on
// up to three live problems, stale and forged handles mixed in. The model
// predicts every status code from include/espice.h alone: what was
// appended, which directive text maps to which id, what run_all completed,
// what was destroyed. `advance` leaves the queries it may have touched
// `.unknown` until a query_info observes them. Problems come from
// `create(testing.allocator)`, so a leak on any path fails the test.
test "C ABI stateful fuzz: random call sequences match a model of valid states and leak nothing" {
    const t = std.testing;
    const ok: u32 = 0;
    const code = struct {
        fn of(c: Code) u32 {
            return @backingInt(c);
        }
    }.of;
    const deck = "C fuzz\n.param rv=1k\nV1 in 0 dc 1 ac 1\nR1 in out {rv}\nC1 out 0 1u\n.op\n.end\n";
    const directives = [_][]const u8{ ".ac dec 1 1 10", ".tran 1u 3u", ".dc v1 0 1 0.5", ".op" };
    const State = enum { pending, complete, unknown };
    const Live = struct {
        token: *Token,
        /// Query ids `0..n`, with the model's state of each.
        states: [32]State = undefined,
        n: u32 = 0,
        /// The id `.op` got since the last set_params: only an OP is
        /// shared by an identical append; every other card is a new query.
        op: ?u32 = null,
        last_error: []const u8 = "",
    };
    const live_name = [_]Bytes{.{ .data = "rv", .len = 2 }};
    for ([_]u64{ 1, 2, 3, 4 }) |seed| {
        var prng = std.Random.DefaultPrng.init(seed);
        const random = prng.random();
        var live: [3]Live = undefined;
        var live_len: usize = 0;
        var stale: [64]*Token = undefined;
        var stale_len: usize = 0;
        defer for (live[0..live_len]) |p| {
            _ = espice_destroy(p.token);
        };
        for (0..200) |_| {
            const step = random.uintLessThan(u8, 12);
            if (live_len == 0 or (step == 0 and live_len < live.len)) {
                var options: CreateOptions = undefined;
                espice_default_options(&options);
                options.source_kind = 1;
                options.source = .{ .data = deck.ptr, .len = deck.len };
                options.live_params = &live_name;
                options.live_param_count = 1;
                live[live_len] = .{ .token = try create(t.allocator, &options) };
                live[live_len].states[0] = .pending; // the deck's .op
                live[live_len].n = 1;
                live[live_len].op = 0;
                live_len += 1;
                continue;
            }
            const p = &live[random.uintLessThan(usize, live_len)];
            const k = random.uintLessThan(u32, p.n);
            var n: usize = 0;
            var u: u32 = 0;
            switch (step) {
                0, 1 => if (p.n + 2 <= p.states.len) {
                    const which = random.uintLessThan(usize, directives.len);
                    const text = directives[which];
                    var id: u32 = no_query;
                    // A sizing call first: nothing is appended.
                    try t.expectEqual(code(.buffer_too_small), espice_append_directives(p.token, .{ .data = text.ptr, .len = text.len }, null, 0, &n));
                    try t.expectEqual(@as(usize, 1), n);
                    try t.expectEqual(ok, espice_query_count(p.token, &u));
                    try t.expectEqual(p.n, u);
                    p.last_error = "BufferTooSmall";
                    try t.expectEqual(ok, espice_append_directives(p.token, .{ .data = text.ptr, .len = text.len }, (&id)[0..1], 1, &n));
                    if (text[1] == 'o' and p.op != null) try t.expectEqual(p.op.?, id) else {
                        // A new card, perhaps with its .op prerequisite; an
                        // .op text may instead be such an existing prerequisite.
                        try t.expectEqual(ok, espice_query_count(p.token, &u));
                        try t.expect(id < u and (id >= p.n or text[1] == 'o'));
                        for (p.states[p.n..u]) |*s| s.* = .pending;
                        p.n = u;
                        if (text[1] == 'o') p.op = id;
                    }
                },
                2 => {
                    // Malformed input: a device card, or no analysis card.
                    const text = if (random.boolean()) "R9 in 0 1k" else "* nothing";
                    var id: u32 = no_query;
                    try t.expectEqual(code(.invalid_argument), espice_append_directives(p.token, .{ .data = text.ptr, .len = text.len }, (&id)[0..1], 1, &n));
                    try t.expectEqual(no_query, id);
                    try t.expectEqual(ok, espice_query_count(p.token, &u));
                    try t.expectEqual(p.n, u);
                    p.last_error = if (text[0] == 'R') "UnsupportedDirectiveMutation" else "InvalidAnalysisArguments";
                },
                3 => {
                    var event: Advance = undefined;
                    const bad = random.uintLessThan(u8, 4) == 0;
                    const id = if (bad) p.n + random.uintAtMost(u32, 3) else k;
                    if (bad) {
                        try t.expectEqual(code(.invalid_query), espice_advance(p.token, id, &event));
                        p.last_error = "InvalidQuery";
                    } else {
                        try t.expectEqual(ok, espice_advance(p.token, id, &event));
                        try t.expectEqual(id, event.requested);
                        for (p.states[0..p.n]) |*s| if (s.* == .pending) {
                            s.* = .unknown;
                        };
                    }
                },
                4 => {
                    var ids: [32]u32 = undefined;
                    try t.expectEqual(ok, espice_ready_queries(p.token, .{ .kind = 0, .id = 0 }, &ids, ids.len, &n));
                    var events: [32]Advance = undefined;
                    var required: usize = 0;
                    if (n > 0 and random.boolean()) {
                        // A duplicate frontier starts nothing.
                        const twice = [_]u32{ ids[0], ids[0] };
                        try t.expectEqual(code(.invalid_argument), espice_advance_ready(p.token, &twice, 2, 2, &events, events.len, &required));
                        p.last_error = "DuplicateQuery";
                    } else {
                        try t.expectEqual(ok, espice_advance_ready(p.token, &ids, n, random.uintAtMost(u32, 3) + 1, &events, events.len, &required));
                        try t.expectEqual(n, required);
                        for (ids[0..n], events[0..n]) |id, e| try t.expectEqual(id, e.requested);
                        for (p.states[0..p.n]) |*s| if (s.* == .pending) {
                            s.* = .unknown;
                        };
                    }
                },
                5 => {
                    try t.expectEqual(ok, espice_run_all(p.token));
                    @memset(p.states[0..p.n], .complete);
                },
                6 => {
                    var info: QueryInfo = undefined;
                    try t.expectEqual(ok, espice_get_query_info(p.token, k, &info));
                    try t.expectEqual(k, info.id);
                    try t.expectEqual(@as(u32, 0), info.failure_code);
                    const complete = header.ESPICE_COMPLETE;
                    switch (p.states[k]) {
                        .pending => try t.expectEqual(@as(u32, header.ESPICE_PENDING), info.status),
                        .complete => try t.expectEqual(@as(u32, complete), info.status),
                        .unknown => {
                            try t.expect(info.status <= complete); // pending, paused or complete
                            if (info.status == complete) p.states[k] = .complete;
                        },
                    }
                },
                7 => {
                    var result: ResultInfo = undefined;
                    const got = espice_get_result_info(p.token, k, &result);
                    var view: ?[*]const f64 = null;
                    var len: usize = 7;
                    const viewed = espice_result_view(p.token, k, &view, &len);
                    try t.expectEqual(got, viewed);
                    if (got == ok) {
                        try t.expect(p.states[k] != .pending);
                        p.states[k] = .complete;
                        try t.expectEqual(result.value_count, len);
                        // Any capacity: short is BUFFER_TOO_SMALL and untouched.
                        var values: [512]f64 = @splat(-1);
                        const capacity = random.uintAtMost(usize, @min(values.len, len + 1));
                        try t.expectEqual(if (capacity < len) code(.buffer_too_small) else ok, espice_copy_result(p.token, k, &values, capacity, &n));
                        try t.expectEqual(len, n);
                        if (capacity < len) {
                            try t.expect(std.mem.allEqual(f64, &values, -1));
                            p.last_error = "BufferTooSmall";
                        } else try t.expectEqualSlices(f64, view.?[0..len], values[0..len]);
                    } else {
                        try t.expectEqual(code(.result_unavailable), got);
                        try t.expect(p.states[k] != .complete);
                        try t.expect(view == null and len == 0);
                        p.last_error = "ResultUnavailable";
                    }
                },
                8 => {
                    // One fault per call: the header leaves the order of an
                    // unknown name and a NaN in one call unspecified.
                    const unknown = random.uintLessThan(u8, 4) == 0;
                    const value: f64 = if (!unknown and random.uintLessThan(u8, 3) == 0)
                        std.math.nan(f64)
                    else
                        500 + @as(f64, @floatFromInt(random.uintLessThan(u32, 5000)));
                    const names = [_]Bytes{.{ .data = "zz", .len = 2 }};
                    const values = [_]f64{value};
                    const got = espice_set_params(p.token, if (unknown) &names else &live_name, &values, 1);
                    if (unknown or std.math.isNan(value)) {
                        try t.expectEqual(code(.invalid_argument), got);
                        p.last_error = if (unknown) "UnknownParameter" else "InvalidArgument";
                    } else {
                        try t.expectEqual(ok, got);
                        // `.op` now names a new OP, at the new value.
                        p.op = null;
                    }
                },
                9 => {
                    // The diagnostic names the last failure, and reading it
                    // (a sizing failure included) does not replace it.
                    var message: [64]u8 = undefined;
                    try t.expectEqual(code(.buffer_too_small), espice_error_message(p.token, null, 0, &n));
                    try t.expectEqual(p.last_error.len + 1, n);
                    try t.expectEqual(ok, espice_error_message(p.token, &message, message.len, &n));
                    try t.expectEqualStrings(p.last_error, std.mem.sliceTo(&message, 0));
                },
                10 => {
                    // A forged token: never a live one's bits, by construction.
                    const bits = random.int(usize) | 1;
                    const forged: *Token = @ptrFromInt(bits);
                    if (for (live[0..live_len]) |q| {
                        if (q.token == forged) break true;
                    } else false) continue;
                    try t.expectEqual(code(.invalid_handle), espice_query_count(forged, &u));
                    try t.expectEqual(code(.invalid_handle), espice_destroy(forged));
                    try t.expectEqual(code(.invalid_argument), espice_run_all(null));
                },
                11 => {
                    try t.expectEqual(ok, espice_destroy(p.token));
                    if (stale_len < stale.len) {
                        stale[stale_len] = p.token;
                        stale_len += 1;
                    }
                    p.* = live[live_len - 1];
                    live_len -= 1;
                },
                else => unreachable, // else: uintLessThan(u8, 12)
            }
            if (stale_len > 0) {
                const old = stale[random.uintLessThan(usize, stale_len)];
                try t.expectEqual(code(.invalid_handle), espice_query_count(old, &u));
                try t.expectEqual(code(.invalid_handle), espice_destroy(old));
            }
        }
    }
}
