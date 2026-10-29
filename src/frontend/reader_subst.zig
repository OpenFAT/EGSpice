//! Parameter substitution: compiled expression text spliced with the
//! scopes' values and `.func` bodies, probes mapped to nets.
//! Cites: ngspice numparam (xpressn.c).
//!
//! Contents:
//! - `pub fn Methods(comptime S: type) type`
//! - `pub fn subst(r: *R, src: *const expr.Scratch, from: usize, to: usize, frame: *const Frame, scopes: []const *const Scope, depth: u8) Error!void`
//! - `pub fn bound(bindings: []const Binding, name: []const u8) ?[]const expr.Op`
//! - `pub fn inline_(r: *R, src: *const expr.Scratch, op: expr.Op, frame: *const Frame, scopes: []const *const Scope, depth: u8) Error!void`
//! - `pub fn compiledText(r: *R, text: []const u8) Error!*const expr.Scratch`
//! - `pub fn levelPath(frame: *const Frame, level: usize) []const u8`

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
const netlist_view = @import("netlist_view.zig");
const reader_frame = @import("reader_frame.zig");
const none = card_rows.none;
const Error = card_rows.Error;
const siteKey = netlist_view.siteKey;
const nameKey = netlist_view.nameKey;
const Scope = reader_frame.Scope;
const Frame = reader_frame.Frame;

/// The stage's `Reader` methods over dialect syntax `S`; `Reader(S)`
/// aliases each one as its own decl.
pub fn Methods(comptime S: type) type {
    return struct {
        const R = reader.Reader(S);
        const Binding = R.Binding;
        const find = R.find;

        /// Copies `src`'s ops `from..to` into the netlist's ops. A name a scope
        /// defines is replaced by its definition, read in the scope that
        /// defined it; probe nets are mapped through `frame`.
        pub fn subst(r: *R, src: *const expr.Scratch, from: usize, to: usize, frame: *const Frame, scopes: []const *const Scope, depth: u8) Error!void {
            if (depth == 64) return error.ParseError;
            // `expr.Code.ident`'s final operand: declared geometry, then the
            // B-source tape's simulator variables.
            const ident_class = std.StaticStringMap(u32).initComptime(.{ .{ "l", 1 }, .{ "w", 1 }, .{ "mult", 1 }, .{ "time", 2 }, .{ "temper", 3 } });
            var i = from;
            while (i < to) : (i += 1) {
                const op = src.ops.items[i];
                switch (op.code) {
                    .num => try r.emitNum(src.consts.items[op.a]),
                    .ident => {
                        const name = src.names.items[op.a];
                        // A formal of the function being expanded: its argument.
                        if (bound(r.bindings.items, name)) |ops| {
                            try r.ops.appendSlice(r.arena, ops);
                            continue;
                        }
                        const hit = find(scopes, name) orelse {
                            try r.ops.append(r.arena, .{ .code = .ident, .a = ident_class.get(name) orelse 0 });
                            continue;
                        };
                        if (hit.level == 0) if (r.live_names.getIndex(name)) |k| {
                            try r.ops.append(r.arena, .{ .code = .live, .a = @intCast(k) });
                            continue;
                        };
                        switch (hit.entry) {
                            .num => |n| try r.emitNum(n),
                            .text => |text| {
                                const code = try r.compiledText(text);
                                // Calls inside a parameter's text draw once per
                                // scope that defines it: globally, or per instance.
                                const owner = r.site_owner;
                                const ordinal = r.site_ordinal;
                                defer {
                                    r.site_owner = owner;
                                    r.site_ordinal = ordinal;
                                }
                                if (r.monte) r.site_owner = nameKey(nameKey(0, levelPath(frame, hit.level)), name);
                                r.site_ordinal = 0;
                                // A parameter's text never sees the formals of a
                                // function that reads it.
                                const saved = r.bindings;
                                r.bindings = .empty;
                                defer r.bindings = saved;
                                try r.subst(code, 0, code.ops.items.len, frame, scopes[0 .. hit.level + 1], depth + 1);
                            },
                        }
                    },
                    .ucall => try r.inline_(src, op, frame, scopes, depth),
                    // The device by its flattened card name (`commit`), for a B-source tape.
                    .iprobe => {
                        const a = if (op.a == none) none else a: {
                            const dev = src.names.items[op.a];
                            const flat = if (frame.path) |path| try r.joined(&.{ dev[0..1], ".", path, ".", dev }) else dev;
                            break :a (try r.internName(flat)).index();
                        };
                        try r.ops.append(r.arena, .{ .code = .iprobe, .a = a, .b = none });
                    },
                    .vprobe => try r.ops.append(r.arena, .{
                        .code = .vprobe,
                        .a = if (op.a == none) none else (try r.netOf(frame, src.names.items[op.a])).index(),
                        .b = if (op.b == none) none else (try r.netOf(frame, src.names.items[op.b])).index(),
                    }),
                    else => {
                        if (r.monte and op.code == .call and expr.isDistribution(@fromBackingInt(@intCast(op.a)))) {
                            try r.site_ops.append(r.arena, @intCast(r.ops.items.len));
                            try r.site_keys.append(r.arena, siteKey(r.site_owner, r.site_ordinal));
                            r.site_ordinal += 1;
                        }
                        try r.ops.append(r.arena, op);
                    },
                }
            }
        }

        /// The ops a formal named `name` stands for, the innermost binding.
        pub fn bound(bindings: []const Binding, name: []const u8) ?[]const expr.Op {
            var i = bindings.len;
            while (i > 0) {
                i -= 1;
                if (std.mem.eql(u8, bindings[i].name, name)) return bindings[i].ops;
            }
            return null;
        }

        /// Emits a call `op` of a function outside `expr.Fn`: a `.func`
        /// expands to its body with each formal bound to its argument (the
        /// arguments' final ops, the last `op.b` subtrees emitted); any
        /// other stays a `call` of `Fn.other`, which nothing folds.
        pub fn inline_(r: *R, src: *const expr.Scratch, op: expr.Op, frame: *const Frame, scopes: []const *const Scope, depth: u8) Error!void {
            const name = src.names.items[op.a];
            const func = r.funcs.get(name) orelse
                return r.ops.append(r.arena, .{ .code = .call, .a = @backingInt(expr.Fn.other), .b = op.b });
            if (func.formals.len != op.b) return error.ParseError;
            // Lift the argument subtrees off the output, last argument first.
            const args = try r.arena.alloc([]const expr.Op, op.b);
            var end = r.ops.items.len;
            var k = args.len;
            while (k > 0) {
                k -= 1;
                const start = expr.subtreeStart(r.ops.items[0..end], end - 1);
                args[k] = try r.arena.dupe(expr.Op, r.ops.items[start..end]);
                end = start;
            }
            r.ops.shrinkRetainingCapacity(end);
            const code = try r.compiledText(func.body);
            const outer = r.bindings.items.len;
            defer r.bindings.shrinkRetainingCapacity(outer);
            for (func.formals, args) |formal, a| try r.bindings.append(r.arena, .{ .name = formal, .ops = a });
            try r.subst(code, 0, code.ops.items.len, frame, scopes, depth + 1);
        }

        /// `text` compiled, once per distinct text: a parameter's or a
        /// function's text is read wherever it is named, once per instance
        /// under a subcircuit. Compiling reads nothing but the text, so a
        /// cached copy is what a fresh compile gives; a text that fails to
        /// compile is not kept and fails the same way again.
        pub fn compiledText(r: *R, text: []const u8) Error!*const expr.Scratch {
            const got = try r.compiled_texts.getOrPut(r.arena, text);
            if (got.found_existing) return got.value_ptr.*;
            errdefer r.compiled_texts.removeByPtr(got.key_ptr);
            // Its names are slices of the text, so the key owns a copy.
            const key = try r.arena.dupe(u8, text);
            const code = try r.arena.create(expr.Scratch);
            code.* = .{};
            try expr.compileAll(S, r.arena, code, key);
            got.key_ptr.* = key;
            got.value_ptr.* = code;
            return code;
        }

        /// The instance path whose scope sits at `level` of `frame`'s
        /// scopes: its first `level` dotted components, "" for globals.
        pub fn levelPath(frame: *const Frame, level: usize) []const u8 {
            const path = frame.path orelse return "";
            if (level == 0) return "";
            var seen: usize = 0;
            for (path, 0..) |c, i| if (c == '.') {
                seen += 1;
                if (seen == level) return path[0..i];
            };
            return path;
        }
    };
}
