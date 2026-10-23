//! Solver workspaces kept between the queries of one Problem, so a sweep's
//! points reuse each other's symbolic LU, scratch and allocations instead of
//! building and freeing them per point. A workspace is handed out only for
//! the same template pattern and the same solver knobs, and `reset` returns
//! it to its just-built state, so a reused one factors bitwise as a fresh one.

const std = @import("std");
const solver = @import("solver");
const converger = solver.converger;
const direct = solver.direct;

pub const WsPool = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    /// Idle workspaces kept at most; more are freed on `give`. A sweep needs
    /// one per running query.
    cap: usize,
    mutex: std.Io.Mutex = .init,
    idle: std.ArrayList(Entry) = .empty,

    const Entry = struct { key: usize, newton: converger.Knobs, lu: direct.Knobs, ws: converger.Workspace };

    /// Reserves room for `cap` workspaces, so `give` never allocates.
    pub fn init(gpa: std.mem.Allocator, io: std.Io, cap: usize) !WsPool {
        var self: WsPool = .{ .gpa = gpa, .io = io, .cap = cap };
        try self.idle.ensureTotalCapacity(gpa, cap);
        return self;
    }

    /// An idle workspace built for pattern `key` with these knobs, or null.
    /// The caller resets it before use.
    pub fn take(self: *WsPool, key: usize, newton: converger.Knobs, lu: direct.Knobs) ?converger.Workspace {
        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);
        for (self.idle.items, 0..) |e, i| {
            if (e.key == key and std.meta.eql(e.newton, newton) and std.meta.eql(e.lu, lu))
                return self.idle.swapRemove(i).ws;
        }
        return null;
    }

    /// Keeps `ws` for a later `take`, or frees it when the pool is full.
    /// `ws` must have been built with `gpa`.
    pub fn give(self: *WsPool, key: usize, newton: converger.Knobs, lu: direct.Knobs, ws: converger.Workspace) void {
        var w = ws;
        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);
        if (self.idle.items.len >= self.cap) return w.deinit(self.gpa);
        self.idle.appendAssumeCapacity(.{ .key = key, .newton = newton, .lu = lu, .ws = w });
    }

    pub fn deinit(self: *WsPool) void {
        for (self.idle.items) |*e| e.ws.deinit(self.gpa);
        self.idle.deinit(self.gpa);
    }
};
