//! Worker: runs one analysis on its own thread and parks it at every
//! progress checkpoint until the coordinator resumes or cancels it, and the
//! checkpoint protocol itself (`Event`, `Callback`). Independent of circuit
//! and solver state.
const std = @import("std");
const builtin = @import("builtin");
const requests = @import("core").query;

/// The analysis stage an `Event` reports from. Only `.nonlinear` is
/// filtered: workers park on it only under `Options.report_nonlinear`.
pub const Phase = enum(u8) {
    prepare,
    nonlinear,
    dc,
    frequency,
    transient,
    periodic,
    harmonic,
    sweep,
    postprocess,
};

/// Progress within the current phase invocation; a restarted solve may reset
/// it. `total == 0` means unknown. u64 because long runs pass 2^32 units.
pub const Event = struct {
    phase: Phase,
    completed: u64,
    total: u64 = 0,
    /// Transient only: time reached, the step just taken and the next one (s),
    /// and the accepted-point count.
    simulation_time: ?f64 = null,
    step_size: ?f64 = null,
    next_step: ?f64 = null,
    accepted: ?u32 = null,
};

/// Called by the analysis's own worker thread at a resumable boundary.
pub const Callback = struct {
    ctx: *anyopaque,
    yield_fn: *const fn (*anyopaque, Event) error{QueryCancelled}!void,

    /// Reports `event` and may block until resumed. Fails when the query was
    /// cancelled; the analysis must unwind.
    pub fn checkpoint(self: Callback, event: Event) error{QueryCancelled}!void {
        return self.yield_fn(self.ctx, event);
    }
};

/// Where a quantum started by `start` may end.
pub const Quantum = enum(u8) {
    /// At the next reported checkpoint: callers that step or interleave
    /// queries see every transient point.
    checkpoint,
    /// At the first reported checkpoint once `min_quantum_ns` of work ran
    /// since the quantum began or resumed: progress about every
    /// millisecond, without a park (two thread handoffs) per step.
    interval,
    /// Only at completion; checkpoints just poll cancellation. A parked
    /// checkpoint costs two thread handoffs per transient step.
    completion,
};

/// The least work an `.interval` quantum runs before it parks: a park is
/// two thread handoffs, which cost 2-11x a small transient's step.
pub const min_quantum_ns: i96 = 1_000_000;

/// Stack reserved per analysis thread, and per `ParEval` worker.
pub const stack_size: usize = 512 * 1024 * 1024;

/// Reusable analysis threads, each on a `stack_size` stack: a sweep of 10k
/// points spawns as many threads as ever run at once, not two per point.
/// A thread runs one task at a time and returns to the idle list when
/// `join` collects it; tasks that park at checkpoints keep their thread.
/// Must stay at a fixed address while any task runs; `deinit` after every
/// task was joined.
pub const Threads = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    mutex: std.Io.Mutex = .init,
    idle: std.ArrayList(*Slot) = .empty,
    all: std.ArrayList(*Slot) = .empty,

    pub const Slot = struct {
        pool: *Threads,
        thread: std.Thread = undefined,
        mutex: std.Io.Mutex = .init,
        changed: std.Io.Condition = .init,
        task: ?Task = null,
        done: bool = false,
        quit: bool = false,

        fn loop(slot: *Slot) void {
            const io = slot.pool.io;
            slot.mutex.lockUncancelable(io);
            defer slot.mutex.unlock(io);
            while (true) {
                while (slot.task == null and !slot.quit) slot.changed.waitUncancelable(io, &slot.mutex);
                const t = slot.task orelse return;
                slot.mutex.unlock(io);
                t.run(t.ctx);
                slot.mutex.lockUncancelable(io);
                slot.task = null;
                slot.done = true;
                slot.changed.broadcast(io);
            }
        }
    };

    pub const Task = struct { run: *const fn (*anyopaque) void, ctx: *anyopaque };

    pub fn init(gpa: std.mem.Allocator, io: std.Io) Threads {
        return .{ .gpa = gpa, .io = io };
    }

    /// Runs `task` on an idle thread, or a new one.
    pub fn spawn(self: *Threads, task: Task) !*Slot {
        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);
        const slot = self.idle.pop() orelse blk: {
            try self.all.ensureUnusedCapacity(self.gpa, 1);
            try self.idle.ensureTotalCapacity(self.gpa, self.all.items.len + 1);
            const s = try self.gpa.create(Slot);
            errdefer self.gpa.destroy(s);
            s.* = .{ .pool = self };
            s.thread = try std.Thread.spawn(.{ .stack_size = stack_size }, Slot.loop, .{s});
            self.all.appendAssumeCapacity(s);
            break :blk s;
        };
        slot.mutex.lockUncancelable(self.io);
        defer slot.mutex.unlock(self.io);
        slot.done = false;
        slot.task = task;
        slot.changed.broadcast(self.io);
        return slot;
    }

    /// Waits until `slot`'s task returned, then makes its thread idle.
    pub fn join(self: *Threads, slot: *Slot) void {
        slot.mutex.lockUncancelable(self.io);
        while (!slot.done) slot.changed.waitUncancelable(self.io, &slot.mutex);
        slot.mutex.unlock(self.io);
        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);
        // `spawn` reserved room for every slot.
        self.idle.appendAssumeCapacity(slot);
    }

    pub fn deinit(self: *Threads) void {
        std.debug.assert(self.idle.items.len == self.all.items.len);
        for (self.all.items) |s| {
            s.mutex.lockUncancelable(self.io);
            s.quit = true;
            s.changed.broadcast(self.io);
            s.mutex.unlock(self.io);
            s.thread.join();
            self.gpa.destroy(s);
        }
        self.all.deinit(self.gpa);
        self.idle.deinit(self.gpa);
    }
};

/// Per-worker settings, fixed at `init`.
pub const Options = struct {
    /// Park on `.nonlinear` checkpoints too. Only `.op` does; other queries
    /// park at outer boundaries and only poll cancellation inside Newton.
    report_nonlinear: bool = false,
    /// Print per-quantum timing under this query id.
    timing_query: ?requests.QueryId = null,
    /// Run on a pooled thread instead of a fresh one.
    threads: ?*Threads = null,
};

/// A resumable run of `run(ctx)` producing `Product` or failing with one of
/// `Error`, which must hold `error.QueryCancelled` (a cancel finishes the run
/// with it). Must stay at a fixed address from `callback` until `deinit`: the
/// thread and the callback hold it.
///
/// ponytail: one retained thread stack per started query; move to explicit
/// phase storage if stacks become the limiting resource.
pub fn Worker(comptime Error: type, comptime Product: type) type {
    comptime std.debug.assert(@typeInfo(Error) == .error_set);
    return struct {
        io: std.Io,
        ctx: *anyopaque,
        run: *const fn (*anyopaque) Error!Product,
        options: Options,
        mutex: std.Io.Mutex = .init,
        changed: std.Io.Condition = .init,
        thread: ?Thread = null,
        state: State = .created,
        /// The running quantum's end rule; written by `start` before the
        /// worker thread resumes, read only by that thread.
        quantum: Quantum = .checkpoint,
        cancel_requested: std.atomic.Value(bool) = .init(false),
        /// Posted once when the running quantum stops (a park or the end),
        /// then cleared: the coordinator's shared wake-up (`Session.Pool`).
        signal: ?*std.Io.Semaphore = null,
        timed_from: ?std.Io.Timestamp = null,
        active_ns: i96 = 0,
        /// When the running quantum began or last resumed; set and read
        /// only by the worker thread.
        quantum_from: ?std.Io.Timestamp = null,

        const Self = @This();
        const Thread = union(enum) {
            own: std.Thread,
            pooled: *Threads.Slot,

            fn join(t: Thread) void {
                switch (t) {
                    .own => |th| th.join(),
                    .pooled => |s| s.pool.join(s),
                }
            }
        };
        const State = union(enum(u8)) {
            created,
            running,
            paused: Event,
            finished: Error!Product,
        };

        /// What one `wait` returns: a parked checkpoint, the product, the
        /// run's error, or a cancel (`error.QueryCancelled` reads as one).
        pub const Outcome = union(enum(u8)) {
            progress: Event,
            complete: Product,
            failed: Error,
            cancelled,
        };

        /// An unstarted worker; nothing runs until `start`. `ctx` must
        /// outlive it.
        pub fn init(io: std.Io, ctx: *anyopaque, execute_fn: *const fn (*anyopaque) Error!Product, options: Options) Self {
            return .{ .io = io, .ctx = ctx, .run = execute_fn, .options = options };
        }

        /// Returns the checkpoint callback the analysis must call.
        pub fn callback(self: *Self) Callback {
            return .{ .ctx = self, .yield_fn = checkpoint };
        }

        /// Starts or resumes one quantum without waiting, so the coordinator
        /// can start several, and posts `signal` once when it stops. A failed
        /// thread spawn leaves the worker unstarted and retryable;
        /// `error.AlreadyRunning` if a quantum is in flight. Single-threaded,
        /// the first `start` runs the analysis to completion before it returns.
        pub fn start(self: *Self, quantum: Quantum, signal: ?*std.Io.Semaphore) !void {
            self.mutex.lockUncancelable(self.io);
            defer self.mutex.unlock(self.io);
            if (self.state != .running) {
                self.quantum = quantum;
                self.signal = signal;
            }
            switch (self.state) {
                .created => {
                    self.state = .running;
                    if (builtin.single_threaded) {
                        // No threads (the wasm playground): the whole run
                        // happens here, with checkpoints only polling cancel.
                        self.quantum = .completion;
                        self.mutex.unlock(self.io);
                        defer self.mutex.lockUncancelable(self.io);
                        self.main();
                    } else {
                        self.thread = self.spawnThread() catch |err| {
                            self.state = .created;
                            self.signal = null;
                            return err;
                        };
                    }
                },
                .paused => {
                    self.state = .running;
                    self.changed.broadcast(self.io);
                },
                .running => return error.AlreadyRunning,
                .finished => {},
            }
        }

        /// Whether the quantum `start` began has stopped, so `wait` returns
        /// without blocking.
        pub fn stopped(self: *Self) bool {
            self.mutex.lockUncancelable(self.io);
            defer self.mutex.unlock(self.io);
            return switch (self.state) {
                .running => false,
                .paused => !self.cancel_requested.load(.monotonic),
                .created, .finished => true,
            };
        }

        /// Waits for the quantum `start` began, or returns the terminal
        /// outcome again. Joins the thread once the run is over.
        /// `error.NotStarted` before the first `start`.
        pub fn wait(self: *Self) !Outcome {
            const outcome = try self.waitOutcome();
            if (outcome != .progress) if (self.thread) |thread| {
                thread.join();
                self.thread = null;
            };
            return outcome;
        }

        fn waitOutcome(self: *Self) !Outcome {
            self.mutex.lockUncancelable(self.io);
            defer self.mutex.unlock(self.io);
            while (true) switch (self.state) {
                .created => return error.NotStarted,
                .running => self.changed.waitUncancelable(self.io, &self.mutex),
                .paused => |event| {
                    if (self.cancel_requested.load(.monotonic)) {
                        self.changed.waitUncancelable(self.io, &self.mutex);
                    } else return .{ .progress = event };
                },
                .finished => |result| {
                    const value = result catch |err| return if (err == error.QueryCancelled)
                        .cancelled
                    else
                        .{ .failed = err };
                    return .{ .complete = value };
                },
            };
        }

        /// `start` then `wait`, one checkpoint quantum.
        pub fn advance(self: *Self) !Outcome {
            try self.start(.checkpoint, null);
            return self.wait();
        }

        /// Requests cancellation. A running analysis sees it at its next
        /// checkpoint; an unstarted one never runs.
        pub fn cancel(self: *Self) void {
            self.mutex.lockUncancelable(self.io);
            defer self.mutex.unlock(self.io);
            if (self.state == .finished) return;
            self.cancel_requested.store(true, .monotonic);
            if (self.state == .created) self.state = .{ .finished = error.QueryCancelled };
            self.changed.broadcast(self.io);
        }

        /// Cancels and joins, so the analysis's defers have run on return.
        pub fn deinit(self: *Self) void {
            self.cancel();
            if (self.thread) |thread| thread.join();
        }

        fn checkpoint(ctx: *anyopaque, event: Event) error{QueryCancelled}!void {
            const self: *Self = @ptrCast(@alignCast(ctx));
            // Suppressed checkpoints only poll cancellation, lock-free.
            if (self.quantum == .completion or (event.phase == .nonlinear and !self.options.report_nonlinear) or (self.quantum == .interval and !self.quantumSpent())) {
                if (self.cancel_requested.load(.monotonic)) return error.QueryCancelled;
                return;
            }
            if (self.options.timing_query) |id| {
                const elapsed = self.timeQuantum();
                if (event.simulation_time) |t| {
                    std.debug.print("timing: query {d} {s} attempt={d} accepted={?d} t={e:.6}s dt={?e:.6}s next_dt={?e:.6}s: {d:.6}ms\n", .{
                        @backingInt(id), @tagName(event.phase), event.completed, event.accepted, t, event.step_size, event.next_step, milliseconds(elapsed),
                    });
                } else {
                    std.debug.print("timing: query {d} {s} checkpoint={d}/{d}: {d:.6}ms\n", .{
                        @backingInt(id), @tagName(event.phase), event.completed, event.total, milliseconds(elapsed),
                    });
                }
            }
            self.mutex.lockUncancelable(self.io);
            defer self.mutex.unlock(self.io);
            if (self.cancel_requested.load(.monotonic)) return error.QueryCancelled;

            self.state = .{ .paused = event };
            self.changed.broadcast(self.io);
            self.post();
            while (self.state == .paused and !self.cancel_requested.load(.monotonic))
                self.changed.waitUncancelable(self.io, &self.mutex);
            if (self.cancel_requested.load(.monotonic)) return error.QueryCancelled;
            if (self.options.timing_query != null) self.timed_from = std.Io.Timestamp.now(self.io, .awake);
            self.quantum_from = std.Io.Timestamp.now(self.io, .awake);
        }

        /// Whether the running quantum has done `min_quantum_ns` of work.
        fn quantumSpent(self: *Self) bool {
            const from = self.quantum_from orelse return true;
            return from.durationTo(std.Io.Timestamp.now(self.io, .awake)).nanoseconds >= min_quantum_ns;
        }

        fn timeQuantum(self: *Self) i96 {
            const before = self.timed_from orelse return 0;
            const elapsed = before.durationTo(std.Io.Timestamp.now(self.io, .awake)).nanoseconds;
            self.timed_from = null;
            self.active_ns += elapsed;
            return elapsed;
        }

        fn spawnThread(self: *Self) !Thread {
            if (self.options.threads) |pool| return .{ .pooled = try pool.spawn(.{ .run = mainOpaque, .ctx = self }) };
            return .{ .own = try std.Thread.spawn(.{ .stack_size = stack_size }, main, .{self}) };
        }

        fn mainOpaque(ctx: *anyopaque) void {
            main(@ptrCast(@alignCast(ctx)));
        }

        fn main(self: *Self) void {
            if (self.options.timing_query != null) self.timed_from = std.Io.Timestamp.now(self.io, .awake);
            self.quantum_from = std.Io.Timestamp.now(self.io, .awake);
            const result = self.run(self.ctx);
            if (self.options.timing_query) |id| {
                const tail = self.timeQuantum();
                const status = if (result) |_| "complete" else |err| @errorName(err);
                std.debug.print("timing: query {d} {s}: final={d:.6}ms active_total={d:.6}ms\n", .{
                    @backingInt(id), status, milliseconds(tail), milliseconds(self.active_ns),
                });
            }
            self.mutex.lockUncancelable(self.io);
            defer self.mutex.unlock(self.io);
            self.state = .{ .finished = if (self.cancel_requested.load(.monotonic)) error.QueryCancelled else result };
            self.changed.broadcast(self.io);
            self.post();
        }

        /// Posts and clears `signal`; the caller holds `mutex`. A later
        /// stop, such as a cancel at teardown, posts nothing.
        fn post(self: *Self) void {
            const s = self.signal orelse return;
            self.signal = null;
            s.post(self.io);
        }
    };
}

fn milliseconds(ns: i96) f64 {
    return @as(f64, @floatFromInt(ns)) / 1e6;
}

test "Threads: tasks reuse idle threads, join waits for each" {
    const io = std.testing.io;
    var pool: Threads = .init(std.testing.allocator, io);
    defer pool.deinit();
    const Count = struct {
        n: std.atomic.Value(u32) = .init(0),
        fn bump(ctx: *anyopaque) void {
            const c: *@This() = @ptrCast(@alignCast(ctx));
            _ = c.n.fetchAdd(1, .monotonic);
        }
    };
    var c: Count = .{};
    for (0..20) |_| {
        const a = try pool.spawn(.{ .run = Count.bump, .ctx = &c });
        const b = try pool.spawn(.{ .run = Count.bump, .ctx = &c });
        pool.join(a);
        pool.join(b);
    }
    try std.testing.expectEqual(@as(u32, 40), c.n.load(.monotonic));
    try std.testing.expectEqual(@as(usize, 2), pool.all.items.len);
}
