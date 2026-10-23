const std = @import("std");

test "runtime optional error equality" {
    var errors = [_]?anyerror{ null, null, error.ResultUnavailable };
    std.mem.doNotOptimizeAway(&errors);
    const found = std.mem.indexOfNone(?anyerror, &errors, &.{null});
    std.debug.print("optional error size={d}, last={?t}, findNone={?d}\n", .{ @sizeOf(?anyerror), errors[2], found });
    try std.testing.expectEqual(@as(?usize, 2), found);
}
