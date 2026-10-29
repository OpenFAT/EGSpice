//! One analysis card to its query: the dispatch spine over `Kind`, one
//! prong per card, each reading its arguments in the file that owns that
//! family (analysis_time.zig, analysis_small.zig, analysis_rf.zig).
//! Cites: the card readers each family file cites.
//!
//! Contents:
//! - `pub fn buildJob(a: netlist.Analysis, sources: core.QueryBindings, cards: []const requests.CardRef, ctx: analysis_args.CardContext) !?Job`

const std = @import("std");
const assert = std.debug.assert;
const testing = std.testing;
const Allocator = std.mem.Allocator;
const core = @import("core");
const requests = core.query;
const numerics = core.numerics;
const netlist = @import("netlist");
const Value = netlist.Value;
const nameAt = Value.nameAt;
const numberAt = Value.numberAt;
const Job = requests.Query;
const GROUND = core.GROUND;
const Library = @import("device").Library;
const analysis_args = @import("analysis_args.zig");
const analysis_time = @import("analysis_time.zig");
const analysis_small = @import("analysis_small.zig");
const analysis_rf = @import("analysis_rf.zig");

/// The query one analysis card asks for; null for a card that only
/// configures the deck (single-value `.temp`). Errors name the argument
/// that is wrong: `InvalidAnalysisArguments`, `AnalysisNodeNotFound`,
/// `AnalysisSourceNotFound`, `UnsupportedFrequencySweep`,
/// `UnsupportedAnalysisOutput`, and `MissingAnalysisCard` for an HSPICE form
/// whose deck lacks the card it reads (`ctx`). Point lists and labels are
/// allocated in `ctx.arena`; `OutOfMemory` when it fails.
pub fn buildJob(a: netlist.Analysis, sources: core.QueryBindings, cards: []const requests.CardRef, ctx: analysis_args.CardContext) !?Job {
    assert(a.args.len < std.math.maxInt(u32));
    return switch (a.kind) {
        .op => try analysis_time.opJob(a),
        .tran, .tran_noise, .matex => try analysis_time.tranJob(a, ctx),
        .ac => try analysis_small.acJob(a, ctx),
        .disto => try analysis_small.distoJob(a, sources),
        .dc => try analysis_time.dcJob(a, cards, ctx),
        .noise => try analysis_small.noiseJob(a, sources, ctx),
        .pnoise => try analysis_rf.pnoiseJob(a, sources, ctx),
        .tf => try analysis_small.tfJob(a, sources),
        .sens, .dcmatch => try analysis_small.sensJob(a, ctx),
        .acmatch, .dcsens => try analysis_small.acmatchJob(a, ctx),
        .four => try analysis_time.fourJob(a, ctx),
        .fft => .{ .fft = try analysis_time.fft(ctx, a.args, a.pos, a.neg) },
        .pz => try analysis_small.pzJob(a, sources),
        .pss => try analysis_rf.pssJob(a),
        .hb => try analysis_rf.hbJob(a, ctx),
        .qpss => try analysis_rf.qpssJob(a),
        .hbac, .hbxf, .hbnoise => try analysis_rf.hbacJob(a, sources),
        .hblin => try analysis_rf.hblinJob(a, sources, ctx),
        .phasenoise => try analysis_rf.phasenoiseJob(a),
        .pac, .pxf => try analysis_rf.pacJob(a, ctx),
        .sp => try analysis_rf.spJob(a, sources, ctx),
        .stb => try analysis_small.stbJob(a, sources),
        .lstb => .{ .lstb = try analysis_small.lstbJob(a.args, sources, ctx.ac, a.pos) },
        .dcxf => try analysis_small.dcxfJob(a, sources),
        .acxf => try analysis_small.acxfJob(a, sources),
        .dcinc => try analysis_small.dcincJob(a),
        .envelope => try analysis_time.envelopeJob(a),
        .mc => try analysis_time.mcJob(a),
        .temp => try analysis_time.tempJob(a),
    };
}

// ---- private ----
