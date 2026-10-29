//! The `Problem` facade's root: the names main.zig and c_api.zig import, and
//! nothing else. One netlist prepared once, a DAG of analysis queries run
//! over it, and their results delivered to one output; the code is in
//! src/espice/.
//! Cites: none.
//!
//! Contents:
//! - `pub const Problem = @import("espice/Problem.zig")`
//! - `pub const Options = config.Options`
//! - `pub const MeasureWriters = config.MeasureWriters`
//! - `pub const HdlToolchain = config.HdlToolchain`
//! - `pub const DeliveryError = publish.DeliveryError`
//! - `pub const requests = core.query`
//! - `pub const QueryId = requests.QueryId`
//! - `pub const Query = requests.Query`
//! - `pub const Source = frontend.Source`
//! - `pub const Dialect = frontend.Dialect`
//! - `pub const parseDialect = frontend.parseDialect`
//! - `pub const detectDialect = frontend.detectDialect`
//! - `pub const Request = analysis.ExecutionConfig.Backend`
//! - `pub const gpu_available = analysis.gpu_available`
//! - `pub const Status = analysis.session.Status`
//! - `pub const Scope = analysis.session.Scope`
//! - `pub const Limits = analysis.session.Limits`
//! - `pub const Preview = analysis.session.Preview`
//! - `pub const PrintOptions = analysis.session.PrintOptions`
//! - `pub const QueryInfo = analysis.session.QueryInfo`
//! - `pub const Advance = analysis.session.Advance`
//! - `pub const Result = output.Result`
//! - `pub const Format = output.Format`
//! - `pub const Selection = output.Selection`
//! - `pub const parseFormat = output.parseFormat`
//! - `pub const Vpi = frontend.Library.Vpi`
//! - `pub const VpiError = frontend.Library.VpiError`
//! - `pub const VpiAnalogPoint = frontend.Library.AnalogPoint`
//! - `pub const VpiVerdict = frontend.Library.Verdict`

const analysis = @import("analysis");
const core = @import("core");
const frontend = @import("frontend");
const output = @import("output");
const config = @import("espice/config.zig");
const publish = @import("espice/publish.zig");

/// A prepared circuit, its query DAG and the output its results go to.
pub const Problem = @import("espice/Problem.zig");
/// Everything `Problem.init` needs. Its slices are copied during init.
pub const Options = config.Options;
/// `Options.measures`: the `.meas` values and the cards' error lines.
pub const MeasureWriters = config.MeasureWriters;
/// Where runtime `.hdl` compilation finds its toolchain (`Options.hdl_toolchain`).
pub const HdlToolchain = config.HdlToolchain;
/// Why publication stopped (`Problem.output_error`).
pub const DeliveryError = publish.DeliveryError;
/// Query request types (`core.query`): what `append_queries` takes.
pub const requests = core.query;
/// Dense DAG index, `0..query_count()`. Appends never renumber an ID.
pub const QueryId = requests.QueryId;
/// One analysis request; `append_queries` copies its slices.
pub const Query = requests.Query;
/// A netlist path, or in-memory text with the origin it is named by.
pub const Source = frontend.Source;
/// The netlist syntax family, which decides SPICE letter and LEVEL meaning.
pub const Dialect = frontend.Dialect;
/// Maps a CLI dialect name (`ngspice`, `hspice`, `spectre`); null if unknown.
pub const parseDialect = frontend.parseDialect;
/// The dialect a deck's text is in, from markers only HSPICE or Spectre
/// decks carry; ngspice otherwise (the CLI's `--tokenizer` default).
pub const detectDialect = frontend.detectDialect;
/// Requested compute backend; `Problem.init` rejects one the build lacks; a
/// query run with an explicit cuda/hip fails when no GPU is usable.
pub const Request = analysis.ExecutionConfig.Backend;
/// This build carries GPU kernel images.
pub const gpu_available = analysis.gpu_available;
/// A query's lifecycle state; `terminal()` says whether it can still run.
pub const Status = analysis.session.Status;
/// Which part of the DAG a call looks at: all, one query's chain, or one component.
pub const Scope = analysis.session.Scope;
/// Concurrency and quantum for one advancing call.
pub const Limits = analysis.session.Limits;
/// Which call's next frontier `print` marks.
pub const Preview = analysis.session.Preview;
/// `print` settings; a null `limits` takes the Problem's.
pub const PrintOptions = analysis.session.PrintOptions;
/// A query's kind, status, prerequisite, component and progress.
pub const QueryInfo = analysis.session.QueryInfo;
/// What one quantum did: which query ran, where it and its target stand.
pub const Advance = analysis.session.Advance;
/// A published plot. Its slices borrow Problem storage until `deinit`.
pub const Result = output.Result;
/// Output file format; some (touchstone, citi) hold only S-parameter queries.
pub const Format = output.Format;
/// Output format and path. A null path keeps results in memory only.
pub const Selection = output.Selection;
/// Maps a CLI format name (`binary`, `csv`, ...); null if unknown.
pub const parseFormat = output.parseFormat;
/// The VPI host interface (`Options.vpi`), which vpi_host.zig implements.
pub const Vpi = frontend.Library.Vpi;
/// Why `Vpi.load_app` or `Vpi.attach` failed; Library.zig names each cause.
pub const VpiError = frontend.Library.VpiError;
/// The solution an analysis is attempting, as `Vpi.point` hands it over.
pub const VpiAnalogPoint = frontend.Library.AnalogPoint;
/// What the VPI callbacks decided about the point (`Vpi.verdict`).
pub const VpiVerdict = frontend.Library.Verdict;
