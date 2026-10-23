**Verdict:** the plan in `docs/performance/gpu-batching.md` has the right shape, and I verified its per-round account against the lane backend, the device kernels and the host step machine. Three things it misses change the priorities. The host keeps a whole circuit, run and solver workspace per lane, so host RAM caps N before device RAM does. The LTE tape ring is the only lane-minor array, which leaves four kernels uncoalesced and explains why LTE cost half of eval in the recorded trace. And every per-round upload is a synchronous copy from pageable memory, which fences the whole device, so a round carries about 16 fences on top of the two syncs per Newton tick. One ceiling also has to be said plainly: E1 measured the 4060 at the work of 12 to 15 host cores per instance evaluation, so even a perfect device-resident loop lands near one 16-core CPU on that card. The 10k to 100k point case pays off on FP64-class parts, where the same kernels project 4 to 69 times faster, or on consumer cards only if the f32-Jacobian experiment works.

## Where a round goes at 10,000 lanes

Verified from the round loop in `src/analysis/gpu/lanes.zig:767-1150` and its host driver at `src/analysis/tran/tran.zig:1714-1860`.

| Phase   | Host work, one thread                                | Device fences                                                       |
| ------- | ---------------------------------------------------- | ------------------------------------------------------------------- |
| begin   | step machine per lane                                | 0                                                                   |
| predict |                                                      | 4 synchronous uploads                                               |
| newton  |                                                      | 4 uploads, then 2 download-and-sync per tick, 1 download at the end |
| probe   | probe-row scatter per lane                           | 2 uploads, 3 async downloads, 1 sync                                |
| end     | step machine per lane, waveform append, measure feed | 0                                                                   |
| apply   |                                                      | 6 uploads, 1 sync                                                   |

The uploads fence because the lane stream is a blocking stream and the plain copy runs on the legacy stream. The graph capture code relies on exactly that ordering.

Estimates from E1's measured kernel rate and the design note's host cost per lane:

| Lane circuit        | Device per tick, RTX 4060 | Host per round | Host share |
| ------------------- | ------------------------- | -------------- | ---------- |
| one transistor      | 1 to 2 ms                 | 20 to 40 ms    | over 90 %  |
| 48-transistor chain | about 90 ms               | 20 to 40 ms    | 20 to 30 % |
| either, H100 class  | 7 to 69 times smaller     | unchanged      | dominant   |

So for the "10k points of one device" case the GPU idles behind the host even on the 4060. Device-side step control is the fix, not faster copies.

## What to change, cheapest first

1. **Pinned host mirrors, async copies on the lane stream.** gompute already exposes page-locked allocation, but the lane backend's mirrors come from the general allocator, so even its async calls stage and block. Converting them removes about 16 fences per round with no numerical change. Smallest diff on this list.

2. **Make the LTE tape ring lane-major.** In the LTE, charge, accept and rebase kernels adjacent lanes read addresses a whole history block apart, so a warp touches 32 cache lines per load. Lane-major keeps each lane's operation order, so the body stays bitwise its host oracle. It only coalesces.

3. **Host stopgaps.** The two scalar passes are per-lane independent and write disjoint slots, so they can run on the pool's worker threads for roughly a 10× cut. Two half-batches on two streams would overlap one half's host pass with the other's Newton, but only after item 1, since synchronous uploads fence both streams.

4. **Step control on the device.** The state the step machine mutates per lane is about 15 scalars plus counters. The coefficient, growth-cap and method-ladder helpers are small scalar functions that need restating std-only beside `stepBound` with the same bitwise host tests. The one non-scalar piece is the breakpoint search, which calls a device hook per batch on the host. Enumerate each lane's breakpoints once at init into a sorted table with a cursor. Eligibility is already gated to the shapes this covers. Waveform rows go to a bounded device ring drained per launch, a reject-at or bound-step request parks that one lane with a yield flag, and the progress callback fires per launch.

5. **Multi-round graphs with masks, no conditional nodes.** Capture R rounds, each with its iteration limit unrolled. Every kernel already exits on the active and round masks. Only the three unconditional memsets in the eval pass and the counter zeroing need lane-masked clear kernels, so a lane that failed its refactor keeps its matrix until the host peels at the launch boundary. Waste is launch cost on idle ticks, about 14 nodes per tick at a few microseconds each. The coordination probe measured unrolled chunks faster than a conditional WHILE graph, and HIP has no conditional nodes, so this is the portable form. Conditional nodes stay a later experiment needing about five optional driver entry points in gompute, following its existing optional-function pattern, plus the device-side set-conditional extern.

6. **Peel without downloading the world.** A gather kernel compacts failed lanes' matrix and right-hand side into a staging buffer, so the download scales with failures. For small n, a dense partial-pivot LU from a device scratch pool removes the host from the path entirely. Every recorded run had zero peels, so this is about not stalling the other lanes.

7. **Measures on the device, subset only.** The measure module is 2,247 lines of ngspice measure semantics. Port only the per-step accumulators: find-at, max, min, avg, rms, integ, and when or trig-targ with linear crossing. Other cards keep the probe-row ring. Measures-only mode is the default when nothing reads waveforms, so this subset makes the common sweep fully resident, with early stop as a device flag.

8. **Host memory is the real cap on N.** Each lane materialises a whole circuit by copying every batch, plus a run and step machine holding about 20 n-vectors, the matrix copy and the charge rings, plus a solver workspace with its own factors. At 1,000 unknowns that is roughly 0.5 to 1 MB per lane, so 10 GB at 10k lanes, and the materialise loop is serial by design. The wave plan must bound the host too: materialise one resident wave at a time and free a lane's circuit when it finishes, keeping only the instance rows the latch check reads back.

9. **Occupancy and the datacenter underfill.** At 255 registers the eval kernel holds 8 warps per SM. 10k lanes is 1.6 waves on the 4060 but about 30 % of an H100's resident threads, so thread-per-lane underfills a datacenter part below 30k to 70k lanes. The alternative is threads per instance and lane with colouring and hub staging, which the phase-4 plan reserved. The 20.8 KB stack per thread exceeds L2 at full residency on both parts, so the VerA frame reduction and the f32 Jacobian are the device-side levers.

## Ceiling and what to measure

Rough budget for 10k single-transistor transients of 3,500 steps, estimates only:

| Configuration                  | Wall time                     |
| ------------------------------ | ----------------------------- |
| today, RTX 4060                | about 80 s                    |
| after items 1 to 5, RTX 4060   | about 12 s                    |
| after items 1 to 5, H100 class | 1 to 2 s                      |
| 16-core CPU                    | comparable to the 4060 result |

- **DC and OP sweeps have no GPU lane path.** The structural sweep solves points serially per circuit and the DC lanes are CPU-wide only. The Newton kernels already serve this case; the missing piece is a thin DC driver over the existing Newton with a CPU fallback for lanes that need gmin or source stepping.
- **Measure first** with `ESPICE_GPU_STATS=1` under an explicit CUDA backend, since the automatic policy keeps lanes on the CPU for the 4060. It prints the host, predict, Newton, probe and apply split and the idle fractions, which will confirm the host-bound regime at 10k lanes before any of the above is built.
