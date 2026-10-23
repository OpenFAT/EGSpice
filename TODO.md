# TODO

Open issues and decisions from the 2026-10-22/23 performance round and the VerA
integration. Performance items in detail: [docs/performance/TODO/REMAINING.md](docs/performance/TODO/REMAINING.md).

## Release and pins

Nothing is pushed or tagged until ESPice, VerA and Gompute sync perfectly (user decision); until
then pins are local: ESPice `dev/performance` (local), VerA via a local tarball of the integration
worktree, then one coordinated push of all three.

- **VerA v0.13.0**: the candidate is VerA branch `dev/performance` (release commit f9ff7a2e,
  the suite at 3dad22b). The user refines it and merges to VerA main; pin ESPice to that merged
  main tip (not a branch commit), then apply `$S/agents/vera/espice-sim-reads.patch`
  (`sim_reads` replacing the `bypass_models` allowlist, 11 more models bypass-eligible).

## VerA ↔ ESPice integration (agent `bridge`, with session vera-26)

- **Call `setup` before `say`** in every host path that can reach a display task (with the pin).

## Performance round 2 (agents still running)

| Agent        | Remaining                                                                                                                                                                          |
| ------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| gpu-resident | 02 2.7 graph capture → multi-GPU stages 0–3 (device table and fake-device test knob in Gompute, sweeps across GPUs, per-device placement and constants, measured per-device rates) |
| gpu-tune     | 02 2.7 admission bars, tail-cut refit; 05 5.3(a) host stamping beside the device eval; `-Dgpu-jac-f32` removal gate                                                                |
| gpu-lanes2   | 06 6.9 OP/DC lanes gate, 6.12 charge-kernel colouring, 6.8 staged cards and `par()`, 6.15 residue, 03 3.9 host-stepped batches on the device                                       |
| host-solver  | chain_bsim4_100k cold OP: gmin-stepping fixes (budget-relative thresholds, gentler back-off on LU refusals, ngspice's 1.00005 stop) and the gmin-assisted source start, unmeasured |
| frontend2    | 07 7.3 body-token memo and 7.5 parser threads (parked: need builds)                                                                                                                |
| freq2        | 09 9.3 GCRO-DR and oscillator period column, 9.2 GPU complex frequency kernel and resident factors, 9.4(a) HB on the GPU, `factorEach` allocations                                 |
