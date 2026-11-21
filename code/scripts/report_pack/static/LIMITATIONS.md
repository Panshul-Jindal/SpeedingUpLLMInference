# Limitations and threats to validity

State these in the report. Each one limits how far a finding generalizes; none invalidates the findings as stated
in `FINDINGS.md`.

## About the workload

- **Synthetic text.** Prompts are random words, not real chat, code or documents. Speculative-decoding acceptance
  and prefix-cache hit rates depend strongly on real text, so absolute speedups for `spec` and `prefix` may differ
  on production traffic. Relative patterns (e.g. acceptance collapsing with prompt length) are more likely to
  transfer than exact numbers.
- **Fixed output length** (256 tokens, `ignore_eos`) and **closed-loop load** (constant in-flight count). Real
  traffic has variable lengths and bursty arrivals.
- **One model** (Llama-3.1-8B-Instruct) and **one draft head** for EAGLE-3. Larger models shift the balance between
  memory-bound and compute-bound work and may change which optimizations matter.

## About the hardware and environment

- **One node, two GPUs, power-capped at 400 W.** Clocks fall under load; on an uncapped or different GPU (e.g.
  H100 SXM) the same configurations may rank differently.
- **Shared, non-isolated cluster.** Another user's process silently shared GPU 0 for part of phase 1 (71 runs).
  Those runs are excluded or replaced, but other, smaller interference cannot be ruled out for the remaining runs.
  Evidence against it: 13 phase-1 cells re-run cleanly in phase 2 matched within 3%, and both GPUs show the same
  clocks and power where clean.
- **Two GPUs measured at once.** Tested on one cell (10 reps: busy vs idle neighbour, no difference); not tested
  for every configuration.

## About the design and statistics

- **16 cells have no clean value** (contaminated, never re-run) at high concurrency, 0% reuse and 90% reuse; a
  better configuration could be among them.
- **Half-fraction design off-centre.** At the 6 off-centre points only even-parity configurations were run in
  phase 1, so some interactions there are model estimates (labelled `model`). Phase 2 measured the most important
  ones directly.
- **Few repetitions** (2-3 per cell). Rep-to-rep variation was small (median 0.2% per cell, 90% of cells below 0.7%, worst 2.0%), but rare slow events could be missed.
- **"No-interaction" baseline.** Synergy is strongest evidence where all single-optimization speedups were measured
  (`independent_basis = product of measured singles`); comparisons against the fitted main-effects model are weaker.
- **Accuracy.** One greedy run per setting; HumanEval has only 164 problems (95% CI about +-5 points), so small
  drops there cannot be confirmed or excluded. Settings quant + spec, kvq + spec and all three were not
  paired-tested, so policy B's winners rely on an untested setting.

## About the tooling

- Two deep-telemetry signals in phase 2 are **not usable**: per-process CPU (a collector bug) and vLLM's FLOP/byte
  counters (not enabled). Both were fixed afterwards but not re-run. The bottleneck finding (F9) rests on profiler
  traces and NVML instead.
- Interpretations of mechanisms (F4, F5, F6) are consistent with the telemetry but were not proven with
  controlled ablations or kernel-level profiles of every configuration.
