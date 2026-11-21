# LLM serving optimization study: report pack

Results, figures and notes for writing up the study of five vLLM serving optimizations on Llama-3.1-8B-Instruct
(2 x H100 NVL, November 2026). Everything here was generated from the measured results; nothing is estimated.

## Start here

| If you are... | Read |
|---|---|
| writing the report or slides | `FINDINGS.md`, then `figures/` |
| checking a number | `CLAIMS.csv` (claim -> evidence file) and `DATA_DICTIONARY.md` |
| describing the setup | `METHODOLOGY.md` |
| writing the limitations section | `LIMITATIONS.md` |
| explaining a specific configuration | `CONFIG_NOTES.md` |

## Contents

- `FINDINGS.md`: ten findings (F1-F10) with tables, each marked measured / interpretation / open question.
- `CLAIMS.csv`: one row per claim: statement, type, confidence, evidence file, figure, caveat.
- `CONFIG_NOTES.md`: every configuration at every workload, observed vs predicted speedup and the reason, from
  telemetry.
- `METHODOLOGY.md`, `LIMITATIONS.md`, `DATA_DICTIONARY.md`.
- `figures/`: 9 PNG charts (200 dpi, slide-ready):

| File | Shows | Finding |
|---|---|---|
| 01_contamination_timeline.png | Run throughput vs clean re-run over time, by GPU | F1 |
| 02_best_speedup_corrected.png | Best speedup per workload, first-reported vs corrected | F2 |
| 03_eagle3_long_context.png | EAGLE-3 acceptance by draft position and speedup vs prompt length | F6 |
| 04_prefix_gain_vs_reuse.png | Prefix-caching gain vs reuse, with and without spec | F4 |
| 05_composition_observed_vs_predicted.png | Observed vs product-of-singles speedup for every combination | F3 |
| 06_main_effects_heatmap.png | Effect of each optimization at each workload | F8 |
| 07_accuracy_paired.png | Paired accuracy change vs Default with 95% CI | F7 |
| 08_power_cap_clocks.png | SM clock and power by workload | F9 |
| 09_outlier_rerun.png | The noisy cell re-run 10 times | F10 |

- `data/`: 26 CSVs (see `DATA_DICTIONARY.md`). The main ones: `cells_combined_clean.csv` (one row per workload x
  configuration), `composition_per_config.csv`, `quality_paired_vs_default.csv`, `final_configs_by_policy.csv`.

## One-paragraph summary

We measured five vLLM optimizations (FP8 weights, FP8 KV cache, EAGLE-3 speculative decoding, prefix caching, tuned
batching) alone and in combination at seven workloads, then re-ran 60 anomalous cells with deep telemetry. The
re-runs exposed a silent co-tenant process that had slowed 18% of the first runs by ~40%; after correcting it, the
best configurations reach 1.4x-2.6x the Default throughput under the original accuracy rule. The optimizations
never cancel: their speedups multiply, and with speculative decoding they compound beyond that (up to +31%), with
measured mechanisms for prefix x spec and FP8 x spec. EAGLE-3 stops helping beyond ~4k-token prompts, FP8 weights
and FP8 KV are each accuracy-neutral but cost 2.6 GSM8K points together, and throughput is bounded by GEMM work on a
power-capped GPU. Details and caveats: `FINDINGS.md`, `LIMITATIONS.md`.
