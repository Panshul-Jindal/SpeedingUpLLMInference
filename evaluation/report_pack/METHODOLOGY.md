# Methodology

How the data in this pack was produced. Numbers for results are in `FINDINGS.md`; this file describes the setup.

## System under test

| Item | Value |
|---|---|
| Hardware | 1 node, 2 x NVIDIA H100 NVL (94 GB, compute capability 9.0, 400 W power limit), PCIe; Slurm node `ccs-srv004`, 16 CPU cores per job |
| Model | Llama-3.1-8B-Instruct (`unsloth/Llama-3.1-8B-Instruct`, an ungated mirror with identical weights) |
| Serving engine | vLLM 0.31.0 (CUDA 12.9 build), PyTorch 2.13.0, one `vllm serve` per GPU |
| Cell size | One GPU per configuration (tensor parallel = 1); two configurations measured at once, one per GPU |
| Driver | NVIDIA 590.48.01 |

## The five optimizations (each on or off)

| # | Name in data | Off (Default) | On |
|---|---|---|---|
| 1 | `quant` | BF16 weights | FP8 W8A8 (`--quantization fp8`, online dynamic) |
| 2 | `kvq` | BF16 KV cache | FP8 KV cache (`--kv-cache-dtype fp8`) |
| 3 | `spec` | no speculation | EAGLE-3 draft head `yuhuili/EAGLE3-LLaMA3.1-Instruct-8B`, 3 draft tokens |
| 4 | `prefix` | prefix caching off | prefix caching on |
| 5 | `batch` | engine default `max_num_seqs` (256) | `max_num_seqs` tuned per workload by a sweep (16-256) |

Chunked prefill and multi-GPU parallelism (optimizations 6-7 of the original 7-optimization design) were held off
in this phase. Configurations are written as 7-bit strings, optimization 1 first (`1011100` = quant + spec +
prefix + batch).

## Workload

- Synthetic prompts of random words with exact token lengths (model tokenizer), a pool of 4 shared prefixes making
  up 75% of a prompt for the "reuse" share of requests, and a fixed 256-token output (`ignore_eos`, greedy).
- Closed-loop load: exactly C requests in flight at all times.
- 7 workload points: centre (2048-token prompts, 32 in flight, 50% reuse) plus one regime changed at a time:
  prompt length 256 / 8192, concurrency 2 / 128, reuse 0% / 90%.
- Every configuration at a point replays the same saved, hashed request trace.

## Measurement

- Each run: 30 s warm-up (discarded), then a 120 s window; only requests that start and finish inside the window
  count. Throughput = output tokens per second over the window.
- Latency: time to first token (TTFT) and time per output token (TPOT), p50 and p99, measured client-side.
- Telemetry: GPU utilisation, clocks, power and memory (NVML); vLLM's Prometheus metrics (KV-cache usage, running
  and waiting requests, prefix-cache hits, speculative acceptance).
- Gates: p99 TPOT <= 100 ms, p99 TTFT <= 5 s, accuracy drop <= 1 point per task vs Default. Value = speedup over
  Default if all gates pass, else 1.0.

## Phase 1: factorial plan (`fast` profile)

- 32 configurations at the centre (full 2^5 factorial) and the 16 even-parity configurations (a 2^(5-1)
  half-fraction) at each of the 6 other points: 128 cells x 3 reps = 384 runs, about 11 h on 2 GPUs.
- Main effects and two-way interactions per point by OLS on log throughput with +-1 coding.
- Accuracy: lm-evaluation-harness against the running server for each of the 8 settings of quant/kvq/spec
  (GSM8K 500 questions, HumanEval 164, 4 MMLU subjects) plus exact-match agreement with Default on 100 prompts.

## Phase 2: focused investigation (`investigate` profile)

60 cells chosen from anomalies in phase 1, re-run with 2-5 reps and deep telemetry: NVML every 0.25 s, the full
vLLM metrics (incl. histograms and per-position speculative acceptance) every 1 s, and process CPU every 1 s.

| Investigation | Question | Cells |
|---|---|---|
| spec_prefix_no_reuse | Is spec + prefix really slower than Default at 0% reuse? | 4 reuse levels x 6 configs |
| spec_long_context | Where does EAGLE-3 acceptance collapse? | prompts 2048 / 4096 / 6144 / 8192 x 2 configs |
| high_load_dealias | Measure the 2x2 squares the half-fraction only modelled | 2 points x 11 configs |
| kvq_spec_outlier (+ _alone) | Was a noisy cell caused by the neighbouring GPU? | 1 cell x 5 reps, other GPU busy vs idle |
| binding_check | What limits throughput? | 3 points x 2 configs, other GPU idle, profiled |

- **Profiler passes:** 18 extra short runs with vLLM's built-in torch profiler recording 64 engine steps under
  the same load (GPU-busy fraction, top kernels). Their throughput is not used.
- **Paired accuracy:** settings Default, quant, kvq, quant + kvq and spec; full GSM8K (1,319), HumanEval (164) and
  8 MMLU subjects (1,687), with every question's answer logged so each setting is compared with Default on the
  same questions (McNemar test, paired 95% CI).

## Data cleaning (applied to everything in this pack)

1. **Detection.** Every phase-1 cell that phase 2 re-ran was compared run by run. All runs on GPU 0 from
   11:24 UTC (7 Nov) to the end of phase 1 were 0.56-0.69x of the re-run; GPU 1 and earlier GPU 0 runs matched
   within 2%. NVML showed extra device memory, higher clocks and lower power on GPU 0 in that window: another
   process was sharing the GPU (the cluster does not isolate GPUs).
2. **Correction.** All 71 runs (24 cells) in that window are flagged `contaminated`. Cells that phase 2 re-ran
   are replaced by the re-run (8 cells); the rest are excluded (16 cells). Cells that only phase 2 measured at
   the 7 standard points (e.g. single optimizations off-centre) are added (16 cells).
3. **Re-analysis.** The study's own analysis code (`harness/analysis`) was re-run on the cleaned dataset
   (`data/clean_*.csv`); the extra analyses in this pack are in `scripts/report_pack/`.

## Reproducing

From the repository: `python scripts/report_pack/build.py` (data and figures) then
`python scripts/report_pack/docs.py` (FINDINGS, CLAIMS, CONFIG_NOTES, DATA_DICTIONARY). Raw results:
`results/fast/` and `results/investigate/`.
