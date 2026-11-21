# SMOKE_TEST_REPORT — test machine (1 × RTX PRO 4000 Blackwell SFF)

Status: **complete** — `python -m harness smoke --env test` ran all 7 steps with rc=0 on 2026-10-29 (18:28–19:26),
followed by a quality rerun (19:26–19:42) after fixing the stop-string issue in §3. Every one of optimizations
1–6 was switched on in at least one measured run; all report files were produced from real measurements
(`results/smoke/report/`).

## 1. Environment

| Item | Value |
|---|---|
| GPU | NVIDIA RTX PRO 4000 Blackwell **SFF Edition**, 24 GB, compute capability 12.0 (verified), **70 W power cap** |
| Driver / CUDA | 595.91.07 (CUDA 13.2 driver); wheels are CUDA 12.9 builds (main machine: driver 575.64 / CUDA 12.9) |
| Python | 3.11.18 (uv-managed; system Python is 3.10) |
| vLLM / torch | 0.31.0+cu129 / 2.13.0+cu129 (torch arch list includes sm_86 → runs on sm_89, and sm_120) |
| FlashInfer | flashinfer-python 0.7.0.post1 + flashinfer-jit-cache 0.7.0.post1+cu129 (sm80/89/90a/100a/103a/120f) + flashinfer-cubin 0.7.0.post1 |
| nvcc | user-level CUDA 12.9.86 toolchain from conda-forge in `.cuda-toolkit/` (no system toolkit, no root) |
| lm-eval | 0.4.13 |
| Model | `unsloth/Llama-3.1-8B-Instruct` (ungated mirror of meta-llama/Llama-3.1-8B-Instruct; same weights) |

Smoke-test numbers come from a 70 W small-form-factor card running at its power cap (SM clock ≈ 1.3 GHz under
load). They prove the code path only; no conclusions should be drawn from them (spec §14).

## 2. What ran, with timings

Profile `smoke`: points `center` (1,024-token prompts, concurrency 8, 50% reuse) and `context_high` (2,048 tokens);
configurations Default, each single optimization 1–6, all-on (8 per point); 1 rep; 10 s warm-up + 20 s window;
64 output tokens; quality `--limit 20`.

| Step | Time | Result |
|---|---|---|
| doctor | 12 s | PASS (GPU, cc 12.0, CUDA 12.9, vLLM 0.31.0+cu129, FP8 supported, 23.4 GiB free, all flags, all model repos) |
| traces | 7 s | 2 traces × 1,500 requests, exact token lengths, reuse 0.50, saved + hashed |
| startup-check | 561 s | 8/8 configurations start and answer (engine start 42–105 s each) |
| sweep | 703 s | 8 runs; centre → `max_num_seqs=64`, `max_num_batched_tokens=512`; `context_high`: no candidate passed the latency gates → placeholders kept (warning logged, runs marked `used_placeholder`) |
| quality (first pass) | 724 s | 5 numeric settings; MMLU ok, GSM8K/HumanEval failed (see §3) |
| run | 1,442 s | 16/16 ok (≈ 90 s per run: engine start 38–105 s + 30 s load + teardown) |
| analyze | 0.5 s | all 9 report files |
| quality (rerun, `--force`) | 997 s | 5/5 ok, all 3 tasks (≈ 200 s per numeric setting) |

Total ≈ 74 min of GPU time.

Measured (proves the code path only — 70 W card, 20 s windows, 1 rep):

| point | config | tok/s | p99 TPOT ms | p99 TTFT ms | notes |
|---|---|---|---|---|---|
| center | default | 102.5 | 71 | 2,168 | engine-reported `max_num_seqs` = 256 |
| center | prefix | 128.0 | 61 | 1,390 | prefix hit rate 0.43 |
| center | quant (FP8 W8A8) | 179.2 | 43 | 1,360 | |
| center | spec (ngram) | 151.2 | 47 | 1,350 | acceptance 0.97 (see caveat) |
| center | all-on | 416.2 | 21 | 462 | Value 4.06 |
| context_high | default | 76.8 | **104** | 4,364 | **Default fails the p99 TPOT gate** → point flagged |
| context_high | all-on | 245.9 | 34 | 935 | Value 3.20 |

Quality (20 samples per task, points): Default GSM8K 50 / HumanEval 85 / MMLU 42.5; agreement with Default outputs:
FP8 weights 0.40, FP8 KV 0.55, ngram spec 0.75, all three 0.35.

Caveats seen in the data (for the main run):
- **Window quantization.** With identical prompt/output lengths the closed loop runs in lockstep cohorts, so a short
  window counts whole cohorts (e.g. exactly 2,048 tokens = 32 × 64 in 20 s for several configurations). The 120 s
  main window reduces the granularity; consider it when reading small effects.
- **Synthetic text flatters n-gram speculation.** Prompts are random words from a small vocabulary and outputs
  (with `ignore_eos`) become repetitive, so ngram acceptance was ≈ 0.95. The main run uses EAGLE-3, but the
  workload source (spec §14: ShareGPT later) matters for speculative decoding.
- **Interactions** are "insufficient data" in smoke by design (no pairs are run); the full plan fits them.

## 3. Infeasible or failed configurations

No configuration was infeasible and no measured run failed in the smoke sequence.

Problems found and fixed during the smoke test:
1. **Quality, first pass:** GSM8K and HumanEval failed with HTTP 400 — vLLM 0.31 accepts at most 4 stop strings
   (`VLLM_MAX_STOP_STRINGS`, default 4) and HumanEval uses 5. Fix: servers start with `VLLM_MAX_STOP_STRINGS=16`
   (no effect on measured runs, which send no stop strings), each lm-eval task now runs in its own call, and a
   quality run with a missing task is `failed` (not `ok`). Rerun with `quality --force`: 5/5 ok.
2. **Before the smoke run** (probing single servers): FP8-KV configurations could not start — FlashInfer needed
   `nvcc` (see §4). These would have been recorded as infeasible rows with the error; fixed in the environment.
3. **Bandwidth estimate with speculative decoding:** the spec's steps/s ≈ tok/s ÷ running batch overcounts when a
   step emits several tokens; it now also divides by (1 + accepted/drafts) from `vllm:spec_decode_num_drafts_total`.
   (Smoke rows predate this metric; the binding label is unaffected because the peaks are null.)

## 4. Differences from the spec (flags, metric names, behaviour)

**vLLM flags (0.31.0).** Every flag in `harness/engine/vllm_flags.py` exists in `vllm serve --help=all`
(saved in `docs/vllm_serve_help_0.31.0.txt`; checked by `doctor` and before every GPU command).
- `--no-enable-chunked-prefill` is accepted and honoured (`/server_info` reports `enable_chunked_prefill=false`),
  with a warning: "This model does not officially support disabling chunked prefill". The off state also passes
  `--max-num-batched-tokens 16384` (spec fallback), so prompts are never split either way.
- The engine does **not log** the resolved `max_num_seqs`. The harness reads the effective scheduler/cache
  settings from `GET /server_info?config_format=json` (enabled by `VLLM_SERVER_DEV_MODE=1`) and records them in
  every row (`engine_info`). Default `max_num_seqs` reported on this machine: **256**.
- `VLLM_SERVER_DEV_MODE=1` also enables `POST /reset_prefix_cache`, used between repetitions.

**Metric names (0.31.0)** — all resolved through the fallback table in `harness/telemetry.py`:

| Wanted | Name found |
|---|---|
| KV-cache usage | `vllm:kv_cache_usage_perc` (old `vllm:gpu_cache_usage_perc` kept as fallback) |
| running / waiting | `vllm:num_requests_running`, `vllm:num_requests_waiting` |
| preemptions | `vllm:num_preemptions_total` |
| prefix-cache hits / queries | `vllm:prefix_cache_hits_total`, `vllm:prefix_cache_queries_total` |
| speculative draft / accepted | `vllm:spec_decode_num_draft_tokens_total`, `vllm:spec_decode_num_accepted_tokens_total` — **only exported when speculative decoding is on**; logged as missing (not an error) otherwise |
| queueing time | `vllm:request_queue_time_seconds` (histogram) |

**Installation issues found and fixed** (all in `requirements.txt` / `scripts/setup_env.sh`):
1. PyPI `torchcodec` is a CUDA 13 build (`libnvrtc.so.13`) and breaks `import vllm` with cu129 torch →
   pinned `torchcodec==0.16.0+cu129`.
2. vLLM's FlashInfer sampler and the FP8-KV attention path (FlashInfer XQA decode on sm_120) JIT-compile kernels
   with `nvcc`. vLLM treats FlashInfer as unavailable without `nvcc` or `flashinfer-cubin`, and the XQA kernel
   for FP8 KV + speculative decoding is not in the precompiled cache. Fix: `flashinfer-jit-cache` +
   `flashinfer-cubin` wheels, and a user-level CUDA 12.9 `nvcc` (conda-forge, via micromamba) that
   `setup_env.sh` installs into `.cuda-toolkit/` when no system `nvcc` exists. The launcher sets `CUDA_HOME`
   for vLLM servers only (`toolchain_env()`); it is not part of the engine flags. `doctor` checks for it.

## 5. What could not be tested on one GPU, and how it was covered

- **Optimization 7 (parallelism)** and all multi-GPU layouts (`tp4`, `tp2x2`, `tp1x4`, `tp2pp2`): n = 6 here
  (bit 7 disabled by the design module). Covered by:
  - unit tests (`tests/test_flags.py`): `CUDA_VISIBLE_DEVICES` and ports per replica for every layout, TP/PP flags;
  - the dry run (`run --profile dry`, env `main`, n = 7): the FakeEngine validates that each launch partitions
    the 4 GPUs with distinct ports, and the orchestrator/analysis run the full 512-cell plan including bit 7;
  - round-robin over replicas in the load driver is exercised by the dry run endpoints and the driver code path,
    but not against multiple real servers.
- **EAGLE-3** (main machine's speculative method): not run here (smoke uses `ngram` per spec). The repo exists
  (`doctor`); vLLM 0.31 accepts `method: eagle3`. First real load happens in the main start-up check.
- **Full-size regimes** (8,192-token prompts, concurrency 128) and 120 s windows: smoke uses the smaller levels.

## 6. Additions beyond the spec (spec §9 "keep it simple")

- `run --baseline` (only Default at the centre: RUNBOOK step 1) and `--retry-failed` (rerun failed/timeout keys).
- `scripts/run_detached.sh`: runs a command in a detached tmux session (SSH sessions on the test machine drop).
- Effective engine settings read from `/server_info`; prefix cache reset between repetitions.
- `doctor` checks free VRAM and names the processes holding it (the test GPU is shared), and checks for `nvcc`.
- `VLLM_MAX_STOP_STRINGS=16` for servers (quality tasks); speculative tokens-per-step in the bandwidth estimate.
- Model: ungated mirror instead of the gated meta-llama repo (team decision; identical weights).
- `fast` profile (team decision after the smoke test; RUNBOOK §2b): `optimizations` subset in the
  profile, `run.gpus_per_cell` with one worker process per GPU slot (`run_on_slots`; result appends are
  file-locked), per-task quality limits (`quality.task_limits`), quality settings run concurrently with
  agreement filled in afterwards, and a per-profile tuned file. With `gpus_per_cell: null` (the `full`
  profile) behaviour is unchanged. Covered by `tests/test_dry_run.py::test_fast_profile_*` (FakeEngine,
  4 slots); not run on real GPUs here (one GPU).
- Slurm support for the 2 × H100 node (RUNBOOK §2c): `configs/env/h100.yaml` with `gpu_ids: auto`
  (from `CUDA_VISIBLE_DEVICES`), `HARNESS_BASE_PORT` per job, a pre-start port check (busy port =
  retryable `failed` row, never another server's `/health`), `CUDA_DEVICE_ORDER=PCI_BUS_ID` for servers plus a
  `doctor` CUDA/NVML UUID check, a job deadline (`HARNESS_DEADLINE`) checked before each engine group and
  quality setting, `pipeline` and `status` commands, the repo CUDA 12.9 toolkit preferred over a system nvcc,
  and `scripts/slurm/{harness.sbatch,submit.sh,prefetch.sh}`. Tested: `tests/test_slurm.py`, a dry
  `pipeline` cut off by its deadline and then resumed to completion, and one real vLLM start on the test GPU
  with the new launch settings. Not tested here: Slurm itself (no scheduler on the test machine).
- Investigation profile (RUNBOOK §2d), added after the H100 fast-plan results: `investigations:` in a profile
  (cells shared between investigations run once; derived points use the base point's tuned values),
  `harness/deep_telemetry.py` (time series of NVML, raw vLLM /metrics, process CPU), profiler passes with
  vLLM's `--profiler-config` and `/start_profile`, exclusive cells, `quality.settings` and
  `quality.log_samples` with a paired McNemar comparison, the `investigate` command and
  `harness/analysis/investigate.py`. Tested with unit tests and a FakeEngine dry run only (no further GPU runs
  were made on the test machine); deep telemetry and profiler passes are first exercised on the cluster.
