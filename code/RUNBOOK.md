# RUNBOOK — full run on the main machine (4 × RTX 6000 Ada)

Everything is driven by configuration; no code changes are needed. All commands run from the repository
root with the venv active (`source .venv/bin/activate`). Every command is resumable (see §4).

## 1. Setup

```bash
git clone <repo-url> llm-serving-study && cd llm-serving-study      # or copy the directory
scripts/setup_env.sh               # uv + Python 3.11 venv + pinned cu129 wheels (vLLM 0.31.0, torch 2.13.0)
source .venv/bin/activate
export HF_TOKEN=...                # optional: configured models are ungated mirrors; needed only for gated repos
python -m harness doctor --env main
```

`doctor` must print `PASS`. Check in particular:
- 4 GPUs visible, compute capability 8.9, FP8 supported, ≥ 44 GiB free on each (`min_free_vram_gb`);
- `vllm 0.31.0+cu129` / `torch 2.13.0+cu129` (same as the test machine; driver 575.64 supports CUDA 12.9);
- "vllm flags: all flags present", all model repos exist;
- the interconnect line (`nvidia-smi topo -m`): expect PCIe, no NVLink.

Then **fill in the per-GPU peaks** in `configs/env/main.yaml` (`peaks.mem_bandwidth_gbps`, `bf16_tflops`,
`fp8_tflops`) from the RTX 6000 Ada datasheet. Without them the bandwidth/compute binding checks report
"unavailable" (everything else still works).

Optional before starting: review `configs/experiment.yaml` (regime levels are tentative; gates; window 120 s;
3 reps) and `configs/env/main.yaml` (`speculative.method: eagle3`, `quantization_on: fp8`).

## 2. Run order

| # | Step | Command | Estimate |
|---|------|---------|----------|
| 1 | Default at the centre (baseline, noise) | `python -m harness run --profile full --env main --baseline` | ≈ 10 min |
| 2 | Sweeps (opt. 5, 6, 7 at all 7 points) | `python -m harness sweep --profile full --env main` | ≈ 6 h |
| 3 | Full start-up check (128 configurations) | `python -m harness startup-check --profile full --env main` | ≈ 4 h |
| 4 | Quality runs (8 numeric settings) | `python -m harness quality --profile full --env main` | ≈ 4–6 h |
| 5 | Full plan (512 cells × 3 reps) | `python -m harness run --profile full --env main` | ≈ 80 h |
| 6 | Analysis | `python -m harness analyze --profile full` | minutes |

Notes:
- Step 1 writes the Default/centre rows of the full plan; step 5 skips them (resume), so nothing is run twice.
  Check `results/full/report/model_fit.csv` → `noise_sd_log` after step 5 (or look at the 3 baseline rows).
- Step 2 writes `configs/tuned.yaml`. Runs before the sweep would use the placeholders in
  `configs/experiment.yaml` and are marked `used_placeholder: true`; run the sweep first.
  If no candidate passes the latency gates at a point, the placeholder stays (logged as a warning).
- Step 3 records configurations that cannot start (OOM, unsupported combinations such as speculative decoding
  with pipeline parallelism) as `infeasible` in `results/full/startup.jsonl`. The run (step 5) will record them
  as infeasible rows again; that is expected.
- Step 4 needs `HF_ALLOW_CODE_EVAL=1` (set automatically) and executes model-generated code (HumanEval).
- Progress lines look like `[run] center 1011000 quant+spec+prefix rep=2 ok 1834.2 tok/s`.

**Total GPU budget: ≈ 95 node-hours** (the study plan budgets about 100 node-hours).

How the estimates were derived (from smoke timings, scaled to the full plan). Run time is dominated by fixed
windows, so it barely depends on how fast the GPUs are:
- smoke engine start 38–105 s (torch.compile cache warm after the first start); assume ≈ 90 s on the main machine
  (TP=4 over PCIe, larger KV allocation), + ≈ 10 s teardown.
- **run:** each cell = one engine start + 3 reps × (30 s warm-up + 120 s window + prefix-cache reset)
  ≈ 100 s + 3 × 155 s ≈ 9.4 min; 512 cells ≈ 80 h.
- **sweep:** 7 points × 13 candidates (5 + 4 + 4 layouts), 1 rep each ≈ 91 × 4.2 min ≈ 6.4 h.
- **startup-check:** smoke ≈ 70 s per configuration incl. probe and teardown; 128 × ≈ 1.75 min ≈ 3.7 h.
- **quality:** smoke ≈ 200 s per numeric setting at 20 samples per task; full GSM8K (1,319), HumanEval (164) and the
  8-subject MMLU subset are ~50× more samples but run batched on 4 Ada GPUs: budget 30–45 min per setting × 8.
- The first start of each new configuration compiles kernels (torch.compile, FlashInfer JIT); later starts reuse
  `~/.cache/vllm` and `~/.cache/flashinfer`.

## 2b. Fast plan (current choice): `--profile fast`

A reduced plan agreed as a deviation from the spec, to finish in under a working day. Same commands as §2
with `--profile fast` (results in `results/fast/`, tuned values in `configs/tuned_fast.yaml`):

- **5 optimizations:** quant, kvq, spec + prefix, batch. Chunked prefill and parallelism stay off.
  n = 5 gives 32 configurations at the centre + 6 × 16 = **128 cells** × 3 reps = 384 runs.
- **One GPU per cell, 4 cells at once** (`run.gpus_per_cell: 1`). Default is TP = 1 on one GPU, not TP = 4.
  Each GPU is a slot with its own worker process, engine and ports (8100, 8110, 8120, 8130); all four
  GPUs must be free. Progress lines carry a `[gpuN]` tag.
- **Quality:** GSM8K limited to 500 samples, full HumanEval, 4 MMLU subjects, agreement kept. Evals run only
  for the 8 settings of quant/kvq/spec (prefix and batch do not change outputs), 4 at once.

| # | Step | Estimate (wall clock) |
|---|------|----------|
| 1 | `run --profile fast --env main --baseline` | ≈ 10 min |
| 2 | `sweep --profile fast --env main` (batch only: 7 points × 5 values) | ≈ 40 min |
| 3 | `startup-check --profile fast --env main` (32 configurations) | ≈ 15 min |
| 4 | `quality --profile fast --env main` | ≈ 1 h |
| 5 | `run --profile fast --env main` | ≈ 5–6 h |
| 6 | `analyze --profile fast` | minutes |

**Total ≈ 7–8 h wall clock (≈ 30 GPU-hours).** Things to keep in mind:
- On one 48 GB GPU the KV cache is about a quarter of the TP = 4 one, so `context_high` and `concurrency_high`
  will be more KV-bound than in the full plan. That is a real effect of the cell size, not a bug.
- Four engines and four load drivers share the host CPU. If `doctor` shows few cores, or per-step overhead
  dominates the binding labels, compare against a few runs with `gpus_per_cell: 4`.
- The multi-slot path was tested with the FakeEngine only (the test machine has one GPU).

## 2c. Slurm node with 2 × H100 (current setup): `--env h100`, jobs ≤ 8 h

The fast plan (§2b) runs on a 2 × H100 80 GB Slurm node as a **chain of resumable jobs**. Each job runs
`python -m harness pipeline` (doctor → traces → baseline → sweep → startup-check → quality → run →
analyze): finished keys are skipped, failed keys are retried once, and nothing is started that would not end
before the job's end time minus 15 min (`HARNESS_DEADLINE`, set from `squeue`). A cut-off job loses nothing.

```bash
# once, on the login node, from the repo root
scripts/setup_env.sh                 # uv venv + pinned cu129 wheels + CUDA 12.9 nvcc in .cuda-toolkit
scripts/slurm/prefetch.sh            # models + lm-eval datasets into HF_HOME (default ~/gemma/hf-cache)
# edit configs/env/h100.yaml: peaks for your H100 variant (doctor prints the GPU name)

scripts/slurm/submit.sh 3            # 3 chained jobs (afterany); the last exits at once if all is done
python -m harness status --profile fast --env h100      # progress, any time, from the login node
```

- **Job settings** (`scripts/slurm/harness.sbatch`) follow `run_qwen_slurm.sh`: partition `gpu`, node
  `ccs-srv003`, 16 CPUs, 128 GB, 8 h. Uncomment `--gres=gpu:2` if the cluster schedules GPUs as gres.
  Overrides via the environment: `PROFILE`, `ENV_NAME`, `MARGIN_MIN`, `HF_HOME`, `HF_HUB_OFFLINE=1`.
- **Do not load the `cuda-13.1` module.** The wheels are CUDA 12.9 builds and FlashInfer's JIT uses the
  CUDA 12.9 toolkit in `.cuda-toolkit` (the harness prefers it over any system `nvcc`). The conda envs from
  the old script are not used; everything runs from `.venv`.
- **GPUs:** `gpu_ids: auto` takes the GPUs Slurm allocates (`CUDA_VISIBLE_DEVICES`); with
  `CUDA_DEVICE_ORDER=PCI_BUS_ID`, `doctor` checks that each index names the same GPU for CUDA and NVML.
  One cell per GPU → 2 cells at once.
- **Ports:** each job uses its own range (`HARNESS_BASE_PORT` = 20000 + 10 × (job id mod 3000)). A port that is
  already taken is never used: the run is recorded `failed` (port_in_use) and retried by the next job.
- **Exit codes:** 0 = all done, 10 = work left (the job still reports success), 3 = GPU memory not released
  (check `nvidia-smi`, then resubmit), 1 = doctor failed (e.g. GPUs busy) — the next chained job tries again.
- **Logs:** `logs/slurm-<jobid>.out|err`, per-step timings in `results/fast/pipeline/timings_<jobid>.json`,
  structured logs in `results/fast/logs/`.

| Step | Estimate on 2 × H100 (2 slots) |
|---|---|
| baseline | ≈ 10 min |
| sweep (batch: 7 points × 5 values) | ≈ 1.25 h |
| startup-check (32 configurations) | ≈ 25 min |
| quality (8 settings, 2 at once) | ≈ 1.5 h |
| run (128 cells × 3 reps) | ≈ 10 h |
| **total** | **≈ 13.5 h → 2 jobs; submit 3 for slack** |

Estimates come from the smoke timings (engine start 40–105 s, fixed 30 s warm-up + 120 s window per rep);
the run is dominated by the fixed windows, so faster GPUs shorten it little. To shorten it, lower
`run.window_s` / `run.reps` in `configs/experiment.yaml` (weaker noise estimate).

## 2d. Investigation re-run: `--profile investigate` (after the fast plan)

Re-runs only the anomalous cells of the fast-plan results, with deep telemetry, profiler passes for selected
cells and a paired accuracy test. Defined in `configs/profiles/investigate.yaml` (each investigation says
why it exists). One Slurm job:

```bash
python -m harness plan --profile investigate --env h100 -v      # cells + estimate, no GPU
PROFILE=investigate scripts/slurm/submit.sh 1                    # doctor -> investigate -> quality -> analyze
python -m harness analyze --profile investigate                  # results/investigate/report/investigation.md
```

| Investigation | Cells | Question |
|---|---|---|
| spec_prefix_no_reuse | 4 reuse levels × 6 configs | Why spec+prefix is slower than Default at 0% reuse |
| spec_long_context | 4 context lengths × 2 | Where EAGLE-3 acceptance collapses (0.78 → 0.11 at 8192 tokens) |
| high_load_dealias | 2 points × 11 | Measured (not modelled) batch and prefix interactions |
| kvq_spec_outlier (+ _alone) | 1 cell × 5 reps, shared and alone | Is the 2716 vs 1770 tok/s rep CPU contention between slots? |
| binding_check | 3 points × 2, alone, profiled | Is "per-step overhead" right (GPU-busy fraction, EngineCore CPU)? |

- **Deep telemetry** (every measured run, `harness/deep_telemetry.py`): NVML every 0.25 s (SM and memory
  utilisation, clocks, power, temperature, throttle reasons), every server's `/metrics` every 1 s raw
  (histogram buckets included), CPU% of the API server / EngineCore / host every 1 s. Series in
  `results/investigate/runs/<key>/telemetry_*.parquet`; summaries (`deep_*`) in each row.
- **Profiler passes**: a separate engine start with vLLM's `--profiler-config`; under the same load,
  `/start_profile` records 64 engine steps. Traces in `results/investigate/profiles/` (open in Perfetto);
  their throughput is not a result.
- **Exclusive cells** run after the shared ones on one GPU with the other idle.
- **Quality**: settings 000/100/010/110/001, full GSM8K and 8 MMLU subjects, `--log_samples`; the report
  gives a paired difference with 95% CI and a McNemar test per task.

**Estimate on 2 × H100 NVL: ≈ 5.2 h wall clock (10 GPU-hours)**: traces 5 min, shared cells 3.25 h,
exclusive cells 1.2 h, quality 0.6 h. From the fast run: engine start 55 s median (70 s budgeted),
30 s warm-up + 120 s window per rep, profiler pass ≈ 3 min. One 8 h job; resubmit if it stops early.

## 3. Outputs

- `results/full/runs.jsonl` — one row per run (key: profile, env, config bits, point, rep), with versions,
  exact flags, tuned/placeholder marker, status + error, latency/throughput, telemetry summaries.
- `results/full/runs/<run_key>/` — `requests.parquet` (per-request timings), `flags.json`, `trace_hash.txt`,
  `metrics_snapshot.txt`, symlinks to the server logs (`results/full/engine_logs/`).
- `results/full/{sweep,startup,quality}.jsonl`, `results/full/quality/<numeric bits>/` (lm-eval output).
- `results/full/traces/` — the saved, hashed traces (one per point; every configuration replays the same one).
- `results/full/report/` — the nine report files; start with `summary.md`.
- `results/full/logs/` — structured JSON logs per command.

## 4. Interruptions and resume

- Every command appends one row per finished unit and skips keys already present. After a crash, reboot or
  Ctrl-C, **re-run the same command**; it continues where it stopped.
- `--retry-failed` reruns keys whose status is `failed`/`timeout` (e.g. after fixing a transient problem);
  `--force` reruns everything for that command.
- If a command exits with `ENVIRONMENT FAULT` (GPU memory not released after a server stopped), check
  `nvidia-smi` for leftover `vllm`/`EngineCore` processes, kill them, and re-run the command.
- Leftover servers after a hard kill: `pkill -f "vllm serve"`; then confirm with `nvidia-smi` that memory is free.
- Traces are regenerated only if their settings change; otherwise the saved file (verified by hash) is reused.

## 5. Known differences from the test machine

- Optimization 7 (parallelism) and the replica layouts (`tp2x2`, `tp1x4`, `tp2pp2`) were only exercised with
  the FakeEngine on the test machine (see SMOKE_TEST_REPORT.md). Watch the first `par` runs: each replica is a
  separate `vllm serve` with its own `CUDA_VISIBLE_DEVICES` and port (8100+).
- EAGLE-3 (`yuhuili/EAGLE3-LLaMA3.1-Instruct-8B`) is used on the main machine; the smoke test used `ngram`.
  Run `startup-check` first: if the head fails to load, every configuration with bit 3 on is recorded as
  infeasible with the error. Switch the repo in `configs/env/main.yaml` (an alternative is listed there) and
  re-run `startup-check --force`. Infeasible rows count as completed, so they are never retried
  automatically; `--force` reruns them.
