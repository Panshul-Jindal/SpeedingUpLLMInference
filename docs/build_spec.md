# Harness Build Spec: LLM Serving Optimization Study

This document specifies what the harness must do, how it is verified, and what is out of scope. It was written before
the implementation; where the final study differs (five optimizations, a 2 × H100 node), see
`docs/Architecture_Decision_Records.md`.

---

## 1. Task

Build a complete, runnable benchmarking harness in Python for the study described in Section 2.

- **The code must be complete.** Every command runs end to end. The full experiment must be launchable later by changing configuration only, with no code changes.
- **Verification in this phase is limited** to:
  - unit tests;
  - a dry run with a fake engine;
  - short smoke tests on the test machine, which has one GPU.
- **Do not run the full experiment here.** It runs later on the main machine (four GPUs), after the code is copied there and the environment is rebuilt.

The definition of done is in Section 13.

---

## 2. The study in brief

The study measures how seven infrastructure-level LLM serving optimizations behave alone and in combination. It also measures under which workload conditions each one helps, and where each one stops helping.

**Fixed for all runs:**

- one serving engine: vLLM, at one pinned version;
- one model: Llama-3.1-8B-Instruct;
- one hardware setup per environment.

**Factorial design.**

- Each optimization is a switch: on or off.
- A configuration is a 7-bit vector, giving 127 combinations plus Default (all off), for 128 configurations in total.

**Workload regimes.** There are three regimes, each with Low, Mid and High levels:

- context length;
- concurrency;
- prompt reuse.

Exactly one regime varies at a time while the other two stay at Mid. This gives 7 workload points: the central Mid–Mid–Mid point plus 6 off-centre points.

**Value.**

- Value is the throughput speedup over Default at the same workload point.
- A speedup counts only if p99 latency and accuracy stay within bounds. Otherwise Value is recorded as 1.0 (no benefit), and the failing gate is recorded.

**Attribution.** For each workload point, the analysis computes:

- the main effects;
- the pairwise interaction ratios;
- Shapley shares of each final configuration's gain;
- the binding constraint of each final configuration.

---

## 3. Environments

| | Test environment (build and smoke test here) | Main environment (full run later) |
|---|---|---|
| GPUs | 1 × NVIDIA RTX PRO 4000 Blackwell | 4 × NVIDIA RTX 6000 Ada |
| VRAM | 24 GB | 48 GB each |
| Architecture | Blackwell, compute capability 12.0 (verify) | Ada Lovelace, compute capability 8.9 |
| FP8 compute | Yes | Yes |
| Interconnect | n/a | PCIe; no NVLink on RTX 6000 Ada (verify with `nvidia-smi topo -m`) |
| Role | Unit tests, dry run, smoke tests | Sweeps, start-up check, quality runs, full run |

**Requirements that follow from the hardware:**

- **Version pin.** Use the same vLLM and PyTorch versions in both environments. Blackwell (sm_120) needs a CUDA 12.8+ build of PyTorch and vLLM.
  - Pick a recent stable vLLM release whose wheels support both sm_89 and sm_120.
  - Pin it in `requirements.txt`.
  - Record the exact versions in every result row.
- **FP8 is supported in both environments.** An earlier design assumed an Ampere GPU and planned a W8A16 workaround. That assumption no longer holds. Use true FP8 (W8A8) for the quantized-weights "on" state and an FP8 KV cache.
- **Optimization 7 (parallelism) on one GPU.** It cannot be exercised on the test machine. The design module must therefore work with a variable number of active optimizations, n:
  - n = 6 on the test machine, with bit 7 disabled;
  - n = 7 on the main machine.
- **Hardware peaks.** Do not hard-code GPU peak memory bandwidth or FLOP/s.
  - Put them in the environment config as `null`, with a comment telling the user to fill them in from the vendor datasheet.
  - The roofline part of the analysis must report "unavailable" rather than fail when they are missing.

---

## 4. Design decisions that must not change

1. **Default really is "all off".**
   - Prefix caching is explicitly disabled.
   - Chunked prefill is disabled. If the pinned version cannot fully disable it, the off state is a token budget of 16,384, larger than any prompt, so prompts are never split.
   - The batch limit is the engine default. Record the actual value the engine reports at start-up.
   - Parallelism is tensor parallel across all GPUs.
2. **Value** = throughput speedup over Default, gated by p99 latency and accuracy, as in Section 2.
3. **Run plan:**
   - All 2^n configurations at the central point.
   - At each of the six off-centre points, a 2^(n−1) half-fraction: the configurations with an even number of optimizations switched on. This keeps Default in every design and separates every main effect and every two-way interaction.
   - On the main machine this gives 128 + 6 × 64 = **512** (configuration, point) cells.
   - Each cell is run **3 times** after a warm-up.
4. **Comparable runs.**
   - Every configuration at a workload point replays the same saved, seeded request trace.
   - Decoding is greedy (temperature 0).
   - Output length is fixed (`max_tokens` = `min_tokens`, `ignore_eos: true`).
5. **Quality runs once per numeric setting.** Only optimizations 1–3 change numerics, so 2^3 = 8 quality runs cover all configurations.
6. **Failures are results.** A configuration that cannot start (out of memory, unsupported combination, timeout) is recorded as an infeasible row with its error. The orchestrator continues with the next run and never crashes the experiment.

---

## 5. Optimization definitions

Flag names below are what recent vLLM versions use. **Verify every flag against `vllm serve --help` for the pinned version**, and keep all flag mapping in one file (`harness/engine/vllm_flags.py`) so it is easy to adjust.

| # | Optimization | Off (Default) | On | Changes numerics |
|---|---|---|---|---|
| 1 | Quantized weights | BF16 weights | FP8 W8A8 via `--quantization fp8` (online dynamic). Alternative, selectable in config: an INT4 AWQ checkpoint | Yes |
| 2 | Reduced-precision KV cache | `--kv-cache-dtype auto` (BF16) | `--kv-cache-dtype fp8` | Yes |
| 3 | Speculative decoding | none | `--speculative-config` JSON. Main: EAGLE-3 draft head for Llama-3.1-8B-Instruct. Smoke: `ngram` method, which needs no extra model | Yes (treated as numeric per design) |
| 4 | Prefix reuse | `--no-enable-prefix-caching` | `--enable-prefix-caching` | No |
| 5 | Tuned batching | Engine default `max_num_seqs` | `--max-num-seqs` set to the per-point value found by the sweep | No |
| 6 | Prefill scheduling | Chunked prefill off, or a 16,384-token budget if it cannot be disabled | Chunked prefill on, `--max-num-batched-tokens` set to the per-point value found by the sweep | No |
| 7 | Parallelism (needs ≥ 2 GPUs) | `--tensor-parallel-size` = number of GPUs | Best layout from the sweep: TP = 4, TP = 2 × 2 replicas, 4 replicas, or a pipeline-parallel layout | No |

**Notes for implementation:**

- **Data-parallel replicas.** Implement these as k independent `vllm serve` processes. Pin each to its GPUs with `CUDA_VISIBLE_DEVICES` and give each its own port. The load driver distributes requests round-robin across them.
  - Do not depend on vLLM's built-in data-parallel mode; it can be added as an option later.
- **Draft model for speculative decoding.**
  - The repository name of the EAGLE-3 head is a config value. Verify that it exists and that the pinned vLLM supports it.
  - If it cannot be loaded, fail that configuration with a clear error. Do not substitute another method silently.
- **Sweeps.** The "on" values for optimizations 5, 6 and 7 are found by sweeps (Section 7.3). Until a sweep has run, use placeholder values from config and mark the results as using untuned values.
- **Incompatible combinations.** Some combinations may be unsupported in the pinned version, for example speculative decoding with pipeline parallelism. Do not special-case them in code. They are found by the start-up check and recorded as infeasible.

---

## 6. Workload

### 6.1 Regime levels

| Regime | Main Low | Main Mid | Main High | Smoke Low | Smoke Mid | Smoke High |
|---|---|---|---|---|---|---|
| Context length (prompt tokens) | 256 | 2,048 | 8,192 | 256 | 1,024 | 2,048 |
| Concurrency (in-flight requests) | 2 | 32 | 128 | 2 | 8 | 16 |
| Prompt reuse (share of requests with a shared prefix) | 0% | 50% | 90% | 0% | 50% | 90% |

- **Main levels are tentative.** Keep them in `configs/experiment.yaml`, not in code.
- **Why the smoke levels are smaller.** On 24 GB, BF16 Llama-3.1-8B uses about 16 GB for weights. The KV cache costs about 128 KiB per token (32 layers × 8 KV heads × 128 dims × 2 for K and V × 2 bytes). That leaves room for roughly 30–40k tokens of KV. Use `max_model_len` = 4,096 for smoke runs.

### 6.2 Request traces

- **Prompt lengths.** Prompts have exact token lengths, measured with the model's own tokenizer. Build them from a seeded text source: synthetic text first, with an optional ShareGPT or long-document source later.
- **Prompt reuse.**
  - A fraction r of requests (the reuse level) start with a prefix drawn from a small pool. The pool size and the prefix share of the prompt are config values; defaults are 4 prefixes and a prefix of 75% of the prompt.
  - The remaining part of each prompt is unique.
- **Output length** is fixed. Defaults: 256 tokens for main runs and 64 for smoke runs.
- **Saving traces.** Each trace is saved to disk with its seed and a content hash. Every configuration at that workload point replays exactly the same trace.

### 6.3 Load generation

- **Closed-loop client.**
  - The client keeps exactly C requests in flight, where C is the concurrency level.
  - It uses async HTTP streaming against the OpenAI-compatible `/v1/completions` endpoint.
  - Each request's timestamps are taken client-side: send time, first token, every token, and completion.
- **Measurement window.**
  1. Run a warm-up and discard the warm-up requests.
  2. Measure over a fixed window (main: 120 s; smoke: 20 s; config values).
  3. Only requests that start and finish inside the window count.

---

## 7. Measurements, gates and analysis

### 7.1 Per-run metrics

- **Latency.** TTFT, TPOT (mean time between output tokens after the first) and end-to-end latency, each at P50 and P99.
- **Throughput.** Output tokens per second and requests per second over the window.
- **Telemetry**, sampled during the measurement window only:
  - GPU: utilization, memory used, power, SM clock (via NVML using `nvidia-ml-py`; DCGM is optional);
  - vLLM: scraped from each server's `/metrics` endpoint.
- **vLLM metric names.** These change between versions (for example, KV-cache usage was renamed in newer releases).
  - Discover the available names at runtime.
  - Map them through one table with fallbacks.
  - Log any expected metric that is missing instead of failing.
- **vLLM metrics wanted:**
  - KV-cache usage;
  - running and waiting requests;
  - preemptions;
  - prefix-cache hits and queries;
  - speculative draft and accepted tokens;
  - queueing time.

### 7.2 Gates and Value

Gates are config values. The current defaults:

- p99 TPOT ≤ 100 ms;
- p99 TTFT ≤ 5 s;
- accuracy drop ≤ 1 point compared with Default on each quality task.

Value is computed as follows:

- **All gates pass:** `Value = throughput / Default_throughput`, both at the same point.
- **Any gate fails:** `Value = 1.0`, and the failed gate is recorded.
- **Default fails a gate itself:** flag the point; do not silently pass it.

### 7.3 Sweeps

Each sweep is run on its own, at every workload point:

| Optimization | Swept value | Default candidates (config) |
|---|---|---|
| 5. Tuned batching | `max_num_seqs` | 16, 32, 64, 128, 256 |
| 6. Prefill scheduling | `max_num_batched_tokens` | 512, 1,024, 2,048, 4,096 |
| 7. Parallelism | Layout | Main machine only |

The best throughput that passes the latency gates is written back to `configs/tuned.yaml`. On the smoke profile, sweep two values per optimization only, to prove the code path.

### 7.4 Quality checker

- **Tool.** Use `lm-evaluation-harness`, pointed at the running server (`local-completions` model type).
- **Tasks.** GSM8K, an MMLU subset and HumanEval.
  - HumanEval executes generated code, so it needs `HF_ALLOW_CODE_EVAL=1` and `--confirm_run_unsafe_code`.
- **Loglikelihood tasks (MMLU).** If these do not work through the server, run them with lm-eval's offline `vllm` backend using the same engine arguments. Speculative decoding does not affect loglikelihood scoring.
- **Agreement score.** Also compute exact-match agreement with Default outputs on a fixed, versioned prompt set (`data/agreement_prompts_v1.jsonl`, 100 prompts).
- **Sample limits.** Main: full task subsets as configured. Smoke: `--limit 20` per task.

### 7.5 Analysis

All analysis is on log-throughput, per workload point.

- **Main effects and interactions.**
  - Fit `log T = β0 + Σ βi·xi + Σ βij·xi·xj` by OLS with statsmodels, using ±1 effect coding.
  - Report each main effect as a throughput multiplier with its standard error.
  - Report the higher-order residual and R².
- **Interaction ratio.** For each pair, `I_AB = (T_AB × T_0) / (T_A × T_B)`.
  - Label each value "measured" where all four configurations were run, otherwise "model".
- **Shapley shares** for each final configuration F, using `v(S) = log(T_S / T_0)`:
  - `φ_i = Σ over S ⊆ F\{i} of |S|!(|F|−|S|−1)!/|F|! · [v(S ∪ {i}) − v(S)]`
  - Shares are exact where all subsets were measured. Elsewhere, missing subsets are filled from the fitted model and the shares are labelled approximate.
- **Final configuration** per point: the configuration with the highest Value.
- **Binding constraint** of each final configuration. Assign one label using these first-order rules, with thresholds in config:
  - **KV capacity:** KV usage > 90%, or preemptions occurring.
  - **Memory bandwidth:** estimated achieved bandwidth > 70% of peak.
    - Per decode step, the bytes read are the weight bytes per GPU plus the KV bytes of the running requests.
    - Steps per second ≈ output tokens/s ÷ mean running batch.
  - **Compute:** estimated FLOP/s > 60% of peak, using FLOPs ≈ 2 × parameters × processed tokens/s.
  - **Scheduler/batch limit:** running requests equal to `max_num_seqs`, waiting requests above 0, and KV usage below 90%.
  - **Per-step overhead:** none of the above.
- **No-benefit list.** Each optimization whose main-effect gain is under 2% at a point, with a first-pass reason taken from telemetry. Example: prefix-cache hit rate near 0 at 0% reuse.
- **Partial data.** The analysis must run on partial data, such as a smoke run with 8 configurations. It fits what it can and marks everything else "insufficient data". It must never crash.

### 7.6 Report outputs

Write the following to `results/<profile>/report/`:

| File | Contents |
|---|---|
| `config_x_regime.csv` | Value for every configuration at every point, plus a final-configuration row |
| `value_per_config.csv` | Speedup, gate results and Value per run |
| `main_effects.csv` | Main effects with standard errors |
| `interactions.csv` | Interaction ratios, each labelled measured or model |
| `model_fit.csv` | R², residual and noise level per point |
| `shapley.csv` | Shapley shares |
| `final_configs.csv` | Final configuration and binding constraint per point |
| `no_benefit.csv` | No-benefit list with reasons |
| `summary.md` | Short human-readable summary |

---

## 8. Code architecture

```
harness/
  cli.py                  # entry point: `python -m harness <command>`
  config.py               # load + validate YAML (pydantic), merge experiment + env + profile
  optimizations.py        # 7-bit encoding, names, which bits are numeric, active-bit mask
  regimes.py              # workload points from regime levels
  design.py               # run plan: full factorial at centre, even-parity half-fractions elsewhere
  engine/
    base.py               # Engine protocol: start(config) -> endpoints, stop(), health()
    vllm_flags.py         # the ONLY place configuration -> vLLM CLI flags is defined
    vllm_server.py        # launches `vllm serve` processes, health checks, logs, teardown
    fake.py               # FakeEngine: synthetic latencies/throughput with planted effects
  workload.py             # trace generation, tokenizer-exact lengths, prefix pool, save/load
  load_driver.py          # closed-loop async client, warm-up, window, per-request timestamps
  telemetry.py            # NVML sampling + Prometheus scraping with metric-name discovery
  quality.py              # lm-eval runs + agreement check
  results.py              # append-only results store, resume by run key
  orchestrator.py         # walks the run plan, orders runs to minimize restarts
  analysis/
    value.py  effects.py  shapley.py  binding.py  report.py
  doctor.py               # environment checks
configs/
  experiment.yaml         # regimes, gates, run settings, sweep candidates
  env/test.yaml           # 1 GPU, model paths, peaks (null), max_model_len 4096
  env/main.yaml           # 4 GPUs, peaks (null), max_model_len 16384
  profiles/{dry,smoke,full}.yaml
  tuned.yaml              # written by sweeps
data/agreement_prompts_v1.jsonl
tests/                    # pytest, no GPU needed
scripts/setup_env.sh      # creates venv, installs pinned requirements
RUNBOOK.md                # how to run on the main machine
SMOKE_TEST_REPORT.md      # what was tested here, what was not
```

### 8.1 CLI commands

```
python -m harness doctor        --env test
python -m harness plan          --profile full --env main     # print run plan, do not run
python -m harness traces        --profile smoke --env test
python -m harness startup-check --profile smoke --env test     # one request per configuration
python -m harness sweep         --profile smoke --env test
python -m harness quality       --profile smoke --env test
python -m harness run           --profile smoke --env test
python -m harness analyze       --profile smoke
python -m harness smoke         --env test    # runs the whole smoke sequence above in order
```

All commands accept `--dry-run`, which uses `FakeEngine` and runs without a GPU.

### 8.2 Profiles

| Profile | Engine | Configurations | Points | Reps | Window | Quality |
|---|---|---|---|---|---|---|
| `dry` | FakeEngine | full plan for n = 6 and n = 7 | all 7 | 3 | instant | faked |
| `smoke` | vLLM, test machine | Default, each single optimization, all-on (8 with n = 6) | central, plus 1 off-centre point | 1 | 20 s | `--limit 20` |
| `full` | vLLM, main machine | 512-cell plan | all 7 | 3 | 120 s | full |

The smoke start-up check covers the same 8 configurations by default. `--all` runs all 64 configurations on the test machine, which takes about 1–2 hours. This is optional.

### 8.3 Results schema

Each run is one row, keyed by `(profile, env, config_bits, point_id, rep)`. Each row also records:

- versions: vLLM, torch, CUDA, driver, harness git hash;
- the exact engine flags;
- whether tuned or placeholder values were used;
- status: `ok`, `infeasible`, `failed` or `timeout`, with the error message;
- all metrics from Section 7.1, with the telemetry summaries.

Store raw per-request timings separately, one Parquet file per run. Each run also gets a directory `results/<profile>/runs/<run_key>/` with the server log, the flags, the trace hash and a metrics snapshot.

---

## 9. Implementation rules

- **Code style.** Python 3.11. Type hints throughout. Pydantic for config. No global state.
- **Process control.**
  - Start each vLLM server in its own process group.
  - Wait for `/health` with a timeout.
  - Capture stdout and stderr to the run directory.
  - On teardown, kill the whole group, then confirm through NVML that GPU memory has been released before the next start.
- **Fewer restarts.** Order runs at each point by engine flags, and restart the engine only when the flags change.
- **Resumable.** Re-running a command skips completed run keys. A `--force` flag reruns them.
- **Deterministic.** Seeds are in config and recorded. Traces are saved and hashed.
- **Version check at start-up.** Check that every flag used in `vllm_flags.py` appears in `vllm serve --help` for the installed version. If any is missing, fail early with a clear message.
- **No secrets in the repository.** Read the Hugging Face token from `HF_TOKEN`; Llama-3.1-8B-Instruct is a gated model.
- **Logging.** Structured logging to the console and to a file. Show a progress line per run: configuration, point, rep, status and throughput.
- **Keep it simple.** Do not add features beyond this spec without a note in `SMOKE_TEST_REPORT.md`.

---

## 10. Model checkpoints

Put these in `configs/env/*.yaml`, and verify each one exists before relying on it:

| Role | Default |
|---|---|
| Target model | `meta-llama/Llama-3.1-8B-Instruct` (gated) |
| Quantized weights, FP8 | None needed: online `--quantization fp8` from the BF16 weights |
| Quantized weights, INT4 alternative | An AWQ INT4 checkpoint of Llama-3.1-8B-Instruct (config value) |
| Speculative decoding, main | EAGLE-3 draft head for Llama-3.1-8B-Instruct (config value; check vLLM support) |
| Speculative decoding, smoke | `ngram` method, no extra model |
| Fast-iteration fallback | `Qwen/Qwen2.5-1.5B-Instruct` (ungated), for wiring tests only, never for results |

---

## 11. Testing

### 11.1 Unit tests

These run without a GPU (`pytest`):

- Configuration encoding: bits ↔ names, and the numeric-bit mask.
- Design:
  - the counts are 2^n and 2^(n−1);
  - every half-fraction configuration has an even number of optimizations switched on;
  - Default is present at every point;
  - main effects and two-way interactions are not aliased with each other.
- Flag mapping:
  - Default produces the explicit off flags (prefix caching off, chunked-prefill handling, TP = number of GPUs);
  - the replica layouts produce the correct `CUDA_VISIBLE_DEVICES` and ports.
- Workload:
  - prompt lengths are exact with a real tokenizer (use the small fallback tokenizer in CI);
  - the measured reuse fraction matches the target;
  - the same seed produces the same trace hash.
- Load driver against a local mock streaming server: TTFT, TPOT, percentiles and the window filtering.
- Results store resume.
- Analysis on synthetic data with planted main effects and two planted interactions:
  - the fit recovers them within tolerance;
  - Shapley shares sum to the total log gain.

### 11.2 Dry run

`python -m harness run --profile dry` followed by `analyze` completes end to end with `FakeEngine` and writes every file in Section 7.6. The `FakeEngine` must also exercise:

- infeasible configurations;
- gate failures;
- one injected crash, which the orchestrator must survive;
- a 4-GPU layout. This is the only place the parallelism code path is exercised before the main machine.

### 11.3 GPU smoke test

On the test machine, `python -m harness smoke --env test` runs these steps in order:

1. doctor;
2. traces;
3. start-up check;
4. sweep;
5. quality;
6. run;
7. analyze.

It should finish in roughly an hour. Every step must complete, and the report files must be produced from real measurements.

---

## 12. Handoff to the main machine

Write `RUNBOOK.md` with the exact steps for the main machine:

1. Clone the repository.
2. Run `scripts/setup_env.sh`.
3. Set `HF_TOKEN`.
4. Run `doctor --env main`.
5. Fill in the GPU peaks in `configs/env/main.yaml`.
6. Run, in order:
   1. Default at the central point (baseline and noise level);
   2. the sweeps;
   3. the full start-up check (128 configurations);
   4. the quality runs;
   5. `run --profile full`;
   6. `analyze`.

The runbook must also:

- give a time estimate per step, based on the smoke-test timings, scaled to the full plan;
- state the total GPU budget, about 100 node-hours;
- explain how to resume after an interruption.

---

## 13. Definition of done

- [ ] `pytest` passes on a machine without a GPU.
- [ ] `run --profile dry` and `analyze` complete and write every report file.
- [ ] `plan --profile full --env main` prints 512 cells, with 128 at the central point and 64 at each off-centre point, and n = 7.
- [ ] `doctor --env test` passes on the test machine, and reports the GPU, compute capability, CUDA version, vLLM version and FP8 support.
- [ ] `smoke --env test` completes on the test machine with real vLLM runs, and each of optimizations 1–6 is switched on in at least one run.
- [ ] `SMOKE_TEST_REPORT.md` covers:
  - what ran, with timings per step;
  - every infeasible or failed configuration and its error;
  - any vLLM flag or metric name that differed from this spec;
  - what could not be tested on one GPU (optimization 7 and the 4-GPU layouts) and how it was covered instead (FakeEngine).
- [ ] `RUNBOOK.md` is written for the main machine.

---

## 14. Out of scope for this phase

- Running the full plan, or drawing conclusions from smoke-test numbers.
- Choosing final latency targets or quality tolerances; keep them as config values.
- Simulator-based pruning (Vidur), SGLang cross-checks, joint-extreme workload points, and ShareGPT or long-document prompt sources. Leave clean extension points for these, but do not build them.
