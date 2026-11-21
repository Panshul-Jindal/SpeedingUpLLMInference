# Architecture Decision Records — Project 14, Group 2

**Project:** Combining infrastructure-level optimizations for LLM serving (CS5903, IIT Hyderabad)
**Members:** Panshul Jindal, Mohd Shozab
**Location in repository:** `docs/Architecture_Decision_Records.md`

**About this log.** Written up on **21 November 2026**, after the experiments. Every record is reconstructed from
sources that are in this repository or were submitted to the course, and cites them:

- course documents: Assignment 1 (problem statement), Assignment 2 (proposed approach and system design), the
  progress report;
- the harness source (`harness/`, `configs/`, `scripts/`, `cluster/`, `tests/`) and the build spec `docs/build_spec.md`;
- the commit history: `6ebfa6d`, `67db3a6`, `a8454d5`, `bba6cee` (29 Oct), `3526c58` (2 Nov), `c42acca` (5 Nov);
  the investigation update (`configs/profiles/investigate.yaml`, `harness/deep_telemetry.py`,
  `harness/analysis/investigate.py`) came after `c42acca`;
- `RUNBOOK.md`, `SMOKE_TEST_REPORT.md`;
- the report pack `evaluation/report_pack/` (data exported from the run logs; `FINDINGS.md`, `METHODOLOGY.md`,
  `LIMITATIONS.md`).

Dates are those of the commit or data that shows a decision in effect; where only a range is known, a range is
given. Nothing is back-dated. Superseded or insufficient records are kept.

**Citation shorthand.** `data/x.csv` = `evaluation/report_pack/data/x.csv`. F1–F10 / C1–C17 = findings and claims
in the report pack. `S5.2` etc. = checks printed by `python evaluation/check_report_numbers.py` (ADR-020). M1–M8 =
milestones in the report (§6).

| ADR | Title | In effect | Status |
|---|---|---|---|
| [000](#adr-000) | Methodology record (9 October 2026): factorial study on one engine | by 9 Oct | Reaffirmed; amended by 007, 010, 012, 016 |
| [001](#adr-001) | Serving engine: vLLM, pinned to 0.31.0 (cu129) | 29 Oct | Accepted |
| [002](#adr-002) | Model checkpoint: ungated mirror of Llama-3.1-8B-Instruct | 29 Oct | Accepted |
| [003](#adr-003) | Default baseline is genuinely "all off" and verified from the engine | 29 Oct | Accepted |
| [004](#adr-004) | Value: speedup over Default, gated by p99 latency and accuracy | 29 Oct | Accepted; accuracy part refined by 018 |
| [005](#adr-005) | Workload: synthetic token-exact traces and an own closed-loop client | 29 Oct | Accepted |
| [006](#adr-006) | Design: full factorial at the centre, even-parity half-fractions elsewhere | 29 Oct | Accepted; factors reduced by 012 |
| [007](#adr-007) | Quantized weights: FP8 W8A8 instead of W8A16 | 29 Oct | Accepted |
| [008](#adr-008) | Speculative decoding: EAGLE-3 for results, n-gram only for smoke tests | 29 Oct | Accepted |
| [009](#adr-009) | Quality: lm-eval against the running server, once per numeric setting | 29 Oct | Accepted; refined by 018 |
| [010](#adr-010) | Telemetry: NVML + vLLM metrics (+ profiler passes) instead of DCGM/Nsight | 29 Oct | Accepted |
| [011](#adr-011) | Append-only results store; failures are results; resumable runs | 29 Oct / 5 Nov | Accepted |
| [012](#adr-012) | Scope: measure 5 of the 7 optimizations; chunked prefill and parallelism dropped | 2 Nov | Accepted |
| [013](#adr-013) | One configuration per GPU; cells on different GPUs measured concurrently | 2 Nov | Accepted |
| [014](#adr-014) | Execution platform: 2 × H100 NVL Slurm node, chained resumable jobs | 5 Nov | Accepted |
| [015](#adr-015) | GPU sharing on the Slurm node: pre-flight VRAM check only | 5 Nov | Insufficient (see 017); replacement open |
| [016](#adr-016) | Phase 2: focused re-runs of anomalies with deep telemetry | by 11 Nov | Accepted |
| [017](#adr-017) | Contaminated runs: replace or exclude, never rescale | 11–14 Nov | Accepted |
| [018](#adr-018) | Accuracy gate: paired re-test and three reported policies | 11–14 Nov | Accepted |
| [019](#adr-019) | Binding constraint: profiler traces override the first-order rule | 11–14 Nov | Accepted |
| [020](#adr-020) | GPU-free check of every report number against the exported data | 21 Nov | Accepted |

---

<a id="adr-000"></a>
## ADR-000 — Methodology record (9 October 2026): factorial study on one engine

**Date, status & scope.** Methodology at the 9 October 2026, 11:59 PM IST checkpoint: **reaffirmed**, not changed.
Documented 21 Nov 2026. Scope: the whole study. Later changes, each with its own record: FP8 W8A8 instead of W8A16
(ADR-007), telemetry stack (ADR-010), 5 instead of 7 optimizations (ADR-012), a second measurement phase (ADR-016).

**Context & decision drivers.** Assignment 1: improve the latency and throughput of a served open-weight model by
combining infrastructure-level optimizations, and establish which contribute, in which regimes (context length,
concurrency, prompt reuse) and where they stop helping. Published speedups use different baselines and interact
(Assignment 2, "Why this approach"), so they cannot be combined. The method must measure every optimization against
one baseline, expose interactions, and fit about 100 GPU-node-hours (Assignment 2, §C).

**Options considered.** (1) Combine published speedups: no common baseline. (2) One-factor-at-a-time ablation:
blind to interactions. (3) Factorial measurement on one engine, model and hardware set-up, varying one regime at a
time.

**Decision & rationale.** Option 3, as in Assignment 2 §A: seven on/off optimizations (127 combinations + Default),
one pinned vLLM version, an ~8B instruct model, seven workload points, Value = gated throughput speedup, attribution
by main effects, pairwise interaction ratios and Shapley shares, binding constraint from telemetry.

**Consequences & trade-offs.** A harness had to be built before any result existed. Results are specific to one
model, engine version and GPU. One-at-a-time regimes miss joint extremes (Assignment 2 §C.2); they were not added.

**Evidence & validation.** Assignment 1; Assignment 2 §A, §B, §D; progress report §3–§6; `docs/build_spec.md` §2, which carries
the same design into the build spec.

**Revisit trigger.** A change to the optimization set, model, engine or regime definitions (triggered: ADR-012).

---

<a id="adr-001"></a>
## ADR-001 — Serving engine: vLLM, pinned to 0.31.0 (cu129)

**Date, status & scope.** Engine chosen in Assignment 2 §C.1; version pinned 29 Oct 2026 (`6ebfa6d`,
`requirements.in`). Accepted. Scope: engine layer, all measurements.

**Context & decision drivers.** Each optimization must be switchable by configuration alone, with the same numerics
on the test GPU (Blackwell, sm_120) and the main GPUs, and the engine must export KV-cache use, prefix-cache hits and
speculative acceptance.

**Options considered.** vLLM (all seven optimizations are flags; OpenAI-compatible API; Prometheus `/metrics`;
EAGLE-3); SGLang (similar features; Assignment 2 kept it only as an optional prefix-reuse cross-check); TensorRT-LLM
(quantization and speculation are engine-build choices, so each configuration needs a build).

**Decision & rationale.** vLLM 0.31.0+cu129 with PyTorch 2.13.0+cu129, whose wheels cover sm_89, sm_90 and sm_120.
All flag mapping is in `harness/engine/vllm_flags.py`; `harness/doctor.py` (`check_flags`) runs `vllm serve
--help=all` and fails if a mapped flag is missing. The 0.31.0 help text is saved in `docs/vllm_serve_help_0.31.0.txt`.

**Consequences & trade-offs.** Results hold for vLLM 0.31 only; the SGLang cross-check was not done. CUDA 12.9 wheels
need a matching driver; FlashInfer's JIT kernels need `nvcc`, so `scripts/setup_env.sh` installs a user-level CUDA
12.9 toolkit, and `torchcodec` is pinned to its cu129 build (`SMOKE_TEST_REPORT.md` §4).

**Evidence & validation.** `vllm_version` = `0.31.0+cu129`, `torch_version` = `2.13.0+cu129` in all 510 rows of
`data/runs_fast_plan_all.csv` and `data/runs_investigation.csv`; smoke test 29 Oct: all flags present, 8/8
configurations start (`SMOKE_TEST_REPORT.md` §2).

**Revisit trigger.** A required flag is missing or behaves differently in another vLLM version.

---

<a id="adr-002"></a>
## ADR-002 — Model checkpoint: ungated mirror of Llama-3.1-8B-Instruct

**Date, status & scope.** 29 Oct 2026 (`6ebfa6d`, `models.target` in `configs/env/*.yaml`). Accepted. Scope: every run
and quality check.

**Context & decision drivers.** The official repository `meta-llama/Llama-3.1-8B-Instruct` is gated, so every
machine would need a personal Hugging Face token. The EAGLE-3 head (ADR-008) is trained for this model, so another
model family would also break speculative decoding.

**Options considered.** (1) The gated repository with `HF_TOKEN`; (2) the ungated mirror
`unsloth/Llama-3.1-8B-Instruct`; (3) another ungated model (`Qwen/Qwen2.5-1.5B-Instruct`, kept only as a
wiring-test `fallback`).

**Decision & rationale.** Option 2 (`SMOKE_TEST_REPORT.md` §6): same architecture and tokenizer, no tokens on shared
machines (`HF_TOKEN` is still used if set). `doctor` compares the mirror's `config.json` (layers, KV heads, head
dimension) with `model_info`, which the analysis uses.

**Consequences & trade-offs.** Provenance goes through a third-party mirror; checksums were not compared with the
official weights. The Llama 3.1 Community License applies unchanged.

**Evidence & validation.** `configs/env/h100.yaml`; `harness/doctor.py` (model and `model_info` checks).

**Revisit trigger.** A checksum mismatch, the mirror being withdrawn, or a requirement to use the official repository.

---

<a id="adr-003"></a>
## ADR-003 — Default baseline is genuinely "all off" and verified from the engine

**Date, status & scope.** 29 Oct 2026 (`6ebfa6d`: `vllm_flags.py`; `a8454d5`: settings read from `/server_info`).
Accepted. Scope: every Value.

**Context & decision drivers.** Assignment 2 §A: Default must also disable what vLLM enables on its own (prefix
caching, chunked prefill); otherwise every gain is understated. Some versions cannot fully disable chunked prefill
(Assignment 2 §C.2).

**Options considered.** (1) vLLM's shipped defaults; (2) explicit all-off with the engine's own batch default;
(3) all-off with a hand-picked batch limit.

**Decision & rationale.** Option 2: BF16 weights and KV cache, no speculation, `--no-enable-prefix-caching`,
`--no-enable-chunked-prefill` plus `--max-num-batched-tokens 16384` (larger than any prompt; `chunked_prefill_off`
in `configs/experiment.yaml`), engine-default `max_num_seqs`, and tensor parallelism across all GPUs of a cell
(`default_layout` in `vllm_flags.py`; one GPU per cell after ADR-013, so TP = 1). The engine does not log its resolved
settings, so the harness reads them from `GET /server_info` and stores them in every row.

**Consequences & trade-offs.** Gains are larger than against vLLM's shipped defaults; the report says so. Because the
default `max_num_seqs` (256) exceeds every in-flight count measured, tuned batching never binds (F8, C14).

**Evidence & validation.** All 45 Default runs: engine-reported prefix caching and chunked prefill off, and
`max_num_seqs_effective` = 256; launch flags stored per row (check `S2-base`).

**Revisit trigger.** A vLLM upgrade changes defaults, or `/server_info` reports an optimization on in a Default run.

---

<a id="adr-004"></a>
## ADR-004 — Value: speedup over Default, gated by p99 latency and accuracy

**Date, status & scope.** Defined in Assignment 2 §A; thresholds fixed 29 Oct 2026 (`6ebfa6d`: `gates` in
`configs/experiment.yaml`, `harness/analysis/value.py`). Accepted; accuracy part refined by ADR-018. Scope: ranking of
configurations (M5).

**Context & decision drivers.** Assignment 1 asks for one Value per configuration and regime. Raw throughput would
reward configurations that are fast in aggregate but slow per user or less accurate. Assignment 2 left the bounds open.

**Options considered.** Raw throughput; goodput at an SLO; throughput speedup with hard gates; a weighted score.

**Decision & rationale.** Value = output-token throughput ÷ Default throughput at the same point, counted only if p99
TPOT ≤ 100 ms, p99 TTFT ≤ 5 s and the accuracy drop ≤ 1 point per task; otherwise 1.0 with the failing gate recorded.
If Default itself fails a gate, the point is flagged (`docs/build_spec.md` §7.2). Hard gates are easy to audit.

**Consequences & trade-offs.** A configuration just over a gate scores 1.0, a cliff rather than a slope. In the study
no cell failed a latency gate; all 86 gate failures were accuracy failures under the original rule.

**Evidence & validation.** `data/cells_combined_clean.csv` (`latency_gates_pass`, `failed_gates`); check
`S5.1-gates`. The Default-fails path was exercised in the smoke test (2,048-token prompts on the 70 W test GPU,
`SMOKE_TEST_REPORT.md` §2).

**Revisit trigger.** An application with a stated SLO, or Default failing a gate on the main hardware.

---

<a id="adr-005"></a>
## ADR-005 — Workload: synthetic token-exact traces and an own closed-loop client

**Date, status & scope.** 29 Oct 2026 (`6ebfa6d`: `harness/workload.py`, `harness/load_driver.py`). Accepted. Scope:
all measured runs.

**Context & decision drivers.** Configurations must see identical requests, and each regime must be set exactly.
Assignment 2 §C.1 planned vLLM's serving benchmark plus an async client, with synthetic prompts, ShareGPT and long
documents.

**Options considered.** Sources: ShareGPT; long documents; seeded synthetic text with exact token lengths and a pooled
prefix. Load: vLLM's serving benchmark (open-loop request rate) or an own closed-loop client.

**Decision & rationale.** Synthetic text and an own client. Prompts are random words from a fixed list sent as token
IDs, so the server sees exactly the requested length; 4 pooled prefixes make up 75% of a reusing prompt; outputs are
256 tokens (`ignore_eos`, greedy). One trace per point is saved with seed 20261029 and a content hash and replayed by
every configuration. The client keeps exactly C requests in flight over streaming `/v1/completions`, which makes
concurrency a controlled variable and gives per-token timestamps; only requests that start and finish in the 120 s
window count. Real-text sources remain an extension point (`workload.source`).

**Consequences & trade-offs.** Synthetic text affects speculative acceptance and prefix hit rates, so absolute `spec`
and `prefix` numbers may not transfer (`LIMITATIONS.md`). Equal lengths make the closed loop move in cohorts, which
makes short windows coarse (`SMOKE_TEST_REPORT.md` §2).

**Evidence & validation.** One trace hash per point, 0 request errors and 0 wrong output lengths in 510 runs
(`S4-trace`); `tests/test_workload.py`, `tests/test_load_driver.py`.

**Revisit trigger.** A real-traffic run whose main effects differ from the synthetic ones by more than 10%.

---

<a id="adr-006"></a>
## ADR-006 — Design: full factorial at the centre, even-parity half-fractions elsewhere

**Date, status & scope.** Planned in Assignment 2 §A; implemented 29 Oct 2026 (`harness/design.py`,
`harness/analysis/effects.py`, `harness/analysis/shapley.py`). Accepted; factor count reduced by ADR-012. Scope: run
plan (M4), attribution (M7).

**Context & decision drivers.** With 7 optimizations, a full factorial at all 7 points is 7 × 128 = 896 cells, each
run 3 times.

**Options considered.** One-factor-at-a-time (8 cells per point); full factorial everywhere (896 cells); full
factorial at the centre and a half-fraction at the other six points (128 + 6 × 64 = 512 cells).

**Decision & rationale.** All 2ⁿ configurations at the centre, so Shapley shares there are exact, and the 2ⁿ⁻¹
even-parity half-fraction elsewhere. That fraction keeps Default in every design and leaves every main effect and
two-way interaction estimable. The module works for any n. Analysis is OLS on log throughput with ±1 coding; terms the
data cannot support are reported as "insufficient data".

**Consequences & trade-offs.** Off-centre interactions are model estimates unless measured directly (phase 2 measured
the important ones, ADR-016); off-centre Shapley shares are approximate.

**Evidence & validation.** `tests/test_design.py`, `tests/test_analysis.py`; R² ≥ 0.9976 at every point
(`S5.3-fit`); Shapley exact at 3 of 7 points (`S5.3-shap`).

**Revisit trigger.** A model-estimated interaction drives a recommendation without a direct measurement.

---

<a id="adr-007"></a>
## ADR-007 — Quantized weights: FP8 W8A8 instead of W8A16

**Date, status & scope.** 29 Oct 2026 (`quantization_on: fp8` in `configs/env/*.yaml`). Accepted; supersedes the
W8A16 plan of Assignment 2 §C.2 and the progress report's risk table. Scope: optimization 1 (`quant`).

**Context & decision drivers.** Assignment 2 assumed an Ampere GPU without FP8 compute and planned vLLM's W8A16 mode.
The test GPU (Blackwell) and the main-run GPUs (Ada, later Hopper) support FP8 compute (`docs/build_spec.md` §3).

**Options considered.** W8A16 (memory saving only); FP8 W8A8 via `--quantization fp8` (online, no extra checkpoint);
INT4 AWQ (larger saving, larger accuracy risk, separate checkpoint).

**Decision & rationale.** FP8 W8A8: it uses FP8 tensor cores, needs no download and goes through the same quality
pipeline. AWQ INT4 stays selectable (`models.awq_int4`) but was not measured.

**Consequences & trade-offs.** Activations are quantized too. Alone, FP8 weights were indistinguishable from BF16 in
the paired test; together with FP8 KV they gave the only significant drop (−2.58 GSM8K points, F7).

**Evidence & validation.** `quant` main effect 1.22–1.34× at every point (`S5.3-eff-*`); paired accuracy (`S5.6-*`).

**Revisit trigger.** A GPU without FP8 compute, or a paired drop above 1 point for `quant` alone.

---

<a id="adr-008"></a>
## ADR-008 — Speculative decoding: EAGLE-3 for results, n-gram only for smoke tests

**Date, status & scope.** 29 Oct 2026 (`6ebfa6d`: `speculative.method: eagle3` in `configs/env/main.yaml`); confirmed
by the smoke result (`bba6cee`). Accepted. Scope: optimization 3 (`spec`).

**Context & decision drivers.** Assignment 2 allowed "EAGLE-style draft heads or a small draft model". The smoke test
used n-gram (no extra model) and it accepted 0.97 of drafts on random-word prompts (`SMOKE_TEST_REPORT.md` §2),
which would overstate speculation.

**Options considered.** n-gram; a separate small draft model (extra memory, tokenizer constraints); an EAGLE-3 head for
this model (drafts from the target's own features; supported by vLLM 0.31).

**Decision & rationale.** EAGLE-3, `yuhuili/EAGLE3-LLaMA3.1-Instruct-8B`, 3 draft tokens, with an alternative head in
config. If the head fails to load, `spec` configurations are recorded as infeasible; no silent substitution.

**Consequences & trade-offs.** Results depend on one draft head. Its speedup falls from 1.35× at 2,048-token prompts to
1.00× at 6,144 (F6, `S5.5-*`); a shorter training context is a plausible cause (C12, low confidence), not checked.

**Evidence & validation.** `data/spec_vs_context.csv` (per-position acceptance).

**Revisit trigger.** Acceptance below 0.3 at a workload of interest, or a long-context draft head becomes available.

---

<a id="adr-009"></a>
## ADR-009 — Quality: lm-eval against the running server, once per numeric setting

**Date, status & scope.** 29 Oct 2026 (`6ebfa6d`: `harness/quality.py`; `bba6cee`: stop-string fix, one lm-eval call
per task). Accepted; refined by ADR-018. Scope: accuracy gate (M6).

**Context & decision drivers.** Only optimizations 1–3 change numerics, so 2³ = 8 quality runs cover every
configuration (Assignment 2 §B.1). The evaluation should use the same server path as the measurements, speculation
included.

**Options considered.** lm-eval's offline `vllm` backend (skips the server's speculation path); one evaluation per
configuration; one evaluation through the server per numeric setting.

**Decision & rationale.** lm-eval 0.4.13 `local-completions` against the running server for each numeric setting:
GSM8K and HumanEval (generative), MMLU (log-likelihood, with an offline fallback using the same engine arguments). Each
task runs in its own call. Servers start with `VLLM_MAX_STOP_STRINGS=16` because HumanEval sends 5 stop strings and
vLLM accepts 4 by default (found in the smoke test). Exact-match agreement with Default on 100 fixed prompts
(`data/agreement_prompts_v1.jsonl`) complements the benchmarks.

**Consequences & trade-offs.** HumanEval executes model-written code: the harness runs lm-eval as a subprocess with
`HF_ALLOW_CODE_EVAL=1` and `--confirm_run_unsafe_code`, in the same job as the servers and without a sandbox. The
trimmed phase-1 setting (GSM8K 500 questions, 4 MMLU subjects; ADR-012) proved too noisy for a 1-point gate (ADR-018).

**Evidence & validation.** `data/quality_trimmed_fast_plan.csv` (8 settings); `data/quality_paired_vs_default.csv`;
smoke re-run 5/5 settings ok.

**Revisit trigger.** A task whose confidence interval is wider than the gate (it happened: ADR-018).

---

<a id="adr-010"></a>
## ADR-010 — Telemetry: NVML + vLLM metrics (+ profiler passes) instead of DCGM/Nsight

**Date, status & scope.** 29 Oct 2026 (`6ebfa6d`: `harness/telemetry.py`); deep telemetry and profiler passes with the
investigation update (`harness/deep_telemetry.py`, first used 11 Nov). Accepted; amends ADR-000. Scope: binding
constraint (M7), integrity (M8).

**Context & decision drivers.** Assignment 2 §C.1 and the progress report planned DCGM or nvidia-smi, Nsight Systems
and vLLM's Prometheus metrics. The harness must run as an ordinary user on shared machines, and vLLM's metric names
change between versions (`docs/build_spec.md` §7.1).

**Options considered.** DCGM profiling counters and Nsight (extra installation and privileges); NVML through
`nvidia-ml-py` plus vLLM `/metrics` (no privileges, no kernel view); the same plus vLLM's built-in torch profiler for
short passes.

**Decision & rationale.** NVML (utilisation, clocks, power, memory, throttle reasons) and `/metrics` read through one
table of fallback names (`METRIC_TABLE`); DCGM optional. For phase 2, time series (NVML every 0.25 s, metrics every 1 s)
and 64-step profiler passes for the GPU-busy fraction and top kernels.

**Consequences & trade-offs.** No hardware FLOP or byte counters, so the binding constraint rests on profiler traces
and NVML (ADR-019). Two phase-2 signals were unusable: per-process CPU and vLLM's FLOP/byte counters read 0 in all 60
cells of `data/investigation_cells.csv`. NVML device memory is what exposed the co-tenant process (ADR-017).

**Evidence & validation.** `data/profiler_passes.csv` (18 passes), `data/binding_check.csv`; `tests/test_telemetry.py`.

**Revisit trigger.** Privileged profiling becomes available, or a conclusion rests on a counter that was not measured.

---

<a id="adr-011"></a>
## ADR-011 — Append-only results store; failures are results; resumable runs

**Date, status & scope.** 29 Oct 2026 (`harness/results.py`, `harness/orchestrator.py`); job deadline, `pipeline` and
`status` commands and Slurm chaining 5 Nov 2026 (`c42acca`). Accepted. Scope: reliability, data model.

**Context & decision drivers.** Runs last many hours on machines that can crash or end jobs at a time limit; some
configurations may not start. Neither may lose data or stop the experiment (`docs/build_spec.md` §4.6).

**Options considered.** A database service; one file per run; one append-only JSONL per kind of work with a
deterministic key.

**Decision & rationale.** `runs.jsonl` (plus `startup`, `sweep`, `quality`) keyed by `profile__env__point__bits__rep`;
the latest row per key wins; finished keys are skipped on restart. Start-up failures become `infeasible` or `timeout`
rows, measurement errors `failed` rows, and the walk continues; only GPU memory not being released
(`EnvironmentFault`) stops it. `harness pipeline` checks a job deadline before each engine group, so a job ended by
Slurm loses nothing.

**Consequences & trade-offs.** Each row stores `git rev-parse` of the harness, falling back to `nogit`. All 510 rows
say `nogit`, so the run directory on the cluster was not a git checkout and the exact revision is not in the data.

**Evidence & validation.** `tests/test_results.py`, `tests/test_slurm.py`; dry `pipeline` cut off by its deadline and
resumed (`SMOKE_TEST_REPORT.md` §6); GPU-free dry pipeline of the 7-optimization plan records 20 infeasible
configurations and 1 injected crash and finishes (`configs/profiles/dry.yaml`); 384/384 and 126/126 real runs `ok`
(`S4-runs`); `S8-git`.

**Revisit trigger.** Writers on more than one node, or a lost or duplicated run key.

---

<a id="adr-012"></a>
## ADR-012 — Scope: measure 5 of the 7 optimizations; chunked prefill and parallelism dropped

**Date, status & scope.** 2 Nov 2026 (`3526c58`: `configs/profiles/fast.yaml`, `optimizations: [quant, kvq, spec,
prefix, batch]`), while the 4 × RTX 6000 Ada node was still the plan; kept after the downgrade to the 2 × H100 node
(ADR-014). Accepted. Amends ADR-000 and ADR-006. Scope: milestones M4,
M7. **Mentor approval:** none on record. This is a change after the 9 October checkpoint, and no mentor feedback was
received during the project (report §7); the change is disclosed here and in the report instead.

**Context & decision drivers.** The problem statement names seven optimizations. Measuring all seven was estimated at
≈ 95 node-hours on a 4-GPU machine (`RUNBOOK.md` §2: sweeps 6 h, start-up check 4 h, quality 4–6 h, 512 cells × 3 reps
≈ 80 h), roughly the whole budget of Assignment 2 (≈ 100 node-hours), with nothing left for follow-up. The two
optimizations dropped are the ones that cost the most and were expected to add the least to a throughput metric:

- **Parallelism (opt. 7) needs more than one GPU per cell.** Every layout must use all GPUs of the cell
  (`vllm_flags.py` rejects any layout that does not), and its off state is tensor parallelism across those GPUs. With
  parallelism in the design, every cell, Default included, occupies the whole node, so cells can no longer run side by
  side on separate GPUs (ADR-013). On the 2-GPU node that alone halves the speed of the experiment. The planned layouts
  (`tp4`, `tp2x2`, `tp1x4`, `tp2pp2` in `configs/experiment.yaml`) also need 4 GPUs; on 2 GPUs only TP-2 versus two
  replicas remains. For an 8B model that fits on one 94 GB GPU, splitting it over PCIe was expected to help little.
- **Chunked prefill (opt. 6)** doubles the cells again and needs its own per-point sweep of
  `max_num_batched_tokens` (4 values × 7 points). It mainly protects tail latency rather than raising throughput; the
  factorial vLLM study cited in the progress report found its effect limited at default settings.

**Options considered.** (1) All 7 optimizations (512 cells); (2) 7 optimizations with 1 rep (no noise estimate);
(3) 7 optimizations at the centre only (no regime answers); (4) 5 optimizations at all 7 points with 3 reps.

**Decision & rationale.** Option 4: n = 5 gives 32 configurations at the centre + 6 × 16 = 128 cells × 3 reps = 384
runs, one GPU per cell, two cells at a time on the 2-GPU node. Measured cost: 11.5 h (7 Nov, 03:23–14:52 UTC;
`S4-time`). For comparison, at the same ≈ 10.8 GPU-minutes per cell, the 7-optimization plan would be 4× the cells
(512) and, with whole-node cells, half the concurrency: about 8× longer, ≈ 90 h of node time for the factorial alone
(≈ 12 jobs of 8 h), before the extra sweeps. The saved budget paid for phase 2 (ADR-016), which found the
contamination (ADR-017).

**Consequences & trade-offs.** Chunked prefill and parallelism are unmeasured; their rows of the Configuration ×
Regime table are empty. They stay implemented and tested so they can be measured by configuration only: flags and
layouts in `vllm_flags.py`, `tests/test_flags.py`, and the GPU-free dry pipeline, which runs the full 7-optimization
plan (1,536 runs) on a simulated engine. Phase-1 quality was trimmed in the same profile (ADR-009).

**Evidence & validation.** `configs/profiles/fast.yaml`, `configs/profiles/full.yaml`; `RUNBOOK.md` §2, §2b; all
phase-1 rows have `profile = fast` (`data/runs_fast_plan_all.csv`).

**Revisit trigger.** A node with ≥ 4 GPUs and ≈ 25 free node-hours for chunked prefill and parallelism at the centre.

---

<a id="adr-013"></a>
## ADR-013 — One configuration per GPU; cells on different GPUs measured concurrently

**Date, status & scope.** 2 Nov 2026 (`3526c58`: `run.gpus_per_cell`, `run_on_slots` in `orchestrator.py`, file lock
in `results.py`), written for 4 slots; run with 2 slots on the H100 node (`gpu_ids: auto`, `c42acca`). Accepted.
Scope: orchestrator, results store, integrity (M8).

**Context & decision drivers.** Once parallelism is out of the design (ADR-012), an 8B model needs only one GPU per
cell, and measuring one cell at a time would leave the other GPUs idle.

**Options considered.** Cells one at a time on one GPU; one cell per GPU in parallel slots; all GPUs per cell.

**Decision & rationale.** Each GPU is a slot with its own worker process, `vllm serve`, port range and load driver;
runs are grouped by engine flags to minimise restarts; appends to `runs.jsonl` take a file lock. Phase-2 cells can be
marked `exclusive` to run with the other slot idle.

**Consequences & trade-offs.** Two engines and two load drivers share 16 CPU cores and the host's PCIe links, so
interference had to be tested. The per-GPU view also made the contamination visible as a GPU-0-only effect (ADR-017).

**Evidence & validation.** Phase-2 `kvq_spec_outlier` test: one cell 5 times with the neighbour busy and 5 times idle,
2,690–2,740 vs. 2,666–2,713 tok/s, CV < 1% each (C17, `data/outlier_rerun.csv`, `S5.7`).

**Revisit trigger.** A busy-versus-idle difference above 2%, or a GPU with more CPU work per step.

---

<a id="adr-014"></a>
## ADR-014 — Execution platform: 2 × H100 NVL Slurm node, chained resumable jobs

**Date, status & scope.** 5 Nov 2026 (`c42acca`: `configs/env/h100.yaml`, `scripts/slurm/{harness.sbatch,submit.sh,
prefetch.sh}`, `pipeline`/`status` commands). Accepted. Scope: all 510 measured runs.

**Context & decision drivers.** The project had a node with 4 × RTX 6000 Ada (48 GB, Ada FP8), which the build spec and
`configs/env/main.yaml` target. On 5 Nov it was downgraded to a Slurm node with two newer GPUs, 2 × H100 NVL (94 GB,
Hopper FP8, 400 W cap), where jobs last at most 8 h and GPUs are shared with other users.

**Options considered.** Keep the 4-GPU node (no longer available); run the plan interactively on one machine; adapt the
harness to the scheduler.

**Decision & rationale.** Adapt the harness: `gpu_ids: auto` takes the GPUs Slurm assigns; each job uses its own port
range (`HARNESS_BASE_PORT`) and never reuses a busy port; `HARNESS_DEADLINE` stops new work 15 min before the job ends;
`submit.sh` chains jobs, each running `harness pipeline`; finished keys are skipped (ADR-011). GPU peaks for the
analysis are filled in `h100.yaml` from the datasheet of the variant `doctor` reported (H100 NVL: 3,900 GB/s,
835.5 / 1,670.5 dense BF16 / FP8 TFLOP/s).

**Consequences & trade-offs.** Two GPUs instead of four sharpen the cost of parallelism (ADR-012). One 94 GB GPU per
cell gives more KV cache than one 48 GB GPU would have. The node does not isolate GPUs (ADR-015), and the GPUs run at
their power cap (F9). The jobs did not use `scripts/setup_env.sh`: the same pinned vLLM and PyTorch wheels were
installed in a conda environment on the cluster (Python 3.10, the cluster's `nvcc` 13.1 for FlashInfer's JIT kernels),
set up by the scripts in `cluster/` and launched with `run_study.sbatch`; the versions recorded in every row match the
pins (ADR-001). A first attempt under Python 3.10.8 failed at engine start for every configuration and was discarded.

**Evidence & validation.** `RUNBOOK.md` §2c; `tests/test_slurm.py`; all 510 rows have `env = h100` and driver
590.48.01.

**Revisit trigger.** A move to another cluster or GPU type.

---

<a id="adr-015"></a>
## ADR-015 — GPU sharing on the Slurm node: pre-flight VRAM check only

**Date, status & scope.** 5 Nov 2026 (`c42acca`: `--gres=gpu:2` left commented in `scripts/slurm/harness.sbatch`;
`min_free_vram_gb: 70` in `h100.yaml`; `gpu_checks` in `doctor.py`). Accepted; **shown insufficient** by the
contamination (ADR-017); replacement open. Kept, not deleted. Scope: validity of every cluster run.

**Context & decision drivers.** It was not known whether the cluster schedules GPUs as a generic resource, and a wrong
`--gres` request makes `sbatch` fail (comment in `harness.sbatch`). Without it, Slurm assigns no GPUs and other users'
processes can use the same GPUs.

**Options considered.** (1) Request GPUs or an exclusive node; (2) a pre-flight check: `doctor` refuses to start if a
GPU has less than 70 GiB free and names the processes holding it; (3) the pre-flight check plus a per-run check for
foreign GPU memory.

**Decision & rationale.** Option 2, judged sufficient. It did stop one start: a pre-flight check found another user's
vLLM server holding 85.6 GiB on GPU 1 (`FINDINGS.md`, F1).

**Consequences & trade-offs.** A co-tenant that arrived during a job went undetected: from 11:24 UTC on 7 Nov, 71 runs
on GPU 0 ran at 0.56–0.69× of their true speed (ADR-017). Per-run device memory was enough to detect it afterwards; 16
cells were lost.

**Evidence & validation.** `harness/doctor.py`; `data/contamination_by_phase.csv` (`S5.2-tab`).

**Revisit trigger.** Already triggered: before further measurement use option 1 if the cluster supports it, otherwise
option 3.

---

<a id="adr-016"></a>
## ADR-016 — Phase 2: focused re-runs of anomalies with deep telemetry

**Date, status & scope.** Investigation update after `c42acca` (`configs/profiles/investigate.yaml`,
`harness/deep_telemetry.py`, `harness/analysis/investigate.py`, `tests/test_investigate.py`, `cluster/*_investigate.sh`);
measured 11 Nov 2026, 19:08–23:31 UTC. Accepted; amends ADR-000. Scope: M6–M8.

**Context & decision drivers.** Phase 1 left five anomalies, each with its `why:` in `investigate.yaml`: spec + prefix
slower than Default at 0% reuse; EAGLE-3 acceptance collapsing at 8k tokens; a 1.24–1.33× batching effect where the
batch limit never binds; one noisy kvq + spec cell; every final configuration labelled "per-step overhead".

**Options considered.** (1) Spend the remaining budget on more optimizations or points; (2) more repetitions of all
cells; (3) one targeted investigation per anomaly.

**Decision & rationale.** Option 3: six investigations (`spec_prefix_no_reuse`, `spec_long_context`,
`high_load_dealias`, `kvq_spec_outlier`, `kvq_spec_outlier_alone`, `binding_check`), 60 cells, 2–5 reps, time-series
telemetry, 18 profiler passes, and a paired accuracy re-test (ADR-018). Unexplained numbers cannot support
recommendations, so explanation came before breadth.

**Consequences & trade-offs.** Re-running phase-1 cells exposed the contamination (ADR-017). Three anomalies (spec +
prefix at 0% reuse, the batching effect, the noisy cell) were in contaminated cells; the other two (EAGLE-3 at long
context, the binding label) are real (F6, F9).

**Evidence & validation.** `data/runs_investigation.csv` (126 runs), `data/investigation_cells.csv` (60 cells)
(`S4-runs`); `data/contamination_matched_runs.csv`; figure 09.

**Revisit trigger.** A new anomaly larger than 5% in a cell that drives a recommendation.

---

<a id="adr-017"></a>
## ADR-017 — Contaminated runs: replace or exclude, never rescale

**Date, status & scope.** Between the phase-2 run (11 Nov) and the report pack (14 Nov). Accepted. Scope: all phase-1
data and every headline result.

**Context & decision drivers.** From 11:24 UTC on 7 Nov, every phase-1 run on GPU 0 ran at 0.56–0.69× of its clean
re-run (71 runs, 24 cells), with +4.3 GB foreign device memory, a higher SM clock and lower power, while GPU 1 stayed at
1.006×. Repetitions inside a contaminated cell still agreed (CV 0.2–3.8%), so noise statistics hid the problem.

**Options considered.** (1) Keep the data; (2) divide by a constant factor (≈ 0.6); (3) replace the re-run cells and
exclude the rest; (4) re-run all contaminated cells (no budget).

**Decision & rationale.** Option 3. The slowdown varied from 0.56× to 0.69×, so one factor is not justified. All 71
runs are flagged `contaminated`; 8 cells are replaced by clean re-runs, 16 excluded
(`data/cells_missing_after_cleaning.csv`), 16 phase-2-only cells at the standard points added: 128 cells.

**Consequences & trade-offs.** Best speedups fell from 4.50× to 2.62× (90% reuse) and from 2.64× to 1.67× (high
concurrency). Sixteen cells have no value. The pack's claim C2 ("< 1%") is wrong and is corrected in the report.

**Evidence & validation.** `data/contamination_by_phase.csv`, `data/contamination_matched_runs.csv` (the 40
uncontaminated phase-1 runs that were re-run match within 2.2%), figure 01; `S5.2`.

**Revisit trigger.** Exclusive GPUs become available (re-run the 16 cells), or a per-run foreign-memory check is added.

---

<a id="adr-018"></a>
## ADR-018 — Accuracy gate: paired re-test and three reported policies

**Date, status & scope.** Re-test configured in `investigate.yaml` (`quality.settings: 000 100 010 110 001`,
`log_samples: true`, full task limits) and run with phase 2; analysis by 14 Nov. Accepted; refines ADR-004 and ADR-009.
Scope: M6, M7.

**Context & decision drivers.** The trimmed phase-1 evaluation excluded FP8 weights and FP8 KV individually: FP8
weights −3.0 points on HumanEval (164 problems), FP8 KV −2.8 on GSM8K (500) and −2.1 on MMLU (4 subjects)
(`data/quality_trimmed_fast_plan.csv`). At these sizes an unpaired 95% interval (≈ ±3–5 points) is much wider than the
1-point gate.

**Options considered.** Keep the phase-1 rule; a larger unpaired re-run; a paired re-test on full GSM8K (1,319),
HumanEval (164) and 8 MMLU subjects (1,687), with every answer logged, McNemar test and paired CI.

**Decision & rationale.** The paired re-test for Default, quant, kvq, quant + kvq and spec (paired test in
`harness/analysis/investigate.py`). The untested settings (quant + spec, kvq + spec, all three) cannot be cleared, so
results are given under three policies: **A**, original rule (main result); **B**, paired verdicts with untested
settings assumed to pass (upper bound); **C**, latency gates only.

**Consequences & trade-offs.** Policy B (up to 3.30×) is unproven until the spec combinations are tested. The re-test
found the real risk: quant + kvq together lose 2.58 GSM8K points (95% CI −4.9 to −0.3, p = 0.03).

**Evidence & validation.** `data/quality_paired_vs_default.csv`, `data/final_configs_by_policy.csv`, figure 07;
`S5.6-*`, `S5.1-B-*`.

**Revisit trigger.** Paired results for the spec combinations.

---

<a id="adr-019"></a>
## ADR-019 — Binding constraint: profiler traces override the first-order rule

**Date, status & scope.** `binding_check` investigation (phase 2); applied in the analysis by 14 Nov. Accepted.
Scope: binding constraint of each final configuration (M7).

**Context & decision drivers.** The rule in `harness/analysis/binding.py` (`docs/build_spec.md` §7.5) labels a configuration
KV-, bandwidth-, compute- or scheduler-bound from estimated bytes and FLOPs, and "per-step overhead" otherwise. It
labelled every final configuration "per-step overhead" (bandwidth ≤ 50% and compute ≤ 41% of peak; `investigate.yaml`).

**Options considered.** Accept the label; measure directly with privileged counters (unavailable, ADR-010); profile
the configurations with vLLM's torch profiler and read NVML throttle reasons.

**Decision & rationale.** Profile. In all 18 traces the GPU executed kernels 96–99% of the time, mostly cuBLAS GEMMs,
and NVML reported the software power-cap throttle in all 126 phase-2 runs. There is no idle GPU time for CPU overhead
to explain, so the report states the constraint as GEMM work at power-capped clocks (F9) and keeps the harness label
only as a superseded estimate.

**Consequences & trade-offs.** The first-order rule needs measured counters to be trustworthy; until then the report
does not use its label.

**Evidence & validation.** `data/profiler_passes.csv`, `data/binding_check.csv`, figure 08; `S5.7`.

**Revisit trigger.** Measured byte/FLOP counters available, or a different GPU without a power cap.

---

<a id="adr-020"></a>
## ADR-020 — GPU-free check of every report number against the exported data

**Date, status & scope.** 21 Nov 2026 (`evaluation/check_report_numbers.py`, `evaluation/make_report_figures.py`).
Accepted. Scope: the final report (M8).

**Context & decision drivers.** Evaluators have no H100s, the report was drafted with AI assistance, and the report
pack's own claims contained an error (C2). Every number in the report needs a check anyone can run from a clean
checkout.

**Options considered.** Trust the pack's text; re-run `python -m harness analyze`, which needs the raw `results/`
directories from the cluster; a script that recomputes each quoted number from the committed CSVs.

**Decision & rationale.** The script: each check recomputes a count, median, CV, ratio, best configuration or
paired-test row from `evaluation/report_pack/data/` and compares it with the report; check ids name the report section.
Figures 10–12 and Appendix A are generated from the same CSVs, so they cannot drift.

**Consequences & trade-offs.** It verifies arithmetic on exported data, not the measurements themselves (those rest on
the harness and its 69 unit tests). It found C2 and one rounding error (centre Default p99 TPOT is 18.0 ms).

**Evidence & validation.** `python evaluation/check_report_numbers.py` → 289/289 pass (21 Nov 2026).

**Revisit trigger.** A number in the report that no check covers, or a change to the data.