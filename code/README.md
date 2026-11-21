# Combining Infrastructure-Level Optimizations for LLM Serving

CS5903 Distributed AI Training, IIT Hyderabad · Project 14, Group 2 · Panshul Jindal, Mohd Shozab

A measurement harness for vLLM that switches serving optimizations on and off independently, and a factorial
study of five of them on Llama-3.1-8B-Instruct: which optimizations contribute, under which workloads, and where
they stop helping.

- **Report:** [`docs/Project_Report.pdf`](docs/Project_Report.pdf) (source in `docs/report/`)
- **Architecture Decision Records:** [`docs/Architecture_Decision_Records.md`](docs/Architecture_Decision_Records.md)
- **Results, findings and figures:** [`evaluation/report_pack/`](evaluation/report_pack/README.md)
- **Slides:** `docs/presentation/`

## Status

| Part | State |
|---|---|
| Harness (`harness/`) | Complete: config, design, vLLM launcher, workload, load driver, telemetry, quality, analysis, Slurm support |
| Study | 5 optimizations (FP8 weights, FP8 KV cache, EAGLE-3 speculative decoding, prefix caching, tuned batching) x 7 workload points on 2 x H100 NVL: 384 factorial runs + 126 investigation runs + 18 profiler traces + paired accuracy re-test |
| Not measured | Chunked prefill and parallelism (implemented and tested on a simulated engine only); 16 cells lost to GPU contamination |

Headline results (policy A, all latency gates met): the best configuration reaches 1.39x-2.62x Default throughput;
speedups multiply and, with speculative decoding, compound beyond the product (up to +31%); EAGLE-3 stops helping
beyond ~4k-token prompts; FP8 weights + FP8 KV together cost 2.6 GSM8K points. Details and caveats: report §5, §8.

## Architecture

```
configs/*.yaml ─► config (pydantic) ─► design (run plan) ─► orchestrator ─► vllm_flags ─► vllm serve (1 per GPU)
                                                               │                              ▲
                                       workload (hashed traces)┴► load_driver (closed loop) ──┘
                                       telemetry (NVML + /metrics) · quality (lm-eval) ─► results/*.jsonl ─► analysis
```

| Path | Responsibility |
|---|---|
| `harness/config.py`, `configs/` | Validated YAML: experiment, environment (`env/`), profiles (`profiles/`), tuned values |
| `harness/design.py`, `harness/optimizations.py` | 7-bit configurations; full factorial at the centre, even-parity half-fractions elsewhere |
| `harness/engine/vllm_flags.py` | The only mapping from a configuration to `vllm serve` flags |
| `harness/engine/vllm_server.py`, `fake.py` | Server lifecycle and failure classification; simulated engine for GPU-free runs |
| `harness/workload.py`, `harness/load_driver.py` | Token-exact synthetic traces; async closed-loop client |
| `harness/telemetry.py`, `harness/deep_telemetry.py` | NVML and vLLM Prometheus metrics; time series and profiler parsing |
| `harness/quality.py` | lm-eval against the running server, per numeric setting; paired per-question logs |
| `harness/orchestrator.py`, `harness/cli.py` | Resumable runs, GPU slots, job deadlines; `python -m harness <command>` |
| `harness/analysis/` | Value and gates, OLS effects, interactions, Shapley shares, binding constraint, investigation report |
| `evaluation/` | GPU-free re-derivation of every report number; report figures 10-12 and Appendix A |
| `scripts/report_pack/` | Builds `evaluation/report_pack/` (cleaned dataset, CSVs, figures 01-09, findings) from `results/` |
| `cluster/`, `run_study.sbatch` | Slurm scripts used for the H100 runs (conda environment) |
| `scripts/setup_env.sh`, `scripts/slurm/` | Reference environment (uv venv) and generic Slurm job chain |

## Prerequisites

- Linux x86-64, Python 3.11 (3.10 also works), NVIDIA driver >= 570 for GPU runs. Pinned dependencies:
  `requirements.txt` (vLLM 0.31.0+cu129, PyTorch 2.13.0+cu129, FlashInfer 0.7.0.post1, lm-eval 0.4.13).
- GPU-free use (tests, dry run, number checks) needs no GPU, no model download and no credentials.
- Figure scripts additionally need `matplotlib`.

## Setup

```bash
scripts/setup_env.sh                 # uv, Python 3.11 venv, pinned wheels, CUDA 12.9 nvcc for FlashInfer JIT
source .venv/bin/activate
cp .env.example .env                 # optional; nothing secret is required (models are ungated)
uv pip install matplotlib            # only for the figure scripts
```

There is no database; results are append-only JSONL files under `results/<profile>/` (not committed).

### Environment variables (`.env.example`)

| Variable | Purpose |
|---|---|
| `HF_TOKEN` | Only for gated model repositories (the configured models are ungated) |
| `HF_HOME` | Model and dataset cache location |
| `HF_HUB_OFFLINE` | `1` on compute nodes without internet (after `scripts/slurm/prefetch.sh`) |
| `HARNESS_BASE_PORT` | First port for this job's vLLM servers (set per Slurm job) |
| `HARNESS_DEADLINE` | Unix time after which no new engine group starts (set from the Slurm job end) |

## Run and test

GPU-free (anyone can run these from a clean checkout):

```bash
pytest -q                                                     # 69 tests; 68 pass + 1 skipped without the tokenizer download
python -m harness pipeline --profile dry --env main --dry-run # full 7-optimization plan on the simulated engine
python evaluation/check_report_numbers.py                     # re-derives every report number: 288/288
python evaluation/make_report_figures.py                      # figures 10-12 and Appendix A from the CSVs
python -m harness plan --profile fast --env h100              # print the measured run plan (128 cells)
```

Expected output of the check script: one `PASS` line per check and `288/288 checks passed`, exit code 0.
The dry pipeline ends with `pipeline complete: every key attempted` and writes a report under
`results/dry/report/`.

GPU runs (as done on the 2 x H100 NVL Slurm node; see `RUNBOOK.md` §2b-§2d and `cluster/README.md`):

```bash
python -m harness doctor --env h100                           # environment checks; must print PASS
PROFILE=fast sbatch run_study.sbatch                          # phase 1: factorial plan, self-chaining 8 h jobs
PROFILE=investigate sbatch run_study.sbatch                   # phase 2: investigation + paired quality
python -m harness status --profile fast --env h100            # progress
python -m harness analyze --profile fast                      # standard report in results/fast/report/
```

Regenerating the report pack from raw results (`results/fast/`, `results/investigate/`):

```bash
python scripts/report_pack/build.py && python evaluation/make_report_figures.py
```

## Operations

- Servers run in their own process groups and are stopped by the harness; leftovers of an interrupted job:
  `pkill -u $USER -f "vllm serve"`, then confirm with `nvidia-smi`. Every command resumes where it stopped;
  `--retry-failed` reruns failed keys, `--force` reruns everything (`RUNBOOK.md` §4).
- Logs: `logs/slurm-<jobid>.out|err`, structured logs in `results/<profile>/logs/`, server logs in
  `results/<profile>/engine_logs/`.
- External resources: Hugging Face downloads (~16 GB model + EAGLE-3 head + eval datasets); a GPU node with
  FP8-capable GPUs. The raw per-run results (~2 GB: per-request timings, telemetry series, profiler traces) are not
  in the repository; the exported run-level data is in `evaluation/report_pack/data/`.

## Limitations

One model, one draft head, synthetic prompts with fixed 256-token outputs, a power-capped H100 NVL, 2-3 repetitions
per cell; the cluster node does not isolate GPUs. Full list: report §8 and `evaluation/report_pack/LIMITATIONS.md`.
