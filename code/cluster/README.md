# How to run on the Slurm cluster (2 × H100, conda env llm_haskell, cuda-13.1)

Everything below runs from this folder (the repo root). Only `run_study.sbatch` uses GPUs.
Every script loads the environment the same way (`cluster/common.sh`):
```bash
module load Anaconda3/
conda activate llm_haskell      # Python 3.10
module load cuda-13.1
```

## 0. Check the settings in `cluster/common.sh`
```bash
ANACONDA_MODULE=Anaconda3/
CUDA_MODULE=cuda-13.1
CONDA_ENV=llm_haskell
HF_HOME=$HOME/gemma/hf-cache      # where models/datasets are cached (same for download and jobs)
```

## 1. One-time setup (login node, ~30 min)
```bash
cluster/1_setup.sh       # adds lm-eval, statsmodels, ... to llm_haskell; installs vLLM 0.31.0 only if vLLM is missing
cluster/2_download.sh    # Llama-3.1-8B-Instruct (~16 GB), EAGLE-3 head, GSM8K/HumanEval/MMLU
cluster/3_check_gpu.sh   # must end with "doctor: PASS"   (GRES=gpu:2 cluster/3_check_gpu.sh if needed)
```
Then edit `configs/env/h100.yaml` → `peaks:` using the GPU name doctor printed:
- "H100 80GB HBM3" (SXM): `mem_bandwidth_gbps: 3350`, `bf16_tflops: 989`, `fp8_tflops: 1979`
- H100 PCIe: `2000`, `756`, `1513`

If your cluster needs a GPU request, uncomment `#SBATCH --gres=gpu:2` in `run_study.sbatch`.

## 2. Submit (one command)
```bash
sbatch run_study.sbatch
# compute nodes without internet:
HF_HUB_OFFLINE=1 sbatch run_study.sbatch
```
Each job runs up to 8 h and resubmits itself while work is left (max 4 jobs; ~2 are needed, ~13.5 GPU-h).

## 3. Watch progress
```bash
cluster/status.sh                 # todo counts per step + squeue
tail -f logs/slurm-<jobid>.out
```
Done when every step shows `todo 0` or the log says `Study complete`. Report: `results/fast/report/summary.md`.

## 4. Send back
```bash
cluster/pack_results.sh           # -> llm-serving-results-h100.tar.gz
```

## If something fails
After fixing it, just `sbatch run_study.sbatch` again — finished work is skipped, nothing runs twice.

| End of `logs/slurm-<jobid>.out` | Fix |
|---|---|
| `PRE-FLIGHT FAIL: ...` | Do what the message says (setup / download / paths in `cluster/common.sh`) |
| `doctor: FAIL`, "free VRAM" | Someone else is using a GPU; wait and resubmit |
| `doctor: FAIL`, "gpu mapping" | Send the doctor output to the study owner |
| `doctor: FAIL`, "nvcc"/"vllm"/missing flags | Re-run `cluster/1_setup.sh`; if your vLLM isn't 0.31.0, `FORCE_PINNED=1 cluster/1_setup.sh` |
| Many FP8-KV (bit 1) rows infeasible, nvcc/FlashInfer errors in `results/fast/engine_logs/` | nvcc 13.1 vs CUDA-12.9 torch mismatch; see the note below |
| `ENVIRONMENT FAULT` (exit 3) | `nvidia-smi` on the node, kill only *your* leftover `vllm`/`EngineCore`, resubmit |
| Many infeasible rows with spec (bit 3) | `scancel` and send `results/fast/startup.jsonl` to the study owner |
| `LocalEntryNotFoundError` | Re-run `cluster/2_download.sh` with the same HF_HOME; submit with `HF_HUB_OFFLINE=1` |

## Note on Python 3.10 and cuda-13.1
The repo was pinned for Python 3.11 and CUDA 12.9, and its own guide says not to load cuda-13.1. The code itself has
no 3.11-only syntax, so it is run straight from this folder (`PYTHONPATH`), not pip-installed. Doctor will show
`python WARN` (3.10) — that is expected. vLLM/torch bundle their own CUDA runtime; cuda-13.1 only supplies `nvcc`
for FlashInfer's JIT kernels (FP8 KV cache). If those fail to compile, the affected configurations are recorded
as infeasible rather than crashing the study.
