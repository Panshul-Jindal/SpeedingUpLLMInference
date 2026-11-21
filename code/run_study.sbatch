#!/bin/bash
#SBATCH --job-name=llm-serving
#SBATCH --partition=gpu
#SBATCH --nodelist=ccs-srv004
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
##SBATCH --gres=gpu:2          # uncomment if your cluster schedules GPUs as gres (most do)
#
# LLM Serving Study: one self-resubmitting job (2 x H100, Llama-3.1-8B, profile "fast").
# Environment: module load Anaconda3/ ; conda activate llm_haskell (Python 3.10) ; module load cuda-13.1
# (all set in cluster/common.sh).
#
# Before the first submit, on the login node from this folder (once): see HOW_TO_RUN.md
#   cluster/1_setup.sh  &&  cluster/2_download.sh  &&  cluster/3_check_gpu.sh
#   fill `peaks:` in configs/env/h100.yaml (SXM: 3350/989/1979, PCIe: 2000/756/1513)
#
# Submit (from this folder):
#   sbatch run_study.sbatch
#   HF_HUB_OFFLINE=1 sbatch run_study.sbatch       # if compute nodes have no internet
#
# The harness pipeline is resumable: it skips finished work, retries failures once and starts nothing it
# cannot finish MARGIN_MIN minutes before the 8 h limit. When it exits 10 (work left), this script submits
# the next job itself (afterany on this one), up to MAX_CHAIN jobs. Total work is ~13.5 GPU-hours (2 jobs).
#
# Overrides (environment): PROFILE (fast), ENV_NAME (h100), MARGIN_MIN (15), MAX_CHAIN (4), HF_HUB_OFFLINE.

set -euo pipefail

# Under sbatch, $0 is a spooled copy, so work from the submit directory (must be this folder).
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

PROFILE="${PROFILE:-fast}"
ENV_NAME="${ENV_NAME:-h100}"
MARGIN_MIN="${MARGIN_MIN:-15}"
MAX_CHAIN="${MAX_CHAIN:-4}"
CHAIN_INDEX="${CHAIN_INDEX:-1}"

fail() { echo "PRE-FLIGHT FAIL: $*" >&2; exit 2; }
[ -f cluster/common.sh ] || fail "submit from the llm-serving-study folder (no cluster/common.sh in $PWD)"

# ---- environment: Anaconda3 module -> conda activate llm_haskell -> cuda-13.1 module ---------------------
source cluster/common.sh
load_env || fail "could not load $ANACONDA_MODULE / conda env $CONDA_ENV / $CUDA_MODULE"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"

# ---- pre-flight: fail fast instead of burning GPU time ---------------------------------------------------
[ -d "$HF_HOME" ] || fail "HF_HOME=$HF_HOME does not exist: run cluster/2_download.sh (HF_HOME is set in cluster/common.sh)"
python - <<'PY' || fail "packages missing in conda env $CONDA_ENV: run cluster/1_setup.sh"
import importlib.metadata as m
for pkg in ("vllm", "torch", "lm_eval"):
    print(f"[env] {pkg:8s} {m.version(pkg)}")
import pydantic, statsmodels, pyarrow, aiohttp, pynvml  # noqa: F401
PY
if grep -qE '^\s*mem_bandwidth_gbps:\s*null' "configs/env/${ENV_NAME}.yaml"; then
    echo "WARNING: peaks in configs/env/${ENV_NAME}.yaml are not filled in; bandwidth/compute checks will say 'unavailable'."
fi
if [ -x .cuda-toolkit/bin/nvcc ]; then
    echo "NOTE: .cuda-toolkit/ exists; the harness prefers its nvcc over $CUDA_MODULE for FlashInfer JIT."
fi

# GPU indices mean the same to CUDA, NVML and Slurm (doctor verifies this).
export CUDA_DEVICE_ORDER=PCI_BUS_ID
# Per-job port range so other users' servers (and our previous job) never collide.
export HARNESS_BASE_PORT=$(( 20000 + (SLURM_JOB_ID % 3000) * 10 ))

# Deadline = scheduled job end minus margin; the harness starts no work that would overrun it.
END_ISO=$(squeue -h -j "$SLURM_JOB_ID" -o %e 2>/dev/null || true)
if ! END_TS=$(date -d "$END_ISO" +%s 2>/dev/null); then
    END_TS=$(( $(date +%s) + 8 * 3600 ))
fi
export HARNESS_DEADLINE=$(( END_TS - MARGIN_MIN * 60 ))

echo "============================================================"
echo "Job $SLURM_JOB_ID (chain $CHAIN_INDEX/$MAX_CHAIN) on $(hostname), ${SLURM_CPUS_PER_TASK:-?} CPUs"
echo "Repo commit:          $(git rev-parse --short HEAD 2>/dev/null || echo '?')"
echo "Conda env / CUDA:     $CONDA_ENV / $CUDA_MODULE"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<not set>}"
echo "Profile/env:          $PROFILE / $ENV_NAME"
echo "HF_HOME:              $HF_HOME (offline=$HF_HUB_OFFLINE)"
echo "Ports from:           $HARNESS_BASE_PORT"
echo "Job ends:             ${END_ISO:-unknown}; no new work after $(date -d "@$HARNESS_DEADLINE")"
echo "============================================================"
nvidia-smi || true

# Stop only the vLLM servers this job started (its own port range) on exit/cancel.
cleanup() {
    for p in $(seq "$HARNESS_BASE_PORT" $(( HARNESS_BASE_PORT + 39 ))); do
        pkill -u "$USER" -f "vllm serve .*--port $p\$" 2>/dev/null || true
    done
}
trap cleanup EXIT INT TERM

# ---- run -----------------------------------------------------------------------------------------------
EXIT_CODE=0
python -u -m harness pipeline --profile "$PROFILE" --env "$ENV_NAME" || EXIT_CODE=$?

echo "============================================================"
case "$EXIT_CODE" in
    0)  echo "Study complete. Report: results/$PROFILE/report/summary.md" ;;
    10) echo "Work left for the next job." ;;
    3)  echo "ENVIRONMENT FAULT (GPU memory not released). Check nvidia-smi on the node, kill only YOUR leftover vllm/EngineCore processes, then resubmit." ;;
    1)  echo "Pipeline/doctor failed (e.g. GPUs busy, setup incomplete). See logs/slurm-$SLURM_JOB_ID.err and results/$PROFILE/logs/." ;;
    *)  echo "Pipeline exited with $EXIT_CODE; see logs/slurm-$SLURM_JOB_ID.err and results/$PROFILE/logs/." ;;
esac

# ---- self-chain: only when there is work left and no hard failure ---------------------------------------
if [ "$EXIT_CODE" -eq 10 ]; then
    if [ "$CHAIN_INDEX" -lt "$MAX_CHAIN" ]; then
        NEXT=$(env -u CUDA_VISIBLE_DEVICES sbatch --parsable --dependency="afterany:$SLURM_JOB_ID" \
                      --export=ALL,CHAIN_INDEX=$(( CHAIN_INDEX + 1 )) \
                      "$SLURM_SUBMIT_DIR/run_study.sbatch") \
            && echo "Submitted next job $NEXT (chain $(( CHAIN_INDEX + 1 ))/$MAX_CHAIN)." \
            || echo "Could not resubmit; run: sbatch run_study.sbatch"
    else
        echo "Reached MAX_CHAIN=$MAX_CHAIN with work left; resubmit with: sbatch run_study.sbatch"
    fi
    exit 0
fi
exit "$EXIT_CODE"
