#!/bin/bash
#SBATCH --job-name=llm-serving
#SBATCH --partition=gpu
#SBATCH --nodelist=ccs-srv003
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
##SBATCH --gres=gpu:2        # uncomment if the cluster schedules GPUs as a generic resource

# One time-limited slice of the study. Submit a chain with scripts/slurm/submit.sh, from the repo root.
# Each job runs `python -m harness pipeline`, which skips finished work, retries failed keys once, and
# starts nothing that cannot finish before the job's end (minus MARGIN_MIN). Exit code 10 = work left.
#
# Overrides (environment, passed through by sbatch): PROFILE (fast), ENV_NAME (h100), MARGIN_MIN (15),
# HF_HOME, HF_HUB_OFFLINE (set to 1 if the compute node has no internet; run scripts/slurm/prefetch.sh first).

set -euo pipefail

# Under sbatch, $0 is a spooled copy of this script, so use the submit directory (the repo root).
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

PROFILE="${PROFILE:-fast}"
ENV_NAME="${ENV_NAME:-h100}"
MARGIN_MIN="${MARGIN_MIN:-15}"

# Do NOT `module load cuda-13.1`: the pinned vLLM/torch/FlashInfer are CUDA 12.9 builds, and FlashInfer's JIT
# uses the CUDA 12.9 toolkit in .cuda-toolkit (installed by scripts/setup_env.sh).
export HF_HOME="${HF_HOME:-$HOME/gemma/hf-cache}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
# GPU indices mean the same to CUDA, NVML and Slurm (doctor verifies this).
export CUDA_DEVICE_ORDER=PCI_BUS_ID
# Per-job port base so a shared node's other servers (and our own previous job) never collide.
export HARNESS_BASE_PORT=$(( 20000 + (SLURM_JOB_ID % 3000) * 10 ))

# Deadline: the job's scheduled end time minus a margin for analysis and clean shutdown.
END_ISO=$(squeue -h -j "$SLURM_JOB_ID" -o %e 2>/dev/null || true)
if ! END_TS=$(date -d "$END_ISO" +%s 2>/dev/null); then
    END_TS=$(( $(date +%s) + 8 * 3600 ))
fi
export HARNESS_DEADLINE=$(( END_TS - MARGIN_MIN * 60 ))

if [ -z "${HF_TOKEN:-}" ]; then
    echo "NOTE: HF_TOKEN is not set (fine: the configured models are ungated)."
fi

echo "============================================================"
echo "Job $SLURM_JOB_ID on $(hostname), partition ${SLURM_JOB_PARTITION:-?}, ${SLURM_CPUS_PER_TASK:-?} CPUs"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<not set>}"
echo "Profile/env:          $PROFILE / $ENV_NAME"
echo "HF_HOME:              $HF_HOME (offline=$HF_HUB_OFFLINE)"
echo "Ports from:           $HARNESS_BASE_PORT"
echo "Job ends:             ${END_ISO:-unknown}; no new work after $(date -d "@$HARNESS_DEADLINE")"
echo "============================================================"
nvidia-smi

# If the job is cancelled, stop the vLLM servers this job started (they run in their own process groups;
# Slurm normally reaps them with the job). Only this job's ports: base .. base+39.
cleanup() {
    for p in $(seq "$HARNESS_BASE_PORT" $(( HARNESS_BASE_PORT + 39 ))); do
        pkill -u "$USER" -f "vllm serve .*--port $p\$" 2>/dev/null || true
    done
}
trap cleanup EXIT INT TERM

source .venv/bin/activate
EXIT_CODE=0
python -u -m harness pipeline --profile "$PROFILE" --env "$ENV_NAME" || EXIT_CODE=$?

echo "============================================================"
case "$EXIT_CODE" in
    0)  echo "Study complete. Report: results/$PROFILE/report/summary.md" ;;
    10) echo "Work left: the next job in the chain resumes (or resubmit: scripts/slurm/submit.sh 1)." ;;
    3)  echo "ENVIRONMENT FAULT (GPU memory not released). Check nvidia-smi on the node, then resubmit." ;;
    *)  echo "Pipeline exited with $EXIT_CODE; see logs/slurm-$SLURM_JOB_ID.err and results/$PROFILE/logs/." ;;
esac
# 10 is expected mid-chain; report it as success so `afterok` chains would still work.
[ "$EXIT_CODE" -eq 10 ] && exit 0
exit "$EXIT_CODE"
