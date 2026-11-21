#!/bin/bash
# Submit a chain of N jobs (default 3), each starting after the previous one ends (whatever its exit code).
# Every job resumes where the last stopped; a job that finds nothing left exits within minutes.
#
#   scripts/slurm/submit.sh            # 3 jobs, PROFILE=fast ENV_NAME=h100
#   PROFILE=fast scripts/slurm/submit.sh 2
#
# Run from anywhere; jobs run from the repo root. Progress: `python -m harness status --profile fast --env h100`.
set -euo pipefail
cd "$(dirname "$0")/../.."
N="${1:-3}"
mkdir -p logs            # sbatch opens --output=logs/... at submit time

if [ ! -x .venv/bin/python ]; then
    echo "No .venv: run scripts/setup_env.sh first (on the login node is fine)." >&2
    exit 1
fi

prev=""
for i in $(seq 1 "$N"); do
    dep=()
    [ -n "$prev" ] && dep=(--dependency="afterany:$prev")
    prev=$(sbatch --parsable ${dep[@]+"${dep[@]}"} scripts/slurm/harness.sbatch)
    echo "job $i/$N: $prev${dep[0]:+ (after ${dep[0]#--dependency=afterany:})}"
done
echo "Logs: logs/slurm-<jobid>.out   Cancel the rest of the chain with: scancel <jobid> ..."
