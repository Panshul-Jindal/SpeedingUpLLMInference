#!/bin/bash
# Progress at any time (login node): todo counts per step, plus your queued/running jobs.
cd "$(dirname "$0")/.."
source cluster/common.sh
load_env >/dev/null
python -m harness status --profile fast --env h100
squeue -u "$USER"
