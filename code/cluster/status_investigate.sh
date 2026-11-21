#!/bin/bash
# Progress of the investigation run (todo counts) + your queued jobs.
cd "$(dirname "$0")/.."
source cluster/common.sh
load_env >/dev/null
python -m harness status --profile investigate --env h100
squeue -u "$USER"
