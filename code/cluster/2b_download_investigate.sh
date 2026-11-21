#!/bin/bash
# Investigation run: fetch what cluster/2_download.sh (profile "fast") did not: the 4 extra MMLU subjects
# (the paired quality test uses all 8). Same HF_HOME. Login node, from the repo root, once.
set -euo pipefail
cd "$(dirname "$0")/.."
source cluster/common.sh
load_env
sed -e "/^cd \"\$(dirname/d" -e "s#\.venv/bin/python#python#" scripts/slurm/prefetch.sh | bash -s -- h100 investigate
