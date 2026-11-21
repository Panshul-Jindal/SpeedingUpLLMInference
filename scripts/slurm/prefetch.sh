#!/bin/bash
# Download the models and the quality datasets into HF_HOME before submitting, so job time is not spent on
# downloads (and so jobs can run with HF_HUB_OFFLINE=1 if compute nodes have no internet).
# Run on the login node from the repo root, after scripts/setup_env.sh:
#   scripts/slurm/prefetch.sh [ENV_NAME] [PROFILE]      # defaults: h100 fast
set -euo pipefail
cd "$(dirname "$0")/../.."
ENV_NAME="${1:-h100}"
PROFILE="${2:-fast}"
export HF_HOME="${HF_HOME:-$HOME/gemma/hf-cache}"
mkdir -p "$HF_HOME"
echo "HF_HOME=$HF_HOME"

.venv/bin/python - "$ENV_NAME" "$PROFILE" <<'PY'
import sys
from huggingface_hub import snapshot_download
from harness.config import load_settings

s = load_settings(sys.argv[2], sys.argv[1])
env, q = s.env, s.experiment.quality
repos = [env.models.target]
spec_on = s.profile.optimizations is None or "spec" in s.profile.optimizations
if spec_on and env.speculative.method == "eagle3" and env.models.eagle3_draft:
    repos.append(env.models.eagle3_draft)
if env.quantization_on == "awq_int4" and env.models.awq_int4:
    repos.append(env.models.awq_int4)
for repo in repos:
    print(f"[prefetch] model {repo}")
    snapshot_download(repo, allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.jinja"])

from lm_eval.tasks import TaskManager, get_task_dict
tasks = [t for t in q.tasks if t != "mmlu"] + (list(q.mmlu_subtasks) if "mmlu" in q.tasks else [])
print(f"[prefetch] lm-eval datasets: {tasks}")
get_task_dict(tasks, TaskManager())
print("[prefetch] done")
PY
