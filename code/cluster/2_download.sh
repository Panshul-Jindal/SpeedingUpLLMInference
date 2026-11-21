#!/bin/bash
# STEP 2 (login node, once, ~10 min): download the model, the EAGLE-3 head and the eval datasets into HF_HOME.
# Run cluster/1_setup.sh first (needs lm_eval in the conda env). Safe to re-run: finished downloads are skipped.
set -euo pipefail
cd "$(dirname "$0")/.."
source cluster/common.sh
load_env
mkdir -p "$HF_HOME"
echo "HF_HOME=$HF_HOME"

if ! python -c 'import lm_eval, huggingface_hub' 2>/dev/null; then
    echo "ERROR: lm_eval is not installed in conda env '$CONDA_ENV'. Run cluster/1_setup.sh first, then re-run this." >&2
    exit 1
fi

# scripts/slurm/prefetch.sh hard-codes .venv/bin/python; run its body with the conda env's python instead.
sed -e "/^cd \"\$(dirname/d" -e "s#\.venv/bin/python#python#" scripts/slurm/prefetch.sh | bash -s -- h100 fast

# prefetch.sh only fetches *.safetensors weights, but the EAGLE-3 head ships pytorch_model.bin:
# fetch the head's weights explicitly, then check that every model has weights in the cache.
python - <<'PY'
from huggingface_hub import snapshot_download
from harness.config import load_settings

env = load_settings("fast", "h100").env
draft = env.models.eagle3_draft
if draft:
    print(f"[download] EAGLE-3 head weights: {draft}")
    snapshot_download(draft, allow_patterns=["*.json", "*.bin", "*.safetensors"])

import os
missing = []
for repo in filter(None, [env.models.target, draft]):
    path = snapshot_download(repo, local_files_only=False)
    weights = [f for f in os.listdir(path) if f.endswith((".safetensors", ".bin"))]
    print(f"[check] {repo}: {len(weights)} weight file(s)")
    if not weights:
        missing.append(repo)
if missing:
    raise SystemExit(f"[check] FAIL: no weights for {missing}")
print("[download] all done")
PY
