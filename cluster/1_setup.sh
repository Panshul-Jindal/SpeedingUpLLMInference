#!/bin/bash
# STEP 1 (login node, once): add the study's Python packages to the llm_haskell conda env.
#
# - If vLLM is NOT in llm_haskell yet, installs the versions the study pins (vllm 0.31.0, torch 2.13.0).
# - If vLLM IS already there, leaves torch/vllm/transformers exactly as they are and only adds the small
#   harness dependencies (lm-eval, statsmodels, ...). Set FORCE_PINNED=1 to install the pinned versions anyway.
set -euo pipefail
cd "$(dirname "$0")/.."
source cluster/common.sh
load_env

python -c 'import sys; v=sys.version_info[:2]; print(f"[setup] Python {v[0]}.{v[1]}"); assert v >= (3, 10), "need Python >= 3.10"'

INDEX=(--extra-index-url https://wheels.vllm.ai/0.31.0/cu129
       --extra-index-url https://download.pytorch.org/whl/cu129
       --extra-index-url https://flashinfer.ai/whl/cu129)

if ! python -c 'import vllm' 2>/dev/null || [ "${FORCE_PINNED:-0}" = 1 ]; then
    echo "[setup] installing pinned vLLM 0.31.0 / torch 2.13.0 (CUDA 12.9 builds) into $CONDA_ENV"
    python -m pip install "${INDEX[@]}" \
        "vllm==0.31.0+cu129" "torch==2.13.0+cu129" "torchcodec==0.16.0+cu129" \
        "flashinfer-jit-cache==0.7.0.post1+cu129" "flashinfer-cubin==0.7.0.post1"
fi

# Keep whatever torch/vllm/transformers are installed now; only add what is missing around them.
CONSTRAINTS="$(mktemp)"; trap 'rm -f "$CONSTRAINTS"' EXIT
python -m pip freeze | grep -iE '^(torch|torchvision|torchaudio|vllm|transformers|flashinfer[a-z_-]*)==' > "$CONSTRAINTS" || true
echo "[setup] keeping: $(tr '\n' ' ' < "$CONSTRAINTS")"
python -m pip install -c "$CONSTRAINTS" \
    "lm-eval[api]" nvidia-ml-py aiohttp "pydantic>=2" pyyaml numpy pandas pyarrow statsmodels \
    prometheus-client psutil huggingface_hub pytest pytest-asyncio

python - <<'PY'
import torch, vllm, lm_eval, statsmodels, pydantic
print(f"[setup] OK: torch {torch.__version__} (CUDA {torch.version.cuda}), vllm {vllm.__version__}")
if not vllm.__version__.startswith("0.31.0"):
    print("[setup] NOTE: the study was built for vllm 0.31.0; doctor will check that every flag it uses exists.")
PY
VER=$(python -c "import vllm; print(vllm.__version__)")
mkdir -p docs && vllm serve --help=all > "docs/vllm_serve_help_${VER}.txt" 2>&1 || true
echo "[setup] done"
