#!/usr/bin/env bash
# Create .venv (Python 3.11) and install the pinned requirements (CUDA 12.9 builds).
# Idempotent: safe to re-run. Needs no root.
set -euo pipefail
cd "$(dirname "$0")/.."

MIN_DRIVER_MAJOR=570   # cu129 wheels need driver >= 570 (CUDA 12.9 forward-compat floor 575 recommended)

export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
  echo "[setup] installing uv to ~/.local/bin"
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  drv=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
  echo "[setup] NVIDIA driver $drv"
  if [ "${drv%%.*}" -lt "$MIN_DRIVER_MAJOR" ]; then
    echo "[setup] WARNING: driver $drv is older than $MIN_DRIVER_MAJOR; the cu129 torch/vLLM builds may not load." >&2
  fi
else
  echo "[setup] WARNING: nvidia-smi not found (fine for CPU-only unit tests / dry runs)" >&2
fi

uv python install 3.11
[ -d .venv ] || uv venv .venv --python 3.11
uv pip sync requirements.txt --python .venv/bin/python --index-strategy unsafe-best-match
uv pip install --python .venv/bin/python --no-deps -e .

# FlashInfer JIT-compiles some kernels (e.g. FP8-KV XQA decode) and needs nvcc + CUDA headers that match
# the cu129 wheels. Unless a CUDA 12.x nvcc is already on PATH, install a user-level CUDA 12.9 toolchain from
# conda-forge (no root). A cluster's cuda-13 module does not count: do not load it for these jobs.
sys_nvcc_major=$(nvcc --version 2>/dev/null | sed -n 's/.*release \([0-9]*\)\..*/\1/p' || true)
if [ "$sys_nvcc_major" != "12" ] && [ ! -x .cuda-toolkit/bin/nvcc ]; then
  echo "[setup] no CUDA 12 nvcc (found: ${sys_nvcc_major:-none}); installing CUDA 12.9 toolchain into .cuda-toolkit"
  MM="$HOME/.local/bin/micromamba"
  if [ ! -x "$MM" ]; then
    mkdir -p "$HOME/.local/bin"
    curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C "$HOME/.local" bin/micromamba
  fi
  "$MM" create -y -q -r "$HOME/.micromamba" -p "$(pwd)/.cuda-toolkit" -c conda-forge \
    "cuda-nvcc=12.9.86" "cuda-cudart-dev=12.9" "cuda-cccl=12.9" "cuda-nvrtc-dev=12.9" \
    "libcublas-dev=12.9" "cuda-profiler-api=12.9"
fi
# FlashInfer links with -L$CUDA_HOME/lib64; conda puts libraries in lib/.
[ -d .cuda-toolkit ] && [ ! -e .cuda-toolkit/lib64 ] && ln -s lib .cuda-toolkit/lib64

.venv/bin/python - <<'PY'
import torch, vllm
print(f"[setup] torch {torch.__version__} (CUDA {torch.version.cuda}), vllm {vllm.__version__}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"[setup] GPU{i}: {torch.cuda.get_device_name(i)} cc={torch.cuda.get_device_capability(i)}")
PY
VER=$(.venv/bin/python -c "import vllm; print(vllm.__version__)")
.venv/bin/vllm serve --help=all > "docs/vllm_serve_help_${VER}.txt" 2>&1 || true
echo "[setup] done. Activate with: source .venv/bin/activate"
