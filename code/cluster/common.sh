# Shared settings + environment loading for run_study.sbatch and the cluster/ scripts.
# Edit the values below if your cluster differs (or override them in the environment).
ANACONDA_MODULE="${ANACONDA_MODULE:-Anaconda3/}"
CUDA_MODULE="${CUDA_MODULE:-cuda-13.1}"
CONDA_ENV="${CONDA_ENV:-llm_haskell}"                 # Python 3.10
export HF_HOME="${HF_HOME:-$HOME/gemma/hf-cache}"    # model/dataset cache, same for download and jobs
# lm-eval's HumanEval task runs model-generated code; it refuses to load without this (the study needs it).
export HF_ALLOW_CODE_EVAL=1

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# The harness is run as `python -m harness` from the repo root (not pip-installed: pyproject says 3.11).
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

load_env() {
    set +u   # module/conda init scripts reference unset variables
    # Batch shells may not define `module`; source the usual init scripts if needed.
    if ! type module >/dev/null 2>&1; then
        for f in /etc/profile.d/modules.sh /etc/profile.d/lmod.sh /usr/share/lmod/lmod/init/bash \
                 /usr/share/Modules/init/bash; do
            [ -f "$f" ] && source "$f" && break
        done
    fi
    module load "$ANACONDA_MODULE"   || { echo "[env] module load $ANACONDA_MODULE failed" >&2; return 1; }
    eval "$(conda shell.bash hook)"  || { echo "[env] conda not available" >&2; return 1; }
    conda activate "$CONDA_ENV"      || { echo "[env] conda activate $CONDA_ENV failed" >&2; return 1; }
    module load "$CUDA_MODULE"       || { echo "[env] module load $CUDA_MODULE failed" >&2; return 1; }
    # (cuda loaded after conda so the module's nvcc comes first on PATH)
    set -u
    echo "[env] python $(python -c 'import platform; print(platform.python_version())') ($(command -v python))"
    echo "[env] nvcc   $(nvcc --version 2>/dev/null | sed -n 's/.*release \([0-9.]*\),.*/\1/p') ($(command -v nvcc || echo 'not found'))"
}
