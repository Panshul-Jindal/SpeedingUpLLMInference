#!/bin/bash
# STEP 3 (login node, recommended, ~5 min): environment check on the GPU node. Must end with "doctor: PASS".
# If your cluster needs a GPU request:  GRES=gpu:2 cluster/3_check_gpu.sh
set -euo pipefail
cd "$(dirname "$0")/.."
srun -p gpu -w ccs-srv004 -c 4 --mem 32G -t 00:20:00 ${GRES:+--gres=$GRES} bash -c '
    source cluster/common.sh && load_env && export CUDA_DEVICE_ORDER=PCI_BUS_ID &&
    python -m harness doctor --env h100'
