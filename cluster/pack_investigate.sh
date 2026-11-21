#!/bin/bash
# When the investigation run is finished: bundle its results + logs to send back.
# Traces are left out (regenerated from the seed); add NO_PROFILES=1 if the archive gets too big.
set -euo pipefail
cd "$(dirname "$0")/.."
EXTRA=()
[ "${NO_PROFILES:-0}" = 1 ] && EXTRA=(--exclude='results/investigate/profiles')
tar czf llm-serving-investigate-results.tar.gz --exclude='results/investigate/traces' ${EXTRA[@]+"${EXTRA[@]}"} \
    results/investigate logs configs/tuned_fast.yaml configs/env/h100.yaml
ls -lh llm-serving-investigate-results.tar.gz
