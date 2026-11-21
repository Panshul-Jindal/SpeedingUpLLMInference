#!/bin/bash
# When finished: bundle results + logs to send back (a few hundred MB).
set -euo pipefail
cd "$(dirname "$0")/.."
tar czf llm-serving-results-h100.tar.gz --exclude='results/fast/traces' \
    results/fast logs configs/tuned_fast.yaml configs/env/h100.yaml
ls -lh llm-serving-results-h100.tar.gz
