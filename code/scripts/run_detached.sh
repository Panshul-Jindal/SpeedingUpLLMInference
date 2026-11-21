#!/usr/bin/env bash
# Run a harness command in a detached tmux session so it survives SSH/terminal disconnects.
#   scripts/run_detached.sh <session-name> <harness args...>
#   e.g. scripts/run_detached.sh smoke smoke --env test
# Re-attach: tmux attach -t <session-name>   (detach again: Ctrl-b d)
# Output is also written to logs/<session-name>.log; the exit code is appended as EXIT=<n>.
set -euo pipefail
cd "$(dirname "$0")/.."
name="$1"; shift
mkdir -p logs
if tmux has-session -t "$name" 2>/dev/null; then
  echo "tmux session '$name' already exists (tmux attach -t $name)" >&2; exit 1
fi
cmd="cd $(pwd) && source .venv/bin/activate && python -u -m harness $* 2>&1 | tee -a logs/$name.log; echo EXIT=\${PIPESTATUS[0]} >> logs/$name.log"
tmux new-session -d -s "$name" "bash -c '$cmd'"
echo "started '$name': tmux attach -t $name   |   tail -f logs/$name.log"
