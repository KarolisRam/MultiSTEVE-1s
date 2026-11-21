#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

CONDA_ENV="${CONDA_ENV:-steve-1-creative}"

if [[ $# -lt 1 ]]; then
  cat >&2 <<'EOF'
usage: run_agent/episode_queue.sh enqueue|worker|status|reset [orchestrator args]

examples:
  run_agent/episode_queue.sh enqueue --campaign configs/eval_campaigns/full_partial_eval.json --dry-run
  run_agent/episode_queue.sh enqueue --campaign configs/eval_campaigns/full_partial_eval.json
  run_agent/episode_queue.sh worker --campaign configs/eval_campaigns/full_partial_eval.json --worker-id worker-1
  run_agent/episode_queue.sh status --campaign configs/eval_campaigns/full_partial_eval.json
EOF
  exit 2
fi

conda run --no-capture-output -n "${CONDA_ENV}" env \
  "PYTHONPATH=${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
  python -m steve1.run_agent.episode_queue_orchestrator "$@"
