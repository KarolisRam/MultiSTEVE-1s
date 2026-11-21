#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 CAMPAIGN_ROOT [extra analyzer args]" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

CAMPAIGN_ROOT="$1"
shift

conda run -n steve-1-creative env "PYTHONPATH=${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
  python -m steve1.run_agent.analyze_teleport_campaign \
  --repo_root "${REPO_ROOT}" \
  --campaign_root "${CAMPAIGN_ROOT}" \
  "$@"
