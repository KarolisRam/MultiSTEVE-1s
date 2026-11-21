#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

OUT_ROOT="${1:-data/teleport_100_model_campaign_$(date +%Y%m%d-%H%M%S)}"
shift || true

WEIGHTS_ROOTS=("${MULTISTEVE_MODEL_ZOO_ROOT:-data/model_zoo}/05_final_init_and_data_100")
if [[ $# -gt 0 ]]; then
  WEIGHTS_ROOTS=("$@")
fi

cmd=(
  conda run -n steve-1-creative env "PYTHONPATH=${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" python -m steve1.run_agent.import_teleport_campaign
  --repo_root "${REPO_ROOT}"
  --campaign_root "${OUT_ROOT}"
)
for root in "${WEIGHTS_ROOTS[@]}"; do
  cmd+=(--weights_root "${root}")
done

"${cmd[@]}"
