#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-steve-1-creative}"

run_module() {
  conda run -n "$CONDA_ENV" env PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" python -m "$@"
}

OUTPUT_DIR_CONTRACTOR=data/dataset_contractor/
N_EPISODES_CONTRACTOR="${N_EPISODES_CONTRACTOR:-5}"

for index in 8.x 9.x 10.x; do
  run_module steve1.data.generation.convert_from_contractor \
    --batch_size 32 \
    --num_episodes "$N_EPISODES_CONTRACTOR" \
    --worker_id 0 \
    --output_dir "$OUTPUT_DIR_CONTRACTOR" \
    --index "$index"
done

OUTPUT_DIR_MIXED_AGENTS=data/dataset_mixed_agents/
N_EPISODES_MIXED_AGENTS="${N_EPISODES_MIXED_AGENTS:-5}"

run_module steve1.data.generation.gen_mixed_agents \
  --output_dir "$OUTPUT_DIR_MIXED_AGENTS" \
  --max_timesteps 7200 \
  --min_timesteps 1000 \
  --switch_agent_prob 0.001 \
  --perform_spin_prob 0.00133 \
  --batch_size 4 \
  --num_episodes "$N_EPISODES_MIXED_AGENTS"
