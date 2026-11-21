#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-steve-1-creative}"

OUTPUT_DIR_MIXED_AGENTS=data/dataset_mixed_agents/
N_EPISODES_MIXED_AGENTS="${N_EPISODES_MIXED_AGENTS:-2000}"

conda run -n "$CONDA_ENV" env PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" python -m steve1.data.generation.gen_mixed_agents \
  --output_dir "$OUTPUT_DIR_MIXED_AGENTS" \
  --max_timesteps 7200 \
  --min_timesteps 1000 \
  --switch_agent_prob 0.001 \
  --perform_spin_prob 0.00133 \
  --batch_size 4 \
  --num_episodes "$N_EPISODES_MIXED_AGENTS"
