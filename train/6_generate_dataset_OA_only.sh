#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-steve-1-creative}"

OUTPUT_DIR_CONTRACTOR=data/dataset_contractor/
N_EPISODES_CONTRACTOR="${N_EPISODES_CONTRACTOR:-2700}"

for index in 8.x 9.x 10.x; do
  conda run -n "$CONDA_ENV" env PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" python -m steve1.data.generation.convert_from_contractor \
    --batch_size 32 \
    --num_episodes "$N_EPISODES_CONTRACTOR" \
    --worker_id 0 \
    --output_dir "$OUTPUT_DIR_CONTRACTOR" \
    --index "$index"
done
