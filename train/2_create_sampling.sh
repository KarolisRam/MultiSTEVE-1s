#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-steve-1-creative}"
cd "$REPO_ROOT"

conda run -n "$CONDA_ENV" env PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" python -m steve1.data.sampling.generate_sampling \
  --type neurips \
  --name "${SAMPLING_NAME:-neurips}" \
  --output_dir data/samplings/ \
  --val_frames "${VAL_FRAMES:-1000000}" \
  --train_frames "${TRAIN_FRAMES:--1}"
