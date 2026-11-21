#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-steve-1-creative}"
cd "$REPO_ROOT"

conda run -n "$CONDA_ENV" env PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" python -m steve1.data.text_alignment.vae_pipeline.train_vae \
  --data_path data/prior_dataset/data.pkl \
  --output_path data/weights/vae/trained_vae.pt
