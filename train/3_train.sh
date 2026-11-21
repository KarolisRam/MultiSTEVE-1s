#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-steve-1-creative}"
cd "$REPO_ROOT"

checkpoint_dir="${CHECKPOINT_DIR:-data/training_checkpoint}"
timestamp="$(date +"%Y-%m-%d_%H-%M-%S")"
log_file="$checkpoint_dir/train_$timestamp.log"
mkdir -p "$checkpoint_dir"

export PYTHONUNBUFFERED=1
conda run -n "$CONDA_ENV" env PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
  accelerate launch --num_processes 1 --mixed_precision bf16 --module steve1.training.train \
  --in_model data/weights/vpt/2x.model \
  --in_weights data/weights/vpt/rl-from-foundation-2x.weights \
  --out_weights "$checkpoint_dir/trained_with_script.weights" \
  --trunc_t 64 \
  --T 640 \
  --batch_size 2 \
  --gradient_accumulation_steps 4 \
  --num_workers 1 \
  --weight_decay 0.039428 \
  --n_frames 100_000 \
  --learning_rate 4e-5 \
  --warmup_frames 10_000 \
  --p_uncond 0.1 \
  --min_btwn_goals 15 \
  --max_btwn_goals 200 \
  --checkpoint_dir "$checkpoint_dir" \
  --val_freq 1000000 \
  --val_freq_begin 1000000 \
  --val_freq_switch_steps 50000 \
  --val_every_nth 1000 \
  --save_each_val False \
  --sampling "${SAMPLING_NAME:-neurips}" \
  --sampling_dir data/samplings/ \
  --shareable_checkpoint_every_n_frames 50_000_000 |& tee "$log_file"
