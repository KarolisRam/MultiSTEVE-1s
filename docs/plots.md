# Paper Plot Suite

The paper figure suite is centralized under `multisteve.plot_suite`. The release includes cached summaries under `outputs/plot_suite/cache/` so paper-style PDFs can be regenerated quickly without raw rollout logs.

List enabled figure families:

```bash
conda run -n steve-1-creative env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
  python -m multisteve.plot_suite.cli list --config configs/plot_suite/final.json
```

Regenerate all figures from cache:

```bash
conda run -n steve-1-creative env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
  python -m multisteve.plot_suite.cli run \
  --from-cache \
  --config configs/plot_suite/final.json \
  --no-overleaf
```

Outputs are written to `outputs/plot_suite/runs/<run-name>/figures/`. Generated PDFs are intentionally not part of the source repository.

To create only the Overleaf-ready subset after regenerating figures, omit `--no-overleaf`; the selected PDFs are copied according to `configs/plot_suite/final.json` and the include list in `multisteve.plot_suite.config`.

## Cached Inputs

The cache contains CSV summaries for full evals, partial evals, tree prompt 100, villager passes, tree-vs-villager comparisons, bitflip summaries, head-fake stability, and T-maze scatter data. The only non-CSV caches are small `.npz` derived trajectory coordinate arrays under `outputs/plot_suite/cache/agent_trajectories/`; these contain numeric top-down coordinates, not frames, observations, videos, or model activations.

## Raw Result Rebuilds

If raw rollout folders are available, omit `--from-cache` and update `configs/plot_suite/final.json` or `multisteve.plot_suite.config.DEFAULT_CONFIG` to point at those raw result roots. Villager analysis helpers live under `tools/villager_analysis/`.

## Curated Paper Figures

The default Overleaf list is encoded in `multisteve/plot_suite/config.py`. The Overleaf folder is generated locally and is not included in the repository.

The current paper figure set covered by the cached plot suite includes:

- Full evaluation prompt figures: `text_dirt`, `text_logs`, `text_max_distance`, `text_seeds`, `visual_dirt`, `visual_logs`, `visual_max_distance`, `visual_seeds`.
- Partial evaluation figures for groups 01-04 and checkpoint-over-time reward evolution.
- Group-05 tree-prompt 100-agent log collection.
- T-maze choice scatters for seeds 49 and 481, with and without confidence intervals.
- Agent-131 and Agent-132 seed-49 top-down T-maze trajectories from derived coordinate caches.
- Agent-111/114 villager seed sweep, 100-agent villager damage/kill-rate sweeps, bitflip villager plots, and head-fake stability.

Paper screenshots that are not generated figure outputs, such as raw first-frame images, attention dashboard screenshots, 3D viewer screenshots, and seed-grid images, are external paper artifacts. The release includes source hooks to regenerate them when the corresponding raw rollout/world artifacts are available.

## Group-06 Coverage

The camera-ready release adds 100 group-06 bit-flip models. Existing cached figures retain their original measured populations; the caches do not provide group-06 evaluation results. Generate new rollouts and summaries before making group-06 performance plots. The new evaluation recipes are documented in [Model zoo](model_zoo.md).
