# Reproducibility Notes

Run all Python commands from the repository root with this checkout first on `PYTHONPATH`:

```bash
conda run -n steve-1-creative env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
  python -m multisteve.plot_suite.cli list
```

Avoid `python path/to/script.py` for repo modules; use `python -m package.module` or the provided shell wrappers.

## What Is Reproducible From The Repository Alone

- Training command generation for all 240 model-zoo agents.
- Queue manifest generation.
- Cached paper figure regeneration.
- Import/help-level checks for rollout, plotting, and analysis modules.

## What Requires External Artifacts

- Full training requires datasets, VPT weights, MineCLIP weights, and GPUs.
- Evaluation campaigns require trained model weights and MineRL runtime support.
- Attention dashboards require generated attention arrays and observations.
- Interactive scatter/world viewers require generated rollout frame bundles or world exports.

## Known Limits

Training bitwise identity is hardware- and software-sensitive. CUDA, GPU model, PyTorch, and driver differences can change floating-point behavior. The model zoo did not rely on resume checkpoints for the released runs, though resumable checkpointing exists in the training code.

MineRL evaluations also require a working Minecraft display/runtime setup. If running on a headless server, use the same MineRL/Xvfb/display setup you normally use for STEVE-1 rollouts before launching queue workers.

Creative-command rollouts, including the T-maze and villager-tree placement helpers, use [KarolisRam/minerl, branch `dev`](https://github.com/KarolisRam/minerl/tree/dev), which adds chat-command support. `environment.yml` pins this fork to commit `d3d51b403b8750823fe6bfe3e82af8d80a8344dd` and MineCLIP to commit `e6c06a0245fac63dceb38bc9bd4fecd033dae735`. See [Data and weights](data_and_weights.md) for fresh installation instructions. The release environment uses pinned Git revisions; record any deviations when reproducing experiments. The package version alone (`minerl==1.0.0` in the recorded package list) does not identify the fork or commit.

## Licensing And Bundled Assets

This source release does not bundle model weights, Minecraft assets, MineRL worlds, generated videos, generated dashboard assets, or hosted-demo runtime bundles. It does include source code that imports STEVE-1/VPT/MineRL/MineDojo-related dependencies and can optionally fetch or embed JavaScript runtimes when generating local viewers. The upstream STEVE-1 repository did not include a `LICENSE` file when checked for this release, so the top-level `LICENSE` file covers only the new MultiSTEVE-specific release additions and does not relicense third-party/upstream code.
