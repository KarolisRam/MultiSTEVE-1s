# Interpretability Suite

To explore recorded experiments without installing the code or downloading model weights, use the [attention demo](https://multisteve-1s.netlify.app/) or download the [interpretability bundle](https://huggingface.co/datasets/MultiSTEVE-1s/interactive-explorers). The bundle includes an attention-pattern viewer, a 3D T-maze trajectory viewer, two scatter explorers, and their local data. Extract the entire ZIP and open `index.html` in a desktop browser.

The source release includes code paths for attention logging, action-probability replay, T-maze construction, top-down trajectory plotting, interactive scatter generation hooks, and world-viewer bundle generation. It excludes generated dashboards, frames, videos, attention arrays, observations archives, and hosted-demo assets.

## Attention Logging

Rollout runners support logging extra artifacts when `--save_additional_data` and attention/action-probability options are enabled. Relevant modules:

- `steve1.run_agent.run_agent_with_logging`
- `steve1.run_agent.run_agent_with_logging_teleport`
- `steve1.run_agent.run_agent_with_logging_batch_fake_env`

Typical generated artifacts are `attn_patterns.npy`, `observations.zip`, action arrays, camera arrays, inventory logs, location logs, and hit/kill logs. Those artifacts are intentionally external to the source repository.

Viewer builders live in `tools/attention_dashboard/`. The multi-rollout builder recursively discovers complete episodes, including the nested model/prompt/seed layout produced by the queue:

```bash
conda run -n steve-1-creative env PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
  python -m tools.attention_dashboard.build_attention_hover_viewer \
  --episode-dir data/example_rollout/episode \
  --output-dir data/generated_viewers/single_rollout

conda run -n steve-1-creative env PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
  python -m tools.attention_dashboard.build_multi_rollout_attention_viewer_multi_action \
  --rollouts-root data/example_rollouts \
  --output-dir data/generated_viewers/multi_rollout
```

## Action Probabilities And Fake-Env Replay

Source-only tools live in:

```text
multisteve/action_probabilities/
tools/action_probability_analysis/
```

The fake-env workflow replays a fixed rollout while ignoring sampled actions and filling recurrent memory from the rollout sequence. This supports exact action-agreement and semantic button-agreement comparisons across nondeterminism sources.

Start from:

```bash
conda run -n steve-1-creative env PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
  python -m multisteve.action_probabilities.run_real_prompt_suite --help
```

Paper-scale action-probability comparisons use 14 fixed text-prompt rollouts of 1,000 actions from the `dataset_seed1-weight_init_seed1` checkpoint, then replay those observations through model groups while ignoring the sampled actions. Metrics include exact action agreement and semantic button agreement; comparison/rendering helpers are under `tools/action_probability_analysis/`.

## T-Maze And Villager-Tree Construction

Creative-command helpers are in `multisteve.mc_creative` and the teleport rollout path. T-maze rollouts are launched with `steve1.run_agent.run_agent_with_logging_teleport`, which places regular trees and villager-trees, can place logs above villagers, and can periodically teleport the agent back to spawn.

Use placement strings such as `side=left,villager=false`, `side=right,villager=true`, and `side=left,villager=true,logs_above_villager=true` through queue configs or direct runner invocations.

These creative-command placement workflows use [KarolisRam/minerl, branch `dev`](https://github.com/KarolisRam/minerl/tree/dev) for chat-command support. `environment.yml` installs the fork; see [Data and weights](data_and_weights.md) for fresh installation instructions.

## Top-Down Trajectories And Scatter Plots

Top-down trajectory figures and T-maze choice scatters are generated through the plot suite. Cached coordinate summaries are included, but raw first-frame images and per-rollout videos are external.

The source release includes two T-maze scatter paths:

- `steve1.run_agent.analyze_teleport_layout_scatter` rebuilds the original analysis CSV/PNG/HTML from raw left-layout and right-layout rollout summaries.
- `tools.interactive_scatter.build_tmaze_scatter_explorer` builds an interactive explorer from the cached scatter CSVs.

Generate the cache-based explorer:

```bash
conda run -n steve-1-creative env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
  python -m tools.interactive_scatter.build_tmaze_scatter_explorer \
  --out-html outputs/interactive/tmaze_choice_scatter_explorer.html \
  --show-ci-default
```

If raw rollout folders are available, pass `--artifact-root` to link or embed `trajectory_topdown_reset.png` and `first_frame_fullres.png` assets referenced by the cache CSV run-directory columns:

```bash
conda run -n steve-1-creative env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
  python -m tools.interactive_scatter.build_tmaze_scatter_explorer \
  --artifact-root /path/to/raw/artifact/root \
  --embed-assets \
  --out-html outputs/interactive/tmaze_choice_scatter_explorer_with_assets.html
```

## 3D World Viewer

`tools/world_viewer/` contains world extraction and isometric viewer scripts. The release does not bundle viewer build outputs, MineRL saves, or third-party runtime assets.

Example:

```bash
conda run -n steve-1-creative env PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
  python -m tools.world_viewer.build_interactive_isometric_html \
  --world data/minecraft_saves/example_world \
  --spawn-location-stats data/example_rollout/location_stats.log \
  --trajectory-location-stats data/example_rollout/location_stats.log \
  --out-html data/generated_viewers/world_view.html \
  --no-offline-bundle
```

## Paper-Discussed Attention Heads

The paper discusses attention heads 0.8 for inventory, 0.9 for previous-frame attention, 2.2 for camera-up/jump, 1.3 for rightward movement, and 1.14 for the frame-802 inventory-related attention line. Reproducing those examples requires the selected rollout observations and attention arrays, which are external to this source repository.
