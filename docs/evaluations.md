# Evaluations And Rollout Campaigns

Large rollout campaigns use `steve1.run_agent.episode_queue_orchestrator`, a SQLite-backed queue with pending/running/completed/failed states, worker leases, heartbeats, retry delays, and missing-weight retries.

Use the wrapper:

```bash
run_agent/episode_queue.sh enqueue --campaign configs/eval_campaigns/full_partial_eval.json --dry-run
run_agent/episode_queue.sh enqueue --campaign configs/eval_campaigns/full_partial_eval.json
run_agent/episode_queue.sh worker --campaign configs/eval_campaigns/full_partial_eval.json --worker-id worker-1
run_agent/episode_queue.sh status --campaign configs/eval_campaigns/full_partial_eval.json
```

Use [KarolisRam/minerl](https://github.com/KarolisRam/minerl/tree/dev), installed by `environment.yml`, for creative-command evaluations. See [installation instructions](data_and_weights.md) for a fresh environment.

Keep MineRL concurrency low by default. The release configs assume one MineRL evaluation at a time unless explicitly changed.

## Included Campaign Configs

- `configs/eval_campaigns/full_partial_eval.json`: full, partial, checkpoint-over-time, and tree-prompt evaluation campaigns.
- `configs/eval_campaigns/group05_tree_prompt_100.json`: 100-agent tree-prompt evaluation for agents 101-200.
- `configs/eval_campaigns/villager_group05_100_agents.json`: 100-agent villager-tree sweeps across logs-above true/false and short/engineered prompts.
- `configs/eval_campaigns/villager_bitflip10_engineered.json`: bitflip-10 villager-tree engineered-prompt sweep.

Full evaluations compare the original STEVE-1 checkpoint against the `dataset_seed1-weight_init_seed1` MultiSTEVE-1s checkpoint over 14 text prompts and 11 visual prompts, 20 environment seeds, and 1,000-frame episodes. Partial evaluations cover groups 01-04 using the dirt/wood visual prompts and engineered dirt/wood text prompts, again with 20 seeds and 1,000-frame episodes. Checkpoint-over-time partial evaluations use the same reduced prompt set over 28 checkpoints from one training run. The group-05 minimal tree-prompt evaluation covers agents 101-200 with `chop a tree` over 20 seeds.

Expected single-agent rollout hardware is roughly 3.5 GB GPU VRAM and about 3 CPU cores per MineRL instance. Scaling beyond one or two concurrent MineRL environments on one workstation tends to have poor returns; keep `max_parallel_minerl_evals` low unless you have verified local capacity.

The two very large repeated-rollout manifests are generated on demand:

```bash
conda run -n steve-1-creative env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
  python -m scripts.build_eval_campaigns \
  --campaign agent_111_114_seed_sweep \
  --out data/generated_campaigns/villager_agent_111_114_seed_sweep.json
```

The paper also reports the bitflip-2 versus bitflip-8 villager-tree sweep. Generate and run that manifest the same way:

```bash
conda run -n steve-1-creative env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
  python -m scripts.build_eval_campaigns \
  --campaign bitflip2_vs_8_sweep \
  --out data/generated_campaigns/villager_bitflip2_vs_8_sweep.json

run_agent/episode_queue.sh enqueue --campaign data/generated_campaigns/villager_bitflip2_vs_8_sweep.json --dry-run
run_agent/episode_queue.sh enqueue --campaign data/generated_campaigns/villager_bitflip2_vs_8_sweep.json
run_agent/episode_queue.sh worker --campaign data/generated_campaigns/villager_bitflip2_vs_8_sweep.json --worker-id worker-1
```

Use `--repeats 1` or another small value for smoke tests.

Generate the release smoke campaign for short path-coverage checks:

```bash
conda run -n steve-1-creative env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
  python -m scripts.build_eval_campaigns \
  --campaign release_smoke_17_episodes \
  --out data/generated_campaigns/release_smoke_17_episodes.json

run_agent/episode_queue.sh enqueue --campaign data/generated_campaigns/release_smoke_17_episodes.json --dry-run
```

This campaign is intentionally small: 17 short episodes covering text, visual, group-05 tree prompt, villager prompt/log settings, T-maze object orders, Agent-111/114 paths, bitflip-2/8 paths, and one attention-logging rollout. It checks code paths only.

## T-Maze Campaigns

Canonical T-maze campaign wrappers:

```bash
run_agent/build_tmaze_campaign.sh data/tmaze_100_model_campaign
run_agent/run_tmaze_campaign.sh data/tmaze_100_model_campaign --max_parallel 1
run_agent/analyze_tmaze_campaign.sh data/tmaze_100_model_campaign
run_agent/tmaze_campaign_control.sh data/tmaze_100_model_campaign status
```

The T-maze setup uses environment seeds 49 and 481, two object orders, a regular tree, a villager-tree, and group-05 agents 101-200. The source release does not include raw rollout videos or frames.

Each T-maze paper condition uses 100 very short rollouts of about three seconds per agent/layout. The scatter plots are regenerated from cached CSV summaries in `outputs/plot_suite/cache/teleport_layout/`; raw first frames and trajectory images are external rollout artifacts.

## Villager-Tree Campaigns

Villager-tree sweeps place a single villager under tree foliage, optionally with a log above the villager. The paper-relevant text prompts are:

- `chop a tree`
- `chop down the tree, gather wood, pick up wood, chop it down, break tree`

The 100-agent group-05 sweep covers four settings: logs above true/false crossed with short/engineered prompt. The Agent-111 versus Agent-114 seed sweep uses 10 environment seeds, villager-tree on left/right, `chop a tree`, logs above true, and 100 repeats per side/seed/agent for 2,000 episodes per agent. The bitflip-2 versus bitflip-8 sweep uses the same generator path with bitflip group-04 models.

Qualitative inspection artifacts, when generated, live next to each rollout: `video.mp4`, `first_frame_fullres.png`, `location_stats.log`, `hits.log`, and `kill_step.log`. These are intentionally excluded from the source repository.

## Smoke-Test Budget

A practical local smoke test is about 15-20 short MineRL episodes total:

- 3 episodes for full/partial/minimal evaluation runners.
- 4 episodes for villager chop settings.
- 4 episodes for T-maze object-order checks.
- 2 episodes for Agent-111/Agent-114 seed-sweep paths.
- 1 episode for attention logging.
- 1 episode for fake-env/action-probability replay if it cannot reuse an existing short rollout.

This smoke suite checks code paths only; it does not reproduce paper statistics.

## Camera-Ready Group 06

- `configs/eval_campaigns/group06_tree_prompt_100.json`: bit-flip seeds 101–200, 20 environment seeds, one short tree prompt (2,000 episodes).
- `configs/eval_campaigns/villager_group06_bitflip100_engineered.json`: the same 100 models, 20 passes, engineered wood prompt, both log-placement settings (4,000 episodes).

Use these configs to evaluate group 06. The bitflip-2/8 configs target group 04; group-05 configs target the models with both seeds varied. See `docs/model_zoo.md` for commands, including a group-06 T-maze campaign.
