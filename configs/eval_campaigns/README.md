# Evaluation Campaign Configs

These JSON files are inputs to `steve1.run_agent.episode_queue_orchestrator`, launched through:

```bash
run_agent/episode_queue.sh enqueue --campaign configs/eval_campaigns/full_partial_eval.json --dry-run
run_agent/episode_queue.sh enqueue --campaign configs/eval_campaigns/full_partial_eval.json
run_agent/episode_queue.sh worker --campaign configs/eval_campaigns/full_partial_eval.json --worker-id worker-1
run_agent/episode_queue.sh status --campaign configs/eval_campaigns/full_partial_eval.json
```

Included compact configs:

- `full_partial_eval.json`: full, partial, checkpoint-over-time, and tree-prompt evaluation campaigns.
- `group05_tree_prompt_100.json`: 100-agent group-05 tree-chop minimal evaluation.
- `villager_group05_100_agents.json`: 100-agent villager-tree sweep across logs-above and prompt settings.
- `villager_bitflip10_engineered.json`: bitflip-10 villager-tree engineered-prompt sweep.

Large repeated-rollout manifests are generated instead of stored:

```bash
conda run -n steve-1-creative env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
  python -m scripts.build_eval_campaigns \
  --campaign agent_111_114_seed_sweep \
  --out data/generated_campaigns/villager_agent_111_114_seed_sweep.json

conda run -n steve-1-creative env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
  python -m scripts.build_eval_campaigns \
  --campaign bitflip2_vs_8_sweep \
  --out data/generated_campaigns/villager_bitflip2_vs_8_sweep.json

conda run -n steve-1-creative env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
  python -m scripts.build_eval_campaigns \
  --campaign release_smoke_17_episodes \
  --out data/generated_campaigns/release_smoke_17_episodes.json
```

Use `--repeats` to make a smaller smoke-test version.

## Camera-Ready Group 06

- `configs/eval_campaigns/group06_tree_prompt_100.json`: bit-flip seeds 101–200, 20 environment seeds, one short tree prompt (2,000 episodes).
- `configs/eval_campaigns/villager_group06_bitflip100_engineered.json`: the same 100 models, 20 passes, engineered wood prompt, both log-placement settings (4,000 episodes).

Use these configs to evaluate group 06. The bitflip-2/8 configs target group 04; group-05 configs target the models with both seeds varied. See `docs/model_zoo.md` for commands, including a group-06 T-maze campaign.
