# Seed Lists

Known paper seed lists are stored in `configs/seed_lists.json`.

- `tmaze_layout_seeds`: seeds 49 and 481 for the regular-tree versus villager-tree T-maze layout experiments.
- `villager_agent_111_114_seed_sweep`: ten seeds used for Agent 111 versus Agent 114 villager-tree seed sweeps.

The first-observation seed library contains images for over 17,000 environment seeds. These large image assets are external to the source repository.

The images are available in [MultiSTEVE-1s/seeds](https://huggingface.co/datasets/MultiSTEVE-1s/seeds). Download them from the repository root with the environment activated:

```bash
hf download MultiSTEVE-1s/seeds --repo-type dataset --local-dir data/seeds
```

The download retains the `batch_000/`–`batch_017/` directories and curated `flat-ground-for-T-maze/`, `nearly-random/`, and `villages/` directories. See the [project website](https://sites.google.com/view/multisteve-1s) for the project overview.
