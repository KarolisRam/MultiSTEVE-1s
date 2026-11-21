# Model Zoo

MultiSTEVE-1s releases 240 agents and 1,520 checkpoint files. The camera-ready release adds group 06: 100 models trained with different single least-significant-bit flips before training, while holding dataset and weight-initialization seeds fixed at 1.

## Groups

| Folder | Runs | Dataset seed | Weight-init seed | Bit-flip seed | Released snapshots per run |
| --- | ---: | --- | --- | --- | ---: |
| `01_init_seed_only` | 10 | 1 | 1–10 | None | 28 |
| `02_data_seed_only` | 10 | 1–10 | 1 | None | 28 |
| `03_init_and_data` | 10 | 1–10, paired | 1–10, paired | None | 28 |
| `04_single_lsb_flip` | 10 | 1 | 1 | 1–10 | 28 |
| `05_final_init_and_data_100` | 100 | 101–200, paired | 101–200, paired | None | 2 |
| `06_final_single_lsb_flip_100` | 100 | 1 | 1 | 101–200 | 2 |

Counts describe released runs, including the identical baseline controls in groups 01–03. Groups 01–04 contribute 1,120 checkpoint files; groups 05–06 contribute 400. Group 06 extends group 04; it does not replace it. For example, bit-flip seed 101 is distinct from group-05 Agent-101, which varies both dataset and initialization seeds.

## Artifact Layout

All six groups are published in [MultiSTEVE-1s/MultiSTEVE-1s](https://huggingface.co/MultiSTEVE-1s/MultiSTEVE-1s), preserving the group/run folder hierarchy. Use the [Hugging Face download commands](data_and_weights.md#download-from-hugging-face) for one checkpoint, one group, or the complete release. The code expects the groups under `data/model_zoo/`:

```text
data/model_zoo/06_final_single_lsb_flip_100/
  dataset_seed1-weight_init_seed1-bitflip_seed101/
    STEVE-1_snapshot_000000000.weights
    STEVE-1_snapshot_160005120.weights
    STEVE-1_data_order.txt
    train_<job_id>.out
    train_<job_id>.err
  ...
  dataset_seed1-weight_init_seed1-bitflip_seed200/
    ...
```

Each group-06 run includes both snapshots. `000000000` is the initial checkpoint; `160005120` is the final checkpoint after approximately 160M frames. Log filenames depend on the training job and are not required for inference. No large model files are bundled in this code repository.

## Training

Print the 100 group-06 training commands from the repository root:

```bash
conda run -n steve-1-creative env PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
  python -m scripts.train_zoo --group 06
```

`--group all` includes all 240 runs. Group-06 commands use `--dataset_seed 1 --weight_init_seed 1 --bitflip --bitflip_seed <101..200>`. The shared training configuration can generate intermediate checkpoints; the released group-06 artifacts retain only the initial and final snapshots. See [Training](training.md) for hyperparameters and reproducibility limitations.

## Evaluation

Use the following recipes to evaluate group 06.

Tree-chopping evaluation (100 models × 20 environment seeds = 2,000 episodes):

```bash
run_agent/episode_queue.sh enqueue --campaign configs/eval_campaigns/group06_tree_prompt_100.json --dry-run
run_agent/episode_queue.sh enqueue --campaign configs/eval_campaigns/group06_tree_prompt_100.json
run_agent/episode_queue.sh worker --campaign configs/eval_campaigns/group06_tree_prompt_100.json --worker-id worker-1
```

Villager-tree evaluation with the engineered prompt (100 models × 20 passes × 2 log-placement settings = 4,000 episodes):

```bash
run_agent/episode_queue.sh enqueue --campaign configs/eval_campaigns/villager_group06_bitflip100_engineered.json --dry-run
run_agent/episode_queue.sh enqueue --campaign configs/eval_campaigns/villager_group06_bitflip100_engineered.json
run_agent/episode_queue.sh worker --campaign configs/eval_campaigns/villager_group06_bitflip100_engineered.json --worker-id worker-1
```

Use the existing T-maze wrapper with an explicit group-06 weights root and a separate campaign output directory:

```bash
run_agent/build_tmaze_campaign.sh data/tmaze_group06_campaign \
  data/model_zoo/06_final_single_lsb_flip_100
run_agent/run_tmaze_campaign.sh data/tmaze_group06_campaign --max_parallel 1
run_agent/analyze_tmaze_campaign.sh data/tmaze_group06_campaign
```

The paper caches cover groups 01–05. Group-06 plots require new evaluation results. The `release_smoke_17_episodes` recipe covers groups 01–05; use the group-06 campaigns above to evaluate the additional bit-flip models.

## Ongoing Expansion

Additional bit-flip runs are published separately in [MultiSTEVE-1s/200bitflips](https://huggingface.co/MultiSTEVE-1s/200bitflips). This expanding collection is not included in the 240-agent/1,520-checkpoint totals or the six-group training and evaluation recipes above. Its run directories sit at the repository root; see [Data and weights](data_and_weights.md#download-from-hugging-face) for a separate download destination.
