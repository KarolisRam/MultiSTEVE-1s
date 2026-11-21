# Training

The model zoo has 240 STEVE-1 agents across six randomness groups:

- `01_init_seed_only`: `dataset_seed=1`, `weight_init_seed=1..10`.
- `02_data_seed_only`: `dataset_seed=1..10`, `weight_init_seed=1`.
- `03_init_and_data`: paired `dataset_seed=1..10`, `weight_init_seed=1..10`.
- `04_single_lsb_flip`: parent `dataset_seed=1`, `weight_init_seed=1`, with `bitflip_seed=1..10`.
- `05_final_init_and_data_100`: paired `dataset_seed=101..200`, `weight_init_seed=101..200`.
- `06_final_single_lsb_flip_100`: parent `dataset_seed=1`, `weight_init_seed=1`, with `bitflip_seed=101..200` (100 additional camera-ready models).

Group 06 uses the same single least-significant-bit perturbation mechanism as group 04. The bit-flip seed selects the perturbed weight; it does not change the dataset or weight-initialization seed. See [Model zoo](model_zoo.md) for released checkpoint names and counts.

The embedded datasets are available from [MultiSTEVE-1s/steve1-training-data](https://huggingface.co/datasets/MultiSTEVE-1s/steve1-training-data). See [download instructions](data_and_weights.md#download-from-hugging-face); downloaded episodes already include MineCLIP embeddings. Create sampling files after downloading the embedded data:

```bash
train/2_create_sampling.sh
```

This writes `data/samplings/neurips_train.txt` and `neurips_val.txt`, matching `scripts.train_zoo` and `train/3_train.sh`. By default, it uses all available frames apart from a validation holdout targeting 1M frames; splits contain whole episodes, so actual counts vary. This creates a new deterministic split from the locally available episodes, rather than downloading the original split files.

For a smaller training run, use a separate sampling name:

```bash
SAMPLING_NAME=neurips-small TRAIN_FRAMES=3000000 train/2_create_sampling.sh
SAMPLING_NAME=neurips-small train/3_train.sh
```

`VAL_FRAMES` adjusts the validation target. `train/3_train.sh` runs a short 100K-frame training job; use the generated model-zoo commands for full 160M-frame runs.

Print all training commands:

```bash
conda run -n steve-1-creative env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
  python -m scripts.train_zoo --group all
```

Print only one group:

```bash
conda run -n steve-1-creative env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
  python -m scripts.train_zoo --group 06
```

The shared hyperparameters are encoded in `scripts.train_zoo`: `T=640`, `trunc_t=64`, `batch_size=12`, `gradient_accumulation_steps=4`, `n_frames=160_000_000`, `learning_rate=4e-5`, `warmup_frames=10_000_000`, `weight_decay=0.039428`, `sampling=neurips`, checkpoint every 10M frames, resumable checkpoint every 20M frames, and validation every 10k frames.

The generator emits the same pattern as the paper training command, with per-run output paths under `data/model_zoo/<group>/<run_name>/`:

```text
STEVE-1.weights
checkpoint_dir/
```

Shareable checkpoints are emitted according to `--shareable_checkpoint_every_n_frames 10_000_000`; resumable checkpoints use `--resumable_checkpoint_every_n_frames 20_000_000`. The release retains 28 snapshots per run in groups 01–04 and only the initial and final snapshots in groups 05–06; generated intermediate snapshots need not all be uploaded. Training logs, data-order logs, and any WandB logs are runtime outputs and are not bundled in the source release.

The primary training module is:

```bash
conda run -n steve-1-creative env PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
  python -m steve1.training.train --help
```

Training requires a sufficiently large GPU. `scripts.train_zoo` prints commands without starting training.

## Reproducibility Checks

The training code preserves separate randomness controls for weight initialization, data ordering/goals/dropout, bitflip selection, and global seed invariance. To audit trained checkpoints:

```bash
conda run -n steve-1-creative env PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
  python -m multisteve.compare_weight_init --help
```

Use `multisteve.measure_model_l2` and `multisteve.measure_model_l2_matrix` for checkpoint distance checks across groups. CUDA, driver, and PyTorch differences can break bitwise identity even with the same seeds.

The model zoo contains duplicate `dataset_seed1-weight_init_seed1` runs across groups 01-03. Those are the bitwise-identical training controls when run on matching hardware/software. Initial-checkpoint checks should verify:

- Identical weight-initialization seeds produce identical initial MineCLIP-layer weights.
- Different weight-initialization seeds differ only in the initialized MineCLIP layer at step zero.
- Bitflip runs differ from the parent `dataset_seed1-weight_init_seed1` initialization in exactly one selected MineCLIP-layer weight.

The released training runs did not rely on resumed training, even though the code can write resumable checkpoints. This avoids a known source of occasional bitwise divergence in resumed runs.
