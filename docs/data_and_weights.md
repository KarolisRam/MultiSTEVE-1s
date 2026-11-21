# Data And Weights

The source repository does not include model weights, datasets, raw rollout logs, videos, attention arrays, hosted-demo assets, or generated paper PDFs. Place external artifacts under `data/` using the paths below, or adapt configs to your storage layout.

## Fresh Installation

Install Conda, Java JDK 8, Git, and a C/C++ build toolchain (for MineRL and its native dependencies), then run these commands from the repository root:

```bash
conda env create -f environment.yml
conda activate steve-1-creative
```

`environment.yml` installs a pinned revision of [KarolisRam/minerl, branch `dev`](https://github.com/KarolisRam/minerl/tree/dev) directly, including the chat-command support required by creative-mode placement, T-maze, and villager-tree workflows. No prior MineRL installation or separate MineRL install command is required. The environment includes CUDA 11.7 PyTorch wheels, MineCLIP, and `gym3`. The packaging-tool and NumPy pins keep the legacy Gym/MineRL dependency stack compatible.

On a headless Linux machine, install Xvfb and prefix rollout commands with `xvfb-run -a`. A working NVIDIA driver is required for GPU inference.

## Required Base Artifacts

```text
data/weights/vpt/2x.model
data/weights/vpt/rl-from-foundation-2x.weights
data/weights/mineclip/attn.pth
data/weights/steve1/steve1.weights
data/weights/steve1/steve1_prior.pt
data/visual_prompt_embeds/
data/prior_dataset/data.pkl
```

The helper script `download_weights.sh` downloads the public base STEVE-1/VPT assets used by the original project. It does not download the MultiSTEVE-1s model zoo or raw paper rollout results.

## Public External Links

| Artifact | Hugging Face repository | Local destination |
| --- | --- | --- |
| 240-agent model zoo | [MultiSTEVE-1s/MultiSTEVE-1s](https://huggingface.co/MultiSTEVE-1s/MultiSTEVE-1s) | `data/model_zoo/` |
| Embedded training data | [MultiSTEVE-1s/steve1-training-data](https://huggingface.co/datasets/MultiSTEVE-1s/steve1-training-data) | `data/` |
| Seed first-observation images | [MultiSTEVE-1s/seeds](https://huggingface.co/datasets/MultiSTEVE-1s/seeds) | `data/seeds/` |

See the [project website](https://sites.google.com/view/multisteve-1s) and [attention dashboard demo](https://multisteve-1s.netlify.app/). [Paper (OpenReview PDF)](https://openreview.net/pdf?id=sR3tHXbQxm).

## Download From Hugging Face

Activate the environment and run from the repository root. `environment.yml` includes the Hugging Face `hf` CLI. Downloads preserve the repository-relative paths beneath `--local-dir`, so no extra `out_weights/` directory or manual renaming is needed. The commands use the repository's `main` revision; add `--revision <commit-sha>` to reproduce a specific version. Artifact terms are provided in the linked repositories.

One final checkpoint (approximately 1 GB):

```bash
hf download MultiSTEVE-1s/MultiSTEVE-1s \
  03_init_and_data/dataset_seed1-weight_init_seed1/STEVE-1_snapshot_160005120.weights \
  --local-dir data/model_zoo
```

All 100 final group-06 checkpoints (approximately 100 GB):

```bash
hf download MultiSTEVE-1s/MultiSTEVE-1s \
  --include "06_final_single_lsb_flip_100/*/STEVE-1_snapshot_160005120.weights" \
  --local-dir data/model_zoo
```

Use the corresponding group folder in `--include` for another group. For all snapshots and logs of group 06, use `--include "06_final_single_lsb_flip_100/*"`. To download the complete camera-ready model zoo (approximately 1.5 TB):

```bash
hf download MultiSTEVE-1s/MultiSTEVE-1s --local-dir data/model_zoo
```

Embedded training data (approximately 352 GB):

```bash
hf download MultiSTEVE-1s/steve1-training-data --repo-type dataset \
  --include "dataset_contractor/*" "dataset_mixed_agents/*" \
  --local-dir data
```

The two top-level dataset folders land directly at `data/dataset_contractor/` and `data/dataset_mixed_agents/`. These files already contain MineCLIP embeddings; do not rerun raw-data generation or embedding just to use the released data. Generate the sampling files needed by your training configuration separately.

Seed images:

```bash
hf download MultiSTEVE-1s/seeds --repo-type dataset --local-dir data/seeds
```

This preserves `batch_000/` through `batch_017/`, `flat-ground-for-T-maze/`, `nearly-random/`, and `villages/` inside `data/seeds/`.

The separate [200bitflips expansion](https://huggingface.co/MultiSTEVE-1s/200bitflips) is an ongoing extension, outside the 240-agent camera-ready model zoo. Its run directories are at the repository root. To use that collection, keep it in a separate local root and point custom campaigns at it:

```bash
hf download MultiSTEVE-1s/200bitflips --local-dir data/model_zoo_extra/200bitflips
```

## Model Zoo Layout

Training commands and evaluation configs expect this layout:

```text
data/model_zoo/
  01_init_seed_only/
    dataset_seed1-weight_init_seed1/
      STEVE-1_snapshot_000000000.weights
      ...
      STEVE-1_snapshot_160005120.weights
  02_data_seed_only/
  03_init_and_data/
  04_single_lsb_flip/
  05_final_init_and_data_100/
    dataset_seed101-weight_init_seed101/
      STEVE-1_snapshot_000000000.weights
      STEVE-1_snapshot_160005120.weights
  06_final_single_lsb_flip_100/
    dataset_seed1-weight_init_seed1-bitflip_seed101/
      STEVE-1_snapshot_000000000.weights
      STEVE-1_snapshot_160005120.weights
```

Most evaluation configs reference `STEVE-1_snapshot_160005120.weights`, the 160M-frame checkpoint.

Group 06 is stored alongside groups 01–05 in the same Hugging Face model repository, using the same group/run folder hierarchy. Locally, place it at:

```text
data/model_zoo/06_final_single_lsb_flip_100/
  dataset_seed1-weight_init_seed1-bitflip_seed101/
    STEVE-1_snapshot_000000000.weights
    STEVE-1_snapshot_160005120.weights
  ...
  dataset_seed1-weight_init_seed1-bitflip_seed200/
    STEVE-1_snapshot_000000000.weights
    STEVE-1_snapshot_160005120.weights
```

The initial and final snapshots are present for all 100 group-06 runs. The model repository also includes data-order and training logs. See [Model zoo](model_zoo.md) for the full inventory. Download commands above preserve this layout.

## Training Datasets

The paper uses the OpenAI Contractor Dataset and VPT-generated MineCLIP-embedded trajectories. The source tree includes dataset conversion and sampling scripts under `train/`, but not the underlying datasets.

Expected generated paths:

```text
data/dataset_contractor/
data/dataset_mixed_agents/
data/samplings/
```

If regenerating data instead of downloading the embedded release, run dataset scripts from the repository root so imports come from this checkout:

```bash
train/1_generate_dataset.sh
```

The generation scripts accept episode-count environment variables; their defaults differ. `train/1_generate_dataset.sh` generates a small sample of both datasets. `train/5_generate_dataset_VPT_only.sh` and `train/6_generate_dataset_OA_only.sh` generate the individual datasets. Full dataset generation requires substantial storage.

MineCLIP embeddings are generated during the dataset preparation flow used by `train/1_generate_dataset.sh`, `train/5_generate_dataset_VPT_only.sh`, and `train/6_generate_dataset_OA_only.sh`. After downloading or generating the data, create sampling files:

```bash
train/2_create_sampling.sh
```

This creates `data/samplings/neurips_train.txt` and `neurips_val.txt`, using the available data with a validation holdout targeting 1M frames. Both training entry points use `neurips` by default. See [Training](training.md) for smaller-run overrides.

## Artifact Exclusions

The source repository deliberately excludes all model/checkpoint files (`.weights`, `.model`, `.pth`), raw MineRL rollouts, videos, observations archives, attention arrays, generated dashboards, generated PDFs, and hosted-demo bundles. Keep these large artifacts in external storage or regenerate them locally using the paths documented above.
