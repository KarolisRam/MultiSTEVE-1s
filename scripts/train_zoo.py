from __future__ import annotations

import argparse
import shlex
from dataclasses import dataclass
from pathlib import Path


GROUPS = ("01", "02", "03", "04", "05", "06")


@dataclass(frozen=True)
class TrainSpec:
    group: str
    run_name: str
    dataset_seed: int
    weight_init_seed: int
    bitflip_seed: int | None = None


COMMON_ARGS: tuple[str, ...] = (
    "--in_model", "data/weights/vpt/2x.model",
    "--in_weights", "data/weights/vpt/rl-from-foundation-2x.weights",
    "--trunc_t", "64",
    "--T", "640",
    "--batch_size", "12",
    "--gradient_accumulation_steps", "4",
    "--num_workers", "1",
    "--weight_decay", "0.039428",
    "--n_frames", "160_000_000",
    "--learning_rate", "4e-5",
    "--warmup_frames", "10_000_000",
    "--p_uncond", "0.1",
    "--min_btwn_goals", "15",
    "--max_btwn_goals", "200",
    "--val_freq", "10000",
    "--val_freq_begin", "10000",
    "--val_freq_switch_steps", "500",
    "--shareable_checkpoint_every_n_frames", "10_000_000",
    "--val_every_nth", "1",
    "--save_each_val", "False",
    "--sampling", "neurips",
    "--sampling_dir", "data/samplings/",
    "--resumable_checkpoint_every_n_frames", "20_000_000",
)


def specs_for_group(group: str) -> list[TrainSpec]:
    if group == "01":
        return [TrainSpec(group, f"dataset_seed1-weight_init_seed{w}", 1, w) for w in range(1, 11)]
    if group == "02":
        return [TrainSpec(group, f"dataset_seed{d}-weight_init_seed1", d, 1) for d in range(1, 11)]
    if group == "03":
        return [TrainSpec(group, f"dataset_seed{s}-weight_init_seed{s}", s, s) for s in range(1, 11)]
    if group == "04":
        return [
            TrainSpec(group, f"dataset_seed1-weight_init_seed1-bitflip_seed{b}", 1, 1, bitflip_seed=b)
            for b in range(1, 11)
        ]
    if group == "05":
        return [TrainSpec(group, f"dataset_seed{s}-weight_init_seed{s}", s, s) for s in range(101, 201)]
    if group == "06":
        return [
            TrainSpec(group, f"dataset_seed1-weight_init_seed1-bitflip_seed{b}", 1, 1, bitflip_seed=b)
            for b in range(101, 201)
        ]
    raise ValueError(f"Unknown group: {group}")


def group_dir(group: str) -> str:
    return {
        "01": "01_init_seed_only",
        "02": "02_data_seed_only",
        "03": "03_init_and_data",
        "04": "04_single_lsb_flip",
        "05": "05_final_init_and_data_100",
        "06": "06_final_single_lsb_flip_100",
    }[group]


def training_args_for_spec(spec: TrainSpec, output_root: Path) -> list[str]:
    out_dir = output_root / group_dir(spec.group) / spec.run_name
    args = [
        "python", "-m", "steve1.training.train",
        *COMMON_ARGS,
        "--out_weights", str(out_dir / "STEVE-1.weights"),
        "--checkpoint_dir", str(out_dir / "checkpoint_dir"),
        "--dataset_seed", str(spec.dataset_seed),
        "--weight_init_seed", str(spec.weight_init_seed),
        "--wandb_run_name", spec.run_name,
    ]
    if spec.bitflip_seed is not None:
        args.extend(["--bitflip", "--bitflip_seed", str(spec.bitflip_seed)])
    return args


def command_for_spec(spec: TrainSpec, output_root: Path) -> str:
    args = training_args_for_spec(spec, output_root)
    return (
        'conda run -n steve-1-creative env PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" '
        + shlex.join(args)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print MultiSTEVE-1s model-zoo training commands.")
    parser.add_argument("--group", choices=("all", *GROUPS), default="all")
    parser.add_argument("--output-root", type=Path, default=Path("data/model_zoo"))
    parser.add_argument("--count-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    groups = GROUPS if args.group == "all" else (args.group,)
    specs = [spec for group in groups for spec in specs_for_group(group)]
    if args.count_only:
        print(len(specs))
        return 0
    for spec in specs:
        print(command_for_spec(spec, args.output_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
