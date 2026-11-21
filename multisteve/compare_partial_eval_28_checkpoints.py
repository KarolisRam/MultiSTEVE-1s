#!/usr/bin/env python3
"""Plot reward evolution across checkpoint snapshots for partial evals."""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from multisteve.plot_suite.style import NEURIPS_TEXT_WIDTH_IN, PALETTE, save_figure

try:
    from scipy.stats import t as student_t
except Exception:
    student_t = None

SNAPSHOT_RE = re.compile(r"^STEVE-1_snapshot_(\d+)$")
RUN_SEED_RE = re.compile(r"^seed_(\d+)_")

PROMPT_ORDER = [
    ("text", "dirt-engineered"),
    ("text", "wood-engineered"),
    ("visual", "dirt"),
    ("visual", "wood"),
]


def parse_args() -> argparse.Namespace:
    default_eval_root = Path("final-evals/03_partial_eval_28_checkpoints")
    default_output_dir = Path(__file__).resolve().parent / "plots" / "03_partial_eval_28_checkpoints_reward_evolution"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval-root",
        type=Path,
        default=default_eval_root,
        help=f"Root folder containing snapshot subfolders (default: {default_eval_root})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir,
        help=f"Folder for generated plots and CSV files (default: {default_output_dir})",
    )
    parser.add_argument(
        "--expected-eval-seeds",
        type=int,
        default=20,
        help="Expected eval seeds per (snapshot, prompt); warning only.",
    )
    return parser.parse_args()


def parse_snapshot(snapshot_name: str) -> tuple[str, int] | None:
    match = SNAPSHOT_RE.match(snapshot_name)
    if not match:
        return None
    return snapshot_name, int(match.group(1))


def parse_run_seed(run_dir: Path) -> int | None:
    match = RUN_SEED_RE.match(run_dir.name)
    return int(match.group(1)) if match else None


def reward_metric_for_prompt(prompt: str) -> str | None:
    if "wood" in prompt:
        return "logs"
    if "dirt" in prompt:
        return "dirt"
    return None


def compute_inventory_metrics(inventory_path: Path) -> dict[str, float]:
    if not inventory_path.exists():
        return {"dirt": float("nan"), "logs": float("nan")}

    inventory: dict[str, int] = {}
    max_dirt = 0
    max_logs = 0

    with inventory_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            item = row.get("Item")
            if not item:
                continue
            try:
                count = int(float(row.get("Count", "0")))
            except ValueError:
                continue

            inventory[item] = count
            curr_dirt = sum(v for k, v in inventory.items() if "dirt" in k)
            curr_logs = sum(v for k, v in inventory.items() if "_log" in k)
            max_dirt = max(max_dirt, curr_dirt)
            max_logs = max(max_logs, curr_logs)

    return {"dirt": float(max_dirt), "logs": float(max_logs)}


def collect_latest_runs(
    eval_root: Path,
) -> dict[tuple[str, int, str, str, int], Path]:
    latest: dict[tuple[str, int, str, str, int], Path] = {}
    snapshot_dirs = sorted(p for p in eval_root.iterdir() if p.is_dir())

    for snapshot_dir in snapshot_dirs:
        parsed = parse_snapshot(snapshot_dir.name)
        if parsed is None:
            continue
        snapshot_name, step = parsed

        for modality, prompt in PROMPT_ORDER:
            prompt_dir = snapshot_dir / modality / prompt
            if not prompt_dir.exists():
                continue

            for run_dir in sorted(p for p in prompt_dir.iterdir() if p.is_dir()):
                eval_seed = parse_run_seed(run_dir)
                if eval_seed is None:
                    continue

                key = (snapshot_name, step, modality, prompt, eval_seed)
                prev = latest.get(key)
                if prev is None or run_dir.name > prev.name:
                    latest[key] = run_dir

    return latest


def build_per_run_df(latest_runs: dict[tuple[str, int, str, str, int], Path]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (snapshot_name, step, modality, prompt, eval_seed), run_dir in sorted(latest_runs.items()):
        metrics = compute_inventory_metrics(run_dir / "inventory_updates.log")
        reward_metric = reward_metric_for_prompt(prompt)
        reward = metrics[reward_metric] if reward_metric else float("nan")
        rows.append(
            {
                "snapshot_name": snapshot_name,
                "step": int(step),
                "modality": modality,
                "prompt": prompt,
                "eval_seed": eval_seed,
                "dirt": metrics["dirt"],
                "logs": metrics["logs"],
                "reward_metric": reward_metric,
                "reward": reward,
                "run_dir": str(run_dir),
            }
        )
    return pd.DataFrame(rows)


def mean_ci95(values: np.ndarray) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = int(arr.size)
    if n == 0:
        return float("nan"), float("nan")
    mean = float(np.mean(arr))
    if n < 2:
        return mean, 0.0

    sem = float(np.std(arr, ddof=1) / math.sqrt(n))
    if student_t is not None:
        critical = float(student_t.ppf(0.975, df=n - 1))
    else:
        critical = 1.96
    return mean, critical * sem


def build_agg_df(per_run_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    grouped = per_run_df.groupby(
        ["snapshot_name", "step", "modality", "prompt", "reward_metric"],
        sort=True,
    )
    for (snapshot_name, step, modality, prompt, reward_metric), group in grouped:
        mean, ci95 = mean_ci95(group["reward"].to_numpy(dtype=float))
        rows.append(
            {
                "snapshot_name": snapshot_name,
                "step": int(step),
                "modality": modality,
                "prompt": prompt,
                "reward_metric": reward_metric,
                "n_eval_seeds": int(group["eval_seed"].nunique()),
                "reward_mean": mean,
                "reward_ci95": ci95,
            }
        )

    agg_df = pd.DataFrame(rows).sort_values(["step", "modality", "prompt"])
    step_order = {step: idx for idx, step in enumerate(sorted(agg_df["step"].unique().tolist()))}
    agg_df["snapshot_index"] = agg_df["step"].map(step_order)
    return agg_df


def format_step(step: int) -> str:
    if step >= 1_000_000:
        return f"{step/1_000_000:.1f}M"
    if step >= 1_000:
        return f"{step/1_000:.1f}K"
    return str(step)


def choose_tick_indices(n: int, max_ticks: int = 8) -> list[int]:
    if n <= 0:
        return []
    if n <= max_ticks:
        return list(range(n))
    idx = np.linspace(0, n - 1, num=max_ticks, dtype=int)
    return sorted(set(int(x) for x in idx))


def get_prompt_label(modality: str, prompt: str, reward_metric: str) -> str:
    prompt_label = f"{prompt.removesuffix('-engineered')}*" if prompt.endswith("-engineered") else prompt
    return f"{modality.title()} Prompt | {prompt_label}"


def plot_reward_evolution(agg_df: pd.DataFrame, spacing_mode: str, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(NEURIPS_TEXT_WIDTH_IN, 4.2))
    axes_flat = axes.flatten()

    panel_prompts = [p for p in PROMPT_ORDER if p in set(zip(agg_df["modality"], agg_df["prompt"]))]

    for i, (modality, prompt) in enumerate(panel_prompts):
        ax = axes_flat[i]
        subset = agg_df[(agg_df["modality"] == modality) & (agg_df["prompt"] == prompt)].copy()
        subset = subset.sort_values("step")
        if subset.empty:
            ax.axis("off")
            continue

        steps = subset["step"].to_numpy(dtype=float)
        means = subset["reward_mean"].to_numpy(dtype=float)
        ci = subset["reward_ci95"].to_numpy(dtype=float)
        reward_metric = str(subset["reward_metric"].iloc[0])
        step_labels = [format_step(int(x)) for x in steps]

        if spacing_mode == "equal":
            x = np.arange(len(subset), dtype=float)
            tick_idx = choose_tick_indices(len(subset), max_ticks=8)
            xticks = [x[j] for j in tick_idx]
            xticklabels = [step_labels[j] for j in tick_idx]
            xlabel = "Checkpoint order"
        elif spacing_mode == "linear":
            x = steps
            tick_idx = choose_tick_indices(len(subset), max_ticks=8)
            xticks = [x[j] for j in tick_idx]
            xticklabels = [step_labels[j] for j in tick_idx]
            xlabel = "Training steps"
        elif spacing_mode == "log":
            positive_steps = steps[steps > 0]
            min_positive = positive_steps.min() if positive_steps.size else 1.0
            x = np.where(steps > 0, steps, min_positive / 2.0)
            tick_idx = choose_tick_indices(len(subset), max_ticks=8)
            xticks = [x[j] for j in tick_idx]
            xticklabels = [step_labels[j] for j in tick_idx]
            xlabel = "Training steps"
        else:
            raise ValueError(f"Unsupported spacing mode: {spacing_mode}")

        color = PALETTE["blue"] if reward_metric == "logs" else PALETTE["orange"]
        ax.plot(x, means, marker="o", linewidth=1.45, markersize=3.2, color=color)
        ax.fill_between(x, np.maximum(0.0, means - ci), means + ci, alpha=0.22, color=color)
        ax.set_title(get_prompt_label(modality, prompt, reward_metric), pad=3)
        ax.set_ylabel("Reward (mean max count)")
        ax.set_xlabel(xlabel)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels, rotation=35, ha="right")
        if spacing_mode == "log":
            ax.set_xscale("log")
        ax.grid(alpha=0.25)

    for i in range(len(panel_prompts), 4):
        axes_flat[i].axis("off")

    title_map = {
        "equal": "Reward Evolution Across Checkpoints",
        "linear": "Reward Evolution Across Checkpoints (Linear Steps)",
        "log": "Reward Evolution Across Checkpoints (Log Steps)",
    }
    fig.suptitle(title_map[spacing_mode], fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.955], pad=0.14, h_pad=1.05)
    save_figure(fig, output_path)


def main() -> None:
    args = parse_args()
    eval_root = args.eval_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not eval_root.exists():
        raise FileNotFoundError(f"Evaluation root does not exist: {eval_root}")

    latest_runs = collect_latest_runs(eval_root)
    per_run_df = build_per_run_df(latest_runs)
    if per_run_df.empty:
        raise RuntimeError(f"No valid runs found under: {eval_root}")

    agg_df = build_agg_df(per_run_df)

    per_run_path = output_dir / "per_run_prompt_rewards.csv"
    agg_path = output_dir / "snapshot_prompt_reward_summary.csv"
    per_run_df.to_csv(per_run_path, index=False)
    agg_df.to_csv(agg_path, index=False)

    plot_reward_evolution(agg_df, "equal", output_dir / "reward_evolution_equal_spacing.png")
    plot_reward_evolution(agg_df, "linear", output_dir / "reward_evolution_linear_steps.png")
    plot_reward_evolution(agg_df, "log", output_dir / "reward_evolution_log_steps.png")

    seed_counts = agg_df[["snapshot_name", "modality", "prompt", "n_eval_seeds"]].copy()
    expected = args.expected_eval_seeds
    incomplete = seed_counts[seed_counts["n_eval_seeds"] != expected]
    if incomplete.empty:
        print(f"All snapshot/prompt combinations have {expected} eval seeds.")
    else:
        print(
            f"Warning: {len(incomplete)} snapshot/prompt combinations differ from expected {expected} eval seeds."
        )
        print(incomplete.to_string(index=False))

    print(f"Snapshots found: {agg_df['snapshot_name'].nunique()}")
    print(f"Wrote per-run rewards: {per_run_path}")
    print(f"Wrote snapshot summary: {agg_path}")
    print(f"Wrote plots to: {output_dir}")


if __name__ == "__main__":
    main()
