#!/usr/bin/env python3
"""Compare models within each partial-eval group on dirt/log collection metrics."""

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

RUN_DIR_SEED_RE = re.compile(r"^seed_(\d+)_")
GROUP_ID_RE = re.compile(r"^(?P<group_id>\d{2})_")

PROMPT_ORDER = [
    ("text", "dirt-engineered"),
    ("text", "wood-engineered"),
    ("visual", "dirt"),
    ("visual", "wood"),
]

GROUP_NAMES = {
    "01": "Group 1 - Init Seed Sweep",
    "02": "Group 2 - Data Seed Sweep",
    "03": "Group 3 - Init+Data Seed Sweep",
    "04": "Group 4 - Single LSB Flip Sweep",
}


def parse_args() -> argparse.Namespace:
    default_eval_root = Path("final-evals/02_partial_eval_40_models")
    default_output_dir = (
        Path(__file__).resolve().parent / "plots" / "02_partial_eval_40_models_compare_prompt_aligned"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval-root",
        type=Path,
        default=default_eval_root,
        help=f"Root directory containing the 40 model folders (default: {default_eval_root})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir,
        help=f"Output directory for plots/CSV files (default: {default_output_dir})",
    )
    parser.add_argument(
        "--expected-eval-seeds",
        type=int,
        default=20,
        help="Expected number of eval seeds per (group, model, prompt); warning only.",
    )
    return parser.parse_args()


def parse_group_id(model_dir_name: str) -> str:
    match = GROUP_ID_RE.match(model_dir_name)
    return match.group("group_id") if match else "unknown"


def parse_model_sort_seed(group_id: str, model_dir_name: str) -> int:
    if group_id == "01":
        m = re.search(r"weight_init_seed(\d+)", model_dir_name)
        return int(m.group(1)) if m else 10_000
    if group_id == "02":
        m = re.search(r"dataset_seed(\d+)", model_dir_name)
        return int(m.group(1)) if m else 10_000
    if group_id == "03":
        m = re.search(r"dataset_seed(\d+)", model_dir_name)
        return int(m.group(1)) if m else 10_000
    if group_id == "04":
        m = re.search(r"bitflip_seed(\d+)", model_dir_name)
        return int(m.group(1)) if m else 10_000
    return 10_000


def parse_model_label(group_id: str, model_dir_name: str) -> str:
    seed = parse_model_sort_seed(group_id, model_dir_name)
    if seed >= 10_000:
        return model_dir_name
    if group_id == "01":
        return f"init_s{seed}"
    if group_id == "02":
        return f"data_s{seed}"
    if group_id == "03":
        return f"both_s{seed}"
    if group_id == "04":
        return f"flip_s{seed}"
    return model_dir_name


def parse_eval_seed(run_dir: Path) -> int | None:
    match = RUN_DIR_SEED_RE.match(run_dir.name)
    return int(match.group(1)) if match else None


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


def collect_latest_runs(eval_root: Path) -> dict[tuple[str, str, str, str, int], Path]:
    latest: dict[tuple[str, str, str, str, int], Path] = {}

    model_dirs = sorted(p for p in eval_root.iterdir() if p.is_dir())
    for model_dir in model_dirs:
        model_dir_name = model_dir.name
        group_id = parse_group_id(model_dir_name)

        for modality, prompt in PROMPT_ORDER:
            prompt_dir = model_dir / modality / prompt
            if not prompt_dir.exists():
                continue

            for run_dir in sorted(p for p in prompt_dir.iterdir() if p.is_dir()):
                eval_seed = parse_eval_seed(run_dir)
                if eval_seed is None:
                    continue
                key = (group_id, model_dir_name, modality, prompt, eval_seed)
                prev = latest.get(key)
                if prev is None or run_dir.name > prev.name:
                    latest[key] = run_dir
    return latest


def build_per_run_df(latest_runs: dict[tuple[str, str, str, str, int], Path]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (group_id, model_dir_name, modality, prompt, eval_seed), run_dir in sorted(latest_runs.items()):
        metrics = compute_inventory_metrics(run_dir / "inventory_updates.log")
        rows.append(
            {
                "group_id": group_id,
                "group_name": GROUP_NAMES.get(group_id, f"Group {group_id}"),
                "model_dir_name": model_dir_name,
                "model_sort_seed": parse_model_sort_seed(group_id, model_dir_name),
                "model_label": parse_model_label(group_id, model_dir_name),
                "modality": modality,
                "prompt": prompt,
                "eval_seed": eval_seed,
                "dirt": metrics["dirt"],
                "logs": metrics["logs"],
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
        critical_value = float(student_t.ppf(0.975, df=n - 1))
    else:
        critical_value = 1.96
    return mean, critical_value * sem


def build_agg_df(per_run_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    grouped = per_run_df.groupby(
        ["group_id", "group_name", "model_dir_name", "model_label", "model_sort_seed", "modality", "prompt"],
        sort=True,
    )

    for keys, group in grouped:
        (
            group_id,
            group_name,
            model_dir_name,
            model_label,
            model_sort_seed,
            modality,
            prompt,
        ) = keys
        dirt_mean, dirt_ci95 = mean_ci95(group["dirt"].to_numpy(dtype=float))
        logs_mean, logs_ci95 = mean_ci95(group["logs"].to_numpy(dtype=float))
        rows.append(
            {
                "group_id": group_id,
                "group_name": group_name,
                "model_dir_name": model_dir_name,
                "model_label": model_label,
                "model_sort_seed": int(model_sort_seed),
                "modality": modality,
                "prompt": prompt,
                "n_eval_seeds": int(group["eval_seed"].nunique()),
                "dirt": dirt_mean,
                "dirt_ci95": dirt_ci95,
                "logs": logs_mean,
                "logs_ci95": logs_ci95,
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["group_id", "model_sort_seed", "modality", "prompt", "model_dir_name"]
    )


def get_prompt_panels(group_df: pd.DataFrame) -> list[tuple[str, str]]:
    available = set(zip(group_df["modality"], group_df["prompt"]))
    ordered = [x for x in PROMPT_ORDER if x in available]
    remaining = sorted(available - set(ordered))
    return ordered + remaining


def metric_for_prompt(prompt: str) -> str | None:
    if "wood" in prompt:
        return "logs"
    if "dirt" in prompt:
        return "dirt"
    return None


def display_prompt_label(modality: str, prompt: str, metric: str) -> str:
    prompt_label = f"{prompt.removesuffix('-engineered')}*" if prompt.endswith("-engineered") else prompt
    return f"{modality.title()} Prompt | {prompt_label}"


def plot_group_prompt_aligned(group_df: pd.DataFrame, group_name: str, output_path: Path) -> None:
    prompt_panels = get_prompt_panels(group_df)
    if not prompt_panels:
        return

    n_panels = len(prompt_panels)
    nrows, ncols = 2, 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(NEURIPS_TEXT_WIDTH_IN, 4.2))
    axes_flat = axes.flatten()

    for panel_idx, (modality, prompt) in enumerate(prompt_panels):
        ax = axes_flat[panel_idx]
        subset = group_df[(group_df["modality"] == modality) & (group_df["prompt"] == prompt)].copy()
        subset = subset.sort_values(["model_sort_seed", "model_label", "model_dir_name"])
        metric = metric_for_prompt(prompt)
        if metric is None:
            ax.axis("off")
            continue

        x = np.arange(len(subset), dtype=float)
        y = subset[metric].to_numpy(dtype=float)
        ci = subset[f"{metric}_ci95"].to_numpy(dtype=float)
        low = np.maximum(0.0, y - ci)
        high = y + ci
        yerr = np.vstack([y - low, high - y])
        labels = subset["model_label"].tolist()

        color = PALETTE["blue"] if metric == "logs" else PALETTE["orange"]
        ax.bar(
            x,
            y,
            yerr=yerr,
            capsize=1.3,
            color=color,
            alpha=0.9,
            edgecolor="black",
            linewidth=0.22,
            error_kw={"elinewidth": 0.45, "capthick": 0.45},
        )
        metric_title = "logs" if metric == "logs" else "dirt"
        ax.set_title(display_prompt_label(modality, prompt, metric), pad=3)
        ax.set_ylabel(f"Average max {metric_title} count")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.grid(axis="y", alpha=0.25)

    for panel_idx in range(n_panels, nrows * ncols):
        axes_flat[panel_idx].axis("off")

    fig.suptitle(
        f"Partial Evaluations | {group_name}",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.955], pad=0.14)
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
        raise RuntimeError(f"No evaluation runs found under: {eval_root}")

    agg_df = build_agg_df(per_run_df)

    per_run_path = output_dir / "per_run_metrics.csv"
    agg_path = output_dir / "aggregated_model_prompt_metrics.csv"
    per_run_df.to_csv(per_run_path, index=False)
    agg_df.to_csv(agg_path, index=False)

    for group_id in sorted(agg_df["group_id"].unique().tolist()):
        group_df = agg_df[agg_df["group_id"] == group_id].copy()
        group_name = GROUP_NAMES.get(group_id, f"Group {group_id}")
        plot_group_prompt_aligned(group_df, group_name, output_dir / f"group_{group_id}_prompt_aligned.png")

    seed_counts = (
        agg_df[["group_id", "model_label", "modality", "prompt", "n_eval_seeds"]]
        .sort_values(["group_id", "model_label", "modality", "prompt"])
        .copy()
    )
    expected = args.expected_eval_seeds
    incomplete = seed_counts[seed_counts["n_eval_seeds"] != expected]
    if incomplete.empty:
        print(f"All model/prompt combinations have {expected} eval seeds.")
    else:
        print(
            f"Warning: {len(incomplete)} model/prompt combinations differ from expected {expected} eval seeds."
        )
        print(incomplete.to_string(index=False))

    group_model_counts = (
        agg_df[["group_id", "model_dir_name"]]
        .drop_duplicates()
        .groupby("group_id", as_index=False)
        .size()
        .rename(columns={"size": "n_models"})
        .sort_values("group_id")
    )
    print("Models per group:")
    print(group_model_counts.to_string(index=False))
    print(f"Wrote per-run metrics: {per_run_path}")
    print(f"Wrote aggregated metrics: {agg_path}")
    print(f"Wrote plots to: {output_dir}")


if __name__ == "__main__":
    main()
