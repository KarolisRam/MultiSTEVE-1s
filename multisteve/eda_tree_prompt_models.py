#!/usr/bin/env python3
"""Generate EDA plots for tree-log rollouts across a model sweep."""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from multisteve.plot_suite.style import NEURIPS_TEXT_WIDTH_IN, PALETTE, SIDE_BY_SIDE_FIGURE_WIDTH_IN, save_figure

try:
    from scipy.stats import t as student_t
except Exception:
    student_t = None

MODEL_SEED_RE = re.compile(r"dataset_seed(\d+)-weight_init_seed(\d+)")
RUN_SEED_RE = re.compile(r"^seed_(\d+)_")


def parse_args() -> argparse.Namespace:
    default_eval_root = Path("final-evals/04_tree_prompt_100_models")
    default_output_dir = Path(__file__).resolve().parent / "plots" / "04_tree_prompt_100_models_eda"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval-root",
        type=Path,
        default=default_eval_root,
        help=f"Root directory containing tree-prompt model folders (default: {default_eval_root})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir,
        help=f"Output directory for plots and CSVs (default: {default_output_dir})",
    )
    parser.add_argument(
        "--expected-eval-seeds",
        type=int,
        default=20,
        help="Expected eval seeds per model; warning only.",
    )
    return parser.parse_args()


def parse_model_info(model_dir_name: str) -> tuple[int, int]:
    match = MODEL_SEED_RE.search(model_dir_name)
    if not match:
        return (10_000, 10_000)
    return int(match.group(1)), int(match.group(2))


def parse_run_seed(run_dir_name: str) -> int | None:
    match = RUN_SEED_RE.match(run_dir_name)
    return int(match.group(1)) if match else None


def compute_max_logs(inventory_path: Path) -> float:
    if not inventory_path.exists():
        return float("nan")

    inventory: dict[str, int] = {}
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
            current_logs = sum(v for k, v in inventory.items() if "_log" in k)
            max_logs = max(max_logs, current_logs)
    return float(max_logs)


def collect_latest_runs(eval_root: Path) -> dict[tuple[str, int], Path]:
    latest: dict[tuple[str, int], Path] = {}
    for model_dir in sorted(p for p in eval_root.iterdir() if p.is_dir()):
        prompt_dir = model_dir / "text" / "wood"
        if not prompt_dir.exists():
            continue
        for run_dir in sorted(p for p in prompt_dir.iterdir() if p.is_dir()):
            eval_seed = parse_run_seed(run_dir.name)
            if eval_seed is None:
                continue
            key = (model_dir.name, eval_seed)
            prev = latest.get(key)
            if prev is None or run_dir.name > prev.name:
                latest[key] = run_dir
    return latest


def build_per_run_df(latest_runs: dict[tuple[str, int], Path]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (model_dir_name, eval_seed), run_dir in sorted(latest_runs.items()):
        dataset_seed, weight_seed = parse_model_info(model_dir_name)
        bitflip_match = re.search(r"bitflip_seed(\d+)", model_dir_name)
        model_label = f"bitflip-{bitflip_match.group(1)}" if bitflip_match else f"s{dataset_seed}"
        rows.append(
            {
                "model_dir_name": model_dir_name,
                "dataset_seed": dataset_seed,
                "weight_seed": weight_seed,
                "model_label": model_label,
                "eval_seed": eval_seed,
                "max_logs": compute_max_logs(run_dir / "inventory_updates.log"),
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


def build_model_summary_df(per_run_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    grouped = per_run_df.groupby(
        ["model_dir_name", "dataset_seed", "weight_seed", "model_label"],
        sort=True,
    )
    for (model_dir_name, dataset_seed, weight_seed, model_label), group in grouped:
        mean, ci95 = mean_ci95(group["max_logs"].to_numpy(dtype=float))
        rows.append(
            {
                "model_dir_name": model_dir_name,
                "dataset_seed": int(dataset_seed),
                "weight_seed": int(weight_seed),
                "model_label": model_label,
                "n_eval_seeds": int(group["eval_seed"].nunique()),
                "logs_mean": mean,
                "logs_ci95": ci95,
                "logs_std": float(group["max_logs"].std(ddof=1)),
                "logs_median": float(group["max_logs"].median()),
                "logs_min": float(group["max_logs"].min()),
                "logs_max": float(group["max_logs"].max()),
            }
        )
    return pd.DataFrame(rows).sort_values(["dataset_seed", "weight_seed", "model_dir_name"])


def choose_tick_positions(n: int, max_ticks: int = 12) -> list[int]:
    if n <= max_ticks:
        return list(range(n))
    idx = np.linspace(0, n - 1, num=max_ticks, dtype=int)
    return sorted(set(int(x) for x in idx))


def choose_agent_seed_ticks(labels: list[str]) -> list[int]:
    tick_indices: list[int] = []
    for idx, label in enumerate(labels):
        try:
            value = int(label.removeprefix("s"))
        except ValueError:
            continue
        if value == 101 or (105 <= value <= 200 and value % 5 == 0):
            tick_indices.append(idx)
    return tick_indices or choose_tick_positions(len(labels), max_ticks=14)


def clipped_ci_errors(means: np.ndarray, ci: np.ndarray) -> np.ndarray:
    low = np.maximum(0.0, means - ci)
    high = means + ci
    return np.vstack([means - low, high - means])


def plot_model_means_in_order(summary_df: pd.DataFrame, ax: plt.Axes) -> None:
    df = summary_df.sort_values(["dataset_seed", "weight_seed", "model_dir_name"]).reset_index(drop=True)
    x = np.arange(len(df), dtype=float)
    y = df["logs_mean"].to_numpy(dtype=float)
    yerr = clipped_ci_errors(y, df["logs_ci95"].to_numpy(dtype=float))
    ax.errorbar(x, y, yerr=yerr, fmt="o-", capsize=1.4, capthick=0.5, color=PALETTE["blue"], ecolor="black", elinewidth=0.5, linewidth=1.2, markersize=3.2)
    tick_idx = choose_tick_positions(len(df), max_ticks=12)
    ax.set_xticks(tick_idx)
    ax.set_xticklabels(df.iloc[tick_idx]["model_label"].tolist(), rotation=35, ha="right")
    ax.set_title("Model Means In Seed Order")
    ax.set_xlabel("Model")
    ax.set_ylabel("Mean max tree logs")
    ax.grid(alpha=0.25)


def plot_model_means_ranked(summary_df: pd.DataFrame, ax: plt.Axes) -> None:
    df = summary_df.sort_values(["logs_mean", "dataset_seed"], ascending=[False, True]).reset_index(drop=True)
    x = np.arange(len(df), dtype=float)
    y = df["logs_mean"].to_numpy(dtype=float)
    yerr = clipped_ci_errors(y, df["logs_ci95"].to_numpy(dtype=float))
    ax.errorbar(x, y, yerr=yerr, fmt="o-", capsize=1.4, capthick=0.5, color=PALETTE["red"], ecolor="black", elinewidth=0.5, linewidth=1.2, markersize=3.2)
    tick_idx = choose_tick_positions(len(df), max_ticks=12)
    ax.set_xticks(tick_idx)
    ax.set_xticklabels(df.iloc[tick_idx]["model_label"].tolist(), rotation=35, ha="right")
    ax.set_title("Model Means Ranked")
    ax.set_xlabel("Ranked model")
    ax.set_ylabel("Mean max tree logs")
    ax.grid(alpha=0.25)


def plot_histogram(summary_df: pd.DataFrame, ax: plt.Axes) -> None:
    values = summary_df["logs_mean"].to_numpy(dtype=float)
    bins = min(12, max(6, int(round(math.sqrt(len(values))))))
    ax.hist(values, bins=bins, color=PALETTE["green"], edgecolor="white", alpha=0.9)
    ax.set_title("Distribution Of Model Means")
    ax.set_xlabel("Mean max tree logs")
    ax.set_ylabel("Number of models")
    ax.grid(axis="y", alpha=0.25)


def plot_ecdf(summary_df: pd.DataFrame, ax: plt.Axes) -> None:
    values = np.sort(summary_df["logs_mean"].to_numpy(dtype=float))
    y = np.arange(1, len(values) + 1, dtype=float) / len(values)
    ax.step(values, y, where="post", color=PALETTE["purple"], linewidth=1.8)
    ax.set_title("ECDF Of Model Means")
    ax.set_xlabel("Mean max tree logs")
    ax.set_ylabel("Fraction of models")
    ax.grid(alpha=0.25)


def save_individual_plot(plot_fn, summary_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(SIDE_BY_SIDE_FIGURE_WIDTH_IN, 2.3))
    plot_fn(summary_df, ax)
    fig.tight_layout(pad=0.2)
    save_figure(fig, output_path, bbox_inches=None)


def plot_model_means_bar_vertical(summary_df: pd.DataFrame, output_path: Path) -> None:
    df = summary_df.sort_values(["dataset_seed", "weight_seed", "model_dir_name"]).reset_index(drop=True)
    x = np.arange(len(df), dtype=float)
    y = df["logs_mean"].to_numpy(dtype=float)
    yerr = clipped_ci_errors(y, df["logs_ci95"].to_numpy(dtype=float))
    labels = df["model_label"].astype(str).tolist()
    tick_idx = choose_agent_seed_ticks(labels)

    fig, ax = plt.subplots(figsize=(NEURIPS_TEXT_WIDTH_IN, 2.85))
    ax.bar(x, y, color=PALETTE["blue"], alpha=0.9, edgecolor="black", linewidth=0.16, width=0.78)
    ax.errorbar(x, y, yerr=yerr, fmt="none", ecolor="black", elinewidth=0.42, capsize=1.1, capthick=0.42, zorder=3)
    ax.set_title("Tree Prompt 100 Agents", pad=3)
    ax.set_xlabel("Agent")
    ax.set_ylabel("Mean max tree logs")
    ax.set_xticks(tick_idx)
    ax.set_xticklabels([labels[i] for i in tick_idx], rotation=45, ha="right")
    ax.set_xlim(-0.6, len(df) - 0.4)
    ax.margins(x=0.0, y=0.02)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(pad=0.12)
    save_figure(fig, output_path, bbox_inches=None)


def plot_model_means_bar_horizontal(summary_df: pd.DataFrame, output_path: Path) -> None:
    df = summary_df.sort_values(["dataset_seed", "weight_seed", "model_dir_name"]).reset_index(drop=True)
    y_pos = np.arange(len(df), dtype=float)
    means = df["logs_mean"].to_numpy(dtype=float)
    yerr = clipped_ci_errors(means, df["logs_ci95"].to_numpy(dtype=float))
    labels = df["model_label"].astype(str).tolist()
    tick_idx = choose_agent_seed_ticks(labels)

    fig, ax = plt.subplots(figsize=(NEURIPS_TEXT_WIDTH_IN, 7.8))
    ax.barh(y_pos, means, color=PALETTE["blue"], alpha=0.9, edgecolor="black", linewidth=0.16, height=0.78)
    ax.errorbar(means, y_pos, xerr=yerr, fmt="none", ecolor="black", elinewidth=0.42, capsize=1.1, capthick=0.42, zorder=3)
    ax.set_title("Tree Prompt 100 Agents", pad=3)
    ax.set_xlabel("Mean max tree logs")
    ax.set_ylabel("Agent")
    ax.set_yticks(tick_idx)
    ax.set_yticklabels([labels[i] for i in tick_idx])
    ax.set_ylim(len(df) - 0.4, -0.6)
    ax.margins(x=0.02, y=0.0)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout(pad=0.12)
    save_figure(fig, output_path, bbox_inches=None)


def plot_overview(summary_df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(NEURIPS_TEXT_WIDTH_IN, 4.2))
    plot_model_means_in_order(summary_df, axes[0, 0])
    plot_model_means_ranked(summary_df, axes[0, 1])
    plot_histogram(summary_df, axes[1, 0])
    plot_ecdf(summary_df, axes[1, 1])
    fig.suptitle("Tree-Log EDA Across Model Sweep", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95], pad=0.2)
    save_figure(fig, output_path)


def main() -> None:
    args = parse_args()
    eval_root = args.eval_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not eval_root.exists():
        raise FileNotFoundError(f"Eval root does not exist: {eval_root}")

    latest_runs = collect_latest_runs(eval_root)
    per_run_df = build_per_run_df(latest_runs)
    if per_run_df.empty:
        raise RuntimeError(f"No valid tree-prompt runs found under: {eval_root}")

    summary_df = build_model_summary_df(per_run_df)
    ranking_df = summary_df.sort_values(["logs_mean", "dataset_seed"], ascending=[False, True]).reset_index(drop=True)
    ranking_df.insert(0, "rank", np.arange(1, len(ranking_df) + 1))

    per_run_path = output_dir / "tree_logs_per_run.csv"
    summary_path = output_dir / "tree_logs_model_summary.csv"
    ranking_path = output_dir / "tree_logs_model_ranking.csv"
    per_run_df.to_csv(per_run_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    ranking_df.to_csv(ranking_path, index=False)

    plot_overview(summary_df, output_dir / "tree_logs_eda_overview.png")
    save_individual_plot(plot_model_means_in_order, summary_df, output_dir / "tree_logs_model_means_in_order.png")
    save_individual_plot(plot_model_means_ranked, summary_df, output_dir / "tree_logs_model_means_ranked.png")
    save_individual_plot(plot_histogram, summary_df, output_dir / "tree_logs_model_mean_histogram.png")
    save_individual_plot(plot_ecdf, summary_df, output_dir / "tree_logs_model_mean_ecdf.png")

    seed_counts = summary_df[["model_label", "n_eval_seeds"]].copy()
    expected = args.expected_eval_seeds
    incomplete = seed_counts[seed_counts["n_eval_seeds"] != expected]
    if incomplete.empty:
        print(f"All models have {expected} eval seeds.")
    else:
        print(f"Warning: {len(incomplete)} models differ from expected {expected} eval seeds.")
        print(incomplete.to_string(index=False))

    print(f"Models found: {summary_df['model_dir_name'].nunique()}")
    print(f"Wrote per-run data: {per_run_path}")
    print(f"Wrote model summary: {summary_path}")
    print(f"Wrote model ranking: {ranking_path}")
    print(f"Wrote plots to: {output_dir}")


if __name__ == "__main__":
    main()
