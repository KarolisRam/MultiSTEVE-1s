#!/usr/bin/env python3
"""Relate villager metrics to tree-chopping competence for overlapping models."""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patheffects

try:
    from scipy.stats import t as student_t
except Exception:
    student_t = None

MODEL_SEED_RE = re.compile(r"dataset_seed(\d+)")
RUN_SEED_RE = re.compile(r"^seed_(\d+)_")
SETTING_ORDER = [
    ("logs_above_villager_false", "wood"),
    ("logs_above_villager_false", "wood-engineered"),
    ("logs_above_villager_true", "wood"),
    ("logs_above_villager_true", "wood-engineered"),
]
SETTING_LABELS = {
    ("logs_above_villager_false", "wood"): 'logs=false | prompt="chop a tree"',
    ("logs_above_villager_false", "wood-engineered"): "logs=false | engineered prompt",
    ("logs_above_villager_true", "wood"): 'logs=true | prompt="chop a tree"',
    ("logs_above_villager_true", "wood-engineered"): "logs=true | engineered prompt",
}
SETTING_COLORS = {
    ("logs_above_villager_false", "wood"): "#2a9d8f",
    ("logs_above_villager_false", "wood-engineered"): "#1d3557",
    ("logs_above_villager_true", "wood"): "#e76f51",
    ("logs_above_villager_true", "wood-engineered"): "#8c564b",
}
SETTING_MARKERS = {
    ("logs_above_villager_false", "wood"): "o",
    ("logs_above_villager_false", "wood-engineered"): "s",
    ("logs_above_villager_true", "wood"): "^",
    ("logs_above_villager_true", "wood-engineered"): "D",
}
Z_95 = 1.959963984540054


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--villager-eval-root",
        type=Path,
        default=Path("data/evals-villager"),
        help="Root containing villager `pass_XX` folders.",
    )
    parser.add_argument(
        "--tree-eval-root",
        type=Path,
        default=Path("data/final-evals/04_tree_prompt_100_models"),
        help="Root containing tree prompt model sweep folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir,
        help="Directory to write scatter plots and joined CSV files.",
    )
    parser.add_argument("--pass-start", type=int, default=1, help="First villager pass to include.")
    parser.add_argument("--pass-end", type=int, default=18, help="Last villager pass to include.")
    parser.add_argument(
        "--logs-setting",
        type=str,
        default=None,
        help="Optional villager setting filter, e.g. logs_above_villager_true.",
    )
    parser.add_argument(
        "--prompt-slug",
        type=str,
        default=None,
        help='Optional prompt filter, e.g. "wood" or "wood-engineered".',
    )
    return parser.parse_args()


def parse_model_seed(name: str) -> int:
    match = MODEL_SEED_RE.search(name)
    return int(match.group(1)) if match else 10_000


def parse_run_seed(name: str) -> int | None:
    match = RUN_SEED_RE.match(name)
    return int(match.group(1)) if match else None


def latest_run_dir(prompt_dir: Path) -> Path | None:
    run_dirs = sorted((p for p in prompt_dir.iterdir() if p.is_dir()), key=lambda p: p.name)
    return run_dirs[-1] if run_dirs else None


def mean_ci95(values: list[float]) -> tuple[float, float]:
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
        critical = Z_95
    return mean, critical * sem


def wilson_ci(successes: int, n: int) -> tuple[float, float]:
    if n <= 0:
        return float("nan"), float("nan")
    phat = successes / n
    denom = 1.0 + (Z_95 * Z_95) / n
    center = (phat + (Z_95 * Z_95) / (2.0 * n)) / denom
    half = (Z_95 / denom) * math.sqrt((phat * (1.0 - phat) / n) + (Z_95 * Z_95) / (4.0 * n * n))
    return center - half, center + half


def read_kill_step(run_dir: Path) -> int:
    return int((run_dir / "kill_step.log").read_text(encoding="utf-8").strip())


def read_hit_sum(run_dir: Path) -> float:
    total = 0.0
    with (run_dir / "hits.log").open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                total += float(row.get("Damage Dealt", "0"))
            except ValueError:
                continue
    return total


def collect_villager_per_pass(
    villager_root: Path,
    pass_start: int,
    pass_end: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for pass_num in range(pass_start, pass_end + 1):
        pass_dir = villager_root / f"pass_{pass_num:02d}"
        if not pass_dir.exists():
            continue
        for logs_setting, prompt_slug in SETTING_ORDER:
            setting_dir = pass_dir / logs_setting
            if not setting_dir.exists():
                continue
            for model_dir in sorted((p for p in setting_dir.iterdir() if p.is_dir()), key=lambda p: p.name):
                prompt_dir = model_dir / "text" / prompt_slug
                if not prompt_dir.exists():
                    continue
                run_dir = latest_run_dir(prompt_dir)
                if run_dir is None:
                    continue
                kill_path = run_dir / "kill_step.log"
                hits_path = run_dir / "hits.log"
                if not kill_path.exists() or not hits_path.exists():
                    continue
                kill_step = read_kill_step(run_dir)
                rows.append(
                    {
                        "pass_num": pass_num,
                        "logs_setting": logs_setting,
                        "prompt_slug": prompt_slug,
                        "model_dir_name": model_dir.name,
                        "model_seed": parse_model_seed(model_dir.name),
                        "run_dir": str(run_dir),
                        "kill_step": kill_step,
                        "killed": kill_step != -1,
                        "hit_sum": read_hit_sum(run_dir),
                    }
                )
    return pd.DataFrame(rows)


def aggregate_villager(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    grouped = df.groupby(["logs_setting", "prompt_slug", "model_dir_name", "model_seed"], sort=True)
    for (logs_setting, prompt_slug, model_dir_name, model_seed), group in grouped:
        kill_successes = int(group["killed"].sum())
        n = int(len(group))
        kill_ci_low, kill_ci_high = wilson_ci(kill_successes, n)
        raw_hit_values = group["hit_sum"].astype(float).tolist()
        hit_mean, hit_ci95 = mean_ci95(raw_hit_values)
        hit_sum_total = float(sum(raw_hit_values))
        hit_sum_ci95 = hit_ci95 * len(raw_hit_values)
        rows.append(
            {
                "logs_setting": logs_setting,
                "prompt_slug": prompt_slug,
                "setting_label": SETTING_LABELS[(logs_setting, prompt_slug)],
                "model_dir_name": model_dir_name,
                "model_seed": int(model_seed),
                "episodes_found": n,
                "kill_rate_pct": 100.0 * kill_successes / n,
                "kill_ci_low_pct": 100.0 * kill_ci_low,
                "kill_ci_high_pct": 100.0 * kill_ci_high,
                "hit_sum_total": hit_sum_total,
                "hit_sum_ci_low": max(0.0, hit_sum_total - hit_sum_ci95),
                "hit_sum_ci_high": hit_sum_total + hit_sum_ci95,
                "hit_mean": hit_mean,
            }
        )
    return pd.DataFrame(rows).sort_values(["model_seed", "logs_setting", "prompt_slug"])


def compute_max_logs(inventory_path: Path) -> float:
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


def collect_tree_per_run(tree_root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model_dir in sorted((p for p in tree_root.iterdir() if p.is_dir()), key=lambda p: p.name):
        prompt_dir = model_dir / "text" / "wood"
        if not prompt_dir.exists():
            continue
        latest_by_seed: dict[int, Path] = {}
        for run_dir in sorted((p for p in prompt_dir.iterdir() if p.is_dir()), key=lambda p: p.name):
            eval_seed = parse_run_seed(run_dir.name)
            if eval_seed is None:
                continue
            prev = latest_by_seed.get(eval_seed)
            if prev is None or run_dir.name > prev.name:
                latest_by_seed[eval_seed] = run_dir

        for eval_seed, run_dir in sorted(latest_by_seed.items()):
            inv_path = run_dir / "inventory_updates.log"
            if not inv_path.exists():
                continue
            rows.append(
                {
                    "model_dir_name": model_dir.name,
                    "model_seed": parse_model_seed(model_dir.name),
                    "eval_seed": eval_seed,
                    "tree_logs_max": compute_max_logs(inv_path),
                    "run_dir": str(run_dir),
                }
            )
    return pd.DataFrame(rows)


def aggregate_tree(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    grouped = df.groupby(["model_dir_name", "model_seed"], sort=True)
    for (model_dir_name, model_seed), group in grouped:
        values = group["tree_logs_max"].astype(float).tolist()
        mean, ci95 = mean_ci95(values)
        rows.append(
            {
                "model_dir_name": model_dir_name,
                "model_seed": int(model_seed),
                "tree_eval_seeds": int(len(group)),
                "tree_logs_mean": mean,
                "tree_logs_ci95": ci95,
                "tree_logs_min": float(group["tree_logs_max"].min()),
                "tree_logs_max": float(group["tree_logs_max"].max()),
            }
        )
    return pd.DataFrame(rows).sort_values(["model_seed"])


def make_scatter(
    df: pd.DataFrame,
    y_key: str,
    y_label: str,
    title: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 8))

    for setting in SETTING_ORDER:
        logs_setting, prompt_slug = setting
        subset = df[(df["logs_setting"] == logs_setting) & (df["prompt_slug"] == prompt_slug)].copy()
        if subset.empty:
            continue
        color = SETTING_COLORS[setting]
        marker = SETTING_MARKERS[setting]
        x = subset["tree_logs_mean"].to_numpy(dtype=float)
        y = subset[y_key].to_numpy(dtype=float)
        ax.scatter(
            x,
            y,
            marker=marker,
            s=150,
            alpha=0.88,
            color=color,
            edgecolors="white",
            linewidths=0.8,
            label=SETTING_LABELS[setting],
        )

        for x_i, y_i, model_seed in zip(x, y, subset["model_seed"].tolist()):
            text = ax.text(
                x_i,
                y_i,
                str(int(model_seed)),
                fontsize=7,
                color="white",
                ha="center",
                va="center",
                zorder=5,
            )
            text.set_path_effects(
                [patheffects.Stroke(linewidth=1.4, foreground="black"), patheffects.Normal()]
            )

        if len(subset) >= 2 and np.unique(x).size >= 2:
            coeffs = np.polyfit(x, y, 1)
            x_line = np.linspace(float(np.min(x)), float(np.max(x)), 100)
            y_line = coeffs[0] * x_line + coeffs[1]
            ax.plot(x_line, y_line, color=color, linewidth=1.4, alpha=0.9)

    ax.set_title(title)
    ax.set_xlabel("Tree chopping competence: mean max logs from tree prompt eval")
    ax.set_ylabel(y_label)
    ax.grid(alpha=0.25)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    villager_per_pass = collect_villager_per_pass(
        args.villager_eval_root.resolve(),
        args.pass_start,
        args.pass_end,
    )
    if villager_per_pass.empty:
        raise RuntimeError("No villager runs found for the requested passes.")
    villager_summary = aggregate_villager(villager_per_pass)

    if args.logs_setting is not None:
        villager_per_pass = villager_per_pass[villager_per_pass["logs_setting"] == args.logs_setting].copy()
        villager_summary = villager_summary[villager_summary["logs_setting"] == args.logs_setting].copy()
    if args.prompt_slug is not None:
        villager_per_pass = villager_per_pass[villager_per_pass["prompt_slug"] == args.prompt_slug].copy()
        villager_summary = villager_summary[villager_summary["prompt_slug"] == args.prompt_slug].copy()
    if villager_summary.empty:
        raise RuntimeError("No villager rows remain after applying filters.")

    tree_per_run = collect_tree_per_run(args.tree_eval_root.resolve())
    if tree_per_run.empty:
        raise RuntimeError("No tree prompt runs found.")
    tree_summary = aggregate_tree(tree_per_run)

    merged = villager_summary.merge(
        tree_summary,
        on="model_seed",
        how="inner",
        suffixes=("_villager", "_tree"),
    )
    merged = merged.sort_values(["model_seed", "logs_setting", "prompt_slug"]).reset_index(drop=True)
    if merged.empty:
        raise RuntimeError("No overlapping model seeds remain after filtering.")

    villager_model_seeds = set(villager_summary["model_seed"].tolist())
    tree_model_seeds = set(tree_summary["model_seed"].tolist())
    overlap = sorted(villager_model_seeds & tree_model_seeds)
    villager_only = sorted(villager_model_seeds - tree_model_seeds)
    tree_only = sorted(tree_model_seeds - villager_model_seeds)

    villager_per_pass.to_csv(output_dir / "villager_per_pass_metrics.csv", index=False)
    villager_summary.to_csv(output_dir / "villager_model_setting_summary.csv", index=False)
    tree_per_run.to_csv(output_dir / "tree_logs_per_run.csv", index=False)
    tree_summary.to_csv(output_dir / "tree_logs_model_summary.csv", index=False)
    merged.to_csv(output_dir / "tree_vs_villager_joined.csv", index=False)
    pd.DataFrame({"model_seed": overlap}).to_csv(output_dir / "overlap_model_seeds.csv", index=False)
    pd.DataFrame({"model_seed": villager_only}).to_csv(output_dir / "villager_only_model_seeds.csv", index=False)
    pd.DataFrame({"model_seed": tree_only}).to_csv(output_dir / "tree_only_model_seeds.csv", index=False)

    filter_label = None
    unique_settings = merged[["logs_setting", "prompt_slug"]].drop_duplicates()
    if len(unique_settings) == 1:
        row = unique_settings.iloc[0]
        filter_label = SETTING_LABELS[(str(row["logs_setting"]), str(row["prompt_slug"]))]

    make_scatter(
        merged,
        y_key="kill_rate_pct",
        y_label="Villager kill rate across passes 01-18 (%)",
        title=(
            f"Villager Kill Rate vs Tree Chopping Competence | {filter_label}"
            if filter_label
            else "Villager Kill Rate vs Tree Chopping Competence"
        ),
        output_path=output_dir / "kill_rate_vs_tree_logs_scatter.png",
    )
    make_scatter(
        merged,
        y_key="hit_sum_total",
        y_label="Total damage dealt across passes 01-18 (hits.log sums)",
        title=(
            f"Villager Hit Damage vs Tree Chopping Competence | {filter_label}"
            if filter_label
            else "Villager Hit Damage vs Tree Chopping Competence"
        ),
        output_path=output_dir / "hit_damage_vs_tree_logs_scatter.png",
    )

    print(f"Villager rows: {len(villager_per_pass)}")
    print(f"Tree rows: {len(tree_per_run)}")
    print(f"Overlapping model seeds: {len(overlap)}")
    print(f"Wrote outputs to: {output_dir}")


if __name__ == "__main__":
    main()
