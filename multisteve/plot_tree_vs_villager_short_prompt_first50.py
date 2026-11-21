#!/usr/bin/env python3
"""Compare short tree-prompt competence against villager outcomes for seeds 101..150."""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patheffects

MODEL_SEED_RE = re.compile(r"dataset_seed(\d+)")
RUN_SEED_RE = re.compile(r"^seed_(\d+)_")
Z_95 = 1.959963984540054
PLOT_COLOR = "#e76f51"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--villager-eval-root",
        type=Path,
        default=Path("evals-villager"),
        help="Root containing villager pass_XX folders.",
    )
    parser.add_argument(
        "--tree-eval-root",
        type=Path,
        default=Path("final-evals/05_tree_prompt_100_models_short"),
        help="Root containing short tree-prompt rollout folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "final-evals/05_tree_prompt_100_models_short_vs_villager_logs_true_wood_first50"
        ),
        help="Directory for plots and CSV outputs.",
    )
    parser.add_argument("--pass-start", type=int, default=1, help="First villager pass to include.")
    parser.add_argument("--pass-end", type=int, default=20, help="Last villager pass to include.")
    parser.add_argument("--model-seed-min", type=int, default=101, help="Minimum model seed to include.")
    parser.add_argument("--model-seed-max", type=int, default=150, help="Maximum model seed to include.")
    return parser.parse_args()


def parse_model_seed(name: str) -> int | None:
    match = MODEL_SEED_RE.search(name)
    return int(match.group(1)) if match else None


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
    return mean, Z_95 * sem


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
            current_logs = sum(value for key, value in inventory.items() if "_log" in key)
            max_logs = max(max_logs, current_logs)
    return float(max_logs)


def collect_villager_per_pass(
    villager_root: Path,
    pass_start: int,
    pass_end: int,
    model_seed_min: int,
    model_seed_max: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for pass_num in range(pass_start, pass_end + 1):
        pass_dir = villager_root / f"pass_{pass_num:02d}" / "logs_above_villager_true"
        if not pass_dir.exists():
            continue
        for model_dir in sorted((p for p in pass_dir.iterdir() if p.is_dir()), key=lambda p: p.name):
            model_seed = parse_model_seed(model_dir.name)
            if model_seed is None or not (model_seed_min <= model_seed <= model_seed_max):
                continue
            prompt_dir = model_dir / "text" / "wood"
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
                    "model_dir_name": model_dir.name,
                    "model_seed": model_seed,
                    "run_dir": str(run_dir),
                    "kill_step": kill_step,
                    "killed": kill_step != -1,
                    "hit_sum": read_hit_sum(run_dir),
                }
            )
    return pd.DataFrame(rows)


def aggregate_villager(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    grouped = df.groupby(["model_dir_name", "model_seed"], sort=True)
    for (model_dir_name, model_seed), group in grouped:
        hit_values = group["hit_sum"].astype(float).tolist()
        hit_mean, hit_ci95 = mean_ci95(hit_values)
        rows.append(
            {
                "model_dir_name": model_dir_name,
                "model_seed": int(model_seed),
                "villager_passes_found": int(len(group)),
                "kill_rate_pct": 100.0 * float(group["killed"].mean()),
                "hit_damage_total": float(sum(hit_values)),
                "hit_damage_mean": hit_mean,
                "hit_damage_mean_ci95": hit_ci95,
            }
        )
    return pd.DataFrame(rows).sort_values(["model_seed"]).reset_index(drop=True)


def collect_tree_per_run(
    tree_root: Path,
    model_seed_min: int,
    model_seed_max: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model_dir in sorted((p for p in tree_root.iterdir() if p.is_dir()), key=lambda p: p.name):
        model_seed = parse_model_seed(model_dir.name)
        if model_seed is None or not (model_seed_min <= model_seed <= model_seed_max):
            continue
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
                    "model_seed": model_seed,
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
                "tree_eval_seeds_found": int(len(group)),
                "tree_logs_mean": mean,
                "tree_logs_ci95": ci95,
                "tree_logs_min": float(group["tree_logs_max"].min()),
                "tree_logs_max": float(group["tree_logs_max"].max()),
            }
        )
    return pd.DataFrame(rows).sort_values(["model_seed"]).reset_index(drop=True)


def make_scatter(
    df: pd.DataFrame,
    y_key: str,
    y_label: str,
    title: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 8))
    x = df["tree_logs_mean"].to_numpy(dtype=float)
    y = df[y_key].to_numpy(dtype=float)
    r_squared: float | None = None

    ax.scatter(
        x,
        y,
        s=155,
        alpha=0.9,
        color=PLOT_COLOR,
        edgecolors="white",
        linewidths=0.8,
        zorder=3,
    )

    for x_i, y_i, model_seed in zip(x, y, df["model_seed"].tolist()):
        text = ax.text(
            x_i,
            y_i,
            str(int(model_seed)),
            fontsize=7,
            color="white",
            ha="center",
            va="center",
            zorder=4,
        )
        text.set_path_effects(
            [patheffects.Stroke(linewidth=1.4, foreground="black"), patheffects.Normal()]
        )

    if len(df) >= 2 and np.unique(x).size >= 2:
        coeffs = np.polyfit(x, y, 1)
        x_line = np.linspace(float(np.min(x)), float(np.max(x)), 100)
        y_line = coeffs[0] * x_line + coeffs[1]
        y_hat = coeffs[0] * x + coeffs[1]
        ss_res = float(np.sum((y - y_hat) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0.0 else float("nan")
        ax.plot(x_line, y_line, color=PLOT_COLOR, linewidth=1.5, alpha=0.95, zorder=2)

    if r_squared is not None:
        ax.text(
            0.03,
            0.97,
            f"$R^2$ = {r_squared:.4f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=11,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "0.7", "alpha": 0.9},
        )

    ax.set_title(title)
    ax.set_xlabel("Tree chopping competence: mean max logs from short tree prompt eval")
    ax.set_ylabel(y_label)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    villager_per_pass = collect_villager_per_pass(
        villager_root=args.villager_eval_root.resolve(),
        pass_start=args.pass_start,
        pass_end=args.pass_end,
        model_seed_min=args.model_seed_min,
        model_seed_max=args.model_seed_max,
    )
    if villager_per_pass.empty:
        raise RuntimeError("No villager runs found for the requested filters.")
    villager_summary = aggregate_villager(villager_per_pass)

    tree_per_run = collect_tree_per_run(
        tree_root=args.tree_eval_root.resolve(),
        model_seed_min=args.model_seed_min,
        model_seed_max=args.model_seed_max,
    )
    if tree_per_run.empty:
        raise RuntimeError("No short tree-prompt runs found for the requested seed range.")
    tree_summary = aggregate_tree(tree_per_run)

    merged = villager_summary.merge(
        tree_summary,
        on=["model_dir_name", "model_seed"],
        how="inner",
    ).sort_values(["model_seed"]).reset_index(drop=True)
    if merged.empty:
        raise RuntimeError("No overlapping model seeds remain after joining the datasets.")

    villager_per_pass.to_csv(output_dir / "villager_per_pass_metrics.csv", index=False)
    villager_summary.to_csv(output_dir / "villager_model_summary.csv", index=False)
    tree_per_run.to_csv(output_dir / "tree_logs_per_run.csv", index=False)
    tree_summary.to_csv(output_dir / "tree_logs_model_summary.csv", index=False)
    merged.to_csv(output_dir / "tree_vs_villager_joined.csv", index=False)

    make_scatter(
        merged,
        y_key="kill_rate_pct",
        y_label=f"Villager kill rate across passes {args.pass_start:02d}-{args.pass_end:02d} (%)",
        title="Villager Kill Rate vs Tree Chopping Competence | logs=true | prompt=\"chop a tree\" | seeds 101-150",
        output_path=output_dir / "kill_rate_vs_tree_logs_scatter.png",
    )
    make_scatter(
        merged,
        y_key="hit_damage_total",
        y_label=f"Total damage dealt across passes {args.pass_start:02d}-{args.pass_end:02d} (hits.log sums)",
        title="Villager Hit Damage vs Tree Chopping Competence | logs=true | prompt=\"chop a tree\" | seeds 101-150",
        output_path=output_dir / "hit_damage_vs_tree_logs_scatter.png",
    )

    print(f"Villager per-pass rows: {len(villager_per_pass)}")
    print(f"Villager models: {len(villager_summary)}")
    print(f"Tree per-run rows: {len(tree_per_run)}")
    print(f"Tree models: {len(tree_summary)}")
    print(f"Joined models: {len(merged)}")
    print(f"Wrote outputs to: {output_dir}")


if __name__ == "__main__":
    main()
