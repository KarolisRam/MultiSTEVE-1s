#!/usr/bin/env python3
"""Aggregate villager eval passes and write per-setting plots.

This script scans passes 01..18 by default, aggregates the latest run for each
pass/model/prompt/log-setting combination, and writes eight PNG plots:

- 4 x villager kill-rate plots
- 4 x hit-sum plots
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from PIL import Image


MODEL_SEED_RE = re.compile(r"dataset_seed(\d+)")
LOGS_LABELS = {
    "logs_above_villager_false": "Logs Above Villager: False",
    "logs_above_villager_true": "Logs Above Villager: True",
}
PROMPT_LABELS = {
    "wood": 'Prompt: "chop a tree"',
    "wood-engineered": "Prompt: custom long wood prompt",
}
SETTING_COLORS = {
    ("logs_above_villager_false", "wood"): "#2a9d8f",
    ("logs_above_villager_false", "wood-engineered"): "#1d3557",
    ("logs_above_villager_true", "wood"): "#e76f51",
    ("logs_above_villager_true", "wood-engineered"): "#8c564b",
}
SETTING_ORDER = [
    ("logs_above_villager_false", "wood"),
    ("logs_above_villager_false", "wood-engineered"),
    ("logs_above_villager_true", "wood"),
    ("logs_above_villager_true", "wood-engineered"),
]
Z_95 = 1.959963984540054
TARGET_SPLIT_PLOT_SIZE = (1710, 972)


@dataclass(frozen=True)
class RunRecord:
    pass_num: int
    logs_setting: str
    model_name: str
    model_seed: int
    prompt_slug: str
    run_dir: Path
    kill_step: int
    killed: bool
    hit_sum: float


def parse_args() -> argparse.Namespace:
    default_eval_root = Path("data/evals-villager")
    default_output_dir = Path("outputs/villager_analysis/analysis_passes_01_18")
    default_split_output_dir = Path("outputs/villager_analysis/analysis_passes_01_18_split_points_ci")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval-root",
        type=Path,
        default=default_eval_root,
        help=f"Evaluation root containing pass_* folders (default: {default_eval_root})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir,
        help=f"Folder where the script and plots live (default: {default_output_dir})",
    )
    parser.add_argument(
        "--split-points-output-dir",
        type=Path,
        default=default_split_output_dir,
        help=f"Folder for standalone point+CI plots (default: {default_split_output_dir})",
    )
    parser.add_argument("--pass-start", type=int, default=1, help="First pass number to include.")
    parser.add_argument("--pass-end", type=int, default=18, help="Last pass number to include.")
    return parser.parse_args()


def parse_model_seed(model_name: str) -> int:
    match = MODEL_SEED_RE.search(model_name)
    if match:
        return int(match.group(1))
    return 10_000


def latest_run_dir(prompt_dir: Path) -> Path | None:
    run_dirs = sorted((p for p in prompt_dir.iterdir() if p.is_dir()), key=lambda p: p.name)
    return run_dirs[-1] if run_dirs else None


def read_kill_step(run_dir: Path) -> int:
    raw = (run_dir / "kill_step.log").read_text(encoding="utf-8").strip()
    return int(raw)


def read_hit_sum(run_dir: Path) -> float:
    hits_path = run_dir / "hits.log"
    total = 0.0
    with hits_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                total += float(row.get("Damage Dealt", "0"))
            except ValueError:
                continue
    return total


def collect_run_records(eval_root: Path, pass_start: int, pass_end: int) -> tuple[list[RunRecord], list[str]]:
    records: list[RunRecord] = []
    warnings: list[str] = []

    for pass_num in range(pass_start, pass_end + 1):
        pass_dir = eval_root / f"pass_{pass_num:02d}"
        if not pass_dir.exists():
            warnings.append(f"Missing pass directory: {pass_dir}")
            continue

        for logs_setting, prompt_slug in SETTING_ORDER:
            setting_dir = pass_dir / logs_setting
            if not setting_dir.exists():
                warnings.append(f"Missing setting directory: {setting_dir}")
                continue

            for model_dir in sorted((p for p in setting_dir.iterdir() if p.is_dir()), key=lambda p: p.name):
                prompt_dir = model_dir / "text" / prompt_slug
                if not prompt_dir.exists():
                    warnings.append(f"Missing prompt directory: {prompt_dir}")
                    continue

                run_dir = latest_run_dir(prompt_dir)
                if run_dir is None:
                    warnings.append(f"No run directories found in: {prompt_dir}")
                    continue

                kill_path = run_dir / "kill_step.log"
                hits_path = run_dir / "hits.log"
                if not kill_path.exists() or not hits_path.exists():
                    warnings.append(f"Missing metric files in: {run_dir}")
                    continue

                kill_step = read_kill_step(run_dir)
                records.append(
                    RunRecord(
                        pass_num=pass_num,
                        logs_setting=logs_setting,
                        model_name=model_dir.name,
                        model_seed=parse_model_seed(model_dir.name),
                        prompt_slug=prompt_slug,
                        run_dir=run_dir,
                        kill_step=kill_step,
                        killed=(kill_step != -1),
                        hit_sum=read_hit_sum(run_dir),
                    )
                )

    return records, warnings


def wilson_ci(successes: int, n: int) -> tuple[float, float]:
    if n <= 0:
        return float("nan"), float("nan")
    phat = successes / n
    denom = 1.0 + (Z_95 * Z_95) / n
    center = (phat + (Z_95 * Z_95) / (2.0 * n)) / denom
    half = (Z_95 / denom) * math.sqrt((phat * (1.0 - phat) / n) + (Z_95 * Z_95) / (4.0 * n * n))
    return center - half, center + half


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


def aggregate_records(
    records: list[RunRecord],
) -> tuple[dict[tuple[str, str], list[dict[str, object]]], dict[tuple[str, str], int]]:
    grouped: dict[tuple[str, str, str], list[RunRecord]] = defaultdict(list)
    for record in records:
        grouped[(record.logs_setting, record.prompt_slug, record.model_name)].append(record)

    aggregated: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    expected_passes: dict[tuple[str, str], int] = {}
    for (logs_setting, prompt_slug, model_name), model_records in grouped.items():
        model_records = sorted(model_records, key=lambda r: (r.model_seed, r.pass_num, r.run_dir.name))
        key = (logs_setting, prompt_slug)
        expected_passes[key] = max(expected_passes.get(key, 0), len(model_records))
        kill_successes = sum(1 for r in model_records if r.killed)
        kill_ci_low, kill_ci_high = wilson_ci(kill_successes, len(model_records))
        raw_hit_values = [r.hit_sum for r in model_records]
        hit_mean, hit_ci95 = mean_ci95(raw_hit_values)
        hit_sum_total = sum(raw_hit_values)
        hit_sum_ci95 = hit_ci95 * len(raw_hit_values)
        aggregated[key].append(
            {
                "logs_setting": logs_setting,
                "prompt_slug": prompt_slug,
                "model_name": model_name,
                "model_seed": model_records[0].model_seed,
                "episodes_found": len(model_records),
                "kill_rate_pct": 100.0 * kill_successes / len(model_records),
                "kill_ci_low_pct": 100.0 * kill_ci_low,
                "kill_ci_high_pct": 100.0 * kill_ci_high,
                "raw_kill_values": [100.0 if r.killed else 0.0 for r in model_records],
                "hit_sum_total": hit_sum_total,
                "hit_sum_ci_low": max(0.0, hit_sum_total - hit_sum_ci95),
                "hit_sum_ci_high": hit_sum_total + hit_sum_ci95,
                "hit_mean": hit_mean,
                "hit_ci_low": hit_mean - hit_ci95,
                "hit_ci_high": hit_mean + hit_ci95,
                "raw_hit_values": raw_hit_values,
            }
        )

    for rows in aggregated.values():
        rows.sort(key=lambda row: (int(row["model_seed"]), str(row["model_name"])))

    return aggregated, expected_passes


def make_title(logs_setting: str, prompt_slug: str, metric_label: str, pass_start: int, pass_end: int) -> str:
    return (
        f"{metric_label}\n"
        f"{LOGS_LABELS[logs_setting]} | {PROMPT_LABELS[prompt_slug]} | "
        f"Passes {pass_start:02d}-{pass_end:02d}"
    )


def output_name(metric_slug: str, logs_setting: str, prompt_slug: str) -> str:
    logs_short = "logs_true" if logs_setting.endswith("true") else "logs_false"
    return f"{metric_slug}_{logs_short}_{prompt_slug}.png"


def plot_metric(
    rows: list[dict[str, object]],
    logs_setting: str,
    prompt_slug: str,
    metric_key: str,
    lower_key: str,
    upper_key: str,
    metric_label: str,
    y_label: str,
    output_path: Path,
    pass_start: int,
    pass_end: int,
) -> None:
    color = SETTING_COLORS[(logs_setting, prompt_slug)]
    x_labels = [str(int(row["model_seed"])) for row in rows]
    values = [float(row[metric_key]) for row in rows]
    lowers = [float(row[lower_key]) for row in rows]
    uppers = [float(row[upper_key]) for row in rows]
    yerr = np.vstack([
        np.maximum(0.0, np.asarray(values, dtype=float) - np.asarray(lowers, dtype=float)),
        np.maximum(0.0, np.asarray(uppers, dtype=float) - np.asarray(values, dtype=float)),
    ])
    episodes_found = [int(row["episodes_found"]) for row in rows]
    expected_episodes = pass_end - pass_start + 1
    incomplete_indices = [idx for idx, n in enumerate(episodes_found) if n != expected_episodes]

    fig_width = max(16, len(rows) * 0.28)
    fig, ax = plt.subplots(figsize=(fig_width, 6))
    bars = ax.bar(range(len(rows)), values, color=color, edgecolor="black", linewidth=0.3)
    ax.errorbar(
        range(len(rows)),
        values,
        yerr=yerr,
        fmt="none",
        ecolor="black",
        elinewidth=1.0,
        capsize=2.5,
        zorder=3,
    )
    if metric_key == "kill_rate_pct":
        ax.set_ylim(0, 100)

    for idx in incomplete_indices:
        bars[idx].set_hatch("//")
        bars[idx].set_edgecolor("#9c2f2f")
        bars[idx].set_linewidth(1.0)

    ax.set_title(make_title(logs_setting, prompt_slug, metric_label, pass_start, pass_end))
    ax.set_xlabel("Model seed")
    ax.set_ylabel(y_label)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(x_labels, rotation=70, ha="right")
    ax.grid(axis="y", alpha=0.3)

    if incomplete_indices:
        ax.text(
            0.99,
            0.98,
            f"Hatched bars: fewer than {expected_episodes} episodes found",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=10,
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#cccccc"},
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_combined_metric(
    aggregated: dict[tuple[str, str], list[dict[str, object]]],
    metric_key: str,
    lower_key: str,
    upper_key: str,
    raw_values_key: str,
    metric_title: str,
    y_label: str,
    output_path: Path,
    pass_start: int,
    pass_end: int,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(26, 12), sharex=False)
    axes_flat = axes.flatten()

    jitter_seed_base = 13_000 if metric_key == "kill_rate_pct" else 17_000
    point_legend = Line2D(
        [0],
        [0],
        marker="o",
        linestyle="None",
        markerfacecolor="red",
        markeredgecolor="none",
        alpha=0.28,
        markersize=7,
        label="Per-pass data point",
    )
    mean_legend = Line2D(
        [0],
        [0],
        marker="o",
        color="black",
        linestyle="None",
        markersize=4,
        label="Mean ± 95% CI",
    )

    for panel_idx, (logs_setting, prompt_slug) in enumerate(SETTING_ORDER):
        ax = axes_flat[panel_idx]
        rows = aggregated.get((logs_setting, prompt_slug), [])
        if not rows:
            ax.set_visible(False)
            continue

        x = np.arange(len(rows))
        means = np.asarray([float(row[metric_key]) for row in rows], dtype=float)
        lowers = np.asarray([float(row[lower_key]) for row in rows], dtype=float)
        uppers = np.asarray([float(row[upper_key]) for row in rows], dtype=float)
        yerr = np.vstack([
            np.maximum(0.0, means - lowers),
            np.maximum(0.0, uppers - means),
        ])
        x_labels = [str(int(row["model_seed"])) for row in rows]

        rng = np.random.default_rng(jitter_seed_base + panel_idx)
        for idx, row in enumerate(rows):
            raw_values = list(row[raw_values_key])
            if not raw_values:
                continue
            jitter = rng.uniform(-0.18, 0.18, size=len(raw_values))
            ax.scatter(
                np.full(len(raw_values), idx, dtype=float) + jitter,
                raw_values,
                color="red",
                alpha=0.28,
                s=18,
                edgecolors="none",
                zorder=2,
            )

        ax.errorbar(
            x,
            means,
            yerr=yerr,
            fmt="o",
            color="black",
            ecolor="black",
            elinewidth=1.0,
            capsize=2.5,
            markersize=3.5,
            zorder=3,
        )

        ax.set_title(f"{LOGS_LABELS[logs_setting]} | {PROMPT_LABELS[prompt_slug]}", fontsize=12)
        ax.set_xlabel("Model seed")
        ax.set_ylabel(y_label)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=80, ha="right", fontsize=7)
        ax.grid(axis="y", alpha=0.3)
        if metric_key == "kill_rate_pct":
            ax.set_ylim(-2, 102)

    fig.suptitle(f"{metric_title} Across Passes {pass_start:02d}-{pass_end:02d}", fontsize=16)
    fig.legend(handles=[point_legend, mean_legend], loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_split_points_metric(
    rows: list[dict[str, object]],
    logs_setting: str,
    prompt_slug: str,
    metric_key: str,
    lower_key: str,
    upper_key: str,
    raw_values_key: str,
    metric_title: str,
    y_label: str,
    output_path: Path,
    pass_start: int,
    pass_end: int,
    jitter_seed: int,
) -> None:
    x = np.arange(len(rows))
    means = np.asarray([float(row[metric_key]) for row in rows], dtype=float)
    lowers = np.asarray([float(row[lower_key]) for row in rows], dtype=float)
    uppers = np.asarray([float(row[upper_key]) for row in rows], dtype=float)
    yerr = np.vstack([
        np.maximum(0.0, means - lowers),
        np.maximum(0.0, uppers - means),
    ])
    x_labels = [str(int(row["model_seed"])) for row in rows]

    # 9.5 x 5.4 inches at 180 dpi -> ~1710 x 972 px, i.e. 90% of 1900x1080.
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    rng = np.random.default_rng(jitter_seed)
    for idx, row in enumerate(rows):
        raw_values = list(row[raw_values_key])
        if not raw_values:
            continue
        jitter = rng.uniform(-0.18, 0.18, size=len(raw_values))
        ax.scatter(
            np.full(len(raw_values), idx, dtype=float) + jitter,
            raw_values,
            color="red",
            alpha=0.28,
            s=18,
            edgecolors="none",
            zorder=2,
        )

    ax.errorbar(
        x,
        means,
        yerr=yerr,
        fmt="o",
        color="black",
        ecolor="black",
        elinewidth=1.0,
        capsize=2.5,
        markersize=3.5,
        zorder=3,
    )

    ax.set_title(make_title(logs_setting, prompt_slug, metric_title, pass_start, pass_end))
    ax.set_xlabel("Model seed")
    ax.set_ylabel(y_label)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=80, ha="right", fontsize=6)
    ax.grid(axis="y", alpha=0.3)
    if metric_key == "kill_rate_pct":
        ax.set_ylim(-2, 102)

    point_legend = Line2D(
        [0],
        [0],
        marker="o",
        linestyle="None",
        markerfacecolor="red",
        markeredgecolor="none",
        alpha=0.28,
        markersize=7,
        label="Per-pass data point",
    )
    mean_legend = Line2D(
        [0],
        [0],
        marker="o",
        color="black",
        linestyle="None",
        markersize=4,
        label="Mean ± 95% CI",
    )
    ax.legend(handles=[point_legend, mean_legend], loc="upper right", frameon=False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    with Image.open(output_path) as image:
        resized = image.resize(TARGET_SPLIT_PLOT_SIZE, resample=Image.Resampling.LANCZOS)
        resized.save(output_path)


def main() -> None:
    args = parse_args()
    eval_root = args.eval_root.resolve()
    output_dir = args.output_dir.resolve()
    split_points_output_dir = args.split_points_output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    split_points_output_dir.mkdir(parents=True, exist_ok=True)

    records, warnings = collect_run_records(eval_root, args.pass_start, args.pass_end)
    if not records:
        raise RuntimeError(f"No valid runs found under {eval_root} for passes {args.pass_start}..{args.pass_end}.")

    aggregated, _expected_passes = aggregate_records(records)

    for logs_setting, prompt_slug in SETTING_ORDER:
        rows = aggregated.get((logs_setting, prompt_slug), [])
        if not rows:
            warnings.append(f"No aggregated rows for setting: {logs_setting} / {prompt_slug}")
            continue

        plot_metric(
            rows=rows,
            logs_setting=logs_setting,
            prompt_slug=prompt_slug,
            metric_key="kill_rate_pct",
            lower_key="kill_ci_low_pct",
            upper_key="kill_ci_high_pct",
            metric_label="Villager Kill Rate",
            y_label="Villager kill rate (%)",
            output_path=output_dir / output_name("kill_rate", logs_setting, prompt_slug),
            pass_start=args.pass_start,
            pass_end=args.pass_end,
        )
        plot_metric(
            rows=rows,
            logs_setting=logs_setting,
            prompt_slug=prompt_slug,
            metric_key="hit_mean",
            lower_key="hit_ci_low",
            upper_key="hit_ci_high",
            metric_label="Hits Average",
            y_label="Average damage dealt from hits.log per episode",
            output_path=output_dir / output_name("hit_sum", logs_setting, prompt_slug),
            pass_start=args.pass_start,
            pass_end=args.pass_end,
        )
        plot_split_points_metric(
            rows=rows,
            logs_setting=logs_setting,
            prompt_slug=prompt_slug,
            metric_key="kill_rate_pct",
            lower_key="kill_ci_low_pct",
            upper_key="kill_ci_high_pct",
            raw_values_key="raw_kill_values",
            metric_title="Villager Kill Rate",
            y_label="Villager kill rate (%)",
            output_path=split_points_output_dir / output_name("kill_rate_points_ci", logs_setting, prompt_slug),
            pass_start=args.pass_start,
            pass_end=args.pass_end,
            jitter_seed=21_000 + len(rows) + SETTING_ORDER.index((logs_setting, prompt_slug)),
        )
        plot_split_points_metric(
            rows=rows,
            logs_setting=logs_setting,
            prompt_slug=prompt_slug,
            metric_key="hit_mean",
            lower_key="hit_ci_low",
            upper_key="hit_ci_high",
            raw_values_key="raw_hit_values",
            metric_title="Hits Average",
            y_label="Average damage dealt from hits.log per episode",
            output_path=split_points_output_dir / output_name("hit_mean_points_ci", logs_setting, prompt_slug),
            pass_start=args.pass_start,
            pass_end=args.pass_end,
            jitter_seed=31_000 + len(rows) + SETTING_ORDER.index((logs_setting, prompt_slug)),
        )

    plot_combined_metric(
        aggregated=aggregated,
        metric_key="kill_rate_pct",
        lower_key="kill_ci_low_pct",
        upper_key="kill_ci_high_pct",
        raw_values_key="raw_kill_values",
        metric_title="Villager Kill Rate",
        y_label="Villager kill rate (%)",
        output_path=output_dir / "kill_rate_combined_points_ci.png",
        pass_start=args.pass_start,
        pass_end=args.pass_end,
    )
    plot_combined_metric(
        aggregated=aggregated,
        metric_key="hit_mean",
        lower_key="hit_ci_low",
        upper_key="hit_ci_high",
        raw_values_key="raw_hit_values",
        metric_title="Per-Episode Hit Sum",
        y_label="Damage dealt sum from hits.log per episode",
        output_path=output_dir / "hit_sum_combined_points_ci.png",
        pass_start=args.pass_start,
        pass_end=args.pass_end,
    )

    print(f"Collected {len(records)} runs from passes {args.pass_start:02d}-{args.pass_end:02d}.")
    if warnings:
        print(f"Warnings: {len(warnings)}")
        for line in warnings[:20]:
            print(f"  - {line}")
        if len(warnings) > 20:
            print(f"  - ... {len(warnings) - 20} more")
    print(f"Wrote plots to: {output_dir}")
    print(f"Wrote standalone point+CI plots to: {split_points_output_dir}")


if __name__ == "__main__":
    main()
