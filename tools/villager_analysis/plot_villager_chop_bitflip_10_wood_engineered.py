#!/usr/bin/env python3
"""Aggregate villager-chop bitflip results for the wood-engineered prompt."""

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
from matplotlib.lines import Line2D

try:
    from scipy.stats import t as student_t
except Exception:
    student_t = None

BITFLIP_RE = re.compile(r"bitflip_seed(\d+)")
PROMPT_SLUG = "wood-engineered"
SETTING_ORDER = [
    "logs_above_villager_false",
    "logs_above_villager_true",
]
SETTING_LABELS = {
    "logs_above_villager_false": "logs=false | engineered prompt",
    "logs_above_villager_true": "logs=true | engineered prompt",
}
Z_95 = 1.959963984540054
RAW_POINT_ALPHA = 0.28
RAW_POINT_SIZE = 18
RAW_JITTER_HALF_WIDTH = 0.18
RAW_POINT_COLOR = "#D55E00"


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval-root",
        type=Path,
        default=Path("data/evals-villager/villager_chop_bitflip_10_wood_engineered"),
        help="Root containing villager_chop_bitflip_10_wood_engineered pass folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir,
        help="Directory for plots and CSV summaries.",
    )
    parser.add_argument("--pass-start", type=int, default=1, help="First pass number to include.")
    parser.add_argument("--pass-end", type=int, default=20, help="Last pass number to include.")
    return parser.parse_args()


def parse_bitflip_seed(model_name: str) -> int:
    match = BITFLIP_RE.search(model_name)
    return int(match.group(1)) if match else 10_000


def latest_run_dir(prompt_dir: Path) -> Path | None:
    run_dirs = sorted((p for p in prompt_dir.iterdir() if p.is_dir()), key=lambda p: p.name)
    return run_dirs[-1] if run_dirs else None


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


def collect_run_records(eval_root: Path, pass_start: int, pass_end: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for pass_num in range(pass_start, pass_end + 1):
        pass_dir = eval_root / f"pass_{pass_num:02d}"
        if not pass_dir.exists():
            continue
        for logs_setting in SETTING_ORDER:
            setting_dir = pass_dir / logs_setting
            if not setting_dir.exists():
                continue
            for model_dir in sorted((p for p in setting_dir.iterdir() if p.is_dir()), key=lambda p: p.name):
                prompt_dir = model_dir / "text" / PROMPT_SLUG
                if not prompt_dir.exists():
                    continue
                run_dir = latest_run_dir(prompt_dir)
                if run_dir is None:
                    continue
                if not (run_dir / "kill_step.log").exists() or not (run_dir / "hits.log").exists():
                    continue
                kill_step = read_kill_step(run_dir)
                rows.append(
                    {
                        "pass_num": pass_num,
                        "logs_setting": logs_setting,
                        "model_name": model_dir.name,
                        "bitflip_seed": parse_bitflip_seed(model_dir.name),
                        "model_label": str(parse_bitflip_seed(model_dir.name)),
                        "run_dir": str(run_dir),
                        "kill_step": kill_step,
                        "killed": kill_step != -1,
                        "hit_sum": read_hit_sum(run_dir),
                    }
                )
    return pd.DataFrame(rows)


def aggregate_records(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    grouped = df.groupby(["logs_setting", "bitflip_seed", "model_label", "model_name"], sort=True)
    for (logs_setting, bitflip_seed, model_label, model_name), group in grouped:
        n = int(len(group))
        kill_successes = int(group["killed"].sum())
        kill_ci_low, kill_ci_high = wilson_ci(kill_successes, n)
        kill_raw_values_pct = [100.0 if bool(value) else 0.0 for value in group["killed"].tolist()]
        hit_raw_values = group["hit_sum"].astype(float).tolist()
        hit_mean, hit_ci95 = mean_ci95(hit_raw_values)
        rows.append(
            {
                "logs_setting": logs_setting,
                "setting_label": SETTING_LABELS[logs_setting],
                "bitflip_seed": int(bitflip_seed),
                "model_label": model_label,
                "model_name": model_name,
                "episodes_found": n,
                "kill_rate_pct": 100.0 * kill_successes / n,
                "kill_ci_low_pct": 100.0 * kill_ci_low,
                "kill_ci_high_pct": 100.0 * kill_ci_high,
                "kill_raw_values_pct": kill_raw_values_pct,
                "hit_mean": hit_mean,
                "hit_ci_low": hit_mean - hit_ci95,
                "hit_ci_high": hit_mean + hit_ci95,
                "hit_raw_values": hit_raw_values,
            }
        )
    return pd.DataFrame(rows).sort_values(["logs_setting", "bitflip_seed", "model_name"]).reset_index(drop=True)


def metric_spec(metric_key: str) -> dict[str, str]:
    if metric_key == "kill_rate_pct":
        return {
            "low_key": "kill_ci_low_pct",
            "high_key": "kill_ci_high_pct",
            "raw_values_key": "kill_raw_values_pct",
            "y_label": "Villager kill rate across passes 01-20 (%)",
            "title": "Kill Rate",
        }
    if metric_key == "hit_mean":
        return {
            "low_key": "hit_ci_low",
            "high_key": "hit_ci_high",
            "raw_values_key": "hit_raw_values",
            "y_label": "Average damage dealt per pass",
            "title": "Average Hit Damage",
        }
    raise KeyError(f"Unsupported metric key: {metric_key}")


def plot_setting_with_raw_points(
    ax: plt.Axes,
    subset: pd.DataFrame,
    metric_key: str,
    low_key: str,
    high_key: str,
    raw_values_key: str,
    setting_label: str,
    jitter_seed: int,
) -> None:
    if subset.empty:
        ax.set_visible(False)
        return

    subset = subset.sort_values(["bitflip_seed", "model_name"]).reset_index(drop=True)
    x = np.arange(len(subset))
    means = subset[metric_key].to_numpy(dtype=float)
    lowers = subset[low_key].to_numpy(dtype=float)
    uppers = subset[high_key].to_numpy(dtype=float)
    yerr = np.vstack([np.maximum(0.0, means - lowers), np.maximum(0.0, uppers - means)])
    x_labels = subset["model_label"].astype(str).tolist()

    rng = np.random.default_rng(jitter_seed)
    for idx, row in subset.iterrows():
        raw_values = list(row[raw_values_key])
        if not raw_values:
            continue
        jitter = rng.uniform(-RAW_JITTER_HALF_WIDTH, RAW_JITTER_HALF_WIDTH, size=len(raw_values))
        ax.scatter(
            np.full(len(raw_values), idx, dtype=float) + jitter,
            raw_values,
            color=RAW_POINT_COLOR,
            alpha=RAW_POINT_ALPHA,
            s=RAW_POINT_SIZE,
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

    ax.set_title(setting_label, fontsize=12)
    ax.set_xlabel("Bitflip seed")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.grid(axis="y", alpha=0.3)
    if metric_key == "kill_rate_pct":
        ax.set_ylim(-2, 102)


def make_combined_plot(summary_df: pd.DataFrame, metric_key: str, output_path: Path) -> None:
    spec = metric_spec(metric_key)
    fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharey=True)

    for idx, logs_setting in enumerate(SETTING_ORDER):
        subset = summary_df[summary_df["logs_setting"] == logs_setting].copy()
        plot_setting_with_raw_points(
            ax=axes[idx],
            subset=subset,
            metric_key=metric_key,
            low_key=spec["low_key"],
            high_key=spec["high_key"],
            raw_values_key=spec["raw_values_key"],
            setting_label=SETTING_LABELS[logs_setting],
            jitter_seed=(13_000 if metric_key == "kill_rate_pct" else 17_000) + idx,
        )
        axes[idx].set_ylabel(spec["y_label"])

    point_legend = Line2D(
        [0],
        [0],
        marker="o",
        linestyle="None",
        markerfacecolor=RAW_POINT_COLOR,
        markeredgecolor="none",
        alpha=RAW_POINT_ALPHA,
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
    fig.suptitle(f"Bitflip 10 | {spec['title']} | wood-engineered", fontsize=16, y=0.985)
    fig.legend(
        handles=[point_legend, mean_legend],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=2,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def make_split_plots(summary_df: pd.DataFrame, metric_key: str, output_dir: Path) -> None:
    spec = metric_spec(metric_key)
    point_legend = Line2D(
        [0],
        [0],
        marker="o",
        linestyle="None",
        markerfacecolor=RAW_POINT_COLOR,
        markeredgecolor="none",
        alpha=RAW_POINT_ALPHA,
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

    for idx, logs_setting in enumerate(SETTING_ORDER):
        subset = summary_df[summary_df["logs_setting"] == logs_setting].copy()
        if subset.empty:
            continue
        fig, ax = plt.subplots(figsize=(9.5, 5.4))
        plot_setting_with_raw_points(
            ax=ax,
            subset=subset,
            metric_key=metric_key,
            low_key=spec["low_key"],
            high_key=spec["high_key"],
            raw_values_key=spec["raw_values_key"],
            setting_label=SETTING_LABELS[logs_setting],
            jitter_seed=(21_000 if metric_key == "kill_rate_pct" else 31_000) + idx,
        )
        ax.set_ylabel(spec["y_label"])
        ax.legend(handles=[point_legend, mean_legend], loc="upper right", frameon=False)
        fig.suptitle(f"{SETTING_LABELS[logs_setting]} | {spec['title']}", fontsize=14)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(output_dir / f"{metric_key}_{logs_setting}.png", dpi=180)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    per_pass_df = collect_run_records(args.eval_root.resolve(), args.pass_start, args.pass_end)
    if per_pass_df.empty:
        raise RuntimeError("No villager chop runs found.")
    summary_df = aggregate_records(per_pass_df)

    per_pass_df.to_csv(output_dir / "villager_chop_bitflip_10_wood_engineered_per_pass.csv", index=False)
    summary_df.to_csv(output_dir / "villager_chop_bitflip_10_wood_engineered_summary.csv", index=False)
    (
        summary_df[["logs_setting", "bitflip_seed", "episodes_found"]]
        .sort_values(["logs_setting", "bitflip_seed"])
        .to_csv(output_dir / "villager_chop_bitflip_10_wood_engineered_episode_counts.csv", index=False)
    )

    make_combined_plot(summary_df, metric_key="kill_rate_pct", output_path=output_dir / "kill_rate_combined.png")
    make_combined_plot(summary_df, metric_key="hit_mean", output_path=output_dir / "hit_mean_combined.png")
    make_split_plots(summary_df, metric_key="kill_rate_pct", output_dir=output_dir)
    make_split_plots(summary_df, metric_key="hit_mean", output_dir=output_dir)

    counts = (
        summary_df[["bitflip_seed"]]
        .drop_duplicates()
        .sort_values("bitflip_seed")
        .shape[0]
    )
    print(f"Per-pass rows: {len(per_pass_df)}")
    print(f"Summary rows: {len(summary_df)}")
    print(f"Bitflip models found: {counts}")
    print(f"Wrote outputs to: {output_dir}")


if __name__ == "__main__":
    main()
