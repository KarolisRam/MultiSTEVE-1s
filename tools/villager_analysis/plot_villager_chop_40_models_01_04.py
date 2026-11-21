#!/usr/bin/env python3
"""Aggregate villager-chop results for the 01..04 model groups across passes."""

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
from matplotlib.lines import Line2D

try:
    from scipy.stats import t as student_t
except Exception:
    student_t = None

MODEL_SEED_RE = re.compile(r"dataset_seed(\d+)")
GROUP_ID_RE = re.compile(r"^(0[1-4])_")
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
    ("logs_above_villager_false", "wood"): "#0072B2",
    ("logs_above_villager_false", "wood-engineered"): "#E69F00",
    ("logs_above_villager_true", "wood"): "#009E73",
    ("logs_above_villager_true", "wood-engineered"): "#CC79A7",
}
SETTING_MARKERS = {
    ("logs_above_villager_false", "wood"): "o",
    ("logs_above_villager_false", "wood-engineered"): "s",
    ("logs_above_villager_true", "wood"): "^",
    ("logs_above_villager_true", "wood-engineered"): "D",
}
GROUP_LABELS = {
    "01": "Group 1: Init Seed Only",
    "02": "Group 2: Data Seed Only",
    "03": "Group 3: Init + Data",
    "04": "Group 4: Single LSB Flip",
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
        default=Path("data/evals-villager/villager_chop_40_models_01_04"),
        help="Root containing villager_chop_40_models_01_04 pass folders.",
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


def parse_model_seed(name: str) -> int:
    match = MODEL_SEED_RE.search(name)
    return int(match.group(1)) if match else 10_000


def parse_group_id(model_name: str) -> str:
    match = GROUP_ID_RE.match(model_name)
    return match.group(1) if match else "unknown"


def parse_group_sort_value(model_name: str) -> int:
    group_id = parse_group_id(model_name)
    if group_id == "01":
        match = re.search(r"weight_init_seed(\d+)", model_name)
        return int(match.group(1)) if match else 10_000
    if group_id in {"02", "03"}:
        match = re.search(r"dataset_seed(\d+)", model_name)
        return int(match.group(1)) if match else 10_000
    if group_id == "04":
        match = re.search(r"bitflip_seed(\d+)", model_name)
        return int(match.group(1)) if match else 10_000
    return 10_000


def group_model_label(model_name: str) -> str:
    sort_value = parse_group_sort_value(model_name)
    return str(sort_value) if sort_value < 10_000 else model_name


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
                if not (run_dir / "kill_step.log").exists() or not (run_dir / "hits.log").exists():
                    continue
                kill_step = read_kill_step(run_dir)
                rows.append(
                    {
                        "pass_num": pass_num,
                        "logs_setting": logs_setting,
                        "prompt_slug": prompt_slug,
                        "model_name": model_dir.name,
                        "model_seed": parse_model_seed(model_dir.name),
                        "group_id": parse_group_id(model_dir.name),
                        "group_label": GROUP_LABELS.get(parse_group_id(model_dir.name), parse_group_id(model_dir.name)),
                        "group_sort_value": parse_group_sort_value(model_dir.name),
                        "group_model_label": group_model_label(model_dir.name),
                        "run_dir": str(run_dir),
                        "kill_step": kill_step,
                        "killed": kill_step != -1,
                        "hit_sum": read_hit_sum(run_dir),
                    }
                )
    return pd.DataFrame(rows)


def aggregate_records(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    grouped = df.groupby(
        [
            "group_id",
            "group_label",
            "group_sort_value",
            "group_model_label",
            "model_name",
            "model_seed",
            "logs_setting",
            "prompt_slug",
        ],
        sort=True,
    )
    for keys, group in grouped:
        (
            group_id,
            group_label,
            group_sort_value,
            group_model_label,
            model_name,
            model_seed,
            logs_setting,
            prompt_slug,
        ) = keys
        n = int(len(group))
        kill_successes = int(group["killed"].sum())
        kill_ci_low, kill_ci_high = wilson_ci(kill_successes, n)
        kill_raw_values_pct = [100.0 if bool(value) else 0.0 for value in group["killed"].tolist()]
        hit_raw_values = group["hit_sum"].astype(float).tolist()
        hit_mean, hit_ci95 = mean_ci95(hit_raw_values)
        rows.append(
            {
                "group_id": group_id,
                "group_label": group_label,
                "group_sort_value": int(group_sort_value),
                "group_model_label": group_model_label,
                "model_name": model_name,
                "model_seed": int(model_seed),
                "logs_setting": logs_setting,
                "prompt_slug": prompt_slug,
                "setting_label": SETTING_LABELS[(logs_setting, prompt_slug)],
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
    return pd.DataFrame(rows).sort_values(
        ["group_id", "group_sort_value", "logs_setting", "prompt_slug", "model_name"]
    )


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
    setting: tuple[str, str],
    jitter_seed: int,
) -> None:
    if subset.empty:
        ax.set_visible(False)
        return
    subset = subset.sort_values(["group_sort_value", "model_name"]).reset_index(drop=True)
    x = np.arange(len(subset))
    means = subset[metric_key].to_numpy(dtype=float)
    lowers = subset[low_key].to_numpy(dtype=float)
    uppers = subset[high_key].to_numpy(dtype=float)
    yerr = np.vstack([np.maximum(0.0, means - lowers), np.maximum(0.0, uppers - means)])
    x_labels = subset["group_model_label"].astype(str).tolist()

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

    ax.set_title(SETTING_LABELS[setting], fontsize=12)
    ax.set_xlabel("Model index within group")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=80, ha="right", fontsize=7)
    ax.grid(axis="y", alpha=0.3)
    if metric_key == "kill_rate_pct":
        ax.set_ylim(-2, 102)


def make_group_combined_plot(
    group_df: pd.DataFrame,
    metric_key: str,
    output_path: Path,
) -> None:
    spec = metric_spec(metric_key)
    group_label = str(group_df["group_label"].iloc[0])
    fig, axes = plt.subplots(2, 2, figsize=(26, 12), sharex=False)
    axes_flat = axes.flatten()

    for panel_idx, setting in enumerate(SETTING_ORDER):
        logs_setting, prompt_slug = setting
        subset = group_df[
            (group_df["logs_setting"] == logs_setting) & (group_df["prompt_slug"] == prompt_slug)
        ].copy()
        ax = axes_flat[panel_idx]
        plot_setting_with_raw_points(
            ax=ax,
            subset=subset,
            metric_key=metric_key,
            low_key=spec["low_key"],
            high_key=spec["high_key"],
            raw_values_key=spec["raw_values_key"],
            setting=setting,
            jitter_seed=(13_000 if metric_key == "kill_rate_pct" else 17_000) + panel_idx,
        )
        ax.set_ylabel(spec["y_label"])

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
    fig.suptitle(f"{group_label} | {spec['title']}", fontsize=16)
    fig.legend(handles=[point_legend, mean_legend], loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def make_group_split_plots(
    group_df: pd.DataFrame,
    metric_key: str,
    output_dir: Path,
) -> None:
    spec = metric_spec(metric_key)
    group_id = str(group_df["group_id"].iloc[0])
    group_label = str(group_df["group_label"].iloc[0])
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
    for panel_idx, setting in enumerate(SETTING_ORDER):
        logs_setting, prompt_slug = setting
        subset = group_df[
            (group_df["logs_setting"] == logs_setting) & (group_df["prompt_slug"] == prompt_slug)
        ].copy()
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
            setting=setting,
            jitter_seed=(21_000 if metric_key == "kill_rate_pct" else 31_000) + panel_idx,
        )
        ax.set_ylabel(spec["y_label"])
        ax.legend(handles=[point_legend, mean_legend], loc="upper right", frameon=False)
        fig.suptitle(f"{group_label} | {SETTING_LABELS[setting]} | {spec['title']}", fontsize=14)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(output_dir / f"{metric_key}_{logs_setting}_{prompt_slug}.pdf", dpi=220)
        plt.close(fig)


def _build_group_setting_panel(
    ax: plt.Axes,
    subset: pd.DataFrame,
    metric_key: str,
    spec: dict[str, str],
    setting: tuple[str, str],
    jitter_seed: int,
) -> None:
    if subset.empty:
        ax.set_visible(False)
        return
    plot_setting_with_raw_points(
        ax,
        subset=subset,
        metric_key=metric_key,
        low_key=spec["low_key"],
        high_key=spec["high_key"],
        raw_values_key=spec["raw_values_key"],
        setting=setting,
        jitter_seed=jitter_seed,
    )


def make_group_setting_stack(
    group_df: pd.DataFrame,
    metric_key: str,
    output_dir: Path,
    *,
    orientation: str = "vertical",
) -> None:
    spec = metric_spec(metric_key)
    if orientation == "horizontal":
        fig, axes = plt.subplots(1, 4, figsize=(18.0, 4.5), sharey=True)
        seed_base = 51_000
    else:
        fig, axes = plt.subplots(4, 1, figsize=(7.6, 14.5), sharex=True)
        seed_base = 61_000
    axes_flat = np.array(axes).reshape(-1)
    for panel_idx, setting in enumerate(SETTING_ORDER):
        logs_setting, prompt_slug = setting
        subset = group_df[
            (group_df["logs_setting"] == logs_setting) & (group_df["prompt_slug"] == prompt_slug)
        ].copy()
        _build_group_setting_panel(
            axes_flat[panel_idx],
            subset,
            metric_key,
            spec,
            setting,
            seed_base + panel_idx,
        )
        axes_flat[panel_idx].set_title(SETTING_LABELS[setting], fontsize=9.2)
        if orientation == "horizontal":
            axes_flat[panel_idx].set_xlabel("Model index within group")
            if panel_idx == 0:
                axes_flat[panel_idx].set_ylabel(spec["y_label"])
        else:
            axes_flat[panel_idx].set_xlabel("Model index within group")
            axes_flat[panel_idx].set_ylabel(spec["y_label"] if panel_idx == 0 else "")
    for axis in axes_flat[len(SETTING_ORDER) :]:
        axis.set_visible(False)
    fig.tight_layout(h_pad=1.0, w_pad=0.9)
    stem = "settings_stacked_vertical" if orientation != "horizontal" else "settings_stacked_side_by_side"
    fig.savefig(output_dir / f"{metric_key}_{stem}.pdf", dpi=220)
    plt.close(fig)


def make_group_folder_outputs(
    summary_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    for group_id in ["01", "02", "03", "04"]:
        group_df = summary_df[summary_df["group_id"] == group_id].copy()
        if group_df.empty:
            continue
        group_output_dir = output_dir / f"group_{group_id}"
        group_output_dir.mkdir(parents=True, exist_ok=True)
        make_group_combined_plot(
            group_df=group_df,
            metric_key="kill_rate_pct",
            output_path=group_output_dir / "kill_rate_combined.pdf",
        )
        make_group_combined_plot(
            group_df=group_df,
            metric_key="hit_mean",
            output_path=group_output_dir / "hit_mean_combined.pdf",
        )
        make_group_split_plots(group_df=group_df, metric_key="kill_rate_pct", output_dir=group_output_dir)
        make_group_split_plots(group_df=group_df, metric_key="hit_mean", output_dir=group_output_dir)

        make_group_setting_stack(group_df=group_df, metric_key="kill_rate_pct", output_dir=group_output_dir, orientation="vertical")
        make_group_setting_stack(group_df=group_df, metric_key="kill_rate_pct", output_dir=group_output_dir, orientation="horizontal")
        make_group_setting_stack(group_df=group_df, metric_key="hit_mean", output_dir=group_output_dir, orientation="vertical")
        make_group_setting_stack(group_df=group_df, metric_key="hit_mean", output_dir=group_output_dir, orientation="horizontal")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    per_pass_df = collect_run_records(args.eval_root.resolve(), args.pass_start, args.pass_end)
    if per_pass_df.empty:
        raise RuntimeError("No villager chop runs found.")
    summary_df = aggregate_records(per_pass_df)

    per_pass_df.to_csv(output_dir / "villager_chop_40_models_per_pass.csv", index=False)
    summary_df.to_csv(output_dir / "villager_chop_40_models_summary.csv", index=False)

    counts_df = (
        summary_df[["group_id", "group_label", "group_model_label", "episodes_found"]]
        .drop_duplicates()
        .sort_values(["group_id", "group_model_label"])
    )
    counts_df.to_csv(output_dir / "villager_chop_40_models_episode_counts.csv", index=False)
    make_group_folder_outputs(summary_df=summary_df, output_dir=output_dir)

    model_counts = (
        summary_df[["group_id", "group_model_label"]]
        .drop_duplicates()
        .groupby("group_id", as_index=False)
        .size()
        .rename(columns={"size": "n_models"})
        .sort_values("group_id")
    )
    print(f"Per-pass rows: {len(per_pass_df)}")
    print(f"Summary rows: {len(summary_df)}")
    print("Models per group:")
    print(model_counts.to_string(index=False))
    print(f"Wrote outputs to: {output_dir}")


if __name__ == "__main__":
    main()
