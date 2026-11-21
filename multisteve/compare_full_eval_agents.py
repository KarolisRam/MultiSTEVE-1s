#!/usr/bin/env python3
"""Compare two agents on full-eval logs and create per-prompt average plots."""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from multisteve.plot_suite.style import PALETTE, SIDE_BY_SIDE_FIGURE_WIDTH_IN, save_figure

try:
    from scipy.stats import t as student_t
except Exception:
    student_t = None

RUN_DIR_SEED_RE = re.compile(r"^seed_(\d+)_")

METRIC_SPECS = [
    ("max_distance", "Maximum Distance Traveled", "Distance (blocks)"),
    ("dirt", "Dirt Blocks Collected", "Average max count"),
    ("logs", "Logs Collected", "Average max count"),
    ("seeds", "Seeds Collected", "Average max count"),
]


def parse_args() -> argparse.Namespace:
    default_eval_root = Path("final-evals/01_full_eval")
    default_output_dir = Path(__file__).resolve().parent / "plots" / "01_full_eval_agent_compare"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval-root",
        type=Path,
        default=default_eval_root,
        help=f"Root directory with agent folders (default: {default_eval_root})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir,
        help=f"Directory where plots/CSVs will be written (default: {default_output_dir})",
    )
    parser.add_argument(
        "--expected-seeds",
        type=int,
        default=20,
        help="Expected number of seeds per (agent, modality, prompt); warning only.",
    )
    return parser.parse_args()


def parse_seed(run_dir: Path) -> int | None:
    match = RUN_DIR_SEED_RE.match(run_dir.name)
    if match:
        return int(match.group(1))

    seed_log = run_dir / "seed.log"
    if not seed_log.exists():
        return None
    text = seed_log.read_text(encoding="utf-8", errors="ignore")
    numbers = re.findall(r"-?\d+", text)
    return int(numbers[0]) if numbers else None


def compute_max_distance(location_path: Path) -> float:
    if not location_path.exists():
        return float("nan")

    with location_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        first_row = next(reader, None)
        if first_row is None:
            return 0.0

        start_x = float(first_row["xpos"])
        start_z = float(first_row["zpos"])
        max_dist = 0.0

        first_dist = math.hypot(float(first_row["xpos"]) - start_x, float(first_row["zpos"]) - start_z)
        max_dist = max(max_dist, first_dist)

        for row in reader:
            x = float(row["xpos"])
            z = float(row["zpos"])
            max_dist = max(max_dist, math.hypot(x - start_x, z - start_z))

    return max_dist


def compute_inventory_metrics(inventory_path: Path) -> dict[str, float]:
    if not inventory_path.exists():
        return {"dirt": float("nan"), "logs": float("nan"), "seeds": float("nan")}

    inventory: dict[str, int] = {}
    max_dirt = 0
    max_logs = 0
    max_seeds = 0

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
            current_dirt = sum(v for k, v in inventory.items() if "dirt" in k)
            current_logs = sum(v for k, v in inventory.items() if "_log" in k)
            current_seeds = sum(v for k, v in inventory.items() if "seed" in k)
            max_dirt = max(max_dirt, current_dirt)
            max_logs = max(max_logs, current_logs)
            max_seeds = max(max_seeds, current_seeds)

    return {"dirt": float(max_dirt), "logs": float(max_logs), "seeds": float(max_seeds)}


def collect_latest_runs(eval_root: Path) -> dict[tuple[str, str, str, int], Path]:
    latest_runs: dict[tuple[str, str, str, int], Path] = {}
    for agent_dir in sorted(p for p in eval_root.iterdir() if p.is_dir()):
        agent = agent_dir.name
        for modality in ("text", "visual"):
            modality_dir = agent_dir / modality
            if not modality_dir.exists():
                continue
            for prompt_dir in sorted(p for p in modality_dir.iterdir() if p.is_dir()):
                prompt = prompt_dir.name
                for run_dir in sorted(p for p in prompt_dir.iterdir() if p.is_dir()):
                    seed = parse_seed(run_dir)
                    if seed is None:
                        continue
                    key = (agent, modality, prompt, seed)
                    prev = latest_runs.get(key)
                    if prev is None or run_dir.name > prev.name:
                        latest_runs[key] = run_dir
    return latest_runs


def build_per_run_df(latest_runs: dict[tuple[str, str, str, int], Path]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (agent, modality, prompt, seed), run_dir in sorted(latest_runs.items()):
        inv_metrics = compute_inventory_metrics(run_dir / "inventory_updates.log")
        row = {
            "agent": agent,
            "modality": modality,
            "prompt": prompt,
            "seed": seed,
            "max_distance": compute_max_distance(run_dir / "location_stats.log"),
            "dirt": inv_metrics["dirt"],
            "logs": inv_metrics["logs"],
            "seeds": inv_metrics["seeds"],
            "run_dir": str(run_dir),
        }
        rows.append(row)
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
    grouped = per_run_df.groupby(["agent", "modality", "prompt"], sort=True)

    for (agent, modality, prompt), group in grouped:
        row: dict[str, object] = {
            "agent": agent,
            "modality": modality,
            "prompt": prompt,
            "n_seeds": int(group["seed"].nunique()),
        }
        for metric_key, _, _ in METRIC_SPECS:
            mean, ci95 = mean_ci95(group[metric_key].to_numpy(dtype=float))
            row[metric_key] = mean
            row[f"{metric_key}_ci95"] = ci95
        rows.append(row)

    return pd.DataFrame(rows).sort_values(["modality", "prompt", "agent"])


def get_agent_order(df: pd.DataFrame) -> list[str]:
    unique_agents = df["agent"].drop_duplicates().tolist()
    preferred = [x for x in ("Original", "Ours") if x in unique_agents]
    remaining = sorted(x for x in unique_agents if x not in preferred)
    return preferred + remaining


def display_agent_label(agent: str) -> str:
    if agent == "Original":
        return "STEVE-1"
    if agent == "Ours":
        return "MultiSTEVE-1s"
    return agent


def display_prompt_label(prompt: str) -> str:
    aliases = {
        "max_distance": "dist.",
    }
    if prompt.endswith("-engineered"):
        return f"{prompt.removesuffix('-engineered')}*"
    return aliases.get(prompt, prompt)


def plot_metric(
    agg_df: pd.DataFrame,
    modality: str,
    metric_key: str,
    metric_title: str,
    y_label: str,
    output_path: Path,
) -> None:
    subset = agg_df[agg_df["modality"] == modality]
    if subset.empty:
        return

    prompts = sorted(subset["prompt"].unique().tolist())
    agents = get_agent_order(subset)
    x = np.arange(len(prompts), dtype=float)
    width = 0.8 / max(len(agents), 1)
    agent_colors = {
        "Original": PALETTE["gray"],
        "Ours": PALETTE["blue"],
    }

    fig, ax = plt.subplots(figsize=(SIDE_BY_SIDE_FIGURE_WIDTH_IN, 2.7))

    for idx, agent in enumerate(agents):
        vals = []
        errs = []
        for prompt in prompts:
            rows = subset[(subset["agent"] == agent) & (subset["prompt"] == prompt)]
            if rows.empty:
                vals.append(np.nan)
                errs.append(0.0)
                continue
            vals.append(float(rows.iloc[0][metric_key]))
            errs.append(float(rows.iloc[0][f"{metric_key}_ci95"]))

        bar_x = x - 0.4 + width * (idx + 0.5)
        y = np.asarray(vals, dtype=float)
        ci = np.asarray(errs, dtype=float)
        low = np.maximum(0.0, y - ci)
        high = y + ci
        yerr = np.vstack([y - low, high - y])
        valid = np.isfinite(y)
        if np.any(valid):
            ax.bar(
                bar_x[valid],
                y[valid],
                width=width * 0.82,
                yerr=yerr[:, valid],
                capsize=1.3,
                color=agent_colors.get(agent, PALETTE["cyan"]),
                edgecolor="black",
                linewidth=0.22,
                error_kw={"elinewidth": 0.45, "capthick": 0.45},
                label=display_agent_label(agent),
            )

    short_title = {
        "Maximum Distance Traveled": "Max Distance",
        "Dirt Blocks Collected": "Dirt Collected",
        "Logs Collected": "Logs Collected",
        "Seeds Collected": "Seeds Collected",
    }.get(metric_title, metric_title)
    ax.set_title(f"{short_title} | {modality.title()} Prompt", pad=3)
    ax.set_ylabel(y_label)
    ax.set_xlabel("")
    ax.set_xticks(x)
    ax.set_xticklabels([display_prompt_label(prompt) for prompt in prompts], rotation=72, ha="right", fontsize=7.8)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title=None, handlelength=1.0, handletextpad=0.35, borderaxespad=0.2, loc="best")
    fig.tight_layout(pad=0.08)
    save_figure(fig, output_path, bbox_inches=None)


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
        raise RuntimeError(f"No runs found under: {eval_root}")

    agg_df = build_agg_df(per_run_df)

    per_run_path = output_dir / "per_run_metrics.csv"
    agg_path = output_dir / "aggregated_prompt_means.csv"
    per_run_df.to_csv(per_run_path, index=False)
    agg_df.to_csv(agg_path, index=False)

    for modality in ("text", "visual"):
        for metric_key, metric_title, y_label in METRIC_SPECS:
            out_name = f"{modality}_{metric_key}.png"
            plot_metric(
                agg_df=agg_df,
                modality=modality,
                metric_key=metric_key,
                metric_title=metric_title,
                y_label=y_label,
                output_path=output_dir / out_name,
            )

    seed_counts = agg_df[["agent", "modality", "prompt", "n_seeds"]].copy()
    expected = args.expected_seeds
    incomplete = seed_counts[seed_counts["n_seeds"] != expected]
    if incomplete.empty:
        print(f"All prompt groups have {expected} seeds.")
    else:
        print(f"Warning: {len(incomplete)} prompt groups differ from expected {expected} seeds.")
        print(incomplete.to_string(index=False))

    print(f"Wrote per-run metrics: {per_run_path}")
    print(f"Wrote aggregated metrics: {agg_path}")
    print(f"Wrote plots to: {output_dir}")


if __name__ == "__main__":
    main()
