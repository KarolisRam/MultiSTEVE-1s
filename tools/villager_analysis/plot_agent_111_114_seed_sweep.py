#!/usr/bin/env python3
"""Plot per-seed villager sweep results for agents 111 and 114.

Writes one kill-rate plot and one hit-mean plot per seed. Each plot shows the
groups among:
- left / agent 111
- left / agent 114
- right / agent 111
- right / agent 114
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from PIL import Image

ROOT = Path("data/evals-villager")
SWEEP_ROOT = ROOT / "agent_111_114_short_logs_above_true_seed_sweep"
OUT_DIR = ROOT / "analysis_agent_111_114_seed_sweep"
TARGET_SIZE = (1710, 972)
Z_95 = 1.959963984540054
GROUP_ORDER = [
    ("01_left", "agent_111"),
    ("01_left", "agent_114"),
    ("02_right", "agent_111"),
    ("02_right", "agent_114"),
]
SIDE_LABELS = {
    "01_left": "Villager Tree on the Left",
    "02_right": "Villager Tree on the Right",
}
AGENT_LABELS = {
    "agent_111": "Agent 111",
    "agent_114": "Agent 114",
}


@dataclass(frozen=True)
class RolloutRecord:
    seed_label: str
    seed_value: int
    side_key: str
    agent_key: str
    rep_key: str
    kill_step: int
    killed: bool
    hit_sum: float
    run_dir: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-root", type=Path, default=SWEEP_ROOT)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    return parser.parse_args()


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


def parse_seed_value(seed_label: str) -> int:
    return int(seed_label.split("_")[-1])


def collect_records(sweep_root: Path) -> list[RolloutRecord]:
    records: list[RolloutRecord] = []
    for kill_path in sorted(sweep_root.glob("seed_*/0*_*/agent_*/rep_*/dataset_*/text/wood/seed_*/kill_step.log")):
        run_dir = kill_path.parent
        hits_path = run_dir / "hits.log"
        if not hits_path.exists():
            continue
        rel = kill_path.relative_to(sweep_root)
        seed_label, side_key, agent_key, rep_key = rel.parts[:4]
        kill_step = int(kill_path.read_text(encoding="utf-8").strip())
        records.append(
            RolloutRecord(
                seed_label=seed_label,
                seed_value=parse_seed_value(seed_label),
                side_key=side_key,
                agent_key=agent_key,
                rep_key=rep_key,
                kill_step=kill_step,
                killed=(kill_step != -1),
                hit_sum=read_hit_sum(run_dir),
                run_dir=run_dir,
            )
        )
    return records


def aggregate(records: list[RolloutRecord]) -> dict[str, list[dict[str, object]]]:
    by_seed: dict[str, dict[tuple[str, str], list[RolloutRecord]]] = {}
    for record in records:
        seed_entry = by_seed.setdefault(record.seed_label, {})
        seed_entry.setdefault((record.side_key, record.agent_key), []).append(record)

    aggregated: dict[str, list[dict[str, object]]] = {}
    for seed_label, groups in sorted(by_seed.items(), key=lambda item: parse_seed_value(item[0])):
        rows: list[dict[str, object]] = []
        for side_key, agent_key in GROUP_ORDER:
            group_records = groups.get((side_key, agent_key), [])
            if not group_records:
                continue
            group_records = sorted(group_records, key=lambda r: r.rep_key)
            kill_successes = sum(1 for r in group_records if r.killed)
            kill_ci_low, kill_ci_high = wilson_ci(kill_successes, len(group_records))
            hit_values = [r.hit_sum for r in group_records]
            hit_mean, hit_ci95 = mean_ci95(hit_values)
            rows.append(
                {
                    "seed_label": seed_label,
                    "seed_value": parse_seed_value(seed_label),
                    "side_key": side_key,
                    "agent_key": agent_key,
                    "group_label": f"{AGENT_LABELS[agent_key]} | {SIDE_LABELS[side_key]}",
                    "x_label": f"{AGENT_LABELS[agent_key]}\n{SIDE_LABELS[side_key]}",
                    "episodes_found": len(group_records),
                    "kill_rate_pct": 100.0 * kill_successes / len(group_records),
                    "kill_ci_low_pct": 100.0 * kill_ci_low,
                    "kill_ci_high_pct": 100.0 * kill_ci_high,
                    "raw_kill_values": [100.0 if r.killed else 0.0 for r in group_records],
                    "hit_mean": hit_mean,
                    "hit_ci_low": hit_mean - hit_ci95,
                    "hit_ci_high": hit_mean + hit_ci95,
                    "raw_hit_values": hit_values,
                }
            )
        if rows:
            aggregated[seed_label] = rows
    return aggregated


def resize_plot(path: Path) -> None:
    with Image.open(path) as image:
        resized = image.resize(TARGET_SIZE, resample=Image.Resampling.LANCZOS)
        resized.save(path)


def plot_seed_metric(
    rows: list[dict[str, object]],
    seed_label: str,
    metric_key: str,
    lower_key: str,
    upper_key: str,
    raw_values_key: str,
    title_prefix: str,
    y_label: str,
    output_path: Path,
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
    x_labels = [str(row["x_label"]) for row in rows]
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

    ax.set_title(
        f"{title_prefix}\nSeed {parse_seed_value(seed_label):03d} | logs_above_villager=True | prompt: \"chop a tree\""
    )
    ax.set_xlabel("Agent / setting")
    ax.set_ylabel(y_label)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    if metric_key == "kill_rate_pct":
        ax.set_ylim(-2, 102)

    point_legend = Line2D(
        [0], [0], marker="o", linestyle="None", markerfacecolor="red", markeredgecolor="none",
        alpha=0.28, markersize=7, label="Per-rollout data point"
    )
    mean_legend = Line2D(
        [0], [0], marker="o", color="black", linestyle="None", markersize=4, label="Mean ± 95% CI"
    )
    ax.legend(handles=[point_legend, mean_legend], loc="upper right", frameon=False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    resize_plot(output_path)


def write_summary_csv(aggregated: dict[str, list[dict[str, object]]], out_path: Path) -> None:
    fieldnames = [
        "seed_label",
        "seed_value",
        "side_key",
        "agent_key",
        "episodes_found",
        "kill_rate_pct",
        "kill_ci_low_pct",
        "kill_ci_high_pct",
        "hit_mean",
        "hit_ci_low",
        "hit_ci_high",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for seed_label in sorted(aggregated.keys(), key=parse_seed_value):
            for row in aggregated[seed_label]:
                writer.writerow({key: row[key] for key in fieldnames})


def main() -> int:
    args = parse_args()
    sweep_root = args.sweep_root.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    records = collect_records(sweep_root)
    aggregated = aggregate(records)
    summary_csv = out_dir / "summary_by_seed.csv"
    write_summary_csv(aggregated, summary_csv)

    seeds_written: list[str] = []
    for seed_label in sorted(aggregated.keys(), key=parse_seed_value):
        rows = aggregated[seed_label]
        if not rows:
            continue
        seeds_written.append(seed_label)
        plot_seed_metric(
            rows=rows,
            seed_label=seed_label,
            metric_key="kill_rate_pct",
            lower_key="kill_ci_low_pct",
            upper_key="kill_ci_high_pct",
            raw_values_key="raw_kill_values",
            title_prefix="Villager Kill Rate",
            y_label="Kill rate (%)",
            output_path=out_dir / f"kill_rate_points_ci_{seed_label}.png",
            jitter_seed=13_000 + parse_seed_value(seed_label),
        )
        plot_seed_metric(
            rows=rows,
            seed_label=seed_label,
            metric_key="hit_mean",
            lower_key="hit_ci_low",
            upper_key="hit_ci_high",
            raw_values_key="raw_hit_values",
            title_prefix="Hits Average",
            y_label="Average hit sum per rollout",
            output_path=out_dir / f"hit_mean_points_ci_{seed_label}.png",
            jitter_seed=17_000 + parse_seed_value(seed_label),
        )

    print(f"Sweep root: {sweep_root}")
    print(f"Output dir: {out_dir}")
    print(f"Completed rollouts found: {len(records)}")
    print(f"Seeds with plots: {len(seeds_written)}")
    for seed_label in seeds_written:
        counts = ", ".join(
            f"{row['agent_key']}:{row['side_key']}={row['episodes_found']}"
            for row in aggregated[seed_label]
        )
        print(f"  {seed_label}: {counts}")
    print(f"Summary CSV: {summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
