from __future__ import annotations

import argparse
import json
import math
import os
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Rectangle

from .style import NEURIPS_TEXT_WIDTH_IN, PALETTE, apply_style, save_figure


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = REPO_ROOT.parent
DEFAULT_AGENT_TOTALS = WORKSPACE_ROOT / "evals-villager" / "analysis_agent_111_114_seed_sweep" / "combined_views" / "agent_totals.csv"
DEFAULT_PAPER_ZIP = Path(os.environ.get("MULTISTEVE_PAPER_SOURCE_ZIP", "paper_source.zip"))
DEFAULT_ACTION_SWEEP_ROOT = (
    WORKSPACE_ROOT
    / "action-probabilities"
    / "experiments"
    / "model_group_comparisons"
    / "fake_env_main_episode_05_setting_sweep"
    / "20260331-114939"
)
DEFAULT_ACTION_SUMMARY = DEFAULT_ACTION_SWEEP_ROOT / "summary" / "family_summary.csv"
DEFAULT_ACTION_COMPARISONS = DEFAULT_ACTION_SWEEP_ROOT / "comparisons"
NEURIPS_TEXT_HEIGHT_IN = 9.0
FIGURE1_HEIGHT_IN = NEURIPS_TEXT_HEIGHT_IN / 3.0
FIGURE1_QUARTER_HEIGHT_IN = NEURIPS_TEXT_HEIGHT_IN / 4.0
NEURIPS_MAIN_TEXT_PT = 10.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render Figure 1 concept PDFs for the MultiSTEVE-1s paper.")
    parser.add_argument("--agent-totals", type=Path, default=DEFAULT_AGENT_TOTALS)
    parser.add_argument("--action-summary", type=Path, default=DEFAULT_ACTION_SUMMARY)
    parser.add_argument("--action-comparisons-dir", type=Path, default=DEFAULT_ACTION_COMPARISONS)
    parser.add_argument("--paper-zip", type=Path, default=DEFAULT_PAPER_ZIP)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "plot_suite" / "figure1" / f"figure1_versions_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    return parser.parse_args()


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = successes / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = z / denom * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))
    return center - half, center + half


def load_agent_rates(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    out = df.copy()
    out["kill_rate_pct"] = 100.0 * out["kills"].astype(float) / out["rollouts"].astype(float)
    lows = []
    highs = []
    for row in out.itertuples(index=False):
        lo, hi = wilson_interval(int(row.kills), int(row.rollouts))
        lows.append(100.0 * lo)
        highs.append(100.0 * hi)
    out["ci_low_pct"] = lows
    out["ci_high_pct"] = highs
    out["agent_num"] = out["agent_key"].str.extract(r"(\d+)").astype(int)
    out = out.sort_values("agent_num")
    return out


def load_lsb_randomness_summary(summary_path: Path, comparisons_dir: Path) -> dict[str, dict[str, object]]:
    df = pd.read_csv(summary_path)
    specs = {
        "full": ("03_init_and_data", "Full randomness", "init + data"),
        "lsb": ("04_single_lsb_flip", "Single LSB flip", "one-bit change"),
    }
    out: dict[str, dict[str, object]] = {}
    for key, (family_name, label, short_label) in specs.items():
        row = df.loc[df["family_name"] == family_name]
        if row.empty:
            raise ValueError(f"Missing {family_name} in {summary_path}")
        record = row.iloc[0]
        summary_json = comparisons_dir / family_name / "summary.json"
        pairwise_exact_agree: list[float] = []
        if summary_json.exists():
            payload = json.loads(summary_json.read_text(encoding="utf-8"))
            pairwise_exact_agree = [100.0 * float(v) for v in payload["exact_full_action_agreement"]["pairwise_means"]]
        exact_agree_pct = 100.0 * float(record["exact_full_action_agreement"])
        out[key] = {
            "family_name": family_name,
            "label": label,
            "short_label": short_label,
            "n_models": int(record["num_models"]),
            "n_pairs": int(record["num_pairs"]),
            "exact_agree_pct": exact_agree_pct,
            "exact_disagree_pct": 100.0 - exact_agree_pct,
            "semantic_disagree_pct": 100.0 * (1.0 - float(record["semantic_button_group_agreement"])),
            "camera_disagree_pct": 100.0 * (1.0 - float(record["camera_action_agreement"])),
            "pairwise_exact_agree_pct": pairwise_exact_agree,
            "pairwise_exact_disagree_pct": [100.0 - v for v in pairwise_exact_agree],
        }
    return out


def setup_axis(ax: plt.Axes) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")


def save_exact_figure(fig: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"savefig.bbox": None, "savefig.pad_inches": 0.0}):
        fig.savefig(output_path, bbox_inches=None, pad_inches=0.0)
    plt.close(fig)


def add_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    wh: tuple[float, float],
    title: str,
    body: str = "",
    *,
    fc: str = "white",
    ec: str = "#4B5563",
    title_color: str = PALETTE["black"],
    body_color: str = "#475569",
    lw: float = 0.9,
    title_size: float = 8.0,
    body_size: float = 6.7,
    body_weight: str = "normal",
    rounding: float = 0.035,
) -> None:
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.012,rounding_size={rounding}",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h * 0.68, title, ha="center", va="center", fontsize=title_size, color=title_color, weight="semibold")
    if body:
        ax.text(
            x + w / 2,
            y + h * 0.32,
            body,
            ha="center",
            va="center",
            fontsize=body_size,
            color=body_color,
            weight=body_weight,
            linespacing=1.05,
        )


def add_arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = "#64748B",
    lw: float = 1.0,
    connectionstyle: str | None = None,
) -> None:
    kwargs = {"connectionstyle": connectionstyle} if connectionstyle else {}
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=8, linewidth=lw, color=color, zorder=1.5, **kwargs))


def label_panel(ax: plt.Axes, label: str, title: str) -> None:
    ax.text(0.0, 1.04, label, transform=ax.transAxes, ha="left", va="bottom", fontsize=9.5, weight="bold", color=PALETTE["black"])
    ax.text(0.095, 1.04, title, transform=ax.transAxes, ha="left", va="bottom", fontsize=9.0, weight="semibold", color=PALETTE["black"])


def add_agent_bars(ax: plt.Axes, rates: pd.DataFrame, *, title: str = "Focused villager-tree evaluation", compact: bool = False) -> None:
    colors = [PALETTE["blue"], PALETTE["red"]]
    x = np.arange(len(rates))
    y = rates["kill_rate_pct"].to_numpy()
    yerr = np.vstack([y - rates["ci_low_pct"].to_numpy(), rates["ci_high_pct"].to_numpy() - y])
    bars = ax.bar(x, y, width=0.58, color=colors, edgecolor="white", linewidth=0.7, zorder=3)
    ax.errorbar(x, y, yerr=yerr, fmt="none", ecolor="#334155", elinewidth=0.9, capsize=2.2, zorder=4)
    for bar, row in zip(bars, rates.itertuples(index=False)):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 5.0,
            f"{int(row.kills)}/{int(row.rollouts)}\n{row.kill_rate_pct:.1f}%",
            ha="center",
            va="bottom",
            fontsize=7.2 if compact else 8.0,
            weight="semibold",
            color=PALETTE["black"],
            linespacing=0.95,
        )
    ax.set_xticks(x, [label.replace(" ", "\n") for label in rates["agent_label"]])
    ax.set_ylim(0, 60)
    if compact:
        ax.set_ylabel("")
        ax.text(0.03, 0.97, "Kill rate (%)", transform=ax.transAxes, ha="left", va="top", fontsize=6.2, color="#475569")
    else:
        ax.set_ylabel("Villager kill rate (%)")
    ax.set_title(title, pad=5)
    ax.grid(axis="y", alpha=0.35, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(
        0.98,
        0.04,
        "n=2,000 each\n95% Wilson CI",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.6 if compact else 6.2,
        color="#475569",
        linespacing=0.95,
    )


def add_agent_bubbles(ax: plt.Axes, rates: pd.DataFrame) -> None:
    setup_axis(ax)
    label_panel(ax, "B", "Two outliers after screening")
    positions = [(0.33, 0.52), (0.74, 0.52)]
    colors = [PALETTE["blue"], PALETTE["red"]]
    max_rate = rates["kill_rate_pct"].max()
    for (x, y), color, row in zip(positions, colors, rates.itertuples(index=False)):
        radius = 0.09 + 0.13 * math.sqrt(row.kill_rate_pct / max_rate)
        ax.add_patch(Circle((x, y), radius, facecolor=color, edgecolor="white", linewidth=1.2, alpha=0.92))
        ax.text(x, y + 0.025, f"{row.kill_rate_pct:.1f}%", ha="center", va="center", fontsize=14, weight="bold", color="white")
        ax.text(x, y - 0.065, f"{int(row.kills)}/{int(row.rollouts)}", ha="center", va="center", fontsize=7.3, color="white")
        ax.text(x, 0.19, row.agent_label, ha="center", va="center", fontsize=9.0, weight="semibold", color=PALETTE["black"])
    ax.text(0.5, 0.87, "Villagers killed in focused evaluation", ha="center", va="center", fontsize=8.7, weight="semibold")
    ax.text(0.5, 0.075, "Agent 114 kills about 13x as often as Agent 111", ha="center", va="center", fontsize=7.2, color="#475569")


def draw_variant_pipeline_bars(rates: pd.DataFrame, out_path: Path) -> None:
    fig = plt.figure(figsize=(NEURIPS_TEXT_WIDTH_IN, 3.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.08, 0.92], wspace=0.22)
    left = fig.add_subplot(gs[0, 0])
    right = fig.add_subplot(gs[0, 1])
    setup_axis(left)
    label_panel(left, "A", "Model-zoo construction")
    add_box(left, (0.04, 0.67), (0.24, 0.18), "VPT", "Minecraft\npolicy", fc="#EFF6FF", ec=PALETTE["blue"])
    add_box(left, (0.38, 0.67), (0.28, 0.18), "Fine-tune", "fixed STEVE-1\nrecipe", fc="#F8FAFC")
    add_box(left, (0.76, 0.67), (0.2, 0.18), "140", "agents", fc="#F0FDF4", ec=PALETTE["green"], title_size=10.0)
    add_arrow(left, (0.29, 0.76), (0.37, 0.76))
    add_arrow(left, (0.67, 0.76), (0.75, 0.76))

    group_rows = [
        ("01", "init seed", "10 x 28 ckpts"),
        ("02", "data seed", "10 x 28 ckpts"),
        ("03", "init + data", "10 x 28 ckpts"),
        ("04", "single bit flip", "10 x 28 ckpts"),
        ("05", "init + data", "100 final agents"),
    ]
    y0 = 0.48
    for i, (gid, source, count) in enumerate(group_rows):
        y = y0 - i * 0.083
        fc = "#FFF7ED" if gid == "05" else "#F8FAFC"
        ec = PALETTE["orange"] if gid == "05" else "#CBD5E1"
        left.add_patch(FancyBboxPatch((0.05, y), 0.9, 0.052, boxstyle="round,pad=0.008,rounding_size=0.015", facecolor=fc, edgecolor=ec, linewidth=0.7))
        left.text(0.075, y + 0.026, gid, ha="left", va="center", fontsize=7.2, weight="bold", color=PALETTE["black"])
        left.text(0.20, y + 0.026, source, ha="left", va="center", fontsize=7.0, color=PALETTE["black"])
        left.text(0.90, y + 0.026, count, ha="right", va="center", fontsize=6.8, color="#475569")
    left.text(0.5, 0.05, "100-agent group supports outlier search", ha="center", va="center", fontsize=7.0, color=PALETTE["orange"], weight="semibold")

    add_agent_bars(right, rates, title="OOD villager kills")
    right.text(-0.16, 1.04, "B", transform=right.transAxes, ha="left", va="bottom", fontsize=9.5, weight="bold", color=PALETTE["black"])
    save_figure(fig, out_path)


def draw_variant_zoo_bubbles(rates: pd.DataFrame, out_path: Path) -> None:
    fig = plt.figure(figsize=(NEURIPS_TEXT_WIDTH_IN, 3.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.16)
    left = fig.add_subplot(gs[0, 0])
    right = fig.add_subplot(gs[0, 1])
    setup_axis(left)
    label_panel(left, "A", "Five randomness settings")
    add_box(left, (0.20, 0.78), (0.60, 0.14), "Same STEVE-1 recipe", "fixed hyperparameters", fc="#F8FAFC")
    add_arrow(left, (0.50, 0.78), (0.50, 0.70))
    branch_xs = [0.12, 0.31, 0.50, 0.69, 0.88]
    labels = ["init", "data", "init+\ndata", "bit\nflip", "100\nfinal"]
    counts = ["10", "10", "10", "10", "100"]
    for x, label, count in zip(branch_xs, labels, counts):
        add_arrow(left, (0.50, 0.70), (x, 0.58), lw=0.8)
        color = PALETTE["orange"] if count == "100" else PALETTE["gray"]
        left.add_patch(Circle((x, 0.48), 0.078, facecolor="#FFF7ED" if count == "100" else "#EFF6FF", edgecolor=color, linewidth=0.9))
        left.text(x, 0.495, count, ha="center", va="center", fontsize=10, weight="bold", color=color)
        left.text(x, 0.415, label, ha="center", va="top", fontsize=6.4, color=PALETTE["black"], linespacing=0.95)
    left.text(0.5, 0.235, "40 agents with 28 checkpoints each", ha="center", va="center", fontsize=7.2, color="#475569")
    left.text(0.5, 0.135, "+ 100 final agents for behavioural outlier search", ha="center", va="center", fontsize=7.2, color=PALETTE["orange"], weight="semibold")
    left.text(0.5, 0.055, "140 agents total", ha="center", va="center", fontsize=10.0, color=PALETTE["black"], weight="bold")
    add_agent_bubbles(right, rates)
    save_figure(fig, out_path)


def draw_variant_grid_focus(rates: pd.DataFrame, out_path: Path) -> None:
    fig = plt.figure(figsize=(NEURIPS_TEXT_WIDTH_IN, 3.05))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 0.95], wspace=0.18)
    left = fig.add_subplot(gs[0, 0])
    right = fig.add_subplot(gs[0, 1])
    setup_axis(left)
    label_panel(left, "A", "140 comparable agents")
    left.add_patch(Rectangle((0.07, 0.50), 0.32, 0.28, facecolor="#EFF6FF", edgecolor=PALETTE["blue"], linewidth=0.8))
    left.add_patch(Rectangle((0.49, 0.50), 0.44, 0.28, facecolor="#FFF7ED", edgecolor=PALETTE["orange"], linewidth=0.8))
    left.text(0.23, 0.82, "40 agents", ha="center", va="bottom", fontsize=8.2, weight="semibold")
    left.text(0.71, 0.82, "100 final agents", ha="center", va="bottom", fontsize=8.2, weight="semibold")
    left.text(0.23, 0.46, "4 randomness settings\n28 checkpoints each", ha="center", va="top", fontsize=6.9, color="#475569")
    left.text(0.71, 0.46, "same init+data setting\nused to find outliers", ha="center", va="top", fontsize=6.9, color="#475569")
    for idx in range(40):
        row, col = divmod(idx, 10)
        left.add_patch(Circle((0.095 + col * 0.031, 0.745 - row * 0.055), 0.008, facecolor=PALETTE["blue"], edgecolor="none", alpha=0.72))
    for idx in range(100):
        row, col = divmod(idx, 20)
        model_id = 101 + idx
        color = PALETTE["red"] if model_id == 114 else PALETTE["blue"] if model_id == 111 else "#94A3B8"
        radius = 0.010 if model_id in (111, 114) else 0.006
        left.add_patch(Circle((0.515 + col * 0.020, 0.745 - row * 0.045), radius, facecolor=color, edgecolor="white", linewidth=0.25))
    left.add_patch(Circle((0.60, 0.30), 0.010, facecolor=PALETTE["blue"], edgecolor="white", linewidth=0.25))
    left.text(0.62, 0.30, "Agent 111", ha="left", va="center", fontsize=6.5, color=PALETTE["black"])
    left.add_patch(Circle((0.60, 0.24), 0.010, facecolor=PALETTE["red"], edgecolor="white", linewidth=0.25))
    left.text(0.62, 0.24, "Agent 114", ha="left", va="center", fontsize=6.5, color=PALETTE["black"])
    left.text(0.5, 0.11, "Controlled randomness creates many comparable agents", ha="center", va="center", fontsize=7.3, color=PALETTE["black"])

    add_agent_bars(right, rates, title="Focused comparison")
    right.text(-0.16, 1.04, "B", transform=right.transAxes, ha="left", va="bottom", fontsize=9.5, weight="bold", color=PALETTE["black"])
    save_figure(fig, out_path)


def add_model_grid(
    ax: plt.Axes,
    *,
    origin: tuple[float, float],
    columns: int,
    rows: int,
    count: int,
    dx: float,
    dy: float,
    base_color: str,
    highlight: dict[int, str] | None = None,
    start_model: int = 1,
    radius: float = 0.006,
) -> None:
    highlight = highlight or {}
    x0, y0 = origin
    for idx in range(count):
        row, col = divmod(idx, columns)
        model_id = start_model + idx
        color = highlight.get(model_id, base_color)
        size = radius * (1.55 if model_id in highlight else 1.0)
        ax.add_patch(
            Circle(
                (x0 + col * dx, y0 - row * dy),
                size,
                facecolor=color,
                edgecolor="white",
                linewidth=0.25 if model_id in highlight else 0.0,
                alpha=0.98 if model_id in highlight else 0.78,
            )
        )


def draw_synthesis_fullwidth(rates: pd.DataFrame, out_path: Path) -> None:
    fig = plt.figure(figsize=(NEURIPS_TEXT_WIDTH_IN, FIGURE1_HEIGHT_IN))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.65, 0.95], wspace=0.20, left=0.025, right=0.985, top=0.90, bottom=0.12)
    left = fig.add_subplot(gs[0, 0])
    right = fig.add_subplot(gs[0, 1])
    setup_axis(left)

    label_panel(left, "A", "Constructing 140 comparable STEVE-1 agents")
    box_y = 0.74
    add_box(left, (0.04, box_y), (0.20, 0.15), "VPT", "Minecraft\npolicy", fc="#EFF6FF", ec=PALETTE["blue"], title_size=7.8, body_size=6.1)
    add_box(left, (0.33, box_y), (0.23, 0.15), "Fine-tune", "fixed STEVE-1\nrecipe", fc="#F8FAFC", title_size=7.8, body_size=6.1)
    add_box(left, (0.66, box_y), (0.16, 0.15), "140", "agents", fc="#F0FDF4", ec=PALETTE["green"], title_size=9.5, body_size=6.3)
    add_arrow(left, (0.245, box_y + 0.075), (0.325, box_y + 0.075))
    add_arrow(left, (0.565, box_y + 0.075), (0.655, box_y + 0.075))

    group_rows = [
        "01_init_seed_only",
        "02_data_seed_only",
        "03_init_and_data",
        "04_single_lsb_flip",
        "05_final_init_and_data_100",
    ]
    left.text(0.03, 0.64, "Five model groups", ha="left", va="center", fontsize=7.4, weight="semibold", color=PALETTE["black"])
    for idx, name in enumerate(group_rows):
        y = 0.575 - idx * 0.052
        is_final = idx == 4
        fc = "#FFF7ED" if is_final else "#F8FAFC"
        ec = PALETTE["orange"] if is_final else "#CBD5E1"
        left.add_patch(FancyBboxPatch((0.03, y), 0.48, 0.038, boxstyle="round,pad=0.005,rounding_size=0.009", facecolor=fc, edgecolor=ec, linewidth=0.65))
        left.text(0.045, y + 0.019, name, ha="left", va="center", fontsize=6.0, color=PALETTE["black"], weight="semibold" if is_final else "normal")
    left.text(0.035, 0.285, "Groups 01-04: 40 agents with 28 checkpoints each", ha="left", va="center", fontsize=6.1, color="#475569")

    left.text(0.58, 0.64, "Agents as dots", ha="left", va="center", fontsize=7.4, weight="semibold", color=PALETTE["black"])
    left.add_patch(Rectangle((0.58, 0.41), 0.17, 0.15, facecolor="#EFF6FF", edgecolor=PALETTE["blue"], linewidth=0.75))
    left.text(0.665, 0.585, "Groups 01-04", ha="center", va="bottom", fontsize=6.4, weight="semibold")
    left.text(0.665, 0.385, "40 agents", ha="center", va="top", fontsize=6.0, color="#475569")
    add_model_grid(left, origin=(0.596, 0.535), columns=10, rows=4, count=40, dx=0.0145, dy=0.030, base_color=PALETTE["blue"], start_model=1, radius=0.0048)

    left.add_patch(Rectangle((0.79, 0.35), 0.20, 0.21, facecolor="#FFF7ED", edgecolor=PALETTE["orange"], linewidth=0.75))
    left.text(0.89, 0.585, "Group 05", ha="center", va="bottom", fontsize=6.4, weight="semibold")
    left.text(0.89, 0.325, "100 agents for\noutlier search", ha="center", va="top", fontsize=5.8, color=PALETTE["orange"], weight="semibold", linespacing=0.95)
    add_model_grid(
        left,
        origin=(0.810, 0.535),
        columns=10,
        rows=10,
        count=100,
        dx=0.017,
        dy=0.018,
        base_color="#94A3B8",
        highlight={111: PALETTE["blue"], 114: PALETTE["red"]},
        start_model=101,
        radius=0.0037,
    )
    left.add_patch(Circle((0.805, 0.245), 0.006, facecolor=PALETTE["blue"], edgecolor="white", linewidth=0.25))
    left.text(0.820, 0.245, "Agent 111", ha="left", va="center", fontsize=5.8, color=PALETTE["black"])
    left.add_patch(Circle((0.805, 0.195), 0.006, facecolor=PALETTE["red"], edgecolor="white", linewidth=0.25))
    left.text(0.820, 0.195, "Agent 114", ha="left", va="center", fontsize=5.8, color=PALETTE["black"])

    add_agent_bars(right, rates, title="Focused comparison", compact=True)
    right.text(-0.10, 1.04, "B", transform=right.transAxes, ha="left", va="bottom", fontsize=9.5, weight="bold", color=PALETTE["black"])
    right.set_xlabel("")
    save_exact_figure(fig, out_path)


def make_story_canvas() -> tuple[plt.Figure, plt.Axes]:
    fig = plt.figure(figsize=(NEURIPS_TEXT_WIDTH_IN, FIGURE1_HEIGHT_IN))
    ax = fig.add_axes([0, 0, 1, 1])
    setup_axis(ax)
    return fig, ax


def make_compact_canvas() -> tuple[plt.Figure, plt.Axes]:
    fig = plt.figure(figsize=(NEURIPS_TEXT_WIDTH_IN, FIGURE1_QUARTER_HEIGHT_IN))
    ax = fig.add_axes([0, 0, 1, 1])
    setup_axis(ax)
    return fig, ax


def pt(size: float) -> float:
    """Clamp figure text to NeurIPS normal text size."""
    return min(size, NEURIPS_MAIN_TEXT_PT)


def rxy(rect: tuple[float, float, float, float], x_frac: float, y_frac: float) -> tuple[float, float]:
    x, y, w, h = rect
    return x + w * x_frac, y + h * y_frac


def rrect(rect: tuple[float, float, float, float], x_frac: float, y_frac: float, w_frac: float, h_frac: float) -> tuple[float, float, float, float]:
    x, y, w, h = rect
    return x + w * x_frac, y + h * y_frac, w * w_frac, h * h_frac


def add_card(
    ax: plt.Axes,
    rect: tuple[float, float, float, float],
    title: str,
    *,
    fc: str,
    ec: str,
    title_color: str = PALETTE["black"],
    title_size: float = 7.4,
) -> None:
    x, y, w, h = rect
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.008,rounding_size=0.020",
            facecolor=fc,
            edgecolor=ec,
            linewidth=0.8,
        )
    )
    ax.text(x + 0.014, y + h - 0.030, title, ha="left", va="top", fontsize=pt(title_size), weight="semibold", color=title_color)


def draw_compact_barplot(
    ax: plt.Axes,
    *,
    rect: tuple[float, float, float, float],
    title: str,
    y_label: str,
    values: list[float],
    labels: list[str],
    colors: list[str],
    annotations: list[str],
    message: str = "",
    value_font: float = 5.4,
    axis_label: str | None = None,
    label_font: float = 4.7,
) -> None:
    x0, y0, w, h = rect
    if title:
        ax.text(x0 + w / 2, y0 + h + 0.022, title, ha="center", va="bottom", fontsize=pt(6.0), weight="semibold", color=PALETTE["black"])
    if axis_label:
        ax.text(x0 - 0.034, y0 + h / 2, axis_label, ha="center", va="center", rotation=90, fontsize=pt(4.8), color="#475569", weight="semibold")
    ax.text(x0 - 0.001, y0 + h + 0.006, y_label, ha="left", va="bottom", fontsize=pt(4.7), color="#475569")
    for tick in [0, 50, 100]:
        ty = y0 + h * tick / 100.0
        ax.plot([x0, x0 + w], [ty, ty], color="#E2E8F0", linewidth=0.55)
        ax.text(x0 - 0.004, ty, str(tick), ha="right", va="center", fontsize=pt(4.8), color="#475569")
    ax.plot([x0, x0], [y0, y0 + h], color=PALETTE["black"], linewidth=0.65)
    ax.plot([x0, x0 + w], [y0, y0], color=PALETTE["black"], linewidth=0.65)
    xs = [x0 + w * 0.32, x0 + w * 0.72]
    bar_w = w * 0.20
    for xpos, val, label, color, annotation in zip(xs, values, labels, colors, annotations):
        bar_h = h * val / 100.0
        ax.add_patch(Rectangle((xpos - bar_w / 2, y0), bar_w, bar_h, facecolor=color, edgecolor="white", linewidth=0.55))
        ax.text(xpos, y0 + bar_h + 0.010, annotation, ha="center", va="bottom", fontsize=pt(value_font), weight="semibold", linespacing=0.88)
        ax.text(xpos, y0 - 0.024, label, ha="center", va="top", fontsize=pt(label_font), linespacing=0.88)
    if message:
        ax.text(x0 + w, y0 + h + 0.010, message, ha="right", va="bottom", fontsize=pt(6.2), color=PALETTE["orange"], weight="semibold")


def action_values(lsb_data: dict[str, dict[str, object]]) -> tuple[list[float], list[str], str]:
    vals = [lsb_mean(lsb_data, "full"), lsb_mean(lsb_data, "lsb")]
    annotations = [f"{vals[0]:.1f}%", f"{vals[1]:.1f}%"]
    message = f"{abs(vals[0] - vals[1]):.1f} pp apart"
    return vals, annotations, message


def villager_values(rates: pd.DataFrame) -> tuple[list[float], list[str], str]:
    vals = [float(v) for v in rates["kill_rate_pct"].to_list()]
    annotations = [f"{int(row.kills)}/{int(row.rollouts)}\n{row.kill_rate_pct:.1f}%" for row in rates.itertuples(index=False)]
    ratio = vals[1] / vals[0] if vals[0] else float("inf")
    return vals, annotations, f"{ratio:.0f}x apart"


def draw_compact_group05_grid(ax: plt.Axes, rect: tuple[float, float, float, float], *, label: bool = True) -> None:
    x, y, w, h = rect
    ax.add_patch(Rectangle((x, y), w, h, facecolor="#FFF7ED", edgecolor=PALETTE["orange"], linewidth=0.7))
    if label:
        ax.text(x + w / 2, y + h + 0.017, "Group 05", ha="center", va="bottom", fontsize=pt(5.5), weight="semibold")
        ax.text(x + w / 2, y + h + 0.002, "100-agent outlier search", ha="center", va="bottom", fontsize=pt(4.7), color=PALETTE["orange"], weight="semibold")
    origin = (x + 0.012, y + h - 0.016)
    dx = (w - 0.024) / 9
    dy = (h - 0.032) / 9
    add_model_grid(
        ax,
        origin=origin,
        columns=10,
        rows=10,
        count=100,
        dx=dx,
        dy=dy,
        base_color="#94A3B8",
        highlight={111: PALETTE["blue"], 114: PALETTE["red"]},
        start_model=101,
        radius=min(w, h) * 0.016,
    )
    selected_center = (origin[0] + 1.5 * dx, origin[1] - 1.0 * dy)
    ax.add_patch(
        Ellipse(
            selected_center,
            width=max(0.060, 4.7 * dx),
            height=max(0.036, 2.7 * dy),
            fill=False,
            edgecolor=PALETTE["orange"],
            linewidth=1.0,
        )
    )


def draw_pairwise_agreement_matrix(
    ax: plt.Axes,
    *,
    rect: tuple[float, float, float, float],
    values: list[float],
    label: str,
    color: str,
) -> None:
    x, y, w, h = rect
    ax.add_patch(Rectangle((x, y), w, h, facecolor="#FFFFFF", edgecolor=color, linewidth=0.75))
    ax.text(x + w / 2, y + h + 0.008, label, ha="center", va="bottom", fontsize=pt(4.65), color=PALETTE["black"], weight="semibold")
    inferred_n = int((1 + math.sqrt(1 + 8 * len(values))) / 2)
    n = inferred_n if inferred_n * (inferred_n - 1) // 2 == len(values) else 10
    grid_n = n - 1
    label_pad_x = w * 0.120
    label_pad_y = h * 0.135
    cell = min((w - label_pad_x) / grid_n, (h - label_pad_y) / grid_n)
    x0 = x + label_pad_x + (w - label_pad_x - grid_n * cell) / 2
    y0 = y + label_pad_y + (h - label_pad_y - grid_n * cell) / 2
    lo, hi = 35.0, 50.0
    vals = list(values)
    idx = 0
    for row in range(grid_n):
        second_seed = row + 2
        for col in range(grid_n):
            first_seed = col + 1
            cx = x0 + col * cell
            cy = y0 + (grid_n - 1 - row) * cell
            if first_seed >= second_seed:
                ax.add_patch(Rectangle((cx, cy), cell * 0.86, cell * 0.86, facecolor="#F8FAFC", edgecolor="none", alpha=0.35))
                continue
            if idx >= len(vals):
                continue
            val = vals[idx]
            idx += 1
            alpha = 0.22 + 0.70 * max(0.0, min(1.0, (val - lo) / (hi - lo)))
            ax.add_patch(Rectangle((cx, cy), cell * 0.90, cell * 0.90, facecolor=color, edgecolor="white", linewidth=0.08, alpha=alpha))
    for idx in range(grid_n):
        label_text = f"s{idx + 1}"
        ax.text(x0 + (idx + 0.45) * cell, y0 - cell * 0.36, label_text, ha="center", va="top", fontsize=pt(2.25), color="#64748B")
        left_label = f"s{idx + 2}"
        ax.text(x0 - cell * 0.25, y0 + (grid_n - idx - 0.55) * cell, left_label, ha="right", va="center", fontsize=pt(2.25), color="#64748B")
    if vals:
        ax.text(x + w / 2, y - 0.010, f"mean {np.mean(vals):.1f}%", ha="center", va="top", fontsize=pt(5.5), color="#475569")


def draw_model_zoo_compact(
    ax: plt.Axes,
    rect: tuple[float, float, float, float],
    *,
    variant: str,
) -> dict[str, tuple[float, float]]:
    x, y, w, h = rect
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.008,rounding_size=0.020",
            facecolor="#F8FAFC",
            edgecolor="#CBD5E1",
            linewidth=0.8,
        )
    )

    def box(
        name: str,
        body: str,
        fc: str,
        ec: str,
        title_size: float,
        body_size: float,
        body_weight: str,
        xy: tuple[float, float],
        wh: tuple[float, float],
    ) -> tuple[float, float, float, float]:
        add_box(ax, xy, wh, name, body, fc=fc, ec=ec, title_size=pt(title_size), body_size=pt(body_size), body_weight=body_weight, rounding=0.016)
        return xy[0], xy[1], wh[0], wh[1]

    def center(r: tuple[float, float, float, float]) -> tuple[float, float]:
        return r[0] + r[2] / 2, r[1] + r[3] / 2

    def right(r: tuple[float, float, float, float]) -> tuple[float, float]:
        return r[0] + r[2], r[1] + r[3] / 2

    def left(r: tuple[float, float, float, float]) -> tuple[float, float]:
        return r[0], r[1] + r[3] / 2

    def bottom(r: tuple[float, float, float, float]) -> tuple[float, float]:
        return r[0] + r[2] / 2, r[1]

    def top(r: tuple[float, float, float, float]) -> tuple[float, float]:
        return r[0] + r[2] / 2, r[1] + r[3]

    specs = {
        "vpt": ("VPT", "Minecraft policy", "#EFF6FF", PALETTE["blue"], 6.7, 5.65, "normal"),
        "fine": ("Fine-tune", "fixed STEVE-1 recipe\ndifferent randomness", "#FFFFFF", "#4B5563", 6.6, 5.65, "normal"),
        "zoo": ("140", "agents", "#F0FDF4", PALETTE["green"], 8.2, 8.2, "semibold"),
    }
    vpt_w = w * 0.360
    box_h = h * 0.148
    pipe_x = x + w * 0.050
    vpt = box(*specs["vpt"], (pipe_x, y + h * 0.625), (vpt_w, box_h))
    fine = box(*specs["fine"], (pipe_x, y + h * 0.435), (vpt_w, box_h))
    zoo = box(*specs["zoo"], (pipe_x, y + h * 0.245), (vpt_w, box_h))
    add_arrow(ax, bottom(vpt), top(fine), lw=0.75)
    add_arrow(ax, bottom(fine), top(zoo), lw=0.75)
    source = right(zoo)
    group_x = x + w * 0.560

    group_h = h * 0.116
    group_ys = [y + h * 0.880, y + h * 0.755, y + h * 0.630, y + h * 0.505, y + h * 0.200]
    group_rows = [
        ("01 init seed", "10"),
        ("02 data seed", "10"),
        ("03 init + data", "10"),
        ("04 single LSB", "10"),
        ("05 final init+data", "100"),
    ]
    group_w = min(w * 0.420, x + w - group_x - 0.018)
    anchors: dict[str, tuple[float, float]] = {"zoo_source": source}
    for idx, ((label, count), cy) in enumerate(zip(group_rows, group_ys), start=1):
        is_final = idx == 5
        gy = cy - group_h / 2
        ax.add_patch(
            FancyBboxPatch(
                (group_x, gy),
                group_w,
                group_h,
                boxstyle="round,pad=0.003,rounding_size=0.006",
                facecolor="#FFF7ED" if is_final else "#FFFFFF",
                edgecolor=PALETTE["orange"] if is_final else "#CBD5E1",
                linewidth=0.6,
            )
        )
        ax.text(group_x + 0.008, cy, label, ha="left", va="center", fontsize=pt(7.0), weight="semibold" if is_final else "normal")
        ax.text(group_x + group_w - 0.008, cy, count, ha="right", va="center", fontsize=pt(7.0), color=PALETTE["orange"] if is_final else "#475569", weight="semibold")
        add_arrow(ax, source, (group_x, cy), color=PALETTE["orange"] if is_final else "#94A3B8", lw=0.85 if is_final else 0.55)
        anchors[f"group_{idx}"] = (group_x + group_w, cy)

    ax.text(group_x + group_w, y + h * 0.415, "01-04: 10 agents x 28 checkpoints", ha="right", va="center", fontsize=pt(6.05), color="#64748B")
    ax.text(group_x + group_w, y + h * 0.080, "05: 100-agent outlier-search set", ha="right", va="center", fontsize=pt(6.05), color="#64748B")
    return anchors


def draw_similarity_card(
    ax: plt.Axes,
    rect: tuple[float, float, float, float],
    lsb_data: dict[str, dict[str, object]],
    *,
    style: str,
) -> None:
    add_card(ax, rect, "Within group action agreement", fc="#F1F8FF", ec=PALETTE["blue"], title_size=6.8)
    vals, annotations, message = action_values(lsb_data)
    x, y, w, h = rect
    ax.text(x + w - 0.018, y + h - 0.030, "NEARLY IDENTICAL", ha="right", va="top", fontsize=pt(7.1), color=PALETTE["orange"], weight="bold")
    ax.text(
        x + w - 0.018,
        y + h - 0.064,
        "minimum possible difference before\ntraining matches full randomness",
        ha="right",
        va="top",
        fontsize=pt(4.75),
        color="#475569",
        weight="semibold",
        linespacing=0.88,
    )
    draw_pairwise_agreement_matrix(
        ax,
        rect=rrect(rect, 0.030, 0.125, 0.260, 0.470),
        values=lsb_pairwise(lsb_data, "full"),
        label="03 init + data",
        color=PALETTE["blue"],
    )
    draw_pairwise_agreement_matrix(
        ax,
        rect=rrect(rect, 0.305, 0.125, 0.260, 0.470),
        values=lsb_pairwise(lsb_data, "lsb"),
        label="04 single LSB",
        color=PALETTE["orange"],
    )
    draw_compact_barplot(
        ax,
        rect=rrect(rect, 0.675, 0.095, 0.295, 0.475),
        title="",
        y_label="%",
        values=vals,
        labels=["03", "04"],
        colors=[PALETTE["blue"], PALETTE["orange"]],
        annotations=annotations,
        message="",
        value_font=5.0,
        axis_label="Mean agreement",
        label_font=4.9,
    )


def draw_outlier_card(
    ax: plt.Axes,
    rect: tuple[float, float, float, float],
    rates: pd.DataFrame,
    *,
    style: str,
) -> None:
    add_card(ax, rect, "Behavioral outlier search", fc="#FFF7ED", ec=PALETTE["orange"], title_size=6.6)
    vals, annotations, message = villager_values(rates)
    x, y, w, h = rect
    ax.text(x + w - 0.018, y + h - 0.038, "13x DIFFERENCE", ha="right", va="top", fontsize=pt(7.1), color=PALETTE["orange"], weight="bold")
    if style == "with_grid":
        plot_rect = rrect(rect, 0.585, 0.150, 0.350, 0.500)
    elif style == "wide":
        plot_rect = rrect(rect, 0.145, 0.175, 0.725, 0.430)
    elif style == "short":
        plot_rect = rrect(rect, 0.125, 0.195, 0.735, 0.390)
    else:
        plot_rect = rrect(rect, 0.185, 0.190, 0.650, 0.455)
    draw_compact_barplot(
        ax,
        rect=plot_rect,
        title="",
        y_label="%",
        values=vals,
        labels=["agent-111", "agent-114"],
        colors=[PALETTE["blue"], "#D95F02"],
        annotations=annotations,
        message="",
        value_font=5.1,
        axis_label="Villagers killed",
    )


def draw_compact_figure_layout(
    rates: pd.DataFrame,
    lsb_data: dict[str, dict[str, object]],
    out_path: Path,
    *,
    zoo_rect: tuple[float, float, float, float],
    sim_rect: tuple[float, float, float, float],
    outlier_rect: tuple[float, float, float, float],
    zoo_variant: str = "vertical",
    sim_style: str = "bars",
    outlier_style: str = "default",
    grid_rect: tuple[float, float, float, float] | None = None,
    pick_center: tuple[float, float] | None = None,
) -> None:
    fig, ax = make_compact_canvas()
    ax.text(zoo_rect[0], 0.958, "MultiSTEVE-1s Model Zoo", ha="left", va="top", fontsize=pt(9.0), weight="semibold", color=PALETTE["black"])
    ax.text(sim_rect[0], 0.958, "Example research it enables", ha="left", va="top", fontsize=pt(9.0), weight="semibold", color=PALETTE["black"])
    anchors = draw_model_zoo_compact(ax, zoo_rect, variant=zoo_variant)
    draw_similarity_card(ax, sim_rect, lsb_data, style=sim_style)
    draw_outlier_card(ax, outlier_rect, rates, style=outlier_style)

    sx, sy, sw, sh = sim_rect
    ox, oy, ow, oh = outlier_rect
    add_arrow(ax, anchors["group_3"], (sx, sy + sh * 0.62), color=PALETTE["blue"], lw=0.80)
    add_arrow(ax, anchors["group_4"], (sx, sy + sh * 0.40), color=PALETTE["orange"], lw=0.85)

    if grid_rect is not None:
        gx, gy, gw, gh = grid_rect
        source = anchors["group_5"]
        for target_y in np.linspace(gy + gh * 0.18, gy + gh * 0.82, 7):
            ax.plot([source[0], gx], [source[1], target_y], color=PALETTE["orange"], linewidth=0.45, alpha=0.32)
        add_arrow(ax, source, (gx, gy + gh * 0.50), color=PALETTE["orange"], lw=0.9)
        draw_compact_group05_grid(ax, grid_rect, label=False)
        label_x = gx + gw * 0.655
        label_y = gy + gh * 0.865
        ax.add_patch(
            FancyBboxPatch(
                (label_x - gw * 0.240, label_y - gh * 0.083),
                gw * 0.480,
                gh * 0.166,
                boxstyle="round,pad=0.002,rounding_size=0.004",
                facecolor="#FFF7ED",
                edgecolor=PALETTE["orange"],
                linewidth=0.45,
                alpha=0.96,
            )
        )
        ax.text(label_x, label_y, "choose agents\n111 and 114", ha="center", va="center", fontsize=pt(3.95), color=PALETTE["orange"], weight="semibold", linespacing=0.88)
        add_arrow(
            ax,
            (label_x + gw * 0.240, label_y + gh * 0.083),
            (ox + ow * 0.670, oy + oh * 0.440),
            color=PALETTE["orange"],
            lw=0.85,
            connectionstyle="arc3,rad=-0.22",
        )
    else:
        add_arrow(ax, anchors["group_5"], (ox, oy + oh * 0.46), color=PALETTE["orange"], lw=0.95)

    save_exact_figure(fig, out_path)


def draw_compact_layout_v1(rates: pd.DataFrame, lsb_data: dict[str, dict[str, object]], out_path: Path) -> None:
    draw_compact_figure_layout(
        rates,
        lsb_data,
        out_path,
        zoo_rect=(0.015, 0.065, 0.470, 0.805),
        sim_rect=(0.515, 0.489, 0.470, 0.386),
        outlier_rect=(0.515, 0.065, 0.470, 0.386),
        zoo_variant="row",
        sim_style="bars",
        outlier_style="with_grid",
        grid_rect=(0.535, 0.115, 0.165, 0.215),
    )


def draw_compact_layout_v2(rates: pd.DataFrame, lsb_data: dict[str, dict[str, object]], out_path: Path) -> None:
    draw_compact_figure_layout(
        rates,
        lsb_data,
        out_path,
        zoo_rect=(0.015, 0.065, 0.470, 0.805),
        sim_rect=(0.515, 0.489, 0.470, 0.386),
        outlier_rect=(0.515, 0.065, 0.470, 0.386),
        sim_style="bars",
        outlier_style="with_grid",
        grid_rect=(0.535, 0.115, 0.165, 0.215),
        zoo_variant="top_elbow",
    )


def draw_compact_layout_v3(rates: pd.DataFrame, lsb_data: dict[str, dict[str, object]], out_path: Path) -> None:
    draw_compact_figure_layout(
        rates,
        lsb_data,
        out_path,
        zoo_rect=(0.015, 0.065, 0.470, 0.805),
        sim_rect=(0.515, 0.489, 0.470, 0.386),
        outlier_rect=(0.515, 0.065, 0.470, 0.386),
        sim_style="bars",
        outlier_style="with_grid",
        grid_rect=(0.535, 0.115, 0.165, 0.215),
        zoo_variant="two_tier",
    )


def draw_compact_layout_v4(rates: pd.DataFrame, lsb_data: dict[str, dict[str, object]], out_path: Path) -> None:
    draw_compact_figure_layout(
        rates,
        lsb_data,
        out_path,
        zoo_rect=(0.015, 0.065, 0.470, 0.805),
        sim_rect=(0.515, 0.489, 0.470, 0.386),
        outlier_rect=(0.515, 0.065, 0.470, 0.386),
        sim_style="bars",
        outlier_style="with_grid",
        grid_rect=(0.535, 0.115, 0.165, 0.215),
        zoo_variant="stair",
    )


def draw_compact_layout_v5(rates: pd.DataFrame, lsb_data: dict[str, dict[str, object]], out_path: Path) -> None:
    draw_compact_figure_layout(
        rates,
        lsb_data,
        out_path,
        zoo_rect=(0.015, 0.065, 0.470, 0.805),
        sim_rect=(0.515, 0.489, 0.470, 0.386),
        outlier_rect=(0.515, 0.065, 0.470, 0.386),
        sim_style="bars",
        outlier_style="with_grid",
        grid_rect=(0.535, 0.115, 0.165, 0.215),
        zoo_variant="vertical",
    )


def draw_manual_bars(
    ax: plt.Axes,
    rates: pd.DataFrame,
    *,
    rect: tuple[float, float, float, float],
    title: str = "Focused comparison",
    max_pct: float = 100.0,
) -> None:
    x0, y0, w, h = rect
    ax.text(x0 + w / 2, y0 + h + 0.045, title, ha="center", va="bottom", fontsize=7.5, weight="semibold", color=PALETTE["black"])
    ax.text(x0 - 0.002, y0 + h + 0.012, "Kill rate (%)", ha="left", va="bottom", fontsize=5.8, color="#475569")
    for tick in [0, 50, 100]:
        ty = y0 + h * tick / max_pct
        ax.plot([x0, x0 + w], [ty, ty], color="#E2E8F0", linewidth=0.6, zorder=0)
        ax.text(x0 - 0.014, ty, str(tick), ha="right", va="center", fontsize=5.8, color=PALETTE["black"])
    ax.plot([x0, x0], [y0, y0 + h], color=PALETTE["black"], linewidth=0.75)
    ax.plot([x0, x0 + w], [y0, y0], color=PALETTE["black"], linewidth=0.75)

    colors = [PALETTE["blue"], PALETTE["red"]]
    xs = [x0 + w * 0.32, x0 + w * 0.73]
    bar_w = w * 0.24
    for xpos, color, row in zip(xs, colors, rates.itertuples(index=False)):
        rate = float(row.kill_rate_pct)
        lo = float(row.ci_low_pct)
        hi = float(row.ci_high_pct)
        bar_h = h * rate / max_pct
        ax.add_patch(Rectangle((xpos - bar_w / 2, y0), bar_w, bar_h, facecolor=color, edgecolor="white", linewidth=0.6))
        cy0 = y0 + h * lo / max_pct
        cy1 = y0 + h * hi / max_pct
        ax.plot([xpos, xpos], [cy0, cy1], color="#334155", linewidth=0.9)
        ax.plot([xpos - bar_w * 0.22, xpos + bar_w * 0.22], [cy0, cy0], color="#334155", linewidth=0.9)
        ax.plot([xpos - bar_w * 0.22, xpos + bar_w * 0.22], [cy1, cy1], color="#334155", linewidth=0.9)
        ax.text(xpos, y0 + bar_h + 0.022, f"{int(row.kills)}/{int(row.rollouts)}\n{rate:.1f}%", ha="center", va="bottom", fontsize=6.5, weight="semibold", linespacing=0.9)
        ax.text(xpos, y0 - 0.040, row.agent_label.replace(" ", "\n"), ha="center", va="top", fontsize=6.3, linespacing=0.9)


def draw_compact_pipeline(
    ax: plt.Axes,
    *,
    y: float,
    xs: tuple[float, float, float] = (0.06, 0.25, 0.44),
) -> dict[str, tuple[float, float]]:
    vpt_x, ft_x, zoo_x = xs
    add_box(ax, (vpt_x, y), (0.13, 0.13), "VPT", "Minecraft\npolicy", fc="#EFF6FF", ec=PALETTE["blue"], title_size=7.5, body_size=5.7)
    add_box(ax, (ft_x, y), (0.15, 0.13), "Fine-tune", "fixed STEVE-1\nrecipe", fc="#F8FAFC", title_size=7.4, body_size=5.6)
    add_box(ax, (zoo_x, y), (0.11, 0.13), "140", "agents", fc="#F0FDF4", ec=PALETTE["green"], title_size=9.0, body_size=5.8)
    mid_y = y + 0.065
    add_arrow(ax, (vpt_x + 0.13, mid_y), (ft_x, mid_y), lw=0.9)
    add_arrow(ax, (ft_x + 0.15, mid_y), (zoo_x, mid_y), lw=0.9)
    return {
        "vpt": (vpt_x + 0.13, mid_y),
        "fine_tune": (ft_x + 0.15, mid_y),
        "zoo": (zoo_x + 0.055, y),
        "zoo_right": (zoo_x + 0.11, mid_y),
    }


def draw_group_stack(ax: plt.Axes, *, x: float, y_top: float, w: float, row_h: float = 0.038) -> list[tuple[float, float]]:
    rows = [
        ("01 init seed only", "10"),
        ("02 data seed only", "10"),
        ("03 init + data", "10"),
        ("04 single LSB flip", "10"),
        ("05 final init + data", "100"),
    ]
    centers = []
    ax.text(x, y_top + 0.047, "Split into five groups", ha="left", va="center", fontsize=7.0, weight="semibold")
    for idx, (name, count) in enumerate(rows):
        y = y_top - idx * (row_h + 0.006)
        is_final = idx == 4
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                row_h,
                boxstyle="round,pad=0.004,rounding_size=0.007",
                facecolor="#FFF7ED" if is_final else "#F8FAFC",
                edgecolor=PALETTE["orange"] if is_final else "#CBD5E1",
                linewidth=0.65,
            )
        )
        ax.text(x + 0.010, y + row_h / 2, name, ha="left", va="center", fontsize=5.7, weight="semibold" if is_final else "normal")
        ax.text(x + w - 0.010, y + row_h / 2, count, ha="right", va="center", fontsize=5.7, color=PALETTE["orange"] if is_final else "#475569", weight="semibold")
        centers.append((x + w, y + row_h / 2))
    return centers


def draw_group_dots(
    ax: plt.Axes,
    *,
    group40_rect: tuple[float, float, float, float],
    group100_rect: tuple[float, float, float, float],
    show_top_labels: bool = True,
    show_bottom_labels: bool = True,
) -> None:
    x, y, w, h = group40_rect
    ax.add_patch(Rectangle((x, y), w, h, facecolor="#EFF6FF", edgecolor=PALETTE["blue"], linewidth=0.7))
    if show_top_labels:
        ax.text(x + w / 2, y + h + 0.025, "Groups 01-04", ha="center", va="bottom", fontsize=6.1, weight="semibold")
    if show_bottom_labels:
        ax.text(x + w / 2, y - 0.020, "40 agents", ha="center", va="top", fontsize=5.6, color="#475569")
    add_model_grid(ax, origin=(x + 0.015, y + h - 0.022), columns=10, rows=4, count=40, dx=(w - 0.03) / 9, dy=(h - 0.044) / 3, base_color=PALETTE["blue"], start_model=1, radius=0.0039)

    x, y, w, h = group100_rect
    ax.add_patch(Rectangle((x, y), w, h, facecolor="#FFF7ED", edgecolor=PALETTE["orange"], linewidth=0.7))
    if show_top_labels:
        ax.text(x + w / 2, y + h + 0.025, "Group 05: 100 agents", ha="center", va="bottom", fontsize=6.1, weight="semibold")
    if show_bottom_labels:
        ax.text(x + w / 2, y - 0.020, "outlier search", ha="center", va="top", fontsize=5.5, color=PALETTE["orange"], weight="semibold")
    add_model_grid(
        ax,
        origin=(x + 0.014, y + h - 0.018),
        columns=10,
        rows=10,
        count=100,
        dx=(w - 0.028) / 9,
        dy=(h - 0.036) / 9,
        base_color="#94A3B8",
        highlight={111: PALETTE["blue"], 114: PALETTE["red"]},
        start_model=101,
        radius=0.0032,
    )


def draw_selection_legend(ax: plt.Axes, *, x: float, y: float) -> None:
    ax.add_patch(Circle((x, y), 0.0055, facecolor=PALETTE["blue"], edgecolor="white", linewidth=0.2))
    ax.text(x + 0.014, y, "Agent 111", ha="left", va="center", fontsize=5.7)
    ax.add_patch(Circle((x, y - 0.040), 0.0055, facecolor=PALETTE["red"], edgecolor="white", linewidth=0.2))
    ax.text(x + 0.014, y - 0.040, "Agent 114", ha="left", va="center", fontsize=5.7)


def draw_pick_callout(ax: plt.Axes, *, center: tuple[float, float], text: str = "pick\n111 & 114") -> None:
    x, y = center
    ax.add_patch(
        FancyBboxPatch(
            (x - 0.035, y - 0.034),
            0.070,
            0.068,
            boxstyle="round,pad=0.006,rounding_size=0.018",
            facecolor="#FFF7ED",
            edgecolor=PALETTE["orange"],
            linewidth=0.75,
        )
    )
    ax.text(x, y, text, ha="center", va="center", fontsize=5.45, color=PALETTE["orange"], weight="semibold", linespacing=0.9)


def draw_group05_grid(ax: plt.Axes, *, rect: tuple[float, float, float, float]) -> None:
    x, y, w, h = rect
    ax.add_patch(Rectangle((x, y), w, h, facecolor="#FFF7ED", edgecolor=PALETTE["orange"], linewidth=0.75))
    ax.text(x + w / 2, y + h + 0.022, "Group 05: 100 agents", ha="center", va="bottom", fontsize=6.0, weight="semibold")
    ax.text(x + w / 2, y + h + 0.003, "outlier search", ha="center", va="bottom", fontsize=5.3, color=PALETTE["orange"], weight="semibold")
    add_model_grid(
        ax,
        origin=(x + 0.014, y + h - 0.020),
        columns=10,
        rows=10,
        count=100,
        dx=(w - 0.028) / 9,
        dy=(h - 0.040) / 9,
        base_color="#94A3B8",
        highlight={111: PALETTE["blue"], 114: PALETTE["red"]},
        start_model=101,
        radius=0.0032,
    )


def draw_requested_story_base(ax: plt.Axes, rates: pd.DataFrame) -> dict[str, tuple[float, float, float, float]]:
    ax.text(0.022, 0.950, "A", fontsize=9.2, weight="bold", color=PALETTE["black"])
    ax.text(0.058, 0.950, "Constructing agents and selecting outliers", fontsize=8.2, weight="semibold", color=PALETTE["black"])
    ax.text(0.615, 0.950, "B", fontsize=9.2, weight="bold", color=PALETTE["black"])
    ax.text(0.650, 0.950, "One-bit perturbations", fontsize=8.2, weight="semibold", color=PALETTE["black"])
    ax.plot([0.575, 0.575], [0.535, 0.925], color="#E2E8F0", linewidth=0.8)
    ax.plot([0.590, 0.985], [0.535, 0.535], color="#E2E8F0", linewidth=0.8)

    pipe_x = 0.045
    pipe_w = 0.125
    pipe_h = 0.102
    pipe_specs = [
        (0.780, "VPT", "Minecraft\npolicy", "#EFF6FF", PALETTE["blue"]),
        (0.610, "Fine-tune", "fixed STEVE-1\nrecipe", "#F8FAFC", "#4B5563"),
        (0.440, "140", "agents", "#F0FDF4", PALETTE["green"]),
    ]
    for idx, (y, title, body, fc, ec) in enumerate(pipe_specs):
        add_box(ax, (pipe_x, y), (pipe_w, pipe_h), title, body, fc=fc, ec=ec, title_size=7.5 if title != "140" else 9.0, body_size=5.45)
        if idx < len(pipe_specs) - 1:
            next_y = pipe_specs[idx + 1][0] + pipe_h
            add_arrow(ax, (pipe_x + pipe_w / 2, y), (pipe_x + pipe_w / 2, next_y), color="#64748B", lw=0.85)

    zoo_source = (pipe_x + pipe_w, 0.440 + pipe_h / 2)

    group_x = 0.245
    group_w = 0.260
    group_h = 0.040
    groups = [
        (0.780, "01 init seed", "10"),
        (0.705, "02 data seed", "10"),
        (0.630, "03 init + data", "10"),
        (0.555, "04 single LSB flip", "10"),
        (0.420, "05 final init + data", "100"),
    ]
    group_centers: dict[str, tuple[float, float]] = {}
    for idx, (cy, name, count) in enumerate(groups):
        is_final = idx == 4
        y = cy - group_h / 2
        ax.add_patch(
            FancyBboxPatch(
                (group_x, y),
                group_w,
                group_h,
                boxstyle="round,pad=0.004,rounding_size=0.009",
                facecolor="#FFF7ED" if is_final else "#F8FAFC",
                edgecolor=PALETTE["orange"] if is_final else "#CBD5E1",
                linewidth=0.70,
            )
        )
        ax.text(group_x + 0.010, cy, name, ha="left", va="center", fontsize=5.7, weight="semibold" if is_final else "normal")
        ax.text(group_x + group_w - 0.010, cy, count, ha="right", va="center", fontsize=5.7, color=PALETTE["orange"] if is_final else "#475569", weight="semibold")
        add_arrow(ax, zoo_source, (group_x, cy), color=PALETTE["orange"] if is_final else "#94A3B8", lw=0.95 if is_final else 0.65)
        group_centers[f"group_{idx + 1}"] = (group_x + group_w, cy)
    ax.text(group_x, 0.505, "Groups 01-04: 10 agents x 28 checkpoints", ha="left", va="top", fontsize=5.4, color="#64748B")
    ax.text(group_x + 0.010, 0.382, "100 final agents for outlier search", ha="left", va="top", fontsize=5.5, color=PALETTE["orange"], weight="semibold")

    add_arrow(ax, group_centers["group_3"], (0.600, 0.790), color=PALETTE["blue"], lw=0.80)
    add_arrow(ax, group_centers["group_4"], (0.600, 0.705), color=PALETTE["orange"], lw=0.90)

    grid_rect = (0.540, 0.105, 0.155, 0.260)
    source = group_centers["group_5"]
    fan_targets = np.linspace(grid_rect[1] + 0.030, grid_rect[1] + grid_rect[3] - 0.030, 9)
    for target_y in fan_targets:
        ax.plot([source[0], grid_rect[0]], [source[1], target_y], color=PALETTE["orange"], linewidth=0.55, alpha=0.32)
    add_arrow(ax, source, (grid_rect[0], grid_rect[1] + grid_rect[3] / 2), color=PALETTE["orange"], lw=1.05)
    draw_group05_grid(ax, rect=grid_rect)

    draw_pick_callout(ax, center=(0.742, 0.235))
    add_arrow(ax, (grid_rect[0] + grid_rect[2], 0.235), (0.704, 0.235), color=PALETTE["orange"], lw=0.9)
    add_arrow(ax, (0.778, 0.235), (0.810, 0.285), color=PALETTE["orange"], lw=0.9)

    c_frame = (0.795, 0.060, 0.185, 0.445)
    ax.add_patch(
        FancyBboxPatch(
            (c_frame[0], c_frame[1]),
            c_frame[2],
            c_frame[3],
            boxstyle="round,pad=0.008,rounding_size=0.018",
            facecolor="#FFFFFF",
            edgecolor="#CBD5E1",
            linewidth=0.75,
        )
    )
    ax.text(0.805, 0.488, "C", fontsize=9.2, weight="bold", color=PALETTE["black"])
    draw_manual_bars(ax, rates, rect=(0.830, 0.125, 0.125, 0.300), title="Villager kills")

    return {
        "lsb_inset": (0.600, 0.590, 0.375, 0.325),
        "bar_rect": (0.830, 0.125, 0.125, 0.300),
        "grid_rect": grid_rect,
    }


def draw_inset_frame(ax: plt.Axes, rect: tuple[float, float, float, float], title: str, subtitle: str = "") -> None:
    x, y, w, h = rect
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.008,rounding_size=0.018",
            facecolor="#FFFFFF",
            edgecolor="#CBD5E1",
            linewidth=0.75,
        )
    )
    ax.text(x + 0.015, y + h - 0.030, title, ha="left", va="top", fontsize=6.8, weight="semibold", color=PALETTE["black"])
    if subtitle:
        ax.text(x + 0.015, y + h - 0.065, subtitle, ha="left", va="top", fontsize=5.25, color="#64748B")


def lsb_mean(data: dict[str, dict[str, object]], key: str, field: str = "exact_disagree_pct") -> float:
    return float(data[key][field])


def lsb_pairwise(data: dict[str, dict[str, object]], key: str) -> list[float]:
    return [float(v) for v in data[key]["pairwise_exact_disagree_pct"]]


def draw_lsb_inset_bar_pair(ax: plt.Axes, rect: tuple[float, float, float, float], data: dict[str, dict[str, object]]) -> None:
    draw_inset_frame(ax, rect, "Exact-action agreement")
    x, y, w, h = rect
    plot_x = x + 0.055
    plot_y = y + 0.070
    plot_w = w - 0.105
    plot_h = h - 0.165
    for tick in [0, 50, 100]:
        ty = plot_y + plot_h * tick / 100.0
        ax.plot([plot_x, plot_x + plot_w], [ty, ty], color="#E2E8F0", linewidth=0.55)
        ax.text(plot_x - 0.012, ty, str(tick), ha="right", va="center", fontsize=5.1, color="#475569")
    ax.text(plot_x - 0.012, plot_y + plot_h + 0.015, "%", ha="right", va="bottom", fontsize=5.1, color="#475569")
    vals = [lsb_mean(data, "full"), lsb_mean(data, "lsb")]
    labels = ["full\nrandomness", "single\nLSB flip"]
    colors = [PALETTE["blue"], PALETTE["orange"]]
    xs = [plot_x + plot_w * 0.33, plot_x + plot_w * 0.72]
    bar_w = plot_w * 0.19
    for xpos, val, label, color in zip(xs, vals, labels, colors):
        bh = plot_h * val / 100.0
        ax.add_patch(Rectangle((xpos - bar_w / 2, plot_y), bar_w, bh, facecolor=color, edgecolor="white", linewidth=0.5, alpha=0.90))
        ax.text(xpos, plot_y + bh + 0.013, f"{val:.1f}%", ha="center", va="bottom", fontsize=5.7, weight="semibold")
        ax.text(xpos, plot_y - 0.020, label, ha="center", va="top", fontsize=5.2, linespacing=0.9)
    ax.text(x + w - 0.018, y + h - 0.052, f"only {abs(vals[0] - vals[1]):.1f} pp apart", ha="right", va="top", fontsize=5.7, color=PALETTE["orange"], weight="semibold")


def draw_lsb_inset_dot_strip(ax: plt.Axes, rect: tuple[float, float, float, float], data: dict[str, dict[str, object]]) -> None:
    draw_inset_frame(ax, rect, "Pairwise exact-action agreement", "each dot is one model pair")
    x, y, w, h = rect
    vals_full = lsb_pairwise(data, "full")
    vals_lsb = lsb_pairwise(data, "lsb")
    lo, hi = 0.0, 100.0
    axis_x0 = x + 0.125
    axis_x1 = x + w - 0.030
    axis_y = y + 0.078
    ax.plot([axis_x0, axis_x1], [axis_y, axis_y], color="#CBD5E1", linewidth=0.65)
    band0 = axis_x0 + (38.0 - lo) / (hi - lo) * (axis_x1 - axis_x0)
    band1 = axis_x0 + (46.0 - lo) / (hi - lo) * (axis_x1 - axis_x0)
    ax.add_patch(Rectangle((band0, y + 0.115), band1 - band0, 0.125, facecolor="#FEF3C7", edgecolor="none", alpha=0.55))
    for tick in [0, 50, 100]:
        tx = axis_x0 + (tick - lo) / (hi - lo) * (axis_x1 - axis_x0)
        ax.plot([tx, tx], [axis_y - 0.008, axis_y + 0.008], color="#CBD5E1", linewidth=0.6)
        ax.text(tx, axis_y - 0.020, f"{tick:.0f}", ha="center", va="top", fontsize=4.9, color="#475569")
    ax.text(axis_x1, axis_y - 0.045, "% agreement", ha="right", va="top", fontsize=4.9, color="#475569")

    rows = [("full", vals_full, PALETTE["blue"], y + 0.210), ("LSB", vals_lsb, PALETTE["orange"], y + 0.145)]
    for label, vals, color, row_y in rows:
        ax.text(x + 0.018, row_y, label, ha="left", va="center", fontsize=5.6, weight="semibold", color=PALETTE["black"])
        for i, val in enumerate(sorted(vals)):
            tx = axis_x0 + (val - lo) / (hi - lo) * (axis_x1 - axis_x0)
            jitter = ((i % 7) - 3) * 0.0028
            ax.add_patch(Circle((tx, row_y + jitter), 0.0036, facecolor=color, edgecolor="none", alpha=0.68))
        mean = float(np.mean(vals))
        mx = axis_x0 + (mean - lo) / (hi - lo) * (axis_x1 - axis_x0)
        ax.plot([mx, mx], [row_y - 0.023, row_y + 0.023], color=PALETTE["black"], linewidth=0.75)
        ax.text(mx, row_y + 0.033, f"{mean:.1f}", ha="center", va="bottom", fontsize=5.0, weight="semibold")
    ax.text(x + w - 0.018, y + h - 0.052, "same narrow band", ha="right", va="top", fontsize=5.7, color=PALETTE["orange"], weight="semibold")


def draw_lsb_inset_ruler(ax: plt.Axes, rect: tuple[float, float, float, float], data: dict[str, dict[str, object]]) -> None:
    draw_inset_frame(ax, rect, "One bit lands beside full randomness", "exact-action agreement")
    x, y, w, h = rect
    axis_x0 = x + 0.055
    axis_x1 = x + w - 0.045
    axis_y = y + 0.145
    lo, hi = 0.0, 100.0
    ax.plot([axis_x0, axis_x1], [axis_y, axis_y], color="#CBD5E1", linewidth=2.0, solid_capstyle="round")
    for tick in [0, 50, 100]:
        tx = axis_x0 + (tick - lo) / (hi - lo) * (axis_x1 - axis_x0)
        ax.plot([tx, tx], [axis_y - 0.016, axis_y + 0.016], color="#94A3B8", linewidth=0.75)
        ax.text(tx, axis_y - 0.034, str(tick), ha="center", va="top", fontsize=5.0, color="#475569")
    markers = [
        ("Full randomness", lsb_mean(data, "full"), PALETTE["blue"], 0.055, 0.074),
        ("Single LSB flip", lsb_mean(data, "lsb"), PALETTE["orange"], -0.050, -0.070),
    ]
    for label, val, color, dx, dy in markers:
        tx = axis_x0 + (val - lo) / (hi - lo) * (axis_x1 - axis_x0)
        ax.add_patch(Circle((tx, axis_y), 0.011, facecolor=color, edgecolor="white", linewidth=0.6))
        ax.text(tx + dx, axis_y + dy, f"{label}\n{val:.1f}%", ha="center", va="center", fontsize=5.4, color=PALETTE["black"], weight="semibold", linespacing=0.9)
    ax.text(x + w - 0.020, y + 0.035, f"Delta = {abs(markers[0][1] - markers[1][1]):.1f} pp", ha="right", va="bottom", fontsize=5.6, color=PALETTE["orange"], weight="semibold")


def draw_lsb_inset_metric_rows(ax: plt.Axes, rect: tuple[float, float, float, float], data: dict[str, dict[str, object]]) -> None:
    draw_inset_frame(ax, rect, "Same outcome from very different causes")
    x, y, w, h = rect
    full_val = lsb_mean(data, "full")
    lsb_val = lsb_mean(data, "lsb")
    ax.text(x + 0.085, y + 0.185, f"{full_val:.1f}%", ha="center", va="center", fontsize=13.0, color=PALETTE["blue"], weight="bold")
    ax.text(x + 0.085, y + 0.123, "full randomness", ha="center", va="center", fontsize=5.4, color="#475569")
    ax.text(x + w / 2, y + 0.163, "≈", ha="center", va="center", fontsize=20.0, color=PALETTE["black"], weight="bold")
    ax.text(x + w - 0.085, y + 0.185, f"{lsb_val:.1f}%", ha="center", va="center", fontsize=13.0, color=PALETTE["orange"], weight="bold")
    ax.text(x + w - 0.085, y + 0.123, "single LSB flip", ha="center", va="center", fontsize=5.4, color="#475569")
    axis_x0 = x + 0.052
    axis_x1 = x + w - 0.052
    axis_y = y + 0.068
    ax.plot([axis_x0, axis_x1], [axis_y, axis_y], color="#CBD5E1", linewidth=0.65)
    for tick in [0, 50, 100]:
        tx = axis_x0 + tick / 100.0 * (axis_x1 - axis_x0)
        ax.plot([tx, tx], [axis_y - 0.007, axis_y + 0.007], color="#CBD5E1", linewidth=0.6)
        ax.text(tx, axis_y - 0.018, str(tick), ha="center", va="top", fontsize=4.7, color="#64748B")
    ax.text(x + w - 0.018, y + h - 0.052, "nearly identical", ha="right", va="top", fontsize=5.7, color=PALETTE["orange"], weight="semibold")


def draw_lsb_inset_pair_matrices(ax: plt.Axes, rect: tuple[float, float, float, float], data: dict[str, dict[str, object]]) -> None:
    draw_inset_frame(ax, rect, "Pairwise maps share a color band", "exact-action agreement")
    x, y, w, h = rect
    mats = [("Full randomness", lsb_pairwise(data, "full"), PALETTE["blue"], x + 0.045), ("Single LSB", lsb_pairwise(data, "lsb"), PALETTE["orange"], x + 0.215)]
    for label, vals, color, mx in mats:
        size = 0.016
        vals = sorted(vals)
        vmin, vmax = min(vals), max(vals)
        idx = 0
        for row in range(6):
            for col in range(row):
                if idx >= len(vals):
                    continue
                val = vals[idx]
                idx += 1
                alpha = 0.25 + 0.62 * (val - vmin) / max(vmax - vmin, 1e-6)
                ax.add_patch(Rectangle((mx + col * size, y + 0.095 + (5 - row) * size), size * 0.86, size * 0.86, facecolor=color, edgecolor="none", alpha=alpha))
        ax.text(mx + size * 2.4, y + 0.220, label, ha="center", va="bottom", fontsize=5.4, weight="semibold")
        ax.text(mx + size * 2.4, y + 0.065, f"mean {float(np.mean(vals)):.1f}%", ha="center", va="top", fontsize=5.2, color="#475569")
    axis_x0 = x + 0.055
    axis_x1 = x + w - 0.055
    axis_y = y + 0.045
    ax.plot([axis_x0, axis_x1], [axis_y, axis_y], color="#CBD5E1", linewidth=0.65)
    for tick in [0, 50, 100]:
        tx = axis_x0 + tick / 100.0 * (axis_x1 - axis_x0)
        ax.plot([tx, tx], [axis_y - 0.006, axis_y + 0.006], color="#CBD5E1", linewidth=0.55)
        ax.text(tx, axis_y - 0.017, str(tick), ha="center", va="top", fontsize=4.6, color="#64748B")
    ax.text(x + w - 0.018, y + h - 0.070, "same band", ha="right", va="top", fontsize=5.4, color=PALETTE["orange"], weight="semibold")


def draw_requested_layout(
    rates: pd.DataFrame,
    lsb_data: dict[str, dict[str, object]],
    out_path: Path,
    inset_renderer: Callable[[plt.Axes, tuple[float, float, float, float], dict[str, dict[str, object]]], None],
) -> None:
    fig, ax = make_story_canvas()
    rects = draw_requested_story_base(ax, rates)
    inset_renderer(ax, rects["lsb_inset"], lsb_data)
    save_exact_figure(fig, out_path)


def draw_requested_v1_bars(rates: pd.DataFrame, lsb_data: dict[str, dict[str, object]], out_path: Path) -> None:
    draw_requested_layout(rates, lsb_data, out_path, draw_lsb_inset_bar_pair)


def draw_requested_v2_dot_strip(rates: pd.DataFrame, lsb_data: dict[str, dict[str, object]], out_path: Path) -> None:
    draw_requested_layout(rates, lsb_data, out_path, draw_lsb_inset_dot_strip)


def draw_requested_v3_ruler(rates: pd.DataFrame, lsb_data: dict[str, dict[str, object]], out_path: Path) -> None:
    draw_requested_layout(rates, lsb_data, out_path, draw_lsb_inset_ruler)


def draw_requested_v4_metric_rows(rates: pd.DataFrame, lsb_data: dict[str, dict[str, object]], out_path: Path) -> None:
    draw_requested_layout(rates, lsb_data, out_path, draw_lsb_inset_metric_rows)


def draw_requested_v5_pair_matrices(rates: pd.DataFrame, lsb_data: dict[str, dict[str, object]], out_path: Path) -> None:
    draw_requested_layout(rates, lsb_data, out_path, draw_lsb_inset_pair_matrices)


def draw_connected_v1_branching(rates: pd.DataFrame, out_path: Path) -> None:
    fig, ax = make_story_canvas()
    ax.text(0.02, 0.94, "A", fontsize=9.5, weight="bold")
    ax.text(0.065, 0.94, "Branching construction and outlier selection", fontsize=8.7, weight="semibold")
    ax.text(0.80, 0.94, "B", fontsize=9.5, weight="bold")
    pipe = draw_compact_pipeline(ax, y=0.76, xs=(0.04, 0.205, 0.375))
    centers = draw_group_stack(ax, x=0.045, y_top=0.55, w=0.36)

    spine_x = 0.430
    add_arrow(ax, pipe["zoo"], (spine_x, centers[0][1] + 0.045), color="#64748B", lw=0.8)
    ax.plot([spine_x, spine_x], [centers[0][1] + 0.037, centers[-1][1]], color="#94A3B8", linewidth=0.8)
    for idx, end in enumerate(centers):
        add_arrow(ax, (spine_x, end[1]), (end[0], end[1]), color=PALETTE["orange"] if idx == 4 else "#94A3B8", lw=0.75 if idx != 4 else 1.0)

    draw_group_dots(
        ax,
        group40_rect=(0.490, 0.445, 0.105, 0.115),
        group100_rect=(0.625, 0.335, 0.165, 0.225),
        show_bottom_labels=False,
    )
    add_arrow(ax, centers[4], (0.625, 0.445), color=PALETTE["orange"], lw=1.0)
    draw_pick_callout(ax, center=(0.805, 0.295))
    add_arrow(ax, (0.775, 0.365), (0.785, 0.330), color=PALETTE["orange"], lw=0.8)
    add_arrow(ax, (0.832, 0.315), (0.845, 0.430), color=PALETTE["orange"], lw=0.8)
    draw_manual_bars(ax, rates, rect=(0.845, 0.18, 0.135, 0.62), title="Kill rates")
    save_exact_figure(fig, out_path)


def draw_connected_v2_sankey(rates: pd.DataFrame, out_path: Path) -> None:
    fig, ax = make_story_canvas()
    ax.text(0.02, 0.94, "A", fontsize=9.5, weight="bold")
    ax.text(0.065, 0.94, "Flow from recipe to screened outliers", fontsize=8.7, weight="semibold")
    ax.text(0.80, 0.94, "B", fontsize=9.5, weight="bold")
    pipe = draw_compact_pipeline(ax, y=0.76, xs=(0.045, 0.215, 0.385))
    group_y = [0.640, 0.575, 0.510, 0.445, 0.380]
    group_names = ["01 init seed", "02 data seed", "03 init+data", "04 bit flip", "05 final init+data"]
    counts = ["10", "10", "10", "10", "100"]
    ax.text(0.535, 0.675, "split", ha="center", va="center", fontsize=6.4, weight="semibold")
    for idx, (gy, name, count) in enumerate(zip(group_y, group_names, counts)):
        is_final = idx == 4
        color = PALETTE["orange"] if is_final else "#CBD5E1"
        lw = 5.6 if is_final else 2.4
        ax.plot([pipe["zoo_right"][0], 0.535], [pipe["zoo_right"][1], gy], color=color, linewidth=lw, alpha=0.48, solid_capstyle="round")
        ax.add_patch(
            FancyBboxPatch(
                (0.548, gy - 0.020),
                0.205,
                0.040,
                boxstyle="round,pad=0.004,rounding_size=0.009",
                facecolor="#FFF7ED" if is_final else "#F8FAFC",
                edgecolor=PALETTE["orange"] if is_final else "#CBD5E1",
                linewidth=0.6,
            )
        )
        ax.text(0.558, gy, name, ha="left", va="center", fontsize=5.7, weight="semibold" if is_final else "normal")
        ax.text(0.742, gy, count, ha="right", va="center", fontsize=5.7, color=PALETTE["orange"] if is_final else "#475569", weight="semibold")

    draw_group_dots(
        ax,
        group40_rect=(0.465, 0.150, 0.110, 0.118),
        group100_rect=(0.630, 0.105, 0.170, 0.215),
        show_top_labels=False,
        show_bottom_labels=False,
    )
    ax.text(0.520, 0.284, "40 agents", ha="center", va="bottom", fontsize=5.8, color="#475569", weight="semibold")
    ax.text(0.715, 0.087, "100-agent outlier search", ha="center", va="top", fontsize=5.8, color=PALETTE["orange"], weight="semibold")
    add_arrow(ax, (0.650, 0.425), (0.715, 0.320), color=PALETTE["orange"], lw=0.9)
    draw_pick_callout(ax, center=(0.800, 0.265))
    add_arrow(ax, (0.795, 0.230), (0.768, 0.250), color=PALETTE["orange"], lw=0.8)
    add_arrow(ax, (0.835, 0.290), (0.845, 0.455), color=PALETTE["orange"], lw=0.9)
    draw_manual_bars(ax, rates, rect=(0.845, 0.18, 0.135, 0.62), title="Kill rates")
    save_exact_figure(fig, out_path)


def draw_connected_v3_zoom(rates: pd.DataFrame, out_path: Path) -> None:
    fig, ax = make_story_canvas()
    ax.text(0.02, 0.94, "A", fontsize=9.5, weight="bold")
    ax.text(0.065, 0.94, "Zooming from the zoo into two agents", fontsize=8.7, weight="semibold")
    ax.text(0.80, 0.94, "B", fontsize=9.5, weight="bold")
    pipe = draw_compact_pipeline(ax, y=0.76, xs=(0.045, 0.215, 0.385))
    centers = draw_group_stack(ax, x=0.050, y_top=0.550, w=0.315)
    add_arrow(ax, pipe["zoo"], (0.210, 0.625), color="#64748B", lw=0.8)
    add_arrow(ax, centers[4], (0.535, 0.455), color=PALETTE["orange"], lw=1.0)
    draw_group_dots(
        ax,
        group40_rect=(0.395, 0.425, 0.110, 0.120),
        group100_rect=(0.535, 0.330, 0.220, 0.245),
        show_bottom_labels=False,
    )
    ax.add_patch(Circle((0.585, 0.537), 0.064, fill=False, edgecolor=PALETTE["orange"], linewidth=1.0))
    ax.text(0.645, 0.278, "screen 100\nfor outliers", ha="center", va="center", fontsize=5.9, color=PALETTE["orange"], weight="semibold", linespacing=0.9)
    draw_pick_callout(ax, center=(0.785, 0.430))
    add_arrow(ax, (0.650, 0.520), (0.760, 0.450), color=PALETTE["orange"], lw=0.9)
    add_arrow(ax, (0.825, 0.430), (0.845, 0.430), color=PALETTE["orange"], lw=0.9)
    draw_manual_bars(ax, rates, rect=(0.845, 0.18, 0.135, 0.62), title="Kill rates")
    save_exact_figure(fig, out_path)


def draw_connected_v4_staircase(rates: pd.DataFrame, out_path: Path) -> None:
    fig, ax = make_story_canvas()
    ax.text(0.02, 0.94, "A", fontsize=9.5, weight="bold")
    ax.text(0.065, 0.94, "Stepwise screen from zoo to comparison", fontsize=8.7, weight="semibold")
    ax.text(0.80, 0.94, "B", fontsize=9.5, weight="bold")
    pipe = draw_compact_pipeline(ax, y=0.76, xs=(0.040, 0.205, 0.375))
    add_arrow(ax, pipe["zoo"], (0.440, 0.640), color="#64748B", lw=0.8)
    centers = draw_group_stack(ax, x=0.255, y_top=0.570, w=0.325)

    add_box(ax, (0.055, 0.235), (0.155, 0.095), "01-04", "40 agents", fc="#EFF6FF", ec=PALETTE["blue"], title_size=7.2, body_size=5.8)
    add_box(ax, (0.455, 0.185), (0.170, 0.105), "05", "100-agent\noutlier search", fc="#FFF7ED", ec=PALETTE["orange"], title_size=7.2, body_size=5.6)
    add_arrow(ax, (0.255, 0.390), (0.132, 0.330), color="#94A3B8", lw=0.75)
    add_arrow(ax, centers[4], (0.455, 0.245), color=PALETTE["orange"], lw=1.0)
    draw_group_dots(
        ax,
        group40_rect=(0.075, 0.075, 0.115, 0.115),
        group100_rect=(0.635, 0.155, 0.170, 0.220),
        show_top_labels=False,
        show_bottom_labels=False,
    )
    add_arrow(ax, (0.625, 0.245), (0.635, 0.265), color=PALETTE["orange"], lw=0.9)
    draw_pick_callout(ax, center=(0.795, 0.430))
    add_arrow(ax, (0.735, 0.365), (0.775, 0.430), color=PALETTE["orange"], lw=0.9)
    add_arrow(ax, (0.835, 0.430), (0.845, 0.430), color=PALETTE["orange"], lw=0.9)
    draw_manual_bars(ax, rates, rect=(0.845, 0.18, 0.135, 0.62), title="Kill rates")
    save_exact_figure(fig, out_path)


def draw_connected_v5_bracket_lanes(rates: pd.DataFrame, out_path: Path) -> None:
    fig, ax = make_story_canvas()
    ax.text(0.02, 0.94, "A", fontsize=9.5, weight="bold")
    ax.text(0.065, 0.94, "Lane view of the split and selected outliers", fontsize=8.7, weight="semibold")
    ax.text(0.80, 0.94, "B", fontsize=9.5, weight="bold")
    pipe = draw_compact_pipeline(ax, y=0.76, xs=(0.040, 0.205, 0.375))
    lane_x = 0.055
    lane_y = [0.585, 0.520, 0.455, 0.390, 0.270]
    lane_names = ["01 init seed only", "02 data seed only", "03 init + data", "04 single LSB flip", "05 final init + data"]
    for idx, (y, name) in enumerate(zip(lane_y, lane_names)):
        is_final = idx == 4
        ax.add_patch(Rectangle((lane_x, y - 0.020), 0.475, 0.040, facecolor="#FFF7ED" if is_final else "#F8FAFC", edgecolor=PALETTE["orange"] if is_final else "#CBD5E1", linewidth=0.65))
        ax.text(lane_x + 0.010, y, name, ha="left", va="center", fontsize=5.7, weight="semibold" if is_final else "normal")
        ax.text(lane_x + 0.452, y, "100 agents" if is_final else "10 agents", ha="right", va="center", fontsize=5.6, color=PALETTE["orange"] if is_final else "#475569", weight="semibold" if is_final else "normal")
        add_arrow(ax, pipe["zoo"], (lane_x, y), color=PALETTE["orange"] if is_final else "#94A3B8", lw=0.70 if not is_final else 0.95)

    bracket_x = 0.545
    ax.plot([bracket_x, bracket_x], [lane_y[3], lane_y[0]], color="#94A3B8", linewidth=0.8)
    ax.plot([bracket_x, 0.580], [lane_y[0], lane_y[0]], color="#94A3B8", linewidth=0.8)
    ax.plot([bracket_x, 0.580], [lane_y[3], lane_y[3]], color="#94A3B8", linewidth=0.8)
    ax.text(0.662, 0.590, "Groups 01-04: 40 agents", ha="center", va="bottom", fontsize=5.6, color="#475569", weight="semibold")

    draw_group_dots(
        ax,
        group40_rect=(0.610, 0.455, 0.105, 0.115),
        group100_rect=(0.610, 0.185, 0.175, 0.220),
        show_top_labels=False,
        show_bottom_labels=False,
    )
    add_arrow(ax, (lane_x + 0.475, lane_y[4]), (0.610, 0.295), color=PALETTE["orange"], lw=1.0)
    draw_pick_callout(ax, center=(0.805, 0.295))
    add_arrow(ax, (0.785, 0.295), (0.760, 0.295), color=PALETTE["orange"], lw=0.8)
    add_arrow(ax, (0.845, 0.305), (0.845, 0.450), color=PALETTE["orange"], lw=0.9)
    draw_manual_bars(ax, rates, rect=(0.845, 0.18, 0.135, 0.62), title="Kill rates")
    save_exact_figure(fig, out_path)


def write_metadata(
    out_dir: Path,
    paths: list[Path],
    rates: pd.DataFrame,
    paper_zip: Path,
    agent_totals: Path,
    action_summary: Path | None = None,
    action_comparisons_dir: Path | None = None,
    lsb_data: dict[str, dict[str, object]] | None = None,
) -> None:
    metadata = {
        "paper_zip": str(paper_zip.resolve()),
        "paper_text_member": "neurips_2026.tex",
        "agent_totals_source": str(agent_totals.resolve()),
        "agent_rates": rates[["agent_key", "agent_label", "rollouts", "kills", "kill_rate_pct", "ci_low_pct", "ci_high_pct"]].to_dict(orient="records"),
        "outputs": [str(path.resolve()) for path in paths],
    }
    if action_summary is not None:
        metadata["action_summary_source"] = str(action_summary.resolve())
    if action_comparisons_dir is not None:
        metadata["action_comparisons_dir"] = str(action_comparisons_dir.resolve())
    if lsb_data is not None:
        metadata["lsb_vs_full_randomness"] = lsb_data
    (out_dir / "figure1_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    apply_style({"font_size": 8.0, "title_font_size": 8.8, "label_font_size": 8.0, "pad_inches": 0.01})
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    rates = load_agent_rates(args.agent_totals.expanduser().resolve())
    action_summary = args.action_summary.expanduser().resolve()
    action_comparisons_dir = args.action_comparisons_dir.expanduser().resolve()
    lsb_data = load_lsb_randomness_summary(action_summary, action_comparisons_dir)
    variants: list[tuple[str, Callable[[pd.DataFrame, dict[str, dict[str, object]], Path], None]]] = [
        ("figure1_zoo_research.pdf", draw_compact_layout_v1),
    ]
    output_paths: list[Path] = []
    for filename, renderer in variants:
        path = out_dir / filename
        renderer(rates, lsb_data, path)
        output_paths.append(path)
    write_metadata(out_dir, output_paths, rates, args.paper_zip, args.agent_totals, action_summary, action_comparisons_dir, lsb_data)
    print(f"Output folder: {out_dir}")
    for path in output_paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
