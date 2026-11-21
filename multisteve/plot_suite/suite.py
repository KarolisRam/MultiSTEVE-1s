from __future__ import annotations

import ast
import fnmatch
import json
import math
import re
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patheffects
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnnotationBbox, DrawingArea
from matplotlib.patches import Polygon, Rectangle
from matplotlib.ticker import FormatStrFormatter, MaxNLocator

from .config import EVALS_VILLAGER_ROOT, REPO_ROOT
from .loaders import import_module_from_path
from .style import NEURIPS_TEXT_WIDTH_IN, SIDE_BY_SIDE_FIGURE_WIDTH_IN, PALETTE, apply_style, save_figure

PASS_BAR_COLORWAYS: list[dict[str, str]] = [
    {"slug": "01_blue_vermillion", "label": "Blue bars, vermillion dots", "bar": "#0072B2", "dot": "#D55E00"},
    {"slug": "02_teal_orange", "label": "Teal bars, orange dots", "bar": "#009E73", "dot": "#E69F00"},
    {"slug": "03_indigo_orange", "label": "Indigo bars, orange dots", "bar": "#332288", "dot": "#EE7733"},
    {"slug": "04_sky_blue_magenta", "label": "Sky-blue bars, magenta dots", "bar": "#56B4E9", "dot": "#AA4499"},
    {"slug": "05_gray_vermillion", "label": "Gray bars, vermillion dots", "bar": "#6B7280", "dot": "#D55E00"},
]


@dataclass
class Artifact:
    id: str
    path: str
    family: str
    description: str


@dataclass
class RunContext:
    config: dict[str, Any]
    run_dir: Path
    figures_dir: Path
    cache_root: Path
    overleaf_dir: Path
    from_cache: bool
    only: set[str] | None
    artifacts: list[Artifact]
    warnings: list[str]


def make_run_name() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def path_from_config(value: str) -> Path:
    return Path(value).expanduser().resolve()


def enabled(config: dict[str, Any], family: str) -> bool:
    return bool(config["families"].get(family, {}).get("enabled", False))


def should_run(ctx: RunContext, family: str) -> bool:
    return ctx.only is None or family in ctx.only


def add_artifact(ctx: RunContext, family: str, artifact_id: str, path: Path, description: str) -> None:
    if not path.exists():
        ctx.warnings.append(f"Expected artifact was not written: {path}")
        return
    ctx.artifacts.append(
        Artifact(
            id=artifact_id,
            path=str(path),
            family=family,
            description=description,
        )
    )


def clipped_interval_errors(
    means: np.ndarray,
    lows: np.ndarray,
    highs: np.ndarray,
    limits: tuple[float | None, float | None],
) -> np.ndarray:
    low_bound, high_bound = limits
    clipped_lows = np.asarray(lows, dtype=float).copy()
    clipped_highs = np.asarray(highs, dtype=float).copy()
    if low_bound is not None:
        clipped_lows = np.maximum(clipped_lows, float(low_bound))
    if high_bound is not None:
        clipped_highs = np.minimum(clipped_highs, float(high_bound))
    means = np.asarray(means, dtype=float)
    return np.vstack([np.maximum(0.0, means - clipped_lows), np.maximum(0.0, clipped_highs - means)])


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def update_latest_link(link_path: Path, target_path: Path, warnings: list[str]) -> None:
    if link_path.is_symlink() or link_path.is_file():
        link_path.unlink()
    if link_path.exists():
        warnings.append(f"Did not replace non-symlink latest path: {link_path}")
        return
    link_path.symlink_to(target_path, target_is_directory=True)


def parse_list_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for column in columns:
        if column not in out.columns:
            continue
        out[column] = out[column].apply(
            lambda value: ast.literal_eval(value)
            if isinstance(value, str) and value.strip().startswith("[")
            else ([] if pd.isna(value) else value)
        )
    return out


def rows_by_setting(df: pd.DataFrame) -> dict[tuple[str, str], list[dict[str, object]]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for (logs_setting, prompt_slug), group in df.groupby(["logs_setting", "prompt_slug"], sort=False):
        grouped[(str(logs_setting), str(prompt_slug))] = group.to_dict(orient="records")
    return grouped


def load_villager_passes_module():
    return import_module_from_path(
        "plot_suite_villager_passes",
        REPO_ROOT / "tools" / "villager_analysis" / "plot_villager_passes_01_18.py",
    )


def load_tree_vs_villager_module():
    return import_module_from_path(
        "plot_suite_tree_vs_villager",
        REPO_ROOT / "tools" / "villager_analysis" / "plot_tree_vs_villager_passes_01_18.py",
    )


def load_villager_40_module():
    return import_module_from_path(
        "plot_suite_villager_40",
        REPO_ROOT / "tools" / "villager_analysis" / "plot_villager_chop_40_models_01_04.py",
    )


def load_bitflip_module():
    return import_module_from_path(
        "plot_suite_villager_bitflip",
        REPO_ROOT / "tools" / "villager_analysis" / "plot_villager_chop_bitflip_10_wood_engineered.py",
    )


def load_agent_111_114_seed_sweep_module():
    return import_module_from_path(
        "plot_suite_agent_111_114_seed_sweep",
        REPO_ROOT / "tools" / "villager_analysis" / "plot_agent_111_114_seed_sweep.py",
    )


class PlotSuite:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    def run(
        self,
        *,
        run_name: str | None = None,
        from_cache: bool = False,
        only: set[str] | None = None,
        build_overleaf: bool = True,
    ) -> RunContext:
        output_root = path_from_config(self.config["outputs"]["root"])
        cache_root = path_from_config(self.config["outputs"]["cache_root"])
        overleaf_root = path_from_config(self.config["outputs"]["overleaf_root"])
        run_name = run_name or make_run_name()
        run_dir = output_root / "runs" / run_name
        ctx = RunContext(
            config=self.config,
            run_dir=run_dir,
            figures_dir=run_dir / "figures",
            cache_root=cache_root,
            overleaf_dir=overleaf_root / run_name,
            from_cache=from_cache,
            only=only,
            artifacts=[],
            warnings=[],
        )
        ctx.figures_dir.mkdir(parents=True, exist_ok=True)
        ctx.cache_root.mkdir(parents=True, exist_ok=True)

        apply_style(self.config.get("style", {}))

        for family, method in [
            ("full_eval", self.render_full_eval),
            ("partial_eval_40", self.render_partial_eval_40),
            ("partial_eval_28_checkpoints", self.render_partial_eval_28),
            ("tree_prompt_100", self.render_tree_prompt_100),
            ("villager_passes", self.render_villager_passes),
            ("tree_vs_villager", self.render_tree_vs_villager),
            ("villager_chop_40", self.render_villager_chop_40),
            ("villager_bitflip", self.render_villager_bitflip),
            ("agent_111_114_seed_sweep", self.render_agent_111_114_seed_sweep),
            ("head_fake_stability", self.render_head_fake_stability),
            ("teleport_layout", self.render_teleport_layout),
            ("agent_trajectories", self.render_agent_trajectories),
        ]:
            if enabled(self.config, family) and should_run(ctx, family):
                method(ctx)

        if build_overleaf:
            self.build_overleaf_bundle(ctx)

        manifest = {
            "run_dir": str(ctx.run_dir),
            "cache_root": str(ctx.cache_root),
            "overleaf_dir": str(ctx.overleaf_dir),
            "from_cache": ctx.from_cache,
            "artifacts": [asdict(artifact) for artifact in ctx.artifacts],
            "warnings": ctx.warnings,
            "config": ctx.config,
        }
        write_json(ctx.run_dir / "manifest.json", manifest)
        update_latest_link(path_from_config(self.config["outputs"]["root"]) / "latest", ctx.run_dir, ctx.warnings)
        if build_overleaf:
            update_latest_link(path_from_config(self.config["outputs"]["overleaf_root"]) / "latest", ctx.overleaf_dir, ctx.warnings)
        return ctx

    def render_full_eval(self, ctx: RunContext) -> None:
        import multisteve.compare_full_eval_agents as source

        family = "full_eval"
        cfg = ctx.config["families"][family]
        eval_root = Path(cfg.get("eval_root", Path(ctx.config["data_roots"]["final_evals"]) / "01_full_eval"))
        cache_dir = ctx.cache_root / family
        fig_dir = ctx.figures_dir / "final_eval" / "full"
        cache_dir.mkdir(parents=True, exist_ok=True)
        fig_dir.mkdir(parents=True, exist_ok=True)
        agg_path = cache_dir / "aggregated_prompt_means.csv"
        per_run_path = cache_dir / "per_run_metrics.csv"

        if ctx.from_cache and agg_path.exists():
            agg_df = pd.read_csv(agg_path)
        else:
            latest = source.collect_latest_runs(eval_root)
            per_run_df = source.build_per_run_df(latest)
            agg_df = source.build_agg_df(per_run_df)
            per_run_df.to_csv(per_run_path, index=False)
            agg_df.to_csv(agg_path, index=False)

        for modality in ("text", "visual"):
            for metric_key, metric_title, y_label in source.METRIC_SPECS:
                pdf_path = fig_dir / f"{modality}_{metric_key}.pdf"
                source.plot_metric(agg_df, modality, metric_key, metric_title, y_label, pdf_path)
                add_artifact(
                    ctx,
                    family,
                    f"final_eval/full/{modality}_{metric_key}",
                    pdf_path,
                    f"Full eval {modality} {metric_key} comparison.",
                )

    def render_partial_eval_40(self, ctx: RunContext) -> None:
        import multisteve.compare_partial_eval_40_models as source

        family = "partial_eval_40"
        cfg = ctx.config["families"][family]
        eval_root = Path(cfg.get("eval_root", Path(ctx.config["data_roots"]["final_evals"]) / "02_partial_eval_40_models"))
        cache_dir = ctx.cache_root / family
        fig_dir = ctx.figures_dir / "final_eval" / "partial_eval_40"
        cache_dir.mkdir(parents=True, exist_ok=True)
        fig_dir.mkdir(parents=True, exist_ok=True)
        agg_path = cache_dir / "aggregated_model_prompt_metrics.csv"
        per_run_path = cache_dir / "per_run_metrics.csv"

        if ctx.from_cache and agg_path.exists():
            agg_df = pd.read_csv(agg_path, dtype={"group_id": str})
        else:
            latest = source.collect_latest_runs(eval_root)
            per_run_df = source.build_per_run_df(latest)
            agg_df = source.build_agg_df(per_run_df)
            per_run_df.to_csv(per_run_path, index=False)
            agg_df.to_csv(agg_path, index=False)

        agg_df["group_id"] = agg_df["group_id"].astype(str).str.zfill(2)
        for group_id in sorted(agg_df["group_id"].unique().tolist()):
            group_df = agg_df[agg_df["group_id"] == group_id].copy()
            group_name = source.GROUP_NAMES.get(group_id, f"Group {group_id}")
            pdf_path = fig_dir / f"group_{group_id}_prompt_aligned.pdf"
            source.plot_group_prompt_aligned(group_df, group_name, pdf_path)
            add_artifact(
                ctx,
                family,
                f"final_eval/partial_eval_40/group_{group_id}_prompt_aligned",
                pdf_path,
                f"Partial eval 40-model prompt-aligned group {group_id}.",
            )

    def render_partial_eval_28(self, ctx: RunContext) -> None:
        import multisteve.compare_partial_eval_28_checkpoints as source

        family = "partial_eval_28_checkpoints"
        cfg = ctx.config["families"][family]
        eval_root = Path(
            cfg.get("eval_root", Path(ctx.config["data_roots"]["final_evals"]) / "03_partial_eval_28_checkpoints")
        )
        cache_dir = ctx.cache_root / family
        fig_dir = ctx.figures_dir / "final_eval" / "partial_eval_28_checkpoints"
        cache_dir.mkdir(parents=True, exist_ok=True)
        fig_dir.mkdir(parents=True, exist_ok=True)
        agg_path = cache_dir / "snapshot_prompt_reward_summary.csv"
        per_run_path = cache_dir / "per_run_prompt_rewards.csv"

        if ctx.from_cache and agg_path.exists():
            agg_df = pd.read_csv(agg_path)
        else:
            latest = source.collect_latest_runs(eval_root)
            per_run_df = source.build_per_run_df(latest)
            agg_df = source.build_agg_df(per_run_df)
            per_run_df.to_csv(per_run_path, index=False)
            agg_df.to_csv(agg_path, index=False)

        for spacing in ("equal", "linear", "log"):
            pdf_path = fig_dir / f"reward_evolution_{spacing}_steps.pdf"
            if spacing == "equal":
                pdf_path = fig_dir / "reward_evolution_equal_spacing.pdf"
            source.plot_reward_evolution(agg_df, spacing, pdf_path)
            add_artifact(
                ctx,
                family,
                f"final_eval/partial_eval_28_checkpoints/{pdf_path.stem}",
                pdf_path,
                f"Checkpoint reward evolution with {spacing} spacing.",
            )

    def render_tree_prompt_100(self, ctx: RunContext) -> None:
        import multisteve.eda_tree_prompt_models as source

        family = "tree_prompt_100"
        cfg = ctx.config["families"][family]
        eval_root = Path(
            cfg.get("eval_root", Path(ctx.config["data_roots"]["final_evals"]) / "04_tree_prompt_100_models")
        )
        cache_dir = ctx.cache_root / family
        fig_dir = ctx.figures_dir / "final_eval" / "tree_prompt_100"
        cache_dir.mkdir(parents=True, exist_ok=True)
        fig_dir.mkdir(parents=True, exist_ok=True)
        summary_path = cache_dir / "tree_logs_model_summary.csv"
        per_run_path = cache_dir / "tree_logs_per_run.csv"
        ranking_path = cache_dir / "tree_logs_model_ranking.csv"

        if ctx.from_cache and summary_path.exists():
            summary_df = pd.read_csv(summary_path)
        else:
            latest = source.collect_latest_runs(eval_root)
            per_run_df = source.build_per_run_df(latest)
            summary_df = source.build_model_summary_df(per_run_df)
            ranking_df = summary_df.sort_values(["logs_mean", "dataset_seed"], ascending=[False, True]).reset_index(drop=True)
            ranking_df.insert(0, "rank", np.arange(1, len(ranking_df) + 1))
            per_run_df.to_csv(per_run_path, index=False)
            summary_df.to_csv(summary_path, index=False)
            ranking_df.to_csv(ranking_path, index=False)

        outputs = [
            ("tree_logs_eda_overview", lambda path: source.plot_overview(summary_df, path)),
            (
                "tree_logs_model_means_in_order",
                lambda path: source.save_individual_plot(source.plot_model_means_in_order, summary_df, path),
            ),
            (
                "tree_logs_model_means_bar_vertical",
                lambda path: source.plot_model_means_bar_vertical(summary_df, path),
            ),
            (
                "tree_logs_model_means_bar_horizontal",
                lambda path: source.plot_model_means_bar_horizontal(summary_df, path),
            ),
            (
                "tree_logs_model_means_ranked",
                lambda path: source.save_individual_plot(source.plot_model_means_ranked, summary_df, path),
            ),
            (
                "tree_logs_model_mean_histogram",
                lambda path: source.save_individual_plot(source.plot_histogram, summary_df, path),
            ),
            (
                "tree_logs_model_mean_ecdf",
                lambda path: source.save_individual_plot(source.plot_ecdf, summary_df, path),
            ),
        ]
        for stem, render_fn in outputs:
            pdf_path = fig_dir / f"{stem}.pdf"
            render_fn(pdf_path)
            add_artifact(ctx, family, f"final_eval/tree_prompt_100/{stem}", pdf_path, f"Tree prompt 100-model {stem}.")

    def render_villager_passes(self, ctx: RunContext) -> None:
        family = "villager_passes"
        cfg = ctx.config["families"][family]
        source = load_villager_passes_module()
        eval_root = Path(cfg.get("eval_root", ctx.config["data_roots"]["villager"]))
        pass_start = int(cfg.get("pass_start", 1))
        pass_end = int(cfg.get("pass_end", 20))
        cache_dir = ctx.cache_root / family
        fig_dir = ctx.figures_dir / "villager" / "passes"
        cache_dir.mkdir(parents=True, exist_ok=True)
        fig_dir.mkdir(parents=True, exist_ok=True)
        summary_path = cache_dir / "villager_passes_summary.csv"
        per_run_path = cache_dir / "villager_passes_per_run.csv"

        if ctx.from_cache and summary_path.exists():
            summary_df = parse_list_columns(pd.read_csv(summary_path), ["raw_kill_values", "raw_hit_values"])
            aggregated = rows_by_setting(summary_df)
        else:
            records, warnings = source.collect_run_records(eval_root, pass_start, pass_end)
            ctx.warnings.extend(warnings[:50])
            per_run_df = pd.DataFrame([{k: str(v) if isinstance(v, Path) else v for k, v in r.__dict__.items()} for r in records])
            aggregated, _expected = source.aggregate_records(records)
            summary_df = pd.DataFrame([row for rows in aggregated.values() for row in rows])
            per_run_df.to_csv(per_run_path, index=False)
            summary_df.to_csv(summary_path, index=False)

        self._render_villager_passes_point_panels(
            source,
            aggregated,
            {
                "metric_key": "kill_rate_pct",
                "low_key": "kill_ci_low_pct",
                "high_key": "kill_ci_high_pct",
                "raw_key": "raw_kill_values",
                "title": "Villager Kill Rate",
                "axis_label": "Villager kill rate (%)",
                "limit": (0.0, 100.0),
                "show_raw_points": False,
            },
            fig_dir / "kill_rate_combined_points_ci.pdf",
            pass_start,
            pass_end,
        )
        add_artifact(
            ctx,
            family,
            "villager/passes/kill_rate_combined_points_ci",
            fig_dir / "kill_rate_combined_points_ci.pdf",
            "Villager pass kill-rate point plus CI summary.",
        )
        self._render_villager_passes_point_panels(
            source,
            aggregated,
            {
                "metric_key": "hit_mean",
                "low_key": "hit_ci_low",
                "high_key": "hit_ci_high",
                "raw_key": "raw_hit_values",
                "title": "Average Damage to the Villager",
                "axis_label": "Average damage per rollout",
                "limit": (0.0, 200.0),
                "show_raw_points": True,
            },
            fig_dir / "hit_sum_combined_points_ci.pdf",
            pass_start,
            pass_end,
        )
        add_artifact(
            ctx,
            family,
            "villager/passes/hit_sum_combined_points_ci",
            fig_dir / "hit_sum_combined_points_ci.pdf",
            "Villager pass hit-sum point plus CI summary.",
        )
        self._render_villager_passes_bar_panels(
            ctx,
            source,
            aggregated,
            fig_dir,
            pass_start,
            pass_end,
        )

        for logs_setting, prompt_slug in source.SETTING_ORDER:
            rows = aggregated.get((logs_setting, prompt_slug), [])
            if not rows:
                continue
            for metric_key, lower, upper, label, ylabel, stem in [
                ("kill_rate_pct", "kill_ci_low_pct", "kill_ci_high_pct", "Villager Kill Rate", "Villager kill rate (%)", "kill_rate"),
                ("hit_mean", "hit_ci_low", "hit_ci_high", "Hits Average", "Average damage dealt per episode", "hit_mean"),
            ]:
                pdf_path = fig_dir / f"{stem}_{'logs_true' if logs_setting.endswith('true') else 'logs_false'}_{prompt_slug}.pdf"
                raw_key = "raw_kill_values" if metric_key == "kill_rate_pct" else "raw_hit_values"
                self._render_villager_passes_single_bar_plot(
                    source,
                    rows,
                    (logs_setting, prompt_slug),
                    {
                        "metric_key": metric_key,
                        "low_key": lower,
                        "high_key": upper,
                        "raw_key": raw_key,
                        "title": label,
                        "axis_label": ylabel,
                        "limit": (0.0, 100.0) if metric_key == "kill_rate_pct" else (0.0, 200.0),
                        "show_raw_points": metric_key != "kill_rate_pct",
                    },
                    pdf_path,
                    pass_start,
                    pass_end,
                )
                add_artifact(
                    ctx,
                    family,
                    f"villager/passes/{pdf_path.stem}",
                    pdf_path,
                    f"Villager passes {label} for {logs_setting}/{prompt_slug}.",
                )

    def _render_villager_passes_single_bar_plot(
        self,
        source: Any,
        rows: list[dict[str, object]],
        setting: tuple[str, str],
        spec: dict[str, object],
        pdf_path: Path,
        pass_start: int,
        pass_end: int,
    ) -> None:
        fig, ax = plt.subplots(figsize=(NEURIPS_TEXT_WIDTH_IN, 2.6))
        self._plot_villager_passes_bar_panel(
            ax,
            source,
            setting,
            rows,
            spec,
            horizontal=False,
            jitter_seed=61_000 + source.SETTING_ORDER.index(setting),
        )
        ax.set_title(
            f"{spec['title']} | {self._villager_setting_label(source, setting)} | Passes {pass_start:02d}-{pass_end:02d}",
            fontsize=11.0,
        )
        fig.tight_layout(pad=0.2)
        save_figure(fig, pdf_path)

    def _render_villager_passes_point_panels(
        self,
        source: Any,
        aggregated: dict[tuple[str, str], list[dict[str, object]]],
        spec: dict[str, object],
        pdf_path: Path,
        pass_start: int,
        pass_end: int,
    ) -> None:
        fig, axes = plt.subplots(2, 2, figsize=(NEURIPS_TEXT_WIDTH_IN, 4.4), squeeze=False)
        for idx, setting in enumerate(source.SETTING_ORDER):
            ax = axes.ravel()[idx]
            rows = sorted(aggregated.get(setting, []), key=lambda row: int(row["model_seed"]))
            if not rows:
                ax.set_visible(False)
                continue
            positions = np.arange(len(rows), dtype=float)
            means = np.asarray([float(row[str(spec["metric_key"])]) for row in rows], dtype=float)
            lows = np.asarray([float(row[str(spec["low_key"])]) for row in rows], dtype=float)
            highs = np.asarray([float(row[str(spec["high_key"])]) for row in rows], dtype=float)
            err = clipped_interval_errors(means, lows, highs, tuple(spec["limit"]))
            if bool(spec.get("show_raw_points", True)):
                rng = np.random.default_rng(67_000 + idx)
                for pos, row in zip(positions, rows):
                    raw_values = np.asarray(list(row[str(spec["raw_key"])]), dtype=float)
                    jitter = rng.uniform(-0.18, 0.18, size=len(raw_values))
                    ax.scatter(
                        np.full(len(raw_values), pos) + jitter,
                        raw_values,
                        color=PASS_BAR_COLORWAYS[0]["dot"],
                        alpha=0.32,
                        s=3.5,
                        edgecolors="none",
                        zorder=2,
                    )
            ax.errorbar(
                positions,
                means,
                yerr=err,
                fmt="o",
                color="black",
                ecolor="black",
                elinewidth=0.45,
                capsize=1.5,
                markersize=2.2,
                zorder=3,
            )
            labels = [str(int(row["model_seed"])) for row in rows]
            tick_idx = self._agent_tick_indices(labels)
            ax.set_xticks(tick_idx)
            ax.set_xticklabels([labels[i] for i in tick_idx], rotation=45, ha="right", fontsize=7.0)
            ax.set_title(self._villager_setting_label(source, setting), fontsize=10.0)
            ax.set_ylabel(str(spec["axis_label"]), fontsize=10.0)
            ax.set_ylim(*tuple(spec["limit"]))
            ax.grid(axis="y", alpha=0.25)
        fig.suptitle(f"{spec['title']} Across Passes {pass_start:02d}-{pass_end:02d}", fontsize=11.0)
        fig.tight_layout(rect=(0, 0, 1, 0.96), pad=0.2)
        save_figure(fig, pdf_path)

    def _render_villager_passes_bar_panels(
        self,
        ctx: RunContext,
        source: Any,
        aggregated: dict[tuple[str, str], list[dict[str, object]]],
        fig_dir: Path,
        pass_start: int,
        pass_end: int,
    ) -> None:
        specs = [
            {
                "metric_key": "hit_mean",
                "low_key": "hit_ci_low",
                "high_key": "hit_ci_high",
                "raw_key": "raw_hit_values",
                "stem": "hit_mean_bar",
                "title": "Average Damage to the Villager",
                "axis_label": "Average damage per rollout",
                "limit": (0.0, 200.0),
                "show_raw_points": True,
            },
            {
                "metric_key": "kill_rate_pct",
                "low_key": "kill_ci_low_pct",
                "high_key": "kill_ci_high_pct",
                "raw_key": "raw_kill_values",
                "stem": "kill_rate_pct_bar",
                "title": "Villager Kill Rate",
                "axis_label": "Villager kill rate (%)",
                "limit": (0.0, 100.0),
                "show_raw_points": False,
            },
        ]

        colorway = PASS_BAR_COLORWAYS[0]
        for spec in specs:
            self._render_villager_passes_vertical_bar_stack(
                ctx,
                source,
                aggregated,
                fig_dir,
                pass_start,
                pass_end,
                spec,
                colorway=colorway,
            )
            self._render_villager_passes_horizontal_bar_stack(
                ctx,
                source,
                aggregated,
                fig_dir,
                pass_start,
                pass_end,
                spec,
                colorway=colorway,
            )

    def _render_villager_passes_vertical_bar_stack(
        self,
        ctx: RunContext,
        source: Any,
        aggregated: dict[tuple[str, str], list[dict[str, object]]],
        fig_dir: Path,
        pass_start: int,
        pass_end: int,
        spec: dict[str, object],
        *,
        colorway: dict[str, str],
        variant_suffix: str = "",
    ) -> None:
        fig, axes = plt.subplots(4, 1, figsize=(NEURIPS_TEXT_WIDTH_IN, 9.2), squeeze=False)
        axes_flat = axes.ravel()
        for idx, setting in enumerate(source.SETTING_ORDER):
            rows = aggregated.get(setting, [])
            self._plot_villager_passes_bar_panel(
                axes_flat[idx],
                source,
                setting,
                rows,
                spec,
                horizontal=False,
                jitter_seed=53_000 + idx,
                colorway=colorway,
            )
        fig.suptitle(str(spec["title"]), fontsize=11.0, y=0.986)
        fig.tight_layout(rect=(0, 0, 1, 0.966), h_pad=0.25)
        pdf_path = fig_dir / f"{spec['stem']}_settings_stacked_vertical{variant_suffix}.pdf"
        save_figure(fig, pdf_path)
        add_artifact(
            ctx,
            "villager_passes",
            f"villager/passes/{pdf_path.stem}",
            pdf_path,
            f"Villager passes 100-agent {spec['title']} vertical bar panels ({colorway['label']}).",
        )

    def _render_villager_passes_horizontal_bar_stack(
        self,
        ctx: RunContext,
        source: Any,
        aggregated: dict[tuple[str, str], list[dict[str, object]]],
        fig_dir: Path,
        pass_start: int,
        pass_end: int,
        spec: dict[str, object],
        *,
        colorway: dict[str, str],
        variant_suffix: str = "",
        explicit_path: Path | None = None,
    ) -> None:
        fig, axes = plt.subplots(1, 4, figsize=(NEURIPS_TEXT_WIDTH_IN, 8.0), squeeze=False)
        axes_flat = axes.ravel()
        for idx, setting in enumerate(source.SETTING_ORDER):
            rows = aggregated.get(setting, [])
            self._plot_villager_passes_bar_panel(
                axes_flat[idx],
                source,
                setting,
                rows,
                spec,
                horizontal=True,
                jitter_seed=57_000 + idx,
                colorway=colorway,
            )
            axes_flat[idx].set_xlabel("")
        fig.suptitle(str(spec["title"]), fontsize=11.0, y=0.976)
        fig.tight_layout(rect=(0, 0.045, 1, 0.952), w_pad=0.32)
        fig.text(0.5, 0.035, str(spec["axis_label"]), ha="center", va="center", fontsize=10.0)
        pdf_path = explicit_path or fig_dir / f"{spec['stem']}_settings_stacked_side_by_side{variant_suffix}.pdf"
        save_figure(fig, pdf_path)
        add_artifact(
            ctx,
            "villager_passes",
            f"villager/passes/{pdf_path.stem}",
            pdf_path,
            f"Villager passes 100-agent {spec['title']} side-by-side horizontal bar panels ({colorway['label']}).",
        )

    def _plot_villager_passes_bar_panel(
        self,
        ax: plt.Axes,
        source: Any,
        setting: tuple[str, str],
        rows: list[dict[str, object]],
        spec: dict[str, object],
        *,
        horizontal: bool,
        jitter_seed: int,
        colorway: dict[str, str] | None = None,
    ) -> None:
        if not rows:
            ax.set_visible(False)
            return

        metric_key = str(spec["metric_key"])
        low_key = str(spec["low_key"])
        high_key = str(spec["high_key"])
        raw_key = str(spec["raw_key"])
        limit = tuple(spec["limit"])
        rows = sorted(rows, key=lambda row: int(row["model_seed"]))
        labels = [str(int(row["model_seed"])) for row in rows]
        means = np.asarray([float(row[metric_key]) for row in rows], dtype=float)
        lows = np.asarray([float(row[low_key]) for row in rows], dtype=float)
        highs = np.asarray([float(row[high_key]) for row in rows], dtype=float)
        err = clipped_interval_errors(means, lows, highs, limit)
        positions = np.arange(len(rows), dtype=float)
        colorway = colorway or PASS_BAR_COLORWAYS[0]
        color = colorway["bar"]
        raw_point_color = colorway["dot"]
        show_raw_points = bool(spec.get("show_raw_points", True))
        rng = np.random.default_rng(jitter_seed)

        if horizontal:
            ax.barh(positions, means, color=color, edgecolor="black", linewidth=0.2, alpha=0.88, height=0.78)
            if show_raw_points:
                for pos, row in zip(positions, rows):
                    raw_values = np.asarray(list(row[raw_key]), dtype=float)
                    jitter = rng.uniform(-0.18, 0.18, size=len(raw_values))
                    ax.scatter(raw_values, np.full(len(raw_values), pos) + jitter, color=raw_point_color, alpha=0.32, s=4, edgecolors="none", zorder=4)
            ax.errorbar(
                means,
                positions,
                xerr=err,
                fmt="none",
                color="black",
                ecolor="black",
                elinewidth=0.42,
                capsize=1.2,
                capthick=0.42,
                zorder=5,
            )
            tick_idx = self._agent_tick_indices(labels)
            ax.set_yticks(tick_idx)
            ax.set_yticklabels([labels[i] for i in tick_idx], fontsize=8.0)
            ax.set_xlabel(str(spec["axis_label"]), fontsize=10.0)
            ax.set_xlim(*limit)
            ax.set_ylim(len(rows) - 0.4, -0.6)
            ax.margins(x=0.01, y=0.0)
        else:
            ax.bar(positions, means, color=color, edgecolor="black", linewidth=0.2, alpha=0.88, width=0.78)
            if show_raw_points:
                for pos, row in zip(positions, rows):
                    raw_values = np.asarray(list(row[raw_key]), dtype=float)
                    jitter = rng.uniform(-0.18, 0.18, size=len(raw_values))
                    ax.scatter(np.full(len(raw_values), pos) + jitter, raw_values, color=raw_point_color, alpha=0.32, s=4, edgecolors="none", zorder=4)
            ax.errorbar(
                positions,
                means,
                yerr=err,
                fmt="none",
                color="black",
                ecolor="black",
                elinewidth=0.42,
                capsize=1.2,
                capthick=0.42,
                zorder=5,
            )
            tick_idx = self._agent_tick_indices(labels)
            ax.set_xticks(tick_idx)
            ax.set_xticklabels([labels[i] for i in tick_idx], rotation=45, ha="right", fontsize=8.0)
            ax.set_ylabel(str(spec["axis_label"]), fontsize=10.0)
            ax.set_ylim(*limit)
            ax.set_xlim(-0.6, len(rows) - 0.4)
            ax.margins(x=0.0, y=0.01)

        ax.set_title(self._villager_setting_label(source, setting), fontsize=9.6, pad=1.5)
        ax.tick_params(axis="both", which="major", pad=1.5)
        ax.grid(axis="x" if horizontal else "y", alpha=0.25)

    def _sparse_tick_indices(self, n: int, *, max_ticks: int) -> list[int]:
        if n <= max_ticks:
            return list(range(n))
        return sorted(set(int(x) for x in np.linspace(0, n - 1, max_ticks)))

    def _agent_tick_indices(self, labels: list[str]) -> list[int]:
        tick_indices = []
        for idx, label in enumerate(labels):
            try:
                value = int(label)
            except ValueError:
                continue
            if value == 101 or (value >= 105 and value <= 200 and value % 5 == 0):
                tick_indices.append(idx)
        return tick_indices or self._sparse_tick_indices(len(labels), max_ticks=10)

    def _villager_setting_label(self, source: Any, setting: tuple[str, str]) -> str:
        logs_setting, prompt_slug = setting
        logs = "Logs above" if logs_setting.endswith("true") else "No logs above"
        prompt = "long prompt" if prompt_slug == "wood-engineered" else "short prompt"
        return f"{logs},\n{prompt}"

    def _setting_color(self, setting: tuple[str, str]) -> str:
        color_map = {
            ("logs_above_villager_false", "wood"): PALETTE["blue"],
            ("logs_above_villager_false", "wood-engineered"): PALETTE["orange"],
            ("logs_above_villager_true", "wood"): PALETTE["green"],
            ("logs_above_villager_true", "wood-engineered"): PALETTE["purple"],
        }
        return color_map.get(setting, PALETTE["gray"])

    def render_tree_vs_villager(self, ctx: RunContext) -> None:
        family = "tree_vs_villager"
        cfg = ctx.config["families"][family]
        source = load_tree_vs_villager_module()
        villager_root = Path(cfg.get("villager_eval_root", ctx.config["data_roots"]["villager"]))
        tree_root = Path(
            cfg.get("tree_eval_root", Path(ctx.config["data_roots"]["final_evals"]) / "04_tree_prompt_100_models")
        )
        pass_start = int(cfg.get("pass_start", 1))
        pass_end = int(cfg.get("pass_end", 20))
        cache_dir = ctx.cache_root / family
        fig_dir = ctx.figures_dir / "villager" / "tree_vs_villager"
        cache_dir.mkdir(parents=True, exist_ok=True)
        fig_dir.mkdir(parents=True, exist_ok=True)
        joined_path = cache_dir / "tree_vs_villager_joined.csv"
        per_pass_path = cache_dir / "villager_per_pass_metrics.csv"

        if ctx.from_cache and joined_path.exists():
            merged = pd.read_csv(joined_path)
            villager_per_pass = pd.read_csv(per_pass_path) if per_pass_path.exists() else None
        else:
            villager_per_pass = source.collect_villager_per_pass(villager_root, pass_start, pass_end)
            villager_summary = source.aggregate_villager(villager_per_pass)
            tree_per_run = source.collect_tree_per_run(tree_root)
            tree_summary = source.aggregate_tree(tree_per_run)
            merged = villager_summary.merge(tree_summary, on="model_seed", how="inner", suffixes=("_villager", "_tree"))
            merged = merged.sort_values(["model_seed", "logs_setting", "prompt_slug"]).reset_index(drop=True)
            villager_per_pass.to_csv(cache_dir / "villager_per_pass_metrics.csv", index=False)
            villager_summary.to_csv(cache_dir / "villager_model_setting_summary.csv", index=False)
            tree_per_run.to_csv(cache_dir / "tree_logs_per_run.csv", index=False)
            tree_summary.to_csv(cache_dir / "tree_logs_model_summary.csv", index=False)
            merged.to_csv(joined_path, index=False)

        for y_key, y_label, stem, title in [
            (
                "kill_rate_pct",
                f"Villager kill rate across passes {pass_start:02d}-{pass_end:02d} (%)",
                "kill_rate_vs_tree_logs_scatter",
                "Villager Kill Rate vs Tree Chopping Competence",
            ),
            (
                "hit_sum_total",
                f"Total damage dealt across passes {pass_start:02d}-{pass_end:02d}",
                "hit_damage_vs_tree_logs_scatter",
                "Villager Hit Damage vs Tree Chopping Competence",
            ),
        ]:
            for ci_suffix, show_ci in [("no_ci", False), ("with_ci", True)]:
                pdf_path = fig_dir / f"all_settings_{stem}_{ci_suffix}.pdf"
                self._plot_tree_vs_villager_scatter(merged, source, y_key, y_label, title, pdf_path, show_ci=show_ci)
                add_artifact(ctx, family, f"villager/tree_vs_villager/all_settings_{stem}_{ci_suffix}", pdf_path, title)
            legacy_path = fig_dir / f"all_settings_{stem}.pdf"
            self._plot_tree_vs_villager_scatter(merged, source, y_key, y_label, title, legacy_path, show_ci=False)
            add_artifact(ctx, family, f"villager/tree_vs_villager/all_settings_{stem}", legacy_path, title)

        logs_setting = cfg.get("filtered_logs_setting")
        prompt_slug = cfg.get("filtered_prompt_slug")
        if logs_setting and prompt_slug:
            filtered = merged[(merged["logs_setting"] == logs_setting) & (merged["prompt_slug"] == prompt_slug)].copy()
            if not filtered.empty:
                filter_label = source.SETTING_LABELS[(logs_setting, prompt_slug)]
                for y_key, y_label, stem, title in [
                    (
                        "kill_rate_pct",
                        f"Villager kill rate across passes {pass_start:02d}-{pass_end:02d} (%)",
                        "kill_rate_vs_tree_logs_scatter",
                        f"Villager Kill Rate vs Tree Chopping Competence | {filter_label}",
                    ),
                    (
                        "hit_sum_total",
                        f"Total damage dealt across passes {pass_start:02d}-{pass_end:02d}",
                        "hit_damage_vs_tree_logs_scatter",
                        f"Villager Hit Damage vs Tree Chopping Competence | {filter_label}",
                    ),
                ]:
                    for ci_suffix, show_ci in [("no_ci", False), ("with_ci", True)]:
                        pdf_path = fig_dir / f"filtered_{stem}_{ci_suffix}.pdf"
                        self._plot_tree_vs_villager_scatter(filtered, source, y_key, y_label, title, pdf_path, show_ci=show_ci)
                        add_artifact(ctx, family, f"villager/tree_vs_villager/filtered_{stem}_{ci_suffix}", pdf_path, title)
                    legacy_path = fig_dir / f"filtered_{stem}.pdf"
                    self._plot_tree_vs_villager_scatter(filtered, source, y_key, y_label, title, legacy_path, show_ci=False)
                    add_artifact(ctx, family, f"villager/tree_vs_villager/filtered_{stem}", legacy_path, title)

    def _plot_tree_vs_villager_scatter(
        self,
        df: pd.DataFrame,
        source: Any,
        y_key: str,
        y_label: str,
        title: str,
        output_path: Path,
        *,
        show_ci: bool,
    ) -> None:
        fig, ax = plt.subplots(figsize=(SIDE_BY_SIDE_FIGURE_WIDTH_IN, 2.55))
        for setting in source.SETTING_ORDER:
            logs_setting, prompt_slug = setting
            subset = df[(df["logs_setting"] == logs_setting) & (df["prompt_slug"] == prompt_slug)].copy()
            if subset.empty:
                continue
            color = self._setting_color(setting)
            marker = source.SETTING_MARKERS[setting]
            x = subset["tree_logs_mean"].to_numpy(dtype=float)
            y = subset[y_key].to_numpy(dtype=float)
            if show_ci:
                x_ci = subset["tree_logs_ci95"].to_numpy(dtype=float)
                xerr = clipped_interval_errors(x, x - x_ci, x + x_ci, (0.0, None))
                if y_key == "kill_rate_pct":
                    y_low = subset["kill_ci_low_pct"].to_numpy(dtype=float)
                    y_high = subset["kill_ci_high_pct"].to_numpy(dtype=float)
                    y_limits: tuple[float | None, float | None] = (0.0, 100.0)
                else:
                    y_low = subset["hit_sum_ci_low"].to_numpy(dtype=float)
                    y_high = subset["hit_sum_ci_high"].to_numpy(dtype=float)
                    y_limits = (0.0, None)
                yerr = clipped_interval_errors(y, y_low, y_high, y_limits)
                ax.errorbar(
                    x,
                    y,
                    xerr=xerr,
                    yerr=yerr,
                    fmt="none",
                    ecolor=color,
                    elinewidth=0.45,
                    alpha=0.23,
                    zorder=1,
                )
            ax.scatter(
                x,
                y,
                marker=marker,
                s=22,
                alpha=0.88,
                color=color,
                edgecolors="white",
                linewidths=0.4,
                label=self._compact_villager_setting_label(setting),
                zorder=2,
            )
            if len(subset) >= 2 and np.unique(x).size >= 2:
                coeffs = np.polyfit(x, y, 1)
                x_line = np.linspace(float(np.min(x)), float(np.max(x)), 100)
                ax.plot(x_line, coeffs[0] * x_line + coeffs[1], color=color, linewidth=0.8, alpha=0.85)

        ax.set_title(title, fontsize=11.0, pad=3)
        ax.set_xlabel("Mean max tree logs", fontsize=10.0)
        ax.set_ylabel(y_label, fontsize=10.0)
        ax.grid(alpha=0.25)
        ax.legend(loc="best", fontsize=7.0, frameon=False, handletextpad=0.25, borderaxespad=0.15)
        fig.tight_layout(pad=0.15)
        save_figure(fig, output_path, bbox_inches=None)

    def _compact_villager_setting_label(self, setting: tuple[str, str]) -> str:
        logs_setting, prompt_slug = setting
        logs = "logs" if logs_setting.endswith("true") else "no logs"
        prompt = "long" if prompt_slug == "wood-engineered" else "short"
        return f"{logs}, {prompt}"

    def _render_tree_vs_villager_setting_panels(
        self,
        ctx: RunContext,
        source: Any,
        merged: pd.DataFrame,
        villager_per_pass: pd.DataFrame | None,
        fig_dir: Path,
        y_key: str,
        y_label: str,
        pass_start: int,
        pass_end: int,
        stem: str,
    ) -> None:
        if villager_per_pass is None or villager_per_pass.empty:
            return

        if y_key == "kill_rate_pct":
            metric_label = f"Villager kill rate across passes {pass_start:02d}-{pass_end:02d} (%)"
            raw_key = "killed"
        else:
            metric_label = f"Total damage dealt across passes {pass_start:02d}-{pass_end:02d}"
            raw_key = "hit_sum"

        layouts = [
            ("settings_stacked_vertical", (4, 1), (7.6, 14.5)),
            ("settings_stacked_side_by_side", (1, 4), (18.0, 4.5)),
        ]
        for layout_name, (nrows, ncols), fig_size in layouts:
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=fig_size, squeeze=False)
            axes_flat = axes.ravel()
            for idx, setting in enumerate(source.SETTING_ORDER):
                logs_setting, prompt_slug = setting
                setting_subset = merged[
                    (merged["logs_setting"] == logs_setting) & (merged["prompt_slug"] == prompt_slug)
                ].copy()
                raw_subset = villager_per_pass[
                    (villager_per_pass["logs_setting"] == logs_setting)
                    & (villager_per_pass["prompt_slug"] == prompt_slug)
                ].copy()
                self._plot_tree_vs_villager_single_panel(
                    axes_flat[idx],
                    setting_subset,
                    raw_subset,
                    y_key,
                    raw_key,
                    metric_label,
                )
                axes_flat[idx].set_title(source.SETTING_LABELS[(logs_setting, prompt_slug)], fontsize=9.2)

            for axis in axes_flat[len(source.SETTING_ORDER) :]:
                axis.set_visible(False)

            if nrows == 4:
                for axis_idx, axis in enumerate(axes_flat[:4]):
                    axis.set_xlabel("Tree chopping competence: mean max logs")
                    if axis_idx == 0:
                        axis.set_ylabel(y_label)
            else:
                for axis_idx, axis in enumerate(axes_flat[:4]):
                    axis.set_title(source.SETTING_LABELS[source.SETTING_ORDER[axis_idx]], fontsize=9.2)
                    axis.set_xlabel("Tree chopping competence: mean max logs")
                    axis.set_ylabel(y_label)
                    axis.tick_params(axis="x", labelrotation=20)
            fig.tight_layout(h_pad=1.0, w_pad=0.8)
            pdf_path = fig_dir / f"all_settings_{stem}_{layout_name}.pdf"
            save_figure(fig, pdf_path)
            add_artifact(
                ctx,
                "tree_vs_villager",
                f"villager/tree_vs_villager/all_settings_{stem}_{layout_name}",
                pdf_path,
                f"Tree chopping competence vs {metric_label.lower()} ({layout_name.replace('_', ' ')}).",
            )

    def _plot_tree_vs_villager_single_panel(
        self,
        ax: plt.Axes,
        setting_subset: pd.DataFrame,
        raw_subset: pd.DataFrame,
        y_key: str,
        raw_key: str,
        y_label: str,
    ) -> None:
        if setting_subset.empty:
            return

        x = setting_subset["tree_logs_mean"].to_numpy(dtype=float)
        y = setting_subset[y_key].to_numpy(dtype=float)
        ax.scatter(
            x,
            y,
            marker="o",
            s=38,
            alpha=0.88,
            color="black",
            edgecolors="white",
            linewidths=0.7,
        )

        if len(x) >= 2 and np.unique(x).size >= 2:
            coeffs = np.polyfit(x, y, 1)
            x_line = np.linspace(float(np.min(x)), float(np.max(x)), 200)
            y_line = coeffs[0] * x_line + coeffs[1]
            ax.plot(x_line, y_line, color=PALETTE["gray"], linewidth=1.1, alpha=0.75)

        for model_seed, x_val, y_val in zip(
            setting_subset["model_seed"].astype(int).tolist(),
            setting_subset["tree_logs_mean"].astype(float).tolist(),
            setting_subset[y_key].astype(float).tolist(),
        ):
            ax.text(
                x_val,
                y_val,
                str(model_seed),
                color="white",
                fontsize=6,
                ha="center",
                va="center",
                zorder=5,
            )

        for model_seed in setting_subset["model_seed"].astype(int).tolist():
            model_raw = raw_subset[raw_subset["model_seed"] == model_seed]
            if model_raw.empty:
                continue
            x_val = float(setting_subset.loc[setting_subset["model_seed"] == model_seed, "tree_logs_mean"].iloc[0])
            raw_values = model_raw[raw_key].astype(float).to_numpy()
            if raw_key == "killed":
                raw_values = raw_values * 100.0
            jitter = np.random.default_rng(31_000 + model_seed).normal(0.0, 0.18, size=len(raw_values))
            ax.scatter(
                np.full(len(raw_values), x_val, dtype=float) + jitter,
                raw_values,
                color=PASS_BAR_COLORWAYS[0]["dot"],
                alpha=0.25,
                s=9,
                edgecolors="none",
            )

        ax.set_ylabel(y_label)

    def render_villager_chop_40(self, ctx: RunContext) -> None:
        family = "villager_chop_40"
        cfg = ctx.config["families"][family]
        source = load_villager_40_module()
        if hasattr(source, "SETTING_COLORS"):
            source.SETTING_COLORS.update({setting: self._setting_color(setting) for setting in source.SETTING_ORDER})
        if hasattr(source, "RAW_POINT_ALPHA"):
            source.RAW_POINT_ALPHA = 0.32
        eval_root = Path(
            cfg.get("eval_root", EVALS_VILLAGER_ROOT / "villager_chop_40_models_01_04")
        )
        pass_start = int(cfg.get("pass_start", 1))
        pass_end = int(cfg.get("pass_end", 20))
        cache_dir = ctx.cache_root / family
        fig_dir = ctx.figures_dir / "villager" / "chop_40_models"
        cache_dir.mkdir(parents=True, exist_ok=True)
        fig_dir.mkdir(parents=True, exist_ok=True)
        summary_path = cache_dir / "villager_chop_40_models_summary.csv"

        if ctx.from_cache and summary_path.exists():
            summary_df = parse_list_columns(pd.read_csv(summary_path, dtype={"group_id": str}), ["kill_raw_values_pct", "hit_raw_values"])
        else:
            per_pass_df = source.collect_run_records(eval_root, pass_start, pass_end)
            summary_df = source.aggregate_records(per_pass_df)
            per_pass_df.to_csv(cache_dir / "villager_chop_40_models_per_pass.csv", index=False)
            summary_df.to_csv(summary_path, index=False)

        summary_df["group_id"] = summary_df["group_id"].astype(str).str.zfill(2)
        for group_id in sorted(summary_df["group_id"].unique().tolist()):
            group_df = summary_df[summary_df["group_id"] == group_id].copy()
            group_dir = fig_dir / f"group_{group_id}"
            group_dir.mkdir(parents=True, exist_ok=True)
            for metric_key in ("kill_rate_pct", "hit_mean"):
                pdf_path = group_dir / f"{metric_key}_combined.pdf"
                source.make_group_combined_plot(group_df, metric_key, pdf_path)
                add_artifact(
                    ctx,
                    family,
                    f"villager/chop_40_models/group_{group_id}/{metric_key}_combined",
                    pdf_path,
                    f"Villager 40-model group {group_id} {metric_key} combined plot.",
                )
                self._render_villager_40_split_plots(ctx, source, group_df, group_dir, group_id, metric_key)
                self._render_villager_40_panel_plots(
                    ctx,
                    source,
                    group_df,
                    group_dir,
                    group_id,
                    metric_key,
                )

    def _render_villager_40_split_plots(
        self,
        ctx: RunContext,
        source: Any,
        group_df: pd.DataFrame,
        group_dir: Path,
        group_id: str,
        metric_key: str,
    ) -> None:
        spec = source.metric_spec(metric_key)
        point_legend = Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor="red",
            markeredgecolor="none",
            alpha=source.RAW_POINT_ALPHA,
            markersize=7,
            label="Per-pass data point",
        )
        mean_legend = Line2D([0], [0], marker="o", color="black", linestyle="None", markersize=4, label="Mean +/- 95% CI")
        for panel_idx, setting in enumerate(source.SETTING_ORDER):
            logs_setting, prompt_slug = setting
            subset = group_df[(group_df["logs_setting"] == logs_setting) & (group_df["prompt_slug"] == prompt_slug)].copy()
            if subset.empty:
                continue
            fig, ax = plt.subplots(figsize=(5.8, 4.6))
            source.plot_setting_with_raw_points(
                ax,
                subset,
                metric_key,
                spec["low_key"],
                spec["high_key"],
                spec["raw_values_key"],
                setting,
                (21_000 if metric_key == "kill_rate_pct" else 31_000) + panel_idx,
            )
            ax.set_ylabel(spec["y_label"])
            ax.legend(handles=[point_legend, mean_legend], loc="upper right", frameon=False)
            fig.suptitle(f"{group_df['group_label'].iloc[0]} | {source.SETTING_LABELS[setting]} | {spec['title']}", fontsize=11)
            fig.tight_layout(rect=(0, 0, 1, 0.95))
            pdf_path = group_dir / f"{metric_key}_{logs_setting}_{prompt_slug}.pdf"
            pdf_path.parent.mkdir(parents=True, exist_ok=True)
            save_figure(fig, pdf_path)
            add_artifact(
                ctx,
                "villager_chop_40",
                f"villager/chop_40_models/group_{group_id}/{pdf_path.stem}",
                pdf_path,
                f"Villager 40-model group {group_id} split {metric_key} plot.",
            )

    def _render_villager_40_panel_plots(
        self,
        ctx: RunContext,
        source: Any,
        group_df: pd.DataFrame,
        group_dir: Path,
        group_id: str,
        metric_key: str,
    ) -> None:
        spec = source.metric_spec(metric_key)
        for layout_name, nrows, ncols, fig_size in [
            ("settings_stacked_vertical", 4, 1, (7.6, 14.5)),
            ("settings_stacked_side_by_side", 1, 4, (18.0, 4.5)),
        ]:
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=fig_size, squeeze=False)
            axes_flat = axes.ravel()
            for panel_idx, setting in enumerate(source.SETTING_ORDER):
                logs_setting, prompt_slug = setting
                subset = group_df[
                    (group_df["logs_setting"] == logs_setting) & (group_df["prompt_slug"] == prompt_slug)
                ].copy()
                if subset.empty:
                    continue
                source.plot_setting_with_raw_points(
                    axes_flat[panel_idx],
                    subset,
                    metric_key,
                    spec["low_key"],
                    spec["high_key"],
                    spec["raw_values_key"],
                    setting,
                    (39_000 if metric_key == "kill_rate_pct" else 49_000) + panel_idx,
                )
                axes_flat[panel_idx].set_title(source.SETTING_LABELS[setting], fontsize=9.6)
                axes_flat[panel_idx].set_ylabel(spec["y_label"] if panel_idx == 0 or ncols == 1 else "")
            for axis in axes_flat[len(source.SETTING_ORDER) :]:
                axis.set_visible(False)
            fig.tight_layout(h_pad=1.0, w_pad=0.9)
            fig.suptitle(
                f"{group_df['group_label'].iloc[0]} | {spec['title']} | {layout_name.replace('_', ' ')}",
                fontsize=10.5,
            )
            pdf_path = group_dir / f"{metric_key}_{layout_name}.pdf"
            save_figure(fig, pdf_path)
            add_artifact(
                ctx,
                "villager_chop_40",
                f"villager/chop_40_models/group_{group_id}/{metric_key}_{layout_name}",
                pdf_path,
                f"Villager 40-model group {group_id} {metric_key} {layout_name.replace('_', ' ')}.",
            )

    def render_villager_bitflip(self, ctx: RunContext) -> None:
        family = "villager_bitflip"
        cfg = ctx.config["families"][family]
        source = load_bitflip_module()
        eval_root = Path(
            cfg.get("eval_root", EVALS_VILLAGER_ROOT / "villager_chop_bitflip_10_wood_engineered")
        )
        pass_start = int(cfg.get("pass_start", 1))
        pass_end = int(cfg.get("pass_end", 20))
        cache_dir = ctx.cache_root / family
        fig_dir = ctx.figures_dir / "villager" / "bitflip_10_wood_engineered"
        cache_dir.mkdir(parents=True, exist_ok=True)
        fig_dir.mkdir(parents=True, exist_ok=True)
        summary_path = cache_dir / "villager_chop_bitflip_10_wood_engineered_summary.csv"

        if ctx.from_cache and summary_path.exists():
            summary_df = parse_list_columns(pd.read_csv(summary_path), ["kill_raw_values_pct", "hit_raw_values"])
        else:
            per_pass_df = source.collect_run_records(eval_root, pass_start, pass_end)
            summary_df = source.aggregate_records(per_pass_df)
            per_pass_df.to_csv(cache_dir / "villager_chop_bitflip_10_wood_engineered_per_pass.csv", index=False)
            summary_df.to_csv(summary_path, index=False)

        for metric_key in ("kill_rate_pct", "hit_mean"):
            pdf_path = fig_dir / f"{metric_key}_combined.pdf"
            source.make_combined_plot(summary_df, metric_key, pdf_path)
            add_artifact(
                ctx,
                family,
                f"villager/bitflip_10_wood_engineered/{metric_key}_combined",
                pdf_path,
                f"Villager bitflip wood-engineered {metric_key} combined plot.",
            )

    def render_agent_111_114_seed_sweep(self, ctx: RunContext) -> None:
        family = "agent_111_114_seed_sweep"
        cfg = ctx.config["families"][family]
        source = load_agent_111_114_seed_sweep_module()
        sweep_root = Path(cfg.get("sweep_root", EVALS_VILLAGER_ROOT / "agent_111_114_short_logs_above_true_seed_sweep"))
        cache_dir = ctx.cache_root / family
        fig_dir = ctx.figures_dir / "villager" / "agent_111_114_seed_sweep"
        cache_dir.mkdir(parents=True, exist_ok=True)
        fig_dir.mkdir(parents=True, exist_ok=True)
        summary_path = cache_dir / "agent_111_114_summary_by_seed.csv"

        if ctx.from_cache and summary_path.exists():
            summary_df = parse_list_columns(pd.read_csv(summary_path), ["raw_kill_values", "raw_hit_values"])
            aggregated = self._agent_111_114_rows_by_seed(summary_df)
        else:
            records = source.collect_records(sweep_root)
            aggregated = source.aggregate(records)
            summary_df = pd.DataFrame([row for rows in aggregated.values() for row in rows])
            summary_df.to_csv(summary_path, index=False)

        pdf_path = fig_dir / "agent_111_114_seed_sweep_side_facet_bars.pdf"
        self._plot_agent_111_114_seed_sweep(aggregated, pdf_path)
        add_artifact(
            ctx,
            family,
            "villager/agent_111_114_seed_sweep/agent_111_114_seed_sweep_side_facet_bars",
            pdf_path,
            "Agent 111 vs Agent 114 villager chop damage and kill-rate comparison across environment seeds.",
        )

    def _agent_111_114_rows_by_seed(self, df: pd.DataFrame) -> dict[str, list[dict[str, object]]]:
        out: dict[str, list[dict[str, object]]] = {}
        for seed_label, group in df.groupby("seed_label", sort=False):
            out[str(seed_label)] = group.to_dict(orient="records")
        return out

    def _plot_agent_111_114_seed_sweep(self, aggregated: dict[str, list[dict[str, object]]], output_path: Path) -> None:
        if not aggregated:
            return
        seeds = sorted(
            [int(str(seed_label).split("_")[-1]) for seed_label in aggregated],
        )
        rows_by_key: dict[tuple[int, str, str], dict[str, object]] = {}
        for seed_label, rows in aggregated.items():
            seed = int(str(seed_label).split("_")[-1])
            for row in rows:
                rows_by_key[(seed, str(row["agent_key"]), str(row["side_key"]))] = row

        metrics = [
            ("kill_rate_pct", "kill_ci_low_pct", "kill_ci_high_pct", "raw_kill_values", "Villager Kill Rate", "Kill rate (%)", (0.0, 100.0), False),
            ("hit_mean", "hit_ci_low", "hit_ci_high", "raw_hit_values", "Average Damage to the Villager", "Average damage per rollout", (0.0, 200.0), True),
        ]
        sides = [("01_left", "Villager tree on the left"), ("02_right", "Villager tree on the right")]
        agents = [("agent_111", "Agent 111"), ("agent_114", "Agent 114")]
        agent_colors = {"agent_111": PALETTE["blue"], "agent_114": PALETTE["green"]}
        raw_point_color = PASS_BAR_COLORWAYS[0]["dot"]
        x = np.arange(len(seeds), dtype=float)
        width = 0.34
        offsets = {"agent_111": -width / 2.0, "agent_114": width / 2.0}

        fig, axes = plt.subplots(2, 2, figsize=(NEURIPS_TEXT_WIDTH_IN, 4.5), sharex=True, squeeze=False)
        for row_idx, (metric_key, low_key, high_key, raw_key, _title, y_label, ylimit, show_raw) in enumerate(metrics):
            for col_idx, (side_key, side_label) in enumerate(sides):
                ax = axes[row_idx, col_idx]
                for agent_key, _agent_label in agents:
                    xpos = x + offsets[agent_key]
                    means = []
                    interval_lows = []
                    interval_highs = []
                    for seed in seeds:
                        row = rows_by_key.get((seed, agent_key, side_key))
                        if row is None:
                            means.append(np.nan)
                            interval_lows.append(np.nan)
                            interval_highs.append(np.nan)
                            continue
                        value = float(row[metric_key])
                        means.append(value)
                        interval_lows.append(float(row[low_key]))
                        interval_highs.append(float(row[high_key]))
                    means_arr = np.asarray(means, dtype=float)
                    valid = np.isfinite(means_arr)
                    ax.bar(
                        xpos[valid],
                        means_arr[valid],
                        width=width,
                        color=agent_colors[agent_key],
                        edgecolor="black",
                        linewidth=0.22,
                        alpha=0.9,
                        label=_agent_label if row_idx == 0 and col_idx == 0 else None,
                        zorder=2,
                    )
                    err = clipped_interval_errors(
                        means_arr[valid],
                        np.asarray(interval_lows, dtype=float)[valid],
                        np.asarray(interval_highs, dtype=float)[valid],
                        ylimit,
                    )
                    ax.errorbar(
                        xpos[valid],
                        means_arr[valid],
                        yerr=err,
                        fmt="none",
                        ecolor="black",
                        elinewidth=0.45,
                        capsize=1.3,
                        capthick=0.45,
                        zorder=3,
                    )
                    if show_raw:
                        for seed_idx, seed in enumerate(seeds):
                            row = rows_by_key.get((seed, agent_key, side_key))
                            if row is None:
                                continue
                            raw_values = np.asarray(list(row[raw_key]), dtype=float)
                            if raw_values.size == 0:
                                continue
                            rng = np.random.default_rng(7_000 + 100 * seed + 10 * col_idx + (1 if agent_key == "agent_114" else 0))
                            jitter = rng.uniform(-width * 0.25, width * 0.25, size=raw_values.size)
                            ax.scatter(
                                np.full(raw_values.size, xpos[seed_idx], dtype=float) + jitter,
                                raw_values,
                                color=raw_point_color,
                                alpha=0.28,
                                s=5,
                                edgecolors="none",
                                zorder=4,
                            )
                if row_idx == 0:
                    ax.set_title(side_label, fontsize=10.0, pad=2)
                if col_idx == 0:
                    ax.set_ylabel(y_label)
                ax.set_ylim(*ylimit)
                ax.set_xticks(x)
                ax.set_xticklabels([str(seed) for seed in seeds], rotation=45, ha="right", fontsize=8.0)
                ax.grid(axis="y", alpha=0.25)
        axes[1, 0].set_xlabel("Environment seed")
        axes[1, 1].set_xlabel("Environment seed")
        raw_legend = Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor=raw_point_color,
            markeredgecolor="none",
            alpha=0.28,
            markersize=4,
            label="Damage rollouts",
        )
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles=[*handles, raw_legend], labels=[*labels, "Damage rollouts"], loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.0))
        fig.suptitle("Agent 111 vs Agent 114 Villager killing statistics", fontsize=11.0, y=0.965)
        fig.tight_layout(rect=(0, 0.105, 1, 0.99), h_pad=0.55, w_pad=0.45)
        save_figure(fig, output_path)

    def render_head_fake_stability(self, ctx: RunContext) -> None:
        family = "head_fake_stability"
        cfg = ctx.config["families"][family]
        summary_csv = path_from_config(str(cfg.get("summary_csv", "")))
        fig_dir = ctx.figures_dir / "attention"
        fig_dir.mkdir(parents=True, exist_ok=True)
        if not summary_csv.exists():
            ctx.warnings.append(f"Missing head fake stability summary CSV: {summary_csv}")
            return
        df = pd.read_csv(summary_csv)
        conditional = df[df["branch"].astype(str) == "conditional"].copy()
        if conditional.empty:
            ctx.warnings.append(f"No conditional rows in head fake stability summary CSV: {summary_csv}")
            return
        layers = sorted(conditional["layer"].astype(int).unique().tolist())
        heads = sorted(conditional["head"].astype(int).unique().tolist())
        grid = np.full((max(layers) + 1, max(heads) + 1), np.nan, dtype=float)
        for row in conditional.itertuples(index=False):
            grid[int(row.layer), int(row.head)] = float(row.fake_pair_mean_corr)
        pdf_path = fig_dir / "head_fake_pair_stability.pdf"
        self._plot_head_fake_stability_grid(grid, pdf_path)
        add_artifact(
            ctx,
            family,
            "attention/head_fake_pair_stability",
            pdf_path,
            "Conditional head fake-model pairwise stability.",
        )

    def _plot_head_fake_stability_grid(self, grid: np.ndarray, output_path: Path) -> None:
        fig, ax = plt.subplots(figsize=(NEURIPS_TEXT_WIDTH_IN, 2.65))
        finite = grid[np.isfinite(grid)]
        vmin = float(np.nanmin(finite)) if finite.size else 0.0
        vmax = float(np.nanmax(finite)) if finite.size else 1.0
        im = ax.imshow(grid, cmap="cividis", aspect="auto", vmin=vmin, vmax=vmax)
        ax.set_title("Conditional Head Fake-Model Pairwise Stability", fontsize=11.0, pad=3)
        ax.set_xlabel("Head")
        ax.set_ylabel("Layer")
        ax.set_xticks(np.arange(grid.shape[1]))
        ax.set_yticks(np.arange(grid.shape[0]))
        ax.tick_params(axis="both", labelsize=8.0, pad=1.5)
        for layer in range(grid.shape[0]):
            for head in range(grid.shape[1]):
                value = grid[layer, head]
                if not np.isfinite(value):
                    continue
                color = "white" if value < (vmin + 0.35 * (vmax - vmin)) else PALETTE["black"]
                ax.text(head, layer, f"{value:.2f}", ha="center", va="center", fontsize=6.0, color=color)
        cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
        cbar.set_label("Mean pairwise corr.", fontsize=9.0)
        cbar.ax.tick_params(labelsize=8.0, pad=1.5)
        fig.tight_layout(pad=0.1)
        save_figure(fig, output_path)

    def render_teleport_layout(self, ctx: RunContext) -> None:
        family = "teleport_layout"
        cfg = ctx.config["families"][family]
        cache_dir = ctx.cache_root / family
        fig_dir = ctx.figures_dir / "teleport_layout"
        cache_dir.mkdir(parents=True, exist_ok=True)
        fig_dir.mkdir(parents=True, exist_ok=True)
        campaign_root = Path(str(cfg.get("campaign_root", ""))).expanduser()
        if campaign_root.exists() and (campaign_root / "rollout_status.csv").exists():
            self._render_teleport_layout_from_campaign(ctx, cfg, campaign_root.resolve(), cache_dir, fig_dir)
            return

        for source_cfg in cfg.get("sources", []):
            name = str(source_cfg["name"])
            cached_csv = cache_dir / f"{name}.csv"
            if not (ctx.from_cache and cached_csv.exists()):
                src_csv = Path(source_cfg["csv"]).expanduser().resolve()
                if not src_csv.exists():
                    ctx.warnings.append(f"Missing teleport layout CSV: {src_csv}")
                    continue
                shutil.copy2(src_csv, cached_csv)
            df = pd.read_csv(cached_csv)
            highlight_labels = [str(label) for label in cfg.get("highlight_labels", {}).get(name, [])]
            seed = self._seed_from_layout_name(name)
            for ci_suffix, show_ci in [("with_ci", True), ("no_ci", False)]:
                pdf_path = fig_dir / f"{name}_left_choice_layout_scatter_{ci_suffix}.pdf"
                self._plot_teleport_layout(
                    df,
                    "T-Maze Choice Scatter",
                    pdf_path,
                    show_ci=show_ci,
                    highlight_labels=highlight_labels,
                    seed=seed,
                )
                add_artifact(
                    ctx,
                    family,
                    f"teleport_layout/{name}_left_choice_layout_scatter_{ci_suffix}",
                    pdf_path,
                    f"Teleport layout choice scatter for {name} ({ci_suffix.replace('_', ' ')}).",
                )
            legacy_path = fig_dir / f"{name}_left_choice_layout_scatter.pdf"
            self._plot_teleport_layout(
                df,
                "T-Maze Choice Scatter",
                legacy_path,
                show_ci=True,
                highlight_labels=highlight_labels,
                seed=seed,
            )
            add_artifact(
                ctx,
                family,
                f"teleport_layout/{name}_left_choice_layout_scatter",
                legacy_path,
                f"Teleport layout choice scatter for {name}.",
            )

    def _render_teleport_layout_from_campaign(
        self,
        ctx: RunContext,
        cfg: dict[str, Any],
        campaign_root: Path,
        cache_dir: Path,
        fig_dir: Path,
    ) -> None:
        status_path = campaign_root / "rollout_status.csv"
        first_segments = int(cfg.get("first_segments", 100))
        status_df = pd.read_csv(status_path)
        condition_pairs = [
            (49, "seed049_villager_left", "seed049_villager_right", "seed49_first100"),
            (481, "seed481_villager_left", "seed481_villager_right", "seed481_first100"),
        ]

        condition_cache: dict[str, pd.DataFrame] = {}
        needed_conditions = sorted({condition for _, left, right, _ in condition_pairs for condition in (left, right)})
        for condition in needed_conditions:
            condition_csv = cache_dir / f"{condition}_model_comparison_first{first_segments}.csv"
            if ctx.from_cache and condition_csv.exists():
                condition_df = pd.read_csv(condition_csv)
            else:
                condition_df = self._build_teleport_condition_compare(status_df, condition, first_segments, ctx)
                condition_df.to_csv(condition_csv, index=False)
            condition_cache[condition] = condition_df

        for seed, left_key, right_key, name in condition_pairs:
            merged_csv = cache_dir / f"{name}.csv"
            if ctx.from_cache and merged_csv.exists():
                merged_df = pd.read_csv(merged_csv)
            else:
                merged_df = self._merge_teleport_layout_rows(condition_cache[left_key], condition_cache[right_key])
                merged_df.to_csv(merged_csv, index=False)
            highlight_labels = [str(label) for label in cfg.get("highlight_labels", {}).get(name, [])]
            for ci_suffix, show_ci in [("with_ci", True), ("no_ci", False)]:
                pdf_path = fig_dir / f"{name}_left_choice_layout_scatter_{ci_suffix}.pdf"
                self._plot_teleport_layout(
                    merged_df,
                    "T-Maze Choice Scatter",
                    pdf_path,
                    show_ci=show_ci,
                    highlight_labels=highlight_labels,
                    seed=seed,
                )
                add_artifact(
                    ctx,
                    "teleport_layout",
                    f"teleport_layout/{name}_left_choice_layout_scatter_{ci_suffix}",
                    pdf_path,
                    f"Teleport layout choice scatter for seed {seed}, first {first_segments} segments ({ci_suffix.replace('_', ' ')}).",
                )
            legacy_path = fig_dir / f"{name}_left_choice_layout_scatter.pdf"
            self._plot_teleport_layout(
                merged_df,
                "T-Maze Choice Scatter",
                legacy_path,
                show_ci=True,
                highlight_labels=highlight_labels,
                seed=seed,
            )
            add_artifact(
                ctx,
                "teleport_layout",
                f"teleport_layout/{name}_left_choice_layout_scatter",
                legacy_path,
                f"Teleport layout choice scatter for seed {seed}, first {first_segments} segments.",
            )

    def _build_teleport_condition_compare(
        self,
        status_df: pd.DataFrame,
        condition: str,
        first_segments: int,
        ctx: RunContext,
    ) -> pd.DataFrame:
        from steve1.run_agent import analyze_teleport_model_compare as compare
        from steve1.run_agent.teleport_campaign_common import model_sort_key

        rows: list[dict[str, object]] = []
        condition_df = status_df[(status_df["condition"] == condition) & (status_df["status"] == "done")].copy()
        for _, row in condition_df.sort_values("model_id").iterrows():
            model_id = str(row["model_id"])
            run_dir = Path(str(row["run_dir"]))
            loc_path = run_dir / "location_stats.log"
            tp_path = run_dir / "teleport_events.log"
            if not loc_path.exists() or not tp_path.exists():
                ctx.warnings.append(f"Skipping teleport {condition}/{model_id}; missing location or teleport log.")
                continue
            x, z = compare._load_locations(loc_path)
            breaks = compare._load_break_indices(tp_path, len(x))
            endpoint_indices = compare._segment_endpoint_indices(len(x), breaks)[:first_segments]
            left, right, tie, _endpoint_rows = compare._classify_endpoints(x, z, endpoint_indices)
            non_tie = left + right
            left_rate = left / non_tie if non_tie else float("nan")
            ci_low, ci_high = compare._wilson_ci(left, non_tie)
            rows.append(
                {
                    "model_id": model_id,
                    "run_dir": str(run_dir),
                    "segments_total": len(endpoint_indices),
                    "left_segments": left,
                    "right_segments": right,
                    "tie_segments": tie,
                    "left_rate_pct_excluding_ties": left_rate * 100.0,
                    "left_ci95_low_pct": ci_low * 100.0,
                    "left_ci95_high_pct": ci_high * 100.0,
                }
            )
        rows.sort(key=lambda r: model_sort_key(str(r["model_id"])))
        return pd.DataFrame(rows)

    def _merge_teleport_layout_rows(self, left_df: pd.DataFrame, right_df: pd.DataFrame) -> pd.DataFrame:
        left_rows = {str(row["model_id"]): row for _, row in left_df.iterrows()}
        right_rows = {str(row["model_id"]): row for _, row in right_df.iterrows()}
        model_ids = sorted(set(left_rows) & set(right_rows), key=self._model_id_sort_key)
        out_rows: list[dict[str, object]] = []
        for model_id in model_ids:
            left_row = left_rows[model_id]
            right_row = right_rows[model_id]
            left_pct = float(left_row["left_rate_pct_excluding_ties"])
            right_pct = float(right_row["left_rate_pct_excluding_ties"])
            left_side_bias = 0.5 * (left_pct + right_pct)
            villager_bias = 0.5 * (left_pct + (100.0 - right_pct))
            out_rows.append(
                {
                    "model_id": model_id,
                    "short_label": self._seed_label(model_id),
                    "left_run_dir": left_row["run_dir"],
                    "right_run_dir": right_row["run_dir"],
                    "left_pct_villager_left": left_pct,
                    "right_pct_villager_left": 100.0 - left_pct,
                    "left_pct_villager_right": right_pct,
                    "left_ci_low_villager_left": float(left_row["left_ci95_low_pct"]),
                    "left_ci_high_villager_left": float(left_row["left_ci95_high_pct"]),
                    "left_ci_low_villager_right": float(right_row["left_ci95_low_pct"]),
                    "left_ci_high_villager_right": float(right_row["left_ci95_high_pct"]),
                    "left_side_bias_pct": left_side_bias,
                    "villager_choice_bias_pct": villager_bias,
                    "segments_left_layout": int(left_row["segments_total"]),
                    "segments_right_layout": int(right_row["segments_total"]),
                }
            )
        return pd.DataFrame(out_rows)

    def _seed_label(self, model_id: str) -> str:
        import re

        match = re.search(r"dataset_seed(\d+)", model_id)
        return match.group(1) if match else model_id

    def _model_id_sort_key(self, model_id: str) -> tuple[int, str]:
        import re

        match = re.search(r"dataset_seed(\d+)", model_id)
        return (int(match.group(1)) if match else 10**9, model_id)

    def _seed_from_layout_name(self, name: str) -> int | None:
        match = re.search(r"seed(\d+)", name)
        return int(match.group(1)) if match else None

    def _plot_teleport_layout(
        self,
        df: pd.DataFrame,
        title: str,
        output_path: Path,
        *,
        show_ci: bool = True,
        highlight_labels: list[str] | None = None,
        seed: int | None = None,
    ) -> None:
        x = 100.0 - df["left_pct_villager_left"].astype(float).to_numpy()
        y = df["left_pct_villager_right"].astype(float).to_numpy()
        labels = df["short_label"].astype(str).tolist()
        fig, ax = plt.subplots(figsize=(SIDE_BY_SIDE_FIGURE_WIDTH_IN, 2.82))
        if show_ci and {"left_ci_low_villager_left", "left_ci_high_villager_left", "left_ci_low_villager_right", "left_ci_high_villager_right"}.issubset(df.columns):
            left_low = np.clip(df["left_ci_low_villager_left"].astype(float).to_numpy(), 0.0, 100.0)
            left_high = np.clip(df["left_ci_high_villager_left"].astype(float).to_numpy(), 0.0, 100.0)
            right_low = np.clip(df["left_ci_low_villager_right"].astype(float).to_numpy(), 0.0, 100.0)
            right_high = np.clip(df["left_ci_high_villager_right"].astype(float).to_numpy(), 0.0, 100.0)
            xerr = np.vstack([np.maximum(0.0, x - (100.0 - left_high)), np.maximum(0.0, (100.0 - left_low) - x)])
            yerr = np.vstack([np.maximum(0.0, y - right_low), np.maximum(0.0, right_high - y)])
            ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt="none", ecolor="#9AA3AD", elinewidth=0.45, alpha=0.35, zorder=1)
        ax.scatter(x, y, s=16, alpha=0.8, color=PALETTE["blue"], edgecolors="white", linewidths=0.35, zorder=2)
        highlight_set = set(highlight_labels or [])
        for xi, yi, label in zip(x, y, labels):
            if label not in highlight_set:
                continue
            dx = -2.0 if xi > 82.0 else 2.0
            dy = -4.0 if yi > 85.0 else 2.0
            ha = "right" if dx < 0 else "left"
            va = "top" if dy < 0 else "bottom"
            ax.scatter(
                [xi],
                [yi],
                s=37,
                alpha=0.98,
                facecolors="none",
                edgecolors="black",
                linewidths=0.9,
                zorder=5,
            )
            text = ax.text(
                xi + dx,
                yi + dy,
                label,
                fontsize=8.5,
                fontweight="semibold",
                color=PALETTE["black"],
                ha=ha,
                va=va,
                zorder=6,
            )
            text.set_path_effects([patheffects.Stroke(linewidth=1.6, foreground="white"), patheffects.Normal()])
        ax.axvline(50.0, color="#7B8188", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.axhline(50.0, color="#7B8188", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.plot([0, 100], [0, 100], color=PALETTE["green"], linestyle=":", linewidth=0.9, alpha=0.8)
        ax.text(2, 98, "always\nleft", ha="left", va="top", fontsize=10.0, linespacing=0.9)
        ax.text(98, 98, "regular\ntree", ha="right", va="top", fontsize=10.0, linespacing=0.9)
        ax.text(2, 2, "villager\ntree", ha="left", va="bottom", fontsize=10.0, linespacing=0.9)
        ax.text(98, 2, "always\nright", ha="right", va="bottom", fontsize=10.0, linespacing=0.9)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ticks = [0, 25, 50, 75, 100]
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Right choices, villager-left (%)")
        ax.set_ylabel("Left choices, villager-right (%)")
        ax.set_title(title, fontsize=11.0, pad=1)
        ax.grid(alpha=0.3)
        fig.tight_layout(pad=0.02)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_figure(fig, output_path)

    def render_agent_trajectories(self, ctx: RunContext) -> None:
        family = "agent_trajectories"
        cfg = ctx.config["families"][family]
        sweep_root = Path(cfg["sweep_root"]).expanduser().resolve()
        cache_dir = ctx.cache_root / family
        fig_dir = ctx.figures_dir / "agent_trajectories"
        cache_dir.mkdir(parents=True, exist_ok=True)
        fig_dir.mkdir(parents=True, exist_ok=True)
        for spec in cfg.get("sets", []):
            name = str(spec["name"])
            trajectories = self._load_or_collect_trajectories(ctx, sweep_root, spec, cache_dir / name)
            if bool(spec.get("filter_start_cluster", True)):
                trajectories = self._filter_trajectory_start_cluster(ctx, spec, trajectories)
            if not trajectories:
                ctx.warnings.append(f"No trajectories found for trajectory set {name}")
                continue
            self._render_trajectory_set(ctx, spec, trajectories, fig_dir / name)

    def _load_or_collect_trajectories(
        self,
        ctx: RunContext,
        sweep_root: Path,
        spec: dict[str, Any],
        cache_dir: Path,
    ) -> dict[str, list[np.ndarray]]:
        mode = str(spec.get("mode", "sweep"))
        if mode == "teleport_condition":
            return self._load_or_collect_teleport_condition_trajectories(ctx, spec, cache_dir)
        if mode != "sweep":
            raise ValueError(f"Unsupported trajectory collection mode for {spec['name']}: {mode}")

        cache_dir.mkdir(parents=True, exist_ok=True)
        agents = [str(agent) for agent in spec.get("agents", [])]
        trajectories: dict[str, list[np.ndarray]] = {}
        if ctx.from_cache and all((cache_dir / f"{agent}.npz").exists() for agent in agents):
            for agent in agents:
                payload = np.load(cache_dir / f"{agent}.npz")
                trajectories[agent] = [payload[key] for key in sorted(payload.files)]
            return trajectories

        seed_label = f"seed_{int(spec['seed']):03d}"
        side = str(spec["side"])
        full_points = int(spec.get("full_episode_points", 0))
        manifest_rows = []
        for agent in agents:
            agent_root = sweep_root / seed_label / side / agent
            agent_trajs: list[np.ndarray] = []
            for location_path in sorted(agent_root.rglob("location_stats.log")):
                traj = self._read_trajectory(location_path)
                if full_points and traj.shape[0] < full_points:
                    continue
                agent_trajs.append(traj)
                manifest_rows.append({"set": spec["name"], "agent": agent, "points": traj.shape[0], "path": str(location_path)})
            trajectories[agent] = agent_trajs
            np.savez_compressed(cache_dir / f"{agent}.npz", **{f"traj_{idx:04d}": traj for idx, traj in enumerate(agent_trajs)})
        pd.DataFrame(manifest_rows).to_csv(cache_dir / "trajectories_manifest.csv", index=False)
        return trajectories

    def _load_or_collect_teleport_condition_trajectories(
        self,
        ctx: RunContext,
        spec: dict[str, Any],
        cache_dir: Path,
    ) -> dict[str, list[np.ndarray]]:
        cache_dir.mkdir(parents=True, exist_ok=True)
        models = list(spec.get("models", []))
        agents = [str(item["agent"]) for item in models]
        trajectories: dict[str, list[np.ndarray]] = {}
        if ctx.from_cache and agents and all((cache_dir / f"{agent}.npz").exists() for agent in agents):
            for agent in agents:
                payload = np.load(cache_dir / f"{agent}.npz")
                trajectories[agent] = [payload[key] for key in sorted(payload.files)]
            return trajectories

        campaign_root = path_from_config(str(spec["campaign_root"]))
        status_path = campaign_root / "rollout_status.csv"
        status_df = pd.read_csv(status_path)
        condition = str(spec["condition"])
        first_segments = int(spec.get("first_segments", 0))
        manifest_rows = []

        for item in models:
            agent = str(item["agent"])
            model_id = str(item["model_id"])
            rows = status_df[
                (status_df["condition"].astype(str) == condition)
                & (status_df["model_id"].astype(str) == model_id)
                & (status_df["status"].astype(str) == "done")
            ]
            if rows.empty:
                ctx.warnings.append(f"No completed teleport rollout found for {condition} / {model_id}")
                trajectories[agent] = []
                continue
            run_dir = path_from_config(str(rows.iloc[0]["run_dir"]))
            agent_trajs = self._read_teleport_reset_trajectories(run_dir, first_segments=first_segments)
            trajectories[agent] = agent_trajs
            np.savez_compressed(cache_dir / f"{agent}.npz", **{f"traj_{idx:04d}": traj for idx, traj in enumerate(agent_trajs)})
            for idx, traj in enumerate(agent_trajs):
                manifest_rows.append(
                    {
                        "set": spec["name"],
                        "agent": agent,
                        "model_id": model_id,
                        "condition": condition,
                        "segment_index": idx,
                        "points": traj.shape[0],
                        "run_dir": str(run_dir),
                    }
                )

        pd.DataFrame(manifest_rows).to_csv(cache_dir / "trajectories_manifest.csv", index=False)
        return trajectories

    def _read_teleport_reset_trajectories(self, run_dir: Path, *, first_segments: int = 0) -> list[np.ndarray]:
        from steve1.run_agent import analyze_teleport_model_compare as teleport_compare

        location_path = run_dir / "location_stats.log"
        teleport_path = run_dir / "teleport_events.log"
        x, z = teleport_compare._load_locations(location_path)
        break_indices = teleport_compare._load_break_indices(teleport_path, len(x))
        segments = teleport_compare._build_segments(x, z, break_indices)
        if first_segments > 0:
            segments = segments[:first_segments]
        return [np.column_stack([sx, sz]) for sx, sz in segments]

    def _filter_trajectory_start_cluster(
        self,
        ctx: RunContext,
        spec: dict[str, Any],
        trajectories: dict[str, list[np.ndarray]],
    ) -> dict[str, list[np.ndarray]]:
        starts = [traj[0] for agent_trajs in trajectories.values() for traj in agent_trajs if traj.size]
        if not starts:
            return trajectories
        center = np.median(np.vstack(starts), axis=0)
        max_dist = float(spec.get("start_distance_max", 20.0))
        filtered: dict[str, list[np.ndarray]] = {}
        dropped = 0
        for agent, agent_trajs in trajectories.items():
            kept = []
            for traj in agent_trajs:
                if float(np.linalg.norm(traj[0] - center)) <= max_dist:
                    kept.append(traj)
                else:
                    dropped += 1
            filtered[agent] = kept
        if dropped:
            ctx.warnings.append(
                f"Dropped {dropped} trajectory outlier(s) from {spec['name']} with start farther than {max_dist:g} blocks from the modal spawn."
            )
        return filtered

    def _read_trajectory(self, location_path: Path) -> np.ndarray:
        df = pd.read_csv(location_path)
        return df[["xpos", "zpos"]].astype(float).to_numpy()

    def _render_trajectory_set(
        self,
        ctx: RunContext,
        spec: dict[str, Any],
        trajectories: dict[str, list[np.ndarray]],
        fig_dir: Path,
    ) -> None:
        all_trajs = [traj for agent_trajs in trajectories.values() for traj in agent_trajs]
        stacked = np.vstack(all_trajs)
        spawn = all_trajs[0][0]
        left_tree, right_tree = self._tree_centers(spawn)
        plot_points = np.vstack([stacked, left_tree.reshape(1, 2), right_tree.reshape(1, 2), spawn.reshape(1, 2)])
        x_half = max(5.2, float(np.nanmax(np.abs(plot_points[:, 0] - spawn[0]))) + 0.25)
        top_offset = max(11.0, float(np.nanmax(plot_points[:, 1] - spawn[1])) + 0.75)
        bottom_offset = top_offset / 3.0
        xlim = (float(spawn[0] - x_half), float(spawn[0] + x_half))
        ylim = (float(spawn[1] - bottom_offset), float(spawn[1] + top_offset))
        color_cycle = [PALETTE["blue"], PALETTE["orange"], PALETTE["cyan"], PALETTE["red"], PALETTE["brown"]]
        agent_colors = {agent: color_cycle[idx % len(color_cycle)] for idx, agent in enumerate(trajectories)}
        agent_labels = {agent: self._default_agent_label(agent) for agent in trajectories}
        agent_labels.update({str(agent): str(label) for agent, label in spec.get("agent_labels", {}).items()})
        rasterized = bool(spec.get("rasterize_lines", True))
        seed = int(spec["seed"])
        side = str(spec["side"])
        villager_side = self._villager_side_from_side_key(side)

        for agent, agent_trajs in trajectories.items():
            if not agent_trajs:
                continue
            pdf_path = fig_dir / f"{agent}_topdown_trajectories.pdf"
            self._plot_trajectories(
                {agent: agent_trajs},
                agent_colors,
                agent_labels,
                "T-Maze Top-Down Trajectories",
                xlim,
                ylim,
                villager_side,
                rasterized,
                pdf_path,
            )
            add_artifact(
                ctx,
                "agent_trajectories",
                f"agent_trajectories/{spec['name']}/{agent}_topdown_trajectories",
                pdf_path,
                f"Top-down trajectory overlay for {agent} in {spec['name']}.",
            )

        if not bool(spec.get("include_combined", False)):
            return

        combined_path = fig_dir / "combined_topdown_trajectories.pdf"
        self._plot_trajectories(
            trajectories,
            agent_colors,
            agent_labels,
            "T-Maze Top-Down Trajectories",
            xlim,
            ylim,
            villager_side,
            rasterized,
            combined_path,
        )
        add_artifact(
            ctx,
            "agent_trajectories",
            f"agent_trajectories/{spec['name']}/combined_topdown_trajectories",
            combined_path,
            f"Combined top-down trajectory overlay for {spec['name']}.",
        )

    def _padded_limits(self, values: np.ndarray, *, fraction: float = 0.06, min_pad: float = 1.0) -> tuple[float, float]:
        lo = float(np.nanmin(values))
        hi = float(np.nanmax(values))
        pad = max((hi - lo) * fraction, min_pad)
        return lo - pad, hi + pad

    def _tree_centers(self, spawn: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        left_offset = np.array([4.0, 7.0], dtype=float)
        right_offset = np.array([-4.0, 7.0], dtype=float)
        return spawn + left_offset, spawn + right_offset

    def _villager_side_from_side_key(self, side_key: str) -> str:
        if "left" in side_key:
            return "left"
        if "right" in side_key:
            return "right"
        return "left"

    def _default_agent_label(self, agent: str) -> str:
        if agent.startswith("agent_"):
            return f"Agent {agent.removeprefix('agent_')}"
        return agent

    def _draw_regular_tree_icon(self, ax: plt.Axes, xy: np.ndarray) -> None:
        icon = DrawingArea(14, 16, 0, 0)
        icon.add_artist(Rectangle((5.6, 1.0), 2.8, 4.9, facecolor="#8C613C", edgecolor="#5B3A24", linewidth=0.45))
        icon.add_artist(
            Polygon(
                [(7, 15.4), (1.4, 4.9), (12.6, 4.9)],
                closed=True,
                facecolor=PALETTE["green"],
                edgecolor="#3F6F2C",
                linewidth=0.5,
            )
        )
        ax.add_artist(AnnotationBbox(icon, xy, frameon=False, box_alignment=(0.5, 0.35), zorder=6))

    def _draw_villager_icon(self, ax: plt.Axes, xy: np.ndarray) -> None:
        icon = DrawingArea(24, 18, 0, 0)
        scale = 0.315
        cx, cy = 12.0, 7.0

        def sx(x: float) -> float:
            return (x - 16.0) * scale + cx

        def sy(y: float) -> float:
            return (y - 14.0) * scale + cy

        icon.add_artist(
            Polygon(
                [(12, 16.1), (0, 10.5), (24, 10.5)],
                closed=True,
                facecolor=PALETTE["green"],
                edgecolor="#3F6F2C",
                linewidth=0.5,
            )
        )
        icon.add_artist(Rectangle((sx(5), sy(3)), 22.0 * scale, 22.0 * scale, facecolor="#C99762", edgecolor="#6F4B2A", linewidth=0.5))
        icon.add_artist(Rectangle((sx(8), sy(18)), 16.0 * scale, 3.0 * scale, facecolor="#4A2C18", edgecolor="none"))
        icon.add_artist(Rectangle((sx(9), sy(14)), 4.0 * scale, 3.0 * scale, facecolor="#24321F", edgecolor="none"))
        icon.add_artist(Rectangle((sx(19), sy(14)), 4.0 * scale, 3.0 * scale, facecolor="#24321F", edgecolor="none"))
        icon.add_artist(Rectangle((sx(14), sy(8)), 4.0 * scale, 9.0 * scale, facecolor="#A7794F", edgecolor="#6F4B2A", linewidth=0.35))
        ax.add_artist(AnnotationBbox(icon, xy, frameon=False, box_alignment=(0.5, 0.45), zorder=6))

    def _plot_trajectories(
        self,
        trajectories: dict[str, list[np.ndarray]],
        agent_colors: dict[str, str],
        agent_labels: dict[str, str],
        title: str,
        xlim: tuple[float, float],
        ylim: tuple[float, float],
        villager_side: str,
        rasterized: bool,
        output_path: Path,
    ) -> None:
        fig, ax = plt.subplots(figsize=(SIDE_BY_SIDE_FIGURE_WIDTH_IN, 2.62))
        single_agent = len([agent for agent, trajs in trajectories.items() if trajs]) == 1
        for agent, agent_trajs in trajectories.items():
            color = agent_colors.get(agent, PALETTE["gray"])
            for traj in agent_trajs:
                line_kwargs: dict[str, object] = {"alpha": 0.58 if single_agent else 0.18, "linewidth": 0.45, "rasterized": rasterized}
                if not single_agent:
                    line_kwargs["color"] = color
                ax.plot(traj[:, 0], traj[:, 1], **line_kwargs)
                start_color = "black" if single_agent else color
                ax.scatter(
                    [traj[0, 0]],
                    [traj[0, 1]],
                    color=start_color,
                    alpha=0.18,
                    s=7,
                    edgecolors="none",
                    rasterized=rasterized,
                )

        first_agent = next(agent for agent, trajs in trajectories.items() if trajs)
        spawn = trajectories[first_agent][0][0]
        left_tree, right_tree = self._tree_centers(spawn)
        if villager_side == "left":
            left_icon = self._draw_villager_icon
            left_label = "villager tree"
            right_icon = self._draw_regular_tree_icon
            right_label = "regular tree"
        else:
            left_icon = self._draw_regular_tree_icon
            left_label = "regular tree"
            right_icon = self._draw_villager_icon
            right_label = "villager tree"

        ax.scatter([spawn[0]], [spawn[1]], marker="*", s=90, c="black", zorder=5)
        ax.annotate(
            "spawn",
            xy=spawn,
            xytext=(6, -5),
            textcoords="offset points",
            ha="left",
            va="top",
            fontsize=10.0,
            color=PALETTE["black"],
        )
        left_icon(ax, left_tree)
        right_icon(ax, right_tree)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.invert_xaxis()
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("X coordinate")
        ax.set_ylabel("Z coordinate")
        ax.set_title(title, fontsize=11.0, pad=1)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=7, integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=7, integer=True))
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.0f"))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))
        ax.grid(True, alpha=0.25)
        fig.tight_layout(pad=0.02)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_figure(fig, output_path)

    def build_overleaf_bundle(self, ctx: RunContext) -> None:
        include_patterns = ctx.config.get("overleaf", {}).get("include", ["**"])
        exclude_patterns = ctx.config.get("overleaf", {}).get("exclude", [])
        ctx.overleaf_dir.mkdir(parents=True, exist_ok=True)

        selected = []
        for artifact in ctx.artifacts:
            include = any(fnmatch.fnmatch(artifact.id, pattern) for pattern in include_patterns)
            exclude = any(fnmatch.fnmatch(artifact.id, pattern) for pattern in exclude_patterns)
            if include and not exclude:
                selected.append(artifact)

        for artifact in selected:
            src = Path(artifact.path)
            if src.suffix.lower() != ".pdf":
                continue
            dest = ctx.overleaf_dir / f"{artifact.id}.pdf"
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)

        self._build_curated_overleaf_figures(ctx, selected)

    def _build_curated_overleaf_figures(self, ctx: RunContext, selected: list[Artifact]) -> None:
        figure_names = [str(name) for name in ctx.config.get("overleaf", {}).get("figures", [])]
        if not figure_names:
            return
        figures_dir = ctx.overleaf_dir / "figures"
        if figures_dir.exists():
            shutil.rmtree(figures_dir)
        figures_dir.mkdir(parents=True, exist_ok=True)

        artifacts_by_name: dict[str, Path] = {}
        for artifact in selected:
            path = Path(artifact.path)
            if path.suffix.lower() == ".pdf":
                artifacts_by_name.setdefault(path.name, path)

        static_figures = {
            str(name): Path(str(path)).expanduser().resolve()
            for name, path in ctx.config.get("overleaf", {}).get("static_figures", {}).items()
        }

        missing: list[str] = []
        for name in figure_names:
            src = artifacts_by_name.get(name)
            if src is None:
                static_src = static_figures.get(name)
                if static_src is not None and static_src.exists():
                    src = static_src
            if src is None or not src.exists():
                missing.append(name)
                continue
            shutil.copy2(src, figures_dir / name)

        if missing:
            ctx.warnings.append(f"Missing curated Overleaf figure(s): {', '.join(missing)}")
