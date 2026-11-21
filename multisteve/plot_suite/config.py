from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = REPO_ROOT.parent
EVALS_VILLAGER_ROOT = REPO_ROOT / "data" / "evals-villager"
FINAL_EVALS_ROOT = REPO_ROOT / "data" / "final-evals"


DEFAULT_CONFIG: dict[str, Any] = {
    "outputs": {
        "root": str(REPO_ROOT / "outputs" / "plot_suite"),
        "cache_root": str(REPO_ROOT / "outputs" / "plot_suite" / "cache"),
        "overleaf_root": str(REPO_ROOT / "outputs" / "plot_suite" / "overleaf_figures"),
    },
    "style": {
        "dpi": 220,
        "pad_inches": 0.0,
        "font_size": 10.0,
        "title_font_size": 11.0,
        "label_font_size": 10.0,
        "target_text_width_in": 5.5,
        "side_by_side_width_fraction": 0.49,
    },
    "data_roots": {
        "final_evals": str(FINAL_EVALS_ROOT),
        "villager": str(EVALS_VILLAGER_ROOT),
        "repo_data": str(REPO_ROOT / "data"),
    },
    "families": {
        "full_eval": {"enabled": True},
        "partial_eval_40": {"enabled": True},
        "partial_eval_28_checkpoints": {"enabled": True},
        "tree_prompt_100": {"enabled": True},
        "villager_passes": {"enabled": True, "pass_start": 1, "pass_end": 20},
        "tree_vs_villager": {
            "enabled": True,
            "pass_start": 1,
            "pass_end": 20,
            "filtered_logs_setting": "logs_above_villager_true",
            "filtered_prompt_slug": "wood",
        },
        "villager_chop_40": {"enabled": True, "pass_start": 1, "pass_end": 20},
        "villager_bitflip": {"enabled": True, "pass_start": 1, "pass_end": 20},
        "agent_111_114_seed_sweep": {
            "enabled": True,
            "sweep_root": str(EVALS_VILLAGER_ROOT / "agent_111_114_short_logs_above_true_seed_sweep"),
        },
        "head_fake_stability": {
            "enabled": True,
            "summary_csv": str(REPO_ROOT / "outputs" / "plot_suite" / "cache" / "head_fake_stability" / "head_summary.csv"),
        },
        "teleport_layout": {
            "enabled": True,
            "campaign_root": str(REPO_ROOT / "data" / "teleport_100_model_campaign_20260415-085856"),
            "first_segments": 100,
            "highlight_labels": {
                "seed49_first100": ["131", "132"]
            },
            "sources": [
                {
                    "name": "seed49_first100",
                    "title": "Seed 49 T-maze Choice Scatter",
                    "csv": str(
                        REPO_ROOT
                        / "data"
                        / "teleport_100_model_campaign_20260415-085856"
                        / "analysis"
                        / "seed49_layout_scatter_first100"
                        / "left_choice_layout_scatter.csv"
                    ),
                },
                {
                    "name": "seed481_first100",
                    "title": "Seed 481 T-maze Choice Scatter",
                    "csv": str(
                        REPO_ROOT
                        / "data"
                        / "teleport_100_model_campaign_20260415-085856"
                        / "analysis"
                        / "seed481_layout_scatter_first100"
                        / "left_choice_layout_scatter.csv"
                    ),
                },
            ],
        },
        "agent_trajectories": {
            "enabled": True,
            "sweep_root": str(EVALS_VILLAGER_ROOT / "agent_111_114_short_logs_above_true_seed_sweep"),
            "sets": [
                {
                    "name": "seed049_left",
                    "seed": 49,
                    "side": "01_left",
                    "agents": ["agent_111", "agent_114"],
                    "full_episode_points": 1001,
                    "start_distance_max": 20.0,
                    "rasterize_lines": True,
                },
                {
                    "name": "seed049_villager_right_agents131_132",
                    "mode": "teleport_condition",
                    "campaign_root": str(REPO_ROOT / "data" / "teleport_100_model_campaign_20260415-085856"),
                    "condition": "seed049_villager_right",
                    "seed": 49,
                    "side": "seed049_villager_right",
                    "models": [
                        {"agent": "agent_131", "model_id": "dataset_seed131-weight_init_seed131"},
                        {"agent": "agent_132", "model_id": "dataset_seed132-weight_init_seed132"},
                    ],
                    "agent_labels": {"agent_131": "Agent 131", "agent_132": "Agent 132"},
                    "first_segments": 100,
                    "filter_start_cluster": False,
                    "rasterize_lines": True,
                }
            ],
        },
    },
    "overleaf": {
        "include": ["**"],
        "exclude": [],
        "figures": [
            "figure1.pdf",
            "text_dirt.pdf",
            "text_logs.pdf",
            "text_max_distance.pdf",
            "text_seeds.pdf",
            "visual_dirt.pdf",
            "visual_logs.pdf",
            "visual_max_distance.pdf",
            "visual_seeds.pdf",
            "group_01_prompt_aligned.pdf",
            "group_02_prompt_aligned.pdf",
            "group_03_prompt_aligned.pdf",
            "group_04_prompt_aligned.pdf",
            "reward_evolution_equal_spacing.pdf",
            "tree_logs_model_means_bar_vertical.pdf",
            "hit_mean_bar_settings_stacked_side_by_side.pdf",
            "kill_rate_pct_bar_settings_stacked_side_by_side.pdf",
            "seed49_first100_left_choice_layout_scatter_no_ci.pdf",
            "seed49_first100_left_choice_layout_scatter_with_ci.pdf",
            "seed481_first100_left_choice_layout_scatter_no_ci.pdf",
            "seed481_first100_left_choice_layout_scatter_with_ci.pdf",
            "agent_131_topdown_trajectories.pdf",
            "agent_132_topdown_trajectories.pdf",
            "agent_111_114_seed_sweep_side_facet_bars.pdf",
            "head_fake_pair_stability.pdf",
        ],
        "static_figures": {
            "figure1.pdf": str(REPO_ROOT / "outputs" / "plot_suite" / "static_overleaf_figures" / "figure1.pdf")
        },
    },
}


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return deepcopy(DEFAULT_CONFIG)
    with path.open("r", encoding="utf-8") as f:
        override = json.load(f)
    return deep_merge(DEFAULT_CONFIG, override)
