from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ENGINEERED_WOOD_PROMPT = "chop down the tree, gather wood, pick up wood, chop it down, break tree"
DEFAULT_ENV_SEEDS = [49, 2, 3, 86, 129, 135, 139, 147, 148, 195]
CHECKPOINT_NAME = "STEVE-1_snapshot_160005120.weights"


def base_campaign(*, name: str, episodes_root: str) -> dict[str, Any]:
    return {
        "name": name,
        "queue_db_path": f"{episodes_root}/_queue/state.sqlite3",
        "state_root": f"{episodes_root}/_queue",
        "episodes_root": episodes_root,
        "runner_script": "steve1/run_agent/run_agent_with_logging_teleport.py",
        "default_episode_seeds": [49],
        "common_args": {
            "in_model": "data/weights/vpt/2x.model",
            "prior_weights": "data/weights/steve1/steve1_prior.pt",
            "gameplay_length": 1000,
            "text_cond_scale": 6.0,
            "visual_cond_scale": 7.0,
            "save_additional_data": False,
            "fake_env_zip": None,
            "wipe_hidden_state_each_step": False,
            "placement": [],
            "extra_args": ["--teleport_to_spawn_every", "0"],
        },
        "runtime": {
            "conda_env": "steve-1-creative",
            "heartbeat_seconds": 30,
            "lease_timeout_seconds": 1200,
            "missing_weights_retry_seconds": 900,
            "failed_retry_seconds": 300,
            "failed_max_retries": -1,
            "idle_sleep_seconds": 5,
        },
        "metadata": {},
        "experiments": [],
    }


def explicit_model(label: str, group: str) -> dict[str, str]:
    return {
        "label": label,
        "weights_path": f"data/model_zoo/{group}/{label}/{CHECKPOINT_NAME}",
    }


def original_model() -> dict[str, str]:
    return {"label": "Original", "weights_path": "data/weights/steve1/steve1.weights"}


def smoke_model(label: str, group: str) -> dict[str, str]:
    return explicit_model(label, group)


def release_smoke_17_episodes() -> dict[str, Any]:
    campaign = base_campaign(
        name="release_smoke_17_episodes",
        episodes_root="data/smoke-evals",
    )
    campaign["runner_script"] = "steve1/run_agent/run_agent_with_logging_teleport.py"
    campaign["default_episode_seeds"] = [49]
    campaign["common_args"]["gameplay_length"] = 120
    campaign["common_args"]["extra_args"] = ["--teleport_to_spawn_every", "0"]
    campaign["metadata"] = {
        "description": "Short path-coverage smoke campaign; not a statistical reproduction.",
        "episode_budget": 17,
        "expected_runtime_scope": "one model, one prompt, one environment seed, short rollouts",
    }

    model_ours = smoke_model("dataset_seed1-weight_init_seed1", "03_init_and_data")
    agent_101 = smoke_model("dataset_seed101-weight_init_seed101", "05_final_init_and_data_100")
    agent_111 = smoke_model("dataset_seed111-weight_init_seed111", "05_final_init_and_data_100")
    agent_114 = smoke_model("dataset_seed114-weight_init_seed114", "05_final_init_and_data_100")
    agent_131 = smoke_model("dataset_seed131-weight_init_seed131", "05_final_init_and_data_100")
    agent_132 = smoke_model("dataset_seed132-weight_init_seed132", "05_final_init_and_data_100")
    bitflip_2 = smoke_model("dataset_seed1-weight_init_seed1-bitflip_seed2", "04_single_lsb_flip")
    bitflip_8 = smoke_model("dataset_seed1-weight_init_seed1-bitflip_seed8", "04_single_lsb_flip")

    def add_text(
        name: str,
        model: dict[str, str],
        prompt_name: str,
        prompt_text: str,
        *,
        seed: int = 49,
        placement: list[str] | None = None,
        save_additional_data: bool = False,
    ) -> None:
        override: dict[str, Any] = {}
        if placement is not None:
            override["placement"] = placement
        if save_additional_data:
            override["save_additional_data"] = True
        campaign["experiments"].append(
            {
                "name": name,
                "output_dir": f"release_smoke/{name}",
                "episode_seeds": [seed],
                "text_custom_seed_mode": "fixed",
                "text_custom_seed": 13,
                "expected_text_prompt_count": 1,
                "expected_visual_prompt_count": 0,
                "model_source": {"type": "explicit_models", "models": [model]},
                "text_prompts": [{"name": prompt_name, "text": prompt_text}],
                "visual_prompts": [],
                **({"common_args_override": override} if override else {}),
            }
        )

    def add_visual(name: str, model: dict[str, str], visual_prompt: str, *, seed: int = 49) -> None:
        campaign["experiments"].append(
            {
                "name": name,
                "output_dir": f"release_smoke/{name}",
                "episode_seeds": [seed],
                "expected_text_prompt_count": 0,
                "expected_visual_prompt_count": 1,
                "model_source": {"type": "explicit_models", "models": [model]},
                "text_prompts": [],
                "visual_prompts": [visual_prompt],
            }
        )

    add_text("01_original_text_wood", original_model(), "wood", "chop a tree")
    add_text("02_ours_text_wood", model_ours, "wood", "chop a tree")
    add_visual("03_ours_visual_wood", model_ours, "wood")
    add_text("04_partial_dirt_engineered", model_ours, "dirt_engineered", "get dirt, dig hole, dig dirt, gather a ton of dirt, collect dirt")
    add_text("05_group05_tree_prompt_agent101", agent_101, "wood", "chop a tree")
    add_text("06_attention_logging_agent101", agent_101, "wood", "chop a tree", save_additional_data=True)

    villager_left = ["side=left,villager=true,logs_above_villager=false"]
    villager_left_logs = ["side=left,villager=true,logs_above_villager=true"]
    add_text("07_villager_short_logs_false", agent_101, "wood", "chop a tree", placement=villager_left)
    add_text("08_villager_engineered_logs_false", agent_101, "wood_engineered", ENGINEERED_WOOD_PROMPT, placement=villager_left)
    add_text("09_villager_short_logs_true", agent_101, "wood", "chop a tree", placement=villager_left_logs)
    add_text("10_villager_engineered_logs_true", agent_101, "wood_engineered", ENGINEERED_WOOD_PROMPT, placement=villager_left_logs)

    tmaze_right = ["side=left,villager=false", "side=right,villager=true"]
    tmaze_left = ["side=left,villager=true", "side=right,villager=false"]
    add_text("11_tmaze_seed49_agent131_villager_right", agent_131, "wood", "chop a tree", placement=tmaze_right)
    add_text("12_tmaze_seed49_agent132_villager_right", agent_132, "wood", "chop a tree", placement=tmaze_right)
    add_text("13_tmaze_seed481_agent131_villager_left", agent_131, "wood", "chop a tree", seed=481, placement=tmaze_left)

    add_text("14_agent111_seed_sweep_path", agent_111, "wood", "chop a tree", placement=villager_left_logs)
    add_text("15_agent114_seed_sweep_path", agent_114, "wood", "chop a tree", placement=villager_left_logs)
    add_text("16_bitflip2_seed_sweep_path", bitflip_2, "wood", "chop a tree", placement=villager_left)
    add_text("17_bitflip8_seed_sweep_path", bitflip_8, "wood", "chop a tree", placement=villager_left)

    campaign["metadata"]["total_jobs"] = len(campaign["experiments"])
    return campaign


def seed_sweep_agent_111_114(repeats: int) -> dict[str, Any]:
    campaign = base_campaign(
        name="villager_agent_111_114_seed_sweep",
        episodes_root="data/evals-villager",
    )
    campaign["metadata"] = {
        "selected_seeds": DEFAULT_ENV_SEEDS,
        "repeats_per_agent_side_seed": repeats,
        "top_output_dir": "agent_111_114_short_logs_above_true_seed_sweep",
        "description": "Agent 111 vs Agent 114 villager-tree seed sweep.",
    }
    agents = [
        ("agent_111", "dataset_seed111-weight_init_seed111"),
        ("agent_114", "dataset_seed114-weight_init_seed114"),
    ]
    sides = [("01_left", "left"), ("02_right", "right")]
    for seed_idx, seed in enumerate(DEFAULT_ENV_SEEDS, start=1):
        for side_key, side in sides:
            for agent_key, model_label in agents:
                for rep in range(1, repeats + 1):
                    campaign["experiments"].append(
                        {
                            "name": f"seedseq_{seed_idx:02d}__seed_{seed:03d}__{side_key}__{agent_key}__rep_{rep:03d}",
                            "output_dir": (
                                "agent_111_114_short_logs_above_true_seed_sweep/"
                                f"seed_{seed:03d}/{side_key}/{agent_key}/rep_{rep:03d}"
                            ),
                            "episode_seeds": [seed],
                            "text_custom_seed_mode": "fixed",
                            "text_custom_seed": 13,
                            "expected_text_prompt_count": 1,
                            "expected_visual_prompt_count": 0,
                            "model_source": {
                                "type": "explicit_models",
                                "models": [explicit_model(model_label, "05_final_init_and_data_100")],
                                "skip_missing_weights": False,
                            },
                            "text_prompts": [{"name": "wood", "text": "chop a tree"}],
                            "visual_prompts": [],
                            "common_args_override": {
                                "placement": [f"side={side},villager=true,logs_above_villager=true"],
                            },
                        }
                    )
    campaign["metadata"]["total_jobs"] = len(campaign["experiments"])
    return campaign


def bitflip2_vs_8_sweep(repeats: int) -> dict[str, Any]:
    campaign = base_campaign(
        name="villager_bitflip2_vs_8_short_logs_above_false_seed_sweep",
        episodes_root="data/evals-villager",
    )
    campaign["metadata"] = {
        "selected_seeds": DEFAULT_ENV_SEEDS,
        "repeats_per_model_side_seed": repeats,
        "top_output_dir": "bitflip_seed2_vs_seed8_short_logs_above_false_seed_sweep",
        "description": "Optional bitflip seed 2 vs seed 8 villager-tree comparison.",
    }
    models = [
        ("model_01_bitflip_seed2", "dataset_seed1-weight_init_seed1-bitflip_seed2"),
        ("model_02_bitflip_seed8", "dataset_seed1-weight_init_seed1-bitflip_seed8"),
    ]
    sides = [("01_left", "left"), ("02_right", "right")]
    for seed_idx, seed in enumerate(DEFAULT_ENV_SEEDS, start=1):
        for side_key, side in sides:
            for model_key, model_label in models:
                bitflip_label = model_label.rsplit("-", 1)[-1]
                for rep in range(1, repeats + 1):
                    campaign["experiments"].append(
                        {
                            "name": (
                                f"bitflip_seed2_vs_seed8__seed_{seed:03d}__{side_key}__"
                                f"{model_key}__rep_{rep:03d}"
                            ),
                            "output_dir": (
                                "bitflip_seed2_vs_seed8_short_logs_above_false_seed_sweep/"
                                f"seed_{seed:03d}/{side_key}/{bitflip_label}/rep_{rep:03d}"
                            ),
                            "episode_seeds": [seed],
                            "text_custom_seed_mode": "fixed",
                            "text_custom_seed": 13,
                            "expected_text_prompt_count": 1,
                            "expected_visual_prompt_count": 0,
                            "model_source": {
                                "type": "explicit_models",
                                "models": [explicit_model(model_label, "04_single_lsb_flip")],
                                "skip_missing_weights": False,
                            },
                            "text_prompts": [{"name": "wood", "text": "chop a tree"}],
                            "visual_prompts": [],
                            "common_args_override": {
                                "placement": [f"side={side},villager=true,logs_above_villager=false"],
                            },
                        }
                    )
    campaign["metadata"]["total_jobs"] = len(campaign["experiments"])
    return campaign


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate large episode-queue campaign manifests from compact recipes.")
    parser.add_argument(
        "--campaign",
        choices=["agent_111_114_seed_sweep", "bitflip2_vs_8_sweep", "release_smoke_17_episodes"],
        required=True,
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=100)
    args = parser.parse_args()

    if args.repeats < 1:
        raise SystemExit("--repeats must be positive")

    if args.campaign == "agent_111_114_seed_sweep":
        payload = seed_sweep_agent_111_114(args.repeats)
    elif args.campaign == "bitflip2_vs_8_sweep":
        payload = bitflip2_vs_8_sweep(args.repeats)
    elif args.campaign == "release_smoke_17_episodes":
        payload = release_smoke_17_episodes()
    else:
        raise AssertionError(args.campaign)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    print(f"experiments={len(payload['experiments'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
