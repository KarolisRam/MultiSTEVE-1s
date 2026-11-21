from __future__ import annotations

import re
import os
from dataclasses import dataclass
from pathlib import Path


DEFAULT_CHECKPOINT_NAME = "STEVE-1_snapshot_160005120.weights"
DEFAULT_WEIGHTS_ROOT = Path(os.environ.get("MULTISTEVE_MODEL_ZOO_ROOT", "data/model_zoo")) / "05_final_init_and_data_100"

DEFAULT_RUNNER_ARGS = {
    "in_model": "data/weights/vpt/2x.model",
    "prior_weights": "data/weights/steve1/steve1_prior.pt",
    "text_cond_scale": 6.0,
    "visual_cond_scale": 7.0,
    "gameplay_length": 6060,
    "teleport_every": 60,
    "creative_command_wait_steps": 10,
    "teleport_wait_steps": 10,
    "teleport_feedback_mode": "gamerule_strict",
    "prompt": "chop tree",
    "suppress_attack_actions": True,
}


@dataclass(frozen=True)
class CampaignCondition:
    key: str
    env_seed: int
    villager_side: str
    source_dir: str
    description: str

    @property
    def regular_side(self) -> str:
        return "right" if self.villager_side == "left" else "left"

    @property
    def placement_args(self) -> list[str]:
        if self.villager_side == "left":
            return ["side=left,villager=true", "side=right,villager=false"]
        if self.villager_side == "right":
            return ["side=left,villager=false", "side=right,villager=true"]
        raise ValueError(f"Unknown villager_side: {self.villager_side}")


CONDITIONS: tuple[CampaignCondition, ...] = (
    CampaignCondition(
        key="seed049_villager_left",
        env_seed=49,
        villager_side="left",
        source_dir="data/external_tmaze_imports/seed049_villager_left",
        description="Environment seed 49, villager tree on left, regular tree on right.",
    ),
    CampaignCondition(
        key="seed049_villager_right",
        env_seed=49,
        villager_side="right",
        source_dir="data/external_tmaze_imports/seed049_villager_right",
        description="Environment seed 49, regular tree on left, villager tree on right.",
    ),
    CampaignCondition(
        key="seed481_villager_left",
        env_seed=481,
        villager_side="left",
        source_dir="data/external_tmaze_imports/seed481_villager_left",
        description="Environment seed 481, villager tree on left, regular tree on right.",
    ),
    CampaignCondition(
        key="seed481_villager_right",
        env_seed=481,
        villager_side="right",
        source_dir="data/external_tmaze_imports/seed481_villager_right",
        description="Environment seed 481, regular tree on left, villager tree on right.",
    ),
)

CONDITION_BY_KEY = {condition.key: condition for condition in CONDITIONS}
CONDITION_ORDER = {condition.key: i for i, condition in enumerate(CONDITIONS)}

CORRECTED_IMPORT_OVERRIDES = {
    "seed049_villager_right": {
        "dataset_seed122-weight_init_seed122": "data/external_tmaze_imports/seed049_villager_right_agent122_corrected",
    },
    "seed481_villager_left": {
        "dataset_seed101-weight_init_seed101": "data/external_tmaze_imports/seed481_villager_left_agent101_corrected",
    },
}


def model_seed(model_id: str) -> int | None:
    match = re.search(r"bitflip_seed(\d+)", model_id) or re.search(r"dataset_seed(\d+)", model_id)
    if not match:
        return None
    return int(match.group(1))


def model_sort_key(model_id: str) -> tuple[int, str]:
    seed = model_seed(model_id)
    if seed is None:
        return (10**9, model_id)
    return (seed, model_id)


def discover_weights(roots: list[Path], checkpoint_name: str = DEFAULT_CHECKPOINT_NAME) -> dict[str, Path]:
    weights: dict[str, Path] = {}
    for root in roots:
        root = root.expanduser().resolve()
        if not root.exists():
            continue
        for weight_path in sorted(root.glob(f"*/{checkpoint_name}")):
            weights[weight_path.parent.name] = weight_path.resolve()
    return dict(sorted(weights.items(), key=lambda item: model_sort_key(item[0])))


def condition_manifest_path(campaign_root: Path, condition_key: str) -> Path:
    return campaign_root / "conditions" / condition_key / "manifest.csv"


def condition_runs_dir(campaign_root: Path, condition_key: str) -> Path:
    return campaign_root / "conditions" / condition_key / "runs"
