import argparse
import csv
import itertools
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from steve1.VPT.lib.action_mapping import CameraHierarchicalMapping

REPO_ROOT = Path(os.environ.get("MULTISTEVE_REPO_ROOT", Path(__file__).resolve().parents[2])).resolve()
OUTPUT_ROOT = Path(
    os.environ.get(
        "MULTISTEVE_ACTION_PROB_OUTPUT_ROOT",
        "outputs/action_probabilities/model_group_comparisons/fake_env_main_episode_05_setting_sweep",
    )
).resolve()
FAKE_ENV_ZIP = Path(
    os.environ.get("MULTISTEVE_FAKE_ENV_ZIP", "data/fake_env_rollouts/main_episode_observations.zip")
).resolve()
MODEL_ZOO_ROOT = Path(os.environ.get("MULTISTEVE_MODEL_ZOO_ROOT", "data/model_zoo")).resolve()
IN_MODEL = "data/weights/vpt/2x.model"
PRIOR_WEIGHTS = "data/weights/steve1/steve1_prior.pt"
PROMPT = "chop a tree"
PROMPT_SEED = 13
ROLLOUT_SEED = 49
GAMEPLAY_LENGTH = 1000
SNAPSHOT_NAME = "STEVE-1_snapshot_160005120.weights"

DEFAULT_SCREEN_WIDTH = 1900
DEFAULT_SCREEN_HEIGHT = 1080
DEFAULT_DPI = 100

EXACT_ACTION_METRIC = "exact_full_action_agreement"
SEMANTIC_METRIC = "semantic_button_group_agreement"
CAMERA_METRIC = "camera_action_agreement"
METRIC_ORDER = [EXACT_ACTION_METRIC, SEMANTIC_METRIC, CAMERA_METRIC]
METRIC_TITLES = {
    EXACT_ACTION_METRIC: "Exact Full-Action Agreement",
    SEMANTIC_METRIC: "Semantic Button-Group Agreement",
    CAMERA_METRIC: "Camera-Action Agreement",
}
METRIC_COLORS = {
    EXACT_ACTION_METRIC: "#1f556c",
    SEMANTIC_METRIC: "#d86c45",
    CAMERA_METRIC: "#3f7f4c",
}
METRIC_SHORT_NAMES = {
    EXACT_ACTION_METRIC: "exact_action",
    SEMANTIC_METRIC: "semantic_buttons",
    CAMERA_METRIC: "camera_action",
}

FAMILY_SPECS = {
    "01_init_seed_only": {
        "models_root": (MODEL_ZOO_ROOT / "01_init_seed_only").resolve(),
        "display_name": "01 Init Seed Only",
    },
    "02_data_seed_only": {
        "models_root": (MODEL_ZOO_ROOT / "02_data_seed_only").resolve(),
        "display_name": "02 Data Seed Only",
    },
    "03_init_and_data": {
        "models_root": (MODEL_ZOO_ROOT / "03_init_and_data").resolve(),
        "display_name": "03 Init And Data",
    },
    "04_single_lsb_flip": {
        "models_root": (MODEL_ZOO_ROOT / "04_single_lsb_flip").resolve(),
        "display_name": "04 Single LSB Flip",
    },
}


@dataclass(frozen=True)
class ModelSpec:
    family_name: str
    family_display_name: str
    label: str
    model_dir: str
    weights_path: str
    model_dir_name: str


def _natural_sort_key(value: str):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value)]


def _extract_seed(root_name: str, field_name: str) -> int | None:
    match = re.search(rf"{field_name}(\d+)", root_name)
    if match is None:
        return None
    return int(match.group(1))


def _label_for_model(family_name: str, model_dir_name: str) -> str:
    dataset_seed = _extract_seed(model_dir_name, "dataset_seed")
    weight_seed = _extract_seed(model_dir_name, "weight_init_seed")
    bitflip_seed = _extract_seed(model_dir_name, "bitflip_seed")

    if family_name == "01_init_seed_only":
        return f"w{weight_seed}"
    if family_name == "02_data_seed_only":
        return f"d{dataset_seed}"
    if family_name == "03_init_and_data":
        return f"s{dataset_seed}"
    if family_name == "04_single_lsb_flip":
        return f"b{bitflip_seed}"
    return model_dir_name


def _discover_model_specs(family_name: str, snapshot_name: str) -> tuple[list[ModelSpec], list[dict]]:
    family_spec = FAMILY_SPECS[family_name]
    models_root = family_spec["models_root"]
    model_dirs = sorted(
        [path for path in models_root.iterdir() if path.is_dir()],
        key=lambda path: _natural_sort_key(path.name),
    )

    model_specs = []
    skipped = []
    for model_dir in model_dirs:
        weight_path = model_dir / snapshot_name
        if not weight_path.exists():
            skipped.append(
                {
                    "model_dir": str(model_dir),
                    "model_dir_name": model_dir.name,
                    "reason": f"missing {snapshot_name}",
                }
            )
            continue
        model_specs.append(
            ModelSpec(
                family_name=family_name,
                family_display_name=family_spec["display_name"],
                label=_label_for_model(family_name, model_dir.name),
                model_dir=str(model_dir),
                weights_path=str(weight_path),
                model_dir_name=model_dir.name,
            )
        )
    return model_specs, skipped


def _resolve_run_dir(path_str: str | Path) -> Path:
    path = Path(path_str).resolve()
    if (path / "buttons.npy").exists() and (path / "camera.npy").exists():
        return path
    candidate_children = []
    if path.exists():
        for child in sorted(path.iterdir()):
            if child.is_dir() and (child / "buttons.npy").exists() and (child / "camera.npy").exists():
                candidate_children.append(child)
    if len(candidate_children) != 1:
        raise ValueError(f"Expected exactly one rollout run directory under {path}, found {len(candidate_children)}")
    return candidate_children[0]


def _python_env() -> dict[str, str]:
    env = dict(os.environ)
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = f"{REPO_ROOT}{':' + existing if existing else ''}"
    return env


def _launch_rollout(model_spec: ModelSpec, family_rollouts_dir: Path) -> dict:
    rollout_root = family_rollouts_dir / model_spec.label
    rollout_root.mkdir(parents=True, exist_ok=True)

    try:
        run_dir = _resolve_run_dir(rollout_root)
        status = "reused_existing"
    except ValueError:
        log_path = rollout_root / "launcher.log"
        cmd = [
            sys.executable,
            "-m",
            "steve1.run_agent.run_agent_with_logging",
            "--in_model",
            IN_MODEL,
            "--in_weights",
            model_spec.weights_path,
            "--prior_weights",
            PRIOR_WEIGHTS,
            "--custom_text_prompt",
            PROMPT,
            "--gameplay_length",
            str(GAMEPLAY_LENGTH),
            "--save_dirpath",
            str(rollout_root),
            "--seed",
            str(ROLLOUT_SEED),
            "--save_additional_data",
            "--fake_env_zip",
            str(FAKE_ENV_ZIP),
            "--custom_text_seed",
            str(PROMPT_SEED),
        ]
        start_time = time.time()
        with open(log_path, "a") as log_file:
            log_file.write(f"[{datetime.now().isoformat()}] Launching rollout\n")
            log_file.write(" ".join(cmd) + "\n\n")
            log_file.flush()
            subprocess.run(cmd, cwd=REPO_ROOT, env=_python_env(), check=True, stdout=log_file, stderr=subprocess.STDOUT)
            log_file.write(f"\n[{datetime.now().isoformat()}] Completed in {time.time() - start_time:.2f}s\n")
        run_dir = _resolve_run_dir(rollout_root)
        status = "ran_new"

    metadata = {
        "family_name": model_spec.family_name,
        "family_display_name": model_spec.family_display_name,
        "label": model_spec.label,
        "model_dir": model_spec.model_dir,
        "weights_path": model_spec.weights_path,
        "rollout_root": str(rollout_root),
        "run_dir": str(run_dir),
        "status": status,
    }
    (rollout_root / "model_spec.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))
    return metadata


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits.astype(np.float64) - np.max(logits, axis=-1, keepdims=True)
    np.exp(shifted, out=shifted)
    shifted /= np.sum(shifted, axis=-1, keepdims=True)
    return shifted.astype(np.float32)


def _button_group_matrices():
    mapping = CameraHierarchicalMapping(n_camera_bins=11)
    base_groups = dict(mapping.BUTTONS_GROUPS)
    semantic_groups = dict(base_groups)
    semantic_groups["inventory"] = ["none", "inventory"]

    num_button_classes = len(mapping.BUTTONS_IDX_TO_COMBINATION)
    matrices = {}
    option_indices = {}
    for group_name, options in semantic_groups.items():
        matrices[group_name] = np.zeros((num_button_classes, len(options)), dtype=np.float32)
        option_indices[group_name] = {option: idx for idx, option in enumerate(options)}

    camera_on_mask = np.zeros(num_button_classes, dtype=bool)
    for button_idx in range(num_button_classes):
        combination = mapping.BUTTONS_IDX_TO_COMBINATION[button_idx]
        if combination == "inventory":
            for group_name in semantic_groups.keys():
                chosen_option = "inventory" if group_name == "inventory" else "none"
                matrices[group_name][button_idx, option_indices[group_name][chosen_option]] = 1.0
            continue

        for group_name, options in base_groups.items():
            group_index = list(base_groups.keys()).index(group_name)
            chosen_option = combination[group_index]
            matrices[group_name][button_idx, option_indices[group_name][chosen_option]] = 1.0
            if group_name == "camera" and chosen_option == "camera":
                camera_on_mask[button_idx] = True
        matrices["inventory"][button_idx, option_indices["inventory"]["none"]] = 1.0

    return {
        "semantic_groups": semantic_groups,
        "matrices": matrices,
        "camera_on_mask": camera_on_mask,
        "camera_off_mask": ~camera_on_mask,
        "camera_on_option_index": option_indices["camera"]["camera"],
    }


def _load_model_probs(run_dir: str | Path, group_info: dict) -> dict:
    buttons_logits = np.load(Path(run_dir) / "buttons.npy", allow_pickle=True)
    camera_logits = np.load(Path(run_dir) / "camera.npy", allow_pickle=True)
    if buttons_logits.shape[0] != camera_logits.shape[0]:
        raise ValueError(f"Buttons/camera length mismatch in {run_dir}: {buttons_logits.shape} vs {camera_logits.shape}")

    button_probs = _softmax(buttons_logits)
    camera_probs = _softmax(camera_logits)
    semantic_group_probs = {
        group_name: button_probs @ matrix for group_name, matrix in group_info["matrices"].items()
    }
    return {
        "button_probs": button_probs,
        "camera_probs": camera_probs,
        "semantic_group_probs": semantic_group_probs,
        "num_steps": int(button_probs.shape[0]),
    }


def _exact_full_action_agreement(model_a: dict, model_b: dict, group_info: dict) -> np.ndarray:
    on_mask = group_info["camera_on_mask"]
    off_mask = group_info["camera_off_mask"]
    camera_match = np.sum(model_a["camera_probs"] * model_b["camera_probs"], axis=-1, dtype=np.float64)
    button_off_agreement = np.sum(
        model_a["button_probs"][:, off_mask] * model_b["button_probs"][:, off_mask],
        axis=-1,
        dtype=np.float64,
    )
    button_on_agreement = np.sum(
        model_a["button_probs"][:, on_mask] * model_b["button_probs"][:, on_mask],
        axis=-1,
        dtype=np.float64,
    )
    return button_off_agreement + button_on_agreement * camera_match


def _semantic_button_group_agreement(model_a: dict, model_b: dict, group_info: dict) -> np.ndarray:
    agreements = []
    for group_name in group_info["semantic_groups"].keys():
        p = model_a["semantic_group_probs"][group_name]
        q = model_b["semantic_group_probs"][group_name]
        agreements.append(np.sum(p * q, axis=-1, dtype=np.float64))
    return np.mean(np.stack(agreements, axis=0), axis=0)


def _camera_action_agreement(model_a: dict, model_b: dict, group_info: dict) -> np.ndarray:
    camera_on_idx = group_info["camera_on_option_index"]
    camera_on_a = model_a["semantic_group_probs"]["camera"][:, camera_on_idx]
    camera_on_b = model_b["semantic_group_probs"]["camera"][:, camera_on_idx]
    camera_match = np.sum(model_a["camera_probs"] * model_b["camera_probs"], axis=-1, dtype=np.float64)
    return (1.0 - camera_on_a) * (1.0 - camera_on_b) + camera_on_a * camera_on_b * camera_match


def _metric_series_for_pair(model_a: dict, model_b: dict, group_info: dict) -> dict[str, np.ndarray]:
    return {
        EXACT_ACTION_METRIC: _exact_full_action_agreement(model_a, model_b, group_info),
        SEMANTIC_METRIC: _semantic_button_group_agreement(model_a, model_b, group_info),
        CAMERA_METRIC: _camera_action_agreement(model_a, model_b, group_info),
    }


def _plot_dimensions():
    return DEFAULT_SCREEN_WIDTH * 0.9 / DEFAULT_DPI, DEFAULT_SCREEN_HEIGHT * 0.9 / DEFAULT_DPI, DEFAULT_DPI


def _write_matrix_csv(output_path: Path, matrix: np.ndarray, labels: list[str]) -> None:
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model"] + labels)
        for label, row in zip(labels, matrix):
            writer.writerow([label] + [f"{value:.8f}" for value in row])


def _plot_annotated_matrix(matrix: np.ndarray, labels: list[str], title: str, output_path: Path) -> None:
    width_in, height_in, dpi = _plot_dimensions()
    fig, ax = plt.subplots(figsize=(width_in, height_in), dpi=dpi)
    im = ax.imshow(matrix, cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_title(title)
    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            value = matrix[row_idx, col_idx]
            text_color = "white" if value < 0.55 else "black"
            ax.text(col_idx, row_idx, f"{value:.3f}", ha="center", va="center", color=text_color, fontsize=9)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Agreement")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_mean_pairwise_series(metric_series_by_name: dict[str, np.ndarray], output_path: Path) -> None:
    width_in, height_in, dpi = _plot_dimensions()
    fig, ax = plt.subplots(figsize=(width_in, height_in), dpi=dpi)
    frames = np.arange(next(iter(metric_series_by_name.values())).shape[1])
    for metric_name in METRIC_ORDER:
        series = metric_series_by_name[metric_name]
        mean = np.mean(series, axis=0)
        std = np.std(series, axis=0)
        ax.plot(frames, mean, color=METRIC_COLORS[metric_name], linewidth=2.2, label=METRIC_TITLES[metric_name])
        ax.fill_between(frames, mean - std, mean + std, color=METRIC_COLORS[metric_name], alpha=0.16)
    ax.set_xlim(0, len(frames) - 1)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Pairwise agreement")
    ax.set_title("Mean Pairwise Agreement Over Time")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_pairwise_heatmap(series: np.ndarray, pair_labels: list[str], title: str, output_path: Path) -> None:
    width_in, height_in, dpi = _plot_dimensions()
    fig, ax = plt.subplots(figsize=(width_in, height_in), dpi=dpi)
    im = ax.imshow(series, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0, interpolation="nearest")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Model pair")
    ax.set_title(title)
    ax.set_yticks(np.arange(len(pair_labels)))
    ax.set_yticklabels(pair_labels, fontsize=8)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Agreement")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _write_mean_series_csv(output_path: Path, metric_series_by_name: dict[str, np.ndarray]) -> None:
    frames = np.arange(next(iter(metric_series_by_name.values())).shape[1])
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "frame",
                "exact_action_mean",
                "exact_action_std",
                "semantic_buttons_mean",
                "semantic_buttons_std",
                "camera_action_mean",
                "camera_action_std",
            ]
        )
        for frame_idx in frames:
            row = [frame_idx]
            for metric_name in METRIC_ORDER:
                values = metric_series_by_name[metric_name][:, frame_idx]
                row.extend([f"{float(np.mean(values)):.8f}", f"{float(np.std(values)):.8f}"])
            writer.writerow(row)


def _write_pairwise_series_csv(output_path: Path, series: np.ndarray, pair_labels: list[str]) -> None:
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["pair"] + [f"frame_{idx}" for idx in range(series.shape[1])])
        for pair_label, values in zip(pair_labels, series):
            writer.writerow([pair_label] + [f"{float(value):.8f}" for value in values])


def _summary_stats(metric_series_by_name: dict[str, np.ndarray]) -> dict:
    summary = {}
    for metric_name, series in metric_series_by_name.items():
        pair_means = np.mean(series, axis=1)
        summary[metric_name] = {
            "mean_pairwise": float(np.mean(series)),
            "std_pairwise": float(np.std(series)),
            "min_pairwise": float(np.min(pair_means)),
            "max_pairwise": float(np.max(pair_means)),
            "pairwise_means": [float(value) for value in pair_means],
        }
    return summary


def _pair_labels(model_labels: list[str]) -> list[str]:
    return [f"{model_labels[i]} vs {model_labels[j]}" for i, j in itertools.combinations(range(len(model_labels)), 2)]


def _compare_family_rollouts(family_name: str, rollout_metadata: list[dict], output_dir: Path) -> dict:
    group_info = _button_group_matrices()
    model_labels = [item["label"] for item in rollout_metadata]
    run_dirs = [item["run_dir"] for item in rollout_metadata]
    loaded_models = [_load_model_probs(run_dir, group_info) for run_dir in run_dirs]
    num_steps = loaded_models[0]["num_steps"]
    for loaded_model, run_dir in zip(loaded_models, run_dirs):
        if loaded_model["num_steps"] != num_steps:
            raise ValueError(f"Rollout length mismatch in {run_dir}")

    matrices_dir = output_dir / "matrices"
    plots_grids_dir = output_dir / "plots" / "grids"
    plots_timelines_dir = output_dir / "plots" / "timelines"
    series_dir = output_dir / "series"
    matrices_dir.mkdir(parents=True, exist_ok=True)
    plots_grids_dir.mkdir(parents=True, exist_ok=True)
    plots_timelines_dir.mkdir(parents=True, exist_ok=True)
    series_dir.mkdir(parents=True, exist_ok=True)

    num_models = len(loaded_models)
    pair_indices = list(itertools.combinations(range(num_models), 2))
    pair_labels = _pair_labels(model_labels)
    metric_matrices = {metric_name: np.eye(num_models, dtype=np.float64) for metric_name in METRIC_ORDER}
    metric_pairwise_series = {
        metric_name: np.zeros((len(pair_indices), num_steps), dtype=np.float64) for metric_name in METRIC_ORDER
    }

    for pair_idx, (i, j) in enumerate(pair_indices):
        pair_metrics = _metric_series_for_pair(loaded_models[i], loaded_models[j], group_info)
        for metric_name, values in pair_metrics.items():
            metric_pairwise_series[metric_name][pair_idx] = values
            metric_matrices[metric_name][i, j] = float(np.mean(values))
            metric_matrices[metric_name][j, i] = float(np.mean(values))

    for metric_name in METRIC_ORDER:
        short_name = METRIC_SHORT_NAMES[metric_name]
        _write_matrix_csv(matrices_dir / f"{short_name}_matrix.csv", metric_matrices[metric_name], model_labels)
        _plot_annotated_matrix(
            metric_matrices[metric_name],
            model_labels,
            METRIC_TITLES[metric_name],
            plots_grids_dir / f"{short_name}_matrix.png",
        )
        _write_pairwise_series_csv(
            series_dir / f"{short_name}_pairwise_over_time.csv",
            metric_pairwise_series[metric_name],
            pair_labels,
        )
        np.save(series_dir / f"{short_name}_pairwise_over_time.npy", metric_pairwise_series[metric_name], allow_pickle=False)
        _plot_pairwise_heatmap(
            metric_pairwise_series[metric_name],
            pair_labels,
            f"{METRIC_TITLES[metric_name]} Over Time",
            plots_timelines_dir / f"{short_name}_pairwise_over_time_heatmap.png",
        )

    _write_mean_series_csv(series_dir / "mean_pairwise_agreement_over_time.csv", metric_pairwise_series)
    _plot_mean_pairwise_series(metric_pairwise_series, plots_timelines_dir / "mean_pairwise_agreement_over_time.png")

    summary = _summary_stats(metric_pairwise_series)
    summary["family_name"] = family_name
    summary["family_display_name"] = rollout_metadata[0]["family_display_name"]
    summary["model_labels"] = model_labels
    summary["pair_labels"] = pair_labels
    summary["num_models"] = len(model_labels)
    summary["num_steps"] = num_steps
    summary["models"] = rollout_metadata
    summary["semantic_groups"] = {group_name: list(options) for group_name, options in group_info["semantic_groups"].items()}

    summary_json_path = output_dir / "summary.json"
    summary_txt_path = output_dir / "summary.txt"
    summary_json_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    summary_txt_path.write_text(
        "\n".join(
            [
                f"{summary['family_display_name']} Agreement Summary",
                f"Models: {', '.join(model_labels)}",
                "",
                f"Exact full-action agreement: {summary[EXACT_ACTION_METRIC]['mean_pairwise']:.4f}",
                f"Semantic button-group agreement: {summary[SEMANTIC_METRIC]['mean_pairwise']:.4f}",
                f"Camera-action agreement: {summary[CAMERA_METRIC]['mean_pairwise']:.4f}",
            ]
        )
        + "\n"
    )

    manifest = {
        "created_at": datetime.now().isoformat(),
        "family_name": family_name,
        "family_display_name": summary["family_display_name"],
        "output_dir": str(output_dir),
        "summary_json": str(summary_json_path),
        "summary_txt": str(summary_txt_path),
        "models": rollout_metadata,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return summary


def _plot_cross_family_metric_bars(family_summaries: list[dict], output_path: Path) -> None:
    width_in, height_in, dpi = _plot_dimensions()
    fig, ax = plt.subplots(figsize=(width_in, height_in), dpi=dpi)
    family_labels = [item["family_name"].split("_", 1)[0] for item in family_summaries]
    x = np.arange(len(family_labels))
    bar_width = 0.22

    for idx, metric_name in enumerate(METRIC_ORDER):
        values = [item[metric_name]["mean_pairwise"] for item in family_summaries]
        offset = (idx - 1) * bar_width
        ax.bar(x + offset, values, width=bar_width, color=METRIC_COLORS[metric_name], label=METRIC_TITLES[metric_name])
        for xpos, value in zip(x + offset, values):
            ax.text(xpos, value + 0.006, f"{value:.3f}", ha="center", va="bottom", fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels(family_labels)
    ax.set_ylim(0.0, 1.02)
    ax.set_ylabel("Mean pairwise agreement")
    ax.set_title("Agreement Comparison Across Model Families")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _write_cross_family_summary(output_dir: Path, family_summaries: list[dict], skipped_by_family: dict[str, list[dict]]) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    for summary in family_summaries:
        family_name = summary["family_name"]
        summary_rows.append(
            {
                "family_name": family_name,
                "family_display_name": summary["family_display_name"],
                "num_models": summary["num_models"],
                "num_pairs": len(summary["pair_labels"]),
                "skipped_count": len(skipped_by_family.get(family_name, [])),
                "exact_full_action_agreement": summary[EXACT_ACTION_METRIC]["mean_pairwise"],
                "semantic_button_group_agreement": summary[SEMANTIC_METRIC]["mean_pairwise"],
                "camera_action_agreement": summary[CAMERA_METRIC]["mean_pairwise"],
                "models": summary["model_labels"],
            }
        )

    csv_path = output_dir / "family_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "family_name",
                "family_display_name",
                "num_models",
                "num_pairs",
                "skipped_count",
                "exact_full_action_agreement",
                "semantic_button_group_agreement",
                "camera_action_agreement",
                "models",
            ]
        )
        for row in summary_rows:
            writer.writerow(
                [
                    row["family_name"],
                    row["family_display_name"],
                    row["num_models"],
                    row["num_pairs"],
                    row["skipped_count"],
                    f"{row['exact_full_action_agreement']:.8f}",
                    f"{row['semantic_button_group_agreement']:.8f}",
                    f"{row['camera_action_agreement']:.8f}",
                    " ".join(row["models"]),
                ]
            )

    summary_json = {
        "created_at": datetime.now().isoformat(),
        "source_fake_env_zip": str(FAKE_ENV_ZIP),
        "prompt": PROMPT,
        "prompt_seed": PROMPT_SEED,
        "rollout_seed": ROLLOUT_SEED,
        "snapshot_name": SNAPSHOT_NAME,
        "families": summary_rows,
        "skipped_models": skipped_by_family,
    }
    json_path = output_dir / "family_summary.json"
    json_path.write_text(json.dumps(summary_json, indent=2, sort_keys=True))

    txt_lines = [
        "Fake Rollout Setting Sweep Summary",
        f"Source fake env zip: {FAKE_ENV_ZIP}",
        f"Prompt: {PROMPT!r}",
        f"Prompt seed: {PROMPT_SEED}",
        f"Rollout seed: {ROLLOUT_SEED}",
        f"Checkpoint: {SNAPSHOT_NAME}",
        "",
    ]
    for row in summary_rows:
        txt_lines.append(f"{row['family_display_name']} ({row['num_models']} models, {row['num_pairs']} pairs)")
        txt_lines.append(f"  Exact full-action agreement: {row['exact_full_action_agreement']:.4f}")
        txt_lines.append(f"  Semantic button-group agreement: {row['semantic_button_group_agreement']:.4f}")
        txt_lines.append(f"  Camera-action agreement: {row['camera_action_agreement']:.4f}")
        if skipped_by_family.get(row["family_name"]):
            skipped_names = ", ".join(item["model_dir_name"] for item in skipped_by_family[row["family_name"]])
            txt_lines.append(f"  Skipped: {skipped_names}")
        txt_lines.append("")
    txt_path = output_dir / "summary.txt"
    txt_path.write_text("\n".join(txt_lines))

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    _plot_cross_family_metric_bars(family_summaries, plots_dir / "family_metric_comparison.png")

    manifest = {
        "created_at": datetime.now().isoformat(),
        "family_summary_csv": str(csv_path),
        "family_summary_json": str(json_path),
        "summary_txt": str(txt_path),
        "plots": {"family_metric_comparison_png": str(plots_dir / "family_metric_comparison.png")},
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    summary_json["manifest_path"] = str(manifest_path)
    return summary_json


def run_sweep(output_dir: Path) -> dict:
    rollouts_root = output_dir / "rollouts"
    comparisons_root = output_dir / "comparisons"
    summary_root = output_dir / "summary"
    rollouts_root.mkdir(parents=True, exist_ok=True)
    comparisons_root.mkdir(parents=True, exist_ok=True)
    summary_root.mkdir(parents=True, exist_ok=True)

    all_rollout_metadata = {}
    family_summaries = []
    skipped_by_family = {}

    for family_name in FAMILY_SPECS.keys():
        model_specs, skipped = _discover_model_specs(family_name, SNAPSHOT_NAME)
        skipped_by_family[family_name] = skipped
        family_rollouts_dir = rollouts_root / family_name
        family_rollouts_dir.mkdir(parents=True, exist_ok=True)

        rollout_metadata = []
        for model_spec in model_specs:
            print(f"[rollout] {family_name} {model_spec.label} -> {model_spec.weights_path}", flush=True)
            rollout_metadata.append(_launch_rollout(model_spec, family_rollouts_dir))
        all_rollout_metadata[family_name] = rollout_metadata

        family_comparison_dir = comparisons_root / family_name
        print(f"[compare] {family_name} ({len(rollout_metadata)} models)", flush=True)
        family_summaries.append(_compare_family_rollouts(family_name, rollout_metadata, family_comparison_dir))

    aggregate_summary = _write_cross_family_summary(summary_root, family_summaries, skipped_by_family)
    top_manifest = {
        "created_at": datetime.now().isoformat(),
        "output_dir": str(output_dir),
        "rollouts_root": str(rollouts_root),
        "comparisons_root": str(comparisons_root),
        "summary_root": str(summary_root),
        "source_fake_env_zip": str(FAKE_ENV_ZIP),
        "prompt": PROMPT,
        "prompt_seed": PROMPT_SEED,
        "rollout_seed": ROLLOUT_SEED,
        "snapshot_name": SNAPSHOT_NAME,
        "families": {family_name: rollout_metadata for family_name, rollout_metadata in all_rollout_metadata.items()},
        "aggregate_summary_manifest": str(summary_root / "manifest.json"),
    }
    (output_dir / "manifest.json").write_text(json.dumps(top_manifest, indent=2, sort_keys=True))
    return aggregate_summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=Path, default=None)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve() if args.output_dir else OUTPUT_ROOT / datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = run_sweep(output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
