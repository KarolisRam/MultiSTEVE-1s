import argparse
import csv
import itertools
import json
import os
import re
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from steve1.VPT.lib.action_mapping import CameraHierarchicalMapping

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


def _softmax(logits):
    shifted = logits.astype(np.float64) - np.max(logits, axis=-1, keepdims=True)
    np.exp(shifted, out=shifted)
    shifted /= np.sum(shifted, axis=-1, keepdims=True)
    return shifted.astype(np.float32)


def _resolve_run_dir(path_str):
    path = Path(path_str).resolve()
    if (path / "buttons.npy").exists() and (path / "camera.npy").exists():
        return path
    candidate_children = []
    for child in sorted(path.iterdir()):
        if child.is_dir() and (child / "buttons.npy").exists() and (child / "camera.npy").exists():
            candidate_children.append(child)
    if len(candidate_children) != 1:
        raise ValueError(f"Expected exactly one rollout run directory under {path}, found {len(candidate_children)}")
    return candidate_children[0]


def _model_label_from_root(path_str):
    root_name = Path(path_str).name
    match = re.search(r"dataset_seed(\d+)-weight_init_seed(\d+)", root_name)
    if match is None:
        return root_name
    dataset_seed = match.group(1)
    weight_seed = match.group(2)
    if dataset_seed == weight_seed:
        return f"s{dataset_seed}"
    return f"d{dataset_seed}_w{weight_seed}"


def _model_metadata(path_str):
    root = Path(path_str).resolve()
    run_dir = _resolve_run_dir(path_str)
    return {
        "input_root": str(root),
        "run_dir": str(run_dir),
        "label": _model_label_from_root(path_str),
        "root_name": root.name,
        "run_name": run_dir.name,
    }


def _button_group_matrices():
    mapping = CameraHierarchicalMapping(n_camera_bins=11)
    base_groups = OrderedDict(mapping.BUTTONS_GROUPS)
    semantic_groups = OrderedDict(base_groups)
    semantic_groups["inventory"] = ["none", "inventory"]

    num_button_classes = len(mapping.BUTTONS_IDX_TO_COMBINATION)
    matrices = {}
    option_indices = {}
    for group_name, options in semantic_groups.items():
        matrices[group_name] = np.zeros((num_button_classes, len(options)), dtype=np.float32)
        option_indices[group_name] = {option: idx for idx, option in enumerate(options)}

    camera_on_mask = np.zeros(num_button_classes, dtype=bool)
    button_class_labels = []
    for button_idx in range(num_button_classes):
        combination = mapping.BUTTONS_IDX_TO_COMBINATION[button_idx]
        if combination == "inventory":
            button_class_labels.append("inventory")
            for group_name, options in semantic_groups.items():
                chosen_option = "inventory" if group_name == "inventory" else "none"
                matrices[group_name][button_idx, option_indices[group_name][chosen_option]] = 1.0
            continue

        button_class_labels.append("|".join(combination))
        for group_index, (group_name, options) in enumerate(base_groups.items()):
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
        "button_class_labels": button_class_labels,
    }


def _load_model_probs(run_dir, group_info):
    buttons_logits = np.load(Path(run_dir) / "buttons.npy", allow_pickle=True)
    camera_logits = np.load(Path(run_dir) / "camera.npy", allow_pickle=True)
    if buttons_logits.shape[0] != camera_logits.shape[0]:
        raise ValueError(f"Buttons/camera length mismatch in {run_dir}: {buttons_logits.shape} vs {camera_logits.shape}")

    button_probs = _softmax(buttons_logits)
    camera_probs = _softmax(camera_logits)
    semantic_group_probs = {
        group_name: button_probs @ matrix
        for group_name, matrix in group_info["matrices"].items()
    }

    return {
        "button_probs": button_probs,
        "camera_probs": camera_probs,
        "semantic_group_probs": semantic_group_probs,
        "num_steps": int(button_probs.shape[0]),
    }


def _exact_full_action_agreement(model_a, model_b, group_info):
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


def _semantic_button_group_agreement(model_a, model_b, group_info):
    agreements = []
    for group_name in group_info["semantic_groups"].keys():
        p = model_a["semantic_group_probs"][group_name]
        q = model_b["semantic_group_probs"][group_name]
        agreements.append(np.sum(p * q, axis=-1, dtype=np.float64))
    return np.mean(np.stack(agreements, axis=0), axis=0)


def _camera_action_agreement(model_a, model_b, group_info):
    camera_on_idx = group_info["camera_on_option_index"]
    camera_on_a = model_a["semantic_group_probs"]["camera"][:, camera_on_idx]
    camera_on_b = model_b["semantic_group_probs"]["camera"][:, camera_on_idx]
    camera_match = np.sum(model_a["camera_probs"] * model_b["camera_probs"], axis=-1, dtype=np.float64)
    return (1.0 - camera_on_a) * (1.0 - camera_on_b) + camera_on_a * camera_on_b * camera_match


def _metric_series_for_pair(model_a, model_b, group_info):
    return {
        EXACT_ACTION_METRIC: _exact_full_action_agreement(model_a, model_b, group_info),
        SEMANTIC_METRIC: _semantic_button_group_agreement(model_a, model_b, group_info),
        CAMERA_METRIC: _camera_action_agreement(model_a, model_b, group_info),
    }


def _plot_dimensions():
    return DEFAULT_SCREEN_WIDTH * 0.9 / DEFAULT_DPI, DEFAULT_SCREEN_HEIGHT * 0.9 / DEFAULT_DPI, DEFAULT_DPI


def _write_matrix_csv(output_path, matrix, labels):
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model"] + labels)
        for label, row in zip(labels, matrix):
            writer.writerow([label] + [f"{value:.8f}" for value in row])


def _plot_annotated_matrix(matrix, labels, title, output_path):
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


def _plot_mean_pairwise_series(metric_series_by_name, output_path):
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


def _plot_pairwise_heatmap(series, pair_labels, title, output_path):
    width_in, height_in, dpi = _plot_dimensions()
    fig, ax = plt.subplots(figsize=(width_in, height_in), dpi=dpi)
    im = ax.imshow(series, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0, interpolation="nearest")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Model pair")
    ax.set_title(title)
    tick_positions = np.arange(len(pair_labels))
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(pair_labels, fontsize=8)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Agreement")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _write_mean_series_csv(output_path, metric_series_by_name):
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


def _write_pairwise_series_csv(output_path, series, pair_labels):
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["pair"] + [f"frame_{idx}" for idx in range(series.shape[1])])
        for pair_label, values in zip(pair_labels, series):
            writer.writerow([pair_label] + [f"{float(value):.8f}" for value in values])


def _pair_labels(model_labels):
    return [f"{model_labels[i]} vs {model_labels[j]}" for i, j in itertools.combinations(range(len(model_labels)), 2)]


def _summary_stats(metric_series_by_name):
    summary = {}
    for metric_name, series in metric_series_by_name.items():
        summary[metric_name] = {
            "mean_pairwise": float(np.mean(series)),
            "std_pairwise": float(np.std(series)),
            "min_pairwise": float(np.min(np.mean(series, axis=1))),
            "max_pairwise": float(np.max(np.mean(series, axis=1))),
        }
    return summary


def _write_summary_txt(output_path, summary, model_labels):
    lines = [
        "Model Group Agreement Summary",
        f"Models: {', '.join(model_labels)}",
        "",
        f"Exact full-action agreement: {summary[EXACT_ACTION_METRIC]['mean_pairwise']:.4f}",
        f"Semantic button-group agreement: {summary[SEMANTIC_METRIC]['mean_pairwise']:.4f}",
        f"Camera-action agreement: {summary[CAMERA_METRIC]['mean_pairwise']:.4f}",
        "",
        "Interpretation:",
        "- Exact full-action agreement is the probability that two sampled models would produce the same final action, including camera direction when camera is active.",
        "- Semantic button-group agreement averages agreement over grouped button decisions such as hotbar, fore/back, left/right, sprint/sneak, attack, jump, and inventory.",
        "- Camera-action agreement measures whether the effective look action matches, accounting for the camera on/off meta-action.",
    ]
    output_path.write_text("\n".join(lines) + "\n")


def _output_dir_for_run(model_metadata):
    first_root_name = Path(model_metadata[0]["input_root"]).name
    output_root = Path(os.environ.get("MULTISTEVE_ACTION_PROB_OUTPUT_ROOT", "outputs/action_probabilities/model_group_comparisons"))
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return output_root / first_root_name / timestamp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout_root", nargs="+", required=True)
    parser.add_argument("--output_dir")
    args = parser.parse_args()

    model_metadata = [_model_metadata(path_str) for path_str in args.rollout_root]
    model_labels = [item["label"] for item in model_metadata]
    run_dirs = [item["run_dir"] for item in model_metadata]

    group_info = _button_group_matrices()
    loaded_models = [_load_model_probs(run_dir, group_info) for run_dir in run_dirs]
    num_steps = loaded_models[0]["num_steps"]
    for loaded_model, run_dir in zip(loaded_models, run_dirs):
        if loaded_model["num_steps"] != num_steps:
            raise ValueError(f"Rollout length mismatch in {run_dir}")

    output_dir = Path(args.output_dir).resolve() if args.output_dir else _output_dir_for_run(model_metadata)
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
        matrix_csv = matrices_dir / f"{short_name}_matrix.csv"
        matrix_png = plots_grids_dir / f"{short_name}_matrix.png"
        pairwise_csv = series_dir / f"{short_name}_pairwise_over_time.csv"
        pairwise_npy = series_dir / f"{short_name}_pairwise_over_time.npy"
        pairwise_heatmap_png = plots_timelines_dir / f"{short_name}_pairwise_over_time_heatmap.png"

        _write_matrix_csv(matrix_csv, metric_matrices[metric_name], model_labels)
        _plot_annotated_matrix(metric_matrices[metric_name], model_labels, METRIC_TITLES[metric_name], matrix_png)
        _write_pairwise_series_csv(pairwise_csv, metric_pairwise_series[metric_name], pair_labels)
        np.save(pairwise_npy, metric_pairwise_series[metric_name], allow_pickle=False)
        _plot_pairwise_heatmap(
            metric_pairwise_series[metric_name],
            pair_labels,
            f"{METRIC_TITLES[metric_name]} Over Time",
            pairwise_heatmap_png,
        )

    mean_series_csv = series_dir / "mean_pairwise_agreement_over_time.csv"
    mean_series_png = plots_timelines_dir / "mean_pairwise_agreement_over_time.png"
    _write_mean_series_csv(mean_series_csv, metric_pairwise_series)
    _plot_mean_pairwise_series(metric_pairwise_series, mean_series_png)

    summary = _summary_stats(metric_pairwise_series)
    summary["model_labels"] = model_labels
    summary["pair_labels"] = pair_labels
    summary["num_steps"] = num_steps
    summary["semantic_groups"] = {
        group_name: list(options) for group_name, options in group_info["semantic_groups"].items()
    }

    summary_json_path = output_dir / "summary.json"
    summary_txt_path = output_dir / "summary.txt"
    summary_json_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    _write_summary_txt(summary_txt_path, summary, model_labels)

    manifest = {
        "created_at": datetime.now().isoformat(),
        "output_dir": str(output_dir),
        "models": model_metadata,
        "num_steps": num_steps,
        "metrics": {
            EXACT_ACTION_METRIC: {
                "description": "Probability that two sampled models would take the same exact full action, including camera direction when camera is active.",
                "matrix_csv": str(matrices_dir / "exact_action_matrix.csv"),
                "matrix_png": str(plots_grids_dir / "exact_action_matrix.png"),
                "pairwise_csv": str(series_dir / "exact_action_pairwise_over_time.csv"),
                "pairwise_npy": str(series_dir / "exact_action_pairwise_over_time.npy"),
                "pairwise_heatmap_png": str(plots_timelines_dir / "exact_action_pairwise_over_time_heatmap.png"),
            },
            SEMANTIC_METRIC: {
                "description": "Average agreement over grouped button distributions such as hotbar, fore/back, left/right, sprint/sneak, attack, jump, camera meta, and inventory.",
                "matrix_csv": str(matrices_dir / "semantic_buttons_matrix.csv"),
                "matrix_png": str(plots_grids_dir / "semantic_buttons_matrix.png"),
                "pairwise_csv": str(series_dir / "semantic_buttons_pairwise_over_time.csv"),
                "pairwise_npy": str(series_dir / "semantic_buttons_pairwise_over_time.npy"),
                "pairwise_heatmap_png": str(plots_timelines_dir / "semantic_buttons_pairwise_over_time_heatmap.png"),
            },
            CAMERA_METRIC: {
                "description": "Probability that the effective camera action matches, accounting for the camera on/off meta-action and camera-bin agreement when active.",
                "matrix_csv": str(matrices_dir / "camera_action_matrix.csv"),
                "matrix_png": str(plots_grids_dir / "camera_action_matrix.png"),
                "pairwise_csv": str(series_dir / "camera_action_pairwise_over_time.csv"),
                "pairwise_npy": str(series_dir / "camera_action_pairwise_over_time.npy"),
                "pairwise_heatmap_png": str(plots_timelines_dir / "camera_action_pairwise_over_time_heatmap.png"),
            },
        },
        "mean_pairwise_plot_png": str(mean_series_png),
        "mean_pairwise_csv": str(mean_series_csv),
        "summary_json": str(summary_json_path),
        "summary_txt": str(summary_txt_path),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
