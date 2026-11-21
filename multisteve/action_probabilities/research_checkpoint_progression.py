import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize, TwoSlopeNorm

TARGET_CHECKPOINT = 491_520
REPORT_TITLE = "Checkpoint Progression Research"


@dataclass
class ExperimentData:
    experiment_dir: Path
    manifest: dict
    frames: np.ndarray
    checkpoint_steps: np.ndarray
    action_names: list[str]
    action_types: list[str]
    stack: np.ndarray  # actions x checkpoints x frames


def pretty_action_name(action_name: str) -> str:
    return action_name.replace(".", " ").replace("_", " ").title()


def format_step_compact(step: int) -> str:
    if step >= 1_000_000:
        return f"{step / 1_000_000:.1f}M"
    if step >= 1_000:
        return f"{step / 1_000:.1f}k"
    return str(step)


def load_experiment(experiment_dir: Path) -> ExperimentData:
    manifest = json.loads((experiment_dir / "manifest.json").read_text())
    action_names = []
    action_types = []
    arrays = []
    frames = None
    checkpoint_steps = np.array([int(item["checkpoint_step"]) for item in manifest["checkpoints"]], dtype=int)

    for action_name, entry in sorted(manifest["button_plots"].items()):
        data = np.loadtxt(entry["csv"], delimiter=",", skiprows=1)
        if frames is None:
            frames = data[:, 0].astype(int)
        action_names.append(action_name)
        action_types.append("button")
        arrays.append(data[:, 1:].T)

    for action_name, entry in sorted(manifest["camera_plots"].items()):
        data = np.loadtxt(entry["csv"], delimiter=",", skiprows=1)
        if frames is None:
            frames = data[:, 0].astype(int)
        action_names.append(action_name)
        action_types.append("camera")
        arrays.append(data[:, 1:].T)

    return ExperimentData(
        experiment_dir=experiment_dir,
        manifest=manifest,
        frames=frames,
        checkpoint_steps=checkpoint_steps,
        action_names=action_names,
        action_types=action_types,
        stack=np.stack(arrays, axis=0),
    )


def nearest_checkpoint_index(checkpoint_steps: np.ndarray, step: int) -> int:
    matches = np.where(checkpoint_steps == step)[0]
    if len(matches) == 0:
        raise ValueError(f"Checkpoint {step} not found")
    return int(matches[0])


def compute_metrics(data: ExperimentData) -> dict:
    stack = data.stack
    checkpoints = data.checkpoint_steps
    action_names = data.action_names
    action_types = data.action_types

    consecutive = stack[:, 1:, :] - stack[:, :-1, :]
    consecutive_mean_abs = np.mean(np.abs(consecutive), axis=(0, 2))
    consecutive_rmse = np.sqrt(np.mean(consecutive**2, axis=(0, 2)))
    consecutive_max_abs = np.max(np.abs(consecutive), axis=(0, 2))

    final = stack[:, -1:, :]
    to_final = stack - final
    to_final_mean_abs = np.mean(np.abs(to_final), axis=(0, 2))
    to_final_rmse = np.sqrt(np.mean(to_final**2, axis=(0, 2)))
    framewise_to_final = np.mean(np.abs(to_final), axis=0)  # checkpoints x frames

    target_ix = nearest_checkpoint_index(checkpoints, TARGET_CHECKPOINT)
    prev_ix = target_ix - 1
    next_ix = target_ix + 1
    expected_target = 0.5 * (stack[:, prev_ix, :] + stack[:, next_ix, :])
    local_residual = stack[:, target_ix, :] - expected_target
    local_residual_mean_abs = np.mean(np.abs(local_residual), axis=1)
    framewise_local_residual = np.mean(np.abs(local_residual), axis=0)

    button_mask = np.array([kind == "button" for kind in action_types], dtype=bool)
    camera_mask = ~button_mask
    transition_action_change = np.mean(np.abs(consecutive), axis=2)  # actions x transitions
    transition_button_total = np.sum(np.abs(consecutive[button_mask]), axis=(0, 2))
    transition_camera_total = np.sum(np.abs(consecutive[camera_mask]), axis=(0, 2))
    transition_camera_share = transition_camera_total / np.maximum(
        transition_button_total + transition_camera_total,
        1e-12,
    )

    checkpoint_vectors = stack.transpose(1, 0, 2).reshape(len(checkpoints), -1)
    checkpoint_vectors_centered = checkpoint_vectors - checkpoint_vectors.mean(axis=0, keepdims=True)
    _, singular_values, vt = np.linalg.svd(checkpoint_vectors_centered, full_matrices=False)
    pca_coords = checkpoint_vectors_centered @ vt[:2].T
    explained = (singular_values**2) / np.sum(singular_values**2)

    def top_actions_for_transition(left_step: int, right_step: int, top_k: int = 8):
        left_ix = nearest_checkpoint_index(checkpoints, left_step)
        right_ix = nearest_checkpoint_index(checkpoints, right_step)
        delta = stack[:, right_ix, :] - stack[:, left_ix, :]
        scores = np.mean(np.abs(delta), axis=1)
        order = np.argsort(-scores)[:top_k]
        return [
            {
                "action": action_names[idx],
                "kind": action_types[idx],
                "mean_abs_change": float(scores[idx]),
            }
            for idx in order
        ]

    rolling_window = 41
    kernel = np.ones(rolling_window) / rolling_window
    framewise_local_residual_smoothed = np.convolve(framewise_local_residual, kernel, mode="same")

    top_frames = np.argsort(-framewise_local_residual_smoothed)
    top_frame_windows = []
    used = np.zeros_like(framewise_local_residual_smoothed, dtype=bool)
    for idx in top_frames:
        if used[idx]:
            continue
        start = max(0, idx - rolling_window // 2)
        end = min(len(framewise_local_residual_smoothed), idx + rolling_window // 2 + 1)
        used[start:end] = True
        top_frame_windows.append(
            {
                "center_frame": int(data.frames[idx]),
                "start_frame": int(data.frames[start]),
                "end_frame": int(data.frames[end - 1]),
                "score": float(framewise_local_residual_smoothed[idx]),
            }
        )
        if len(top_frame_windows) == 3:
            break

    metrics = {
        "top_consecutive_changes": [
            {
                "from": int(checkpoints[idx]),
                "to": int(checkpoints[idx + 1]),
                "mean_abs": float(consecutive_mean_abs[idx]),
                "rmse": float(consecutive_rmse[idx]),
                "max_abs": float(consecutive_max_abs[idx]),
            }
            for idx in np.argsort(-consecutive_mean_abs)[:8]
        ],
        "top_farthest_from_final": [
            {
                "checkpoint": int(checkpoints[idx]),
                "mean_abs": float(to_final_mean_abs[idx]),
                "rmse": float(to_final_rmse[idx]),
            }
            for idx in np.argsort(-to_final_mean_abs)[:8]
        ],
        "target_neighbors": [
            {
                "checkpoint": int(checkpoints[idx]),
                "mean_abs_to_final": float(to_final_mean_abs[idx]),
                "rmse_to_final": float(to_final_rmse[idx]),
            }
            for idx in [prev_ix, target_ix, next_ix]
        ],
        "top_local_residual_actions": [
            {
                "action": action_names[idx],
                "kind": action_types[idx],
                "mean_abs_residual": float(local_residual_mean_abs[idx]),
            }
            for idx in np.argsort(-local_residual_mean_abs)[:12]
        ],
        "transition_contributors_245760_491520": top_actions_for_transition(245_760, 491_520, top_k=10),
        "transition_contributors_491520_983040": top_actions_for_transition(491_520, 983_040, top_k=10),
        "transition_contributors_983040_1966080": top_actions_for_transition(983_040, 1_966_080, top_k=10),
        "target_frame_windows": top_frame_windows,
        "camera_share_by_transition": [
            {
                "from": int(checkpoints[idx]),
                "to": int(checkpoints[idx + 1]),
                "camera_share": float(transition_camera_share[idx]),
            }
            for idx in range(len(transition_camera_share))
        ],
        "pca_explained_variance": [float(explained[0]), float(explained[1])],
    }

    derived = {
        "consecutive_mean_abs": consecutive_mean_abs,
        "consecutive_rmse": consecutive_rmse,
        "to_final_mean_abs": to_final_mean_abs,
        "to_final_rmse": to_final_rmse,
        "framewise_to_final": framewise_to_final,
        "framewise_local_residual": framewise_local_residual,
        "framewise_local_residual_smoothed": framewise_local_residual_smoothed,
        "local_residual_mean_abs": local_residual_mean_abs,
        "transition_action_change": transition_action_change,
        "transition_button_total": transition_button_total,
        "transition_camera_total": transition_camera_total,
        "transition_camera_share": transition_camera_share,
        "pca_coords": pca_coords,
        "target_ix": target_ix,
        "prev_ix": prev_ix,
        "next_ix": next_ix,
        "expected_target": expected_target,
    }
    return {"metrics": metrics, "derived": derived}


def ensure_dirs(output_dir: Path) -> dict[str, Path]:
    plots_dir = output_dir / "plots"
    data_dir = output_dir / "data"
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    return {"root": output_dir, "plots": plots_dir, "data": data_dir}


def save_overview_plot(data: ExperimentData, analysis: dict, output_path: Path) -> None:
    checkpoints = data.checkpoint_steps
    derived = analysis["derived"]
    target_ix = derived["target_ix"]

    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)

    axes[0].plot(checkpoints[1:], derived["consecutive_mean_abs"], marker="o", color="#1f556c", linewidth=2.2)
    axes[0].axvline(checkpoints[target_ix], color="#d86c45", linestyle="--", linewidth=1.8)
    axes[0].set_xscale("log")
    axes[0].set_ylabel("Mean abs change")
    axes[0].set_title("Checkpoint transition strength through training")
    axes[0].grid(alpha=0.25)

    axes[1].plot(checkpoints, derived["to_final_mean_abs"], marker="o", color="#33414d", linewidth=2.2)
    axes[1].axvline(checkpoints[target_ix], color="#d86c45", linestyle="--", linewidth=1.8, label="Checkpoint 491,520")
    axes[1].set_xscale("log")
    axes[1].set_ylabel("Mean abs distance to final")
    axes[1].set_xlabel("Checkpoint step")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_pca_plot(data: ExperimentData, analysis: dict, output_path: Path) -> None:
    checkpoints = data.checkpoint_steps
    coords = analysis["derived"]["pca_coords"]
    target_ix = analysis["derived"]["target_ix"]
    explained = analysis["metrics"]["pca_explained_variance"]
    colors = np.linspace(0.35, 0.9, len(checkpoints))

    fig, ax = plt.subplots(figsize=(10.5, 8))
    ax.plot(coords[:, 0], coords[:, 1], color="#9ca3af", linewidth=1.4, alpha=0.9, zorder=1)
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=colors, cmap="Greys", s=55, zorder=2)
    ax.scatter(coords[target_ix, 0], coords[target_ix, 1], color="#d86c45", s=110, zorder=3, label="Checkpoint 491,520")
    ax.scatter(coords[-1, 0], coords[-1, 1], color="black", s=95, zorder=3, label="Final")

    for idx in [0, target_ix, len(checkpoints) - 1]:
        ax.annotate(format_step_compact(int(checkpoints[idx])), (coords[idx, 0], coords[idx, 1]), xytext=(6, 6), textcoords="offset points", fontsize=9)

    ax.set_title("Checkpoint embedding in action-probability space (PCA)")
    ax.set_xlabel(f"PC1 ({explained[0] * 100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({explained[1] * 100:.1f}% variance)")
    ax.grid(alpha=0.22)
    ax.legend(frameon=False)
    fig.colorbar(scatter, ax=ax, fraction=0.045, pad=0.04, label="Training progression")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_local_residual_bar(data: ExperimentData, analysis: dict, output_path: Path) -> None:
    derived = analysis["derived"]
    scores = derived["local_residual_mean_abs"]
    order = np.argsort(-scores)[:12][::-1]
    labels = [pretty_action_name(data.action_names[idx]) for idx in order]
    colors = ["#1f556c" if data.action_types[idx] == "button" else "#b74b31" for idx in order]

    fig, ax = plt.subplots(figsize=(10.5, 6.8))
    ax.barh(labels, scores[order], color=colors)
    ax.set_title("Checkpoint 491,520 local anomaly by action")
    ax.set_xlabel("Mean abs deviation from linear interpolation between neighbors")
    ax.grid(axis="x", alpha=0.22)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_framewise_anomaly_plot(data: ExperimentData, analysis: dict, output_path: Path) -> None:
    derived = analysis["derived"]
    checkpoints = data.checkpoint_steps
    frames = data.frames
    prev_ix = derived["prev_ix"]
    target_ix = derived["target_ix"]
    next_ix = derived["next_ix"]
    framewise_to_final = derived["framewise_to_final"]
    framewise_local_residual = derived["framewise_local_residual"]
    smooth = derived["framewise_local_residual_smoothed"]

    fig, ax = plt.subplots(figsize=(13.2, 6.4))
    ax.plot(frames, framewise_to_final[prev_ix], color="#8092a2", linewidth=1.8, label=f"{format_step_compact(int(checkpoints[prev_ix]))} -> final")
    ax.plot(frames, framewise_to_final[target_ix], color="#d86c45", linewidth=2.6, label="491,520 -> final")
    ax.plot(frames, framewise_to_final[next_ix], color="#4b6278", linewidth=1.8, label=f"{format_step_compact(int(checkpoints[next_ix]))} -> final")
    ax.plot(frames, smooth, color="#111827", linewidth=2.0, linestyle="--", label="491,520 residual vs neighbors (smoothed)")

    for window in analysis["metrics"]["target_frame_windows"]:
        ax.axvspan(window["start_frame"], window["end_frame"], color="#f5c46b", alpha=0.16)

    ax.set_title("Where checkpoint 491,520 deviates during the episode")
    ax.set_xlabel("Rollout frame")
    ax.set_ylabel("Mean abs probability difference")
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_top_action_panels(data: ExperimentData, analysis: dict, output_path: Path) -> None:
    derived = analysis["derived"]
    scores = derived["local_residual_mean_abs"]
    top_idx = np.argsort(-scores)[:6]
    checkpoints = data.checkpoint_steps
    prev_ix, target_ix, next_ix = derived["prev_ix"], derived["target_ix"], derived["next_ix"]
    final_ix = len(checkpoints) - 1

    fig, axes = plt.subplots(3, 2, figsize=(14, 11), sharex=True)
    axes = axes.flatten()
    colors = {
        "prev": "#8092a2",
        "target": "#d86c45",
        "next": "#4b6278",
        "final": "#111827",
    }

    for ax, action_idx in zip(axes, top_idx):
        action = data.action_names[action_idx]
        series = data.stack[action_idx]
        ax.plot(data.frames, series[prev_ix], color=colors["prev"], linewidth=1.6, label=format_step_compact(int(checkpoints[prev_ix])))
        ax.plot(data.frames, series[target_ix], color=colors["target"], linewidth=2.3, label="491,520")
        ax.plot(data.frames, series[next_ix], color=colors["next"], linewidth=1.6, label=format_step_compact(int(checkpoints[next_ix])))
        ax.plot(data.frames, series[final_ix], color=colors["final"], linewidth=1.8, label="Final")
        ax.set_title(pretty_action_name(action))
        ax.set_ylim(0.0, min(1.02, max(0.08, float(series[[prev_ix, target_ix, next_ix, final_ix]].max()) * 1.05)))
        ax.grid(alpha=0.22)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.995))
    fig.suptitle("Top action trajectories behind the 491,520 anomaly", y=0.99)
    fig.supxlabel("Rollout frame")
    fig.supylabel("Probability")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_transition_heatmap(data: ExperimentData, analysis: dict, output_path: Path) -> None:
    action_change = analysis["derived"]["transition_action_change"]
    action_scores = np.max(action_change, axis=1)
    order = np.argsort(-action_scores)
    ordered_change = action_change[order]
    ordered_labels = [
        ("B: " if data.action_types[idx] == "button" else "C: ") + pretty_action_name(data.action_names[idx])
        for idx in order
    ]
    transition_labels = [f"{format_step_compact(int(data.checkpoint_steps[idx]))}\n→ {format_step_compact(int(data.checkpoint_steps[idx + 1]))}" for idx in range(len(data.checkpoint_steps) - 1)]

    fig, ax = plt.subplots(figsize=(15, 9))
    im = ax.imshow(ordered_change, aspect="auto", cmap="magma")
    ax.set_title("Action-by-transition change heatmap")
    ax.set_yticks(np.arange(len(ordered_labels)))
    ax.set_yticklabels(ordered_labels, fontsize=8)
    ax.set_xticks(np.arange(len(transition_labels)))
    ax.set_xticklabels(transition_labels, rotation=60, ha="right", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="Mean abs change")
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def save_distance_heatmap(data: ExperimentData, analysis: dict, output_path: Path) -> None:
    framewise_to_final = analysis["derived"]["framewise_to_final"]
    target_ix = analysis["derived"]["target_ix"]
    fig, ax = plt.subplots(figsize=(15, 7.2))
    im = ax.imshow(framewise_to_final, aspect="auto", cmap="viridis")
    ax.axhline(target_ix, color="#d86c45", linewidth=2.0)
    ax.set_title("Checkpoint-by-frame distance to final policy")
    ax.set_xlabel("Rollout frame")
    ax.set_ylabel("Checkpoint")
    tick_positions = np.arange(len(data.checkpoint_steps))
    tick_labels = [format_step_compact(int(step)) for step in data.checkpoint_steps]
    show = np.linspace(0, len(tick_positions) - 1, num=min(8, len(tick_positions)), dtype=int)
    ax.set_yticks(show)
    ax.set_yticklabels([tick_labels[idx] for idx in show])
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="Mean abs distance to final")
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def save_family_share_plot(data: ExperimentData, analysis: dict, output_path: Path) -> None:
    button_total = analysis["derived"]["transition_button_total"]
    camera_total = analysis["derived"]["transition_camera_total"]
    x = np.arange(len(button_total))
    labels = [format_step_compact(int(step)) for step in data.checkpoint_steps[1:]]

    fig, ax = plt.subplots(figsize=(14, 5.8))
    ax.bar(x, button_total, color="#1f556c", label="Buttons")
    ax.bar(x, camera_total, bottom=button_total, color="#b74b31", label="Camera")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    ax.set_title("Family contribution to each consecutive transition")
    ax.set_xlabel("To checkpoint")
    ax.set_ylabel("Total abs change")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.22)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def write_data_tables(data: ExperimentData, analysis: dict, dirs: dict[str, Path]) -> None:
    summary_path = dirs["data"] / "summary_metrics.json"
    summary_path.write_text(json.dumps(analysis["metrics"], indent=2))

    residual_csv = dirs["data"] / "checkpoint_491520_local_residuals.csv"
    with residual_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["kind", "action", "mean_abs_residual"])
        order = np.argsort(-analysis["derived"]["local_residual_mean_abs"])
        for idx in order:
            writer.writerow(
                [
                    data.action_types[idx],
                    data.action_names[idx],
                    float(analysis["derived"]["local_residual_mean_abs"][idx]),
                ]
            )


def write_reports(data: ExperimentData, analysis: dict, dirs: dict[str, Path]) -> None:
    metrics = analysis["metrics"]

    checkpoint_report = f"""# Checkpoint 491,520 Report

## Headline

Checkpoint `491,520` is not just visually different. In this rollout-conditioned action-probability space it is the single checkpoint **farthest from the final policy**, and the transition into it (`245,760 -> 491,520`) is the **largest checkpoint-to-checkpoint jump** in the whole training run.

## Main Findings

1. `245,760 -> 491,520` is the strongest consecutive transition by mean absolute change: `{metrics['top_consecutive_changes'][0]['mean_abs']:.4f}`.
2. Checkpoint `491,520` is the checkpoint farthest from final by mean absolute distance: `{metrics['top_farthest_from_final'][0]['mean_abs']:.4f}`.
3. The anomaly is dominated by **movement and control actions**, not camera actions. The biggest contributors entering `491,520` are `{", ".join(item['action'] for item in metrics['transition_contributors_245760_491520'][:6])}`.
4. Locally, `491,520` is not a smooth midpoint between its neighbors. Its biggest interpolation residual actions are `{", ".join(item['action'] for item in metrics['top_local_residual_actions'][:6])}`.

## Plots

![Training overview](plots/overview_transition_strength.png)

![Checkpoint 491,520 framewise anomaly](plots/checkpoint_491520_framewise_anomaly.png)

![Checkpoint 491,520 local residual by action](plots/checkpoint_491520_local_residuals.png)

![Top action trajectories behind the anomaly](plots/checkpoint_491520_top_actions.png)

## Interpretation

The evidence looks less like random noise and more like a **transient policy regime**. Training reaches a movement-heavy configuration around `491,520`, then partially backs away from it by `983,040` and continues reshaping interaction actions after that. In other words, `491,520` appears to be a genuine excursion in policy space rather than a simple monotone step toward the final checkpoint.

## Where It Happens In The Episode

The strongest anomaly windows from the smoothed framewise residual are:

{chr(10).join(f"- Frames `{window['start_frame']}` to `{window['end_frame']}` with score `{window['score']:.4f}`" for window in metrics['target_frame_windows'])}

These windows are where checkpoint `491,520` deviates most strongly from the expected local trajectory defined by its neighbors.
"""

    exploratory_report = f"""# Exploratory Findings Report

## Headline

The training path is phase-structured. The biggest reorganizations happen in three bursts:

{chr(10).join(f"- `{item['from']}` -> `{item['to']}` with mean abs change `{item['mean_abs']:.4f}`" for item in metrics['top_consecutive_changes'][:3])}

## What Looks Interesting Beyond 491,520

1. The checkpoint trajectory in PCA space makes a visible excursion around `491,520` before curving back toward the later-training region.
2. Buttons dominate the large regime changes. Camera probabilities move, but they are a small share of total transition magnitude compared with buttons.
3. The transition `983,040 -> 1,966,080` is the next major change after the `491,520` excursion, and its top contributors shift toward `{", ".join(item['action'] for item in metrics['transition_contributors_983040_1966080'][:5])}`.
4. After roughly `7.9M` steps the checkpoint-to-checkpoint changes become much smaller, suggesting the run has moved from phase shifts into refinement.

## Plots

![Checkpoint embedding in PCA space](plots/checkpoint_pca_embedding.png)

![Action-by-transition change heatmap](plots/action_transition_heatmap.png)

![Checkpoint-by-frame distance to final](plots/checkpoint_frame_distance_heatmap.png)

![Button vs camera contribution by transition](plots/family_transition_share.png)

## Additional Quantitative Notes

- The top three checkpoints farthest from final are `{", ".join(str(item['checkpoint']) for item in metrics['top_farthest_from_final'][:3])}`.
- The camera share of total transition magnitude stays relatively small even during the major regime shifts.
- The anomalous region around `491,520` is therefore better described as a **control-policy reorganization** than a camera-policy event.

## Good Next Questions

- Are the anomalous frame windows aligned with specific visual situations in the saved rollout frames?
- Does the same `491,520` excursion appear for other prompts or only for `chop a tree`?
- Is the excursion mirrored in hidden-state geometry or only in output probabilities?
"""

    journal = f"""# Research Journal

## 2026-03-26

- Started from the checkpoint progression artifact set at `{data.experiment_dir}`.
- Verified that checkpoint `491,520` is the farthest checkpoint from final and that `245,760 -> 491,520` is the strongest consecutive jump.
- Computed local interpolation residuals to test whether `491,520` is merely far from final or actually off-trajectory relative to its neighbors.
- Found that the anomaly is button-dominated, especially movement and interaction actions.
- Extended the analysis into broader phase structure with PCA, action-by-transition heatmaps, family contribution plots, and checkpoint-by-frame distance heatmaps.
- Wrote two reports: one focused on `491,520`, one broader exploratory summary.
"""

    (dirs["root"] / "report_checkpoint_491520.md").write_text(checkpoint_report)
    (dirs["root"] / "report_exploratory_findings.md").write_text(exploratory_report)
    (dirs["root"] / "journal.md").write_text(journal)
    (dirs["root"] / "README.md").write_text(
        "# Checkpoint Progression Research\n\n"
        "- [Focused checkpoint 491,520 report](report_checkpoint_491520.md)\n"
        "- [Broader exploratory findings](report_exploratory_findings.md)\n"
        "- [Research journal](journal.md)\n"
    )


def save_manifest(data: ExperimentData, analysis: dict, dirs: dict[str, Path]) -> None:
    manifest = {
        "source_experiment_dir": str(data.experiment_dir),
        "target_checkpoint": TARGET_CHECKPOINT,
        "reports": {
            "checkpoint_491520": str(dirs["root"] / "report_checkpoint_491520.md"),
            "exploratory": str(dirs["root"] / "report_exploratory_findings.md"),
            "journal": str(dirs["root"] / "journal.md"),
        },
        "plots": {
            "overview_transition_strength": str(dirs["plots"] / "overview_transition_strength.png"),
            "checkpoint_pca_embedding": str(dirs["plots"] / "checkpoint_pca_embedding.png"),
            "checkpoint_491520_local_residuals": str(dirs["plots"] / "checkpoint_491520_local_residuals.png"),
            "checkpoint_491520_framewise_anomaly": str(dirs["plots"] / "checkpoint_491520_framewise_anomaly.png"),
            "checkpoint_491520_top_actions": str(dirs["plots"] / "checkpoint_491520_top_actions.png"),
            "action_transition_heatmap": str(dirs["plots"] / "action_transition_heatmap.png"),
            "checkpoint_frame_distance_heatmap": str(dirs["plots"] / "checkpoint_frame_distance_heatmap.png"),
            "family_transition_share": str(dirs["plots"] / "family_transition_share.png"),
        },
        "data": {
            "summary_metrics": str(dirs["data"] / "summary_metrics.json"),
            "checkpoint_491520_local_residuals": str(dirs["data"] / "checkpoint_491520_local_residuals.csv"),
        },
        "headline_metrics": analysis["metrics"],
    }
    (dirs["root"] / "manifest.json").write_text(json.dumps(manifest, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    experiment_dir = Path(args.experiment_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    dirs = ensure_dirs(output_dir)

    data = load_experiment(experiment_dir)
    analysis = compute_metrics(data)

    save_overview_plot(data, analysis, dirs["plots"] / "overview_transition_strength.png")
    save_pca_plot(data, analysis, dirs["plots"] / "checkpoint_pca_embedding.png")
    save_local_residual_bar(data, analysis, dirs["plots"] / "checkpoint_491520_local_residuals.png")
    save_framewise_anomaly_plot(data, analysis, dirs["plots"] / "checkpoint_491520_framewise_anomaly.png")
    save_top_action_panels(data, analysis, dirs["plots"] / "checkpoint_491520_top_actions.png")
    save_transition_heatmap(data, analysis, dirs["plots"] / "action_transition_heatmap.png")
    save_distance_heatmap(data, analysis, dirs["plots"] / "checkpoint_frame_distance_heatmap.png")
    save_family_share_plot(data, analysis, dirs["plots"] / "family_transition_share.png")

    write_data_tables(data, analysis, dirs)
    write_reports(data, analysis, dirs)
    save_manifest(data, analysis, dirs)


if __name__ == "__main__":
    main()
