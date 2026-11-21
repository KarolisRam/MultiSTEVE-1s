import argparse
import json
import os
import shlex
import sys
import time
import zipfile

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from tqdm import tqdm

from steve1.config import DEVICE, PRIOR_INFO
from steve1.data.text_alignment.vae import load_vae_model
from steve1.utils.embed_utils import get_prior_embed
from steve1.utils.mineclip_agent_env_utils import load_mineclip_wconfig, make_agent
from steve1.VPT.agent import ACTION_TRANSFORMER_KWARGS
from steve1.VPT.lib.actions import ActionTransformer, Buttons
from steve1.VPT.lib.action_mapping import CameraHierarchicalMapping

SCREEN_WIDTH = 1900
SCREEN_HEIGHT = 1080
SCREEN_FRACTION = 0.9
PLOT_DPI = 100
PLOT_WIDTH_PX = int(round(SCREEN_WIDTH * SCREEN_FRACTION))
PLOT_HEIGHT_PX = int(round(SCREEN_HEIGHT * SCREEN_FRACTION))
CAMERA_DIRECTION_THRESHOLD_DEGREES = 3.0
EARLIEST_GRAY = 0.5
EARLIEST_LINEWIDTH = 0.7
LATEST_LINEWIDTH = 3.4


def _decode_frames_from_zip(zip_path):
    with zipfile.ZipFile(zip_path, "r") as zf:
        frame_names = sorted(name for name in zf.namelist() if name.endswith(".png"))
        frames = []
        for name in frame_names:
            encoded = np.frombuffer(zf.read(name), dtype=np.uint8)
            frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    return frames


def _make_obs(frame):
    return {
        "pov": frame,
        "location_stats": {},
        "inventory": {},
        "damage_dealt": {"damage_dealt": np.array(0)},
        "kill_entity": {"villager": np.array(0)},
    }


def _discover_checkpoints(weights_dir):
    checkpoint_names = sorted(
        entry
        for entry in os.listdir(weights_dir)
        if entry.startswith("STEVE-1_snapshot_") and entry.endswith(".weights")
    )
    if not checkpoint_names:
        raise ValueError(f"No STEVE-1 checkpoints found in {weights_dir}")
    checkpoints = []
    for checkpoint_name in checkpoint_names:
        checkpoint_step = checkpoint_name[len("STEVE-1_snapshot_") : -len(".weights")]
        checkpoints.append((checkpoint_step, os.path.join(weights_dir, checkpoint_name)))
    return checkpoints


def _run_exact_fake_rollout(frames, prompt_embed, in_model, in_weights, cond_scale):
    agent = make_agent(in_model, in_weights, cond_scale=cond_scale)
    buttons = []
    camera = []
    with tqdm(total=len(frames), leave=False) as pbar:
        for frame in frames:
            obs = _make_obs(frame)
            with torch.cuda.amp.autocast():
                _, pd = agent.get_action_with_pd(obs, prompt_embed)
            buttons.append(pd["buttons"].cpu().numpy().squeeze())
            camera.append(pd["camera"].cpu().numpy().squeeze())
            pbar.update(1)
    return np.stack(buttons), np.stack(camera)


def _save_rollout_outputs(run_dir, checkpoint_step, checkpoint_weights, buttons, camera, prompt, seed, command_text):
    os.makedirs(run_dir, exist_ok=True)
    np.save(os.path.join(run_dir, "buttons.npy"), buttons, allow_pickle=True)
    np.save(os.path.join(run_dir, "camera.npy"), camera, allow_pickle=True)
    with open(os.path.join(run_dir, "prompt.log"), "w") as f:
        f.write(prompt)
    with open(os.path.join(run_dir, "seed.log"), "w") as f:
        f.write(str(seed))
    with open(os.path.join(run_dir, "checkpoint_step.log"), "w") as f:
        f.write(checkpoint_step)
    with open(os.path.join(run_dir, "model_weights.log"), "w") as f:
        f.write(checkpoint_weights)
    with open(os.path.join(run_dir, "command.log"), "w") as f:
        f.write(command_text + "\n")


def _softmax_from_scores(scores):
    scores = scores.astype(np.float64, copy=False)
    shifted = scores - np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    return exp_scores / exp_scores.sum(axis=1, keepdims=True)


def _button_probability_series_by_name(button_scores):
    mapping = CameraHierarchicalMapping(n_camera_bins=11)
    joint_probs = _softmax_from_scores(button_scores)
    factored_matrix = mapping.BUTTON_IDX_TO_FACTORED.astype(np.float64)
    factored_probs = joint_probs @ factored_matrix
    return {
        button_name: factored_probs[:, idx]
        for idx, button_name in enumerate(Buttons.ALL)
    }


def _camera_direction_probability_series(camera_scores, button_scores):
    mapping = CameraHierarchicalMapping(n_camera_bins=11)
    transformer = ActionTransformer(**ACTION_TRANSFORMER_KWARGS)

    button_joint_probs = _softmax_from_scores(button_scores)
    camera_joint_probs = _softmax_from_scores(camera_scores)

    camera_on_mask = np.array(
        [
            button_comb != "inventory" and button_comb[-1] == "camera"
            for button_comb in mapping.BUTTONS_COMBINATIONS
        ],
        dtype=np.float64,
    )
    camera_on_prob = button_joint_probs @ camera_on_mask

    camera_actions = transformer.undiscretize_camera(mapping.CAMERA_IDX_TO_FACTORED)
    pitch_vals = camera_actions[:, 0]
    yaw_vals = camera_actions[:, 1]

    direction_masks = {
        "look_up": pitch_vals <= -CAMERA_DIRECTION_THRESHOLD_DEGREES,
        "look_down": pitch_vals >= CAMERA_DIRECTION_THRESHOLD_DEGREES,
        "look_left": yaw_vals <= -CAMERA_DIRECTION_THRESHOLD_DEGREES,
        "look_right": yaw_vals >= CAMERA_DIRECTION_THRESHOLD_DEGREES,
    }

    return {
        direction: camera_on_prob * camera_joint_probs[:, mask].sum(axis=1)
        for direction, mask in direction_masks.items()
    }


def _plot_dimensions():
    return PLOT_WIDTH_PX / PLOT_DPI, PLOT_HEIGHT_PX / PLOT_DPI


def _prettify_label(label):
    return label.replace("_", " ").replace(".", " ")


def _slugify_label(label):
    return label.replace(".", "_").replace(" ", "_")


def _checkpoint_style(index, total):
    if total <= 1:
        return "0.0", LATEST_LINEWIDTH
    frac = index / (total - 1)
    gray = EARLIEST_GRAY * (1.0 - frac)
    linewidth = EARLIEST_LINEWIDTH + (LATEST_LINEWIDTH - EARLIEST_LINEWIDTH) * frac
    return str(gray), linewidth


def _checkpoint_legend_handles(checkpoint_series):
    sample_indices = sorted(set([0, len(checkpoint_series) // 2, len(checkpoint_series) - 1]))
    handles = []
    for idx in sample_indices:
        checkpoint_step, _ = checkpoint_series[idx]
        color, linewidth = _checkpoint_style(idx, len(checkpoint_series))
        label = f"ckpt {checkpoint_step}"
        if idx == len(checkpoint_series) - 1:
            label += " (final)"
        handles.append(Line2D([0], [0], color=color, linewidth=linewidth, label=label))
    return handles


def _plot_checkpoint_series(output_path, title, y_label, checkpoint_series):
    fig, ax = plt.subplots(figsize=_plot_dimensions(), dpi=PLOT_DPI)
    x = np.arange(len(checkpoint_series[0][1]))

    ymax = 0.0
    for idx, (_, series) in enumerate(checkpoint_series):
        color, linewidth = _checkpoint_style(idx, len(checkpoint_series))
        ax.plot(x, series, color=color, linewidth=linewidth, alpha=1.0, zorder=2 + idx)
        ymax = max(ymax, float(series.max()))

    ax.set_xlim(0, len(x) - 1)
    ax.set_ylim(0.0, min(1.02, max(0.05, ymax * 1.05)))
    ax.set_xlabel("Frame")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(
        handles=_checkpoint_legend_handles(checkpoint_series),
        loc="center left",
        bbox_to_anchor=(0.80, 0.5),
        frameon=False,
        fontsize=11,
        title="Training progression",
        title_fontsize=12,
    )
    ax.text(
        0.80,
        0.16,
        f"{len(checkpoint_series)} checkpoints\n50% gray -> black",
        transform=ax.transAxes,
        fontsize=11,
        va="top",
        ha="left",
    )
    fig.subplots_adjust(left=0.07, right=0.78, top=0.92, bottom=0.10)
    fig.savefig(output_path, dpi=PLOT_DPI)
    plt.close(fig)


def _write_series_csv(output_path, checkpoint_series):
    labels = [f"ckpt_{checkpoint_step}" for checkpoint_step, _ in checkpoint_series]
    arrays = [series for _, series in checkpoint_series]
    stacked = np.column_stack([np.arange(len(arrays[0]))] + arrays)
    header = ",".join(["frame"] + labels)
    np.savetxt(output_path, stacked, delimiter=",", header=header, comments="")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights_dir", required=True)
    parser.add_argument("--in_model", default="data/weights/vpt/2x.model")
    parser.add_argument("--zip_path", required=True)
    parser.add_argument("--reference_run_dir", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--prompt_seed", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--text_cond_scale", type=float, default=6.0)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    replays_dir = os.path.join(args.output_dir, "replays")
    plots_dir = os.path.join(args.output_dir, "plots")
    plot_buttons_dir = os.path.join(plots_dir, "buttons")
    plot_camera_dir = os.path.join(plots_dir, "camera")
    curves_dir = os.path.join(args.output_dir, "curves")
    curves_buttons_dir = os.path.join(curves_dir, "buttons")
    curves_camera_dir = os.path.join(curves_dir, "camera")
    os.makedirs(replays_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(plot_buttons_dir, exist_ok=True)
    os.makedirs(plot_camera_dir, exist_ok=True)
    os.makedirs(curves_buttons_dir, exist_ok=True)
    os.makedirs(curves_camera_dir, exist_ok=True)

    command_text = " ".join(shlex.quote(arg) for arg in [sys.executable, *sys.argv])
    checkpoints = _discover_checkpoints(args.weights_dir)

    reference_buttons = np.load(os.path.join(args.reference_run_dir, "buttons.npy"), allow_pickle=True)
    reference_camera = np.load(os.path.join(args.reference_run_dir, "camera.npy"), allow_pickle=True)

    manifest = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "weights_dir": os.path.abspath(args.weights_dir),
        "zip_path": os.path.abspath(args.zip_path),
        "reference_run_dir": os.path.abspath(args.reference_run_dir),
        "plot_dimensions_pixels": {"width": PLOT_WIDTH_PX, "height": PLOT_HEIGHT_PX, "dpi": PLOT_DPI},
        "camera_direction_threshold_degrees": CAMERA_DIRECTION_THRESHOLD_DEGREES,
        "camera_direction_definition": "softmax(camera_scores) direction mass multiplied by softmax(button_scores) camera-meta-on mass",
        "checkpoints": [],
        "button_plots": {},
        "camera_plots": {},
    }

    frames = None
    prompt_embed = None
    plotted_button_series = {button_name: [] for button_name in Buttons.ALL}
    plotted_camera_series = {direction: [] for direction in ["look_up", "look_down", "look_left", "look_right"]}
    final_buttons = None
    final_camera = None

    for checkpoint_idx, (checkpoint_step, checkpoint_weights) in enumerate(tqdm(checkpoints, desc="Checkpoints")):
        run_dir = os.path.join(replays_dir, f"checkpoint_{checkpoint_step}")
        buttons_path = os.path.join(run_dir, "buttons.npy")
        camera_path = os.path.join(run_dir, "camera.npy")
        if os.path.exists(buttons_path) and os.path.exists(camera_path):
            buttons = np.load(buttons_path, allow_pickle=True)
            camera = np.load(camera_path, allow_pickle=True)
        else:
            if frames is None:
                frames = _decode_frames_from_zip(args.zip_path)
            if prompt_embed is None:
                mineclip = load_mineclip_wconfig()
                prior = load_vae_model(PRIOR_INFO)
                prompt_embed = get_prior_embed(args.prompt, mineclip, prior, DEVICE, seed=args.prompt_seed)
            buttons, camera = _run_exact_fake_rollout(
                frames=frames,
                prompt_embed=prompt_embed,
                in_model=args.in_model,
                in_weights=checkpoint_weights,
                cond_scale=args.text_cond_scale,
            )
            _save_rollout_outputs(
                run_dir=run_dir,
                checkpoint_step=checkpoint_step,
                checkpoint_weights=checkpoint_weights,
                buttons=buttons,
                camera=camera,
                prompt=args.prompt,
                seed=args.seed,
                command_text=command_text,
            )

        button_series = _button_probability_series_by_name(buttons)
        camera_series = _camera_direction_probability_series(camera, buttons)
        for button_name, series in button_series.items():
            plotted_button_series[button_name].append((checkpoint_step, series))
        for direction, series in camera_series.items():
            plotted_camera_series[direction].append((checkpoint_step, series))

        manifest["checkpoints"].append(
            {
                "checkpoint_index": checkpoint_idx,
                "checkpoint_step": checkpoint_step,
                "checkpoint_weights": checkpoint_weights,
                "run_dir": run_dir,
            }
        )

        if checkpoint_idx == len(checkpoints) - 1:
            final_buttons = buttons
            final_camera = camera

    manifest["final_checkpoint_match_to_reference"] = {
        "checkpoint_step": checkpoints[-1][0],
        "buttons_exact_equal": bool(np.array_equal(final_buttons, reference_buttons)),
        "camera_exact_equal": bool(np.array_equal(final_camera, reference_camera)),
    }

    for button_name in Buttons.ALL:
        slug = _slugify_label(button_name)
        png_path = os.path.join(plot_buttons_dir, f"{slug}.png")
        csv_path = os.path.join(curves_buttons_dir, f"{slug}.csv")
        _plot_checkpoint_series(
            output_path=png_path,
            title=f"Button Probability Through Training: {_prettify_label(button_name).title()}",
            y_label="Button probability",
            checkpoint_series=plotted_button_series[button_name],
        )
        _write_series_csv(
            output_path=csv_path,
            checkpoint_series=plotted_button_series[button_name],
        )
        manifest["button_plots"][button_name] = {"png": png_path, "csv": csv_path}
        if button_name == "inventory":
            inventory_png_path = os.path.join(plots_dir, "inventory_probability_overlay.png")
            _plot_checkpoint_series(
                output_path=inventory_png_path,
                title="Button Probability Through Training: Inventory",
                y_label="Button probability",
                checkpoint_series=plotted_button_series[button_name],
            )
            manifest["inventory_plot_png"] = inventory_png_path

    for direction in ["look_up", "look_down", "look_left", "look_right"]:
        png_path = os.path.join(plot_camera_dir, f"{direction}.png")
        csv_path = os.path.join(curves_camera_dir, f"{direction}.csv")
        _plot_checkpoint_series(
            output_path=png_path,
            title=f"Camera Probability Through Training: {_prettify_label(direction).title()}",
            y_label="Camera action probability",
            checkpoint_series=plotted_camera_series[direction],
        )
        _write_series_csv(
            output_path=csv_path,
            checkpoint_series=plotted_camera_series[direction],
        )
        manifest["camera_plots"][direction] = {"png": png_path, "csv": csv_path}

    with open(os.path.join(args.output_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
