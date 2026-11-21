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
from tqdm import tqdm

from steve1.config import PRIOR_INFO, DEVICE
from steve1.data.text_alignment.vae import load_vae_model
from steve1.utils.embed_utils import get_prior_embed
from steve1.utils.mineclip_agent_env_utils import load_mineclip_wconfig, make_agent
from steve1.VPT.lib.actions import ActionTransformer, Buttons
from steve1.VPT.lib.action_mapping import CameraHierarchicalMapping

SCREEN_WIDTH = 1900
SCREEN_HEIGHT = 1080
SCREEN_FRACTION = 0.9
PLOT_DPI = 100
PLOT_WIDTH_PX = int(round(SCREEN_WIDTH * SCREEN_FRACTION))
PLOT_HEIGHT_PX = int(round(SCREEN_HEIGHT * SCREEN_FRACTION))
CAMERA_DIRECTION_THRESHOLD_DEGREES = 3.0


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


def _discover_first_n_models(models_root, num_models, snapshot_name):
    model_dirs = sorted(
        os.path.join(models_root, entry)
        for entry in os.listdir(models_root)
        if os.path.isdir(os.path.join(models_root, entry))
    )
    selected = []
    for model_dir in model_dirs:
        weight_path = os.path.join(model_dir, snapshot_name)
        if os.path.exists(weight_path):
            selected.append((os.path.basename(model_dir), weight_path))
        if len(selected) == num_models:
            break
    if len(selected) < num_models:
        raise ValueError(f"Requested {num_models} models, found only {len(selected)} matching {snapshot_name}")
    return selected


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


def _save_rollout_outputs(run_dir, model_label, model_weights, buttons, camera, prompt, seed, command_text):
    os.makedirs(run_dir, exist_ok=True)
    np.save(os.path.join(run_dir, "buttons.npy"), buttons, allow_pickle=True)
    np.save(os.path.join(run_dir, "camera.npy"), camera, allow_pickle=True)
    with open(os.path.join(run_dir, "prompt.log"), "w") as f:
        f.write(prompt)
    with open(os.path.join(run_dir, "seed.log"), "w") as f:
        f.write(str(seed))
    with open(os.path.join(run_dir, "model_label.log"), "w") as f:
        f.write(model_label)
    with open(os.path.join(run_dir, "model_weights.log"), "w") as f:
        f.write(model_weights)
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
    transformer = ActionTransformer()

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


def _prettify_label(action_name):
    return action_name.replace("_", " ").replace(".", " ")


def _slugify_label(action_name):
    return action_name.replace(".", "_").replace(" ", "_")


def _plot_series(output_path, title, y_label, reference_label, reference_series, model_series):
    fig, ax = plt.subplots(figsize=_plot_dimensions(), dpi=PLOT_DPI)
    x = np.arange(len(reference_series))

    ax.plot(x, reference_series, linewidth=3.4, color="black", label=reference_label, zorder=5)
    cmap = plt.get_cmap("tab10")
    for idx, (label, series) in enumerate(model_series):
        ax.plot(x, series, linewidth=1.2, alpha=0.9, color=cmap(idx % 10), label=label)

    ymax = max(float(reference_series.max()), *(float(series.max()) for _, series in model_series))
    ax.set_xlim(0, len(reference_series) - 1)
    ax.set_ylim(0.0, min(1.02, max(0.05, ymax * 1.05)))
    ax.set_xlabel("Frame")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(
        loc="center left",
        bbox_to_anchor=(0.80, 0.5),
        frameon=False,
        fontsize=11,
    )
    fig.subplots_adjust(left=0.07, right=0.78, top=0.92, bottom=0.10)
    fig.savefig(output_path, dpi=PLOT_DPI)
    plt.close(fig)


def _write_series_csv(output_path, reference_label, reference_series, model_series):
    labels = [reference_label] + [label for label, _ in model_series]
    arrays = [reference_series] + [series for _, series in model_series]
    stacked = np.column_stack([np.arange(len(reference_series))] + arrays)
    header = ",".join(["frame"] + labels)
    np.savetxt(output_path, stacked, delimiter=",", header=header, comments="")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models_root", required=True)
    parser.add_argument("--num_models", type=int, default=10)
    parser.add_argument("--snapshot_name", default="STEVE-1_snapshot_160005120.weights")
    parser.add_argument("--in_model", default="data/weights/vpt/2x.model")
    parser.add_argument("--zip_path", required=True)
    parser.add_argument("--reference_run_dir", required=True)
    parser.add_argument("--reference_label", required=True)
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
    models = _discover_first_n_models(args.models_root, args.num_models, args.snapshot_name)

    reference_buttons = np.load(os.path.join(args.reference_run_dir, "buttons.npy"), allow_pickle=True)
    reference_camera = np.load(os.path.join(args.reference_run_dir, "camera.npy"), allow_pickle=True)
    reference_button_series = _button_probability_series_by_name(reference_buttons)
    reference_camera_series = _camera_direction_probability_series(reference_camera, reference_buttons)

    manifest = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "models_root": os.path.abspath(args.models_root),
        "zip_path": os.path.abspath(args.zip_path),
        "reference_run_dir": os.path.abspath(args.reference_run_dir),
        "reference_label": args.reference_label,
        "plot_dimensions_pixels": {"width": PLOT_WIDTH_PX, "height": PLOT_HEIGHT_PX, "dpi": PLOT_DPI},
        "camera_direction_threshold_degrees": CAMERA_DIRECTION_THRESHOLD_DEGREES,
        "camera_direction_definition": "softmax(camera_scores) direction mass multiplied by softmax(button_scores) camera-meta-on mass",
        "models": [],
        "button_plots": {},
        "camera_plots": {},
    }

    frames = None
    prompt_embed = None
    plotted_button_series = {button_name: [] for button_name in Buttons.ALL}
    plotted_camera_series = {direction: [] for direction in ["look_up", "look_down", "look_left", "look_right"]}
    for model_label, model_weights in tqdm(models, desc="Models"):
        run_dir = os.path.join(replays_dir, model_label)
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
                in_weights=model_weights,
                cond_scale=args.text_cond_scale,
            )
            _save_rollout_outputs(
                run_dir=run_dir,
                model_label=model_label,
                model_weights=model_weights,
                buttons=buttons,
                camera=camera,
                prompt=args.prompt,
                seed=args.seed,
                command_text=command_text,
            )
        button_series = _button_probability_series_by_name(buttons)
        camera_series = _camera_direction_probability_series(camera, buttons)
        for button_name, series in button_series.items():
            plotted_button_series[button_name].append((model_label, series))
        for direction, series in camera_series.items():
            plotted_camera_series[direction].append((model_label, series))
        manifest["models"].append(
            {
                "model_label": model_label,
                "model_weights": model_weights,
                "run_dir": run_dir,
            }
        )

    for button_name in Buttons.ALL:
        slug = _slugify_label(button_name)
        png_path = os.path.join(plot_buttons_dir, f"{slug}.png")
        csv_path = os.path.join(curves_buttons_dir, f"{slug}.csv")
        _plot_series(
            output_path=png_path,
            title=f"Button Probability: {_prettify_label(button_name).title()}",
            y_label="Button probability",
            reference_label=args.reference_label,
            reference_series=reference_button_series[button_name],
            model_series=plotted_button_series[button_name],
        )
        _write_series_csv(
            output_path=csv_path,
            reference_label=args.reference_label,
            reference_series=reference_button_series[button_name],
            model_series=plotted_button_series[button_name],
        )
        manifest["button_plots"][button_name] = {"png": png_path, "csv": csv_path}
        if button_name == "inventory":
            inventory_png_path = os.path.join(plots_dir, "inventory_probability_overlay.png")
            _plot_series(
                output_path=inventory_png_path,
                title="Button Probability: Inventory",
                y_label="Button probability",
                reference_label=args.reference_label,
                reference_series=reference_button_series[button_name],
                model_series=plotted_button_series[button_name],
            )
            manifest["inventory_plot_png"] = inventory_png_path

    for direction in ["look_up", "look_down", "look_left", "look_right"]:
        png_path = os.path.join(plot_camera_dir, f"{direction}.png")
        csv_path = os.path.join(curves_camera_dir, f"{direction}.csv")
        _plot_series(
            output_path=png_path,
            title=f"Camera Probability: {_prettify_label(direction).title()}",
            y_label="Camera action probability",
            reference_label=args.reference_label,
            reference_series=reference_camera_series[direction],
            model_series=plotted_camera_series[direction],
        )
        _write_series_csv(
            output_path=csv_path,
            reference_label=args.reference_label,
            reference_series=reference_camera_series[direction],
            model_series=plotted_camera_series[direction],
        )
        manifest["camera_plots"][direction] = {"png": png_path, "csv": csv_path}

    with open(os.path.join(args.output_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
