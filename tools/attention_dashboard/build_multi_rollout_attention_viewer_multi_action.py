import argparse
import ast
import base64
import csv
import hashlib
import html
import json
import re
import shlex
import textwrap
from functools import lru_cache
from pathlib import Path

import numpy as np

try:
    from .build_attention_hover_viewer import extract_frames, load_model_config_from_command
except ImportError:
    from build_attention_hover_viewer import extract_frames, load_model_config_from_command
from steve1.VPT.agent import ACTION_TRANSFORMER_KWARGS
from steve1.VPT.lib.action_mapping import CameraHierarchicalMapping
from steve1.VPT.lib.actions import ActionTransformer, Buttons


REQUIRED_EPISODE_FILES = (
    "attn_patterns.npy",
    "observations.zip",
    "command.log",
    "buttons.npy",
    "camera.npy",
    "actions.log",
    "location_stats.log",
)
ACTION_PLOT_DEFAULT_VALUE = "button:attack"
CAMERA_DIRECTION_THRESHOLD_DEGREES = 3.0
CAMERA_STRONG_THRESHOLD_DEGREES = 5.0
CAMERA_GRID_SIDE = 11
CAMERA_ACTION_SCALE_DEGREES = 10.0
RUN_AGENT_WITH_LOGGING_PATH = Path(__file__).resolve().parents[2] / "steve1" / "run_agent" / "run_agent_with_logging.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollouts-root", required=True, help="Directory containing rollout folders with episode subdirs")
    parser.add_argument("--output-dir", required=True, help="Viewer output directory")
    parser.add_argument(
        "--diff-strategy",
        choices=["linear", "percentile", "asinh"],
        default="linear",
        help="How to colorize conditional-unconditional differences",
    )
    parser.add_argument("--viewer-title", default="Interactive Attention Maps", help="Main title shown at the top")
    parser.add_argument(
        "--default-episode-dir",
        default=None,
        help="Optional episode dir to load first. If omitted, the first discovered rollout is used.",
    )
    return parser.parse_args()


def discover_episode_dirs(rollouts_root: Path) -> list[Path]:
    # Queue outputs include model/prompt/seed directories, while direct runners
    # use shallower rollout/episode directories. Accept either layout.
    return sorted(
        path.parent
        for path in rollouts_root.rglob("attn_patterns.npy")
        if all((path.parent / filename).is_file() for filename in REQUIRED_EPISODE_FILES)
    )


def make_slug(episode_dir: Path, seen: set[str]) -> str:
    base = f"{episode_dir.parent.name}__{episode_dir.name}"
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", base)
    if slug not in seen:
        seen.add(slug)
        return slug

    suffix = 2
    while f"{slug}_{suffix}" in seen:
        suffix += 1
    final_slug = f"{slug}_{suffix}"
    seen.add(final_slug)
    return final_slug


def hash_file(path: Path) -> str:
    digest = hashlib.blake2b(digest_size=16)
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def parse_command_arguments(raw_command: str) -> dict[str, object]:
    tokens = shlex.split(raw_command)
    parsed: dict[str, object] = {}
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if not token.startswith("--"):
            index += 1
            continue

        key = token[2:]
        if index + 1 < len(tokens) and not tokens[index + 1].startswith("--"):
            value: object = tokens[index + 1]
            index += 2
        else:
            value = True
            index += 1

        if key in parsed:
            existing = parsed[key]
            if isinstance(existing, list):
                existing.append(value)
            else:
                parsed[key] = [existing, value]
        else:
            parsed[key] = value
    return parsed


def ensure_list(value: object | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def append_rollout_info_row(rows: list[dict[str, object]], label: str, value: object | None) -> None:
    if value is None:
        return
    if isinstance(value, list) and not value:
        return
    rows.append({"label": label, "value": value})


@lru_cache(maxsize=1)
def load_run_agent_defaults() -> dict[str, object]:
    source = RUN_AGENT_WITH_LOGGING_PATH.read_text()
    tree = ast.parse(source, filename=str(RUN_AGENT_WITH_LOGGING_PATH))
    defaults: dict[str, object] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "add_argument":
            continue
        if not node.args:
            continue
        first_arg = node.args[0]
        if not isinstance(first_arg, ast.Constant) or not isinstance(first_arg.value, str):
            continue
        if not first_arg.value.startswith("--"):
            continue
        default_kw = next((kw.value for kw in node.keywords if kw.arg == "default"), None)
        if default_kw is None:
            continue
        try:
            defaults[first_arg.value[2:]] = ast.literal_eval(default_kw)
        except Exception:
            continue
    return defaults


def format_arg_value(value: object) -> object:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, list):
        return [format_arg_value(item) for item in value]
    if value is None:
        return "None"
    return value


def value_or_default(parsed_args: dict[str, object], defaults: dict[str, object], key: str) -> object:
    if key in parsed_args:
        return format_arg_value(parsed_args[key])
    default = defaults.get(key, None)
    formatted_default = format_arg_value(default)
    if isinstance(formatted_default, list):
        if not formatted_default:
            return ["(default) none"]
        return [f"{item} (default)" for item in formatted_default]
    return f"{formatted_default} (default)"


def build_rollout_info(episode_dir: Path, meta: dict) -> dict[str, object]:
    raw_command = (episode_dir / "command.log").read_text().strip()
    parsed_args = parse_command_arguments(raw_command)
    defaults = load_run_agent_defaults()
    placements = ensure_list(parsed_args.get("placement"))
    fake_env_sources = ensure_list(parsed_args.get("fake_env_zip"))
    is_fake_rollout = bool(fake_env_sources)

    rows: list[dict[str, object]] = []
    append_rollout_info_row(rows, "Rollout Type", "Fake environment rollout" if is_fake_rollout else "Real MineRL rollout")
    append_rollout_info_row(rows, "Rollout Name", episode_dir.parent.name)
    append_rollout_info_row(rows, "Episode", episode_dir.name)
    append_rollout_info_row(rows, "Prompt", value_or_default(parsed_args, defaults, "custom_text_prompt"))
    append_rollout_info_row(rows, "Text Seed", value_or_default(parsed_args, defaults, "custom_text_seed"))
    append_rollout_info_row(rows, "Visual Prompt", value_or_default(parsed_args, defaults, "visual_prompt"))
    append_rollout_info_row(rows, "Environment Seed", value_or_default(parsed_args, defaults, "seed"))
    append_rollout_info_row(rows, "Gameplay Length", value_or_default(parsed_args, defaults, "gameplay_length"))
    append_rollout_info_row(rows, "Text Cond Scale", value_or_default(parsed_args, defaults, "text_cond_scale"))
    append_rollout_info_row(rows, "Visual Cond Scale", value_or_default(parsed_args, defaults, "visual_cond_scale"))
    append_rollout_info_row(rows, "Save Additional Data", value_or_default(parsed_args, defaults, "save_additional_data"))
    append_rollout_info_row(rows, "Fake Env Source", value_or_default(parsed_args, defaults, "fake_env_zip"))
    append_rollout_info_row(rows, "Wipe Hidden State Each Step", value_or_default(parsed_args, defaults, "wipe_hidden_state_each_step"))
    append_rollout_info_row(rows, "Placements", placements or value_or_default(parsed_args, defaults, "placement"))
    append_rollout_info_row(rows, "Model", value_or_default(parsed_args, defaults, "in_model"))
    append_rollout_info_row(rows, "Weights", value_or_default(parsed_args, defaults, "in_weights"))
    append_rollout_info_row(rows, "Prior Weights", value_or_default(parsed_args, defaults, "prior_weights"))
    append_rollout_info_row(rows, "Frame Count", meta.get("frameCount"))
    append_rollout_info_row(rows, "Layers", meta.get("layerCount"))
    append_rollout_info_row(rows, "Heads", meta.get("configuredHeads"))
    append_rollout_info_row(rows, "Logged Maps Per Layer", meta.get("loggedMapsPerLayer"))
    append_rollout_info_row(rows, "Context Positions", meta.get("contextPositions"))

    return {
        "isFakeRollout": is_fake_rollout,
        "rows": rows,
        "rawCommand": raw_command,
    }


def reshape_attention(episode_dir: Path) -> tuple[np.ndarray, dict]:
    model_cfg = load_model_config_from_command(episode_dir)
    configured_layers = int(model_cfg["n_recurrence_layers"])
    configured_heads = int(model_cfg["attention_heads"])

    raw = np.load(episode_dir / "attn_patterns.npy", allow_pickle=True)
    if raw.ndim != 3 or raw.shape[0] % configured_layers != 0:
        raise ValueError(f"Unexpected attention tensor shape in {episode_dir}: {raw.shape}")

    n_frames = raw.shape[0] // configured_layers
    logged_maps_per_layer = raw.shape[1]
    attn = raw.reshape(n_frames, configured_layers, logged_maps_per_layer, raw.shape[2])[:, :, :, 1:]
    context_positions = int(attn.shape[-1])

    if logged_maps_per_layer % configured_heads != 0:
        raise ValueError(
            f"Logged maps per layer {logged_maps_per_layer} not divisible by configured heads {configured_heads} in {episode_dir}"
        )

    branch_count = logged_maps_per_layer // configured_heads
    if branch_count == 2:
        branch_labels = ["conditional", "unconditional"]
    else:
        branch_labels = [f"branch {i}" for i in range(branch_count)]

    meta = {
        "frameCount": int(n_frames),
        "layerCount": int(configured_layers),
        "configuredHeads": int(configured_heads),
        "loggedMapsPerLayer": int(logged_maps_per_layer),
        "contextPositions": context_positions,
        "branchCount": int(branch_count),
        "branchLabels": branch_labels,
        "rawShape": list(raw.shape),
        "viewerTensorShape": list(attn.shape),
    }
    return attn, meta


def encode_float16_base64(array: np.ndarray) -> str:
    words = np.ascontiguousarray(array.astype(np.float16, copy=False)).view(np.uint16)
    return base64.b64encode(words.tobytes()).decode("ascii")


def encode_float32_base64(array: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(array.astype(np.float32, copy=False)).tobytes()).decode("ascii")


def encode_uint8_base64(array: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(array.astype(np.uint8, copy=False)).tobytes()).decode("ascii")


def stable_softmax(log_scores: np.ndarray) -> np.ndarray:
    shifted = log_scores - np.max(log_scores, axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)


def humanize_button_label(name: str) -> str:
    if name.startswith("hotbar."):
        return f"Hotbar {name.split('.', 1)[1]} probability"
    return f"{name.replace('.', ' ').title()} probability"


def humanize_actual_button_label(name: str) -> str:
    if name.startswith("hotbar."):
        return f"Actual Hotbar {name.split('.', 1)[1]}"
    return f"Actual {name.replace('.', ' ').title()}"


def parse_performed_actions(episode_dir: Path, frame_count: int) -> tuple[list[str], list[str], np.ndarray, np.ndarray]:
    with (episode_dir / "actions.log").open(newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        rows = [row for row in reader if row]

    if header is None or len(header) < 3:
        raise ValueError(f"actions.log header is missing or malformed in {episode_dir}")
    if len(rows) != frame_count:
        raise ValueError(f"actions.log frame count mismatch in {episode_dir}: {len(rows)} vs {frame_count}")

    button_names = header[:-2]
    camera_names = header[-2:]
    performed_buttons = np.asarray([[int(float(value)) for value in row[:-2]] for row in rows], dtype=np.uint8)
    performed_camera = np.asarray([[float(row[-2]), float(row[-1])] for row in rows], dtype=np.float32)

    if performed_buttons.shape != (frame_count, len(button_names)):
        raise ValueError(
            f"Unexpected performed button shape in {episode_dir}: {performed_buttons.shape} vs {(frame_count, len(button_names))}"
        )
    if performed_camera.shape != (frame_count, 2):
        raise ValueError(f"Unexpected performed camera shape in {episode_dir}: {performed_camera.shape}")

    return button_names, camera_names, performed_buttons, performed_camera


def shift_series_to_next_frame(series: np.ndarray) -> np.ndarray:
    shifted = np.full(series.shape, np.nan, dtype=np.float32)
    if len(series) > 1:
        shifted[1:] = series[:-1]
    return shifted


def compute_frame_delta(series: np.ndarray) -> np.ndarray:
    deltas = np.full(series.shape, np.nan, dtype=np.float32)
    if len(series) > 1:
        deltas[1:] = series[1:] - series[:-1]
    return deltas


def parse_location_deltas(episode_dir: Path, frame_count: int) -> dict[str, np.ndarray]:
    with (episode_dir / "location_stats.log").open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [row for row in reader]

    if not reader.fieldnames or not {"xpos", "ypos", "zpos"}.issubset(set(reader.fieldnames)):
        nan_series = np.full(frame_count, np.nan, dtype=np.float32)
        return {"xpos": nan_series.copy(), "ypos": nan_series.copy(), "zpos": nan_series.copy()}

    if len(rows) == frame_count + 1:
        rows = rows[:frame_count]
    if len(rows) != frame_count:
        raise ValueError(f"location_stats.log frame count mismatch in {episode_dir}: {len(rows)} vs {frame_count}")

    deltas: dict[str, np.ndarray] = {}
    for key in ("xpos", "ypos", "zpos"):
        values = np.asarray([float(row[key]) for row in rows], dtype=np.float32)
        deltas[key] = compute_frame_delta(values)
    return deltas


def infer_inventory_open_series(button_names: list[str], performed_buttons: np.ndarray, frame_count: int) -> np.ndarray:
    if "inventory" not in button_names:
        raise ValueError("actions.log is missing the inventory button column")

    inventory_idx = button_names.index("inventory")
    inventory_pressed = performed_buttons[:, inventory_idx].astype(bool, copy=False)
    inventory_open = np.zeros(frame_count, dtype=np.float32)
    is_open = False
    prev_pressed = False
    for t in range(frame_count - 1):
        pressed = bool(inventory_pressed[t])
        if pressed and not prev_pressed:
            is_open = not is_open
        inventory_open[t + 1] = 1.0 if is_open else 0.0
        prev_pressed = pressed
    return inventory_open


def build_action_plot_payload(
    episode_dir: Path,
    frame_count: int,
    performed_camera: np.ndarray,
    button_names: list[str],
    performed_buttons: np.ndarray,
) -> tuple[np.ndarray, list[dict], dict]:
    button_scores = np.load(episode_dir / "buttons.npy", allow_pickle=True)
    camera_scores = np.load(episode_dir / "camera.npy", allow_pickle=True)

    if button_scores.shape[0] != frame_count:
        raise ValueError(f"buttons.npy frame count mismatch in {episode_dir}: {button_scores.shape[0]} vs {frame_count}")
    if camera_scores.shape[0] != frame_count:
        raise ValueError(f"camera.npy frame count mismatch in {episode_dir}: {camera_scores.shape[0]} vs {frame_count}")

    mapping = CameraHierarchicalMapping(n_camera_bins=11)
    transformer = ActionTransformer(**ACTION_TRANSFORMER_KWARGS)

    action_series: list[np.ndarray] = []
    action_options: list[dict] = []

    location_deltas = parse_location_deltas(episode_dir, frame_count)
    delta_series_specs = [
        ("Position Delta", "delta:pos_x", "X position delta", location_deltas["xpos"]),
        ("Position Delta", "delta:pos_y", "Y position delta", location_deltas["ypos"]),
        ("Position Delta", "delta:pos_z", "Z position delta", location_deltas["zpos"]),
        ("Camera Delta", "delta:camera_pitch", "Camera pitch delta", shift_series_to_next_frame(performed_camera[:, 0])),
        ("Camera Delta", "delta:camera_yaw", "Camera yaw delta", shift_series_to_next_frame(performed_camera[:, 1])),
    ]
    for group, value, label, series in delta_series_specs:
        action_series.append(series.astype(np.float32, copy=False))
        action_options.append(
            {
                "group": group,
                "value": value,
                "label": label,
                "index": len(action_options),
            }
        )

    inventory_open = infer_inventory_open_series(button_names, performed_buttons, frame_count)
    action_series.append(inventory_open.astype(np.float32, copy=False))
    action_options.append(
        {
            "group": "State",
            "value": "state:inventory_open",
            "label": "Inventory open state",
            "index": len(action_options),
        }
    )

    actual_action_specs = [
        (
            "actual:camera_pitch",
            "Actual vertical camera move (scaled)",
            shift_series_to_next_frame((performed_camera[:, 0] / CAMERA_ACTION_SCALE_DEGREES).astype(np.float32, copy=False)),
        ),
        (
            "actual:camera_yaw",
            "Actual horizontal camera move (scaled)",
            shift_series_to_next_frame((performed_camera[:, 1] / CAMERA_ACTION_SCALE_DEGREES).astype(np.float32, copy=False)),
        ),
    ]
    for button_idx, button_name in enumerate(button_names):
        actual_action_specs.append(
            (
                f"actual_button:{button_name}",
                humanize_actual_button_label(button_name),
                shift_series_to_next_frame(performed_buttons[:, button_idx].astype(np.float32, copy=False)),
            )
        )

    for value, label, series in actual_action_specs:
        action_series.append(series.astype(np.float32, copy=False))
        action_options.append(
            {
                "group": "Actual Actions",
                "value": value,
                "label": label,
                "index": len(action_options),
            }
        )

    button_probs = stable_softmax(button_scores.astype(np.float32, copy=False)).astype(np.float32, copy=False)
    button_factored = mapping.BUTTON_IDX_TO_FACTORED.astype(np.float32, copy=False)
    button_series = button_probs @ button_factored

    for idx, button_name in enumerate(Buttons.ALL):
        action_series.append(shift_series_to_next_frame(button_series[:, idx].astype(np.float32, copy=False)))
        action_options.append(
            {
                "group": "Buttons",
                "value": f"button:{button_name}",
                "label": humanize_button_label(button_name),
                "index": len(action_options),
            }
        )

    camera_probs = stable_softmax(camera_scores.astype(np.float32, copy=False)).astype(np.float32, copy=False)
    camera_bin_values = np.array(
        [float(transformer.undiscretize_camera(np.array([i]))[0]) for i in range(mapping.n_camera_bins)],
        dtype=np.float32,
    )
    camera_factored = mapping.CAMERA_IDX_TO_FACTORED
    pitch_values = camera_bin_values[camera_factored[:, 0]]
    yaw_values = camera_bin_values[camera_factored[:, 1]]

    direction_threshold = CAMERA_DIRECTION_THRESHOLD_DEGREES
    strong_threshold = CAMERA_STRONG_THRESHOLD_DEGREES
    summary_masks = [
        ("camera_summary:active", "Camera active probability", (np.abs(pitch_values) > 1e-9) | (np.abs(yaw_values) > 1e-9)),
        ("camera_summary:neutral", "Camera neutral probability", (np.abs(pitch_values) <= 1e-9) & (np.abs(yaw_values) <= 1e-9)),
        ("camera_summary:pitch_up", "Pitch up probability", pitch_values <= -direction_threshold),
        ("camera_summary:pitch_down", "Pitch down probability", pitch_values >= direction_threshold),
        ("camera_summary:pitch_neutral", "Pitch neutral probability", np.abs(pitch_values) < direction_threshold),
        ("camera_summary:yaw_left", "Yaw left probability", yaw_values <= -direction_threshold),
        ("camera_summary:yaw_right", "Yaw right probability", yaw_values >= direction_threshold),
        ("camera_summary:yaw_neutral", "Yaw neutral probability", np.abs(yaw_values) < direction_threshold),
        ("camera_summary:strong_pitch_up", "Strong pitch up probability", pitch_values <= -strong_threshold),
        ("camera_summary:strong_pitch_down", "Strong pitch down probability", pitch_values >= strong_threshold),
        ("camera_summary:strong_yaw_left", "Strong yaw left probability", yaw_values <= -strong_threshold),
        ("camera_summary:strong_yaw_right", "Strong yaw right probability", yaw_values >= strong_threshold),
    ]

    for value, label, mask in summary_masks:
        action_series.append(shift_series_to_next_frame(camera_probs[:, mask].sum(axis=1).astype(np.float32, copy=False)))
        action_options.append(
            {
                "group": "Camera Summary",
                "value": value,
                "label": label,
                "index": len(action_options),
            }
        )

    camera_joint_start_index = len(action_options)
    for camera_idx in range(camera_probs.shape[1]):
        pitch = float(pitch_values[camera_idx])
        yaw = float(yaw_values[camera_idx])
        action_series.append(shift_series_to_next_frame(camera_probs[:, camera_idx].astype(np.float32, copy=False)))
        action_options.append(
            {
                "group": "Camera Joint",
                "value": f"camera_joint:{camera_idx}",
                "label": f"Pitch {pitch:+.2f}, Yaw {yaw:+.2f}",
                "index": len(action_options),
            }
        )

    extra_metadata = {
        "cameraGridSide": int(mapping.n_camera_bins),
        "cameraBinValues": [float(value) for value in camera_bin_values.tolist()],
        "cameraJointToFactored": mapping.CAMERA_IDX_TO_FACTORED.astype(int, copy=False).tolist(),
        "cameraJointStartIndex": int(camera_joint_start_index),
        "cameraJointCount": int(camera_probs.shape[1]),
    }
    return np.stack(action_series, axis=0), action_options, extra_metadata


def write_bundle_js(
    bundle_path: Path,
    slug: str,
    metadata: dict,
    attn_b64: str,
    action_series_b64: str,
    performed_buttons_b64: str,
    performed_camera_b64: str,
) -> None:
    bundle_payload = {
        "metadata": metadata,
        "attnBase64": attn_b64,
        "actionSeriesBase64": action_series_b64,
        "performedButtonsBase64": performed_buttons_b64,
        "performedCameraBase64": performed_camera_b64,
    }
    bundle_js = (
        "window.__ATTN_ROLLOUT_BUNDLES = window.__ATTN_ROLLOUT_BUNDLES || {};\n"
        f"window.__ATTN_ROLLOUT_BUNDLES[{json.dumps(slug)}] = {json.dumps(bundle_payload, separators=(',', ':'))};\n"
    )
    bundle_path.write_text(bundle_js)


def build_browser_html(viewer_title: str, diff_strategy: str) -> str:
    html_template = textwrap.dedent(
        """
        <!DOCTYPE html>
        <html lang="en">
        <head>
          <meta charset="utf-8">
          <meta name="viewport" content="width=device-width, initial-scale=1">
          <title>__VIEWER_TITLE__</title>
          <style>
            :root {
              color-scheme: light;
              --panel: #fffdf8;
              --panel-alt: #f7f2e8;
              --border: #c9bfaf;
              --text: #1d1a15;
              --muted: #675f56;
              --shadow: 0 14px 36px rgba(29, 26, 21, 0.09);
            }
            * {
              box-sizing: border-box;
            }
            body {
              margin: 0;
              font-family: Georgia, "Times New Roman", serif;
              color: var(--text);
              background: linear-gradient(180deg, #faf7f1 0%, #efe8dc 100%);
            }
            .page {
              max-width: 1720px;
              margin: 0 auto;
              padding: 22px;
            }
            .header {
              margin-bottom: 16px;
            }
            .header h1 {
              margin: 0;
              font-size: 28px;
              line-height: 1.05;
            }
            .top-layout {
              display: grid;
              grid-template-columns: minmax(0, 1fr) 360px;
              gap: 18px;
              align-items: start;
            }
            .panel {
              background: var(--panel);
              border: 1px solid var(--border);
              box-shadow: var(--shadow);
            }
            .heatmap-panel {
              padding: 14px;
            }
            .controls-panel {
              padding: 14px;
              display: grid;
              gap: 14px;
              align-content: start;
              background: linear-gradient(180deg, #fffdf8 0%, #f6f0e6 100%);
            }
            .control-group label {
              display: block;
              margin-bottom: 6px;
              font-size: 12px;
              color: var(--muted);
              text-transform: uppercase;
              letter-spacing: 0.04em;
            }
            .control-row-two {
              display: grid;
              grid-template-columns: repeat(2, minmax(0, 1fr));
              gap: 10px;
            }
            .control-label-row {
              display: flex;
              align-items: center;
              justify-content: space-between;
              gap: 10px;
              margin-bottom: 6px;
            }
            .control-group select {
              width: 100%;
              padding: 10px 12px;
              font-size: 15px;
              color: var(--text);
              border: 1px solid var(--border);
              background: var(--panel);
            }
            .stepper-controls {
              display: inline-flex;
              align-items: center;
              gap: 8px;
            }
            .stepper-controls .tiny-button {
              width: 28px;
              height: 28px;
            }
            .stepper-readout {
              min-width: 44px;
              text-align: center;
              font-size: 13px;
              color: var(--muted);
            }
            .toggle-controls {
              display: inline-flex;
              align-items: center;
              gap: 0;
            }
            .toggle-button {
              min-width: 52px;
              height: 28px;
              border: 1px solid var(--border);
              background: #fff;
              color: var(--muted);
              font: inherit;
              font-size: 13px;
              line-height: 1;
              cursor: pointer;
              padding: 0 10px;
            }
            .toggle-button + .toggle-button {
              margin-left: -1px;
            }
            .toggle-button.active {
              background: #f1e7d7;
              color: var(--text);
              font-weight: 600;
            }
            .tiny-button {
              width: 24px;
              height: 24px;
              border: 1px solid var(--border);
              background: #fff;
              color: var(--text);
              font-size: 16px;
              line-height: 1;
              cursor: pointer;
              padding: 0;
            }
            .action-row-stack {
              display: grid;
              gap: 8px;
            }
            .action-row {
              display: grid;
              grid-template-columns: minmax(0, 1fr) 28px;
              gap: 8px;
              align-items: center;
            }
            .action-row .tiny-button {
              width: 28px;
              height: 28px;
            }
            .readout {
              padding: 10px 12px;
              border: 1px solid var(--border);
              background: var(--panel-alt);
              font-size: 14px;
              line-height: 1.45;
            }
            .scroll-shell {
              width: 100%;
              overflow-x: auto;
              overflow-y: hidden;
              border: 1px solid #000;
              background: white;
            }
            .scroll-content {
              width: max-content;
            }
            #heatmapCanvas {
              display: block;
              image-rendering: pixelated;
              image-rendering: crisp-edges;
              cursor: crosshair;
              background: white;
            }
            #actionPlotCanvas {
              display: block;
              cursor: crosshair;
              background: #fffdf8;
              border-top: 1px solid #a99f90;
            }
            .action-legend {
              margin-top: 10px;
              display: flex;
              flex-wrap: wrap;
              gap: 10px 14px;
              min-height: 20px;
              color: var(--muted);
              font-size: 13px;
            }
            .legend-item {
              display: inline-flex;
              align-items: center;
              gap: 6px;
            }
            .legend-swatch {
              width: 12px;
              height: 12px;
              border: 1px solid rgba(0, 0, 0, 0.18);
            }
            .axis-note {
              margin-top: 10px;
              display: flex;
              flex-wrap: wrap;
              justify-content: space-between;
              gap: 10px;
              color: var(--muted);
              font-size: 13px;
            }
            .bottom-layout {
              margin-top: 18px;
              display: grid;
              grid-template-columns: minmax(0, 1.25fr) minmax(240px, 300px) minmax(220px, 1fr) minmax(220px, 1fr);
              gap: 18px;
              align-items: start;
            }
            .stats-panel {
              padding: 14px;
              font-size: 14px;
              line-height: 1.55;
              background: linear-gradient(180deg, #fffdf8 0%, #f8f2e8 100%);
            }
            .stats-tabs {
              display: flex;
              gap: 8px;
              margin: 0 0 12px;
            }
            .stats-tab-button {
              border: 1px solid var(--border);
              background: #f8f2e8;
              color: var(--muted);
              padding: 7px 10px;
              font: inherit;
              font-size: 13px;
              cursor: pointer;
            }
            .stats-tab-button.active {
              background: #fffdf8;
              color: var(--text);
              font-weight: 600;
            }
            .stats-tab-panel[hidden] {
              display: none !important;
            }
            .rollout-info-grid {
              display: grid;
              grid-template-columns: minmax(120px, 145px) minmax(0, 1fr);
              gap: 6px 12px;
              align-items: start;
            }
            .rollout-info-label {
              color: var(--muted);
              font-size: 12px;
              text-transform: uppercase;
              letter-spacing: 0.04em;
            }
            .rollout-info-value {
              min-width: 0;
              overflow-wrap: anywhere;
            }
            .rollout-info-command-title {
              margin-top: 12px;
              color: var(--muted);
              font-size: 12px;
              text-transform: uppercase;
              letter-spacing: 0.04em;
            }
            .rollout-info-command {
              margin: 6px 0 0;
              padding: 10px 12px;
              border: 1px solid var(--border);
              background: var(--panel-alt);
              font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
              font-size: 12px;
              line-height: 1.45;
              white-space: pre-wrap;
              overflow-wrap: anywhere;
            }
            .stats-panel h2,
            .frame-panel h2,
            .camera-grid-panel h2 {
              margin: 0;
              font-size: 16px;
            }
            .camera-grid-panel {
              padding: 12px;
              background: linear-gradient(180deg, #fffdf8 0%, #f7f1e8 100%);
            }
            .camera-grid-shell {
              display: grid;
              justify-items: center;
              gap: 10px;
            }
            #cameraGridCanvas {
              display: block;
              width: 220px;
              height: 220px;
              border: 1px solid var(--border);
              background: #fff;
              image-rendering: pixelated;
            }
            .camera-grid-note {
              text-align: center;
              color: var(--muted);
              font-size: 12px;
              line-height: 1.45;
            }
            .frame-panel {
              padding: 12px;
              background: linear-gradient(180deg, #fffdf8 0%, #f7f1e8 100%);
              text-align: center;
            }
            .frame-panel-header {
              display: flex;
              justify-content: space-between;
              align-items: center;
              gap: 12px;
              margin-bottom: 10px;
            }
            .frame-title {
              display: inline-flex;
              align-items: baseline;
              gap: 8px;
              flex-wrap: wrap;
            }
            .frame-title-value {
              color: var(--muted);
              font-size: 15px;
              font-weight: 400;
            }
            .frame-size-controls {
              display: inline-flex;
              align-items: center;
              gap: 6px;
            }
            .frame-size-controls button {
              width: 24px;
              height: 24px;
              border: 1px solid var(--border);
              background: #fff;
              color: var(--text);
              font-size: 16px;
              line-height: 1;
              cursor: pointer;
              padding: 0;
            }
            .frame-size-controls span {
              min-width: 44px;
              font-size: 12px;
              color: var(--muted);
              text-align: center;
            }
            .frame-panel img {
              width: 90%;
              aspect-ratio: 1 / 1;
              object-fit: contain;
              border: 1px solid var(--border);
              background: #fff;
              image-rendering: pixelated;
              display: block;
              margin: 0 auto;
            }
            #tooltip {
              position: fixed;
              display: none;
              z-index: 30;
              pointer-events: none;
              background: rgba(18, 18, 17, 0.94);
              color: #faf7f1;
              border: 1px solid rgba(255, 255, 255, 0.14);
              padding: 8px 10px;
              font-size: 12px;
              line-height: 1.45;
              white-space: normal;
              max-width: 520px;
            }
            @media (max-width: 1200px) {
              .top-layout {
                grid-template-columns: 1fr;
              }
              .bottom-layout {
                grid-template-columns: 1fr;
              }
            }
          </style>
          <script>
            window.__ATTN_VIEWER_CONFIG = {
              diffStrategy: __DIFF_STRATEGY__,
              differenceLabel: __DIFFERENCE_LABEL__
            };
          </script>
          <script src="./manifest.js"></script>
          <script src="./app.js" defer></script>
        </head>
        <body>
          <div class="page">
            <div class="header">
              <h1>__VIEWER_TITLE__</h1>
            </div>

            <div class="top-layout">
              <div class="panel heatmap-panel">
                <div class="scroll-shell" id="scrollShell">
                  <div class="scroll-content">
                    <canvas id="heatmapCanvas" width="1000" height="128"></canvas>
                    <canvas id="actionPlotCanvas" width="1000" height="150"></canvas>
                  </div>
                </div>
                <div id="actionLegend" class="action-legend"></div>
                <div class="axis-note">
                  <span id="positionAxisNote">Attention position: 0 to 127</span>
                  <span id="actionAxisNote">Selected line value: 0 to 1</span>
                  <span id="frameAxisNote">Episode frame: 0 to 999</span>
                </div>
              </div>

              <div class="panel controls-panel">
                <div class="control-group">
                  <label for="rolloutSelect">Rollout</label>
                  <select id="rolloutSelect"></select>
                </div>
                <div class="control-row-two">
                  <div class="control-group">
                    <label for="layerSelect">Layer</label>
                    <select id="layerSelect"></select>
                  </div>
                  <div class="control-group">
                    <label for="headSelect">Head</label>
                    <select id="headSelect"></select>
                  </div>
                </div>
                <div class="control-group">
                  <label for="modeSelect">View</label>
                  <select id="modeSelect"></select>
                </div>
                <div class="control-group">
                  <div class="control-label-row">
                    <label for="actionMetricRows">Metric Lines</label>
                    <button id="addActionMetricButton" class="tiny-button" type="button">+</button>
                  </div>
                  <div id="actionMetricRows" class="action-row-stack"></div>
                </div>
                <div class="control-row-two">
                  <div class="control-group">
                    <div class="control-label-row">
                      <label for="scaleReadout">Size</label>
                      <div class="stepper-controls">
                        <button id="scaleDecrease" class="tiny-button" type="button">-</button>
                        <span id="scaleReadout" class="stepper-readout">2 px</span>
                        <button id="scaleIncrease" class="tiny-button" type="button">+</button>
                      </div>
                    </div>
                  </div>
                  <div class="control-group">
                    <div class="control-label-row">
                      <label>Plot</label>
                      <div class="toggle-controls">
                        <button id="plotStyleLines" class="toggle-button active" type="button">Lines</button>
                        <button id="plotStyleBars" class="toggle-button" type="button">Bars</button>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            <div class="bottom-layout">
              <div class="panel stats-panel">
                <div class="stats-tabs">
                  <button id="hoverTabButton" class="stats-tab-button active" type="button" data-stats-tab="hover">Hover Details</button>
                  <button id="rolloutInfoTabButton" class="stats-tab-button" type="button" data-stats-tab="rollout-info">Rollout Info</button>
                </div>
                <div id="hoverTabPanel" class="stats-tab-panel">
                  <div id="hoverStats">Hover over the attention map or the action plot to inspect a value and the corresponding frames.</div>
                </div>
                <div id="rolloutInfoTabPanel" class="stats-tab-panel" hidden>
                  <div id="rolloutInfo">Select a rollout to inspect its metadata.</div>
                </div>
              </div>

              <div class="panel camera-grid-panel">
                <h2>Camera Distribution</h2>
                <div class="camera-grid-shell">
                  <canvas id="cameraGridCanvas" width="11" height="11"></canvas>
                  <div id="cameraGridCaption" class="camera-grid-note">Hover over the attention map to inspect the camera distribution for the current frame.</div>
                </div>
              </div>

              <div class="panel frame-panel">
                <div class="frame-panel-header">
                  <h2 class="frame-title">Current Frame <span id="currentFrameNumber" class="frame-title-value">-</span></h2>
                  <div class="frame-size-controls">
                    <button id="frameSizeDecrease" type="button">-</button>
                    <span id="frameSizeReadout">90%</span>
                    <button id="frameSizeIncrease" type="button">+</button>
                  </div>
                </div>
                <img id="currentFrameImg" alt="Current frame preview">
              </div>

              <div class="panel frame-panel">
                <div class="frame-panel-header">
                  <h2 class="frame-title">Attended Frame <span id="attendedFrameNumber" class="frame-title-value">-</span></h2>
                </div>
                <img id="attendedFrameImg" alt="Attended frame preview">
              </div>
            </div>
          </div>

          <div id="tooltip"></div>
        </body>
        </html>
        """
    )
    return (
        html_template.replace("__VIEWER_TITLE__", html.escape(viewer_title))
        .replace("__DIFF_STRATEGY__", json.dumps(diff_strategy))
        .replace("__DIFFERENCE_LABEL__", json.dumps(f"Difference ({diff_strategy})"))
    )


def main() -> None:
    args = parse_args()
    rollouts_root = Path(args.rollouts_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    episode_dirs = discover_episode_dirs(rollouts_root)
    if not episode_dirs:
        raise ValueError(f"No rollout episodes found under {rollouts_root}")

    default_episode_dir = Path(args.default_episode_dir).resolve() if args.default_episode_dir else None
    if default_episode_dir is not None:
        episode_dirs.sort(key=lambda path: (path.resolve() != default_episode_dir, path.parent.name, path.name))

    assets_root = output_dir / "assets"
    assets_root.mkdir(parents=True, exist_ok=True)
    shared_frames_root = output_dir / "shared_frames"
    shared_frames_root.mkdir(parents=True, exist_ok=True)
    viewer_app_source = Path(__file__).with_name("multi_rollout_viewer_app_multi_action.js")

    manifest_entries: list[dict] = []
    seen_slugs: set[str] = set()
    frame_cache: dict[str, tuple[str, int]] = {}

    for episode_dir in episode_dirs:
        slug = make_slug(episode_dir, seen_slugs)
        asset_dir = assets_root / slug
        asset_dir.mkdir(parents=True, exist_ok=True)

        attn, meta = reshape_attention(episode_dir)
        observations_zip = episode_dir / "observations.zip"
        frames_key = hash_file(observations_zip)
        if frames_key not in frame_cache:
            shared_frames_dir = shared_frames_root / frames_key
            frame_count = extract_frames(observations_zip, shared_frames_dir)
            frame_cache[frames_key] = (f"./shared_frames/{frames_key}", frame_count)
        frames_dir_rel, frame_count = frame_cache[frames_key]
        if frame_count != meta["frameCount"]:
            raise ValueError(f"Frame count mismatch in {episode_dir}: {frame_count} vs {meta['frameCount']}")

        attn_b64 = encode_float16_base64(attn)
        performed_button_names, performed_camera_names, performed_buttons, performed_camera = parse_performed_actions(
            episode_dir, meta["frameCount"]
        )
        action_series, action_options, action_plot_meta = build_action_plot_payload(
            episode_dir, meta["frameCount"], performed_camera, performed_button_names, performed_buttons
        )
        action_series_b64 = encode_float16_base64(action_series)
        performed_buttons_b64 = encode_uint8_base64(performed_buttons)
        performed_camera_b64 = encode_float32_base64(performed_camera)

        metadata = {
            **meta,
            "id": slug,
            "rolloutName": episode_dir.parent.name,
            "episodeName": episode_dir.name,
            "framesDir": frames_dir_rel,
            "framesKey": frames_key,
            "rolloutInfo": build_rollout_info(episode_dir, meta),
            "actionPlotOptions": action_options,
            "actionSeriesCount": int(action_series.shape[0]),
            "defaultActionValue": ACTION_PLOT_DEFAULT_VALUE,
            "defaultActionValues": [ACTION_PLOT_DEFAULT_VALUE],
            "performedButtonNames": performed_button_names,
            "performedCameraNames": performed_camera_names,
            **action_plot_meta,
        }
        write_bundle_js(
            asset_dir / "bundle.js",
            slug,
            metadata,
            attn_b64,
            action_series_b64,
            performed_buttons_b64,
            performed_camera_b64,
        )
        (asset_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

        manifest_entries.append(
            {
                "id": slug,
                "label": f"{episode_dir.parent.name} / {episode_dir.name}",
                "bundlePath": f"./assets/{slug}/bundle.js",
                "framesDir": frames_dir_rel,
                "rolloutName": episode_dir.parent.name,
                "episodeName": episode_dir.name,
            }
        )

        print(
            f"bundled rollout={episode_dir.parent.name} episode={episode_dir.name} "
            f"shape={tuple(meta['viewerTensorShape'])} frames={frame_count} slug={slug}"
        )

    default_rollout_id = manifest_entries[0]["id"]
    manifest_js = (
        f"window.__ATTN_ROLLOUT_MANIFEST = {json.dumps(manifest_entries, indent=2)};\n"
        f"window.__ATTN_DEFAULT_ROLLOUT = {json.dumps(default_rollout_id)};\n"
    )
    (output_dir / "manifest.js").write_text(manifest_js)
    (output_dir / "index.html").write_text(build_browser_html(args.viewer_title, args.diff_strategy))
    (output_dir / "app.js").write_text(viewer_app_source.read_text())

    print(f"rollouts_root={rollouts_root}")
    print(f"rollout_count={len(manifest_entries)}")
    print(f"default_rollout_id={default_rollout_id}")
    print(output_dir / "index.html")


if __name__ == "__main__":
    main()
