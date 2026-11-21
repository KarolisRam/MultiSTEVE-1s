import argparse
import base64
import html
import json
import re
import textwrap
from pathlib import Path

import numpy as np

try:
    from .build_attention_hover_viewer import extract_frames, load_model_config_from_command
except ImportError:
    from build_attention_hover_viewer import extract_frames, load_model_config_from_command
from steve1.VPT.agent import ACTION_TRANSFORMER_KWARGS
from steve1.VPT.lib.action_mapping import CameraHierarchicalMapping
from steve1.VPT.lib.actions import ActionTransformer, Buttons


REQUIRED_EPISODE_FILES = ("attn_patterns.npy", "observations.zip", "command.log", "buttons.npy", "camera.npy")
ACTION_PLOT_DEFAULT_VALUE = "button:attack"
CAMERA_DIRECTION_THRESHOLD_DEGREES = 3.0
CAMERA_STRONG_THRESHOLD_DEGREES = 5.0


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
    episodes: list[Path] = []
    for rollout_dir in sorted(path for path in rollouts_root.iterdir() if path.is_dir()):
        for episode_dir in sorted(path for path in rollout_dir.iterdir() if path.is_dir()):
            if all((episode_dir / filename).exists() for filename in REQUIRED_EPISODE_FILES):
                episodes.append(episode_dir)
    return episodes


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


def stable_softmax(log_scores: np.ndarray) -> np.ndarray:
    shifted = log_scores - np.max(log_scores, axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)


def humanize_button_label(name: str) -> str:
    if name.startswith("hotbar."):
        return f"Hotbar {name.split('.', 1)[1]} probability"
    return f"{name.replace('.', ' ').title()} probability"


def build_action_plot_payload(episode_dir: Path, frame_count: int) -> tuple[np.ndarray, list[dict]]:
    button_scores = np.load(episode_dir / "buttons.npy", allow_pickle=True)
    camera_scores = np.load(episode_dir / "camera.npy", allow_pickle=True)

    if button_scores.shape[0] != frame_count:
        raise ValueError(f"buttons.npy frame count mismatch in {episode_dir}: {button_scores.shape[0]} vs {frame_count}")
    if camera_scores.shape[0] != frame_count:
        raise ValueError(f"camera.npy frame count mismatch in {episode_dir}: {camera_scores.shape[0]} vs {frame_count}")

    mapping = CameraHierarchicalMapping(n_camera_bins=11)
    transformer = ActionTransformer(**ACTION_TRANSFORMER_KWARGS)

    button_probs = stable_softmax(button_scores.astype(np.float32, copy=False)).astype(np.float32, copy=False)
    button_factored = mapping.BUTTON_IDX_TO_FACTORED.astype(np.float32, copy=False)
    button_series = button_probs @ button_factored

    action_series: list[np.ndarray] = []
    action_options: list[dict] = []

    for idx, button_name in enumerate(Buttons.ALL):
        action_series.append(button_series[:, idx].astype(np.float32, copy=False))
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
        action_series.append(camera_probs[:, mask].sum(axis=1).astype(np.float32, copy=False))
        action_options.append(
            {
                "group": "Camera Summary",
                "value": value,
                "label": label,
                "index": len(action_options),
            }
        )

    for camera_idx in range(camera_probs.shape[1]):
        pitch = float(pitch_values[camera_idx])
        yaw = float(yaw_values[camera_idx])
        action_series.append(camera_probs[:, camera_idx].astype(np.float32, copy=False))
        action_options.append(
            {
                "group": "Camera Joint",
                "value": f"camera_joint:{camera_idx}",
                "label": f"Pitch {pitch:+.2f}, Yaw {yaw:+.2f}",
                "index": len(action_options),
            }
        )

    return np.stack(action_series, axis=0), action_options


def write_bundle_js(bundle_path: Path, slug: str, metadata: dict, attn_b64: str, action_series_b64: str) -> None:
    bundle_payload = {
        "metadata": metadata,
        "attnBase64": attn_b64,
        "actionSeriesBase64": action_series_b64,
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
            .control-group select {
              width: 100%;
              padding: 10px 12px;
              font-size: 15px;
              color: var(--text);
              border: 1px solid var(--border);
              background: var(--panel);
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
              grid-template-columns: minmax(0, 1.2fr) minmax(220px, 1fr) minmax(220px, 1fr);
              gap: 18px;
              align-items: start;
            }
            .stats-panel {
              padding: 14px;
              font-size: 14px;
              line-height: 1.55;
              background: linear-gradient(180deg, #fffdf8 0%, #f8f2e8 100%);
              min-height: 100%;
            }
            .stats-panel h2,
            .frame-panel h2 {
              margin: 0;
              font-size: 16px;
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
            .caption {
              margin-top: 8px;
              color: var(--muted);
              font-size: 13px;
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
              white-space: nowrap;
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
                <div class="axis-note">
                  <span id="positionAxisNote">Attention position: 0 to 127</span>
                  <span id="actionAxisNote">Selected action probability: 0 to 1</span>
                  <span id="frameAxisNote">Episode frame: 0 to 999</span>
                </div>
              </div>

              <div class="panel controls-panel">
                <div class="control-group">
                  <label for="rolloutSelect">Rollout</label>
                  <select id="rolloutSelect"></select>
                </div>
                <div class="control-group">
                  <label for="layerSelect">Layer</label>
                  <select id="layerSelect"></select>
                </div>
                <div class="control-group">
                  <label for="headSelect">Head</label>
                  <select id="headSelect"></select>
                </div>
                <div class="control-group">
                  <label for="modeSelect">View</label>
                  <select id="modeSelect"></select>
                </div>
                <div class="control-group">
                  <label for="actionMetricSelect">Action Plot</label>
                  <select id="actionMetricSelect"></select>
                </div>
                <div class="control-group">
                  <label for="scaleSelect">Cell Size</label>
                  <select id="scaleSelect">
                    <option value="1">1 px per cell</option>
                    <option value="2" selected>2 px per cell</option>
                    <option value="3">3 px per cell</option>
                    <option value="4">4 px per cell</option>
                    <option value="5">5 px per cell</option>
                  </select>
                </div>
                <div id="modeReadout" class="readout"></div>
              </div>
            </div>

            <div class="bottom-layout">
              <div class="panel stats-panel">
                <h2>Hover Details</h2>
                <div id="hoverStats">Hover over the attention map or the action plot to inspect a value and the corresponding frames.</div>
              </div>

              <div class="panel frame-panel">
                <div class="frame-panel-header">
                  <h2>Current Frame</h2>
                  <div class="frame-size-controls">
                    <button id="frameSizeDecrease" type="button">-</button>
                    <span id="frameSizeReadout">90%</span>
                    <button id="frameSizeIncrease" type="button">+</button>
                  </div>
                </div>
                <img id="currentFrameImg" alt="Current frame preview">
                <div id="currentFrameCaption" class="caption">Frame -</div>
              </div>

              <div class="panel frame-panel">
                <div class="frame-panel-header">
                  <h2>Attended Frame</h2>
                </div>
                <img id="attendedFrameImg" alt="Attended frame preview">
                <div id="attendedFrameCaption" class="caption">Frame -</div>
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
    viewer_app_source = Path(__file__).with_name("multi_rollout_viewer_app.js")

    manifest_entries: list[dict] = []
    seen_slugs: set[str] = set()

    for episode_dir in episode_dirs:
        slug = make_slug(episode_dir, seen_slugs)
        asset_dir = assets_root / slug
        asset_dir.mkdir(parents=True, exist_ok=True)

        attn, meta = reshape_attention(episode_dir)
        frames_dir = asset_dir / "frames"
        frame_count = extract_frames(episode_dir / "observations.zip", frames_dir)
        if frame_count != meta["frameCount"]:
            raise ValueError(f"Frame count mismatch in {episode_dir}: {frame_count} vs {meta['frameCount']}")

        attn_b64 = encode_float16_base64(attn)
        action_series, action_options = build_action_plot_payload(episode_dir, meta["frameCount"])
        action_series_b64 = encode_float16_base64(action_series)

        metadata = {
            **meta,
            "id": slug,
            "rolloutName": episode_dir.parent.name,
            "episodeName": episode_dir.name,
            "framesDir": f"./assets/{slug}/frames",
            "actionPlotOptions": action_options,
            "actionSeriesCount": int(action_series.shape[0]),
            "defaultActionValue": ACTION_PLOT_DEFAULT_VALUE,
        }
        write_bundle_js(asset_dir / "bundle.js", slug, metadata, attn_b64, action_series_b64)
        (asset_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

        manifest_entries.append(
            {
                "id": slug,
                "label": f"{episode_dir.parent.name} / {episode_dir.name}",
                "bundlePath": f"./assets/{slug}/bundle.js",
                "framesDir": f"./assets/{slug}/frames",
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
