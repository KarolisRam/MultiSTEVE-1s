import csv
import json
from pathlib import Path

import numpy as np
from plotly.offline.offline import get_plotlyjs

DEFAULT_SCREEN_WIDTH = 1900
DEFAULT_SCREEN_HEIGHT = 1080


def load_manifest(experiment_dir):
    manifest_path = Path(experiment_dir) / "manifest.json"
    return json.loads(manifest_path.read_text()), manifest_path


def load_curve_csv(csv_path):
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    frames = data[:, 0].astype(int)
    checkpoint_steps = np.array([int(column.removeprefix("ckpt_")) for column in header[1:]], dtype=int)
    series = data[:, 1:].T
    return frames, checkpoint_steps, series


def plot_dimensions(manifest):
    dims = manifest.get("plot_dimensions_pixels")
    if dims is None:
        return int(DEFAULT_SCREEN_WIDTH * 0.9), int(DEFAULT_SCREEN_HEIGHT * 0.9)
    return int(dims["width"]), int(dims["height"])


def slugify_action_name(action_name):
    return action_name.replace(".", "_").replace(" ", "_")


def pretty_action_name(action_name):
    return action_name.replace(".", " ").replace("_", " ").title()


def format_step_compact(step):
    if step >= 1_000_000:
        return f"{step / 1_000_000:.1f}M"
    if step >= 1_000:
        return f"{step / 1_000:.1f}k"
    return f"{step}"


def guide_indices(num_checkpoints):
    raw = [
        0,
        round((num_checkpoints - 1) * 0.25),
        round((num_checkpoints - 1) * 0.50),
        round((num_checkpoints - 1) * 0.75),
        num_checkpoints - 1,
    ]
    indices = []
    for index in raw:
        if index not in indices:
            indices.append(index)
    return indices


def tick_indices(num_checkpoints, max_ticks=8):
    raw = np.linspace(0, num_checkpoints - 1, num=min(max_ticks, num_checkpoints), dtype=int).tolist()
    indices = []
    for index in raw:
        if index not in indices:
            indices.append(index)
    if indices[-1] != num_checkpoints - 1:
        indices.append(num_checkpoints - 1)
    return indices


def write_plotly_assets(output_dir, extra_js=None, css_text=None):
    assets_dir = Path(output_dir) / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    (assets_dir / "plotly.min.js").write_text(get_plotlyjs())
    if extra_js is not None:
        (assets_dir / "app.js").write_text(extra_js)
    if css_text is not None:
        (assets_dir / "styles.css").write_text(css_text)
    return assets_dir
