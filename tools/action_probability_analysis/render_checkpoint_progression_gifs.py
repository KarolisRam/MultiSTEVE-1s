import argparse
import csv
import io
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from PIL import Image

DEFAULT_SCREEN_WIDTH = 1900
DEFAULT_SCREEN_HEIGHT = 1080
DEFAULT_DPI = 100
EARLIEST_GRAY = 0.5
LATEST_GRAY = 0.1
TRAIL_LIGHTEN_STEP = 0.12
FINAL_LINEWIDTH = 3.4
ANIMATED_LINEWIDTH = 2.0
TRAIL_LINEWIDTH = 1.6
DEFAULT_DURATION_SECONDS = 6.0
DEFAULT_FINAL_HOLD_SECONDS = 2.0
DEFAULT_TRAIL_LENGTH = 5


def _load_manifest(experiment_dir):
    manifest_path = os.path.join(experiment_dir, "manifest.json")
    with open(manifest_path) as f:
        return json.load(f), manifest_path


def _load_curve_csv(csv_path):
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    frames = data[:, 0]
    checkpoint_steps = [column.removeprefix("ckpt_") for column in header[1:]]
    series = [(checkpoint_step, data[:, idx + 1]) for idx, checkpoint_step in enumerate(checkpoint_steps)]
    return frames, series


def _plot_dimensions(manifest):
    dims = manifest.get("plot_dimensions_pixels")
    if dims is None:
        return DEFAULT_SCREEN_WIDTH * 0.9 / DEFAULT_DPI, DEFAULT_SCREEN_HEIGHT * 0.9 / DEFAULT_DPI, DEFAULT_DPI
    return dims["width"] / dims["dpi"], dims["height"] / dims["dpi"], dims["dpi"]


def _checkpoint_gray_value(index, total):
    if total <= 1:
        return LATEST_GRAY
    frac = index / (total - 1)
    gray = EARLIEST_GRAY + frac * (LATEST_GRAY - EARLIEST_GRAY)
    return gray


def _trail_gray_value(index, total, age_from_current):
    base_gray = _checkpoint_gray_value(index, total)
    return min(0.9, base_gray + age_from_current * TRAIL_LIGHTEN_STEP)


def _legend_handles(current_color=None, trail_color=None, converged=False):
    if converged:
        return [
            Line2D([0], [0], color="black", linewidth=FINAL_LINEWIDTH, label="Final checkpoint"),
        ]
    return [
        Line2D([0], [0], color="black", linewidth=FINAL_LINEWIDTH, label="Final checkpoint"),
        Line2D([0], [0], color=current_color, linewidth=ANIMATED_LINEWIDTH, label="Current checkpoint"),
        Line2D([0], [0], color=trail_color, linewidth=TRAIL_LINEWIDTH, label="Recent trail"),
    ]


def _render_frame_image(
    width_in,
    height_in,
    dpi,
    title,
    y_label,
    x,
    checkpoint_series,
    checkpoint_index,
    trail_length,
):
    fig, ax = plt.subplots(figsize=(width_in, height_in), dpi=dpi)
    total = len(checkpoint_series)
    current_step, current_series = checkpoint_series[checkpoint_index]
    final_step, final_series = checkpoint_series[-1]
    converged = checkpoint_index == total - 1
    current_gray = _checkpoint_gray_value(checkpoint_index, total)
    current_color = str(current_gray)
    trail_start = max(0, checkpoint_index - trail_length + 1)
    trail_entries = checkpoint_series[trail_start : checkpoint_index + 1]

    ymax = max(float(series.max()) for _, series in checkpoint_series)
    ax.plot(x, final_series, color="black", linewidth=FINAL_LINEWIDTH, zorder=2)
    trail_legend_color = str(min(0.9, current_gray + TRAIL_LIGHTEN_STEP))
    if not converged:
        for trail_offset, (_, trail_series) in enumerate(trail_entries[:-1]):
            trail_index = trail_start + trail_offset
            age_from_current = checkpoint_index - trail_index
            trail_color = str(_trail_gray_value(trail_index, total, age_from_current))
            ax.plot(
                x,
                trail_series,
                color=trail_color,
                linewidth=TRAIL_LINEWIDTH,
                zorder=3 + trail_offset,
            )
        ax.plot(x, current_series, color=current_color, linewidth=ANIMATED_LINEWIDTH, zorder=10)

    ax.set_xlim(0, len(x) - 1)
    ax.set_ylim(0.0, min(1.02, max(0.05, ymax * 1.05)))
    ax.set_xlabel("Frame")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(
        handles=_legend_handles(current_color, trail_legend_color, converged=converged),
        loc="center left",
        bbox_to_anchor=(0.80, 0.55),
        frameon=False,
        fontsize=11,
    )
    if converged:
        status_text = f"Converged: {final_step}\nFinal frame"
    else:
        status_text = (
            f"Current: {current_step}\nFinal: {final_step}\n"
            f"Frame {checkpoint_index + 1}/{total}\nTrail: {min(trail_length, checkpoint_index + 1)} checkpoints"
        )
    ax.text(
        0.80,
        0.24,
        status_text,
        transform=ax.transAxes,
        fontsize=12,
        va="top",
        ha="left",
    )
    fig.subplots_adjust(left=0.07, right=0.78, top=0.92, bottom=0.10)

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi)
    plt.close(fig)
    buffer.seek(0)
    image = Image.open(buffer).convert("RGB")
    image.load()
    buffer.close()
    return image


def _save_gif(output_path, frames, total_duration_seconds, final_hold_seconds):
    per_frame_duration_ms = max(20, int(round((total_duration_seconds / len(frames)) * 1000)))
    durations = [per_frame_duration_ms] * len(frames)
    durations[-1] = max(per_frame_duration_ms, int(round(final_hold_seconds * 1000)))
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=False,
        disposal=2,
    )


def _render_gif_for_curve(
    csv_path,
    output_path,
    title,
    y_label,
    width_in,
    height_in,
    dpi,
    duration_seconds,
    final_hold_seconds,
    trail_length,
):
    x, checkpoint_series = _load_curve_csv(csv_path)
    frames = [
        _render_frame_image(
            width_in=width_in,
            height_in=height_in,
            dpi=dpi,
            title=title,
            y_label=y_label,
            x=x,
            checkpoint_series=checkpoint_series,
            checkpoint_index=checkpoint_index,
            trail_length=trail_length,
        )
        for checkpoint_index in range(len(checkpoint_series))
    ]
    _save_gif(
        output_path,
        frames,
        total_duration_seconds=duration_seconds,
        final_hold_seconds=final_hold_seconds,
    )
    return checkpoint_series[-1][0], len(checkpoint_series)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_dir", required=True)
    parser.add_argument("--duration_seconds", type=float, default=DEFAULT_DURATION_SECONDS)
    parser.add_argument("--final_hold_seconds", type=float, default=DEFAULT_FINAL_HOLD_SECONDS)
    parser.add_argument("--trail_length", type=int, default=DEFAULT_TRAIL_LENGTH)
    args = parser.parse_args()

    manifest, manifest_path = _load_manifest(args.experiment_dir)
    width_in, height_in, dpi = _plot_dimensions(manifest)

    gifs_dir = os.path.join(args.experiment_dir, "gifs")
    gifs_buttons_dir = os.path.join(gifs_dir, "buttons")
    gifs_camera_dir = os.path.join(gifs_dir, "camera")
    os.makedirs(gifs_dir, exist_ok=True)
    os.makedirs(gifs_buttons_dir, exist_ok=True)
    os.makedirs(gifs_camera_dir, exist_ok=True)

    gif_manifest = {
        "created_at": manifest["created_at"],
        "source_manifest": manifest_path,
        "duration_seconds": args.duration_seconds,
        "final_hold_seconds": args.final_hold_seconds,
        "trail_length": args.trail_length,
        "button_gifs": {},
        "camera_gifs": {},
    }

    for button_name, entry in manifest["button_plots"].items():
        slug = button_name.replace(".", "_")
        gif_path = os.path.join(gifs_buttons_dir, f"{slug}.gif")
        _render_gif_for_curve(
            csv_path=entry["csv"],
            output_path=gif_path,
            title=f"Button Probability Through Training: {button_name.replace('.', ' ').title()}",
            y_label="Button probability",
            width_in=width_in,
            height_in=height_in,
            dpi=dpi,
            duration_seconds=args.duration_seconds,
            final_hold_seconds=args.final_hold_seconds,
            trail_length=args.trail_length,
        )
        gif_manifest["button_gifs"][button_name] = gif_path
        if button_name == "inventory":
            inventory_gif_path = os.path.join(gifs_dir, "inventory_probability_overlay.gif")
            _render_gif_for_curve(
                csv_path=entry["csv"],
                output_path=inventory_gif_path,
                title="Button Probability Through Training: Inventory",
                y_label="Button probability",
                width_in=width_in,
                height_in=height_in,
                dpi=dpi,
                duration_seconds=args.duration_seconds,
                final_hold_seconds=args.final_hold_seconds,
                trail_length=args.trail_length,
            )
            gif_manifest["inventory_gif"] = inventory_gif_path

    for direction, entry in manifest["camera_plots"].items():
        gif_path = os.path.join(gifs_camera_dir, f"{direction}.gif")
        _render_gif_for_curve(
            csv_path=entry["csv"],
            output_path=gif_path,
            title=f"Camera Probability Through Training: {direction.replace('_', ' ').title()}",
            y_label="Camera action probability",
            width_in=width_in,
            height_in=height_in,
            dpi=dpi,
            duration_seconds=args.duration_seconds,
            final_hold_seconds=args.final_hold_seconds,
            trail_length=args.trail_length,
        )
        gif_manifest["camera_gifs"][direction] = gif_path

    with open(os.path.join(gifs_dir, "manifest.json"), "w") as f:
        json.dump(gif_manifest, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
