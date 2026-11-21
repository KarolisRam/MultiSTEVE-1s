import argparse
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from plot_l2_heatmap import load_matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render an L2 distance matrix as a 3D surface.")
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-png", type=Path, required=True)
    parser.add_argument("--title", type=str, default="L2 Distance Surface")
    parser.add_argument("--width-px", type=int, default=864)
    parser.add_argument("--height-px", type=int, default=864)
    parser.add_argument("--dpi", type=int, default=100)
    parser.add_argument("--elev", type=float, default=28.0)
    parser.add_argument("--azim", type=float, default=-132.0)
    parser.add_argument("--label-mode", choices=["compact", "short", "index"], default="compact")
    return parser.parse_args()


def transform_labels(labels: List[str], label_mode: str) -> tuple[List[str], str, str, float, float, float]:
    if label_mode == "compact":
        return labels, "Init Seed X", "Init Seed Y", 62.0, 8.0, 8.5
    if label_mode == "short":
        short = [label.replace("seed", "s") for label in labels]
        return short, "Init Seed X", "Init Seed Y", 42.0, 9.0, 9.0
    if label_mode == "index":
        index_labels = [str(i + 1) for i in range(len(labels))]
        return index_labels, "Init Seed Index X", "Init Seed Index Y", 0.0, 10.0, 10.0
    raise ValueError(f"Unsupported label mode: {label_mode}")


def render_surface(
    labels: List[str],
    matrix: np.ndarray,
    output_path: Path,
    title: str,
    width_px: int,
    height_px: int,
    dpi: int,
    elev: float = 28.0,
    azim: float = -132.0,
    triangle: str = "upper",
    z_min: float | None = None,
    z_max: float | None = None,
    color_min: float | None = None,
    color_max: float | None = None,
    label_mode: str = "compact",
) -> None:
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figsize = (width_px / dpi, height_px / dpi)
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")

    x = np.arange(len(labels), dtype=float)
    y = np.arange(len(labels), dtype=float)
    xx, yy = np.meshgrid(x, y)
    zz = np.array(matrix, dtype=float, copy=True)
    if triangle == "upper":
        zz[np.tril_indices_from(zz, k=0)] = np.nan
    elif triangle == "lower":
        zz[np.triu_indices_from(zz, k=0)] = np.nan
    elif triangle != "full":
        raise ValueError(f"Unsupported triangle mode: {triangle}")

    valid_values = zz[~np.isnan(zz)]
    if valid_values.size == 0:
        raise ValueError("No valid surface values remain after masking")
    z_lower = float(valid_values.min()) if z_min is None else float(z_min)
    z_upper = float(valid_values.max()) if z_max is None else float(z_max)
    c_lower = z_lower if color_min is None else float(color_min)
    c_upper = z_upper if color_max is None else float(color_max)
    if z_upper <= z_lower:
        z_upper = z_lower + 1.0
    if c_upper <= c_lower:
        c_upper = c_lower + 1.0

    tick_labels, x_axis_label, y_axis_label, x_rotation, x_fontsize, y_fontsize = transform_labels(labels, label_mode)

    surface = ax.plot_surface(
        xx,
        yy,
        zz,
        cmap="magma",
        vmin=c_lower,
        vmax=c_upper,
        linewidth=0.35,
        edgecolor=(0.1, 0.1, 0.1, 0.3),
        antialiased=True,
        rstride=1,
        cstride=1,
    )

    ax.set_title(title, fontsize=18, pad=18)
    ax.set_xlabel(x_axis_label, fontsize=12, labelpad=10)
    ax.set_ylabel(y_axis_label, fontsize=12, labelpad=12)
    ax.set_zlabel("L2 distance", fontsize=12, labelpad=8)

    ax.set_xticks(x)
    ax.set_yticks(y)
    ax.set_xticklabels(tick_labels, rotation=x_rotation, ha="right" if x_rotation else "center", fontsize=x_fontsize)
    ax.set_yticklabels(tick_labels, fontsize=y_fontsize, ha="right")
    ax.tick_params(axis="x", pad=1)
    ax.tick_params(axis="y", pad=1)

    ax.set_zlim(z_lower, z_upper)
    ax.view_init(elev=elev, azim=azim)
    ax.set_box_aspect((1.0, 1.0, 0.55))

    ax.xaxis.pane.set_facecolor((0.96, 0.96, 0.96, 1.0))
    ax.yaxis.pane.set_facecolor((0.96, 0.96, 0.96, 1.0))
    ax.zaxis.pane.set_facecolor((0.985, 0.985, 0.985, 1.0))
    ax.grid(True, alpha=0.28)

    colorbar = fig.colorbar(surface, ax=ax, shrink=0.62, aspect=22, pad=0.06)
    colorbar.set_label("L2 distance", rotation=90, labelpad=10, fontsize=12)
    colorbar.ax.tick_params(labelsize=10)

    fig.subplots_adjust(left=0.18, right=0.94, top=0.92, bottom=0.04)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    labels, matrix = load_matrix(args.input_csv.expanduser().resolve())
    render_surface(
        labels=labels,
        matrix=matrix,
        output_path=args.output_png,
        title=args.title,
        width_px=args.width_px,
        height_px=args.height_px,
        dpi=args.dpi,
        elev=args.elev,
        azim=args.azim,
        label_mode=args.label_mode,
    )
    print(f"input_csv: {args.input_csv.expanduser().resolve()}")
    print(f"output_png: {args.output_png.expanduser().resolve()}")


if __name__ == "__main__":
    main()
