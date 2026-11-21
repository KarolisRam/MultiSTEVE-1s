import argparse
import csv
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render an L2 distance matrix CSV as a heatmap image.")
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-png", type=Path, required=True)
    parser.add_argument("--title", type=str, default="L2 Distance Heatmap")
    parser.add_argument("--width-px", type=int, default=1900)
    parser.add_argument("--height-px", type=int, default=1080)
    parser.add_argument("--dpi", type=int, default=100)
    return parser.parse_args()


def load_matrix(path: Path) -> Tuple[List[str], np.ndarray]:
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        rows = list(reader)

    if len(rows) < 2:
        raise ValueError(f"CSV has no matrix rows: {path}")

    labels = rows[0][1:]
    matrix_rows = []
    for row in rows[1:]:
        matrix_rows.append([float(value) for value in row[1:]])
    matrix = np.asarray(matrix_rows, dtype=float)

    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Expected square matrix, got shape {matrix.shape}")
    if matrix.shape[0] != len(labels):
        raise ValueError(f"Label count {len(labels)} does not match matrix shape {matrix.shape}")
    return labels, matrix


def render_heatmap(
    labels: List[str],
    matrix: np.ndarray,
    output_path: Path,
    title: str,
    width_px: int,
    height_px: int,
    dpi: int,
) -> None:
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    is_square_canvas = width_px == height_px
    figsize = (width_px / dpi, height_px / dpi)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    im = ax.imshow(
        matrix,
        cmap="magma",
        interpolation="nearest",
        aspect="equal" if is_square_canvas else "auto",
    )

    ax.set_title(title, fontsize=20, pad=14)
    ax.set_xticks(np.arange(len(labels)), labels=labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(labels)), labels=labels)
    ax.tick_params(axis="both", labelsize=12)

    colorbar = fig.colorbar(
        im,
        ax=ax,
        orientation="vertical",
        fraction=0.046 if is_square_canvas else 0.035,
        pad=0.03 if is_square_canvas else 0.02,
    )
    colorbar.set_label("L2 distance", rotation=90, labelpad=12, fontsize=13)
    colorbar.ax.tick_params(labelsize=11)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            text = ax.text(
                j,
                i,
                f"{value:.1f}",
                ha="center",
                va="center",
                color="black",
                fontsize=11,
            )
            text.set_path_effects([pe.withStroke(linewidth=2.4, foreground="white")])

    ax.set_xlabel("Model", fontsize=13)
    ax.set_ylabel("Model", fontsize=13)
    if is_square_canvas:
        fig.subplots_adjust(left=0.12, right=0.91, top=0.90, bottom=0.18)
    else:
        fig.subplots_adjust(left=0.11, right=0.94, top=0.90, bottom=0.18)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    labels, matrix = load_matrix(args.input_csv.expanduser().resolve())
    render_heatmap(
        labels=labels,
        matrix=matrix,
        output_path=args.output_png,
        title=args.title,
        width_px=args.width_px,
        height_px=args.height_px,
        dpi=args.dpi,
    )

    print(f"input_csv: {args.input_csv.expanduser().resolve()}")
    print(f"output_png: {args.output_png.expanduser().resolve()}")


if __name__ == "__main__":
    main()
