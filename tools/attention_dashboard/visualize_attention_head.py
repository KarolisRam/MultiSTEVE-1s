import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode-dir", required=True, help="Episode directory containing attn_patterns.npy")
    parser.add_argument("--layer", type=int, required=True, help="Layer index after reshape")
    parser.add_argument("--head", type=int, required=True, help="Head index after reshape")
    parser.add_argument("--output", required=True, help="Output PNG path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    episode_dir = Path(args.episode_dir)
    attn_path = episode_dir / "attn_patterns.npy"

    raw = np.load(attn_path, allow_pickle=True)
    if raw.ndim != 3 or raw.shape[0] % 4 != 0:
        raise ValueError(f"Unexpected attention tensor shape: {raw.shape}")

    attn = raw.reshape(-1, 4, raw.shape[1], raw.shape[2])
    n_frames, n_layers, n_heads, context = attn.shape

    if not 0 <= args.layer < n_layers:
        raise ValueError(f"Layer {args.layer} out of range 0..{n_layers - 1}")
    if not 0 <= args.head < n_heads:
        raise ValueError(f"Head {args.head} out of range 0..{n_heads - 1}")

    # Drop the first position so the visualization is 128 x 1000 rather than 129 x 1000.
    grid = attn[:, args.layer, args.head, 1:].T
    if grid.shape != (128, n_frames):
        raise ValueError(f"Unexpected visualization grid shape: {grid.shape}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(20, 6), dpi=200)
    ax.imshow(grid, cmap="gray_r", vmin=0.0, vmax=1.0, aspect="auto", interpolation="nearest", origin="upper")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Attention Position")
    ax.set_title(f"Layer {args.layer}, Head {args.head} ({grid.shape[0]} x {grid.shape[1]})")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

    print(f"raw_shape={raw.shape}")
    print(f"reshaped_shape={attn.shape}")
    print(f"layers={n_layers}")
    print(f"heads={n_heads}")
    print(f"grid_shape={grid.shape}")
    print(output_path)


if __name__ == "__main__":
    main()
