import argparse
import json
import pickle
import shlex
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode-dir", required=True, help="Episode directory containing attn_patterns.npy")
    parser.add_argument("--output-dir", required=True, help="Directory to write PNGs and metadata")
    return parser.parse_args()


def load_model_config_from_command(episode_dir: Path) -> dict:
    command_text = (episode_dir / "command.log").read_text().strip()
    argv = shlex.split(command_text)
    in_model = None
    if "--in_model" in argv:
        idx = argv.index("--in_model")
        if idx + 1 < len(argv):
            in_model = argv[idx + 1]
    if in_model is None:
        raise ValueError(f"Could not find --in_model in {episode_dir / 'command.log'}")

    with open(in_model, "rb") as f:
        agent_parameters = pickle.load(f)
    return agent_parameters["model"]["args"]["net"]["args"]


def save_grid_png(grid: np.ndarray, output_path: Path) -> None:
    if grid.ndim != 2:
        raise ValueError(f"Expected 2D grid, got shape {grid.shape}")
    pixels = np.clip(np.rint((1.0 - grid) * 255.0), 0, 255).astype(np.uint8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels, mode="L").save(output_path)


def main() -> None:
    args = parse_args()
    episode_dir = Path(args.episode_dir)
    output_dir = Path(args.output_dir)
    attn_path = episode_dir / "attn_patterns.npy"

    raw = np.load(attn_path, allow_pickle=True)
    model_cfg = load_model_config_from_command(episode_dir)
    configured_layers = int(model_cfg["n_recurrence_layers"])
    configured_heads = int(model_cfg["attention_heads"])

    if raw.ndim != 3:
        raise ValueError(f"Unexpected attention tensor ndim: {raw.ndim}")
    if raw.shape[0] % configured_layers != 0:
        raise ValueError(
            f"First dimension {raw.shape[0]} is not divisible by configured layer count {configured_layers}"
        )

    n_frames = raw.shape[0] // configured_layers
    logged_maps_per_layer = raw.shape[1]
    context = raw.shape[2]
    attn = raw.reshape(n_frames, configured_layers, logged_maps_per_layer, context)

    branch_count = None
    branch_labels = None
    if logged_maps_per_layer % configured_heads == 0:
        branch_count = logged_maps_per_layer // configured_heads
        if branch_count == 2:
            branch_labels = ["conditional", "unconditional"]
        else:
            branch_labels = [f"branch{i}" for i in range(branch_count)]

    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "episode_dir": str(episode_dir),
        "attn_path": str(attn_path),
        "raw_shape": list(raw.shape),
        "reshaped_shape": list(attn.shape),
        "configured_layers": configured_layers,
        "configured_attention_heads_per_layer": configured_heads,
        "logged_maps_per_layer": logged_maps_per_layer,
        "logged_maps_total": configured_layers * logged_maps_per_layer,
        "visualized_grid_shape": [context - 1, n_frames],
        "branch_count": branch_count,
        "branch_labels": branch_labels,
        "note": (
            "When cond_scale is used, the policy duplicates batch size for classifier-free guidance. "
            "That can make logged attention maps per layer exceed configured architectural heads."
        ),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    for layer in range(configured_layers):
        layer_maps = attn[:, layer, :, 1:]
        layer_dir = output_dir / f"layer_{layer:02d}"
        layer_dir.mkdir(parents=True, exist_ok=True)

        for logged_idx in range(logged_maps_per_layer):
            grid = layer_maps[:, logged_idx, :].T

            if branch_count and branch_labels:
                branch_idx = logged_idx // configured_heads
                head_idx = logged_idx % configured_heads
                filename = (
                    f"layer{layer:02d}_map{logged_idx:02d}_"
                    f"{branch_labels[branch_idx]}_head{head_idx:02d}.png"
                )
            else:
                filename = f"layer{layer:02d}_map{logged_idx:02d}.png"

            save_grid_png(grid, layer_dir / filename)

        max_grid = np.max(layer_maps, axis=1).T
        save_grid_png(max_grid, output_dir / f"layer{layer:02d}_max_over_logged_maps.png")

    print(f"raw_shape={raw.shape}")
    print(f"reshaped_shape={attn.shape}")
    print(f"configured_layers={configured_layers}")
    print(f"configured_attention_heads_per_layer={configured_heads}")
    print(f"logged_maps_per_layer={logged_maps_per_layer}")
    print(f"logged_maps_total={configured_layers * logged_maps_per_layer}")
    if branch_count is not None:
        print(f"branch_count={branch_count}")
    print(output_dir)


if __name__ == "__main__":
    main()
