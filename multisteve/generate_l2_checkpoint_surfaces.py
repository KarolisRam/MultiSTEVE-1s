import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

from generate_l2_checkpoint_heatmaps import (
    build_gif,
    common_snapshot_steps,
    label_for_run_dir,
    select_run_dirs,
    weight_paths_for_step,
)
from measure_model_l2_matrix import compute_matrix, load_state_dict, validate_models
from plot_l2_surface import render_surface


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate checkpoint-by-checkpoint 3D L2 surfaces and assemble them into a GIF."
    )
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--chunk-elements", type=int, default=1_000_000)
    parser.add_argument("--width-px", type=int, default=864)
    parser.add_argument("--height-px", type=int, default=864)
    parser.add_argument("--dpi", type=int, default=100)
    parser.add_argument("--elev", type=float, default=28.0)
    parser.add_argument("--elev-end", type=float)
    parser.add_argument("--azim", type=float, default=-132.0)
    parser.add_argument("--azim-end", type=float)
    parser.add_argument("--label-mode", choices=["compact", "short", "index"], default="compact")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--matrix-dir", type=Path)
    parser.add_argument("--gif-path", type=Path, required=True)
    parser.add_argument("--first-frame-ms", type=int, default=1000)
    parser.add_argument("--middle-frame-ms", type=int, default=200)
    parser.add_argument("--last-frame-ms", type=int, default=3000)
    parser.add_argument("--max-steps", type=int)
    return parser.parse_args()


def matrix_to_numpy(matrix) -> np.ndarray:
    return np.asarray(matrix.tolist(), dtype=float)


def main() -> None:
    args = parse_args()
    start_time = time.perf_counter()

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    gif_path = args.gif_path.expanduser().resolve()
    matrix_dir = args.matrix_dir.expanduser().resolve() if args.matrix_dir else (output_dir / "matrices")
    matrix_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = select_run_dirs(args.experiment_dir, args.limit)
    labels = [label_for_run_dir(run_dir) for run_dir in run_dirs]
    steps = common_snapshot_steps(run_dirs)
    if args.max_steps is not None:
        steps = steps[: args.max_steps]

    manifest: Dict[str, object] = {
        "experiment_dir": str(args.experiment_dir.expanduser().resolve()),
        "labels": labels,
        "run_dirs": [str(run_dir) for run_dir in run_dirs],
        "steps": steps,
        "frames": [],
        "view": {
            "elev_start": args.elev,
            "elev_end": args.elev_end,
            "azim_start": args.azim,
            "azim_end": args.azim_end,
        },
        "triangle": "upper",
        "matrix_dir": str(matrix_dir),
        "label_mode": args.label_mode,
    }

    frame_paths: List[Path] = []
    num_parameters = None
    matrices: List[np.ndarray] = []
    global_min = None
    global_max = None
    upper_idx = None

    for index, step in enumerate(steps):
        frame_start = time.perf_counter()
        matrix_name = f"matrix_{index:02d}_step_{step:09d}.npy"
        matrix_path = matrix_dir / matrix_name
        weight_paths = weight_paths_for_step(run_dirs, step)
        cache_hit = matrix_path.exists()
        if cache_hit:
            matrix_np = np.load(matrix_path)
        else:
            states = [load_state_dict(path) for path in weight_paths]
            keys = validate_models(states)
            if num_parameters is None:
                num_parameters = sum(states[0][key].numel() for key in keys)
            matrix, _, _ = compute_matrix(states, keys, args.chunk_elements)
            matrix_np = matrix_to_numpy(matrix)
            np.save(matrix_path, matrix_np)
            del states, matrix

        matrices.append(matrix_np)
        if upper_idx is None:
            upper_idx = np.triu_indices(matrix_np.shape[0], k=1)
        upper_values = matrix_np[upper_idx]
        frame_min = float(np.min(upper_values))
        frame_max = float(np.max(upper_values))
        global_min = frame_min if global_min is None else min(global_min, frame_min)
        global_max = frame_max if global_max is None else max(global_max, frame_max)

        elapsed = time.perf_counter() - frame_start
        manifest["frames"].append(
            {
                "index": index,
                "step": step,
                "matrix_path": str(matrix_path),
                "weight_paths": [str(path) for path in weight_paths],
                "compute_elapsed_sec": elapsed,
                "matrix_min": frame_min,
                "matrix_max": frame_max,
                "cache_hit": cache_hit,
            }
        )
        print(
            f"[compute {index + 1}/{len(steps)}] step={step} cache_hit={cache_hit} elapsed_sec={elapsed:.2f}",
            flush=True,
        )
        del matrix_np

    if global_min is None or global_max is None:
        raise ValueError("No matrices were computed")

    manifest["global_z_min"] = global_min
    manifest["global_z_max"] = global_max
    if args.azim_end is None:
        azim_values = [args.azim] * len(steps)
    else:
        azim_values = np.linspace(args.azim, args.azim_end, num=len(steps)).tolist()
    if args.elev_end is None:
        elev_values = [args.elev] * len(steps)
    else:
        elev_values = np.linspace(args.elev, args.elev_end, num=len(steps)).tolist()

    for index, (step, matrix_np, frame_elev, frame_azim) in enumerate(zip(steps, matrices, elev_values, azim_values)):
        render_start = time.perf_counter()
        frame_name = f"surface_{index:02d}_step_{step:09d}.png"
        frame_path = output_dir / frame_name
        title = f"01 Init Seed Only Unique-Pair L2 Surface, Step {step:,}"
        render_surface(
            labels=labels,
            matrix=matrix_np,
            output_path=frame_path,
            title=title,
            width_px=args.width_px,
            height_px=args.height_px,
            dpi=args.dpi,
            elev=frame_elev,
            azim=frame_azim,
            triangle="upper",
            z_min=global_min,
            z_max=global_max,
            color_min=global_min,
            color_max=global_max,
            label_mode=args.label_mode,
        )
        frame_paths.append(frame_path)
        manifest["frames"][index]["frame_path"] = str(frame_path)
        manifest["frames"][index]["render_elapsed_sec"] = time.perf_counter() - render_start
        manifest["frames"][index]["elev"] = frame_elev
        manifest["frames"][index]["azim"] = frame_azim
        print(
            f"[render {index + 1}/{len(steps)}] step={step} elev={frame_elev:.2f} azim={frame_azim:.2f} frame={frame_path.name}",
            flush=True,
        )

    build_gif(
        frame_paths=frame_paths,
        gif_path=gif_path,
        first_ms=args.first_frame_ms,
        middle_ms=args.middle_frame_ms,
        last_ms=args.last_frame_ms,
    )

    manifest["gif_path"] = str(gif_path)
    manifest["num_parameters_per_model"] = num_parameters
    manifest["total_elapsed_sec"] = time.perf_counter() - start_time
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"frames_dir: {output_dir}", flush=True)
    print(f"gif_path: {gif_path}", flush=True)
    print(f"manifest_path: {manifest_path}", flush=True)
    print(f"total_elapsed_sec: {manifest['total_elapsed_sec']:.2f}", flush=True)


if __name__ == "__main__":
    main()
