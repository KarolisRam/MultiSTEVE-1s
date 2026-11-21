import argparse
import json
import re
import time
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
from PIL import Image

from measure_model_l2 import list_run_dirs
from measure_model_l2_matrix import compute_matrix, load_state_dict, validate_models
from plot_l2_heatmap import render_heatmap


SNAPSHOT_RE = re.compile(r"^STEVE-1_snapshot_(\d+)\.weights$")
SEED_RE = re.compile(r"weight_init_seed(\d+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate checkpoint-by-checkpoint L2 heatmaps and assemble them into a GIF."
    )
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--chunk-elements", type=int, default=1_000_000)
    parser.add_argument("--width-px", type=int, default=864)
    parser.add_argument("--height-px", type=int, default=864)
    parser.add_argument("--dpi", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gif-path", type=Path, required=True)
    parser.add_argument("--first-frame-ms", type=int, default=1000)
    parser.add_argument("--middle-frame-ms", type=int, default=200)
    parser.add_argument("--last-frame-ms", type=int, default=40000)
    parser.add_argument("--max-steps", type=int)
    return parser.parse_args()


def label_for_run_dir(run_dir: Path) -> str:
    match = SEED_RE.search(run_dir.name)
    if match:
        return f"seed{int(match.group(1))}"
    return run_dir.name


def select_run_dirs(experiment_dir: Path, limit: int) -> List[Path]:
    run_dirs = list_run_dirs(experiment_dir.expanduser().resolve())
    if len(run_dirs) < limit:
        raise ValueError(f"Requested {limit} runs but only found {len(run_dirs)} in {experiment_dir}")
    return run_dirs[:limit]


def snapshot_steps(run_dir: Path) -> List[int]:
    steps = []
    for child in run_dir.iterdir():
        match = SNAPSHOT_RE.match(child.name)
        if match:
            steps.append(int(match.group(1)))
    return sorted(steps)


def common_snapshot_steps(run_dirs: Sequence[Path]) -> List[int]:
    common = None
    for run_dir in run_dirs:
        steps = set(snapshot_steps(run_dir))
        common = steps if common is None else common & steps
    if not common:
        raise ValueError("No common snapshot steps found across selected runs")
    return sorted(common)


def weight_paths_for_step(run_dirs: Sequence[Path], step: int) -> List[Path]:
    filename = f"STEVE-1_snapshot_{step:09d}.weights"
    paths = [run_dir / filename for run_dir in run_dirs]
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing checkpoint for step {step}: {missing[0]}")
    return paths


def matrix_to_numpy(matrix) -> np.ndarray:
    return np.asarray(matrix.tolist(), dtype=float)


def build_gif(frame_paths: Sequence[Path], gif_path: Path, first_ms: int, middle_ms: int, last_ms: int) -> None:
    if not frame_paths:
        raise ValueError("No frames provided for GIF generation")

    frames = [Image.open(path).convert("P", palette=Image.ADAPTIVE) for path in frame_paths]
    durations = [middle_ms] * len(frames)
    durations[0] = first_ms
    durations[-1] = last_ms

    gif_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=False,
        disposal=2,
    )


def main() -> None:
    args = parse_args()
    start_time = time.perf_counter()

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    gif_path = args.gif_path.expanduser().resolve()

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
    }

    frame_paths: List[Path] = []
    last_keys = None
    num_parameters = None

    for index, step in enumerate(steps):
        frame_start = time.perf_counter()
        weight_paths = weight_paths_for_step(run_dirs, step)
        states = [load_state_dict(path) for path in weight_paths]
        keys = validate_models(states)
        if last_keys is None:
            last_keys = keys
            num_parameters = sum(states[0][key].numel() for key in keys)
        matrix, _, _ = compute_matrix(states, keys, args.chunk_elements)
        matrix_np = matrix_to_numpy(matrix)

        frame_name = f"frame_{index:02d}_step_{step:09d}.png"
        frame_path = output_dir / frame_name
        title = f"01 Init Seed Only L2 Distances, Step {step:,}"
        render_heatmap(
            labels=labels,
            matrix=matrix_np,
            output_path=frame_path,
            title=title,
            width_px=args.width_px,
            height_px=args.height_px,
            dpi=args.dpi,
        )

        frame_paths.append(frame_path)
        elapsed = time.perf_counter() - frame_start
        manifest["frames"].append(
            {
                "index": index,
                "step": step,
                "frame_path": str(frame_path),
                "weight_paths": [str(path) for path in weight_paths],
                "elapsed_sec": elapsed,
            }
        )
        print(f"[{index + 1}/{len(steps)}] step={step} frame={frame_path.name} elapsed_sec={elapsed:.2f}", flush=True)
        del states, matrix, matrix_np

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
