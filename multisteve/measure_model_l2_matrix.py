import argparse
import csv
import json
import re
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import psutil
import torch

from measure_model_l2 import format_bytes, list_run_dirs, resolve_weight_path


SEED_RE = re.compile(r"weight_init_seed(\d+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute an all-pairs L2 distance matrix for the latest STEVE-1 checkpoints in an experiment folder."
    )
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--chunk-elements", type=int, default=1_000_000)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def label_for_run_dir(run_dir: Path) -> str:
    match = SEED_RE.search(run_dir.name)
    if match:
        return f"seed{int(match.group(1))}"
    return run_dir.name


def select_models(experiment_dir: Path, limit: int) -> Tuple[List[str], List[Path]]:
    run_dirs = list_run_dirs(experiment_dir.expanduser().resolve())
    if len(run_dirs) < limit:
        raise ValueError(f"Requested {limit} runs but only found {len(run_dirs)} in {experiment_dir}")
    chosen = run_dirs[:limit]
    labels = [label_for_run_dir(run_dir) for run_dir in chosen]
    weight_paths = [resolve_weight_path(run_dir) for run_dir in chosen]
    return labels, weight_paths


def load_state_dict(path: Path) -> Dict[str, torch.Tensor]:
    state = torch.load(path, map_location="cpu")
    if not isinstance(state, dict):
        raise TypeError(f"Expected a state dict in {path}, got {type(state).__name__}")
    return state


def tensor_keys(state: Dict[str, torch.Tensor]) -> List[str]:
    return [key for key, value in state.items() if torch.is_tensor(value)]


def validate_models(states: Sequence[Dict[str, torch.Tensor]]) -> List[str]:
    if not states:
        raise ValueError("No models loaded")

    reference_keys = sorted(tensor_keys(states[0]))
    reference_shapes = {key: tuple(states[0][key].shape) for key in reference_keys}

    for model_index, state in enumerate(states[1:], start=1):
        keys = sorted(tensor_keys(state))
        if keys != reference_keys:
            missing = sorted(set(reference_keys) - set(keys))
            extra = sorted(set(keys) - set(reference_keys))
            raise ValueError(
                f"Model {model_index} keys do not match reference: "
                f"{len(missing)} missing, {len(extra)} extra"
            )
        for key in reference_keys:
            if tuple(state[key].shape) != reference_shapes[key]:
                raise ValueError(
                    f"Shape mismatch for model {model_index}, key {key}: "
                    f"{tuple(state[key].shape)} vs {reference_shapes[key]}"
                )
            if not state[key].is_floating_point():
                raise TypeError(f"Expected floating tensor for key {key}, got {state[key].dtype}")

    return reference_keys


def compute_matrix(
    states: Sequence[Dict[str, torch.Tensor]],
    keys: Sequence[str],
    chunk_elements: int,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    n_models = len(states)
    gram = torch.zeros((n_models, n_models), dtype=torch.float64)
    sq_norm = torch.zeros(n_models, dtype=torch.float64)
    num_parameters = 0

    for key in keys:
        flats = [state[key].reshape(-1) for state in states]
        tensor_numel = flats[0].numel()
        num_parameters += tensor_numel

        for start in range(0, tensor_numel, chunk_elements):
            end = min(start + chunk_elements, tensor_numel)
            block = torch.stack([flat[start:end] for flat in flats], dim=0).to(dtype=torch.float64)
            sq_norm += (block * block).sum(dim=1)
            gram += block @ block.T

    dist_sq = sq_norm[:, None] + sq_norm[None, :] - 2.0 * gram
    dist_sq.clamp_(min=0.0)
    matrix = torch.sqrt(dist_sq)
    return matrix, sq_norm, num_parameters


def write_csv(path: Path, labels: Sequence[str], matrix: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", *labels])
        for label, row in zip(labels, matrix.tolist()):
            writer.writerow([label, *[f"{value:.15f}" for value in row]])


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def format_matrix(labels: Sequence[str], matrix: torch.Tensor) -> str:
    cell_width = 12
    header = ["".ljust(cell_width), *[label.rjust(cell_width) for label in labels]]
    lines = [" ".join(header)]
    rows = matrix.tolist()
    for label, row in zip(labels, rows):
        values = [f"{value:.3f}".rjust(cell_width) for value in row]
        lines.append(" ".join([label.ljust(cell_width), *values]))
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    process = psutil.Process()

    select_start = time.perf_counter()
    labels, weight_paths = select_models(args.experiment_dir, args.limit)
    select_sec = time.perf_counter() - select_start

    load_start = time.perf_counter()
    states = [load_state_dict(path) for path in weight_paths]
    load_sec = time.perf_counter() - load_start

    validate_start = time.perf_counter()
    keys = validate_models(states)
    validate_sec = time.perf_counter() - validate_start

    compute_start = time.perf_counter()
    matrix, sq_norm, num_parameters = compute_matrix(states, keys, args.chunk_elements)
    compute_sec = time.perf_counter() - compute_start

    peak_rss_bytes = process.memory_info().rss
    bytes_per_model = sum(path.stat().st_size for path in weight_paths) / len(weight_paths)

    payload: Dict[str, object] = {
        "experiment_dir": str(args.experiment_dir.expanduser().resolve()),
        "labels": labels,
        "weight_paths": [str(path) for path in weight_paths],
        "num_models": len(labels),
        "num_tensors": len(keys),
        "num_parameters_per_model": num_parameters,
        "avg_checkpoint_file_bytes": bytes_per_model,
        "timings_sec": {
            "select_models_sec": select_sec,
            "load_models_sec": load_sec,
            "validate_models_sec": validate_sec,
            "compute_matrix_sec": compute_sec,
            "total_sec": select_sec + load_sec + validate_sec + compute_sec,
        },
        "peak_rss_bytes": peak_rss_bytes,
        "matrix": matrix.tolist(),
        "sq_norm": sq_norm.tolist(),
    }

    if args.output_csv:
        write_csv(args.output_csv.expanduser().resolve(), labels, matrix)
    if args.output_json:
        write_json(args.output_json.expanduser().resolve(), payload)

    print(f"experiment_dir: {payload['experiment_dir']}")
    print(f"labels: {', '.join(labels)}")
    print(f"num_models: {payload['num_models']}")
    print(f"num_tensors: {payload['num_tensors']}")
    print(f"num_parameters_per_model: {payload['num_parameters_per_model']}")
    print(
        "avg_checkpoint_file_bytes: "
        f"{int(bytes_per_model)} ({format_bytes(bytes_per_model)})"
    )
    for key, value in payload["timings_sec"].items():
        print(f"{key}: {value:.6f}")
    print(f"peak_rss_bytes: {peak_rss_bytes} ({format_bytes(peak_rss_bytes)})")
    if args.output_csv:
        print(f"output_csv: {args.output_csv.expanduser().resolve()}")
    if args.output_json:
        print(f"output_json: {args.output_json.expanduser().resolve()}")
    print("l2_matrix:")
    print(format_matrix(labels, matrix))


if __name__ == "__main__":
    main()
