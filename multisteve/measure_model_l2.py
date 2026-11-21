import argparse
import gc
import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import psutil
import torch


SNAPSHOT_RE = re.compile(r"^STEVE-1_snapshot_(\d+)\.weights$")


def natural_key(text: str) -> List[object]:
    parts = re.split(r"(\d+)", text)
    key: List[object] = []
    for part in parts:
        if part.isdigit():
            key.append(int(part))
        else:
            key.append(part)
    return key


def format_bytes(num_bytes: float) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} TiB"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute the L2 distance between two STEVE-1 weight files with timing and memory stats."
    )
    parser.add_argument("--model-a", type=Path, help="Weight file or run directory for model A.")
    parser.add_argument("--model-b", type=Path, help="Weight file or run directory for model B.")
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        help="If set, automatically pick the first two run directories in natural sort order.",
    )
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--sample-interval-sec", type=float, default=0.05)
    parser.add_argument("--json", action="store_true", help="Emit results as JSON.")
    return parser.parse_args()


def list_run_dirs(experiment_dir: Path) -> List[Path]:
    run_dirs = [child for child in experiment_dir.iterdir() if child.is_dir()]
    return sorted(run_dirs, key=lambda path: natural_key(path.name))


def find_latest_snapshot(run_dir: Path) -> Path:
    snapshots: List[Tuple[int, Path]] = []
    for child in run_dir.iterdir():
        match = SNAPSHOT_RE.match(child.name)
        if match:
            snapshots.append((int(match.group(1)), child))
    if not snapshots:
        raise FileNotFoundError(f"No STEVE-1 snapshots found in {run_dir}")
    snapshots.sort(key=lambda item: item[0])
    return snapshots[-1][1]


def resolve_weight_path(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.is_file():
        return path
    if path.is_dir():
        return find_latest_snapshot(path)
    raise FileNotFoundError(f"Path does not exist: {path}")


def choose_model_pair(args: argparse.Namespace) -> Tuple[Path, Path]:
    if args.experiment_dir is not None:
        run_dirs = list_run_dirs(args.experiment_dir.expanduser().resolve())
        if len(run_dirs) < 2:
            raise ValueError(f"Need at least two run directories in {args.experiment_dir}")
        return resolve_weight_path(run_dirs[0]), resolve_weight_path(run_dirs[1])
    if args.model_a is None or args.model_b is None:
        raise ValueError("Provide either --experiment-dir or both --model-a and --model-b")
    return resolve_weight_path(args.model_a), resolve_weight_path(args.model_b)


class MemoryMonitor:
    def __init__(self, device: torch.device, sample_interval_sec: float) -> None:
        self._device = device
        self._sample_interval_sec = sample_interval_sec
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._process = psutil.Process(os.getpid())
        self.peak_rss_bytes = 0
        self.peak_vram_allocated_bytes = 0
        self.peak_vram_reserved_bytes = 0

    def start(self) -> None:
        self._sample()
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._thread.join()
        self._sample()

    def _run(self) -> None:
        while not self._stop_event.wait(self._sample_interval_sec):
            self._sample()

    def _sample(self) -> None:
        rss = self._process.memory_info().rss
        if rss > self.peak_rss_bytes:
            self.peak_rss_bytes = rss
        if self._device.type == "cuda" and torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(self._device)
            reserved = torch.cuda.memory_reserved(self._device)
            if allocated > self.peak_vram_allocated_bytes:
                self.peak_vram_allocated_bytes = allocated
            if reserved > self.peak_vram_reserved_bytes:
                self.peak_vram_reserved_bytes = reserved


def load_state_dict(path: Path, device: torch.device) -> Dict[str, torch.Tensor]:
    del device
    state = torch.load(path, map_location="cpu")
    if not isinstance(state, dict):
        raise TypeError(f"Expected a state dict in {path}, got {type(state).__name__}")
    return state


def tensor_keys(state: Dict[str, torch.Tensor]) -> List[str]:
    return [key for key, value in state.items() if torch.is_tensor(value)]


def move_state_dict_to_device(state: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    if device.type == "cpu":
        return state

    moved: Dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device=device, non_blocking=False)
        else:
            moved[key] = value
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return moved


def compare_state_dicts(
    state_a: Dict[str, torch.Tensor],
    state_b: Dict[str, torch.Tensor],
    device: torch.device,
) -> Dict[str, object]:
    keys_a = tensor_keys(state_a)
    keys_b = tensor_keys(state_b)
    missing = sorted(set(keys_a) - set(keys_b))
    extra = sorted(set(keys_b) - set(keys_a))
    if missing or extra:
        raise ValueError(
            "State dict keys do not match: "
            f"{len(missing)} missing in model B, {len(extra)} extra in model B"
        )

    keys = sorted(keys_a)
    sum_sq = torch.zeros((), dtype=torch.float64, device=device)
    num_tensors = 0
    num_parameters = 0
    bytes_per_model = 0

    compute_start = time.perf_counter()
    for key in keys:
        tensor_a = state_a[key]
        tensor_b = state_b[key]
        if tensor_a.shape != tensor_b.shape:
            raise ValueError(f"Shape mismatch for {key}: {tuple(tensor_a.shape)} vs {tuple(tensor_b.shape)}")
        if tensor_a.device != device or tensor_b.device != device:
            raise ValueError(f"Tensors for {key} are not on the requested device {device}")
        if not tensor_a.is_floating_point() or not tensor_b.is_floating_point():
            raise TypeError(f"Expected floating tensors for {key}, got {tensor_a.dtype} and {tensor_b.dtype}")
        sum_sq += (tensor_a - tensor_b).pow(2).sum(dtype=torch.float64)
        num_tensors += 1
        num_parameters += tensor_a.numel()
        bytes_per_model += tensor_a.numel() * tensor_a.element_size()

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    compute_sec = time.perf_counter() - compute_start

    return {
        "l2_distance": torch.sqrt(sum_sq).item(),
        "num_tensors": num_tensors,
        "num_parameters": num_parameters,
        "bytes_per_model": bytes_per_model,
        "compute_sec": compute_sec,
    }


def main() -> None:
    args = parse_args()
    model_a_path, model_b_path = choose_model_pair(args)
    device = torch.device("cuda:0" if args.device == "cuda" else "cpu")

    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
        torch.cuda.init()
        torch.cuda.set_device(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    timings: Dict[str, float] = {}
    file_sizes = {
        "model_a_file_bytes": model_a_path.stat().st_size,
        "model_b_file_bytes": model_b_path.stat().st_size,
    }

    total_start = time.perf_counter()
    monitor = MemoryMonitor(device=device, sample_interval_sec=args.sample_interval_sec)
    monitor.start()

    load_a_start = time.perf_counter()
    state_a = load_state_dict(model_a_path, device=device)
    timings["load_model_a_sec"] = time.perf_counter() - load_a_start
    if device.type == "cuda":
        move_a_start = time.perf_counter()
        state_a = move_state_dict_to_device(state_a, device=device)
        gc.collect()
        torch.cuda.synchronize(device)
        timings["move_model_a_to_cuda_sec"] = time.perf_counter() - move_a_start

    load_b_start = time.perf_counter()
    state_b = load_state_dict(model_b_path, device=device)
    timings["load_model_b_sec"] = time.perf_counter() - load_b_start
    if device.type == "cuda":
        move_b_start = time.perf_counter()
        state_b = move_state_dict_to_device(state_b, device=device)
        gc.collect()
        torch.cuda.synchronize(device)
        timings["move_model_b_to_cuda_sec"] = time.perf_counter() - move_b_start

    compare_result = compare_state_dicts(state_a=state_a, state_b=state_b, device=device)
    timings["compute_l2_sec"] = compare_result.pop("compute_sec")
    timings["total_sec"] = time.perf_counter() - total_start

    monitor.stop()

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    result = {
        "device": str(device),
        "model_a_path": str(model_a_path),
        "model_b_path": str(model_b_path),
        **file_sizes,
        **compare_result,
        "timings_sec": timings,
        "peak_rss_bytes": monitor.peak_rss_bytes,
        "peak_vram_allocated_bytes": monitor.peak_vram_allocated_bytes,
        "peak_vram_reserved_bytes": monitor.peak_vram_reserved_bytes,
    }

    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    print(f"device: {result['device']}")
    print(f"model_a_path: {result['model_a_path']}")
    print(f"model_b_path: {result['model_b_path']}")
    print(f"model_a_file_bytes: {result['model_a_file_bytes']} ({format_bytes(result['model_a_file_bytes'])})")
    print(f"model_b_file_bytes: {result['model_b_file_bytes']} ({format_bytes(result['model_b_file_bytes'])})")
    print(f"num_tensors: {result['num_tensors']}")
    print(f"num_parameters: {result['num_parameters']}")
    print(f"bytes_per_model: {result['bytes_per_model']} ({format_bytes(result['bytes_per_model'])})")
    print(f"l2_distance: {result['l2_distance']:.15f}")
    for key, value in result["timings_sec"].items():
        print(f"{key}: {value:.6f}")
    print(f"peak_rss_bytes: {result['peak_rss_bytes']} ({format_bytes(result['peak_rss_bytes'])})")
    print(
        "peak_vram_allocated_bytes: "
        f"{result['peak_vram_allocated_bytes']} ({format_bytes(result['peak_vram_allocated_bytes'])})"
    )
    print(
        "peak_vram_reserved_bytes: "
        f"{result['peak_vram_reserved_bytes']} ({format_bytes(result['peak_vram_reserved_bytes'])})"
    )


if __name__ == "__main__":
    main()
