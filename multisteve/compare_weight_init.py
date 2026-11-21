# Compare the seeded implementation of weight init to the default one to check if the distributions match. They do.
import argparse
import math
from typing import Tuple

import numpy as np
import torch


def default_linear(in_features: int, out_features: int, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    lin = torch.nn.Linear(in_features, out_features, bias=True)
    return lin.weight.detach().cpu(), lin.bias.detach().cpu()


def seeded_linear(in_features: int, out_features: int, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator()
    generator.manual_seed(seed)
    weight = torch.empty(out_features, in_features)
    bias = torch.empty(out_features)
    bound = 1 / math.sqrt(in_features) if in_features > 0 else 0.0
    weight.uniform_(-bound, bound, generator=generator)
    bias.uniform_(-bound, bound, generator=generator)
    return weight, bias


def moments(tensor: torch.Tensor) -> dict:
    data = tensor.numpy().ravel()
    mean = float(np.mean(data))
    std = float(np.std(data))
    if std > 0:
        centered = data - mean
        skew = float(np.mean(centered ** 3) / (std ** 3))
        kurt = float(np.mean(centered ** 4) / (std ** 4) - 3.0)
    else:
        skew = 0.0
        kurt = 0.0
    return {
        "mean": mean,
        "std": std,
        "min": float(np.min(data)),
        "max": float(np.max(data)),
        "skew": skew,
        "kurtosis": kurt,
    }


def print_moments(label: str, tensor: torch.Tensor) -> None:
    stats = moments(tensor)
    print(f"{label}:")
    print(
        "  mean={mean:.6g} std={std:.6g} min={min:.6g} max={max:.6g} "
        "skew={skew:.6g} kurtosis={kurtosis:.6g}".format(**stats)
    )


def histogram(data: torch.Tensor, bins: int, data_min: float, data_max: float) -> Tuple[np.ndarray, np.ndarray]:
    arr = data.numpy().ravel()
    counts, edges = np.histogram(arr, bins=bins, range=(data_min, data_max))
    return counts, edges


def print_histogram(label: str, counts: np.ndarray, edges: np.ndarray, width: int = 40) -> None:
    print(f"{label} histogram:")
    max_count = int(counts.max()) if counts.size else 0
    for i, count in enumerate(counts):
        left = edges[i]
        right = edges[i + 1]
        bar_len = int((count / max_count) * width) if max_count > 0 else 0
        bar = "#" * bar_len
        print(f"  [{left: .4g}, {right: .4g}): {count:8d} {bar}")


def compare(name: str, a: torch.Tensor, b: torch.Tensor, bins: int) -> None:
    print(f"\n== {name} ==")
    print_moments("default", a)
    print_moments("seeded", b)
    data_min = float(min(a.min().item(), b.min().item()))
    data_max = float(max(a.max().item(), b.max().item()))
    counts_a, edges = histogram(a, bins, data_min, data_max)
    counts_b, _ = histogram(b, bins, data_min, data_max)
    print_histogram("default", counts_a, edges)
    print_histogram("seeded", counts_b, edges)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_features", type=int, default=512)
    parser.add_argument("--out_features", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--bins", type=int, default=30)
    args = parser.parse_args()

    w_default, b_default = default_linear(args.in_features, args.out_features, args.seed)
    w_seeded, b_seeded = seeded_linear(args.in_features, args.out_features, args.seed+1)

    compare("weights", w_default, w_seeded, args.bins)
    compare("bias", b_default, b_seeded, args.bins)


if __name__ == "__main__":
    main()
