import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


LEFT_OFFSET = np.array([4.0, 7.0], dtype=np.float64)
RIGHT_OFFSET = np.array([-4.0, 7.0], dtype=np.float64)
Z_95 = 1.959963984540054


@dataclass
class ModelResult:
    model_id: str
    run_dir: Path
    segments_total: int
    left: int
    right: int
    tie: int
    left_rate: float
    left_ci_low: float
    left_ci_high: float


def _infer_villager_side(manifest_path: Path) -> str | None:
    name = manifest_path.parent.name.lower()
    if "villager_left" in name:
        return "left"
    if "villager_right" in name:
        return "right"
    return None


def _wilson_ci(successes: int, n: int, z: float = Z_95) -> tuple[float, float]:
    if n <= 0:
        return float("nan"), float("nan")
    phat = successes / n
    denom = 1.0 + (z * z) / n
    center = (phat + (z * z) / (2.0 * n)) / denom
    half = (z / denom) * math.sqrt((phat * (1.0 - phat) / n) + (z * z) / (4.0 * n * n))
    return center - half, center + half


def _load_locations(path: Path) -> tuple[np.ndarray, np.ndarray]:
    xs = []
    zs = []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            xs.append(float(row["xpos"]))
            zs.append(float(row["zpos"]))
    if not xs:
        raise ValueError(f"No rows in {path}")
    return np.asarray(xs, dtype=np.float64), np.asarray(zs, dtype=np.float64)


def _load_break_indices(path: Path, n_points: int) -> list[int]:
    teleport_steps = []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            teleport_steps.append(int(row["step"]))
    return sorted({s + 1 for s in teleport_steps if 0 <= s + 1 < n_points})


def _build_segments(x: np.ndarray, z: np.ndarray, break_indices: Iterable[int]) -> list[tuple[np.ndarray, np.ndarray]]:
    segments: list[tuple[np.ndarray, np.ndarray]] = []
    start = 0
    for b in break_indices:
        sx = x[start:b]
        sz = z[start:b]
        if sx.size >= 2:
            segments.append((sx, sz))
        start = b
    sx = x[start:]
    sz = z[start:]
    if sx.size >= 2:
        segments.append((sx, sz))
    return segments


def _segment_endpoint_indices(n_points: int, break_indices: Iterable[int]) -> list[int]:
    end_indices = []
    for b in break_indices:
        end_indices.append(b - 1)
    end_indices.append(n_points - 1)
    return end_indices


def _classify_endpoints(x: np.ndarray, z: np.ndarray, endpoint_indices: Iterable[int]) -> tuple[int, int, int, list[dict]]:
    spawn = np.array([x[0], z[0]], dtype=np.float64)
    left_center = spawn + LEFT_OFFSET
    right_center = spawn + RIGHT_OFFSET

    left = 0
    right = 0
    tie = 0
    rows = []
    for seg_idx, idx in enumerate(endpoint_indices):
        p = np.array([x[idx], z[idx]], dtype=np.float64)
        dl = float(np.linalg.norm(p - left_center))
        dr = float(np.linalg.norm(p - right_center))
        if dl < dr:
            side = "left"
            left += 1
        elif dr < dl:
            side = "right"
            right += 1
        else:
            side = "tie"
            tie += 1
        rows.append(
            {
                "segment_index": int(seg_idx),
                "end_row_index": int(idx),
                "end_x": float(x[idx]),
                "end_z": float(z[idx]),
                "distance_left_tree": dl,
                "distance_right_tree": dr,
                "classified_side": side,
            }
        )
    return left, right, tie, rows


def _plot_trajectory(
    run_dir: Path,
    model_id: str,
    x: np.ndarray,
    z: np.ndarray,
    segments: list[tuple[np.ndarray, np.ndarray]],
    villager_side: str,
) -> Path:
    spawn_x = float(x[0])
    spawn_z = float(z[0])
    left_center = np.array([spawn_x, spawn_z]) + LEFT_OFFSET
    right_center = np.array([spawn_x, spawn_z]) + RIGHT_OFFSET

    if villager_side == "left":
        left_label = "left villager tree"
        left_color = "tab:green"
        right_label = "right tree"
        right_color = "tab:orange"
    elif villager_side == "right":
        left_label = "left tree"
        left_color = "tab:orange"
        right_label = "right villager tree"
        right_color = "tab:green"
    else:
        raise ValueError(f"Unsupported villager_side: {villager_side}")

    plt.figure(figsize=(8, 8))
    for i, (sx, sz) in enumerate(segments):
        plt.plot(sx, sz, linewidth=1.25, alpha=0.95, label="trajectory" if i == 0 else None)
        plt.scatter([sx[0]], [sz[0]], s=8)

    plt.scatter([spawn_x], [spawn_z], marker="*", s=120, c="black", label="spawn")
    plt.scatter([left_center[0]], [left_center[1]], marker="^", s=70, c=left_color, label=left_label)
    plt.scatter([right_center[0]], [right_center[1]], marker="^", s=70, c=right_color, label=right_label)

    ax = plt.gca()
    ax.invert_xaxis()
    plt.title(f"Top-Down Trajectory (Teleport Jumps Removed): {model_id}")
    plt.xlabel("x")
    plt.ylabel("z")
    plt.axis("equal")
    plt.grid(True, alpha=0.25)
    plt.legend(loc="best")
    plt.tight_layout()
    out_path = run_dir / "trajectory_topdown_reset.png"
    plt.savefig(out_path, dpi=180)
    plt.close()
    return out_path


def _plot_comparison(out_dir: Path, results: list[ModelResult]) -> Path:
    model_ids = [r.model_id for r in results]
    pcts = [r.left_rate * 100.0 for r in results]
    lows = [(r.left_rate - r.left_ci_low) * 100.0 for r in results]
    highs = [(r.left_ci_high - r.left_rate) * 100.0 for r in results]

    x = np.arange(len(results))
    plt.figure(figsize=(12, 6))
    plt.bar(x, pcts, color="tab:blue", alpha=0.85)
    plt.errorbar(x, pcts, yerr=[lows, highs], fmt="none", ecolor="black", capsize=3, linewidth=1.0)
    plt.xticks(x, model_ids, rotation=35, ha="right")
    plt.ylabel("Left Choice Rate (%)")
    plt.title("Left-vs-Right Endpoint Choice by Model (95% Wilson CI)")
    plt.ylim(0, 100)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out_path = out_dir / "left_choice_rate_ci.png"
    plt.savefig(out_path, dpi=180)
    plt.close()
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, help="CSV manifest emitted by rollout script")
    parser.add_argument("--out_dir", default=None, help="Output directory for aggregate summaries")
    parser.add_argument("--villager_side", choices=["left", "right"], default=None)
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    out_dir = Path(args.out_dir) if args.out_dir else manifest_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    villager_side = args.villager_side or _infer_villager_side(manifest_path) or "left"

    results: list[ModelResult] = []
    skipped = []
    with manifest_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            model_id = row["model_id"]
            status = row["status"]
            run_dir_raw = row["run_dir"].strip()
            if status != "ok" or not run_dir_raw:
                skipped.append({"model_id": model_id, "status": status, "run_dir": run_dir_raw})
                continue

            run_dir = Path(run_dir_raw)
            loc_path = run_dir / "location_stats.log"
            tp_path = run_dir / "teleport_events.log"
            if not loc_path.exists() or not tp_path.exists():
                skipped.append({"model_id": model_id, "status": "missing_logs", "run_dir": str(run_dir)})
                continue

            x, z = _load_locations(loc_path)
            breaks = _load_break_indices(tp_path, len(x))
            segments = _build_segments(x, z, breaks)
            endpoint_indices = _segment_endpoint_indices(len(x), breaks)
            left, right, tie, endpoint_rows = _classify_endpoints(x, z, endpoint_indices)

            non_tie = left + right
            left_rate = (left / non_tie) if non_tie > 0 else float("nan")
            ci_low, ci_high = _wilson_ci(left, non_tie)

            _plot_trajectory(run_dir, model_id, x, z, segments, villager_side)

            per_model_json = {
                "model_id": model_id,
                "run_dir": str(run_dir),
                "segments_total": len(endpoint_indices),
                "left_segments": left,
                "right_segments": right,
                "tie_segments": tie,
                "left_rate_excluding_ties": left_rate,
                "left_rate_ci95_wilson_excluding_ties": [ci_low, ci_high],
                "villager_side": villager_side,
                "method": "Endpoint of each short segment (between teleports) is assigned to nearest tree center in x-z plane.",
                "segment_endpoint_rows": endpoint_rows,
            }
            (run_dir / "trajectory_endpoint_split.json").write_text(json.dumps(per_model_json, indent=2))

            results.append(
                ModelResult(
                    model_id=model_id,
                    run_dir=run_dir,
                    segments_total=len(endpoint_indices),
                    left=left,
                    right=right,
                    tie=tie,
                    left_rate=left_rate,
                    left_ci_low=ci_low,
                    left_ci_high=ci_high,
                )
            )

    results.sort(key=lambda r: r.model_id)

    compare_csv = out_dir / "model_comparison.csv"
    with compare_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "model_id",
                "run_dir",
                "segments_total",
                "left_segments",
                "right_segments",
                "tie_segments",
                "left_rate_pct_excluding_ties",
                "left_ci95_low_pct",
                "left_ci95_high_pct",
            ]
        )
        for r in results:
            writer.writerow(
                [
                    r.model_id,
                    str(r.run_dir),
                    r.segments_total,
                    r.left,
                    r.right,
                    r.tie,
                    r.left_rate * 100.0,
                    r.left_ci_low * 100.0,
                    r.left_ci_high * 100.0,
                ]
            )

    compare_md = out_dir / "model_comparison.md"
    with compare_md.open("w") as f:
        f.write("# Model Comparison: Left vs Right Endpoint Choice\n\n")
        f.write("Method: endpoint of each short segment (teleport-to-teleport) classified by nearest tree center in x-z plane.\n\n")
        f.write("| model_id | segments | left | right | tie | left% (excl ties) | 95% CI |\n")
        f.write("|---|---:|---:|---:|---:|---:|---|\n")
        for r in results:
            f.write(
                f"| {r.model_id} | {r.segments_total} | {r.left} | {r.right} | {r.tie} | "
                f"{r.left_rate * 100.0:.2f}% | [{r.left_ci_low * 100.0:.2f}%, {r.left_ci_high * 100.0:.2f}%] |\n"
            )

    aggregate = {
        "manifest": str(manifest_path),
        "num_models_ok": len(results),
        "num_models_skipped": len(skipped),
        "skipped": skipped,
        "comparison_csv": str(compare_csv),
        "comparison_md": str(compare_md),
    }
    if results:
        plot_path = _plot_comparison(out_dir, results)
        aggregate["comparison_plot"] = str(plot_path)
    (out_dir / "model_comparison_summary.json").write_text(json.dumps(aggregate, indent=2))

    print(f"wrote: {compare_csv}")
    print(f"wrote: {compare_md}")
    if results:
        print(f"wrote: {out_dir / 'left_choice_rate_ci.png'}")
    print(f"wrote: {out_dir / 'model_comparison_summary.json'}")
    print(f"models analyzed: {len(results)}, skipped: {len(skipped)}")


if __name__ == "__main__":
    main()
