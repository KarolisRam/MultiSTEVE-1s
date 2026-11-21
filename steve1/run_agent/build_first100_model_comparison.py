from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

Z_95 = 1.959963984540054
FIELDS = [
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


def _wilson_ci(successes: int, n: int, z: float = Z_95) -> tuple[float, float]:
    if n <= 0:
        return float("nan"), float("nan")
    phat = successes / n
    denom = 1.0 + (z * z) / n
    center = (phat + (z * z) / (2.0 * n)) / denom
    half = (z / denom) * math.sqrt((phat * (1.0 - phat) / n) + (z * z) / (4.0 * n * n))
    return center - half, center + half


def _first_n_counts(run_dir: Path, max_segments: int) -> tuple[int, int, int, int]:
    json_path = run_dir / "trajectory_endpoint_split.json"
    data = json.loads(json_path.read_text())
    rows = data["segment_endpoint_rows"][:max_segments]
    left = sum(1 for row in rows if row.get("classified_side") == "left")
    right = sum(1 for row in rows if row.get("classified_side") == "right")
    tie = sum(1 for row in rows if row.get("classified_side") == "tie")
    return len(rows), left, right, tie


def build(input_csv: Path, out_csv: Path, max_segments: int) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with input_csv.open("r", newline="") as f:
        rows = list(csv.DictReader(f))

    out_rows = []
    for row in rows:
        run_dir = Path(row["run_dir"])
        total, left, right, tie = _first_n_counts(run_dir, max_segments)
        non_tie = left + right
        left_rate = left / non_tie if non_tie else float("nan")
        ci_low, ci_high = _wilson_ci(left, non_tie)
        out_rows.append(
            {
                "model_id": row["model_id"],
                "run_dir": str(run_dir),
                "segments_total": total,
                "left_segments": left,
                "right_segments": right,
                "tie_segments": tie,
                "left_rate_pct_excluding_ties": left_rate * 100.0,
                "left_ci95_low_pct": ci_low * 100.0,
                "left_ci95_high_pct": ci_high * 100.0,
            }
        )

    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"wrote: {out_csv}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--max_segments", type=int, default=100)
    args = parser.parse_args()
    build(Path(args.input_csv), Path(args.out_csv), args.max_segments)


if __name__ == "__main__":
    main()
