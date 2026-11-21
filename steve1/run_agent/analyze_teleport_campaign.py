from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from steve1.run_agent.teleport_campaign_common import CONDITIONS


def _run(repo_root: Path, args: list[str]) -> None:
    subprocess.run([sys.executable, "-m", *args], cwd=repo_root, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign_root", required=True)
    parser.add_argument("--repo_root", default=".")
    parser.add_argument("--first_segments", type=int, default=100)
    parser.add_argument("--max_agent_seed", type=int, default=None)
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    campaign_root = Path(args.campaign_root)
    if not campaign_root.is_absolute():
        campaign_root = repo_root / campaign_root
    campaign_root = campaign_root.resolve()
    analysis_dir = campaign_root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    first_n_csvs: dict[str, Path] = {}
    for condition in CONDITIONS:
        condition_dir = campaign_root / "conditions" / condition.key
        manifest = condition_dir / "manifest.csv"
        if not manifest.exists():
            print(f"skip missing manifest: {manifest}")
            continue
        _run(
            repo_root,
            [
                "steve1.run_agent.analyze_teleport_model_compare",
                "--manifest",
                str(manifest),
                "--out_dir",
                str(condition_dir),
                "--villager_side",
                condition.villager_side,
            ],
        )
        full_csv = condition_dir / "model_comparison.csv"
        first_csv = condition_dir / f"model_comparison_first{args.first_segments}.csv"
        _run(
            repo_root,
            [
                "steve1.run_agent.build_first100_model_comparison",
                "--input_csv",
                str(full_csv),
                "--out_csv",
                str(first_csv),
                "--max_segments",
                str(args.first_segments),
            ],
        )
        first_n_csvs[condition.key] = first_csv

    for seed, left_key, right_key in [
        (49, "seed049_villager_left", "seed049_villager_right"),
        (481, "seed481_villager_left", "seed481_villager_right"),
    ]:
        if left_key not in first_n_csvs or right_key not in first_n_csvs:
            print(f"skip seed {seed} scatter; missing CSV")
            continue
        out_dir = analysis_dir / f"seed{seed}_layout_scatter_first{args.first_segments}"
        scatter_args = [
            "steve1.run_agent.analyze_teleport_layout_scatter",
            "--left_first_csv",
            str(first_n_csvs[left_key]),
            "--right_first_csv",
            str(first_n_csvs[right_key]),
            "--out_dir",
            str(out_dir),
            "--page_title",
            f"Seed {seed} T-Maze Choice Scatter",
            "--chart_title",
            f"Seed {seed} T-Maze Choice Scatter",
        ]
        if args.max_agent_seed is not None:
            scatter_args.extend(["--max_agent_seed", str(args.max_agent_seed)])
        _run(repo_root, scatter_args)

    print(f"analysis_dir={analysis_dir}")


if __name__ == "__main__":
    main()
