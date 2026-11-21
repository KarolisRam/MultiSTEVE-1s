from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config
from .suite import PlotSuite


FAMILIES = [
    "full_eval",
    "partial_eval_40",
    "partial_eval_28_checkpoints",
    "tree_prompt_100",
    "villager_passes",
    "tree_vs_villager",
    "villager_chop_40",
    "villager_bitflip",
    "agent_111_114_seed_sweep",
    "head_fake_stability",
    "teleport_layout",
    "agent_trajectories",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect cached data and render paper-ready plot PDFs.")
    parser.add_argument("command", nargs="?", default="run", choices=["run", "list"])
    parser.add_argument("--config", type=Path, default=Path("configs/plot_suite/final.json"))
    parser.add_argument("--from-cache", action="store_true", help="Skip raw data parsing when cache CSV/NPZ files exist.")
    parser.add_argument("--run-name", type=str, default=None, help="Optional output run name.")
    parser.add_argument("--only", action="append", choices=FAMILIES, help="Run only one family. Can be repeated.")
    parser.add_argument(
        "--include",
        action="append",
        help="Glob pattern for artifacts to copy into overleaf folder. Can be repeated.",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        help="Glob pattern for artifacts to skip from overleaf folder. Can be repeated.",
    )
    parser.add_argument("--no-overleaf", action="store_true", help="Do not build the PDF-only Overleaf bundle.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config if args.config.exists() else None)
    if args.include:
        config.setdefault("overleaf", {})["include"] = args.include
    if args.exclude:
        config.setdefault("overleaf", {})["exclude"] = args.exclude

    if args.command == "list":
        for family in FAMILIES:
            enabled = config["families"].get(family, {}).get("enabled", False)
            print(f"{family}: {'enabled' if enabled else 'disabled'}")
        print("Overleaf include patterns:")
        for pattern in config.get("overleaf", {}).get("include", ["**"]):
            print(f"  {pattern}")
        return 0

    suite = PlotSuite(config)
    ctx = suite.run(
        run_name=args.run_name,
        from_cache=args.from_cache,
        only=set(args.only) if args.only else None,
        build_overleaf=not args.no_overleaf,
    )
    print(f"Run directory: {ctx.run_dir}")
    print(f"Cache root: {ctx.cache_root}")
    if not args.no_overleaf:
        print(f"Overleaf PDF bundle: {ctx.overleaf_dir}")
    print(f"PDF artifacts: {len(ctx.artifacts)}")
    if ctx.warnings:
        print(f"Warnings: {len(ctx.warnings)}")
        for warning in ctx.warnings[:20]:
            print(f"  - {warning}")
        if len(ctx.warnings) > 20:
            print(f"  - ... {len(ctx.warnings) - 20} more")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
