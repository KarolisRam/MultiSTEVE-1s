from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from datetime import datetime
from pathlib import Path

from steve1.run_agent.teleport_campaign_common import (
    CONDITIONS,
    CONDITION_BY_KEY,
    CORRECTED_IMPORT_OVERRIDES,
    DEFAULT_CHECKPOINT_NAME,
    DEFAULT_RUNNER_ARGS,
    DEFAULT_WEIGHTS_ROOT,
    condition_manifest_path,
    condition_runs_dir,
    discover_weights,
    model_sort_key,
)

STATUS_FIELDS = [
    "condition",
    "model_id",
    "weight_path",
    "status",
    "source",
    "run_dir",
    "staging_dir",
    "attempts",
    "last_return_code",
    "started_at",
    "finished_at",
    "updated_at",
    "notes",
]

MANIFEST_FIELDS = ["model_id", "weight_path", "save_base", "run_dir", "status", "return_code"]
MODEL_FIELDS = ["model_id", "weight_path", "checkpoint_name", "source_group"]


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    os.replace(tmp, path)


def _resolve_repo_path(repo_root: Path, raw: str) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def _load_manifest_by_model_and_run(repo_root: Path, source_dirs: list[Path]) -> tuple[dict[str, dict[str, str]], dict[Path, dict[str, str]]]:
    by_model: dict[str, dict[str, str]] = {}
    by_run: dict[Path, dict[str, str]] = {}
    for source_dir in source_dirs:
        manifest = source_dir / "manifest.csv"
        for row in _read_csv(manifest):
            model_id = row.get("model_id", "")
            run_dir_raw = row.get("run_dir", "")
            if model_id:
                by_model[model_id] = row
            if run_dir_raw:
                by_run[_resolve_repo_path(repo_root, run_dir_raw)] = row
    return by_model, by_run


def _load_corrected_rows(repo_root: Path, condition_key: str) -> tuple[dict[str, dict[str, str]], list[Path]]:
    condition = CONDITION_BY_KEY[condition_key]
    source_dirs = [_resolve_repo_path(repo_root, condition.source_dir)]
    source_by_model: dict[str, dict[str, str]] = {}

    base_csv = source_dirs[0] / "model_comparison.csv"
    for row in _read_csv(base_csv):
        source_by_model[row["model_id"]] = row

    for model_id, override_dir_raw in CORRECTED_IMPORT_OVERRIDES.get(condition_key, {}).items():
        override_dir = _resolve_repo_path(repo_root, override_dir_raw)
        source_dirs.append(override_dir)
        override_rows = {row["model_id"]: row for row in _read_csv(override_dir / "model_comparison.csv")}
        if model_id not in override_rows:
            raise FileNotFoundError(f"Override row for {condition_key}/{model_id} not found in {override_dir}")
        source_by_model[model_id] = override_rows[model_id]

    return source_by_model, source_dirs


def _copytree_atomic(src: Path, final: Path) -> None:
    final.parent.mkdir(parents=True, exist_ok=True)
    if final.exists():
        return
    tmp = final.parent / f".incoming_{final.name}_{os.getpid()}"
    if tmp.exists():
        shutil.rmtree(tmp)
    try:
        shutil.copytree(src, tmp)
        os.replace(tmp, final)
    finally:
        if tmp.exists():
            shutil.rmtree(tmp)


def _existing_status_rows(campaign_root: Path) -> dict[tuple[str, str], dict[str, str]]:
    rows = {}
    for row in _read_csv(campaign_root / "rollout_status.csv"):
        rows[(row.get("condition", ""), row.get("model_id", ""))] = row
    return rows


def _write_readme(campaign_root: Path, weights_roots: list[Path], num_models: int) -> None:
    lines = [
        "# Teleport 100-Model Campaign",
        "",
        "Canonical folder for the villager-tree versus regular-tree teleport rollouts.",
        "",
        "## Fixed Rollout Setup",
        "",
        f"- Prompt: `{DEFAULT_RUNNER_ARGS['prompt']}`",
        f"- Episode length: `{DEFAULT_RUNNER_ARGS['gameplay_length']}` policy steps",
        f"- Teleport interval: every `{DEFAULT_RUNNER_ARGS['teleport_every']}` policy steps",
        f"- Creative-command wait: `{DEFAULT_RUNNER_ARGS['creative_command_wait_steps']}` idle steps",
        f"- Teleport wait: `{DEFAULT_RUNNER_ARGS['teleport_wait_steps']}` idle steps",
        "- Attack actions suppressed: `true`",
        "- Hidden state is reset after every teleport by `run_agent_with_logging_teleport.py`.",
        "- Command feedback is suppressed with `--teleport_feedback_mode gamerule_strict`.",
        "",
        "## Conditions",
        "",
    ]
    for condition in CONDITIONS:
        lines.append(f"- `{condition.key}`: {condition.description}")
    lines.extend(
        [
            "",
            "## Imported Existing Data",
            "",
        "This release path queues all discovered models from the configured weights roots. If you are reconstructing an internal campaign with pre-existing rollout folders, pass `--import_existing` to import those corrected source runs before queuing missing pairs.",
            "",
            "## Model Discovery",
            "",
            f"Imported/queued model count at creation time: `{num_models}`.",
            "",
            "Weights roots used:",
        ]
    )
    for root in weights_roots:
        lines.append(f"- `{root}`")
    lines.extend(
        [
            "",
            "## Running Pending Rollouts",
            "",
            "Start or resume the queue from repo root:",
            "",
            "```bash",
            f"run_agent/run_tmaze_campaign.sh {campaign_root} --max_parallel 2",
            "```",
            "",
            "The queue holds a campaign lock so only one queue process runs for this folder. On startup it resets stale `running` rows to `pending` unless their final run directory was already copied.",
            "",
            "Status and graceful stop:",
            "",
            "```bash",
            f"run_agent/tmaze_campaign_control.sh {campaign_root} status",
            f"run_agent/tmaze_campaign_control.sh {campaign_root} stop",
            "```",
            "",
            "`stop` sends SIGTERM only to the queue controller. Active rollouts are allowed to finish and copy into the canonical folders; the queue then exits without launching more rollouts.",
            "",
            "For graceful interruption, press Ctrl-C once. The queue stops launching new rollouts and waits for already-running rollouts to finish and copy into final condition folders. Partial/interrupted rollouts stay under `staging/` and are not used by analysis.",
            "",
            "## Analysis",
            "",
            "After imports or new runs, regenerate summaries and scatter plots:",
            "",
            "```bash",
            f"run_agent/analyze_tmaze_campaign.sh {campaign_root}",
            "```",
        ]
    )
    (campaign_root / "README.md").write_text("\n".join(lines) + "\n")


def import_campaign(
    repo_root: Path,
    campaign_root: Path,
    weights_roots: list[Path],
    checkpoint_name: str,
    *,
    import_existing: bool = False,
) -> None:
    campaign_root.mkdir(parents=True, exist_ok=True)
    (campaign_root / "conditions").mkdir(exist_ok=True)
    (campaign_root / "staging").mkdir(exist_ok=True)
    (campaign_root / "logs").mkdir(exist_ok=True)
    (campaign_root / "analysis").mkdir(exist_ok=True)

    weights = discover_weights(weights_roots, checkpoint_name=checkpoint_name)
    existing = _existing_status_rows(campaign_root)
    now = _now()

    all_status: dict[tuple[str, str], dict[str, object]] = dict(existing)
    imported_models: set[str] = set()
    imported_pairs: set[tuple[str, str]] = set()

    for condition in CONDITIONS:
        condition_dir = campaign_root / "conditions" / condition.key
        runs_dir = condition_runs_dir(campaign_root, condition.key)
        condition_dir.mkdir(parents=True, exist_ok=True)
        runs_dir.mkdir(parents=True, exist_ok=True)
        manifest_rows: list[dict[str, object]] = []
        if not import_existing:
            _write_csv(condition_manifest_path(campaign_root, condition.key), MANIFEST_FIELDS, manifest_rows)
            continue

        corrected_rows, source_dirs = _load_corrected_rows(repo_root, condition.key)
        manifest_by_model, manifest_by_run = _load_manifest_by_model_and_run(repo_root, source_dirs)

        for model_id, comparison_row in sorted(corrected_rows.items(), key=lambda item: model_sort_key(item[0])):
            src_run = _resolve_repo_path(repo_root, comparison_row["run_dir"])
            if not src_run.exists():
                raise FileNotFoundError(f"Missing source run dir for {condition.key}/{model_id}: {src_run}")
            source_manifest_row = manifest_by_run.get(src_run) or manifest_by_model.get(model_id, {})
            weight_path = str(weights.get(model_id, Path(source_manifest_row.get("weight_path", ""))))
            timestamp = src_run.name
            final_dir = runs_dir / model_id / timestamp
            _copytree_atomic(src_run, final_dir)
            save_base = final_dir.parent
            note = "imported from existing corrected source"
            if model_id in CORRECTED_IMPORT_OVERRIDES.get(condition.key, {}):
                note = "imported corrective rerun; supersedes bad original"
            manifest_rows.append(
                {
                    "model_id": model_id,
                    "weight_path": weight_path,
                    "save_base": str(save_base),
                    "run_dir": str(final_dir),
                    "status": "ok",
                    "return_code": "0",
                }
            )
            all_status[(condition.key, model_id)] = {
                "condition": condition.key,
                "model_id": model_id,
                "weight_path": weight_path,
                "status": "done",
                "source": "imported_corrected",
                "run_dir": str(final_dir),
                "staging_dir": "",
                "attempts": existing.get((condition.key, model_id), {}).get("attempts", "0"),
                "last_return_code": "0",
                "started_at": "",
                "finished_at": now,
                "updated_at": now,
                "notes": note,
            }
            imported_models.add(model_id)
            imported_pairs.add((condition.key, model_id))

        _write_csv(condition_manifest_path(campaign_root, condition.key), MANIFEST_FIELDS, manifest_rows)

    for model_id, weight_path in weights.items():
        for condition in CONDITIONS:
            key = (condition.key, model_id)
            if key in imported_pairs:
                continue
            if key in all_status and all_status[key].get("status") == "done":
                continue
            all_status[key] = {
                "condition": condition.key,
                "model_id": model_id,
                "weight_path": str(weight_path),
                "status": all_status.get(key, {}).get("status", "pending"),
                "source": "queued_weight_discovery",
                "run_dir": all_status.get(key, {}).get("run_dir", ""),
                "staging_dir": all_status.get(key, {}).get("staging_dir", ""),
                "attempts": all_status.get(key, {}).get("attempts", "0"),
                "last_return_code": all_status.get(key, {}).get("last_return_code", ""),
                "started_at": all_status.get(key, {}).get("started_at", ""),
                "finished_at": all_status.get(key, {}).get("finished_at", ""),
                "updated_at": now,
                "notes": all_status.get(key, {}).get("notes", "pending rollout"),
            }

    status_rows = sorted(
        all_status.values(),
        key=lambda row: (model_sort_key(str(row.get("model_id", ""))), row.get("condition", "")),
    )
    _write_csv(campaign_root / "rollout_status.csv", STATUS_FIELDS, status_rows)

    model_rows = []
    for model_id, weight_path in weights.items():
        source_group = "imported_existing" if model_id in imported_models else "queued_from_weights"
        model_rows.append(
            {
                "model_id": model_id,
                "weight_path": str(weight_path),
                "checkpoint_name": checkpoint_name,
                "source_group": source_group,
            }
        )
    _write_csv(campaign_root / "models.csv", MODEL_FIELDS, model_rows)

    run_config = {
        "created_or_updated_at": now,
        "campaign_root": str(campaign_root),
        "checkpoint_name": checkpoint_name,
        "weights_roots": [str(root) for root in weights_roots],
        "runner_args": DEFAULT_RUNNER_ARGS,
        "conditions": [condition.__dict__ for condition in CONDITIONS],
        "import_existing": import_existing,
        "corrected_import_overrides": CORRECTED_IMPORT_OVERRIDES if import_existing else {},
        "status_fields": STATUS_FIELDS,
        "manifest_fields": MANIFEST_FIELDS,
    }
    (campaign_root / "run_config.json").write_text(json.dumps(run_config, indent=2) + "\n")
    _write_readme(campaign_root, weights_roots, len(weights))

    done_count = sum(1 for row in status_rows if row.get("status") == "done")
    pending_count = sum(1 for row in status_rows if row.get("status") == "pending")
    print(f"campaign_root={campaign_root}")
    print(f"models={len(weights)} done_pairs={done_count} pending_pairs={pending_count}")
    print(f"status={campaign_root / 'rollout_status.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign_root", required=True)
    parser.add_argument("--repo_root", default=".")
    parser.add_argument("--weights_root", action="append", default=None)
    parser.add_argument("--checkpoint_name", default=DEFAULT_CHECKPOINT_NAME)
    parser.add_argument("--import_existing", action="store_true")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    campaign_root = Path(args.campaign_root)
    if not campaign_root.is_absolute():
        campaign_root = repo_root / campaign_root
    weights_roots = [Path(p) for p in (args.weights_root or [str(DEFAULT_WEIGHTS_ROOT)])]
    import_campaign(
        repo_root,
        campaign_root.resolve(),
        weights_roots,
        args.checkpoint_name,
        import_existing=bool(args.import_existing),
    )


if __name__ == "__main__":
    main()
