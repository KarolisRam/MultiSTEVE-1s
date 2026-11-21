#!/usr/bin/env python3
"""Episode queue orchestrator for STEVE-1 rollout campaigns.

Features:
- Campaign-based enqueue for large prompt/model/seed matrices.
- SQLite-backed queue state with statuses: pending/running/completed/failed.
- Multi-process safe claiming (run one or two worker processes).
- Lease + heartbeat so stale running jobs are re-queued automatically.
- Final output-only episode folders (no attempt folders in episode output tree).

Typical usage:
  run_agent/episode_queue.sh enqueue \
      --campaign configs/eval_campaigns/full_partial_eval.json --dry-run

  run_agent/episode_queue.sh enqueue \
      --campaign configs/eval_campaigns/full_partial_eval.json

  run_agent/episode_queue.sh worker \
      --campaign configs/eval_campaigns/full_partial_eval.json --worker-id worker-1

  run_agent/episode_queue.sh status \
      --campaign configs/eval_campaigns/full_partial_eval.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import signal
import sqlite3
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CAMPAIGN_PATH = REPO_ROOT / "configs" / "eval_campaigns" / "full_partial_eval.json"
SNAPSHOT_RE = re.compile(r"^STEVE-1_snapshot_(\d+)\.weights$")

STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"
ALL_STATUSES = [STATUS_PENDING, STATUS_RUNNING, STATUS_COMPLETED, STATUS_FAILED]


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def utc_after(seconds: int) -> str:
    return (datetime.now(timezone.utc) + timedelta(seconds=seconds)).strftime("%Y-%m-%dT%H:%M:%SZ")


def dt_token() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def resolve_path(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def slugify(text: str, max_len: int = 80) -> str:
    text = text.strip().lower()
    out = []
    for ch in text:
        if ch.isalnum():
            out.append(ch)
        elif ch in {" ", "-", "_", ".", "/"}:
            out.append("-")
    slug = "".join(out)
    while "--" in slug:
        slug = slug.replace("--", "-")
    slug = slug.strip("-")
    if not slug:
        slug = "item"
    return slug[:max_len]


def safe_component(text: str, max_len: int = 120) -> str:
    text = text.strip()
    if not text:
        return "item"
    out = []
    for ch in text:
        if ch.isalnum() or ch in {"-", "_", "."}:
            out.append(ch)
        else:
            out.append("_")
    value = "".join(out).strip("_")
    while "__" in value:
        value = value.replace("__", "_")
    if not value:
        value = "item"
    return value[:max_len]


def parse_snapshot_step(path: Path) -> Optional[int]:
    match = SNAPSHOT_RE.match(path.name)
    if match is None:
        return None
    return int(match.group(1))


def list_snapshot_files(model_dir: Path) -> List[Tuple[int, Path]]:
    if not model_dir.exists() or not model_dir.is_dir():
        return []
    out: List[Tuple[int, Path]] = []
    for child in model_dir.iterdir():
        if not child.is_file():
            continue
        step = parse_snapshot_step(child)
        if step is None:
            continue
        out.append((step, child))
    out.sort(key=lambda item: item[0])
    return out


def resolve_checkpoint(
    model_dir: Path,
    checkpoint_step: Optional[int],
    fallback_to_latest: bool,
) -> Optional[Path]:
    if checkpoint_step is not None:
        target = model_dir / f"STEVE-1_snapshot_{int(checkpoint_step):09d}.weights"
        if target.exists():
            return target
        if not fallback_to_latest:
            return None

    snapshots = list_snapshot_files(model_dir)
    if not snapshots:
        return None
    return snapshots[-1][1]


def detect_new_run_dir(parent: Path, before_dirs: Iterable[str]) -> Optional[Path]:
    if not parent.exists():
        return None
    existing = [p for p in parent.iterdir() if p.is_dir()]
    if not existing:
        return None
    before = set(before_dirs)
    fresh = [p for p in existing if p.name not in before]
    candidates = fresh if fresh else existing
    return max(candidates, key=lambda p: p.stat().st_mtime)


def interrupt_process(proc: subprocess.Popen[Any]) -> None:
    if proc.poll() is not None:
        return
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=20)
        return
    except subprocess.TimeoutExpired:
        pass

    proc.terminate()
    try:
        proc.wait(timeout=10)
        return
    except subprocess.TimeoutExpired:
        pass

    proc.kill()
    proc.wait(timeout=5)


def ensure_unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    for idx in range(1, 1000):
        candidate = Path(f"{path}__dup{idx:03d}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not allocate unique output path for {path}")


def parse_int_list(raw: Sequence[Any], key: str) -> List[int]:
    out: List[int] = []
    for item in raw:
        try:
            out.append(int(item))
        except Exception as exc:
            raise ValueError(f"Invalid int in '{key}': {item}") from exc
    if not out:
        raise ValueError(f"'{key}' must not be empty")
    return out


@dataclass(frozen=True)
class ModelSpec:
    label: str
    weights_path: str


@dataclass(frozen=True)
class PromptSpec:
    prompt_type: str  # text | visual
    slug: str
    value: str


class StopController:
    def __init__(self) -> None:
        self.stop_requested = False
        self.signal_name: Optional[str] = None

    def request_stop(self, signum: int) -> None:
        self.stop_requested = True
        try:
            self.signal_name = signal.Signals(signum).name
        except Exception:
            self.signal_name = str(signum)

    def install(self) -> None:
        def _handler(signum: int, _frame: Any) -> None:
            self.request_stop(signum)
            print(
                f"[worker] stop requested via {self.signal_name}; "
                "will interrupt current subprocess and re-queue that job.",
                flush=True,
            )

        signal.signal(signal.SIGINT, _handler)
        signal.signal(signal.SIGTERM, _handler)


class JobStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30.0, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    experiment_name TEXT NOT NULL,
                    model_label TEXT NOT NULL,
                    prompt_type TEXT NOT NULL,
                    prompt_slug TEXT NOT NULL,
                    prompt_value TEXT NOT NULL,
                    seed INTEGER NOT NULL,
                    weights_path TEXT NOT NULL,
                    output_rel_template TEXT NOT NULL,
                    custom_text_seed INTEGER,
                    payload_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    worker_id TEXT,
                    lease_expires_at TEXT,
                    heartbeat_at TEXT,
                    started_at TEXT,
                    finished_at TEXT,
                    return_code INTEGER,
                    error_message TEXT,
                    final_output_dir TEXT,
                    log_file TEXT,
                    next_eligible_at TEXT,
                    defer_count INTEGER NOT NULL DEFAULT 0,
                    failed_retry_count INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )
            self._ensure_column(conn, "jobs", "next_eligible_at", "TEXT")
            self._ensure_column(conn, "jobs", "defer_count", "INTEGER NOT NULL DEFAULT 0")
            self._ensure_column(conn, "jobs", "failed_retry_count", "INTEGER NOT NULL DEFAULT 0")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_jobs_exp_status ON jobs(experiment_name, status);"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_jobs_lease ON jobs(status, lease_expires_at);"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_jobs_next_eligible ON jobs(status, next_eligible_at);"
            )

    @staticmethod
    def _ensure_column(conn: sqlite3.Connection, table: str, column: str, definition: str) -> None:
        cols = [str(row["name"]) for row in conn.execute(f"PRAGMA table_info({table})").fetchall()]
        if column in cols:
            return
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition};")

    def insert_jobs(self, jobs: Sequence[Dict[str, Any]]) -> Tuple[int, int]:
        inserted = 0
        existing = 0
        now = utc_now()
        with self._connect() as conn:
            for job in jobs:
                cursor = conn.execute(
                    """
                    INSERT OR IGNORE INTO jobs (
                        id, experiment_name, model_label, prompt_type, prompt_slug, prompt_value,
                        seed, weights_path, output_rel_template, custom_text_seed, payload_json,
                        status, next_eligible_at, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        job["id"],
                        job["experiment_name"],
                        job["model_label"],
                        job["prompt_type"],
                        job["prompt_slug"],
                        job["prompt_value"],
                        int(job["seed"]),
                        job["weights_path"],
                        job["output_rel_template"],
                        job.get("custom_text_seed"),
                        json.dumps(job, sort_keys=True),
                        STATUS_PENDING,
                        now,
                        now,
                        now,
                    ),
                )
                if cursor.rowcount == 1:
                    inserted += 1
                else:
                    existing += 1
        return inserted, existing

    def requeue_stale_running(self) -> int:
        now = utc_now()
        with self._connect() as conn:
            cursor = conn.execute(
                """
                UPDATE jobs
                SET status = ?, worker_id = NULL, lease_expires_at = NULL, heartbeat_at = NULL,
                    started_at = NULL, next_eligible_at = ?, updated_at = ?
                WHERE status = ? AND lease_expires_at IS NOT NULL AND lease_expires_at < ?
                """,
                (STATUS_PENDING, now, now, STATUS_RUNNING, now),
            )
            return int(cursor.rowcount)

    def claim_next_job(self, worker_id: str, lease_timeout_seconds: int) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            try:
                conn.execute("BEGIN IMMEDIATE;")
                now = utc_now()
                row = conn.execute(
                    """
                    SELECT id
                    FROM jobs
                    WHERE status = ?
                      AND (next_eligible_at IS NULL OR next_eligible_at <= ?)
                    ORDER BY
                        experiment_name ASC,
                        model_label ASC,
                        prompt_type ASC,
                        prompt_slug ASC,
                        seed ASC,
                        id ASC
                    LIMIT 1
                    """,
                    (STATUS_PENDING, now),
                ).fetchone()
                if row is None:
                    conn.execute("COMMIT;")
                    return None

                lease = utc_after(lease_timeout_seconds)
                updated = conn.execute(
                    """
                    UPDATE jobs
                    SET status = ?, worker_id = ?, started_at = ?, heartbeat_at = ?,
                        lease_expires_at = ?, updated_at = ?, error_message = NULL
                    WHERE id = ? AND status = ?
                    """,
                    (STATUS_RUNNING, worker_id, now, now, lease, now, row["id"], STATUS_PENDING),
                )
                if updated.rowcount != 1:
                    conn.execute("ROLLBACK;")
                    return None

                job = conn.execute("SELECT * FROM jobs WHERE id = ?", (row["id"],)).fetchone()
                conn.execute("COMMIT;")
                if job is None:
                    return None
                return dict(job)
            except Exception:
                conn.execute("ROLLBACK;")
                raise

    def refresh_lease(self, job_id: str, worker_id: str, lease_timeout_seconds: int) -> None:
        now = utc_now()
        lease = utc_after(lease_timeout_seconds)
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE jobs
                SET heartbeat_at = ?, lease_expires_at = ?, updated_at = ?
                WHERE id = ? AND status = ? AND worker_id = ?
                """,
                (now, lease, now, job_id, STATUS_RUNNING, worker_id),
            )

    def mark_completed(
        self,
        job_id: str,
        worker_id: str,
        return_code: int,
        final_output_dir: str,
        log_file: str,
    ) -> None:
        now = utc_now()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE jobs
                SET status = ?, worker_id = NULL, lease_expires_at = NULL, heartbeat_at = NULL,
                    finished_at = ?, updated_at = ?, return_code = ?, error_message = NULL,
                    final_output_dir = ?, log_file = ?
                WHERE id = ? AND status = ? AND worker_id = ?
                """,
                (
                    STATUS_COMPLETED,
                    now,
                    now,
                    int(return_code),
                    final_output_dir,
                    log_file,
                    job_id,
                    STATUS_RUNNING,
                    worker_id,
                ),
            )

    def mark_failed(
        self,
        job_id: str,
        worker_id: str,
        return_code: int,
        error_message: str,
        log_file: str,
    ) -> None:
        now = utc_now()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE jobs
                SET status = ?, worker_id = NULL, lease_expires_at = NULL, heartbeat_at = NULL,
                    finished_at = ?, updated_at = ?, return_code = ?, error_message = ?, log_file = ?
                WHERE id = ? AND status = ? AND worker_id = ?
                """,
                (
                    STATUS_FAILED,
                    now,
                    now,
                    int(return_code),
                    error_message,
                    log_file,
                    job_id,
                    STATUS_RUNNING,
                    worker_id,
                ),
            )

    def requeue_running_job(self, job_id: str, worker_id: str, reason: str) -> None:
        now = utc_now()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE jobs
                SET status = ?, worker_id = NULL, lease_expires_at = NULL, heartbeat_at = NULL,
                    started_at = NULL, next_eligible_at = ?, updated_at = ?, error_message = ?
                WHERE id = ? AND status = ? AND worker_id = ?
                """,
                (STATUS_PENDING, now, now, reason, job_id, STATUS_RUNNING, worker_id),
            )

    def defer_running_job(
        self,
        job_id: str,
        worker_id: str,
        reason: str,
        retry_delay_seconds: int,
    ) -> None:
        now = utc_now()
        next_eligible = utc_after(retry_delay_seconds)
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE jobs
                SET status = ?, worker_id = NULL, lease_expires_at = NULL, heartbeat_at = NULL,
                    started_at = NULL, next_eligible_at = ?, updated_at = ?, error_message = ?,
                    defer_count = COALESCE(defer_count, 0) + 1
                WHERE id = ? AND status = ? AND worker_id = ?
                """,
                (
                    STATUS_PENDING,
                    next_eligible,
                    now,
                    reason,
                    job_id,
                    STATUS_RUNNING,
                    worker_id,
                ),
            )

    def retry_running_failed_job(
        self,
        job_id: str,
        worker_id: str,
        reason: str,
        return_code: int,
        retry_delay_seconds: int,
    ) -> None:
        now = utc_now()
        next_eligible = utc_after(retry_delay_seconds)
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE jobs
                SET status = ?, worker_id = NULL, lease_expires_at = NULL, heartbeat_at = NULL,
                    started_at = NULL, next_eligible_at = ?, finished_at = ?, updated_at = ?,
                    return_code = ?, error_message = ?, failed_retry_count = COALESCE(failed_retry_count, 0) + 1
                WHERE id = ? AND status = ? AND worker_id = ?
                """,
                (
                    STATUS_PENDING,
                    next_eligible,
                    now,
                    now,
                    int(return_code),
                    reason,
                    job_id,
                    STATUS_RUNNING,
                    worker_id,
                ),
            )

    def requeue_retryable_failed_jobs(
        self,
        max_retries: int,
    ) -> int:
        now = utc_now()
        with self._connect() as conn:
            if max_retries < 0:
                cursor = conn.execute(
                    """
                    UPDATE jobs
                    SET status = ?, worker_id = NULL, lease_expires_at = NULL, heartbeat_at = NULL,
                        started_at = NULL, next_eligible_at = ?, updated_at = ?
                    WHERE status = ? AND (next_eligible_at IS NULL OR next_eligible_at <= ?)
                    """,
                    (STATUS_PENDING, now, now, STATUS_FAILED, now),
                )
            else:
                cursor = conn.execute(
                    """
                    UPDATE jobs
                    SET status = ?, worker_id = NULL, lease_expires_at = NULL, heartbeat_at = NULL,
                        started_at = NULL, next_eligible_at = ?, updated_at = ?
                    WHERE status = ?
                      AND (next_eligible_at IS NULL OR next_eligible_at <= ?)
                      AND COALESCE(failed_retry_count, 0) < ?
                    """,
                    (STATUS_PENDING, now, now, STATUS_FAILED, now, int(max_retries)),
                )
            return int(cursor.rowcount)

    def totals(self) -> Dict[str, int]:
        out = {status: 0 for status in ALL_STATUSES}
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT status, COUNT(*) AS n FROM jobs GROUP BY status"
            ).fetchall()
        for row in rows:
            out[str(row["status"])] = int(row["n"])
        return out

    def per_experiment_totals(self) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT experiment_name, status, COUNT(*) AS n
                FROM jobs
                GROUP BY experiment_name, status
                ORDER BY experiment_name ASC
                """
            ).fetchall()

        by_exp: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            exp = str(row["experiment_name"])
            status = str(row["status"])
            n = int(row["n"])
            if exp not in by_exp:
                by_exp[exp] = {"experiment_name": exp, "total": 0, **{s: 0 for s in ALL_STATUSES}}
            by_exp[exp][status] = n
            by_exp[exp]["total"] += n
        return [by_exp[key] for key in sorted(by_exp.keys())]

    def list_by_status(self, status: str, limit: int) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, experiment_name, model_label, prompt_type, prompt_slug, seed,
                       worker_id, started_at, error_message
                FROM jobs
                WHERE status = ?
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (status, int(limit)),
            ).fetchall()
        return [dict(row) for row in rows]


def apply_campaign_defaults(campaign: Dict[str, Any]) -> Dict[str, Any]:
    cfg = dict(campaign)
    cfg.setdefault("queue_db_path", "data/episode_queue/state.sqlite3")
    cfg.setdefault("state_root", "data/episode_queue")
    cfg.setdefault("episodes_root", "data/episodes")
    cfg.setdefault("runner_script", "steve1/run_agent/run_agent_with_logging.py")
    cfg.setdefault("default_episode_seeds", list(range(1, 21)))
    cfg.setdefault("experiments", [])

    common = cfg.setdefault("common_args", {})
    common.setdefault("in_model", "data/weights/vpt/2x.model")
    common.setdefault("prior_weights", "data/weights/steve1/steve1_prior.pt")
    common.setdefault("gameplay_length", 1000)
    common.setdefault("text_cond_scale", 6.0)
    common.setdefault("visual_cond_scale", 7.0)
    common.setdefault("save_additional_data", False)
    common.setdefault("fake_env_zip", None)
    common.setdefault("wipe_hidden_state_each_step", False)
    common.setdefault("placement", [])
    common.setdefault("extra_args", [])

    runtime = cfg.setdefault("runtime", {})
    runtime.setdefault("conda_env", "steve-1-creative")
    runtime.setdefault("heartbeat_seconds", 30)
    runtime.setdefault("lease_timeout_seconds", 1200)
    runtime.setdefault("missing_weights_retry_seconds", 900)
    runtime.setdefault("failed_retry_seconds", 300)
    runtime.setdefault("failed_max_retries", -1)
    runtime.setdefault("idle_sleep_seconds", 5)

    return cfg


def parse_text_prompts(raw: Any, experiment_name: str) -> List[PromptSpec]:
    if not raw:
        return []
    if not isinstance(raw, list):
        raise ValueError(f"{experiment_name}: 'text_prompts' must be a list")

    out: List[PromptSpec] = []
    for idx, item in enumerate(raw):
        if isinstance(item, str):
            text = item.strip()
            name = text
        elif isinstance(item, dict):
            text = str(item.get("text", "")).strip()
            name = str(item.get("name") or text).strip()
        else:
            raise ValueError(f"{experiment_name}: invalid text prompt at index {idx}")
        if not text:
            raise ValueError(f"{experiment_name}: empty text prompt at index {idx}")
        out.append(PromptSpec(prompt_type="text", slug=slugify(name), value=text))
    return out


def parse_visual_prompts(raw: Any, experiment_name: str) -> List[PromptSpec]:
    if not raw:
        return []
    if not isinstance(raw, list):
        raise ValueError(f"{experiment_name}: 'visual_prompts' must be a list")

    out: List[PromptSpec] = []
    for idx, item in enumerate(raw):
        if not isinstance(item, str):
            raise ValueError(f"{experiment_name}: invalid visual prompt at index {idx}")
        value = item.strip()
        if not value:
            raise ValueError(f"{experiment_name}: empty visual prompt at index {idx}")
        out.append(PromptSpec(prompt_type="visual", slug=slugify(value), value=value))
    return out


def discover_models(exp: Dict[str, Any]) -> Tuple[List[ModelSpec], List[str]]:
    name = str(exp.get("name", "experiment"))
    source = exp.get("model_source", {})
    if not isinstance(source, dict):
        raise ValueError(f"{name}: 'model_source' must be an object")

    source_type = str(source.get("type", "")).strip()
    if not source_type:
        raise ValueError(f"{name}: model_source.type is required")

    warnings: List[str] = []
    models: List[ModelSpec] = []

    if source_type == "explicit_models":
        raw_models = source.get("models", [])
        if not isinstance(raw_models, list) or not raw_models:
            raise ValueError(f"{name}: explicit_models requires non-empty 'models'")
        for idx, item in enumerate(raw_models):
            if not isinstance(item, dict):
                raise ValueError(f"{name}: invalid model spec at index {idx}")
            label = str(item.get("label", "")).strip()
            weights_path = str(item.get("weights_path", "")).strip()
            if not label or not weights_path:
                raise ValueError(f"{name}: model[{idx}] requires 'label' and 'weights_path'")
            resolved = resolve_path(weights_path)
            if not resolved.exists():
                warnings.append(f"{name}: missing weights path: {resolved}")
                if not bool(source.get("skip_missing_weights", False)):
                    raise ValueError(f"{name}: required weights file not found: {resolved}")
                continue
            models.append(ModelSpec(label=label, weights_path=str(resolved)))

    elif source_type == "models_from_roots":
        roots_raw = source.get("roots", [])
        if not isinstance(roots_raw, list) or not roots_raw:
            raise ValueError(f"{name}: models_from_roots requires non-empty 'roots'")
        checkpoint_step = source.get("checkpoint_step")
        checkpoint_step = int(checkpoint_step) if checkpoint_step is not None else None
        fallback_to_latest = bool(source.get("fallback_to_latest", True))
        include_root = bool(source.get("include_root_in_label", True))
        skip_missing_weights = bool(source.get("skip_missing_weights", True))
        enqueue_missing = bool(source.get("enqueue_missing_weights_jobs", False))

        for root_raw in roots_raw:
            root = resolve_path(str(root_raw))
            if not root.exists() or not root.is_dir():
                warnings.append(f"{name}: missing model root: {root}")
                continue
            children = sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.name)
            for child in children:
                resolved = resolve_checkpoint(child, checkpoint_step, fallback_to_latest)
                if resolved is None:
                    msg = (
                        f"{name}: no checkpoint found in {child} "
                        f"(checkpoint_step={checkpoint_step}, fallback_to_latest={fallback_to_latest})"
                    )
                    warnings.append(msg)
                    if enqueue_missing and checkpoint_step is not None:
                        label = f"{root.name}__{child.name}" if include_root else child.name
                        expected = child / f"STEVE-1_snapshot_{int(checkpoint_step):09d}.weights"
                        models.append(ModelSpec(label=label, weights_path=str(expected)))
                        continue
                    if not skip_missing_weights:
                        raise ValueError(msg)
                    continue
                label = f"{root.name}__{child.name}" if include_root else child.name
                models.append(ModelSpec(label=label, weights_path=str(resolved.resolve())))

    elif source_type == "checkpoints_in_run":
        run_dir_raw = source.get("run_dir")
        if not run_dir_raw:
            raise ValueError(f"{name}: checkpoints_in_run requires 'run_dir'")
        run_dir = resolve_path(str(run_dir_raw))
        include_step_zero = bool(source.get("include_step_zero", False))
        if not run_dir.exists() or not run_dir.is_dir():
            raise ValueError(f"{name}: run_dir does not exist: {run_dir}")
        snapshots = list_snapshot_files(run_dir)
        if not snapshots:
            raise ValueError(f"{name}: no STEVE-1 snapshots found in {run_dir}")
        for step, path in snapshots:
            if step == 0 and not include_step_zero:
                continue
            label = path.stem
            models.append(ModelSpec(label=label, weights_path=str(path.resolve())))

    elif source_type == "seed_range_template":
        root_raw = source.get("root")
        dir_template = str(source.get("dir_template", "")).strip()
        seed_start = source.get("seed_start")
        seed_end = source.get("seed_end")
        if not root_raw or not dir_template:
            raise ValueError(f"{name}: seed_range_template requires 'root' and 'dir_template'")
        if seed_start is None or seed_end is None:
            raise ValueError(f"{name}: seed_range_template requires seed_start and seed_end")

        root = resolve_path(str(root_raw))
        checkpoint_step = source.get("checkpoint_step")
        checkpoint_step = int(checkpoint_step) if checkpoint_step is not None else None
        fallback_to_latest = bool(source.get("fallback_to_latest", False))
        skip_missing_dirs = bool(source.get("skip_missing_dirs", True))
        skip_missing_weights = bool(source.get("skip_missing_weights", True))
        enqueue_missing = bool(source.get("enqueue_missing_weights_jobs", False))
        missing_dir_count = 0
        missing_dir_samples: List[str] = []
        missing_weights_count = 0
        missing_weights_samples: List[str] = []

        for seed in range(int(seed_start), int(seed_end) + 1):
            model_dir = root / dir_template.format(seed=seed)
            if not model_dir.exists() or not model_dir.is_dir():
                msg = f"{name}: missing model directory for seed {seed}: {model_dir}"
                if not skip_missing_dirs:
                    raise ValueError(msg)
                missing_dir_count += 1
                if len(missing_dir_samples) < 5:
                    missing_dir_samples.append(str(model_dir))
                if enqueue_missing and checkpoint_step is not None:
                    expected = model_dir / f"STEVE-1_snapshot_{int(checkpoint_step):09d}.weights"
                    models.append(ModelSpec(label=model_dir.name, weights_path=str(expected)))
                continue

            resolved = resolve_checkpoint(model_dir, checkpoint_step, fallback_to_latest)
            if resolved is None:
                msg = (
                    f"{name}: missing checkpoint in {model_dir} "
                    f"(checkpoint_step={checkpoint_step}, fallback_to_latest={fallback_to_latest})"
                )
                if not skip_missing_weights:
                    raise ValueError(msg)
                missing_weights_count += 1
                if len(missing_weights_samples) < 5:
                    missing_weights_samples.append(str(model_dir))
                if enqueue_missing and checkpoint_step is not None:
                    expected = model_dir / f"STEVE-1_snapshot_{int(checkpoint_step):09d}.weights"
                    models.append(ModelSpec(label=model_dir.name, weights_path=str(expected)))
                continue
            models.append(ModelSpec(label=model_dir.name, weights_path=str(resolved.resolve())))

        if missing_dir_count > 0:
            action = "enqueued for retry" if enqueue_missing else "skipped"
            warnings.append(
                f"{name}: missing model directories: {missing_dir_count} ({action}) "
                f"(sample: {', '.join(missing_dir_samples)})"
            )
        if missing_weights_count > 0:
            action = "enqueued for retry" if enqueue_missing else "skipped"
            warnings.append(
                f"{name}: missing checkpoints: {missing_weights_count} ({action}) "
                f"(sample: {', '.join(missing_weights_samples)})"
            )

    else:
        raise ValueError(f"{name}: unsupported model_source.type '{source_type}'")

    expected_model_count = source.get("expected_model_count")
    if expected_model_count is not None:
        expected = int(expected_model_count)
        if len(models) != expected:
            warnings.append(f"{name}: expected {expected} models, discovered {len(models)}")

    if not models:
        warnings.append(f"{name}: discovered 0 models")
    return models, warnings


def build_prompts(exp: Dict[str, Any]) -> List[PromptSpec]:
    name = str(exp.get("name", "experiment"))
    text_prompts = parse_text_prompts(exp.get("text_prompts", []), name)
    visual_prompts = parse_visual_prompts(exp.get("visual_prompts", []), name)

    expected_text = exp.get("expected_text_prompt_count")
    if expected_text is not None and len(text_prompts) != int(expected_text):
        raise ValueError(
            f"{name}: expected_text_prompt_count={expected_text}, got {len(text_prompts)}"
        )

    expected_visual = exp.get("expected_visual_prompt_count")
    if expected_visual is not None and len(visual_prompts) != int(expected_visual):
        raise ValueError(
            f"{name}: expected_visual_prompt_count={expected_visual}, got {len(visual_prompts)}"
        )

    prompts = [*text_prompts, *visual_prompts]
    if not prompts:
        raise ValueError(f"{name}: no prompts configured")
    return prompts


def make_job_id(payload: Dict[str, Any]) -> str:
    key = {
        "experiment_name": payload["experiment_name"],
        "model_label": payload["model_label"],
        "weights_path": payload["weights_path"],
        "prompt_type": payload["prompt_type"],
        "prompt_value": payload["prompt_value"],
        "seed": int(payload["seed"]),
        "output_rel_template": payload["output_rel_template"],
    }
    if "common_args_override" in payload:
        key["common_args_override"] = payload["common_args_override"]
    raw = json.dumps(key, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
    prefix = slugify(str(payload["experiment_name"]), max_len=20)
    return f"{prefix}__{digest}"


def build_jobs_from_campaign(
    campaign: Dict[str, Any],
    selected_experiments: Optional[Sequence[str]],
) -> Tuple[List[Dict[str, Any]], List[str], Dict[str, int]]:
    only = set(selected_experiments or [])
    default_seeds = parse_int_list(campaign.get("default_episode_seeds", list(range(1, 21))), "default_episode_seeds")
    jobs: List[Dict[str, Any]] = []
    warnings: List[str] = []
    per_experiment: Dict[str, int] = {}
    seen_ids: set[str] = set()

    experiments = campaign.get("experiments", [])
    if not isinstance(experiments, list) or not experiments:
        raise ValueError("Campaign 'experiments' must be a non-empty list")

    for exp in experiments:
        if not isinstance(exp, dict):
            raise ValueError("Each experiment entry must be an object")
        name = str(exp.get("name", "")).strip()
        if not name:
            raise ValueError("Each experiment requires a non-empty 'name'")
        if only and name not in only:
            continue

        exp_seeds_raw = exp.get("episode_seeds", default_seeds)
        exp_seeds = parse_int_list(exp_seeds_raw, f"{name}.episode_seeds")
        output_dir = str(exp.get("output_dir", name)).strip()
        if not output_dir:
            raise ValueError(f"{name}: output_dir must not be empty")

        prompts = build_prompts(exp)
        models, model_warnings = discover_models(exp)
        warnings.extend(model_warnings)

        text_custom_seed_mode = str(exp.get("text_custom_seed_mode", "episode_seed")).strip().lower()
        fixed_custom_seed = exp.get("text_custom_seed")
        common_args_override = exp.get("common_args_override")
        if common_args_override is not None and not isinstance(common_args_override, dict):
            raise ValueError(f"{name}: common_args_override must be an object when provided")

        start_count = len(jobs)
        for model in models:
            model_component = safe_component(model.label)
            for prompt in prompts:
                prompt_component = safe_component(prompt.slug)
                for seed in exp_seeds:
                    custom_text_seed: Optional[int] = None
                    if prompt.prompt_type == "text":
                        if text_custom_seed_mode == "episode_seed":
                            custom_text_seed = int(seed)
                        elif text_custom_seed_mode == "fixed":
                            if fixed_custom_seed is None:
                                raise ValueError(
                                    f"{name}: text_custom_seed_mode='fixed' requires text_custom_seed"
                                )
                            custom_text_seed = int(fixed_custom_seed)
                        elif text_custom_seed_mode == "none":
                            custom_text_seed = None
                        else:
                            raise ValueError(
                                f"{name}: unsupported text_custom_seed_mode '{text_custom_seed_mode}'"
                            )

                    output_rel_template = (
                        f"{output_dir}/{model_component}/{prompt.prompt_type}/{prompt_component}/"
                        f"seed_{int(seed)}_{{dt}}"
                    )
                    payload: Dict[str, Any] = {
                        "experiment_name": name,
                        "model_label": model.label,
                        "weights_path": model.weights_path,
                        "prompt_type": prompt.prompt_type,
                        "prompt_slug": prompt.slug,
                        "prompt_value": prompt.value,
                        "seed": int(seed),
                        "custom_text_seed": custom_text_seed,
                        "output_rel_template": output_rel_template,
                    }
                    if common_args_override is not None:
                        payload["common_args_override"] = common_args_override
                    payload["id"] = make_job_id(payload)
                    if payload["id"] in seen_ids:
                        raise ValueError(
                            f"{name}: duplicate job id generated for model={model.label}, "
                            f"prompt={prompt.slug}, seed={seed}"
                        )
                    seen_ids.add(payload["id"])
                    jobs.append(payload)

        per_experiment[name] = len(jobs) - start_count

    if only:
        missing = sorted(exp_name for exp_name in only if exp_name not in per_experiment)
        if missing:
            raise ValueError(f"Unknown experiment name(s) in --only: {', '.join(missing)}")

    return jobs, warnings, per_experiment


def build_runner_command(
    campaign: Dict[str, Any],
    payload: Dict[str, Any],
    save_dirpath: Path,
) -> List[str]:
    runtime = campaign["runtime"]
    common = dict(campaign["common_args"])
    common_args_override = payload.get("common_args_override")
    if common_args_override is not None:
        if not isinstance(common_args_override, dict):
            raise ValueError("payload.common_args_override must be an object")
        common.update(common_args_override)
    conda_env = str(runtime["conda_env"])

    runner_module = str(campaign.get("runner_module", "")).strip()
    if not runner_module:
        runner_script = str(campaign.get("runner_script", "")).strip()
        if not runner_script:
            raise ValueError("Campaign requires either 'runner_module' or 'runner_script'")
        if runner_script.endswith(".py"):
            runner_module = runner_script[:-3].replace("/", ".")
        else:
            runner_module = runner_script
    cmd = [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        conda_env,
        "env",
        f"PYTHONPATH={REPO_ROOT}{os.pathsep}{os.environ.get('PYTHONPATH', '')}"
        if os.environ.get("PYTHONPATH")
        else f"PYTHONPATH={REPO_ROOT}",
        "python",
        "-m",
        runner_module,
        "--in_model",
        str(common["in_model"]),
        "--in_weights",
        str(payload["weights_path"]),
        "--prior_weights",
        str(common["prior_weights"]),
        "--gameplay_length",
        str(int(common["gameplay_length"])),
        "--save_dirpath",
        str(save_dirpath),
        "--seed",
        str(int(payload["seed"])),
    ]

    if payload["prompt_type"] == "text":
        cmd.extend(["--custom_text_prompt", str(payload["prompt_value"])])
        custom_text_seed = payload.get("custom_text_seed")
        if custom_text_seed is not None:
            cmd.extend(["--custom_text_seed", str(int(custom_text_seed))])
        cmd.extend(["--text_cond_scale", str(common["text_cond_scale"])])
    elif payload["prompt_type"] == "visual":
        cmd.extend(["--visual_prompt", str(payload["prompt_value"])])
        cmd.extend(["--visual_cond_scale", str(common["visual_cond_scale"])])
    else:
        raise ValueError(f"Unknown prompt_type: {payload['prompt_type']}")

    if bool(common.get("save_additional_data", False)):
        cmd.append("--save_additional_data")

    fake_env_zip = common.get("fake_env_zip")
    if fake_env_zip:
        cmd.extend(["--fake_env_zip", str(fake_env_zip)])

    if bool(common.get("wipe_hidden_state_each_step", False)):
        cmd.append("--wipe_hidden_state_each_step")

    placement_specs = common.get("placement", [])
    if placement_specs:
        if not isinstance(placement_specs, list):
            raise ValueError("common_args.placement must be a list")
        for spec in placement_specs:
            cmd.extend(["--placement", str(spec)])

    extra_args = common.get("extra_args", [])
    if extra_args:
        if not isinstance(extra_args, list):
            raise ValueError("common_args.extra_args must be a list")
        cmd.extend(str(arg) for arg in extra_args)

    return cmd


def run_one_job(
    store: JobStore,
    campaign: Dict[str, Any],
    job_row: Dict[str, Any],
    worker_id: str,
    stop_ctrl: StopController,
    state_root: Path,
    episodes_root: Path,
    heartbeat_seconds: int,
    lease_timeout_seconds: int,
) -> None:
    payload = json.loads(str(job_row["payload_json"]))
    job_id = str(job_row["id"])
    missing_weights_retry_seconds = int(campaign["runtime"].get("missing_weights_retry_seconds", 900))
    failed_retry_seconds = int(campaign["runtime"].get("failed_retry_seconds", 300))
    failed_max_retries = int(campaign["runtime"].get("failed_max_retries", -1))
    failed_retry_count = int(job_row.get("failed_retry_count") or 0)

    def _retry_or_fail(reason: str, code: int) -> None:
        nonlocal failed_retry_count
        if failed_max_retries < 0 or failed_retry_count < failed_max_retries:
            next_retry = failed_retry_count + 1
            store.retry_running_failed_job(
                job_id=job_id,
                worker_id=worker_id,
                reason=reason,
                return_code=code,
                retry_delay_seconds=failed_retry_seconds,
            )
            print(
                f"[worker] warning {job_id}: {reason}; deferred retry in "
                f"{failed_retry_seconds}s (retry {next_retry}"
                + ("" if failed_max_retries < 0 else f"/{failed_max_retries}")
                + ").",
                flush=True,
            )
            failed_retry_count = next_retry
            return
        store.mark_failed(
            job_id=job_id,
            worker_id=worker_id,
            return_code=code,
            error_message=reason,
            log_file=str(log_path),
        )
        print(
            f"[worker] job failed {job_id}: {reason} "
            f"(retry limit reached: {failed_retry_count}/{failed_max_retries})",
            flush=True,
        )

    weights_path = Path(str(payload["weights_path"]))
    if not weights_path.exists():
        reason = f"missing-weights:{weights_path}"
        store.defer_running_job(
            job_id=job_id,
            worker_id=worker_id,
            reason=reason,
            retry_delay_seconds=missing_weights_retry_seconds,
        )
        print(
            f"[worker] warning {job_id}: missing weights {weights_path}; "
            f"deferred for retry in {missing_weights_retry_seconds}s.",
            flush=True,
        )
        return

    logs_dir = state_root / "logs"
    tmp_root = state_root / "tmp"
    logs_dir.mkdir(parents=True, exist_ok=True)
    tmp_root.mkdir(parents=True, exist_ok=True)
    episodes_root.mkdir(parents=True, exist_ok=True)

    job_tmp_dir = tmp_root / job_id
    if job_tmp_dir.exists():
        shutil.rmtree(job_tmp_dir)
    run_parent = job_tmp_dir / "runner_save_dir"
    run_parent.mkdir(parents=True, exist_ok=True)
    before_dirs = {p.name for p in run_parent.iterdir() if p.is_dir()}

    log_path = logs_dir / f"{job_id}.log"
    cmd = build_runner_command(campaign, payload, run_parent)

    interrupted = False
    return_code = 1

    with log_path.open("a", encoding="utf-8", buffering=1) as out:
        out.write(f"\n=== Job {job_id} started at {utc_now()} ===\n")
        out.write(f"Worker: {worker_id}\n")
        out.write(f"Experiment: {payload['experiment_name']}\n")
        out.write(f"Model: {payload['model_label']}\n")
        out.write(f"Prompt: [{payload['prompt_type']}] {payload['prompt_value']}\n")
        out.write(f"Seed: {payload['seed']}\n")
        out.write(f"Command: {' '.join(cmd)}\n\n")
        out.flush()

        # Ensure runner imports modules from this repository first (not another repo on PYTHONPATH).
        child_env = os.environ.copy()
        repo_root_str = str(REPO_ROOT)
        existing_pythonpath = child_env.get("PYTHONPATH", "")
        if existing_pythonpath:
            child_env["PYTHONPATH"] = f"{repo_root_str}:{existing_pythonpath}"
        else:
            child_env["PYTHONPATH"] = repo_root_str

        proc = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            env=child_env,
            stdout=out,
            stderr=subprocess.STDOUT,
            text=True,
        )

        next_hb = time.monotonic() + float(heartbeat_seconds)
        while True:
            polled = proc.poll()
            if polled is not None:
                return_code = int(polled)
                break

            if stop_ctrl.stop_requested:
                interrupted = True
                interrupt_process(proc)
                return_code = int(proc.returncode) if proc.returncode is not None else 130
                break

            now_mono = time.monotonic()
            if now_mono >= next_hb:
                store.refresh_lease(job_id, worker_id, lease_timeout_seconds)
                next_hb = now_mono + float(heartbeat_seconds)

            time.sleep(1.0)

        out.write(f"\nFinished at: {utc_now()}\n")
        out.write(f"Return code: {return_code}\n")

    if interrupted:
        store.requeue_running_job(job_id, worker_id, "interrupted-by-signal")
        shutil.rmtree(job_tmp_dir, ignore_errors=True)
        print(f"[worker] re-queued {job_id} after interruption.", flush=True)
        return

    if return_code != 0:
        _retry_or_fail(f"runner-exit:{return_code}", return_code)
        shutil.rmtree(job_tmp_dir, ignore_errors=True)
        return

    run_dir = detect_new_run_dir(run_parent, before_dirs)
    if run_dir is None:
        _retry_or_fail("runner produced no output directory", return_code)
        shutil.rmtree(job_tmp_dir, ignore_errors=True)
        return

    rendered_rel = str(payload["output_rel_template"]).format(
        dt=dt_token(),
        seed=int(payload["seed"]),
    )
    final_dir = ensure_unique_path(episodes_root / rendered_rel)
    final_dir.parent.mkdir(parents=True, exist_ok=True)

    try:
        shutil.move(str(run_dir), str(final_dir))
    except Exception as exc:
        _retry_or_fail(f"failed to move output directory: {exc}", return_code)
        shutil.rmtree(job_tmp_dir, ignore_errors=True)
        return

    store.mark_completed(
        job_id=job_id,
        worker_id=worker_id,
        return_code=return_code,
        final_output_dir=str(final_dir),
        log_file=str(log_path),
    )
    shutil.rmtree(job_tmp_dir, ignore_errors=True)
    print(f"[worker] completed {job_id} -> {final_dir}", flush=True)


def print_status(store: JobStore, show_running: int, show_failed: int) -> None:
    totals = store.totals()
    total = sum(totals.values())
    completed = totals[STATUS_COMPLETED]
    pending = totals[STATUS_PENDING]
    running = totals[STATUS_RUNNING]
    failed = totals[STATUS_FAILED]

    remaining_not_completed = total - completed
    remaining_queue = pending + running

    print("Episode Queue Status")
    print(f"  total episodes tracked: {total}")
    print(f"  completed episodes: {completed}")
    print(f"  remaining episodes (not completed): {remaining_not_completed}")
    print(f"  queue-active remaining (pending + running): {remaining_queue}")
    print(f"  pending: {pending}")
    print(f"  running: {running}")
    print(f"  failed: {failed}")

    by_exp = store.per_experiment_totals()
    if by_exp:
        print("\nPer-experiment")
        for row in by_exp:
            exp_total = int(row["total"])
            exp_completed = int(row[STATUS_COMPLETED])
            exp_remaining = exp_total - exp_completed
            print(
                f"  {row['experiment_name']}: total={exp_total}, completed={exp_completed}, "
                f"remaining={exp_remaining}, pending={row[STATUS_PENDING]}, "
                f"running={row[STATUS_RUNNING]}, failed={row[STATUS_FAILED]}"
            )

    if show_running > 0:
        running_rows = store.list_by_status(STATUS_RUNNING, limit=show_running)
        if running_rows:
            print(f"\nRunning jobs (up to {show_running})")
            for row in running_rows:
                print(
                    f"  {row['id']} | {row['experiment_name']} | {row['model_label']} | "
                    f"{row['prompt_type']}:{row['prompt_slug']} | seed={row['seed']} | worker={row['worker_id']}"
                )

    if show_failed > 0:
        failed_rows = store.list_by_status(STATUS_FAILED, limit=show_failed)
        if failed_rows:
            print(f"\nFailed jobs (up to {show_failed})")
            for row in failed_rows:
                print(
                    f"  {row['id']} | {row['experiment_name']} | {row['model_label']} | "
                    f"{row['prompt_type']}:{row['prompt_slug']} | seed={row['seed']} | "
                    f"error={row['error_message']}"
                )


def cmd_enqueue(args: argparse.Namespace, campaign: Dict[str, Any]) -> int:
    only = args.only if args.only else None
    jobs, warnings, per_exp = build_jobs_from_campaign(campaign, selected_experiments=only)

    print("Enqueue plan")
    print(f"  jobs discovered: {len(jobs)}")
    for exp_name in sorted(per_exp.keys()):
        print(f"  {exp_name}: {per_exp[exp_name]} jobs")

    if warnings:
        print("\nDiscovery warnings")
        for line in warnings:
            print(f"  - {line}")

    preview_n = max(0, int(args.preview))
    if preview_n > 0:
        print(f"\nPreview (first {preview_n})")
        for job in jobs[:preview_n]:
            rendered = str(job["output_rel_template"]).format(dt="YYYYMMDD-HHMMSS", seed=job["seed"])
            print(
                f"  {job['id']} | {job['experiment_name']} | {job['model_label']} | "
                f"{job['prompt_type']}:{job['prompt_slug']} | seed={job['seed']} | {rendered}"
            )

    if args.dry_run:
        print("\nDry run only. No rows inserted.")
        return 0

    db_path = resolve_path(str(campaign["queue_db_path"]))
    store = JobStore(db_path)
    inserted, existing = store.insert_jobs(jobs)
    print(f"\nQueue update: inserted={inserted}, existing={existing}")
    return 0


def cmd_worker(args: argparse.Namespace, campaign: Dict[str, Any]) -> int:
    db_path = resolve_path(str(campaign["queue_db_path"]))
    state_root = resolve_path(str(campaign["state_root"]))
    episodes_root = resolve_path(str(campaign["episodes_root"]))
    runtime = campaign["runtime"]

    store = JobStore(db_path)
    stop_ctrl = StopController()
    stop_ctrl.install()

    worker_id = str(args.worker_id or f"worker-{os.getpid()}")
    heartbeat_seconds = int(args.heartbeat_seconds or runtime["heartbeat_seconds"])
    lease_timeout_seconds = int(args.lease_timeout_seconds or runtime["lease_timeout_seconds"])
    failed_max_retries = int(runtime.get("failed_max_retries", -1))
    idle_sleep_seconds = int(args.idle_sleep_seconds or runtime["idle_sleep_seconds"])
    max_jobs = args.max_jobs if args.max_jobs is None else int(args.max_jobs)
    if max_jobs is not None and max_jobs < 0:
        raise ValueError("--max-jobs must be >= 0")

    print(
        f"[worker] starting worker_id={worker_id} "
        f"(heartbeat={heartbeat_seconds}s, lease={lease_timeout_seconds}s, "
        f"missing-retry={int(runtime.get('missing_weights_retry_seconds', 900))}s, "
        f"idle={idle_sleep_seconds}s)",
        flush=True,
    )
    print(f"[worker] queue db: {db_path}", flush=True)
    print(f"[worker] episodes root: {episodes_root}", flush=True)

    processed = 0
    while True:
        if max_jobs is not None and processed >= max_jobs:
            print(f"[worker] reached --max-jobs={max_jobs}; exiting.", flush=True)
            break

        requeued_failed = store.requeue_retryable_failed_jobs(max_retries=failed_max_retries)
        if requeued_failed > 0:
            print(f"[worker] re-queued retryable failed jobs: {requeued_failed}", flush=True)

        stale = store.requeue_stale_running()
        if stale > 0:
            print(f"[worker] re-queued stale running jobs: {stale}", flush=True)

        job = store.claim_next_job(worker_id=worker_id, lease_timeout_seconds=lease_timeout_seconds)
        if job is None:
            if stop_ctrl.stop_requested:
                print("[worker] stop requested while idle; exiting.", flush=True)
                break
            time.sleep(float(idle_sleep_seconds))
            continue

        print(
            f"[worker] claimed {job['id']} ({job['experiment_name']} | {job['model_label']} | "
            f"{job['prompt_type']}:{job['prompt_slug']} | seed={job['seed']})",
            flush=True,
        )

        run_one_job(
            store=store,
            campaign=campaign,
            job_row=job,
            worker_id=worker_id,
            stop_ctrl=stop_ctrl,
            state_root=state_root,
            episodes_root=episodes_root,
            heartbeat_seconds=heartbeat_seconds,
            lease_timeout_seconds=lease_timeout_seconds,
        )
        processed += 1

        if stop_ctrl.stop_requested:
            print("[worker] stop requested after finishing current job; exiting.", flush=True)
            break

    return 0


def cmd_status(args: argparse.Namespace, campaign: Dict[str, Any]) -> int:
    db_path = resolve_path(str(campaign["queue_db_path"]))
    store = JobStore(db_path)
    print_status(
        store=store,
        show_running=int(args.show_running),
        show_failed=int(args.show_failed),
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Episode queue orchestrator")
    sub = parser.add_subparsers(dest="command", required=True)

    enqueue = sub.add_parser("enqueue", help="Build jobs from campaign and insert pending jobs")
    enqueue.add_argument("--campaign", type=str, default=str(DEFAULT_CAMPAIGN_PATH))
    enqueue.add_argument("--dry-run", action="store_true", default=False)
    enqueue.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Optional list of experiment names to enqueue",
    )
    enqueue.add_argument(
        "--preview",
        type=int,
        default=12,
        help="How many jobs to print in preview",
    )

    worker = sub.add_parser("worker", help="Run one worker process")
    worker.add_argument("--campaign", type=str, default=str(DEFAULT_CAMPAIGN_PATH))
    worker.add_argument("--worker-id", type=str, default=None)
    worker.add_argument("--max-jobs", type=int, default=None)
    worker.add_argument("--heartbeat-seconds", type=int, default=None)
    worker.add_argument("--lease-timeout-seconds", type=int, default=None)
    worker.add_argument("--idle-sleep-seconds", type=int, default=None)

    status = sub.add_parser("status", help="Show queue progress summary")
    status.add_argument("--campaign", type=str, default=str(DEFAULT_CAMPAIGN_PATH))
    status.add_argument("--show-running", type=int, default=10)
    status.add_argument("--show-failed", type=int, default=10)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    campaign_path = resolve_path(str(args.campaign))
    campaign = apply_campaign_defaults(load_json(campaign_path))

    if args.command == "enqueue":
        return cmd_enqueue(args, campaign)
    if args.command == "worker":
        return cmd_worker(args, campaign)
    if args.command == "status":
        return cmd_status(args, campaign)

    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
