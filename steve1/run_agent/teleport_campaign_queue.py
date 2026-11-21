from __future__ import annotations

import argparse
import csv
import fcntl
import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from steve1.run_agent.teleport_campaign_common import (
    CONDITION_BY_KEY,
    CONDITION_ORDER,
    CONDITIONS,
    DEFAULT_RUNNER_ARGS,
    condition_manifest_path,
    condition_runs_dir,
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


@dataclass
class ActiveJob:
    row_key: tuple[str, str]
    condition: str
    model_id: str
    staging_save_base: Path
    stdout_log: Path
    process: subprocess.Popen


stop_requested = False
hard_stop_requested = False


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _read_status(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing rollout status: {path}")
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def _write_status(path: Path, rows: list[dict[str, object]]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=STATUS_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in STATUS_FIELDS})
    os.replace(tmp, path)


def _status_sort(row: dict[str, str]) -> tuple[tuple[int, str], int]:
    return (model_sort_key(row.get("model_id", "")), CONDITION_ORDER.get(row.get("condition", ""), 10**6))


def _row_key(row: dict[str, str]) -> tuple[str, str]:
    return (row["condition"], row["model_id"])


def _update_row(path: Path, key: tuple[str, str], updates: dict[str, object]) -> None:
    rows = _read_status(path)
    found = False
    for row in rows:
        if _row_key(row) == key:
            row.update({k: str(v) for k, v in updates.items()})
            row["updated_at"] = _now()
            found = True
            break
    if not found:
        raise KeyError(key)
    _write_status(path, rows)


def _reset_stale_running(status_path: Path) -> int:
    rows = _read_status(status_path)
    changed = 0
    for row in rows:
        if row.get("status") == "running":
            run_dir = row.get("run_dir", "")
            if run_dir and Path(run_dir).exists():
                row["status"] = "done"
                row["notes"] = "recovered completed final run after interruption"
            else:
                row["status"] = "pending"
                row["notes"] = "reset stale running row to pending at queue startup"
            row["staging_dir"] = ""
            row["started_at"] = ""
            row["updated_at"] = _now()
            changed += 1
    if changed:
        _write_status(status_path, rows)
    return changed


def _next_pending(status_path: Path) -> dict[str, str] | None:
    rows = sorted(_read_status(status_path), key=_status_sort)
    for row in rows:
        if row.get("status") == "pending":
            return row
    return None


def _build_command(repo_root: Path, row: dict[str, str], staging_save_base: Path) -> list[str]:
    condition = CONDITION_BY_KEY[row["condition"]]
    cmd = [
        sys.executable,
        "-m",
        "steve1.run_agent.run_agent_with_logging_teleport",
        "--in_model",
        DEFAULT_RUNNER_ARGS["in_model"],
        "--in_weights",
        row["weight_path"],
        "--prior_weights",
        DEFAULT_RUNNER_ARGS["prior_weights"],
        "--text_cond_scale",
        str(DEFAULT_RUNNER_ARGS["text_cond_scale"]),
        "--visual_cond_scale",
        str(DEFAULT_RUNNER_ARGS["visual_cond_scale"]),
        "--gameplay_length",
        str(DEFAULT_RUNNER_ARGS["gameplay_length"]),
        "--save_dirpath",
        str(staging_save_base),
        "--custom_text_prompt",
        DEFAULT_RUNNER_ARGS["prompt"],
        "--custom_text_seed",
        str(condition.env_seed),
        "--seed",
        str(condition.env_seed),
        "--teleport_to_spawn_every",
        str(DEFAULT_RUNNER_ARGS["teleport_every"]),
        "--creative_command_wait_steps",
        str(DEFAULT_RUNNER_ARGS["creative_command_wait_steps"]),
        "--teleport_wait_steps",
        str(DEFAULT_RUNNER_ARGS["teleport_wait_steps"]),
        "--teleport_feedback_mode",
        DEFAULT_RUNNER_ARGS["teleport_feedback_mode"],
    ]
    for placement in condition.placement_args:
        cmd.extend(["--placement", placement])
    if DEFAULT_RUNNER_ARGS["suppress_attack_actions"]:
        cmd.append("--suppress_attack_actions")
    return cmd


def _line_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("rb") as f:
        return sum(1 for _ in f)


def _validate_run_dir(run_dir: Path) -> tuple[bool, str]:
    gameplay_length = int(DEFAULT_RUNNER_ARGS["gameplay_length"])
    teleport_every = int(DEFAULT_RUNNER_ARGS["teleport_every"])
    expected_actions = gameplay_length + 1
    expected_locations = gameplay_length + 2
    teleport_count = (gameplay_length - 1) // teleport_every if teleport_every > 0 else 0
    # The runner writes the CSV header only when at least one teleport occurs.
    expected_teleports = teleport_count + 1 if teleport_count else 0
    checks = {
        "actions.log": _line_count(run_dir / "actions.log"),
        "location_stats.log": _line_count(run_dir / "location_stats.log"),
        "teleport_events.log": _line_count(run_dir / "teleport_events.log"),
        "video.mp4": 1 if (run_dir / "video.mp4").exists() else 0,
    }
    if checks["actions.log"] < expected_actions:
        return False, f"actions.log too short: {checks['actions.log']} < {expected_actions}"
    if checks["location_stats.log"] < expected_locations:
        return False, f"location_stats.log too short: {checks['location_stats.log']} < {expected_locations}"
    if checks["teleport_events.log"] != expected_teleports:
        return False, f"teleport_events.log line count {checks['teleport_events.log']} != {expected_teleports}"
    if checks["video.mp4"] != 1:
        return False, "missing video.mp4"
    return True, "ok"


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


def _append_or_replace_manifest(manifest_path: Path, row: dict[str, object]) -> None:
    rows: list[dict[str, str]] = []
    if manifest_path.exists():
        with manifest_path.open("r", newline="") as f:
            rows = list(csv.DictReader(f))
    rows = [r for r in rows if r.get("model_id") != row["model_id"]]
    rows.append({k: str(v) for k, v in row.items()})
    rows.sort(key=lambda r: model_sort_key(r.get("model_id", "")))
    tmp = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    with tmp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for r in rows:
            writer.writerow({field: r.get(field, "") for field in MANIFEST_FIELDS})
    os.replace(tmp, manifest_path)


def _handle_signal(signum, frame) -> None:  # noqa: ANN001
    global stop_requested, hard_stop_requested
    if stop_requested:
        hard_stop_requested = True
        print(f"received signal {signum}; hard stop requested", flush=True)
    else:
        stop_requested = True
        print(f"received signal {signum}; will finish active rollouts and stop", flush=True)


def _start_job(repo_root: Path, campaign_root: Path, status_path: Path, worker_id: int, row: dict[str, str]) -> ActiveJob:
    key = _row_key(row)
    attempts = int(row.get("attempts") or 0) + 1
    staging_save_base = campaign_root / "staging" / f"worker_{worker_id}" / row["condition"] / row["model_id"]
    staging_save_base.mkdir(parents=True, exist_ok=True)
    log_dir = campaign_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_log = log_dir / f"worker_{worker_id}_{row['condition']}_{row['model_id']}_attempt{attempts}.log"

    _update_row(
        status_path,
        key,
        {
            "status": "running",
            "staging_dir": staging_save_base,
            "attempts": attempts,
            "started_at": _now(),
            "last_return_code": "",
            "notes": f"running on worker {worker_id}",
        },
    )

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{repo_root}{os.pathsep}{env['PYTHONPATH']}" if env.get("PYTHONPATH") else str(repo_root)
    cmd = _build_command(repo_root, row, staging_save_base)
    with stdout_log.open("wb") as log_f:
        log_f.write(("Command: " + " ".join(cmd) + "\n").encode("utf-8"))
    log_f = stdout_log.open("ab")
    process = subprocess.Popen(cmd, cwd=repo_root, env=env, stdout=log_f, stderr=subprocess.STDOUT)
    process._steve_log_handle = log_f  # type: ignore[attr-defined]
    print(f"[start] worker={worker_id} {row['condition']} {row['model_id']} attempt={attempts}", flush=True)
    return ActiveJob(key, row["condition"], row["model_id"], staging_save_base, stdout_log, process)


def _finish_job(campaign_root: Path, status_path: Path, job: ActiveJob) -> None:
    rc = job.process.returncode
    log_handle = getattr(job.process, "_steve_log_handle", None)
    if log_handle is not None:
        log_handle.close()

    rows_by_key = {_row_key(row): row for row in _read_status(status_path)}
    row = rows_by_key[job.row_key]
    latest_runs = sorted([p for p in job.staging_save_base.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True)
    latest_run = latest_runs[0] if latest_runs else None
    if rc == 0 and latest_run is not None:
        valid, note = _validate_run_dir(latest_run)
    else:
        valid = False
        note = f"return_code={rc}; no run dir" if latest_run is None else f"return_code={rc}"

    if rc == 0 and latest_run is not None and valid:
        final_dir = condition_runs_dir(campaign_root, job.condition) / job.model_id / latest_run.name
        _copytree_atomic(latest_run, final_dir)
        manifest_path = condition_manifest_path(campaign_root, job.condition)
        _append_or_replace_manifest(
            manifest_path,
            {
                "model_id": job.model_id,
                "weight_path": row["weight_path"],
                "save_base": final_dir.parent,
                "run_dir": final_dir,
                "status": "ok",
                "return_code": "0",
            },
        )
        _update_row(
            status_path,
            job.row_key,
            {
                "status": "done",
                "source": "queue_rollout",
                "run_dir": final_dir,
                "staging_dir": "",
                "last_return_code": "0",
                "finished_at": _now(),
                "notes": "completed and copied into canonical condition folder",
            },
        )
        print(f"[done] {job.condition} {job.model_id}: {final_dir}", flush=True)
    else:
        _update_row(
            status_path,
            job.row_key,
            {
                "status": "failed",
                "last_return_code": rc if rc is not None else "",
                "finished_at": _now(),
                "notes": note,
            },
        )
        print(f"[fail] {job.condition} {job.model_id}: {note}; log={job.stdout_log}", flush=True)


def run_queue(repo_root: Path, campaign_root: Path, max_parallel: int, retry_failed: bool) -> int:
    status_path = campaign_root / "rollout_status.csv"
    lock_path = campaign_root / "queue.lock"
    lock_f = lock_path.open("w")
    try:
        fcntl.flock(lock_f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(f"queue already running for {campaign_root} (lock: {lock_path})", file=sys.stderr)
        return 2

    (campaign_root / "queue_controller.pid").write_text(f"{os.getpid()}\n")

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    reset_count = _reset_stale_running(status_path)
    if reset_count:
        print(f"reset stale running rows: {reset_count}", flush=True)

    if retry_failed:
        rows = _read_status(status_path)
        changed = 0
        for row in rows:
            if row.get("status") == "failed":
                row["status"] = "pending"
                row["notes"] = "failed row reset to pending by --retry_failed"
                row["updated_at"] = _now()
                changed += 1
        if changed:
            _write_status(status_path, rows)
            print(f"reset failed rows: {changed}", flush=True)

    active: dict[int, ActiveJob] = {}
    worker_ids = list(range(max_parallel))

    while True:
        for worker_id, job in list(active.items()):
            if job.process.poll() is not None:
                _finish_job(campaign_root, status_path, job)
                del active[worker_id]

        if hard_stop_requested:
            for job in active.values():
                job.process.terminate()
            return 130

        if not stop_requested:
            while len(active) < max_parallel:
                row = _next_pending(status_path)
                if row is None:
                    break
                worker_id = next(w for w in worker_ids if w not in active)
                active[worker_id] = _start_job(repo_root, campaign_root, status_path, worker_id, row)

        if not active:
            pending = [row for row in _read_status(status_path) if row.get("status") == "pending"]
            if pending and not stop_requested:
                time.sleep(2)
                continue
            break

        time.sleep(5)

    rows = _read_status(status_path)
    counts: dict[str, int] = {}
    for row in rows:
        counts[row.get("status", "")] = counts.get(row.get("status", ""), 0) + 1
    print("queue stopped; status counts: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())), flush=True)
    pid_path = campaign_root / "queue_controller.pid"
    if pid_path.exists() and pid_path.read_text().strip() == str(os.getpid()):
        pid_path.unlink()
    return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign_root", required=True)
    parser.add_argument("--repo_root", default=".")
    parser.add_argument("--max_parallel", type=int, default=2)
    parser.add_argument("--retry_failed", action="store_true")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    campaign_root = Path(args.campaign_root)
    if not campaign_root.is_absolute():
        campaign_root = repo_root / campaign_root
    campaign_root = campaign_root.resolve()
    if args.max_parallel < 1:
        raise ValueError("--max_parallel must be >= 1")
    raise SystemExit(run_queue(repo_root, campaign_root, args.max_parallel, args.retry_failed))


if __name__ == "__main__":
    main()
