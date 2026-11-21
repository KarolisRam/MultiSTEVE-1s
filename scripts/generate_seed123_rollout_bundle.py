#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
from nbt.nbt import NBTFile


REPO = Path(os.environ.get('MULTISTEVE_REPO_ROOT', Path(__file__).resolve().parents[1])).resolve()
PYTHON_EXE = os.environ.get('PYTHON', sys.executable)
RUN_MODULE = os.environ.get(
    'MULTISTEVE_RUN_MODULE',
    'steve1.run_agent.run_agent_with_logging_teleport',
)
BUILD_HTML_MODULE = os.environ.get(
    'MULTISTEVE_WORLD_VIEWER_BUILDER_MODULE',
    'tools.world_viewer.build_interactive_isometric_html',
)
THREE_JS_FILE = (
    Path(os.environ['MULTISTEVE_THREE_JS_FILE'])
    if os.environ.get('MULTISTEVE_THREE_JS_FILE')
    else None
)
TEMPLATE_HTML = (
    Path(os.environ['MULTISTEVE_WORLD_VIEWER_TEMPLATE'])
    if os.environ.get('MULTISTEVE_WORLD_VIEWER_TEMPLATE')
    else None
)
SAVES_DIR = Path(os.environ.get('MINERL_SAVES_DIR', 'data/minerl_saves'))

SEEDS = [1, 2, 3]
FPS = 20
GAMEPLAY_SECONDS = 20
GAMEPLAY_LENGTH = FPS * GAMEPLAY_SECONDS
PROMPT = 'chop tree'
MAX_RUN_ATTEMPTS = 3


def _run(cmd: list[str], cwd: Path = REPO) -> None:
    print('RUN:', ' '.join(cmd))
    env = os.environ.copy()
    repo_path = str(REPO)
    existing_pythonpath = env.get('PYTHONPATH', '').strip()
    if existing_pythonpath:
        env['PYTHONPATH'] = repo_path + os.pathsep + existing_pythonpath
    else:
        env['PYTHONPATH'] = repo_path
    subprocess.run(cmd, cwd=str(cwd), check=True, env=env)


def _cleanup_minerl_processes() -> None:
    patterns = [
        'mcprec-6.13.jar',
        'MCP-Reborn/launchClient.sh -port',
        'minerl.utils.process_watcher',
    ]
    for pat in patterns:
        subprocess.run(['pkill', '-f', pat], check=False)


def _get_tag(compound, name: str):
    for tag in compound.tags:
        if tag.name == name:
            return tag
    return None


def read_level_meta(world_dir: Path) -> dict[str, Any] | None:
    level = world_dir / 'level.dat'
    if not level.exists():
        return None
    try:
        nbt = NBTFile(str(level), 'rb')
        data = _get_tag(nbt, 'Data')
        if data is None:
            return None
        wgs = _get_tag(data, 'WorldGenSettings')
        ver = _get_tag(data, 'Version')
        out = {
            'world_dir': str(world_dir),
            'world_name': world_dir.name,
            'level_name': getattr(_get_tag(data, 'LevelName'), 'value', None),
            'seed': getattr(_get_tag(wgs, 'seed'), 'value', None) if wgs is not None else None,
            'spawn_x': getattr(_get_tag(data, 'SpawnX'), 'value', None),
            'spawn_y': getattr(_get_tag(data, 'SpawnY'), 'value', None),
            'spawn_z': getattr(_get_tag(data, 'SpawnZ'), 'value', None),
            'data_version': getattr(_get_tag(data, 'DataVersion'), 'value', None),
            'mc_version': getattr(_get_tag(ver, 'Name'), 'value', None) if ver is not None else None,
            'level_dat_mtime': level.stat().st_mtime,
        }
        return out
    except Exception:
        return None


def first_pose(location_stats_csv: Path) -> dict[str, float]:
    with location_stats_csv.open('r', newline='') as f:
        reader = csv.DictReader(f)
        row = next(reader)
    return {
        'x': float(row['xpos']),
        'y': float(row['ypos']),
        'z': float(row['zpos']),
        'yaw': float(row.get('yaw', 0.0)),
        'pitch': float(row.get('pitch', 0.0)),
    }


def pick_world_for_seed(seed: int, start_ts: float, pose: dict[str, float]) -> dict[str, Any]:
    worlds: list[dict[str, Any]] = []
    for p in SAVES_DIR.iterdir():
        if not p.is_dir():
            continue
        meta = read_level_meta(p)
        if meta is None:
            continue
        if meta.get('seed') != seed:
            continue
        sx, sz = meta.get('spawn_x'), meta.get('spawn_z')
        if sx is None or sz is None:
            dist = 1e18
        else:
            dx = float(sx) - pose['x']
            dz = float(sz) - pose['z']
            dist = (dx * dx + dz * dz) ** 0.5
        meta['spawn_dist_to_pose'] = dist
        worlds.append(meta)

    if not worlds:
        raise RuntimeError(f'No MineRL world found with seed={seed}')

    recent = [w for w in worlds if float(w['level_dat_mtime']) >= (start_ts - 2.0)]
    pool = recent if recent else worlds
    pool.sort(key=lambda w: (float(w['level_dat_mtime']), -float(w['spawn_dist_to_pose'])), reverse=True)
    return pool[0]


def zip_world(world_dir: Path, zip_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, mode='w', compression=zipfile.ZIP_DEFLATED) as zf:
        for fp in sorted(world_dir.rglob('*')):
            if fp.is_dir():
                continue
            arcname = f'{world_dir.name}/{fp.relative_to(world_dir)}'
            zf.write(fp, arcname)


def extract_first_frame(video_path: Path, out_png: Path, run_dir: Path) -> None:
    fullres_first = run_dir / 'first_frame_fullres.png'
    if fullres_first.exists():
        out_png.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(fullres_first, out_png)
        return

    cap = cv2.VideoCapture(str(video_path))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f'Failed to read first frame from {video_path}')
    out_png.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_png), frame)


def extract_const_line(text: str, const_name: str) -> str:
    pattern = rf"  const {re.escape(const_name)} = .*?;\n"
    m = re.search(pattern, text, flags=re.S)
    if not m:
        raise RuntimeError(f'Could not extract const {const_name}')
    return m.group(0)


def build_v22_style_html(world_dir: Path, location_stats: Path, out_html: Path, tmp_base_html: Path) -> None:
    cmd = [
        PYTHON_EXE,
        '-m',
        BUILD_HTML_MODULE,
        '--world', str(world_dir),
        '--spawn-location-stats', str(location_stats),
        '--trajectory-location-stats', str(location_stats),
        '--out-html', str(tmp_base_html),
        '--radius', '42',
        '--layers-below', '20',
        '--layers-above', '20',
        '--hide-plants',
        '--trajectory-step', '1',
        '--offline-bundle',
    ]
    if THREE_JS_FILE is not None:
        cmd.extend(['--three-js-file', str(THREE_JS_FILE)])
    _run(cmd)

    if TEMPLATE_HTML is None:
        out_html.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(tmp_base_html, out_html)
        return

    if not TEMPLATE_HTML.exists():
        raise FileNotFoundError(TEMPLATE_HTML)

    base_text = tmp_base_html.read_text(encoding='utf-8')
    template_text = TEMPLATE_HTML.read_text(encoding='utf-8')
    for name in ('META', 'VOXELS', 'TRAJECTORIES'):
        repl = extract_const_line(base_text, name)
        template_text = re.sub(rf"  const {re.escape(name)} = .*?;\n", repl, template_text, count=1, flags=re.S)

    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(template_text, encoding='utf-8')


def find_new_run_dir(runs_dir: Path, existing: set[str], start_ts: float) -> Path:
    current = [p for p in runs_dir.iterdir() if p.is_dir()]
    new_dirs = [p for p in current if p.name not in existing]
    if len(new_dirs) == 1:
        return new_dirs[0]
    if new_dirs:
        new_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return new_dirs[0]

    # Fallback if timestamp collision happened.
    candidates = [p for p in current if p.stat().st_mtime >= start_ts - 2.0]
    if not candidates:
        raise RuntimeError('Could not identify new rollout directory')
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--seeds',
        nargs='+',
        type=int,
        default=SEEDS,
        help='Seeds to run, e.g. --seeds 1 2 3',
    )
    parser.add_argument(
        '--output-prefix',
        type=str,
        default=None,
        help='Optional output folder prefix under data/. Default: seed<seeds>_rollouts',
    )
    parser.add_argument(
        '--max-run-attempts',
        type=int,
        default=MAX_RUN_ATTEMPTS,
        help='Retry attempts per seed when MineRL run fails.',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_seeds = args.seeds
    if not run_seeds:
        raise RuntimeError('At least one seed is required')

    dt = datetime.now().strftime('%Y%m%d-%H%M%S')
    default_prefix = f"seed{''.join(str(s) for s in run_seeds)}_rollouts"
    out_prefix = args.output_prefix if args.output_prefix else default_prefix
    out_root = REPO / 'data' / f'{out_prefix}_{dt}'
    worlds_dir = out_root / '01_worlds'
    frames_dir = out_root / '02_first_screenshots'
    videos_dir = out_root / '03_rollout_videos'
    html_dir = out_root / '04_3d_html'
    runs_dir = out_root / '_tmp_runs'
    tmp_html_dir = out_root / '_tmp_base_html'

    for d in (worlds_dir, frames_dir, videos_dir, html_dir, runs_dir, tmp_html_dir):
        d.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        'created_at': dt,
        'gameplay_seconds': GAMEPLAY_SECONDS,
        'gameplay_length_steps': GAMEPLAY_LENGTH,
        'fps': FPS,
        'prompt': PROMPT,
        'seeds_requested': run_seeds,
        'seeds': [],
    }

    for seed_idx, seed in enumerate(run_seeds):
        print(f'\n===== SEED {seed} =====')
        run_dir: Path | None = None
        video_src: Path | None = None
        loc_stats: Path | None = None
        start_ts: float | None = None
        last_exc: Exception | None = None

        for attempt in range(1, args.max_run_attempts + 1):
            existing_runs = {p.name for p in runs_dir.iterdir() if p.is_dir()}
            start_ts = time.time()

            cmd = [
                PYTHON_EXE,
                '-m',
                RUN_MODULE,
                '--in_model', 'data/weights/vpt/2x.model',
                '--in_weights', 'data/weights/steve1/steve1.weights',
                '--prior_weights', 'data/weights/steve1/steve1_prior.pt',
                '--text_cond_scale', '6.0',
                '--visual_cond_scale', '7.0',
                '--gameplay_length', str(GAMEPLAY_LENGTH),
                '--save_dirpath', str(runs_dir),
                '--custom_text_prompt', PROMPT,
                '--custom_text_seed', str(seed),
                '--seed', str(seed),
                '--placement', 'side=left,villager=true',
                '--placement', 'side=right,villager=false',
                '--teleport_to_spawn_every', '0',
                '--teleport_feedback_mode', 'gamerule_strict',
                '--suppress_attack_actions',
            ]
            if seed_idx == 0:
                cmd.append('--dummy_reset_before_seed')

            try:
                print(f'SEED {seed} attempt {attempt}/{args.max_run_attempts}')
                _run(cmd)
                run_dir = find_new_run_dir(runs_dir, existing_runs, start_ts)
                video_src = run_dir / 'video.mp4'
                loc_stats = run_dir / 'location_stats.log'
                if not video_src.exists() or not loc_stats.exists():
                    raise RuntimeError(f'Run dir missing expected files: {run_dir}')
                break
            except Exception as exc:
                last_exc = exc
                print(f'WARN: seed {seed} attempt {attempt} failed: {exc}')
                _cleanup_minerl_processes()
                time.sleep(3.0)
                run_dir = None
                video_src = None
                loc_stats = None

        if run_dir is None or video_src is None or loc_stats is None or start_ts is None:
            raise RuntimeError(
                f'Failed seed {seed} after {args.max_run_attempts} attempts'
            ) from last_exc

        pose = first_pose(loc_stats)
        world_meta = pick_world_for_seed(seed, start_ts, pose)
        world_dir = Path(world_meta['world_dir'])

        world_zip = worlds_dir / f'seed_{seed}.zip'
        frame_png = frames_dir / f'seed_{seed}_first_frame.png'
        video_dst = videos_dir / f'seed_{seed}.mp4'
        base_html = tmp_html_dir / f'seed_{seed}_base.html'
        html_dst = html_dir / f'seed_{seed}.html'

        zip_world(world_dir, world_zip)
        extract_first_frame(video_src, frame_png, run_dir)
        video_dst.write_bytes(video_src.read_bytes())
        build_v22_style_html(world_dir, loc_stats, html_dst, base_html)

        summary['seeds'].append({
            'seed': seed,
            'run_dir': str(run_dir),
            'world_dir': str(world_dir),
            'world_name': world_meta['world_name'],
            'world_level_name': world_meta.get('level_name'),
            'world_spawn': {
                'x': world_meta.get('spawn_x'),
                'y': world_meta.get('spawn_y'),
                'z': world_meta.get('spawn_z'),
            },
            'world_seed': world_meta.get('seed'),
            'world_version': {
                'mc_version': world_meta.get('mc_version'),
                'data_version': world_meta.get('data_version'),
            },
            'episode_start_pose': pose,
            'artifacts': {
                'world_zip': str(world_zip),
                'first_frame_png': str(frame_png),
                'video_mp4': str(video_dst),
                'html_3d': str(html_dst),
            },
        })

    summary_path = out_root / 'summary.json'
    summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')

    print('\nDONE')
    print(f'Output root: {out_root}')
    print(f'Summary: {summary_path}')


if __name__ == '__main__':
    main()
