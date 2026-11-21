#!/usr/bin/env python3
import argparse
import csv
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path

import anvil
import cv2
import numpy as np


IGNORE_AIR_IDS = {'air', 'cave_air', 'void_air'}
PLANT_IDS = {
    'grass',
    'short_grass',
    'tall_grass',
    'fern',
    'large_fern',
    'dandelion',
    'poppy',
    'blue_orchid',
    'allium',
    'azure_bluet',
    'red_tulip',
    'orange_tulip',
    'white_tulip',
    'pink_tulip',
    'oxeye_daisy',
    'cornflower',
    'lily_of_the_valley',
    'sunflower',
    'lilac',
    'rose_bush',
    'peony',
}

# Matches existing top-down palette so colors stay consistent across views.
COLOR_MAP = {
    'grass_block': (95, 159, 53),
    'dirt': (134, 96, 67),
    'coarse_dirt': (121, 85, 58),
    'podzol': (99, 74, 52),
    'sand': (219, 211, 160),
    'gravel': (134, 134, 134),
    'stone': (124, 124, 124),
    'cobblestone': (115, 115, 115),
    'water': (64, 100, 255),
    'seagrass': (45, 120, 80),
    'tall_seagrass': (45, 120, 80),
    'oak_log': (102, 81, 51),
    'birch_log': (216, 206, 171),
    'oak_leaves': (52, 90, 42),
    'birch_leaves': (78, 116, 54),
    'spruce_leaves': (45, 76, 39),
    'short_grass': (106, 173, 60),
    'grass': (106, 173, 60),
    'tall_grass': (116, 183, 69),
    'fern': (88, 141, 52),
    'dandelion': (230, 210, 48),
    'poppy': (190, 45, 45),
    'torch': (245, 198, 74),
    'oak_planks': (163, 130, 79),
    'birch_planks': (197, 184, 136),
    'crafting_table': (150, 122, 74),
    'netherrack': (130, 70, 70),
    'magma_block': (200, 110, 40),
    'stone_slab': (140, 140, 140),
    'stone_brick_stairs': (145, 145, 145),
    'barrier': (255, 80, 80),
}


@dataclass
class Pose:
    x: float
    y: float
    z: float
    yaw: float


@dataclass
class SurfaceBlock:
    x: int
    y: int
    z: int
    block_id: str


class WorldSampler:
    def __init__(self, world_dir: Path):
        self.world_dir = world_dir
        self.region_dir = world_dir / 'region'
        self.region_cache: dict[tuple[int, int], anvil.Region | None] = {}
        self.chunk_cache: dict[tuple[int, int], anvil.Chunk | None] = {}

    def _get_region(self, rx: int, rz: int):
        key = (rx, rz)
        if key in self.region_cache:
            return self.region_cache[key]
        rf = self.region_dir / f'r.{rx}.{rz}.mca'
        if not rf.exists():
            self.region_cache[key] = None
            return None
        self.region_cache[key] = anvil.Region.from_file(str(rf))
        return self.region_cache[key]

    def _get_chunk(self, cx: int, cz: int):
        key = (cx, cz)
        if key in self.chunk_cache:
            return self.chunk_cache[key]
        region = self._get_region(cx // 32, cz // 32)
        if region is None:
            self.chunk_cache[key] = None
            return None
        try:
            chunk = anvil.Chunk.from_region(region, cx, cz)
        except Exception:
            chunk = None
        self.chunk_cache[key] = chunk
        return chunk

    def top_block(self, x: int, z: int, max_y: int) -> tuple[str, int]:
        chunk = self._get_chunk(x // 16, z // 16)
        if chunk is None:
            return 'air', 0
        lx, lz = x % 16, z % 16
        for y in range(max_y, -1, -1):
            try:
                bid = chunk.get_block(lx, y, lz).id
            except Exception:
                continue
            if bid not in IGNORE_AIR_IDS:
                return bid, y
        return 'air', 0

    def block_id(self, x: int, y: int, z: int) -> str:
        chunk = self._get_chunk(x // 16, z // 16)
        if chunk is None:
            return 'air'
        lx, lz = x % 16, z % 16
        try:
            return chunk.get_block(lx, y, lz).id
        except Exception:
            return 'air'


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Render isometric block-color view from world data, optionally with trajectory.')
    p.add_argument('--world', required=True)
    p.add_argument('--spawn-location-stats', required=True, help='CSV with at least xpos,ypos,zpos,yaw in first row')
    p.add_argument('--trajectory-location-stats', default=None, help='Optional rollout location_stats.log to overlay trajectory')
    p.add_argument('--out-blocks', required=True)
    p.add_argument('--out-with-trajectory', default=None)
    p.add_argument('--radius', type=int, default=40)
    p.add_argument('--layers-below', type=int, default=20,
                   help='Number of layers below spawn Y to render as voxels')
    p.add_argument('--layers-above', type=int, default=20,
                   help='Number of layers above spawn Y to render as voxels')
    p.add_argument('--hide-plants', action='store_true',
                   help='Skip flowers and grass-like decorative plants in rendering')
    p.add_argument('--max-y', type=int, default=255)
    p.add_argument('--width', type=int, default=1400)
    p.add_argument('--height', type=int, default=1000)
    p.add_argument('--margin', type=int, default=50)
    p.add_argument('--elevation-deg', type=float, default=35.0)
    p.add_argument('--behind-side-mix', type=float, default=0.85, help='Diagonal offset from pure behind view')
    p.add_argument('--behind-side', choices=['right', 'left'], default='right')
    p.add_argument('--trajectory-step', type=int, default=2)
    p.add_argument('--trajectory-clip-radius', type=float, default=None,
                   help='If set, only draw trajectory points within this x/z distance from spawn. '
                        'Default uses --radius.')
    return p.parse_args()


def read_first_pose(path: Path) -> Pose:
    with path.open('r', newline='') as f:
        reader = csv.DictReader(f)
        first = next(reader)
    return Pose(
        x=float(first['xpos']),
        y=float(first['ypos']),
        z=float(first['zpos']),
        yaw=float(first.get('yaw', 0.0)),
    )


def read_trajectory(path: Path, step: int) -> list[Pose]:
    out: list[Pose] = []
    with path.open('r', newline='') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i % max(1, step) != 0:
                continue
            out.append(
                Pose(
                    x=float(row['xpos']),
                    y=float(row['ypos']),
                    z=float(row['zpos']),
                    yaw=float(row.get('yaw', 0.0)),
                )
            )
    return out


def clip_trajectory_to_radius(points: list[Pose], spawn: Pose, radius: float) -> list[Pose]:
    clipped = [p for p in points if abs(p.x - spawn.x) <= radius and abs(p.z - spawn.z) <= radius]
    return clipped


def color_for_block(block_id: str) -> tuple[int, int, int]:
    if block_id in COLOR_MAP:
        return COLOR_MAP[block_id]
    h = hashlib.md5(block_id.encode('utf-8')).digest()
    return (70 + (h[0] % 120), 70 + (h[1] % 120), 70 + (h[2] % 120))


def collect_volume_blocks(
    world_dir: Path,
    spawn: Pose,
    radius: int,
    layers_below: int,
    layers_above: int,
    max_y: int,
    hide_plants: bool,
) -> tuple[list[SurfaceBlock], int, int]:
    sampler = WorldSampler(world_dir)
    cx = math.floor(spawn.x)
    cz = math.floor(spawn.z)
    sy = math.floor(spawn.y)
    y_min = max(0, sy - max(0, layers_below))
    y_max = min(max_y, sy + max(0, layers_above))
    out: list[SurfaceBlock] = []
    for dz in range(-radius, radius + 1):
        z = cz + dz
        for dx in range(-radius, radius + 1):
            x = cx + dx
            for y in range(y_min, y_max + 1):
                bid = sampler.block_id(x, y, z)
                if bid in IGNORE_AIR_IDS:
                    continue
                if hide_plants and bid in PLANT_IDS:
                    continue
                out.append(SurfaceBlock(x=x, y=y, z=z, block_id=bid))
    return out, y_min, y_max


def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-9:
        return v
    return v / n


def build_camera_basis(yaw_deg: float, elevation_deg: float, side_mix: float, side: str):
    yaw = math.radians(yaw_deg)
    elev = math.radians(elevation_deg)

    fwd = np.array([-math.sin(yaw), 0.0, math.cos(yaw)], dtype=np.float64)
    right = normalize(np.cross(fwd, np.array([0.0, 1.0, 0.0], dtype=np.float64)))
    behind = -fwd

    side_sign = 1.0 if side == 'right' else -1.0
    horiz = normalize(behind + side_sign * side_mix * right)

    # Vector from scene center toward camera.
    to_camera = np.array([
        horiz[0] * math.cos(elev),
        math.sin(elev),
        horiz[2] * math.cos(elev),
    ], dtype=np.float64)
    to_camera = normalize(to_camera)

    cam_right = normalize(np.cross(np.array([0.0, 1.0, 0.0], dtype=np.float64), to_camera))
    cam_up = normalize(np.cross(to_camera, cam_right))
    return cam_right, cam_up, to_camera


def project_point(p_world: np.ndarray, center: np.ndarray, cam_right: np.ndarray, cam_up: np.ndarray, to_camera: np.ndarray):
    rel = p_world - center
    u = float(np.dot(rel, cam_right))
    v = float(np.dot(rel, cam_up))
    d = float(np.dot(rel, to_camera))
    return u, v, d


def cube_faces(x: float, y: float, z: float):
    v = np.array(
        [
            [x, y, z],
            [x + 1.0, y, z],
            [x, y + 1.0, z],
            [x + 1.0, y + 1.0, z],
            [x, y, z + 1.0],
            [x + 1.0, y, z + 1.0],
            [x, y + 1.0, z + 1.0],
            [x + 1.0, y + 1.0, z + 1.0],
        ],
        dtype=np.float64,
    )
    faces = [
        (np.array([0.0, 1.0, 0.0]), (0, 1, 0), [2, 3, 7, 6], 1.00),
        (np.array([1.0, 0.0, 0.0]), (1, 0, 0), [1, 5, 7, 3], 0.78),
        (np.array([-1.0, 0.0, 0.0]), (-1, 0, 0), [0, 2, 6, 4], 0.78),
        (np.array([0.0, 0.0, 1.0]), (0, 0, 1), [4, 6, 7, 5], 0.66),
        (np.array([0.0, 0.0, -1.0]), (0, 0, -1), [0, 1, 3, 2], 0.66),
    ]
    return v, faces


def render_isometric(
    blocks: list[SurfaceBlock],
    spawn: Pose,
    trajectory: list[Pose] | None,
    out_path: Path,
    y_min: int,
    y_max: int,
    width: int,
    height: int,
    margin: int,
    elevation_deg: float,
    side_mix: float,
    side: str,
):
    cam_right, cam_up, to_camera = build_camera_basis(
        yaw_deg=spawn.yaw,
        elevation_deg=elevation_deg,
        side_mix=side_mix,
        side=side,
    )

    center = np.array([spawn.x, spawn.y, spawn.z], dtype=np.float64)

    face_items = []
    u_vals = []
    v_vals = []
    occupancy = {(b.x, b.y, b.z) for b in blocks}

    for b in blocks:
        base_rgb = np.array(color_for_block(b.block_id), dtype=np.float64)
        vertices, faces = cube_faces(float(b.x), float(b.y), float(b.z))

        for normal, neighbor_dxyz, idxs, shade in faces:
            nx, ny, nz = neighbor_dxyz
            if (b.x + nx, b.y + ny, b.z + nz) in occupancy:
                continue
            if float(np.dot(normal, to_camera)) <= 0.0:
                continue

            poly_uvd = []
            depths = []
            for vid in idxs:
                u, v, d = project_point(vertices[vid], center, cam_right, cam_up, to_camera)
                poly_uvd.append((u, v))
                depths.append(d)
                u_vals.append(u)
                v_vals.append(v)

            hshade = float(np.clip(0.86 + (b.y - spawn.y) * 0.015, 0.58, 1.24))
            color = np.clip(base_rgb * shade * hshade, 0, 255).astype(np.uint8)
            depth = float(np.mean(depths))
            face_items.append((depth, poly_uvd, (int(color[0]), int(color[1]), int(color[2]))))

    traj_uv = []
    if trajectory:
        for t in trajectory:
            # Lift slightly so path is visible above grass surface.
            p = np.array([t.x, t.y + 0.2, t.z], dtype=np.float64)
            u, v, _ = project_point(p, center, cam_right, cam_up, to_camera)
            traj_uv.append((u, v))
            u_vals.append(u)
            v_vals.append(v)

    # Agent marker uses eye-level point.
    agent_point = np.array([spawn.x, spawn.y + 1.6, spawn.z], dtype=np.float64)
    au, av, _ = project_point(agent_point, center, cam_right, cam_up, to_camera)
    u_vals.append(au)
    v_vals.append(av)

    if not u_vals or not v_vals:
        raise RuntimeError('No projection data to render')

    u_min, u_max = min(u_vals), max(u_vals)
    v_min, v_max = min(v_vals), max(v_vals)

    span_u = max(1e-6, u_max - u_min)
    span_v = max(1e-6, v_max - v_min)
    scale = min((width - 2 * margin) / span_u, (height - 2 * margin) / span_v)

    def to_px(u: float, v: float) -> tuple[int, int]:
        px = int(round((u - u_min) * scale + margin))
        py = int(round((v_max - v) * scale + margin))
        return px, py

    img = np.full((height, width, 3), (22, 25, 30), dtype=np.uint8)

    # Draw far faces first.
    face_items.sort(key=lambda item: item[0])
    for _depth, poly_uv, color in face_items:
        pts = np.array([to_px(u, v) for (u, v) in poly_uv], dtype=np.int32)
        cv2.fillConvexPoly(img, pts, color, lineType=cv2.LINE_AA)

    # Draw trajectory as gradient if provided.
    if len(traj_uv) >= 2:
        traj_px = [to_px(u, v) for (u, v) in traj_uv]
        for i in range(1, len(traj_px)):
            t = i / max(1, len(traj_px) - 1)
            color = (int(255 * t), int(255 * (1.0 - t)), 255)
            cv2.line(img, traj_px[i - 1], traj_px[i], color, 2, cv2.LINE_AA)
        cv2.circle(img, traj_px[0], 4, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(img, traj_px[-1], 5, (255, 80, 80), -1, cv2.LINE_AA)

    # Agent marker at spawn.
    a_px = to_px(au, av)
    cv2.circle(img, a_px, 6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.circle(img, a_px, 2, (255, 0, 0), -1, cv2.LINE_AA)

    title = f'Isometric voxel view (behind agent), Y [{y_min}, {y_max}]'
    cv2.putText(img, title, (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (240, 240, 240), 2, cv2.LINE_AA)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def main():
    args = parse_args()
    world = Path(args.world)
    spawn_stats = Path(args.spawn_location_stats)

    spawn = read_first_pose(spawn_stats)
    blocks, y_min, y_max = collect_volume_blocks(
        world,
        spawn,
        radius=args.radius,
        layers_below=args.layers_below,
        layers_above=args.layers_above,
        max_y=args.max_y,
        hide_plants=args.hide_plants,
    )

    render_isometric(
        blocks=blocks,
        spawn=spawn,
        trajectory=None,
        out_path=Path(args.out_blocks),
        y_min=y_min,
        y_max=y_max,
        width=args.width,
        height=args.height,
        margin=args.margin,
        elevation_deg=args.elevation_deg,
        side_mix=args.behind_side_mix,
        side=args.behind_side,
    )

    if args.trajectory_location_stats and args.out_with_trajectory:
        traj = read_trajectory(Path(args.trajectory_location_stats), step=args.trajectory_step)
        clip_r = float(args.trajectory_clip_radius) if args.trajectory_clip_radius is not None else float(args.radius)
        traj_clipped = clip_trajectory_to_radius(traj, spawn=spawn, radius=clip_r)
        if len(traj_clipped) >= 2:
            traj = traj_clipped
        render_isometric(
            blocks=blocks,
            spawn=spawn,
            trajectory=traj,
            out_path=Path(args.out_with_trajectory),
            y_min=y_min,
            y_max=y_max,
            width=args.width,
            height=args.height,
            margin=args.margin,
            elevation_deg=args.elevation_deg,
            side_mix=args.behind_side_mix,
            side=args.behind_side,
        )


if __name__ == '__main__':
    main()
