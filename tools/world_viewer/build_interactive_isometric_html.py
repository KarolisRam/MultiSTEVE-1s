#!/usr/bin/env python3
import argparse
import base64
import csv
import hashlib
import json
import math
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import anvil


IGNORE_AIR_IDS = {'air', 'cave_air', 'void_air'}
BARRIER_IDS = {'barrier'}
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

THREE_VERSION = '0.160.0'
THREE_JS_URL = f'https://unpkg.com/three@{THREE_VERSION}/build/three.min.js'

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


class WorldSampler:
    def __init__(self, world_dir: Path):
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
    p = argparse.ArgumentParser(description='Build self-contained interactive isometric HTML from world voxel data.')
    p.add_argument('--world', required=True)
    p.add_argument('--spawn-location-stats', required=True)
    p.add_argument('--trajectory-location-stats', default=None)
    p.add_argument('--trajectory-dir', default=None,
                   help='Directory containing per-episode subfolders with location_stats.log trajectories')
    p.add_argument('--max-trajectories', type=int, default=10,
                   help='When using --trajectory-dir, use this many first (sorted) trajectories')
    p.add_argument('--out-html', required=True)
    p.add_argument('--radius', type=int, default=42)
    p.add_argument('--layers-below', type=int, default=20)
    p.add_argument('--layers-above', type=int, default=20)
    p.add_argument('--hide-plants', action='store_true',
                   help='Skip flowers and grass-like decorative plants in voxel extraction')
    p.add_argument('--max-y', type=int, default=255)
    p.add_argument('--trajectory-step', type=int, default=2)
    p.add_argument('--trajectory-clip-radius', type=float, default=None,
                   help='Optional x/z clip radius for trajectory around spawn. '
                        'If omitted, trajectory is not clipped.')
    p.add_argument('--elevation-deg', type=float, default=35.0)
    p.add_argument('--behind-side-mix', type=float, default=0.85)
    p.add_argument('--behind-side', choices=['right', 'left'], default='right')
    p.add_argument('--offline-bundle', action='store_true', default=True,
                   help='Embed Three.js runtime scripts in output HTML (default: enabled)')
    p.add_argument('--no-offline-bundle', action='store_false', dest='offline_bundle',
                   help='Use CDN runtime script URLs instead of embedding')
    p.add_argument('--three-js-file', default=None,
                   help='Optional local path to three.min.js for offline bundling')
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


def clip_trajectory(points: list[Pose], spawn: Pose, radius: float) -> list[Pose]:
    return [p for p in points if abs(p.x - spawn.x) <= radius and abs(p.z - spawn.z) <= radius]


def truncate_trajectory_at_bounds(points: list[Pose], spawn: Pose, radius: float) -> list[Pose]:
    out: list[Pose] = []
    for p in points:
        if abs(p.x - spawn.x) <= radius and abs(p.z - spawn.z) <= radius:
            out.append(p)
        else:
            break
    return out


def color_for_block(block_id: str) -> tuple[int, int, int]:
    if block_id in COLOR_MAP:
        return COLOR_MAP[block_id]
    h = hashlib.md5(block_id.encode('utf-8')).digest()
    return (70 + (h[0] % 120), 70 + (h[1] % 120), 70 + (h[2] % 120))


def rgb_to_hex(rgb: tuple[int, int, int]) -> int:
    return (int(rgb[0]) << 16) | (int(rgb[1]) << 8) | int(rgb[2])


def collect_visible_voxels(
    world_dir: Path,
    spawn: Pose,
    radius: int,
    layers_below: int,
    layers_above: int,
    max_y: int,
    hide_plants: bool,
):
    sampler = WorldSampler(world_dir)
    sx = math.floor(spawn.x)
    sy = math.floor(spawn.y)
    sz = math.floor(spawn.z)

    y_min = max(0, sy - max(0, layers_below))
    y_max = min(max_y, sy + max(0, layers_above))

    occ: dict[tuple[int, int, int], str] = {}
    for dz in range(-radius, radius + 1):
        z = sz + dz
        for dx in range(-radius, radius + 1):
            x = sx + dx
            for y in range(y_min, y_max + 1):
                bid = sampler.block_id(x, y, z)
                if bid in IGNORE_AIR_IDS:
                    continue
                if bid in BARRIER_IDS:
                    continue
                if hide_plants and bid in PLANT_IDS:
                    continue
                occ[(x, y, z)] = bid

    dirs = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
    visible = []
    for (x, y, z), bid in occ.items():
        if any((x + dx, y + dy, z + dz) not in occ for dx, dy, dz in dirs):
            rgb = color_for_block(bid)
            visible.append([x - sx, y - sy, z - sz, rgb_to_hex(rgb)])

    return visible, (sx, sy, sz), (y_min, y_max)


def _sanitize_inline_js(js_text: str) -> str:
    return js_text.replace('</script', '<\\/script')


def load_runtime_sources(args: argparse.Namespace) -> str:
    def read_file(path: Path) -> str:
        return path.read_text(encoding='utf-8')

    if args.three_js_file:
        three_path = Path(args.three_js_file)
    else:
        runtime_dir = Path(__file__).resolve().parent / 'runtime_js'
        runtime_dir.mkdir(parents=True, exist_ok=True)
        three_path = runtime_dir / f'three-{THREE_VERSION}.min.js'
        if not three_path.exists():
            urllib.request.urlretrieve(THREE_JS_URL, str(three_path))
    return _sanitize_inline_js(read_file(three_path))


def build_html(
    meta: dict,
    voxels: list[list[int]],
    trajectories: list[list[list[float]]],
    offline_bundle: bool,
    three_source: str,
) -> str:
    meta_json = json.dumps(meta, separators=(',', ':'))
    voxels_json = json.dumps(voxels, separators=(',', ':'))
    trajs_json = json.dumps(trajectories, separators=(',', ':'))

    runtime_block = ''
    if offline_bundle:
        three_b64 = base64.b64encode(three_source.encode('utf-8')).decode('ascii')
        runtime_block = (
            '<script>\n'
            '(function(){\n'
            f'  const __three_b64 = "{three_b64}";\n'
            '  (0, eval)(atob(__three_b64));\n'
            '})();\n'
            '</script>'
        )
    else:
        runtime_block = f'<script src="{THREE_JS_URL}"></script>'

    return f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width,initial-scale=1\" />
  <title>Interactive Isometric Voxel View</title>
  <style>
    html, body {{ margin: 0; width: 100%; height: 100%; background: #11141b; overflow: hidden; }}
    #app {{ position: fixed; inset: 0; }}
    #hud {{
      position: fixed; left: 12px; top: 12px; z-index: 10;
      background: rgba(12, 16, 24, 0.78); color: #e8edf5;
      font: 13px/1.35 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      padding: 10px 12px; border: 1px solid rgba(255,255,255,0.15); border-radius: 8px;
      max-width: 460px;
    }}
    #hud .row {{ margin-top: 4px; }}
    #hud .title {{ font-weight: 700; margin-bottom: 6px; }}
    #trajControls {{
      display: flex; align-items: center; gap: 6px; margin-top: 8px;
    }}
    #trajControls button {{
      border: 1px solid rgba(255,255,255,0.18); border-radius: 5px;
      background: rgba(255,255,255,0.08); color: #e8edf5; padding: 3px 8px;
      font: inherit; cursor: pointer;
    }}
    #trajControls button:disabled {{ opacity: 0.45; cursor: default; }}
    #trajProgress {{ width: 190px; }}
    #trajTime {{ min-width: 78px; text-align: right; font-variant-numeric: tabular-nums; }}
    #fallback {{
      position: fixed; inset: 0; display: none; place-items: center; text-align: center;
      color: white; background: #0f1218; padding: 24px; font: 16px/1.5 system-ui, sans-serif;
    }}
  </style>
</head>
<body>
  <div id=\"app\"></div>
  <div id=\"hud\">
    <div class=\"title\">Interactive Isometric Voxel View</div>
    <div class=\"row\" id=\"meta1\"></div>
    <div class=\"row\" id=\"meta2\"></div>
    <div class=\"row\">Mouse: drag rotate, wheel zoom, right-drag pan</div>
    <div class=\"row\">Keys: <b>R</b> reset view, <b>T</b> toggle paths, <b>Space</b> play/pause, <b>B</b> rewind</div>
    <div class=\"row\" id=\"trajControls\">
      <button id=\"playPauseBtn\">Play</button>
      <button id=\"rewindBtn\">Rewind</button>
      <input id=\"trajProgress\" type=\"range\" min=\"0\" max=\"0\" step=\"1\" value=\"0\" />
      <span id=\"trajTime\">0.00s</span>
    </div>
  </div>
  <div id=\"fallback\">Failed to load 3D runtime.<br/>If this is an offline-bundled file, the embedded runtime may be corrupted.</div>

  {runtime_block}
  <script>
  const META = {meta_json};
  const VOXELS = {voxels_json};
  const TRAJECTORIES = {trajs_json};

  if (!window.THREE) {{
    document.getElementById('fallback').style.display = 'grid';
    throw new Error('Three.js failed to load');
  }}

  const app = document.getElementById('app');
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x11141b);

  const camera = new THREE.PerspectiveCamera(55, window.innerWidth / window.innerHeight, 0.1, 5000);
  const renderer = new THREE.WebGLRenderer({{ antialias: true }});
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setSize(window.innerWidth, window.innerHeight);
  if ('outputColorSpace' in renderer && THREE.SRGBColorSpace) {{
    renderer.outputColorSpace = THREE.SRGBColorSpace;
  }} else if ('outputEncoding' in renderer && THREE.sRGBEncoding !== undefined) {{
    renderer.outputEncoding = THREE.sRGBEncoding;
  }}
  app.appendChild(renderer.domElement);

  const hemi = new THREE.HemisphereLight(0xffffff, 0x5a5d66, 0.95);
  scene.add(hemi);
  const ambient = new THREE.AmbientLight(0xffffff, 0.16);
  scene.add(ambient);
  const sun = new THREE.DirectionalLight(0xffffff, 0.85);
  sun.position.set(70, 140, 60);
  scene.add(sun);
  const fill = new THREE.DirectionalLight(0xaeb9ff, 0.18);
  fill.position.set(-90, 55, -70);
  scene.add(fill);

  const boxGeo = new THREE.BoxGeometry(1, 1, 1);
  const dummy = new THREE.Object3D();
  const voxelsByColor = new Map();
  for (let i = 0; i < VOXELS.length; i++) {{
    const v = VOXELS[i];
    const c = (v[3] >>> 0);
    if (!voxelsByColor.has(c)) voxelsByColor.set(c, []);
    voxelsByColor.get(c).push(v);
  }}
  for (const [hexColor, voxList] of voxelsByColor.entries()) {{
    const mat = new THREE.MeshLambertMaterial({{ color: hexColor, flatShading: true }});
    const mesh = new THREE.InstancedMesh(boxGeo, mat, voxList.length);
    mesh.instanceMatrix.setUsage(THREE.StaticDrawUsage);
    for (let i = 0; i < voxList.length; i++) {{
      const v = voxList[i];
      dummy.position.set(v[0] + 0.5, v[1] + 0.5, v[2] + 0.5);
      dummy.updateMatrix();
      mesh.setMatrixAt(i, dummy.matrix);
    }}
    mesh.instanceMatrix.needsUpdate = true;
    scene.add(mesh);
  }}

  const spawnGroup = new THREE.Group();
  scene.add(spawnGroup);
  const spawnMarker = new THREE.Mesh(
    new THREE.SphereGeometry(0.32, 16, 16),
    new THREE.MeshBasicMaterial({{ color: 0xff3a3a }})
  );
  spawnMarker.position.set(META.spawn_rel[0], META.spawn_rel[1] + 1.6, META.spawn_rel[2]);
  spawnGroup.add(spawnMarker);

  const ring = new THREE.Mesh(
    new THREE.TorusGeometry(0.55, 0.06, 12, 28),
    new THREE.MeshBasicMaterial({{ color: 0xffffff }})
  );
  ring.rotation.x = Math.PI / 2;
  ring.position.copy(spawnMarker.position);
  spawnGroup.add(ring);

  const TRAJECTORY_PALETTE = [
    0xff4d4d, 0xffa64d, 0xffde59, 0x7ddc59, 0x59d3ff,
    0x7a8dff, 0xc778ff, 0xff7ab8, 0x6cf0c5, 0xf0f06c
  ];

  let trajPathGroup = null;
  let trajDotGroup = null;
  const trajDots = [];
  const trajHistoryLines = [];
  function buildTrajectoryLine(points, colorHex) {{
    if (!points || points.length < 2) return null;
    const pos = new Float32Array(points.length * 3);
    for (let i = 0; i < points.length; i++) {{
      pos[i*3+0] = points[i][0];
      pos[i*3+1] = points[i][1] + 0.22;
      pos[i*3+2] = points[i][2];
    }}
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    const m = new THREE.LineBasicMaterial({{ color: colorHex, linewidth: 2 }});
    const line = new THREE.Line(g, m);
    return line;
  }}

  const dotGeo = new THREE.SphereGeometry(0.34, 14, 14);
  trajPathGroup = new THREE.Group();
  trajDotGroup = new THREE.Group();
  for (let i = 0; i < TRAJECTORIES.length; i++) {{
    const pts = TRAJECTORIES[i];
    const colorHex = TRAJECTORY_PALETTE[i % TRAJECTORY_PALETTE.length];
    let historyLine = null;
    const line = buildTrajectoryLine(pts, colorHex);
    if (line) {{
      line.geometry.setDrawRange(0, Math.min(1, pts.length));
      line.visible = false;
      trajPathGroup.add(line);
      historyLine = line;
    }}
    trajHistoryLines.push(historyLine);
    const dot = new THREE.Mesh(
      dotGeo,
      new THREE.MeshBasicMaterial({{ color: colorHex }})
    );
    dot.visible = false;
    trajDotGroup.add(dot);
    trajDots.push(dot);
  }}
  // Path lines are progressively revealed during playback.
  trajPathGroup.visible = true;
  scene.add(trajPathGroup);
  scene.add(trajDotGroup);

  const orbit = {{
    target: new THREE.Vector3(0, 0, 0),
    radius: META.camera_distance,
    theta: 0.0,
    phi: Math.PI / 3.0,
    rotateSpeed: 0.0045,
    panSpeed: 0.0018,
    zoomSpeed: 0.0012,
  }};

  function syncOrbitFromCamera() {{
    const off = camera.position.clone().sub(orbit.target);
    orbit.radius = Math.max(1e-3, off.length());
    orbit.theta = Math.atan2(off.x, off.z);
    orbit.phi = Math.acos(THREE.MathUtils.clamp(off.y / orbit.radius, -1.0, 1.0));
  }}

  function updateCameraFromOrbit() {{
    orbit.radius = THREE.MathUtils.clamp(orbit.radius, 6.0, 3000.0);
    orbit.phi = THREE.MathUtils.clamp(orbit.phi, 0.05, Math.PI - 0.05);
    const sinPhiR = Math.sin(orbit.phi) * orbit.radius;
    camera.position.set(
      orbit.target.x + sinPhiR * Math.sin(orbit.theta),
      orbit.target.y + Math.cos(orbit.phi) * orbit.radius,
      orbit.target.z + sinPhiR * Math.cos(orbit.theta),
    );
    camera.lookAt(orbit.target);
  }}

  function resetCamera() {{
    const yaw = META.spawn_yaw_deg * Math.PI / 180.0;
    const fwd = new THREE.Vector3(-Math.sin(yaw), 0, Math.cos(yaw));
    const up = new THREE.Vector3(0, 1, 0);
    const right = new THREE.Vector3().crossVectors(fwd, up).normalize();
    const behind = fwd.clone().multiplyScalar(-1);
    const sideSign = META.behind_side === 'right' ? 1 : -1;
    const horiz = behind.clone().add(right.multiplyScalar(sideSign * META.behind_side_mix)).normalize();

    const elev = META.elevation_deg * Math.PI / 180.0;
    const toCam = new THREE.Vector3(
      horiz.x * Math.cos(elev),
      Math.sin(elev),
      horiz.z * Math.cos(elev)
    ).normalize();

    orbit.target.set(META.spawn_rel[0], META.spawn_rel[1] + 1.0, META.spawn_rel[2]);
    camera.position.copy(orbit.target.clone().add(toCam.multiplyScalar(META.camera_distance)));
    syncOrbitFromCamera();
    updateCameraFromOrbit();
  }}

  resetCamera();

  let isDragging = false;
  let dragButton = 0;
  let lastX = 0;
  let lastY = 0;
  renderer.domElement.addEventListener('contextmenu', (e) => e.preventDefault());
  renderer.domElement.addEventListener('pointerdown', (e) => {{
    isDragging = true;
    dragButton = e.button;
    lastX = e.clientX;
    lastY = e.clientY;
    renderer.domElement.setPointerCapture(e.pointerId);
  }});
  renderer.domElement.addEventListener('pointerup', (e) => {{
    isDragging = false;
    renderer.domElement.releasePointerCapture(e.pointerId);
  }});
  renderer.domElement.addEventListener('pointermove', (e) => {{
    if (!isDragging) return;
    const dx = e.clientX - lastX;
    const dy = e.clientY - lastY;
    lastX = e.clientX;
    lastY = e.clientY;

    if (dragButton === 0) {{
      orbit.theta -= dx * orbit.rotateSpeed;
      orbit.phi -= dy * orbit.rotateSpeed;
      updateCameraFromOrbit();
      return;
    }}

    if (dragButton === 2) {{
      const rightVec = new THREE.Vector3();
      const upVec = new THREE.Vector3();
      camera.getWorldDirection(upVec);
      rightVec.crossVectors(upVec, camera.up).normalize();
      upVec.copy(camera.up).normalize();
      const scale = orbit.radius * orbit.panSpeed;
      orbit.target.addScaledVector(rightVec, -dx * scale);
      orbit.target.addScaledVector(upVec, dy * scale);
      updateCameraFromOrbit();
    }}
  }});
  renderer.domElement.addEventListener('wheel', (e) => {{
    e.preventDefault();
    const zoom = Math.exp(e.deltaY * orbit.zoomSpeed);
    orbit.radius *= zoom;
    updateCameraFromOrbit();
  }}, {{ passive: false }});

  const trajectoryPointCount = TRAJECTORIES.reduce((acc, t) => acc + t.length, 0);
  const playbackFps = 20.0;
  const maxFrame = TRAJECTORIES.reduce((acc, t) => Math.max(acc, Math.max(0, t.length - 1)), 0);
  document.getElementById('meta1').textContent =
    `Voxels: ${{VOXELS.length.toLocaleString()}} | Trajectories: ${{TRAJECTORIES.length.toLocaleString()}} | Traj points: ${{trajectoryPointCount.toLocaleString()}} | Frames: ${{(maxFrame + 1).toLocaleString()}}`;
  document.getElementById('meta2').textContent =
    `Y range: [${{META.y_min}}, ${{META.y_max}}] | Radius: ${{META.radius}} | Spawn yaw: ${{META.spawn_yaw_deg.toFixed(1)}}°`;

  const playPauseBtn = document.getElementById('playPauseBtn');
  const rewindBtn = document.getElementById('rewindBtn');
  const trajProgress = document.getElementById('trajProgress');
  const trajTime = document.getElementById('trajTime');

  let playbackFrame = 0.0;
  let isPlaying = false;
  let lastTickMs = performance.now();
  const hasPlayback = TRAJECTORIES.length > 0 && maxFrame > 0;

  trajProgress.max = String(Math.max(0, maxFrame));
  playPauseBtn.disabled = !hasPlayback;
  rewindBtn.disabled = !hasPlayback;
  trajProgress.disabled = !hasPlayback;

  function updatePlaybackLabel() {{
    const frameInt = Math.max(0, Math.min(maxFrame, Math.floor(playbackFrame + 1e-9)));
    const tSec = frameInt / playbackFps;
    const totalSec = Math.max(0, maxFrame) / playbackFps;
    trajTime.textContent = `${{tSec.toFixed(2)}}s/${{totalSec.toFixed(2)}}s`;
  }}

  function setPlaybackFrame(frame) {{
    playbackFrame = THREE.MathUtils.clamp(frame, 0.0, Math.max(0, maxFrame));
    const frameInt = Math.max(0, Math.min(maxFrame, Math.floor(playbackFrame + 1e-9)));
    trajProgress.value = String(frameInt);
    for (let i = 0; i < TRAJECTORIES.length; i++) {{
      const pts = TRAJECTORIES[i];
      const dot = trajDots[i];
      const historyLine = trajHistoryLines[i];
      if (!dot || !pts || pts.length < 1) continue;
      const idx = Math.min(frameInt, pts.length - 1);
      const p = pts[idx];
      dot.position.set(p[0], p[1] + 0.35, p[2]);
      dot.visible = true;
      if (historyLine) {{
        historyLine.geometry.setDrawRange(0, idx + 1);
        historyLine.visible = idx >= 1;
      }}
    }}
    updatePlaybackLabel();
  }}

  function syncPlayButton() {{
    playPauseBtn.textContent = isPlaying ? 'Pause' : 'Play';
  }}

  playPauseBtn.addEventListener('click', () => {{
    if (!hasPlayback) return;
    isPlaying = !isPlaying;
    syncPlayButton();
  }});
  rewindBtn.addEventListener('click', () => {{
    isPlaying = false;
    setPlaybackFrame(0.0);
    syncPlayButton();
  }});
  trajProgress.addEventListener('input', () => {{
    isPlaying = false;
    setPlaybackFrame(Number(trajProgress.value));
    syncPlayButton();
  }});

  setPlaybackFrame(0.0);
  syncPlayButton();

  window.addEventListener('keydown', (e) => {{
    const tag = (e.target && e.target.tagName ? e.target.tagName : '').toLowerCase();
    if (tag === 'input' || tag === 'button') return;
    const key = e.key.toLowerCase();
    if (key === 'r') {{
      resetCamera();
    }} else if (key === 't' && trajPathGroup) {{
      trajPathGroup.visible = !trajPathGroup.visible;
    }} else if (key === ' ' || key === 'space') {{
      e.preventDefault();
      if (!hasPlayback) return;
      isPlaying = !isPlaying;
      syncPlayButton();
    }} else if (key === 'b') {{
      isPlaying = false;
      setPlaybackFrame(0.0);
      syncPlayButton();
    }}
  }});

  window.addEventListener('resize', () => {{
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
  }});

  renderer.setAnimationLoop((tMs) => {{
    const dt = Math.max(0.0, Math.min(0.25, (tMs - lastTickMs) / 1000.0));
    lastTickMs = tMs;
    if (isPlaying && hasPlayback) {{
      const next = playbackFrame + dt * playbackFps;
      if (next >= maxFrame) {{
        setPlaybackFrame(maxFrame);
        isPlaying = false;
        syncPlayButton();
      }} else {{
        setPlaybackFrame(next);
      }}
    }}
    renderer.render(scene, camera);
  }});
  </script>
</body>
</html>
"""


def main():
    args = parse_args()

    spawn = read_first_pose(Path(args.spawn_location_stats))
    voxels, (sx, sy, sz), (y_min, y_max) = collect_visible_voxels(
        world_dir=Path(args.world),
        spawn=spawn,
        radius=args.radius,
        layers_below=args.layers_below,
        layers_above=args.layers_above,
        max_y=args.max_y,
        hide_plants=args.hide_plants,
    )

    traj_rels: list[list[list[float]]] = []
    if args.trajectory_dir:
        tdir = Path(args.trajectory_dir)
        files = sorted(tdir.glob('*/location_stats.log'))[:max(0, int(args.max_trajectories))]
        for tfile in files:
            traj = read_trajectory(tfile, step=args.trajectory_step)
            traj = truncate_trajectory_at_bounds(traj, spawn=spawn, radius=float(args.radius))
            if args.trajectory_clip_radius is not None:
                traj = clip_trajectory(traj, spawn=spawn, radius=float(args.trajectory_clip_radius))
            rel = [[p.x - sx, p.y - sy, p.z - sz] for p in traj]
            if len(rel) >= 2:
                traj_rels.append(rel)
    elif args.trajectory_location_stats:
        traj = read_trajectory(Path(args.trajectory_location_stats), step=args.trajectory_step)
        if args.trajectory_clip_radius is not None:
            traj = clip_trajectory(traj, spawn=spawn, radius=float(args.trajectory_clip_radius))
        rel = [[p.x - sx, p.y - sy, p.z - sz] for p in traj]
        if len(rel) >= 2:
            traj_rels.append(rel)

    camera_distance = max(70.0, args.radius * 3.2)

    meta = {
        'radius': args.radius,
        'layers_below': args.layers_below,
        'layers_above': args.layers_above,
        'y_min': y_min,
        'y_max': y_max,
        'spawn_yaw_deg': float(spawn.yaw),
        'spawn_rel': [spawn.x - sx, spawn.y - sy, spawn.z - sz],
        'behind_side_mix': float(args.behind_side_mix),
        'behind_side': args.behind_side,
        'elevation_deg': float(args.elevation_deg),
        'camera_distance': float(camera_distance),
    }

    three_source = ''
    if args.offline_bundle:
        three_source = load_runtime_sources(args)

    html = build_html(
        meta=meta,
        voxels=voxels,
        trajectories=traj_rels,
        offline_bundle=args.offline_bundle,
        three_source=three_source,
    )
    out = Path(args.out_html)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding='utf-8')

    print(f'Wrote {out}')
    print(f'Visible voxels: {len(voxels)}')
    print(f'Trajectories: {len(traj_rels)}')
    print(f'Trajectory points: {sum(len(t) for t in traj_rels)}')


if __name__ == '__main__':
    main()
