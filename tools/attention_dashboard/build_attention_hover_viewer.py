import argparse
import base64
import json
import pickle
import shlex
import zipfile
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode-dir", required=True, help="Episode directory containing attention and observation files")
    parser.add_argument("--output-dir", required=True, help="Viewer output directory")
    parser.add_argument("--output-html", default="index.html", help="HTML filename to write inside output-dir")
    parser.add_argument(
        "--diff-strategy",
        choices=["linear", "percentile", "asinh"],
        default="linear",
        help="How to colorize conditional-unconditional differences",
    )
    parser.add_argument("--viewer-title", default=None, help="Optional title suffix shown in the page")
    return parser.parse_args()


def load_model_config_from_command(episode_dir: Path) -> dict:
    command_text = (episode_dir / "command.log").read_text().strip()
    argv = shlex.split(command_text)
    in_model = None
    if "--in_model" in argv:
        idx = argv.index("--in_model")
        if idx + 1 < len(argv):
            in_model = argv[idx + 1]
    if in_model is None:
        raise ValueError(f"Could not find --in_model in {(episode_dir / 'command.log')}")

    with open(in_model, "rb") as f:
        agent_parameters = pickle.load(f)
    return agent_parameters["model"]["args"]["net"]["args"]


def extract_frames(zip_path: Path, frames_dir: Path) -> int:
    frames_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        frame_names = sorted(name for name in zf.namelist() if name.endswith(".png"))
        missing = [name for name in frame_names if not (frames_dir / Path(name).name).exists()]
        for name in missing:
            target = frames_dir / Path(name).name
            with zf.open(name) as src, open(target, "wb") as dst:
                dst.write(src.read())
    return len(list(frames_dir.glob("frame_*.png")))


def build_html(
    attn_b64: str,
    n_frames: int,
    n_layers: int,
    logged_maps_per_layer: int,
    context_positions: int,
    configured_heads: int,
    branch_count: int,
    branch_labels: list[str],
    diff_strategy: str,
    viewer_title: str | None,
) -> str:
    layer_options_html = "\n".join(
        [f'<option value="{i}"{" selected" if i == 0 else ""}>{i}</option>' for i in range(n_layers)]
        + ['<option value="all">All Layers (max)</option>']
    )
    default_head = 9 if configured_heads > 9 else 0
    head_options_html = "\n".join(
        [f'<option value="{i}"{" selected" if i == default_head else ""}>{i}</option>' for i in range(configured_heads)]
        + ['<option value="all">All Heads (max)</option>']
    )
    scale_options_html = "\n".join(
        f'<option value="{i}"{" selected" if i == 2 else ""}>{i} px per cell</option>' for i in range(1, 6)
    )

    if branch_count == 2:
        mode_options_html = "\n".join(
            [
                '<option value="conditional" selected>Conditional</option>',
                '<option value="unconditional">Unconditional</option>',
                f'<option value="difference">Difference ({diff_strategy})</option>',
            ]
        )
    else:
        mode_options_html = "\n".join(
            f'<option value="branch_{i}"{" selected" if i == 0 else ""}>{label}</option>'
            for i, label in enumerate(branch_labels)
        )

    title_suffix = viewer_title or {
        "linear": "Difference Fix 1: Linear Diverging",
        "percentile": "Difference Fix 2: Percentile Clip",
        "asinh": "Difference Fix 3: Asinh Boost",
    }[diff_strategy]

    branch_labels_json = json.dumps(branch_labels)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Attention Viewer - {title_suffix}</title>
  <style>
    :root {{
      color-scheme: light;
      --panel: #fffdf8;
      --panel-alt: #f7f2e8;
      --border: #c9bfaf;
      --text: #1d1a15;
      --muted: #675f56;
      --shadow: 0 14px 36px rgba(29, 26, 21, 0.09);
    }}
    * {{
      box-sizing: border-box;
    }}
    body {{
      margin: 0;
      font-family: Georgia, "Times New Roman", serif;
      color: var(--text);
      background: linear-gradient(180deg, #faf7f1 0%, #efe8dc 100%);
    }}
    .page {{
      max-width: 1720px;
      margin: 0 auto;
      padding: 22px;
    }}
    .header {{
      margin-bottom: 16px;
    }}
    .header h1 {{
      margin: 0 0 6px;
      font-size: 28px;
      line-height: 1.05;
    }}
    .header p {{
      margin: 0;
      color: var(--muted);
      font-size: 14px;
    }}
    .top-layout {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) 320px;
      gap: 18px;
      align-items: start;
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--border);
      box-shadow: var(--shadow);
    }}
    .heatmap-panel {{
      padding: 14px;
    }}
    .controls-panel {{
      padding: 14px;
      display: grid;
      gap: 14px;
      align-content: start;
      background: linear-gradient(180deg, #fffdf8 0%, #f6f0e6 100%);
    }}
    .control-group label {{
      display: block;
      margin-bottom: 6px;
      font-size: 12px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }}
    .control-group select {{
      width: 100%;
      padding: 10px 12px;
      font-size: 15px;
      color: var(--text);
      border: 1px solid var(--border);
      background: var(--panel);
    }}
    .readout {{
      padding: 10px 12px;
      border: 1px solid var(--border);
      background: var(--panel-alt);
      font-size: 14px;
      line-height: 1.45;
    }}
    .scroll-shell {{
      width: 100%;
      overflow: auto;
      border: 1px solid #000;
      background: white;
    }}
    #heatmapCanvas {{
      display: block;
      image-rendering: pixelated;
      image-rendering: crisp-edges;
      cursor: crosshair;
      background: white;
    }}
    .axis-note {{
      margin-top: 10px;
      display: flex;
      justify-content: space-between;
      gap: 10px;
      color: var(--muted);
      font-size: 13px;
    }}
    .bottom-layout {{
      margin-top: 18px;
      display: grid;
      grid-template-columns: minmax(0, 1.2fr) minmax(220px, 1fr) minmax(220px, 1fr);
      gap: 18px;
      align-items: start;
    }}
    .stats-panel {{
      padding: 14px;
      font-size: 14px;
      line-height: 1.55;
      background: linear-gradient(180deg, #fffdf8 0%, #f8f2e8 100%);
      min-height: 100%;
    }}
    .stats-panel h2,
    .frame-panel h2 {{
      margin: 0 0 10px;
      font-size: 16px;
    }}
    .frame-panel {{
      padding: 12px;
      background: linear-gradient(180deg, #fffdf8 0%, #f7f1e8 100%);
      text-align: center;
    }}
    .frame-panel img {{
      width: 90%;
      aspect-ratio: 1 / 1;
      object-fit: contain;
      border: 1px solid var(--border);
      background: #fff;
      image-rendering: pixelated;
      display: block;
      margin: 0 auto;
    }}
    .caption {{
      margin-top: 8px;
      color: var(--muted);
      font-size: 13px;
    }}
    #tooltip {{
      position: fixed;
      display: none;
      z-index: 30;
      pointer-events: none;
      background: rgba(18, 18, 17, 0.94);
      color: #faf7f1;
      border: 1px solid rgba(255, 255, 255, 0.14);
      padding: 8px 10px;
      font-size: 12px;
      line-height: 1.45;
      white-space: nowrap;
    }}
    @media (max-width: 1200px) {{
      .top-layout {{
        grid-template-columns: 1fr;
      }}
      .bottom-layout {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <div class="header">
      <h1>{title_suffix}</h1>
    </div>

    <div class="top-layout">
      <div class="panel heatmap-panel">
        <div class="scroll-shell" id="scrollShell">
          <canvas id="heatmapCanvas" width="{n_frames}" height="{context_positions}"></canvas>
        </div>
        <div class="axis-note">
          <span>Attention position: 0 to {context_positions - 1}</span>
          <span>Episode frame: 0 to {n_frames - 1}</span>
        </div>
      </div>

      <div class="panel controls-panel">
        <div class="control-group">
          <label for="layerSelect">Layer</label>
          <select id="layerSelect">{layer_options_html}</select>
        </div>
        <div class="control-group">
          <label for="headSelect">Head</label>
          <select id="headSelect">{head_options_html}</select>
        </div>
        <div class="control-group">
          <label for="modeSelect">View</label>
          <select id="modeSelect">{mode_options_html}</select>
        </div>
        <div class="control-group">
          <label for="scaleSelect">Cell Size</label>
          <select id="scaleSelect">{scale_options_html}</select>
        </div>
        <div id="modeReadout" class="readout"></div>
      </div>
    </div>

    <div class="bottom-layout">
      <div class="panel stats-panel">
        <h2>Hover Details</h2>
        <div id="hoverStats">Hover over the heatmap to inspect a value and the corresponding frames.</div>
      </div>

      <div class="panel frame-panel">
        <h2>Current Frame</h2>
        <img id="currentFrameImg" alt="Current frame preview">
        <div id="currentFrameCaption" class="caption">Frame -</div>
      </div>

      <div class="panel frame-panel">
        <h2>Attended Frame</h2>
        <img id="attendedFrameImg" alt="Attended frame preview">
        <div id="attendedFrameCaption" class="caption">Frame -</div>
      </div>
    </div>
  </div>

  <div id="tooltip"></div>

  <script>
    const FRAME_COUNT = {n_frames};
    const LAYER_COUNT = {n_layers};
    const LOGGED_MAPS_PER_LAYER = {logged_maps_per_layer};
    const CONTEXT_POSITIONS = {context_positions};
    const CONFIGURED_HEADS = {configured_heads};
    const BRANCH_COUNT = {branch_count};
    const BRANCH_LABELS = {branch_labels_json};
    const ATTN_BASE64 = "{attn_b64}";
    const DIFF_STRATEGY = "{diff_strategy}";
    const DIFFERENCE_ENABLED = {str(branch_count == 2).lower()};

    function decodeBase64(base64) {{
      const binary = atob(base64);
      const bytes = new Uint8Array(binary.length);
      for (let i = 0; i < binary.length; i++) {{
        bytes[i] = binary.charCodeAt(i);
      }}
      return bytes;
    }}

    function float16ToFloat32(bits) {{
      const sign = (bits & 0x8000) ? -1 : 1;
      const exponent = (bits >> 10) & 0x1f;
      const fraction = bits & 0x03ff;
      if (exponent === 0) {{
        if (fraction === 0) return sign * 0;
        return sign * Math.pow(2, -14) * (fraction / 1024);
      }}
      if (exponent === 31) {{
        return fraction === 0 ? sign * Infinity : NaN;
      }}
      return sign * Math.pow(2, exponent - 15) * (1 + fraction / 1024);
    }}

    const attnBytes = decodeBase64(ATTN_BASE64);
    const attnWords = new Uint16Array(attnBytes.buffer);
    const attnValues = new Float32Array(attnWords.length);
    for (let i = 0; i < attnWords.length; i++) {{
      attnValues[i] = float16ToFloat32(attnWords[i]);
    }}

    const canvas = document.getElementById("heatmapCanvas");
    const ctx = canvas.getContext("2d", {{ alpha: false }});
    const tooltip = document.getElementById("tooltip");
    const scrollShell = document.getElementById("scrollShell");

    const layerSelect = document.getElementById("layerSelect");
    const headSelect = document.getElementById("headSelect");
    const modeSelect = document.getElementById("modeSelect");
    const scaleSelect = document.getElementById("scaleSelect");
    const modeReadout = document.getElementById("modeReadout");
    const hoverStats = document.getElementById("hoverStats");
    const currentFrameImg = document.getElementById("currentFrameImg");
    const currentFrameCaption = document.getElementById("currentFrameCaption");
    const attendedFrameImg = document.getElementById("attendedFrameImg");
    const attendedFrameCaption = document.getElementById("attendedFrameCaption");

    function selectedLayerValue() {{
      return layerSelect.value;
    }}

    function selectedHeadValue() {{
      return headSelect.value;
    }}

    function selectedMode() {{
      return modeSelect.value;
    }}

    function selectedScale() {{
      return Number(scaleSelect.value);
    }}

    function selectedLayerIndices() {{
      if (selectedLayerValue() === "all") {{
        return Array.from({{ length: LAYER_COUNT }}, (_, i) => i);
      }}
      return [Number(selectedLayerValue())];
    }}

    function selectedHeadIndices() {{
      if (selectedHeadValue() === "all") {{
        return Array.from({{ length: CONFIGURED_HEADS }}, (_, i) => i);
      }}
      return [Number(selectedHeadValue())];
    }}

    function selectedBranchIndex() {{
      const mode = selectedMode();
      if (mode === "conditional") return 0;
      if (mode === "unconditional") return 1;
      if (mode.startsWith("branch_")) return Number(mode.split("_")[1]);
      return 0;
    }}

    function arrayIndex(frame, layer, loggedMap, position) {{
      return (((frame * LAYER_COUNT + layer) * LOGGED_MAPS_PER_LAYER + loggedMap) * CONTEXT_POSITIONS + position);
    }}

    function rawAttentionValue(frame, layer, loggedMap, position) {{
      return attnValues[arrayIndex(frame, layer, loggedMap, position)];
    }}

    function branchValue(frame, position, branchIndex) {{
      const layers = selectedLayerIndices();
      const heads = selectedHeadIndices();
      let maxValue = -Infinity;
      for (let li = 0; li < layers.length; li++) {{
        const layer = layers[li];
        for (let hi = 0; hi < heads.length; hi++) {{
          const head = heads[hi];
          const value = rawAttentionValue(frame, layer, branchIndex * CONFIGURED_HEADS + head, position);
          if (value > maxValue) {{
            maxValue = value;
          }}
        }}
      }}
      return maxValue;
    }}

    function conditionalValue(frame, position) {{
      return branchValue(frame, position, 0);
    }}

    function unconditionalValue(frame, position) {{
      return branchValue(frame, position, 1);
    }}

    function selectionLabel() {{
      const layerText = selectedLayerValue() === "all" ? "All Layers (max)" : "Layer " + selectedLayerValue();
      const headText = selectedHeadValue() === "all" ? "All Heads (max)" : "Head " + selectedHeadValue();
      return layerText + " / " + headText;
    }}

    function framePath(frameIndex) {{
      const clamped = Math.max(0, Math.min(FRAME_COUNT - 1, frameIndex));
      return "frames/frame_" + String(clamped).padStart(6, "0") + ".png";
    }}

    function attendedFrameIndex(currentFrame, position) {{
      return Math.max(0, Math.min(FRAME_COUNT - 1, currentFrame - (CONTEXT_POSITIONS - position)));
    }}

    function applyScale() {{
      const scale = selectedScale();
      canvas.style.width = String(FRAME_COUNT * scale) + "px";
      canvas.style.height = String(CONTEXT_POSITIONS * scale) + "px";
    }}

    function percentile(sortedValues, q) {{
      if (sortedValues.length === 0) return 0;
      const idx = Math.max(0, Math.min(sortedValues.length - 1, Math.floor((sortedValues.length - 1) * q)));
      return sortedValues[idx];
    }}

    function lerp(a, b, t) {{
      return Math.round(a + (b - a) * t);
    }}

    function divergingRgb(normalized) {{
      const t = Math.max(0, Math.min(1, normalized));
      const blue = [45, 92, 255];
      const white = [255, 255, 255];
      const red = [214, 41, 41];
      if (t <= 0.5) {{
        const u = t / 0.5;
        return [
          lerp(blue[0], white[0], u),
          lerp(blue[1], white[1], u),
          lerp(blue[2], white[2], u),
        ];
      }}
      const u = (t - 0.5) / 0.5;
      return [
        lerp(white[0], red[0], u),
        lerp(white[1], red[1], u),
        lerp(white[2], red[2], u),
      ];
    }}

    function buildDifferenceGrid() {{
      const diffs = new Float32Array(FRAME_COUNT * CONTEXT_POSITIONS);
      const absVals = new Float32Array(FRAME_COUNT * CONTEXT_POSITIONS);
      let maxAbs = 0;
      let p = 0;

      for (let y = 0; y < CONTEXT_POSITIONS; y++) {{
        for (let x = 0; x < FRAME_COUNT; x++) {{
          const diff = conditionalValue(x, y) - unconditionalValue(x, y);
          const absDiff = Math.abs(diff);
          diffs[p] = diff;
          absVals[p] = absDiff;
          if (absDiff > maxAbs) maxAbs = absDiff;
          p += 1;
        }}
      }}

      const sortedAbs = Array.from(absVals).sort((a, b) => a - b);
      const p995 = percentile(sortedAbs, 0.995);
      return {{ diffs, maxAbs, p995 }};
    }}

    function mapDifferenceToRgb(diff, stats) {{
      if (stats.maxAbs === 0) {{
        return [255, 255, 255];
      }}

      let normalized;
      if (DIFF_STRATEGY === "linear") {{
        normalized = 0.5 + diff / (2 * stats.maxAbs);
      }} else if (DIFF_STRATEGY === "percentile") {{
        const scale = stats.p995 > 0 ? stats.p995 : stats.maxAbs;
        const clipped = Math.max(-scale, Math.min(scale, diff));
        normalized = 0.5 + clipped / (2 * scale);
      }} else {{
        const scale = stats.p995 > 0 ? stats.p995 / 6 : (stats.maxAbs / 6);
        const safe = scale > 0 ? scale : 1e-6;
        const denom = Math.asinh(stats.maxAbs / safe);
        normalized = 0.5 + Math.asinh(diff / safe) / (2 * denom);
      }}

      return divergingRgb(normalized);
    }}

    function renderHeatmap() {{
      const mode = selectedMode();
      const imageData = ctx.createImageData(FRAME_COUNT, CONTEXT_POSITIONS);
      let ptr = 0;
      let diffStats = null;
      if (mode === "difference") {{
        diffStats = buildDifferenceGrid();
      }}

      let diffPtr = 0;
      for (let y = 0; y < CONTEXT_POSITIONS; y++) {{
        for (let x = 0; x < FRAME_COUNT; x++) {{
          let r, g, b;
          if (mode === "conditional") {{
            const value = conditionalValue(x, y);
            const shade = 255 - Math.max(0, Math.min(255, Math.round(value * 255)));
            r = g = b = shade;
          }} else if (mode === "unconditional") {{
            const value = unconditionalValue(x, y);
            const shade = 255 - Math.max(0, Math.min(255, Math.round(value * 255)));
            r = g = b = shade;
          }} else if (mode.startsWith("branch_")) {{
            const value = branchValue(x, y, selectedBranchIndex());
            const shade = 255 - Math.max(0, Math.min(255, Math.round(value * 255)));
            r = g = b = shade;
          }} else {{
            const rgb = mapDifferenceToRgb(diffStats.diffs[diffPtr], diffStats);
            r = rgb[0];
            g = rgb[1];
            b = rgb[2];
            diffPtr += 1;
          }}

          imageData.data[ptr++] = r;
          imageData.data[ptr++] = g;
          imageData.data[ptr++] = b;
          imageData.data[ptr++] = 255;
        }}
      }}

      ctx.putImageData(imageData, 0, 0);

      if (mode === "difference") {{
        modeReadout.innerHTML =
          "<strong>Selection</strong>: " + selectionLabel() + "<br>" +
          "<strong>Mode</strong>: conditional - unconditional<br>" +
          "<strong>Max abs diff</strong>: " + diffStats.maxAbs.toFixed(6) + "<br>" +
          "<strong>99.5th pct abs diff</strong>: " + diffStats.p995.toFixed(6);
      }} else {{
        modeReadout.innerHTML =
          "<strong>Selection</strong>: " + selectionLabel() + "<br>" +
          "<strong>Mode</strong>: " + mode + "<br>" +
          "<strong>Value range</strong>: 0 to 1";
      }}

      applyScale();
    }}

    function updatePreview(frame, attended, position) {{
      const cond = conditionalValue(frame, position);
      const uncond = unconditionalValue(frame, position);
      const diff = cond - uncond;
      let displayed;
      const mode = selectedMode();
      if (mode === "conditional") {{
        displayed = cond;
      }} else if (mode === "unconditional") {{
        displayed = uncond;
      }} else if (mode.startsWith("branch_")) {{
        displayed = branchValue(frame, position, selectedBranchIndex());
      }} else {{
        displayed = diff;
      }}

      currentFrameImg.src = framePath(frame);
      attendedFrameImg.src = framePath(attended);
      currentFrameCaption.textContent = "Frame " + frame;
      attendedFrameCaption.textContent = "Frame " + attended;

      hoverStats.innerHTML =
        "<strong>Selection</strong>: " + selectionLabel() + "<br>" +
        "<strong>View</strong>: " + mode + "<br>" +
        "<strong>Current frame</strong>: " + frame + "<br>" +
        "<strong>Attended frame</strong>: " + attended + "<br>" +
        "<strong>Attention position</strong>: " + position + "<br>" +
        "<strong>Conditional value</strong>: " + cond.toFixed(6) + "<br>" +
        "<strong>Unconditional value</strong>: " + uncond.toFixed(6) + "<br>" +
        "<strong>Difference</strong>: " + diff.toFixed(6) + "<br>" +
        "<strong>Displayed value</strong>: " + displayed.toFixed(6);

      return {{ cond, uncond, diff, displayed }};
    }}

    function onHover(event) {{
      const rect = canvas.getBoundingClientRect();
      const rawX = Math.floor((event.clientX - rect.left) * FRAME_COUNT / rect.width);
      const rawY = Math.floor((event.clientY - rect.top) * CONTEXT_POSITIONS / rect.height);
      const x = Math.max(0, Math.min(FRAME_COUNT - 1, rawX));
      const y = Math.max(0, Math.min(CONTEXT_POSITIONS - 1, rawY));
      const attended = attendedFrameIndex(x, y);
      const values = updatePreview(x, attended, y);

      tooltip.style.display = "block";
      tooltip.style.left = String(event.clientX + 16) + "px";
      tooltip.style.top = String(event.clientY + 16) + "px";
      tooltip.innerHTML =
        "frame: " + x + "<br>" +
        "attended: " + attended + "<br>" +
        "position: " + y + "<br>" +
        "conditional: " + values.cond.toFixed(6) + "<br>" +
        "unconditional: " + values.uncond.toFixed(6) + "<br>" +
        "difference: " + values.diff.toFixed(6) + "<br>" +
        "displayed: " + values.displayed.toFixed(6);
    }}

    function hideTooltip() {{
      tooltip.style.display = "none";
    }}

    function initialPreview() {{
      const frame = 0;
      const position = CONTEXT_POSITIONS - 1;
      const attended = attendedFrameIndex(frame, position);
      updatePreview(frame, attended, position);
    }}

    layerSelect.addEventListener("change", function() {{
      renderHeatmap();
      initialPreview();
    }});
    headSelect.addEventListener("change", function() {{
      renderHeatmap();
      initialPreview();
    }});
    modeSelect.addEventListener("change", function() {{
      renderHeatmap();
      initialPreview();
    }});
    scaleSelect.addEventListener("change", applyScale);
    canvas.addEventListener("mousemove", onHover);
    canvas.addEventListener("mouseleave", hideTooltip);

    renderHeatmap();
    initialPreview();
    scrollShell.scrollLeft = 0;
  </script>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    episode_dir = Path(args.episode_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_cfg = load_model_config_from_command(episode_dir)
    configured_layers = int(model_cfg["n_recurrence_layers"])
    configured_heads = int(model_cfg["attention_heads"])

    raw = np.load(episode_dir / "attn_patterns.npy", allow_pickle=True)
    if raw.ndim != 3 or raw.shape[0] % configured_layers != 0:
        raise ValueError(f"Unexpected attention tensor shape: {raw.shape}")

    n_frames = raw.shape[0] // configured_layers
    logged_maps_per_layer = raw.shape[1]
    attn = raw.reshape(n_frames, configured_layers, logged_maps_per_layer, raw.shape[2])[:, :, :, 1:]
    context_positions = attn.shape[-1]

    if logged_maps_per_layer % configured_heads != 0:
        raise ValueError(
            f"Logged maps per layer {logged_maps_per_layer} not divisible by configured heads {configured_heads}"
        )
    branch_count = logged_maps_per_layer // configured_heads
    if branch_count == 2:
        branch_labels = ["conditional", "unconditional"]
    else:
        branch_labels = [f"branch {i}" for i in range(branch_count)]

    frames_dir = output_dir / "frames"
    frame_count = extract_frames(episode_dir / "observations.zip", frames_dir)
    if frame_count != n_frames:
        raise ValueError(f"Frame count mismatch: extracted {frame_count}, attention uses {n_frames}")

    attn_words = np.ascontiguousarray(attn.astype(np.float16, copy=False)).view(np.uint16)
    attn_b64 = base64.b64encode(attn_words.tobytes()).decode("ascii")

    html = build_html(
        attn_b64=attn_b64,
        n_frames=n_frames,
        n_layers=configured_layers,
        logged_maps_per_layer=logged_maps_per_layer,
        context_positions=context_positions,
        configured_heads=configured_heads,
        branch_count=branch_count,
        branch_labels=branch_labels,
        diff_strategy=args.diff_strategy,
        viewer_title=args.viewer_title,
    )

    output_path = output_dir / args.output_html
    output_path.write_text(html)

    print(f"episode_dir={episode_dir}")
    print(f"raw_shape={raw.shape}")
    print(f"viewer_tensor_shape={attn.shape}")
    print(f"configured_layers={configured_layers}")
    print(f"configured_heads={configured_heads}")
    print(f"logged_maps_per_layer={logged_maps_per_layer}")
    print(f"branch_count={branch_count}")
    print(f"frame_count={frame_count}")
    print(output_path)


if __name__ == "__main__":
    main()
