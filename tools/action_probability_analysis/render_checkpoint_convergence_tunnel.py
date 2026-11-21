import argparse
import json
import os
from html import escape
from pathlib import Path

import numpy as np

from checkpoint_viz_common import (
    format_step_compact,
    guide_indices,
    load_curve_csv,
    load_manifest,
    plot_dimensions,
    pretty_action_name,
    slugify_action_name,
    tick_indices,
    write_plotly_assets,
)

DEFAULT_OUTPUT_SUBDIR = "convergence_tunnel"
FRAME_DOWNSAMPLE = 4
ANIMATION_FRAMES = 84

APP_JS = """
(function () {
  const page = JSON.parse(document.getElementById("page-config").textContent);
  const host = document.getElementById("tunnel-plot");
  const actionSelect = document.getElementById("action-select");
  const playButton = document.getElementById("play-button");
  const pauseButton = document.getElementById("pause-button");
  const cameraLabel = document.getElementById("camera-label");
  const statusBox = document.getElementById("status-box");

  const frames = page.frames;
  const checkpointSteps = page.checkpoint_steps;
  const checkpointLabels = page.checkpoint_labels;
  const probabilities = page.probabilities;
  const yValues = page.y_values;
  const animation = page.animation;
  const guideIndices = page.guide_indices;
  const guideSpecs = page.guide_specs;
  const finalSeries = probabilities[probabilities.length - 1];

  function populateActionSelect() {
    for (const [groupName, entries] of Object.entries(page.action_groups)) {
      const optgroup = document.createElement("optgroup");
      optgroup.label = groupName === "buttons" ? "Button actions" : "Camera actions";
      entries.forEach((entry) => {
        const option = document.createElement("option");
        option.value = entry.href;
        option.textContent = entry.label;
        if (entry.href === page.current_action_href) {
          option.selected = true;
        }
        optgroup.appendChild(option);
      });
      actionSelect.appendChild(optgroup);
    }
    actionSelect.addEventListener("change", () => {
      window.location.href = actionSelect.value;
    });
  }

  function buildSurfaceTrace() {
    const absDelta = probabilities.map((series) => series.map((value, frameIndex) => Math.abs(value - finalSeries[frameIndex])));
    const xGrid = checkpointSteps.map(() => frames.slice());
    const yGrid = yValues.map((y) => Array(frames.length).fill(y));
    const customdata = probabilities.map((series, checkpointIndex) =>
      series.map((value, frameIndex) => [checkpointSteps[checkpointIndex], frames[frameIndex], absDelta[checkpointIndex][frameIndex]])
    );
    return {
      type: "surface",
      x: xGrid,
      y: yGrid,
      z: probabilities,
      surfacecolor: absDelta,
      cmin: 0,
      cmax: Math.max(page.abs_delta_max, 1e-9),
      colorscale: page.surface_colorscale,
      opacity: 0.88,
      customdata: customdata,
      contours: {
        z: {
          show: true,
          usecolormap: false,
          color: "rgba(0,0,0,0.16)",
          width: 1,
        },
      },
      colorbar: {
        title: "|p - p_final|",
        thickness: 15,
        len: 0.74,
      },
      hovertemplate:
        "Frame %{customdata[1]:,.0f}<br>" +
        "Checkpoint %{customdata[0]:,.0f}<br>" +
        "Probability %{z:.4f}<br>" +
        "|Δ to final| %{customdata[2]:.4f}<extra></extra>",
      name: "Probability surface",
    };
  }

  function buildGuideTraces() {
    return guideIndices.map((checkpointIndex, index) => ({
      type: "scatter3d",
      mode: "lines",
      x: frames,
      y: Array(frames.length).fill(yValues[checkpointIndex]),
      z: probabilities[checkpointIndex],
      line: {
        color: guideSpecs[index].color,
        width: guideSpecs[index].width,
      },
      name: guideSpecs[index].label + " checkpoint (" + checkpointLabels[checkpointIndex] + ")",
      hovertemplate:
        "Frame %{x:.0f}<br>" +
        "Checkpoint " + checkpointSteps[checkpointIndex].toLocaleString("en-US") + "<br>" +
        "Probability %{z:.4f}<extra></extra>",
    }));
  }

  function buildHeroTrace(frameIndex) {
    return {
      type: "scatter3d",
      mode: "lines",
      x: animation[frameIndex].x,
      y: animation[frameIndex].y,
      z: animation[frameIndex].z,
      line: {
        color: "#d86c45",
        width: 10,
      },
      name: "Animated checkpoint",
      hovertemplate:
        "Interpolated checkpoint %{customdata[0]:,.0f}<br>" +
        "Frame %{x:.0f}<br>" +
        "Probability %{z:.4f}<extra></extra>",
      customdata: animation[frameIndex].customdata,
    };
  }

  function layoutForFrame(frameIndex) {
    const animationFrame = animation[frameIndex];
    cameraLabel.textContent = "Camera path: " + animationFrame.camera_label;
    statusBox.innerHTML =
      '<div class="status-row"><span class="status-key">Checkpoint</span><span class="status-value">' +
      animationFrame.step_label +
      " (" +
      animationFrame.step.toLocaleString("en-US") +
      ")</span></div>" +
      '<div class="status-row"><span class="status-key">Frame in animation</span><span class="status-value">' +
      (frameIndex + 1) +
      " / " +
      animation.length +
      "</span></div>" +
      '<div class="status-row"><span class="status-key">Interpolation</span><span class="status-value">' +
      animationFrame.interpolation_label +
      "</span></div>";

    return {
      autosize: true,
      height: page.plot_height,
      paper_bgcolor: "#fffaf2",
      plot_bgcolor: "#fffaf2",
      margin: {l: 0, r: 0, t: 10, b: 0},
      font: {family: "IBM Plex Sans, Arial, sans-serif", size: 14, color: "#1f1d19"},
      legend: {
        bgcolor: "rgba(255,250,242,0.72)",
        bordercolor: "rgba(0,0,0,0.08)",
        borderwidth: 1,
        x: 0.01,
        y: 0.99,
      },
      scene: {
        bgcolor: "#fffaf2",
        camera: animationFrame.camera,
        aspectmode: "manual",
        aspectratio: {x: 1.85, y: 1.10, z: 0.88},
        xaxis: {
          title: "Rollout frame",
          backgroundcolor: "rgba(255,255,255,0)",
          gridcolor: "rgba(0,0,0,0.10)",
          zerolinecolor: "rgba(0,0,0,0.18)",
        },
        yaxis: {
          title: "Checkpoint step",
          tickmode: "array",
          tickvals: page.tick_y_values,
          ticktext: page.tick_labels,
          backgroundcolor: "rgba(255,255,255,0)",
          gridcolor: "rgba(0,0,0,0.10)",
          zerolinecolor: "rgba(0,0,0,0.18)",
        },
        zaxis: {
          title: "Probability",
          range: [0, page.zmax],
          backgroundcolor: "rgba(255,255,255,0)",
          gridcolor: "rgba(0,0,0,0.10)",
          zerolinecolor: "rgba(0,0,0,0.18)",
        },
      },
    };
  }

  let currentFrameIndex = 0;
  let timerId = null;
  let plotReady = false;

  function renderFrame(frameIndex) {
    currentFrameIndex = frameIndex;
    const data = [buildSurfaceTrace()].concat(buildGuideTraces()).concat([buildHeroTrace(frameIndex)]);
    const layout = layoutForFrame(frameIndex);
    if (!plotReady) {
      Plotly.newPlot(host, data, layout, {displayModeBar: true, responsive: true});
      plotReady = true;
    } else {
      Plotly.react(host, data, layout, {displayModeBar: true, responsive: true});
    }
  }

  function stopAnimation() {
    if (timerId !== null) {
      window.clearInterval(timerId);
      timerId = null;
    }
  }

  function startAnimation(restart) {
    if (restart) {
      currentFrameIndex = 0;
      renderFrame(currentFrameIndex);
    }
    stopAnimation();
    timerId = window.setInterval(() => {
      currentFrameIndex += 1;
      if (currentFrameIndex >= animation.length) {
        stopAnimation();
        currentFrameIndex = animation.length - 1;
        renderFrame(currentFrameIndex);
        return;
      }
      renderFrame(currentFrameIndex);
    }, page.frame_interval_ms);
  }

  populateActionSelect();
  renderFrame(0);
  playButton.addEventListener("click", () => startAnimation(true));
  pauseButton.addEventListener("click", () => stopAnimation());
  window.addEventListener("resize", () => {
    if (plotReady) {
      Plotly.Plots.resize(host);
    }
  });
  window.setTimeout(() => startAnimation(true), 300);
})();
"""

CSS = """
:root {
  --bg: #f6f1e8;
  --panel: rgba(255, 250, 242, 0.92);
  --ink: #1f1d19;
  --muted: #6c6459;
  --line: rgba(0, 0, 0, 0.10);
  --accent: #1f556c;
  --accent-2: #d86c45;
  --shadow: 0 14px 40px rgba(0, 0, 0, 0.08);
}
* { box-sizing: border-box; }
body {
  margin: 0;
  color: var(--ink);
  font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
  background:
    radial-gradient(circle at top left, rgba(31, 85, 108, 0.10), transparent 28%),
    radial-gradient(circle at bottom right, rgba(216, 108, 69, 0.12), transparent 24%),
    var(--bg);
}
.shell {
  max-width: 1710px;
  margin: 0 auto;
  padding: 24px 22px 28px;
}
.hero {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 320px;
  gap: 18px;
  margin-bottom: 18px;
}
.card {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 22px;
  box-shadow: var(--shadow);
}
.hero-copy {
  padding: 20px 22px 18px;
}
.eyebrow {
  color: var(--accent);
  font-size: 0.82rem;
  font-weight: 700;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  margin-bottom: 8px;
}
h1 {
  margin: 0 0 10px;
  font-size: 2.35rem;
  line-height: 0.98;
  letter-spacing: -0.04em;
}
.subtitle {
  margin: 0;
  color: var(--muted);
  font-size: 1.02rem;
  line-height: 1.45;
}
.hero-meta {
  padding: 18px 20px;
  display: flex;
  flex-direction: column;
  gap: 14px;
}
.back-link {
  display: inline-flex;
  width: fit-content;
  text-decoration: none;
  color: var(--accent);
  font-weight: 600;
}
.meta-box {
  border: 1px solid var(--line);
  border-radius: 16px;
  background: rgba(255, 255, 255, 0.58);
  padding: 14px 16px;
}
.meta-kicker {
  margin: 0 0 6px;
  font-size: 0.76rem;
  font-weight: 700;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--muted);
}
.meta-copy {
  margin: 0;
  line-height: 1.44;
}
.workspace {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 320px;
  gap: 18px;
}
.plot-card {
  padding: 14px;
}
.plot-topline {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  align-items: baseline;
  padding: 0 4px 6px;
}
.plot-title {
  margin: 0;
  font-size: 1.02rem;
}
.plot-caption {
  margin: 0;
  color: var(--muted);
  font-size: 0.88rem;
}
.plot-frame {
  border-radius: 16px;
  overflow: hidden;
  background: rgba(255, 255, 255, 0.98);
  border: 1px solid rgba(0, 0, 0, 0.06);
}
.sidebar {
  padding: 18px;
  display: flex;
  flex-direction: column;
  gap: 18px;
  position: sticky;
  top: 18px;
  align-self: start;
}
.section-label {
  margin: 0 0 8px;
  font-size: 0.8rem;
  font-weight: 700;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--muted);
}
select {
  width: 100%;
  border: 1px solid rgba(0, 0, 0, 0.12);
  border-radius: 12px;
  padding: 12px 14px;
  font: inherit;
  color: var(--ink);
  background: rgba(255, 255, 255, 0.9);
}
.control-buttons {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px;
}
button {
  border: 1px solid rgba(0, 0, 0, 0.10);
  border-radius: 12px;
  background: rgba(255, 255, 255, 0.82);
  color: var(--ink);
  font: inherit;
  padding: 11px 12px;
  cursor: pointer;
}
button:hover {
  border-color: rgba(31, 85, 108, 0.30);
}
.status-box {
  border: 1px solid var(--line);
  border-radius: 16px;
  background: rgba(255, 255, 255, 0.58);
  padding: 12px 14px 8px;
}
.status-row {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  padding: 5px 0;
  border-bottom: 1px solid rgba(0, 0, 0, 0.05);
}
.status-row:last-child { border-bottom: 0; }
.status-key { color: var(--muted); }
.status-value { font-weight: 600; text-align: right; }
.note {
  margin: 0;
  color: var(--muted);
  line-height: 1.46;
}
@media (max-width: 1320px) {
  .hero, .workspace { grid-template-columns: 1fr; }
  .sidebar { position: static; }
}
"""


def downsample_frames(frames, series, step=FRAME_DOWNSAMPLE):
    indices = list(range(0, len(frames), step))
    if indices[-1] != len(frames) - 1:
        indices.append(len(frames) - 1)
    indices = np.array(indices, dtype=int)
    return frames[indices], series[:, indices]


def build_action_groups(button_specs, camera_specs, current_output_path):
    action_groups = {"buttons": [], "camera": []}
    for action_name, _, output_path in button_specs:
        action_groups["buttons"].append(
            {"label": f"Button: {pretty_action_name(action_name)}", "href": os.path.relpath(output_path, current_output_path.parent)}
        )
    for action_name, _, output_path in camera_specs:
        action_groups["camera"].append(
            {"label": f"Camera: {pretty_action_name(action_name)}", "href": os.path.relpath(output_path, current_output_path.parent)}
        )
    return action_groups


def interpolate_series(checkpoint_steps, series, fraction_index):
    lower_index = int(np.floor(fraction_index))
    upper_index = min(len(checkpoint_steps) - 1, lower_index + 1)
    frac = float(fraction_index - lower_index)
    if lower_index == upper_index:
        blended_series = series[lower_index]
        blended_step = checkpoint_steps[lower_index]
    else:
        blended_series = (1.0 - frac) * series[lower_index] + frac * series[upper_index]
        blended_step = (1.0 - frac) * checkpoint_steps[lower_index] + frac * checkpoint_steps[upper_index]
    return blended_series, float(blended_step), lower_index, upper_index, frac


def camera_for_t(fraction):
    start = np.array([2.4, -2.8, 1.55], dtype=float)
    end = np.array([0.86, -0.68, 0.72], dtype=float)
    side_bias = np.array([0.26 * np.sin(np.pi * fraction), -0.18 * np.cos(np.pi * fraction), 0.14 * np.sin(np.pi * fraction)], dtype=float)
    eye = start * (1.0 - fraction) + end * fraction + side_bias
    return {"eye": {"x": float(eye[0]), "y": float(eye[1]), "z": float(eye[2])}}


def build_animation_payload(frames, checkpoint_steps, y_values, series):
    positions = np.linspace(0, len(checkpoint_steps) - 1, ANIMATION_FRAMES)
    payload = []
    for position in positions:
        line_series, step_value, lower_index, upper_index, frac = interpolate_series(checkpoint_steps, series, position)
        y_value = (1.0 - frac) * y_values[lower_index] + frac * y_values[upper_index]
        customdata = [[step_value, int(frame)] for frame in frames]
        if lower_index == upper_index:
            interpolation_label = f"Exact checkpoint {format_step_compact(int(checkpoint_steps[lower_index]))}"
        else:
            interpolation_label = (
                f"{format_step_compact(int(checkpoint_steps[lower_index]))} -> "
                f"{format_step_compact(int(checkpoint_steps[upper_index]))} ({frac:.2f})"
            )
        fraction = float(position / max(1, len(checkpoint_steps) - 1))
        payload.append(
            {
                "x": frames.tolist(),
                "y": [float(y_value)] * len(frames),
                "z": np.round(line_series, 10).tolist(),
                "customdata": customdata,
                "step": float(step_value),
                "step_label": format_step_compact(int(round(step_value))),
                "camera": camera_for_t(fraction),
                "camera_label": f"{int(round(fraction * 100))}% through tunnel",
                "interpolation_label": interpolation_label,
            }
        )
    return payload


def page_payload(
    action_name,
    category_name,
    frames,
    checkpoint_steps,
    series,
    plot_height,
    action_groups,
    current_output_path,
    gallery_path,
):
    frames_ds, series_ds = downsample_frames(frames, series)
    tick_idx = tick_indices(len(checkpoint_steps))
    guide_idx = guide_indices(len(checkpoint_steps))
    y_values = checkpoint_steps.astype(float)
    return {
        "action_pretty": pretty_action_name(action_name),
        "category_name": category_name,
        "current_action_href": os.path.relpath(current_output_path, current_output_path.parent),
        "gallery_href": os.path.relpath(gallery_path, current_output_path.parent),
        "frames": frames_ds.tolist(),
        "checkpoint_steps": checkpoint_steps.tolist(),
        "checkpoint_labels": [format_step_compact(int(step)) for step in checkpoint_steps],
        "probabilities": np.round(series_ds, 10).tolist(),
        "y_values": y_values.tolist(),
        "tick_y_values": [float(y_values[index]) for index in tick_idx],
        "tick_labels": [format_step_compact(int(checkpoint_steps[index])) for index in tick_idx],
        "guide_indices": guide_idx,
        "guide_specs": [
            {"label": label, "color": color, "width": width}
            for (label, color, width) in [
                ("Initial", "#8a8a8a", 3),
                ("25%", "#64847d", 3),
                ("50%", "#4c6278", 4),
                ("75%", "#2e3d4f", 4),
                ("Final", "#000000", 8),
            ][: len(guide_idx)]
        ],
        "action_groups": action_groups,
        "surface_colorscale": [
            [0.00, "#17324d"],
            [0.25, "#236c8a"],
            [0.55, "#7db8a5"],
            [0.78, "#f5c46b"],
            [1.00, "#d86c45"],
        ],
        "abs_delta_max": float(np.abs(series_ds - series_ds[-1][None, :]).max()),
        "zmax": min(1.02, max(0.08, float(series_ds.max()) * 1.03)),
        "plot_height": plot_height,
        "frame_interval_ms": 90,
        "animation": build_animation_payload(frames_ds, checkpoint_steps, y_values, series_ds),
    }


def write_action_page(output_path, plotly_js_rel, app_js_rel, css_rel, payload):
    title = f"{payload['category_name']}: {payload['action_pretty']}"
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(title)}</title>
  <link rel="stylesheet" href="{escape(css_rel)}">
</head>
<body>
  <div class="shell">
    <header class="hero">
      <section class="card hero-copy">
        <div class="eyebrow">{escape(payload["category_name"])}</div>
        <h1>{escape(payload["action_pretty"])}</h1>
        <p class="subtitle">This is the tunnel view: a moving camera flies from the earliest checkpoint wall toward the final policy while an ember guide rail sweeps through training. It is meant to make phase changes feel spatial instead of stacked.</p>
      </section>
      <aside class="card hero-meta">
        <a class="back-link" href="{escape(payload["gallery_href"])}">Back to tunnel gallery</a>
        <div class="meta-box">
          <p class="meta-kicker">How To Read It</p>
          <p class="meta-copy">The surface stays fixed. The orange line is the moving training frontier. The thick black line is the converged final checkpoint.</p>
        </div>
      </aside>
    </header>
    <section class="workspace">
      <article class="card plot-card">
        <div class="plot-topline">
          <h2 class="plot-title">Convergence Tunnel</h2>
          <p class="plot-caption" id="camera-label">Camera path</p>
        </div>
        <div class="plot-frame"><div id="tunnel-plot"></div></div>
      </article>
      <aside class="card sidebar">
        <section>
          <p class="section-label">Action</p>
          <select id="action-select"></select>
        </section>
        <section>
          <p class="section-label">Playback</p>
          <div class="control-buttons">
            <button id="play-button" type="button">Restart</button>
            <button id="pause-button" type="button">Pause</button>
          </div>
        </section>
        <section>
          <p class="section-label">Current State</p>
          <div class="status-box" id="status-box"></div>
        </section>
        <section>
          <p class="section-label">Why This Exists</p>
          <p class="note">The static surface is useful, but the camera motion makes cliffs, late jumps, and smooth convergence ramps feel more legible. It is the cinematic version of the 3D plot.</p>
        </section>
      </aside>
    </section>
  </div>
  <script src="{escape(plotly_js_rel)}"></script>
  <script id="page-config" type="application/json">{json.dumps(payload, separators=(",", ":"))}</script>
  <script src="{escape(app_js_rel)}"></script>
</body>
</html>
"""
    output_path.write_text(html)


def write_index(output_dir, button_outputs, camera_outputs):
    button_links = "\n".join(
        f'<a class="tile" href="buttons/{escape(filename)}"><span>{escape(pretty_action_name(label))}</span><span class="meta">button</span></a>'
        for label, filename in button_outputs
    )
    camera_links = "\n".join(
        f'<a class="tile" href="camera/{escape(filename)}"><span>{escape(pretty_action_name(label))}</span><span class="meta">camera</span></a>'
        for label, filename in camera_outputs
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Convergence Tunnel Gallery</title>
  <style>
    body {{
      margin: 0;
      font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
      background:
        radial-gradient(circle at top left, rgba(31, 85, 108, 0.10), transparent 28%),
        radial-gradient(circle at bottom right, rgba(216, 108, 69, 0.12), transparent 24%),
        #f6f1e8;
      color: #1f1d19;
    }}
    main {{
      max-width: 1520px;
      margin: 0 auto;
      padding: 30px 26px 42px;
    }}
    .panel {{
      background: rgba(255, 250, 242, 0.92);
      border: 1px solid rgba(0,0,0,0.10);
      border-radius: 22px;
      padding: 20px 22px;
      box-shadow: 0 14px 40px rgba(0, 0, 0, 0.08);
    }}
    .hero {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) 320px;
      gap: 18px;
      margin-bottom: 24px;
    }}
    h1 {{
      margin: 0 0 10px;
      font-size: 2.35rem;
      letter-spacing: -0.04em;
      line-height: 0.98;
    }}
    p {{
      margin: 0;
      color: #6c6459;
      line-height: 1.46;
    }}
    h2 {{
      margin: 0 0 14px;
      color: #6c6459;
      letter-spacing: 0.10em;
      text-transform: uppercase;
      font-size: 1.04rem;
    }}
    .section {{ margin-top: 22px; }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
      gap: 12px;
    }}
    .tile {{
      display: flex;
      justify-content: space-between;
      gap: 10px;
      padding: 14px 16px;
      border-radius: 14px;
      border: 1px solid rgba(0,0,0,0.10);
      background: rgba(255,255,255,0.82);
      text-decoration: none;
      color: inherit;
    }}
    .meta {{
      color: #6c6459;
      font-size: 0.84rem;
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }}
  </style>
</head>
<body>
  <main>
    <section class="hero">
      <div class="panel">
        <h1>Convergence Tunnel</h1>
        <p>This is the cinematic version of training convergence. Each page autoplays a camera move from the earliest checkpoint wall toward the final policy while an ember guide rail sweeps through training.</p>
      </div>
      <div class="panel">
        <p><strong>Suggested first look:</strong> open <a href="buttons/inventory.html">Inventory</a> or <a href="buttons/attack.html">Attack</a> and let the camera play through once before rotating manually.</p>
      </div>
    </section>
    <section class="section panel">
      <h2>Button Actions</h2>
      <div class="grid">{button_links}</div>
    </section>
    <section class="section panel">
      <h2>Camera Actions</h2>
      <div class="grid">{camera_links}</div>
    </section>
  </main>
</body>
</html>
"""
    index_path = Path(output_dir) / "index.html"
    index_path.write_text(html)
    return index_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_dir", required=True)
    parser.add_argument("--output_dir")
    args = parser.parse_args()

    experiment_dir = Path(args.experiment_dir).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else experiment_dir / DEFAULT_OUTPUT_SUBDIR
    output_dir.mkdir(parents=True, exist_ok=True)
    button_dir = output_dir / "buttons"
    camera_dir = output_dir / "camera"
    button_dir.mkdir(parents=True, exist_ok=True)
    camera_dir.mkdir(parents=True, exist_ok=True)

    manifest, manifest_path = load_manifest(experiment_dir)
    width_px, height_px = plot_dimensions(manifest)
    plot_height = max(520, int(height_px * 0.74))
    assets_dir = write_plotly_assets(output_dir, extra_js=APP_JS, css_text=CSS)

    button_specs = [(name, entry, button_dir / f"{slugify_action_name(name)}.html") for name, entry in sorted(manifest["button_plots"].items())]
    camera_specs = [(name, entry, camera_dir / f"{slugify_action_name(name)}.html") for name, entry in sorted(manifest["camera_plots"].items())]

    output_manifest = {
        "source_manifest": str(manifest_path),
        "output_dir": str(output_dir),
        "assets_dir": str(assets_dir),
        "buttons": {},
        "camera": {},
        "animation_frames": ANIMATION_FRAMES,
        "frame_downsample": FRAME_DOWNSAMPLE,
        "plot_dimensions_pixels": {"width": width_px, "height": plot_height},
    }

    gallery_path = output_dir / "index.html"

    for action_name, entry, output_path in button_specs:
        frames, checkpoint_steps, series = load_curve_csv(entry["csv"])
        action_groups = build_action_groups(button_specs, camera_specs, output_path)
        payload = page_payload(
            action_name=action_name,
            category_name="Convergence Tunnel",
            frames=frames,
            checkpoint_steps=checkpoint_steps,
            series=series,
            plot_height=plot_height,
            action_groups=action_groups,
            current_output_path=output_path,
            gallery_path=gallery_path,
        )
        write_action_page(
            output_path=output_path,
            plotly_js_rel=os.path.relpath(assets_dir / "plotly.min.js", output_path.parent),
            app_js_rel=os.path.relpath(assets_dir / "app.js", output_path.parent),
            css_rel=os.path.relpath(assets_dir / "styles.css", output_path.parent),
            payload=payload,
        )
        output_manifest["buttons"][action_name] = {"html": str(output_path), "source_csv": entry["csv"]}

    for action_name, entry, output_path in camera_specs:
        frames, checkpoint_steps, series = load_curve_csv(entry["csv"])
        action_groups = build_action_groups(button_specs, camera_specs, output_path)
        payload = page_payload(
            action_name=action_name,
            category_name="Convergence Tunnel",
            frames=frames,
            checkpoint_steps=checkpoint_steps,
            series=series,
            plot_height=plot_height,
            action_groups=action_groups,
            current_output_path=output_path,
            gallery_path=gallery_path,
        )
        write_action_page(
            output_path=output_path,
            plotly_js_rel=os.path.relpath(assets_dir / "plotly.min.js", output_path.parent),
            app_js_rel=os.path.relpath(assets_dir / "app.js", output_path.parent),
            css_rel=os.path.relpath(assets_dir / "styles.css", output_path.parent),
            payload=payload,
        )
        output_manifest["camera"][action_name] = {"html": str(output_path), "source_csv": entry["csv"]}

    index_path = write_index(
        output_dir,
        button_outputs=[(name, output_path.name) for name, _, output_path in button_specs],
        camera_outputs=[(name, output_path.name) for name, _, output_path in camera_specs],
    )
    output_manifest["index_html"] = str(index_path)

    (output_dir / "manifest.json").write_text(json.dumps(output_manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
