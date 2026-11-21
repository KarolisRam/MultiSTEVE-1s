import argparse
import json
import os
from html import escape
from pathlib import Path

import numpy as np

from checkpoint_viz_common import (
    format_step_compact,
    load_curve_csv,
    load_manifest,
    plot_dimensions,
    pretty_action_name,
    slugify_action_name,
    tick_indices,
    write_plotly_assets,
)

DEFAULT_OUTPUT_SUBDIR = "delta_to_final_heatwall"

APP_JS = """
(function () {
  const page = JSON.parse(document.getElementById("page-config").textContent);
  const heatmapHost = document.getElementById("heatmap-plot");
  const checkpointHost = document.getElementById("checkpoint-delta-plot");
  const frameHost = document.getElementById("frame-delta-plot");
  const scaleSelect = document.getElementById("scale-select");
  const actionSelect = document.getElementById("action-select");
  const hoverSummary = document.getElementById("hover-summary");

  const frames = page.frames;
  const checkpointSteps = page.checkpoint_steps;
  const checkpointLabels = page.checkpoint_labels;
  const delta = page.delta;

  const scaleModes = {
    actual: {
      label: "Actual checkpoint steps",
      axisTitle: "Checkpoint step",
      values: checkpointSteps.slice(),
      tickvals: page.tick_indices.map((index) => checkpointSteps[index]),
      ticktext: page.tick_indices.map((index) => checkpointLabels[index]),
    },
    log: {
      label: "Log checkpoint steps",
      axisTitle: "log10(checkpoint step + 1)",
      values: checkpointSteps.map((step) => Math.log10(step + 1)),
      tickvals: page.tick_indices.map((index) => Math.log10(checkpointSteps[index] + 1)),
      ticktext: page.tick_indices.map((index) => checkpointLabels[index]),
    },
    even: {
      label: "Evenly spaced checkpoints",
      axisTitle: "Checkpoint order",
      values: checkpointSteps.map((_, index) => index),
      tickvals: page.tick_indices.slice(),
      ticktext: page.tick_indices.map((index) => checkpointLabels[index]),
    },
  };

  let currentScale = "actual";
  let currentCheckpointIndex = page.default_checkpoint_index;
  let currentFrameIndex = page.default_frame_index;

  function toComma(value) {
    return Number(value).toLocaleString("en-US");
  }

  function toDelta(value) {
    const number = Number(value);
    const sign = number > 0 ? "+" : "";
    return sign + number.toFixed(4);
  }

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

  function populateScaleSelect() {
    [["actual", "Actual"], ["log", "Log"], ["even", "Evenly spaced"]].forEach(([value, label]) => {
      const option = document.createElement("option");
      option.value = value;
      option.textContent = label;
      if (value === currentScale) {
        option.selected = true;
      }
      scaleSelect.appendChild(option);
    });
    scaleSelect.addEventListener("change", () => {
      currentScale = scaleSelect.value;
      renderHeatmap();
      renderFrameSlice();
      updateHoverSummary();
    });
  }

  function heatmapLayout() {
    const mode = scaleModes[currentScale];
    return {
      autosize: true,
      height: page.heatmap_height,
      paper_bgcolor: "#fffaf2",
      plot_bgcolor: "#fffaf2",
      margin: {l: 60, r: 18, t: 14, b: 48},
      font: {family: "IBM Plex Sans, Arial, sans-serif", size: 14, color: "#1f1d19"},
      xaxis: {
        title: "Rollout frame",
        gridcolor: "rgba(0,0,0,0.06)",
      },
      yaxis: {
        title: mode.axisTitle,
        tickmode: "array",
        tickvals: mode.tickvals,
        ticktext: mode.ticktext,
        gridcolor: "rgba(0,0,0,0.06)",
      },
    };
  }

  function renderHeatmap() {
    const mode = scaleModes[currentScale];
    const customdata = delta.map((series, checkpointIndex) =>
      series.map((value, frameIndex) => [checkpointSteps[checkpointIndex], checkpointIndex, frames[frameIndex]])
    );
    const trace = {
      type: "heatmap",
      x: frames,
      y: mode.values,
      z: delta,
      zmid: 0,
      zmin: -page.delta_limit,
      zmax: page.delta_limit,
      colorscale: page.colorscale,
      customdata: customdata,
      colorbar: {
        title: "Δ to final",
        thickness: 16,
        len: 0.76,
      },
      hovertemplate:
        "Frame %{customdata[2]:,.0f}<br>" +
        "Checkpoint %{customdata[0]:,.0f}<br>" +
        "Δ to final %{z:+.4f}<extra></extra>",
    };
    Plotly.react(heatmapHost, [trace], heatmapLayout(), {displayModeBar: false, responsive: true});
  }

  function checkpointLayout() {
    return {
      autosize: true,
      height: page.slice_height,
      paper_bgcolor: "#fffaf2",
      plot_bgcolor: "#fffaf2",
      margin: {l: 54, r: 20, t: 54, b: 44},
      font: {family: "IBM Plex Sans, Arial, sans-serif", size: 13, color: "#1f1d19"},
      xaxis: {
        title: "Frame",
        gridcolor: "rgba(0,0,0,0.08)",
      },
      yaxis: {
        title: "Δ to final",
        range: [-page.delta_limit, page.delta_limit],
        gridcolor: "rgba(0,0,0,0.08)",
      },
      title: {
        text: "Checkpoint slice: " + checkpointLabels[currentCheckpointIndex] + " (" + toComma(checkpointSteps[currentCheckpointIndex]) + ")",
        x: 0.02,
        xanchor: "left",
        font: {size: 15},
      },
      showlegend: false,
    };
  }

  function renderCheckpointSlice() {
    const y = delta[currentCheckpointIndex];
    const traces = [
      {
        type: "scatter",
        mode: "lines",
        x: frames,
        y: y,
        line: {color: "#1f556c", width: 2.4},
        hovertemplate: "Frame %{x:.0f}<br>Δ to final %{y:+.4f}<extra></extra>",
      },
      {
        type: "scatter",
        mode: "lines",
        x: [frames[0], frames[frames.length - 1]],
        y: [0, 0],
        line: {color: "rgba(0,0,0,0.28)", width: 1, dash: "dash"},
        hoverinfo: "skip",
      },
      {
        type: "scatter",
        mode: "markers",
        x: [frames[currentFrameIndex]],
        y: [y[currentFrameIndex]],
        marker: {color: "#d86c45", size: 9, line: {color: "#ffffff", width: 1.2}},
        hovertemplate: "Selected frame %{x:.0f}<br>Δ %{y:+.4f}<extra></extra>",
      },
    ];
    Plotly.react(checkpointHost, traces, checkpointLayout(), {displayModeBar: false, responsive: true});
  }

  function frameLayout() {
    const mode = scaleModes[currentScale];
    return {
      autosize: true,
      height: page.slice_height,
      paper_bgcolor: "#fffaf2",
      plot_bgcolor: "#fffaf2",
      margin: {l: 56, r: 20, t: 54, b: 52},
      font: {family: "IBM Plex Sans, Arial, sans-serif", size: 13, color: "#1f1d19"},
      xaxis: {
        title: mode.axisTitle,
        tickmode: "array",
        tickvals: mode.tickvals,
        ticktext: mode.ticktext,
        gridcolor: "rgba(0,0,0,0.08)",
      },
      yaxis: {
        title: "Δ to final",
        range: [-page.delta_limit, page.delta_limit],
        gridcolor: "rgba(0,0,0,0.08)",
      },
      title: {
        text: "Frame slice: " + toComma(frames[currentFrameIndex]) + " across checkpoints",
        x: 0.02,
        xanchor: "left",
        font: {size: 15},
      },
      showlegend: false,
    };
  }

  function renderFrameSlice() {
    const mode = scaleModes[currentScale];
    const y = delta.map((series) => series[currentFrameIndex]);
    const traces = [
      {
        type: "scatter",
        mode: "lines+markers",
        x: mode.values,
        y: y,
        customdata: checkpointSteps,
        marker: {size: 6, color: "#33414d"},
        line: {width: 2.2, color: "#33414d"},
        hovertemplate: "Checkpoint %{customdata:,.0f}<br>Δ %{y:+.4f}<extra></extra>",
      },
      {
        type: "scatter",
        mode: "lines",
        x: [mode.values[0], mode.values[mode.values.length - 1]],
        y: [0, 0],
        line: {color: "rgba(0,0,0,0.28)", width: 1, dash: "dash"},
        hoverinfo: "skip",
      },
      {
        type: "scatter",
        mode: "markers",
        x: [mode.values[currentCheckpointIndex]],
        y: [y[currentCheckpointIndex]],
        customdata: [checkpointSteps[currentCheckpointIndex]],
        marker: {color: "#d86c45", size: 10, line: {color: "#ffffff", width: 1.2}},
        hovertemplate: "Selected checkpoint %{customdata:,.0f}<br>Δ %{y:+.4f}<extra></extra>",
      },
    ];
    Plotly.react(frameHost, traces, frameLayout(), {displayModeBar: false, responsive: true});
  }

  function updateHoverSummary() {
    const value = delta[currentCheckpointIndex][currentFrameIndex];
    hoverSummary.innerHTML = [
      ["Scale", scaleModes[currentScale].label],
      ["Checkpoint", checkpointLabels[currentCheckpointIndex] + " (" + toComma(checkpointSteps[currentCheckpointIndex]) + ")"],
      ["Frame", toComma(frames[currentFrameIndex])],
      ["Δ to final", toDelta(value)],
      ["Sign", value > 0 ? "Above final" : (value < 0 ? "Below final" : "At final")],
    ]
      .map(([key, val]) => '<div class="summary-row"><span class="summary-key">' + key + '</span><span class="summary-value">' + val + '</span></div>')
      .join("");
  }

  heatmapHost.on = heatmapHost.on || function () {};

  function handleHover(eventData) {
    if (!eventData || !eventData.points || !eventData.points.length) {
      return;
    }
    const point = eventData.points[0];
    currentCheckpointIndex = Number(point.customdata[1]);
    currentFrameIndex = Number(point.customdata[2]);
    renderCheckpointSlice();
    renderFrameSlice();
    updateHoverSummary();
  }

  populateActionSelect();
  populateScaleSelect();
  renderHeatmap();
  renderCheckpointSlice();
  renderFrameSlice();
  updateHoverSummary();
  heatmapHost.on("plotly_hover", handleHover);
  window.addEventListener("resize", () => {
    Plotly.Plots.resize(heatmapHost);
    Plotly.Plots.resize(checkpointHost);
    Plotly.Plots.resize(frameHost);
  });
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
.left-column { min-width: 0; }
.cross-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 18px;
  margin-bottom: 18px;
}
.plot-card, .main-card {
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
.hover-summary {
  border: 1px solid var(--line);
  border-radius: 16px;
  background: rgba(255, 255, 255, 0.58);
  padding: 12px 14px 8px;
}
.summary-row {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  padding: 5px 0;
  border-bottom: 1px solid rgba(0, 0, 0, 0.05);
}
.summary-row:last-child { border-bottom: 0; }
.summary-key { color: var(--muted); }
.summary-value { font-weight: 600; text-align: right; }
.note {
  margin: 0;
  color: var(--muted);
  line-height: 1.46;
}
@media (max-width: 1320px) {
  .hero, .workspace { grid-template-columns: 1fr; }
  .sidebar { position: static; }
}
@media (max-width: 980px) {
  .cross-grid { grid-template-columns: 1fr; }
}
"""


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


def page_payload(
    action_name,
    category_name,
    frames,
    checkpoint_steps,
    series,
    heatmap_height,
    slice_height,
    action_groups,
    current_output_path,
    gallery_path,
):
    final_series = series[-1]
    delta = series - final_series[None, :]
    return {
        "action_pretty": pretty_action_name(action_name),
        "category_name": category_name,
        "current_action_href": os.path.relpath(current_output_path, current_output_path.parent),
        "gallery_href": os.path.relpath(gallery_path, current_output_path.parent),
        "frames": frames.tolist(),
        "checkpoint_steps": checkpoint_steps.tolist(),
        "checkpoint_labels": [format_step_compact(int(step)) for step in checkpoint_steps],
        "delta": np.round(delta, 10).tolist(),
        "delta_limit": float(np.abs(delta).max()),
        "tick_indices": tick_indices(len(checkpoint_steps)),
        "default_checkpoint_index": len(checkpoint_steps) - 1,
        "default_frame_index": int(np.argmax(np.abs(delta[:-1]).max(axis=0))),
        "action_groups": action_groups,
        "heatmap_height": heatmap_height,
        "slice_height": slice_height,
        "colorscale": [
            [0.00, "#17324d"],
            [0.20, "#236c8a"],
            [0.50, "#f8f4ee"],
            [0.80, "#f0a15d"],
            [1.00, "#b74b31"],
        ],
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
        <p class="subtitle">Blue means this checkpoint assigns less probability than the final policy at that frame. Ember means it overshoots the final policy. White means it has already converged there.</p>
      </section>
      <aside class="card hero-meta">
        <a class="back-link" href="{escape(payload["gallery_href"])}">Back to heat-wall gallery</a>
        <div class="meta-box">
          <p class="meta-kicker">Why This Works</p>
          <p class="meta-copy">The sign of the difference matters. A plain absolute-distance plot hides whether training is still under-confident, over-confident, or oscillating around the final policy.</p>
        </div>
      </aside>
    </header>
    <section class="workspace">
      <div class="left-column">
        <div class="cross-grid">
          <article class="card plot-card">
            <div class="plot-topline">
              <h2 class="plot-title">Checkpoint Slice</h2>
              <p class="plot-caption">One checkpoint across all frames</p>
            </div>
            <div class="plot-frame"><div id="checkpoint-delta-plot"></div></div>
          </article>
          <article class="card plot-card">
            <div class="plot-topline">
              <h2 class="plot-title">Frame Slice</h2>
              <p class="plot-caption">One frame across all checkpoints</p>
            </div>
            <div class="plot-frame"><div id="frame-delta-plot"></div></div>
          </article>
        </div>
        <article class="card main-card">
          <div class="plot-topline">
            <h2 class="plot-title">Signed Delta-To-Final Heat Wall</h2>
            <p class="plot-caption">Hover to lock the two slices above</p>
          </div>
          <div class="plot-frame"><div id="heatmap-plot"></div></div>
        </article>
      </div>
      <aside class="card sidebar">
        <section>
          <p class="section-label">Action</p>
          <select id="action-select"></select>
        </section>
        <section>
          <p class="section-label">Checkpoint Scale</p>
          <select id="scale-select"></select>
        </section>
        <section>
          <p class="section-label">Current Hover</p>
          <div class="hover-summary" id="hover-summary"></div>
        </section>
        <section>
          <p class="section-label">Interpretation</p>
          <p class="note">Use <strong>Actual</strong> to preserve real training step spacing. Use <strong>Log</strong> to open up early learning. Use <strong>Evenly spaced</strong> when you care about checkpoint-to-checkpoint pattern changes more than time.</p>
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
  <title>Delta-To-Final Heat Wall Gallery</title>
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
        <h1>Delta-To-Final Heat Wall</h1>
        <p>This view makes the sign of the training error visible. Blue is below-final probability, ember is above-final probability, and white is already converged.</p>
      </div>
      <div class="panel">
        <p><strong>Suggested first look:</strong> open <a href="buttons/inventory.html">Inventory</a> and toggle between Actual and Log checkpoint scales. Late corrections and early overshoot read very differently there.</p>
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
    heatmap_height = max(460, int(height_px * 0.54))
    slice_height = max(220, int(height_px * 0.235))
    assets_dir = write_plotly_assets(output_dir, extra_js=APP_JS, css_text=CSS)

    button_specs = [(name, entry, button_dir / f"{slugify_action_name(name)}.html") for name, entry in sorted(manifest["button_plots"].items())]
    camera_specs = [(name, entry, camera_dir / f"{slugify_action_name(name)}.html") for name, entry in sorted(manifest["camera_plots"].items())]

    output_manifest = {
        "source_manifest": str(manifest_path),
        "output_dir": str(output_dir),
        "assets_dir": str(assets_dir),
        "buttons": {},
        "camera": {},
        "plot_dimensions_pixels": {"width": width_px, "height": height_px},
    }

    gallery_path = output_dir / "index.html"

    for action_name, entry, output_path in button_specs:
        frames, checkpoint_steps, series = load_curve_csv(entry["csv"])
        action_groups = build_action_groups(button_specs, camera_specs, output_path)
        payload = page_payload(
            action_name=action_name,
            category_name="Delta-To-Final Heat Wall",
            frames=frames,
            checkpoint_steps=checkpoint_steps,
            series=series,
            heatmap_height=heatmap_height,
            slice_height=slice_height,
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
            category_name="Delta-To-Final Heat Wall",
            frames=frames,
            checkpoint_steps=checkpoint_steps,
            series=series,
            heatmap_height=heatmap_height,
            slice_height=slice_height,
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
