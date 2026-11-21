import argparse
import csv
import json
import os
from html import escape
from pathlib import Path

import numpy as np
from plotly.offline.offline import get_plotlyjs

DEFAULT_SCREEN_WIDTH = 1900
DEFAULT_SCREEN_HEIGHT = 1080
DEFAULT_OUTPUT_SUBDIR = "interactive_3d"
MAIN_PLOT_HEIGHT_RATIO = 0.66
CROSS_SECTION_HEIGHT_RATIO = 0.235

GUIDE_LINE_SPECS = [
    {"label": "Initial", "color": "#8a8a8a", "width": 3},
    {"label": "25%", "color": "#64847d", "width": 3},
    {"label": "50%", "color": "#4c6278", "width": 4},
    {"label": "75%", "color": "#2e3d4f", "width": 4},
    {"label": "Final", "color": "#000000", "width": 8},
]

CAMERA_PRESETS = [
    {"label": "Perspective", "camera": {"eye": {"x": 1.55, "y": -1.55, "z": 0.95}}},
    {"label": "Top Down", "camera": {"eye": {"x": 0.01, "y": 0.01, "z": 2.20}}},
    {"label": "Training Side", "camera": {"eye": {"x": 0.12, "y": -2.35, "z": 0.56}}},
    {"label": "Frame Side", "camera": {"eye": {"x": 2.35, "y": -0.10, "z": 0.56}}},
]

DASHBOARD_CSS = """
:root {
  --bg: #f6f1e8;
  --panel: rgba(255, 250, 242, 0.92);
  --panel-strong: rgba(255, 255, 255, 0.98);
  --ink: #1f1d19;
  --muted: #6c6459;
  --line: rgba(0, 0, 0, 0.10);
  --accent: #1f556c;
  --accent-2: #d86c45;
  --shadow: 0 14px 40px rgba(0, 0, 0, 0.08);
}
* {
  box-sizing: border-box;
}
body {
  margin: 0;
  color: var(--ink);
  font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
  background:
    radial-gradient(circle at top left, rgba(31, 85, 108, 0.10), transparent 28%),
    radial-gradient(circle at bottom right, rgba(216, 108, 69, 0.12), transparent 24%),
    var(--bg);
}
a {
  color: var(--accent);
}
.shell {
  max-width: 1710px;
  margin: 0 auto;
  padding: 24px 22px 28px;
}
.hero {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 290px;
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
  max-width: 980px;
  color: var(--muted);
  font-size: 1.02rem;
  line-height: 1.45;
}
.hero-meta {
  padding: 18px 20px;
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  gap: 16px;
}
.back-link {
  display: inline-flex;
  width: fit-content;
  text-decoration: none;
  font-weight: 600;
}
.meta-box {
  border: 1px solid var(--line);
  border-radius: 16px;
  background: rgba(255, 255, 255, 0.55);
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
  line-height: 1.42;
  color: var(--ink);
}
.workspace {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 308px;
  gap: 18px;
}
.left-column {
  min-width: 0;
}
.cross-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 18px;
  margin-bottom: 18px;
}
.plot-card {
  padding: 14px 14px 10px;
}
.plot-topline {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 10px;
  padding: 0 4px 6px;
}
.plot-title {
  margin: 0;
  font-size: 1rem;
  letter-spacing: 0.01em;
}
.plot-caption {
  margin: 0;
  color: var(--muted);
  font-size: 0.87rem;
}
.plot-frame {
  border-radius: 16px;
  overflow: hidden;
  background: var(--panel-strong);
  border: 1px solid rgba(0, 0, 0, 0.06);
}
.plot-frame > div {
  width: 100%;
}
.main-card {
  padding: 14px 14px 8px;
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
.sidebar-section {
  display: flex;
  flex-direction: column;
  gap: 10px;
}
.section-label {
  margin: 0;
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
.camera-buttons {
  display: grid;
  gap: 8px;
}
.camera-button {
  text-align: left;
  border: 1px solid rgba(0, 0, 0, 0.10);
  border-radius: 12px;
  background: rgba(255, 255, 255, 0.74);
  color: var(--ink);
  font: inherit;
  padding: 11px 12px;
  cursor: pointer;
  transition: background 120ms ease, border-color 120ms ease, transform 120ms ease;
}
.camera-button:hover {
  transform: translateY(-1px);
  border-color: rgba(31, 85, 108, 0.28);
}
.camera-button.active {
  background: rgba(31, 85, 108, 0.12);
  border-color: rgba(31, 85, 108, 0.42);
}
.hover-summary {
  border: 1px solid var(--line);
  border-radius: 16px;
  background: rgba(255, 255, 255, 0.58);
  padding: 14px 14px 12px;
}
.hover-summary-row {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  padding: 5px 0;
  border-bottom: 1px solid rgba(0, 0, 0, 0.05);
}
.hover-summary-row:last-child {
  border-bottom: 0;
}
.hover-key {
  color: var(--muted);
}
.hover-value {
  font-weight: 600;
  text-align: right;
}
.note {
  margin: 0;
  color: var(--muted);
  line-height: 1.48;
  font-size: 0.93rem;
}
@media (max-width: 1320px) {
  .hero,
  .workspace {
    grid-template-columns: 1fr;
  }
  .sidebar {
    position: static;
  }
}
@media (max-width: 980px) {
  .cross-grid {
    grid-template-columns: 1fr;
  }
  h1 {
    font-size: 1.95rem;
  }
}
"""

APP_JS = """
(function () {
  const configNode = document.getElementById("page-config");
  if (!configNode) {
    return;
  }

  const page = JSON.parse(configNode.textContent);
  const mainPlot = document.getElementById("main-plot");
  const checkpointPlot = document.getElementById("checkpoint-cross-section");
  const framePlot = document.getElementById("frame-cross-section");
  const scaleSelect = document.getElementById("scale-select");
  const actionSelect = document.getElementById("action-select");
  const cameraButtonsHost = document.getElementById("camera-buttons");
  const hoverSummary = document.getElementById("hover-summary");

  const frames = page.frames;
  const checkpointSteps = page.checkpoint_steps;
  const checkpointLabels = page.checkpoint_step_labels;
  const probabilities = page.probabilities;
  const guideIndices = page.guide_indices;
  const guideSpecs = page.guide_specs;
  const cameraPresets = page.camera_presets;

  const finalSeries = probabilities[probabilities.length - 1];
  const absDelta = probabilities.map((series) => series.map((value, frameIndex) => Math.abs(value - finalSeries[frameIndex])));
  const surfaceCustomData = probabilities.map((series, checkpointIndex) =>
    series.map((value, frameIndex) => [checkpointSteps[checkpointIndex], checkpointIndex, frames[frameIndex], absDelta[checkpointIndex][frameIndex]])
  );
  const guideCustomData = probabilities.map((series, checkpointIndex) =>
    series.map((value, frameIndex) => [checkpointSteps[checkpointIndex], checkpointIndex, frames[frameIndex], absDelta[checkpointIndex][frameIndex]])
  );
  const surfaceHoverText = probabilities.map((series, checkpointIndex) =>
    series.map((value, frameIndex) =>
      "Frame " +
      toComma(frames[frameIndex]) +
      "<br>Checkpoint " +
      toComma(checkpointSteps[checkpointIndex]) +
      "<br>Probability " +
      toProbability(value) +
      "<br>|Δ to final| " +
      toProbability(absDelta[checkpointIndex][frameIndex])
    )
  );
  const guideHoverText = probabilities.map((series, checkpointIndex) =>
    series.map((value, frameIndex) =>
      "Frame " +
      toComma(frames[frameIndex]) +
      "<br>Checkpoint " +
      toComma(checkpointSteps[checkpointIndex]) +
      "<br>Probability " +
      toProbability(value) +
      "<br>|Δ to final| " +
      toProbability(absDelta[checkpointIndex][frameIndex])
    )
  );

  const selectedTickIndices = page.tick_indices;
  const scaleModes = {
    actual: {
      label: "Actual checkpoint steps",
      axisTitle: "Checkpoint step",
      values: checkpointSteps.slice(),
      tickvals: selectedTickIndices.map((index) => checkpointSteps[index]),
      ticktext: selectedTickIndices.map((index) => checkpointLabels[index]),
    },
    log: {
      label: "Log checkpoint steps",
      axisTitle: "log10(checkpoint step + 1)",
      values: checkpointSteps.map((step) => Math.log10(step + 1)),
      tickvals: selectedTickIndices.map((index) => Math.log10(checkpointSteps[index] + 1)),
      ticktext: selectedTickIndices.map((index) => checkpointLabels[index]),
    },
    even: {
      label: "Evenly spaced checkpoints",
      axisTitle: "Checkpoint order",
      values: checkpointSteps.map((_, index) => index),
      tickvals: selectedTickIndices.slice(),
      ticktext: selectedTickIndices.map((index) => checkpointLabels[index]),
    },
  };

  let currentScale = page.default_scale;
  let currentCheckpointIndex = page.default_checkpoint_index;
  let currentFrameIndex = page.default_frame_index;
  let currentCamera = cameraPresets[0].camera;
  let mainPlotReady = false;

  function toComma(value) {
    return Number(value).toLocaleString("en-US");
  }

  function toProbability(value) {
    return Number(value).toFixed(4);
  }

  function prettyActionName(name) {
    return name.replace(/_/g, " ").replace(/\\./g, " ");
  }

  function populateScaleSelect() {
    [
      ["actual", "Actual"],
      ["log", "Log"],
      ["even", "Evenly spaced"],
    ].forEach(([value, label]) => {
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
      renderMainPlot();
      renderFrameCrossSection();
      updateHoverSummary();
    });
  }

  function populateActionSelect() {
    for (const [groupName, options] of Object.entries(page.action_groups)) {
      const optgroup = document.createElement("optgroup");
      optgroup.label = groupName === "buttons" ? "Button actions" : "Camera actions";
      options.forEach((entry) => {
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

  function renderCameraButtons() {
    cameraPresets.forEach((preset, index) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "camera-button";
      button.textContent = preset.label;
      if (index === 0) {
        button.classList.add("active");
      }
      button.addEventListener("click", () => {
        currentCamera = preset.camera;
        if (mainPlotReady) {
          Plotly.relayout(mainPlot, {"scene.camera": preset.camera});
        }
        cameraButtonsHost.querySelectorAll(".camera-button").forEach((node) => node.classList.remove("active"));
        button.classList.add("active");
      });
      cameraButtonsHost.appendChild(button);
    });
  }

  function axisRange(values) {
    return [Math.min.apply(null, values), Math.max.apply(null, values)];
  }

  function nearestIndex(values, target) {
    let bestIndex = 0;
    let bestDistance = Infinity;
    for (let index = 0; index < values.length; index += 1) {
      const distance = Math.abs(values[index] - target);
      if (distance < bestDistance) {
        bestDistance = distance;
        bestIndex = index;
      }
    }
    return bestIndex;
  }

  function surfaceTraceForScale(scaleKey) {
    const mode = scaleModes[scaleKey];
    const yGrid = mode.values.map((value) => Array(frames.length).fill(value));
    const xGrid = checkpointSteps.map(() => frames.slice());
    return {
      type: "surface",
      x: xGrid,
      y: yGrid,
      z: probabilities,
      surfacecolor: absDelta,
      cmin: 0,
      cmax: Math.max(page.abs_delta_max, 1e-9),
      colorscale: page.surface_colorscale,
      opacity: 0.92,
      customdata: surfaceCustomData,
      text: surfaceHoverText,
      hoverinfo: "skip",
      contours: {
        z: {
          show: true,
          usecolormap: false,
          color: "rgba(0,0,0,0.18)",
          width: 1,
        },
      },
      colorbar: {
        title: "|p - p_final|",
        thickness: 16,
        len: 0.72,
      },
      name: "Probability surface",
      showscale: true,
    };
  }

  function surfaceProbeTraceForScale(scaleKey) {
    const mode = scaleModes[scaleKey];
    const x = [];
    const y = [];
    const z = [];
    const customdata = [];
    const text = [];
    for (let checkpointIndex = 0; checkpointIndex < probabilities.length; checkpointIndex += 1) {
      for (let frameIndex = 0; frameIndex < frames.length; frameIndex += 1) {
        x.push(frames[frameIndex]);
        y.push(mode.values[checkpointIndex]);
        z.push(probabilities[checkpointIndex][frameIndex]);
        customdata.push(surfaceCustomData[checkpointIndex][frameIndex]);
        text.push(surfaceHoverText[checkpointIndex][frameIndex]);
      }
    }
    return {
      type: "scatter3d",
      mode: "markers",
      x: x,
      y: y,
      z: z,
      customdata: customdata,
      text: text,
      marker: {
        size: 4,
        color: "#000000",
        opacity: 0.001,
      },
      hovertemplate: "%{text}<extra></extra>",
      name: "Hover probes",
      showlegend: false,
    };
  }

  function guideTracesForScale(scaleKey) {
    const mode = scaleModes[scaleKey];
    return guideIndices.map((checkpointIndex, index) => {
      const spec = guideSpecs[index];
      const step = checkpointSteps[checkpointIndex];
      return {
        type: "scatter3d",
        mode: "lines",
        x: frames,
        y: Array(frames.length).fill(mode.values[checkpointIndex]),
        z: probabilities[checkpointIndex],
        customdata: guideCustomData[checkpointIndex],
        text: guideHoverText[checkpointIndex],
        line: {
          color: spec.color,
          width: spec.width,
        },
        name: spec.label + " checkpoint (" + checkpointLabels[checkpointIndex] + ")",
        hoverinfo: "skip",
      };
    });
  }

  function mainLayoutForScale(scaleKey) {
    const mode = scaleModes[scaleKey];
    return {
      autosize: true,
      height: page.main_plot_height,
      paper_bgcolor: "#fffaf2",
      plot_bgcolor: "#fffaf2",
      margin: {l: 0, r: 0, t: 22, b: 0},
      font: {family: "IBM Plex Sans, Arial, sans-serif", size: 14, color: "#1f1d19"},
      legend: {
        bgcolor: "rgba(255,250,242,0.70)",
        bordercolor: "rgba(0,0,0,0.08)",
        borderwidth: 1,
        x: 0.01,
        y: 0.99,
      },
      scene: {
        bgcolor: "#fffaf2",
        camera: currentCamera,
        aspectmode: "manual",
        aspectratio: {x: 1.85, y: 1.10, z: 0.86},
        xaxis: {
          title: "Rollout frame",
          backgroundcolor: "rgba(255,255,255,0)",
          gridcolor: "rgba(0,0,0,0.10)",
          zerolinecolor: "rgba(0,0,0,0.18)",
        },
        yaxis: {
          title: mode.axisTitle,
          tickmode: "array",
          tickvals: mode.tickvals,
          ticktext: mode.ticktext,
          range: axisRange(mode.values),
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

  function renderMainPlot() {
    const data = [surfaceTraceForScale(currentScale)]
      .concat(guideTracesForScale(currentScale))
      .concat([surfaceProbeTraceForScale(currentScale)]);
    const layout = mainLayoutForScale(currentScale);
    if (!mainPlotReady) {
      Plotly.newPlot(mainPlot, data, layout, {displayModeBar: true, responsive: true});
      mainPlot.on("plotly_hover", handleHover);
      mainPlot.on("plotly_relayout", (eventData) => {
        if (eventData && eventData["scene.camera"]) {
          currentCamera = eventData["scene.camera"];
        }
      });
      mainPlotReady = true;
    } else {
      Plotly.react(mainPlot, data, layout, {displayModeBar: true, responsive: true});
    }
  }

  function checkpointCrossSectionLayout() {
    return {
      autosize: true,
      height: page.cross_section_height,
      paper_bgcolor: "#fffaf2",
      plot_bgcolor: "#fffaf2",
      margin: {l: 54, r: 20, t: 54, b: 44},
      font: {family: "IBM Plex Sans, Arial, sans-serif", size: 13, color: "#1f1d19"},
      xaxis: {
        title: "Frame",
        gridcolor: "rgba(0,0,0,0.08)",
        zerolinecolor: "rgba(0,0,0,0.12)",
      },
      yaxis: {
        title: "Probability",
        range: [0, page.zmax],
        gridcolor: "rgba(0,0,0,0.08)",
        zerolinecolor: "rgba(0,0,0,0.12)",
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

  function renderCheckpointCrossSection() {
    const series = probabilities[currentCheckpointIndex];
    const selectedProbability = series[currentFrameIndex];
    const traces = [
      {
        type: "scatter",
        mode: "lines",
        x: frames,
        y: series,
        line: {color: "#1f556c", width: 2.7},
        hovertemplate: "Frame %{x:.0f}<br>Probability %{y:.4f}<extra></extra>",
      },
      {
        type: "scatter",
        mode: "markers",
        x: [frames[currentFrameIndex]],
        y: [selectedProbability],
        marker: {color: "#d86c45", size: 9, line: {color: "#ffffff", width: 1.2}},
        hovertemplate:
          "Selected frame %{x:.0f}<br>" +
          "Probability %{y:.4f}<extra></extra>",
      },
    ];
    Plotly.react(checkpointPlot, traces, checkpointCrossSectionLayout(), {displayModeBar: false, responsive: true});
  }

  function frameCrossSectionLayout() {
    const mode = scaleModes[currentScale];
    return {
      autosize: true,
      height: page.cross_section_height,
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
        zerolinecolor: "rgba(0,0,0,0.12)",
      },
      yaxis: {
        title: "Probability",
        range: [0, page.zmax],
        gridcolor: "rgba(0,0,0,0.08)",
        zerolinecolor: "rgba(0,0,0,0.12)",
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

  function renderFrameCrossSection() {
    const mode = scaleModes[currentScale];
    const series = probabilities.map((checkpointSeries) => checkpointSeries[currentFrameIndex]);
    const xValues = mode.values;
    const selectedX = xValues[currentCheckpointIndex];
    const selectedY = series[currentCheckpointIndex];
    const traces = [
      {
        type: "scatter",
        mode: "lines+markers",
        x: xValues,
        y: series,
        marker: {size: 6, color: "#33414d"},
        line: {width: 2.3, color: "#33414d"},
        hovertemplate:
          "Checkpoint %{customdata:,.0f}<br>" +
          "Probability %{y:.4f}<extra></extra>",
        customdata: checkpointSteps,
      },
      {
        type: "scatter",
        mode: "markers",
        x: [selectedX],
        y: [selectedY],
        marker: {color: "#d86c45", size: 10, line: {color: "#ffffff", width: 1.2}},
        hovertemplate:
          "Selected checkpoint %{customdata:,.0f}<br>" +
          "Probability %{y:.4f}<extra></extra>",
        customdata: [checkpointSteps[currentCheckpointIndex]],
      },
    ];
    Plotly.react(framePlot, traces, frameCrossSectionLayout(), {displayModeBar: false, responsive: true});
  }

  function updateHoverSummary() {
    const probability = probabilities[currentCheckpointIndex][currentFrameIndex];
    const delta = absDelta[currentCheckpointIndex][currentFrameIndex];
    hoverSummary.innerHTML = [
      ["Action", prettyActionName(page.action_name)],
      ["Scale", scaleModes[currentScale].label],
      ["Checkpoint", checkpointLabels[currentCheckpointIndex] + " (" + toComma(checkpointSteps[currentCheckpointIndex]) + ")"],
      ["Frame", toComma(frames[currentFrameIndex])],
      ["Probability", toProbability(probability)],
      ["|Δ to final|", toProbability(delta)],
    ]
      .map(([key, value]) => (
        '<div class="hover-summary-row"><span class="hover-key">' +
        key +
        '</span><span class="hover-value">' +
        value +
        '</span></div>'
      ))
      .join("");
  }

  function handleHover(eventData) {
    if (!eventData || !eventData.points || !eventData.points.length) {
      return;
    }
    const point = eventData.points[0];
    const custom = point.customdata;
    if (custom && custom.length >= 4) {
      currentCheckpointIndex = Number(custom[1]);
      currentFrameIndex = Number(custom[2]);
    } else {
      if (typeof point.x === "number") {
        currentFrameIndex = nearestIndex(frames, Number(point.x));
      }
      if (typeof point.y === "number") {
        currentCheckpointIndex = nearestIndex(scaleModes[currentScale].values, Number(point.y));
      }
    }
    renderCheckpointCrossSection();
    renderFrameCrossSection();
    updateHoverSummary();
  }

  function initialRender() {
    populateScaleSelect();
    populateActionSelect();
    renderCameraButtons();
    renderMainPlot();
    renderCheckpointCrossSection();
    renderFrameCrossSection();
    updateHoverSummary();
  }

  initialRender();
  window.addEventListener("resize", function () {
    Plotly.Plots.resize(mainPlot);
    Plotly.Plots.resize(checkpointPlot);
    Plotly.Plots.resize(framePlot);
  });
})();
"""


def _load_manifest(experiment_dir):
    manifest_path = Path(experiment_dir) / "manifest.json"
    return json.loads(manifest_path.read_text()), manifest_path


def _load_curve_csv(csv_path):
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    frames = data[:, 0].astype(int)
    checkpoint_steps = np.array([int(column.removeprefix("ckpt_")) for column in header[1:]], dtype=int)
    probabilities = data[:, 1:].T
    return frames, checkpoint_steps, probabilities


def _plot_dimensions(manifest):
    dims = manifest.get("plot_dimensions_pixels")
    if dims is None:
        return int(DEFAULT_SCREEN_WIDTH * 0.9), int(DEFAULT_SCREEN_HEIGHT * 0.9)
    return int(dims["width"]), int(dims["height"])


def _slugify(action_name):
    return action_name.replace(".", "_").replace(" ", "_")


def _pretty_action_name(action_name):
    return action_name.replace(".", " ").replace("_", " ").title()


def _format_step_compact(step):
    if step >= 1_000_000:
        return f"{step / 1_000_000:.1f}M"
    if step >= 1_000:
        return f"{step / 1_000:.1f}k"
    return f"{step}"


def _guide_indices(num_checkpoints):
    raw = [
        0,
        round((num_checkpoints - 1) * 0.25),
        round((num_checkpoints - 1) * 0.50),
        round((num_checkpoints - 1) * 0.75),
        num_checkpoints - 1,
    ]
    indices = []
    for index in raw:
        if index not in indices:
            indices.append(index)
    return indices


def _tick_indices(num_checkpoints):
    raw = np.linspace(0, num_checkpoints - 1, num=min(8, num_checkpoints), dtype=int).tolist()
    indices = []
    for index in raw:
        if index not in indices:
            indices.append(index)
    if indices[-1] != num_checkpoints - 1:
        indices.append(num_checkpoints - 1)
    return indices


def _write_assets(output_dir):
    assets_dir = output_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    (assets_dir / "plotly.min.js").write_text(get_plotlyjs())
    (assets_dir / "app.js").write_text(APP_JS)
    (assets_dir / "dashboard.css").write_text(DASHBOARD_CSS)
    return assets_dir


def _build_action_groups(button_specs, camera_specs, current_output_path):
    action_groups = {"buttons": [], "camera": []}
    for action_name, _, output_path in button_specs:
        action_groups["buttons"].append(
            {
                "label": f"Button: {_pretty_action_name(action_name)}",
                "href": os.path.relpath(output_path, current_output_path.parent),
            }
        )
    for action_name, _, output_path in camera_specs:
        action_groups["camera"].append(
            {
                "label": f"Camera: {_pretty_action_name(action_name)}",
                "href": os.path.relpath(output_path, current_output_path.parent),
            }
        )
    return action_groups


def _page_payload(
    action_name,
    category_name,
    frames,
    checkpoint_steps,
    probabilities,
    width_px,
    height_px,
    action_groups,
    current_output_path,
    gallery_path,
):
    guide_indices = _guide_indices(len(checkpoint_steps))
    tick_indices = _tick_indices(len(checkpoint_steps))
    final_series = probabilities[-1]
    default_frame_index = int(np.argmax(final_series))
    return {
        "action_name": action_name,
        "action_pretty": _pretty_action_name(action_name),
        "category_name": category_name,
        "current_action_href": os.path.relpath(current_output_path, current_output_path.parent),
        "gallery_href": os.path.relpath(gallery_path, current_output_path.parent),
        "frames": frames.tolist(),
        "checkpoint_steps": checkpoint_steps.tolist(),
        "checkpoint_step_labels": [_format_step_compact(int(step)) for step in checkpoint_steps],
        "probabilities": np.round(probabilities, 10).tolist(),
        "guide_indices": guide_indices,
        "guide_specs": GUIDE_LINE_SPECS[: len(guide_indices)],
        "tick_indices": tick_indices,
        "camera_presets": CAMERA_PRESETS,
        "surface_colorscale": [
            [0.00, "#17324d"],
            [0.25, "#236c8a"],
            [0.55, "#7db8a5"],
            [0.78, "#f5c46b"],
            [1.00, "#d86c45"],
        ],
        "abs_delta_max": float(np.abs(probabilities - final_series[None, :]).max()),
        "zmax": min(1.02, max(0.08, float(probabilities.max()) * 1.03)),
        "main_plot_height": max(460, int(round(height_px * MAIN_PLOT_HEIGHT_RATIO))),
        "cross_section_height": max(220, int(round(height_px * CROSS_SECTION_HEIGHT_RATIO))),
        "default_scale": "actual",
        "default_checkpoint_index": len(checkpoint_steps) - 1,
        "default_frame_index": default_frame_index,
        "action_groups": action_groups,
    }


def _write_action_page(output_path, plotly_js_rel, app_js_rel, css_rel, payload):
    page_title = f"{payload['category_name']}: {payload['action_pretty']}"
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(page_title)}</title>
  <link rel="stylesheet" href="{escape(css_rel)}">
</head>
<body>
  <div class="shell">
    <header class="hero">
      <section class="card hero-copy">
        <div class="eyebrow">{escape(payload["category_name"])}</div>
        <h1>{escape(payload["action_pretty"])}</h1>
        <p class="subtitle">Rotate the 3D surface, switch checkpoint scales, and hover anywhere on the surface or guide rails. The two plots above update live to show a checkpoint slice over rollout time and a frame slice over training checkpoints.</p>
      </section>
      <aside class="card hero-meta">
        <a class="back-link" href="{escape(payload["gallery_href"])}">Back to 3D gallery</a>
        <div class="meta-box">
          <p class="meta-kicker">Surface Encoding</p>
          <p class="meta-copy">Height is action probability. Color is absolute distance from the final checkpoint. The thick black guide rail is the final policy.</p>
        </div>
      </aside>
    </header>

    <section class="workspace">
      <div class="left-column">
        <div class="cross-grid">
          <article class="card plot-card">
            <div class="plot-topline">
              <h2 class="plot-title">Checkpoint Slice</h2>
              <p class="plot-caption">One checkpoint across all 1000 frames</p>
            </div>
            <div class="plot-frame"><div id="checkpoint-cross-section"></div></div>
          </article>
          <article class="card plot-card">
            <div class="plot-topline">
              <h2 class="plot-title">Frame Slice</h2>
              <p class="plot-caption">One frame across all checkpoints</p>
            </div>
            <div class="plot-frame"><div id="frame-cross-section"></div></div>
          </article>
        </div>

        <article class="card main-card">
          <div class="plot-topline">
            <h2 class="plot-title">3D Probability Surface</h2>
            <p class="plot-caption">Use the right rail for camera presets and scale mode</p>
          </div>
          <div class="plot-frame"><div id="main-plot"></div></div>
        </article>
      </div>

      <aside class="card sidebar">
        <section class="sidebar-section">
          <p class="section-label">Action</p>
          <select id="action-select"></select>
        </section>

        <section class="sidebar-section">
          <p class="section-label">Checkpoint Scale</p>
          <select id="scale-select"></select>
        </section>

        <section class="sidebar-section">
          <p class="section-label">Camera Presets</p>
          <div class="camera-buttons" id="camera-buttons"></div>
        </section>

        <section class="sidebar-section">
          <p class="section-label">Current Hover</p>
          <div class="hover-summary" id="hover-summary"></div>
        </section>

        <section class="sidebar-section">
          <p class="section-label">How To Read It</p>
          <p class="note">Try <strong>Actual</strong> to respect training step spacing, <strong>Log</strong> to reveal early training motion, and <strong>Evenly spaced</strong> to compare checkpoints without the huge late-stage gaps dominating the view.</p>
          <p class="note">Then switch to <strong>Training Side</strong> and hover the ridge or cliff you care about. The upper plots will lock onto that hovered checkpoint and frame.</p>
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


def _write_index_html(output_dir, button_outputs, camera_outputs):
    index_path = output_dir / "index.html"
    button_links = "\n".join(
        f'<a class="card" href="buttons/{escape(filename)}"><span>{escape(_pretty_action_name(label))}</span><span class="meta">button</span></a>'
        for label, filename in button_outputs
    )
    camera_links = "\n".join(
        f'<a class="card" href="camera/{escape(filename)}"><span>{escape(_pretty_action_name(label))}</span><span class="meta">camera</span></a>'
        for label, filename in camera_outputs
    )
    html = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Checkpoint Progression 3D Gallery</title>
  <style>
    :root {
      --bg: #f6f1e8;
      --panel: rgba(255, 250, 242, 0.92);
      --ink: #1f1d19;
      --muted: #6c6459;
      --line: rgba(0, 0, 0, 0.10);
      --accent: #1f556c;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(31, 85, 108, 0.10), transparent 28%),
        radial-gradient(circle at bottom right, rgba(216, 108, 69, 0.12), transparent 24%),
        var(--bg);
    }
    main {
      max-width: 1520px;
      margin: 0 auto;
      padding: 30px 26px 42px;
    }
    h1 {
      margin: 0 0 10px;
      font-size: 2.35rem;
      line-height: 0.98;
      letter-spacing: -0.04em;
    }
    p {
      margin: 0;
      color: var(--muted);
      line-height: 1.46;
      font-size: 1.03rem;
    }
    .hero {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 310px;
      gap: 18px;
      margin-bottom: 26px;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 22px;
      padding: 20px 22px;
      box-shadow: 0 14px 40px rgba(0, 0, 0, 0.08);
    }
    .callout {
      display: flex;
      flex-direction: column;
      gap: 10px;
      justify-content: space-between;
    }
    h2 {
      margin: 0 0 14px;
      font-size: 1.04rem;
      letter-spacing: 0.10em;
      text-transform: uppercase;
      color: var(--muted);
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
      gap: 12px;
    }
    .card {
      display: flex;
      justify-content: space-between;
      gap: 10px;
      padding: 14px 16px;
      border: 1px solid var(--line);
      border-radius: 14px;
      background: rgba(255, 255, 255, 0.82);
      color: inherit;
      text-decoration: none;
      transition: transform 120ms ease, border-color 120ms ease, background 120ms ease;
    }
    .card:hover {
      transform: translateY(-2px);
      border-color: rgba(31, 85, 108, 0.30);
      background: white;
    }
    .meta {
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.06em;
      font-size: 0.84rem;
    }
    .section {
      margin-top: 22px;
    }
    @media (max-width: 960px) {
      .hero { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <main>
    <section class="hero">
      <div class="panel">
        <h1>Training Convergence In 3D</h1>
        <p>Each action now opens as a small dashboard instead of a lone plot. You get the rotatable 3D surface, a checkpoint-slice plot, a frame-slice plot, a checkpoint-scale dropdown, and direct action navigation from inside the page.</p>
      </div>
      <div class="panel callout">
        <p><strong>Suggested first look:</strong> open <a href="buttons/inventory.html">Inventory</a>, switch checkpoint scale between Actual and Evenly spaced, then hover the late ridge and watch the two cross-sections above update.</p>
      </div>
    </section>
    <section class="section panel">
      <h2>Button Actions</h2>
      <div class="grid">
        __BUTTON_LINKS__
      </div>
    </section>
    <section class="section panel">
      <h2>Camera Actions</h2>
      <div class="grid">
        __CAMERA_LINKS__
      </div>
    </section>
  </main>
</body>
</html>
"""
    html = html.replace("__BUTTON_LINKS__", button_links).replace("__CAMERA_LINKS__", camera_links)
    index_path.write_text(html)
    return index_path


def _write_ideas_markdown(output_dir):
    ideas_path = output_dir / "visualization_ideas.md"
    content = """# Next Visualization Ideas

## 1. Convergence Tunnel Video
Fly a camera through the 3D surface from the earliest checkpoint wall toward the final line, with the surface color pulsing on frames where the model is still far from its final policy. This would feel more cinematic than the GIF and would make sudden training phase changes read as cliffs.

## 2. Delta-To-Final Heat Wall
Build a wall plot where x is rollout frame, y is checkpoint, and color is signed difference to the final checkpoint. Use blue for below-final probability and ember for above-final probability. This would make overshoot, collapse, and late correction immediately visible without any rotation.

## 3. Policy Sculpture
Pick three semantically linked actions at once, for example inventory, attack, and look_left, then trace each checkpoint as a 3D trajectory through action-probability space over the rollout. Training becomes a moving sculpture rather than 24 separate single-action plots, which could reveal coupled strategy shifts that the per-action charts hide.
"""
    ideas_path.write_text(content)
    return ideas_path


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

    manifest, manifest_path = _load_manifest(experiment_dir)
    width_px, height_px = _plot_dimensions(manifest)
    assets_dir = _write_assets(output_dir)

    button_specs = [
        (action_name, entry, button_dir / f"{_slugify(action_name)}.html")
        for action_name, entry in sorted(manifest["button_plots"].items())
    ]
    camera_specs = [
        (action_name, entry, camera_dir / f"{_slugify(action_name)}.html")
        for action_name, entry in sorted(manifest["camera_plots"].items())
    ]

    output_manifest = {
        "source_manifest": str(manifest_path),
        "output_dir": str(output_dir),
        "assets_dir": str(assets_dir),
        "buttons": {},
        "camera": {},
        "plot_dimensions_pixels": {"width": width_px, "height": height_px},
    }

    gallery_index_path = output_dir / "index.html"

    for action_name, entry, output_path in button_specs:
        frames, checkpoint_steps, probabilities = _load_curve_csv(entry["csv"])
        action_groups = _build_action_groups(button_specs, camera_specs, output_path)
        payload = _page_payload(
            action_name=action_name,
            category_name="Button Action Evolution",
            frames=frames,
            checkpoint_steps=checkpoint_steps,
            probabilities=probabilities,
            width_px=width_px,
            height_px=height_px,
            action_groups=action_groups,
            current_output_path=output_path,
            gallery_path=gallery_index_path,
        )
        _write_action_page(
            output_path=output_path,
            plotly_js_rel=os.path.relpath(assets_dir / "plotly.min.js", output_path.parent),
            app_js_rel=os.path.relpath(assets_dir / "app.js", output_path.parent),
            css_rel=os.path.relpath(assets_dir / "dashboard.css", output_path.parent),
            payload=payload,
        )
        output_manifest["buttons"][action_name] = {"html": str(output_path), "source_csv": entry["csv"]}

    for action_name, entry, output_path in camera_specs:
        frames, checkpoint_steps, probabilities = _load_curve_csv(entry["csv"])
        action_groups = _build_action_groups(button_specs, camera_specs, output_path)
        payload = _page_payload(
            action_name=action_name,
            category_name="Camera Action Evolution",
            frames=frames,
            checkpoint_steps=checkpoint_steps,
            probabilities=probabilities,
            width_px=width_px,
            height_px=height_px,
            action_groups=action_groups,
            current_output_path=output_path,
            gallery_path=gallery_index_path,
        )
        _write_action_page(
            output_path=output_path,
            plotly_js_rel=os.path.relpath(assets_dir / "plotly.min.js", output_path.parent),
            app_js_rel=os.path.relpath(assets_dir / "app.js", output_path.parent),
            css_rel=os.path.relpath(assets_dir / "dashboard.css", output_path.parent),
            payload=payload,
        )
        output_manifest["camera"][action_name] = {"html": str(output_path), "source_csv": entry["csv"]}

    index_path = _write_index_html(
        output_dir=output_dir,
        button_outputs=[(action_name, output_path.name) for action_name, _, output_path in button_specs],
        camera_outputs=[(action_name, output_path.name) for action_name, _, output_path in camera_specs],
    )
    ideas_path = _write_ideas_markdown(output_dir)
    output_manifest["index_html"] = str(index_path)
    output_manifest["ideas_markdown"] = str(ideas_path)

    (output_dir / "manifest.json").write_text(json.dumps(output_manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
