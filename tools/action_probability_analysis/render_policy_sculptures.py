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
    write_plotly_assets,
)

DEFAULT_OUTPUT_SUBDIR = "policy_sculpture"
FRAME_DOWNSAMPLE = 4

TRIPLETS = [
    ("inventory", "attack", "use"),
    ("forward", "left", "right"),
    ("jump", "sprint", "sneak"),
    ("look_left", "look_right", "look_up"),
    ("inventory", "hotbar.1", "hotbar.2"),
    ("attack", "look_left", "look_up"),
]

APP_JS = """
(function () {
  const page = JSON.parse(document.getElementById("page-config").textContent);
  const host = document.getElementById("sculpture-plot");
  const tripletSelect = document.getElementById("triplet-select");
  const slider = document.getElementById("checkpoint-slider");
  const sliderLabel = document.getElementById("checkpoint-slider-label");
  const playButton = document.getElementById("play-button");
  const pauseButton = document.getElementById("pause-button");
  const hoverSummary = document.getElementById("hover-summary");

  let currentCheckpointIndex = page.default_checkpoint_index;
  let timerId = null;

  function populateTripletSelect() {
    page.triplet_options.forEach((entry) => {
      const option = document.createElement("option");
      option.value = entry.href;
      option.textContent = entry.label;
      if (entry.href === page.current_triplet_href) {
        option.selected = true;
      }
      tripletSelect.appendChild(option);
    });
    tripletSelect.addEventListener("change", () => {
      window.location.href = tripletSelect.value;
    });
  }

  function toComma(value) {
    return Number(value).toLocaleString("en-US");
  }

  function updateSliderLabel() {
    const step = page.checkpoint_steps[currentCheckpointIndex];
    const label = page.checkpoint_labels[currentCheckpointIndex];
    sliderLabel.textContent = "Highlighted checkpoint: " + label + " (" + toComma(step) + ")";
  }

  function traceStyle(checkpointIndex) {
    const isFinal = checkpointIndex === page.checkpoint_steps.length - 1;
    const isSelected = checkpointIndex === currentCheckpointIndex;
    if (isFinal) {
      return {color: "#000000", width: isSelected ? 10 : 8, opacity: 1.0};
    }
    if (isSelected) {
      return {color: "#d86c45", width: 7, opacity: 0.95};
    }
    return {
      color: page.checkpoint_colors[checkpointIndex],
      width: 2,
      opacity: 0.22 + (checkpointIndex / Math.max(1, page.checkpoint_steps.length - 1)) * 0.10,
    };
  }

  function buildTraces() {
    const traces = page.trajectories.map((trajectory, checkpointIndex) => {
      const style = traceStyle(checkpointIndex);
      return {
        type: "scatter3d",
        mode: "lines",
        x: trajectory.x,
        y: trajectory.y,
        z: trajectory.z,
        customdata: trajectory.customdata,
        line: {
          color: style.color,
          width: style.width,
        },
        opacity: style.opacity,
        name: page.checkpoint_labels[checkpointIndex],
        hovertemplate:
          "Checkpoint %{customdata[0]}<br>" +
          "Frame %{customdata[1]:,.0f}<br>" +
          page.axis_titles[0] + " %{x:.4f}<br>" +
          page.axis_titles[1] + " %{y:.4f}<br>" +
          page.axis_titles[2] + " %{z:.4f}<extra></extra>",
      };
    });

    traces.push({
      type: "scatter3d",
      mode: "markers",
      x: page.anchor_points.x,
      y: page.anchor_points.y,
      z: page.anchor_points.z,
      customdata: page.anchor_points.customdata,
      marker: {
        color: "#000000",
        size: 3.4,
        opacity: 0.78,
      },
      name: "Final anchor frames",
      hovertemplate:
        "Final checkpoint<br>" +
        "Frame %{customdata[0]:,.0f}<br>" +
        page.axis_titles[0] + " %{x:.4f}<br>" +
        page.axis_titles[1] + " %{y:.4f}<br>" +
        page.axis_titles[2] + " %{z:.4f}<extra></extra>",
    });
    return traces;
  }

  function updateSummary() {
    const step = page.checkpoint_steps[currentCheckpointIndex];
    const label = page.checkpoint_labels[currentCheckpointIndex];
    hoverSummary.innerHTML = [
      ["Triplet", page.triplet_label],
      ["Highlighted checkpoint", label + " (" + toComma(step) + ")"],
      ["Final checkpoint", page.checkpoint_labels[page.checkpoint_steps.length - 1]],
      ["Frames sampled", toComma(page.frames_sampled)],
    ]
      .map(([key, value]) => '<div class="summary-row"><span class="summary-key">' + key + '</span><span class="summary-value">' + value + '</span></div>')
      .join("");
  }

  function layout() {
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
        camera: {eye: {x: 1.55, y: -1.45, z: 1.10}},
        aspectmode: "cube",
        xaxis: {
          title: page.axis_titles[0],
          range: [0, page.axis_max],
          backgroundcolor: "rgba(255,255,255,0)",
          gridcolor: "rgba(0,0,0,0.10)",
          zerolinecolor: "rgba(0,0,0,0.18)",
        },
        yaxis: {
          title: page.axis_titles[1],
          range: [0, page.axis_max],
          backgroundcolor: "rgba(255,255,255,0)",
          gridcolor: "rgba(0,0,0,0.10)",
          zerolinecolor: "rgba(0,0,0,0.18)",
        },
        zaxis: {
          title: page.axis_titles[2],
          range: [0, page.axis_max],
          backgroundcolor: "rgba(255,255,255,0)",
          gridcolor: "rgba(0,0,0,0.10)",
          zerolinecolor: "rgba(0,0,0,0.18)",
        },
      },
    };
  }

  function render() {
    Plotly.react(host, buildTraces(), layout(), {displayModeBar: true, responsive: true});
    updateSliderLabel();
    updateSummary();
  }

  function stopPlayback() {
    if (timerId !== null) {
      window.clearInterval(timerId);
      timerId = null;
    }
  }

  function startPlayback() {
    stopPlayback();
    timerId = window.setInterval(() => {
      currentCheckpointIndex += 1;
      if (currentCheckpointIndex >= page.checkpoint_steps.length) {
        currentCheckpointIndex = 0;
      }
      slider.value = String(currentCheckpointIndex);
      render();
    }, 480);
  }

  populateTripletSelect();
  slider.max = String(page.checkpoint_steps.length - 1);
  slider.value = String(currentCheckpointIndex);
  slider.addEventListener("input", () => {
    currentCheckpointIndex = Number(slider.value);
    render();
  });
  playButton.addEventListener("click", startPlayback);
  pauseButton.addEventListener("click", stopPlayback);
  render();
  window.addEventListener("resize", () => {
    Plotly.Plots.resize(host);
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
select, input[type="range"] {
  width: 100%;
}
select {
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
.summary-box {
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
"""


def downsample_matrix(matrix, step=FRAME_DOWNSAMPLE):
    indices = list(range(0, matrix.shape[1], step))
    if indices[-1] != matrix.shape[1] - 1:
        indices.append(matrix.shape[1] - 1)
    indices = np.array(indices, dtype=int)
    return matrix[:, indices], indices


def find_curve_path(manifest, action_name):
    if action_name in manifest["button_plots"]:
        return manifest["button_plots"][action_name]["csv"], "button"
    return manifest["camera_plots"][action_name]["csv"], "camera"


def checkpoint_colors(num_checkpoints):
    colors = []
    for index in range(num_checkpoints):
        fraction = index / max(1, num_checkpoints - 1)
        gray = 0.74 - fraction * 0.58
        colors.append(f"rgba({int(gray * 255)},{int(gray * 255)},{int(gray * 255)},1.0)")
    return colors


def build_triplet_options(specs, current_output_path):
    return [
        {
            "label": spec["triplet_label"],
            "href": os.path.relpath(spec["output_path"], current_output_path.parent),
        }
        for spec in specs
    ]


def page_payload(spec, plot_height, current_output_path, gallery_path, triplet_specs):
    trajectories = []
    x_matrix = spec["series_map"][spec["triplet"][0]]
    _, indices = downsample_matrix(x_matrix[None, :], step=FRAME_DOWNSAMPLE)
    sampled_frames = spec["frames"][indices]
    for checkpoint_index in range(len(spec["checkpoint_steps"])):
        customdata = [[format_step_compact(int(spec["checkpoint_steps"][checkpoint_index])), int(frame)] for frame in sampled_frames]
        trajectories.append(
            {
                "x": np.round(spec["series_map"][spec["triplet"][0]][checkpoint_index, indices], 10).tolist(),
                "y": np.round(spec["series_map"][spec["triplet"][1]][checkpoint_index, indices], 10).tolist(),
                "z": np.round(spec["series_map"][spec["triplet"][2]][checkpoint_index, indices], 10).tolist(),
                "customdata": customdata,
            }
        )

    final_index = len(spec["checkpoint_steps"]) - 1
    anchor_every = max(1, len(indices) // 18)
    anchor_positions = indices[::anchor_every]
    if anchor_positions[-1] != indices[-1]:
        anchor_positions = np.append(anchor_positions, indices[-1])
    anchor_points = {
        "x": np.round(spec["series_map"][spec["triplet"][0]][final_index, anchor_positions], 10).tolist(),
        "y": np.round(spec["series_map"][spec["triplet"][1]][final_index, anchor_positions], 10).tolist(),
        "z": np.round(spec["series_map"][spec["triplet"][2]][final_index, anchor_positions], 10).tolist(),
        "customdata": spec["frames"][anchor_positions].astype(int).tolist(),
    }
    axis_max = float(
        max(
            spec["series_map"][spec["triplet"][0]].max(),
            spec["series_map"][spec["triplet"][1]].max(),
            spec["series_map"][spec["triplet"][2]].max(),
        )
        * 1.03
    )
    axis_max = min(1.02, max(axis_max, 0.08))
    return {
        "triplet_label": spec["triplet_label"],
        "triplet_pretty": " / ".join(pretty_action_name(name) for name in spec["triplet"]),
        "current_triplet_href": os.path.relpath(current_output_path, current_output_path.parent),
        "gallery_href": os.path.relpath(gallery_path, current_output_path.parent),
        "triplet_options": build_triplet_options(triplet_specs, current_output_path),
        "axis_titles": [pretty_action_name(name) for name in spec["triplet"]],
        "checkpoint_steps": spec["checkpoint_steps"].tolist(),
        "checkpoint_labels": [format_step_compact(int(step)) for step in spec["checkpoint_steps"]],
        "checkpoint_colors": checkpoint_colors(len(spec["checkpoint_steps"])),
        "trajectories": trajectories,
        "anchor_points": anchor_points,
        "default_checkpoint_index": len(spec["checkpoint_steps"]) - 1,
        "plot_height": plot_height,
        "axis_max": axis_max,
        "frames_sampled": int(len(sampled_frames)),
    }


def write_action_page(output_path, plotly_js_rel, app_js_rel, css_rel, payload):
    title = f"Policy Sculpture: {payload['triplet_pretty']}"
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
        <div class="eyebrow">Policy Sculpture</div>
        <h1>{escape(payload["triplet_pretty"])}</h1>
        <p class="subtitle">Each line is one checkpoint moving through a three-action probability space over rollout time. The final checkpoint is the thick black sculpture. Earlier checkpoints are the ghosted scaffolding that training had to pass through.</p>
      </section>
      <aside class="card hero-meta">
        <a class="back-link" href="{escape(payload["gallery_href"])}">Back to sculpture gallery</a>
        <div class="meta-box">
          <p class="meta-kicker">Why This Exists</p>
          <p class="meta-copy">Per-action plots hide coupling. This view is for moments where two or three policies seem to learn together, collapse together, or take a coordinated late turn.</p>
        </div>
      </aside>
    </header>
    <section class="workspace">
      <article class="card plot-card">
        <div class="plot-topline">
          <h2 class="plot-title">3D Trajectory Sculpture</h2>
          <p class="plot-caption">Use the slider or playback to highlight checkpoints</p>
        </div>
        <div class="plot-frame"><div id="sculpture-plot"></div></div>
      </article>
      <aside class="card sidebar">
        <section>
          <p class="section-label">Triplet</p>
          <select id="triplet-select"></select>
        </section>
        <section>
          <p class="section-label">Checkpoint Highlight</p>
          <div id="checkpoint-slider-label" class="note"></div>
          <input id="checkpoint-slider" type="range" min="0" step="1">
          <div class="control-buttons">
            <button id="play-button" type="button">Play</button>
            <button id="pause-button" type="button">Pause</button>
          </div>
        </section>
        <section>
          <p class="section-label">Current State</p>
          <div id="hover-summary" class="summary-box"></div>
        </section>
        <section>
          <p class="section-label">How To Read It</p>
          <p class="note">If the scaffolding is wide and tangled, training changed the joint policy a lot. If it compresses smoothly into the black line, the triplet converged in a coordinated way. Look for late twists and detached loops.</p>
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


def write_index(output_dir, triplet_specs):
    tiles = "\n".join(
        f'<a class="tile" href="triplets/{escape(spec["output_path"].name)}"><span>{escape(spec["triplet_label"])}</span><span class="meta">triplet</span></a>'
        for spec in triplet_specs
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Policy Sculpture Gallery</title>
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
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
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
        <h1>Policy Sculpture</h1>
        <p>These are curated three-action spaces. Each checkpoint becomes a trajectory through that space over rollout time, which turns training into a bundle of moving sculptures instead of 24 independent charts.</p>
      </div>
      <div class="panel">
        <p><strong>Suggested first look:</strong> start with <a href="triplets/inventory__attack__use.html">Inventory / Attack / Use</a> and <a href="triplets/forward__left__right.html">Forward / Left / Right</a>. Those usually show the clearest coupling.</p>
      </div>
    </section>
    <section class="panel">
      <div class="grid">{tiles}</div>
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
    triplet_dir = output_dir / "triplets"
    triplet_dir.mkdir(parents=True, exist_ok=True)

    manifest, manifest_path = load_manifest(experiment_dir)
    width_px, height_px = plot_dimensions(manifest)
    plot_height = max(580, int(height_px * 0.78))
    assets_dir = write_plotly_assets(output_dir, extra_js=APP_JS, css_text=CSS)

    loaded_curves = {}
    triplet_specs = []
    for triplet in TRIPLETS:
        series_map = {}
        checkpoint_steps = None
        frames = None
        for action_name in triplet:
            csv_path, action_type = find_curve_path(manifest, action_name)
            if action_name not in loaded_curves:
                frames_arr, checkpoint_steps_arr, series_arr = load_curve_csv(csv_path)
                loaded_curves[action_name] = (frames_arr, checkpoint_steps_arr, series_arr)
            frames_arr, checkpoint_steps_arr, series_arr = loaded_curves[action_name]
            series_map[action_name] = series_arr
            if checkpoint_steps is None:
                checkpoint_steps = checkpoint_steps_arr
                frames = frames_arr
        filename = "__".join(slugify_action_name(name) for name in triplet) + ".html"
        triplet_specs.append(
            {
                "triplet": triplet,
                "triplet_label": " / ".join(pretty_action_name(name) for name in triplet),
                "series_map": series_map,
                "checkpoint_steps": checkpoint_steps,
                "frames": frames,
                "output_path": triplet_dir / filename,
            }
        )

    output_manifest = {
        "source_manifest": str(manifest_path),
        "output_dir": str(output_dir),
        "assets_dir": str(assets_dir),
        "triplets": {},
        "frame_downsample": FRAME_DOWNSAMPLE,
        "plot_dimensions_pixels": {"width": width_px, "height": plot_height},
    }

    gallery_path = output_dir / "index.html"

    for spec in triplet_specs:
        payload = page_payload(
            spec=spec,
            plot_height=plot_height,
            current_output_path=spec["output_path"],
            gallery_path=gallery_path,
            triplet_specs=triplet_specs,
        )
        write_action_page(
            output_path=spec["output_path"],
            plotly_js_rel=os.path.relpath(assets_dir / "plotly.min.js", spec["output_path"].parent),
            app_js_rel=os.path.relpath(assets_dir / "app.js", spec["output_path"].parent),
            css_rel=os.path.relpath(assets_dir / "styles.css", spec["output_path"].parent),
            payload=payload,
        )
        output_manifest["triplets"][spec["triplet_label"]] = {
            "html": str(spec["output_path"]),
            "actions": list(spec["triplet"]),
        }

    index_path = write_index(output_dir, triplet_specs)
    output_manifest["index_html"] = str(index_path)

    (output_dir / "manifest.json").write_text(json.dumps(output_manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
