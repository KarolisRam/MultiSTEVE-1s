import argparse
import base64
import csv
import json
import html
import re
import zipfile
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from steve1.VPT.agent import AGENT_RESOLUTION, resize_image

SCREEN_WIDTH_PX = 1900
SCREEN_HEIGHT_PX = 1080
SCREEN_FIT_RATIO = 0.90
PNG_SIDE_PX = int(min(SCREEN_WIDTH_PX, SCREEN_HEIGHT_PX) * SCREEN_FIT_RATIO)
PNG_DPI = 100


def _load_rows(paths: list[Path]) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    for path in paths:
        with path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows[row["model_id"]] = row
    return rows


def _seed_label(model_id: str) -> str:
    bitflip = re.search(r"bitflip_seed(\d+)", model_id)
    if bitflip:
        return f"bitflip-{bitflip.group(1)}"
    match = re.search(r"dataset_seed(\d+)", model_id)
    if match:
        return match.group(1)
    return model_id


def _seed_number(model_id: str) -> int | None:
    match = re.search(r"bitflip_seed(\d+)", model_id) or re.search(r"dataset_seed(\d+)", model_id)
    if match:
        return int(match.group(1))
    return None


def _seed_in_range(model_id: str, min_seed: int | None, max_seed: int | None) -> bool:
    if min_seed is None and max_seed is None:
        return True
    seed = _seed_number(model_id)
    if seed is None:
        return False
    if min_seed is not None and seed < min_seed:
        return False
    if max_seed is not None and seed > max_seed:
        return False
    return True


def _mean(left_pct_when_left: float, left_pct_when_right: float) -> tuple[float, float]:
    left_side_bias = 0.5 * (left_pct_when_left + left_pct_when_right)
    villager_bias = 0.5 * (left_pct_when_left + (100.0 - left_pct_when_right))
    return left_side_bias, villager_bias


def _plot(rows: list[dict[str, object]], out_path: Path, title: str) -> None:
    x = np.asarray([100.0 - float(r["left_pct_villager_left"]) for r in rows], dtype=np.float64)
    y = np.asarray([float(r["left_pct_villager_right"]) for r in rows], dtype=np.float64)
    labels = [str(r["short_label"]) for r in rows]

    plt.figure(figsize=(PNG_SIDE_PX / PNG_DPI, PNG_SIDE_PX / PNG_DPI), dpi=PNG_DPI)
    ax = plt.gca()

    ax.scatter(x, y, s=48, alpha=0.88, color="tab:blue", edgecolors="black", linewidths=0.5)
    for xi, yi, label in zip(x, y, labels):
        ax.text(xi + 0.8, yi + 0.8, label, fontsize=8, alpha=0.88)

    ax.axvline(50.0, color="gray", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.axhline(50.0, color="gray", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.plot([0, 100], [0, 100], color="tab:green", linestyle=":", linewidth=1.0, alpha=0.7)

    ax.text(2, 98, "always left", ha="left", va="top", fontsize=12, color="black")
    ax.text(98, 98, "regular-tree-seeking", ha="right", va="top", fontsize=12, color="black")
    ax.text(2, 2, "villager-seeking", ha="left", va="bottom", fontsize=12, color="black")
    ax.text(98, 2, "always right", ha="right", va="bottom", fontsize=12, color="black")

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Right choice rate (%) when villager tree is on the left")
    ax.set_ylabel("Left choice rate (%) when villager tree is on the right")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    plt.tight_layout(pad=2.0)
    plt.savefig(out_path, dpi=PNG_DPI)
    plt.close()


def _fmt_pct(value: object) -> str:
    return f"{float(value):.2f}%"


def _file_data_uri(path: Path, mime_type: str) -> str | None:
    if not path.exists():
        return None
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{payload}"


def _first_frame_data_uri(run_dir: str) -> str | None:
    fullres_path = Path(run_dir) / "first_frame_fullres.png"
    if fullres_path.exists():
        image = cv2.imread(str(fullres_path), cv2.IMREAD_COLOR)
        if image is not None:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = resize_image(image, AGENT_RESOLUTION)
            encoded_ok, buf = cv2.imencode(".png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            if encoded_ok:
                payload = base64.b64encode(buf.tobytes()).decode("ascii")
                return f"data:image/png;base64,{payload}"

    zip_path = Path(run_dir) / "observations.zip"
    if zip_path.exists():
        with zipfile.ZipFile(zip_path, "r") as zf:
            names = [name for name in zf.namelist() if name.lower().endswith(".png")]
            if names:
                first_name = sorted(names)[0]
                image_bytes = zf.read(first_name)
                image_np = np.frombuffer(image_bytes, dtype=np.uint8)
                image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
                if image is not None:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image = resize_image(image, AGENT_RESOLUTION)
                    encoded_ok, buf = cv2.imencode(".png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                    if encoded_ok:
                        payload = base64.b64encode(buf.tobytes()).decode("ascii")
                        return f"data:image/png;base64,{payload}"

    video_path = Path(run_dir) / "video.mp4"
    if not video_path.exists():
        return None

    cap = cv2.VideoCapture(str(video_path))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = resize_image(frame, AGENT_RESOLUTION)
    encoded_ok, buf = cv2.imencode(".png", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    if not encoded_ok:
        return None
    payload = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/png;base64,{payload}"


def _image_file_data_uri(image_path: Path) -> str | None:
    return _file_data_uri(image_path, "image/png")


def _write_interactive_html(rows: list[dict[str, object]], out_path: Path, page_title: str, chart_title: str) -> None:
    points = []
    for row in rows:
        point = dict(row)
        point["left_trajectory_uri"] = _image_file_data_uri(Path(str(row["left_run_dir"])) / "trajectory_topdown_reset.png")
        point["right_trajectory_uri"] = _image_file_data_uri(Path(str(row["right_run_dir"])) / "trajectory_topdown_reset.png")
        point["left_first_frame_uri"] = _first_frame_data_uri(str(row["left_run_dir"]))
        point["right_first_frame_uri"] = _first_frame_data_uri(str(row["right_run_dir"]))
        points.append(point)

    data_json = json.dumps(points)
    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(page_title)}</title>
  <style>
    :root {{
      --bg: #f5f1e8;
      --panel: #fffdf8;
      --ink: #17120f;
      --muted: #5d544d;
      --grid: rgba(38, 29, 23, 0.16);
      --cross: rgba(38, 29, 23, 0.45);
      --diag: rgba(24, 110, 71, 0.65);
      --point: #1d5fa7;
      --point-stroke: #0e223d;
      --tooltip: #17120f;
      --tooltip-ink: #fffdf8;
    }}
    * {{
      box-sizing: border-box;
    }}
    body {{
      margin: 0;
      font-family: "IBM Plex Sans", "Helvetica Neue", Arial, sans-serif;
      background:
        radial-gradient(circle at top left, rgba(201, 168, 114, 0.16), transparent 28%),
        linear-gradient(180deg, #f3ecdf 0%, #efe6d6 100%);
      color: var(--ink);
    }}
    .page {{
      width: min(90vw, 1710px);
      margin: 24px auto;
      padding: 20px 20px 16px;
      background: var(--panel);
      border: 1px solid rgba(23, 18, 15, 0.08);
      box-shadow: 0 18px 50px rgba(50, 38, 28, 0.12);
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 26px;
      line-height: 1.1;
    }}
    p {{
      margin: 0 0 14px;
      color: var(--muted);
      font-size: 15px;
      line-height: 1.45;
    }}
    .content {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) 720px;
      gap: 18px;
      align-items: start;
    }}
    .plot-panel {{
      min-width: 0;
    }}
    .plot-wrap {{
      position: relative;
      width: 100%;
      aspect-ratio: 1 / 1;
      background: #fffdfa;
      border: 1px solid rgba(23, 18, 15, 0.08);
      overflow: hidden;
    }}
    svg {{
      display: block;
      width: 100%;
      height: 100%;
    }}
    .side-panel {{
      padding: 14px 14px 12px;
      background: #fffdfa;
      border: 1px solid rgba(23, 18, 15, 0.08);
    }}
    .side-panel h2 {{
      margin: 0 0 8px;
      font-size: 20px;
      line-height: 1.15;
    }}
    .model-id {{
      margin-top: 14px;
      margin-bottom: 8px;
      font-size: 13px;
      color: var(--muted);
      word-break: break-word;
    }}
    .stats {{
      display: grid;
      grid-template-columns: 1fr;
      gap: 6px;
      margin-top: 0;
      margin-bottom: 0;
      font-size: 14px;
    }}
    .stats strong {{
      color: var(--ink);
    }}
    .media-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
      align-items: start;
    }}
    .media-column {{
      display: grid;
      grid-template-columns: 1fr;
      gap: 12px;
    }}
    .media-column h3 {{
      margin: 0;
      font-size: 14px;
      color: var(--muted);
      letter-spacing: 0.02em;
      text-transform: uppercase;
    }}
    .thumb {{
      margin: 0;
      width: 100%;
    }}
    .thumb figcaption {{
      margin-bottom: 4px;
      font-size: 12px;
      color: var(--muted);
    }}
    .thumb img {{
      display: block;
      width: 100%;
      height: auto;
      border-radius: 6px;
      border: 1px solid rgba(23, 18, 15, 0.12);
    }}
    .legend {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px 16px;
      margin-top: 14px;
      font-size: 14px;
      color: var(--muted);
    }}
    .legend strong {{
      color: var(--ink);
    }}
    .axis-note {{
      margin-top: 10px;
      font-size: 14px;
      color: var(--muted);
    }}
    .hover-note {{
      margin-top: 8px;
      font-size: 13px;
      color: var(--muted);
    }}
    @media (max-width: 1200px) {{
      .content {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <h1>{html.escape(page_title)}</h1>
    <p>Each point is one model. Hover or click a point to update the right-side panel with the corresponding top-down trajectories and first-frame views.</p>
    <div class="content">
      <div class="plot-panel">
        <div class="plot-wrap" id="plot-wrap">
          <svg id="plot" viewBox="0 0 1000 1000" aria-label="{html.escape(page_title)}"></svg>
        </div>
        <div class="legend">
          <div><strong>Top-left:</strong> always left</div>
          <div><strong>Top-right:</strong> regular-tree-seeking</div>
          <div><strong>Bottom-left:</strong> villager-seeking</div>
          <div><strong>Bottom-right:</strong> always right</div>
        </div>
        <div class="axis-note">x-axis: right choice rate when villager tree is on the left. y-axis: left choice rate when villager tree is on the right.</div>
        <div class="hover-note">Hover or click a point to pin the corresponding trajectories and first-frame views on the right.</div>
      </div>
      <aside class="side-panel">
        <h2 id="detail-title">agent</h2>
        <div class="media-grid">
          <div class="media-column">
            <h3>Top-Down Trajectories</h3>
            <figure class="thumb">
              <figcaption>Villager left</figcaption>
              <a id="left-trajectory-link" target="_blank" rel="noopener noreferrer">
                <img id="left-trajectory" alt="Villager left top-down trajectory">
              </a>
            </figure>
            <figure class="thumb">
              <figcaption>Villager right</figcaption>
              <a id="right-trajectory-link" target="_blank" rel="noopener noreferrer">
                <img id="right-trajectory" alt="Villager right top-down trajectory">
              </a>
            </figure>
          </div>
          <div class="media-column">
            <h3>First Frames</h3>
            <figure class="thumb">
              <figcaption>Villager left</figcaption>
              <a id="left-frame-link" target="_blank" rel="noopener noreferrer">
                <img id="left-frame" alt="Villager left first frame">
              </a>
            </figure>
            <figure class="thumb">
              <figcaption>Villager right</figcaption>
              <a id="right-frame-link" target="_blank" rel="noopener noreferrer">
                <img id="right-frame" alt="Villager right first frame">
              </a>
            </figure>
          </div>
        </div>
        <div id="detail-model-id" class="model-id"></div>
        <div id="detail-stats" class="stats"></div>
      </aside>
    </div>
  </div>
  <script>
    const points = {data_json};
    const svg = document.getElementById("plot");
    const detailTitle = document.getElementById("detail-title");
    const detailModelId = document.getElementById("detail-model-id");
    const detailStats = document.getElementById("detail-stats");
    const leftTrajectory = document.getElementById("left-trajectory");
    const rightTrajectory = document.getElementById("right-trajectory");
    const leftFrame = document.getElementById("left-frame");
    const rightFrame = document.getElementById("right-frame");
    const leftTrajectoryLink = document.getElementById("left-trajectory-link");
    const rightTrajectoryLink = document.getElementById("right-trajectory-link");
    const leftFrameLink = document.getElementById("left-frame-link");
    const rightFrameLink = document.getElementById("right-frame-link");
    const W = 1000;
    const H = 1000;
    const M = {{ left: 104, right: 28, top: 42, bottom: 92 }};
    const innerW = W - M.left - M.right;
    const innerH = H - M.top - M.bottom;

    function sx(v) {{
      return M.left + (v / 100) * innerW;
    }}
    function sy(v) {{
      return H - M.bottom - (v / 100) * innerH;
    }}

    function add(tag, attrs, parent) {{
      const el = document.createElementNS("http://www.w3.org/2000/svg", tag);
      Object.entries(attrs).forEach(([k, v]) => el.setAttribute(k, String(v)));
      parent.appendChild(el);
      return el;
    }}

    add("rect", {{ x: 0, y: 0, width: W, height: H, fill: "#fffdfa" }}, svg);

    for (let t = 0; t <= 100; t += 10) {{
      add("line", {{
        x1: sx(t), y1: sy(0), x2: sx(t), y2: sy(100),
        stroke: "var(--grid)", "stroke-width": 1
      }}, svg);
      add("line", {{
        x1: sx(0), y1: sy(t), x2: sx(100), y2: sy(t),
        stroke: "var(--grid)", "stroke-width": 1
      }}, svg);
    }}

    add("line", {{
      x1: sx(50), y1: sy(0), x2: sx(50), y2: sy(100),
      stroke: "var(--cross)", "stroke-width": 2, "stroke-dasharray": "8 8"
    }}, svg);
    add("line", {{
      x1: sx(0), y1: sy(50), x2: sx(100), y2: sy(50),
      stroke: "var(--cross)", "stroke-width": 2, "stroke-dasharray": "8 8"
    }}, svg);
    add("line", {{
      x1: sx(0), y1: sy(0), x2: sx(100), y2: sy(100),
      stroke: "var(--diag)", "stroke-width": 2, "stroke-dasharray": "5 6"
    }}, svg);

    add("rect", {{
      x: sx(0), y: sy(100), width: innerW, height: innerH,
      fill: "none", stroke: "#1a1511", "stroke-width": 2
    }}, svg);

    function label(text, x, y, anchor, size = 16, weight = 500, fill = "#17120f") {{
      const el = add("text", {{
        x, y, "text-anchor": anchor, "font-size": size,
        "font-weight": weight, fill
      }}, svg);
      el.textContent = text;
      return el;
    }}

    label("always left", sx(2), sy(98), "start", 18, 600);
    label("regular-tree-seeking", sx(98), sy(98), "end", 18, 600);
    label("villager-seeking", sx(2), sy(2), "start", 18, 600);
    label("always right", sx(98), sy(2), "end", 18, 600);

    label("Right choice rate (%) when villager tree is on the left", W / 2, H - 24, "middle", 18, 500);
    const yAxis = label("Left choice rate (%) when villager tree is on the right", 24, H / 2, "middle", 18, 500);
    yAxis.setAttribute("transform", `rotate(-90 24 ${{H / 2}})`);
    label({json.dumps(chart_title)}, W / 2, 24, "middle", 22, 700);

    for (let t = 0; t <= 100; t += 10) {{
      label(String(t), sx(t), H - 56, "middle", 14, 400, "#5d544d");
      label(String(t), 76, sy(t) + 5, "end", 14, 400, "#5d544d");
    }}

    function setDetail(point) {{
      detailTitle.textContent = `agent-${'{'}point.short_label{'}'}`;
      detailModelId.textContent = point.model_id;
      detailStats.innerHTML =
        `<div><strong>Left% when villager left:</strong> ${'{'}Number(point.left_pct_villager_left).toFixed(2){'}'}%</div>` +
        `<div><strong>Left% when villager right:</strong> ${'{'}Number(point.left_pct_villager_right).toFixed(2){'}'}%</div>` +
        `<div><strong>Left-bias:</strong> ${'{'}Number(point.left_side_bias_pct).toFixed(2){'}'}%</div>` +
        `<div><strong>Villager-bias:</strong> ${'{'}Number(point.villager_choice_bias_pct).toFixed(2){'}'}%</div>` +
        `<div><strong>Segments:</strong> ${'{'}point.segments_left_layout{'}'} / ${'{'}point.segments_right_layout{'}'}</div>`;

      if (point.left_trajectory_uri) {{
        leftTrajectory.src = point.left_trajectory_uri;
        leftTrajectoryLink.href = point.left_trajectory_uri;
        leftTrajectory.style.visibility = "visible";
      }} else {{
        leftTrajectory.removeAttribute("src");
        leftTrajectoryLink.removeAttribute("href");
        leftTrajectory.style.visibility = "hidden";
      }}

      if (point.right_trajectory_uri) {{
        rightTrajectory.src = point.right_trajectory_uri;
        rightTrajectoryLink.href = point.right_trajectory_uri;
        rightTrajectory.style.visibility = "visible";
      }} else {{
        rightTrajectory.removeAttribute("src");
        rightTrajectoryLink.removeAttribute("href");
        rightTrajectory.style.visibility = "hidden";
      }}

      if (point.left_first_frame_uri) {{
        leftFrame.src = point.left_first_frame_uri;
        leftFrameLink.href = point.left_first_frame_uri;
        leftFrame.style.visibility = "visible";
      }} else {{
        leftFrame.removeAttribute("src");
        leftFrameLink.removeAttribute("href");
        leftFrame.style.visibility = "hidden";
      }}

      if (point.right_first_frame_uri) {{
        rightFrame.src = point.right_first_frame_uri;
        rightFrameLink.href = point.right_first_frame_uri;
        rightFrame.style.visibility = "visible";
      }} else {{
        rightFrame.removeAttribute("src");
        rightFrameLink.removeAttribute("href");
        rightFrame.style.visibility = "hidden";
      }}
    }}

    points.forEach((point) => {{
      const circle = add("circle", {{
        cx: sx(point.right_pct_villager_left),
        cy: sy(point.left_pct_villager_right),
        r: 6,
        fill: "var(--point)",
        stroke: "var(--point-stroke)",
        "stroke-width": 1.25,
        opacity: 0.9
      }}, svg);

      const txt = label(
        point.short_label,
        sx(point.right_pct_villager_left) + 8,
        sy(point.left_pct_villager_right) - 8,
        "start",
        12,
        500,
        "#1d2735"
      );

      [circle, txt].forEach((node) => {{
        node.addEventListener("mouseenter", () => setDetail(point));
        node.addEventListener("click", () => setDetail(point));
      }});
    }});

    if (points.length > 0) {{
      setDetail(points[0]);
    }}
  </script>
</body>
</html>
"""
    out_path.write_text(html_doc)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--left_first_csv",
        action="append",
        required=True,
        help="CSV where villager tree is on the left. Can be passed multiple times; later files override earlier rows by model_id.",
    )
    parser.add_argument(
        "--right_first_csv",
        action="append",
        required=True,
        help="CSV where villager tree is on the right. Can be passed multiple times; later files override earlier rows by model_id.",
    )
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--page_title", default="Layout Scatter")
    parser.add_argument("--chart_title", default="Left-choice Scatter Across Tree Layouts")
    parser.add_argument("--min_agent_seed", type=int, default=None)
    parser.add_argument("--max_agent_seed", type=int, default=None)
    args = parser.parse_args()

    left_first_paths = [Path(p) for p in args.left_first_csv]
    right_first_paths = [Path(p) for p in args.right_first_csv]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    left_rows = _load_rows(left_first_paths)
    right_rows = _load_rows(right_first_paths)

    model_ids = sorted(set(left_rows) & set(right_rows))
    model_ids = [
        model_id
        for model_id in model_ids
        if _seed_in_range(model_id, args.min_agent_seed, args.max_agent_seed)
    ]
    combined_rows: list[dict[str, object]] = []
    skipped = sorted(
        model_id
        for model_id in (set(left_rows) ^ set(right_rows))
        if _seed_in_range(model_id, args.min_agent_seed, args.max_agent_seed)
    )
    for model_id in model_ids:
        left_row = left_rows[model_id]
        right_row = right_rows[model_id]

        left_pct = float(left_row["left_rate_pct_excluding_ties"])
        right_pct = float(right_row["left_rate_pct_excluding_ties"])
        left_side_bias, villager_bias = _mean(left_pct, right_pct)

        combined_rows.append(
            {
                "model_id": model_id,
                "short_label": _seed_label(model_id),
                "left_run_dir": left_row["run_dir"],
                "right_run_dir": right_row["run_dir"],
                "left_pct_villager_left": left_pct,
                "right_pct_villager_left": 100.0 - left_pct,
                "left_pct_villager_right": right_pct,
                "left_ci_low_villager_left": float(left_row["left_ci95_low_pct"]),
                "left_ci_high_villager_left": float(left_row["left_ci95_high_pct"]),
                "left_ci_low_villager_right": float(right_row["left_ci95_low_pct"]),
                "left_ci_high_villager_right": float(right_row["left_ci95_high_pct"]),
                "left_side_bias_pct": left_side_bias,
                "villager_choice_bias_pct": villager_bias,
                "segments_left_layout": int(left_row["segments_total"]),
                "segments_right_layout": int(right_row["segments_total"]),
            }
        )

    merged_csv = out_dir / "left_choice_layout_scatter.csv"
    with merged_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model_id",
                "short_label",
                "left_run_dir",
                "right_run_dir",
                "left_pct_villager_left",
                "right_pct_villager_left",
                "left_pct_villager_right",
                "left_ci_low_villager_left",
                "left_ci_high_villager_left",
                "left_ci_low_villager_right",
                "left_ci_high_villager_right",
                "left_side_bias_pct",
                "villager_choice_bias_pct",
                "segments_left_layout",
                "segments_right_layout",
            ],
        )
        writer.writeheader()
        writer.writerows(combined_rows)

    plot_path = out_dir / "left_choice_layout_scatter.png"
    _plot(combined_rows, plot_path, args.chart_title)
    html_path = out_dir / "left_choice_layout_scatter.html"
    _write_interactive_html(combined_rows, html_path, args.page_title, args.chart_title)

    summary_md = out_dir / "left_choice_layout_scatter.md"
    with summary_md.open("w") as f:
        f.write(f"# {args.page_title}\n\n")
        f.write("Each point is one model.\n\n")
        f.write("- x-axis: right choice rate when villager tree is on the left\n")
        f.write("- y-axis: left choice rate when villager tree is on the right\n")
        f.write("- top-left: always go left\n")
        f.write("- top-right: seek regular tree\n")
        f.write("- bottom-left: seek villager tree\n")
        f.write("- bottom-right: always go right\n")

    summary_json = out_dir / "left_choice_layout_scatter_summary.json"
    summary_json.write_text(
        json.dumps(
            {
                "left_first_csvs": [str(path) for path in left_first_paths],
                "right_first_csvs": [str(path) for path in right_first_paths],
                "min_agent_seed": args.min_agent_seed,
                "max_agent_seed": args.max_agent_seed,
                "num_models": len(combined_rows),
                "skipped_model_ids": skipped,
                "merged_csv": str(merged_csv),
                "plot": str(plot_path),
                "html": str(html_path),
                "summary_md": str(summary_md),
            },
            indent=2,
        )
    )

    print(f"wrote: {merged_csv}")
    print(f"wrote: {plot_path}")
    print(f"wrote: {html_path}")
    print(f"wrote: {summary_md}")
    print(f"wrote: {summary_json}")
    print(f"models merged: {len(combined_rows)}")


if __name__ == "__main__":
    main()
