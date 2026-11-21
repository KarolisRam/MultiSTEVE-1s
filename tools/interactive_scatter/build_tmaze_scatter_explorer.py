from __future__ import annotations

import argparse
import base64
import csv
import html
import json
import mimetypes
import re
from pathlib import Path
from typing import Any


DEFAULT_CACHE_FILES = (
    Path("outputs/plot_suite/cache/teleport_layout/seed49_first100.csv"),
    Path("outputs/plot_suite/cache/teleport_layout/seed481_first100.csv"),
)


def infer_seed_label(path: Path) -> str:
    match = re.search(r"seed0*(\d+)", path.name)
    if match:
        return f"Seed {match.group(1)}"
    return path.stem


def parse_input(value: str) -> tuple[str | None, Path]:
    if "=" not in value:
        return None, Path(value)
    label, raw_path = value.split("=", 1)
    label = label.strip()
    if not label:
        raise ValueError(f"empty label in --input {value!r}")
    return label, Path(raw_path)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def number(row: dict[str, str], key: str, default: float | None = None) -> float | None:
    raw = row.get(key, "")
    if raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def data_uri(path: Path) -> str:
    mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{payload}"


def candidate_asset(
    run_dir: str,
    names: tuple[str, ...],
    artifact_root: Path | None,
    embed_assets: bool,
) -> dict[str, Any]:
    if not run_dir:
        return {"available": False, "href": None, "path": None}

    run_path = Path(run_dir)
    candidates: list[Path] = []
    if run_path.is_absolute():
        candidates.extend(run_path / name for name in names)
    else:
        candidates.extend(run_path / name for name in names)
        if artifact_root is not None:
            candidates.extend(artifact_root / run_path / name for name in names)

    for path in candidates:
        if path.exists():
            href = data_uri(path) if embed_assets else path.as_posix()
            return {"available": True, "href": href, "path": path.as_posix()}

    first = candidates[0] if candidates else run_path / names[0]
    return {"available": False, "href": None, "path": first.as_posix()}


def build_point(
    row: dict[str, str],
    seed_label: str,
    artifact_root: Path | None,
    embed_assets: bool,
) -> dict[str, Any]:
    left_pct_villager_left = number(row, "left_pct_villager_left")
    right_pct_villager_left = number(row, "right_pct_villager_left")
    left_pct_villager_right = number(row, "left_pct_villager_right")
    if left_pct_villager_left is None or left_pct_villager_right is None:
        raise ValueError(f"missing T-maze choice-rate columns for {row.get('model_id', '<unknown>')}")
    if right_pct_villager_left is None:
        right_pct_villager_left = 100.0 - left_pct_villager_left

    left_ci_low = number(row, "left_ci_low_villager_left")
    left_ci_high = number(row, "left_ci_high_villager_left")
    right_ci_low = number(row, "left_ci_low_villager_right")
    right_ci_high = number(row, "left_ci_high_villager_right")
    x_ci_low = None if left_ci_high is None else max(0.0, 100.0 - left_ci_high)
    x_ci_high = None if left_ci_low is None else min(100.0, 100.0 - left_ci_low)
    y_ci_low = None if right_ci_low is None else max(0.0, right_ci_low)
    y_ci_high = None if right_ci_high is None else min(100.0, right_ci_high)

    left_run_dir = row.get("left_run_dir", "")
    right_run_dir = row.get("right_run_dir", "")
    return {
        "model_id": row.get("model_id", ""),
        "short_label": row.get("short_label", row.get("model_id", "")),
        "seed_label": seed_label,
        "x": right_pct_villager_left,
        "y": left_pct_villager_right,
        "x_ci_low": x_ci_low,
        "x_ci_high": x_ci_high,
        "y_ci_low": y_ci_low,
        "y_ci_high": y_ci_high,
        "left_pct_villager_left": left_pct_villager_left,
        "right_pct_villager_left": right_pct_villager_left,
        "left_pct_villager_right": left_pct_villager_right,
        "left_side_bias_pct": number(row, "left_side_bias_pct"),
        "villager_choice_bias_pct": number(row, "villager_choice_bias_pct"),
        "segments_left_layout": row.get("segments_left_layout", ""),
        "segments_right_layout": row.get("segments_right_layout", ""),
        "left_run_dir": left_run_dir,
        "right_run_dir": right_run_dir,
        "left_trajectory": candidate_asset(
            left_run_dir,
            ("trajectory_topdown_reset.png", "trajectory_topdown.png"),
            artifact_root,
            embed_assets,
        ),
        "right_trajectory": candidate_asset(
            right_run_dir,
            ("trajectory_topdown_reset.png", "trajectory_topdown.png"),
            artifact_root,
            embed_assets,
        ),
        "left_first_frame": candidate_asset(
            left_run_dir,
            ("first_frame_fullres.png", "first_frame.png"),
            artifact_root,
            embed_assets,
        ),
        "right_first_frame": candidate_asset(
            right_run_dir,
            ("first_frame_fullres.png", "first_frame.png"),
            artifact_root,
            embed_assets,
        ),
    }


def load_datasets(
    input_specs: list[str],
    artifact_root: Path | None,
    embed_assets: bool,
) -> dict[str, list[dict[str, Any]]]:
    if input_specs:
        parsed = [parse_input(item) for item in input_specs]
    else:
        parsed = [(None, path) for path in DEFAULT_CACHE_FILES if path.exists()]
    if not parsed:
        raise FileNotFoundError(
            "no --input files supplied and no default T-maze cache CSVs were found"
        )

    datasets: dict[str, list[dict[str, Any]]] = {}
    for explicit_label, path in parsed:
        if not path.exists():
            raise FileNotFoundError(path)
        label = explicit_label or infer_seed_label(path)
        datasets[label] = [
            build_point(row, label, artifact_root, embed_assets) for row in read_rows(path)
        ]
    return datasets


def render_html(
    datasets: dict[str, list[dict[str, Any]]],
    title: str,
    subtitle: str,
    show_ci_default: bool,
) -> str:
    data_json = json.dumps(datasets, separators=(",", ":")).replace("</", "<\\/")
    template = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>__TITLE_TEXT__</title>
  <style>
    :root {
      --bg: #f3efe3;
      --panel: #fffdf7;
      --ink: #16120f;
      --muted: #665c52;
      --line: rgba(22, 18, 15, 0.18);
      --point: #0072b2;
      --point-stroke: #0f1720;
      --ci: rgba(0, 114, 178, 0.38);
      --accent: #d55e00;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      color: var(--ink);
      font-family: ui-sans-serif, "Helvetica Neue", Arial, sans-serif;
      background:
        radial-gradient(circle at 8% 4%, rgba(213, 94, 0, 0.12), transparent 24rem),
        linear-gradient(180deg, #f6f0e4 0%, var(--bg) 100%);
    }
    .page {
      width: min(96vw, 1440px);
      margin: 20px auto;
      padding: 18px;
      background: var(--panel);
      border: 1px solid rgba(22, 18, 15, 0.10);
      box-shadow: 0 18px 45px rgba(60, 45, 32, 0.13);
    }
    h1 { margin: 0 0 4px; font-size: 26px; line-height: 1.12; }
    p { margin: 0; color: var(--muted); font-size: 14px; line-height: 1.4; }
    .toolbar {
      display: flex;
      flex-wrap: wrap;
      gap: 10px 14px;
      align-items: center;
      margin: 14px 0;
      padding: 10px 12px;
      background: #f8f3e8;
      border: 1px solid rgba(22, 18, 15, 0.08);
    }
    label { color: var(--muted); font-size: 13px; }
    select, input[type="search"] {
      margin-left: 5px;
      padding: 5px 7px;
      border: 1px solid rgba(22, 18, 15, 0.18);
      background: #fffdfa;
      color: var(--ink);
      font: inherit;
    }
    input[type="checkbox"] { transform: translateY(1px); }
    .content {
      display: grid;
      grid-template-columns: minmax(520px, 1fr) 420px;
      gap: 16px;
      align-items: start;
    }
    .plot-card, .detail {
      background: #fffefa;
      border: 1px solid rgba(22, 18, 15, 0.09);
    }
    .plot-wrap { width: 100%; aspect-ratio: 1 / 1; }
    svg { display: block; width: 100%; height: 100%; }
    .detail { padding: 14px; min-height: 520px; }
    .detail h2 { margin: 0 0 5px; font-size: 20px; line-height: 1.16; }
    .model-id { color: var(--muted); font-size: 12px; word-break: break-word; }
    .stats {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 6px 10px;
      margin: 12px 0;
      font-size: 13px;
    }
    .stats strong { display: block; color: var(--muted); font-weight: 500; }
    .media {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
      margin-top: 10px;
    }
    figure { margin: 0; }
    figcaption { margin: 0 0 4px; color: var(--muted); font-size: 12px; }
    .thumb {
      display: grid;
      place-items: center;
      min-height: 116px;
      padding: 8px;
      border: 1px dashed rgba(22, 18, 15, 0.18);
      background: #faf5eb;
      color: var(--muted);
      font-size: 12px;
      text-align: center;
      text-decoration: none;
    }
    .thumb img { max-width: 100%; height: auto; display: block; }
    .runs {
      margin-top: 12px;
      padding-top: 10px;
      border-top: 1px solid rgba(22, 18, 15, 0.10);
      color: var(--muted);
      font-size: 12px;
      word-break: break-word;
    }
    .tooltip {
      position: fixed;
      pointer-events: none;
      z-index: 20;
      display: none;
      padding: 6px 8px;
      background: #16120f;
      color: #fffdf7;
      border-radius: 4px;
      font-size: 12px;
      box-shadow: 0 10px 24px rgba(0, 0, 0, 0.22);
    }
    @media (max-width: 980px) {
      .content { grid-template-columns: 1fr; }
      .plot-wrap { min-height: 520px; }
    }
  </style>
</head>
<body>
  <main class="page">
    <h1>__TITLE_HTML__</h1>
    <p>__SUBTITLE_HTML__</p>
    <div class="toolbar">
      <label>Environment seed
        <select id="seed-select"></select>
      </label>
      <label>
        <input id="show-ci" type="checkbox" __SHOW_CI_CHECKED__>
        show confidence intervals
      </label>
      <label>Highlight agent
        <input id="agent-search" type="search" placeholder="e.g. 131">
      </label>
    </div>
    <section class="content">
      <div class="plot-card">
        <div class="plot-wrap">
          <svg id="plot" viewBox="0 0 1000 1000" role="img" aria-label="T-maze choice scatter"></svg>
        </div>
      </div>
      <aside class="detail">
        <h2 id="detail-title">Select an agent</h2>
        <div id="detail-model-id" class="model-id"></div>
        <div id="detail-stats" class="stats"></div>
        <div class="media">
          <figure>
            <figcaption>Villager left trajectory</figcaption>
            <a id="left-trajectory" class="thumb">not bundled</a>
          </figure>
          <figure>
            <figcaption>Villager right trajectory</figcaption>
            <a id="right-trajectory" class="thumb">not bundled</a>
          </figure>
          <figure>
            <figcaption>Villager left first frame</figcaption>
            <a id="left-frame" class="thumb">not bundled</a>
          </figure>
          <figure>
            <figcaption>Villager right first frame</figcaption>
            <a id="right-frame" class="thumb">not bundled</a>
          </figure>
        </div>
        <div id="detail-runs" class="runs"></div>
      </aside>
    </section>
  </main>
  <div id="tooltip" class="tooltip"></div>
  <script>
    const datasets = __DATA_JSON__;
    const W = 1000;
    const H = 1000;
    const M = { left: 102, right: 34, top: 46, bottom: 92 };
    const innerW = W - M.left - M.right;
    const innerH = H - M.top - M.bottom;
    const seedSelect = document.getElementById("seed-select");
    const showCi = document.getElementById("show-ci");
    const search = document.getElementById("agent-search");
    const svg = document.getElementById("plot");
    const tooltip = document.getElementById("tooltip");
    let activeSeed = Object.keys(datasets)[0];
    let pinned = null;

    Object.keys(datasets).forEach((seed) => {
      const opt = document.createElement("option");
      opt.value = seed;
      opt.textContent = seed;
      seedSelect.appendChild(opt);
    });

    function sx(v) { return M.left + (v / 100) * innerW; }
    function sy(v) { return H - M.bottom - (v / 100) * innerH; }
    function clearSvg() { while (svg.firstChild) svg.removeChild(svg.firstChild); }
    function add(tag, attrs, parent = svg) {
      const el = document.createElementNS("http://www.w3.org/2000/svg", tag);
      Object.entries(attrs).forEach(([k, v]) => el.setAttribute(k, String(v)));
      parent.appendChild(el);
      return el;
    }
    function text(label, x, y, anchor, size, weight = 500, fill = "#16120f") {
      const el = add("text", {
        x, y, "text-anchor": anchor, "font-size": size,
        "font-weight": weight, fill
      });
      el.textContent = label;
      return el;
    }
    function fmt(v) {
      if (v === null || v === undefined || Number.isNaN(Number(v))) return "n/a";
      return `${Number(v).toFixed(1)}%`;
    }
    function pointMatches(point) {
      const q = search.value.trim().toLowerCase();
      if (!q) return true;
      return String(point.short_label).toLowerCase().includes(q)
        || String(point.model_id).toLowerCase().includes(q);
    }

    function drawAxes() {
      add("rect", { x: 0, y: 0, width: W, height: H, fill: "#fffefa" });
      for (let t = 0; t <= 100; t += 25) {
        add("line", { x1: sx(t), y1: sy(0), x2: sx(t), y2: sy(100), stroke: "var(--line)", "stroke-width": 1 });
        add("line", { x1: sx(0), y1: sy(t), x2: sx(100), y2: sy(t), stroke: "var(--line)", "stroke-width": 1 });
        text(String(t), sx(t), H - 56, "middle", 15, 400, "#665c52");
        text(String(t), 76, sy(t) + 5, "end", 15, 400, "#665c52");
      }
      add("line", { x1: sx(50), y1: sy(0), x2: sx(50), y2: sy(100), stroke: "rgba(22,18,15,0.42)", "stroke-width": 2, "stroke-dasharray": "8 8" });
      add("line", { x1: sx(0), y1: sy(50), x2: sx(100), y2: sy(50), stroke: "rgba(22,18,15,0.42)", "stroke-width": 2, "stroke-dasharray": "8 8" });
      add("line", { x1: sx(0), y1: sy(0), x2: sx(100), y2: sy(100), stroke: "rgba(0, 158, 115, 0.70)", "stroke-width": 2, "stroke-dasharray": "6 6" });
      add("rect", { x: sx(0), y: sy(100), width: innerW, height: innerH, fill: "none", stroke: "#16120f", "stroke-width": 2 });
      text("always left", sx(2), sy(98), "start", 18, 650);
      text("regular tree", sx(98), sy(98), "end", 18, 650);
      text("villager tree", sx(2), sy(2), "start", 18, 650);
      text("always right", sx(98), sy(2), "end", 18, 650);
      text("Right choice rate when villager tree is on the left (%)", W / 2, H - 22, "middle", 18, 500);
      const yLabel = text("Left choice rate when villager tree is on the right (%)", 24, H / 2, "middle", 18, 500);
      yLabel.setAttribute("transform", `rotate(-90 24 ${H / 2})`);
    }

    function drawCi(point, dimmed) {
      if (!showCi.checked) return;
      if ([point.x_ci_low, point.x_ci_high, point.y_ci_low, point.y_ci_high].some((v) => v === null || v === undefined)) return;
      const stroke = dimmed ? "rgba(0, 114, 178, 0.12)" : "var(--ci)";
      add("line", { x1: sx(point.x_ci_low), y1: sy(point.y), x2: sx(point.x_ci_high), y2: sy(point.y), stroke, "stroke-width": 2 });
      add("line", { x1: sx(point.x), y1: sy(point.y_ci_low), x2: sx(point.x), y2: sy(point.y_ci_high), stroke, "stroke-width": 2 });
    }

    function drawPoint(point) {
      const matched = pointMatches(point);
      const selected = pinned && pinned.model_id === point.model_id && pinned.seed_label === point.seed_label;
      const radius = selected ? 8.5 : 5.5;
      const opacity = matched ? 0.82 : 0.16;
      const circle = add("circle", {
        cx: sx(point.x),
        cy: sy(point.y),
        r: radius,
        fill: "var(--point)",
        stroke: selected || matched ? "var(--point-stroke)" : "none",
        "stroke-width": selected ? 3 : 1.2,
        opacity
      });
      circle.style.cursor = "pointer";
      circle.addEventListener("mouseenter", (event) => {
        tooltip.style.display = "block";
        tooltip.textContent = `Agent ${point.short_label}: x ${fmt(point.x)}, y ${fmt(point.y)}`;
        setDetail(point, false);
      });
      circle.addEventListener("mousemove", (event) => {
        tooltip.style.left = `${event.clientX + 12}px`;
        tooltip.style.top = `${event.clientY + 12}px`;
      });
      circle.addEventListener("mouseleave", () => {
        tooltip.style.display = "none";
        if (pinned) setDetail(pinned, false);
      });
      circle.addEventListener("click", () => {
        pinned = point;
        setDetail(point, true);
        draw();
      });
    }

    function setAsset(link, asset) {
      link.replaceChildren();
      link.removeAttribute("href");
      link.removeAttribute("target");
      if (asset && asset.available && asset.href) {
        link.href = asset.href;
        link.target = "_blank";
        link.rel = "noopener noreferrer";
        const img = document.createElement("img");
        img.src = asset.href;
        img.alt = asset.path || "rollout asset";
        link.appendChild(img);
      } else {
        link.textContent = asset && asset.path ? `not bundled: ${asset.path}` : "not bundled";
      }
    }

    function setDetail(point) {
      if (!point) return;
      document.getElementById("detail-title").textContent = `Agent ${point.short_label} - ${point.seed_label}`;
      document.getElementById("detail-model-id").textContent = point.model_id;
      document.getElementById("detail-stats").innerHTML = [
        ["x: right choice, villager left", fmt(point.x)],
        ["y: left choice, villager right", fmt(point.y)],
        ["left bias", fmt(point.left_side_bias_pct)],
        ["villager choice bias", fmt(point.villager_choice_bias_pct)],
        ["villager-left segments", point.segments_left_layout || "n/a"],
        ["villager-right segments", point.segments_right_layout || "n/a"],
      ].map(([k, v]) => `<div><strong>${k}</strong>${v}</div>`).join("");
      setAsset(document.getElementById("left-trajectory"), point.left_trajectory);
      setAsset(document.getElementById("right-trajectory"), point.right_trajectory);
      setAsset(document.getElementById("left-frame"), point.left_first_frame);
      setAsset(document.getElementById("right-frame"), point.right_first_frame);
      document.getElementById("detail-runs").innerHTML =
        `<strong>Villager-left run:</strong> ${point.left_run_dir || "n/a"}<br>` +
        `<strong>Villager-right run:</strong> ${point.right_run_dir || "n/a"}`;
    }

    function draw() {
      clearSvg();
      drawAxes();
      const points = datasets[activeSeed] || [];
      points.forEach((point) => drawCi(point, !pointMatches(point)));
      points.forEach(drawPoint);
      if (!pinned || pinned.seed_label !== activeSeed) {
        pinned = points[0] || null;
      }
      setDetail(pinned);
    }

    seedSelect.addEventListener("change", () => {
      activeSeed = seedSelect.value;
      pinned = null;
      draw();
    });
    showCi.addEventListener("change", draw);
    search.addEventListener("input", draw);
    seedSelect.value = activeSeed;
    draw();
  </script>
</body>
</html>
"""
    return (
        template.replace("__TITLE_TEXT__", html.escape(title, quote=True))
        .replace("__TITLE_HTML__", html.escape(title))
        .replace("__SUBTITLE_HTML__", html.escape(subtitle))
        .replace("__SHOW_CI_CHECKED__", "checked" if show_ci_default else "")
        .replace("__DATA_JSON__", data_json)
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a standalone interactive explorer for cached T-maze choice scatter CSVs."
    )
    parser.add_argument(
        "--input",
        action="append",
        default=[],
        help=(
            "Cache CSV to include. Use LABEL=PATH to override the dropdown label. "
            "Defaults to the included seed49 and seed481 plot-suite caches if omitted."
        ),
    )
    parser.add_argument(
        "--out-html",
        type=Path,
        default=Path("outputs/interactive/tmaze_choice_scatter_explorer.html"),
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=None,
        help="Optional root containing raw rollout folders referenced by the cache CSV run_dir columns.",
    )
    parser.add_argument(
        "--embed-assets",
        action="store_true",
        help="Embed trajectory/first-frame PNG assets as data URIs when they exist under --artifact-root.",
    )
    parser.add_argument("--show-ci-default", action="store_true")
    parser.add_argument("--title", default="Interactive T-Maze Choice Scatter")
    parser.add_argument(
        "--subtitle",
        default=(
            "Each point is one group-05 agent. Hover or click to inspect choice rates; "
            "optional raw rollout assets add top-down trajectories and first frames."
        ),
    )
    args = parser.parse_args()

    artifact_root = args.artifact_root.resolve() if args.artifact_root is not None else None
    datasets = load_datasets(args.input, artifact_root, args.embed_assets)
    args.out_html.parent.mkdir(parents=True, exist_ok=True)
    args.out_html.write_text(
        render_html(datasets, args.title, args.subtitle, args.show_ci_default),
        encoding="utf-8",
    )
    total_points = sum(len(points) for points in datasets.values())
    print(f"wrote_html={args.out_html}")
    print(f"datasets={len(datasets)}")
    print(f"points={total_points}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
