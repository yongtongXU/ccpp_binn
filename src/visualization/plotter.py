from __future__ import annotations

import json
from pathlib import Path
import os

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/sbinn_matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
from matplotlib.colors import ListedColormap

from src.core.cell_map import COVERED, OBSTACLE, UNCOVERED, CellMap
from src.core.gbnn_field import GBNNField
from src.core.usv import USV


def _setup_path(path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def plot_trajectory(cell_map: CellMap, usv: USV, path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))
    base = np.zeros_like(cell_map.grid, dtype=int)
    base[cell_map.grid == UNCOVERED] = 0
    base[cell_map.grid == COVERED] = 1
    base[cell_map.grid == OBSTACLE] = 2
    cmap = ListedColormap(["#f7f7f7", "#d9ead3", "#303030"])
    ax.imshow(base, origin="lower", cmap=cmap, interpolation="nearest")

    def draw_segments(mode: str, escape_type: str | None, color: str, linestyle: str, label: str) -> None:
        first = True
        for i in range(1, len(usv.path)):
            if usv.mode_history[i] != mode:
                continue
            if escape_type is not None and usv.escape_type_history[i] != escape_type:
                continue
            a, b = usv.path[i - 1], usv.path[i]
            ax.plot([a[0], b[0]], [a[1], b[1]], color=color, linestyle=linestyle, linewidth=1.2, label=label if first else None)
            first = False

    draw_segments("normal", None, "#1f77b4", "-", "normal")
    draw_segments("escape", "backtracking", "#2ca02c", "--", "backtracking escape")
    draw_segments("escape", "dijkstra", "#ff7f0e", "--", "dijkstra escape")
    if usv.path:
        ax.scatter([usv.path[0][0]], [usv.path[0][1]], c="#00aa00", s=45, marker="o", label="start", zorder=5)
        ax.scatter([usv.path[-1][0]], [usv.path[-1][1]], c="#cc0000", s=45, marker="x", label="end", zorder=5)
    ax.set_title("Coverage trajectory")
    ax.set_xlim(-0.5, cell_map.width - 0.5)
    ax.set_ylim(-0.5, cell_map.height - 0.5)
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(color="#bbbbbb", linewidth=0.15)
    fig.tight_layout()
    fig.savefig(_setup_path(path), dpi=300)
    plt.close(fig)


def plot_activity_map(gbnn_field: GBNNField, path: str | Path) -> None:
    activity = gbnn_field.normalized_activity()
    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(activity, origin="lower", cmap="viridis", interpolation="nearest")
    ax.set_title("Final GBNN activity")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(_setup_path(path), dpi=300)
    plt.close(fig)


def plot_coverage_map(cell_map: CellMap, path: str | Path) -> None:
    base = np.zeros_like(cell_map.grid, dtype=int)
    base[cell_map.grid == UNCOVERED] = 0
    base[cell_map.grid == COVERED] = 1
    base[cell_map.grid == OBSTACLE] = 2
    cmap = ListedColormap(["#ffffff", "#74c476", "#222222"])
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(base, origin="lower", cmap=cmap, interpolation="nearest")
    ax.set_title("Final coverage map")
    ax.set_aspect("equal")
    ax.grid(color="#bbbbbb", linewidth=0.15)
    fig.tight_layout()
    fig.savefig(_setup_path(path), dpi=300)
    plt.close(fig)


def plot_planning_animation(
    cell_map: CellMap,
    path_rows: list[dict],
    path: str | Path,
    *,
    fps: int = 12,
    max_frames: int = 300,
) -> None:
    if not path_rows:
        return

    frames = _sample_frame_indices(len(path_rows), max_frames)
    obstacle_mask = cell_map.grid == OBSTACLE
    coverage_stack = _build_coverage_stack(cell_map, path_rows, frames)
    path_cells = [(int(row["x"]), int(row["y"])) for row in path_rows]
    modes = [str(row.get("mode", "normal")) for row in path_rows]
    escape_types = [str(row.get("escape_type", "none")) for row in path_rows]

    cmap = ListedColormap(["#ffffff", "#83c77b", "#222222"])
    fig, ax = plt.subplots(figsize=(10, 7))
    image = ax.imshow(coverage_stack[0], origin="lower", cmap=cmap, vmin=0, vmax=2, interpolation="nearest")
    normal_line, = ax.plot([], [], color="#1f77b4", linewidth=1.4, label="normal")
    backtrack_line, = ax.plot([], [], color="#2ca02c", linestyle="--", linewidth=1.4, label="backtracking escape")
    dijkstra_line, = ax.plot([], [], color="#ff7f0e", linestyle="--", linewidth=1.4, label="dijkstra escape")
    start_marker = ax.scatter([path_cells[0][0]], [path_cells[0][1]], c="#00aa00", s=45, marker="o", label="start", zorder=5)
    current_marker = ax.scatter([path_cells[0][0]], [path_cells[0][1]], c="#cc0000", s=55, marker="x", label="current", zorder=6)
    title = ax.set_title("")

    ax.set_xlim(-0.5, cell_map.width - 0.5)
    ax.set_ylim(-0.5, cell_map.height - 0.5)
    ax.set_aspect("equal")
    ax.grid(color="#bbbbbb", linewidth=0.15)
    ax.legend(loc="upper right", fontsize=8)

    total_free = int(np.count_nonzero(~obstacle_mask))

    def update(frame_pos: int):
        row_idx = frames[frame_pos]
        image.set_data(coverage_stack[frame_pos])
        _set_segment_line(normal_line, path_cells, modes, escape_types, row_idx, "normal", None)
        _set_segment_line(backtrack_line, path_cells, modes, escape_types, row_idx, "escape", "backtracking")
        _set_segment_line(dijkstra_line, path_cells, modes, escape_types, row_idx, "escape", "dijkstra")
        current_cell = path_cells[row_idx]
        current_marker.set_offsets(np.array([[current_cell[0], current_cell[1]]]))
        covered = int(np.count_nonzero((coverage_stack[frame_pos] == 1) & ~obstacle_mask))
        coverage_rate = covered / total_free if total_free else 1.0
        title.set_text(f"Coverage planning process | step {row_idx} | coverage {coverage_rate:.1%}")
        return image, normal_line, backtrack_line, dijkstra_line, start_marker, current_marker, title

    interval_ms = int(1000 / max(1, fps))
    anim = FuncAnimation(fig, update, frames=len(frames), interval=interval_ms, blit=False)
    output_path = _setup_path(path)
    anim.save(output_path, writer=PillowWriter(fps=max(1, fps)))
    plt.close(fig)


def _sample_frame_indices(length: int, max_frames: int) -> list[int]:
    if length <= 1:
        return [0]
    max_frames = max(2, int(max_frames))
    if length <= max_frames:
        return list(range(length))
    indices = np.linspace(0, length - 1, max_frames, dtype=int).tolist()
    indices[-1] = length - 1
    return sorted(set(indices))


def _build_coverage_stack(cell_map: CellMap, path_rows: list[dict], frames: list[int]) -> list[np.ndarray]:
    base = np.zeros_like(cell_map.grid, dtype=int)
    base[cell_map.grid == OBSTACLE] = 2
    stack: list[np.ndarray] = []
    visited = np.zeros_like(cell_map.grid, dtype=bool)
    frame_set = set(frames)
    for idx, row in enumerate(path_rows):
        x, y = int(row["x"]), int(row["y"])
        if 0 <= x < cell_map.width and 0 <= y < cell_map.height and cell_map.grid[y, x] != OBSTACLE:
            visited[y, x] = True
        if idx in frame_set:
            frame = base.copy()
            frame[visited & (cell_map.grid != OBSTACLE)] = 1
            stack.append(frame)
    return stack


def _set_segment_line(
    line,
    path_cells: list[tuple[int, int]],
    modes: list[str],
    escape_types: list[str],
    row_idx: int,
    mode: str,
    escape_type: str | None,
) -> None:
    xs: list[float] = []
    ys: list[float] = []
    for i in range(1, row_idx + 1):
        if modes[i] != mode:
            continue
        if escape_type is not None and escape_types[i] != escape_type:
            continue
        a, b = path_cells[i - 1], path_cells[i]
        xs.extend([a[0], b[0], np.nan])
        ys.extend([a[1], b[1], np.nan])
    line.set_data(xs, ys)


def plot_planning_viewer(
    cell_map: CellMap,
    path_rows: list[dict],
    decision_rows: list[dict],
    path: str | Path,
    *,
    playback_speed: float = 1.0,
) -> None:
    if not path_rows:
        return
    payload = {
        "width": cell_map.width,
        "height": cell_map.height,
        "obstacles": _obstacle_cells(cell_map),
        "pathRows": [_clean_row(row) for row in path_rows],
        "decisionRows": [_clean_decision(row) for row in decision_rows],
        "playbackSpeed": float(playback_speed),
    }
    html = _viewer_html(json.dumps(payload, ensure_ascii=True))
    output_path = _setup_path(path)
    output_path.write_text(html, encoding="utf-8")


def _obstacle_cells(cell_map: CellMap) -> list[list[int]]:
    ys, xs = np.where(cell_map.grid == OBSTACLE)
    return [[int(x), int(y)] for x, y in zip(xs.tolist(), ys.tolist())]


def _clean_row(row: dict) -> dict:
    return {
        "step": int(row.get("step", 0)),
        "x": int(row.get("x", 0)),
        "y": int(row.get("y", 0)),
        "mode": str(row.get("mode", "normal")),
        "escape_type": str(row.get("escape_type", "none")),
        "coverage_rate": float(row.get("coverage_rate", 0.0)),
        "repeated_coverage_rate": float(row.get("repeated_coverage_rate", 0.0)),
    }


def _clean_decision(row: dict) -> dict:
    candidates = row.get("candidate_branches", [])
    if isinstance(candidates, str):
        try:
            candidates = json.loads(candidates)
        except json.JSONDecodeError:
            candidates = []
    tree = row.get("candidate_tree", {"levels": []})
    if isinstance(tree, str):
        try:
            tree = json.loads(tree)
        except json.JSONDecodeError:
            tree = {"levels": []}
    return {
        "step": int(row.get("step", 0)),
        "current": [int(row.get("current_x", 0)), int(row.get("current_y", 0))],
        "selected": [int(row.get("selected_x", 0)), int(row.get("selected_y", 0))],
        "mode": str(row.get("mode", "normal")),
        "escape_type": str(row.get("escape_type", "none")),
        "method": str(row.get("method", "")),
        "score": _safe_float(row.get("branch_score")),
        "selected_branch": _parse_branch(str(row.get("selected_branch", ""))),
        "candidate_branches": candidates,
        "candidate_tree": tree,
    }


def _safe_float(value) -> float | None:
    try:
        if value is None:
            return None
        result = float(value)
        if np.isnan(result):
            return None
        return result
    except (TypeError, ValueError):
        return None


def _parse_branch(value: str) -> list[list[int]]:
    result: list[list[int]] = []
    for item in value.split(";"):
        if not item:
            continue
        x, y = item.split(":")
        result.append([int(x), int(y)])
    return result


def _viewer_html(payload_json: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Coverage Planning Viewer</title>
  <style>
    :root {{
      color-scheme: light;
      font-family: Arial, Helvetica, sans-serif;
      --bg: #f4f6f8;
      --panel: #ffffff;
      --ink: #17202a;
      --muted: #5d6d7e;
      --grid: #c9d1d9;
      --normal: #1f77b4;
      --escape-back: #2ca02c;
      --escape-dij: #ff7f0e;
      --selected: #d62728;
      --candidate: #7b61ff;
    }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
    }}
    main {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) 320px;
      gap: 16px;
      min-height: 100vh;
      padding: 16px;
      box-sizing: border-box;
    }}
    .stage, .side {{
      background: var(--panel);
      border: 1px solid #d8dee6;
      border-radius: 8px;
      box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04);
    }}
    .stage {{
      display: flex;
      align-items: center;
      justify-content: center;
      min-height: 520px;
      overflow: hidden;
    }}
    canvas {{
      width: 100%;
      height: 100%;
      max-height: calc(100vh - 34px);
      display: block;
    }}
    .side {{
      padding: 14px;
      box-sizing: border-box;
    }}
    h1 {{
      font-size: 18px;
      margin: 0 0 12px;
    }}
    .stat {{
      display: grid;
      grid-template-columns: 112px 1fr;
      gap: 6px;
      font-size: 13px;
      margin: 6px 0;
    }}
    .stat span:first-child {{
      color: var(--muted);
    }}
    .controls {{
      display: grid;
      gap: 10px;
      margin-top: 14px;
    }}
    .buttons {{
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 8px;
    }}
    button {{
      height: 34px;
      border: 1px solid #c6ced8;
      border-radius: 6px;
      background: #f8fafc;
      color: var(--ink);
      cursor: pointer;
      font-size: 13px;
    }}
    button:hover {{
      background: #eef3f7;
    }}
    label {{
      display: grid;
      gap: 4px;
      color: var(--muted);
      font-size: 12px;
    }}
    input[type="range"] {{
      width: 100%;
    }}
    .legend {{
      margin-top: 14px;
      display: grid;
      gap: 7px;
      font-size: 12px;
      color: var(--muted);
    }}
    .swatch {{
      display: inline-block;
      width: 18px;
      height: 3px;
      vertical-align: middle;
      margin-right: 7px;
      background: var(--normal);
    }}
    .candidate {{ background: var(--candidate); }}
    .selected {{ background: var(--selected); }}
    .escapeBack {{ background: var(--escape-back); }}
    .escapeDij {{ background: var(--escape-dij); }}
    @media (max-width: 900px) {{
      main {{
        grid-template-columns: 1fr;
      }}
      .stage {{
        min-height: 420px;
      }}
    }}
  </style>
</head>
<body>
  <main>
    <section class="stage">
      <canvas id="canvas"></canvas>
    </section>
    <aside class="side">
      <h1>Planning Process</h1>
      <div class="stat"><span>step</span><strong id="stepText">0</strong></div>
      <div class="stat"><span>coverage</span><strong id="coverageText">0%</strong></div>
      <div class="stat"><span>mode</span><strong id="modeText">normal</strong></div>
      <div class="stat"><span>selected</span><strong id="selectedText">-</strong></div>
      <div class="stat"><span>score</span><strong id="scoreText">-</strong></div>
      <div class="stat"><span>candidates</span><strong id="candidateText">0</strong></div>
      <div class="controls">
        <div class="buttons">
          <button id="prevBtn">Prev</button>
          <button id="playBtn">Pause</button>
          <button id="nextBtn">Next</button>
        </div>
        <label>Progress
          <input id="stepSlider" type="range" min="0" max="0" value="0">
        </label>
        <label>Speed <strong id="speedText">1.00x</strong>
          <input id="speedSlider" type="range" min="0.1" max="5" step="0.1" value="1">
        </label>
      </div>
      <div class="legend">
        <div><span class="swatch"></span>executed normal path</div>
        <div><span class="swatch escapeBack"></span>backtracking escape</div>
        <div><span class="swatch escapeDij"></span>dijkstra escape</div>
        <div><span class="swatch candidate"></span>candidate branches</div>
        <div><span class="swatch selected"></span>selected branch and current cell</div>
      </div>
    </aside>
  </main>
  <script>
    const data = {payload_json};
    const canvas = document.getElementById("canvas");
    const ctx = canvas.getContext("2d");
    const stepSlider = document.getElementById("stepSlider");
    const speedSlider = document.getElementById("speedSlider");
    const playBtn = document.getElementById("playBtn");
    const stepText = document.getElementById("stepText");
    const coverageText = document.getElementById("coverageText");
    const modeText = document.getElementById("modeText");
    const selectedText = document.getElementById("selectedText");
    const scoreText = document.getElementById("scoreText");
    const candidateText = document.getElementById("candidateText");
    const speedText = document.getElementById("speedText");
    const decisions = new Map(data.decisionRows.map(row => [row.step, row]));
    const obstacles = new Set(data.obstacles.map(cell => `${{cell[0]}},${{cell[1]}}`));
    let index = 0;
    let playing = true;
    let speed = Number(data.playbackSpeed || 1);
    let lastTime = 0;

    stepSlider.max = Math.max(0, data.pathRows.length - 1);
    speedSlider.value = String(speed);
    speedText.textContent = `${{speed.toFixed(2)}}x`;

    function resizeCanvas() {{
      const box = canvas.parentElement.getBoundingClientRect();
      const ratio = window.devicePixelRatio || 1;
      canvas.width = Math.max(320, Math.floor(box.width * ratio));
      canvas.height = Math.max(320, Math.floor(box.height * ratio));
      draw();
    }}

    function toCanvas(cell, metrics) {{
      return [
        metrics.left + (cell[0] + 0.5) * metrics.cell,
        metrics.top + (data.height - cell[1] - 0.5) * metrics.cell,
      ];
    }}

    function metrics() {{
      const padding = 26 * (window.devicePixelRatio || 1);
      const cell = Math.min((canvas.width - padding * 2) / data.width, (canvas.height - padding * 2) / data.height);
      return {{
        cell,
        left: (canvas.width - cell * data.width) / 2,
        top: (canvas.height - cell * data.height) / 2,
      }};
    }}

    function coveredCells(stepIndex) {{
      const covered = new Set();
      for (let i = 0; i <= stepIndex; i++) {{
        const row = data.pathRows[i];
        covered.add(`${{row.x}},${{row.y}}`);
      }}
      return covered;
    }}

    function drawGrid(m, covered) {{
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      for (let y = 0; y < data.height; y++) {{
        for (let x = 0; x < data.width; x++) {{
          const key = `${{x}},${{y}}`;
          const px = m.left + x * m.cell;
          const py = m.top + (data.height - y - 1) * m.cell;
          ctx.fillStyle = obstacles.has(key) ? "#222222" : covered.has(key) ? "#83c77b" : "#f8fafc";
          ctx.fillRect(px, py, m.cell, m.cell);
          ctx.strokeStyle = "#c9d1d9";
          ctx.lineWidth = Math.max(0.5, m.cell * 0.025);
          ctx.strokeRect(px, py, m.cell, m.cell);
        }}
      }}
    }}

    function strokePath(cells, m, color, width, dash = []) {{
      if (!cells || cells.length === 0) return;
      ctx.save();
      ctx.strokeStyle = color;
      ctx.lineWidth = width;
      ctx.setLineDash(dash);
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      ctx.beginPath();
      cells.forEach((cell, i) => {{
        const [x, y] = toCanvas(cell, m);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }});
      ctx.stroke();
      ctx.restore();
    }}

    function drawPathNodes(cells, m, color, radiusScale) {{
      if (!cells || cells.length === 0) return;
      ctx.save();
      ctx.fillStyle = color;
      cells.forEach((cell) => {{
        const [x, y] = toCanvas(cell, m);
        ctx.beginPath();
        ctx.arc(x, y, Math.max(2.5, m.cell * radiusScale), 0, Math.PI * 2);
        ctx.fill();
      }});
      ctx.restore();
    }}

    function drawExecuted(m) {{
      let normal = [];
      let back = [];
      let dij = [];
      for (let i = 1; i <= index; i++) {{
        const prev = data.pathRows[i - 1];
        const row = data.pathRows[i];
        const segment = [[prev.x, prev.y], [row.x, row.y]];
        if (row.mode === "escape" && row.escape_type === "backtracking") back.push(segment);
        else if (row.mode === "escape" && row.escape_type === "dijkstra") dij.push(segment);
        else normal.push(segment);
      }}
      normal.forEach(seg => strokePath(seg, m, "#1f77b4", Math.max(2, m.cell * 0.1)));
      back.forEach(seg => strokePath(seg, m, "#2ca02c", Math.max(2, m.cell * 0.1), [8, 5]));
      dij.forEach(seg => strokePath(seg, m, "#ff7f0e", Math.max(2, m.cell * 0.1), [8, 5]));
    }}

    function drawDecision(m, decision) {{
      if (!decision) return;
      const origin = decision.current;
      drawCandidateTree(m, origin, decision.candidate_tree);
      const candidates = decision.candidate_branches || [];
      candidates.slice().reverse().forEach((candidate, rank) => {{
        const alpha = Math.max(0.12, 0.45 - rank * 0.015);
        const cells = [origin, ...(candidate.path || [])];
        ctx.globalAlpha = alpha;
        strokePath(cells, m, "#7b61ff", Math.max(1.4, m.cell * 0.045));
        drawPathNodes(cells.slice(1), m, "#7b61ff", 0.075);
        ctx.globalAlpha = 1;
      }});
      const selectedCells = [origin, ...(decision.selected_branch || [])];
      strokePath(selectedCells, m, "#d62728", Math.max(2.5, m.cell * 0.13));
      drawPathNodes(selectedCells.slice(1), m, "#d62728", 0.1);
    }}

    function drawCandidateTree(m, origin, tree) {{
      const levels = tree && Array.isArray(tree.levels) ? tree.levels : [];
      levels.forEach((level) => {{
        const depth = Math.max(1, Number(level.depth || 1));
        const branches = Array.isArray(level.branches) ? level.branches : [];
        const alpha = Math.max(0.12, 0.5 - depth * 0.08);
        const width = Math.max(1.1, m.cell * (0.07 - Math.min(depth, 5) * 0.006));
        ctx.globalAlpha = alpha;
        branches.forEach((branch) => {{
          const cells = [origin, ...(branch.path || [])];
          strokePath(cells, m, "#4f67c8", width);
        }});
        ctx.globalAlpha = Math.min(0.6, alpha + 0.1);
        branches.forEach((branch) => {{
          drawPathNodes((branch.path || []).slice(-1), m, "#4f67c8", 0.055);
        }});
        ctx.globalAlpha = 1;
      }});
    }}

    function drawMarker(cell, m, color, radiusScale) {{
      const [x, y] = toCanvas(cell, m);
      ctx.beginPath();
      ctx.fillStyle = color;
      ctx.arc(x, y, Math.max(4, m.cell * radiusScale), 0, Math.PI * 2);
      ctx.fill();
    }}

    function draw() {{
      const row = data.pathRows[index];
      const decision = decisions.get(row.step + 1) || decisions.get(row.step);
      const m = metrics();
      drawGrid(m, coveredCells(index));
      drawExecuted(m);
      drawDecision(m, decision);
      drawMarker([data.pathRows[0].x, data.pathRows[0].y], m, "#00aa00", 0.18);
      drawMarker([row.x, row.y], m, "#d62728", 0.22);
      stepText.textContent = String(row.step);
      coverageText.textContent = `${{(row.coverage_rate * 100).toFixed(1)}}%`;
      modeText.textContent = row.escape_type && row.escape_type !== "none" ? `${{row.mode}} / ${{row.escape_type}}` : row.mode;
      selectedText.textContent = decision ? `${{decision.selected[0]}}, ${{decision.selected[1]}}` : "-";
      scoreText.textContent = decision && decision.score !== null ? decision.score.toFixed(2) : "-";
      candidateText.textContent = decision ? String((decision.candidate_branches || []).length) : "0";
      stepSlider.value = String(index);
    }}

    function setIndex(next) {{
      index = Math.max(0, Math.min(data.pathRows.length - 1, next));
      draw();
    }}

    function tick(time) {{
      const delay = 500 / Math.max(0.1, speed);
      if (playing && time - lastTime >= delay) {{
        setIndex(index >= data.pathRows.length - 1 ? 0 : index + 1);
        lastTime = time;
      }}
      requestAnimationFrame(tick);
    }}

    playBtn.addEventListener("click", () => {{
      playing = !playing;
      playBtn.textContent = playing ? "Pause" : "Play";
    }});
    document.getElementById("prevBtn").addEventListener("click", () => setIndex(index - 1));
    document.getElementById("nextBtn").addEventListener("click", () => setIndex(index + 1));
    stepSlider.addEventListener("input", event => setIndex(Number(event.target.value)));
    speedSlider.addEventListener("input", event => {{
      speed = Number(event.target.value);
      speedText.textContent = `${{speed.toFixed(2)}}x`;
    }});
    window.addEventListener("resize", resizeCanvas);
    resizeCanvas();
    requestAnimationFrame(tick);
  </script>
</body>
</html>
"""
