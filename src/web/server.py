from __future__ import annotations

import argparse
import json
import mimetypes
import threading
import time
from copy import deepcopy
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

import numpy as np

from src.core.coverage_planner import CoveragePlanner
from src.core.cell_map import OBSTACLE, CellMap
from src.utils.config import load_config


ROOT = Path(__file__).resolve().parents[2]
WEB_ROOT = ROOT / "web"
SCENARIO_ROOT = ROOT / "configs" / "scenarios"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "web_runs"
RUN_CONTROLS: dict[str, "RunControl"] = {}


class RunControl:
    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._paused = False
        self._stopped = False

    def pause(self) -> None:
        with self._condition:
            self._paused = True

    def resume(self) -> None:
        with self._condition:
            self._paused = False
            self._condition.notify_all()

    def stop(self) -> None:
        with self._condition:
            self._stopped = True
            self._paused = False
            self._condition.notify_all()

    def wait_if_paused(self) -> bool:
        with self._condition:
            while self._paused and not self._stopped:
                self._condition.wait()
            return not self._stopped


class PlannerWebHandler(SimpleHTTPRequestHandler):
    server_version = "CCPPPlannerWeb/1.0"

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send_file(WEB_ROOT / "index.html")
            return
        if parsed.path == "/api/scenarios":
            self._send_json({"scenarios": list_scenarios()})
            return
        if parsed.path == "/api/scenario":
            params = parse_qs(parsed.query)
            try:
                self._send_json(load_scenario_map(params.get("path", [""])[0]))
            except Exception as exc:  # noqa: BLE001
                self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return
        if parsed.path.startswith("/outputs/"):
            self._send_safe_file(ROOT / unquote(parsed.path.lstrip("/")), ROOT / "outputs")
            return
        self._send_safe_file(WEB_ROOT / unquote(parsed.path.lstrip("/")), WEB_ROOT)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/control":
            self._handle_control()
            return
        if parsed.path == "/api/run-stream":
            self._handle_run_stream()
            return
        if parsed.path != "/api/run":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            payload = self._read_json()
            result = run_planner(payload)
        except Exception as exc:  # noqa: BLE001 - keep web errors visible to local users.
            self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return
        self._send_json(result)

    def _handle_run_stream(self) -> None:
        try:
            payload = self._read_json()
            run_id = str(payload.get("run_id") or "")
            if not run_id:
                raise ValueError("run_id is required for streaming runs")
            scenario_path = resolve_scenario_path(str(payload.get("scenario", "")))
            cfg = load_config(scenario_path, default_path=ROOT / "configs" / "default.yaml")
            apply_web_overrides(cfg, payload)
            planner = CoveragePlanner(cfg)
            output_root = Path(cfg.get("output", {}).get("root", DEFAULT_OUTPUT_ROOT))
            delay_ms = max(0, int(payload.get("stream_delay_ms") or 0))
            control = RunControl()
            RUN_CONTROLS[run_id] = control
        except Exception as exc:  # noqa: BLE001
            self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return

        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        try:
            events = planner.run_events(output_root, save_outputs=bool(payload.get("save_outputs", True)))
            while control.wait_if_paused():
                event = next(events)
                if event["type"] == "init":
                    event.update(
                        {
                            "run_id": run_id,
                            "scenario": planner.scenario,
                            "map": planner_map(planner),
                            "outputs": output_links(output_root / planner.scenario),
                        }
                    )
                elif event["type"] == "decision":
                    event["decisionRow"] = normalize_decisions([event["decisionRow"]])[0]
                elif event["type"] == "step":
                    event["decisionRow"] = normalize_decisions([event["decisionRow"]])[0]
                elif event["type"] == "done":
                    event["outputs"] = output_links(output_root / planner.scenario)
                self._write_json_line(event)
                if event["type"] == "done":
                    break
                if delay_ms:
                    time.sleep(delay_ms / 1000)
        except StopIteration:
            return
        except BrokenPipeError:
            return
        except Exception as exc:  # noqa: BLE001
            self._write_json_line({"type": "error", "error": str(exc)})
        finally:
            RUN_CONTROLS.pop(run_id, None)

    def _handle_control(self) -> None:
        try:
            payload = self._read_json()
            run_id = str(payload.get("run_id") or "")
            action = str(payload.get("action") or "")
            control = RUN_CONTROLS.get(run_id)
            if control is None:
                raise ValueError(f"run not found: {run_id}")
            if action == "pause":
                control.pause()
            elif action == "resume":
                control.resume()
            elif action == "stop":
                control.stop()
            else:
                raise ValueError(f"unknown control action: {action}")
        except Exception as exc:  # noqa: BLE001
            self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return
        self._send_json({"ok": True, "run_id": run_id, "action": action})

    def _write_json_line(self, event: dict[str, Any]) -> None:
        line = json.dumps(to_jsonable(event), ensure_ascii=False).encode("utf-8") + b"\n"
        self.wfile.write(line)
        self.wfile.flush()

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[web] {self.address_string()} - {format % args}")

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length).decode("utf-8") if length else "{}"
        data = json.loads(raw or "{}")
        if not isinstance(data, dict):
            raise ValueError("Request body must be a JSON object")
        return data

    def _send_json(self, data: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(to_jsonable(data), ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path) -> None:
        if not path.exists() or not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        body = path.read_bytes()
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_safe_file(self, path: Path, base: Path) -> None:
        try:
            resolved = path.resolve()
            resolved.relative_to(base.resolve())
        except ValueError:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        self._send_file(resolved)


def list_scenarios() -> list[dict[str, str]]:
    scenarios = []
    for path in sorted(SCENARIO_ROOT.glob("*.yaml")):
        cfg = load_config(path, default_path=ROOT / "configs" / "default.yaml")
        name = cfg.get("scenario", {}).get("name") or path.stem
        scenarios.append({"name": str(name), "path": str(path.relative_to(ROOT))})
    return scenarios


def load_scenario_map(value: str) -> dict[str, Any]:
    scenario_path = resolve_scenario_path(value)
    cfg = load_config(scenario_path, default_path=ROOT / "configs" / "default.yaml")
    cell_map = CellMap.from_config(cfg)
    start_cfg = cfg.get("start", {"x": 0, "y": 0})
    return {
        "scenario": cfg.get("scenario", {}).get("name") or scenario_path.stem,
        "config": web_config_snapshot(cfg),
        "map": {
            "width": cell_map.width,
            "height": cell_map.height,
            "obstacles": obstacle_cells_from_map(cell_map),
            "start": [int(start_cfg.get("x", 0)), int(start_cfg.get("y", 0))],
        },
    }


def run_planner(payload: dict[str, Any]) -> dict[str, Any]:
    scenario_path = resolve_scenario_path(str(payload.get("scenario", "")))
    cfg = load_config(scenario_path, default_path=ROOT / "configs" / "default.yaml")
    apply_web_overrides(cfg, payload)

    planner = CoveragePlanner(cfg)
    output_root = Path(cfg.get("output", {}).get("root", DEFAULT_OUTPUT_ROOT))
    metrics = planner.run(output_root)
    output_dir = output_root / planner.scenario
    return {
        "scenario": planner.scenario,
        "metrics": metrics,
        "map": {
            **planner_map(planner),
        },
        "pathRows": planner.path_rows,
        "decisionRows": normalize_decisions(planner.decision_rows),
        "outputs": output_links(output_dir),
    }


def resolve_scenario_path(value: str) -> Path:
    if not value:
        raise ValueError("scenario is required")
    path = (ROOT / value).resolve()
    try:
        path.relative_to(SCENARIO_ROOT.resolve())
    except ValueError as exc:
        raise ValueError("scenario must be under configs/scenarios") from exc
    if not path.exists() or path.suffix not in {".yaml", ".yml"}:
        raise ValueError(f"scenario file not found: {value}")
    return path


def apply_web_overrides(cfg: dict[str, Any], payload: dict[str, Any]) -> None:
    output_root = Path(payload.get("output_root") or DEFAULT_OUTPUT_ROOT)
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    cfg.setdefault("output", {})["root"] = str(output_root)
    cfg.setdefault("planner", {})["max_steps"] = int(payload.get("max_steps") or cfg.get("planner", {}).get("max_steps", 30000))
    if payload.get("target_coverage_rate") is not None:
        cfg.setdefault("planner", {})["target_coverage_rate"] = float(payload["target_coverage_rate"])
    if payload.get("method"):
        cfg.setdefault("method", {})["name"] = str(payload["method"])
    cfg.setdefault("gbnn", {})["enabled"] = not bool(payload.get("no_gbnn", False))
    cfg.setdefault("rolling_optimizer", {})["enabled"] = not bool(payload.get("no_rolling", False))
    cfg.setdefault("escape", {})["enabled"] = not bool(payload.get("no_escape", False))
    apply_numeric_overrides(
        cfg.setdefault("rolling_optimizer", {}),
        payload,
        {
            "horizon": int,
            "beam_width": int,
            "record_candidate_count": int,
            "max_repeat_in_branch": int,
            "w_new_coverage": float,
            "w_activity": float,
            "w_direction": float,
            "w_turn": float,
            "w_repeat": float,
            "w_dead_zone": float,
            "w_obstacle": float,
            "w_structure": float,
            "w_immediate_backtrack": float,
        },
    )
    if "allow_immediate_backtrack" in payload:
        cfg.setdefault("rolling_optimizer", {})["allow_immediate_backtrack"] = bool(payload["allow_immediate_backtrack"])
    apply_numeric_overrides(
        cfg.setdefault("gbnn", {}),
        payload,
        {
            "iterations_per_step": int,
            "external_excitation": float,
            "obstacle_inhibition": float,
            "covered_input": float,
            "neighbor_weight": float,
            "transfer_beta": float,
        },
    )
    apply_numeric_overrides(
        cfg.setdefault("escape", {}),
        payload,
        {
            "backtracking_max_steps": int,
            "dijkstra_max_expansion": int,
            "min_uncovered_neighbors": int,
        },
    )

    animation = deepcopy(cfg.get("output", {}).get("animation", {}) or {})
    animation["enabled"] = bool(payload.get("save_animation", False))
    animation["fps"] = int(payload.get("fps") or animation.get("fps", 12))
    animation["max_frames"] = int(payload.get("max_frames") or animation.get("max_frames", 300))
    animation["playback_speed"] = float(payload.get("playback_speed") or animation.get("playback_speed", 1.0))
    cfg.setdefault("output", {})["animation"] = animation


def apply_numeric_overrides(target: dict[str, Any], payload: dict[str, Any], schema: dict[str, Any]) -> None:
    for key, caster in schema.items():
        if key in payload and payload[key] not in (None, ""):
            target[key] = caster(payload[key])


def web_config_snapshot(cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "planner": cfg.get("planner", {}),
        "method": cfg.get("method", {}),
        "rolling_optimizer": cfg.get("rolling_optimizer", {}),
        "gbnn": cfg.get("gbnn", {}),
        "escape": cfg.get("escape", {}),
        "output": cfg.get("output", {}),
    }


def planner_map(planner: CoveragePlanner) -> dict[str, Any]:
    return {
        "width": planner.cell_map.width,
        "height": planner.cell_map.height,
        "obstacles": obstacle_cells_from_map(planner.cell_map),
        "start": [int(planner.usv.path[0][0]), int(planner.usv.path[0][1])],
    }


def obstacle_cells_from_map(cell_map: CellMap) -> list[list[int]]:
    ys, xs = np.where(cell_map.grid == OBSTACLE)
    return [[int(x), int(y)] for x, y in zip(xs.tolist(), ys.tolist())]


def output_links(output_dir: Path) -> dict[str, str]:
    return {
        "base": output_url(output_dir),
        "trajectory": output_url(output_dir / "figures" / "trajectory.png"),
        "coverage": output_url(output_dir / "figures" / "coverage_map.png"),
        "activity": output_url(output_dir / "figures" / "activity_map.png"),
        "gif": output_url(output_dir / "animations" / "planning_process.gif"),
        "viewer": output_url(output_dir / "animations" / "planning_viewer.html"),
    }


def normalize_decisions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        item = dict(row)
        item["candidate_branches"] = parse_candidates(item.get("candidate_branches"))
        item["selected_branch"] = parse_branch(str(item.get("selected_branch", "")))
        result.append(item)
    return result


def parse_candidates(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return value
    if not value:
        return []
    try:
        parsed = json.loads(str(value))
        return parsed if isinstance(parsed, list) else []
    except json.JSONDecodeError:
        return []


def parse_branch(value: str) -> list[list[int]]:
    cells = []
    for item in value.split(";"):
        if not item:
            continue
        x, y = item.split(":")
        cells.append([int(x), int(y)])
    return cells


def output_url(path: Path) -> str:
    try:
        return "/" + str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def run_server(host: str = "127.0.0.1", port: int = 8000) -> None:
    server = ThreadingHTTPServer((host, port), PlannerWebHandler)
    print(f"Planner web app running at http://{host}:{port}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping planner web app.")
    finally:
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local web UI for the coverage planner")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    run_server(args.host, args.port)


if __name__ == "__main__":
    main()
