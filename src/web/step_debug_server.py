from __future__ import annotations

import argparse
import csv
import json
import mimetypes
from copy import deepcopy
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import numpy as np

from src.core.cell_map import OBSTACLE, Cell
from src.core.coverage_planner import CoveragePlanner
from src.core.metrics import compute_metrics
from src.core.rolling_optimizer import PLANNING_MODE_LABELS
from src.core.strategy import RollingGBNNStrategy, StepDecision
from src.utils.config import load_config


ROOT = Path(__file__).resolve().parents[2]
WEB_ROOT = ROOT / "web"
SCENARIO_PATH = ROOT / "configs" / "scenarios" / "four_obstacles_15x15.yaml"
DEBUG_OUTPUT_ROOT = ROOT / "outputs" / "step_debug"
PLANNING_MODES = [
    "auto",
    "open_water",
    "junction_search",
    "corridor",
    "boundary_contact",
    "obstacle_edge",
    "pocket_entry",
    "dead_zone",
    "frontier_recovery",
    "frontier_following",
]


class StepDebugSession:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> dict[str, Any]:
        cfg = load_config(SCENARIO_PATH, default_path=ROOT / "configs" / "default.yaml")
        cfg["planner"]["max_steps"] = 2000
        cfg.setdefault("output", {}).setdefault("animation", {})["enabled"] = False
        cfg["rolling_optimizer"]["record_candidate_count"] = 80
        cfg["rolling_optimizer"]["record_tree_count"] = 80
        self.config = cfg
        self.planner = CoveragePlanner(cfg)
        self.step_index = 0
        self.history: list[dict[str, Any]] = []
        self.annotations: list[dict[str, Any]] = []
        self.step_log: list[dict[str, Any]] = []
        self.custom_modes: list[str] = []
        self.planner.cell_map.mark_covered(self.planner.usv.current_cell)
        self.planner.gbnn.initialize(self.planner.cell_map)
        self.planner._record_path_row(0, "normal", "none")
        self.planner.coverage_history.append(self.planner.cell_map.coverage_rate())
        self.planner.strategy.after_step(self.planner.cell_map.coverage_rate())
        self._save_debug_log()
        return self.state("reset")

    def state(self, event: str = "state") -> dict[str, Any]:
        planner = self.planner
        current_state = self._classify_now()
        metrics = compute_metrics(
            planner.scenario,
            planner.cell_map.coverage_rate() >= 1.0,
            planner.cell_map,
            planner.usv,
            planner.deadlock_count,
            planner.escape_rows,
            planner.failure_reason,
            method=planner.strategy.name,
        )
        return {
            "event": event,
            "scenario": planner.scenario,
            "step": self.step_index,
            "finished": planner.cell_map.coverage_rate() >= 1.0,
            "current": [int(planner.usv.current_cell[0]), int(planner.usv.current_cell[1])],
            "heading": planner.usv.heading,
            "coverage_rate": planner.cell_map.coverage_rate(),
            "repeated_coverage_rate": planner.cell_map.repeated_coverage_rate(),
            "metrics": metrics,
            "planning_state": current_state,
            "grid": planner.cell_map.grid.astype(int).tolist(),
            "visit_count": planner.cell_map.visit_count.astype(int).tolist(),
            "activity": planner.gbnn.activity.tolist() if planner.gbnn.activity is not None else [],
            "normalized_activity": planner.gbnn.normalized_activity().tolist(),
            "path": [[int(x), int(y)] for x, y in planner.usv.path],
            "neighbors": self._neighbor_options(),
            "last_decision": self._last_decision(),
            "annotations": self.annotations,
            "save_paths": self._save_paths(),
            "can_undo": bool(self.history),
            "planning_modes": self._planning_modes(),
        }

    def step(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.planner.cell_map.coverage_rate() >= 1.0:
            return self.state("finished")
        self._push_history()
        pre_step_state = self._classify_now()
        pre_cell = self.planner.usv.current_cell
        pre_step = self.step_index
        pre_coverage_rate = self.planner.cell_map.coverage_rate()
        pre_repeated_coverage_rate = self.planner.cell_map.repeated_coverage_rate()
        user_mode = self._user_mode(payload, pre_step_state)
        self._apply_debug_overrides(payload)
        self.step_index += 1
        self.planner.gbnn.update(self.planner.cell_map)
        decision = self.planner.strategy.choose_next(self.step_index, self.planner.usv, self.planner.cell_map, self.planner.gbnn)
        self._clear_forced_mode()
        suggested = decision.next_cell
        manual_cell = self._manual_cell(payload)
        if manual_cell is not None:
            decision = self._manual_decision(manual_cell, decision)
        if decision.deadlock:
            self.planner.deadlock_count += 1
        if decision.escape_record:
            self.planner.escape_rows.append(decision.escape_record)
        if decision.next_cell is None:
            self.planner.failure_reason = decision.failure_reason or decision.details.get("reason", "strategy_stopped")
            return self.state("stopped")

        self.planner.usv.move_to(
            decision.next_cell,
            decision.mode,
            decision.escape_type,
            advance_strip=decision.advance_strip,
        )
        self.planner.cell_map.mark_covered(decision.next_cell)
        decision.details["planner_suggested"] = [int(suggested[0]), int(suggested[1])] if suggested else None
        self.planner._record_decision(self.step_index, decision, decision.next_cell)
        self.planner.decision_rows[-1]["pre_step_mode"] = pre_step_state.get("mode")
        self.planner.decision_rows[-1]["pre_step_mode_label"] = pre_step_state.get("mode_label")
        self.planner.decision_rows[-1]["pre_step_mode_reason"] = pre_step_state.get("reason")
        self.planner.coverage_history.append(self.planner.cell_map.coverage_rate())
        self.planner.strategy.after_step(self.planner.cell_map.coverage_rate())
        self.planner._record_path_row(self.step_index, self.planner.usv.mode_history[-1], self.planner.usv.escape_type_history[-1])
        self.step_log.append(
            self._step_record(
                step=pre_step,
                cell=pre_cell,
                decision=decision,
                suggested=suggested,
                planning_state=pre_step_state,
                user_mode=user_mode,
                coverage_rate=pre_coverage_rate,
                repeated_coverage_rate=pre_repeated_coverage_rate,
            )
        )
        self._save_debug_log()
        return self.state("step")

    def undo(self) -> dict[str, Any]:
        if not self.history:
            return self.state("undo_empty")
        snapshot = self.history.pop()
        self.config = snapshot["config"]
        self.planner = snapshot["planner"]
        self.step_index = snapshot["step_index"]
        self.annotations = snapshot["annotations"]
        self.step_log = snapshot["step_log"]
        self.custom_modes = snapshot["custom_modes"]
        self._save_debug_log()
        return self.state("undo")

    def annotate_mode(self, payload: dict[str, Any]) -> dict[str, Any]:
        mode = self._mode_from_payload(payload, "corrected_mode", "custom_mode")
        if not mode:
            raise ValueError("corrected_mode is required")
        self._remember_mode(mode)
        current_state = self._classify_now()
        annotation = {
            "step": self.step_index,
            "cell": [int(self.planner.usv.current_cell[0]), int(self.planner.usv.current_cell[1])],
            "algorithm_mode": current_state.get("mode"),
            "algorithm_mode_label": current_state.get("mode_label"),
            "corrected_mode": mode,
            "corrected_mode_label": mode_label(mode),
            "note": str(payload.get("note") or ""),
        }
        self.annotations.append(annotation)
        self._save_debug_log()
        return self.state("annotate_mode")

    def save(self) -> dict[str, Any]:
        self._save_debug_log()
        data = self.state("save")
        data["saved"] = True
        return data

    def _rolling_config(self) -> dict[str, Any]:
        return self.config.setdefault("rolling_optimizer", {})

    def _apply_debug_overrides(self, payload: dict[str, Any]) -> None:
        rolling_cfg = self._rolling_config()
        forced_mode = self._mode_from_payload(payload, "forced_mode", "custom_forced_mode") or "auto"
        self._remember_mode(forced_mode)
        rolling_cfg["debug_forced_planning_mode"] = "" if forced_mode == "auto" else forced_mode
        if isinstance(self.planner.strategy, RollingGBNNStrategy):
            self.planner.strategy.rolling.config["debug_forced_planning_mode"] = rolling_cfg["debug_forced_planning_mode"]

    def _clear_forced_mode(self) -> None:
        self._rolling_config()["debug_forced_planning_mode"] = ""
        if isinstance(self.planner.strategy, RollingGBNNStrategy):
            self.planner.strategy.rolling.config["debug_forced_planning_mode"] = ""

    def _manual_cell(self, payload: dict[str, Any]) -> Cell | None:
        raw = payload.get("manual_next")
        if not raw:
            return None
        if isinstance(raw, str):
            parts = [p.strip() for p in raw.replace("，", ",").split(",") if p.strip()]
            if len(parts) != 2:
                raise ValueError("manual_next must be formatted as x,y")
            cell = (int(parts[0]), int(parts[1]))
        else:
            cell = (int(raw[0]), int(raw[1]))
        if cell not in self.planner.cell_map.neighbors8(self.planner.usv.current_cell):
            raise ValueError(f"manual next cell {cell} is not an adjacent traversable cell")
        return cell

    def _manual_decision(self, cell: Cell, suggested: StepDecision) -> StepDecision:
        details = {}
        if isinstance(self.planner.strategy, RollingGBNNStrategy):
            details = self.planner.strategy.rolling.score_branch(self.planner.usv, self.planner.cell_map, self.planner.gbnn, [cell])
        details["manual_override"] = True
        details["planner_suggested"] = [int(suggested.next_cell[0]), int(suggested.next_cell[1])] if suggested.next_cell else None
        details["suggested_details"] = suggested.details
        return StepDecision(next_cell=cell, branch=[cell], details=details, mode="normal", escape_type="none")

    def _mode_from_payload(self, payload: dict[str, Any], select_key: str, custom_key: str) -> str:
        custom = str(payload.get(custom_key) or "").strip()
        if custom:
            return custom
        return str(payload.get(select_key) or "").strip()

    def _user_mode(self, payload: dict[str, Any], planning_state: dict[str, Any]) -> str:
        mode = self._mode_from_payload(payload, "corrected_mode", "custom_mode")
        if mode:
            self._remember_mode(mode)
            return mode
        mode = self._mode_from_payload(payload, "forced_mode", "custom_forced_mode")
        if not mode or mode == "auto":
            return str(planning_state.get("mode") or "")
        return mode

    def _remember_mode(self, mode: str) -> None:
        if not mode or mode == "auto" or mode in PLANNING_MODES or mode in self.custom_modes:
            return
        self.custom_modes.append(mode)

    def _planning_modes(self) -> list[str]:
        return [mode_option(mode) for mode in [*PLANNING_MODES, *self.custom_modes]]

    def _push_history(self) -> None:
        self.history.append(
            {
                "config": deepcopy(self.config),
                "planner": deepcopy(self.planner),
                "step_index": self.step_index,
                "annotations": deepcopy(self.annotations),
                "step_log": deepcopy(self.step_log),
                "custom_modes": deepcopy(self.custom_modes),
            }
        )

    def _step_record(
        self,
        *,
        step: int,
        cell: Cell,
        decision: StepDecision,
        suggested: Cell | None,
        planning_state: dict[str, Any],
        user_mode: str,
        coverage_rate: float,
        repeated_coverage_rate: float,
    ) -> dict[str, Any]:
        selected = decision.next_cell
        return {
            "step": int(step),
            "x": int(cell[0]),
            "y": int(cell[1]),
            "mode": decision.mode,
            "next_pos_x": int(suggested[0]) if suggested else "",
            "next_pos_y": int(suggested[1]) if suggested else "",
            "user_next_pos_x": int(selected[0]) if selected else "",
            "user_next_pos_y": int(selected[1]) if selected else "",
            "planning_mode": planning_state.get("mode"),
            "planning_mode_label": planning_state.get("mode_label"),
            "user_planning_mode": user_mode,
            "user_planning_mode_label": mode_label(user_mode),
            "escape_type": decision.escape_type,
            "coverage_rate": coverage_rate,
            "repeated_coverage_rate": repeated_coverage_rate,
        }

    def _save_paths(self) -> dict[str, str]:
        base = DEBUG_OUTPUT_ROOT / self.planner.scenario
        return {
            "json": str(base / "step_debug_log.json"),
            "csv": str(base / "step_debug_steps.csv"),
            "annotations_csv": str(base / "mode_annotations.csv"),
        }

    def _save_debug_log(self) -> None:
        paths = self._save_paths()
        base = Path(paths["json"]).parent
        base.mkdir(parents=True, exist_ok=True)
        Path(paths["json"]).write_text(json.dumps(to_jsonable(self.step_log), ensure_ascii=False, indent=2), encoding="utf-8")
        self._write_step_csv(Path(paths["csv"]))
        self._write_annotations_csv(Path(paths["annotations_csv"]))

    def _write_step_csv(self, path: Path) -> None:
        columns = [
            "step",
            "x",
            "y",
            "mode",
            "next_pos_x",
            "next_pos_y",
            "user_next_pos_x",
            "user_next_pos_y",
            "planning_mode",
            "planning_mode_label",
            "user_planning_mode",
            "user_planning_mode_label",
            "escape_type",
            "coverage_rate",
            "repeated_coverage_rate",
        ]
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            for row in self.step_log:
                writer.writerow({key: row.get(key, "") for key in columns})

    def _write_annotations_csv(self, path: Path) -> None:
        columns = ["step", "x", "y", "algorithm_mode", "algorithm_mode_label", "corrected_mode", "corrected_mode_label", "note"]
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            for row in self.annotations:
                cell = row.get("cell") or ["", ""]
                writer.writerow(
                    {
                        "step": row.get("step"),
                        "x": cell[0],
                        "y": cell[1],
                        "algorithm_mode": row.get("algorithm_mode"),
                        "algorithm_mode_label": row.get("algorithm_mode_label"),
                        "corrected_mode": row.get("corrected_mode"),
                        "corrected_mode_label": row.get("corrected_mode_label"),
                        "note": row.get("note"),
                    }
                )

    def _classify_now(self) -> dict[str, Any]:
        if not isinstance(self.planner.strategy, RollingGBNNStrategy):
            return {}
        rolling = self.planner.strategy.rolling
        old_forced = rolling.config.get("debug_forced_planning_mode", "")
        rolling.config["debug_forced_planning_mode"] = ""
        try:
            state = rolling.classify_planning_state(self.planner.usv, self.planner.cell_map)
        finally:
            rolling.config["debug_forced_planning_mode"] = old_forced
        return {
            "mode": state.mode,
            "mode_label": mode_label(state.mode),
            "reason": state.reason,
            "traversable_neighbors": state.traversable_neighbors,
            "uncovered_neighbors": state.uncovered_neighbors,
            "obstacle_pressure": state.obstacle_pressure,
            "current_dead_zone": state.current_dead_zone,
        }

    def _neighbor_options(self) -> list[dict[str, Any]]:
        current = self.planner.usv.current_cell
        result = []
        for cell in sorted(self.planner.cell_map.neighbors8(current), key=lambda c: (c[1], c[0])):
            x, y = cell
            result.append(
                {
                    "cell": [int(x), int(y)],
                    "state": int(self.planner.cell_map.grid[y, x]),
                    "visit_count": int(self.planner.cell_map.visit_count[y, x]),
                    "activity": float(self.planner.gbnn.get_activity(cell)),
                    "uncovered": bool(self.planner.cell_map.is_uncovered(cell)),
                }
            )
        return result

    def _last_decision(self) -> dict[str, Any] | None:
        if not self.planner.decision_rows:
            return None
        row = deepcopy(self.planner.decision_rows[-1])
        for key in ("candidate_branches", "candidate_tree"):
            if isinstance(row.get(key), str):
                try:
                    row[key] = json.loads(row[key])
                except json.JSONDecodeError:
                    pass
        return row


class StepDebugHandler(SimpleHTTPRequestHandler):
    server_version = "CCPPStepDebug/1.0"

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send_file(WEB_ROOT / "step_debug.html")
            return
        if parsed.path == "/api/state":
            self._send_json(SESSION.state())
            return
        self._send_safe_file(WEB_ROOT / unquote(parsed.path.lstrip("/")), WEB_ROOT)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            payload = self._read_json()
            if parsed.path == "/api/reset":
                result = SESSION.reset()
            elif parsed.path == "/api/step":
                result = SESSION.step(payload)
            elif parsed.path == "/api/undo":
                result = SESSION.undo()
            elif parsed.path == "/api/annotate-mode":
                result = SESSION.annotate_mode(payload)
            elif parsed.path == "/api/save":
                result = SESSION.save()
            else:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
        except Exception as exc:  # noqa: BLE001
            self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return
        self._send_json(result)

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[step-debug] {self.address_string()} - {format % args}")

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


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def mode_label(mode: str) -> str:
    return PLANNING_MODE_LABELS.get(mode, mode)


def mode_option(mode: str) -> dict[str, str]:
    return {"value": mode, "label": mode_label(mode)}


SESSION = StepDebugSession()


def main() -> None:
    parser = argparse.ArgumentParser(description="Step-by-step debugger for the fixed 15x15 four-obstacle scenario")
    parser.add_argument("--port", type=int, default=8015)
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), StepDebugHandler)
    print(f"[step-debug] open http://127.0.0.1:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
