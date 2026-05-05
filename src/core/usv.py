from __future__ import annotations

from dataclasses import dataclass, field

from src.utils.geometry import Point


@dataclass
class USV:
    id: str
    start: Point
    position: Point | None = None
    heading: float = 0.0
    max_speed: float = 1.0
    coverage_width: float = 4.0
    coverage_radius: float = 2.0
    min_turn_radius: float = 1.0
    max_heading_change: float = 90.0
    endurance: float = 10000.0
    path: list[Point] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.position is None:
            self.position = self.start
        if not self.path:
            self.path = [self.start]

    @classmethod
    def from_config(cls, cfg: dict) -> "USV":
        coverage_width = float(cfg.get("coverage_width", 4.0))
        return cls(
            id=str(cfg.get("id", "USV1")),
            start=tuple(cfg.get("start", [0.5, 0.5])),
            heading=float(cfg.get("heading", 0.0)),
            max_speed=float(cfg.get("max_speed", 1.0)),
            coverage_width=coverage_width,
            coverage_radius=float(cfg.get("coverage_radius", coverage_width / 2.0)),
            min_turn_radius=float(cfg.get("min_turn_radius", 1.0)),
            max_heading_change=float(cfg.get("max_heading_change", 90.0)),
            endurance=float(cfg.get("endurance", 10000.0)),
        )
