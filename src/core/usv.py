from __future__ import annotations

from dataclasses import dataclass, field

Cell = tuple[int, int]

DIR_TO_HEADING = {
    (1, 0): 0,
    (1, 1): 1,
    (0, 1): 2,
    (-1, 1): 3,
    (-1, 0): 4,
    (-1, -1): 5,
    (0, -1): 6,
    (1, -1): 7,
}


@dataclass
class USV:
    current_cell: Cell
    heading: int | None = None
    current_strip_id: int | None = None
    strip_direction: int = 1
    strip_progress: int = 0
    recent_strip_history: list[int] = field(default_factory=list)
    path: list[Cell] = field(default_factory=list)
    mode_history: list[str] = field(default_factory=list)
    visit_history: list[Cell] = field(default_factory=list)
    escape_type_history: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.current_strip_id is None:
            self.current_strip_id = self.coverage_strip_id(self.current_cell)
        if not self.path:
            self.path.append(self.current_cell)
        if not self.visit_history:
            self.visit_history.append(self.current_cell)
        if not self.mode_history:
            self.mode_history.append("normal")
        if not self.escape_type_history:
            self.escape_type_history.append("none")
        if not self.recent_strip_history:
            self.recent_strip_history.append(self.current_strip_id)

    def move_to(self, cell: Cell, mode: str, escape_type: str = "none", advance_strip: bool | None = None) -> None:
        new_heading = self.heading_to(cell)
        old_strip = self.current_strip_id
        new_strip = self.coverage_strip_id(cell)
        self.current_cell = cell
        self.heading = new_heading if new_heading is not None else self.heading
        if mode == "normal":
            should_advance_strip = new_strip != old_strip if advance_strip is None else advance_strip
            if should_advance_strip and new_strip != old_strip:
                self.strip_direction *= -1
                self.strip_progress = 0
                self.current_strip_id = new_strip
            else:
                dx = cell[0] - self.path[-1][0]
                self.strip_progress = self.strip_progress + 1 if dx * self.strip_direction > 0 else max(0, self.strip_progress - 1)
            self.recent_strip_history.append(self.current_strip_id)
            self.recent_strip_history = self.recent_strip_history[-20:]
        elif mode == "escape":
            self.current_strip_id = new_strip
            self.recent_strip_history.append(new_strip)
            self.recent_strip_history = self.recent_strip_history[-20:]
        self.path.append(cell)
        self.visit_history.append(cell)
        self.mode_history.append(mode)
        self.escape_type_history.append(escape_type)

    @staticmethod
    def coverage_strip_id(cell: Cell, scan_axis: str = "horizontal") -> int:
        return cell[1] if scan_axis == "horizontal" else cell[0]

    def heading_to(self, cell: Cell) -> int | None:
        dx = cell[0] - self.current_cell[0]
        dy = cell[1] - self.current_cell[1]
        dx = 0 if dx == 0 else (1 if dx > 0 else -1)
        dy = 0 if dy == 0 else (1 if dy > 0 else -1)
        if dx == 0 and dy == 0:
            return self.heading
        return DIR_TO_HEADING.get((dx, dy))

    def heading_change_to(self, cell: Cell) -> int:
        target = self.heading_to(cell)
        if self.heading is None or target is None:
            return 0
        diff = abs(target - self.heading) % 8
        return min(diff, 8 - diff)

    def recent_heading(self, window: int = 5) -> int | None:
        if len(self.path) < 2:
            return self.heading
        headings: list[int] = []
        segment = self.path[-window:]
        for a, b in zip(segment, segment[1:]):
            dx = 0 if b[0] == a[0] else (1 if b[0] > a[0] else -1)
            dy = 0 if b[1] == a[1] else (1 if b[1] > a[1] else -1)
            h = DIR_TO_HEADING.get((dx, dy))
            if h is not None:
                headings.append(h)
        return headings[-1] if headings else self.heading
