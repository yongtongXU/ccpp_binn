from typing import List, Tuple, Set
import matplotlib.pyplot as plt

Cell = Tuple[int, int]
Path = List[Cell]

# 8 邻域动作
DIRECTIONS_8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]


def in_bounds(cell: Cell, width: int, height: int) -> bool:
    x, y = cell
    return 0 <= x < width and 0 <= y < height


def is_traversable(cell: Cell, obstacles: Set[Cell], width: int, height: int) -> bool:
    return in_bounds(cell, width, height) and cell not in obstacles


def get_neighbors8(cell: Cell, width: int, height: int, obstacles: Set[Cell]) -> List[Cell]:
    x, y = cell
    neighbors = []

    for dx, dy in DIRECTIONS_8:
        nxt = (x + dx, y + dy)
        if is_traversable(nxt, obstacles, width, height):
            neighbors.append(nxt)

    return neighbors


def generate_candidate_paths(
    start: Cell,
    width: int,
    height: int,
    obstacles: Set[Cell],
    horizon: int,
    allow_revisit: bool = True,
    allow_immediate_backtrack: bool = False,
) -> List[Path]:
    """
    从 start 出发，生成未来 horizon 步的所有候选路径。
    每条路径格式：[start, cell_1, ..., cell_H]
    """

    candidate_paths: List[Path] = []

    def dfs(path: Path):
        if len(path) == horizon + 1:
            candidate_paths.append(path.copy())
            return

        current = path[-1]
        neighbors = get_neighbors8(current, width, height, obstacles)

        for nxt in neighbors:
            if not allow_immediate_backtrack and len(path) >= 2:
                if nxt == path[-2]:
                    continue

            if not allow_revisit and nxt in path:
                continue

            path.append(nxt)
            dfs(path)
            path.pop()

    dfs([start])
    return candidate_paths


def plot_candidate_paths(
    paths: List[Path],
    start: Cell,
    obstacles: Set[Cell],
    width: int,
    height: int,
    title: str = "Rolling Candidate Path Tree",
    save_path: str | None = None,
):
    fig, ax = plt.subplots(figsize=(8, 8))

    # 画网格
    ax.set_xlim(-0.5, width - 0.5)
    ax.set_ylim(-0.5, height - 0.5)
    ax.set_xticks(range(width))
    ax.set_yticks(range(height))
    ax.grid(True, linewidth=0.8, alpha=0.5)

    # 让 y 轴方向更像矩阵坐标：y 向下增大
    ax.invert_yaxis()
    ax.set_aspect("equal")

    # 画障碍
    for obs in obstacles:
        x, y = obs
        ax.add_patch(
            plt.Rectangle(
                (x - 0.5, y - 0.5),
                1,
                1,
                facecolor="black",
                edgecolor="black",
                alpha=0.9,
            )
        )

    # 画所有候选路径
    for path in paths:
        xs = [p[0] for p in path]
        ys = [p[1] for p in path]
        ax.plot(xs, ys, linestyle="--", linewidth=1.0, alpha=0.25, color="purple")

    # 画候选路径末端
    end_cells = [path[-1] for path in paths]
    if end_cells:
        end_x = [p[0] for p in end_cells]
        end_y = [p[1] for p in end_cells]
        ax.scatter(end_x, end_y, s=25, color="red", alpha=0.5, label="Branch endpoints")

    # 画起点
    ax.scatter([start[0]], [start[1]], s=140, color="blue", edgecolor="white", linewidth=1.5, label="Current cell")

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="upper right")

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300)
        print(f"Saved figure to: {save_path}")

    plt.show()


if __name__ == "__main__":
    width = 8
    height = 8
    start = (3, 3)

    obstacles = {
        
        (5, 4),
    }

    horizon = 3

    paths = generate_candidate_paths(
        start=start,
        width=width,
        height=height,
        obstacles=obstacles,
        horizon=horizon,
        allow_revisit=True,
        allow_immediate_backtrack=False,
    )

    print(f"候选路径数量: {len(paths)}")
    for i, path in enumerate(paths):
        print(f"{i:03d}: {path}")

    plot_candidate_paths(
        paths=paths,
        start=start,
        obstacles=obstacles,
        width=width,
        height=height,
        title=f"Candidate Path Tree, H={horizon}",
        save_path="candidate_tree.png",
    )