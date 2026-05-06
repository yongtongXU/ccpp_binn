# 单艇元胞级 GBNN 辅助全覆盖路径规划

本项目实现单艇、栅格元胞级的 GBNN 辅助全覆盖路径规划算法。一个元胞对应一个栅格、一个神经元和无人艇一次探测宽度；无人艇进入可通行元胞即认为完成覆盖。

## 核心算法定位

- 全覆盖路径规划是核心任务。
- GBNN 只构造覆盖需求场，不直接按全图最大活性跳转。
- 正常阶段由滚动优化算子展开有限深度候选路径树，并只执行最优分支第一步。
- 死区阶段由回溯候选和 Dijkstra 扩张候选融合选择逃逸点。
- A* / Dijkstra 只作为连接到逃逸点的图搜索工具。
- 当前仅支持单艇、静态障碍、8 邻域运动。

## 环境

```bash
conda env create -f environment.yml
conda activate BINN
```

如果本机已有 `BINN` 环境，可直接运行。

## 运行

单场景：

```bash
python main.py --scenario configs/scenarios/open_water.yaml
python main.py --scenario configs/scenarios/single_obstacle.yaml
python main.py --scenario configs/scenarios/island_obstacles.yaml
python main.py --scenario configs/scenarios/concave_area.yaml
```

全部场景：

```bash
python main.py --all
```

可选参数：

```bash
python main.py --all --output outputs/test_run --max-steps 10000
python main.py --scenario configs/scenarios/open_water.yaml --no-gbnn
python main.py --scenario configs/scenarios/open_water.yaml --no-rolling
python main.py --scenario configs/scenarios/open_water.yaml --no-escape
```

## 输出

每个场景输出到 `outputs/<scenario_name>/`：

- `figures/trajectory.png`
- `figures/activity_map.png`
- `figures/coverage_map.png`
- `data/path.csv`
- `data/decisions.csv`
- `data/escapes.csv`
- `data/metrics.csv`

全部场景汇总写入：

- `outputs/summary.csv`

## 测试

```bash
pytest -q
```
