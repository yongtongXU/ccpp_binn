# 单艇元胞级 GBNN 辅助全覆盖路径规划

本项目实现单艇、栅格元胞级的 GBNN 辅助全覆盖路径规划算法。一个元胞对应一个栅格、一个神经元和无人艇一次探测宽度；无人艇进入可通行元胞即认为完成覆盖。

## 核心算法定位

- 全覆盖路径规划是核心任务。
- GBNN 只构造覆盖需求场，不直接按全图最大活性跳转。
- 正常阶段由滚动优化算子展开有限深度候选路径树，并只执行最优分支第一步。
- 死区阶段由回溯候选和 Dijkstra 扩张候选融合选择逃逸点。
- A* / Dijkstra 只作为连接到逃逸点的图搜索工具。
- 当前仅支持单艇、静态障碍、8 邻域运动。

## 方法入口

仿真入口通过 `method.name` 选择覆盖策略，便于后续加入其他方法做对比实验：

- `rolling_gbnn`：本文方法，使用 GBNN 覆盖需求场、滚动候选分支评价和逃逸点选择。
- `gbnn_greedy`：传统单步 GBNN 基线，在当前位置 8 邻域内按神经元活性和航向连续性选点。

新增方法时，在 `src/core/strategy.py` 中实现同样的 `choose_next()` / `after_step()` 接口，并注册到 `create_strategy()` 的 registry 中即可复用同一套场景、指标和输出文件。

## 环境

```bash
conda env create -f environment.yml
conda activate BINN
```

如果本机已有 `BINN` 环境，可直接运行。

## 运行

Web 控制台：

```bash
python web_server.py --port 8000
```

然后打开 `http://127.0.0.1:8000`，可在网页中选择场景、设置算法参数、运行规划并回放每一步候选路径和最终选择。场景切换后地图会立即加载；点击运行后后端一次性完成规划，页面用进度条提示等待，并在结果返回后播放完整过程。

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
python main.py --scenario configs/scenarios/open_water.yaml --method rolling_gbnn
python main.py --scenario configs/scenarios/open_water.yaml --method gbnn_greedy
python main.py --scenario configs/scenarios/open_water.yaml --no-gbnn
python main.py --scenario configs/scenarios/open_water.yaml --no-rolling
python main.py --scenario configs/scenarios/open_water.yaml --no-escape
```

## 输出

每个场景输出到 `outputs/<scenario_name>/`：

- `figures/trajectory.png`
- `figures/activity_map.png`
- `figures/coverage_map.png`
- `animations/planning_process.gif`
- `animations/planning_viewer.html`
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
