## 面向复杂水域异构无人艇集群的连通性保持型全覆盖路径规划方法

本项目实现二维栅格仿真下的连通性保持型全覆盖路径规划。主路径由候选扫描方向、平行扫描线、障碍切割覆盖段和牛耕式排序生成；GBNN 仅作为覆盖需求场可视化与残留补扫排序辅助，不参与主路径生成。

### 环境

```bash
conda env create -f environment.yml
conda activate BINN
```

### 运行

```bash
python main.py --scenario configs/scenarios/open_water.yaml
python main.py --scenario configs/scenarios/single_obstacle.yaml
python main.py --scenario configs/scenarios/concave_area.yaml
python main.py --scenario configs/scenarios/heterogeneous_cluster.yaml
```

可选参数：

```bash
python main.py --scenario configs/scenarios/island_obstacles.yaml --output outputs/island_run --animate
python main.py --scenario configs/scenarios/open_water.yaml --no-gbnn
python main.py --scenario configs/scenarios/open_water.yaml --animate --animation-step 1
```

### 输出

图片输出到 `outputs/figures`：

- `trajectory.png`
- `segments.png`
- `coverage_map.png`
- `gbnn_activity.png`
- `metrics_curve.png`
- `../animations/coverage.gif`，使用 `--animate` 时生成，逐步展示路径执行、当前位置和覆盖区域更新。

数据输出到 `outputs/data`：

- `paths.csv`
- `segments.csv`
- `final_metrics.csv`

### 当前简化假设

- 以二维栅格覆盖规划为主，不做 USV 动力学积分。
- 段间连接优先直线可航，必要时使用 A*。
- 异构集群采用段级启发式贪心分配，不使用 CVT、CBBA、Voronoi、强化学习或 LLM。
- GBNN 只表达未覆盖需求与残留补扫优先级，不生成主路径。
