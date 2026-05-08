# Rolling GBNN 代码阅读记录

本文档记录当前工程中 `rolling_gbnn` 方法的真实代码结构、运行流程和可修改点。后续可直接在本文档中增删、批注或标记希望修改的部分。

## 1. 总体定位

当前工程的 `rolling_gbnn` 不是单纯的传统 GBNN 邻域最大活性选点算法，而是一个混合式全覆盖路径规划流程：

1. 用 `GBNNField` 维护覆盖需求活性场。
2. 用 `RollingGBNNStrategy` 作为策略入口。
3. 优先尝试全局条带扫描计划 `global_strip_plan`。
4. 在必要时调用 `RollingOptimizer` 生成有限深度候选分支树并评分。
5. 当正常规划停滞或进入死区时，调用 `EscapeSelector` 进行回溯或 Dijkstra 逃逸。
6. 每一步只执行当前选中分支的第一步，然后更新覆盖状态并重新规划。

相关入口文件：

- `main.py`：命令行入口，加载场景配置并运行规划器。
- `src/core/coverage_planner.py`：主循环、记录路径、写输出文件。
- `src/core/strategy.py`：策略选择与 `rolling_gbnn` 方法入口。
- `src/core/rolling_optimizer.py`：滚动候选树生成、分支评分和条带优先规则。
- `src/core/gbnn_field.py`：GBNN 活性场更新。
- `src/core/escape_selector.py`：死区逃逸、回溯候选、Dijkstra 候选。
- `src/core/graph_search.py`：A* / Dijkstra 图搜索。
- `src/core/usv.py`：USV 状态、航向、路径、条带方向。
- `src/core/cell_map.py`：栅格状态、8 邻域、覆盖率。

## 2. 配置入口

默认配置位于 `configs/default.yaml`。

与 rolling GBNN 直接相关的配置块：

```yaml
method:
  name: rolling_gbnn

gbnn:
  enabled: true
  iterations_per_step: 1
  external_excitation: 1.0
  obstacle_inhibition: -2.0
  covered_input: 0.0
  neighbor_weight: 0.2
  transfer_beta: 0.6
  activity_min: -1.0
  activity_max: 1.0

rolling_optimizer:
  enabled: true
  use_global_strip_plan: true
  use_priority_strip: true
  horizon: 3
  beam_width: 30
  w_new_coverage: 8.0
  w_activity: 1.0
  w_direction: 4.0
  w_turn: 3.0
  w_repeat: 2.0
  w_dead_zone: 5.0
  w_obstacle: 1.5
  w_structure: 3.0
  w_branch_urgency: 12.0
  w_missed_branch: 45.0
  w_immediate_backtrack: 1000.0

escape:
  enabled: true
  method: hybrid
  backtracking_enabled: true
  dijkstra_enabled: true
```

需要注意：代码中还使用了若干未显式写在默认配置里的评分权重，例如 `w_strip_forward`、`w_strip_transition`、`w_strip_reverse`、`w_strip_cross`、`w_strip_loop`。这些在 `RollingOptimizer.score_branch()` 中有默认值。

## 3. 主循环

主循环在 `CoveragePlanner.run()` 和 `CoveragePlanner.run_events()` 中。二者逻辑基本一致，区别是 `run_events()` 会为 Web 前端逐步返回事件。

每一步流程：

1. 初始化时将起点标记为已覆盖。
2. 初始化 GBNN 活性场。
3. 记录第 0 步路径。
4. 进入循环，直到覆盖率达到目标或超过最大步数。
5. 每步先调用 `self.gbnn.update(self.cell_map)` 更新活性场。
6. 调用 `self.strategy.choose_next(...)` 选择下一元胞。
7. 如果 `decision.next_cell is None`，规划停止并记录失败原因。
8. 调用 `usv.move_to(...)` 更新 USV 状态。
9. 调用 `cell_map.mark_covered(...)` 更新覆盖状态和访问次数。
10. 记录决策、路径、覆盖率、逃逸记录。
11. 结束后计算指标并输出 CSV、PNG、GIF、HTML。

路径输出：

- `outputs/<scenario>/data/path.csv`：每一步位置、模式、逃逸类型、覆盖率。
- `outputs/<scenario>/data/decisions.csv`：每步选点、候选分支、候选树和评分分项。
- `outputs/<scenario>/data/escapes.csv`：逃逸发生点、目标点、逃逸类型和长度。

## 4. GBNN 活性场

实现位置：`src/core/gbnn_field.py`。

当前 GBNN 场是单层二维数组 `activity`，尺寸与栅格地图相同。

外部输入：

- 未覆盖元胞：`external_excitation`，默认 `1.0`。
- 已覆盖元胞：`covered_input`，默认 `0.0`。
- 障碍元胞：`obstacle_inhibition`，默认 `-2.0`。

每次更新：

1. 取当前正活性部分 `positive_activity`。
2. 对 8 邻域做扩散累加。
3. 计算 `raw = external + neighbor_weight * spread / 8.0`。
4. 如果启用模板匹配，则额外叠加局部结构分数。
5. 通过分段传递函数 `_transfer()` 限制活性范围。
6. 障碍元胞强制设为 `activity_min`。

传递函数逻辑：

- `raw < 0`：设为 `activity_min`。
- `0 <= raw < 1`：设为 `beta * raw`。
- `raw >= 1`：设为 `activity_max`。

当前 GBNN 的作用主要是提供候选分支评分中的 `activity_score`。它不是直接决定下一步的唯一依据。

## 5. 策略入口

实现位置：`src/core/strategy.py`。

`create_strategy()` 根据配置中的 `method.name` 选择策略：

- `rolling_gbnn`：当前主方法。
- `gbnn_greedy`：单步贪心基线。
- `original_binn`：只按 GBNN 活性选邻域点。
- `improved_binn`：加入未覆盖、航向、未来邻域和重复访问项。

`RollingGBNNStrategy.choose_next()` 当前优先级如下：

1. 如果启用 `use_global_strip_plan`，先执行 `_next_strip_plan_step()`。
2. 如果有正在执行的逃逸路径，则继续 `_continue_escape()`。
3. 调用 `RollingOptimizer.select_next_cell()` 做滚动候选选择。
4. 如果当前位置是死区，或滚动优化无法给出有效下一步，则尝试 `_start_escape()`。
5. 如果逃逸不允许，则尝试 `_local_strip_fallback()`。
6. 返回 `StepDecision`。

关键点：当前代码中，全局条带计划的优先级高于滚动候选树。因此很多正常路径其实首先来自 `global_strip_plan`，不是来自候选树评分。

## 6. 全局条带计划

实现位置：`RollingGBNNStrategy._build_strip_plan()`。

它按行生成类似 boustrophedon 的扫描路线：

1. 从 `y = 0` 到 `height - 1` 逐行扫描。
2. 每行用 `_row_segments()` 找出障碍切分后的可通行连续区间。
3. 奇数行反向扫描。
4. 如果当前点不在下一个 segment 内，用 A* 从当前位置连接到 segment 起点。
5. 把 segment 内元胞按顺序加入计划。

执行时 `_next_strip_plan_step()` 会跳过已经访问过的计划点。如果下一个未访问计划点不在当前 8 邻域，则用 A* 连接，并只执行 A* 路径的第一步。

潜在影响：

- 开阔水域会形成规则横向扫描线。
- 障碍将行切成多个 segment 后，A* 连接线可能导致局部折线或绕障。
- 因为该计划优先级最高，`RollingOptimizer` 的候选树在很多正常阶段不会真正主导决策。

## 7. 滚动候选树

实现位置：`src/core/rolling_optimizer.py`。

`RollingOptimizer.select_next_cell()` 是滚动优化的核心入口。

候选选择优先级：

1. 如果 `rolling_optimizer.enabled` 为 false，则退化为找一个未覆盖邻居。
2. 如果启用 `use_priority_strip`，先尝试 `_priority_strip_step()`。
3. 若优先规则没有返回结果，则调用 `build_candidate_tree()`。
4. 如果候选树为空，则尝试直接条带步或局部 fallback。
5. 对候选分支取 `branch_score` 最大者。
6. 如果最佳分支没有新增覆盖且当前条带前方也没有未覆盖点，则返回失败原因，交给逃逸逻辑。
7. 最终只返回最佳分支的第一步。

候选树生成：

- 深度由 `horizon` 控制，默认 3。
- 每层从上一层 beam 中继续扩展。
- 每个节点从当前位置或分支末端取 `cell_map.neighbors8(root)`。
- 当前 `_allowed_next()` 直接返回 `True`，因此没有实际过滤立即回退、重复访问或斜向移动。
- 每个新分支都调用 `score_branch()` 评分。
- 每层按分数排序，并保留 `beam_width` 个分支继续扩展。

当前实现里，候选树是 beam search，不是完全穷举。文件中存在 `_exhaustive_candidate_tree()`，但正常流程未调用。

## 8. 分支评分

实现位置：`RollingOptimizer.score_branch()`。

评分分为奖励项和惩罚项。

奖励项：

- `new_coverage_score`：分支中新增未覆盖元胞数量。
- `activity_score`：分支上 GBNN 正活性均值。
- `direction_score`：航向连续性。
- `structure_score`：局部未覆盖邻居数量。
- `branch_urgency_score`：当前邻域存在多个未覆盖分支时，对低度数/障碍压力分支加权。
- `missed_branch_score`：跨到相邻条带补可能遗漏的分支。
- `strip_forward_score`：沿当前条带方向前进。
- `strip_transition_score`：当前条带无前方覆盖机会时，转入相邻条带。

惩罚项：

- `turn_penalty`：航向变化较大。
- `repeat_penalty`：重复访问或虚拟分支内重复。
- `dead_zone_penalty`：局部未覆盖邻居少，死区风险高。
- `obstacle_penalty`：靠近障碍或边界。
- `loop_penalty`：无意义往复。
- `strip_reverse_penalty`：沿条带反方向走。
- `strip_cross_penalty`：跨多条带或当前条带仍有前方未覆盖时提前跨条带。
- `strip_loop_penalty`：分支中重复访问。
- `immediate_backtrack_penalty`：立即回到上一步或分支内二步回退。

最终：

```text
score =
  覆盖收益
+ GBNN 活性
+ 航向连续
+ 结构收益
+ 分支紧迫性
+ 遗漏分支补偿
+ 条带前进/转移收益
- 转向/重复/死区/障碍/回退/跨条带等惩罚
```

需要注意：有些行为没有在 `_allowed_next()` 中硬过滤，而是通过大惩罚项压低分数。例如立即回退默认惩罚权重为 `1000.0`。

## 9. 条带优先规则

`RollingOptimizer._priority_strip_step()` 会先尝试：

1. `_missed_branch_step()`：如果相邻条带存在可能遗漏的未覆盖分支，优先补侧向分支。
2. `_direct_strip_step()`：沿当前条带方向向前走。

`_direct_strip_step()` 逻辑：

- 若当前行前方仍有未覆盖元胞，并且前方一格可通行，则直接返回前方一格。
- 若当前条带前方没有未覆盖，则尝试垂直进入相邻条带的同 x 位置。

这意味着候选树前面还有一层强规则，会让路径更像条带扫描。

## 10. 逃逸机制

实现位置：`src/core/escape_selector.py`。

触发场景：

- 当前元胞是死区。
- rolling optimizer 无法返回有效下一步。
- 覆盖长时间停滞，且当前或相邻条带没有继续覆盖机会。

逃逸候选：

1. `backtracking`：沿历史路径向后找一个仍有未覆盖邻居的历史元胞。
2. `dijkstra`：从当前位置用 Dijkstra 搜索最近/低代价的未覆盖元胞。

混合模式下：

- 通常先找 backtracking，再找 dijkstra。
- 若原因是 `dead_zone`、`no_strip_new_coverage_candidate`、`no_traversable_neighbor`，则偏向直接找未覆盖目标。
- 最终选择 `evaluate_escape_candidate()` 分数最低的候选。

Dijkstra 代价：

```text
move_cost
+ 0.8 * repeat
+ 0.3 * obstacle_proximity
+ 0.4 * narrow
```

其中 `move_cost()` 对斜向移动的代价是 `sqrt(2)`，并且 `cell_map.neighbors8()` 允许 8 邻域运动。因此当前 Dijkstra 逃逸会自然生成较长的斜向跨图路径。

逃逸候选评分：

- 距离越长，分数越高。
- 重复访问越多，分数越高。
- 转向越多，分数越高。
- 目标连通块越大，分数越低。
- 未来覆盖潜力越大，分数越低。

## 11. 地图和运动模型

实现位置：`src/core/cell_map.py`。

当前所有邻域扩展都基于 `neighbors8()`：

- 正常候选树使用 8 邻域。
- GBNN 活性扩散使用 8 邻域。
- A* / Dijkstra 使用 8 邻域。
- 死区判断使用 8 邻域是否存在未覆盖邻居。

这会带来两个明显效果：

1. 算法允许斜向移动。
2. 绘图中连续斜向移动会表现为跨越很远的斜线，尤其是 Dijkstra 逃逸路径。

如果希望路径更像实际覆盖航迹，可能需要将运动邻域、逃逸邻域或绘图解释区分开，例如正常覆盖只允许 4 邻域，逃逸允许 8 邻域，或者对斜向移动施加强惩罚。

## 12. USV 状态

实现位置：`src/core/usv.py`。

`USV` 记录：

- `current_cell`：当前位置。
- `heading`：当前航向，8 个方向编码。
- `current_strip_id`：当前条带，默认等于 y 坐标。
- `strip_direction`：条带扫描方向，初始为 1。
- `strip_progress`：条带前进进度。
- `recent_strip_history`：最近条带历史。
- `path`：完整路径。
- `mode_history`：每步是 normal 还是 escape。
- `escape_type_history`：每步逃逸类型。

`move_to()` 会根据移动模式更新航向、条带方向、路径和历史记录。

当 normal 模式跨到新条带时，如果 `advance_strip` 为真，会反转 `strip_direction`，形成往复式扫描。

## 13. 输出与可视化

实现位置：`src/visualization/plotter.py` 和 `web/index.html`。

轨迹图：

- 蓝色实线：normal。
- 绿色虚线：backtracking escape。
- 橙色虚线：dijkstra escape。
- 绿色圆点：起点。
- 红色叉号：终点。

`plot_trajectory()` 是逐段画 `usv.path[i - 1] -> usv.path[i]`，没有对同色非连续点做错误连接。因此如果图中出现长斜线，通常说明路径中确实存在连续斜向移动，而不是绘图乱序。

`planning_viewer.html` 和 Web 页面还会显示：

- 执行路径。
- 当前选中分支。
- 候选分支。
- 候选树。

## 14. 当前代码与文字算法描述的偏差

README / `text/03.md` 中的文字流程更像“候选树筛选 + 评分 + 死区逃逸”的理想算法。但当前代码存在以下偏差：

1. `use_global_strip_plan` 默认开启，且优先级高于滚动候选树。
2. `_allowed_next()` 未实际筛除立即回退、重复访问、斜向移动、过多转向等，只是通过评分惩罚。
3. 候选树使用 beam search，不是完整枚举所有深度为 H 的路径。
4. 条带优先规则 `_priority_strip_step()` 会在候选树之前返回直接步骤。
5. Dijkstra 逃逸使用 8 邻域，并会为了补剩余未覆盖块产生长斜向路径。
6. `record_candidate_count` 和 `record_tree_count` 默认极大，输出的 `decisions.csv` 可能非常重。

这些偏差不一定都是错误，但如果论文方法希望强调 rolling GBNN 分支决策，就需要决定是否弱化或关闭全局条带计划。

## 15. 最近输出暴露的问题

从当前 `outputs` 检查结果看：

- `path.csv` 没有乱序跳点，相邻路径点距离最大为 1。
- `open_water` 基本是规则条带扫描。
- `single_obstacle` 主要仍是条带扫描，绕障局部有转折。
- `concave_area` 和 `island_obstacles` 中出现较明显的橙色斜线，主要来自 Dijkstra escape。
- `island_obstacles` 中有多次较长 Dijkstra 逃逸，例如跨越几十个元胞去补剩余未覆盖块。

因此“路径乱”的主要来源不是 CSV 写错或绘图乱连，而是当前逃逸策略在 8 邻域下允许长距离斜向补洞。

## 16. 后续可修改方向

下面列出一些可能的修改入口，供后续批注。

### 16.1 是否保留全局条带计划

位置：`RollingGBNNStrategy.choose_next()` 和 `_next_strip_plan_step()`。

可选方向：

- 关闭 `use_global_strip_plan`，让 rolling optimizer 真正主导正常阶段。
- 保留全局条带计划，但只作为低优先级 fallback。
- 将全局条带计划改为只生成参考方向，不直接返回下一步。

### 16.2 候选树合法性筛选

位置：`RollingOptimizer._allowed_next()`。

当前直接返回 `True`。可加入：

- 禁止立即回退。
- 限制分支内重复访问。
- 限制斜向移动。
- 限制连续大角度转向。
- 禁止穿越障碍角点。

### 16.3 分支评分权重

位置：`RollingOptimizer.score_branch()`。

可重点调整：

- 提高 `w_new_coverage`。
- 提高 `w_repeat`。
- 提高 `w_turn`。
- 提高 `w_strip_cross`。
- 降低对全局条带的强约束。
- 增加对长斜向移动的惩罚。

### 16.4 逃逸目标选择

位置：`EscapeSelector.find_dijkstra_candidate()` 和 `evaluate_escape_candidate()`。

可选方向：

- Dijkstra 只允许 4 邻域。
- 对斜向移动加更高代价。
- 限制逃逸最大距离。
- 优先选择当前条带或相邻条带的 frontier。
- 目标不再是任意未覆盖元胞，而是局部连通块入口。
- 增大长距离逃逸惩罚，降低大连通块吸引力。

### 16.5 绘图表达

位置：`plot_trajectory()`、`planning_viewer.html`。

可选方向：

- 将 normal path 和 escape path 分图展示。
- 对逃逸路径降低透明度。
- 为每次 escape 标注编号和起止点。
- 只显示最终覆盖顺序，不叠加候选树。
- 输出一张按时间渐变的路径图，减少静态叠线误读。

## 17. 修改约定

后续你可以直接在本文档中：

- 删除你不认可的描述。
- 在任意小节下写“修改为：...”。
- 用“保留 / 删除 / 重构 / 待确认”标记代码段。
- 在第 16 节列出具体希望实现的算法变化。

我会优先按本文档中的修改说明调整代码。
