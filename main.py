import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import math
from dataclasses import dataclass

# 定义点结构
@dataclass
class Point:
    position: np.ndarray
    covered: int = 0
    x1: int = 0
    x2: float = 0.0
    x3: float = 0.0

# 机器人类
class Robot3:
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.L = 0
        self.X0 = None
        self.Y0 = None
        
    @property
    def s(self):
        """计算到所有点的距离"""
        s = np.zeros((self.L, self.L))
        for i in range(self.L):
            for j in range(self.L):
                s[i, j] = math.hypot(self.X0[i, j] - self.x, self.Y0[i, j] - self.y)
        return s

# 判断点是否在地图内
def in_map(point, map_size):
    return 1 <= point[0] <= map_size and 1 <= point[1] <= map_size

# 移动规则函数
def move_rule(max_index, around_points, current_index, map_size):
    if (around_points[6].x1 != 100 and around_points[7].x1 != 100 and 
        around_points[3].x1 == 100):
        return 4
    elif (around_points[0].x1 != 100 and around_points[1].x1 != 100 and 
          around_points[3].x1 == 100):
        return 4
        
    # 处理边界情况
    if (around_points[0].x1 != 100 and around_points[3].x1 != 100 and 
        around_points[5].x1 != 100 and around_points[1].x1 == 100 and 
        around_points[6].x1 == 100):
        if current_index[0] == map_size or current_index[0] == 0:
            if current_index[1] >= map_size / 2:
                return 2
            else:
                return 7
    return max_index

# 生成地图和障碍物
def create_map(map_size, obs_num, model=1):
    obstacles = np.zeros((obs_num, 4))
    fig, ax = plt.subplots()
    ax.set_xlim(0, map_size)
    ax.set_ylim(0, map_size)
    ax.set_aspect('equal')
    
    if model == 1:
        # 固定障碍物
        rects = [
            [0, 5, 2, 2],
            [7, 2, 2, 2],
            [4, 3, 2, 2],
            [7, 7, 1, 1]
        ]
        for i, (x, y, w, h) in enumerate(rects[:obs_num]):
            ax.add_patch(plt.Rectangle((x, y), w, h, facecolor='gray'))
            obstacles[i] = [x, y, x+w, y+h]
    
    # 绘制网格
    for i in range(map_size + 1):
        ax.plot([0, map_size], [i, i], 'k-', linewidth=0.5)
        ax.plot([i, i], [0, map_size], 'k-', linewidth=0.5)
    
    return obstacles, fig, ax

# 判断是否为障碍物
def is_obstacle(pos, obstacles):
    x, y = pos
    for obs in obstacles:
        if obs[0] <= x < obs[2] and obs[1] <= y < obs[3]:
            return True
    return False

# 路径规划主函数
def binn(map_size, obstacles):
    # 初始化点网格
    points = [[Point(np.array([i, j])) for j in range(map_size + 1)] 
             for i in range(map_size + 1)]
    current_index = np.array([1, 1])
    path = [points[current_index[0]][current_index[1]]]
    count = 0
    
    while True:
        around_points = []
        arrund_flag = [0] * 8
        directions = [
            [-1, 1], [0, 1], [1, 1],
            [-1, 0],          [1, 0],
            [-1, -1], [0, -1], [1, -1]
        ]
        
        # 计算周围点
        for i, (dx, dy) in enumerate(directions):
            grid = current_index + np.array([dx, dy])
            if in_map(grid, map_size):
                around_points.append(points[grid[0]][grid[1]])
                arrund_flag[i] = 1
            else:
                arrund_flag[i] = 0
        
        # 计算点属性
        max_sum = -np.inf
        max_index = 0
        for i, p in enumerate(around_points):
            if not is_obstacle(p.position, obstacles):
                if p.covered == 0:
                    p.x1 = 100
                else:
                    p.x1 = 0
            else:
                p.x1 = -100
                
            # 计算x2 (距离惩罚)
            dx = p.position[0] - current_index[0]
            dy = p.position[1] - current_index[1]
            p.x2 = -2 * math.hypot(dx, dy)
            p.x3 = 0
            
            # 寻找最大权重
            current_sum = p.x1 + p.x2 + p.x3
            if current_sum > max_sum:
                max_sum = current_sum
                max_index = i
        
        # 应用移动规则
        if len(around_points) > 0:
            max_index = move_rule(max_index, around_points, current_index, map_size)
            current_index = around_points[max_index].position
            points[current_index[0]][current_index[1]].covered = 1
            points[current_index[0]][current_index[1]].x1 = 100
            path.append(points[current_index[0]][current_index[1]])
        
        # 检查终止条件
        count += 1
        if count == 2000:
            print("未能完全覆盖地图")
            break
            
        # 检查是否全部覆盖
        all_covered = True
        for i in range(1, map_size + 1):
            for j in range(1, map_size + 1):
                if not is_obstacle(np.array([i, j]), obstacles) and points[i][j].covered == 0:
                    all_covered = False
                    break
            if not all_covered:
                break
        if all_covered:
            print("完全覆盖路径规划成功!")
            break
            
    return points, path

# A*算法函数
def a_star_function(point1, point2, target_point):
    g = math.hypot(
        point1.position[0] - point2.position[0],
        point1.position[1] - point2.position[1]
    )
    h = math.hypot(
        point2.position[0] - target_point.position[0],
        point2.position[1] - target_point.position[1]
    )
    return g + h

# A*路径搜索
def a_star_search(current_point, points, obstacles, map_size):
    min_dis = map_size ** 2
    current_point.covered = 1
    target_point = None
    path = [current_point]
    count = 0
    
    # 寻找目标点（最近的未覆盖点）
    for i in range(1, map_size + 1):
        for j in range(1, map_size + 1):
            p = points[i][j]
            if not is_obstacle(p.position, obstacles) and p.covered == 0:
                dis = math.hypot(
                    current_point.position[0] - p.position[0],
                    current_point.position[1] - p.position[1]
                )
                if dis < min_dis:
                    min_dis = dis
                    target_point = p
    
    if target_point is None:
        return path, current_point
    
    last_position = np.array([0, 0])
    while True:
        around_points = []
        directions = [
            [-1, 1], [0, 1], [1, 1],
            [-1, 0],          [1, 0],
            [-1, -1], [0, -1], [1, -1]
        ]
        
        # 生成周围点
        for dx, dy in directions:
            pos = current_point.position + np.array([dx, dy])
            if in_map(pos, map_size) and not is_obstacle(pos, obstacles):
                around_points.append(points[pos[0]][pos[1]])
        
        if not around_points:
            print("周围没有可达点")
            break
        
        # 移除回退点
        for i in range(len(around_points)):
            if np.array_equal(around_points[i].position, last_position):
                around_points.pop(i)
                break
        
        # 选择最佳下一个点
        f_min = a_star_function(current_point, around_points[0], target_point)
        min_index = 0
        for i, p in enumerate(around_points):
            f = a_star_function(current_point, p, target_point)
            if f < f_min:
                f_min = f
                min_index = i
        
        last_position = current_point.position
        current_point = around_points[min_index]
        current_point.covered = 1
        path.append(current_point)
        
        count += 1
        if count == 3000:
            print("超时，未能到达目标点")
            break
            
        if np.array_equal(current_point.position, target_point.position):
            print("到达目标点")
            break
            
    return path, current_point

# 主实验函数
def experiment():
    map_size = 10
    M = 3  # 机器人数量
    d = 10  # 环境尺寸
    
    # 创建地图和障碍物
    obstacles, fig, ax = create_map(map_size, 4)
    
    # 初始化机器人
    robots = [Robot3() for _ in range(M)]
    colors = ['r', 'g', 'b']
    lines = [ax.plot([], [], color=colors[i], lw=2)[0] for i in range(M)]
    points = [ax.plot([], [], 'o', color=colors[i])[0] for i in range(M)]
    
    # 设置机器人初始位置
    for i in range(M):
        robots[i].x = 0.5 * (i + 1)
        robots[i].y = 0.5
        x0 = np.arange(0, d, 0.0625)
        y0 = np.arange(0, d, 0.0625)
        robots[i].X0, robots[i].Y0 = np.meshgrid(x0, y0)
        robots[i].L = len(x0)
    
    # 执行路径规划
    points_grid, path = binn(map_size, obstacles)
    
    # 动画更新函数
    def update(frame):
        for i in range(M):
            # 简单的路径跟踪
            if frame < len(path):
                pos = path[frame].position
                robots[i].x = pos[0]
                robots[i].y = pos[1]
            
            # 更新轨迹
            xdata = lines[i].get_xdata()
            ydata = lines[i].get_ydata()
            lines[i].set_xdata(np.append(xdata, robots[i].x))
            lines[i].set_ydata(np.append(ydata, robots[i].y))
            points[i].set_data(robots[i].x, robots[i].y)
        return lines + points
    
    # 创建动画
    ani = FuncAnimation(fig, update, frames=len(path), interval=100, blit=True)
    plt.show()

if __name__ == "__main__":
    experiment()