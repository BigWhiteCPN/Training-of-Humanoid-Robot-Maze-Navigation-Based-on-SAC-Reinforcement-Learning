import os
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
import mujoco.viewer
from scipy.spatial.transform import Rotation as R
import math
import torch
import torch.nn as nn
from skimage.draw import line as skimage_line
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates, binary_dilation, binary_closing, distance_transform_edt
from skimage.graph import route_through_array
from collections import deque
from scipy.interpolate import splprep, splev

# 辅助函数

def pd_control(target_q, q, kp, target_dq, dq, kd):
    """简单的PD控制器函数"""
    return (target_q - q) * kp + (target_dq - dq) * kd

# 1. 迷宫生成器 (保持 20米/8x8 配置)
class MazeGridGenerator:
    def __init__(self, world_size=20.0, grid_dim=8, remove_wall_prob=0.25):
        self.world_size = world_size
        self.grid_dim = grid_dim
        self.cell_size = world_size / grid_dim
        self.remove_wall_prob = remove_wall_prob
        self.wall_thickness = 0.1 
        self.wall_length = self.cell_size 

    def generate(self, np_random):
        visited = np.zeros((self.grid_dim, self.grid_dim), dtype=bool)
        stack =[(0, 0)]
        visited[0, 0] = True
        passages = set()
        
        while stack:
            r, c = stack[-1]
            neighbors = []
            for dr, dc in[(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.grid_dim and 0 <= nc < self.grid_dim and not visited[nr, nc]:
                    neighbors.append((nr, nc))
            
            if neighbors:
                idx = np_random.integers(0, len(neighbors))
                nr, nc = neighbors[idx]
                p1, p2 = (r, c), (nr, nc)
                if p1 > p2: p1, p2 = p2, p1
                passages.add((p1, p2))
                visited[nr, nc] = True
                stack.append((nr, nc))
            else:
                stack.pop()
        
        walls =[]
        for r in range(self.grid_dim):
            for c in range(self.grid_dim - 1):
                p1, p2 = (r, c), (r, c+1)
                if (p1, p2) not in passages:
                    if np_random.random() > self.remove_wall_prob:
                        center_x = (c + 1) * self.cell_size - self.world_size/2.0
                        center_y = (self.world_size/2.0) - (r + 0.5) * self.cell_size 
                        walls.append({
                            'pos': np.array([center_x, center_y, 1.0]),
                            'is_vertical': True,
                            'size': np.array([self.wall_thickness, self.cell_size/2.0 + 0.05, 1.0])
                        })

        for r in range(self.grid_dim - 1):
            for c in range(self.grid_dim):
                p1, p2 = (r, c), (r+1, c)
                if (p1, p2) not in passages:
                    if np_random.random() > self.remove_wall_prob:
                        center_x = (c + 0.5) * self.cell_size - self.world_size/2.0
                        center_y = (self.world_size/2.0) - (r + 1) * self.cell_size
                        walls.append({
                            'pos': np.array([center_x, center_y, 1.0]),
                            'is_vertical': False,
                            'size': np.array([self.cell_size/2.0 + 0.05, self.wall_thickness, 1.0])
                        })
                        
        return walls


class PathPlanner:
    def __init__(self, grid_map_obj, obstacle_threshold=0.65, robot_radius=0.5):
        self.grid_map = grid_map_obj
        self.obstacle_threshold = obstacle_threshold
        self.robot_radius = robot_radius

    def _world_to_grid(self, world_pos):
        grid_coords = (world_pos[:2] + self.grid_map.world_origin_offset_m) * self.grid_map.resolution
        r = self.grid_map.num_cells_world - 1 - int(grid_coords[1])
        c = int(grid_coords[0])
        return np.array([
            np.clip(r, 0, self.grid_map.num_cells_world - 1), 
            np.clip(c, 0, self.grid_map.num_cells_world - 1)
        ])

    def _grid_to_world(self, grid_pos):
        y_grid, x_grid = self.grid_map.num_cells_world - 1 - grid_pos[0], grid_pos[1]
        world_coords = np.array([x_grid, y_grid]) / self.grid_map.resolution - self.grid_map.world_origin_offset_m
        return world_coords

    def _compute_gradient_cost_map(self, prob_grid):
        obstacles = prob_grid > self.obstacle_threshold
        structure = np.ones((3, 3), dtype=bool)
        closed_obstacles = binary_closing(obstacles, structure=structure, iterations=2)
        dist_grid = distance_transform_edt(np.logical_not(closed_obstacles))
        dist_meters = dist_grid / self.grid_map.resolution
        
        cost_map = np.ones_like(prob_grid, dtype=np.float32)
        
        # Zone 1: 致命硬墙
        lethal_radius = self.robot_radius * 1.2 
        cost_map[dist_meters < lethal_radius] = np.inf
        
        # Zone 2: 膨胀势场
        inflation_zone_mask = (dist_meters >= lethal_radius) & (dist_meters < (lethal_radius + 0.8))
        decay_factor = 2.5
        dist_diff = dist_meters[inflation_zone_mask] - lethal_radius
        potential_cost = 50.0 * np.exp(-decay_factor * dist_diff)
        cost_map[inflation_zone_mask] += potential_cost
        
        # Zone 3: 未知区域
        unknown_mask = (prob_grid > 0.45) & (prob_grid <= 0.55)
        valid_unknown = unknown_mask & (cost_map != np.inf)
        cost_map[valid_unknown] += 20.0 
        
        return cost_map

    def _prune_path(self, path, start_pos):
        if len(path) < 3: return path
        dists = np.linalg.norm(path - start_pos, axis=1)
        closest_idx = np.argmin(dists)
        pruned_path = path[closest_idx:]
        if len(pruned_path) < 2: return path
        if len(pruned_path) > 2:
            dist0 = np.linalg.norm(pruned_path[0] - start_pos)
            dist1 = np.linalg.norm(pruned_path[1] - start_pos)
            if dist1 < dist0:
                pruned_path = pruned_path[1:]
        return pruned_path

    def _smooth_path(self, path, s=0.1): 
        if len(path) < 3: return path
        try:
            keep_indices = [0]
            for i in range(1, len(path)):
                if np.linalg.norm(path[i] - path[keep_indices[-1]]) > 0.2:
                    keep_indices.append(i)
            if keep_indices[-1] != len(path) - 1:
                keep_indices.append(len(path) - 1)
            path = path[keep_indices]
            if len(path) < 3: return path

            x, y = path[:, 0], path[:, 1]
            k_order = min(3, len(path) - 1)
            tck, u = splprep([x, y], s=s, k=k_order) 
            num_points = max(10, int(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)) / 0.1))
            u_new = np.linspace(u.min(), u.max(), num_points)
            x_new, y_new = splev(u_new, tck)
            return np.c_[x_new, y_new]
        except Exception:
            return path

    def _find_nearest_valid_point(self, grid_pos, cost_grid, search_radius=30):
        if cost_grid[grid_pos[0], grid_pos[1]] != np.inf:
            return grid_pos
        rows, cols = cost_grid.shape
        for r in range(1, search_radius + 1):
            r_min, r_max = max(0, grid_pos[0]-r), min(rows-1, grid_pos[0]+r)
            c_min, c_max = max(0, grid_pos[1]-r), min(cols-1, grid_pos[1]+r)
            local_area = cost_grid[r_min:r_max+1, c_min:c_max+1]
            valid_indices = np.argwhere(local_area != np.inf)
            if len(valid_indices) > 0:
                candidates = valid_indices + np.array([r_min, c_min])
                dists = np.sum((candidates - grid_pos)**2, axis=1)
                best_idx = np.argmin(dists)
                return candidates[best_idx]
        return None

    def find_path(self, start_world, goal_world):
        raw_prob_grid = self.grid_map._log_odds_to_prob(self.grid_map.grid)
        # 注意：PathPlanner 内部计算依然需要原始方向，不翻转
        cost_map = self._compute_gradient_cost_map(raw_prob_grid)
        
        start_grid = self._world_to_grid(start_world)
        goal_grid = self._world_to_grid(goal_world)

        safe_start = self._find_nearest_valid_point(start_grid, cost_map)
        safe_goal = self._find_nearest_valid_point(goal_grid, cost_map)
        
        if safe_start is None or safe_goal is None:
            return None

        try:
            path_indices, cost = route_through_array(
                cost_map, start=safe_start, end=safe_goal,
                fully_connected=True, geometric=True 
            )
            if not path_indices: return None

            path_world = np.array([self._grid_to_world(pos) for pos in path_indices])
            
            filtered_path = [path_world[0]]
            for i in range(1, len(path_world)-1):
                if np.linalg.norm(path_world[i] - filtered_path[-1]) > 0.15: 
                    filtered_path.append(path_world[i])
            filtered_path.append(path_world[-1])
            filtered_path = np.array(filtered_path)
            
            if cost_map[goal_grid[0], goal_grid[1]] != np.inf:
                filtered_path[-1] = goal_world
            
            pruned_path = self._prune_path(filtered_path, start_world)
            smoothed_path = self._smooth_path(pruned_path, s=0.1)
            
            return smoothed_path
        except Exception:
            return None

# DynamicObstacleController
class DynamicObstacleController:
    def __init__(self, static_walls: list, world_size_m: float, resolution: int, 
                 obstacle_radius: float, speed: float = 0.4):
        self.speed = speed
        self.world_size_m = world_size_m
        self.resolution = resolution
        self.num_cells = int(world_size_m * resolution)
        self.world_origin_offset_m = np.array([world_size_m / 2.0, world_size_m / 2.0])
        self.obstacle_radius = obstacle_radius

        self.cost_grid = self._create_cost_grid_from_walls(static_walls)
        self.free_indices = np.argwhere(np.isfinite(self.cost_grid))
        if len(self.free_indices) == 0:
            raise RuntimeError("地图中没有空闲区域，无法生成动态障碍物！")

        self.current_path = None
        self.current_path_index = 0
        self.current_pos = np.array([0.0, 0.0]) 
        self.is_waiting = False
        self.wait_timer = 0.0
        self.wait_duration = 0.0

    def update_walls(self, static_walls):
        self.cost_grid = self._create_cost_grid_from_walls(static_walls)
        self.free_indices = np.argwhere(np.isfinite(self.cost_grid))
        if len(self.free_indices) == 0:
            print("[警告] 地图中没有空闲区域，动态障碍物可能无法工作！")

    def _world_to_grid(self, world_pos):
        grid_coords = (world_pos[:2] + self.world_origin_offset_m) * self.resolution
        r = self.num_cells - 1 - int(grid_coords[1])
        c = int(grid_coords[0])
        return np.array([
            np.clip(r, 0, self.num_cells - 1),
            np.clip(c, 0, self.num_cells - 1)
        ])

    def _grid_to_world(self, grid_pos):
        y_grid, x_grid = self.num_cells - 1 - grid_pos[0], grid_pos[1]
        world_coords = np.array([x_grid, y_grid]) / self.resolution - self.world_origin_offset_m
        return world_coords

    def _create_cost_grid_from_walls(self, walls):
        wall_map = np.zeros((self.num_cells, self.num_cells), dtype=bool)
        for wall in walls:
            pos, size = wall['pos'], wall['size']
            x_min, x_max = pos[0] - size[0], pos[0] + size[0]
            y_min, y_max = pos[1] - size[1], pos[1] + size[1]
            
            grid_x_min = int((x_min + self.world_origin_offset_m[0]) * self.resolution)
            grid_x_max = int((x_max + self.world_origin_offset_m[0]) * self.resolution)
            grid_y_min_flipped = self.num_cells - 1 - int((y_max + self.world_origin_offset_m[1]) * self.resolution)
            grid_y_max_flipped = self.num_cells - 1 - int((y_min + self.world_origin_offset_m[1]) * self.resolution)
            
            grid_x_min, grid_x_max = np.clip([grid_x_min, grid_x_max], 0, self.num_cells-1)
            grid_y_min_flipped, grid_y_max_flipped = np.clip([grid_y_min_flipped, grid_y_max_flipped], 0, self.num_cells-1)
            wall_map[grid_y_min_flipped:grid_y_max_flipped+1, grid_x_min:grid_x_max+1] = True

        inflation_radius_grid = math.ceil(self.obstacle_radius * self.resolution) + 1
        structure = np.ones((3, 3), dtype=bool)
        inflated_wall_map = binary_dilation(wall_map, structure=structure, iterations=inflation_radius_grid)
        cost_grid = np.ones_like(wall_map, dtype=np.float32)
        cost_grid[inflated_wall_map] = np.inf
        return cost_grid

    def _smooth_path(self, path, num_points=None, s=0.5):
        if len(path) < 4: return path
        try:
            if num_points is None:
                dist = np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1))
                num_points = max(5, int(dist * 10))
            tck, u = splprep([path[:, 0], path[:, 1]], s=s, k=3)
            u_new = np.linspace(u.min(), u.max(), num_points)
            x_new, y_new = splev(u_new, tck)
            return np.c_[x_new, y_new]
        except Exception: return path

    def _find_valid_start_grid(self, current_grid):
        if self.cost_grid[current_grid[0], current_grid[1]] != np.inf:
            return current_grid
        max_search_radius = 6
        for r in range(1, max_search_radius + 1):
            r_min = max(0, current_grid[0] - r)
            r_max = min(self.num_cells - 1, current_grid[0] + r)
            c_min = max(0, current_grid[1] - r)
            c_max = min(self.num_cells - 1, current_grid[1] + r)
            sub_grid = self.cost_grid[r_min:r_max+1, c_min:c_max+1]
            valid_indices = np.argwhere(np.isfinite(sub_grid))
            if len(valid_indices) > 0:
                found = valid_indices[0] + np.array([r_min, c_min])
                return found
        return None 

    def _find_random_target_and_plan(self):
        start_grid_raw = self._world_to_grid(self.current_pos)
        start_grid = self._find_valid_start_grid(start_grid_raw)
        if start_grid is None:
            return False
        attempts = 0
        while attempts < 10:
            idx = np.random.randint(0, len(self.free_indices))
            end_grid = self.free_indices[idx]
            dist_grid = np.linalg.norm(start_grid - end_grid)
            if dist_grid < (2.0 * self.resolution): 
                attempts += 1
                continue
            try:
                path_indices, _ = route_through_array(
                    self.cost_grid, start=start_grid, end=end_grid,
                    fully_connected=True, geometric=True
                )
                if not path_indices or len(path_indices) < 2:
                    attempts += 1
                    continue
                path_world = np.array([self._grid_to_world(pos) for pos in path_indices])
                self.current_path = self._smooth_path(path_world)
                self.current_path_index = 0
                return True
            except Exception:
                attempts += 1
        return False

    def reset(self):
        idx = np.random.randint(0, len(self.free_indices))
        start_grid = self.free_indices[idx]
        self.current_pos = self._grid_to_world(start_grid)
        self.is_waiting = False
        self.current_path = None
        self._find_random_target_and_plan()

    def update(self, dt: float) -> np.ndarray:
        if self.is_waiting:
            self.wait_timer += dt
            if self.wait_timer >= self.wait_duration:
                self.is_waiting = False
                success = self._find_random_target_and_plan()
                if not success: return self.current_pos
            else:
                return self.current_pos

        if self.current_path is None or len(self.current_path) == 0:
             success = self._find_random_target_and_plan()
             if not success: return self.current_pos 

        distance_to_move = self.speed * dt
        while distance_to_move > 0:
            if self.current_path is None: break 
            next_index = self.current_path_index + 1
            if next_index >= len(self.current_path):
                self.is_waiting = True
                self.wait_timer = 0.0
                self.wait_duration = np.random.uniform(0.2, 0.4)
                self.current_path = None
                break
            target_pos = self.current_path[next_index]
            vector_to_target = target_pos - self.current_pos
            dist_to_target = np.linalg.norm(vector_to_target)
            if dist_to_target < 1e-6:
                self.current_path_index = next_index
                continue
            if distance_to_move >= dist_to_target:
                self.current_pos = target_pos.copy()
                distance_to_move -= dist_to_target
                self.current_path_index = next_index
            else:
                move_vec = (vector_to_target / dist_to_target) * distance_to_move
                self.current_pos += move_vec
                distance_to_move = 0
        return self.current_pos
    
    @property
    def full_trajectory(self):
        return self.current_path
    
    def get_current_position(self) -> np.ndarray:
        return self.current_pos

# GlobalGridMap (修复可视化方向)

class GlobalGridMap:
    def __init__(self, world_size_m=20.0, local_map_size_m=10.0, resolution=6):
        self.world_size_m = world_size_m
        self.resolution = resolution
        self.num_cells_world = int(self.world_size_m * self.resolution)
        self.world_origin_offset_m = np.array([self.world_size_m / 2.0, self.world_size_m / 2.0])
        self.grid = np.zeros((self.num_cells_world, self.num_cells_world), dtype=np.float32)

        self.prob_hit = 0.70
        self.prob_miss = 0.4
        self.log_odds_hit = np.log(self.prob_hit / (1 - self.prob_hit))
        self.log_odds_miss = np.log(self.prob_miss / (1 - self.prob_miss))
        
        self.log_odds_max = 5.0
        self.log_odds_min = -5.0
        self.decay_rate = 0.99 
        
        self.local_map_size_m = local_map_size_m
        self.num_cells_local = int(self.local_map_size_m * self.resolution)

        half_local = self.num_cells_local / 2.0
        local_x, local_y = np.meshgrid(np.arange(self.num_cells_local), np.arange(self.num_cells_local))
        self.local_coords_base = np.stack((local_x.flatten() - half_local, 
                                           local_y.flatten() - half_local), axis=1)

        self.fig, self.ax = None, None
        self.im = None

    def _log_odds_to_prob(self, log_odds_grid):
        return 1.0 - 1.0 / (1.0 + np.exp(log_odds_grid))

    def reset(self):
        self.grid.fill(0)

    def _world_to_grid_indices(self, pos):
        c = int((pos[0] + self.world_origin_offset_m[0]) * self.resolution)
        # 注意: 这里的 'r' 是矩阵行索引，0在最上方。world y+ 对应 row 0 (inverted)
        r = self.num_cells_world - 1 - int((pos[1] + self.world_origin_offset_m[1]) * self.resolution)
        return r, c

    def update_from_lidar(self, robot_pos, hit_points, valid_mask):
        r0, c0 = self._world_to_grid_indices(robot_pos)
        
        for i, point in enumerate(hit_points):
            r1, c1 = self._world_to_grid_indices(point)
            rr, cc = skimage_line(r0, c0, r1, c1)
            valid_idx = (rr >= 0) & (rr < self.num_cells_world) & (cc >= 0) & (cc < self.num_cells_world)
            rr, cc = rr[valid_idx], cc[valid_idx]
            if len(rr) == 0: continue
            
            if valid_mask[i] and len(rr) > 1:
                rr_free, cc_free = rr[:-1], cc[:-1]
            else:
                rr_free, cc_free = rr, cc
                
            if len(rr_free) > 0:
                self.grid[rr_free, cc_free] = self.grid[rr_free, cc_free] * self.decay_rate + self.log_odds_miss

        hit_points_valid = hit_points[valid_mask]
        if len(hit_points_valid) > 0:
            hits_indices = ((hit_points_valid + self.world_origin_offset_m) * self.resolution).astype(int)
            r_hits = self.num_cells_world - 1 - hits_indices[:, 1]
            c_hits = hits_indices[:, 0]
            valid_hits = (r_hits >= 0) & (r_hits < self.num_cells_world) & (c_hits >= 0) & (c_hits < self.num_cells_world)
            r_hits, c_hits = r_hits[valid_hits], c_hits[valid_hits]
            np.add.at(self.grid, (r_hits, c_hits), self.log_odds_hit)

        np.clip(self.grid, self.log_odds_min, self.log_odds_max, out=self.grid)

    def get_local_map(self, robot_pos_world, robot_yaw_world):
        c, s = np.cos(-robot_yaw_world), np.sin(-robot_yaw_world)
        rotation_matrix = np.array([[c, -s], [s, c]])
        world_aligned_local_coords = self.local_coords_base @ rotation_matrix.T
        
        grid_x_center = (robot_pos_world[0] + self.world_origin_offset_m[0]) * self.resolution
        grid_y_center_inverted = self.num_cells_world - 1 - (robot_pos_world[1] + self.world_origin_offset_m[1]) * self.resolution
        
        # local x (right) -> grid col (increase)
        # local y (up)    -> grid row (decrease, go up in matrix because row 0 is top)
        sampling_cols = grid_x_center + world_aligned_local_coords[:, 0]
        sampling_rows = grid_y_center_inverted - world_aligned_local_coords[:, 1] 
        
        sampling_coords = np.stack([sampling_rows, sampling_cols])

        local_map_flat_log_odds = map_coordinates(
            self.grid, 
            sampling_coords, 
            order=0, 
            cval=0.0,
            prefilter=False 
        )
        
        local_map_prob = self._log_odds_to_prob(local_map_flat_log_odds)
        local_map = local_map_prob.reshape((self.num_cells_local, self.num_cells_local))
        return local_map.copy()

    def visualize(self, robot_pos_world, robot_yaw_world, goal_pos_world, 
                robot_path=None, obstacle_path=None, obstacle_pos_world=None,
                debug_target_pos=None, debug_desired_dir=None):
        
        half_world = self.world_size_m / 2.0
        prob_grid = self._log_odds_to_prob(self.grid)
        
        if self.fig is None:
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(8, 8))
            # [核心修复] 不使用 flipud
            self.im = self.ax.imshow(prob_grid, cmap='gray_r',
                                vmin=0, vmax=1,
                                extent=[-half_world, half_world, -half_world, half_world])
            self.ax.set_title("Lidar Grid Map")
            
            robot_circle = plt.Circle((0, 0), 0.2, color='blue', zorder=4)
            self.robot_patch = self.ax.add_patch(robot_circle)
            self.robot_arrow_patch = self.ax.add_patch(plt.Arrow(0, 0, 0.5, 0, width=0.2, color='blue', zorder=4))
            self.goal_patch, = self.ax.plot([], [], '*', color='red', markersize=15, zorder=5)
            self.path_patch, = self.ax.plot([], [], '-', color='cyan', linewidth=2)
            self.obstacle_traj_patch, = self.ax.plot([], [], '--', color='lime', linewidth=1.5)
            self.obstacle_patch = self.ax.add_patch(plt.Circle((0, 0), 0.3, color='orange', zorder=3))
            self.debug_target_patch, = self.ax.plot([], [], 'x', color='lime', markersize=10)
            self.debug_heading_arrow = None
        
        if not plt.fignum_exists(self.fig.number): return

        try:
            # [核心修复] 更新时也不 flipud
            self.im.set_data(prob_grid)
            
            self.robot_patch.center = (robot_pos_world[0], robot_pos_world[1])
            arrow_dx, arrow_dy = 0.5 * np.cos(robot_yaw_world), 0.5 * np.sin(robot_yaw_world)
            if self.robot_arrow_patch: self.robot_arrow_patch.remove()
            self.robot_arrow_patch = plt.Arrow(robot_pos_world[0], robot_pos_world[1], 
                                            arrow_dx, arrow_dy, width=0.2, 
                                            color='blue', zorder=4)
            self.ax.add_patch(self.robot_arrow_patch)
            
            self.goal_patch.set_data([goal_pos_world[0]], [goal_pos_world[1]])
            
            if robot_path is not None and len(robot_path) > 0:
                self.path_patch.set_data(robot_path[:, 0], robot_path[:, 1])
            else:
                self.path_patch.set_data([], [])

            if obstacle_path is not None and len(obstacle_path) > 0:
                self.obstacle_traj_patch.set_data(obstacle_path[:, 0], obstacle_path[:, 1])
            else:
                self.obstacle_traj_patch.set_data([], [])

            if obstacle_pos_world is not None:
                self.obstacle_patch.center = (obstacle_pos_world[0], obstacle_pos_world[1])
                self.obstacle_patch.set_visible(True)
            else:
                self.obstacle_patch.set_visible(False) 

            if debug_target_pos is not None:
                self.debug_target_patch.set_data([debug_target_pos[0]], [debug_target_pos[1]])
                self.debug_target_patch.set_visible(True)
            else:
                self.debug_target_patch.set_visible(False)

            if self.debug_heading_arrow: self.debug_heading_arrow.remove()
            if debug_desired_dir is not None:
                d_dx, d_dy = debug_desired_dir[0] * 0.8, debug_desired_dir[1] * 0.8
                self.debug_heading_arrow = plt.Arrow(robot_pos_world[0], robot_pos_world[1], 
                                                d_dx, d_dy, width=0.15, color='magenta', zorder=5)
                self.ax.add_patch(self.debug_heading_arrow)

            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
        except Exception as e:
            print(f"Visualize error: {e}")

    def close_visualization(self):
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None

class VelocitySmoother:
    def __init__(self, dt=0.01, max_accel=[0.5, 0.3, 0.8], max_jerk=[2.0, 1.5, 3.0]):
        self.dt = dt
        self.max_accel = np.array(max_accel, dtype=np.double)
        self.max_jerk = np.array(max_jerk, dtype=np.double)
        self.current_vel = np.zeros(3, dtype=np.double)
        self.current_accel = np.zeros(3, dtype=np.double)
        self.target_vel = np.zeros(3, dtype=np.double)

    def update(self, target_vel):
        accel_target = (target_vel - self.current_vel) / self.dt
        jerk = (accel_target - self.current_accel) / self.dt
        jerk = np.clip(jerk, -self.max_jerk, self.max_jerk)
        self.current_accel += jerk * self.dt
        self.current_accel = np.clip(self.current_accel, -self.max_accel, self.max_accel)
        self.current_vel += self.current_accel * self.dt
        return self.current_vel.copy()

    def reset(self):
        self.current_vel.fill(0)
        self.current_accel.fill(0)
        self.target_vel.fill(0)

class LocomotionController:
    def __init__(self, policy, model, data, cfg, parent_env):
        self.policy = policy
        self.model = model
        self.data = data
        self.cfg = cfg
        self.env = parent_env
        self.action = np.zeros(12, dtype=np.double)
        self.last_action = np.zeros(12, dtype=np.double)
        self.default_joint_pos = np.array([0.0, 0.0, 0.0, 0.0, -0.26, -0.26, 0.52, 0.52, -0.26, -0.26, 0.0, 0.0])
        self.default_joint_pos_mujoco = np.array([0.0, 0.0, -0.26, 0.52, -0.26, 0.0, 0.0, 0.0, -0.26, 0.52, -0.26, 0.0])
        self.obs_history_length = 10
        obs_buffer_size = 48 * self.obs_history_length
        self.obs_history = np.zeros(obs_buffer_size, dtype=np.float32)
        self.first_obs = True
        self.count_lowlevel = 0
        self.velocity_smoother = VelocitySmoother(
            dt=cfg.sim_config.dt * cfg.sim_config.decimation,
            max_accel=[0.5, 0.3, 1.0],
            max_jerk=[2.0, 1.5, 3.0]
        )

    def reset(self):
        self.first_obs = True
        self.last_action.fill(0)
        self.obs_history.fill(0)
        self.count_lowlevel = 0
        self.velocity_smoother.reset()

    def get_obs_from_sim(self):
        q = self.data.qpos.astype(np.double)
        dq = self.data.qvel.astype(np.double)
        omega = self.data.sensor('angular-velocity').data.astype(np.double)
        quat = self.data.sensor('orientation').data[[1, 2, 3, 0]].astype(np.double)
        r = R.from_quat(quat)
        euler_angle = r.as_euler('xyz', degrees=False)
        joint_pos = q[-28:]
        isaac_joint_pos = joint_pos[[0, 14, 2, 16, 4, 18, 7, 21, 10, 24, 12, 26]]
        joint_vel = dq[-28:]
        isaac_joint_vel = joint_vel[[0, 14, 2, 16, 4, 18, 7, 21, 10, 24, 12, 26]]
        return isaac_joint_pos, isaac_joint_vel, omega, euler_angle

    def step(self, cmd_vx, cmd_vy, cmd_dyaw):
        smoothed_vel = self.velocity_smoother.update(np.array([cmd_vx, cmd_vy, cmd_dyaw]))
        cmd_vx_s, cmd_vy_s, cmd_dyaw_s = smoothed_vel

        isaac_joint_pos, isaac_joint_vel, omega, euler_angle = self.get_obs_from_sim()
        new_obs = np.zeros(48, dtype=np.float32)
        new_obs[0:3] = [cmd_vx_s, cmd_vy_s, cmd_dyaw_s]
        new_obs[3] = 0.8
        new_obs[4:16] = isaac_joint_pos - self.default_joint_pos
        new_obs[16:28] = isaac_joint_vel
        new_obs[28:31] = omega
        new_obs[31:34] = euler_angle
        new_obs[34:46] = self.last_action
        phase_time = self.count_lowlevel * self.cfg.sim_config.dt
        new_obs[46] = math.sin(2 * math.pi * phase_time / 0.8)
        new_obs[47] = math.cos(2 * math.pi * phase_time / 0.8)
        
        group_sizes = [3, 1, 12, 12, 3, 3, 12, 1, 1]
        start_idx_obs, start_idx_new_obs = 0, 0
        
        if self.first_obs:
            for group_size in group_sizes:
                group_data = new_obs[start_idx_new_obs : start_idx_new_obs + group_size]
                self.obs_history[start_idx_obs : start_idx_obs + group_size * self.obs_history_length] = np.tile(group_data, self.obs_history_length)
                start_idx_obs += group_size * self.obs_history_length
                start_idx_new_obs += group_size
            self.first_obs = False
        else:
            for group_size in group_sizes:
                group_data = new_obs[start_idx_new_obs : start_idx_new_obs + group_size]
                obs_slice = self.obs_history[start_idx_obs : start_idx_obs + group_size * self.obs_history_length]
                rolled_slice = np.roll(obs_slice, -group_size)
                rolled_slice[-group_size:] = group_data
                self.obs_history[start_idx_obs : start_idx_obs + group_size * self.obs_history_length] = rolled_slice
                start_idx_obs += group_size * self.obs_history_length
                start_idx_new_obs += group_size
                
        self.action[:] = self.policy(torch.tensor(self.obs_history).unsqueeze(0))[0].detach().numpy()
        mujoco_action = self.action[[0, 2, 4, 6, 8, 10, 1, 3, 5, 7, 9, 11]]
        self.last_action[:] = self.action
        target_joint_pos = self.cfg.robot_config.action_scale * mujoco_action + self.default_joint_pos_mujoco

        for _ in range(self.cfg.sim_config.decimation):
            q = self.data.qpos.astype(np.double)
            dq = self.data.qvel.astype(np.double)
            mujoco_joint_pos = q[-28:][[0, 2, 4, 7, 10, 12, 14, 16, 18, 21, 24, 26]]
            mujoco_joint_vel = dq[-28:][[0, 2, 4, 7, 10, 12, 14, 16, 18, 21, 24, 26]]
            tau = pd_control(target_joint_pos, mujoco_joint_pos, self.cfg.robot_config.kps,
                             np.zeros(12), mujoco_joint_vel, self.cfg.robot_config.kds)
            self.data.ctrl[:] = np.clip(tau, -self.cfg.robot_config.tau_limit, self.cfg.robot_config.tau_limit)
            mujoco.mj_step(self.model, self.data)
            self.count_lowlevel += 1
            if self.env.render_mode == 'human':
                if self.env.viewer_handle is None:
                    self.env.viewer_handle = mujoco.viewer.launch_passive(self.model, self.data)
                if self.count_lowlevel % self.env.render_decimation == 0:
                    self.env.viewer_handle.sync()

class RobotVisualEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}

    def __init__(self, model_path, low_level_policy_path, render_mode='rgb_array', 
                 render_decimation=5, action_repeat=4, history_length=15, 
                 enable_dynamic_obstacles=False): 
        super().__init__()
        self.render_mode = render_mode
        self.render_decimation = render_decimation
        self.action_repeat = action_repeat
        self.enable_dynamic_obstacles = enable_dynamic_obstacles 
        
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.model.opt.timestep = 0.001
        self.data = mujoco.MjData(self.model)
        
        self.grid_map = GlobalGridMap(world_size_m=20.0, local_map_size_m=10.0, resolution=6)
        
        self.lidar_num_rays = 180       
        self.lidar_max_range = 15.0     
        self.lidar_fov = np.pi          
        self.lidar_angles = np.linspace(-self.lidar_fov/2, self.lidar_fov/2, self.lidar_num_rays)
        
        self.history_length = history_length
        self.state_feature_dim = 9
        self.state_history_buffer = deque(maxlen=self.history_length)

        low_level_policy = torch.jit.load(low_level_policy_path)
        class LocomotionCfg:
            class sim_config: dt = 0.001; decimation = 10
            class robot_config:
                kps = np.array([200, 200, 350, 350, 35, 35] * 2, dtype=np.double)
                kds = np.array([10] * 12, dtype=np.double)
                tau_limit = np.array([240, 240, 240, 240, 40, 40] * 2, dtype=np.double)
                action_scale = 0.25
        self.locomotion_controller = LocomotionController(low_level_policy, self.model, self.data, LocomotionCfg(), parent_env=self)

        self.observation_space = spaces.Dict({
            "grid_map": spaces.Box(low=0, high=1, 
                                   shape=(1, self.grid_map.num_cells_local, self.grid_map.num_cells_local), 
                                   dtype=np.float32),
            "state_history": spaces.Box(low=-np.inf, high=np.inf, shape=(self.history_length * self.state_feature_dim,), dtype=np.float32)
        })
        
        action_low = np.array([-0.6, -0.5, -0.85], dtype=np.float32)
        action_high = np.array([0.8, 0.5, 0.85], dtype=np.float32)
        self.action_space = spaces.Box(low=action_low, high=action_high, dtype=np.float32)
        self.goal_pos = np.array([5.0, 0.0])
        self.max_episode_steps = 15000
        self.current_step = 0
        self.last_applied_action = np.zeros(3, dtype=np.float32)
        self.max_action_rate = np.array([0.25, 0.2, 0.25]) 
        self.fall_threshold = 0.6
        
        self.reward_scales = {
            "success": 1500.0, "collision_penalty": -400.0, "fall_penalty": -300.0,
            "path_following": 10.0, "cte_penalty": 0.05, "velocity_to_goal": 1.0,
            "heading_to_goal": 0.6, "obstacle_avoidance": 0.03, "turn_penalty": 0.000001,
            "action_rate_penalty": 0.01, "unstable_penalty": -0.00001, "exist_penalty": -0.00001,
            "distance_to_goal": 0.0, "exploration_turn": 0.0, "goal_discovery": 0.0, "safety_margin": 0.5,
        }
        
        self.path_planner = PathPlanner(self.grid_map)
        self.current_path = None
        self.current_waypoint_index = 0
        self.path_update_freq = 25
        self.path_update_counter = 0
        self.viewer_handle = None
        self.debug_target_pos = None
        self.debug_desired_dir = None

        self.static_walls_info = [] 
        for i in range(self.model.ngeom):
            geom_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, i)
            if geom_name and 'static_wall' in geom_name:
                if self.model.geom_type[i] == mujoco.mjtGeom.mjGEOM_BOX:
                    pos = self.model.geom_pos[i]
                    size = self.model.geom_size[i]
                    self.static_walls_info.append({'pos': pos, 'size': size})

        self.dyn_obs_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'dynamic_obstacle')
        if self.enable_dynamic_obstacles and self.dyn_obs_body_id != -1:
            self.dyn_obs_mocap_id = self.model.body_mocapid[self.dyn_obs_body_id]
            dyn_obs_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'dynamic_obstacle_geom')
            obstacle_radius = self.model.geom_size[dyn_obs_geom_id][0]
            self.obstacle_controller = DynamicObstacleController(
                static_walls=self.static_walls_info, 
                world_size_m=self.grid_map.world_size_m, 
                resolution=self.grid_map.resolution,
                obstacle_radius=obstacle_radius,
                speed=0.2 
            )
        else:
            self.obstacle_controller = None
            if self.dyn_obs_body_id != -1:
                self.dyn_obs_mocap_id = self.model.body_mocapid[self.dyn_obs_body_id]
            else:
                self.dyn_obs_mocap_id = -1

        self.obstacle_geom_ids = set()
        for i in range(self.model.ngeom):
            geom_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, i)
            if geom_name and ('static_wall' in geom_name or 'dynamic_obstacle_geom' in geom_name):
                self.obstacle_geom_ids.add(i)
        
        self.floor_geom_id = -1
        for i in range(self.model.ngeom):
            geom_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, i)
            if geom_name and ('floor' in geom_name or 'ground' in geom_name):
                self.floor_geom_id = i; break
        if self.floor_geom_id == -1:
            for i in range(self.model.ngeom):
                if self.model.geom_type[i] == mujoco.mjtGeom.mjGEOM_PLANE: self.floor_geom_id = i; break
        
        self.robot_base_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'base_link')
        self.depth_camera_name = "d435i_depth"
        self.camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, self.depth_camera_name)
        
        self.goal_zones = [
            {'x_range': [-4.5, -0.5], 'y_range': [1.5, 4.5]}, {'x_range': [0.5, 4.5], 'y_range': [1.5, 4.5]},
            {'x_range': [-4.5, -0.5], 'y_range': [-4.5, -1.5]}, {'x_range': [0.5, 4.5], 'y_range': [-4.5, -1.5]},
            {'x_range': [-4.5, 4.5], 'y_range': [-0.9, 0.9]},
        ]
        
        s = self.grid_map.num_cells_local
        y, x = np.ogrid[:s, :s]
        center = s / 2.0
        dist_in_cells = np.sqrt((x - center)**2 + (y - center)**2)
        dist_in_meters = dist_in_cells / self.grid_map.resolution
        sigma = 0.6 
        self.repulsion_kernel = np.exp(- (dist_in_meters ** 2) / (2 * sigma ** 2))
        self.repulsion_kernel[dist_in_meters < 0.3] = 0.0

    def _randomize_map(self):
        generator = MazeGridGenerator(world_size=20.0, grid_dim=8)
        generated_walls = generator.generate(self.np_random)
        
        wall_idx = 0
        self.static_walls_info =[]
        for i in range(self.model.ngeom):
            geom_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, i)
            if geom_name and 'static_wall' in geom_name:
                if wall_idx < len(generated_walls):
                    wall_data = generated_walls[wall_idx]
                    self.model.geom_pos[i] = wall_data['pos']
                    self.model.geom_size[i] = wall_data['size']
                    self.static_walls_info.append({
                        'pos': wall_data['pos'].copy(), 
                        'size': wall_data['size'].copy()
                    })
                    wall_idx += 1
                else:
                    # 将用不到的墙体模块藏在地下深处
                    self.model.geom_pos[i] = np.array([0, 0, -100])
                    self.model.geom_size[i] = np.array([0.1, 0.1, 0.1])
                    
        mujoco.mj_forward(self.model, self.data)

    def _update_path(self):
        robot_pos = self.data.xpos[self.robot_base_body_id][:2]
        path = self.path_planner.find_path(robot_pos, self.goal_pos)
        if path is not None and len(path) > 1:
            self.current_path = path
            self.current_waypoint_index = 1
        else:
            self.current_path = None

    def _calculate_cross_track_error(self, robot_pos, path, start_idx=0):
        if path is None or len(path) < 2: return 0.0
        search_radius = 20
        start = max(0, start_idx - 5)
        end = min(len(path), start_idx + search_radius)
        local_path = path[start:end]
        if len(local_path) < 2:
            local_path = path; start = 0
        dists = np.linalg.norm(local_path - robot_pos, axis=1)
        min_local_idx = np.argmin(dists)
        global_idx = start + min_local_idx
        if global_idx + 1 < len(path):
            p1 = path[global_idx]; p2 = path[global_idx + 1]
        elif global_idx - 1 >= 0:
            p1 = path[global_idx - 1]; p2 = path[global_idx]
        else:
            return 0.0
        segment_vec = p2 - p1
        segment_len_sq = np.dot(segment_vec, segment_vec)
        if segment_len_sq < 1e-6: return dists[min_local_idx]
        robot_vec = robot_pos - p1
        t = np.dot(robot_vec, segment_vec) / segment_len_sq
        t = np.clip(t, 0.0, 1.0)
        closest_point = p1 + t * segment_vec
        cte = np.linalg.norm(robot_pos - closest_point)
        return cte

    def _get_robot_pose(self):
        robot_pos_2d = self.data.xpos[self.robot_base_body_id][:2].copy()
        mat = self.data.xmat[self.robot_base_body_id].reshape(3, 3)
        robot_yaw = math.atan2(mat[1, 0], mat[0, 0])
        return robot_pos_2d, robot_yaw
    
    def _update_waypoint_index(self, robot_pos, robot_vel_norm):
        if self.current_path is None or len(self.current_path) < 2: return
        max_crawl_steps = 8 
        best_idx = self.current_waypoint_index
        current_dist = np.linalg.norm(robot_pos - self.current_path[best_idx])
        for i in range(max_crawl_steps):
            next_idx = best_idx + 1
            if next_idx >= len(self.current_path): break
            next_dist = np.linalg.norm(robot_pos - self.current_path[next_idx])
            if next_dist < current_dist:
                best_idx = next_idx; current_dist = next_dist
            else:
                break
        self.current_waypoint_index = best_idx
        lookahead_dist = 0.5 + 0.4 * robot_vel_norm 
        target_idx = self.current_waypoint_index
        accumulated_dist = 0.0
        for i in range(self.current_waypoint_index, len(self.current_path) - 1):
            segment_len = np.linalg.norm(self.current_path[i+1] - self.current_path[i])
            accumulated_dist += segment_len
            if accumulated_dist > lookahead_dist:
                target_idx = i + 1; break
        self.current_target_waypoint = self.current_path[target_idx]

    def _sense_lidar(self):
        if self.camera_id != -1:
            lidar_origin = self.data.cam_xpos[self.camera_id].copy()
        else:
            lidar_origin = self.data.xpos[self.robot_base_body_id].copy()
            lidar_origin[2] += 0.2

        body_mat = self.data.xmat[self.robot_base_body_id].reshape(3, 3)

        local_vx = np.cos(self.lidar_angles)
        local_vy = np.sin(self.lidar_angles)
        local_vz = np.zeros_like(local_vx)
        local_rays = np.stack([local_vx, local_vy, local_vz], axis=0) 

        global_rays = body_mat @ local_rays
        
        hit_points = []
        valid_mask = []
        
        geomgroup = np.array([1, 1, 1, 1, 1, 1], dtype=np.uint8) 
        flg_static = 1 
        body_exclude = int(self.robot_base_body_id) 
        geomid_out = np.zeros(1, dtype=np.int32)
        
        for i in range(self.lidar_num_rays):
            vec = np.ascontiguousarray(global_rays[:, i], dtype=np.float64) 
            dist = mujoco.mj_ray(self.model, self.data, lidar_origin, vec, geomgroup, flg_static, body_exclude, geomid_out)
            
            if dist != -1 and dist < self.lidar_max_range:
                hit_pos = lidar_origin + vec * dist
                hit_points.append(hit_pos[:2])
                valid_mask.append(True)
            else:
                end_pos = lidar_origin + vec * self.lidar_max_range
                hit_points.append(end_pos[:2])
                valid_mask.append(False)
        
        return np.array(hit_points), np.array(valid_mask), lidar_origin
        
    def _update_perception(self):
        hit_points, valid_mask, lidar_origin_3d = self._sense_lidar()
        
        lidar_pos_2d = lidar_origin_3d[:2]
        self.grid_map.update_from_lidar(lidar_pos_2d, hit_points, valid_mask)
        
        if self.render_mode == 'human':
            import matplotlib
            if matplotlib.get_backend() != 'Agg':
                pos, yaw = self._get_robot_pose()
                obstacle_traj = None
                obstacle_pos = None
                if self.obstacle_controller is not None:
                    obstacle_traj = self.obstacle_controller.full_trajectory
                    obstacle_pos = self.data.mocap_pos[self.dyn_obs_mocap_id]

                self.grid_map.visualize(
                    robot_pos_world=pos, 
                    robot_yaw_world=yaw, 
                    goal_pos_world=self.goal_pos, 
                    robot_path=self.current_path, 
                    obstacle_path=obstacle_traj,
                    obstacle_pos_world=obstacle_pos,
                    debug_target_pos=self.debug_target_pos,
                    debug_desired_dir=self.debug_desired_dir
                )
            
    def _get_obs(self):
        robot_pos_2d, robot_yaw = self._get_robot_pose()
        local_grid_map = self.grid_map.get_local_map(robot_pos_2d, robot_yaw)
        local_grid_map = local_grid_map[..., np.newaxis] 
        local_grid_map = np.transpose(local_grid_map, (2, 0, 1)) 
        
        robot_pos_3d = self.data.qpos[0:3]
        robot_quat = self.data.qpos[3:7] 
        r = R.from_quat([robot_quat[1], robot_quat[2], robot_quat[3], robot_quat[0]])
        world_lin_vel = self.data.qvel[0:3]
        world_ang_vel = self.data.qvel[3:6]
        robot_lin_vel = r.apply(world_lin_vel, inverse=True) 
        
        world_goal_vec = self.goal_pos - robot_pos_2d
        rel_goal_pos = r.apply(np.array([world_goal_vec[0], world_goal_vec[1], 0]), inverse=True)[:2]
        
        if self.current_path is None or not hasattr(self, 'current_target_waypoint'):
            target_wp = self.goal_pos
        else:
            target_wp = self.current_target_waypoint
        world_wp_vec = target_wp - robot_pos_2d
        rel_waypoint_pos = r.apply(np.array([world_wp_vec[0], world_wp_vec[1], 0]), inverse=True)[:2]

        current_cte = 0.0
        current_heading_error = 0.0
        
        if self.current_path is not None and len(self.current_path) > 1:
            current_cte = self._calculate_cross_track_error(robot_pos_2d, self.current_path, self.current_waypoint_index)
            dists = np.linalg.norm(self.current_path - robot_pos_2d, axis=1)
            nearest_idx = np.argmin(dists)
            next_idx = min(nearest_idx + 3, len(self.current_path) - 1)
            if next_idx > nearest_idx:
                path_vec = self.current_path[next_idx] - self.current_path[nearest_idx]
                target_yaw = math.atan2(path_vec[1], path_vec[0])
                diff = target_yaw - robot_yaw
                diff = (diff + np.pi) % (2 * np.pi) - np.pi
                current_heading_error = diff

        current_state_features = np.concatenate([
            rel_goal_pos, rel_waypoint_pos, robot_lin_vel[:2], [world_ang_vel[2]], 
            [current_cte], [current_heading_error]
        ]).astype(np.float32)

        self.state_history_buffer.append(current_state_features)
        history_list = list(self.state_history_buffer)
        if len(history_list) < self.history_length:
            padding = [history_list[0]] * (self.history_length - len(history_list))
            history_list = padding + history_list
        state_history = np.concatenate(history_list).astype(np.float32)

        return {
            "grid_map": local_grid_map, 
            "state_history": state_history
        }

    def _check_collision(self):
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1 = contact.geom1; geom2 = contact.geom2
            g1_is_obs = geom1 in self.obstacle_geom_ids
            g2_is_obs = geom2 in self.obstacle_geom_ids
            if g1_is_obs and g2_is_obs: continue
            if not g1_is_obs and not g2_is_obs: continue
            other_geom = geom2 if g1_is_obs else geom1
            if other_geom != self.floor_geom_id: return True
        return False
        
    def _reset_goal(self):
        zone = self.goal_zones[self.np_random.integers(0, len(self.goal_zones))]
        self.goal_pos = self.np_random.uniform(low=[zone['x_range'][0], zone['y_range'][0]], high=[zone['x_range'][1], zone['y_range'][1]])
        target_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, 'target_goal')
        if target_site_id != -1: self.model.site_pos[target_site_id][:2] = self.goal_pos

    def _is_valid_spawn_pos(self, pos, min_dist_to_wall=0.8, min_dist_to_goal=3.0):
        half_size = self.grid_map.world_size_m / 2.0 - 0.5 
        if abs(pos[0]) > half_size or abs(pos[1]) > half_size: return False
        robot_radius = 0.4; safe_margin = robot_radius + min_dist_to_wall 
        for wall in self.static_walls_info:
            w_pos, w_size = wall['pos'], wall['size']
            x_overlap = (w_pos[0] - w_size[0] - safe_margin) < pos[0] < (w_pos[0] + w_size[0] + safe_margin)
            y_overlap = (w_pos[1] - w_size[1] - safe_margin) < pos[1] < (w_pos[1] + w_size[1] + safe_margin)
            if x_overlap and y_overlap: return False 
        if self.obstacle_controller is not None:
             dyn_obs_pos = self.obstacle_controller.get_current_position()
             if np.linalg.norm(pos - dyn_obs_pos) < 2.0: return False
        if np.linalg.norm(pos - self.goal_pos) < min_dist_to_goal: return False
        return True

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # --- 1. 重新生成迷宫并同步到物理世界 ---
        self._randomize_map()
        
        # --- 2. 如果有动态障碍物，刷新其避障地图 ---
        if self.obstacle_controller is not None:
            self.obstacle_controller.update_walls(self.static_walls_info)
            
        mujoco.mj_resetData(self.model, self.data)
        
        self.grid_map.reset()
        self.locomotion_controller.reset()
        self.state_history_buffer.clear()
        self.current_step = 0
        self.last_applied_action.fill(0)
        self._reset_goal() 
        
        if self.obstacle_controller is not None:
            self.obstacle_controller.reset()
            initial_obs_pos_2d = self.obstacle_controller.get_current_position()
            self.data.mocap_pos[self.dyn_obs_mocap_id] = [initial_obs_pos_2d[0], initial_obs_pos_2d[1], 0.5]
        elif self.dyn_obs_mocap_id != -1:
            self.data.mocap_pos[self.dyn_obs_mocap_id] =[0.0, 0.0, -10.0]

        robot_spawn_pos = None; robot_spawn_yaw = 0.0; valid_pos_found = False
        for _ in range(200): 
            # 将机器人随机重置范围扩大为20x20内的 -8.5 到 8.5
            x = self.np_random.uniform(-8.5, 8.5)
            y = self.np_random.uniform(-8.5, 8.5)
            candidate_pos = np.array([x, y])
            if self._is_valid_spawn_pos(candidate_pos, min_dist_to_wall=0.8):
                robot_spawn_pos = candidate_pos
                robot_spawn_yaw = self.np_random.uniform(-math.pi, math.pi)
                valid_pos_found = True
                break
        
        if not valid_pos_found: 
            return self.reset(seed=seed, options=options)
        
        self.data.qpos[0] = robot_spawn_pos[0]
        self.data.qpos[1] = robot_spawn_pos[1]
        self.data.qpos[2] = 0.05 
        quat_xyzw = R.from_euler('z', robot_spawn_yaw).as_quat()
        self.data.qpos[3:7] =[quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]

        mujoco.mj_forward(self.model, self.data)
        try:
            for _ in range(20):
                self.data.ctrl[:] = 0
                mujoco.mj_step(self.model, self.data)
        except Exception: 
            return self.reset(seed=seed, options=options)

        is_colliding = self._check_collision()
        robot_quat = self.data.qpos[3:7]
        r = R.from_quat([robot_quat[1], robot_quat[2], robot_quat[3], robot_quat[0]])
        z_axis = r.apply([0, 0, 1])
        is_fallen = z_axis[2] < 0.5

        if is_colliding or is_fallen: 
            return self.reset(seed=seed, options=options)
        
        self.dist_to_goal_start = np.linalg.norm(self.data.qpos[:2] - self.goal_pos)
        if self.dist_to_goal_start < 0.1: self.dist_to_goal_start = 0.1
        self.prev_dist_to_goal = self.dist_to_goal_start 
        
        self._update_perception()
        self.path_update_counter = 0
        self._update_path()
        
        if self.current_path is None: 
            return self.reset(seed=seed, options=options)
            
        return self._get_obs(), {}
    
    def _update_mocap_obstacle(self, dt):
        if self.obstacle_controller is None: return
        new_pos_2d = self.obstacle_controller.update(dt)
        self.data.mocap_pos[self.dyn_obs_mocap_id] = [new_pos_2d[0], new_pos_2d[1], 0.5]

    def step(self, action):
        clipped_delta = np.clip(action - self.last_applied_action, -self.max_action_rate, self.max_action_rate)
        smoothed_action = self.last_applied_action + clipped_delta
        self.last_applied_action = smoothed_action

        total_reward = 0.0; terminated = False; truncated = False
        self.path_update_counter += 1
        if self.path_update_counter >= self.path_update_freq:
            self._update_path(); self.path_update_counter = 0

        for _ in range(self.action_repeat):
            sub_step_dt = self.locomotion_controller.cfg.sim_config.decimation * self.model.opt.timestep
            self._update_mocap_obstacle(sub_step_dt)
            self.locomotion_controller.step(*smoothed_action)
            
            pos_after = self.data.xpos[self.robot_base_body_id][:2]
            lin_vel_world = self.data.qvel[:2] 
            robot_vel_norm = np.linalg.norm(lin_vel_world)
            self._update_waypoint_index(pos_after, robot_vel_norm)
            
            cte = 0.0; path_tangent_dir = None
            if self.current_path is not None and len(self.current_path) > 1:
                cte = self._calculate_cross_track_error(pos_after, self.current_path, self.current_waypoint_index)
                dists = np.linalg.norm(self.current_path - pos_after, axis=1)
                nearest_idx = np.argmin(dists)
                next_idx = min(nearest_idx + 3, len(self.current_path) - 1) 
                if next_idx > nearest_idx:
                    vec = self.current_path[next_idx] - self.current_path[nearest_idx]
                    path_tangent_dir = vec / (np.linalg.norm(vec) + 1e-6)

            if self.current_path is not None and hasattr(self, 'current_target_waypoint'):
                target_pos_world = self.current_target_waypoint
                to_target_vec = target_pos_world - pos_after
                dist_to_target = np.linalg.norm(to_target_vec)
                to_target_dir = to_target_vec / (dist_to_target + 1e-6)
                # Pure-pursuit direction toward the current lookahead waypoint.
                desired_dir = to_target_dir
            else:
                target_pos_world = self.goal_pos
                to_target_vec = target_pos_world - pos_after
                desired_dir = to_target_vec / (np.linalg.norm(to_target_vec) + 1e-6)

            self.debug_target_pos = target_pos_world; self.debug_desired_dir = desired_dir

            mat = self.data.xmat[self.robot_base_body_id].reshape(3, 3)
            robot_yaw = math.atan2(mat[1, 0], mat[0, 0])
            forward_dir = np.array([np.cos(robot_yaw), np.sin(robot_yaw)])

            reward_path = 0.0 
            reward_cte = -cte * self.reward_scales["cte_penalty"]
            if cte > 0.4: reward_cte *= 2.0
            
            heading_alignment = np.dot(forward_dir, desired_dir)
            reward_heading = heading_alignment * self.reward_scales["heading_to_goal"]

            vel_projected = np.dot(lin_vel_world, desired_dir)
            if heading_alignment < 0 and vel_projected > 0: valid_velocity = vel_projected * 0.1 
            else: valid_velocity = vel_projected
            reward_velocity = np.clip(valid_velocity, -0.5, 1.0) * self.reward_scales["velocity_to_goal"]
            
            local_map = self.grid_map.get_local_map(pos_after, robot_yaw)
            obstacle_mask = local_map > 0.6
            reward_safety = 0.0
            obs_indices = np.argwhere(obstacle_mask)
            if len(obs_indices) > 0:
                center = self.grid_map.num_cells_local // 2
                dists = np.sqrt(np.sum((obs_indices - [center, center])**2, axis=1))
                min_dist_meters = np.min(dists) / self.grid_map.resolution
                if min_dist_meters < 0.35:
                     reward_safety = -self.reward_scales["safety_margin"] * (0.35 - min_dist_meters)

            repulsion_field = local_map * self.repulsion_kernel
            total_repulsion = np.sum(repulsion_field[obstacle_mask])
            reward_obstacle = -self.reward_scales["obstacle_avoidance"] * total_repulsion

            ang_vel_z = self.data.qvel[5]
            penalty_turn = -self.reward_scales["turn_penalty"] * (abs(ang_vel_z) + 0.1 * (ang_vel_z ** 2))
            
            z_axis_world = mat[:, 2] 
            up_alignment = z_axis_world[2]
            
            penalty_unstable = 0.0
            if up_alignment < 0.8: penalty_unstable = self.reward_scales["unstable_penalty"]

            penalty_act_rate = -self.reward_scales["action_rate_penalty"] * np.linalg.norm(action - self.last_applied_action)
            penalty_exist = self.reward_scales["exist_penalty"]

            sub_reward = (
                reward_path + reward_cte + reward_velocity + 
                reward_heading + reward_obstacle + reward_safety + 
                penalty_turn + penalty_unstable + 
                penalty_act_rate + penalty_exist
            )
            
            dist_to_final_goal = np.linalg.norm(pos_after - self.goal_pos)
            succeeded = dist_to_final_goal < 0.6
            collision = self._check_collision()
            fell = up_alignment < self.fall_threshold

            self.current_step += 1
            truncated = self.current_step >= self.max_episode_steps
            
            term_reward = 0.0
            termination_reason = "running" 

            if succeeded:
                term_reward = self.reward_scales["success"]
                terminated = True
                termination_reason = "success"
                print(f"Goal Reached! Reward: {term_reward}")
            elif collision or fell or truncated:
                terminated = True
                if collision:
                    term_reward += self.reward_scales["collision_penalty"]
                    termination_reason = "collision"
                elif fell:
                    term_reward += self.reward_scales["fall_penalty"]
                    termination_reason = "fall"
                elif truncated:
                    termination_reason = "timeout"
                
                progress_ratio = 1.0 - (dist_to_final_goal / self.dist_to_goal_start)
                progress_ratio = np.clip(progress_ratio, 0.0, 1.0)
                progress_bonus = 600.0 * progress_ratio
                term_reward += progress_bonus

            total_reward += sub_reward + term_reward
            if terminated or truncated: break
        
        self._update_perception()
        
        info_dict = {
            "is_success": succeeded,
            "distance_to_goal": dist_to_final_goal,
            "termination_reason": termination_reason,
            "robot_pos": pos_after.copy(),
            "rewards": {
                "path": reward_path, "cte": reward_cte, "vel": reward_velocity,
                "head": reward_heading, "safe": reward_safety, "dist": dist_to_final_goal 
            }
        }

        return self._get_obs(), total_reward, terminated, truncated, info_dict

    def render(self): pass
    def close(self):
        if self.viewer_handle:
            if self.viewer_handle.is_running(): self.viewer_handle.close()
            self.viewer_handle = None
        self.grid_map.close_visualization()
