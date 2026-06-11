"""
End-to-end navigation environment with curriculum learning and geodesic shaping.

The observation contains a 4-channel grid map and a 12-dimensional state vector.
Global shortest-path potentials are used for reward shaping during training.
"""

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
from skimage.graph import MCP_Geometric
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates, binary_dilation, distance_transform_edt
from collections import deque
import random
import matplotlib
import heapq

try:
    from core.mpc_tracker import WaypointTracker
except ImportError:
    from .core.mpc_tracker import WaypointTracker


# 辅助函数

def pd_control(target_q, q, kp, target_dq, dq, kd):
    return (target_q - q) * kp + (target_dq - dq) * kd


# 1. 迷宫生成器 (支持课程学习: 可控墙体数量)
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
        stack = [(0, 0)]
        visited[0, 0] = True
        passages = set()

        while stack:
            r, c = stack[-1]
            neighbors = []
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
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

        walls = []
        for r in range(self.grid_dim):
            for c in range(self.grid_dim - 1):
                p1, p2 = (r, c), (r, c + 1)
                if (p1, p2) not in passages:
                    if np_random.random() > self.remove_wall_prob:
                        center_x = (c + 1) * self.cell_size - self.world_size / 2.0
                        center_y = (self.world_size / 2.0) - (r + 0.5) * self.cell_size
                        walls.append({
                            'pos': np.array([center_x, center_y, 1.0]),
                            'is_vertical': True,
                            'size': np.array([self.wall_thickness, self.cell_size / 2.0 + 0.05, 1.0])
                        })

        for r in range(self.grid_dim - 1):
            for c in range(self.grid_dim):
                p1, p2 = (r, c), (r + 1, c)
                if (p1, p2) not in passages:
                    if np_random.random() > self.remove_wall_prob:
                        center_x = (c + 0.5) * self.cell_size - self.world_size / 2.0
                        center_y = (self.world_size / 2.0) - (r + 1) * self.cell_size
                        walls.append({
                            'pos': np.array([center_x, center_y, 1.0]),
                            'is_vertical': False,
                            'size': np.array([self.cell_size / 2.0 + 0.05, self.wall_thickness, 1.0])
                        })

        return walls


# 2. 动态障碍物控制器 (复用原版)
class DynamicObstacleController:
    def __init__(self, static_walls: list, world_size_m: float, resolution: int,
                 obstacle_radius: float, speed: float = 0.4):
        self.speed = speed
        self.world_size_m = world_size_m
        self.resolution = resolution
        self.num_cells = int(world_size_m * resolution)
        self.world_origin_offset_m = np.array([world_size_m / 2.0, world_size_m / 2.0])
        self.obstacle_radius = obstacle_radius
        self.update_map(static_walls)

    def update_map(self, static_walls):
        self.cost_grid = self._create_cost_grid_from_walls(static_walls)
        self.free_indices = np.argwhere(np.isfinite(self.cost_grid))
        if len(self.free_indices) == 0:
            self.cost_grid[:] = 1.0
            self.free_indices = np.argwhere(np.isfinite(self.cost_grid))
        self.reset()

    def _world_to_grid(self, world_pos):
        grid_coords = (world_pos[:2] + self.world_origin_offset_m) * self.resolution
        r = self.num_cells - 1 - int(grid_coords[1])
        c = int(grid_coords[0])
        return np.array([np.clip(r, 0, self.num_cells - 1), np.clip(c, 0, self.num_cells - 1)])

    def _grid_to_world(self, grid_pos):
        y_grid, x_grid = self.num_cells - 1 - grid_pos[0], grid_pos[1]
        return np.array([x_grid, y_grid]) / self.resolution - self.world_origin_offset_m

    def _create_cost_grid_from_walls(self, walls):
        from scipy.ndimage import binary_dilation as _dilation
        wall_map = np.zeros((self.num_cells, self.num_cells), dtype=bool)
        for wall in walls:
            pos, size = wall['pos'], wall['size']
            x_min, x_max = pos[0] - size[0], pos[0] + size[0]
            y_min, y_max = pos[1] - size[1], pos[1] + size[1]
            grid_x_min = int((x_min + self.world_origin_offset_m[0]) * self.resolution)
            grid_x_max = int((x_max + self.world_origin_offset_m[0]) * self.resolution)
            grid_y_min_flipped = self.num_cells - 1 - int((y_max + self.world_origin_offset_m[1]) * self.resolution)
            grid_y_max_flipped = self.num_cells - 1 - int((y_min + self.world_origin_offset_m[1]) * self.resolution)
            grid_x_min, grid_x_max = np.clip([grid_x_min, grid_x_max], 0, self.num_cells - 1)
            grid_y_min_flipped, grid_y_max_flipped = np.clip([grid_y_min_flipped, grid_y_max_flipped], 0, self.num_cells - 1)
            wall_map[grid_y_min_flipped:grid_y_max_flipped + 1, grid_x_min:grid_x_max + 1] = True
        inflation_radius_grid = math.ceil(self.obstacle_radius * self.resolution) + 1
        structure = np.ones((3, 3), dtype=bool)
        inflated = _dilation(wall_map, structure=structure, iterations=inflation_radius_grid)
        cost_grid = np.ones_like(wall_map, dtype=np.float32)
        cost_grid[inflated] = np.inf
        return cost_grid

    def _smooth_path(self, path, num_points=None, s=0.5):
        from scipy.interpolate import splprep, splev
        if len(path) < 4: return path
        try:
            if num_points is None:
                dist = np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1))
                num_points = max(5, int(dist * 10))
            tck, u = splprep([path[:, 0], path[:, 1]], s=s, k=3)
            u_new = np.linspace(u.min(), u.max(), num_points)
            x_new, y_new = splev(u_new, tck)
            return np.c_[x_new, y_new]
        except Exception:
            return path

    def _find_valid_start_grid(self, current_grid):
        if self.cost_grid[current_grid[0], current_grid[1]] != np.inf:
            return current_grid
        for r in range(1, 7):
            r_min, r_max = max(0, current_grid[0] - r), min(self.num_cells - 1, current_grid[0] + r)
            c_min, c_max = max(0, current_grid[1] - r), min(self.num_cells - 1, current_grid[1] + r)
            sub = self.cost_grid[r_min:r_max + 1, c_min:c_max + 1]
            valid = np.argwhere(np.isfinite(sub))
            if len(valid) > 0:
                return valid[0] + np.array([r_min, c_min])
        return None

    def _find_random_target_and_plan(self):
        from skimage.graph import route_through_array
        start_grid = self._find_valid_start_grid(self._world_to_grid(self.current_pos))
        if start_grid is None: return False
        for _ in range(10):
            idx = np.random.randint(0, len(self.free_indices))
            end_grid = self.free_indices[idx]
            if np.linalg.norm(start_grid - end_grid) < 2.0 * self.resolution: continue
            try:
                path_indices, _ = route_through_array(self.cost_grid, start=start_grid, end=end_grid, fully_connected=True, geometric=True)
                if not path_indices or len(path_indices) < 2: continue
                path_world = np.array([self._grid_to_world(p) for p in path_indices])
                self.current_path = self._smooth_path(path_world)
                self.current_path_index = 0
                return True
            except Exception: continue
        return False

    def reset(self):
        if len(self.free_indices) > 0:
            idx = np.random.randint(0, len(self.free_indices))
            self.current_pos = self._grid_to_world(self.free_indices[idx])
        self.is_waiting = False
        self.current_path = None
        self._find_random_target_and_plan()

    def update(self, dt: float) -> np.ndarray:
        if self.is_waiting:
            self.wait_timer += dt
            if self.wait_timer >= self.wait_duration:
                self.is_waiting = False
                if not self._find_random_target_and_plan(): return self.current_pos
            else:
                return self.current_pos
        if self.current_path is None or len(self.current_path) == 0:
            if not self._find_random_target_and_plan(): return self.current_pos
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
            target = self.current_path[next_index]
            vec = target - self.current_pos
            dist = np.linalg.norm(vec)
            if dist < 1e-6:
                self.current_path_index = next_index
                continue
            if distance_to_move >= dist:
                self.current_pos = target.copy()
                distance_to_move -= dist
                self.current_path_index = next_index
            else:
                self.current_pos += (vec / dist) * distance_to_move
                distance_to_move = 0
        return self.current_pos

    @property
    def full_trajectory(self):
        return self.current_path

    def get_current_position(self) -> np.ndarray:
        return self.current_pos


# 3. 栅格地图 (复用原版)
class GlobalGridMap:
    def __init__(self, world_size_m=20.0, local_map_size_m=10.0, resolution=6):
        self.world_size_m = world_size_m
        self.resolution = resolution
        self.num_cells_world = int(self.world_size_m * self.resolution)
        self.world_origin_offset_m = np.array([self.world_size_m / 2.0, self.world_size_m / 2.0])
        self.grid = np.zeros((self.num_cells_world, self.num_cells_world), dtype=np.float32)
        self.visited_grid = np.zeros((self.num_cells_world, self.num_cells_world), dtype=np.float32)
        self.visited_decay_rate = 0.999
        self.prob_hit = 0.70
        self.prob_miss = 0.45
        self.log_odds_hit = np.log(self.prob_hit / (1 - self.prob_hit))
        self.log_odds_miss = np.log(self.prob_miss / (1 - self.prob_miss))
        self.log_odds_max = 5.0
        self.log_odds_min = -5.0
        self.decay_rate = 0.99
        self.local_map_size_m = local_map_size_m
        self.num_cells_local = int(self.local_map_size_m * self.resolution)
        half_local = self.num_cells_local / 2.0
        local_x, local_y = np.meshgrid(np.arange(self.num_cells_local), np.arange(self.num_cells_local))
        self.local_coords_base = np.stack((local_x.flatten() - half_local, local_y.flatten() - half_local), axis=1)
        self.fig, self.ax = None, None
        self.im = None

    def _log_odds_to_prob(self, log_odds_grid):
        return 1.0 - 1.0 / (1.0 + np.exp(log_odds_grid))

    def reset(self):
        self.grid.fill(0)
        self.visited_grid.fill(0)

    def _world_to_grid_indices(self, pos):
        c = int((pos[0] + self.world_origin_offset_m[0]) * self.resolution)
        r = self.num_cells_world - 1 - int((pos[1] + self.world_origin_offset_m[1]) * self.resolution)
        return r, c

    def update_visited_footprint(self, robot_pos_world):
        self.visited_grid *= self.visited_decay_rate
        r, c = self._world_to_grid_indices(robot_pos_world)
        radius = max(1, int(0.5 * self.resolution))
        r_min, r_max = np.clip(r - radius, 0, self.num_cells_world - 1), np.clip(r + radius + 1, 0, self.num_cells_world)
        c_min, c_max = np.clip(c - radius, 0, self.num_cells_world - 1), np.clip(c + radius + 1, 0, self.num_cells_world)
        self.visited_grid[r_min:r_max, c_min:c_max] = 1.0

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
        hit_valid = hit_points[valid_mask]
        if len(hit_valid) > 0:
            hits_idx = ((hit_valid + self.world_origin_offset_m) * self.resolution).astype(int)
            r_hits = self.num_cells_world - 1 - hits_idx[:, 1]
            c_hits = hits_idx[:, 0]
            v = (r_hits >= 0) & (r_hits < self.num_cells_world) & (c_hits >= 0) & (c_hits < self.num_cells_world)
            np.add.at(self.grid, (r_hits[v], c_hits[v]), self.log_odds_hit)
        np.clip(self.grid, self.log_odds_min, self.log_odds_max, out=self.grid)

    def get_local_maps(self, robot_pos_world, robot_yaw_world):
        c, s = np.cos(-robot_yaw_world), np.sin(-robot_yaw_world)
        rot = np.array([[c, -s], [s, c]])
        world_aligned = self.local_coords_base @ rot.T
        gx = (robot_pos_world[0] + self.world_origin_offset_m[0]) * self.resolution
        gy = self.num_cells_world - 1 - (robot_pos_world[1] + self.world_origin_offset_m[1]) * self.resolution
        sampling_cols = gx + world_aligned[:, 0]
        sampling_rows = gy - world_aligned[:, 1]
        coords = np.stack([sampling_rows, sampling_cols])
        local_log = map_coordinates(self.grid, coords, order=0, cval=0.0, prefilter=False)
        local_prob = self._log_odds_to_prob(local_log).reshape((self.num_cells_local, self.num_cells_local))
        local_vis = map_coordinates(self.visited_grid, coords, order=0, cval=0.0, prefilter=False).reshape((self.num_cells_local, self.num_cells_local))
        return local_prob.copy(), local_vis.copy()

    def generate_goal_map(self, robot_pos_world, robot_yaw_world, goal_pos_world):
        """在机器人局部坐标系下，生成目标方向的高斯热力图"""
        # goal in robot local frame
        dx = goal_pos_world[0] - robot_pos_world[0]
        dy = goal_pos_world[1] - robot_pos_world[1]
        c, s = np.cos(-robot_yaw_world), np.sin(-robot_yaw_world)
        local_gx = c * dx - s * dy
        local_gy = s * dx + c * dy

        # convert to grid coordinates (center of local map = robot position)
        half = self.num_cells_local / 2.0
        goal_col = local_gx * self.resolution + half
        goal_row = -local_gy * self.resolution + half  # y is flipped in grid

        # generate Gaussian heatmap
        y, x = np.ogrid[:self.num_cells_local, :self.num_cells_local]
        dist_sq = (x - goal_col) ** 2 + (y - goal_row) ** 2
        sigma = max(1.0, self.num_cells_local / 8.0)  # adaptive sigma
        goal_map = np.exp(-dist_sq / (2 * sigma ** 2))

        # also mark the exact goal position with a bright spot
        gr, gc = int(np.clip(goal_row, 0, self.num_cells_local - 1)), int(np.clip(goal_col, 0, self.num_cells_local - 1))
        goal_map[gr, gc] = 1.0

        return goal_map.astype(np.float32)

    def sample_local_field(self, world_field, robot_pos_world, robot_yaw_world,
                           fill_value=0.0, normalize_max=None, invert=False):
        c, s = np.cos(-robot_yaw_world), np.sin(-robot_yaw_world)
        rot = np.array([[c, -s], [s, c]])
        world_aligned = self.local_coords_base @ rot.T
        gx = (robot_pos_world[0] + self.world_origin_offset_m[0]) * self.resolution
        gy = self.num_cells_world - 1 - (robot_pos_world[1] + self.world_origin_offset_m[1]) * self.resolution
        sampling_cols = gx + world_aligned[:, 0]
        sampling_rows = gy - world_aligned[:, 1]
        coords = np.stack([sampling_rows, sampling_cols])
        local = map_coordinates(world_field, coords, order=1, cval=fill_value, prefilter=False)
        local = local.reshape((self.num_cells_local, self.num_cells_local)).astype(np.float32)
        if normalize_max is not None:
            local = np.nan_to_num(local, nan=normalize_max, posinf=normalize_max, neginf=normalize_max)
            local = np.clip(local / max(normalize_max, 1e-6), 0.0, 1.0)
        if invert:
            local = 1.0 - local
        return local

    def visualize(self, robot_pos_world, robot_yaw_world, goal_pos_world,
                  obstacle_pos_world=None, debug_target_pos=None, debug_desired_dir=None):
        half_world = self.world_size_m / 2.0
        prob_grid = self._log_odds_to_prob(self.grid)
        if self.fig is None:
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(8, 8))
            self.im = self.ax.imshow(prob_grid, cmap='gray_r', vmin=0, vmax=1,
                                     extent=[-half_world, half_world, -half_world, half_world])
            self.ax.set_title("End-to-End Navigation (No Planner)")
            robot_circle = plt.Circle((0, 0), 0.2, color='blue', zorder=4)
            self.robot_patch = self.ax.add_patch(robot_circle)
            self.goal_patch, = self.ax.plot([], [], '*', color='red', markersize=15, zorder=5)
            self.obstacle_patch = self.ax.add_patch(plt.Circle((0, 0), 0.3, color='orange', zorder=3))
            self.robot_arrow_patch = None
        if not plt.fignum_exists(self.fig.number): return
        try:
            self.im.set_data(prob_grid)
            self.robot_patch.center = (robot_pos_world[0], robot_pos_world[1])
            arrow_dx, arrow_dy = 0.5 * np.cos(robot_yaw_world), 0.5 * np.sin(robot_yaw_world)
            if self.robot_arrow_patch is not None:
                if self.robot_arrow_patch in self.ax.patches:
                    self.robot_arrow_patch.remove()
            self.robot_arrow_patch = plt.Arrow(robot_pos_world[0], robot_pos_world[1], arrow_dx, arrow_dy, width=0.2, color='blue', zorder=4)
            self.ax.add_patch(self.robot_arrow_patch)
            self.goal_patch.set_data([goal_pos_world[0]], [goal_pos_world[1]])
            if obstacle_pos_world is not None:
                self.obstacle_patch.center = (obstacle_pos_world[0], obstacle_pos_world[1])
                self.obstacle_patch.set_visible(True)
            else:
                self.obstacle_patch.set_visible(False)
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
        except Exception as e:
            print(f"Visualize error: {e}")

    def close_visualization(self):
        if self.fig is not None:
            plt.close(self.fig)
            self.fig, self.ax, self.im = None, None, None
            self.robot_patch, self.robot_arrow_patch, self.goal_patch = None, None, None
            self.obstacle_patch = None


# 4. 运动速度平滑器
class VelocitySmoother:
    def __init__(self, dt=0.01, max_accel=[0.5, 0.3, 0.8], max_jerk=[2.0, 1.5, 3.0]):
        self.dt = dt
        self.max_accel = np.array(max_accel, dtype=np.double)
        self.max_jerk = np.array(max_jerk, dtype=np.double)
        self.current_vel = np.zeros(3, dtype=np.double)
        self.current_accel = np.zeros(3, dtype=np.double)

    def update(self, target_vel):
        accel_target = (target_vel - self.current_vel) / self.dt
        jerk = np.clip((accel_target - self.current_accel) / self.dt, -self.max_jerk, self.max_jerk)
        self.current_accel = np.clip(self.current_accel + jerk * self.dt, -self.max_accel, self.max_accel)
        self.current_vel += self.current_accel * self.dt
        return self.current_vel.copy()

    def reset(self):
        self.current_vel.fill(0)
        self.current_accel.fill(0)


# 5. 运动控制器 (复用原版)
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
        self.obs_history = np.zeros(48 * self.obs_history_length, dtype=np.float32)
        self.first_obs = True
        self.count_lowlevel = 0
        self.velocity_smoother = VelocitySmoother(dt=cfg.sim_config.dt * cfg.sim_config.decimation)

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
        smoothed = self.velocity_smoother.update(np.array([cmd_vx, cmd_vy, cmd_dyaw]))
        isaac_joint_pos, isaac_joint_vel, omega, euler_angle = self.get_obs_from_sim()
        new_obs = np.zeros(48, dtype=np.float32)
        new_obs[0:3] = smoothed
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
        si, sn = 0, 0
        if self.first_obs:
            for gs in group_sizes:
                self.obs_history[si:si + gs * self.obs_history_length] = np.tile(new_obs[sn:sn + gs], self.obs_history_length)
                si += gs * self.obs_history_length; sn += gs
            self.first_obs = False
        else:
            for gs in group_sizes:
                sl = self.obs_history[si:si + gs * self.obs_history_length]
                rolled = np.roll(sl, -gs)
                rolled[-gs:] = new_obs[sn:sn + gs]
                self.obs_history[si:si + gs * self.obs_history_length] = rolled
                si += gs * self.obs_history_length; sn += gs

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


# 6. 课程配置
class CurriculumConfig:
    """
    课程学习配置，控制难度等级:
    Level 0: 目标距离 1~3m, 无迷宫墙(remove_wall_prob=1.0)
    Level 1: 目标距离 2~5m, 少量墙(remove_wall_prob=0.7)
    Level 2: 目标距离 3~7m, 中等墙(remove_wall_prob=0.5)
    Level 3: 目标距离 5~10m, 较多墙(remove_wall_prob=0.35)
    Level 4+: 逐步接近完整迷宫，避免一次性引入距离和墙密度双重跳变
    """
    LEVELS = [
        {'min_dist': 1.0, 'max_dist': 3.0, 'remove_wall_prob': 1.0, 'name': 'open_near'},
        {'min_dist': 2.0, 'max_dist': 5.0, 'remove_wall_prob': 0.7,  'name': 'open_medium'},
        {'min_dist': 3.0, 'max_dist': 7.0, 'remove_wall_prob': 0.5,  'name': 'partial_maze'},
        {'min_dist': 5.0, 'max_dist': 10.0, 'remove_wall_prob': 0.35, 'name': 'moderate_maze'},
        {'min_dist': 5.0, 'max_dist': 11.0, 'remove_wall_prob': 0.32, 'name': 'maze_bridge_1'},
        {'min_dist': 5.0, 'max_dist': 12.0, 'remove_wall_prob': 0.30, 'name': 'maze_bridge_2'},
        {'min_dist': 5.0, 'max_dist': 13.0, 'remove_wall_prob': 0.27, 'name': 'maze_bridge_3'},
        {'min_dist': 5.0, 'max_dist': 14.0, 'remove_wall_prob': 0.25, 'name': 'full_maze'},
    ]

    def __init__(self):
        self.current_level = 0
        self.success_buffer = deque(maxlen=300)
        self.promote_threshold = 0.65
        self.demote_threshold = 0.25
        self.min_episodes_for_eval = 100

    @property
    def config(self):
        return self.LEVELS[self.current_level]

    def update(self, success: bool):
        self.success_buffer.append(success)
        if len(self.success_buffer) < self.min_episodes_for_eval:
            return
        rate = sum(self.success_buffer) / len(self.success_buffer)
        if rate >= self.promote_threshold and self.current_level < len(self.LEVELS) - 1:
            self.current_level += 1
            self.success_buffer.clear()
            print(f"\n*** 课程升级 -> Level {self.current_level}: {self.config['name']} ***\n")
        elif rate <= self.demote_threshold and self.current_level > 0:
            self.current_level -= 1
            self.success_buffer.clear()
            print(f"\n*** 课程降级 -> Level {self.current_level}: {self.config['name']} ***\n")


# 7. End-to-End 环境
class RobotVisualEnd2EndEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}

    def __init__(self, model_path, low_level_policy_path, render_mode='rgb_array',
                 render_decimation=5, action_repeat=4, enable_dynamic_obstacles=False,
                 action_mode="waypoint"):
        super().__init__()
        self.render_mode = render_mode
        self.render_decimation = render_decimation
        self.action_repeat = action_repeat
        self.enable_dynamic_obstacles = enable_dynamic_obstacles
        self.action_mode = action_mode

        # MuJoCo 模型
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.model.opt.timestep = 0.001
        self.data = mujoco.MjData(self.model)

        # 栅格地图: 6m 局部窗口，比原 4m 更有上下文，同时避免 40x40 带来的采样开销
        self.grid_map = GlobalGridMap(world_size_m=20.0, local_map_size_m=6.0, resolution=5)
        self.local_cells = self.grid_map.num_cells_local

        # LiDAR 配置
        self.lidar_num_rays = 180
        self.lidar_max_range = 25.0
        self.lidar_fov = np.pi
        self.lidar_angles = np.linspace(-self.lidar_fov / 2, self.lidar_fov / 2, self.lidar_num_rays)

        # 课程学习
        self.curriculum = CurriculumConfig()

        # 低层运动策略
        low_level_policy = torch.jit.load(low_level_policy_path)
        class LocomotionCfg:
            class sim_config: dt = 0.001; decimation = 10
            class robot_config:
                kps = np.array([200, 200, 350, 350, 35, 35] * 2, dtype=np.double)
                kds = np.array([10] * 12, dtype=np.double)
                tau_limit = np.array([240, 240, 240, 240, 40, 40] * 2, dtype=np.double)
                action_scale = 0.25
        self.locomotion_controller = LocomotionController(low_level_policy, self.model, self.data, LocomotionCfg(), parent_env=self)
        # 观测空间: 4通道栅格地图 + 12维状态向量
        # grid: obstacle / visited / local goal heatmap / global geodesic potential
        # state: rel_goal(2) + lin_vel(2) + ang_vel(1) + euclid_dist(1)
        #        + geodesic_dist(1) + heading(2) + prev_action(3)
        self.observation_space = spaces.Dict({
            "grid_map": spaces.Box(
                low=0, high=255,
                shape=(4, self.local_cells, self.local_cells),
                dtype=np.uint8
            ),
            "state": spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(12,),
                dtype=np.float32
            )
        })

        # 动作空间
        # velocity: 直接输出 [vx, vy, yaw_rate]
        # waypoint: 输出 3 个 body-frame waypoint + speed_scale，由中层 tracker 转成速度命令
        if self.action_mode == "velocity":
            action_low = np.array([-0.6, -0.5, -0.85], dtype=np.float32)
            action_high = np.array([0.8, 0.5, 0.85], dtype=np.float32)
        elif self.action_mode == "waypoint":
            action_low = np.array([-3.0, -3.0, -3.0, -3.0, -3.0, -3.0, 0.2], dtype=np.float32)
            action_high = np.array([3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 1.0], dtype=np.float32)
        else:
            raise ValueError(f"Unsupported action_mode: {self.action_mode}")
        self.action_space = spaces.Box(low=action_low, high=action_high, dtype=np.float32)

        # 目标与步数
        self.goal_pos = np.array([5.0, 0.0])
        self.max_episode_steps = 18000
        self.current_step = 0
        self.last_applied_action = np.zeros(3, dtype=np.float32)
        self.last_high_level_action = np.zeros(self.action_space.shape[0], dtype=np.float32)
        self.max_action_rate = np.array([0.25, 0.2, 0.25])
        self.max_waypoint_command_rate = np.array([0.20, 0.16, 0.25], dtype=np.float32)
        self.fall_threshold = 0.6
        self.sensor_dropout = 0.0
        self.action_noise = 0.0
        self.latency_steps = 0
        self.action_latency_buffer = deque()
        self.stuck_counter = 0
        self.stuck_progress_epsilon = 0.005
        self.stuck_penalty_after = 240
        self.stuck_terminate_after = 900
        self.waypoint_tracker = WaypointTracker()
        # Reward 设计: 用 privileged geodesic potential 解决迷宫绕路信用分配
        self.reward_scales = {
            "success": 400.0,               # 从 800 降到 400, 降低"赌博"动机
            "collision_penalty": -300.0,     # 从 -150 提高到 -300, 让碰撞代价更高
            "fall_penalty": -100.0,
            "geodesic_progress": 10.0,  # 沿迷宫最短路势能下降
            "euclidean_progress": 0.5,  # 只保留弱直线距离 shaping
            "heading_to_goal": 0.15,
            "velocity_to_goal": 0.25,
            "obstacle_proximity": 3.0,       # 从 1.0 提高到 3.0, 更强的障碍物回避引导
            "novelty": 0.04,
            "turn_penalty": 0.0001,     # 原地打转惩罚
            "action_rate_penalty": 0.005,
            "unstable_penalty": -0.001,
            "exist_penalty": -0.002,    # 存活惩罚(鼓励快速到达)
            "stuck_penalty": -0.1,
        }
        # 障碍物距离惩罚参数: penalty = scale * exp(-dist / decay)
        self.obstacle_decay = 0.8  # 从 0.3 提高到 0.8, 更早感知障碍物

        # 目标区域
        self.goal_zones = [
            {'x_range': [-9.5, -0.5], 'y_range': [0.5, 9.5]},
            {'x_range': [0.5, 9.5], 'y_range': [0.5, 9.5]},
            {'x_range': [-9.5, -0.5], 'y_range': [-9.5, -0.5]},
            {'x_range': [0.5, 9.5], 'y_range': [-9.5, -0.5]},
            {'x_range': [-9.5, 9.5], 'y_range': [-0.9, 0.9]},
        ]

        # 障碍物排斥核
        s = self.local_cells
        y, x = np.ogrid[:s, :s]
        center = s / 2.0
        dist_m = np.sqrt((x - center) ** 2 + (y - center) ** 2) / self.grid_map.resolution
        sigma = 0.6
        self.repulsion_kernel = np.exp(-(dist_m ** 2) / (2 * sigma ** 2))
        self.repulsion_kernel[dist_m < 0.3] = 0.0

        # MuJoCo 相关 ID
        self.static_walls_info = []
        self.wall_mocap_indices = []
        boundary_names = ["boundary_north", "boundary_south", "boundary_east", "boundary_west"]
        for name in boundary_names:
            i = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if i != -1:
                self.static_walls_info.append({'pos': self.model.geom_pos[i], 'size': self.model.geom_size[i]})
        for i in range(120):
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"gen_wall_body_{i}")
            if body_id != -1:
                self.wall_mocap_indices.append(self.model.body_mocapid[body_id])

        self.maze_gen = MazeGridGenerator(world_size=20.0, grid_dim=8, remove_wall_prob=0.25)

        # 动态障碍物
        self.dyn_obs_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'dynamic_obstacle')
        if self.enable_dynamic_obstacles and self.dyn_obs_body_id != -1:
            self.dyn_obs_mocap_id = self.model.body_mocapid[self.dyn_obs_body_id]
            geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'dynamic_obstacle_geom')
            self.obstacle_controller = DynamicObstacleController(
                static_walls=self.static_walls_info, world_size_m=self.grid_map.world_size_m,
                resolution=self.grid_map.resolution, obstacle_radius=self.model.geom_size[geom_id][0], speed=0.2)
        else:
            self.obstacle_controller = None
            self.dyn_obs_mocap_id = self.model.body_mocapid[self.dyn_obs_body_id] if self.dyn_obs_body_id != -1 else -1

        self.obstacle_geom_ids = set()
        for i in range(self.model.ngeom):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, i)
            if name and ('static_wall' in name or 'dynamic_obstacle_geom' in name or 'gen_wall' in name or 'boundary' in name):
                self.obstacle_geom_ids.add(i)

        self.floor_geom_id = -1
        for i in range(self.model.ngeom):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, i)
            if name and ('floor' in name or 'ground' in name):
                self.floor_geom_id = i; break
        if self.floor_geom_id == -1:
            for i in range(self.model.ngeom):
                if self.model.geom_type[i] == mujoco.mjtGeom.mjGEOM_PLANE:
                    self.floor_geom_id = i; break

        self.robot_base_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'base_link')
        self.depth_camera_name = "d435i_depth"
        self.camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, self.depth_camera_name)
        self.viewer_handle = None
        self.nav_cost_grid = None
        self.goal_distance_field = None
        self.prev_geodesic_dist = None

    # ------------------------------------------------------------------
    # 地图随机化 (支持课程学习)
    # ------------------------------------------------------------------
    def _randomize_map(self):
        if not self.wall_mocap_indices:
            return

        self.static_walls_info = []
        boundary_names = ["boundary_north", "boundary_south", "boundary_east", "boundary_west"]
        for name in boundary_names:
            i = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if i != -1:
                self.static_walls_info.append({'pos': self.model.geom_pos[i].copy(), 'size': self.model.geom_size[i].copy()})

        # 课程学习: 调整 remove_wall_prob
        cfg = self.curriculum.config
        self.maze_gen.remove_wall_prob = cfg['remove_wall_prob']
        new_walls = self.maze_gen.generate(self.np_random)

        if len(new_walls) > len(self.wall_mocap_indices):
            new_walls = new_walls[:len(self.wall_mocap_indices)]

        for i, wall in enumerate(new_walls):
            mocap_idx = self.wall_mocap_indices[i]
            self.data.mocap_pos[mocap_idx] = wall['pos']
            self.data.mocap_quat[mocap_idx] = [0.7071068, 0, 0, 0.7071068] if wall.get('is_vertical', False) else [1, 0, 0, 0]
            self.static_walls_info.append(wall)

        for i in range(len(new_walls), len(self.wall_mocap_indices)):
            self.data.mocap_pos[self.wall_mocap_indices[i]] = [0, 0, -10]

        mujoco.mj_forward(self.model, self.data)

    # ------------------------------------------------------------------
    # 训练专用导航势能场: policy 不直接拿路径，只拿局部 potential 图和 reward shaping
    # ------------------------------------------------------------------
    def _world_to_nav_grid(self, pos):
        c = int((pos[0] + self.grid_map.world_origin_offset_m[0]) * self.grid_map.resolution)
        r = self.grid_map.num_cells_world - 1 - int((pos[1] + self.grid_map.world_origin_offset_m[1]) * self.grid_map.resolution)
        return np.array([
            np.clip(r, 0, self.grid_map.num_cells_world - 1),
            np.clip(c, 0, self.grid_map.num_cells_world - 1),
        ], dtype=np.int32)

    def _build_navigation_cost_grid(self):
        n = self.grid_map.num_cells_world
        wall_map = np.zeros((n, n), dtype=bool)
        for wall in self.static_walls_info:
            pos, size = wall['pos'], wall['size']
            x_min, x_max = pos[0] - size[0], pos[0] + size[0]
            y_min, y_max = pos[1] - size[1], pos[1] + size[1]
            gx_min = int((x_min + self.grid_map.world_origin_offset_m[0]) * self.grid_map.resolution)
            gx_max = int((x_max + self.grid_map.world_origin_offset_m[0]) * self.grid_map.resolution)
            gy_min = n - 1 - int((y_max + self.grid_map.world_origin_offset_m[1]) * self.grid_map.resolution)
            gy_max = n - 1 - int((y_min + self.grid_map.world_origin_offset_m[1]) * self.grid_map.resolution)
            gx_min, gx_max = np.clip([gx_min, gx_max], 0, n - 1)
            gy_min, gy_max = np.clip([gy_min, gy_max], 0, n - 1)
            wall_map[gy_min:gy_max + 1, gx_min:gx_max + 1] = True

        clearance_cells = max(1, math.ceil(0.35 * self.grid_map.resolution))
        inflated = binary_dilation(wall_map, structure=np.ones((3, 3), dtype=bool), iterations=clearance_cells)
        clearance_m = distance_transform_edt(~inflated) / self.grid_map.resolution
        cost_grid = 1.0 + 2.0 * np.exp(-clearance_m / 0.8)
        cost_grid[inflated] = np.inf
        return cost_grid.astype(np.float32)

    def _nearest_free_cell(self, grid_pos, max_radius=12):
        if self.nav_cost_grid is None:
            return None
        r0, c0 = int(grid_pos[0]), int(grid_pos[1])
        if np.isfinite(self.nav_cost_grid[r0, c0]):
            return np.array([r0, c0], dtype=np.int32)
        n = self.nav_cost_grid.shape[0]
        for radius in range(1, max_radius + 1):
            r_min, r_max = max(0, r0 - radius), min(n - 1, r0 + radius)
            c_min, c_max = max(0, c0 - radius), min(n - 1, c0 + radius)
            window = self.nav_cost_grid[r_min:r_max + 1, c_min:c_max + 1]
            valid = np.argwhere(np.isfinite(window))
            if len(valid) > 0:
                dists = np.sum((valid + np.array([r_min, c_min]) - grid_pos) ** 2, axis=1)
                return valid[np.argmin(dists)] + np.array([r_min, c_min])
        return None

    def _compute_goal_distance_field(self):
        self.nav_cost_grid = self._build_navigation_cost_grid()
        n = self.nav_cost_grid.shape[0]
        goal_grid = self._nearest_free_cell(self._world_to_nav_grid(self.goal_pos))
        if goal_grid is None:
            self.goal_distance_field = np.full((n, n), np.inf, dtype=np.float32)
            return

        try:
            mcp = MCP_Geometric(self.nav_cost_grid, fully_connected=True)
            costs, _ = mcp.find_costs(starts=[tuple(goal_grid)])
            self.goal_distance_field = costs.astype(np.float32) / self.grid_map.resolution
            return
        except Exception:
            pass

        dist = np.full((n, n), np.inf, dtype=np.float32)
        moves = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
                 (-1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)),
                 (1, -1, math.sqrt(2)), (1, 1, math.sqrt(2))]
        dist[goal_grid[0], goal_grid[1]] = 0.0
        heap = [(0.0, int(goal_grid[0]), int(goal_grid[1]))]
        while heap:
            cur_dist, r, c = heapq.heappop(heap)
            if cur_dist > dist[r, c]:
                continue
            for dr, dc, step_cells in moves:
                nr, nc = r + dr, c + dc
                if nr < 0 or nr >= n or nc < 0 or nc >= n:
                    continue
                if not np.isfinite(self.nav_cost_grid[nr, nc]):
                    continue
                if dr != 0 and dc != 0:
                    if (not np.isfinite(self.nav_cost_grid[r + dr, c]) or
                            not np.isfinite(self.nav_cost_grid[r, c + dc])):
                        continue
                step_m = step_cells / self.grid_map.resolution
                edge_cost = step_m * 0.5 * (self.nav_cost_grid[r, c] + self.nav_cost_grid[nr, nc])
                new_dist = cur_dist + edge_cost
                if new_dist < dist[nr, nc]:
                    dist[nr, nc] = new_dist
                    heapq.heappush(heap, (float(new_dist), nr, nc))
        self.goal_distance_field = dist

    def _lookup_geodesic_distance(self, pos):
        if self.goal_distance_field is None:
            return np.linalg.norm(self.goal_pos - pos)
        grid_pos = self._world_to_nav_grid(pos)
        value = self.goal_distance_field[grid_pos[0], grid_pos[1]]
        if np.isfinite(value):
            return float(value)
        nearest = self._nearest_free_cell(grid_pos, max_radius=8)
        if nearest is not None:
            value = self.goal_distance_field[nearest[0], nearest[1]]
            if np.isfinite(value):
                return float(value)
        return float(np.linalg.norm(self.goal_pos - pos))

    def _nav_grid_to_world(self, grid_pos):
        y_grid = self.grid_map.num_cells_world - 1 - grid_pos[0]
        x_grid = grid_pos[1]
        return np.array([x_grid, y_grid], dtype=np.float32) / self.grid_map.resolution - self.grid_map.world_origin_offset_m

    def _trace_geodesic_waypoints(self, start_pos, lookahead_distances=(0.8, 1.6, 2.4)):
        if self.goal_distance_field is None or self.nav_cost_grid is None:
            direction = self.goal_pos - start_pos
            norm = np.linalg.norm(direction) + 1e-6
            unit = direction / norm
            return np.array([start_pos + unit * min(d, norm) for d in lookahead_distances], dtype=np.float32)

        current = self._nearest_free_cell(self._world_to_nav_grid(start_pos), max_radius=8)
        if current is None:
            return np.tile(start_pos[None, :], (len(lookahead_distances), 1)).astype(np.float32)

        waypoints = []
        travelled = 0.0
        last_world = self._nav_grid_to_world(current)
        max_steps = int(self.grid_map.world_size_m * self.grid_map.resolution * 2)
        target_idx = 0
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

        for _ in range(max_steps):
            cur_dist = self.goal_distance_field[current[0], current[1]]
            if not np.isfinite(cur_dist) or cur_dist < 0.25:
                break

            best = current
            best_dist = cur_dist
            for dr, dc in moves:
                nr, nc = current[0] + dr, current[1] + dc
                if nr < 0 or nr >= self.grid_map.num_cells_world or nc < 0 or nc >= self.grid_map.num_cells_world:
                    continue
                if not np.isfinite(self.nav_cost_grid[nr, nc]):
                    continue
                cand_dist = self.goal_distance_field[nr, nc]
                if cand_dist < best_dist:
                    best_dist = cand_dist
                    best = np.array([nr, nc], dtype=np.int32)

            if np.array_equal(best, current):
                break

            current = best
            world = self._nav_grid_to_world(current)
            travelled += float(np.linalg.norm(world - last_world))
            last_world = world

            while target_idx < len(lookahead_distances) and travelled >= lookahead_distances[target_idx]:
                waypoints.append(world.copy())
                target_idx += 1

            if target_idx >= len(lookahead_distances):
                break

        while len(waypoints) < len(lookahead_distances):
            waypoints.append(last_world.copy())
        return np.array(waypoints, dtype=np.float32)

    def _world_points_to_body(self, points_world, robot_pos, robot_yaw):
        c, s = np.cos(-robot_yaw), np.sin(-robot_yaw)
        rot = np.array([[c, -s], [s, c]], dtype=np.float32)
        rel = points_world - robot_pos[None, :]
        return rel @ rot.T

    def get_teacher_action(self):
        """Return planner-teacher action in the current action space."""
        robot_pos, robot_yaw = self._get_robot_pose()
        if self.action_mode == "velocity":
            waypoints = self._trace_geodesic_waypoints(robot_pos, (0.9,))
            wp_body = self._world_points_to_body(waypoints, robot_pos, robot_yaw)[0]
            cmd = self._waypoint_to_velocity_command(np.r_[wp_body, 1.0])
            return cmd.astype(np.float32)

        waypoints = self._trace_geodesic_waypoints(robot_pos, (0.8, 1.6, 2.4))
        wp_body = self._world_points_to_body(waypoints, robot_pos, robot_yaw)
        action = np.zeros(7, dtype=np.float32)
        action[:6] = np.clip(wp_body.reshape(-1), -3.0, 3.0)
        action[6] = 1.0
        return np.clip(action, self.action_space.low, self.action_space.high).astype(np.float32)

    def set_dynamic_obstacles(self, enabled):
        self.enable_dynamic_obstacles = bool(enabled)
        if not self.enable_dynamic_obstacles or self.dyn_obs_body_id == -1:
            self.obstacle_controller = None
            if self.dyn_obs_mocap_id != -1:
                self.data.mocap_pos[self.dyn_obs_mocap_id] = [0.0, 0.0, -10.0]
            return
        if self.obstacle_controller is None:
            geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'dynamic_obstacle_geom')
            radius = self.model.geom_size[geom_id][0] if geom_id != -1 else 0.25
            self.obstacle_controller = DynamicObstacleController(
                static_walls=self.static_walls_info,
                world_size_m=self.grid_map.world_size_m,
                resolution=self.grid_map.resolution,
                obstacle_radius=radius,
                speed=0.25,
            )
        self.obstacle_controller.update_map(self.static_walls_info)

    def _estimate_local_clearance(self):
        robot_pos, robot_yaw = self._get_robot_pose()
        local_map, _ = self.grid_map.get_local_maps(robot_pos, robot_yaw)
        obstacle_mask = local_map > 0.5
        if not np.any(obstacle_mask):
            return 2.0
        center = self.local_cells // 2
        obs_indices = np.argwhere(obstacle_mask)
        dists = np.sqrt(np.sum((obs_indices - [center, center]) ** 2, axis=1))
        return float(np.min(dists) / self.grid_map.resolution)

    def _waypoint_to_velocity_command(self, action):
        action = np.asarray(action, dtype=np.float32)
        clearance = self._estimate_local_clearance()
        level = int(getattr(self.curriculum, "current_level", 0))
        wp_x = float(np.clip(action[0], -3.0, 3.0))
        wp_y = float(np.clip(action[1], -3.0, 3.0))
        speed_scale = float(np.clip(action[6] if len(action) > 6 else 1.0, 0.2, 1.0))
        if level >= 2:
            clearance_scale = float(np.clip((clearance - 0.20) / 0.75, 0.25, 1.0))
            speed_scale *= clearance_scale
        heading = math.atan2(wp_y, max(0.2, wp_x))
        dist = math.sqrt(wp_x * wp_x + wp_y * wp_y)

        cmd_vx = np.clip(0.55 * wp_x, -0.35, 0.8) * speed_scale
        cmd_vy = np.clip(0.45 * wp_y, -0.45, 0.45) * speed_scale
        cmd_yaw = np.clip(1.2 * heading, -0.85, 0.85)

        if dist < 0.35:
            cmd_vx *= 0.4
            cmd_vy *= 0.4
        base = np.array([cmd_vx, cmd_vy, cmd_yaw], dtype=np.float32)

        tracked = self.waypoint_tracker.mpc(action, clearance=clearance, prev_cmd=self.last_applied_action)
        command = 0.55 * tracked + 0.45 * base
        low = np.array([-0.35, -0.35, -0.75], dtype=np.float32)
        high = np.array([0.65, 0.35, 0.75], dtype=np.float32)
        if level >= 2:
            high = np.array([0.48, 0.28, 0.70], dtype=np.float32)
            low = np.array([-0.28, -0.28, -0.70], dtype=np.float32)
        if clearance < 0.65:
            scale = float(np.clip((clearance - 0.15) / 0.50, 0.20, 1.0))
            high[:2] *= scale
            low[:2] *= scale
        if clearance < 0.40:
            command[:2] *= 0.45
        return np.clip(command, low, high).astype(np.float32)

    def _convert_high_level_action(self, action):
        if self.action_mode == "velocity":
            return np.asarray(action, dtype=np.float32)
        return self._waypoint_to_velocity_command(action)

    # ------------------------------------------------------------------
    # 碰撞检测
    # ------------------------------------------------------------------
    def _check_collision(self):
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            if (contact.geom1 == self.floor_geom_id) or (contact.geom2 == self.floor_geom_id):
                continue
            if contact.geom1 in self.obstacle_geom_ids or contact.geom2 in self.obstacle_geom_ids:
                return True
        return False

    # ------------------------------------------------------------------
    # 目标重置 (支持课程学习: 控制目标距离)
    # ------------------------------------------------------------------
    def _reset_goal(self):
        cfg = self.curriculum.config
        min_dist = cfg['min_dist']
        max_dist = cfg['max_dist']
        robot_pos = self.data.xpos[self.robot_base_body_id][:2]

        for _ in range(100):
            angle = self.np_random.uniform(-math.pi, math.pi)
            dist = self.np_random.uniform(min_dist, max_dist)
            candidate = robot_pos + np.array([dist * math.cos(angle), dist * math.sin(angle)])

            # 检查边界
            half = self.grid_map.world_size_m / 2.0 - 0.5
            if abs(candidate[0]) > half or abs(candidate[1]) > half:
                continue

            # 检查是否在墙内
            valid = True
            for wall in self.static_walls_info:
                p, s = wall['pos'], wall['size']
                if abs(candidate[0] - p[0]) < s[0] + 0.3 and abs(candidate[1] - p[1]) < s[1] + 0.3:
                    valid = False
                    break
            if valid:
                self.goal_pos = candidate
                break

        target_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, 'target_goal')
        if target_site_id != -1:
            self.model.site_pos[target_site_id][:2] = self.goal_pos

    # ------------------------------------------------------------------
    # 生成位置有效性检查
    # ------------------------------------------------------------------
    def _is_valid_spawn_pos(self, pos, min_dist_to_wall=0.8):
        half = self.grid_map.world_size_m / 2.0 - 0.5
        if abs(pos[0]) > half or abs(pos[1]) > half:
            return False
        for wall in self.static_walls_info:
            w_pos, w_size = wall['pos'], wall['size']
            if abs(pos[0] - w_pos[0]) < (w_size[0] + min_dist_to_wall) and abs(pos[1] - w_pos[1]) < (w_size[1] + min_dist_to_wall):
                return False
        if self.obstacle_controller is not None:
            if np.linalg.norm(pos - self.obstacle_controller.get_current_position()) < 2.0:
                return False
        return True

    # ------------------------------------------------------------------
    # 机器人姿态
    # ------------------------------------------------------------------
    def _get_robot_pose(self):
        pos_2d = self.data.xpos[self.robot_base_body_id][:2].copy()
        mat = self.data.xmat[self.robot_base_body_id].reshape(3, 3)
        yaw = math.atan2(mat[1, 0], mat[0, 0])
        return pos_2d, yaw

    # ------------------------------------------------------------------
    # LiDAR 感知
    # ------------------------------------------------------------------
    def _sense_lidar(self):
        if self.camera_id != -1:
            origin = self.data.cam_xpos[self.camera_id].copy()
        else:
            origin = self.data.xpos[self.robot_base_body_id].copy()
            origin[2] += 0.2

        body_mat = self.data.xmat[self.robot_base_body_id].reshape(3, 3)
        local_rays = np.stack([np.cos(self.lidar_angles), np.sin(self.lidar_angles), np.zeros_like(self.lidar_angles)], axis=0)
        global_rays = body_mat @ local_rays

        hit_points, valid_mask = [], []
        geomgroup = np.array([1, 1, 1, 1, 1, 1], dtype=np.uint8)
        body_exclude = int(self.robot_base_body_id)
        geomid_out = np.zeros(1, dtype=np.int32)

        for i in range(self.lidar_num_rays):
            vec = np.ascontiguousarray(global_rays[:, i], dtype=np.float64)
            dist = mujoco.mj_ray(self.model, self.data, origin, vec, geomgroup, 1, body_exclude, geomid_out)
            if dist != -1 and dist < self.lidar_max_range:
                hit_points.append((origin + vec * dist)[:2])
                valid_mask.append(True)
            else:
                hit_points.append((origin + vec * self.lidar_max_range)[:2])
                valid_mask.append(False)

        valid_mask = np.array(valid_mask)
        if self.sensor_dropout > 0.0:
            keep = self.np_random.random(len(valid_mask)) > self.sensor_dropout
            valid_mask = valid_mask & keep
        return np.array(hit_points), valid_mask

    # ------------------------------------------------------------------
    # 感知更新
    # ------------------------------------------------------------------
    def _update_perception(self):
        hit_points, valid_mask = self._sense_lidar()
        lidar_pos = self.data.xpos[self.robot_base_body_id][:2]
        self.grid_map.update_from_lidar(lidar_pos, hit_points, valid_mask)

        if self.render_mode == 'human':
            import matplotlib
            if matplotlib.get_backend() != 'Agg':
                pos, yaw = self._get_robot_pose()
                obs_pos = self.data.mocap_pos[self.dyn_obs_mocap_id] if self.obstacle_controller is not None and self.dyn_obs_mocap_id != -1 else None
                self.grid_map.visualize(
                    robot_pos_world=pos, robot_yaw_world=yaw,
                    goal_pos_world=self.goal_pos, obstacle_pos_world=obs_pos)

    # ------------------------------------------------------------------
    # 观测 (4通道地图 + 12维状态向量)
    # ------------------------------------------------------------------
    def _get_obs(self):
        robot_pos, robot_yaw = self._get_robot_pose()

        local_prob, local_visited = self.grid_map.get_local_maps(robot_pos, robot_yaw)
        goal_map = self.grid_map.generate_goal_map(robot_pos, robot_yaw, self.goal_pos)
        if self.goal_distance_field is None:
            potential_map = np.zeros_like(goal_map, dtype=np.float32)
        else:
            potential_map = self.grid_map.sample_local_field(
                self.goal_distance_field, robot_pos, robot_yaw,
                fill_value=self.grid_map.world_size_m,
                normalize_max=self.grid_map.world_size_m * 2.0,
                invert=True)

        grid_4ch = np.stack([
            local_prob,           # 障碍物
            local_visited,        # 已访问
            goal_map,             # 目标方向
            potential_map,        # 全局最短路势能
        ], axis=0)
        grid_uint8 = (grid_4ch * 255.0).clip(0, 255).astype(np.uint8)

        # 12维状态向量 (全在机器人本体坐标系下)
        robot_quat = self.data.qpos[3:7]
        r = R.from_quat([robot_quat[1], robot_quat[2], robot_quat[3], robot_quat[0]])
        world_goal_vec = self.goal_pos - robot_pos
        rel_goal = r.apply(np.array([world_goal_vec[0], world_goal_vec[1], 0]), inverse=True)[:2]

        world_lin_vel = self.data.qvel[0:3]
        robot_lin_vel = r.apply(world_lin_vel, inverse=True)
        world_ang_vel = self.data.qvel[3:6]

        # 目标距离 (标量进度信号)
        goal_distance = np.linalg.norm(world_goal_vec)
        geodesic_distance = self._lookup_geodesic_distance(robot_pos)
        # 目标朝向 (cos, sin, 比rel_goal更平滑)
        heading_to_goal = math.atan2(world_goal_vec[1], world_goal_vec[0]) - robot_yaw
        heading_to_goal = (heading_to_goal + np.pi) % (2 * np.pi) - np.pi  # normalize to [-pi, pi]
        heading_cos_sin = np.array([np.cos(heading_to_goal), np.sin(heading_to_goal)], dtype=np.float32)
        # 上一步动作 (动量信息)
        prev_action = self.last_applied_action.copy()

        state = np.concatenate([
            rel_goal,                     # 2D: 相对目标位置 (body frame)
            robot_lin_vel[:2],            # 2D: 线速度 (body frame)
            [world_ang_vel[2]],           # 1D: 角速度 (yaw)
            [goal_distance],              # 1D: 到目标距离
            [geodesic_distance],           # 1D: 迷宫最短路距离
            heading_cos_sin,              # 2D: 目标朝向 cos/sin
            prev_action,                  # 3D: 上一步动作
        ]).astype(np.float32)

        return {"grid_map": grid_uint8, "state": state}

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.grid_map.reset()
        self.locomotion_controller.reset()
        self.current_step = 0
        self.last_applied_action.fill(0)
        self.last_high_level_action.fill(0)
        self.action_latency_buffer.clear()
        self.stuck_counter = 0

        self._randomize_map()

        if self.obstacle_controller is not None:
            self.obstacle_controller.update_map(self.static_walls_info)
            pos2d = self.obstacle_controller.get_current_position()
            self.data.mocap_pos[self.dyn_obs_mocap_id] = [pos2d[0], pos2d[1], 0.5]
        elif self.dyn_obs_mocap_id != -1:
            self.data.mocap_pos[self.dyn_obs_mocap_id] = [0.0, 0.0, -10.0]

        # 先生成随机机器人位置, 再基于课程设置目标
        robot_spawn_pos = None
        for _ in range(500):
            x = self.np_random.uniform(-9.0, 9.0)
            y = self.np_random.uniform(-9.0, 9.0)
            if self._is_valid_spawn_pos(np.array([x, y])):
                robot_spawn_pos = np.array([x, y])
                robot_spawn_yaw = self.np_random.uniform(-math.pi, math.pi)
                break

        if robot_spawn_pos is None:
            return self.reset(seed=seed, options=options)

        self.data.qpos[0] = robot_spawn_pos[0]
        self.data.qpos[1] = robot_spawn_pos[1]
        self.data.qpos[2] = 0.05
        quat_xyzw = R.from_euler('z', robot_spawn_yaw).as_quat()
        self.data.qpos[3:7] = [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]

        mujoco.mj_forward(self.model, self.data)
        try:
            for _ in range(20):
                self.data.ctrl[:] = 0
                mujoco.mj_step(self.model, self.data)
        except Exception:
            return self.reset(seed=seed, options=options)

        if self._check_collision():
            return self.reset(seed=seed, options=options)
        robot_quat = self.data.qpos[3:7]
        r = R.from_quat([robot_quat[1], robot_quat[2], robot_quat[3], robot_quat[0]])
        if r.apply([0, 0, 1])[2] < 0.5:
            return self.reset(seed=seed, options=options)

        # 基于课程设置目标距离
        self._reset_goal()
        self._compute_goal_distance_field()

        self.dist_to_goal_start = np.linalg.norm(self.data.qpos[:2] - self.goal_pos)
        if self.dist_to_goal_start < 0.1:
            self.dist_to_goal_start = 0.1
        self.prev_dist_to_goal = self.dist_to_goal_start
        self.prev_geodesic_dist = self._lookup_geodesic_distance(self.data.qpos[:2])

        self._update_perception()
        return self._get_obs(), {}

    # ------------------------------------------------------------------
    # 动态障碍物更新
    # ------------------------------------------------------------------
    def _update_mocap_obstacle(self, dt):
        if self.obstacle_controller is None: return
        new_pos = self.obstacle_controller.update(dt)
        self.data.mocap_pos[self.dyn_obs_mocap_id] = [new_pos[0], new_pos[1], 0.5]

    # ------------------------------------------------------------------
    # Step (核心: 新 reward 设计)
    # ------------------------------------------------------------------
    def step(self, action):
        high_action = np.clip(np.asarray(action, dtype=np.float32), self.action_space.low, self.action_space.high)
        if self.action_noise > 0.0:
            high_action = high_action + self.np_random.normal(0.0, self.action_noise, size=high_action.shape).astype(np.float32)
            high_action = np.clip(high_action, self.action_space.low, self.action_space.high)
        if self.latency_steps > 0:
            self.action_latency_buffer.append(high_action.copy())
            if len(self.action_latency_buffer) <= self.latency_steps:
                high_action = self.last_high_level_action.copy()
            else:
                high_action = self.action_latency_buffer.popleft()
        self.last_high_level_action = high_action.copy()
        target_command = self._convert_high_level_action(high_action)

        prev_action = self.last_applied_action.copy()
        clipped_delta = np.clip(target_command - prev_action, -self.max_waypoint_command_rate, self.max_waypoint_command_rate)
        smoothed_action = prev_action + clipped_delta
        self.last_applied_action = smoothed_action

        total_reward = 0.0
        terminated = False
        truncated = False

        for _ in range(self.action_repeat):
            sub_step_dt = self.locomotion_controller.cfg.sim_config.decimation * self.model.opt.timestep
            self._update_mocap_obstacle(sub_step_dt)
            self.locomotion_controller.step(*smoothed_action)

            pos_after = self.data.xpos[self.robot_base_body_id][:2]
            visited_r, visited_c = self.grid_map._world_to_grid_indices(pos_after)
            visited_r = np.clip(visited_r, 0, self.grid_map.num_cells_world - 1)
            visited_c = np.clip(visited_c, 0, self.grid_map.num_cells_world - 1)
            visited_before = self.grid_map.visited_grid[visited_r, visited_c]
            self.grid_map.update_visited_footprint(pos_after)

            # 基础信息
            mat = self.data.xmat[self.robot_base_body_id].reshape(3, 3)
            robot_yaw = math.atan2(mat[1, 0], mat[0, 0])
            forward_dir = np.array([np.cos(robot_yaw), np.sin(robot_yaw)])
            lin_vel_world = self.data.qvel[:2]

            # --- 目标方向 ---
            to_goal_vec = self.goal_pos - pos_after
            dist_to_goal = np.linalg.norm(to_goal_vec)
            to_goal_dir = to_goal_vec / (dist_to_goal + 1e-6)

            # --- Reward 计算 ---

            # 1. 进度奖励: 距离减少量
            euclidean_progress = self.prev_dist_to_goal - dist_to_goal
            reward_euclidean_progress = euclidean_progress * self.reward_scales["euclidean_progress"]
            self.prev_dist_to_goal = dist_to_goal

            geodesic_dist = self._lookup_geodesic_distance(pos_after)
            geodesic_progress = self.prev_geodesic_dist - geodesic_dist
            reward_geodesic_progress = geodesic_progress * self.reward_scales["geodesic_progress"]
            self.prev_geodesic_dist = geodesic_dist
            if geodesic_progress > self.stuck_progress_epsilon:
                self.stuck_counter = 0
            else:
                self.stuck_counter += 1

            # 2. 朝向目标奖励
            heading_alignment = np.dot(forward_dir, to_goal_dir)
            reward_heading = heading_alignment * self.reward_scales["heading_to_goal"]

            # 3. 速度向目标投影奖励
            vel_projected = np.dot(lin_vel_world, to_goal_dir)
            if heading_alignment < 0 and vel_projected > 0:
                valid_velocity = vel_projected * 0.1
            else:
                valid_velocity = vel_projected
            reward_velocity = np.clip(valid_velocity, -0.5, 1.0) * self.reward_scales["velocity_to_goal"]

            # 4. 障碍物距离惩罚 (连续指数衰减, 类似高程图代价)
            local_map, _ = self.grid_map.get_local_maps(pos_after, robot_yaw)
            obstacle_mask = local_map > 0.6
            reward_obstacle_proximity = 0.0
            min_obs_dist = np.inf
            obs_indices = np.argwhere(obstacle_mask)
            if len(obs_indices) > 0:
                center = self.local_cells // 2
                dists = np.sqrt(np.sum((obs_indices - [center, center]) ** 2, axis=1))
                min_obs_dist = np.min(dists) / self.grid_map.resolution
                # 指数衰减: 距离0时惩罚最大, 距离>1m时趋近于0
                reward_obstacle_proximity = -self.reward_scales["obstacle_proximity"] * np.exp(-min_obs_dist / self.obstacle_decay)

            reward_novelty = self.reward_scales["novelty"] * (1.0 - visited_before)

            # 6. 打转惩罚
            ang_vel_z = self.data.qvel[5]
            penalty_turn = -self.reward_scales["turn_penalty"] * (abs(ang_vel_z) + 0.1 * ang_vel_z ** 2)

            # 7. 不稳定惩罚
            up_alignment = mat[:, 2][2]
            penalty_unstable = self.reward_scales["unstable_penalty"] if up_alignment < 0.8 else 0.0

            # 8. 动作变化率惩罚
            penalty_act_rate = -self.reward_scales["action_rate_penalty"] * np.linalg.norm(smoothed_action - prev_action)

            # 9. 存活惩罚
            penalty_exist = self.reward_scales["exist_penalty"]
            penalty_stuck = self.reward_scales["stuck_penalty"] if self.stuck_counter > self.stuck_penalty_after else 0.0

            sub_reward = (reward_geodesic_progress + reward_euclidean_progress +
                          reward_heading + reward_velocity +
                          reward_obstacle_proximity + reward_novelty + penalty_turn +
                          penalty_unstable + penalty_act_rate + penalty_exist + penalty_stuck)

            # --- 终止条件 ---
            succeeded = dist_to_goal < 0.6
            collision = self._check_collision()
            fell = up_alignment < self.fall_threshold
            stuck = self.stuck_counter > self.stuck_terminate_after
            self.current_step += 1
            truncated = self.current_step >= self.max_episode_steps

            term_reward = 0.0
            termination_reason = "running"

            if succeeded:
                term_reward = self.reward_scales["success"]
                terminated = True
                termination_reason = "success"
            elif collision:
                term_reward = self.reward_scales["collision_penalty"]
                terminated = True
                termination_reason = "collision"
            elif fell:
                term_reward = self.reward_scales["fall_penalty"]
                terminated = True
                termination_reason = "fall"
            elif stuck:
                terminated = True
                termination_reason = "stuck"
            elif truncated:
                terminated = True
                termination_reason = "timeout"

            total_reward += sub_reward + term_reward
            if terminated or truncated:
                break

        self._update_perception()

        # 课程更新
        if terminated or truncated:
            self.curriculum.update(termination_reason == "success")

        info_dict = {
            "is_success": termination_reason == "success",
            "distance_to_goal": dist_to_goal,
            "termination_reason": termination_reason,
            "robot_pos": pos_after.copy(),
            "curriculum_level": self.curriculum.current_level,
            "teacher_action": self.get_teacher_action(),
            "applied_command": smoothed_action.copy(),
            "rewards": {
                "geodesic_progress": reward_geodesic_progress,
                "euclidean_progress": reward_euclidean_progress,
                "heading": reward_heading,
                "velocity": reward_velocity, "obs_proximity": reward_obstacle_proximity,
                "novelty": reward_novelty,
                "min_obs_dist": min_obs_dist, "dist": dist_to_goal,
                "geodesic_dist": geodesic_dist
            }
        }

        return self._get_obs(), total_reward, terminated, truncated, info_dict

    def render(self): pass

    def close(self):
        if self.viewer_handle:
            if self.viewer_handle.is_running(): self.viewer_handle.close()
            self.viewer_handle = None
        self.grid_map.close_visualization()
