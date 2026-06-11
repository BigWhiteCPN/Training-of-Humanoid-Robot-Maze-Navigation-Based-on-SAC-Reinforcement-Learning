import random
from collections import deque, defaultdict
import numpy as np
import torch


class EpisodeReplayBuffer:
    def __init__(
        self,
        capacity=1_000_000,
        gamma=0.995,
        horizon_steps=50,
        her_samples_per_episode=8,
        demo_capacity=300_000,
        demo_sample_fraction=0.35,
        current_level_sample_fraction=0.35,
        adjacent_level_sample_fraction=0.20,
        rebuild_interval=100_000,
    ):
        self.capacity = capacity
        self.demo_capacity = demo_capacity
        self.demo_sample_fraction = demo_sample_fraction
        self.current_level_sample_fraction = current_level_sample_fraction
        self.adjacent_level_sample_fraction = adjacent_level_sample_fraction
        self.gamma = gamma
        self.horizon_steps = horizon_steps
        self.her_samples_per_episode = her_samples_per_episode
        self.transitions = deque(maxlen=capacity)
        self.demo_transitions = deque(maxlen=demo_capacity)
        self.episodes = deque(maxlen=max(100, capacity // 200))
        self.by_level = defaultdict(list)
        self.buckets = defaultdict(list)
        self._added_since_rebuild = 0
        self._rebuild_interval = rebuild_interval

    def add_episode(self, episode):
        if not episode:
            return
        self._assign_horizon_labels(episode)
        is_demo = any(t.get("teacher", False) for t in episode)
        if not is_demo:
            self.episodes.append(episode)
        for t in episode:
            self._append_transition(t, is_demo=is_demo)

    def _append_transition(self, t, is_demo=False):
        if is_demo:
            self.demo_transitions.append(t)
        else:
            self.transitions.append(t)
        self._added_since_rebuild += 1
        if self._added_since_rebuild >= self._rebuild_interval:
            self._rebuild_index()
        else:
            self.by_level[int(t.get("level", 0))].append(t)
            for bucket in self._transition_buckets(t):
                self.buckets[bucket].append(t)

    def _rebuild_index(self):
        self.by_level = defaultdict(list)
        self.buckets = defaultdict(list)
        for t in list(self.demo_transitions) + list(self.transitions):
            self.by_level[int(t.get("level", 0))].append(t)
            for bucket in self._transition_buckets(t):
                self.buckets[bucket].append(t)
        self._added_since_rebuild = 0

    def _transition_buckets(self, t):
        buckets = ["uniform"]
        if t.get("is_success", False):
            buckets.append("success")
        if t.get("near_success", False):
            buckets.append("near_success")
        if t.get("collision_label", 0.0) > 0.5:
            buckets.append("collision")
        if t.get("stuck", False):
            buckets.append("stuck")
        if t.get("teacher", False):
            buckets.append("teacher")
        if t.get("teacher_success_episode", False):
            buckets.append("teacher_success")
        if t.get("her_relabel", False):
            buckets.append("her")
        return buckets

    def _assign_horizon_labels(self, episode):
        n = len(episode)
        for i, t in enumerate(episode):
            j = min(n - 1, i + self.horizon_steps)
            future = episode[i:j + 1]
            t["collision_label"] = float(any(x.get("termination_reason") == "collision" for x in future))
            t["stuck_label"] = float(any(x.get("termination_reason") == "stuck" for x in future))
            geo_i = float(t.get("geodesic_dist", 0.0))
            geo_j = float(episode[j].get("geodesic_dist", geo_i))
            t["progress_label"] = geo_i - geo_j
            if "start_geodesic_dist" in t and t["start_geodesic_dist"] > 1e-6:
                t["progress_ratio"] = (t["start_geodesic_dist"] - geo_j) / t["start_geodesic_dist"]

    def __len__(self):
        return len(self.demo_transitions) + len(self.transitions)

    def stats(self):
        return {
            "total": len(self),
            "demo": len(self.demo_transitions),
            "online": len(self.transitions),
            "capacity": self.demo_capacity + self.capacity,
        }

    def _sample_level(self, level):
        pool = self.by_level.get(int(level), [])
        return random.choice(pool) if pool else None

    def _sample_adjacent_level(self, current_level):
        levels = [int(current_level) - 1, int(current_level) + 1]
        pools = []
        for level in levels:
            if self.by_level.get(level):
                pools.extend(self.by_level[level])
        return random.choice(pools) if pools else None

    def sample_items(self, batch_size, current_level=None):
        if len(self) == 0:
            raise RuntimeError("Replay buffer is empty")
        items = []
        levels = sorted(self.by_level.keys())
        hard = max(levels) if levels else 0
        for _ in range(batch_size):
            r = random.random()
            if r < self.demo_sample_fraction and self.demo_transitions:
                if self.buckets.get("teacher_success") and random.random() < 0.8:
                    items.append(random.choice(self.buckets["teacher_success"]))
                else:
                    items.append(random.choice(self.demo_transitions))
                continue

            r = random.random()
            if current_level is not None and r < self.current_level_sample_fraction:
                item = self._sample_level(current_level)
                items.append(item if item is not None else random.choice(self.transitions or self.demo_transitions))
            elif current_level is not None and r < self.current_level_sample_fraction + self.adjacent_level_sample_fraction:
                item = self._sample_adjacent_level(current_level)
                items.append(item if item is not None else random.choice(self.transitions or self.demo_transitions))
            elif r < 0.25 and self.by_level.get(hard):
                items.append(random.choice(self.by_level[hard]))
            elif r < 0.45 and self.by_level.get(max(0, hard - 1)):
                items.append(random.choice(self.by_level[max(0, hard - 1)]))
            elif r < 0.58 and (self.buckets.get("success") or self.buckets.get("near_success")):
                pool = self.buckets.get("success", []) + self.buckets.get("near_success", [])
                items.append(random.choice(pool))
            elif r < 0.66 and self.buckets.get("teacher"):
                items.append(random.choice(self.buckets["teacher"]))
            elif r < 0.95 and self.buckets.get("collision"):
                items.append(random.choice(self.buckets["collision"]))
            elif r < 0.98 and self.buckets.get("stuck"):
                items.append(random.choice(self.buckets["stuck"]))
            else:
                pool = self.transitions if self.transitions else self.demo_transitions
                items.append(random.choice(pool))
        return items

    def sample(self, batch_size, device, current_level=None):
        items = self.sample_items(batch_size, current_level=current_level)
        batch = {}
        for key in ["grid_map", "state_history", "graph_summary", "action", "next_grid_map", "next_state_history", "next_graph_summary"]:
            arr = np.asarray([t[key] for t in items])
            batch[key] = torch.as_tensor(arr, dtype=torch.float32 if "grid" not in key else torch.uint8, device=device)
        for key in ["reward", "done", "collision_label", "progress_label", "teacher_action"]:
            arr = np.asarray([t.get(key, 0.0) for t in items], dtype=np.float32)
            batch[key] = torch.as_tensor(arr, dtype=torch.float32, device=device)
        if batch["teacher_action"].ndim == 1:
            batch["teacher_action"] = batch["teacher_action"].view(batch_size, -1)
        return batch

    def add_hindsight_goals(self, episode):
        """Relabel a subset of transitions to future achieved goals.

        This implementation rewrites the local goal heatmap and the last state in
        the history using saved robot pose/yaw/future achieved goal. It cannot
        reconstruct a past global SLAM map exactly, so it preserves obstacle and
        visited channels and uses Euclidean potential for the relabeled local goal.
        """
        if not episode:
            return []
        relabeled = []
        indices = np.linspace(0, max(0, len(episode) - 2), min(self.her_samples_per_episode, len(episode)), dtype=int)
        for i in indices:
            t = episode[int(i)]
            future = random.choice(episode[int(i):])
            new_t = dict(t)
            goal = np.asarray(future.get("achieved_goal", future.get("robot_pos", [0.0, 0.0])), dtype=np.float32)
            new_t["grid_map"] = self._relabel_grid(t["grid_map"], t, goal)
            new_t["state_history"] = self._relabel_state_history(t["state_history"], t, goal)
            new_t["next_grid_map"] = self._relabel_grid(t["next_grid_map"], future, goal)
            new_t["next_state_history"] = self._relabel_state_history(t["next_state_history"], future, goal)
            new_t["near_success"] = True
            dist = float(np.linalg.norm(np.asarray(t.get("robot_pos", goal), dtype=np.float32) - goal))
            next_dist = float(np.linalg.norm(np.asarray(future.get("robot_pos", goal), dtype=np.float32) - goal))
            new_t["geodesic_dist"] = dist
            new_t["start_geodesic_dist"] = max(dist, 1e-6)
            new_t["progress_label"] = dist - next_dist
            new_t["progress_ratio"] = (dist - next_dist) / max(dist, 1e-6)
            new_t["reward"] = 800.0 if next_dist < 0.6 else max(float(new_t.get("reward", 0.0)), 5.0 * (dist - next_dist))
            new_t["done"] = float(next_dist < 0.6)
            new_t["her_relabel"] = True
            relabeled.append(new_t)
        return relabeled

    def _relabel_grid(self, grid, transition, goal):
        grid = np.asarray(grid).copy()
        if grid.ndim != 3 or grid.shape[0] < 4:
            return grid
        robot_pos = np.asarray(transition.get("robot_pos", [0.0, 0.0]), dtype=np.float32)
        yaw = float(transition.get("robot_yaw", 0.0))
        rel = goal - robot_pos
        c, s = np.cos(-yaw), np.sin(-yaw)
        local_x = c * rel[0] - s * rel[1]
        local_y = s * rel[0] + c * rel[1]
        h, w = grid.shape[1], grid.shape[2]
        resolution = w / 6.0
        col = local_x * resolution + w / 2.0
        row = -local_y * resolution + h / 2.0
        yy, xx = np.ogrid[:h, :w]
        sigma = max(1.0, w / 8.0)
        goal_map = np.exp(-((xx - col) ** 2 + (yy - row) ** 2) / (2 * sigma ** 2))
        potential = 1.0 - np.clip(np.sqrt((xx - col) ** 2 + (yy - row) ** 2) / max(w, 1), 0.0, 1.0)
        grid[2] = (goal_map * 255.0).clip(0, 255).astype(np.uint8)
        grid[3] = (potential * 255.0).clip(0, 255).astype(np.uint8)
        return grid

    def _relabel_state_history(self, state_history, transition, goal):
        hist = np.asarray(state_history, dtype=np.float32).copy()
        if hist.shape[0] < 12:
            return hist
        robot_pos = np.asarray(transition.get("robot_pos", [0.0, 0.0]), dtype=np.float32)
        yaw = float(transition.get("robot_yaw", 0.0))
        rel = goal - robot_pos
        c, s = np.cos(-yaw), np.sin(-yaw)
        local_x = c * rel[0] - s * rel[1]
        local_y = s * rel[0] + c * rel[1]
        dist = float(np.linalg.norm(rel))
        heading = np.arctan2(rel[1], rel[0]) - yaw
        heading = (heading + np.pi) % (2 * np.pi) - np.pi
        last = hist[-12:].copy()
        last[0:2] = [local_x, local_y]
        last[5] = dist
        last[6] = dist
        last[7:9] = [np.cos(heading), np.sin(heading)]
        hist[-12:] = last
        return hist
