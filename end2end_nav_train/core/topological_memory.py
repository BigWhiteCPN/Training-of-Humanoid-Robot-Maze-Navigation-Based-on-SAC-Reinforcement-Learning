from dataclasses import dataclass
import numpy as np


@dataclass
class TopoNode:
    pos: np.ndarray
    visited_count: int = 1
    dead_end_score: float = 0.0
    frontier_score: float = 0.0


class TopologicalMemory:
    """Lightweight online topological summary.

    This is intentionally small and deterministic so it can run in both train and play.
    It stores coarse visited nodes and exports a fixed-size summary vector.
    """

    def __init__(self, node_spacing=0.8):
        self.node_spacing = node_spacing
        self.nodes = []
        self.last_node_idx = None

    def reset(self):
        self.nodes = []
        self.last_node_idx = None

    def update(self, robot_pos, geodesic_dist=None, local_visited_mean=0.0):
        robot_pos = np.asarray(robot_pos, dtype=np.float32)
        if not self.nodes:
            self.nodes.append(TopoNode(robot_pos.copy()))
            self.last_node_idx = 0
            return

        dists = np.array([np.linalg.norm(n.pos - robot_pos) for n in self.nodes])
        nearest = int(np.argmin(dists))
        if dists[nearest] > self.node_spacing:
            self.nodes.append(TopoNode(robot_pos.copy()))
            self.last_node_idx = len(self.nodes) - 1
        else:
            self.nodes[nearest].visited_count += 1
            if local_visited_mean > 0.8:
                self.nodes[nearest].dead_end_score = min(1.0, self.nodes[nearest].dead_end_score + 0.02)
            self.last_node_idx = nearest

    def summary(self, robot_pos, goal_pos=None):
        robot_pos = np.asarray(robot_pos, dtype=np.float32)
        if not self.nodes:
            return np.zeros(8, dtype=np.float32)

        dists = np.array([np.linalg.norm(n.pos - robot_pos) for n in self.nodes])
        nearest_idx = int(np.argmin(dists))
        nearest = self.nodes[nearest_idx]

        visits = np.array([n.visited_count for n in self.nodes], dtype=np.float32)
        dead = np.array([n.dead_end_score for n in self.nodes], dtype=np.float32)

        frontier_dir = np.zeros(2, dtype=np.float32)
        if len(self.nodes) > 1:
            low_visit_idx = int(np.argmin(visits))
            vec = self.nodes[low_visit_idx].pos - robot_pos
            norm = np.linalg.norm(vec)
            if norm > 1e-6:
                frontier_dir = vec / norm

        goal_dir = np.zeros(2, dtype=np.float32)
        if goal_pos is not None:
            vec = np.asarray(goal_pos, dtype=np.float32) - robot_pos
            norm = np.linalg.norm(vec)
            if norm > 1e-6:
                goal_dir = vec / norm

        return np.array([
            min(len(self.nodes) / 128.0, 1.0),
            min(nearest.visited_count / 50.0, 1.0),
            min(float(np.mean(visits)) / 50.0, 1.0),
            float(np.mean(dead)),
            frontier_dir[0],
            frontier_dir[1],
            goal_dir[0],
            goal_dir[1],
        ], dtype=np.float32)

