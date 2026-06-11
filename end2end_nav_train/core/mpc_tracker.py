import math
import numpy as np


class WaypointTracker:
    """Waypoint-sequence to velocity command adapter.

    The low-level locomotion policy remains unchanged. This tracker only produces
    [vx, vy, yaw_rate] commands from the selected high-level waypoint sequence.
    """

    def __init__(self):
        self.max_cmd = np.array([0.8, 0.45, 0.85], dtype=np.float32)
        self.dt = 0.45
        self.horizon = 5

    def pure_pursuit(self, waypoint_action):
        action = np.asarray(waypoint_action, dtype=np.float32)
        wp_x, wp_y = float(action[0]), float(action[1])
        speed_scale = float(np.clip(action[6] if len(action) > 6 else 1.0, 0.2, 1.0))
        heading = math.atan2(wp_y, max(0.2, wp_x))
        dist = math.sqrt(wp_x * wp_x + wp_y * wp_y)
        cmd = np.array([
            np.clip(0.55 * wp_x, -0.35, 0.8) * speed_scale,
            np.clip(0.45 * wp_y, -0.45, 0.45) * speed_scale,
            np.clip(1.2 * heading, -0.85, 0.85),
        ], dtype=np.float32)
        if dist < 0.35:
            cmd[:2] *= 0.4
        return cmd

    def score_command(self, cmd, waypoint_action, clearance=1.0, prev_cmd=None):
        action = np.asarray(waypoint_action, dtype=np.float32)
        waypoints = action[:6].reshape(3, 2) if action.shape[0] >= 6 else action[:2].reshape(1, 2)
        pos = np.zeros(2, dtype=np.float32)
        yaw = 0.0
        tracking_cost = 0.0
        for k in range(self.horizon):
            c, s = math.cos(yaw), math.sin(yaw)
            body_vel = np.array([cmd[0], cmd[1]], dtype=np.float32)
            world_vel = np.array([c * body_vel[0] - s * body_vel[1], s * body_vel[0] + c * body_vel[1]], dtype=np.float32)
            pos = pos + world_vel * self.dt
            yaw = yaw + float(cmd[2]) * self.dt
            target = waypoints[min(k * len(waypoints) // self.horizon, len(waypoints) - 1)]
            tracking_cost += float(np.linalg.norm(target - pos))
        smoothness = 0.0 if prev_cmd is None else float(np.linalg.norm(cmd - prev_cmd))
        speed = float(np.linalg.norm(cmd[:2]))
        clearance_risk = float(np.exp(-max(clearance, 0.0) / 0.35))
        speed_cost = (0.03 + 0.35 * clearance_risk) * speed
        turn_cost = 0.04 * abs(float(cmd[2])) * clearance_risk
        safety_bonus = 0.16 * float(np.clip(clearance, 0.0, 2.0))
        return float(-tracking_cost - 0.10 * smoothness - speed_cost - turn_cost + safety_bonus)

    def mpc(self, waypoint_action, clearance=1.0, prev_cmd=None):
        base = self.pure_pursuit(waypoint_action)
        max_cmd = self.max_cmd.copy()
        if clearance < 0.8:
            scale = float(np.clip((clearance - 0.2) / 0.6, 0.25, 1.0))
            max_cmd[:2] *= scale
            base[:2] *= scale
        if clearance < 0.45:
            max_cmd[0] = min(max_cmd[0], 0.28)
            max_cmd[1] = min(max_cmd[1], 0.18)
            max_cmd[2] = min(max_cmd[2], 0.65)
        candidates = [base]
        sx_values = [0.35, 0.55, 0.75, 1.0] if clearance < 0.8 else [0.55, 0.75, 1.0, 1.15]
        sy_values = [0.25, 0.5, 0.75] if clearance < 0.8 else [0.4, 0.75, 1.0]
        wz_values = [0.6, 0.85, 1.0] if clearance < 0.8 else [0.75, 1.0, 1.15]
        for sx in sx_values:
            for sy in sy_values:
                for wz in wz_values:
                    cand = base.copy()
                    cand[0] *= sx
                    cand[1] *= sy
                    cand[2] *= wz
                    cand = np.clip(cand, -max_cmd, max_cmd)
                    candidates.append(cand.astype(np.float32))
        if prev_cmd is not None:
            prev_cmd = np.asarray(prev_cmd, dtype=np.float32)
            for blend in [0.35, 0.65]:
                cand = base.copy()
                cand = blend * cand + (1.0 - blend) * prev_cmd
                cand = np.clip(cand, -max_cmd, max_cmd)
                candidates.append(cand.astype(np.float32))
        scores = [self.score_command(c, waypoint_action, clearance, prev_cmd) for c in candidates]
        return candidates[int(np.argmax(scores))]
