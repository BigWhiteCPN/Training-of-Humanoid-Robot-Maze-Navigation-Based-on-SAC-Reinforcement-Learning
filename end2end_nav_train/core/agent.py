import copy
import numpy as np
import torch
import torch.nn.functional as F

from .models import AdvancedNavNetworks
from .topological_memory import TopologicalMemory


class AdvancedNavigationAgent:
    def __init__(self, config, grid_shape=(4, 30, 30)):
        self.cfg = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.networks = AdvancedNavNetworks(
            grid_shape=grid_shape,
            state_dim=config.state_dim,
            history_length=config.history_length,
            graph_dim=config.graph_dim,
            action_dim=config.action_dim,
            flow_steps=config.flow_steps,
            flow_loss_weight=config.flow_loss_weight,
            mean_bc_loss_weight=config.mean_bc_loss_weight,
            deterministic_candidate_spread=config.deterministic_candidate_spread,
        ).to(self.device)
        self.target_networks = copy.deepcopy(self.networks).to(self.device)
        self.target_networks.eval()
        self.memory = TopologicalMemory()
        self.policy_opt = torch.optim.Adam(
            list(self.networks.encoder.parameters()) + list(self.networks.policy.parameters()),
            lr=config.lr_policy,
        )
        critic_params = (
            list(self.networks.q1.parameters()) + list(self.networks.q2.parameters())
            + list(self.networks.safety.parameters()) + list(self.networks.progress.parameters())
        )
        self.critic_opt = torch.optim.Adam(critic_params, lr=config.lr_critic)
        self.last_candidate_diversity = 0.0
        self.last_predicted_safe = True
        self.last_obstacle_penalty = 0.0

    def reset_memory(self):
        self.memory.reset()

    def graph_summary(self, env):
        pos, _ = env.unwrapped._get_robot_pose()
        local_visited_mean = 0.0
        try:
            _, local_visited = env.unwrapped.grid_map.get_local_maps(pos, env.unwrapped._get_robot_pose()[1])
            local_visited_mean = float(np.mean(local_visited))
        except Exception:
            pass
        self.memory.update(pos, getattr(env.unwrapped, "prev_geodesic_dist", None), local_visited_mean)
        return self.memory.summary(pos, env.unwrapped.goal_pos)

    def _tensor_obs(self, obs, graph_summary):
        return {
            "grid_map": torch.as_tensor(obs["grid_map"][None], dtype=torch.uint8, device=self.device),
            "state_history": torch.as_tensor(obs["state_history"][None], dtype=torch.float32, device=self.device),
            "graph_summary": torch.as_tensor(graph_summary[None], dtype=torch.float32, device=self.device),
        }

    @torch.no_grad()
    def act(self, obs, graph_summary, deterministic=False):
        batch = self._tensor_obs(obs, graph_summary)
        latent = self.networks.encode_batch(batch)
        candidates = self.networks.policy.sample(
            latent,
            num_candidates=self.cfg.candidate_count,
            deterministic=deterministic,
        )[0]
        latent_rep = latent.expand(candidates.shape[0], -1)
        q = torch.min(self.networks.q1(latent_rep, candidates), self.networks.q2(latent_rep, candidates)).squeeze(-1)
        safety = torch.sigmoid(self.networks.safety(latent_rep, candidates)).squeeze(-1)
        progress = self.networks.progress(latent_rep, candidates).squeeze(-1)
        curvature = torch.mean(torch.abs(candidates[:, 2:6] - candidates[:, 0:4]), dim=-1)
        obstacle_np, invalid_np = self._candidate_obstacle_assessment(obs, candidates)
        obstacle_penalty = torch.as_tensor(
            obstacle_np,
            dtype=torch.float32,
            device=self.device,
        )
        invalid = torch.as_tensor(invalid_np, dtype=torch.bool, device=self.device)
        score = (
            q + self.cfg.progress_policy_weight * progress
            - self.cfg.safety_policy_weight * safety
            - self.cfg.obstacle_policy_weight * obstacle_penalty
            - 0.05 * curvature
        )
        self.last_candidate_diversity = float(torch.mean(torch.std(candidates, dim=0)).detach().cpu().item())
        safe = safety < self.cfg.safety_threshold
        valid = safe & (~invalid)
        if torch.any(valid):
            masked = score.clone()
            masked[~valid] = -1e9
            idx = int(torch.argmax(masked).item())
        elif torch.any(~invalid):
            masked = score.clone()
            masked[invalid] = -1e9
            idx = int(torch.argmax(masked).item())
        elif torch.any(safe):
            idx = int(torch.argmin(obstacle_penalty + 0.25 * safety).item())
        else:
            idx = int(torch.argmin(safety + 0.5 * obstacle_penalty).item())
        self.last_predicted_safe = bool((safety[idx].item() < self.cfg.safety_threshold) and not invalid[idx].item())
        self.last_obstacle_penalty = float(obstacle_penalty[idx].detach().cpu().item())
        return candidates[idx].detach().cpu().numpy().astype(np.float32)

    def _candidate_obstacle_assessment(self, obs, candidates):
        obstacle = np.asarray(obs["grid_map"][0], dtype=np.float32) / 255.0
        obstacle_idx = np.argwhere(obstacle > self.cfg.obstacle_hard_threshold)
        cand_np = candidates.detach().cpu().numpy()
        if len(obstacle_idx) == 0:
            return np.zeros(cand_np.shape[0], dtype=np.float32), np.zeros(cand_np.shape[0], dtype=bool)
        h, w = obstacle.shape
        cells_per_meter = w / 6.0
        center = np.array([h / 2.0, w / 2.0], dtype=np.float32)
        penalties = []
        invalid = []
        for cand in cand_np:
            waypoints = cand[:6].reshape(3, 2)
            max_penalty = 0.0
            blocked = False
            for wp in waypoints:
                samples = np.linspace(0.15, 1.0, 7, dtype=np.float32)[:, None] * wp[None, :]
                for p in samples:
                    row = center[0] - p[1] * cells_per_meter
                    col = center[1] + p[0] * cells_per_meter
                    if row < 0 or row >= h or col < 0 or col >= w:
                        max_penalty = max(max_penalty, 1.0)
                        blocked = True
                        continue
                    ri = int(np.clip(round(row), 0, h - 1))
                    ci = int(np.clip(round(col), 0, w - 1))
                    if obstacle[ri, ci] > self.cfg.obstacle_hard_threshold:
                        blocked = True
                    d_pix = np.sqrt((obstacle_idx[:, 0] - row) ** 2 + (obstacle_idx[:, 1] - col) ** 2)
                    min_dist_m = float(np.min(d_pix) / cells_per_meter)
                    if min_dist_m < self.cfg.waypoint_min_clearance:
                        blocked = True
                    max_penalty = max(max_penalty, float(np.exp(-min_dist_m / 0.35)))
            penalties.append(max_penalty)
            invalid.append(blocked)
        return np.asarray(penalties, dtype=np.float32), np.asarray(invalid, dtype=bool)

    def _candidate_obstacle_penalty(self, obs, candidates):
        penalties, _ = self._candidate_obstacle_assessment(obs, candidates)
        return penalties

    @torch.no_grad()
    def estimate_candidate_diversity(self, obs, graph_summary):
        batch = self._tensor_obs(obs, graph_summary)
        latent = self.networks.encode_batch(batch)
        candidates = self.networks.policy.sample(latent, num_candidates=self.cfg.candidate_count, deterministic=False)[0]
        return float(torch.mean(torch.std(candidates, dim=0)).detach().cpu().item())

    def bc_update(self, batch):
        latent = self.networks.encode_batch(batch)
        loss = self.networks.policy.bc_loss(latent, batch["teacher_action"])
        self.policy_opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.networks.parameters(), 1.0)
        self.policy_opt.step()
        self.networks.policy.log_std.data.clamp_(self.cfg.log_std_min, self.cfg.log_std_max)
        return float(loss.detach().cpu().item())

    def critic_update(self, batch):
        latent = self.networks.encode_batch(batch)
        with torch.no_grad():
            next_latent = self.target_networks.encode_batch({
                "grid_map": batch["next_grid_map"],
                "state_history": batch["next_state_history"],
                "graph_summary": batch["next_graph_summary"],
            })
            next_action = self.target_networks.policy.deterministic_action(next_latent)
            target_q = torch.min(
                self.target_networks.q1(next_latent, next_action),
                self.target_networks.q2(next_latent, next_action),
            ).squeeze(-1)
            reward = batch["reward"] * self.cfg.reward_scale
            y = reward + self.cfg.gamma * (1.0 - batch["done"]) * target_q

        q1 = self.networks.q1(latent, batch["action"]).squeeze(-1)
        q2 = self.networks.q2(latent, batch["action"]).squeeze(-1)
        q_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)
        safety_loss = F.binary_cross_entropy_with_logits(
            self.networks.safety(latent, batch["action"]).squeeze(-1),
            batch["collision_label"],
        )
        progress_loss = F.mse_loss(
            self.networks.progress(latent, batch["action"]).squeeze(-1),
            batch["progress_label"] * self.cfg.progress_scale,
        )
        loss = q_loss + safety_loss + progress_loss
        self.critic_opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.networks.q1.parameters()) + list(self.networks.q2.parameters())
            + list(self.networks.safety.parameters()) + list(self.networks.progress.parameters()),
            1.0,
        )
        self.critic_opt.step()
        return {
            "critic_loss": float(loss.detach().cpu().item()),
            "q_loss": float(q_loss.detach().cpu().item()),
            "safety_loss": float(safety_loss.detach().cpu().item()),
            "progress_loss": float(progress_loss.detach().cpu().item()),
        }

    def policy_update(self, batch, bc_weight=0.0):
        latent = self.networks.encode_batch(batch)
        candidates = self.networks.policy.sample(latent, num_candidates=1).squeeze(1)
        q = torch.min(self.networks.q1(latent, candidates), self.networks.q2(latent, candidates)).squeeze(-1)
        safety = torch.sigmoid(self.networks.safety(latent, candidates)).squeeze(-1)
        progress = self.networks.progress(latent, candidates).squeeze(-1)
        rl_loss = -(q + self.cfg.progress_policy_weight * progress - self.cfg.safety_policy_weight * safety).mean()
        entropy_bonus = self.networks.policy.log_std.mean()
        loss = self.cfg.rl_loss_weight * rl_loss - self.cfg.entropy_coef * entropy_bonus
        anchor_loss = torch.tensor(0.0, device=self.device)
        if bc_weight > 0.0:
            bc_loss = self.networks.policy.bc_loss(latent, batch["teacher_action"])
            anchor_loss = F.mse_loss(candidates, batch["teacher_action"])
            loss = loss + bc_weight * bc_loss
            loss = loss + self.cfg.action_anchor_weight * anchor_loss
        else:
            bc_loss = torch.tensor(0.0, device=self.device)
        self.policy_opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.networks.encoder.parameters()) + list(self.networks.policy.parameters()),
            1.0,
        )
        self.policy_opt.step()
        self.networks.policy.log_std.data.clamp_(self.cfg.log_std_min, self.cfg.log_std_max)
        return {
            "policy_loss": float(loss.detach().cpu().item()),
            "rl_loss": float(rl_loss.detach().cpu().item()),
            "entropy_bonus": float(entropy_bonus.detach().cpu().item()),
            "policy_std": float(torch.exp(self.networks.policy.log_std).mean().detach().cpu().item()),
            "bc_loss": float(bc_loss.detach().cpu().item()),
            "anchor_loss": float(anchor_loss.detach().cpu().item()),
            "policy_safety": float(safety.mean().detach().cpu().item()),
            "policy_progress": float(progress.mean().detach().cpu().item()),
        }

    def soft_update_targets(self):
        tau = self.cfg.tau
        for target, source in zip(self.target_networks.parameters(), self.networks.parameters()):
            target.data.mul_(1.0 - tau).add_(source.data, alpha=tau)
