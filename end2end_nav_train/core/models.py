import torch
import torch.nn as nn
import torch.nn.functional as F


class MapStateEncoder(nn.Module):
    def __init__(self, grid_shape=(4, 30, 30), state_dim=12, history_length=10, graph_dim=8, latent_dim=256):
        super().__init__()
        c, h, w = grid_shape
        self.state_dim = state_dim
        self.history_length = history_length
        self.cnn = nn.Sequential(
            nn.Conv2d(c, 16, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            flat = self.cnn(torch.zeros(1, c, h, w)).shape[1]
        self.map_head = nn.Sequential(nn.Linear(flat, 128), nn.LayerNorm(128), nn.ReLU())
        self.state_embed = nn.Linear(state_dim, 128)
        self.gru = nn.GRU(128, 128, num_layers=2, batch_first=True)
        self.graph_head = nn.Sequential(nn.Linear(graph_dim, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
        self.fusion = nn.Sequential(nn.Linear(128 + 128 + 64, latent_dim), nn.LayerNorm(latent_dim), nn.ReLU())

    def forward(self, grid_map, state_history, graph_summary):
        grid = grid_map.float() / 255.0
        map_feat = self.map_head(self.cnn(grid))
        batch = state_history.shape[0]
        seq = state_history.view(batch, self.history_length, self.state_dim)
        seq = F.relu(self.state_embed(seq))
        gru_out, _ = self.gru(seq)
        state_feat = gru_out[:, -1]
        graph_feat = self.graph_head(graph_summary.float())
        return self.fusion(torch.cat([map_feat, state_feat, graph_feat], dim=-1))


class FlowWaypointPolicy(nn.Module):
    """Flow-matching waypoint sequence policy."""

    def __init__(
        self,
        latent_dim=256,
        action_dim=7,
        flow_steps=4,
        flow_loss_weight=1.0,
        mean_bc_loss_weight=0.25,
        deterministic_candidate_spread=0.55,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.flow_steps = flow_steps
        self.flow_loss_weight = flow_loss_weight
        self.mean_bc_loss_weight = mean_bc_loss_weight
        self.deterministic_candidate_spread = deterministic_candidate_spread
        self.mean = nn.Sequential(nn.Linear(latent_dim, 256), nn.ReLU(), nn.Linear(256, action_dim))
        self.vector_field = nn.Sequential(
            nn.Linear(latent_dim + action_dim + 1, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, action_dim),
        )
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.8))
        self.register_buffer("action_low", torch.tensor([-3, -3, -3, -3, -3, -3, 0.2], dtype=torch.float32))
        self.register_buffer("action_high", torch.tensor([3, 3, 3, 3, 3, 3, 1.0], dtype=torch.float32))

    def clamp_action(self, action):
        return torch.max(torch.min(action, self.action_high), self.action_low)

    def deterministic_action(self, latent):
        action = self.mean(latent)
        for step in range(self.flow_steps):
            t = torch.full((action.shape[0], 1), float(step) / max(self.flow_steps, 1), device=latent.device)
            v = self.vector_field(torch.cat([latent, action, t], dim=-1))
            action = action + v / max(self.flow_steps, 1)
        return self.clamp_action(action)

    def deterministic_candidate_offsets(self, num_candidates, device):
        offsets = torch.zeros(num_candidates, self.action_dim, device=device)
        if num_candidates <= 1:
            return offsets
        lateral = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0], device=device)
        forward = torch.tensor([-0.5, 0.0, 0.5], device=device)
        speed = torch.tensor([-0.15, 0.0, 0.1], device=device)
        for i in range(1, num_candidates):
            lat = lateral[i % len(lateral)]
            fwd = forward[(i // len(lateral)) % len(forward)]
            spd = speed[(i // (len(lateral) * len(forward))) % len(speed)]
            scale = self.deterministic_candidate_spread * (0.65 + 0.15 * float(i % 4))
            offsets[i, 0] = fwd * scale
            offsets[i, 1] = lat * scale
            offsets[i, 2] = 0.8 * fwd * scale
            offsets[i, 3] = 0.8 * lat * scale
            offsets[i, 4] = 1.2 * fwd * scale
            offsets[i, 5] = 1.2 * lat * scale
            offsets[i, 6] = spd
        return offsets

    def sample(self, latent, num_candidates=1, deterministic=False):
        if deterministic:
            action = self.deterministic_action(latent)
            offsets = self.deterministic_candidate_offsets(num_candidates, latent.device)
            candidates = action[:, None, :] + offsets[None, :, :]
            return self.clamp_action(candidates)
        batch = latent.shape[0]
        latent_rep = latent[:, None, :].expand(batch, num_candidates, -1).reshape(batch * num_candidates, -1)
        std = torch.exp(self.log_std).view(1, -1)
        action = torch.randn(batch * num_candidates, self.action_dim, device=latent.device) * std
        for step in range(self.flow_steps):
            t = torch.full((action.shape[0], 1), float(step) / max(self.flow_steps, 1), device=latent.device)
            v = self.vector_field(torch.cat([latent_rep, action, t], dim=-1))
            action = action + v / max(self.flow_steps, 1)
        return self.clamp_action(action).view(batch, num_candidates, self.action_dim)

    def flow_matching_loss(self, latent, target):
        noise = torch.randn_like(target)
        t = torch.rand(target.shape[0], 1, device=target.device)
        x_t = (1.0 - t) * noise + t * target
        target_v = target - noise
        pred_v = self.vector_field(torch.cat([latent, x_t, t], dim=-1))
        return F.mse_loss(pred_v, target_v)

    def bc_loss(self, latent, target):
        mean_loss = F.mse_loss(self.mean(latent), target)
        flow_loss = self.flow_matching_loss(latent, target)
        return self.flow_loss_weight * flow_loss + self.mean_bc_loss_weight * mean_loss


class Critic(nn.Module):
    def __init__(self, latent_dim=256, action_dim=7, output_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, output_dim),
        )

    def forward(self, latent, action):
        return self.net(torch.cat([latent, action], dim=-1))


class AdvancedNavNetworks(nn.Module):
    def __init__(
        self,
        grid_shape=(4, 30, 30),
        state_dim=12,
        history_length=10,
        graph_dim=8,
        action_dim=7,
        flow_steps=4,
        flow_loss_weight=1.0,
        mean_bc_loss_weight=0.25,
        deterministic_candidate_spread=0.55,
    ):
        super().__init__()
        self.encoder = MapStateEncoder(grid_shape, state_dim, history_length, graph_dim)
        self.policy = FlowWaypointPolicy(
            action_dim=action_dim,
            flow_steps=flow_steps,
            flow_loss_weight=flow_loss_weight,
            mean_bc_loss_weight=mean_bc_loss_weight,
            deterministic_candidate_spread=deterministic_candidate_spread,
        )
        self.q1 = Critic(action_dim=action_dim)
        self.q2 = Critic(action_dim=action_dim)
        self.safety = Critic(action_dim=action_dim)
        self.progress = Critic(action_dim=action_dim)

    def encode_batch(self, batch):
        return self.encoder(batch["grid_map"], batch["state_history"], batch["graph_summary"])
