import torch
import torch.nn as nn
import math

class DiffusionSinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class ConditionalUnet1D(nn.Module):
    """简化的1D Conditional U-Net，用于轨迹去噪"""
    def __init__(self, input_dim=2, cond_dim=128, diff_step_embed_dim=32):
        super().__init__()
        self.input_dim = input_dim
        self.step_embed = DiffusionSinusoidalPosEmb(diff_step_embed_dim)
        
        # 降采样
        self.down1 = nn.Conv1d(input_dim, 64, kernel_size=5, padding=2)
        self.down2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        
        # 融合 Condition (Map features + Goal)
        self.cond_proj = nn.Linear(cond_dim + diff_step_embed_dim, 128 * 2) # *2 for bias modulation

        # 中间层
        self.mid = nn.Conv1d(128, 128, kernel_size=5, padding=2)
        
        # 上采样
        self.up1 = nn.Conv1d(128, 64, kernel_size=5, padding=2)
        self.final = nn.Conv1d(64, input_dim, kernel_size=5, padding=2)
        
        self.act = nn.Mish()

    def forward(self, sample, timestep, global_cond):
        # sample: (B, 2, T), timestep: (B,), global_cond: (B, cond_dim)
        
        # 1. Embed Time & Condition
        t_emb = self.step_embed(timestep)
        cond = torch.cat([global_cond, t_emb], dim=-1)
        scale, shift = self.cond_proj(cond).chunk(2, dim=1)
        scale = scale.unsqueeze(-1)
        shift = shift.unsqueeze(-1)

        # 2. Down
        x = self.act(self.down1(sample))
        x = self.act(self.down2(x))
        
        # 3. FiLM Conditioning (Feature-wise Linear Modulation)
        x = x * (1 + scale) + shift
        
        # 4. Mid
        x = self.act(self.mid(x))
        
        # 5. Up
        x = self.act(self.up1(x))
        out = self.final(x)
        
        return out

class MapEncoder(nn.Module):
    """简单的CNN，用于将局部栅格地图编码为特征向量"""
    def __init__(self, input_channels=1, feature_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            # 输入 60x60
            nn.Conv2d(input_channels, 16, 5, 2, 2), nn.ReLU(), # -> 30x30
            nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU(),             # -> 15x15
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),             # -> 8x8
            nn.Flatten(),
            # [修正] 这里是 64 * 8 * 8 = 4096
            nn.Linear(64 * 8 * 8, feature_dim), 
            nn.LayerNorm(feature_dim)
        )
    def forward(self, x):
        return self.net(x)