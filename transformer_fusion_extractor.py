"""Cross-attention feature extractor shared by transformer train and play scripts."""

from __future__ import annotations

import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CrossAttentionFusionExtractor(BaseFeaturesExtractor):
    """Fuse robot state history with spatial map tokens using cross-attention.

    The GRU state token is the query. CNN map tokens are the keys and values.
    The returned feature is one fused latent state consumed by the SAC actor
    and critics.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        state_feature_dim: int = 9,
        history_length: int = 15,
        d_model: int = 128,
        num_heads: int = 4,
        ffn_dim: int = 256,
        features_dim: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__(observation_space, features_dim)

        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads}).")

        self.state_feature_dim = state_feature_dim
        self.history_length = history_length

        map_shape = observation_space["grid_map"].shape
        state_shape = observation_space["state_history"].shape
        expected_state_dim = history_length * state_feature_dim
        if len(map_shape) != 3:
            raise ValueError(f"grid_map must be CHW, received shape {map_shape}.")
        if state_shape[0] != expected_state_dim:
            raise ValueError(
                f"state_history has {state_shape[0]} values, expected "
                f"{history_length} x {state_feature_dim} = {expected_state_dim}."
            )

        # Keep a spatial feature grid instead of flattening the complete map.
        self.map_encoder = nn.Sequential(
            nn.Conv2d(map_shape[0], 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, d_model, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        with torch.no_grad():
            sample_map = torch.zeros(1, *map_shape)
            encoded_map = self.map_encoder(sample_map)
        self.map_height = encoded_map.shape[-2]
        self.map_width = encoded_map.shape[-1]
        self.num_map_tokens = self.map_height * self.map_width

        # Learned 2-D location identity for every map token.
        self.map_position_embedding = nn.Parameter(
            torch.zeros(1, self.num_map_tokens, d_model)
        )
        nn.init.trunc_normal_(self.map_position_embedding, std=0.02)
        self.map_token_norm = nn.LayerNorm(d_model)

        # Encode the ordered robot-state history.
        self.state_embedding = nn.Linear(state_feature_dim, d_model)
        self.state_input_norm = nn.LayerNorm(d_model)
        self.state_time_embedding = nn.Parameter(
            torch.zeros(1, history_length, d_model)
        )
        nn.init.trunc_normal_(self.state_time_embedding, std=0.02)
        self.state_gru = nn.GRU(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=2,
            batch_first=True,
        )
        self.query_norm = nn.LayerNorm(d_model)

        # State-conditioned map lookup.
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attention_residual_norm = nn.LayerNorm(d_model)

        # Transformer feed-forward sub-layer for nonlinear modal interaction.
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
        )
        self.feed_forward_norm = nn.LayerNorm(d_model)

        self.output_projection = nn.Sequential(
            nn.Linear(d_model, features_dim),
            nn.LayerNorm(features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        # Preserve the normalization convention used by the original random-map
        # train/play implementation.
        grid_map = observations["grid_map"].float() / 255.0

        map_features = self.map_encoder(grid_map)
        map_tokens = map_features.flatten(2).transpose(1, 2)
        map_tokens = self.map_token_norm(
            map_tokens + self.map_position_embedding
        )

        batch_size = observations["state_history"].shape[0]
        state_sequence = observations["state_history"].reshape(
            batch_size, self.history_length, self.state_feature_dim
        )
        state_tokens = self.state_embedding(state_sequence)
        state_tokens = self.state_input_norm(
            state_tokens + self.state_time_embedding
        )
        state_tokens = torch.relu(state_tokens)
        state_output, _ = self.state_gru(state_tokens)

        # The last recurrent state summarizes the 15-step history and asks the
        # map which spatial regions matter for the current navigation decision.
        state_query = self.query_norm(state_output[:, -1:, :])
        attended_map, _ = self.cross_attention(
            query=state_query,
            key=map_tokens,
            value=map_tokens,
            need_weights=False,
        )

        fused_token = self.attention_residual_norm(state_query + attended_map)
        fused_token = self.feed_forward_norm(
            fused_token + self.feed_forward(fused_token)
        )
        return self.output_projection(fused_token[:, 0, :])

