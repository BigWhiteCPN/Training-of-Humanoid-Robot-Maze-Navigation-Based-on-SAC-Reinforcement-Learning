"""Difficulty-balanced replay sampling for random-map SAC training."""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np
from stable_baselines3.common.buffers import DictReplayBuffer
from stable_baselines3.common.type_aliases import DictReplayBufferSamples
from stable_baselines3.common.vec_env import VecNormalize


class DifficultyBalancedDictReplayBuffer(DictReplayBuffer):
    """Sample each SAC batch with a more even mix of episode difficulties.

    The buffer still stores every transition normally. Only sampling changes:
    transitions are bucketed by a scalar difficulty value from ``info`` and each
    training batch is filled across those buckets when possible.
    """

    def __init__(
        self,
        *args: Any,
        difficulty_key: str = "difficulty_path_len",
        bin_edges: Sequence[float] = (5.0, 8.0, 12.0, 16.0),
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.difficulty_key = difficulty_key
        self.bin_edges = np.asarray(bin_edges, dtype=np.float32)
        self.num_difficulty_bins = len(self.bin_edges) + 1
        self.difficulty_bins = np.zeros(
            (self.buffer_size, self.n_envs), dtype=np.int16
        )

    def _difficulty_to_bin(self, value: float) -> int:
        return int(np.searchsorted(self.bin_edges, value, side="right"))

    def add(  # type: ignore[override]
        self,
        obs: dict[str, np.ndarray],
        next_obs: dict[str, np.ndarray],
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: list[dict[str, Any]],
    ) -> None:
        current_pos = self.pos
        for env_idx, info in enumerate(infos):
            difficulty = float(info.get(self.difficulty_key, 0.0))
            self.difficulty_bins[current_pos, env_idx] = self._difficulty_to_bin(
                difficulty
            )
        super().add(obs, next_obs, action, reward, done, infos)

    def sample(  # type: ignore[override]
        self,
        batch_size: int,
        env: Optional[VecNormalize] = None,
    ) -> DictReplayBufferSamples:
        upper_bound = self.buffer_size if self.full else self.pos
        if upper_bound <= 1:
            return super().sample(batch_size=batch_size, env=env)

        valid_bins = self.difficulty_bins[:upper_bound].reshape(-1)
        valid_flat_indices = np.arange(upper_bound * self.n_envs)
        sampled_flat_indices = []

        per_bin = max(1, batch_size // self.num_difficulty_bins)
        for bin_id in range(self.num_difficulty_bins):
            bin_flat_indices = valid_flat_indices[valid_bins == bin_id]
            if len(bin_flat_indices) == 0:
                continue
            take = min(per_bin, batch_size - len(sampled_flat_indices))
            if take <= 0:
                break
            sampled = np.random.choice(
                bin_flat_indices,
                size=take,
                replace=len(bin_flat_indices) < take,
            )
            sampled_flat_indices.extend(sampled.tolist())

        remaining = batch_size - len(sampled_flat_indices)
        if remaining > 0:
            sampled = np.random.choice(
                valid_flat_indices,
                size=remaining,
                replace=len(valid_flat_indices) < remaining,
            )
            sampled_flat_indices.extend(sampled.tolist())

        sampled_flat_indices = np.asarray(sampled_flat_indices, dtype=np.int64)
        np.random.shuffle(sampled_flat_indices)
        batch_inds = sampled_flat_indices // self.n_envs
        env_indices = sampled_flat_indices % self.n_envs
        return self._get_samples_for_env_indices(batch_inds, env_indices, env)

    def _get_samples_for_env_indices(
        self,
        batch_inds: np.ndarray,
        env_indices: np.ndarray,
        env: Optional[VecNormalize] = None,
    ) -> DictReplayBufferSamples:
        obs_ = self._normalize_obs(
            {
                key: obs[batch_inds, env_indices, :]
                for key, obs in self.observations.items()
            },
            env,
        )
        next_obs_ = self._normalize_obs(
            {
                key: obs[batch_inds, env_indices, :]
                for key, obs in self.next_observations.items()
            },
            env,
        )

        assert isinstance(obs_, dict)
        assert isinstance(next_obs_, dict)
        observations = {key: self.to_torch(obs) for key, obs in obs_.items()}
        next_observations = {
            key: self.to_torch(obs) for key, obs in next_obs_.items()
        }

        dones = self.dones[batch_inds, env_indices]
        dones = dones * (1 - self.timeouts[batch_inds, env_indices])
        rewards = self.rewards[batch_inds, env_indices].reshape(-1, 1)

        return DictReplayBufferSamples(
            observations=observations,
            actions=self.to_torch(self.actions[batch_inds, env_indices]),
            next_observations=next_observations,
            dones=self.to_torch(dones.reshape(-1, 1)),
            rewards=self.to_torch(self._normalize_reward(rewards, env)),
        )
