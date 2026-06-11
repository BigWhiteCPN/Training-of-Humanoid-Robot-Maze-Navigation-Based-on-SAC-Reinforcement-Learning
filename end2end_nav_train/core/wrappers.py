import numpy as np
import gymnasium as gym
from gymnasium import spaces


class StateHistoryWrapper(gym.Wrapper):
    """Convert env state observations into a flattened state history."""

    def __init__(self, env, history_length, state_dim):
        super().__init__(env)
        self.history_length = history_length
        self.state_dim = state_dim
        self.history = np.zeros(history_length * state_dim, dtype=np.float32)
        self.observation_space = spaces.Dict({
            "grid_map": env.observation_space["grid_map"],
            "state_history": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(history_length * state_dim,),
                dtype=np.float32,
            ),
        })

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.history[:] = 0.0
        self._append_state(obs["state"])
        return self._make_obs(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._append_state(obs["state"])
        return self._make_obs(obs), reward, terminated, truncated, info

    def _append_state(self, state):
        self.history = np.roll(self.history, -self.state_dim)
        self.history[-self.state_dim:] = state

    def _make_obs(self, obs):
        return {
            "grid_map": obs["grid_map"],
            "state_history": self.history.copy(),
        }

