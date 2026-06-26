"""SAC with PopArt-normalized critic targets."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch as th
from stable_baselines3 import SAC
from stable_baselines3.common.utils import polyak_update
from torch.nn import functional as F


class PopArtSAC(SAC):
    """SAC variant that normalizes Q targets with PopArt.

    The critic learns normalized Q values, while actor and target computation use
    denormalized Q values. When running target statistics change, the final
    critic layers are rescaled so their denormalized predictions are preserved.
    """

    def __init__(
        self,
        *args: Any,
        use_popart: bool = True,
        popart_beta: float = 3e-4,
        popart_epsilon: float = 1e-5,
        popart_min_std: float = 1.0,
        **kwargs: Any,
    ) -> None:
        self.use_popart = use_popart
        self.popart_beta = popart_beta
        self.popart_epsilon = popart_epsilon
        self.popart_min_std = popart_min_std
        self.popart_initialized = False
        super().__init__(*args, **kwargs)
        self._init_popart_stats()

    def _init_popart_stats(self) -> None:
        device = self.device
        self.popart_mean = th.zeros(1, device=device)
        self.popart_second_moment = th.ones(1, device=device)
        self.popart_std = th.ones(1, device=device)

    def _get_torch_save_params(self) -> tuple[list[str], list[str]]:
        state_dicts, torch_vars = super()._get_torch_save_params()
        torch_vars += ["popart_mean", "popart_second_moment", "popart_std"]
        return state_dicts, torch_vars

    def _last_q_layers(self):
        for critic in (self.critic, self.critic_target):
            for q_net in critic.q_networks:
                yield q_net[-1]

    def _denormalize_q(self, q_value: th.Tensor) -> th.Tensor:
        if not self.use_popart:
            return q_value
        return q_value * self.popart_std + self.popart_mean

    def _normalize_q_target(self, q_target: th.Tensor) -> th.Tensor:
        if not self.use_popart:
            return q_target
        return (q_target - self.popart_mean) / self.popart_std

    def _update_popart_stats(self, target_q_values: th.Tensor) -> None:
        if not self.use_popart:
            return

        with th.no_grad():
            old_mean = self.popart_mean.clone()
            old_std = self.popart_std.clone()

            batch_mean = target_q_values.mean()
            batch_second_moment = th.mean(target_q_values.square())
            if not self.popart_initialized:
                new_mean = batch_mean
                new_second_moment = batch_second_moment
                self.popart_initialized = True
            else:
                beta = self.popart_beta
                new_mean = (1.0 - beta) * old_mean + beta * batch_mean
                new_second_moment = (
                    (1.0 - beta) * self.popart_second_moment
                    + beta * batch_second_moment
                )

            min_var = self.popart_min_std**2
            new_var = th.clamp(
                new_second_moment - new_mean.square(),
                min=min_var,
            )
            new_std = th.sqrt(new_var + self.popart_epsilon)

            self._rescale_critic_outputs(old_mean, old_std, new_mean, new_std)
            self.popart_mean.data.copy_(new_mean.reshape_as(self.popart_mean))
            self.popart_second_moment.data.copy_(
                new_second_moment.reshape_as(self.popart_second_moment)
            )
            self.popart_std.data.copy_(new_std.reshape_as(self.popart_std))

    def _rescale_critic_outputs(
        self,
        old_mean: th.Tensor,
        old_std: th.Tensor,
        new_mean: th.Tensor,
        new_std: th.Tensor,
    ) -> None:
        scale = old_std / new_std
        bias_shift = (old_mean - new_mean) / new_std
        for last_layer in self._last_q_layers():
            last_layer.weight.data.mul_(scale)
            last_layer.bias.data.mul_(scale).add_(bias_shift)

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        self.policy.set_training_mode(True)
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []

        for gradient_step in range(gradient_steps):
            # PopArt owns target normalization, so avoid VecNormalize reward
            # normalization here. Observations are not normalized in this setup.
            sample_env = None if self.use_popart else self._vec_normalize_env
            replay_data = self.replay_buffer.sample(batch_size, env=sample_env)  # type: ignore[union-attr]

            if self.use_sde:
                self.actor.reset_noise()

            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                ent_coef = th.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(
                    self.log_ent_coef * (log_prob + self.target_entropy).detach()
                ).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor

            ent_coefs.append(ent_coef.item())

            if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            with th.no_grad():
                next_actions, next_log_prob = self.actor.action_log_prob(
                    replay_data.next_observations
                )
                next_q_values = th.cat(
                    self.critic_target(replay_data.next_observations, next_actions),
                    dim=1,
                )
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                next_q_values = self._denormalize_q(next_q_values)
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(
                    -1, 1
                )
                target_q_values = (
                    replay_data.rewards
                    + (1 - replay_data.dones) * self.gamma * next_q_values
                )
                self._update_popart_stats(target_q_values)
                normalized_target_q_values = self._normalize_q_target(
                    target_q_values
                )

            current_q_values = self.critic(
                replay_data.observations, replay_data.actions
            )
            critic_loss = 0.5 * sum(
                F.mse_loss(current_q, normalized_target_q_values)
                for current_q in current_q_values
            )
            assert isinstance(critic_loss, th.Tensor)
            critic_losses.append(critic_loss.item())

            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            q_values_pi = th.cat(
                self.critic(replay_data.observations, actions_pi), dim=1
            )
            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
            min_qf_pi = self._denormalize_q(min_qf_pi)
            actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
            actor_losses.append(actor_loss.item())

            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            if gradient_step % self.target_update_interval == 0:
                polyak_update(
                    self.critic.parameters(), self.critic_target.parameters(), self.tau
                )
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        if self.use_popart:
            self.logger.record("train/popart_mean", self.popart_mean.item())
            self.logger.record("train/popart_std", self.popart_std.item())
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))
