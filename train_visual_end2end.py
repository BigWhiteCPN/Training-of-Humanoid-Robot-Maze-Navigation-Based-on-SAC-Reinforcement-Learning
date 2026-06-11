"""
End-to-end navigation training with curriculum learning.

Notes:
1. Uses RobotVisualEnd2EndEnv.
2. Adapts the network to 4-channel grid maps and 12-dimensional state features.
3. Logs curriculum statistics to TensorBoard.
"""

import os
import time
import multiprocessing
import sys
from collections import deque, defaultdict
from typing import Callable
from pathlib import Path

sys.path.append("/home/iansten/code/IsaacLabExtensionTemplate/scripts/visual_train/")

try:
    from robot_visual_env_end2end import RobotVisualEnd2EndEnv
except ImportError:
    raise ImportError("无法找到 'robot_visual_env_end2end.py'。请确保该文件在同一目录下。")


# 1. 全局配置
class Config:
    # 训练设置
    num_envs = 1
    total_timesteps = 40_000_000
    lr = 1e-4
    seed = 42

    # 显示模式
    render = False

    resume_from = None

    # 路径
    model_xml = "/home/iansten/code/IsaacLabExtensionTemplate/scripts/resources/mjcf/Linnxil_fifteen_angle_bs_copy_20260302.xml"
    policy_path = "/home/iansten/code/IsaacLabExtensionTemplate/scripts/visual_train/policy_20251026.pt"
    log_dir = "./sac_end2end_logs/"
    teacher_dataset_path = "./sac_end2end_logs/teacher_waypoint_dataset.npz"
    final_playable_path = "./sac_end2end_logs/final_playable_model.zip"

    # 一体化训练阶段
    enable_teacher_dataset = True
    teacher_dataset_version = 10
    teacher_episodes_per_level = 24
    teacher_hard_extra_episodes = 80
    teacher_hard_start_level = 1
    teacher_max_steps_per_episode = 3500
    teacher_hard_max_steps_per_episode = 3500
    bc_pretrain_updates = 10000
    bc_batch_size = 512
    bc_lr = 3e-4
    bc_regularization_updates = 5
    bc_regularization_freq = 5
    bc_regularization_weight = 0.5
    bc_hard_sample_fraction = 0.45
    bc_success_sample_fraction = 0.35

    # 课程升级保护: 升级后临时提高探索和 teacher 约束
    level_transition_boost_steps = 500_000
    level_transition_ent_coef_floor = 0.12
    level_transition_sde_sample_freq = 4
    level_transition_bc_freq = 5
    level_transition_bc_updates = 4
    level_transition_bc_weight_multiplier = 2.0

    # 环境参数
    decimation = 50
    action_repeat = 4
    history_length = 10
    state_feature_dim = 12  # rel_goal(2) + lin_vel(2) + ang_vel(1) + euclid_dist(1) + unknown_ratio(1) + heading(2) + prev_action(3)

    enable_dynamic_obstacles = False
    action_mode = "velocity"

    # 底层执行加速: 不改变 RL 采样/更新比例
    enable_tf32 = True
    enable_channels_last = True
    enable_torch_compile = False
    torch_compile_mode = "reduce-overhead"


# 2. Matplotlib 后端
import matplotlib
if not Config.render:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt


# 3. 依赖
import torch
import torch.nn as nn
import torch._dynamo
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback, EvalCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


def configure_torch_runtime():
    torch._dynamo.config.suppress_errors = True
    if Config.enable_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    torch.backends.cudnn.benchmark = True


configure_torch_runtime()


# 4. 状态历史包装器
class StateHistoryWrapper(gym.Wrapper):
    """将 env 输出的 state 扩展为 state_history 以供 GRU 使用"""
    def __init__(self, env, history_length, state_dim):
        super().__init__(env)
        self.history_length = history_length
        self.state_dim = state_dim
        self.history = np.zeros(history_length * state_dim, dtype=np.float32)
        self.observation_space = spaces.Dict({
            "grid_map": env.observation_space["grid_map"],
            "state_history": spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(history_length * state_dim,),
                dtype=np.float32
            )
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
        return {"grid_map": obs["grid_map"], "state_history": self.history.copy()}


# 5. 网络结构: 轻量 CoordConv-CNN + 状态GRU
class End2EndExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict,
                 state_feature_dim=12, history_length=10,
                 d_model=128, features_dim=256):
        super().__init__(observation_space, features_dim)

        # --- 地图编码: 4通道局部地图 + CoordConv(x/y/r) ---
        n_input_channels = observation_space['grid_map'].shape[0] + 3
        self.map_encoder = nn.Sequential(
            nn.Conv2d(n_input_channels, 24, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(num_groups=4, num_channels=24),
            nn.SiLU(),
            nn.Conv2d(24, 48, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=48),
            nn.SiLU(),
            nn.Conv2d(48, 96, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=96),
            nn.SiLU(),
        )
        if Config.enable_channels_last:
            self.map_encoder = self.map_encoder.to(memory_format=torch.channels_last)

        self.map_head = nn.Sequential(
            nn.Linear(96 * 2, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU()
        )

        # --- 状态序列 GRU ---
        self.state_sub_dim = state_feature_dim
        self.seq_len = history_length

        self.state_embedding = nn.Sequential(
            nn.LayerNorm(state_feature_dim),
            nn.Linear(state_feature_dim, d_model),
            nn.SiLU(),
        )
        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=2,
            batch_first=True,
            dropout=0.05,
        )

        # --- 融合层 ---
        self.fusion_layer = nn.Sequential(
            nn.Linear(d_model + d_model + state_feature_dim, features_dim),
            nn.LayerNorm(features_dim),
            nn.SiLU(),
        )

    def _add_coord_channels(self, grid_map):
        batch_size, _, height, width = grid_map.shape
        y = torch.linspace(-1.0, 1.0, height, device=grid_map.device, dtype=grid_map.dtype)
        x = torch.linspace(-1.0, 1.0, width, device=grid_map.device, dtype=grid_map.dtype)
        yy = y.view(1, 1, height, 1).expand(batch_size, 1, height, width)
        xx = x.view(1, 1, 1, width).expand(batch_size, 1, height, width)
        rr = torch.sqrt(torch.clamp(xx * xx + yy * yy, min=0.0)) / 1.41421356237
        grid_with_coords = torch.cat([grid_map, xx, yy, rr], dim=1)
        if Config.enable_channels_last:
            grid_with_coords = grid_with_coords.contiguous(memory_format=torch.channels_last)
        return grid_with_coords

    def forward(self, observations):
        # 栅格地图 (4通道: obstacle/visited/goal_map/unknown_map)
        grid_map = observations['grid_map'].float() / 255.0
        if Config.enable_channels_last:
            grid_map = grid_map.contiguous(memory_format=torch.channels_last)
        map_feat = self.map_encoder(self._add_coord_channels(grid_map))

        # 状态序列
        batch_size = observations['state_history'].shape[0]
        state_seq = observations['state_history'].view(batch_size, self.seq_len, self.state_sub_dim)
        x = self.state_embedding(state_seq)
        gru_out, _ = self.gru(x)
        temporal_feat = gru_out[:, -1, :]
        current_state = state_seq[:, -1, :]

        map_avg = torch.mean(map_feat, dim=(2, 3))
        map_max = torch.amax(map_feat, dim=(2, 3))
        map_feat = self.map_head(torch.cat([map_avg, map_max], dim=1))

        combined = torch.cat([map_feat, temporal_feat, current_state], dim=1)
        return self.fusion_layer(combined)


# 5. 回调函数
class LevelTransitionBoostState:
    def __init__(self):
        self.boost_until_step = 0
        self.boost_level = 0

    def trigger(self, current_step: int, level: int, duration_steps: int):
        self.boost_level = int(level)
        self.boost_until_step = max(self.boost_until_step, int(current_step) + int(duration_steps))

    def remaining(self, current_step: int) -> int:
        return max(0, self.boost_until_step - int(current_step))

    def is_active(self, current_step: int) -> bool:
        return self.remaining(current_step) > 0


class CurriculumCallback(BaseCallback):
    """记录课程等级，并在课程升级后临时提高探索和 teacher 约束。"""
    def __init__(self, boost_state=None, verbose=1):
        super().__init__(verbose)
        self.boost_state = boost_state
        self.last_level = None
        self.base_sde_sample_freq = None
        self.boost_was_active = False

    def _on_training_start(self) -> None:
        self.base_sde_sample_freq = getattr(self.model, "sde_sample_freq", None)

    def _set_entropy_floor(self, floor: float):
        log_ent_coef = getattr(self.model, "log_ent_coef", None)
        if log_ent_coef is None:
            return None, None
        with torch.no_grad():
            before = float(torch.exp(log_ent_coef.detach()).mean().cpu().item())
            if before < floor:
                log_ent_coef.data.fill_(float(np.log(floor)))
            after = float(torch.exp(log_ent_coef.detach()).mean().cpu().item())
        return before, after

    def _activate_boost(self, level: int):
        if self.boost_state is None or Config.level_transition_boost_steps <= 0:
            return
        self.boost_state.trigger(self.num_timesteps, level, Config.level_transition_boost_steps)
        if hasattr(self.model, "sde_sample_freq"):
            self.model.sde_sample_freq = Config.level_transition_sde_sample_freq
        ent_before, ent_after = self._set_entropy_floor(Config.level_transition_ent_coef_floor)
        ent_msg = ""
        if ent_before is not None:
            ent_msg = f" | ent_coef {ent_before:.4f}->{ent_after:.4f}"
        print(
            f"[Curriculum] exploration boost: level={level} "
            f"until_step={self.boost_state.boost_until_step} "
            f"sde_freq={Config.level_transition_sde_sample_freq}{ent_msg}"
        )

    def _maintain_boost(self):
        if self.boost_state is None:
            return
        active = self.boost_state.is_active(self.num_timesteps)
        if active:
            if hasattr(self.model, "sde_sample_freq"):
                self.model.sde_sample_freq = Config.level_transition_sde_sample_freq
            self.logger.record("curriculum/level_transition_boost", 1)
            self.logger.record("curriculum/boost_remaining_steps", self.boost_state.remaining(self.num_timesteps))
            self.logger.record("curriculum/boost_level", self.boost_state.boost_level)
        else:
            if self.boost_was_active and self.base_sde_sample_freq is not None and hasattr(self.model, "sde_sample_freq"):
                self.model.sde_sample_freq = self.base_sde_sample_freq
                print(f"[Curriculum] exploration boost ended; sde_freq={self.base_sde_sample_freq}")
            self.logger.record("curriculum/level_transition_boost", 0)
        self.boost_was_active = active

    def _on_step(self) -> bool:
        for info in self.locals['infos']:
            if 'curriculum_level' in info:
                level = int(info['curriculum_level'])
                self.logger.record("curriculum/level", level)
                if self.last_level is None:
                    self.last_level = level
                elif level != self.last_level:
                    previous_level = self.last_level
                    self.last_level = level
                    print(f"[Curriculum] Level changed {previous_level} -> {level}")
                    if level > previous_level:
                        self._activate_boost(level)
        self._maintain_boost()
        # Entropy floor 保护: 防止 ent_coef 坍塌到接近 0
        if hasattr(self.model, 'log_ent_coef'):
            ent_coef_val = float(torch.exp(self.model.log_ent_coef).item())
            if ent_coef_val < 0.15:
                self.model.log_ent_coef.data.fill_(np.log(0.15))
        return True


class FailureAnalysisCallback(BaseCallback):
    def __init__(self, save_dir, check_freq=5000, verbose=1):
        super().__init__(verbose)
        self.save_dir = os.path.join(save_dir, "analysis_plots")
        os.makedirs(self.save_dir, exist_ok=True)
        self.check_freq = check_freq
        self.history_len = 1000
        self.reasons = deque(maxlen=self.history_len)
        self.positions = deque(maxlen=self.history_len)

    def _on_step(self) -> bool:
        for info in self.locals['infos']:
            if 'termination_reason' in info:
                reason = info['termination_reason']
                if reason != "running":
                    self.reasons.append(reason)
                    self.positions.append((info['robot_pos'], reason))
        if self.n_calls % self.check_freq == 0 and len(self.reasons) > 0:
            self._generate_report()
        return True

    def _generate_report(self):
        total = len(self.reasons)
        counts = {
            "success": self.reasons.count("success"),
            "collision": self.reasons.count("collision"),
            "fall": self.reasons.count("fall"),
            "timeout": self.reasons.count("timeout"),
            "stuck": self.reasons.count("stuck")
        }
        self.logger.record("analysis/success_rate", counts['success'] / total)
        self.logger.record("analysis/collision_rate", counts['collision'] / total)
        self.logger.record("analysis/stuck_rate", counts['stuck'] / total)
        self.logger.record("analysis/timeout_rate", counts['timeout'] / total)
        self._plot_death_map(counts)

    def _plot_death_map(self, counts):
        fig, ax = plt.subplots(figsize=(10, 10))
        coll_x, coll_y, fall_x, fall_y, time_x, time_y, stuck_x, stuck_y, succ_x, succ_y = [], [], [], [], [], [], [], [], [], []
        for pos, reason in self.positions:
            if reason == 'collision': coll_x.append(pos[0]); coll_y.append(pos[1])
            elif reason == 'fall': fall_x.append(pos[0]); fall_y.append(pos[1])
            elif reason == 'timeout': time_x.append(pos[0]); time_y.append(pos[1])
            elif reason == 'stuck': stuck_x.append(pos[0]); stuck_y.append(pos[1])
            elif reason == 'success': succ_x.append(pos[0]); succ_y.append(pos[1])
        ax.scatter(coll_x, coll_y, c='red', marker='x', label=f'Collision ({counts["collision"]})', alpha=0.6)
        ax.scatter(fall_x, fall_y, c='orange', marker='^', label=f'Fall ({counts["fall"]})', alpha=0.6)
        ax.scatter(time_x, time_y, c='blue', marker='o', label=f'Timeout ({counts["timeout"]})', alpha=0.3)
        ax.scatter(stuck_x, stuck_y, c='purple', marker='d', label=f'Stuck ({counts["stuck"]})', alpha=0.4)
        ax.scatter(succ_x, succ_y, c='green', marker='*', label=f'Success ({counts["success"]})', alpha=0.3)
        ax.set_xlim(-10, 10); ax.set_ylim(-10, 10)
        ax.set_title(f"Termination Map (Step {self.num_timesteps}) - End2End")
        ax.legend(); ax.grid(True)
        plt.savefig(os.path.join(self.save_dir, f"death_map_{self.num_timesteps}.png"))
        plt.close(fig)


class DetailedRewardAnalysisCallback(BaseCallback):
    def __init__(self, log_freq=100, verbose=1):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.reward_buffer = defaultdict(list)

    def _on_step(self) -> bool:
        for info in self.locals['infos']:
            if 'rewards' in info:
                for key, value in info['rewards'].items():
                    self.reward_buffer[key].append(value)
                if 'distance_to_goal' in info:
                    self.reward_buffer['Dist'].append(info['distance_to_goal'])
        if self.n_calls % self.log_freq == 0:
            self._log_and_print_stats()
            self.reward_buffer = defaultdict(list)
        return True

    def _log_and_print_stats(self):
        sorted_keys = sorted(self.reward_buffer.keys())
        log_items = []
        for key in sorted_keys:
            values = self.reward_buffer[key]
            if not values: continue
            mean_val = np.mean(values)
            if "Dist" in key:
                self.logger.record(f"analysis_metrics/{key}", mean_val)
            else:
                self.logger.record(f"detailed_rewards/{key}_mean", mean_val)
            log_items.append(f"{key}: {mean_val:.3f}")
        print(f"[Step: {self.num_timesteps:<8}] | " + " | ".join(log_items))


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func


def warmup_linear_schedule(initial_value: float, warmup_frac: float = 0.1) -> Callable[[float], float]:
    """Warmup + linear decay: 前 10% 线性升温到 initial_value，之后线性衰减。"""
    def func(progress_remaining: float) -> float:
        progress = 1.0 - progress_remaining
        if progress < warmup_frac:
            # Warmup: 从 initial_value*0.1 线性升到 initial_value
            warmup_ratio = 0.1 + 0.9 * (progress / warmup_frac)
            return initial_value * warmup_ratio * progress_remaining
        return progress_remaining * initial_value
    return func


def make_env_fn(rank, seed, env_kwargs):
    def _init():
        try:
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        env = RobotVisualEnd2EndEnv(**env_kwargs)
        env = StateHistoryWrapper(env, Config.history_length, Config.state_feature_dim)
        env.reset(seed=seed + rank)
        return env
    return _init


# 6. Teacher 数据和 BC 训练
def make_single_env(render_mode=None):
    env = RobotVisualEnd2EndEnv(
        model_path=Config.model_xml,
        low_level_policy_path=Config.policy_path,
        render_mode=render_mode,
        render_decimation=Config.decimation,
        action_repeat=Config.action_repeat,
        enable_dynamic_obstacles=Config.enable_dynamic_obstacles,
        action_mode=Config.action_mode,
    )
    return StateHistoryWrapper(env, Config.history_length, Config.state_feature_dim)


def collect_teacher_dataset(dataset_path):
    print(f"--> [Stage 1] 采集 planner teacher 数据: {dataset_path}")
    os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
    env = make_single_env(render_mode=None)
    grids, states, actions, levels, episode_success = [], [], [], [], []
    episode_meta = []

    try:
        for level in range(len(env.unwrapped.curriculum.LEVELS)):
            env.unwrapped.curriculum.current_level = level
            level_name = env.unwrapped.curriculum.config["name"]
            num_episodes = Config.teacher_episodes_per_level
            if level >= Config.teacher_hard_start_level:
                num_episodes += Config.teacher_hard_extra_episodes
            max_steps = Config.teacher_max_steps_per_episode
            if level >= Config.teacher_hard_start_level:
                max_steps = Config.teacher_hard_max_steps_per_episode
            successes = 0
            level_transitions = 0
            for ep in range(num_episodes):
                env.unwrapped.curriculum.current_level = level
                env.unwrapped.curriculum.success_buffer.clear()
                obs, _ = env.reset(seed=Config.seed + level * 1000 + ep)
                done = False
                steps = 0
                ep_grids, ep_states, ep_actions = [], [], []
                info = {}
                while not done and steps < max_steps:
                    teacher_action = env.unwrapped.get_teacher_action()
                    ep_grids.append(obs["grid_map"].copy())
                    ep_states.append(obs["state_history"].copy())
                    ep_actions.append(teacher_action.copy())
                    obs, _, terminated, truncated, info = env.step(teacher_action)
                    done = terminated or truncated
                    steps += 1
                reason = info.get("termination_reason", "teacher_timeout" if steps >= max_steps else "unknown")
                success = reason == "success"
                successes += int(success)
                level_transitions += len(ep_actions)
                grids.extend(ep_grids)
                states.extend(ep_states)
                actions.extend(ep_actions)
                levels.extend([level] * len(ep_actions))
                episode_success.extend([success] * len(ep_actions))
                episode_meta.append((level, steps, reason))
                print(
                    f"    [teacher] level={level} ({level_name}) "
                    f"ep={ep + 1}/{num_episodes} success={successes}/{ep + 1} "
                    f"max_steps={max_steps} steps={steps} reason={reason}"
                )
            print(
                f"    level {level} ({level_name}): episodes={num_episodes} "
                f"success={successes}/{num_episodes} max_steps={max_steps} transitions={level_transitions}"
            )
    finally:
        env.close()

    np.savez_compressed(
        dataset_path,
        grid_map=np.asarray(grids, dtype=np.uint8),
        state_history=np.asarray(states, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.float32),
        levels=np.asarray(levels, dtype=np.int64),
        episode_success=np.asarray(episode_success, dtype=np.bool_),
        meta=np.asarray(episode_meta, dtype=object),
        version=np.asarray([Config.teacher_dataset_version], dtype=np.int64),
    )
    print(f"--> Teacher 数据完成: {len(actions)} transitions")


def load_teacher_dataset(dataset_path):
    data = np.load(dataset_path, allow_pickle=True)
    dataset = {
        "grid_map": data["grid_map"],
        "state_history": data["state_history"],
        "actions": data["actions"],
    }
    n = len(dataset["actions"])
    dataset["levels"] = data["levels"] if "levels" in data else np.zeros(n, dtype=np.int64)
    dataset["episode_success"] = data["episode_success"] if "episode_success" in data else np.zeros(n, dtype=np.bool_)
    return dataset


def teacher_dataset_is_current(dataset_path):
    if not os.path.exists(dataset_path):
        return False
    try:
        data = np.load(dataset_path, allow_pickle=True)
        version = int(data["version"][0]) if "version" in data else -1
        has_hard_labels = "levels" in data and "episode_success" in data
        return version == Config.teacher_dataset_version and has_hard_labels
    except Exception:
        return False


def bc_update(model, dataset, batch_size, weight=1.0, optimizer=None):
    n = len(dataset["actions"])
    if n == 0:
        return 0.0
    levels = dataset.get("levels")
    successes = dataset.get("episode_success")
    if levels is not None and len(levels) == n:
        max_level = int(np.max(levels)) if n else 0
        hard_mask = levels >= max(0, max_level - 1)
        success_mask = successes.astype(bool) if successes is not None and len(successes) == n else np.zeros(n, dtype=bool)
        hard_idx = np.flatnonzero(hard_mask)
        hard_success_idx = np.flatnonzero(hard_mask & success_mask)
        n_hard_success = min(batch_size, int(batch_size * Config.bc_success_sample_fraction))
        n_hard = min(batch_size - n_hard_success, int(batch_size * Config.bc_hard_sample_fraction))
        n_any = batch_size - n_hard_success - n_hard
        parts = []
        if len(hard_success_idx) > 0 and n_hard_success > 0:
            parts.append(np.random.choice(hard_success_idx, size=n_hard_success, replace=True))
        else:
            n_any += n_hard_success
        if len(hard_idx) > 0 and n_hard > 0:
            parts.append(np.random.choice(hard_idx, size=n_hard, replace=True))
        else:
            n_any += n_hard
        if n_any > 0:
            parts.append(np.random.randint(0, n, size=n_any))
        indices = np.concatenate(parts)
        np.random.shuffle(indices)
    else:
        indices = np.random.randint(0, n, size=batch_size)
    device = model.device
    obs_tensor = {
        "grid_map": torch.as_tensor(dataset["grid_map"][indices], device=device),
        "state_history": torch.as_tensor(dataset["state_history"][indices], device=device),
    }
    target_actions_np = dataset["actions"][indices].astype(np.float32)
    if getattr(model.policy, "squash_output", False):
        target_actions_np = model.policy.scale_action(target_actions_np)
    target_actions = torch.as_tensor(target_actions_np, dtype=torch.float32, device=device)
    actor = model.policy.actor
    opt = optimizer if optimizer is not None else actor.optimizer

    was_training = model.policy.training
    model.policy.set_training_mode(True)
    try:
        pred_actions = actor(obs_tensor, deterministic=True)
        loss = torch.nn.functional.mse_loss(pred_actions, target_actions) * weight
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
        opt.step()
        return float(loss.detach().cpu().item())
    finally:
        model.policy.set_training_mode(was_training)


def pretrain_actor_with_bc(model, dataset):
    if Config.bc_pretrain_updates <= 0:
        return
    print(f"--> [Stage 2] BC 预训练 actor: {Config.bc_pretrain_updates} updates")
    optimizer = torch.optim.Adam(model.policy.actor.parameters(), lr=Config.bc_lr)
    loss_window = deque(maxlen=100)
    for update in range(1, Config.bc_pretrain_updates + 1):
        loss = bc_update(model, dataset, Config.bc_batch_size, optimizer=optimizer)
        loss_window.append(loss)
        if update % 200 == 0:
            print(f"    BC update {update}/{Config.bc_pretrain_updates} | loss={np.mean(loss_window):.5f}")


class BCRegularizationCallback(BaseCallback):
    def __init__(self, dataset, boost_state=None, verbose=0):
        super().__init__(verbose)
        self.dataset = dataset
        self.boost_state = boost_state

    def _on_step(self) -> bool:
        if Config.bc_regularization_updates <= 0:
            return True
        boost_active = self.boost_state is not None and self.boost_state.is_active(self.num_timesteps)
        freq = Config.level_transition_bc_freq if boost_active else Config.bc_regularization_freq
        updates = Config.level_transition_bc_updates if boost_active else Config.bc_regularization_updates
        weight = Config.bc_regularization_weight
        if boost_active:
            weight *= Config.level_transition_bc_weight_multiplier
        if self.n_calls % freq != 0:
            return True
        losses = []
        for _ in range(updates):
            losses.append(bc_update(
                self.model,
                self.dataset,
                Config.bc_batch_size,
                weight=weight,
            ))
        if losses:
            self.logger.record("train/bc_regularization_loss", float(np.mean(losses)))
            self.logger.record("train/bc_regularization_weight", weight)
            self.logger.record("curriculum/bc_boost_active", int(boost_active))
        return True


def maybe_compile_policy_modules(model):
    if not Config.enable_torch_compile:
        return
    if not hasattr(torch, "compile"):
        print("--> torch.compile 不可用，跳过编译加速")
        return

    compiled = []
    for module_name in ("actor", "critic", "critic_target"):
        module = getattr(model.policy, module_name, None)
        if module is None:
            continue
        try:
            try:
                module.forward = torch.compile(
                    module.forward,
                    mode=Config.torch_compile_mode,
                    fullgraph=False,
                    dynamic=True,
                )
            except TypeError:
                module.forward = torch.compile(module.forward, mode=Config.torch_compile_mode)
            compiled.append(module_name)
        except Exception as exc:
            print(f"--> torch.compile 跳过 {module_name}: {exc}")

    if compiled:
        print(f"--> torch.compile 已启用: {', '.join(compiled)}")


# 6. 主函数
def main():
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    num_cpu = multiprocessing.cpu_count()

    teacher_dataset = None
    if Config.enable_teacher_dataset:
        dataset_path = Config.teacher_dataset_path
        if not teacher_dataset_is_current(dataset_path):
            if os.path.exists(dataset_path):
                print(f"--> [Stage 1] teacher 数据版本过旧，重新采集: {dataset_path}")
            collect_teacher_dataset(dataset_path)
        else:
            print(f"--> [Stage 1] 复用已有 teacher 数据: {dataset_path}")
        teacher_dataset = load_teacher_dataset(dataset_path)
        print(f"--> Teacher transitions: {len(teacher_dataset['actions'])}")

    if Config.render:
        actual_num_envs = 1
        use_subproc = False
        render_mode = 'human'
        print(f"--> [Mode] 观看模式 (GUI)")
    else:
        actual_num_envs = max(1, min(Config.num_envs, num_cpu - 2))
        use_subproc = True
        render_mode = None
        print(f"--> [Mode] 训练模式: {actual_num_envs} 并行环境")

    env_kwargs = {
        "model_path": Config.model_xml,
        "low_level_policy_path": Config.policy_path,
        "render_mode": render_mode,
        "render_decimation": Config.decimation,
        "action_repeat": Config.action_repeat,
        "enable_dynamic_obstacles": Config.enable_dynamic_obstacles,
        "action_mode": Config.action_mode,
    }

    # VecEnv
    env_fns = [make_env_fn(i, Config.seed, env_kwargs) for i in range(actual_num_envs)]
    if use_subproc and actual_num_envs > 1:
        env = SubprocVecEnv(env_fns)
    else:
        env = DummyVecEnv(env_fns)
    env = VecMonitor(env, os.path.join(Config.log_dir, "train_monitor"))

    # 评估环境
    eval_env_kwargs = env_kwargs.copy()
    eval_env_kwargs['render_mode'] = None
    eval_env = DummyVecEnv([make_env_fn(999, Config.seed + 999, eval_env_kwargs)])
    eval_env = VecMonitor(eval_env, os.path.join(Config.log_dir, "eval_monitor"))

    # 回调
    eval_callback = EvalCallback(
        eval_env, best_model_save_path=os.path.join(Config.log_dir, "best_model"),
        log_path=os.path.join(Config.log_dir, "eval_logs"),
        eval_freq=max(500000 // actual_num_envs, 1),
        deterministic=True, render=False, n_eval_episodes=10
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=max(20000000 // actual_num_envs, 1),
        save_path=os.path.join(Config.log_dir, "checkpoints/"),
        name_prefix="sac_end2end", save_replay_buffer=False, verbose=1
    )
    fail_callback = FailureAnalysisCallback(save_dir=Config.log_dir, check_freq=500000)
    reward_callback = DetailedRewardAnalysisCallback(log_freq=500)
    boost_state = LevelTransitionBoostState()
    curriculum_callback = CurriculumCallback(boost_state=boost_state)

    callbacks = [eval_callback, checkpoint_callback, fail_callback, reward_callback, curriculum_callback]
    if teacher_dataset is not None:
        callbacks.append(BCRegularizationCallback(teacher_dataset, boost_state=boost_state))

    # 网络
    policy_kwargs = dict(
        features_extractor_class=End2EndExtractor,
        features_extractor_kwargs=dict(
            state_feature_dim=Config.state_feature_dim,
            history_length=Config.history_length,
            d_model=128,
            features_dim=256,
        ),
        net_arch=dict(pi=[256, 256], qf=[256, 256]),
        share_features_extractor=False
    )

    lr_schedule = warmup_linear_schedule(Config.lr)

    # 加载或新建
    if Config.resume_from and os.path.exists(Config.resume_from):
        print(f"--> 恢复训练: {Config.resume_from}")
        model = SAC.load(Config.resume_from, env=env, learning_rate=lr_schedule, device="cuda",
                         custom_objects={"policy_kwargs": policy_kwargs})
    else:
        print("--> 初始化全新 SAC 模型 (End-to-End with Curriculum)")
        model = SAC(
            "MultiInputPolicy", env,
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log=Config.log_dir,
            learning_rate=lr_schedule,
            buffer_size=5_000_000,
            batch_size=1024,
            gamma=0.993,
            tau=0.001,
            ent_coef='auto_0.5',
            use_sde=True,
            use_sde_at_warmup=True,
            sde_sample_freq=8,
            train_freq=(100, "step"),
            gradient_steps=10,
            learning_starts=50000,
            device="cuda"
        )

    if teacher_dataset is not None and not Config.resume_from:
        pretrain_actor_with_bc(model, teacher_dataset)

    maybe_compile_policy_modules(model)

    print(f"--- [Stage 3] 开始 SAC + BC regularization 训练 (Total Steps: {Config.total_timesteps}) ---")
    try:
        model.learn(
            total_timesteps=Config.total_timesteps,
            callback=callbacks,
            tb_log_name="SAC_End2End_Curriculum",
            reset_num_timesteps=(Config.resume_from is None),
            progress_bar=True
        )
        final_path = os.path.join(Config.log_dir, "sac_end2end_final.zip")
        model.save(final_path)
        model.save(Config.final_playable_path)
        print(f"训练完成。最终模型保存至: {final_path}")
        print(f"Playable 模型保存至: {Config.final_playable_path}")
    except KeyboardInterrupt:
        print("\n训练被用户中断。正在保存当前模型...")
        model.save(os.path.join(Config.log_dir, "sac_end2end_interrupted.zip"))
    finally:
        env.close()
        eval_env.close()


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    os.makedirs(Config.log_dir, exist_ok=True)
    torch.backends.cudnn.benchmark = True
    main()
