"""Train SAC navigation with GRU-to-map transformer cross-attention."""

from __future__ import annotations

import multiprocessing
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecMonitor,
    VecNormalize,
)

from difficulty_balanced_replay_buffer import DifficultyBalancedDictReplayBuffer
from robot_visual_env_random_map_transformer import RobotVisualEnv
from train_visual_random_map import (
    Config as RandomMapConfig,
    DetailedRewardAnalysisCallback,
    FailureAnalysisCallback,
    linear_schedule,
)
from transformer_fusion_extractor import CrossAttentionFusionExtractor


class Config(RandomMapConfig):
    """Keep the original experiment settings and separate transformer outputs."""

    log_dir = "./sac_lidar_logs_random_transformer/"
    resume_from = None

    # T3 baseline: less conservative than T5, but still stable enough for the
    # transformer critic.
    replay_buffer_size = 3_000_000
    batch_size = 512
    checkpoint_freq = 500_000
    min_free_gpu_memory_gib = 2.0

    normalize_rewards = True
    reward_clip = 10.0
    resume_vecnormalize = None

    difficulty_balanced_replay = True
    difficulty_replay_bins = (5.0, 8.0, 12.0, 16.0)

    transformer_d_model = 128
    transformer_num_heads = 4
    transformer_ffn_dim = 256
    transformer_dropout = 0.05


def make_env_fn(rank, seed, env_kwargs):
    def _init():
        env = RobotVisualEnv(**env_kwargs)
        env.reset(seed=seed + rank)
        return env

    return _init


def main():
    num_cpu = multiprocessing.cpu_count()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by this training configuration.")

    free_gpu_bytes, total_gpu_bytes = torch.cuda.mem_get_info()
    gib = 1024**3
    print(
        "--> [GPU] "
        f"{free_gpu_bytes / gib:.2f} GiB free / "
        f"{total_gpu_bytes / gib:.2f} GiB total"
    )
    if free_gpu_bytes < Config.min_free_gpu_memory_gib * gib:
        raise RuntimeError(
            "Insufficient free GPU memory for transformer SAC training. "
            f"At least {Config.min_free_gpu_memory_gib:.1f} GiB is required. "
            "Stop other CUDA training/simulation processes and retry."
        )

    if Config.render:
        actual_num_envs = 1
        use_subproc = False
        render_mode = "human"
        print("--> [Mode] Transformer watch mode: one GUI environment")
    else:
        actual_num_envs = max(1, min(Config.num_envs, num_cpu - 2))
        use_subproc = True
        render_mode = None
        print(f"--> [Mode] Transformer training mode: {actual_num_envs} environments")

    env_kwargs = {
        "model_path": Config.model_xml,
        "low_level_policy_path": Config.policy_path,
        "render_mode": render_mode,
        "render_decimation": Config.decimation,
        "action_repeat": Config.action_repeat,
        "history_length": Config.history_length,
        "enable_dynamic_obstacles": Config.enable_dynamic_obstacles,
    }

    env_fns = [
        make_env_fn(i, Config.seed, env_kwargs) for i in range(actual_num_envs)
    ]
    if use_subproc and actual_num_envs > 1:
        env = SubprocVecEnv(env_fns)
    else:
        env = DummyVecEnv(env_fns)
    env = VecMonitor(env, os.path.join(Config.log_dir, "train_monitor"))
    if Config.normalize_rewards:
        if Config.resume_vecnormalize and os.path.exists(Config.resume_vecnormalize):
            print(f"--> Loading VecNormalize stats: {Config.resume_vecnormalize}")
            env = VecNormalize.load(Config.resume_vecnormalize, env)
            env.training = True
            env.norm_obs = False
            env.norm_reward = True
        else:
            env = VecNormalize(
                env,
                training=True,
                norm_obs=False,
                norm_reward=True,
                clip_reward=Config.reward_clip,
                gamma=0.993,
            )

    eval_env_kwargs = env_kwargs.copy()
    eval_env_kwargs["render_mode"] = None
    eval_env = DummyVecEnv(
        [make_env_fn(999, Config.seed + 999, eval_env_kwargs)]
    )
    eval_env = VecMonitor(
        eval_env, os.path.join(Config.log_dir, "eval_monitor")
    )
    if Config.normalize_rewards:
        eval_env = VecNormalize(
            eval_env,
            training=False,
            norm_obs=False,
            norm_reward=False,
            clip_reward=Config.reward_clip,
            gamma=0.993,
        )

    replay_buffer_class = None
    replay_buffer_kwargs = None
    if Config.difficulty_balanced_replay:
        replay_buffer_class = DifficultyBalancedDictReplayBuffer
        replay_buffer_kwargs = {
            "difficulty_key": "difficulty_path_len",
            "bin_edges": Config.difficulty_replay_bins,
        }

    callbacks = [
        EvalCallback(
            eval_env,
            best_model_save_path=os.path.join(Config.log_dir, "best_model"),
            log_path=os.path.join(Config.log_dir, "eval_logs"),
            eval_freq=max(500000 // actual_num_envs, 1),
            deterministic=True,
            render=False,
            n_eval_episodes=5,
        ),
        CheckpointCallback(
            save_freq=max(Config.checkpoint_freq // actual_num_envs, 1),
            save_path=os.path.join(Config.log_dir, "checkpoints"),
            name_prefix="sac_lidar_random_transformer",
            save_replay_buffer=False,
            save_vecnormalize=Config.normalize_rewards,
            verbose=1,
        ),
        FailureAnalysisCallback(
            save_dir=Config.log_dir, check_freq=20000000
        ),
        DetailedRewardAnalysisCallback(log_freq=500),
    ]

    policy_kwargs = {
        "features_extractor_class": CrossAttentionFusionExtractor,
        "features_extractor_kwargs": {
            "state_feature_dim": Config.state_feature_dim,
            "history_length": Config.history_length,
            "d_model": Config.transformer_d_model,
            "num_heads": Config.transformer_num_heads,
            "ffn_dim": Config.transformer_ffn_dim,
            "features_dim": 256,
            "dropout": Config.transformer_dropout,
        },
        "net_arch": {"pi": [256, 256], "qf": [256, 256]},
        "share_features_extractor": False,
    }

    lr_schedule = linear_schedule(Config.lr)

    if Config.resume_from and os.path.exists(Config.resume_from):
        print(f"--> Resuming transformer model: {Config.resume_from}")
        model = SAC.load(
            Config.resume_from,
            env=env,
            learning_rate=lr_schedule,
            device="cuda",
            custom_objects={
                "policy_kwargs": policy_kwargs,
                "learning_rate": lr_schedule,
                "replay_buffer_class": replay_buffer_class,
                "replay_buffer_kwargs": replay_buffer_kwargs,
            },
        )
    else:
        print(
            "--> Initializing SAC with state-query/map-key-value "
            "cross-attention"
        )
        model = SAC(
            "MultiInputPolicy",
            env,
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log=Config.log_dir,
            learning_rate=2e-5,
            buffer_size=Config.replay_buffer_size,
            batch_size=Config.batch_size,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            gamma=0.993,
            tau=0.0005,
            ent_coef="auto",
            train_freq=(100, "step"),
            gradient_steps=15,
            learning_starts=5000,
            device="cuda",
        )

    print(
        f"--- Transformer training started "
        f"(total steps: {Config.total_timesteps}) ---"
    )
    try:
        model.learn(
            total_timesteps=Config.total_timesteps,
            callback=callbacks,
            tb_log_name="SAC_Lidar_Random_Map_Transformer",
            reset_num_timesteps=Config.resume_from is None,
            progress_bar=True,
        )
        final_path = os.path.join(
            Config.log_dir, "sac_lidar_transformer_final.zip"
        )
        model.save(final_path)
        if Config.normalize_rewards and model.get_vec_normalize_env() is not None:
            model.get_vec_normalize_env().save(
                os.path.join(Config.log_dir, "vecnormalize_final.pkl")
            )
        print(f"Training complete. Model saved to: {final_path}")
    except KeyboardInterrupt:
        interrupted_path = os.path.join(
            Config.log_dir, "sac_lidar_transformer_interrupted.zip"
        )
        print(f"\nTraining interrupted. Saving to: {interrupted_path}")
        model.save(interrupted_path)
        if Config.normalize_rewards and model.get_vec_normalize_env() is not None:
            model.get_vec_normalize_env().save(
                os.path.join(Config.log_dir, "vecnormalize_interrupted.pkl")
            )
    finally:
        env.close()
        eval_env.close()


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    os.makedirs(Config.log_dir, exist_ok=True)
    torch.backends.cudnn.benchmark = True
    main()
