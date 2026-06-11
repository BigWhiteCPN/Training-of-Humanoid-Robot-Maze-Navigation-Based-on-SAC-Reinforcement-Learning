import os
import time
import multiprocessing
import sys
from collections import deque
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Tuple, Union

# 1. 全局配置中心 (必须放在 import matplotlib 之前！)
class Config:
    # --- 训练设置 ---
    # 并行环境数量 (训练时建议 4~16，观看时会自动变为 1)
    num_envs = 1                
    total_timesteps = 2_000_000 
    lr = 3e-4                   
    seed = 42                   
    
    # --- 显示与调试模式开关 ---
    # True  = 观看模式 (有窗口，单进程，能看到机器人动)
    # False = 训练模式 (无窗口，多进程，速度快，后台绘图)
    render = False              
    
    # resume_from = None
    resume_from = "./sac_lidar_logs/best_model.zip"
    
    # --- 路径设置 ---
    # 请确保路径正确
    model_xml = "/home/iansten/code/IsaacLabExtensionTemplate/scripts/resources/mjcf/Linnxil_fifteen_angle_bs_copy_20260302.xml"
    policy_path = "/home/iansten/code/IsaacLabExtensionTemplate/scripts/visual_train/policy_20251026.pt"
    log_dir = "./sac_lidar_logs/"
    
    # --- 环境参数 ---
    decimation = 50             
    history_length = 15         
    state_feature_dim = 9       
    # 注意: vision_feature_dim 已移除，因为不再使用图像
    
    # 动态障碍物配置
    enable_dynamic_obstacles = False 

# 2. 动态设置 Matplotlib 后端 (防止崩溃的关键)
import matplotlib

if not Config.render:
    # 【训练模式】：强制使用 Agg 后端
    # Agg 是非交互式的，不依赖 X11 或 GUI 线程，绝对不会崩溃，但无法弹窗。
    print("--> [System] 训练模式: 启用 Agg 后端 (无窗口，安全稳定)")
    matplotlib.use('Agg') 
else:
    # 【观看模式】：使用默认 GUI 后端 (如 TkAgg, Qt5Agg)
    print("--> [System] 观看模式: 启用 GUI 后端 (有窗口)")
    # matplotlib.use('TkAgg') # 如有需要可取消注释

import matplotlib.pyplot as plt

# 3. 其他依赖导入
import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback, EvalCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# 确保 robot_visual_env.py 在路径下
from robot_visual_env import RobotVisualEnv

# 4. 自定义网络结构 (Grid Map CNN + State GRU)
class SequenceFusionExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, 
                 state_feature_dim=9, 
                 history_length=15,
                 d_model=256):
        # features_dim 是最终输出给 SAC Policy 的特征维度
        features_dim = 512
        super().__init__(observation_space, features_dim)
        # 1. 栅格地图处理 CNN (动态计算维度)
        n_input_channels = observation_space['grid_map'].shape[0] 
        
        # 定义卷积层主体 (加宽通道数)
        self.cnn_body = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=5, stride=2, padding=2), # 16 -> 32
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),               # 32 -> 64
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),              # 64 -> 128
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # 动态计算 Linear 输入大小
        with torch.no_grad():
            # 创建一个假输入来推导 Flatten 后的维度
            sample_input = torch.zeros(1, *observation_space['grid_map'].shape)
            n_flatten = self.cnn_body(sample_input).shape[1]
            
        # 完整的 Map CNN
        self.map_cnn = nn.Sequential(
            self.cnn_body,
            nn.Linear(n_flatten, 256),
            nn.LayerNorm(256),
            nn.ReLU()
        )
        # 2. 状态序列处理 GRU
        self.state_sub_dim = state_feature_dim
        self.seq_len = history_length  
        
        self.state_embedding = nn.Linear(state_feature_dim, d_model)
        self.gru = nn.GRU(input_size=d_model, hidden_size=d_model, num_layers=2, batch_first=True)
        # 3. 融合层
        self.fusion_layer = nn.Sequential(
            nn.Linear(256 + d_model, features_dim),
            nn.LayerNorm(features_dim),
            nn.ReLU()
        )

    def forward(self, observations):
        # --- 处理 Grid Map ---
        grid_map = observations['grid_map']
        map_feat = self.map_cnn(grid_map)
        
        # --- 处理状态序列 ---
        batch_size = observations['state_history'].shape[0]
        
        # 将平铺的历史状态 reshape 回 (Batch, Length, Dim)
        state_seq = observations['state_history'].view(batch_size, self.seq_len, self.state_sub_dim)
        
        x = self.state_embedding(state_seq)
        x = torch.relu(x)
        gru_out, _ = self.gru(x)
        temporal_feat = gru_out[:, -1, :] # 取最后一个时间步
        
        # --- 融合 ---
        combined = torch.cat([map_feat, temporal_feat], dim=1)
        output = self.fusion_layer(combined)
        return output
    
# 5. 回调函数 (保持不变，用于分析训练状态)
class FailureAnalysisCallback(BaseCallback):
    """
    分析失败原因并绘制分布图。
    """
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
            "timeout": self.reasons.count("timeout")
        }
        
        # 记录到 TensorBoard
        self.logger.record("analysis/success_rate", counts['success'] / total)
        self.logger.record("analysis/collision_rate", counts['collision'] / total)
        
        self._plot_death_map(counts)

    def _plot_death_map(self, counts):
        fig, ax = plt.subplots(figsize=(10, 10))
        
        coll_x, coll_y = [], []
        fall_x, fall_y = [], []
        time_x, time_y = [], []
        succ_x, succ_y = [], []
        
        for pos, reason in self.positions:
            if reason == 'collision':
                coll_x.append(pos[0]); coll_y.append(pos[1])
            elif reason == 'fall':
                fall_x.append(pos[0]); fall_y.append(pos[1])
            elif reason == 'timeout':
                time_x.append(pos[0]); time_y.append(pos[1])
            elif reason == 'success':
                succ_x.append(pos[0]); succ_y.append(pos[1])

        ax.scatter(coll_x, coll_y, c='red', marker='x', label=f'Collision ({counts["collision"]})', alpha=0.6)
        ax.scatter(fall_x, fall_y, c='orange', marker='^', label=f'Fall ({counts["fall"]})', alpha=0.6)
        ax.scatter(time_x, time_y, c='blue', marker='o', label=f'Timeout ({counts["timeout"]})', alpha=0.3)
        ax.scatter(succ_x, succ_y, c='green', marker='*', label=f'Success ({counts["success"]})', alpha=0.3)

        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_title(f"Termination Map (Step {self.num_timesteps})")
        ax.legend()
        ax.grid(True)
        
        save_path = os.path.join(self.save_dir, f"death_map_{self.num_timesteps}.png")
        plt.savefig(save_path)
        plt.close(fig) 

class DetailedRewardAnalysisCallback(BaseCallback):
    def __init__(self, log_freq: int = 1000, verbose: int = 1):
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

        log_str = f"[Step: {self.num_timesteps:<8}] | " + " | ".join(log_items)
        print(log_str)

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func

# 辅助函数：创建环境
def make_env_fn(rank, seed, model_path, policy_path, render_mode=None, decimation=50, history_length=15, enable_dynamic_obstacles=False):
    def _init():
        env = RobotVisualEnv(
            model_path=model_path,
            low_level_policy_path=policy_path,
            render_mode=render_mode,
            render_decimation=decimation,
            history_length=history_length,
            enable_dynamic_obstacles=enable_dynamic_obstacles 
        )
        env.reset(seed=seed + rank) 
        return env
    return _init

# 6. 主函数
def main():
    num_cpu = multiprocessing.cpu_count()
    
    # ---------------------------------------------------------
    # 根据 Config.render 决定环境配置
    # ---------------------------------------------------------
    if Config.render:
        # 【观看模式】
        actual_num_envs = 1
        use_subproc = False 
        render_mode = 'human'
        print(f"--> [Mode] 观看模式 (GUI): 单环境, render_mode='human'")
    else:
        # 【训练模式】
        actual_num_envs = max(1, min(Config.num_envs, num_cpu - 2))
        use_subproc = True 
        render_mode = None 
        print(f"--> [Mode] 训练模式 (Headless): {actual_num_envs} 并行环境, render_mode=None")

    env_kwargs = {
        "model_path": Config.model_xml,
        "policy_path": Config.policy_path,
        "render_mode": render_mode,
        "decimation": Config.decimation,
        "history_length": Config.history_length,
        "enable_dynamic_obstacles": Config.enable_dynamic_obstacles
    }

    # 创建 VecEnv
    if use_subproc and actual_num_envs > 1:
        env = SubprocVecEnv([make_env_fn(i, Config.seed, **env_kwargs) for i in range(actual_num_envs)])
    else:
        env = DummyVecEnv([make_env_fn(0, Config.seed, **env_kwargs)])

    # 包装 Monitor
    env = VecMonitor(env, os.path.join(Config.log_dir, "train_monitor"))

    # 创建评估环境
    eval_env_kwargs = env_kwargs.copy()
    eval_env_kwargs['render_mode'] = None 
    
    eval_env = DummyVecEnv([make_env_fn(999, Config.seed + 999, **eval_env_kwargs)])
    eval_env = VecMonitor(eval_env, os.path.join(Config.log_dir, "eval_monitor"))
    
    # 配置回调
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(Config.log_dir, "best_model"),
        log_path=os.path.join(Config.log_dir, "eval_logs"),
        eval_freq=max(100000 // actual_num_envs, 1), 
        deterministic=True,
        render=False,
        n_eval_episodes=5
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=max(200000 // actual_num_envs, 1),
        save_path=os.path.join(Config.log_dir, "checkpoints/"),
        name_prefix="sac_lidar",
        save_replay_buffer=False,
        verbose=1
    )

    fail_analysis_callback = FailureAnalysisCallback(save_dir=Config.log_dir, check_freq=20000)
    analysis_callback = DetailedRewardAnalysisCallback(log_freq=100)
    
    callbacks = [eval_callback, checkpoint_callback, fail_analysis_callback, analysis_callback]

    # 网络参数 - 注意：不再传递 vision_feature_dim
    policy_kwargs = dict(
        features_extractor_class=SequenceFusionExtractor,
        features_extractor_kwargs=dict(
            state_feature_dim=Config.state_feature_dim,
            history_length=Config.history_length,
            d_model=256
        ),
        net_arch=dict(pi=[256, 256], qf=[256, 256]),
        share_features_extractor=False
    )
    
    lr_schedule = linear_schedule(Config.lr)

    # 加载或新建模型
    if Config.resume_from and os.path.exists(Config.resume_from):
        print(f"--> 正在从检查点恢复: {Config.resume_from}")
        model = SAC.load(Config.resume_from, env=env, learning_rate=lr_schedule, device="cuda", custom_objects=policy_kwargs)
    else:
        print("--> 初始化全新 SAC 模型 (With Lidar GridMap + GRU)")
        model = SAC(
            "MultiInputPolicy",
            env,
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log=Config.log_dir,
            learning_rate=1e-4,
            buffer_size=800_000,
            batch_size=256,
            gamma=0.99,
            tau=0.005,
            ent_coef='auto',
            # ent_coef=0.04,
            train_freq=(4, "step"), 
            gradient_steps=1,       
            learning_starts=5000,
            device="cuda"
        )

    print(f"--- 开始训练 (Total Steps: {Config.total_timesteps}) ---")
    try:
        model.learn(
            total_timesteps=Config.total_timesteps,
            callback=callbacks,
            tb_log_name="SAC_Lidar_GRU",
            reset_num_timesteps=(Config.resume_from is None),
            progress_bar=True
        )
        final_path = os.path.join(Config.log_dir, "sac_lidar_final.zip")
        model.save(final_path)
        print(f"训练完成。最终模型保存至: {final_path}")
        
    except KeyboardInterrupt:
        print("\n训练被用户中断。正在保存当前模型...")
        model.save(os.path.join(Config.log_dir, "sac_lidar_interrupted.zip"))
    finally:
        env.close()
        eval_env.close()

if __name__ == '__main__':
    # 显式设置启动方法，增加稳定性
    multiprocessing.set_start_method('spawn', force=True)
    
    os.makedirs(Config.log_dir, exist_ok=True)
    torch.backends.cudnn.benchmark = True
    main()
