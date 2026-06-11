import argparse
import time
import os
import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
import matplotlib

# GUI 后端设置
# 在导入环境之前设置 matplotlib 后端为 GUI 模式
try:
    matplotlib.use('TkAgg')
    print("--> Using TkAgg backend for visualization.")
except ImportError:
    try:
        matplotlib.use('Qt5Agg')
        print("--> Using Qt5Agg backend for visualization.")
    except:
        print("--> Warning: No interactive backend found. Visualization might fail.")

from stable_baselines3 import SAC
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from robot_visual_env import RobotVisualEnv

# --- Configuration ---
# 请确保这些路径与你的系统一致
MODEL_XML_PATH = "/home/iansten/code/IsaacLabExtensionTemplate/scripts/resources/mjcf/Linnxil_fifteen_angle_bs_copy_20260302.xml"
LOW_LEVEL_POLICY_PATH = "/home/iansten/code/IsaacLabExtensionTemplate/scripts/visual_train/policy_20251026.pt"

# 与训练配置保持一致
HISTORY_LENGTH = 15
STATE_FEATURE_DIM = 9
DECIMATION = 50

# 1. 自定义网络结构
class SequenceFusionExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, 
                 state_feature_dim=9, 
                 history_length=15,
                 d_model=128):
        
        # features_dim 是最终输出给 SAC Policy 的特征维度
        features_dim = 256
        super().__init__(observation_space, features_dim)
        # 1. 栅格地图处理 CNN
        # 自动获取输入通道数 (grid_map 通常为 1)
        n_input_channels = observation_space['grid_map'].shape[0] 
        
        # 定义卷积层主体
        self.cnn_body = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # 动态计算 Linear 输入大小 (关键修复：使用 dummy input 计算维度)
        with torch.no_grad():
            sample_input = torch.zeros(1, *observation_space['grid_map'].shape)
            n_flatten = self.cnn_body(sample_input).shape[1]
            
        # 完整的 Map CNN
        self.map_cnn = nn.Sequential(
            self.cnn_body,
            nn.Linear(n_flatten, 128),
            nn.LayerNorm(128),
            nn.ReLU()
        )
        # 2. 状态序列处理 GRU (必须与 train.py 一致)
        self.state_sub_dim = state_feature_dim
        self.seq_len = history_length
        
        self.state_embedding = nn.Linear(state_feature_dim, d_model)
        self.gru = nn.GRU(input_size=d_model, hidden_size=d_model, num_layers=2, batch_first=True)
        # 3. 融合层
        self.fusion_layer = nn.Sequential(
            nn.Linear(128 + d_model, features_dim),
            nn.LayerNorm(features_dim),
            nn.ReLU()
        )

    def forward(self, observations):
        # --- 处理 Grid Map ---
        # 注意：这里用的 key 是 'grid_map'，不是 'vision_features'
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

# 2. 主函数
def main(args):
    print("==================================================")
    print(f"Loading model from: {args.model_path}")
    print("==================================================")

    # 1. 初始化环境
    # 注意：render_mode='human' 会弹窗显示
    env = RobotVisualEnv(
        model_path=MODEL_XML_PATH,
        low_level_policy_path=LOW_LEVEL_POLICY_PATH,
        render_mode='human', 
        render_decimation=DECIMATION, # 必须与训练一致，否则物理步长不同步
        history_length=HISTORY_LENGTH,
        enable_dynamic_obstacles=args.enable_obstacles 
    )

    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at '{args.model_path}'")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 2. 加载模型
    try:
        # 如果加载 .zip (Stable Baselines3 保存的完整模型)
        if args.model_path.endswith(".zip"):
            print("Loading full SAC agent from .zip file...")
            
            # [关键] 传入 custom_objects，告诉 SB3 如何实例化我们的自定义提取器
            custom_objects = {
                "SequenceFusionExtractor": SequenceFusionExtractor,
                # 如果你的 Python 环境版本与训练时不同，可能还需要映射 learning_rate
                "learning_rate": 0.0, 
                "lr_schedule": lambda _: 0.0,
                "clip_range": lambda _: 0.0,
            }
            
            # 加载模型
            model = SAC.load(args.model_path, env=env, device=device, custom_objects=custom_objects)
        else:
            print("Error: Please provide a .zip file saved by Stable Baselines3.")
            return

        print("Model loaded successfully!")

        # 3. 运行推理循环
        episode_count = 0
        while True:
            episode_count += 1
            obs, info = env.reset()
            done = False
            episode_reward = 0
            episode_length = 0
            
            print(f"\n--- Episode {episode_count} Start ---")
            
            while not done:
                # [关键] deterministic=True 用于评估/演示，关闭随机探索
                action, _ = model.predict(obs, deterministic=True)
                
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                episode_reward += reward
                episode_length += 1
                
                # 只有在非常快的时候才需要 sleep，通常 Mujoco 的 human 模式会限制 FPS
                # time.sleep(0.005) 
            
            print(f"--- Episode Finished ---")
            print(f"Result: {'Success' if info.get('is_success') else 'Failed'}")
            print(f"Reason: {info.get('termination_reason', 'unknown')}")
            print(f"Distance to Goal: {info.get('distance_to_goal', 0.0):.2f}")
            print(f"Total Reward: {episode_reward:.2f}")
            
            # 每一轮结束后暂停一下，方便观察最终状态
            time.sleep(1.0)

    except KeyboardInterrupt:
        print("\nStopped by user.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Closing environment.")
        env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 只需要传入模型路径
    parser.add_argument("model_path", type=str, help="Path to best_model.zip or sac_lidar_final.zip")
    parser.add_argument("--enable_obstacles", action="store_true", help="Enable dynamic obstacles in playback")
    args = parser.parse_args()
    
    main(args)
