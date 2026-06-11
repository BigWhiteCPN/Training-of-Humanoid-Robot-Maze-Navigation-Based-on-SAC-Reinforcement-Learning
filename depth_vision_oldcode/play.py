import argparse
import time
import os
import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
import matplotlib

# 在导入 RobotVisualEnv 之前设置后端，防止某些系统上冲突
# 播放模式下我们需要 GUI 窗口
try:
    matplotlib.use('TkAgg')
except:
    pass # 如果失败则使用默认

from stable_baselines3 import SAC
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
# 确保 robot_visual_env.py 在同一目录下
from robot_visual_env import RobotVisualEnv

# --- Configuration ---
# 请根据实际路径修改 XML 和 Policy 路径
MODEL_XML_PATH = "/home/iansten/code/IsaacLabExtensionTemplate/scripts/resources/mjcf/Linnxil_fifteen_angle_bs_copy_20260302.xml"
LOW_LEVEL_POLICY_PATH = "/home/iansten/code/IsaacLabExtensionTemplate/scripts/visual_train/policy_20251026.pt"

# 必须与训练时的配置完全一致
HISTORY_LENGTH = 15
STATE_FEATURE_DIM = 9

# 1. 自定义网络结构 (必须与训练代码完全一致)
class SequenceFusionExtractor(BaseFeaturesExtractor):
    """
    必须包含这个类定义，并且结构必须与训练脚本中的完全一致，
    否则加载权重时会出现 shape mismatch 错误。
    """
    def __init__(self, observation_space: gym.spaces.Dict, 
                 state_feature_dim=9, 
                 history_length=15,
                 d_model=128):
        
        # 这里的 features_dim 是最终输出给 SAC Policy 的特征维度
        features_dim = 256
        super().__init__(observation_space, features_dim)
        
        # 1. 视觉处理 CNN (必须与训练代码一致)
        # 自动获取输入通道数 (Frame Stack)
        n_input_channels = observation_space['vision_features'].shape[0] # 通常是 3
        
        self.vision_cnn = nn.Sequential(
            # 输入: [Batch, 3, 48, 64]
            nn.Conv2d(n_input_channels, 16, kernel_size=5, stride=2, padding=2), 
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            # 计算 Flatten 后的维度: 64通道 * 6高 * 8宽 = 3072
            nn.Linear(3072, 128), 
            nn.ReLU()
        )

        # 2. 状态序列处理 (GRU)
        self.state_sub_dim = state_feature_dim
        self.seq_len = history_length
        
        self.state_embedding = nn.Linear(state_feature_dim, d_model)
        self.gru = nn.GRU(input_size=d_model, hidden_size=d_model, num_layers=2, batch_first=True)
        
        # 3. 融合层
        self.fusion_layer = nn.Sequential(
            nn.Linear(128 + d_model, features_dim),
            nn.ReLU()
        )

    def forward(self, observations):
        # --- 处理图像 ---
        vision_input = observations['vision_features']
        vis_feat = self.vision_cnn(vision_input)
        
        # --- 处理状态序列 ---
        batch_size = observations['state_history'].shape[0]
        state_seq = observations['state_history'].view(batch_size, self.seq_len, self.state_sub_dim)
        
        x = self.state_embedding(state_seq)
        x = torch.relu(x)
        gru_out, _ = self.gru(x)
        temporal_feat = gru_out[:, -1, :] # 取最后一个时间步
        
        # --- 融合 ---
        combined = torch.cat([vis_feat, temporal_feat], dim=1)
        output = self.fusion_layer(combined)
        return output

# 2. 主函数
def main(args):
    print("==================================================")
    print(f"Loading model/policy from: {args.model_path}")
    print("==================================================")

    # 1. 初始化环境 (参数需与训练一致)
    env = RobotVisualEnv(
        model_path=MODEL_XML_PATH,
        low_level_policy_path=LOW_LEVEL_POLICY_PATH,
        render_mode='human',  # 开启渲染窗口
        render_decimation=5, # 渲染帧率 (越小越流畅，但越慢)
        history_length=HISTORY_LENGTH,
        enable_dynamic_obstacles=False # 根据需要开启
    )

    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at '{args.model_path}'")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 2. 加载模型
    try:
        # 准备 Policy 参数 (如果是 .pth 需要，如果是 .zip 主要是为了 custom_objects)
        policy_kwargs = dict(
            features_extractor_class=SequenceFusionExtractor,
            features_extractor_kwargs=dict(
                # [关键] 这里不再传入 vision_feature_dim
                state_feature_dim=STATE_FEATURE_DIM,
                history_length=HISTORY_LENGTH,
                d_model=128
            ),
            net_arch=dict(pi=[256, 256], qf=[256, 256]),
            share_features_extractor=False
        )

        if args.model_path.endswith(".zip"):
            print("Loading full SAC agent from .zip file...")
            # 传入 custom_objects 以确保 SB3 能找到我们定义的类
            custom_objects = {
                "SequenceFusionExtractor": SequenceFusionExtractor
            }
            agent = SAC.load(args.model_path, env=env, device=device, custom_objects=custom_objects)

        elif args.model_path.endswith(".pth"):
            print("Loading policy weights from .pth file...")
            
            # 初始化一个空的 Agent，结构必须与训练时完全一致
            agent = SAC("MultiInputPolicy", env, policy_kwargs=policy_kwargs, device=device)
            
            # 加载权重
            # 注意：这里假设 .pth 只保存了 policy 的 state_dict
            # 如果保存的是整个 model 的 state_dict，可能需要 agent.policy.load_state_dict(checkpoint['policy'])
            state_dict = torch.load(args.model_path, map_location=torch.device(device))
            agent.policy.load_state_dict(state_dict)
            
        else:
            print(f"Error: Unknown format. Please use .zip or .pth")
            return

        print("Model loaded successfully!")

        # 3. 开始运行循环
        episode_count = 0
        while True:
            episode_count += 1
            obs, info = env.reset()
            done = False
            episode_reward = 0
            episode_length = 0
            
            print(f"\n--- Episode {episode_count} Start ---")
            
            while not done:
                # [关键] deterministic=True 用于评估，去除随机探索噪声
                action, _ = agent.predict(obs, deterministic=True)
                
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                episode_reward += reward
                episode_length += 1
                
                # 只有在 render_mode='human' 且没有自动 sync 时才需要手动 sleep
                # Mujoco 的 viewer 通常有自己的同步机制，如果不流畅可以打开下面这行
                # time.sleep(0.01)
            
            print(f"--- Episode Finished ---")
            print(f"Success: {info.get('is_success', False)}")
            print(f"Termination: {info.get('termination_reason', 'unknown')}")
            print(f"Final Distance: {info.get('distance_to_goal', 0.0):.2f}")
            print(f"Total Reward: {episode_reward:.2f}")
            print(f"Steps: {episode_length}")
            
            time.sleep(1.0) # 休息一下再开始下一轮

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
    parser.add_argument("model_path", type=str, help="Path to .zip or .pth file")
    args = parser.parse_args()
    
    main(args)
