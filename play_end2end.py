"""
End-to-End Navigation 可视化评估脚本

用法:
    python play_end2end.py
    python play_end2end.py --model ./sac_end2end_logs/final_playable_model.zip
"""

import os
import sys
import argparse

sys.path.append("/home/iansten/code/IsaacLabExtensionTemplate/scripts/visual_train/")

import torch
import numpy as np
from stable_baselines3 import SAC
from robot_visual_env_end2end import RobotVisualEnd2EndEnv

# 复用训练脚本中的网络定义和包装器
from train_visual_end2end import End2EndExtractor, StateHistoryWrapper


def main():
    parser = argparse.ArgumentParser(description="End-to-End 导航可视化")
    parser.add_argument("--model", type=str, default="./sac_end2end_logs/final_playable_model.zip",
                        help="模型路径")
    parser.add_argument("--episodes", type=int, default=10, help="评估回合数")
    parser.add_argument("--difficulty", type=int, default=-1,
                        help="强制课程等级 (-1=自动)")
    args = parser.parse_args()

    # 环境配置
    model_xml = "/home/iansten/code/IsaacLabExtensionTemplate/scripts/resources/mjcf/Linnxil_fifteen_angle_bs_copy_20260302.xml"
    policy_path = "/home/iansten/code/IsaacLabExtensionTemplate/scripts/visual_train/policy_20251026.pt"

    env_kwargs = {
        "model_path": model_xml,
        "low_level_policy_path": policy_path,
        "render_mode": 'human',
        "render_decimation": 5,
        "action_repeat": 4,
        "enable_dynamic_obstacles": False,
        "action_mode": "waypoint",
    }

    env = RobotVisualEnd2EndEnv(**env_kwargs)

    # 如果指定难度, 强制设置课程等级
    if args.difficulty >= 0:
        env.curriculum.current_level = min(args.difficulty, len(env.curriculum.LEVELS) - 1)
        print(f"强制课程等级: {env.curriculum.current_level} ({env.curriculum.config['name']})")

    # 包装状态历史
    history_length = 10
    state_feature_dim = 12
    env = StateHistoryWrapper(env, history_length, state_feature_dim)

    # 加载模型
    if not os.path.exists(args.model):
        print(f"模型文件不存在: {args.model}")
        print("可用的模型路径:")
        for root, dirs, files in os.walk("./sac_end2end_logs/"):
            for f in files:
                if f.endswith(".zip"):
                    print(f"  {os.path.join(root, f)}")
        return

    print(f"加载模型: {args.model}")
    policy_kwargs = dict(
        features_extractor_class=End2EndExtractor,
        features_extractor_kwargs=dict(
            state_feature_dim=state_feature_dim,
            history_length=history_length,
            d_model=128
        ),
        net_arch=dict(pi=[256, 256], qf=[256, 256]),
        share_features_extractor=False
    )

    model = SAC.load(args.model, env=env, device="cuda",
                     custom_objects={"policy_kwargs": policy_kwargs})

    # 评估循环
    success_count = 0
    collision_count = 0
    fall_count = 0
    timeout_count = 0
    total_rewards = []
    total_distances = []

    for ep in range(args.episodes):
        obs, info = env.reset()
        done = False
        ep_reward = 0.0
        step_count = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated
            step_count += 1

        reason = info.get("termination_reason", "unknown")
        dist = info.get("distance_to_goal", -1)
        total_rewards.append(ep_reward)
        total_distances.append(dist)

        if reason == "success":
            success_count += 1
            tag = "SUCCESS"
        elif reason == "collision":
            collision_count += 1
            tag = "COLLISION"
        elif reason == "fall":
            fall_count += 1
            tag = "FALL"
        else:
            timeout_count += 1
            tag = "TIMEOUT"

        level = info.get("curriculum_level", "?")
        print(f"Episode {ep+1}/{args.episodes} | {tag} | Reward: {ep_reward:.1f} | "
              f"Dist: {dist:.2f}m | Steps: {step_count} | Level: {level}")

    print("\n" + "=" * 50)
    print(f"评估结果 ({args.episodes} episodes)")
    print(f"  成功率: {success_count/args.episodes*100:.1f}% ({success_count}/{args.episodes})")
    print(f"  碰撞率: {collision_count/args.episodes*100:.1f}%")
    print(f"  摔倒率: {fall_count/args.episodes*100:.1f}%")
    print(f"  超时率: {timeout_count/args.episodes*100:.1f}%")
    print(f"  平均奖励: {np.mean(total_rewards):.1f}")
    print(f"  平均最终距离: {np.mean(total_distances):.2f}m")
    print("=" * 50)

    env.close()


if __name__ == '__main__':
    main()
