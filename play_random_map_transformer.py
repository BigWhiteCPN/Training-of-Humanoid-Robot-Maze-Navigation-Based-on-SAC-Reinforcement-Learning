"""Run a trained random-map SAC transformer policy."""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import matplotlib
import torch

try:
    matplotlib.use("TkAgg")
    print("--> [System] Using TkAgg backend for visualization.")
except ImportError:
    try:
        matplotlib.use("Qt5Agg")
        print("--> [System] Using Qt5Agg backend for visualization.")
    except ImportError:
        print("--> [System] No interactive matplotlib backend found.")

from stable_baselines3 import SAC

from robot_visual_env_random_map_transformer import RobotVisualEnv
from transformer_fusion_extractor import CrossAttentionFusionExtractor


MODEL_XML_PATH = (
    "/home/iansten/code/IsaacLabExtensionTemplate/scripts/resources/mjcf/"
    "Linnxil_fifteen_angle_bs_copy_20260302.xml"
)
LOW_LEVEL_POLICY_PATH = (
    "/home/iansten/code/IsaacLabExtensionTemplate/scripts/visual_train/"
    "policy_20251026.pt"
)

HISTORY_LENGTH = 15
DECIMATION = 50
ACTION_REPEAT = 4


def main(args):
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model not found: {args.model_path}")
    if not args.model_path.endswith(".zip"):
        raise ValueError("Expected a Stable-Baselines3 .zip model.")

    env = RobotVisualEnv(
        model_path=MODEL_XML_PATH,
        low_level_policy_path=LOW_LEVEL_POLICY_PATH,
        render_mode="human",
        render_decimation=DECIMATION,
        action_repeat=ACTION_REPEAT,
        history_length=HISTORY_LENGTH,
        enable_dynamic_obstacles=args.enable_obstacles,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("==================================================")
    print(f"Loading transformer model: {args.model_path}")
    print(f"Device: {device}")
    print("Fusion: GRU state query -> CNN map key/value cross-attention")
    print("==================================================")

    try:
        # The shared extractor module is imported above, so SB3 can resolve the
        # class stored in policy_kwargs when reconstructing the saved policy.
        model = SAC.load(
            args.model_path,
            env=env,
            device=device,
            custom_objects={
                "CrossAttentionFusionExtractor": CrossAttentionFusionExtractor,
                "learning_rate": 0.0,
                "lr_schedule": lambda _: 0.0,
            },
        )
        print("Model loaded successfully.")

        episode_count = 0
        while True:
            episode_count += 1
            print(
                f"\n--- Episode {episode_count} "
                "(new random map) ---"
            )
            obs, _ = env.reset()
            done = False
            episode_reward = 0.0
            step_count = 0

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                episode_reward += float(reward)
                step_count += 1

            print(
                "Result: "
                f"{'Success' if info.get('is_success') else 'Failed'}"
            )
            print(f"Reason: {info.get('termination_reason', 'unknown')}")
            print(f"Steps: {step_count}")
            print(
                f"Distance to goal: "
                f"{info.get('distance_to_goal', 0.0):.2f} m"
            )
            print(f"Total reward: {episode_reward:.2f}")
            time.sleep(1.5)
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model_path",
        type=str,
        help="Path to best_model.zip or sac_lidar_transformer_final.zip",
    )
    parser.add_argument(
        "--enable_obstacles",
        action="store_true",
        help="Enable dynamic obstacles during playback",
    )
    main(parser.parse_args())

