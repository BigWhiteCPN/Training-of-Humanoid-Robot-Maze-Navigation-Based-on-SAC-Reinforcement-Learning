import argparse
import os
import sys
import numpy as np
import torch

VISUAL_TRAIN = "/home/iansten/code/IsaacLabExtensionTemplate/scripts/visual_train"
if VISUAL_TRAIN not in sys.path:
    sys.path.append(VISUAL_TRAIN)

if __package__ is None or __package__ == "":
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from core.agent import AdvancedNavigationAgent
    from core.checkpoint import load_checkpoint
    from core.config import AdvancedNavConfig
    from core.teacher import make_env
else:
    from .core.agent import AdvancedNavigationAgent
    from .core.checkpoint import load_checkpoint
    from .core.config import AdvancedNavConfig
    from .core.teacher import make_env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="./advanced_end2end_nav_logs/final_model")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--difficulty", type=int, default=-1)
    args = parser.parse_args()

    cfg = AdvancedNavConfig(final_dir=args.checkpoint)
    env = make_env(cfg, render_mode="human")
    obs, _ = env.reset()
    agent = AdvancedNavigationAgent(cfg, grid_shape=obs["grid_map"].shape)
    load_checkpoint(args.checkpoint, agent, map_location=agent.device)
    agent.networks.eval()

    if args.difficulty >= 0:
        env.unwrapped.curriculum.current_level = min(args.difficulty, len(env.unwrapped.curriculum.LEVELS) - 1)

    success = 0
    try:
        for ep in range(args.episodes):
            obs, _ = env.reset()
            agent.reset_memory()
            done = False
            total_reward = 0.0
            steps = 0
            info = {}
            while not done:
                graph = agent.graph_summary(env)
                action = agent.act(obs, graph, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                steps += 1
                done = terminated or truncated
            reason = info.get("termination_reason", "unknown")
            success += int(reason == "success")
            print(f"Episode {ep + 1}: {reason} reward={total_reward:.1f} steps={steps} dist={info.get('distance_to_goal', np.nan):.2f}")
        print(f"Success rate: {success / max(args.episodes, 1) * 100:.1f}%")
    finally:
        env.close()


if __name__ == "__main__":
    with torch.no_grad():
        main()
