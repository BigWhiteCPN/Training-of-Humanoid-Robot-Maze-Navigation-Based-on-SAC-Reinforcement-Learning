# test_env.py

import time
import numpy as np
import mujoco.viewer # 导入 viewer 用于手动启动

# 从我们自己创建的文件中导入环境
from robot_visual_env import RobotVisualEnv

# --- 脚本配置 ---
MODEL_XML_PATH = "/home/iansten/code/IsaacLabExtensionTemplate/scripts/resources/mjcf/Linnxil_fifteen_angle_bs_copy_20260302.xml"
LOW_LEVEL_POLICY_PATH = "/home/iansten/code/IsaacLabExtensionTemplate/scripts/visual_train/policy_20251026.pt"

def main():
    """
    创建一个环境实例，生成一个随机房间，并启动一个交互式查看器来显示它。
    """
    print("==================================================")
    print("环境可视化测试脚本 (交互式)")
    print("  - 将生成一个随机房间并打开一个查看器窗口。")
    print("  - 您可以自由拖动、缩放、旋转视角。")
    print("  - 关闭查看器窗口或在终端中按 Ctrl+C 来结束程序。")
    print("==================================================")

    # --- 1. 创建环境 ---
    # 在这个模式下，我们不需要设置 render_mode，因为我们将手动启动 viewer
    env = RobotVisualEnv(
        model_path=MODEL_XML_PATH,
        low_level_policy_path=LOW_LEVEL_POLICY_PATH
    )

    try:
        # --- 2. 重置环境以生成随机房间 ---
        print("\n--- 正在生成随机房间... ---")
        obs, info = env.reset()
        
        print("机器人初始位置:", env.data.qpos[:2])
        print("目标位置:", env.goal_pos)
        print(f"激活的障碍物数量: {env.num_active_obstacles}")
        print("\n启动查看器... (如果窗口没有出现，请检查您的图形环境设置)")

        # --- 3. 【核心修改】使用 mujoco.viewer.launch 手动启动一个交互式查看器 ---
        # launch() 会阻塞程序的执行，直到你关闭窗口
        mujoco.viewer.launch(env.model, env.data)

    except KeyboardInterrupt:
        print("\n测试结束。")
    finally:
        # --- 4. 清理 ---
        print("正在关闭环境...")
        env.close()

if __name__ == '__main__':
    main()
