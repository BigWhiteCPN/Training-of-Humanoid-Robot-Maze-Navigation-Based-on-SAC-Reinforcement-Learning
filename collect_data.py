import os
import numpy as np
import torch
from robot_visual_env_defusion import RobotVisualEnv
from train_visual_defusion import Config
from scipy.spatial.transform import Rotation as R
import pickle
from tqdm import tqdm

def world_to_local(robot_pos, robot_yaw, world_points):
    """
    将世界坐标系的点转换到机器人局部坐标系
    :param robot_pos: (2,) [x, y]
    :param robot_yaw: float (rad)
    :param world_points: (N, 2)
    :return: (N, 2)
    """
    if world_points is None or len(world_points) == 0:
        return np.zeros((0, 2))
    
    # 平移
    centered = world_points - robot_pos
    
    # 旋转 (逆时针旋转 -yaw)
    c, s = np.cos(-robot_yaw), np.sin(-robot_yaw)
    rot_mat = np.array([[c, -s], [s, c]])
    
    # (N, 2) @ (2, 2) -> (N, 2)
    local_points = centered @ rot_mat.T
    return local_points

def collect_expert_data():
    # 1. 配置
    TOTAL_EPISODES = 200   # 收集多少轮数据
    HORIZON = 100          # 预测未来多少个点
    SAVE_DIR = "./expert_data"
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 2. 初始化环境 (强制使用 GUI 模式以便 debug，或者为了速度用 None)
    # 必须确保 enable_dynamic_obstacles=False，或者根据需要调整，让 A* 容易跑通
    env = RobotVisualEnv(
        model_path=Config.model_xml,
        low_level_policy_path=Config.policy_path,
        render_mode=None, # 为了速度不渲染
        history_length=Config.history_length,
        enable_dynamic_obstacles=False, 
        use_diffusion_planner=False # 收集数据时必须关闭 Diffusion，用 A*
    )

    data_buffer = []
    
    print("开始收集 Expert Data...")
    
    for episode in tqdm(range(TOTAL_EPISODES)):
        obs, _ = env.reset()
        done = False
        
        while not done:
            # 环境每一步都会自动更新 PathPlanner
            # 我们只需要获取当前的动作并执行，主要是为了推进环境
            # 这里的 action 可以是随机的，或者是简单的纯跟踪，
            # 只要 robot 能动起来，A* 就会不断重规划
            
            # 为了让数据质量高，我们最好让机器人真的在沿着路径走
            # 这里简单复用环境内部的控制器逻辑，或者传一个全0 action (如果环境自带path follower)
            # 在你的代码中，step 需要 action，我们用简单的逻辑生成一个往前走的 action
            action = np.array([0.5, 0.0, 0.0]) # 简单向前
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # --- 核心采集逻辑 ---
            # 1. 获取机器人位姿
            robot_pos = info['robot_pos']
            # 获取 Yaw (需要从 env 内部获取，或者 step 返回)
            # 这是一个 hack，直接访问 env data
            quat = env.data.qpos[3:7]
            r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
            robot_yaw = r.as_euler('xyz')[2]

            # 2. 获取当前的 A* 路径 (World Frame)
            full_path = env.current_path
            
            if full_path is not None and len(full_path) > HORIZON:
                # 3. 截取未来 H 步的路径
                # 找到路径上离机器人最近的点作为起点
                dists = np.linalg.norm(full_path - robot_pos, axis=1)
                start_idx = np.argmin(dists)
                
                if start_idx + HORIZON < len(full_path):
                    expert_traj_world = full_path[start_idx : start_idx + HORIZON]
                    
                    # 4. 坐标转换 World -> Local
                    expert_traj_local = world_to_local(robot_pos, robot_yaw, expert_traj_world)
                    
                    # 5. 目标点转换
                    goal_local = world_to_local(robot_pos, robot_yaw, env.goal_pos[None, :])[0]
                    
                    # 6. 获取局部地图 (Obs 中已包含，并且通常已经是 robot-centric)
                    # obs['grid_map'] shape is (H, W, 1) -> 需要转成 (1, H, W) 存起来更省空间
                    local_map = obs['grid_map'].transpose(2, 0, 1).copy() # (1, 60, 60)
                    
                    # 存储数据
                    data_sample = {
                        'map': local_map,             # (1, H, W)
                        'goal': goal_local,           # (2,)
                        'traj': expert_traj_local     # (HORIZON, 2)
                    }
                    data_buffer.append(data_sample)

            # 定期保存，防止内存溢出
            if len(data_buffer) >= 2000:
                file_idx = len(os.listdir(SAVE_DIR))
                save_path = os.path.join(SAVE_DIR, f"data_{file_idx}.pkl")
                with open(save_path, 'wb') as f:
                    pickle.dump(data_buffer, f)
                data_buffer = []

    # 保存剩余数据
    if len(data_buffer) > 0:
        file_idx = len(os.listdir(SAVE_DIR))
        with open(os.path.join(SAVE_DIR, f"data_{file_idx}.pkl"), 'wb') as f:
            pickle.dump(data_buffer, f)
            
    print(f"数据收集完成，保存在 {SAVE_DIR}")
    env.close()

if __name__ == '__main__':
    collect_expert_data()