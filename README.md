# Humanoid Robot Maze Navigation with SAC

本仓库包含基于 Isaac Lab/MuJoCo 的人形机器人迷宫导航训练代码。主要任务是在随机地图和迷宫环境中，根据视觉观测、占据栅格和机器人状态训练高层导航策略，并复用已有的低层 locomotion policy 执行速度指令。

项目包含 SAC 训练、课程学习、教师数据采集、端到端导航、waypoint 策略、候选路径评估和 MPC 跟踪等模块，可用于视觉导航、局部规划和强化学习训练实验。

## Files

- `train_visual.py`: 基础视觉导航 SAC 训练入口。
- `train_visual_random_map.py`: 随机地图训练入口。
- `train_visual_end2end.py`: 端到端课程学习训练入口。
- `end2end_nav_train/`: 分阶段训练流程，包括数据采集、策略预训练、critic 更新和 checkpoint 导出。
- `play.py`, `play_random_map.py`, `play_end2end.py`: 模型测试和可视化入口。
