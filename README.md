# Humanoid Robot Maze Navigation with SAC

本仓库包含基于 Isaac Lab/MuJoCo 的人形机器人迷宫导航训练代码。主要任务是在随机地图和迷宫环境中，根据视觉观测、占据栅格和机器人状态训练高层导航策略，并复用已有的低层 locomotion policy 执行速度指令。

项目包含 SAC 训练、课程学习、教师数据采集、端到端导航、waypoint 策略、候选路径评估和 MPC 跟踪等模块，可用于视觉导航、局部规划和强化学习训练实验。

## Files

- `train_visual.py`: 基础视觉导航 SAC 训练入口。
- `train_visual_random_map.py`: 随机地图训练入口。
- `train_visual_random_map_transformer.py`: 随机地图 Transformer 融合版 SAC 训练入口。
- `transformer_fusion_extractor.py`: GRU 状态序列到局部栅格地图 token 的 cross-attention 特征融合器。
- `train_visual_end2end.py`: 端到端课程学习训练入口。
- `end2end_nav_train/`: 分阶段训练流程，包括数据采集、策略预训练、critic 更新和 checkpoint 导出。
- `play.py`, `play_random_map.py`, `play_random_map_transformer.py`, `play_end2end.py`: 模型测试和可视化入口。

## Transformer Random Map SAC

Transformer 版本复用 `robot_visual_env_random_map.py` 的环境、观测、奖励和终止逻辑，只替换 SAC 的 feature extractor。`CrossAttentionFusionExtractor` 使用 CNN 保留局部地图的空间 token，用 GRU 编码 15 帧机器人状态历史，并以状态 token 作为 query 对地图 token 做 cross-attention，输出给 SAC actor 和 critic。

默认训练配置：

- 日志目录：`sac_lidar_logs_random_transformer/`
- replay buffer：`3_000_000`
- batch size：`512`
- transformer：`d_model=128`, `num_heads=4`, `ffn_dim=256`, `dropout=0.05`
- SAC：`learning_rate=2e-5`, `gamma=0.993`, `tau=0.0005`, `ent_coef="auto"`

启动训练：

```bash
python3 train_visual_random_map_transformer.py
```

播放已训练模型：

```bash
python3 play_random_map_transformer.py sac_lidar_logs_random_transformer/best_model/best_model.zip
```

如需测试动态障碍物：

```bash
python3 play_random_map_transformer.py sac_lidar_logs_random_transformer/best_model/best_model.zip --enable_obstacles
```
