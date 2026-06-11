# End-to-End Navigation Notes

本文档记录端到端视觉迷宫导航的结构设计和训练流程。系统保留已有低层 locomotion policy，主要调整高层导航策略和 waypoint 到速度指令的中间接口。

约束条件：

```text
High-Level Navigation Policy   可大改
Waypoint / Velocity Interface  可大改
Low-Level Locomotion Policy    不动
MuJoCo Robot / PD / JIT Policy 不动
```

训练目标是将高层导航从单步速度输出改为局部 waypoint 序列预测：

```text
Topological Memory
    + Diffusion / Flow Waypoint Sequence Policy
    + Value Critic
    + Safety Critic
    + Progress Critic
    + Candidate Reranking
    + MPC / Pure-Pursuit Command Tracker
```

推理时仍然是三层：

```text
local map + state + graph memory
        -> high-level policy proposes waypoint sequences
        -> critics rank candidates
        -> middle tracker converts first waypoint to [vx, vy, yaw_rate]
        -> existing low-level locomotion policy
```

## 1. 核心问题

当前 level4 成功率卡住，根因是长程迷宫导航问题被压成了单步连续控制问题：

- 欧氏距离 progress 会惩罚正确绕路行为。
- 直接输出 `[vx, vy, yaw_rate]` 太短视，难以表达拐弯、绕路和回头。
- 普通高斯 actor 是单峰分布，迷宫岔路是多模态决策，平均动作容易撞墙。
- 局部观测缺少拓扑记忆，难以避免死胡同和重复探索。
- collision penalty 只是软惩罚，不能可靠约束安全。
- 普通 replay buffer 会被低级别样本、timeout、stuck 和 collision 污染。

当前实现保留低层运动控制，将主要训练压力放在高层局部规划上。

## 2. 系统架构

### 2.1 三层结构

```text
Layer 1: High-Level Learned Navigator
    输入:
        local occupancy map
        visited map
        goal heatmap
        geodesic potential
        frontier / information gain
        clearance map
        robot state history
        topological memory summary
    输出:
        K-step waypoint sequence candidates

Layer 2: Command Tracker
    输入:
        selected waypoint sequence
    输出:
        [vx, vy, yaw_rate]

Layer 3: Existing Low-Level Locomotion
    输入:
        velocity command
    输出:
        joint actions
```

低层 locomotion policy 不改，导航相关改动集中在高层策略和中间接口。

### 2.2 高层动作

将高层动作从单步速度改为局部路径片段：

```python
action = [
    wp1_x_body, wp1_y_body,
    wp2_x_body, wp2_y_body,
    wp3_x_body, wp3_y_body,
]
```

默认设置：

```text
K = 3
waypoint range = [-3.0m, 3.0m]
execute only wp1, then replan
```

这是 receding horizon control：

```text
predict short local path
execute first segment
observe again
replan
```

优势：

- 能表达“先绕开再接近目标”。
- 比单 waypoint 更适合拐角和窄通道。
- 比直接速度更容易从 planner teacher 学习。
- 支持多候选采样和 critic reranking。

## 3. Diffusion / Flow Waypoint Policy

### 3.1 为什么不用普通 SAC actor

普通 SAC actor 通常输出单峰高斯分布：

```text
pi(a | obs) = Gaussian(mu, sigma)
```

迷宫导航是多模态问题：

```text
left route valid
right route valid
turn back valid
go straight invalid
```

单峰 actor 在多条合理路径之间容易输出平均动作，平均动作可能正好撞墙。因此高层 policy 推荐改成 diffusion 或 flow matching policy：

```text
obs + noise -> denoise waypoint sequence
```

### 3.2 训练目标

使用 planner teacher 轨迹训练 diffusion policy：

```text
teacher path -> waypoint sequence
waypoint sequence + noise -> model predicts denoised sequence
```

损失：

```text
L_diffusion = || epsilon_pred - epsilon ||^2
```

或者 flow matching：

```text
L_flow = || v_pred(z_t, t, obs) - v_target ||^2
```

推荐优先 flow matching，因为推理步数可更少：

```text
diffusion: 5-20 denoise steps
flow: 1-4 integration steps
```

如果训练速度优先，可先实现 small diffusion with 4 denoise steps。

### 3.3 推理

每个高层 step 采样多个候选：

```python
candidates = policy.sample(obs, num_candidates=16)
```

每个 candidate 是一个 waypoint sequence：

```python
candidate_i = [wp1, wp2, wp3]
```

候选不直接执行，要经过 value / safety / progress critics 排序。

## 4. Critic Reranking

最终高层不是“actor 输出什么就执行什么”，而是：

```text
generate candidates
filter unsafe candidates
rank remaining candidates
execute first waypoint of best candidate
```

### 4.1 Value Critic

标准 RL critic：

```python
Q(obs, waypoint_sequence) -> expected return
```

训练来源：

- online rollout
- teacher replay
- HER relabeled replay
- hard maze replay

### 4.2 Safety Critic

独立安全 critic：

```python
C(obs, waypoint_sequence) -> P(collision within H seconds)
```

标签：

```text
future collision within horizon H -> 1
no collision -> 0
```

推荐 horizon：

```text
H = 1.5s ~ 3.0s
```

损失：

```text
L_safety = BCE(C(obs, action), collision_label)
```

推理过滤：

```python
safe_candidates = [a for a in candidates if C(obs, a) < 0.2]
```

安全 critic 比 collision penalty 更可靠，因为碰撞应该是约束，不只是负奖励。

### 4.3 Progress Critic

预测 waypoint sequence 的 geodesic progress：

```python
P(obs, waypoint_sequence) -> expected geodesic progress over H seconds
```

标签：

```python
progress_label = geodesic_dist_t - geodesic_dist_t+H
```

它帮助候选排序时区分“安全但原地绕圈”和“安全且接近目标”。

### 4.4 Candidate Score

最终候选打分：

```python
score(a) =
    Q(obs, a)
    + beta_progress * P(obs, a)
    - beta_safety * C(obs, a)
    - beta_smooth * action_curvature(a)
```

执行：

```python
best = argmax(score)
execute(best.wp1)
```

如果所有候选都 unsafe：

```text
fallback to safest candidate
reduce speed_scale
increase yaw-to-free-space behavior
```

## 5. Topological Memory

### 5.1 为什么需要拓扑记忆

完整迷宫不是单纯局部避障。策略必须知道：

- 哪些岔路走过。
- 哪些区域是死胡同。
- 如何回到上一个 junction。
- 哪些 frontier 还没探索。

GRU 只能提供弱记忆，不适合长程拓扑结构。推荐维护在线拓扑图。

### 5.2 图结构

```python
Node:
    id
    world_pos
    local_map_embedding
    visited_count
    frontier_score
    geodesic_estimate
    dead_end_score

Edge:
    node_i
    node_j
    traversed_distance
    collision_count
    success_count
    estimated_cost
```

节点创建条件：

```text
distance from nearest node > 0.8m
or local map embedding changed significantly
or junction detected
```

边创建条件：

```text
robot traversed between two nearby nodes without collision
```

### 5.3 Graph Summary 输入

不需要一开始上 GNN。可以先把图压成固定维度 summary：

```text
nearest_node_visited_count
nearest_frontier_direction_body_xy
best_return_direction_body_xy
dead_end_score
loop_closure_score
frontier_count
visited_area_ratio
```

拼到 state vector：

```python
state = concat(robot_state, graph_summary)
```

进阶版本使用 GNN：

```text
topological graph -> GraphSAGE / GAT -> graph embedding
```

### 5.4 Graph-Guided Candidate Bias

Diffusion policy 的候选可以被图记忆引导：

```text
if stuck:
    bias candidates toward nearest frontier or return node
if dead_end_score high:
    suppress forward candidates
if loop closure likely:
    prefer return path
```

这比单纯 visited map 更适合 level4。

## 6. Middle Layer: MPC / Pure-Pursuit Tracker

低层 locomotion 不动，但 waypoint 到速度的中间层可以升级。

### 6.1 基础 Pure-Pursuit Tracker

```python
target = waypoint_sequence[0]
cmd_vx = clip(kx * target.x, -0.4, 0.8)
cmd_vy = clip(ky * target.y, -0.4, 0.4)
cmd_yaw = clip(k_yaw * atan2(target.y, target.x), -0.8, 0.8)
```

### 6.2 MPC Command Tracker

扩展版本：

```text
sample velocity command sequences for 1-2 seconds
roll out simple command dynamics
score by:
    waypoint tracking error
    clearance
    smoothness
    safety critic
execute first command
```

评分：

```python
score(cmd_seq) =
    - waypoint_error
    + clearance_bonus
    - smoothness_cost
    - safety_cost
```

这个中层 MPC 不碰底层 joint policy，只优化给低层的速度命令。

## 7. Privileged Planner Teacher

### 7.1 Teacher Path

训练时使用完整地图：

```text
walls + boundary
    -> inflated traversability cost grid
    -> Dijkstra / A*
    -> global path
    -> local waypoint sequence
```

teacher sequence：

```python
teacher_action = [
    path_point_at_0.8m,
    path_point_at_1.6m,
    path_point_at_2.4m,
]
```

全部转换到 body frame。

### 7.2 Teacher Dataset

保存：

```text
obs
graph_summary
teacher_waypoint_sequence
teacher_velocity_command
reward
done
level
termination_reason
geodesic_dist
clearance
```

### 7.3 Teacher 用途

```text
1. diffusion / flow policy pretraining
2. safety critic pretraining
3. value critic pretraining
4. replay buffer expert mixture
5. online BC regularization
```

## 8. Offline-to-Online Training Pipeline

### Stage A: Teacher Data Collection

```text
levels: all levels
episodes per level: 1000-5000
dynamic obstacles: off initially
sensor noise: low
save failures too
```

### Stage B: Diffusion / Flow BC Pretraining

```text
input: obs + graph_summary
target: teacher waypoint sequence
loss: diffusion noise prediction or flow matching
```

Stop criteria:

```text
teacher waypoint MSE plateau
teacher rollout success on validation mazes > 0.7
```

### Stage C: Critic Pretraining

Train:

```text
Q critic from teacher/offline returns
safety critic from collision labels
progress critic from geodesic progress labels
```

### Stage D: Offline RL

Use teacher dataset and collected rollout dataset:

```text
TD3-BC
SAC+BC
IQL-style value learning
```

Recommended first implementation:

```text
SAC+BC with fixed lambda_bc schedule
```

### Stage E: Online Fine-Tuning

Online loop:

```text
sample candidates from diffusion policy
rerank by critics
execute selected candidate
store episode
HER relabel
level-aware replay
update policy and critics
```

BC weight schedule:

```text
lambda_bc = 0.3 -> 0.05 -> 0.0
```

Safety critic remains active throughout.

### Stage F: Hard Maze Robustness

Enable:

```text
hard levels only
dynamic obstacles optional
lidar/dropout noise
pose noise
command latency
low-level tracking error randomization
wall thickness randomization
```

## 9. HER / Goal Relabeling

Goal relabeling remains important.

Relabel target choices:

```text
future achieved position
episode final position
max geodesic progress position
frontier-near position
near-success position
```

For each relabeled goal, recompute:

```text
goal heatmap
geodesic potential
geodesic progress reward
success flag
teacher waypoint sequence if needed
```

This converts failed hard-maze episodes into useful training data.

## 10. Level-Aware Prioritized Replay

Each episode stores:

```python
{
    "level": int,
    "termination_reason": str,
    "is_success": bool,
    "near_success": bool,
    "start_geodesic_dist": float,
    "final_geodesic_dist": float,
    "progress_ratio": float,
    "collision": bool,
    "stuck": bool,
    "teacher_mse": float,
}
```

Sampling mix:

```text
25% current hardest level
20% current hardest level - 1
20% success / near-success
15% teacher / expert
10% collision recovery
5% stuck recovery
5% uniform random
```

This prevents full-maze success and near-success samples from being drowned by easy or failed samples.

## 11. Curriculum

Replace single current_level with distributional curriculum:

```python
level_probs = adaptive_distribution(level_stats)
```

Per-level stats:

```text
success_rate
collision_rate
stuck_rate
progress_ratio
teacher_mse
```

Sampling rule:

```text
success_rate 0.35-0.75 -> increase sampling
success_rate > 0.85 -> reduce sampling
success_rate < 0.15 -> add teacher ratio or back off temporarily
```

Hard levels should always remain in the replay mix once introduced.

## 12. Reward

Reward should support but not replace teacher and critics:

```text
r =
  + 8.0  * geodesic_progress
  + 0.5  * euclidean_progress
  + 0.3  * waypoint_tracking_progress
  + 0.1  * progress_critic_consistency
  + 0.05 * new_area_bonus
  + 0.03 * clearance_bonus
  - 0.5  * obstacle_proximity
  - 0.02 * time_penalty
  - 0.1  * stuck_penalty
  - 0.005 * command_smoothness
  + 800  * success
  - 150  * collision
  - 100  * fall
```

Stuck termination:

```text
if geodesic_dist improves < 0.2m over 3s:
    apply stuck penalty

if geodesic_dist improves < 0.3m over 8s:
    terminate as stuck
```

## 13. Network Components

### 13.1 Encoder

```text
map encoder: lightweight ResNet CNN
state encoder: GRU or temporal conv
graph encoder: MLP first, GNN later
fusion: MLP
```

### 13.2 Policy Head

```text
diffusion / flow waypoint sequence head
```

### 13.3 Critics

```text
Q critic: expected return
safety critic: collision probability
progress critic: expected geodesic progress
optional stuck critic: stuck probability
```

Critics share encoder optionally, but for stability:

```text
policy encoder and critic encoder can start separate
```

## 14. Inference Procedure

One high-level control cycle:

```python
obs = get_local_obs()
graph.update(obs, pose)
graph_summary = graph.get_summary()

candidates = diffusion_policy.sample(obs, graph_summary, n=16)

scores = []
for a in candidates:
    q = q_critic(obs, graph_summary, a)
    c = safety_critic(obs, graph_summary, a)
    p = progress_critic(obs, graph_summary, a)
    smooth = curvature_cost(a)
    score = q + beta_p * p - beta_c * c - beta_s * smooth
    scores.append(score)

safe_candidates = filter(candidates, safety < threshold)
best = argmax_score(safe_candidates)

cmd = tracker(best.waypoints)
low_level_policy(cmd)
```

Fallback:

```text
if no safe candidate:
    choose lowest safety risk candidate
    reduce speed
    bias yaw toward free space / frontier
```

## 15. Evaluation Metrics

Must log:

```text
success_rate
SPL
collision_rate
fall_rate
timeout_rate
stuck_rate
mean_geodesic_progress
progress_ratio
final_geodesic_dist
episode_length_success
episode_length_failure
level_distribution
teacher_action_mse
diffusion candidate diversity
safety critic AUC
safety false negative rate
progress critic error
HER relabel success ratio
topological node count
dead-end escape rate
```

Important derived metric:

```python
progress_ratio = (start_geodesic_dist - final_geodesic_dist) / start_geodesic_dist
```

Safety metric:

```python
safety_false_negative = predicted_safe_but_collided / predicted_safe
```

This must be low before enabling faster commands.

## 16. Expected Benefit

Compared with plain SAC velocity policy:

```text
geodesic shaping             fixes maze credit assignment
teacher pretraining          fixes cold-start exploration
diffusion waypoint sequence  handles multimodal route choices
safety critic                reduces collisions as constraint
progress critic              filters safe but useless actions
topological memory           reduces dead ends and repeated exploration
HER                          turns failures into useful data
level-aware replay           protects rare hard-level successes
MPC tracker                  improves command smoothness and wall clearance
```

Expected outcome if implemented well:

```text
full_maze success_rate: 0.3 -> 0.7+
collision_rate: lower
stuck/timeout: much lower
generalization to unseen maze layouts: better
training stability: much better
```

## 17. Implementation Order

Although the final target is the full system, implement in dependency order:

```text
1. waypoint sequence action + tracker
2. planner teacher waypoint sequence generation
3. teacher dataset collector
4. BC pretraining
5. safety/progress critics
6. candidate sampling + reranking
7. online SAC+BC fine-tuning
8. HER relabeling
9. level-aware replay
10. topological memory summary
11. diffusion/flow policy replacement
12. MPC tracker
```

The final architecture should not be judged until at least steps 1-7 are in place. Diffusion policy without teacher data and critics will not show its advantage.

## 18. References

- DD-PPO: Learning Near-Perfect PointGoal Navigators from 2.5 Billion Frames  
  https://arxiv.org/abs/1911.00357

- Active Neural SLAM  
  https://arxiv.org/abs/2004.05155

- SLIM: Sim-to-Real Legged Instructive Manipulation via Long-Horizon Visuomotor Learning  
  https://arxiv.org/abs/2501.09905

- TD-MPC2: Scalable, Robust World Models for Continuous Control  
  https://www.tdmpc2.com/

- DreamerV3: Mastering Diverse Domains through World Models  
  https://arxiv.org/abs/2301.04104

- Stable-Baselines3 SAC documentation  
  https://stable-baselines3.readthedocs.io/en/v2.4.1/modules/sac.html
