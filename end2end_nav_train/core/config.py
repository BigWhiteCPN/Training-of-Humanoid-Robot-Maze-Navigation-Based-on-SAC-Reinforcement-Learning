from dataclasses import dataclass
from pathlib import Path


ROOT = Path("/home/iansten/code/IsaacLabExtensionTemplate/scripts/visual_train")


@dataclass
class AdvancedNavConfig:
    model_xml: str = str(ROOT / "../resources/mjcf/Linnxil_fifteen_angle_bs_copy_20260302.xml")
    low_level_policy: str = str(ROOT / "policy_20251026.pt")
    log_dir: str = "./advanced_end2end_nav_logs"
    tensorboard_dir: str = "./advanced_end2end_nav_logs/tensorboard"
    checkpoint_dir: str = "./advanced_end2end_nav_logs/checkpoints"
    final_dir: str = "./advanced_end2end_nav_logs/final_model"
    best_dir: str = "./advanced_end2end_nav_logs/best_model"
    level_best_root: str = "./advanced_end2end_nav_logs/level_best"
    teacher_dataset: str = "./advanced_end2end_nav_logs/teacher_dataset.npz"

    seed: int = 42
    device: str = "cuda"
    render: bool = False
    action_repeat: int = 4
    render_decimation: int = 50
    history_length: int = 10
    state_dim: int = 12
    graph_dim: int = 8
    action_dim: int = 7
    action_mode: str = "waypoint"

    teacher_episodes_per_level: int = 24
    teacher_successes_per_level: int = 16
    teacher_max_attempts_per_level: int = 72
    teacher_successes_by_level: tuple = (16, 20, 24, 32, 40, 48, 56, 64)
    teacher_min_episodes_by_level: tuple = (24, 28, 32, 44, 56, 72, 88, 104)
    teacher_max_attempts_by_level: tuple = (48, 60, 72, 96, 120, 144, 168, 192)
    teacher_extra_steps_by_level: tuple = (0, 0, 450, 600, 750, 900, 1050, 1200)
    teacher_max_steps: int = 900
    teacher_hard_level_extra_steps: int = 450
    teacher_dataset_version: int = 3
    teacher_rebuild_if_poor: bool = True
    teacher_min_bc_success_rate: float = 0.35
    bc_success_sample_fraction: float = 0.85
    bc_updates: int = 15_000
    critic_pretrain_updates: int = 8_000
    online_steps: int = 10_000_000
    robustness_steps: int = 500_000
    batch_size: int = 1024
    replay_capacity: int = 5_500_000
    demo_replay_capacity: int = 500_000
    replay_rebuild_interval: int = 100_000
    demo_sample_fraction: float = 0.40
    current_level_sample_fraction: float = 0.35
    adjacent_level_sample_fraction: float = 0.20
    horizon_steps: int = 50
    her_samples_per_episode: int = 8

    lr_policy: float = 1e-4
    lr_critic: float = 1e-4
    reward_scale: float = 0.01
    progress_scale: float = 0.1
    gamma: float = 0.995
    tau: float = 0.001
    candidate_count: int = 64
    flow_steps: int = 4
    flow_loss_weight: float = 1.0
    mean_bc_loss_weight: float = 0.25
    safety_threshold: float = 0.20
    bc_weight_start: float = 2.0
    bc_weight_end: float = 0.5
    rl_loss_weight: float = 0.05
    action_anchor_weight: float = 1.0
    safety_policy_weight: float = 10.0
    progress_policy_weight: float = 0.25
    obstacle_policy_weight: float = 3.0
    waypoint_min_clearance: float = 0.35
    obstacle_hard_threshold: float = 0.55
    deterministic_candidate_spread: float = 0.75
    policy_warmup_steps: int = 50_000
    online_train_freq_steps: int = 50
    online_gradient_steps: int = 25
    critic_updates_per_episode: int = 3
    policy_update_interval: int = 1
    eval_level_bonus: float = 0.15
    best_score_min_delta: float = 0.02
    rollback_score_margin: float = 0.25
    collision_freeze_threshold: float = 0.45
    policy_recovery_steps: int = 150_000
    max_bad_evals_before_stop: int = 4
    rollback_demote_level: bool = True
    entropy_coef: float = 0.08
    log_std_min: float = -1.2
    log_std_max: float = 0.5

    save_every_steps: int = 100_000
    eval_every_steps: int = 500_000
    eval_episodes: int = 10
    show_progress_bar: bool = True
    robustness_enable_dynamic_obstacles: bool = True
    robustness_sensor_dropout: float = 0.03
    robustness_action_noise: float = 0.03
    robustness_latency_steps: int = 2
