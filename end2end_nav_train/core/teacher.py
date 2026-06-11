import os
import sys
import numpy as np

VISUAL_TRAIN = "/home/iansten/code/IsaacLabExtensionTemplate/scripts/visual_train"
if VISUAL_TRAIN not in sys.path:
    sys.path.append(VISUAL_TRAIN)
END2END_DIR = "/home/iansten/code/IsaacLabExtensionTemplate/scripts/visual_train/end2end_nav_train"
if END2END_DIR not in sys.path:
    sys.path.append(END2END_DIR)

from env import RobotVisualEnd2EndEnv
from .wrappers import StateHistoryWrapper


def _level_value(config, name, level, default):
    values = getattr(config, name, None)
    if values is None:
        return default
    if level < len(values):
        return values[level]
    return values[-1]


def _episode_success_mask(dones, reasons):
    mask = np.zeros(len(dones), dtype=bool)
    start = 0
    for i, done in enumerate(dones):
        if bool(done):
            success = str(reasons[i]) == "success"
            mask[start:i + 1] = success
            start = i + 1
    if start < len(dones):
        mask[start:] = False
    return mask


def _dataset_needs_rebuild(path, config):
    if not os.path.exists(path):
        return True
    data = np.load(path, allow_pickle=True)
    version = int(data["dataset_version"][0]) if "dataset_version" in data else 0
    if version < config.teacher_dataset_version:
        print(f"[teacher] Rebuilding old dataset version={version}")
        return True
    if not config.teacher_rebuild_if_poor:
        return False
    dones = data["dones"].astype(bool)
    reasons = data["reasons"]
    terminal = reasons[dones]
    if len(terminal) == 0:
        print("[teacher] Rebuilding dataset with no terminal episodes")
        return True
    success_rate = float(np.mean([str(x) == "success" for x in terminal]))
    if success_rate < config.teacher_min_bc_success_rate:
        print(f"[teacher] Rebuilding poor dataset success_rate={success_rate:.3f}")
        return True
    return False


def make_env(config, render_mode=None):
    env = RobotVisualEnd2EndEnv(
        model_path=config.model_xml,
        low_level_policy_path=config.low_level_policy,
        render_mode=render_mode,
        render_decimation=config.render_decimation,
        action_repeat=config.action_repeat,
        enable_dynamic_obstacles=False,
        action_mode=config.action_mode,
    )
    return StateHistoryWrapper(env, config.history_length, config.state_dim)


def collect_teacher_dataset(config):
    os.makedirs(os.path.dirname(config.teacher_dataset), exist_ok=True)
    if not _dataset_needs_rebuild(config.teacher_dataset, config):
        print(f"[teacher] Reusing {config.teacher_dataset}")
        return
    print(f"[teacher] Collecting dataset -> {config.teacher_dataset}")
    env = make_env(config, render_mode=None)
    grids, states, graphs, actions, rewards, dones, levels, reasons = [], [], [], [], [], [], [], []
    geodesic_dists, robot_positions, robot_yaws, episode_ids = [], [], [], []
    episode_id = 0
    try:
        for level in range(len(env.unwrapped.curriculum.LEVELS)):
            successes = 0
            attempts = 0
            target_successes = int(_level_value(config, "teacher_successes_by_level", level, config.teacher_successes_per_level))
            min_episodes = int(_level_value(config, "teacher_min_episodes_by_level", level, config.teacher_episodes_per_level))
            max_attempts = int(_level_value(config, "teacher_max_attempts_by_level", level, config.teacher_max_attempts_per_level))
            extra_steps = int(_level_value(config, "teacher_extra_steps_by_level", level, config.teacher_hard_level_extra_steps if level >= 2 else 0))
            max_attempts = max(min_episodes, max_attempts)
            while attempts < max_attempts and (attempts < min_episodes or successes < target_successes):
                env.unwrapped.curriculum.current_level = level
                env.unwrapped.curriculum.success_buffer.clear()
                obs, _ = env.reset(seed=config.seed + level * 1000 + attempts)
                done = False
                step = 0
                graph = np.zeros(config.graph_dim, dtype=np.float32)
                ep_indices = []
                terminal_reason = "teacher_timeout"
                max_steps = config.teacher_max_steps + extra_steps
                while not done and step < max_steps:
                    action = env.unwrapped.get_teacher_action()
                    robot_pos, robot_yaw = env.unwrapped._get_robot_pose()
                    next_obs, reward, terminated, truncated, info = env.step(action)
                    done_flag = bool(terminated or truncated)
                    terminal_reason = info.get("termination_reason", "running") if done_flag else "running"
                    grids.append(obs["grid_map"].copy())
                    states.append(obs["state_history"].copy())
                    graphs.append(graph.copy())
                    actions.append(action.copy())
                    rewards.append(float(reward))
                    dones.append(float(done_flag))
                    levels.append(level)
                    reasons.append(terminal_reason)
                    geodesic_dists.append(float(info.get("rewards", {}).get("geodesic_dist", env.unwrapped.prev_geodesic_dist)))
                    robot_positions.append(robot_pos.copy())
                    robot_yaws.append(float(robot_yaw))
                    episode_ids.append(episode_id)
                    ep_indices.append(len(dones) - 1)
                    obs = next_obs
                    done = done_flag
                    step += 1
                if not done and ep_indices:
                    dones[ep_indices[-1]] = 1.0
                    reasons[ep_indices[-1]] = "teacher_timeout"
                    terminal_reason = "teacher_timeout"
                if terminal_reason == "success":
                    successes += 1
                attempts += 1
                episode_id += 1
                print(
                    f"[teacher] level={level} attempt={attempts}/{max_attempts} "
                    f"success={successes}/{target_successes} min_ep={min_episodes} "
                    f"max_steps={max_steps} steps={step} reason={terminal_reason}"
                )
    finally:
        env.close()

    success_mask = _episode_success_mask(np.asarray(dones, dtype=np.float32), np.asarray(reasons, dtype=object))
    np.savez_compressed(
        config.teacher_dataset,
        dataset_version=np.asarray([config.teacher_dataset_version], dtype=np.int32),
        teacher_successes_by_level=np.asarray(config.teacher_successes_by_level, dtype=np.int32),
        teacher_min_episodes_by_level=np.asarray(config.teacher_min_episodes_by_level, dtype=np.int32),
        teacher_max_attempts_by_level=np.asarray(config.teacher_max_attempts_by_level, dtype=np.int32),
        teacher_extra_steps_by_level=np.asarray(config.teacher_extra_steps_by_level, dtype=np.int32),
        grid_map=np.asarray(grids, dtype=np.uint8),
        state_history=np.asarray(states, dtype=np.float32),
        graph_summary=np.asarray(graphs, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.float32),
        rewards=np.asarray(rewards, dtype=np.float32),
        dones=np.asarray(dones, dtype=np.float32),
        levels=np.asarray(levels, dtype=np.int32),
        reasons=np.asarray(reasons, dtype=object),
        geodesic_dist=np.asarray(geodesic_dists, dtype=np.float32),
        robot_pos=np.asarray(robot_positions, dtype=np.float32),
        robot_yaw=np.asarray(robot_yaws, dtype=np.float32),
        episode_id=np.asarray(episode_ids, dtype=np.int32),
        episode_success=success_mask.astype(np.float32),
    )
    print(f"[teacher] transitions={len(actions)}")


def load_teacher_dataset(path):
    data = np.load(path, allow_pickle=True)
    out = {k: data[k] for k in data.files}
    if "episode_success" not in out:
        out["episode_success"] = _episode_success_mask(out["dones"], out["reasons"]).astype(np.float32)
    return out
