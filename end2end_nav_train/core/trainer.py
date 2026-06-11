import os
import numpy as np
import torch
from collections import defaultdict, deque
try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

from .agent import AdvancedNavigationAgent
from .checkpoint import load_checkpoint, save_checkpoint
from .config import AdvancedNavConfig
from .replay import EpisodeReplayBuffer
from .teacher import collect_teacher_dataset, load_teacher_dataset, make_env


class ConsoleMetricPrinter:
    def __init__(self, window=50):
        self.window = window
        self.buffers = defaultdict(lambda: deque(maxlen=window))

    def add_episode(self, info):
        reason = info.get("termination_reason", "running")
        self.buffers["success"].append(float(info.get("is_success", False)))
        self.buffers["collision"].append(float(reason == "collision"))
        self.buffers["stuck"].append(float(reason == "stuck"))
        self.buffers["ep_len"].append(float(info.get("episode_len", 0)))
        self.buffers["geo_dist"].append(float(info.get("geodesic_dist", 0.0)))
        self.buffers["level"].append(float(info.get("level", 0)))

    def summary(self):
        out = {}
        for key, values in self.buffers.items():
            if values:
                out[key] = float(np.mean(values))
        return out


def teacher_batch(dataset, batch_size, device, success_fraction=0.85):
    n = len(dataset["actions"])
    success_mask = dataset.get("episode_success", np.zeros(n, dtype=np.float32)).astype(bool)
    success_idx = np.flatnonzero(success_mask)
    all_idx = np.arange(n)
    if len(success_idx) > 0 and success_fraction > 0.0:
        n_success = int(batch_size * success_fraction)
        n_other = batch_size - n_success
        idx = np.concatenate([
            np.random.choice(success_idx, size=n_success, replace=True),
            np.random.choice(all_idx, size=n_other, replace=True),
        ])
        np.random.shuffle(idx)
    else:
        idx = np.random.randint(0, n, size=batch_size)
    grid = dataset["grid_map"][idx]
    state = dataset["state_history"][idx]
    graph = dataset["graph_summary"][idx] if "graph_summary" in dataset else np.zeros((batch_size, 8), dtype=np.float32)
    action = dataset["actions"][idx].astype(np.float32)
    return {
        "grid_map": torch.as_tensor(grid, dtype=torch.uint8, device=device),
        "state_history": torch.as_tensor(state, dtype=torch.float32, device=device),
        "graph_summary": torch.as_tensor(graph, dtype=torch.float32, device=device),
        "teacher_action": torch.as_tensor(action, dtype=torch.float32, device=device),
    }


class AdvancedNavTrainer:
    def __init__(self, config=None):
        self.cfg = config or AdvancedNavConfig()
        os.makedirs(self.cfg.log_dir, exist_ok=True)
        os.makedirs(self.cfg.checkpoint_dir, exist_ok=True)
        os.makedirs(self.cfg.tensorboard_dir, exist_ok=True)
        self.writer = SummaryWriter(self.cfg.tensorboard_dir) if SummaryWriter is not None else None
        self.env = make_env(self.cfg, render_mode=None)
        sample_obs, _ = self.env.reset(seed=self.cfg.seed)
        grid_shape = sample_obs["grid_map"].shape
        self.agent = AdvancedNavigationAgent(self.cfg, grid_shape=grid_shape)
        self.replay = EpisodeReplayBuffer(
            self.cfg.replay_capacity,
            self.cfg.gamma,
            horizon_steps=self.cfg.horizon_steps,
            her_samples_per_episode=self.cfg.her_samples_per_episode,
            demo_capacity=self.cfg.demo_replay_capacity,
            demo_sample_fraction=self.cfg.demo_sample_fraction,
            current_level_sample_fraction=self.cfg.current_level_sample_fraction,
            adjacent_level_sample_fraction=self.cfg.adjacent_level_sample_fraction,
            rebuild_interval=self.cfg.replay_rebuild_interval,
        )
        self.teacher_data = None
        self.best_eval_score = -np.inf
        self.best_level_scores = {}
        self.bad_eval_count = 0
        self.freeze_policy_until = 0
        self.console = ConsoleMetricPrinter(window=50)
        self.progress_bar = None

    def log_scalar(self, key, value, step):
        if self.writer is not None:
            self.writer.add_scalar(key, float(value), int(step))

    def log_dict(self, prefix, values, step):
        for key, value in values.items():
            if isinstance(value, (int, float, np.integer, np.floating)):
                self.log_scalar(f"{prefix}/{key}", value, step)

    def print_status(self, message):
        if self.progress_bar is not None:
            self.progress_bar.write(message)
        else:
            print(message)

    def eval_score(self, metrics):
        return (
            metrics["success"]
            - metrics["collision"]
            - 0.5 * metrics["stuck"]
            + 0.25 * metrics["spl"]
            + self.cfg.eval_level_bonus * metrics.get("level", 0.0)
        )

    def load_best_if_available(self):
        ckpt = os.path.join(self.cfg.best_dir, "agent.pt")
        if os.path.exists(ckpt):
            step = load_checkpoint(self.cfg.best_dir, self.agent, map_location=self.agent.device)
            self.print_status(f"[rollback] loaded best checkpoint step={step} from {self.cfg.best_dir}")
            return True
        return False

    def level_best_dir(self, level):
        return os.path.join(self.cfg.level_best_root, f"level_{int(level)}")

    def save_level_best(self, level, step):
        save_checkpoint(self.level_best_dir(level), self.agent, self.cfg, step=step)

    def load_level_best_if_available(self, level):
        path = self.level_best_dir(level)
        ckpt = os.path.join(path, "agent.pt")
        if os.path.exists(ckpt):
            step = load_checkpoint(path, self.agent, map_location=self.agent.device)
            self.print_status(f"[rollback] loaded level {int(level)} best checkpoint step={step} from {path}")
            return True
        return False

    def enter_recovery(self, step, reason):
        self.freeze_policy_until = max(self.freeze_policy_until, step + self.cfg.policy_recovery_steps)
        if self.cfg.rollback_demote_level:
            env = self.env.unwrapped
            old_level = int(env.curriculum.current_level)
            env.curriculum.current_level = max(0, old_level - 1)
            env.curriculum.success_buffer.clear()
            self.print_status(f"[recovery] {reason}; freeze_actor_until={self.freeze_policy_until}; level {old_level}->{env.curriculum.current_level}")
        else:
            self.print_status(f"[recovery] {reason}; freeze_actor_until={self.freeze_policy_until}")

    def close(self):
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
        self.env.close()

    def stage_teacher(self):
        collect_teacher_dataset(self.cfg)
        self.teacher_data = load_teacher_dataset(self.cfg.teacher_dataset)

    def stage_bc(self):
        if self.teacher_data is None:
            self.teacher_data = load_teacher_dataset(self.cfg.teacher_dataset)
        print(f"[stage BC] updates={self.cfg.bc_updates}")
        losses = []
        for i in range(1, self.cfg.bc_updates + 1):
            batch = teacher_batch(
                self.teacher_data,
                self.cfg.batch_size,
                self.agent.device,
                success_fraction=self.cfg.bc_success_sample_fraction,
            )
            loss = self.agent.bc_update(batch)
            losses.append(loss)
            self.log_scalar("bc/loss", loss, i)
            if i % 250 == 0:
                print(f"[stage BC] {i}/{self.cfg.bc_updates} loss={np.mean(losses[-250:]):.5f}")
        save_checkpoint(os.path.join(self.cfg.checkpoint_dir, "bc"), self.agent, self.cfg, step=0)

    def prefill_replay_from_teacher(self):
        if self.teacher_data is None:
            self.teacher_data = load_teacher_dataset(self.cfg.teacher_dataset)
        n = len(self.teacher_data["actions"])
        episode = []
        episode_start_geo = 1.0
        for i in range(n - 1):
            done = bool(self.teacher_data["dones"][i])
            next_i = i if done else i + 1
            if not episode:
                episode_start_geo = float(self.teacher_data["geodesic_dist"][i]) if "geodesic_dist" in self.teacher_data else 1.0
            t = {
                "grid_map": self.teacher_data["grid_map"][i],
                "state_history": self.teacher_data["state_history"][i],
                "graph_summary": self.teacher_data["graph_summary"][i],
                "action": self.teacher_data["actions"][i],
                "teacher_action": self.teacher_data["actions"][i],
                "next_grid_map": self.teacher_data["grid_map"][next_i],
                "next_state_history": self.teacher_data["state_history"][next_i],
                "next_graph_summary": self.teacher_data["graph_summary"][next_i],
                "reward": self.teacher_data["rewards"][i],
                "done": float(done),
                "collision_label": float(str(self.teacher_data["reasons"][i]) == "collision"),
                "progress_label": max(float(self.teacher_data["rewards"][i]), 0.0),
                "level": int(self.teacher_data["levels"][i]),
                "is_success": str(self.teacher_data["reasons"][i]) == "success",
                "near_success": False,
                "teacher": True,
                "teacher_success_episode": bool(self.teacher_data.get("episode_success", np.zeros(n))[i]),
                "termination_reason": str(self.teacher_data["reasons"][i]),
                "geodesic_dist": float(self.teacher_data["geodesic_dist"][i]) if "geodesic_dist" in self.teacher_data else 0.0,
                "start_geodesic_dist": episode_start_geo,
                "robot_pos": self.teacher_data["robot_pos"][i] if "robot_pos" in self.teacher_data else np.zeros(2, dtype=np.float32),
                "robot_yaw": float(self.teacher_data["robot_yaw"][i]) if "robot_yaw" in self.teacher_data else 0.0,
                "achieved_goal": self.teacher_data["robot_pos"][i] if "robot_pos" in self.teacher_data else np.zeros(2, dtype=np.float32),
            }
            episode.append(t)
            if done:
                self.replay.add_episode(episode)
                self.replay.add_episode(self.replay.add_hindsight_goals(episode))
                episode = []
        stats = self.replay.stats()
        self.log_dict("replay", stats, 0)
        print(
            f"[replay] prefilled total={stats['total']} "
            f"demo={stats['demo']} online={stats['online']} capacity={stats['capacity']}"
        )

    def stage_critic_pretrain(self):
        self.prefill_replay_from_teacher()
        print(f"[stage critics] updates={self.cfg.critic_pretrain_updates}")
        for i in range(1, self.cfg.critic_pretrain_updates + 1):
            batch = self.replay.sample(self.cfg.batch_size, self.agent.device)
            metrics = self.agent.critic_update(batch)
            self.log_dict("critic_pretrain", metrics, i)
            if i % 250 == 0:
                print(f"[stage critics] {i}/{self.cfg.critic_pretrain_updates} loss={metrics['critic_loss']:.4f}")
            self.agent.soft_update_targets()
        save_checkpoint(os.path.join(self.cfg.checkpoint_dir, "critics"), self.agent, self.cfg, step=0)

    def set_robustness(self, enabled):
        env = self.env.unwrapped
        env.sensor_dropout = self.cfg.robustness_sensor_dropout if enabled else 0.0
        env.action_noise = self.cfg.robustness_action_noise if enabled else 0.0
        env.latency_steps = self.cfg.robustness_latency_steps if enabled else 0
        if hasattr(env, "set_dynamic_obstacles"):
            env.set_dynamic_obstacles(enabled and self.cfg.robustness_enable_dynamic_obstacles)

    def rollout_episode(self, deterministic=False, progress_callback=None):
        obs, _ = self.env.reset()
        self.agent.reset_memory()
        episode = []
        done = False
        last_geo = getattr(self.env.unwrapped, "prev_geodesic_dist", 0.0)
        while not done:
            graph = self.agent.graph_summary(self.env)
            action = self.agent.act(obs, graph, deterministic=deterministic)
            teacher_action = self.env.unwrapped.get_teacher_action()
            robot_pos, robot_yaw = self.env.unwrapped._get_robot_pose()
            start_geo = float(getattr(self.env.unwrapped, "dist_to_goal_start", max(last_geo, 1.0)))
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            next_graph = self.agent.graph_summary(self.env)
            geo = info.get("rewards", {}).get("geodesic_dist", last_geo)
            progress = float(last_geo - geo)
            last_geo = geo
            t = {
                "grid_map": obs["grid_map"].copy(),
                "state_history": obs["state_history"].copy(),
                "graph_summary": graph.copy(),
                "action": action.copy(),
                "teacher_action": teacher_action.copy(),
                "next_grid_map": next_obs["grid_map"].copy(),
                "next_state_history": next_obs["state_history"].copy(),
                "next_graph_summary": next_graph.copy(),
                "reward": float(reward),
                "done": float(terminated or truncated),
                "collision_label": float(info.get("termination_reason") == "collision"),
                "progress_label": progress,
                "geodesic_dist": float(geo),
                "start_geodesic_dist": start_geo,
                "level": int(info.get("curriculum_level", 0)),
                "is_success": bool(info.get("is_success", False)),
                "near_success": float(info.get("distance_to_goal", 999.0)) < 2.0,
                "termination_reason": info.get("termination_reason", "running"),
                "stuck": info.get("termination_reason", "running") == "stuck",
                "robot_pos": robot_pos.copy(),
                "robot_yaw": float(robot_yaw),
                "achieved_goal": info.get("robot_pos", robot_pos).copy(),
            }
            episode.append(t)
            if progress_callback is not None:
                progress_callback(1)
            obs = next_obs
            done = terminated or truncated
        self.replay.add_episode(episode)
        self.replay.add_episode(self.replay.add_hindsight_goals(episode))
        final = dict(episode[-1])
        final["episode_len"] = len(episode)
        return final

    def stage_online(self, total_steps, name="online"):
        print(f"[stage {name}] steps={total_steps}")
        self.set_robustness(name == "robustness")
        step = 0
        update_count = 0
        next_log = 1_000
        next_eval = self.cfg.eval_every_steps
        next_save = self.cfg.save_every_steps
        last_critic_metrics = {}
        last_policy_metrics = {}
        pbar = None
        if self.cfg.show_progress_bar and tqdm is not None:
            pbar = tqdm(total=total_steps, desc=f"{name}", unit="it", dynamic_ncols=True, smoothing=0.05, mininterval=0.5)
        self.progress_bar = pbar
        pbar_count = 0

        def update_progress(n=1):
            nonlocal pbar_count
            if pbar is None:
                return
            remaining = total_steps - pbar_count
            if remaining <= 0:
                return
            inc = min(int(n), remaining)
            pbar.update(inc)
            pbar_count += inc

        try:
            while step < total_steps:
                info = self.rollout_episode(deterministic=False, progress_callback=update_progress)
                episode_len = int(info.get("episode_len", 1))
                step += episode_len
                self.console.add_episode(info)
                self.log_scalar(f"{name}/episode_len", info.get("episode_len", 0), step)
                self.log_scalar(f"{name}/curriculum_level", info.get("level", 0), step)
                self.log_scalar(f"{name}/success_episode", float(info.get("is_success", False)), step)
                self.log_scalar(f"{name}/collision_episode", float(info.get("termination_reason") == "collision"), step)
                self.log_scalar(f"{name}/stuck_episode", float(info.get("termination_reason") == "stuck"), step)
                self.log_scalar(f"{name}/final_geodesic_dist", info.get("geodesic_dist", 0.0), step)
                if len(self.replay) >= self.cfg.batch_size:
                    current_level = int(self.env.unwrapped.curriculum.current_level)
                    train_blocks = max(1, int(np.ceil(episode_len / max(1, self.cfg.online_train_freq_steps))))
                    scheduled_updates = train_blocks * max(1, self.cfg.online_gradient_steps)
                    num_updates = max(self.cfg.critic_updates_per_episode, scheduled_updates)
                    for update_i in range(num_updates):
                        batch = self.replay.sample(self.cfg.batch_size, self.agent.device, current_level=current_level)
                        critic_metrics = self.agent.critic_update(batch)
                        last_critic_metrics = critic_metrics
                        self.log_dict(f"{name}_critic", critic_metrics, step)
                        actor_frozen = step < self.cfg.policy_warmup_steps or step < self.freeze_policy_until
                        self.log_scalar(f"{name}/actor_frozen", float(actor_frozen), step)
                        if actor_frozen:
                            if update_i == 0 and self.teacher_data is not None:
                                bc_batch = teacher_batch(
                                    self.teacher_data,
                                    self.cfg.batch_size,
                                    self.agent.device,
                                    success_fraction=self.cfg.bc_success_sample_fraction,
                                )
                                warmup_bc = self.agent.bc_update(bc_batch)
                                self.log_scalar(f"{name}/warmup_bc_loss", warmup_bc, step)
                            self.agent.soft_update_targets()
                            continue
                        update_count += 1
                        if update_count % max(1, self.cfg.policy_update_interval) != 0:
                            self.agent.soft_update_targets()
                            continue
                        frac = min(1.0, step / max(total_steps, 1))
                        bc_weight = self.cfg.bc_weight_start * (1.0 - frac) + self.cfg.bc_weight_end * frac
                        policy_metrics = self.agent.policy_update(batch, bc_weight=bc_weight)
                        last_policy_metrics = policy_metrics
                        self.log_scalar(f"{name}/bc_weight", bc_weight, step)
                        self.log_dict(f"{name}_policy", policy_metrics, step)
                        self.agent.soft_update_targets()
                if step >= next_log:
                    stats = self.replay.stats()
                    self.log_dict("replay", stats, step)
                    summary = self.console.summary()
                    postfix = {
                        "lvl": int(info.get("level", 0)),
                        "succ50": f"{summary.get('success', 0.0):.2f}",
                        "coll50": f"{summary.get('collision', 0.0):.2f}",
                        "stuck50": f"{summary.get('stuck', 0.0):.2f}",
                        "geo": f"{summary.get('geo_dist', 0.0):.2f}",
                        "replay": f"{stats['total']/1e6:.2f}M",
                        "critic": f"{last_critic_metrics.get('critic_loss', 0.0):.3f}",
                    }
                    if last_policy_metrics:
                        postfix["policy"] = f"{last_policy_metrics.get('policy_loss', 0.0):.3f}"
                        postfix["p_safe"] = f"{last_policy_metrics.get('policy_safety', 0.0):.3f}"
                    if pbar is not None:
                        pbar.set_postfix(postfix, refresh=False)
                    log_items = [
                        f"stage: {name}",
                        f"level: {info.get('level', 0)}",
                        f"succ50: {summary.get('success', 0.0):.2f}",
                        f"coll50: {summary.get('collision', 0.0):.2f}",
                        f"stuck50: {summary.get('stuck', 0.0):.2f}",
                        f"ep_len50: {summary.get('ep_len', 0.0):.0f}",
                        f"geo: {summary.get('geo_dist', 0.0):.2f}",
                        f"replay: {stats['total']/1e6:.2f}M",
                        f"demo: {stats['demo']/1e3:.0f}k",
                        f"online: {stats['online']/1e6:.2f}M",
                        f"critic: {last_critic_metrics.get('critic_loss', 0.0):.3f}",
                    ]
                    if last_policy_metrics:
                        log_items.append(f"policy: {last_policy_metrics.get('policy_loss', 0.0):.3f}")
                        log_items.append(f"p_safe: {last_policy_metrics.get('policy_safety', 0.0):.3f}")
                    if step < self.freeze_policy_until:
                        log_items.append(f"frozen_until: {self.freeze_policy_until}")
                    self.print_status(f"[Step: {step:<8}] | " + " | ".join(log_items))
                    while next_log <= step:
                        next_log += 1_000
                if step >= next_eval:
                    metrics = self.evaluate()
                    self.log_dict(f"{name}_eval", metrics, step)
                    self.print_status("[eval] " + " ".join(f"{k}={v:.3f}" for k, v in metrics.items()))
                    eval_score = self.eval_score(metrics)
                    self.log_scalar(f"{name}_eval/score", eval_score, step)
                    improved = eval_score > self.best_eval_score + self.cfg.best_score_min_delta
                    unsafe = metrics["collision"] >= self.cfg.collision_freeze_threshold
                    regressed = eval_score < self.best_eval_score - self.cfg.rollback_score_margin
                    if improved:
                        self.best_eval_score = eval_score
                        self.bad_eval_count = 0
                        save_checkpoint(self.cfg.best_dir, self.agent, self.cfg, step=step)
                        self.print_status(f"[best] score={eval_score:.3f} checkpoint={self.cfg.best_dir}")
                    level = int(metrics.get("level", self.env.unwrapped.curriculum.current_level))
                    level_best = self.best_level_scores.get(level, -np.inf)
                    level_improved = eval_score > level_best + self.cfg.best_score_min_delta
                    if level_improved:
                        self.best_level_scores[level] = eval_score
                        self.save_level_best(level, step)
                        self.print_status(f"[level-best] level={level} score={eval_score:.3f} checkpoint={self.level_best_dir(level)}")
                    if (not improved) and (not level_improved) and (unsafe or regressed):
                        self.bad_eval_count += 1
                        reason = f"unsafe={unsafe} regressed={regressed} score={eval_score:.3f} best={self.best_eval_score:.3f}"
                        level = int(metrics.get("level", self.env.unwrapped.curriculum.current_level))
                        if self.load_level_best_if_available(level) or self.load_best_if_available():
                            self.enter_recovery(step, reason)
                        self.log_scalar(f"{name}_eval/bad_eval_count", self.bad_eval_count, step)
                        if self.bad_eval_count >= self.cfg.max_bad_evals_before_stop:
                            self.print_status(f"[early-stop] {name} stopped after {self.bad_eval_count} bad evals; keeping best checkpoint")
                            save_checkpoint(os.path.join(self.cfg.checkpoint_dir, f"{name}_early_stop_{step}"), self.agent, self.cfg, step=step)
                            break
                    while next_eval <= step:
                        next_eval += self.cfg.eval_every_steps
                if step >= next_save:
                    save_checkpoint(os.path.join(self.cfg.checkpoint_dir, f"{name}_{step}"), self.agent, self.cfg, step=step)
                    while next_save <= step:
                        next_save += self.cfg.save_every_steps
        finally:
            self.progress_bar = None
            if pbar is not None:
                pbar.close()
        save_checkpoint(os.path.join(self.cfg.checkpoint_dir, name), self.agent, self.cfg, step=step)

    def evaluate(self):
        successes = 0
        collisions = 0
        stuck = 0
        spl_values = []
        progress_ratios = []
        final_dists = []
        safety_false_negative = 0
        predicted_safe = 0
        diversities = []
        for ep in range(self.cfg.eval_episodes):
            obs, _ = self.env.reset(seed=self.cfg.seed + 900000 + ep)
            self.agent.reset_memory()
            done = False
            path_len = 0.0
            last_pos, _ = self.env.unwrapped._get_robot_pose()
            start_geo = float(getattr(self.env.unwrapped, "prev_geodesic_dist", 1.0))
            info = {}
            while not done:
                graph = self.agent.graph_summary(self.env)
                action = self.agent.act(obs, graph, deterministic=True)
                predicted_safe += int(self.agent.last_predicted_safe)
                diversities.append(self.agent.estimate_candidate_diversity(obs, graph))
                obs, _, terminated, truncated, info = self.env.step(action)
                pos, _ = self.env.unwrapped._get_robot_pose()
                path_len += float(np.linalg.norm(pos - last_pos))
                last_pos = pos
                done = terminated or truncated
            reason = info.get("termination_reason", "unknown")
            successes += int(reason == "success")
            collisions += int(reason == "collision")
            safety_false_negative += int(reason == "collision" and self.agent.last_predicted_safe)
            stuck += int(reason == "stuck")
            final_geo = float(info.get("rewards", {}).get("geodesic_dist", info.get("distance_to_goal", start_geo)))
            progress_ratios.append((start_geo - final_geo) / max(start_geo, 1e-6))
            final_dists.append(float(info.get("distance_to_goal", np.nan)))
            spl_values.append(float(reason == "success") * start_geo / max(path_len, start_geo, 1e-6))
        return {
            "success": successes / max(self.cfg.eval_episodes, 1),
            "collision": collisions / max(self.cfg.eval_episodes, 1),
            "stuck": stuck / max(self.cfg.eval_episodes, 1),
            "level": float(getattr(self.env.unwrapped.curriculum, "current_level", 0)),
            "spl": float(np.mean(spl_values)),
            "progress": float(np.mean(progress_ratios)),
            "final_dist": float(np.nanmean(final_dists)),
            "candidate_diversity": float(np.mean(diversities)) if diversities else 0.0,
            "safety_fn": safety_false_negative / max(predicted_safe, 1),
        }

    def run_all(self):
        try:
            self.stage_teacher()
            self.stage_bc()
            self.stage_critic_pretrain()
            self.stage_online(self.cfg.online_steps, name="online")
            self.load_best_if_available()
            self.bad_eval_count = 0
            self.freeze_policy_until = self.cfg.policy_recovery_steps
            self.stage_online(self.cfg.robustness_steps, name="robustness")
            self.load_best_if_available()
            save_checkpoint(self.cfg.final_dir, self.agent, self.cfg, step=self.cfg.online_steps + self.cfg.robustness_steps)
            print(f"[done] final playable model: {self.cfg.final_dir}")
        finally:
            self.close()
