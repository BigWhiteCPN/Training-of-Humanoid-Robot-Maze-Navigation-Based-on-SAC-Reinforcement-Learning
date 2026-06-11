# End2End Navigation Trainer

This directory contains the integrated end-to-end navigation training system.

It keeps the existing three-layer robot stack:

```text
advanced high-level navigator
    -> waypoint / command interface
    -> existing low-level locomotion policy
```

## Files

- `config.py`: central training and path configuration.
- `models.py`: encoder, flow-matching waypoint policy, Q/safety/progress critics.
- `agent.py`: candidate sampling, critic reranking, updates, target networks.
- `replay.py`: episode replay, horizon safety/progress labels, level/bucket-aware sampling, HER relabeling.
- `topological_memory.py`: lightweight online topological summary.
- `mpc_tracker.py`: waypoint-sequence MPC tracker used by `env.py`.
- `teacher.py`: planner teacher dataset collection using the existing environment.
- `trainer.py`: one-command staged pipeline.
- `checkpoint.py`: save/load final playable agent.
- `train.py`: training entrypoint.
- `play.py`: agent play entrypoint.

## Train

Run from the repository root:

```bash
python -m scripts.visual_train.end2end_nav_train.train
```

The pipeline automatically runs:

```text
1. teacher dataset collection
2. flow waypoint BC pretraining
3. safety/progress/Q critic pretraining
4. online fine-tuning with candidate reranking
5. robustness fine-tuning with sensor dropout, action noise, latency, and dynamic obstacles
6. final checkpoint export
```

Final checkpoint:

```text
./advanced_end2end_nav_logs/final_model
```

## Play

```bash
python -m scripts.visual_train.end2end_nav_train.play \
  --checkpoint ./advanced_end2end_nav_logs/final_model
```

## Notes

This package is intentionally separate from the SB3 SAC training script. SB3 is
not a good fit for flow-matching candidate generation, horizon safety/progress
critics, topological memory, HER episode relabeling, custom replay buckets, and
custom candidate reranking.
