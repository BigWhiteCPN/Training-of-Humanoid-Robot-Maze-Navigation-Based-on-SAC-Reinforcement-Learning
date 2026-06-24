"""Transformer experiment environment.

The environment intentionally inherits the original random-map environment
without changing simulation, observations, rewards, actions, or termination
logic. Only the policy feature extractor differs in the transformer scripts.
"""

from robot_visual_env_random_map import RobotVisualEnv as RandomMapRobotVisualEnv


class RobotVisualEnv(RandomMapRobotVisualEnv):
    pass

