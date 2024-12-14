from typing import Optional

import gymnasium as gym
import numpy as np

from flatland.core.env_observation_builder import AgentHandle
from flatland.envs.observations import TreeObsForRailEnv, Node
from flatland.ml.observations.gym_observation_builder import GymObservationBuilder


# from https://github.com/aiAdrian/flatland_solver_policy/blob/main/observation/flatland/flatten_tree_observation_for_rail_env/flatten_tree_observation_for_rail_env_utils.py
# initially from https://github.com/instadeepai/Mava/blob/0.0.9/mava/wrappers/flatland.py
def max_lt(seq, val):
    """
    Return greatest item in seq for which item < val applies.
    None is returned if seq was empty or all items in seq were >= val.
    """
    max = 0
    idx = len(seq) - 1
    while idx >= 0:
        if seq[idx] < val and seq[idx] >= 0 and seq[idx] > max:
            max = seq[idx]
        idx -= 1
    return max


def min_gt(seq, val):
    """
    Return smallest item in seq for which item > val applies.
    None is returned if seq was empty or all items in seq were >= val.
    """
    min = np.inf
    idx = len(seq) - 1
    while idx >= 0:
        if seq[idx] >= val and seq[idx] < min:
            min = seq[idx]
        idx -= 1
    return min


# TODO documentataion
def norm_obs_clip(obs, clip_min=-1, clip_max=1, fixed_radius=0, normalize_to_range=False):
    """
    This function returns the difference between min and max value of an observation.

    Parameters
    ----------
    obs
        Observation that should be normalized
    clip_min
        min value where observation will be clipped
    clip_max
        max value where observation will be clipped
    fixed_radius
    normalize_to_range

    Returns
    -------
    normalized and clipped observation
    """
    if fixed_radius > 0:
        max_obs = fixed_radius
    else:
        max_obs = max(1, max_lt(obs, 1000)) + 1

    min_obs = 0
    if normalize_to_range:
        min_obs = min_gt(obs, 0)
    if min_obs > max_obs:
        min_obs = max_obs
    if max_obs == min_obs:
        return np.clip(np.array(obs) / max_obs, clip_min, clip_max)
    norm = np.abs(max_obs - min_obs)
    return np.clip((np.array(obs) - min_obs) / norm, clip_min, clip_max)


def _split_node_into_feature_groups(node: Node) -> (np.ndarray, np.ndarray, np.ndarray):
    data = np.zeros(6)
    distance = np.zeros(1)
    agent_data = np.zeros(5)

    data[0] = node.dist_own_target_encountered
    data[1] = node.dist_other_target_encountered
    data[2] = node.dist_other_agent_encountered
    data[3] = node.dist_potential_conflict
    data[4] = node.dist_unusable_switch
    data[5] = node.dist_to_next_branch

    distance[0] = node.dist_min_to_target

    agent_data[0] = node.num_agents_same_direction
    agent_data[1] = node.num_agents_opposite_direction
    agent_data[2] = node.num_agents_malfunctioning
    agent_data[3] = node.speed_min_fractional
    agent_data[4] = node.num_agents_ready_to_depart

    return data, distance, agent_data


def _split_subtree_into_feature_groups(node, current_tree_depth: int, max_tree_depth: int) -> (
    np.ndarray, np.ndarray, np.ndarray):
    if node == -np.inf:
        remaining_depth = max_tree_depth - current_tree_depth
        # reference: https://stackoverflow.com/questions/515214/total-number-of-nodes-in-a-tree-data-structure
        num_remaining_nodes = int((4 ** (remaining_depth + 1) - 1) / (4 - 1))
        return [-np.inf] * num_remaining_nodes * 6, [-np.inf] * num_remaining_nodes, [-np.inf] * num_remaining_nodes * 5

    data, distance, agent_data = _split_node_into_feature_groups(node)

    if not node.childs:
        return data, distance, agent_data

    for direction in TreeObsForRailEnv.tree_explored_actions_char:
        sub_data, sub_distance, sub_agent_data = _split_subtree_into_feature_groups(node.childs[direction],
                                                                                    current_tree_depth + 1,
                                                                                    max_tree_depth)
        data = np.concatenate((data, sub_data))
        distance = np.concatenate((distance, sub_distance))
        agent_data = np.concatenate((agent_data, sub_agent_data))

    return data, distance, agent_data


# TODO document feature groups
def split_tree_into_feature_groups(tree, max_tree_depth: int) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    This function splits the tree into three difference arrays of values
    """
    data, distance, agent_data = _split_node_into_feature_groups(tree)

    for direction in TreeObsForRailEnv.tree_explored_actions_char:
        sub_data, sub_distance, sub_agent_data = _split_subtree_into_feature_groups(tree.childs[direction], 1,
                                                                                    max_tree_depth)
        data = np.concatenate((data, sub_data))
        distance = np.concatenate((distance, sub_distance))
        agent_data = np.concatenate((agent_data, sub_agent_data))

    return data, distance, agent_data


# TODO make normalization optional? Re-factor, so we first have flattening and re-arranging in groups and then normlization afterwards.
def normalize_observation(observation, tree_depth: int, observation_radius=0):
    """
    This function normalizes the observation used by the RL algorithm
    """
    data, distance, agent_data = split_tree_into_feature_groups(observation, tree_depth)

    data = norm_obs_clip(data, fixed_radius=observation_radius)
    distance = norm_obs_clip(distance, normalize_to_range=True)
    agent_data = np.clip(agent_data, -1, 1)
    normalized_obs = np.concatenate((np.concatenate((data, distance)), agent_data))
    return normalized_obs


# TODO passive_env_checker.py:164: UserWarning: WARN: The obs returned by the `reset()` method was expecting numpy array dtype to be float32, actual type: float64
class FlattenTreeObsForRailEnv(GymObservationBuilder[np.ndarray], TreeObsForRailEnv):
    """
    Gym-ified and flattend normalized tree observation.
    """

    NUM_FEATURES = 12
    NUM_BRANCHES = 4

    def __init__(self, observation_radius: int = 2, **kwargs):
        super().__init__(**kwargs)
        self.observation_radius = observation_radius

    def get(self, handle: Optional[AgentHandle] = 0) -> np.ndarray:
        obs = super(FlattenTreeObsForRailEnv, self).get(handle)
        obs = normalize_observation(obs, tree_depth=self.max_depth, observation_radius=self.observation_radius)
        return obs

    def get_observation_space(self, handle: int = 0) -> gym.Space:
        # max_depth=1 -> 60, max_depth=2 -> 240, max_depth=3 -> 972, ...
        k = FlattenTreeObsForRailEnv.NUM_FEATURES
        for _ in range(self.max_depth):
            k = k * FlattenTreeObsForRailEnv.NUM_BRANCHES + FlattenTreeObsForRailEnv.NUM_FEATURES
        return gym.spaces.Box(-1, 2, (k,), dtype=np.float64)
