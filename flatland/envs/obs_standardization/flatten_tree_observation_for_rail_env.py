import numpy as np
#from flatland.envs.observations import TreeObsForRailEnv


class TreeObsStandardizer:

    tree_explored_actions_char = ['L', 'F', 'R', 'B']

    def __init__(self):
        return

    def max_lt(self, seq, val):
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


    def min_gt(self, seq, val):
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


    def norm_obs_clip(self, obs, clip_min=-1, clip_max=1, fixed_radius=0, normalize_to_range=False):
        """
        This function returns the difference between min and max value of an observation
        :param obs: observation that should be normalized
        :param clip_min: min value where observation will be clipped
        :param clip_max: max value where observation will be clipped
        :return: returns normalized and clipped observation
        """
        if fixed_radius > 0:
            max_obs = fixed_radius
        else:
            max_obs = max(1, self.max_lt(obs, 1000)) + 1

        min_obs = 0  # min(max_obs, min_gt(obs, 0))
        if normalize_to_range:
            min_obs = self.min_gt(obs, 0)
        if min_obs > max_obs:
            min_obs = max_obs
        if max_obs == min_obs:
            return np.clip(np.array(obs) / max_obs, clip_min, clip_max)
        norm = np.abs(max_obs - min_obs)
        return np.clip((np.array(obs) - min_obs) / norm, clip_min, clip_max)


    def _split_node_into_feature_groups(self, node) -> (np.ndarray, np.ndarray, np.ndarray):
        data = np.zeros(6)
        distance = np.zeros(1)
        agent_data = np.zeros(4)

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

        return data, distance, agent_data


    def _split_subtree_into_feature_groups(self, node, current_tree_depth: int, max_tree_depth: int) -> (
            np.ndarray, np.ndarray, np.ndarray):
        if node == -np.inf:
            remaining_depth = max_tree_depth - current_tree_depth
            # reference: https://stackoverflow.com/questions/515214/total-number-of-nodes-in-a-tree-data-structure
            num_remaining_nodes = int((4 ** (remaining_depth + 1) - 1) / (4 - 1))
            return [-np.inf] * num_remaining_nodes * 6, [-np.inf] * num_remaining_nodes, [-np.inf] * num_remaining_nodes * 4

        data, distance, agent_data = self._split_node_into_feature_groups(node)

        if not node.childs:
            return data, distance, agent_data

        for direction in self.tree_explored_actions_char:
            sub_data, sub_distance, sub_agent_data = self._split_subtree_into_feature_groups(node.childs[direction],
                                                                                        current_tree_depth + 1,
                                                                                        max_tree_depth)
            data = np.concatenate((data, sub_data))
            distance = np.concatenate((distance, sub_distance))
            agent_data = np.concatenate((agent_data, sub_agent_data))

        return data, distance, agent_data


    def split_tree_into_feature_groups(self, tree, max_tree_depth: int) -> (np.ndarray, np.ndarray, np.ndarray):
        """
        This function splits the tree into three difference arrays of values
        """
        data, distance, agent_data = self._split_node_into_feature_groups(tree)

        for direction in self.tree_explored_actions_char:
            sub_data, sub_distance, sub_agent_data = self._split_subtree_into_feature_groups(tree.childs[direction], 1,
                                                                                        max_tree_depth)
            data = np.concatenate((data, sub_data))
            distance = np.concatenate((distance, sub_distance))
            agent_data = np.concatenate((agent_data, sub_agent_data))

        return data, distance, agent_data


    def standardize_tree_observation(self, observation, tree_depth: int, observation_radius=0):
        """
        This function normalizes the observation used by the RL algorithm
        """
        data, distance, agent_data = self.split_tree_into_feature_groups(observation, tree_depth)

        data = self.norm_obs_clip(data, fixed_radius=observation_radius)
        distance = self.norm_obs_clip(distance, normalize_to_range=True)
        agent_data = np.clip(agent_data, -1, 1)
        normalized_obs = np.concatenate((np.concatenate((data, distance)), agent_data))
        return normalized_obs
