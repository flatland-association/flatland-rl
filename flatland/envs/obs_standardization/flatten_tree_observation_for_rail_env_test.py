from typing import Optional, List

import numpy as np
from flatland.envs.observations import TreeObsForRailEnv

from flatland.envs.obs_standardization.flatten_tree_observation_for_rail_env import normalize_tree_observation

class FlattenTreeObsForRailEnv(TreeObsForRailEnv):

    def _normalize_observation_tree_obs(self, agent_handle, raw_states, observation_radius=2):
        return normalize_tree_observation(raw_states[agent_handle],
                                          self.max_depth,
                                          observation_radius=observation_radius)

    def _transform_state(self, states):
        ret_obs = None
        # Update replay buffer and train agent
        for agent_handle in self.env.get_agent_handles():

            # Preprocess the new observations
            if states[agent_handle]:
                normalized_obs = self._normalize_observation_tree_obs(agent_handle, states)
                if ret_obs is None:
                    ret_obs = np.zeros((self.env.get_num_agents(), len(normalized_obs)))

                ret_obs[agent_handle] = normalized_obs

        return ret_obs

    def get_many(self, handles: Optional[List[int]] = None):
        return self._transform_state(super(FlattenTreeObsForRailEnv, self).get_many(handles))
