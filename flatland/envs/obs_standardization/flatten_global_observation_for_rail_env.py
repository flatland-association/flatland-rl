import numpy as np
from typing import List, Optional

class GlobalObsStandardizer:

    def __init__(self):
        return

    def standardize_global_observation(self, transition_maps, agent_states, agent_targets, env_width, env_height):

        transition_maps_flat = transition_maps.reshape(env_width * env_height * 16)
        # TODO: shall we really use a vector of 16 elements per env-cell or convert the 16 bits to a single integer value?

        agent_states_flat = agent_states.reshape(env_width * env_height * 5)

        agent_targets_flat = agent_targets.reshape(env_width * env_height * 2)

        normalized_obs = np.concatenate((transition_maps_flat, agent_states_flat, agent_targets_flat))

        return normalized_obs

    def _transform_state(self, states):

        ret_obs = None

        for agent_handle in self.env.get_agent_handles():

            env_state = states[agent_handle]

            transition_maps = env_state[0]
            transition_maps_flat = transition_maps.reshape((self.env.width * self.env.height * 16))
            # TODO: shall we really use a vector of 16 elements per env-cell or convert the 16 bits to a single integer value?

            agent_states = env_state[1]
            agent_states_flat = agent_states.reshape((self.env.width * self.env.height * 5))

            agent_targets = env_state[2]
            agent_targets_flat = agent_targets.reshape((self.env.width * self.env.height * 2))

            normalized_obs = np.concatenate(transition_maps_flat, agent_states_flat, agent_targets_flat)

            if ret_obs is None:
                ret_obs = np.zeros((self.env.get_num_agents(), len(normalized_obs)))

            ret_obs[agent_handle] = normalized_obs

        return ret_obs

