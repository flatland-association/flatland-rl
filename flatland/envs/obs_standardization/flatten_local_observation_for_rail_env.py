import numpy as np
from typing import List, Optional

class LocalObsStandardizer:

    def __init__(self):
        return

    def standardize_local_observation(self, local_rail_obs, obs_map_state, obs_other_agents_state, direction, view_size):

        local_rail_obs_flat = local_rail_obs.reshape(view_size * 16)
        # TODO: shall we really use a vector of 16 elements per env-cell or convert the 16 bits to a single integer value?

        obs_map_state_flat = obs_map_state.reshape(view_size * 2)

        obs_other_agents_state_flat = obs_other_agents_state.reshape(view_size * 4)

        normalized_obs = np.concatenate((local_rail_obs_flat, obs_map_state_flat, obs_other_agents_state_flat, direction))

        return normalized_obs
