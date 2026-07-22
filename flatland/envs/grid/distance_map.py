from typing import List, Tuple

import numpy as np

from flatland.core.distance_map import AgentSourceTargetDistanceMap
from flatland.envs.agent_utils import EnvAgent
from flatland.envs.rail_grid_transition_map import RailGridTransitionMap
from flatland.envs.rail_trainrun_data_structures import Waypoint


def _waypoint(c):
    return Waypoint(*c)


class DistanceMap(
    AgentSourceTargetDistanceMap[RailGridTransitionMap, np.ndarray, Tuple[Tuple[int, int], int], Waypoint]
):
    def __init__(self, agents: List[EnvAgent], env_height: int, env_width: int):
        super().__init__(agents=agents, waypoint_init=_waypoint)
        self.env_height = env_height
        self.env_width = env_width

    def reset(self, agents: List[EnvAgent], rail: RailGridTransitionMap):
        """
        Reset the distance map
        """
        super().reset(agents=agents, rail=rail)
        self.env_height = rail.height
        self.env_width = rail.width

    def _new_distance_map(self, num_agents: int) -> np.ndarray:
        return np.full(shape=(num_agents, self.env_height, self.env_width, 4), fill_value=np.inf)

    def _valid_targets(self, agent: EnvAgent, rail: RailGridTransitionMap) -> List[Tuple[Tuple[int, int], int]]:
        return [target for target in agent.targets if rail.is_valid_configuration(target)]

    def _copy_agent_distance(self, target_nr: int, source_target_nr: int):
        # just copy the distance map from other agent with same target (performance)
        self.distance_map[target_nr, :, :, :] = np.copy(self.distance_map[source_target_nr, :, :, :])

    def _set_agent_distance(self, source_configuration: Tuple[Tuple[int, int], int], target_nr: int, new_distance: int):
        (r, c), direction = source_configuration
        self.distance_map[target_nr, r, c, direction] = new_distance
