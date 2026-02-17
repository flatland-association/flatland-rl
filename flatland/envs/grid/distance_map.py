from typing import List, Tuple

import numpy as np

from flatland.core.distance_map import AbstractDistanceMap
from flatland.core.distance_map_walker import DistanceMapWalker
from flatland.envs.agent_utils import EnvAgent
from flatland.envs.rail_grid_transition_map import RailGridTransitionMap
from flatland.envs.rail_trainrun_data_structures import Waypoint


def _waypoint(c):
    return Waypoint(*c)

class DistanceMap(AbstractDistanceMap[RailGridTransitionMap, np.ndarray, Tuple[Tuple[int, int], int], Waypoint]):
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

    def _compute(self, agents: List[EnvAgent], rail: RailGridTransitionMap):
        """
        This function computes the distance maps for each unique target. Thus, if several targets are the same
        we only compute the distance for them once and copy to all targets with the same position.
        :param agents: All the agents in the environment, independent of their current status
        :param rail: The rail transition map

        """
        self.agents_previous_computation = self.agents
        self.distance_map = np.full(shape=(len(agents),
                                           self.env_height,
                                           self.env_width,
                                           4),
                                    fill_value=np.inf
                                    )
        distance_map_walker = DistanceMapWalker[DistanceMap, RailGridTransitionMap, Tuple[Tuple[int, int], int]](self)
        computed_targets = []
        for i, agent in enumerate(agents):
            # TODO safe?
            if agent.target not in computed_targets:
                targets = [target for target in agent.targets if rail.is_valid_configuration(target)]
                distance_map_walker._distance_map_walker(rail, i, targets)
            else:
                # just copy the distance map form other agent with same target (performance)
                self.distance_map[i, :, :, :] = np.copy(
                    self.distance_map[computed_targets.index(agent.target), :, :, :])
            computed_targets.append(agent.target)

    def _set_distance(self, configuration: Tuple[Tuple[int, int], int], target_nr: int, new_distance: int):
        position, direction = configuration
        self.distance_map[target_nr, *position, direction] = new_distance

    def _get_distance(self, configuration: Tuple[Tuple[int, int], int], target_nr: int):
        distance_map = self.get()
        position, direction = configuration
        return distance_map[target_nr, *position, direction]
