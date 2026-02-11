from typing import Dict, List, Optional, Generic, TypeVar

import numpy as np

from flatland.core.transition_map import TransitionMap
from flatland.envs.agent_utils import EnvAgent
from flatland.envs.rail_grid_transition_map import RailGridTransitionMap
from flatland.envs.rail_trainrun_data_structures import Waypoint

UnderlyingTransitionMapType = TypeVar('UnderlyingTransitionMapType', bound=TransitionMap)
UnderlyingDistanceMapType = TypeVar('UnderlyingDistanceMapType')
UnderlyingConfigurationType = TypeVar('UnderlyingConfigurationType')


class AbstractDistanceMap(Generic[UnderlyingTransitionMapType, UnderlyingDistanceMapType, UnderlyingConfigurationType]):
    def __init__(self, agents: List[EnvAgent]):
        self.distance_map = None
        self.agents_previous_computation = None
        self.reset_was_called = False
        self.agents: List[EnvAgent] = agents
        self.rail: Optional[RailGridTransitionMap] = None

    def set(self, distance_map: np.ndarray):
        """
        Set the distance map
        """
        self.distance_map = distance_map

    def get(self) -> np.ndarray:
        """
        Get the distance map
        """
        if self.reset_was_called:
            self.reset_was_called = False

            compute_distance_map = True
            # Don't compute the distance map if it was loaded
            if self.agents_previous_computation is None and self.distance_map is not None:
                compute_distance_map = False

            if compute_distance_map:
                self._compute(self.agents, self.rail)

        elif self.distance_map is None:
            self._compute(self.agents, self.rail)

        return self.distance_map

    def reset(self, agents: List[EnvAgent], rail: UnderlyingTransitionMapType):
        """
        Reset the distance map
        """
        self.reset_was_called = True
        self.agents: List[EnvAgent] = agents
        self.rail = rail

    def get_shortest_paths(self, max_depth: Optional[int] = None, agent_handle: Optional[int] = None) -> Dict[int, Optional[List[Waypoint]]]:
        raise NotImplementedError()

    def _compute(self, agents: List[EnvAgent], rail: UnderlyingTransitionMapType):
        raise NotImplementedError()

    def _set_distance(self, configuration: UnderlyingConfigurationType, target_nr: int, new_distance: int):
        raise NotImplementedError()

    def _get_distance(self, configuration: UnderlyingConfigurationType, target_nr: int):
        raise NotImplementedError()
