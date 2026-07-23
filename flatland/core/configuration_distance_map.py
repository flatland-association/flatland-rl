import math
from collections import defaultdict
from typing import Dict, List, Optional, Generic, TypeVar, Callable, Tuple

from flatland.core.transition_map import TransitionMap
from flatland.envs.agent_utils import EnvAgent
from flatland.envs.rail_grid_transition_map import RailGridTransitionMap

UnderlyingTransitionMapType = TypeVar('UnderlyingTransitionMapType', bound=TransitionMap)
UnderlyingDistanceMapType = TypeVar('UnderlyingDistanceMapType')
UnderlyingConfigurationType = TypeVar('UnderlyingConfigurationType')
UnderlyingWaypointType = TypeVar('UnderlyingWaypointType')


def _infinite_distance():
    return math.inf


class ConfigurationDistanceMap(Generic[UnderlyingTransitionMapType, UnderlyingDistanceMapType, UnderlyingConfigurationType, UnderlyingWaypointType]):
    """
    Base distance map collecting the distance from every configuration visited during the BFS walk to the
    effective target configuration reached, keyed by (source_configuration, target_configuration) - agnostic
    of any numeric target_nr (agent handle), which `DistanceMapWalker` has no notion of.
    """

    def __init__(self, agents: List[EnvAgent], waypoint_init: Callable[[UnderlyingConfigurationType], UnderlyingWaypointType]):
        self.agents: List[EnvAgent] = agents
        self.rail: Optional[RailGridTransitionMap] = None
        self.waypoint_init = waypoint_init
        self.distances: Dict[
            Tuple[UnderlyingConfigurationType, UnderlyingConfigurationType], int
        ] = defaultdict(_infinite_distance)

    def reset(self, agents: List[EnvAgent], rail: UnderlyingTransitionMapType):
        """
        Reset the distance map
        """
        self.agents: List[EnvAgent] = agents
        self.rail = rail
        self.distances: Dict[
            Tuple[UnderlyingConfigurationType, UnderlyingConfigurationType], int
        ] = defaultdict(_infinite_distance)

    def _set_distance(self, source_configuration: UnderlyingConfigurationType,
                      target_configuration: UnderlyingConfigurationType, new_distance: int):
        self.distances[(source_configuration, target_configuration)] = new_distance

    def _get_distance(self, source_configuration: UnderlyingConfigurationType,
                      target_configuration: UnderlyingConfigurationType) -> int:
        return self.distances[(source_configuration, target_configuration)]
