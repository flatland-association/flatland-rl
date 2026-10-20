import math
from collections import defaultdict
from typing import Dict, List, Optional, Generic, TypeVar, Callable, Tuple

from flatland.core.transition_map import TransitionMap
from flatland.envs.agent_utils import EnvAgent
from flatland.envs.rail_grid_transition_map import RailGridTransitionMap

TransitionMapT = TypeVar('TransitionMapT', bound=TransitionMap)
DistanceMapT = TypeVar('DistanceMapT')
EntryPointT = TypeVar('EntryPointT')
WaypointT = TypeVar('WaypointT')


def _infinite_distance():
    return math.inf


class EntryPointDistanceMap(Generic[TransitionMapT, DistanceMapT, EntryPointT, WaypointT]):
    """
    Base distance map collecting the distance from every entry point visited during the BFS walk to the
    effective target entry point reached, keyed by (source_entry_point, target_entry_point) - agnostic
    of any numeric target_nr (agent handle), which `DistanceMapWalker` has no notion of.
    """

    def __init__(self, agents: List[EnvAgent], waypoint_init: Callable[[EntryPointT], WaypointT]):
        self.agents: List[EnvAgent] = agents
        self.rail: Optional[RailGridTransitionMap] = None
        self.waypoint_init = waypoint_init
        self.distances: Dict[
            Tuple[EntryPointT, EntryPointT], int
        ] = defaultdict(_infinite_distance)

    def reset(self, agents: List[EnvAgent], rail: TransitionMapT):
        """
        Reset the distance map
        """
        self.agents: List[EnvAgent] = agents
        self.rail = rail
        self.distances: Dict[
            Tuple[EntryPointT, EntryPointT], int
        ] = defaultdict(_infinite_distance)

    def _set_distance(self, source_entry_point: EntryPointT,
                      target_entry_point: EntryPointT, new_distance: int):
        self.distances[(source_entry_point, target_entry_point)] = new_distance

    def _get_distance(self, source_entry_point: EntryPointT,
                      target_entry_point: EntryPointT) -> int:
        return self.distances[(source_entry_point, target_entry_point)]
