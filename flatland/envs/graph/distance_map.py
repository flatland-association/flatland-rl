from typing import Dict, List, Optional, Tuple, Any

from flatland.core.distance_map import AbstractDistanceMap
from flatland.envs.agent_utils import EnvAgent
from flatland.envs.graph.rail_graph_transition_map import GraphTransitionMap
from flatland.envs.rail_trainrun_data_structures import Waypoint


class GraphDistanceMap(AbstractDistanceMap[GraphTransitionMap, Any, str]):
    # TODO implement/generalize distance map for graph
    def _compute(self, agents: List[EnvAgent], rail: GraphTransitionMap):
        pass

    def get_shortest_paths(self, max_depth: Optional[int] = None, agent_handle: Optional[int] = None) -> Dict[int, Optional[List[Waypoint]]]:
        if agent_handle is not None:
            return {agent_handle: []}
        return {a.handle: [] for a in self.agents}

    def _set_distance(self, configuration: Tuple[Tuple[int, int], int], target_nr: int, new_distance: int):
        # TODO
        pass

    def _get_distance(self, configuration: Tuple[Tuple[int, int], int], target_nr: int):
        # TODO
        pass
