from collections import defaultdict
from typing import List, Dict

import numpy as np

from flatland.core.distance_map import AgentSourceTargetDistanceMap
from flatland.envs.agent_utils import EnvAgent
from flatland.envs.graph.rail_graph_transition_map import GraphTransitionMap


class GraphDistanceMap(AgentSourceTargetDistanceMap[GraphTransitionMap, Dict[int, Dict[str, int]], str, str]):
    def __init__(self, agents: List[EnvAgent]):
        super().__init__(agents=agents, waypoint_init=str)

    def _new_distance_map(self, num_agents: int) -> Dict[int, Dict[str, int]]:
        return defaultdict(lambda: defaultdict(lambda: np.inf))

    def _valid_targets(self, agent: EnvAgent, rail: GraphTransitionMap):
        return agent.targets

    def _all_configurations(self, rail: GraphTransitionMap):
        return rail.g.nodes

    def _copy_agent_distance(self, target_nr: int, source_target_nr: int):
        self.distance_map[target_nr] = self.distance_map[source_target_nr]

    def _set_agent_distance(self, source_configuration: str, target_nr: int, new_distance: int):
        self.distance_map[target_nr][source_configuration] = new_distance
