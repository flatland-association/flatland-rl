from collections import defaultdict
from typing import List, Dict

import numpy as np

from flatland.core.distance_map import AbstractDistanceMap
from flatland.core.distance_map_walker import DistanceMapWalker
from flatland.envs.agent_utils import EnvAgent
from flatland.envs.graph.rail_graph_transition_map import GraphTransitionMap


class GraphDistanceMap(AbstractDistanceMap[GraphTransitionMap, Dict[int, Dict[str, int]], str, str]):
    def __init__(self, agents: List[EnvAgent]):
        super().__init__(agents=agents, waypoint_init=str)

    def _compute(self, agents: List[EnvAgent], rail: GraphTransitionMap):
        self.agents_previous_computation = self.agents
        self.distance_map = defaultdict(lambda: defaultdict(lambda: np.inf))
        distance_map_walker = DistanceMapWalker[GraphDistanceMap, GraphTransitionMap, str](self)
        computed_targets = []
        for i, agent in enumerate(agents):
            if agent.targets not in computed_targets:
                distance_map_walker._distance_map_walker(rail, agent.handle, agent.targets)
            else:
                self.distance_map[i] = self.distance_map[computed_targets.index(agent.targets)]
            computed_targets.append(agent.targets)

    def _set_distance(self, configuration: str, target_nr: int, new_distance: int):
        self.distance_map[target_nr][configuration] = new_distance

    def _get_distance(self, configuration: str, target_nr: int):
        return self.get()[target_nr][configuration]
