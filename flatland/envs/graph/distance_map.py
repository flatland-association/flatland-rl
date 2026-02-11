from collections import defaultdict
from typing import List, Dict

import numpy as np

from flatland.core.distance_map import AbstractDistanceMap
from flatland.core.distance_map_walker import DistanceMapWalker
from flatland.envs.agent_utils import EnvAgent
from flatland.envs.graph.rail_graph_transition_map import GraphTransitionMap


class GraphDistanceMap(AbstractDistanceMap[GraphTransitionMap, Dict[str, Dict[str, int]], str, str]):
    def __init__(self, agents: List[EnvAgent]):
        super().__init__(agents=agents, waypoint_init=str)

    def _compute(self, agents: List[EnvAgent], rail: GraphTransitionMap):
        self.agents_previous_computation = self.agents
        self.distance_map = defaultdict(lambda: defaultdict(lambda: np.inf))
        distance_map_walker = DistanceMapWalker[GraphDistanceMap, GraphTransitionMap, str, str](self)
        computed_targets = set()
        for i, agent in enumerate(agents):
            for target in agent.targets:
                if rail.is_valid_configuration(target) and target not in computed_targets:
                    # let's keep distance to all possible targets, not just best target.
                    # TODO we could pull-up this behaviour and generalize the grid distance map on this. Here, we index by target node, there we index by agend id (needing to duplicate slices).
                    # On the other hand, we could also index the nodes and have a an all-to-one slice for all targets.
                    distance_map_walker._distance_map_walker(rail, target, [target])
                    computed_targets.add(target)

    def _set_distance(self, configuration: str, target_nr: str, new_distance: int):
        self.distance_map[configuration][target_nr] = new_distance

    def _get_distance(self, configuration: str, target_nr: str):
        return self.get()[configuration][target_nr]
