from collections import defaultdict
from typing import List, Tuple

import networkx as nx
from attr import attrs
from lxml.etree import attrib

from core.grid.grid4_utils import get_new_position
from core.transition_map import GridTransitionMap
from envs.rail_env import RailEnv


class GraphTransitionMap:
    def __init__(self, g: nx.DiGraph):
        self.g = g
        self.cell_in_pins = defaultdict(lambda: set())
        self.cell_out_pins = defaultdict(lambda: set())
        for n in self.g:
            cell = n[:2]
            self.cell_in_pins[cell].add(n)
            for succ in g.successors(n):
                succ_cell = succ[:2]
                self.cell_out_pins[succ_cell].add(cell)

    @staticmethod
    def grid_to_digraph(transition_map: GridTransitionMap) -> nx.DiGraph:
        g = nx.DiGraph()
        for r in range(transition_map.height):
            for c in range(transition_map.width):
                for d in range(4):
                    possible_transitions = transition_map.get_transitions(r, c, d)
                    for new_direction in range(4):
                        if possible_transitions[new_direction]:
                            new_position = get_new_position((r, c), new_direction)
                            g.add_edge((r, c, d), (*new_position, new_direction))
        return g

    @staticmethod
    def fromRailEnv(env: RailEnv) -> "GraphTransitionMap":
        return GraphTransitionMap(GraphTransitionMap.grid_to_digraph(env.rail))


@attrs
class DecisionPointGraphEdgeData:
    path = attrib(type=List[Tuple[int, int, int]])


class DecisionPointGraph:
    def __init__(self, g: GraphTransitionMap):
        self.g = g

    @staticmethod
    def fromGraphTransitionMap(g: GraphTransitionMap):
        # TODO _create_simplified_graph https://github.com/aiAdrian/flatland_railway_extension/blob/e2b15bdd851ad32fb26c1a53f04621a3ca38fc00/flatland_railway_extension/FlatlandGraphBuilder.py

        return DecisionPointGraph(g)
