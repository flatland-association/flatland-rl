from typing import List, Tuple

import networkx as nx
from attr import attrs, attrib

from flatland.core.graph.grid_to_graph import GraphTransitionMap

GridNode = Tuple[int, int, int]


@attrs
class DecisionPointGraphEdgeData:
    path = attrib(type=List[GridNode])


# TODO naming: decision point graph, simplified graph, collapsed graph, ....?
class DecisionPointGraph:
    def __init__(self, g: nx.MultiDiGraph):
        self.g = g

    @staticmethod
    def _explore_branch(g: nx.DiGraph, u: GridNode, v: GridNode) -> List[GridNode]:
        branch = [u, v]
        successors = list(g.successors(v))
        assert len(successors) > 0
        while len(successors) == 1:
            successor = successors[0]
            branch.append(successor)
            successors = list(g.successors(successor))
            assert len(successors) > 0
        return branch

    @staticmethod
    def fromGraphTransitionMap(gtm: GraphTransitionMap) -> "DecisionPointGraph":
        g = nx.MultiDiGraph()

        # find decision points (ie. nodes with more than one successor (=neighbor in the directed graph))
        micro = gtm.g
        decision_nodes = {s for s in micro.nodes if len(list(micro.successors(s))) > 1}

        # add edge for
        for dp in decision_nodes:
            for n in micro.successors(dp):
                branch = DecisionPointGraph._explore_branch(micro, dp, n)
                g.add_edge(branch[0], branch[-1], d=DecisionPointGraphEdgeData(path=branch))
        return DecisionPointGraph(g)
