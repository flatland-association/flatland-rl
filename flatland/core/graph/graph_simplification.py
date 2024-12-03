from typing import List, Tuple

import networkx as nx
from attr import attrs, attrib

from flatland.core.graph.grid_to_graph import GraphTransitionMap

GridNode = Tuple[int, int, int]  # row, column, heading (at cell entry)


@attrs
class DecisionPointGraphEdgeData:
    """
    The edge data of a decision point overlay graph.

    Attributes
    ----------
    path: List[GridNode]
        The list of collapsed cells, starting on a facing switch (e.g. where a decision has been taken upon entering).
    len: int
        Number of collapsed cells.

    """
    path = attrib(type=List[GridNode])
    len = attrib(type=int)


class DecisionPointGraph:
    """
    Overlay on top of Flatland 3 grid where consecutive cells where agents cannot choose between alternative paths are collapsed into a single edge.
    A reference to the underlying grid nodes is maintained.
    The edge length is the number of cells "collapsed" into this edge.
    See `DecisionPointGraphEdgeData`.

    Attributes
    ----------
    g: nx.MultiDiGraph
        The decision point graph. Nodes have type `GridNode` and edge data has an attributed `d` of type `DecisionPointGraphEdgeData`.
    """

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
        """
        Factory method to derive `DecisionPointGraph` from `GraphTransitionMap`.

        Parameters
        ----------
        gtm: GraphTransitionMap
             The Flatland 3 graph transition map.

        Returns
        -------
        The overlaid decision point graph.
        """
        # multiple paths can be between consecutive decision points
        g = nx.MultiDiGraph()

        # find decision points (ie. nodes with more than one successor (=neighbor in the directed graph))
        micro = gtm.g
        decision_nodes = {s for s in micro.nodes if len(list(micro.successors(s))) > 1}

        # add edge for
        for dp in decision_nodes:
            for n in micro.successors(dp):
                branch = DecisionPointGraph._explore_branch(micro, dp, n)
                g.add_edge(branch[0], branch[-1], d=DecisionPointGraphEdgeData(path=branch, len=len(branch)))
        return DecisionPointGraph(g)
