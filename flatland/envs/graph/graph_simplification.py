from typing import List

import networkx as nx
from attr import attrs, attrib

from flatland.envs.graph.rail_graph_transition_map import GraphTransitionMap, GridNode
from flatland.envs.rail_trainrun_data_structures import Waypoint


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
    Overlay on top of Flatland 3 grid where agents need to choose between alternative paths before entering are collapsed into a single edge.
    A reference to the underlying grid nodes is maintained: all but last positions have only one neighbor.
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
            if successor == u:
                # loop
                break
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

        # add edge starting at each decision point
        closed = set()
        for dp in decision_nodes:
            for n in micro.successors(dp):
                branch = DecisionPointGraph._explore_branch(micro, dp, n)
                g.add_edge(branch[0], branch[-1], d=DecisionPointGraphEdgeData(path=branch, len=len(branch)))
                for u, v, in zip(branch, branch[1:]):
                    closed.add((u, v))

        # special cases closed loops
        open = set(micro.edges) - closed
        while not len(open) == 0:
            u, v = next(iter(open))
            branch = DecisionPointGraph._explore_branch(micro, u, v)
            g.add_edge(branch[0], branch[-1], d=DecisionPointGraphEdgeData(path=branch, len=len(branch)))
            for u_, v_, in zip(branch, branch[1:]):
                open.discard((u_, v_))

        return DecisionPointGraph(g)

    @staticmethod
    def micro_edge_to_waypoint(p1: GridNode, p2: GridNode) -> Waypoint:
        """
        Micro edge ((u,v,_), (_,_,d)) <=> directed cell (u,v,d).

        Parameters
        ----------
        p1 : GridNode
            starting vertex of directed edge
        p2 : GridNode
            end vertex of directed edge

        Returns
        -------
        Waypoint
            the waypoint (agent position and direction), identifying the cell occupied and direction in which the agent is moving (specifying the next neighbor cell).
        """
        return Waypoint(position=(p1[0], p1[1]), direction=p2[2])
