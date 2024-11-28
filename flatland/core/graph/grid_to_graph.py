"""
A Flatland (microscopic) topology can be represented by different kinds of graphs.
The topology must reflect the possible paths through the rail network - it must not be possible to traverse a switch in the acute angle.
With the help of the graph it is very easy to calculate the shortest connection from node A to node B. The API makes it possible to solve such tasks very efficiently. Moreover, the graph can be simplified so that only decision-relevant nodes remain in the graph and all other nodes are merged. A decision node is a node or flatland cell (track) that reasonably allows the agent to stop, go, or branch off. For straight track edges within a route, it makes little sense to wait in many situations. This is because the agent would block many resources, i.e., if an agent does not drive to the decision point: a cell before a crossing, the agent blocks the area in between. This makes little sense from an optimization point of view.

Two (dual, equivalent) approaches are possible:
- agents are positioned on the nodes
- agents are positioned on the edges.
The second approach makes it easier to visualize agents moving forward on edges. Hence, we choose the second approach.

Our directed graph consists of nodes and edges:
* A node in the graph is defined by position and direction. The position corresponds to the position of the underlying cell in the original flatland topology, and the direction corresponds to the direction in which an agent reaches the cell. Thus, the node is defined by (r, c, d), where c (column) is the index of the horizontal cell grid position, r (row) is the index of the vertical cell grid position, and d (direction) is the direction of cell entry. In the Flatland (2d grid), not every of the eight neighbors cell can be reached from every direction. Therefore, the entry direction information is key.
* An edge is defined by "from-node" u and "to-node" v such that for the edge e = (u, v).  Edges reflect feasible transition from node u to node v exist.

The implementation uses networkX, so there are also many graph functions available.

References:
- Egli, Adrian. FlatlandGraphBuilder. https://github.com/aiAdrian/flatland_railway_extension/blob/e2b15bdd851ad32fb26c1a53f04621a3ca38fc00/flatland_railway_extension/FlatlandGraphBuilder.py
- Nygren, E., Eichenberger, Ch., Frejinger, E. Scope Restriction for Scalable Real-Time Railway Rescheduling: An Exploratory Study. https://arxiv.org/abs/2305.03574
TODO action into cell behaviour - add to documentation --> "flatland-book"
TODO docs with illustration of the mapping and "pins". --> "flatland-book"
TODO illustration simplification and edge cases (pun intended): non-facing switches and diamond-crossings, and multi-edges. --> notebook or "flatland-book"?
"""
from collections import defaultdict

import networkx as nx

from flatland.core.grid.grid4_utils import get_new_position
from flatland.core.transition_map import GridTransitionMap
from flatland.envs.rail_env import RailEnv


class GraphTransitionMap:
    """
    Flatland 3 Transition map represented by a directed graph.

    Attributes
    ----------
    g: nx.DiGraph
    """

    def __init__(self, g: nx.DiGraph):
        self.g = g

        self.cell_in_pins = defaultdict(lambda: set())
        self.cell_out_pins = defaultdict(lambda: set())
        for n in self.g:
            cell = n[:2]
            self.cell_in_pins[cell].add(n)
            for succ in g.successors(n):
                succ_cell = succ[:2]
                self.cell_out_pins[succ_cell].add(n)

    @staticmethod
    def grid_to_digraph(transition_map: GridTransitionMap) -> nx.DiGraph:
        """
        The graph representation of a grid transition map.

        Parameters
        ----------
        transition_map

        Returns
        -------

        """
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
    def from_rail_env(env: RailEnv) -> "GraphTransitionMap":
        """
        Factory method to create a graph transition map from a rail env.

        Parameters
        ----------
        env: RailEnv

        Returns
        -------
        GraphTransitionMap
            The graph transition map.
        """
        return GraphTransitionMap(GraphTransitionMap.grid_to_digraph(env.rail))
