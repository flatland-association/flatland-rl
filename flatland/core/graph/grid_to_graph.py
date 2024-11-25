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
TODO split file: env creation, graph derivation, simplification, rendering, notebook
TODO action into cell behaviour - add to documentation
TODO docs with illustration of the mapping and "pins".
TODO illustration simplification and edge cases (pun intended): non-facing switches and diamond-crossings, and multi-edges.
TODO tests for conversion
"""
from collections import defaultdict
from typing import List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from attr import attrs, attrib

from flatland.core.grid.grid4_utils import get_new_position
from flatland.core.transition_map import GridTransitionMap
from flatland.envs.rail_env_utils import env_creator
from flatland.envs.rail_env import RailEnv
from flatland.utils.rendertools import RenderTool

GridNode = Tuple[int, int, int]


# TODO naming?
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
                self.cell_out_pins[succ_cell].add(n)

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


# TODO separation of concerns: graph building and rendering...
delta = 0.2
offsets = {
    # N
    0: [0.5, -delta],
    # E
    1: [-delta, -0.5],
    # S
    2: [-0.5, delta],
    # W
    3: [delta, 0.5]
}


def _add_flatland_styling(env: RailEnv, ax):
    env_renderer = RenderTool(env)
    img = env_renderer.render_env(show=False, frames=True, show_observations=False, show_predictions=False, return_image=True)
    ax.set_ylim(env.height - 0.5, -0.5)
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    ax.set_xticks(np.arange(0, env.width, 1))
    ax.set_yticks(np.arange(0, env.height, 1))
    # TODO image does not fill extent entirely - why?
    ax.imshow(np.fliplr(np.rot90(np.rot90(img))), extent=[-0.5, env.width - 0.5, -0.5, env.height - 0.5])
    ax.set_xticks(np.arange(-0.5, env.width + 0.5, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, env.height + 0.5, 1), minor=True)
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    ax.grid(which="minor")


# TODO move to notebook
if __name__ == '__main__':
    env = env_creator()

    micro = GraphTransitionMap.grid_to_digraph(env.rail)
    gtm = GraphTransitionMap(micro)
    decision_point_graph = DecisionPointGraph.fromGraphTransitionMap(gtm)
    collapsed = decision_point_graph.g
    fig, axs = plt.subplots(1, 2)

    micro1 = nx.subgraph_view(micro, filter_edge=lambda u, v: len(list(micro.successors(v))) == 1)
    nx.draw_networkx(micro1,
                     pos={(r, c, d): (c + offsets[d][1], r + offsets[d][0]) for (r, c, d) in micro1},
                     ax=axs[0],
                     node_size=2,
                     with_labels=False,
                     arrows=False
                     )
    micro2 = nx.subgraph_view(micro, filter_node=lambda v: len(list(micro.successors(v))) == 2)
    nx.draw_networkx(micro2,
                     pos={(r, c, d): (c + offsets[d][1], r + offsets[d][0]) for (r, c, d) in micro2},
                     ax=axs[0],
                     node_size=8,
                     node_color="red",
                     with_labels=False,
                     )
    micro3 = nx.subgraph_view(micro, filter_edge=lambda u, v: len(list(micro.successors(v))) == 2)
    nx.draw_networkx(micro3,
                     pos={(r, c, d): (c + offsets[d][1], r + offsets[d][0]) for (r, c, d) in micro3},
                     ax=axs[0],
                     arrows=True,
                     node_size=1,
                     with_labels=False
                     )

    nx.draw_networkx(collapsed,
                     pos={(r, c, d): (c + offsets[d][1], r + offsets[d][0]) for (r, c, d) in collapsed},
                     ax=axs[1],
                     node_size=2,
                     with_labels=False
                     )
    _add_flatland_styling(env, axs[1])
    _add_flatland_styling(env, axs[0])

    axs[0].set_title('micro')
    axs[1].set_title('collapsed')
    plt.show()
