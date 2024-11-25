import matplotlib.pyplot as plt
import networkx as nx

from flatland.core.graph.graph_rendering import get_positions, add_flatland_styling
from flatland.core.graph.graph_simplification import DecisionPointGraph
from flatland.core.graph.grid_to_graph import GraphTransitionMap
from flatland.envs.rail_env_utils import env_creator

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
                     pos=get_positions(micro1),
                     ax=axs[0],
                     node_size=2,
                     with_labels=False,
                     arrows=False
                     )
    micro2 = nx.subgraph_view(micro, filter_node=lambda v: len(list(micro.successors(v))) == 2)
    nx.draw_networkx(micro2,
                     pos=get_positions(micro2),
                     ax=axs[0],
                     node_size=8,
                     node_color="red",
                     with_labels=False,
                     )
    micro3 = nx.subgraph_view(micro, filter_edge=lambda u, v: len(list(micro.successors(v))) == 2)
    nx.draw_networkx(micro3,
                     pos=get_positions(micro3),
                     ax=axs[0],
                     arrows=True,
                     node_size=1,
                     with_labels=False
                     )

    nx.draw_networkx(collapsed,
                     pos=get_positions(collapsed),
                     ax=axs[1],
                     node_size=2,
                     with_labels=False
                     )
    add_flatland_styling(env, axs[1])
    add_flatland_styling(env, axs[0])

    axs[0].set_title('micro')
    axs[1].set_title('collapsed')
    # plt.show()
