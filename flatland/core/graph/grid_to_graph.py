"""
TODO description
TODO split file
TODO documentation/notebook
TODO reference to _create_simplified_graph https://github.com/aiAdrian/flatland_railway_extension/blob/e2b15bdd851ad32fb26c1a53f04621a3ca38fc00/flatland_railway_extension/FlatlandGraphBuilder.py
"""
from collections import defaultdict
from typing import List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from attr import attrs, attrib

from core.grid.grid4_utils import get_new_position
from core.transition_map import GridTransitionMap
from envs.line_generators import sparse_line_generator
from envs.malfunction_generators import NoMalfunctionGen
from envs.observations import TreeObsForRailEnv
from envs.predictions import ShortestPathPredictorForRailEnv
from envs.rail_env import RailEnv
from envs.rail_generators import sparse_rail_generator
from utils.rendertools import RenderTool


# TODO move method
# defaults from Flatland 3 Round 2 Test_0, see https://flatland.aicrowd.com/challenges/flatland3/envconfig.html
def env_creator(n_agents=7,
                x_dim=30,
                y_dim=30,
                n_cities=2,
                max_rail_pairs_in_city=4,
                grid_mode=False,
                max_rails_between_cities=2,
                malfunction_duration_min=20,
                malfunction_duration_max=50,
                malfunction_interval=540,
                speed_ratios=None,
                seed=42,
                obs_builder_object=None) -> RailEnv:
    if speed_ratios is None:
        speed_ratios = {1.0: 0.25, 0.5: 0.25, 0.33: 0.25, 0.25: 0.25}
    if obs_builder_object is None:
        obs_builder_object = TreeObsForRailEnv(max_depth=3, predictor=ShortestPathPredictorForRailEnv(max_depth=50))

    env = RailEnv(
        width=x_dim,
        height=y_dim,
        rail_generator=sparse_rail_generator(
            max_num_cities=n_cities,
            seed=seed,
            grid_mode=grid_mode,
            max_rails_between_cities=max_rails_between_cities,
            max_rail_pairs_in_city=max_rail_pairs_in_city
        ),
        malfunction_generator=NoMalfunctionGen(),
        # TODO ignore malfunctions for now
        # ParamMalfunctionGen(MalfunctionParameters(
        #     min_duration=malfunction_duration_min, max_duration=malfunction_duration_max, malfunction_rate=1.0 / malfunction_interval)),
        #
        line_generator=sparse_line_generator(speed_ratio_map=speed_ratios, seed=seed),
        number_of_agents=n_agents,
        obs_builder_object=obs_builder_object,
        record_steps=True,
        random_seed=seed
    )
    # TODO not deterministic grrrrr!
    env.reset(random_seed=seed)
    return env


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
    path = attrib(type=List[Tuple[int, int, int]])


# TODO naming: decision point graph, simplified graph, collapsed graph, ....?
class DecisionPointGraph:
    def __init__(self, g: nx.MultiDiGraph):
        self.g = g

    @staticmethod
    def explore_branch(g: nx.DiGraph, u: Tuple[int, int, int], v: Tuple[int, int, int]) -> List[Tuple[int, int, int]]:
        branch = [u, v]
        successors = list(g.successors(v))
        assert len(successors) > 0
        while len(successors) == 1:
            succ = successors[0]
            branch.append(succ)
            successors = list(g.successors(succ))
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
                branch = DecisionPointGraph.explore_branch(micro, dp, n)
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
