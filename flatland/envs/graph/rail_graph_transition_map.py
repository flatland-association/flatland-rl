"""
A Flatland (microscopic) topology can be represented by different kinds of graphs.
The topology must reflect the possible paths through the rail network - it must not be possible to traverse a switch in the acute angle.
With the help of the graph it is very easy to calculate the shortest connection from node A to node B. The API makes it possible to solve such tasks very efficiently. Moreover, the graph can be simplified so that only decision-relevant nodes remain in the graph and all other nodes are merged. A decision node is a node or flatland cell (track) that reasonably allows the agent to stop, go, or branch off. For straight track edges within a route, it makes little sense to wait in many situations. This is because the agent would block many resources, i.e., if an agent does not drive to the decision point: a cell before a crossing, the agent blocks the area in between. This makes little sense from an optimization point of view.

Two (dual, equivalent) approaches are possible:
- agents are positioned on the nodes
- agents are positioned on the edges.
The second approach makes it easier to visualize agents moving forward on edges. Hence, we choose the second approach.

Our directed graph consists of nodes and edges:
* A node in the graph is defined by position and direction. The position corresponds to the position of the underlying cell in the original flatland topology, and the direction corresponds to the direction in which an agent reaches the cell. Thus, the node is defined by (r, c, d), where c (column) is the index of the horizontal cell grid position, r (row) is the index of the vertical cell grid position, and d (direction) is the direction of cell entry. In the Flatland (2d grid), not every of the eight neighbor cells can be reached from every direction. Therefore, the entry direction information is key.
* An edge is defined by "from-node" u and "to-node" v such that for the edge e = (u, v).  Edges reflect feasible transition from node u to node v. We can think of the suggestive notation $[u,v)$ in terms of resource occupation of the underlying cell, as the "from-node" refers to the underlying cell entered, and the "to-node" refers to the neighbor cell entered when the edge is left.

The implementation uses networkX, so there are also many graph functions available.

References:
- Egli, Adrian. FlatlandGraphBuilder. https://github.com/aiAdrian/flatland_railway_extension/blob/e2b15bdd851ad32fb26c1a53f04621a3ca38fc00/flatland_railway_extension/FlatlandGraphBuilder.py
- Nygren, E., Eichenberger, Ch., Frejinger, E. Scope Restriction for Scalable Real-Time Railway Rescheduling: An Exploratory Study. https://arxiv.org/abs/2305.03574
"""
import ast
from collections import defaultdict
from functools import lru_cache
from typing import Tuple, Optional, Set

import networkx as nx

from flatland.core.grid.grid4_utils import get_new_position
from flatland.core.transition_map import GridTransitionMap, TransitionMap
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_env_action import RailEnvActions, RailEnvNextAction

GridNode = Tuple[Tuple[int, int], int]
GridEdge = Tuple[GridNode, GridNode]


class GraphTransitionMap(TransitionMap[GridNode, GridEdge, bool, RailEnvActions]):
    """
    Flatland 3 Transition map represented by a directed graph.

    The grid transition map contains for all cells a set of pairs (heading at cell entry, heading at cell exit).
      E.g. horizontal straight is {(E,E), (W,W)}.
    The directed graph's nodes are entry pins (cell + plus heading at entry).
    Edges always go from entry pin at one cell to entry pin of a neighboring cell.
    The outgoing heading for the grid transition map is the incoming heading at a neighboring cell.

    Incoming heading:

                   S
                   ⌄
                   |
           E   >---+---< W
                   |
                   ^
                   N

    Outgoing heading (=incoming at neighbor cell):
                   N (of cell-to-the-north)
                   ^
                   |
           E   <---+---> E (of cell-to-the-east)
    (of cell-to-   |
     the-east)     ⌄
                   S (of cell-to-the-south)


    Attributes
    ----------
    g: nx.DiGraph
    """

    def __init__(self, g: nx.DiGraph):
        self.g = g
        self.cell_in_pins = defaultdict(lambda: set())

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
                    covered_actions = set()
                    possible_transitions = transition_map.get_transitions(((r, c), d))
                    for new_direction in range(4):
                        if possible_transitions[new_direction]:
                            new_position = get_new_position((r, c), new_direction)
                            if (new_direction - d) % 4 == 0:
                                action = "F"
                            elif (new_direction - d) % 4 == 1:
                                action = "R"
                            elif (new_direction - d) % 4 == (-1 % 4):
                                action = "L"
                            elif (new_direction - d) % 4 == 2:
                                # dead-end
                                action = "F"
                            else:
                                raise
                            r2, c2 = new_position
                            d2 = new_direction
                            g.add_edge(
                                GraphTransitionMap.grid_configuration_to_graph_configuration(r, c, d),
                                GraphTransitionMap.grid_configuration_to_graph_configuration(r2, c2, d2), action=action)
                            covered_actions.add(action)
                    u = GraphTransitionMap.grid_configuration_to_graph_configuration(r, c, d)
                    if u not in g.nodes:
                        continue
                    for a in RailEnvActions:
                        new_cell_valid, ((r2, c2), d2), transition_valid, preprocessed_action, action_valid = transition_map.check_action_on_agent(a,
                                                                                                                                                   ((r, c), d))

                        v = GraphTransitionMap.grid_configuration_to_graph_configuration(r2, c2, d2)
                        if new_cell_valid:
                            if ((r2, c2), d2) != ((r, c), d):
                                g[u][v].setdefault("actions", []).append(action)
                        else:
                            g.nodes[u].setdefault("prohibited_actions", []).append(action)
                    # if len(covered_actions) == 2:
                    #     assert  "symmetric" in RailEnvTransitionsEnum(transition_map.get_full_transitions(r, c)).name
                    for a in RailEnvActions:
                        new_cell_valid, ((r2, c2), d2), transition_valid, preprocessed_action, _ = transition_map.check_action_on_agent(a, ((r, c), d))
                        v = GraphTransitionMap.grid_configuration_to_graph_configuration(r2, c2, d2)
                        if (u, v) in g.edges:
                            g[u][v].setdefault("_grid_check_action_on_agent", []).append(
                                (new_cell_valid, ((r2, c2), d2), transition_valid, preprocessed_action, action_valid))

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

    @lru_cache(maxsize=1_000_000)
    def apply_action_independent(self, action: RailEnvActions, configuration: Tuple[Tuple[int, int], int]) -> Optional[Tuple[GridNode, bool]]:
        new_cell_valid, new_configuration, transition_valid, preprocessed_action, action_valid = self.check_action_on_agent(action, configuration)
        if action_valid:
            return new_configuration, transition_valid
        else:
            return None

    def check_action_on_agent(self, action: RailEnvActions, configuration: GridNode) -> Tuple[bool, GridNode, bool, RailEnvActions]:
        succs = list(self.g.successors(configuration))
        if False:
            if action in self.g.nodes[configuration].get("prohibited_actions", set()):
                # TODO does not work - currently the new position derived from grid is returned -> first refactor core without returning a new invalid position (return e.g. None)
                return True, configuration, True, action

            for v in succs:
                if self.g.get_edge_data(configuration, v)["action"] == action:
                    # TODO does not work - currently preprocessed action is returned which can differ from raw action in some cases -> first refactor core without preprocessing
                    return True, v, True, action

        assert 1 <= len(succs) <= 2

        graph_action = None
        new_configuration = None
        if len(succs) == 1:
            succ = list(succs)[0]
            new_configuration = succ
            graph_action = self.g.get_edge_data(configuration, succ)["action"]
        else:
            if action == RailEnvActions.MOVE_LEFT:
                for v in succs:
                    if self.g.get_edge_data(configuration, v)["action"] == "L":
                        new_configuration = v
                        graph_action = "L"
                        break

            elif action == RailEnvActions.MOVE_RIGHT:
                for v in succs:
                    if self.g.get_edge_data(configuration, v)["action"] == "R":
                        new_configuration = v
                        graph_action = "R"
                        break
            if new_configuration is None:
                for v in succs:
                    if self.g.get_edge_data(configuration, v)["action"] == "F":
                        new_configuration = v
                        graph_action = "F"
                        break
        if graph_action is None:
            # symmetric switches
            return False, new_configuration, False, RailEnvActions.STOP_MOVING, False
        transition_valid = True
        preprocessed_action = action
        if action == RailEnvActions.MOVE_LEFT and graph_action != "L":
            transition_valid = False
            preprocessed_action = RailEnvActions.MOVE_FORWARD
        elif action == RailEnvActions.MOVE_RIGHT and graph_action != "R":
            transition_valid = False
            preprocessed_action = RailEnvActions.MOVE_FORWARD

        new_cell_valid = new_configuration in self.g.nodes
        return new_cell_valid, new_configuration, transition_valid, preprocessed_action, True

    @lru_cache
    def get_valid_move_actions(self, configuration: GridNode) -> Set[RailEnvNextAction]:
        return [RailEnvNextAction(a, s) for s in self.g.successors(configuration) for a in self.g.get_edge_data(configuration, s)["action"]]

    def get_transitions(self, configuration: GridNode) -> Tuple[bool]:
        return True,

    @staticmethod
    @lru_cache
    def grid_configuration_to_graph_configuration(r: int, c: int, d: int) -> str:
        return f"{int(r), int(c), int(d)}"

    @staticmethod
    @lru_cache
    def graph_configuration_to_grid_configuration(s: str) -> Optional[Tuple[Tuple[int, int], int]]:
        if s is None:
            return None
        r, c, d = ast.literal_eval(s)
        return ((r, c), d)

    @lru_cache
    def get_successor_configurations(self, configuration: GridNode) -> Set[GridNode]:
        return set(self.g.successors(configuration))

    @lru_cache
    def get_predecessor_configurations(self, configuration: GridNode) -> Set[GridNode]:
        return set(self.g.predecessors(configuration))

    @lru_cache
    def is_valid_configuration(self, configuration: GridNode) -> bool:
        return configuration in self.g.nodes
