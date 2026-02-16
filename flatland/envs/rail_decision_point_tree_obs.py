from collections import defaultdict
from typing import Dict, Set, List, Optional, Tuple, TypeVar, Callable

import numpy as np

from flatland.core.env_observation_builder import ObservationBuilder, AgentHandle
from flatland.envs.agent_utils import EnvAgent
from flatland.envs.graph.graph_simplification import DecisionPointGraph, DecisionPointGraphEdgeData
from flatland.envs.graph.rail_graph_transition_map import GraphTransitionMap, GridEdge, GridNode
from flatland.envs.observations import Node
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_trainrun_data_structures import Waypoint

TreeObsNodeType = TypeVar('TreeObsNodeType')


class DecisionPointTreeObs(ObservationBuilder[RailEnv, TreeObsNodeType]):

    def __init__(self, max_depth: int, branch_to_node: Callable[[Optional[List[Waypoint]]], TreeObsNodeType]):
        super().__init__()
        self.max_depth = max_depth
        self.dpg = None
        self.waypoint_edge_mapping: Optional[Dict[Waypoint, Set[Tuple[GridEdge, int]]]] = None
        self.curr_edges: Optional[Dict[AgentHandle, GridEdge]] = None
        self.curr_edge_remaining: Optional[Dict[AgentHandle, List[GridNode]]] = None
        self._collect_branch = branch_to_node

    def reset(self):
        gtm = GraphTransitionMap(GraphTransitionMap.grid_to_digraph(self.env.rail))
        self.dpg = DecisionPointGraph.fromGraphTransitionMap(gtm)
        self.waypoint_edge_mapping = defaultdict(set)
        for u, v, d in self.dpg.g.edges.data():
            data: DecisionPointGraphEdgeData = d["d"]
            for offset, (p1, p2) in enumerate(zip(data.path, data.path[1:])):
                wp = DecisionPointGraph.micro_edge_to_waypoint(p1, p2)
                self.waypoint_edge_mapping[wp].add(((u, v), offset))
        self.curr_edges = defaultdict(lambda: None)
        self.curr_edge_remaining = defaultdict(lambda: None)

    def get(self, handle: AgentHandle = 0) -> TreeObsNodeType:
        agent = self.env.agents[handle]
        self._update_agent_position(agent)
        curr_edge = self.curr_edges.get(handle, None)
        if curr_edge is not None:
            edge_remaining_vertices = self.curr_edge_remaining.get(handle, None)
            edge_remaining_cells = len(edge_remaining_vertices) - 1
            assert edge_remaining_cells >= 1
            return self._traverse_branch(curr_edge, path=edge_remaining_vertices, depth=0, max_depth=self.max_depth)
        return self._collect_branch(None)

    def _traverse_branch(self, edge: GridEdge, path: List[GridNode], depth: int, max_depth) -> Node:
        wps = [DecisionPointGraph.micro_edge_to_waypoint(p1, p2) for p1, p2 in zip(path, path[1:])]
        r = self._collect_branch(wps)
        if depth < max_depth:
            _, u = edge
            # N.B. multi-di-graph: successor can be the same but with different path (in the data attribute).
            outgoing_edges_with_data = list(self.dpg.g.edges(u, True))
            # TODO warning only: in the case of loops, there is only one outgoing branch!
            assert len(outgoing_edges_with_data) == 2
            for i, (_, v, edge_data) in enumerate(outgoing_edges_with_data):
                # TODO interface for .childs
                r.childs[i] = self._traverse_branch((u, v), edge_data["d"].path, depth + 1, max_depth)
        return r

    def _update_agent_position(self, agent: EnvAgent):
        handle = agent.handle
        if agent.position is None:
            self.curr_edges.pop(handle, None)
            self.curr_edge_remaining.pop(handle, None)
        else:
            waypoint = Waypoint(agent.position, agent.direction)
            if agent.position != agent.old_position:
                if agent.old_position is None:
                    # take any branch (there may be multiple after dead-ends and "mergers")
                    assert len(list(self.waypoint_edge_mapping[waypoint])) > 0, (waypoint, self.waypoint_edge_mapping)
                    edge, offset = list(self.waypoint_edge_mapping[waypoint])[0]
                    self.curr_edges[handle] = edge
                    self.curr_edge_remaining[handle] = self.dpg.g.get_edge_data(*edge)[0]["d"].path[offset:]
                else:
                    assert len(self.curr_edge_remaining) >= 2
                    if len(self.curr_edge_remaining) == 2:
                        # update unique next branch from waypoint (unique because merger results in separate micro-edges)
                        edges_offsets = list(self.waypoint_edge_mapping[waypoint])
                        assert len(edges_offsets) == 1, edges_offsets
                        edge, offset = edges_offsets[0]
                        assert offset == 0
                        self.curr_edges[handle] = edge
                        self.curr_edge_remaining[handle] = self.dpg.g.data(edge).path[offset:]
                    else:
                        # take unique next cell in current edge
                        self.curr_edge_remaining[handle] = self.curr_edge_remaining[handle][1:]
                        assert DecisionPointGraph.micro_edge_to_waypoint(*self.curr_edge_remaining[handle][:2]) == waypoint


def standard_branch_to_node(wps: Optional[List[Waypoint]]) -> Node:
    return Node(*[-np.inf] * 12, {})
