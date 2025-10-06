from collections import defaultdict

from flatland.core.env_observation_builder import ObservationBuilder, AgentHandle
from flatland.envs.graph.graph_simplification import DecisionPointGraph, DecisionPointGraphEdgeData
from flatland.envs.graph.rail_graph_transition_map import GraphTransitionMap
from flatland.envs.observations import Node
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_trainrun_data_structures import Waypoint


class DecisionPointTreeObs(ObservationBuilder[RailEnv, Node]):

    def __init__(self, depth: int):
        self.depth = depth
        self.dpg = None
        self.waypoint_edge_mapping = None
        self.curr_edges = None
        self.curr_edge_remaining = None

    def reset(self):
        gtm = GraphTransitionMap(GraphTransitionMap.grid_to_digraph(self.env.rail))
        self.dpg = DecisionPointGraph.fromGraphTransitionMap(gtm)
        self.waypoint_edge_mapping = defaultdict(set)
        for u, v, d in self.dpg.g.edges.data():
            print(u, v, d)
            data: DecisionPointGraphEdgeData = d["d"]
            for offset, (p1, p2) in enumerate(zip(data.path, data.path[1:])):
                wp = DecisionPointGraph.micro_edge_to_waypoint(p1, p2)
                self.waypoint_edge_mapping[wp].add(((u, v), offset))
        self.curr_edges = defaultdict(lambda: None)
        self.curr_edge_remaining = defaultdict(lambda: None)
        print(self.waypoint_edge_mapping)

    def get(self, handle: AgentHandle = 0) -> Node:
        agent = self.env.agents[handle]
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
                    self.curr_edges[handle] = handle
                    self.curr_edge_remaining[handle] = self.dpg.g.data(edge).path[offset:]
                else:
                    assert len(self.curr_edge_remaining) >= 2
                    if len(self.curr_edge_remaining) == 2:
                        # update unique next branch from dcell (unique because merger results in separate micro-edges)
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

        return None
