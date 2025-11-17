import json
from pathlib import Path

import numpy as np
from networkx.readwrite import json_graph

from flatland.env_generation.env_generator import env_generator
from flatland.envs.graph.graph_simplification import DecisionPointGraph
from flatland.envs.graph.rail_graph_transition_map import GraphTransitionMap
from flatland.envs.rail_trainrun_data_structures import Waypoint


# https://stackoverflow.com/questions/50916422/python-typeerror-object-of-type-int64-is-not-json-serializable
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, Waypoint):
            return (obj.position, obj.direction)
        return super(NpEncoder, self).default(obj)


if __name__ == '__main__':
    env, _, _ = env_generator(seed=66)

    micro = GraphTransitionMap.grid_to_digraph(env.rail)
    decision_point_graph = DecisionPointGraph.fromGraphTransitionMap(GraphTransitionMap(micro))
    collapsed = decision_point_graph.g
    d = {
        "micro_graph": json_graph.node_link_data(micro),
        "decision_point_graph": json_graph.node_link_data(micro),
        "timetable": {
            "waypoints": {agent.handle: agent.waypoints for agent in env.agents},
            "waypoints_earliest_departure": {agent.handle: agent.waypoints_earliest_departure for agent in env.agents},
            "waypoints_latest_arrival": {agent.handle: agent.waypoints_latest_arrival for agent in env.agents}
        },
        "max_speeds": {
            agent.handle: agent.speed_counter.max_speed for agent in env.agents
        }
    }
    print(json.dumps(d, indent=4, cls=NpEncoder))
    with Path("first.json").open("w") as f:
        f.write(json.dumps(d, indent=4, cls=NpEncoder))
