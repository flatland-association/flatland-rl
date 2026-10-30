import tempfile
from functools import partial
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pytest

from flatland.callbacks.generate_movie_callbacks import GenerateMovieCallbacks
from flatland.env_generation.env_generator import env_generator_legacy
from flatland.envs.graph.graph_routing import greedy_graph_route_through_waypoints
from flatland.envs.graph.rail_graph_transition_map import GraphTransitionMap
from flatland.envs.graph_rail_env import GraphRailEnv
from flatland.envs.grid.grid_routing import greedy_grid_route_through_waypoints
from flatland.envs.grid.rail_env_grid import RailEnvTransitions
from flatland.envs.observations import FullEnvObservation
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_env_policies import ShortestPathPolicy
from flatland.envs.rail_generators import rail_from_grid_transition_map
from flatland.envs.rail_grid_transition_map import RailGridTransitionMap
from flatland.envs.rail_trainrun_data_structures import Waypoint
from flatland.envs.timetable_utils import Line, Timetable
from flatland.trajectories.policy_runner import PolicyRunner


def test_shortest_path_policy_no_intermediate_target():
    with tempfile.TemporaryDirectory() as tmpdirname:
        data_dir = Path(tmpdirname)
        trajectory = PolicyRunner.create_from_policy(
            policy=ShortestPathPolicy(),
            data_dir=data_dir,
            snapshot_interval=5,
            env=env_generator_legacy(obs_builder_object=FullEnvObservation(), seed=42, )[0],
        )
        assert np.isclose(trajectory.trains_arrived_lookup()["success_rate"], 1 / 7)


def test_shortest_path_policy_with_intermediate_targets(gen_movies=False):
    with tempfile.TemporaryDirectory() as tmpdirname:
        data_dir = Path(tmpdirname)
        trajectory = PolicyRunner.create_from_policy(
            policy=ShortestPathPolicy(),
            data_dir=data_dir,
            snapshot_interval=5,
            env=env_generator_legacy(n_cities=5, line_length=3, obs_builder_object=FullEnvObservation(), seed=42, )[0],
            callbacks=GenerateMovieCallbacks() if gen_movies else None,
        )
        assert np.isclose(trajectory.trains_arrived_lookup()["success_rate"], 2 / 7)


# A straight corridor of 9 cells with dead ends, stations A, B, C on the eastbound track.
CORRIDOR_LENGTH = 9
A, B, C = ((0, 1), 1), ((0, 4), 1), ((0, 7), 1)


def _corridor_rail() -> RailGridTransitionMap:
    transitions = RailEnvTransitions()
    cells = transitions.transition_list
    straight = transitions.rotate_transition(cells[1], 90)
    opens_east = transitions.rotate_transition(cells[7], 270)
    opens_west = transitions.rotate_transition(cells[7], 90)
    rail = RailGridTransitionMap(width=CORRIDOR_LENGTH, height=1, transitions=transitions)
    rail.grid = np.array([[opens_east] + [straight] * (CORRIDOR_LENGTH - 2) + [opens_west]], dtype=np.uint16)
    return rail


def _corridor_env(graph: bool, waypoints, earliest_departures, latest_arrivals, speed: float):
    """ One agent on the corridor, as `RailEnv` or as `GraphRailEnv` built from the same grid. """

    def timetable_generator(*args, **kwargs) -> Timetable:
        return Timetable(earliest_departures=[earliest_departures], latest_arrivals=[latest_arrivals], max_episode_steps=60)

    if graph:
        g = GraphTransitionMap.grid_to_digraph(_corridor_rail())
        to_graph = GraphTransitionMap.grid_entry_point_to_graph_entry_point
        graph_waypoints = [[to_graph(*position, direction) for position, direction in group] for group in waypoints]
        return GraphRailEnv.from_graph(g, GraphTransitionMap.grid_to_resource_map(g), {0: graph_waypoints}, {0: speed},
                                       observation_builder=FullEnvObservation(),
                                       timetable_generator=timetable_generator)

    def line_generator(*args, **kwargs) -> Line:
        return Line(agent_waypoints={0: [[Waypoint(position, direction) for position, direction in group] for group in waypoints]},
                    agent_speeds=[speed])

    env = RailEnv(width=CORRIDOR_LENGTH, height=1, rail_generator=rail_from_grid_transition_map(_corridor_rail()), line_generator=line_generator,
                  timetable_generator=timetable_generator, number_of_agents=1, obs_builder_object=FullEnvObservation())
    env.reset()
    return env


def _run(env, policy: ShortestPathPolicy) -> Tuple[Dict[int, Optional[Tuple[Tuple[int, int], int]]], float]:
    """ Runs the policy until done, returns the agent's entry point per step (as grid entry point) and the total reward. """
    obs, _ = env.reset()
    trace = {}
    total_reward = 0
    done = False
    while not done:
        actions = policy.act_many(env.get_agent_handles(), observations=list(obs.values()))
        obs, rewards, dones, _ = env.step(actions)
        entry_point = env.agents[0].current_entry_point
        if isinstance(entry_point, str):
            entry_point = GraphTransitionMap.graph_entry_point_to_grid_entry_point(entry_point)
        trace[env._elapsed_steps] = entry_point
        total_reward += rewards[0]
        done = dones["__all__"]
    return trace, total_reward


def _policy(graph: bool, serve_intermediate_stops: bool) -> ShortestPathPolicy:
    if graph:
        return ShortestPathPolicy(router=greedy_graph_route_through_waypoints, serve_intermediate_stops=serve_intermediate_stops)
    return ShortestPathPolicy(serve_intermediate_stops=serve_intermediate_stops)


@pytest.mark.parametrize("graph", [False, True])
@pytest.mark.parametrize("speed, earliest_departure_b, expected_trace", [
    (1.0, 10, {1: None, 2: A, 3: ((0, 2), 1), 4: ((0, 3), 1), 5: B, 6: B, 7: B, 8: B, 9: B, 10: B, 11: ((0, 5), 1), 12: ((0, 6), 1),
               13: None}),
    (0.5, 14, {1: None, 2: A, 3: A, 4: ((0, 2), 1), 5: ((0, 2), 1), 6: ((0, 3), 1), 7: ((0, 3), 1), 8: B, 9: B, 10: B, 11: B, 12: B, 13: B,
               14: B, 15: B, 16: ((0, 5), 1), 17: ((0, 5), 1), 18: ((0, 6), 1), 19: ((0, 6), 1), 20: None}),
])
def test_shortest_path_policy_waits_at_intermediate_stop_until_earliest_departure(graph, speed, earliest_departure_b, expected_trace):
    """
    A -> B -> C, stopping at B until its earliest departure there, on grid and graph: the agent enters B and stops,
    and moves again in the step of its earliest departure at B. The default rewards count B as served and on
    time: total reward 0.
    """
    env = _corridor_env(graph, [[A], [B], [C]], earliest_departures=[2, earliest_departure_b, None],
                        latest_arrivals=[None, earliest_departure_b, 40], speed=speed)
    trace, total_reward = _run(env, _policy(graph, serve_intermediate_stops=True))
    assert trace == expected_trace
    assert total_reward == 0


@pytest.mark.parametrize("graph", [False, True])
def test_shortest_path_policy_stops_once_at_intermediate_stop_when_late(graph):
    """ Arriving at B after its earliest departure there, the agent still stops at B, then moves on at once. """
    env = _corridor_env(graph, [[A], [B], [C]], earliest_departures=[2, 3, None], latest_arrivals=[None, 5, 40], speed=1.0)
    trace, total_reward = _run(env, _policy(graph, serve_intermediate_stops=True))
    assert trace == {1: None, 2: A, 3: ((0, 2), 1), 4: ((0, 3), 1), 5: B, 6: B, 7: ((0, 5), 1), 8: ((0, 6), 1), 9: None}
    assert total_reward == 0


@pytest.mark.parametrize("graph", [False, True])
def test_shortest_path_policy_ignores_intermediate_stop_by_default(graph):
    """
    Without `serve_intermediate_stops` (the default, the previous behaviour), the agent routes straight to C and
    drives through B: the default rewards count B as not served (intermediate-not-served penalty 1).
    """
    env = _corridor_env(graph, [[A], [B], [C]], earliest_departures=[2, 10, None], latest_arrivals=[None, 10, 40], speed=1.0)
    trace, total_reward = _run(env, _policy(graph, serve_intermediate_stops=False))
    assert trace == {1: None, 2: A, 3: ((0, 2), 1), 4: ((0, 3), 1), 5: B, 6: ((0, 5), 1), 7: ((0, 6), 1), 8: None}
    assert total_reward == -1


@pytest.mark.parametrize("router, expected_stop", [
    # default router with serve_intermediate_stops: the nearest alternative
    (None, B),
    # first reachable alternative, the rule without serve_intermediate_stops
    (partial(greedy_grid_route_through_waypoints, nearest_alternative=False), ((0, 6), 1)),
])
def test_shortest_path_policy_default_router_picks_nearest_alternative_when_serving_intermediate_stops(router, expected_stop):
    """ An intermediate stop with alternatives `(0, 6)` (listed first) and B (nearer): the agent halts at the alternative picked. """
    env = _corridor_env(False, [[A], [((0, 6), 1), B], [C]], earliest_departures=[2, 10, None], latest_arrivals=[None, 10, 40], speed=1.0)
    trace, _ = _run(env, ShortestPathPolicy(router=router, serve_intermediate_stops=True))
    assert [entry_point for step, entry_point in trace.items() if 7 <= step <= 9] == [expected_stop] * 3
