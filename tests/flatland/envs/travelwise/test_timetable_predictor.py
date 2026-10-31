from fractions import Fraction
from typing import Dict, List, Optional, Tuple

import pytest

from flatland.envs.graph.graph_routing import greedy_graph_route_through_waypoints
from flatland.envs.graph.rail_graph_transition_map import GraphTransitionMap
from flatland.envs.graph_rail_env import GraphRailEnv
from flatland.envs.rail_env_policies import RailEnvStaticData, ShortestPathObservationBuilder, ShortestPathPolicy
from flatland.envs.step_utils.states import TrainState
from flatland.envs.timetable_utils import Timetable
from flatland.envs.travelwise.timetable_predictor import ProjectedEntryPoint, UnilateralTimetablePredictor
from tests.flatland.envs.travelwise.test_passenger_env_spike import _make_three_station_rail

EAST = 1
# the eastbound entry point of each corridor cell, by column
EASTBOUND = [GraphTransitionMap.grid_entry_point_to_graph_entry_point(0, column, EAST) for column in range(9)]
A, B, C = EASTBOUND[1], EASTBOUND[4], EASTBOUND[7]


def _through_train_env(max_speed: Fraction, earliest_departures: List[Optional[int]]) -> GraphRailEnv:
    """ A single through train A -> B -> C on the corridor, stopping at B. """
    g = GraphTransitionMap.grid_to_digraph(_make_three_station_rail())
    timetable = Timetable(earliest_departures=[earliest_departures], latest_arrivals=[[None, 30, 60]], max_episode_steps=60)
    return GraphRailEnv.from_graph(g, GraphTransitionMap.grid_to_resource_map(g), {0: [[A], [B], [C]]}, {0: float(max_speed)},
                                   observation_builder=ShortestPathObservationBuilder(), timetable_generator=lambda *args, **kwargs: timetable)


def _run(env: GraphRailEnv) -> Tuple[Dict[int, List[ProjectedEntryPoint]], Dict[int, Optional[str]], List[Tuple[str, int]]]:
    """
    Runs the train with `ShortestPathPolicy`, serving the stop at B. Returns the prediction made at each step, the
    train's entry point at each step and the entry points it actually entered, each with the step it entered it.
    """
    observations, infos = env.reset()
    static = RailEnvStaticData.from_env(env)
    predictor = UnilateralTimetablePredictor()
    predictor.reset(env)
    policy = ShortestPathPolicy(router=greedy_graph_route_through_waypoints, serve_intermediate_stops=True)
    agent = env.agents[0]
    predictions = {0: predictor.get(0)[0]}
    entry_points = {0: None}
    entered = []
    while not env.dones['__all__']:
        observations, _, _, infos = env.step(policy.act_many([0], [observations[0]], infos=dict(infos, static=static)))
        step = env._elapsed_steps
        if agent.state == TrainState.DONE:
            entered.append((agent.target_entry_point, step))
        elif agent.current_entry_point is not None and (len(entered) == 0 or entered[-1][0] != agent.current_entry_point):
            entered.append((agent.current_entry_point, step))
        predictions[step] = predictor.get(0)[0]
        entry_points[step] = agent.current_entry_point
    return predictions, entry_points, entered


@pytest.mark.parametrize("max_speed", [Fraction(1), Fraction(1, 2), Fraction(1, 3)])
@pytest.mark.parametrize("earliest_departure_a", [1, 3])
def test_timetable_predictor_is_exact_for_a_single_vehicle(max_speed: Fraction, earliest_departure_a: int):
    """
    A train alone on the map (no conflicts, no malfunctions, default acceleration) drives A -> B -> C, waiting at B
    until its earliest departure 12: the prediction made at any step equals the steps at which the train actually
    enters each later entry point, including the wait at B and the arrival at C. With `earliest_departure_a=3`, the
    first predictions are made before the train is on the map; at `max_speed=1/3`, it then reaches B exactly at its
    earliest departure, and still halts there for one step.
    """
    predictions, entry_points, entered = _run(_through_train_env(max_speed, [earliest_departure_a, 12, None]))

    for step, prediction in predictions.items():
        expected = [(entry_point, entered_step) for entry_point, entered_step in entered if entered_step > step]
        if entry_points[step] is not None:
            expected = [(entry_points[step], step)] + expected
        assert [(p.entry_point, p.step) for p in prediction] == expected, f"prediction made at step {step}"


def test_timetable_predictor_marks_stops_and_drops_reached_ones():
    """
    The predictions mark the waypoint groups the train stops at: A (0) and B (1) while it is not on the map; B (1),
    while it stands at B, including the wait there until its earliest departure 12; only C (2) once it has left B.
    Empty once the train has arrived.
    """
    predictions, _, _ = _run(_through_train_env(Fraction(1), [1, 12, None]))

    assert predictions[0] == [ProjectedEntryPoint(A, 1, 0), ProjectedEntryPoint(EASTBOUND[2], 2), ProjectedEntryPoint(EASTBOUND[3], 3),
                              ProjectedEntryPoint(B, 4, 1), ProjectedEntryPoint(EASTBOUND[5], 13), ProjectedEntryPoint(EASTBOUND[6], 14),
                              ProjectedEntryPoint(C, 15, 2)]
    # standing at B before its departure: moves again at 12, enters the next cell at 13
    assert predictions[7] == [ProjectedEntryPoint(B, 7, 1), ProjectedEntryPoint(EASTBOUND[5], 13), ProjectedEntryPoint(EASTBOUND[6], 14),
                              ProjectedEntryPoint(C, 15, 2)]
    assert predictions[13] == [ProjectedEntryPoint(EASTBOUND[5], 13), ProjectedEntryPoint(EASTBOUND[6], 14), ProjectedEntryPoint(C, 15, 2)]
    assert predictions[15] == []
