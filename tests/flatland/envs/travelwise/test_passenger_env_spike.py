from typing import Dict, List, Optional, Tuple

import numpy as np

from flatland.core.grid.grid4 import Grid4TransitionsEnum
from flatland.envs.graph.graph_routing import greedy_graph_route_through_waypoints
from flatland.envs.graph.rail_graph_transition_map import GraphTransitionMap
from flatland.envs.graph_rail_env import GraphRailEnv
from flatland.envs.grid.rail_env_grid import RailEnvTransitions
from flatland.envs.rail_env_policies import ShortestPathPolicy
from flatland.envs.rail_grid_transition_map import RailGridTransitionMap
from flatland.envs.step_utils.states import TrainState
from flatland.envs.timetable_utils import Timetable
from flatland.envs.travelwise.passenger_env import PassengerEnv, PassengerJourney
from flatland.envs.travelwise.passenger_env_observations import PassengerObservationBuilder
from flatland.envs.travelwise.passenger_env_policies import BoardForTargetPassengerPolicy
from flatland.envs.travelwise.vehicle_observations import VehicleObservationBuilder

# A single straight corridor with three stations A, B, C, evenly spaced 3 cells apart, capped by dead
# ends well beyond either station so no agent ever reaches an end of the corridor.
STATION_A = (0, 1)
STATION_B = (0, 4)
STATION_C = (0, 7)
STATION_DISTANCE = 3
N_CELLS = 9

# One departure every HEADWAY steps, round-robin over the four legs below (leg 1 is the deliberate
# exception - see LEG_EARLIEST_DEPARTURE[1] below) - regular enough that no two agents ever contend for
# a station at the same time, given each leg only takes STATION_DISTANCE steps.
HEADWAY = 6


def _modes(modes: Dict[int, Tuple[Tuple[int, int], Optional[int]]]) -> Dict[int, Tuple[str, Optional[int]]]:
    """ Passenger modes per step, given with grid cells, as resources. """
    return {step: (GraphTransitionMap.grid_entry_point_to_graph_resource(*position), riding_handle) for step, (position, riding_handle) in modes.items()}


EAST, WEST = Grid4TransitionsEnum.EAST, Grid4TransitionsEnum.WEST

# Four one-way legs: two shuttling A<->B, two shuttling B<->C.
LEG_WAYPOINTS: Dict[int, List[List[str]]] = {
    0: [[GraphTransitionMap.grid_entry_point_to_graph_entry_point(*STATION_A, EAST)],  # A -> B
        [GraphTransitionMap.grid_entry_point_to_graph_entry_point(*STATION_B, EAST)]],
    1: [[GraphTransitionMap.grid_entry_point_to_graph_entry_point(*STATION_B, EAST)],  # B -> C
        [GraphTransitionMap.grid_entry_point_to_graph_entry_point(*STATION_C, EAST)]],
    2: [[GraphTransitionMap.grid_entry_point_to_graph_entry_point(*STATION_B, WEST)],  # B -> A
        [GraphTransitionMap.grid_entry_point_to_graph_entry_point(*STATION_A, WEST)]],
    3: [[GraphTransitionMap.grid_entry_point_to_graph_entry_point(*STATION_C, WEST)],  # C -> B
        [GraphTransitionMap.grid_entry_point_to_graph_entry_point(*STATION_B, WEST)]],
}
LEG_EARLIEST_DEPARTURE: Dict[int, int] = {handle: (handle + 1) * HEADWAY for handle in LEG_WAYPOINTS}
# Leg 1 (B->C) is deliberately postponed well past its own regular headway slot (12) - still clear of
# leg 2's own use of B (step 18, departing the other way) - so a passenger at B heading to C has to
# wait for however long it actually takes a train stopping there to show up, not just within one
# headway period, and sees leg 2 (not stopping at C) come and go first.
LEG_EARLIEST_DEPARTURE[1] = 20
# Per leg and stop (waypoint group), the earliest departure and the latest arrival, as `Timetable` holds them.
LEG_EARLIEST_DEPARTURES: Dict[int, List[Optional[int]]] = {handle: [departure, None] for handle, departure in LEG_EARLIEST_DEPARTURE.items()}
LEG_LATEST_ARRIVALS: Dict[int, List[Optional[int]]] = {handle: [None, departure + STATION_DISTANCE] for handle, departure in LEG_EARLIEST_DEPARTURE.items()}
LEG_TARGET_ENTRY_POINT: Dict[int, str] = {handle: leg[-1][0] for handle, leg in LEG_WAYPOINTS.items()}


def _make_three_station_rail() -> RailGridTransitionMap:
    transitions = RailEnvTransitions()
    cells = transitions.transition_list
    dead_end = cells[7]
    straight = transitions.rotate_transition(cells[1], 90)
    opens_east = transitions.rotate_transition(dead_end, 270)  # connects only to its east neighbor
    opens_west = transitions.rotate_transition(dead_end, 90)  # connects only to its west neighbor
    row = [opens_east] + [straight] * (N_CELLS - 2) + [opens_west]
    rail = RailGridTransitionMap(width=N_CELLS, height=1, transitions=transitions)
    rail.grid = np.array([row], dtype=np.uint16)
    return rail


def _timetable_generator(agents, distance_map, hints, np_random) -> Timetable:
    return Timetable(earliest_departures=[LEG_EARLIEST_DEPARTURES[handle] for handle in range(len(agents))],
                     latest_arrivals=[LEG_LATEST_ARRIVALS[handle] for handle in range(len(agents))],
                     max_episode_steps=max(LEG_EARLIEST_DEPARTURE.values()) + STATION_DISTANCE)


def _make_three_station_graph_env() -> GraphRailEnv:
    """ The corridor as `GraphRailEnv`, each node's resource being its grid cell. """
    # TODO build the corridor graph natively instead of converting a grid.
    g = GraphTransitionMap.grid_to_digraph(_make_three_station_rail())
    return GraphRailEnv.from_graph(g, GraphTransitionMap.grid_to_resource_map(g), LEG_WAYPOINTS, observation_builder=VehicleObservationBuilder(),
                                   timetable_generator=_timetable_generator)


def _assert_predicted_targets(passenger_env: PassengerEnv):
    """ Each train's plan, as observed, ends at its target at its latest arrival, until it has arrived - then it is empty. """
    for handle, predictions in enumerate(passenger_env.vehicle_observations[0].vehicles.plans):
        if passenger_env.rail_env.agents[handle].state == TrainState.DONE:
            assert predictions == ()
        else:
            assert (predictions[-1].entry_point, predictions[-1].step, predictions[-1].stop) == (
                LEG_TARGET_ENTRY_POINT[handle], LEG_EARLIEST_DEPARTURE[handle] + STATION_DISTANCE, 1)


def _vehicle_policy() -> ShortestPathPolicy:
    return ShortestPathPolicy(router=greedy_graph_route_through_waypoints, serve_intermediate_stops=True)


def test_shortest_path_policy_runs_shuttle_trains_exactly_on_timetable():
    """
    Three stations A=(0,1), B=(0,4), C=(0,7) on one straight track, 3 cells apart. A rail env runs
    four trains, each at max speed 1, one leg each: two shuttle A<->B (legs 0: A->B, 2: B->A), two
    shuttle B<->C (legs 1: B->C, 3: C->B). Their earliest departures are staggered by a fixed headway
    of 6 steps, round-robin over the four legs (leg *h* departs at `(h+1) * HEADWAY`) - except leg 1
    (B->C), whose departure is deliberately postponed to step 20 instead of its regular slot at step
    12 (see `LEG_EARLIEST_DEPARTURE[1]`) - so no two trains ever need station B - the only cell shared
    between the two routes - at the same time. The corridor is a `GraphRailEnv`; positions below are the
    grid entry points `((row, col), direction)` of its nodes (direction 1 east, 3 west).

    A `PassengerEnv` without passengers steps the rail env's four legs with its `vehicle_policy`, a
    `ShortestPathPolicy`:

    - A `ShortestPathPolicy` with the graph router drives every one of the rail env's four trains
      from its start waypoint to its target waypoint, one cell per step (speed 1, no malfunctions, no
      other traffic in the way).
    - Each train stays off the track until its own earliest departure, appears at its start
      station on exactly that step, advances by exactly one cell per following step, and reaches its
      target - leaving the track immediately - exactly `STATION_DISTANCE` steps after departing,
      matching its timetabled latest arrival exactly.
    - No train is ever stopped: the rail env's headway leaves enough
      clearance that its four trains never contend for a cell.
    - The vehicle observations hold no plan after `reset()`. From the first step on, each train's plan ends at its
      target - its leg's own target waypoint - at the step of its timetabled latest arrival, until it arrives,
      whether that train is currently on or off the track.
    - `reset()` and `step()` return no passenger observations, no rewards, and the static data in the infos.
    """
    passenger_env = PassengerEnv(_make_three_station_graph_env(), vehicle_policy=_vehicle_policy(), passenger_journeys={},
                                 obs_builder=PassengerObservationBuilder())
    observations, infos = passenger_env.reset()
    assert observations == {}
    assert list(infos) == ["static"] and infos["static"].passenger_journeys == {}
    assert passenger_env.get_agent_handles() == []
    assert passenger_env.vehicle_observations[0].vehicles.plans == ((), (), (), ())

    # verbatim expected timetable for the rail env: one entry per leg, in handle order 0..3
    # (A->B, B->C, B->A, C->B), departures at HEADWAY=6, 12, 18, 24 and arrivals STATION_DISTANCE=3
    # steps later - except leg 1, postponed to depart (and so arrive) 8 steps later than its regular
    # slot (20/23 instead of 12/15).
    assert _timetable_generator(passenger_env.rail_env.agents, passenger_env.rail_env.distance_map, None, None) == Timetable(
        earliest_departures=[[6, None], [20, None], [18, None], [24, None]],
        latest_arrivals=[[None, 9], [None, 23], [None, 21], [None, 27]],
        max_episode_steps=27,
    )
    for handle, agent in enumerate(passenger_env.rail_env.agents):
        assert agent.earliest_departure == LEG_EARLIEST_DEPARTURE[handle]
        assert agent.latest_arrival == LEG_EARLIEST_DEPARTURE[handle] + STATION_DISTANCE

    rail_env_position_by_step: Dict[int, Dict[int, Tuple[int, int]]] = {handle: {} for handle in LEG_WAYPOINTS}
    while not passenger_env.done:
        obs, rewards, dones, infos = passenger_env.step({})
        step = passenger_env.rail_env._elapsed_steps
        assert (obs, rewards, list(infos)) == ({}, {}, ["static"])
        assert dones == {'__all__': passenger_env.rail_env.dones['__all__']}
        _assert_predicted_targets(passenger_env)
        for handle, agent in enumerate(passenger_env.rail_env.agents):
            rail_env_position_by_step[handle][step] = GraphTransitionMap.graph_entry_point_to_grid_entry_point(agent.current_entry_point)
            assert agent.state != TrainState.STOPPED

    # verbatim expected position trace for the rail env: for each leg (dict key), its position
    # at every elapsed step from 1 to 27 - `None` off track, `((0, col), direction)` on track, `None` again once
    # removed on arrival.
    assert rail_env_position_by_step == {
        0: {1: None, 2: None, 3: None, 4: None, 5: None, 6: ((0, 1), 1), 7: ((0, 2), 1), 8: ((0, 3), 1), 9: None, 10: None,
            11: None, 12: None, 13: None, 14: None, 15: None, 16: None, 17: None, 18: None, 19: None, 20: None,
            21: None, 22: None, 23: None, 24: None, 25: None, 26: None, 27: None},
        1: {1: None, 2: None, 3: None, 4: None, 5: None, 6: None, 7: None, 8: None, 9: None, 10: None,
            11: None, 12: None, 13: None, 14: None, 15: None, 16: None, 17: None, 18: None, 19: None, 20: ((0, 4), 1),
            21: ((0, 5), 1), 22: ((0, 6), 1), 23: None, 24: None, 25: None, 26: None, 27: None},
        2: {1: None, 2: None, 3: None, 4: None, 5: None, 6: None, 7: None, 8: None, 9: None, 10: None,
            11: None, 12: None, 13: None, 14: None, 15: None, 16: None, 17: None, 18: ((0, 4), 3), 19: ((0, 3), 3), 20: ((0, 2), 3),
            21: None, 22: None, 23: None, 24: None, 25: None, 26: None, 27: None},
        3: {1: None, 2: None, 3: None, 4: None, 5: None, 6: None, 7: None, 8: None, 9: None, 10: None,
            11: None, 12: None, 13: None, 14: None, 15: None, 16: None, 17: None, 18: None, 19: None, 20: None,
            21: None, 22: None, 23: None, 24: ((0, 7), 3), 25: ((0, 6), 3), 26: ((0, 5), 3), 27: None},
    }

    for handle, leg in LEG_WAYPOINTS.items():
        (start_position, direction) = GraphTransitionMap.graph_entry_point_to_grid_entry_point(leg[0][0])
        target_position, _ = GraphTransitionMap.graph_entry_point_to_grid_entry_point(leg[-1][0])
        step_towards_target = 1 if target_position[1] > start_position[1] else -1
        departure_step = LEG_EARLIEST_DEPARTURE[handle]
        arrival_step = departure_step + STATION_DISTANCE

        for step in range(1, departure_step):
            assert rail_env_position_by_step[handle][step] is None, f"leg {handle}: on track before its earliest departure at step {step}"
        for offset in range(STATION_DISTANCE):
            step = departure_step + offset
            expected_position = (start_position[0], start_position[1] + offset * step_towards_target)
            assert rail_env_position_by_step[handle][step] == (expected_position, direction), f"leg {handle} at step {step}"
        assert rail_env_position_by_step[handle][arrival_step] is None, f"leg {handle}: still on track at its latest arrival step {arrival_step}"


def test_multiple_passengers_share_trains():
    """
    Five passengers on the same four trains as above, each with its own journey:

    - 0: A->C, a single ride: no train stops at C coming from A (only A->B and B->C shuttles), so it
      never boards and waits at A until the end, without arriving.
    - 1: A->B, boards leg 0 at A and gets off at B.
    - 2: B->C, waits at B from the first step and boards leg 1.
    - 3: B->A, waits at B alongside passenger 2, and alone boards leg 2 (B->A), which is at B at step 18 - passenger 2
      stays, as leg 2 doesn't stop at its target.

    The passengers board in the step after they observe their train at their origin, before the train moves on: their
    mode changes from waiting at the origin straight to riding on the next cell.
    - 4: C->B, the only passenger on leg 3.

    All journeys have earliest departure 1 at the origin. Each passenger's mode changes (step -> `passenger_mode[h]`) are asserted verbatim; while riding,
    the passenger's resource equals its vehicle's every step.
    """
    a, b, c = STATION_A, STATION_B, STATION_C
    passenger_journeys = {handle: PassengerJourney(
        waypoints=[[GraphTransitionMap.grid_entry_point_to_graph_entry_point(*origin, direction)],
                   [GraphTransitionMap.grid_entry_point_to_graph_entry_point(*target, direction)]],
        waypoints_earliest_departure=[1, None], waypoints_latest_arrival=[None, 30])
        for handle, (origin, target, direction) in enumerate([(a, c, EAST), (a, b, EAST), (b, c, EAST), (b, a, WEST), (c, b, WEST)])}
    passenger_env = PassengerEnv(_make_three_station_graph_env(), vehicle_policy=_vehicle_policy(), passenger_journeys=passenger_journeys,
                                 obs_builder=PassengerObservationBuilder())
    passenger_policy = BoardForTargetPassengerPolicy()
    handles = passenger_env.get_agent_handles()
    observations, infos = passenger_env.reset()

    mode_changes: Dict[int, Dict[int, Tuple]] = {handle: {} for handle in passenger_journeys}
    previous_modes = {handle: None for handle in passenger_journeys}
    while not passenger_env.done:
        observations, _, _, infos = passenger_env.step(
            passenger_policy.act_many(handles, observations=[observations[handle] for handle in handles], infos=infos))
        step = passenger_env.rail_env._elapsed_steps
        for handle in passenger_journeys:
            mode = passenger_env.passenger_mode[handle]
            if mode != previous_modes[handle]:
                mode_changes[handle][step] = mode
            previous_modes[handle] = mode
            passenger_resource, riding_handle = mode
            if riding_handle is not None:
                vehicle_entry_point = passenger_env.rail_env.agents[riding_handle].current_entry_point
                assert passenger_env.rail_env.resource_map.get_resource(vehicle_entry_point, None) == passenger_resource

    assert mode_changes == {handle: _modes(changes) for handle, changes in {
        0: {1: (a, None)},
        1: {1: (a, None), 7: ((0, 2), 0), 8: ((0, 3), 0), 9: (b, None)},
        2: {1: (b, None), 21: ((0, 5), 1), 22: ((0, 6), 1), 23: (c, None)},
        3: {1: (b, None), 19: ((0, 3), 2), 20: ((0, 2), 2), 21: (a, None)},
        4: {1: (c, None), 25: ((0, 6), 3), 26: ((0, 5), 3), 27: (b, None)},
    }.items()}
    assert passenger_env.passenger_arrival_time == {0: None, 1: 9, 2: 23, 3: 21, 4: 27}
    assert passenger_env.passenger_target_entry_point == {handle: journey.waypoints[-1][0] if handle > 0 else None
                                                          for handle, journey in passenger_journeys.items()}
