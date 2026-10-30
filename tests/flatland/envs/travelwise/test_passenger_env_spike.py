from typing import Dict, List, Optional, Tuple

import numpy as np

from flatland.core.grid.grid4 import Grid4TransitionsEnum
from flatland.envs.graph.graph_routing import greedy_graph_route_through_waypoints
from flatland.envs.graph.rail_graph_transition_map import GraphTransitionMap
from flatland.envs.graph_rail_env import GraphRailEnv
from flatland.envs.grid.rail_env_grid import RailEnvTransitions
from flatland.envs.observations import FullEnvObservation
from flatland.envs.rail_env_policies import ShortestPathPolicy
from flatland.envs.rail_grid_transition_map import RailGridTransitionMap
from flatland.envs.step_utils.states import TrainState
from flatland.envs.timetable_utils import Timetable
from flatland.envs.travelwise.passenger_env import PassengerEnv

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
# Leg 1 (B->C) is deliberately postponed well past both its own regular headway slot (12) and the
# passenger's own arrival at B via leg 0 (step 9, riding leg 0 from A) - still clear of leg 2's own
# use of B (step 18, departing the other way) - so the passenger has to wait at B for however long
# it actually takes a train heading to its next stop to show up, not just within one headway period,
# before it can hop on leg 1 and ride it all the way to its own final target, C.
LEG_EARLIEST_DEPARTURE[1] = 20
# Per leg and stop (waypoint group), the earliest departure and the latest arrival, as `Timetable` holds them.
LEG_EARLIEST_DEPARTURES: Dict[int, List[Optional[int]]] = {handle: [departure, None] for handle, departure in LEG_EARLIEST_DEPARTURE.items()}
LEG_LATEST_ARRIVALS: Dict[int, List[Optional[int]]] = {handle: [None, departure + STATION_DISTANCE] for handle, departure in LEG_EARLIEST_DEPARTURE.items()}
LEG_TARGET_ENTRY_POINT: Dict[int, str] = {handle: leg[-1][0] for handle, leg in LEG_WAYPOINTS.items()}

# The single passenger's own route: A->B->C, stopping at B on the way rather than skipping
# straight to C - one ordered list of stops (entry points) per passenger handle.
PASSENGER_JOURNEYS: Dict[int, List[str]] = {
    0: [GraphTransitionMap.grid_entry_point_to_graph_entry_point(*STATION_A, EAST),
        GraphTransitionMap.grid_entry_point_to_graph_entry_point(*STATION_B, EAST),
        GraphTransitionMap.grid_entry_point_to_graph_entry_point(*STATION_C, EAST)],
}


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
    return GraphRailEnv.from_graph(g, GraphTransitionMap.grid_to_resource_map(g), LEG_WAYPOINTS, observation_builder=FullEnvObservation(),
                                   timetable_generator=_timetable_generator)


def _assert_predicted_targets(passenger_env: PassengerEnv):
    """ Each train's prediction ends at its target at its latest arrival, until it has arrived - then it is empty. """
    for handle, predictions in passenger_env.vehicle_predictions.items():
        if passenger_env.rail_env.agents[handle].state == TrainState.DONE:
            assert predictions == []
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
    grid entry points `((row, col), direction)` of its nodes (direction 1 east, 3 west), the resources
    being the strings of their cells, e.g. `'(0, 1)'`.

    A `PassengerEnv` wraps this `GraphRailEnv` (the "rail env") and adds a passenger, travelling the
    journey `PASSENGER_JOURNEYS[0]` (A->B->C) on the rail env's trains. It shares only the corridor's
    topology with the rail env - never its agents, timetable or motion checks, and never a real rail env/`EnvAgent` of its own. `PassengerEnv` implements the
    `Environment` interface for the passenger: `step()` takes the passenger's (empty) actions, steps the
    rail env's four legs with its `vehicle_policy`, a `ShortestPathPolicy`, and returns empty
    observations, rewards and infos. The passenger never moves under its own action: it
    either waits at one of its own stops for a train heading to its next stop, or - once one shows up -
    rides that train's position exactly, one train at a time, boarding and alighting
    automatically. Its `passenger_policy` is still a required constructor argument, but is not
    exercised (`None` here) - `PassengerEnv` ignores
    passengers' own actions entirely while riding mode is in place.

    - A `ShortestPathPolicy` with the graph router drives every one of the rail env's four trains from its start
      waypoint to its target waypoint, one cell per step (speed 1, no malfunctions, no other traffic
      in the way).
    - Each train stays off the track until its own earliest departure, appears at its start
      station on exactly that step, advances by exactly one cell per following step, and reaches its
      target - leaving the track immediately - exactly `STATION_DISTANCE` steps after departing,
      matching its timetabled latest arrival exactly.
    - The passenger is off map after `reset()`, shows up at its first stop, A, on the first step, and
      waits there. Leg 0 (A->B) is the
      first train heading to the passenger's next stop (B) to ever occupy A - the
      passenger boards it the instant it does, rides its exact position cell by cell, and alights
      onto B the same step that train reaches its own target and is removed from the rail env.
      It then waits at B considerably longer - 11 steps, from step 9 to step 20 - than it did at A,
      since leg 1 (B->C), the only train ever heading to its next stop, C, is the one deliberately
      postponed above: the passenger simply keeps waiting until leg 1 actually shows up, boards it
      the instant it does, and alights onto C, its own final target, the step leg 1 arrives there and
      is removed. Once at C - its own last stop - the passenger simply stays there, with
      `passenger_arrival_time`/`passenger_target_entry_point` recorded the same step its position is
      written as C.
    - No train is ever stopped: the rail env's headway leaves enough
      clearance that its four trains never contend for a cell.
    - `PassengerEnv`'s `vehicle_predictor` predicts each train's route up to its target - its leg's own
      target waypoint - at the step of its timetabled latest arrival, throughout the run until it
      arrives, whether that train is currently on or off the track. This is exactly what the
      passenger's boarding decision reads to tell which train, if any currently sharing its cell, is
      headed towards its own next stop.
    """
    passenger_env = PassengerEnv(_make_three_station_graph_env(), vehicle_policy=_vehicle_policy(), passenger_policy=None,
                                 passenger_journeys=PASSENGER_JOURNEYS)
    assert passenger_env.reset() == ({}, {})
    assert passenger_env.get_agent_handles() == [0]

    # the predictions already end at each leg's target entry point, at its latest arrival, right after
    # reset(), before any train has moved.
    _assert_predicted_targets(passenger_env)

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

    # the passenger is off map (mode 1) after reset(), until the first step.
    assert passenger_env.passenger_mode == {0: (None, None)}
    assert passenger_env.passenger_current_entry_point == {0: None}

    rail_env_position_by_step: Dict[int, Dict[int, Tuple[int, int]]] = {handle: {} for handle in LEG_WAYPOINTS}
    passenger_position_by_step: Dict[int, Tuple[int, int]] = {}
    passenger_mode_by_step: Dict[int, Tuple] = {}
    while not passenger_env.done:
        obs, rewards, dones, infos = passenger_env.step({})
        step = passenger_env.rail_env._elapsed_steps
        assert (obs, rewards, infos) == ({}, {}, {})
        assert dones == {'__all__': passenger_env.rail_env.dones['__all__']}
        _assert_predicted_targets(passenger_env)
        for handle, agent in enumerate(passenger_env.rail_env.agents):
            rail_env_position_by_step[handle][step] = GraphTransitionMap.graph_entry_point_to_grid_entry_point(agent.current_entry_point)
            assert agent.state != TrainState.STOPPED
        passenger_resource, riding_handle = passenger_env.passenger_mode[0]
        passenger_position_by_step[step] = GraphTransitionMap.graph_entry_point_to_grid_entry_point(passenger_env.passenger_current_entry_point[0])
        passenger_mode_by_step[step] = passenger_env.passenger_mode[0]
        # while riding (mode 3), the ridden train's own position matches the
        # passenger's position exactly, every step - the invariant `passenger_mode[h]`'s riding entry itself
        # is a handle (not a position) relies on.
        if riding_handle is not None:
            vehicle_entry_point = passenger_env.rail_env.agents[riding_handle].current_entry_point
            assert passenger_env.rail_env.resource_map.get_resource(vehicle_entry_point, None) == passenger_resource

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

    # verbatim expected position trace for the passenger: waiting at A from step 1 until step 6, when leg 0 (A->B) reaches A and it boards;
    # rides leg 0's exact position for steps 6-8; alights onto B the moment leg 0 arrives and is
    # removed, step 9, then waits there considerably longer - until step 20, when the deliberately
    # postponed leg 1 (B->C) finally reaches B and it boards; rides leg 1's exact position for steps
    # 20-22; alights onto C, its own final target, at step 23 and stays there for the remainder of the
    # run (steps 23-27), since there is no next stop left to wait for.
    assert passenger_position_by_step == {
        1: ((0, 1), 1), 2: ((0, 1), 1), 3: ((0, 1), 1), 4: ((0, 1), 1), 5: ((0, 1), 1), 6: ((0, 1), 1), 7: ((0, 2), 1), 8: ((0, 3), 1), 9: ((0, 4), 1),
        10: ((0, 4), 1), 11: ((0, 4), 1), 12: ((0, 4), 1), 13: ((0, 4), 1), 14: ((0, 4), 1), 15: ((0, 4), 1), 16: ((0, 4), 1), 17: ((0, 4), 1),
        18: ((0, 4), 1), 19: ((0, 4), 1), 20: ((0, 4), 1), 21: ((0, 5), 1), 22: ((0, 6), 1), 23: ((0, 7), 1), 24: ((0, 7), 1), 25: ((0, 7), 1),
        26: ((0, 7), 1), 27: ((0, 7), 1),
    }

    # matching mode trace: waiting at a stop, not riding (mode 2) while
    # `passenger_mode[0][1]` (the ridden train's handle) is `None`; riding (mode 3) - that
    # handle, with the ridden agent's own position always equal to the passenger's - while it
    # isn't. Boarding/alighting steps are exactly where `passenger_position_by_step` above holds steady
    # across the transition: step 6, waiting -> riding leg 0 (A->B) at A; step 9, riding -> waiting at
    # B; step 20, waiting -> riding leg 1 (B->C) at B, after waiting there far longer than at A; step
    # 23, riding -> waiting at C, permanently, since C is the passenger's own last stop.
    assert passenger_mode_by_step == _modes({
        1: ((0, 1), None), 2: ((0, 1), None), 3: ((0, 1), None), 4: ((0, 1), None), 5: ((0, 1), None),
        6: ((0, 1), 0), 7: ((0, 2), 0), 8: ((0, 3), 0),
        9: ((0, 4), None), 10: ((0, 4), None), 11: ((0, 4), None), 12: ((0, 4), None), 13: ((0, 4), None),
        14: ((0, 4), None), 15: ((0, 4), None), 16: ((0, 4), None), 17: ((0, 4), None), 18: ((0, 4), None),
        19: ((0, 4), None),
        20: ((0, 4), 1), 21: ((0, 5), 1), 22: ((0, 6), 1),
        23: ((0, 7), None), 24: ((0, 7), None), 25: ((0, 7), None), 26: ((0, 7), None), 27: ((0, 7), None),
    })

    # the passenger's own target entry point - C, arriving from the east, as a graph node - and the
    # step its arrival is recorded on, the same step its position is actually written as C.
    assert passenger_env.passenger_target_entry_point[0] == GraphTransitionMap.grid_entry_point_to_graph_entry_point(*STATION_C, EAST)
    assert passenger_env.passenger_arrival_time[0] == 23

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

    # from the first step on, the passenger is never off map again - unlike a train, it is never
    # removed from the map once it boards its first train.
    for step in range(1, 28):
        assert passenger_position_by_step[step] is not None, f"passenger: off map at step {step}"

    # the passenger only ever rides a leg that is, at that same step, at exactly the
    # passenger's own recorded position - cross-checked here against the recorded traces (the loop above
    # already asserts the same against the live agents, every step).
    for step, (passenger_resource, riding_handle) in passenger_mode_by_step.items():
        if riding_handle is not None:
            assert GraphTransitionMap.grid_entry_point_to_graph_resource(*rail_env_position_by_step[riding_handle][step][0]) == passenger_resource


def test_multiple_passengers_share_trains():
    """
    Five passengers on the same four trains as above, each with its own journey:

    - 0: A->B->C, as in the single-passenger test.
    - 1: A->B, boards leg 0 at A together with passenger 0, but gets off for good at B.
    - 2: B->C, waits at B from the first step and boards leg 1 together with passenger 0.
    - 3: B->A, waits at B alongside passengers 0 and 2, and alone boards leg 2 (B->A) at step 18 - the
      others stay, as leg 2 doesn't head to their next stop.
    - 4: C->B, the only passenger on leg 3.

    Each passenger's mode changes (step -> `passenger_mode[h]`) are asserted verbatim; while riding,
    the passenger's resource equals its vehicle's every step.
    """
    a, b, c = STATION_A, STATION_B, STATION_C
    passenger_journeys = {
        0: [GraphTransitionMap.grid_entry_point_to_graph_entry_point(*a, EAST),
            GraphTransitionMap.grid_entry_point_to_graph_entry_point(*b, EAST),
            GraphTransitionMap.grid_entry_point_to_graph_entry_point(*c, EAST)],
        1: [GraphTransitionMap.grid_entry_point_to_graph_entry_point(*a, EAST), GraphTransitionMap.grid_entry_point_to_graph_entry_point(*b, EAST)],
        2: [GraphTransitionMap.grid_entry_point_to_graph_entry_point(*b, EAST), GraphTransitionMap.grid_entry_point_to_graph_entry_point(*c, EAST)],
        3: [GraphTransitionMap.grid_entry_point_to_graph_entry_point(*b, WEST), GraphTransitionMap.grid_entry_point_to_graph_entry_point(*a, WEST)],
        4: [GraphTransitionMap.grid_entry_point_to_graph_entry_point(*c, WEST), GraphTransitionMap.grid_entry_point_to_graph_entry_point(*b, WEST)],
    }
    passenger_env = PassengerEnv(_make_three_station_graph_env(), vehicle_policy=_vehicle_policy(), passenger_policy=None,
                                 passenger_journeys=passenger_journeys)
    passenger_env.reset()

    mode_changes: Dict[int, Dict[int, Tuple]] = {handle: {} for handle in passenger_journeys}
    previous_modes = {handle: None for handle in passenger_journeys}
    while not passenger_env.done:
        passenger_env.step({})
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
        0: {1: (a, None), 6: (a, 0), 7: ((0, 2), 0), 8: ((0, 3), 0), 9: (b, None),
            20: (b, 1), 21: ((0, 5), 1), 22: ((0, 6), 1), 23: (c, None)},
        1: {1: (a, None), 6: (a, 0), 7: ((0, 2), 0), 8: ((0, 3), 0), 9: (b, None)},
        2: {1: (b, None), 20: (b, 1), 21: ((0, 5), 1), 22: ((0, 6), 1), 23: (c, None)},
        3: {1: (b, None), 18: (b, 2), 19: ((0, 3), 2), 20: ((0, 2), 2), 21: (a, None)},
        4: {1: (c, None), 24: (c, 3), 25: ((0, 6), 3), 26: ((0, 5), 3), 27: (b, None)},
    }.items()}
    assert passenger_env.passenger_arrival_time == {0: 23, 1: 9, 2: 23, 3: 21, 4: 27}
    assert passenger_env.passenger_target_entry_point == {handle: journey[-1] for handle, journey in passenger_journeys.items()}
