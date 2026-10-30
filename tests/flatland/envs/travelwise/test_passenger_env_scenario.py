from typing import Dict, Tuple

from flatland.envs.graph.graph_routing import greedy_graph_route_through_waypoints
from flatland.envs.graph.rail_graph_transition_map import GraphTransitionMap
from flatland.envs.graph_rail_env import GraphRailEnv
from flatland.envs.observations import FullEnvObservation
from flatland.envs.rail_env_policies import ShortestPathPolicy
from flatland.envs.timetable_utils import Timetable
from flatland.envs.travelwise.passenger_env import PassengerEnv, PassengerJourney
from tests.flatland.envs.travelwise.test_passenger_env_spike import _make_three_station_rail, _modes

EAST, WEST = 1, 3
A, B, C = (0, 1), (0, 4), (0, 7)
A_EAST, B_EAST, C_EAST = [GraphTransitionMap.grid_entry_point_to_graph_entry_point(*station, EAST) for station in (A, B, C)]
A_WEST, B_WEST, C_WEST = [GraphTransitionMap.grid_entry_point_to_graph_entry_point(*station, WEST) for station in (A, B, C)]

# Vehicles on the corridor A-B-C, scheduled so they never contend for a cell:
# - T0: through train A -> B -> C, stopping at B until its earliest departure 10;
# - T1: C -> B;
# - T2: a later A -> B.
VEHICLE_WAYPOINTS = {0: [[A_EAST], [B_EAST], [C_EAST]], 1: [[C_WEST], [B_WEST]], 2: [[A_EAST], [B_EAST]]}
VEHICLE_TIMETABLE = Timetable(earliest_departures=[[2, 10, None], [14, None], [15, None]],
                              latest_arrivals=[[None, 8, 20], [None, 17], [None, 18]], max_episode_steps=25)


def _journey(origin: str, target: str, earliest_departure: int, latest_arrival: int) -> PassengerJourney:
    return PassengerJourney(waypoints=[[origin], [target]], waypoints_earliest_departure=[earliest_departure, None],
                            waypoints_latest_arrival=[None, latest_arrival])


PASSENGER_JOURNEYS = {
    0: _journey(A_EAST, C_EAST, 1, 15),  # rides T0 through B
    1: _journey(A_EAST, B_EAST, 1, 4),  # rides T0, hops off at its intermediate stop B - late
    2: _journey(B_EAST, C_EAST, 1, 15),  # boards T0 at B
    3: _journey(A_EAST, B_EAST, 3, 20),  # earliest departure after T0 has left A: takes T2
    4: _journey(C_WEST, B_WEST, 1, 20),  # rides T1 to its final stop
    5: _journey(C_WEST, A_WEST, 1, 20),  # no vehicle stops at A heading west: never boards
}


def _make_scenario_env() -> PassengerEnv:
    g = GraphTransitionMap.grid_to_digraph(_make_three_station_rail())
    rail_env = GraphRailEnv.from_graph(g, GraphTransitionMap.grid_to_resource_map(g), VEHICLE_WAYPOINTS, observation_builder=FullEnvObservation(),
                                       timetable_generator=lambda *args, **kwargs: VEHICLE_TIMETABLE)
    vehicle_policy = ShortestPathPolicy(router=greedy_graph_route_through_waypoints, serve_intermediate_stops=True)
    return PassengerEnv(rail_env, vehicle_policy=vehicle_policy, passenger_policy=None, passenger_journeys=PASSENGER_JOURNEYS)


def test_passenger_env_scenario_single_rides():
    """
    Single-ride journeys on the scenario's vehicles - mode changes (step -> `passenger_mode[h]`, given with grid cells)
    and arrivals asserted verbatim:

    - 0: A -> C boards T0 at A at step 2, its earliest departure there being 1, stays on board at T0's intermediate
      stop B (5-10), as B is not its target, and arrives at C at 13, T0's final stop.
    - 1: A -> B boards T0 together with passenger 0 and alights at T0's intermediate stop B, in the step T0 stops
      there (5).
    - 2: B -> C waits at B and boards T0 in the step it stops at B (5).
    - 3: A -> B with earliest departure 3 does not board T0, which is at A only at step 2, and takes T2 at 15.
    - 4: C -> B rides T1 to its final stop B, where Flatland removes T1 in the step it arrives (17).
    - 5: C -> A never boards: T1 stops at B only, so it waits at C until the end, without arriving.
    """
    passenger_env = _make_scenario_env()
    passenger_env.reset()

    mode_changes: Dict[int, Dict[int, Tuple]] = {handle: {} for handle in PASSENGER_JOURNEYS}
    previous_modes = {handle: None for handle in PASSENGER_JOURNEYS}
    while not passenger_env.done:
        passenger_env.step({})
        step = passenger_env.rail_env._elapsed_steps
        for handle in PASSENGER_JOURNEYS:
            mode = passenger_env.passenger_mode[handle]
            if mode != previous_modes[handle]:
                mode_changes[handle][step] = mode
            previous_modes[handle] = mode

    assert mode_changes == {handle: _modes(changes) for handle, changes in {
        0: {1: (A, None), 2: (A, 0), 3: ((0, 2), 0), 4: ((0, 3), 0), 5: (B, 0), 11: ((0, 5), 0), 12: ((0, 6), 0), 13: (C, None)},
        1: {1: (A, None), 2: (A, 0), 3: ((0, 2), 0), 4: ((0, 3), 0), 5: (B, None)},
        2: {1: (B, None), 5: (B, 0), 11: ((0, 5), 0), 12: ((0, 6), 0), 13: (C, None)},
        3: {1: (A, None), 15: (A, 2), 16: ((0, 2), 2), 17: ((0, 3), 2), 18: (B, None)},
        4: {1: (C, None), 14: (C, 1), 15: ((0, 6), 1), 16: ((0, 5), 1), 17: (B, None)},
        5: {1: (C, None)},
    }.items()}
    assert passenger_env.passenger_arrival_time == {0: 13, 1: 5, 2: 13, 3: 18, 4: 17, 5: None}
    assert passenger_env.passenger_target_entry_point == {0: C_EAST, 1: B_EAST, 2: C_EAST, 3: B_EAST, 4: B_WEST, 5: None}
