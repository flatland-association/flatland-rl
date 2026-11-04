from typing import Dict, List, Tuple

import numpy as np
import pytest

from flatland.core.policy import Policy
from flatland.envs.graph.graph_routing import greedy_graph_route_through_waypoints
from flatland.envs.graph.rail_graph_transition_map import GraphTransitionMap
from flatland.envs.graph_rail_env import GraphRailEnv
from flatland.envs.rail_env_policies import ShortestPathPolicy
from flatland.envs.timetable_utils import Timetable
from flatland.envs.travelwise.passenger_env import PassengerEnv, PassengerJourney
from flatland.envs.travelwise.passenger_env_action import PassengerAction, PassengerActionT
from flatland.envs.travelwise.passenger_env_observations import PassengerObservation, PassengerObservationBuilder
from flatland.envs.travelwise.passenger_env_policies import BoardForTargetPassengerPolicy
from flatland.envs.travelwise.timetable_predictor import ProjectedEntryPoint
from flatland.envs.travelwise.travelwise_policy_runner import TravelWisePolicyRunner
from flatland.envs.travelwise.vehicle_observations import VehicleObservationBuilder
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
    3: _journey(A_EAST, B_EAST, 4, 20),  # earliest departure after T0 has left A: takes T2
    4: _journey(C_WEST, B_WEST, 1, 20),  # rides T1 to its final stop
    5: _journey(C_WEST, A_WEST, 1, 20),  # no vehicle stops at A heading west: never boards
}


def _make_scenario_env() -> PassengerEnv:
    g = GraphTransitionMap.grid_to_digraph(_make_three_station_rail())
    rail_env = GraphRailEnv.from_graph(g, GraphTransitionMap.grid_to_resource_map(g), VEHICLE_WAYPOINTS, observation_builder=VehicleObservationBuilder(),
                                       timetable_generator=lambda *args, **kwargs: VEHICLE_TIMETABLE)
    vehicle_policy = ShortestPathPolicy(router=greedy_graph_route_through_waypoints, serve_intermediate_stops=True)
    return PassengerEnv(rail_env, vehicle_policy=vehicle_policy, passenger_journeys=PASSENGER_JOURNEYS, obs_builder=PassengerObservationBuilder())


def _run(passenger_env: PassengerEnv, passenger_policy: Policy) -> Tuple[Dict[int, Dict[int, PassengerObservation]], Dict[int, Dict[int, Tuple]],
                                                                         Dict[int, Dict[int, int]], Dict]:
    """
    Drives the passengers of `passenger_env` with `passenger_policy` (`TravelWisePolicyRunner`) until the rail env is
    done: per step, the policy acts on the observations of the previous step. Returns the observations per step (0
    after `reset()`), the mode changes per passenger (step -> `passenger_mode[h]`), the non-zero rewards per passenger
    (step -> reward) and the last dones.
    """
    handles = passenger_env.get_agent_handles()
    runner = TravelWisePolicyRunner(passenger_env, passenger_policy)
    observations, _ = runner.reset()
    observations_by_step = {0: observations}
    mode_changes: Dict[int, Dict[int, Tuple]] = {handle: {} for handle in handles}
    previous_modes = {handle: None for handle in handles}
    rewards_by_step: Dict[int, Dict[int, int]] = {handle: {} for handle in handles}
    dones = {}
    while not passenger_env.done:
        observations, rewards, dones, _ = runner.step()
        step = passenger_env.rail_env._elapsed_steps
        observations_by_step[step] = observations
        for handle in handles:
            if rewards[handle] != 0:
                rewards_by_step[handle][step] = rewards[handle]
            mode = passenger_env.passenger_mode[handle]
            if mode != previous_modes[handle]:
                mode_changes[handle][step] = mode
            previous_modes[handle] = mode
    return observations_by_step, mode_changes, rewards_by_step, dones


def test_passenger_env_scenario_single_rides():
    """
    Single-ride journeys on the scenario's vehicles - mode changes (step -> `passenger_mode[h]`, given with grid cells)
    and arrivals asserted verbatim:

    The passengers act on what they observed in the previous step: they board and alight in the step after they
    observe the vehicle, before it moves on.

    - 0: A -> C observes T0 at A at step 2, its earliest departure there being 1, boards it at step 3 as T0 moves on,
      stays on board at T0's intermediate stop B (5-10), as B is not its target, and arrives at C at 13, T0's final
      stop.
    - 1: A -> B boards T0 together with passenger 0, observes T0 stopping at its target B at step 5 and alights there
      at step 6.
    - 2: B -> C waits at B, observes T0 stopping at B at step 5 and boards it at step 6.
    - 3: A -> B with earliest departure 4: its policy does not board T0, observed at A only at step 2 - boarding in
      step 3 would be before the earliest departure - and takes T2, at A at step 15, at step 16.
    - 4: C -> B boards T1 at step 15, at C at step 14, and rides it to its final stop B, where Flatland removes T1 in
      the step it arrives (17).
    - 5: C -> A never boards: T1 stops at B only, so it waits at C until the end, without arriving.

    Rewards (step -> reward): at arrival, the latest arrival minus the arrival step - 1 is late (latest arrival 4,
    arrives at 6); 5 gets `-max_episode_steps` when the rail env is done (18, all vehicles arrived).
    """
    passenger_env = _make_scenario_env()
    _, mode_changes, rewards, dones = _run(passenger_env, BoardForTargetPassengerPolicy())

    assert mode_changes == {handle: _modes(changes) for handle, changes in {
        0: {1: (A, None), 3: ((0, 2), 0), 4: ((0, 3), 0), 5: (B, 0), 11: ((0, 5), 0), 12: ((0, 6), 0), 13: (C, None)},
        1: {1: (A, None), 3: ((0, 2), 0), 4: ((0, 3), 0), 5: (B, 0), 6: (B, None)},
        2: {1: (B, None), 6: (B, 0), 11: ((0, 5), 0), 12: ((0, 6), 0), 13: (C, None)},
        3: {1: (A, None), 16: ((0, 2), 2), 17: ((0, 3), 2), 18: (B, None)},
        4: {1: (C, None), 15: ((0, 6), 1), 16: ((0, 5), 1), 17: (B, None)},
        5: {1: (C, None)},
    }.items()}
    assert passenger_env.passenger_arrival_time == {0: 13, 1: 6, 2: 13, 3: 18, 4: 17, 5: None}
    assert passenger_env.passenger_target_entry_point == {0: C_EAST, 1: B_EAST, 2: C_EAST, 3: B_EAST, 4: B_WEST, 5: None}
    assert rewards == {0: {13: 2}, 1: {6: -2}, 2: {13: 2}, 3: {18: 2}, 4: {17: 3}, 5: {18: -25}}
    assert dones == {0: True, 1: True, 2: True, 3: True, 4: True, 5: False, '__all__': True}


class _ScriptedPassengerPolicy(BoardForTargetPassengerPolicy):
    """
    `BoardForTargetPassengerPolicy`, except for the actions scripted per (passenger handle, step the action is applied
    in), counting the steps itself.
    """

    def __init__(self, script: Dict[Tuple[int, int], PassengerActionT]):
        super().__init__()
        self.script = script
        self._step = 0

    def act_many(self, handles: List[int], observations: List[PassengerObservation], **kwargs) -> Dict[int, PassengerActionT]:
        self._step += 1
        actions = super().act_many(handles, observations, **kwargs)
        return {handle: self.script.get((handle, self._step), action) for handle, action in actions.items()}


def test_passenger_env_scenario_ignores_invalid_actions_and_hops_off_anywhere():
    """
    The scenario with scripted actions on top of `BoardForTargetPassengerPolicy`:

    - 0: hops off T0 at step 4, at (0, 2) between stations: it waits there, without arriving - no vehicle serving C
      ever comes by.
    - 5 (C -> A): hops on T1 at C at step 15, which doesn't stop at A - valid, as whether a vehicle serves the
      passenger is the policy's decision: it is forced off at T1's final stop B (17), without arriving.
    - 3 (A -> B, earliest departure 4): hops on T0 at A at step 3, having observed it there at step 2, one step before
      its earliest departure - feasible, as respecting the earliest departure is the policy's decision, but penalized
      (-1); it rides T0 and arrives at B at 6, alighting the step after observing T0 stop there (+14).
    - Ignored, as infeasible: 2 hops on T0 at step 3, while T0 is at A, not at B where 2 waits; 1 hops on T0 at B at
      step 8, after having arrived there.
    """
    passenger_env = _make_scenario_env()
    _, mode_changes, rewards, dones = _run(passenger_env, _ScriptedPassengerPolicy({
        (0, 4): (PassengerAction.HOP_OFF, None),
        (3, 3): (PassengerAction.HOP_ON, 0),
        (2, 3): (PassengerAction.HOP_ON, 0),
        (5, 15): (PassengerAction.HOP_ON, 1),
        (1, 8): (PassengerAction.HOP_ON, 0),
    }))

    assert mode_changes[0] == _modes({1: (A, None), 3: ((0, 2), 0), 4: ((0, 2), None)})
    assert mode_changes[2] == _modes({1: (B, None), 6: (B, 0), 11: ((0, 5), 0), 12: ((0, 6), 0), 13: (C, None)})
    assert mode_changes[3] == _modes({1: (A, None), 3: ((0, 2), 0), 4: ((0, 3), 0), 5: (B, 0), 6: (B, None)})
    assert mode_changes[5] == _modes({1: (C, None), 15: ((0, 6), 1), 16: ((0, 5), 1), 17: (B, None)})
    assert passenger_env.passenger_arrival_time == {0: None, 1: 6, 2: 13, 3: 6, 4: 17, 5: None}
    assert rewards == {0: {18: -25}, 1: {6: -2}, 2: {13: 2}, 3: {3: -1, 6: 14}, 4: {17: 3}, 5: {18: -25}}
    assert dones == {0: False, 1: True, 2: True, 3: True, 4: True, 5: False, '__all__': True}
    assert passenger_env.passenger_current_entry_point[0] == GraphTransitionMap.grid_entry_point_to_graph_entry_point(0, 2, EAST)


@pytest.mark.parametrize("action", [(PassengerAction.HOP_ON, -3), (PassengerAction.HOP_ON, 3), (PassengerAction.HOP_ON, "0"), PassengerAction.HOP_ON,
                                    (PassengerAction.HOP_ON, np.int64(-3))],
                         ids=["negative handle", "handle out of range", "handle not an int", "not a pair", "negative numpy handle"])
def test_passenger_env_scenario_warns_on_unknown_vehicle_handles_and_malformed_actions(action):
    """
    Passenger 2 waits at B, where T0 (vehicle 0 of 3) stands from step 5 to 10. An invalid action at step 6 is ignored -
    a negative handle (-3, T0 if indexed from the end), one out of range, one that is not an int, or an action that is
    not an `(action, vehicle handle)` pair - with a warning, and the passenger boards T0 one step later, by its policy.
    """
    passenger_env = _make_scenario_env()
    with pytest.warns(UserWarning, match="Passenger 2: .*, ignored."):
        _, mode_changes, rewards, _ = _run(passenger_env, _ScriptedPassengerPolicy({(2, 6): action}))

    assert mode_changes[2] == _modes({1: (B, None), 7: (B, 0), 11: ((0, 5), 0), 12: ((0, 6), 0), 13: (C, None)})
    assert passenger_env.passenger_arrival_time[2] == 13
    assert rewards[2] == {13: 2}


def test_passenger_env_scenario_observations():
    """
    The observations `reset()`/`step()` return, which the passenger policy acts on in the next step, for passengers 0
    (A -> C) and 1 (A -> B): off map after `reset()`; at step 2, waiting at A, with T0 to board - stopping at C at 13
    for 0, at B at 5 for 1; riding T0 from step 3 on; at step 5, T0 stopping at B, 1's target; at step 6, 1 has
    arrived. Passenger 3 (A -> B, earliest departure 4) observes T0 at A at step 2 as a vehicle to board too: whether
    to board before the earliest departure is its policy's decision. Each passenger observes the index of its journey's
    group it is at or last left: its origin (0) until it arrives at its target (1).
    """
    passenger_env = _make_scenario_env()
    observations_by_step, _, _, _ = _run(passenger_env, BoardForTargetPassengerPolicy())
    cell = GraphTransitionMap.grid_entry_point_to_graph_resource

    assert observations_by_step[0][1] == PassengerObservation(elapsed_steps=0, mode=(None, None), stop_index=0,
                                                              vehicles_to_board={}, vehicle_stops_at_next_stop=False)
    assert observations_by_step[2][0] == PassengerObservation(elapsed_steps=2, mode=(cell(*A), None), stop_index=0,
                                                              vehicles_to_board={0: [ProjectedEntryPoint(C_EAST, 13, 2)]},
                                                              vehicle_stops_at_next_stop=False)
    assert observations_by_step[2][1] == PassengerObservation(elapsed_steps=2, mode=(cell(*A), None), stop_index=0,
                                                              vehicles_to_board={0: [ProjectedEntryPoint(B_EAST, 5, 1)]},
                                                              vehicle_stops_at_next_stop=False)
    assert observations_by_step[2][3] == PassengerObservation(elapsed_steps=2, mode=(cell(*A), None), stop_index=0,
                                                              vehicles_to_board={0: [ProjectedEntryPoint(B_EAST, 5, 1)]},
                                                              vehicle_stops_at_next_stop=False)
    assert observations_by_step[3][1] == PassengerObservation(elapsed_steps=3, mode=(cell(0, 2), 0), stop_index=0,
                                                              vehicles_to_board={}, vehicle_stops_at_next_stop=False)
    assert observations_by_step[5][0] == PassengerObservation(elapsed_steps=5, mode=(cell(*B), 0), stop_index=0,
                                                              vehicles_to_board={}, vehicle_stops_at_next_stop=False)
    assert observations_by_step[5][1] == PassengerObservation(elapsed_steps=5, mode=(cell(*B), 0), stop_index=0,
                                                              vehicles_to_board={}, vehicle_stops_at_next_stop=True)
    assert observations_by_step[6][1] == PassengerObservation(elapsed_steps=6, mode=(cell(*B), None), stop_index=1,
                                                              vehicles_to_board={}, vehicle_stops_at_next_stop=False)


def test_board_for_target_passenger_policy_waits_for_the_earliest_departure_at_its_stop_index():
    """
    The policy boards not before the earliest departure of the group at its observed stop index, also on a journey
    visiting a group twice: on A -> B -> A, waiting at B (stop index 1, earliest departure 10) for A. Boarding applies
    in the step after the one observed.
    """
    cell = GraphTransitionMap.grid_entry_point_to_graph_resource
    journey = PassengerJourney(waypoints=[[A_EAST], [B_EAST], [A_EAST]], waypoints_earliest_departure=[1, 10, None],
                               waypoints_latest_arrival=[None, 8, 30])

    def observation(elapsed_steps: int) -> PassengerObservation:
        return PassengerObservation(elapsed_steps=elapsed_steps, mode=(cell(*B), None), stop_index=1,
                                    vehicles_to_board={1: [ProjectedEntryPoint(A_EAST, 20, 2)]}, vehicle_stops_at_next_stop=False)

    policy = BoardForTargetPassengerPolicy()
    assert policy.act(observation(8), journey=journey) == (PassengerAction.WAIT, None)
    assert policy.act(observation(9), journey=journey) == (PassengerAction.HOP_ON, 1)
