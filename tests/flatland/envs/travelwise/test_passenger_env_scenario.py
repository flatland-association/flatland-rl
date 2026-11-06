from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
import pytest

from flatland.core.policy import Policy
from flatland.envs.graph.graph_routing import greedy_graph_route_through_waypoints
from flatland.envs.graph.rail_graph_transition_map import GraphTransitionMap
from flatland.envs.graph_rail_env import GraphRailEnv
from flatland.envs.rail_env_action import RailEnvActions
from flatland.envs.rail_env_policies import ShortestPathPolicy
from flatland.envs.timetable_utils import Timetable
from flatland.envs.travelwise.passenger_env import PassengerEnv, PassengerJourney
from flatland.envs.travelwise.passenger_env_action import PassengerAction, PassengerActionT
from flatland.envs.travelwise.passenger_env_observations import PassengerObservation, PassengerObservationBuilder
from flatland.envs.travelwise.passenger_env_policies import BoardForTargetPassengerPolicy
from flatland.envs.travelwise.passenger_env_rewards import PassengerRewards, PassengerTransitionData
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


def _make_scenario_env(rewards: PassengerRewards = None, serve_intermediate_stops: bool = True) -> PassengerEnv:
    g = GraphTransitionMap.grid_to_digraph(_make_three_station_rail())
    rail_env = GraphRailEnv.from_graph(g, GraphTransitionMap.grid_to_resource_map(g), VEHICLE_WAYPOINTS, observation_builder=VehicleObservationBuilder(),
                                       timetable_generator=lambda *args, **kwargs: VEHICLE_TIMETABLE)
    vehicle_policy = ShortestPathPolicy(router=greedy_graph_route_through_waypoints, serve_intermediate_stops=serve_intermediate_stops)
    return PassengerEnv(rail_env, vehicle_policy=vehicle_policy, passenger_journeys=PASSENGER_JOURNEYS, obs_builder=PassengerObservationBuilder(),
                        rewards=rewards)


def _run(passenger_env: PassengerEnv, passenger_policy: Policy) -> Tuple[Dict[int, Dict[int, PassengerObservation]], Dict[int, Dict[int, Tuple]],
                                                                         Dict[int, Dict[int, int]], Dict]:
    """
    Drives the passengers of `passenger_env` with `passenger_policy` (`TravelWisePolicyRunner`) until the rail env is
    done: per step, the policy acts on the observations of the previous step. Returns the observations per step (0
    after `reset()`), the mode changes per passenger (step -> `passengers[h].mode`), the non-zero rewards per passenger
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
            mode = passenger_env.passengers[handle].mode
            if mode != previous_modes[handle]:
                mode_changes[handle][step] = mode
            previous_modes[handle] = mode
    return observations_by_step, mode_changes, rewards_by_step, dones


def test_passenger_env_scenario_single_rides():
    """
    Single-ride journeys on the scenario's vehicles - mode changes (step -> `passengers[h].mode`, given with grid cells)
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
    assert {h: p.arrival_time for h, p in passenger_env.passengers.items()} == {0: 13, 1: 6, 2: 13, 3: 18, 4: 17, 5: None}
    assert {h: p.target_entry_point for h, p in passenger_env.passengers.items()} == {0: C_EAST, 1: B_EAST, 2: C_EAST, 3: B_EAST, 4: B_WEST, 5: None}
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


def test_passenger_env_scenario_ignores_invalid_actions():
    """
    The scenario with scripted actions on top of `BoardForTargetPassengerPolicy`:

    - 0: hops off T0 at step 4, at (0, 2) between stations - ignored, as T0 does not stop there: it rides on and
      arrives at C at 13, as without the script.
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

    assert mode_changes[0] == _modes({1: (A, None), 3: ((0, 2), 0), 4: ((0, 3), 0), 5: (B, 0), 11: ((0, 5), 0), 12: ((0, 6), 0), 13: (C, None)})
    assert mode_changes[2] == _modes({1: (B, None), 6: (B, 0), 11: ((0, 5), 0), 12: ((0, 6), 0), 13: (C, None)})
    assert mode_changes[3] == _modes({1: (A, None), 3: ((0, 2), 0), 4: ((0, 3), 0), 5: (B, 0), 6: (B, None)})
    assert mode_changes[5] == _modes({1: (C, None), 15: ((0, 6), 1), 16: ((0, 5), 1), 17: (B, None)})
    assert {h: p.arrival_time for h, p in passenger_env.passengers.items()} == {0: 13, 1: 6, 2: 13, 3: 6, 4: 17, 5: None}
    assert rewards == {0: {13: 2}, 1: {6: -2}, 2: {13: 2}, 3: {3: -1, 6: 14}, 4: {17: 3}, 5: {18: -25}}
    assert dones == {0: True, 1: True, 2: True, 3: True, 4: True, 5: False, '__all__': True}


def test_passenger_env_scenario_passenger_alights_at_a_stop_not_its_next_stop_and_boards_again():
    """
    Passenger 0 (A -> C) rides T0 from A and hops off at B at step 6, while T0 stands still there (steps 5 to 9): B is
    a stop of T0, not of 0's journey. It waits at B with the same next stop C (stop index 0), boards T0 again at step
    7, by its policy, and arrives at C at 13, as without the hop-off.
    """
    passenger_env = _make_scenario_env()
    observations, mode_changes, _, _ = _run(passenger_env, _ScriptedPassengerPolicy({(0, 6): (PassengerAction.HOP_OFF, None)}))

    assert mode_changes[0] == _modes({1: (A, None), 3: ((0, 2), 0), 4: ((0, 3), 0), 5: (B, 0), 6: (B, None), 7: (B, 0), 11: ((0, 5), 0),
                                      12: ((0, 6), 0), 13: (C, None)})
    assert observations[6][0].stop_index == 0
    assert passenger_env.passengers[0].arrival_time == 13


def test_passenger_env_scenario_passengers_neither_board_nor_alight_where_the_vehicle_passes_through():
    """
    The scenario with an auto-pilot not serving intermediate stops: T0 passes through B (`B_EAST`, an entry point of
    its waypoint group `[B_EAST]`) at speed 1 at step 5, without stopping, and is at (0, 5) at step 6. Passengers act by
    `BoardForTargetPassengerPolicy`, except the scripted hop-off of 1 and hop-on of 2:

    - 1 (A -> B) rides T0 from A, does not observe it stopping at B at step 5, and hops off at step 6 - ignored: it
      rides on to T0's final stop C, where it is forced off at 8, without arriving.
    - 2 (B -> C) waits at B, observes no vehicle to board at step 5, and hops on T0 at step 6 - ignored: it never
      boards.
    - 0 (A -> C) rides T0 through B and arrives at C at 8, T0's final stop.
    """
    assert B_EAST in VEHICLE_WAYPOINTS[0][1]
    passenger_env = _make_scenario_env(serve_intermediate_stops=False)
    observations, mode_changes, _, _ = _run(passenger_env, _ScriptedPassengerPolicy({(1, 6): (PassengerAction.HOP_OFF, None),
                                                                                     (2, 6): (PassengerAction.HOP_ON, 0)}))

    # T0 is at B at step 5 only, moving on to (0, 5) at step 6: it does not stand still at B
    assert mode_changes[1] == _modes({1: (A, None), 3: ((0, 2), 0), 4: ((0, 3), 0), 5: (B, 0), 6: ((0, 5), 0), 7: ((0, 6), 0), 8: (C, None)})
    assert not observations[5][1].vehicle_stops_at_next_stop
    assert observations[5][2].vehicles_to_board == {}
    assert mode_changes[2] == _modes({1: (B, None)})
    assert {h: passenger_env.passengers[h].arrival_time for h in (0, 1, 2)} == {0: 8, 1: None, 2: None}


def test_passenger_env_vehicle_stops_at_its_origin_only_in_the_step_it_enters_the_map():
    """
    T0 enters the map at its origin A at step 2, at speed 1, and is at (0, 2) at step 3, also at speed 1 with distance 0.

    - Step 2: T0 stops at A, though moving - it cannot enter the map standing still.
    - Step 3: with (0, 2) taken as T0's origin, as for a route passing its origin again, T0 does not stop there: it
      passes through, coming from A.
    """
    passenger_env = _make_scenario_env()
    passenger_env.reset()
    vehicle = passenger_env.rail_env.agents[0]
    _run_steps(passenger_env, 2)
    assert vehicle.current_entry_point == A_EAST and vehicle.speed_counter.speed == 1
    assert passenger_env.vehicle_stops(0)

    _run_steps(passenger_env, 1)
    assert vehicle.current_entry_point == GraphTransitionMap.grid_entry_point_to_graph_entry_point(0, 2, EAST)
    assert vehicle.speed_counter.speed == 1 and vehicle.speed_counter.distance == 0
    vehicle.initial_entry_point = vehicle.current_entry_point
    assert not passenger_env.vehicle_stops(0)


def _run_steps(passenger_env: PassengerEnv, steps: int):
    """ Steps the passenger env `steps` times, the passengers waiting. """
    for _ in range(steps):
        passenger_env.step({})


@pytest.mark.parametrize("hop_on_step, boards", [(10, True), (11, False)], ids=["standing still", "moving on"])
def test_passenger_env_scenario_passenger_boards_while_the_vehicle_stands_still_not_as_it_moves_on(hop_on_step, boards):
    """
    Passenger 2 (B -> C) waits at B, where T0 stands still (speed 0) at steps 5 to 9, and is at speed 1, still at B, at
    step 10, the step it moves on. 2 waits until it hops on T0:

    - at step 10, having observed T0 standing still at step 9: it boards, though T0 starts moving in the same step, as
      the passengers' actions are applied before the vehicles move; it arrives at C at 13;
    - at step 11, having observed T0 moving at step 10: ignored, it never boards.
    """
    script = {(2, step): (PassengerAction.WAIT, None) for step in range(1, hop_on_step)}
    script[(2, hop_on_step)] = (PassengerAction.HOP_ON, 0)
    passenger_env = _make_scenario_env()
    _, mode_changes, _, _ = _run(passenger_env, _ScriptedPassengerPolicy(script))

    if boards:
        assert mode_changes[2] == _modes({1: (B, None), 10: (B, 0), 11: ((0, 5), 0), 12: ((0, 6), 0), 13: (C, None)})
        assert passenger_env.passengers[2].arrival_time == 13
    else:
        assert mode_changes[2] == _modes({1: (B, None)})
        assert passenger_env.passengers[2].arrival_time is None


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
    assert passenger_env.passengers[2].arrival_time == 13
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


def _make_two_track_station_env(passenger_journeys: Dict[int, PassengerJourney]) -> PassengerEnv:
    """
    A line A -> S -> {B1, B2} -> J -> C -> D (track beyond C, never reached): at the switch S, `MOVE_LEFT` leads to track B2, the other actions to track B1;
    each node is its own resource. One vehicle A -> B -> C, its stop at B the group `[B2, B1]`: both tracks are equally
    far, so its router picks B2, the first of the group. It enters A in step 2 and halts at B2 until its earliest
    departure.
    """
    g = nx.DiGraph()
    all_actions = set(RailEnvActions)
    for u, v in [("A", "S"), ("B1", "J"), ("B2", "J"), ("J", "C"), ("C", "D")]:
        g.add_edge(u, v, actions=all_actions)
    g.add_edge("S", "B1", actions=all_actions - {RailEnvActions.MOVE_LEFT})
    g.add_edge("S", "B2", actions={RailEnvActions.MOVE_LEFT})
    timetable = Timetable(earliest_departures=[[2, 7, None]], latest_arrivals=[[None, 5, 12]], max_episode_steps=15)
    rail_env = GraphRailEnv.from_graph(g, {node: node for node in g.nodes}, {0: [["A"], ["B2", "B1"], ["C"]]},
                                       observation_builder=VehicleObservationBuilder(), timetable_generator=lambda *args, **kwargs: timetable)
    vehicle_policy = ShortestPathPolicy(router=greedy_graph_route_through_waypoints, serve_intermediate_stops=True)
    return PassengerEnv(rail_env, vehicle_policy=vehicle_policy, passenger_journeys=passenger_journeys, obs_builder=PassengerObservationBuilder())


def test_board_for_target_passenger_policy_boards_only_vehicles_planned_to_stop_at_its_next_stop():
    """
    Two passengers at A, on a station B with two tracks B1 and B2, and one vehicle planned to stop at B2:

    - vehicle 0: waypoints `[["A"], ["B2", "B1"], ["C"]]` - its router picks B2, the first of the equally far tracks;
    - passenger 0: waypoints `[["A"], ["B1"]]` - does not board: the vehicle's group at B shares B1 with its next
      stop, but the vehicle stops on the other track. It stays waiting at A and never arrives;
    - passenger 1: waypoints `[["A"], ["B2"]]` - boards at A, hops off at B2 and arrives.
    """
    passenger_env = _make_two_track_station_env({0: _journey("A", "B1", 1, 20), 1: _journey("A", "B2", 1, 20)})
    TravelWisePolicyRunner(passenger_env, BoardForTargetPassengerPolicy()).run()

    assert {h: p.mode for h, p in passenger_env.passengers.items()} == {0: ("A", None), 1: ("B2", None)}
    assert passenger_env.passengers[0].arrival_time is None
    assert passenger_env.passengers[1].arrival_time is not None


class _ArrivalCountingRewards(PassengerRewards):
    """ 1 for arriving, -100 for not arriving by the end of the episode. """

    def step_reward(self, journey: PassengerJourney, transition: PassengerTransitionData, elapsed_steps: int) -> int:
        return 1 if transition.arrival_step is not None else 0

    def end_of_episode_reward(self, journey: PassengerJourney, arrived: bool, max_episode_steps: int) -> int:
        return 0 if arrived else -100


def test_passenger_env_scenario_takes_its_rewards_from_a_passenger_rewards_class():
    """
    The scenario's single rides, with a rewards class passed to `PassengerEnv` that gives 1 for arriving and -100 for
    not arriving: passengers 0-4 arrive and total 1, passenger 5 (C -> A, no vehicle) totals -100.
    """
    total_rewards = TravelWisePolicyRunner(_make_scenario_env(rewards=_ArrivalCountingRewards()), BoardForTargetPassengerPolicy()).run()

    assert total_rewards == {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: -100}
