from typing import Dict, List, Tuple

import numpy as np

from flatland.core.grid.grid4 import Grid4TransitionsEnum
from flatland.envs.grid.rail_env_grid import RailEnvTransitions
from flatland.envs.observations import FullEnvObservation
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_env_heading_info_wrapper import RailEnvHeadingInfoWrapper
from flatland.envs.rail_env_policies import ShortestPathPolicy
from flatland.envs.rail_generators import rail_from_grid_transition_map
from flatland.envs.rail_grid_transition_map import RailGridTransitionMap
from flatland.envs.rail_trainrun_data_structures import Waypoint
from flatland.envs.step_utils.states import TrainState
from flatland.envs.timetable_utils import Line, Timetable
from flatland.envs.travelwise_overlay_env import TravelwiseOverlayEnv

# A single straight corridor with three stations A, B, C, evenly spaced 3 cells apart, capped by dead
# ends well beyond either station so no agent ever reaches an end of the corridor.
STATION_A = (0, 1)
STATION_B = (0, 4)
STATION_C = (0, 7)
STATION_DISTANCE = 3
N_CELLS = 9

# One departure every HEADWAY steps, round-robin over the four legs below - regular enough that no two
# agents ever contend for a station at the same time, given each leg only takes STATION_DISTANCE steps.
HEADWAY = 6

# Four one-way legs: two shuttling A<->B, two shuttling B<->C.
LEG_WAYPOINTS: Dict[int, List[List[Waypoint]]] = {
    0: [[Waypoint(STATION_A, Grid4TransitionsEnum.EAST)], [Waypoint(STATION_B, Grid4TransitionsEnum.EAST)]],  # A -> B
    1: [[Waypoint(STATION_B, Grid4TransitionsEnum.EAST)], [Waypoint(STATION_C, Grid4TransitionsEnum.EAST)]],  # B -> C
    2: [[Waypoint(STATION_B, Grid4TransitionsEnum.WEST)], [Waypoint(STATION_A, Grid4TransitionsEnum.WEST)]],  # B -> A
    3: [[Waypoint(STATION_C, Grid4TransitionsEnum.WEST)], [Waypoint(STATION_B, Grid4TransitionsEnum.WEST)]],  # C -> B
}
LEG_EARLIEST_DEPARTURE: Dict[int, int] = {handle: (handle + 1) * HEADWAY for handle in LEG_WAYPOINTS}
LEG_TARGET_WAYPOINT: Dict[int, Waypoint] = {handle: leg[-1][0] for handle, leg in LEG_WAYPOINTS.items()}


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


def _line_generator(rail, num_agents, hints, num_resets, np_random) -> Line:
    return Line(agent_waypoints=LEG_WAYPOINTS, agent_speeds=[1.0] * len(LEG_WAYPOINTS))


def _timetable_generator(agents, distance_map, hints, np_random) -> Timetable:
    earliest_departures = [[LEG_EARLIEST_DEPARTURE[handle], None] for handle in range(len(agents))]
    latest_arrivals = [[None, LEG_EARLIEST_DEPARTURE[handle] + STATION_DISTANCE] for handle in range(len(agents))]
    return Timetable(earliest_departures=earliest_departures, latest_arrivals=latest_arrivals,
                     max_episode_steps=max(LEG_EARLIEST_DEPARTURE.values()) + STATION_DISTANCE)


def test_shortest_path_policy_runs_shuttle_trains_exactly_on_timetable():
    """
    Three stations A=(0,1), B=(0,4), C=(0,7) on one straight track, 3 cells apart. A `RailEnv` runs
    four trains, each at max speed 1, one leg each: two shuttle A<->B (legs 0: A->B, 2: B->A), two
    shuttle B<->C (legs 1: B->C, 3: C->B). Their earliest departures are staggered by a fixed headway
    of 6 steps, round-robin over the four legs (leg *h* departs at `(h+1) * HEADWAY`), so no two trains
    ever need station B - the only cell shared between the two routes - at the same time.

    A `TravelwiseOverlayEnv` wraps this `RailEnv` (the "underlying env") together with the
    `ShortestPathPolicy` that drives its four legs, and layers on a fifth, wholly independent train:
    the overlay agent, running the full A->C span on its own separate `RailEnv` and its own
    `ShortestPathPolicy`, sharing only the corridor's topology with the underlying env - never its
    agents, timetable or motion checks.

    - A `ShortestPathPolicy` alone drives every train, underlying and overlay, from its start waypoint
      to its target waypoint, one cell per step (speed 1, no malfunctions, no other traffic in the
      way).
    - Each underlying train stays off the track until its own earliest departure, appears at its start
      station on exactly that step, advances by exactly one cell per following step, and reaches its
      target - leaving the track immediately - exactly `STATION_DISTANCE` steps after departing,
      matching its timetabled latest arrival exactly.
    - The overlay train follows the same pattern over the full A->C span (twice `STATION_DISTANCE`),
      on its own earliest departure/latest arrival, entirely independent of the underlying env's own
      schedule.
    - No train, underlying or overlay, is ever stopped: the underlying env's headway leaves enough
      clearance that its four trains never contend for a cell, and the overlay train runs alone on its
      own separate env.
    - The underlying env is wrapped in `RailEnvHeadingInfoWrapper`, so its info dict's `heading` entry
      always names the waypoint (position, direction) each of its four trains is heading to - its
      leg's own target waypoint - throughout the run, regardless of whether that train is currently on
      or off the track.
    """
    rail = _make_three_station_rail()
    rail_env = RailEnvHeadingInfoWrapper(RailEnv(width=N_CELLS, height=1,
                                                 rail_generator=rail_from_grid_transition_map(rail),
                                                 line_generator=_line_generator,
                                                 timetable_generator=_timetable_generator,
                                                 number_of_agents=len(LEG_WAYPOINTS),
                                                 obs_builder_object=FullEnvObservation()))
    overlay = TravelwiseOverlayEnv(rail_env, ShortestPathPolicy())
    overlay.reset()

    # the heading info wrapper's info dict already carries each leg's target waypoint right after
    # reset(), before any train has moved.
    assert overlay.rail_env_info['heading'] == LEG_TARGET_WAYPOINT

    # verbatim expected timetable for the underlying env: one entry per leg, in handle order 0..3
    # (A->B, B->C, B->A, C->B), departures at HEADWAY=6, 12, 18, 24 and arrivals STATION_DISTANCE=3
    # steps later.
    assert _timetable_generator(overlay.rail_env.agents, overlay.rail_env.distance_map, None, None) == Timetable(
        earliest_departures=[[6, None], [12, None], [18, None], [24, None]],
        latest_arrivals=[[None, 9], [None, 15], [None, 21], [None, 27]],
        max_episode_steps=27,
    )
    for handle, agent in enumerate(overlay.rail_env.agents):
        assert agent.earliest_departure == LEG_EARLIEST_DEPARTURE[handle]
        assert agent.latest_arrival == LEG_EARLIEST_DEPARTURE[handle] + STATION_DISTANCE

    # the overlay agent's own route/timetable: derived to span the underlying env's full corridor
    # (A->C), on a fixed earliest departure clear of RailEnv's off-map departure timing edge cases.
    assert overlay.overlay_start == STATION_A
    assert overlay.overlay_target == STATION_C
    assert overlay.overlay_earliest_departure == 2
    assert overlay.overlay_latest_arrival == 2 + 2 * STATION_DISTANCE
    overlay_agent = overlay.overlay_env.agents[0]
    assert overlay_agent.earliest_departure == 2
    assert overlay_agent.latest_arrival == 2 + 2 * STATION_DISTANCE

    rail_env_position_by_step: Dict[int, Dict[int, Tuple[int, int]]] = {handle: {} for handle in LEG_WAYPOINTS}
    overlay_position_by_step: Dict[int, Tuple[int, int]] = {}
    while not overlay.done:
        overlay.step()
        step = overlay.rail_env._elapsed_steps
        assert overlay.rail_env_info['heading'] == LEG_TARGET_WAYPOINT
        for handle, agent in enumerate(overlay.rail_env.agents):
            rail_env_position_by_step[handle][step] = agent.current_entry_point[0] if agent.current_entry_point is not None else None
            assert agent.state != TrainState.STOPPED
        overlay_position_by_step[step] = overlay_agent.current_entry_point[0] if overlay_agent.current_entry_point is not None else None
        assert overlay_agent.state != TrainState.STOPPED

    # verbatim expected position trace for the underlying env: for each leg (dict key), its position
    # at every elapsed step from 1 to 27 - `None` off track, `(0, col)` on track, `None` again once
    # removed on arrival.
    assert rail_env_position_by_step == {
        0: {1: None, 2: None, 3: None, 4: None, 5: None, 6: (0, 1), 7: (0, 2), 8: (0, 3), 9: None, 10: None,
            11: None, 12: None, 13: None, 14: None, 15: None, 16: None, 17: None, 18: None, 19: None, 20: None,
            21: None, 22: None, 23: None, 24: None, 25: None, 26: None, 27: None},
        1: {1: None, 2: None, 3: None, 4: None, 5: None, 6: None, 7: None, 8: None, 9: None, 10: None,
            11: None, 12: (0, 4), 13: (0, 5), 14: (0, 6), 15: None, 16: None, 17: None, 18: None, 19: None, 20: None,
            21: None, 22: None, 23: None, 24: None, 25: None, 26: None, 27: None},
        2: {1: None, 2: None, 3: None, 4: None, 5: None, 6: None, 7: None, 8: None, 9: None, 10: None,
            11: None, 12: None, 13: None, 14: None, 15: None, 16: None, 17: None, 18: (0, 4), 19: (0, 3), 20: (0, 2),
            21: None, 22: None, 23: None, 24: None, 25: None, 26: None, 27: None},
        3: {1: None, 2: None, 3: None, 4: None, 5: None, 6: None, 7: None, 8: None, 9: None, 10: None,
            11: None, 12: None, 13: None, 14: None, 15: None, 16: None, 17: None, 18: None, 19: None, 20: None,
            21: None, 22: None, 23: None, 24: (0, 7), 25: (0, 6), 26: (0, 5), 27: None},
    }

    # verbatim expected position trace for the overlay agent (A->C, departure 2, arrival 8): on track
    # from step 2 to 7, `None` before, at, and after that window.
    assert overlay_position_by_step == {
        1: None, 2: (0, 1), 3: (0, 2), 4: (0, 3), 5: (0, 4), 6: (0, 5), 7: (0, 6), 8: None, 9: None, 10: None,
        11: None, 12: None, 13: None, 14: None, 15: None, 16: None, 17: None, 18: None, 19: None, 20: None,
        21: None, 22: None, 23: None, 24: None, 25: None, 26: None, 27: None,
    }

    for handle, leg in LEG_WAYPOINTS.items():
        start_position = leg[0][0].position
        target_position = leg[-1][0].position
        step_towards_target = 1 if target_position[1] > start_position[1] else -1
        departure_step = LEG_EARLIEST_DEPARTURE[handle]
        arrival_step = departure_step + STATION_DISTANCE

        for step in range(1, departure_step):
            assert rail_env_position_by_step[handle][step] is None, f"leg {handle}: on track before its earliest departure at step {step}"
        for offset in range(STATION_DISTANCE):
            step = departure_step + offset
            expected_position = (start_position[0], start_position[1] + offset * step_towards_target)
            assert rail_env_position_by_step[handle][step] == expected_position, f"leg {handle} at step {step}"
        assert rail_env_position_by_step[handle][arrival_step] is None, f"leg {handle}: still on track at its latest arrival step {arrival_step}"

    # same shape of check for the overlay agent's A->C leg.
    overlay_departure_step = overlay.overlay_earliest_departure
    overlay_arrival_step = overlay.overlay_latest_arrival
    for step in range(1, overlay_departure_step):
        assert overlay_position_by_step[step] is None, f"overlay: on track before its earliest departure at step {step}"
    for offset in range(2 * STATION_DISTANCE):
        step = overlay_departure_step + offset
        expected_position = (STATION_A[0], STATION_A[1] + offset)
        assert overlay_position_by_step[step] == expected_position, f"overlay at step {step}"
    assert overlay_position_by_step[overlay_arrival_step] is None, f"overlay: still on track at its latest arrival step {overlay_arrival_step}"
