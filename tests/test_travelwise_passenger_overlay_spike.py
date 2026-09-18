from typing import Dict, List, Tuple

import numpy as np

from flatland.core.grid.grid4 import Grid4TransitionsEnum
from flatland.envs.agent_utils import EnvAgent
from flatland.envs.grid.rail_env_grid import RailEnvTransitions
from flatland.envs.observations import FullEnvObservation
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_env_action import RailEnvActions
from flatland.envs.rail_env_heading_info_wrapper import RailEnvHeadingInfoWrapper
from flatland.envs.rail_env_policies import ShortestPathPolicy
from flatland.envs.rail_env_policy import RailEnvPolicy
from flatland.envs.rail_generators import rail_from_grid_transition_map
from flatland.envs.rail_grid_transition_map import RailGridTransitionMap
from flatland.envs.rail_env_shortest_paths import get_k_shortest_paths
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

# One departure every HEADWAY steps, round-robin over the four legs below (leg 1 is the deliberate
# exception - see LEG_EARLIEST_DEPARTURE[1] below) - regular enough that no two agents ever contend for
# a station at the same time, given each leg only takes STATION_DISTANCE steps.
HEADWAY = 6

# Four one-way legs: two shuttling A<->B, two shuttling B<->C.
LEG_WAYPOINTS: Dict[int, List[List[Waypoint]]] = {
    0: [[Waypoint(STATION_A, Grid4TransitionsEnum.EAST)], [Waypoint(STATION_B, Grid4TransitionsEnum.EAST)]],  # A -> B
    1: [[Waypoint(STATION_B, Grid4TransitionsEnum.EAST)], [Waypoint(STATION_C, Grid4TransitionsEnum.EAST)]],  # B -> C
    2: [[Waypoint(STATION_B, Grid4TransitionsEnum.WEST)], [Waypoint(STATION_A, Grid4TransitionsEnum.WEST)]],  # B -> A
    3: [[Waypoint(STATION_C, Grid4TransitionsEnum.WEST)], [Waypoint(STATION_B, Grid4TransitionsEnum.WEST)]],  # C -> B
}
LEG_EARLIEST_DEPARTURE: Dict[int, int] = {handle: (handle + 1) * HEADWAY for handle in LEG_WAYPOINTS}
# Leg 1 (B->C) is deliberately postponed well past both its own regular headway slot (12) and the
# overlay agent's own arrival at B via leg 0 (step 9, riding leg 0 from A) - still clear of leg 2's own
# use of B (step 18, departing the other way) - so the overlay agent has to wait at B for however long
# it actually takes a train heading to its next stop to show up, not just within one headway period,
# before it can hop on leg 1 and ride it all the way to its own final target, C.
LEG_EARLIEST_DEPARTURE[1] = 20
LEG_TARGET_WAYPOINT: Dict[int, Waypoint] = {handle: leg[-1][0] for handle, leg in LEG_WAYPOINTS.items()}

# The single overlay agent's own route: A->B->C, stopping at B on the way rather than skipping
# straight to C - one ordered list of stops per overlay agent handle.
OVERLAY_STOPS: Dict[int, List[Waypoint]] = {
    0: [Waypoint(STATION_A, Grid4TransitionsEnum.EAST), Waypoint(STATION_B, Grid4TransitionsEnum.EAST), Waypoint(STATION_C, Grid4TransitionsEnum.EAST)],
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


def _line_generator(rail, num_agents, hints, num_resets, np_random) -> Line:
    return Line(agent_waypoints=LEG_WAYPOINTS, agent_speeds=[1.0] * len(LEG_WAYPOINTS))


def _timetable_generator(agents, distance_map, hints, np_random) -> Timetable:
    earliest_departures = [[LEG_EARLIEST_DEPARTURE[handle], None] for handle in range(len(agents))]
    latest_arrivals = [[None, LEG_EARLIEST_DEPARTURE[handle] + STATION_DISTANCE] for handle in range(len(agents))]
    return Timetable(earliest_departures=earliest_departures, latest_arrivals=latest_arrivals,
                     max_episode_steps=max(LEG_EARLIEST_DEPARTURE.values()) + STATION_DISTANCE)


def _always_first_waypoint_from_flexible_groups(waypoint_groups: List[List[Waypoint]]) -> List[List[Waypoint]]:
    """
    Reduces a list of waypoint-alternatives groups to a single, non-flexible path plan: every
    intermediate group is narrowed to its first alternative (arbitrary but stable), while the *final*
    group is passed through in full.
    """
    if len(waypoint_groups) == 0:
        return []
    return [[pp[0]] for pp in waypoint_groups[:-1]] + [waypoint_groups[-1]]


class SetPathPolicy(RailEnvPolicy[RailEnv, RailEnv, RailEnvActions]):
    """
    Ported from flatland-baselines'
    `flatland_baselines.deadlock_avoidance_heuristic.policy.set_path_policy.SetPathPolicy`, adapted to
    this checkout's `EnvAgent` API (`agent.current_entry_point` in place of the pinned baselines
    version's separate `agent.position`/`agent.direction`, `env.rail.apply_action_independent()` in
    place of `env.rail.check_action_on_agent()`) and stripped of the baseline-specific `self.audit`/
    `self.rail_env` debug hooks and the (unused here) segment-plotting debug path.

    Works with `FullEnvObservation` only. Unlike `ShortestPathPolicy`, which paths straight from an
    agent's start to its target ignoring anything in between, `SetPathPolicy` computes and caches each
    agent's full path once - through every intermediate waypoint group in order when
    `use_always_first_strategy` is set, straight start->target otherwise - and then just follows that
    cached path, one cell per step.
    """

    def __init__(self, k_shortest_path_cutoff: int = None, use_always_first_strategy: int = None):
        super().__init__()
        self._set_paths: Dict[int, Tuple[Waypoint, ...]] = {}
        self.k_shortest_path_cutoff = k_shortest_path_cutoff
        self.use_always_first_strategy = use_always_first_strategy

    def _act(self, env: RailEnv, agent: EnvAgent):
        if agent.current_entry_point is None:
            return RailEnvActions.MOVE_FORWARD

        if len(self._set_paths[agent.handle]) == 0:
            return RailEnvActions.DO_NOTHING

        for a in {RailEnvActions.MOVE_FORWARD, RailEnvActions.MOVE_LEFT, RailEnvActions.MOVE_RIGHT}:
            result = env.rail.apply_action_independent(RailEnvActions.from_value(a), agent.current_entry_point)
            if result is not None:
                new_position, new_direction = result
                next_waypoint = self._set_paths[agent.handle][1]
                if new_position == next_waypoint.position and new_direction == next_waypoint.direction:
                    return a
        raise Exception("Invalid state")

    def act_many(self, handles: List[int], observations: List[RailEnv], **kwargs):
        actions = {}
        for handle, env in zip(handles, observations):
            agent = env.agents[handle]
            self._update_agent(agent, env)
            actions[handle] = self._act(env, agent)
        return actions

    def _update_agent(self, agent: EnvAgent, env: RailEnv):
        """ Build `_set_paths`. """
        if agent.state == TrainState.DONE:
            self._set_paths.pop(agent.handle, None)
            return

        if agent.handle not in self._set_paths:
            if self.use_always_first_strategy:
                waypoint_groups = _always_first_waypoint_from_flexible_groups(agent.waypoints)
            else:
                waypoint_groups = _always_first_waypoint_from_flexible_groups([agent.waypoints[0], agent.waypoints[-1]])
            self._set_paths[agent.handle] = self._shortest_path_from_non_flexible_waypoints(waypoint_groups, env.rail)

        if self._set_paths[agent.handle] is None or agent.current_entry_point is None:
            return

        position = agent.current_entry_point[0]
        while len(self._set_paths[agent.handle]) > 0 and self._set_paths[agent.handle][0].position != position:
            self._set_paths[agent.handle] = self._set_paths[agent.handle][1:]
        assert self._set_paths[agent.handle][0].position == position

    def _shortest_path_from_non_flexible_waypoints(self, waypoint_groups: List[List[Waypoint]], rail) -> List[Waypoint]:
        """
        Computes the shortest path built by routing the shortest path between non-flexible waypoints;
        only the target may have flexibility.
        """
        p: List[Waypoint] = []
        for g1, g2 in zip(waypoint_groups, waypoint_groups[1:]):
            assert len(g1) == 1
            p1 = g1[0]
            if len(p) > 0:
                assert p[-1] == p1, (p[-1], p1)

            arrival_directions = {wp.direction for wp in g2}
            target_direction = next(iter(arrival_directions)) if len(arrival_directions) == 1 else None

            path_segment_candidates: List[Tuple[Waypoint]] = get_k_shortest_paths(
                None, p1.position, p1.direction, g2[0].position, rail=rail,
                target_direction=target_direction, cutoff=self.k_shortest_path_cutoff)
            assert len(path_segment_candidates) > 0, f"Not found next path from {p1} to any of {g2}."
            next_path_segment = path_segment_candidates[0]
            assert g2[0].position == next_path_segment[-1].position
            if len(p) > 0:
                p += next_path_segment[1:]
            else:
                p += next_path_segment
        return p


def test_shortest_path_policy_runs_shuttle_trains_exactly_on_timetable():
    """
    Three stations A=(0,1), B=(0,4), C=(0,7) on one straight track, 3 cells apart. A `RailEnv` runs
    four trains, each at max speed 1, one leg each: two shuttle A<->B (legs 0: A->B, 2: B->A), two
    shuttle B<->C (legs 1: B->C, 3: C->B). Their earliest departures are staggered by a fixed headway
    of 6 steps, round-robin over the four legs (leg *h* departs at `(h+1) * HEADWAY`) - except leg 1
    (B->C), whose departure is deliberately postponed to step 20 instead of its regular slot at step
    12 (see `LEG_EARLIEST_DEPARTURE[1]`) - so no two trains ever need station B - the only cell shared
    between the two routes - at the same time.

    A `TravelwiseOverlayEnv` wraps this `RailEnv` (the "underlying env") and layers on a fifth agent -
    the overlay agent, a passenger rather than a train - riding `OVERLAY_STOPS[0]` (A->B->C), sharing
    only the corridor's topology with the underlying env - never its agents, timetable or motion
    checks, and never a real `RailEnv`/`EnvAgent` of its own. `TravelwiseOverlayEnv` implements the
    `Environment` interface as a thin pass-through to the underlying env - `step()` takes an
    `action_dict` for the underlying env's own four legs, exactly `RailEnv.step()`'s own contract - so
    a `ShortestPathPolicy`, driven from the outside exactly as it would drive a plain `RailEnv`, is what
    actually picks each leg's action every step. The overlay agent never moves under its own action: it
    either waits at one of its own stops for a train heading to its next stop, or - once one shows up -
    rides that train's position exactly, one underlying-env agent at a time, boarding and alighting
    automatically. Its `overlay_policy` (`SetPathPolicy`, ported from flatland-baselines) is still a
    required constructor argument, but is not currently exercised - `TravelwiseOverlayEnv` ignores
    overlay agents' own actions entirely while riding mode is in place.

    - A `ShortestPathPolicy` drives every one of the underlying env's four trains from its start
      waypoint to its target waypoint, one cell per step (speed 1, no malfunctions, no other traffic
      in the way).
    - Each underlying train stays off the track until its own earliest departure, appears at its start
      station on exactly that step, advances by exactly one cell per following step, and reaches its
      target - leaving the track immediately - exactly `STATION_DISTANCE` steps after departing,
      matching its timetabled latest arrival exactly.
    - The overlay agent shows up at its first stop, A, at a fixed step (`overlay_earliest_departure`)
      clear of `RailEnv`'s off-map departure timing edge cases, and waits there. Leg 0 (A->B) is the
      first underlying train heading to the overlay agent's next stop (B) to ever occupy A - the
      overlay agent boards it the instant it does, rides its exact position cell by cell, and alights
      onto B the same step that train reaches its own target and is removed from the underlying env.
      It then waits at B considerably longer - 11 steps, from step 9 to step 20 - than it did at A,
      since leg 1 (B->C), the only train ever heading to its next stop, C, is the one deliberately
      postponed above: the overlay agent simply keeps waiting until leg 1 actually shows up, boards it
      the instant it does, and alights onto C, its own final target, the step leg 1 arrives there and
      is removed. Once at C - its own last stop - the overlay agent simply stays there, with
      `overlay_arrival_time`/`overlay_target_entry_point` recorded the same step its position is
      written as C.
    - No train, underlying or overlay, is ever stopped: the underlying env's headway leaves enough
      clearance that its four trains never contend for a cell.
    - The underlying env is wrapped in `RailEnvHeadingInfoWrapper`, so its info dict's `heading` entry
      always names the waypoint (position, direction) each of its four trains is heading to - its
      leg's own target waypoint - throughout the run, regardless of whether that train is currently on
      or off the track. This is exactly what the overlay agent's boarding decision reads to tell which
      underlying train, if any currently sharing its cell, is headed towards its own next stop.
    """
    rail = _make_three_station_rail()
    rail_env = RailEnvHeadingInfoWrapper(RailEnv(width=N_CELLS, height=1,
                                                 rail_generator=rail_from_grid_transition_map(rail),
                                                 line_generator=_line_generator,
                                                 timetable_generator=_timetable_generator,
                                                 number_of_agents=len(LEG_WAYPOINTS),
                                                 obs_builder_object=FullEnvObservation()))
    policy = ShortestPathPolicy()
    overlay = TravelwiseOverlayEnv(rail_env, overlay_policy=SetPathPolicy(use_always_first_strategy=1),
                                   overlay_stops=OVERLAY_STOPS)
    obs, _ = overlay.reset()

    # the heading info wrapper's info dict already carries each leg's target waypoint right after
    # reset(), before any train has moved.
    assert overlay.rail_env_info['heading'] == LEG_TARGET_WAYPOINT

    # verbatim expected timetable for the underlying env: one entry per leg, in handle order 0..3
    # (A->B, B->C, B->A, C->B), departures at HEADWAY=6, 12, 18, 24 and arrivals STATION_DISTANCE=3
    # steps later - except leg 1, postponed to depart (and so arrive) 8 steps later than its regular
    # slot (20/23 instead of 12/15).
    assert _timetable_generator(overlay.rail_env.agents, overlay.rail_env.distance_map, None, None) == Timetable(
        earliest_departures=[[6, None], [20, None], [18, None], [24, None]],
        latest_arrivals=[[None, 9], [None, 23], [None, 21], [None, 27]],
        max_episode_steps=27,
    )
    for handle, agent in enumerate(overlay.rail_env.agents):
        assert agent.earliest_departure == LEG_EARLIEST_DEPARTURE[handle]
        assert agent.latest_arrival == LEG_EARLIEST_DEPARTURE[handle] + STATION_DISTANCE

    # the overlay agent's own route/timetable, computed from OVERLAY_STOPS[0] (A, B, C - not skipping
    # over intermediate B) - real arrival at B/C is no longer on this fixed schedule (see
    # `_update_overlay_agents()`), so only the very first entry, `overlay_earliest_departure`, is still
    # behaviorally meaningful: it gates the step the overlay agent shows up waiting at its first stop, A.
    assert overlay.overlay_earliest_departures == {0: [2, 2 + STATION_DISTANCE, None]}
    assert overlay.overlay_latest_arrivals == {0: [None, 2 + STATION_DISTANCE, 2 + 2 * STATION_DISTANCE]}
    assert overlay.overlay_earliest_departure == {0: 2}
    assert overlay.overlay_latest_arrival == {0: 2 + 2 * STATION_DISTANCE}

    rail_env_position_by_step: Dict[int, Dict[int, Tuple[int, int]]] = {handle: {} for handle in LEG_WAYPOINTS}
    overlay_position_by_step: Dict[int, Tuple[int, int]] = {}
    overlay_mode_by_step: Dict[int, Tuple] = {}
    while not overlay.done:
        actions = policy.act_many(overlay.get_agent_handles(), observations=list(obs.values()))
        obs, _, _, _ = overlay.step(actions)
        step = overlay.rail_env._elapsed_steps
        assert overlay.rail_env_info['heading'] == LEG_TARGET_WAYPOINT
        for handle, agent in enumerate(overlay.rail_env.agents):
            rail_env_position_by_step[handle][step] = agent.current_entry_point[0] if agent.current_entry_point is not None else None
            assert agent.state != TrainState.STOPPED
        overlay_position, riding_handle = overlay.overlay_mode[0]
        overlay_current_entry_point = overlay.overlay_current_entry_point[0]
        overlay_position_by_step[step] = overlay_current_entry_point[0] if overlay_current_entry_point is not None else None
        overlay_mode_by_step[step] = overlay.overlay_mode[0]
        # while riding (mode 3), the ridden underlying agent's own position matches the overlay
        # agent's position exactly, every step - the invariant `overlay_mode[h]`'s riding entry itself
        # is a handle (not a position) relies on.
        if riding_handle is not None:
            assert overlay.rail_env.agents[riding_handle].current_entry_point[0] == overlay_position

    # verbatim expected position trace for the underlying env: for each leg (dict key), its position
    # at every elapsed step from 1 to 27 - `None` off track, `(0, col)` on track, `None` again once
    # removed on arrival.
    assert rail_env_position_by_step == {
        0: {1: None, 2: None, 3: None, 4: None, 5: None, 6: (0, 1), 7: (0, 2), 8: (0, 3), 9: None, 10: None,
            11: None, 12: None, 13: None, 14: None, 15: None, 16: None, 17: None, 18: None, 19: None, 20: None,
            21: None, 22: None, 23: None, 24: None, 25: None, 26: None, 27: None},
        1: {1: None, 2: None, 3: None, 4: None, 5: None, 6: None, 7: None, 8: None, 9: None, 10: None,
            11: None, 12: None, 13: None, 14: None, 15: None, 16: None, 17: None, 18: None, 19: None, 20: (0, 4),
            21: (0, 5), 22: (0, 6), 23: None, 24: None, 25: None, 26: None, 27: None},
        2: {1: None, 2: None, 3: None, 4: None, 5: None, 6: None, 7: None, 8: None, 9: None, 10: None,
            11: None, 12: None, 13: None, 14: None, 15: None, 16: None, 17: None, 18: (0, 4), 19: (0, 3), 20: (0, 2),
            21: None, 22: None, 23: None, 24: None, 25: None, 26: None, 27: None},
        3: {1: None, 2: None, 3: None, 4: None, 5: None, 6: None, 7: None, 8: None, 9: None, 10: None,
            11: None, 12: None, 13: None, 14: None, 15: None, 16: None, 17: None, 18: None, 19: None, 20: None,
            21: None, 22: None, 23: None, 24: (0, 7), 25: (0, 6), 26: (0, 5), 27: None},
    }

    # verbatim expected position trace for the overlay agent: off map before step 2; waiting at A from
    # step 2 (its overlay_earliest_departure) until step 6, when leg 0 (A->B) reaches A and it boards;
    # rides leg 0's exact position for steps 6-8; alights onto B the moment leg 0 arrives and is
    # removed, step 9, then waits there considerably longer - until step 20, when the deliberately
    # postponed leg 1 (B->C) finally reaches B and it boards; rides leg 1's exact position for steps
    # 20-22; alights onto C, its own final target, at step 23 and stays there for the remainder of the
    # run (steps 23-27), since there is no next stop left to wait for.
    assert overlay_position_by_step == {
        1: None, 2: (0, 1), 3: (0, 1), 4: (0, 1), 5: (0, 1), 6: (0, 1), 7: (0, 2), 8: (0, 3), 9: (0, 4),
        10: (0, 4), 11: (0, 4), 12: (0, 4), 13: (0, 4), 14: (0, 4), 15: (0, 4), 16: (0, 4), 17: (0, 4),
        18: (0, 4), 19: (0, 4), 20: (0, 4), 21: (0, 5), 22: (0, 6), 23: (0, 7), 24: (0, 7), 25: (0, 7),
        26: (0, 7), 27: (0, 7),
    }

    # matching mode trace: off map (mode 1) up to step 1; waiting at a stop, not riding (mode 2) while
    # `overlay_mode[0][1]` (the ridden underlying agent's handle) is `None`; riding (mode 3) - that
    # handle, with the ridden agent's own position always equal to the overlay agent's - while it
    # isn't. Boarding/alighting steps are exactly where `overlay_position_by_step` above holds steady
    # across the transition: step 6, waiting -> riding leg 0 (A->B) at A; step 9, riding -> waiting at
    # B; step 20, waiting -> riding leg 1 (B->C) at B, after waiting there far longer than at A; step
    # 23, riding -> waiting at C, permanently, since C is the overlay agent's own last stop.
    assert overlay_mode_by_step == {
        1: (None, None),
        2: ((0, 1), None), 3: ((0, 1), None), 4: ((0, 1), None), 5: ((0, 1), None),
        6: ((0, 1), 0), 7: ((0, 2), 0), 8: ((0, 3), 0),
        9: ((0, 4), None), 10: ((0, 4), None), 11: ((0, 4), None), 12: ((0, 4), None), 13: ((0, 4), None),
        14: ((0, 4), None), 15: ((0, 4), None), 16: ((0, 4), None), 17: ((0, 4), None), 18: ((0, 4), None),
        19: ((0, 4), None),
        20: ((0, 4), 1), 21: ((0, 5), 1), 22: ((0, 6), 1),
        23: ((0, 7), None), 24: ((0, 7), None), 25: ((0, 7), None), 26: ((0, 7), None), 27: ((0, 7), None),
    }

    # the overlay agent's own target entry point - C, arriving from the east, as a plain
    # (position, direction) pair (not a `Waypoint`; `EntryPointT` is generic) - and the step its
    # arrival is recorded on, the same step its position is actually written as C.
    assert overlay.overlay_target_entry_point[0] == (STATION_C, Grid4TransitionsEnum.EAST)
    assert overlay.overlay_arrival_time[0] == 23

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

    # the overlay agent is off map before its own overlay_earliest_departure, then never off map again -
    # unlike an underlying leg, it is never removed from the overlay map once it boards its first train.
    overlay_departure_step = overlay.overlay_earliest_departure[0]
    for step in range(1, overlay_departure_step):
        assert overlay_position_by_step[step] is None, f"overlay: on map before its earliest departure at step {step}"
    for step in range(overlay_departure_step, 28):
        assert overlay_position_by_step[step] is not None, f"overlay: off map at step {step}"

    # the overlay agent only ever rides a leg that is, at that same step, at exactly the overlay
    # agent's own recorded position - cross-checked here against the recorded traces (the loop above
    # already asserts the same against the live agents, every step).
    for step, (overlay_position, riding_handle) in overlay_mode_by_step.items():
        if riding_handle is not None:
            assert rail_env_position_by_step[riding_handle][step] == overlay_position
