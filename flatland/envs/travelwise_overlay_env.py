from fractions import Fraction
from typing import Dict, List, Optional, Tuple

from flatland.core.policy import Policy
from flatland.envs.agent_utils import _sanitize_entry_point
from flatland.envs.observations import FullEnvObservation
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_env_action import RailEnvActions
from flatland.envs.rail_generators import rail_from_grid_transition_map
from flatland.envs.rail_trainrun_data_structures import Waypoint
from flatland.envs.timetable_utils import Line, Timetable

# `RailEnv`'s off-map departure timing has documented/undocumented edge cases at earliest_departure 0
# and 1 (e.g. rail_env.py's own "an earliest_departure=0 agent dispatches directly on the very first
# movement action" comment) where the step an agent actually appears on the map doesn't equal
# earliest_departure exactly. 2 is the smallest value clear of both, so every overlay agent's own
# earliest-departure bookkeeping stays clean, even though (see below) it no longer gates real movement.
OVERLAY_EARLIEST_DEPARTURE = 2

# An overlay agent's `overlay_mode[h]` tuple - a position (not a full (position, direction) entry
# point), then a handle - holds in each of its 3 modes:
# - off map:                       (None, None)
# - on the overlay map, waiting:   (overlay_position, None)
# - riding an underlying train:    (overlay_position, underlying agent handle), with the underlying
#   agent's own position always equal to overlay_position
OverlayMode = Tuple[Optional[Tuple[int, int]], Optional[int]]


class TravelwiseOverlayEnv:
    """
    Layers a second env, with its own overlay agents, over a `RailEnv` ("the underlying env"): the
    overlay env shares the underlying env's rail topology (transition map), but is otherwise wholly
    independent of it - separate agents, separate timetable, separate motion checks. Each overlay
    agent (handle `h`) follows its own fixed, ordered list of stops `overlay_stops[h]`.

    Movement is temporarily not action-driven at all - `overlay_policy`'s actions are computed
    (still required to be a `SetPathPolicy` configured to respect intermediate stops) but ignored;
    every step, the overlay env's own agents are instead passed a `DO_NOTHING` and then have their
    real position/mode overridden by `_update_overlay_agents()`, based purely on the underlying env's
    agents and its info dict's `heading` entry (see `RailEnvHeadingInfoWrapper`). An overlay agent has
    exactly 3 modes, tracked in `overlay_mode[h]` as an `OverlayMode` (position, handle) pair:

    1. off map (`(None, None)`) - before showing up at its first stop.
    2. on the overlay map, waiting (`(overlay_position, None)`) - standing at one of its own stops,
       not currently riding a train.
    3. riding (`(overlay_position, underlying agent handle)`, the underlying agent's own position
       always equal to `overlay_position`) - an underlying train is currently at the overlay agent's
       position and heading (per the info dict) for the overlay agent's *next* stop; the overlay agent
       boards it and, every following step, its position tracks that train's - until the train arrives
       (is removed from the underlying env), at which point the overlay agent gets off, back into
       mode 2, now waiting at that next stop.

    Because overlay agents' positions are overridden out of band rather than reached through
    `RailEnv.step()`'s own action-driven motion, the overlay env's per-step invariant assertions (see
    `check_step_pre_post_conditions` on `RailEnv`) would otherwise fail on the very next step - the
    resource/position-consistency checks they run assume every position change came from `step()`
    itself. They're disabled on the overlay env for exactly this reason.
    """

    def __init__(self, rail_env: RailEnv, policy: Policy, overlay_policy: Policy, overlay_stops: Dict[int, List[Waypoint]]):
        self.rail_env = rail_env
        self.policy = policy
        self.overlay_policy = overlay_policy
        self.overlay_stops = overlay_stops

        self.overlay_env: Optional[RailEnv] = None
        self.overlay_earliest_departures: Optional[Dict[int, List[Optional[int]]]] = None
        self.overlay_latest_arrivals: Optional[Dict[int, List[Optional[int]]]] = None
        self.overlay_earliest_departure: Optional[Dict[int, int]] = None
        self.overlay_latest_arrival: Optional[Dict[int, int]] = None

        # OverlayMode per handle - see the class docstring. Its own second entry, once riding (mode 3),
        # already *is* the handle of the underlying agent being ridden - no separate bookkeeping needed.
        self.overlay_mode: Dict[int, OverlayMode] = {}
        # index into overlay_stops[h] of the stop the agent is at/last departed, once it's shown up.
        self._overlay_stop_index: Dict[int, int] = {}

        self._rail_env_obs = None
        self._overlay_obs = None
        self._rail_env_done = False
        self._overlay_env_done = False
        self.rail_env_info: Optional[Dict] = None

    def reset(self):
        self._rail_env_obs, self.rail_env_info = self.rail_env.reset()
        self._rail_env_done = False
        self._overlay_env_done = False

        self.overlay_earliest_departures = {}
        self.overlay_latest_arrivals = {}
        self.overlay_earliest_departure = {}
        self.overlay_latest_arrival = {}
        for handle, stops in self.overlay_stops.items():
            # cumulative distance (cells) from this agent's first stop up to each of its stops, at
            # speed 1 - no dwell at an intermediate stop, so its earliest departure and latest arrival
            # are the same step: the step the train reaches it. Purely informational now (see the
            # class docstring) - real movement no longer follows this schedule.
            cumulative_distances = [0]
            for stop, next_stop in zip(stops, stops[1:]):
                cumulative_distances.append(cumulative_distances[-1] + abs(next_stop.position[1] - stop.position[1]))
            arrival_steps = [OVERLAY_EARLIEST_DEPARTURE + distance for distance in cumulative_distances]
            self.overlay_earliest_departures[handle] = arrival_steps[:-1] + [None]
            self.overlay_latest_arrivals[handle] = [None] + arrival_steps[1:]
            self.overlay_earliest_departure[handle] = arrival_steps[0]
            self.overlay_latest_arrival[handle] = arrival_steps[-1]

        def _line_generator(rail, num_agents, hints, num_resets, np_random) -> Line:
            return Line(agent_waypoints={handle: [[stop] for stop in stops] for handle, stops in self.overlay_stops.items()},
                       agent_speeds=[1.0] * len(self.overlay_stops))

        def _timetable_generator(agents, distance_map, hints, np_random) -> Timetable:
            handles = sorted(self.overlay_stops)
            # `overlay_latest_arrivals`' own values no longer predict when an overlay agent actually
            # reaches a stop - real arrival now depends on when a train going there happens to pass
            # through (see `_update_overlay_agents()`) - so `max_episode_steps` is sized against the
            # underlying env's own episode length instead, purely so the overlay env's episode can't
            # end on its stale schedule before the underlying env (and therefore the ride) is done.
            max_episode_steps = max(max(self.overlay_latest_arrival.values()), self.rail_env._max_episode_steps)
            return Timetable(earliest_departures=[self.overlay_earliest_departures[handle] for handle in handles],
                             latest_arrivals=[self.overlay_latest_arrivals[handle] for handle in handles],
                             max_episode_steps=max_episode_steps)

        self.overlay_env = RailEnv(width=self.rail_env.width, height=self.rail_env.height,
                                   rail_generator=rail_from_grid_transition_map(self.rail_env.rail),
                                   line_generator=_line_generator,
                                   timetable_generator=_timetable_generator,
                                   number_of_agents=len(self.overlay_stops),
                                   obs_builder_object=FullEnvObservation(),
                                   check_step_pre_post_conditions=False)
        self._overlay_obs, _ = self.overlay_env.reset()

        self.overlay_mode = {handle: (None, None) for handle in self.overlay_stops}
        self._overlay_stop_index = {handle: 0 for handle in self.overlay_stops}

        return (self._rail_env_obs, self._overlay_obs), (self.rail_env_info, None)

    def step(self) -> Tuple[Dict, Dict]:
        """
        Advances the underlying env with its own policy, then updates every overlay agent's mode/
        position from the result (see `_update_overlay_agents()`) - the overlay env itself is stepped
        with `DO_NOTHING` for every agent, its own action-driven movement ignored. A finished
        underlying env is left untouched (`RailEnv.step()` itself refuses a call once its episode is
        done); overlay agents simply stop changing once there's nothing new to react to.

        Returns
        -------
        Tuple[Dict, Dict]
            The underlying env's own `dones`, and an overlay `dones` dict (one entry per overlay
            handle, `True` once that agent has ridden through to its last stop, plus `'__all__'`).
        """
        rail_env_dones = {'__all__': self._rail_env_done}
        if not self._rail_env_done:
            actions = self.policy.act_many(self.rail_env.get_agent_handles(), observations=list(self._rail_env_obs.values()))
            self._rail_env_obs, _, rail_env_dones, self.rail_env_info = self.rail_env.step(actions)
            self._rail_env_done = rail_env_dones['__all__']

            # writing an overlay agent's last stop straight onto its `current_entry_point` (mode 2,
            # once `_overlay_stop_index` reaches its final value) is itself exactly the condition
            # `RailEnv`'s own `handle_done_state()` reads as "reached its target" - so the overlay env
            # marks that agent (and, once every overlay agent has, itself) done on its own the very
            # next time its `step()` runs, same as it would for a real, action-driven arrival. Once
            # that's happened, `RailEnv.step()` itself refuses further calls - stop calling it, same
            # guard as `_rail_env_done` above; `_update_overlay_agents()` needs no further calls either,
            # since every overlay agent's own `_is_overlay_agent_done()` is true by construction here.
            if not self._overlay_env_done:
                do_nothing = {handle: RailEnvActions.DO_NOTHING for handle in self.overlay_env.get_agent_handles()}
                self._overlay_obs, _, overlay_env_dones, _ = self.overlay_env.step(do_nothing)
                self._overlay_env_done = overlay_env_dones['__all__']
                self._update_overlay_agents()

        overlay_dones = {handle: self._is_overlay_agent_done(handle) for handle in self.overlay_stops}
        overlay_dones['__all__'] = all(overlay_dones.values())
        return rail_env_dones, overlay_dones

    def _is_overlay_agent_done(self, handle: int) -> bool:
        """ Reached its last stop and isn't (or never was) mid-ride away from it. """
        stops = self.overlay_stops[handle]
        return self._overlay_stop_index[handle] == len(stops) - 1 and self.overlay_mode[handle][1] is None

    def _update_overlay_agents(self):
        """
        One mode-transition pass per overlay agent, from the underlying env's freshly-stepped agents
        and its info dict's `heading` entry - see the class docstring for the 3 modes/transitions.
        Also writes the resulting position straight onto the matching `overlay_env` agent, so it
        reflects the override rather than whatever `DO_NOTHING` left it at.
        """
        heading = self.rail_env_info['heading']
        underlying_positions = [agent.current_entry_point[0] if agent.current_entry_point is not None else None
                                for agent in self.rail_env.agents]

        for handle, stops in self.overlay_stops.items():
            overlay_position, riding_handle = self.overlay_mode[handle]
            direction = stops[self._overlay_stop_index[handle]].direction

            if overlay_position is None:
                # mode 1 -> mode 2: show up at the first stop once its own earliest departure is reached.
                if self.rail_env._elapsed_steps >= self.overlay_earliest_departure[handle]:
                    overlay_position = stops[0].position
                    self.overlay_mode[handle] = (overlay_position, None)
                else:
                    self._sync_overlay_env_agent(handle, None, direction)
                    continue

            if riding_handle is not None:
                # mode 3: riding - follow the train, or get off once it's gone (arrived/removed).
                riding_position = underlying_positions[riding_handle]
                if riding_position is None:
                    self._overlay_stop_index[handle] += 1
                    overlay_position = stops[self._overlay_stop_index[handle]].position
                    direction = stops[self._overlay_stop_index[handle]].direction
                    self.overlay_mode[handle] = (overlay_position, None)
                else:
                    self.overlay_mode[handle] = (riding_position, riding_handle)
                    direction = self.rail_env.agents[riding_handle].current_entry_point[1]
                self._sync_overlay_env_agent(handle, self.overlay_mode[handle][0], direction)
                continue

            # mode 2: waiting on the overlay map for a train heading to the next stop.
            stop_index = self._overlay_stop_index[handle]
            if stop_index + 1 < len(stops):
                next_stop_position = stops[stop_index + 1].position
                for underlying_handle, position in enumerate(underlying_positions):
                    if position == overlay_position and heading[underlying_handle].position == next_stop_position:
                        self.overlay_mode[handle] = (overlay_position, underlying_handle)
                        direction = self.rail_env.agents[underlying_handle].current_entry_point[1]
                        break
            self._sync_overlay_env_agent(handle, self.overlay_mode[handle][0], direction)

    def _sync_overlay_env_agent(self, handle: int, position: Optional[Tuple[int, int]], direction: int):
        """
        Writes the mode-computed position straight onto the matching `overlay_env` agent, standing in
        for the `RailEnv.step()`-driven commit that would otherwise do it. `speed_counter` is pinned to
        a stationary `(0, 0)` whenever on map - `DO_NOTHING` (every overlay agent's only action, see
        `step()`) then always resolves to its own "keep moving mid-cell" branch at that pinned speed,
        leaving `current_entry_point` exactly as just written until the next override - rather than
        `None`, which would desync from `current_entry_point` being non-`None` and misroute the
        underlying `RailEnv.step()` call this override runs *between* into its off-map branches.
        """
        agent = self.overlay_env.agents[handle]
        if position is None:
            agent.current_entry_point = None
            agent.speed_counter.set(None, None)
        else:
            agent.current_entry_point = _sanitize_entry_point((position, direction))
            agent.speed_counter.set(Fraction(0), Fraction(0))

    @property
    def done(self) -> bool:
        return self._rail_env_done and all(self._is_overlay_agent_done(handle) for handle in self.overlay_stops)
