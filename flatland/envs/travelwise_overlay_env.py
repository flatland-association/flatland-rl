import copy
from typing import Dict, List, Optional, Tuple

from flatland.core.env import Environment
from flatland.core.policy import Policy
from flatland.envs.agent_utils import EntryPointT, _sanitize_entry_point
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_env_action import RailEnvActions
from flatland.envs.rail_trainrun_data_structures import Waypoint

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


class TravelwiseOverlayEnv(Environment):
    """
    Layers a set of overlay agents over a `RailEnv` ("the underlying env"): the overlay agents share
    the underlying env's rail topology, but are otherwise wholly independent of it - separate agents,
    separate timetable, separate motion checks. Each overlay agent (handle `h`) follows its own fixed,
    ordered list of stops `overlay_stops[h]`.

    Implements the `Environment` interface as a thin pass-through to the underlying env: `step()`
    takes an `action_dict` for the underlying env's own agents (exactly `RailEnv.step()`'s own contract
    - `get_agent_handles()` returns the underlying env's handles, not the overlay agents') and returns
    its `(obs, rewards, dones, info)` verbatim. Overlay agents never take an action of their own -
    `overlay_policy`'s actions are computed (still required to be a `SetPathPolicy` configured to
    respect intermediate stops) but ignored; every step, each overlay agent's position/mode is instead
    derived purely from the just-stepped underlying env's agents and its info dict's `heading` entry
    (see `RailEnvHeadingInfoWrapper`) in `_update_overlay_agents()`, as a side effect alongside the
    pass-through return value - not itself part of the `Environment` contract. An overlay agent has
    exactly 3 modes, tracked in `overlay_mode[h]` as an `OverlayMode` (position, handle) pair:

    1. off map (`(None, None)`) - before showing up at its first stop.
    2. on the overlay map, waiting (`(overlay_position, None)`) - standing at one of its own stops,
       not currently riding a train.
    3. riding (`(overlay_position, underlying agent handle)`, the underlying agent's own position
       always equal to `overlay_position`) - an underlying train is currently at the overlay agent's
       position and heading (per the info dict) for the overlay agent's *next* stop; the overlay agent
       boards it and, every following step, its position tracks that train's - until the train arrives
       (is removed from the underlying env), at which point the overlay agent gets off, back into
       mode 2, now waiting at that next stop (or, at its own last stop, done - see
       `_is_overlay_agent_done()`).
    """

    def __init__(self, rail_env: RailEnv, overlay_policy: Policy, overlay_stops: Dict[int, List[Waypoint]]):
        super().__init__()
        self.rail_env = rail_env
        self.overlay_policy = overlay_policy
        self.overlay_stops = overlay_stops
        self.action_space = self.rail_env.action_space

        self.overlay_earliest_departures: Optional[Dict[int, List[Optional[int]]]] = None
        self.overlay_latest_arrivals: Optional[Dict[int, List[Optional[int]]]] = None
        self.overlay_earliest_departure: Optional[Dict[int, int]] = None
        self.overlay_latest_arrival: Optional[Dict[int, int]] = None

        # OverlayMode per handle - see the class docstring. Its own second entry, once riding (mode 3),
        # already *is* the handle of the underlying agent being ridden - no separate bookkeeping needed.
        self.overlay_mode: Dict[int, OverlayMode] = {}
        # index into overlay_stops[h] of the stop the agent is at/last departed, once it's shown up.
        self._overlay_stop_index: Dict[int, int] = {}

        # per-handle bookkeeping mirroring the entry-point/arrival fields an `EnvAgent` would carry,
        # since an overlay agent is never a real `EnvAgent` - written directly by `_update_overlay_agents()`.
        self.overlay_current_entry_point: Dict[int, Optional[EntryPointT]] = {}
        self.overlay_target_entry_point: Dict[int, Optional[EntryPointT]] = {}
        self.overlay_arrival_time: Dict[int, Optional[int]] = {}

        self._rail_env_obs = None
        self._rail_env_done = False
        self.rail_env_info: Optional[Dict] = None

    def reset(self) -> Tuple[Dict, Dict]:
        self._rail_env_obs, self.rail_env_info = self.rail_env.reset()
        self._rail_env_done = False

        self.overlay_earliest_departures = {}
        self.overlay_latest_arrivals = {}
        self.overlay_earliest_departure = {}
        self.overlay_latest_arrival = {}
        for handle, stops in self.overlay_stops.items():
            # cumulative distance (cells) from this agent's first stop up to each of its stops, at
            # speed 1 - no dwell at an intermediate stop, so its earliest departure and latest arrival
            # are the same step: the step the train reaches it. Purely informational now (see the
            # class docstring) - real movement no longer follows this schedule, except for
            # `overlay_earliest_departure` gating the step the agent shows up at its first stop.
            cumulative_distances = [0]
            for stop, next_stop in zip(stops, stops[1:]):
                cumulative_distances.append(cumulative_distances[-1] + abs(next_stop.position[1] - stop.position[1]))
            arrival_steps = [OVERLAY_EARLIEST_DEPARTURE + distance for distance in cumulative_distances]
            self.overlay_earliest_departures[handle] = arrival_steps[:-1] + [None]
            self.overlay_latest_arrivals[handle] = [None] + arrival_steps[1:]
            self.overlay_earliest_departure[handle] = arrival_steps[0]
            self.overlay_latest_arrival[handle] = arrival_steps[-1]

        self.overlay_mode = {handle: (None, None) for handle in self.overlay_stops}
        self._overlay_stop_index = {handle: 0 for handle in self.overlay_stops}
        self.overlay_current_entry_point = {handle: None for handle in self.overlay_stops}
        self.overlay_target_entry_point = {handle: None for handle in self.overlay_stops}
        self.overlay_arrival_time = {handle: None for handle in self.overlay_stops}

        return self._rail_env_obs, self.rail_env_info

    def get_agent_handles(self) -> List[int]:
        return self.rail_env.get_agent_handles()

    def clone_from(self, env: 'TravelwiseOverlayEnv', **kwargs) -> None:
        self.rail_env.clone_from(env.rail_env)
        self.overlay_policy = env.overlay_policy
        self.overlay_stops = env.overlay_stops
        self.overlay_earliest_departures = copy.deepcopy(env.overlay_earliest_departures)
        self.overlay_latest_arrivals = copy.deepcopy(env.overlay_latest_arrivals)
        self.overlay_earliest_departure = copy.deepcopy(env.overlay_earliest_departure)
        self.overlay_latest_arrival = copy.deepcopy(env.overlay_latest_arrival)
        self.overlay_mode = copy.deepcopy(env.overlay_mode)
        self._overlay_stop_index = copy.deepcopy(env._overlay_stop_index)
        self.overlay_current_entry_point = copy.deepcopy(env.overlay_current_entry_point)
        self.overlay_target_entry_point = copy.deepcopy(env.overlay_target_entry_point)
        self.overlay_arrival_time = copy.deepcopy(env.overlay_arrival_time)
        self._rail_env_obs = copy.deepcopy(env._rail_env_obs)
        self._rail_env_done = env._rail_env_done
        self.rail_env_info = copy.deepcopy(env.rail_env_info)

    def step(self, action_dict: Dict[int, RailEnvActions]) -> Tuple[Dict, Dict, Dict, Dict]:
        """
        Delegates `action_dict` straight to the underlying env's own `step()` and returns its
        `(obs, rewards, dones, info)` verbatim - overlay agents take no actions and aren't part of this
        contract. Also updates every overlay agent's mode/position from the result as a side effect
        (see `_update_overlay_agents()`). Once the underlying env is done, `RailEnv.step()` itself
        refuses further calls - further calls here are a no-op instead, repeating the last observation/
        info alongside empty rewards and an all-done `dones` dict.

        Returns
        -------
        Tuple[Dict, Dict, Dict, Dict]
            The underlying env's own `(obs, rewards, dones, info)`. An overlay agent's derived
            position/mode/arrival bookkeeping is available via `overlay_mode`/`overlay_current_entry_point`/
            `overlay_target_entry_point`/`overlay_arrival_time`, and `_is_overlay_agent_done()`/`done`.
        """
        if self._rail_env_done:
            dones = {handle: True for handle in self.rail_env.get_agent_handles()}
            dones['__all__'] = True
            return self._rail_env_obs, {}, dones, self.rail_env_info

        self._rail_env_obs, rewards, dones, self.rail_env_info = self.rail_env.step(action_dict)
        self._rail_env_done = dones['__all__']
        self._update_overlay_agents()
        return self._rail_env_obs, rewards, dones, self.rail_env_info

    def _is_overlay_agent_done(self, handle: int) -> bool:
        """ Reached its last stop and isn't (or never was) mid-ride away from it. """
        stops = self.overlay_stops[handle]
        return self._overlay_stop_index[handle] == len(stops) - 1 and self.overlay_mode[handle][1] is None

    def _update_overlay_agents(self):
        """
        One mode-transition pass per overlay agent, from the underlying env's freshly-stepped agents
        and its info dict's `heading` entry - see the class docstring for the 3 modes/transitions. Also
        writes the resulting position straight into `overlay_current_entry_point`, and, the step an
        overlay agent's mode-transition reaches its own last stop, its `overlay_target_entry_point`/
        `overlay_arrival_time` - mirroring what `RailEnv.handle_done_state()` would set on a real
        `EnvAgent` reaching its target, computed directly rather than through a second `RailEnv`.
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
                    self._sync_overlay_agent(handle, None, direction)
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
                self._sync_overlay_agent(handle, self.overlay_mode[handle][0], direction)
                if self._is_overlay_agent_done(handle):
                    self.overlay_target_entry_point[handle] = self.overlay_current_entry_point[handle]
                    self.overlay_arrival_time[handle] = self.rail_env._elapsed_steps
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
            self._sync_overlay_agent(handle, self.overlay_mode[handle][0], direction)

    def _sync_overlay_agent(self, handle: int, position: Optional[Tuple[int, int]], direction: int):
        """ Writes the mode-computed position straight into `overlay_current_entry_point`. """
        if position is None:
            self.overlay_current_entry_point[handle] = None
        else:
            self.overlay_current_entry_point[handle] = _sanitize_entry_point((position, direction))

    @property
    def done(self) -> bool:
        return self._rail_env_done and all(self._is_overlay_agent_done(handle) for handle in self.overlay_stops)
