import copy
from typing import Dict, List, Optional, Tuple

from flatland.core.env import Environment
from flatland.core.policy import Policy
from flatland.envs.agent_utils import EntryPointT, _sanitize_entry_point
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_env_action import RailEnvActions
from flatland.envs.rail_trainrun_data_structures import Waypoint

# A passenger's `passenger_mode[h]` tuple - a position (not a full (position, direction) entry
# point), then a handle - holds in each of its 3 modes:
# - off map:                       (None, None)
# - on map, waiting:               (passenger_position, None)
# - riding a vehicle:              (passenger_position, vehicle handle), with the vehicle's own
#   position always equal to passenger_position
PassengerMode = Tuple[Optional[Tuple[int, int]], Optional[int]]


class PassengerEnv(Environment):
    """
    TravelWise passenger env: passengers (handle `h`) travel on the vehicles (trains, ships etc.) of a
    `RailEnv` ("the rail env"), see the TravelWise data model
    (https://github.com/flatland-association/flatland-book/blob/main/resources/travelwise/data_model.md).
    The passengers share the rail env's topology, but are otherwise independent of it - separate
    agents, separate timetable, separate motion checks. Each passenger follows its own fixed journey:
    an ordered list of stops `passenger_journeys[h]`.

    Implements the `Environment` interface as a thin pass-through to the rail env: `step()`
    takes an `action_dict` for the rail env's vehicles (exactly `RailEnv.step()`'s own contract
    - `get_agent_handles()` returns the vehicles' handles, not the passengers') and returns
    its `(obs, rewards, dones, info)` verbatim. Passengers never take an action of their own -
    `passenger_policy`'s actions are computed (still required to be a `SetPathPolicy` configured to
    respect intermediate stops) but ignored; every step, each passenger's position/mode is instead
    derived purely from the just-stepped rail env's vehicles and its info dict's `heading` entry
    (see `RailEnvHeadingInfoWrapper`) in `_update_passengers()`, as a side effect alongside the
    pass-through return value - not itself part of the `Environment` contract. A passenger has
    exactly 3 modes, tracked in `passenger_mode[h]` as a `PassengerMode` (position, handle) pair:

    1. off map (`(None, None)`) - after `reset()`, before the first step shows it up at its first stop.
    2. on map, waiting (`(passenger_position, None)`) - standing at one of its own stops, not
       currently riding a vehicle.
    3. riding (`(passenger_position, vehicle handle)`, the vehicle's own position always equal to
       `passenger_position`) - a vehicle is currently at the passenger's position and heading (per
       the info dict) for the passenger's *next* stop; the passenger boards it and, every following
       step, its position tracks the vehicle's - until the vehicle arrives (is removed from the rail
       env), at which point the passenger gets off, back into
       mode 2, now waiting at that next stop (or, at its own last stop, done - see
       `_is_passenger_done()`).
    """

    def __init__(self, rail_env: RailEnv, passenger_policy: Policy, passenger_journeys: Dict[int, List[Waypoint]]):
        super().__init__()
        self.rail_env = rail_env
        self.passenger_policy = passenger_policy
        self.passenger_journeys = passenger_journeys
        self.action_space = self.rail_env.action_space

        # PassengerMode per handle - see the class docstring. Its own second entry, once riding (mode 3),
        # already *is* the handle of the vehicle being ridden - no separate bookkeeping needed.
        self.passenger_mode: Dict[int, PassengerMode] = {}
        # index into passenger_journeys[h] of the stop the passenger is at/last departed, once it's shown up.
        self._passenger_stop_index: Dict[int, int] = {}

        # per-handle bookkeeping mirroring the entry-point/arrival fields an `EnvAgent` would carry,
        # since a passenger is never a real `EnvAgent` - written directly by `_update_passengers()`.
        self.passenger_current_entry_point: Dict[int, Optional[EntryPointT]] = {}
        self.passenger_target_entry_point: Dict[int, Optional[EntryPointT]] = {}
        self.passenger_arrival_time: Dict[int, Optional[int]] = {}

        self._rail_env_obs = None
        self._rail_env_done = False
        self.rail_env_info: Optional[Dict] = None

    def reset(self) -> Tuple[Dict, Dict]:
        self._rail_env_obs, self.rail_env_info = self.rail_env.reset()
        self._rail_env_done = False

        self.passenger_mode = {handle: (None, None) for handle in self.passenger_journeys}
        self._passenger_stop_index = {handle: 0 for handle in self.passenger_journeys}
        self.passenger_current_entry_point = {handle: None for handle in self.passenger_journeys}
        self.passenger_target_entry_point = {handle: None for handle in self.passenger_journeys}
        self.passenger_arrival_time = {handle: None for handle in self.passenger_journeys}

        return self._rail_env_obs, self.rail_env_info

    def get_agent_handles(self) -> List[int]:
        return self.rail_env.get_agent_handles()

    def clone_from(self, env: 'PassengerEnv', **kwargs) -> None:
        self.rail_env.clone_from(env.rail_env)
        self.passenger_policy = env.passenger_policy
        self.passenger_journeys = env.passenger_journeys
        self.passenger_mode = copy.deepcopy(env.passenger_mode)
        self._passenger_stop_index = copy.deepcopy(env._passenger_stop_index)
        self.passenger_current_entry_point = copy.deepcopy(env.passenger_current_entry_point)
        self.passenger_target_entry_point = copy.deepcopy(env.passenger_target_entry_point)
        self.passenger_arrival_time = copy.deepcopy(env.passenger_arrival_time)
        self._rail_env_obs = copy.deepcopy(env._rail_env_obs)
        self._rail_env_done = env._rail_env_done
        self.rail_env_info = copy.deepcopy(env.rail_env_info)

    def step(self, action_dict: Dict[int, RailEnvActions]) -> Tuple[Dict, Dict, Dict, Dict]:
        """
        Delegates `action_dict` straight to the rail env's own `step()` and returns its
        `(obs, rewards, dones, info)` verbatim - passengers take no actions and aren't part of this
        contract. Also updates every passenger's mode/position from the result as a side effect
        (see `_update_passengers()`). Once the rail env is done, `RailEnv.step()` itself
        refuses further calls - further calls here are a no-op instead, repeating the last observation/
        info alongside empty rewards and an all-done `dones` dict.

        Returns
        -------
        Tuple[Dict, Dict, Dict, Dict]
            The rail env's own `(obs, rewards, dones, info)`. A passenger's derived
            position/mode/arrival bookkeeping is available via `passenger_mode`/`passenger_current_entry_point`/
            `passenger_target_entry_point`/`passenger_arrival_time`, and `_is_passenger_done()`/`done`.
        """
        if self._rail_env_done:
            dones = {handle: True for handle in self.rail_env.get_agent_handles()}
            dones['__all__'] = True
            return self._rail_env_obs, {}, dones, self.rail_env_info

        self._rail_env_obs, rewards, dones, self.rail_env_info = self.rail_env.step(action_dict)
        self._rail_env_done = dones['__all__']
        self._update_passengers()
        return self._rail_env_obs, rewards, dones, self.rail_env_info

    def _is_passenger_done(self, handle: int) -> bool:
        """ Reached its last stop and isn't (or never was) mid-ride away from it. """
        stops = self.passenger_journeys[handle]
        return self._passenger_stop_index[handle] == len(stops) - 1 and self.passenger_mode[handle][1] is None

    def _update_passengers(self):
        """
        One mode-transition pass per passenger, from the rail env's freshly-stepped vehicles
        and its info dict's `heading` entry - see the class docstring for the 3 modes/transitions. Also
        writes the resulting position straight into `passenger_current_entry_point`, and, the step an
        passenger's mode-transition reaches its own last stop, its `passenger_target_entry_point`/
        `passenger_arrival_time` - mirroring what `RailEnv.handle_done_state()` would set on a real
        `EnvAgent` reaching its target, computed directly rather than through a second `RailEnv`.
        """
        heading = self.rail_env_info['heading']
        vehicle_positions = [agent.current_entry_point[0] if agent.current_entry_point is not None else None
                                for agent in self.rail_env.agents]

        for handle, stops in self.passenger_journeys.items():
            passenger_position, riding_handle = self.passenger_mode[handle]
            direction = stops[self._passenger_stop_index[handle]].direction

            if passenger_position is None:
                # mode 1 -> mode 2: show up at the first stop on the first step.
                passenger_position = stops[0].position
                self.passenger_mode[handle] = (passenger_position, None)

            if riding_handle is not None:
                # mode 3: riding - follow the vehicle, or get off once it's gone (arrived/removed).
                riding_position = vehicle_positions[riding_handle]
                if riding_position is None:
                    self._passenger_stop_index[handle] += 1
                    passenger_position = stops[self._passenger_stop_index[handle]].position
                    direction = stops[self._passenger_stop_index[handle]].direction
                    self.passenger_mode[handle] = (passenger_position, None)
                else:
                    self.passenger_mode[handle] = (riding_position, riding_handle)
                    direction = self.rail_env.agents[riding_handle].current_entry_point[1]
                self._sync_passenger(handle, self.passenger_mode[handle][0], direction)
                if self._is_passenger_done(handle):
                    self.passenger_target_entry_point[handle] = self.passenger_current_entry_point[handle]
                    self.passenger_arrival_time[handle] = self.rail_env._elapsed_steps
                continue

            # mode 2: waiting on map for a vehicle heading to the next stop.
            stop_index = self._passenger_stop_index[handle]
            if stop_index + 1 < len(stops):
                next_stop_position = stops[stop_index + 1].position
                for vehicle_handle, position in enumerate(vehicle_positions):
                    if position == passenger_position and heading[vehicle_handle].position == next_stop_position:
                        self.passenger_mode[handle] = (passenger_position, vehicle_handle)
                        direction = self.rail_env.agents[vehicle_handle].current_entry_point[1]
                        break
            self._sync_passenger(handle, self.passenger_mode[handle][0], direction)

    def _sync_passenger(self, handle: int, position: Optional[Tuple[int, int]], direction: int):
        """ Writes the mode-computed position straight into `passenger_current_entry_point`. """
        if position is None:
            self.passenger_current_entry_point[handle] = None
        else:
            self.passenger_current_entry_point[handle] = _sanitize_entry_point((position, direction))

    @property
    def done(self) -> bool:
        return self._rail_env_done and all(self._is_passenger_done(handle) for handle in self.passenger_journeys)
