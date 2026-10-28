import copy
from typing import Dict, List, Optional, Tuple

from flatland.core.env import Environment
from flatland.core.policy import Policy
from flatland.core.resource_map import ResourceT
from flatland.envs.agent_utils import EntryPointT, _sanitize_entry_point
from flatland.envs.rail_env import AbstractRailEnv
from flatland.envs.rail_env_action import RailEnvActions

# A passenger's `passenger_mode[h]` tuple - a resource (grid cell, graph node's resource; not a full
# entry point), then a handle - holds in each of its 3 modes:
# - off map:                       (None, None)
# - on map, waiting:               (passenger_resource, None)
# - riding a vehicle:              (passenger_resource, vehicle handle), with the vehicle's own
#   resource always equal to passenger_resource
PassengerMode = Tuple[Optional[ResourceT], Optional[int]]


class PassengerEnv(Environment):
    """
    TravelWise passenger env: passengers (handle `h`) travel on the vehicles (trains, ships etc.) of an
    `AbstractRailEnv` ("the rail env", grid or graph), see the TravelWise data model
    (https://github.com/flatland-association/flatland-book/blob/main/resources/travelwise/data_model.md).
    The passengers share the rail env's topology, but are otherwise independent of it - separate
    agents, separate timetable, separate motion checks. Each passenger follows its own fixed journey:
    an ordered list of stops `passenger_journeys[h]`, each an entry point of the rail env.

    Implements the `Environment` interface as a thin pass-through to the rail env: `step()`
    takes an `action_dict` for the rail env's vehicles (exactly `AbstractRailEnv.step()`'s own contract
    - `get_agent_handles()` returns the vehicles' handles, not the passengers') and returns
    its `(obs, rewards, dones, info)` verbatim. Passengers never take an action of their own -
    `passenger_policy`'s actions are computed (still required to be a `SetPathPolicy` configured to
    respect intermediate stops) but ignored; every step, each passenger's location/mode is instead
    derived purely from the just-stepped rail env's vehicles and its info dict's `heading` entry
    (see `RailEnvHeadingInfoWrapper`) in `_update_passengers()`, as a side effect alongside the
    pass-through return value - not itself part of the `Environment` contract. A passenger has
    exactly 3 modes, tracked in `passenger_mode[h]` as a `PassengerMode` (resource, handle) pair:

    1. off map (`(None, None)`) - after `reset()`, before the first step shows it up at its first stop.
    2. on map, waiting (`(passenger_resource, None)`) - standing at one of its own stops, not
       currently riding a vehicle.
    3. riding (`(passenger_resource, vehicle handle)`, the vehicle's own resource always equal to
       `passenger_resource`) - a vehicle is currently at the passenger's resource and heading (per
       the info dict) for the passenger's *next* stop; the passenger boards it and, every following
       step, its location tracks the vehicle's - until the vehicle arrives (is removed from the rail
       env), at which point the passenger gets off, back into
       mode 2, now waiting at that next stop (or, at its own last stop, done - see
       `_is_passenger_done()`).
    """

    def __init__(self, rail_env: AbstractRailEnv, passenger_policy: Policy, passenger_journeys: Dict[int, List[EntryPointT]]):
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
        contract. Also updates every passenger's mode/location from the result as a side effect
        (see `_update_passengers()`). Once the rail env is done, `AbstractRailEnv.step()` itself
        refuses further calls - further calls here are a no-op instead, repeating the last observation/
        info alongside empty rewards and an all-done `dones` dict.

        Returns
        -------
        Tuple[Dict, Dict, Dict, Dict]
            The rail env's own `(obs, rewards, dones, info)`. A passenger's derived
            location/mode/arrival bookkeeping is available via `passenger_mode`/`passenger_current_entry_point`/
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
        writes the resulting entry point straight into `passenger_current_entry_point`, and, the step a
        passenger's mode-transition reaches its own last stop, its `passenger_target_entry_point`/
        `passenger_arrival_time` - mirroring what `AbstractRailEnv.handle_done_state()` would set on a real
        `EnvAgent` reaching its target, computed directly rather than through a second rail env.
        Locations are compared as resources (`_resource()`), not as entry points: a passenger at a stop
        boards a vehicle at the same cell/node, whatever direction the vehicle enters it from.
        """
        heading = self.rail_env_info['heading']
        vehicle_entry_points = [agent.current_entry_point for agent in self.rail_env.agents]
        vehicle_resources = [self._resource(entry_point) for entry_point in vehicle_entry_points]

        for handle, stops in self.passenger_journeys.items():
            passenger_resource, riding_handle = self.passenger_mode[handle]
            entry_point = stops[self._passenger_stop_index[handle]]

            if passenger_resource is None:
                # mode 1 -> mode 2: show up at the first stop on the first step.
                passenger_resource = self._resource(stops[0])
                self.passenger_mode[handle] = (passenger_resource, None)

            if riding_handle is not None:
                # mode 3: riding - follow the vehicle, or get off once it's gone (arrived/removed).
                if vehicle_entry_points[riding_handle] is None:
                    self._passenger_stop_index[handle] += 1
                    entry_point = stops[self._passenger_stop_index[handle]]
                    self.passenger_mode[handle] = (self._resource(entry_point), None)
                else:
                    entry_point = vehicle_entry_points[riding_handle]
                    self.passenger_mode[handle] = (vehicle_resources[riding_handle], riding_handle)
                self._sync_passenger(handle, entry_point)
                if self._is_passenger_done(handle):
                    self.passenger_target_entry_point[handle] = self.passenger_current_entry_point[handle]
                    self.passenger_arrival_time[handle] = self.rail_env._elapsed_steps
                continue

            # mode 2: waiting on map for a vehicle heading to the next stop.
            stop_index = self._passenger_stop_index[handle]
            if stop_index + 1 < len(stops):
                next_stop_resource = self._resource(stops[stop_index + 1])
                for vehicle_handle, vehicle_resource in enumerate(vehicle_resources):
                    if vehicle_resource == passenger_resource and self._resource(heading[vehicle_handle]) == next_stop_resource:
                        self.passenger_mode[handle] = (passenger_resource, vehicle_handle)
                        entry_point = vehicle_entry_points[vehicle_handle]
                        break
            self._sync_passenger(handle, entry_point)

    def _resource(self, entry_point: Optional[EntryPointT]) -> Optional[ResourceT]:
        """ The resource (grid cell, graph node's resource) at `entry_point` - `None` off map. """
        return self.rail_env.resource_map.get_resource(entry_point, None)

    def _sync_passenger(self, handle: int, entry_point: EntryPointT):
        """ Writes the mode-computed entry point straight into `passenger_current_entry_point`. """
        self.passenger_current_entry_point[handle] = _sanitize_entry_point(entry_point)

    @property
    def done(self) -> bool:
        return self._rail_env_done and all(self._is_passenger_done(handle) for handle in self.passenger_journeys)
