from dataclasses import dataclass
from types import MappingProxyType
from typing import Dict, List, Mapping, NamedTuple, Optional, Tuple

from flatland.core.env import Environment
from flatland.core.env_observation_builder import ObservationBuilder
from flatland.core.policy import Policy
from flatland.core.resource_map import ResourceT
from flatland.envs.agent_utils import EntryPointT, _sanitize_entry_point
from flatland.envs.rail_env import AbstractRailEnv
from flatland.envs.rail_env_policies import RailEnvStaticData
from flatland.envs.step_utils.states import TrainState
from flatland.envs.travelwise.passenger_env_action import PassengerAction, PassengerActionT

# A passenger's `passenger_mode[h]` tuple - a resource (grid cell, graph node's resource; not a full
# entry point), then a handle - holds in each of its 3 modes:
# - off map:                       (None, None)
# - on map, waiting:               (passenger_resource, None)
# - riding a vehicle:              (passenger_resource, vehicle handle), with the vehicle's own
#   resource always equal to passenger_resource
PassengerMode = Tuple[Optional[ResourceT], Optional[int]]


@dataclass(frozen=True)
class PassengerJourney:
    """
    A passenger's journey, with the same fields and conventions as a vehicle's `EnvAgent`:

    - `waypoints`: the waypoint groups in order, each a list of accepted entry points (e.g. the tracks of a station):
      the origin, the stops where the passenger changes vehicles, and the target. Each pair of consecutive groups is
      one ride. The origin group has exactly one entry point, where the passenger waits.
    - `waypoints_earliest_departure`: per group, the earliest step to leave it - `None` at the target.
    - `waypoints_latest_arrival`: per group, the latest step to reach it - `None` at the origin.
    """
    waypoints: List[List[EntryPointT]]
    waypoints_earliest_departure: List[Optional[int]]
    waypoints_latest_arrival: List[Optional[int]]

    def __post_init__(self):
        assert len(self.waypoints) >= 2, f"at least origin and target, got {self.waypoints}"
        assert len(self.waypoints[0]) == 1, f"the origin must have exactly one entry point, got {self.waypoints[0]}"
        assert len(self.waypoints_earliest_departure) == len(self.waypoints_latest_arrival) == len(self.waypoints)


class PassengerEnvStaticData(NamedTuple):
    """
    `PassengerEnv`'s static data for an episode, immutable: the passengers' journeys by handle and the rail env's static
    data (the vehicles' timetables, the map). Not part of the observations: handed out in the infos (key `"static"`).
    """
    passenger_journeys: Mapping[int, PassengerJourney]
    rail_env: RailEnvStaticData


class PassengerEnv(Environment):
    """
    TravelWise passenger env: passengers (handle `h`) travel on the vehicles (trains, ships etc.) of an
    `AbstractRailEnv` ("the rail env"; a `GraphRailEnv` with a `VehicleObservationBuilder`), see the TravelWise data model
    (https://github.com/flatland-association/flatland-book/blob/main/resources/travelwise/data_model.md).
    The passengers share the rail env's topology, but are otherwise independent of it - separate
    agents, separate timetable, separate motion checks. Each passenger travels its own journey
    `passenger_journeys[h]`, a `PassengerJourney` from an origin to a target, changing vehicles at given stops.

    Implements the `Environment` interface for the passengers: `get_agent_handles()` returns the
    passengers' handles, and `step()` takes an `action_dict` with the passengers' `PassengerActionT` actions. The rail
    env's vehicles are not agents of this env, but part of the passengers' environment: `step()` drives them itself,
    with the actions of `vehicle_policy` (the vehicles' "auto-pilot"). For the vehicles, the env is the runner: it
    keeps the vehicle observations (`vehicle_observations`, built by the rail env's `VehicleObservationBuilder`) and
    infos the rail env returned, and the vehicle policy acts on them in the next step, with the rail env's static data
    in the infos (`infos["static"]`, a `RailEnvStaticData`).

    The passengers decide themselves when to board and when to leave a vehicle, based on what they observe of the
    vehicles: `reset()`/`step()` return the passengers' observations, built by `obs_builder` (e.g.
    `PassengerObservationBuilder`, observing the vehicles' positions and plans in `vehicle_observations`), and infos
    with the env's static data (`infos["static"]`, a `PassengerEnvStaticData`). A controller, `TravelWisePolicyRunner`,
    acts on them with a passenger policy, e.g. `BoardForTargetPassengerPolicy`, and passes the actions to the next
    `step()`:

        total_rewards = TravelWisePolicyRunner(passenger_env, passenger_policy).run()

    Rewards per passenger, in the step they occur (0 otherwise), integers that can be positive (an early arrival) -
    unlike Flatland's `DefaultRewards`, which never reward arriving early (TODO align with them or keep the bonus
    once the TravelWise rewards are settled):

    - arriving: `waypoints_latest_arrival[-1] - arrival time` - positive when early, negative when late;
    - boarding before its earliest departure at the group it waits at: `-(earliest departure - boarding step)` -
      respecting it is the policy's decision, the env does not enforce it;
    - not arrived when the rail env is done: `-max_episode_steps`.

    `dones` holds per passenger whether it has arrived, and `'__all__'`: whether the rail env is done.
    Read the passengers' state from `passenger_mode`/
    `passenger_current_entry_point`/`passenger_target_entry_point`/`passenger_arrival_time`. A passenger
    has exactly 3 modes, tracked in `passenger_mode[h]` as a `PassengerMode` (resource, handle) pair:

    1. off map (`(None, None)`) - after `reset()`, before the first step shows it up at its origin.
    2. on map, waiting (`(passenger_resource, None)`) - at its origin, at its target once arrived, or where a vehicle
       set it down outside its target.
    3. riding (`(passenger_resource, vehicle handle)`, the vehicle's own resource always equal to
       `passenger_resource`).

    The env only applies the passengers' actions if they are physically feasible, and ignores them otherwise; what to
    do is the passenger policy's decision - e.g. whether a vehicle serves the passenger's target, and respecting the
    passenger's earliest departure.

    Boarding (2 -> 3): `HOP_ON` a vehicle, feasible if `can_hop_on()` - the vehicle on the map at the passenger's
    resource. A passenger on a vehicle not stopping at its target is forced off at the vehicle's final stop, without
    arriving.

    Alighting (3 -> 2): `HOP_OFF`, at the vehicle's current entry point; and forced at the vehicle's final stop, where
    Flatland removes it in the step it reaches its target, at the entry point it reached
    (`EnvAgent.target_entry_point`). Alighting at an entry point of its next stop (`next_stop()`) completes the ride:
    at a change, the passenger waits there for the next vehicle; at its target, it has arrived, stays there and its
    further actions are ignored. Alighting elsewhere leaves it waiting there, with the same next stop.

    Limitations:

    - The changes are given, not planned: a passenger set down outside its next stop continues only on a vehicle
      from there that stops at its next stop. No minimum transfer time. TODO plan connections with the query view.
    - A passenger may board a vehicle at its resource even where the vehicle doesn't stop, e.g. passing through.
      TODO board only at a vehicle's stop.
    - A passenger may hop off at any step. TODO hop off only where the vehicle stops.
    - Missed stop: a vehicle serving the passenger may stop at another alternative of the group sharing an entry
      point with the target; then `BoardForTargetPassengerPolicy` stays on board, up to the vehicle's final stop.
      TODO not covered by a test, as every group on the test corridor has one entry point.
    - The vehicles' plans are per-vehicle predictions, optimistic (`UnilateralTimetablePredictor`); there is no query view
      of all vehicles' future stops yet.
    - Stops are entry points (station tracks), not stations.
    - Graph edges count as length 1 (https://github.com/flatland-association/flatland-rl/pull/542).
    """

    def __init__(self, rail_env: AbstractRailEnv, vehicle_policy: Policy, passenger_journeys: Dict[int, PassengerJourney],
                 obs_builder: ObservationBuilder):
        super().__init__()
        self.rail_env = rail_env
        self.vehicle_policy = vehicle_policy
        self.passenger_journeys = passenger_journeys
        self.obs_builder = obs_builder

        # PassengerMode per handle - see the class docstring. Its own second entry, once riding (mode 3),
        # already *is* the handle of the vehicle being ridden - no separate bookkeeping needed.
        self.passenger_mode: Dict[int, PassengerMode] = {}
        # index into passenger_journeys[h].waypoints of the group the passenger is at/last departed, once it's shown up.
        self._passenger_stop_index: Dict[int, int] = {}

        # per-handle bookkeeping mirroring the entry-point/arrival fields an `EnvAgent` would carry,
        # since a passenger is never a real `EnvAgent` - written directly by `_update_passengers()`.
        self.passenger_current_entry_point: Dict[int, Optional[EntryPointT]] = {}
        self.passenger_target_entry_point: Dict[int, Optional[EntryPointT]] = {}
        self.passenger_arrival_time: Dict[int, Optional[int]] = {}

        # the rail env's last vehicle observations and infos, which the vehicle policy acts on in the next step
        self.vehicle_observations: Dict[int, object] = {}
        self._vehicle_infos: Dict = {}
        self._static: Optional[PassengerEnvStaticData] = None
        self._passenger_obs: Dict = {}
        self._rewards: Dict[int, int] = {}
        self._rail_env_done = False

    def reset(self) -> Tuple[Dict, Dict]:
        """ Resets the rail env and the passengers. Returns the passengers' observations and the infos with the static data. """
        self.vehicle_observations, vehicle_infos = self.rail_env.reset()
        rail_env_static = RailEnvStaticData.from_env(self.rail_env)
        self._vehicle_infos = dict(vehicle_infos, static=rail_env_static)
        self._static = PassengerEnvStaticData(passenger_journeys=MappingProxyType(dict(self.passenger_journeys)), rail_env=rail_env_static)
        self._rail_env_done = False

        self.passenger_mode = {handle: (None, None) for handle in self.passenger_journeys}
        self._passenger_stop_index = {handle: 0 for handle in self.passenger_journeys}
        self.passenger_current_entry_point = {handle: None for handle in self.passenger_journeys}
        self.passenger_target_entry_point = {handle: None for handle in self.passenger_journeys}
        self.passenger_arrival_time = {handle: None for handle in self.passenger_journeys}

        self.obs_builder.reset(self)
        self._passenger_obs = self.obs_builder.get_many(self.get_agent_handles())
        return self._passenger_obs, {"static": self._static}

    def get_agent_handles(self) -> List[int]:
        return list(self.passenger_journeys)

    def step(self, action_dict: Dict[int, PassengerActionT]) -> Tuple[Dict, Dict, Dict, Dict]:
        """
        Applies the passengers' actions (`_update_passengers()`) on the state they observed, then steps the rail env
        with `vehicle_policy`'s actions on the vehicle observations and infos kept from the previous step, keeps the
        new ones, moves the passengers with the vehicles (`_move_passengers()`) and builds the passengers'
        observations. So passengers and vehicles decide on the same state. Once the rail env is done,
        `AbstractRailEnv.step()` itself refuses further calls - further calls here are a no-op instead.

        Parameters
        ----------
        action_dict : Dict[int, PassengerActionT]
            Per passenger handle, its action - a missing passenger waits. Invalid actions are ignored.

        Returns
        -------
        Tuple[Dict, Dict, Dict, Dict]
            The passengers' observations, their rewards (see the class docstring), `dones` per passenger (arrived) and
            `'__all__'` (true once the rail env is done, as no passenger can move after that), and the infos with the
            static data.
        """
        self._rewards = {handle: 0 for handle in self.passenger_journeys}
        if not self._rail_env_done:
            self._update_passengers(action_dict)
            vehicle_handles = self.rail_env.get_agent_handles()
            vehicle_actions = self.vehicle_policy.act_many(vehicle_handles, observations=[self.vehicle_observations[handle] for handle in vehicle_handles],
                                                           infos=self._vehicle_infos)
            self.vehicle_observations, _, rail_env_dones, vehicle_infos = self.rail_env.step(vehicle_actions)
            self._vehicle_infos = dict(vehicle_infos, static=self._static.rail_env)
            self._rail_env_done = rail_env_dones['__all__']
            self._move_passengers()
            if self._rail_env_done:
                for handle in self.passenger_journeys:
                    if self.passenger_arrival_time[handle] is None:
                        self._rewards[handle] -= self.rail_env._max_episode_steps
            self._passenger_obs = self.obs_builder.get_many(self.get_agent_handles())
        dones = {handle: self.passenger_arrival_time[handle] is not None for handle in self.passenger_journeys}
        return self._passenger_obs, self._rewards, dict(dones, __all__=self._rail_env_done), {"static": self._static}

    def next_stop(self, passenger_handle: int) -> Optional[List[EntryPointT]]:
        """ The waypoint group the passenger travels to next (a change or its target) - `None` once arrived. """
        waypoints = self.passenger_journeys[passenger_handle].waypoints
        stop_index = self._passenger_stop_index[passenger_handle]
        return waypoints[stop_index + 1] if stop_index + 1 < len(waypoints) else None

    def can_hop_on(self, vehicle_handle: int, passenger_handle: int) -> bool:
        """
        Whether the passenger can hop on the vehicle physically: it waits, and the vehicle is on the map at its
        resource. A passenger who has arrived is done: its actions are ignored.
        """
        passenger_resource, riding_handle = self.passenger_mode[passenger_handle]
        if riding_handle is not None or passenger_resource is None or self.next_stop(passenger_handle) is None:
            return False
        vehicle_entry_point = self.rail_env.agents[vehicle_handle].current_entry_point
        return vehicle_entry_point is not None and self._resource(vehicle_entry_point) == passenger_resource

    def _move_passengers(self):
        """
        Moves the passengers with the rail env's freshly-stepped vehicles, before they observe: they show up at
        their origin on the first step (mode 1 -> mode 2), riding passengers follow their vehicles, and are forced off
        at a vehicle's final stop, where Flatland removes it in the step it reaches its target (mode 3 -> mode 2).
        """
        for handle, journey in self.passenger_journeys.items():
            passenger_resource, riding_handle = self.passenger_mode[handle]
            if passenger_resource is None:
                self._set_passenger(handle, journey.waypoints[0][0], None)
            elif riding_handle is not None:
                vehicle = self.rail_env.agents[riding_handle]
                if vehicle.state == TrainState.DONE:
                    self._alight(handle, vehicle.target_entry_point, self.rail_env._elapsed_steps)
                else:
                    self._set_passenger(handle, vehicle.current_entry_point, riding_handle)

    def _update_passengers(self, actions: Dict[int, PassengerActionT]):
        """
        Applies the passengers' actions, before the rail env steps, on the state they observed - see the class docstring
        for the 3 modes/transitions. Invalid actions are ignored. Also writes the resulting entry point straight into
        `passenger_current_entry_point`, and, the step a passenger arrives at its target (the step being executed), its
        `passenger_target_entry_point`/`passenger_arrival_time` - mirroring what `AbstractRailEnv.handle_done_state()`
        sets on an `EnvAgent` reaching its target.
        """
        for handle in self.passenger_journeys:
            action, vehicle_handle = actions.get(handle, (PassengerAction.WAIT, None))
            riding_handle = self.passenger_mode[handle][1]
            if riding_handle is not None and action == PassengerAction.HOP_OFF:
                self._alight(handle, self.rail_env.agents[riding_handle].current_entry_point, self.rail_env._elapsed_steps + 1)
            elif riding_handle is None and action == PassengerAction.HOP_ON and vehicle_handle is not None and self.can_hop_on(vehicle_handle, handle):
                self._set_passenger(handle, self.rail_env.agents[vehicle_handle].current_entry_point, vehicle_handle)
                earliest_departure = self.passenger_journeys[handle].waypoints_earliest_departure[self._passenger_stop_index[handle]]
                boarding_step = self.rail_env._elapsed_steps + 1
                if earliest_departure is not None and boarding_step < earliest_departure:
                    self._rewards[handle] -= earliest_departure - boarding_step

    def _alight(self, handle: int, entry_point: EntryPointT, step: int):
        """
        Sets the passenger down at `entry_point` in `step`. If `entry_point` is in its next stop, the ride is complete:
        the passenger travels on to the following group; it has arrived if the next stop was its target.
        """
        next_stop = self.next_stop(handle)
        self._set_passenger(handle, entry_point, None)
        if entry_point in next_stop:
            self._passenger_stop_index[handle] += 1
            if self.next_stop(handle) is not None:
                return
            self.passenger_target_entry_point[handle] = self.passenger_current_entry_point[handle]
            self.passenger_arrival_time[handle] = step
            self._rewards[handle] += self.passenger_journeys[handle].waypoints_latest_arrival[-1] - step

    def _resource(self, entry_point: Optional[EntryPointT]) -> Optional[ResourceT]:
        """ The resource (grid cell, graph node's resource) at `entry_point` - `None` off map. """
        return self.rail_env.resource_map.get_resource(entry_point, None)

    def _set_passenger(self, handle: int, entry_point: EntryPointT, riding_handle: Optional[int]):
        """ Writes the passenger's entry point into `passenger_current_entry_point` and its mode into `passenger_mode`. """
        self.passenger_current_entry_point[handle] = _sanitize_entry_point(entry_point)
        self.passenger_mode[handle] = (self._resource(entry_point), riding_handle)

    @property
    def done(self) -> bool:
        """ The rail env is done: no passenger can move after that. """
        return self._rail_env_done
