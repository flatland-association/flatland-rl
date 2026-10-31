from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from flatland.core.env import Environment
from flatland.core.policy import Policy
from flatland.core.resource_map import ResourceT
from flatland.envs.agent_utils import EntryPointT, _sanitize_entry_point
from flatland.envs.rail_env import AbstractRailEnv
from flatland.envs.rail_env_policies import RailEnvStaticData
from flatland.envs.step_utils.states import TrainState
from flatland.envs.travelwise.timetable_predictor import ProjectedEntryPoint, TimetablePredictor, UnilateralTimetablePredictor

# A passenger's `passenger_mode[h]` tuple - a resource (grid cell, graph node's resource; not a full
# entry point), then a handle - holds in each of its 3 modes:
# - off map:                       (None, None)
# - on map, waiting:               (passenger_resource, None)
# - riding a vehicle:              (passenger_resource, vehicle handle), with the vehicle's own
#   resource always equal to passenger_resource
PassengerMode = Tuple[Optional[ResourceT], Optional[int]]


@dataclass
class PassengerJourney:
    """
    A passenger's journey, with the same fields and conventions as a vehicle's `EnvAgent`:

    - `waypoints`: the waypoint groups in order, each a list of accepted entry points (e.g. the tracks of a station).
      The origin group has exactly one entry point, where the passenger waits.
    - `waypoints_earliest_departure`: per group, the earliest step to leave it - `None` at the target.
    - `waypoints_latest_arrival`: per group, the latest step to reach it - `None` at the origin.

    Single ride for now: exactly two groups, origin and target.
    """
    waypoints: List[List[EntryPointT]]
    waypoints_earliest_departure: List[Optional[int]]
    waypoints_latest_arrival: List[Optional[int]]

    def __post_init__(self):
        assert len(self.waypoints) == 2, f"single ride only: origin and target, got {self.waypoints}"
        assert len(self.waypoints[0]) == 1, f"the origin must have exactly one entry point, got {self.waypoints[0]}"
        assert len(self.waypoints_earliest_departure) == len(self.waypoints_latest_arrival) == len(self.waypoints)


class PassengerEnv(Environment):
    """
    TravelWise passenger env: passengers (handle `h`) travel on the vehicles (trains, ships etc.) of an
    `AbstractRailEnv` ("the rail env"; a `GraphRailEnv` with the default `vehicle_predictor`), see the TravelWise data model
    (https://github.com/flatland-association/flatland-book/blob/main/resources/travelwise/data_model.md).
    The passengers share the rail env's topology, but are otherwise independent of it - separate
    agents, separate timetable, separate motion checks. Each passenger travels its own journey
    `passenger_journeys[h]`, a `PassengerJourney` from an origin to a target on a single vehicle.

    Implements the `Environment` interface for the passengers: `get_agent_handles()` returns the
    passengers' handles, and `step()` takes an `action_dict` for the passengers. The rail env's
    vehicles are not agents of this env: `step()` drives them itself, with the actions of
    `vehicle_policy` (the vehicles' "auto-pilot") on the rail env's observations, with the rail env's static data
    (`RailEnvStaticData`) in its infos (`infos["static"]`), before it updates the passengers. After each rail env step,
    `vehicle_predictor` predicts each vehicle's remaining route and its stops (`vehicle_predictions`).

    The passenger side of the interface is formal for now: passengers have no actions (`action_dict` is
    ignored, as are `passenger_policy`'s actions), no observations, no rewards and no infos, so
    `reset()`/`step()` return empty dicts and `dones` holds only `'__all__'`. Every step, each passenger's
    location/mode is derived from the just-stepped rail env's vehicles and their predictions in
    `_update_passengers()`; read it from `passenger_mode`/
    `passenger_current_entry_point`/`passenger_target_entry_point`/`passenger_arrival_time`. A passenger
    has exactly 3 modes, tracked in `passenger_mode[h]` as a `PassengerMode` (resource, handle) pair:

    1. off map (`(None, None)`) - after `reset()`, before the first step shows it up at its origin.
    2. on map, waiting (`(passenger_resource, None)`) - at its origin, at its target once arrived, or where a vehicle
       set it down outside its target.
    3. riding (`(passenger_resource, vehicle handle)`, the vehicle's own resource always equal to
       `passenger_resource`).

    Boarding (2 -> 3), automatic: a passenger waiting at its origin, from its earliest departure on, boards a vehicle
    on the map at its resource that serves it (`serves()`). With several, it boards first one whose predicted route
    stops at an entry point of its target, then the one predicted to stop there soonest.

    Alighting (3 -> 2), automatic: in the step the vehicle stops at an entry point of the passenger's target
    (`stops_at_next_stop()`); and at the vehicle's final stop, where Flatland removes it in the step it reaches its
    target, at the entry point it reached (`EnvAgent.target_entry_point`). The passenger has arrived if it alights at
    an entry point of its target; then it stays there.

    Limitations:

    - Single ride: no change of vehicles. A passenger set down outside its target stays there and doesn't arrive.
    - A passenger boards a vehicle at its resource even where the vehicle doesn't stop, e.g. passing through.
      TODO board only at a vehicle's stop.
    - Missed stop: a vehicle serving the passenger may stop at another alternative of the group sharing an entry
      point with the target; then the passenger stays on board, up to its final stop. TODO not covered by a test, as
      every group on the test corridor has one entry point.
    """

    def __init__(self, rail_env: AbstractRailEnv, vehicle_policy: Policy, passenger_policy: Policy,
                 passenger_journeys: Dict[int, PassengerJourney], vehicle_predictor: Optional[TimetablePredictor] = None):
        super().__init__()
        self.rail_env = rail_env
        self.vehicle_policy = vehicle_policy
        self.passenger_policy = passenger_policy
        self.passenger_journeys = passenger_journeys
        self.vehicle_predictor = vehicle_predictor if vehicle_predictor is not None else UnilateralTimetablePredictor()
        # per vehicle handle, its predicted remaining route, updated after each rail env step
        self.vehicle_predictions: Dict[int, List[ProjectedEntryPoint]] = {}

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

        self._rail_env_obs = None
        self._rail_env_done = False
        self.rail_env_info: Optional[Dict] = None
        # the rail env's static data, added to the infos the vehicle policy gets
        self._rail_env_static: Optional[RailEnvStaticData] = None

    def reset(self) -> Tuple[Dict, Dict]:
        """ Resets the rail env and the passengers. Returns empty passenger observations and infos. """
        self._rail_env_obs, self.rail_env_info = self.rail_env.reset()
        self._rail_env_static = RailEnvStaticData.from_env(self.rail_env)
        self._rail_env_done = False
        self.vehicle_predictor.reset(self.rail_env)
        self.vehicle_predictions = self.vehicle_predictor.get()

        self.passenger_mode = {handle: (None, None) for handle in self.passenger_journeys}
        self._passenger_stop_index = {handle: 0 for handle in self.passenger_journeys}
        self.passenger_current_entry_point = {handle: None for handle in self.passenger_journeys}
        self.passenger_target_entry_point = {handle: None for handle in self.passenger_journeys}
        self.passenger_arrival_time = {handle: None for handle in self.passenger_journeys}

        return {}, {}

    def get_agent_handles(self) -> List[int]:
        return list(self.passenger_journeys)

    def step(self, action_dict: Dict[int, Any]) -> Tuple[Dict, Dict, Dict, Dict]:
        """
        Steps the rail env with `vehicle_policy`'s actions for its vehicles, then updates every
        passenger's mode/location (see `_update_passengers()`). `action_dict` holds the passengers'
        actions, which are ignored for now. Once the rail env is done, `AbstractRailEnv.step()` itself
        refuses further calls - further calls here are a no-op instead.

        Returns
        -------
        Tuple[Dict, Dict, Dict, Dict]
            Empty passenger observations, rewards and infos, and `dones` with only `'__all__'`: true
            once the rail env is done, as no passenger can move after that.
        """
        if not self._rail_env_done:
            vehicle_actions = self.vehicle_policy.act_many(self.rail_env.get_agent_handles(), observations=list(self._rail_env_obs.values()),
                                                           infos=dict(self.rail_env_info, static=self._rail_env_static))
            self._rail_env_obs, _, rail_env_dones, self.rail_env_info = self.rail_env.step(vehicle_actions)
            self._rail_env_done = rail_env_dones['__all__']
            self.vehicle_predictions = self.vehicle_predictor.get()
            self._update_passengers()
        return {}, {}, {'__all__': self._rail_env_done}, {}

    def serves(self, vehicle_handle: int, passenger_handle: int) -> bool:
        """
        Whether the vehicle serves the passenger: one of the vehicle's future stop groups - after the group it stands
        on or last left, as predicted - shares an entry point with the passenger's next stop (its target). The vehicle
        stops at one entry point of such a group, chosen by its policy, possibly one outside the passenger's next stop.
        """
        next_stop = self._next_stop(passenger_handle)
        return next_stop is not None and len(self._future_stops_sharing(vehicle_handle, next_stop)) > 0

    def stops_at_next_stop(self, vehicle_handle: int, passenger_handle: int) -> bool:
        """ Whether the vehicle is at the entry point it stops at for one of its groups, and it is in the passenger's next stop. """
        next_stop = self._next_stop(passenger_handle)
        predictions = self.vehicle_predictions.get(vehicle_handle, [])
        return next_stop is not None and len(predictions) > 0 and predictions[0].stop is not None and predictions[0].entry_point in next_stop

    def _next_stop(self, passenger_handle: int) -> Optional[List[EntryPointT]]:
        """ The waypoint group the passenger travels to next - `None` once arrived. """
        waypoints = self.passenger_journeys[passenger_handle].waypoints
        stop_index = self._passenger_stop_index[passenger_handle]
        return waypoints[stop_index + 1] if stop_index + 1 < len(waypoints) else None

    def _future_stops_sharing(self, vehicle_handle: int, next_stop: List[EntryPointT]) -> List[ProjectedEntryPoint]:
        """ The vehicle's predicted future stops whose group shares an entry point with `next_stop`. """
        vehicle_waypoints = self.rail_env.agents[vehicle_handle].waypoints
        return [prediction for prediction in self.vehicle_predictions.get(vehicle_handle, [])[1:]
                if prediction.stop is not None and not set(next_stop).isdisjoint(vehicle_waypoints[prediction.stop])]

    def _update_passengers(self):
        """
        One mode-transition pass per passenger, from the rail env's freshly-stepped vehicles
        and their predictions (`vehicle_predictions`) - see the class docstring for the 3 modes/transitions. Also
        writes the resulting entry point straight into `passenger_current_entry_point`, and, the step a
        passenger arrives at its target, its `passenger_target_entry_point`/`passenger_arrival_time` - mirroring what
        `AbstractRailEnv.handle_done_state()` sets on an `EnvAgent` reaching its target.
        Locations are compared as resources (`_resource()`), not as entry points: a passenger at a stop
        boards a vehicle at the same cell/node, whatever direction the vehicle enters it from.
        """
        for handle, journey in self.passenger_journeys.items():
            passenger_resource, riding_handle = self.passenger_mode[handle]
            if passenger_resource is None:
                # mode 1 -> mode 2: show up at the origin on the first step.
                self._set_passenger(handle, journey.waypoints[0][0], None)
            if riding_handle is not None:
                self._update_riding(handle, riding_handle)
            else:
                self._update_waiting(handle, journey)

    def _update_riding(self, handle: int, riding_handle: int):
        """ Mode 3: follow the vehicle, or alight at the passenger's next stop or at the vehicle's final stop. """
        vehicle = self.rail_env.agents[riding_handle]
        if vehicle.state == TrainState.DONE:
            # Flatland removes a vehicle in the step it reaches its target
            self._alight(handle, vehicle.target_entry_point)
        elif self.stops_at_next_stop(riding_handle, handle):
            self._alight(handle, vehicle.current_entry_point)
        else:
            self._set_passenger(handle, vehicle.current_entry_point, riding_handle)

    def _alight(self, handle: int, entry_point: EntryPointT):
        """ Sets the passenger down at `entry_point`; it has arrived if `entry_point` is in its next stop (its target). """
        next_stop = self._next_stop(handle)
        self._set_passenger(handle, entry_point, None)
        if entry_point in next_stop:
            self._passenger_stop_index[handle] += 1
            self.passenger_target_entry_point[handle] = self.passenger_current_entry_point[handle]
            self.passenger_arrival_time[handle] = self.rail_env._elapsed_steps

    def _update_waiting(self, handle: int, journey: PassengerJourney):
        """ Mode 2: board a vehicle serving the passenger, if waiting at its origin from its earliest departure on. """
        stop_index = self._passenger_stop_index[handle]
        if stop_index > 0 or self.passenger_current_entry_point[handle] not in journey.waypoints[stop_index]:
            # arrived, or set down outside its target (single ride)
            return
        earliest_departure = journey.waypoints_earliest_departure[stop_index]
        if earliest_departure is not None and self.rail_env._elapsed_steps < earliest_departure:
            return
        passenger_resource = self.passenger_mode[handle][0]
        next_stop = self._next_stop(handle)
        candidates = [vehicle for vehicle in self.rail_env.agents
                      if vehicle.current_entry_point is not None and self._resource(vehicle.current_entry_point) == passenger_resource
                      and self.serves(vehicle.handle, handle)]
        if len(candidates) == 0:
            return
        # first a vehicle stopping at an entry point of the next stop, then the one stopping there soonest
        vehicle = min(candidates, key=lambda candidate: min((prediction.entry_point not in next_stop, prediction.step)
                                                            for prediction in self._future_stops_sharing(candidate.handle, next_stop)))
        self._set_passenger(handle, vehicle.current_entry_point, vehicle.handle)

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
