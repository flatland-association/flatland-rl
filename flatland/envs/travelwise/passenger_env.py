import warnings
from numbers import Integral
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
from flatland.envs.travelwise.passenger import Passenger
from flatland.envs.travelwise.passenger_env_action import PassengerAction, PassengerActionT
from flatland.envs.travelwise.passenger_env_rewards import DefaultPassengerRewards, PassengerRewards, PassengerTransitionData
from flatland.envs.travelwise.passenger_journey import PassengerJourney


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

    Rewards per passenger and step come from `rewards` (a `PassengerRewards`, `DefaultPassengerRewards` by default),
    as a rail env's from its `Rewards`: the step reward after the vehicles and the riding passengers moved, and the
    end-of-episode reward once the rail env is done.

    `dones` holds per passenger whether it has arrived, and `'__all__'`: whether the rail env is done.
    Read the passengers' state from `passengers`, a `Passenger` per handle. A passenger has exactly 3 modes, tracked in
    its `mode`, a `PassengerMode` (resource, handle) pair:

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
    (`EnvAgent.target_entry_point`). Alighting at an entry point of its next stop (`Passenger.next_stop()`) completes the ride:
    at a change, the passenger waits there for the next vehicle; at its target, it has arrived, stays there and its
    further actions are ignored. Alighting elsewhere leaves it waiting there, with the same next stop.

    Limitations:

    - The changes are given, not planned: a passenger set down outside its next stop continues only on a vehicle
      from there that stops at its next stop. No minimum transfer time.
      TODO https://github.com/flatland-association/flatland-rl/issues/562 plan connections with the query view.
    - A passenger may board a vehicle at its resource even where the vehicle doesn't stop, e.g. passing through.
      TODO board only at a vehicle's stop.
    - A passenger may hop off at any step. TODO hop off only where the vehicle stops.
    - The vehicles' plans are per-vehicle predictions, optimistic (`UnilateralTimetablePredictor`); there is no query view
      of all vehicles' future stops yet.
    - Stops are entry points (station tracks), not stations.
    - Graph edges count as length 1 (https://github.com/flatland-association/flatland-rl/pull/542).
    """

    def __init__(self, rail_env: AbstractRailEnv, vehicle_policy: Policy, passenger_journeys: Dict[int, PassengerJourney],
                 obs_builder: ObservationBuilder, rewards: PassengerRewards = None):
        super().__init__()
        self.rail_env = rail_env
        self.rewards = rewards if rewards is not None else DefaultPassengerRewards()
        self.vehicle_policy = vehicle_policy
        self.passenger_journeys = passenger_journeys
        self.obs_builder = obs_builder

        # the passengers' state by handle, see the class docstring
        self.passengers: Dict[int, Passenger] = {}

        # the rail env's last vehicle observations and infos, which the vehicle policy acts on in the next step
        self.vehicle_observations: Dict[int, object] = {}
        self._vehicle_infos: Dict = {}
        self._static: Optional[PassengerEnvStaticData] = None
        self._passenger_obs: Dict = {}
        self._rewards: Dict[int, int] = {}
        # per passenger, what its current step did - for the step reward
        self._transitions: Dict[int, PassengerTransitionData] = {}
        self._rail_env_done = False

    def reset(self) -> Tuple[Dict, Dict]:
        """ Resets the rail env and the passengers. Returns the passengers' observations and the infos with the static data. """
        self.vehicle_observations, vehicle_infos = self.rail_env.reset()
        rail_env_static = RailEnvStaticData.from_env(self.rail_env)
        self._vehicle_infos = dict(vehicle_infos, static=rail_env_static)
        self._static = PassengerEnvStaticData(passenger_journeys=MappingProxyType(dict(self.passenger_journeys)), rail_env=rail_env_static)
        self._rail_env_done = False

        self.passengers = {handle: Passenger(handle=handle, journey=journey) for handle, journey in self.passenger_journeys.items()}

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
            Per passenger handle, its action - a missing passenger waits. Infeasible actions are ignored, malformed ones with a warning.

        Returns
        -------
        Tuple[Dict, Dict, Dict, Dict]
            The passengers' observations, their rewards (see the class docstring), `dones` per passenger (arrived) and
            `'__all__'` (true once the rail env is done, as no passenger can move after that), and the infos with the
            static data.
        """
        self._rewards = {handle: self.rewards.empty() for handle in self.passenger_journeys}
        self._transitions = {handle: PassengerTransitionData() for handle in self.passenger_journeys}
        if not self._rail_env_done:
            self._update_passengers(action_dict)
            vehicle_handles = self.rail_env.get_agent_handles()
            vehicle_actions = self.vehicle_policy.act_many(vehicle_handles, observations=[self.vehicle_observations[handle] for handle in vehicle_handles],
                                                           infos=self._vehicle_infos)
            self.vehicle_observations, _, rail_env_dones, vehicle_infos = self.rail_env.step(vehicle_actions)
            self._vehicle_infos = dict(vehicle_infos, static=self._static.rail_env)
            self._rail_env_done = rail_env_dones['__all__']
            self._move_passengers()
            for handle, journey in self.passenger_journeys.items():
                self._rewards[handle] = self.rewards.cumulate(
                    self._rewards[handle], self.rewards.step_reward(journey, self._transitions[handle], self.rail_env._elapsed_steps))
                if self._rail_env_done:
                    self._rewards[handle] = self.rewards.cumulate(
                        self._rewards[handle],
                        self.rewards.end_of_episode_reward(journey, self.passengers[handle].arrival_time is not None, self.rail_env._max_episode_steps))
            self._passenger_obs = self.obs_builder.get_many(self.get_agent_handles())
        dones = {handle: passenger.arrival_time is not None for handle, passenger in self.passengers.items()}
        return self._passenger_obs, self._rewards, dict(dones, __all__=self._rail_env_done), {"static": self._static}

    def can_hop_on(self, vehicle_handle: int, passenger_handle: int) -> bool:
        """
        Whether the passenger can hop on the vehicle physically: it waits, and the vehicle is on the map at its
        resource. A passenger who has arrived is done: its actions are ignored.
        """
        passenger = self.passengers[passenger_handle]
        passenger_resource, riding_handle = passenger.mode
        if riding_handle is not None or passenger_resource is None or passenger.next_stop() is None:
            return False
        vehicle_entry_point = self.rail_env.agents[vehicle_handle].current_entry_point
        return vehicle_entry_point is not None and self._resource(vehicle_entry_point) == passenger_resource

    def _move_passengers(self):
        """
        Moves the passengers with the rail env's freshly-stepped vehicles, before they observe: they show up at
        their origin on the first step (mode 1 -> mode 2), riding passengers follow their vehicles, and are forced off
        at a vehicle's final stop, where Flatland removes it in the step it reaches its target (mode 3 -> mode 2).
        """
        for handle, passenger in self.passengers.items():
            passenger_resource, riding_handle = passenger.mode
            if passenger_resource is None:
                self._set_passenger(handle, passenger.journey.waypoints[0][0], None)
            elif riding_handle is not None:
                vehicle = self.rail_env.agents[riding_handle]
                if vehicle.state == TrainState.DONE:
                    self._alight(handle, vehicle.target_entry_point, self.rail_env._elapsed_steps)
                else:
                    self._set_passenger(handle, vehicle.current_entry_point, riding_handle)

    def _update_passengers(self, actions: Dict[int, PassengerActionT]):
        """
        Applies the passengers' actions, before the rail env steps, on the state they observed - see the class docstring
        for the 3 modes/transitions. Infeasible actions are ignored, malformed ones with a warning (`_is_well_formed()`).
        Also writes the resulting entry point straight into the passenger's `current_entry_point`, and, the step a
        passenger arrives at its target (the step being executed), its `target_entry_point`/`arrival_time` - mirroring
        what `AbstractRailEnv.handle_done_state()` sets on an `EnvAgent` reaching its target.
        """
        for handle in self.passenger_journeys:
            action = actions.get(handle, (PassengerAction.WAIT, None))
            if not self._is_well_formed(handle, action):
                continue
            action, vehicle_handle = action
            riding_handle = self.passengers[handle].mode[1]
            if riding_handle is not None and action == PassengerAction.HOP_OFF:
                self._alight(handle, self.rail_env.agents[riding_handle].current_entry_point, self.rail_env._elapsed_steps + 1)
            elif riding_handle is None and action == PassengerAction.HOP_ON and self.can_hop_on(vehicle_handle, handle):
                self._set_passenger(handle, self.rail_env.agents[vehicle_handle].current_entry_point, vehicle_handle)
                self._transitions[handle] = self._transitions[handle]._replace(
                    boarding_step=self.rail_env._elapsed_steps + 1, boarding_stop_index=self.passengers[handle].stop_index)

    def _is_well_formed(self, handle: int, action) -> bool:
        """
        Whether the passenger's action is an `(action, vehicle handle)` pair, with a vehicle handle of the rail env for
        `HOP_ON`. Otherwise warns: the passenger policy has a bug (e.g. an RL policy's action space not bounding the
        vehicle handle), and the passenger waits. The rail env's handles are its agents' indices `0, ..., n - 1`, so
        the vehicle handle is checked by range: used as an index, a negative handle would board a vehicle counted from
        the end, and one out of range would raise. A well-formed action can still be infeasible, e.g. hopping on a
        vehicle elsewhere: such actions are part of the passengers' decisions and ignored without a warning.
        """
        if not isinstance(action, tuple) or len(action) != 2:
            warnings.warn(f"Passenger {handle}: action {action!r} is not an (action, vehicle handle) pair, ignored.")
            return False
        action, vehicle_handle = action
        if action == PassengerAction.HOP_ON and not (isinstance(vehicle_handle, Integral) and 0 <= vehicle_handle < self.rail_env.get_num_agents()):
            warnings.warn(f"Passenger {handle}: HOP_ON vehicle handle {vehicle_handle!r} is not one of the rail env's "
                          f"0, ..., {self.rail_env.get_num_agents() - 1}, ignored.")
            return False
        return True

    def _alight(self, handle: int, entry_point: EntryPointT, step: int):
        """
        Sets the passenger down at `entry_point` in `step`. If `entry_point` is in its next stop, the ride is complete:
        the passenger travels on to the following group; it has arrived if the next stop was its target.
        """
        passenger = self.passengers[handle]
        next_stop = passenger.next_stop()
        self._set_passenger(handle, entry_point, None)
        if entry_point in next_stop:
            passenger.stop_index += 1
            if passenger.next_stop() is not None:
                return
            passenger.target_entry_point = passenger.current_entry_point
            passenger.arrival_time = step
            self._transitions[handle] = self._transitions[handle]._replace(arrival_step=step)

    def _resource(self, entry_point: Optional[EntryPointT]) -> Optional[ResourceT]:
        """ The resource (grid cell, graph node's resource) at `entry_point` - `None` off map. """
        return self.rail_env.resource_map.get_resource(entry_point, None)

    def _set_passenger(self, handle: int, entry_point: EntryPointT, riding_handle: Optional[int]):
        """ Writes the passenger's entry point into its `current_entry_point` and its mode into its `mode`. """
        passenger = self.passengers[handle]
        passenger.current_entry_point = _sanitize_entry_point(entry_point)
        passenger.mode = (self._resource(entry_point), riding_handle)

    @property
    def done(self) -> bool:
        """ The rail env is done: no passenger can move after that. """
        return self._rail_env_done
