from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from flatland.core.env_observation_builder import AgentHandle, ObservationBuilder
from flatland.core.resource_map import ResourceT
from flatland.envs.agent_utils import EntryPointT
from flatland.envs.travelwise.passenger_env import PassengerEnv
from flatland.envs.travelwise.timetable_predictor import ProjectedEntryPoint
from flatland.envs.travelwise.vehicle_observations import VehiclePositionsAndPlans


@dataclass(frozen=True)
class PassengerObservation:
    """
    What a passenger observes to decide on its action:

    - `elapsed_steps`: the env's elapsed steps (TODO move to the infos, as for the vehicles);
    - `mode`: its `(resource, riding vehicle handle)`, see `PassengerEnv`;
    - `next_stop`: the entry points of the waypoint group it travels to next (a change or its target) - `None` once
      arrived;
    - `vehicles_to_board`: per vehicle it can hop on now (`PassengerEnv.can_hop_on()`) and that serves it, the
      vehicle's planned future stops serving its next stop (`PassengerObservationBuilder.future_stops_serving()`);
    - `vehicle_stops_at_next_stop`: whether the vehicle it rides stops at an entry point of its next stop now
      (`PassengerObservationBuilder.stops_at_next_stop()`).

    After `reset()`, the passenger is off the map and observes no vehicle.
    """
    elapsed_steps: int
    mode: Tuple[Optional[ResourceT], Optional[int]]
    next_stop: Optional[List[EntryPointT]]
    vehicles_to_board: Dict[int, List[ProjectedEntryPoint]]
    vehicle_stops_at_next_stop: bool


class PassengerObservationBuilder(ObservationBuilder[PassengerEnv, PassengerObservation]):
    """
    Builds each passenger's `PassengerObservation`: the passengers observe the vehicles' positions and plans from the
    vehicle observations `PassengerEnv` keeps (`VehicleObservation.vehicles`), and read their journeys and the
    vehicles' timetables from the env.
    """

    def __init__(self):
        super().__init__()
        self._vehicles: Optional[VehiclePositionsAndPlans] = None

    def get_many(self, handles: Optional[List[AgentHandle]] = None) -> Dict[AgentHandle, PassengerObservation]:
        # the positions and plans are shared by all vehicle observations
        self._vehicles = next(iter(self.env.vehicle_observations.values())).vehicles if self.env.vehicle_observations else None
        return super().get_many(handles)

    def get(self, handle: AgentHandle = 0) -> PassengerObservation:
        riding_handle = self.env.passenger_mode[handle][1]
        return PassengerObservation(
            elapsed_steps=self.env.rail_env._elapsed_steps,
            mode=self.env.passenger_mode[handle],
            next_stop=self.env.next_stop(handle),
            vehicles_to_board={vehicle.handle: self.future_stops_serving(vehicle.handle, handle) for vehicle in self.env.rail_env.agents
                               if self.env.can_hop_on(vehicle.handle, handle) and self.serves(vehicle.handle, handle)},
            vehicle_stops_at_next_stop=riding_handle is not None and self.stops_at_next_stop(riding_handle, handle),
        )

    def future_stops_serving(self, vehicle_handle: int, passenger_handle: int) -> List[ProjectedEntryPoint]:
        """
        The vehicle's future stops, as planned - after the group it stands on or last left - whose group shares an
        entry point with the passenger's next stop. The vehicle stops at one entry point of such a group, chosen by its
        policy, possibly one outside the passenger's next stop.
        """
        next_stop = self.env.next_stop(passenger_handle)
        if next_stop is None:
            return []
        vehicle_waypoints = self.env.rail_env.agents[vehicle_handle].waypoints
        return [prediction for prediction in self._vehicles.plans[vehicle_handle][1:]
                if prediction.stop is not None and not set(next_stop).isdisjoint(vehicle_waypoints[prediction.stop])]

    def serves(self, vehicle_handle: int, passenger_handle: int) -> bool:
        """ Whether the vehicle serves the passenger: it has a future stop serving the passenger's next stop, see `future_stops_serving()`. """
        return len(self.future_stops_serving(vehicle_handle, passenger_handle)) > 0

    def stops_at_next_stop(self, vehicle_handle: int, passenger_handle: int) -> bool:
        """ Whether the vehicle is at the entry point it stops at for one of its groups, and it is in the passenger's next stop. """
        next_stop = self.env.next_stop(passenger_handle)
        plan = self._vehicles.plans[vehicle_handle]
        return next_stop is not None and len(plan) > 0 and plan[0].stop is not None and plan[0].entry_point in next_stop
