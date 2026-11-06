from typing import Dict, List, Optional

from flatland.core.policy import Policy
from flatland.envs.travelwise.passenger_env import PassengerEnv, PassengerJourney
from flatland.envs.travelwise.passenger_env_action import PassengerAction, PassengerActionT
from flatland.envs.travelwise.passenger_env_observations import PassengerObservation


class BoardForTargetPassengerPolicy(Policy[PassengerEnv, PassengerObservation, PassengerActionT]):
    """
    Acts on `PassengerObservation`s, with the passengers' journeys from the static data in the infos passed to
    `act_many()` (`infos["static"]`, a `PassengerEnvStaticData`). Each passenger boards a vehicle that serves its
    next stop (a change or its target), not before its earliest departure, and hops off where the vehicle stops at
    its next stop:

    - riding, the vehicle stopping at an entry point of its next stop (`vehicle_stops_at_next_stop`): `HOP_OFF`;
    - waiting, boarding not before its earliest departure (the action applies in the step after the one observed, as
      for the vehicles), with vehicles it can hop on and that serve it
      (`vehicles_to_board`, planned to stop at an entry point of the next stop): `HOP_ON` the vehicle planned to stop
      there soonest;
    - otherwise `WAIT`.

    Limitations: set down outside its next stop (e.g. forced off at a vehicle's final stop), a passenger catches up with
    its next stop only directly, on a vehicle stopping where it waits and planned to stop at its next stop. It does not
    re-route via another change, does not skip ahead to a later group of its journey (e.g. straight to its target), and
    goes back only if such a vehicle comes by after it was carried past its next stop.
    TODO https://github.com/flatland-association/flatland-rl/issues/562 plan connections with the query view.
    """

    def act_many(self, handles: List[int], observations: List[PassengerObservation], infos: Optional[Dict] = None, **kwargs) -> Dict[int, PassengerActionT]:
        journeys = infos["static"].passenger_journeys
        return {handle: self.act(observation, journey=journeys[handle]) for handle, observation in zip(handles, observations)}

    def act(self, observation: PassengerObservation, *, journey: PassengerJourney, **kwargs) -> PassengerActionT:
        if observation.vehicle_stops_at_next_stop:
            return PassengerAction.HOP_OFF, None
        if len(observation.vehicles_to_board) == 0 or not self._may_depart(observation, journey):
            return PassengerAction.WAIT, None
        vehicle_handle = min(observation.vehicles_to_board,
                             key=lambda candidate: min(stop.step for stop in observation.vehicles_to_board[candidate]))
        return PassengerAction.HOP_ON, vehicle_handle

    @staticmethod
    def _may_depart(observation: PassengerObservation, journey: PassengerJourney) -> bool:
        """
        Whether boarding now respects the passenger's earliest departure at the group it waits at: the action applies in
        step `elapsed_steps + 1`, as `ShortestPathPolicy` departs a vehicle.
        """
        earliest_departure = journey.waypoints_earliest_departure[observation.stop_index]
        return earliest_departure is None or observation.elapsed_steps + 1 >= earliest_departure
