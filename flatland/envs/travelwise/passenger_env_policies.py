from typing import Dict, List, Optional

from flatland.core.policy import Policy
from flatland.envs.travelwise.passenger_env import PassengerEnv, PassengerJourney
from flatland.envs.travelwise.passenger_env_action import PassengerAction, PassengerActionT
from flatland.envs.travelwise.passenger_env_observations import PassengerObservation


class BoardForTargetPassengerPolicy(Policy[PassengerEnv, PassengerObservation, PassengerActionT]):
    """
    Acts on `PassengerObservation`s, with the passengers' journeys from the static data in the infos passed to
    `act_many()` (`infos["static"]`, a `PassengerEnvStaticData`). Each passenger boards a vehicle that serves its
    target, not before its earliest departure, and hops off where the vehicle stops at its target:

    - riding, the vehicle stopping at an entry point of its target (`vehicle_stops_at_next_stop`): `HOP_OFF`;
    - waiting, from its earliest departure on (the step it observed), with vehicles it can hop on and that serve it
      (`vehicles_to_board`): `HOP_ON` a vehicle - first one
      whose predicted route stops at an entry point of the target, then the one predicted to stop there soonest;
    - otherwise `WAIT`.

    A vehicle stopping at another alternative of the target's group is not left: the passenger stays on board up to
    the vehicle's final stop.
    """

    def act_many(self, handles: List[int], observations: List[PassengerObservation], infos: Optional[Dict] = None, **kwargs) -> Dict[int, PassengerActionT]:
        journeys = infos["static"].passenger_journeys
        return {handle: self.act(observation, journey=journeys[handle]) for handle, observation in zip(handles, observations)}

    def act(self, observation: PassengerObservation, journey: PassengerJourney = None, **kwargs) -> PassengerActionT:
        if observation.vehicle_stops_at_next_stop:
            return PassengerAction.HOP_OFF, None
        if len(observation.vehicles_to_board) == 0 or not self._may_depart(observation, journey):
            return PassengerAction.WAIT, None
        # first a vehicle stopping at an entry point of the next stop, then the one stopping there soonest
        vehicle_handle = min(observation.vehicles_to_board,
                             key=lambda candidate: min((stop.entry_point not in observation.next_stop, stop.step)
                                                       for stop in observation.vehicles_to_board[candidate]))
        return PassengerAction.HOP_ON, vehicle_handle

    @staticmethod
    def _may_depart(observation: PassengerObservation, journey: PassengerJourney) -> bool:
        """ Whether the passenger's earliest departure at the group it waits at has passed, in the step it observed. """
        earliest_departure = journey.waypoints_earliest_departure[journey.waypoints.index(observation.next_stop) - 1]
        return earliest_departure is None or observation.elapsed_steps >= earliest_departure
