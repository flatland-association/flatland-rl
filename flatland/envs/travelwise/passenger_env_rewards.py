import abc
from typing import NamedTuple, Optional

from flatland.envs.travelwise.passenger_journey import PassengerJourney


class PassengerTransitionData(NamedTuple):
    """
    What a passenger's step did: the step and the stop index (the journey's group it waited at) it boarded at, and the
    step it arrived in - `None` if it did not board or arrive in this step.
    """
    boarding_step: Optional[int] = None
    boarding_stop_index: Optional[int] = None
    arrival_step: Optional[int] = None


class PassengerRewards(metaclass=abc.ABCMeta):
    """
    The rewards of `PassengerEnv`'s passengers, as `Rewards` are a rail env's: a reward per passenger and step, and one
    at the end of the episode, cumulated per step. Integers.
    """

    @abc.abstractmethod
    def step_reward(self, journey: PassengerJourney, transition: PassengerTransitionData, elapsed_steps: int) -> int:
        """
        The passenger's reward for the step `elapsed_steps`, after the vehicles and the riding passengers moved.

        Parameters
        ----------
        journey : PassengerJourney
            the passenger's journey
        transition : PassengerTransitionData
            what the passenger's step did
        elapsed_steps : int
            the step
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def end_of_episode_reward(self, journey: PassengerJourney, arrived: bool, max_episode_steps: int) -> int:
        """
        The passenger's reward when the rail env is done.

        Parameters
        ----------
        journey : PassengerJourney
            the passenger's journey
        arrived : bool
            whether the passenger has arrived at its target
        max_episode_steps : int
            the rail env's episode length
        """
        raise NotImplementedError()

    def cumulate(self, *rewards: int) -> int:
        return sum(rewards)

    def empty(self) -> int:
        return 0


class DefaultPassengerRewards(PassengerRewards):
    """
    Rewards in the step they occur, integers that can be positive (an early arrival) - unlike Flatland's
    `DefaultRewards`, which never reward arriving early:

    - arriving: `waypoints_latest_arrival[-1] - arrival step` - positive when early, negative when late;
    - boarding before its earliest departure at the group it waits at: `-(earliest departure - boarding step)` -
      respecting it is the policy's decision, the env does not enforce it;
    - not arrived when the rail env is done: `-max_episode_steps`.
    """

    def step_reward(self, journey: PassengerJourney, transition: PassengerTransitionData, elapsed_steps: int) -> int:
        reward = 0
        if transition.boarding_step is not None:
            earliest_departure = journey.waypoints_earliest_departure[transition.boarding_stop_index]
            if earliest_departure is not None and transition.boarding_step < earliest_departure:
                reward -= earliest_departure - transition.boarding_step
        if transition.arrival_step is not None:
            reward += journey.waypoints_latest_arrival[-1] - transition.arrival_step
        return reward

    def end_of_episode_reward(self, journey: PassengerJourney, arrived: bool, max_episode_steps: int) -> int:
        return 0 if arrived else -max_episode_steps
