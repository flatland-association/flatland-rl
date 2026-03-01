from collections import defaultdict
from typing import Generic, TypeVar, Tuple, Dict, Set, Optional, List

import numpy as np
from fastenum import fastenum

from flatland.core.env_observation_builder import AgentHandle
from flatland.envs.agent_utils import EnvAgent
from flatland.envs.grid.distance_map import DistanceMap
from flatland.envs.rail_trainrun_data_structures import Waypoint
from flatland.envs.step_utils.env_utils import AgentTransitionData
from flatland.envs.step_utils.states import TrainState

RewardType = TypeVar('RewardType')


class Rewards(Generic[RewardType]):
    """
    Reward Function Interface.
    """

    def step_reward(self, agent: EnvAgent, agent_transition_data: AgentTransitionData, distance_map: DistanceMap, elapsed_steps: int) -> RewardType:
        """
        Handles end-of-step-reward for a particular agent.

        Parameters
        ----------
        agent: EnvAgent
        agent_transition_data: AgentTransitionData
        distance_map: DistanceMap
        elapsed_steps: int
        """
        raise NotImplementedError()

    def end_of_episode_reward(self, agent: EnvAgent, distance_map: DistanceMap, elapsed_steps: int) -> RewardType:
        """
        Handles end-of-episode reward for a particular agent.

        Parameters
        ----------
        agent: EnvAgent
        distance_map: DistanceMap
        elapsed_steps: int
        """
        raise NotImplementedError()

    def cumulate(self, *rewards: RewardType) -> RewardType:
        """
        Cumulate multiple rewards to one.

        Parameters
        ----------
        rewards

        Returns
        -------
        Cumulative rewards

        """
        raise NotImplementedError()

    def empty(self) -> RewardType:
        """
        Return empty initial value neutral for the cumulation.
        """
        raise NotImplementedError()

    def normalize(self, *rewards: RewardType, num_agents: int, max_episode_steps: int) -> Optional[float]:
        """
        Return normalized cumulated rewards. Can be `None` for some rewards.

        Parameters
        ----------
        rewards : List[RewardType]
        num_agents : int
        max_episode_steps : int

        Returns
        -------

        """
        return None


def defaultdict_set():
    return defaultdict(lambda: set())


def defaultdict_list():
    return defaultdict(lambda: [])


class DefaultPenalties(fastenum.Enum):
    COLLISION = "COLLISION"
    TARGET_LATE_ARRIVAL = "TARGET_LATE_ARRIVAL"
    CANCELLATION = "CANCELLATION"
    TARGET_NOT_REACHED = "TARGET_NOT_REACHED"
    INTERMEDIATE_NOT_SERVED = "INTERMEDIATE_NOT_SERVED"
    INTERMEDIATE_LATE_ARRIVAL = "INTERMEDIATE_LATE_ARRIVAL"
    INTERMEDIATE_EARLY_DEPARTURE = "INTERMEDIATE_EARLY_DEPARTURE"


class BaseDefaultRewards(Rewards[Dict[str, float]]):
    r"""
    Reward Function.

    This scoring function is designed to capture key operational metrics such as punctuality, efficiency in responding to disruptions, and safety.

    Punctuality and schedule adherence are rewarded based on the difference between actual and target arrival and departure times at each stop respectively,
    as well as penalties for intermediate stops not served or even journeys not started.

    Safety measures are implemented as penalties for collisions which are directly proportional to the train’s speed at impact, ensuring that high-speed operations are managed with extra caution.

    Parameters
    ----------
    cancellation_factor : float
        Cancellation factor :math:`\phi \geq 0`. defaults to  1.
    cancellation_time_buffer : float
        Cancellation time buffer :math:`\pi \geq 0`. Defaults to 0.
    intermediate_not_served_penalty : float
       Intermediate stop not served penalty :math:`\mu \geq 0`. Applied if one of the intermediates is not served or only run through without stopping. Defaults to 1.
    intermediate_late_arrival_penalty_factor : float
        Intermediate late arrival penalty factor :math:`\alpha \geq 0`. Defaults to 0.2.
    intermediate_early_departure_penalty_factor : float
        Intermediate early departure penalty factor :math:`\delta \geq 0`. Defaults to 0.5.
    collision_factor : float
        Crash penalty factor :math:`\kappa \geq 0`. Defaults to 0.0.
    """

    def __init__(self,
                 cancellation_factor: float = 1,
                 cancellation_time_buffer: float = 0,
                 intermediate_not_served_penalty: float = 1,
                 intermediate_late_arrival_penalty_factor: float = 0.2,
                 intermediate_early_departure_penalty_factor: float = 0.5,
                 collision_factor: float = 0.0
                 ):
        self.collision_factor = collision_factor
        self.intermediate_early_departure_penalty_factor = intermediate_early_departure_penalty_factor
        self.intermediate_late_arrival_penalty_factor = intermediate_late_arrival_penalty_factor
        self.intermediate_not_served_penalty = intermediate_not_served_penalty
        self.cancellation_time_buffer = cancellation_time_buffer
        self.cancellation_factor = cancellation_factor
        assert self.collision_factor >= 0
        assert self.intermediate_early_departure_penalty_factor >= 0
        assert self.intermediate_late_arrival_penalty_factor >= 0
        assert self.intermediate_not_served_penalty >= 0
        assert self.cancellation_time_buffer >= 0
        assert self.cancellation_factor >= 0
        # https://stackoverflow.com/questions/16439301/cant-pickle-defaultdict
        self.arrivals: Dict[AgentHandle, Dict[Waypoint, List[int]]] = defaultdict(defaultdict_list)
        self.departures: Dict[AgentHandle, Dict[Waypoint, List[int]]] = defaultdict(defaultdict_list)
        self.states: Dict[AgentHandle, Dict[Waypoint, Set[TrainState]]] = defaultdict(defaultdict_set)

    def step_reward(self, agent: EnvAgent, agent_transition_data: AgentTransitionData, distance_map: DistanceMap, elapsed_steps: int) -> Dict[str, float]:
        d = self.empty()
        if agent.current_configuration is not None:
            wp = Waypoint(agent.position, agent.direction)
            self.states[agent.handle][wp].add(agent.state)

            # Only record arrival if this is a new waypoint (not dwelling at same position)
            if agent.old_configuration != agent.current_configuration:
                assert wp is not None
                assert elapsed_steps is not None
                self.arrivals[agent.handle][wp].append(elapsed_steps)
                # Only record departure from old position when we arrive from on-map position
                if agent.old_configuration is not None:
                    old_wp = Waypoint(agent.old_position, agent.old_direction)
                    self.departures[agent.handle][old_wp].append(elapsed_steps)
        elif agent.old_configuration is not None:
            old_wp = Waypoint(agent.old_position, agent.old_direction)
            self.departures[agent.handle][old_wp].append(elapsed_steps)

        if agent.state_machine.previous_state == TrainState.MOVING and agent.state == TrainState.STOPPED:
            # agent_transition_data.speed has speed after action is applied at start of step(), not set to 0 upon motion check.
            # - if braking, reduced speed
            # - if not braking, still full speed
            # TODO https://github.com/flatland-association/flatland-rl/issues/280 revise design, should we penalize invalid actions upon symmetric switch?
            # - if invalid action, speed set to 0
            d[DefaultPenalties.COLLISION.value] = -1 * agent_transition_data.speed * self.collision_factor

        if agent.state == TrainState.DONE and agent.state_machine.previous_state != TrainState.DONE:
            self._agent_done_or_max_episode_steps_reward(agent, distance_map, elapsed_steps, d)
        return d

    def end_of_episode_reward(self, agent: EnvAgent, distance_map: DistanceMap, elapsed_steps: int) -> Dict[str, float]:
        d = self.empty()
        # If agent finished during episode, reward already calculated in step_reward()
        if agent.state == TrainState.DONE:
            return d
        # Calculate penalty for not reaching target before episode end
        return self._agent_done_or_max_episode_steps_reward(agent, distance_map, elapsed_steps, d)

    def _agent_done_or_max_episode_steps_reward(self, agent, distance_map, elapsed_steps, d: Dict[str, float]):
        """
        Calculate final rewards/penalties for an agent.

        Called in two contexts:
        1. From step_reward(): when agent transitions to DONE during episode
        2. From end_of_episode_reward(): when episode ends and agent didn't finish

        Handles both completed and incomplete journeys.
        """
        if agent.state == TrainState.DONE:
            # delay at target
            # if agent arrived earlier or on time = 0
            # if agent arrived later = -ve reward based on how late
            d[DefaultPenalties.TARGET_LATE_ARRIVAL.value] = min(agent.latest_arrival - agent.arrival_time, 0)
        else:
            if agent.state.is_off_map_state():
                # journey not started
                d[DefaultPenalties.CANCELLATION.value] = -1 * self.cancellation_factor * \
                                                         (agent.get_travel_time_on_shortest_path(distance_map) + self.cancellation_time_buffer)

            # target not reached
            if agent.state.is_on_map_state():
                d[DefaultPenalties.TARGET_NOT_REACHED.value] = agent.get_current_delay(elapsed_steps, distance_map)
        for intermediate_alternatives, la, ed in zip(agent.waypoints[1:-1], agent.waypoints_latest_arrival[1:-1],
                                                     agent.waypoints_earliest_departure[1:-1]):
            agent_arrivals: Set[Waypoint] = set(self.arrivals[agent.handle])
            intermediate_alternatives: Set[Waypoint] = set(intermediate_alternatives)
            wps_intersection: Set[Waypoint] = intermediate_alternatives.intersection(agent_arrivals)
            if len(wps_intersection) == 0 or TrainState.STOPPED not in self.states[agent.handle][list(wps_intersection)[0]]:
                # stop not served or served but not stopped
                d[DefaultPenalties.INTERMEDIATE_NOT_SERVED.value] += -1 * self.intermediate_not_served_penalty
            else:
                lates = []
                earlies = []
                # take best time window (minimum penalty sum) over all alternatives and all arrival/departures
                for wp in list(wps_intersection):
                    # `+ [None]` is for arrival but no departure
                    for arrival, departure in zip(self.arrivals[agent.handle][wp], self.departures[agent.handle][wp] + [None]):
                        # late arrival
                        lates.append(self.intermediate_late_arrival_penalty_factor * min(la - arrival, 0))
                        # early departure
                        # N.B. if arrival but not departure, handled by above by departed but never reached.
                        if departure is not None:
                            earlies.append(self.intermediate_early_departure_penalty_factor * min(departure - ed, 0))
                        else:
                            earlies.append(0)
                totals = [l + e for l, e in zip(lates, earlies)]
                # argmax as penalty is negative reward
                d[DefaultPenalties.INTERMEDIATE_LATE_ARRIVAL.value] += lates[np.argmax(totals)]
                d[DefaultPenalties.INTERMEDIATE_EARLY_DEPARTURE.value] += earlies[np.argmax(totals)]
        return d

    def cumulate(self, *rewards: float) -> Dict[str, float]:
        return {p.value: sum([r[p.value] for r in rewards]) for p in DefaultPenalties}

    def normalize(self, *rewards: float, num_agents: int, max_episode_steps: int) -> float:
        # https://flatland-association.github.io/flatland-book/challenges/flatland3/eval.html
        return sum([sum([r[p.value] for r in rewards]) / (max_episode_steps * num_agents) for p in DefaultPenalties]) + 1

    def empty(self) -> Dict[str, float]:
        return {p.value: 0 for p in DefaultPenalties}


class DefaultRewards(Rewards[float]):
    """
    Aggregates `FineDefaultRewards` to single `float`.
    """

    def __init__(self,
                 cancellation_factor: float = 1,
                 cancellation_time_buffer: float = 0,
                 intermediate_not_served_penalty: float = 1,
                 intermediate_late_arrival_penalty_factor: float = 0.2,
                 intermediate_early_departure_penalty_factor: float = 0.5,
                 collision_factor: float = 0.0
                 ):
        self._proxy = BaseDefaultRewards(
            cancellation_factor=cancellation_factor,
            cancellation_time_buffer=cancellation_time_buffer,
            intermediate_not_served_penalty=intermediate_not_served_penalty,
            intermediate_late_arrival_penalty_factor=intermediate_late_arrival_penalty_factor,
            intermediate_early_departure_penalty_factor=intermediate_early_departure_penalty_factor,
            collision_factor=collision_factor
        )

    @property
    def collision_factor(self):
        return self._proxy.collision_factor

    @property
    def intermediate_early_departure_penalty_factor(self):
        return self._proxy.intermediate_early_departure_penalty_factor

    @property
    def intermediate_late_arrival_penalty_factor(self):
        return self._proxy.intermediate_late_arrival_penalty_factor

    @property
    def intermediate_not_served_penalty(self):
        return self._proxy.intermediate_not_served_penalty

    @property
    def cancellation_time_buffer(self):
        return self._proxy.cancellation_time_buffer

    @property
    def cancellation_factor(self):
        return self._proxy.cancellation_factor

    @collision_factor.setter
    def collision_factor(self, v):
        self._proxy.collision_factor = v

    @intermediate_early_departure_penalty_factor.setter
    def intermediate_early_departure_penalty_factor(self, v):
        self._proxy.intermediate_early_departure_penalty_factor = v

    @intermediate_late_arrival_penalty_factor.setter
    def intermediate_late_arrival_penalty_factor(self, v):
        self._proxy.intermediate_late_arrival_penalty_factor = v

    @intermediate_not_served_penalty.setter
    def intermediate_not_served_penalty(self, v):
        self._proxy.intermediate_not_served_penalty = v

    @cancellation_time_buffer.setter
    def cancellation_time_buffer(self, v):
        self._proxy.cancellation_time_buffer = v

    @cancellation_factor.setter
    def cancellation_factor(self, v):
        self._proxy.cancellation_factor = v

    def step_reward(self, agent: EnvAgent, agent_transition_data: AgentTransitionData, distance_map: DistanceMap, elapsed_steps: int) -> float:
        return sum(self._proxy.step_reward(agent, agent_transition_data, distance_map, elapsed_steps).values())

    def end_of_episode_reward(self, agent: EnvAgent, distance_map: DistanceMap, elapsed_steps: int) -> float:
        return sum(self._proxy.end_of_episode_reward(agent, distance_map, elapsed_steps).values())

    def cumulate(self, *rewards: float) -> float:
        return sum(rewards)

    def normalize(self, *rewards: float, num_agents: int, max_episode_steps: int) -> float:
        # https://flatland-association.github.io/flatland-book/challenges/flatland3/eval.html
        return sum(rewards) / (max_episode_steps * num_agents) + 1.0

    def empty(self) -> float:
        return 0


class BasicMultiObjectiveRewards(DefaultRewards, Rewards[Tuple[float, float, float]]):
    """
    Basic MORL (Multi-Objective Reinforcement Learning) Rewards: with 3 items
     - default score
     - energy efficiency: - square of (speed/max_speed).
     - smoothness: - square of speed differences
    For illustration purposes.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._previous_speeds = {}

    def step_reward(self, agent: EnvAgent, agent_transition_data: AgentTransitionData, distance_map: DistanceMap, elapsed_steps: int) -> Tuple[
        float, float, float]:
        default_reward = super().step_reward(agent=agent, agent_transition_data=agent_transition_data, distance_map=distance_map, elapsed_steps=elapsed_steps)

        # TODO https://github.com/flatland-association/flatland-rl/issues/280 revise design: speed_counter currently is not set to 0 during malfunctions.
        # N.B. enforces penalization before/after malfunction
        current_speed = agent.speed_counter.speed if agent.state == TrainState.MOVING else 0

        energy_efficiency = -(current_speed / agent.speed_counter.max_speed) ** 2
        smoothness = 0
        if agent.handle in self._previous_speeds:
            smoothness = -(current_speed - self._previous_speeds[agent.handle]) ** 2
        self._previous_speeds[agent.handle] = current_speed
        return default_reward, float(energy_efficiency), float(smoothness)

    def end_of_episode_reward(self, agent: EnvAgent, distance_map: DistanceMap, elapsed_steps: int) -> Tuple[float, float, float]:
        default_reward = super().end_of_episode_reward(agent=agent, distance_map=distance_map, elapsed_steps=elapsed_steps)
        energy_efficency = 0
        smoothness = 0
        return default_reward, energy_efficency, smoothness

    def cumulate(self, *rewards: Tuple[float, float, float]) -> Tuple[float, float, float]:
        return tuple([sum([r[i] for r in rewards]) for i in range(3)])

    def empty(self) -> Tuple[float, float, float]:
        return 0, 0, 0

    def normalize(self, *rewards: float, num_agents: int, max_episode_steps: int) -> float:
        return None


class PunctualityRewards(Rewards[Tuple[int, int]]):
    """
    Punctuality: n_stops_on_time / n_stops
    An agent is deemed not punctual at a stop if it arrives too late, departs too early or does not serve the stop at all. If an agent is punctual at a stop, n_stops_on_time is increased by 1.

    The implementation returns the tuple `(n_stops_on_time, n_stops)`.
    """

    def __init__(self):
        # https://stackoverflow.com/questions/16439301/cant-pickle-defaultdict
        self.arrivals = defaultdict(defaultdict_list)
        self.departures = defaultdict(defaultdict_list)

    def step_reward(self, agent: EnvAgent, agent_transition_data: AgentTransitionData, distance_map: DistanceMap, elapsed_steps: int) -> Tuple[int, int]:
        if agent.position is None and agent.state_machine.state == TrainState.DONE and agent.target not in self.arrivals[agent.handle]:
            self.arrivals[agent.handle][agent.target].append(elapsed_steps)

        if agent.position is not None and agent.position not in self.arrivals[agent.handle]:
            self.arrivals[agent.handle][agent.position].append(elapsed_steps)
            self.departures[agent.handle][agent.old_position].append(elapsed_steps)

        return 0, 0

    def end_of_episode_reward(self, agent: EnvAgent, distance_map: DistanceMap, elapsed_steps: int) -> Tuple[int, int]:
        n_stops_on_time = 0
        # by design, initial waypoint is unique
        initial_wp = agent.waypoints[0][0]
        if initial_wp.position in self.departures[agent.handle]:
            stop_on_time = False
            for departure in self.departures[agent.handle][initial_wp.position]:
                if departure >= agent.waypoints_earliest_departure[0]:
                    stop_on_time = True
                    break
            if stop_on_time:
                n_stops_on_time += 1
        for i, (wps, la, ed) in enumerate(zip(
            agent.waypoints[1:-1],
            agent.waypoints_latest_arrival[1:-1],
            agent.waypoints_earliest_departure[1:-1]
        )):
            stop_on_time = False
            # has any alternative with any arrival/departure been served on time?
            for wp in wps:
                if wp.position not in self.arrivals[agent.handle] or wp.position not in self.departures[agent.handle]:
                    # intermediate stop not served
                    continue
                for arrival, departure in zip(self.arrivals[agent.handle][wp.position], self.departures[agent.handle][wp.position]):
                    if arrival <= agent.waypoints_latest_arrival[i + 1] and departure >= agent.waypoints_earliest_departure[i + 1]:
                        stop_on_time = True
                        break
            if stop_on_time:
                n_stops_on_time += 1
                break
        # by design, target is only one cell
        target_wp = agent.waypoints[-1][0]
        if target_wp.position in self.arrivals[agent.handle] and self.arrivals[agent.handle][target_wp.position][0] <= agent.waypoints_latest_arrival[-1]:
            n_stops_on_time += 1
        n_stops = len(agent.waypoints)
        return n_stops_on_time, n_stops

    def cumulate(self, *rewards: Tuple[int, int]) -> Tuple[int, int]:
        return sum([r[0] for r in rewards]), sum([r[1] for r in rewards])

    def empty(self) -> Tuple[int, int]:
        return 0, 0
