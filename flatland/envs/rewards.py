from collections import defaultdict
from typing import Generic, TypeVar, Tuple

from flatland.envs.agent_utils import EnvAgent
from flatland.envs.distance_map import DistanceMap
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


def defaultdict_set():
    return defaultdict(lambda: set())

class DefaultRewards(Rewards[float]):
    """
    Reward Function.

    This scoring function is designed to capture key operational metrics such as punctuality, efficiency in responding to disruptions, and safety.

    Punctuality and schedule adherence are rewarded based on the difference between actual and target arrival and departure times at each stop respectively,
    as well as penalties for intermediate stops not served or even journeys not started.

    Safety measures are implemented as penalties for collisions which are directly proportional to the train’s speed at impact, ensuring that high-speed operations are managed with extra caution.
    """

    def __init__(self,
                 epsilon: float = 0.01,
                 cancellation_factor: float = 1,
                 cancellation_time_buffer: float = 0,
                 intermediate_not_served_penalty: float = 1,
                 intermediate_late_arrival_penalty_factor: float = 0.2,
                 intermediate_early_departure_penalty_factor: float = 0.5,
                 crash_penalty_factor: float = 0.0
                 ):
        """
        Parameters
        ----------
        epsilon : float
            avoid rounding errors, defaults to 0.01.
        cancellation_factor : float
            Cancellation factor $\phi \geq 0$. defaults to  1.
        cancellation_time_buffer : float
            Cancellation time buffer $\pi \geq 0$. Defaults to 0.
        intermediate_not_served_penalty : float
           Intermediate stop not served penalty $\mu \geq 0$. Applied if one of the intermediates is not served or only run through without stopping. Defaults to 1.
        intermediate_late_arrival_penalty_factor : float
            Intermediate late arrival penalty factor $\alpha \geq 0$. Defaults to 0.2.
        intermediate_early_departure_penalty_factor : float
            Intermediate early departure penalty factor $\delta \geq 0$. Defaults to 0.5.
        crash_penalty_factor : float
            Crash penalty factor $\kappa \geq 0$. Defaults to 0.0.
        """
        self.crash_penalty_factor = crash_penalty_factor
        self.intermediate_early_departure_penalty_factor = intermediate_early_departure_penalty_factor
        self.intermediate_late_arrival_penalty_factor = intermediate_late_arrival_penalty_factor
        self.intermediate_not_served_penalty = intermediate_not_served_penalty
        self.cancellation_time_buffer = cancellation_time_buffer
        self.cancellation_factor = cancellation_factor
        assert self.crash_penalty_factor >= 0
        assert self.intermediate_early_departure_penalty_factor >= 0
        assert self.intermediate_late_arrival_penalty_factor >= 0
        assert self.intermediate_not_served_penalty >= 0
        assert self.cancellation_time_buffer >= 0
        assert self.cancellation_factor >= 0
        # https://stackoverflow.com/questions/16439301/cant-pickle-defaultdict
        self.arrivals = defaultdict(defaultdict)
        self.departures = defaultdict(defaultdict)
        self.states = defaultdict(defaultdict_set)

    def step_reward(self, agent: EnvAgent, agent_transition_data: AgentTransitionData, distance_map: DistanceMap, elapsed_steps: int) -> float:
        reward = 0
        if agent.position is not None:
            self.states[agent.handle][agent.position].add(agent.state)
        if agent.position not in self.arrivals[agent.handle]:
            self.arrivals[agent.handle][agent.position] = elapsed_steps
            self.departures[agent.handle][agent.old_position] = elapsed_steps
        if agent.state_machine.previous_state == TrainState.MOVING and agent.state == TrainState.STOPPED and not agent_transition_data.state_transition_signal.stop_action_given:
            reward += -1 * agent_transition_data.speed * self.crash_penalty_factor
        return reward

    def end_of_episode_reward(self, agent: EnvAgent, distance_map: DistanceMap, elapsed_steps: int) -> float:
        reward = None

        if agent.state == TrainState.DONE:
            # delay at target
            # if agent arrived earlier or on time = 0
            # if agent arrived later = -ve reward based on how late
            reward = min(agent.latest_arrival - agent.arrival_time, 0)
        else:
            if agent.state.is_off_map_state():
                # journey not started
                reward = -1 * self.cancellation_factor * \
                         (agent.get_travel_time_on_shortest_path(distance_map) + self.cancellation_time_buffer)

            # target not reached
            if agent.state.is_on_map_state():
                reward = agent.get_current_delay(elapsed_steps, distance_map)

        for wps, la, ed in zip(agent.waypoints[1:-1], agent.waypoints_latest_arrival[1:-1], agent.waypoints_earliest_departure[1:-1]):
            agent_arrivals = set(self.arrivals[agent.handle])
            wps_intersection = set(wps).intersection(agent_arrivals)
            if len(wps_intersection) == 0 or TrainState.STOPPED not in self.states[agent.handle][list(wps_intersection)[0]]:
                # stop not served or served but not stopped
                reward += -1 * self.intermediate_not_served_penalty
            else:
                wp = list(wps_intersection)[0]
                # late arrival
                reward += self.intermediate_late_arrival_penalty_factor * min(la - self.arrivals[agent.handle][wp], 0)
                # early departure
                # N.B. if arrival but not departure, handled by above by departed but never reached.
                if wp in self.departures[agent.handle]:
                    reward += self.intermediate_early_departure_penalty_factor * min(self.departures[agent.handle][wp] - ed, 0)
        return reward

    def cumulate(self, *rewards: int) -> RewardType:
        return sum(rewards)

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

    def __init__(self):
        super().__init__()
        self._previous_speeds = {}

    def step_reward(self, agent: EnvAgent, agent_transition_data: AgentTransitionData, distance_map: DistanceMap, elapsed_steps: int) -> Tuple[
        float, float, float]:
        default_reward = super().step_reward(agent=agent, agent_transition_data=agent_transition_data, distance_map=distance_map, elapsed_steps=elapsed_steps)

        # TODO revise design: speed_counter currently is not set to 0 during malfunctions.
        # N.B. enforces penalization before/after malfunction
        current_speed = agent.speed_counter.speed if agent.state == TrainState.MOVING else 0

        energy_efficiency = -(current_speed / agent.speed_counter.max_speed) ** 2
        smoothness = 0
        if agent.handle in self._previous_speeds:
            smoothness = -(current_speed - self._previous_speeds[agent.handle]) ** 2
        self._previous_speeds[agent.handle] = current_speed
        return default_reward, energy_efficiency, smoothness

    def end_of_episode_reward(self, agent: EnvAgent, distance_map: DistanceMap, elapsed_steps: int) -> Tuple[float, float, float]:
        default_reward = super().end_of_episode_reward(agent=agent, distance_map=distance_map, elapsed_steps=elapsed_steps)
        energy_efficency = 0
        smoothness = 0
        return default_reward, energy_efficency, smoothness

    def cumulate(self, *rewards: Tuple[float, float, float]) -> Tuple[float, float, float]:
        return tuple([sum([r[i] for r in rewards]) for i in range(3)])

    def empty(self) -> Tuple[float, float, float]:
        return 0, 0, 0


class PunctualityRewards(Rewards[Tuple[int, int]]):
    """
    Punctuality: n_stops_on_time / n_stops
    An agent is deemed not punctual at a stop if it arrives to late, departs to early or does not serve the stop at all. If an agent is punctual at a stop, n_stops_on_time is increased by 1.

    The implementation returns the tuple `(n_stops_on_time, n_stops)`.
    """

    def __init__(self):
        # https://stackoverflow.com/questions/16439301/cant-pickle-defaultdict
        self.arrivals = defaultdict(defaultdict)
        self.departures = defaultdict(defaultdict)

    def step_reward(self, agent: EnvAgent, agent_transition_data: AgentTransitionData, distance_map: DistanceMap, elapsed_steps: int) -> Tuple[int, int]:
        if agent.position is None and agent.state_machine.state == TrainState.DONE and agent.target not in self.arrivals[agent.handle]:
            self.arrivals[agent.handle][agent.target] = elapsed_steps

        if agent.position is not None and agent.position not in self.arrivals[agent.handle]:
            self.arrivals[agent.handle][agent.position] = elapsed_steps
            self.departures[agent.handle][agent.old_position] = elapsed_steps

        return 0, 0

    def end_of_episode_reward(self, agent: EnvAgent, distance_map: DistanceMap, elapsed_steps: int) -> Tuple[int, int]:
        n_stops_on_time = 0
        initial_wp = agent.waypoints[0][0]
        if initial_wp.position in self.departures[agent.handle] and self.departures[agent.handle][initial_wp.position] >= agent.waypoints_earliest_departure[0]:
            n_stops_on_time += 1
        for i, (wps, la, ed) in enumerate(zip(
            agent.waypoints[1:-1],
            agent.waypoints_latest_arrival[1:-1],
            agent.waypoints_earliest_departure[1:-1]
        )):
            for wp in wps:
                if wp.position not in self.arrivals[agent.handle] or wp.position not in self.departures[agent.handle]:
                    # intermediate stop not served
                    continue
                if self.arrivals[agent.handle][wp.position] > agent.waypoints_latest_arrival[i + 1]:
                    # intermediate late arrival
                    continue
                if self.departures[agent.handle][wp.position] < agent.waypoints_earliest_departure[i + 1]:
                    # intermediate early departure
                    continue
                n_stops_on_time += 1
                break
        target_wp = agent.waypoints[-1][0]
        if target_wp.position in self.arrivals[agent.handle] and self.arrivals[agent.handle][target_wp.position] <= agent.waypoints_latest_arrival[-1]:
            n_stops_on_time += 1
        n_stops = len(agent.waypoints)
        return n_stops_on_time, n_stops

    def cumulate(self, *rewards: Tuple[int, int]) -> Tuple[int, int]:
        return sum([r[0] for r in rewards]), sum([r[1] for r in rewards])

    def empty(self) -> Tuple[int, int]:
        return 0, 0
