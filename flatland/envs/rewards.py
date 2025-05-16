from collections import defaultdict

from flatland.envs.agent_utils import EnvAgent
from flatland.envs.distance_map import DistanceMap
from flatland.envs.step_utils.env_utils import AgentTransitionData
from flatland.envs.step_utils.states import TrainState


class Rewards:
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
           Intermediate stop not served penalty $\mu \geq 0$. Defaults to 1.
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

    def step_reward(self, agent: EnvAgent, agent_transition_data: AgentTransitionData, distance_map: DistanceMap, elapsed_steps: int):
        """
        Handles end-of-step-reward for a particular agent.

        Parameters
        ----------
        agent: EnvAgent
        agent_transition_data: AgentTransitionData
        distance_map: DistanceMap
        elapsed_steps: int
        """
        reward = 0
        if agent.position not in self.arrivals[agent.handle]:
            self.arrivals[agent.handle][agent.position] = elapsed_steps
            self.departures[agent.handle][agent.old_position] = elapsed_steps
        if agent.state_machine.previous_state == TrainState.MOVING and agent.state == TrainState.STOPPED and not agent_transition_data.state_transition_signal.stop_action_given:
            reward += -1 * agent_transition_data.speed * self.crash_penalty_factor
        return reward

    def end_of_episode_reward(self, agent: EnvAgent, distance_map: DistanceMap, elapsed_steps: int) -> int:
        """
        Handles end-of-episode reward for a particular agent.

        Parameters
        ----------
        agent: EnvAgent
        distance_map: DistanceMap
        elapsed_steps: int
        """
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

        for et, la, ed in zip(agent.waypoints[1:-1], agent.waypoints_latest_arrival[1:-1], agent.waypoints_earliest_departure[1:-1]):
            if et not in self.arrivals[agent.handle]:
                # stop not served
                reward += -1 * self.intermediate_not_served_penalty
            else:
                # late arrival
                reward += self.intermediate_late_arrival_penalty_factor * min(la - self.arrivals[agent.handle][et], 0)
                # early departure
                # N.B. if arrival but not departure, handled by above by departed but never reached.
                if et in self.departures[agent.handle]:
                    reward += self.intermediate_early_departure_penalty_factor * min(self.departures[agent.handle][et] - ed, 0)
        return reward
