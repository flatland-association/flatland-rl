from collections import defaultdict

from flatland.envs.agent_utils import EnvAgent
from flatland.envs.distance_map import DistanceMap
from flatland.envs.step_utils.env_utils import AgentTransitionData
from flatland.envs.step_utils.states import TrainState


class Rewards:
    """
    Reward Function:

    It costs each agent a step_penalty for every time-step taken in the environment. Independent of the movement
    of the agent. Currently all other penalties such as penalty for stopping, starting and invalid actions are set to 0.

    alpha = 0
    beta = 0
    Reward function parameters:

    - invalid_action_penalty = 0
    - step_penalty = -alpha
    - global_reward = beta
    - epsilon = avoid rounding errors
    - stop_penalty = 0  # penalty for stopping a moving agent
    - start_penalty = 0  # penalty for starting a stopped agent
    - intermediate_not_served_penalty = -1
    - intermediate_late_arrival_penalty_factor = 0.2
    - intermediate_early_departure_penalty_factor = 0.5
    """
    # Epsilon to avoid rounding errors
    epsilon = 0.01
    # NEW : REW: Sparse Reward
    alpha = 0
    beta = 0
    step_penalty = -1 * alpha
    global_reward = 1 * beta
    invalid_action_penalty = 0  # previously -2; GIACOMO: we decided that invalid actions will carry no penalty
    stop_penalty = 0  # penalty for stopping a moving agent
    start_penalty = 0  # penalty for starting a stopped agent
    cancellation_factor = 1
    cancellation_time_buffer = 0
    intermediate_not_served_penalty = -1
    intermediate_late_arrival_penalty_factor = 0.2
    intermediate_early_departure_penalty_factor = 0.5
    crash_penalty_factor = 0.0  # penalty for stopping train in conflict

    def __init__(self):
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
            reward += agent_transition_data.speed * self.crash_penalty_factor
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
        # agent done? (arrival_time is not None)
        if agent.state == TrainState.DONE:
            # if agent arrived earlier or on time = 0
            # if agent arrived later = -ve reward based on how late
            reward = min(agent.latest_arrival - agent.arrival_time, 0)

        # Agents not done (arrival_time is None)
        else:
            # CANCELLED check (never departed)
            if (agent.state.is_off_map_state()):
                reward = -1 * self.cancellation_factor * \
                         (agent.get_travel_time_on_shortest_path(distance_map) + self.cancellation_time_buffer)

            # Departed but never reached
            if (agent.state.is_on_map_state()):
                reward = agent.get_current_delay(elapsed_steps, distance_map)

        for et, la, ed in zip(agent.waypoints[1:-1], agent.waypoints_latest_arrival[1:-1], agent.waypoints_earliest_departure[1:-1]):
            if et not in self.arrivals[agent.handle]:
                reward += self.intermediate_not_served_penalty
            else:
                reward += self.intermediate_late_arrival_penalty_factor * min(la - self.arrivals[agent.handle][et], 0)
                # if arrival but not departure, handled by above by departed but never reached.
                if et in self.departures[agent.handle]:
                    reward += self.intermediate_early_departure_penalty_factor * min(self.departures[agent.handle][et] - ed, 0)
        return reward
