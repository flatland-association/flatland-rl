from flatland.envs.agent_utils import EnvAgent
from flatland.envs.distance_map import DistanceMap
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
    """

    def __init__(self):
        # Epsilon to avoid rounding errors
        self.epsilon = 0.01
        # NEW : REW: Sparse Reward
        self.alpha = 0
        self.beta = 0
        self.step_penalty = -1 * self.alpha
        self.global_reward = 1 * self.beta
        self.invalid_action_penalty = 0  # previously -2; GIACOMO: we decided that invalid actions will carry no penalty
        self.stop_penalty = 0  # penalty for stopping a moving agent
        self.start_penalty = 0  # penalty for starting a stopped agent
        self.cancellation_factor = 1
        self.cancellation_time_buffer = 0

    def step_reward(self, agent: EnvAgent, distance_map: DistanceMap, elapsed_steps: int):
        """
        Handles end-of-step-reward for a particular agent.

        Parameters
        ----------
        agent: EnvAgent
        distance_map: DistanceMap
        elapsed_steps: int
        """
        return 0

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

        return reward
