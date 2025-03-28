from pathlib import Path
from typing import Optional

from flatland.callbacks.callbacks import FlatlandCallbacks
from flatland.envs.rail_env import RailEnv
from flatland.envs.step_utils.states import TrainState


class FlatlandEvaluatorCallbacks(FlatlandCallbacks):
    """
    Implements Flatland evaluation similar to FlatlandRemoteEvaluationService for just one scenario and in offline mode.

    The result dict is similar to its `evaluation_state`.

    The following features are not implemented as they concern the evaluation of a full test with several trajectories in interactive mode:
    - INTIAL_PLANNING_TIMEOUT
    - PER_STEP_TIMEOUT
    - OVERALL_TIMEOUT
    - DEFAULT_COMMAND_TIMEOUT
    """

    def __init__(self):
        self._cumulative_reward = None
        self._normalized_reward = None
        self._percentage_complete = None
        self._termination_cause = None

    def on_episode_end(
        self,
        *,
        env: Optional[RailEnv] = None,
        data_dir: Path = None,
        **kwargs,
    ) -> None:
        # cumulative reward of all agents
        self._cumulative_reward = sum(env.rewards_dict.values())

        """
        The normalized rewards normalize the reward for an
        episode by dividing the whole reward by max-time-steps
        allowed in that episode, and the number of agents present in
        that episode
        """
        self._normalized_reward = (self._cumulative_reward / (
            env._max_episode_steps *
            env.get_num_agents()
        ))
        # Compute percentage complete
        complete = 0
        for i_agent in range(env.get_num_agents()):
            agent = env.agents[i_agent]
            if agent.state == TrainState.DONE:
                complete += 1
        self._percentage_complete = complete * 1.0 / env.get_num_agents()

    def get_evaluation(self) -> dict:
        """
        Evaluation for the trajectory.

        Returns
        -------
        reward : float
            cumulative reward of all agents.
        normalized reward : float
            The normalized rewards normalize the reward for an episode by dividing the whole reward by max-time-steps allowed in that episode, and the number of agents present in that episode.
        termination_cause : Optional[str]
            if timeout occurs.
        percentage_complete : float
            ratio of agents done.
        """
        return {
            "normalized_reward": self._normalized_reward,
            "termination_cause": self._termination_cause,
            "reward": self._cumulative_reward,
            "percentage_complete": self._percentage_complete
        }
