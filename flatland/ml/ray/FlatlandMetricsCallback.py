from typing import List, Union, Optional, Dict

import gymnasium as gym
from ray.rllib import BaseEnv
from ray.rllib.callbacks.callbacks import RLlibCallback
from ray.rllib.core.rl_module import RLModule
from ray.rllib.env.vector.vector_multi_agent_env import VectorMultiAgentEnv
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.utils.metrics.metrics_logger import MetricsLogger
from ray.rllib.utils.typing import EpisodeType, PolicyID

from flatland.core.policy import Policy
from flatland.envs.rail_env import RailEnv
from flatland.envs.step_utils.states import TrainState
from flatland.ml.ray.ray_multi_agent_rail_env import RayMultiAgentWrapper


# See also for https://gitlab.aicrowd.com/flatland/neurips2020-flatland-baselines/-/blob/master/train.py?ref_type=heads
class FlatlandMetricsCallback(RLlibCallback):
    """
    Add `normalized_reward` and `percentage_complete` evaluation metrics.
    """

    def on_episode_end(
        self,
        *,
        episode: Union[EpisodeType, EpisodeV2],
        prev_episode_chunks: Optional[List[EpisodeType]] = None,
        env_runner: Optional["EnvRunner"] = None,
        metrics_logger: Optional[MetricsLogger] = None,
        env: Optional[gym.Env] = None,
        env_index: int,
        rl_module: Optional[RLModule] = None,
        # TODO (sven): Deprecate these args.
        worker: Optional["EnvRunner"] = None,
        base_env: Optional[BaseEnv] = None,
        policies: Optional[Dict[PolicyID, Policy]] = None,
        **kwargs,
    ) -> None:
        rail_env: RailEnv = self._unwrap_rail_env(env)

        episode_done_agents = 0
        for agent in rail_env.agents:
            if agent.state == TrainState.DONE:
                episode_done_agents += 1

        # https://flatland-association.github.io/flatland-book/challenges/flatland3/eval.html
        num_agents = rail_env.get_num_agents()
        episode_rewards: Dict[str, List[float]] = episode.get_rewards()
        normalized_reward = sum([sum(agent_rewards) for agent_rewards in episode_rewards.values()]) / (
            rail_env._max_episode_steps *
            num_agents
        )

        metrics_logger.log_value(
            "normalized_reward",
            normalized_reward,
            reduce="sum",
        )

        percentage_complete = float(episode_done_agents) / num_agents
        metrics_logger.log_value(
            "percentage_complete",
            percentage_complete,
        )

    def _unwrap_rail_env(self, env: gym.Env) -> RailEnv:
        # If we have a vector env, only render the sub-env at index 0.
        if isinstance(env.unwrapped, (gym.vector.VectorEnv, VectorMultiAgentEnv)):
            unwrapped = env.unwrapped.envs[0]
        else:
            unwrapped = env.unwrapped
        while not isinstance(unwrapped, RayMultiAgentWrapper):
            unwrapped = unwrapped.unwrapped
        rail_env: RailEnv = unwrapped._wrap
        return rail_env
