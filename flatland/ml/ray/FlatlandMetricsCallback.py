import gymnasium as gym
from ray.rllib.callbacks.callbacks import RLlibCallback
from ray.rllib.env.multi_agent_episode import MultiAgentEpisode
from ray.rllib.env.vector.vector_multi_agent_env import VectorMultiAgentEnv

from flatland.envs.rail_env import RailEnv
from flatland.ml.ray.ray_multi_agent_rail_env import RayMultiAgentWrapper


# See also for https://gitlab.aicrowd.com/flatland/neurips2020-flatland-baselines/-/blob/master/train.py?ref_type=heads
class FlatlandMetricsCallback(RLlibCallback):
    """
    Add `normalized_reward` and `percentage_complete` evaluation metrics.
    """

    def on_episode_end(
        self,
        *,
        episode: MultiAgentEpisode,
        env_runner,
        metrics_logger,
        env,
        env_index,
        rl_module,
        **kwargs,
    ) -> None:
        # If we have a vector env, only render the sub-env at index 0.
        if isinstance(env.unwrapped, (gym.vector.VectorEnv, VectorMultiAgentEnv)):
            unwrapped = env.unwrapped.envs[0]
        else:
            unwrapped = env.unwrapped
        while not isinstance(unwrapped, RayMultiAgentWrapper):
            unwrapped = unwrapped.unwrapped
        rail_env: RailEnv = unwrapped._wrap

        rewards_dict = episode.get_rewards(-1)
        episode.get_state()
        episode_done_agents = 0
        for h in rail_env.get_agent_handles():
            if rail_env.dones[h]:
                episode_done_agents += 1

        episode_num_agents = len(rewards_dict)
        assert episode_num_agents == rail_env.get_num_agents()
        assert sum(list(rewards_dict.values())) == sum(list(rail_env.rewards_dict.values()))
        # https://flatland-association.github.io/flatland-book/challenges/flatland3/eval.html
        normalized_reward = sum(list(rewards_dict.values())) / (
            rail_env._max_episode_steps *
            episode_num_agents
        )

        metrics_logger.log_value(
            "normalized_reward",
            normalized_reward,
            reduce="sum",
        )

        percentage_complete = float(episode_done_agents) / episode_num_agents
        metrics_logger.log_value(
            "percentage_complete",
            percentage_complete,
        )
