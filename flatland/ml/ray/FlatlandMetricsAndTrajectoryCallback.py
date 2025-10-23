import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Union, List

import gymnasium as gym
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.env.vector.vector_multi_agent_env import VectorMultiAgentEnv
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.policy import Policy
from ray.rllib.utils.metrics.metrics_logger import MetricsLogger
from ray.rllib.utils.typing import EpisodeType, PolicyID

from flatland.envs.rail_env import RailEnv
from flatland.envs.step_utils.states import TrainState
from flatland.evaluators.trajectory_evaluator import TrajectoryEvaluator
from flatland.ml.ray.FlatlandMetricsCallback import FlatlandMetricsCallback
from flatland.ml.ray.ray_multi_agent_rail_env import RayMultiAgentWrapper
from flatland.trajectories.trajectories import Trajectory, SERIALISED_STATE_SUBDIR


class FlatlandMetricsAndTrajectoryCallback(FlatlandMetricsCallback):
    """
    In addition to `FlatlandMetricsCallback`, creates trajectory for the episodes.
    """

    def on_episode_start(
        self,
        *,
        episode: Union[EpisodeType, EpisodeV2],
        env_runner: Optional["EnvRunner"] = None,
        metrics_logger: Optional[MetricsLogger] = None,
        env: Optional[gym.Env] = None,
        env_index: int,
        rl_module: Optional[RLModule] = None,
        worker: Optional["EnvRunner"] = None,
        base_env: Optional[BaseEnv] = None,
        policies: Optional[Dict[PolicyID, Policy]] = None,
        **kwargs, ) -> None:

        rail_env = self._unwrap_rail_env(env)
        data_dir = Path("trajectories") / episode.id_
        (data_dir / SERIALISED_STATE_SUBDIR).mkdir(parents=True)
        trajectory = Trajectory(data_dir=data_dir, ep_id=episode.id_)
        trajectory.load()
        trajectory.save_initial(rail_env)
        assert rail_env._elapsed_steps == 0

        episode.custom_data["trajectory"] = trajectory
        episode.custom_data["dones"] = defaultdict(dict)

        # N.B. initial position not stored in trajectory.

    def on_episode_step(
        self,
        *,
        episode: Union[EpisodeType, EpisodeV2],
        env_runner: Optional["EnvRunner"] = None,
        metrics_logger: Optional[MetricsLogger] = None,
        env: Optional[gym.Env] = None,
        env_index: int,
        rl_module: Optional[RLModule] = None,
        worker: Optional["EnvRunner"] = None,
        base_env: Optional[BaseEnv] = None,
        policies: Optional[Dict[PolicyID, Policy]] = None,
        **kwargs,
    ) -> None:
        rail_env: RailEnv = self._unwrap_rail_env(env)
        trajectory = episode.custom_data["trajectory"]
        dones = episode.custom_data["dones"]
        # N.B. initial position not stored in trajectory.
        for agent in rail_env.agents:
            trajectory.position_collect(rail_env._elapsed_steps, agent.handle, (agent.position, agent.direction))
            dones[rail_env._elapsed_steps][agent.handle] = rail_env.dones[agent.handle]

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
        worker: Optional["EnvRunner"] = None,
        base_env: Optional[BaseEnv] = None,
        policies: Optional[Dict[PolicyID, Policy]] = None,
        **kwargs,
    ) -> None:
        # metrics logging
        super().on_episode_end(episode=episode, env=env, env_index=env_index, metrics_logger=metrics_logger)

        rail_env = self._unwrap_rail_env(env)

        episode_done_agents = 0
        for agent in rail_env.agents:
            if agent.state == TrainState.DONE:
                episode_done_agents += 1

        episode_done_agents = 0
        for agent in rail_env.agents:
            if agent.state == TrainState.DONE:
                episode_done_agents += 1

        # https://flatland-association.github.io/flatland-book/challenges/flatland3/eval.html
        num_agents = rail_env.get_num_agents()
        percentage_complete = float(episode_done_agents) / num_agents

        episode_length = rail_env._elapsed_steps
        episode_rewards = episode.get_rewards()
        episode_infos = episode.get_infos()
        episode_actions = episode.get_actions()

        trajectory: Trajectory = episode.custom_data["trajectory"]
        dones = episode.custom_data["dones"]

        trajectory.arrived_collect(episode_length, percentage_complete)
        for agent_id in rail_env.get_agent_handles():
            for env_time, action in enumerate(episode_actions[str(agent_id)]):
                trajectory.action_collect(env_time, agent_id, action)

            # info from reset - we do not store it in trajectory.
            agent_episode_length = len(episode_rewards[str(agent_id)])
            assert agent_episode_length == len(episode_infos[str(agent_id)][1:])
            last_info = None

            for env_time, (rewards, infos) in enumerate(zip(episode_rewards[str(agent_id)], episode_infos[str(agent_id)][1:])):
                last_info = infos
                trajectory.rewards_dones_infos_collect(
                    env_time + 1,
                    agent_id,
                    rewards,
                    infos,
                    dones[env_time + 1][agent_id]
                )
            # comply with policy runner behaviour (which continues collecting after single agent end)
            for env_time in range(agent_episode_length + 1, episode_length + 1):
                trajectory.rewards_dones_infos_collect(
                    env_time,
                    agent_id,
                    0,
                    last_info,
                    True
                )
        trajectory.persist()
        TrajectoryEvaluator(trajectory).evaluate(tqdm_kwargs={"disable": True})
        shutil.make_archive(trajectory.data_dir.name, "zip", base_dir=str(trajectory.data_dir))
        with Path(f"{trajectory.data_dir.name}.zip").open("rb") as f:
            metrics_logger.log_value(
                "trajectory",
                f.read().hex(),
                reduce=None,
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
