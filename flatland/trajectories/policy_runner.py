import os
import sys
from pathlib import Path
from typing import Tuple, Any, Optional

import click
import numpy as np
import tqdm

from flatland.callbacks.callbacks import FlatlandCallbacks, make_multi_callbacks
from flatland.core.effects_generator import EffectsGenerator
from flatland.core.env_observation_builder import ObservationBuilder
from flatland.core.policy import Policy
from flatland.env_generation.env_generator import env_generator, env_generator_legacy
from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.persistence import RailEnvPersister
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.envs.rail_env import RailEnv, AbstractRailEnv
from flatland.envs.rewards import DefaultRewards, Rewards
from flatland.trajectories.trajectories import Trajectory
from flatland.utils.cli_utils import resolve_type


class PolicyRunner:
    def __init__(self,
                 policy: Policy,
                 trajectory: Trajectory,
                 by_pass_env: RailEnv = None,
                 callbacks: Optional[FlatlandCallbacks] = None,
                 end_step=None,
                 obs_builder: Optional[ObservationBuilder[Any, RailEnv]] = None,
                 rewards: Optional["Rewards"] = None,
                 effects_generator: Optional[EffectsGenerator[RailEnv]] = None,
                 ):
        self._policy = policy
        trajectory_env_time = trajectory.trains_rewards_dones_infos["env_time"].max()
        trajectory_env_time = 0 if np.isnan(trajectory_env_time) else trajectory_env_time
        if by_pass_env is not None:
            self.env = by_pass_env = by_pass_env
            assert by_pass_env._elapsed_steps == trajectory_env_time, \
                f"Expected env at {trajectory_env_time}, found {by_pass_env._elapsed_steps}."
        else:
            self.env = trajectory.load_env(trajectory_env_time, obs_builder=obs_builder, rewards=rewards, effects_generator=effects_generator)

        self.trajectory = trajectory
        # TODO extract to public interface?
        self.observations = self.env._get_observations()
        self.callbacks = callbacks
        self.n_agents = self.env.get_num_agents()
        self.end_step = end_step if end_step is not None else self.env._max_episode_steps
        self.done = False

    @property
    def policy(self):
        return self._policy

    def fork(self):
        pass

    @property
    def env_time(self):
        return self.env._elapsed_steps

    def change_policy(self, policy: Policy, obs_builder: ObservationBuilder):
        self._policy = policy
        self.env.obs_builder = obs_builder
        self.env.obs_builder.set_env(self.env)
        self.observations = self.env._get_observations()

    def step(self, persist: bool = False) -> Tuple["Trajectory", bool]:
        """Execute one environment step. Returns (trajectory, done)."""
        env_time = self.env_time
        assert env_time == self.env._elapsed_steps

        action_dict = self._policy.act_many(self.env.get_agent_handles(), observations=list(self.observations.values()))
        for handle, action in action_dict.items():
            self.trajectory.action_collect(env_time=env_time, agent_id=handle, action=action)

        self.observations, rewards, dones, infos = self.env.step(action_dict)

        for agent_id in range(self.n_agents):
            agent = self.env.agents[agent_id]
            self.trajectory.position_collect(env_time=env_time + 1, agent_id=agent_id, position=agent.current_configuration)
            self.trajectory.rewards_dones_infos_collect(env_time=env_time + 1, agent_id=agent_id, reward=rewards.get(agent_id, 0.0),
                                                        info={k: v[agent_id] for k, v in infos.items()},
                                                        done=dones[agent_id])

        self.done = dones['__all__']

        if self.callbacks is not None:
            self.callbacks.on_episode_step(env=self.env, data_dir=self.trajectory.outputs_dir)

        if self.done:
            if self.callbacks is not None:
                self.callbacks.on_episode_end(env=self.env, data_dir=self.trajectory.outputs_dir)
            actual_success_rate = sum([agent.state == 6 for agent in self.env.agents]) / self.n_agents
            # not persisted yet, need to get df from collected buffer
            collected_rewards = self.trajectory._collected_trains_rewards_dones_infos_to_df()["reward"]
            normalized_reward = self.env.rewards.normalize(*collected_rewards, max_episode_steps=self.env._max_episode_steps,
                                                           num_agents=self.env.get_num_agents())
            self.trajectory.arrived_collect(env_time, actual_success_rate, normalized_reward)

        if persist:
            self.trajectory.persist()
        return self.trajectory, self.done

    @staticmethod
    def create_from_policy(
        policy: Policy,
        data_dir: Path,
        env: AbstractRailEnv = None,
        snapshot_interval: int = 1,
        ep_id: str = None,
        callbacks: FlatlandCallbacks = None,
        tqdm_kwargs: dict = None,
        end_step: int = None,
        no_save: bool = False,
    ) -> Trajectory:
        trajectory = Trajectory.create_empty(data_dir=data_dir, env=env if not no_save else None, ep_id=ep_id, )
        return PolicyRunner.resume(
            policy=policy,
            trajectory=trajectory,
            by_pass_env=env if no_save else None,
            snapshot_interval=snapshot_interval,
            callbacks=callbacks,
            tqdm_kwargs=tqdm_kwargs,
            end_step=end_step,
            rewards=env.rewards,
            obs_builder=env.obs_builder,
            effects_generator=env.effects_generator,
        )

    @staticmethod
    def resume(
        policy: Policy,
        trajectory: Trajectory,
        by_pass_env: RailEnv = None,
        snapshot_interval: int = 1,
        callbacks: FlatlandCallbacks = None,
        tqdm_kwargs: dict = None,
        end_step: int = None,
        obs_builder: Optional[ObservationBuilder[Any, RailEnv]] = None,
        rewards: Optional[Rewards] = None,
        effects_generator: Optional[EffectsGenerator[RailEnv]] = None,
    ) -> Trajectory:
        """
        Creates trajectory by running submission (policy and obs builder).

        Always backs up the actions and positions for steps executed in the tsvs.
        Can start from existing trajectory.

        Parameters
        ----------
        policy : Policy
            the submission's policy
        trajectory : Trajectory
        by_pass_env : RailEnv
            Deprecated: pass env directly to avoid loading env (graph envs do not support env persistence yet).
        snapshot_interval : int
            interval to write pkl snapshots
        callbacks: FlatlandCallbacks
            callbacks to run during trajectory creation
        tqdm_kwargs: dict
            additional kwargs for tqdm
        end_step : int
            stop evaluation at intermediate step excl. Capped by env's max_episode_steps
        Returns
        -------
        Trajectory

        """
        if tqdm_kwargs is None:
            tqdm_kwargs = {}

        if snapshot_interval > 0:
            from flatland.trajectories.trajectory_snapshot_callbacks import TrajectorySnapshotCallbacks
            if callbacks is None:
                callbacks = TrajectorySnapshotCallbacks(trajectory, snapshot_interval=snapshot_interval, data_dir_override=trajectory.data_dir)
            else:
                callbacks = make_multi_callbacks(callbacks,
                                                 TrajectorySnapshotCallbacks(trajectory, snapshot_interval=snapshot_interval,
                                                                             data_dir_override=trajectory.data_dir))

        runner = PolicyRunner(
            policy=policy,
            trajectory=trajectory,
            by_pass_env=by_pass_env,
            callbacks=callbacks,
            end_step=end_step,
            effects_generator=effects_generator,
            rewards=rewards,
            obs_builder=obs_builder
        )
        for _ in tqdm.tqdm(range(runner.env_time, runner.end_step), **tqdm_kwargs):
            if callbacks is not None and runner.env_time == 0:
                callbacks.on_episode_start(env=runner.env, data_dir=trajectory.outputs_dir)
            _, done = runner.step()
            if done:
                break
        runner.trajectory.persist()
        return runner.trajectory


@click.command()
@click.option('--data-dir',
              type=click.Path(exists=True, path_type=Path),
              help="Path to folder containing Flatland episode",
              required=True
              )
@click.option('--policy',
              type=str,
              help=" Policy's fully qualified name. Can also be provided through env var POLICY (command-line option takes priority).",
              required=False,
              default=None,
              )
@click.option('--policy-pkg',
              type=str,
              help="DEPRECATED: use --policy instead. Policy's fully qualified package name. Can also be provided through env var POLICY_PKG (command-line option takes priority).",
              required=False,
              default=None,
              )
@click.option('--policy-cls',
              type=str,
              help="DEPRECATED: use --policy instead. Policy class name. Can also be provided through env var POLICY_CLS  (command-line option takes priority).",
              required=False,
              default=None,
              )
@click.option('--obs-builder',
              type=str,
              help="Can also be provided through env var OBS_BUILDER (command-line option takes priority). Defaults to `TreeObsForRailEnv(max_depth=3, predictor=ShortestPathPredictorForRailEnv(max_depth=50))`",
              required=False,
              default=None,
              )
@click.option('--obs-builder-pkg',
              type=str,
              help="DEPRECATED: use --obs-builder instead. Can also be provided through env var OBS_BUILDER_PKG. Defaults to `TreeObsForRailEnv(max_depth=3, predictor=ShortestPathPredictorForRailEnv(max_depth=50))`",
              required=False,
              default=None,
              )
@click.option('--obs-builder-cls',
              type=str,
              help="DEPRECATED: use --obs-builder instead. Can also be provided through env var OBS_BUILDER_CLS. Defaults to `TreeObsForRailEnv(max_depth=3, predictor=ShortestPathPredictorForRailEnv(max_depth=50))`",
              required=False,
              default=None,
              )
@click.option('--rewards',
              type=str,
              help="Defaults to `flatland.envs.rewards.DefaultRewards`. Can also be provided through env var REWARDS (command-line option takes priority).",
              required=False,
              default=None,
              )
@click.option('--rewards-pkg',
              type=str,
              help="DEPRECATED: use --rewards instead. Defaults to `flatland.envs.rewards.DefaultRewards`. Can also be provided through env var REWARDS_PKG (command-line option takes priority).",
              required=False,
              default=None,
              )
@click.option('--rewards-cls',
              type=str,
              help="DEPRECATED: use --rewards instead. Defaults to `flatland.envs.rewards.DefaultRewards. Can also be provided through env var REWARDS_CLS (command-line option takes priority).",
              required=False,
              default=None,
              )
@click.option('--n-agents',
              type=int,
              help="Number of agents.",
              required=False,
              default=7)
@click.option('--x-dim',
              type=int,
              help="Number of columns.",
              required=False,
              default=30)
@click.option('--y-dim',
              type=int,
              help="Number of rows.",
              required=False,
              default=30)
@click.option('--n-cities',
              type=int,
              help="Max number of cities to build. The generator tries to achieve this numbers given all the parameters. Goes into `sparse_rail_generator`. ",
              required=False,
              default=2)
@click.option('--max-rail-pairs-in-city',
              type=int,
              help="Number of parallel tracks in the city. This represents the number of tracks in the train stations. Goes into `sparse_rail_generator`.",
              required=False,
              default=4)
@click.option('--grid-mode',
              type=bool,
              help="How to distribute the cities in the path, either equally in a grid or random. Goes into `sparse_rail_generator`.",
              required=False,
              default=False)
@click.option('--max-rails-between-cities',
              type=int,
              help="Max number of rails connecting to a city. This is only the number of connection points at city boarder.",
              required=False,
              default=2)
@click.option('--malfunction-duration-min',
              type=int,
              help="Minimal duration of malfunction. Goes into `ParamMalfunctionGen`.",
              required=False,
              default=20)
@click.option('--malfunction-duration-max',
              type=int,
              help="Max duration of malfunction. Goes into `ParamMalfunctionGen`.",
              required=False,
              default=50)
@click.option('--malfunction-interval',
              type=int,
              help="Inverse of rate of malfunction occurrence. Goes into `ParamMalfunctionGen`.",
              required=False,
              default=540)
@click.option('--speed-ratios',
              multiple=True,
              nargs=2,
              type=click.Tuple(types=[float, float]),
              help="Speed ratios of all agents. They are probabilities of all different speeds and have to add up to 1. Goes into `sparse_line_generator`. Defaults to `{1.0: 0.25, 0.5: 0.25, 0.33: 0.25, 0.25: 0.25}`.",
              required=False,
              default=None)
@click.option('--seed',
              type=int,
              help="Initiate random seed generators. Goes into `reset`",
              required=False, default=None)
@click.option('--effects-generator',
              type=str,
              help="Use to override options for `ParamMalfunctionGen`. Defaults to `None`. Can also be provided through env var EFFECTS_GENERATOR (command-line option takes priority).",
              required=False,
              default=None
              )
@click.option('--effects-generator-pkg',
              type=str,
              help="DEPRECATED: use --effects-generator instead. Use to override options for `ParamMalfunctionGen`. Defaults to `None`. Can also be provided through env var EFFECTS_GENERATOR_PKG (command-line option takes priority).",
              required=False,
              default=None
              )
@click.option('--effects-generator-cls',
              type=str,
              help="DEPRECATED: use --effects-generator instead. Use to override options for `ParamMalfunctionGen`. Defaults to `None`. Can also be provided through env var EFFECTS_GENERATOR_CLS (command-line option takes priority).",
              required=False,
              default=None
              )
@click.option('--effects-generator-kwargs',
              multiple=True,
              nargs=2,
              type=click.Tuple(types=[str, str]),
              help="Keyworard args passed to effects generator.",
              required=False,
              default=None)
@click.option('--snapshot-interval',
              type=int,
              help="Interval to right snapshots. Use 0 to switch off, 1 for every step, ....",
              required=False,
              default=1)
@click.option('--ep-id',
              type=str,
              help="Set the episode ID used - if not set, a UUID will be sampled.",
              required=False)
@click.option('--env-path',
              type=click.Path(exists=True, path_type=Path),
              help="Path to existing RailEnv to start trajectory from",
              required=False
              )
@click.option('--start-step',
              type=int,
              help="Path to existing RailEnv to start trajectory from",
              required=False, default=0
              )
@click.option('--end-step',
              type=int,
              help="Path to existing RailEnv to start trajectory from",
              required=False, default=None
              )
@click.option('--fork-data-dir',
              type=click.Path(exists=True, path_type=Path),
              help="Path to existing RailEnv to start trajectory from",
              required=False, default=None
              )
@click.option('--fork-ep-id',
              type=int,
              help="Path to existing RailEnv to start trajectory from",
              required=False, default=None
              )
@click.option('--callbacks',
              type=str,
              help="Pass FlatlandCallbacks during policy run. Defaults to `None`. Can also be provided through env var CALLBACKS (command-line option takes priority).",
              required=False,
              default=None
              )
@click.option('--callbacks-pkg',
              type=str,
              help="Pass FlatlandCallbacks during policy run. Defaults to `None`. Can also be provided through env var CALLBACKS_PKG (command-line option takes priority).",
              required=False,
              default=None
              )
@click.option('--callbacks-cls',
              type=str,
              help="Pass FlatlandCallbacks during policy run. Defaults to `None`. Can also be provided through env var CALLBACKS_CLS (command-line option takes priority).",
              required=False,
              default=None
              )
@click.option('--legacy-env-generator',
              type=bool,
              default=False,
              help="DEPRECATED: use the patched env_generator. Keep only for regression tests. Update tests and drop in separate pr.",
              required=False
              )
@click.option('--post-seed',
              type=int,
              help="Initiate random seed after the env is generated, goes into second `reset` with `regenerate_rail=False, regenerate_schedule=False`.",
              required=False, default=None)
def generate_trajectory_from_policy(
    data_dir: Path,
    policy: str = None,
    policy_pkg: str = None,
    policy_cls: str = None,
    obs_builder: str = None,
    obs_builder_pkg: str = None,
    obs_builder_cls: str = None,
    rewards: str = None,
    rewards_pkg: str = None,
    rewards_cls: str = None,
    n_agents=7,
    x_dim=30,
    y_dim=30,
    n_cities=2,
    max_rail_pairs_in_city=4,
    grid_mode=False,
    max_rails_between_cities=2,
    malfunction_duration_min=20,
    malfunction_duration_max=50,
    malfunction_interval=540,
    speed_ratios=None,
    seed: int = None,
    effects_generator: str = None,
    effects_generator_pkg: str = None,
    effects_generator_cls: str = None,
    effects_generator_kwargs: str = None,
    snapshot_interval: int = 1,
    ep_id: str = None,
    env_path: Path = None,
    start_step: int = 0,
    end_step: int = None,
    fork_data_dir: Path = None,
    fork_ep_id: str = None,
    callbacks: str = None,
    callbacks_pkg: str = None,
    callbacks_cls: str = None,
    legacy_env_generator: bool = False,
    post_seed: int = None,
):
    if policy is None:
        policy = os.environ.get("POLICY", None)
    if policy_pkg is None:
        policy_pkg = os.environ.get("POLICY_PKG", None)
    if policy_cls is None:
        policy_cls = os.environ.get("POLICY_CLS", None)
    policy_cls = resolve_type(policy, policy_pkg, policy_cls)

    if obs_builder is None:
        obs_builder = os.environ.get("OBS_BUILDER", None)
    if obs_builder_pkg is None:
        obs_builder_pkg = os.environ.get("OBS_BUILDER_PKG", None)
    if obs_builder_cls is None:
        obs_builder_cls = os.environ.get("OBS_BUILDER_CLS", None)

    obs_builder = resolve_type(obs_builder, obs_builder_pkg, obs_builder_cls)
    if obs_builder is None:
        obs_builder = TreeObsForRailEnv(max_depth=3, predictor=ShortestPathPredictorForRailEnv())
    else:
        obs_builder = obs_builder()

    if rewards is None:
        rewards = os.environ.get("REWARDS", None)
    if rewards_pkg is None:
        rewards_pkg = os.environ.get("REWARDS_PKG", None)
    if rewards_cls is None:
        rewards_cls = os.environ.get("REWARDS_CLS", None)
    rewards = resolve_type(rewards, rewards_pkg, rewards_cls) or DefaultRewards
    rewards = rewards()

    if effects_generator is None:
        effects_generator = os.environ.get("EFFECTS_GENERATOR", None)
    if effects_generator_pkg is None:
        effects_generator_pkg = os.environ.get("EFFECTS_GENERATOR_PKG", None)
    if effects_generator_cls is None:
        effects_generator_cls = os.environ.get("EFFECTS_GENERATOR_CLS", None)
    effects_generator_kwargs = dict(effects_generator_kwargs) if len(effects_generator_kwargs) > 0 else {}
    effects_generator = resolve_type(effects_generator, effects_generator_pkg, effects_generator_cls)
    if effects_generator is not None:
        effects_generator = effects_generator(**effects_generator_kwargs)

    if callbacks is None:
        callbacks = os.environ.get("CALLBACKS", None)
    if callbacks_pkg is None:
        callbacks_pkg = os.environ.get("CALLBACKS_PKG", None)
    if callbacks_cls is None:
        callbacks_cls = os.environ.get("CALLBACKS_CLS", None)
    callbacks = resolve_type(callbacks, callbacks_pkg, callbacks_cls)
    if callbacks is not None:
        callbacks = callbacks()

    if env_path is not None:
        env, _ = RailEnvPersister.load_new(str(env_path), obs_builder=obs_builder, rewards=rewards, effects_generator=effects_generator)
        assert seed is None, "When using --env-path, use --post-seed to do a reset on the loaded env instead. Currently, loaded will always load the same rail/line/timetable from file."
        if post_seed is not None:
            env.reset(random_seed=post_seed)
        # TODO https://github.com/flatland-association/flatland-rl/issues/278 a bit hacky for now, clean up later...
        if malfunction_interval == -1 and effects_generator is not None:
            env.effects_generator = effects_generator
    elif legacy_env_generator:
        env, _, _ = env_generator_legacy(
            n_agents=n_agents,
            x_dim=x_dim,
            y_dim=y_dim,
            n_cities=n_cities,
            max_rail_pairs_in_city=max_rail_pairs_in_city,
            grid_mode=grid_mode,
            max_rails_between_cities=max_rails_between_cities,
            malfunction_duration_min=malfunction_duration_min,
            malfunction_duration_max=malfunction_duration_max,
            malfunction_interval=malfunction_interval,
            effects_generator=effects_generator,
            speed_ratios=dict(speed_ratios) if len(speed_ratios) > 0 else None,
            seed=seed,
            obs_builder_object=obs_builder,
            rewards=rewards,
            post_seed=post_seed,
        )
    else:
        env, _, _ = env_generator(
            n_agents=n_agents,
            x_dim=x_dim,
            y_dim=y_dim,
            n_cities=n_cities,
            max_rail_pairs_in_city=max_rail_pairs_in_city,
            grid_mode=grid_mode,
            max_rails_between_cities=max_rails_between_cities,
            malfunction_duration_min=malfunction_duration_min,
            malfunction_duration_max=malfunction_duration_max,
            malfunction_interval=malfunction_interval,
            effects_generator=effects_generator,
            speed_ratios=dict(speed_ratios) if len(speed_ratios) > 0 else None,
            seed=seed,
            obs_builder_object=obs_builder,
            rewards=rewards,
            post_seed=post_seed,
        )

    if fork_data_dir is not None and fork_ep_id is not None:
        trajectory = Trajectory.load_existing(data_dir=fork_data_dir, ep_id=fork_ep_id).fork(data_dir=data_dir, ep_id=fork_ep_id, start_step=start_step, )
    else:
        trajectory = Trajectory.create_empty(data_dir=data_dir, ep_id=ep_id, env=env)

    PolicyRunner.resume(
        policy=policy_cls(),
        trajectory=trajectory,
        snapshot_interval=snapshot_interval,
        end_step=end_step,
        callbacks=callbacks,
        effects_generator=effects_generator,
        rewards=rewards,
        obs_builder=obs_builder,
    )


if __name__ == "__main__":
    sys.exit(generate_trajectory_from_policy())  # pragma: no cover
