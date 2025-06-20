import importlib
from pathlib import Path
from typing import Optional

import click
import tqdm

from flatland.callbacks.callbacks import FlatlandCallbacks, make_multi_callbacks
from flatland.core.env_observation_builder import ObservationBuilder
from flatland.core.policy import Policy
from flatland.env_generation.env_generator import env_generator
from flatland.envs.persistence import RailEnvPersister
from flatland.envs.rail_env import RailEnv
from flatland.evaluators.trajectory_evaluator import TrajectoryEvaluator
from flatland.trajectories.trajectories import Trajectory, SERIALISED_STATE_SUBDIR


class PolicyRunner:
    @staticmethod
    def create_from_policy(
            policy: Policy,
            data_dir: Path,
            env: RailEnv = None,
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
            seed=42,
            obs_builder: Optional[ObservationBuilder] = None,
            snapshot_interval: int = 1,
            ep_id: str = None,
            callbacks: FlatlandCallbacks = None,
            tqdm_kwargs: dict = None,
            start_step: int = 0,
            end_step: int = None,
            fork_from_trajectory: "Trajectory" = None,
    ) -> "Trajectory":
        """
        Creates trajectory by running submission (policy and obs builder).

        Always backs up the actions and positions for steps executed in the tsvs.
        Can start from existing trajectory.

        Parameters
        ----------
        policy : Policy
            the submission's policy
        data_dir : Path
            the path to write the trajectory to
        env: RailEnv
            directly inject env, skip env generation
        n_agents: int
            number of agents
        x_dim: int
            number of columns
        y_dim: int
            number of rows
        n_cities: int
           Max number of cities to build. The generator tries to achieve this numbers given all the parameters. Goes into `sparse_rail_generator`.
        max_rail_pairs_in_city: int
            Number of parallel tracks in the city. This represents the number of tracks in the train stations. Goes into `sparse_rail_generator`.
        grid_mode: bool
            How to distribute the cities in the path, either equally in a grid or random. Goes into `sparse_rail_generator`.
        max_rails_between_cities: int
            Max number of rails connecting to a city. This is only the number of connection points at city boarder.
        malfunction_duration_min: int
            Minimal duration of malfunction. Goes into `ParamMalfunctionGen`.
        malfunction_duration_max: int
            Max duration of malfunction. Goes into `ParamMalfunctionGen`.
        malfunction_interval: int
            Inverse of rate of malfunction occurrence. Goes into `ParamMalfunctionGen`.
        speed_ratios: Dict[float, float]
            Speed ratios of all agents. They are probabilities of all different speeds and have to add up to 1. Goes into `sparse_line_generator`. Defaults to `{1.0: 0.25, 0.5: 0.25, 0.33: 0.25, 0.25: 0.25}`.
        seed: int
             Initiate random seed generators. Goes into `reset`.
        obs_builder: Optional[ObservationBuilder]
            Defaults to `TreeObsForRailEnv(max_depth=3, predictor=ShortestPathPredictorForRailEnv(max_depth=50))`
        snapshot_interval : int
            interval to write pkl snapshots
        ep_id: str
            episode ID to store data under. If not provided, generate one.
        callbacks: FlatlandCallbacks
            callbacks to run during trajectory creation
        tqdm_kwargs: dict
            additional kwargs for tqdm
        start_step : int
            start evaluation from intermediate step incl. (requires snapshot to be present); take actions from start_step and first step executed is start_step + 1. Defaults to 0 with first elapsed step 1.
        end_step : int
            stop evaluation at intermediate step excl. Capped by env's max_episode_steps
        fork_from_trajectory : Trajectory
            copy data from this trajectory up to start step and run policy from there on
        Returns
        -------
        Trajectory

        """
        if ep_id is not None:
            trajectory = Trajectory(data_dir=data_dir, ep_id=ep_id)
        else:
            trajectory = Trajectory(data_dir=data_dir)

        trajectory.load()

        # ensure to start with new empty df to avoid inconsistencies:
        assert len(trajectory.trains_positions) == 0
        assert len(trajectory.actions) == 0
        assert len(trajectory.trains_arrived) == 0
        assert len(trajectory.trains_rewards_dones_infos) == 0

        if fork_from_trajectory is not None:
            env = fork_from_trajectory.restore_episode(start_step=start_step)
            fork_from_trajectory.load(episode_only=True)

            # will run action start_step into step start_step+1
            trajectory.actions = fork_from_trajectory.actions[fork_from_trajectory.actions["env_time"] < start_step]
            trajectory.trains_positions = fork_from_trajectory.trains_positions[fork_from_trajectory.trains_positions["env_time"] <= start_step]
            trajectory.trains_arrived = fork_from_trajectory.trains_arrived[fork_from_trajectory.trains_arrived["env_time"] <= start_step]
            trajectory.trains_rewards_dones_infos = fork_from_trajectory.trains_rewards_dones_infos[
                fork_from_trajectory.trains_rewards_dones_infos["env_time"] <= start_step]
            trajectory.actions["episode_id"] = trajectory.ep_id
            trajectory.trains_positions["episode_id"] = trajectory.ep_id
            trajectory.trains_arrived["episode_id"] = trajectory.ep_id
            trajectory.trains_rewards_dones_infos["episode_id"] = trajectory.ep_id

            trajectory.persist()

            if env is None:
                env = fork_from_trajectory.restore_episode()
                (trajectory.data_dir / SERIALISED_STATE_SUBDIR).mkdir(parents=True)
                RailEnvPersister.save(env, trajectory.data_dir / SERIALISED_STATE_SUBDIR / f"{trajectory.ep_id}.pkl")
                env = TrajectoryEvaluator(trajectory=trajectory, callbacks=callbacks).evaluate(end_step=start_step)
                trajectory.load()
                # TODO bad code smell - private method - check num resets?
                observations = env._get_observations()
        elif env is not None:
            # TODO bad code smell - private method - check num resets?
            observations = env._get_observations()
        else:
            env, observations, _ = env_generator(
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
                speed_ratios=speed_ratios,
                seed=seed,
                obs_builder_object=obs_builder)

        assert start_step == env._elapsed_steps, f"Expected env at {start_step}, found {env._elapsed_steps}."

        if tqdm_kwargs is None:
            tqdm_kwargs = {}

        (data_dir / SERIALISED_STATE_SUBDIR).mkdir(parents=True, exist_ok=True)
        RailEnvPersister.save(env, str(data_dir / SERIALISED_STATE_SUBDIR / f"{trajectory.ep_id}.pkl"))

        if snapshot_interval > 0:
            from flatland.trajectories.trajectory_snapshot_callbacks import TrajectorySnapshotCallbacks
            if callbacks is None:
                callbacks = TrajectorySnapshotCallbacks(trajectory, snapshot_interval=snapshot_interval, data_dir_override=data_dir)
            else:
                callbacks = make_multi_callbacks(callbacks,
                                                 TrajectorySnapshotCallbacks(trajectory, snapshot_interval=snapshot_interval, data_dir_override=data_dir))

        trajectory.outputs_dir.mkdir(exist_ok=True)

        n_agents = env.get_num_agents()
        assert len(env.agents) == n_agents

        env_time = start_step
        if end_step is None:
            end_step = env._max_episode_steps
        env_time_range = range(start_step, end_step)

        if callbacks is not None and start_step == 0:
            callbacks.on_episode_start(env=env, data_dir=trajectory.outputs_dir)

        for env_time in tqdm.tqdm(env_time_range, **tqdm_kwargs):
            assert env_time == env._elapsed_steps

            action_dict = policy.act_many(env.get_agent_handles(), observations)
            for handle, action in action_dict.items():
                trajectory.action_collect(env_time=env_time, agent_id=handle, action=action)

            observations, rewards, dones, infos = env.step(action_dict)

            for agent_id in range(n_agents):
                agent = env.agents[agent_id]
                actual_position = (agent.position, agent.direction)
                trajectory.position_collect(env_time=env_time + 1, agent_id=agent_id, position=actual_position)
                trajectory.rewards_dones_infos_collect(env_time=env_time + 1, agent_id=agent_id, reward=rewards.get(agent_id, 0.0),
                                                       info={k: v[agent_id] for k, v in infos.items()},
                                                       done=dones[agent_id])

            done = dones['__all__']

            if callbacks is not None:
                callbacks.on_episode_step(env=env, data_dir=trajectory.outputs_dir)

            if done:
                if callbacks is not None:
                    callbacks.on_episode_end(env=env, data_dir=trajectory.outputs_dir)
                break
        actual_success_rate = sum([agent.state == 6 for agent in env.agents]) / n_agents
        if done:
            trajectory.arrived_collect(env_time, actual_success_rate)
        trajectory.persist()
        return trajectory


@click.command()
@click.option('--data-dir',
              type=click.Path(exists=True, path_type=Path),
              help="Path to folder containing Flatland episode",
              required=True
              )
@click.option('--policy-pkg',
              type=str,
              help="Policy's fully qualified package name.",
              required=True
              )
@click.option('--policy-cls',
              type=str,
              help="Policy class name.",
              required=True
              )
@click.option('--obs-builder-pkg',
              type=str,
              help="Defaults to `TreeObsForRailEnv(max_depth=3, predictor=ShortestPathPredictorForRailEnv(max_depth=50))`",
              required=False,
              default=None
              )
@click.option('--obs-builder-cls',
              type=str,
              help="Defaults to `TreeObsForRailEnv(max_depth=3, predictor=ShortestPathPredictorForRailEnv(max_depth=50))`",
              required=False,
              default=None
              )
@click.option('--n_agents',
              type=int,
              help="Number of agents.",
              required=False,
              default=7)
@click.option('--x_dim',
              type=int,
              help="Number of columns.",
              required=False,
              default=30)
@click.option('--y_dim',
              type=int,
              help="Number of rows.",
              required=False,
              default=30)
@click.option('--n_cities',
              type=int,
              help="Max number of cities to build. The generator tries to achieve this numbers given all the parameters. Goes into `sparse_rail_generator`. ",
              required=False,
              default=2)
@click.option('--max_rail_pairs_in_city',
              type=int,
              help="Number of parallel tracks in the city. This represents the number of tracks in the train stations. Goes into `sparse_rail_generator`.",
              required=False,
              default=4)
@click.option('--grid_mode',
              type=bool,
              help="How to distribute the cities in the path, either equally in a grid or random. Goes into `sparse_rail_generator`.",
              required=False,
              default=False)
@click.option('--max_rails_between_cities',
              type=int,
              help="Max number of rails connecting to a city. This is only the number of connection points at city boarder.",
              required=False,
              default=2)
@click.option('--malfunction_duration_min',
              type=int,
              help="Minimal duration of malfunction. Goes into `ParamMalfunctionGen`.",
              required=False,
              default=20)
@click.option('--malfunction_duration_max',
              type=int,
              help="Max duration of malfunction. Goes into `ParamMalfunctionGen`.",
              required=False,
              default=50)
@click.option('--malfunction_interval',
              type=int,
              help="Inverse of rate of malfunction occurrence. Goes into `ParamMalfunctionGen`.",
              required=False,
              default=540)
@click.option('--speed_ratios',
              multiple=True,
              nargs=2,
              type=click.Tuple(types=[float, float]),
              help="Speed ratios of all agents. They are probabilities of all different speeds and have to add up to 1. Goes into `sparse_line_generator`. Defaults to `{1.0: 0.25, 0.5: 0.25, 0.33: 0.25, 0.25: 0.25}`.",
              required=False,
              default=None)
@click.option('--seed',
              type=int,
              help="Initiate random seed generators. Goes into `reset`.",
              required=False, default=42)
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
def generate_trajectory_from_policy(
        data_dir: Path,
        policy_pkg: str, policy_cls: str,
        obs_builder_pkg: str, obs_builder_cls: str,
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
        seed: int = 42,
        snapshot_interval: int = 1,
        ep_id: str = None,
        env_path: Path = None,
        start_step: int = 0,
        end_step: int = None,
        fork_data_dir: Path = None,
        fork_ep_id: str = None,
):
    module = importlib.import_module(policy_pkg)
    policy_cls = getattr(module, policy_cls)

    obs_builder = None
    if obs_builder_pkg is not None and obs_builder_cls is not None:
        module = importlib.import_module(obs_builder_pkg)
        obs_builder_cls = getattr(module, obs_builder_cls)
        obs_builder = obs_builder_cls()
    env = None
    if env_path is not None:
        env, _ = RailEnvPersister.load_new(str(env_path))
    fork_from_trajectory = None
    if fork_data_dir is not None and fork_ep_id is not None:
        fork_from_trajectory = Trajectory(data_dir=fork_data_dir, ep_id=fork_ep_id)
    PolicyRunner.create_from_policy(
        policy=policy_cls(),
        data_dir=data_dir,
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
        speed_ratios=dict(speed_ratios) if len(speed_ratios) > 0 else None,
        seed=seed,
        obs_builder=obs_builder,
        snapshot_interval=snapshot_interval,
        ep_id=ep_id,
        env=env,
        start_step=start_step,
        end_step=end_step,
        fork_from_trajectory=fork_from_trajectory,
    )
