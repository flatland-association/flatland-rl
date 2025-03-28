import ast
import importlib
import os
import uuid
from pathlib import Path
from typing import Optional, Tuple

import click
import pandas as pd
import tqdm
from attr import attrs, attrib

from flatland.callbacks.callbacks import FlatlandCallbacks, make_multi_callbacks
from flatland.core.env_observation_builder import ObservationBuilder
from flatland.core.policy import Policy
from flatland.env_generation.env_generator import env_generator
from flatland.envs.persistence import RailEnvPersister
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_env_action import RailEnvActions

EVENT_LOGS_SUBDIR = 'event_logs'
DISCRETE_ACTION_FNAME = os.path.join(EVENT_LOGS_SUBDIR, "ActionEvents.discrete_action.tsv")
TRAINS_ARRIVED_FNAME = os.path.join(EVENT_LOGS_SUBDIR, "TrainMovementEvents.trains_arrived.tsv")
TRAINS_POSITIONS_FNAME = os.path.join(EVENT_LOGS_SUBDIR, "TrainMovementEvents.trains_positions.tsv")
SERIALISED_STATE_SUBDIR = 'serialised_state'
OUTPUTS_SUBDIR = 'outputs'


def _uuid_str():
    return str(uuid.uuid4())


@attrs
class Trajectory:
    """
    Encapsulates episode data (actions, positions etc.) for one or multiple episodes for further analysis/evaluation.

    Aka. Episode
    Aka. Recording

    In contrast to rllib (https://github.com/ray-project/ray/blob/master/rllib/env/multi_agent_episode.py), we use a tabular approach (tsv-backed) instead of `dict`s.

    Directory structure:
    - event_logs
        ActionEvents.discrete_action 		 -- holds set of action to be replayed for the related episodes.
        TrainMovementEvents.trains_arrived 	 -- holds success rate for the related episodes.
        TrainMovementEvents.trains_positions -- holds the positions for the related episodes.
    - serialised_state
        <ep_id>.pkl                          -- Holds the pickled environment version for the episode.
    """
    data_dir = attrib(type=Path)
    ep_id = attrib(type=str, factory=_uuid_str)

    def read_actions(self):
        """Returns pd df with all actions for all episodes."""
        f = os.path.join(self.data_dir, DISCRETE_ACTION_FNAME)
        if not os.path.exists(f):
            return pd.DataFrame(columns=['episode_id', 'env_time', 'agent_id', 'action'])
        return pd.read_csv(f, sep='\t')

    def read_trains_arrived(self):
        """Returns pd df with success rate for all episodes."""
        f = os.path.join(self.data_dir, TRAINS_ARRIVED_FNAME)
        if not os.path.exists(f):
            return pd.DataFrame(columns=['episode_id', 'env_time', 'success_rate'])
        return pd.read_csv(f, sep='\t')

    def read_trains_positions(self) -> pd.DataFrame:
        """Returns pd df with all trains' positions for all episodes."""
        f = os.path.join(self.data_dir, TRAINS_POSITIONS_FNAME)
        if not os.path.exists(f):
            return pd.DataFrame(columns=['episode_id', 'env_time', 'agent_id', 'position'])
        return pd.read_csv(f, sep='\t')

    def write_trains_positions(self, df: pd.DataFrame):
        """Store pd df with all trains' positions for all episodes."""
        f = os.path.join(self.data_dir, TRAINS_POSITIONS_FNAME)
        Path(f).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(f, sep='\t', index=False)

    def write_actions(self, df: pd.DataFrame):
        """Store pd df with all trains' actions for all episodes."""
        f = os.path.join(self.data_dir, DISCRETE_ACTION_FNAME)
        Path(f).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(f, sep='\t', index=False)

    def write_trains_arrived(self, df: pd.DataFrame):
        """Store pd df with all trains' success rates for all episodes."""
        f = os.path.join(self.data_dir, TRAINS_ARRIVED_FNAME)
        Path(f).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(f, sep='\t', index=False)

    def restore_episode(self, start_step: int = None) -> RailEnv:
        """Restore an episode.

        Parameters
        ----------
        start_step : Optional[int]
            start from snapshot (if it exists)
        Returns
        -------
        RailEnv
            the episode
        """
        if start_step is None:
            f = os.path.join(self.data_dir, SERIALISED_STATE_SUBDIR, f'{self.ep_id}.pkl')
            env, _ = RailEnvPersister.load_new(f)
            return env
        else:
            env, _ = RailEnvPersister.load_new(os.path.join(self.data_dir, SERIALISED_STATE_SUBDIR, f"{self.ep_id}_step{start_step:04d}.pkl"))
            return env

    def position_collect(self, df: pd.DataFrame, env_time: int, agent_id: int, position: Tuple[Tuple[int, int], int]):
        df.loc[len(df)] = {'episode_id': self.ep_id, 'env_time': env_time, 'agent_id': agent_id, 'position': position}

    def action_collect(self, df: pd.DataFrame, env_time: int, agent_id: int, action: RailEnvActions):
        df.loc[len(df)] = {'episode_id': self.ep_id, 'env_time': env_time, 'agent_id': agent_id, 'action': action}

    def arrived_collect(self, df: pd.DataFrame, env_time: int, success_rate: float):
        df.loc[len(df)] = {'episode_id': self.ep_id, 'env_time': env_time, 'success_rate': success_rate}

    def position_lookup(self, df: pd.DataFrame, env_time: int, agent_id: int) -> Tuple[Tuple[int, int], int]:
        """Method used to retrieve the stored position (if available).

        Parameters
        ----------
        df: pd.DataFrame
            Data frame from ActionEvents.discrete_action.tsv
        env_time: int
            position before (!) step env_time
        agent_id: int
            agent ID
        Returns
        -------
        Tuple[Tuple[int, int], int]
            The position in the format ((row, column), direction).
        """
        pos = df.loc[(df['env_time'] == env_time) & (df['agent_id'] == agent_id) & (df['episode_id'] == self.ep_id)]['position']
        if len(pos) != 1:
            print(f"Found {len(pos)} positions for {self.ep_id} {env_time} {agent_id}")
            print(df[(df['agent_id'] == agent_id) & (df['episode_id'] == self.ep_id)]["env_time"])
        assert len(pos) == 1, f"Found {len(pos)} positions for {self.ep_id} {env_time} {agent_id}"
        iloc_ = pos.iloc[0]
        iloc_ = iloc_.replace("<Grid4TransitionsEnum.NORTH: 0>", "0").replace("<Grid4TransitionsEnum.EAST: 1>", "1").replace("<Grid4TransitionsEnum.SOUTH: 2>",
                                                                                                                             "2").replace(
            "<Grid4TransitionsEnum.WEST: 3>", "3")
        return ast.literal_eval(iloc_)

    def action_lookup(self, actions_df: pd.DataFrame, env_time: int, agent_id: int) -> RailEnvActions:
        """Method used to retrieve the stored action (if available). Defaults to 2 = MOVE_FORWARD.

        Parameters
        ----------
        actions_df: pd.DataFrame
            Data frame from ActionEvents.discrete_action.tsv
        env_time: int
            action going into step env_time
        agent_id: int
            agent ID
        Returns
        -------
        RailEnvActions
            The action to step the env.
        """
        action = actions_df.loc[
            (actions_df['env_time'] == env_time) &
            (actions_df['agent_id'] == agent_id) &
            (actions_df['episode_id'] == self.ep_id)
            ]['action'].to_numpy()
        if len(action) == 0:
            return RailEnvActions(2)
        return RailEnvActions(action[0])

    def trains_arrived_lookup(self, movements_df: pd.DataFrame) -> pd.Series:
        """Method used to retrieve the trains arrived for the episode.

        Parameters
        ----------
        movements_df: pd.DataFrame
            Data frame from event_logs/TrainMovementEvents.trains_arrived.tsv
        Returns
        -------
        pd.Series
            The trains arrived data.
        """
        movement = movements_df.loc[(movements_df['episode_id'] == self.ep_id)]

        if len(movement) == 1:
            return movement.iloc[0]
        raise

    @property
    def outputs_dir(self) -> Path:
        return self.data_dir / OUTPUTS_SUBDIR

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
        callbacks: FlatlandCallbacks = None
    ) -> "Trajectory":
        """
        Creates trajectory by running submission (policy and obs builder).

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

        Returns
        -------
        Trajectory

        """
        if env is not None:
            observations, _ = env.reset()
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
        if ep_id is not None:
            trajectory = Trajectory(data_dir=data_dir, ep_id=ep_id)
        else:
            trajectory = Trajectory(data_dir=data_dir)
        (data_dir / SERIALISED_STATE_SUBDIR).mkdir(parents=True, exist_ok=True)
        RailEnvPersister.save(env, str(data_dir / SERIALISED_STATE_SUBDIR / f"{trajectory.ep_id}.pkl"))

        if snapshot_interval > 0:
            from flatland.trajectories.trajectory_snapshot_callbacks import TrajectorySnapshotCallbacks
            if callbacks is None:
                callbacks = TrajectorySnapshotCallbacks(trajectory, snapshot_interval=snapshot_interval, data_dir_override=data_dir)
            else:
                callbacks = make_multi_callbacks(callbacks,
                                                 TrajectorySnapshotCallbacks(trajectory, snapshot_interval=snapshot_interval, data_dir_override=data_dir))

        trains_positions = trajectory.read_trains_positions()
        actions = trajectory.read_actions()
        trains_arrived = trajectory.read_trains_arrived()

        trajectory.outputs_dir.mkdir(exist_ok=True)
        if callbacks is not None:
            callbacks.on_episode_start(env=env, data_dir=trajectory.outputs_dir)
        n_agents = env.get_num_agents()
        assert len(env.agents) == n_agents
        env_time = 0

        for env_time in tqdm.tqdm(range(env._max_episode_steps)):

            action_dict = policy.act_many(env.get_agent_handles(), observations)
            for handle, action in action_dict.items():
                trajectory.action_collect(actions, env_time=env_time, agent_id=handle, action=action)

            _, _, dones, _ = env.step(action_dict)

            for agent_id in range(n_agents):
                agent = env.agents[agent_id]
                actual_position = (agent.position, agent.direction)
                trajectory.position_collect(trains_positions, env_time=env_time + 1, agent_id=agent_id, position=actual_position)
            done = dones['__all__']

            if callbacks is not None:
                callbacks.on_episode_step(env=env, data_dir=trajectory.outputs_dir)

            if done:
                break
        if callbacks is not None:
            callbacks.on_episode_end(env=env, data_dir=trajectory.outputs_dir)

        actual_success_rate = sum([agent.state == 6 for agent in env.agents]) / n_agents
        trajectory.arrived_collect(trains_arrived, env_time, actual_success_rate)
        trajectory.write_trains_positions(trains_positions)
        trajectory.write_actions(actions)
        trajectory.write_trains_arrived(trains_arrived)
        return trajectory


@click.command()
@click.option('--data-dir',
              type=click.Path(exists=True),
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
              type=click.Path(exists=True),
              help="Path to existing RailEnv to start trajectory from",
              required=False
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
    env_path: Path = None
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
    Trajectory.create_from_policy(
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
        env=env
    )


def generate_trajectories_from_metadata(
    metadata_csv: Path,
    data_dir: Path,
    policy_pkg: str, policy_cls: str,
    obs_builder_pkg: str, obs_builder_cls: str):
    metadata = pd.read_csv(metadata_csv)
    for k, v in metadata.iterrows():
        try:
            test_folder = data_dir / v["test_id"] / v["env_id"]
            test_folder.mkdir(parents=True, exist_ok=True)
            generate_trajectory_from_policy(
                ["--data-dir", test_folder,
                 "--policy-pkg", policy_pkg, "--policy-cls", policy_cls,
                 "--obs-builder-pkg", obs_builder_pkg, "--obs-builder-cls", obs_builder_cls,
                 "--n_agents", v["n_agents"],
                 "--x_dim", v["x_dim"],
                 "--y_dim", v["y_dim"],
                 "--n_cities", v["n_cities"],
                 "--max_rail_pairs_in_city", v["max_rail_pairs_in_city"],
                 "--grid_mode", v["grid_mode"],
                 "--max_rails_between_cities", v["max_rails_between_cities"],
                 "--malfunction_duration_min", v["malfunction_duration_min"],
                 "--malfunction_duration_max", v["malfunction_duration_max"],
                 "--malfunction_interval", v["malfunction_interval"],
                 "--speed_ratios", "1.0", "0.25",
                 "--speed_ratios", "0.5", "0.25",
                 "--speed_ratios", "0.33", "0.25",
                 "--speed_ratios", "0.25", "0.25",
                 "--seed", v["seed"],
                 "--snapshot-interval", 0,
                 "--ep-id", v["test_id"] + "_" + v["env_id"]
                 ])
        except SystemExit as exc:
            assert exc.code == 0


if __name__ == '__main__':
    metadata_csv = Path("../../episodes/malfunction_deadlock_avoidance_heuristics/metadata.csv")
    data_dir = Path("../../episodes/malfunction_deadlock_avoidance_heuristics")
    generate_trajectories_from_metadata(
        metadata_csv=metadata_csv,
        data_dir=data_dir,
        # TODO https://github.com/flatland-association/flatland-rl/issues/101 import heuristic baseline as example
        policy_pkg="src.policy.deadlock_avoidance_policy",
        policy_cls="DeadLockAvoidancePolicy",
        obs_builder_pkg="src.observation.full_state_observation",
        obs_builder_cls="FullStateObservationBuilder"
    )
