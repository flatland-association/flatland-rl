import ast
import importlib
import os
import uuid
from pathlib import Path
from typing import Optional, Any, Tuple

import click
import pandas as pd
import tqdm
from attr import attrs, attrib

from flatland.core.env_observation_builder import ObservationBuilder
from flatland.env_generation.env_generator import env_generator
from flatland.envs.persistence import RailEnvPersister
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_env_action import RailEnvActions
from flatland.utils.rendertools import RenderTool

DISCRETE_ACTION_FNAME = "event_logs/ActionEvents.discrete_action.tsv"
TRAINS_ARRIVED_FNAME = "event_logs/TrainMovementEvents.trains_arrived.tsv"
TRAINS_POSITIONS_FNAME = "event_logs/TrainMovementEvents.trains_positions.tsv"
SERIALISED_STATE_SUBDIR = 'serialised_state'


class Policy:
    def act(self, handle: int, observation: Any, **kwargs) -> RailEnvActions:
        pass


def _uuid_str():
    return str(uuid.uuid4())


@attrs
class Trajectory:
    """
    Aka. Episode
    Aka. Recording

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

    def evaluate(self, start_step: int = None, rendering=False, snapshot_interval=0):
        """
        The data is structured as follows:
            -30x30 map
                Contains the data to replay the episodes.
                - <n>_trains                                 -- for n in 10,15,20,50
                    - event_logs
                        ActionEvents.discrete_action 		 -- holds set of action to be replayed for the related episodes.
                        TrainMovementEvents.trains_arrived 	 -- holds success rate for the related episodes.
                        TrainMovementEvents.trains_positions -- holds the positions for the related episodes.
                    - serialised_state
                        <ep_id>.pkl                          -- Holds the pickled environment version for the episode.

        All these episodes are with constant speed of 1 and malfunctions free.
         Parameters
        ----------
        start_step : int
            start evaluation from intermediate step (requires snapshot to be present)
        rendering : bool
            render while evaluating
        snapshot_interval : int
            interval to write pkl snapshots. 1 means at every step. 0 means never.
        """

        trains_positions = self.read_trains_positions()
        actions = self.read_actions()
        trains_arrived = self.read_trains_arrived()

        env = self.restore_episode(start_step)
        n_agents = env.get_num_agents()
        assert len(env.agents) == n_agents
        if start_step is None:
            start_step = 0

        if rendering:
            renderer = RenderTool(env)
            renderer.render_env(show=True, frames=False, show_observations=False)

        if rendering:
            renderer.render_env(show=True, show_observations=True)
        for env_time in tqdm.tqdm(range(start_step, env._max_episode_steps)):

            if snapshot_interval > 0 and env_time % snapshot_interval == 0:
                RailEnvPersister.save(env, os.path.join(self.data_dir, SERIALISED_STATE_SUBDIR, f"{self.ep_id}_step{env_time:04d}.pkl"))
            action = {agent_id: self.action_lookup(actions, env_time=env_time, agent_id=agent_id) for agent_id in range(n_agents)}
            _, _, dones, _ = env.step(action)
            done = dones['__all__']

            if rendering:
                renderer.render_env(show=True, show_observations=True)

            for agent_id in range(n_agents):
                agent = env.agents[agent_id]
                actual_position = (agent.position, agent.direction)
                expected_position = self.position_lookup(trains_positions, env_time=env_time + 1, agent_id=agent_id)
                assert actual_position == expected_position, (self.data_dir, self.ep_id, env_time + 1, agent_id, actual_position, expected_position)

            if done:
                break

        trains_arrived_episode = self.trains_arrived_lookup(trains_arrived)
        expected_success_rate = trains_arrived_episode['success_rate']
        actual_success_rate = sum([agent.state == 6 for agent in env.agents]) / n_agents
        print(f"{actual_success_rate * 100}% trains arrived. Expected {expected_success_rate * 100}%.")
        assert expected_success_rate == actual_success_rate

    @staticmethod
    def create_from_policy(
            policy: Policy,
            data_dir: Path,
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
            ep_id: str = None
    ) -> "Trajectory":
        """
        Creates trajectory by running submission (policy and obs builder).

        Parameters
        ----------
        policy : Policy
            the submission's policy
        data_dir : Path
            the path to write the trajectory to
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
            if not provide, generate one.
        Returns
        -------
        Trajectory

        """
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

        trains_positions = trajectory.read_trains_positions()
        actions = trajectory.read_actions()
        trains_arrived = trajectory.read_trains_arrived()

        n_agents = env.get_num_agents()
        assert len(env.agents) == n_agents
        env_time = 0
        for env_time in tqdm.tqdm(range(env._max_episode_steps)):
            if snapshot_interval > 0 and env_time % snapshot_interval == 0:
                RailEnvPersister.save(env, str(data_dir / SERIALISED_STATE_SUBDIR / f"{trajectory.ep_id}_step{env_time:04d}.pkl"))
            action_dict = dict()
            for handle in env.get_agent_handles():
                action = policy.act(handle, observations[handle])
                action_dict.update({handle: action})
                trajectory.action_collect(actions, env_time=env_time, agent_id=handle, action=action)

            _, _, dones, _ = env.step(action_dict)

            for agent_id in range(n_agents):
                agent = env.agents[agent_id]
                actual_position = (agent.position, agent.direction)
                trajectory.position_collect(trains_positions, env_time=env_time + 1, agent_id=agent_id, position=actual_position)
            done = dones['__all__']

            if snapshot_interval > 0 and done and (env_time + 1) % snapshot_interval == 0:
                RailEnvPersister.save(env, str(data_dir / SERIALISED_STATE_SUBDIR / f"{trajectory.ep_id}_step{env_time + 1:04d}.pkl"))
            if done:
                break

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
@click.option('--ep-id',
              type=str,
              help="Episode ID.",
              required=True
              )
def evaluate_trajectory(data_dir: Path, ep_id: str):
    Trajectory(data_dir=data_dir, ep_id=ep_id).evaluate()


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
        ep_id: str = None
):
    module = importlib.import_module(policy_pkg)
    policy_cls = getattr(module, policy_cls)

    obs_builder = None
    if obs_builder_pkg is not None and obs_builder_cls is not None:
        module = importlib.import_module(obs_builder_pkg)
        obs_builder_cls = getattr(module, obs_builder_cls)
        obs_builder = obs_builder_cls()
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
        ep_id=ep_id
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
