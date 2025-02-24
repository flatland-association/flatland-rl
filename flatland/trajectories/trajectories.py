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

DISCRETE_ACTION_FNAME = "event_logs/ActionEvents.discrete_action.tsv"
TRAINS_ARRIVED_FNAME = "event_logs/TrainMovementEvents.trains_arrived.tsv"
TRAINS_POSITIONS_FNAME = "event_logs/TrainMovementEvents.trains_positions.tsv"
SERIALISED_STATE_SUBDIR = 'serialised_state'


# TODO add wrapper for rllib/Pettingzoo policy from checkpoint
class Policy:
    def act(self, handle: int, observation: Any, **kwargs) -> RailEnvActions:
        pass


def _uuid_str():
    return str(uuid.uuid4())


# TODO one subdirectory per trajectory?
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

    def restore_episode(self) -> RailEnv:
        """Restore an episode.

        Parameters
        ----------

        Returns
        -------
        RailEnv
            the episode
        """

        f = os.path.join(self.data_dir, SERIALISED_STATE_SUBDIR, f'{self.ep_id}.pkl')
        env, _ = RailEnvPersister.load_new(f)
        return env

    def position_collect(self, df: pd.DataFrame, env_time: int, agent_id: int, position: Tuple[Tuple[int, int], int]):
        df.loc[len(df)] = {'episode_id': self.ep_id, 'env_time': env_time, 'agent_id': agent_id, 'position': position}

    # TODO re-encode regression without agent_ prefix
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
        return ast.literal_eval(pos.iloc[0])

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

    # TODO same as verify? Finalize naming
    # TODO add rendering?
    # TODO add collect stats rewards etc from evaluator...?
    def run(self, prefix=""):
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
        data_sub_dir subdirectory within BENCHMARK_EPISODES_FOLDER
        ep_id the episode ID
        """

        trains_positions = self.read_trains_positions()
        actions = self.read_actions()
        trains_arrived = self.read_trains_arrived()

        env = self.restore_episode()
        n_agents = env.get_num_agents()
        assert len(env.agents) == n_agents

        for env_time in tqdm.tqdm(range(env._max_episode_steps)):
            # TODO re-encode regression data to have action for step i, starts at 1 for step 0
            action = {agent_id: self.action_lookup(actions, env_time=env_time - 1, agent_id=agent_id if not prefix else f"{prefix}{agent_id}") for agent_id in
                      range(n_agents)}
            _, _, dones, _ = env.step(action)
            done = dones['__all__']

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

    # TODO generate a subfolder with generated episode_id as name for the new trajectory?
    # TODO finalize naming
    # TODO extract params for env generation to interface
    @staticmethod
    def from_submission(policy: Policy, data_dir: Path, obs_builder: Optional[ObservationBuilder] = None, snapshot_interval: int = 1) -> "Trajectory":
        """
        Creates trajectory by running submission (policy and obs builder).

        Parameters
        ----------
        policy : Policy
            the submission's policy
        data_dir : Path
            the path to write the trajectory to
        obs_builder : ObservationBuilder
            the submission's obs builder
        snapshot_interval : int
            interval to write pkl snapshots

        Returns
        -------
        Trajectory

        """
        env, observations, _ = env_generator(obs_builder_object=obs_builder)
        trajectory = Trajectory(data_dir=data_dir)
        (data_dir / SERIALISED_STATE_SUBDIR).mkdir(parents=True, exist_ok=True)
        RailEnvPersister.save(env, str(data_dir / SERIALISED_STATE_SUBDIR / f"{trajectory.ep_id}.pkl"))

        trains_positions = trajectory.read_trains_positions()
        actions = trajectory.read_actions()
        trains_arrived = trajectory.read_trains_arrived()

        n_agents = env.get_num_agents()
        assert len(env.agents) == n_agents

        for env_time in tqdm.tqdm(range(env._max_episode_steps)):
            if env_time % snapshot_interval == 0:
                RailEnvPersister.save(env, str(data_dir / SERIALISED_STATE_SUBDIR / f"{trajectory.ep_id}_step{env_time:04d}.pkl"))
            action_dict = dict()
            for handle in env.get_agent_handles():
                action = policy.act(handle, observations[handle])
                action_dict.update({handle: action})
                # TODO re-encode regression episodes instead
                trajectory.action_collect(actions, env_time=env_time - 1, agent_id=handle, action=action)

            _, _, dones, _ = env.step(action_dict)

            for agent_id in range(n_agents):
                agent = env.agents[agent_id]
                actual_position = (agent.position, agent.direction)
                trajectory.position_collect(trains_positions, env_time=env_time + 1, agent_id=agent_id, position=actual_position)
            done = dones['__all__']

            if done and (env_time + 1) % snapshot_interval == 0:
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
def cli_run(data_dir: Path, ep_id: str):
    Trajectory(data_dir=data_dir, ep_id=ep_id).run()


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
def cli_from_submission(data_dir: Path, policy_pkg: str, policy_cls: str):
    module = importlib.import_module(policy_pkg)
    policy_cls = getattr(module, policy_cls)

    Trajectory.from_submission(policy=policy_cls(), data_dir=data_dir)
