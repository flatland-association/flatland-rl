import ast
import os
import uuid
from pathlib import Path

import pandas as pd
from attr import attrs, attrib

from flatland.envs.malfunction_generators import NoMalfunctionGen
from flatland.envs.persistence import RailEnvPersister
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_env_action import RailEnvActions
from flatland.envs.rail_trainrun_data_structures import Waypoint

DISCRETE_ACTION_FNAME = "event_logs/ActionEvents.discrete_action.tsv"
TRAINS_ARRIVED_FNAME = "event_logs/TrainMovementEvents.trains_arrived.tsv"
TRAINS_POSITIONS_FNAME = "event_logs/TrainMovementEvents.trains_positions.tsv"
SERIALISED_STATE_SUBDIR = 'serialised_state'

COLLECT_POSITIONS = False


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
    ep_id = attrib(type=str, factory=uuid.uuid4)

    def read_actions(self):
        """Returns pd df with all actions for all episodes."""
        tmp_dir = os.path.join(self.data_dir, DISCRETE_ACTION_FNAME)
        return pd.read_csv(tmp_dir, sep='\t')

    def read_trains_arrived(self):
        """Returns pd df with success rate for all episodes."""
        tmp_dir = os.path.join(self.data_dir, TRAINS_ARRIVED_FNAME)
        return pd.read_csv(tmp_dir, sep='\t')

    def read_trains_positions(self) -> pd.DataFrame:
        """Returns pd df with all trains' positions for all episodes."""
        f = os.path.join(self.data_dir, TRAINS_POSITIONS_FNAME)
        if not os.path.exists(f):
            return pd.DataFrame(columns=['env_time', 'agent_id', 'episode_id', 'position'])
        return pd.read_csv(f, sep='\t')

    def write_trains_positions(self, df: pd.DataFrame):
        """Store pd df with all trains' positions for all episodes."""
        f = os.path.join(self.data_dir, TRAINS_POSITIONS_FNAME)
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

        # TODO episodes contain strings for malfunction_rate etc. instead of ints - we should fix the serialized pkls?
        env.malfunction_generator = NoMalfunctionGen()
        return env

    def position_collect(self, df: pd.DataFrame, env_time: int, agent_id: int, position: Waypoint):
        df.loc[len(df)] = {'env_time': env_time, 'agent_id': agent_id, 'episode_id': self.ep_id, 'position': position}

    def position_lookup(self, df: pd.DataFrame, env_time: int, agent_id: int) -> Waypoint:
        """Method used to retrieve the stored position (if available).

        Parameters
        ----------
        df: pd.DataFrame
            Data frame from ActionEvents.discrete_action.tsv
        env_time: int
            episode step
        agent_id: int
            agent ID
        Returns
        -------
        Waypoint
            The position in the format ((row, column), direction).
        """
        pos = df.loc[(df['env_time'] == env_time) & (df['agent_id'] == agent_id) & (df['episode_id'] == self.ep_id)]['position']
        if len(pos) != 1:
            print(f"Found {len(pos)} positions for {self.ep_id} {env_time} {agent_id}")
            print(df[(df['agent_id'] == agent_id) & (df['episode_id'] == self.ep_id)]["env_time"])
        assert len(pos) == 1, f"Found {len(pos)} positions for {self.ep_id} {env_time} {agent_id}"
        return Waypoint(*ast.literal_eval(pos.iloc[0]))

    def action_lookup(self, actions_df: pd.DataFrame, env_time: int, agent_id: int) -> RailEnvActions:
        """Method used to retrieve the stored action (if available). Defaults to 2 = MOVE_FORWARD.

        Parameters
        ----------
        actions_df: pd.DataFrame
            Data frame from ActionEvents.discrete_action.tsv
        env_time: int
            episode step
        agent_id: int
            agent ID
        Returns
        -------
        RailEnvActions
            The action to step the env.
        """
        action = actions_df.loc[
            (actions_df['env_time'] == env_time) &
            (actions_df['agent_id'] == f'agent_{agent_id}') &
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

    def run(self):
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
        done = False
        n_agents = env.get_num_agents()
        assert len(env.agents) == n_agents
        i = 0
        while not done:
            action = {agent_id: self.action_lookup(actions, env_time=env._elapsed_steps - 1, agent_id=agent_id) for agent_id in range(n_agents)}
            _, _, dones, _ = env.step(action)
            done = dones['__all__']
            i += 1

            for agent_id in range(n_agents):
                agent = env.agents[agent_id]
                actual_position = Waypoint(agent.position, agent.direction)
                if COLLECT_POSITIONS:
                    self.position_collect(trains_positions, env_time=i, agent_id=agent_id, position=actual_position)
                else:
                    expected_position = self.position_lookup(trains_positions, env_time=i, agent_id=agent_id)
                    assert actual_position == expected_position, (self.data_dir, self.ep_id, agent_id, i, actual_position, expected_position)

        trains_arrived_episode = self.trains_arrived_lookup(trains_arrived)
        expected_success_rate = trains_arrived_episode['success_rate']
        actual_success_rate = sum([agent.state == 6 for agent in env.agents]) / n_agents
        print(f"{actual_success_rate * 100}% trains arrived. Expected {expected_success_rate * 100}%.")
        assert expected_success_rate == actual_success_rate
        if COLLECT_POSITIONS:
            self.write_trains_positions(trains_positions)
