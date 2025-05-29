import ast
import os
import uuid
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from attr import attrs, attrib

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

    Indexing:
        - actions for step i are index i-1 (i.e. starting at 0)
        - positions before step i are indexed i-1 (i.e. starting at 0)
        - positions after step are indexed i (i.e. starting at 1)
    """
    data_dir = attrib(type=Path)
    ep_id = attrib(type=str, factory=_uuid_str)

    def read_actions(self, episode_only: bool = False):
        """Returns pd df with all actions for all episodes.

        Parameters
        ----------
        episode_only : bool
            Filter df to contain only this episode.
        """
        f = os.path.join(self.data_dir, DISCRETE_ACTION_FNAME)
        if not os.path.exists(f):
            return pd.DataFrame(columns=['episode_id', 'env_time', 'agent_id', 'action'])
        df = pd.read_csv(f, sep='\t')
        if episode_only:
            return df[df['episode_id'] == self.ep_id]
        return df

    def read_trains_arrived(self, episode_only: bool = False):
        """Returns pd df with success rate for all episodes.

            Parameters
            ----------
            episode_only : bool
                Filter df to contain only this episode.
        """
        f = os.path.join(self.data_dir, TRAINS_ARRIVED_FNAME)
        if not os.path.exists(f):
            return pd.DataFrame(columns=['episode_id', 'env_time', 'success_rate'])
        df = pd.read_csv(f, sep='\t')
        if episode_only:
            return df[df['episode_id'] == self.ep_id]
        return df

    def read_trains_positions(self, episode_only: bool = False) -> pd.DataFrame:
        """Returns pd df with all trains' positions for all episodes.

        Parameters
        ----------
        episode_only : bool
            Filter df to contain only this episode.
        """
        f = os.path.join(self.data_dir, TRAINS_POSITIONS_FNAME)
        if not os.path.exists(f):
            return pd.DataFrame(columns=['episode_id', 'env_time', 'agent_id', 'position'])
        df = pd.read_csv(f, sep='\t')
        if episode_only:
            return df[df['episode_id'] == self.ep_id]
        return df

    def write_trains_positions(self, df: pd.DataFrame):
        """Store pd df with all trains' positions for all episodes."""
        f = os.path.join(self.data_dir, TRAINS_POSITIONS_FNAME)
        Path(f).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(f, sep='\t', index=False)

    def write_actions(self, df: pd.DataFrame):
        """Store pd df with all trains' actions for all episodes."""
        f = os.path.join(self.data_dir, DISCRETE_ACTION_FNAME)
        Path(f).parent.mkdir(parents=True, exist_ok=True)
        df["action"] = df["action"].map(lambda a: a.value if isinstance(a, RailEnvActions) else a)
        df.to_csv(f, sep='\t', index=False)

    def write_trains_arrived(self, df: pd.DataFrame):
        """Store pd df with all trains' success rates for all episodes."""
        f = os.path.join(self.data_dir, TRAINS_ARRIVED_FNAME)
        Path(f).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(f, sep='\t', index=False)

    def restore_episode(self, start_step: int = None) -> Optional[RailEnv]:
        """Restore an episode.

        Parameters
        ----------
        start_step : Optional[int]
            start from snapshot (if it exists)
        Returns
        -------
        RailEnv
            the rail env or None if the snapshot at the step does not exist
        """
        if start_step is None:
            f = os.path.join(self.data_dir, SERIALISED_STATE_SUBDIR, f'{self.ep_id}.pkl')
            env, _ = RailEnvPersister.load_new(f)
            return env
        else:
            f = os.path.join(self.data_dir, SERIALISED_STATE_SUBDIR, f"{self.ep_id}_step{start_step:04d}.pkl")
            if not os.path.isfile(f):
                return None
            env, _ = RailEnvPersister.load_new(f)
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
            return RailEnvActions.MOVE_FORWARD
        return RailEnvActions.from_value(action[0])

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
        raise Exception(f"No entry for {self.ep_id} found in data frame.")

    @property
    def outputs_dir(self) -> Path:
        return self.data_dir / OUTPUTS_SUBDIR
