import ast
import os
import uuid
from pathlib import Path
from typing import Optional, Tuple, Any, Dict

import pandas as pd
from attr import attrs, attrib

from flatland.envs.persistence import RailEnvPersister
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_env_action import RailEnvActions
from flatland.envs.step_utils.states import TrainState

EVENT_LOGS_SUBDIR = 'event_logs'
DISCRETE_ACTION_FNAME = os.path.join(EVENT_LOGS_SUBDIR, "ActionEvents.discrete_action.tsv")
TRAINS_ARRIVED_FNAME = os.path.join(EVENT_LOGS_SUBDIR, "TrainMovementEvents.trains_arrived.tsv")
TRAINS_POSITIONS_FNAME = os.path.join(EVENT_LOGS_SUBDIR, "TrainMovementEvents.trains_positions.tsv")
trains_rewards_dones_infos_FNAME = os.path.join(EVENT_LOGS_SUBDIR, "TrainMovementEvents.trains_rewards_dones_infos.tsv")
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
        TrainMovementEvents.trains_rewards_dones_infos   -- holds the rewards for the related episodes.
    - serialised_state
        <ep_id>.pkl                          -- Holds the pickled environment version for the episode.

    Indexing:
        - actions for step i are index i-1 (i.e. starting at 0)
        - positions before step i are indexed i-1 (i.e. starting at 0)
        - positions after step are indexed i (i.e. starting at 1)
    """
    data_dir = attrib(type=Path)
    ep_id = attrib(type=str, factory=_uuid_str)
    trains_positions = attrib(type=pd.DataFrame, default=None)
    actions = attrib(type=pd.DataFrame, default=None)
    trains_arrived = attrib(type=pd.DataFrame, default=None)
    trains_rewards_dones_infos = attrib(type=pd.DataFrame, default=None)

    def load(self, episode_only: bool = False):
        self.trains_positions = self._read_trains_positions(episode_only=episode_only)
        self.actions = self._read_actions(episode_only=episode_only)
        self.trains_arrived = self._read_trains_arrived(episode_only=episode_only)
        self.trains_rewards_dones_infos = self._read_trains_rewards_dones_infos(episode_only=episode_only)

    def persist(self):
        self._write_actions(self.actions)
        self._write_trains_positions(self.trains_positions)
        self._write_trains_arrived(self.trains_arrived)
        self._write_trains_rewards_dones_infos(self.trains_rewards_dones_infos)

    def _read_actions(self, episode_only: bool = False) -> pd.DataFrame:
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
            df = df[df['episode_id'] == self.ep_id]
        df["action"] = df["action"].map(RailEnvActions.from_value)
        return df

    def _read_trains_arrived(self, episode_only: bool = False) -> pd.DataFrame:
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

    def _read_trains_positions(self, episode_only: bool = False) -> pd.DataFrame:
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
        df["position"] = df["position"].map(ast.literal_eval).map(lambda p: (p[0], int(p[1])))
        if episode_only:
            return df[df['episode_id'] == self.ep_id]
        return df

    def _read_trains_rewards_dones_infos(self, episode_only: bool = False) -> pd.DataFrame:
        """Returns pd df with all trains' rewards, dones, infos for all episodes.

        Parameters
        ----------
        episode_only : bool
            Filter df to contain only this episode.
        """
        f = os.path.join(self.data_dir, trains_rewards_dones_infos_FNAME)
        if not os.path.exists(f):
            return pd.DataFrame(columns=['episode_id', 'env_time', 'agent_id', 'reward', 'info', 'done'])
        df = pd.read_csv(f, sep='\t')
        if episode_only:
            df = df[df['episode_id'] == self.ep_id]
        df["info"] = df["info"].map(lambda s: s.replace("<TrainState.WAITING: 0>", "0").replace("<TrainState.READY_TO_DEPART: 1>", "1").replace(
            "<TrainState.MALFUNCTION_OFF_MAP: 2>", "2").replace("<TrainState.MOVING: 3>", "3").replace("<TrainState.STOPPED: 4>", "4").replace(
            "<TrainState.MALFUNCTION: 5>", "5").replace("<TrainState.DONE: 6>", "6"))
        df["info"] = df["info"].map(ast.literal_eval)
        df["info"] = df["info"].map(lambda d: {k: (v if k != "state" else TrainState(v)) for k, v in d.items()})
        return df

    def _write_trains_positions(self, df: pd.DataFrame):
        """Store pd df with all trains' positions for all episodes."""
        f = os.path.join(self.data_dir, TRAINS_POSITIONS_FNAME)
        Path(f).parent.mkdir(parents=True, exist_ok=True)
        df["position"] = df["position"].map(lambda p: (p[0], int(p[1])))
        df.to_csv(f, sep='\t', index=False)

    def _write_actions(self, df: pd.DataFrame):
        """Store pd df with all trains' actions for all episodes."""
        f = os.path.join(self.data_dir, DISCRETE_ACTION_FNAME)
        Path(f).parent.mkdir(parents=True, exist_ok=True)
        df["action"] = df["action"].map(lambda a: a.value if isinstance(a, RailEnvActions) else a)
        df.to_csv(f, sep='\t', index=False)

    def _write_trains_arrived(self, df: pd.DataFrame):
        """Store pd df with all trains' success rates for all episodes."""
        f = os.path.join(self.data_dir, TRAINS_ARRIVED_FNAME)
        Path(f).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(f, sep='\t', index=False)

    def _write_trains_rewards_dones_infos(self, df: pd.DataFrame):
        """Store pd df with all trains' rewards for all episodes."""
        f = os.path.join(self.data_dir, trains_rewards_dones_infos_FNAME)
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

    def position_collect(self, env_time: int, agent_id: int, position: Tuple[Tuple[int, int], int]):
        df = self.trains_positions
        df.loc[len(df)] = {'episode_id': self.ep_id, 'env_time': env_time, 'agent_id': agent_id, 'position': position}

    def action_collect(self, env_time: int, agent_id: int, action: RailEnvActions):
        df = self.actions
        df.loc[len(df)] = {'episode_id': self.ep_id, 'env_time': env_time, 'agent_id': agent_id, 'action': action}

    def arrived_collect(self, env_time: int, success_rate: float):
        df = self.trains_arrived
        df.loc[len(df)] = {'episode_id': self.ep_id, 'env_time': env_time, 'success_rate': success_rate}

    def rewards_dones_infos_collect(self, env_time: int, agent_id: int, reward: float, info: Any, done: bool):
        df = self.trains_rewards_dones_infos
        df.loc[len(df)] = {'episode_id': self.ep_id, 'env_time': env_time, 'agent_id': agent_id, 'reward': reward, 'info': info, 'done': done}

    def position_lookup(self, env_time: int, agent_id: int) -> Tuple[Tuple[int, int], int]:
        """Method used to retrieve the stored position (if available).

        Parameters
        ----------
        env_time: int
            position before (!) step env_time
        agent_id: int
            agent ID
        Returns
        -------
        Tuple[Tuple[int, int], int]
            The position in the format ((row, column), direction).
        """
        df = self.trains_positions
        pos = df.loc[(df['env_time'] == env_time) & (df['agent_id'] == agent_id) & (df['episode_id'] == self.ep_id)]['position']
        if len(pos) != 1:
            print(f"Found {len(pos)} positions for {self.ep_id} {env_time} {agent_id}")
            print(df[(df['agent_id'] == agent_id) & (df['episode_id'] == self.ep_id)]["env_time"])
        assert len(pos) == 1, f"Found {len(pos)} positions for {self.ep_id} {env_time} {agent_id}"
        # fail fast
        p, d = pos.iloc[0]
        return (p, d)

    def action_lookup(self, env_time: int, agent_id: int) -> RailEnvActions:
        """Method used to retrieve the stored action (if available). Defaults to 2 = MOVE_FORWARD.

        Parameters
        ----------
        env_time: int
            action going into step env_time
        agent_id: int
            agent ID
        Returns
        -------
        RailEnvActions
            The action to step the env.
        """
        actions_df = self.actions
        action = actions_df.loc[
            (actions_df['env_time'] == env_time) &
            (actions_df['agent_id'] == agent_id) &
            (actions_df['episode_id'] == self.ep_id)
            ]['action'].to_numpy()
        if len(action) == 0:
            return RailEnvActions.MOVE_FORWARD
        return RailEnvActions.from_value(action[0])

    def trains_arrived_lookup(self) -> pd.Series:
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
        movements_df = self.trains_arrived
        movement = movements_df.loc[(movements_df['episode_id'] == self.ep_id)]

        if len(movement) == 1:
            return movement.iloc[0]
        raise Exception(f"No entry for {self.ep_id} found in data frame.")

    def trains_rewards_dones_infos_lookup(self, env_time: int, agent_id: int) -> Tuple[float, bool, Dict]:
        """Method used to retrieve the rewards for the episode.

        Parameters
        ----------
        env_time: int
            action going into step env_time
        agent_id: int
            agent ID
        Returns
        -------
        pd.DataFrame
            The trains arrived data.
        """
        rewards_df = self.trains_rewards_dones_infos
        data = rewards_df.loc[(rewards_df['env_time'] == env_time) & (rewards_df['agent_id'] == agent_id) & (rewards_df['episode_id'] == self.ep_id)]
        assert len(data) == 1
        data = data.iloc[0]
        return data["reward"], data["done"], data["info"]

    @property
    def outputs_dir(self) -> Path:
        return self.data_dir / OUTPUTS_SUBDIR

    def compare_actions(self, other: "Trajectory", start_step: int = None, end_step: int = None) -> pd.DataFrame:
        df = self._read_actions(episode_only=True)
        other_df = other._read_actions(episode_only=True)
        return self._compare(df, other_df, end_step, start_step)

    def compare_positions(self, other: "Trajectory", start_step: int = None, end_step: int = None) -> pd.DataFrame:
        df = self._read_trains_positions(episode_only=True)
        other_df = other._read_trains_positions(episode_only=True)
        return self._compare(df, other_df, end_step, start_step)

    def compare_arrived(self, other: "Trajectory", start_step: int = None, end_step: int = None) -> pd.DataFrame:
        df = self._read_trains_arrived(episode_only=True)
        other_df = other._read_trains_arrived(episode_only=True)
        return self._compare(df, other_df, end_step, start_step)

    def compare_rewards_dones_infos(self, other: "Trajectory", start_step: int = None, end_step: int = None) -> pd.DataFrame:
        df = self._read_trains_rewards_dones_infos(episode_only=True)
        other_df = other._read_trains_rewards_dones_infos(episode_only=True)
        return self._compare(df, other_df, end_step, start_step)

    def _compare(self, df, other_df, end_step, start_step):
        if start_step is not None:
            df = df[df["env_time"] >= start_step]
            other_df = other_df[other_df["env_time"] >= start_step]
        if end_step is not None:
            df = df[df["env_time"] < end_step]
            other_df = other_df[other_df["env_time"] < end_step]
        df.reset_index(drop=True, inplace=True)
        other_df.reset_index(drop=True, inplace=True)
        df.drop(columns="episode_id", inplace=True)
        other_df.drop(columns="episode_id", inplace=True)
        diff = df.compare(other_df)
        return diff
