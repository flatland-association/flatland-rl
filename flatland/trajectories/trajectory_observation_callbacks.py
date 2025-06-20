import pickle
from pathlib import Path
from typing import Optional, Literal

import msgpack

from flatland.callbacks.callbacks import FlatlandCallbacks
from flatland.envs.rail_env import RailEnv
from flatland.trajectories.trajectories import SERIALISED_STATE_SUBDIR
from flatland.trajectories.trajectories import Trajectory


class TrajectoryObservationCallbacks(FlatlandCallbacks):
    """
    FlatlandCallbacks to write observations.

    Parameters
    ----------
    trajectory: Trajectory
        the trajectory
    data_dir_override : Path
        use this override instead of the `data_dir` passed in the callback.
    """

    def __init__(self, trajectory: Trajectory, data_dir_override: Path = None, format: Literal["pkl", "mpk"] = "pkl"):
        self.trajectory = trajectory
        self.data_dir_override = data_dir_override
        self.format = format

    def _dump(self, data_dir: Path, env: RailEnv):
        if self.format == "pkl":
            data = pickle.dumps(env._get_observations())
        elif self.format == "mpk":
            data = msgpack.packb(env._get_observations())
        else:
            raise ValueError("Format must be \"mpk\" (msgpack) or \"pkl\" for pickle")
        (data_dir / SERIALISED_STATE_SUBDIR).mkdir(exist_ok=True, parents=True)
        with (data_dir / SERIALISED_STATE_SUBDIR / f"{self.trajectory.ep_id}_obs{env._elapsed_steps:04d}.{self.format}").open("wb") as f:
            f.write(data)

    def on_episode_start(
        self,
        *,
        env: Optional[RailEnv] = None,
        data_dir: Path = None,
        **kwargs,
    ) -> None:
        if self.data_dir_override is not None:
            data_dir = self.data_dir_override
        self._dump(data_dir, env)

    def on_episode_step(
        self,
        *,
        env: Optional[RailEnv] = None,
        data_dir: Path = None,
        **kwargs,
    ) -> None:
        if self.data_dir_override is not None:
            data_dir = self.data_dir_override
        self._dump(data_dir, env)
