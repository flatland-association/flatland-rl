from pathlib import Path
from typing import Optional

from flatland.callbacks.callbacks import FlatlandCallbacks
from flatland.envs.persistence import RailEnvPersister
from flatland.envs.rail_env import RailEnv
from flatland.trajectories.trajectories import SERIALISED_STATE_SUBDIR
from flatland.trajectories.trajectories import Trajectory


class TrajectorySnapshotCallbacks(FlatlandCallbacks):
    def __init__(self, trajectory: Trajectory, data_dir_override: Path = None, snapshot_interval: int = None):
        self.trajectory = trajectory
        self.snapshot_interval = snapshot_interval
        self.data_dir_override = data_dir_override

    def on_episode_start(
        self,
        *,
        env: Optional[RailEnv] = None,
        data_dir: Path = None,
        **kwargs,
    ) -> None:
        if self.data_dir_override is not None:
            data_dir = self.data_dir_override
        if self.snapshot_interval > 0:
            (data_dir / SERIALISED_STATE_SUBDIR).mkdir(exist_ok=True)
            RailEnvPersister.save(env, str(data_dir / SERIALISED_STATE_SUBDIR / f"{self.trajectory.ep_id}_step{env._elapsed_steps:04d}.pkl"))

    def on_episode_step(
        self,
        *,
        env: Optional[RailEnv] = None,
        data_dir: Path = None,
        **kwargs,
    ) -> None:
        if self.data_dir_override is not None:
            data_dir = self.data_dir_override
        elapsed_steps = env._elapsed_steps
        if self.snapshot_interval > 0 and elapsed_steps % self.snapshot_interval == 0:
            (data_dir / SERIALISED_STATE_SUBDIR).mkdir(exist_ok=True)
            RailEnvPersister.save(env, str(data_dir / SERIALISED_STATE_SUBDIR / f"{self.trajectory.ep_id}_step{elapsed_steps :04d}.pkl"))
