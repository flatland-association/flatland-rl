from pathlib import Path
from typing import Optional

from flatland.callbacks.callbacks import FlatlandCallbacks
from flatland.envs.persistence import RailEnvPersister
from flatland.envs.rail_env import RailEnv
from flatland.trajectories.trajectories import Trajectory, SERIALISED_STATE_SUBDIR


class TrajectorySnapshotCallbacks(FlatlandCallbacks):
    def __init__(self, trajectory: Trajectory, snapshot_interval: int = None):
        self.trajectory = trajectory
        self.snapshot_interval = snapshot_interval

    def on_episode_start(
        self,
        *,
        env: Optional[RailEnv] = None,
        data_dir: Path = None,
        **kwargs,
    ) -> None:
        if self.snapshot_interval > 0:
            RailEnvPersister.save(env, str(self.trajectory.data_dir / SERIALISED_STATE_SUBDIR / f"{self.trajectory.ep_id}_step{env._elapsed_steps:04d}.pkl"))

    def on_episode_step(
        self,
        *,
        env: Optional[RailEnv] = None,
        data_dir: Path = None,
        **kwargs,
    ) -> None:
        elapsed_steps = env._elapsed_steps
        if self.snapshot_interval > 0 and elapsed_steps % self.snapshot_interval == 0:
            RailEnvPersister.save(env, str(self.trajectory.data_dir / SERIALISED_STATE_SUBDIR / f"{self.trajectory.ep_id}_step{elapsed_steps :04d}.pkl"))
