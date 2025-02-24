import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from flatland.trajectories.trajectories import Policy, Trajectory, DISCRETE_ACTION_FNAME, TRAINS_ARRIVED_FNAME, TRAINS_POSITIONS_FNAME, SERIALISED_STATE_SUBDIR


def test_from_submission():
    class RandomPolicy(Policy):
        def __init__(self, action_size: int = 5):
            super(RandomPolicy, self).__init__()
            self.action_size = action_size

        def act(self, handle: int, observation: Any):
            return np.random.choice(self.action_size)

    with tempfile.TemporaryDirectory() as tmpdirname:
        data_dir = Path(tmpdirname)
        trajectory = Trajectory.from_submission(policy=RandomPolicy(), data_dir=data_dir)

        assert (data_dir / DISCRETE_ACTION_FNAME).exists()
        assert (data_dir / TRAINS_ARRIVED_FNAME).exists()
        assert (data_dir / TRAINS_POSITIONS_FNAME).exists()
        assert (data_dir / SERIALISED_STATE_SUBDIR / f"{trajectory.ep_id}.pkl").exists()

        assert "episode_id	env_time	agent_id	action" in (data_dir / DISCRETE_ACTION_FNAME).read_text()
        assert "episode_id	env_time	success_rate" in (data_dir / TRAINS_ARRIVED_FNAME).read_text()
        assert "episode_id	env_time	agent_id	position" in (data_dir / TRAINS_POSITIONS_FNAME).read_text()

        trajectory.run()
