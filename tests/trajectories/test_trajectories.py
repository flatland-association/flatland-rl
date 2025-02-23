import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from flatland.trajectories.trajectories import Policy, Trajectory


def test_from_submission():
    class RandomPolicy(Policy):
        def __init__(self, action_size: int = 5):
            super(RandomPolicy, self).__init__()
            self.action_size = action_size

        def act(self, handle: int, observation: Any):
            return np.random.choice(self.action_size)

    with tempfile.TemporaryDirectory() as tmpdirname:
        Trajectory.from_submission(policy=RandomPolicy(), data_dir=Path(tmpdirname))
        # TODO check files created and content
