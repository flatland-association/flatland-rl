import tempfile
from pathlib import Path

from flatland.envs.observations import FullEnvObservation
from flatland.envs.rail_env_policies import ShortestPathPolicy
from flatland.trajectories.policy_runner import PolicyRunner


def test_shortest_path_policy():
    with tempfile.TemporaryDirectory() as tmpdirname:
        data_dir = Path(tmpdirname)
        trajectory = PolicyRunner.create_from_policy(
            policy=ShortestPathPolicy(),
            obs_builder=FullEnvObservation(),
            data_dir=data_dir,
            snapshot_interval=5,
        )
        print(trajectory.trains_arrived_lookup())
