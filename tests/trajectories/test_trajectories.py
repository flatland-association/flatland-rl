import tempfile
from pathlib import Path
from typing import Any

from flatland.env_generation.env_generator import env_generator
from flatland.trajectories.trajectories import Policy, Trajectory, DISCRETE_ACTION_FNAME, TRAINS_ARRIVED_FNAME, TRAINS_POSITIONS_FNAME, SERIALISED_STATE_SUBDIR
from flatland.utils.seeding import np_random, random_state_to_hashablestate


class RandomPolicy(Policy):
    def __init__(self, action_size: int = 5, seed=42):
        super(RandomPolicy, self).__init__()
        self.action_size = action_size
        self.np_random, _ = np_random(seed=seed)

    def act(self, handle: int, observation: Any):
        return self.np_random.choice(self.action_size)


def test_from_episode():
    with tempfile.TemporaryDirectory() as tmpdirname:
        data_dir = Path(tmpdirname)
        trajectory = Trajectory.from_submission(policy=RandomPolicy(), data_dir=data_dir, snapshot_interval=5)
        # np_random in loaded episode is same as if it comes directly from env_generator incl. reset()!
        env = trajectory.restore_episode()
        gen, _, _ = env_generator()
        assert random_state_to_hashablestate(env.np_random) == random_state_to_hashablestate(gen.np_random)

        gen.reset(random_seed=42)
        assert random_state_to_hashablestate(env.np_random) == random_state_to_hashablestate(gen.np_random)


def test_from_submission():
    with tempfile.TemporaryDirectory() as tmpdirname:
        data_dir = Path(tmpdirname)
        trajectory = Trajectory.from_submission(policy=RandomPolicy(), data_dir=data_dir, snapshot_interval=5)

        assert (data_dir / DISCRETE_ACTION_FNAME).exists()
        assert (data_dir / TRAINS_ARRIVED_FNAME).exists()
        assert (data_dir / TRAINS_POSITIONS_FNAME).exists()
        assert (data_dir / SERIALISED_STATE_SUBDIR / f"{trajectory.ep_id}.pkl").exists()

        assert (data_dir / SERIALISED_STATE_SUBDIR / f"{trajectory.ep_id}.pkl").exists()
        snapshots = list((data_dir / SERIALISED_STATE_SUBDIR).glob("*step*"))
        assert len(snapshots) == 95
        assert set(snapshots) == {data_dir / SERIALISED_STATE_SUBDIR / f"{trajectory.ep_id}_step{i * 5:04d}.pkl" for i in range(95)}

        assert "episode_id	env_time	agent_id	action" in (data_dir / DISCRETE_ACTION_FNAME).read_text()
        assert "episode_id	env_time	success_rate" in (data_dir / TRAINS_ARRIVED_FNAME).read_text()
        assert "episode_id	env_time	agent_id	position" in (data_dir / TRAINS_POSITIONS_FNAME).read_text()

        trajectory.run()
