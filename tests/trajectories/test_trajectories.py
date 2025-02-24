import importlib
import re
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from flatland.env_generation.env_generator import env_generator
from flatland.trajectories.trajectories import Policy, Trajectory, DISCRETE_ACTION_FNAME, TRAINS_ARRIVED_FNAME, TRAINS_POSITIONS_FNAME, SERIALISED_STATE_SUBDIR, \
    cli_from_submission, cli_run, gen_trajectories_from_metadata
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


def test_cli_from_submission():
    with tempfile.TemporaryDirectory() as tmpdirname:
        data_dir = Path(tmpdirname)
        with pytest.raises(SystemExit) as e_info:
            cli_from_submission(["--data-dir", data_dir, "--policy-pkg", "tests.trajectories.test_trajectories", "--policy-cls", "RandomPolicy"])
        assert e_info.value.code == 0

        ep_id = re.sub(r"_step.*", "", str(next((data_dir / SERIALISED_STATE_SUBDIR).glob("*step*.pkl")).name))

        assert (data_dir / DISCRETE_ACTION_FNAME).exists()
        assert (data_dir / TRAINS_ARRIVED_FNAME).exists()
        assert (data_dir / TRAINS_POSITIONS_FNAME).exists()
        assert (data_dir / SERIALISED_STATE_SUBDIR / f"{ep_id}.pkl").exists()

        assert (data_dir / SERIALISED_STATE_SUBDIR / f"{ep_id}.pkl").exists()
        snapshots = list((data_dir / SERIALISED_STATE_SUBDIR).glob("*step*"))
        assert len(snapshots) == 472
        assert set(snapshots) == {data_dir / SERIALISED_STATE_SUBDIR / f"{ep_id}_step{i:04d}.pkl" for i in range(472)}

        assert "episode_id	env_time	agent_id	action" in (data_dir / DISCRETE_ACTION_FNAME).read_text()
        assert "episode_id	env_time	success_rate" in (data_dir / TRAINS_ARRIVED_FNAME).read_text()
        assert "episode_id	env_time	agent_id	position" in (data_dir / TRAINS_POSITIONS_FNAME).read_text()

        with pytest.raises(SystemExit) as e_info:
            cli_run(["--data-dir", data_dir, "--ep-id", ep_id])
        assert e_info.value.code == 0


@pytest.mark.skip  # TODO import heuristic baseline as example, add rendering to cli
def test_gen_trajectories_from_metadata():
    metadata_csv = importlib.resources.files("env_data.tests.service_test").joinpath("metadata.csv")
    with tempfile.TemporaryDirectory() as tmpdirname:
        with importlib.resources.as_file(metadata_csv) as template_file:
            tmpdir = Path(tmpdirname)
            gen_trajectories_from_metadata(template_file, tmpdir)
            df = pd.read_csv(tmpdir / TRAINS_ARRIVED_FNAME, sep="\t")
            assert df["success_rate"].to_list() == [0.8571428571428571, 1.0, 0.8571428571428571, 1.0]
            assert df["env_time"].to_list() == [391, 165, 391, 165]

# TODO test cli after installed! paths and tuples etc.
