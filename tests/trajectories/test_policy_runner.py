import re
import tempfile
from pathlib import Path
from typing import Optional, Any

import pytest

from flatland.callbacks.callbacks import FlatlandCallbacks, make_multi_callbacks
from flatland.core.policy import Policy
from flatland.env_generation.env_generator import env_generator
from flatland.envs.rail_env import RailEnv
from flatland.evaluators.trajectory_evaluator import TrajectoryEvaluator, evaluate_trajectory
from flatland.trajectories.policy_runner import PolicyRunner, generate_trajectory_from_policy
from flatland.trajectories.trajectories import DISCRETE_ACTION_FNAME, TRAINS_ARRIVED_FNAME, TRAINS_POSITIONS_FNAME, SERIALISED_STATE_SUBDIR
from flatland.utils.seeding import random_state_to_hashablestate, np_random


class RandomPolicy(Policy):
    def __init__(self, action_size: int = 5, seed=42):
        super(RandomPolicy, self).__init__()
        self.action_size = action_size
        self.np_random, _ = np_random(seed=seed)

    def act(self, handle: int, observation: Any, **kwargs):
        return self.np_random.choice(self.action_size)


def test_from_episode():
    with tempfile.TemporaryDirectory() as tmpdirname:
        data_dir = Path(tmpdirname)
        trajectory = PolicyRunner.create_from_policy(policy=RandomPolicy(), data_dir=data_dir, snapshot_interval=5)
        # np_random in loaded episode is same as if it comes directly from env_generator incl. reset()!
        env = trajectory.restore_episode()
        gen, _, _ = env_generator()
        assert random_state_to_hashablestate(env.np_random) == random_state_to_hashablestate(gen.np_random)

        gen.reset(random_seed=42)
        assert random_state_to_hashablestate(env.np_random) == random_state_to_hashablestate(gen.np_random)


def test_from_submission():
    with tempfile.TemporaryDirectory() as tmpdirname:
        data_dir = Path(tmpdirname)
        trajectory = PolicyRunner.create_from_policy(policy=RandomPolicy(), data_dir=data_dir, snapshot_interval=5)

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

        class DummyCallbacks(FlatlandCallbacks):
            def on_episode_step(
                self,
                *,
                env: Optional[RailEnv] = None,
                **kwargs,
            ) -> None:
                (data_dir / f"step{env._elapsed_steps - 1}").touch()

        TrajectoryEvaluator(trajectory, callbacks=make_multi_callbacks(DummyCallbacks())).evaluate()
        for i in range(471):
            assert (data_dir / f"step{i}").exists()
        assert not (data_dir / f"step471").exists()

        TrajectoryEvaluator(trajectory).evaluate(start_step=5)
        with pytest.raises(FileNotFoundError):
            TrajectoryEvaluator(trajectory).evaluate(start_step=4)


def test_cli_from_submission():
    with tempfile.TemporaryDirectory() as tmpdirname:
        data_dir = Path(tmpdirname)
        with pytest.raises(SystemExit) as e_info:
            generate_trajectory_from_policy(
                ["--data-dir", data_dir, "--policy-pkg", "tests.trajectories.test_policy_runner", "--policy-cls", "RandomPolicy"])
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
            evaluate_trajectory(["--data-dir", data_dir, "--ep-id", ep_id])
        assert e_info.value.code == 0


def test_fork_and_run_from_intermediate_step():
    with tempfile.TemporaryDirectory() as tmpdirname:
        data_dir = Path(tmpdirname)
        trajectory = PolicyRunner.create_from_policy(policy=RandomPolicy(), data_dir=data_dir / "trajectory", snapshot_interval=0)
        print(trajectory.read_actions())
        print(trajectory.read_trains_arrived())
        print(trajectory.read_trains_positions())
        fork = PolicyRunner.create_from_policy(
            data_dir=data_dir / "fork",
            policy=RandomPolicy(),
            start_step=7,
            end_step=17,
            fork_from_trajectory=trajectory
        )

        print(fork.read_actions())
        print(fork.read_trains_arrived())
        print(fork.read_trains_positions())


def test_evaluation_snapshots():
    with tempfile.TemporaryDirectory() as tmpdirname:
        data_dir = Path(tmpdirname)
        trajectory = PolicyRunner.create_from_policy(policy=RandomPolicy(), data_dir=data_dir, snapshot_interval=0)
        print(list(trajectory.data_dir.rglob("**/*step*.pkl")))
        assert len(list(trajectory.data_dir.rglob("**/*step*.pkl"))) == 0
        TrajectoryEvaluator(trajectory).evaluate(snapshot_interval=1)
        print(list(trajectory.data_dir.rglob("**/*step*.pkl")))
        assert len(list((trajectory.data_dir / "outputs" / "serialised_state").rglob("**/*step*.pkl"))) == 472
