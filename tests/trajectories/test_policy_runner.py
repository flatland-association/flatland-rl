import re
import tempfile
from pathlib import Path
from typing import Optional, Any

import pytest

from flatland.callbacks.callbacks import FlatlandCallbacks, make_multi_callbacks
from flatland.core.env_observation_builder import ObservationBuilder, AgentHandle, ObservationType
from flatland.core.policy import Policy
from flatland.env_generation.env_generator import env_generator
from flatland.envs.persistence import RailEnvPersister
from flatland.envs.rail_env import RailEnv
from flatland.evaluators.trajectory_evaluator import TrajectoryEvaluator, evaluate_trajectory
from flatland.trajectories.policy_runner import PolicyRunner, generate_trajectory_from_policy
from flatland.trajectories.trajectories import DISCRETE_ACTION_FNAME, TRAINS_ARRIVED_FNAME, TRAINS_POSITIONS_FNAME, SERIALISED_STATE_SUBDIR
from flatland.utils.seeding import random_state_to_hashablestate, np_random


class RandomPolicy(Policy):
    """
    Random action with reset of random sequence to allow synchronization with partial trajectory.
    """

    def __init__(self, action_size: int = 5, seed=42, reset_at: int = None):
        """

        Parameters
        ----------
        reset_at : Optional[int] actions applied in env step reset_at+1 (e.g. reset at 7 to start at step 8)
        """
        super(RandomPolicy, self).__init__()
        self.action_size = action_size
        self._seed = seed
        self.reset_at = reset_at
        self.np_random, _ = np_random(seed=self._seed)

    def act(self, handle: int, observation: Any, **kwargs):
        if handle == 0 and self.reset_at is not None and observation == self.reset_at:
            self.np_random, _ = np_random(seed=self._seed)
        return self.np_random.choice(self.action_size)


class EnvStepObservationBuilder(ObservationBuilder[int]):
    """Returns elapsed steps as observation."""

    def get(self, handle: AgentHandle = 0) -> ObservationType:
        return self.env._elapsed_steps

    def reset(self):
        pass


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


def test_fork_and_run_from_intermediate_step(verbose: bool = False):
    with tempfile.TemporaryDirectory() as tmpdirname:
        data_dir = Path(tmpdirname)
        trajectory = PolicyRunner.create_from_policy(
            policy=RandomPolicy(reset_at=7),
            obs_builder=EnvStepObservationBuilder(),
            data_dir=data_dir / "trajectory",
            snapshot_interval=0)
        if verbose:
            import pandas as pd
            pd.set_option('display.max_rows', None)
            pd.set_option('display.max_columns', None)
            print(trajectory.actions)
            print(trajectory.trains_arrived)
            print(trajectory.trains_positions)
            print(trajectory.trains_rewards_dones_infos)

        fork = PolicyRunner.create_from_policy(
            data_dir=data_dir / "fork",
            policy=RandomPolicy(),
            obs_builder=EnvStepObservationBuilder(),
            # no snapshot here, PolicyRunner needs to start from a previous snapshot and run forward to starting step:
            start_step=7,
            end_step=17,
            fork_from_trajectory=trajectory
        )
        if verbose:
            print(fork.actions)
            print(fork.trains_arrived)
            print(fork.trains_positions)
            print(fork.trains_rewards_dones_infos)
        actions_diff = trajectory.compare_actions(other=fork, start_step=7, end_step=17)
        positions_diff = trajectory.compare_positions(other=fork, start_step=8, end_step=17)
        arrived_diff = trajectory.compare_arrived(other=fork, start_step=8, end_step=17)
        rewards_dones_infos_diff = trajectory.compare_rewards_dones_infos(other=fork, start_step=8, end_step=17)
        if verbose:
            print(actions_diff)
            print(positions_diff)
            print(arrived_diff)
        assert len(actions_diff) == 0
        assert len(positions_diff) == 0
        assert len(arrived_diff) == 0
        assert len(rewards_dones_infos_diff) == 0


def test_run_from_intermediate_step_pkl(verbose: bool = False):
    with tempfile.TemporaryDirectory() as tmpdirname:
        data_dir = Path(tmpdirname)
        trajectory = PolicyRunner.create_from_policy(
            policy=RandomPolicy(reset_at=7),
            obs_builder=EnvStepObservationBuilder(),
            data_dir=data_dir / "trajectory",
            snapshot_interval=1
        )
        if verbose:
            import pandas as pd
            pd.set_option('display.max_rows', None)
            pd.set_option('display.max_columns', None)
            print(trajectory.actions)
            print(trajectory.trains_arrived)
            print(trajectory.trains_positions)
            print(trajectory.trains_rewards_dones_infos)
        other = PolicyRunner.create_from_policy(
            data_dir=data_dir / "other",
            policy=RandomPolicy(),
            start_step=7,
            end_step=17,
            env=RailEnvPersister.load_new(data_dir / "trajectory" / SERIALISED_STATE_SUBDIR / f"{trajectory.ep_id}_step0007.pkl")[0]
        )
        if verbose:
            print(other.actions)
            print(other.trains_arrived)
            print(other.trains_positions)
            print(other.trains_rewards_dones_infos)
        actions_diff = trajectory.compare_actions(other=other, start_step=7, end_step=17)
        positions_diff = trajectory.compare_positions(other=other, start_step=8, end_step=17)
        arrived_diff = trajectory.compare_arrived(other=other, start_step=8, end_step=17)
        rewards_dones_infos_diff = trajectory.compare_rewards_dones_infos(other=other, start_step=8, end_step=17)
        if verbose:
            print(actions_diff)
            print(positions_diff)
            print(arrived_diff)
        assert len(actions_diff) == 0
        assert len(positions_diff) == 0
        assert len(arrived_diff) == 0
        assert len(rewards_dones_infos_diff) == 0


def test_failing_from_wrong_intermediate_step():
    with tempfile.TemporaryDirectory() as tmpdirname:
        data_dir = Path(tmpdirname)
    trajectory = PolicyRunner.create_from_policy(
        policy=RandomPolicy(reset_at=7),
        obs_builder=EnvStepObservationBuilder(),
        data_dir=data_dir / "trajectory",
        snapshot_interval=1
    )
    with pytest.raises(AssertionError) as e_info:
        PolicyRunner.create_from_policy(
            data_dir=data_dir / "other",
            policy=RandomPolicy(),
            start_step=8,
            end_step=17,
            env=RailEnvPersister.load_new(data_dir / "trajectory" / SERIALISED_STATE_SUBDIR / f"{trajectory.ep_id}_step0007.pkl")[0]
        )
    assert str(e_info.value) == 'Expected env at 8, found 7.'


def test_evaluation_snapshots():
    with tempfile.TemporaryDirectory() as tmpdirname:
        data_dir = Path(tmpdirname)
        trajectory = PolicyRunner.create_from_policy(policy=RandomPolicy(), data_dir=data_dir, snapshot_interval=0)
        print(list(trajectory.data_dir.rglob("**/*step*.pkl")))
        assert len(list(trajectory.data_dir.rglob("**/*step*.pkl"))) == 0
        TrajectoryEvaluator(trajectory).evaluate(snapshot_interval=1)
        print(list(trajectory.data_dir.rglob("**/*step*.pkl")))
        assert len(list((trajectory.data_dir / "outputs" / "serialised_state").rglob("**/*step*.pkl"))) == 472
