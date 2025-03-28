import importlib
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import pytest

from flatland.callbacks.callbacks import FlatlandCallbacks, make_multi_callbacks
from flatland.env_generation.env_generator import env_generator
from flatland.envs.persistence import RailEnvPersister
from flatland.envs.rail_env import RailEnv
from flatland.evaluators.trajectory_evaluator import TrajectoryEvaluator, evaluate_trajectory
from flatland.trajectories.trajectories import Policy
from flatland.trajectories.trajectories import Trajectory, DISCRETE_ACTION_FNAME, TRAINS_ARRIVED_FNAME, TRAINS_POSITIONS_FNAME, SERIALISED_STATE_SUBDIR, \
    generate_trajectory_from_policy, generate_trajectories_from_metadata
from flatland.utils.seeding import np_random, random_state_to_hashablestate


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
        trajectory = Trajectory.create_from_policy(policy=RandomPolicy(), data_dir=data_dir, snapshot_interval=5)
        # np_random in loaded episode is same as if it comes directly from env_generator incl. reset()!
        env = trajectory.restore_episode()
        gen, _, _ = env_generator()
        assert random_state_to_hashablestate(env.np_random) == random_state_to_hashablestate(gen.np_random)

        gen.reset(random_seed=42)
        assert random_state_to_hashablestate(env.np_random) == random_state_to_hashablestate(gen.np_random)


def test_from_submission():
    with tempfile.TemporaryDirectory() as tmpdirname:
        data_dir = Path(tmpdirname)
        trajectory = Trajectory.create_from_policy(policy=RandomPolicy(), data_dir=data_dir, snapshot_interval=5)

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
            generate_trajectory_from_policy(["--data-dir", data_dir, "--policy-pkg", "tests.trajectories.test_trajectories", "--policy-cls", "RandomPolicy"])
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


@pytest.mark.skip  # TODO https://github.com/flatland-association/flatland-rl/issues/101 import heuristic baseline as example
def test_gen_trajectories_from_metadata():
    metadata_csv_path = importlib.resources.files("env_data.tests.service_test").joinpath("metadata.csv")
    with tempfile.TemporaryDirectory() as tmpdirname:
        with importlib.resources.as_file(metadata_csv_path) as metadata_csv:
            tmpdir = Path(tmpdirname)
            generate_trajectories_from_metadata(
                metadata_csv=metadata_csv,
                data_dir=tmpdir,
                # TODO https://github.com/flatland-association/flatland-rl/issues/101 import heuristic baseline as example
                policy_pkg="src.policy.deadlock_avoidance_policy",
                policy_cls="DeadLockAvoidancePolicy",
                obs_builder_pkg="src.observation.full_state_observation",
                obs_builder_cls="FullStateObservationBuilder"
            )
            metadata = pd.read_csv(metadata_csv)
            for sr, t, (k, v) in zip([0.8571428571428571, 1.0, 0.8571428571428571, 1.0], [391, 165, 391, 165], metadata.iterrows()):
                df = pd.read_csv(tmpdir / v["test_id"] / v["env_id"] / TRAINS_ARRIVED_FNAME, sep="\t")
                assert df["success_rate"].to_list() == [sr]
                assert df["env_time"].to_list() == [t]


@pytest.mark.parametrize(
    'seed',
    [43, 44, 1001, 249385789, 289435789]
)
def test_persistence_reset(seed):
    rail_env, _, _ = env_generator(seed=seed, x_dim=40, y_dim=57, )
    np_random_generated = random_state_to_hashablestate(rail_env.np_random)
    dict_generated = RailEnvPersister.get_full_state(rail_env)

    with tempfile.TemporaryDirectory() as tmpdirname:
        RailEnvPersister.save(rail_env, os.path.join(tmpdirname, "env.pkl"))
        reloaded, _ = RailEnvPersister.load_new(os.path.join(tmpdirname, "env.pkl"))
    np_random_reloaded = random_state_to_hashablestate(reloaded.np_random)
    dict_reloaded = RailEnvPersister.get_full_state(reloaded)

    assert np_random_generated == np_random_reloaded
    assert dict_generated == dict_reloaded

    rail_env.reset(regenerate_rail=True, regenerate_schedule=True, random_seed=seed)
    np_random_reset = random_state_to_hashablestate(rail_env.np_random)
    dict_reset = RailEnvPersister.get_full_state(rail_env)

    assert np_random_generated == np_random_reset
    assert dict_generated == dict_reset

    # CAVEAT: if we pass the seed but do not regenerate, this results in a different state!
    rail_env.reset(regenerate_rail=False, regenerate_schedule=False, random_seed=seed)
    np_random_reset_no_regenerate_same_seed = random_state_to_hashablestate(rail_env.np_random)
    dict_reset_no_regenerate_same_seed = RailEnvPersister.get_full_state(rail_env)

    assert np_random_generated != np_random_reset_no_regenerate_same_seed
    assert dict_generated != dict_reset_no_regenerate_same_seed

    # however, if we do not regenerate and pass no seed, this results in the same state.
    rail_env.reset(regenerate_rail=False, regenerate_schedule=False)
    np_random_reset_no_regenerate_no_seed = random_state_to_hashablestate(rail_env.np_random)
    dict_reset_no_regenerate_no_seed = RailEnvPersister.get_full_state(rail_env)

    assert np_random_generated != np_random_reset_no_regenerate_no_seed
    assert dict_generated != dict_reset_no_regenerate_no_seed


def test_evaluation_snapshots():
    with tempfile.TemporaryDirectory() as tmpdirname:
        data_dir = Path(tmpdirname)
        trajectory = Trajectory.create_from_policy(policy=RandomPolicy(), data_dir=data_dir, snapshot_interval=0)
        print(list(trajectory.data_dir.rglob("**/*step*.pkl")))
        assert len(list(trajectory.data_dir.rglob("**/*step*.pkl"))) == 0
        TrajectoryEvaluator(trajectory).evaluate(snapshot_interval=1)
        print(list(trajectory.data_dir.rglob("**/*step*.pkl")))
        assert len(list((trajectory.data_dir / "outputs" / "serialised_state").rglob("**/*step*.pkl"))) == 472
