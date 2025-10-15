import os
import tempfile
from pathlib import Path

import pytest

from benchmarks.benchmark_episodes import DOWNLOAD_INSTRUCTIONS
from flatland.envs.persistence import RailEnvPersister
from flatland.trajectories.policy_runner import PolicyRunner
from flatland.trajectories.trajectories import Trajectory
from flatland_baselines.deadlock_avoidance_heuristic.observation.full_env_observation import FullEnvObservation
from flatland_baselines.deadlock_avoidance_heuristic.policy.deadlock_avoidance_policy import DeadLockAvoidancePolicy


@pytest.mark.parametrize("data_sub_dir,ep_id", [
    # trajectories generated with DLA
    ("malfunction_deadlock_avoidance_heuristics/Test_00/Level_0", "Test_00_Level_0"),
    ("malfunction_deadlock_avoidance_heuristics/Test_00/Level_1", "Test_00_Level_1"),
    ("malfunction_deadlock_avoidance_heuristics/Test_00/Level_2", "Test_00_Level_2"),
    ("malfunction_deadlock_avoidance_heuristics/Test_00/Level_3", "Test_00_Level_3"),
    ("malfunction_deadlock_avoidance_heuristics/Test_00/Level_4", "Test_00_Level_4"),
    ("malfunction_deadlock_avoidance_heuristics/Test_00/Level_5", "Test_00_Level_5"),
    ("malfunction_deadlock_avoidance_heuristics/Test_00/Level_6", "Test_00_Level_6"),
    ("malfunction_deadlock_avoidance_heuristics/Test_00/Level_7", "Test_00_Level_7"),
    ("malfunction_deadlock_avoidance_heuristics/Test_00/Level_8", "Test_00_Level_8"),
    ("malfunction_deadlock_avoidance_heuristics/Test_00/Level_9", "Test_00_Level_9"),
    ("malfunction_deadlock_avoidance_heuristics/Test_01/Level_0", "Test_01_Level_0"),
    ("malfunction_deadlock_avoidance_heuristics/Test_01/Level_1", "Test_01_Level_1"),
    ("malfunction_deadlock_avoidance_heuristics/Test_01/Level_2", "Test_01_Level_2"),
    ("malfunction_deadlock_avoidance_heuristics/Test_01/Level_3", "Test_01_Level_3"),
    ("malfunction_deadlock_avoidance_heuristics/Test_01/Level_4", "Test_01_Level_4"),
    ("malfunction_deadlock_avoidance_heuristics/Test_01/Level_5", "Test_01_Level_5"),
    ("malfunction_deadlock_avoidance_heuristics/Test_01/Level_6", "Test_01_Level_6"),
    ("malfunction_deadlock_avoidance_heuristics/Test_01/Level_7", "Test_01_Level_7"),
    ("malfunction_deadlock_avoidance_heuristics/Test_01/Level_8", "Test_01_Level_8"),
    ("malfunction_deadlock_avoidance_heuristics/Test_01/Level_9", "Test_01_Level_9"),
    ("malfunction_deadlock_avoidance_heuristics/Test_02/Level_0", "Test_02_Level_0"),
    ("malfunction_deadlock_avoidance_heuristics/Test_02/Level_1", "Test_02_Level_1"),
    ("malfunction_deadlock_avoidance_heuristics/Test_02/Level_2", "Test_02_Level_2"),
    ("malfunction_deadlock_avoidance_heuristics/Test_02/Level_3", "Test_02_Level_3"),
    ("malfunction_deadlock_avoidance_heuristics/Test_02/Level_4", "Test_02_Level_4"),
    ("malfunction_deadlock_avoidance_heuristics/Test_02/Level_5", "Test_02_Level_5"),
    ("malfunction_deadlock_avoidance_heuristics/Test_02/Level_6", "Test_02_Level_6"),
    ("malfunction_deadlock_avoidance_heuristics/Test_02/Level_7", "Test_02_Level_7"),
    ("malfunction_deadlock_avoidance_heuristics/Test_02/Level_8", "Test_02_Level_8"),
    ("malfunction_deadlock_avoidance_heuristics/Test_02/Level_9", "Test_02_Level_9"),
    ("malfunction_deadlock_avoidance_heuristics/Test_03/Level_0", "Test_03_Level_0"),
    ("malfunction_deadlock_avoidance_heuristics/Test_03/Level_1", "Test_03_Level_1"),
    ("malfunction_deadlock_avoidance_heuristics/Test_03/Level_2", "Test_03_Level_2"),
    ("malfunction_deadlock_avoidance_heuristics/Test_03/Level_3", "Test_03_Level_3"),
    ("malfunction_deadlock_avoidance_heuristics/Test_03/Level_4", "Test_03_Level_4"),
    ("malfunction_deadlock_avoidance_heuristics/Test_03/Level_5", "Test_03_Level_5"),
])
def test_episode(data_sub_dir: str, ep_id: str):
    _dir = os.getenv("BENCHMARK_EPISODES_FOLDER")
    assert _dir is not None, (DOWNLOAD_INSTRUCTIONS, _dir)
    assert os.path.exists(_dir), (DOWNLOAD_INSTRUCTIONS, _dir)
    data_dir = Path(os.path.join(_dir, data_sub_dir))
    re_run_episode(data_dir, ep_id)


def re_run_episode(data_dir: str, ep_id: str, rendering=False, snapshot_interval=0, start_step=None):
    """
    The data is structured as follows:
        -30x30 map
            Contains the data to replay the episodes.
            - <n>_trains                                 -- for n in 10,15,20,50
                - event_logs
                    ActionEvents.discrete_action 		 -- holds set of action to be replayed for the related episodes.
                    TrainMovementEvents.trains_arrived 	 -- holds success rate for the related episodes.
                    TrainMovementEvents.trains_positions -- holds the positions for the related episodes.
                - serialised_state
                    <ep_id>.pkl                          -- Holds the pickled environment version for the episode.

    All these episodes are with constant speed of 1 and malfunctions free.

    Parameters
    ----------
    data_dir: str
        data dir with trajectory
    ep_id : str
        the episode ID
    start_step : int
        start evaluation from intermediate step (requires snapshot to be present)
    rendering : bool
        render while evaluating
    snapshot_interval : int
        interval to write pkl snapshots. 1 means at every step. 0 means never.
    """
    expected_trajectory = Trajectory(data_dir=data_dir, ep_id=ep_id)
    env_pkl = str((data_dir / "serialised_state" / f"{ep_id}.pkl").resolve())
    env, _ = RailEnvPersister.load_new(env_pkl, obs_builder=FullEnvObservation())

    with tempfile.TemporaryDirectory() as tmpdirname:
        temp_data_dir = Path(tmpdirname)

        recreated_trajectory = PolicyRunner.create_from_policy(
            policy=DeadLockAvoidancePolicy(),
            data_dir=temp_data_dir,
            env=env,
            snapshot_interval=0,
            ep_id=ep_id + "_regen",
        )

        # we optimize and do not consider opposing agents when in state WAITING any more as before perf optimization
        assert len(expected_trajectory.compare_actions(recreated_trajectory, ignoring_waiting=True)) == 0
        assert len(expected_trajectory.compare_positions(recreated_trajectory)) == 0
        assert len(expected_trajectory.compare_arrived(recreated_trajectory)) == 0
        assert len(expected_trajectory.compare_rewards_dones_infos(recreated_trajectory)) == 0
